import time
import math
import csv
import os
import threading

import rclpy
from rclpy.node import Node
from dronekit import connect, VehicleMode
from std_msgs.msg import String
from pymavlink import mavutil
from geopy.distance import geodesic
from apriltag_interfaces.msg import TagPoseStamped
import math
import time
from geopy.distance import geodesic

class SimulatedMovingTarget:
    def __init__(self, initial_lat, initial_lon, heading_deg, initial_offset_m=40.0, speed_mps=16.0):
        self.start_time = time.time()
        self.heading_rad = math.radians(heading_deg)
        self.speed_mps = speed_mps
        self.earth_radius = 6378137.0

        # Offset 40 meters ahead of UAV
        d_lat = (initial_offset_m * math.cos(self.heading_rad)) / self.earth_radius
        d_lon = (initial_offset_m * math.sin(self.heading_rad)) / (
            self.earth_radius * math.cos(math.radians(initial_lat))
        )

        self.start_lat = initial_lat + math.degrees(d_lat)
        self.start_lon = initial_lon + math.degrees(d_lon)

    def get_position(self):
        t = time.time() - self.start_time
        dist = self.speed_mps * t

        d_lat = (dist * math.cos(self.heading_rad)) / self.earth_radius
        d_lon = (dist * math.sin(self.heading_rad)) / (
            self.earth_radius * math.cos(math.radians(self.start_lat))
        )

        new_lat = self.start_lat + math.degrees(d_lat)
        new_lon = self.start_lon + math.degrees(d_lon)
        return new_lat, new_lon

# --- PID Controller ---
class PID:
    def __init__(self, kp, ki, kd, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_limit = integral_limit

    def update(self, error, dt):
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll/2)
    sr = math.sin(roll/2)
    cp = math.cos(pitch/2)
    sp = math.sin(pitch/2)
    cy = math.cos(yaw/2)
    sy = math.sin(yaw/2)
    return [
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ]

class UAVFollower(Node):
    def __init__(self):
        super().__init__('uav_follower')

        self.serial_path = 'udpout:127.0.0.1:14550'
        self.baudrate = 57600
        self.create_subscription(TagPoseStamped, '/apriltag/pose_in_base', self.apriltag_callback, 10)

        self.fixed_altitude = 30.0
        self.target_distance = 15.0  # target following distance
        self.base_throttle = 0.55
        self.last_apriltag_time = 0.0
        self.latest_apriltag_pose = None

        self.trajectory_triggered = False
        self.trajectory_start_time = None
        self.trajectory_duration = 10.0  # seconds to complete descent
        self.simulate_fake_distance = True
        self.sim_start_time = time.time()
        self.low_altitude = 25.0
        self.low_distance = 10.0
        self.original_altitude = self.fixed_altitude
        self.original_distance = self.target_distance

        # PID Controllers
        self.distance_pid = PID(1.0, 0.0, 0.0, integral_limit=5.0)
        self.airspeed_pid = PID(0.08, 0.04, 0.0, integral_limit=5.0)
        self.altitude_pid = PID(0.052, 0.0095, 0.25, integral_limit=5.5)
        self.lateral_pid = PID(kp=0.0013, ki=0.0014, kd=0.16, integral_limit=1.0)

        self.target_lat = None
        self.target_lon = None
        self.last_lat = None
        self.last_lon = None
        self.last_time = time.time()
        self.latest_tags = {}

        self.connect_vehicle()
        self.subscription = self.create_subscription(
            String,
            'uav1/location',
            self.location_callback,
            10
        )

        uav_loc = self.vehicle.location.global_relative_frame
        heading_deg = self.vehicle.heading
        self.target_sim = SimulatedMovingTarget(
            uav_loc.lat, uav_loc.lon, heading_deg, initial_offset_m=40.0, speed_mps=16.0
        )
        self.start_attitude_thread()

    def connect_vehicle(self):
        self.get_logger().info(f"Connecting to vehicle at {self.serial_path}...")
        self.vehicle = connect(self.serial_path)
        self.get_logger().info(f"Connected! Firmware: {self.vehicle.version}")

    def apriltag_callback(self, msg: TagPoseStamped):
        tag_id = getattr(msg, 'id', 0)
        self.latest_tags[tag_id] = (msg.pose, time.time())

    def location_callback(self, msg: String):
        try:
            parts = msg.data.strip().split(',')
            lat = float(parts[0].split(':')[1].strip())
            lon = float(parts[1].split(':')[1].strip())

            # smooth target
            if self.last_lat is not None:
                self.target_lat = 0.8 * self.last_lat + 0.2 * lat
                self.target_lon = 0.8 * self.last_lon + 0.2 * lon
            else:
                self.target_lat = lat
                self.target_lon = lon

            self.last_lat = self.target_lat
            self.last_lon = self.target_lon

        except Exception as e:
            self.get_logger().error(f"Error parsing location: {e}")

    def start_attitude_thread(self):
        thread = threading.Thread(target=self.attitude_loop)
        thread.daemon = True
        thread.start()

    def attitude_loop(self):
        while rclpy.ok():
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            if self.target_lat is None or self.target_lon is None:
                time.sleep(0.05)
                continue

            if self.vehicle.mode.name != "GUIDED":
                time.sleep(0.05)
                continue

            # Trigger trajectory only once when entering GUIDED
            if not self.trajectory_triggered:
                self.trajectory_triggered = True
                self.trajectory_start_time = now
                self.get_logger().info("🛫 GUIDED mode detected — starting trajectory descent.")

            # Apply trajectory if triggered
            if self.trajectory_triggered:
                elapsed_traj = now - self.trajectory_start_time
                if elapsed_traj < self.trajectory_duration:
                    factor = elapsed_traj / self.trajectory_duration
                    self.fixed_altitude = self.original_altitude - factor * (self.original_altitude - self.low_altitude)
                    self.target_distance = self.original_distance - factor * (self.original_distance - self.low_distance)
                else:
                    self.fixed_altitude = self.low_altitude
                    self.target_distance = self.low_distance

            current = self.vehicle.location.global_relative_frame
            uav_lat = current.lat
            uav_lon = current.lon
            uav_alt = current.alt or self.fixed_altitude
            airspeed = self.vehicle.airspeed or 0.0

            pose = None
            tag_id = None
            for tid, (p, t) in self.latest_tags.items():
                if time.time() - t < 0.5:
                    pose = p
                    tag_id = tid
                    break

            use_cv = pose is not None

            if use_cv:
                lateral_error = pose.position.y
                distance_error = pose.position.z - self.target_distance
                dist_to_target = pose.position.z

            else:
                # --- Simulated Car Using World GPS ---
                sim_lat, sim_lon = self.target_sim.get_position()

                dist_to_target = geodesic((uav_lat, uav_lon), (sim_lat, sim_lon)).meters
                distance_error = dist_to_target - self.target_distance

                d_east = geodesic((sim_lat, sim_lon), (sim_lat, uav_lon)).meters
                if uav_lon < sim_lon:
                    d_east *= -1
                lateral_error = d_east

            # PID control
            target_airspeed = 16.0 + self.distance_pid.update(distance_error, dt)
            target_airspeed = max(10.0, min(22.0, target_airspeed))

            airspeed_error = target_airspeed - airspeed
            throttle_adjust = self.airspeed_pid.update(airspeed_error, dt)
            throttle_cmd = self.base_throttle + throttle_adjust
            throttle_cmd = max(0.0, min(throttle_cmd, 1.0))

            altitude_error = self.fixed_altitude - uav_alt
            pitch_cmd = self.altitude_pid.update(altitude_error, dt)
            pitch_cmd = max(min(pitch_cmd, math.radians(15)), math.radians(-15))

            roll_cmd = self.lateral_pid.update(lateral_error, dt)
            roll_cmd = max(min(roll_cmd, math.radians(3.0)), math.radians(-3.0))

            q = euler_to_quaternion(roll_cmd, pitch_cmd, 0.0)
            self.vehicle._master.mav.set_attitude_target_send(
                int((now - self.last_time) * 1000),
                self.vehicle._master.target_system,
                self.vehicle._master.target_component,
                0b00000100,
                q,
                0.0, 0.0, 0.0,
                throttle_cmd
            )
            log_msg = (
                f"[{'Tag ' + str(tag_id) if use_cv else 'GPS'}]  "
                f"Dist: {dist_to_target:6.2f} m   |  "
                f"Target AS: {target_airspeed:5.2f}  |  "
                f"Airspeed: {airspeed:5.2f}  |  "
                f"Alt: {uav_alt:5.2f} m  |  "
                f"Throttle: {throttle_cmd:4.2f}  |  "
                f"Pitch: {math.degrees(pitch_cmd):6.2f}°  |  "
                f"Lateral: {lateral_error:6.2f} m  |  "
                f"Roll: {math.degrees(roll_cmd):6.2f}° |  " 
                f"Target Atitude: {self.fixed_altitude:5.2f} m  |  "
                f"Target Distance: {self.target_distance:5.2f} m  |  "
            )
            self.get_logger().info(log_msg)
            time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = UAVFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vehicle.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
