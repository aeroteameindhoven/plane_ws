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
from dronekit import LocationGlobalRelative


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
        super().__init__('gps_follower_single_loop')

        self.get_logger().info("Connecting to UAV...")
        self.vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
        self.get_logger().info("Connected to UAV!")

        self.get_logger().info("Setting mode to GUIDED...")
        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 15)

        self.fixed_altitude = 20.0
        self.last_lat, self.last_lon = None, None
        self.uav_lat, self.uav_lon = None, None
        self.start_time = time.time()
        self.control_start_time = time.time()  # Start immediately
        self.run_duration = 400.0
        self.last_time = time.time()
        self.pid_integral = 0.0
        self.prev_error = 0.0
        self.last_target_airspeed = 14.0
        self.smoothed_target_airspeed = None
        self.trajectory_start_time = None
        self.trajectory_duration = 10.0
        self.initial_ref_distance = None
        self.final_ref_distance = None
        self.initial_altitude = None
        self.final_altitude = None

        # Logging lists
        self.log_time = []
        self.log_distance_error = []
        self.log_chirp_input = []
        self.log_altitude = []
        self.log_ref_distance = []
        self.log_actual_airspeed = []
        self.log_actual_distance = []
        self.log_horizontal_distance = []
        self.log_actual_lat = []
        self.log_actual_lon = []
        self.log_target_lat = []
        self.log_target_lon = []

        # Set UAV position
        loc = self.vehicle.location.global_relative_frame
        self.uav_lat = loc.lat
        self.uav_lon = loc.lon

        self.trajectory_triggered = False

        # Altitude and distance descent parameters
        self.original_altitude = self.vehicle.location.global_relative_frame.alt or 20.0
        self.low_altitude = self.original_altitude - 5.0
        self.original_distance = 40.0
        self.low_distance = self.original_distance - 5.0
        self.target_distance = self.original_distance


        self.initialize_moving_target()
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

    def initialize_moving_target(self):
        heading = self.vehicle.heading
        self.target_sim = SimulatedMovingTarget(
            self.uav_lat,
            self.uav_lon,
            heading_deg=heading
        )

    def attitude_loop(self):
        while rclpy.ok():
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            # Get current UAV location
            current = self.vehicle.location.global_relative_frame
            self.uav_lat = current.lat
            self.uav_lon = current.lon
            uav_alt = current.alt or self.fixed_altitude
            airspeed = self.vehicle.airspeed or 0.0

            # Wait for GUIDED mode
            if self.vehicle.mode.name != "GUIDED":
                time.sleep(0.05)
                continue

            # Trigger trajectory once
            if not self.trajectory_triggered:
                self.trajectory_triggered = True
                self.trajectory_start_time = now
                self.get_logger().info("🛫 GUIDED mode detected — starting trajectory.")

            elapsed_traj = now - self.trajectory_start_time
            if elapsed_traj < self.trajectory_duration:
                factor = elapsed_traj / self.trajectory_duration
                self.fixed_altitude = self.original_altitude - factor * (self.original_altitude - self.low_altitude)
                self.target_distance = self.original_distance - factor * (self.original_distance - self.low_distance)
            else:
                self.fixed_altitude = self.low_altitude
                self.target_distance = self.low_distance

            # Get simulated car location
            sim_lat, sim_lon = self.target_sim.get_position()
            dist_to_target = geodesic((self.uav_lat, self.uav_lon), (sim_lat, sim_lon)).meters
            distance_error = dist_to_target - self.target_distance

            # Airspeed control like first code
            k_p = 1.0
            correction = k_p * distance_error
            base_speed = 16.0
            target_airspeed = base_speed + correction
            target_airspeed = max(14.0, min(20.0, target_airspeed))

            # Send movement command
            self.vehicle.simple_goto(LocationGlobalRelative(sim_lat, sim_lon, self.fixed_altitude))

            # Send MAVLink airspeed command
            self.vehicle._master.mav.command_long_send(
                self.vehicle._master.target_system,
                self.vehicle._master.target_component,
                mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                0,
                0,  # airspeed type
                target_airspeed,
                -1, 0, 0, 0, 0
            )

            # Logging
            log_msg = (
                f"[GPS] Dist: {dist_to_target:.1f} m | Ref: {self.target_distance:.1f} | "
                f"AS_cmd: {target_airspeed:.1f} | AS_act: {airspeed:.1f} | Alt: {uav_alt:.1f}"
            )
            self.get_logger().info(log_msg)
            time.sleep(0.2)

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
