import rclpy
from rclpy.node import Node
from dronekit import connect, VehicleMode, LocationGlobalRelative
from std_msgs.msg import String
from pymavlink import mavutil
from geopy.distance import geodesic
import time
import csv
import os
from apriltag_interfaces.msg import TagPoseStamped

class UAV2Node(Node):
    def __init__(self):
        super().__init__('uav2_node')
        self.serial_path = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A906H62E-if00-port0'
        self.vehicle = None
        self.fixed_alt = 20.0
        self.target_distance = 20.0  # meters behind
        self.kp = 1.0
        self.base_speed = 16.0  # default speed
        self.start_time = time.time()
        self.control_start_time = None
        self.latest_apriltag = None
        self.last_apriltag_time = 0.0

        self.latest_tags = {}  # Dict of {id: (pose, timestamp)}
        self.create_subscription(
            TagPoseStamped,
            '/apriltag/pose_in_base',
            self.apriltag_callback,
            10
        )

        self.log_filename = "uav2_control_log.csv"
        self.init_logger_file()

        self.connect_vehicle()

    def init_logger_file(self):
        # Create CSV and write header
        if not os.path.exists(self.log_filename):
            with open(self.log_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Distance_m', 'Error_m', 'TargetAirspeed_mps', 'ActualAirspeed_mps'])

    def apriltag_callback(self, msg: TagPoseStamped):
        tag_id = msg.id if hasattr(msg, 'id') else 0
        self.latest_tags[tag_id] = (msg.pose.pose, time.time())

    def log_to_file(self, distance, error, target_airspeed, actual_airspeed):
        with open(self.log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{distance:.2f}",
                f"{error:.2f}",
                f"{target_airspeed:.2f}",
                f"{actual_airspeed:.2f}"
            ])

    def connect_vehicle(self):
        try:
            self.get_logger().info(f"[UAV2] Connecting to {self.serial_path}...")
            self.vehicle = connect('udpout:127.0.1:14550')
            self.get_logger().info(f"[UAV2] Connected! Firmware: {self.vehicle.version}")

            self.subscription = self.create_subscription(
                String,
                'uav1/location',
                self.location_callback,
                10
            )
            self.get_logger().info("[UAV2] Subscribed to uav1/location")
        except Exception as e:
            self.get_logger().error(f"[UAV2] Connection error: {e}")

    def location_callback(self, msg: String):
        try:
            parts = msg.data.strip().split(',')
            lat = float(parts[0].split(':')[1].strip())
            lon = float(parts[1].split(':')[1].strip())

            if self.vehicle.mode.name != "GUIDED":
                self.get_logger().warn("[UAV2] Not in GUIDED mode. Skipping movement.")
                return

            current_loc = self.vehicle.location.global_relative_frame
            if current_loc.lat is None or current_loc.lon is None:
                self.get_logger().warn("[UAV2] Current GPS not available.")
                return

            uav_lat = current_loc.lat
            uav_lon = current_loc.lon
            uav_alt = current_loc.alt or self.fixed_alt

            horizontal_distance = geodesic((uav_lat, uav_lon), (lat, lon)).meters

            now = time.time()
            selected_pose = None
            # --- AprilTag priority check ---
            for tag_id in [0, 1]:  # ID priority list
                pose, t = self.latest_tags.get(tag_id, (None, 0.0))
                if pose and (now - t < 0.5):
                    selected_pose = pose
                    self.get_logger().info(f"[UAV2] Using AprilTag ID {tag_id}")
                    self.get_logger().info("[MODE] Control source: AprilTag")
                    break

            if selected_pose:
                tag_dist = selected_pose.position.z
                error = tag_dist - self.target_distance
                correction = self.kp * error
                airspeed_cmd = self.base_speed + correction
                airspeed_cmd = max(14.0, min(20.0, airspeed_cmd))

                actual_airspeed = self.vehicle.airspeed or 0.0
                self.get_logger().info(
                    f"[UAV2][Tag] Dist: {tag_dist:.1f} m | Cmd AS: {airspeed_cmd:.1f} | AS: {actual_airspeed:.1f}"
                )

                self.log_to_file(tag_dist, error, airspeed_cmd, actual_airspeed)
                return  # ✅ Skip GPS follow

            elapsed_total = now - self.start_time

            if self.control_start_time is None and elapsed_total >= 3.0:
                self.control_start_time = now

            if self.control_start_time is not None:
                self.get_logger().info("[MODE] Control source: GPS")
                error = horizontal_distance - self.target_distance
                correction = self.kp * error
                airspeed_cmd = self.base_speed + correction
                airspeed_cmd = max(14.0, min(20.0, airspeed_cmd))

                # Send airspeed command
                self.vehicle._master.mav.command_long_send(
                    self.vehicle._master.target_system,
                    self.vehicle._master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                    0,
                    0,  # 0 = airspeed
                    airspeed_cmd,
                    -1, 0, 0, 0, 0
                )

                actual_airspeed = self.vehicle.airspeed or 0.0
                self.get_logger().info(
                    f"[UAV2] Dist: {horizontal_distance:.1f} m | Cmd AS: {airspeed_cmd:.1f} m/s | Actual AS: {actual_airspeed:.1f} m/s"
                )

                # Log to CSV
                self.log_to_file(horizontal_distance, error, airspeed_cmd, actual_airspeed)

            # Always GoTo
            target = LocationGlobalRelative(lat, lon, self.fixed_alt)
            self.vehicle.simple_goto(target)

        except Exception as e:
            self.get_logger().error(f"[UAV2] Error parsing location or sending command: {e}")
            self.get_logger().error(f"[UAV2] Raw msg: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = UAV2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
