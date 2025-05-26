import rclpy
from rclpy.node import Node
from dronekit import connect, Vehicle
from std_msgs.msg import String
import time

class UAV1Node(Node):
    def __init__(self):
        super().__init__('uav1_node')
        self.publisher = self.create_publisher(String, 'uav1/location', 10)
        self.serial_path = 'udpout:127.0.0.1:14551'
        self.vehicle: Vehicle = None
        self.connecting = False

        # Timers
        self.connect_vehicle()
        self.create_timer(1 / 5.0, self.publish_location)

    def connect_vehicle(self):
        try:
            self.connecting = True
            self.get_logger().info(f"[UAV1] Connecting to {self.serial_path}...")
            self.vehicle = connect('udpout:127.0.0.1:14551')
            self.get_logger().info(f"[UAV1] Connected with firmware: {self.vehicle.version}")
        except Exception as e:
            self.vehicle = None
            self.get_logger().error(f"[UAV1] Connection error: {e}")
        finally:
            self.connecting = False


    def publish_location(self):
        if self.vehicle is not None and self.vehicle.location.global_frame:
            loc = self.vehicle.location.global_frame
            heading = getattr(self.vehicle, 'heading', None)

            if loc.lat is not None and loc.lon is not None and loc.alt is not None:
                msg = String()
                if heading is not None:
                    msg.data = f"Lat: {loc.lat:.7f}, Lon: {loc.lon:.7f}, Alt: {loc.alt:.2f}, Heading: {heading:.1f}"
                else:
                    msg.data = f"Lat: {loc.lat:.7f}, Lon: {loc.lon:.7f}, Alt: {loc.alt:.2f}, Heading: N/A"
                self.publisher.publish(msg)
            else:
                self.get_logger().warn("[UAV1] GPS location not available yet.")
        else:
            self.get_logger().warn("[UAV1] Skipping publish; vehicle not connected or data missing.")


def main(args=None):
    rclpy.init(args=args)
    node = UAV1Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
