from setuptools import setup
import os
from glob import glob

package_name = 'dual_pid_trajectory'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clara',
    maintainer_email='clara@example.com',
    description='Dual UAV connection via DroneKit using ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'uav2_node = dual_pid_trajectory.uav2_node:main',
        ],
    },
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('dual_pid_trajectory/launch/*.launch.py')),
    ],
)
