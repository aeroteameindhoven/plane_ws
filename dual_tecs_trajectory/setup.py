from setuptools import setup
import os
from glob import glob

package_name = 'dual_tecs_trajectory'

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
            'uav1_node = dual_tecs_trajectory.uav1_node:main',
            'uav2_node = dual_tecs_trajectory.uav2_node:main',
        ],
    },
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('dual_tecs_trajectory/launch/*.launch.py')),
    ],
)
