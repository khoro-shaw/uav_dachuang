from setuptools import find_packages
from distutils.core import setup

setup(
    name="uav_dachuang",
    version="1.0.0",
    author="Khoro Shaw",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="u202217270@hust.edu.cn",
    description="UAV Path Tracking with Reinforcement Learning",
    install_requires=[
        "torch",
        "numpy",
        "gymnasium",
        "matplotlib",
        "setuptools",
        "GitPython",
        "onnx",
    ],
)
