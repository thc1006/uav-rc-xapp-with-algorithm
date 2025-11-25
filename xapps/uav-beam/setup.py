"""
Setup script for UAV Beam Tracking xApp
"""

from setuptools import setup, find_packages

setup(
    name="uav-beam-xapp",
    version="0.1.0",
    description="UAV Beam Tracking xApp for O-RAN Near-RT RIC",
    author="UAV O-RAN Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "flask>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "ml": [
            "torch>=1.9.0",
            "scikit-learn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "uav-beam-xapp=uav_beam.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Communications",
        "Topic :: Scientific/Engineering",
    ],
)
