"""Setup configuration for uav-policy package."""

from setuptools import setup, find_packages

setup(
    name="uav-policy",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "flask>=2.3.0",
        "werkzeug>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "uav-policy=uav_policy.main:main",
        ],
    },
)
