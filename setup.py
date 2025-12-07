from setuptools import setup, find_packages

setup(
    name="miniml",
    version="1.0.1",
    packages=find_packages(include=['miniml*', 'estimators*', 'adapters*']),
    python_requires=">=3.7",
    install_requires=[],
)