from setuptools import setup, find_packages

setup(
    name="miniml",
    version="1.1.2",
    author="Wilner Manzanares (Michego Takoro 'Shuuida')",
    description="Edge AI and Embedded Machine Learning Framework (Zero-Dependencies)",
    packages=find_packages(include=['miniml*', 'estimators*', 'adapters*', 'miniml_cli*']),
    python_requires=">=3.7",
    install_requires=[],
    entry_points={
        "console_scripts": [
            "miniml=miniml_cli.main:main", 
        ],
    },
)