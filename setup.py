from setuptools import setup, find_packages


setup(
    name="banana-navigation",
    version="0.1.0",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "torch>=1.0, <2.0",
        "plotnine>=0.2.0",
        "click>=7.0.0",
        "unityagents>=0.4.0", 'pandas'
    ],
    author="Darius Aliulis",
    author_email="darius.aliulis@gmail.com",
    description="Deep Q-Learning Agent for Unity Banana Navigation Environment",
    url="https://github.com/daraliu/drl-banana-navigation",
)
