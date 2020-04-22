from setuptools import setup, find_packages


setup(
    name="banana-navigation",
    version="0.1.0",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "pandas",
        "plotnine>=0.2.0",
        "click>=7.0.0",
    ],
    author="Darius Aliulis",
    author_email="darius.aliulis@gmail.com",
    description="Deep Q-Learning Agent for Unity Banana Navigation Environment",
    url="https://github.com/daraliu/drl-banana-navigation",
    entry_points='''
        [console_scripts]
        banana-nav=banana_nav.cli:cli
    '''
)
