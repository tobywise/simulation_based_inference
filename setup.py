from setuptools import find_packages, setup

setup(
    name='simulation_based_inference',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'sbi==0.22.0',
        'tqdm'
    ]
)