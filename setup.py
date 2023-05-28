from setuptools import find_packages
from setuptools import setup

setup(
    name='adapmen',
    author='Shengyi Jiang & Xu-Hui Liu',
    author_email='shengyi.jiang@outlook.com',
    packages=find_packages(),
    # python_requires='=3.9.7',
    install_requires=[
        'torch',
        'gym[atari]==0.23.1',
        'gym[accept-rom-license]==0.23.1',
        'scipy',
        'numpy',
        'munch',
        'pyyaml',
        'colorama',
        'pandas',
        'mujoco_py',
        'tensorboard'
    ],
    package_data={
        # include default config files
        "": ["*.yaml", "*.xml"],
    }
)
