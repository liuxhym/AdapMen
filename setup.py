from setuptools import find_packages
from setuptools import setup

setup(
    name='adapmen',
    author='Shengyi Jiang & Xu-Hui Liu',
    author_email='shengyi.jiang@outlook.com',
    packages=find_packages(),
    # python_requires='=3.9.7',
    install_requires=[
        'torch<=2.0.1',
        'gym[atari]==0.23.1',
        'gym[accept-rom-license]==0.23.1',
        'scipy==1.7.3',
        'munch',
        'pyyaml',
        'colorama',
        'pandas',
        'mujoco_py',
        'tensorboard',
        'metadrive-simulator==0.2.5.1',
        'numpy',
        'dm_tree',
        'ray'
    ],
    package_data={
        # include default config files
        "": ["*.yaml", "*.xml"],
    }
)
