#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from importlib import util as import_util
from setuptools import setup, find_packages

spec = import_util.spec_from_file_location('_metadata', 'rls/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()
long_description += '\n\nFor more information see our [github repository](https://github.com/StepNeverStop/RLs).'

setup(
    name="RLs",
    version=_metadata.__version__,
    description="Reinforcement Learning Algorithm Based On TensorFlow 2.x.",
    keywords='reinforcement learning gym ml-agents tf2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/StepNeverStop/RLs',
    author='Keavnn',
    author_email='keavnn.wjs@gmail.com',
    maintainer='Keavnn',
    maintainer_email='keavnn.wjs@gmail.com',
    license="Apache License, Version 2.0",
    packages=find_packages(exclude=['test', 'test.*']),
    python_requires='>=3.6',
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        # Indicate who your project is intended for
        'Intended Audience :: Reinforcement Learning Researchers',
        'Topic :: Artificial Intelligence :: Reinforcement Learning',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'docopt',
        'numpy',
        'gym>=0.15.0',
        'pyyaml',
        'tqdm',
        'tensorflow>=2.0.0, <=2.1.0',
        'tensorflow_probability==0.9.0',
        'gast==0.2.2'
    ],
    extras_require={
        'windows': [
            'pywin32'
        ],
        'non-windows': [
            'ray',
            'ray[debug]',
        ],
        'unity': [
            'mlagents-envs==0.17.0'
        ],
        'atari': [
            'atari_py',
            'opencv-python',
        ],
        'mujoco': [
            'mujoco_py'
        ],
        'pybullet': [
            'PyBullet'
        ],
        'gym-minigrid': [
            'gym-minigrid'
        ]
    },
)
