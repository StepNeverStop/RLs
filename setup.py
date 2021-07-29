#!/usr/bin/env python3
# encoding: utf-8
import platform
from importlib import util as import_util
from setuptools import setup, find_packages

spec = import_util.spec_from_file_location('_metadata', 'rls/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()
long_description += '\n\nFor more information see our [github repository](https://github.com/StepNeverStop/RLs).'

systembased_extras = {
    'windows': [
        # 'swig',   # install it with conda.
        # 'box2d-py',   # install it manually.
        'cmake',
        'pywin32'
    ],
    'nowindows': []
}
extras = {
    'ray': ['pytest-runner', 'ray'],
    'unity': ['mlagents-envs==0.27.0'],
    'mujoco': ['mujoco_py'],
    'pybullet': ['PyBullet'],
    'gym-minigrid': ['gym-minigrid'],
    'gym_donkeycar': ['gym_donkeycar']
}

all_deps = []
for name in extras:
    all_deps += extras[name]

if platform.system() == "Windows":
    extras['windows'] = systembased_extras['windows']
    all_deps += systembased_extras['windows']
else:
    extras['nowindows'] = systembased_extras['nowindows']
    all_deps += systembased_extras['nowindows']
extras['all'] = all_deps

setup(
    name="RLs",
    version=_metadata.__version__,
    description="Reinforcement Learning Algorithm Based On PyTorch.",
    keywords='reinforcement learning gym ml-agents pytorch',
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
        'Programming Language :: Python :: 3.8',
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'gym>=0.15.4',
        "torch>=1.9.0",
        'numpy>=1.19.0',
        'pyyaml',
        'tqdm',
        'tensorboard',
        'colored_traceback',
        # 'imageio'
    ],
    extras_require=extras,
)
