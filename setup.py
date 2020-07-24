from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

setup(
    name="RLs",
    version='UnReleased',
    description="Reinforcement Learning Algorithm Based On TensorFlow 2.x.",
    keywords='reinforcement learning gym ml-agents tf2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/StepNeverStop/RLs',
    author='Keavnn',
    author_email='keavnn.wjs@gmail.com',
    maintainer='Keavnn',
    maintainer_email='keavnn.wjs@gmail.com',
    license="Apache-2.0 License",
    packages=find_packages(exclude=['test', 'test.*']),
    python_requires='>=3.6',
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Reinforcement Learning Researchers',
        'Topic :: Artificial Intelligence :: Reinforcement Learning',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache-2.0 License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'gym>=0.15.0',
        'cloudpickle==1.2.2',
        'tensorflow-gpu>=2.0.0, <=2.2.0',
        'tensorflow_probability==0.7.0',
        'docopt',
        'numpy',
        'opencv-python',
        'imageio',
        'pyyaml',
        'protobuf',
        'grpcio>=1.24.3',
        'pandas',
        'openpyxl',
        'tqdm'
    ],
    extras_require={
        'windows': [
            'pywin32'
        ],
        'non-windows': [
            'ray',
            'ray[debug]',
        ],
        'atari': [
            'atari_py',
            'cv2',
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
