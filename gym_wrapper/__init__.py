import platform
if platform.system() == "Windows":
    from .wrapper_win import gym_envs
else:
    from .wrapper_linux import gym_envs
