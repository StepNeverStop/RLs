import platform
if platform.system() == "Windows":
    from .wrapper_win import gym_envs
else:
    use_ray = False
    if use_ray:
        from .wrapper_linux import gym_envs
    else:
        from .wrapper_win import gym_envs
