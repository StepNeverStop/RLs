# LunarLander-v2

- Convergence episode: 16
- max step per episode: 1000
- algorithm: sac+gumbel[[code](https://github.com/StepNeverStop/RLs/blob/master/algos/tf2algos/sac.py)]
- Actor
- - 64 -> 32 -> logits
- Critic_Q
- - 64 -> 64 -> 1
- Critic_V
- - 32 -> 32 -> 1
- n_step: 4
- step_before_train(random): 10000
- agents in parrallel: 10

Parameters:
```
'alpha': 0.2,
'auto_adaption': True,
'gamma': 0.999,
'ployak': 0.995,
'lr': 2.0e-3,
'discrete_tau': 1.0,
'max_train_episode': 50000,
'batch_size': 8192,
'buffer_size': 200000,
'use_priority': False,
'n_step': True
```

Result:

![](./result.gif)

![](./training_process.png)

![](./training_process2.png)

![](./training_curve.png)

