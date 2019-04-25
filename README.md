# RLs

This project include some state-of-art or classic RL(reinforcement learning) algorithms used for training agents by interactive with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents) v0.8.1.

```
Usage:
    python [options]

Options:
    -h,--help                   显示帮助
    -i,--inference              推断 [default: False]
    -a,--algorithm=<name>       算法 [default: ppo]
    -c,--config-file=<file>     指定模型的超参数config文件 [default: None]
    -e,--env=<file>             指定环境名称 [default: None]
    -p,--port=<n>               端口 [default: 5005]
    -u,--unity                  是否使用unity客户端 [default: False]
    -g,--graphic                是否显示图形界面 [default: False]
    -m,--model-string=<name>    训练的名字/载入模型的路径 [default: None]
    -s,--save-frequency=<n>     保存频率 [default: None]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -m test
```
I am very appreciate to my best friend - [BlueFisher](https://github.com/BlueFisher) - who always look down on my coding style and competence(Although he **is** right, at least for now, but will be **was**).

Any questions about this project or errors about my bad grammer, plz let me know in [this](https://github.com/StepNeverStop/RLs/issues/new).