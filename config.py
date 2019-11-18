import os
import time
import platform
import tensorflow as tf

unity_file = {
    '3DBall': {
        'simple500': r'C:\UnityBuild\Ball\simple500\train.exe',
        'hard1': r'C:\UnityBuild\Ball\hard1\hard1.exe',
        'hard3': r'C:\UnityBuild\Ball\hard3\hard3.exe'
    },
    'RollerBall': {
        'OneFloor': r'C:/UnityBuild/RollerBall/OneFloor/RollerBall-custom.exe',
        'PureCamera': r'C:\UnityBuild\RollerBall\PureCamera\train.exe'
    },
    'Boat': {
        'first': r'C:/UnityBuild/Boat/first/BoatTrain.exe',
        'second': r'C:/UnityBuild/Boat/second/BoatTrain.exe',
        'interval1': r'C:/UnityBuild/Boat/interval1/BoatTrain.exe',
        'no_border': r'C:/UnityBuild/Boat/no_border/BoatTrain.exe',
        'no_border2': r'C:/UnityBuild/Boat/no_border2/BoatTrain.exe'
    }
}

train_config = {
    'name': time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())),
    'save_frequency': 20,
    'max_step': 2000,
    'max_episode': 5000,
    # share_args
    'share': {
        'base_dir': f'C:/RLData/tf{tf.__version__[0]}' if platform.system() == "Windows" else os.environ['HOME'] + f'/RLData/tf{tf.__version__[0]}',
        'logger2file': False,
        'out_graph': False,
        'ma': {
            'batch_size': 10,
            'capacity': 1000
        }
    },
    # unity default_args
    'unity': {
        'no_op_steps': 100,
        'exe_file': unity_file['3DBall']['simple500'],
        'reset_config': {
            #    'copy': 10
        },
    },
    # gym default_args
    'gym': {
        'random_steps': 10000,
        'render': False,
        'render_episode': 50000,
        'render_mode': 'random_1', # first, last, [list], random_[num], or all.
        'eval_while_train': False,
        'max_eval_episode': 100,
    },

}
