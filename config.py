import platform

unity_file = {
    '3DBall': {

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
    'base_dir': r'C:/RLData' if platform.system() == "Windows" else r'/RLData',
    'exe_file': unity_file['RollerBall']['PureCamera'],
    'reset_config': {
        'copy': 5
    },
    'save_frequency': 20,
    'max_episode': 50000,
    'max_step': 10000,
    'name': '0'
}
