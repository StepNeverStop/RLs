import os
import time
import json
import tensorflow as tf
from tensorflow.core.util import event_pb2

DIR = r''   # where to save .json file.
BASE_DIR = r''
BASE_TAG = [] # ['MAIN/reward_max', 'MAIN/reward_mean', 'MAIN/reward_min', 'MAIN/sma_max', 'MAIN/sma_mean', 'MAIN/sma_min', 'MAIN/step']
file_paths = [
    {
      'path': r'*.v2',  # tf2 summary file path.
      'tags': ['loss/actor_loss']   # tags needed to load and save to json.
    },
]

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def tf2summary2dict(path, tags=[]):
    serialized_examples = tf.data.TFRecordDataset(path)
    data = {}
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            if value.tag in tags or tags == []:
                t = tf.make_ndarray(value.tensor)
                t = float(t)
                try:
                    data[f'{value.tag}'].append([t, event.step])
                except:
                    data[f'{value.tag}'] = [[t, event.step]]
                    pass
    return data

def data2json(directory, data):
    for k, v in data.items():
        mkdir(directory=directory)
        file_name = directory + k.replace('/', '-').replace('\\', '-') + '.json'
        with open(file_name, 'a') as f:
            json.dump(v, f)

def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)
        value, step = list(zip(*data))
        return (value, step)
    return None

def search_file(path, text):
    try:
        files = os.listdir(path)
        file_path_list = []
        for f in files:
            f1 = os.path.join(path, f)
            if os.path.isdir(f1):
                file_path_list.extend(search_file(f1, text))
            elif os.path.isfile(f1) and os.path.splitext(f1)[1] == text:
                file_path_list.append(f1)
        return file_path_list
    except Exception as e:
        print(e)

if __name__ == "__main__":
    if BASE_DIR != '':
        file_paths = search_file(BASE_DIR, '.v2')
        file_paths = [{'path': path, 'tags':BASE_TAG} for path in file_paths]

    for index, d in enumerate(file_paths):
        path = d.get('path')
        if os.path.exists(path):
            if DIR == '':
                save_dir = (path.split('events.out.tfevents')[0]+ 'json_data/').strip()
            else:
                save_dir = DIR
            tags = d.get('tags', [])
            data = tf2summary2dict(path=path, tags=tags)
            data2json(directory=save_dir, data=data)
            print(f'The number {index} summary has transformed to .json successfully.')
        else:
            print(f'path: {path} could not find summary file.')