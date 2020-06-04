import tensorflow as tf

class ConsistentLearningRate:
    
    def __init__(self, lr, *args, **kwargs):
        self.lr = lr

    def __call__(self, *args, **kwargs):
        return self.lr

