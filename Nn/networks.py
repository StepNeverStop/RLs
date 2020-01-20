import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Flatten

activation_fn = 'tanh'

class VisualNet(tf.keras.Model):
    '''
    Processing image input observation information.
    If there has multiple cameras, Conv3D will be used, otherwise Conv2D will be used. The feature obtained by forward propagation will be concatenate with the vector input.
    If there is no visual image input, Conv layers won't be built and initialized.
    '''

    def __init__(self, name, visual_dim=[]):
        super().__init__(name=name)
        self.vd = len(visual_dim) + 1
        if len(visual_dim) == 4:
            self.net = Sequential([
                Conv3D(filters=32, kernel_size=[1, 8, 8], strides=[1, 4, 4], padding='valid', activation='relu'),
                Conv3D(filters=64, kernel_size=[1, 4, 4], strides=[1, 2, 2], padding='valid', activation='relu'),
                Conv3D(filters=64, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='valid', activation='relu'),
                Flatten(),
                Dense(128, activation_fn)
            ])
            self.hdim = 128
            self(tf.keras.Input(shape=visual_dim))
        elif len(visual_dim) == 3:
            self.net = Sequential([
                Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding='valid', activation='relu'),
                Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding='valid', activation='relu'),
                Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='valid', activation='relu'),
                Flatten(),
                Dense(128, activation_fn)
            ])
            self.hdim = 128
            self(tf.keras.Input(shape=visual_dim))
        else:
            self.net = lambda vs: vs
            self.hdim = 0

    def call(self, visual_input):
        if len(visual_input.shape) == 5 and self.vd == 4:
            # LSTM
            b = visual_input.shape[0]
            visual_input = tf.reshape(visual_input, [-1]+list(visual_input.shape)[-3:])
            f = self.net(visual_input)
            f = tf.reshape(f, [b, -1, self.hdim])
        else:
            f = self.net(visual_input)
        return f