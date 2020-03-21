import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Flatten
from utils.tf2_utils import get_device
from Nn.layers import ConvLayer

activation_fn = 'tanh'

class ObsRNN(tf.keras.Model):
    '''输入状态的RNN
    '''
    def __init__(self, dim, hidden_units, batch_size, use_rnn):
        super().__init__()
        self.use_rnn = use_rnn
        if use_rnn:
            self.dim = dim
            self.batch_size = batch_size
            self.masking = tf.keras.layers.Masking(mask_value=0.)
            self.lstm_net = tf.keras.layers.LSTM(hidden_units, return_state=True, return_sequences=True)
            self(tf.keras.layers.Input(shape=(None,self.dim)))
            self.hdim = hidden_units
        else:
            self.hdim = dim

    def call(self, s, state=None, train=True):
        if self.use_rnn:
            if train:   # [B*T, ...] => [B, T, ...] 
                s = tf.reshape(s, [self.batch_size, -1, s.shape[-1]])
            else:
                s = tf.reshape(s, [-1, 1, s.shape[-1]])
            if state is None:
                x, h, c = self.lstm_net(s)
            else:
                x, h, c = self.lstm_net(s, state)
            x = tf.reshape(x, [-1, x.shape[-1]])    # [B, T, ...] => [B*T, ...]
            return (x, (h, c))
        else:
            return (s, None)

class VisualNet(tf.keras.Model):
    '''
    Processing image input observation information.
    If there has multiple cameras, Conv3D will be used, otherwise Conv2D will be used. The feature obtained by forward propagation will be concatenate with the vector input.
    If there is no visual image input, Conv layers won't be built and initialized.
    '''

    def __init__(self, vector_dim, visual_dim=[], visual_feature=128):
        super().__init__()
        self.vdl = len(visual_dim)
        if 2 < self.vdl < 5:
            if self.vdl == 4:
                self.net = ConvLayer(Conv3D, [32,64,64], [[1,8,8],[1,4,4],[1,3,3]], [[1,4,4],[1,2,2],[1,1,1]], padding='valid', activation='relu')
            elif self.vdl == 3:
                self.net = ConvLayer(Conv2D, [32,64,64], [[8,8],[4,4],[3,3]], [[4,4],[2,2],[1,1]], padding='valid', activation='relu')
            self.net.add(Dense(visual_feature, activation_fn))
            self.hdim = vector_dim + visual_feature
            self(tf.keras.Input(shape=visual_dim))
        else:
            self.net = lambda vs: vs
            self.hdim = vector_dim

    def call(self, vector_input, visual_input):
        # if len(visual_input.shape) == 5 and self.vdl == 3:   # [B, T, H, W, C]
        #     # LSTM, Conv2D后接LSTM时候的处理
        #     b = visual_input.shape[0]   # Batchsize
        #     visual_input = tf.reshape(visual_input, [-1]+list(visual_input.shape)[-3:]) # [B*T, H, W, C]
        #     f = self.net(visual_input)  # [B*T, Hidden]
        #     f = tf.reshape(f, [b, -1, self.hdim])   # [B, T, Hidden]
        # else:
        f = self.net(visual_input)
        f = tf.concat([vector_input, f], axis=-1)
        return f

class CuriosityModel(tf.keras.Model):
    '''
    Model of Intrinsic Curiosity Module (ICM).
    Curiosity-driven Exploration by Self-supervised Prediction, https://arxiv.org/abs/1705.05363
    '''

    def __init__(self, is_continuous, vector_dim, action_dim, visual_dim=[], visual_feature=128,
                 *, eta=0.2, lr=1.0e-3, beta=0.2, loss_weight=10.):
        '''
        params:
            is_continuous: sepecify whether action space is continuous(True) or discrete(False)
            vector_dim: dimension of vector state input
            action_dim: dimension of action
            visual_dim: dimension of visual state input
            visual_feature: dimension of visual feature map
            eta: weight of intrinsic reward
            lr: the learning rate of curiosity model
            beta: weight factor of loss between inverse_dynamic_net and forward_net
            loss_weight: weight factor of loss between policy gradient and curiosity model
        '''
        super().__init__()
        self.device = get_device()
        self.eta = eta
        self.beta = beta
        self.loss_weight = loss_weight
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        vdl = len(visual_dim)
        if vdl == 4 or vdl == 3:
            self.use_visual = True
            if vdl == 4:
                self.net = ConvLayer(Conv3D, [32,64,64], [[1,8,8],[1,4,4],[1,3,3]], [[1,4,4],[1,2,2],[1,1,1]], padding='valid', activation='elu')
            else:
                self.net = ConvLayer(Conv2D, [32,64,64], [[8,8],[4,4],[3,3]], [[4,4],[2,2],[1,1]], padding='valid', activation='elu')
            self.net.add(Dense(visual_feature, activation_fn))
            self.s_dim = visual_feature + vector_dim

            # S, S' => A
            self.inverse_dynamic_net = Sequential([
                Dense(self.s_dim*2, activation_fn),
                Dense(action_dim, 'tanh' if is_continuous else None)
            ])
        else:
            self.use_visual = False
            self.net = lambda vs: vs
            self.s_dim = vector_dim

        # S, A => S'
        self.forward_net = Sequential([
            Dense(self.s_dim+action_dim, activation_fn),
            Dense(self.s_dim, None)
        ]) 
        self.initial_weights(tf.keras.Input(shape=visual_dim), tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=action_dim))

        self.tv = []
        if self.use_visual:
            self.tv += self.net.trainable_variables
            self.tv += self.inverse_dynamic_net.trainable_variables
        self.tv += self.forward_net.trainable_variables   


    def initial_weights(self, visual_input, vector_input, action):
        f = self.net(visual_input)
        s = tf.concat((f, vector_input), -1)
        if self.use_visual:
            self.inverse_dynamic_net(tf.concat((s, s), -1))
        self.forward_net(tf.concat((s, action), -1))

    @tf.function(experimental_relax_shapes=True)
    def call(self, s, visual_s, a, s_, visual_s_):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                fs = self.net(visual_s)
                fs_ = self.net(visual_s_)

                fsa = tf.concat((fs, s, a), axis=-1)            # <S, A>
                s_target = tf.concat((fs_, s_), axis=-1)        # S'
                s_eval = self.forward_net(fsa)                  # <S, A> => S'
                LF = 0.5 * tf.reduce_sum(tf.square(s_target - s_eval), axis=-1, keepdims=True)    # [B, 1]
                intrinsic_reward = self.eta * LF
                loss_forward = tf.reduce_mean(LF)

                if self.use_visual:
                    f = tf.concat((fs, s, fs_, s_), axis=-1)
                    a_eval = self.inverse_dynamic_net(f)
                    if self.is_continuous:
                        loss_inverse = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(a_eval-a), axis=-1))
                    else:
                        idx = tf.argmax(a, axis=-1) #[B, ]
                        loss_inverse = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=idx, logits=a_eval))
                    loss = (1-self.beta)*loss_inverse+self.beta*loss_forward
                else:
                    loss = loss_forward
            
            grads = tape.gradient(loss, self.tv)
            self.optimizer.apply_gradients(zip(grads, self.tv))
            summaries = dict([
                ['LOSS/curiosity_loss', loss],
                ['LOSS/forward_loss', loss_forward]
            ])
            if self.use_visual:
                summaries.update({
                    'LOSS/inverse_loss': loss_inverse
                })
            return intrinsic_reward, loss*self.loss_weight, summaries
