import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from utils.tf2_utils import get_device
from Nn.layers import ConvLayer

activation_fn = 'tanh'

class ObsRNN(tf.keras.Model):
    '''输入状态的RNN
    '''
    def __init__(self, dim, hidden_units, use_rnn):
        super().__init__()
        self.use_rnn = use_rnn
        if use_rnn:
            self.dim = dim
            # self.masking = tf.keras.layers.Masking(mask_value=0.)
            self.lstm_net = tf.keras.layers.LSTM(hidden_units, return_state=True, return_sequences=True)
            self(tf.keras.layers.Input(shape=(None, self.dim)))
            self.hdim = hidden_units
        else:
            self.hdim = dim

    def call(self, s, state=None):
        if self.use_rnn:
            # s = self.masking(s)
            if state is None:
                x, h, c = self.lstm_net(s) # 如果没指定初始化隐状态，就用burn_in的， 或者 None
            else:
                x, h, c = self.lstm_net(s, state)
            return (x, (h, c))
        else:
            return (s, None)


class VisualNet(tf.keras.Model):
    '''
    Processing image input observation information.
    The feature obtained by forward propagation will be concatenate with the vector input.
    If there is no visual image input, Conv layers won't be built and initialized.
    '''

    def __init__(self, vector_dim, visual_dim=[], visual_feature=128):
        super().__init__()
        if len(visual_dim) == 0:
            self.camera_num = 0
        else:
            self.camera_num = visual_dim[0]
        
        self.nets = []
        for _ in range(self.camera_num):
            net = ConvLayer(Conv2D, [32,64,64], [[8,8],[4,4],[3,3]], [[4,4],[2,2],[1,1]], padding='valid', activation='relu')
            net.add(Dense(visual_feature, activation_fn))
            self.nets.append(net)

        self.hdim = vector_dim + (visual_feature * self.camera_num) * (self.camera_num > 0)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        f = [self.nets[i](visual_input[:, i]) for i in range(self.camera_num)]
        f = tf.concat([vector_input, *f], axis=-1)
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
        self.is_continuous = is_continuous

        if len(visual_dim) == 0:
            self.use_visual = False
            self.camera_num = 0
        else:
            self.use_visual = True
            self.camera_num = visual_dim[0]
        
        self.nets = []
        for _ in range(self.camera_num):
            net = ConvLayer(Conv2D, [32,64,64], [[8,8],[4,4],[3,3]], [[4,4],[2,2],[1,1]], padding='valid', activation='elu')
            net.add(Dense(visual_feature, activation_fn))
            self.nets.append(net)

        self.s_dim = vector_dim + (visual_feature * self.camera_num) * (self.camera_num > 0)

        if self.use_visual:
            # S, S' => A
            self.inverse_dynamic_net = Sequential([
                Dense(self.s_dim*2, activation_fn),
                Dense(action_dim, 'tanh' if is_continuous else None)
            ])

        # S, A => S'
        self.forward_net = Sequential([
            Dense(self.s_dim+action_dim, activation_fn),
            Dense(self.s_dim, None)
        ]) 
        self.initial_weights(tf.keras.Input(shape=visual_dim), tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=action_dim))

        self.tv = []
        if self.use_visual:
            for net in self.nets:
                self.tv += net.trainable_variables
            self.tv += self.inverse_dynamic_net.trainable_variables
        self.tv += self.forward_net.trainable_variables   


    def initial_weights(self, visual_input, vector_input, action):
        f = [self.nets[i](visual_input[:, i]) for i in range(self.camera_num)]
        s = tf.concat((*f, vector_input), -1)
        if self.use_visual:
            self.inverse_dynamic_net(tf.concat((s, s), -1))
        self.forward_net(tf.concat((s, action), -1))

    @tf.function(experimental_relax_shapes=True)
    def call(self, s, visual_s, a, s_, visual_s_):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                fs = [self.nets[i](visual_s[:, i]) for i in range(self.camera_num)]
                fs_ = [self.nets[i](visual_s_[:, i]) for i in range(self.camera_num)]

                fsa = tf.concat((*fs, s, a), axis=-1)            # <S, A>
                s_target = tf.concat((*fs_, s_), axis=-1)        # S'
                s_eval = self.forward_net(fsa)                  # <S, A> => S'
                LF = 0.5 * tf.reduce_sum(tf.square(s_target - s_eval), axis=-1, keepdims=True)    # [B, 1]
                intrinsic_reward = self.eta * LF
                loss_forward = tf.reduce_mean(LF)

                if self.use_visual:
                    f = tf.concat((*fs, s, *fs_, s_), axis=-1)
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
