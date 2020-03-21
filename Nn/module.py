import tensorflow as tf


class VisualObsRNN(tf.keras.Model):
    def __init__(
        self, 
        net,
        visual_net, 
        visual_net_grad=True,
        visual_net_update=True,
        rnn_net=None, 
        rnn_net_grad=True,
        rnn_net_update=True
        ):
        '''
        神经网络模组，CNN->RNN->后续网络

        param net: 后续网络模型
        param visual_net: 处理图像的CNN部分网络
        param visual_net_grad: 指定该模组是否对CNN部分传导梯度
        param visual_net_updata: 指定在具有target network时，是否对CNN部分进行soft/hard更新
        param rnn_net: 处理局部可观测的RNN部分网络
        param rnn_net_grad: 指定该模组是否对RNN部分传导梯度
        param rnn_net_update: 指定在具有target network时，是否对RNN部分进行soft/hard更新
        '''
        super().__init__()
        self.net = net
        self.visual_net = visual_net
        self.visual_net_grad = visual_net_grad
        self.visual_net_update = visual_net_update
        self.rnn_net = rnn_net
        self.rnn_net_grad = rnn_net_grad
        self.rnn_net_update = rnn_net_update
        self._cell_state = None # RNN部分cell的隐藏状态
    
    def _call(self, vector_input, visual_input, *args, train=True, use_cs=False, record_cs=False, **kwargs):
        state = self.visual_net(vector_input, visual_input)
        state, cell_state = self.rnn_net(state, self._cell_state, train=train) if use_cs else self.rnn_net(state, train=train)
        if record_cs:
            self._cell_state = cell_state
        ret = self.net(state, *args, **kwargs)
        return ret

    def call(self, vector_input, visual_net, *args, **kwargs):
        return self._call(vector_input, visual_net, *args, train=True, use_cs=False, record_cs=False, **kwargs)

    def choose(self, vector_input, visual_net, *args, **kwargs):
        return self._call(vector_input, visual_net, *args, train=False, use_cs=True, record_cs=True, **kwargs)

    def get_cell_state(self):
        return self._cell_state

    @property
    def weights(self):
        return self.visual_net.weights + self.net.weights + self.rnn_net.weights

    @property
    def tv(self):
        tv = self.net.trainable_variables
        if self.visual_net_update:
            tv += self.visual_net.trainable_variables
        if self.rnn_net_grad:
            tv += self.rnn_net.trainable_variables
        return tv

    @property
    def uv(self):
        tv = self.net.weights
        if self.visual_net_grad:
            tv += self.visual_net.weights
        if self.rnn_net_update:
            tv += self.rnn_net.weights
        return tv