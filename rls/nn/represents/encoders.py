

from torch.nn import Identity, Linear, Sequential

from rls.nn.activations import Act_REGISTER, default_act

End_REGISTER = {}


class EncoderIdentityNetwork(Sequential):

    def __init__(self, in_dim, *args, **kwargs):
        super().__init__()
        self.h_dim = self.in_dim = in_dim
        self.add_module(f'identity', Identity())


class EncoderMlpNetwork(Sequential):

    def __init__(self, in_dim, h_dim=16, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.add_module('linear', Linear(self.in_dim, self.h_dim))
        self.add_module('activation', Act_REGISTER[default_act]())


End_REGISTER['identity'] = EncoderIdentityNetwork
End_REGISTER['mlp'] = EncoderMlpNetwork
