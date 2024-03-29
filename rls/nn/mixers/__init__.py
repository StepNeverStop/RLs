from .qatten import QattenMixer
from .qmix import QMixer
from .qplex.qplex import QPLEXMixer
from .qtran_base import QTranBase
from .vdn import VDNMixer

Mixer_REGISTER = {}

Mixer_REGISTER['vdn'] = VDNMixer
Mixer_REGISTER['qmix'] = QMixer
Mixer_REGISTER['qatten'] = QattenMixer
Mixer_REGISTER['qtran-base'] = QTranBase
Mixer_REGISTER['qplex'] = QPLEXMixer
