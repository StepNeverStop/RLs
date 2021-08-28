from .qatten import QattenMixer
from .qmix import QMixer
from .vdn import VDNMixer

Mixer_REGISTER = {}

Mixer_REGISTER['vdn'] = VDNMixer
Mixer_REGISTER['qmix'] = QMixer
Mixer_REGISTER['qatten'] = QattenMixer
