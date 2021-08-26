from .vdn import VDNMixer
from .qmix import QMixer

Mixer_REGISTER = {}

Mixer_REGISTER['vdn'] = VDNMixer
Mixer_REGISTER['qmix'] = QMixer
