from .base_oscillation import BaseOscillation
from .cosine import Cosine
from .custom_input import CustomInput
from .cylinder_bell_funnel import CylinderBellFunnel
from .dirichlet import Dirichlet
from .ecg import ECG
from .formula import Formula  # type: ignore  # mypy ends up in recursion
from .interface import BaseOscillationInterface
from .mls import MLS
from .polynomial import Polynomial
from .random_mode_jump import RandomModeJump
from .random_walk import RandomWalk
from .sawtooth import Sawtooth
from .shared_noise_sine import SharedNoiseSine
from .sine import Sine
from .square import Square
