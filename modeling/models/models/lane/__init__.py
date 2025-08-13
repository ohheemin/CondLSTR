from .cond_lstr_2d_res10 import CondLSTR2DRes10
from .cond_lstr_2d_res18 import CondLSTR2DRes18
from .cond_lstr_2d_res34 import CondLSTR2DRes34

__factory = {
    'CondLSTR2DRes10': CondLSTR2DRes10,
    'CondLSTR2DRes18': CondLSTR2DRes18,
    'CondLSTR2DRes34': CondLSTR2DRes34,
}

def names():
    return list(__factory.keys())

def create(name, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)

