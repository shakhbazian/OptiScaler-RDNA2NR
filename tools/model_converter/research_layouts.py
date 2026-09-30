"""Address-only research helpers for pinned window-family layout candidates.

Bijective addresses prove implementation consistency, not original model semantics.
"""
import numpy as np
from qmma_layout import physical_offsets


def n128_offsets(inputs, outputs):
    if inputs <= 0 or inputs % 32 or outputs <= 0 or outputs % 128:
        raise ValueError('n128 layout requires K multiple of 32, N multiple of 128')
    row = np.arange(inputs)[:, None]
    col = np.arange(outputs)[None, :]
    within = physical_offsets(row, col, outputs) % 1024
    tile = (col//128*(inputs//32)+row//32)*4+col%128//32
    return tile*1024+within


def hmma_offsets(outputs):
    if outputs <= 0 or outputs % 16:
        raise ValueError('HMMA layout requires N multiple of 16')
    row = np.arange(16)[:, None]
    col = np.arange(outputs)[None, :]
    lane = col%8*4+row%8//2
    element = col%16//8*4+row//8*2+row%2
    return col//16*256+lane*8+element


def bias_offsets():
    """Physical accumulator offset for every logical 8x8-window token pair."""
    token = np.arange(64)
    y, x = token//8, token%8
    internal = (y//4*2+x//4)*16+y%4*4+x%4
    q, key = internal[:, None], internal[None, :]
    tile = q//16*4+key//16
    lane = q%8*4+key%8//2
    element = key%16//8*4+q%16//8*2+key%2
    return tile*256+lane*8+element
