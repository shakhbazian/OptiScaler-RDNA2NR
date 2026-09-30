"""Decode the packed E4M3 matrix layout used by the supported model version.

The row permutation was checked against the matching model's weight records.
Layout research references MLX-DLSS 0ca2deab (Apache-2.0).
"""
import math
import numpy as np


def physical_offsets(k, n, outputs, permute_rows=True):
    # Rotate bits 1..3 inside each 16-row group: [0,1,4,5,...,14,15].
    k_fragment = (k // 16 * 16 + k % 2 + (k % 8 // 2) * 4 + (k % 16 // 8) * 2
                  if permute_rows else k)
    tile = (k // 32 * (outputs // 32) + n // 32) * 1024
    pair = n % 32 // 16
    lane = (n % 8) * 4 + k_fragment % 16 // 4
    fragment = n % 16 // 8
    element = k_fragment % 4 + (k_fragment % 32 // 16) * 4
    return tile + pair * 512 + lane * 16 + fragment * 8 + element


def unpack(packed, inputs, outputs, *, permute_rows=True):
    if inputs <= 0 or outputs <= 0 or inputs % 32 or outputs % 32:
        raise ValueError('K and N must be positive multiples of 32')
    if inputs * outputs > 16 * 1024 * 1024:
        raise ValueError('matrix exceeds research limit')
    raw = np.frombuffer(packed, dtype=np.uint8)
    if raw.size != inputs * outputs:
        raise ValueError('packed matrix byte count mismatch')
    k = np.arange(inputs, dtype=np.int64)[:, None]
    n = np.arange(outputs, dtype=np.int64)[None, :]
    return np.ascontiguousarray(raw[physical_offsets(k, n, outputs, permute_rows)])


def decode_e4m3(raw):
    table = []
    for x in range(256):
        exponent, fraction = (x >> 3) & 15, x & 7
        value = (math.nan if exponent == 15 and fraction == 7 else
                 math.ldexp(fraction, -9) if exponent == 0 else
                 math.ldexp(1 + fraction / 8, exponent - 7))
        table.append(math.copysign(value, -1 if x & 128 else 1))
    return np.array(table, dtype=np.float32)[np.asarray(raw, dtype=np.uint8)]
