"""Paper B command curriculum helpers."""

from __future__ import annotations

import math


def paper_b_vx_max(step, initial_max=1.0, final_max=3.5, k=0.002, midpoint=1000):
    return initial_max + (final_max - initial_max) / (1.0 + math.exp(-k * (step - midpoint)))


def paper_b_vx_range(step, initial_range=(-0.5, 1.0), final_range=(-1.75, 3.5), k=0.002, midpoint=1000):
    vx_max = paper_b_vx_max(step, initial_range[1], final_range[1], k, midpoint)
    vx_max = min(max(vx_max, initial_range[1]), final_range[1])
    negative_ratio = abs(final_range[0]) / final_range[1]
    vx_min = -negative_ratio * vx_max
    return max(vx_min, final_range[0]), vx_max
