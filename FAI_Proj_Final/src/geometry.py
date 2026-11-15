"""
Arena boundary geometry helpers shared by the tracker, predictor and firing solver.

The arena is bounded (ArenaConfig.toroidal_wrap = False): asteroids bounce off the
walls elastically instead of wrapping to the opposite side.  Any prediction that
extrapolates a straight line past a wall must therefore *reflect* the point back
into the arena (and flip the corresponding velocity component) rather than taking
a modulo, which would teleport the predicted object to the far side of the screen.
"""

from typing import Tuple
import numpy as np


def fold_axis(x: float, v: float, lo: float, hi: float) -> Tuple[float, float, bool]:
    """
    Reflect a 1-D coordinate into [lo, hi] as a perfectly elastic wall would.
    Returns (folded_x, folded_v, flipped) where flipped is True when an odd number of
    reflections occurred (velocity sign reversed).
    """
    span = hi - lo
    if span <= 0.0:
        return float(np.clip(x, min(lo, hi), max(lo, hi))), v, False
    m = (x - lo) % (2.0 * span)
    if m <= span:
        return lo + m, v, False
    return lo + 2.0 * span - m, -v, True


def reflect_into_arena(
    pos: np.ndarray,
    vel: np.ndarray,
    width: float,
    height: float,
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[bool, bool]]:
    """
    Reflect a 2-D position/velocity pair into the bounded arena
    [margin, width - margin] x [margin, height - margin].
    Returns (position, velocity, (flipped_x, flipped_y)).
    """
    x, vx, fx = fold_axis(float(pos[0]), float(vel[0]), margin, width - margin)
    y, vy, fy = fold_axis(float(pos[1]), float(vel[1]), margin, height - margin)
    return np.array([x, y], dtype=np.float64), np.array([vx, vy], dtype=np.float64), (fx, fy)


def extrapolate(
    pos: np.ndarray,
    vel: np.ndarray,
    seconds: float,
    width: float,
    height: float,
    wrap: bool = False,
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-velocity extrapolation honouring the arena boundary model."""
    raw = np.asarray(pos, dtype=np.float64) + np.asarray(vel, dtype=np.float64) * seconds
    if wrap:
        return np.array([raw[0] % width, raw[1] % height], dtype=np.float64), np.asarray(vel, dtype=np.float64).copy()
    p, v, _ = reflect_into_arena(raw, np.asarray(vel, dtype=np.float64), width, height, margin)
    return p, v
