#!/usr/bin/env python3
#
# mouse_emulation.py
# Tools to emulate mouse movement.
#

import os
import sys

sys_move_mouse = None
if sys.platform == "win32":
    from .win_mouse_tools import move_mouse
    sys_move_mouse = move_mouse
else:
    raise NotImplementedError("Only Windows platform is supported for running aimbots.")

def move_mouse(delta_x, delta_y):
    """
    Move mouse by delta_x, delta_y amount, relative
    to current position.

    Positive delta_x: Right (on screen)
    Positive delta_y: Down (on screen)
    """
    if delta_x == 0 and delta_y == 0:
        return

    sys_move_mouse(int(round(delta_x)), int(round(delta_y)))
