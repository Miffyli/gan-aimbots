#!/usr/bin/env python3
#
# windows_mouse.py 
# Tools to emulate mouse movement on Windows
#

from ctypes import *
from ctypes.wintypes import DWORD, LONG, ULONG, WORD

M1_VK = 0x1
M2_VK = 0x2
MOUSEEVENTF_MOVE        = 0x0001 #/* mouse move */
MOUSEEVENTF_ABSOLUTE    = 0x8000 #/* absolute move */
MOUSEEVENTF_LEFTDOWN    = 0x0002
MOUSEEVENTF_LEFTUP      = 0x0004
MOUSEEVENTF_RIGHTDOWN   = 0x0008
MOUSEEVENTF_RIGHTUP     = 0x0010
INPUT_MOUSE = 0
INPUT_KEYBD = 1

class MOUSEINPUT(Structure):
    _fields_ = [ ("dx", LONG),
                 ("dy", LONG),
                 ("mouseData", DWORD),
                 ("dwFlags", DWORD),
                 ("time", DWORD),
                 ("dwExtraInfo", POINTER(ULONG))
                ]

class KEYBDINPUT(Structure):
    _fields_ = [ ("wVk", WORD),
                 ("wScan", WORD),
                 ("dwFlags", DWORD),
                 ("time", DWORD),
                 ("dwExtraInfo", POINTER(ULONG))
                ]
    
class HARDWAREINPUT(Structure):
    _fields_ = [ ("uMsg", DWORD),
                 ("wParamL", WORD),
                 ("wParamH", WORD),
                ]

class _INPUT_UNION(Union):
    _fields_ = [("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT),
                ]

class INPUT(Structure):
    _anonymous_ = ("iu",)
    _fields_ = [ ("type", DWORD),
                 ("iu", _INPUT_UNION)]
minput = INPUT()
minput.type = INPUT_MOUSE
minput.mi.dwFlags = MOUSEEVENTF_MOVE

def move_mouse(delta_x, delta_y):
    global minput
    if delta_x == 0 and delta_y == 0:
        return (0, 0)

    minput.mi.dx = delta_x
    minput.mi.dy = delta_y
    windll.User32.SendInput(1, byref(minput), sizeof(minput))
