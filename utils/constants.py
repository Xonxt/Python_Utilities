#----------------------
import os

#----------------------
# KEYCODES
KEY_ENTER = 13
KEY_SPACEBAR = 32
KEY_ESC = 27
KEY_BACKSPACE = 8
KEY_TAB = 9

KEY_F1 = 7340032 if os.name == 'nt' else 65470
KEY_F2 = 7405568 if os.name == 'nt' else 65471
KEY_F3 = 7471104 if os.name == 'nt' else 65472
KEY_F4 = 7536640 if os.name == 'nt' else 65473

KEY_ARROW_LEFT = 2424832  if os.name == 'nt' else 65361
KEY_ARROW_RIGHT = 2555904 if os.name == 'nt' else 65363
KEY_ARROW_UP = 2490368    if os.name == 'nt' else 65362
KEY_ARROW_DOWN = 2621440  if os.name == 'nt' else 65364

KEY_PAGE_UP = 2162688   if os.name == 'nt' else 65365
KEY_PAGE_DOWN = 2228224 if os.name == 'nt' else 65366

KEY_DEL = 3014656 if os.name == 'nt' else 65535
KEY_HOME = 2359296
KEY_END = 2293760
KEY_INSERT = 2949120

# PRIMARY COLORS
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_GOLD = (0, 215, 255)
COLOR_MAROON = (0, 0, 128)
COLOR_PURPLE = (128, 0, 128)
COLOR_NAVY = (128, 0, 0)
COLOR_TEAL = (128, 128, 0)
COLOR_ORANGE = (0, 69, 255)
COLOR_OLIVE = (0, 128, 128)
COLOR_KHAKI = (140, 230, 240)
COLOR_FOREST = (0, 128, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREY = (128, 128, 128)
COLOR_DARK_GREY = (200, 200, 200)
COLOR_BLACK = (0, 0,0 )