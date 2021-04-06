# Example of using the util functions:

#-----------
import os, sys
import numpy as np
import cv2
#-----------
from utils.utils import *
#-----------

def main():

    # prepare a sample iamge
    blank_image = np.random.randint(0, 256, 1280 * 720 * 3, dtype=np.uint8).reshape((720, 1280, 3))
    blank_image = cv2.morphologyEx(blank_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    blank_image = np.clip(blank_image * 0.1, 0, 255)
    blank_image = cv2.blur(blank_image, (31, 31))
    blank_image = np.interp(blank_image, (blank_image.min(), blank_image.max()), (0, 255)).astype(np.uint8)

    while True:

        display_image = blank_image.copy()
        
        # show just one line:
        draw_text(display_image, "string in left corner", (0, 0))        
        
        draw_text(display_image, "string in right corner", (1280, 0), align=Align.RIGHT, font_width=2, font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=1)
        draw_text(display_image, "string in the middle", (1280 // 2, 20), align=Align.CENTER, font_width=2, font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=2, font_color=COLOR_NAVY)
        
        draw_text(display_image, "string with red outline", (1280 // 2, 90), align=Align.CENTER, font_width=2, font_face=cv2.FONT_HERSHEY_DUPLEX, 
                  font_size=1, font_color=COLOR_WHITE, outline=True, outline_width=3, outline_color=COLOR_RED)
        
        # show a bunch of strings with a background and more options:
        text = ["This text has a dark background", 
                "With a black outline and antialiasing", 
                "And padded 30 horizontally, 20 vertically and 10 internally"]
        draw_text(display_image, text, (1280 // 2, 150), 
                  align=Align.CENTER, font_width=2, font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=0.75, font_color=COLOR_WHITE,
                  outline=True, outline_width=4, outline_color=COLOR_BLACK, line_type=cv2.LINE_AA,
                  darken_background=0.5, inner_pad=10, outer_pad=[30, 20])
        
        # put text with a coloured background, and show username:
        draw_text(display_image, f"Current user: '{get_username()}'", (100, 300), 
                  font_width=2, font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=1, font_color=COLOR_WHITE,
                  background_color=COLOR_BLUE)
        
        # draw a long string, and split it automatically:
        draw_text(display_image, split_long_line("this text with be automatically split into several lines of a maximum size of 20 characters", 20), org=(1280-50, 400), 
                  align=Align.RIGHT, valign=Valign.MIDDLE,
                  font_width=2, font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=0.8, font_color=COLOR_GREEN,
                  background_color=COLOR_PURPLE, darken_background=0.7)
        
        # put strings in bottom left corner
        draw_text(display_image, ["bunch of strings", "in bottom left corner", "not sorted"], (0, 720), 
                  valign=Valign.BOTTOM, font_width=1, font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=1)
        
        # put strings in bottom right corner, with sorting, padding and dark background
        draw_text(display_image, ["by length", "these three strings", "are sorted"], (1280-10, 720-10), 
                  align=Align.RIGHT, valign=Valign.BOTTOM, sort=Sorting.DSCN,
                  font_width=2, font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=1, font_color=COLOR_RED, 
                  darken_background=0.5, inner_pad=10, outer_pad=[20, 30])
        
        cv2.imshow("image", display_image)

        key = cv2.waitKeyEx(1)

        if key == KEY_ESC or is_key(key, 'Q', case_sensitive=False):
            break

#-----------
if __name__ == "__main__":
    main()

