import os, sys
import numpy as np
import cv2
import time

from utils.utils import *

vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

def __video_time_remaining(frame_count, fps, current_frame=0):
    """
    Returns the Hours, Minutes and Seconds remaining in the video of a given number of frames, FPS and current frame number
    """
    seconds = (frame_count - current_frame) / fps

    minutes = seconds // 60
    seconds = seconds - (minutes * 60)

    hours = minutes // 60
    minutes = minutes - (hours * 60)

    return int(hours), int(minutes), seconds

def play_video(path_to_video, video_name=None):

    if path_to_video is None:
        return

    if not os.path.exists(path_to_video):
        return
    
    if os.path.splitext(path_to_video)[-1] not in vid_formats:
        return

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        return

    # get video properties
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get window name for the video (if not given, use the file name with no extension)
    video_name = os.path.splitext(os.path.basename(path_to_video))[0] if video_name is None else video_name

    cv2.namedWindow(video_name, cv2.WINDOW_AUTOSIZE)

    for i in range(frame_count):
        ret, frame = cap.read()

        if not ret or not is_valid_image(frame):
            continue

        # resize the frame to a more uniform size (pad, if necessary)
        frame_resized = resize_ratio(frame, (720, 1280), pad_value=0)[0]
        hi, wi = frame_resized.shape[:2]
        
        # add a seek-bar:
        seek_bar = np.ones((40, wi, 3), dtype=np.uint8) * 30
        cv2.rectangle(seek_bar, (0, 0), (int(wi * ((i + 1) / frame_count)), seek_bar.shape[0]), COLOR_MAROON, -1)

        # get the hours:minutes:seconds remaining and draw them on the seek bar
        h,m,s = __video_time_remaining(frame_count, fps, i)
        draw_text(seek_bar, f"{h}:{m:02}:{s:04.1f}", (seek_bar.shape[1] // 2, seek_bar.shape[0] // 2), 
                  align=Align.CENTER, valign=Valign.MIDDLE,
                  font_face=cv2.FONT_HERSHEY_DUPLEX, font_size=1, font_width=2,
                  outline=True)
        
        # add the seek bar at the bottom of the frame
        frame_resized = np.vstack([frame_resized, seek_bar])

        cv2.imshow(video_name, frame_resized)

        key = cv2.waitKey(1)
        if key == KEY_ESC:
            break

    cv2.destroyWindow(video_name)