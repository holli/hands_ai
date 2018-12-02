# import ipdb
#import ipdb # ; ipdb.set_trace()
import pdb
import sys
import cv2
import random
import logging
import re
import time
import shutil
import numpy as np
import time
import datetime
import os
import glob
import math
import pickle
import json
import collections
from collections import defaultdict



def print_help():
    # for i, s in enumerate(cap_classes):
    #     print(str(i) + '     : ' + s )
    print("""
c     : toggle python console (for debugging)
t     : start_tracking
r     : reset tracking
esc   : exit
""")


def get_flat_list(obj):
    if not isinstance(obj, collections.Iterable):
        return [obj]
    res = []
    for o in obj:
        res += get_flat_list(o)
    return res

def scale_p(x, y, scale):
    return int(x*scale), int(y*scale)

def draw_label(img, lab, color, scale=1, thickness=2):
    if len(lab) < 4:
        cv2.circle(img, scale_p(lab[1], lab[2], scale), 5, color, thickness=thickness)
        y_text = lab[2]
    else:
        cv2_arrow(img, scale_p(lab[1], lab[2], scale), scale_p(lab[3], lab[4], scale), color, thickness)
        y_text = max(lab[2], lab[4])

    y_text += 30/scale
    cv2.putText(img, lab[0], scale_p(lab[1], y_text, scale), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def cv2_arrow(im_display, xy_start, xy_end, color, line_width, arrow_length=20):
    # cv2.line(im_display, (lab[1], lab[2]),(lab[3], lab[4]), (255,0,0), 2)
    cv2.line(im_display, xy_start, xy_end, color, line_width)

    angle = math.atan2(xy_end[1]-xy_start[1], xy_end[0]-xy_start[0]) + math.pi
    arrow_angl = 0.7
    x1 = int(xy_end[0] + arrow_length * math.cos(angle - arrow_angl))
    y1 = int(xy_end[1] + arrow_length * math.sin(angle - arrow_angl))
    x2 = int(xy_end[0] + arrow_length * math.cos(angle + arrow_angl))
    y2 = int(xy_end[1] + arrow_length * math.sin(angle + arrow_angl))
    cv2.line(im_display, (x1, y1), xy_end, color, line_width)
    cv2.line(im_display, (x2, y2), xy_end, color, line_width)


# Class for reading the video
# opencv skips duplicate frames, which results inconsistencies when
# jumping to specific frames https://github.com/opencv/opencv/issues/9053
# ffmpeg -i 2018-09-17_31_open_hand.mp4 tmp/%4d.jpg
class CaptureVideo(object):
    def __init__(self, video_fname):
        self.video_fname = video_fname
        self.restart()

    def restart(self):
        self.cap = cv2.VideoCapture(self.video_fname)
        self.frames = collections.deque(maxlen=1000)
        self.__idx = 0; self.delta = 0
        self.read()
        self.__idx = 0 # start index from zero
        self.eof = False

    def frame_idx(self):
        return self.__idx + self.delta

    def info(self):
        return f"Current frame idx {self.frame_idx()} (opencv next_frame: {self.cap.get(cv2.CAP_PROP_POS_FRAMES)}, delta: {self.delta})"

    def current(self):
        return self.frames[self.delta-1]

    def read(self):
        if self.delta > 0:
            raise f"CaptureVideo error. Positive delta {self.delta}"
    
        if self.delta == 0:
            ret, frame = self.cap.read()
            if ret:
                self.__idx += 1 
                self.frames.append(frame)
            else:
                self.eof = True
        elif self.delta < 0:
            self.delta += 1

        return self.current()
            
    def previous(self):
        if -self.delta + 1 < len(self.frames):
            self.delta -= 1
        else:
            print(f"CaptureVideo delta ({self.delta}) at first possible frame. Use reset (r) to start from beginning.")
        return self.current()
            


class VideoLabelFile(object):
    VIDEO_FNAME_EXTENSIONS = ('.mp4', '.avi')

    def __init__(self, video_fname, fname_add='frame_labels'):
        self.video_fname = video_fname
        self.fname_add = fname_add

    def get_fname(self):
        s = self.video_fname
        for rep in self.VIDEO_FNAME_EXTENSIONS: s = s.replace(rep, f"_{self.fname_add}_*.pk")
        return s

    def last_previous_fname(self):
        fnames = sorted(glob.glob(self.get_fname()))
        if fnames:
            return fnames[-1]
        else:
            return None

    def load_previous(self):
        prev_fname = self.last_previous_fname()
        if not prev_fname:
            return defaultdict(lambda: [])
        else:
            try:
                dic = pickle.load(open(prev_fname, "rb"))
                return defaultdict(lambda: [], dic)
            except:
                print(f"Error loading file {prev_fname}:", sys.exc_info()[0])
                print("\n")
                raise

    def get_next_fname(self):
        idx = 1
        prev_fname = self.last_previous_fname()
        if prev_fname:
            idx = int(re.search('(\d+)\.pk', prev_fname)[1]) + 1
        fname = self.get_fname().replace("*", "{:03d}").format(idx)
        return fname
    
    def save_current_labels(self, labels_current, labels_all_previous=None, append_previous=True, custom_lists=False):
        if labels_all_previous is None:
            if append_previous:
                labels_all_previous = self.load_previous()
            else:
                labels_all_previous = defaultdict(lambda: [])
        prev_count = len(labels_all_previous)
        frame_keys = sorted(set(list(labels_current.keys()) + list(labels_all_previous.keys())))
        dic = {}
        for f in frame_keys:
            f_labs = []
            if custom_lists:
                all_arr = labels_current[f] + labels_all_previous[f]
            else:
                all_arr = [labels_current[f]] + labels_all_previous[f]
            for x in all_arr:
                if x:
                    if not custom_lists:
                        x = [x[0]] + [int(i) for i in x[1:]]
                    if x not in f_labs: # remove duplicates and empty
                        f_labs.append(x)
            if f_labs:
                dic[f] = f_labs

        fname = self.get_next_fname()
        print(f"Total saved frames {len(dic)} (prev: {prev_count}). Saving to file {fname}.")

        pickle.dump(dic, open(fname, "wb"))
        return dic
