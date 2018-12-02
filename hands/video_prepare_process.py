import warnings
import argparse
from hands.basics import *
from hands.utils import *
import cv2

"""
use instead of "ret, im_input_fullsize = cap.read()"

from video_threaded_prepare import *
capt = CaptureVideoThreaded()
capt.start()
time.sleep(1)

im_input = capt.read()

capt.stop()
capt.join(2)
"""

# https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
# https://docs.python.org/3/library/multiprocessing.html


from multiprocessing import Process, Value
from multiprocessing import RawArray, Array
import multiprocessing.sharedctypes
import ctypes
import sys
import cv2
import time

class VideoCaptureFile(cv2.VideoCapture):
    def __init__(self, s, size):
        super().__init__(s)
        self.size = size
        self.is_file = True

    def read(self):
        grabbed, im_input = super().read()
        if grabbed:
            im_input = im_crop_and_resize(im_input, self.size)

        return grabbed, im_input

    def stop(self):
        return self.release()


class VideoCaptureProcess(Process):
    def __init__(self, capture_input, model_size, flip_mode=1, img_size=False, cap_args=False):
        self.capture_input = capture_input
        self.model_size = model_size
        self.flip_mode = flip_mode
        self.is_file = False
        self.cap_args = cap_args

        if img_size:
            self.img_size = img_size
            print(f"VideoCaptureProcess: Video input ??? -> Model input {self.img_size}")
        else:
            print("VideoCaptureProcess: Figuring out model input size.")
            cap = cv2.VideoCapture(self.capture_input)
            if cap_args:
                for i, j in cap_args: cap.set(i, j)
            _, im_input_org = cap.read()
            im_input = im_crop_and_resize(im_input_org, self.model_size)
            self.img_size = im_input.shape
            cap.release()
            print(f"VideoCaptureProcess: Video input {im_input_org.shape} -> Model input {self.img_size}")

        self.stopped = Value(ctypes.c_bool, False)
        self.new_value = Value(ctypes.c_bool, False)
        self.shared_array = multiprocessing.sharedctypes.RawArray(ctypes.c_ubyte, int(np.prod(self.img_size)))

        super().__init__()

    def read(self):
        return (not self.stopped.value), np.frombuffer(self.shared_array, dtype='uint8').reshape(self.img_size)

    def read_new(self):
        if self.new_value.value and not self.stopped.value:
            self.new_value.value = False
            return True, np.frombuffer(self.shared_array, dtype='uint8').reshape(self.img_size)
        else:
            return False, np.array([])

    def stop(self):
        self.stopped.value = True
        self.join(2)

    def run(self):
        try:
            print("VideoCaptureProcess: Starting the process")

            cap = cv2.VideoCapture(self.capture_input)
            if self.cap_args:
                for i, j in self.cap_args: cap.set(i, j)
            i = 0

            _, shared_array_np = self.read()

            while not self.stopped.value:
                i += 1

                grab_c = 1 # 1 # few grabs makes sure that we are reading the last frame (and not from queue). Needed for slow machines
                for i in range(grab_c): _ = cap.grab()
                grabbed, im_input = cap.read()
                if grabbed:
                    im_input = im_crop_and_resize(im_input, self.model_size)

                    if self.flip_mode:
                        im_input = cv2.flip(im_input, self.flip_mode)

                    np.copyto(shared_array_np, im_input)
                    self.new_value.value = True
                else:
                    self.stopped.value = True
                time.sleep(0.001)

            cap.release()
            # print("VideoCaptureProcess: Ending feed")
        except Exception as e:
            print("VideoCaptureProcess: Exception in video reading / preparing")
            print(e)
            self.stopped.value = True
            raise


