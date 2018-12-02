
"""
use instead of "ret, im_input_fullsize = cap.read()"

from capture_video_threaded import *
fvs = CaptureVideoThreaded()
fvs.start()
time.sleep(1)

im_input_fullsize = fvs.read()

fvs.stop()
fvs.join(2)
"""

# https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
# https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
# https://docs.python.org/3/library/multiprocessing.html


from multiprocessing import Process, Queue, Value
import ctypes
import sys
import cv2
import time

class CaptureVideoThreaded(Process):
# class CaptureVideoThreaded():
    def __init__(self, queue_size=20):
        self.stopped = Value(ctypes.c_bool, False)
        self.Q = Queue(maxsize=queue_size)
        super(CaptureVideoThreaded, self).__init__()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped.value = True

    def run(self):
        cap = cv2.VideoCapture(sys.argv[-1]) # have to open stream only here so multithreading works, otherwise it would halt in read

        i = 0
        while not self.stopped.value:
            i += 1

            # if self.Q.full():
            #     print("CAP_THREAD {}: full {}".format(i, self.Q.qsize()), flush=True)
            # else:
            #     print("CAP_THREAD {}: not_full {}".format(i, self.Q.qsize()), flush=True)

            if not self.Q.full():
                # read the next frame from the file
                (grabbed, im_input) = cap.read()

                # fgmask_mog2 = fgbg_mog2.apply(im_input)  # should this be im_input_gray
                # im_moved = cv2.bitwise_and(im_input, im_input, mask=fgmask_mog2)

                # if the `grabbed` boolean is `False`, then we have reached the end of the video file
                if not grabbed:
                    print("Stopping CaptureVideoThread because not grabbed")
                    self.stop()
                else:
                    self.Q.put(im_input)
            else:
                time.sleep(0.001)

        cap.release()
        self.Q.close()
        self.Q.cancel_join_thread()


