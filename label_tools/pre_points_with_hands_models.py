# import ipdb
from label_utils import *
# from config import *
# import ipdb # ; ipdb.set_trace()

# Python path fixing so we can import libraries
import sys
import os
sys_paths = ['../../hands/custom_01/',
            ]
for p in sys_paths:
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.append(p)


import cv2
import argparse

from hands.basics import *
from hands.utils import *
from hands.models.model_yolov3_tiny_backbone import *
from hands.models.model_yolov3_plain_darknet import *
from video_prepare_process import *

def main(cap, im_scale=2, view_results=False):
    debug_i = 0
    fps_timer_arr = [0] * 16
    fps = 0
    first_run = True

    video_label_file = VideoLabelFile(cap.video_fname, fname_add='pre_points_hands')
    labels_current = defaultdict(lambda: [])
    # labels_all_previous = video_label_file.load_previous()

    if view_results: cv2.namedWindow('display')

    # load model
    model = ModelYoloV3PlainDarknet(num_classes=12).cuda().eval()
    model.load_state_dict(torch.load('/home/ohu/koodi/kesken/hands/custom_01/data/models/18/7.pth'))

    while (not cap.eof):
        fps_time_begin = time.perf_counter()
        debug_i += 1

        im_input = cap.read()
        current_frame_id = cap.frame_idx()
        # print(cap.info())

        im_sized, im_sized_padding = im_crop_and_resize(im_input, 416, return_crop_pad_size=True)
        #im = cv2.resize(im_input, (round(im_input_shape[1]/im_scale), round(im_input_shape[0]/im_scale)))

        if first_run:
            print(f"Video size {im_input.shape} -> Model input size {im_sized.shape} (padding: {im_sized_padding})")
            first_run = False

        im_tensor = im2tensor(im_sized).cuda()
        results = predict_img(model, im_tensor)
        for label_i, x, y, dx, dy, obj_p, label_p in results[0]:
            xy = (int(round(x*im_input.shape[1])+im_sized_padding[1]), int(round(y*im_input.shape[0])+im_sized_padding[1]))
            dxy_size = 27
            dxy = tuple([int(a+b*dxy_size) for a, b in zip(xy, (dx, dy))])
            #f_points = list(xy) + list(dxy)
            # labels_current[current_frame_id].append([list(xy) + list(dxy)])
            labels_current[current_frame_id].append([list(xy)] + [list(dxy)])


        im_display = cv2.resize(im_input, (round(im_input.shape[1]/im_scale), round(im_input.shape[0]/im_scale)))
        for l in labels_current[current_frame_id]:
            cv2.circle(im_display, (round(l[0][0]/im_scale), round(l[0][1]/im_scale)), 10, (255, 0, 0), 2)
            cv2.circle(im_display, (round(l[1][0]/im_scale), round(l[1][1]/im_scale)), 10, (0, 255, 0), 2)

        cv2.putText(im_display, f"frame {int(current_frame_id)}, fps: {int(fps)}.", (10,im_display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if view_results:
            #cv2.imshow('display', im_display)
            cv2.imshow('display', im_display)
        else:
            print(".", end="")
            sys.stdout.flush()

        k = cv2.waitKey(5)
        if k == 27:  # esc
            break
        elif k == ord('c'):
            import ipdb; ipdb.set_trace()

        fps_timer_arr[debug_i%16] = time.perf_counter() - fps_time_begin
        fps = int(len(fps_timer_arr) * 1 / sum(fps_timer_arr))

    print(". ")
    # cap.release()
    video_label_file.save_current_labels(labels_current, append_previous=False, custom_lists=True)

    if view_results: cv2.destroyAllWindows()




if __name__ == "__main__":
    print('Python: ', sys.version.split()[0], ', OpenCV2:', cv2.__version__)

    parser = argparse.ArgumentParser(description='Pose detector hand points')
    parser.add_argument('video_fname', help='')
    parser.add_argument('--scale', '-s', default=1, type=float, help='what scaling used before feedint the model')
    args = parser.parse_args()

    # if sys.argv and sys.argv[-1].endswith(('.avi', 'mpg', 'mp4')):
    #video_fname = sys.argv[-1]
    # else:
        #video_fname = "/home/ohu/koodi/kesken/hands_data/data/videos/3/2018-09-17_31_open_hand.mp4"

    cap = CaptureVideo(args.video_fname)
    # for i in range(1): im_input = cap.read()

    main(cap, im_scale=args.scale, view_results=True)

