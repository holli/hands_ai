# import ipdb
from label_utils import *
# from config import *
# import ipdb # ; ipdb.set_trace()

# Python path fixing so we can import libraries
import sys
import os
sys_paths = ['../../Chainer_Realtime_Multi-Person_Pose_Estimation/',
            ]
for p in sys_paths:
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.append(p)


import cv2
import argparse
import chainer
from entity import params
from pose_detector import PoseDetector, draw_person_pose
# from face_detector import FaceDetector, draw_face_keypoints
from hand_detector import HandDetector, draw_hand_keypoints


def main(cap, im_scale=2, view_results=False):
    debug_i = 0
    fps_timer_arr = [0] * 16
    fps = 0

    # load model
    pose_device = 0
    pose_model_dir = '../../Chainer_Realtime_Multi-Person_Pose_Estimation/models'
    pose_detector = PoseDetector("posenet", f"{pose_model_dir}/coco_posenet.npz", device=pose_device)
    hand_detector = HandDetector("handnet", f"{pose_model_dir}/handnet.npz", device=pose_device)

    # cv2.namedWindow('display', flags=(cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE))
    if view_results: cv2.namedWindow('display')

    video_label_file = VideoLabelFile(cap.video_fname, fname_add='pre_points_pose')
    labels_current = defaultdict(lambda: [])
    labels_all_previous = video_label_file.load_previous()

    im_input = cap.read()
    im_input_shape = im_input.shape[0:2]

    first_run = True

    while (not cap.eof):
        fps_time_begin = time.perf_counter()
        debug_i += 1

        im_input = cap.read()
        current_frame_id = cap.frame_idx()
        # print(cap.info())

        im_pose = cv2.resize(im_input, (round(im_input_shape[1]/im_scale), round(im_input_shape[0]/im_scale)))
        if first_run:
            print(f"Video size {im_input.shape} -> Model input size {im_pose.shape}")
            first_run = False

        ##########################################
        person_pose_array, _ = pose_detector(im_pose)
        im_display = cv2.addWeighted(im_pose, 0.6, draw_person_pose(im_pose, person_pose_array), 0.4, 0)

        for person_pose in person_pose_array:
            unit_length = pose_detector.get_unit_length(person_pose)

            # arr = np.array([a for a in person_pose if a is not None])
            # if arr.any():
            #     arr[:, 0:2] *= im_scale
            #     labels_current[current_frame_id].append(['pre_person_pose', arr.tolist()])

            # hands estimation
            hands = pose_detector.crop_hands(im_pose, person_pose, unit_length)
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                im_display = draw_hand_keypoints(im_display, hand_keypoints, (bbox[0], bbox[1]))
                cv2.rectangle(im_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

                if hand_keypoints[5] and hand_keypoints[8]:
                    f_points = np.array([hand_keypoints[5][:2], hand_keypoints[8][:2]])
                    f_points = (f_points + np.array([bbox[0], bbox[1]]))*im_scale
                    #f_points = tuple(map(tuple, f_points.astype(int)))
                    f_points = f_points.astype(int).tolist()
                    labels_current[current_frame_id].append(f_points)

            if hands["right"] is not None:
                hand_img = hands["right"]["img"]
                bbox = hands["right"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="right")
                im_display = draw_hand_keypoints(im_display, hand_keypoints, (bbox[0], bbox[1]))
                cv2.rectangle(im_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

                if hand_keypoints[5] and hand_keypoints[8]:
                    f_points = np.array([hand_keypoints[5][:2], hand_keypoints[8][:2]])
                    f_points = (f_points + np.array([bbox[0], bbox[1]]))*im_scale
                    #f_points = tuple(map(tuple, f_points.astype(int)))
                    f_points = f_points.astype(int).tolist()
                    labels_current[current_frame_id].append(f_points)



        #############################################
        for l in labels_current[current_frame_id]:
            cv2.circle(im_display, (round(l[0][0]/im_scale), round(l[0][1]/im_scale)), 10, (255, 0, 0), 2)
            cv2.circle(im_display, (round(l[1][0]/im_scale), round(l[1][1]/im_scale)), 10, (0, 255, 0), 2)

        cv2.putText(im_display, f"frame {int(current_frame_id)}, fps: {int(fps)}.", (10,im_display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


        if view_results:
            #cv2.imshow('display', im_display)
            cv2.imshow('display', im_pose)
        else:
            print(".", end="")
            sys.stdout.flush()

        # labels_current[current_frame_id].append

        #############################################
        ## KEYBOARD

        k = cv2.waitKey(5)
        if k == 27:  # esc
            break
        elif k == ord('c'):
            import ipdb; ipdb.set_trace()
            # ipdb.set_trace()
            # pdb.set_trace()

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
    for i in range(1): im_input = cap.read()

    #main(cap, im_scale=2, view_results=False)
    main(cap, im_scale=args.scale, view_results=True)

