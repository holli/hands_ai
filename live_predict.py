import warnings
import argparse

# sklearn forces some deprecation warnings, we don't want to see them
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=DeprecationWarning)
#     def _warn_ignore(*args, **kwargs):
#         pass
#     _warn_original = warnings.warn
#     warnings.warn = _warn_ignore
#     import sklearn
#     warnings.warn = _warn_original

from hands.basics import *
from hands.utils import *
# from hands.models.model_yolov3_tiny_backbone import *
# from hands.models.model_yolov3_plain_darknet import *
from hands.models.model_darknet_customized import *
import cv2
import collections

from hands.video_prepare_process import *

def cv2_arrow(im, xy_start, xy_end, color, line_width, arrow_length=20):
    # cv2.line(im_display, (lab[1], lab[2]),(lab[3], lab[4]), (255,0,0), 2)
    cv2.line(im, xy_start, xy_end, color, line_width)

    angle = math.atan2(xy_end[1]-xy_start[1], xy_end[0]-xy_start[0]) + math.pi
    arrow_angl = 0.7
    x1 = int(xy_end[0] + arrow_length * math.cos(angle - arrow_angl))
    y1 = int(xy_end[1] + arrow_length * math.sin(angle - arrow_angl))
    x2 = int(xy_end[0] + arrow_length * math.cos(angle + arrow_angl))
    y2 = int(xy_end[1] + arrow_length * math.sin(angle + arrow_angl))
    cv2.line(im, (x1, y1), xy_end, color, line_width)
    cv2.line(im, (x2, y2), xy_end, color, line_width)


def main(cap, model, display_size, display_size_screen=1, display_output=True, cuda=True,
         stabilize_frames_c=0, record_fname='output.avi', hue_lights=False):

    if hue_lights:
        from phue_custom import Bridge
        hue_ip = '192.168.1.102'
        print(f"Using hue bridge at ip {hue_ip}")
        hue_b = Bridge(hue_ip)
        hue_b.connect()
        hue_b_wait = 0
        hue_b_wait_max = 4

    debug_i = 0
    sleep_time = .0
    fps_timer_arr = [0] * 32
    fps = 0
    play = True
    cap_frame = 0
    first_process = True
    stabilize_frames = collections.deque([[] for _ in range(stabilize_frames_c)], stabilize_frames_c)
    if cuda: model = model.cuda()
    cap_is_file = cap.is_file
    if display_output:
        # cv2.namedWindow('display', flags=(cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE))
        cv2.namedWindow('display')

    #class_names = ('unknown', 'open_hand', 'finger_point', 'fist', 'pinch')
    class_names = ('unknown', 'open_hand', 'finger_point', 'fist', 'pinch', 'one', 'two', 'three', 'four', 'thumbs_up', 'thumbs_down', 'finger_gun')
    class_colors = ((0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,0,255))
    text_color = (255,255,255)

    pointing_targets = [[0, -.2, .4, [2]], [.68, .1, .2, [1, 3]], [.95, .56, .2, [0]]] # x, y, r, id
    # for p in pointing_targets: p[0:3] = [int(x) for x in [p[0]*im_display.shape[1], p[1]*im_display.shape[0], p[2]*im_display.shape[0]]]

    record_cap, record_start, record_stop = False, 0, 0

    running = True
    while (running):
        fps_time_begin = time.perf_counter()
        debug_i += 1

        #############################################
        ## Keyboard - ui handling

        k = cv2.waitKey(5)
        if k == 27:  # esc
            break
        elif k == ord(' '):
            play = not play
        elif k == ord('q'):
            if cap.is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)-(32*2))
        elif k == ord('w'):
            if cap.is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+(32*2))
        elif k == ord('z'):
            sleep_time = max(sleep_time-0.05, .0)
        elif k == ord('x'):
            sleep_time += 0.05
        elif k == ord('s'):
            stabilize_frames_c = (stabilize_frames_c + 1)%5
            stabilize_frames = collections.deque([[] for _ in range(stabilize_frames_c)], stabilize_frames_c)
            print(f"Stabilizing with frames: {stabilize_frames_c}")
        elif k == ord('r'):
            if record_cap:
                record_stop = 30
            else:
                record_start = 20
                record_fname = 'output.avi'
                print(f"Creating a video output {record_fname}: {im_display.shape[:2]}")
                # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
                # record_cap_fourcc = cv2.VideoWriter_fourcc(*'XVID')
                record_cap_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                record_cap = cv2.VideoWriter(record_fname, record_cap_fourcc, 20.0, (im_display.shape[1], im_display.shape[0]), 1)
        elif k == ord('c'):
            import ipdb; ipdb.set_trace()

        ########################################
        # Getting new frame
        if play:
            if isinstance(cap, VideoCaptureProcess):
                _, im_input = cap.read_new()
                if not _:
                    if cap.stopped.value:
                        running = False
                        break
                    else:
                        time.sleep(.02)
                        continue
            else:
                _, im_input = cap.read()
                if not _:
                    running = False
                    break

            if cap_is_file:
                cap_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                cap_frame += 1

        im_tensor = im2tensor(im_input)
        if cuda: im_tensor = im_tensor.cuda()
        results = predict_img(model, im_tensor, conf_thresh=.3)[0]

        ########################################
        # Stabilizing frames
        if stabilize_frames_c > 0:
            results_org = results
            for results_frame in stabilize_frames:
                _, results = results, []
                for r in _:
                    for rp in results_frame:
                        if (r[0] == rp[0]) and (abs(r[1]-rp[1]) < 0.05) and (abs(r[2]-rp[2]) < 0.05):
                            results.append(r)
                            break
            stabilize_frames.append(results_org)

        ########################################
        # Output displaying
        info_str = ""
        if cap_is_file:
            info_str = f"f{int(cap_frame)}. "
        #info_str += f"Fps {fps}. "
        if sleep_time: info_str += "Sleep_time {:.2}. ".format(sleep_time)
        if record_cap:
            if record_stop > 0: info_str += f"Ending recording in {record_stop}. "
            elif record_start > 0: info_str += f"Starting record in {record_start}. "
            else: info_str += f"Recording to {record_fname}. "

        command_str = ""
        if display_output:
            # im_display = cv2.resize(im_input, (display_size, display_size))
            im_display = cv2.resize(im_input, (int(im_input.shape[1]*display_size), int(im_input.shape[0]*display_size)))
            #im_display = cv2.resize(im_input, (display_size, display_size))

            if first_process:
                for p in pointing_targets:
                    p[0:3] = [int(x) for x in [p[0]*im_display.shape[1], p[1]*im_display.shape[0], p[2]*im_display.shape[0]]]
            for p in pointing_targets:
                cv2.circle(im_display, tuple(p[:2]), p[2], (100,100,100), thickness=5, )

            for label_i, x, y, dx, dy, obj_p, label_p in results:
                color = class_colors[label_i%len(class_colors)]
                xy = (int(round(x*im_display.shape[1])), int(round(y*im_display.shape[0])))
                dxy_size = 10000 # 100 # 25
                dxy = tuple([int(a+b*dxy_size) for a, b in zip(xy, (dx, dy))])
                cv2.circle(im_display, xy, 5, color, thickness=2)

                label_s = class_names[label_i]
                cv2.putText(im_display, label_s, (xy[0]-10, xy[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                label_s_info = "{:.2f} > {:.2f}".format(obj_p, label_p)
                cv2.putText(im_display, label_s_info, (xy[0]-10, xy[1]+30+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

                if label_s in ('one', 'two', 'three', 'four'):
                    command_str += f"Setting mode {label_s}"
                    if hue_lights:
                        if label_s == 'one':
                            hue_b.activate_scene('1',  '2Ag8NvbZKbfbSnb') # Demo1
                        if label_s == 'two':
                            hue_b.activate_scene('1',  'GtGEYxbOCAsdnEr') # Arctic
                        if label_s == 'three':
                            hue_b.activate_scene('1',  'vD9ehQd-CxgWN2V') # Bright
                        hue_b_wait = hue_b_wait_max

                if label_s == 'thumbs_up' and record_cap and record_stop >= 0:
                    command_str += f"Thumbs up: Stopping recording. "
                    if record_stop == 0: record_stop = 20

                if label_s in ('open_hand', 'finger_point', 'pinch', 'finger_gun'):
                    cv2_arrow(im_display, xy, dxy, color, 1)

                    for p in pointing_targets:
                        dist = np.linalg.norm(np.cross(np.array(xy)-np.array(dxy), np.array(xy)-p[:2]))/np.linalg.norm(np.array(xy)-np.array(dxy))
                        hit = dist < p[2]
                        # point_info = "HIT" if hit else "MIS"
                        # point_info += f" ({int(dist)})"
                        # cv2.putText(im_display, point_info, (xy[0]-10, xy[1]+30+20+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                        if hit:
                            cv2.circle(im_display, tuple(p[:2]), p[2], (255,255,255), thickness=5, )
                            if hue_lights:
                                if label_s == 'finger_point':
                                    command_str += f"Off {p[-1]}. "
                                    for idx in p[-1]: b.get_light_objects()[idx].on = False
                                    hue_b_wait = hue_b_wait_max
                                elif label_s == 'finger_gun':
                                    command_str += f"On {p[-1]}. "
                                    for idx in p[-1]: b.get_light_objects()[idx].on = True
                                    hue_b_wait = hue_b_wait_max


            #############################################
            cv2.putText(im_display, command_str, (10,im_display.shape[0]-10-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(im_display, info_str, (10,im_display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            if record_cap:
                if record_start > 0: record_start -= 1
                if record_start == 0:
                    record_cap.write(im_display)
                if record_stop > 0:
                    if record_stop == 1:
                        record_cap.release()
                        record_cap = None
                    record_stop -= 1


            im_display_screen = cv2.resize(im_display,
                                    (int(im_display.shape[1]*display_size_screen), int(im_display.shape[0]*display_size_screen)))
            cv2.imshow('display', im_display_screen)

        else:
            for label_i, x, y, dx, dy, obj_p, label_p in results:
                info_str += " - {} ({:.2f} > {:.2f})".format(class_names[label_i], obj_p, label_p)
            print(info_str)

        #############################################
        # Statistics
        if sleep_time > 0:
            time.sleep(sleep_time)

        fps_timer_arr[debug_i%16] = time.perf_counter() - fps_time_begin
        fps = int(len(fps_timer_arr) * 1 / sum(fps_timer_arr))
        first_process = False
        if hue_lights: b_wait = max(0, b_wait-1)

    if record_cap: record_cap.release()
    # cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Python:', sys.version.split()[0], ', OpenCV2:', cv2.__version__, ', Torch:', torch.__version__)

    # -m 'ModelYoloV3TinyBackbone.load_default_320()'
    # -m 'ModelYoloV3TinyBackbone.load_default_224()'
    # -m 'ModelYoloV3PlainDarknet.load_default_320()'

    # --cap_args 3 640 4 480 --img_size 416 544
    # --cap_args 3 1280 4 720 --img_size 416 736
    # --cap_args 3 640 4 480 --img_size 320 416

    default_model = "ModelDarknetCustomized.load_default_03_320()"

    parser = argparse.ArgumentParser(description='Live prediction with hands models.')
    parser.add_argument('-i', '--input', default='cv2video0', help='cv2video0/9, gstreamer, file')#, default="cv2.VideoCapture(0)")
    parser.add_argument('-m', '--model', help=f"Python command returning model, def:'{default_model}'", default=default_model)
    parser.add_argument('-f', '--input-file', help='path to the file')
    parser.add_argument('--stabilize-frames', type=int, default=0, help="Only show predicts that are seen in this many previous frames")
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    #parser.add_argument('--display-size', type=int, default=448, help="window output size, def 448")
    parser.add_argument('--display-size', type=float, default=1.0, help="Video/Window output size compared to input size, def 1")
    parser.add_argument('--display-size-screen', type=float, default=1.0, help="Screen size compared to video, def 1")
    parser.add_argument('--record_fname', type=str, default=False, help="Filename for recording (output.avi)")
    parser.add_argument('--img_size', type=int, default=False, nargs=2, help="If automatic figuring doesn't work, list them here. E.g. --img_size 320 544")
    parser.add_argument('--cap_args', type=int, default=False, nargs='*', help="Opencv cap set infos. E.g. '3 1280 4 720' or '3 640 4 480'")
    args = parser.parse_args()

    # import pdb; pdb.set_trace()
    # import ipdb; ipdb.set_trace()

    model_exec = 'model = ' + args.model
    exec(model_exec)
    if not args.no_cuda: model.cuda()

    if args.input_file:
        video_fname = args.input_file
        # cap = cv2.VideoCapture(video_fname)
        cap = VideoCaptureFile(video_fname, model.default_size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

    elif args.input == 'gstreamer':
        gs = [l for l in cv2.getBuildInformation().splitlines() if re.findall(r'gstream', l, re.IGNORECASE)]
        if not gs or gs[0].endswith('NO'):
            warnings.warn("NO GSTREAM SUPPORT? If using stream options make sure that cv2 is compiled with gstream support, (see cv2.getBuildInformation()")

        width=640; height=480
        #width=960; height=720
        s = f"nvcamerasrc queue-size=6 sensor-id=0 fpsRange='60 60' ! video/x-raw(memory:NVMM), width={width}, height={height}, format=I420, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

        # s = "gst-launch-1.0 udpsrc port=5000 !  application/x-rtp, encoding-name=JPEG,payload=26 !  rtpjpegdepay !  jpegdec ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        # s = "videotestsrc ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

        # cap = cv2.VideoCapture(s)
        cap = VideoCaptureProcess(s, model.default_size)
        cap.start()
    else:
        cv_input_int = int(re.search('[0-9]$', args.input)[0])
        # cap = cv2.VideoCapture(cv_input_int)

        img_size = args.img_size
        if img_size: img_size = tuple(img_size + [3])
        cap_args = args.cap_args
        if cap_args: cap_args = [[args.cap_args[i*2], args.cap_args[i*2+1]] for i in range(len(args.cap_args)//2)]
        cap = VideoCaptureProcess(cv_input_int, model.default_size, img_size=img_size, cap_args=cap_args)
        cap.start()

    #video_out = args.video_out

    main(cap, model, display_size=args.display_size, display_size_screen=args.display_size_screen,
            display_output=(not args.no_display),
            cuda=(not args.no_cuda), stabilize_frames_c=args.stabilize_frames,
            record_fname=args.record_fname,
            )

    cap.stop()







