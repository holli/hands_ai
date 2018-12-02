# import ipdb
from label_utils import *
# from config import *
# import ipdb # ; ipdb.set_trace()



def mouse_callback(event, x, y, flags, param):
    global l_xy, l_xy_pointer, l_ready, c_play, im_scale
    x, y = int(x/im_scale), int(y/im_scale)
    # if event == cv2.EVENT_MOUSEMOVE:
    #     #cv2.circle(img,(x,y),100,(255,0,0),-1)
    if event == cv2.EVENT_LBUTTONDOWN:
        if c_play:
            c_play = False
        else:
            l_xy[l_xy_pointer] = [x, y]
            if l_xy_pointer == len(l_xy):
                l_ready = True
            l_xy_pointer = (l_xy_pointer + 1) % len(l_xy)

def scale_p(x, y):
    global im_scale
    return int(x*im_scale), int(y*im_scale)

def reset_l_xy(size=None):
    global l_xy, l_xy_pointer
    if size == None:
        size = len(l_xy)
    l_xy = [None for i in range(size)]
    l_xy_pointer = 0

def draw_label_pred(im, im_scale, l, i):
    cv2.circle(im, scale_p(*l[0]), 10, (255, 0, 0), 1)
    if len(l_xy) > 1:
        cv2.circle(im, scale_p(*l[1]), 10, (0, 255, 0), 1)
    x = (l[0][0]+l[0][0])//2 # l[1][0]-5
    y = min(l[0][1], l[1][1])-20/im_scale  # l[1][1]-30/im_scale
    cv2.putText(im, f"f{i+1}", scale_p(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)


def change_classes(labels_all_previous, start_idx, end_idx, new_label):
    keys_change = np.array(list(labels_all_previous.keys()))
    keys_change = keys_change[(keys_change >= start_idx) & (keys_change <= end_idx)]
    for k in keys_change:
        if labels_all_previous[k] and labels_all_previous[k][0]:
            labels_all_previous[k][0][0] = new_label

# def del_labels(start_idx, end_idx):
#     keys_change = np.array(list(labels_all_previous.keys()))
#     keys_change = keys_change[keys_change > current_frame_id]
#     for k in keys_change: labels_all_previous[k][0][0] = 'finger_gun'



def main(cap, im_scale_fact=1):
    debug_i = 0
    fps_timer_arr = [0] * 16
    fps = 0

    global l_xy, l_xy_pointer, l_ready, c_play, im_scale
    l_ready = False
    reset_l_xy(2)
    # l_xy = [None, None, None, None]
    im_scale = im_scale_fact

    # cv2.namedWindow('display', flags=(cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE))
    cv2.namedWindow('display')
    cv2.setMouseCallback('display', mouse_callback)

    c_read_next = c_play = True
    l_class = '0_unkown'

    video_label_file = VideoLabelFile(cap.video_fname)
    labels_current = defaultdict(lambda: [])
    labels_all_previous = video_label_file.load_previous()

    labels_pred_show = True
    labels_pred_pose = VideoLabelFile(cap.video_fname, fname_add='pre_points_pose').load_previous()
    labels_pred_hands = VideoLabelFile(cap.video_fname, fname_add='pre_points_hands').load_previous()
    labels_pred_f1fx_keys = [190, 191, 192, 193, 194, 195, 196, 197] # f1, f2, f3, f4...

    im_input = cap.read()
    #im_input = cv2.resize(im_input, (int(im_input.shape[0]*im_scale), int(im_input.shape[1]*im_scale)))
    im_input_gray = cv2.cvtColor(im_input, cv2.COLOR_BGR2GRAY)
    tracking_flow_p0 = np.array([])

    while (True):
        fps_time_begin = time.perf_counter()
        debug_i += 1

        if c_read_next or c_play:
            c_read_next = False

            im_input = cap.read()
            #im_input = cv2.resize(im_input, (int(im_input.shape[0]*im_scale), int(im_input.shape[1]*im_scale)))
            #print(int(im_input.shape[0]*im_scale), int(im_input.shape[1]*im_scale))
            # print(cap.info())

            im_input_gray_old = im_input_gray
            im_input_gray = cv2.cvtColor(im_input, cv2.COLOR_BGR2GRAY)
            if len(tracking_flow_p0) > 0:
                tracking_lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                tracking_flow_p1, tracking_st, tracking_err = cv2.calcOpticalFlowPyrLK(im_input_gray_old, im_input_gray, tracking_flow_p0, None, **tracking_lk_params)
                tracking_flow_p0 = np.array([])
                if tracking_st.all():
                    l_xy = tracking_flow_p1.astype('int')
                #     track_xy = tracking_flow_p1.astype('int')
                # else:
                #     track_xy = None
                    # track_xy = [tracking_flow_p1[0, 0, 0], tracking_flow_p1[0, 0, 1], tracking_flow_p1[1, 0, 0], tracking_flow_p1[1, 0, 1]]
                    # track_xy = [int(round(v)) for v in track_xy]
                # for p in tracking_flow_p1: cv2.circle(im_display, (p[0,0], p[0,1]), 10, (0,255,0), 1)

        current_frame_id = cap.frame_idx()
        im_input = cap.current()

        im_display = im_input.copy()
        im_display = cv2.resize(im_display, (int(im_input.shape[1]*im_scale), int(im_input.shape[0]*im_scale)))

        #############################################

        for lab in labels_all_previous[current_frame_id]:
            draw_label(im_display, lab, (255,0,0), scale=im_scale)
        lab = labels_current[current_frame_id]
        if lab:
            draw_label(im_display, lab, (200,0,200), scale=im_scale)
        l_color = (255,0,255)
        if l_xy[0] is not None:
            cv2.circle(im_display, scale_p(*l_xy[0]), 5, l_color, 1)
            y_text = max(l_xy[0][1], l_xy[1][1]) if len(l_xy) > 1 and l_xy[1] is not None else l_xy[0][1]
            y_text += 30/im_scale
            cv2.putText(im_display, l_class, scale_p(l_xy[0][0], y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_color, 1)
        if len(l_xy) > 1 and l_xy[1] is not None:
            cv2_arrow(im_display, scale_p(*l_xy[0]), scale_p(*l_xy[1]), l_color, 1)

        if labels_pred_show:
            if labels_pred_pose[current_frame_id]:
                #for i, l in enumerate(labels_pred[current_frame_id][0]): # bug in old preprocessing
                for i, l in enumerate(labels_pred_pose[current_frame_id]):
                    draw_label_pred(im_display, im_scale, l, i)
            if labels_pred_hands[current_frame_id]:
                for i, l in enumerate(labels_pred_hands[current_frame_id]):
                    draw_label_pred(im_display, im_scale, l, i+4)
                    # cv2.circle(im_display, scale_p(*l[0]), 10, (255, 0, 0), 1)
                    # cv2.circle(im_display, scale_p(*l[1]), 10, (0, 255, 0), 1)
                    # cv2.putText(im_display, f"f{i+5}", scale_p(l[1][0]-5, l[1][1]-30/im_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        #############################################
        #cv2.putText(im_display, f"frame {int(current_frame_id)}, fps {int(fps)}", (10,im_input.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(im_display, f"frame {int(current_frame_id)}, l_class: {l_class}.", (10,im_display.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(im_display, f"l_class: {l_class}.", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        # cv2.putText(im_display_flipped, f"mouse {mouse_position}", (10,im_input.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('display', im_display)


        #############################################
        ## CAPTURING INFORMATION

        if l_ready:
            if all(get_flat_list(l_xy)):
                tracking_flow_p0 = np.array(l_xy).astype('float32')
                labels_current[current_frame_id] = [l_class] + [int(i) for i in get_flat_list(l_xy)]
                reset_l_xy()
                c_read_next = True
            l_ready = False


        #############################################
        ## KEYBOARD

        k = cv2.waitKey(10)
        if k == 27:  # esc
            # video_label_file.save_current_labels(labels_current)
            # labels_all_previous = video_label_file.load_previous()
            # labels_current = defaultdict(lambda: [])
            break
        elif k == ord(' '):
            if c_play:
                c_play = False
            else:
                l_ready = True
                # c_read_next = True
        elif k == ord('q'):
            c_play = False
            tracking_flow_p0 = np.array([])
            cap.previous()
            reset_l_xy()
        elif k == ord('w'):
            c_play = False
            c_read_next = True
            reset_l_xy()
        elif k == ord('c'):
            import ipdb; ipdb.set_trace()
            # ipdb.set_trace()
            # pdb.set_trace()
        elif k == ord('e'):
            c_play = not c_play
            reset_l_xy()
        elif k == ord('r'):
            cap.restart()
            reset_l_xy()
        elif k == ord('s'):
            video_label_file.save_current_labels(labels_current, labels_all_previous=labels_all_previous)
            labels_all_previous = video_label_file.load_previous()
            labels_current = defaultdict(lambda: [])
        elif k == ord('d'):
            del(labels_current[current_frame_id])
            del(labels_all_previous[current_frame_id])
            c_read_next = True
            reset_l_xy()
        elif k == ord('1'):
            l_class = 'open_hand'
            reset_l_xy(2)
        elif k == ord('2'):
            l_class = 'fist'
            reset_l_xy(1)
        elif k == ord('3'):
            l_class = 'finger_point'
            reset_l_xy(2)
        elif k == ord('4'):
            l_class = 'pinch'
            reset_l_xy(2)
        elif k == ord('5'):
            l_class = 'one'
            reset_l_xy(1)
        elif k == ord('6'):
            l_class = 'two'
            reset_l_xy(1)
        elif k == ord('7'):
            l_class = 'three'
            reset_l_xy(1)
        elif k == ord('8'):
            l_class = 'four'
            reset_l_xy(1)
        elif k == ord('9'):
            l_class = 'thumbs_up'
            reset_l_xy(1)
        elif k == ord('0'):
            l_class = 'thumbs_down'
            reset_l_xy(1)
        elif k == ord('p'):
            l_class = 'finger_gun'
            reset_l_xy(2)
        elif k == ord('t'):
            labels_pred_show = not labels_pred_show
            print(f"Showing previos predictions: {labels_pred_show}")
        elif k in labels_pred_f1fx_keys:
            labels_pred_idx = labels_pred_f1fx_keys.index(k)
            if labels_pred_idx < 4:
                labels_pred_copy = labels_pred_pose
            else:
                labels_pred_copy = labels_pred_hands
                labels_pred_idx -= 4

            if labels_pred_copy[current_frame_id] and len(labels_pred_copy[current_frame_id]) > labels_pred_idx:
                l_xy = np.array(labels_pred_copy[current_frame_id][labels_pred_idx])[:len(l_xy)].astype('int')
                #print(f"Copying label {labels_pre_idx} : f{l_xy}")


        fps_timer_arr[debug_i%16] = time.perf_counter() - fps_time_begin
        fps = int(len(fps_timer_arr) * 1 / sum(fps_timer_arr))

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Python: ', sys.version.split()[0], ', OpenCV2:', cv2.__version__)

    # if sys.argv and sys.argv[-1].endswith(('.avi', 'mpg', 'mp4')):
    video_fname = sys.argv[-1]
    # else:
        #video_fname = "/home/ohu/koodi/kesken/hands_data/data/videos/3/2018-09-17_31_open_hand.mp4"

    cap = CaptureVideo(video_fname)
    # for i in range(1): im_input = cap.read()

    main(cap, im_scale_fact=1.3)