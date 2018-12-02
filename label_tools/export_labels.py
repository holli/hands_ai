from label_utils import *
# import ipdb # ; ipdb.set_trace()

def main(cap, im_scale_fact):
    global im_scale
    im_scale = im_scale_fact

    debug_i = 0
    fps_timer_arr = [0] * 16
    fps = 0
    jpg_count = 0; label_count = 0

    video_label_file = VideoLabelFile(cap.video_fname)
    labels_all = video_label_file.load_previous()
    label_max_idx = max(labels_all.keys())

    export_dir = cap.video_fname.replace('data/videos', 'data/jpg_exp')
    for rep in VideoLabelFile.VIDEO_FNAME_EXTENSIONS:
        export_dir = export_dir.replace(rep, "/")

    print(f"Creating directory {export_dir}")
    os.makedirs(os.path.dirname(export_dir), exist_ok=True)
    
    cv2.namedWindow('display')
    
    # import ipdb; ipdb.set_trace()

    while (True):
        fps_time_begin = time.perf_counter()
        debug_i += 1

        im_input = cap.read()
        fps_time_begin = time.perf_counter()

        if cap.eof or cap.frame_idx() > label_max_idx:
            print(".")
            break
        labels = labels_all[cap.frame_idx()]
        if not labels:
            continue
        
        print(str(len(labels)), end='', flush=True)
        jpg_count += 1
        label_count += len(labels)

        #############################################
        jpg_fname = os.path.join(export_dir, f"{cap.frame_idx()}.jpg")
        labels_fname = os.path.join(export_dir, f"{cap.frame_idx()}.json")

        cv2.imwrite(jpg_fname, im_input, params=[cv2.IMWRITE_JPEG_QUALITY, 95])
        for lab in labels:
            assert (isinstance(lab[0], str) and (len(lab) == 3 or len(lab) == 5)
                    ), f"Problem with label syntax, frame: {cap.frame_idx()}, labels: {labels}"
        labels = [[lab[0]] + [int(l) for l in lab[1:]] for lab in labels]  # so that if there are floats or numpy.int64 left etc
        with open(labels_fname, "w") as write_file:
            json.dump(labels, write_file)
        
        #############################################
        im_display = im_input.copy()
        im_display = cv2.resize(im_display, (int(im_input.shape[1]*im_scale), int(im_input.shape[0]*im_scale)))

        for lab in labels:
            draw_label(im_display, lab, (255,0,0), im_scale)
            # cv2_arrow(im_display, (lab[1], lab[2]),(lab[3], lab[4]), (255,0,0), 2)
            # cv2.putText(im_display, lab[0], (lab[1], lab[2]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        cv2.putText(im_display, f"frame {cap.frame_idx()}, saved_imgs {jpg_count}, fps {fps}", (10,im_display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.imshow('display', im_display)
        
        #############################################
        # time.sleep(0.01)
        k = cv2.waitKey(10)

        fps_timer_arr[debug_i%16] = time.perf_counter() - fps_time_begin
        fps = int(len(fps_timer_arr) * 1 / sum(fps_timer_arr))

    print("Last files:", jpg_fname)
    print("".rjust(len("Last files:"), ' '), labels_fname)
    print()
    print(f"Total files written: {jpg_count}, label_count: {label_count}")
    print()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Python: ', sys.version.split()[0], ', OpenCV2:', cv2.__version__)

    # if sys.argv and sys.argv[-1].endswith(('.avi', 'mpg', 'mp4')):
    video_fname = sys.argv[-1]

    cap = CaptureVideo(video_fname)
    for i in range(120): im_input = cap.read()

    main(cap, im_scale_fact=0.5)