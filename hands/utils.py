from hands.basics import *
from hands.data import *
#from scipy.spatial import KDTree
import scipy.spatial
# import sklearn
import sklearn.preprocessing
import fastprogress
import cv2

###################################
# CV2 Image related functions
def im2tensor(im):
    im = im[...,::-1]  # bgr to rgb
    im = np.transpose(im, (2,0,1)) # y,x,c => c,y,x
    t = torch.tensor(np.ascontiguousarray(im))
    t = t.float().div_(255)
    return t

def im_crop_32_center(im, square=False):
    shape = im.shape[:2]
    y, x = shape
    if square:
        size = min(shape)
        dy = (y-size)//2
        dx = (x-size)//2
        return im[dy:dy+size, dx:dx+size]
    else:
        dy = (y%32)/2
        dx = (x%32)/2
        return im[math.floor(dy):y-math.ceil(dy), math.floor(dx):x-math.ceil(dx)]

def im_crop_and_resize(im, size_or_sizes, return_resize_ratio=False):
    square = isinstance(size_or_sizes, tuple)
    size = size_or_sizes if square else (size_or_sizes, size_or_sizes)

    resize_ratio = (size[0]/im.shape[0], size[1]/im.shape[1])
    resize_ratio = max(resize_ratio)
    assert resize_ratio <= 1, f"Video input should be at least as big as network input. video:{im.shape} >= model:{size}."
    # assert min(im_input.shape[:-1]) >= size, f"Video input should be at least as big as network input. tensor:{im_input.shape} >= input:{self.size}."

    im_resized = cv2.resize(im, (round(im.shape[1]*resize_ratio), round(im.shape[0]*resize_ratio)))
    im = im_crop_32_center(im_resized, square=square)

    # if return_crop_pad_size:
    #     return im, np.array([(im_resized.shape[0] - im.shape[0])//2, (im_resized.shape[1] - im.shape[1])//2])
    if return_resize_ratio:
        return im, resize_ratio, np.array([(im_resized.shape[0] - im.shape[0])/2, (im_resized.shape[1] - im.shape[1])/2])
    else:
        return im

###################################
# Predicting related functions

#def predict_img(model, imgs, get_results_args={}):
def predict_img(model, imgs, **get_results_args):
    # imgs = imgs.cuda()
    if len(imgs.shape) == 3: imgs = imgs.unsqueeze(0)

    with torch.no_grad():
        outputs = model(imgs)

    results = get_results(outputs, **get_results_args)
    return results


# return [[label_i, x, y, dx, dy, obj_p, label_p],[...],...]
def get_results(outputs, conf_thresh=.4, nms=.05):
    results_all = [[] for _ in range(len(outputs))]

    outputs_sig = outputs[:, :5].sigmoid()
    obj_p_confs = outputs_sig[:, 4, :, :]
    idxs = (obj_p_confs > conf_thresh).nonzero()
    #idxs = idxs.cpu().numpy()
    idxs = idxs.cpu().tolist()

    # sort by obj_p confidence
    idxs.sort(key=lambda x: -obj_p_confs[tuple(x)])

    for b, i, j in idxs:
        obj_p = outputs_sig[b,4,i,j].item()
        labels = outputs[b,5:,i,j].softmax(0)
        label_i = int(labels.argmax(0))
        label_p = labels[label_i].item()

        xy_cent = outputs_sig[b, 0:2, i, j].cpu().detach().numpy()
        xy_cent = (xy_cent + (i, j)) / outputs.shape[2:] # * grid_sz
        xy_cent = xy_cent.T[[1, 0]]

        too_close = False
        for _, res_x, res_y, *res in results_all[b]:
            if math.hypot(xy_cent[0]-res_x, xy_cent[1]-res_y) < nms:
                too_close = True
                break
        if too_close: continue

        xy_angle = outputs[b, 2:4, i, j].tanh().cpu().detach().numpy()
        xy_angle = xy_angle.T[[1, 0]]
        xy_angle = sklearn.preprocessing.normalize(xy_angle[None])[0]
        #angle = math.atan2(*xy_angle.T[[1, 0]])
        #angle = math.degrees(angle)
        #results_all[b].append([label_i, *xy_cent.T[[1,0]], d_angle, *xy_angle.T[[1, 0]], obj_p, label_p])
        results_all[b].append([label_i, *xy_cent, *xy_angle, obj_p, label_p])

    return results_all

###################################
# Accuracy related functions

def compare_single_result_target(results, targets, distance_ok=.05, angle_ok=10):
    len_targ = len((targets[:, 0] > 0).nonzero())
    if len(results) > len_targ: return 'too_many_preds'
    elif len(results) < len_targ: return 'not_enough_preds'

    # preds = preds.copy()
    # preds.sort(key=lambda x: -x[5])

    # from yx to xy
    targets = targets.cpu()
    targets = torch.index_select(targets, 1, torch.LongTensor([0, 2, 1, 4, 3]))
    # targets = list(targets[:len_targ])
    idxs = list(range(0, len_targ))

    for p in results:
        idx_t = torch.LongTensor(idxs)
        kdtree = scipy.spatial.KDTree(targets.index_select(0, idx_t)[:, 1:3])
        dist, idx = kdtree.query(p[1:3])
        #idxs.remove(idx)
        t = targets[idxs[idx]]
        del(idxs[idx])

        if(int(t[0]) != int(p[0])):
            return 'wrong_class'

        if dist>distance_ok:
            return 'too_far'

        if t[3] != -100:
            angle_t = sklearn.preprocessing.normalize(t[3:].cpu().numpy()[None])[0]
            angle_d = math.degrees(math.acos((angle_t @ p[3:5]).round(decimals=4)))
            if angle_d > angle_ok:
                return 'wrong_direction'

    return True


def calculate_accuracy(model, dataloader, max_samples=0, save_idxs=False, get_results_args={}):
    model.eval()
    #wrongs = defaultdict(int) # premade it so its always in specific order
    wrongs = {'not_enough_preds': 0, 'too_many_preds': 0, 'wrong_class': 0, 'too_far': 0, 'wrong_direction': 0}
    wrong_idxs = defaultdict(list)
    wrong_samples = []
    wrong_i, correct_i, total = 0.0, 0.0, 0
    #total = len(databunch.valid_dl.dataset)
    with torch.no_grad():
        for img_tensors, targets in fastprogress.progress_bar(dataloader):
            outputs = model(img_tensors)
            results = get_results(outputs, **get_results_args)
            for idx, (r, t, img) in enumerate(zip(results, targets, img_tensors)):
                correct = compare_single_result_target(r, t)
                if correct is not True:
                    wrong_i += 1
                    wrongs[correct] += 1
                    if save_idxs:
                        wrong_idxs[correct].append(int(total+idx))
                    if max_samples > 0:
                        arr = (img.cpu(), r, t.cpu(), correct, total+idx)
                        if len(wrong_samples) < max_samples: wrong_samples.append(arr)
                        else: wrong_samples[random.randint(0, max_samples-1)] = arr
                else:
                    correct_i += 1
            total += len(img_tensors)

    return correct_i/total, wrongs, wrong_samples, wrong_idxs


# Single file loaded at a time. Makes it possible to predict with varying network sizes
# corr, wrongs, wrong_samples, wrong_idxs = calculate_accuracy_files(model, img_fnames_arr[2][300:320], 416, max_samples=10)
# plot_results(*list(zip(*wrong_samples[:6])))
def calculate_accuracy_files(model, file_names, size, max_samples=0, save_idxs=True, get_results_args={}):
    model.eval()
    wrongs = {'not_enough_preds': 0, 'too_many_preds': 0, 'wrong_class': 0, 'too_far': 0, 'wrong_direction': 0}
    wrong_idxs = defaultdict(list)
    wrong_samples = []
    wrong_i, correct_i, total = 0.0, 0.0, 0
    with torch.no_grad():
        for idx, fname in fastprogress.progress_bar(enumerate(file_names), total=len(file_names)):
            img_org = cv2.imread(fname)
            img_input, resize_ratio, crop_size = im_crop_and_resize(img_org, size, True)
            img_tensor = im2tensor(img_input).cuda()

            results = predict_img(model, img_tensor, **get_results_args)

            labels = []
            for l in HandsDataset.read_label(fname):
                l[1] = l[1]*resize_ratio-math.ceil(crop_size[1])
                l[2] = l[2]*resize_ratio-math.ceil(crop_size[0])
                if len(l) > 3:
                    l[3] = l[3]*resize_ratio-math.ceil(crop_size[1])
                    l[4] = l[4]*resize_ratio-math.ceil(crop_size[0])
                labels.append(l)
            t = HandItems.create(labels, h=img_input.shape[0], w=img_input.shape[1]).data

            r = results[0]
            img = img_tensor

            correct = compare_single_result_target(r, t)
            if correct is not True:
                wrong_i += 1
                wrongs[correct] += 1
                if save_idxs:
                    wrong_idxs[correct].append(int(idx))
                if max_samples > 0:
                    arr = (img.cpu(), r, t.cpu(), correct, idx)
                    if len(wrong_samples) < max_samples: wrong_samples.append(arr)
                    else: wrong_samples[random.randint(0, max_samples-1)] = arr
            else:
                correct_i += 1

            total += 1

    return correct_i/total, wrongs, wrong_samples, wrong_idxs
