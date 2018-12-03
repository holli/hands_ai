from hands.basics import *

# _ = torch.manual_seed(42); np.random.seed(42); random.seed(42)
import sklearn
import functools


def get_default_databunch(img_fnames_arr, tfms_arr=None, size=224, max_lines=None, bs=32):
    if not tfms_arr:
        tfms_xtra = [fastai.vision.transform.jitter(magnitude=(-.005, .005), p=.75),
                     fastai.vision.transform.rand_zoom(scale=(0.8,1.4), p=1)]
        tfms_arr = fastai.vision.get_transforms(max_zoom=1, max_lighting=0.5, max_warp=0.2,
                                                max_rotate=20, xtra_tfms=tfms_xtra)
        # tfms_arr = [tfms_arr[0],
        #             [fastai.vision.transform.crop_pad(row_pct=(0,1), col_pct=(0,1))], # randomize crop position, not always on the center
        #             ]
    if max_lines:
        img_fnames_arr = [a[:max_lines] for a in img_fnames_arr]

    i = 0
    tfms = sorted(fastai.core.listify(tfms_arr[i]), key=lambda o: o.tfm.order)
    train_ds = HandsDataset(img_fnames_arr[i], tfms=tfms, size=size)

    i = 1
    tfms = sorted(fastai.core.listify(tfms_arr[i]), key=lambda o: o.tfm.order)
    valid_ds = HandsDataset(img_fnames_arr[i], tfms=tfms, size=size)

    test_ds = None
    if len(img_fnames_arr) > 2:
        test_ds = HandsDataset(img_fnames_arr[2], tfms=tfms, size=size)

    databunch = fastai.basic_data.DataBunch.create(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, bs=bs)
    return databunch


class HandsDataset(Dataset):
    def __init__(self, fnames, tfms=None, size, return_data=True, img_padding_mode='zeros'):
        self.fnames = fnames
        self.tfms = tfms
        self.size = size
        self.read_all_labels()
        self.return_data = return_data
        self.img_padding_mode = img_padding_mode

    # HandsDataset.read_label(path)
    @staticmethod
    def read_label(path):
        with open(path.replace('.jpg', '.json')) as f:
            labels = json.load(f)
        return labels

    def read_all_labels(self,):
        self.labels = []
        for path in self.fnames:
            lab = self.read_label(path)
            self.labels.append(lab)

    def get_img(self, idx):
        return fastai.vision.image.open_image(self.fnames[idx])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_org = self.get_img(idx)
        hands_org = HandItems.create(self.labels[idx], *img_org.size)

        for _ in range(4): # try x times to get img with hands in it
            # img = fastai.vision.apply_tfms(self.tfms, img_org, size=self.size)
            # hands = fastai.vision.apply_tfms(self.tfms, hands_org, size=self.size, do_resolve=False)
            img = img_org.apply_tfms(self.tfms, size=self.size, padding_mode=self.img_padding_mode)
            hands = hands_org.apply_tfms(self.tfms, size=self.size, do_resolve=False)
            if len(hands.get_hands()) > 0: break

        if self.return_data: return img.data, hands.data
        else: return img, hands


class HandItems(fastai.vision.ImagePoints):
    # CLASS_NAMES_ALIASES = (('unknown',), ('open_hand', 'hand_open', '1_hand_open'), ('finger_point', '3_finger_point',), ('fist',), ('pinch',),
    #                         ('one',), ('two',), ('three',), ('four',),
    #                         ('thumbs_up',), ('thumbs_down',), ('finger_gun', ))
    # CLASS_NAMES = tuple([arr[0] for arr in CLASS_NAMES_ALIASES])
    CLASS_NAMES = ('unknown', 'open_hand', 'finger_point', 'fist', 'pinch', 'one', 'two', 'three', 'four', 'thumbs_up', 'thumbs_down', 'finger_gun')

    def __init__(self, labels, label_points_count, flow, scale:bool=True, y_first:bool=True, output_sz=3, *args, **kwargs):
        self.labels = labels
        self.label_points_count = label_points_count
        self.output_sz = output_sz
        super().__init__(flow, scale, y_first)

    @classmethod
    def create(cls, y, h, w, *args, **kwargs):
        labels = [cls.get_label_idx(arr[0]) for arr in y]

        locs = [a[1:] for a in y]
        label_points_count = [len(a)//2 for a in locs]

        locs = [item for sublist in locs for item in sublist] # flatten list
        locs = np.array(locs).reshape(-1, 2)
        locs[:,[0, 1]] = locs[:,[1, 0]] # swap xy to yx
        #locs = np.array([[arr[2], arr[1], arr[4], arr[3]] for arr in y]).reshape(-1, 2)
        flow = fastai.vision.image.FlowField((h, w), torch.tensor(locs).float())

        return cls(labels, label_points_count, flow, *args, **kwargs)

    def clone(self):
        "Mimic the behavior of torch.clone for `Image` objects."
        return self.__class__(self.labels.copy(), self.label_points_count,
                              fastai.vision.image.FlowField(self.size, self.flow.flow.clone()),
                              scale=False, y_first=False, output_sz=self.output_sz)

    def __repr__(self): return f'{self.__class__.__name__} ({self.flow.size}: {self.labels})'
    def _repr_png_(self): return None
    def _repr_jpeg_(self): return None

    def coord(self, func, *args, **kwargs):
        "Put `func` with `args` and `kwargs` in `self.flow_func` for later."
        if 'invert' in kwargs: kwargs['invert'] = True
        elif func.__name__ not in ('jitter'):
            warn(f"{func.__name__} isn't implemented for {self.__class__}.")
        self.flow_func.append(functools.partial(func, *args, **kwargs))
        return self

    @staticmethod
    def get_label_idx(label):
        return HandItems.CLASS_NAMES.index(label)
        # for i, aliases in enumerate(HandItems.CLASS_NAMES_ALIASES):
        #     if label in aliases:
        #         return i
        # raise ValueError(f"HandItems.CLASS_NAMES_ALIASES does not include {label}")

    def get_hands(self):
        flow = self.flow #This updates flow before we test if some transforms happened
        flow = flow.flow.flip(1)
        # flow = ((flow + 1) / 2)

        hands = []
        # for i in range(len(flow)//2):
        loc_i = 0
        for i in range(len(self.labels)):
            loc = None
            lab = self.labels[i]
            if self.label_points_count[i] == 1:
                loc = flow[loc_i:loc_i+1].flatten()
                loc_i += 1
            elif self.label_points_count[i] == 2:
                loc = flow[loc_i:loc_i+2].flatten()
                loc_i += 2

            loc = ((loc + 1) / 2).flatten()

            if ((loc > 0) & (loc < 1)).all():
                if self.label_points_count[i] == 1:
                    hands.append([self.labels[i], *loc, -100, -100])
                else:
                    # loc[2:] -= loc[:2]
                    dxy = (loc[2:] - loc[:2]) * torch.tensor(self.size[:2], dtype=torch.float)
                    loc[2:] = torch.tensor(sklearn.preprocessing.normalize(dxy[None])[0], dtype=torch.float)
                    hands.append([self.labels[i], *loc])

        return np.array(hands)

    @property
    def data(self):
        hands = self.get_hands()
        hands = torch.tensor(hands).float()
        #hands = F.pad(hands, (0, 0, 0, (self.output_sz-len(hands))))
        hands = torch.cat((hands, torch.zeros(((self.output_sz-len(hands)),5))))
        return hands

    def show(self, img, ax:plt.Axes=None, figsize:tuple=(6,6), hide_axis=True, lw=2):
        if ax is None: _,ax = plt.subplots(figsize=figsize)
        img.show(ax, hide_axis=hide_axis)

        colors = np.array([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
        for ic, (lab, y, x, dy, dx) in enumerate(self.get_hands()):
            lab = self.CLASS_NAMES[int(lab)]
            y = float(y)*self.size[0]; x = float(x)*self.size[1]
            color = colors[ic%len(colors)]
            _ = ax.add_patch(patches.Circle([x, y], 0.01*self.size[1], fill=False, edgecolor=color, lw=lw))
            if dy != -100 and dx != -100:
                #dy *= self.size[0]; dx *= self.size[1]
                dy *= min(self.size)//10; dx *= min(self.size)//10
                _ = ax.add_patch(patches.Arrow(x, y, dx, dy, color=color, lw=lw))
            patch = ax.text(max(0, x-20), y+self.size[1]*0.03, lab, verticalalignment='top', color=color, fontsize=15, weight='normal')
            patch.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black', alpha=0.5), patheffects.Normal()])

