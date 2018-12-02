from hands.data import *
from hands.multiloss import *
from hands.utils import *
import sklearn
import sklearn.preprocessing


class HandsAccuracy(fastai.Callback):
    def __init__(self):
        self.name = 'accuracy'

    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        results = get_results(last_output, conf_thresh=.4, nms=.05)

        for res, tar in zip(results, last_target):
            correct = compare_single_result_target(res, tar, distance_ok=.05, angle_ok=10)
            if correct is True:
                self.correct += 1

        #self.total += len(last_target)
        self.total += len(last_output)

    def on_epoch_end(self, **kwargs):
        self.metric = self.correct/self.total


class HandsLoss(AvgMultiLoss):
    def __call__(self, output, target):
        # global conf, coord, angle, classes, tconf, tcoord, tangle, tclasses, \
        #         classes_mask, coord_mask, angle_mask, loss_conf, loss_coord, loss, nB, nH, nW, nC
        device = output.device
        nB = output.data.size(0)    # batch size
        #nA = len(anchors)
        nC = output.data.size(1) - 5
        nH = output.data.size(2)
        nW = output.data.size(3)

        ix = torch.LongTensor(range(0,output.size(1))).to(device)
        coord = output.index_select(1, ix[0:2]).contiguous().sigmoid()
        # angle = output.index_select(1, ix[2:4]).contiguous().sigmoid()
        angle = output.index_select(1, ix[2:4]).contiguous().tanh()
        conf = output.index_select(1, ix[4]).view(nB, nH, nW).contiguous().sigmoid()
        classes = output.index_select(1, ix[5:])

        #coord = output.index_select(1, ix[0:2]).view(nB, -1, nH*nW).transpose(0,1).contiguous().view(4,cls_anchor_dim)  # x, y, w, h

        tconf = torch.zeros(nB, nH, nW)
        tcoord = torch.zeros(nB, 2, nH, nW)
        coord_mask = torch.zeros(nB, 2, nH, nW)
        tangle = torch.zeros(nB, 2, nH, nW)
        angle_mask = torch.zeros(nB, 2, nH, nW)

        tclasses = torch.zeros(nB, nH, nW).long()
        tclasses[:,:,:] = -100

        # tconf.shape == conf.shape
        for b in range(nB):
            for t in target[b, :]:
                if t[0] == 0: break
                gx, gy = t[1]*nH, t[2]*nW
                gi, gj = int(gx), int(gy)
                tconf[b, gi, gj] = 1

                tclasses[b, gi, gj] = int(t[0])

                coord_mask[b, :, gi, gj] = 1
                tcoord[b, 0, gi, gj] = gx-gi
                tcoord[b, 1, gi, gj] = gy-gj

                if t[4] != -100: # if target has angle also calculate it
                    dx, dy = sklearn.preprocessing.normalize(t[3:5].cpu()[None])[0]
                    angle_mask[b, :, gi, gj] = 1
                    tangle[b, 0, gi, gj] = dx
                    tangle[b, 1, gi, gj] = dy


        tclasses = tclasses.to(device); # classes_mask = classes_mask.to(device)
        tconf = tconf.to(device); tcoord = tcoord.to(device); tangle = tangle.to(device)
        coord_mask = coord_mask.to(device); angle_mask = angle_mask.to(device)

        self.loss_conf = nn.MSELoss(reduction='sum')(conf, tconf) / 10 # / 10 to get it at the same range as rest
        self.loss_coord = nn.MSELoss(reduction='sum')(coord*coord_mask, tcoord*coord_mask)
        self.loss_angle = nn.MSELoss(reduction='sum')(angle*angle_mask, tangle*angle_mask)

        #self.loss_classes = nn.CrossEntropyLoss(reduction='sum')(classes, tclasses)
        self.loss_classes = F.cross_entropy(classes, tclasses, ignore_index=-100, reduction='sum')

        loss = self.loss_conf + self.loss_coord + self.loss_angle + self.loss_classes
        return loss

    def loss_names(self):
        return ['loss_conf', 'loss_cls', 'loss_coord', 'loss_angle']
    def losses(self):
        return [self.loss_conf.detach(), self.loss_classes.detach(), self.loss_coord.detach(), self.loss_angle.detach()]
