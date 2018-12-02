# Some magick to track multiple losses within fastai. If loss class inherits AvgMultiLoss
# then save the information to learn.recorder
#
# see https://forums.fast.ai/t/allow-for-more-than-one-output-for-loss-and-metric/21991/40
# and https://github.com/fastai/fastai_docs/blob/master/dev_nb/100_add_metrics.ipynb
#
# learn = fastai.Learner(data=databunch, model=model, loss_func=HandsLoss(track_train=False))
# learn.callback_fns.append(HandleMultiLoss)
#
# class ExampleMultiLoss(AvgMultiLoss):
#     def __call__(self, output, target):
#         pct = uniform(0,1)
#         loss = F.cross_entropy(output, target)
#         self.loss_conf = pct * loss
#         self.loss_coord = (1-pct) * loss
#         return loss
#     def loss_names(self):
#         return ['loss_conf', 'loss_coord']
#     def losses(self):
#         return [self.loss_conf.detach(), self.loss_coord.detach()]


from hands.basics import *
from abc import ABC, ABCMeta, abstractmethod

class AvgMultiLoss(ABC):
    def __init__(self, track_train=False):
        self.track_train=track_train
        
    @abstractmethod
    def __call__(self, output, target):
        pass
    @abstractmethod
    def loss_names(self):
        pass
    @abstractmethod
    def losses(self):
        pass
    

class HandleAvgMultiLoss(fastai.LearnerCallback):
    _order = -20 #Needs to run before the recorder
    
    def on_train_begin(self, **kwargs):
        self.in_use = isinstance(self.learn.loss_func, AvgMultiLoss)
        if self.in_use:
            loss_names = self.learn.loss_func.loss_names()
            self.loss_c = len(loss_names)
            self.track_train = self.learn.loss_func.track_train
            if self.track_train:
                names = ["t_"+n for n in loss_names]
                names.extend(["v_"+n for n in loss_names])
                self.learn.recorder.add_metric_names(names)
            else:
                self.learn.recorder.add_metric_names(loss_names)
    
    def on_epoch_begin(self, **kwargs):
        if self.in_use:
            self.avgs_train = [0. for _ in range(self.loss_c)]
            self.nums_train = 0.
            self.avgs_val = [0. for _ in range(self.loss_c)]
            self.nums_val = 0.
    
    def on_batch_end(self, last_target, train, **kwargs):
        if self.in_use:
            bs = last_target.size(0)
            if train and self.track_train:
                losses = self.learn.loss_func.losses()
                for i in range(self.loss_c):
                    self.avgs_train[i] += bs * losses[i]
                self.nums_train += bs                
            if not train:
                losses = self.learn.loss_func.losses()
                for i in range(self.loss_c):
                    self.avgs_val[i] += bs * losses[i]
                self.nums_val += bs
    
    def on_epoch_end(self, **kwargs):
        if self.in_use:
            metrics = []
            if self.track_train:
                m = [avg/self.nums_train for avg in self.avgs_train] if self.nums_train > 0 else [0 for _ in self.avgs_train]
                metrics.extend(m)
            m = [avg/self.nums_val for avg in self.avgs_val] if self.nums_val > 0 else [0 for _ in self.avgs_val]
            metrics.extend(m)

            self.learn.recorder.add_metrics(metrics)

