from hands.basics import *
#from hands.data import *
#from hands.multiloss import *
import yolov3_pytorch.yolov3_tiny
from yolov3_pytorch.yolov3_tiny import Yolov3TinyBackbone


import yolov3_pytorch
from yolov3_pytorch import yolov3

class ModelDarknetCustomized(torch.nn.Module):
    def __init__(self, num_classes, darknet_layers, darknet_output):
        super().__init__()
        self.backbone = yolov3.Darknet(darknet_layers)

        self.num_classes = num_classes

        self.pre_0 = nn.Sequential(OrderedDict([
            ('14_convbatch',    fastai.layers.conv_layer(darknet_output, 512, 3, 1)),
            # dx + dy + ax + ay + objectness + classes
            ('15_conv',         nn.Conv2d(512, 1*(5+self.num_classes), 1, 1, 0)),
        ]))


    def forward(self, x):
        x_b_full = self.backbone(x)[-1]
        y = self.pre_0(x_b_full)
        return y


    @classmethod
    def load_default_4_416(cls):
        model = cls(num_classes=12, darknet_layers=[1,2,8,8], darknet_output=512)
        h5path = 'data/models/22_1/5_big.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 416
        return model.eval()


    @classmethod
    def load_default_3_416(cls):
        model = cls(num_classes=12, darknet_layers=[1,2,8], darknet_output=256)
        h5path = 'data/models/22_2/5_big.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 416
        return model.eval()


    @classmethod
    def load_default_3_320(cls):
        model = cls(num_classes=12, darknet_layers=[1,2,8], darknet_output=256)
        h5path = 'data/models/22_2/3_small_3.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 320
        return model.eval()





