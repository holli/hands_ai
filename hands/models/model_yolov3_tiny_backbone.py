from hands.basics import *
#from hands.data import *
#from hands.multiloss import *
# import yolov3_pytorch.yolov3_tiny
# from yolov3_pytorch.yolov3_tiny import Yolov3TinyBackbone
# from yolov3_pytorch.yolov3_base import Upsample



class ModelYoloV3TinyBackboneWithUp(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Yolov3TinyBackbone()

        self.num_classes = num_classes

        self.up_1 = nn.Sequential(OrderedDict([
            ('17_convbatch',    fastai.layers.conv_layer(256, 128, 1, 1)),
            ('18_upsample',     Upsample(2)),
        ]))

        self.yolo_1_pre = nn.Sequential(OrderedDict([
            ('19_convbatch',    fastai.layers.conv_layer(128+256, 512, 3, 1)),
            ('20_conv',         nn.Conv2d(512, 1*(5+self.num_classes), 1, 1, 0)),
        ]))


    def forward(self, x):
        x_b_0, x_b_full = self.backbone(x)

        x_up = self.up_1(x_b_full)
        x_up = torch.cat((x_up, x_b_0), 1)
        y1 = self.yolo_1_pre(x_up)

        return y1

    @classmethod
    def load_default_416(cls):
        model = cls(num_classes=12).cuda()
        model.default_size = 416
        model.load_state_dict(torch.load('data/models/21_1/5_big.pth'))
        return model.eval()



class ModelYoloV3TinyBackbone(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Yolov3TinyBackbone()

        self.num_classes = num_classes

        self.pre_0 = nn.Sequential(OrderedDict([
            #('14_convbatch',    ConvBN(256, 512, 3, 1, 1)),
            ('14_convbatch',    fastai.layers.conv_layer(256, 512, 3, 1)),
            # dx + dy + ax + ay + objectness + classes
            ('15_conv',         nn.Conv2d(512, 1*(5+self.num_classes), 1, 1, 0)),
        ]))


    def forward(self, x):
        x_b_0, x_b_full = self.backbone(x)

        y = self.pre_0(x_b_full)

        return y

    @classmethod
    def load_default_224(cls):
        model = cls(num_classes=12).cuda()
        model.default_size = 224
        model.load_state_dict(torch.load('data/models/12/5.pth'))
        return model.eval()

    @classmethod
    def load_default_320(cls):
        model = cls(num_classes=12).cuda()
        model.default_size = 320
        model.load_state_dict(torch.load('data/models/17/4.pth'))
        return model.eval()

    @classmethod
    def load_default_416(cls):
        model = cls(num_classes=12).cuda()
        model.default_size = 416
        model.load_state_dict(torch.load('data/models/17_3/7.pth'))
        return model.eval()

    @classmethod
    def load_default_512(cls):
        model = cls(num_classes=12).cuda()
        model.default_size = 512
        model.load_state_dict(torch.load('data/models/17_3/9.pth'))
        return model.eval()

    @classmethod
    def load_default_608(cls):
        model = cls(num_classes=12).cuda()
        model.default_size = 608
        model.load_state_dict(torch.load('data/models/17_3/11.pth'))
        return model.eval()







