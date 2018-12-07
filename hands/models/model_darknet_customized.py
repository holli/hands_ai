from hands.basics import *

class ModelDarknetCustomized(torch.nn.Module):
    def __init__(self, num_classes, darknet_layers, darknet_output):
        super().__init__()
        self.backbone = Darknet(darknet_layers)

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

    ###########################
    # Full sizes

    @classmethod
    def load_full_416(cls, models_path='data/models/'):
        model = cls(num_classes=12, darknet_layers=[1,2,8,8,4], darknet_output=1024)
        h5path = models_path + 'hands_darknet_full_416_v01.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 416
        return model.eval()

    @classmethod
    def load_full_512(cls, models_path='data/models/'):
        model = cls(num_classes=12, darknet_layers=[1,2,8,8,4], darknet_output=1024)
        h5path = models_path + 'hands_darknet_full_512_v01.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 512
        return model.eval()

    @classmethod
    def load_full_608(cls, models_path='data/models/'):
        model = cls(num_classes=12, darknet_layers=[1,2,8,8,4], darknet_output=1024)
        h5path = models_path + 'hands_darknet_full_608_v01.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 608
        return model.eval()

    ###########################
    # Half sizes

    @classmethod
    def load_03_320(cls, models_path='data/models/'):
        model = cls(num_classes=12, darknet_layers=[1,2,8], darknet_output=256)
        h5path = models_path + 'hands_darknet_03_320_v01.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 320
        return model.eval()

    @classmethod
    def load_03_416(cls, models_path='data/models/'):
        model = cls(num_classes=12, darknet_layers=[1,2,8], darknet_output=256)
        h5path = models_path + 'hands_darknet_03_416_v01.pth'
        model.load_state_dict(torch.load(h5path))
        model.default_size = 416
        return model.eval()


###################################################################
## Backbone and helper modules

class Darknet(nn.Module):
    def __init__(self, num_blocks, start_nf=32):
        super().__init__()
        nf = start_nf
        self.base = ConvBN(3, nf, kernel_size=3, stride=1) #, padding=1)
        self.layers = []
        for i, nb in enumerate(num_blocks):
            # dn_layer = make_group_layer(nf, nb, stride=(1 if i==-1 else 2))
            dn_layer = self.make_group_layer(nf, nb, stride=2)
            self.add_module(f"darknet_{i}", dn_layer)
            self.layers.append(dn_layer)
            nf *= 2

    def make_group_layer(self, ch_in, num_blocks, stride=2):
        layers = [ConvBN(ch_in, ch_in*2, stride=stride)]
        for i in range(num_blocks): layers.append(DarknetBlock(ch_in*2))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = [self.base(x)]
        for l in self.layers:
            y.append(l(y[-1]))
        return y

# from fastai.models.darknet import ConvBN
class ConvBN(nn.Module):
    "convolutional layer then batchnorm"
    def __init__(self, ch_in, ch_out, kernel_size = 3, stride=1, padding=None):
        super().__init__()
        if padding is None: padding = (kernel_size - 1) // 2 # we should never need to set padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in//2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x): return self.conv2(self.conv1(x)) + x
