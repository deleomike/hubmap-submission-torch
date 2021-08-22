import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplab.batchnorm import SynchronizedBatchNorm2d
from models.deeplab.aspp import build_aspp
from models.deeplab.decoder import build_decoder
from models.deeplab.xception import AlignedXception
from models.deeplab.drn import drn_d_54

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == "xception":
        return AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn_d_54(BatchNorm)

class DeepLab(nn.Module):
    def __init__(self, backbone='drn', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            # if isinstance(param, own_state[]):
            #     # backwards compatibility for serialized parameters
            #     param = param.data
            if name == "decoder.last_conv.8.weight":
                own_state[name] = param[0][None, :, :]
            elif name == "decoder.last_conv.8.bias":
                own_state[name] = torch.tensor([param[0]])
            else:
                own_state[name].copy_(param)

        self.load_state_dict(own_state)

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='drn', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())

    state_dict = torch.load("./pretrained/deeplab-drn.pth")["state_dict"]

    model.load_state_dict(state_dict)