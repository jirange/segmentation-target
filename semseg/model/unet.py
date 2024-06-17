import torch
import torch.nn as nn

import model.resnet50 as models


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class UNet(nn.Module):
    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(UNet, self).__init__()
        assert layers in [50, 101, 152]
        assert classes > 1
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained,unet = True)

        in_filters  = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # upsampling
        # 64,64,512
        self.up_concat4 = UNetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = UNetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = UNetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = UNetUp(in_filters[0], out_filters[0])

        
        self.final_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2), 
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], classes, 1),
        )

    def forward(self,  x, y=None):

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        up4 = self.up_concat4(x3, x4)
        up3 = self.up_concat3(x2, up4)
        up2 = self.up_concat2(x1, up3)
        up1 = self.up_concat1(x0, up2)

        final = self.final_conv(up1)

        if self.training:
            main_loss = self.criterion(final, y)

            return final.max(1)[1], main_loss
        else:
            return final
        
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 512, 512).cuda()
    model = UNet(layers=50, classes=2, pretrained=True).cuda()
    model.eval()
    output = model(input)
    print('UNet', output.size())


