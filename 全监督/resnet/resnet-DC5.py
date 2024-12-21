import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ModifiedResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        # 初始层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Layer1, Layer2, Layer3 保持不变
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        
        # 修改 Layer4 为膨胀卷积
        self.layer4 = self._modify_layer4(resnet.layer4)
    
    def _modify_layer4(self, layer4):
        # 修改 Layer4 的第一个 Bottleneck 模块
        for i, bottleneck in enumerate(layer4):
            if i == 0:
                # 修改第一个 Bottleneck 的步幅和膨胀率
                bottleneck.conv2 = nn.Conv2d(
                    in_channels=bottleneck.conv1.in_channels,
                    out_channels=bottleneck.conv2.out_channels,
                    kernel_size=3,
                    stride=1,  # 将步幅从2改为1
                    padding=2,  # 根据膨胀率调整 padding
                    dilation=2,
                    bias=False
                )
                if bottleneck.downsample is not None:
                    bottleneck.downsample[0] = nn.Conv2d(
                        in_channels=bottleneck.downsample[0].in_channels,
                        out_channels=bottleneck.downsample[0].out_channels,
                        kernel_size=1,
                        stride=1,  # 将步幅从2改为1
                        bias=False
                    )
        return layer4
    
    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer1, Layer2, Layer3
        c2 = self.layer1(x)  # C2
        c3 = self.layer2(c2) # C3
        c4 = self.layer3(c3) # C4
        c5 = self.layer4(c4) # C5 (使用膨胀卷积)
        
        return c2, c3, c4, c5

# 示例使用
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModifiedResNet50(pretrained=True).to(device)
    input_tensor = torch.randn(1, 3, 800, 800).to(device)
    c2, c3, c4, c5 = model(input_tensor)
    print(f"C2 shape: {c2.shape}")  # 1, 256, 200, 200
    print(f"C3 shape: {c3.shape}")  # 1, 512, 100, 100
    print(f"C4 shape: {c4.shape}")  # 1, 1024, 50, 50
    print(f"C5 shape: {c5.shape}")  # 1, 2048, 50, 50 （保持1/16）
