import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        """
        FPN模块
        
        Args:
            in_channels_list (list[int]): 每个输入特征图的通道数（C2, C3, C4, C5）。
            out_channels (int): FPN输出特征图的通道数。
        """
        super(FPN, self).__init__()
        # 横向连接：1x1卷积
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        
        # 自顶向下路径：3x3卷积
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs (list[torch.Tensor]): 输入的特征图列表，按照C2, C3, C4, C5顺序。
        
        Returns:
            dict[str, torch.Tensor]: 输出的特征金字塔，包含P2, P3, P4, P5。
        """
        # 横向连接
        lateral_features = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]
        
        # 自顶向下融合
        num_layers = len(lateral_features)
        for i in range(num_layers - 1, 0, -1):
            # 上采样
            upsampled = F.interpolate(lateral_features[i], scale_factor=2, mode='nearest')
            # 融合
            lateral_features[i-1] += upsampled
        
        # 输出卷积
        output_features = [output_conv(x) for output_conv, x in zip(self.output_convs, lateral_features)]
        
        # 构建字典形式的特征金字塔
        out = {}
        out['P2'] = output_features[0]
        out['P3'] = output_features[1]
        out['P4'] = output_features[2]
        out['P5'] = output_features[3]
        
        return out

# 示例使用
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256).to(device)
    
    # 假设有四个特征图，C2, C3, C4, C5
    c2 = torch.randn(1, 256, 200, 200).to(device)
    c3 = torch.randn(1, 512, 100, 100).to(device)
    c4 = torch.randn(1, 1024, 50, 50).to(device)
    c5 = torch.randn(1, 2048, 25, 25).to(device)  # 已修改为1/16分辨率
    
    fpn_features = fpn([c2, c3, c4, c5])
    for key, feature in fpn_features.items():
        print(f"{key} shape: {feature.shape}")
    # 输出:
    # P2 shape: torch.Size([1, 256, 200, 200])
    # P3 shape: torch.Size([1, 256, 100, 100])
    # P4 shape: torch.Size([1, 256, 50, 50])
    # P5 shape: torch.Size([1, 256, 50, 50])
