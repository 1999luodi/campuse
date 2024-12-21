
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        # 主干网络
        self.backbone = models.resnet50(pretrained=True)
        # 去掉ResNet的最后一个全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 区域提议网络（RPN）
        self.rpn = RPN()
        
        # RoI池化层
        """在主干网络（如ResNet）中，输入图像经过卷积和下采样后，特征图的尺寸通常是原始图像的1/16。例如，输入图像大小为800x800，经过ResNet50后，特征图可能为50x50。
        spatial_scale=1/16 的作用是将RoI在原始图像上的坐标转换为特征图上的坐标。例如，原始图像上的一个RoI坐标（x1, y1, x2, y2）在特征图上的对应坐标就是（x1 * 1/16, y1 * 1/16, x2 * 1/16, y2 * 1/16）。
        这样，RoIPool能够正确地从特征图中提取对应的区域进行池化。"""
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16) 
        
        # 分类和回归网络
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        """
            详细说明：
            分类层（cls_score）：
                输出维度为num_classes，表示每个RoI属于各个类别的得分。
                如果有20个类别，输出将是20维的。
            回归层（bbox_pred）：
                输出维度为num_classes * 4，即每个类别有4个边界框回归参数。
                这是因为在多类别检测中，模型需要为每个类别预测一个独立的边界框调整参数。
                例如，对于20个类别，输出将是80维的（20 * 4）。
            注意：
                有些实现中，为了简化模型，可能只对背景类别之外的类别进行边界框回归，具体取决于实现细节。
        """
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
    
    def forward(self, images, targets=None):
        # 提取特征
        features = self.backbone(images)
        
        # RPN生成区域提议
        rpn_outputs = self.rpn(features, images, targets)
        proposals = rpn_outputs['proposals']
        
        # RoI池化
        """
        features：主干网络提取的特征图，形状为(batch_size, C, H', W')。
        proposals：由RPN生成的候选区域（RoIs），通常是一个列表，每个元素对应一个图像的RoI坐标，形状为(num_proposals, 4)，坐标格式为(x1, y1, x2, y2)。
        """
        pooled_features = self.roi_pool(features, proposals)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # 分类和回归
        cls_features = self.classifier(pooled_features)
        scores = self.cls_score(cls_features)
        bbox_deltas = self.bbox_pred(cls_features)
        
        # 如果是训练模式，计算损失
        if self.training:
            loss = {}
            loss['loss_rpn'] = rpn_outputs['loss_rpn']
            loss['loss_cls'] = F.cross_entropy(scores, targets['labels'])
            loss['loss_bbox'] = smooth_l1_loss(bbox_deltas, targets['bbox_targets'])
            return loss
        else:
            # 推理模式，返回检测结果
            detections = post_process(scores, bbox_deltas, proposals)
            return detections
        
class RPN(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=512, num_anchors=9):
        super(RPN, self).__init__()
        # 3x3卷积
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        # 分类层
        self.cls_conv = nn.Conv2d(mid_channels, num_anchors * 2, 1)
        # 回归层
        self.bbox_conv = nn.Conv2d(mid_channels, num_anchors * 4, 1)
        
    def forward(self, features, images, targets=None):
        x = self.relu(self.conv(features))
        # 分类预测
        cls_logits = self.cls_conv(x)
        # 回归预测
        bbox_preds = self.bbox_conv(x)
        
        if self.training:
            # 计算损失（分类损失和回归损失）
            loss_rpn = compute_rpn_loss(cls_logits, bbox_preds, targets)
            proposals = generate_proposals(cls_logits, bbox_preds, images)
            return {'loss_rpn': loss_rpn, 'proposals': proposals}
        else:
            # 生成区域提议
            proposals = generate_proposals(cls_logits, bbox_preds, images)
            return {'proposals': proposals}