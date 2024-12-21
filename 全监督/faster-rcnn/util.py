import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

def decode_boxes(anchors, bbox_preds):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = bbox_preds[:, 0]
    dy = bbox_preds[:, 1]
    dw = bbox_preds[:, 2]
    dh = bbox_preds[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(bbox_preds)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def generate_proposals(cls_logits, bbox_preds, images, anchors, nms_thresh=0.7, pre_nms_top_n=6000, post_nms_top_n=2000, device='cuda'):
    batch_size = cls_logits.shape[0]
    num_anchors = anchors.shape[0]
    H, W = cls_logits.shape[2], cls_logits.shape[3]
    
    # 分类得分处理
    cls_logits = cls_logits.view(batch_size, num_anchors, 2, H, W)
    cls_scores = F.softmax(cls_logits, dim=2)[:, :, 1, :, :]  # 前景概率
    cls_scores = cls_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)  # (batch_size, num_anchors * H * W)
    
    # 边界框回归参数处理
    bbox_preds = bbox_preds.view(batch_size, num_anchors * 4, H, W)
    bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # (batch_size, num_anchors * H * W, 4)
    
    # 解码边界框
    decoded_boxes = decode_boxes(anchors.expand(batch_size, -1, -1).reshape(batch_size, -1, 4), bbox_preds)  # (batch_size, N, 4)
    
    proposals = []
    for i in range(batch_size):
        scores = cls_scores[i]  # (N,)
        boxes = decoded_boxes[i]  # (N, 4)
        
        # 筛选得分较高的区域
        topk_scores, topk_inds = scores.topk(pre_nms_top_n, sorted=True)
        topk_boxes = boxes[topk_inds]
        
        # 应用NMS
        keep = nms(topk_boxes, topk_scores, nms_thresh)
        keep = keep[:post_nms_top_n]
        proposals.append(topk_boxes[keep])
    
    return proposals

# 示例使用
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 假设batch_size=1, num_anchors=9, H=W=50
    batch_size = 1
    num_anchors = 9
    H, W = 50, 50
    images = torch.randn(batch_size, 3, 800, 800).to(device)
    
    # RPN输出
    cls_logits = torch.randn(batch_size, num_anchors * 2, H, W).to(device)
    bbox_preds = torch.randn(batch_size, num_anchors * 4, H, W).to(device)
    
    # 生成锚框
    anchor_sizes = [128, 256, 512]
    aspect_ratios = [0.5, 1.0, 2.0]
    anchors = generate_anchors((H, W), anchor_sizes, aspect_ratios, device)
    
    # 生成候选区域
    proposals = generate_proposals(cls_logits, bbox_preds, images, anchors, device=device)
    
    for i, prop in enumerate(proposals):
        print(f"Image {i} proposals shape: {prop.shape}")  # 例如: torch.Size([2000, 4])
