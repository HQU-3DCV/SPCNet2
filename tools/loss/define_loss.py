import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class SPContrastiveLoss(nn.Module):
    def __init__(self, num_classes, temperature=0.1, ignore_index=255):
        """
        超点级别的监督对比损失（内置伪标签生成）
        :param num_classes: 数据集的总类别数 (例如 ScanNet 是 20)
        :param temperature: 对比损失的温度系数
        :param ignore_index: 真实标签中需要忽略的类别索引 (如 255 或 -1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(self, sp_features, point_to_sp_index, raw_labels):
        """
        :param sp_features:       (M, C)  当前批次的所有超点特征，例如 (M, 256)
        :param point_to_sp_index: (N, )   有效原始点所属的超点索引
        :param raw_labels:        (N, )   有效原始点的真实语义标签
        """
        # =======================================================
        # 1. 内部计算超点伪标签 (多数投票)
        # =======================================================
        # 过滤掉 ignore_index
        valid_mask = (raw_labels >= 0) & (raw_labels != self.ignore_index)
        valid_labels = raw_labels[valid_mask].long()
        valid_index = point_to_sp_index[valid_mask].long()

        # 转 One-Hot 并统计每个超点的类别票数
        label_one_hot = F.one_hot(valid_labels, num_classes=self.num_classes).float()
        M = sp_features.shape[0]
        sp_label_counts = scatter_add(label_one_hot, valid_index, dim=0, dim_size=M)

        # 找出每个超点得票最多的类别作为硬标签
        _, sp_labels = torch.max(sp_label_counts, dim=1)

        # 找出真正包含有效点的超点 (防止有些超点全都是 ignore 的点)
        sp_valid_mask = sp_label_counts.sum(dim=1) > 0

        # =======================================================
        # 2. 计算对比损失
        # =======================================================
        features = sp_features[sp_valid_mask]
        labels = sp_labels[sp_valid_mask]

        # 如果有效超点少于2个，没法算对比
        if features.shape[0] < 2:
            return sp_features.new_tensor(0.0)

        # L2 归一化 (极度关键，映射到单位球面)
        features = F.normalize(features, p=2, dim=1)

        # 计算余弦相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 构建正样本掩码 (同类为 1，异类为 0)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # 对角线清零 (不和自己作对比)
        N = features.shape[0]
        eye = torch.eye(N, device=features.device)
        mask = mask * (1.0 - eye)

        # 数值稳定性处理
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        # 计算 InfoNCE 分母
        exp_logits = torch.exp(logits) * (1.0 - eye)
        denominator = exp_logits.sum(dim=1, keepdim=True) + 1e-8

        log_prob = logits - torch.log(denominator)

        # 找出有同类可以拉近的有效 anchor (防止有些类别在当前 batch 只有一个超点)
        mask_sum = mask.sum(dim=1)
        valid_anchor_mask = mask_sum > 0

        if not valid_anchor_mask.any():
            return sp_features.new_tensor(0.0)

        # 仅在有正样本的超点上计算损失
        mean_log_prob_pos = (mask[valid_anchor_mask] * log_prob[valid_anchor_mask]).sum(dim=1) / mask_sum[
            valid_anchor_mask]

        loss = -mean_log_prob_pos.mean()
        return loss