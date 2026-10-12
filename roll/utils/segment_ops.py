"""Shared masked reduction, extracted unchanged from ROLL."""
import torch

def compute_segment_masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    对每段连续的1分别计算 masked_mean，不连续的段不相乘。

    Args:
        tensor: [batch_size, seq_len] 要计算的值
        mask: [batch_size, seq_len] mask，1表示有效位置，0表示无效位置

    Returns:
        [batch_size, seq_len] 结果，每段连续的1位置填充该段的 masked_mean
    """
    batch_size, seq_len = mask.shape
    device = mask.device
    result = torch.zeros_like(tensor)

    # 对每个样本分别处理
    for b in range(batch_size):
        sample_mask = mask[b]  # [seq_len]
        sample_tensor = tensor[b]  # [seq_len]

        # 找到所有连续的1的段
        # 使用 diff 找到边界：1->0 和 0->1 的位置
        diff = torch.diff(sample_mask, prepend=torch.tensor([0], device=device))
        # 找到段的开始位置（0->1）
        segment_starts = torch.where(diff == 1)[0]
        # 找到段的结束位置（1->0），diff[i]==-1 表示 mask[i-1]==1 且 mask[i]==0，所以段的结束位置是 i（不包括i）
        segment_ends = torch.where(diff == -1)[0]

        # 如果最后一个位置是1，需要添加结束位置
        if sample_mask[-1] == 1:
            segment_ends = torch.cat([segment_ends, torch.tensor([seq_len], device=device)])

        # 确保 segment_starts 和 segment_ends 长度匹配
        if len(segment_starts) != len(segment_ends):
            # 如果长度不匹配，只处理能匹配的部分
            min_len = min(len(segment_starts), len(segment_ends))
            segment_starts = segment_starts[:min_len]
            segment_ends = segment_ends[:min_len]

        # 对每段分别计算 masked_mean
        for start, end in zip(segment_starts, segment_ends):
            # 获取这段的索引
            segment_indices = torch.arange(start, end, device=device)
            segment_mask = sample_mask[segment_indices]  # 这段的mask
            segment_tensor = sample_tensor[segment_indices]  # 这段的值

            if segment_mask.sum() > 0:
                # 计算这段的 masked_mean（只考虑mask为1的位置）
                segment_mean = (segment_tensor * segment_mask).sum() / (segment_mask.sum() + 1e-8)
                # 将结果填充到这段内mask为1的位置
                result[b, segment_indices] = segment_mean * segment_mask

    return result
