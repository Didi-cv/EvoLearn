import torch
import torch.nn.functional as F

def diversity_aware_prototype(support_features, tau=0.1, tau_d=0.95):
    """
    多样性感知鲁棒原型聚合 (DRA-Lite)
    输入:
        support_features: [K, D]  K张支撑样本的特征向量 (已通过encoder提取)
        tau: 温度参数，控制可靠性权重分布
        tau_d: 多样性阈值，高于此值的样本对被判定为"近重复"
    输出:
        p_c: [D] 聚合后的原型向量
        weights: [K] 每个样本的最终权重（用于调试）
    """
    K, D = support_features.shape
    z = support_features  # [K, D]

    # ===== 信号1: 增强一致性 (detect blur/lighting anomalies) =====
    # 注意: 这里假设你已经有数据增强函数augment()
    # 如果当前没有，可以先用随机噪声模拟，后续替换为真实增强
    # 我假设你会在外部传入z_aug，这里做占位处理
    # 实际使用时，请将support_features分别通过原始encoder和augment+encoder得到两组特征
    # 这里为了代码能跑通，暂时用z自身模拟（你后续替换为真实z_aug）
    z_aug = z + 0.01 * torch.randn_like(z)  # 临时占位，后续替换
    sim_aug = F.cosine_similarity(z, z_aug, dim=1)  # [K]
    C_aug = torch.exp(- (1 - sim_aug) / tau)  # 一致性越高，C越大

    # ===== 信号2: 内部共识 (detect clear but mislabeled samples) =====
    # 每张图与同support内其他图的median cosine similarity
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [K, K]
    # 去掉对角线（自己与自己的相似度=1，会干扰median）
    mask = ~torch.eye(K, dtype=torch.bool, device=z.device)
    sim_matrix_masked = sim_matrix[mask].view(K, K-1)
    C_cons = sim_matrix_masked.median(dim=1).values  # [K]

    # ===== 信号3: 综合可靠性分数 =====
    # 简单相乘（也可加权求和）
    r = C_aug * C_cons  # [K]

    # ===== 信号4: 多样性惩罚 (suppress near-duplicate frames) =====
    # 对高度相似的样本对进行降权
    penalty = torch.ones(K, device=z.device)
    for i in range(K):
        # 统计与样本i相似度超过阈值的其他样本数量
        dup_count = (sim_matrix[i] > tau_d).sum().item() - 1  # 去掉自己
        penalty[i] = 1.0 + dup_count

    # 最终权重
    weights = r / penalty  # [K]
    weights = weights / (weights.sum() + 1e-8)  # 归一化

    # 加权原型
    p_c = (weights.unsqueeze(1) * z).sum(dim=0)  # [D]

    return p_c, weights


# ===== 测试函数（可选，跑通验证用）=====
if __name__ == "__main__":
    # 模拟5张图片的特征（512维）
    torch.manual_seed(42)
    dummy_features = torch.randn(5, 512)
    
    p, w = diversity_aware_prototype(dummy_features)
    print("原型形状:", p.shape)
    print("权重:", w)
    print("权重和:", w.sum().item())
