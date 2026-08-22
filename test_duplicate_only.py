import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from collections import defaultdict
from diversity_prototype import diversity_aware_prototype
from models.protonet import PrototypicalNetwork
from PIL import Image, ImageFilter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_support_and_query(test_set, class_id, n_shot=5, n_query=20):
    targets = torch.tensor(test_set.targets)
    idx = (targets == class_id).nonzero().squeeze()
    if idx.numel() < n_shot + n_query:
        perm = idx[torch.randperm(len(idx))]
        support_idx = perm[:min(n_shot, len(perm))]
        query_idx = perm[min(n_shot, len(perm)):min(n_shot+n_query, len(perm))]
    else:
        perm = idx[torch.randperm(len(idx))]
        support_idx = perm[:n_shot]
        query_idx = perm[n_shot:n_shot+n_query]
    
    support_imgs = torch.stack([transform(test_set.data[i]) for i in support_idx])
    query_imgs = torch.stack([transform(test_set.data[i]) for i in query_idx])
    return support_imgs.cuda(), query_imgs.cuda()

def compute_prototype_mean(features):
    return features.mean(dim=0)

def compute_prototype_dra(features, tau=0.1, tau_d=0.95):
    proto, _ = diversity_aware_prototype(features, tau=tau, tau_d=tau_d)
    return proto

def run_single_experiment(encoder, test_set, n_classes=5, n_shot=5, n_query=20, seed=42):
    """
    只运行 duplicate 污染实验
    返回: (acc_mean, acc_dra) 即普通平均和DRA的准确率
    """
    set_seed(seed)
    
    novel_classes = random.sample(range(60, 100), n_classes)
    
    # 基类原型
    base_prototypes = []
    for c in range(60):
        sup, _ = get_support_and_query(test_set, c, n_shot=1, n_query=1)
        with torch.no_grad():
            feat = encoder(sup).cpu()
        base_prototypes.append(feat.mean(dim=0))
    base_prototypes = torch.stack(base_prototypes)  # [60, D]
    
    correct_mean = 0
    correct_dra = 0
    total = 0
    
    for class_id in novel_classes:
        support_imgs, query_imgs = get_support_and_query(test_set, class_id, n_shot, n_query)
        
        # ===== 应用 duplicate 污染 =====
        # 复制第1张图片到第2、3、4张 (替换原来的2、3、4)
        support_dirty = support_imgs.clone()
        for i in range(1, 4):  # 索引1,2,3
            support_dirty[i] = support_imgs[0].clone() + 0.01 * torch.randn_like(support_imgs[0])
        
        with torch.no_grad():
            features = encoder(support_dirty).cpu()
            query_feats = encoder(query_imgs).cpu()
        
        # ---- 普通平均 ----
        proto_mean = compute_prototype_mean(features)
        proto_mean = proto_mean / (proto_mean.norm() + 1e-8)
        
        # ---- DRA ----
        proto_dra = compute_prototype_dra(features, tau=0.1, tau_d=0.95)
        proto_dra = proto_dra / (proto_dra.norm() + 1e-8)
        
        # 合并基类和新类原型
        all_protos_mean = torch.cat([base_prototypes, proto_mean.unsqueeze(0)], dim=0)
        all_protos_dra = torch.cat([base_prototypes, proto_dra.unsqueeze(0)], dim=0)
        
        all_protos_mean = all_protos_mean / (all_protos_mean.norm(dim=1, keepdim=True) + 1e-8)
        all_protos_dra = all_protos_dra / (all_protos_dra.norm(dim=1, keepdim=True) + 1e-8)
        
        sim_mean = query_feats @ all_protos_mean.T
        sim_dra = query_feats @ all_protos_dra.T
        
        pred_mean = sim_mean.argmax(dim=1)
        pred_dra = sim_dra.argmax(dim=1)
        
        correct_mean += (pred_mean == 60).sum().item()
        correct_dra += (pred_dra == 60).sum().item()
        total += n_query
    
    return correct_mean / total, correct_dra / total

def main():
    print("=" * 70)
    print("专用实验: duplicate 污染 (普通平均 vs DRA)")
    print("污染方式: 将支撑集第2、3、4张替换为第1张+小噪声")
    print("=" * 70)
    
    model = PrototypicalNetwork().cuda()
    model.load_state_dict(torch.load("checkpoints/protonet_resnet18.pth"))
    model.eval()
    encoder = model.encoder
    
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=None
    )
    
    n_seeds = 3
    results_mean = []
    results_dra = []
    
    for seed in range(1, n_seeds + 1):
        acc_mean, acc_dra = run_single_experiment(
            encoder, test_set,
            n_classes=5, n_shot=5, n_query=20,
            seed=seed
        )
        results_mean.append(acc_mean)
        results_dra.append(acc_dra)
        print(f"种子 {seed}: 普通平均={acc_mean:.4f}, DRA={acc_dra:.4f}")
    
    # 汇总统计
    mean_mean = np.mean(results_mean)
    mean_dra = np.mean(results_dra)
    std_mean = np.std(results_mean)
    std_dra = np.std(results_dra)
    diff = mean_dra - mean_mean
    
    print("\n" + "=" * 70)
    print("汇总统计 (3个种子)")
    print("=" * 70)
    print(f"{'方法':<12} {'平均准确率':<12} {'标准差':<12}")
    print("-" * 40)
    print(f"{'普通平均':<12} {mean_mean*100:.2f}%       {std_mean*100:.2f}%")
    print(f"{'DRA':<12} {mean_dra*100:.2f}%       {std_dra*100:.2f}%")
    print("-" * 40)
    print(f"DRA 提升: {diff*100:+.2f} 个百分点")
    print("=" * 70)

if __name__ == "__main__":
    main()
