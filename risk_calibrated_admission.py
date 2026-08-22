"""
risk_calibrated_admission.py
RCA（风险校准类别准入）模块

功能：
1. 从基类数据构造伪增量场景（pseudo-incremental episodes）
2. 提取每个候选类别的风险特征向量
3. 训练一个轻量级逻辑回归分类器，预测"是否值得学习"
4. 端侧推理时，仅需一次前向传播即可判断是否准入
"""

import torch
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

# CIFAR-100 超类映射 (用于构造有意义的错标样本)
SUPERCLASS_MAP = {
    0: [0,1,2,3,4],       # 水生哺乳动物
    1: [5,6,7,8,9],       # 鱼类
    2: [10,11,12,13,14],  # 花卉
    3: [15,16,17,18,19],  # 食品容器
    4: [20,21,22,23,24],  # 水果和蔬菜
    5: [25,26,27,28,29],  # 家用电器
    6: [30,31,32,33,34],  # 家具
    7: [35,36,37,38,39],  # 昆虫
    8: [40,41,42,43,44],  # 大型食肉动物
    9: [45,46,47,48,49],  # 大型人造户外物品
    10: [50,51,52,53,54], # 大型自然户外场景
    11: [55,56,57,58,59], # 大型杂食/草食动物
    12: [60,61,62,63,64], # 中型哺乳动物
    13: [65,66,67,68,69], # 非昆虫无脊椎动物
    14: [70,71,72,73,74], # 人物
    15: [75,76,77,78,79], # 爬行动物
    16: [80,81,82,83,84], # 小型哺乳动物
    17: [85,86,87,88,89], # 树木
    18: [90,91,92,93,94], # 交通工具1
    19: [95,96,97,98,99], # 交通工具2
}

CLASS_TO_SUPER = {}
for super_id, members in SUPERCLASS_MAP.items():
    for c in members:
        CLASS_TO_SUPER[c] = super_id

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

class RiskCalibratedAdmission:
    def __init__(self, feature_dim=5):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.threshold = None
        self.is_trained = False
        self.feature_names = ['U_c', 'D_c', 'N_eff', 'r_bar', 'V_c']
    
    def extract_features(self, support_features, existing_prototypes=None):
        """
        从支撑集中提取风险特征向量
        修复：处理设备不匹配问题
        """
        K, D = support_features.shape
        
        # 1. 类内一致性 U_c
        proto = support_features.mean(dim=0)
        distances = torch.norm(support_features - proto.unsqueeze(0), dim=1)
        U_c = distances.mean().item()
        
        # 2. 与最近旧类的距离 D_c
        # ===== 修复：设备迁移 =====
        if existing_prototypes is not None and len(existing_prototypes) > 0:
            # 确保 existing_prototypes 与 proto 在同一设备上
            if existing_prototypes.device != proto.device:
                existing_prototypes = existing_prototypes.to(proto.device)
            dist_to_old = torch.cdist(proto.unsqueeze(0), existing_prototypes)
            D_c = dist_to_old.min().item()
        else:
            D_c = 10.0
        
        # 3. 有效样本数 N_eff (用均匀权重近似)
        weights = torch.ones(K)
        N_eff = (weights.sum() ** 2) / (weights ** 2).sum()
        
        # 4. 平均可靠性 r_bar
        r_bar = 1.0 / (1.0 + distances.mean().item())
        
        # 5. 样本多样性 V_c
        if K > 1:
            sim_matrix = torch.cdist(support_features, support_features)
            mask = ~torch.eye(K, dtype=torch.bool, device=support_features.device)
            V_c = sim_matrix[mask].mean().item()
        else:
            V_c = 0.0
        
        return torch.tensor([U_c, D_c, N_eff.item(), r_bar, V_c], dtype=torch.float32)
    
    def get_support_and_query(self, test_set, class_id, n_shot=5, n_query=20):
        """从测试集中取支撑集和查询集 (不重复)"""
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
    
    def evaluate_prototype(self, encoder, test_set, class_id, support_imgs, query_imgs, existing_prototypes):
        """
        给定一个候选类别，用普通平均构造原型，评估其分类性能
        返回: 新类准确率
        """
        with torch.no_grad():
            support_feats = encoder(support_imgs).cpu()
            query_feats = encoder(query_imgs).cpu()
        
        # 普通平均原型
        proto = support_feats.mean(dim=0)
        proto = proto / (proto.norm() + 1e-8)
        
        # 合并原型
        if existing_prototypes is not None and len(existing_prototypes) > 0:
            all_protos = torch.cat([existing_prototypes, proto.unsqueeze(0)], dim=0)
        else:
            all_protos = proto.unsqueeze(0)
        all_protos = all_protos / (all_protos.norm(dim=1, keepdim=True) + 1e-8)
        
        # 查询集分类
        sim = query_feats @ all_protos.T
        preds = sim.argmax(dim=1)
        new_class_idx = len(existing_prototypes) if existing_prototypes is not None else 0
        new_acc = (preds == new_class_idx).float().mean().item()
        
        return new_acc
    
    def generate_pseudo_episode(self, encoder, test_set, n_shot=5, n_query=20):
        """
        构造一个伪增量场景：
        1. 从基类(0-59)中随机选一个类作为"新类"
        2. 构造支撑集（可能添加污染）和查询集
        3. 评估"该支撑集是否值得学习" (用性能增益作为标签)
        返回: (features, label) 其中label=1表示值得学，0表示不值得学
        """
        # 随机选一个基类作为"新类"
        class_id = random.randint(0, 59)
        
        # 选一些其他基类作为"旧类"
        other_classes = random.sample([c for c in range(60) if c != class_id], min(5, 60))
        
        # 构造旧类原型 (从测试集中取1张图作为代表)
        old_prototypes = []
        for c in other_classes:
            sup, _ = self.get_support_and_query(test_set, c, n_shot=1, n_query=1)
            with torch.no_grad():
                feat = encoder(sup).cpu().mean(dim=0)
            old_prototypes.append(feat)
        old_prototypes = torch.stack(old_prototypes) if old_prototypes else None
        
        # 获取当前类的支撑集和查询集
        support_imgs, query_imgs = self.get_support_and_query(test_set, class_id, n_shot, n_query)
        
        # 随机决定是否添加污染 (以0.3概率)
        add_corruption = random.random() < 0.3
        if add_corruption:
            corr_type = random.choice(['blur', 'duplicate'])
            support_dirty = support_imgs.clone()
            if corr_type == 'blur':
                from PIL import ImageFilter
                img_pil = transforms.ToPILImage()(support_dirty[0].cpu())
                blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=3))
                blurred_t = transforms.ToTensor()(blurred).cuda()
                blurred_t = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(blurred_t)
                support_dirty[0] = blurred_t
            elif corr_type == 'duplicate':
                for i in range(1, 4):
                    support_dirty[i] = support_imgs[0].clone() + 0.01 * torch.randn_like(support_imgs[0])
            support_imgs = support_dirty
        
        # 提取特征
        with torch.no_grad():
            support_feats = encoder(support_imgs).cpu()
        
        # 提取风险特征
        g = self.extract_features(support_feats, old_prototypes)
        
        # 评估标签：用普通平均构造原型，看新类查询集准确率
        new_acc = self.evaluate_prototype(encoder, test_set, class_id, support_imgs, query_imgs, old_prototypes)
        
        # ===== 相对性能标签 =====
        if old_prototypes is not None and len(old_prototypes) > 0:
            random_baseline = 1.0 / (len(old_prototypes) + 1)
        else:
            random_baseline = 1.0
        
        if new_acc > 2.0 * random_baseline:
            label = 1
        else:
            label = 0
        
        return g.numpy(), label
    
    def train_on_pseudo_episodes(self, encoder, test_set, n_episodes=300, n_shot=5, n_query=20):
        """
        在基类数据上构造伪增量场景，训练RCA分类器
        """
        print(f"开始构造 {n_episodes} 个伪增量场景...")
        X = []
        y = []
        
        for i in range(n_episodes):
            g, label = self.generate_pseudo_episode(encoder, test_set, n_shot, n_query)
            X.append(g)
            y.append(label)
            if (i+1) % 50 == 0:
                print(f"  已构造 {i+1}/{n_episodes} 个场景")
        
        X = np.array(X)
        y = np.array(y)
        
        # 标准化
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 训练逻辑回归
        self.classifier.fit(X_scaled, y)
        
        # 在训练集上计算最优阈值 (最大化F1)
        probs = self.classifier.predict_proba(X_scaled)[:, 1]
        best_f1 = 0
        best_threshold = 0.5
        for thr in np.arange(0.3, 0.8, 0.05):
            preds = (probs >= thr).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            if tp + fp == 0 or tp + fn == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr
        
        self.threshold = best_threshold
        self.is_trained = True
        
        # 打印训练统计
        train_acc = (self.classifier.predict(X_scaled) == y).mean()
        print(f"\n训练完成:")
        print(f"  训练集准确率: {train_acc:.3f}")
        print(f"  正例比例: {y.mean():.3f}")
        print(f"  最优阈值: {self.threshold:.3f}")
        print(f"  最优F1: {best_f1:.3f}")
        
        # 特征重要性
        if hasattr(self.classifier, 'coef_'):
            print(f"  特征权重: {dict(zip(self.feature_names, self.classifier.coef_[0]))}")
        
        return self
    
    def predict(self, support_features, existing_prototypes):
        """判断当前支撑集是否应该被接受为新类别"""
        if not self.is_trained:
            print("警告: RCA模型尚未训练，默认接受")
            return True, 0.5
        
        g = self.extract_features(support_features, existing_prototypes)
        g_scaled = self.scaler.transform(g.unsqueeze(0).numpy())
        prob = self.classifier.predict_proba(g_scaled)[0, 1]
        
        if self.threshold is None:
            self.threshold = 0.5
        
        return prob >= self.threshold, prob
    
    def save(self, path="rca_model.pkl"):
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'classifier': self.classifier,
                'threshold': self.threshold,
                'feature_dim': self.feature_dim,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names
            }, f)
        print(f"RCA模型已保存到 {path}")
    
    def load(self, path="rca_model.pkl"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.classifier = data['classifier']
        self.threshold = data['threshold']
        self.feature_dim = data['feature_dim']
        self.is_trained = data['is_trained']
        self.feature_names = data.get('feature_names', self.feature_names)
        print(f"RCA模型已从 {path} 加载")


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("RCA模块训练测试 (使用真实CIFAR-100数据)")
    print("=" * 60)
    
    from models.protonet import PrototypicalNetwork
    
    # 加载模型
    model = PrototypicalNetwork().cuda()
    model.load_state_dict(torch.load("checkpoints/protonet_resnet18.pth"))
    model.eval()
    encoder = model.encoder
    
    # 加载CIFAR-100测试集
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=None
    )
    
    # 初始化RCA
    rca = RiskCalibratedAdmission()
    
    # 训练 (300个伪场景)
    rca.train_on_pseudo_episodes(encoder, test_set, n_episodes=300)
    
    # 保存模型
    rca.save("rca_model.pkl")
    
    print("\n模块运行正常 ✅")
