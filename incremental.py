import torch
import torch.nn.functional as F
from models.protonet import PrototypicalNetwork
from diversity_prototype import diversity_aware_prototype
from risk_calibrated_admission import RiskCalibratedAdmission

class IncrementalLearner:
    def __init__(self, model_path="checkpoints/protonet_resnet18.pth", base_proto_path="checkpoints/base_prototypes_resnet.pth"):
        self.model = PrototypicalNetwork().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.encoder = self.model.encoder
        self.prototypes = []
        self.labels = []
        self.buffer = {}
        self.max_buffer_per_class = 20

        base_data = torch.load(base_proto_path)
        self.prototypes = base_data['prototypes']
        self.labels = base_data['labels']
        print(f"已加载 {len(self.prototypes)} 个基类原型 (ResNet-18)")

        # 边界约束参数
        self.margin = 0.4
        self.push_strength = 0.03

        # 加载RCA模型（延迟初始化，第一次使用时加载）
        self.rca = None

    def _load_rca(self):
        """懒加载RCA模型"""
        if self.rca is None:
            self.rca = RiskCalibratedAdmission()
            try:
                self.rca.load("rca_model.pkl")
                print("RCA模型加载成功")
            except FileNotFoundError:
                print("警告: rca_model.pkl 未找到，RCA将默认接受所有类别")
                self.rca.is_trained = False

    def extract_prototype(self, images):
        """保留旧方法备用，但不使用"""
        with torch.no_grad():
            embs = self.encoder(images)
            proto = embs.mean(dim=0)
            return proto

    def add_class(self, images, label_name, tau=0.1, tau_d=0.95):
        """
        使用 DRA 计算原型，集成RCA准入判断
        """
        if not images.is_cuda:
            images = images.cuda()

        # ---- 1. 提取所有图像的特征 ----
        with torch.no_grad():
            features = self.encoder(images)  # [K, D]

        # ---- 2. RCA 准入判断 ----
        self._load_rca()
        if self.rca is not None and self.rca.is_trained:
            existing_protos = torch.stack(self.prototypes).cpu() if len(self.prototypes) > 0 else None
            should_accept, confidence = self.rca.predict(features, existing_protos)

            if not should_accept:
                print(f"RCA拒绝学习 {label_name}: 置信度={confidence:.3f}, 阈值={self.rca.threshold:.3f}")
                return
            else:
                print(f"RCA通过: {label_name} (置信度={confidence:.3f})")
        else:
            print(f"RCA未启用，默认接受 {label_name}")

        # ---- 3. DRA 计算加权原型 ----
        prototype, weights = diversity_aware_prototype(features, tau=tau, tau_d=tau_d)
        prototype = prototype / prototype.norm()

        # ---- 4. 边界约束：推离最近旧类 ----
        if len(self.prototypes) > 0:
            prototypes_tensor = torch.stack(self.prototypes)
            distances = torch.cdist(prototype.cpu().unsqueeze(0), prototypes_tensor)
            min_dist, closest_idx = distances.min(dim=1)
            min_dist_val = min_dist.item()
            if min_dist_val < self.margin:
                closest_proto = prototypes_tensor[closest_idx.item()]
                push_dir = (prototype.cpu() - closest_proto) / (torch.norm(prototype.cpu() - closest_proto) + 1e-8)
                push_amount = (self.margin - min_dist_val) * self.push_strength
                prototype = prototype.cpu() + push_dir * push_amount
                prototype = prototype / prototype.norm()

        self.prototypes.append(prototype.cpu())
        self.labels.append(label_name)
        print(f"已添加新类别: {label_name} (DRA加权原型，基于 {images.shape[0]} 张图像)")

    def predict(self, image):
        if len(self.prototypes) == 0:
            return "无已学习类别", 0.0

        # ===== 修正：判断输入维度，如果已经是4维则不再unsqueeze =====
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.cuda()

        with torch.no_grad():
            emb = self.encoder(image).cpu()

        prototypes = torch.stack(self.prototypes)
        distances = torch.cdist(emb, prototypes)
        min_dist, idx = distances.min(dim=1)

        confidence = 1.0 / (1.0 + min_dist.item())
        return self.labels[idx.item()], confidence

    def get_class_count(self):
        return len(self.labels)
