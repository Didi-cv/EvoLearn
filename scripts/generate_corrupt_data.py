"""
EvoBench-Corrupt: 非理想支撑集污染数据生成脚本
用于生成模糊、错标、重复三类污染的CIFAR-100支撑集
iCAN 2026 · EvoLearn
"""

import os
import pickle
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

# ==================== 配置参数 ====================
DATA_ROOT = "./data"          # CIFAR-100 下载目录
SAVE_ROOT = "./corrupted_data" # 污染数据保存目录
NUM_SEEDS = 3                 # 随机种子数（3个种子）
NUM_WAY = 5                   # 每阶段5个新类
NUM_SHOT = 5                  # 每类5张支撑样本

# 污染配置
BLUR_SIGMAS = [1.0, 2.0, 3.0]          # 模糊程度
LABEL_FLIP_COUNTS = [1, 2, 3]          # 错标张数（5张中错几张）
DUPLICATE_COUNTS = [2, 3, 4]           # 重复张数（5张中重复几张）

# ==================== 工具函数 ====================

def apply_gaussian_blur(img_np, sigma):
    """
    对 numpy 格式的图片施加高斯模糊（HWC, 0-255）
    """
    blurred = np.zeros_like(img_np)
    for c in range(img_np.shape[2]):
        blurred[:, :, c] = gaussian_filter(img_np[:, :, c], sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def corrupt_support_set(images, labels, corrupt_type, severity, seed):
    """
    对一组支撑样本施加污染

    参数:
        images: list of np.array, 每张图 HWC uint8
        labels: list of int, 对应标签
        corrupt_type: 'blur' | 'label_flip' | 'duplicate'
        severity: 强度（int 或 float）
        seed: 随机种子

    返回:
        corrupted_images, corrupted_labels
    """
    np.random.seed(seed)
    n = len(images)
    corrupted_imgs = list(images)
    corrupted_lbls = list(labels)

    if corrupt_type == 'blur':
        # 模糊污染：对前 severity 张图片加模糊
        sigma = float(severity)
        for i in range(min(n, 2)):  # 对前2张加模糊，模拟轻度污染
            corrupted_imgs[i] = apply_gaussian_blur(images[i], sigma)

    elif corrupt_type == 'label_flip':
        # 错标污染：将 severity 张图片的标签改成随机错误类别
        flip_count = int(severity)
        # 选择要翻转的索引（在5张中选severity张）
        flip_indices = np.random.choice(n, size=flip_count, replace=False)
        for idx in flip_indices:
            # 随机选一个不同于原标签的类别（简单起见，取原标签+5，绕回）
            # 实际使用时可以更精细
            original_label = labels[idx]
            new_label = (original_label + np.random.randint(1, 10)) % 100
            corrupted_lbls[idx] = new_label

    elif corrupt_type == 'duplicate':
        # 重复污染：将 severity 张图片替换为第0张的副本（加微小噪声）
        dup_count = int(severity)
        original_img = images[0]
        for i in range(1, min(dup_count + 1, n)):
            # 加一点小噪声，避免完全像素相同（更真实）
            noise = np.random.randint(0, 10, size=original_img.shape, dtype=np.uint8)
            corrupted_imgs[i] = np.clip(original_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return corrupted_imgs, corrupted_lbls


# ==================== 主流程 ====================

def main():
    print("=" * 50)
    print("EvoBench-Corrupt 污染数据生成器")
    print("=" * 50)

    # 1. 加载 CIFAR-100 数据集
    print("\n[1] 加载 CIFAR-100 数据...")
    transform = transforms.ToTensor()
    train_set = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform)
    
    # 将数据转为 numpy 格式（方便处理）
    all_images_np = []
    all_labels = []
    for idx in range(len(train_set)):
        img_tensor, label = train_set[idx]
        # 转换为 HWC uint8 (0-255)
        img_np = (img_tensor.numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
        all_images_np.append(img_np)
        all_labels.append(label)
    
    # 按类别组织
    class_to_indices = {}
    for idx, lbl in enumerate(all_labels):
        class_to_indices.setdefault(lbl, []).append(idx)

    print(f"  加载完成，共 {len(all_images_np)} 张图片，{len(class_to_indices)} 个类别")

    # 2. 生成污染配置
    print("\n[2] 生成污染配置...")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # 只取前60类作为基类（FSCIL标准协议），后40类作为增量
    base_classes = list(range(60))
    # 对每个污染配置生成数据
    configs = []

    # 模糊
    for sigma in BLUR_SIGMAS:
        configs.append(('blur', sigma, f'blur_sigma_{sigma}'))
    # 错标
    for cnt in LABEL_FLIP_COUNTS:
        configs.append(('label_flip', cnt, f'label_flip_{cnt}of5'))
    # 重复
    for cnt in DUPLICATE_COUNTS:
        configs.append(('duplicate', cnt, f'duplicate_{cnt}of5'))

    # 额外加一个干净对照组
    configs.append(('clean', 0, 'clean'))

    # 3. 为每个配置生成数据
    print("\n[3] 生成污染数据...")
    results = {}

    for corrupt_type, severity, name in configs:
        print(f"  处理: {name}")
        seed_results = []
        
        for seed in range(NUM_SEEDS):
            # 为每个种子随机选择5个基类，每个基类取5张图片
            # 这里简化：固定取前5个类别，每类前5张
            class_sample = base_classes[:NUM_WAY]
            sample_images = []
            sample_labels = []
            
            for cls_id in class_sample:
                indices = class_to_indices[cls_id][:NUM_SHOT]
                for idx in indices:
                    sample_images.append(all_images_np[idx])
                    sample_labels.append(cls_id)
            
            # 应用污染（如果配置不是clean）
            if corrupt_type != 'clean':
                corrupted_imgs, corrupted_lbls = corrupt_support_set(
                    sample_images, sample_labels, corrupt_type, severity, seed
                )
            else:
                corrupted_imgs, corrupted_lbls = sample_images, sample_labels
            
            # 保存这个seed的结果
            seed_results.append({
                'images': corrupted_imgs,
                'labels': corrupted_lbls,
                'class_ids': class_sample,
                'seed': seed
            })
        
        results[name] = seed_results

    # 4. 保存到文件
    print("\n[4] 保存数据...")
    for name, data in results.items():
        save_path = os.path.join(SAVE_ROOT, f"{name}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  已保存: {save_path}")

    # 保存配置清单
    config_summary = {
        'num_seeds': NUM_SEEDS,
        'num_way': NUM_WAY,
        'num_shot': NUM_SHOT,
        'configs': configs,
        'base_classes': base_classes
    }
    with open(os.path.join(SAVE_ROOT, "config_summary.pkl"), 'wb') as f:
        pickle.dump(config_summary, f)

    print("\n" + "=" * 50)
    print("✅ 所有污染数据生成完成！")
    print(f"   数据保存在: {SAVE_ROOT}")
    print("=" * 50)


if __name__ == "__main__":
    main()