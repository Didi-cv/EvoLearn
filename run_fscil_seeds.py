import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
from incremental import IncrementalLearner

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

BASE_CLASSES = list(range(60))
SESSION_CLASSES = [
    list(range(60, 65)), list(range(65, 70)), list(range(70, 75)),
    list(range(75, 80)), list(range(80, 85)), list(range(85, 90)),
    list(range(90, 95)), list(range(95, 100))
]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_support_images(test_set, class_id, n_shot=5):
    targets = torch.tensor(test_set.targets)
    idx = (targets == class_id).nonzero().squeeze()
    if idx.numel() < n_shot:
        n_shot = idx.numel()
    perm = idx[torch.randperm(len(idx))][:n_shot]
    images = torch.stack([transform(test_set.data[i]) for i in perm])
    return images

def evaluate_all_seen(learner, test_set, all_classes, n_query=20):
    correct = 0
    total = 0
    targets = torch.tensor(test_set.targets)
    if len(learner.prototypes) == 0:
        return 0.0
    for c in all_classes:
        idx = (targets == c).nonzero().squeeze()
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(len(idx))][:n_query]
        for i in perm:
            img = transform(test_set.data[i]).cuda().unsqueeze(0)
            with torch.no_grad():
                emb = learner.encoder(img).cpu()
            prototypes = torch.stack(learner.prototypes)
            distances = torch.cdist(emb, prototypes)
            _, pred_idx = distances.min(dim=1)
            total += 1
            if learner.labels[pred_idx.item()] == str(c):
                correct += 1
    return correct / total if total > 0 else 0.0

def run_fscil_benchmark(seed):
    set_seed(seed)
    print(f"\n========== 随机种子: {seed} ==========")
    
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=None
    )

    learner = IncrementalLearner()
    learner.encoder.cuda()

    all_seen = []
    accuracies = []

    for session_idx, new_classes in enumerate(SESSION_CLASSES, 1):
        for c in new_classes:
            images = get_support_images(test_set, c, n_shot=5)
            learner.add_class(images, str(c))
            all_seen.append(c)
        acc = evaluate_all_seen(learner, test_set, BASE_CLASSES + all_seen)
        accuracies.append(acc)
        print(f"阶段{session_idx}: {acc:.4f}")

    final_acc = accuracies[-1]
    print(f"种子{seed} 最终准确率: {final_acc:.4f}")
    return final_acc, accuracies

if __name__ == "__main__":
    seeds = [1, 2, 3]
    results = []
    for s in seeds:
        final, _ = run_fscil_benchmark(s)
        results.append(final)
    
    print("\n" + "="*40)
    print("3次运行统计:")
    print(f"种子 {seeds}: {[f'{r:.4f}' for r in results]}")
    print(f"平均值: {sum(results)/len(results):.4f}")
    print(f"最大值: {max(results):.4f}, 最小值: {min(results):.4f}")
    print(f"差值: {max(results) - min(results):.4f}")
    print("="*40)
