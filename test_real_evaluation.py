import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from incremental import IncrementalLearner

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

learner = IncrementalLearner()

support_dir = '/home/didi/evobench_real/support'
categories = sorted(os.listdir(support_dir))

print("添加真实类别...")
for cat in categories:
    cat_path = os.path.join(support_dir, cat)
    if not os.path.isdir(cat_path):
        continue
    imgs = os.listdir(cat_path)
    tensors = []
    for img_name in imgs:
        img = Image.open(os.path.join(cat_path, img_name)).convert('RGB')
        tensors.append(transform(img))
    support_tensor = torch.stack(tensors).cuda()
    learner.add_class(support_tensor, cat)

print(f"\n已添加 {learner.get_class_count() - 60} 个真实类别")

test_dir = '/home/didi/evobench_real/test'
correct = 0
total = 0

print("\n测试真实图片...")
for cat in categories:
    cat_path = os.path.join(test_dir, cat)
    if not os.path.isdir(cat_path):
        continue
    imgs = os.listdir(cat_path)
    for img_name in imgs:
        img = Image.open(os.path.join(cat_path, img_name)).convert('RGB')
        img_t = transform(img).unsqueeze(0).cuda()
        pred, conf = learner.predict(img_t)
        if pred == cat:
            correct += 1
        total += 1

print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")
