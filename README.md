# EvoLearn

> 端侧自进化少样本学习引擎 · iCAN 2026 AI+前沿技术

## 一句话

让嵌入式设备通过5-15张现场照片安全地持续学习新类别，即便照片包含模糊、重复或错标，也能自主判断“能不能学、学哪些样本、怎样安全写入”。

## 核心能力

- **DRA**：多样性感知鲁棒原型聚合，在重复样本污染（2/5重复）下相比普通平均提升**5.10个百分点**（10种子平均）
- **RCA**：风险校准类别准入，将“用户提交即学习”升级为“评估→准入→写入”的可信流程，端侧推理<1ms
- **Rollback**：可回滚知识写入，支持一键撤销错误学习

## 核心数据（10种子平均）

| 指标 | 数值 |
|------|------|
| CIFAR-100 8阶段最终准确率 | **47.31%** |
| 性能下降（PD） | **13.42pp** |
| 平均准确率（AA） | **52.10%** |
| 端侧推理延迟 | **52ms/帧**（Jetson Nano 4GB） |
| 新类学习时间 | **~5秒**（5张照片） |
| 峰值内存 | **<2GB** |

> 基于10个随机种子，最终准确率标准差±1.05%

## 快速开始

```bash
# 环境要求：Python 3.8+, PyTorch 1.10+, torchvision 0.11+
git clone https://github.com/Didi-cv/EvoLearn.git
cd EvoLearn
pip install -r requirements.txt

# 运行标准FSCIL实验 (CIFAR-100, 8阶段, 5-shot, 10种子)
python run_fscil_seeds.py --seeds 10 --output ./results

# 生成EvoBench-Corrupt污染支撑集
python scripts/generate_corrupt_data.py

# 运行增量学习演示 (Jetson Nano)
python incremental.py
