# Cross-Modal Alignment of Local and Global Features for Zero-Shot Chinese Character Recognition

本仓库是 ICIP 论文 **Cross-Modal Alignment of Local and Global Features for Zero-Shot Chinese Character Recognition** 的代码。

图像数据集和模型权重未包含在本仓库中。部首分解与字符表保留在 `data/IDS/` 和 `char_dict`。使用前请自行准备训练与测试图像，并修改 `cfgs/` 中的数据路径。

## 环境

```bash
conda create -n ICIP-CCR python=3.7
conda activate ICIP-CCR
pip install torch torchvision pillow lmdb numpy matplotlib tqdm
```

## 训练

```bash
python pretrain_main.py
```

配置文件为 `cfgs/fdtv2_config.py`。

## 测试与可视化

```bash
python val_fdt.py
python visualization.py
```
