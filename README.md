# ML_Foundations
2020 Winter Machine Learning Foundations Experiments
Final Project

#### 选题：
图像去噪与重建

#### 要求：
	现有一批不同程度噪声影响的MRI测试数据噪声影响程度分别为3%、5%、7%、9%，希望开发一个机器学习算法，能最大程度的去除MRI图像的噪声并重建出尽可能真实的图像。

#### 评估指标：
	PSNR(Peak Signal to Noise Ratio)峰值信噪比和SSIM(Structural Similarity Index)结构相似性。

#### 数据文件：
	共有5个子文件夹NoNoise、NoiseLevel3、NoiseLevel5、NoiseLevel7、NoiseLevel9，分别表示无噪声影响的和不同噪声程度影响的MRI图像，每个子文件夹下包含15 张MRI图像。

#### 进展：
任务状态![status](https://img.shields.io/badge/status-working(1%2F4)-yellow)

- [x] 环境配置
- [ ] 初级算法构建（算法去噪）
- [ ] CNN卷积神经网络构建
- [ ] 数据对比、书写论文

#### 环境配置：
Anaconda3 (Python 3.8.x)
OpenCV
