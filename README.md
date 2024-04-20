# MM 805 Project Winter 2024

## Group Member
- Mingwei Lu: mlu1
- Dulong Sang: dulong

## Dataset
[Garbage Classification -- Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data)

The Garbage Classification Dataset contains 6 classifications: cardboard (393), glass (491), metal (400), paper (584), plastic (472) and trash (127).

## Requirements
- `python` 3.9 or above
- CPU or NVIDIA GPU + CUDA CuDNN

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Models

### VGGNet16 (Very Deep Convolutional Networks)
```bash
python3 VGG16.py [--batch_size=64] [--learning_rate=1e-4] [--num_epochs=20] [--dataset_dir='dataset/Garbage classification'] [--model_save_path='vgg16.pth'] [--no-cuda]
```

### SVM (Support Vector Machine)
```bash
python3 SVM.py [--dataset_dir='dataset/Garbage classification'] [--model_save_path='svm.pkl']
```

## Windows 10/11 CUDA Support Installation
Guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

- Install CUDA Toolkit:
https://developer.nvidia.com/cuda-downloads

- Install CUDA compiled torch and torchvision:
https://pytorch.org/get-started/locally/
