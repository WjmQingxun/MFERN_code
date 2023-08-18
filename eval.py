# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data

import numpy as np
import seaborn as sns
import imageio
import matplotlib.pyplot as plt
import os
from utils import metrics, show_results
from datasets import get_dataset, DATASETS_CONFIG
from models.get_model import get_model
from train import test

from sklearn.metrics import confusion_matrix

save_flg = True
import argparse

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='PaviaU', choices=dataset_names,
                    help="PaviaU, IndianPines, Salinas")
parser.add_argument('--model', type=str, default='SSRN',
                    help="Model to train. Available:\n"
                         "SSRN, DFFN, ResNet, pResNet, ASKResNet"
                         "pResNet, HybridSN, RSSAN, SSTN, DCRN, MFERN"
                    )
parser.add_argument('--folder', type=str, help="Folder where to store the datasets.",
                    default="./Datasets/")
parser.add_argument('--patch_size', type=int, default=7,
                    help="Input patch size")
parser.add_argument('--cuda', type=str, default='0',
                    help="Specify CUDA device")
parser.add_argument('--weights', type=str, default='./checkpoints/SSRN/PaviaU/0',
                    help="Folder to the weights used for evaluation")
parser.add_argument('--output', type=str, default='./results',
                    help="Folder to store results")

args = parser.parse_args()

if int(args.cuda) < 0:
    print("Computation on CPU")
    device = torch.device('cpu')
elif torch.cuda.is_available():
    print("Computation on CUDA GPU device {}".format(args.cuda))
    device = torch.device('cuda:{}'.format(args.cuda))
else:
    print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
    device = torch.device('cpu')

# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Weights to evaluate
WEIGHTS = args.weights
# Folder to restore results
OUTPUT = args.output
# Patch size
PATCH_SIZE = args.patch_size
#Batch size
BATCH_SIZE = 128

print('Dataset: %s' % DATASET)
print('patch size: %d' % PATCH_SIZE)
print('Model: %s' % (MODEL))

# Load the dataset
img, gt, LABEL_VALUES = get_dataset(DATASET,FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Generate color palette
palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", N_CLASSES+1)):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d

#load model and weights
model = get_model(MODEL, args.dataset, N_CLASSES, N_BANDS, PATCH_SIZE)
print('Loading weights from %s' % WEIGHTS + '/model_best.pth')
model = model.to(device)
model.load_state_dict(torch.load(WEIGHTS + '/model_best.pth'))
model.eval()

#testing model
probabilities = test(model, WEIGHTS, img, PATCH_SIZE, N_CLASSES, device=device)
prediction = np.argmax(probabilities, axis=-1)

run_results = metrics(prediction, gt, n_classes=N_CLASSES)

prediction[gt < 0] = -1

#color results
colored_gt = color_results(gt+1, palette)
colored_pred = color_results(prediction+1, palette)

outfile = os.path.join(OUTPUT, DATASET, MODEL)
os.makedirs(outfile, exist_ok=True)

imageio.imsave(os.path.join(outfile, DATASET+'_gt.png'), colored_gt)
imageio.imsave(os.path.join(outfile, DATASET+'_'+MODEL+'_out.png'), colored_pred)

confusion = show_results(run_results, label_values=LABEL_VALUES)

#绘制混淆矩阵
plt.figure(figsize=(18, 17))  # 设置图片大小


# 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
plt.colorbar()  # 右边的colorbar

# 2.设置坐标轴显示列表
indices = range(len(confusion))
dic = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
labelnum = len(LABEL_VALUES)

lt = [dic[i] for i in range(labelnum)]
classes = lt
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, classes,fontsize=18)  # 设置横坐标方向，rotation=45为45度倾斜
plt.yticks(indices, classes,fontsize=18)

# 3.设置全局字体
# 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
# ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 4.设置坐标轴标题、字体
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.title('Confusion matrix', fontsize=18)


# 5.显示数据
normalize = False
fmt = '.2f' if normalize else 'd'
thresh = confusion.max() / 2.

for i in range(len(confusion)):  # 第几行
    for j in range(len(confusion[i])):  # 第几列
        plt.text(j, i, format(confusion[i][j], fmt),
                 fontsize=17,  # 矩阵字体大小
                 horizontalalignment="center",  # 水平居中。
                 verticalalignment="center",  # 垂直居中。
                 color="white" if confusion[i, j] > thresh else "black")

# 6.保存图片
if save_flg:
    dress = "./picture/" + args.dataset + "/" + args.model
    if not os.path.isdir(dress):
        os.makedirs(dress, exist_ok=True)
    place = dress + "/confusion_matrix.png"
    plt.savefig(place, dpi=800, bbox_inches = 'tight')

# 7.显示
plt.show()


del model


