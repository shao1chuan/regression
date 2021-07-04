# 重要语法
# 1 filenames = os.listdir(root)  取得目录下文件夹名
import torch
from torch.utils.data  import Dataset, DataLoader
import os,glob,csv,random
from torchvision import transforms
from PIL import Image

from ImageDatastep5 import ImageData
def main():
    ds = ImageData(r"pokemon\data","224","train")
    x,y = next(iter(ds))
    print(x.shape,y.shape)
if __name__ == '__main__':
    main()

