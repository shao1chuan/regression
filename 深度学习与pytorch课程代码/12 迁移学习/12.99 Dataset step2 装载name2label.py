# 重要语法
# 1 filenames = os.listdir(root)  取得目录下文件夹名
from torch.utils.data  import Dataset, DataLoader
import os

class ImageData(Dataset):
    def __init__(self,root,resize,mode):
        # root 路径，  resize 图片大小， mode 训练模式
        super(ImageData,self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.name2label = {}
        #  "name ":0
        filenames = os.listdir(root)
        print(root)
        for idx, name in sorted(enumerate(filenames)):
            self.name2label[name] = idx
        print(self.name2label)
    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
def main():

    root = r"../../../use/data/pokemon/"

    ds = ImageData(root,"224","train")
    # print(ds.__getitem__(1),ds.__len__())
if __name__ == '__main__':
    main()

