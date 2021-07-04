# 重要语法
# 1 filenames = os.listdir(root)  取得目录下文件夹名
from torch.utils.data  import Dataset, DataLoader
import os,glob

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
            # 去掉文件 只保留文件夹
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name] = idx
        print(self.name2label)
    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def load_csv(self,filename):
        images = []
        for name in self.name2label.keys():
            print(self.root)
            images+=glob.glob(os.path.join(self.root,name,"*.png"))
            images += glob.glob(os.path.join(self.root, name, "*.jpg"))
            images += glob.glob(os.path.join(self.root, name, "*.jpeg"))
        print(len(images),images)
        # 1167['pokemon\\data\\bulbasaur\\00000000.png',
def main():

    ds = ImageData(r"pokemon\data","224","train")
    ds.load_csv("a")
if __name__ == '__main__':  
    main()

