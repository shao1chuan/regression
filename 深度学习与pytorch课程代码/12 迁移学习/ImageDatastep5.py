# 重要语法
# 1 filenames = os.listdir(root)  取得目录下文件夹名
import torch
from torch.utils.data  import Dataset, DataLoader
import os,glob,csv,random
from torchvision import transforms
from PIL import Image

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
        self.images,self.labels = self.load_csv('images.csv')
        if mode == 'train':
            self.images,self.labels = self.images[:int(0.6*len(self.images))],self.labels[:int(0.6*len(self.images))]
        elif mode =='val':
            self.images, self.labels = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))], self.labels[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
        elif mode =='test':
            self.images, self.labels = self.images[int(0.8 * len(self.images)):],self.labels[int(0.8 * len(self.images)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 'pokemon\\data\\squirtle\\00000205.jpg' 变成文件
        img,label  = self.images[idx],self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img,label
    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 4, 4]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x
    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images+= glob.glob(os.path.join(self.root,name,'*.png'))
                images+= glob.glob(os.path.join(self.root,name,'*.jpg'))
                images+= glob.glob(os.path.join(self.root,name,'*.jpeg'))
            print(len(images),images)
            random.shuffle(images)

            with open(os.path.join(self.root, filename),mode="w",newline='') as f:
                    writer = csv.writer(f)
                    for image in images:
                        # 'pokemon\\data\\squirtle\\00000205.jpg'
                        name = image.split(os.sep)[-2]
                        label = self.name2label[name]
                        writer.writerow([image,label])
                    print('writen into csv file:', filename)
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images,labels
def main():
    ds = ImageData(r"pokemon\data","224","train")
    ds.load_csv(r"images.csv")
if __name__ == '__main__':
    main()

