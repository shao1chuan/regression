import os
import glob,random
import torch.utils.data.dataset as ds
import torch
import csv
from    torchvision import transforms
from    PIL import Image

class Pokemon(ds.Dataset):
    def __init__(self,root):
        super(Pokemon,self).__init__()
        self.name2lable = {}
        self.root = root

        for name in os.listdir(root):
            if not os.path.isdir(os.path.join(root,name)):
                # print(os.path.join(root,name))
                continue
            self.name2lable[name] = len(self.name2lable.keys())
        # print(self.name2lable)
        # image, label
        self.images, self.labels = self.load_csv('myimages.csv')


    def __getitem__(self,idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'data\\bulbasaur\\00000000.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]

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

        return img, label

    def __len__(self):
        return len(self.images)

    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images,labels = [],[]
            for name in self.name2lable.keys():
                images += glob.glob(os.path.join(self.root,name,"*.jpg"))
                images += glob.glob(os.path.join(self.root, name, "*.png"))
                images += glob.glob(os.path.join(self.root, name, "*.jpeg"))
            # print(images)
            random.shuffle(images)
            with open(os.path.join(self.root,filename),mode = 'w',newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2lable[name]
                    images.append(img)
                    labels.append(label)
                    writer.writerow([img,label])
        else:
            images,labels = [],[]
            with open(os.path.join(self.root, filename), mode='r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    # data\mewtwo\00000237.png, 2
                    img,label = row
                    print(img,label)
                    images.append(img)
                    labels.append(label)
        assert len(images) == len(labels)
        return images,labels


def main():
    pp = Pokemon('data')
    images,labels = pp.load_csv("myimages.csv")
if __name__ == '__main__':
    main()