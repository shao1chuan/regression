
# python -m visdom.server
import torch
from torch.utils.data  import Dataset, DataLoader
import os,glob,csv,random
from torchvision import transforms
from PIL import Image
import  visdom
import  time


from ImageDatastep5 import ImageData
def main():
    ds = ImageData(r"pokemon\data",64,"train")
    x,y = next(iter(ds))
    print(x.shape,y.shape)

    viz = visdom.Visdom()
    viz.image(ds.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=8)

    for x,y in loader:
        viz.images(ds.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)

if __name__ == '__main__':
    main()

