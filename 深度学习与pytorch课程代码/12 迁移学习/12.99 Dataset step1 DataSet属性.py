from torch.utils.data  import Dataset, DataLoader

class NumberDataSet(Dataset):
    def __init__(self,train=True):
        if train :
            self.ds = list(range(1,1000))
        else :
            self.ds = list(range(1001,1500))
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        return self.ds[item]
def main():
    ds = NumberDataSet()
    print(ds.__getitem__(1),ds.__len__())
if __name__ == '__main__':
    main()

