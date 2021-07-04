import csv
import os
def writecsv(name):
    print("write")
    with open(name, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(['a1ab', "1"])
        writer.writerow(['a2ab', "2"])

def readcsv(name):
    print("read")
    with open(name, mode="r", newline="") as f:
        reader = csv.reader(f)
        img1,label1,img2,label2 = [],[],[],[]
        for row in reader:
            a1,a2=row
            print(a1,a2)
            img1.append(a1)
            label1.append(a2)
            img2+=a1
            label2+=a2
        print(img1,label1,img2,label2)

def main():
    name ="test.csv"
    if not os.path.exists(os.path.join(name)):
        writecsv(name)
    else :
        readcsv(name)
if __name__ == '__main__':
    main()