import os
def main():
    path1 = "1"
    path2 = "2"
    path3 = "3"
    path = os.path.join(path1,path2,path3)
    print(path)
if __name__ == '__main__':
    main()