import os

def main():
    root = "pokemon/data"
    filenames = os.listdir(root)
    print(f"filenames is :{filenames}")
    for name in filenames:
        if os.path.isdir(os.path.join(root,name)):
            print(f"dir is :{name}")
        if os.path.isfile(os.path.join(root,name)):
            print(f"file is :{name}")
if __name__ == '__main__':
    main()