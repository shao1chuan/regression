import glob
import os

def main():
    root = 'pokemon'
    filetype = '*.py'
    print(os.path.join(root, filetype))
    filename = glob.glob(os.path.join(root, filetype))
    print(filename)
    # 获取上级目录的所有.py文件
    print(glob.glob(r'../*.py'))  # 相对路径


if __name__ == '__main__':
    main()