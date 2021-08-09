import glob
import os

def main():
    root = 'glob'
    filetype = '*.py'
    print(os.path.join(root, filetype))
    filename = glob.glob(os.path.join(root, filetype))
    print(f"filename is {filename}")
    # 获取上级目录的所有.py文件
    print(f"glob.glob(r'../4/*.py') is {glob.glob(r'../4/*.py')}")  # 相对路径


if __name__ == '__main__':
    main()