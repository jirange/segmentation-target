import os
import glob
import numpy as np

def create_trainval(dataset_dir):
    # 设置据集目录
    images_dir = 'image/'
    labels_dir = 'label-1/'
    # 要生成的trainval.txt文件的路径
    output_file = os.path.join(dataset_dir, 'list/trainval.txt')

    # 获取所有bmp文件的路径 in
    bmp_files = glob.glob(os.path.join(dataset_dir, images_dir, '*.bmp'))
    # 创建或覆盖train.txt文件
    with open(output_file, 'w') as f:
        for bmp_file in bmp_files:
            # 构造对应的png文件名
            png_file = bmp_file.replace('.bmp', '.png')
            png_file = png_file.replace(images_dir, labels_dir)
            # 确保对应的png文件存在
            if os.path.isfile(png_file):
                # 将bmp和png文件的相对路径写入train.txt
                f.write(f"{os.path.relpath(bmp_file, dataset_dir)} {os.path.relpath(png_file, dataset_dir)}\n")
            else:
                print(f"Corresponding PNG file not found for {bmp_file}")

    print(f"{output_file} has been created.")

def split_train_val(dataset_dir):
    train_path =os.path.join(dataset_dir, 'list/train.txt')
    val_path = os.path.join(dataset_dir, 'list/val.txt')
    trainval_path =os.path.join(dataset_dir, 'list/trainval.txt')
    # 读取原始train.txt文件
    with open(trainval_path, 'r') as file:
        lines = file.readlines()

    # 随机打乱数据行顺序
    np.random.shuffle(lines)

    # 确定分割点，80% train之后再进行数据增广
    split_ratio = 0.8
    split_index = int(len(lines) * split_ratio)

    # 分割数据为训练集和验证集
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]

    # 写入新的train.txt和val.txt文件
    with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
        for line in train_lines:
            f_train.write(line)  # 写入训练集
        for line in val_lines:
            f_val.write(line)  # 写入验证集
    print(f"{train_path} {val_path}has been created.")

if __name__ == "__main__":
    dataset_dir = 'semseg/dataset/target/'
    np.random.seed(7)
    create_trainval(dataset_dir)
    split_train_val(dataset_dir)