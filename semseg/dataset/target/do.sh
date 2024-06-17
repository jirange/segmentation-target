#!/bin/bash

# 将下面的路径替换为你的txt文件和目标文件夹的路径
input_file="/dataset/vsitongwu/LJC/TargetSeg/semseg/dataset/target/list/val.txt"
destination_folder="/dataset/vsitongwu/LJC/TargetSeg/test_result"

# 确保目标文件夹存在
mkdir -p "$destination_folder"

# 读取文件并复制图片
while IFS=' ' read -r source_path1 source_path2; do
    # 假设每行的格式是 "source_path1 source_path2"
    # 根据空格分割每行，并将结果赋值给source_path1和source_path2

    # 复制文件
    cp "$source_path1" "$destination_folder"
    # cp "$source_path2" "$destination_folder"
done < "$input_file"