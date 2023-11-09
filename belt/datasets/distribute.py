import os
import shutil
import random

# 设置数据集根目录和各子目录
dataset_root = "./"
images_root = os.path.join(dataset_root, "images")
labels_root = os.path.join(dataset_root, "labels")

# 创建train、val和test目录
split_folders = ["train", "val", "test"]
for folder in split_folders:
    split_dir = os.path.join(images_root, folder)
    os.makedirs(split_dir, exist_ok=True)

# 获取所有图像文件和标签文件
image_files = os.listdir(os.path.join(images_root, "all"))
label_files = os.listdir(os.path.join(labels_root, "all"))

# 随机打乱图像和标签的顺序，但确保它们一一对应
random.seed(42)  # 设置随机种子以确保可重复性
random.shuffle(image_files)
random.shuffle(label_files)

# 计算分配的数量
total_files = len(image_files)
train_size = int(total_files * 0.6)  # 60%
val_size = int(total_files * 0.2)    # 20%
test_size = total_files - train_size - val_size

# 分配图像和标签到train、val和test目录
for i in range(total_files):
    image_filename = image_files[i]
    label_filename = image_filename.replace(".jpg", ".txt")  # 根据图像文件名构建标签文件名
    if i < train_size:
        folder = "train"
    elif i < train_size + val_size:
        folder = "val"
    else:
        folder = "test"
    
    # 复制图像文件
    source_image_path = os.path.join(images_root, "all", image_filename)
    target_image_path = os.path.join(images_root, folder, image_filename)
    shutil.copy(source_image_path, target_image_path)

    # 复制标签文件
    source_label_path = os.path.join(labels_root, "all", label_filename)
    target_label_path = os.path.join(labels_root, folder, label_filename)
    shutil.copy(source_label_path, target_label_path)

print("Distribution Complete")
