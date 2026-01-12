# encoding: utf-8
import glob
import re
import os.path as osp
#from .bases import BaseImageDataset
import os
import glob
import random
from shutil import copy2
def split_ship_dataset(root_dir, output_dir, train_ratio = 0.6, query_ratio=0.1, seed=42, category='rgb'):
    random.seed(seed)

    # 获取所有 ship 文件夹
    train_root_dir = os.path.join(root_dir, 'train')
    test_root_dir = os.path.join(root_dir, 'test')
    if category  == 'rgb':
        train_root_dir = os.path.join(train_root_dir, 'rgb_crops')
        test_root_dir = os.path.join(test_root_dir, 'rgb_crops')
    elif category == 'sar':
        train_root_dir = os.path.join(train_root_dir, 'sar_crops')
        test_root_dir = os.path.join(test_root_dir, 'sar_crops')
    ship_train_dirs = [d for d in os.listdir(train_root_dir) if os.path.isdir(os.path.join(train_root_dir, d))]
    ship_test_dirs = [d for d in os.listdir(test_root_dir) if os.path.isdir(os.path.join(test_root_dir, d))]
    ship_train_dirs.sort()  # 保证 pid 一致性
    ship_test_dirs.sort()
    pid_mapping = {ship: pid for pid, ship in enumerate(ship_train_dirs)}

    train_set, query_set, gallery_set = [], [], []
    for ship in ship_train_dirs:
        pid = pid_mapping[ship]
        train_img_paths = glob.glob(os.path.join(train_root_dir, ship, '*.jpg'))
        test_img_paths = glob.glob(os.path.join(test_root_dir, ship, '*.jpg'))
        #print(train_img_paths)
        train_img_paths.sort()
        test_img_paths.sort()

        random.shuffle(train_img_paths)
        random.shuffle(test_img_paths)
        
        train_n_total = len(train_img_paths)
        test_n_total = len(test_img_paths)
        n_total = train_n_total + test_n_total
        #print(train_n_total)
        n_query = max(1, int(n_total * query_ratio))  # 至少 1 张 query
        n_train = int(n_total * train_ratio)

        train_imgs = train_img_paths[:n_train]

        train_imgs_paths_left = train_img_paths[n_train:]
        imgs_paths_left = train_imgs_paths_left + test_img_paths

        query_imgs = imgs_paths_left[:n_query]
        gallery_imgs = imgs_paths_left[n_query:]

        # 保存路径和标签 (camid 统一设 0)
        query_set.extend([(p, pid) for p in query_imgs])
        train_set.extend([(p, pid) for p in train_imgs])
        gallery_set.extend([(p, pid) for p in gallery_imgs])

    # 拷贝图片到新目录
    def copy_images(data, subset):
        subset_dir = os.path.join(output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        if category == 'rgb':
            camid = 0
        else:
            camid = 1
        for img_path, pid in data:
            # 文件名格式：pid_c0_xxx.jpg
            filename = f"{pid:02d}_c{camid}_{os.path.basename(img_path)}"
            dest_path = os.path.join(subset_dir, filename)
            copy2(img_path, dest_path)

    copy_images(train_set, 'bounding_box_train')
    copy_images(query_set, 'query')
    copy_images(gallery_set, 'bounding_box_test')

    print(" ReID dataset created at:", output_dir)
    print(f"Total:{len(train_set)+len(query_set)+len(gallery_set)}")
    print(f"Number of ID:{len(ship_train_dirs)}")
    print(f"Train: {len(train_set)} images")
    print(f"Query: {len(query_set)} images")
    print(f"Gallery: {len(gallery_set)} images")

# 使用示例
split_ship_dataset(
    root_dir='./todataset/market1501',         # 你的原始船只数据集
    output_dir='./todataset/ship_reid',  # 输出的新 ReID 数据集         # 50% 用于训练
    train_ratio=0.6,
    query_ratio=0.1,
    category='rgb'           # 10% 做 query
)
split_ship_dataset(
    root_dir='./todataset/market1501',         # 你的原始船只数据集
    output_dir='./todataset/ship_reid',  # 输出的新 ReID 数据集         # 50% 用于训练
    train_ratio=0.6,
    query_ratio=0.1,
    category='sar'           # 10% 做 query
)
