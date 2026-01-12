from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as F
from train import GlobalConfig
def pad_to_square(img, fill=0):
    w, h = img.size
    max_wh = max(w, h)
    pad_left = 0
    pad_top = 0
    pad_right = max_wh - w
    pad_bottom = max_wh - h
    padding = (pad_left, pad_top, pad_right, pad_bottom)  # 左上右下
    img_padded = F.pad(img, padding, fill=fill)
    return img_padded
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
def extend_to_max_len(img_list, max_len):
    """
    扩展 img_list 到 max_len:
    - 先保留原始数据
    - 不足的部分用随机采样补齐
    """
    n = len(img_list)
    if n >= max_len:
        # 如果已经够长，直接打乱后返回前 max_len 个
        return random.sample(img_list, max_len)
    else:
        # 先保留原始数据
        ext_list = img_list.copy()
        # 再补齐剩余部分
        extra = [random.choice(img_list) for _ in range(max_len - n)]
        ext_list.extend(extra)
        # 打乱顺序
        random.shuffle(ext_list)
        return ext_list

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views
    def get_imagedata_info_pair(self, data):
        pids, cams, tracks = [], [], []

        for img in data:
            for _, pid, camid, trackid in img:#会循环两次，第一次解包rgb元组，第二次解包sar元组
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views
    def print_dataset_statistics(self):
        raise NotImplementedError

"""
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
   train    |    22 |     1416 |         2
   query    |    22 |      223 |         2
   gallery  |    22 |      726 |         2
  train_pair|    22 |     1037 |         2
   query_x  |    22 |      165 |         1
   gallery_x|    22 |      524 |         1
   query_y  |    22 |       58 |         1
   gallery_y|    22 |      202 |         1
  ----------------------------------------

"""
class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, train_x, train_y, query, gallery, query_x, gallery_x, query_y, gallery_y):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        #num_train_pair_pids, num_train_pair_imgs, num_train_pair_cams, num_train_pair_vids = self.get_imagedata_info_pair(train_pair)
        num_train_x_pids, num_train_x_imgs, num_train_x_cams, _ = self.get_imagedata_info(train_x)
        num_train_y_pids, num_train_y_imgs, num_train_y_cams, _ = self.get_imagedata_info(train_y)

        num_query_x_pids, num_query_x_imgs, num_query_x_cams, num_train_x_views = self.get_imagedata_info(query_x)
        num_gallery_x_pids, num_gallery_x_imgs, num_gallery_x_cams, num_train_x_views = self.get_imagedata_info(gallery_x)
        
        num_query_y_pids, num_query_y_imgs, num_query_y_cams, num_train_y_views = self.get_imagedata_info(query_y)
        num_gallery_y_pids, num_gallery_y_imgs, num_gallery_y_cams, num_train_y_views = self.get_imagedata_info(gallery_y)
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("   train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("   query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("   gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        #print("  train_pair| {:5d} | {:8d} | {:9d}".format(num_train_pair_pids, num_train_pair_imgs, num_train_pair_cams))
        print("   train_x    | {:5d} | {:8d} | {:9d}".format(num_train_x_pids, num_train_x_imgs, num_train_x_cams))
        print("   train_y    | {:5d} | {:8d} | {:9d}".format(num_train_y_pids, num_train_y_imgs, num_train_y_cams))
        print("   query_x  | {:5d} | {:8d} | {:9d}".format(num_query_x_pids, num_query_x_imgs, num_query_x_cams))
        print("   query_y  | {:5d} | {:8d} | {:9d}".format(num_query_y_pids, num_query_y_imgs, num_query_y_cams))
        print("   gallery_x| {:5d} | {:8d} | {:9d}".format(num_gallery_x_pids, num_gallery_x_imgs, num_gallery_x_cams))
        print("   gallery_y| {:5d} | {:8d} | {:9d}".format(num_gallery_y_pids, num_gallery_y_imgs, num_gallery_y_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]
    

class ImageDatasetPair(Dataset):
    def __init__(self, dataset_x, dataset_y, transform=None):
        """
        dataset_x: [(img_path, pid, camid, trackid), ...]  RGB
        dataset_y: [(img_path, pid, camid, trackid), ...]  SAR
        """
        self.transform = transform
        
        # 1. 按 pid 分类
        pid2rgb = {}
        pid2sar = {}
        for path, pid, camid, trackid in dataset_x:
            pid2rgb.setdefault(pid, []).append((path, pid, camid, trackid))
        for path, pid, camid, trackid in dataset_y:
            pid2sar.setdefault(pid, []).append((path, pid, camid, trackid))

        # 2. 扩展每个 pid 的数据到相同数量
        self.paired_data = []
        for pid in sorted(set(pid2rgb.keys()) & set(pid2sar.keys())):
            rgb_list = pid2rgb[pid]
            sar_list = pid2sar[pid]
            n_rgb, n_sar = len(rgb_list), len(sar_list)
            if n_rgb == 0 or n_sar == 0:
                continue

            max_len = max(80, n_rgb, n_sar)
            rgb_ext = extend_to_max_len(rgb_list, max_len)
            sar_ext = extend_to_max_len(sar_list, max_len)

            # 打乱顺序
            random.shuffle(rgb_ext)
            random.shuffle(sar_ext)

            # 一一配对
            for rgb_item, sar_item in zip(rgb_ext, sar_ext):
                rgb_path, pid1, camid1, trackid1 = rgb_item
                sar_path, pid2, camid2, trackid2 = sar_item
                self.paired_data.append((
                    (rgb_path, pid1, camid1, trackid1, rgb_path.split('/')[-1]),
                    (sar_path, pid2, camid2, trackid2, sar_path.split('/')[-1])
                ))

        # 最后再整体打乱
        random.shuffle(self.paired_data)
        

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, index):
        img1_info, img2_info = self.paired_data[index]
        img_path1, pid1, camid1, trackid1, fname1 = img1_info
        img_path2, pid2, camid2, trackid2, fname2 = img2_info

        # 加载图片并应用 transform
        img1 = read_image(img_path1)
        img2 = read_image(img_path2)
        img1 = pad_to_square(img1)
        img2 = pad_to_square(img2)
        w1, h1 = img1.size  # 注意顺序是 (W, H)
        #print('w1,h1', w1, h1)
        #GlobalConfig.W1.append(w1)
        #GlobalConfig.H1.append(h1)
        #print('H1',max(GlobalConfig.H1))
        #print('W1',max(GlobalConfig.W1))
        #w2, h2 = img2.size  # 注意顺序是 (W, H)
        #GlobalConfig.W2.append(w2)
        #GlobalConfig.H2.append(h2)
        #print('H2',max(GlobalConfig.H2))
        #print('W2',max(GlobalConfig.W2))
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, pid1, camid1, trackid1, fname1), (img2, pid2, camid2, trackid2, fname2)