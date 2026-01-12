# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import itertools
class Shipreid(BaseImageDataset):
    dataset_dir = 'ship_reid'
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Shipreid, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train, train_x, train_y = self._process_dir_train(self.train_dir, relabel=False)

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        query_x = self._process_dir_0(self.query_dir, relabel=False)
        gallery_x = self._process_dir_0(self.gallery_dir, relabel=False)
        query_y = self._process_dir_1(self.query_dir, relabel=False)
        gallery_y = self._process_dir_1(self.gallery_dir, relabel=False)
        if verbose:
            print("=> Shipreid loaded")
            self.print_dataset_statistics(train, train_x, train_y, query, gallery, query_x, gallery_x, query_y, gallery_y)

        self.train = train
        #self.train_pair = train_pair
        self.train_x = train_x
        self.train_y = train_y
        self.query = query
        self.gallery = gallery
        self.query_x = query_x
        self.gallery_x = gallery_x
        self.query_y = query_y
        self.gallery_y = gallery_y

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_train_x_pids, self.num_train_x_imgs, self.num_train_x_cams, self.num_train_x_vids = self.get_imagedata_info(self.train_x)
        self.num_train_y_pids, self.num_train_y_imgs, self.num_train_y_cams, self.num_train_y_vids = self.get_imagedata_info(self.train_y)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        #self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(self.train_pair)
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
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir_0(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]
            if camid == 0:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
    
    def _process_dir_1(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]
            if camid == 1:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
    
    
    def _process_dir_train(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        pid2rgb, pid2sar = {}, {}

        # 按pid分类RGB和SAR图像
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # 忽略无效图片
            pid_container.add(pid)
            if camid == 0:  # RGB图像
                pid2rgb.setdefault(pid, []).append(img_path)
            elif camid == 1:  # SAR图像
                pid2sar.setdefault(pid, []).append(img_path)

        # relabel pid
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        dataset_x = []
        dataset_y = []

        for pid in sorted(pid_container):
            rgb_list = pid2rgb.get(pid, [])
            sar_list = pid2sar.get(pid, [])

            if not rgb_list or not sar_list:
                continue  # 如果缺少RGB或SAR，跳过此pid

            # 重新编号pid
            if relabel:
                mapped_pid = pid2label[pid]
            else:
                mapped_pid = pid


            # 配对：每个RGB与一个SAR匹配
            for rgb_path in rgb_list:
                rgb_tuple = (rgb_path, self.pid_begin + mapped_pid, 0, 1)  # camid=0
                dataset_x.append(rgb_tuple)
            for sar_path in sar_list:
                sar_tuple = (sar_path, self.pid_begin + mapped_pid, 1, 1)  # camid=1
                dataset_y.append(sar_tuple)

        return dataset, dataset_x, dataset_y
#用训练集测试时，用下面这个
class Shipreid1(BaseImageDataset):
    dataset_dir = 'ship_reid'
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Shipreid1, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train, train_x, train_y = self._process_dir_train(self.train_dir, relabel=True)

        query = self._process_dir(self.train_dir, relabel=False)#都在这块
        #gallery = self._process_dir(self.gallery_dir, relabel=False)

        query_x = self._process_dir_0(self.train_dir, relabel=False)#都在这块

        #gallery_x = self._process_dir_0(self.gallery_dir, relabel=False)

        query_y = self._process_dir_1(self.train_dir, relabel=False)#都在这块

        #gallery_y = self._process_dir_1(self.gallery_dir, relabel=False)
        #if verbose:
            #print("=> Shipreid loaded")
            #self.print_dataset_statistics(train, train_x, train_y, query, gallery, query_x, gallery_x, query_y, gallery_y)

        self.train = train
        self.train_x = train_x
        self.train_y = train_y
        
        self.query = query
        #self.gallery = gallery
        
        self.query_x = query_x
        #self.gallery_x = gallery_x

        self.query_y = query_y
        #self.gallery_y = gallery_y

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        #self.num_train_x_pids, self.num_train_x_imgs, self.num_train_x_cams, self.num_train_x_vids = self.get_imagedata_info(self.train_x)
        #self.num_train_y_pids, self.num_train_y_imgs, self.num_train_y_cams, self.num_train_y_vids = self.get_imagedata_info(self.train_y)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        #self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        #self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(self.train_pair)
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
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        print('query', len(dataset))
        return dataset

    def _process_dir_0(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]
            if camid == 0:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        print('query_x', len(dataset))
        return dataset
    
    def _process_dir_1(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]
            if camid == 1:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        print('query_y', len(dataset))
        return dataset
    
    
    def _process_dir_train(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        pid2rgb, pid2sar = {}, {}

        # 按pid分类RGB和SAR图像
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # 忽略无效图片
            pid_container.add(pid)
            if camid == 0:  # RGB图像
                pid2rgb.setdefault(pid, []).append(img_path)
            elif camid == 1:  # SAR图像
                pid2sar.setdefault(pid, []).append(img_path)

        # relabel pid
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 21  # pid == 0 means background
            assert 0 <= camid <= 1
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        dataset_x = []
        dataset_y = []

        for pid in sorted(pid_container):
            rgb_list = pid2rgb.get(pid, [])
            sar_list = pid2sar.get(pid, [])

            if not rgb_list or not sar_list:
                continue  # 如果缺少RGB或SAR，跳过此pid

            # 重新编号pid
            if relabel:
                mapped_pid = pid2label[pid]
            else:
                mapped_pid = pid


            # 配对：每个RGB与一个SAR匹配
            for rgb_path in rgb_list:
                rgb_tuple = (rgb_path, self.pid_begin + mapped_pid, 0, 1)  # camid=0
                dataset_x.append(rgb_tuple)
            for sar_path in sar_list:
                sar_tuple = (sar_path, self.pid_begin + mapped_pid, 1, 1)  # camid=1
                dataset_y.append(sar_tuple)

        return dataset, dataset_x, dataset_y
