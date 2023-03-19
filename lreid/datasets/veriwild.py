from __future__ import division, print_function, absolute_import
import os
import copy
import os.path as osp
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
from lreid.data.datasets import ImageDataset
import re
import glob

class IncrementalSamples4veriwild(IncrementalPersonReIDSamples):
    '''
    Duke dataset
    '''
    # duke_path = 'dukemtmc-reid/DukeMTMC-reID/'
    veriwild_path = 'veriwild/'
    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root, self.veriwild_path)
        # self.train_dir = osp.join(
        #     root, 'bounding_box_train'
        # )
        self.train_dir = osp.join(
            root, 'image_train'
        )
        # self.query_dir = osp.join(root, 'query')
        self.query_dir = osp.join(root, 'image_query')
        # self.gallery_dir = osp.join(
        #     root, 'bounding_box_test'
        # )
        self.gallery_dir = osp.join(
            root, 'image_test'
        )

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # /media/lzs/de2ef254-eaa4-4486-b00b-ab367ed2a6d8/home/lzs/LifelongReID/dataset/dukemtmc-reid/DukeMTMC-reID/0001_c2_f0046182.jpg
        # pattern = re.compile(r'([-\d]+)_c(\d)') # 匹配0001_c2_f0046182.jpg的 0001和c2
        pattern = re.compile(r'n([\d]+)_.*?_c([\d]+)')  # 匹配n00002_000007_c41.jpg' 的n00002和c41

        pid_container = set()
        for img_path in img_paths: # 每张图片
            pid, _ = map(int, pattern.search(img_path).groups()) # (pid,cid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 0 <= camid <= 174
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append([img_path, pid, camid, 'dukemtmcreid', pid])

        return data




class VeriWild(ImageDataset):
    """VeriWild.

   
    Dataset statistics(part):
        - identities: 900 (train + query).
        - images:6248 (train) + 3230 (query) + 3230 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'veriwild'
    # dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = osp.join(
            self.dataset_dir, 'image_train'
        )
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'image_test'
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(VeriWild, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        pattern = re.compile(r'n([\d]+)_.*?_c([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 0 <= camid <= 174
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, 'veriwild', pid))

        return data
