from __future__ import division, print_function, absolute_import
import os
import copy
import os.path as osp
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
from lreid.data.datasets import ImageDataset
import re
import glob

class IncrementalSamples4veri776(IncrementalPersonReIDSamples):
    '''
    VeRi776
    '''
    duke_path = 'VeRi/'
    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root, self.duke_path)
        self.train_dir = osp.join(
            root, 'image_train'
        )
        self.query_dir = osp.join(root, 'image_query')
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
        pattern = re.compile(r'([-\d]+)_c([\d]+)') #匹配0002_c002_00030640_0.jpg中的 0002 和 c002

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print("camid", camid)
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append([img_path, pid, camid, 'veri776', pid])

        return data




class VeRi776(ImageDataset):
    """VeRi776.

    Dataset statistics:
        - identities: (576+200) (train + query).
        - images:37778 (train) + 1678 (query) + 11579 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'VeRi'
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

        super(VeRi776, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print("camid:", camid)
            assert 1 <= camid <= 20
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, 'veri776', pid))

        return data
