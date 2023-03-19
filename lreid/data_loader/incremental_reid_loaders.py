import sys
sys.path.append('../')
import os
from lreid.data_loader.incremental_datasets import IncrementalReIDDataSet, \
    Incremental_combine_train_samples, Incremental_combine_test_samples, IncrementalPersonReIDSamples
import copy
from lreid.datasets import (IncrementalSamples4subcuhksysu, IncrementalSamples4market,
                               IncrementalSamples4duke, IncrementalSamples4sensereid,
                               IncrementalSamples4msmt17, IncrementalSamples4cuhk03,
                               IncrementalSamples4cuhk01, IncrementalSamples4cuhk02,
                               IncrementalSamples4viper, IncrementalSamples4ilids,
                               IncrementalSamples4prid, IncrementalSamples4grid,
                               IncrementalSamples4mix)
from lreid.data_loader.loader import ClassUniformlySampler4Incremental, data, IterLoader, ClassUniformlySampler
import torch
import torchvision.transforms as transforms
from lreid.data_loader.transforms2 import RandomErasing
from collections import defaultdict
from IPython import embed

class IncrementalReIDLoaders:

    def __init__(self, config):
        self.config = config

        # resize --> flip --> pad+crop --> colorjitor(optional) --> totensor+norm --> rea (optional)
        transform_train = [
            transforms.Resize(self.config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(self.config.image_size)]
        if self.config.use_colorjitor: # use colorjitor
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if self.config.use_rea: # use rea
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        # resize --> totensor --> norm
        self.transform_test = transforms.Compose([
            transforms.Resize(self.config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = ['market', 'duke', 'cuhksysu', 'subcuhksysu', 'msmt17', 'cuhk03',
                         'mix', 'sensereid',
                         'cuhk01', 'cuhk02', 'viper', 'ilids', 'prid', 'grid', 'generalizable',
                         'allgeneralizable', 'partgeneralizable', 'finalgeneralizable']

        # dataset
        # print("loaders------line53")
        # embed()

        for a_train_dataset in self.config.train_dataset + self.config.test_dataset:
            # print("incremental_reid_loaders.py line 55-----------------------")
            # embed()
            assert a_train_dataset in self.datasets, a_train_dataset
        # self.config.train_dataset
        # ['market', 'subcuhksysu', 'duke', 'msmt17', 'cuhk03']
        # self.config.test_dataset
        # ['duke', 'market', 'cuhk03', 'allgeneralizable']



        # batch size

        self.if_init_show_loader = self.config.output_featuremaps #output_featuremaps=Flase 'IF During training visualize featuremaps'

        self.use_local_label4validation = self.config.use_local_label4validation # defalut=True validation use global pid label or not

        self.total_step = len(self.config.train_dataset) # 5
        # self.config.train_dataset         
        # ['market', 'subcuhksysu', 'duke', 'msmt17', 'cuhk03']

        # load
        self._load() # init train dataset
        self._init_device()
        self.continual_train_iter_dict = self.incremental_train_iter_dict
        # incremental_train_iter_dict 字典: '1':market的dataloader  , '2':duke的dataloader


        self.continual_num_pid_per_step = [len(v) for v in self.global_pids_per_step_dict.values()]
        self.continual_num_cid_per_step = [len(v) for v in self.global_cids_per_step_dict.values()]
        print(
            f'Show incremental_num_pid_per_step {self.continual_num_pid_per_step}\n')
        print(
            f'Show incremental_num_cid_per_step {self.continual_num_cid_per_step}\n')
        print(f'Show incremental_train_iter_dict (size = {len(self.continual_train_iter_dict)}): \n {self.continual_train_iter_dict} \n--------end \n')


    def _init_device(self):
        self.device = torch.device('cuda')

    def _load(self):

        '''init train dataset'''
        train_samples = self._get_train_samples(self.config.train_dataset)
        # train_samples : [img_path, global_pid, global_cid, dataset_name, local_pid] 5个域的放在了一起
        
        self.incremental_train_iter_dict = {}

        total_pid_list, total_cid_list = [], []
        temp_dict = copy.deepcopy(self.global_pids_per_step_dict)
        for step_index, pid_per_step in self.global_pids_per_step_dict.items():
            if self.config.num_identities_per_domain is -1:
                one_step_pid_list = sorted(list(pid_per_step))
            else:
                one_step_pid_list = sorted(list(pid_per_step))[0:self.config.num_identities_per_domain] # 截取每个域训练集的前500个identity
            temp_dict[step_index] = one_step_pid_list
            # temp中每个step对应500个pid
            total_pid_list.extend(one_step_pid_list)
# 上面的for的作用：每个域的训练集中选500个identity
        num_of_real_train = 0

        # train_samples : [img_path, global_pid, global_cid, dataset_name, local_pid] 5个域的放在了一起
        for item in train_samples:
            # 再筛选一下,计划选500个,实际上可能没到
            if item[1] in total_pid_list:
                num_of_real_train +=1
        print(f'with {self.config.num_identities_per_domain} per domain, the num_of_real_train :{num_of_real_train}')
        # with 500 per domain, the num_of_real_train :25077 
        


        for cid_per_step in self.global_cids_per_step_dict.values():
            total_cid_list.extend(cid_per_step)
        del self.global_pids_per_step_dict
        if self.config.joint_train: # 联合训练
            del self.global_cids_per_step_dict
            self.global_pids_per_step_dict = {0: total_pid_list} # 联合训练时所有训练数据放到一起
            self.global_cids_per_step_dict = {0: total_cid_list}
        else:
            self.global_pids_per_step_dict = temp_dict 

        

        for step_number, one_step_pid_list in self.global_pids_per_step_dict.items():
            # incremental_train_iter_dict 字典: '1':market的dataloader  , '2':duke的dataloader
            self.incremental_train_iter_dict[step_number] = self._get_uniform_incremental_iter(train_samples,
                                                                                                   self.transform_train,
                                                                                                   self.config.p,
                                                                                                   self.config.k,
                                                                                                   one_step_pid_list)


        # from IPython import embed
        # print("incremental_reid_loaders.py-------------------------------------line 149")
        # embed()

        # self.train_iter = self._get_uniform_iter(train_samples, self.transform_train, self.p, self.k)
        if self.if_init_show_loader:
            self.train_vae_iter = self._get_uniform_iter(train_samples, self.transform_test, 4, 2)
            
        '''init test dataset'''
        self.test_loader_dict = defaultdict(list)
        query_sample, gallery_sample = [], []
        for one_test_dataset in self.config.test_dataset: #--test_dataset default=['duke','market','cuhk03','allgeneralizable']
            temp_query_samples, temp_gallery_samples = self._get_test_samples(one_test_dataset)
            query_sample += temp_query_samples
            gallery_sample += temp_gallery_samples
            temp_query_loader = self._get_loader(temp_query_samples, self.transform_test, self.config.test_batch_size)
            temp_gallery_loader = self._get_loader(temp_gallery_samples, self.transform_test,
                                                   self.config.test_batch_size)
            self.test_loader_dict[one_test_dataset].append(temp_query_loader)
            self.test_loader_dict[one_test_dataset].append(temp_gallery_loader)


        # print("incremental_reid_loaders.py----------------------line170")
        # embed()

        IncrementalPersonReIDSamples._show_info(None, train_samples, query_sample, gallery_sample,
                                                name=str(self.config.train_dataset), if_show=True)


    def _get_train_samples(self, train_dataset):
        '''get train samples, support multi-dataset'''
        samples_list = []
        for a_train_dataset in train_dataset:
            if a_train_dataset == 'market':
                samples = IncrementalSamples4market(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'duke':
                samples = IncrementalSamples4duke(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhksysu':
                samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all, use_subset_train=False).train
            elif a_train_dataset == 'subcuhksysu':
                samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all, use_subset_train=True).train
            elif a_train_dataset == 'mix':
                samples = IncrementalSamples4mix(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'sensereid':
                samples = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'msmt17':
                samples = IncrementalSamples4msmt17(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhk03':
                samples = IncrementalSamples4cuhk03(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhk01':
                samples = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhk02':
                samples = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'viper':
                samples = IncrementalSamples4viper(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'ilids':
                samples = IncrementalSamples4ilids(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'prid':
                samples = IncrementalSamples4prid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'grid':
                samples = IncrementalSamples4grid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            samples_list.append(samples)
            # train:samples_list中是5个sample对象

        samples, global_pids_per_step_dict, global_cids_per_step_dict = Incremental_combine_train_samples(samples_list)
        # samples = IncrementalPersonReIDSamples._relabels_incremental(None, samples, 1)
        
        # all_samples : [img_path, global_pid, global_cid, dataset_name, local_pid] 5个域的放在了一起    
        # global_pids_per_step_dict :  {'1':[22,25,65,78,98,101,...], '2':[24,47,58,93,203,...], '3':[...], ...}
        # global_cids_per_step_dict :  {'1':[...], '2':[...], '3':[...], ...}

        self.global_pids_per_step_dict = global_pids_per_step_dict
        self.global_cids_per_step_dict = global_cids_per_step_dict
        return samples


    def _get_test_samples(self, a_test_dataset):
        if a_test_dataset == 'market':
            samples = IncrementalSamples4market(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'duke':
            samples = IncrementalSamples4duke(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhksysu':
            samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all,
                                                     use_subset_train=False)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'subcuhksysu':
            samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all,
                                                     use_subset_train=True)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'mix':
            samples = IncrementalSamples4mix(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'sensereid':
            samples = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'msmt17':
            samples = IncrementalSamples4msmt17(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhk03':
            samples = IncrementalSamples4cuhk03(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhk01':
            samples = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhk02':
            samples = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'viper':
            samples = IncrementalSamples4viper(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'ilids':
            samples = IncrementalSamples4ilids(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'prid':
            samples = IncrementalSamples4prid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'grid':
            samples = IncrementalSamples4grid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'generalizable':

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                               combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                               combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                              combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                              combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(samples_list=[samples4viper,samples4ilids,samples4prid,samples4grid])
        elif a_test_dataset == 'allgeneralizable':

            samples4sensereid = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            samples4cuhk01 = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True,
                                                combineall=self.config.combine_all)

            samples4cuhk02 = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True,
                                                       combineall=self.config.combine_all)

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                                     combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                                     combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(
                samples_list=[samples4viper, samples4ilids, samples4prid, samples4grid,
                              samples4sensereid, samples4cuhk01, samples4cuhk02])
        elif a_test_dataset == 'finalgeneralizable':
            samples4cuhk03 = IncrementalSamples4cuhk03(self.config.datasets_root, relabel=True,
                                                combineall=self.config.combine_all)

            samples4sensereid = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            samples4cuhk01 = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True,
                                                combineall=self.config.combine_all)

            samples4cuhk02 = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True,
                                                       combineall=self.config.combine_all)

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                                     combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                                     combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(
                samples_list=[samples4viper, samples4ilids, samples4prid, samples4grid,
                              samples4sensereid, samples4cuhk01, samples4cuhk02, samples4cuhk03])
        elif a_test_dataset == 'partgeneralizable':

            samples4sensereid = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            # samples4cuhk01 = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True,
            #                                     combineall=self.config.combine_all)
            #
            # samples4cuhk02 = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True,
            #                                            combineall=self.config.combine_all)

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                                     combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                                     combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(
                samples_list=[samples4viper, samples4ilids, samples4prid, samples4grid,
                              samples4sensereid])

        return query, gallery


    def _get_uniform_incremental_iter(self, samples, transform, p, k, pid_list):
        '''
               load person reid data_loader from images_folder
               and uniformly sample according to class for continual
               '''
        # dataset.sample is list  dataset.transform
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        # ClassUniformlySampler
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False,
                                 sampler=ClassUniformlySampler4Incremental(dataset, class_position=1, k=k, pid_list=pid_list))
        iters = IterLoader(loader)
        return iters



    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        # dataset.sample is list  dataset.transform
        dataset = IncrementalReIDDataSet(samples,self.total_step, transform=transform)
        # ClassUniformlySampler
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters




    def _get_random_iter(self, samples, transform, batch_size):
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)
        return iters

    def _get_random_loader(self, samples, transform, batch_size):
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader

    def _get_loader(self, samples, transform, batch_size):
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

