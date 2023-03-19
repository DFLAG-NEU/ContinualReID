# ContinualReID
Implementation of our [Knowledge-Preserving Continual Person Re-Identification using Graph Attention Network]

# Install
## Enviornment
conda create -n creid python=3.7
source activate creid
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.0 -c pytorch
conda install opencv
pip install Cython sklearn numpy prettytable easydict tqdm matplotlib

## Dataset prepration
Please follow [Torchreid_Dataset_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) to download datasets and unzip them to your data path (we refer to 'machine_dataset_path' in train_test.py). Alternatively, you could download some of never-seen domain datasets in [DualNorm](https://github.com/BJTUJia/person_reID_DualNorm).

## Train & Test
python train_test.py

# Acknowledgement
The code is based on the [LifelongReID](https://github.com/TPCD/LifelongReID) and [Graph Attention Network](https://github.com/Diego999/pyGAT)


# Citation
@article{liu2023knowledge,
  title={Knowledge-Preserving continual person re-identification using Graph Attention Network},
  author={Liu, Zhaoshuo and Feng, Chaolu and Chen, Shuaizheng and Hu, Jun},
  journal={Neural Networks},
  volume={161},
  pages={105--115},
  year={2023},
  publisher={Elsevier}
}

