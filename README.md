# UniDA
The domain adaptation (DA) approaches available to date are usually not well suited for practical DA scenarios of remote sensing image classification, since these methods (such as unsupervised DA) rely on rich prior knowledge about the relationship between label sets of source and target domains, and source data are often not accessible  due to privacy or confidentiality issues. To this end, we propose a practical universal domain adaptation setting for remote sensing image scene classification that requires no prior knowledge on the label sets. Furthermore,  a novel universal domain adaptation method without source data is proposed for cases when the source data is unavailable. The architecture of the model is divided into two parts: the source data generation stage and the model adaptation stage. The first stage estimates the conditional distribution of source data from the pre-trained model using the knowledge of class-separability in the source domain and then synthesizes the source data. With this synthetic source data in hand, it becomes a universal DA task to classify a target sample correctly if it belongs to any category in the source label set, or mark it as ``unknown" otherwise. In the second stage, a novel transferable weight that distinguishes the shared and private label sets in each domain promotes the adaptation in the automatically discovered shared label set and recognizes the 'unknown' samples successfully. Empirical results show that the proposed model is effective and practical for remote sensing image scene classification, regardless of whether the source data is available or not.
# Universal Domain Adaptation for Remote Sensing Image Scene Classification
![Alt text](https://github.com/zhu-xlab/UniDA/blob/main/SGD-MA/image/fig1.jpg)
Fig. 1. Different domain adaptation scenarios. (a) Closed-set DA, which assumes that the source domain and the target domain have shared label sets. (b) Partial DA, which assumes that target label sets are considered a subset of source label sets. (c) Open-set DA, which assumes that source label sets are considered a subset of target label sets. (d) Universal DA, which imposes no prior knowledge on the label sets. Label sets are divided into shared and private label sets in each domain. (e) Universal DA without source data. The source dataset is not available in the practical universal DA scenarios of remote sensing.
# SDG-MA
![Alt text](https://github.com/zhu-xlab/UniDA/blob/main/SGD-MA/image/fig22.jpg)
Fig. 2. Overview of the proposed UniDA without source data (SDG-MA). The model consists of a source data generation stage and a model adaptation stage.
## Dataset and configuration
`AID2NWPU-train-config.yaml` is the configuration for the experiment from AID dataset to NWPU-RESISC45 dataset.
`RSSCN2AID-train-config.yaml` is the configuration for the experiment from RSSCN7 dataset to AID dataset.
`RSSCN2NWPU-train-config.yaml` is the configuration for the experiment from RSSCN7 dataset to NWPU-RESISC45 dataset.
`RSSCN2UCL-train-config.yaml` is the configuration for the experiment from RSSCN7 dataset to UC Merced dataset.
## Pre-tranined model on source data
The pre-tranined models on AID dataset and RSSCN7 dataset are available at [Google Drive](https://drive.google.com/drive/folders/1C6sauYyc0Z4ABWSb6jwKbMsX1LBrpnX_?usp=share_link)
## run
After configuring the file path, run the code `python main.py`
# Citation
@ARTICLE{10043671,  
  author={Xu, Qingsong and Shi, Yilei and Yuan, Xin and Zhu, Xiao Xiang},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},   
  title={Universal Domain Adaptation for Remote Sensing Image Scene Classification},   
  year={2023},  
  volume={61},  
  number={},  
  pages={1-15},  
  doi={10.1109/TGRS.2023.3235988}}  

