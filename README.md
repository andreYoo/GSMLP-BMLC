# GSMLP-SMLC
Unsupervised Person Re-identification via Multi-Label Prediction and Classification based on Graph-Structural Insight
<<<<<<< HEAD
=======

~~~
This paper addresses unsupervised person re-identification (Re-ID) using multi-label prediction and classification based on graph-structural insight. Our method extracts features from person images and produces a graph that consists of the features and a pairwise similarity of them as nodes and edges, respectively.
Based on the graph, the proposed graph structure based multi-label prediction (GSMLP) method predicts multi-labels by considering the pairwise similarity and the adjacency node distribution of each node. The multi-labels created by GSMLP are applied to the proposed selective multi-label classification (SMLC) loss. SMLC integrates a hard-sample mining scheme and a multi-label classification. The proposed GSMLP and SMLC boost the performance of unsupervised person Re-ID without any pre-labelled dataset. Experimental results justify the superiority of the proposed method in unsupervised person Re-ID by producing state-of-the-art performance. 
~~~
>>>>>>> 4a1803e839f69db80d782f727c7cfe825016e632

## Abastract
~~~
This paper addresses unsupervised person re-identification (Re-ID) using multi-label prediction and classification based on graph-structural insight. Our method extracts features from person images and produces a graph that consists of the features and a pairwise similarity of them as nodes and edges, respectively. Based on the graph, the proposed graph structure based multi-label prediction (GSMLP) method predicts multi-labels by considering the pairwise similarity and the adjacency node distribution of each node. The multi-labels created by GSMLP are applied to the proposed selective multi-label classification (SMLC) loss. SMLC integrates a hard-sample mining scheme and a multi-label classification. The proposed GSMLP and SMLC boost the performance of unsupervised person Re-ID without any pre-labelled dataset. Experimental results justify the superiority of the proposed method in unsupervised person Re-ID by producing state-of-the-art performance. 
~~~


##Dependencies

This project mainly complied with Python3.6, Pytorch 1.3. All details are included in the 'requirement.txt'
~~~
pip install -r requirements.txt
~~~


## File configuration
.<br>
├── abla.py<br>
├── arun.sh<br>
├── config<br>
│   ├── config.txt<br>
│   ├── kinetics-skeleton<br>
│   │   ├── test_bone.yaml<br>
│   │   ├── test_joint_mutual_none.yaml<br>
│   │   ├── test_joint_mutual_tmp.yaml<br>
│   │   ├── test_joint_mutual.yaml<br>
│   │   ├── test_joint.yaml<br>
...


## Dataset reference
~~~
Refer [ECN](https://github.com/zhunzhong07/ECN) to prepare DukeMTMC-ReID dataset, Market-1501 dataset,and MSMT17 dataset.
~~~



## How to train and test
~~~
#Train
python tools/train.py --experiments ./experiments/***.yml --gpus 0,1

#Test
python tools/test.py --experiments ./experiments/***.yml --gpus 0,1
~~~


## Reproduce the experimental results

You can download the checkpoint files to reproduct the experiment results from [here](https://drive.google.com/file/d/1CsKlNc06ZQbMlThEPkowmDSlMJlLmdq8/view?usp=sharing).

~~~
#For Duke
python ./tools/test.py --experiments/duke_eval.yml --gpus 0,1

#For Market-1501
python ./tools/test.py --experiments/market_eval.yml --gpus 0,1

#For MSMT1
python ./tools/test.py --experiments/msmt17_eval.yml --gpus 0,1

~~~


## Code reference.
~~~
<<<<<<< HEAD
* The code is mainly encouraged by [ECN](https://github.com/zhunzhong07/ECN) and [MLCReID](https://github.com/kennethwdk/MLCReID)
=======
Wang, Dongkai, and Shiliang Zhang. "Unsupervised person re-identification via multi-label classification." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
'https://github.com/kennethwdk/MLCReID'
>>>>>>> 4a1803e839f69db80d782f727c7cfe825016e632
~~~

