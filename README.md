# GSMLP-SMLC
Unsupervised Person Re-identification via Multi-Label Prediction and Classification based on Graph-Structural Insight

~~~
This paper addresses unsupervised person re-identification (Re-ID) using multi-label prediction and classification based on graph-structural insight. Our method extracts features from person images and produces a graph that consists of the features and a pairwise similarity of them as nodes and edges, respectively.
Based on the graph, the proposed graph structure based multi-label prediction (GSMLP) method predicts multi-labels by considering the pairwise similarity and the adjacency node distribution of each node. The multi-labels created by GSMLP are applied to the proposed selective multi-label classification (SMLC) loss. SMLC integrates a hard-sample mining scheme and a multi-label classification. The proposed GSMLP and SMLC boost the performance of unsupervised person Re-ID without any pre-labelled dataset. Experimental results justify the superiority of the proposed method in unsupervised person Re-ID by producing state-of-the-art performance. 
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




## How to train
~~~
python main2.py --config ./config/kinetics-skeleton/train_joint_mutual.yaml
~~~



## Reproduce the experimental results
~~~
#For Duke and Market datasets
tensorboard --logdir=./wordir

#For MSMT17 dataset
tensorboard --logdir=./wordir

~~~


## Code reference.
~~~
Wang, Dongkai, and Shiliang Zhang. "Unsupervised person re-identification via multi-label classification." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
'https://github.com/kennethwdk/MLCReID'
~~~

