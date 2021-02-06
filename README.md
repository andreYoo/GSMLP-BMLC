# GSMLP-SMLC
Unsupervised Person Re-identification via Multi-Label Prediction and Classification based on Graph-Structural Insight



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

