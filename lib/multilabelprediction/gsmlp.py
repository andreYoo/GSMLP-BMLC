import torch
import pdb

EVAL_MLP = 'true'

class GSMLP(object):
    def __init__(self, t=0.5,l=10,EVAL_MLP =False) :
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = t
        self.topk = l
        self.eval_mlp = EVAL_MLP
    
    def predict_graph_only(self,memory, targets):
        mem_vec = memory[targets]
        #Condition - all features are normalised to |x|=1
        node_sim = mem_vec.mm(memory.t()) #similarity matrix
        all_sim =memory.mm(memory.t()).fill_diagonal_(0.0) #matrix for neightborhood similiarty
        n_sim_batch = torch.cdist(all_sim[targets],all_sim,p=2.0)
        node_sim[node_sim<self.tau]=-1.0
        m, n = node_sim.size() #m is scale of batch, n is the number of images on memory.
        node_sim_sorted, index_sorted = torch.sort(node_sim, dim=1, descending=True)

        n_sim_sorted, index_n_sim_sorted = torch.sort(n_sim_batch,dim=1,descending=False)
        nodeclass = torch.zeros(node_sim.size()).to(self.device)
        mask_num = torch.sum(node_sim_sorted != -1.0, dim=1) #listing candiate using node similiarity.
        for i in range(m):
            topk = int(mask_num[i])
            topk = max(topk,self.topk)
            topk_idx = index_sorted[i,:topk]
            #print('node distance based:')
            #print(topk_idx)
            topk_idx_nbased = index_n_sim_sorted[i,:topk+1]
            if mask_num[i].item()==1:
                nodeclass[i,targets[i]] = float(1)
            else:
                nodeclass[i,topk_idx_nbased] = float(1)
        targets = torch.unsqueeze(targets, 1)
        nodeclass.scatter_(1, targets, float(1))
        if self.eval_mlp=='true':
            return nodeclass,
        else:
            return nodeclass


    def predict(self,memory, targets):
        mem_vec = memory[targets]
        #Condition - all features are normalised to |x|=1
        node_sim = mem_vec.mm(memory.t()) #similarity matrix
        all_sim =memory.mm(memory.t()).fill_diagonal_(0.0) #matrix for neightborhood similiarty
        n_sim_batch = torch.cdist(all_sim[targets],all_sim,p=2.0)
        node_sim[node_sim<self.tau]=-1.0
        m, n = node_sim.size() #m is scale of batch, n is the number of images on memory.
        node_sim_sorted, index_sorted = torch.sort(node_sim, dim=1, descending=True)

        n_sim_sorted, index_n_sim_sorted = torch.sort(n_sim_batch,dim=1,descending=False)
        nodeclass = torch.zeros(node_sim.size()).to(self.device)
        mask_num = torch.sum(node_sim_sorted != -1.0, dim=1) #listing candiate using node similiarity.

        _pos_list = []
        for i in range(m):
            topk = int(mask_num[i].item())
            topk = max(topk,self.topk)
            topk_idx = index_sorted[i,:topk]
            #print('node distance based:')
            #print(topk_idx)
            topk_idx_nbased = index_n_sim_sorted[i,:topk+1]
            #print('Neighborhood similairty based:')
            #print(topk_idx_nbased)
            #nodeclass[i, topk_idx_nbased] = float(1)
            if mask_num[i].item()==1:
                nodeclass[i,targets[i]] = float(1)
                _pos_list.append(targets[i])
            else:
                nodeclass[i,targets[i]] = float(1)
                for j in range(topk):
                    if topk_idx[j] in topk_idx_nbased:continue
                    else:
                        topk_idx[j]=-1
                tmp = topk_idx[topk_idx!=-1]
                _pos_list.append(tmp)
                #print('[%d] similarity %d/%d (%.3f)'%(i,len(tmp),topk,len(tmp)/topk))
                nodeclass[i, topk_idx[topk_idx!=-1]] = float(1)
                nodeclass[i, targets[i]] = float(1)
        targets = torch.unsqueeze(targets, 1)
        nodeclass.scatter_(1, targets, float(1))
        if EVAL_MLP==True:
            return nodeclass, _pos_list
        else:
            return nodeclass



class KNN(object):
    def __init__(self,k=8,l=10,EVAL_MLP =False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.topk = l
        self.eval_mlp = EVAL_MLP

    def predict(self, memory, targets):
        mem_vec = memory[targets]
        # Condition - all features are normalised to |x|=1
        node_sim = mem_vec.mm(memory.t())  # similarity matrix
        m, n = node_sim.size()  # m is scale of batch, n is the number of images on memory.
        node_sim_sorted, index_sorted = torch.sort(node_sim, dim=1, descending=True)
        nodeclass = torch.zeros(node_sim.size()).to(self.device)
        _pos_list = []
        for i in range(m):
            topk = self.k
            topk_idx = index_sorted[i,:topk]
            nodeclass[i, topk_idx] = float(1)
            _pos_list.append(topk_idx)
        targets = torch.unsqueeze(targets, 1)
        nodeclass.scatter_(1, targets, float(1))
        if self.eval_mlp==True:
            return nodeclass, _pos_list
        else:
            return nodeclass

class SS(object):
    def __init__(self,t=0.6, l=10,EVAL_MLP =False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = t
        self.topk = l
        self.eval_mlp = EVAL_MLP

    def predict(self, memory, targets):
        mem_vec = memory[targets]
        # Condition - all features are normalised to |x|=1
        node_sim = mem_vec.mm(memory.t())  # similarity matrix
        node_sim[node_sim<self.tau]=-1.0
        m, n = node_sim.size()  # m is scale of batch, n is the number of images on memory.
        node_sim_sorted, index_sorted = torch.sort(node_sim, dim=1, descending=True)
        nodeclass = torch.zeros(node_sim.size()).to(self.device)
        _pos_list = []
        mask_num = torch.sum(node_sim_sorted != -1.0, dim=1)  # listing candiate using node similiarity.
        for i in range(m):
            topk = int(mask_num[i])
            topk = max(topk, self.topk)
            topk_idx = index_sorted[i,:topk]
            _pos_list.append(topk_idx)
            nodeclass[i, topk_idx] = float(1)
        targets = torch.unsqueeze(targets, 1)
        nodeclass.scatter_(1, targets, float(1))
        if self.eval_mlp==True:
            return nodeclass, _pos_list
        else:
            return nodeclass

