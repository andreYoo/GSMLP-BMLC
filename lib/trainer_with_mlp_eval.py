import time
import numpy as np
import torch
from .utils.meters import AverageMeter
from .utils.plot_figures import utils_for_fig3
from .utils.mlp_statistics import precision_recall
from .loss import SMCL
from .multilabelprediction import GSMLP, KNN, SS
import pdb
class Trainer(object):
    def __init__(self, cfg, model, graph):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.graph = graph
        self.eval_mlp= True

        if cfg.MLP.TYPE=='SMLC':
            self.labelpred = GSMLP(cfg.GSMLP.TAU,cfg.GSMLP.L,EVAL_MLP=self.eval_mlp)
        if cfg.MLP.TYPE=='KNN':
            self.labelpred = KNN(l=cfg.GSMLP.L,EVAL_MLP=self.eval_mlp)
        if cfg.MLP.TYPE=='SS':
            self.labelpred = SS(t=cfg.GSMLP.T,l=cfg.GSMLP.L,EVAL_MLP=self.eval_mlp)

        self.criterion = MMCL(cfg.SMCL.DELTA, cfg.SMCL.R).to(self.device)



    def train(self, epoch, data_loader, optimizer,writer, print_freq=1):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        if epoch==0:
            print('True pid table')
            with torch.no_grad():
                _tmax = 0
                for i, inputs in enumerate(data_loader):
                    inputs, pids,tpids = self._parse_data(inputs)
                    self.graph.tpid_memory[pids.cpu().numpy()]=tpids.cpu().numpy()
                    _max = np.max(tpids.cpu().numpy())
                    if _max > _tmax:
                        _tmax = _max
                    #self.graph.global_normalisation()
                #print statistics
                pdb.set_trace()
                _avg_array = np.zeros([_tmax])
                for _t in range(_tmax):
                    _avg_array[_t]=np.sum(self.graph.tpid_memory==_t)
                print('average samples per identity: %.3f'%(np.mean(_avg_array)))

        if epoch % 5 == 0 and epoch != 0:
            print('Look-up table Overhaul - [reinitialising]')
            with torch.no_grad():
                for i, inputs in enumerate(data_loader):
                    inputs, pids,_ = self._parse_data(inputs)
                    outputs = self.model(inputs, 'l2feat')
                    self.graph.store(outputs,pids)
                    #self.graph.global_normalisation()

        precision = 0.0
        recall = 0.0
        _num_positive = 0
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, pids, tpids  = self._parse_data(inputs)
            outputs = self.model(inputs, 'l2feat')
            #Batch output - so start on this section
            logits = self.graph(outputs, pids, epoch)


            if epoch > 5: #Original value is  > 5mux
                if self.eval_mlp==True:
                    multilabel, positive_labels = self.labelpred.predict(self.graph.mem.detach().clone(), pids.detach().clone())
                    batch_precision, batch_recall = precision_recall(tpids.cpu().numpy(),np.array(positive_labels),self.graph.tpid_memory)
                    precision += batch_precision
                    recall += batch_recall
                    #_num_positive += len(positive_labels)
                else:
                    multilabel, _ = self.labelpred.predict(self.graph.mem.detach().clone(), pids.detach().clone())

                loss, _num_pos = self.criterion(logits, multilabel, True)
                _num_positive += _num_pos
            else:
                loss, _ = self.criterion(logits, pids)


            losses.update(loss.item(), outputs.size(0))
            writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader)+i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}][{}/{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f})" \
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg)
                print(log)
            torch.cuda.empty_cache()
        if epoch > 5:  # Original value is  > 5mux
            plog = "[Epoch {}]Average # of positive {} Prediction {:.3f} Recall {:.3f}".format(epoch,_num_positive/len(data_loader),precision/len(data_loader), recall/len(data_loader))
            print(plog)
            writer.add_scalar("Positive_label_num/train", _num_positive/len(data_loader), epoch)
            writer.add_scalar("Precision/train", precision/len(data_loader), epoch)
            writer.add_scalar("Recall/train", recall/len(data_loader), epoch)

    def _parse_data(self, inputs):
        imgs, _t1,true_pids, pids = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        true_pids = true_pids.to('cpu')
        return inputs, pids, true_pids
