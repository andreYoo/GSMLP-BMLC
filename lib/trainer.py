import time
import numpy as np
import torch
from .utils.meters import AverageMeter
from .loss import SMCL
from .multilabelprediction import GSMLP,KNN,SS

class Trainer(object):
    def __init__(self, cfg, model, graph):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.graph = graph
        self.labelpred = GSMLP(cfg.GSMLP.TAU,cfg.GSMLP.L) #default

        if cfg.MLP.TYPE=='GSMLP':
            self.labelpred = GSMLP(cfg.GSMLP.TAU,cfg.GSMLP.L)
        if cfg.MLP.TYPE=='KNN':
            self.labelpred = KNN(l=cfg.GSMLP.L)
        if cfg.MLP.TYPE=='SS':
            self.labelpred = SS(l=cfg.GSMLP.L)

        self.criterion = SMCL(cfg.SMCL.DELTA, cfg.SMCL.GAMMA).to(self.device)



    def train(self, epoch, data_loader, optimizer,writer,gi=False, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        if gi==True:
            print('Graph Re-initisliation')
            with torch.no_grad():
                for i, inputs in enumerate(data_loader):
                    inputs, pids = self._parse_data(inputs)

                    outputs = self.model(inputs, 'l2feat')
                    self.graph.store(outputs,pids)
                    #self.graph.global_normalisation()


        if epoch % 5 == 0 and epoch != 0:
            print('Look-up table Overhaul - [reinitialising]')
            with torch.no_grad():
                for i, inputs in enumerate(data_loader):
                    inputs, pids = self._parse_data(inputs)

                    outputs = self.model(inputs, 'l2feat')
                    self.graph.store(outputs,pids)
                    #self.graph.global_normalisation()



        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, pids  = self._parse_data(inputs)
            outputs = self.model(inputs, 'l2feat')
            #Batch output - so start on this section

            logits = self.graph(outputs, pids, epoch)


            if epoch >= 1: #Original value is  > 5mux
                multilabel = self.labelpred.predict(self.graph.mem.detach().clone(), pids.detach().clone())
                loss = self.criterion(logits, multilabel, True)
            else:
                loss = self.criterion(logits, pids)


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

    def _parse_data(self, inputs):
        imgs, _t1,_t2, pids = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        #print(_t1)
        #print(_t2)
        return inputs, pids
