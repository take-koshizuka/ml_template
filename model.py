# ======================
# Import Libs
# ======================

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from torchmetrics import MetricCollection, Accuracy

try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False


# from models.resnet import ResNet50

MODEL_NAME = {
    # "model_name" : model class
    # "resnet50" : ResNet50
}

class Net(nn.Module):
    def __init__(self, net, device):
        super(Net, self).__init__()
        self.net = net
        self.device = device
        self.criterion = self.net.criterion
        metrics = MetricCollection([Accuracy(), ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        """ example
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        out = self.forward(x)
        loss = self.criterion(out, y)
        score = self.train_metrics(out, y)
        score = { k: v.item() for k, v in score.items() }
        score['loss'] = loss
        return score
        """
        self.train()
        return
        
    def validation_step(self, batch, batch_idx):
        """ example
        with torch.no_grad():
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            out = self.forward(x)
            loss = self.criterion(out, y)
            score = self.valid_metrics(out, y)
            score = { k: v.item() for k, v in score.items() }
            score['val_loss'] = loss
            return score
        """
        self.eval()
        with torch.no_grad():
            pass
        return

    def training_epoch_end(self, outputs):
        """ example
        avg_loss = torch.mean(torch.tensor([ out['loss'].item() for out in outputs ]).flatten()).item()
        logs = self.train_metrics.compute()
        logs = { k: v.item() for k, v in logs.items() }
        logs['avg_loss'] = avg_loss
        return { 'avg_loss' : avg_loss, 'log' : logs }
        """
        return
        

    def validation_epoch_end(self, outputs):
        """ example
        avg_loss = torch.mean(torch.tensor([ out['val_loss'].item() for out in outputs ]).flatten()).item()
        logs = self.valid_metrics.compute()
        logs = { k: v.item() for k, v in logs.items() }
        logs['avg_loss'] = avg_loss
        return { 'avg_loss' : avg_loss, 'log' : logs }
        """

    def state_dict(self, optimizer, scheduler=None):
        dic =  {
            "net": deepcopy(self.net.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict())
        }
        if not scheduler is None:
            dic["scheduler"] = deepcopy(scheduler.state_dict())
        
        if AMP:
            dic['amp'] = deepcopy(amp.state_dict())
        return dic

    def load_model(self, checkpoint, amp=False):
        self.net.load_state_dict(checkpoint["net"])
        if amp:
            amp.load_state_dict(checkpoint["amp"])

    def parameters_lr(self):
        return self.net.parameters_lr()
