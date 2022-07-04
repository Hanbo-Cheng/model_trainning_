
import pytorch_lightning as pl
from model.google_net import *
import torch.optim as optim
## Standard libraries
import os
import numpy as np
import random
from PIL import Image
from types import SimpleNamespace
act_fn_by_name={
    "tanh":nn.Tanh,
    "relu":nn.ReLU,
    "leakyrelu":nn.LeakyReLU,
    "gelu":nn.GELU
}
#model_dict["GoogleNet"] = GoogleNet
model_dict={
    "GoogleNet":GoogleNet
}

def create_model(model_name,model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"

class CIFARModule(pl.LightningModule):
    
    def __init__(self,model_name,model_hparams,optimizer_name,optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model 
            model_hparams -Hyperparameters for the model, as dictionary
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams -Hyperparameters for the optimizer, as dictionary. This includes lr, weight decay...
        """
        
        super().__init__()
        # Exports the hyperparameters to a YAML file
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name,model_hparams)
        # Create loss module
        self.loss_module= nn.CrossEntropyLoss()
        # Example input for visualizeing the graph in Tensorboard
        self.example_input_array = torch.zeros((1,3,32,32),dtype=torch.float32)
        
    def forward(self,imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)
    
    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers
        if self.hparams.optimizer_name == 'Adam':
            optimizer=optim.AdamW(
                self.parameters(),**self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer=optim.SGD(self.parameters(),**self.hparams.optimizer_hparams)
            
        else:
            assert False, f"Unknow optimizer :\"{self.hparams.optimizer_name}\""
            
        scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,milestones=[100,150],gamma=0.1)
        return [optimizer],[scheduler]
    
    def training_step(self,batch,batch_idx):
        #"batch" is output of the training data loader
        imgs,labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds,labels)
        acc = (preds.argmax(dim=-1)==labels).float().mean()
        
        self.log('train_acc',acc,on_step=False, on_epoch=True)
        self.log('train_loss',loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        imgs,labels=batch
        preds=self.model(imgs).argmax(dim=-1)
        acc=(labels==preds).float().mean()
        self.log('val_acc',acc)
        
    def test_step(self, batch, batch_idx):
        imgs,labels=batch
        preds, labels= batch
        preds = self.model(imgs).argmax(dim=-1)
        acc=(labels==preds).float().mean()
        
        self.log("test_acc",acc)
        
        