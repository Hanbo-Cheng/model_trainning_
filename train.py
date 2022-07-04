import pytorch_lightning as pl
import os
from CIFA_MODULE import *
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def train_model(model_name,train_loader,val_loader,save_name=None,**kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    CHECKPOINT_PATH = "/saved_models/tutorial5"
    if save_name is None:
        save_name = model_name
    ## Create a PyTorch Lightning trainer with the generation callback
    trainer=pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH,save_name),
                      gpus=1 if str(device)=='cuda' else 0,
                      max_epochs=180,                                                                   # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                      callbacks=[ModelCheckpoint(save_weights_only=True,mode='max',monitor='val_acc'), # Log learning rate every epoch
                                LearningRateMonitor('epoch')])   
    trainer.logger._log_graph=True
    trainer.logger._default_hp_metric=None
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename=os.path.join(CHECKPOINT_PATH,save_name+".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading ...")
        model = CIFARModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = CIFARModule(model_name=model_name,**kwargs)
        trainer.fit(model,train_loader,val_loader)   # 相当于那一堆反向传播，梯度归零，step之类的复杂步骤
        model=CIFARModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
    val_result = trainer.test(model,val_loader,verbose=False)
    test_result=trainer.test(model,val_loader,verbose=False)
    result={"test0":test_result[0]['test_acc'],'val':val_result[0]['test_acc']}
    
    return model,result

