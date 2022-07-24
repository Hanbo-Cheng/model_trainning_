import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.utils.data as data
import torch.optim as optim
import scheduler

def scaled_dot_product(q, k, v, mask=None):
    d_k=q.size(-1)
    attn_logits=torch.matmul(q,k.transpose(-2,-1))
    attn_logits=attn_logits/math.sqrt(d_k)
    if mask is not None:
        attn_logits=attn_logits.mask_fill(mask==0,-9e15)
    attention = F.softmax(attn_logits,dim=-1)
    values=torch.matmul(attention,v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0,"Embedding dimension must be 0 modulo number of heads."
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads

        self.qkv_proj = nn.Linear(input_dim,3*embed_dim)
        self.o_proj = nn.Linear(embed_dim,embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self,x,mask=None,return_attention=False):
        batch_size,seq_len,embed_dim=x.size()
        qkv=self.qkv_proj(x)
        qkv=qkv.reshape(batch_size,seq_len,self.num_heads,3*self.embed_dim)
        qkv=qkv.premute(0,2,1,3)
        q,k,v=qkv.chunk(3,dim=-1)

        values,attention=scaled_dot_product(q,k,v,mask=mask)
        values=values.permute(0,2,1,3)
        values=values.reshape(batch_size,seq_len,embed_dim)
        o=self.o_proj(values)
        if(return_attention):
            return o,attention
        else:
            return o

class EncoderBlock(nn.Module):

    def __init__(self,input_dim,num_heads,dim_feedforward,dropout=0.0):
        super().__init__()

        self.self_attn=MultiheadAttention(input_dim,input_dim,num_heads)

        self.linear_net=nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        #两层Linear
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        attn_out = self.self_attn(x,mask=mask)
        x=x+self.dropout(attn_out)
        x=self.norm1(x)

        linear_out = self.linear_net(x)
        x=x+self.dropout(linear_out)
        x=self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,**block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self,x,mask=None):
        for l in self.layers:
            x=l(x,mask=mask)
        return x

def get_attention_maps(self, x, mask=None):
    attention_maps = []
    for l in self.layers:
        _,attn_map=l.self_attn(x, mask=mask, return_attention=True)
        attention_maps.append(attn_map)
        x=l(x)
        return attention_maps


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]= torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe',pe, persistent=False)

    def forward(self,x):
        x=x+self.pe[:,:x.size(1)]
        return x

class TransformerPredictor(pl.LightningModule):

    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers
                 , #lr, warmup, max_iters,
                 dropout=0.0, input_dropout=0.0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
    #
    #     # Apply lr scheduler per step
    #     lr_scheduler = scheduler.CosineWarmupScheduler(optimizer,
    #                                          warmup=self.hparams.warmup,
    #                                          max_iters=self.hparams.max_iters)
    #     return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

class AnomalyPredictor(TransformerPredictor):

    def _calculate_loss(self, batch, mode="train"):
        img_sets, _, labels = batch
        preds = self.forward(img_sets, add_positional_encoding=False) # No positional encodings as it is a set, not a sequence!
        preds = preds.squeeze(dim=-1) # Shape: [Batch_size, set_size]
        loss = F.cross_entropy(preds, labels) # Softmax/CE over set dimension
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
