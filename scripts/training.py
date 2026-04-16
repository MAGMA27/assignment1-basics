import numpy as np
import torch
from cs336_basics.data_loading import data_loading
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import learning_rate_schedule
import datetime
import wandb 
from argparse import Namespace

wandb.login()

# to do: implement the command line training tools
config = Namespace(
    project_name='wandb_demo'
)

project ='wandb_demo'

config = {
    # hyperparameters
    ## training loop
    'total_steps': 1000,
    'batch_size': 32,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lazy_load': False,
    ## model
    'vocab_size': 10000,
    'context_length': 256,
    'd_model': 512,
    'd_ff': 1344,
    'rope_theta': 10000,
    'num_layers': 4,
    'num_heads': 16,
    ## optimizer
    'lr_max': 1e-1,
    'lr_min': 1e-4,
    'betas': (0.999, 0.9),
    'eps': 1e-6,
    'weight_decay': 0.01,
    'l2_max': 1,
    ## tokens id file
    'tk_file': r""
}

config['lr_T_w'] = int(np.around(config['total_steps'] * 0.1))
config['lr_T_c'] = int(np.around(config['total_steps'] * 0.85))

## tokens id file
if config['lazy_load']:
    tokens = np.load(config['tk_file'], mmap_mode='r')
else:
    tokens = np.load(config['tk_file'])

# initial model
model = TransformerLM(
    config['vocab_size'], config['context_length'], config['num_layers'], config['d_model'],
    config['num_heads'], config['d_ff'], config['rope_theta'], device=config['device']
)

# initial optimizer
opt = AdamW(model.parameters(), lr=config['lr_max'], weight_decay=config['weight_decay'], 
            betas=config['betas'], eps=config['eps'])


with wandb.init(project=project, config=config) as run:
    # training loop implementation
    for it in range(config['total_steps']):
        ''''''
        #======================================================================
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #======================================================================
        ## get batch
        sequences, targets = data_loading(tokens, config['batch_size'], config['context_length'], device=config['device'])
        ## zero grad
        opt.zero_grad()
        ## learning rate schedule
        lr = learning_rate_schedule(it, config['lr_max'], config['lr_min'], config['lr_T_w'], config['lr_T_c'])
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        ## forward
        lm_head = TransformerLM(sequences)
        ## loss
        loss = cross_entropy(lm_head, targets)
        #======================================================================
        wandb.log({'it':it, 'loss': loss})
        #======================================================================
        ## backward
        loss.backward()
        ## grad clipping
        gradient_clipping(model.parameters(), config['l2_max'], eps=config['eps'])
        ## opt step
        opt.step()

#======================================================================
wandb.finish()
#======================================================================

