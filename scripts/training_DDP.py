import torch
import os
import torch
import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb 
import numpy as np
from argparse import Namespace
from cs336_basics.data_loading import data_loading
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class TokenSequenceDataset(Dataset):
    def __init__(self, x_np: np.array, context_length: int):
        self.context_length = context_length
        self.seq = torch.from_numpy(x_np[:-1]).long()
        self.target = torch.from_numpy(x_np[1:]).long()
        
        self.num_samples = len(self.seq) - self.context_length + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq = self.seq[idx : idx + self.context_length]
        target_seq = self.target[idx : idx + self.context_length]
        return input_seq, target_seq

def main(rank, project, config, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        wandb.init(
            project=project,
            config=config
        )

    # initial model
    model = TransformerLM(
        config['vocab_size'], config['context_length'], config['num_layers'], config['d_model'],
        config['num_heads'], config['d_ff'], config['rope_theta'], device=device
    )
    if config['load_ckpt_path'] != '':
        it_loaded = load_checkpoint(config['load_ckpt_path'], model, opt)
        print(f'it = {it_loaded}')
    else:
        it_loaded = 0
    model = model.to(device)

    # initial optimizer
    opt = AdamW(model.parameters(), lr=config['lr_max'], weight_decay=config['weight_decay'], 
                betas=config['betas'], eps=config['eps'])
    

    model = DDP(model, device_ids=[rank])

    ## tokens id file
    if config['lazy_load']:
        tokens_train = np.memmap(config['tk_train_file'], dtype=np.uint16, mode='r')
        tokens_valid = np.memmap(config['tk_valid_file'], dtype=np.uint16, mode='r')
    else:
        tokens_train = np.fromfile(config['tk_train_file'], dtype=np.uint16)
        tokens_valid = np.fromfile(config['tk_valid_file'], dtype=np.uint16)
    
    dataset = TokenSequenceDataset(tokens_train, config['context_length'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler, shuffle=False, drop_last=True)
    
    # training loop implementation
    for epoch in range(100):
        sampler.set_epoch(epoch)
        for it, (sequences, targets) in enumerate(dataloader):
            it += it_loaded
            ''''''
            sequences, targets = sequences.to(device), targets.to(device)
            #======================================================================
            start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #======================================================================
            ## zero grad
            opt.zero_grad()
            ## learning rate schedule
            lr = learning_rate_schedule(it, config['lr_max'], config['lr_min'], config['lr_T_w'], config['lr_T_c'])
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            ## forward
            lm_head = model(sequences)
            ## loss
            loss = cross_entropy(lm_head, targets)
            ## backward
            loss.backward()
            ## grad clipping
            l2p = gradient_clipping(model.parameters(), config['l2_max'], eps=config['eps'])
            ## opt step
            opt.step()
            #======================================================================
            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #======================================================================
            # valid loss
            seq_val, tar_val = data_loading(tokens_valid, config['batch_size'], 
                                            config['context_length'], device=device)
            lm_head_val = model(seq_val)
            valid_loss = cross_entropy(lm_head_val, tar_val)
            # print and log
            #======================================================================
            if rank == 0:
                print(f'step:{it} from {start_time} to {end_time} training_loss={loss} valid_loss={valid_loss}, learning_rate={lr}, l2p={l2p}')
                wandb.log({'it':it, 'training_loss': loss, 'valid_loss': valid_loss, 'learning_rate': lr, 'l2p': l2p})
            #======================================================================
                if (it+1) % 2500 == 0:
                    out_dir = f'{config['save_ckpt_path']}_run{config['run_idx']}'
                    os.makedirs(out_dir, exist_ok=True)
                    output_path = f'{out_dir}//{project}_run{config['run_idx']}_it{it}.pt'
                    save_checkpoint(model, opt, it, output_path)
            
            if it >= config['total_steps']:
                break
        if it >= config['total_steps']:
                break
        
    if rank == 0:
        wandb.finish()
        
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 2
    wandb.login()

    # to do: implement the command line training tools
    config = Namespace(
        project_name='ddp-single-node-demo'
    )

    project ='ddp-single-node-demo'
    run_idx = 2
    config = {
        # hyperparameters
        ## training loop
        'total_steps': 50000,
        'batch_size': 32,
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
        'lr_max': 1e-3,
        'lr_min': 1e-6,
        'betas': (0.9, 0.999),
        'eps': 1e-6,
        'weight_decay': 0.01,
        'l2_max': 5,
        ## tokens id file
        'tk_train_file': r"data/tokens_TinyStoriesV2_train.npy",
        'tk_valid_file': r"data/tokens_TinyStoriesV2_valid.npy",
        ## checkpoint
        'load_ckpt_path': r'',
        'save_ckpt_path': r'check_points',
        ## run idx
        'run_idx': run_idx
    }

    config['lr_T_w'] = int(np.around(config['total_steps'] * 0.1))
    config['lr_T_c'] = int(np.around(config['total_steps'] * 0.85))

    # ddp
    mp.spawn(fn=main, args=(project, config, world_size), 
             nprocs=world_size, join=True)

