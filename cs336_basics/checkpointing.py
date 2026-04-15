import torch
import os
import typing


def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer , 
                    iteration: int, 
                    outstr: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
                    ) -> None:
    '''
    model: torch.nn.Module  
    optimizer: torch.optim.Optimizer  
    iteration: int  
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] 
    '''
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "it": iteration 
        }
    
    torch.save(obj, outstr)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  , 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer 
                    ) -> int:
    '''
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  
    model: torch.nn.Module  
    optimizer: torch.optim.Optimizer 
    '''
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])

    return obj['it']