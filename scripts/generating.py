import torch
import torch.nn.functional as F
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    logits: [vocab_size] or [batch, vocab_size]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    mask = (cum_probs - sorted_probs) <= p
    masked_probs = sorted_probs.masked_fill(~mask, 0.0)
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
    
    sampled_idx = torch.multinomial(masked_probs, 1)   # [batch, 1]
    
    original_token_ids = sorted_indices.gather(dim=-1, index=sampled_idx)  # [batch, 1]
    sampled_prob = probs[original_token_ids]
    
    return original_token_ids.squeeze(-1), sampled_prob  # [batch]

if __name__ == '__main__':
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        ## model
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 1344,
        'rope_theta': 10000,
        'num_layers': 4,
        'num_heads': 16,
        ## checkpoint
        'load_ckpt_path': r'check_points\check_points_run2\ddp-single-node-demo_run2_it49999.pt',
    }

    model = TransformerLM(
        config['vocab_size'], config['context_length'], config['num_layers'], config['d_model'],
        config['num_heads'], config['d_ff'], config['rope_theta'], device=config['device']
    )

    obj = torch.load(config['load_ckpt_path'])
    state_dict = obj['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.to(config['device'])
    model.eval()

    prompt = "Once upon a time there was a little boy named Ben."

    vocab_filepath = r'data\vocab_TinyStoriesV2.json'
    merges_filepath = r'data\merges_TinyStoriesV2.json'
    special_tokens = ['<|endoftext|>']
    format='local'
    tker = Tokenizer.from_files(vocab_filepath, merges_filepath, 
                                special_tokens=special_tokens, format=format)
    
    end_token = tker.encode(special_tokens[0])[0]
    max_length = 256
    input_ids = tker.encode(prompt)
    input_ids = torch.tensor(input_ids, device=config['device']).unsqueeze(0) # [1, seq_len]
    prob_lst = []

    with torch.no_grad():
        while input_ids.shape[1] < max_length:
            logits = model(input_ids)[0, -1, :]
            next_id, prob = top_p_sampling(logits, p=0.9, temperature=1)
            prob_lst.append(prob.detach().cpu())
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0).unsqueeze(0)], dim=-1)
            
            if next_id.item() == end_token:
                break
    
    prob_lst = torch.tensor(prob_lst)
    outputs = tker.decode(input_ids.detach().cpu().squeeze(0).tolist())
    print(outputs)
    # print(prob_lst)
    print(f'平均概率: {torch.mean(prob_lst)*100:.2f}%')