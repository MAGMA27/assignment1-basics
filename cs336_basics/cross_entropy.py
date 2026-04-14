import torch

def cross_entropy(
        inputs: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
    '''
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]
    '''
    class_dim = -1

    z_max = torch.max(inputs, dim=-1, keepdim=True).values
    inputs_stable = inputs - z_max

    targets_expanded = targets.unsqueeze(class_dim)
    z_t_stable = torch.gather(inputs_stable, class_dim, targets_expanded)

    # l = -z_t_stable + torch.log(torch.sum(torch.exp(inputs_stable)))
    log_sum_exp = torch.logsumexp(inputs_stable, dim=class_dim)
    nll = log_sum_exp - z_t_stable.squeeze(class_dim)

    return torch.mean(nll)