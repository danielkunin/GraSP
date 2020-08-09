import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import types

def score(model, dataloader, device):
    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            param.mul_(signs[name])
    
    signs = linearize(model)

    (data, _) = next(iter(dataloader))
    input_dim = list(data[0,:].shape)
    input = torch.ones([1] + input_dim).to(device)
    output = model(input)
    torch.sum(output).backward()

    scores = dict()
    old_modules = list(model.modules())
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            scores[old_modules[idx]] = torch.clone(layer.weight.data * layer.weight.grad).detach().abs_()

    nonlinearize(model, signs)

    return scores

def mask(scores, sparsity):
    global_scores = torch.cat([torch.flatten(v) for v in scores.values()])
    k = int((1.0 - sparsity) * global_scores.numel())
    keep_masks = dict()
    if not k < 1:
        threshold, _ = torch.kthvalue(global_scores, k)
        for mask, score in scores.items():
            zero = torch.tensor([0.]).to(mask.device)
            one = torch.tensor([1.]).to(mask.device)
            keep_masks[m] = torch.where(score <= threshold, zero, one)
    return keep_masks


def apply_mask(model, masks):
    old_modules = list(model.modules())
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.mul_(masks[old_modules[idx]])


def SynFlow(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net).eval()
    net.zero_grad()

    weights = []
    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)

    # rescale_weights(net)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal(layer.weight)
            weights.append(layer.weight)

    # Prune model
    epochs = 100
    for epoch in tqdm(range(epochs)):
        scores = score(model, train_dataloader, device)
        sparsity = keep_ratio**((epoch + 1) / epochs) # Exponential
        masks = mask(scores, sparsity)
        apply_mask(model, masks)


    return keep_masks
