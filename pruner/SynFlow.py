import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
import types


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


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
        pruner.apply_mask()
        pruner.score(model, loss, dataloader, device)
        if linear_schedule:
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs) # Linear
        else:
            sparse = sparsity**((epoch + 1) / epochs) # Exponential
        pruner.mask(sparse, scope)


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
    
    signs = linearize(net)

    (data, _) = next(iter(train_dataloader))
    input_dim = list(data[0,:].shape)
    input = torch.ones([1] + input_dim).to(device)
    output = net(input)
    torch.sum(output).backward()

    scores = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = torch.clone(layer.weight.data * layer.weight.grad).detach().abs_()

    nonlinearize(net, signs)


    global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
    k = int((1.0 - sparsity) * global_scores.numel())
    keep_masks = dict()
    if not k < 1:
        threshold, _ = torch.kthvalue(global_scores, k)
        for mask, score in grads.items():
            zero = torch.tensor([0.]).to(mask.device)
            one = torch.tensor([1.]).to(mask.device)
            keep_masks[m] = torch.where(score <= threshold, zero, one)

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks
