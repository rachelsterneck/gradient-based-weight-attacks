from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_topk_idx(grad): 
  grad_list =  grad.detach().cpu().tolist()
  stack = [enumerate(grad_list)]
  path = [None]
  while stack:
      for path[-1], x in stack[-1]:
          if isinstance(x, list):
              stack.append(enumerate(x))
              path.append(None)
          else:
              yield x, tuple(path)
          break
      else:
          stack.pop()
          path.pop()

# def get_topk_grads(grad, num_gradients):
#   topk_idx = get_topk_idx(grad)
#   idx_map = [i for i in topk_idx]

#   grad_flat = torch.abs(grad).flatten()
#   topk_values, topk_indices = grad_flat.topk(num_gradients)
#   topk_tensor = torch.cuda.FloatTensor(grad.size()).fill_(0)

#   for i in range(len(topk_indices)):
#     idx_expanded = idx_map[topk_indices[i]][1]
#     topk_tensor[idx_expanded[0], idx_expanded[1], idx_expanded[2], idx_expanded[3]] = grad[idx_expanded[0], idx_expanded[1], idx_expanded[2], idx_expanded[3]]

#   return topk_tensor

def get_topk_grads(grad, num_gradients):
  grad_flat = torch.abs(grad).flatten()
  topk_values, topk_indices = grad_flat.topk(num_gradients)
  topk_tensor = torch.cuda.FloatTensor(grad.size()).fill_(0)
  for k in topk_values:
    a = torch.eq(torch.abs(grad), k)*1
    topk_tensor += grad*a
  
  return topk_tensor

def test_attack(model, device, testloader, weight_epsilon, num_gradients, layer_idx, criterion):
  correct = 0
  total = 0
  conv_layers = []

  for i in layer_idx:
    count = 0
    for layer in model.modules():
      if isinstance(layer, nn.Conv2d):
        if count == int(i):
          conv_layers.append(layer)
        count+=1
  
  for batch_idx, (inputs, targets) in enumerate(testloader):
    w_origs = [copy.deepcopy(c.weight.data) for c in conv_layers]
    
    inputs, targets = inputs.to(device), targets.to(device)
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    for conv_layer in conv_layers:
      dldw = conv_layer.weight.grad 
      topk_grads_tensor = get_topk_grads(dldw.data, num_gradients)
      perturbation = weight_epsilon * torch.sign(topk_grads_tensor) ## attack same top k indices
      w_adv = conv_layer.weight + perturbation
      conv_layer.weight = torch.nn.Parameter(w_adv)    

    # predict on input images with new weights
    outputs_adv = model(inputs)
    outputs_adv = F.softmax(outputs_adv.data,dim=1) ## go back to original weight

    vals, predicted = torch.max(outputs_adv.data, 1)

    total += targets.size(0)
    correct += (predicted == targets).sum().item()

    # return weight to original value
    for (conv_layer, w_orig) in zip(conv_layers, w_origs): 
      conv_layer.weight = torch.nn.Parameter(w_orig)

  final_acc = (100 * correct)/ total
  return final_acc, activations_adv, vals 


def attack_layers(model, device, testloader, epsilons, num_weights, layer_idx):
    accs_all = []
    confs_all = []
    criterion = nn.CrossEntropyLoss()

    for e in epsilons:
        acc, adv, confs = test_attack(model, device, testloader, int(e), num_weights, layer_idx, criterion)
        
        accs_all.append(acc/100)
        confs_all.append(torch.mean(confs).item())

        print("Epsilon = {}\t Test Accuracy = {}\t Avg Confidence = {}\n".format(e, acc, torch.mean(confs).item()))
    return accs_all, confs_all
