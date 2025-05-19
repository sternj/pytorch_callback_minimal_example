import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import logging
from sys import argv
from ._C import _begin_fwd_op, _begin_bwd_op, _end_bwd_op

@torch._dynamo.disable
def begin_forward_hook(module, args):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg.requires_grad_()
    bw_cbid = _begin_fwd_op(module.unique_id)
    module.bw_cbid = bw_cbid
    return None

@torch._dynamo.disable
def begin_backward_hook(module, grad_output):
    _begin_bwd_op(module.bw_cbid)
    return None

@torch._dynamo.disable
def end_backward_hook(module, grad_input, grad_output):
    _end_bwd_op(module.bw_cbid)

counter = 0 
def get_unique_id():
    global counter
    counter += 1
    return counter
class Operator1(nn.Module):
    def __init__(self):
        super().__init__()
        self.unique_id = get_unique_id()
        self.register_forward_pre_hook(begin_forward_hook)
        self.register_full_backward_pre_hook(begin_backward_hook)
        self.register_full_backward_hook(end_backward_hook)
    def forward(self, x):

        x = x ** 2
        x = torch.sin(x)
        x = x + 1
        x = torch.exp(x)
        return x


class Operator2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(begin_forward_hook)
        self.register_full_backward_pre_hook(begin_backward_hook)
        self.register_full_backward_hook(end_backward_hook)
        self.operator1 = Operator1()
        self.unique_id = get_unique_id()
    def forward(self, x):

        x = self.operator1(x)

        weight = torch.randn(x.size(1), x.size(1), device=x.device)
        x = torch.matmul(x, weight)
        x = x + 3
        
        return x


class CustomNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.unique_id = get_unique_id()
        self.register_forward_pre_hook(begin_forward_hook)
        self.register_full_backward_pre_hook(begin_backward_hook)
        self.register_full_backward_hook(end_backward_hook)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 10)

        self.operator2 = Operator2()

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.operator2(x)

        x = self.fc4(x)
        
        return x






def main():
    device = 'cuda'

    model = CustomNN().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = torch.randn(100, 64).to(device)
    train_labels = torch.randint(0, 10, (100,)).to(device)

    num_epochs = 1



    # torch.cuda.cudart().cudaProfilerStart()
    # with torch.autograd.profiler.emit_nvtx():
    if True:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    # torch.cuda.cudart().cudaProfilerStop()

if __name__ == '__main__': 
    main()