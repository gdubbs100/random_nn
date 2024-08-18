import torch
import torchvision
import torch.nn as nn
import numpy as np


class RandomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(0.01)

def reinitialise_network(network):

    return None

if __name__=="__main__":
    loss = nn.CrossEntropyLoss()
    trainset = torchvision.datasets.MNIST(
        root = './data', train=True,
        download=True
    )
    X = trainset.data
    y = trainset.targets
    n_tries = 2
    acc_res = []
    log_loss_res = []
    for i in range(n_tries):
        random_net = RandomNetwork()
        outputs = random_net(X.float())
        class_preds = torch.argmax(outputs, axis=1)
        log_loss = loss(outputs, y)
        acc = torch.sum(class_preds==y) / y.size(0)
        print(f"{i} LOSS: {log_loss}")
        print(f"{i} ACCURACY: {acc}")
        log_loss_res.append(log_loss.detach().numpy())
        acc_res.append(acc.detach().numpy())
    
    log_loss_res = np.array(log_loss_res)
    acc = np.array(acc)
    print(f"Overall loss: {np.mean(log_loss_res)} +/- {np.std(log_loss_res)}")
    print(f"Overall accuracy: {np.mean(acc)} +/- {np.std(acc)}")

    print("all good!")