import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            torch.nn.init.xavier_normal_(self.net[0].weight)
            torch.nn.init.constant_(self.net[0].bias, 0.1)

        def forward(self, x):
            return self.net(x)

    def __init__(self, layers=None, n_input_channels=3):
        super().__init__()
        if layers is None:
            layers = [128, 256]
        L = [
            torch.nn.Conv2d(n_input_channels, 64, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        c = 64
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 1)

        torch.nn.init.zeros_(self.classifier.weight)
        torch.nn.init.xavier_normal_(self.network[0].weight)
        torch.nn.init.constant_(self.network[0].bias, 0.1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        # compute features
        z = self.network(x)
        # global average pooling
        z = z.mean(dim=[2, 3])
        # classify
        return self.classifier(z)

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model_CNN():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r