import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, image_size, ngf, extra_layers, nz=100):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # empty module
        self.main = nn.Sequential()
        size = image_size // 8
        layer = 1

        # first layer
        self.main.add_module(
            f"Conv({layer})",
            nn.ConvTranspose2d(nz, ngf*size, 4, 1, 0, bias=False))
        self.main.add_module(
            f"BatchNorm({layer})",
            nn.BatchNorm2d(ngf*size))
        self.main.add_module(
            f"ReLU({layer})",
            nn.ReLU(True))
        layer += 1
        size //= 2

        while size > 0:
            self.main.add_module(
                f"Conv({layer})",
                nn.ConvTranspose2d(ngf*size*2, ngf*size, 4, 2, 1, bias=False))
            self.main.add_module(
                f"BatchNorm({layer})",
                nn.BatchNorm2d(ngf*size))
            self.main.add_module(
                f"ReLU({layer})",
                nn.ReLU(True))
            layer += 1
            size //= 2

        # extra layers
        for e in range(extra_layers):
            self.main.add_module(
                f"Conv({layer})",
                nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False))
            self.main.add_module(
                f"BatchNorm({layer})",
                nn.BatchNorm2d(ngf))
            self.main.add_module(
                f"ReLU({layer})",
                nn.ReLU(True))
            layer += 1

        # last layer
        self.main.add_module(
            f"Conv({layer})",
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False))
        self.main.add_module(
            f"Tanh({layer})",
            nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
