import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ngpu, image_size, ndf, extra_layers):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential()
        layer = 1

        # first layer
        self.main.add_module(
            f"Conv({layer})",
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False))
        self.main.add_module(
            f"LeakyReLU({layer})",
            nn.LeakyReLU(0.2, inplace=True))
        layer += 1

        # extra layers
        for e in range(extra_layers):
            self.main.add_module(
                f"Conv({layer})",
                nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            self.main.add_module(
                f"BatchNorm({layer})",
                nn.BatchNorm2d(ndf))
            self.main.add_module(
                f"LeakyReLU({layer})",
                nn.LeakyReLU(0.2, inplace=True))
            layer += 1

        size = 2
        while size < image_size // 4:
            self.main.add_module(
                f"Conv({layer})",
                nn.Conv2d(ndf*size//2, ndf*size, 4, 2, 1, bias=False))
            self.main.add_module(
                f"BatchNorm({layer})",
                nn.BatchNorm2d(ndf*size))
            self.main.add_module(
                f"LeakyReLU({layer})",
                nn.LeakyReLU(0.2, inplace=True))
            layer += 1
            size *= 2

        # last layer
        self.main.add_module(
            f"Conv({layer})",
            nn.Conv2d(ndf*size//2, 1, 4, 1, 0, bias=False))
        self.main.add_module(
            f"Sigmoid({layer})",
            nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
