from torch import nn
import math


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # P1
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.max = nn.MaxPool2d(2, 2)
        # P2
        self.conv2 = nn.Conv2d(64, 192, 3, 1, 1)
        # P3
        self.conv3 = nn.Conv2d(192, 128, 1, 1, 0)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 1, 1, 0)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        # P4
        self.conv7 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 1, 1, 0)
        self.conv10 = nn.Conv2d(512, 1024, 3, 1, 1)
        # P5
        self.conv11 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv12 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv13 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv14 = nn.Conv2d(1024, 1024, 3, 2, 1)
        # P6
        self.conv15 = nn.Conv2d(1024, 1024, 3, 1, 1)
        # Linear layer
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 30 * 7 * 7)
        # Activation function
        self.l_relu = nn.LeakyReLU(0.1)
        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img_input):
        x = self.l_relu(self.conv1(img_input))
        x = self.max(x)

        x = self.l_relu(self.conv2(x))
        x = self.max(x)

        x = self.l_relu(self.conv3(x))
        x = self.l_relu(self.conv4(x))
        x = self.l_relu(self.conv5(x))
        x = self.l_relu(self.conv6(x))

        x = self.l_relu(self.conv7(x))
        x = self.l_relu(self.conv8(x))
        x = self.l_relu(self.conv7(x))
        x = self.l_relu(self.conv8(x))
        x = self.l_relu(self.conv7(x))
        x = self.l_relu(self.conv8(x))
        x = self.l_relu(self.conv7(x))
        x = self.l_relu(self.conv8(x))
        x = self.l_relu(self.conv9(x))
        x = self.l_relu(self.conv10(x))
        x = self.max(x)

        x = self.l_relu(self.conv11(x))
        x = self.l_relu(self.conv12(x))
        x = self.l_relu(self.conv11(x))
        x = self.l_relu(self.conv12(x))
        x = self.l_relu(self.conv13(x))
        x = self.l_relu(self.conv14(x))

        x = self.l_relu(self.conv15(x))
        x = self.l_relu(self.conv15(x))

        x = self.l_relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 7, 7, 13)
        return x


def __init__weights(network):
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2.0 / n))  # normalize(mean=0, std=...)
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            layer.weight.data.normal_(0, 0.001)
            layer.bias.data.zero_()


my_net = MyModule()
my_net.apply(__init__weights)
