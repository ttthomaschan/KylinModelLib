import torch
import torch.nn as nn

### 1. tensor.register_hook()
flag1 = 0
if flag1:
    def grad_hook(grad):
        y_grad.append(grad)

    y_grad = list()
    x = torch.tensor([[1.,2.], [3.,4.]], requires_grad=True)
    y = x + 1
    y.register_hook(grad_hook)
    z = torch.mean(y*y)
    z.backward()

    print("type(y): ", type(y))
    print("y.grad: ", y.grad)
    print("y_grad[0]: ", y_grad[0])
#############################################################


### 2. nn.Module.register_forward_hook()
flag2 = 1
if flag2:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1,2,3)
            self.pool1 = nn.MaxPool2d(2,2)
        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x

    def farward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    net = Net()
    net.conv1.weight[0].fill_(1)
    net.conv1.weight[1].fill_(2)
    net.conv1.bias.data.zero_()

    fmap_block = list()
    input_block = []
    net.conv1.register_forward_hook(farward_hook)

    fake_img = torch.ones((1,1,4,4))
    output = net(fake_img)

    print("output shape: {}\noutput value: {}\n".format(output.shape, output))
    print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
    print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))