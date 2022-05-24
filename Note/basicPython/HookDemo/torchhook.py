import torch

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