import torch


'''1. 保存和读取'''
# 方式一：保存加载整个state_dict（推荐）
# 保存
torch.save(model.state_dict(), PATH)
# 加载
model.load_state_dict(torch.load(PATH))
# 测试时不启用 BatchNormalization 和 Dropout
model.eval()


# 方式二：保存加载整个模型
# 保存
torch.save(model, PATH)
# 加载
model = torch.load(PATH)
model.eval()


# 方式三：保存用于继续训练的checkpoint或者多个模型
# 保存
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # ... 
            }, PATH)
# 加载
checkpoint = torch.load(PATH)
start_epoch=checkpoint['epoch']
model.load_state_dict(checkpoint['model_state_dict'])
# 测试时
model.eval()
# or 训练时
model.train()


'''2. 跨 GPU/CPU 处理'''
# 一、GPU上保存，CPU上加载
# 保存
torch.save(model.state_dict(), PATH)
# 加载
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))
# 如果是多gpu保存，需要去除关键字中的module，见下

# 解决1：加载时将module去掉
# 创建一个不包含`module.`的新OrderedDict
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # 去掉 `module.`
    new_state_dict[name] = v
# 加载参数
model.load_state_dict(new_state_dict)

# 解决2：保存checkpoint时不保存module
torch.save(model.module.state_dict(), PATH)


# 二、GPU上保存，GPU上加载
# 保存
torch.save(model.state_dict(), PATH)
# 加载
device = torch.device("cuda")
model.load_state_dict(torch.load(PATH))
model.to(device)


# 三、CPU上保存，GPU上加载
# 保存
torch.save(model.state_dict(), PATH)
# 加载
device = torch.device("cuda")
# 选择希望使用的GPU
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  
model.to(device)


'''3. 打印 '''
# 打印模型的 state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
