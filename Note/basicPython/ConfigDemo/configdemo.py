'''
### Config 用法汇总 ###
1. 通过 dict 生成 config
2. 通过配置文件生成 config
3. 自动替换预定义变量
4. 导入自定义模块
5. 合并多个配置文件
1）从 base 文件中合并
2）从多个 base 文件中合并
3）合并字典到配置中
4）allow_list_keys 模式合并
5）允许删掉特定内容
6. pretty_text 和 dump
'''

from mmcv import Config

## 1. 通过 dict 生成 config
cfg = Config(dict(a=1, b=dict(b1=[0, 1])))

# 可以通过 .属性方式访问
print(cfg.b.b1) # --> [0,1]

# 比较
cfg1 = dict(type="10", data=10)
cfg2 = {"type":"10", "data":10}
print(type(cfg1))  # --> <class 'dict'>
print(type(cfg2))  # --> <class 'dict'>
print(cfg1 == cfg2)  # --> True

# print(cfg2.type)  # --> Error
cfg = Config(cfg2)
print(cfg.type)  # --> 10
##################################################

## 2. 通过配置文件生成 config
cfg = Config.fromfile("./cfg.py")

print(cfg.filename)
print(cfg.type)
###################################################

## 3. 自动替换预定义变量
# # 假设 h.py 文件里面存储的内容是：
# # cfg_dict = dict(
# #         item1='{{fileBasename}}',
# #         item2='{{fileDirname}}',
# #         item3='abc_{{fileBasenameNoExtension }}')
#
# # 则可以通过参数 use_predefined_variables 实现自动替换预定义变量功能
# # cfg_file 文件名是 h.py
# cfg_file = ""
# cfg = Config.fromfile(cfg_file, use_predefined_variables=True)
# print(cfg.pretty_text)
#
# # 输出
# item1 = 'h.py'
# item2 = 'config 文件路径'
# item3 = 'abc_h'

# 该参数主要用途是自动替换 Config 类中已经预定义好的变量模板为真实值，在某些场合有用，
# 目前只支持 4 个变量：fileDirname、fileBasename、fileBasenameNoExtension 和 fileExtname
###################################################

## 4. 导入自定义模块
# Config.fromfile 函数除了有 filename 和 use_predefined_variables 参数外，还有 import_custom_modules，默认是 True。
# 一个典型用法是：
# 假设你在 mmdet 中新增了自定义模型 MobileNet ，
# 你需要在 mmdet/models/backbones/__init__.py 里面加入如下代码，否则在调用时候会提示该模块没有被注册进去：
# from .mobilenet import MobileNet
# 但是上述做法在某些场景下会比较麻烦。例如该模块处于非常深的层级，那么就需要逐层修改 __init__.py.

# # .py 文件里面存储如下内容
# custom_imports = dict(
#     imports=['mmdet.models.backbones.mobilenet'],
#     allow_failed_imports=False)
#
# # 自动导入 mmdet.models.backbones.mobilenet
# Config.fromfile(cfg_file, import_custom_modules=True)
#####################################################

## 5. 合并多个配置文件
# 1）从 base 文件中合并
# mmcv.Config 支持基于单个 base 配置文件，然后合并和汇总其余配置。

# cfg.py 中的内容
# _base_ = './base.py'
# ...

# 用法
cfg = Config.fromfile('./cfg.py')

# 2) 从多个 base 文件中合并
# 只需将上述非 base 配置文件中将类似 _base_ = './base.py'改成 _base_ = ['./base.py',...] 即可。
# base 文件的 key 是不允许改的，必须是 _base_ ，否则程序不知道哪个字段才是 base
# 多个 base 以 list 方式并行构建模式下，不允许多个 base 文件中有相同字段，程序会报 Duplicate Key Error，因为此时不知道以哪个配置为主。

# 3）合并字典到配置
# 通过 cfg.merge_from_dict 函数接口可以实现对字典内容进行合并

input_options = {'item2.a': 1, 'item2.b': 0.1, 'item3': False}
cfg.merge_from_dict(input_options)

# 4) allow_list_keys 参数
# cfg.merge_from_dict(input_options, allow_list_keys=True)

# 5）允许删除特定内容 -- 参数 _delete_ = True
# 假设 base.py 中有如下 bbox 回归损失函数配置：
# loss_bbox=dict(type='L1Loss', loss_weight=1.0，其他参数)
# 若此时需要更换 Loss Function:
loss_bbox=dict(
    _delete_=True,  ## 关键参数，可以忽略 base 相关配置，直接采用新配置
    type='IoULoss',
    eps=1e-6,
    loss_weight=1.0,
    reduction='none')
#####################################################

## 6. pretty_text 和 dump
# pretty_text 函数可以将字典内容按照 PEP8 格式打印，输出结构更加清晰。

# 直接打印
print(cfg._cfg_dict)
# 输出
# {'item1': [1, 2], 'item2': {'a': 1, 'b': 0.1}, 'item3': False, 'item4': 'test'}

print(cfg.pretty_text)
# 输出
# item1 = [1, 2]
# item2 = dict(a=1, b=0.1)
# item3 = False
# item4 = 'test'