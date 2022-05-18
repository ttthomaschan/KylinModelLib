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


