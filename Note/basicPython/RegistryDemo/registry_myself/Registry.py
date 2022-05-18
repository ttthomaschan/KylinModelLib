'''Registry类实现'''

# ###（1）最简实现 ###
# # 需要一个全局变量作注册表
# _module_dict = dict()
#
# # 定义装饰器函数
# def register_module(name):
#     def _register(cls):
#         _module_dict[name] = cls
#         return cls
#
#     return _register
#
# # 装饰器用法
# @register_module('one_class')
# class OneTest(object):
#     pass
#
# @register_module('two_class')
# class TwoTest(object):
#     pass

# #####################################
# if __name__ == '__main__':
#     one_test = _module_dict['one_class']()
#     print(one_test)


# ###（2）实现无需传入参数，自动根据类名初始化类 ###
# _module_dict = dict()
#
# # 定义装饰器函数
# def register_module(module_name=None):
#     def _register(cls):
#         name = module_name
#         # 如果 module_name 没有给，则自动获取
#         if module_name is None:
#             name = cls.__name__
#         _module_dict[name] = cls
#         return cls
#
#     return _register
#
# @register_module('one_class')
# class OneTest(object):
#     pass
#
# @register_module()
# class TwoTest(object):
#     pass
#
# ###########################################
# if __name__ == '__main__':
#     print(_module_dict)
#     one_test = _module_dict['one_class']
#     print(one_test)
#     two_test = _module_dict['TwoTest']
#     print(two_test)


# ###（3）实现重名注册强制报错功能（新增参数force） ###
# _module_dict = dict()
# def register_module(module_name=None, force=False):
#     def _register(cls):
#         name = module_name
#         if module_name is None:
#             name = cls.__name__
#         if not force and name in _module_dict:
#             raise KeyError(f'{module_name} is already registered in {name}')
#         _module_dict[name] = cls
#         return cls
#
#     return _register

### mmcv.Registry类 完整实现 ###
import inspect
class Registry:
    def __init__(self, name):
        # 可实现注册类细分功能
        self._name = name
        # 内部核心内容，维护所有的已经注册好的 class
        self._module_dict = dict()

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class,'
                            f'but got {type(module_class)}')
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')

        # 最核心代码
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        if module is not None:
            # 如果已经是 module，直接增加到字典中即可
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # 最标准用法 @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register

# 在 MMCV 中所有的类实例化都是通过 build_from_cfg 函数实现
# 实质是给定 module_name，然后从 self._module_dict 提取即可。
def build_from_cfg(cfg, registry, default_args=None):
    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type') # 注册 str 类名
    if is_str(obj_type):
        obj_cls = registry.get(obj_type) # --> 等价于 self._module_dict[obj_type]
        if obj_type is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')

    # 如果已经实例化，那就直接返回
    elif inspect.isclass(obj_type):
        obj_cls = obj_type

    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

    # 最终初始化模块类，并且返回，就完成了一个类的实例化过程
    return obj_cls(**args)

#######################################################3
'''调用示例'''
CONVERTERS = Registry('converter')

@CONVERTERS._register_module()
class Converter1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

converter_cfg = dict(type='Converter1', a=a_value, b=b_value)
converter = build_from_cfg(converter_cfg, CONVERTERS)


