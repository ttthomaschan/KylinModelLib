'''
大型项目中有很多不同的函数和类需要组装使用，把函数或类整合到一个字典（key是函数名、类名，value是对应的函数对象或者类）。
将一个函数或类放入字典中，成为对该函数或类完成注册。
'''

'''
# Method_1： 最简单的注册方式（不优雅且麻烦）
register = {}

def func1():
    pass

f = lambda x: x

class cls1(object):
    pass

register[func1.__name__] = func1
register[f.__name__] = f
register[cls1.__name__] = cls1
----------------------------------------
'''

# Method_2: 直接在装饰器中完成对传入的函数或者类的注册
class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    # 只需要判断输入的参数是否可调用对象，然后做相应处理。谨记装饰器函数最终返回的都是函数对象。
    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                print(f"warning:\033[33m{value.__name__} has been registered before.\033[0m")
            self._dict[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x : add_register_item(target, x)  ## 如果不可调用，说明额外说明了注册的可调用对象的名字

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

#######################################################################################
if __name__ == "__main__":
    register_functions = Register()

    @register_functions.register
    def add(a, b):
        return a+b

    @register_functions.register("mymultiply")
    def multiply(a, b):
        return a*b

    @register_functions  ## 直接调用了 __call__()函数
    def minus(a, b):
        return a-b

    print(register_functions.items())