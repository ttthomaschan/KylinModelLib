# -*- coding:utf-8 -*-
"""
# @file name  : decorator.py
# @author     : JLChen
# @date       : 2021-04
# @brief      : 装饰器演示
"""
'''
Note:
装饰器工作逻辑：当编译器遇到 @xxx 时，就会将原函数进行修饰。可以理解为原函数在编译阶段就被“替换”为装饰器返回的那个函数[decorator()]!

@clock
def foo(seconds):
等价于 ==>

def foo(seconds):
    pass
foo = clock(foo)
'''

import time

def foo(seconds=2.1):
    time.sleep(seconds)

def clock_func(your_func):
    t0 = time.perf_counter()
    results = your_func()
    elapsed = time.perf_counter() - t0
    print("Running time: {:.8f}".format(elapsed))
    return results


def clock(func):
    '''
    装饰器就是一个函数，它接收函数（原函数），返回函数（新函数）
    :param func:
    :return:
    '''
    def decorator():
        t0 = time.perf_counter()
        results = func()
        elapsed = time.perf_counter() - t0
        print("Running time: {:.8f}".format(elapsed))
        return results, elapsed   ### 原函数 foo_decorator() 已经被装饰为 此装饰函数 decorator(), 因此返回值也跟此函数相同。

    return decorator

@clock
def foo_decorator(seconds=2.2):
    time.sleep(seconds)


if __name__ == "__main__":

    ## stage 1: 面向过程
    # t0 = time.perf_counter()
    # foo()
    # elapsed = time.perf_counter() - t0
    # print("Running time: {:.8f}".format(elapsed))


    ## stage 2: 面向对象，抽象形成函数
    # results = clock_func(your_func=foo)
    # print(results)

    ## stage 3: Pythonic, 装饰器装饰原函数，原函数已具备计算运算时间功能
    res = foo_decorator()
    print("Done!")
