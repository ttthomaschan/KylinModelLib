from .builder import CLASS

@CLASS.register_module()
class test(object):
    def __init__(self, color):
        self.color = color

    def printf(self):
        print(self.color)