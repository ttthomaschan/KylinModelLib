from registry import CLASS

cfg = {"type":'test', "color":"red"}
test = CLASS.build(cfg)
test.printf()