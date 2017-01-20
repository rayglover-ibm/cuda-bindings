import cufoo

print("[py] version={}".format(cufoo.version()), flush=True)

x = cufoo.add(3, 5)
assert(x == 8)

print("[py] {}".format(x), flush=True)
