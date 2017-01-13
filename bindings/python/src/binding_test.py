import cufoopy

print("[py] version={}".format(cufoopy.version()), flush=True)

x = cufoopy.add(3, 5)
assert(x == 8)

print("[py] {}".format(x), flush=True)
