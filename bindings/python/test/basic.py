import libsrpy

print("[py] version = {}".format(libsrpy.version()), flush=True)

x = libsrpy.add(3, 5)
assert(x == 8)

print("[py] {}".format(x), flush=True)
