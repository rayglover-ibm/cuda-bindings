import mwe

print("[py] version={}".format(mwe.version()))

def test_add():
    a = 3
    b = 5

    c = mwe.add(a, b)
    assert(c == 8)

    print("[py] {}".format(c))


def test_add_all():
    import numpy

    a = numpy.array([1,2,3,4], dtype=numpy.int32)
    b = numpy.array([5,6,7,8], dtype=numpy.int32)

    c = mwe.add(a, b)
    assert(numpy.array_equal(c, a + b))

    print("[py] {}".format(c))


test_add()
test_add_all()