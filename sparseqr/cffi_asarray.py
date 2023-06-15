from __future__ import print_function, division, absolute_import

import numpy

# Create the dictionary mapping ctypes to numpy dtypes.
ctype_to_dtype = {}

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2 ** log_bytes))
        dtype = '%s%d' % (prefix[0], 2 ** log_bytes)
        # print( ctype )
        # print( dtype )
        ctype_to_dtype[ctype] = numpy.dtype(dtype)

# Floating point types
ctype_to_dtype['float'] = numpy.dtype('f4')
ctype_to_dtype['double'] = numpy.dtype('f8')


# print( ctype2dtype )

def as_array(ffi, ptr, length):
    # Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof(ptr).item)
    # print( T )
    # print( ffi.sizeof( T ) )

    if T not in ctype_to_dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % T)

    return numpy.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(T)), ctype_to_dtype[T])


def test():
    from cffi import FFI
    ffi = FFI()

    N = 10
    ptr = ffi.new("float[]", N)

    arr = as_array(ffi, ptr, N)
    arr[:] = numpy.arange(N)

    for i in range(N):
        print(arr[i], ptr[i])


if __name__ == '__main__':
    test()
