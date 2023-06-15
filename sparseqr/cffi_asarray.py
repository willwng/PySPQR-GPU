from __future__ import print_function, division, absolute_import

import numpy

# Create the dictionary mapping ctypes to numpy dtypes.
ctype_to_dtype = {}

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2 ** log_bytes))
        dtype = '%s%d' % (prefix[0], 2 ** log_bytes)
        ctype_to_dtype[ctype] = numpy.dtype(dtype)

# Floating point types
ctype_to_dtype['float'] = numpy.dtype('f4')
ctype_to_dtype['double'] = numpy.dtype('f8')


def as_array(ffi, ptr, length):
    # Get the canonical C type of the elements of ptr as a string.
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    if c_type not in ctype_to_dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % c_type)

    return numpy.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(c_type)), ctype_to_dtype[c_type])
