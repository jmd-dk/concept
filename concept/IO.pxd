from species cimport Particles

cdef class Gadget_snapshot:
    # Data attributes
    cdef dict header
    cdef Particles particles
    cdef unsigned int[::1] ID
    # Methods
    cdef populate(self, Particles particles, double a)
    cdef save(self, str filename)
    cdef load(self, str filename, bint write_msg=*)
    cdef read(self, object f, str fmt)  # f is an io.TextIOWrapper instance
    cdef size_t new_block(self, object f, size_t offset)  # f is an io.TextIOWrapper instance


cdef save_standard(Particles particles, double a, str filename)
cdef Particles load_standard(str filename)

cdef save_gadget(Particles particles, double a, str filename)
cdef Particles load_gadget(str filename)

cdef save(Particles particles, double a, str filename)
cdef Particles load(str filename)
