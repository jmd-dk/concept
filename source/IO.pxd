from species cimport Particles


cdef class Gadget_snapshot:
    # Data attributes
    cdef size_t offset
    cdef dict HEAD
    cdef object f  # This is actually an io.TextIOWrapper instance
    cdef Particles particles
    # Methods
    cdef load(self, str filename)
    cdef read(self, str fmt)
    cdef new_block(self)

cdef Particles load(str filename)

cdef Particles save(Particles particles, str filename)
