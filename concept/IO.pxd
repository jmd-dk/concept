from species cimport Particles

cdef class Gadget_snapshot:
    cdef:
        # Data attributes
        dict header
        Particles particles
        unsigned int[::1] ID
        # Methods (f is an io.TextIOWrapper instance)
        populate(self, Particles particles, double a)
        save(self, str filename)
        load(self, str filename, bint write_msg=*)
        read(self, object f, str fmt)  
        size_t new_block(self, object f, size_t offset)

cdef:
    save(Particles particles, double a, str filename)
    Particles load(str filename)
    save_standard(Particles particles, double a, str filename)
    Particles load_standard(str filename)
    save_gadget(Particles particles, double a, str filename)
    Particles load_gadget(str filename)
