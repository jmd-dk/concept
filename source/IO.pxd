from species cimport Particles

cdef Particles load(str filename)

cdef Particles save(Particles particles, str filename)
