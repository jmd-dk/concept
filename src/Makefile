# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2021 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This is the build makefile for the CO𝘕CEPT code.
# This makefile accepts the options
#   native_optimizations
#   no_lto
#   no_optimizations
#   no_update_timestamps
#   unsafe_building
# all of which may be either 'True' or 'False',
# with unset values being equivalent to 'False'.

# Use the Bash shell
SHELL = /usr/bin/env bash



##############################
# Specification of filenames #
##############################
# Modules which should be cythonised and compiled
pyfiles =         \
    analysis      \
    commons       \
    communication \
    ewald         \
    fluid         \
    graphics      \
    gravity       \
    integration   \
    interactions  \
    linear        \
    main          \
    mesh          \
    snapshot      \
    species       \
    utilities     \

# Filename of the module holding common definitions
commons = commons.py
# Filename of the .pyx preprocessor script
pyxpp = pyxpp.py
# Filename of the temporary .pyx file storing the custom types
types = .types.pyx



#####################
# Main build target #
#####################
# This builds all the Python modules.
# When finished, time stamps are updated so that all dependencies are
# synchronized. This is needed because .pxd files are only updated
# when their contents actually change. That is, `make pyfile.pxd` does
# not necessarily update the timestamp of pyfile.pxd, even though its
# rule is run. This makes .c (and therefore .o and .so) files
# dependent on such .pxd files not up-to-date. The types file is also
# only updated when its content actually changes.
# Once all timestamps are updated, further update just those of the
# compiled module files, so that all of these are newer than any of
# source files.
all_build: $(addsuffix .so, $(pyfiles))
	@if [ "$(no_update_timestamps)" != "True" ]; then   \
	    no_update_timestamps=True $(MAKE) -s -t $@;     \
	    sleep 1;                                        \
	    touch $^;                                       \
	fi;
.PHONY: all_build

# Import variables from the main makefile
# (must be placed below main target).
include ../Makefile
# Set heading text
building = modules
ifneq ($(build),)
    ifneq ($(build),$(build_dir))
        ifneq ($(build_dir),.)
            building += within $(call sensible_path,$(build),$(concept_dir))
        endif
     endif
endif



########################################
# Settings for compilation and linking #
########################################
# Process ID of this make process,
# used for unique naming of temporary files.
pid=$(shell echo $${PPID})
# The python configuration script
python_config = $(shell readlink -f $$(which $(python)))-config
# Options passed when cythonising .pyx files
python_version_major = $(shell $(python) -B -c "import sys; print(sys.version_info.major)")
cythonflags = -$(python_version_major) -D -a

# Includes
fftw_incl   = -I$(fftw_dir)/include
gsl_incl    = -I$(gsl_dir)/include
mpi_incl    = -I$(mpi_includedir)
python_incl = $(shell $(python_config) --includes)
includes    = $(fftw_incl) $(gsl_incl) $(mpi_incl) $(python_incl)
# Compiler options
CC = $(mpicc)
compiler = $(shell                                           \
    info="$$($(CC) --version 2>/dev/null | head -n 1 || :)"; \
    if echo "$${info}" | grep -i icc >/dev/null; then        \
        echo "icc";                                          \
    elif echo "$${info}" | grep -i intel >/dev/null; then    \
        echo "icc";                                          \
    elif echo "$${info}" | grep -i gcc >/dev/null; then      \
        echo "gcc";                                          \
    elif echo "$${info}" | grep -i clang >/dev/null; then    \
        echo "clang";                                        \
    else                                                     \
        echo "gcc";                                          \
    fi                                                       \
)
python_cflags = $(filter-out              \
    $(addprefix -W,$(unwanted_warnings))  \
    -g -g0 -g1 -g2 -g3                    \
    -O -O0 -O1 -O2 -O3 -O4 -Og -Os -Ofast \
    -fwrapv                               \
    ,                                     \
    $(shell $(python_config) --cflags)    \
)
unwanted_warnings_general += \
    uninitialized            \
    unknown-warning-option   \

ifeq ($(compiler),icc)
    unwanted_warnings_icc +=   111  # Unreachable statement
    unwanted_warnings_icc += 10006  # Ignoring unknown option
    unwanted_warnings_icc += 10148  # Option not supported
    unwanted_warnings_icc += 11074  # Inlining inhibited
    unwanted_warnings_icc += 11076  # How to get full report
endif
warnings =                                          \
    -Wall                                           \
    -Wextra                                         \
    $(addprefix -Wno-,$(unwanted_warnings_general)) \
    $(addprefix -wd,$(unwanted_warnings_icc))       \

other_cflags = \
    -std=c99   \
    -pthread   \
    -fPIC      \

# Optimization options
no_optimizations_flag = --no-optimizations
ifneq ($(no_optimizations),True)
    no_optimizations_flag =
    # General optimizations
    optimizations        += -DNDEBUG -O3 -funroll-loops
    optimizations_linker += -O3
    # Python/Cython optimizations
    optimizations += -DCYTHON_WITHOUT_ASSERTIONS
    # Floating-point optimizations
    ifeq ($(compiler),gcc)
        optimizations += -ffast-math
    else ifeq ($(compiler),clang)
        optimizations += -ffast-math
    else ifeq ($(compiler),icc)
        optimizations += -fp-model fast=2
    endif
    # Native optimizations
    ifeq ($(native_optimizations),True)
        optimizations += -march=native
    endif
    # Link time optimizations
    ifneq ($(no_lto),True)
        check_fname = .check_lto_$(pid)
        lto_warning = $(shell                                                    \
            echo "int main(void){ return 0; }" > $(check_fname).c;               \
            $(CC) -flto -c -o $(check_fname).o $(check_fname).c 2>&1 | grep lto; \
            $(CC) $(check_fname).o -o $(check_fname) -flto      2>&1 | grep lto; \
            ./$(check_fname) 2>/dev/null || echo "lto error";                    \
            $(RM) $(check_fname)*;                                               \
        )
        ifeq ("$(lto_warning)","")
            optimizations        += -flto
            optimizations_linker += -flto
        endif
    endif
endif
CFLAGS += $(call unique, $(call sensible_path, \
    $(python_cflags)                           \
    $(other_cflags)                            \
    $(optimizations)                           \
    $(warnings)                                \
    $(includes)                                \
))

# Libraries to link
fftw_libs = -L$(fftw_dir)/lib -Wl,-rpath=$(fftw_dir)/lib -lfftw3_mpi -lfftw3
ifneq ("$(wildcard $(blas_dir)/lib/libopenblas.*)","")
    # OpenBLAS found
    gsl_blas_libs = -L$(blas_dir)/lib -Wl,-rpath=$(blas_dir)/lib -lopenblas
else ifneq ("$(wildcard $(blas_dir)/lib/libblas.*)","")
    # Some BLAS found
    gsl_blas_libs = -L$(blas_dir)/lib -Wl,-rpath=$(blas_dir)/lib -lblas
else
    # No BLAS found
    gsl_blas_libs = -lgslcblas
endif
gsl_libs = -L$(gsl_dir)/lib -Wl,-rpath=$(gsl_dir)/lib -lgsl $(gsl_blas_libs) -lm
mpi_libs = -L$(mpi_libdir) -Wl,-rpath=$(mpi_libdir) -lmpi
ifneq ("$(wildcard $(mpi_symlinkdir))","")
    mpi_libs += -L$(mpi_symlinkdir) -Wl,-rpath=$(mpi_symlinkdir)
endif
python_libdir = $(shell $(python) -B -c "pass; \
    import sysconfig;                          \
    print(sysconfig.get_config_var('LIBDIR')); \
")
python_libs = -L$(python_libdir) -Wl,-rpath=$(python_libdir) $(shell $(python_config) --libs)
LDLIBS += $(call unique, $(call sensible_path, \
    $(fftw_libs)                               \
    $(gsl_libs)                                \
    $(mpi_libs)                                \
    $(python_libs)                             \
))

# Linker options
python_ldflags = $(shell $(python_config) --ldflags)
comma = ,
LDFLAGS += $(call unique, $(call sensible_path,      \
    $(python_ldflags)                                \
    $(filter-out -ffast-math,$(optimizations))       \
    $(addprefix -Wl$(comma),$(optimizations_linker)) \
    $(warnings)                                      \
))



#################
# Build targets #
#################
# The below recipes are responsible for the
# .py --> ((.pyx --> types) --> .pxd) --> (.c + .html) --> .o --> .so
# build chain.

# Link object files into shared object Python modules
$(addsuffix .so, $(pyfiles)): %.so: %.o
	@$(call heading)
	$(CC) $< -o $@ $(call unique, $(LDFLAGS) -shared $(LDLIBS))

# Compile C source files into object files
$(addsuffix .o, $(pyfiles)): %.o: %.c
	@$(call heading)
	$(CC) $(CFLAGS) -c -o $@ $<

# Cythonise .pyx and .pxd files into C source files
# (and also produce the annotated .html file).
$(addsuffix .c, $(pyfiles)): %.c: %.pyx %.pxd
	@$(call heading)
	$(python) -m cython $(cythonflags) $<
# Same as above (for explicit builds of html)
$(addsuffix .html, $(pyfiles)): %.html: %.pyx %.pxd
	$(python) -m cython $(cythonflags) $<

# Write .pxd files from .pyx files together with the types file
# using the pyxpp script.
$(addsuffix .pxd, $(pyfiles)): %.pxd: %.pyx $(types)
	@$(call heading)
	$(mpiexec) -n 1 $(python) -B $(pyxpp) $< $(commons) $(types)

# Write the types file from the .pyx files using the pyxpp script
$(types): $(addsuffix .pyx, $(pyfiles))
	@$(call heading)
	$(mpiexec) -n 1 $(python) -B $(pyxpp) $@ $(commons) $@ $^

# Write .pyx files from .py files using the pyxpp script
$(addsuffix .pyx, $(pyfiles)): %.pyx: %.py
	@$(call heading)
	$(mpiexec) -n 1 $(python) -B $(pyxpp) $< $(commons) $(no_optimizations_flag)



###########################
# Additional dependencies #
###########################
# Additional target dependencies
$(foreach ext,c html,$(addsuffix .$(ext), mesh)): fft.c
# Target dependencies which strictly speaking should be
# taken into account, but can be ignored using the --unsafe-building flag.
ifneq ($(unsafe_building),True)
    # Use of inline iterators defined in other modules
    # makes the .pyx files depend on the .py file
    # of the module implementing the iterator.
    gravity.pyx:      interactions.py  # particle_particle()
    analysis.pyx:     mesh.py          # fourier_loop()
    interactions.pyx: mesh.py          # fourier_loop()
    linear.pyx:       mesh.py          # fourier_loop()
    # Because modules cimport stuff from each other, some .c
    # (and therefore also .html) files depend on the .pxd files of
    # other modules.
    $(foreach ext,c html,$(addsuffix .$(ext), $(pyfiles))): $(addsuffix .pxd, $(pyfiles))
    # Every module cimports everything from the commons module
    $(foreach ext,pyx pxd,$(addsuffix .$(ext), $(pyfiles))): $(commons)
    $(types): $(commons)
    # The pyxpp script is used to create .pyx files,
    # .pxd files, .c files and the types file.
    $(foreach ext,pyx pxd c,$(addsuffix .$(ext), $(pyfiles))): $(pyxpp)
    $(types): $(pyxpp)
    # Let the bottom of the build chain depend on this makefile
    # as well as on the main makefile.
    $(addsuffix .pyx, $(pyfiles)): $(MAKEFILE_LIST)
endif



#################
# Other targets #
#################
# For checking whether the NAN and INFINITY macros are supported.
# Each supported macro will be printed.
check_fname = .check_float_literals_$(pid)
check_float_literals:
	@printf "\
	    #include <stdio.h>\n\
	    #include <math.h>\n\
	    int main(void){\n\
	      #ifdef NAN\n\
	        printf(\"NAN\\\\n\");\n\
	      #endif\n\
	      #ifdef INFINITY\n\
	        printf(\"INFINITY\\\\n\");\n\
	      #endif\n\
	      return 0;\n\
	    }\
	" > $(check_fname).c
	@$(CC) $(CFLAGS) -c -o $(check_fname).o $(check_fname).c
	@$(CC) $(check_fname).o -o $(check_fname) \
	    $(call unique, $(LDFLAGS) $(LDLIBS))
	@./$(check_fname)
	@$(RM) $(check_fname)*
.PHONY: check_float_literals



##################
# Cleanup target #
##################
# Remove all compile files
clean_build:
	$(RM) -r $(foreach ext,pyc pyx pxd c html o so so_,\
	    $(addsuffix .$(ext), $(pyfiles))) $(types) __pycache__
.PHONY: clean_build

