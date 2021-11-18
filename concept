#!/usr/bin/env bash

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



# This script runs the CO𝘕CEPT code.
# Run the script with the -h option to get help.

# Unless this file is being sourced,
# automatically export all variables when set.
being_sourced="True"
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    being_sourced="False"
fi
if [ "${being_sourced}" == "False" ]; then
    set -a
fi

# If this file is being sourced, backups of 'this_file' and 'this_dir'
# are needed not to alter the values of these variables.
this_file_backup="${this_file}"
this_dir_backup="${this_dir}"

# Absolute paths to this file and its directory
this_file="$(readlink -f "${BASH_SOURCE[0]}")"
this_dir="$(dirname "${this_file}")"

# The user's current working directory
if [ -z "${workdir}" ]; then
    workdir="$(pwd)"
fi

# For the terminal to be able to print Unicode characters correctly,
# we need to use a UTF-8 locale.
set_locale() {
    # This function will set the locale through the LC_ALL and LANG
    # environment variables. We want to use a supported UTF-8 locale.
    # The preference order is as follows:
    #   en_US.UTF-8
    #   en_*.UTF-8
    #   C.UTF-8
    #   POSIX.UTF-8
    #   *.UTF-8
    # We consider the suffix (UTF-8) valid regardless of the case and
    # presence of the dash.
    # Get all available locals.
    locales="$(locale -a 2>/dev/null || :)"
    if [ -z "${locales}" ]; then
        return
    fi
    # Look for available UTF-8 locale
    for prefix in "en_US" "en_*" "C" "POSIX" "*"; do
        for suffix in "UTF-8" "UTF8" "utf-8" "utf8"; do
            pattern="${prefix}.${suffix}"
            for loc in ${locales}; do
                if [[ "${loc}" == ${pattern} ]]; then
                    export LC_ALL="${loc}"
                    export LANG="${loc}"
                    return
                fi
            done
        done
    done
}
set_locale
# Set the terminal if unset or broken
if [ -z "${TERM}" ] || [ "${TERM}" == "dumb" ]; then
    export TERM="linux"
fi

# ANSI/VT100 escape sequences
esc="\x1b"
# Text formatting
esc_normal="${esc}[0m"
esc_bold="${esc}[1m"
esc_italic="${esc}[3m"
esc_no_italic="${esc}[23m"
esc_red="${esc}[91m"
# The name of the program, nicely typeset
if [ -z "${esc_concept}" ]; then
    esc_concept="CO${esc_italic}N${esc_no_italic}CEPT"
else
    esc_concept="${esc_concept//\$\{esc_italic\}/${esc_italic}}"
    esc_concept="${esc_concept//\$\{esc_no_italic\}/${esc_no_italic}}"
fi

# Enable extended globbing, used for negative wildcards !(...)
shopt -s extglob

# Load paths from the .path file
curr="${this_dir}"
while :; do
    if [ -f "${curr}/.path" ]; then
        source "${curr}/.path"
        break
    fi
    if [ "${curr}" == "/" ]; then
        # Print out error message and exit
        printf "${esc_bold}${esc_red}Could not find the .path file!${esc_normal}\n" >&2
        exit 1
    fi
    curr="$(dirname "${curr}")"
done

# Load environment variables from the .env file
env_backup_vars=(mpi_executor make_jobs)
env_backup_vals=()
for env_var in "${env_backup_vars[@]}"; do
    eval "env_val=\"\${${env_var}}\""
    env_backup_vals=("${env_backup_vals[@]}" "${env_val}")
done
source "${env}"
for ((index=0; index<${#env_backup_vars[@]}; index+=1)); do
    env_val="${env_backup_vals[${index}]}"
    if [ -n "${env_val}" ]; then
        env_var="${env_backup_vars[${index}]}"
        eval "${env_var}=\"${env_val}\""
        env_backups=("${env_backups[@]}" "${env_val}")
    fi
done

# Add the src directory to searched paths
# when importing modules in Python.
export PYTHONPATH="${src_dir}:${PYTHONPATH}"

# Disable Python hash randomization (salting) for reproducibility
export PYTHONHASHSEED=0

# Some Python packages may need access to libraries at runtime
for lib in "blas" "fftw" "gsl" "hdf5" "ncurses" "python" "zlib"; do
    eval "lib_dir=\"\${${lib}_dir}\""
    if [ -z "${lib_dir}" ]; then
        continue
    fi
    lib_dir="${lib_dir}/lib"
    if [ ! -d "${lib_dir}" ]; then
        continue
    fi
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH}"
done

# A bug in HDF5 makes the code crash on certain file systems.
# A workaround is to set an environment variable as below.
export HDF5_USE_FILE_LOCKING=FALSE

# The MPI executables and libraries should be
# on the PATH and LD_LIBRARY_PATH, respectively.
export PATH="${mpi_bindir}:${PATH}"
export LD_LIBRARY_PATH="${mpi_libdir}:${LD_LIBRARY_PATH}"
# Additional symlinks to MPI libraries might be placed in mpi_symlinkdir
if [ -d "${mpi_symlinkdir}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${mpi_symlinkdir}"
    # If the special symlink named "ld_preload.so" is present
    # in mpi_symlinkdir, we should include this symlink in LD_PRELOAD.
    if [ -f "${mpi_symlinkdir}/ld_preload.so" ]; then
        export LD_PRELOAD="${LD_PRELOAD} ${mpi_symlinkdir}/ld_preload.so"
    fi
fi

# The time before any computation begins.
# This time is saved both in seconds after the Unix epoch
# and in a human readable format.
start_time_epoch="$("${python}" -B -c "
import datetime
print(datetime.datetime.now().timestamp())
")"
start_time_human="$("${python}" -B -c "
import datetime
print(str(datetime.datetime.fromtimestamp(${start_time_epoch}))[:-3])
")"
# Further transform the human readable version
start_time_human_sec="${start_time_human%.*}"
start_time_human_nosep="${start_time_human//-/}"
start_time_human_nosep="${start_time_human_nosep// /}"
start_time_human_nosep="${start_time_human_nosep//:/}"
start_time_human_nosep="${start_time_human_nosep//./}"

# Check whether this script is run locally or remotely via ssh
ssh="True"
if [ -z "${SSH_CLIENT}" ] && [ -z "${SSH_TTY}" ]; then
    ssh="False"
fi

# Function for printing coloured messages
colorprint() {
    # Arguments: Message, colour
    "${python}" -B -c "
import sys
from blessings import Terminal
terminal = Terminal(force_styling=True)
print(terminal.bold_${2}('${1}'), file=(sys.stderr if '${2}' == 'red' else sys.stdout))"
}

# Function for printing out a nice CO𝘕CEPT logo
print_logo() {
    logo='
   ____     ____             __  ____    _____   ____   _____
  / __ \   / __ \     /\    / / / __ \  |  ___| |  _ \ |_   _|
 | /  \_| | /  \ |   /  \  / / | /  \_| | |__   | |_) |  | |
 ||    _  ||    ||  / /\ \/ /  ||    _  |  __|  |  __/   | |
 | \__/ | | \__/ | / /  \  /   | \__/ | | |___  | |      | |
  \____/   \____/ /_/    \/     \____/  |_____| |_|      |_|
'
    # Plot the logo via Python's matplotlib.
    # While the colour of the 𝘕 is fixed, the colour used for the rest
    # is determined from the timestamp of this file.
    # This uses up the 16th and 17th colour of the terminal.
    "${python}" -B -c "
import sys, matplotlib
# Generate colour based on input number
color_for_N = 'darkorange'
brightness = lambda color: 0.241*color[0]**2 + 0.691*color[1]**2 + 0.068*color[2]**2
blim = (0.3, 0.97)
color_choices = [
    matplotlib.colors.ColorConverter().to_rgb(color)
    for name, color in
    matplotlib.colors.CSS4_COLORS.items()
    if name != color_for_N
]
color_choices = [color for color in color_choices if blim[0] < brightness(color) < blim[1]]
colors = (
    color_choices[int(sys.argv[1]) % len(color_choices)],
    color_for_N,
)
# Apply colormap
for i, color in enumerate(colors):
    colorhex = matplotlib.colors.rgb2hex(color)
    print('\\x1b]4;{};rgb:{}/{}/{}\\x1b\\\\'
        .format(16 + i, colorhex[1:3], colorhex[3:5], colorhex[5:]), end='')
# Construct the coloured logo
logo=r'''${logo}'''
logo = logo[1:-1]
rows = logo.split('\\n')
ANSI = []
for row in rows:
    ANSI.append('\\x1b[1m')
    for i, c in enumerate(row):
        if i in {0, 18, 31}:
            colornumber = 17 if i == 18 else 16
            ANSI.append(f'\\x1b[38;5;{colornumber}m')
        ANSI.append(c)
    ANSI.append('\\x1b[0m\\n')
# Print the ANSI image
print(''.join(ANSI), end='', flush=True)
" \
    $(stat -c '%Y' "${this_file}")
}
# Print out the logo the first time an execution reaches this point
if [ "${logo_printed}" != "True" ] && [ "${being_sourced}" == "False" ]; then
    print_logo
    export logo_printed="True"
fi

# Function for converting paths to absolute paths
absolute_path() {
    # Arguments: Path, [working directory]
    local path="${1}"
    local current_dir="$(pwd)"
    if [[ "${path}" == /* ]]; then
        # The path is already absolute
        echo "${path}"
        return
    elif [ -n "${2}" ]; then
        # Explicit working directory supplied
        cd "${2}"
    elif [[ "${path}" == "./" ]] || [[ "${path}" == "../" ]]; then
        # The path is explicitly written as relative
        # to the user's current directory.
        cd "${workdir}"
    elif [ -f "${workdir}/${path}" ] || [ -d "${workdir}/${path}" ]; then
        # Path exists relative to the user's current working directory
        cd "${workdir}"
    elif [ -f "${concept_dir}/${path}" ] || [ -d "${concept_dir}/${path}" ]; then
        # Path exists relative to the concept directory
        cd "${concept_dir}"
    else
        # Assume the path is relative to the user's current working directory
        cd "${workdir}"
    fi
    # Place backslashes before spaces and dollar signs.
    # These are needed when expanding tilde, but they will not persist.
    path="${path// /\\ }"
    path="${path//$/\\$}"
    # Expand tilde
    eval path="${path}"
    # Convert to absolute path
    path=$(readlink -m "${path}")
    cd "${current_dir}"
    if [ -z "${path}" ]; then
        colorprint "Cannot convert \"${1}\" to an absolute path!" "red"
        exit 1
    fi
    # Print out result
    echo "${path}"
}

# Function for converting an absolute path to its "sensible" form.
# That is, this function returns the relative path with respect to the
# concept directory, if it is no more than one directory above the
# concept directory. Otherwise, return the absolute path back again.
sensible_path() {
    "${python}" -B -c "
path = '${1}'
from os.path import relpath
rel = relpath(path, '${concept_dir}')
print(path if rel.startswith('../../../') else rel)"
}

# Function which prints the absolute path of a given command.
# If the command is not an executable file on the PATH but instead a
# known function, the input command is printed as is. If the command
# cannot be found at all, nothing is printed and an exit code of 1
# is returned.
get_command() {
    command_name="${1}"
    # Use the type built-in to locate the command
    local path="$(type "${command_name}" 2>/dev/null | awk '{print $NF}')"
    if [[ "${path}" == "/"* ]]; then
        # The command is a path
        path="$(readlink -f "${path}")"
        echo "${path}"
        return 0
    elif [ -n "${path}" ]; then
        # The command exists as a function
        echo "${command_name}"
        return 0
    fi
    # The command does not exist
    return 1
}

# Function which prints a passed Bash array
# in the format of a Python list.
bash_array2python_list() {
    # Call like this: bash_array2python_list "${array[@]}"
    local list=''
    local element
    for element in "$@"; do
        # If element is a string, encapsulate it in quotation marks
        element=$("${python}" -B -c "
try:
    eval(\"${element}\")
    print(\"${element}\")
except:
    print('\"{}\"'.format(\"${element}\"))
")
        # Append element to list
        list="$(echo "${list}")${element}, "
    done
    list="[$(echo "${list}")]"
    echo "${list}"
}

# Function for recursively removing empty sub-directories
# within the tmp directory.
cleanup_empty_tmp() {
    local tmp_subdir
    tmp_subdir="$1"
    if [ -z "${tmp_subdir}" ]; then
        tmp_subdir="${tmp_dir}"
    fi
    if [ ! -d "${tmp_subdir}" ]; then
        return
    fi
    sleep_before_rm="0.01"
    for d in "${tmp_subdir}/"* "${tmp_subdir}/"*; do
        if [ ! -d "${d}" ]; then
            continue
        fi
        if [ -z "$(ls -A "${d}")" ]; then
            sleep ${sleep_before_rm}
            if [ -z "$(ls -A "${d}")" ]; then
                rm -rf "${d}" || :
            fi
        else
            cleanup_empty_tmp "${d}"
        fi
    done
    if [ ! -d "${tmp_dir}" ]; then
        return
    fi
    if [ "${tmp_subdir}" == "${tmp_dir}" ] && [ -z "$(ls -A "${tmp_dir}")" ]; then
        sleep ${sleep_before_rm}
        if [ -z "$(ls -A "${tmp_dir}")" ]; then
            rm -rf "${tmp_dir}" || :
        fi
    fi
}

# Determine the MPI implementation and version
if [ -f "${mpi_bindir}/ompi_info" ] || [ -f "${mpi_compilerdir}/ompi_info" ]; then
    # OpenMPI
    mpi_implementation="openmpi"
    mpi_version="$("${mpiexec}" --version 2>&1 | head -n 1 | awk '{print $NF}')"
elif [ -f "${mpi_bindir}/mpichversion" ] || [ -f "${mpi_compilerdir}/mpichversion" ]; then
    # MPICH or MVAPICH
    for f in "${mpi_bindir}/mpichversion" "${mpi_compilerdir}/mpichversion"; do
        if [ -f "${f}" ]; then
            line="$("${mpi_bindir}/mpichversion" 2>&1 | head -n 1)"
            mpi_implementation="$(echo "${line}" | awk '{print $1}' | tr '[:upper:]' '[:lower:]')"
            if [[ "${mpi_implementation}" == "mvapich"* ]]; then
                mpi_implementation="mvapich"
            else
                mpi_implementation="mpich"
            fi
            mpi_version="$(echo "${line}" | awk '{print $NF}')"
            break
        fi
    done
else
    # Unknown MPI implementation
    mpi_implementation="unknown"
    mpi_version="unknown"
fi

# MPI implementation specifics.
# While we always construct the contents of mpi_env,
# that of mpiexec_args is only constructed when this script is run
# as opposed to being sourced. We do this as mpiexec_args is only used
# when executing the program, and its construction requires test running
# mpiexec, which may cause trouble on some clusters.
if [ "${being_sourced}" == "False" ]; then
    mpiexec_args=""
fi
mpi_env=""
if [ "${mpi_implementation}" == "openmpi" ]; then
    # Disable aggregation of OpenMPI warnings
    mpi_env="${mpi_env}
export OMPI_MCA_orte_base_help_aggregate=0"
    # Disable OpenMPI warning about forking
    mpi_env="${mpi_env}
export OMPI_MCA_mpi_warn_on_fork=0"
    # By default, OpenMPI disallows any usage by the root user
    mpi_env="${mpi_env}
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    # In OpenMPI 3 and 4, oversubscription (having more MPI processes
    # than physical cores) is disallowed by default.
    mpi_version_major="${mpi_version:0:1}"
    if [ "${mpi_version_major}" -ge 3 2>/dev/null ]; then
        mpi_env="${mpi_env}
export OMPI_MCA_rmaps_base_oversubscribe=1"
    fi
    # In OpenMPI 4, InfiniBand ports are disabled by default in favour
    # of UCX. If this leads to an error about not initializing an
    # OpenFabrics device, we overwrite this default.
    if [ -z "${OMPI_MCA_btl_openib_allow_ib}" ] \
        && [ "${mpi_version_major}" -ge 4 2>/dev/null ]; then
        # As we are importing MPI4Py we ought to invoke Python via
        # mpiexec, though this may lead to trouble when sourcing.
        if [ "${being_sourced}" == "False" ]; then
            mpi_warning="$("${mpiexec}" -n 1 "${python}" -B -c '
import mpi4py.rc; mpi4py.rc.threads = False  # Do not use threads
from mpi4py import MPI
' 2>&1 || :)"
        else
            mpi_warning="$("${python}" -B -c '
import mpi4py.rc; mpi4py.rc.threads = False  # Do not use threads
from mpi4py import MPI
' 2>&1 || :)"
        fi
        if [ -n "${mpi_warning}" ]; then
            if echo "${mpi_warning}" | grep "error" | grep "OpenFabrics" >/dev/null; then
                if echo "${mpi_warning}" | grep "btl_openib_allow_ib" >/dev/null; then
                    mpi_env="${mpi_env}
export OMPI_MCA_btl_openib_allow_ib=1"
                fi
            fi
        fi
    fi
    # Disable automatic process binding/affinity, allowing OpenMP
    # threads to be assigned to cores in a one-to-one fashion.
    # This is off by default prior to OpenMPI version 1.7.
    # All of 1.7 -- 4.0 supports the --bind-to none option to mpiexec.
    # Here we add this option if it is understood.
    if [ "${being_sourced}" == "False" ]; then
        mpiexec_output="$("${mpiexec}" --bind-to none -n 1 echo "success" 2>&1 || :)"
        if [ "${mpiexec_output}" == "success" ]; then
            mpiexec_args="${mpiexec_args} --bind-to none"
        fi
    fi
    # For OpenMPI 1.8 -- 4.0 process binding can also be deactivated
    # using the following environment variable.
    mpi_env="${mpi_env}
export OMPI_MCA_hwloc_base_binding_policy=none"
elif [ "${mpi_implementation}" == "mpich" ]; then
    # Though MPICH should not use process binding by default,
    # it still supports a -bind-to none or --bind-to none option.
    # Here we check if this is so, and apply it.
    if [ "${being_sourced}" == "False" ]; then
        mpiexec_output="$("${mpiexec}" -bind-to none -n 1 echo "success" 2>&1 || :)"
        if [ "${mpiexec_output}" == "success" ]; then
            mpiexec_args="${mpiexec_args} -bind-to none"
        else
            mpiexec_output="$("${mpiexec}" --bind-to none -n 1 echo "success" 2>&1 || :)"
            if [ "${mpiexec_output}" == "success" ]; then
                mpiexec_args="${mpiexec_args} --bind-to none"
            fi
        fi
    fi
elif [ "${mpi_implementation}" == "mvapich" ]; then
    # Disable automatic process binding, allowing OpenMP threads to be
    # assigned to cores in a one-to-one fashion.
    mpi_env="${mpi_env}
export MV2_ENABLE_AFFINITY=0"
fi
if [ -n "${mpi_env}" ]; then
    eval "${mpi_env}"
    mpi_env="# Environment variables${mpi_env}"
fi

# Default values of command-line arguments
interactive_default="False"
jobname_default="concept"
local_default="False"
memory_default="-1"  # -1 implies unset
native_optimizations_default="False"
no_lto_default="False"
no_optimizations_default="False"
no_watching_default="False"
nprocs_default=1
param_default="None"
pure_python_default="False"
unsafe_building_default="False"
walltime_default="00:00:00"  # 00:00:00 implies unset

# Initial but illegal values of some command-line arguments,
# for testing whether these arguments have been supplied.
nprocs_unspecified="-1"
param_unspecified="__none__"
queue_unspecified="__none__"
test_unspecified="__none__"
utility_unspecified="__none__"

# Functions for building the code
set_make_jobs() {
    # Arguments: [--force].
    # Set the "make_jobs" variable, holding the -j option for
    # future make commands, enabling parallel building.
    # Some implementations of make do not support the bare -j
    # option without explicitly specifying a number afterwards.
    # If so, we do not make use of the -j option.
    if [ "$1" != "--force" ] && ([ -n "${make_jobs}" ] || [ "${ssh}" == "True" ]); then
        # Do not change the value of make_jobs
        return
    fi
    tmp_dir_preexists="False"
    if [ -d "${tmp_dir}" ]; then
        tmp_dir_preexists="True"
    fi
    make_jobs_test_dir="${tmp_dir}/make_jobs_test_${start_time_human_nosep}"
    rm -rf "${make_jobs_test_dir}" 2>/dev/null || :
    mkdir -p "${make_jobs_test_dir}"
    printf "
test:
\t@echo success
" > "${make_jobs_test_dir}/Makefile"
    make_jobs_output="$(cd "${make_jobs_test_dir}" && make -j 2>/dev/null)" || :
    if [ "${make_jobs_output}" == "success" ]; then
        make_jobs="-j"
    fi
    if [ "${tmp_dir_preexists}" == "True" ]; then
        rm -rf "${make_jobs_test_dir}" 2>/dev/null || :
    else
        rm -rf "${tmp_dir}" 2>/dev/null || :
    fi
}
check_rebuild_necessary() {
    # The code should be rebuild if the compiled modules are
    # missing or out-of-date with respect to the source.
    # The makefiles check this already, but we would like to
    # avoid invoking make unnecessarily, as make is rather slow
    # due to the large environment inherited by this Bash process.
    source_files=("${src_dir}/"* "${concept_dir}/Makefile")
    t_src=$(                                                 \
        stat -c '%Y'                                         \
            "${source_files[@]}"                             \
        | awk '(FNR==1){x=$1} {x=$1 > x?$1:x} END {print x}' \
    )
    t_build=$(                                               \
        stat -c '%Y'                                         \
            "${build}/"*.so                                  \
            2>/dev/null                                      \
        | awk '(FNR==1){x=$1} {x=$1 < x?$1:x} END {print x}' \
    )
    for f_src in "${source_files[@]}"; do
        if [[ "${f_src}" == *.py ]]; then
            f_base="$(basename "${f_src}")"
            if [ "${f_base}" == "pyxpp.py" ]; then
                continue
            fi
            f_build="${build}/${f_base%.py}.so"
            if [ ! -s "${f_build}" ]; then
                # Flag rebuild
                t_build=""
                break
            fi
        fi
    done
    if [ -z "${t_src}" ] || [ -z "${t_build}" ] || [ "${t_build}" -le "${t_src}" ]; then
        # The build is not up-to-date
        echo "True"
    else
        echo "False"
    fi
}
build_concept() {
    # Never build the code when running one of these utilities
    utilities_nobuild=("play" "update" "watch")
    utility_name="$(basename "${utility}")"
    if [ -n "${utility_name}" ]; then
        for utility_nobuild in ${utilities_nobuild[@]}; do
            if [ "${utility_name}" == "${utility_nobuild}" ]; then
                return
            fi
        done
    fi
    # In case of pure Python mode, run directly off of the Python source
    if [ "${pure_python}" == "True" ]; then
        build="${src_dir}"
        # Note that ${src_dir} is already included in PYTHONPATH
        return
    fi
    # Add the selected build directory as the first searched path
    # when importing modules in Python.
    export PYTHONPATH="${build}:${PYTHONPATH}"
    # Check whether the code should be rebuild
    if [ "${rebuild}" == "True" ]; then
        rm -rf "${build}/"* "${build}/".[^.]*
    elif [ "${rebuild}" == "False" ]; then
        return
    else
        rebuild_actual="$(check_rebuild_necessary)"
        if [ "${rebuild_actual}" == "False" ]; then
            return
        fi
    fi
    # Rebuild the code.
    # As this process requires a lot of memory it may fail,
    # in particular on virtual machines with dynamic memory
    # and a small base memory. Attempting the compilation
    # a few times in a row can help, as the dynamic memory
    # then has a chance to increase.
    start_time_build=$("${python}" -B -c "import time; print(time.time())")
    set_make_jobs
    make_concept() {
        make                                               \
            build="${build}"                               \
            native_optimizations="${native_optimizations}" \
            no_lto="${no_lto}"                             \
            no_optimizations="${no_optimizations}"         \
            unsafe_building="${unsafe_building}"           \
            ${make_jobs}
    }
    n_make_attemps=3
    for ((i = 0; i < ${n_make_attemps}; i += 1)); do
        exec 4>&1
        make_output="$(
            cd "${concept_dir}"                               \
            && make_concept                                   \
                | tee >(cat >&4)                              \
                ; echo "concept_exit_code = ${PIPESTATUS[0]}" \
        )"
        exec 4>&-
        exit_code="$(echo "${make_output}" | grep "concept_exit_code" \
            | awk '{print $NF}' || :)"
        if [ "${exit_code}" == "0" ]; then
            break
        fi
        sleep 1
    done
    if [ "${exit_code}" != "0" ]; then
        colorprint "Failed to compile ${esc_concept}!" "red"
        exit 1
    fi
    if [[ "${make_output}" == *"Building modules"* ]]; then
        "${mpiexec}" -n 1 "${python}" -B -c "from commons import *
print('Build time: {}'.format(time_since(${start_time_build})))"
    fi
}

# Function for setting variables used
# when running CO𝘕CEPT via Python.
prepare_python_options() {
    # Prepare Python options
    if [ "${main_as_command}" == "True" ]; then
        # Run main as Python command
        main_as_library="${main}"
        m_flag="-c"
    else
        if [ "${pure_python}" == "True" ]; then
            # Run main as normal Python script
            main_as_library="${main}"
            m_flag=""
        else
            main_as_library="${main%.*}.so"
            if [ -f "${main_as_library}" ]; then
                # Run main as compiled library module
                main_as_library="$(basename "${main_as_library}")"
                m_flag="-m"
            else
                # Run main as normal Python script,
                # even though the CO𝘕CEPT modules are compiled.
                main_as_library="${main}"
                m_flag=""
            fi
        fi
    fi
    i_flag=""
    if [ "${interactive}" == "True" ]; then
        i_flag="-i"
    fi
}

# Function for printing basic information about a job
print_info() {
    echo "Date:            ${start_time_human_sec}" \
        | tee -a "${job_dir}/${jobid}/log"
    if [ -n "${jobname}" ] && [ "${jobname}" != "${jobname_default}" ]; then
        echo "Job name:        ${jobname}" \
            | tee -a "${job_dir}/${jobid}/log"
    fi
    echo "Job ID:          ${jobid}" \
        | tee -a "${job_dir}/${jobid}/log"
    if [ "${pure_python}" == "False" ] && [ "${build}" != "${build_dir}" ]; then
        echo "Build:           \"$(sensible_path "${build}")\"" \
            | tee -a "${job_dir}/${jobid}/log"
    fi
    if [ "${main_as_command}" == "False" ] \
        && [[ "${main}" != "${src_dir}/main."* ]] \
        && [[ "${main}" != "${build}/main."* ]] \
    ; then
        echo "Entry point:     \"$(sensible_path "${main}")\"" \
            | tee -a "${job_dir}/${jobid}/log"
    fi
    if [ "${param}" == "${param_default}" ]; then
        echo "Parameter file:  ${param}" \
            | tee -a "${job_dir}/${jobid}/log"
    else
        echo "Parameter file:  \"$(sensible_path "${param}")\"" \
            | tee -a "${job_dir}/${jobid}/log"
    fi
    if [ -n "${memory_display}" ]; then
        echo "Memory:          ${memory_display}" \
            | tee -a "${job_dir}/${jobid}/log"
    fi
    if [ -n "${walltime_display}" ]; then
        echo "Wall time:       ${walltime_display}" \
            | tee -a "${job_dir}/${jobid}/log"
    fi
}

# If this file is being sourced, return now
if [ "${being_sourced}" == "True" ]; then
    this_file="${this_file_backup}"
    this_dir="${this_dir_backup}"
    return
fi

# Set up error trapping
ctrl_c() {
    trap : 0
    exit 2
}
abort() {
    exit_code_newest=$?
    colorprint "An error occurred!" "red"
    if [ -n "${exit_code}" ]; then
        exit ${exit_code}
    elif [ ${exit_code_newest} -ne 0 ]; then
        exit ${exit_code_newest}
    else
        exit 1
    fi
}
trap 'ctrl_c' SIGINT
trap 'abort' EXIT
set -e

# Function which prints the resource manager.
# In order, the implemented resource managers are:
# - Slurm
# - TORQUE/PBS
get_resource_manager() {
    # Detect what resource manager is used
    if get_command sbatch >/dev/null; then
        # Slurm is installed. Use this as the resource manager.
        resource_manager="slurm"
    elif get_command qsub >/dev/null; then
        # TORQUE/PBS is installed. Use this as the resource manager.
        resource_manager="torque"
    else
        # No resource manager found
        resource_manager=""
    fi
    echo "${resource_manager}"
}

# Change to the concept directory
cd "${concept_dir}"

# Use Python's argparse module to handle command-line arguments
argparse_finished="False"
argparse_exit_code=""
args=$("${python}" -B -c "
import argparse, math, re, sys
# Function which checks whether input is a representation of
# a positive integer and converts it.
def positive_int(value):
    value_raw = value
    def raise_argparse_exception():
        raise argparse.ArgumentTypeError(\"invalid positive int value: '{}'\".format(value_raw))
    try:
        value = float(eval(value))
    except:
        raise_argparse_exception()
    if value != int(value):
        raise_argparse_exception()
    value = int(value)
    if value < 1:
        raise_argparse_exception()
    return value
# Function which checks whether input is a representation of
# one or two positive integers. If two ints are given,
# separate them by a colon.
def positive_int_or_int_pair(value, value_input=None):
    if value_input is None:
        value_input = value
    def raise_argparse_exception():
        raise argparse.ArgumentTypeError(
            f\"invalid positive int or int pair: '{value_input}'\"
        )
    for sep in ',;':
        value = value.replace(sep, ':')
    if value.count(':') > 1:
        raise_argparse_exception()
    elif value.count(':') == 1:
        values = value.split(':')
        return ':'.join(positive_int_or_int_pair(value, value_input) for value in values)
    else:  # value.count(':') == 0
        try:
            value_eval = eval(value)
        except:
            return positive_int_or_int_pair(value.replace(' ', ':'), value_input)
        try:
            value_float = float(value_eval)
        except:
            raise_argparse_exception()
        value_int = int(value_float)
        if value_int == value_float and value_int > 0:
            return str(value_int)
        raise_argparse_exception()
# Function which checks whether input is a representation of
# a memory size and converts it to bytes.
def memory(value):
    value_raw = value
    def raise_argparse_exception():
        raise argparse.ArgumentTypeError(\"invalid memory value: '{}'\".format(value_raw))
    # Convert to (whole) bytes
    value = value.lower()
    value = value.replace(' ', '').replace('b', '')
    value = re.subn(r'([0-9]+)([a-z]+)', r'\g<1>*\g<2>', value)[0]
    units = {'k': 2**10,
             'm': 2**20,
             'g': 2**30,
             't': 2**40,
             'p': 2**50,
             'e': 2**60,
             'z': 2**70,
             'y': 2**80,
             }
    try:
        value = int(math.ceil(float(eval(value, units))))
    except:
        raise_argparse_exception()
    return value
# Function which converts a time value to the format hh:mm:ss
def time(value):
    # Convert value to integer seconds
    units = {'s': 1}
    units['sec'] = units['secs'] = units['seond'] = units['seonds'] = units['s']
    units['m'] = 60*units['s']
    units['min'] = units['mins'] = units['minute'] = units['minutes'] = units['m']
    units['h'] = 60*units['m']
    units['hr'] = units['hrs'] = units['hs'] = units['hour'] = units['hours'] = units['h']
    units['d'] = 24*units['h']
    units['day'] = units['days'] = units['d']
    units['y'] = 365.25*units['d']
    units['yr'] = units['year'] = units['years'] = units['y']
    # If a pure number is provided, interpret this in seconds
    value = value.replace(' ', '')
    try:
        value = int(value)*units['s']
    except:
        pass
    # Attempt to interpret the time as an expression
    # like '2hr + 30mins'.
    if isinstance(value, str):
        value = value.lower()
        value = re.subn(r'([0-9]+)([a-z]+)', r'\g<1>*\g<2>', value)[0]
        try:
            value = int(math.ceil(float(eval(value, units))))
        except:
            pass
    # Attempt to interpret the time in the format
    # 'm:s' or 'h:m:s' or 'd:h:m:s' or 'd+h:m:s' or 'd+h:m' or 'd+h'.
    if isinstance(value, str):
        plusses_in_value = value.count('+')
        if plusses_in_value > 1:
            raise argparse.ArgumentTypeError(\"error parsing value\")
        for sep in '+ ,;':
            value = value.replace(sep, ':')
        value = value.split(':')
        s = m = h = d = 0
        if len(value) == 1:
            raise argparse.ArgumentTypeError(\"error parsing value\")
        elif len(value) == 2:
            if plusses_in_value:
                # Format d:h
                d, h = value
            else:
                # Format m:s
                m, s = value
        elif len(value) == 3:
            if plusses_in_value:
                # Format d+h:m
                d, h, m = value
            else:
                # Format h:m:s
                h, m, s = value
        elif len(value) == 4:
            # Format d:h:m:s or d+h:m:s
            d, h, m, s = value
        else:
            raise argparse.ArgumentTypeError(\"error parsing value\")
        d, h, m, s = int(d), int(h), int(m), int(s)
        value = int(math.ceil(s*units['s'] + m*units['m'] + h*units['h'] + d*units['d']))
    # Now value should be in integer seconds
    if value < 0:
        raise argparse.ArgumentTypeError(\"the wall time cannot be negative\")
    # Convert to the format hh:mm:ss.
    h = value//units['h']
    value -= h*units['h']
    m = value//units['m']
    value -= m*units['m']
    s = value
    h, m, s = str(h), str(m), str(s)
    if len(h) == 1:
        h = '0' + h
    if len(m) == 1:
        m = '0' + m
    if len(s) == 1:
        s = '0' + s
    value = f'{h}:{m}:{s}'
    return value
# Function which checks whether input is a boolean or is empty
def bool_or_empty(value):
    value_raw = value
    value = str(value).lower()
    if value == '':
        return value
    if value in {'true', 'y', 'yes'}:
        return 'True'
    if value in {'false', 'n', 'no'}:
        return 'False'
    raise argparse.ArgumentTypeError(\"invalid non-empty bool value: '{}'\".format(value_raw))
# Function for setting up a parser for command-line arguments
def get_parser(*, add_help=True):
    parser = argparse.ArgumentParser(
        prog='$(basename "${this_file}")',
        description='Run the ${esc_concept} code',
        add_help=add_help,
    )
    parser.add_argument(
        '-b', '--build',
        help='build directory to use',
        default='${build_dir}',
    )
    parser.add_argument(
        '-c', '--command-line-params',
        help=(
            'specify parameter(s) directly from the command-line. '
            'If a parameter file is specified as well, the command-line '
            'parameters will take precedence. This option may be specified '
            'multiple times.'
        ),
        default=[],
        action='append',
    )
    parser.add_argument(
        '-i', '--interactive',
        help='inspect interactively after program execution',
        default=${interactive_default},
        action='store_true',
    )
    parser.add_argument(
        '-j', '--job-directive',
        help=(
            'specify an additional line to add to the job script header '
            'for remote jobs. This option may be specified multiple times.'
        ),
        default=[],
        action='append',
    )
    parser.add_argument(
        '-m', '--main',
        help='entry point of the code. Can be a Python filename or command.',
        default='${src_dir}/main.py',
    )
    parser.add_argument(
        '--memory',
        help='total memory allocated for remote job',
        type=memory,
        default=${memory_default},
    )
    parser.add_argument(
        '-n', '--nprocs',
        help=(
            'total number of processes '
            'or number of nodes and number of processes per node'
        ),
        type=positive_int_or_int_pair,
        default=${nprocs_unspecified},
    )
    parser.add_argument(
        '-p', '--params',
        help='parameter file to use',
        default='${param_unspecified}',
    )
    parser.add_argument(
        '-q', '--queue',
        help='queue for submission of the remote job',
        default='${queue_unspecified}',
    )
    parser.add_argument(
        '-t', '--tests',
        help=(
            'run test TESTS. TESTS can be any subdirectory of the test directory. '
            'Use TESTS=all to run all tests'
        ),
        default='${test_unspecified}',
    )
    parser.add_argument(
        '-u', '--utility',
        nargs='+',
        help='run utility UTILITY. UTILITY can be any executable in the util directory',
        default=['${utility_unspecified}']*2,  # One for utility, one for utility_args
    )
    parser.add_argument(
        '-w', '--walltime',
        help='maxium allowed wall time for remote job',
        type=time,
        default='${walltime_default}',
    )
    parser.add_argument(
        '--job-name',
        help='name of the job to be used with Slurm/TORQUE/PBS',
        default='${jobname_default}',
    )
    parser.add_argument(
        '--local',
        help='force the run to be done locally, without submitting it as a remote job',
        default=${local_default},
        action='store_true',
    )
    parser.add_argument(
        '--native-optimizations',
        help='allow the compiler to generate non-portable code optimized for this machine',
        default=${native_optimizations_default},
        action='store_true',
    )
    parser.add_argument(
        '--no-lto',
        help='disable link time optimizations',
        default=${no_lto_default},
        action='store_true',
    )
    parser.add_argument(
        '--no-optimizations',
        help='disable compiler optimizations',
        default=${no_optimizations_default},
        action='store_true',
    )
    parser.add_argument(
        '--no-watching',
        help='do not follow the submitted job via the watch utility',
        default=${no_watching_default},
        action='store_true',
    )
    parser.add_argument(
        '--pure-python',
        help='run in pure Python mode',
        default=${pure_python_default},
        action='store_true',
    )
    parser.add_argument(
        '--rebuild',
        nargs='?',
        help='specify whether to rebuild the code before running it',
        type=bool_or_empty,
        default='',
        const='True',
    )
    parser.add_argument(
        '--unsafe-building',
        help='ignore dependencies between modules when building',
        default=${unsafe_building_default},
        action='store_true',
    )
    parser.add_argument(
        '-v', '--version',
        help='print version info',
        default=False,
        action='store_true',
    )
    return parser
# Enables Python to write directly to screen (stderr)
# in case of help request.
stdout = sys.stdout
sys.stdout = sys.stderr
# Parse without -h/--help
parser = get_parser(add_help=False)
args, unknown_args = parser.parse_known_args()
# If a utility is to be used, -h/--help and arguments
# unknown to this script should be passed on to the utility.
utility_args = args.utility[1:]
if args.utility[0] != '${utility_unspecified}':
    utility_args += unknown_args
else:
    # No utility should be used. Parse again,
    # this time including -h/--help and errors
    # in case of invalid arguments.
    parser = get_parser()
    args = parser.parse_args()
# Reinsert stdout
sys.stdout = stdout
# Function for converting a list to a str in the form of a Bash array
def python_list2bash_array(lst):
    return '({})'.format(\"'\" + \"' '\".join(
        [str(val).replace(\"'\", '\"') for val in lst]
    ) + \"'\")
# Print out the arguments.
# These will be captured in the Bash 'args' variable
print('; '.join([
    \"argparse_finished=True\",
    \"build='{}'\".format(str(args.build).replace(\"'\", '\"')),
    \"command_line_params='{}'\".format(' ; '.join(args.command_line_params).replace(\"'\", '\"')),
    \"interactive={}\".format(args.interactive),
    \"job_directive='{}'\".format(' ; '.join(args.job_directive).replace(\"'\", '\"')),
    \"main='{}'\".format(str(args.main).replace(\"'\", '\"')),
    \"memory='{}'\".format(args.memory),
    \"nprocs={}\".format(args.nprocs),
    \"param='{}'\".format(args.params),
    \"queue='{}'\".format(args.queue),
    \"test='{}'\".format(args.tests),
    \"utility='{}'\".format(args.utility[0]),
    \"utility_args={}\".format(python_list2bash_array(utility_args)),
    \"walltime={}\".format(args.walltime),
    \"jobname='{}'\".format(args.job_name),
    \"local={}\".format(args.local),
    \"native_optimizations={}\".format(args.native_optimizations),
    \"no_lto={}\".format(args.no_lto),
    \"no_optimizations={}\".format(args.no_optimizations),
    \"no_watching={}\".format(args.no_watching),
    \"pure_python={}\".format(args.pure_python),
    \"rebuild={}\".format(args.rebuild),
    \"unsafe_building={}\".format(args.unsafe_building),
    \"version={}\".format(args.version),
]))
" "$@" || echo "argparse_exit_code=$?")
# Evaluate the handled arguments into this scope
eval "${args}"

# Exit if argparse exited without finishing
if [ "${argparse_finished}" != "True" ]; then
    if [ -z "${argparse_exit_code}" ]; then
        argparse_exit_code=0
    fi
    if [ ${argparse_exit_code} -eq 0 ]; then
        trap : 0
    fi
    exit ${argparse_exit_code}
fi

# Print version info and exit, if requested
if [ "${version}" == "True" ]; then
    "${concept}" -m 'import commons; print(commons.__version__)' \
        --local --pure-python | tail -n 1
    trap : 0
    exit 0
fi

# Display warning if a build directory is selected
# while running in pure Python mode.
if [ "${pure_python}" == "True" ] && [ "${build}" != "${build_dir}" ]; then
    colorprint "Warning: Ignoring the specified build directory \"${build}\" \
as running in pure Python mode" "red"
fi

# Display warning if the requested memory is below one megabyte
if [ ${memory} -gt 0 ] && [ ${memory} -lt 1048576 ]; then
    colorprint "Warning: The requested memory is below 1 MB. \
Have you forgotten to specify the unit?" "red"
fi

# Display warning if the requested wall time is below one minute
if [ "${walltime}" != "${walltime_default}" ] && [[ "${walltime}" == "00:00:"* ]]; then
    colorprint "Warning: The requested wall time is below 1 minute. \
Have you forgotten to specify the unit?" "red"
fi

# Check whether the main "file" is really a string of commands
main_as_command="False"
if (   [[ "${main}" == *"print("* ]] \
    || [[ "${main}" == *";"*      ]] \
    || [[ "${main}" == *$'\n'*    ]]); then
    main_as_command="True"
fi

# Convert all supplied paths to absolute paths
build="$(absolute_path "${build}")"
if [ "${main_as_command}" == "False" ]; then
    main="$(absolute_path "${main}")"
fi
if [ "${param}" != "${param_unspecified}" ]; then
    param_ori="${param}"
    param="$(absolute_path "${param}")"
    if [ ! -f "${param}" ] && [ -f "${param_dir}/${param_ori}" ]; then
        param="${param_dir}/${param_ori}"
    fi
fi
if [ "${test}" != "${test_unspecified}" ] && [ "${test}" != "all" ]; then
    if [[ "${test}" == *"/run" ]]; then
        test="$(dirname "${test}")"
    fi
    test="${test_dir}/$(basename "${test}")"
fi
if [ "${utility}" != "${utility_unspecified}" ]; then
    utility="${util_dir}/$(basename "${utility}")"
fi

# Function for doing fuzzy comparisons between
# illegal concept options and possible correct ones.
concept_options=("$@")
suggest_correct_invocation() {
    # First argument: The illegal option
    # Second argument: The option type ('test' or 'utility')
    illegal_option="$(basename "$1")"
    invocation="$0"
    for arg in "${concept_options[@]}"; do
        if [ "${replace_next}" == "True" ]; then
            invocation="${invocation} __replace__"
            replace_next="False"
        else
            invocation="${invocation} ${arg}"
        fi
        if ([ "$2" == "test" ] && (        \
               [ "${arg}" == "-t"      ]   \
            || [ "${arg}" == "--test"  ]   \
            || [ "${arg}" == "--tests" ]   \
        )) || ([ "$2" == "utility" ] && (  \
               [ "${arg}" == "-u"        ] \
            || [ "${arg}" == "--util"    ] \
            || [ "${arg}" == "--utility" ] \
        )); then
            replace_next="True"
        fi
    done
    "${python}" -B -c "
import difflib, shutil, os
if '$2' == 'test':
    files = os.listdir('${test_dir}')
    possibilities = [file for file in files if os.path.isdir('${test_dir}/' + file)]
elif '$2' == 'utility':
    files = os.listdir('${util_dir}')
    possibilities = [file for file in files if shutil.which('${util_dir}/' + file)]
max_ratio = 0
for possibility in possibilities:
    ratio = max([
        difflib.SequenceMatcher(a='${illegal_option}', b=possibility).ratio(),
        difflib.SequenceMatcher(a='${illegal_option}'.lower(), b=possibility.lower()).ratio(),
    ])
    if ratio > max_ratio:
        max_ratio = ratio
        closest_match = possibility
if max_ratio > 0.01:
    print('Did you mean:\n${invocation}'.replace('__replace__', closest_match))
                   "
}

# Use the main.{py/so} of the build instead of the source.
# Error out if neither path exists. If a utility is to be run,
# do not do this now, but wait until this script is called back
# from the utility.
if [ "${main_as_command}" == "False" ] && [ "${utility}" == "${utility_unspecified}" ]; then
    main_ori="${main}"
    main_missing=0
    for i in 0 1; do
        if [ ${i} -eq 1 ] && [ "${pure_python}" != "True" ]; then
            # Use the main file of the build instead of the source
            if [ "${main}" == "${src_dir}/main.py" ]; then
                main="${build}/main.py"
            elif [ "${main}" == "${src_dir}/main.so" ]; then
                main="${build}/main.so"
            fi
        fi
        if [ "${pure_python}" == "True" ]; then
            if [ ! -f "${main}" ] && [ ! -f "${main%.*}.py" ]; then
                ((main_missing += 1))
            fi
        else
            if [ ! -f "${main}" ] && [ ! -f "${main%.*}.py" ] && [ ! -f "${main%.*}.so" ]; then
                ((main_missing += 1))
            fi
        fi
    done
    if [ ${main_missing} -eq 2 ]; then
        if [ -d "${main_ori}" ]; then
            colorprint "Error: Entry point \"${main_ori}\" is a directory" "red"
        else
            colorprint "Error: Entry point \"${main_ori}\" does not exist" "red"
        fi
        exit 1
    fi
fi

# Do the supplied paths exist?
if [ "${param}" != "${param_unspecified}" ] && [ ! -f "${param}" ]; then
    if [ -d "${param}" ]; then
        colorprint "Error: Parameter file \"${param}\" is a directory" "red"
    else
        colorprint "Error: Parameter file \"${param}\" does not exist" "red"
    fi
    exit 1
fi
if [ "${test}" != "${test_unspecified}" ] && [ "${test}" != "all" ] && [ ! -d "${test}" ]; then
    if [ -f "${test}" ]; then
        colorprint "Error: Test \"${test}\" is a file but should be a directory" "red"
    else
        colorprint "Error: Test \"${test}\" does not exist" "red"
    fi
    # Suggest closest match
    suggest_correct_invocation "${test}" "test"
    exit 1
fi
if [ "${utility}" != "${utility_unspecified}" ] && [ ! -f "${utility}" ]; then
    if [ -d "${utility}" ]; then
        colorprint "Error: Utility \"${utility}\" is a directory" "red"
    else
        colorprint "Error: Utility \"${utility}\" does not exist" "red"
    fi
    # Suggest closest match
    suggest_correct_invocation "${utility}" "utility"
    exit 1
fi

# Assigned values to unspecified options
if [ "${nprocs}" == "${nprocs_unspecified}" ]; then
    nprocs="${nprocs_default}"
fi
if [ "${param}" == "${param_unspecified}" ]; then
    param="${param_default}"
fi

# Check for syntax errors in supplied parameter file.
# Other types of errors will not be detected before runtime.
if [ "${param}" != "${param_default}" ]; then
    param_traceback="$("${python}" -B -c "
import ast
with open('${param}', mode='r', encoding='utf-8') as f:
    source = f.read()
ast.parse(source)
" 2>&1 || :)"
    if [ -n "${param_traceback}" ]; then
        colorprint "Syntax error in parameter file \"${param}\":" "red"
        echo "${param_traceback}"
        exit 1
    fi
fi
# Check for syntax errors in supplied command-line parameters.
# Other types of errors will not be detected before runtime.
if [ -n "${command_line_params}" ]; then
    command_line_params_traceback="$("${python}" -B -c "
import ast
source = '${command_line_params}'
ast.parse(source)
" 2>&1 || :)"
    if [ -n "${command_line_params_traceback}" ]; then
        if [[ "${command_line_params_traceback}" == *";"* ]]; then
            colorprint "Syntax error in command-line parameters:" "red"
        else
            colorprint "Syntax error in command-line parameter:" "red"
        fi
        echo "${command_line_params_traceback}"
        exit 1
    fi
fi

# Determine whether to run CO𝘕CEPT locally or remotely (via some
# resource manager). Always treat tests as if they were run locally.
remote="False"
if     [ "${local}" == "False"               ] \
    && [ "${test}"  == "${test_unspecified}" ] \
    && [ "${ssh}"   == "True"                ]; then
    remote="True"
fi

# If a test is to be run, run it and exit
if [ "${test}" != "${test_unspecified}" ]; then
    # Function for running a test, completely containing it within an
    # artifact subdirectory.
    run_test() {
        test_name="$1"
        test_name="$(basename "${test_name}")"
        artifact_dir="${test_dir}/${test_name}/artifact"
        rm -rf "${artifact_dir}"
        mkdir "${artifact_dir}"
        for f in "${test_dir}/${test_name}/"*; do
            if [ "${f}" == "${artifact_dir}" ]; then
                continue
            fi
            cp -r "${f}" "${artifact_dir}/"
        done
        "${artifact_dir}/run"
    }
    # Function which can extract parameters from the parameter file
    # given by "${this_dir}/param". This is used by some tests.
    get_param() {
        "${concept}"                              \
            -c="${command_line_params}"           \
            -m="
from commons import *
try:
    print('param_var =', ${1})
except NameError:
    print('param_var =', user_params['''${1}'''])
"                                                 \
            -n=1                                  \
            -p="${this_dir}/param"                \
            --local                               \
            --pure-python                         \
        | grep "param_var"                        \
        | tail -n 1                               \
        | sed 's/^.\{12\}//'
    }
    # If the job directory is clean,
    # we count any generated *_err file as a test failure.
    test_check_err_logs() { :; }
    if [ ! -d "${job_dir}" ] || [ -z "$(ls "${job_dir}")" ]; then
        test_check_err_logs() {
            if ls "${job_dir}/"*/log_err >/dev/null 2>&1; then
                colorprint "It looks like ${test_name} test generated error messages.\n\
This is counted as a test failure." "red"
                exit 1
            fi
        }
    fi
    # Enable warnings that are ignored by default.
    # We do this in order to get notified about deprecated code usage.
    # Due to the above test_check_err_logs(), any such warning will
    # lead to test failure if starting with an empty job directory.
    if [ -z "${PYTHONWARNINGS}" ]; then
        export PYTHONWARNINGS=default
    fi
    # Do the test(s)
    if [ "${test}" == "all" ]; then
        # Find and run all tests
        tests="$(cd "${test_dir}" && "${python}" -B -c "
from glob import glob
# List tests in order of required execution.
# Tests not included here will be run last.
order = (
    # Test whether the code is able to compile and run
    'basic',
    # Tests of the CLASS installation,
    # the Friedmann equation, realisations
    # and the power spectrum functionality.
    'friedmann',
    'realize',
    'powerspec',
    # Test of the GADGET-2 installation
    'gadget',
    # Tests of the particle implementation
    'drift_nohubble',
    'drift',
    'kick_pp_without_ewald',
    'kick_pp_with_ewald',
    # Tests of the PP implementation
    'nprocs_pp',
    'pure_python_pp',
    'concept_vs_gadget_pp',
    # Tests of the PM implementation
    'nprocs_pm',
    'pure_python_pm',
    'concept_vs_class_pm',
    # Tests of the P³M implementation
    'nprocs_p3m',
    'pure_python_p3m',
    'concept_vs_gadget_p3m',
    # Test multi-component simulations (particles only)
    'multicomponent',
    # Test upstream/downstream grid scalings and multi-grid simulations
    'multigrid',
    # Tests of the fluid implementation
    'fluid_drift_rigid_nohubble',
    'fluid_drift_rigid',
    'fluid_gravity_nohubble',
    'fluid_gravity',
    'fluid_vacuum',
    'fluid_vs_particles',
    'fluid_pressure',
    # To test the fluid implementation on a real life scenario,
    # we test linear and non-linear neutrino simulations.
    'neutrino',
    # Test whether the optimizations introduces bugs
    'optimizations',
    # Tests of other functionality
    'render',
)
# Find all tests (directories in ${test_dir}).
# Skip test if its (directory) name has a leading underscore.
tests = (dir[:-1] for dir in glob('*/') if not dir.startswith('_'))
# Sort the tests based on the order given above
sorted_tests = sorted(tests, key=lambda test: order.index(test) if test in order else len(order))
for test in sorted_tests:
    print(test)
"              )"
        # Run all tests in the test directory
        printf "\nThe following tests will be run in order:\n"
        for test_name in ${tests}; do
            echo "    ${test_name}"
        done
        echo
        (cd "${concept_dir}" && make clean-test)
        for test_name in ${tests}; do
            start_time_test=$("${python}" -B -c "import time; print(time.time())")
            colorprint "\nRunning ${test_name} test" "yellow"
            run_test "${test_name}"
            test_check_err_logs
            colorprint "${test_name} test ran successfully" "green"
            # Print out the execution time for this test
            "${mpiexec}" -n 1 "${python}" -B -c "from commons import *
print('Total execution time: {}'.format(time_since(${start_time_test})))"
        done
        colorprint "\nAll tests ran successfully" "green"
    else
        # Run specific test
        test_name="$(basename "${test}")"
        colorprint "\nRunning ${test_name} test" "yellow"
        run_test "${test}"
        test_check_err_logs
        colorprint "${test_name} test ran successfully" "green"
    fi
    # Print out the total time it took to run the test(s)
    "${mpiexec}" -n 1 "${python}" -B -c "from commons import *
print('Total execution time: {}'.format(time_since(${start_time_epoch})))"
    # Deactivate traps and exit
    trap : 0
    exit 0
fi

# Cannot run in interactive mode when running remotely
if [ "${remote}" == "True" ] && [ "${interactive}" == "True" ]; then
    colorprint "Cannot run in interactive mode when run remotely" "red"
    exit 1
fi

# If a utility is to be run, run it and exit
if [ "${utility}" != "${utility_unspecified}" ]; then
    # If no argument was passed after the -u option,
    # utility_args should be empty.
    if [ "${utility_args}" == '""' ] || [ "${utility_args}" == "''" ]; then
        utility_args=""
    fi
    # Set flag variables for the flag command options,
    # so that a utility can call this script with the same flags enabled
    # as was used to invoke this script originally.
    interactive_flag=""
    if [ "${interactive}" == "True" ]; then
        interactive_flag="--interactive"
    fi
    local_flag=""
    if [ "${local}" == "True" ]; then
        local_flag="--local"
    fi
    native_optimizations_flag=""
    if [ "${native_optimizations}" == "True" ]; then
        native_optimizations_flag="--native-optimizations"
    fi
    no_lto_flag=""
    if [ "${no_lto}" == "True" ]; then
        no_lto_flag="--no-lto"
    fi
    no_optimizations_flag=""
    if [ "${no_optimizations}" == "True" ]; then
        no_optimizations_flag="--no-optimizations"
    fi
    no_watching_flag=""
    if [ "${no_watching}" == "True" ]; then
        no_watching_flag="--no-watching"
    fi
    pure_python_flag=""
    if [ "${pure_python}" == "True" ]; then
        pure_python_flag="--pure-python"
    fi
    unsafe_building_flag=""
    if [ "${unsafe_building}" == "True" ]; then
        unsafe_building_flag="--unsafe-building"
    fi
    # Run utility and exit
    colorprint "\nRunning the $(basename "${utility}") utility" "yellow"
    trap : 0
    if [ -z "${utility_args[0]}" ] && [ ${#utility_args[@]} == 1 ]; then
        called_from_concept="True" "${utility}"
    else
        called_from_concept="True" "${utility}" "${utility_args[@]}"
    fi
    # Print out the total execution time for this utility
    "${mpiexec}" -n 1 "${python}" -B -c "from commons import *
print('Total execution time: {}'.format(time_since(${start_time_epoch})))"
    exit 0
fi

# Build the code as necessary. When running remotely,
# the code will be built from within the job script.
if [ "${remote}" == "False" ]; then
    build_concept
fi

# Ensure existence of job directory
mkdir -p "${job_dir}"

# Function for copying the parameter file (including command-line
# arguments) to the job sub-directory of the current run.
# This copy will be the parameter file actually used,
# freeing up the supplied parameter file for editing.
copy_param() {
    if [ "${param}" == "${param_default}" ]; then
        if [ -z "${command_line_params}" ]; then
            return
        fi
        touch "${job_dir}/${jobid}/param"
    else
        # Copy the parameter file
        cp "${param}" "${job_dir}/${jobid}/param"
    fi
    # Insert command-line parameters
    # at the bottom of the copied parameter file.
    "${python}" -B -c "
lines = [
    '',
    '# The above is a copy of the CO𝘕CEPT parameter file \"${param}\".',
    '# Below we apply any additional command-line parameters.',
]
for statement in '${command_line_params}'.split(';'):
    lines.append(statement.strip())
lines.append('')
with open('${job_dir}/${jobid}/param', mode='a') as f:
    f.write('\n'.join(lines))
"
}

# Either stop doing further actions, submit job or run it locally
if [ "${remote}" == "True" ]; then
    # Running remotely.
    # Detect what resource manager is used
    resource_manager="$(get_resource_manager)"
    # Prepare job script header dependent on the resource manager.
    # If no resource manager is used, default to slurm.
    if [ "${resource_manager}" == "slurm" ] || [ -z "${resource_manager}" ]; then
        # Split the 'nprocs' variable up into the number of nodes
        # and the number of processes per node, if both are given.
        if [[ "${nprocs}" == *':'* ]]; then
            nnodes=$("${python}" -B -c "print('${nprocs}'[:'${nprocs}'.index(':')])")
            nprocs_per_node=$("${python}" -B -c "print('${nprocs}'[('${nprocs}'.index(':') + 1):])")
            ((nprocs = nnodes*nprocs_per_node))
        else
            nnodes=0  # Has to be 0, not 1
            nprocs_per_node=${nprocs}
        fi
        # Compute dedicated memory per process and node in megabytes
        mb_per_process=$("${python}" -B -c "print(int(${memory}/(2**20*${nprocs})))")
        if [ "${nnodes}" -gt 0 ]; then
            mb_per_node=$("${python}" -B -c "print(int(${memory}/(2**20*${nnodes})))")
        fi
        # Construct Slurm header
        jobscript_header="$(${python} -B -c "
directive_prefix = '#SBATCH'
lines = []
lines.append(f'{directive_prefix} --job-name=${jobname}')
lines.append(f'{directive_prefix} --partition=${queue}')
if ${nnodes}:
    lines.append(f'{directive_prefix} --nodes=${nnodes}')
    lines.append(f'{directive_prefix} --ntasks-per-node=${nprocs_per_node}')
else:
    lines.append(f'{directive_prefix} --ntasks=${nprocs}')
if ${memory} > 0:
    if ${nnodes}:
        lines.append(f'{directive_prefix} --mem=${mb_per_node}M')
    else:
        lines.append(f'{directive_prefix} --mem-per-cpu=${mb_per_process}M')
elif ${memory} == 0:
    lines.append(f'{directive_prefix} --mem=0')
if '${walltime}' != '${walltime_default}':
    lines.append(f'{directive_prefix} --time=${walltime}')
lines.append(f'{directive_prefix} --ntasks-per-core=1')   # to avoid hyper-threading
lines.append(f'{directive_prefix} --hint=nomultithread')  # to avoid hyper-threading
lines.append(f'{directive_prefix} --cpus-per-task=1')     # map 1 MPI process to 1 OpenMP thread
lines.append(f'{directive_prefix} --output=/dev/null')
lines.append(f'{directive_prefix} --error=/dev/null')
for job_directive in '${job_directive}'.split(';'):
    job_directive = job_directive.strip()
    if not job_directive:
        continue
    if not job_directive.startswith(directive_prefix):
        job_directive = f'{directive_prefix} {job_directive}'
    lines.append(job_directive)
lines.append('')
lines.append('# Get the ID of the current job')
lines.append('jobid=\"\${SLURM_JOB_ID%%.*}\"')
print('\n'.join(lines))
        ")"
        # The Slurm queue (partition) in which the job is running
        queue_display='${SLURM_JOB_PARTITION}'
    elif [ "${resource_manager}" == "torque" ]; then
        # Split the 'nprocs' variable up into the number of nodes
        # and the number of processes per node.
        if [[ "${nprocs}" != *':'* ]]; then
            colorprint "Error: When using TORQUE or PBS you need to specify the number of nodes \
and the number of processes per node" "red"
            exit 1
        fi
        nnodes=$("${python}" -B -c "print('${nprocs}'[:'${nprocs}'.index(':')])")
        nprocs_per_node=$("${python}" -B -c "print('${nprocs}'[('${nprocs}'.index(':') + 1):])")
        ((nprocs = nnodes*nprocs_per_node))
        # Compute dedicated memory per process in megabytes
        mb_per_process=$("${python}" -B -c "print(int(${memory}/(2**20*${nprocs})))")
        # Construct TORQUE header
        jobscript_header="$(${python} -B -c "
directive_prefix = '#PBS'
lines = []
lines.append(f'{directive_prefix} -N ${jobname}')
lines.append(f'{directive_prefix} -q ${queue}')
lines.append(f'{directive_prefix} -l nodes=${nnodes}:ppn=${nprocs_per_node}')
if ${memory} > 0:
    lines.append(f'{directive_prefix} -l pmem=${mb_per_process}mb')
elif ${memory} == 0:
    lines.append(f'{directive_prefix} -l mem=0')
if '${walltime}' != '${walltime_default}':
    lines.append(f'{directive_prefix} -l walltime=${walltime}')
lines.append(f'{directive_prefix} -o /dev/null')
lines.append(f'{directive_prefix} -e /dev/null')
for job_directive in '${job_directive}'.split(';'):
    job_directive = job_directive.strip()
    if not job_directive:
        continue
    if not job_directive.startswith(directive_prefix):
        job_directive = f'{directive_prefix} {job_directive}'
    lines.append(job_directive)
lines.append('')
lines.append('# Get the ID of the current job')
lines.append('jobid=\"\${PBS_JOBID%%.*}\"')
print('\n'.join(lines))
        ")"
        # The TORQUE/PBS queue in which the job is running
        queue_display="${queue}"
    fi
    # Prepare display texts with the total memory consumption
    # and maximum allowed wall time.
    infos_display="$("${mpiexec}" -n 1 "${python}" -B -c "
memory = ${memory}
walltime = '${walltime}'
from commons import *
# Memory
memory_str = ' '
if memory > 0:
    prefixes = iter(['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'])
    for prefix in prefixes:
        if memory < 2**10:
            break
        memory /= 2**10
    memory_str = significant_figures(memory, 3)
    if 'e' in memory_str:
        memory /= 2**10
        prefix = next(prefixes)
        memory_str = significant_figures(memory, 3)
    memory_str = f'{memory_str} {prefix}B'
print(memory_str)
# Wall time
walltime_str = ' '
h, m, s = [int(t) for t in walltime.split(':')]
walltime = s + 60*(m + 60*h)
if walltime > 0:
    walltime_str = nice_time_interval(walltime)
print(walltime_str)
" 2>/dev/null | tail -n 2)"
    memory_display="$(echo "${infos_display}"   | head -n 1 | tail -n 1)"
    if [ -z "${memory_display// }" ]; then
        memory_display=""
    fi
    walltime_display="$(echo "${infos_display}" | head -n 2 | tail -n 1)"
    if [ -z "${walltime_display// }" ]; then
        walltime_display=""
    fi
    # The MPI executor command when run remotely.
    # When using Slurm, it is best to use the srun command rather than
    # calling mpiexec directly.
    # If the mpi_executor variable is already defined, we use this.
    if [ -z "${mpi_executor}" ]; then
        mpi_executor="${mpiexec}${mpiexec_args}"
        if [ "${resource_manager}" == "slurm" ]; then
            if get_command "srun" >/dev/null; then
                # Use the Slurm srun command as the MPI executor.
                # We further disable process binding/affinity in Slurm.
                mpi_executor="srun"
                srun_output="$(srun --help 2>&1 || :)"
                if echo "${srun_output}" | grep "\-\-cpu\-bind" > /dev/null; then
                    mpi_executor="${mpi_executor} --cpu-bind=none"
                elif echo "${srun_output}" | grep "\-\-cpu_bind" > /dev/null; then
                    mpi_executor="${mpi_executor} --cpu_bind=none"
                fi
                # When using OpenMPI with Slurm, it's often necessary to
                # additionally pass the --mpi=openmpi to srun. Here we
                # add this option if it is supported by srun.
                if [ "${mpi_implementation}" == "openmpi" ] \
                    && srun --mpi=list 2>&1 | grep "openmpi" >/dev/null; then
                    mpi_executor="${mpi_executor} --mpi=openmpi"
                fi
            else
                colorprint "Warning: Using Slurm but cannot find the srun command. \
Using mpiexec instead." "red"
            fi
        fi
    fi
    # Add additional environment variables to mpi_env
    # depending on the resource manager.
    if [ "${resource_manager}" == "slurm" ]; then
        # Disable additional output stream buffering by Slurm.
        # With this buffering enabled, Slurm does not respect
        # stream flushing but only flushes after a newline.
        mpi_env="${mpi_env}
export SLURM_UNBUFFEREDIO=1"
    fi
    # Write job script file
    jobscript="${job_dir}/.jobscript_${start_time_human_nosep}"
    printf "#!/usr/bin/env bash
${jobscript_header}

# Exit on error
set -e

# Source the concept script
source '${concept}'

# MPI executor
mpi_executor='${mpi_executor}'

${mpi_env}

# Variables
build=\"${build}\"
jobname=\"${jobname}\"
main=\"${main}\"
param=\"${param}\"
utility=\"${utility}\"
native_optimizations=\"${native_optimizations}\"
no_lto=\"${no_lto}\"
no_optimizations=\"${no_optimizations}\"
pure_python=\"${pure_python}\"
rebuild=\"${rebuild}\"
unsafe_building=\"${unsafe_building}\"
main_as_command=\"${main_as_command}\"
memory_display=\"${memory_display}\"
queue_display=\"${queue_display}\"
walltime_display=\"${walltime_display}\"

# The job sub-directory for the current run
# should have been created by the concept script.
# Allow a little time for this directory to appear.
sleep_time=1
sleep_max=30
slept=0
while [ ! -d \"\${job_dir}/\${jobid}\" ] && [ \${slept} -lt \${sleep_max} ]; do
    sleep \${sleep_time}
    ((slept += 1))
done
if [ ! -d \"\${job_dir}/\${jobid}\" ]; then
    # The job sub-directory did not appear.
    # Interpret this to mean that the job has been submitted manually
    # by the user instead of via the concept script.
    # Make the job sub-directory now and copy over the parameter file
    # and this job script file.
    mkdir -p \"\${job_dir}/\${jobid}\" || :
    cp \"\${param}\" \"\${job_dir}/\${jobid}/param\" || :
    this_file=\"\$(readlink -f \"\${BASH_SOURCE[0]}\")\"
    cp \"\${this_file}\" \"\${job_dir}/\${jobid}/jobscript\" || :
    # Let the user know why they should use
    # the concept script for submission.
    printf \\
\"Detected that this job was not submitted using the concept script.\\\n\\
Note that this comes with the following downsides:\\\n\\
1) The parameter file might have changed on disk\\\n\\
   since the job script was generated.\\\n\\
2) Any command-line parameters will be lost.\\\n\\\n\\
\" >> \"\${job_dir}/\${jobid}/log\"
fi

# Build the code as necessary
set_make_jobs --force
build_concept                                       \\
    >> \"\${job_dir}/\${jobid}/log\"                \\
    2>> >(tee -a \"\${job_dir}/\${jobid}/log_err\")

# Set variables used when invoking Python
prepare_python_options

# Print start messages
pure_python_msg=\"\"
[ \"\${pure_python}\" != \"True\" ] || pure_python_msg=\" in pure Python mode\"
colorprint                                                                                                       \\
    \"\\\nRunning \${esc_concept} job \${jobid}\${pure_python_msg} remotely on \$(hostname -f) (\${queue_display})\" \\
    \"yellow\"                                                                                                   \\
    >> \"\${job_dir}/\${jobid}/log\"
print_info

# Change to the concept directory
cd \"\${concept_dir}\"

# Run the code. Both stdout and stderr are being logged
# to job_dir/jobid/log, while the stderr alone is also logged
# to job_dir/jobid/log_err.
\${mpi_executor}                                         \\
    \"\${python}\" -B \${m_flag} \"\${main_as_library}\" \\
        \"param='\${param}'\"                            \\
        \"jobid='\${jobid}'\"                            \\
    >> \"\${job_dir}/\${jobid}/log\"                     \\
    2>> >(tee -a \"\${job_dir}/\${jobid}/log_err\")      \\

# Run complete. Do cleanup.
if [ -f \"\${job_dir}/\${jobid}/log_err\" ] && [ ! -s \"\${job_dir}/\${jobid}/log_err\" ]; then
    # Remove empty error log
    rm \"\${job_dir}/\${jobid}/log_err\"
else
    colorprint                                                            \\
        \"\\\nSome warnings/errors occurred during \${esc_concept} run!\" \\
        \"red\"                                                           \\
        >> \"\${job_dir}/\${jobid}/log\" 2>&1
    colorprint                                                  \\
        \"Check the following error log for more information:\" \\
        \"red\"                                                 \\
        >> \"\${job_dir}/\${jobid}/log\" 2>&1
    colorprint                                        \\
        \"\\\\\"\${job_dir}/\${jobid}/log_err\\\\\"\" \\
        \"red\"                                       \\
        >> \"\${job_dir}/\${jobid}/log\" 2>&1
fi
" > "${jobscript}"
    # Though not essential, make the job script executable,
    # reflecting the fact that it is a runnable script.
    chmod u+x "${jobscript}" 2>/dev/null || :
    # Exit now if no resource manager was found
    if [ -z "${resource_manager}" ]; then
        colorprint "Could not find any resource manager.\n\
Is Slurm/TORQUE/PBS installed and on the PATH?\n\
An almost complete job script has been saved to\n\
\"${jobscript}\"" "red"
        exit 1
    fi
    # Only submit if a queue is specified
    if [ "${queue}" == "${queue_unspecified}" ]; then
        printf "Job not submitted as no queue is specified.\n\
An almost complete job script has been saved to\n\
\"${jobscript}\"\n"
        trap : 0
        exit 0
    fi
    # Submit the remote job from within the job directory,
    # so that any auto-generated files will be dumped there.
    colorprint "\nSubmitting job" "yellow"
    if [ "${resource_manager}" == 'slurm' ]; then
        jobid=$(cd "${job_dir}" && sbatch "${jobscript}")
    elif [ "${resource_manager}" == 'torque' ]; then
        jobid=$(cd "${job_dir}" && qsub "${jobscript}")
    fi
    jobid=$(echo "${jobid}" | awk '{print $NF}')
    jobid="${jobid%.*}"
    # Create job sub-directory and copy over the parameter file
    mkdir -p "${job_dir}/${jobid}"
    copy_param
    # Move the job script into the job sub-directory.
    # Crucially, the resource manager should have taken a copy
    # for itself prior to this.
    mv "${jobscript}" "${job_dir}/${jobid}/jobscript" || :
    echo "Job ${jobid} submitted to queue ${queue}"
    # Deactivate traps before exiting
    trap : 0
    # Invoke the watch utility on the submitted job
    # unless --no-watching was supplied.
    if [ "${no_watching}" == "False" ]; then
        # Invoke the watch utility on the submitted job
        printf "You can now kill (Ctrl+C) this script without cancelling the job\n"
        "${concept}" -u watch "${jobid}"
    fi
    exit 0
else
    # Run locally
    if [[ "${nprocs}" == *':'* ]]; then
        nnodes=$("${python}" -B -c "print('${nprocs}'[:'${nprocs}'.index(':')])")
        if [ "${nnodes}" == "1" ]; then
            nprocs=$("${python}" -B -c "print('${nprocs}'[('${nprocs}'.index(':') + 1):])")
        else
            colorprint "You may not specify a number of nodes when running locally" "red"
            exit 1
        fi
    fi
    # Construct the next jobid that does not yet exist in the job directory,
    # using exponential followed by binary search.
    # of the job dir, using binary search.
    jobid=0
    while [ -d "${job_dir}/${jobid}" ]; do
        jobid=$((2*jobid + 1))
    done
    jobid_upper=${jobid}
    jobid_lower=$(((jobid - 1)/2))
    while [ ${jobid_lower} -lt ${jobid_upper} ]; do
        jobid=$(((jobid_lower + jobid_upper)/2))
        if [ -d "${job_dir}/${jobid}" ]; then
            jobid_lower=$((jobid + 1))
        else
            jobid_upper=${jobid}
        fi
    done
    jobid=${jobid_lower}
    # Create job sub-directory and copy over the parameter file
    mkdir -p "${job_dir}/${jobid}"
    copy_param
    # Set variables used when invoking Python
    prepare_python_options
    # Print start messages
    pure_python_msg=""
    [ "${pure_python}" != "True" ] || pure_python_msg=" in pure Python mode"
    colorprint "\nRunning ${esc_concept}${pure_python_msg}" "yellow" \
        | tee -a "${job_dir}/${jobid}/log"
    print_info
    # Run the code. Print stdout and stderr to the terminal while at the
    # same time logging them to job_dir/jobid/log. The stderr alone is also
    # logged to job_dir/jobid/log_err.
    # We want the exit code of mpiexec, but it is not easily retrieved
    # due to the logging. We therefore print the exit code to a
    # temporary file which we then read and discard.
    (                                                                   \
        (                                                               \
            "${mpiexec}" -n "${nprocs}" ${mpiexec_args}                 \
                "${python}" -B ${i_flag} ${m_flag} "${main_as_library}" \
                    "param='${param}'"                                  \
                    "jobid='${jobid}'"                                  \
            | tee -a "${job_dir}/${jobid}/log";                         \
            echo "${PIPESTATUS[0]}" > "${job_dir}/${jobid}/.exit_code"  \
        ) 3>&1 1>&2 2>&3                                                \
        | tee -a                                                        \
            "${job_dir}/${jobid}/log"                                   \
            "${job_dir}/${jobid}/log_err"                               \
    ) 3>&1 1>&2 2>&3
    # Get the exit code from the temporary file
    sleep_time=0.01
    sleep_max=3000
    slept=0
    while [ ! -f "${job_dir}/${jobid}/.exit_code" ] && [ ${slept} -lt ${sleep_max} ]; do
        sleep ${sleep_time}
        ((slept += 1))
    done
    exit_code=1
    if [ -f "${job_dir}/${jobid}/.exit_code" ]; then
        exit_code="$(cat "${job_dir}/${jobid}/.exit_code" || :)"
        if [ -z "${exit_code}" ]; then
            exit_code=1
        fi
        rm -f "${job_dir}/${jobid}/.exit_code"
    fi
    # Run complete. Do cleanup.
    if [ -f "${job_dir}/${jobid}/log_err" ]; then
        if [ ! -s "${job_dir}/${jobid}/log_err" ]; then
            # Remove empty error log
            rm "${job_dir}/${jobid}/log_err"
        else
            colorprint "\nSome warnings/errors occurred during ${esc_concept} run!\n\
Check the following error log for more information:\n\
\"${job_dir}/${jobid}/log_err\"" "red" 2>&1 | tee -a "${job_dir}/${jobid}/log" 1>&2
        fi
    fi
    # If the CO𝘕CEPT run exited erroneously, exit now
    if [ "${exit_code}" != "0" ]; then
        # Note that the real exit will be handled by the abort function
        exit ${exit_code}
    fi
    # Deactivate traps and exit
    trap : 0
    exit 0
fi
