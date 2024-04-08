#!/usr/bin/env bash

# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2024 Jeppe Mosgaard Dakin.
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
# Cursor movement
esc_up="${esc}[1A"
esc_erase="${esc}[K"
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

# Do not produce .pyc files
export PYTHONDONTWRITEBYTECODE=1

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
if [ "$(basename "${mpi_libdir}")" == "merged_lib_specified" ]; then
    if [ -d "${mpi_symlinkdir}" ]; then
        # Let mpi_symlinkdir go before merged_lib_specified
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${mpi_symlinkdir}"
    fi
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${mpi_libdir}"
else
    export LD_LIBRARY_PATH="${mpi_libdir}:${LD_LIBRARY_PATH}"
fi
# Additional symlinks to MPI libraries might be placed in mpi_symlinkdir
if [ -d "${mpi_symlinkdir}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${mpi_symlinkdir}"
    # If the special symlink named "ld_preload.so" is present
    # in mpi_symlinkdir, we should include this symlink in LD_PRELOAD.
    if [ -f "${mpi_symlinkdir}/ld_preload.so" ]; then
        if [ -n "${LD_PRELOAD}" ]; then
            export LD_PRELOAD="${LD_PRELOAD} ${mpi_symlinkdir}/ld_preload.so"
        else
            export LD_PRELOAD="${mpi_symlinkdir}/ld_preload.so"
        fi
    fi
fi

# The time before any computation begins.
# This time is saved both in seconds after the Unix epoch
# and in a human readable format.
start_time="$("${python}" -c "
import datetime
start_time_epoch = datetime.datetime.now().timestamp()
start_time_human = str(datetime.datetime.fromtimestamp(start_time_epoch))[:-3]
print(f'{start_time_epoch}_{start_time_human}')
")"
start_time_epoch="${start_time%%_*}"
start_time_human="${start_time#*_}"
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
    "${python}" -c "
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
    "${python}" -c "
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
# Print out the logo the first time an execution reaches this point.
# Do not print the logo if the file is beind sourced or if the version
# should be printed.
if     [ "${logo_printed}"  != "True"  ] \
    && [ "${being_sourced}" == "False" ] \
    && [[ " $@ " != *" -v "           ]] \
    && [[ " $@ " != *" --v "          ]] \
    && [[ " $@ " != *" --ve "         ]] \
    && [[ " $@ " != *" --ver "        ]] \
    && [[ " $@ " != *" --vers "       ]] \
    && [[ " $@ " != *" --versi "      ]] \
    && [[ " $@ " != *" --versio "     ]] \
    && [[ " $@ " != *" --version "    ]] \
; then
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
    # Place backslashes before spaces, dollar signs and parentheses.
    # These are needed when expanding tilde, but they will not persist.
    path="${path// /\\ }"
    path="${path//$/\\$}"
    path="${path//\(/\\(}"
    path="${path//\)/\\)}"
    # Expand tilde
    eval path="${path}"
    # Convert to absolute path
    path="$(readlink -m "${path}")"
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
    "${python}" -c "
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
        element=$("${python}" -c "
try:
    eval(\"${element}\")
    print(\"${element}\")
except Exception:
    print('\"{}\"'.format(\"${element}\"))
")
        # Append element to list
        list="$(echo "${list}")${element}, "
    done
    list="[$(echo "${list}")]"
    echo "${list}"
}

# Function for recursively removing empty sub-directories
# within the tmp directory (including the tmp directly itself).
cleanup_empty_tmp() {
    local d
    if [ ! -d "${tmp_dir}" ]; then
        return
    fi
    tmp_subdirs=("${tmp_dir}")
    deleted_any="False"
    while :; do
        tmp_subdirs_new=()
        for tmp_subdir in "${tmp_subdirs[@]}"; do
            for d in "${tmp_subdir}/"*; do
                if [ ! -d "${d}" ]; then
                    continue
                fi
                is_empty="True"
                for f in "${d}/"*; do
                    if [ "${f}" != "${d}/*" ]; then
                        # Much faster than ls -A
                        is_empty="False"
                        break
                    fi
                done
                if [ "${is_empty}" == "True" ]; then
                    rm -rf "${d}" || :
                    deleted_any="True"
                else
                    tmp_subdirs_new=("${tmp_subdirs_new[@]}" "${d}")
                fi
            done
        done
        if [ ${#tmp_subdirs_new} -eq 0 ]; then
            break
        fi
        tmp_subdirs=("${tmp_subdirs_new[@]}")
    done
    if [ "${deleted_any}" == "True" ]; then
        if [ -z "$(ls -A "${tmp_dir}")" ]; then
            rm -rf "${tmp_dir}" || :
        else
            cleanup_empty_tmp
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
            mpi_warning="$("${mpiexec}" -n 1 "${python}" -c '
import mpi4py.rc; mpi4py.rc.threads = False  # Do not use threads
from mpi4py import MPI
' 2>&1 || :)"
        else
            mpi_warning="$("${python}" -c '
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
build_default="${build_dir}"
command_line_params_default=""
interactive_default="False"
job_directive_default=""
job_name_default="concept"
linktime_optimizations_default="True"
local_default="False"
main_default="${src_dir}/main.py"
memory_default="-1"  # -1 implies unset
native_optimizations_default="False"
nprocs_default="1"
optimizations_default="True"
param_default="None"  # None implies unset
pure_python_default="False"
queue_default=""
rebuild_default=""  # empty implies that a rebuild is triggered by changes to the source
safe_build_default="True"
submit_default=""  # empty implies submission if other remote job options are specified
tests_default=""
utility_default=""
walltime_default="00:00:00"  # 00:00:00 implies unset
watch_default="True"

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
    start_time_build=$("${python}" -c "import time; print(time.time())")
    set_make_jobs
    make_concept() {
        make                                                   \
            build="${build}"                                   \
            optimizations="${optimizations}"                   \
            linktime_optimizations="${linktime_optimizations}" \
            native_optimizations="${native_optimizations}"     \
            safe_build="${safe_build}"                         \
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
        "${mpiexec}" -n 1 "${python}" -c "from commons import *
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
    if [ -n "${job_name}" ] && [ "${job_name}" != "${job_name_default}" ]; then
        echo "Job name:        ${job_name}" \
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
args=$("${python}" -c "
import argparse, math, os, re, sys
# Function which checks whether input is a representation of
# one or two positive integers. If two ints are given,
# separate them by a colon.
def positive_int_or_int_pair(value, value_input=None):
    value = str(value)
    if value_input is None:
        value_input = value
    def raise_argparse_exception():
        raise argparse.ArgumentTypeError(
            f\"invalid positive int or int pair: '{value_input}'\"
        )
    for sep in ',;':
        value = value.replace(sep, ':')
    if value == '' or value.count(':') > 1:
        raise_argparse_exception()
    elif value.count(':') == 1:
        values = value.split(':')
        return ':'.join(positive_int_or_int_pair(value, value_input) for value in values)
    else:  # value.count(':') == 0
        try:
            value_eval = eval(value)
        except Exception:
            return positive_int_or_int_pair(value.replace(' ', ':'), value_input)
        try:
            value_float = float(value_eval)
        except Exception:
            raise_argparse_exception()
        value_int = int(value_float)
        if value_int == value_float and value_int > 0:
            return str(value_int)
        raise_argparse_exception()
# Function which checks whether input is a representation of
# a memory size and converts it to bytes.
def memory(value):
    value = str(value)
    value_raw = value
    def raise_argparse_exception():
        raise argparse.ArgumentTypeError(f\"invalid memory value: '{value_raw}'\")
    # Convert to (whole) bytes
    value = value.lower()
    value = value.replace(' ', '').replace('b', '')
    value = re.subn(r'([0-9]+)([a-z]+)', r'\g<1>*\g<2>', value)[0]
    units = {
        'k': 2**10,
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
    except Exception:
        raise_argparse_exception()
    return value
# Function which converts a time value to the format hh:mm:ss
def time(value):
    value = str(value)
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
    except Exception:
        pass
    # Attempt to interpret the time as an expression
    # like '2hr + 30mins'.
    if isinstance(value, str):
        value = value.lower()
        value = re.subn(r'([0-9]+)([a-z]+)', r'\g<1>*\g<2>', value)[0]
        try:
            value = int(math.ceil(float(eval(value, units))))
        except Exception:
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
# Function which checks whether input is a Boolean
def bool_from_str(value):
    value_raw = value
    value = str(value).lower()
    if value.startswith('y') or value in {'true',  't', 'on',  'enable',  '1', '1.0'}:
        return 'True'
    if value.startswith('n') or value in {'false', 'f', 'off', 'disable', '0', '0.0'}:
        return 'False'
    raise argparse.ArgumentTypeError(f\"invalid bool value: '{value_raw}'\")
# Function which checks whether input is a Boolean or is empty
def bool_or_empty(value):
    value = str(value).strip()
    if not value:
        return value
    return bool_from_str(value)
# Function which checks whether input is a Boolean,
# with empty input defaulting to True.
def bool_truebydefault(value):
    value = str(value).strip()
    if not value:
        return 'True'
    return bool_from_str(value)
# Function for setting up a parser for command-line arguments
def get_parser(*, add_help=True):
    def add_parser_argument(name, group, help, default, useenv=True, shortname=False, **kwargs):
        names = []
        if shortname:
            names.append(f'-{name[0]}')
        names.append(f'--{name}')
        if useenv:
            default = os.environ.get(f'CONCEPT_{name}'.replace('-', '_'), default)
        if kwargs.get('action') != 'count':
            default = str(default)
        if kwargs.get('action') == 'append' or kwargs.get('nargs') == '+':
            default = [default]
        group.add_argument(
            *names,
            help=help,
            default=default,
            **kwargs,
        )
    parser = argparse.ArgumentParser(
        prog='$(basename "${this_file}")',
        description='run the ${esc_concept} code',
        add_help=False,
    )
    # Basics
    group_basic = parser.add_argument_group('basics')
    if add_help:
        add_parser_argument(
            'help',
            group_basic,
            help='show this help message and exit',
            default=False,
            shortname=True,
            action='help'
    )
    add_parser_argument(
        'param',
        group_basic,
        help='parameter file to use',
        default='${param_default}',
        shortname=True,
    )
    add_parser_argument(
        'command-line-params',
        group_basic,
        help=(
            'specify parameter(s) directly from the command-line. '
            'If a parameter file is specified as well, '
            'the command-line parameters will take precedence. '
            'This option may be specified multiple times.'
        ),
        default='${command_line_params_default}',
        shortname=True,
        action='append',
    )
    add_parser_argument(
        'nprocs',
        group_basic,
        help=(
            'total number of processes '
            'or number of nodes and number of processes per node'
        ),
        default='${nprocs_default}',
        shortname=True,
        type=positive_int_or_int_pair,
    )
    add_parser_argument(
        'utility',
        group_basic,
        help=(
            'run utility UTILITY. UTILITY can be any executable '
            'in the util directory.'
        ),
        default='${utility_default}',
        useenv=False,  # do not make use of environment variable
        shortname=True,
        nargs='+',
    )
    # Remote job submission
    group_job = parser.add_argument_group('remote job submission')
    add_parser_argument(
        'submit',
        group_job,
        help=(
            'if True, submits the run as a remote job. '
            'By default submission takes place if other '
            'remote job submission options are specified.'
        ),
        default='${submit_default}',
        nargs='?',
        type=bool_or_empty,
        const='True',
    )
    add_parser_argument(
        'queue',
        group_job,
        help='queue/partition for submission of the remote job',
        default='${queue_default}',
        shortname=True,
    )
    add_parser_argument(
        'walltime',
        group_job,
        help='maxium allowed wall time for remote job',
        default='${walltime_default}',
        shortname=True,
        type=time,
    )
    add_parser_argument(
        'memory',
        group_job,
        help='total memory allocated for remote job',
        default='${memory_default}',
        type=memory,
    )
    add_parser_argument(
        'job-name',
        group_job,
        help='name of the job to be used with Slurm/TORQUE/PBS',
        default='${job_name_default}',
    )
    add_parser_argument(
        'job-directive',
        group_job,
        help=(
            'specify an additional line to add to the job script '
            'header for remote jobs. This option may be specified '
            'multiple times.'
        ),
        default='${job_directive_default}',
        shortname=True,
        action='append',
    )
    add_parser_argument(
        'watch',
        group_job,
        help=(
            'if True, follow submitted jobs via the watch utility '
            'upon remote submission. Defaults to ${watch_default}.'
        ),
        default=${watch_default},
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    # Building and running
    group_build = parser.add_argument_group('building and running')
    add_parser_argument(
        'local',
        group_build,
        help=(
            'if True, forces the run to be carried out locally, '
            'without submitting it as a remote job. '
            'Defaults to ${local_default}.'
        ),
        default='${local_default}',
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    add_parser_argument(
        'pure-python',
        group_build,
        help=(
            'if True, run in pure Python mode. '
            'Defaults to ${pure_python_default}.'
        ),
        default=${pure_python_default},
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    add_parser_argument(
        'build',
        group_build,
        help='build directory to use',
        default='${build_default}',
        shortname=True,
    )
    add_parser_argument(
        'rebuild',
        group_build,
        help=(
            'specify whether to rebuild the code before running it. '
            'By default the code is built if any of the source files '
            'have been changed since the last build.'
        ),
        default='${rebuild_default}',
        nargs='?',
        type=bool_or_empty,
        const='True',
    )
    add_parser_argument(
        'optimizations',
        group_build,
        help=(
            'if True, enables compiler optimizations. '
            'Defaults to ${optimizations_default}.'
        ),
        default='${optimizations_default}',
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    add_parser_argument(
        'linktime-optimizations',
        group_build,
        help=(
            'if True, allows the compiler to make use of '
            'link time optimizations. '
            'Defaults to ${linktime_optimizations_default}.'
        ),
        default='${linktime_optimizations_default}',
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    add_parser_argument(
        'native-optimizations',
        group_build,
        help=(
            'if True, allows the compiler to generate non-portable '
            'code optimized specifically for this machine. '
            'Defaults to ${native_optimizations_default}.'
        ),
        default='${native_optimizations_default}',
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    add_parser_argument(
        'safe-build',
        group_build,
        help=(
            'if False, dependencies between modules will be ignored '
            'when building. Defaults to ${safe_build_default}.'
        ),
        default=${safe_build_default},
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    # Special
    group_special = parser.add_argument_group('specials')
    add_parser_argument(
        'tests',
        group_special,
        help=(
            'run tests TESTS. TESTS can be one or more of the '
            'subdirectories within the test directory. '
            'Use TESTS=all to run all tests.'
        ),
        default='${tests_default}',
        shortname=True,
        nargs='+',
    )
    add_parser_argument(
        'main',
        group_special,
        help=(
            'entry point of the program. '
            'Can be a Python filename or command.'
        ),
        default='${main_default}',
        shortname=True,
    )
    add_parser_argument(
        'interactive',
        group_special,
        help='inspect interactively after program execution',
        default='${interactive_default}',
        shortname=True,
        nargs='?',
        type=bool_truebydefault,
        const='True',
    )
    add_parser_argument(
        'version',
        group_special,
        help='print version and exit',
        default=False,
        shortname=True,
        action='store_true',
    )
    add_parser_argument(
        'eeffoc-ekam'[::-1],
        group_special,
        help='if only ...',
        default=0,
        action='count',
    )
    add_parser_argument(
        'esaelp'[::-1],
        group_special,
        help=argparse.SUPPRESS,
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
utility = args.utility
utility_args = ['${utility_default}']
if isinstance(args.utility, list):
    utility = args.utility[0]
    if len(args.utility) > 1:
        utility_args = args.utility[1:]
if utility != '${utility_default}':
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
print(' ;:; '.join([
    \"argparse_finished=True\",
    \"build='{}'\".format(str(args.build).replace(\"'\", '\"')),
    \"command_line_params='{}'\".format(
        '\\n'.join([
            arg for arg in args.command_line_params if arg.strip()
        ]).replace(\"'\", '\"')
    ),
    \"interactive={}\".format(args.interactive),
    \"job_directive='{}'\".format(
        ' ; '.join([
            arg for arg in args.job_directive if arg.strip(' ;')
        ]).replace(\"'\", '\"')
    ),
    \"main='{}'\".format(str(args.main).replace(\"'\", '\"')),
    \"memory='{}'\".format(args.memory),
    \"nprocs={}\".format(args.nprocs),
    \"param='{}'\".format(args.param),
    \"queue='{}'\".format(args.queue),
    \"tests='{}'\".format(
        ';'.join([
            arg for arg in args.tests if arg.strip(' ;')
        ]).replace(\"'\", '\"')
    ),
    \"utility='{}'\".format(utility),
    \"utility_args={}\".format(python_list2bash_array(
        [arg for arg in utility_args if arg.strip()]
    )),
    \"walltime={}\".format(args.walltime),
    \"job_name='{}'\".format(args.job_name),
    \"local={}\".format(args.local),
    \"submit={}\".format(args.submit),
    \"optimizations={}\".format(args.optimizations),
    \"linktime_optimizations={}\".format(args.linktime_optimizations),
    \"native_optimizations={}\".format(args.native_optimizations),
    \"rebuild={}\".format(args.rebuild),
    \"safe_build={}\".format(args.safe_build),
    \"pure_python={}\".format(args.pure_python),
    \"watch={}\".format(args.watch),
    \"version={}\".format(args.version),
    \"{}={}\".format('eeffoc_ekam', getattr(args, 'eeffoc_ekam'[::-1])),
    \"{}={}\".format('esaelp', getattr(args, 'esaelp'[::-1])),
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
    version_printout="$(
        "${concept}" -m "import commons; print('version', commons.__version__)" \
            --pure-python --local --submit=False 2>/dev/null | tail -n 1
    )"
    if [[ "${version_printout}" == "version "* ]]; then
        version_printout="$(echo "${version_printout}" | awk '{print $NF}')"
        if [ "${version_printout}" != "version" ] && [ -n "${version_printout}" ]; then
            echo "${version_printout}"
            trap : 0
            exit 0
        fi
    fi
    colorprint "Could not determine ${esc_concept} version!" "red"
    exit 1
fi

# I wonder what this obfuscated code does?
reverse() {
    local s="$1"
    local r=""
    for ((i = 0; i < ${#s}; i += 1)); do
        r="${s:i:1}${r}"
    done
    echo "${r}"
}
if [ ${eeffoc_ekam} -gt 0 ]; then
    r="getattr(__import__('modnar'[::-1]), 'modnar'[::-1])()"
    if [ "${esaelp}" == "False" ]; then
        esaelp="$("${python}" -c "print(${r} < 0.15/${eeffoc_ekam}**0.5)")"
    fi
    if [ "${esaelp}" == "False" ]; then
        "${python}" -c "
print(
    getattr(__import__('modnar'[::-1]), 'eciohc'[::-1])(
        ['ylecin ksa uoy fi ebyaM']*10
        + ['retal spahreP']*3
        + ['esaelp yaS']
    )[::-1] + ' ' + '.'*3
)
"
        trap : 0
        exit 1
    fi
    if [ "$("${python}" -c "print(${r} < 0.05)")" == "True" ]; then
        "${python}" -c "
import sys
from blessings import Terminal
terminal = Terminal(force_styling=True)
print(
    terminal.bold_red(
        ' :gnitrobA'[::-1] + getattr(__import__('modnar'[::-1]), 'eciohc'[::-1])(
            ['knat retaw llifeR']
            + ['snaeb hguone toN']
            + ['spuc naelc fo tuO']
            + ['redro fo tuo yliraropmeT']
            + ['egatrohs rewop lacol ot eud elbaliavanu ecivreS']
        )[::-1],
    ),
    file=sys.stderr,
)
"
        trap : 0
        exit 1
    fi
    if [ "${param}" != "${param_default}" ]; then
        colorprint "$(reverse ".desu eb lliw epicer dradnats ehTn\.\
\"$(reverse ${param})\" nihtiw gniwerb eeffoc rof snoitcurtsni tneiciffusnI :gninraW")" "red"
    fi
    if [[ "${nprocs}" == *':'* ]]; then
        nnodes="${nprocs%:*}"
        if [ ${nnodes} -gt 1 ]; then
            colorprint "$(reverse "elbaliava edon eeffoc elgnis a ylnO :gninraW")" "red"
        fi
        nprocs="${nprocs#*:}"
    fi
    if [ "${submit}" == "True" ]; then
        colorprint "$(reverse "devres eb lliw eeffoc lacol ylnO :gninraW")" "red"
    fi
    "${python}" -c "
import itertools
def primes():
    for n in itertools.count(2):
        for i in range(2, 1 + int(n**0.5)):
            if n%i == 0:
                break
        else:
            yield n
getp = lambda n: list(zip(range(n), primes()))[-1][-1]
s = lambda t=None: getattr(__import__('emit'[::-1]), 'peels'[::-1])(
    0.15 + 1.2*${eeffoc_ekam}**0.5/${nprocs}*${r} if t is None else t
)
def werb():
    print(' gniwerB'[::-1], end='', flush=True)
    for i in range(3):
        s()
        if i > 1 and ${r} < 0.05:
            print(chr(5*getp(3*941)), end='', flush=True)
            s(0.7)
            print('!taht tuoba yrroS  '[::-1], end='', flush=True)
            s(1.5)
            print('\r\x1b[K', end='', flush=True)
            s(0.5)
            return werb()
        print(chr(46), end='', flush=True)
    s()
try:
    print('\x1b[?25l', end='')
    werb()
    print('\r\x1b[K', end='', flush=True)
    text = ' :og uoy ereH'[::-1]
    c = chr(getp(3*401))
    v_possibilities = []
    for v_max in range(1, 50):
        v_possibilities += list(range(2, 2 + 2*max(${eeffoc_ekam}, v_max//2), 2))
    __import__('modnar'[::-1]).shuffle(v_possibilities)
    v_values = set()
    for v in v_possibilities:
        v_values.add(v)
        if len(v_values) == ${eeffoc_ekam}:
            break
    v_possibilities = sorted(v_values)
    indices = set()
    for throw in range(${eeffoc_ekam}):
        print(f'\r{text}', end='', flush=True)
        if throw == 0:
            s(0.25)
        if throw == 0:
            s(0.2)
        v = v_possibilities.pop()
        for i in range(v - 1):
            if i > 0 and i not in indices:
                print(f'\x1b[2D ', end='')
            print(c, end='', flush=True)
            v -= 1
            s(0.35/v)
        indices.add(i + 1)
except KeyboardInterrupt:
    pass
finally:
    print('\x1b[?12l\x1b[?25h', flush=True)
"
    trap : 0
    exit 0
fi

# Export command-line arguments as environment variables
while read arg; do
    arg_name="${arg%%=*}"
    arg_val="${arg#*=}"
    arg_val="$(printf "${arg_val}")"  # expand \n
    eval "arg_val=${arg_val}"
    if [ "${arg_name}" == "utility" ]; then
        # If a utility is specified by the path to the corresponding
        # executable, reduce this to the utility name only.
        arg_val="$(basename "${arg_val}" || :)"
    fi
    eval "export CONCEPT_${arg_name}=\${arg_val}"
done <<< "$("${python}" -c "
for arg in '''${args}'''.split(';:;'):
    print(arg.strip().replace('\n', r'\\\\n'))
")"

# Display warning if a build directory is selected
# while running in pure Python mode.
if [ "${pure_python}" == "True" ] && [ "${build}" != "${build_dir}" ]; then
    colorprint "Warning: Ignoring the specified build directory \"${build}\" \
as running in pure Python mode" "red"
fi

# Display warning if the requested memory is below one megabyte
if [ "${memory}" != "${memory_default}" ] && [ ${memory} -ne 0 ] && [ ${memory} -lt 1048576 ]; then
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
if [ "${param}" != "${param_default}" ]; then
    param_ori="${param}"
    param="$(absolute_path "${param}")"
    if [ ! -f "${param}" ] && [ -f "${param_dir}/${param_ori}" ]; then
        param="${param_dir}/${param_ori}"
    fi
fi
if [ "${tests}" != "${tests_default}" ]; then
    IFS=';' read -ra tests_arr <<< "${tests}"
    tests=""
    sep=""
    for test in "${tests_arr[@]}"; do
        if [[ "${test}" == *"/run" ]]; then
            test="$(dirname "${test}")"
        fi
        if [ "${test}" != "all" ]; then
            test="${test_dir}/$(basename "${test}")"
        fi
        tests="${tests}${sep}${test}"
        sep=";"
    done
fi
if [ "${utility}" != "${utility_default}" ]; then
    utility="${util_dir}/$(basename "${utility}")"
fi

# Use the main.{py/so} of the build instead of the source.
# Error out if neither path exists. If a utility is to be run,
# do not do this now, but wait until this script is called back
# from the utility.
if [ "${main_as_command}" == "False" ] && [ "${utility}" == "${utility_default}" ]; then
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
if [ "${param}" != "${param_default}" ] && [ ! -f "${param}" ]; then
    if [ -d "${param}" ]; then
        colorprint "Error: Parameter file \"${param}\" is a directory" "red"
    else
        colorprint "Error: Parameter file \"${param}\" does not exist" "red"
    fi
    exit 1
fi
if [ "${tests}" != "${tests_default}" ]; then
    IFS=';' read -ra tests_arr <<< "${tests}"
    for test in "${tests_arr[@]}"; do
        if [ "${test}" == "all" ] || [ -d "${test}" ]; then
            continue
        fi
        if [ -f "${test}" ]; then
            colorprint "Error: Test \"${test}\" is a file but should be a directory" "red"
        else
            colorprint "Error: Test \"${test}\" does not exist" "red"
        fi
        exit 1
    done
fi
if [ "${utility}" != "${utility_default}" ] && [ ! -f "${utility}" ]; then
    if [ -d "${utility}" ]; then
        colorprint "Error: Utility \"${utility}\" is a directory" "red"
    else
        colorprint "Error: Utility \"${utility}\" does not exist" "red"
    fi
    exit 1
fi

# Check for syntax errors in supplied parameter file.
# Other types of errors will not be detected before runtime.
if [ "${param}" != "${param_default}" ]; then
    param_traceback="$("${python}" -c "
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
    command_line_params_traceback="$("${python}" -c "
import ast
source = '''${command_line_params}'''
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
if     [ "${local}" == "False"            ] \
    && [ "${tests}" == "${tests_default}" ] \
    && [ "${ssh}"   == "True"             ]; then
    remote="True"
fi
if [ "${remote}" == "True" ] && [ "${interactive}" == "True" ]; then
    colorprint "Cannot use interactive mode when running remotely" "red"
    exit 1
fi

# Determine whehter to actually submit remote job
if [ "${remote}" == "False" ] && [ "${submit}" == "True" ]; then
    colorprint "Cannot submit job when running locally" "red"
    exit 1
fi
if [ -z "${submit_actual}" ]; then
    submit_actual="False"
fi
if [ -n "${submit}" ]; then
    submit_actual="${submit}"
elif [ "${called_from_concept}" != "True" ]; then
    if [ "${queue}" != "${queue_default}" ]; then
        if [ "${remote}" == "True" ]; then
            submit_actual="True"
        else
            colorprint "Queue (${queue}) specified while running locally. \
This will be ignored" "red"
        fi
    fi
    if [[ "${nprocs}" == *':'* ]]; then
        if [ "${remote}" == "True" ]; then
            submit_actual="True"
        else
            nnodes="${nprocs%:*}"
            colorprint "Number of nodes (${nnodes}) specified while running locally. \
This will be ignored" "red"
        fi
    fi
    if [ "${walltime}" != "${walltime_default}" ]; then
        if [ "${remote}" == "True" ]; then
            submit_actual="True"
        else
            colorprint "Wall time (${walltime}) specified while running locally. \
This will be ignored" "red"
        fi
    fi
    if [ "${memory}" != "${memory_default}" ]; then
        if [ "${remote}" == "True" ]; then
            submit_actual="True"
        else
            colorprint "Memory (${memory}) specified while running locally. \
This will be ignored" "red"
        fi
    fi
    if [ "${job_name}" != "${job_name_default}" ]; then
        if [ "${remote}" == "True" ]; then
            submit_actual="True"
        else
            colorprint "Job name (${job_name}) specified while running locally. \
This will be ignored" "red"
        fi
    fi
    if [ "${job_directive}" != "${job_directive_default}" ]; then
        if [ "${remote}" == "True" ]; then
            submit_actual="True"
        else
            colorprint "Job directive(s) (${job_directive}) specified while running locally. \
This will be ignored" "red"
        fi
    fi
    if [ "${watch}" != "${watch_default}" ]; then
        if [ "${remote}" == "True" ]; then
            submit_actual="True"
        else
            colorprint "Watch (${watch}) specified while running locally. \
This will be ignored" "red"
        fi
    fi
fi

# If tests are to be run, run them and exit
if [ "${tests}" != "${tests_default}" ]; then
    # Function for running a single test, completely containing it
    # within an artifact subdirectory.
    run_test_specific() {
        test="$1"
        test="$(basename "${test}")"
        start_time_test=$("${python}" -c "import time; print(time.time())")
        colorprint "\nRunning ${test} test" "yellow"
        artifact_dir="${test_dir}/${test}/artifact"
        rm -rf "${artifact_dir}"
        mkdir "${artifact_dir}"
        for f in "${test_dir}/${test}/"*; do
            if [ "${f}" == "${artifact_dir}" ]; then
                continue
            fi
            cp -r "${f}" "${artifact_dir}/"
        done
        "${artifact_dir}/run"
        test_check_err_logs
        colorprint "${test} test ran successfully" "green"
        # Print out the execution time for this test
        "${mpiexec}" -n 1 "${python}" -c "from commons import *
print('Total execution time: {}'.format(time_since(${start_time_test})))"
    }
    # Function for running given test (including "all"),
    # with time measurements added.
    run_test() {
        test="$1"
        if [ "${test}" == "all" ]; then
            # Find and run all tests
            tests_all="$(cd "${test_dir}" && "${python}" -c "
from glob import glob
# List tests in order of required execution.
# Tests not included here will be run last.
order = [
    # Test whether the code is able to compile and run
    'basic',
    # Tests of the CLASS installation,
    # the Friedmann equation, realisations
    # and the power spectrum and bispectrum functionality.
    'friedmann',
    'realize',
    'powerspec',
    'bispec',
    # Test of the GADGET-2 installation
    'gadget',
    # Tests of the particle implementation
    'drift_nohubble',
    'drift',
    'kick_pp_without_ewald',
    'kick_pp_with_ewald',
    'lpt',
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
    # Test particle IDs
    'ids',
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
    'classutil',
    'render',
]
# Find all tests (directories in test_dir).
# Skip test if its (directory) name has a leading underscore.
tests = (dir.rstrip('/') for dir in glob('*/') if not dir.startswith('_'))
# Sort the tests based on the order given above
sorted_tests = sorted(
    tests,
    key=(lambda test: (order.index(test), test) if test in order else (len(order), test)),
)
for test in sorted_tests:
    print(test)
"
            )"
            # Run all tests in the test directory
            printf "\nThe following tests will be run in order:\n"
            for test in ${tests_all}; do
                echo "    ${test}"
            done
            echo
            (cd "${concept_dir}" && make clean-test)
            for test in ${tests_all}; do
                run_test_specific "${test}"
            done
            colorprint "\nAll tests ran successfully" "green"
        else
            # Run specific test
            run_test_specific "${test}"
        fi
    }
    # Function which can extract parameters from the parameter file
    # given by "${this_dir}/param". This is used by some tests.
    get_param() {
        "${concept}"                              \
            -n=1                                  \
            -p="${this_dir}/param"                \
            -c="${command_line_params}"           \
            -m="
from commons import *
${2}
try:
    print('param_var =', ${1})
except NameError:
    print('param_var =', user_params['''${1}'''])
"                                                 \
            --pure-python                         \
            --local --submit=False                \
        | grep "param_var"                        \
        | tail -n 1                               \
        | sed 's/^.\{12\}//'
    }
    # If the job directory is clean,
    # we count any generated *_err file as a test failure.
    test_check_err_logs() { :; }
    if [ ! -d "${job_dir}" ] || [ -z "$(ls "${job_dir}")" ]; then
        test_check_err_logs() {
            for f in "${job_dir}/"*"/log_err"; do
                if [ "${f}" == "${job_dir}/*/log_err" ]; then
                    continue
                fi
                if [ -f "${f}" ] && [ -s "${f}" ]; then
                    colorprint "It looks like ${test} test generated error messages.\n\
This is counted as a test failure." "red"
                    exit 1
                fi
            done
        }
    fi
    # Unset the tests environment variable so that subprocesss are free
    # to perform the actual testing (otherwise we get stuck in the
    # current initializing phase).
    unset CONCEPT_tests
    # Ensure that the testing is carried out locally
    export CONCEPT_local="True"
    export CONCEPT_submit="False"
    # Enable warnings that are ignored by default.
    # We do this in order to get notified about deprecated code usage.
    # Due to the above test_check_err_logs(), any such warning will
    # lead to test failure if starting with an empty job directory.
    if [ -z "${PYTHONWARNINGS}" ]; then
        export PYTHONWARNINGS=default
    fi
    # Do the test(s)
    IFS=';' read -ra tests_arr <<< "${tests}"
    for test in "${tests_arr[@]}"; do
        run_test "${test}"
    done
    # If several tests have been run,
    # print out the total time it took to run them all.
    if [ "${tests}" == "all" ] || [ ${#tests_arr[@]} -gt 1 ]; then
        "${mpiexec}" -n 1 "${python}" -c "from commons import *
print('Total execution time: {}'.format(time_since(${start_time_epoch})))"
    fi
    # Deactivate traps and exit
    trap : 0
    exit 0
fi

# Function implementing functionality common to many utilities,
# typically involving some processing of snapshots.
launch_utility() {
    # Arguments: Product name (singular), product name (plural),
    # yes to defaults, default utility parameters,
    # final utility parameter specifications, paths.
    local product_singular="$1"
    local product_plural="$2"
    local yes_to_defaults="$3"
    local utility_params="$4"
    local utility_params_closing="$5"
    local paths=("${@:6}")
    extract_info() {
        local text="$1"
        local pattern="$2"
        "${python}" -c                                                           \
            "import re; print(re.search(r'${pattern}', '''${text}''').group(1))" \
            2>/dev/null                                                          \
        || :
    }
    # Apply the CONCEPT_local and CONCEPT_submit variables
    if [ -n "${CONCEPT_local}" ]; then
        local="${CONCEPT_local}"
    fi
    if [ "${local}" == "True" ]; then
        remote="False"
    fi
    if [ -n "${CONCEPT_submit}" ]; then
        submit="${CONCEPT_submit}"
        submit_actual="${submit}"
    fi
    # Read in user-supplied parameters
    user_params=""
    if [ "${param}" != "${param_default}" ]; then
        user_params="$(cat "${param}")"
    fi
    # Get filenames of snapshots and create matching
    # (temporary) parameter files.
    snapshot_param_fake="${tmp_dir}/param/${start_time_human_nosep}/${CONCEPT_utility}/None"
    if [ ${#paths[@]} -gt 0 ]; then
        # Convert to absolute paths
        for i in ${!paths[@]}; do
            paths[${i}]="$(absolute_path "${paths[${i}]}" "${workdir}")"
        done
        # Get filenames
        info="$(                                                                       \
            "${concept}"                                                               \
                -u=info                                                                \
                    "$(bash_array2python_list "${paths[@]}")"                          \
                    --generate-param="${tmp_dir}/param/${start_time_human_nosep}/info" \
                --local --submit=False                                                 \
        )"
        snapshot_filenames="$(echo "${info}" | grep -x 'Parameters.*' | grep -o '".*"')"
        snapshot_param_filenames="$(                                 \
            echo "${info}"                                           \
            | grep -x 'The above parameters have been written to .*' \
            | grep -o '".*"'                                         \
        )"
        snapshot_param_filenames="${snapshot_param_filenames//\"/}"
        # Print out the snapshots which will be processed
        n_snapshots="$(echo "${snapshot_filenames}" | wc -l)"
        if [ ${n_snapshots} == 1 ]; then
            echo "${product_singular} will be produced of the following snapshot:"
        else
            echo "${product_plural} will be produced of the following snapshots:"
        fi
        echo "${snapshot_filenames}"
        # Spawning many simultaneous remote jobs which read in snapshots
        # can put the file system under a lot of stress.
        # If several snapshots are to be processed remotely,
        # prompt the user for confirmation.
        if [ "${remote}" == "True" ] && [ ${n_snapshots} -gt 1 ] \
            && [ "${submit_actual}" == "True" ] \
        ; then
            if [ "${yes_to_defaults}" == "True" ]; then
                printf "\nA job will be submitted for each snapshot\n"
            else
                while :; do
                    read -p "
Simultaneous submission of many jobs which read from disk
may put the file system under severe stress.
Would you like to submit a job for each
of the above snapshots anyway? [Y/n] " yn
                    case "${yn}" in
                        [Yy]*)
                            break
                            ;;
                        [Nn]*)
                            trap : 0
                            exit 0
                            ;;
                        "")
                            break
                            ;;
                        *)
                            ;;
                    esac
                done
            fi
        fi
    else
        # No paths given. Run CO𝘕CEPT once.
        n_snapshots=1
        snapshot_filenames="'None'"
        snapshot_param_filenames="${snapshot_param_fake}"
    fi
    # Run the CO𝘕CEPT code for each snapshot
    utility_param_dirname="${tmp_dir}/param/${start_time_human_nosep}/${CONCEPT_utility}"
    mkdir -p "${utility_param_dirname}"
    jobids=()
    for ((i = 1; i <= n_snapshots; i += 1)); do
        # Filename of the parameter file to create and use
        utility_param="${utility_param_dirname}/$((i - 1))"
        # Get the i'th snapshot and generated parameter file
        snapshot_filename="$(echo "${snapshot_filenames}"       | sed "${i}q;d")"
        snapshot_param="$(echo    "${snapshot_param_filenames}" | sed "${i}q;d")"
        # Make temporary parameter file with everything needed
        echo "
##################
# Utility header #
##################
# Set the path to the parameter file to be the path to the actual
# parameter file specified by the user, not this auto-generated
# parameter file.
_param_ori, param = param, type(param)('${param}')

#########################################################
# Parameters deduced from snapshot ${snapshot_filename} #
#########################################################
$([ "${snapshot_param}" == "${snapshot_param_fake}" ] || cat "${snapshot_param}")
# Record the scale factor and boxsize
try:
    _a_begin = a_begin
except NameError:
    _a_begin = None
try:
    _boxsize = boxsize
except NameError:
    _boxsize = None

#########################################
# ${CONCEPT_utility} utility parameters #
#########################################
${utility_params}
special_params['snapshot_filename'] = ${snapshot_filename}

############################
# User-supplied parameters #
############################
${user_params}

##################
# Utility footer #
##################
# Reinsert original path to the parameter file
param = _param_ori
# Use scale factor and boxsize from snapshot
if _a_begin is not None:
    a_begin = _a_begin
if _boxsize is not None:
    boxsize = _boxsize

#################################################
# ${CONCEPT_utility} closing utility parameters #
#################################################
${utility_params_closing}
" > "${utility_param}"
        # Cleanup
        rm -f "${snapshot_param}"
        # Run CO𝘕CEPT to process the snapshot.
        # Submitted jobs should not be watched at this time.
        # Capture the jobid and exit code.
        if [ "${remote}" == "True" ] && [ "${submit_actual}" == "False" ]; then
            echo
        fi
        exec 4>&1
        jobid_and_exit_code="$(                \
            "${concept}"                       \
                -p "${utility_param}"          \
                --watch=False                  \
            | tee >(cat >&4)                   \
            | grep "^Job "                     \
            | head -n 1                        \
            ; echo "exit_code${PIPESTATUS[0]}" \
        )"
        exec 4>&-
        exit_code="$(extract_info "${jobid_and_exit_code}" "exit_code(\d+)")"
        if [ -z "${exit_code}" ]; then
            colorprint "Error capturing exit code" "red"
            exit 1
        elif [ ${exit_code} != 0 ]; then
            exit ${exit_code}
        fi
        jobid="$(extract_info "${jobid_and_exit_code}" "Job .*?(\d+)")"
        jobids=("${jobids[@]}" "${jobid}")
        # Cleanup
        if [ "${remote}" == "False" ] || [ -n "${jobid}" ]; then
            rm -f "${utility_param}"
        fi
    done
    # Cleanup
    rm -rf "${utility_param_dirname}" || :
    # Deactivate traps before watching jobs and/or exiting
    trap : 0
    # Watch remotely submitted jobs in submission order
    any_jobs="False"
    for jobid in "${jobids[@]}"; do
        if [ -n "${jobid}" ]; then
            any_jobs="True"
            break
        fi
    done
    plural=""
    if [ ${n_snapshots} -gt 1 ]; then
        plural="s"
    fi
    if [ "${remote}" == "True" ] && [ "${CONCEPT_watch}" == "True" ] && [ "${any_jobs}" == "True" ]; then
        printf "\nYou can now kill (Ctrl+C) this script without cancelling the job${plural}\n"
        printf "\nWill now watch the submitted job${plural}\n"
        for jobid in "${jobids[@]}"; do
            if [ -n "${jobid}" ]; then
                "${concept}" -u watch "${jobid}"
            fi
        done
    fi
    echo
    # Final message
    if [ "${remote}" == "False" ] || [ "${CONCEPT_watch}" == "True" ]; then
        completely_successful="True"
        for jobid in "${jobids[@]}"; do
            if  [ -z "${jobid}" ] || [ -f "${job_dir}/${jobid}/log_err" ]; then
                completely_successful="False"
            fi
        done
        if [ "${completely_successful}" == "True" ]; then
            colorprint "${CONCEPT_utility} utility finished successfully" "green"
        else
            echo "${CONCEPT_utility} utility finished"
        fi
    else
        echo "${CONCEPT_utility} utility finished. Check the submitted job${plural} for results."
    fi
}

# If a utility is to be run, run it and exit
if [ "${utility}" != "${utility_default}" ]; then
    # If no argument was passed after the -u option,
    # utility_args should be empty.
    if     [ "${utility_args}" == ""   ] \
        || [ "${utility_args}" == '""' ] \
        || [ "${utility_args}" == "''" ] \
    ; then
        utility_args=""
    fi
    # Run utility and exit
    if [ "${utility_heading_printed}" != "True" ]; then
        colorprint "\nRunning the $(basename "${utility}") utility" "yellow"
        export utility_heading_printed="True"
    fi
    trap : 0
    if [ -z "${utility_args[0]}" ] && [ ${#utility_args[@]} == 1 ]; then
        called_from_concept="True" "${utility}"
    else
        called_from_concept="True" "${utility}" "${utility_args[@]}"
    fi
    # Print out the total execution time for this utility
    "${mpiexec}" -n 1 "${python}" -c "from commons import *
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
    "${python}" -c "
lines = [
    '',
    '# The above is a copy of the CO𝘕CEPT parameter file \"${param}\".',
    '# Below we apply any additional command-line parameters.',
]
for statement in '''${command_line_params}'''.split(';'):
    lines.append(statement.strip())
lines.append('')
with open('${job_dir}/${jobid}/param', mode='a', encoding='utf-8') as f:
    f.write('\n'.join(lines))
"
}

# Either stop doing further actions, submit job or run it locally
if [ "${remote}" == "True" ]; then
    # Running remotely.
    # Detect what resource manager is used.
    resource_manager="$(get_resource_manager)"
    resource_manager_nice=""
    if [ "${resource_manager}" == "slurm" ]; then
        resource_manager_nice="Slurm"
    elif [ "${resource_manager}" == "torque" ]; then
        resource_manager_nice="TORQUE/PBS"
        # Try to detect whether we are using
        # - pbs_pro=False: TORQUE or the original OpenPBS.
        # - pbs_pro=True: PBS Pro(fessional) or OpenPBS,
        #   from Altair Engineering.
        pbs_pro="False"
        for pbs_cmd in qsub qstat qdel; do
            pbs_version="$(${pbs_cmd} --version 2>&1 | head -n 1 | awk '{print $NF}' || :)"
            pbs_version="${pbs_version%%.*}"
            if [[ "${pbs_version}" =~ ^[0-9]+$ ]]; then
                break
            fi
        done
        if [ "${pbs_version}" -ge 14 2>/dev/null ]; then
            pbs_pro="True"
        fi
    fi
    # Prepare job script header dependent on the resource manager.
    # If no resource manager is used, default to slurm.
    if [ -z "${resource_manager}" ]; then
        resource_manager="slurm"
    fi
    if [ "${resource_manager}" == "slurm" ]; then
        # Split the 'nprocs' variable up into the number of nodes
        # and the number of processes per node, if both are given.
        if [[ "${nprocs}" == *':'* ]]; then
            nnodes="${nprocs%:*}"
            nprocs_per_node="${nprocs#*:}"
            ((nprocs = nnodes*nprocs_per_node))
        else
            nnodes=0  # has to be 0, not 1
            nprocs_per_node=${nprocs}
        fi
        # Compute dedicated memory per process and node in megabytes
        ((mb_per_process = memory/(2**20*nprocs))) || :
        if [ ${nnodes} -gt 0 ]; then
            ((mb_per_node = memory/(2**20*nnodes))) || :
        fi
        # Construct Slurm header
        jobscript_header="$(${python} -c "
directive_prefix = '#SBATCH'
lines = []
if '${job_name}':
    lines.append(f'{directive_prefix} --job-name=${job_name}')
if '${queue}':
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
lines.append('jobid=\"\${SLURM_JOB_ID%%%%.*}\"')
print('\n'.join(lines))
        ")"
        # The Slurm queue (partition) in which the job is running
        queue_display='${SLURM_JOB_PARTITION}'
    elif [ "${resource_manager}" == "torque" ]; then
        # Split the 'nprocs' variable up into the number of nodes
        # and the number of processes per node.
        if [[ "${nprocs}" != *':'* ]]; then
            colorprint "Warning: You are using TORQUE/PBS but did not specify \
the number of nodes. A single node will be used." "red"
            nprocs="1:${nprocs}"
        fi
        nnodes="${nprocs%:*}"
        nprocs_per_node="${nprocs#*:}"
        ((nprocs = nnodes*nprocs_per_node))
        # Compute dedicated memory per process and node in megabytes
        ((mb_per_process = memory/(2**20*nprocs))) || :
        ((mb_per_node    = memory/(2**20*nnodes))) || :
        # Construct TORQUE header
        jobscript_header="$(${python} -c "
directive_prefix = '#PBS'
lines = []
if '${job_name}':
    lines.append(f'{directive_prefix} -N ${job_name}')
if '${queue}':
    lines.append(f'{directive_prefix} -q ${queue}')
if '${pbs_pro}' == 'True':
    lines.append(
        f'{directive_prefix} -l select=${nnodes}:ncpus=${nprocs_per_node}'
        + ':mem=${mb_per_node}mb'*(${memory} > 0)
    )
    if ${memory} == 0:
        lines.append(f'{directive_prefix} -l mem=0')
    if ${nnodes} > 1:
        lines.append(f'{directive_prefix} -l place=scatter')
else:
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
lines.append('jobid=\"\${PBS_JOBID%%%%.*}\"')
print('\n'.join(lines))
        ")"
        # The TORQUE/PBS queue in which the job is running
        queue_display='${PBS_QUEUE}'
    fi
    # Prepare display texts with the total memory consumption
    # and maximum allowed wall time.
    infos_display="$("${mpiexec}" -n 1 "${python}" -c "
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
        if [ "$(get_resource_manager)" == "slurm" ]; then
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
job_name=\"${job_name}\"
main=\"${main}\"
param=\"${param}\"
utility=\"${utility}\"
optimizations=\"${optimizations}\"
linktime_optimizations=\"${linktime_optimizations}\"
native_optimizations=\"${native_optimizations}\"
rebuild=\"${rebuild}\"
safe_build=\"${safe_build}\"
pure_python=\"${pure_python}\"
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
colorprint                                                                                                           \\
    \"\\\nRunning \${esc_concept} job \${jobid}\${pure_python_msg} remotely on \$(hostname -f) (\${queue_display})\" \\
    \"yellow\"                                                                                                       \\
    >> \"\${job_dir}/\${jobid}/log\"
print_info

# Change to the concept directory
cd \"\${concept_dir}\"

# Run the code. Both stdout and stderr are being logged
# to job_dir/jobid/log, while the stderr alone is also logged
# to job_dir/jobid/log_err.
\${mpi_executor}                                      \\
    \"\${python}\" \${m_flag} \"\${main_as_library}\" \\
        \"param='\${param}'\"                         \\
        \"jobid='\${jobid}'\"                         \\
    >> \"\${job_dir}/\${jobid}/log\"                  \\
    2>> >(tee -a \"\${job_dir}/\${jobid}/log_err\")   \\

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
    # Exit now if we are not to submit
    if [ "${submit_actual}" == "False" ]; then
        printf "A ${resource_manager_nice} job script has been written to\n\"${jobscript}\"\n"
        if [ -z "${submit}" ]; then
            printf "Run with --submit for automatic submission\n"
        fi
        trap : 0
        exit 0
    fi
    # Exit now if no resource manager was found
    if [ -z "$(get_resource_manager)" ]; then
        colorprint "Could not find any resource manager.\n\
Is Slurm/TORQUE/PBS installed and on the PATH?\n\
A ${resource_manager_nice} job script has been saved to\n\
\"${jobscript}\"" "red"
        exit 1
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
    jobid="${jobid%%.*}"
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
    # unless --watch=False is supplied.
    if [ "${watch}" == "True" ]; then
        # Invoke the watch utility on the submitted job
        printf "You can now kill (Ctrl+C) this script without cancelling the job\n"
        "${concept}" -u watch "${jobid}"
    fi
    exit 0
else
    # Run locally
    if [[ "${nprocs}" == *':'* ]]; then
        nnodes="${nprocs%:*}"
        nprocs_per_node="${nprocs#*:}"
        ((nprocs = nnodes*nprocs_per_node))
        if [ ${nnodes} -ne 1 ]; then
            colorprint "You may not specify a number of nodes ≠ 1 when running locally" "red"
            exit 1
        fi
    fi
    # Construct the next jobid that does not yet exist in the
    # job directory, using exponential followed by binary search.
    jobid=0
    while [ -d "${job_dir}/${jobid}" ]; do
        jobid=$((2*jobid + 1))
    done
    jobid_upper=${jobid}
    jobid_lower=$(((jobid - 1)/2)) || :
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
    (                                                                  \
        (                                                              \
            "${mpiexec}" -n "${nprocs}" ${mpiexec_args}                \
                "${python}" ${i_flag} ${m_flag} "${main_as_library}"   \
                    "param='${param}'"                                 \
                    "jobid='${jobid}'"                                 \
            | tee -a "${job_dir}/${jobid}/log";                        \
            echo "${PIPESTATUS[0]}" > "${job_dir}/${jobid}/.exit_code" \
        ) 3>&1 1>&2 2>&3                                               \
        | tee -a                                                       \
            "${job_dir}/${jobid}/log"                                  \
            "${job_dir}/${jobid}/log_err"                              \
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
    if [ ${exit_code} -ne 0 ]; then
        # Note that the real exit will be handled by the abort function
        exit ${exit_code}
    fi
    # Deactivate traps and exit
    trap : 0
    exit 0
fi
