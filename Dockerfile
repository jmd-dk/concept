# Base
FROM debian:11.1-slim
SHELL ["/usr/bin/env", "bash", "-c"]
CMD ["bash"]

# Installation options
ARG concept_version=/source
ARG install_dir="/concept"
ARG slim=True
ARG mpi=mpich
ARG mpi_configure_options="+= --with-device=ch3:sock"
ARG cleanup_concept
ARG make_jobs

# Build
COPY \
    .env* \
    .gitignore* \
    CHANGELOG.md* \
    Dockerfile* \
    LICENSE* \
    Makefile* \
    README.md* \
    concept* \
    install \
    /source/
COPY doc*            /source/doc/
COPY param/example_* /source/param/
COPY src*            /source/src/
COPY test*           /source/test/
COPY util*           /source/util/
COPY .github*        /source/.github/
ARG DEBIAN_FRONTEND=noninteractive
RUN : \
    # Update APT cache
    && apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
        2> >(grep -v 'apt-utils is not installed' >&2) \
    # Install CO𝘕CEPT
    && bash /source/install -y "${install_dir}" \
    && rm -rf /source \
    # Set up CO𝘕CEPT and Python environment
    && sed -i "1i source \"${install_dir}/concept\"" ~/.bashrc \
    && apt-get install -y --no-install-recommends less \
    && echo "[ ! -t 0 ] || alias less='less -r -f'" >> ~/.bashrc \
    && ln -s "${install_dir}/dep/python/bin/python3" "${install_dir}/dep/python/bin/python" \
    # Set up Bash auto-completion
    && apt-get install -y --no-install-recommends bash-completion \
    && echo "[ ! -t 0 ] || source /etc/bash_completion" >> ~/.bashrc \
    # Set up Bash history search with ↑↓
    && echo "[ ! -t 0 ] || bind '\"\e[A\": history-search-backward' 2>/dev/null" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || bind '\"\e[B\": history-search-forward' 2>/dev/null" >> ~/.bashrc \
    # Set up colour prompt
    && echo "[ ! -t 0 ] || PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\\$ '" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || alias grep='grep --color=auto'" >> ~/.bashrc \
    # Remove unnecessary packages and clean APT cache
    && apt-get autoremove -y \
    && apt-get purge -y 'g++*' 'libstdc++*dev' 'gfortran*' 'libgfortran*dev' \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/{apt/lists,cache,log}/* \
    && rm -rf $(ls /var/lib/dpkg/info/* | grep -v "\.list") \
    # Remove other caches
    && rm -rf /tmp/* ~/.cache/* \
    # Remove some files installed with CO𝘕CEPT
    && rm -f $(find "${install_dir}" -name "*.a") \
    && rm -f "${install_dir}"/dep/freetype*/lib/libfreetype*.so* \
    && rm -f "${install_dir}"/dep/python/lib/python*/site-packages/Pillow.libs/lib{freetype,harfbuzz,lcms,png,web}*.so* \
    && rm -f "${install_dir}"/dep/python/lib/python*/site-packages/scipy*/scipy/misc/face.dat \
    # Remove some system files
    && rm -rf /usr/share/{doc,info,man}/* \
    && rm -f $(find / -name "*.a" | grep -v "/libgcc.a\|/libc_nonshared.a") \
    && rm -f /usr/lib/x86_64-linux-gnu/lib{crypto,db-*,*san*}.so* \
    && rm -f /usr/bin/x86_64-linux-gnu-lto-dump-* \
    && :
ENV \
    PATH="${install_dir}:${install_dir}/util:${install_dir}/dep/python/bin:${PATH}" \
    TERM="linux"
WORKDIR "${install_dir}"

