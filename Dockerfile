# Base
FROM ubuntu:20.04
SHELL ["/usr/bin/env", "bash", "-c"]

# Installation options
ARG concept_version=/source
ARG top_dir="/concept"
ARG mpi=mpich
ARG mpi_configure_options="+= --with-device=ch3:sock"
ARG native_optimizations=False

# Update apt cache
ARG DEBIAN_FRONTEND=noninteractive
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
        2> >(grep -v 'apt-utils is not installed' >&2)

# Install CO𝘕CEPT
COPY installer COPYING* Dockerfile* README.md* .env .github* .gitignore* /source/
COPY installer concept* /source/concept/
RUN : \
    && rm -f /source/concept/installer \
    && concept_version=${concept_version} bash /source/installer -y "${top_dir}"

# Set up:
#  - CO𝘕CEPT and Python environment
#  - Bash autocompletion
#  - Bash history search with ↑↓
#  - color prompt
RUN : \
    && sed -i "1i source \"${top_dir}/concept/concept\"" ~/.bashrc \
    && ln -s "${top_dir}/python/bin/python3" "${top_dir}/python/bin/python" \
    && apt-get install -y bash-completion \
    && echo "source /etc/bash_completion" >> ~/.bashrc \
    && echo "bind '\"\e[A\": history-search-backward'" >> ~/.bashrc \
    && echo "bind '\"\e[B\": history-search-forward'" >> ~/.bashrc \
    && sed -i "s/xterm-color)/xterm-color|*-256color)/" ~/.bashrc
ENV PATH="${PATH}:${top_dir}/concept:${top_dir}/python/bin"
ENV TERM="xterm-256color"
WORKDIR "${top_dir}/concept"

# Cleanup
RUN : \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /source \
    && rm -f ${top_dir}/install_log*

