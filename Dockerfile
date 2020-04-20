# Base
FROM ubuntu:18.04
SHELL ["/usr/bin/env", "bash", "-c"]

# Installation options
ARG top_dir=/concept
ARG mpi=mpich
ARG native_optimizations=False

# Update apt cache
ARG DEBIAN_FRONTEND=noninteractive
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
        2> >(grep -v 'apt-utils is not installed' >&2)

# Install CO𝘕CEPT
COPY installer /
RUN : \
    && mpi=${mpi} bash /installer \
        -y \
        $([ "${native_optimizations}" == "True" ] && echo --native-optimizations) \
        "${top_dir}" \
    && rm -f /installer

# Set up bash autocompletion and intelligent search with ↑↓
RUN : \
    && apt-get install -y bash-completion \
    && echo '. /etc/bash_completion' >> /etc/bash.bashrc \
    && echo "bind '\"\e[A\": history-search-backward'" >> ~/.bashrc \
    && echo "bind '\"\e[B\": history-search-forward'" >> ~/.bashrc

# Cleanup
RUN rm -rf /var/lib/apt/lists/*

# Environment
ENV PATH="${top_dir}/concept:${PATH}"
WORKDIR "${top_dir}/concept"

