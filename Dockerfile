# Basics
FROM debian:11.6-slim
SHELL ["/usr/bin/env", "bash", "-c"]
CMD ["bash"]

# Installation options
ARG concept_version=/source
ARG install_dir="/concept"
ARG slim=True
ARG mpi=mpich
ARG mpi_configure_options="--with-device=ch3:sock +="
ARG cleanup_concept
ARG make_jobs

# Build environment as root
RUN : \
    # Update APT cache
    && apt-get update \
    # Install and set up simple text editor
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nano-tiny \
    && ln -s "$(which nano-tiny)" /bin/nano \
    # Install and set up less
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends less \
    && echo "[ ! -t 0 ] || alias less='less -r -f'" >> ~/.bashrc \
    # Install and set up Bash auto-completion
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bash-completion \
    && echo "[ ! -t 0 ] || source /etc/bash_completion" >> ~/.bashrc \
    # Set up Bash history search with ↑↓
    && echo "[ ! -t 0 ] || bind '\"\e[A\": history-search-backward' 2>/dev/null" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || bind '\"\e[B\": history-search-forward' 2>/dev/null" >> ~/.bashrc \
    # Set up colour prompt
    && echo "[ ! -t 0 ] || PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\\$ '" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || alias grep='grep --color=auto'" >> ~/.bashrc \
    # Add concept user
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sudo \
    && echo "Defaults lecture = never" > "/etc/sudoers.d/privacy" \
    && groupadd -r concept \
    && useradd -m -g concept concept \
    && echo    "root:concept" | chpasswd \
    && echo "concept:concept" | chpasswd \
    && adduser concept sudo \
    && echo "concept ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && cp ~/.bashrc /home/concept/ \
    && chown -R "concept:concept" /home/concept \
    # Prepare the CO𝘕CEPT installation directory
    && mkdir -p "${install_dir}" \
    && chown -R "concept:concept" "${install_dir}" \
    # Clean APT cache
    && apt-get autoremove -y \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/{apt/lists,cache,log}/* \
    && rm -rf $(ls /var/lib/dpkg/info/* | grep -v "\.list") \
    # Allow for APT auto-completion
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && :

# Build CO𝘕CEPT as concept user
USER concept
COPY . /source/
ENV TERM="linux"
RUN : \
    # Update APT cache
    && sudo apt-get update \
    # Install CO𝘕CEPT
    && bash /source/install -y "${install_dir}" \
    && sudo rm -rf /source \
    # Set up CO𝘕CEPT and Python environment
    && sed -i "1i source \"${install_dir}/concept\"" ~/.bashrc \
    && ln -s "${install_dir}/dep/python/bin/python3" "${install_dir}/dep/python/bin/python" \
    # Remove unnecessary packages and clean APT cache
    && sudo apt-get autoremove -y \
    && sudo apt-get purge -y 'g++*' 'libstdc++*dev' 'gfortran*' 'libgfortran*dev' \
    && sudo apt-get clean -y \
    && sudo apt-get autoclean -y \
    && sudo rm -rf /var/lib/{apt/lists,cache,log}/* \
    && sudo rm -rf $(sudo ls /var/lib/dpkg/info/* | grep -v "\.list") \
    # Remove other caches
    && sudo rm -rf /tmp/* ~/.cache/* \
    # Remove some system files
    && sudo rm -rf /usr/share/{doc,info,man}/* \
    && sudo rm -f $(sudo find / -name "*.a" 2>/dev/null | grep -v "/libgcc.a\|/libc_nonshared.a") \
    && sudo rm -f /usr/lib/x86_64-linux-gnu/lib{crypto,db-*,*san*}.so* \
    && sudo rm -f /usr/bin/x86_64-linux-gnu-lto-dump-* \
    && :
ENV PATH="${install_dir}:${install_dir}/util:${install_dir}/dep/python/bin:${PATH}"
WORKDIR "${install_dir}"

