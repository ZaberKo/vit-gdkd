ARG IMAGE_TAG=2.6.0-cuda12.4-cudnn9-devel

FROM pytorch/pytorch:${IMAGE_TAG}

ENV TZ=Asia/Shanghai

RUN set -eux; \
    apt update; \
    DEBIAN_FRONTEND=noninteractive apt install -y ca-certificates; \
    rm -rf /var/lib/apt/lists/*

RUN set -ex && \
    sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

RUN set -ex; \
    apt update;  \
    DEBIAN_FRONTEND=noninteractive apt install -y \
        neovim curl git tmux htop zsh iproute2 openssh-server iputils-ping dnsutils \
        ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
        libosmesa6-dev libgl1-mesa-dev libgl1-mesa-glx libglfw3 patchelf; \
    rm -rf /var/lib/apt/lists/*

# ssh server (also for horovod ssh connection)
RUN sed -i '/PermitRootLogin prohibit-password/c\PermitRootLogin yes' /etc/ssh/sshd_config
EXPOSE 22

WORKDIR /root

COPY requirements.txt /root
RUN pip install -r /root/requirements.txt

# fix ssh login envs missing issue
RUN env | egrep -v "^(HOME=|USER=|MAIL=|LS_COLORS=|LANG=|HOSTNAME=|PWD=|TERM=|SHLVL=|LANGUAGE=|_=|SHELL=)" >> /etc/environment

CMD ["/bin/bash"]

