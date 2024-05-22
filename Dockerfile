# FROM nvidia/cudagl:10.1-base-ubuntu18.04
FROM cupy/cupy:v12.0.0


ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key list

RUN apt-get update && \
    apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y \
  python3-dev \
  python3-pip \
  python3-virtualenv \
  libeigen3-dev \
  libopencv-dev \
  libblas-dev \
  ffmpeg \
  cmake \
  tmux \
  vim \
  nano

RUN python3 -m pip install -U virtualenv jupyter jupyterlab

RUN rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt

RUN python3 -m pip install -U pip

RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install gpytorch cvxopt

ENTRYPOINT jupyter notebook --generate-config && \
    echo 'c.NotebookApp.ip="127.0.0.1"' >> /root/.jupyter/jupyter_notebook_config.py && \
    echo 'c.NotebookApp.allow_root = True' >> /root/.jupyter/jupyter_notebook_config.py && \
    cd /root/mfrl && \
    python3 -m pip install libs/pyMulticopterSim-1.0.1-cp310-cp310-linux_x86_64.whl && \
    /bin/bash
