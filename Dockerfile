# Based on https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
# Please use devel instead of runtime since you might need use the compiler (nvcc).
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt remove --purge cmake

RUN apt-get update && apt-get install -y --no-install-recommends \
     build-essential \
     libssl-dev \
     git \
     curl \
     vim \
     wget \
     unzip \
     ca-certificates \
     libjpeg-dev \
     libpng-dev \
     libgtk2.0-dev \
     libopencv-dev \
     libgl1-mesa-glx \
     bash-completion \
     libpcl-dev 
     

RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz && \
     tar -zxvf cmake-3.16.5.tar.gz && \
     cd cmake-3.16.5 && \
     ./bootstrap && \
     make  && \
     make install

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
RUN chmod -R a+w /workspace