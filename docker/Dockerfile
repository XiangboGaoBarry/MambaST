FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get -y install wget



COPY ./requirements.txt requirements.txt
COPY ./vim_requirements.txt vim_requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r vim_requirements.txt

RUN pip install PyYAML==5.3.1 protobuf==3.20.0 gpustat scikit-learn ptflops timm

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 nano git tmux htop vim -y 

ARG USERNAME=your_username
ARG USER_UID=your_useruid
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -l --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME

