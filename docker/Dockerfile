FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="hoang1007"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY ./ /app
RUN python3 -m pip install -v -e .
