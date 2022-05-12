# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
LABEL com.amazon.image.authors.email="sage-learner@amazon.com"
LABEL com.amazon.image.authors.name="Amazon AI"

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-key del 7fa2af80 \
    && rm /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list \
    && apt-get -y update && apt-get install -y --no-install-recommends \
        wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get -y update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        nginx \
        ca-certificates \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    && rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip --version \
    && pip --no-cache-dir install \
        transformers==4.15 \
        deepspeed==0.5.10 \
        flask==2.1.2 \
        gunicorn==20.1.0

# Set up the program in the image
COPY ./src /opt/program
WORKDIR /opt/program
