FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies from OpenSfM's Dockerfile
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
        libopencv-dev \
        libceres-dev \
        python3-dev \
        python3-numpy \
        python3-opencv \
        python3-pip \
        python3-pyproj \
        python3-scipy \
        python3-yaml \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Clone the OpenSfM repository
WORKDIR /app
RUN git clone --recursive https://github.com/mapillary/OpenSfM.git

# Clone SfML GitHub repository (to get the sample dataset)
WORKDIR /app
RUN git clone https://github.com/walter-morawa/structure-from-machine-learning.git sample-dataset 


# Build and install OpenSfM
WORKDIR /app/OpenSfM
RUN pip3 install -r requirements.txt && \
    python3 setup.py build && \
    python3 setup.py install 

# Install quick-stitch dependencies (if any additional ones are needed)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy your quick-stitch project files 
COPY . .

# Command to run your application 
CMD ["python", "main.py"] 
