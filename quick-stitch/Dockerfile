FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies (from OpenSfM's Dockerfile)
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

# Install Miniconda
RUN curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh



# Set up a Conda environment
ENV PATH="/opt/conda/bin:$PATH"
RUN conda create -n myenv python=3.12 
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"] 

# Manually run conda clean with the correct flag (update the path if needed)
#RUN conda run -n myenv conda clean -tipsy 

# Install OpenSfM and Ceres with Conda 
RUN conda run -n myenv conda install -c conda-forge opensfm 
RUN conda run -n myenv conda install -c conda-forge 'ceres=2.1.0' 'ceres-solver-suitesparse' lapack 'suitesparse=5.10'


# Clone your GitHub repository
WORKDIR /app
RUN git clone https://github.com/walter-morawa/structure-from-machine-learning.git . 

# Install other dependencies using pip within the Conda environment
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your application (adjust if your main script is named differently)
CMD ["python", "main.py"] 
