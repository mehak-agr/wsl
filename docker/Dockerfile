# Start with the base CUDA image
# FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
FROM projectmonai/monai

WORKDIR /

ENV TZ=America/Montreal
ARG DEBIAN_FRONTEND=noninteractive

# Install useful programs and build tools
RUN apt update && apt install -y \
       build-essential \
       cifs-utils \
       cmake \
       git \
       vim \
       swig \
       tmux \
       wget && \
       apt clean

# Install GDCM from source with python bindings
RUN   wget -O gdcm.tar.gz "https://sourceforge.net/projects/gdcm/files/gdcm 3.x/GDCM 3.0.7/gdcm-3.0.7.tar.gz" && \
      tar xzf gdcm.tar.gz && \
      rm gdcm.tar.gz && \
      mkdir gdcm-build && \
      mkdir /gdcm_pkg && \
      cd gdcm-build && \
      cmake -DGDCM_BUILD_APPLICATIONS=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DGDCM_WRAP_PYTHON=ON \
            -DGDCM_BUILD_SHARED_LIBS=ON \
            -DGDCM_INSTALL_PYTHONMODULE_DIR=/gdcm_pkg/ \
            -DPYTHON_EXECUTABLE=/opt/conda/bin/python \
            ../gdcm-3.0.7/ && \
      make -j 16 && \
      make install && \
      ldconfig && \
      rm -rf /gdcm-3.0.7 && \
      rm -rf /gdcm-build

# This is necessary so that the GDCM python bindings can be found
ENV PYTHONPATH /gdcm_pkg/

# Install Python packages from a requirements file
COPY requirements.txt /
RUN pip install --upgrade pip==20.3.1 && pip install --no-cache-dir -r requirements.txt
