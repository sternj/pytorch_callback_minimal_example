FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

#install cmake 3.7
ADD https://cmake.org/files/v3.7/cmake-3.31.7-Linux-x86_64.sh /cmake-3.7.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.7.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# install clang 18
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 all && \
    rm llvm.sh

# Verify installation
RUN clang-18 --version && clang++-18 --version

# Optionally set clang-18 as default
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100


RUN