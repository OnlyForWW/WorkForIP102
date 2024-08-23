#!/bin/bash
git clone https://gitcode.net/opencv/opencv.git
git clone https://gitcode.net/opencv/opencv_contrib.git

cd /home/$(whoami)/Downloads/opencv

mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D ENABLE_FAST_MATH=ON \
      -D BUILD_opencv_java=OFF \
      -D BUILD_ZLIB=ON \
      -D BUILD_TIFF=ON \
      -D WITH_GTK_2_X=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_1394=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv4.pc \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF ..

make -j $(nproc)

sudo make install 
