#!/bin/bash

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install libxcb-xinerama0 libgl1-mesa-dev

sudo apt-get install build-essential checkinstall  cmake gdb pkg-config yasm git gfortran libjpeg8-dev libpng-dev

sudo apt-get install qtbase5-dev

sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
sudo apt-get update

sudo apt-get install libjasper1 libjasper-dev ffmpeg libavcodec-dev libavformat-dev libswscale-dev libdc1394-dev

sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

sudo apt-get install libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils
