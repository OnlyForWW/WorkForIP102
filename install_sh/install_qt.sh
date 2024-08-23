#!/bin/bash

sudo wget https://mirrors.ustc.edu.cn/qtproject/archive/online_installers/4.6/qt-unified-linux-x64-4.6.0-online.run  -P /home/$(whoami)/Downloads

sudo chmod 777 /home/$(whoami)/Downloads/qt-unified-linux-x64-4.6.0-online.run

/home/$(whoami)/Downloads/qt-unified-linux-x64-4.6.0-online.run --mirror https://mirrors.ustc.edu.cn/qtproject/
