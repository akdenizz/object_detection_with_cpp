# OBJECT DETECTION WITH C++

##  OPENCV INSTALLATION with *CUDA* and *cuDNN*  SUPPORT


✨ A fresh start, so check for updates

```
$ sudo apt-get update
$ sudo apt-get upgrade
```

🗝 Third-party libraries

```
$ sudo apt-get install build-essential cmake git unzip pkg-config zlib1g-dev
$ sudo apt-get install libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev
$ sudo apt-get install libpng-dev libtiff-dev libglew-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
$ sudo apt-get install python-dev python-numpy python-pip
$ sudo apt-get install python3-dev python3-numpy python3-pip
$ sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev
$ sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev
$ sudo apt-get install gstreamer1.0-tools libgstreamer-plugins-base1.0-dev
$ sudo apt-get install libgstreamer-plugins-good1.0-dev
$ sudo apt-get install libv4l-dev v4l-utils v4l2ucp qv4l2
$ sudo apt-get install libtesseract-dev libxine2-dev libpostproc-dev
$ sudo apt-get install libavresample-dev libvorbis-dev
$ sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev
$ sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
$ sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev
$ sudo apt-get install liblapack-dev liblapacke-dev libeigen3-dev gfortran
$ sudo apt-get install libhdf5-dev libprotobuf-dev protobuf-compiler
$ sudo apt-get install libgoogle-glog-dev libgflags-dev
```

⬇️ Download the latest version

```
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
```

🎊 Unpack

```
$ unzip opencv.zip
$ unzip opencv_contrib.zip
```

📝 Some administration to make live easier later on

```
$ mv opencv-4.5.5 opencv
$ mv opencv_contrib-4.5.5 opencv_contrib
```

🌬 Clean up the zip files

```
$ rm opencv.zip
$ rm opencv_contrib.zip

$ cd ~/opencv
$ mkdir build && cd build
```

🦊 Build make

```
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_C_COMPILER=/usr/bin/gcc-8 \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUDA=ON \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=OFF \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D BUILD_EXAMPLES=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=6.1 .. 
```

> ***!!! CUDA_ARCH_BIN is compute capability of your GPU - You can check your gpu compute capability [here](https://developer.nvidia.com/cuda-gpus)***

💽 Use nproc to know the number of cpu cores

```
$ nproc 
$ make -j8
$ sudo make install
```
🗃 Include the libs in your environment

```
$ sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
$ sudo ldconfig
```
# 🛸 USAGE of OBJECT DETECTION

> 🌠 Clone the repo with `git clone https://github.com/akdenizz/object_detection_with_cpp` !

### 🔮 For yolov4 detection

```
$ cd yolov4
$ g++ yolov4_detect.cpp -o yolov4_out -std=c++11 `pkg-config --cflags --libs opencv4`
$ ./yolov4_out
```
### 🚀 For yolov5 detection

```
$ cd yolov5
$ g++ yolov5_detect.cpp -o yolov5_out -std=c++11 `pkg-config --cflags --libs opencv4`
$ ./yolov5_out
```
# 🌚HAVE FUN!🌝
