#CPP 
#opencv  
#编译

[toc]

#   编译主模块
```bash
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip

# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip 
unzip opencv.zip

# Create build directory
mkdir -p build && cd build

# Configure
cmake ../opencv-master

# Build
cmake --build .
```

#  参考
- https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html