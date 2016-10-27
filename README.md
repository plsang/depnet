# Installation

## Update CuDNN V5.1
  - Get CuDNN v5.1 from /works/csisv15/plsang/cudnn/cudnn-7.5-linux-x64-v5.1.tgz
    
  ```bash
  tar -xvf cudnn-7.5-linux-x64-v5.1.tgz
  sudo cp cuda/include/* /usr/local/cuda/include/
  sudo cp cuda/lib64/* /usr/local/cuda/lib64/
  cd /usr/local/cuda/lib64/
  sudo unlink libcudnn.so
  sudo ln -s libcudnn.so.5 libcudnn.so
  ```
  
## Install Torch7
  - Install torch from scratch
  
  ```bash
  git clone https://github.com/plsang/distro.git ~/torch --recursive
  cd ~/torch; bash install-deps;
  ./install.sh
  source ~/.bashrc
  ```
  - Update Torch
  
  ```bash
  cd ~/torch
  ./update.sh
  ./install.sh
  ```
  
## Install packages
This can be done using [this script](https://github.com/mynlp/depnet/blob/clcv/install_deps.sh). Otherwise, you can manually install them as follows.

### Install standard packages

  ```bash
  luarocks install nn
  luarocks install nngraph
  luarocks install image
  luarocks install cutorch
  luarocks install cunn
  luarocks install loadcaffe
  luarocks install lualogging
  ```
### Install torch-hdf5
  ```
  git clone https://github.com/deepmind/torch-hdf5
  cd torch-hdf5; luarocks make hdf5-0-0.rockspec
  ```
### Install cjson
  ```
  wget http://www.kyne.com.au/~mark/software/download/lua-cjson-2.1.0.tar.gz
  tar -xvf lua-cjson-2.1.0.tar.gz
  cd lua-cjson-2.1.0; luarocks make
  ```
### Install debugger 
  ```
  git clone https://github.com/slembcke/debugger.lua.git torch-debugger
  cd torch-debugger; luarocks make
  ```
### Install customized torch-nn (required by Depnet)
  ```
  git clone https://github.com/plsang/nn.git torch-nn
  cd torch-nn; luarocks make rocks/nn-scm-1.rockspec
  ```
### Install customized torch-cunn (required by Depnet)
  ```
  git clone https://github.com/plsang/cunn.git torch-cunn
  cd torch-cunn; luarocks make rocks/cunn-scm-1.rockspec
  ```
  
More on [clcv branch](https://github.com/mynlp/depnet/tree/clcv)
