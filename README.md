1. Installation
  * Install Torch7
  ```
  git clone https://github.com/torch/distro.git ~/torch --recursive
  cd ~/torch; bash install-deps;
  ./install.sh
  ```
  * Install standard packages
  ```
  luarocks install nn
  luarocks install nngraph
  luarocks install image
  luarocks install cutorch
  luarocks install cunn
  luarocks install loadcaffe
  luarocks install lualogging
  ```
  * Install torch-hdf5
  ```
  git clone https://github.com/deepmind/torch-hdf5
  cd torch-hdf5; luarocks make hdf5-0-0.rockspec
  ```
  * Install cjson
  ```
  wget http://www.kyne.com.au/~mark/software/download/lua-cjson-2.1.0.tar.gz
  tar -xvf lua-cjson-2.1.0.tar.gz
  cd lua-cjson-2.1.0; luarocks make
  ```
  * Install debugger 
  ```
  git clone https://github.com/slembcke/debugger.lua.git torch-debugger
  cd torch-debugger; luarocks make
  ```
  * Install customized torch-nn
  ```
  git clone https://github.com/plsang/nn.git torch-nn
  cd torch-nn; luarocks make rocks/nn-scm-1.rockspec
  ```
  * Install customized torch-cunn
  ```
  git clone https://github.com/plsang/cunn.git torch-cunn
  cd torch-cunn; luarocks make rocks/cunn-scm-1.rockspec
  ```
  * The required packages can be installed using the `install_deps.sh` script.
  
2. Preprocessing
  * Check out the clcv branch
  * Update CLCV_ROOT=/path/to/the/clcv/`resources`/directory in the Makefile
  * Preprocessing:
  ```
  make prepo_vgg     # preprocessing images for training vgg models
  make prepo_msmil   # preprocessing images for training msmil models
  make vgg16-model   # download the standard VGG16 net
  ```
3. Training models
  * Training Options
    * GID: specify the GPU device ID (default: 0)
    * VER: version name (e.g., v1), each models will be saved in a subdirectory specified by this version
    * WD: weight decay, recommend to set it to zero when using ADAM learning method
    * LR: learning rate
    * BS: batch size (number of images per batch) reduce this number if memory is not enough
    * OP: Optimization method (choices=sgd,adam) 
    * EP: number of training epochs (default = 1)
  * VGG models
  ```
  make vgg-train-models   
  ```
  This command will train all the model names (myconceptsv3, mydepsv4, mypasv4, mypasprepv4) sequentially. 
  We can also train each model separately:
  
  ```
  make vgg-myconceptsv3-model 
  make vgg-mydepsv4-model 
  make vgg-mypasv4-model 
  make vgg-mypasprepv4-model
  ```
  * MSMIL models
  ```
  make msmil-train-models OP=adam WD=0  # sgd does not work well for MSMIL models
  ```
  * Multitask models
4. Extracting features
  * VGG models
  ```
  make vgg-extract-fc8
  make vgg-extract-fc7
  ```
  * MSMIL models
  ```
  make msmil-extract-fc8
  ```
  * Multitask models
5. Testing the models
  
  After extracting features from the fc8 layer, we can test the performance of the model on the COCO val set.
 
  * VGG models
  ```
  make vgg-test-models
  ```
  * MSMIL models
  ```
  make msmil-test-models
  ```
