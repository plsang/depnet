## Installation
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
  
## Preprocessing
  * Check out the clcv branch
  * Update CLCV_ROOT=/path/to/the/clcv/`resources`/directory in the Makefile
  * Preprocessing:
  ```
  make prepo_vgg     # preprocessing images for training vgg models
  make prepo_msmil   # preprocessing images for training msmil models
  make vgg16-model   # download the standard VGG16 net
  ```
## Training models
  * Training Options
    * GID: [0] specify the GPU device ID (default: 0)
    * VER: [v1] version name (e.g., v1), each models will be saved in a subdirectory specified by this version
    * WD: [0] weight decay, recommend to set it to zero when using ADAM learning method
    * LR: [1e-5] learning rate
    * BS: [4] batch size (number of images per batch) reduce this number if memory is not enough
    * OP: [adam] Optimization method (choices=sgd,adam).  Use `adam` for faster convergence 
    * EP: [1] number of training epochs (default = 1), set to 4 or 5 for better performance
    * BIAS: [-6.58] set to 0 for VGG models, and -6.58 for MSMIL models
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

## Extracting features
  * VGG models
  ```
  make vgg-extract-fc8
  make vgg-extract-fc7
  ```
  * MSMIL models
  ```
  make msmil-extract-fc8
  ```
  * Format of the feature maps output (at `fc8`)
    * The first K values are the indexes of the top K concepts (corresponding to its index in the vocabulary)
    * The remaining is the faltten array of the reponse maps K x 12 x 12 (flattend in row major order)
   
## Testing the models
  
  After extracting features from the fc8 layer, we can test the performance of the model on the COCO val set.
 
  * VGG models
  ```
  make vgg-test-models
  ```
  * MSMIL models
  ```
  make msmil-test-models
  ```
