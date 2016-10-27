
# Preprocessing
  * Check out the clcv branch
  * Update CLCV_ROOT=/path/to/the/clcv/`resources`/directory in the Makefile
  * Preprocessing:
  
  ```
  make prepo_vgg     # preprocessing images for training vgg models
  make prepo_msmil   # preprocessing images for training msmil models
  make vgg16-model   # download the standard VGG16 net
  ```

# Training models
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

# Extracting features
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
   
# Testing the models
  
  After extracting features from the fc8 layer, we can test the performance of the model on the COCO val set.
 
  * VGG models
  ```
  make vgg-test-models
  ```
  * MSMIL models
  ```
  make msmil-test-models
  ```

# Extract feature interactively

- Start the server 
  
  ```
  th extract_features_server.lua -model_path <path_to_a_depnet_model>
  e.g., th extract_features_server.lua -model_path /clcv/resources/data/cv_models/depnet-vgg-myconceptsv3/v1/model_depnet-dev_epoch1.t7
  ```
- Input format
  - `filename`: path to the image file
  - `layers`: in [`fc6`,`fc7`,`fc8`,`responsemapfc8`]
  - `top_concepts`: number of top concepts (default: 20), used for extracting `responsemapfc8`. 
  ```
  {"filename": "../clcv/resources/corpora/Microsoft_COCO/images/val2014/COCO_val2014_000000029594.jpg", "layers":["fc6","fc7","fc8"]}
  {"filename": "../clcv/resources/corpora/Microsoft_COCO/images/val2014/COCO_val2014_000000029594.jpg", "layers":["responsemapfc8"],"top_concepts":20}
  ```
- Output format
  - For `fc6`,`fc7`,`fc8` layers
  ```
  {"fc6":[0,1.076,0.20], "fc7":[0,1.076,0.20],"fc8":[0,1.076,0.20]}
  ```
  - For `responsemapfc8` layer
  ```
  {"responsemapfc8":{"scores":[[],[],[]],"index":[]}}
  ```
