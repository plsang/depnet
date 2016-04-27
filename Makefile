CLCV_ROOT = ../clcv/resources
CLCV_CORPORA_ROOT = $(CLCV_ROOT)/corpora
CLCV_DATA_ROOT = $(CLCV_ROOT)/data
CLCV_TOOLS_ROOT = $(CLCV_ROOT)/tools
LOG_ROOT = ./log

MSCOCO_SYM = Microsoft_COCO
MSCOCO_ROOT = $(CLCV_CORPORA_ROOT)/$(MSCOCO_SYM)
MSCOCO_DATA_ROOT = $(CLCV_DATA_ROOT)/$(MSCOCO_SYM)
MODEL_ROOT = $(CLCV_DATA_ROOT)/cv_models

MSCOCO_SET = train val

MODEL_SET = myconceptsv3 mydepsv4 mypasv4 mypasprepv4

### DEFAULT PARAMETERS

NDIM?=1000
VER?=v2
GID?=0
WD?=0
LR?=1e-5
BIAS?=0
BS?=1
OP?=adam
EP?=1

### PRE-PROCESSING

prepo_vgg: $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_preprocessedimages_vgg.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	mkdir -p $(LOG_ROOT)/prepo
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 224 \
		2>&1 | tee $(LOG_ROOT)/prepo/mscoco2014_$*_preprocessedimages_vgg.txt

prepo_msmil: $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_preprocessedimages_msmil.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	mkdir -p $(LOG_ROOT)/prepo
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 565 \
		2>&1 | tee $(LOG_ROOT)/prepo/mscoco2014_$*_preprocessedimages_msmil.txt

vgg16-model: $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel
$(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel:
	mkdir -p $(MODEL_ROOT)/pretrained-models/vgg-imagenet
	wget -P $(MODEL_ROOT)/pretrained-models/vgg-imagenet \
          http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
          https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt \
          http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
	cd $(MODEL_ROOT)/pretrained-models/vgg-imagenet; tar -xzf caffe_ilsvrc12.tar.gz; rm caffe_ilsvrc12.tar.gz 
    

###### VGG MODELS

vgg-train-models: vgg-myconceptsv3-model vgg-mydepsv4-model vgg-mypasv4-model vgg-mypasprepv4-model
vgg-test-models: vgg-myconceptsv3-test vgg-mydepsv4-test vgg-mypasv4-test vgg-mypasprepv4-test

vgg-%-model: $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \    $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 \
$(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5 \
$(MSCOCO_DATA_ROOT)/mscoco2014_train_%.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_%.h5
	mkdir -p $(MODEL_ROOT)/vgg-$*/$(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th train.lua -coco_data_root $(MSCOCO_DATA_ROOT) \
		-train_label_file_h5 mscoco2014_train_$*.h5 \
		-val_label_file_h5 mscoco2014_val_$*.h5 \
		-train_image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 \
		-val_image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5 \
		-cnn_proto $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers_deploy.prototxt  \
		-cnn_model $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size $(BS) -optim $(OP) -num_target $(NDIM) -test_interval 1000 -num_test_image 400 -print_log_interval 10 \
		-vocab_file mscoco2014_train_$*vocab.json -model_type vgg \
		-cp_path $(MODEL_ROOT)/vgg-$* -model_id vgg-$* \
		-learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) -max_iters 100 -save_cp_interval 100 \
		2>&1 | tee $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).log
        
vgg-extract-fc8: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc8.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc8.h5,$(VGG_MODELS))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc8.h5:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 \
            -model_type vgg -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc8.h5:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
            
vgg-extract-fc7: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc7.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc7.h5,$(MODEL_SET))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc7.h5:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 \
            -model_type vgg -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 \
            -layer fc7 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc8.h5:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 \
            -layer fc7 -output_file $@
            
vgg-%-test:            
	CUDA_VISIBLE_DEVICES=$(GID) th test.lua -log_mode file \
			-val_image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5 \
			-val_label_file_h5 mscoco2014_val_$*.h5 -model_type vgg -test_mode file \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 \
            -version $(VER)
            
### MSMIL MODEL

msmil-train-models: msmil-myconceptsv3-model msmil-mydepsv4-model msmil-mypasv4-model msmil-mypasprepv4-model
msmil-test-models: msmil-myconceptsv3-test msmil-mydepsv4-test msmil-mypasv4-test msmil-mypasprepv4-test

msmil-%-model: $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \ $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 \
$(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
$(MSCOCO_DATA_ROOT)/mscoco2014_train_%.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_%.h5
	mkdir -p $(MODEL_ROOT)/msmil-$*/$(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th train.lua -coco_data_root $(MSCOCO_DATA_ROOT) \
		-train_label_file_h5 mscoco2014_train_$*.h5 \
		-val_label_file_h5 mscoco2014_val_$*.h5 \
		-train_image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
		-cnn_proto $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers_deploy.prototxt  \
		-cnn_model $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size $(BS) -optim $(OP) -num_target $(NDIM) -test_interval 1000 -num_test_image 400 -print_log_interval 10 \
		-vocab_file mscoco2014_train_$*vocab.json -model_type milmaxnor -cp_path $(MODEL_ROOT)/vgg-$* -model_id msmil-$*\
		-learning_rate $(LR) -weight_decay $(WD) -bias_init -6.58 -version $(VER) -max_iters 100 -save_cp_interval 100 \
		2>&1 | tee $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).log
        
msmil-extract-fc8: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc8.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc8.h5,$(MODEL_SET))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc8.h5: 
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc8.h5:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
            
msmil-extract-fc7: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc7.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc7.h5,$(MODEL_SET))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc7.h5:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc7 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc8.h5:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $(NDIM) -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc7 -output_file $@

msmil-%-test:
	CUDA_VISIBLE_DEVICES=$(GID) th test.lua -log_mode file \
			-val_image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
			-val_label_file_h5 mscoco2014_val_$*.h5 -model_type milmaxnor -test_mode file \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -version $(VER)
