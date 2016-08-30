CLCV_ROOT = ../clcv/resources
CLCV_CORPORA_ROOT = $(CLCV_ROOT)/corpora
CLCV_DATA_ROOT = $(CLCV_ROOT)/data
CLCV_TOOLS_ROOT = $(CLCV_ROOT)/tools
LOG_ROOT = ./log

MSCOCO_SYM = Microsoft_COCO_20160518
MSCOCO_ROOT = $(CLCV_CORPORA_ROOT)/$(MSCOCO_SYM)
MSCOCO_DATA_ROOT = $(CLCV_DATA_ROOT)/$(MSCOCO_SYM)
MODEL_ROOT = $(CLCV_DATA_ROOT)/cv_models

MSCOCO_SET = train val

MODEL_SET = myconceptsv3 mydepsv4 mypasv4 mypasprepv4

### DEFAULT PARAMETERS

VER?=v1
GID?=0
WD?=0
LR?=0.00001
BIAS?=-6.58
BS?=4
OP?=adam
EP?=1

###### PRE-PROCESSING

prepo_vgg: $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_preprocessedimages_vgg.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	mkdir -p $(LOG_ROOT)/prepo
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(MSCOCO_DATA_ROOT)/mscoco2014_$*_preprocessedimages_vgg.json \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 224 \
		2>&1 | tee $(LOG_ROOT)/prepo/mscoco2014_$*_preprocessedimages_vgg.txt

prepo_msmil: $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_preprocessedimages_msmil.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	mkdir -p $(LOG_ROOT)/prepo
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(MSCOCO_DATA_ROOT)/mscoco2014_$*_preprocessedimages_msmil.json \
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
		-train_image_file_h5 mscoco2014_train_preprocessedimages_vgg.h5 \
		-train_index_json mscoco2014_train_preprocessedimages_vgg.json \
		-val_image_file_h5 mscoco2014_val_preprocessedimages_vgg.h5 \
		-val_index_json mscoco2014_val_preprocessedimages_vgg.json \
		-cnn_proto $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers_deploy.prototxt  \
		-cnn_model $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size $(BS) -optim $(OP) -test_interval 1000 -num_test_image 400 -print_log_interval 10 \
		-vocab_file mscoco2014_train_$*vocab.json -model_type vgg \
		-cp_path $(MODEL_ROOT)/vgg-$* -model_id vgg-$* -max_epochs $(EP) \
		-learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) \
		2>&1 | tee $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).log
        
vgg-extract-fc8: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc8.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc8.h5,$(MODEL_SET))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc8.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 \
            -model_type vgg -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc8.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
            
vgg-extract-fc7: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc7.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc7.h5,$(MODEL_SET))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_vgg-%fc7.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
		-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 \
            	-model_type vgg -print_log_interval 1000 -batch_size $(BS) -version $(VER) \
            	-test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 -num_target $${NDIM} \
            	-layer fc7 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc7.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -print_log_interval 1000 -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/vgg-$*/$(VER)/model_vgg-$*_epoch$(EP).t7 -num_target $${NDIM} \
            -layer fc7 -output_file $@
            
vgg-%-test: $(MSCOCO_DATA_ROOT)/mscoco2014_val_vgg-%fc8.h5
	CUDA_VISIBLE_DEVICES=$(GID) th test.lua -log_mode file -coco_data_root $(MSCOCO_DATA_ROOT) \
		-log_dir $(MODEL_ROOT)/vgg-$* \
		-val_image_file_h5 mscoco2014_val_preprocessedimages_vgg.h5 \
		-val_index_json mscoco2014_val_preprocessedimages_vgg.json \
		-val_label_file_h5 mscoco2014_val_$*.h5 -model_type vgg -test_mode file \
            	-test_cp $^ -version $(VER)
            
###### MSMIL MODEL

msmil-train-models: msmil-myconceptsv3-model msmil-mydepsv4-model msmil-mypasv4-model msmil-mypasprepv4-model
msmil-test-models: msmil-myconceptsv3-test msmil-mydepsv4-test msmil-mypasv4-test msmil-mypasprepv4-test

msmil-%-model: $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \ $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 \
$(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
$(MSCOCO_DATA_ROOT)/mscoco2014_train_%.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_%.h5
	mkdir -p $(MODEL_ROOT)/msmil-$*/$(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th train.lua -coco_data_root $(MSCOCO_DATA_ROOT) \
		-train_label_file_h5 mscoco2014_train_$*.h5 \
		-val_label_file_h5 mscoco2014_val_$*.h5 \
		-train_image_file_h5 mscoco2014_train_preprocessedimages_msmil.h5 \
		-train_index_json mscoco2014_train_preprocessedimages_msmil.json \
		-val_image_file_h5 mscoco2014_val_preprocessedimages_msmil.h5 \
		-val_index_json mscoco2014_val_preprocessedimages_msmil.json \
		-cnn_proto $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers_deploy.prototxt  \
		-cnn_model $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size $(BS) -optim $(OP) -test_interval 1000 -num_test_image 400 -print_log_interval 10 \
		-vocab_file mscoco2014_train_$*vocab.json -model_type milmaxnor -cp_path $(MODEL_ROOT)/msmil-$* -model_id msmil-$*\
		-learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) -max_epochs $(EP) \
		2>&1 | tee $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).log
        
msmil-extract-fc8: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc8.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc8.h5,$(MODEL_SET))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc8.h5: 
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc8.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc8 -output_file $@
            
msmil-extract-fc7: $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc7.h5, $(MODEL_SET)) \
    $(patsubst %,$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc7.h5,$(MODEL_SET))
$(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-%fc7.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc7 -output_file $@
$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc7.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_$*vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-$*/$(VER)/model_msmil-$*_epoch$(EP).t7 \
            -layer fc7 -output_file $@
msmil-extract-myconceptsv3responsemapfc8: $(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-myconceptsv3responsemapfc8.h5 \
	$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-myconceptsv3responsemapfc8.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_msmil-myconceptsv3responsemapfc8.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_myconceptsv3vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_$*_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-myconceptsv3/$(VER)/model_msmil-myconceptsv3_epoch$(EP).t7 \
            -layer responsemapfc8 -output_file $@

msmil-extract-myconceptsv3responsemapfc7: $(MSCOCO_DATA_ROOT)/mscoco2014_train_msmil-myconceptsv3responsemapfc7.h5 \
	$(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-myconceptsv3responsemapfc7.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_msmil-myconceptsv3responsemapfc7.h5:
	NDIM=$$(python -c "import json; v=json.load(open('$(MSCOCO_DATA_ROOT)/mscoco2014_train_myconceptsv3vocab.json')); print len(v)") && \
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_$*_preprocessedimages_msmil.h5 \
            -model_type milmaxnor -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
            -test_cp $(MODEL_ROOT)/msmil-myconceptsv3/$(VER)/model_msmil-myconceptsv3_epoch$(EP).t7 \
            -layer responsemapfc7 -output_file $@

msmil-%-test: $(MSCOCO_DATA_ROOT)/mscoco2014_val_msmil-%fc8.h5
	CUDA_VISIBLE_DEVICES=$(GID) th test.lua -log_mode file -log_dir $(MODEL_ROOT)/msmil-$* \
		-val_image_file_h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5 \
		-val_label_file_h5 mscoco2014_val_$*.h5 -model_type milmaxnor -test_mode file \
            	-test_cp $^ -version $(VER)
            

###### MULTITASK MODEL
### myconceptsv3 + mydepsv4
vgg-multitask-train:
	mkdir -p $(MODEL_ROOT)/vgg-multitask/$(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th train_multitask.lua \
		-coco_data_root $(MSCOCO_DATA_ROOT) \
		-cnn_proto $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers_deploy.prototxt \
		-cnn_model $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 mscoco2014_dev1_image_depnet_preprocessedimages_vgg.h5 \
		-train_index_json mscoco2014_dev1_image_depnet_preprocessedimages_vgg.json \
		-val_image_file_h5 mscoco2014_dev2_image_depnet_preprocessedimages_vgg.h5 \
		-val_index_json mscoco2014_dev2_image_depnet_preprocessedimages_vgg.json \
		-train_label_file_h5_task1 mscoco2014_dev1_captions_myconceptsv3.h5 \
		-val_label_file_h5_task1 mscoco2014_dev2_captions_myconceptsv3.h5 \
		-train_label_file_h5_task2 mscoco2014_dev1_captions_mydepsv4.h5 \
		-val_label_file_h5_task2 mscoco2014_dev2_captions_mydepsv4.h5 \
		-test_interval 1000 -num_test_image 400 -max_epochs $(EP) \
		-print_log_interval 10 -model_type vgg -multitask_type 1 \
		-batch_size $(BS) -optim $(OP) -bias_init $(BIAS) -weight_decay $(WD) -version $(VER) -learning_rate $(LR) \
		2>&1 | tee $(MODEL_ROOT)/vgg-multitask/$(VER)/model_vgg-multitask_epoch$(EP).log

### myconceptsv3 + mydepsv4
msmil-multitask-train:
	mkdir -p $(MODEL_ROOT)/msmil-multitask/$(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th train_multitask.lua \
		-coco_data_root $(MSCOCO_DATA_ROOT) \
		-cnn_proto $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers_deploy.prototxt \
		-cnn_model $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 mscoco2014_dev1_image_depnet_preprocessedimages_msmil.h5 \
		-train_index_json mscoco2014_dev1_image_depnet_preprocessedimages_msmil.json \
		-val_image_file_h5 mscoco2014_dev2_image_depnet_preprocessedimages_msmil.h5 \
		-val_index_json mscoco2014_dev2_image_depnet_preprocessedimages_msmil.json \
		-train_label_file_h5_task1 mscoco2014_dev1_captions_myconceptsv3.h5 \
		-val_label_file_h5_task1 mscoco2014_dev2_captions_myconceptsv3.h5 \
		-train_label_file_h5_task2 mscoco2014_dev1_captions_mydepsv4.h5 \
		-val_label_file_h5_task2 mscoco2014_dev2_captions_mydepsv4.h5 \
		-test_interval 1000 -num_test_image 400 -max_epochs $(EP) \
		-print_log_interval 10 -model_type milmaxnor -multitask_type 1 \
		-batch_size $(BS) -optim $(OP) -bias_init $(BIAS) -weight_decay $(WD) -version $(VER) -learning_rate $(LR) \
		2>&1 | tee $(MODEL_ROOT)/msmil-multitask/$(VER)/model_msmil-multitask_epoch$(EP).log
        
vgg-multitask-test:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -num_target 22034 -print_log_interval 1000 -batch_size 32 \
            -test_cp cp/$(VER)/model_multitask_mt1_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP).t7 -version $(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th test_multitask.lua -log_mode file \
			-val_image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
			-model_type vgg -test_mode file -version $(VER) \
            -test_cp data/Microsoft_COCO/$(VER)/model_multitask_mt1_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP)_fc8.h5
            
