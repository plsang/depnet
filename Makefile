CLCV_ROOT = /net/per920a/export/das14a/satoh-lab/plsang
CLCV_CORPORA_ROOT = $(CLCV_ROOT)/corpora
DATA_ROOT = ./data
LOG_ROOT = ./log

MSCOCO_SYM = Microsoft_COCO
MSCOCO_ROOT = $(CLCV_CORPORA_ROOT)/$(MSCOCO_SYM)
MSCOCO_DATA_ROOT = $(DATA_ROOT)/$(MSCOCO_SYM)

TYPE?=myconceptsv3
NDIM?=1000
VER?=v1.9
GID?=0
WD?=0
LR?=1e-5
BIAS?=0
BS?=1
OP?=adam
EP?=1
LAYER?=fc8

### PRE-PROCESSING

prepo_vgg: $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_preprocessedimages_vgg.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 224 \
		2>&1 | tee log/prepo/mscoco2014_$*_preprocessedimages_vgg.txt

prepo_msmil: $(MSCOCO_DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 $(MSCOCO_DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5
$(MSCOCO_DATA_ROOT)/mscoco2014_%_preprocessedimages_msmil.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 565 \
		2>&1 | tee log/prepo/mscoco2014_$*_preprocessedimages_msmil.txt


### CONCEPT MODEL

vgg-$(TYPE)-train: 
	CUDA_VISIBLE_DEVICES=$(GID) th train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		-train_label_file_h5 mscoco2014_train_$(TYPE).h5 \
		-val_label_file_h5 mscoco2014_val_$(TYPE).h5 \
		-train_image_file_h5 data/Microsoft_COCO/mscoco2014_train_preprocessedimages_vgg.h5 \
		-val_image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size $(BS) -optim $(OP) -num_target $(NDIM) -test_interval 1000 -num_test_image 400 -print_log_interval 10 \
		-vocab_file mscoco2014_train_$(TYPE)vocab.json -model_type vgg \
        	-learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) \
		2>&1 | tee log/$(VER)/train_vgg_$(TYPE)_$(OP)_b$(BS)_wd$(WD)_bias$(BIAS)_lr$(LR).log
        
vgg-$(TYPE)-test:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -num_target $(NDIM) -print_log_interval 1000 -batch_size 32 \
            -test_cp cp/$(VER)/model_$(TYPE)_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP).t7 -version $(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th test.lua -log_mode file \
			-val_image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
			-val_label_file_h5 mscoco2014_val_$(TYPE).h5 -model_type vgg -test_mode file \
            -test_cp data/Microsoft_COCO/$(VER)/model_$(TYPE)_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP)_fc8.h5 \
            -version $(VER)
            
vgg-$(TYPE)-med:            
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features_med.lua -log_mode console \
			-video_root /net/per610a/export/das11f/plsang/trecvidmed/preprocessed-vgg \
			-batch_size 32 -model_type vgg -num_target $(NDIM) -layer $(LAYER) -version $(VER) \
           	 	-test_cp cp/$(VER)/model_$(TYPE)_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP).t7
                
### DEPENDENCY MODEL

vgg-mydepsv4-train: 
	CUDA_VISIBLE_DEVICES=$(GID) th train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		-train_label_file_h5 mscoco2014_train_mydepsv4.h5 \
		-val_label_file_h5 mscoco2014_val_mydepsv4.h5 \
		-train_image_file_h5 data/Microsoft_COCO/mscoco2014_train_preprocessedimages_vgg.h5 \
		-val_image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size $(BS) -optim $(OP) -num_target 21034 -test_interval 1000 -num_test_image 400 -print_log_interval 10 \
		-vocab_file mscoco2014_train_mydepsv4vocab.json -model_type vgg \
        -learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) \
		2>&1 | tee log/$(VER)/train_vgg_mydepsv4_$(OP)_b$(BS)_wd$(WD)_bias$(BIAS)_lr$(LR).log
        
vgg-mydepsv4-test:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -num_target 21034 -print_log_interval 1000 -batch_size 32 \
            -test_cp cp/$(VER)/model_mydepsv4_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP).t7 -version $(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th test.lua -log_mode file \
			-val_image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
			-val_label_file_h5 mscoco2014_val_mydepsv4.h5 -model_type vgg -test_mode file \
            -test_cp data/Microsoft_COCO/$(VER)/model_mydepsv4_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP)_fc8.h5 \
            -version $(VER)
            
vgg-mydepsv4-med:            
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features_med.lua -log_mode console \
			-video_root /net/per610a/export/das11f/plsang/trecvidmed/preprocessed-vgg \
			-batch_size 32 -model_type vgg -num_target 21034 -layer $(LAYER) -version $(VER) \
           	 	-test_cp cp/$(VER)/model_mydepsv4_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP).t7
            
### MULTITASK MODEL

vgg-multitask-train:
	CUDA_VISIBLE_DEVICES=$(GID) th train_multitask.lua \
        -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
        -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
        -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/Microsoft_COCO/mscoco2014_train_preprocessedimages_vgg.h5 \
		-val_image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
		-test_interval 1000 -num_test_image 400 -reg_type 2 \
		-print_log_interval 10 -model_type vgg -multitask_type 1 \
        -batch_size $(BS) -optim $(OP) -bias_init $(BIAS) -weight_decay $(WD) -version $(VER) -learning_rate $(LR) \
		2>&1 | tee log/$(VER)/train_vgg_multitask_$(OP)_b$(BS)_wd$(WD)_bias$(BIAS)_lr$(LR).log
        
vgg-multitask-test:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features.lua -log_mode console \
			-image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
            -model_type vgg -num_target 22034 -print_log_interval 1000 -batch_size 32 \
            -test_cp cp/$(VER)/model_multitask_mt1_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP).t7 -version $(VER)
	CUDA_VISIBLE_DEVICES=$(GID) th test_multitask.lua -log_mode file \
			-val_image_file_h5 data/Microsoft_COCO/mscoco2014_val_preprocessedimages_vgg.h5 \
			-model_type vgg -test_mode file -version $(VER) \
            -test_cp data/Microsoft_COCO/$(VER)/model_multitask_mt1_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP)_fc8.h5
        
vgg-multitask-med:
	CUDA_VISIBLE_DEVICES=$(GID) th extract_features_med.lua -log_mode console \
			-video_root /net/per610a/export/das11f/plsang/trecvidmed/preprocessed-vgg \
			-batch_size 32 -model_type vgg -num_target 22034 -layer $(LAYER) -version $(VER) \
            -test_cp cp/$(VER)/model_multitask_mt1_vgg_$(OP)_b$(BS)_bias$(BIAS)_lr$(LR)_wd$(WD)_l2_epoch$(EP).t7

### TEST DETECT CONCEPTS

test_ind?=1
detect:
	python detect_concepts.py --coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO_20160302 \
		--input_file /net/per920a/export/das14a/satoh-lab/plsang/trecvidmed/feature/mydeps/vgg16l-mydepsv4.fc8-conv-sigmoid.h5 \
		--data fc8-conv-sigmoid --test_ind $(test_ind)
	python detect_concepts.py --coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		--input_file /net/per920a/export/das14a/satoh-lab/plsang/trecvidmed/feature/mydeps/depnet_vgg_fc8.h5 \
		--data data --test_ind $(test_ind) 
