CLCV_ROOT = /net/per920a/export/das14a/satoh-lab/plsang
CLCV_CORPORA_ROOT = $(CLCV_ROOT)/corpora
DATA_ROOT = ./data

MSCOCO_SYM = Microsoft_COCO
MSCOCO_ROOT = $(CLCV_CORPORA_ROOT)/$(MSCOCO_SYM)



prepo_vgg: $(DATA_ROOT)/mscoco2014_train_preprocessedimages_vgg.h5 $(DATA_ROOT)/mscoco2014_val_preprocessedimages_vgg.h5
$(DATA_ROOT)/mscoco2014_%_preprocessedimages_vgg.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 224

prepo_msmil: $(DATA_ROOT)/mscoco2014_train_preprocessedimages_msmil.h5 $(DATA_ROOT)/mscoco2014_val_preprocessedimages_msmil.h5
$(DATA_ROOT)/mscoco2014_%_preprocessedimages_msmil.h5: $(MSCOCO_ROOT)/annotations/captions_%2014.json
	python preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 565

train5: 
	CUDA_VISIBLE_DEVICES=5 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size 1 -test_interval 10000 | tee log/train_1.log


train6: 
	CUDA_VISIBLE_DEVICES=6 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size 16 -learning_rate 2e-4 | tee log/train_16.log


train7: 
	CUDA_VISIBLE_DEVICES=7 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size 32 -learning_rate 4e-4 | tee log/train_32.log

train_dep20k: 
	CUDA_VISIBLE_DEVICES=3 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_label_file_h5 
		-batch_size 1 -learning_rate 4e-4 | tee log/train_32.log

test1:
	CUDA_VISIBLE_DEVICES=4 th -i test.lua -test_cp cp/model_b1_iter160000.t7 | tee log/test_b1_iter160000.log

