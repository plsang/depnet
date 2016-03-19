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


vgg-myconceptsv3: 
	CUDA_VISIBLE_DEVICES=4 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size 1 -optim adam -ft_lr_mult 10 | tee log/train_1_vgg_myconceptsv3.log


vgg-mydepsv4: 
	CUDA_VISIBLE_DEVICES=4 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		-train_label_file_h5 mscoco2014_train_mydepsv4.h5 \
		-val_label_file_h5 mscoco2014_val_mydepsv4.h5 \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size 8 -optim adam -num_target 21034 -ft_lr_mult 10 -test_interval 1000 -num_test_image 400 -print_log_interval 1 \
		-vocab_file mscoco2014_train_mydepsv4vocab.json \
		2>&1 | tee log/train_b8_vgg_mydepsv4.log

vgg-mydepsv4-bias: 
	CUDA_VISIBLE_DEVICES=5 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		-train_label_file_h5 mscoco2014_train_mydepsv4.h5 \
		-val_label_file_h5 mscoco2014_val_mydepsv4.h5 \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-batch_size 4 -optim adam -num_target 21034 -ft_lr_mult 10 -test_interval 1000 -num_test_image 400 -print_log_interval 1 \
		-vocab_file mscoco2014_train_mydepsv4vocab.json -bias_init -6.58 \
		2>&1 | tee log/train_b4_vgg_mydepsv4_bias.log

milnor-myconceptsv3-b1:	
	CUDA_VISIBLE_DEVICES=4 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 1 -optim adam -num_test_image 400 -test_interval 1000 -ft_lr_mult 10 -model_type milnor -print_log_interval 1 -bias_init -6.58 \
		| tee log/train_b1_milnor_myconceptsv3.log

milnor-myconceptsv3-b8:	
	CUDA_VISIBLE_DEVICES=6 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 8 -optim adam -num_test_image 400 -test_interval 1000 -ft_lr_mult 10 -model_type milnor -bias_init -6.58 -print_log_interval 1 \
		| tee log/train_b8_milnor_myconceptsv3.log

milnor-myconceptsv3-b4:	
	#CUDA_VISIBLE_DEVICES=6 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
        #        -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
        #        -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
	#	-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
	#	-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
	#	-batch_size 4 -optim adam -num_test_image 400 -test_interval 1000 -ft_lr_mult 10 -model_type milnor -bias_init -6.58 -max_iters 21000 \
	#	| tee log/train_b4_milnor_myconceptsv3.log
	CUDA_VISIBLE_DEVICES=6 th test.lua -log_mode file -test_cp cp/model_adam_milnor_b4_bias-6.580000_lr0.000010_iter20696_myconceptsv3.t7 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 
	CUDA_VISIBLE_DEVICES=6 th train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 4 -optim adam -num_test_image 400 -test_interval 1000 -ft_lr_mult 10 -model_type milnor -bias_init -5 -max_iters 21000 \
		| tee log/train_b4_milnor_myconceptsv3_bias-5.log
	CUDA_VISIBLE_DEVICES=6 th test.lua -log_mode file -test_cp cp/model_myconceptsv3_adam_milnor_b4_bias-5.000000_lr0.000010.t7 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 

milmax-myconceptsv3-b8:	
	CUDA_VISIBLE_DEVICES=7 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 8 -optim adam -num_test_image 400 -test_interval 1000 -ft_lr_mult 10 -model_type milmax -bias_init 0 -print_log_interval 1 \
		| tee log/train_b8_milmax_myconceptsv3.log

milnor-mydepsv4:	
	CUDA_VISIBLE_DEVICES=4 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		-train_label_file_h5 mscoco2014_train_mydepsv4.h5 \
		-val_label_file_h5 mscoco2014_val_mydepsv4.h5 \
		-vocab_file mscoco2014_train_mydepsv4vocab.json \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 4 -optim adam -test_interval 1000 -ft_lr_mult 10 -model_type milnor -num_test_image 400 \
		-num_target 21034 -print_log_interval 1 -bias_init -6.58 \
		2>&1 | tee log/train_b4_milnor_mydepsv4.log

milmaxnor-mydepsv4:	
	CUDA_VISIBLE_DEVICES=4 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		-train_label_file_h5 mscoco2014_train_mydepsv4.h5 \
		-val_label_file_h5 mscoco2014_val_mydepsv4.h5 \
		-vocab_file mscoco2014_train_mydepsv4vocab.json \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 4 -optim adam -test_interval 1000 -ft_lr_mult 10 -model_type milmaxnor -num_test_image 400 \
		-num_target 21034 -print_log_interval 1 -bias_init -6.58 \
		2>&1 | tee log/train_b4_milmaxnor_mydepsv4.log

milmax-mydepsv4:	
	CUDA_VISIBLE_DEVICES=6 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
		-train_label_file_h5 mscoco2014_train_mydepsv4.h5 \
		-val_label_file_h5 mscoco2014_val_mydepsv4.h5 \
		-vocab_file mscoco2014_train_mydepsv4vocab.json \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 4 -optim adam -test_interval 1000 -ft_lr_mult 10 -model_type milmax -num_test_image 400 \
		-print_log_interval 1 -bias_init 0 \
		2>&1 | tee log/train_b4_milmaxnor_mydepsv4.log

train7:	
	CUDA_VISIBLE_DEVICES=7 th -i train.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 1 -optim adam -num_test_image 400 -test_interval 100 \
		-ft_lr_mult 10 -model_type milnor -bias_init -5 -print_log_interval 1 -debug 1

test_vgg:
	#CUDA_VISIBLE_DEVICES=2 th test.lua -log_mode file -test_cp cp/model_adam_b16_iter10348.t7
	#CUDA_VISIBLE_DEVICES=2 th test.lua -log_mode file -test_cp cp/model_adam_b16_iter5174.t7
	#CUDA_VISIBLE_DEVICES=2 th test.lua -log_mode file -test_cp cp/model_adam_b1_iter82783.t7
	#CUDA_VISIBLE_DEVICES=2 th test.lua -log_mode file -test_cp cp/model_adam1_iter80000.t7
	#CUDA_VISIBLE_DEVICES=6 th test.lua -log_mode file -test_cp cp/model_adam_vgg_b4_lr0.000010_iter20696.t7 \
	#	-val_label_file_h5 mscoco2014_val_mydepsv4.h5 -debug 1
	#CUDA_VISIBLE_DEVICES=5 th test.lua -log_mode file -test_cp cp/model_adam_vgg_b8_bias0.000000_lr0.000010_iter10348.t7 \
	#	-val_label_file_h5 mscoco2014_val_mydepsv4.h5

test_mil:
	#CUDA_VISIBLE_DEVICES=2 th test.lua -log_mode file -test_cp cp/model_adam_milnor_b8_lr0.000010_iter10348.t7 -model_type milnor \
	#		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5
	#CUDA_VISIBLE_DEVICES=2 th test.lua -log_mode file -test_cp cp/model_adam_milmax_b8_lr0.000010_iter10348.t7 -model_type milmax \
	#		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5
	CUDA_VISIBLE_DEVICES=7 th test.lua -log_mode file -test_cp cp/model_adam_milnor_b4_lr0.000010_iter20696_mydepsv4.t7 \
			-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
			-val_label_file_h5 mscoco2014_val_mydepsv4.h5 -model_type milnor


#### MULTI TASKS

multitask-milmaxnor-mt1:	
	CUDA_VISIBLE_DEVICES=4 th train_multitask.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 4 -optim adam -test_interval 1000 -ft_lr_mult 10 -num_test_image 400 \
		-print_log_interval 1 -bias_init -6.58 -model_type milmaxnor -multitask_type 1 \
		2>&1 | tee log/train_b4_milmaxnor_multitask_mt1.log

multitask-milmaxnor-mt2:	
	CUDA_VISIBLE_DEVICES=1 th train_multitask.lua -coco_data_root /net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO \
                -cnn_proto /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
                -cnn_model /net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
		-train_image_file_h5 data/mscoco2014_train_preprocessedimages_msmil.h5 \
		-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
		-batch_size 4 -optim adam -test_interval 1000 -ft_lr_mult 10 -num_test_image 400 \
		-print_log_interval 1 -bias_init -6.58 -model_type milmaxnor -multitask_type 2 \
		2>&1 | tee log/train_b4_milmaxnor_multitask_mt2.log

multitask-test:
	CUDA_VISIBLE_DEVICES=4 th test_multitask.lua -log_mode console -test_cp cp/model_myconceptsv3-mydepsv4_milmaxnor_adam_b4_bias-6.580000_lr0.000010_epoch0.t7 \
			-val_image_file_h5 data/mscoco2014_val_preprocessedimages_msmil.h5 \
			-model_type milmaxnor -print_log_interval 10

