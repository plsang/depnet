CLCV_ROOT = ../clcv/resources
CLCV_CORPORA_ROOT = $(CLCV_ROOT)/corpora
CLCV_DATA_ROOT = $(CLCV_ROOT)/data
CLCV_TOOLS_ROOT = $(CLCV_ROOT)/tools
LOG_ROOT = ./log

# corpora/data/tools directories
MSRVTT_SYM = msrvtt
MSRVTT_ROOT = $(CLCV_CORPORA_ROOT)/$(MSRVTT_SYM)
MSRVTT_DATA_ROOT = $(CLCV_DATA_ROOT)/$(MSRVTT_SYM)

TVV2T_SYM = tvv2t
TVV2T_ROOT = $(CLCV_CORPORA_ROOT)/$(TVV2T_SYM)
TVV2T_DATA_ROOT = $(CLCV_DATA_ROOT)/$(TVV2T_SYM)
TVV2T_SETS = train test

MODEL_ROOT = $(CLCV_DATA_ROOT)/cv_models
SHARED_ROOT = /mnt/data

DEPNET_ROOT = ../depnet

MSRVTT_SETS = train val 
MODEL_SET = myconceptsv3 mydepsv4 mypasv4 mypasprepv4 exconceptsv3 exdepsv4 expasv4 expasprepv4
MULTITASK_MODEL_SET = myconceptsv3mydepsv4 myconceptsv3mypasv4 myconceptsv3mypasprepv4 \
			exconceptsv3exdepsv4 exconceptsv3expasv4 exconceptsv3expasprepv4

ATTACH_VOCAB_PY = ./attachvocab2h5.py
ATTACH_IDS_PY = ./attachIDs2h5.py

### DEFAULT PARAMETERS

NDIM?=1000
VER?=v2
GID?=5
WD?=0
LR?=0.00001
BIAS?=0
BS?=16
OP?=adam
EP?=1000

MODEL_ID = depnet-msrvtt

# function to compute set/model/val name from filename
setnameof = $(word 2,$(subst _, ,$(basename $(notdir $(1)))))
modeltypeof = $(word 2,$(subst -, ,$(word 4,$(subst _, ,$(basename $(notdir $(1)))))))
valtypeof = $(word 3,$(subst -, ,$(word 4,$(subst _, ,$(basename $(notdir $(1)))))))
dataprefixof = $(word 1,$(subst _, ,$(basename $(notdir $(1)))))

##### TARGETS

all: prepo_vgg prepo_msmil

extract-vgg: $(patsubst %,extract-vgg-%,$(MODEL_SET) $(MULTITASK_MODEL_SET))
extract-msmil: $(patsubst %,extract-msmil-%,$(MODEL_SET) $(MULTITASK_MODEL_SET))
extract-resmap: $(patsubst %,extract-resmap-%,$(MODEL_SET))

###### PRE-PROCESSING

prepo_vgg: prepo_vgg_rgb prepo_vgg_flow

prepo_vgg_rgb: $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_rgb.h5,$(MSRVTT_SETS)) \
	$(patsubst %,$(TVV2T_DATA_ROOT)/tvv2t_%_image_depnet_preprocessedimages_vgg_rgb.h5,$(TVV2T_SETS))
prepo_vgg_flow: $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_flow.h5,$(MSRVTT_SETS)) \
	$(patsubst %,$(TVV2T_DATA_ROOT)/tvv2t_%_image_depnet_preprocessedimages_vgg_flow.h5,$(TVV2T_SETS))
 
$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_rgb.h5: json_file = $(patsubst %.h5,%.json,$@)
$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_rgb.h5: \
    $(MSRVTT_ROOT)/msrvtt_captions_%.json
	mkdir -p $(LOG_ROOT)/prepo
	python $(DEPNET_ROOT)/preprocess_video.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(json_file) \
		--image_root $(MSRVTT_ROOT)/TrainValFlow \
		--image_height 240 --image_width 320 --type rgb --frame_sample 8 
		2>&1 | tee $(LOG_ROOT)/prepo/msrvtt_$*_image_depnet_preprocessedimages_vgg_rgb.txt

$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_flow.h5: json_file = $(patsubst %.h5,%.json,$@)
$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_flow.h5: \
    $(MSRVTT_ROOT)/msrvtt_captions_%.json
	mkdir -p $(LOG_ROOT)/prepo
	python $(DEPNET_ROOT)/preprocess_video.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(json_file) \
		--image_root $(MSRVTT_ROOT)/TrainValFlow \
		--image_height 240 --image_width 320 --type flow
		2>&1 | tee $(LOG_ROOT)/prepo/msrvtt_$*_image_depnet_preprocessedimages_vgg_flow.txt

$(TVV2T_DATA_ROOT)/tvv2t_%_image_depnet_preprocessedimages_vgg_rgb.h5: json_file = $(patsubst %.h5,%.json,$@)
$(TVV2T_DATA_ROOT)/tvv2t_%_image_depnet_preprocessedimages_vgg_rgb.h5: \
    $(TVV2T_ROOT)/tvv2t_captions_%.json
	mkdir -p $(LOG_ROOT)/prepo
	python $(DEPNET_ROOT)/preprocess_video.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(json_file) \
		--image_root $(TVV2T_ROOT)/frames/$* \
		--image_height 240 --image_width 320 --type rgb --frame_sample 8 
		2>&1 | tee $(LOG_ROOT)/prepo/tvv2t_$*_image_depnet_preprocessedimages_vgg_rgb.txt

$(TVV2T_DATA_ROOT)/tvv2t_%_image_depnet_preprocessedimages_vgg_flow.h5: json_file = $(patsubst %.h5,%.json,$@)
$(TVV2T_DATA_ROOT)/tvv2t_%_image_depnet_preprocessedimages_vgg_flow.h5: \
    $(TVV2T_ROOT)/tvv2t_captions_%.json
	mkdir -p $(LOG_ROOT)/prepo
	python $(DEPNET_ROOT)/preprocess_video.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(json_file) \
		--image_root $(TVV2T_ROOT)/frames/$* \
		--image_height 240 --image_width 320 --type flow
		2>&1 | tee $(LOG_ROOT)/prepo/tvv2t_$*_image_depnet_preprocessedimages_vgg_flow.txt



prepo_msmil: $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_msmil.h5,$(MSRVTT_SETS)) \
             $(patsubst %,$(DAQUAR_DATA_ROOT)/daquar_%_image_depnet_preprocessedimages_msmil.h5,$(DAQUAR_SETS))

$(MSRVTT_DATA_ROOT)/msrvtt_dev%_image_depnet_preprocessedimages_msmil.h5: json_file = $(patsubst %.h5,%.json,$@)
$(MSRVTT_DATA_ROOT)/msrvtt_dev%_image_depnet_preprocessedimages_msmil.h5: \
    $(MSRVTT_DATA_ROOT)/msrvtt_dev%_imageinfo.json
	mkdir -p $(LOG_ROOT)/prepo
	python $(DEPNET_ROOT)/preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(json_file) \
		--images_root $(MSRVTT_ROOT)/images/train2014 \
		--images_size 565 \
		2>&1 | tee $(LOG_ROOT)/prepo/msrvtt_$*_image_depnet_preprocessedimages_msmil.txt
$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_msmil.h5: json_file = $(patsubst %.h5,%.json,$@)
$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_msmil.h5: \
    $(MSRVTT_DATA_ROOT)/msrvtt_%_imageinfo.json
	mkdir -p $(LOG_ROOT)/prepo
	python $(DEPNET_ROOT)/preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(json_file) \
		--images_root $(MSRVTT_ROOT)/images/$*2014 \
		--images_size 565 \
		2>&1 | tee $(LOG_ROOT)/prepo/msrvtt_$*_image_depnet_preprocessedimages_msmil.txt
$(DAQUAR_DATA_ROOT)/daquar_%_image_depnet_preprocessedimages_msmil.h5: json_file = $(patsubst %.h5,%.json,$@)
$(DAQUAR_DATA_ROOT)/daquar_%_image_depnet_preprocessedimages_msmil.h5: \
    $(DAQUAR_DATA_ROOT)/daquar_%_imageinfo.json
	mkdir -p $(LOG_ROOT)/prepo
	python $(DEPNET_ROOT)/preprocess_image.py --input_json $^ \
		--output_h5 $@ \
		--output_json $(json_file) \
		--images_root $(DAQUAR_ROOT)/nyu_depth_images \
		--images_size 565 \
		2>&1 | tee $(LOG_ROOT)/prepo/daquar_$*_image_depnet_preprocessedimages_msmil.txt


vgg16-model: $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel
$(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel:
	mkdir -p $(MODEL_ROOT)/pretrained-models/vgg-imagenet
	wget -P $(MODEL_ROOT)/pretrained-models/vgg-imagenet \
          http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
          https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt \
          http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
	cd $(MODEL_ROOT)/pretrained-models/vgg-imagenet; tar -xzf caffe_ilsvrc12.tar.gz; rm caffe_ilsvrc12.tar.gz 

######  CONVERT TO DEPNET FORMAT FOR TRAINING USING DEPNET (WHICH DOES NOT SUPPORT STRING DATA)
%depnetfmt.h5:
	python $(DEPNET_ROOT)/convert_index_format.py --input_h5 $*.h5 --output_h5 $@

###### CREATE A SYMBOLIC LINK FROM MY*VOCAB TO EX*VOCAB 
$(MSRVTT_DATA_ROOT)/msrvtt_train_captions_ex%vocab.json:
	cp $(MSRVTT_DATA_ROOT)/msrvtt_train_captions_my$*vocab.json $@

###### VGG MODELS

vgg-train-models: $(patsubst %,vgg-%-model,$(MODEL_SET) $(MULTITASK_MODEL_SET))
vgg-test-models: $(patsubst %,vgg-%-test,$(MODEL_SET) $(MULTITASK_MODEL_SET))


###### model training

VGG_MODEL_PATTERN = $(MODEL_ROOT)/depnet-vgg-%/$(VER)/model_$(MODEL_ID)_epoch$(EP).t7
VGG_VOCAB_PATTERN = $(MODEL_ROOT)/depnet-vgg-%/$(VER)/$(MODEL_ID)_vocab.json
vgg-myconceptsv3-model: $(patsubst %,$(VGG_MODEL_PATTERN),myconceptsv3)
vgg-mydepsv4-model: $(patsubst %,$(VGG_MODEL_PATTERN),mydepsv4)
vgg-mypasv4-model: $(patsubst %,$(VGG_MODEL_PATTERN),mypasv4)
vgg-mypasprepv4-model: $(patsubst %,$(VGG_MODEL_PATTERN),mypasprepv4)
vgg-exconceptsv3-model: $(patsubst %,$(VGG_MODEL_PATTERN),exconceptsv3)
vgg-exdepsv4-model: $(patsubst %,$(VGG_MODEL_PATTERN),exdepsv4)
vgg-expasv4-model: $(patsubst %,$(VGG_MODEL_PATTERN),expasv4)
vgg-expasprepv4-model: $(patsubst %,$(VGG_MODEL_PATTERN),expasprepv4)

$(VGG_MODEL_PATTERN) $(VGG_VOCAB_PATTERN): \
    $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
    $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_preprocessedimages_vgg.h5,train val) \
    $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_preprocessedimages_vgg.json,train val) \
    $(MSRVTT_DATA_ROOT)/msrvtt_train_captions_%depnetfmt.h5 \
    $(MSRVTT_DATA_ROOT)/msrvtt_val_captions_%depnetfmt.h5 \
    $(MSRVTT_DATA_ROOT)/msrvtt_train_captions_%vocab.json
	mkdir -p $(MODEL_ROOT)/depnet-vgg-$*/$(VER)
	LUA_PATH='$(DEPNET_ROOT)/?.lua;;' CUDA_VISIBLE_DEVICES=$(GID) \
	th $(DEPNET_ROOT)/train.lua -coco_data_root $(MSRVTT_DATA_ROOT) \
		-train_image_file_h5 $(notdir $(word 2,$^)) \
		-val_image_file_h5 $(notdir $(word 3,$^)) \
		-train_index_json $(notdir $(word 4,$^)) \
		-val_index_json $(notdir $(word 5,$^)) \
		-train_label_file_h5 $(notdir $(word 6,$^)) \
		-val_label_file_h5 $(notdir $(word 7,$^)) \
		-vocab_file $(notdir $(word 8,$^)) \
		-model_type vgg -cnn_proto $<VGG_ILSVRC_16_layers_deploy.prototxt -cnn_model $< \
		-batch_size $(BS) -optim $(OP) -test_interval -1 -num_test_image -1 -print_log_interval 10 \
		-cp_path $(MODEL_ROOT)/depnet-vgg-$* -model_id $(MODEL_ID) -max_epochs $(EP) \
		-learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) -debug 1 -num_img_channel 3 \
		2>&1 | tee $(MODEL_ROOT)/depnet-vgg-$*/$(VER)/model_$(MODEL_ID).log
	cp $(word 8,$^) $(patsubst %,$(VGG_VOCAB_PATTERN),$*)
	rm $(MSRVTT_DATA_ROOT)/msrvtt_dev1_captions_$*depnetfmt.h5 2> /dev/null 
	rm $(MSRVTT_DATA_ROOT)/msrvtt_dev2_captions_$*depnetfmt.h5 2> /dev/null


VGG_TMODEL_PATTERN = $(MODEL_ROOT)/depnet-vgg-%/$(VER)/model_$(MODEL_ID)_t.t7
VGG_TVOCAB_PATTERN = $(MODEL_ROOT)/depnet-vgg-%/$(VER)/$(MODEL_ID)_vocab_t.json
vgg-myconceptsv3-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),myconceptsv3)
vgg-mydepsv4-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),mydepsv4)
vgg-mypasv4-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),mypasv4)
vgg-mypasprepv4-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),mypasprepv4)
vgg-exconceptsv3-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),exconceptsv3)
vgg-exdepsv4-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),exdepsv4)
vgg-expasv4-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),expasv4)
vgg-expasprepv4-tmodel: $(patsubst %,$(VGG_TMODEL_PATTERN),expasprepv4)

$(VGG_TMODEL_PATTERN) $(VGG_TVOCAB_PATTERN): \
    $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
    $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_flow.h5,train val) \
    $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_vgg_flow.json,train val) \
    $(MSRVTT_DATA_ROOT)/msrvtt_train_captions_%depnetfmt.h5 \
    $(MSRVTT_DATA_ROOT)/msrvtt_val_captions_%depnetfmt.h5 \
    $(MSRVTT_DATA_ROOT)/msrvtt_train_captions_%vocab.json
	mkdir -p $(MODEL_ROOT)/depnet-vgg-$*/$(VER)
	LUA_PATH='$(DEPNET_ROOT)/?.lua;;' CUDA_VISIBLE_DEVICES=$(GID) \
	th $(DEPNET_ROOT)/train.lua -coco_data_root $(MSRVTT_DATA_ROOT) \
                -train_image_file_h5 $(notdir $(word 2,$^)) \
                -val_image_file_h5 $(notdir $(word 3,$^)) \
                -train_index_json $(notdir $(word 4,$^)) \
                -val_index_json $(notdir $(word 5,$^)) \
                -train_label_file_h5 $(notdir $(word 6,$^)) \
                -val_label_file_h5 $(notdir $(word 7,$^)) \
                -vocab_file $(notdir $(word 8,$^)) \
                -model_type vgg -cnn_proto $<VGG_ILSVRC_16_layers_deploy.prototxt -cnn_model $< \
                -batch_size $(BS) -optim $(OP) -test_interval -1 -num_test_image -1 -print_log_interval 10 \
                -cp_path $(MODEL_ROOT)/depnet-vgg-$* -model_id $(MODEL_ID)_t -max_epochs $(EP) \
                -learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) -debug 1 -num_img_channel 20 \
                2>&1 | tee $(MODEL_ROOT)/depnet-vgg-$*/$(VER)/model_$(MODEL_ID).log
	cp $(word 8,$^) $(patsubst %,$(VGG_VOCAB_PATTERN),$*)
	rm $(MSRVTT_DATA_ROOT)/msrvtt_dev1_captions_$*depnetfmt.h5 2> /dev/null
	rm $(MSRVTT_DATA_ROOT)/msrvtt_dev2_captions_$*depnetfmt.h5 2> /dev/null

###### feature extracting

VGG_LAYERS = fc8 fc7
VGG_SUFFIXES = $(patsubst %,-%.h5,$(VGG_LAYERS))
vggfilesof = $(foreach suffix,$(VGG_SUFFIXES),$(addsuffix $(suffix),$(patsubst %,$(2)_%_image_depnet-vgg-$(1),$(3))))
extract-vgg-myconceptsv3: \
    $(patsubst %,$(VGG_MODEL_PATTERN),myconceptsv3) \
    $(call vggfilesof,myconceptsv3,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,myconceptsv3,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-mydepsv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),mydepsv4) \
    $(call vggfilesof,mydepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,mydepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-mypasv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),mypasv4) \
    $(call vggfilesof,mypasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,mypasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-mypasprepv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),mypasprepv4) \
    $(call vggfilesof,mypasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,mypasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-myconceptsv3mydepsv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),myconceptsv3mydepsv4) \
    $(call vggfilesof,myconceptsv3mydepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,myconceptsv3mydepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-myconceptsv3mypasv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),myconceptsv3mypasv4) \
    $(call vggfilesof,myconceptsv3mypasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,myconceptsv3mypasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-myconceptsv3mypasprepv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),myconceptsv3mypasprepv4) \
    $(call vggfilesof,myconceptsv3mypasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,myconceptsv3mypasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-exconceptsv3: \
    $(patsubst %,$(VGG_MODEL_PATTERN),exconceptsv3) \
    $(call vggfilesof,exconceptsv3,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,exconceptsv3,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-exdepsv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),exdepsv4) \
    $(call vggfilesof,exdepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,exdepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-expasv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),expasv4) \
    $(call vggfilesof,expasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,expasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-expasprepv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),expasprepv4) \
    $(call vggfilesof,expasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,expasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-exconceptsv3exdepsv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),exconceptsv3exdepsv4) \
    $(call vggfilesof,exconceptsv3exdepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,exconceptsv3exdepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-exconceptsv3expasv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),exconceptsv3expasv4) \
    $(call vggfilesof,exconceptsv3expasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,exconceptsv3expasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-vgg-exconceptsv3expasprepv4: \
    $(patsubst %,$(VGG_MODEL_PATTERN),exconceptsv3expasprepv4) \
    $(call vggfilesof,exconceptsv3expasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call vggfilesof,exconceptsv3expasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))


VGG_FEAT_FILES = $(foreach val,$(MODEL_SET),$(call vggfilesof,$(val),$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS))) \
                 $(foreach val,$(MODEL_SET),$(call vggfilesof,$(val),$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))) \
		 $(foreach val,$(MULTITASK_MODEL_SET),$(call vggfilesof,$(val),$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS))) \
                 $(foreach val,$(MULTITASK_MODEL_SET),$(call vggfilesof,$(val),$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS)))

.SECONDEXPANSION:
$(VGG_FEAT_FILES): dataprefix = $(call dataprefixof,$@)
$(VGG_FEAT_FILES): setname = $(call setnameof,$@)
$(VGG_FEAT_FILES): modeltype = $(call modeltypeof,$@)
$(VGG_FEAT_FILES): valtype = $(call valtypeof,$@)
$(VGG_FEAT_FILES): layer = $(word 4,$(subst -, ,$(word 4,$(subst _, ,$(basename $(notdir $@))))))
$(VGG_FEAT_FILES): %.h5: \
    $$(dir $$@)$$(call dataprefixof,$$@)_$$(call setnameof,$$@)_image_depnet_preprocessedimages_vgg.h5 \
    $(MODEL_ROOT)/depnet-$$(call modeltypeof,$$@)-$$(call valtypeof,$$@)/$(VER)/model_$(MODEL_ID)_epoch$(EP).t7 \
    $(MODEL_ROOT)/depnet-$$(call modeltypeof,$$@)-$$(call valtypeof,$$@)/$(VER)/$(MODEL_ID)_vocab.json
	NDIM=$$(python -c "import json; v=json.load(open('$(word 3,$^)')); print len(v)") && \
	LUA_PATH='$(DEPNET_ROOT)/?.lua;;' CUDA_VISIBLE_DEVICES=$(GID) \
          th $(DEPNET_ROOT)/extract_features.lua -log_mode console \
          -image_file_h5 $< \
          -model_type $(modeltype) -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
          -test_cp $(word 2,$^) \
          -layer $(layer) -output_file $@
	python $(ATTACH_VOCAB_PY) $(if $(subst fc8,,$(layer)),--internal,) $(word 3,$^) $@
	python $(ATTACH_IDS_PY) $(patsubst %.h5,%.json,$<) $@

###### model testing

#vgg-myconceptsv3-test: $(patsubst %,$(MODEL_ROOT)/depnet-vgg-%/$(VER)/test_model_depnet_epoch$(EP).log,myconceptsv3)
#$(MODEL_ROOT)/depnet-vgg-%/$(VER)/test_model_depnet_epoch$(EP).log: \
    $(MODEL_ROOT)/depnet-vgg-%/$(VER)/model_depnet_epoch$(EP).t7 \
    $(MSRVTT_DATA_ROOT)/msrvtt_dev2_image_depnet_preprocessedimages_vgg.h5 \
    $(MSRVTT_DATA_ROOT)/msrvtt_dev2_captions_%.h5
#	LUA_PATH='$(DEPNET_ROOT)/?.lua;;' CUDA_VISIBLE_DEVICES=$(GID) \
          th $(DEPNET_ROOT)/test.lua -log_mode file -log_dir $(MODEL_ROOT)/depnet-vgg-$* \
          -coco_data_root $(MSRVTT_DATA_ROOT) \
          -val_image_file_h5 $(word 2,$^) \
          -val_label_file_h5 msrvtt_dev2_captions_$*.h5 -model_type vgg -test_mode file \
          -test_cp $< \
          -version $(VER)



###### MSMIL MODEL

msmil-train-models: $(patsubst %,msmil-%-model,$(MODEL_SET) $(MULTITASK_MODEL_SET))
msmil-test-models: $(patsubst %,msmil-%-test,$(MODEL_SET) $(MULTITASK_MODEL_SET))


###### model training

MSMIL_MODEL_PATTERN = $(MODEL_ROOT)/depnet-msmil-%/$(VER)/model_$(MODEL_ID)_epoch$(EP).t7
MSMIL_VOCAB_PATTERN = $(MODEL_ROOT)/depnet-msmil-%/$(VER)/$(MODEL_ID)_vocab.json
msmil-myconceptsv3-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),myconceptsv3)
msmil-mydepsv4-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),mydepsv4)
msmil-mypasv4-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),mypasv4)
msmil-mypasprepv4-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),mypasprepv4)
msmil-exconceptsv3-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),exconceptsv3)
msmil-exdepsv4-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),exdepsv4)
msmil-expasv4-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),expasv4)
msmil-expasprepv4-model: $(patsubst %,$(MSMIL_MODEL_PATTERN),expasprepv4)

$(MSMIL_MODEL_PATTERN) $(MSMIL_VOCAB_PATTERN): \
    $(MODEL_ROOT)/pretrained-models/vgg-imagenet/VGG_ILSVRC_16_layers.caffemodel \
    $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_msmil.h5,dev1 dev2) \
    $(patsubst %,$(MSRVTT_DATA_ROOT)/msrvtt_%_image_depnet_preprocessedimages_msmil.json,dev1 dev2) \
    $(MSRVTT_DATA_ROOT)/msrvtt_dev1_captions_%depnetfmt.h5 \
    $(MSRVTT_DATA_ROOT)/msrvtt_dev2_captions_%depnetfmt.h5 \
    $(MSRVTT_DATA_ROOT)/msrvtt_dev1_captions_%vocab.json
	mkdir -p $(MODEL_ROOT)/depnet-msmil-$*/$(VER)
	LUA_PATH='$(DEPNET_ROOT)/?.lua;;' CUDA_VISIBLE_DEVICES=$(GID) \
          th $(DEPNET_ROOT)/train.lua -coco_data_root $(MSRVTT_DATA_ROOT) \
		-train_label_file_h5 msrvtt_dev1_captions_$*depnetfmt.h5 \
		-val_label_file_h5 msrvtt_dev2_captions_$*depnetfmt.h5 \
		-train_image_file_h5 $(notdir $(word 2,$^)) \
		-val_image_file_h5 $(notdir $(word 3,$^)) \
		-train_index_json $(notdir $(word 4,$^)) \
		-val_index_json $(notdir $(word 5,$^)) \
		-cnn_proto $<VGG_ILSVRC_16_layers_deploy.prototxt -cnn_model $< \
		-batch_size $(BS) -optim $(OP) -test_interval 1000 -num_test_image 400 -print_log_interval 10 \
		-vocab_file msrvtt_dev1_captions_$*vocab.json -model_type milmaxnor \
                -cp_path $(MODEL_ROOT)/depnet-msmil-$* -model_id $(MODEL_ID) -max_epochs $(EP) \
		-learning_rate $(LR) -weight_decay $(WD) -bias_init $(BIAS) -version $(VER) \
		2>&1 | tee $(MODEL_ROOT)/depnet-msmil-$*/$(VER)/model_$(MODEL_ID)_epoch$(EP).log
	cp $(word 8,$^) $(patsubst %,$(MSMIL_VOCAB_PATTERN),$*)
	rm $(MSRVTT_DATA_ROOT)/msrvtt_dev1_captions_$*depnetfmt.h5 2> /dev/null
	rm $(MSRVTT_DATA_ROOT)/msrvtt_dev2_captions_$*depnetfmt.h5 2> /dev/null



###### feature extracting

MSMIL_LAYERS = fc8 fc7
MSMIL_SUFFIXES = $(patsubst %,-%.h5,$(MSMIL_LAYERS))
msmilfilesof = $(foreach suffix,$(MSMIL_SUFFIXES),$(addsuffix $(suffix),$(patsubst %,$(2)_%_image_depnet-msmil-$(1),$(3))))
extract-msmil-myconceptsv3: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),myconceptsv3) \
    $(call msmilfilesof,myconceptsv3,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,myconceptsv3,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-mydepsv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),mydepsv4) \
    $(call msmilfilesof,mydepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,mydepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-mypasv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),mypasv4) \
    $(call msmilfilesof,mypasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,mypasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-mypasprepv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),mypasprepv4) \
    $(call msmilfilesof,mypasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,mypasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-myconceptsv3mydepsv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),myconceptsv3mydepsv4) \
    $(call msmilfilesof,myconceptsv3mydepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,myconceptsv3mydepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-myconceptsv3mypasv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),myconceptsv3mypasv4) \
    $(call msmilfilesof,myconceptsv3mypasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,myconceptsv3mypasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-myconceptsv3mypasprepv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),myconceptsv3mypasprepv4) \
    $(call msmilfilesof,myconceptsv3mypasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,myconceptsv3mypasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-exconceptsv3: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),exconceptsv3) \
    $(call msmilfilesof,exconceptsv3,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,exconceptsv3,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-exdepsv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),exdepsv4) \
    $(call msmilfilesof,exdepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,exdepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-expasv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),expasv4) \
    $(call msmilfilesof,expasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,expasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-expasprepv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),expasprepv4) \
    $(call msmilfilesof,expasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,expasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-exconceptsv3exdepsv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),exconceptsv3exdepsv4) \
    $(call msmilfilesof,exconceptsv3exdepsv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,exconceptsv3exdepsv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-exconceptsv3expasv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),exconceptsv3expasv4) \
    $(call msmilfilesof,exconceptsv3expasv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,exconceptsv3expasv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))
extract-msmil-exconceptsv3expasprepv4: \
    $(patsubst %,$(MSMIL_MODEL_PATTERN),exconceptsv3expasprepv4) \
    $(call msmilfilesof,exconceptsv3expasprepv4,$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS)) \
    $(call msmilfilesof,exconceptsv3expasprepv4,$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))

MSMIL_FEAT_FILES = \
    $(foreach val,$(MODEL_SET),$(call msmilfilesof,$(val),$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS))) \
    $(foreach val,$(MODEL_SET),$(call msmilfilesof,$(val),$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS))) \
    $(foreach val,$(MULTITASK_MODEL_SET),$(call msmilfilesof,$(val),$(MSRVTT_DATA_ROOT)/msrvtt,$(MSRVTT_SETS))) \
    $(foreach val,$(MULTITASK_MODEL_SET),$(call msmilfilesof,$(val),$(DAQUAR_DATA_ROOT)/daquar,$(DAQUAR_SETS)))

$(MSMIL_FEAT_FILES): dataprefix = $(call dataprefixof,$@)
$(MSMIL_FEAT_FILES): setname = $(call setnameof,$@)
$(MSMIL_FEAT_FILES): modeltype = $(call modeltypeof,$@)
$(MSMIL_FEAT_FILES): valtype = $(call valtypeof,$@)
$(MSMIL_FEAT_FILES): layer = $(word 4,$(subst -, ,$(word 4,$(subst _, ,$(basename $(notdir $@))))))
$(MSMIL_FEAT_FILES): %.h5: \
    $$(dir $$@)$$(call dataprefixof,$$@)_$$(call setnameof,$$@)_image_depnet_preprocessedimages_msmil.h5 \
    $(MODEL_ROOT)/depnet-$$(call modeltypeof,$$@)-$$(call valtypeof,$$@)/$(VER)/model_$(MODEL_ID)_epoch$(EP).t7 \
    $(MODEL_ROOT)/depnet-$$(call modeltypeof,$$@)-$$(call valtypeof,$$@)/$(VER)/$(MODEL_ID)_vocab.json
	NDIM=$$(python -c "import json; v=json.load(open('$(word 3,$^)')); print len(v)") && \
          LUA_PATH='$(DEPNET_ROOT)/?.lua;;' CUDA_VISIBLE_DEVICES=$(GID) \
          th $(DEPNET_ROOT)/extract_features.lua -log_mode console \
	  -image_file_h5 $< \
          -model_type milmaxnor -print_log_interval 1000 -num_target $${NDIM} -batch_size $(BS) -version $(VER) \
          -test_cp $(word 2,$^) \
          -layer $(layer) -output_file $@
	python $(ATTACH_VOCAB_PY) $(if $(subst fc8,,$(layer)),--internal,) $(word 3,$^) $@
	python $(ATTACH_IDS_PY) $(patsubst %.h5,%.json,$<) $@

