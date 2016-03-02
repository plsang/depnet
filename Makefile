CLCV_ROOT = $(HOME)
CLCV_CORPORA_ROOT = $(CLCV_ROOT)/corpora
DATA_ROOT = ./data

MSCOCO_SYM = Microsoft_COCO
MSCOCO_ROOT = $(CLCV_CORPORA_ROOT)/$(MSCOCO_SYM)


prepo: $(DATA_ROOT)/coco_train.h5 $(DATA_ROOT)/coco_val.h5
$(DATA_ROOT)/coco_%.h5: 
	python preprocess_image.py --input_json $(MSCOCO_ROOT)/annotations/captions_$*2014.json \
		--output_h5 $(DATA_ROOT)/coco_$*.h5 \
		--images_root $(MSCOCO_ROOT)/images/$*2014 \
		--images_size 224

train: 
	th -i train.lua | tee train.log
