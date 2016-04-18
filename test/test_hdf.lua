--[[
Class to train
]]--

require 'nn'
require 'cudnn'

require 'CocoData'
require 'MultilabelCrossEntropyCriterion'
require 'eval_utils'
local model_utils = require 'model_utils'

dbg = require 'debugger'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a caffe model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-coco_data_root', '/net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO', 'path to coco data root')
cmd:option('-train_image_file_h5', 'data/coco_train.h5', 'path to the prepressed image data')
cmd:option('-val_image_file_h5', 'data/coco_val.h5', 'path to the prepressed image data')
cmd:option('-train_label_file_h5', 'mscoco2014_train_mydepsv4.h5', 'file name of the prepressed train label data')
cmd:option('-val_label_file_h5', 'mscoco2014_val_mydepsv4.h5', 'file name of the prepressed val label data')
cmd:option('-num_target', 21034, 'Number of target concepts')
cmd:option('-num_test_image', 40504, 'Number of test image')
cmd:option('-batch_size', 1, 'Number of image per batch')

cmd:text()
local opt = cmd:parse(arg)

print(opt)

local loader = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5),
    num_target = opt.num_target, 
    batch_size = opt.batch_size}

for i=1,10000 do
    print(i)
    data = loader:getBatch(opt.batch_size) 
end

print('done')
