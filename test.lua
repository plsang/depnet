--[[
Class to train
]]--

require 'loadcaffe'
require 'CocoData'
require 'nn'
require 'cudnn'
require 'optim'

local dbg = require 'debugger'
local model_utils = require 'model_utils'
local coco_data_root = '/home/ec2-user/data/Microsoft_COCO'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a caffe model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-train_image_file_h5', 'data/coco_train.h5', 'path to the prepressed image data')
cmd:option('-val_image_file_h5', 'data/coco_val.h5', 'path to the prepressed image data')
cmd:option('-train_label_file_h5', paths.concat(coco_data_root, 'mscoco2014_train_myconceptsv3.h5'), 'path to the prepressed label data')
cmd:option('-val_label_file_h5', paths.concat(coco_data_root, 'mscoco2014_val_myconceptsv3.h5'), 'path to the prepressed label data')
cmd:option('-num_target', 1000, 'Number of target concepts')
cmd:option('-num_test_image', 1600, 'Number of test image. -1 for all')
cmd:option('-batch_size', 16, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-test_cp', '', 'name of the checkpoint to test')
cmd:option('-cp_path', 'cp', 'path to save checkpoints')
cmd:option('-phase', 'test', 'phase (train/test)')

cmd:text()
local opt = cmd:parse(arg)

print(opt)

--val_loader = CocoData{image_file_h5 = opt.val_image_file_h5, label_file_h5 = opt.val_label_file_h5, 
--    num_target = opt.num_target, batch_size = opt.batch_size}

val_loader = CocoData{image_file_h5 = opt.train_image_file_h5, label_file_h5 = opt.train_label_file_h5, 
    num_target = opt.num_target, batch_size = opt.batch_size}

-- vgg_model, criterion = model_utils.load_vgg(opt)
vgg_model = model_utils.define_vgg(opt)

vgg_model:evaluate()  -- this would change the behavior of modules such as dropout (that have randomized factor)

local num_iters = torch.ceil(opt.num_test_image/opt.batch_size)

-- average precision per image
local function average_precision(pred, gold)
    sorted_pred, idx = torch.sort(pred) -- sort descending 
    rank_idx = torch.nonzero(gold):squeeze()
    --print(rank_idx)
    
    -- dbg()
    rank_idx:apply(function(x) local i=idx:eq(x):nonzero()[1][1] return i end)
    sorted_idx = torch.sort(rank_idx)
    --print(sorted_idx)
    ap = 0
    for kk=1,sorted_idx:size(1) do
        ap = ap + kk/sorted_idx[kk]
    end
    ap = ap/sorted_idx:size(1)
    return ap
end
    
local function average_precision_batch(batch_output, batch_label)
    local batch_size = opt.batch_size
    assert(batch_size == batch_output:size(1))
    assert(batch_size == batch_label:size(1))
    local batch_ap = 0
    for i=1, batch_size do
        local pred = batch_output[i]
        local gold = batch_label[i]
        local ap = average_precision(pred, gold)
        batch_ap = batch_ap + ap
    end
    return batch_ap/batch_size
end

local map = 0
for iter=1, num_iters do
    local data = val_loader:getBatch()
    local outputs = vgg_model:forward(data.images:cuda())
    local batch_ap = average_precision_batch(outputs:float(), data.labels)
    map = map + batch_ap
    
    if iter % 20 == 0 then 
        print(string.format('iter %d: map = %.6f', iter, map/iter))
        collectgarbage() 
    end
    
    -- handle the case when the number of test images are not divisible by the batch_size
    if iter == num_iters then
        
    end
end    

map = map/num_iters

