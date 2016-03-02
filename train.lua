--[[
Class to train
]]--

require 'loadcaffe'
require 'CocoData'
require 'nn'
require 'cudnn'
require 'optim'
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
cmd:option('-num_test_image', 1600, 'Number of test image.')
cmd:option('-test_interval', 1000, 'Number of test image.')
cmd:option('-batch_size', 16, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-learning_rate', 1e-4, 'learning rate for sgd')
cmd:option('-learning_rate_decay', 1e-6, 'learning rate for sgd')
cmd:option('-momentum', 0.99, 'momentum for sgd')
cmd:option('-weight_decay', 0.0005, 'momentum for sgd')
cmd:option('-max_iters', 1000000)
cmd:option('-save_cp_interval', 10000, 'to save a check point every interval number of iterations')
cmd:option('-test_cp', '', 'name of the checkpoint to test')
cmd:option('-cp_path', 'cp', 'path to save checkpoints')
cmd:option('-phase', 'train', 'phase (train/test)')
cmd:text()
local opt = cmd:parse(arg)

print(opt)

--- Test loading Coco data
train_loader = CocoData{image_file_h5 = opt.train_image_file_h5, label_file_h5 = opt.train_label_file_h5, 
    num_target = opt.num_target, batch_size = opt.batch_size}

val_loader = CocoData{image_file_h5 = opt.val_image_file_h5, label_file_h5 = opt.val_label_file_h5, 
    num_target = opt.num_target, batch_size = opt.batch_size}

---
print('loading model...')
-- vgg_model, criterion = model_utils.load_vgg(opt)
vgg_model, criterion = model_utils.define_vgg(opt)

local params, grad_params = vgg_model:getParameters()
print('total number of parameters: ', params:nElement(), grad_params:nElement())
assert(params:nElement() == grad_params:nElement())

--
vgg_model:training()
-- vgg_model:evaluate()  -- this would change the behavior of modules such as dropout (that have randomized factor)

local function eval_loss()
    vgg_model:evaluate()
    val_loader:reset() -- reset interator
    print('evaluating...')
    
    local eval_iters = torch.ceil(opt.num_test_image/opt.batch_size)
    local total_loss = 0
    for iter=1, eval_iters do
        local data = val_loader:getBatch()
        local outputs = vgg_model:forward(data.images:cuda())
        local iter_loss = criterion:forward(outputs, data.labels:cuda())
        total_loss = total_loss + iter_loss 

        -- handle the case when the number of test images are not divisible by the batch_size
        if iter == num_iters then

        end
    end    
    
    print ('eval loss = ', total_loss/eval_iters)
    vgg_model:training() -- back to the training mode
end

local feval = function(x)
    if x ~= params then params:copy(x) end
    grad_params:zero()
    
    local data = train_loader:getBatch()
    local outputs = vgg_model:forward(data.images:cuda())
    
    local loss = criterion:forward(outputs, data.labels:cuda())
    local df_do = criterion:backward(outputs, data.labels:cuda())
    
    vgg_model:backward(data.images:cuda(), df_do)
    return loss, grad_params
end

optimState = {
    learningRate = opt.learning_rate,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum,
    learningRateDecay = opt.learning_rate_decay,
}

eval_loss()
local iter = 0
while true do 
    
    iter = iter + 1
    
    local _, loss = optim.sgd(feval, params, optimState)
    
    if iter % 20 == 0 then 
        print(string.format('iter %d: loss = %.6f', iter, loss[1] ))
        collectgarbage() 
    end
   
    -- test loss
    if (iter % opt.test_interval == 0) then
        eval_loss()
    end
    
    -- save checkpoints
    if (iter % opt.save_cp_interval == 0 or iter == opt.max_iters) then
        local cp_path = path.join(opt.cp_path, 'model_iter' .. iter)
        local cp = {}
        cp.opt = opt
        cp.iter = iter
        cp.loss = loss[1]
        cp.params = params
        print('saving checkpoint...')
        torch.save(cp_path .. '.t7', cp)
    end

    if iter >= opt.max_iters then 
        eval_loss()
        break 
    end
end    



