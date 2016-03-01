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
cmd:option('-cp_path', 'cp', 'path to save checkpoints')

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


local feval = function(x)
    if x ~= params then params:copy(x) end
    grad_params:zero()
    
    local data = train_loader:getBatch()
    local outputs = vgg_model:forward(data.images:cuda())
    
    local f = criterion:forward(outputs, data.labels:cuda())
    local df_do = criterion:backward(outputs, data.labels:cuda())
    
    vgg_model:backward(data.images:cuda(), df_do)
    return f, grad_params
end

optimState = {
    learningRate = opt.learning_rate,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum,
    learningRateDecay = opt.learning_rate_decay,
}

local iter = 0
while true do 
    
    iter = iter + 1
    
    _, loss = optim.sgd(feval, params, optimState)
    
    if iter % 20 == 0 then 
        print(string.format('iter %d: loss = %.6f', iter, loss[1] ))
        collectgarbage() 
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

    if iter >= opt.max_iters then break end
end    



