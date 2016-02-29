--[[
Class to train
]]--

require 'loadcaffe'
require 'CocoData'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

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
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-learning_rate', 1e-5, 'learning rate for sgd')
cmd:option('-momentum', 0.99, 'momentum for sgd')
cmd:option('-weight_decay', 0.0005, 'momentum for sgd')
cmd:option('-max_iters', 10)

cmd:text()
local opt = cmd:parse(arg)

print(opt)

--- Test loading Coco data
train_loader = CocoData{image_file_h5 = opt.train_image_file_h5, label_file_h5 = opt.train_label_file_h5, 
    num_target = opt.num_target, batch_size = opt.batch_size}

val_loader = CocoData{image_file_h5 = opt.val_image_file_h5, label_file_h5 = opt.val_label_file_h5, 
    num_target = opt.num_target, batch_size = opt.batch_size}

---
print('Loading...')
vgg_model = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end)

for k,v in pairs(vgg_model) do print(k,v) end

---
print('Removing the softmax layer')
table.remove(vgg_model['modules'])  -- equivalent to pop

print('checking model by sampling temp input...')
print(#vgg_model:cuda():forward(torch.CudaTensor(10, 3, 224, 224)))


---
print('converting first layer conv filters from BGR to RGB...')
local input_layer = vgg_model:get(1)
local w = input_layer.weight:clone()
-- swap weights to R and B channels
input_layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
input_layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])

local params, grad_params = vgg_model:getParameters()
print('total number of parameters: ', params:nElement(), grad_params:nElement())
assert(params:nElement() == grad_params:nElement())

criterion = nn.CrossEntropyCriterion():cuda()

local feval = function(x)
    if x ~= params then params:copy(x) end
    
    vgg_model:training()
    grad_params:zero()
    local data = train_loader:getBatch()
    local outputs = vgg_model:forward(data.images:cuda())
    print(outputs)
    -- print(data.labels)
    local f = criterion:forward(outputs, data.labels:cuda())
    local df_do = criterion:backward(outputs, data.labels:cuda())
    vgg_model:backward(data.images:cuda(), df_do)
    
    return f, grad_params
end

optimState = {
    learningRate = opt.learning_rate,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum,
    -- learningRateDecay = opt.learning_rate_decay,
}

local iter = 0
while true do 
    iter = iter + 1
    
    optim.sgd(feval, params, optimState)
    
    if iter % 10 == 0 then collectgarbage() end
    if opt.max_iters > 0 and iter >= opt.max_iters then break end
end    



