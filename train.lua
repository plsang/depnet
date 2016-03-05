--[[
Class to train
]]--

require 'nn'
require 'cudnn'
require 'CocoData'
require 'MultilabelCrossEntropyCriterion'
require 'eval_utils'

-- dbg = require 'debugger'

local model_utils = require 'model_utils'
local optim_utils = require 'optim_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-coco_data_root', '/home/ec2-user/data/Microsoft_COCO', 'path to coco data root')
cmd:option('-train_image_file_h5', 'data/coco_train.h5', 'path to the prepressed image data')
cmd:option('-val_image_file_h5', 'data/coco_val.h5', 'path to the prepressed image data')
cmd:option('-train_label_file_h5', 'mscoco2014_train_myconceptsv3.h5', 'file name of the prepressed train label data')
cmd:option('-val_label_file_h5', 'mscoco2014_val_myconceptsv3.h5', 'file name of the prepressed val label data')
cmd:option('-num_target', 1000, 'Number of target concepts')
cmd:option('-num_test_image', 1600, 'Number of test image.')
cmd:option('-test_interval', 1000, 'Number of test image.')
cmd:option('-print_log_interval', 20, 'Number of test image.')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-learning_rate', 0.000015625, 'learning rate for sgd')
cmd:option('-gamma_factor', 0.1, 'factor to reduce learning rate')
cmd:option('-learning_rate_decay_interval', 80000, 'learning rate for sgd')
cmd:option('-momentum', 0.99, 'momentum for sgd')
cmd:option('-weight_decay', 0.0005, 'momentum for sgd')
cmd:option('-max_iters', 1000000)
cmd:option('-save_cp_interval', 10000, 'to save a check point every interval number of iterations')
cmd:option('-test_cp', '', 'name of the checkpoint to test')
cmd:option('-cp_path', 'cp', 'path to save checkpoints')
cmd:option('-phase', 'train', 'phase (train/test)')
cmd:option('-model_id', '1', 'id of the model. will be put in the check point name')
cmd:text()
local opt = cmd:parse(arg)

print(opt)

--- Test loading Coco data
local train_loader = CocoData{image_file_h5 = opt.train_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.train_label_file_h5), 
    num_target = opt.num_target, 
    batch_size = opt.batch_size}

local val_loader = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5),
    num_target = opt.num_target, 
    batch_size = opt.batch_size}

local eval = eval_utils()

---
local model = model_utils.finetune_vgg(opt):cuda() -- model must be global, otherwise, C++ exception!!!
local criterion = nn.MultilabelCrossEntropyCriterion():cuda()

print(model.modules)

local params, grad_params = model:getParameters()
print('total number of parameters: ', params:nElement(), grad_params:nElement())
assert(params:nElement() == grad_params:nElement())

--
model:training()
-- model:evaluate()  -- this would change the behavior of modules such as dropout (that have randomized factor)

local function eval_loss()
    model:evaluate()
    val_loader:reset() -- reset interator
    eval:reset()
    
    print('evaluating...')
    
    local eval_iters = torch.ceil(opt.num_test_image/opt.batch_size)
    local total_loss = 0
    for iter=1, eval_iters do
        local data = val_loader:getBatch()
        local outputs = model:forward(data.images:cuda())
        local iter_loss = criterion:forward(outputs, data.labels:cuda())
        total_loss = total_loss + iter_loss 
        
        eval:cal_precision_recall(outputs, data.labels)
        -- handle the case when the number of test images are not divisible by the batch_size
        if iter == num_iters then

        end
    end    
    
    print ('eval loss = ', total_loss/eval_iters)
    eval:print_precision_recall()
    model:training() -- back to the training mode
end

-- local feval = function(x)
local function feval(x)
    if x ~= params then params:copy(x) end
    grad_params:zero()
    
    local data = train_loader:getBatch()
    local outputs = model:forward(data.images:cuda())
    
    local loss = criterion:forward(outputs, data.labels:cuda())
    local df_do = criterion:backward(outputs, data.labels:cuda())
    
    model:backward(data.images:cuda(), df_do)
    return loss, grad_params
end

local optim_state = {
    learningRate = opt.learning_rate,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum,
    learningRateDecay = opt.learning_rate_decay,
}

local function sgd_layer(layer, opt)
    local params_, grad_params_ = layer:getParameters()
    --print(params_:nElement(), grad_params_:nElement(), layer.weight:nElement() + layer.bias:nElement(), 
    --    layer.gradWeight:nElement() + layer.gradBias:nElement())
    wlen = layer.weight:nElement()
    blen = layer.bias:nElement()

    -- add weight decay to weight
    grad_params_[{{1,wlen}}]:add(optim_state.weightDecay*opt.w_wd_mult, params_[{{1,wlen}}])
    if opt.b_wd_mult ~= 0 then
        grad_params_[{{wlen+1, wlen+blen}}]:add(optim_state.weightDecay*opt.b_wd_mult, params_[{{wlen+1, wlen+blen}}])
    end
    
    params_[{{1,wlen}}]:add(-optim_state.learningRate*opt.w_lr_mult, grad_params_[{{1,wlen}}])
    params_[{{wlen+1, wlen+blen}}]:add(-optim_state.learningRate*opt.b_lr_mult, grad_params_[{{wlen+1, wlen+blen}}])
    
    collectgarbage() 
end

local function sgd_group(group)
    if group.opt.w_lr_mult ~= 0 then
        for _,m in ipairs(group.modules) do
            if m.weight and m.bias then
                sgd_layer(m, group.opt)
            elseif m.weight or m.bias then
                error('Layer that has weight but no bias or vice versa')
            end
        end    
    end
end

-- TRAINING LOOP --- 
eval_loss()
local iter = 0
while true do 
    
    iter = iter + 1
    
    timer = torch.Timer() 
    
    -- Call forward/backward
    local loss = feval(params)
    
    -- Now update params acordingly
    for group = 1,#model do
        sgd_group(model.modules[group])
    end
    ----
    
    -- local _, loss = optim_utils.sgd(feval, params, optim_state)
    
    if iter % opt.print_log_interval == 0 then 
        print(string.format('%s: iter %d, loss = %.6f (%.3f s/iter)', os.date(), iter, loss, timer:time().real))
        collectgarbage() 
    end
   
    -- test loss
    if (iter % opt.test_interval == 0) then
        eval_loss()
    end
    
    -- save checkpoints
    if (iter % opt.save_cp_interval == 0 or iter == opt.max_iters) then
        local cp_path = path.join(opt.cp_path, 'model_' .. opt.model_id .. '_iter' .. iter)
        local cp = {}
        cp.opt = opt
        cp.iter = iter
        cp.loss = loss
        cp.params = params
        print('saving checkpoint...')
        torch.save(cp_path .. '.t7', cp)
    end

    if iter % opt.learning_rate_decay_interval == 0 then
        optim_state.learningRate = optim_state.learningRate/opt.gamma_factor
        print('new learning rate', optim_state.learningRate)
    end
    
    if iter >= opt.max_iters then 
        eval_loss()
        break 
    end
end    



