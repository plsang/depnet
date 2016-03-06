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
cmd:option('-learning_rate', 0.00001, 'learning rate for sgd')
cmd:option('-learning_rate_decay', 0, 'decaying rate for sgd')
cmd:option('-gamma_factor', 0.1, 'factor to reduce learning rate, 0.1 ==> drop 10 times')
cmd:option('-learning_rate_decay_interval', 80000, 'learning rate for sgd')
cmd:option('-momentum', 0.99, 'momentum for sgd')
cmd:option('-weight_decay', 0.0005, 'momentum for sgd')
cmd:option('-max_iters', 1000000)
cmd:option('-save_cp_interval', 0, 'to save a check point every interval number of iterations')
cmd:option('-test_cp', '', 'name of the checkpoint to test')
cmd:option('-cp_path', 'cp', 'path to save checkpoints')
cmd:option('-phase', 'train', 'phase (train/test)')
cmd:option('-model_id', '', 'id of the model. will be put in the check point name')
cmd:option('-phase', 'train', 'phase (train/test)')
cmd:option('-weight_init', 0.001, 'std of gausian to initilize weights & bias')
cmd:option('-bias_init', -6.58, 'initilize bias to contant')
cmd:option('-w_lr_mult', 10, 'learning multipier for weight on the finetuning layer')
cmd:option('-b_lr_mult', 20, 'learning multipier for bias on the finetuning layer')
cmd:option('-loss_weight', 1, 'loss multiplier, to display loss as a bigger value')

cmd:text()
local opt = cmd:parse(arg)

-- update decaying interval
opt.learning_rate_decay_interval = opt.learning_rate_decay_interval/opt.batch_size
if opt.model_id == '' then opt.model_id = opt.batch_size end
if opt.save_cp_interval == 0 then opt.save_cp_interval = opt.learning_rate_decay_interval end

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

local function eval_loss()
    model:evaluate()
    val_loader:reset() -- reset interator
    eval:reset()
    
    local eval_iters = torch.ceil(opt.num_test_image/opt.batch_size)
    local total_loss = 0
    local map = 0
    for iter=1, eval_iters do
        local data = val_loader:getBatch()
        local outputs = model:forward(data.images:cuda())
        local iter_loss = criterion:forward(outputs, data.labels:cuda())
        total_loss = total_loss + iter_loss 
        
        eval:cal_precision_recall(outputs, data.labels)
        local batch_map = eval:cal_mean_average_precision(outputs:float(), data.labels)
        map = map + batch_map
        -- handle the case when the number of test images are not divisible by the batch_size
        if iter == num_iters then

        end
    end    
    
    print (' ==> eval loss = ', opt.loss_weight*total_loss/eval_iters)
    print (' ==> eval map = ', map/eval_iters)
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
    return loss
end

--- Setting finetune layer

local sgd_config = {
    learningRate = opt.learning_rate,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum,
    learningRateDecay = opt.learning_rate_decay,
    w_lr_mult = opt.w_lr_mult,
    b_lr_mult = opt.b_lr_mult   
}

local finetune_graph = model.modules[2]
assert(finetune_graph.frozen == false)
local params_finetune, grad_params_finetune = finetune_graph:getParameters()
assert(params_finetune:nElement(), grad_params_finetune:nElement())
print('Number of finetune parameters', params_finetune:nElement())

local bias_indices = {}
local total_elements = 0
for _, m in ipairs(finetune_graph.modules) do
   if m.weight and m.bias then
        
        local wlen = m.weight:nElement()
        local blen = m.bias:nElement()
        local mlen = wlen + blen
        table.insert(bias_indices, total_elements + wlen + 1)
        table.insert(bias_indices, total_elements + mlen)
        
        if m.name == 'fc8' then
            print('Fine tuning layer found!')
            sgd_config.ft_ind_start = total_elements + 1
            sgd_config.ft_ind_end = total_elements + mlen
            sgd_config.ftb_ind_start = total_elements + wlen + 1 -- fine tune bias index start
            sgd_config.ftb_ind_end = total_elements + mlen       -- fine tune bias index end
            
            if opt.weight_init > 0 then
                print('Initialize parameters of the finetuned layer')
                m.weight:normal(0, opt.weight_init) -- gaussian of zero mean
                m.bias:fill(opt.bias_init)          -- constant value    
            end
        end
        
        total_elements = total_elements + mlen
   elseif m.weight or m.bias then
       error('Layer that has either weight or bias')     
   end
end

assert(total_elements == params_finetune:nElement(), 'number of params mismatch')
assert(sgd_config.ft_ind_start, 'Fine tune layer not found')
-- assign bias indices to sgd config
sgd_config.bias_indices = bias_indices

-- TRAINING LOOP --- 
eval_loss()
local iter = 0
while true do 
    
    iter = iter + 1
    
    timer = torch.Timer() 
    
    -- Call forward/backward with full params input
    local loss = feval(params)
    
    -- Now update params acordingly
    optim_utils.sgd_finetune(params_finetune, grad_params_finetune, sgd_config)
    
    -- local _, loss = optim_utils.sgd(feval, params, sgd_conifg)
    
    if iter % opt.print_log_interval == 0 then 
        print(string.format('%s: iter %d, loss = %f, lr = %g (%.3fs/iter)', 
                os.date(), iter, opt.loss_weight*loss, sgd_config.learningRate, timer:time().real))
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
        sgd_config.learningRate = sgd_config.learningRate * opt.gamma_factor
        print('new learning rate', sgd_config.learningRate)
    end
    
    if iter >= opt.max_iters then 
        eval_loss()
        break 
    end
end    



