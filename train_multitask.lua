--[[
Class to train
]]--

require 'nn'
require 'cudnn'
local cjson = require 'cjson'

require 'CocoData'
--require 'MultilabelCrossEntropyCriterion'
require 'nn.MultiLabelCrossEntropyCriterion'
require 'eval_utils'

local model_utils = require 'model_utils'
local optim_utils = require 'optim_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-coco_data_root', '/home/ec2-user/data/Microsoft_COCO', 'path to coco data root')
cmd:option('-train_image_file_h5', 'data/coco_train.h5', 'path to the prepressed image data')
cmd:option('-val_image_file_h5', 'data/coco_val.h5', 'path to the prepressed image data')

cmd:option('-train_label_file_h5_task1', 'mscoco2014_train_myconceptsv3.h5', 'file name of the prepressed train label data')
cmd:option('-val_label_file_h5_task1', 'mscoco2014_val_myconceptsv3.h5', 'file name of the prepressed val label data')
cmd:option('-train_label_file_h5_task2', 'mscoco2014_train_mydepsv4.h5', 'file name of the prepressed train label data')
cmd:option('-val_label_file_h5_task2', 'mscoco2014_val_mydepsv4.h5', 'file name of the prepressed val label data')

cmd:option('-vocab_file_task1', 'mscoco2014_train_myconceptsv3vocab.json', 'saving a copy of the vocabulary that was used for training')
cmd:option('-vocab_file_task2', 'mscoco2014_train_mydepsv4vocab.json', 'saving a copy of the vocabulary that was used for training')
cmd:option('-concept_type', 'multitask', 'name of concept type, e.g., myconceptsv3, mydepsv4, empty for auto detect from train_label_file_h5')
cmd:option('-num_target', -1, 'Number of target concepts, -1 for getting from file')
cmd:option('-num_test_image', 400, 'Number of test image, -1 for testing all (40504)')
cmd:option('-test_interval', 10000, 'Number of test image.')
cmd:option('-print_log_interval', 20, 'Number of test image.')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-max_iters', 1000000)
cmd:option('-max_epochs', 10)
cmd:option('-save_cp_interval', -1, 'to save a check point every interval number of iterations')
cmd:option('-test_cp', '', 'name of the checkpoint to test')
cmd:option('-cp_path', 'cp', 'path to save checkpoints')
cmd:option('-phase', 'train', 'phase (train/test)')
cmd:option('-model_id', '', 'id of the model. will be put in the check point name')
cmd:option('-phase', 'train', 'phase (train/test)')
cmd:option('-weight_init', 0.001, 'std of gausian to initilize weights & bias')
cmd:option('-bias_init', 0, 'initilize bias to contant')
cmd:option('-w_lr_mult', 10, 'learning multipier for weight on the finetuning layer')
cmd:option('-b_lr_mult', 20, 'learning multipier for bias on the finetuning layer')
cmd:option('-ft_lr_mult', 1, 'learning multipier for the finetuning layer, same for weight and bias')
cmd:option('-loss_weight', 20, 'loss multiplier, to display loss as a bigger value, and to scale backward gradient')
cmd:option('-seed', 123, 'random number generator seed, used to generate initial gaussian weights of the finetune layer')
cmd:option('-optim', 'adam', 'optimization method: sgd, adam, adaml21')
cmd:option('-learning_rate', 1e-5, 'learning rate for sgd') -- msmil: 0.000015625
cmd:option('-model_type', 'vgg', 'vgg, vggbn, milmax, milnor, milmaxnor')
cmd:option('-finetune_layer_name', 'fc8', 'name of the finetuning layer')
cmd:option('-debug', 0, 'turn debug mode on/off')
cmd:option('-multitask_type', 1, '1: concate, 2: alternate')
-- these options are for SGD
cmd:option('-learning_rate_decay', 0, 'decaying rate for sgd')
cmd:option('-gamma_factor', 0.1, 'factor to reduce learning rate, 0.1 ==> drop 10 times')
cmd:option('-learning_rate_decay_interval', -1, 'learning rate for sgd')
cmd:option('-momentum', 0.99, 'momentum for sgd')
cmd:option('-weight_decay', 0, 'regularization multiplier. 0 to disable [default]. Typical value: 0.0005')
cmd:option('-reg_type', 2, '1: L1 regularization, 2: L2 regularization, 3: L2,1 regularization')
cmd:option('-fc7dim', 4096, 'fc7 dimension')
-- these options are for Adam
cmd:option('-adam_beta1', 0.9, 'momentum for adam')
cmd:option('-adam_beta2', 0.999, 'momentum for adam')
cmd:option('-adam_epsilon', 1e-8, 'momentum for epsilon')

cmd:text()
local opt = cmd:parse(arg)

if opt.debug == 1 then dbg = require 'debugger' end

-- set the manual seed
torch.manualSeed(opt.seed)

-- loading Coco data
local train_loader_task1 = CocoData{image_file_h5 = opt.train_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.train_label_file_h5_task1), 
    batch_size = opt.batch_size}
local train_loader_task2 = CocoData{image_file_h5 = opt.train_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.train_label_file_h5_task2), 
    batch_size = opt.batch_size}
local val_loader_task1 = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5_task1),
    batch_size = opt.batch_size}
local val_loader_task2 = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5_task2),
    batch_size = opt.batch_size}

-- Update some default options
if opt.num_target == -1 then opt.num_target = train_loader_task1:getNumTargets() + train_loader_task2:getNumTargets() end
if opt.num_test_image == -1 then opt.num_test_image = val_loader:getNumImages() end
if opt.concept_type == '' then opt.concept_type = string.split(paths.basename(opt.train_label_file_h5, '.h5'), '_')[3] end
if opt.model_id == '' then 
    opt.model_id = string.format('%s_%s_mt%d_%s_b%d_bias%f_lr%f_wd%f_l%d', 
            opt.concept_type, opt.model_type, opt.multitask_type, 
            opt.optim, opt.batch_size, opt.bias_init, 
            opt.learning_rate, opt.weight_decay, opt.reg_type)
end
opt.iter_per_epoch = math.ceil(train_loader_task1:getNumImages()/opt.batch_size)
if opt.save_cp_interval <= 0 then 
    opt.save_cp_interval = opt.iter_per_epoch
end
if opt.learning_rate_decay_interval == -1 then
    opt.learning_rate_decay_interval = math.ceil(train_loader_task1:getNumImages()/opt.batch_size)
end
print(opt)
------------------------------------------

local eval_task1 = eval_utils()
local eval_task2 = eval_utils()
local eval_all = eval_utils()

local model = model_utils.load_model(opt):cuda()

-- local criterion = nn.MultilabelCrossEntropyCriterion(opt.loss_weight):cuda() -- Lua version
local criterion = nn.MultiLabelCrossEntropyCriterion(opt.loss_weight):cuda() -- C/Cuda version

print(model.modules)

-- Initialization
model_utils.init_finetuning_params(model, opt)

local params, grad_params = model:getParameters()
print('total number of parameters: ', params:nElement(), grad_params:nElement())

-- note: don't use 'config' as a variable name
local optim_config = {
    learningRate = opt.learning_rate,
    weightDecay = opt.weight_decay,
    reg_type = opt.reg_type,
    fc7dim = opt.fc7dim,       -- l2,1 reg
    w_lr_mult = opt.w_lr_mult,
    b_lr_mult = opt.b_lr_mult,
    ft_lr_mult = opt.ft_lr_mult  -- if w and b have the same learning rate
}

if opt.optim == 'sgd' then
    optim_config.momentum = opt.momentum
    optim_config.learningRateDecay = opt.learning_rate_decay
elseif opt.optim == 'adam' or opt.optim == 'adaml21' then
    optim_config.adam_beta1 = opt.adam_beta1
    optim_config.adam_beta2 = opt.adam_beta2
    optim_config.adam_epsilon = opt.adam_epsilon
else
    error('Unknown optimization method', opt.optim)
end

-- update param indices from model 
model_utils.update_param_indices(model, opt, optim_config)

print('Optimization configurations', optim_config) 

local n1 = train_loader_task1:getNumTargets()
local n2 = train_loader_task2:getNumTargets()


-- MAIN LOOP --- 
local iter = 0
local epoch = 1
local loss_history = {}
local val_loss_history = {}

local function eval_loss()
    model:evaluate()
    val_loader_task1:reset() 
    val_loader_task2:reset() 
    
    eval_task1:reset()
    eval_task2:reset()
    eval_all:reset()
    
    print(' ==> evaluating ...') 
    local eval_iters = torch.ceil(opt.num_test_image/opt.batch_size)
    local sum_loss = 0
    
    local map_task1 = 0
    local map_task2 = 0
    local map_all = 0
    
    for iter=1, eval_iters do
        local data1 = val_loader_task1:getBatch() -- get image and label batches
        local data2 = val_loader_task2:getBatch(true) -- get label only
        local images = data1.images:cuda()
        local labels = torch.cat(data1.labels, data2.labels, 2)
        
        local outputs = model:forward(images)
        local iter_loss = criterion:forward(outputs, labels:cuda())
        sum_loss = sum_loss + iter_loss 
        
        eval_task1:cal_precision_recall(outputs[{{},{1,n1}}], labels[{{},{1,n1}}])
        eval_task2:cal_precision_recall(outputs[{{},{n1+1,n1+n2}}], labels[{{},{n1+1,n1+n2}}])
        eval_all:cal_precision_recall(outputs, labels)
        
        local batch_map_task1 = eval_task1:cal_mean_average_precision(outputs[{{},{1,n1}}]:float(), labels[{{},{1,n1}}])
        local batch_map_task2 = eval_task2:cal_mean_average_precision(outputs[{{},{n1+1,n1+n2}}]:float(), labels[{{},{n1+1,n1+n2}}])
        local batch_map_all = eval_all:cal_mean_average_precision(outputs:float(), labels)
        
        map_task1 = map_task1 + batch_map_task1
        map_task2 = map_task2 + batch_map_task2
        map_all = map_all + batch_map_all
    end    
    
    local loss = sum_loss/eval_iters
    local weight_norm, bias_norm = model_utils.cal_reg_loss(params, optim_config)
    local total_loss = loss + weight_norm * optim_config.weightDecay
    
    print (' ==> eval loss (loss, weight_norm, bias_norm, total loss) = ', opt.loss_weight*loss, weight_norm, bias_norm, 
        opt.loss_weight*total_loss)
    print (' ==> eval map (task1, task2, all) = ', map_task1/eval_iters, map_task2/eval_iters, map_all/eval_iters)
    
    print('-------------- Task 1 -------------- ')
    eval_task1:print_precision_recall()
    print('-------------- Task 2 -------------- ')
    eval_task2:print_precision_recall()
    print('-------------- All -------------- ')
    eval_all:print_precision_recall()
    
    model:training() -- back to the training mode
    return total_loss
end

-- callback function
local function feval(x)
    if x ~= params then params:copy(x) end
    grad_params:zero()
    
    local data1 = train_loader_task1:getBatch() -- get image and label batches
    local data2 = train_loader_task2:getBatch(true) -- get label only
    
    local images = data1.images:cuda()
    local labels = torch.cat(data1.labels, data2.labels, 2):cuda()
    
    local outputs = model:forward(images)
    local loss, tmp_df_do
    local df_do = torch.Tensor(opt.batch_size, opt.num_target):zero():cuda()
    
    if opt.multitask_type == 1 then
        loss = criterion:forward(outputs, labels)
        tmp_df_do = criterion:backward(outputs, labels)
        df_do:copy(tmp_df_do)
        
    elseif opt.multitask_type == 2 then -- alternative learning
        if iter % 2 == 1 then
            loss = criterion:forward(outputs[{{},{1,n1}}], labels[{{},{1,n1}}])
            tmp_df_do = criterion:backward(outputs[{{},{1,n1}}], labels[{{},{1,n1}}])
            df_do[{{},{1,n1}}]:copy(tmp_df_do)
        else
            loss = criterion:forward(outputs[{{},{n1+1,n1+n2}}], labels[{{},{n1+1,n1+n2}}])
            tmp_df_do = criterion:backward(outputs[{{},{n1+1,n1+n2}}], labels[{{},{n1+1,n1+n2}}])
            df_do[{{},{n1+1,n1+n2}}]:copy(tmp_df_do)
        end
    else
        error('Unknown multitask type', opt.multitask_type)
    end
    
    model:backward(images, df_do)
    
    return loss
end

-- Save model
local function save_model()
    local cp_path = path.join(opt.cp_path, 'model_' .. opt.model_id .. '_epoch' .. epoch  .. '.t7')
    local cp = {}
    cp.opt = opt
    cp.iter = iter
    cp.loss_history = loss_history
    cp.val_loss_history = val_loss_history
    cp.params = params
    
    -- saving vocabulary
    local vocab_path_task1 = paths.concat(opt.coco_data_root, opt.vocab_file_task1)
    local vocab_path_task2 = paths.concat(opt.coco_data_root, opt.vocab_file_task2)
    if paths.filep(vocab_path_task1) and paths.filep(vocab_path_task2) then
        local fh = io.open(vocab_path_task1, 'r')
        local json_text = fh:read()
        fh:close()
        local vocab1 = cjson.decode(json_text)
        
        fh = io.open(vocab_path_task2, 'r')
        json_text = fh:read()
        fh:close()
        local vocab2 = cjson.decode(json_text)
        
        -- append vocab 2 to vocab 1
        for k,v in ipairs(vocab2) do 
            table.insert(vocab1, v)
        end
        
        cp.vocab = vocab1
    else
        print('*** Warning ***: Vocab files not found! ', opt.vocab_path_task1, opt.vocab_path_task2)
    end

    print('Saving checkpoint to', cp_path)
    torch.save(cp_path, cp)
end

-- First evaluation
-- eval_loss()

model:training()
local timer = torch.Timer()

-- Training
while true do 

    iter = iter + 1
    timer:reset()
    
    -- Call forward/backward with full params input
    local loss = feval(params)

    -- Now update params acordingly
    if opt.optim == 'sgd' then
        optim_utils.sgd(params, grad_params, optim_config)
    elseif opt.optim == 'adam' then
        optim_utils.adam(params, grad_params, optim_config)   
    elseif opt.optim == 'adaml21' then
        optim_utils.adam_l21(params, grad_params, optim_config)       
    else
        error('Unknow optimization method', opt.optim)
    end
    
    if iter % opt.print_log_interval == 0 or iter == 1 then 
        local elapsed_time = timer:time().real
        local weight_norm, bias_norm = model_utils.cal_reg_loss(params, optim_config)
        local total_loss = loss + weight_norm * optim_config.weightDecay
        loss_history[iter] = opt.loss_weight*total_loss
        
        print(string.format('%s: iter %d, lr = %g, loss = %f, weight_norm = %f, bias_norm = %f, total loss = %f (%.3fs/iter)', 
                os.date(), iter, optim_config.learningRate, 
                opt.loss_weight*loss, weight_norm, bias_norm,
                opt.loss_weight*total_loss, 
                elapsed_time))
    end
   
    -- test loss
    if (iter % opt.test_interval == 0) then
        local val_loss = eval_loss()
        val_loss_history[iter] = opt.loss_weight*val_loss
        collectgarbage()
    end
    
    -- Learning rate decay for SGD
    if opt.optim == 'sgd' and iter % opt.learning_rate_decay_interval == 0 then
        optim_config.learningRate = optim_config.learningRate * opt.gamma_factor
        print('new learning rate', optim_config.learningRate)
    end
    
    if (iter % opt.save_cp_interval == 0) then save_model() end
    
    if iter % opt.iter_per_epoch == 0 then epoch = epoch + 1 end
    
    if iter >= opt.max_iters or epoch >= opt.max_epochs then break end
end    



