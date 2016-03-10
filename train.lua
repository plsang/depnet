--[[
Class to train
]]--

require 'nn'
require 'cudnn'
local cjson = require 'cjson'

require 'CocoData'
require 'MultilabelCrossEntropyCriterion'
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
cmd:option('-train_label_file_h5', 'mscoco2014_train_myconceptsv3.h5', 'file name of the prepressed train label data')
cmd:option('-val_label_file_h5', 'mscoco2014_val_myconceptsv3.h5', 'file name of the prepressed val label data')
cmd:option('-vocab_file', 'mscoco2014_train_myconceptsv3vocab.json', 'saving a copy of the vocabulary that was used for training')
cmd:option('-num_target', 1000, 'Number of target concepts')
cmd:option('-num_test_image', 1600, 'Number of test image.')
cmd:option('-test_interval', 10000, 'Number of test image.')
cmd:option('-print_log_interval', 20, 'Number of test image.')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-max_iters', 1000000)
cmd:option('-save_cp_interval', 0, 'to save a check point every interval number of iterations')
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
cmd:option('-optim', 'adam', 'optimization method: sgd, adam')
cmd:option('-learning_rate', 1e-5, 'learning rate for sgd') -- msmil: 0.000015625
cmd:option('-model_type', 'vgg', 'vgg, vggbn, milmax, milnor')
cmd:option('-finetune_layer_name', 'fc8', 'name of the finetuning layer')
cmd:option('-debug', 0, 'turn debug mode on/off')
-- these options are for SGD
cmd:option('-learning_rate_decay', 0, 'decaying rate for sgd')
cmd:option('-gamma_factor', 0.1, 'factor to reduce learning rate, 0.1 ==> drop 10 times')
cmd:option('-learning_rate_decay_interval', 80000, 'learning rate for sgd')
cmd:option('-momentum', 0.99, 'momentum for sgd')
cmd:option('-weight_decay', 0.0005, 'momentum for sgd')
-- these options are for Adam
cmd:option('-adam_beta1', 0.9, 'momentum for adam')
cmd:option('-adam_beta2', 0.999, 'momentum for adam')
cmd:option('-adam_epsilon', 1e-8, 'momentum for epsilon')
--

cmd:text()
local opt = cmd:parse(arg)

if opt.model_id == '' then 
    -- opt.model_id = ('%s_b%d_lr%f').format(opt.optim, opt.batch_size, opt.learning_rate)
    opt.model_id = string.format('%s_%s_b%d_lr%f', opt.optim, opt.model_type, opt.batch_size, opt.learning_rate)
end

if opt.debug == 1 then dbg = require 'debugger' end

-- set the manual seed
torch.manualSeed(opt.seed)

--- loading Coco data
local train_loader = CocoData{image_file_h5 = opt.train_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.train_label_file_h5), 
    num_target = opt.num_target, 
    batch_size = opt.batch_size}

local val_loader = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5),
    num_target = opt.num_target, 
    batch_size = opt.batch_size}

if opt.save_cp_interval == 0 then 
    opt.save_cp_interval = math.ceil(train_loader:getNumImages()/opt.batch_size)
end

-- 
print(opt)

local eval = eval_utils()

--
local model = model_utils.load_model(opt):cuda()

local criterion = nn.MultilabelCrossEntropyCriterion(opt.loss_weight):cuda()
print(model.modules)

-- Initialization
model_utils.init_finetuning_params(model, opt)

local params, grad_params = model:getParameters()
print('total number of parameters: ', params:nElement(), grad_params:nElement())

-- note: don't use 'config' as a variable name
local optim_config = {
    learningRate = opt.learning_rate,
    w_lr_mult = opt.w_lr_mult,
    b_lr_mult = opt.b_lr_mult,
    ft_lr_mult = opt.ft_lr_mult  -- if w and b have the same learning rate
}

if opt.optim == 'sgd' then
    optim_config.weightDecay = opt.weight_decay
    optim_config.momentum = opt.momentum
    optim_config.learningRateDecay = opt.learning_rate_decay
elseif opt.optim == 'adam' then
    optim_config.adam_beta1 = opt.adam_beta1
    optim_config.adam_beta2 = opt.adam_beta2
    optim_config.adam_epsilon = opt.adam_epsilon
else
    error('Unknow optimization method', opt.optim)
end

-- update param indices from model 
model_utils.update_param_indices(model, opt, optim_config)

print('Optimization configurations', optim_config) 

local function eval_loss()
    model:evaluate()
    val_loader:reset() -- reset interator
    eval:reset()
    
    print(' ==> evaluating ...') 
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
    
    local loss = opt.loss_weight*total_loss/eval_iters
    print (' ==> eval loss = ', loss)
    print (' ==> eval map = ', map/eval_iters)
    
    eval:print_precision_recall()
    model:training() -- back to the training mode
    return loss
end

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

-- MAIN LOOP --- 
local iter = 0
local loss_history = {}
local val_loss_history = {}

-- Save model
local function save_model()
    local cp_path = path.join(opt.cp_path, 'model_' .. opt.model_id .. '_iter' .. iter)
    local cp = {}
    cp.opt = opt
    cp.iter = iter
    cp.loss_history = loss_history
    cp.val_loss_history = val_loss_history
    cp.params = params
    -- saving vocabulary
    local vocab_path = paths.concat(opt.coco_data_root, opt.vocab_file)
    if paths.filep(vocab_path) then
        local fh = io.open(vocab_path, 'r')
        local json_text = fh:read()
        fh:close()
        local vocab = cjson.decode(json_text)
        cp.vocab = vocab
    else
        print('*** Warning ***: Vocab file not found! ', opt.vocab_path)
    end

    print('Saving checkpoint to', cp_path)
    torch.save(cp_path .. '.t7', cp)
end

-- First evaluation
-- eval_loss()

model:training()

-- Training
while true do 

    iter = iter + 1
    local timer = torch.Timer() 
    
    -- Call forward/backward with full params input
    local loss = feval(params)
   
     -- Now update params acordingly
    if opt.optim == 'sgd' then
        optim_utils.sgd(params, grad_params, optim_config)
    elseif opt.optim == 'adam' then
        optim_utils.adam(params, grad_params, optim_config)   
    else
        error('Unknow optimization method', opt.optim)
    end
    
    if iter % opt.print_log_interval == 0 then 
        loss_history[iter] = loss
        print(string.format('%s: iter %d, loss = %f, lr = %g (%.3fs/iter)', 
                os.date(), iter, opt.loss_weight*loss, optim_config.learningRate, timer:time().real))
        collectgarbage() 
    end
   
    -- test loss
    if (iter % opt.test_interval == 0) then
        local val_loss = eval_loss()
        val_loss_history[iter] = val_loss
    end
    
    -- save checkpoints
    if (iter % opt.save_cp_interval == 0 or iter == opt.max_iters) then
        save_model()
    end

    -- Learning rate decay for SGD
    if opt.optim == 'sgd' and iter % opt.learning_rate_decay_interval == 0 then
        config.learningRate = config.learningRate * opt.gamma_factor
        print('new learning rate', config.learningRate)
    end
    
    -- Break condition
    if iter >= opt.max_iters then 
        break 
    end
end    



