--[[
Class to train
]]--

require 'nn'
require 'cudnn'
local cjson = require 'cjson'

require 'VideoData'
require 'nn.MultiLabelCrossEntropyCriterion'
require 'eval_utils'

local model_utils = require 'model_utils'
local optim_utils = require 'optim_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-coco_data_root', '/home/ec2-user/data/Microsoft_COCO', 'path to coco data root')
cmd:option('-val_image_file_h5_s', 'coco_val.h5', 'path to the prepressed image data')
cmd:option('-val_index_json_s', 'coco_val.json', 'path to the index json file')
cmd:option('-val_image_file_h5_t', 'coco_val.h5', 'path to the prepressed image data')
cmd:option('-val_index_json_t', 'coco_val.json', 'path to the index json file')
cmd:option('-train_label_file_h5', 'mscoco2014_train_myconceptsv3.h5', 'file name of the prepressed train label data')
cmd:option('-val_label_file_h5', 'mscoco2014_val_myconceptsv3.h5', 'file name of the prepressed val label data')
cmd:option('-vocab_file', 'mscoco2014_train_myconceptsv3vocab.json', 'saving a copy of the vocabulary that was used for training')
cmd:option('-concept_type', '', 'name of concept type, e.g., myconceptsv3, mydepsv4, empty for auto detect from train_label_file_h5')
cmd:option('-num_target', -1, 'Number of target concepts, -1 for getting from file')
cmd:option('-num_test_image', -1, 'Number of test image, -1 for testing all (40504)')
cmd:option('-test_interval', -1, 'Number of test image.')
cmd:option('-print_log_interval', 20, 'Number of test image.')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-val_batch_size', 1, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-max_iters', 1000000)
cmd:option('-max_epochs', 10)
cmd:option('-save_cp_interval', -1, 'to save a check point every interval number of iterations')
cmd:option('-test_cp_s', '', 'name of the checkpoint to test')
cmd:option('-test_cp_t', '', 'name of the checkpoint to test')
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
cmd:option('-learning_rate', 1e-5, 'learning rate for optim') -- msmil: 0.000015625
cmd:option('-model_type', 'vgg', 'vgg, vggbn, milmax, milnor, milmaxnor')
cmd:option('-finetune_layer_name', 'fc8', 'name of the finetuning layer')
cmd:option('-debug', 0, 'turn debug mode on/off')
cmd:option('-reg_type', 2, '1: L1 regularization, 2: L2 regularization, 3: L2,1 regularization')
-- these options are for SGD
cmd:option('-learning_rate_decay', 0, 'decaying rate for sgd')
cmd:option('-gamma_factor', 0.1, 'factor to reduce learning rate, 0.1 ==> drop 10 times')
cmd:option('-learning_rate_decay_interval', -1, 'learning rate for sgd')
cmd:option('-momentum', 0.99, 'momentum for sgd')
-- these options are for Adam
cmd:option('-adam_beta1', 0.9, 'momentum for adam')
cmd:option('-adam_beta2', 0.999, 'momentum for adam')
cmd:option('-adam_epsilon', 1e-8, 'momentum for epsilon')
cmd:option('-weight_decay', 0, 'regularization multiplier')
cmd:option('-version', 'v0.0', 'release version')    
cmd:option('-num_img_channel', 3, 'number of input channels (3: spatial net, 20: temporal net)')
cmd:option('-fc6_dropout', 0.5, 'Dropout ratio in the fully connected layers (use 0.9 for the temporal net)')
cmd:option('-fc7_dropout', 0.5, 'Dropout ratio in the fully connected layers (use 0.9 for the temporal net)')
cmd:option('-num_val_frame_per_video', 25, 'number of frame to be evaluated per videos')
--

cmd:text()
local opt = cmd:parse(arg)

if opt.debug == 1 then dbg = require 'debugger' end

-- set the manual seed
torch.manualSeed(opt.seed)

local val_loader_s = VideoData{
    image_file_h5 = paths.concat(opt.coco_data_root, opt.val_image_file_h5_s), 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5),
    index_json = paths.concat(opt.coco_data_root, opt.val_index_json_s), 
    num_target = opt.num_target, 
    batch_size = opt.val_batch_size,
    num_img_channel = 3,
    num_val_frame_per_video = opt.num_val_frame_per_video,
    mode = 'test'
}

local val_loader_t = VideoData{
    image_file_h5 = paths.concat(opt.coco_data_root, opt.val_image_file_h5_t), 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5),
    index_json = paths.concat(opt.coco_data_root, opt.val_index_json_t), 
    num_target = opt.num_target, 
    batch_size = opt.val_batch_size,
    num_img_channel = 20,
    num_val_frame_per_video = opt.num_val_frame_per_video,
    mode = 'test'
}

-- Update some default options
if opt.num_target == -1 then opt.num_target = val_loader_s:getNumTargets() end
if opt.num_test_image == -1 then opt.num_test_image = val_loader_s:getNumVideos() end
if opt.concept_type == '' then opt.concept_type = string.split(paths.basename(opt.train_label_file_h5, '.h5'), '_')[3] end
if opt.model_id == '' then 
    opt.model_id = string.format('%s_%s_%s_b%d_bias%g_lr%g_wd%g_l%d', 
            opt.concept_type, opt.model_type,
            opt.optim, opt.batch_size, opt.bias_init, 
            opt.learning_rate, opt.weight_decay, opt.reg_type)
end

print(opt)
------------------------------------------

print('Loading spatial network: ' .. opt.test_cp_s)
local opt_s = opt
opt_s.test_cp = opt.test_cp_s
opt_s.num_img_channel = 3
local model_s = model_utils.load_model(opt_s):cuda()

print('Loading temporal network: ' .. opt.test_cp_t)
local opt_t = opt
opt_t.test_cp = opt.test_cp_t
opt_t.num_img_channel = 20
local model_t = model_utils.load_model(opt_t):cuda()

local eval = eval_utils()
local eval_s = eval_utils()
local eval_t = eval_utils()

local function get_vocab(vocab_path)
    local vocab
    if paths.filep(vocab_path) then
        local fh = io.open(vocab_path, 'r')
        local json_text = fh:read()
        fh:close()
        vocab = cjson.decode(json_text)
    else
        error('*** Warning ***: Vocab file not found! ', vocab_path)
    end
    return vocab
end

local function eval_loss()
    model_s:evaluate()
    model_t:evaluate()
    val_loader_s:reset() -- reset interator
    val_loader_t:reset() -- reset interator
    eval:reset()
    eval_s:reset()
    eval_t:reset()
    
    local vocab_path = paths.concat(opt.coco_data_root, opt.vocab_file)
    local vocab = get_vocab(vocab_path)
    
    print(' ==> evaluating ...') 
    local eval_iters = opt.num_test_image
    local sum_loss = 0
    local map = 0
    local map_s = 0
    local map_t = 0
    for iter=1, eval_iters do
        local data_s = val_loader_s:getBatch()
        local data_t = val_loader_t:getBatch()
        assert(data_s.ids[1] == data_t.ids[1])
        
        local outputs_s = model_s:forward(data_s.images:cuda()):mean(1)
        local outputs_t = model_t:forward(data_t.images:cuda()):mean(1)
        local outputs = torch.add(outputs_s, outputs_t):div(2)
        
        eval:cal_precision_recall(outputs, data_s.labels)
        local batch_map = eval:cal_mean_average_precision(outputs:float(), data_s.labels)
        local batch_map_s = eval_s:cal_mean_average_precision(outputs_s:float(), data_s.labels)
        local batch_map_t = eval_t:cal_mean_average_precision(outputs_t:float(), data_t.labels)
        
        map = map + batch_map
        map_s = map_s + batch_map_s
        map_t = map_t + batch_map_t
        
        -- 
        local idx = torch.nonzero(data_s.labels[1]):view(-1)
        local gold_concept = ''
        for ii=1,idx:size(1) do
            gold_concept = gold_concept .. ' ' .. vocab[idx[ii]]
        end
        print(' ## GOLD: ' .. gold_concept)
        
        local _, idx = torch.sort(outputs_s[1], 1, true)
        local pred_concept = ''
        for ii=1,15 do
            pred_concept = pred_concept .. ' ' .. vocab[idx[ii]]
        end
        print(' ## SPATIAL PRED: ' .. pred_concept)
        
        local _, idx = torch.sort(outputs_t[1], 1, true)
        local pred_concept = ''
        for ii=1,15 do
            pred_concept = pred_concept .. ' ' .. vocab[idx[ii]]
        end
        print(' ## TEMPORAL PRED: ' .. pred_concept)
        
        local _, idx = torch.sort(outputs[1], 1, true)
        local pred_concept = ''
        for ii=1,15 do
            pred_concept = pred_concept .. ' ' .. vocab[idx[ii]]
        end
        print(' ## SPATIAL-TEMPORAL PRED: ' .. pred_concept)
        
        print(string.format('===> [%d/%d] Video_id %d, map_s = %f, map_t = %f, map = %f ', 
                iter, eval_iters, data_s.ids[1], batch_map_s, batch_map_t, batch_map))
        print('--------------------------------------------------------')
        
        collectgarbage()
    end    
    
    print (string.format(' FINAL: map_s = %f, map_t = %f, map = %f', map_s/eval_iters, map_t/eval_iters, map/eval_iters))
        
    eval:print_precision_recall()
end

---- MAIN ----

local timer = torch.Timer() 
eval_loss()
local elapsed_time = timer:time().real
print('Done in ' .. elapsed_time)



