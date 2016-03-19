--[[
Class to train
]]--

require 'nn'
require 'cudnn'
require "logging.console"

require 'CocoData'
require 'MultilabelCrossEntropyCriterion'
require 'eval_utils'
local model_utils = require 'model_utils'
    
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-coco_data_root', '/net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO', 'path to coco data root')
cmd:option('-val_image_file_h5', 'data/coco_val.h5', 'path to the prepressed image data')

cmd:option('-val_label_file_h5_task1', 'mscoco2014_val_myconceptsv3.h5', 'file name of the prepressed val label data')
cmd:option('-val_label_file_h5_task2', 'mscoco2014_val_mydepsv4.h5', 'file name of the prepressed val label data')

cmd:option('-num_target', -1, 'Number of target concepts, -1 for getting from file')
cmd:option('-num_test_image', -1, 'Number of test image, -1 for testing all (40504)')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-test_cp', '', 'name of the checkpoint to test')
cmd:option('-cp_path', 'cp', 'path to save checkpoints')
cmd:option('-loss_weight', 20, 'loss multiplier, to display loss as a bigger value, and to scale backward gradient')
cmd:option('-phase', 'test', 'phase (train/test)')
cmd:option('-log_mode', 'console', 'console/file.  filename is the testing model file + .log')
cmd:option('-log_dir', 'log', 'log dir')
cmd:option('-version', 'v1.4', 'release version')    
cmd:option('-debug', 0, '1 to turn debug on')    
cmd:option('-print_log_interval', 1000, 'Number of test image.')
cmd:option('-model_type', 'vgg', 'vgg, vggbn, milmax, milnor')


cmd:text()
local opt = cmd:parse(arg)

local logger = logging.console()

if opt.log_mode == 'file' then
    require 'logging.file'
    local dirname = paths.concat(opt.log_dir, opt.version)
    if not paths.dirp(dirname) then paths.mkdir(dirname) end
    local filename = paths.basename(opt.test_cp, 't7')
    local logfile = paths.concat(dirname, 'test_' .. filename .. '.log')
    logger = logging.file(logfile)
end

if opt.debug == 1 then dbg = require 'debugger' end

local loader_task1 = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5_task1),
    batch_size = opt.batch_size}
local loader_task2 = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5_task2),
    batch_size = opt.batch_size}

if opt.num_test_image == -1 then opt.num_test_image = loader_task1:getNumImages() end
if opt.num_target == -1 then opt.num_target = loader_task1:getNumTargets() + loader_task2:getNumTargets() end
print(opt)

logger:info('Number of testing images: ' .. opt.num_test_image)
logger:info('Number of labels: ' .. opt.num_target)

logger:info('Logging model. Type: ' .. opt.model_type)
local model = model_utils.load_model(opt):cuda()
local criterion = nn.MultilabelCrossEntropyCriterion(opt.loss_weight):cuda()
model:evaluate()  

local num_iters = torch.ceil(opt.num_test_image/opt.batch_size)

local eval_task1 = eval_utils()
local eval_task2 = eval_utils()
local eval_all = eval_utils()

local map_task1 = 0
local map_task2 = 0
local map_all = 0

local n1 = loader_task1:getNumTargets()
local n2 = loader_task2:getNumTargets()

function print_eval_log()
    logger:info('-------------- Task 1 -------------- ')
    eval_task1:print_precision_recall(logger)
    logger:info('-------------- Task 2 -------------- ')
    eval_task2:print_precision_recall(logger)
    logger:info('-------------- All -------------- ')
    eval_all:print_precision_recall(logger)
end

for iter=1, num_iters do
    
    local data1 = loader_task1:getBatch() -- get image and label batches
    local data2 = loader_task2:getBatch(true) -- get label only
    local images = data1.images:cuda()
    local labels = torch.cat(data1.labels, data2.labels, 2)
    
    local outputs = model:forward(images)
    local iter_loss = criterion:forward(outputs, labels:cuda())
    
    eval_task1:cal_precision_recall(outputs[{{},{1,n1}}], labels[{{},{1,n1}}])
    eval_task2:cal_precision_recall(outputs[{{},{n1+1,n1+n2}}], labels[{{},{n1+1,n1+n2}}])
    eval_all:cal_precision_recall(outputs, labels)

    local batch_map_task1 = eval_task1:cal_mean_average_precision(outputs[{{},{1,n1}}]:float(), labels[{{},{1,n1}}])
    local batch_map_task2 = eval_task2:cal_mean_average_precision(outputs[{{},{n1+1,n1+n2}}]:float(), labels[{{},{n1+1,n1+n2}}])
    local batch_map_all = eval_all:cal_mean_average_precision(outputs:float(), labels)

    map_task1 = map_task1 + batch_map_task1
    map_task2 = map_task2 + batch_map_task2
    map_all = map_all + batch_map_all
    
    if iter % opt.print_log_interval == 0 then 
        logger:info(string.format('iter %d: iter_loss = %.6f, map_task1 = %.6f, map_task2 = %.6f, map_all = %.6f', 
                iter, opt.loss_weight*iter_loss, map_task1/iter, map_task2/iter, map_all/iter))
        print_eval_log()
        collectgarbage() 
    end
end    

logger:info('Final performanace:')
print (string.format(' ==> map (task1, task2, all) = (%.6f, %.6f, %.6f)', 
        map_task1/num_iters, map_task2/num_iters, map_all/num_iters))
print_eval_log()


