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
cmd:option('-val_label_file_h5', 'mscoco2014_val_myconceptsv3.h5', 'file name of the prepressed val label data')
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
cmd:option('-version', 'v2.0', 'release version')    
cmd:option('-debug', 0, '1 to turn debug on')    
cmd:option('-print_log_interval', 10000, 'Number of test image.')
cmd:option('-model_type', 'vgg', 'vgg, vggbn, milmax, milnor')
cmd:option('-test_mode', 'model', 'model/file: test from a model or from a predicted file')

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

local loader = CocoData{image_file_h5 = opt.val_image_file_h5, 
    label_file_h5 = paths.concat(opt.coco_data_root, opt.val_label_file_h5),
    num_target = opt.num_target, 
    batch_size = opt.batch_size}

if opt.num_test_image == -1 then opt.num_test_image = loader:getNumImages() end
if opt.num_target == -1 then opt.num_target = loader:getNumTargets() end
assert(opt.batch_size == 1, 'batch_size > 1 is not supported!')
print(opt)

logger:info('Number of testing images: ' .. opt.num_test_image)
logger:info('Number of labels: ' .. opt.num_target)

logger:info('Logging model. Type: ' .. opt.model_type)

local model, criterion, output_loader
if opt.test_mode == 'model' then
    logger:info('Logging model. Type: ' .. opt.model_type)
    model = model_utils.load_model(opt):cuda()
    criterion = nn.MultilabelCrossEntropyCriterion(opt.loss_weight):cuda()
    model:evaluate()  
else
    output_loader = hdf5.open(opt.test_cp, 'r')
    local data_size = output_loader:read('/data'):dataspaceSize()
    assert(data_size[1] == opt.num_test_image, 'number of test images are not equal')
    assert(data_size[2] == opt.num_target, 'number of targets are not equal')
end

local eval = eval_utils()
local num_iters = torch.ceil(opt.num_test_image/opt.batch_size)

local map = 0
local outputs
local idx = 1

for iter=1, num_iters do
    local data = loader:getBatch()
    
    if opt.test_mode == 'model' then
        outputs = model:forward(data.images:cuda())
    else
        local index = output_loader:read('/index'):partial({idx, idx+opt.batch_size-1})
        assert(torch.all(torch.eq(index, data.image_ids)), 'error: image ids are not matched!!!')
        outputs = output_loader:read('/data'):partial({idx, idx+opt.batch_size-1}, {1, opt.num_target})
        
        idx = idx+opt.batch_size    
    end
    
    outputs = outputs:float()
    
    local batch_map = eval:cal_mean_average_precision(outputs, data.labels)
    map = map + batch_map
    eval:cal_precision_recall(outputs, data.labels)
  
    if iter % opt.print_log_interval == 0 then 
        logger:info(string.format('iter %d: map = %.6f', iter, map/iter))
        eval:print_precision_recall(logger)
        collectgarbage() 
    end
end    

logger:info('Final performanace:')
logger:info('==> map = ' .. map/num_iters)
eval:print_precision_recall(logger)    

