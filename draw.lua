--[[
Class to train
]]--

require 'nn'
require 'cudnn'
require "logging.console"
require 'gnuplot'
local cjson = require 'cjson'

require 'CocoData'
require 'eval_utils'
local model_utils = require 'model_utils'
require 'nn.MultiLabelCrossEntropyCriterion'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-coco_data_root', '/net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO', 'path to coco data root')
cmd:option('-val_image_file_h5', 'data/coco_val.h5', 'path to the prepressed image data')
cmd:option('-val_index_json', 'coco_val.json', 'path to the index json file')
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
cmd:option('-model_type', 'vgg', 'vgg, milmaxnor')
cmd:option('-test_mode', 'model', 'model/file: test from a model or from a predicted file')
cmd:option('-concept_idx', 1, 'Index of the concept to visualize')
cmd:option('-vocab_file', 'mscoco2014_train_captions_mydepsv4vocab.json', 'saving a copy of the vocabulary that was used for training')

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

local loader = CocoData{
    image_file_h5 = paths.concat(opt.coco_data_root, opt.val_image_file_h5),
    index_json = paths.concat(opt.coco_data_root, opt.val_index_json), 
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
local model = model_utils.load_model(opt):cuda()
model:evaluate()  

local function draw(matrix, output_file)
    gnuplot.pngfigure(output_file)
    gnuplot.imagesc(torch.sin(matrix),'color')
    gnuplot.plotflush()
end

local num_iters = torch.ceil(opt.num_test_image/opt.batch_size)

local map = 0
local outputs
local idx = 1
local img_size = 224
if opt.model_type == 'milmaxnor' then img_size = 565 end

-- saving vocabulary
local vocab_path = paths.concat(opt.coco_data_root, opt.vocab_file)
if paths.filep(vocab_path) then
    local fh = io.open(vocab_path, 'r')
    local json_text = fh:read()
    fh:close()
    vocab = cjson.decode(json_text)
else
    print('*** Warning ***: Vocab file not found! ', opt.vocab_path)
end

for iter=1, num_iters do
    local data = loader:getBatch()
    
    num_range = 10
    range = torch.linspace(1, img_size, num_range+1)
    local img_id = loader:getIndex(iter)

    -- whole image to get index of max concept 
    local outputs = model:forward(data.images:cuda())
    
    print('img_id, max concept', img_id, vocab[opt.concept_idx])

    local nel = num_range*num_range
    local img_data = torch.zeros(nel, 3, img_size, img_size)
    
    for ii=1,num_range do
        for jj=1,num_range do
            ii_start = math.floor(range[ii])
            ii_end = math.floor(range[ii+1])
            jj_start = math.floor(range[jj])
            jj_end = math.floor(range[jj+1])
            local eind = (ii-1)*num_range + jj 
            img_data[eind] = data.images:clone()
            img_data[{{eind},{},{ii_start,ii_end},{jj_start,jj_end}}] = 0
        end
    end

    local outputs = torch.zeros(nel)
    local nbatch = 16
    local niters = torch.ceil(nel/nbatch)
    
    for niter = 1,niters do
        local start_idx = (niter-1)*nbatch + 1
        local end_idx = start_idx + nbatch - 1
        if end_idx > nel then end_idx = nel end
        local tmp_outputs = model:forward(img_data[{{start_idx,end_idx}}]:cuda())
        outputs[{{start_idx,end_idx}}] = tmp_outputs[{{},{opt.concept_idx}}]:float()
    end
    local map = outputs:resize(num_range, num_range)
    
    img_name = 'COCO_val2014_' .. string.format("%012d", img_id) 
    draw(map, 'data/map/' .. img_name .. '.png')

    src_file = '/net/per610a/export/das11f/plsang/coco2014/images/val2014/' .. img_name .. '.jpg'
    os.execute('cp ' .. src_file .. ' ' .. 'data/src/' )    
    
    if iter % opt.print_log_interval == 0 then 
        logger:info(string.format('iter %d: ', iter))
        collectgarbage() 
    end
end    

