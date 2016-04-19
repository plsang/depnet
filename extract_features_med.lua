--[[
Class to train
]]--

require 'nn'
require 'cudnn'
require "logging.console"
require 'hdf5'
require 'CocoData'

local model_utils = require 'model_utils'
local cjson = require 'cjson'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-video_root', '/net/per610a/export/das11f/plsang/trecvidmed/preprocessed', 'path to preprocessed data root')
cmd:option('-input_jsonl', '/net/per610a/export/das11f/plsang/trecvidmed/metadata/med_videos.jsonl', 'path to list of video')
cmd:option('-num_target', -1, 'Number of target concepts, -1 for getting from file')
cmd:option('-num_test_video', -1, 'Number of test video, -1 for testing all (40504)')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')
cmd:option('-test_cp', '', 'name of the checkpoint to test')
cmd:option('-cp_path', 'cp', 'path to save checkpoints')
cmd:option('-phase', 'test', 'phase (train/test)')
cmd:option('-log_mode', 'console', 'console/file.  filename is the testing model file + .log')
cmd:option('-log_dir', 'log', 'log dir')
cmd:option('-version', '', 'release version')    
cmd:option('-debug', 0, '1 to turn debug on')    
cmd:option('-model_type', 'milmaxnor', 'vgg, vggbn, milmax, milnor, milmaxnor')
cmd:option('-layer', 'fc8', 'fc8, fc7')
cmd:option('-med_root', 'data/MED', 'path to coco data root')
cmd:option('-output_file', '', 'path to output file')
cmd:option('-start_video', 1, 'index of the start video to extract')
cmd:option('-end_video', -1, 'index of the end video to extract')


cmd:text()
local opt = cmd:parse(arg)

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function read_all_lines(file)
    if not paths.filep(file) then return {} end
    local textlines = {}
    for line in io.lines(file) do 
        textlines[#textlines + 1] = line
    end
    return textlines
end

local logger = logging.console()

if opt.log_mode == 'file' then
    require 'logging.file'
    local dirname = paths.concat(opt.log_dir, opt.version)
    if not paths.dirp(dirname) then paths.mkdir(dirname) end
    local filename = paths.basename(opt.test_cp, 't7')
    local logfile = paths.concat(dirname, 'test_' .. filename .. '.log')
    logger = logging.file(logfile)
end

if opt.output_file == '' then
    local dirname = paths.concat(opt.med_root, opt.version)
    if not paths.dirp(dirname) then paths.mkdir(dirname) end
    local filename = paths.basename(opt.test_cp, 't7')
    opt.output_file = paths.concat(dirname, filename .. '_med_' .. opt.layer .. '.h5')
end

if opt.debug == 1 then dbg = require 'debugger' end

logger:info('Loading video info: ' .. opt.input_jsonl)

local textlines = read_all_lines(opt.input_jsonl)
local videos = {}
for i, line in ipairs(textlines) do
    local info = cjson.decode(line)
    videos[#videos+1] = info
end


if opt.start_video < 1 then opt.start_video = 1 end
if opt.end_video == -1 or opt.end_video > #videos  then opt.end_video = #videos end
if opt.num_test_video == -1 then opt.num_test_video = opt.end_video - opt.start_video + 1 end

print(opt)

logger:info('Total videos: ' .. #videos)
logger:info('Number of testing videos: ' .. opt.num_test_video)
logger:info('Number of labels: ' .. opt.num_target)
logger:info('Model type: ' .. opt.model_type)
logger:info('Loading model: ' .. opt.test_cp)

local model = model_utils.load_model(opt):cuda()

if opt.layer == 'fc7' or opt.layer == 'fc6' then
    model:remove()  -- remove MIL
    model['modules'][2]:remove()  -- remove sigmoid
    model['modules'][2]:remove()  -- remove fc8
    
    if opt.layer == 'fc6' then
        model['modules'][2]:remove()  -- remove dropout
        model['modules'][2]:remove()  -- remove relu
        model['modules'][2]:remove()  -- remove spatial convolution
    end
    
    if opt.model_type == 'milmaxnor' then
    	model:add(nn.SpatialMIL('milmaxnor'):cuda()) 
    end

    opt.num_target = 4096
    print(model['modules'][2])
end

model:evaluate() 

local myFile = hdf5.open(opt.output_file, 'w')

local timer = torch.Timer()

local index = torch.LongTensor(opt.num_test_video):zero()
local data = torch.FloatTensor(opt.num_test_video, opt.num_target):zero()

for ii=opt.start_video, opt.end_video do
    
    local image_file_h5 = paths.concat(opt.video_root, videos[ii]['location'] .. '.h5')
    if not paths.filep(image_file_h5) then 
        logger:info('Warning: file not found ' .. image_file_h5)
    else
        local loader = CocoData{image_file_h5 = image_file_h5, 
            images_key = '/data', noshuffle = true,
            num_target = opt.num_target,
            batch_size = opt.batch_size}

        local num_test_image = loader:getNumImages()
        
        video_id = videos[ii]['video_id']
        logger:info(string.format('[%d/%d] Extracting feature for video %s...', ii, opt.num_test_video, video_id))
        local num_iters = torch.ceil(num_test_image/opt.batch_size)

        local feats = torch.FloatTensor(num_test_image, opt.num_target):zero()
        
        for iter=1, num_iters do
            local batch = loader:getBatch() -- get image and label batches
            local outputs = model:forward(batch.images:cuda())

            local start_idx = (iter-1)*opt.batch_size + 1
            local end_idx = iter*opt.batch_size
            if end_idx > num_test_image then
                end_idx = num_test_image
            end
            
            feats[{{start_idx, end_idx},{}}] = outputs[{{1,end_idx-start_idx+1},{}}]:float()   -- copy cpu ==> gpu 
        end
        
        data[ii-opt.start_video+1] = torch.mean(feats, 1)
        index[ii-opt.start_video+1] = tonumber(string.sub(video_id, 4, #video_id))
       
        loader:close() 
        feats = nil
        collectgarbage()
    end
end

logger:info('Writing indices...')
myFile:write('/index', index, options)
logger:info('Writing data...')
myFile:write('/data', data, options)
myFile:close()
logger:info('Done. Elappsed time: ' .. timer:time().real .. '(s)' )



