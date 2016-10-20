--[[
An interactive script that loads a depnet model running, 
and waiting for input, which is an image path and spcecified layers,
and return the image features from that layers
]]--

require 'nn'
require 'cudnn'
require 'image'
-- require 'logging.console'

local cjson = require 'cjson'
local model_utils = require 'model_utils'

io.stdout:setvbuf 'no'
torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-model_path', '', 'name of the checkpoint to test')
cmd:option('-debug', 0, '1 to turn debug on') 

cmd:text()
local opt = cmd:parse(arg)

-- local logger = logging.console()

if opt.debug == 1 then dbg = require 'debugger' end

function preprocess(img, img_size)
    local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68}) -- BGR
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    
    -- img = img:transpose(2,3)-- nChannel x height x width --> -- nChannel x width x height
    img = image.scale(img, img_size) 
    local width = img:size(2)
    local height = img:size(3)
    local out = torch.FloatTensor(3, img_size, img_size):zero()
    out[{{}, {1,width}, {1,height}}] = img
    return out
end

-- print(opt)
-- logger:info('Loading depnet model from checkpoint: ' .. opt.model_path)
io.stderr:write('Loading depnet model from checkpoint: ' .. opt.model_path .. '\n')
local model, model_info = model_utils.load_depnet_model(opt.model_path)
model:cuda()

-- logger:info('-- Model type: ' .. model_info.model_type)
-- logger:info('-- Image size: ' .. model_info.img_size)
io.stderr:write('-- Model type: ' .. model_info.model_type .. '\n')
io.stderr:write('-- Image size: ' .. model_info.img_size .. '\n')

-- logger:info('Turn evaluation mode on')
io.stderr:write('Turn evaluation mode on' .. '\n')
model:evaluate()

local timer = torch.Timer()

--[[
Load image and do the pre-processing
--]]
local function load_image(img_path, img_size)
    local img = image.load(img_path, 3, 'byte'):float()
    img = preprocess(img, img_size)
    img = img:view(1, 3, img_size, img_size)
    return img
end

--[[
input format: 
{"filename": "../clcv/resources/corpora/Microsoft_COCO/images/COCO_val2014_000000029594.jpg", "layers":["fc6","fc7","fc8"]}
--]]
local function decode_input(json_input)
    local out = cjson.decode(json_input)
    return out
end

--[[
output format: 
{"fc6":[0,1.076,0.20], "fc7":[0,1.076,0.20],"fc8":[0,1.076,0.20]}
--]]
local function encode_output(output)
    local output_feat = {}
    for i = 1,output:size(1) do
        table.insert(output_feat, output[i])
    end
    return output_feat
end

--[[
Main loop
Read a path to an image on stdin and output the features to stdout
--]]
--logger:info('Waiting for a json input. Press Enter to stop!')
io.stderr:write('Waiting for a json input. Press Enter to stop!\n')

while true do
    local input_text = io.read("*line")
    if string.len(input_text)==0 then break end
    
    local decode_status, input_json = pcall(decode_input, input_text)
    
    if decode_status then
        local img_path = input_json.filename
        local load_status, img = pcall(load_image, img_path, model_info.img_size)
        
        if load_status then
            model:forward(img:cuda())
            
            local layers = input_json.layers or {}
            local top_concepts = input_json.top_concepts or 20
            local output_json = {}
            
            for ii=1,#layers do
                layer = layers[ii]
                if model[layer] then
                    local output = model[layer]
		     if model_info.model_type == 'milmaxnor' then
			if layer == 'fc8' then 
		             -- use milmaxnor (that picks the largest value between the max and its noisy-or propability)
		             output =  nn.SpatialMIL('milmaxnor'):cuda():forward(output)
			elseif layer == 'responsemapfc8' then
		             local output_mil =  nn.SpatialMIL('milmaxnor'):cuda():forward(output)
        		     -- outputs should be from the prob layer, select top 
			     local sorted_outputs, sorted_indices = torch.topk(output_mil, top_concepts, 2, true, true)
			     local expanded_indices = sorted_indices:reshape(1, top_concepts, 1, 1):repeatTensor(1, 1, 12, 12)
        
       			     -- the repsonsemaps output 
			     local top_index = sorted_indices:view(-1)
			     local top_output = output:gather(2, expanded_indices):view(top_concepts, -1)
			     output = {top_index, top_output}
			else
		             -- use milmax (that only picks the max value) for the remaining layers (fc6, fc7, etc)
		             output = nn.SpatialMIL('milmax'):cuda():forward(output)
			end
                    end

		   if layer == 'responsemapfc8' then
			local output_featmap = {}
			output_featmap['index'] = encode_output(output[1])
			output_featmap['scores'] = {}
			for jj=1,output[1]:size(1) do
				local feat_jj = output[2][jj]
				output_featmap['scores'][jj] = encode_output(feat_jj)
			end
                        output_json[layer] = output_featmap
	           else
	                output = output:squeeze():view(-1)
                        local output_feat = encode_output(output)
                        output_json[layer] = output_feat
	    	   end
                else
		   -- logger:error('Unknown layer: ' .. layer)
		    output_json[layer] = 'Unknown layer'
                end
            end

            local output_txt = cjson.encode(output_json)
            io.write(output_txt .. '\n')
            io.stdout:flush()
        else
            -- logger:error('Error while loading image: ' .. img_path)
	   io.write('{\"status\": \"Error while loading image\"}\n')
	   io.stdout:flush()
        end
    else
       -- logger:error('Input text is not in json format')
       io.write('{"status": "Input text is not in json format"}\n')
       io.stdout:flush()
    end
end

-- logger:info('Elappsed time: ' .. timer:time().real .. '(s)' )
-- logger:info('Bye')
io.stderr:write('Elappsed time: ' .. timer:time().real .. '(s)\n')
io.stderr:write('Bye\n')

