require 'nn'
require 'cunn' -- otherwise, error: attempt to index field 'THNN' (a nil value)
require 'cudnn'
require 'loadcaffe'
require 'MultilabelNLLCriterion'

local model_utils = {}

function model_utils.load_vgg(opt)
    print('Loading...')
    vgg_model = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end)

    ---
    print('Removing the softmax layer')
    table.remove(vgg_model['modules'])  -- equivalent to pop
    table.insert(vgg_model['modules'], nn.LogSoftMax():cuda())
    
    -- print(vgg_model['modules'])
    
    -- for k,v in pairs(vgg_model) do print(k,v) end
    -- following code will change the parameter of the model
    -- print('checking model by sampling temp input...')
    -- print(#vgg_model:cuda():forward(torch.CudaTensor(1, 3, 224, 224)))

    ---
    print('converting first layer conv filters from BGR to RGB...')
    local input_layer = vgg_model:get(1)
    local w = input_layer.weight:clone()
    -- swap weights to R and B channels
    input_layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
    input_layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    
    criterion = nn.MultilabelNLLCriterion():cuda()
    
    return vgg_model, criterion
end

-- Define the model, then copy parameters
function model_utils.define_vgg(opt)
    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv1_1'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu1_1'
    model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv1_2'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu1_2'
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool1'
    model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv2_1'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu2_1'
    model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv2_2'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu2_2'
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool2'
    model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv3_1'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu3_1'
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv3_2'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu3_2'
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv3_3'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu3_3'
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool3'
    model:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv4_1'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu4_1'
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv4_2'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu4_2'
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv4_3'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu4_3'
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool4'
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv5_1'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu5_1'
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv5_2'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu5_2'
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    model:get(#model).name = 'conv5_3'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu5_3'
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool5'
    model:add(nn.View(-1):setNumInputDims(3))
    model:get(#model).name = 'torch_view'
    model:add(nn.Linear(25088, 4096))
    model:get(#model).name = 'fc6'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu6'
    model:add(nn.Dropout(0.500000))
    model:get(#model).name = 'drop6'
    model:add(nn.Linear(4096, 4096))
    model:get(#model).name = 'fc7'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu7'
    model:add(nn.Dropout(0.500000))
    model:get(#model).name = 'drop7'
    model:add(nn.Linear(4096, 1000))
    model:get(#model).name = 'fc8'
    model:add(nn.LogSoftMax())
    model:get(#model).name = 'logsoftmax'
    
    cudnn.convert(model, cudnn)
    -- convert data to CudaTensor
    for i, m in ipairs(model.modules) do
        m:cuda()
    end
    
    -- for k,v in pairs(model) do print (k,v) end
    
    local model_vgg = model_utils.load_vgg(opt)
    local parameters = model:getParameters() -- all params as one flat variables
    local parameters_vgg = model_vgg:getParameters() -- all params as one flat variables
    
    assert(parameters:nElement() == parameters_vgg:nElement())
    parameters:copy(parameters_vgg)
    parameters_vgg = nil
    model_vgg = nil
    collectgarbage() 
    
    -- freeze conv1_1,  conv1_2,  conv2_1,  conv2_2
    local count = 0
    for i, m in ipairs(model.modules) do
        if count == 4 then break end
        if torch.type(m):find('Convolution') then
            print(i, torch.type(m))
            m.accGradParameters = function() end
            m.updateParameters = function() end
            count = count + 1
        end
    end
    
    criterion = nn.MultilabelNLLCriterion():cuda()
    
    return model, criterion
end

return model_utils
