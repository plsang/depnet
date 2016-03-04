require 'nn'
require 'cunn' -- otherwise, error: attempt to index field 'THNN' (a nil value)
require 'cudnn'
require 'loadcaffe'
require 'MultilabelNLLCriterion'
require 'MultilabelCrossEntropyCriterion'

local model_utils = {}

function model_utils.load_vgg(opt)
    vgg_model = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end)

    ---
    -- print('Removing the softmax layer')
    table.remove(vgg_model['modules'])  -- equivalent to pop
    
    criterion = nn.MultilabelCrossEntropyCriterion():cuda()
    
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
    model:add(nn.Sigmoid())
    model:get(#model).name = 'sigmoid'
    
    if opt.phase == 'train' then
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
    end
    
    cudnn.convert(model, cudnn)
    -- convert data to CudaTensor
    for i, m in ipairs(model.modules) do
        m:cuda()
    end
    
    local parameters = model:getParameters() -- all params as one flat variables
    
    -- for k,v in pairs(model) do print (k,v) end
    if opt.test_cp ~= '' then
        print('loading network from a checkpoint ', opt.test_cp)
        model_cp = torch.load(opt.test_cp)
        local parameters_cp = model_cp.params
        assert(parameters:nElement() == parameters_cp:nElement())
        parameters:copy(parameters_cp)
        model_cp = nil
        parameters_cp = nil
    else
        print('loading network from a vgg pretrained model')
        local model_vgg = model_utils.load_vgg(opt)
        local parameters_vgg = model_vgg:getParameters() -- all params as one flat variables
        assert(parameters:nElement() == parameters_vgg:nElement())
        parameters:copy(parameters_vgg)
        parameters_vgg = nil
        model_vgg = nil
    end
    collectgarbage() 
    
    criterion = nn.MultilabelCrossEntropyCriterion():cuda()
    
    return model, criterion
end


-- Define a fine tuning network, that contains different groups.
-- each group will have different learning rates
function model_utils.finetune_vgg(opt)
    local main_model = nn.Sequential()
    -- group 1: frozen weight
    local model = nn.Sequential()
    
    -- learing rate & weight decay multipliers for this group
    model.w_lr_mult = 0
    model.b_lr_mult = 0
    model.w_wd_mult = 0
    model.b_wd_mult = 0
    
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
    
    main_model:add(model)
    model = nil
    
    -- group 2: normal weight
    local model = nn.Sequential()
    -- learing rate & weight decay multipliers for this group
    model.w_lr_mult = 1
    model.b_lr_mult = 2
    model.w_wd_mult = 1
    model.b_wd_mult = 0
    
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
    
    model:add(cudnn.SpatialConvolution(512, 4096, 7, 7, 1, 1, 0, 0, 1))
    model:get(#model).name = 'fc6'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu6'
    model:add(nn.Dropout(0.500000))
    model:get(#model).name = 'drop6'
    
    model:add(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0, 1))
    model:get(#model).name = 'fc7'
    model:add(cudnn.ReLU(true))
    model:get(#model).name = 'relu7'
    model:add(nn.Dropout(0.500000))
    model:get(#model).name = 'drop7'
    
    main_model:add(model)
    model = nil
    
    -- group 3: tuning weight
    local model = nn.Sequential()
    -- learing rate & weight decay multipliers for this group
    model.w_lr_mult = 10
    model.b_lr_mult = 20
    model.w_wd_mult = 1
    model.b_wd_mult = 0
    
    model:add(cudnn.SpatialConvolution(4096, 1000, 1, 1, 1, 1, 0, 0, 1))
    model:get(#model).name = 'fc8'
    
    model:add(nn.View(-1):setNumInputDims(3)) -- 1x1x1000 --> 1000
    model:get(#model).name = 'torch_view'
    
    model:add(nn.Sigmoid())
    model:get(#model).name = 'sigmoid'
    
    main_model:add(model)
    model = nil
    
    cudnn.convert(main_model, cudnn)
    main_model:cuda()
    
    local parameters = main_model:getParameters() -- all params as one flat variables
    
    -- for k,v in pairs(model) do print (k,v) end
    if opt.test_cp ~= '' then
        print('loading network from a checkpoint ', opt.test_cp)
        model_cp = torch.load(opt.test_cp)
        local parameters_cp = model_cp.params
        assert(parameters:nElement() == parameters_cp:nElement())
        parameters:copy(parameters_cp)
        model_cp = nil
        parameters_cp = nil
    else
        print('loading network from a vgg pretrained model')
        local model_vgg = model_utils.load_vgg(opt)
        local parameters_vgg = model_vgg:getParameters() -- all params as one flat variables
        print(parameters:nElement())
        print(parameters_vgg:nElement())
        assert(parameters:nElement() == parameters_vgg:nElement())
        parameters:copy(parameters_vgg)
        parameters_vgg = nil
        model_vgg = nil
    end
    collectgarbage() 
    
    
    criterion = nn.MultilabelCrossEntropyCriterion():cuda()
    
    return main_model, criterion
end


return model_utils
