require 'nn'
require 'cunn' -- otherwise, error: attempt to index field 'THNN' (a nil value)
require 'cudnn'
require 'loadcaffe'

local model_utils = {}

function model_utils.load_vgg(opt)
    vgg_model = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end)

    ---
    -- print('Removing the softmax layer')
    table.remove(vgg_model['modules'])  -- equivalent to pop
    
    criterion = nn.MultilabelCrossEntropyCriterion():cuda()
    
    return vgg_model, criterion
end

function model_utils.init_finetuning_params(model, opt)
    for _, m in ipairs(model:listModules()) do
        if m.weight and m.bias then
            if m.name == opt.finetune_layer_name then
                if opt.weight_init > 0 then
                    print('Initializing parameters of the finetuned layer')
                    m.weight:zero():normal(0, opt.weight_init) -- gaussian of zero mean
                    m.bias:zero():fill(opt.bias_init)          -- constant value    
                end
            end
        elseif m.weight or m.bias then
            error('Layer that has either weight or bias')     
        end
    end

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
    
    return model
end


-- Define a fine tuning network, that contains different groups.
-- each group will have different learning rates
function model_utils.finetune_vgg(opt)
    local main_model = nn.Sequential()
    -- group 1: frozen weight
    local model = nn.Sequential()
    
    -- learing rate & weight decay multipliers for this group
    model.frozen = true
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
    model.frozen = false
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
    
    model:add(cudnn.SpatialConvolution(4096, 1000, 1, 1, 1, 1, 0, 0, 1))
    model:get(#model).name = 'fc8'
    
    model:add(nn.View(-1):setNumInputDims(3)) -- 1x1x1000 --> 1000
    model:get(#model).name = 'torch_view'
    
    model:add(nn.Sigmoid())
    model:get(#model).name = 'sigmoid'
    
    main_model:add(model)
    model = nil
    
    cudnn.convert(main_model, cudnn)
    
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
        assert(parameters:nElement() == parameters_vgg:nElement())
        parameters:copy(parameters_vgg)
        parameters_vgg = nil
        model_vgg = nil
    end
    collectgarbage() 
    
    return main_model
end

-- a building block for convultion neurals, with batch normalization
function model_utils.build_conv_block(model, nInputDim, nOutputDim, kernel_size, step_size, pad_size, convName, bnName, reluName)
    model:add(cudnn.SpatialConvolution(nInputDim, nOutputDim, kernel_size, kernel_size, step_size, step_size, pad_size, pad_size, 1))
    model:get(#model).name = convName
    model:add(cudnn.SpatialBatchNormalization(nOutputDim, 1e-3))
    model:get(#model).name = bnName
    model:add(cudnn.ReLU(true))
    model:get(#model).name = reluName
end

-- Define a fine tuning network, that contains different groups.
-- each group will have different learning rates
-- support Batch Normalization
function model_utils.finetune_vgg_bn(opt)
    local main_model = nn.Sequential()
    -- group 1: frozen weight
    local model = nn.Sequential()
    
    -- learing rate & weight decay multipliers for this group
    model.frozen = true
    model_utils.build_conv_block(model, 3, 64, 3, 1, 1, 'conv1_1', 'bn1_1', 'relu1_1')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop1'
    
    model_utils.build_conv_block(model, 64, 64, 3, 1, 1, 'conv1_2', 'bn1_2', 'relu1_2')
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool1'
    
    model_utils.build_conv_block(model, 64, 128, 3, 1, 1, 'conv2_1', 'bn2_1', 'relu2_1')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop2'
    
    model_utils.build_conv_block(model, 128, 128, 3, 1, 1, 'conv2_2', 'bn2_2', 'relu2_2')
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool2'
    
    main_model:add(model)
    model = nil
    
    -- group 2: normal weight
    local model = nn.Sequential()
    -- learing rate & weight decay multipliers for this group
    model.frozen = false
    
    model_utils.build_conv_block(model, 128, 256, 3, 1, 1, 'conv3_1', 'bn3_1', 'relu3_1')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop3_1'
    
    model_utils.build_conv_block(model, 256, 256, 3, 1, 1, 'conv3_2', 'bn3_2', 'relu3_2')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop3_2'
    
    model_utils.build_conv_block(model, 256, 256, 3, 1, 1, 'conv3_3', 'bn3_3', 'relu3_3')
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool3'
    
    model_utils.build_conv_block(model, 256, 512, 3, 1, 1, 'conv4_1', 'bn4_1', 'relu4_1')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop4_1'
    
    model_utils.build_conv_block(model, 512, 512, 3, 1, 1, 'conv4_2', 'bn4_2', 'relu4_2')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop4_2'
    
    model_utils.build_conv_block(model, 512, 512, 3, 1, 1, 'conv4_3', 'bn4_3', 'relu4_3')
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool4'
    
    model_utils.build_conv_block(model, 512, 512, 3, 1, 1, 'conv5_1', 'bn5_1', 'relu5_1')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop5_1'
    
    model_utils.build_conv_block(model, 512, 512, 3, 1, 1, 'conv5_2', 'bn5_2', 'relu5_2')
    model:add(nn.Dropout(0.5))
    model:get(#model).name = 'drop5_2'
    
    model_utils.build_conv_block(model, 512, 512, 3, 1, 1, 'conv5_3', 'bn5_3', 'relu5_3')
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    model:get(#model).name = 'pool5'
    
    -- note: padding here is zero, otherwise, the output layer has 25,000 elements?
    model_utils.build_conv_block(model, 512, 4096, 7, 1, 0, 'fc6', 'bn6', 'relu6')
    model:add(nn.Dropout(0.500000))
    model:get(#model).name = 'drop6'
    
    model_utils.build_conv_block(model, 4096, 4096, 1, 1, 0, 'fc7', 'bn7', 'relu7')
    model:add(nn.Dropout(0.500000))
    model:get(#model).name = 'drop7'
    
    -- this layer can be considered as a linear output, don't apply batch normalization here 
    model:add(cudnn.SpatialConvolution(4096, 1000, 1, 1, 1, 1, 0, 0, 1))
    model:get(#model).name = 'fc8'
    
    model:add(nn.View(-1):setNumInputDims(3)) -- 1x1x1000 --> 1000
    model:get(#model).name = 'torch_view'
    
    model:add(nn.Sigmoid())
    model:get(#model).name = 'sigmoid'
    
    main_model:add(model)
    model = nil
    
    cudnn.convert(main_model, cudnn)
    
    local parameters = main_model:getParameters() -- all params as one flat variables
    
    print(main_model.modules)
    
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
        
        print(parameters:nElement(), parameters_vgg:nElement())
        
        -- copy weights from VGG model to the fined tune network.
        -- Since the size of the params is different now (thanks to BN layers),
        -- we have to copy it mananually
        local vgg_idx = 0
        for _, m in ipairs(main_model:listModules()) do
            if m.weight and m.bias then
                
                local wlen = m.weight:nElement()
                local blen = m.bias:nElement()
                local mlen = wlen + blen
                
                -- check if this is not a batch normalization layer
                if string.sub(m.name,1,2) ~= 'bn' then
                    m.weight:copy(parameters_vgg[{{vgg_idx + 1, vgg_idx + wlen}}])
                    m.bias:copy(parameters_vgg[{{vgg_idx + wlen + 1, vgg_idx + mlen}}])
                    vgg_idx = vgg_idx + mlen
                end
                
            elseif m.weight or m.bias then
                error('Layer that has either weight or bias')     
            end
        end
        assert(vgg_idx == parameters_vgg:nElement())
        
        parameters_vgg = nil
        model_vgg = nil
    end
    collectgarbage() 
    
    return main_model
end

-- get params (weights + biases indinces) and save them to the config param
-- used to get invdividual params from each layers, usefule for fine-tuning
function model_utils.update_param_indices(model, opt, optim_config)
    
    local frozen_graph = model.modules[1]
    assert(frozen_graph.frozen == true)

    optim_config.frozen_start = 1

    local total_elements = 0
    for _, m in ipairs(frozen_graph.modules) do
        if m.weight and m.bias then
            local wlen = m.weight:nElement()
            local blen = m.bias:nElement()
            local mlen = wlen + blen
            total_elements = total_elements + mlen
        end
    end
    
    optim_config.frozen_end = total_elements

    local finetune_graph = model.modules[2]
    assert(finetune_graph.frozen == false)

    local bias_indices = {}
    for _, m in ipairs(finetune_graph.modules) do
       if m.weight and m.bias then

            local wlen = m.weight:nElement()
            local blen = m.bias:nElement()
            local mlen = wlen + blen
            table.insert(bias_indices, total_elements + wlen + 1)
            table.insert(bias_indices, total_elements + mlen)

            if m.name == opt.finetune_layer_name then
                print('Fine tuning layer found!')
                optim_config.ft_ind_start = total_elements + 1
                optim_config.ft_ind_end = total_elements + mlen
                optim_config.ftb_ind_start = total_elements + wlen + 1 -- fine tune bias index start
                optim_config.ftb_ind_end = total_elements + mlen       -- fine tune bias index end
            end

            total_elements = total_elements + mlen
       elseif m.weight or m.bias then
           error('Layer that has either weight or bias')     
       end
    end

    optim_config.bias_indices = bias_indices
    assert(optim_config.ft_ind_start, 'Fine tuning layer not found')
    
end

return model_utils
