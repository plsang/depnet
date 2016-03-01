require 'nn'
require 'cunn' -- otherwise, error: attempt to index field 'THNN' (a nil value)
require 'cudnn'
require 'loadcaffe'
require 'MultilabelNLLCriterion'

function create_model(opt)
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
    -- print('converting first layer conv filters from BGR to RGB...')
    -- local input_layer = vgg_model:get(1)
    -- local w = input_layer.weight:clone()
    -- swap weights to R and B channels
    -- input_layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
    -- input_layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    
    criterion = nn.MultilabelNLLCriterion():cuda()
    
    return vgg_model, criterion
end