
require 'loadcaffe'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a caffe model')
cmd:text()
cmd:text('Options')

cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')

cmd:text()
local opt = cmd:parse(arg)

print(opt)

print('Loading...')
model1 = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end):cuda()
model2 = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end):cuda()

model1:evaluate()
model2:evaluate()

params1, grad1 = model1:getParameters()
params2, grad2 = model2:getParameters()

grad1:zero()
grad2:zero()

assert(torch.norm(params1 - params2) < 1e-10)

--- assert layer by layer

for i=1,#model1.modules do
    for k, v in pairs(model1.modules[i]) do
        if type(model1.modules[i][k]) == 'userdata' and model1.modules[i][k].__typename == 'torch.CudaTensor' then
            -- assert(torch.all(torch.eq(model1.modules[i][k], model2.modules[i][k])), k .. i)    
            assert(torch.norm(model1.modules[i][k] - model2.modules[i][k]) < 1e-5,
                string.format('%s %s %s %s', tostring(i), tostring(k), torch.norm(model1.modules[i][k]), torch.norm(model2.modules[i][k])))
        elseif type(model1.modules[i][k]) == 'userdata' and model1.modules[i][k].__typename == 'torch.LongStorage' then
            assert(model1.modules[i][k]:size() == model2.modules[i][k]:size())
        else
            assert(model1.modules[i][k] == model2.modules[i][k],  
                string.format('%s %s %s %s', tostring(i), tostring(k), tostring(model1.modules[i][k]), tostring(model2.modules[i][k])))
        end
    end
    
    print('passed ', i)
end


torch.manualSeed(123)
local inputs = torch.rand(1,3,224,224):uniform():cuda()
print('input norm = ', torch.norm(inputs))
output1 = model1:forward(inputs)
output2 = model2:forward(inputs)

print('ouput1 norm = ', torch.norm(output1))
print('ouput2 norm = ', torch.norm(output2))

