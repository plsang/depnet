require 'loadcaffe'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a caffe model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-back_end', 'cudnn')

cmd:text()
local opt = cmd:parse(arg)

print(opt)

---
print('Loading...')
model = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end)

for k,v in pairs(model) do print (k,v) end

os.exit()

print('checking model by sampling temp input...')
print(#model:cuda():forward(torch.CudaTensor(10, 3, 224, 224)))

---
print('modify model...')
local layer = model:get(1)
local w = layer.weight:clone()
-- swap weights to R and B channels
print('converting first layer conv filters from BGR to RGB...')
layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])

-- parameters, gradParameters = model:parameters()  -- params at each layer
parameters, gradParameters = model:getParameters() -- all params as one flat variables


-- apply a function to each module
model:apply(function(m) print(torch.type(m)) end)

-- traverse the network via traversing the model's .modules table
local total_element = 0
for i, m in ipairs(model.modules) do
   print(torch.type(m))
   if m.weight then
     print(#m.weight)
     total_element = total_element + m.weight:nElement()
   end
   if m.bias then
     print(#m.weight)
     total_element = total_element + m.bias:nElement()   
   end
   if m.ceil_mode then
      print('ceil_mode', m.ceil_mode)  
   end
end

assert(parameters:nElement() == total_element)

local count = 0
for i, m in ipairs(model.modules) do
    if count == 4 then break end
    if torch.type(m):find('Convolution') then
        print(i, torch.type(m))
        -- m.accGradParameters = function() end
        -- m.updateParameters = function() end
        count = count + 1
    end
end