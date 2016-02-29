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
vgg_model = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end)

print('checking model by sampling temp input...')
print(#model:cuda():forward(torch.CudaTensor(10, 3, 224, 224)))

---
print('modify model...')
local input_layer = model:get(1)
local w = layer.weight:clone()
-- swap weights to R and B channels
print('converting first layer conv filters from BGR to RGB...')
layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])

