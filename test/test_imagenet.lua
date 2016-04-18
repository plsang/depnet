require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'loadcaffe'
dbg = require 'debugger'
require 'CocoData'

local coco_data_root = '/home/ec2-user/data/Microsoft_COCO'

cmd = torch.CmdLine()
cmd:text('Options')
-- Data input settings
cmd:option('-network', '../torch-tutorials/7_imagenet_classification/nin_nobn_final.t7')
cmd:option('-vocab', '../torch-tutorials/7_imagenet_classification/synset_words.txt')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
-- cmd:option('-cnn_proto','data/deploy.prototxt','path to CNN prototxt file in Caffe format.')
-- cmd:option('-cnn_model','data/bvlc_reference_caffenet.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-train_image_file_h5', 'data/coco_train.h5', 'path to the prepressed image data')
cmd:option('-train_label_file_h5', paths.concat(coco_data_root, 'mscoco2014_train_myconceptsv3.h5'), 'path to the prepressed label data')

cmd:option('-back_end', 'cudnn')
cmd:option('-image', '../torch-tutorials/7_imagenet_classification/Goldfish3.jpg')
cmd:text()

local opt = cmd:parse(arg)

loader = CocoData{image_file_h5 = opt.train_image_file_h5, label_file_h5 = opt.train_label_file_h5, 
    num_target = opt.num_target, batch_size = opt.batch_size}

-- Rescales and normalizes the image
function preprocess(im, img_mean)
  -- rescale the image  
  im = image.scale(im,224,224,'bilinear'):cuda()
    
  im_mean = torch.Tensor{123.68, 116.779, 103.939}:view(3,1,1):cuda()
  im:add(-1, im_mean:expandAs(im))
    
  -- swap weights to R and B channels
  local im_ = im:clone()
  im[{{1}, {}, {} }]:copy(im_[{{3}, {}, {} }])
  im[{{3}, {}, {} }]:copy(im_[{{1}, {}, {} }])
  
  -- subtract imagenet mean and divide by std
  -- for i=1,3 do im3[i]:add(-img_mean.mean[i]):div(img_mean.std[i]) end
  
  return im
end

print '==> Loading network'
net = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.back_end)
net:evaluate()
print(net)
local params, grad_params = net:getParameters()
print('total number of parameters: ', params:nElement(), grad_params:nElement())

local refnet = torch.load(opt.network):unpack():cuda()

print '==> Loading synsets'
print 'Loads mapping from net outputs to human readable labels'
local synset_words = {}
for line in io.lines(opt.vocab) do table.insert(synset_words, line:sub(11)) end

print '==> Loading image'
local im = image.load(opt.image, 3, 'byte')
-- print(im)

print '==> Preprocessing'
-- Our network has mean and std saved in net.transform
local I = preprocess(im, refnet.transform):view(1,3,224,224):cuda()

print 'Propagate through the network, sort outputs in decreasing order and show 5 best classes'
local _,classes = net:forward(I):view(-1):sort(true)
for i=1,10 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end

for iter=1, 1 do
    print('----------')
    local data = loader:getBatch()
    print('Image id: ', data.image_ids[1])
    local outputs = net:forward(data.images:cuda())

    local _,classes = net:forward(data.images:cuda()):view(-1):sort(true)
    for i=1,10 do
      print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
    end
end    

          