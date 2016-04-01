require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'loadcaffe'
-- dbg = require 'debugger'
require 'CocoData'
local model_utils = require 'model_utils'

local coco_data_root = '/home/ec2-user/data/Microsoft_COCO'

cmd = torch.CmdLine()
cmd:text('Options')
-- Data input settings
cmd:option('-test_cp', 'cp/model_multitask_milmaxnor_mt1_adam_b4_bias-6.580000_lr0.000010_epoch2.t7')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format.')
cmd:option('-batch_size', 1, 'Number of image per batch')
cmd:option('-back_end', 'cudnn')
cmd:option('-num_target', 22034, 'Number of target concepts, -1 for getting from file')
cmd:option('-image', '../torch-tutorials/7_imagenet_classification/Goldfish3.jpg')
cmd:option('-model_type', 'milmaxnor', 'vgg, vggbn, milmax, milnor')
cmd:option('-phase', 'test', 'phase (train/test)')
cmd:text()

local opt = cmd:parse(arg)
print(opt)

-- Rescales and normalizes the image
local function preprocess(im)
  -- rescale the image  
  im = image.scale(im,565,565,'bilinear'):cuda()
    
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
local model = model_utils.load_model(opt):cuda()
model:evaluate()  

local vocab = model.vocab

print '==> Loading image'
local im = image.load(opt.image, 3, 'byte')
-- print(im)

print '==> Preprocessing'
-- Our network has mean and std saved in net.transform
local I = preprocess(im):view(1,3,565,565):cuda()

print 'Propagate through the network, sort outputs in decreasing order and show 5 best classes'
local _,classes = model:forward(I):view(-1):sort(true)

for i=1,10 do
  print('predicted class '..tostring(i)..': ', vocab[classes[i] ])
end

          