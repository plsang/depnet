require 'create_model'

-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)
  -- compute true gradient
  local grad = g(x)
  
  -- compute numeric approximations to gradient
  local eps = eps or 1e-7
  local grad_est = torch.DoubleTensor(grad:size())
  for i = 1, grad:size(1) do
    -- Evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
    x[i] = x[i] + eps
    local e1 = f(x)
    x[i] = x[i] - 2*eps
    local e2 = f(x)
    grad_est[i] = (e1 - e2) / (2 * eps)
    x[i] = x[i] + eps
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  return diff, grad, grad_est
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
function fakedata(n)
    local data = {}
    data.inputs = torch.randn(n, 3, 224, 224):mul(256):floor():cuda()          -- random standard normal distribution for inputs
    data.targets = torch.rand(n, 1000):mul(2):floor():cuda()  -- random {0,1} labels
    return data
end

torch.manualSeed(1)
local data = fakedata(1)

opt = {}
opt.cnn_proto='model/VGG_ILSVRC_16_layers_deploy.prototxt'
opt.cnn_model='model/VGG_ILSVRC_16_layers.caffemodel'
opt.back_end='cudnn'

local model, criterion = create_model(opt)
local parameters, gradParameters = model:getParameters()

-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  return criterion:forward(model:forward(data.inputs), data.targets)
end
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local outputs = model:forward(data.inputs)
  criterion:forward(outputs, data.targets)
  model:backward(data.inputs, criterion:backward(outputs, data.targets))

  return gradParameters
end

local diff = checkgrad(f, g, parameters)
print(diff)

