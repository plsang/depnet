require 'nn'
require 'cudnn'
require 'MultilabelNLLCriterion'
require 'MultilabelCrossEntropyCriterion'


-- NOTE: Assumes input and output to module are 1-dimensional, i.e. doesn't test the module
--       in mini-batch mode. It's easy to modify it to do that if you want, though.
local function jacobian_wrt_input(module, x, eps)
  -- compute true Jacobian (rows = over outputs, cols = over inputs, as in our writeup's equations)
  local z = module:forward(x):clone()
  local jac = torch.DoubleTensor(z:size(1), x:size(1))
  
  -- get true Jacobian, ROW BY ROW
  local one_hot = torch.zeros(z:size())
  for i = 1, z:size(1) do
    one_hot[i] = 1
    jac[i]:copy(module:backward(x, one_hot))
    one_hot[i] = 0
  end
  
  -- compute finite-differences Jacobian, COLUMN BY COLUMN
  local jac_est = torch.DoubleTensor(z:size(1), x:size(1))
  for i = 1, x:size(1) do
    x[i] = x[i] + eps
    local z_offset = module:forward(x)
    jac_est[{{},i}]:copy(z_offset)
    x[i] = x[i] - 2*eps
    z_offset = module:forward(x)
    jac_est[{{},i}]:add(-1, z_offset):div(2*eps)
    x[i] = x[i] + eps
  end

  -- computes (symmetric) relative error of gradient
  local abs_diff = (jac - jac_est):abs()
  return jac, jac_est, torch.mean(abs_diff), torch.min(abs_diff), torch.max(abs_diff)
end

local function crit_check(crit, x, target, eps)
  -- compute true Jacobian (rows = over outputs, cols = over inputs, as in our writeup's equations)
  local loss = crit:forward(x, target)
  print('loss = ', loss)
  local grad = crit:backward(x, target)
 
  local grad_est = torch.Tensor(grad:size()):cuda()
  for i = 1, grad:size(1) do
    for j = 1, grad:size(2) do   
        -- Evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
        x[i][j] = x[i][j] + eps
        local e1 = crit:forward(x, target)
        x[i][j] = x[i][j] - 2*eps
        local e2 = crit:forward(x, target)
        grad_est[i][j] = (e1 - e2) / (2 * eps)
        x[i][j] = x[i][j] + eps
     end
  end
  
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  print(torch.cat(grad, grad_est, 1))
  print(diff)
end


---------------------------------------------------------
-- test our layer in isolation
--
torch.manualSeed(1)
local model = nn.Sequential()
model:add(nn.LogSoftMax())
-- model:add(nn.MultilabelNLLCriterion())
-- crit = nn.MultilabelNLLCriterion():cuda()
crit = nn.MultilabelCrossEntropyCriterion():cuda()

-- note: if input is not probabilistic values (0,1), jac check failed
-- local x = torch.randn(1,10):cuda() -- random input to layer
local x = torch.rand(1,10):cuda() -- random input to layer
local t = torch.rand(1,10):mul(2):floor():cuda()
print(torch.cat(x, t, 1))
eps=1e-6
-- print(jacobian_wrt_input(model, x, 1e-7))
crit_check(crit, x, t, eps)

