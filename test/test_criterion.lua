require 'nn'
require 'cunn'
require 'cudnn'
require 'MultilabelCrossEntropyCriterion'
require 'nn.MultiLabelCrossEntropyCriterion'

nb = 1
nt = 1000000
loss_weight = 20

x = torch.rand(nb, nt)
t = torch.rand(nb, nt):mul(2):floor()

--[[    
m = nn.MultilabelCrossEntropyCriterion(loss_weight):cuda()

timer = torch.Timer()
m:forward(x:cuda(), t:cuda())
m:backward(x:cuda(), t:cuda())
print('Lua time: ', timer:time().real)
--]]
timer = torch.Timer()
m2 = nn.MultiLabelCrossEntropyCriterion(loss_weight)
m2:forward(x, t)
m2:backward(x, t)
print('C time: ', timer:time().real)

--print('forward diff ', m.output - m2.output)
--print('backward diff ', torch.norm(m.gradInput:double() - m2.gradInput))


timer = torch.Timer()
m3 = nn.MultiLabelCrossEntropyCriterion(loss_weight):cuda()
m3:forward(x:cuda(), t:cuda())
m3:backward(x:cuda(), t:cuda())
print('Cuda time: ', timer:time().real)
print('forward diff ', m2.output - m3.output)
print('backward diff ', torch.norm(m2.gradInput:cuda() - m3.gradInput))
