require 'nn'
require 'nn.SpatialMIL'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'MILLayer'

--dbg = require 'debugger'

--torch.setdefaulttensortype('torch.FloatTensor')

nb = 4 
nt = 1000000
mil_type = 'milnor'

x = torch.rand(nb,nt,12,12):uniform(0, 0.1)
y = torch.rand(nb,nt)

--[[
m = nn.MILLayer(mil_type):cuda()

timer = torch.Timer()
m:forward(x:cuda())
m:backward(x:cuda(), y:cuda())
print('Lua time: ', timer:time().real)
--]]

timer = torch.Timer()
m2 = nn.SpatialMIL(mil_type)
m2:forward(x)
m2:backward(x, y)
print('C time: ', timer:time().real)

--print('forward diff ', torch.norm(m.output:double() - m2.output))
--print('backward diff ', torch.norm(m.gradInput:double() - m2.gradInput))

timer = torch.Timer()
m3 = nn.SpatialMIL(mil_type):cuda()
m3:forward(x:cuda())
m3:backward(x:cuda(), y:cuda())
print('Cuda time: ', timer:time().real)

print('forward diff ', torch.norm(m2.output:cuda() - m3.output))
print('backward diff ', torch.norm(m2.gradInput:cuda() - m3.gradInput))


