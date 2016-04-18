require 'nn'
require 'nn.SpatialMIL'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'MILLayer'

--dbg = require 'debugger'

--torch.setdefaulttensortype('torch.FloatTensor')

nb = 4 
nt = 4096
mil_type = 'milmax'

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
for i=1,nb do
    for j=1,nt do
        local w = math.floor(m2.mil_indices[i][j]/12) + 1
        local h = m2.mil_indices[i][j]%12 + 1
        -- print(m2.mil_indices[i][j], w, h, x[i][j][w][h], m2.output[i][j])
        assert(x[i][j][w][h] == m2.output[i][j])
    end
end
m2:backward(x, y)
print('C time: ', timer:time().real)

--print('forward diff ', torch.norm(m.output:double() - m2.output))
--print('backward diff ', torch.norm(m.gradInput:double() - m2.gradInput))

timer = torch.Timer()
m3 = nn.SpatialMIL(mil_type):cuda()
m3:forward(x:cuda())
for i=1,nb do
    for j=1,nt do
        local w = math.floor(m3.mil_indices[i][j]/12) + 1
        local h = m3.mil_indices[i][j]%12 + 1
        -- print(m3.mil_indices[i][j], w, h, x[i][j][w][h], m3.output[i][j])
        assert(x[i][j][w][h] - m3.output[i][j] < 1e-6)
    end
end
m3:backward(x:cuda(), y:cuda())
print('Cuda time: ', timer:time().real)

print('forward diff ', torch.norm(m2.output:cuda() - m3.output))
print('backward diff ', torch.norm(m2.gradInput:cuda() - m3.gradInput))


