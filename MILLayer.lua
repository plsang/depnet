require 'nn'

local MILLayer, parent = torch.class('nn.MILLayer', 'nn.Module')

function MILLayer:__init(mil_type)
    parent.__init(self) -- would inherit gradInput and output variables
    
    if mil_type then
        self.mil_type = mil_type
    else
        self.mil_type = 'milnor' -- noisy OR
    end
    self.max_indices = torch.LongTensor()
    
    self.width = 12
    self.height = 12
    self.tmp = torch.Tensor(self.width, self.height):fill(1):cuda()
end
    
function MILLayer:updateOutput(input)

    local batch_size = input:size(1)
    local num_channels = input:size(2)
    assert(self.width == input:size(3))
    
    self.output:resize(batch_size, num_channels):zero()
    
    if self.mil_type == 'milmax' then
        local max_concepts, max_indices = torch.max(input:view(batch_size, num_channels, -1), 3)
        self.output:copy(max_concepts)
        max_indices = max_indices:squeeze(3) -- remove that 3rd dim
        self.max_indices:resizeAs(max_indices):copy(max_indices)
        
    elseif self.mil_type == 'milnor' then
        for i=1, batch_size do
            for j=1, num_channels do
                local prob = 1
                input[i][j]:apply(function(x) prob = prob * (1 - x) end)
                -- local prob = torch.csub(self.tmp, input[i][j]):prod() -- slower
                local max_prob = torch.max(input[i][j])
                self.output[i][j] = math.max(1 - prob, max_prob)
            end 
        end
    else
        error('Unknown MIL type', self.mil_type)
    end
        
    return self.output
end

function MILLayer:updateGradInput(input, gradOutput)
    
    local batch_size = input:size(1)
    local num_channels = input:size(2)
    assert(self.width == input:size(3))
        
    if self.mil_type == 'milmax' then
        self.gradInput:resizeAs(input):zero()
        for i=1, batch_size do
            for j=1, num_channels do
                local max_idx = self.max_indices[i][j]
                local w_idx = math.ceil(max_idx/self.height)
                local h_idx = max_idx % self.height
                if h_idx == 0 then h_idx = self.height end
                self.gradInput[i][j][w_idx][h_idx] = gradOutput[i][j]
            end
        end
    elseif self.mil_type == 'milnor' then
        self.gradInput:resizeAs(input):fill(1)
        for i=1, batch_size do
            for j=1, num_channels do
                local p = torch.Tensor(self.width, self.height):fill(1-self.output[i][j]):cuda()
                local q = torch.csub(self.tmp, input[i][j])    
                self.gradInput[i][j]:cmin(p:cdiv(q)):mul(gradOutput[i][j])
            end
        end
    else
        error('Unknown MIL type', sefl.mil_type)
    end

    return self.gradInput
end
