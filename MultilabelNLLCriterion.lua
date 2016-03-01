local THNN = require 'nn.THNN'
local MultilabelNLLCriterion, parent = torch.class('nn.MultilabelNLLCriterion', 'nn.Criterion')

function MultilabelNLLCriterion:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
       self.sizeAverage = sizeAverage
    else
       self.sizeAverage = true
    end
    if weights then
       assert(weights:dim() == 1, "weights input should be 1-D Tensor")
       self.weights = weights
    end

    self.output_tensor = torch.zeros(1)
    self.total_weight_tensor = torch.ones(1)
    self.target = torch.zeros(1):long()
end

function MultilabelNLLCriterion:__len()
    if (self.weights) then
        return #self.weights
    else
        return 0
    end
end

--[[
this implementation only penalizes correct labels
--]]

function MultilabelNLLCriterion:updateOutput(input, target)
    if target:type() == 'torch.CudaTensor' then
        
        -- check that input and target have the same shape
        assert(input:size(1) == target:size(1)) -- batch_size
        assert(input:size(2) == target:size(2)) -- num_target
        
        --[[  firstway
        local batch_size = input:size(1)
        local num_target = input:size(2)
        local loss = 0
        for i=1,batch_size do
            for j=1,num_target do
                if target[i][j] == 1 then
                    loss = loss - input[i][j]
                end
            end
        end
        
        --]]
        
        -- second way
        local loss = -torch.sum(input[target])/input:size(1)
        self.output_tensor[1] = loss
        
    else
        error('Only support target type of CudaTensor')
    end
    
    self.output = self.output_tensor[1]
    return self.output, self.total_weight_tensor[1]
end

function MultilabelNLLCriterion:updateGradInput(input, target)
    if target:type() == 'torch.CudaTensor' then
        
        -- check that input and target have the same shape
        assert(input:size(1) == target:size(1)) -- batch_size
        assert(input:size(2) == target:size(2)) -- num_target
        
        self.gradInput:resizeAs(input):zero()
        
        -- local grad = target:cuda():mul(-1)/input:size(1)
        -- self.gradInput:copy(grad)
        
        self.gradInput[target] = -1/input:size(1)
    else
        error('Only support target type of CudaTensor')
    end 
    
    return self.gradInput
end
