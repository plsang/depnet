local THNN = require 'nn.THNN'
local MultilabelCrossEntropyCriterion, parent = torch.class('nn.MultilabelCrossEntropyCriterion', 'nn.Criterion')

function MultilabelCrossEntropyCriterion:__init(loss_weight, weights, sizeAverage)
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
    
    -- support caffe loss weight
    if loss_weight then
        self.loss_weight = loss_weight
    else
        self.loss_weight = 1
    end
    assert(self.loss_weight > 0)
end

function MultilabelCrossEntropyCriterion:__len()
    if (self.weights) then
        return #self.weights
    else
        return 0
    end
end

--[[
this implementation only penalizes correct labels
--]]

function MultilabelCrossEntropyCriterion:updateOutput(input, target)
    if target:type() == 'torch.CudaTensor' then
        
        -- check that input and target have the same shape
        assert(input:size(1) == target:size(1)) -- batch_size
        assert(input:size(2) == target:size(2)) -- num_target
        
        local eps = 1e-5
        local loss = 0
        local l = 0
        for i=1,input:size(1) do
            for j=1,input:size(2) do
                if target[i][j] == 1 then
                    l = -torch.log(math.max(input[i][j], eps))
                else
                    l = -torch.log(math.max(1-input[i][j], eps))
                end
                loss = loss + l
            end
        end
        loss = loss/input:nElement()
        self.output = loss
    else
        error('Only support target type of CudaTensor')
    end
    
    return self.output, self.total_weight_tensor[1]
end

-- Note that loss_weight is multipiled at element-wise
function MultilabelCrossEntropyCriterion:updateGradInput(input, target)
    if target:type() == 'torch.CudaTensor' then
        
        -- check that input and target have the same shape
        assert(input:size(1) == target:size(1)) -- batch_size
        assert(input:size(2) == target:size(2)) -- num_target
        
        self.gradInput:resizeAs(input):zero()
        local eps = 1e-5
        local val = 0
        for i=1,input:size(1) do
            for j=1,input:size(2) do
                if target[i][j] == 1 then
                    val = -self.loss_weight/(math.max(input[i][j], eps) * input:nElement())
                else
                    val = self.loss_weight/(math.max(1-input[i][j], eps) * input:nElement())
                end
                self.gradInput[i][j] = val
            end
        end
    else
        error('Only support target type of CudaTensor')
    end 
    
    return self.gradInput
end
