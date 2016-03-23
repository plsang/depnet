local optim_utils = {}

function optim_utils.sgd(x, dfdx, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-3
    local lrd = config.learningRateDecay or 0
    local wd = config.weightDecay or 0
    local mom = config.momentum or 0
    local damp = config.dampening or mom
    local nesterov = config.nesterov or false
    local lrs = config.learningRates
    local wds = config.weightDecays
    state.evalCounter = state.evalCounter or 0
    local nevals = state.evalCounter
    assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

    -- (1) copy param of the frozen layers
    local frozen_x = x[{{config.frozen_start, config.frozen_end}}]:clone()
   
    -- (2) weight decay with single
    if wd ~= 0 then
        -- this will apply weight decay to bias as well, which is wd*b
        dfdx:add(wd, x)
        -- minus wd*b
        for i=1,#config.bias_indices,2 do 
           dfdx[{{config.bias_indices[i], config.bias_indices[i+1]}}]:add(-wd, x[{{config.bias_indices[i], config.bias_indices[i+1]}}]) 
        end
    end

    -- (3) apply momentum
    if mom ~= 0 then
        if not state.dfdx then
            state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
        else
            state.dfdx:mul(mom):add(1-damp, dfdx)
        end
        if nesterov then
            dfdx:add(mom, state.dfdx)
        else
            dfdx = state.dfdx
        end
    end

    -- (4) learning rate decay (annealing)
    local clr = lr / (1 + nevals*lrd)

    -- (5) parameter update, this apply the base learning rate
    x:add(-clr, dfdx)
    
    -- finetuning layer may need more update
    x[{{config.ft_ind_start, config.ft_ind_end}}]:add(-(config.w_lr_mult-1)*clr, dfdx[{{config.ft_ind_start, config.ft_ind_end}}])
    
    -- bias update twice
    for i=1,#config.bias_indices,2 do 
        x[{{config.bias_indices[i], config.bias_indices[i+1]}}]:add(-clr, dfdx[{{config.bias_indices[i], config.bias_indices[i+1]}}]) 
    end
    -- bias update on fine tuning layer, since it already updated twice, minus 2 to the multiplier
    x[{{config.ftb_ind_start, config.ftb_ind_end}}]:add(-(config.b_lr_mult-2)*clr, dfdx[{{config.ftb_ind_start, config.ftb_ind_end}}])
    
    
    -- (6) update evaluation counter
    state.evalCounter = state.evalCounter + 1

    -- (7) restore frozen_x
    x[{{config.frozen_start, config.frozen_end}}]:copy(frozen_x)
    frozen_x = nil
    -- return x*, f(x) before optimization
    -- return x, fx
end

function optim_utils.adam(x, dfdx, config, state)
    local beta1 = config.adam_beta1 or 0.9
    local beta2 = config.adam_beta2 or 0.999
    local epsilon = config.adam_epsilon or 1e-8
    local state = state or config
    local lr = config.learningRate or 1e-3
    local wd = config.weightDecay or 0
    
    if not state.m then
        --initialization
        state.t = 0
        -- momentum1 m = beta1*m + (1-beta1)*dx
        state.m = x.new(#dfdx):zero()
        -- mementum2 v = beta2*v + (1-beta2)*(dx**2)
        state.v = x.new(#dfdx):zero()
        -- tmp tensor to hold the sqrt(v) + epsilon
        state.tmp = x.new(#dfdx):zero()
    end
    
    if wd ~= 0 then
        -- regularization only at the finetuned layer
        if config.reg_type == 1 then
            dfdx[{{config.ft_ind_start, config.ft_ind_end}}]:add(torch.sign(x[{{config.ft_ind_start, config.ft_ind_end}}]):mul(wd))
        elseif config.reg_type == 2 then
            dfdx[{{config.ft_ind_start, config.ft_ind_end}}]:add(wd, x[{{config.ft_ind_start, config.ft_ind_end}}])
        else
            error('Unknown regularization type: ' .. config.reg_type)
        end
    end
    
    state.m:mul(beta1):add(1-beta1, dfdx)
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
    state.tmp:copy(state.v):sqrt():add(epsilon)
    
    state.t = state.t + 1
    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local clr = lr * math.sqrt(biasCorrection2)/biasCorrection1
    
    x:addcdiv(-clr, state.m, state.tmp)
    
    -- finetuning layer needs more update
    if config.ft_lr_mult > 1 then
        x[{{config.ft_ind_start, config.ft_ind_end}}]:addcdiv(
            -(config.ft_lr_mult-1)*clr, 
            state.m[{{config.ft_ind_start, config.ft_ind_end}}], 
            state.tmp[{{config.ft_ind_start, config.ft_ind_end}}])
    end
    
end

-- adam_l21 version
-- only update pre-finetuning layers
function optim_utils.adam_l21(x, dfdx, config, state)
    local beta1 = config.adam_beta1 or 0.9
    local beta2 = config.adam_beta2 or 0.999
    local epsilon = config.adam_epsilon or 1e-8
    local state = state or config
    local lr = config.learningRate or 1e-3
    local wd = config.weightDecay or 0
    
    local dfdx_tmp = dfdx[{{1, config.ft_ind_start-1}}]
    if not state.m then
        --initialization
        state.t = 0
        -- momentum1 m = beta1*m + (1-beta1)*dx
        state.m = x.new(#dfdx_tmp):zero()
        -- mementum2 v = beta2*v + (1-beta2)*(dx**2)
        state.v = x.new(#dfdx_tmp):zero()
        -- tmp tensor to hold the sqrt(v) + epsilon
        state.tmp = x.new(#dfdx_tmp):zero()
    end
    
    state.m:mul(beta1):add(1-beta1, dfdx_tmp)
    state.v:mul(beta2):addcmul(1-beta2, dfdx_tmp, dfdx_tmp)
    state.tmp:copy(state.v):sqrt():add(epsilon)
    
    state.t = state.t + 1
    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local clr = lr * math.sqrt(biasCorrection2)/biasCorrection1
    
    x[{{1, config.ft_ind_start-1}}]:addcdiv(-clr, state.m, state.tmp)
end

-- input gradients of the fine-tuning layer
-- output new weights of the fine-tuning layer
function optim_utils.reg_l21(x, dfdx, config, state)
    local beta1 = config.adam_beta1 or 0.9
    local beta2 = config.adam_beta2 or 0.999
    local epsilon = config.adam_epsilon or 1e-8
    local state = state or config
    local gamma = config.gamma_l21 or 1
    local wd = config.weightDecay or 0
    local fc7dim = config.fc7dim or 4096
    
    if not state.g then
        --initialization
        state.t = 0
        state.g = x.new(config.ft_ind_end - config.ft_ind_start + 1):zero()
    end
    
    state.t = state.t + 1
    
    -- update average gradients
    state.g = state.g:mul(state.t - 1):div(state.t) + dfdx[{{config.ft_ind_start, config.ft_ind_end}}]:div(state.t)
    
    -- updating new x using closed-form solution
    
    for i=config.ft_ind_start,config.ft_ind_end,fc7dim+1 do
        local gi = i - config.ft_ind_start + 1
        local t1 = math.max(0, 1 - wd/torch.norm(state.g[{{gi,gi+fc7dim}}], 2))
        local t2 = -math.sqrt(state.t)*t1/gamma
        x[{{i,i+fc7dim}}]:copy(state.g[{{gi,gi+fc7dim}}]:mul(t2))
    end
    
end


return optim_utils
