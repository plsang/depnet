local optim_utils = {}

function optim_utils.sgd(x, dfdx, config, state)
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

    local x_nonfrozen = x[{{config.nonfrozen_start, config.nonfrozen_end}}]
    local dfdx_nonfrozen = dfdx[{{config.nonfrozen_start, config.nonfrozen_end}}]
   
     -- regularization on weights only
    if wd ~= 0 then
        if config.reg_type == 1 then
            for i=1,#config.weight_indices,2 do 
                local wi_s = config.weight_indices[i] - config.frozen_end
                local wi_e = config.weight_indices[i+1] - config.frozen_end
                dfdx_nonfrozen[{{wi_s, wi_e}}]:add(torch.sign(x_nonfrozen[{{wi_s, wi_e}}]):mul(wd))
            end
        elseif config.reg_type == 2 then
            for i=1,#config.weight_indices,2 do 
                local wi_s = config.weight_indices[i] - config.frozen_end
                local wi_e = config.weight_indices[i+1] - config.frozen_end
                dfdx_nonfrozen[{{wi_s, wi_e}}]:add(wd, x_nonfrozen[{{wi_s, wi_e}}])
            end
        else
            error('Unknown regularization type: ' .. config.reg_type)
        end
    end

    -- apply momentum
    if mom ~= 0 then
        if not state.dfdx then
            state.dfdx = torch.Tensor():typeAs(dfdx_nonfrozen):resizeAs(dfdx_nonfrozen):copy(dfdx_nonfrozen)
        else
            state.dfdx:mul(mom):add(1-damp, dfdx_nonfrozen)
        end
        if nesterov then
            dfdx_nonfrozen:add(mom, state.dfdx)
        else
            dfdx_nonfrozen = state.dfdx
        end
    end

    -- learning rate decay (annealing)
    local clr = lr / (1 + nevals*lrd)

    -- parameter update, this apply the base learning rate
    x_nonfrozen:add(-clr, dfdx_nonfrozen)
    
    -- update bias twice
    for i=1,#config.bias_indices,2 do 
        local bi_s = config.bias_indices[i] - config.frozen_end
        local bi_e = config.bias_indices[i+1] - config.frozen_end
        x_nonfrozen[{{bi_s, bi_e}}]:add(-clr, dfdx_nonfrozen[{{bi_s, bi_e}}])
    end
    
    -- finetuning layer needs more update
    if config.w_lr_mult > 1 and config.b_lr_mult > 3 then
        
        -- update weight index
        local ft_ind_start = config.ft_ind_start - config.frozen_end
        local ft_ind_end = config.ft_ind_end - config.frozen_end
        
        -- update bias index
        local ftb_ind_start = config.ftb_ind_start - config.frozen_end
        local ftb_ind_end = config.ftb_ind_end - config.frozen_end

        x_nonfrozen[{{ft_ind_start, ft_ind_end}}]:add(-(config.w_lr_mult-1)*clr, dfdx_nonfrozen[{{ft_ind_start, ft_ind_end}}])
        x_nonfrozen[{{ftb_ind_start, ftb_ind_end}}]:add(-(config.b_lr_mult-3)*clr, dfdx_nonfrozen[{{ftb_ind_start, ftb_ind_end}}])
    end
    
    -- copy update back to x
    x[{{config.nonfrozen_start, config.nonfrozen_end}}]:copy(x_nonfrozen)
    
    -- update evaluation counter
    state.evalCounter = state.evalCounter + 1
    
    -- delete tmp variables
    x_nonfrozen = nil
    dfdx_nonfrozen = nil
end

function optim_utils.adam(x, dfdx, config, state)
    local beta1 = config.adam_beta1 or 0.9
    local beta2 = config.adam_beta2 or 0.999
    local epsilon = config.adam_epsilon or 1e-8
    local state = state or config
    local lr = config.learningRate or 1e-3
    local wd = config.weightDecay or 0
    local ws = config.ft_ind_start       -- start index of finetuned weight
    local we = config.ftb_ind_start - 1  -- end index of finetuned weight
    local fc7dim = config.fc7dim or 4096
    
    local x_nonfrozen = x[{{config.nonfrozen_start, config.nonfrozen_end}}]
    local dfdx_nonfrozen = dfdx[{{config.nonfrozen_start, config.nonfrozen_end}}]
    
    if not state.m then
        --initialization
        state.t = 0
        -- momentum1 m = beta1*m + (1-beta1)*dx
        state.m = x.new(#dfdx_nonfrozen):zero()
        -- mementum2 v = beta2*v + (1-beta2)*(dx**2)
        state.v = x.new(#dfdx_nonfrozen):zero()
        -- tmp tensor to hold the sqrt(v) + epsilon
        state.tmp = x.new(#dfdx_nonfrozen):zero()
    end
    
    if wd ~= 0 then
        -- regularization on weights only
        if config.reg_type == 1 then
            for i=1,#config.weight_indices,2 do 
                local wi_s = config.weight_indices[i] - config.frozen_end
                local wi_e = config.weight_indices[i+1] - config.frozen_end
                dfdx_nonfrozen[{{wi_s, wi_e}}]:add(torch.sign(x_nonfrozen[{{wi_s, wi_e}}]):mul(wd))
            end
        elseif config.reg_type == 2 then
            for i=1,#config.weight_indices,2 do 
                local wi_s = config.weight_indices[i] - config.frozen_end
                local wi_e = config.weight_indices[i+1] - config.frozen_end
                dfdx_nonfrozen[{{wi_s, wi_e}}]:add(wd, x_nonfrozen[{{wi_s, wi_e}}])
            end
        else
            error('Unknown regularization type: ' .. config.reg_type)
        end
    end
    
    state.m:mul(beta1):add(1-beta1, dfdx_nonfrozen)
    state.v:mul(beta2):addcmul(1-beta2, dfdx_nonfrozen, dfdx_nonfrozen)
    state.tmp:copy(state.v):sqrt():add(epsilon)
    dfdx_nonfrozen = nil
    
    state.t = state.t + 1
    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local clr = lr * math.sqrt(biasCorrection2)/biasCorrection1
    
    -- parameter update
    x_nonfrozen:addcdiv(-clr, state.m, state.tmp)
    
    -- update bias twice
    for i=1,#config.bias_indices,2 do 
        local bi_s = config.bias_indices[i] - config.frozen_end
        local bi_e = config.bias_indices[i+1] - config.frozen_end
        x_nonfrozen[{{bi_s, bi_e}}]:addcdiv(-clr, state.m[{{bi_s, bi_e}}], state.tmp[{{bi_s, bi_e}}])
    end
    
    -- finetuning layer needs more update
    if config.w_lr_mult > 1 and config.b_lr_mult > 3 then
        
        -- update weight index
        local ft_ind_start = config.ft_ind_start - config.frozen_end
        local ft_ind_end = config.ft_ind_end - config.frozen_end
        
        -- update bias index
        local ftb_ind_start = config.ftb_ind_start - config.frozen_end
        local ftb_ind_end = config.ftb_ind_end - config.frozen_end

        x_nonfrozen[{{ft_ind_start, ft_ind_end}}]:addcdiv(
            -(config.w_lr_mult-1)*clr, 
            state.m[{{ft_ind_start, ft_ind_end}}], 
            state.tmp[{{ft_ind_start, ft_ind_end}}])
        
        x_nonfrozen[{{ftb_ind_start, ftb_ind_end}}]:addcdiv(
            -(config.b_lr_mult-3)*clr, 
            state.m[{{ftb_ind_start, ftb_ind_end}}], 
            state.tmp[{{ftb_ind_start, ftb_ind_end}}])
    end
    
    -- copy update back to x
    x[{{config.nonfrozen_start, config.nonfrozen_end}}]:copy(x_nonfrozen)
    x_nonfrozen = nil
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
    local ws = config.ft_ind_start       -- start index of finetuned weight
    local we = config.ftb_ind_start - 1  -- end index of finetuned weight
    local fc7dim = config.fc7dim or 4096
    
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
    
    state.m:mul(beta1):add(1-beta1, dfdx)
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
    state.tmp:copy(state.v):sqrt():add(epsilon)
    
    state.t = state.t + 1
    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local clr = lr * math.sqrt(biasCorrection2)/biasCorrection1
    
    
    --x:addcdiv(-clr, state.m, state.tmp)
    x[{{1, config.ft_ind_start-1}}]:addcdiv(-clr, 
        state.m[{{1, config.ft_ind_start-1}}], 
        state.tmp[{{1, config.ft_ind_start-1}}])
    
    -- finetuning layer needs more update
    if config.ft_lr_mult > 1 then
        -- update of bias is same 
        x[{{config.ftb_ind_start, config.ftb_ind_end}}]:addcdiv(
            -config.ft_lr_mult*clr, 
            state.m[{{config.ftb_ind_start, config.ftb_ind_end}}], 
            state.tmp[{{config.ftb_ind_start, config.ftb_ind_end}}])
        
        -- update of weights is different
        for i=config.ft_ind_start,config.ftb_ind_start-1,fc7dim do
            local t1 = x[{{i,i+fc7dim-1}}] - torch.cdiv(state.m[{{i,i+fc7dim-1}}], 
                state.tmp[{{i,i+fc7dim-1}}]):mul(config.ft_lr_mult*clr)
            local t2 = math.max(0, 1 - wd/torch.norm(t1, 2))
            x[{{i,i+fc7dim-1}}]:mul(t1, t2) -- x = t2*t1
        end
    end
    
end

return optim_utils
