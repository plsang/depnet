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
    
    if wd ~= 0 then
        -- regularization only at the finetuned layer
        if config.reg_type == 1 then
            dfdx[{{ws,we}}]:add(torch.sign(x[{{ws,we}}]):mul(wd))
        elseif config.reg_type == 2 then
            dfdx[{{ws,we}}]:add(wd, x[{{ws,we}}])
        elseif config.reg_type == 3 then
            for i=ws,we,fc7dim do
                local xi = x[{{i,i+fc7dim-1}}]
                local ni = torch.norm(xi,2)
                if ni > 0 then
                    local d_xi = xi:div(ni)
                    dfdx[{{i,i+fc7dim-1}}]:add(wd, d_xi)
                else
                    print('warning: zero norm detected')
                end
            end
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
