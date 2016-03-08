local optim_utils = {}

function optim_utils.sgd_config(model, opt)
    local sgd_config = {
        learningRate = opt.learning_rate,
        weightDecay = opt.weight_decay,
        momentum = opt.momentum,
        learningRateDecay = opt.learning_rate_decay,
        w_lr_mult = opt.w_lr_mult,
        b_lr_mult = opt.b_lr_mult   
    }

    local frozen_graph = model.modules[1]
    assert(frozen_graph.frozen == true)

    sgd_config.frozen_start = 1

    local total_elements = 0
    for _, m in ipairs(frozen_graph.modules) do
        if m.weight and m.bias then
            local wlen = m.weight:nElement()
            local blen = m.bias:nElement()
            local mlen = wlen + blen
            total_elements = total_elements + mlen
        end
    end
    
    sgd_config.frozen_end = total_elements

    local finetune_graph = model.modules[2]
    assert(finetune_graph.frozen == false)

    local bias_indices = {}
    for _, m in ipairs(finetune_graph.modules) do
       if m.weight and m.bias then

            local wlen = m.weight:nElement()
            local blen = m.bias:nElement()
            local mlen = wlen + blen
            table.insert(bias_indices, total_elements + wlen + 1)
            table.insert(bias_indices, total_elements + mlen)

            if m.name == opt.finetune_layer_name then
                print('Fine tuning layer found!')
                sgd_config.ft_ind_start = total_elements + 1
                sgd_config.ft_ind_end = total_elements + mlen
                sgd_config.ftb_ind_start = total_elements + wlen + 1 -- fine tune bias index start
                sgd_config.ftb_ind_end = total_elements + mlen       -- fine tune bias index end
            end

            total_elements = total_elements + mlen
       elseif m.weight or m.bias then
           error('Layer that has either weight or bias')     
       end
    end

    sgd_config.bias_indices = bias_indices
    assert(sgd_config.ft_ind_start, 'Fine tuning layer not found')
    
    return sgd_config
end

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
    local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
    
    x:addcdiv(-stepSize, state.m, state.tmp)
end


return optim_utils
