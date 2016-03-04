local eval_utils = torch.class('eval_utils')

-- Initialization
function eval_utils:__init()
    self:reset()
end

function eval_utils:reset()
    self.threshold_values = torch.range(0.1, 0.9, 0.1)
    self.num_correct = torch.zeros(self.threshold_values:size(1))
    self.num_pred = torch.zeros(self.threshold_values:size(1))
    self.num_gold = 0
end

function eval_utils:cal_precision_recall(batch_output, batch_label)

    local batch_size = batch_output:size(1)
    
    for i=1, batch_size do
        local pred = batch_output[i]
        local gold = batch_label[i]
        self.num_gold = self.num_gold + gold:nonzero():size(1)

        for j = 1,self.threshold_values:size(1) do
            local t = self.threshold_values[j]

            local pred_t = pred:gt(t):float():nonzero()
            if pred_t:dim() > 0 then
                self.num_pred[j] = self.num_pred[j] + pred_t:size(1)
            end

            local correct_t = pred[gold]:gt(t):float():nonzero()
            if correct_t:dim() > 0 then
                self.num_correct[j] = self.num_correct[j] + correct_t:size(1)
            end
        end
    end   
end

function eval_utils:print_precision_recall()
    for t=1, self.threshold_values:size(1) do
        local precision = self.num_correct[t] / self.num_pred[t]
        local recall = self.num_correct[t] / self.num_gold
        local fscore = 2 * precision * recall / (precision + recall)

        print(string.format('t=%.6f: precision/recall/f-score %.4f/%.4f/%.4f (%d/%d/%d)', 
                self.threshold_values[t], precision, recall, fscore, 
                self.num_correct[t], self.num_pred[t], self.num_gold))
    end
end

return eval_utils

