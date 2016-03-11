--[[
This program test reading a csr (compressed sparse row) matrix
]]

cjson = require 'cjson'
require 'hdf5'

-- Settings
local coco_data_root = '/home/plsang/data/Microsoft_COCO'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a caffe model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-h5_file', paths.concat(coco_data_root, 'mscoco2014_train_mydepsv4.h5'), 'h5 file')
cmd:option('-jsonl_file', paths.concat(coco_data_root, 'mscoco2014_train_mydepsv4.jsonl'), 'jsonl file')
cmd:option('-vocab_file', paths.concat(coco_data_root, 'mscoco2014_train_mydepsv4vocab.json'), 'vocab file')
cmd:option('-concept_type', 'coarse_dependencies', 'type of concepts: coarse_lemmas, coarse_dependencies')
cmd:option('-id', 'image_id', 'id key')
  
local function read_json(path)
    local file = io.open(path, 'r')
    local text = file:read()
    file:close()
    local info = cjson.decode(text)
    return info
end

cmd:text()
local opt = cmd:parse(arg)

print(opt)

if not paths.filep(opt.vocab_file) then 
    print(string.format('Vocab file %s does not exist', opt.vocab_file))
    os.exit()
end
    
print('loading vocab file ', opt.vocab_file)
local vocab = read_json(opt.vocab_file)

assert(paths.filep(opt.jsonl_file), 'Jsonl file does not exist ' .. opt.jsonl_file)
print('loading jsonl file ', opt.jsonl_file)
concepts = {}
for line in io.lines(opt.jsonl_file) do 
    concepts[#concepts + 1] = cjson.decode(line)
end

print('loading h5 file ', opt.h5_file)
h5_data = hdf5.open(opt.h5_file, 'r')

-- Reading data of class STRING is unsupported by torch-hdf5
-- h5_vocab = h5_data:read('/vocab'):all()

local start = os.clock()

for i=1,#concepts do
    row_indptr = h5_data:read('/data/indptr'):partial({i,i+1})
    if row_indptr[1] == row_indptr[2] then 
	print('file with no concept, i =', i)
	print(concepts[i][opt.concept_type])
    else
    col_indptr = h5_data:read('/data/indices'):partial({row_indptr[1]+1, row_indptr[2]})
    
    local function inTable(tbl, item)
        for key, value in pairs(tbl) do
            if value == item then return true end
        end
        return false
    end
    
    for j=1,col_indptr:size(1) do
        if not inTable(concepts[i][opt.concept_type], vocab[col_indptr[j]+1]) then
            print(string.format("image_id [%d]: concept %s not found in the vocab", 
                concepts[i][opt.id], vocab[col_indptr[j]+1]))
            os.exit()
        end    
    end
    end
--    local index = torch.ByteTensor(1000):zero()
--    print(col_indptr)
--    print(col_indptr:long():view(1,-1))
--    index:scatter(1, col_indptr:long():add(1):view(1,-1), 1)
--    print(index)
--    break
end

print('Test passed!')
print(string.format("Time: %.2fs\n", os.clock() - start))

