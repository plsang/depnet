--[[
Class to load COCO data in batch mode
]]--

require 'hdf5'
local cjson = require 'cjson'

local CocoData = torch.class('CocoData')

-- Initialization
function CocoData:__init(opt)
    -- Iterator
    self.iterator = 1
    
    self.images_key = '/images' 
    if opt.images_key then 
        self.images_key = opt.images_key
    end
    
    -- print('Loading image data: ', opt.image_file_h5)
    self.image_data = hdf5.open(opt.image_file_h5, 'r')
    
    local image_data_size = self.image_data:read(self.images_key):dataspaceSize()
    assert(#image_data_size == 4, '/images is a 4D tensor')
    assert(image_data_size[3] == image_data_size[4], 'image width and height are not equal')
    
    if not opt.noshuffle then
        self.image_shuffle = self.image_data:read('/index_shuffle'):all()
    end
    
    self.num_images = image_data_size[1]
    self.num_channels = image_data_size[2]
    self.image_size = image_data_size[3]
    
    if opt.label_file_h5 then
        print('Loading label data: ', opt.label_file_h5)
        self.label_data = hdf5.open(opt.label_file_h5, 'r')
        local label_data_size = self.label_data:read('/data/shape'):all()
        assert(self.num_images == label_data_size[1], 'number of images is different between image and label data')
    
        self.num_target = label_data_size[2]
        print('Number of target concepts: ', self.num_target)
        self.has_label = true
    else
        self.has_label = false
        self.num_target = opt.num_target
    end
    
    if opt.index_json then
        -- load input json
        print('Loading input json that contains index', opt.index_json)
        local fh = io.open(opt.index_json, 'r')
        local json_data = fh:read()
        fh:close()
        self.indexes = cjson.decode(json_data)
    end
    --
    self.batch_size = opt.batch_size
    
end

function CocoData:close()
    if self.image_data then self.image_data:close() end
    if self.label_data then self.label_data:close() end
end

function CocoData:getNumImages()
    return self.num_images
end

function CocoData:getNumTargets()
    return self.num_target
end

function CocoData:getIndex(idx)
    return self.indexes[idx]
end

function CocoData:getBatch(label_only)
    local image_batch = label_only or torch.FloatTensor(self.batch_size, self.num_channels, self.image_size, self.image_size)
    local label_batch = torch.ByteTensor(self.batch_size, self.num_target)
    
    local idx = self.iterator
    
    for i=1, self.batch_size do
        -- check if image indexes are matched
        
        if not label_only then
            -- fetch the image from h5
            image_batch[i] = self.image_data:read(self.images_key):partial({idx,idx},{1, self.num_channels},
                                {1, self.image_size},{1,self.image_size})
        end
        
        if self.has_label then
            local img_id1 = tonumber(self.indexes[idx])
            assert(img_id1, 'currently depnet is supported for training on MS COCO dataset only')
            local shuffle_idx = self.image_shuffle[idx] + 1
            local img_id2 = self.label_data:read('/index'):partial({shuffle_idx, shuffle_idx})
            assert(img_id1 == img_id2[1], 'image ids not matched!!!')
            
            -- fetch label from h5 (FROM SHUFFLED INDEX)
            local label_idx = torch.ByteTensor(1, self.num_target):zero() -- by default, torch does not initialize tensor
            local row_indptr = self.label_data:read('/data/indptr'):partial({shuffle_idx, shuffle_idx+1})
            if row_indptr[1] < row_indptr[2] then -- some row/image has no concept. this would prevent this case
                local col_indptr = self.label_data:read('/data/indices'):partial({row_indptr[1]+1, row_indptr[2]})
                label_idx:scatter(2, col_indptr:long():add(1):view(1,-1), 1) -- add 1 to col_ind (Lua index starts at 1)
            end
            label_batch[i] = label_idx
        end
        
        idx = idx + 1
        if idx > self.num_images then idx = 1 end
    end
    
    self.iterator = idx

    local data = {}
    data.images = image_batch
    data.labels = label_batch
    return data
end

function CocoData:reset()
    self.iterator = 1
end

function CocoData:getCurrentIndex()
    return self.iterator
end

function CocoData:setCurrentIndex(index)
    self.iterator = index
end

