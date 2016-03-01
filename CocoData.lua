--[[
Class to load COCO data in batch mode
]]--

require 'hdf5'

local CocoData = torch.class('CocoData')

-- Initialization
function CocoData:__init(opt)
    -- Iterator
    self.iterator = 1
    
    print('Loading image data: ', opt.image_file_h5)
    self.image_data = hdf5.open(opt.image_file_h5, 'r')
    
    local image_data_size = self.image_data:read('/images'):dataspaceSize()
    assert(#image_data_size == 4, '/images is a 4D tensor')
    assert(image_data_size[3] == image_data_size[4], 'image width and height are not equal')
    
    self.image_shuffle = self.image_data:read('/index_shuffle'):all()
    self.num_images = image_data_size[1]
    self.num_channels = image_data_size[2]
    self.image_size = image_data_size[3]
    
    print('Loading label data: ', opt.label_file_h5)
    self.label_data = hdf5.open(opt.label_file_h5, 'r')
    local label_data_size = self.label_data:read('/data/shape'):all()
    assert(self.num_images == label_data_size[1], 'number of images is different between image and label data')
    
    self.num_target = label_data_size[2]
    print('Number of target concepts: ', self.num_target)
    --
    self.batch_size = opt.batch_size
    
end

function CocoData:getBatch(opt)
    local image_batch = torch.ByteTensor(self.batch_size, self.num_channels, self.image_size, self.image_size)
    local label_batch = torch.ByteTensor(self.batch_size, self.num_target)
    local idx = self.iterator
    
    for i=1, self.batch_size do
        -- check if image indexes are matched
        local img_id1 = self.image_data:read('/index'):partial({idx, idx})
        local shuffle_idx = self.image_shuffle[idx] + 1
        
        local img_id2 = self.label_data:read('/index'):partial({shuffle_idx, shuffle_idx})
        assert(torch.all(torch.eq(img_id1, img_id2)), 'image id not matched!!!')
        
        -- fetch the image from h5
        local img = self.image_data:read('/images'):partial({idx,idx},{1, self.num_channels},
                            {1, self.image_size},{1,self.image_size})
        image_batch[i] = img
        
        -- fetch label from h5 (FROM SHUFFLED INDEX)
        local row_indptr = self.label_data:read('/data/indptr'):partial({shuffle_idx, shuffle_idx+1})
        local col_indptr = self.label_data:read('/data/indices'):partial({row_indptr[1]+1, row_indptr[2]})
        
        local label_idx = torch.ByteTensor(1, self.num_target):zero() -- by default, torch does not initialize tensor
        label_idx:scatter(2, col_indptr:long():add(1):view(1,-1), 1) -- add 1 to col_ind (Lua index starts at 1)
        label_batch[i] = label_idx
        
        idx = idx + 1
        if idx > self.num_images then idx = 1 end
    end
    
    self.iterator = idx
    
    local data = {}
    data.images = image_batch
    data.labels = label_batch
    return data
end