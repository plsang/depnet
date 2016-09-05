--[[
Class to load COCO data in batch mode
]]--

require 'hdf5'
local cjson = require 'cjson'
local t = require '../fb.resnet.torch/datasets/transforms'

local VideoData = torch.class('VideoData')

--[[
preprocesing for VGGNet
input is mage is RGB: 3x240x320
1. RGB--> BGR
2. Mean subtraction
3. RandomCrop
4. 
not that VGG caffe's implemenation is Crop (img - mean) 
cf. https://github.com/KaimingHe/deep-residual-networks/issues/5
--]]
local bgr_mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68}):view(3, 1, 1):expand(3, 240, 320)
local flow_mean_pixel = torch.FloatTensor({128}):expand(20):contiguous():view(20, 1, 1):expand(20,240,320)
local perm = torch.LongTensor{3, 2, 1}
local vgg_transform_train = t.Compose{
    t.RandomCrop(224),
    t.HorizontalFlip(0.5),
}
local vgg_transform_test = t.Compose{
    t.CenterCrop(224),
}
function preprocess_vgg(img, mode, num_img_channel)
    
    if num_img_channel == 3 then
        -- convert RGB --> BGR
        img = img:index(1, perm):float()
        -- mean subtraction
        img:add(-1, bgr_mean_pixel)
    else
        -- flow image
        img:add(-1, flow_mean_pixel)
    end
    
    if mode == 'train' then
        img = vgg_transform_train(img)
    else
        img = vgg_transform_test(img)
    end
    
    return img
end

-- Initialization
function VideoData:__init(opt)
    -- Iterator
    self.iterator = 1
    self.epoch = 1
    self.batch_size = opt.batch_size or 1
    self.crop_size = opt.crop_size or 224
    self.image_height = opt.image_height or 240
    self.image_width = opt.image_width or 320
    self.num_img_channel = opt.num_img_channel or 3
    self.mode = opt.mode or 'train' -- train/test mode: use to do data augmentation/preprocessing
    self.index = torch.Tensor() -- order of video index, to be randomly shuffled every epoch
    
    -- print('Loading image data: ', opt.image_file_h5)
    self.video_data = hdf5.open(opt.image_file_h5, 'r')
    
    -- local image_data_size = self.image_data:read(self.images_key):dataspaceSize()
    
    if opt.label_file_h5 then
        print('Loading label data: ', opt.label_file_h5)
        self.label_data = hdf5.open(opt.label_file_h5, 'r')
        local label_data_size = self.label_data:read('/data/shape'):all()
        
        self.num_videos = label_data_size[1]
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
        
        self.ids = cjson.decode(json_data)
        assert(self.num_videos == #self.ids, 'number of images is different between image and label data')
    end
    
    -- 
    self:shuffle_videos()
    self.num_frames = self:countFrames()
    self.label_map = self:getLabelMap()
end

function VideoData:getBatch()
    local video_batch = torch.FloatTensor(self.batch_size, self.num_img_channel, self.crop_size, self.crop_size)
    local label_batch = torch.ByteTensor(self.batch_size, self.num_target)
    
    local counter = self.iterator
    
    for i=1, self.batch_size do
        local idx = self.index[counter]
        local vid = self.ids[idx]
        local num_frame = self.num_frames[idx]
        
        if self.num_img_channel == 3 then
            local frame_idx = torch.random(1, num_frame)
            -- fetch the image from h5
            local img = self.video_data:read(vid):partial({frame_idx,frame_idx},{1, self.num_img_channel},
                {1, self.image_height},{1,self.image_width})
            -- apply transformation
            video_batch[i] = preprocess_vgg(img[1], self.mode, self.num_img_channel)
        else
            local frame_idx = torch.random(1, num_frame - self.num_img_channel)
            -- fetch the image from h5
            local img = self.video_data:read(vid):partial({frame_idx, frame_idx+self.num_img_channel-1},
                {1, self.image_height},{1,self.image_width})
            -- apply transformation
            video_batch[i] = preprocess_vgg(img:float(), self.mode, self.num_img_channel)
        end
        
        if self.has_label then
            
            local label_idx = self.label_map[vid]
            local label_vid = self.label_data:read('/index'):partial({label_idx, label_idx})
            assert(vid == tostring(label_vid[1]), 'video ids are not matched!!!')
            
            -- fetch label from h5 (FROM SHUFFLED INDEX)
            local label = torch.ByteTensor(1, self.num_target):zero() -- by default, torch does not initialize tensor
            local row_indptr = self.label_data:read('/data/indptr'):partial({label_idx, label_idx+1})
            if row_indptr[1] < row_indptr[2] then -- some row/image has no concept. this would prevent this case
                local col_indptr = self.label_data:read('/data/indices'):partial({row_indptr[1]+1, row_indptr[2]})
                label:scatter(2, col_indptr:long():add(1):view(1,-1), 1) -- add 1 to col_ind (Lua index starts at 1)
            end
            label_batch[i] = label
        end
        
        counter = counter + 1
        if counter > self.num_videos then 
            counter = 1 
            self.epoch = self.epoch + 1
            self:shuffle_videos()
        end
    end
    
    self.iterator = counter

    local data = {}
    data.images = video_batch
    data.labels = label_batch
    return data
end

--[[
Count the number of frames in each video
count['video_id'] = num_frames of this video
--]]
function VideoData:countFrames()
    local counts = {}
    for k,v in ipairs(self.ids) do
        local data_size = self.video_data:read(v):dataspaceSize()
        counts[k] = data_size[1]
    end
    return counts
end

--[[
Get the label map of provided label file
for each video_id, return its index in the label file
--]]
function VideoData:getLabelMap()
    local map = {}
    local label_index = self.label_data:read('index'):all()
    for idx=1,label_index:size(1) do
        local vid = tostring(label_index[idx])
        map[vid] = idx
    end
    return map
end

function VideoData:reset()
    self.iterator = 1
end

function VideoData:getCurrentIndex()
    return self.iterator
end

function VideoData:setCurrentIndex(index)
    self.iterator = index
end

function VideoData:close()
    if self.image_data then self.image_data:close() end
    if self.label_data then self.label_data:close() end
end

function VideoData:getNumVideos()
    return self.num_videos
end

function VideoData:getNumTargets()
    return self.num_target
end

function VideoData:getVideoId(counter)
    return self.ids[counter]
end

function VideoData:shuffle_videos()
    self.index = torch.randperm(self.num_videos)
end    