torch.setdefaulttensortype('torch.FloatTensor')
require("image")
dofile('memoryMap.lua')
-- Dataset settings for training
data_dir = '/misc/vlgscratch3/LecunGroup/provodin/lagr/pp'
image_dir = data_dir..'/quality9075/LAGR2014_img_train/'
save_dir = '/misc/vlgscratch3/LecunGroup/mbhenaff/'
list = torch.load(data_dir.."/index/train.t7b")
load = true

--options
nclasses = 100
nsamples_per_class = 500
njitters = 9
nsample = nclasses*nsamples_per_class
height = 128
width = 128
rot_range = {-10,10}
dx_range = {-0.01,0.01}
dy_range = {-0.01,0.01}
scale_range = {-0.2, 0.2}
xsheer_range = {-0.2,0.2}
ysheer_range = {-0.2,0.2}
crop_method = 'center'


-- we choose the classes with the most samples
counts = torch.Tensor(1000)
for i = 1,1000 do 
   counts[i]=#list.files[i]
end
counts,classes = torch.sort(counts,true)
classes = classes[{{1,nclasses}}]

if true then

local function load_image(dir)
   local im = image.load(dir)
   -- Check channels
   if im:dim() == 2 then
      local new_im = torch.Tensor(3,im:size(1),im:size(2))
      for c = 1,3 do
         new_im:select(1,c):copy(im)
      end
      im = new_im
   elseif im:size(1) == 1 then
      local new_im = torch.Tensor(3,im:size(2),im:size(3))
      for c = 1,3 do
         new_im:select(1,c):copy(im:select(1,1))
      end
      im = new_im
   end
   if im:dim() ~= 3 or im:size(1) ~=3 then
      error("Image channel is not 3")
   end
   return im

end


-- distort function
function distort(i, deg_rot, scale, trans_v, trans_u, xsheer, ysheer)
   -- size:
   local height,width = i:size(2),i:size(3)
   -- x/y grids
   local grid_y = torch.ger( torch.linspace(-1,1,height), torch.ones(width) )
   local grid_x = torch.ger( torch.ones(height), torch.linspace(-1,1,width) )
   local flow = torch.FloatTensor()
   local flow_scale = torch.FloatTensor()
   local flow_rot = torch.FloatTensor()
   -- global flow:
   flow:resize(2,height,width)
   flow:zero()
   local rot_angle
   local rotmat

   -- Apply translation (comes before rotation)
   flow[1]:add(trans_v * height)
   flow[2]:add(trans_u * width)
   -- Apply scale and rotation
   flow_rot:resize(2,height,width)
   flow_rot[1] = grid_y * ((height-1)/2) * -1
   flow_rot[2] = grid_x * ((width-1)/2) * -1
   local view = flow_rot:reshape(2,height*width)

   local function rmat(deg, s)
      local r = deg/180*math.pi
      return torch.FloatTensor{{s * math.cos(r), -s * math.sin(r)},
         {s * math.sin(r), s * math.cos(r)}}
   end

   local function smat(xsheer, ysheer)
      return torch.FloatTensor{{1, xsheer},
         {ysheer,1}}
   end

   rotmat = rmat(deg_rot, 1 + scale)
   shemat = smat(xsheer, ysheer)
   flow_sheerr = torch.mm(shemat, view)
   flow_rotr = torch.mm(rotmat, view)
   flow_sheer = flow_rot - flow_sheerr:reshape(2, height, width)
   flow_rot = flow_rot - flow_rotr:reshape(2, height, width)
   flow:add(flow_rot)
   flow:add(flow_sheer)
   -- apply field
   local result = torch.FloatTensor()
   image.warp(result,i,flow,'lanczos')
   return result, rotmat
end



function crop(im, height, width, method, cropped)
   local cropped = cropped or torch.Tensor(3,height,width)
   local width,height = cropped:size(3),cropped:size(2)
   local cstartx,cstarty,cendx,cendy = 1,1,width,height
   local startx, starty,endx,endy
   -- Determine start and end indices based on type of crop
   if method == "center" or method == nil then
      startx = math.modf((im:size(3) - width)/2) + 1
      starty = math.modf((im:size(2) - height)/2) + 1
   elseif method == "random" then
      startx = math.random(math.min(1,im:size(3) - width + 1), math.max(1,im:size(3) - width + 1))
      starty = math.random(math.min(1,im:size(2) - height + 1), math.max(1,im:size(2) - height + 1))
   elseif method == "leftupper" then
      startx = 1
      starty = 1
   elseif method == "leftlower" then
      startx = 1
      starty = im:size(2) - height + 1
   elseif method == "rightupper" then
      startx = im:size(3) - width + 1
      starty = 1
   elseif method == "rightlower" then
      startx = im:size(3) - width + 1
      starty = im:size(2) - height + 1
   elseif method == "corners" then
      local method = math.random(5)
      if method == 1 then -- Center
         startx = math.modf((im:size(3) - width)/2) + 1
         starty = math.modf((im:size(2) - height)/2) + 1
      elseif method == 2 then -- LeftUpper
         startx = 1
         starty = 1
      elseif method == 3 then -- LeftLower
         startx = 1
         starty = im:size(2) - height + 1
      elseif method == 4 then -- RightUpper
         startx = im:size(3) - width + 1
         starty = 1
      elseif method == 5 then -- RightLower
         startx = im:size(3) - width + 1
         starty = im:size(2) - height + 1
      end
   else
      error("Unrecognized cropping method")
   end
   endx = startx + width - 1
   endy = starty + height - 1
   -- Centering the image patch
   cstartx = startx
   cstarty = starty
   -- Rectify the indices for image
   startx = math.max(startx,1)
   endx = math.min(endx,im:size(3))
   starty = math.max(starty,1)
   endy = math.min(endy,im:size(2))
   -- Rectify end indices for cropped
   cstartx = startx - cstartx + 1
   cstarty = starty - cstarty + 1
   cendx = cstartx + endx - startx
   cendy = cstarty + endy - starty
   cropped:fill(0)
   cropped[{{},{cstarty,cendy},{cstartx,cendx}}]:copy(im[{{},{starty,endy},{startx,endx}}])
   return cropped
end

function shuffle(data,labels)
   local n = data:size(1)
   local tmpdata = data[1]:clone()
   local tmplabel = labels[1]
   for i = 1,n do 
      if i % 1000 == 0 then print(i .. '/' .. n) end
      local j = math.random(1,n)
      tmpdata:copy(data[j])
      data[j]:copy(data[i])
      data[i]:copy(tmpdata)
      tmplabel = labels[j]
      labels[j] = labels[i]
      labels[i] = tmplabel
   end
end

function shuffle_jitter(data,labels)
   local n = data:size(1)
   local tmpdata = data[1]:clone()
   local tmplabel = labels[1]:clone()
   for i = 1,n do 
      if i % 1000 == 0 then print(i .. '/' .. n) end
      local j = math.random(1,n)
      tmpdata:copy(data[j])
      data[j]:copy(data[i])
      data[i]:copy(tmpdata)
      tmplabel:copy(labels[j])
      labels[j]:copy(labels[i])
      labels[i]:copy(tmplabel)
   end
end




if load then 
   print('loading data')
   x = torch.load(save_dir..'ImageNet_50kx10x128x128.t7')
   print('done')
   jitter_dataset = x.data
   -- shuffle so classes will be evenly distributed but jitters do not overlap in train/test sets
   shuffle_jitter(jitter_dataset, x.labels)
   if true then
   trdata = x.data[{{1,40000},{},{},{},{}}]
   tedata = x.data[{{40001,50000},{},{},{},{}}]
   trlabels = x.labels[{{1,40000},{}}]
   telabels = x.labels[{{40001,50000},{}}]
   trdata:resize(40000*10,3,128,128)
   tedata:resize(10000*10,3,128,128)
   trlabels:resize(40000*10)
   telabels:resize(10000*10)
   -- shuffle again so jitters do not appear consecutively
   shuffle(trdata,trlabels)
   shuffle(tedata,telabels)
   -- relabel so we have labels in 1,...,nclasses
   dict = {}
   for i = 1,nclasses do 
      dict[classes[i]]=i
   end
   for i=1,trlabels:size(1) do 
      trlabels[i]=dict[trlabels[i]]
   end

   for i=1,telabels:size(1) do 
      telabels[i]=dict[telabels[i]]
   end
   trdata1 = trdata[{{1,100000}}]:resize(100000*3*128*128)
   trdata2 = trdata[{{100001,200000}}]:resize(100000*3*128*128)
   trdata3 = trdata[{{200001,300000}}]:resize(100000*3*128*128)
   trdata4 = trdata[{{300001,400000}}]:resize(100000*3*128*128)
   tedata = tedata:resize(100000*3*128*128)

   torch.saveMemoryFile(save_dir .. '/imagenet_small/trdata1.th',trdata1:clone())
   collectgarbage()
   torch.saveMemoryFile(save_dir .. '/imagenet_small/trdata2.th',trdata2:clone())
   collectgarbage()
   torch.saveMemoryFile(save_dir .. '/imagenet_small/trdata3.th',trdata3:clone())
   collectgarbage()
   torch.saveMemoryFile(save_dir .. '/imagenet_small/trdata4.th',trdata4:clone())
   collectgarbage()
   torch.saveMemoryFile(save_dir .. '/imagenet_small/tedata.th',tedata:clone())
   collectgarbage()
   
   torch.save(save_dir .. '/imagenet_small/trlabels1.th',trlabels[{{1,100000}}])
   torch.save(save_dir .. '/imagenet_small/trlabels2.th',trlabels[{{100001,200000}}])
   torch.save(save_dir .. '/imagenet_small/trlabels3.th',trlabels[{{200001,300000}}])
   torch.save(save_dir .. '/imagenet_small/trlabels4.th',trlabels[{{300001,400000}}])
   torch.save(save_dir .. '/imagenet_small/telabels.th',telabels)

   
   --torch.saveMemoryFile(save_dir .. 'imagenet_train_400kx3x128x128_mm.t7',trdata)
   --torch.saveMemoryFile(save_dir .. 'imagenet_test_100kx3x128x128_mm.t7',tedata)
   end
else
   jitter_dataset = torch.Tensor(nsample,njitters+1,3,height,width)
   labels = torch.Tensor(nsample,njitters+1)
   k = 1
   for c = 1,nclasses do
      --progress(k,nsample)
      print('class ' .. c)
      class_idx = classes[c]
      for s = 1,nsamples_per_class do
         --file_idx = math.random(#list.files[class_idx])
         file_idx = s
         I = load_image(image_dir..list.files[class_idx][file_idx])
         Ic = crop(I,height,width,crop_method)
         jitter_dataset[k][1]:copy(Ic)
         labels[k]:copy(torch.Tensor(njitters+1):fill(class_idx))

         for i = 2,njitters+1 do
            p1 = math.random()
            p2 = math.random()
            p3 = math.random()
            p4 = math.random()
            p5 = math.random()
            p6 = math.random()

            rot = p1*rot_range[1] + (1-p1)*rot_range[2]
            scale = p2*scale_range[1] + (1-p2)*scale_range[2]
            dx = p3*dx_range[1] + (1-p3)*dx_range[2]
            dy = p4*dy_range[1] + (1-p4)*dy_range[2]
            xsheer = p5*xsheer_range[1] + (1-p5)*xsheer_range[2]
            ysheer = p6*ysheer_range[1] + (1-p6)*ysheer_range[2]
            Id = distort(I, rot, scale, dx, dy, xsheer, ysheer)
            Idc = crop(Id, height, width, 'center')
            jitter_dataset[k][i]:copy(Idc)

         end
         k = k + 1
      end
      collectgarbage()
   end
   -- shuffle the dataset 
   shuffle(jitter_dataset, labels)
end


end




--torch.save(save_dir..'ImageNet_50kx10x128x128.t7',jitter_dataset,labels)

--I = image.toDisplayTensor({input = jitter_dataset:resize((n+1)*njitters,3,114,114), nrow = njitters+1, padding = 1})
--image.save('./sample.png',I) 

