---------------------------------------------------------
-- functions for loading/saving memory-mapped files
---------------------------------------------------------

function torch.saveMemoryFile(filename,data)
	local offset
	local defaultType = torch.getdefaulttensortype()
	torch.setdefaulttensortype(data:type())
	
	if data:type() == 'torch.DoubleTensor' then
		offset = 13
	elseif data:type() == 'torch.FloatTensor' then
       offset = 27
       --offset = 8
       --offset = 31
	elseif data:type() == 'torch.ByteTensor' then
       offset = 107
       --offset = 4
	else
		error('not implemented for this type')
	end
	-- allocate space (removing Torch offset)
	torch.save(filename, torch.Tensor(data:nElement() - offset))
	-- copy the data
	local storage = torch.Storage(filename, true)
    print(storage:size() - data:nElement())
	storage:copy(data:storage())
	-- reset the type
	torch.setdefaulttensortype(defaultType)
end

-- load a memory-mapped file (this may need to be resized)
function torch.loadMemoryFile(filename,dataType)
	local defaultType = torch.getdefaulttensortype()
	torch.setdefaulttensortype(dataType)
	local storage = torch.Storage(filename, true)
	local data = torch.Tensor(storage)
	torch.setdefaulttensortype(defaultType)
	return data
end



--this is how to get the offset size

