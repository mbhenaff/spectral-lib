-- make interpolation kernel, which can be linear, spline, dyadic spline or spatial
require 'image'

kernels_path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/interp_kernels/'
-- for all except spatial2D, N is the input size, M is the output size
-- for spatial2D, we are assuming inputs are NxN images and outputs are MxM
function interpKernel(N, M, type)
   if type == 'spline' then
      return torch.load(kernels_path .. '/spline_kernel_' .. N .. '_' .. M .. '.th'):float()
   elseif type == 'spline_border' then
      return torch.load(kernels_path .. '/spline_border_' .. N .. '_' .. M .. '.th'):float()
   elseif type == 'dyadic_spline' then
      return torch.load(kernels_path .. '/dyadic_spline_kernel_' .. N .. '_' .. M .. '.th'):float()
   elseif type == 'spatial2D' then 
      return torch.load(kernels_path .. '/spatial_kernel_' .. N .. '_' .. M .. '.th'):float()
   elseif type == 'bilinear' then
      if N == M then
         return torch.eye(N)
      else
         local out = torch.zeros(N,M)
         local aux = torch.zeros(N)
         local x = torch.linspace(0,1,N)
         local y = torch.linspace(0,1,M)
         for i = 1,N do
			aux:zero()
			aux[i] = 1
			out[{i,{}}]:copy(linterp(x,aux,y))
         end
         return out:float()
      end
   else
      error('unrecognized type')
   end
end

-- linear interpolation
function linterp(x1,y1,x2)
	local y2 = torch.Tensor(x2:nElement())
	if torch.max(x2) > torch.max(x1) or torch.min(x2) < torch.min(x1) then
		error('range of x2 should be included in range of x1')
	end
	for i=1,x2:nElement() do
		local indx = torch.find(torch.le(x1,x2[i]))
		local j = torch.max(indx)
		if j == x1:nElement() then
			y2[i] = y1[x1:nElement()]
		else
			local slope = (y1[j+1]-y1[j])/(x1[j+1]-x1[j])
			y2[i] = y1[j] + slope*(x2[i]-x1[j])
		end
	end
	return y2
end

		
		
function torch.find(x)
   if x:nDimension() > 1 then
		error('torch.find is only defined for 1D tensors')
	end
   local indx={}
   for i=1,(#x)[1] do
      if x[i]>0 then
         table.insert(indx,i)
      end
   end
   return torch.IntTensor(indx)
end
			
			








