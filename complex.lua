-- simple operations for complex numbers

complex = {}

function complex.new(x)
	local out=torch.zeros(x:nElement(),2)
	out[{{},1}]=x
	return out
end

function complex.abs(x)
	if x:nDimension() == 1 then
		return torch.abs(x)
	else
		return torch.norm(x,2,2)
	end
end

function complex.prod(x,y)
	z=torch.Tensor(x:size())
	if y:nDimension() == 1 then
		z[{{},1}] = torch.cmul(x[{{},1}],y)
		z[{{},2}] = torch.cmul(x[{{},2}],y)
	else
		z[{{},1}] = torch.cmul(x[{{},1}],y[{{},1}]) - torch.cmul(x[{{},2}],y[{{},2}])
		z[{{},2}] = torch.cmul(x[{{},1}],y[{{},2}]) + torch.cmul(x[{{},2}],y[{{},1}])
	end
	return z
end


function complex.prod_fprop(input,kernel,output)
   local nSamples = input:size(1)
   local nInputPlanes = input:size(2)
   local nOutputPlanes = kernel:size(1)
   output:zero()
   for s = 1,nSamples do
      for i = 1,nOutputPlanes do 
         for j = 1,nInputPlanes do
            complex.addcmul(input[s][j],kernel[i][j],output[s][i])
         end
      end
   end
end





function complex.dot(a,b)
   if not(a:dim() == 2 and a:size(2) == 2 and b:dim() == 2 and b:size(2) == 2) then
      error('Inputs have to be 2D Tensor of size Nx2 (complex 1D tensor)')
   end
   if a:size(1) ~= b:size(1) then
      error('Both inputs need to have same number of elements')
   end
   local c = torch.sum(complex.cmul(a,b, true), 1)
   return c
end


function complex.mm(a,b)
   if not(a:dim() == 3 and a:size(3) == 2 and b:dim() == 3 and b:size(3) == 2) then
      error('Inputs have to be 3D Tensor of size NxMx2 (complex 2D tensor)')
   end
   if a:size(2) ~= b:size(1) then
      error('Matrix-Matrix product requires NxM and MxP matrices.')
   end
   local c = torch.zeros(a:size(1), b:size(2), 2):typeAs(a)
   for i=1,c:size(1) do
      for j=1,c:size(2) do
         c[i][j] = complex.dot(a[{i,{},{}}], b[{{},j,{}}])
         -- print(c[i][j])
      end
   end
return c
end

-- component-wise product of x and y, add the result to z
function complex.addcmul(x,y,z)
	local dim = x:nDimension()
	if not ((y:nDimension() == dim) and (z:nDimension() == dim)) then
		error('all inputs should have the same dimension')
	end
	local xReal = x:select(dim,1)
	local xImag = x:select(dim,2)
	local yReal = y:select(dim,1)
	local yImag = y:select(dim,2)	
	local zReal = z:select(dim,1)
	local zImag = z:select(dim,2)
	
	-- real part
	zReal:addcmul(xReal,yReal)
	zReal:addcmul(-1,xImag,yImag)
	-- complex part
	zImag:addcmul(xReal,yImag)
	zImag:addcmul(xImag,yReal)

end

function complex.cmul(x,y)
   local out = torch.zeros(x:size())
   complex.addcmul(x,y,out)
   return out
end


-- conjugate x in-place
function complex.conj(x)
	x:select(x:nDimension(),2):mul(-1)
end

-- returns w^1, w^2,...,w^k where w is the nth root of unity
function complex.roots(k,n)
	local roots = torch.Tensor(k,2)
	for i=0,k-1 do
		roots[i+1][1]=math.cos(-2*math.pi*i/n)
		roots[i+1][2]=math.sin(-2*math.pi*i/n)
	end
	return roots
end
	













