--require 'fftw'
--require 'cunn'
require 'cutorch'
require 'libcufft'
require 'spectralcuda'
--require 'libFFTconv'
dofile('../complex.lua')
cufft = dofile('cufft.lua')




function make_hermitian_weights(nOutputPlanes,nInputPlanes,iH,iW)
    local spatial_weights = torch.zeros(nOutputPlanes,nInputPlanes,iH,iW,2)
    spatial_weights:select(5,1):copy(torch.randn(nOutputPlanes,nInputPlanes,iH,iW))
    local weights = torch.CudaTensor(nOutputPlanes,nInputPlanes,iH,iW,2):zero()
    cufft.fft2d_c2c(spatial_weights:cuda(),weights,1)
    return weights
 end

function test1d()
   nInputLines = 2
   N = 8
   -- real to complex
   x=torch.range(0,nInputLines*N-1):resize(nInputLines,N)
   y=torch.CudaTensor(nInputLines,N/2+1,2)
   x=x:cuda()
   y=y:cuda()
   print('original input:') print(x)
   cufft.fft1d_r2c(x,y)
   print('transform:') print(y)
   x:zero()
   cufft.fft1d_c2r(y,x)
   print('reconstruction:') print(x)

   -- complex to complex
   print('complex to complex')
   x=torch.CudaTensor(nInputLines,N,2):zero()
   y=torch.CudaTensor(nInputLines,N,2):zero()
   x[{{},{},1}]:copy(torch.range(0,nInputLines*N-1))
   print('original input:') print(x)
   cufft.fft1d_c2c(x,y,1)
   print('transform:') print(y)
   x:zero()
   cufft.fft1d_c2c(y,x,-1)
   print('reconstruction:') print(x)
end

--test1d()

function test2d()
   nInputPlanes = 1 --128*96
   N = 8
   M = 8
   -- real to complex
   x = torch.randn(nInputPlanes,N,M):cuda()
   f = torch.CudaTensor(nInputPlanes,N,M/2+1,2)
   r = torch.CudaTensor(nInputPlanes,N,M)
   t = torch.Timer()
   t:reset()
   print('cats')
   --print(x)
   --print(f)
   libcufft.fft2d_r2c(x,f)
   print('hex')
   libcufft.fft2d_c2r(f,r)
   r:div(N*M)
   print('time elapse: ' .. t:time().real)
   err = torch.max(torch.abs(x:float()-r:float()))
   print('error=' .. err)
end

--test2d()

function testHermitian()
   local precision = 1e-5
   nSamples = 1
   nInputPlanes = 1
   N = 16
   M = 16
   -- real to complex
   x1 = torch.randn(nSamples,nInputPlanes,N,M):cuda()
   x2 = torch.zeros(nSamples,nInputPlanes,N,M,2):cuda()
   x2:select(5,1):copy(x1)
   f1 = torch.CudaTensor(nSamples,nInputPlanes,N,M/2+1,2)
   f2 = torch.CudaTensor(nSamples,nInputPlanes,N,M,2)
   cufft.fft2d_r2c(x1,f1)
   cufft.fft2d_c2c(x2,f2,1)
   err1 = torch.max(torch.abs(f2[{{},{},{},{1,M/2+1}}]:float()-f1:float()))
   assert(err1 < precision)
   r1 = x1:clone():zero()
   r2 = x2:clone():zero()
   cufft.fft2d_c2r(f1,r1)
   cufft.fft2d_c2c(f2,r2,-1)
   err2 = torch.max(torch.abs(r1:float()-r2:select(5,1):float()))
   assert(err2 < precision)
   f1 = f1:double():squeeze()
   f2 = f2:double():squeeze()
   print('error1=' .. err1)
   print('error2=' .. err2)
end
--testHermitian()


function testfft()
   N = 8
   M = 4
   input = torch.randn(N,M)
   input2 = torch.zeros(N,M,2)
   input2:select(3,1):copy(input)
   out1 =cufft.fft2dsingle(input2)
   out2 = cufft.fft2d(input,out2)
end
--testfft()

function test2dc2c()
   nSamples = 1
   nInputPlanes = 1
   N = 8
   M = 8
   x = torch.randn(nSamples,nInputPlanes,N,N):cuda()
   f1 = torch.Tensor(nSamples,nInputPlanes,N,N):cuda()
   f2 = torch.Tensor(nSamples,nInputPlanes,N,N):cuda()
   cufft.fft2d_c2c(x,f1,1,false)
   --cufft.fft2d_c2c(x,f2,1,true)
   err = torch.max(torch.abs(f1:float()-f2:float()))
   print('error=' .. err)
end


test2dc2c()

   
--test2d2()








function test_prod_real()
   local nMinibatch = 128 --math.random(1,10)
   local nInputPlanes = 96
   local nOutputPlanes = 256 --math.random(1,16)
   local dim = 2000

   local input = torch.CudaTensor(nMinibatch,nInputPlanes,dim):normal()
   local weight = torch.CudaTensor(nOutputPlanes, nInputPlanes, dim):normal()
   local output = torch.CudaTensor(nMinibatch, nOutputPlanes, dim):zero()
   print('testing real prod')
   print('\nTESTING FPROP')
   local timer = torch.Timer()
   timer:reset()
   spectralcuda.prod_fprop_real(input,weight,output)
   cutorch.synchronize()
   print('Fprop CUDA version took ' .. timer:time().real .. ' sec')
   local output2 = torch.CudaTensor(nMinibatch, nOutputPlanes, dim):zero()
   timer:reset()
   for s=1,nMinibatch do
      for i = 1,nOutputPlanes do
         for j = 1,nInputPlanes do 
			output2[s][i]:addcmul(input[s][j],weight[i][j])
         end
      end
   end
   cutorch.synchronize()
   print('Fprop Torch version took ' .. timer:time().real .. ' sec')
   output:add(-1,output2)
   print('Norm of difference = ' .. output:norm())
   
   print('\nTESTING BPROP')
   local gradInput = input:zero()
   local gradInput2 = gradInput:clone()
   local gradOutput = output:normal()
   weight:normal()
   timer:reset()
   spectralcuda.prod_bprop_real(gradOutput,weight,gradInput)
   cutorch.synchronize()
   print('Bprop CUDA version took ' .. timer:time().real .. ' sec')
   
   for s = 1,nMinibatch do
      for i = 1,nInputPlanes do
         for j = 1,nOutputPlanes do 
            gradInput2[s][i]:addcmul(gradOutput[s][j],weight[j][i])
         end
      end
   end
   gradInput:add(-1,gradInput2)
   print('Norm of difference = ' .. gradInput:norm())
      
   print('\nTESTING ACCGRAD')
   local gradWeight = weight:zero()
   local gradWeight2 = gradWeight:clone()
   gradOutput:normal()
   input:normal()
   timer:reset()
   spectralcuda.prod_accgrad_real(input, gradOutput, gradWeight)
   cutorch.synchronize()
   print('Accgrad CUDA version took ' .. timer:time().real .. ' sec')
   for j = 1,nOutputPlanes do 
      for i = 1,nInputPlanes do
         for s = 1,nMinibatch do
            gradWeight2[j][i]:addcmul(gradOutput[s][j],input[s][i])
         end
      end
   end
   gradWeight:add(-1,gradWeight2)
   print('Norm of difference = ' .. gradWeight:norm())
end


--test_prod_real()



-- test the complex product/accumulation used in fprop, bprop and accGrad
-- WARNING/TODO: this seems to work for powers of 2, but not for certain column 
-- numbers such as 17. Make sure the row/col sizes give the correct answer 
-- before running experiments.   
function test_prod_complex()
   local nMinibatch = 16 --math.random(1,10)
   local nInputPlanes = 4
   local nOutputPlanes = 16 --math.random(1,16)
   local nRows = 32
   local nCols = 32

   local input = torch.CudaTensor(nMinibatch,nInputPlanes,nRows,nCols,2):normal()
   local weight = torch.CudaTensor(nOutputPlanes, nInputPlanes, nRows, nCols, 2):normal()
   local output = torch.CudaTensor(nMinibatch, nOutputPlanes, nRows,nCols,2):zero()

   print('\nTESTING FPROP')
   local timer = torch.Timer()
   timer:reset()
   libFFTconv.prod_fprop(input,weight,output,true)
   cutorch.synchronize()
   print('Fprop CUDA version took ' .. timer:time().real .. ' sec')
   local output2 = torch.CudaTensor(nMinibatch, nOutputPlanes, nRows,nCols,2):zero()
   timer:reset()
   for s=1,nMinibatch do
      for i = 1,nOutputPlanes do
         for j = 1,nInputPlanes do 
			--complex.addcmul(input[s][j],weight[i][j],output2[s][i])
         end
      end
   end
   cutorch.synchronize()
   print('Fprop Torch version took ' .. timer:time().real .. ' sec')
   output:add(-1,output2)
   print('Norm of difference = ' .. output:norm())
   
   print('\nTESTING BPROP')
   local gradInput = input:zero()
   local gradInput2 = gradInput:clone()
   local gradOutput = output:normal()
   weight:normal()
   timer:reset()
   libFFTconv.prod_bprop(gradOutput,weight,gradInput,false)
   cutorch.synchronize()
   print('Bprop CUDA version took ' .. timer:time().real .. ' sec')
   
   for s = 1,nMinibatch do
      for i = 1,nInputPlanes do
         for j = 1,nOutputPlanes do 
            complex.addcmul(gradOutput[s][j],weight[j][i],gradInput2[s][i])
         end
      end
   end
   gradInput:add(-1,gradInput2)
   print('Norm of difference = ' .. gradInput:norm())
      
   print('\nTESTING ACCGRAD')
   local gradWeight = weight:zero()
   local gradWeight2 = gradWeight:clone()
   gradOutput:normal()
   input:normal()
   timer:reset()
   libFFTconv.prod_accgrad(input, gradOutput, gradWeight,false)
   cutorch.synchronize()
   print('Accgrad CUDA version took ' .. timer:time().real .. ' sec')
   for j = 1,nOutputPlanes do 
      for i = 1,nInputPlanes do
         for s = 1,nMinibatch do
            complex.addcmul(gradOutput[s][j],input[s][i],gradWeight2[j][i])
         end
      end
   end
   gradWeight:add(-1,gradWeight2)
   print('Norm of difference = ' .. gradWeight:norm())
end

--test_prod()

function test_fill_hermitian()
   nRows = 64
   nCols = 64
   nInputPlanes = 256
   nMinibatch = 128
   input = torch.randn(nMinibatch,nInputPlanes,nRows,nCols):cuda()
   inputP = torch.zeros(nMinibatch,nInputPlanes,nRows,nCols,2):cuda()
   F1 = torch.CudaTensor(nMinibatch, nInputPlanes, nRows, nCols/2+1,2)
   F2 = torch.CudaTensor(nMinibatch, nInputPlanes, nRows, nCols,2)
   F3 = torch.CudaTensor(nMinibatch, nInputPlanes, nRows, nCols,2)
   local timer = torch.Timer()
   timer:reset()
   inputP:select(5,1):copy(input)
   cufft.fft2d_c2c(inputP,F2,1)
   print('Pad+C2C: ' .. timer:time().real)
   timer:reset()
   cufft.fft2d_r2c(input,F1)
   spectralcuda.fill_hermitian(F1,F3)
   print('R2C+Fill : ' .. timer:time().real)
   print(torch.max(torch.abs(F2:float()-F3:float())))
end

--test_fill_hermitian()



function test_graph_pool()
   dim = 2000
   nMaps = 200
   poolsize = 4
   clusters = torch.randperm(dim)
   clusters:resize(dim/poolsize,poolsize)
   clusters = clusters:cuda()
   input = torch.randn(nMaps,dim):cuda()
   output = torch.CudaTensor(nMaps,dim/poolsize):fill(-99)
   indices = torch.CudaTensor(nMaps,dim/poolsize):fill(-99)
   timer = torch.Timer():reset()
   spectralcuda.graph_pool_fprop(input, output, clusters, indices, nMaps, dim, poolsize)
   cutorch.synchronize()
   print('fprop took ' .. timer:time().real)
   output2=torch.FloatTensor(output:size())
   indices2 = torch.FloatTensor(indices:size())
   for i = 1,nMaps do
      local out, ind = graph_pool_cpu(input[i], clusters)
      output2[i]:copy(out)
      indices2[i]:copy(ind)
   end
   output=output:float()
   indices = indices:float()
   cutorch.synchronize()
   print(torch.norm(output-output2))
   print(torch.norm(indices-indices2))


   -- bprop
   gradOutput = output:clone():normal():cuda()
   gradInput = input:clone():zero():cuda()
   indices = indices:cuda()
   timer:reset()
   spectralcuda.graph_pool_bprop(gradInput, gradOutput, indices)
   cutorch.synchronize()
   print('bprop took ' .. timer:time().real)
   gradInput2 = gradInput:clone():fill(1):float()
   for i = 1,nMaps do 
      graph_pool_bprop_cpu(gradInput2[i], gradOutput[i], indices[i])
   end
   print(torch.norm(gradInput:float()-gradInput2))


end

function graph_pool_cpu(input, clusters)
   local input = input:float()
   local clusters = clusters:float()
   local nClusters = clusters:size(1)
   local poolsize = clusters:size(2)
   local output = torch.FloatTensor(nClusters)
   local indices = torch.FloatTensor(nClusters)
   for i = 1,nClusters do 
      local pool=torch.Tensor(poolsize)
      for j = 1,poolsize do 
         pool[j] = input[clusters[i][j]]
      end
      local s,indx=torch.sort(pool,true)
      indx = indx[1]
      output[i] = pool[indx]
      indices[i] = clusters[i][indx]
   end
   return output, indices
end



function graph_pool_bprop_cpu(gradInput, gradOutput, indices)
   gradInput:zero()
   for i=1,indices:size(1) do 
      gradInput[indices[i]] = gradOutput[i]
   end
   return gradInput
end

test_graph_pool()

      







