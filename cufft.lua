-- Lua wrapper for cufft functions
require 'libspectralnet'

cufft = cufft or {}

function cufft.fft1d(input, output)
   local nSamples = input:size(1)
   local nPlanes = input:size(2)
   local N = input:size(3)
   local M = input:size(4)
   input:resize(nSamples*nPlanes*N, M)
   output:resize(nSamples*nPlanes*N, M/2+1, 2)
   libspectralnet.fft1d_r2c(input, output)
   input:resize(nSamples, nPlanes, N, M)
   output:resize(nSamples, nPlanes, N, M/2+1, 2)
end

function cufft.ifft1d(input, output)
   local nSamples = output:size(1)
   local nPlanes = output:size(2)
   local N = output:size(3)
   local M = output:size(4)
   input:resize(nSamples*nPlanes*N, M/2+1, 2)
   output:resize(nSamples*nPlanes*N, M)
   libspectralnet.fft1d_c2r(input,output)
   output:div(M)
   input:resize(nSamples, nPlanes, N, M/2+1, 2)
   output:resize(nSamples, nPlanes, N, M)
end

function cufft.fft2d_r2c(input,output,debug)
   local debug = debug or false
   local nSamples = input:size(1)
   local nPlanes = input:size(2)
   local N = input:size(3)
   local M = input:size(4)
   input:resize(nSamples*nPlanes, N, M)
   if debug then
      output:resize(nSamples*nPlanes, N, M, 2)
      local dft1 = cufft.dftmatrix(N,1)
      local dft2 = cufft.dftmatrix(M,1)
      for i = 1,output:size(1) do
         local input2 = torch.zeros(N,M,2)
         input2:select(3,1):copy(input[i])
         input2 = complex.mm(input2,dft2)
         input2 = complex.mm(dft1,input2)
         output[i]:copy(input2)
      end
      output:resize(nSamples, nPlanes, N, M, 2)
   else
      output:resize(nSamples*nPlanes, N, M/2+1, 2)
      libspectralnet.fft2d_r2c(input, output)
      output:resize(nSamples, nPlanes, N, M/2+1, 2)
   end
   input:resize(nSamples, nPlanes, N, M)
   --print(input:isContiguous())
end

function cufft.fft2d_c2r(input, output, debug)
   local debug = debug or false
   local nSamples = output:size(1)
   local nPlanes = output:size(2)
   local N = output:size(3)
   local M = output:size(4)
   output:resize(nSamples*nPlanes, N, M)
   --print(input:isContiguous())
   if debug then
      input:resize(nSamples*nPlanes, N, M, 2)
      local dft1 = cufft.dftmatrix(N,-1)
      local dft2 = cufft.dftmatrix(M,-1)
      for i = 1,input:size(1) do
         local input2 = torch.zeros(N,M,2)
         input2:select(3,1):copy(input[i])
         input2 = complex.mm(input2,dft2)
         input2 = complex.mm(dft1,input2)
         output[i]:copy(input2)
      end
      input:resize(nSamples,nPlanes, N, M, 2)
   else
      input:resize(nSamples*nPlanes, N, M/2+1, 2)
      libspectralnet.fft2d_c2r(input,output)
      input:resize(nSamples, nPlanes, N, M/2+1, 2)
   end
   output:div(M*N)
   output:resize(nSamples, nPlanes, N, M)
end

function cufft.fft2d_c2c(input,output,dir,debug)
   local dir = dir or 1
   local debug = debug or false
   local nSamples = output:size(1)
   local nPlanes = output:size(2)
   local N = output:size(3)
   local M = output:size(4)
   input:resize(nSamples*nPlanes, N, M, 2)
   output:resize(nSamples*nPlanes, N, M, 2)
   if debug then
      print('warning, slow')
      local dft1 = cufft.dftmatrix(N,dir)
      local dft2 = cufft.dftmatrix(M,dir)
      for i = 1,input:size(1) do
         local input2 = input[i]:clone():double()
         input2 = complex.mm(input2,dft2)
         input2 = complex.mm(dft1,input2)
         output[i]:copy(input2)
      end
   else
      libspectralnet.fft2d_c2c(input,output,dir)
   end
   input:resize(nSamples, nPlanes, N, M, 2)
   if dir == -1 then
      output:div(M*N)
   end
   output:resize(nSamples, nPlanes, N, M, 2)
end
  







function cufft.fft2dsingle(input,dir)
   local dir = dir or 1
   local dft1 = cufft.dftmatrix(input:size(1),dir)
   local dft2 = cufft.dftmatrix(input:size(2),dir)
   local output = complex.mm(input,dft2)
   output = complex.mm(dft1,output)
   return output
end


function cufft.dftmatrix(n,dir)
   local dir = dir or 1
   local dft = torch.Tensor(n,n,2)
   local real = dft:select(3,1)
   local imag = dft:select(3,2)
   for i = 1,n do 
      for j = 1,n do 
         local theta = 2*math.pi*(i-1)*(j-1)/n
         real[i][j] = math.cos(theta)
         imag[i][j] = -dir*math.sin(theta)
      end
   end
   return dft
end




