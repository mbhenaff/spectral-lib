require 'cutorch'
require 'nn'


function nn.Linear:initReg()
   self.gradReg = self.gradReg or torch.Tensor()
   self.pts = self.pts or torch.Tensor()
   self.Y = self.Y or torch.Tensor()
   self.Z = self.Z or torch.Tensor()
   self.V = self.V or torch.Tensor()
   self.U = self.U or torch.Tensor()
end

function nn.Linear:regularize(L,eta,nSamples)
   local nSamples = nSamples or 100
   local N = L:size(1)
   self.gradReg:resize(N,N)
   self.Y:resize(N, nSamples)
   self.Z:resize(N, nSamples)
   self.V:resize(N, nSamples)
   self.U:resize(N, nSamples)

   local nInputs = self.weight:size(1)
   local nOutputs = self.weight:size(2)
   assert(nInputs % N == 0 and nOutputs % N == 0)
   local nBlocksIn = nInputs / N
   local nBlocksOut = nOutputs / N

   for i = 1,nBlocksOut do 
      for j = 1,nBlocksIn do 
         local W = self.weight:narrow(1,1+i*N,N):narrow(2,1+j*N,N)
         -- sample points on the sphere
         self.pts:resize(N, nSamples)
         self.pts:normal()
         self.pts:cdiv(self.pts:norm(2,1):expandAs(self.pts))

         self.Y:addmm(0,1,L,self.pts)
         self.Z:addmm(0,1,W,self.pts)
         self.V:addmm(0,1,W,self.Y)
         self.U:addmm(0,1,L,self.Z)
         self.gradReg:addmm(0,1/nSamples,(self.V+self.U)*2,self.Y:t())
         self.gradReg:addmm(0,1/nSamples,(L:t()*self.V+L*(L:t()*self.Z*2)+L:t()*self.U),self.pts:t())
         W:add(-eta,self.gradReg)
      end
   end
end



--[[
torch.manualSeed(123)
N = 1000
alpha = 0.5
A = torch.randn(N,N)
B = torch.randn(N,N)
L = (A + A:t())/2
U,S,V=torch.svd(L)
L = U*torch.diag(torch.pow(torch.range(1,N),-alpha))*U:t()
W = (B + B:t())/2
L:div(L:norm())
W:div(W:norm())

L = L:cuda()
W = W:cuda()

lr = 0.01
niter = 10000

cntr = 1
k = 100
loss = torch.Tensor(niter/k):zero()

model = nn.Linear(N,N)
model:initReg()
model = model:cuda()
timer = torch.Timer()
for i = 1,niter do 
   timer:reset()
   model:regularize(L,lr)
   cutorch.synchronize()
   print('time=' .. timer:time().real)
   model.weight:div(model.weight:norm())
   if i % k == 0 then
      loss[cntr] = torch.norm(L*model.weight - model.weight*L)
      print(loss[cntr])
      cntr = cntr + 1
   end
end   
--]]