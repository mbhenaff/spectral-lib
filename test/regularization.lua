require 'cutorch'
require 'nn'
require 'gnuplot'

function nn.Linear:initReg()
   self.gradReg = self.gradReg or torch.Tensor()
   self.pts = self.pts or torch.Tensor()
   self.Y = self.Y or torch.Tensor()
   self.Z = self.Z or torch.Tensor()
   self.V = self.V or torch.Tensor()
   self.U = self.U or torch.Tensor()
   self.LU = self.LU or torch.Tensor()
   self.LV = self.LV or torch.Tensor()
end

function nn.Linear:regularize(L,eta,nSamples)
   local nSamples = nSamples or 10
   local N = L:size(1)
   self.gradReg:resize(N,N)
   self.Y:resize(N, nSamples)
   self.Z:resize(N, nSamples)
   self.V:resize(N, nSamples)
   self.U:resize(N, nSamples)
   self.LU:resize(N, nSamples)
   self.LV:resize(N, nSamples)

   local nInputs = self.weight:size(2)
   local nOutputs = self.weight:size(1)
   assert(nInputs % N == 0 and nOutputs % N == 0)
   local nBlocksIn = nInputs / N
   local nBlocksOut = nOutputs / N

   -- sample points on the sphere
   self.pts:resize(N, nSamples)
   self.pts:normal()
   self.pts:cdiv(self.pts:norm(2,1):expandAs(self.pts))

   for i = 1,nBlocksOut do 
      for j = 1,nBlocksIn do 
         local W = self.weight:narrow(1,1+(i-1)*N,N):narrow(2,1+(j-1)*N,N)
         self.Y:addmm(0,1,L,self.pts)
         self.Z:addmm(0,1,W,self.pts)
         self.V:addmm(0,1,W,self.Y)
         self.U:addmm(0,1,L,self.Z)
         self.LU:addmm(0,1,L,self.U)
         self.LV:addmm(0,1,L:t(),self.V)
         self.gradReg:addmm(0,1/nSamples,self.LU*2,self.pts:t())
         self.gradReg:addmm(0,1/nSamples,-self.LV*2,self.pts:t())
         self.gradReg:addmm(0,1/nSamples,self.V*2,self.Y:t())
         self.gradReg:addmm(0,1/nSamples,-self.U*2,self.Y:t())




--         self.gradReg:addmm(0,1/nSamples,L,(self.U-self.V)*2)
  --       self.gradReg:mm(self.gradReg,self.pts)
    --     self.gradReg:addmm(1/nSamples,(self.V-self.U)*2,self.Y:t())
         --self.gradReg:addmm(0,1/nSamples,(L:t()*self.V*2+L*(L:t()*self.Z)+L:t()*self.U),self.pts:t())
         local norm = W:norm()
         --local p = self.gradReg:dot(W)/W:norm()
--         print('p=' .. p)
--         self.gradReg:add(-p,W)
         W:add(-eta,self.gradReg)
         W:mul(norm/W:norm())
      end
   end
end


if false then
torch.manualSeed(123)
N = 10
alpha = 1
A = torch.randn(N,N)
B = torch.randn(N,N)
L = (A + A:t())/2
_,U=torch.eig(L,'V')
D = torch.diag(torch.pow(torch.range(1,N),-alpha))
--D[{{8,10}}]:zero()
L = U*D*U:t()
W = (B + B:t())/2
L:div(L:norm())
W:div(W:norm())

L = L:cuda()
W = W:cuda()

lr = 0.01
decay=0.99
niter = 10000

cntr = 1
k = 10
loss = torch.Tensor(niter/k):zero()

model = nn.Linear(N,N)
model:initReg()
model = model:cuda()
timer = torch.Timer()
for i = 1,niter do 
   timer:reset()
   model:regularize(L,lr)
   cutorch.synchronize()
   --print('time=' .. timer:time().real)
  -- model.weight:div(model.weight:norm())
   if i % k == 0 then
      loss[cntr] = torch.norm(L*model.weight - model.weight*L)
      --print('norm=' .. model.weight:norm())
      print(loss[cntr])
      --lr = lr*decay
      cntr = cntr + 1
   end
end   
gnuplot.plot(loss)
end

W = model.weight

