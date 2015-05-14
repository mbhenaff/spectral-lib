require 'gnuplot'

function gradComm(L, W, x)
   local y = L*x
   local z = W*x
   local v = W*y
   local w = L*z
   local LLT = L:t()*L
   g = torch.Tensor(W:size()):zero()
   g:addr((v+w)*2,y)
   g:addr((L:t()*v+L*(L:t()*z*2)+L:t()*w),x)
   return g
end

function gradSymm(W)
   g = W - W:t()
   return g
end

N = 10
A = torch.randn(N,N)
B = torch.randn(N,N)
L = (A + A:t())/2
W = (B + B:t())/2
L:div(L:norm())
W:div(W:norm())

lr = 0.001
niter = 10000

cntr = 1
k = 1000
loss = torch.Tensor(niter/k):zero()
for i = 1,niter do
   x = torch.randn(N)
   x:div(math.sqrt(x:norm()))
--   g = grad(L,W,x)
   g = gradSymm(W)
   W:add(-lr,g)
   W:div(W:norm())
   if i % k == 0 then
      loss[cntr] = torch.norm(L*W - W*L)
      print(loss[cntr])
      cntr = cntr + 1
   end
end


   
   
   