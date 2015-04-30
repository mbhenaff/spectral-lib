-- useful functions

function repeatDiag(M,k)
   assert(M:size(1) == M:size(2))
   local n = M:size(1)
   local X = torch.zeros(n*k,n*k)
   for i = 1,k do
      X[{{(i-1)*n+1,i*n},{(i-1)*n+1,i*n}}]:copy(M)
   end
   return X
end
