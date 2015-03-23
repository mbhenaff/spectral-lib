require 'InterpImage'

k = 5
n = 14

m1 = nn.InterpImage(k,k,n,n,'spatial'):float()
m2 = nn.InterpImage(k,k,n,n,'bilinear'):float()
--m3 = nn.InterpImage(k,k,n,n,'spline'):float()
--m4 = nn.InterpImage(k,k,n,n,'dyadic_spline'):float()

s = 10000
input = torch.rand(s,2*k*k):float()
for i = 1,s do 
   input[i]:mul(1/input[i]:norm())
end
input = input:resize(s,1,k,k,2)

out1 = m1:forward(input)
out2 = m2:forward(input)
--out3 = m3:forward(input)
--out4 = m4:forward(input)

out1:resize(s,2*n*n)
out2:resize(s,2*n*n)
--out3:resize(s,2*n*n)
--out4:resize(s,2*n*n)

d1 = out1:norm(2,2)
d2 = out2:norm(2,2)
print(torch.max(d1))
print(torch.max(d2))
print(torch.max(d1)/torch.max(d2))
--d3 = out3:norm(2,2)
--d4 = out4:norm(2,2)






      
      


