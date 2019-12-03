import torch
import torch.nn.functional as F
import torch.nn

#define cosineSimilarity
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

#examples
ex1 = torch.randn(31,768)
ex2 = torch.randn(31,768)
ex3 = torch.randn(31,768)
ex4 = torch.randn(31,768)

#define output
final = torch.zeros([3,1])

#compare ex1 and (ex2, ex3, ex4)
output1 = torch.tensordot(ex1,ex2)
output2 = torch.tensordot(ex1,ex3)
output3 = torch.tensordot(ex1,ex4)

#insert output vector
final[0,:] = output1
final[1,:] = output2
final[2,:] = output3

#softmax
sort_ = F.softmax(final, dim=0)
tt = torch.t(sort_)

#extract top2
top = torch.topk(tt,2)

#extract top2 name
top1 = top[1].tolist()



