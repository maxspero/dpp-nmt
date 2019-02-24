import torch 

from highway import Highway

batch_size = 3
e_word = 5

x = torch.ones((batch_size, e_word))

h = Highway(e_word, e_word)
p = h(x) 
print(p)
