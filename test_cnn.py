import torch 

from cnn import CNN 

batch_size = 13 
e_char = 50
e_word = 290
m_word = 35
kernel_size  = 5

x_reshaped = torch.ones((batch_size, e_char, m_word))

c = CNN(e_char, e_word, kernel_size)
p = c(x_reshaped) 

print(p)
