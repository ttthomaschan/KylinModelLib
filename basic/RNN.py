import torch
import torch.nn as nn

x_input = torch.randn(2,3,10)

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,batch_first=False):
        super(RNN,self).__init__()

        self.rnn_cell = nn.RNNCell(input_size,hidden_size)

        self.hidden_size = hidden_size
        self.batch_first = batch_first

    def _initialize_hidden(self,batch_size):
        return torch.zeros(batch_size,self.hidden_size)

    def forward(self,inputs,initial_hidden=None):
        if self.batch_first:   # 检查是否以batch为第一维度
            batch_size, seq_size, feat_size = inputs.size()
            inputs = inputs.permute(1,0,2)  # 转换维度
        
        else:
            seq_size, batch_size, feat_size = inputs.size()
        
        hiddens = []

        if initial_hidden is None:
            initial_hidden = self._initialize_hidden(batch_size)
            initial_hidden = initial_hidden.to(inputs.device)

        hidden_t = initial_hidden

        for t in range(seq_size):
            hidden_t = self.rnn_cell(inputs[t], hidden_t)
            hiddens.append(hidden_t)

        hiddens = torch.stack(hiddens)

        if self.batch_first:
            hiddens = hiddens.permute(1,0,2)

        return hiddens


model = RNN(10,15,batch_first=True)
outputs= model(x_input)
print(outputs.shape)

        