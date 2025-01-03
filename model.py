import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1) # flatten the features for the FC layer
        features = self.embed(features)
        return features # shape: (batch_size, embed_size)
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1): # input_size = embed_size
        super(DecoderRNN, self).__init__()
        
        # Set the hidden size for init_hidden
        self.hidden_size = hidden_size
        
        # Set the device
        self.device = device
        
        # Embedded layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first= True,
                            dropout = 0)
        
        # Fully Connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden(self, batch_size): # Initialize the hidden state of the LSTM with zeros
        return (torch.zeros(1, batch_size, self.hidden_size, device = device),
                torch.zeros(1, batch_size, self.hidden_size, device = device))
    
    def forward(self, features, captions):
        
        # Initialize the hidden state
        self.hidden = self.init_hidden(features.shape[0])# features is of shape (batch_size, embed_size) so features.shape[0] is the batch_size
        
        
        # Embedding the captions
        embedded = self.embed(captions[:,:-1])
        # print(embedded.shape) # (batch_size, caption_length, embed_size)
        # print(features.unsqueeze(1).shape) # (batch_size, 1, embed_size)
        
        embedded = torch.cat((features.unsqueeze(1), embedded), dim=1) # Concatenate the features and captions to feed into the LSTM
        # print(embedded.shape) 
        
        # LSTM
        lstm_out, self.hidden = self.lstm(embedded, self.hidden)
        
        # Functional component
        out = self.fc(lstm_out)
        return out

    def sample(self, inputs, states=None, max_len=20): 
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Initialize the hidden state
        hidden = self.init_hidden(inputs.shape[0])# features is of shape (batch_size, embed_size) so features.shape[0] is the batch_size
        
        out_list = list()
        word_len = 0
        
        with torch.no_grad():
            while word_len < max_len:
                lstm_out, hidden = self.lstm(inputs, hidden)
                out = self.fc(lstm_out)
                out = out.squeeze(1) # (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
                out = out.argmax(dim=1) # (batch_size, vocab_size) -> (batch_size)
                # print(out)
                out_list.append(out.item()) # Append the word to the list
                
                inputs = self.embed(out.unsqueeze(0)) # (1) -> (1, embed_size)
                
                word_len += 1
                if out == 1:
                    break
        
        return out_list # list of image ids of length max_len (max_len is the maximum length of the sentence)