import torch.nn as nn
import torch.nn.functional as F
import torch

class Final_Model(nn.Module):
    def __init__(self, input_dim, layers, hidden_dim, num_head, ff_weight, drop_out, num_classes):
        super(Final_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1_1 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv1_3 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)

        self.transformer_encoder_1 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dropout=drop_out, nhead=num_head, dim_feedforward= ff_weight * hidden_dim)
        self.transformer_encoder_2 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dropout=drop_out, nhead=num_head, dim_feedforward= ff_weight * hidden_dim)
        self.transformer_encoder_3 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dropout=drop_out, nhead=num_head, dim_feedforward= ff_weight * hidden_dim)

        self.encoder_1 = nn.TransformerEncoder(self.transformer_encoder_1, num_layers=layers)
        self.encoder_2 = nn.TransformerEncoder(self.transformer_encoder_2, num_layers=layers)
        self.encoder_3 = nn.TransformerEncoder(self.transformer_encoder_3, num_layers=layers)

        self.fc = nn.Linear(hidden_dim * 3, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        # First
        x1 = self.conv1_1(x1)       # batch, hidden_dim, feature
        x1 = x1.permute(2, 0, 1)    # feature_dim, batch_size, hidden_dim
        x1 = self.encoder_1(x1)
        x1 = x1.mean(dim=0)
        # Second  
        x2 = self.conv1_2(x2)       # batch, hidden_dim, feature
        x2 = x2.permute(2, 0, 1)    # feature_dim, batch_size, hidden_dim
        x2 = self.encoder_2(x2)
        x2 = x2.mean(dim=0)
        # Third
        x3 = self.conv1_3(x3)       # batch, hidden_dim, feature
        x3 = x3.permute(2, 0, 1)    # feature_dim, batch_size, hidden_dim
        x3 = self.encoder_3(x3)
        x3 = x3.mean(dim=0)

        # Concatenate features
        x = torch.cat((x1, x2, x3), dim=1) # batch_size, hidden_dim * 3
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class Multi_Model(nn.Module):
    def __init__(self, input_dim, layers, hidden_dim, num_head, ff_weight, drop_out, num_classes):
        super(Multi_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1_1 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
        
        self.transformer_encoder_1 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dropout=drop_out, nhead=num_head, dim_feedforward= ff_weight * hidden_dim)
        self.transformer_encoder_2 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dropout=drop_out, nhead=num_head, dim_feedforward= ff_weight * hidden_dim)
        
        self.encoder_1 = nn.TransformerEncoder(self.transformer_encoder_1, num_layers=layers)
        self.encoder_2 = nn.TransformerEncoder(self.transformer_encoder_2, num_layers=layers)
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # First
        x1 = self.conv1_1(x1)       # batch, hidden_dim, feature
        x1 = x1.permute(2, 0, 1)    # feature_dim, batch_size, hidden_dim
        x1 = self.encoder_1(x1)
        x1 = x1.mean(dim=0)
        # Second  
        x2 = self.conv1_2(x2)       # batch, hidden_dim, feature
        x2 = x2.permute(2, 0, 1)    # feature_dim, batch_size, hidden_dim
        x2 = self.encoder_2(x2)
        x2 = x2.mean(dim=0)
        
        # Concatenate features
        x = torch.cat((x1, x2), dim=1) # batch_size, hidden_dim * 3
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, layers, hidden_dim, num_head, ff_weight, drop_out, num_classes):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dropout=drop_out, nhead=num_head, dim_feedforward= ff_weight * hidden_dim)
        self.encoder = nn.TransformerEncoder(self.transformer_encoder, num_layers = layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)       # batch, hidden_dim, feature
        x = x.permute(2, 0, 1)  # Feature_dim, batch_size, hidden_dim
        x = self.encoder(x)
        x = x.mean(dim = 0)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x