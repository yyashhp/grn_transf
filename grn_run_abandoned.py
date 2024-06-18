import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class WENDY_Transformer(nn.Module):
    def __init__(self, max_matrix_size, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=max_matrix_size)
        self.input_projection = nn.Linear(max_matrix_size * max_matrix_size, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, max_matrix_size * max_matrix_size)

    def forward(self, input_matrix, tgt_matrix):
        batch_size, matrix_size, _ = input_matrix.size()

        # Flatten the input matrix
        input_matrix = input_matrix.view(batch_size, -1)
        print(input_matrix.shape)
        # Project the input matrix to the model dimension
        input_matrix = self.input_projection(input_matrix)

        # Add positional encoding and reshape for Transformer input
        input_matrix = self.positional_encoder(input_matrix)
        input_matrix = input_matrix.view(matrix_size, batch_size, self.dim_model).permute(1, 0, 2)

        # Pass through the Transformer module
        output = self.transformer(input_matrix)

        # Reshape the output and pass through the output linear layer
        output = output.permute(1, 0, 2).contiguous().view(batch_size, -1)
        output = self.out(output)

        # Reshape the output to match the original matrix size
        output = output.view(batch_size, matrix_size, matrix_size)

        return output

def pad_matrix(matrix, max_size, pad_value=0):
    pad_size = max_size - len(matrix)
    if pad_size > 0:
        padding = (0, pad_size, 0, pad_size)
        matrix = F.pad(matrix, padding, mode='constant', value=pad_value)
    return matrix

# def batchify_data(input_matrices, true_matrices, batch_size, max_matrix_size):
#     batches = []
#     for idx in range(0, len(input_matrices), batch_size):
#         batch_input = []
#         batch_true = []
#         for i in range(idx, min(idx + batch_size, len(input_matrices))):
#         #    input_matrix = pad_matrix(input_matrices[i], max_matrix_size)
#         #    true_matrix = pad_matrix(true_matrices[i], max_matrix_size)
#             batch_input.append(input_matrices[i])
#             batch_true.append(true_matrices[i])
#         batch_input = torch.stack(batch_input, dim=0)
#         batch_true = torch.stack(batch_true, dim=0)
#         batches.append((batch_input, batch_true))
#     print(type(batches[0][0]))
#     return batches

def batchify_data(input_matrices, true_matrices, batch_size, max_matrix_size):
    batches = []
    for idx in range(0, len(input_matrices), batch_size):
        batch_input = []
        batch_true = []
        for i in range(idx, min(idx + batch_size, len(input_matrices))):
          #  input_matrix = pad_matrix(input_matrices[i], max_matrix_size)
          #  true_matrix = pad_matrix(true_matrices[i], max_matrix_size)
            batch_input.append(input_matrices[i])
            batch_true.append(true_matrices[i])
        batch_input = torch.tensor(np.stack(batch_input), dtype=torch.float32)
        batch_true = torch.tensor(np.stack(batch_true), dtype=torch.float32)
        batches.append((batch_input, batch_true))
    return batches

def train_loop(model, opt, loss_fn, dataloader, device):
    model.train()
    total_loss = 0
    accuracy = 0

    for batch_input, batch_true in dataloader:
        batch_input = batch_input.to(device)
        batch_true = batch_true.to(device)
        print(type(batch_input))
        print("---------------")
        pred = model(batch_input)
        loss = loss_fn(pred, batch_true)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
        accuracy += compute_accuracy(pred, batch_true)

    total_loss /= len(dataloader)
    accuracy /= len(dataloader)

    return total_loss, accuracy

def validation_loop(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    accuracy = 0

    with torch.no_grad():
        for batch_input, batch_true in dataloader:
            batch_input = batch_input.to(device)
            batch_true = batch_true.to(device)

            pred = model(batch_input)
            loss = loss_fn(pred, batch_true)
            total_loss += loss.detach().item()
            accuracy += compute_accuracy(pred, batch_true)

    total_loss /= len(dataloader)
    accuracy /= len(dataloader)

    return total_loss, accuracy

def test_loop(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    accuracy = 0

    with torch.no_grad():
        for batch_input, batch_true in dataloader:
            batch_input = batch_input.to(device)
            batch_true = batch_true.to(device)

            pred = model(batch_input)
            loss = loss_fn(pred, batch_true)
            total_loss += loss.detach().item()
            accuracy += compute_accuracy(pred, batch_true)

    total_loss /= len(dataloader)
    accuracy /= len(dataloader)

    return total_loss, accuracy

def compute_accuracy(pred, true):
    pred = pred.view(-1)
    true = true.view(-1)
    correct = torch.sum(torch.isclose(pred, true, rtol=1e-3))
    total = true.numel()
    return correct.item() / total

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device):
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss, train_acc = train_loop(model, opt, loss_fn, train_dataloader, device)
        val_loss, val_acc = validation_loop(model, loss_fn, val_dataloader, device)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
        #print()

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list

# Example usage

