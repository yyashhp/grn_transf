import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class GRNDataset(Dataset):

    def __init__(self, ds, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        # self.tokenizer_src = tokenizer_src
        # self.tokenizer_tgt = tokenizer_tgt
        # self.src_lang = src_lang
        # self.tgt_lang = tgt_lang
       # self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype = torch.int64)
       # self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor(-1e9, dtype = torch.int64)


    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair.wendy_estimate
        tgt_text = src_target_pair.true_grn

      #  enc_input_tokens = self.tokenizer_src.encode(src_text).ids
       # dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        if len(tgt_text) == 10:
            reshaped_src_text, reshaped_tgt_text = reshape_and_pad(src_text, tgt_text)
        elif len(tgt_text) == 20:
            reshaped_src_text = src_text.flatten()
            reshaped_tgt_text = tgt_text.flatten()
        enc_num_padding_tokens = self.seq_len - len(reshaped_src_text) #we consider eos and sos
        dec_num_padding_tokens = self.seq_len - len(reshaped_tgt_text) #we add sos only to decoder side

        if dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        

    
        #Add sos and eos to source text, and padding
        encoder_input = torch.tensor(reshaped_src_text * 1000, dtype = torch.int64)
        encoder_input = encoder_input + torch.min(encoder_input)
        encoder_input = torch.cat(
            [
                encoder_input,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )
                
        decoder_input = torch.tensor(reshaped_tgt_text * 1000, dtype = torch.int64)
        decoder_input = decoder_input + torch.min(decoder_input)
        #print(decoder_input.shape)
        # Add sos to the decoder input, and padding
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)

            ]
        )

        # Add eos to the label, which is what we expect as output
        label = torch.cat(
           [
            torch.tensor(reshaped_tgt_text, dtype = torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
           ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        #print(encoder_input.shape)
       # print(decoder_input.shape)
        return {
            "encoder_input": encoder_input, #(Seq_len)
            "decoder_input": decoder_input, #(Seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(len(tgt_text)),
            "label": label, #(seq_len)
        }
def causal_mask(size):
    N = size

# Create the mask
    mask = torch.zeros((1, 400, 400), dtype=int)

    # Iterate through each element in the flattened vector
    for r in range(N):
        for c in range(N):
            idx = r * N + c  # Index in the flattened vector
            # Iterate through the flattened vector to set mask values
            for k in range(N*N):
                if k == idx:
                    continue  # Skip the element itself
                original_row = k // N
                original_col = k % N
                if original_row == r or original_col == c:
                    mask[0, idx, k] = 1
    return mask


def reshape_and_pad(x, y):
    original_wendy = x[:10, :10]
    original_wendy = original_wendy.flatten()
    true_grn_flattened = y.flatten()
    return original_wendy, true_grn_flattened
