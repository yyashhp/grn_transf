import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from data_pairs import Data_Wendy_True
import numpy as np

from dataset import BilingualDataset, causal_mask
#from model import build_transformer
from grn_model import build_transformer
from grn_config import get_weights_file_path, get_config
#from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from grn_dataset import GRNDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from pathlib import Path

def greedy_decode(model, source, source_mask, max_len, device):
   

    #Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    print(len(source[0]))
    decoder_input = torch.empty(1,1).fill_(source[0][0]).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
            
        #Build mask for the target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        #Get the next token , we only want projection of the last token, which is the next one in the sequence
        prob = model.project(out[:, -1])

        #Select the token with maximum probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        print(f"counter: {len(decoder_input)}")
        if len(decoder_input) == 400 :
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    # Size of the control window (use default value)

    with torch.no_grad(): #Disable gradient calculation for the following
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask,max_len, device)

            source_text = batch['encoder_input'][0]
            target_text = batch['decoder_input'][0]
            model_out_text = model_out.detach().cpu().numpy()

            # Print to the console
            print_msg('-'*80)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = np.load('./875_samples.npy', allow_pickle=True)

    #Build tokenziers
   # tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    #tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = GRNDataset(train_ds_raw, config['seq_len'])
    val_ds = GRNDataset(val_ds_raw, config['seq_len'])

    max_len_src = 400
    max_len_tgt = 400

   # for item in ds_raw:
        #src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
      #  tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
      #  max_len_src = max(max_len_src, len(src_ids))
      #  max_len_tgt = max(max_len_tgt, len(tgt_ids))
   # print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sequence: {max_len_tgt}")
#
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model 

def train_model(config):
#Define the device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_ds(config)
    model = get_model(config, 400, 400).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=int(-1e9), label_smoothing = 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device) #(batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, seq_len)

            # Run the tensors through model
            print(f"encoder_input; {encoder_input.shape} \n encoder_mask: {encoder_mask.shape}")
            encoder_output = model.encode(encoder_input, encoder_mask) #(Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(Batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (Batch, seq_len)

            #(batch, seq_len, tgt_vocab_size --> (Batch * seq_len, tgt_vocab_size))
            loss = loss_fn(proj_output.view(-1, 400), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            #Backpropogate the loss
            loss.backward()
            
            #Update weights
            optimizer.step()
            optimizer.zero_grad()
#MOVE VALIDATION TO AFTER EVERY EPOCH
#
            global_step += 1
            run_validation(model, val_dataloader,  config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            #Run validation after each epoch
        #Save the model at every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)