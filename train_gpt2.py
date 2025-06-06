import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    r"""
    Inside of this class we will carry out the self-attention computation between tokens of the same prompt.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query and value kind of concatenated in the below line of code
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1   # so we are adding a new attribute to the c_proj object
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # not really a 'bias' more of a mask, but following the OpenAI/ HF naming convention
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # The above "x" comes just after the layer norm operation
        B, T, C = x.size()    # (Batch_size, sequence_length, embedding_dimension)
        # calculate the query, key, values for all the heads before spilting it into multiple number of heads and move head forward to be the batch dim
        # nh is the number of heads, hs is the head_size and C is the number of channels (embedding dimensionality) -> (nh * hs)
        # e.g. in GPT2 (124M parameter model), n_head=12, hs=64 and thus C = nh * hs = 768
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)    # split along the last dimension each of the size n_embed
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)
        # Next comes up the attention computation between tokens and tokens and thus the size is (T, T)
        attn = (q @ k.tranpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))     #k.size(-1) is still referring to the head size
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v   # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, Hs)
        y = y.tranpose(1, 2).contiguous().view(B, T, C)    # reassemble all the heads/ concatenate all the heads side-by-side
        # output projection
        y = self.c_proj(y)
        # A simple explanation of the importance of the output projection:
        # During the concatenation of the different heads, there was no communication between the heads, each of them were computed individually and thus to mix-up their results, we project them through this output projection layer
        return y
    

class MLP(nn.Module):
    r"""
    This class contains the code for the feedforward layers which comes after the self-attention layers
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    r"""
    This class represents "1" transformer block, inside of which you have a "layer normalization" followed by "self-attention" and then doing residual connection with the input embeddings. Then, again followed by a "layer normalization" inside the feed forward network and then lastly passing through the "FFNs" before again doing the residual connection
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    block_size: int = 1024     # max sequence length in any of the samples
    vocab_size: int = 50257    # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <End_of_sentence> token
    n_layer: int = 12          # total number of transformer blocks, how many times are we repeating it
    n_head: int = 12           # number of heads
    n_embed: int = 768         # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)   # So, this layer normalization comes after the feed forward network or the MLP layers
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # So lm_head is the projection layer that takes  the embedding dimensionality for each token and then projects it back to the vocab_size, and we choose the vocab (or the next token from here!)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize the parameters of the sub-modules inside this module
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx in the above case is the token id from the vocabulary of size 50257
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and the position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # shape (T), positions for the tokens that we get from a sequence
        pos_emb = self.transformer.wpe(pos)     # position embeddings (T, n_embed)
        tok_emb = self.transformer.wte(idx)     # Token embeddings (B, T, n_embed)
        # pos_emb will be the same across all the sequences in the batch and thus we will just broadcast it, and thus the missing "batch_dimension" in the above case
        x = tok_emb + pos_emb
        # Next, forward to the block of the transformer
        for block in self.transformer.h:
            x = block(x)
        # Then, forward to the layernorm and the final classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)
        # return logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        r"""
        class method used to load pretrained GPT2 model weights from huggingface
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" %model_type)

        # n_layer, n_head and n_embed are all determined from the model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),    # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024), # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280), # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600)   # 1558M 
        }[model_type]


        # config_args = {"n_layer": 12, "n_head": 12, "n_embed": 768}

        config_args['vocab_size'] = 50257   # always 50257 for GPT checkpoints
        config_args['block_size'] = 1024    # always 1024 seq len for GPT checkpoints
        # Create from scratch an initialized GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]   # We have to discard this causal mask or bias buffer as we also defined before, since this is not a learnable parameter

        # init huggingface transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy the weights from the hugging face model to the model we created above, ensuring that all of the parameters are aligned and matches in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore this, since it's just a buffer and not a parameter
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # Again just a buffer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # Basically the openai checkpoints use a 'Conv1D' module, but we are making use of a vanilla Linear layer, this means that we have to transpose these weights to be copied to the model we implemented
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for Conv1D weights since we need to transpose them
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # just simply copy over all the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    

# Now we are going to define a simple DataLoader class, so that we don't keep on feeding the same class!
class DataLoaderLite:
    
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from the disk and store them in the memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch contains {len(self.tokens) // (B * T)} batches")
        # just for info: 1 epoch passes all the batches forward and backward

        # state, which position in the sequences are we starting from
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]  # +1 in this case is the GT for the very last token
        x = (buf[:-1].view(B, T))  # input to the model
        y = (buf[1:].view(B, T))   # targets for the model
        # Once we have filled in these tokens in the first batch move the current position so that we can retrieve the next fresh set of batches
        self.current_position += B * T
        # if loading the next batch would be out of bounds, just reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

import time
    
# Simple block of code to detect device autonomously
device = "cpu" # default
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # for Mac!
    device = "mps"
print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# num_return_sequences = 5    # the number of sequences in a batch to be processed
# max_length = 30             # This is the maximum length of each of those 5 sequences

train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')     #loading the model from pretrained weights
model = GPT(GPTConfig())    # randomly initialized model
print("Didn't crash while copying the weights from a hugging gpt2 model to our implemented model")

# model.eval()
# model.to('cuda')
model.to(device)

# logits, loss = model(x, y)      # we are passing the above mentioned "x" through an untrained so expect to get some giberish!

# Just a small sanity check, and thus we are overfitting the model to this small batch!
# defining the optimizer and optimizing the parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):  # 50 iterations
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # print(f"Step {i}, loss: {loss.item()}")  # loss.item -> shifting the loss tensor value from the gpu to the cpu and also converting to a float value
    torch.cuda.synchronize()   # wait for all the GPU related task to be finished
    t1 = time.time()
    dt = (t1 - t0)*1000   # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    # so in each time step, the total number of tokens processed will be (batch_size(num_prompts in a batch) * (length of sequence of each prompt))
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec:{tokens_per_sec:.2f}")


import sys; sys.exit(0)

# prefix tokens: The initial set of prompts (later converted to tokens) given by a user/ human, and we start with this set of prompts and later the GPT model starts generating tokens one after the other from here!

# prefix tokens
import tiktoken   # this is the library we use to tokenize the prompt
enc = tiktoken.get_encoding("gpt2")
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I am a language model,")    # Now the model will be completing this prompt
tokens = torch.tensor(tokens, dtype=torch.long)    # (8,) above sentence equates to 8 tokens
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)    # (5, 8) batch size of 5 sents
# x = tokens.to("cuda")
x = tokens.to(device)

# Now comes the generation phase
# x = (B, T), this is the input to the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: # each of the 5 sequences's length must be less than 30 as defined above, only generate new tokens
    with torch.no_grad():
        logits = model(x)    #(B, T, vocab_size)
        # Get the probabilities
        probs = F.softmax(logits, dim=-1)
        # Do top-k sampling of 50 (huggingface pipeline default)
        # top-k_probs here becomes (5, 50), tok_indices is also (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # top-50 probability values itself and also it's corresponding indices
        # Now, just select a single token from the top50 options
        # Note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # Gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)   # (B, 1)
        x = torch.cat((x, xcol), dim=1)    # Add the newly generated tokens to the existing set of tokens or the initial set of prompts, if we are generating tokens for the first time!

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)




############### just block of code to understand DDP in detail
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    """
    Main training function for distributed data parallel (DDP) setup.

    Args:
        rank (int): The rank of the current process (0 <= rank < world_size). Each process is assigned a unique rank.
        world_size (int): Total number of processes involved in the distributed training.
        save_every (int): Frequency of model checkpoint saving, in terms of epochs.
        total_epochs (int): Total number of epochs for training.
        batch_size (int): Number of samples processed in one iteration (forward and backward pass).
    """

    # Set up the distributed environment, including setting the master address, port, and backend.
    ddp_setup(rank, world_size)

    # Load the necessary training objects - dataset, model, and optimizer.
    dataset, model, optimizer = load_train_objs()

    # Prepare the data loader for distributed training. It partitions the dataset across the processes and handles shuffling.
    train_data = prepare_dataloader(dataset, batch_size)

    # Initialize the trainer instance with the loaded model, data, and other configurations.
    trainer = Trainer(model, train_data, optimizer, rank, save_every)

    # Train the model for the specified number of epochs.
    trainer.train(total_epochs)

    # Cleanup the distributed environment after training is complete.
    destroy_process_group()