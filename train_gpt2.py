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
                # Special treatment for COnv1D weights since we need to transpose them
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # just simply copy over all the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    

model = GPT.from_pretrained('gpt2')
print("Didn't crash while copying the weights from a hugging gpt2 model to our implemented model")