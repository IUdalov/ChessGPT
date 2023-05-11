import sys
import os
sys.path.insert(1, os.path.join(os.path.abspath(""), "../third-party/nanoGPT"))

from model import GPTConfig, GPT
from contextlib import nullcontext
from tokenizer import encode, decode, GAME_LEN

import torch


# -----------------------------------------------------------------------------

temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------


class ChessGPT:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = self._load_model(checkpoint)

    def _load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        if compile:
            model = torch.compile(model)  # requires PyTorch 2.0 (optional)
        return model

    def _get_ctx(self):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        return nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    def next_moves(self, moves, n_moves=GAME_LEN, num_samples=3):
        assert num_samples > 0
        assert n_moves >= 1
        assert isinstance(moves, list)

        if len(moves) == 0 or moves[0] != "<beg>":
            moves = ["<beg>"] + moves

        start_ids = encode(moves)
        if len(start_ids) >= GAME_LEN - 2:
            return None

        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        n_tokens_to_gen = min(len(start_ids) + n_moves * 3, GAME_LEN)
        # run generation
        predicted_moves = list()
        with torch.no_grad():
            with self._get_ctx():
                for k in range(num_samples):
                    y = self.model.generate(x, n_tokens_to_gen, temperature=temperature, top_k=top_k)
                    predicted_moves.append(decode(y[0].tolist())[len(moves):len(moves) + n_moves])
        return predicted_moves