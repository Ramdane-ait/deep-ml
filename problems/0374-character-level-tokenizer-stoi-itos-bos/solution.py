import torch
import torch.nn as nn

class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        Includes a PyTorch nn.Embedding layer for the full vocabulary.

        Args:
            text: A string used to build the vocabulary.
        """
        # Your code here
        self.vocab = ['<BOS>', 'EOS'] + sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.stoi = { c:i for c,i in enumerate(self.vocab) }
        self.itos = { i:c for c,i in enumerate(self.vocab)}
        self.embedding = nn.Embedding(self.vocab_size, 1)
        

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a string into a 1D LongTensor of token indices,
        with BOS prepended and EOS appended.

        Args:
            text: The string to encode.
        Returns:
            torch.Tensor of dtype torch.long.
        """
        # Your code here
        seq = ['<BOS>'] + text + ['<EOS>']
        return torch.tensor([self.stoi[c] for c in seq])

    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode a 1D LongTensor (or list) of token indices back into a string.

        Args:
            indices: torch.Tensor or list of integer indices.
        Returns:
            Decoded string.
        """
        # Your code here
        
        return ''.join([self.itos[i] for i in indices])