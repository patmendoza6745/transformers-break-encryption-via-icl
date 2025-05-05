import numpy as np

class EncryptionScheme():
    def enc(self, plain):
        raise NotImplementedError("Must be implemented in subclass")
    
    def dec(self, cipher):
        raise NotImplementedError("Must be implemented in subclass")

class MonoAlphabetic(EncryptionScheme):
    def __init__(self, alpha_ids):
        self.alpha_ids = alpha_ids
        self.perm = np.random.permutation(26)
        self.enc  = {self.alpha_ids[i]: self.alpha_ids[self.perm[i]] for i in range(26)}   # plain→cipher
        self.dec  = {v: k for k, v in self.enc.items()}                          # cipher→plain

    def enc(self, plain):
        return self.enc[plain]

    def dec(self, cipher):
        return self.dec[cipher]

class Vigenere(EncryptionScheme):
    def __init__(self, key_length, alpha_ids):
        self.alpha_ids = alpha_ids
        self.key_length = key_length
        self.key = np.random.choice(self.alpha_ids, size=key_length, replace=True)
        self.enc_idx = 0
        self.dec_idx = 0

    def enc(self, plain):
        shift = self.key[self.enc_idx % self.key_length]
        c = (plain + shift) % len(self.alpha_ids) 
        self.enc_idx += 1
        return c

    def dec(self, cipher):
        shift = self.key[self.dec_idx % self.key_length]
        p = (cipher - shift) % len(self.alpha_ids)
        self.dec_idx += 1
        return p