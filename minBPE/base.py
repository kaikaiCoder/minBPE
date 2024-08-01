class Tokenizer:

  def __init__(self):
    self.patterns = ""
    self.merges = {}
    self.special_tokens = {}
    self.vocab = {}

  def train(self,corpu,vocab_size,verbose=False):
     NotImplementedError()
  
  def encode(self, text):
     NotImplementedError()
  
  def decode(self,ids):
      NotImplementedError()