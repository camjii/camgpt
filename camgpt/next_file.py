from tokenizer import Tokenizer

text = "hello hello hello world"

# base vocab is just the 256 raw byte values
vocab = {i: bytes([i]) for i in range(256)}

# empty dict to store merge mappings
merged = {}

# create tokenizer and train with 5 merges
tok = Tokenizer(text, vocab, num_merges=5, merged=merged)
ids = tok.train()

print(f"Text {text}")
print("Original bytes:", list(text.encode('utf-8')))
print("After merges:  ", ids)
print("Merge mapping: ", tok.merged)
print("Decoded: ", tok.decode(ids) )