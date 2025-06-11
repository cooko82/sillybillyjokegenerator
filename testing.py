from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-cased")
print(tokenizer)