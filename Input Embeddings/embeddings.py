import tiktoken


encoding = tiktoken.get_encoding("cl100k_base")

text = "Hey there! just experimenting with tokenization."

# Turn text into tokens
tokens = encoding.encode(text)
print(f"Tokens: {tokens}")

# Turn tokens back into the actual strings to see the "splits"
parts = [encoding.decode_single_token_bytes(t) for t in tokens]

print(f"Token strings: {parts}")