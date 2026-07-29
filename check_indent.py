with open('transformer.py', 'rb') as f:
    content = f.read()
idx = content.find(b'def sum')
for i, b in enumerate(content[idx:idx+60]):
    ch = chr(b) if 32 <= b < 127 else '.'
    print(f'{idx+i}: {b} = {ch}')