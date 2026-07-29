import base64, zlib

with open(r'D:\Inteligente\rubidium-api\train_pytorch.py', 'r', encoding='utf-8') as f:
    code = f.read()

compressed = zlib.compress(code.encode('utf-8'))
encoded = base64.b64encode(compressed).decode('ascii')

cell_code = 'import base64, zlib\n'
cell_code += '_data = "' + encoded + '"\n'
cell_code += 'code = zlib.decompress(base64.b64decode(_data)).decode("utf-8")\n'
cell_code += 'exec(code)\n'

with open(r'D:\Inteligente\rubidium-api\kaggle_cell_torch.txt', 'w', encoding='utf-8') as f:
    f.write(cell_code)
print(f'Cell: {len(cell_code)} chars')
