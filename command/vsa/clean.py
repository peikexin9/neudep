from command.configs import fields

for field in fields + ['label']:
    for split in ['train', 'valid']:
        with open(f'data-raw/finetune_vsa/{split}.{field}', 'r') as f:
            data = f.read().replace('\n', ' ').split()
        with open(f'data-src/finetune_vsa/{split}.{field}', 'w') as f:
            for i in range(0, len(data), 511):
                f.write(' '.join(data[i:i + 511]) + '\n')
