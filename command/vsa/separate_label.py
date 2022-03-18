labels = ['other', 'stack', 'heap', 'global']
splits = ['train', 'valid']

for label in labels:
    for split in splits:
        with open(f'data-src/finetune_vsa/{split}.label', 'r') as f:
            with open(f'data-src/finetune_vsa/{split}.label_{label}', 'w') as wf:
                for line in f:
                    line_write = []
                    for token in line.split():
                        if token == label:
                            line_write.append(token)
                        else:
                            line_write.append('none')
                    wf.write(' '.join(line_write) + '\n')
