from command.configs import fields

labels = ['other', 'stack', 'heap', 'global']
for field in fields + [f'label_{label}' for label in labels]:
    train_orig = open(f'data-raw/finetune_vsa/train.{field}', 'r')
    valid_orig = open(f'data-raw/finetune_vsa/valid.{field}', 'r')

    train_write = open(f'data-src/finetune_vsa/train.{field}', 'w')
    valid_write = open(f'data-src/finetune_vsa/valid.{field}', 'w')

    for i, line in enumerate(train_orig):
        if i % 5 == 0:
            valid_write.write(line)
        else:
            train_write.write(line)

    for i, line in enumerate(valid_orig):
        if i % 5 == 0:
            valid_write.write(line)
        else:
            train_write.write(line)
