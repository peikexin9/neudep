import shutil
import glob

from command import configs

for field in configs.fields + ['label', 'dep_cmp_emb']:
    with open(f'data-src/finetune_table3_all/valid.{field}', 'wb') as fw:
        for directory in glob.glob('data-src/finetune_table3/*'):
            shutil.copyfileobj(open(f'{directory}/valid.{field}', 'rb'), fw)
