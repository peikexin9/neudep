import subprocess
from multiprocessing import Pool

from command.configs import fields


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain/{field}/dict.txt', '--trainpref',
         f'data-src/finetune_vsa/train.{field}',
         '--validpref',
         f'data-src/finetune_vsa/valid.{field}', '--destdir', f'data-bin/finetune_vsa/{field}',
         '--workers',
         '5'])


with Pool() as pool:
    pool.map(run, fields)

# subprocess.run(
#     ['fairseq-preprocess', '--only-source', '--trainpref',
#      f'data-src/finetune_vsa/train.label',
#      '--validpref',
#      f'data-src/finetune_vsa/valid.label', '--destdir', f'data-bin/finetune_vsa/label',
#      '--workers',
#      '5'])

labels = ['other', 'stack', 'heap', 'global']
for label in labels:
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--trainpref',
         f'data-src/finetune_vsa/train.label_{label}',
         '--validpref',
         f'data-src/finetune_vsa/valid.label_{label}', '--destdir', f'data-bin/finetune_vsa/label_{label}',
         '--workers',
         '5'])
