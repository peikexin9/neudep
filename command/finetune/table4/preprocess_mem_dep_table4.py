import glob
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

from command.configs import fields

for folder in glob.glob('data-src/finetune_table4/*'):
    dest_folder = folder.replace('data-src', 'data-bin')


    def run(field):
        subprocess.run(
            ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain/{field}/dict.txt', '--trainpref',
             f'{folder}/train.{field}',
             '--validpref',
             f'{folder}/valid.{field}', '--destdir', f'{dest_folder}/{field}',
             '--workers',
             '5'])


    with Pool() as pool:
        pool.map(run, fields)

    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--trainpref', f'{folder}/train.dep_cmp_emb',
         '--validpref',
         f'{folder}/valid.dep_cmp_emb', '--destdir', f'{dest_folder}/dep_cmp_emb',
         '--workers',
         '10'])

    Path(f'{dest_folder}/label/').mkdir(parents=True, exist_ok=True)

    shutil.copy(f'{folder}/train.label', f'{dest_folder}/label/')
    shutil.copy(f'{folder}/valid.label', f'{dest_folder}/label/')
