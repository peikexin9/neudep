from multiprocessing import Pool
import os
import subprocess

progs = ['objdump_bcfobf', 'objdump_cffobf', 'objdump_O0', 'objdump_O1', 'objdump_O2', 'objdump_O3', 'objdump_splobf',
         'objdump_subobf']


def run(name):
    os.chdir(f'/root/playground/bda/{name}')
    subprocess.run(['rexe', name])
    subprocess.run(['rexe', '-t', '300', name])
    dep_name = name.split('_')[-1]
    subprocess.run(['rdep', '-d', f'../bda_deps/{dep_name}.dep', name])
    subprocess.run(['rdep', name])


with Pool() as pool:
    pool.map(run, progs)
