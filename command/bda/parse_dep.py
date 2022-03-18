import glob

for fname in glob.glob(f'data-raw/bda_deps_raw/*'):
    with open(fname, 'r') as f:
        flag = fname.split('_')[-1]
        with open(f'data-raw/bda_deps/{flag}.dep', 'w') as wf:
            for line in f:
                line_split = line.split()
                if line_split[1] != '1':
                    continue

                wf.write(f'{hex(int(line_split[2]))[2:]} -> {hex(int(line_split[3]))[2:]}\n')
