flag = 'bcfobf'

pos = set()
neg = set()
with open(f'data-raw/bda_deps_raw/mem_dep_pair_of_addr_bda_{flag}', 'r') as f:
    for line in f:
        line_split = line.split()
        if line_split[1] == '1':
            pos.add(f'{hex(int(line_split[2]))[2:]} -> {hex(int(line_split[3]))[2:]}')
        else:
            neg.add(f'{hex(int(line_split[2]))[2:]} -> {hex(int(line_split[3]))[2:]}')

TP = 0
FP = 0
with open(f'data-raw/bda_deps_pred/objdump_{flag}.dep', 'r') as f:
    for line in f:
        line_split = line.split()
        if len(line_split) != 4:
            continue
        addr1 = line_split[0][2:].lower()
        addr2 = line_split[2][2:].lower()
        if f'{addr1} -> {addr2}' in pos or f'{addr2} -> {addr1}' in pos:
            TP += 1
        if f'{addr1} -> {addr2}' in neg or f'{addr2} -> {addr1}' in neg:
            FP += 1
print(TP, FP)
