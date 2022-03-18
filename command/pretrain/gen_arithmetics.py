import random
from keystone import *

try:
    from command import configs

    fields = configs.fields
except:
    fields = ['static', 'mem_mask', 'op_pos_emb',
              'mem1', 'mem2', 'mem3', 'mem4', 'mem5', 'mem6', 'mem7', 'mem8',
              'byte1', 'byte2', 'byte3', 'byte4', 'byte5', 'byte6', 'byte7', 'byte8']

opcode1 = ['add', 'sub', 'xor', 'or', 'and']
opcode2 = ['mov', 'cmp', 'test', 'push', 'pop', 'add', 'sub', 'mul', 'imul', 'div', 'inc', 'dec', 'xor', 'or', 'and',
           'shl', 'shr', 'sar', 'jmp', 'call']

reg_dict = {}
reg_dict[8] = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14',
               'r15']
reg_dict[4] = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d',
               'r14d', 'r15d']
reg_dict[2] = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp', 'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w',
               'r15w']
reg_dict[1] = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b',
               'r15b']

num_sample = 1000000


def byte2str(b):
    result = hex(b)
    result = result[2:]  # ignore 0x
    if len(result) == 1:
        result = '0' + result
    elif len(result) != 2:
        raise ValueError('result must have length either 1 or 2')

    return result


def get_op_result(opcode, reg_dest, reg_src):
    if opcode == 'add':
        return reg_dest + reg_src
    elif opcode == 'sub':
        return reg_dest - reg_src
    elif opcode == 'xor':
        return reg_dest ^ reg_src
    elif opcode == 'or':
        return reg_dest | reg_src
    elif opcode == 'and':
        return reg_dest & reg_src


def gen_bytes(opcode1_select, reg_len_dest, reg_len_src):
    if reg_len_dest == 8:
        reg_len_dest -= 1
    if reg_len_src == 8:
        reg_len_src -= 1

    reg_dest_value = random.randrange(1, 256 ** reg_len_dest)
    if opcode1_select == 'sub':
        # print(min(256 ** (reg_len_src), reg_dest_value), reg_dest_value, reg_len_dest)
        reg_src_value = random.randrange(
            min(256 ** reg_len_src, reg_dest_value))  # make sure the reg_src is smaller than reg_dest
    else:
        reg_src_value = random.randrange(256 ** reg_len_src)
    reg_dest_bytes = reg_dest_value.to_bytes(8, 'big')
    reg_src_bytes = reg_src_value.to_bytes(8, 'big')

    reg_src_value_after = get_op_result(opcode1_select, reg_dest_value, reg_src_value)
    reg_src_bytes_after = reg_src_value_after.to_bytes(8, 'big')

    reg_dest_list = [byte2str(b) for b in reg_dest_bytes]
    reg_src_list = [byte2str(b) for b in reg_src_bytes]
    reg_desg_list_after = [byte2str(b) for b in reg_src_bytes_after]

    return reg_dest_list, reg_src_list, reg_desg_list_after


def gen_mem(opcode, reg_dest, reg_src, hexvar_byte_list):
    if reg_src == 'hexvar':
        hexvar = '0x' + ''.join(hexvar_byte_list)
        code = f'{opcode} {reg_dest}, {hexvar}'
    else:
        code = f'{opcode} {reg_dest}, {reg_src}'

    ks = Ks(KS_ARCH_X86, KS_MODE_64)
    try:
        inst_bytecount = len(ks.asm(code)[0])
    except:
        # print(code)
        return False, [], []

    mem_curr = random.randrange(256 ** 5, 256 ** 6)
    mem_next = mem_curr + inst_bytecount
    mem_curr_list = [byte2str(b) for b in mem_curr.to_bytes(8, 'big')]
    mem_next_list = [byte2str(b) for b in mem_next.to_bytes(8, 'big')]

    return True, mem_curr_list, mem_next_list


def gen_sample():
    sample = dict()

    isvalid = False
    while not isvalid:
        reg_len_dest = random.choice(list(reg_dict.keys()))
        if reg_len_dest == 8:
            reg_len_src = random.choice([4, 8])
        else:
            reg_len_src = reg_len_dest

        # reg_len_src = random.choice(list(reg_dict.keys()))

        opcode1_select = random.choice(opcode1)
        reg_dest = random.choice(reg_dict[reg_len_dest])
        reg_src = random.choice(reg_dict[reg_len_src] + ['hexvar'])

        opcode2_select = random.choice(opcode2)

        reg_dest_list, reg_src_list, reg_desg_list_after = gen_bytes(opcode1_select, reg_len_dest, reg_len_src)
        isvalid, mem_curr_list, mem_next_list = gen_mem(opcode1_select, reg_dest, reg_src, reg_src_list)

    sample['static'] = f'{opcode1_select} {reg_dest} {reg_src} {opcode2_select} {reg_dest}'
    sample['op_pos_emb'] = '0 1 2 0 1'
    # sample['inst_pos_emb'] = '0 0 0 1 1'
    sample['mem_mask'] = '0 0 0 0 0'
    for i in range(8):
        sample['byte' + str(i + 1)] = f'## {reg_dest_list[i]} {reg_src_list[i]} ## {reg_desg_list_after[i]}'
        sample['mem' + str(
            i + 1)] = f'{mem_curr_list[i]} {mem_curr_list[i]} {mem_curr_list[i]} {mem_next_list[i]} {mem_next_list[i]}'

    return sample


# add n synthetic entries to dataset
def add_arithmetics(n):
    file_dict = {}
    for field in fields:
        file_dict[f'train_{field}'] = open(f'data-src/pretrain/train.{field}', 'a')
        file_dict[f'valid_{field}'] = open(f'data-src/pretrain/valid.{field}', 'a')

    for _ in range(n):
        sample = gen_sample()

        dataset = 'valid_'
        if random.random() > 0.01:
            dataset = 'train_'

        for field in sample.keys():
            file_dict[dataset + field].write(sample[field] + '\n')

    for field in fields:
        file_dict[f'train_{field}'].close()
        file_dict[f'valid_{field}'].close()


add_arithmetics(2000000)
