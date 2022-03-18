import json
import os

# test:
# ../../../ghidra_9.1.2_PUBLIC/support/analyzeHeadless ../../../ghidra_play play -readOnly -import new_bins/addr2line_O0 -postScript ../../command/ghidra/get_inst_ref.py

getCurrentProgram().setImageBase(toAddr(0), 0)

fm = currentProgram.getFunctionManager()
ref = currentProgram.getReferenceManager()

d = {}
for fn in fm.getFunctions(True):
    funcname = fn.name

    if funcname != 'FUN_001352b0':
        continue

    d[funcname] = {}

    instr = getInstructionAt(fn.getBody().getMinAddress())
    if instr is None:
        continue

    print(funcname)
    while instr.getMinAddress() <= fn.getBody().getMaxAddress():
        if fn.getBody().contains(instr.getMinAddress()):
            ref_set = set()
            # iterate every address that forms an instruction
            instr_start = instr.getMinAddress()
            while instr_start <= instr.getMaxAddress():
                refs = ref.getReferencesFrom(instr_start)
                for each_ref in refs:
                    ref_set.add(each_ref.getToAddress().toString())
                instr_start = instr_start.add(1)
            print(instr.getMinAddress(), ref_set)
            print('============')

        instr = instr.getNext()
        if instr is None:
            break
    print('***************')
