import json
import os

# test:
# ../../../ghidra_9.1.2_PUBLIC/support/analyzeHeadless ../../../ghidra_play play -readOnly -import new_bins/addr2line_O0 -postScript ../../command/ghidra/get_inst_ref.py



output_dir = getScriptArgs()[0]

filepath = str(getProgramFile())
filename = filepath.split('/')[-1]

getCurrentProgram().setImageBase(toAddr(0), 0)

fm = currentProgram.getFunctionManager()
ref = currentProgram.getReferenceManager()
listing = currentProgram.getListing()
d = {}
for fn in fm.getFunctions(True):
    funcname = fn.name
    body = fn.getBody()
    function_addresses = body.getAddresses(True)
    ref_set = set()
    while function_addresses.hasNext():
        instruction = listing.getInstructionAt(function_addresses.next())
        if instruction:
            refs = ref.getReferencesFrom(instruction.getMinAddress())
            for each_ref in refs:
                ref_set.add(each_ref.getToAddress().toString())
            d[str(instruction.getMinAddress())] = list(ref_set)
    # instr = getInstructionAt(fn.getBody().getMinAddress())
    # if instr is None:
    #     continue

    # while instr.getMinAddress() <= fn.getBody().getMaxAddress():
    #     if fn.getBody().contains(instr.getMinAddress()):
    #         ref_set = set()
    #         # iterate every address that forms an instruction
    #         instr_start = instr.getMinAddress()
    #         while instr_start <= instr.getMaxAddress():
    #             refs = ref.getReferencesFrom(instr_start)
    #             for each_ref in refs:
    #                 ref_set.add(each_ref.getToAddress().toString())
    #             instr_start = instr_start.add(1)
    #         d[funcname][str(instr.getMinAddress())] = list(ref_set)
            

        # instr = instr.getNext()
        # if instr is None:
        #     break
with open(os.path.join(output_dir, filename + '.json'), 'w') as f:
    f.write(json.dumps(d, indent=4))
