static_field = 'static'
op_pos_field = 'op_pos_emb'  # opcode/operand positional embedding
mem_mask_field = 'mem_mask'  # denote whether a token is memory access instruction

discrete_fields = [static_field] + [op_pos_field] + [mem_mask_field]

byte_fields = [f'byte{i}' for i in range(1, 9)]
byte_len = len(byte_fields)

mem_fields = [f'mem{i}' for i in range(1, 9)]
mem_len = len(mem_fields)

maskable_fields = [static_field] + byte_fields

fields = discrete_fields + byte_fields + mem_fields
