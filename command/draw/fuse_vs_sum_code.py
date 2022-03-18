import json
import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl

fuse = []
sum = []
fuse_old = []
sum_old = []

with open('result/pretrain', 'r') as f:
    for line in f:
        if '\"code_ppl\"' in line:
            line_json = json.loads(line.split('|')[-1])
            fuse.append(float(line_json['code_ppl']))

with open('result/pretrain_sum', 'r') as f:
    for line in f:
        if '\"code_ppl\"' in line:
            line_json = json.loads(line.split('|')[-1])
            sum.append(float(line_json['code_ppl']))

with open('result/pretrain_old', 'r') as f:
    for line in f:
        if '\"code_ppl\"' in line:
            line_json = json.loads(line.split('|')[-1])
            fuse_old.append(float(line_json['code_ppl']))

with open('result/pretrain_sum_old', 'r') as f:
    for line in f:
        if '\"code_ppl\"' in line:
            line_json = json.loads(line.split('|')[-1])
            sum_old.append(float(line_json['code_ppl']))

min_len = min(len(fuse),
              len(sum),
              len(fuse_old),
              len(sum_old),)

x = np.arange(min_len)

fig_legend = plt.figure()
fig, ax = plt.subplots(figsize=(9, 6))

patterns = ['o-', 'x-', '+--', 'o--']
ax1 = ax.plot(fuse[:min_len], patterns[0], linewidth=2, color='C1', label='Fuse')
ax2 = ax.plot(sum[:min_len], patterns[1], linewidth=2, color='C2', label='Sum')
ax3 = ax.plot(fuse_old[:min_len], patterns[2], linewidth=2, color='C3', label='Fuse_old')
ax4 = ax.plot(sum_old[:min_len], patterns[3], linewidth=2, color='C4', label='Sum_old')

# ax.set_xticks(x)
# ax.set_xticklabels(x + 1)
ax.tick_params(axis='both', which='major', labelsize=24)

plt.xlabel('Pretrain Iterations', fontsize=26)
plt.ylabel('Perplexity', fontsize=26)

# plot y-axis
axes = plt.gca()
axes.yaxis.grid(True, ls='--')

ax.legend(loc='best', handlelength=4, fontsize=20, bbox_to_anchor=(0, 1))

# plt.show()
plt.savefig('figs/fuse_vs_sum_code.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
