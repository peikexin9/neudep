import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.rc('font', family='Times New Roman')

tools = ['Xdep', 'Angr', 'SVF']
x = np.arange(4) + 1

Xdep = [.99, .91, 0.9, 0.86]
Angr = [.75, .77, .71, 0.62]
SVF = [.99, .92, 0.9, 0.9]

fig_legend = plt.figure()
fig, ax = plt.subplots(figsize=(9, 6))

patterns = ["+", "x", "//"]
ax1_emb = ax.bar(x - 0.25, Xdep, width=0.25, hatch=patterns[0], align='center', alpha=0.7, edgecolor='black')
ax2_emb = ax.bar(x, Angr, width=0.25, hatch=patterns[1], align='center', alpha=0.7, edgecolor='black')
ax3_emb = ax.bar(x + 0.25, SVF, width=0.25, hatch=patterns[2], align='center', alpha=0.7, edgecolor='black')

ax.tick_params(axis='both', which='major', labelsize=18)

plt.xlim([0.5, len(x) + 0.5])
ax.set_xticks(x)
ax.set_xticklabels(['O0', 'O1', 'O2', 'O3'], fontsize=24)

# plt.legend((ax1_emb, ax2_emb, ax3_emb), tools, loc='best', ncol=1, handlelength=3, fontsize=16)

# plt.ylabel('Accuracy', fontsize=18)
# plt.yscale('log')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# plt.show()
plt.savefig('figs/xdep_angr_svf_precision.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)

fig_legend.legend((ax1_emb, ax2_emb, ax3_emb), tools, loc='upper center', ncol=3,
                  frameon=False, handlelength=5, fontsize=18)
fig_legend.savefig(f'figs/xdep_angr_svf_legend.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
