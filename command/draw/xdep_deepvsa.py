import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

plt.style.use('seaborn-dark-palette')
mpl.rc('font', family='Times New Roman')

tools = ['XDep', 'DeepVSA', 'Bi-LSTM', 'Bi-GRU', 'Bi-RNN', 'CRF', 'HMM']
x = np.arange(4) + 1

xdep = [0.991, 0.99, 0.992, 0.995]  # 0.991
hmm = [0.724, 0.455, 0.778, 0.859, ]  # 0.92
crf = [0.125, 0.146, 0.566, 0.794, ]  # 0.92
birnn = [0.908, 0.779, 0.956, 0.964, ]  # 0.92
bigru = [0.894, 0.827, 0.963, 0.971, ]  # 0.92
bilstm = [0.907, 0.823, 0.961, 0.966, ]  # 0.92
deepvsa = [0.915, 0.953, 0.987, 0.994, ]  # 0.92

colors = ['navy', 'magenta', 'gold', 'deepskyblue', 'mediumpurple', 'mediumseagreen', 'tomato', 'gray']

fig_legend = plt.figure()
fig, ax = plt.subplots(figsize=(10, 3))

# ax1 = ax.bar(x - 0.3, xdep, color=colors[0], width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[0])
# ax2 = ax.bar(x - 0.2, deepvsa, color=colors[1], width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[1])
# ax3 = ax.bar(x - 0.1, bilstm, color=colors[2], width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[2])
# ax4 = ax.bar(x, bigru, color=colors[3], width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[3])
# ax5 = ax.bar(x + 0.1, birnn, color=colors[4], width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[4])
# ax6 = ax.bar(x + 0.2, crf, color=colors[5], width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[5])
# ax7 = ax.bar(x + 0.3, hmm, color=colors[6], width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[6])

ax1 = ax.bar(x - 0.3, xdep, width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[0])
ax2 = ax.bar(x - 0.2, deepvsa, width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[1])
ax3 = ax.bar(x - 0.1, bilstm, width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[2])
ax4 = ax.bar(x, bigru, width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[3])
ax5 = ax.bar(x + 0.1, birnn, width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[4])
ax6 = ax.bar(x + 0.2, crf, width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[5])
ax7 = ax.bar(x + 0.3, hmm, width=0.1, align='center', alpha=0.7, edgecolor='black', label=tools[6])

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xlim([0.5, len(x) + .5])
ax.set_xticks(x)
ax.set_xticklabels(['Global', 'Heap', 'Stack', 'Other'], fontsize=15, fontweight='bold')

plt.legend(ncol=7, loc='upper center', bbox_to_anchor=(.48, 1.17), fontsize=13, columnspacing=0.5, labelspacing=0)

plt.ylabel('F1 score', fontsize=15, fontweight='bold')

plt.grid(color='grey', linestyle=':', linewidth=.5)
plt.grid(axis='x', color='grey', linestyle=':', linewidth=.5, alpha=0)

# plt.show()
plt.savefig('figs/xdep_deepvsa.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)

# fig_legend.legend((ax1, ax2, ax3, ax4, ax5, ax6, ax7), tools, loc='upper center', ncol=3, frameon=False, handlelength=5,
#                   fontsize=18)
# fig_legend.savefig(f'figs/xdep_deepvsa_legend.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
