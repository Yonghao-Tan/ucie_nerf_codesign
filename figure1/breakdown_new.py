# 绘制 breakdown 饼图


# 更学术风格的 breakdown 饼图

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 使用 DejaVu Sans，避免字体警告
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']

labels1 = ['EMA', 'PE Comp.', 'Interpolation', 'Others']
labels1 = ['EMA', 'PE Comp.', 'Others']

# sizes1 = [1.496, 1.763, 0.124, 0.1]
# sizes1 = [1.496, 1.663, 0.114, 0.08]
sizes1 = [2.447, 2.221, 0.156, 0.12] # DDR3
sizes1 = [2.447, 2.221, 0.136+0.08] # DDR3
# colors1 = plt.get_cmap('Set2').colors[:len(labels1)]
colors1 = ['#cdb6de', '#ac9fe8', '#f2f2f2']
colors2 = ['#cdb6de', '#ac9fe8', '#f2f2f2']

labels2 = ['EMA', 'PE Comp.', 'Interpolation', 'Others']
labels2 = ['EMA', 'PE Comp.', 'Others']

# sizes2 = [0.242, 0.344, 0.056, 0.015]
# sizes2 = [0.302, 0.344, 0.041, 0.011]
sizes2 = [0.500, 0.452, 0.046+0.012] # DDR3
# colors2 = plt.get_cmap('Set3').colors[:len(labels2)]

base = 16
fig, axs = plt.subplots(1, 2, figsize=(8.3, 4.3))

wedges1, texts1, autotexts1 = axs[0].pie(
	sizes1,
	labels=labels1,
	colors=colors1,
	autopct='%1.1f%%',
	startangle=140,
	wedgeprops={'edgecolor': 'white', 'linewidth': 2},
	textprops={'fontsize': base, 'fontweight': 'bold'}
)
axs[0].set_title('Energy Breakdown', fontsize=base+6, fontweight='bold', pad=-2000)
axs[0].axis('equal')

wedges2, texts2, autotexts2 = axs[1].pie(
	sizes2,
	labels=labels2,
	colors=colors2,
	autopct='%1.1f%%',
	startangle=140,
	wedgeprops={'edgecolor': 'white', 'linewidth': 2},
	textprops={'fontsize': base, 'fontweight': 'bold'}
)
axs[1].set_title('Latency Breakdown', fontsize=base+6, fontweight='bold', pad=-200)
axs[1].axis('equal')


# 在图的空白处添加注释
# fig.text(
# 	0.5, 0.05,
# 	'Off-chip Memory: LPDDR4-3200, 25.6GB/s\nOn-chip PE Array: 2048 MAC @ 1000MHz',
# 	ha='center', va='center', fontsize=13, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
# )
fig.text(
	0.5, 0.05,
	'Off-chip Memory: DDR3-1600, 25.6GB/s\nOn-chip PE Array: 2048 MAC @ 800MHz',
	ha='center', va='center', fontsize=base-1, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

plt.tight_layout()
plt.savefig('breakdown_pie.png', bbox_inches='tight', dpi=400)
plt.close()
