# 绘制 breakdown 饼图


# 更学术风格的 breakdown 饼图

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 使用 DejaVu Sans，避免字体警告
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
color_list = ['#cdb6de', '#ac9fe8', "#f2f2f2", '#9bb4e9', '#f1ccf0']

# 左侧饼图：计算 breakdown
labels1 = ['EMA', 'Comp\nSRAM', 'Comp PE', 'Interpolation\n', 'Others']
labels1 = ['EMA', 'PE Comp.', 'Others']
sizes1 = [5.16, 1.001, 3.648, 0.539, 0.17]
sizes1 = [2.447, 2.221, 0.136+0.08] # DDR3
# sizes1 = [2.567, 2.221, 0.136+0.08] # DDR3
# colors1 = plt.get_cmap('tab20b').colors[:len(labels1)]
colors1 = color_list[:len(labels1)]

# 右侧饼图：存储 breakdown（示例数据，可自行修改）
labels2 = ['Off-Chip', 'On-Chip\nComp', 'On-Chip\nInterpolation']
labels2 = ['EMA', 'PE Comp.', 'Others']
sizes2 = [3.34, 2.54, 0.46] # 插值算力0.4TFLOPS?
sizes2 = [0.500, 0.452, 0.046+0.012] # DDR3
# colors2 = plt.get_cmap('tab20b').colors[:len(labels2)]
colors2 = color_list[:len(labels2)]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

wedges1, texts1, autotexts1 = axs[0].pie(
	sizes1,
	labels=labels1,
	colors=colors1,
	autopct='%1.1f%%',
	startangle=140,
	wedgeprops={'edgecolor': 'white', 'linewidth': 2},
	textprops={'fontsize': 20, 'fontweight': 'bold'}
)
axs[0].set_title('Energy Breakdown', fontsize=33, fontweight='bold')
axs[0].axis('equal')

wedges2, texts2, autotexts2 = axs[1].pie(
	sizes2,
	labels=labels2,
	colors=colors2,
	autopct='%1.1f%%',
	startangle=140,
	wedgeprops={'edgecolor': 'white', 'linewidth': 2},
	textprops={'fontsize': 20, 'fontweight': 'bold'}
)
axs[1].set_title('Latency Breakdown', fontsize=33, fontweight='bold')
axs[1].axis('equal')


# 在图的空白处添加注释
fig.text(
	0.5, 0.02,
	'Off-chip Memory: DDR3-1600, 12.8GB/s\nOn-chip PE Array: 2048 MAC @ 1200MHz',
	ha='center', va='center', fontsize=19, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

plt.tight_layout()
plt.savefig('breakdown_pie.png', bbox_inches='tight', dpi=300)
plt.close()
