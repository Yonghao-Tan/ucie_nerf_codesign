# 绘制 breakdown 饼图


# 更学术风格的 breakdown 饼图

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 使用 DejaVu Sans，避免字体警告
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']

labels1 = ['EMA', 'PE Comp.', 'Interpolation', 'Others']

# sizes1 = [1.496, 1.763, 0.124, 0.1]
sizes1 = [1.496, 1.663, 0.114, 0.08]
colors1 = plt.get_cmap('Set2').colors[:len(labels1)]

labels2 = ['EMA', 'PE Comp.', 'Interpolation', 'Others']

# sizes2 = [0.242, 0.344, 0.056, 0.015]
sizes2 = [0.302, 0.344, 0.041, 0.011]
colors2 = plt.get_cmap('Set3').colors[:len(labels2)]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

wedges1, texts1, autotexts1 = axs[0].pie(
	sizes1,
	labels=labels1,
	colors=colors1,
	autopct='%1.1f%%',
	startangle=140,
	wedgeprops={'edgecolor': 'white', 'linewidth': 2},
	textprops={'fontsize': 14, 'fontweight': 'bold'}
)
axs[0].set_title('Energy Breakdown', fontsize=16, fontweight='bold')
axs[0].axis('equal')

wedges2, texts2, autotexts2 = axs[1].pie(
	sizes2,
	labels=labels2,
	colors=colors2,
	autopct='%1.1f%%',
	startangle=140,
	wedgeprops={'edgecolor': 'white', 'linewidth': 2},
	textprops={'fontsize': 14, 'fontweight': 'bold'}
)
axs[1].set_title('Latency Breakdown', fontsize=16, fontweight='bold')
axs[1].axis('equal')


# 在图的空白处添加注释
fig.text(
	0.5, 0.05,
	'Off-chip Memory: LPDDR4-3200, 25.6GB/s\nOn-chip PE Array: 2048 MAC @ 1000MHz',
	ha='center', va='center', fontsize=13, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

plt.tight_layout()
plt.savefig('breakdown_pie.png', bbox_inches='tight', dpi=300)
plt.close()
