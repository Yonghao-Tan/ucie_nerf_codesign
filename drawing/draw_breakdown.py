# 绘制 breakdown 饼图


# 更学术风格的 breakdown 饼图

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 使用 DejaVu Sans，避免字体警告
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']

# 左侧饼图：计算 breakdown
labels1 = ['EMA', 'Comp SRAM', 'Comp PE', 'Interpolation', 'Others']
sizes1 = [5.16, 1.001, 3.648, 0.539, 0.17]
colors1 = plt.get_cmap('Set2').colors[:len(labels1)]

# 右侧饼图：存储 breakdown（示例数据，可自行修改）
labels2 = ['Off-chip', 'On-chip Comp', 'On-chip Interpolation']
sizes2 = [3.34, 2.54, 0.46] # 插值算力0.4TFLOPS?
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
	'Off-chip Memory: DDR3-1600, 12.8GB/s\nPE Array: 2048 MAC @ 500MHz -> 2TOPS\nInterpolation: ~0.4TFLOPS\nEMA: 112.54 pJ/B, PE: 0.718 pJ/op, SRAM: 3.153 pJ/B',
	ha='center', va='center', fontsize=13, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

plt.tight_layout()
plt.savefig('breakdown_pie.png', bbox_inches='tight', dpi=300)
plt.close()
