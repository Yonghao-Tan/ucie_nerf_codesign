import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.family'] = 'Calibri'
# 数据
window_size_labels = ['5x5', '10x10', '20x20', '40x40', '80x80', '160x160']

# 三种sample grouping的原始数据

sample_8 = [4556, 2539, 1529, 844, 450, 360]
sample_16 = [6181, 3326, 2030, 1173, 568, 247]
sample_48 = [9556, 4765, 2587, 1574, 938, 507]
max_value = max(max(sample_8), max(sample_16), max(sample_48))

# 对数据进行归一化（除以最大值）
sample_8 = [x / max_value for x in sample_8]
sample_16 = [x / max_value for x in sample_16]
sample_48 = [x / max_value for x in sample_48]

# 设置柱状图的位置和宽度
x = np.arange(len(window_size_labels))  # x轴位置
width = 0.25  # 柱子宽度

# 创建图形
fig, ax = plt.subplots(figsize=(12, 4))
base = 24
# 绘制三组柱状图
lw = 1.5
ec = 'black'
bars1 = ax.bar(x - width, sample_8, width, label='Sample Grouping = 8', color='#354285', alpha=1.0, zorder=5, linewidth=lw, edgecolor=ec)
bars2 = ax.bar(x, sample_16, width, label='Sample Grouping = 16', color='#6394d4', alpha=1.0, zorder=5, linewidth=lw, edgecolor=ec)
bars3 = ax.bar(x + width, sample_48, width, label='w/o Sample Grouping', color='#9bb4d5', alpha=1.0, zorder=5, linewidth=lw, edgecolor=ec)

# ax.plot([0., 1.], [0.5, 0.5], 
#         linestyle='-', color='red', linewidth=1.5, alpha=0.8, zorder=10)
# ax.axhline(y=0.5, color='red', linewidth=2, linestyle='-', zorder=10)

# 设置坐标轴
ax.set_xlabel('Window Size', fontsize=base, fontweight='bold')
ax.set_ylabel('Normalized EMA', fontsize=base, fontweight='bold', labelpad=-0)

ax.set_xticks(x)
ax.set_xticklabels(window_size_labels) # , rotation=45
ax.tick_params(axis='y', which='major', labelsize=base-6)
ax.tick_params(axis='x', which='major', labelsize=base-6, labelcolor='black')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# 设置y轴为对数坐标
# ax.set_yscale('log')

# 添加网格
ax.grid(True, alpha=0.3, axis='y')

# 添加图例
ax.legend(fontsize=base-6, loc='upper right')

# 在每个柱子顶部添加数值标签
# def add_value_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{int(height)}',
#                 ha='center', va='bottom', fontsize=8)

# add_value_labels(bars1)
# add_value_labels(bars2)
# add_value_labels(bars3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/home/ytanaz/access/IBRNet/memory/ema_bar_chart.png', dpi=400, bbox_inches='tight')
print("柱状图已保存到: /home/ytanaz/access/IBRNet/memory/ema_bar_chart.png")

# 显示图片
plt.show()
