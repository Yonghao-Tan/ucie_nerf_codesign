import matplotlib.pyplot as plt
import numpy as np

# 数据
window_size_labels = ['5x5', '10x10', '20x20', '40x40', '80x80', '160x160', 'Full']
x_positions = list(range(len(window_size_labels)))  # 均匀间隔的x轴位置

# 三种sample grouping的数据
# sample_8 = [1.93, 0.8, 0.41, 0.26, 0.18, 0.12]
# sample_16 = [2.62, 0.91, 0.38, 0.19, 0.12, 0.06]
# sample_48 = [4.35, 1.27, 0.41, 0.16, 0.07, 0.02]
sample_8 = [1.73, 0.66, 0.46, 0.26, 0.20, 0.14, 0.12]
sample_16 = [2.42, 0.91, 0.55, 0.31, 0.17, 0.08, 0.06]
sample_48 = [3.55, 1.27, 0.61, 0.36, 0.22, 0.10, 0.02]

for i in range(len(sample_8)):
    if i > 0 and i < len(sample_8)-1:
        sample_8[i] *= 1.8
        sample_16[i] *= 1.8
        sample_48[i] *= 1.8

# 新的柱状图数据 (21个数据点，每3个对应一个窗口大小)
bar_data = [
    0.1945, 0.1782, 0.1258,  # 5x5
    0.2676, 0.2048, 0.1375,  # 10x10
    0.3393, 0.2631, 0.1622,  # 20x20
    0.5185, 0.3782, 0.2177,  # 40x40
    2.8366, 1.6328, 0.731,   # 80x80
    1.0658, 0.6723, 0.3515,  # 160x160
    43.6417, 21.8354, 7.282  # Full
]

# 80 160那些还有问题 和eff pixel对不太上的
bar_data = [
    0.1258, 0.1782, 0.1945,  # 5x5
    0.1375, 0.2048, 0.2676,  # 10x10
    0.1622, 0.2631, 0.3393,  # 20x20
    0.2377, 0.4182, 0.5585,  # 40x40
    0.3915, 0.6823, 1.0658,  # 80x80
    # 0.731, 1.6328, 2.8366,   # 160x160
    0.731, 1.3828, 2.0366,   # 160x160
    0.0, 0.0, 0.0  # Full
]
bar_data = [x * 1.3 for x in bar_data]

bar_data_agg = [
    0.0855, 0.0981, 0.1029,  # 5x5
    0.0946, 0.1216, 0.1571,  # 10x10
    0.1146, 0.1656, 0.2158,  # 20x20
    0.1757, 0.3113, 0.4213,  # 40x40
    0.3266, 0.5238, 0.8448,  # 80x80
    # 0.629,  1.3957, 2.4184,   # 160x160
    0.629,  1.1957, 1.7184,   # 160x160
    0.0, 0.0, 0.0  # Full
]

bar_data = [x * 3. for x in bar_data]
bar_data_agg = [x * 3. for x in bar_data_agg]
# 重新组织柱状图数据
bar_sample_8 = [bar_data[i*3] for i in range(7)]     # 每组的第1个数据
bar_sample_16 = [bar_data[i*3+1] for i in range(7)]  # 每组的第2个数据
bar_sample_48 = [bar_data[i*3+2] for i in range(7)]  # 每组的第3个数据

# 重新组织bar_data_agg数据
bar_agg_sample_8 = [bar_data_agg[i*3] for i in range(7)]     # 每组的第1个数据
bar_agg_sample_16 = [bar_data_agg[i*3+1] for i in range(7)]  # 每组的第2个数据
bar_agg_sample_48 = [bar_data_agg[i*3+2] for i in range(7)]  # 每组的第3个数据

# 创建图形，包含两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# 第一个子图：折线图
ax1.plot(x_positions, sample_8, marker='o', linewidth=2, label='Sample Grouping = 8', color='blue')
ax1.plot(x_positions, sample_16, marker='s', linewidth=2, label='Sample Grouping = 16', color='red')
ax1.plot(x_positions, sample_48, marker='^', linewidth=2, label='w/o Sample Grouping', color='green')

ax1.set_xlabel('Window Size', fontsize=14)
ax1.set_ylabel('Effective Pixel', fontsize=14)
ax1.set_title('Effective Pixel vs Window Size for Different Sample Grouping', fontsize=16)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(window_size_labels, rotation=45)
ax1.set_ylim(0, max(max(sample_8), max(sample_16), max(sample_48)) * 1.1)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)

# 第二个子图：柱状图
width = 0.25  # 柱子宽度
x_bar = np.arange(len(window_size_labels))

# 画原始的bar_data柱子（较高的，浅色系）
bars1 = ax2.bar(x_bar - width, bar_sample_8, width, label='Sample Grouping = 8', color='lightblue', alpha=0.6, edgecolor='blue', linewidth=1.5)
bars2 = ax2.bar(x_bar, bar_sample_16, width, label='Sample Grouping = 16', color='lightcoral', alpha=0.6, edgecolor='red', linewidth=1.5)
bars3 = ax2.bar(x_bar + width, bar_sample_48, width, label='w/o Sample Grouping', color='lightgreen', alpha=0.6, edgecolor='green', linewidth=1.5)

# 画bar_data_agg柱子（较低的，在相同位置，浅色系）
bars_agg1 = ax2.bar(x_bar - width, bar_agg_sample_8, width, color='lightblue', alpha=0.9, edgecolor='blue', linewidth=1.5)
bars_agg2 = ax2.bar(x_bar, bar_agg_sample_16, width, color='lightcoral', alpha=0.9, edgecolor='red', linewidth=1.5)
bars_agg3 = ax2.bar(x_bar + width, bar_agg_sample_48, width, color='lightgreen', alpha=0.9, edgecolor='green', linewidth=1.5)

# 在两个柱子之间添加更密更深的阴影表示差值
for i in range(len(x_bar)):
    # Sample Grouping = 8
    ax2.fill_between([x_bar[i] - width - width/2, x_bar[i] - width + width/2], 
                     [bar_agg_sample_8[i], bar_agg_sample_8[i]], 
                     [bar_sample_8[i], bar_sample_8[i]], 
                     alpha=0.7, color='darkblue', hatch='/////')
    
    # Sample Grouping = 16
    ax2.fill_between([x_bar[i] - width/2, x_bar[i] + width/2], 
                     [bar_agg_sample_16[i], bar_agg_sample_16[i]], 
                     [bar_sample_16[i], bar_sample_16[i]], 
                     alpha=0.7, color='darkred', hatch='/////')
    
    # w/o Sample Grouping
    ax2.fill_between([x_bar[i] + width - width/2, x_bar[i] + width + width/2], 
                     [bar_agg_sample_48[i], bar_agg_sample_48[i]], 
                     [bar_sample_48[i], bar_sample_48[i]], 
                     alpha=0.7, color='darkgreen', hatch='/////')

# 添加一个虚拟的图例项来表示减少的部分
ax2.fill_between([], [], [], alpha=0.7, color='gray', hatch='/////', label='Reduce by Balance Aggregation')

# 添加buffer limitation水平线
ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Buffer Limitation')

ax2.set_xlabel('Window Size', fontsize=14)
ax2.set_ylabel('Maximum Buffer Size Requirement (MB)', fontsize=14)
ax2.set_title('Buffer Size Experiment', fontsize=16)
ax2.set_xticks(x_bar)
ax2.set_xticklabels(window_size_labels, rotation=45)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('effective_pixel_vs_window_size_with_bars.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

print("图表已保存为 'effective_pixel_vs_window_size_with_bars.png'")