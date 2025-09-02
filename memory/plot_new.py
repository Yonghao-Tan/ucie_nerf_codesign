import matplotlib.pyplot as plt
import numpy as np

# 数据
window_size_labels = ['5x5', '10x10', '20x20', '40x40', '80x80', '160x160']
x_positions = list(range(len(window_size_labels)))  # 均匀间隔的x轴位置


base = 28
sample_8 = [9556, 3039, 1147, 544, 328, 240, 144]
sample_16 = [16881, 4826, 1530, 573, 268, 158, 73]
sample_48 = [29053, 7765, 2187, 674, 238, 101, 24]


sample_8 = [4556, 2539, 1529, 844, 450, 360]
sample_16 = [6181, 3326, 2030, 1173, 568, 247]
sample_48 = [9556, 4765, 2587, 1574, 938, 507]
# for i in range(len(sample_8)):
#     if i > 0 and i < len(sample_8)-1:
#         sample_8[i] *= 1.8
#         sample_16[i] *= 1.8
#         sample_48[i] *= 1.8
        
max_value = max(max(sample_8), max(sample_16), max(sample_48))

# 对数据进行归一化（除以最大值）
sample_8_normalized = [x / max_value for x in sample_8]
sample_16_normalized = [x / max_value for x in sample_16]
sample_48_normalized = [x / max_value for x in sample_48]
# for i in range(len(sample_8)):
#     print(f"{sample_8_normalized[i]:.4f}")
#     print(f"{sample_16_normalized[i]:.4f}")
#     print(f"{sample_48_normalized[i]:.4f}")

# 创建图形，包含两个子图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 第一个子图：折线图
ax1.plot(x_positions, sample_8_normalized, marker='o', linewidth=2, label='Sample Grouping = 8', color='blue')
ax1.plot(x_positions, sample_16_normalized, marker='s', linewidth=2, label='Sample Grouping = 16', color='red')
ax1.plot(x_positions, sample_48_normalized, marker='^', linewidth=2, label='w/o Sample Grouping', color='green')

ax1.set_xlabel('Window Size', fontsize=base, fontweight='bold')
ax1.set_ylabel('Normalized EMA', fontsize=base, fontweight='bold')
# ax1.set_title('Normalized EMA vs Window Size for Different Sample Grouping', fontsize=16)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(window_size_labels, rotation=45)
ax1.tick_params(axis='y', which='major', labelsize=base-6)
ax1.tick_params(axis='x', which='major', labelsize=base-8, labelcolor='black')
# ax1.set_yscale('log')  # 设置y轴为对数
# ax1.set_ylim(1e-2, 1.1)  # 避免0导致报错
# 设置Y轴显示为百分比
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=base-8)

# 调整布局
plt.tight_layout()
# 保存图片
plt.savefig('eff_new.png', dpi=400, bbox_inches='tight')