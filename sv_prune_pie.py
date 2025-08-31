import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 数据定义
labels = [f'SV$_{i}$' for i in range(1, 9)]  # 使用 LaTeX 格式生成下标
values = [0.001, 0.35, 0.01, 0.25, 0.17, 0.03, 0.186, 0.003]
values = [0.16, 0.20, 0.02, 0.18, 0.15, 0.21, 0.05, 0.03] # a
# values = [0.25, 0.37, 0.29, 0.06, 0.01, 0.01, 0.01, 0.00] # b

max_value = max(values)
# 创建一个颜色映射（从浅蓝到深蓝）
cmap = plt.cm.get_cmap('Purples')  # 使用 Blues 色系
norm = plt.Normalize(min(values), max(values))  # 归一化值范围
colors = cmap(norm(values))  # 根据值生成对应的颜色

# 创建柱状图
# fig, ax = plt.figure(figsize=(10, 2.5))  # 设置图形大小
fig, ax = plt.subplots(figsize=(10, 2.5))
bars = plt.bar(labels, values, color=colors, edgecolor='black')  # 设置颜色和边缘线

# 在每个柱子上方标注数据值
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.0003, f'{value:.3f}',
             ha='center', va='bottom', fontsize=10)

# 设置标题和标签
# plt.title('Bar Chart of SV Values with Color Intensity')
plt.xlabel('Source Views')
plt.ylabel('Blending Weights')
ax.set_ylim(0, max_value+0.05)
# 添加颜色条（可选）
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 需要设置一个空数组以避免警告
# cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.02)
# cbar.set_label('Value Intensity')


# 调整布局以避免裁剪
plt.tight_layout()

# 保存为 PNG 文件
plt.savefig('bar_chart_colored.png', dpi=300)

# 显示图形（可选）
plt.show()