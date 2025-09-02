import matplotlib.pyplot as plt

# 数据
x_labels = ['600x600', '800x800', '1200x1200']  # 分辨率作为 x 轴标签
base = 14057+9556
y = [base*6*6/(8*8)/1000, base/1000, base*12*12/(8*8)/1000]  # 数值作为 y 轴

# 创建图形
plt.figure(figsize=(6.64, 4.1))  # 设置画布大小
base = 26
# 绘制柱状图，颜色设置为 #d9e2f7
bars = plt.bar(x_labels, y, color='#d0d0d0', edgecolor='black', linewidth=1.2, zorder=10)

# 添加数值标签到柱状图顶部
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=base-6)

# 添加标题和标签
plt.title('Resolution vs EMA', fontsize=base, fontweight='bold', pad=25)
plt.xlabel('Resolution (HxW)', fontsize=base-1)  # X轴名称和单位
plt.ylabel('EMA (GB)', fontsize=base-1)         # Y轴名称和单位

# 调整刻度字体大小
plt.xticks(fontsize=base-6)
plt.yticks(fontsize=base-6)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 60)
# 添加网格
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# 保存图像
plt.savefig('resolution_vs_ema_bar.png', dpi=400, bbox_inches='tight')

# 显示图像
plt.show()