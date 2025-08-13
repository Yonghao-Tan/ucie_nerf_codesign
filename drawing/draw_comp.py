import matplotlib.pyplot as plt

# 数据
x_labels = ['600x600', '800x800', '1200x1200']  # 分辨率作为 x 轴标签
y = [3.69, 6.75, 15.19]  # 数值作为 y 轴

# 创建图形
plt.figure(figsize=(8, 6))  # 设置画布大小

# 绘制柱状图，颜色设置为 #d9e2f7
bars = plt.bar(x_labels, y, color='#d9e2f7', edgecolor='black', linewidth=1.2)

# 添加数值标签到柱状图顶部
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

# 添加标题和标签
plt.title('Resolution vs TFLOPs', fontsize=16, fontweight='bold')
plt.xlabel('Resolution (HxW)', fontsize=14)  # X轴名称和单位
plt.ylabel('TFLOPs', fontsize=14)           # Y轴名称，无单位

# 调整刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 添加网格
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# 保存图像
plt.savefig('resolution_vs_tflops_bar.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()