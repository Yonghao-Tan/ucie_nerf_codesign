import matplotlib.pyplot as plt
import numpy as np

def create_d2d_transfer_chart():
    """
    创建D2D传输总量对比图表
    """
    # 数据
    categories = ['Baseline', 'w/ Full-SR', 'w/ PDU']
    # 计算总和值
    # a, b, c = 28.99, 28.15, 28.15+0.57
    a, b, c = 28.99, 28.15, 28.15+0.71
    a, b, c = 26.14, 26.14-0.54, 26.14-0.54+0.41
    values = [a, b, c]
    colors = ['#A0A0A0', '#707070', '#505050']  # 渐变灰色
    base = 26
    
    # 创建图表 - 调整为更合适的比例
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 创建柱状图
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    
    # 设置坐标轴范围
    ax.set_ylim(25.5, 26.25)
    ax.set_yticks([25.5, 25.75, 26.0, 26.25])
    
    # 在柱子内部添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        # 在柱子中间位置添加文字（相对于柱子的实际位置）
        y_bottom = 25.5  # y轴下限
        y_center = y_bottom + (height - y_bottom) / 2  # 柱子的中间位置
        ax.text(bar.get_x() + bar.get_width()/2., y_center,
                f'{value:.2f}', ha='center', va='center', 
                fontsize=base-4, fontweight='bold', color='white')
    
    # 添加横向网格线
    ax.grid(True, axis='y', alpha=0.7, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 添加斜向箭头和标注
    baseline_bar = bars[0]  # baseline柱子
    middle_bar = bars[1]    # 中间柱子
    last_bar = bars[2]      # 最右边柱子
    
    # 第一个箭头：从baseline斜向指向中间柱子
    baseline_center_x = baseline_bar.get_x() + baseline_bar.get_width() + middle_bar.get_width()/8
    baseline_top_y = values[0]
    middle_center_x = middle_bar.get_x() #- middle_bar.get_width()/8
    middle_top_y = values[1]
    
    # 绘制第一个斜向箭头
    ax.annotate('', xy=(middle_center_x, middle_top_y), 
                xytext=(baseline_center_x, baseline_top_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5, mutation_scale=25))
    
    # 第一个箭头的标签 (-0.84)
    text_x_1 = (baseline_center_x + middle_center_x) / 2 + 0.25
    text_y_1 = (baseline_top_y + middle_top_y) / 2 + 0.1
    ax.text(text_x_1, text_y_1, '-0.54', 
            ha='center', va='center', fontsize=base, fontweight='bold', color='black')
    
    # 第二个箭头：从中间柱子斜向指向最右边柱子
    last_center_x = last_bar.get_x() - middle_bar.get_width()/8
    last_top_y = values[2]
    
    middle_center_x = middle_bar.get_x() + middle_bar.get_width()
    # 绘制第二个斜向箭头
    ax.annotate('', xy=(last_center_x, last_top_y), 
                xytext=(middle_center_x, middle_top_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5, mutation_scale=25))
    
    # 第二个箭头的标签 (+0.57)
    text_x_2 = (middle_center_x + last_center_x) / 2 - 0.36
    text_y_2 = (middle_top_y + last_top_y) / 2 - 0.05
    ax.text(text_x_2, text_y_2, '+0.41', 
            ha='center', va='center', fontsize=base, fontweight='bold', color='black')
    
    # 设置坐标轴样式 - 保留所有边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    
    # 设置刻度样式
    ax.tick_params(axis='y', which='major', labelsize=base-2)
    ax.tick_params(axis='x', which='major', labelsize=base, labelcolor='black')
    
    # 添加Y轴标签
    ax.set_ylabel('Quality* [PSNR]', fontsize=base, fontweight='bold')
    
    # 调整边距以确保Y轴标签显示
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.95)
    
    # 保存图片 - 不使用tight_layout或bbox_inches='tight'
    output_path = '/home/ytanaz/access/IBRNet/memory/statistic_plots/ema_comparison.png'
    plt.savefig(output_path, dpi=400, facecolor='white')
    print(f"D2D传输量对比图表已保存到: {output_path}")
    
    # 不显示图表，直接关闭
    plt.close()

if __name__ == "__main__":
    # 创建D2D传输量对比图表
    create_d2d_transfer_chart()
    print("D2D传输量对比图表创建完成！")
