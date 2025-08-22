import matplotlib.pyplot as plt
import numpy as np

def create_memory_comparison_chart():
    """
    创建内存使用对比图表，包含4个柱子
    """
    # 数据
    categories = ['Baseline', 'Direct Mapping', 'CGU', 'BAU']
    values = [42.72, 12, 2.5, 0.4]  # GB
    colors = ['#A0A0A0', '#707070', '#505050', '#303030']  # 不同深度的灰色
    
    # 计算减少百分比（相对于baseline）
    reduction_percentage = ((values[0] - values[-1]) / values[0]) * 100
    
    # 创建图表 - 减小整体宽度使柱子间距更紧凑
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建柱状图 - 减小width使柱子更窄
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
    
    # 设置坐标轴范围 - 调整xlim使柱子间距更紧凑
    ax.set_ylim(0, 50)
    ax.set_xlim(-0.4, len(categories)-0.5)  # 进一步缩小范围
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    
    # 在柱子内部添加数值标签（白色文字）
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        # 在柱子中间位置添加文字
        if height > 5:  # 只有足够高的柱子才在内部显示文字
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{value}', ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')
        else:  # 太矮的柱子在顶部显示文字
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black')
    
    # 添加横向网格线
    ax.grid(True, axis='y', alpha=0.7, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 添加减少百分比的箭头和标注
    baseline_bar = bars[0]  # Baseline柱子
    baseline_value = values[0]
    bau_bar = bars[3]  # BAU柱子
    
    # 计算虚线的位置和终点
    baseline_right_x = baseline_bar.get_x() + baseline_bar.get_width()
    line_end_x = bau_bar.get_x() + bau_bar.get_width()
    line_y = baseline_value - 0.05  # 虚线的高度与baseline顶部平齐
    
    # 绘制一根水平虚线 - 从baseline右边延展到BAU柱子最右端
    ax.plot([baseline_right_x, line_end_x], [line_y, line_y], 
            linestyle='--', color='black', linewidth=1.5, alpha=0.8)
    
    # 为每个proposed方法（Direct Mapping, CGU, BAU）添加箭头和百分比
    for i in range(1, len(values)):
        proposed_bar = bars[i]
        reduction_percentage = ((baseline_value - values[i]) / baseline_value) * 100
        
        # 计算箭头位置
        proposed_center_x = proposed_bar.get_x() + proposed_bar.get_width()/2
        
        # 绘制垂直箭头从虚线到proposed柱子顶端
        ax.annotate('', xy=(proposed_center_x, values[i]), xytext=(proposed_center_x, line_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # 添加百分比文字 - 放在每个箭头旁边
        text_x = proposed_center_x + 0.15
        text_y = (line_y + values[i]) / 2  # 箭头中间位置
        ax.text(text_x, text_y, f'{reduction_percentage:.1f}%', 
                ha='left', va='center', fontsize=10, fontweight='bold', color='black')
    
    # 设置坐标轴样式 - 保留所有边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    
    # 设置刻度样式
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='x', which='major', labelsize=16, labelcolor='black')
    
    # 添加Y轴标签
    ax.set_ylabel('EMA (GB)', fontsize=14, fontweight='bold')
    
    # 添加标题
    ax.set_title('Memory Usage Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/home/ytanaz/access/IBRNet/memory/statistic_plots/memory_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图表已保存到: {output_path}")
    
    # 不显示图表，直接关闭
    plt.close()

if __name__ == "__main__":
    # 只创建一个图表
    create_memory_comparison_chart()
    print("图表创建完成！")
