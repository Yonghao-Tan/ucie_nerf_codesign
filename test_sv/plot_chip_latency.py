import matplotlib.pyplot as plt
import numpy as np

def create_sr_latency_chart():
    """
    创建D2D传输总量对比图表
    """
    # 数据
    # Baseline: 0.3435s; Solution 2a: 0.0436s
    categories = ['Baseline', '+CFSRE']
    a, b = 0.3435, 0.0436
    a, b = a / a, b / a
    values = [a*100, b*100]
    colors = ['#B0B0B0', '#505050']  # 渐变灰色
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建柱状图
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    
    # 设置坐标轴范围
    # ax.set_ylim(0, 20)
    ax.set_xlim(-0.5, len(categories) - 0.35)  # 适当的右边空间
    # ax.set_yticks([0, 5, 10, 15, 20])
    
    # 在柱子内部添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        # 在柱子中间位置添加文字
        if height > 3:  # 只有足够高的柱子才在内部显示文字
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{value:.2f}', ha='center', va='center', 
                    fontsize=26, fontweight='bold', color='white')
        else:  # 太矮的柱子在顶部显示文字
            ax.text(bar.get_x() + bar.get_width()/2. + 0.25, height + 0.5,
                    f'{value:.2f}', ha='center', va='bottom', 
                    fontsize=26, fontweight='bold', color='black')
    
    # 添加横向网格线
    ax.grid(True, axis='y', alpha=0.7, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 添加减少百分比的箭头和标注
    baseline_bar = bars[0]  # Mode 0 Only作为baseline
    baseline_value = values[0]
    dual_model_bar = bars[1]  # Dual-Model柱子
    
    # 计算虚线的位置和终点
    baseline_right_x = baseline_bar.get_x() + baseline_bar.get_width()
    line_end_x = dual_model_bar.get_x() + dual_model_bar.get_width()
    line_y = baseline_value - 0.025  # 虚线的高度
    
    # 绘制一根水平虚线 - 从baseline右边延展到Dual-Model柱子最右端
    ax.plot([baseline_right_x, line_end_x], [line_y, line_y], 
            linestyle='--', color='black', linewidth=1.5, alpha=0.8)
    
    # 为Mode 1 Only和Dual-Model添加箭头和百分比
    for i in range(1, len(values)):
        proposed_bar = bars[i]
        reduction_percentage = ((baseline_value - values[i]) / baseline_value) * 100
        # reduction_percentage = baseline_value / values[i]
        
        # 计算箭头位置
        proposed_center_x = proposed_bar.get_x() + proposed_bar.get_width()/2
        
        # 绘制垂直箭头从虚线到proposed柱子顶端
        ax.annotate('', xy=(proposed_center_x, values[i]), xytext=(proposed_center_x, line_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.5, mutation_scale=25))
        
        # 添加百分比文字 - 放在每个箭头旁边
        text_x = proposed_center_x + 0.025
        text_y = (line_y + values[i]) / 2  # 箭头中间位置
        ax.text(text_x, text_y, f'{reduction_percentage:.2f}%', 
                ha='left', va='center', fontsize=26, fontweight='bold', color='black')
    
    # 设置坐标轴样式 - 保留所有边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    
    # 设置刻度样式
    ax.tick_params(axis='y', which='major', labelsize=24)
    ax.tick_params(axis='x', which='major', labelsize=26, labelcolor='black')
    
    # 添加Y轴标签
    ax.set_ylabel('Normalized Latency', fontsize=26, fontweight='bold')
    
    # 添加标题
    ax.set_title('Latency Comparison', fontsize=26, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = './cfsre.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"D2D传输量对比图表已保存到: {output_path}")
    
    # 不显示图表，直接关闭
    plt.close()

if __name__ == "__main__":
    # 创建D2D传输量对比图表
    create_sr_latency_chart()
    print("D2D传输量对比图表创建完成！")
