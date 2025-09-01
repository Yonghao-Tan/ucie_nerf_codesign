import matplotlib.pyplot as plt
import numpy as np

def create_d2d_transfer_chart():
    """
    创建D2D传输总量对比图表
    """
    # 数据
    # categories = ['Baseline', '+Unified\nCache', '+PAU']
    categories = ['Baseline', '+UC', '+PAU']
    # 计算总和值
    # a, b, c = 14057+9556, 1248+1147, 200+376
    # a, b, c = 14057+9556, 3990+3039, 200+376
    a, b, c = 14057+9556, 3990+3039, 453+544
    a, b, c = a/a, b/a, c/a
    values = [a*100, b*100, c*100]
    # values = [16.47, 8.01, 2.52]  # 四舍五入后的值
    colors = ['#A0A0A0', '#707070', '#505050']  # 渐变灰色
    base = 30
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 创建柱状图
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    
    # 设置坐标轴范围
    # ax.set_ylim(0, 20)
    ax.set_xlim(-0.5, len(categories) - 0.25)  # 适当的右边空间
    # ax.set_yticks([0, 5, 10, 15, 20])
    
    # 在柱子内部添加数值标签
    
    # 在柱子内部添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        if i > 0:
            height = bar.get_height()
            # 在柱子中间位置添加文字
            if height > 8:  # 只有足够高的柱子才在内部显示文字
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{value:.2f}', ha='center', va='center', 
                        fontsize=base, fontweight='bold', color='white')
            else:  # 太矮的柱子在顶部显示文字
                ax.text(bar.get_x() + bar.get_width()/2. + 0.25, height + 0.5,
                        f'{value:.2f}', ha='center', va='bottom', 
                        fontsize=base, fontweight='bold', color='black')
    
    # 添加横向网格线
    ax.grid(True, axis='y', alpha=0.7, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 添加减少百分比的箭头和标注
    baseline_bar = bars[0]  # Mode 0 Only作为baseline
    baseline_value = values[0]
    dual_model_bar = bars[2]  # Dual-Model柱子
    
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
                    arrowprops=dict(arrowstyle='->', color='black', lw=5.5, mutation_scale=45))
        
        # 添加百分比文字 - 放在每个箭头旁边
        text_x = proposed_center_x + 0.025
        text_y = (line_y + values[i]) / 2  # 箭头中间位置
        ax.text(text_x, text_y, f'{reduction_percentage:.2f}%', 
                ha='left', va='center', fontsize=base, fontweight='bold', color='black')
    
    # 设置坐标轴样式 - 保留所有边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    
    # 设置刻度样式
    ax.tick_params(axis='y', which='major', labelsize=base-2)
    ax.tick_params(axis='x', which='major', labelsize=base, labelcolor='black')
    
    # 添加Y轴标签
    ax.set_ylabel('Normalized EMA* (%)', fontsize=base, fontweight='bold')
    
    # 添加标题
    # ax.set_title('EMA Comparison', fontsize=base, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/home/ytanaz/access/IBRNet/memory/statistic_plots/ema_comparison.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"D2D传输量对比图表已保存到: {output_path}")
    
    # 不显示图表，直接关闭
    plt.close()

if __name__ == "__main__":
    # 创建D2D传输量对比图表
    create_d2d_transfer_chart()
    print("D2D传输量对比图表创建完成！")
