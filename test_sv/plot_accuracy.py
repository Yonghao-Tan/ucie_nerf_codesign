import matplotlib.pyplot as plt
import numpy as np

def create_accuracy_chart():
    """
    创建PSNR准确率对比图表
    """
    # 数据
    categories = ['Baseline', '+CSU', '+CSU+50% SV Prune', '+CSU+DPU(48%)']
    values = [25.06, 25.05, 24.77, 24.96]  # PSNR值
    colors = ['#B0B0B0', '#909090', '#707070', '#505050']  # 渐变灰色
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 创建柱状图
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    
    # 设置坐标轴范围
    ax.set_ylim(24.25, 25.25)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_yticks([24.5, 25])
    
    # 在柱子内部添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        # 在柱子顶端添加文字
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='black')
    
    # 添加横向网格线
    ax.grid(True, axis='y', alpha=0.7, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 添加虚线和箭头标注
    baseline_bar = bars[0]  # Baseline作为参考
    baseline_value = values[0]
    
    # 计算虚线的位置和终点
    baseline_right_x = baseline_bar.get_x() + baseline_bar.get_width()
    line_end_x = bars[3].get_x() + bars[3].get_width()  # 延伸到最后一个柱子
    line_y = baseline_value  # 虚线的高度设置在baseline的高度
    
    # 绘制一根水平虚线 - 从baseline右边延展到最后一个柱子右端
    ax.plot([baseline_right_x, line_end_x], [line_y, line_y], 
            linestyle='--', color='black', linewidth=1.5, alpha=0.8)
    
    # 为第二和第三个柱子添加箭头和差值
    for i in range(1, len(values)):
        proposed_bar = bars[i]
        # 计算与baseline的差值
        difference = values[i] - baseline_value
        
        # 计算箭头位置
        proposed_center_x = proposed_bar.get_x() + proposed_bar.get_width()/2
        
        # 调整箭头起点和终点
        arrow_start_y = line_y
        arrow_end_y = values[i]
        
        # 根据差值正负决定箭头方向和颜色
        if difference < 0:
            # 下降箭头
            ax.annotate('', xy=(proposed_center_x, arrow_end_y), xytext=(proposed_center_x, arrow_start_y),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
            # 添加差值文字
            text_x = proposed_center_x + 0.15
            text_y = (arrow_start_y + arrow_end_y) / 2
            ax.text(text_x, text_y, f'{difference:.2f}', 
                    ha='left', va='center', fontsize=11, fontweight='bold', color='black')
        else:
            # 上升箭头
            ax.annotate('', xy=(proposed_center_x, arrow_end_y), xytext=(proposed_center_x, arrow_start_y),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
            # 添加差值文字
            text_x = proposed_center_x + 0.15
            text_y = (arrow_start_y + arrow_end_y) / 2
            ax.text(text_x, text_y, f'+{difference:.2f}', 
                    ha='left', va='center', fontsize=11, fontweight='bold', color='black')
    
    # 设置坐标轴样式 - 保留所有边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    
    # 设置刻度样式
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='x', which='major', labelsize=14, labelcolor='black')
    
    # 添加Y轴标签
    ax.set_ylabel('PSNR', fontsize=14, fontweight='bold')
    
    # 添加标题
    ax.set_title('PSNR Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = './accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PSNR对比图表已保存到: {output_path}")
    
    # 不显示图表，直接关闭
    plt.close()

if __name__ == "__main__":
    # 创建PSNR对比图表
    create_accuracy_chart()
    print("PSNR对比图表创建完成！")
