import matplotlib.pyplot as plt
import numpy as np

def plot_25mm_eye_diagrams():
    """
    绘制25mm的两个眼图，分别保存
    """
    channel_length = '25mm'
    test_groups = [1, 2]
    
    base = 20
    
    for group in test_groups:
        # 创建单个图形
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
        
        # 读取对应的数据文件
        filename = f'./data/{channel_length}_{group}.txt'
        data = np.loadtxt(filename)
        
        # 第一列是Vref
        vref = data[:, 0]
        
        # 后面32列是两组16 lane的数据
        group1_data = data[:, 1:17]   # 前16列
        group2_data = data[:, 17:33]  # 后16列
        
        # 生成颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, 16))
        
        # 左半边：第一组16个lane的眼图（映射到负CPI区域）
        for i in range(16):
            lane_data = group1_data[:, i]
            # 将数据映射到左半边（负值区域）
            lane_data_left = -np.abs(lane_data)
            ax.plot(lane_data_left, vref, color=colors[i], linewidth=2, alpha=0.8, 
                    marker='o', markersize=5, markerfacecolor=colors[i], markeredgecolor='none')
        
        # 右半边：第二组16个lane的眼图（映射到正CPI区域）
        for i in range(16):
            lane_data = group2_data[:, i]
            # 将数据映射到右半边（正值区域）
            lane_data_right = np.abs(lane_data)
            ax.plot(lane_data_right, vref, color=colors[i], linewidth=2, alpha=0.8,
                    marker='o', markersize=5, markerfacecolor=colors[i], markeredgecolor='none')
        
        # 添加中心参考线
        ax.axvline(x=0, color='orange', linewidth=3, alpha=0.8)
        ax.axhline(y=0, color='gray', linewidth=1, alpha=0.5)
        
        # 设置图形属性（无标题）
        if group == 1:
            # 第一个图显示全称
            ax.set_xlabel('CPI (Clock-path Phase Interpolator)', fontsize=base+2, fontweight='bold')
        else:
            # 第二个图只显示缩写
            ax.set_xlabel('CPI', fontsize=base+2, fontweight='bold')
        
        ax.set_ylabel('Vref', fontsize=base+6, fontweight='bold', labelpad=-8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-32, 32)
        ax.set_ylim(-25, 25)
        
        # 设置刻度
        ax.set_xticks(np.arange(-32, 33, 8))
        ax.set_yticks(np.arange(-24, 25, 8))
        
        # 设置刻度字号
        ax.tick_params(axis='x', labelsize=base)
        ax.tick_params(axis='y', labelsize=base)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存文件
        output_filename = f'./outputs/{channel_length}_{group-1}.png'
        plt.savefig(output_filename, dpi=400, bbox_inches='tight')
        print(f"眼图已保存到: {output_filename}")
        
        plt.close()  # 关闭当前图形

if __name__ == "__main__":
    # 绘制25mm眼图
    plot_25mm_eye_diagrams()
