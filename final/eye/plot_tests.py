import matplotlib.pyplot as plt
import numpy as np

def plot_eye_diagram():
    """
    读取8个txt文件创建4x2眼图
    """
    # 定义文件列表和对应的标题
    channel_lengths = ['5mm', '10mm', '25mm', '50mm']
    test_groups = [1, 2]
    
    # 创建4行2列的子图
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    
    base = 16
    
    for row, length in enumerate(channel_lengths):
        for col, group in enumerate(test_groups):
            ax = axes[row, col]
            
            # 读取对应的数据文件
            filename = f'./data/{length}_{group}.txt'
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
                ax.plot(lane_data_left, vref, color=colors[i], linewidth=1.5, alpha=0.8, 
                        marker='o', markersize=4, markerfacecolor=colors[i], markeredgecolor='none')
            
            # 右半边：第二组16个lane的眼图（映射到正CPI区域）
            for i in range(16):
                lane_data = group2_data[:, i]
                # 将数据映射到右半边（正值区域）
                lane_data_right = np.abs(lane_data)
                ax.plot(lane_data_right, vref, color=colors[i], linewidth=1.5, alpha=0.8,
                        marker='o', markersize=4, markerfacecolor=colors[i], markeredgecolor='none')
            
            # 添加中心参考线
            ax.axvline(x=0, color='orange', linewidth=2, alpha=0.8)
            ax.axhline(y=0, color='gray', linewidth=1, alpha=0.5)
            
            # 设置图形属性
            ax.set_xlabel('CPI', fontsize=base, fontweight='bold')
            ax.set_ylabel('Vref', fontsize=base, fontweight='bold', labelpad=-5)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-32, 32)
            ax.set_ylim(-25, 25)
            
            # 设置刻度
            ax.set_xticks(np.arange(-32, 33, 8))  # 减少刻度密度
            ax.set_yticks(np.arange(-24, 25, 8))
            
            # 设置刻度字号
            ax.tick_params(axis='x', labelsize=base-4)
            ax.tick_params(axis='y', labelsize=base-4)
            
            # 设置标题
            group_index = group - 1  # 转换为0和1
            title = f'Digital Eye [Channel Length = {length}] Test Group {group_index}'
            ax.set_title(title, fontsize=base, fontweight='bold', pad=10)
    
    # 调整子图间距
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.4, wspace=0.45)
    plt.savefig('./outputs/all_eye_diagrams.png', dpi=400, bbox_inches='tight')
    print("所有眼图已保存到: ./outputs/all_eye_diagrams.png")
    plt.show()

if __name__ == "__main__":
    # 绘制所有眼图
    plot_eye_diagram()
