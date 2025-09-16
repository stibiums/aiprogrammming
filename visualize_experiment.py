import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV数据
csv_path = '/home/stibiums/下载/cifar_experiment.csv'
steps = []
values = []

with open(csv_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        steps.append(int(row['Step']))
        values.append(float(row['Value']))

# 创建图表
plt.figure(figsize=(12, 8))
plt.plot(steps, values, 'b-', linewidth=2, marker='o', markersize=4)

# 设置图表标题和标签
plt.title('CIFAR Experiment Training Process', fontsize=16, fontweight='bold')
plt.xlabel('Training Step', fontsize=14)
plt.ylabel('Value', fontsize=14)

# 添加网格
plt.grid(True, alpha=0.3)

# 设置坐标轴
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 自动调整布局
plt.tight_layout()

# 保存图片
output_path = '/media/stibiums/document/PKU/25_autumn/aip/aiprogrammming/cifar_experiment_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# 显示图片
plt.show()

print(f"Plot saved to: {output_path}")