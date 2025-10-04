import matplotlib.pyplot as plt
import re

def plot_loss_from_file(filename, outname=None):
    # 从文件读取数据
    with open(filename, 'r') as file:
        log_data = file.read()
    
    epochs = []
    train_losses = []
    valid_losses = []
    
    # 解析数据
    for line in log_data.strip().split('\n'):
        match = re.search(r'@epoch(\d+)\s+Train loss:([\d.]+)\s+Valid loss:([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            valid_loss = float(match.group(3))
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
    
    # 创建更清晰的图表
    plt.figure(figsize=(12, 7))
    
    # 使用不同的线型和更细的线条
    line1, = plt.plot(epochs, train_losses, color='blue', linestyle='-', 
                      linewidth=1.2, label='Train Loss', marker='o', markersize=3)
    line2, = plt.plot(epochs, valid_losses, color='red', linestyle='--', 
                      linewidth=1.2, label='Valid Loss', marker='s', markersize=3)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    
    # 改进的图例
    plt.legend(handles=[line1, line2], fontsize=11, 
               frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    
    # 更细致的网格
    plt.grid(True, alpha=0.15, linestyle='-')
    
    # 设置刻度
    plt.xticks(epochs, fontsize=10)
    plt.yticks(fontsize=10)
    
    # 美化边框
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    if outname:
        plt.savefig(outname)
    plt.show()
    
    # 打印统计信息
    print(f"训练完成 {len(epochs)} 个epoch")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {valid_losses[-1]:.4f}")
    print(f"训练损失下降: {train_losses[0]-train_losses[-1]:.2f} ({((train_losses[0]-train_losses[-1])/train_losses[0]*100):.1f}%)")
    
    return epochs, train_losses, valid_losses

# 使用示例（如果你的日志保存在文件中）
plot_loss_from_file("train_logs/ca_train/ca_train_train.log", "ca_train_loss_plot.png")
plot_loss_from_file("train_logs/sc_train/sc_train_train.log", "sc_train_loss_plot.png")
