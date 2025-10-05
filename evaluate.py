import os
import re
import glob

def extract_mip_results_from_files(log_folder, folder_name="文件夹"):
    """
    从log文件夹中提取所有instance log文件的Best solution和Best gap，并计算平均值
    
    Args:
        log_folder: log文件夹路径
        folder_name: 文件夹显示名称
    """
    solutions = []
    gaps = []
    
    # 查找所有log文件
    log_pattern = os.path.join(log_folder, "instance_*.lp.log")
    log_files = glob.glob(log_pattern)
    
    print(f"{folder_name}: 找到 {len(log_files)} 个log文件")
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"警告: {log_file} 不存在，跳过")
            continue
            
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取Best solution
            solution_match = re.search(r'Best solution\s*:\s*([\d.]+)', content)
            # 提取Best gap (百分比值)
            gap_match = re.search(r'Best gap\s*:\s*([\d.]+)%', content)
            
            if solution_match and gap_match:
                solution = float(solution_match.group(1))
                gap = float(gap_match.group(1))
                
                solutions.append(solution)
                gaps.append(gap)
                
                instance_name = os.path.basename(log_file)
                print(f"  {instance_name}: Solution = {solution:.2f}, Gap = {gap:.2f}%")
            else:
                print(f"警告: 在 {log_file} 中未找到完整数据")
                
        except Exception as e:
            print(f"错误: 处理 {log_file} 时出错: {e}")
    
    # 计算平均值
    if solutions and gaps:
        avg_solution = sum(solutions) / len(solutions)
        avg_gap = sum(gaps) / len(gaps)
        
        print("\n" + "="*50)
        print(f"{folder_name} 统计结果:")
        print(f"有效instance数量: {len(solutions)}")
        print(f"Best Solution 平均值: {avg_solution:.2f}")
        print(f"Best Gap 平均值: {avg_gap:.4f}%")
        print(f"Best Solution 范围: {min(solutions):.2f} - {max(solutions):.2f}")
        print(f"Best Gap 范围: {min(gaps):.4f}% - {max(gaps):.4f}%")
        print("="*50)
        
        return {
            'folder_name': folder_name,
            'avg_solution': avg_solution,
            'avg_gap': avg_gap,
            'solutions': solutions,
            'gaps': gaps,
            'count': len(solutions)
        }
    else:
        print(f"{folder_name}: 未找到任何有效数据")
        return None

def extract_mip_results_from_subfolders(log_folder, folder_name="文件夹"):
    """
    从log文件夹中提取所有instance子文件夹的Best solution和Best gap，并计算平均值
    
    Args:
        log_folder: log文件夹路径
        folder_name: 文件夹显示名称
    """
    solutions = []
    gaps = []
    
    # 查找所有instance文件夹
    instance_pattern = os.path.join(log_folder, "instance_*.lp")
    instance_folders = glob.glob(instance_pattern)
    
    print(f"{folder_name}: 找到 {len(instance_folders)} 个instance文件夹")
    
    for instance_folder in instance_folders:
        log_file = os.path.join(instance_folder, "3.log")
        
        if not os.path.exists(log_file):
            print(f"警告: {log_file} 不存在，跳过")
            continue
            
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取Best solution
            solution_match = re.search(r'Best solution\s*:\s*([\d.]+)', content)
            # 提取Best gap (百分比值)
            gap_match = re.search(r'Best gap\s*:\s*([\d.]+)%', content)
            
            if solution_match and gap_match:
                solution = float(solution_match.group(1))
                gap = float(gap_match.group(1))
                
                solutions.append(solution)
                gaps.append(gap)
                
                print(f"  {os.path.basename(instance_folder)}: Solution = {solution:.2f}, Gap = {gap:.2f}%")
            else:
                print(f"警告: 在 {log_file} 中未找到完整数据")
                
        except Exception as e:
            print(f"错误: 处理 {log_file} 时出错: {e}")
    
    # 计算平均值
    if solutions and gaps:
        avg_solution = sum(solutions) / len(solutions)
        avg_gap = sum(gaps) / len(gaps)
        
        print("\n" + "="*50)
        print(f"{folder_name} 统计结果:")
        print(f"有效instance数量: {len(solutions)}")
        print(f"Best Solution 平均值: {avg_solution:.2f}")
        print(f"Best Gap 平均值: {avg_gap:.4f}%")
        print(f"Best Solution 范围: {min(solutions):.2f} - {max(solutions):.2f}")
        print(f"Best Gap 范围: {min(gaps):.4f}% - {max(gaps):.4f}%")
        print("="*50)
        
        return {
            'folder_name': folder_name,
            'avg_solution': avg_solution,
            'avg_gap': avg_gap,
            'solutions': solutions,
            'gaps': gaps,
            'count': len(solutions)
        }
    else:
        print(f"{folder_name}: 未找到任何有效数据")
        return None

def compare_results(results1, results2):
    """
    比较两个文件夹的结果
    """
    if not results1 or not results2:
        print("无法比较，至少有一个文件夹没有有效数据")
        return
    
    print("\n" + "="*60)
    print("结果对比:")
    print("="*60)
    print(f"{'指标':<20} {results1['folder_name']:<15} {results2['folder_name']:<15} {'差异':<10}")
    print("-" * 60)
    
    # 比较Solution平均值
    solution_diff = results2['avg_solution'] - results1['avg_solution']
    solution_diff_percent = (solution_diff / results1['avg_solution']) * 100 if results1['avg_solution'] != 0 else 0
    print(f"{'Solution平均值':<20} {results1['avg_solution']:<15.2f} {results2['avg_solution']:<15.2f} {solution_diff:+.2f} ({solution_diff_percent:+.2f}%)")
    
    # 比较Gap平均值
    gap_diff = results2['avg_gap'] - results1['avg_gap']
    print(f"{'Gap平均值(%)':<20} {results1['avg_gap']:<15.4f} {results2['avg_gap']:<15.4f} {gap_diff:+.4f}%")
    
    # 比较实例数量
    count_diff = results2['count'] - results1['count']
    print(f"{'实例数量':<20} {results1['count']:<15} {results2['count']:<15} {count_diff:+d}")

def main():
    # 第一个文件夹结构（子文件夹）
    # folder1 = "logs/ca/ca_COPT_Predect&Search_2025-10-03 14:00:18.484240"
    folder1 = "logs/sc/sc_COPT_Predect&Search_2025-10-03 14:00:49.157188"
    
    # 第二个文件夹结构（直接文件）  
    # folder2 = "test_logs/ca"
    folder2 = "test_logs/sc"

    results1 = None
    results2 = None
    
    # 处理第一个文件夹（子文件夹结构）
    if os.path.exists(folder1):
        print("处理第一个文件夹 (子文件夹结构):")
        results1 = extract_mip_results_from_subfolders(folder1, "原始实验文件夹")
    else:
        print(f"第一个文件夹 {folder1} 不存在，跳过")
    
    print("\n\n")
    
    # 处理第二个文件夹（直接文件结构）
    if os.path.exists(folder2):
        print("处理第二个文件夹 (直接文件结构):")
        results2 = extract_mip_results_from_files(folder2, "测试实验文件夹")
    else:
        print(f"第二个文件夹 {folder2} 不存在，跳过")
    
    # 比较结果
    if results1 and results2:
        compare_results(results1, results2)

if __name__ == "__main__":
    main()