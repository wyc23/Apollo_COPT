import torch
import time
import os

# from helper import get_a_new2, get_a_gp_fast
from copt import solve_copt

# def compare_functions(ins_name, device="cpu", rtol=1e-4, atol=1e-6):
#     results = {}

#     # 运行 get_a_new2
#     start = time.time()
#     A1, v_map1, v_nodes1, c_nodes1, b_vars1 = get_a_new2(ins_name)
#     t1 = time.time() - start

#     # 运行 get_a_gp_fast
#     start = time.time()
#     A2, v_map2, v_nodes2, c_nodes2, b_vars2 = get_a_gp_fast(ins_name, device=device)
#     t2 = time.time() - start

#     # 对比结果
#     results["time_get_a_new2"] = t1
#     results["time_get_a_gp_fast"] = t2

#     # 稀疏矩阵 A 对比
#     results["A_equal"] = torch.allclose(A1.to_dense(), A2.to_dense(), rtol=rtol, atol=atol)

#     # 变量特征对比
#     results["v_nodes_equal"] = torch.allclose(v_nodes1, v_nodes2, rtol=rtol, atol=atol)

#     # 约束特征对比
#     results["c_nodes_equal"] = torch.allclose(c_nodes1, c_nodes2, rtol=rtol, atol=atol)

#     # 二进制变量集合对比
#     results["b_vars_equal"] = torch.equal(b_vars1, b_vars2)

#     # v_map 的键集合对比
#     results["v_map_keys_equal"] = set(v_map1.keys()) == set(v_map2.keys())

#     return results


def main():
    # ins_name = "./instances/cauctions/test_100_500/instance_1.lp"
    # res = compare_functions(ins_name, device="cpu")
    # print(res)

    TaskName = "sc"
    TestNum = 100

    sample_names = sorted(os.listdir(f'./data/instances/{TaskName}/test'))
    log_dir = f"./test_logs/{TaskName}"
    settings = {
        'maxtime': 1000,
        'maxsol': 20,
        'threads': 1
    }
    for ins_num in range(min(len(sample_names),TestNum)):
        sample_name = sample_names[ins_num]
        filepath = os.path.join(f'./data/instances/{TaskName}/test', sample_name)
        print(f"Solving instance {ins_num+1}/{min(len(sample_names),TestNum)}: {sample_name}")
        solve_copt(filepath, log_dir, settings)



if __name__ == "__main__":
    main()
