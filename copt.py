import os.path
import pickle
from multiprocessing import Process, Queue
import numpy as np
import argparse
from helper import get_a_copt_fast

import coptpy as cp
from coptpy import COPT

def solve_copt(filepath, log_dir, settings):
    envconfig = cp.EnvrConfig()
    envconfig.set('nobanner', '1')

    env = cp.Envr()
    m = env.createModel()
    m.read(filepath)

    m.setParam(COPT.Param.TimeLimit, settings['maxtime'])
    m.setParam(COPT.Param.Threads, settings['threads'])

    log_path = os.path.join(log_dir, os.path.basename(filepath) + '.log')
    with open(log_path, 'w'):
        pass

    m.setLogFile(log_path)

    max_sols = settings.get('maxsol', 1)

    m.solve()

    sols = []
    objs = []

    n_sols = m.getAttr(COPT.Attr.PoolSols)

    mvars = m.getVars()
    var_names = [var.getName() for var in mvars]

    for sn in range(min(n_sols, max_sols)):
        sols.append(np.array(m.getPoolSolution(sn, mvars)))
        objs.append(m.getPoolObjVal(sn))

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)

    sol_data = {
        'var_names': var_names,
        'sols': sols,
        'objs': objs,
    }

    return sol_data

def collect(ins_dir,q,sol_dir,log_dir,bg_dir,settings):

    while True:
        filename = q.get()
        if not filename:
            break
        filepath = os.path.join(ins_dir,filename)        
        sol_data = solve_copt(filepath,log_dir,settings)
        #get bipartite graph , binary variables' indices
        A2,v_map2,v_nodes2,c_nodes2,b_vars2=get_a_copt_fast(filepath)
        BG_data=[A2,v_map2,v_nodes2,c_nodes2,b_vars2]
        
        # save data
        pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))
        pickle.dump(BG_data, open(os.path.join(bg_dir, filename+'.bg'), 'wb'))





if __name__ == '__main__':
    #sizes=['small','large']
    # sizes=["ca", "sc", "lb", "ip", "is"]
    # sizes = ["ca", "sc"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./data')
    parser.add_argument('--nWorkers', type=int, default=16)
    parser.add_argument('--maxTime', type=int, default=10)
    parser.add_argument('--maxStoredSol', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--prob', type=str, default='sc', choices=['sc', 'ca'])
    args = parser.parse_args()

    if args.prob == 'sc':
        sizes = ['sc']
    elif args.prob == 'ca':
        sizes = ['ca']
    else:
        raise NotImplementedError


    import multiprocessing
    multiprocessing.set_start_method('spawn')

    for size in sizes:
    

        dataDir = args.dataDir

        TRAIN_INS_DIR = os.path.join(dataDir,f'instances/{size}/train')
        VAL_INS_DIR = os.path.join(dataDir,f'instances/{size}/valid')

        print(f'check {TRAIN_INS_DIR} ...')
        assert os.path.isdir(TRAIN_INS_DIR)
        print(f'check {VAL_INS_DIR} ...')
        assert os.path.isdir(VAL_INS_DIR)


        if not os.path.isdir('./dataset'):
            os.mkdir('./dataset')
        if not os.path.isdir(f'./dataset/{size}'):
            os.mkdir(f'./dataset/{size}')

        if not os.path.isdir(f'./dataset/{size}/train'):
            os.mkdir(f'./dataset/{size}/train')
        if not os.path.isdir(f'./dataset/{size}/train/solution'):
            os.mkdir(f'./dataset/{size}/train/solution')
        if not os.path.isdir(f'./dataset/{size}/train/NBP'):
            os.mkdir(f'./dataset/{size}/train/NBP')
        if not os.path.isdir(f'./dataset/{size}/train/logs'):
            os.mkdir(f'./dataset/{size}/train/logs')
        if not os.path.isdir(f'./dataset/{size}/train/BG'):
            os.mkdir(f'./dataset/{size}/train/BG')

        if not os.path.isdir(f'./dataset/{size}/valid'):
            os.mkdir(f'./dataset/{size}/valid')
        if not os.path.isdir(f'./dataset/{size}/valid/solution'):
            os.mkdir(f'./dataset/{size}/valid/solution')
        if not os.path.isdir(f'./dataset/{size}/valid/NBP'):
            os.mkdir(f'./dataset/{size}/valid/NBP')
        if not os.path.isdir(f'./dataset/{size}/valid/logs'):
            os.mkdir(f'./dataset/{size}/valid/logs')
        if not os.path.isdir(f'./dataset/{size}/valid/BG'):
            os.mkdir(f'./dataset/{size}/valid/BG')

        TRAIN_SOL_DIR =f'./dataset/{size}/train/solution'
        TRAIN_LOG_DIR =f'./dataset/{size}/train/logs'
        TRAIN_BG_DIR =f'./dataset/{size}/train/BG'
        os.makedirs(TRAIN_SOL_DIR, exist_ok=True)
        os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

        os.makedirs(TRAIN_BG_DIR, exist_ok=True)

        VAL_SOL_DIR =f'./dataset/{size}/valid/solution'
        VAL_LOG_DIR =f'./dataset/{size}/valid/logs'
        VAL_BG_DIR =f'./dataset/{size}/valid/BG'
        os.makedirs(VAL_SOL_DIR, exist_ok=True)
        os.makedirs(VAL_LOG_DIR, exist_ok=True)

        os.makedirs(VAL_BG_DIR, exist_ok=True)


        N_WORKERS = args.nWorkers

        # copt settings
        SETTINGS = {
            'maxtime': args.maxTime,
            'maxsol': args.maxStoredSol,
            'threads': args.threads,

        }

        print(f'collecting train {size} ...')

        filenames_train = os.listdir(TRAIN_INS_DIR)
   
        q = Queue()
        # add ins
        for filename in filenames_train:
            if not os.path.exists(os.path.join(TRAIN_BG_DIR,filename+'.bg')):
                q.put(filename)

        total_files = q.qsize()
        print(f'Total training files to process: {total_files}')
        
        # add stop signal
        for i in range(N_WORKERS):
            q.put(None)

        # pbar = tqdm(total=total_files, desc=f'Processing {size} training files')

        ps = []
        for i in range(N_WORKERS):
            p = Process(target=collect,args=(TRAIN_INS_DIR,q,TRAIN_SOL_DIR,TRAIN_LOG_DIR,TRAIN_BG_DIR,SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        # pbar.close()

        print(f'collecting valid {size} ...')

        filenames_val = os.listdir(VAL_INS_DIR)

        q = Queue()
        # add ins
        for filename in filenames_val:
            if not os.path.exists(os.path.join(VAL_BG_DIR,filename+'.bg')):
                q.put(filename)

        total_files = q.qsize()
        print(f'Total validation files to process: {total_files}')

        # add stop signal
        for i in range(N_WORKERS):
            q.put(None)

        # pbar = tqdm(total=total_files, desc=f'Processing {size} validation files')

        ps = []
        for i in range(N_WORKERS):
            p = Process(target=collect,args=(VAL_INS_DIR,q,VAL_SOL_DIR,VAL_LOG_DIR,VAL_BG_DIR,SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
        
        # pbar.close()

        print(f'finish {size} !')
