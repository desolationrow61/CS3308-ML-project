import os
import re
import abc_py
import numpy as np
import pickle
import torch
import random

# Define the RESYN2_CMD
RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"

synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

# 定义从state生成AIG特征和评估得分的函数
def process_state(state):
    cnt = 0
    circuitName, actions = state.split('_')
    circuitDir = f'./circuit/{circuitName}'
    logDir = f'./log/{circuitName}'
    featureDir = f'./feature/{circuitName}'

    # 创建文件夹，如果不存在的话
    if not os.path.exists(circuitDir):
        os.makedirs(circuitDir)
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    if not os.path.exists(featureDir):
        os.makedirs(featureDir)
    
    initState = f'./InitialAIG/train/{circuitName}.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile_reg = f'{logDir}/{circuitName}_reg.log'

    def evaluate(lastState, actionCmd, logFile, nextState):
        abcRunCmd = f"./yosys/yosys-abc -c \"read {lastState} ; {actionCmd} read_lib {libFile} ; write {nextState} ; map ; topo ; stime\" > {logFile}"
        os.system(abcRunCmd)

        if not os.path.exists(logFile):
            raise FileNotFoundError(f"Log file '{logFile}' not found after evaluation command.")
        
        '''with open(logFile) as f:
            lines = f.readlines()
            area_information = re.findall('[a-zA-Z0-9.]+', lines[-1])
            if len(area_information) < 9:
                raise ValueError("Parsed area information does not contain enough elements.")
            
            eval_score = float(area_information[-9]) * float(area_information[-4])
        
        return eval_score'''
    
    def extract_aig_features(aig_path):
        _abc = abc_py.AbcInterface()
        _abc.start()
        _abc.read(aig_path)
        data = {}

        numNodes = _abc.numNodes()
        if numNodes == 0:
            raise ValueError("AIG network is empty.")
        
        data['node_type'] = np.zeros(numNodes, dtype=int)
        data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
        edge_src_index = []
        edge_target_index = []

        for nodeIdx in range(numNodes):
            aigNode = _abc.aigNode(nodeIdx)
            nodeType = aigNode.nodeType()
            data['num_inverted_predecessors'][nodeIdx] = 0
            if nodeType == 0 or nodeType == 2:
                data['node_type'][nodeIdx] = 0
            elif nodeType == 1:
                data['node_type'][nodeIdx] = 1
            else:
                data['node_type'][nodeIdx] = 2
                if nodeType == 4:
                    data['num_inverted_predecessors'][nodeIdx] = 1
                if nodeType == 5:
                    data['num_inverted_predecessors'][nodeIdx] = 2
            if (aigNode.hasFanin0()):
                fanin = aigNode.fanin0()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)
            if (aigNode.hasFanin1()):
                fanin = aigNode.fanin1()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)

        data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
        data['node_type'] = torch.tensor(data['node_type'])
        data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
        data['nodes'] = numNodes
        
        return data
    
    # get baseline
    if not os.path.exists(logFile_reg):
        # baseline = evaluate(initState, RESYN2_CMD, logFile_reg, f'{circuitDir}/{circuitName}_reg.aig')
        evaluate(initState, RESYN2_CMD, logFile_reg, f'{circuitDir}/{circuitName}_reg.aig')
    '''else:
        with open(logFile_reg) as f:
            area_information = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
            baseline = float(area_information[-9]) * float(area_information[-4])
    print(f"{circuitName} baseline: {baseline}")'''

    # evaluate and extract features for initial state
    lastStateName = circuitName + '_'
    logFile_cur = f'{logDir}/{lastStateName}.log'
    lastState = initState
    nextState = f'{circuitDir}/{lastStateName}.aig'
    if not os.path.exists(nextState):
        evaluate(lastState, '', logFile_cur, nextState)
    '''if os.path.exists(nextState):
        print(f"{lastStateName} already exists.")
    else:
        eval_score = evaluate(lastState, '', logFile_cur, nextState)
        eval_score_reg = 1 - eval_score / baseline
        print(f"{lastStateName} evaluation score: {eval_score} after reg: {eval_score_reg}")'''
    lastState = nextState

    aig_features = extract_aig_features(lastState)
    with open(f'{featureDir}/{lastStateName}.pkl', 'wb') as f:
        pickle.dump(aig_features, f)

    for action in actions:
        actionCmd = synthesisOpToPosDic[int(action)] + ';'
        logFile_cur = f'{logDir}/{lastStateName+action}.log'
        nextState = f'{circuitDir}/{lastStateName+action}.aig'
        if not os.path.exists(nextState):
            evaluate(lastState, actionCmd, logFile_cur, nextState)
            cnt += 1
        '''if os.path.exists(nextState):
            print(f"{lastStateName+action} already exists.")
        else:
            eval_score = evaluate(lastState, actionCmd, logFile_cur, nextState)
            eval_score_reg = 1 - eval_score / baseline
            print(f"{lastStateName+action} evaluation score: {eval_score} after reg: {eval_score_reg}")'''
        lastStateName += action
        lastState = nextState
    
        aig_features = extract_aig_features(nextState)
        with open(f'{featureDir}/{lastStateName}.pkl', 'wb') as f:
            pickle.dump(aig_features, f)

    return cnt

filesDir = './project_data'
# circuits = ['frg1', 'ctrl', 'int2float', 'alu2', 'm3', 'c1355', 'max512', 'c2670', 'priority', 'i7']
circuits = ['frg1', 'int2float', 'm3', 'max512', 'priority']
all_files = [f for f in os.listdir(filesDir) if f.endswith('.pkl')]

# 分组文件
grouped_files = {}
for f in all_files:
    circuitName = f.split('_')[0]
    if circuitName not in circuits:
        continue
    if circuitName not in grouped_files:
        grouped_files[circuitName] = []
    grouped_files[circuitName].append(f)
    
# 抽样文件
sampled_files = {}
for circuitName, files in grouped_files.items():
    if len(files) > 200:
        sampled_files[circuitName] = random.sample(files,200)
    else:
        sampled_files[circuitName] = files
        
for circuitName, files in sampled_files.items():
    count = 0
    for filename in files:
        with open(os.path.join(filesDir, filename), 'rb') as f:
            state_data = pickle.load(f)
            input_list = state_data['input']
            target_list = state_data['target']
            input_str = input_list[-1]
            target = target_list[-1]
            try:
                tmp = process_state(input_str)
                count += tmp
            except Exception as e:
                print(f"Error processing state {input_str}: {e}")
                
                
                
                
                

        if count > 400:
            break
