import multiprocessing

def run_command(cmd):
    import subprocess
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [ 
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/1%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/1%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/2%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/2%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/3%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/3%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/4%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/4%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/5%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/5%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/6%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/6%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/7%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/7%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/8%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/8%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 1",

        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/1%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/1%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/2%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/2%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/3%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/3%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/4%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/4%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/5%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/5%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/6%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/6%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/7%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/7%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/8%_node2/Malware --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/8%_node2/Benign --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 2",

        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/1%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/1%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/2%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/2%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/3%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/3%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/4%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/4%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/5%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/5%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/6%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/6%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/7%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/7%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/8%_node3/Malware --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 3",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/8%_node3/Benign --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 3",

        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/1%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/1%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/1%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/2%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/2%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/2%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/3%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/3%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/3%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/4%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/4%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/4%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/5%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/5%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/6%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/6%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/6%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/7%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/7%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/7%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/8%_node4/Malware --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 4",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/8%_node4/Benign --label malware --correlation_json ../sensitiveAPIList/8%.json --threshold 0 --Pnode 4",
        
        
]
    with multiprocessing.Pool(4) as pool:
        pool.map(run_command, commands)