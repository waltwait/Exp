import multiprocessing

def run_command(cmd):
    import subprocess
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [ 
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/5%_node1/Malware --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/5%_node1/Benign --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/android/Malware --label malware --correlation_json ../sensitiveAPIList/android.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/android/Benign --label malware --correlation_json ../sensitiveAPIList/android.json --threshold 0 --Pnode 1",
        
        
]
    with multiprocessing.Pool(4) as pool:
        pool.map(run_command, commands)