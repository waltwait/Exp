import multiprocessing

def run_command(cmd):
    import subprocess
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/TPASSG/Malware --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 1",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/TPASSG/Benign --label malware --correlation_json ../sensitiveAPIList/5%.json --threshold 0 --Pnode 1",
        # SFCGDroid 敏感子圖
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/malware/ --processed_dst ../Data/DataSet/SFCGDroid/Malware --label malware --correlation_json ../sensitiveAPIList/SFCGDroid.json --threshold 0 --Pnode 2",
        "python pruning_graph_by_frequency.py --dir ../../../fcg/Exp_M_B/Test/benign/ --processed_dst ../Data/DataSet/SFCGDroid/Benign --label malware --correlation_json ../sensitiveAPIList/SFCGDroid.json --threshold 0 --Pnode 2",
        
        
]
    with multiprocessing.Pool(4) as pool:
        pool.map(run_command, commands)