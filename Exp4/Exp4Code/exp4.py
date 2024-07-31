import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [
        # DGCNDroid 和 SFCG敏感子圖 
        "python data_preprocess_M_B.py --dir ../../../Exp/fcg/Exp_M_B/Train/malware/ --processed_dst ../Data/Dataset/DGCNDroid/Malware --label malware",
        "python data_preprocess_M_B.py --dir ../../../Exp/fcg/Exp_M_B/Train/benign/ --processed_dst ../Data/Dataset/DGCNDroid/Benign --label benign",

        "python autoprune.py",
        "python autotrain.py",     
    ]
    
    for cmd in commands:
        run_command(cmd)
