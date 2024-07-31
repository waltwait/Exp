import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [
        # comment: 官方API list，找裁減最大行為子圖的最佳參數(1%~8%)
        "python autoprune.py",
        # comment: 訓練shi model
        "python autotrain.py",     
    ]
    
    for cmd in commands:
        run_command(cmd)
