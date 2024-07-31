import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [
        # commnt: 全部AIP list，裁剪出行為子圖重要度為5%的子圖
        "python autoprune.py",
        "python autotrain.py",     
    ]
    
    for cmd in commands:
        run_command(cmd)
