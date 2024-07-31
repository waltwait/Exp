import subprocess
import concurrent.futures
import gc

# def get_gpu_memory_usage():
#     """获取GPU内存使用情况"""
#     gpus = GPUtil.getGPUs()
#     if gpus:
#         return gpus[0].memoryUsed * 1024 * 1024  # 将GB转换为字节
#     return 0

def run_command(command):
    try:
        # 使用 Popen 并设置 stdout 和 stderr 为 subprocess.PIPE 来即时捕获和显示输出
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            while True:
                output = proc.stdout.readline()
                if output == '' and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())
            stderr = proc.stderr.read()
            if stderr:
                print(stderr.strip())
        print("Script completed successfully")
    except subprocess.CalledProcessError as e:
        # 輸出錯誤信息
        print("Error occurred:", e)

def run_test_train(directory, model_destination,hid_dim,pool_ratio,architecture,conv_layers):
    try:
        command = [
            'python', 'testTrain_old.py',
            '--dir', directory,
            '--model_dst', model_destination,
            '--hid_dim',hid_dim,
            '--pool_ratio',pool_ratio,
            '--architecture',architecture,
            '--conv_layers',conv_layers,
        ]
        run_command(command)

    except Exception as e:
        print(f"Failed to run testTrain.py for {directory}: {e}")

try:
    #Exp2 shi model demo 5%
        
    run_test_train('../Data/DataSet/5%_node1', '../Data/Model_Data/5%_node1','128','0.5','hierarchical','2')
    run_test_train('../Data/DataSet/5%_node2', '../Data/Model_Data/5%_node2','128','0.5','hierarchical','2')
    run_test_train('../Data/DataSet/5%_node3', '../Data/Model_Data/5%_node3','128','0.5','hierarchical','2')
    run_test_train('../Data/DataSet/5%_node4', '../Data/Model_Data/5%_node4','128','0.5','hierarchical','2')

except Exception as e:
    print(f"An error occurred in the main execution block: {e}")


