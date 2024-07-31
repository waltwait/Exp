import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [
        ## 敏感子圖
        # DGCNDroid 敏感子圖 
        "python data_preprocess_M_B.py --dir ../../../Exp/fcg/Exp_M_B/Train/malware/ --processed_dst ../Data/Dataset/DGCNDroid/Malware --label malware",
        "python data_preprocess_M_B.py --dir ../../../Exp/fcg/Exp_M_B/Train/benign/ --processed_dst ../Data/Dataset/DGCNDroid/Benign --label benign",
        "python autoprune.py",
        ## 節點遷入
        # comment : 跑 DGCNDroid 的 node embedding
        "python nodeembedding_DGCNDroid.py --dir ../Data/Dataset/DGCNDroid/Benign --processed_dst ../Data/Processed/DGCNDroid/Benign"
        "python nodeembedding_DGCNDroid.py --dir ../Data/Dataset/DGCNDroid/Malware --processed_dst ../Data/Processed/DGCNDroid/Malware"

        # comment: training a word2vec model      
        "python train_word2vec.py --dir ../Data/DataSet/TPASSG --model_dst ../Data/Model_Data/word2vec/TPASSG.pt",
        "python train_word2vec.py --dir ../Data/DataSet/SFCGDroid --model_dst ../Data/Model_Data/word2vec/SFCGDroid.pt",
        
        # comment: 把剛剛訓練的模型根據圖結構的 API name 進行 embedding
        "python word2vec.py --dir ../Data/DataSet/TPASSG/Malware --model_dst ../Data/Model_Data/word2vec/TPASSG.pt --processed_dst ../Data/Processed/TPASSG/Malware",
        "python word2vec.py --dir ../Data/DataSet/TPASSG/Benign --model_dst ../Data/Model_Data/word2vec/TPASSG.pt --processed_dst ../Data/Processed/TPASSG/Benign",
        "python word2vec.py --dir ./Data/DataSet/SFCGDroid/Malware --model_dst ../Data/Model_Data/word2vec/SFCGDroid.pt --processed_dst ../Data/Processed/SFCGDroid/Malware",
        "python word2vec.py --dir ./Data/DataSet/SFCGDroid/Benign --model_dst ../Data/Model_Data/word2vec/SFCGDroid.pt --processed_dst ../Data/Processed/SFCGDroid/Benign",


        # 各檢測系統訓練
        "python autotrain.py",

    ]
    
    for cmd in commands:
        run_command(cmd)

# future work
# 1. 直接把子圖丟給 GNN 找子圖當中最惡意的部分
# 2. 將最惡意部分對應回 APK 反編譯出來的程式碼
# 3. word2vec可以使用 bert