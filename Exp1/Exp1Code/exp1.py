import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    commands = [
        # comment: pre-processing malware and benign dataset
        "python data_preprocess_M_B.py --dir ../../../張華哲實驗原始碼與資料集/fcg/Exp_M_B/Train/malware/ --processed_dst ../Data/DataSet/Malware --label malware",
        "python data_preprocess_M_B.py --dir ../../../張華哲實驗原始碼與資料集/fcg/Exp_M_B/Train/benign/ --processed_dst ../Data/DataSet/Benign --label benign",

        # comment: training a word2vec model      
        "python train_word2vec.py --dir ../Data/DataSet --model_dst ../Data/Model_Data/word2vec/model.pt",
        
        # comment: 把剛剛訓練的模型根據圖結構的 API name 進行 embedding
        "python word2vec.py --dir ../Data/DataSet/Malware --model_dst ../Data/Model_Data/word2vec/model.pt --processed_dst ../Data/Processed/SAA/Malware",
        "python word2vec.py --dir ../Data/DataSet/Benign --model_dst ../Data/Model_Data/word2vec/model.pt --processed_dst ../Data/Processed/SAA/Benign",

        # comment: training a GNN model
        "python main.py --dir ../Data/Processed/SAA --model_dst ../Data/Model_Data/SAA",

        # comment: 進 XAI 解釋性生成行為子圖 (gnn explainer)
        # comment: 因為是根據良性和惡性的樣本去生成，所以要分開處理
        "python gnnexplainer.py --dir ../Data/Processed/SAA/Malware --model_path ../Data/Model_Data/SAA/best_model.pth --size 4 --Xai_dst ../Data/Xai/Malware",
        "python gnnexplainer.py --dir ../Data/Processed/SAA/Benign --model_path ../Data/Model_Data/SAA/best_model.pth --size 4 --Xai_dst ../Data/Xai/Benign",

        # comment: 計算 API degree（每個 API 在樣本內的重要度）
        "python genApidegree.py --dir ../Data/Processed/SAA/Benign ../Data/Processed/SAA/Malware --Xdir ../Data/Xai/Malware ../Data/Xai/Benign --api_dst ../Data/APIdegree/5%.json --percent 5",

        # comment: 計算 API 敏感度（用重要度去計算 API 比較偏向 malware/benign）
        # comment: 計算結果匯出於 SAA Model 當中的所有 .json 檔案當中，>0 表示偏向 malware，<0 表示偏向 benign
        "python frequency.py --json_degree ../Data/APIdegree/5%.json --dst ../Data/APIdegree/5%_frequency.json",
    ]
    
    for cmd in commands:
        run_command(cmd)

# future work
# 1. 直接把子圖丟給 GNN 找子圖當中最惡意的部分
# 2. 將最惡意部分對應回 APK 反編譯出來的程式碼
# 3. word2vec可以使用 bert GraphCodeBERT