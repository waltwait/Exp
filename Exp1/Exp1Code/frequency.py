import argparse
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def count_total_samples(label_file):
    all_label_data = load_data(label_file)
    total_samples = len(all_label_data)
    return total_samples

# Load JSON data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def get_api_importance(api_data):
    malware_count = sum(value for key, value in api_data.items() if key.startswith('malware'))
    benign_count = sum(value for key, value in api_data.items() if key.startswith('benign'))
    total_count = malware_count + benign_count

    malware_frequency = malware_count / total_count if total_count != 0 else 0
    benign_frequency = benign_count / total_count if total_count != 0 else 0

    frequency_diff = malware_frequency - benign_frequency
    return frequency_diff
# def get_api_importance(all_api_data):
#     total_malware_degree = 0.0
#     total_benign_degree = 0.0

#     # Calculate total degrees for all APIs
#     for api, values in all_api_data.items():
#         for key, degree in values.items():
#             if "malware" in key:
#                 total_malware_degree += float(degree)
#             elif "benign" in key:
#                 total_benign_degree += float(degree)

#     # Calculate sensitivity score for each API
#     api_sensitivity_scores = {}
#     for api, values in all_api_data.items():
#         api_malware_degree = sum(float(degree) for key, degree in values.items() if "malware" in key)
#         api_benign_degree = sum(float(degree) for key, degree in values.items() if "benign" in key)

#         malware_ratio = api_malware_degree / total_malware_degree if total_malware_degree > 0 else 0
#         benign_ratio = api_benign_degree / total_benign_degree if total_benign_degree > 0 else 0

#         sensitivity_score = malware_ratio - benign_ratio
#         api_sensitivity_scores[api] = sensitivity_score

#     return api_sensitivity_scores



def calculate_correlation(data_file, output_file, min_sample_count=10):
    # Load JSON data from files
    all_degree_data = load_data(data_file)
    correlation_results = {}
    # sensitivity_scores = get_api_importance(all_degree_data)

    # for api, samples in all_degree_data.items():
    #     total_count = sum(samples.values())
    #     if total_count < min_sample_count:
    #         print(f"API {api} 樣本數 {total_count} 少於下限 {min_sample_count}, 跳過計算。")
    #         continue
    #     correlation_results[api] = {'importance': sensitivity_scores[api]}
    for api, samples in all_degree_data.items():
        total_count = sum(samples.values())
        if total_count < min_sample_count:
            print(f"API {api} 樣本數 {total_count} 少於下限 {min_sample_count}, 跳過計算。")
            continue
        importance = get_api_importance(samples)
        # sensitivity_scores = get_api_importance({api: samples})
        correlation_results[api] = {'importance': importance}

    # 将结果存储为JSON文件
    with open(output_file, 'w') as file:
        json.dump(correlation_results, file, indent=4)
    print(f"结果已保存至 {output_file}")

# def process_api(api, samples, min_sample_count):
#     total_count = sum(samples.values())
#     if total_count < min_sample_count:
#         print(f"API {api} 樣本數 {total_count} 少於下限 {min_sample_count}, 跳過計算。")
#         return None
#     importance = get_api_importance(samples)
#     return (api, {'importance': importance})

# def calculate_correlation(data_file, output_file, min_sample_count=10):
#     all_degree_data = load_data(data_file)
#     correlation_results = {}
#     futures = []

#     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#         for api, samples in all_degree_data.items():
#             futures.append(executor.submit(process_api, api, samples, min_sample_count))

#         for future in as_completed(futures):
#             result = future.result()
#             if result:
#                 api, importance = result
#                 correlation_results[api] = importance

#     with open(output_file, 'w') as file:
#         json.dump(correlation_results, file, indent=4)
#     print(f"結果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process explained graphs and calculate API scores based on all nodes.")
    parser.add_argument("--json_degree", required=True)
    # parser.add_argument("--json_label", required=True)
    parser.add_argument("--dst", required=True)
    # parser.add_argument("--percent", type=float, default=0.1, help="Minimum sample percentage for API calculation (0 to 1)")
    args = parser.parse_args()
    # total_samples = count_total_samples(args.json_label)
    # Percent_float = args.percent / 100.0
    min_sample_count = 0
    calculate_correlation(args.json_degree, args.dst, min_sample_count=min_sample_count)