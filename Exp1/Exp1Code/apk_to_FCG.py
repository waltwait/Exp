# ## use command window to run this scriptfamily
# import os
# import argparse

# def decompiler(instruction, apk):
#     res = os.system(instruction)
#     if res != 0:
#         print("fail to extract " + apk)
#     else:
#         print(apk, ' success!!!')

# parser = argparse.ArgumentParser()
# parser.add_argument("--source")
# parser.add_argument("--dst")
# # parser.add_argument('--types', nargs='+') # train val test
# parser.add_argument("--remove", choices=['Y', 'N'])  # remove apk after transfer to FCG
# parser.add_argument("-obf_method")
# args = parser.parse_args()

# if args.obf_method:
#     obf_method = args.obf_method
# else:
#     obf_method = ""

# for family in os.listdir(args.source):
#     apk_file = args.source + family + '/' + obf_method  + '/'
#     result_file = args.dst + family + '/' + obf_method + '/'
#     if not os.path.exists(args.source + family):
#         os.mkdir(args.source + family)
#     if not os.path.exists(args.source + family + '/' + obf_method):
#         os.mkdir(args.source + family + '/' + obf_method)
#     if not os.path.exists(args.source + family + '/' + obf_method + '/'):
#         os.mkdir(args.source + family + '/' + obf_method  + '/')
#     if not os.path.exists(args.dst + family):
#         os.mkdir(args.dst + family)
#     if not os.path.exists(args.dst + family + '/' + obf_method):
#         os.mkdir(args.dst + family + '/' + obf_method)
#     if not os.path.exists(args.dst + family + '/' + obf_method + '/'):
#         os.mkdir(args.dst + family + '/' + obf_method  + '/')
#     for apk in os.listdir(apk_file):
#         if apk=='obfuscation_working_dir':
#             print("obfuscation_working_dir")
#             continue
#         result_name = apk.replace('apk', 'gexf')

#         if os.path.exists(result_file + result_name):
#             print(result_file + result_name + " has already exists")
#             os.remove(apk_file + apk)
#             print("Delete ", apk_file + apk)
#             continue
#         # res = os.system('androguard cg ' + family_file + family+'/' + var + '/' + binascii.unhexlify(apk) + ' -o ' + fcg_file + family + '/' + var + '/' + result_name)
#         print(family)

#         ## APK to FCG
#         res = os.system('androguard cg ' + apk_file + apk + ' -o ' + result_file + result_name)
#         if res != 0:
#             print("fail to extract " + apk)
#         else:
#             print(apk, ' success!!!')

#         ## delete APK
#         if args.remove=='Y' and os.path.exists(apk_file + apk):
#             os.remove(apk_file + apk)
#             print("Delete ", apk_file + apk)

import os
import argparse

def run_decompiler(command):
    try:
        res = os.system(command)
        if res != 0:
            print("Failed to extract")
            return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False
    return True

parser = argparse.ArgumentParser()
parser.add_argument("--source")
parser.add_argument("--dst")
parser.add_argument("--remove", choices=['Y', 'N'])  # remove apk after transfer to FCG
parser.add_argument("-obf_method")
args = parser.parse_args()

if args.obf_method:
    obf_method = args.obf_method
else:
    obf_method = ""

# 確認目錄存在
if not os.path.exists(args.source):
    print(f"Source directory {args.source} does not exist.")
    exit(1)

if not os.path.exists(args.dst):
    os.makedirs(args.dst)

for apk in os.listdir(args.source):
    apk_path = os.path.join(args.source, apk)
    if not apk.endswith('.apk'):  # 確保處理的是 APK 檔案
        continue

    result_name = apk.replace('.apk', '.gexf')
    result_file_path = os.path.join(args.dst, result_name)

    if os.path.exists(result_file_path):
        print(result_file_path + " already exists")
        if args.remove == 'Y':
            os.remove(apk_path)
            print("Deleted ", apk_path)
        continue

    # 執行 APK 到 FCG 的轉換
    command = f'androguard cg "{apk_path}" -o "{result_file_path}"'
    print(f"Processing: {apk_path}")

    if not run_decompiler(command) and args.remove == 'Y':
            print(f"Skipping deletion due to error in processing {apk}")
            continue
    if args.remove == 'Y':
        os.remove(apk_path)
        print(f"Deleted {apk_path}")

