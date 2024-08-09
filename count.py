def count_numbers_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        numbers = content.split()  # 將文件內容按空格分割成數值列表
        return len(numbers)

# 使用範例
file_path = './outputs/UE4 and Airsim/20240529-180231/results/DepthValues/image_1.txt'  # 請將這裡替換成你的txt檔案的路徑
total_numbers = count_numbers_in_file(file_path)
print(f"總共有 {total_numbers} 個數值")
