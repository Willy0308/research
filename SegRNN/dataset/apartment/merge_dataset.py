import pandas as pd
import os

# 假設所有客戶的CSV檔案都存放在這個目錄下
directory_path = '/home/willy/research/aparment/dataset/apartment/'

# 檔案名稱列表
file_names = os.listdir(directory_path)

# 初始化一個空字典來存儲所有客戶的數據
all_data = {}

# 遍歷每個檔案，加載並轉換數據
for file_name in file_names:
    if file_name.endswith('.csv'):
        # 獲取客戶名稱作為列名
        customer_name = file_name.replace('.csv', '')
        
        # 讀取 CSV 檔案
        df = pd.read_csv(os.path.join(directory_path, file_name))
        
        # 將第一列轉換為 datetime 格式，並設置為索引
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        
        # 將數據轉換為每小時的平均值
        hourly_data = df.resample('H').mean()
        
        # 將處理後的數據存儲到字典中
        all_data[customer_name] = hourly_data.iloc[:, 0]

# 使用 pd.DataFrame.from_dict 一次性創建 DataFrame
combined_df = pd.DataFrame.from_dict(all_data)

# 將合併後的 DataFrame 導出為 CSV
output_path = '/home/willy/research/aparment/dataset/apartment/combined_electricity_2016.csv'
combined_df.to_csv(output_path)

print(f'合併後的數據已保存到 {output_path}')
