import pandas as pd
import re


def extract_hour_values():
    try:
        df = pd.read_csv('chiller_settings_31days.csv')
        new_data = []
        for index, row in df.iterrows():
            if index >= 0 :
                value_str = row['daily_chiller_settings']
                if pd.notnull(value_str):
                    values = re.findall(r'hour\d+=\d+', value_str)
                    for value in values:
                        num = int(re.search(r'=\d+', value).group(0)[1:])
                        new_data.append(num)
        new_df = pd.DataFrame({'extracted_values': new_data})
        new_df.to_csv('rag-chiller.csv', index=False)
        print("数据提取并保存成功！")
    except FileNotFoundError:
        print("找不到文件，请检查文件名是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    extract_hour_values()
