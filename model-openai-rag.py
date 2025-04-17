import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.hub import pull
from tqdm import tqdm

# 设置环境变量（保持原有设置）
os.environ["OPENAI_BASE_URL"] = "https://api.zhizengzeng.com/v1/"
os.environ["OPENAI_API_KEY"] = "sk-zk2b6dbc9cc8056ab439cdb600ae7be5b0266d8391559c91"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# 定义本地保存路径
persist_directory = "embedding"  # 指定保存文件夹路径

# 加载数据
with open('output_sentences2.txt', 'r', encoding='utf-8') as file:
    combined_text = file.read()

# 切分文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
print("Splitting text into chunks...")
splits = text_splitter.split_text(combined_text)

# 检查是否存在已保存的向量存储
if os.path.exists(persist_directory):
    # 如果保存的嵌入文件夹存在，直接加载
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
else:
    # 如果保存文件夹不存在，则创建嵌入并保存
    print("Embedding text...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=[text for text in tqdm(splits, desc="Embedding")],
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()  # 将嵌入保存到本地文件夹

# 创建一个检索器
retriever = vectorstore.as_retriever()
print("Retriever created.")

# Prompt
prompt = pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def read_and_process_data(file_path):
    """读取并处理CSV文件中的数据"""
    # 读取CSV文件中的特定列（保持原有逻辑）
    df1 = pd.read_csv(file_path, usecols=['t_out'])
    df2 = pd.read_csv(file_path, usecols=['energy_total'])
    df3 = pd.read_csv(file_path, usecols=['Equip'])
    df4 = pd.read_csv(file_path, usecols=['light'])
    df5 = pd.read_csv(file_path, usecols=['occ'])
    df6 = pd.read_csv(file_path, usecols=['temperature:drybulb'])

    # 确保数据是数值类型（保持原有逻辑）
    df1['t_out'] = pd.to_numeric(df1['t_out'], errors='coerce')
    df2['energy_total'] = pd.to_numeric(df2['energy_total'], errors='coerce')
    df3['Equip'] = pd.to_numeric(df3['Equip'], errors='coerce')
    df4['light'] = pd.to_numeric(df4['light'], errors='coerce')
    df5['occ'] = pd.to_numeric(df5['occ'], errors='coerce')
    df6['temperature:drybulb'] = pd.to_numeric(df6['temperature:drybulb'], errors='coerce')

    return df1, df2, df3, df4, df5, df6

def calculate_hourly_averages(df):
    """计算每小时的平均值（保持原有逻辑）"""
    var = []
    for i in range(0, len(df), 6):
        if i + 5 < len(df):
            var.append(df.iloc[i:i + 6].mean().values[0])
    return var

def calculate_changes(var):
    """计算每小时的变化量（保持原有逻辑）"""
    changes = []
    for i in range(0, len(var) - 1):
        changes.append(var[i + 1] - var[i])
    return changes

def group_data(var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, group_size=24):
    """将数据按指定大小分组（保持原有逻辑）"""
    grouped_data = {}
    for i in range(0, len(var_1), group_size):
        group_index = i // group_size
        grouped_data[f'var_1_{group_index + 1}'] = var_1[i:i + group_size]
        grouped_data[f'var_2_{group_index + 1}'] = var_2[i:i + group_size]
        grouped_data[f'var_3_{group_index + 1}'] = var_3[i:i + group_size]
        grouped_data[f'var_4_{group_index + 1}'] = var_4[i:i + group_size]
        grouped_data[f'var_5_{group_index + 1}'] = var_5[i:i + group_size]
        grouped_data[f'var_6_{group_index + 1}'] = var_6[i:i + group_size]
        grouped_data[f'var_7_{group_index + 1}'] = var_7[i:i + group_size]
        grouped_data[f'var_8_{group_index + 1}'] = var_8[i:i + group_size]
    return grouped_data

def process_grouped_data(vars_dict):
    """处理分组后的数据并调用模型"""
    all_days_results = []  # 存储所有天的结果

    for i in range(0, 31):
        try:
            # 生成问题
            question = f"""As a building operator in USA, I am responsible for monitoring and adjusting the indoor and outdoor environmental conditions, including the energy usage of HVAC systems. Given that it's a weekday in August with high cooling loads, I need to set optimal indoor chiller temperatures to save the energy savings as much as possible.

            Building Specifications:
            Total area: 46,320 square meters
            Levels: 12 above-ground floors, 1 underground floor (1 basement level and 12 upper levels)
            10-fold multiplier applied to internal load on the 6th floor
            HVAC system coverage: 46,320 square meters
            Occupancy rates: 37.16 square meters per person (underground level) and 18.58 square meters per person (above-ground levels)
            Lighting density: 10.76 watts per square meter
            Equipment density: 10.76 watts per square meter

            Current Conditions:
            Outdoor air temperature: {vars_dict.get(f'var_1_{i + 1}', None)}°C, from 0:00 to 24:00 each hour
            Temperature change: {vars_dict.get(f'var_3_{i + 1}', None)}, from 0:00 to 24:00 each hour
            Equipment situation: {vars_dict.get(f'var_5_{i + 1}', None)}, from 0:00 to 24:00 each hour
            Building's lighting: {vars_dict.get(f'var_6_{i + 1}', None)}, from 0:00 to 24:00 each hour
            Occupancy ratio:  {vars_dict.get(f'var_7_{i + 1}', None)}, from 0:00 to 24:00 each hour
            HVAC air condition temperature setting: {vars_dict.get(f'var_8_{i + 1}', None)}, from 0:00 to 24:00 each hour

            When the current 24-hour water-cooled chiller temperature setting is 12°C, the energy usage of electricity for this building has {vars_dict.get(f'var_2_{i + 1}', None)} [J] from 0:00 to 24:00 each hour, and the change of the energy usage of electricity is {vars_dict.get(f'var_4_{i + 1}', None)} from 0:00 to 24:00 each hour. 

            Goal: To reduce overall electricity consumption as much as possible.

            Please select the optimal water-cooled chiller temperature setting from 6°C,7°C,8°C,9°C,10°C,11°C,12°C for each 24 hour (0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00, 17:00, 18:00, 19:00, 20:00, 21:00, 22:00 23:00) to achieve this goal.

            **Output Format**: Please output EXACTLY 24 lines in the following format, with no extra text:
                hour0=VALUE
                hour1=VALUE
                ...
                hour23=VALUE
            Do not output any additional lines, characters, or explanations.
            """

            # 调用 RAG 链
            respond = rag_chain.invoke(question)
            text = respond.replace("°C", "")  # 统一去除°C
            lines = text.split('\n')[:25]  # 只保留前25行
            day_result_str = '\n'.join(lines)  # 拼接成字符串
            all_days_results.append(day_result_str)  # 存入列表

            # 打印当天结果
            print(f"day {i + 1} 结果：")
            print(day_result_str)
            print("————————————————————————————————————————————————————————————————")

        except Exception as e:
            print(f"模型调用失败: {e}")

    # 保存到CSV，一列存储31天结果
    df = pd.DataFrame({"daily_chiller_settings": all_days_results})
    df.to_csv("chiller_settings_31days.csv", index=False, sep='\t')

    return all_days_results

if __name__ == "__main__":
    file_path = 'result-12.csv'
    df1, df2, df3, df4, df5, df6 = read_and_process_data(file_path)

    # 计算每小时的平均温度和耗电量（保持原有逻辑）
    var_1 = calculate_hourly_averages(df1)
    var_2 = calculate_hourly_averages(df2)
    var_5 = calculate_hourly_averages(df3)
    var_6 = calculate_hourly_averages(df4)
    var_7 = calculate_hourly_averages(df5)
    var_8 = calculate_hourly_averages(df6)

    # 计算每小时的温度和耗电量变化（保持原有逻辑）
    var_3 = calculate_changes(var_1)
    var_4 = calculate_changes(var_2)

    # 每24个数据为一组（保持原有逻辑）
    grouped_data = group_data(var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8)

    # 处理分组数据并保存结果
    results = process_grouped_data(grouped_data)