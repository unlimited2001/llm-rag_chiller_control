import os

# 这里是你的环境变量设置
os.environ["OPENAI_BASE_URL"] = "https://api.zhizengzeng.com/v1/"
os.environ["OPENAI_API_KEY"] = "sk-zk2b6dbc9cc8056ab439cdb600ae7be5b0266d8391559c91"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain import hub


# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 定义本地保存路径（持久化目录）
persist_directory = "embedding"  # 指定保存文件夹路径

# 读取 txt 文件
output_file_path = "output_sentences2.txt"
with open(output_file_path, "r", encoding="utf-8") as f:
    combined_text = f.read()

# 将读取到的字符串包装为 Document
document = Document(page_content=combined_text)

# 按照既定 chunk_size 与 chunk_overlap 切分文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
print("Splitting text into chunks...")
splits = text_splitter.split_documents([document])

# 生成向量索引
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 第 3 步：增大检索数量（k=5，避免遗漏）
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 从 hub 拉取原先的 prompt
prompt = hub.pull("rlm/rag-prompt")

# 用于把检索到的文档拼接成上下文字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构造 RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#------------------------------
# 第 1 步：手动查看检索结果
#------------------------------
original_query = "Please show me the summary for 2018-08-01 between 0:00 and 0:59."

# 先不走 rag_chain，直接用 retriever 看能检索到什么文档
docs = retriever.get_relevant_documents(original_query)
print("Retrieved docs for original_query:")
for i, doc in enumerate(docs):
    print(f"--- Doc #{i+1} ---")
    print(doc.page_content)
    print("---------------")

#------------------------------
# 第 2 步：修改提问方式
#------------------------------
# 更直接的问法，贴近你 txt 文件里的时间和表述：
modified_query = "Summarize data for 2018-08-01 from 00:00 to 01:00 in the Large office."

print("\nRetrieved docs for modified_query:")
docs_modified = retriever.get_relevant_documents(modified_query)
for i, doc in enumerate(docs_modified):
    print(f"--- Doc #{i+1} ---")
    print(doc.page_content)
    print("---------------")

#------------------------------
# 第 4 步：给模型更多上下文提示
#------------------------------
# 这里可以直接在“问题”里加上类似“如果找不到就说 I am not sure”之类的指令
final_query = (
    "Based on the above retrieved documents, please provide a concise summary "
    "of the building data for 2018-08-01 from 00:00 to 01:00 in the Large office. "
    "If the context does not provide an answer, say 'I am not sure.'"
)

print("\n--- Now using rag_chain.invoke with final_query ---")
result = rag_chain.invoke(final_query)
print("模型回答：", result)
