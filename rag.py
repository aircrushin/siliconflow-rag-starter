import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from silicon_flow_llm import SiliconFlowLLM

# 加载.env文件中的环境变量
load_dotenv()

# 加载文档
loader = TextLoader("base.txt")
documents = loader.load()

# 使用 RecursiveCharacterTextSplitter 并指定 "###" 为分隔符
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],  # 使用换行符作为分隔符
    chunk_size=100,  # 减小 chunk_size
    chunk_overlap=0,  # 减少或移除重叠
    keep_separator=False  # 不保留分隔符
)
texts = text_splitter.split_documents(documents)

# 打印切片结果
print(f"总共分割出 {len(texts)} 个文本块")
for i, text in enumerate(texts):
    print(f"\n--- 文本块 {i+1} ---")
    print(f"长度: {len(text.page_content)} 字符")
    print(f"内容: {text.page_content[:100]}...")  # 只打印前100个字符

# 创建嵌入和向量存储
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma.from_documents(texts, embeddings)

# 创建检索器，设置 k 参数
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 创建SiliconFlow语言模型
llm = SiliconFlowLLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="Qwen/Qwen2-7B-Instruct"  # 使用SiliconFlow支持的模型
)

# 创建RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 使用RAG链回答问题
try:
    query = "什么是自然语言处理"
    result = qa_chain.invoke({"query": query})
    print(result["result"])
except Exception as e:
    print(f"发生错误：{e}")
