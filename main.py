"""
小米MiLM 智能文档RAG项目
用于小米大模型Token申请演示
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 小米MiLM 官方接口
MILM_KEY = os.getenv("MILM_KEY")
BASE_URL = "https://api.milim.lm.cn/v1"

# 初始化小米大模型
llm = ChatOpenAI(
    api_key=MILM_KEY,
    base_url=BASE_URL,
    model="milm-large",
    temperature=0.2
)
embeddings = OpenAIEmbeddings(
    api_key=MILM_KEY,
    base_url=BASE_URL
)

# 加载PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# 文本分割
def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

# 构建向量库
def create_vector(docs):
    return FAISS.from_documents(docs, embeddings)

# 创建问答+摘要双Agent
def init_agent(vector_store):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    sum_prompt = PromptTemplate(
        input_variables=["text"],
        template="帮我精简总结这段文档内容：\n{text}"
    )
    sum_chain = LLMChain(llm=llm, prompt=sum_prompt)
    return qa_chain, sum_chain

# 主程序
if __name__ == "__main__":
    print("===== 小米MiLM 文档智能问答系统 =====")
    pdf_path = input("请输入PDF文件路径：")

    docs = load_pdf(pdf_path)
    split_docs = split_text(docs)
    vec_store = create_vector(split_docs)

    qa, summary = init_agent(vec_store)

    print("\n📄 文档自动摘要：")
    print(summary.run(split_docs[0].page_content))

    chat_history = []
    print("\n💬 开始问答，输入 exit 退出")
    while True:
        question = input("\n你的问题：")
        if question.lower() == "exit":
            break
        res = qa.invoke({
            "question": question,
            "chat_history": chat_history
        })
        print("🤖 回答：", res["answer"])
        chat_history.append((question, res["answer"]))