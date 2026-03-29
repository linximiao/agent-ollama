import os
from langchain_core.tools import tool
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ============ 配置区域 ============
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "rag_db",
    "user": "postgres",
    "password": "12369874"
}
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
KNOWLEDGE_DIR = "./knowledge"
COLLECTION_NAME = "knowledge_unified"
# ==================================

def get_connection_string():
    """构建 PostgreSQL 连接字符串"""
    return (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

def build_knowledge_index(overwrite: bool = False):
    """
    【索引构建函数】预加载所有知识库文件到向量数据库
    
    Args:
        overwrite: 是否强制重建索引（清空原集合）
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=get_connection_string(),
    )
    
    if overwrite:
        print(f"🗑️ 正在清空集合 {COLLECTION_NAME} ...")
        vector_store.delete_collection()
        print("✅ 集合已清空")
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=get_connection_string(),
        )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=200,
        chunk_overlap=20,
    )
    
    indexed = 0
    for filename in os.listdir(KNOWLEDGE_DIR):
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf8') as f:
                content = f.read()
            if not content.strip():
                print(f"⚪ 跳过空文件: {filename}")
                continue
            splits = text_splitter.split_text(content)
            if not splits:
                continue
            vector_store.add_texts(
                splits,
                metadatas=[{"source": filename} for _ in splits]
            )
            print(f"✓ 已索引: {filename} ({len(splits)} chunks)")
            indexed += 1
        except Exception as e:
            print(f"❌ 索引失败 {filename}: {e}")
            continue
    print(f"\n🎉 索引完成！共处理 {indexed} 个文件")
    return True

@tool
def check_knowledge(s: str) -> str:
    '''
    【检索函数】在已索引的知识库中查找关于 s 的知识（Top-3）
    '''
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=get_connection_string(),
    )
    docs = vector_store.similarity_search_with_score(s, k=3)
    SCORE_THRESHOLD = 0.3
    results = []
    for i, (d, score) in enumerate(docs, 1):
        if score > SCORE_THRESHOLD:
            continue
        source = d.metadata.get("source", "unknown")
        results.append(f"【片段 {i}】(来源: {source}, 相似度: {score:.2f})\n{d.page_content}")
    return "\n\n".join(results) if results else "未找到相关信息。"

@tool
async def acheck_knowledge(s: str) -> str:
    '''
    【检索函数】在已索引的知识库中查找关于 s 的知识（Top-3）
    '''
    return await check_knowledge(s)

if __name__=='__main__':
    build_knowledge_index(True)
    print(check_knowledge('党参'))

# with open('./knowledge/枸杞.txt', 'r', encoding='utf8') as f:
#     content = f.read()
# text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n"],
#         chunk_size=200,
#         chunk_overlap=20
# )
# split = text_splitter.split_text(content)
# print(len(split))
# print(split)