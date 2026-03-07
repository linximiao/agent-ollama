import os
def new_file(path:str) -> None:
    '''
    新建path文件
    '''
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            pass

def read_file(path:str) -> str:
    '''
    读取path文件并返回内容
    '''
    with open(path, 'r', encoding='utf-8') as f:
        res = f.read()
    return res

def write_file(path:str, way:str, s:str) -> str:
    '''
    将s写入文件path
    path: 文件路径
    way: 写入方式 'w':写模式。如果文件已存在，则清空文件并写入新内容; 'a':追加模式。如果文件已存在，则将新内容添加到文件末尾。
    s: 要写入的内容
    '''
    try:
        with open(path, way, encoding='utf-8') as f:
            f.write(s)
    except Exception as e:
        return e
    else:
        return 'success'

def rename_file(path:str, new_name:str) -> None:
    '''
    将path文件重命名为new_name，new_name是新路径，不是单单一个文件名
    '''
    os.rename(path, new_name)

def check_knowledge(s:str) -> str:
    '''
    根据输入s在知识库中寻找有用的知识
    '''
    with open('./knowledge/knowledge_article.txt', 'r', encoding='utf8') as f:
        know = f.read()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],
        length_function = len,
        chunk_size = 200,
        chunk_overlap = 20)
    all_splits = text_splitter.split_text(know)

    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model='qwen3-embedding:0.6b')

    from langchain_core.vectorstores import InMemoryVectorStore
    vector_store = InMemoryVectorStore(embeddings).from_texts(
        texts=all_splits,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever()
    retrieved_documents = retriever.invoke(s)
    return retrieved_documents[0].page_content