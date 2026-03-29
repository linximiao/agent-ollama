import os
from langchain_core.tools import tool

@tool
def new_file(path:str) -> None:
    '''
    新建path文件
    '''
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            pass

@tool
def read_file(path:str) -> str:
    '''
    读取path文件并返回内容
    '''
    with open(path, 'r', encoding='utf-8') as f:
        res = f.read()
    return res

@tool
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

@tool
def rename_file(path:str, new_name:str) -> None:
    '''
    将path文件重命名为new_name，new_name是新路径，不是单单一个文件名
    '''
    os.rename(path, new_name)
