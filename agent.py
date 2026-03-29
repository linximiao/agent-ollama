import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from typing import List, TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import tool.rag as rag
import tool.image_classify as image_class
from middleware.middlewares import adynamic_model_select, dynamic_prompt, ahandle_tool_errors
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import uuid

POSTGRES_CONN_STRING = "postgresql://postgres:12369874@localhost:5432/checkpoint_db?sslmode=disable"
MCP_CONFIG = {
    "search": {
        "transport": "http",
        "url": "http://localhost:3000/mcp"
    }
}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str 

class Agent:
    def __init__(self):
        self.tool = [rag.check_knowledge, image_class.image_class]
        self.middleware = [adynamic_model_select, ahandle_tool_errors]
        # self.checkpoint = MemorySaver()
        # self.checkpoint = PostgresSaver.from_conn_string(POSTGRES_CONN_STRING)
        self.checkpoint = None
        self.checkpoint_context = None
        with open('prompt.txt', 'r', encoding='utf8') as f:
            self.prompt = f.read()
        self.file_path = None
        self.sys = self.prompt + f'\n用户上传文件路径为{self.file_path}，输出的时候不要输出文件路径'
        self.llm = init_chat_model(
            model="ollama:qwen3:8b",
            temperature=0.7,
            timeout=10
        )
        self.summarizer_llm = init_chat_model(
            model="ollama:qwen3:8b", 
            temperature=0,
            timeout=10
        )
        self.agent = None
        self.thread_id = None

    async def initialize(self, mcp_session):
        self.checkpoint_context = AsyncPostgresSaver.from_conn_string(POSTGRES_CONN_STRING)
        self.checkpoint = await self.checkpoint_context.__aenter__()
        await self.checkpoint.setup()
        mcptools = await load_mcp_tools(mcp_session)
        all_tools = self.tool + mcptools
        self.agent = create_agent(
            model=self.llm,
            tools=all_tools,
            middleware=self.middleware,
            system_prompt=self.prompt,
            checkpointer=self.checkpoint
        )

    @classmethod
    async def creat(cls):
        cls = cls()
        mcp_client = MultiServerMCPClient(MCP_CONFIG)
        mcp_context = mcp_client.session("search")
        mcp_session = await mcp_context.__aenter__()
        await cls.initialize(mcp_session)
        return cls, mcp_context

    def invoke(self, user_input: str, file_path: str = None):
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())
            
        config = {"configurable": {"thread_id": self.thread_id}}
        res = self.agent.invoke({"messages": user_input, "file_path": file_path}, config)
        return res
    
    async def ainvoke(self, user_input: str, file_path: str = None):
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())
            
        config = {"configurable": {"thread_id": self.thread_id}}
        res = await self.agent.ainvoke({"messages": user_input, "file_path": file_path}, config)
        return res
    
    def start_new_conversation(self, thread_id: str = None):
        if thread_id:
            self.thread_id = thread_id
        else:
            self.thread_id = str(uuid.uuid4())
    
    def get_file(self, path:str):
        self.file_path = path
        self.sys = self.prompt + f"用户上传文件路径为{self.file_path}，输出的时候不要输出文件路径"

    def _count_valid_turns(self, messages: list) -> int:
        """
        统计有效对话轮数（HumanMessage + 非空 AIMessage）
        """
        turns = 0
        for msg in messages:
            if isinstance(msg, HumanMessage):
                turns += 1
            elif isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                turns += 1
        return turns

    async def _compress_history(self, messages: list, max_length: int = 400) -> str:
        """
        使用 summarizer_llm 压缩对话历史
        
        Args:
            messages: 原始消息列表
            max_length: 摘要最大字数
            
        Returns:
            压缩后的摘要字符串
        """
        conversation = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                role = "用户" if isinstance(msg, HumanMessage) else "助手"
                content = msg.content.strip() if msg.content else ""
                if content:  # 跳过空消息
                    conversation.append(f"{role}: {content}")
        
        if not conversation:
            return ""
        
        # 2. 构建压缩提示词
        history_text = "\n".join(conversation[-10:])  # 最多取最近 10 条，避免超长
        prompt = f"""请帮我将以下对话历史压缩成一段简洁的摘要，要求：
1. 保留关键事实、用户意图和重要结论
2. 去除重复、寒暄和无关细节
3. 控制在 {max_length} 字以内
4. 以第三人称客观描述

对话历史：
{'-' * 40}
{history_text}
{'-' * 40}
"""
        
        response = await self.summarizer_llm.ainvoke(prompt)
        summary = response.content.strip()
        return summary

    def _apply_compression(self, messages: list, summary: str) -> list:
        """
        用摘要替换旧历史，保留最新一轮对话
        
        Args:
            messages: 原始消息列表
            summary: 压缩后的摘要
            
        Returns:
            新的消息列表：[SystemMessage(摘要), ...最新对话]
        """
        last_human_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        
        # 2. 保留摘要 + 最新一轮对话（用户输入 + 助手回复）
        new_messages = [
            SystemMessage(content=f"【对话历史摘要】{summary}"),
        ]
        
        # 保留最新一轮完整对话
        if last_human_idx >= 0:
            new_messages.append(messages[last_human_idx])  # 用户最新输入
            if last_human_idx + 1 < len(messages) and isinstance(messages[last_human_idx + 1], AIMessage):
                new_messages.append(messages[last_human_idx + 1])
        
        return new_messages


async def amain():
    agent, mcp_context = await Agent.creat()
    res = await agent.ainvoke("什么是党参")
    print(res)
    res = await agent.ainvoke("我刚才说了什么？")
    print(res)
    await mcp_context.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(amain())