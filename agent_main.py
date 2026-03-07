from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
import tool.tools as tools
import tool.data_analysis_tool as data_ana
import tool.image_classify as image_class
import uuid
class State(TypedDict):
    messages: Annotated[list, add_messages]

class Agent:
    def __init__(self):
        self.tool = [tools.new_file, tools.read_file, tools.write_file, tools.rename_file, tools.check_knowledge,
                     data_ana.describe_data, image_class.image_class]
        self.llm = init_chat_model(
            model="ollama:qwen3:8b",
            # base_url='http://loaclhost:11434',
            temperature=0.1,
            timeout=5,
        )
        self.bound_model = self.llm.bind_tools(self.tool)
        self.tool_node = ToolNode(self.tool)
        self.graph = StateGraph(State)
        self.graph.add_node("tools", self.tool_node)
        self.graph.add_node("agent", self.call_model)

        # 编写链接顺序
        self.graph.add_edge(START, "agent")
        self.graph.add_conditional_edges("agent", self.should_continue)
        self.graph.add_edge("tools", "agent")

        # 创建一个 MemorySaver 实例
        self.memory = MemorySaver()
        # 编译工作流，生成一个 LangChain Runnable，并传入 MemorySaver 实例
        self.app = self.graph.compile(checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.file_path = None
        with open('prompt.txt', 'r', encoding='utf8') as f:
            self.prompt = f.read()
        self.sys = self.prompt + f'用户上传文件路径为{self.file_path}，输出的时候不要输出文件路径'

    def call_model(self, state: State):
        response = self.bound_model.invoke(state["messages"])
        # 返回一个列表，因为这将被添加到现有列表中
        return {"messages": response}

    def should_continue(self, state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        # 如果最后一条消息包含工具调用，则返回 "tools"
        if last_message.tool_calls:
            return "tools"
        # 否则返回 "__end__" 以结束流程
        return "__end__"
    
    def start_new_conversation(self, thread_id: str = None):
        if thread_id:
            self.config["configurable"]["thread_id"] = thread_id
        else:
            self.config["configurable"]["thread_id"] = str(uuid.uuid4())
        return self.config
    
    def get_file(self, path:str):
        self.file_path = path
        self.sys = self.prompt + f"用户上传文件路径为{self.file_path}，输出的时候不要输出文件路径"

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
def main():
    agent = Agent()
    while True:
        s = input()
        for event in agent.app.stream({"messages": [HumanMessage(s), SystemMessage(agent.sys)]}, agent.config, stream_mode="values"):
            event["messages"][-1].pretty_print()
            if isinstance(event["messages"][-1], AIMessage):
                print(event["messages"][-1].content)
if __name__ == '__main__':
    main()
