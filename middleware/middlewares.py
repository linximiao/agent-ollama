from langchain.agents.middleware import wrap_model_call, dynamic_prompt, wrap_tool_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from typing import Callable

base_model = init_chat_model(
    model="ollama:qwen3:8b",
    temperture=0.7,
    timeout=5
)

advance_model = init_chat_model(
    model="ollama:qwen3.5:9b",
    temperture=0.7,
    timeout=5
)

@wrap_model_call
def dynamic_model_select(request:ModelRequest, handler:Callable) -> ModelResponse:
    n = len(request.state['messages'])
    if n>=3:
        model = advance_model
    else:
        model = base_model
    return handler(request.override(model = model))

@wrap_model_call
async def adynamic_model_select(request:ModelRequest, handler:Callable) -> ModelResponse:
    n = len(request.state['messages'])
    if n>=3:
        model = advance_model
    else:
        model = base_model
    return await handler(request.override(model = model))

@dynamic_prompt
def dynamic_prompt(request:ModelRequest) -> str:
    pass

@wrap_tool_call
def handle_tool_errors(request:ModelRequest, handler:Callable) -> ModelResponse:
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            tool_call_id=request.tool_call['id'],
            content=f'工具调用错误：{str(e)}'
        )

@wrap_tool_call
async def ahandle_tool_errors(request:ModelRequest, handler:Callable) -> ModelResponse:
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            tool_call_id=request.tool_call['id'],
            content=f'工具调用错误：{str(e)}'
        )

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tool')))
    from langchain.agents import create_agent
    from rag import check_knowledge
    base_model = init_chat_model(
        model="ollama:qwen3:8b",
        temperture=0.7,
        timeout=5
    )
    agent = create_agent(
        model = base_model,
        tools = [check_knowledge],
        system_prompt=f'如果用户询问草药 → 直接调用数据库工具检索',
        middleware=[dynamic_model_select, handle_tool_errors]
    )
    res = agent.invoke({'messages':[{'role':'user', 'content':"什么是枸杞？"}]})
    print(res)
