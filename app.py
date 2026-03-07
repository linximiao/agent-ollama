from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask import send_file, abort
import os
import pandas as pd
from werkzeug.utils import secure_filename
from agent_main import Agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import shutil
import uuid
from datetime import datetime

agent = Agent()
app = Flask(__name__)
app.secret_key = 'your-super-secret-key-for-session'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_new_session():
    """创建新会话"""
    thread_id = str(uuid.uuid4())
    session_id = f"session_{thread_id}"
    
    # 重置agent配置
    agent.start_new_conversation(thread_id)
    
    # 创建新的会话数据
    new_session = {
        'id': session_id,
        'thread_id': thread_id,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'title': '新对话',
        'messages': [
            {"role": "bot", "text": "你好！今天需要什么帮助吗？"}
        ],
        'table_html': None
    }
    
    # 初始化上传目录
    session_path = os.path.join(app.config['UPLOAD_FOLDER'], thread_id)
    os.makedirs(session_path, exist_ok=True)
    
    return new_session

@app.route('/', methods=['GET', 'POST'])
def index():
    # 检查是否需要创建新会话
    if 'current_session_id' not in session or request.args.get('new_session'):
        # 创建新会话
        new_session = create_new_session()
        session['current_session_id'] = new_session['id']
        session['sessions'] = session.get('sessions', [])
        session['sessions'].insert(0, new_session)  # 插入到最前面
        
        # 限制会话数量，只保留最近10个
        if len(session['sessions']) > 10:
            # 删除最旧的会话文件
            old_session = session['sessions'].pop()
            old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_session['thread_id'])
            if os.path.exists(old_path):
                shutil.rmtree(old_path)
    
    # 获取当前会话
    current_session = None
    for sess in session.get('sessions', []):
        if sess['id'] == session['current_session_id']:
            current_session = sess
            break
    
    if not current_session:
        # 如果找不到当前会话，创建一个新的
        current_session = create_new_session()
        session['current_session_id'] = current_session['id']
        session['sessions'] = session.get('sessions', [])
        session['sessions'].insert(0, current_session)
    
    # 设置agent的thread_id
    agent.config["configurable"]["thread_id"] = current_session['thread_id']
    
    table_html = current_session.get('table_html', None)

    if request.method == 'POST':
        if 'clear_chat' in request.form:
            session.pop('conversation', None)
            session.pop('table_html', None)
            path = os.path.join(app.config['UPLOAD_FOLDER'], agent.config["configurable"]["thread_id"])
            if os.path.exists(path):
                shutil.rmtree(path)
            agent.start_new_conversation()
            session.modified = True
            return redirect(url_for('index'))
        elif 'switch_session' in request.form:
            # 切换会话
            session_id = request.form['session_id']
            session['current_session_id'] = session_id
            session.modified = True
            return redirect(url_for('index'))
        elif 'delete_session' in request.form:
            # 删除会话
            session_id = request.form['session_id']
            sessions = session.get('sessions', [])
            
            # 找到要删除的会话
            for i, sess in enumerate(sessions):
                if sess['id'] == session_id:
                    # 删除会话文件
                    path = os.path.join(app.config['UPLOAD_FOLDER'], sess['thread_id'])
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    
                    # 如果删除的是当前会话，创建新的
                    if session_id == session['current_session_id']:
                        new_session = create_new_session()
                        session['current_session_id'] = new_session['id']
                        sessions.insert(0, new_session)
                    
                    # 删除会话
                    sessions.pop(i)
                    break
            
            session['sessions'] = sessions
            session.modified = True
            return redirect(url_for('index'))
        # 情况1: 发送文本消息
        elif 'message' in request.form and request.form['message'].strip():
            user_msg = request.form['message'].strip()
            current_session['messages'].append({"role": "user", "text": user_msg})

            # 更新会话标题（如果还是默认标题）
            if current_session['title'] == '新对话':
                current_session['title'] = user_msg[:20] + '...' if len(user_msg) > 20 else user_msg

            for event in agent.app.stream({"messages": [HumanMessage(user_msg), SystemMessage(agent.sys)]}, agent.config, stream_mode="values"):
                event["messages"][-1].pretty_print()
                if isinstance(event["messages"][-1], AIMessage):
                    current_session['messages'].append({"role": "bot", "text": event["messages"][-1].content.replace('\n', '<br>')})
            session.modified = True

        # 情况2: 上传文件
        elif 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], os.path.join(current_session['thread_id'], filename))
                agent.get_file(filepath)
                file.save(filepath)
                # 添加用户消息
                current_session['messages'].append({"role": "user", "text": f"上传了文件：{filename}"})
                session.modified = True
            else:
                flash("请上传有效的文件")
        else:
            flash("无效操作")

    # 更新当前会话到sessions列表
    for i, sess in enumerate(session.get('sessions', [])):
        if sess['id'] == session['current_session_id']:
            session['sessions'][i] = current_session
            break

    return render_template(
        'index.html',
        conversation=current_session['messages'],
        table_html=table_html,
        sessions=session.get('sessions', []),
        current_session_id=session['current_session_id']
    )

@app.route('/new_session')
def new_session():
    """创建新会话的路由"""
    return redirect(url_for('index', new_session='true'))

if __name__ == '__main__':
    app.run(debug=True)