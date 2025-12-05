import gradio as gr
import subprocess
import time

# --- 1. 設定 ---

# 定義可用的 GraphRAG 專案路徑
# 注意：你的 v1 和 v2 路徑相同，請確認是否正確
PROJECT_PATHS = {
    "Baseline version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\ipmnproject",
    "CSV version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\csv_ipmnproject",
    "TXT version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\modi_ipmnproject",
    "CSV v2 version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\temp_csv_ipmnproject",
    "Tuned CSV version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\tuned_csv_graphrag",
}

# --- 2. 後端函式 ---

def run_graphrag_query(message, history, project_name, method):
    """
    執行 GraphRAG 查詢並返回結果。
    """
    project_path = PROJECT_PATHS[project_name]
    
    # 顯示正在處理的訊息
    yield "⏳ Running query... Please wait."
    
    cmd = [
        "graphrag", "query",
        "--root", project_path,
        "--method", method,
        "--query", message  # 使用 --query 參數傳遞問題
    ]

    
    try:
        # 執行指令
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=project_path,  # 在專案目錄下執行
            shell=True # 建議在 Windows 上使用 shell=True
        )
        
        # 組合 stdout 和 stderr
        response = result.stdout or result.stderr
        
        if not response.strip():
            response = "⚠️ No response from GraphRAG. Please check the console for errors."
            
        # 返回最終結果
        yield response

    except FileNotFoundError:
        yield "❌ Error: 'python' or 'graphrag' command not found. Make sure GraphRAG is installed and in your system's PATH."
    except Exception as e:
        yield f"❌ An unexpected error occurred: {str(e)}"

# --- 3. Gradio 界面 ---

with gr.Blocks(theme=gr.themes.Soft(), title="GraphRAG Query Console") as demo:
    gr.Markdown("# 🤖 GraphRAG Query Console")
    gr.Markdown("Ask questions about your documents and get insights extracted via GraphRAG.")
    
    with gr.Row():
        # 左側設定欄
        with gr.Column(scale=1):
            project_dropdown = gr.Dropdown(
                choices=list(PROJECT_PATHS.keys()),
                value=list(PROJECT_PATHS.keys())[0],
                label="📁 Select GraphRAG Project",
                interactive=True
            )
            method_dropdown = gr.Dropdown(
                choices=["local", "global", "drift", "basic"],
                value="local",
                label="🔍 Search Method",
                interactive=True
            )
            gr.Markdown("""
            **Search Methods:**
            - **Local**: Best for specific questions.
            - **Global**: Best for broad, overview questions.
            - **Drift**: For exploration-based search.
            - **Basic**: Simple keyword-based search.
            """)
        
        # 右側聊天界面
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat History", height=900, bubble_full_width=False)
            msg = gr.Textbox(label="Your Question", placeholder="Type your question here and press Enter...", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")

    # 將函式綁定到界面元件
    def respond(message, chat_history, project_name, method):
        # 將使用者訊息加入歷史紀錄
        chat_history.append([message, None])
        # 為了顯示 "Running query..."，先更新一次界面
        yield chat_history
        
        # 執行查詢並取得生成器
        query_generator = run_graphrag_query(message, chat_history, project_name, method)
        
        # 逐步更新聊天機器人的回覆
        bot_message = ""
        for chunk in query_generator:
            bot_message = chunk
            chat_history[-1][1] = bot_message
            yield chat_history

    # 處理送出事件 (點擊按鈕或按 Enter)
    msg.submit(respond, [msg, chatbot, project_dropdown, method_dropdown], [chatbot]).then(
        lambda: gr.update(value=""), None, [msg], queue=False
    )
    submit_btn.click(respond, [msg, chatbot, project_dropdown, method_dropdown], [chatbot]).then(
        lambda: gr.update(value=""), None, [msg]
    )

    # 處理清除事件
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# 啟動應用
demo.launch(share=False)