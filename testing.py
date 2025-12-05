import gradio as gr
import subprocess
from dotenv import load_dotenv
import os

# --- 1. 初始化與設定 ---

# 載入 .env 檔案 (為了自訂 Pipeline)
load_dotenv()

# 匯入自訂的 Pipeline 類別
# 確保 GraphRAG_baseline.py 和 demo.py 在同一個資料夾
try:
    from GraphRAG_baseline import GraphRAGPipeline
except ImportError:
    print("❌ 錯誤: 找不到 'GraphRAG_baseline.py'。請確保它與 demo.py 在同一個資料夾中。")
    GraphRAGPipeline = None


# --- Microsoft GraphRAG 專案路徑 ---
MS_PROJECT_PATHS = {
    "Baseline version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\ipmnproject",
    "CSV version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\csv_ipmnproject",
    "TXT version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\modi_ipmnproject",
    "CSV v2 version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\temp_csv_ipmnproject",
    "Tuned CSV version": r"C:\Users\User\Desktop\Microsoft_GraphRAG\tuned_csv_graphrag",
}

# --- 新增一個選項來代表我們自訂的 Pipeline ---
CUSTOM_PIPELINE_NAME = "Custom Neo4j Pipeline (Fast)"
ALL_PROJECTS = [CUSTOM_PIPELINE_NAME] + list(MS_PROJECT_PATHS.keys())

# --- 在應用程式啟動時，初始化自訂的 Pipeline ---
custom_pipeline = None
if GraphRAGPipeline:
    try:
        print("Initializing Custom Neo4j Pipeline...")
        custom_pipeline = GraphRAGPipeline()
        print("✅ Custom Pipeline Initialized.")
    except Exception as e:
        print(f"❌ 嚴重錯誤：無法初始化 Custom Neo4j Pipeline。")
        print(f"錯誤細節: {e}")
else:
    # 如果無法匯入，將自訂選項從列表中移除
    ALL_PROJECTS.remove(CUSTOM_PIPELINE_NAME)


# --- 2. 後端函式 ---

def run_ms_graphrag(message, project_path, method):
    """執行 Microsoft GraphRAG 的 subprocess 指令。"""
    cmd = ["graphrag", "query", "--root", project_path, "--method", method, "--query", message]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_path, shell=True)
        response = result.stdout or result.stderr
        return response if response.strip() else "⚠️ No response from Microsoft GraphRAG."
    except Exception as e:
        return f"❌ An unexpected error occurred with Microsoft GraphRAG: {str(e)}"

def run_custom_pipeline(message):
    """執行我們自訂的 Neo4j Pipeline。"""
    if not custom_pipeline:
        return "❌ Custom Pipeline is not available due to an initialization error."
    try:
        response = custom_pipeline.run(message)
        return response if response.strip() else "⚠️ Custom Pipeline returned an empty response."
    except Exception as e:
        return f"❌ An unexpected error occurred with Custom Pipeline: {str(e)}"

def handle_query(message, project_name, method):
    """根據選擇的專案，路由到對應的執行函式。"""
    yield "⏳ Running query... Please wait."
    
    if project_name == CUSTOM_PIPELINE_NAME:
        # 如果選擇了自訂 Pipeline
        response = run_custom_pipeline(message)
    else:
        # 如果選擇了 Microsoft 的專案
        project_path = MS_PROJECT_PATHS[project_name]
        response = run_ms_graphrag(message, project_path, method)
        
    yield response

# --- 3. Gradio 界面 ---

with gr.Blocks(theme=gr.themes.Soft(), title="GraphRAG Query Console") as demo:
    gr.Markdown("# 🤖 GraphRAG Query Console")
    gr.Markdown("Select a project to query. The 'Custom Neo4j Pipeline' uses a direct, optimized RAG flow.")
    
    with gr.Row():
        # 左側設定欄
        with gr.Column(scale=1):
            project_dropdown = gr.Dropdown(
                choices=ALL_PROJECTS,
                value=ALL_PROJECTS[0],
                label="📁 Select Project / Pipeline",
                interactive=True
            )
            method_dropdown = gr.Dropdown(
                choices=["local", "global"],
                value="local",
                label="🔍 Search Method (for MS GraphRAG)",
                interactive=True
            )
            gr.Markdown("""
            **Note:**
            - The **Custom Neo4j Pipeline** is much faster as it's always running.
            - The **Search Method** dropdown only applies to the Microsoft GraphRAG projects.
            """)
        
        # 右側聊天界面
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat History", height=1200, bubble_full_width=False)
            msg = gr.Textbox(label="Your Question", placeholder="Type your question here and press Enter...", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")

    # --- 界面邏輯 ---
    
    # 當下拉選單改變時，如果選的是自訂 Pipeline，就禁用 "Search Method" 選單
    def toggle_method_dropdown(project_name):
        is_custom = (project_name == CUSTOM_PIPELINE_NAME)
        return gr.update(interactive=not is_custom)
    
    project_dropdown.change(toggle_method_dropdown, project_dropdown, method_dropdown)

    # 處理聊天回應
    def respond(message, chat_history, project_name, method):
        chat_history.append([message, None])
        yield chat_history
        
        query_generator = handle_query(message, project_name, method)
        
        bot_message = ""
        for chunk in query_generator:
            bot_message = chunk
            chat_history[-1][1] = bot_message
            yield chat_history

    # 綁定事件
    msg.submit(respond, [msg, chatbot, project_dropdown, method_dropdown], [chatbot]).then(
        lambda: gr.update(value=""), None, [msg], queue=False
    )
    submit_btn.click(respond, [msg, chatbot, project_dropdown, method_dropdown], [chatbot]).then(
        lambda: gr.update(value=""), None, [msg]
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# 啟動應用
if __name__ == "__main__":
    demo.launch(share=False)