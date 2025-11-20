import os
import json
import time
from datetime import datetime
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# --- Configuration ---
DEFAULT_KEY = "Enter Your GEMINI API KEY HERE"
API_KEY = os.getenv("GEMINI_API_KEY", DEFAULT_KEY)

if API_KEY:
    genai.configure(api_key=API_KEY)

# --- Prompt Templates ---
MOOD_DETECTION_PROMPT = """
You are an expert psychological mood analyzer. Analyze the following user input (text and/or audio transcript) and determine the user's current mood.
Classify the mood into one of these labels: Happy, Sad, Anxious, Angry, Neutral, Overwhelmed, Depressive, Suicidal-Risk.
Also provide a confidence score (0-100).
Input: "{input_text}"
Output format (JSON):
{{
  "mood": "Label",
  "confidence": 85,
  "risk_flag": boolean
}}
"""

THERAPY_RESPONSE_PROMPT = """
You are a compassionate, empathetic therapy coach named EmotiCare. The user is feeling "{mood}".
Generate a supportive, non-judgmental response. Acknowledge their feelings, validate them, and offer a comforting perspective.
Keep it warm and conversational. Do not diagnose.
User Input: "{input_text}"
"""

JOURNALING_ANALYSIS_PROMPT = """
Analyze the user's entry for journaling purposes.
Extract 3 key themes or emotions.
Suggest 2 specific journaling prompts.
User Input: "{input_text}"
Mood: "{mood}"
Output format (JSON):
{{
  "themes": ["theme1", "theme2", "theme3"],
  "prompts": ["prompt1", "prompt2"]
}}
"""

CRISIS_CHECK_PROMPT = """
Analyze the following input strictly for self-harm, suicide, or immediate danger.
If ANY risk is detected, output TRUE and a short reason. Otherwise output FALSE.
Input: "{input_text}"
Output format (JSON):
{{
  "is_crisis": boolean,
  "reason": "explanation"
}}
"""

SUMMARY_PROMPT = """
Summarize the user's situation in 7-10 words.
Then provide 3 short, actionable coping mechanisms:
1. Breathing/grounding (Immediate).
2. Small step (Now).
3. Long-term action.
User Input: "{input_text}"
Mood: "{mood}"
Output format (JSON):
{{
  "summary": "short summary",
  "actions": {{
    "breathing": "description",
    "immediate": "description",
    "long_term": "description"
  }}
}}
"""

# --- Helper Functions ---
HISTORY_FILE = "journal_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:20]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    return history

def format_history_html(history):
    if not history:
        return "<p style='color: #9ca3af; text-align: center; margin-top: 20px;'>No entries yet. Start journaling!</p>"
    
    html = "<div class='history-container'>"
    for item in history:
        mood = item.get('mood', 'Neutral')
        color_map = {
            'Happy': '#10b981', 'Neutral': '#6b7280', 
            'Sad': '#3b82f6', 'Depressive': '#1d4ed8',
            'Anxious': '#f59e0b', 'Overwhelmed': '#d97706',
            'Angry': '#ef4444', 'Suicidal-Risk': '#b91c1c'
        }
        color = color_map.get(mood, '#6b7280')
        
        html += f"""
        <div class='history-card' style='border-left: 4px solid {color};'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <span style='font-weight: bold; color: {color};'>{mood}</span>
                <span style='font-size: 0.8em; color: #9ca3af;'>{item['timestamp']}</span>
            </div>
            <p style='font-size: 0.9em; color: #4b5563; margin: 0;'>{item['summary']}</p>
        </div>
        """
    html += "</div>"
    return html

def plot_mood_trend(history):
    if not history:
        return None
    df = pd.DataFrame(history)
    if 'mood' not in df.columns:
        return None
    mood_counts = df['mood'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#f9fafb')
    ax.set_facecolor('#f9fafb')
    
    colors = ['#2dd4bf' for _ in range(len(mood_counts))]
    mood_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_title("Mood Trends", fontsize=10, fontweight='bold', color='#374151')
    ax.set_xlabel("", fontsize=8)
    ax.set_ylabel("Entries", fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=8, colors='#4b5563')
    ax.tick_params(axis='y', labelsize=8, colors='#4b5563')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

# --- ROBUST MODEL GENERATION ---
def generate_with_fallback(prompt):
    """
    Attempts to generate content using a list of models in priority order.
    If one fails (e.g. 404 Not Found), it tries the next.
    """
    # Priority list: 1.5 Flash (Fastest) -> 1.5 Pro (Best) -> Pro (Legacy/Stable)
    candidate_models = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-pro',
        'gemini-2.0-pro'
    ]
    
    last_error = None
    
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response
        except Exception as e:
            last_error = e
            # Continue to next model
            continue
            
    # If all failed, raise the last error
    raise last_error

def analyze_process(text_input, audio_input):
    if not API_KEY:
        return "⚠️ API Key Error.", "", "", "", None, "", gr.update(visible=False)
    if not text_input and not audio_input:
        return "Please share your thoughts first.", "", "", "", None, "", gr.update(visible=False)

    final_input = text_input

    try:
        # 1. Mood Detection
        mood_res = generate_with_fallback(MOOD_DETECTION_PROMPT.format(input_text=final_input))
        mood_data = json.loads(mood_res.text.replace('```json', '').replace('```', ''))
        mood = mood_data.get("mood", "Unknown")
        confidence = mood_data.get("confidence", 0)
        is_risk = mood_data.get("risk_flag", False)

        # 2. Crisis Check
        if is_risk:
            crisis_result = {"is_crisis": True}
        else:
            crisis_res = generate_with_fallback(CRISIS_CHECK_PROMPT.format(input_text=final_input))
            crisis_result = json.loads(crisis_res.text.replace('```json', '').replace('```', ''))

        # Safety Intervention
        if crisis_result.get("is_crisis"):
            risk_msg = f"""
            <div class='risk-alert'>
                <h3>⚠️ CRITICAL SAFETY WARNING</h3>
                <p>We detected content indicating high distress. Please prioritize your safety.</p>
                <p><strong>Immediate Help:</strong></p>
                <ul>
                    <li>🇺🇸 USA: 988</li>
                    <li>🇮🇳 India: 9152987821</li>
                    <li>Global: <a href='https://findahelpline.com' target='_blank'>findahelpline.com</a></li>
                </ul>
            </div>
            """
            return "Crisis Detected", "Please seek help.", "See safety card.", "High Risk", None, format_history_html(load_history()), gr.update(value=risk_msg, visible=True)

        # 3. Therapy Response
        therapy_msg = generate_with_fallback(THERAPY_RESPONSE_PROMPT.format(input_text=final_input, mood=mood)).text

        # 4. Summary & Actions
        summary_res = generate_with_fallback(SUMMARY_PROMPT.format(input_text=final_input, mood=mood))
        summary_data = json.loads(summary_res.text.replace('```json', '').replace('```', ''))
        
        # 5. Journaling
        journal_res = generate_with_fallback(JOURNALING_ANALYSIS_PROMPT.format(input_text=final_input, mood=mood))
        journal_data = json.loads(journal_res.text.replace('```json', '').replace('```', ''))

        # Save History
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "input": final_input[:50] + "...",
            "mood": mood,
            "confidence": confidence,
            "summary": summary_data.get("summary", "")
        }
        history = save_history(entry)
        
        # Format Outputs
        actions_html = f"""
        <div class='action-list'>
            <div class='action-item'><b>🌬️ Breathe:</b> {summary_data['actions'].get('breathing')}</div>
            <div class='action-item'><b>⚡ Do Now:</b> {summary_data['actions'].get('immediate')}</div>
            <div class='action-item'><b>📅 Plan:</b> {summary_data['actions'].get('long_term')}</div>
        </div>
        """
        
        journal_html = f"""
        <div class='journal-box'>
            <p><b>Themes:</b> {', '.join(journal_data.get('themes', []))}</p>
            <p><b>Reflect on this:</b></p>
            <ol>
                <li>{journal_data.get('prompts', [''])[0]}</li>
                <li>{journal_data.get('prompts', [''])[1] if len(journal_data.get('prompts', [])) > 1 else ''}</li>
            </ol>
        </div>
        """

        return (
            f"{mood} ({confidence}%)", 
            therapy_msg, 
            actions_html, 
            journal_html, 
            plot_mood_trend(history), 
            format_history_html(history),
            gr.update(visible=False)
        )

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", None, "", gr.update(visible=False)

def export_session(mood, therapy, actions):
    filename = f"session_{int(time.time())}.txt"
    with open(filename, "w") as f:
        f.write(f"EmotiCare Session\nMood: {mood}\n\nTherapy:\n{therapy}\n\nActions:\n{actions}")
    return filename

# --- UI Styling & Theme ---
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="#f9fafb",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
    button_primary_background_fill="#0d9488",
    button_primary_background_fill_hover="#0f766e",
    button_primary_text_color="white",
)

custom_css = """
.gradio-container {max-width: 1200px !important; margin: auto;}
h1.title {text-align: center; color: #0f766e; font-weight: 800; font-size: 2.5rem; margin-bottom: 0.5rem;}
p.subtitle {text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
.risk-alert {background-color: #fef2f2; border: 1px solid #ef4444; padding: 20px; border-radius: 12px; color: #b91c1c;}
.history-card {background: white; padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.05);}
.history-container {max-height: 500px; overflow-y: auto; padding-right: 5px;}
.action-list .action-item {margin-bottom: 8px; padding: 8px; background: #f0fdfa; border-radius: 6px; border-left: 3px solid #0d9488;}
.journal-box {background: #fff7ed; padding: 15px; border-radius: 8px; border: 1px solid #fed7aa;}
"""

# --- Main App Layout ---
with gr.Blocks(theme=theme, css=custom_css, title="EmotiCare AI") as app:
    gr.HTML("""
    <div style="padding: 20px;">
        <h1 class="title">🌿 EmotiCare AI</h1>
        <p class="subtitle">Your personal AI companion for mood tracking, therapy coaching, and mindfulness.</p>
    </div>
    """)
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            with gr.Group():
                txt_input = gr.Textbox(
                    label="How are you feeling right now?", 
                    placeholder="Take a deep breath and share your thoughts... (e.g., 'I'm feeling anxious about my presentation tomorrow.')", 
                    lines=4,
                    show_label=True
                )
                analyze_btn = gr.Button("✨ Analyze & Support Me", variant="primary", size="lg")
            
            risk_card = gr.HTML(visible=False)
            
            gr.Markdown("### 🧠 Insights & Support")
            
            with gr.Row():
                lbl_mood = gr.Textbox(label="Detected Mood", interactive=False, scale=1)
            
            txt_therapy = gr.Textbox(label="EmotiCare Coach", lines=5, interactive=False, show_copy_button=True)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🛡️ Coping Strategy")
                    html_actions = gr.HTML()
                with gr.Column():
                    gr.Markdown("#### 📔 Journaling Prompts")
                    html_journal = gr.HTML()
            
            with gr.Accordion("📥 Export Session", open=False):
                btn_export = gr.Button("Download Summary as .txt")
                file_output = gr.File(label="Download")

        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📊 Your Journey")
            plot_output = gr.Plot(label="Mood Trend", container=False)
            
            gr.Markdown("### 🕒 Recent Entries")
            history_display = gr.HTML()
            
            gr.Markdown("""
            <div style="font-size: 0.8em; color: #9ca3af; margin-top: 20px; text-align: center;">
            Privacy: Data sent to Gemini API. History stored locally. <br>
            <b>Not a substitute for professional medical advice.</b>
            </div>
            """)

    analyze_btn.click(
        analyze_process,
        inputs=[txt_input, txt_input],
        outputs=[lbl_mood, txt_therapy, html_actions, html_journal, plot_output, history_display, risk_card]
    )
    
    btn_export.click(
        export_session,
        inputs=[lbl_mood, txt_therapy, html_actions],
        outputs=[file_output]
    )
    
    app.load(lambda: (plot_mood_trend(load_history()), format_history_html(load_history())), outputs=[plot_output, history_display])

if __name__ == "__main__":
    print("--- EmotiCare UI Updated ---")

    app.launch()
