# EmotiCare — AI Mood Detector & Therapy Coach

EmotiCare is a local-run Gradio web app that uses Google Gemini to detect your mood, provide supportive therapy-style coaching, and suggest coping mechanisms.

## Features
- **Mood Detection**: Identifies moods like Happy, Sad, Anxious, etc.
- **Therapy Coach**: Generates empathetic, actionable advice.
- **Safety Check**: Detects high-risk language and provides emergency resources.
- **Journaling**: Tracks mood history and suggests prompts.

## Setup & Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Get Gemini API Key**:
    -   Get your free key from [Google AI Studio](https://aistudio.google.com/).

3.  **Set API Key**:
    -   Create a file named `.env` in this folder.
    -   Add this line: `GEMINI_API_KEY=your_actual_api_key_here`
    -   OR set it in your terminal: `export GEMINI_API_KEY=your_key`

4.  **Run the App**:
    ```bash
    python app.py
    ```
    -   Open the link shown in the terminal (usually `http://127.0.0.1:7860`).

## Privacy
This app runs locally. Your data is sent to Google Gemini API for processing but is not stored on any third-party servers by the app itself.