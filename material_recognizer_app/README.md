# 🧪 Material Recognition AI

A professional desktop application for material recognition using deep learning and an integrated AI chatbot (Google Gemini).

## ✨ Features
- 🏷️ **Material Recognition**: Classifies images or webcam frames as different material types (e.g., metal, plastic) using a trained MobileNetV2 model (TensorFlow/Keras).
- 🖥️ **User-Friendly GUI**: Built with Tkinter, including image/video display, prediction controls, and status messages.
- 🏋️ **Model Training**: Train your own model on custom data with a single click.
- 🤖 **AI Chatbot**: Ask questions about materials, the app, or code using Google Gemini (Generative AI) directly in the app.
- 🎤 **Voice Chat**: Speak to the AI and hear its responses (Speech-to-Text and Text-to-Speech integration).

## 🛠️ Requirements
- 🐍 Python 3.9 64-bit
- 📦 See `requirements.txt` for all dependencies

## 🚀 Setup Instructions
1. **Clone the repository**
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   pip install speechrecognition pyttsx3 pyaudio google-generativeai
   ```
3. **Insert your Google Generative AI API key**:
   - Open `app_ui.py`
   - Find the line:
     ```python
     genai.configure(api_key="YOUR_GOOGLE_API_KEY_HERE")
     ```
   - Replace with your own API key from [Google AI Studio](https://aistudio.google.com/).
4. **Prepare your data**:
   - Place training images in `data/<class_name>/` folders (e.g., `data/metal/`, `data/plastic/`).
5. **Run the app**:
   ```sh
   python main.py
   ```

## 📝 Usage
- 🏋️‍♂️ **Train Model**: Click "Train Model" to train on your dataset.
- 🖼️ **Load Image**: Select an image to classify.
- 📷 **Start Webcam**: Use your webcam for real-time material recognition.
- 💬 **AI Chatbot**: Type or speak your question in the chat section. The AI will answer and read the answer aloud.

## 🔒 Security Note
- **API Key**: Never share your Google API key publicly. The code uses a placeholder; insert your own key before use.

## 📄 License
This project is for educational and research purposes. 