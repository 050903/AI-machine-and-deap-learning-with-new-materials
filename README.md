![image](https://github.com/user-attachments/assets/89fbe7f5-3d80-4e79-90dc-a02e2af49a5c)# 🧪 Material Recognition AI

A professional desktop application for material recognition using deep learning and an integrated AI chatbot (Google Gemini).
# Running
![image](https://github.com/user-attachments/assets/876addad-7bf3-4536-bc35-3dffe555b048)
![image](https://github.com/user-attachments/assets/fd790a15-7e31-4a3c-844c-d7265cae1285)
![image](https://github.com/user-attachments/assets/d18d967e-f6b5-405d-bf71-d56f552292d5)
![image](https://github.com/user-attachments/assets/8e909a96-cdea-4665-8e13-6228c11beb4e)
![image](https://github.com/user-attachments/assets/d091b3a4-d0a6-43da-80bf-feab17a23e1c)
![image](https://github.com/user-attachments/assets/bede35eb-e248-4ff8-9ad9-cbb12397a99c)
![image](https://github.com/user-attachments/assets/0de6d201-bbe3-471c-8cad-1759e419c96c)
![image](https://github.com/user-attachments/assets/a3c630cd-b4c4-43a4-8fab-676bfc329a40)
![image](https://github.com/user-attachments/assets/97f6f36f-cc6d-4e10-bcbb-eda5f77d367c)
![image](https://github.com/user-attachments/assets/7e7cc39b-7515-4f51-98ed-6dd48de27ec1)
![image](https://github.com/user-attachments/assets/581ebb47-be36-4d64-a731-766f2bb7dd9d)
![image](https://github.com/user-attachments/assets/7dbdbde1-4cb4-43bd-a372-bbb0ae10c239)
![image](https://github.com/user-attachments/assets/304d473c-5882-41d4-bee5-69a18edfa49c)
![image](https://github.com/user-attachments/assets/00011ea9-0b92-4008-9cc2-9cfb9087ff8f)
![image](https://github.com/user-attachments/assets/8dd68577-8b7e-4774-a4c4-b4fb0511fa1b)
![image](https://github.com/user-attachments/assets/419c9396-0187-4c9c-963a-7ccf1ad9a609)

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
