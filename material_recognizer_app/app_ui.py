# app_ui.py

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import threading
import time # For simulation delay
import os
from tkinter.scrolledtext import ScrolledText
# import openai  # Không dùng nữa
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3

from config import APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, IMG_WIDTH, IMG_HEIGHT, MODEL_PATH
from data_loader import MaterialDataLoader
from model_trainer import MaterialModelTrainer
from material_classifier import MaterialClassifier

class MaterialRecognitionApp:
    """
    Main application class for the Material Recognition AI with a professional UI.
    """
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False) # Fixed window size

        self.data_loader = MaterialDataLoader()
        self.trainer = None # Initialized only when training is requested
        self.classifier = None # Initialized after model is ready

        self.class_names = []

        self.speak_engine = pyttsx3.init()

        self._create_widgets()
        self._initialize_classifier() # Try to load classifier on startup

    def _create_widgets(self):
        """Creates and arranges all GUI elements."""
        # Main Frame for content
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Section: Model Status and Training ---
        model_status_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        model_status_frame.pack(fill=tk.X, pady=5)

        self.model_status_label = ttk.Label(model_status_frame, text="Loading model...", foreground="orange")
        self.model_status_label.pack(side=tk.LEFT, padx=5)

        self.train_button = ttk.Button(model_status_frame, text="Train Model", command=self._start_training_thread)
        self.train_button.pack(side=tk.RIGHT, padx=5)

        # --- Middle Section: Image/Video Display and Controls ---
        content_frame = ttk.Frame(main_frame, padding="5")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left side: Image/Video Display
        display_frame = ttk.LabelFrame(content_frame, text="Live Feed / Image", padding="5")
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.video_label = ttk.Label(display_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind("<Configure>", self._resize_image_on_configure)

        # Right side: Controls for Image/Video
        control_frame = ttk.LabelFrame(content_frame, text="Controls", padding="10")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        ttk.Label(control_frame, text="Prediction:").pack(pady=(0, 5))
        self.prediction_label = ttk.Label(control_frame, text="N/A", font=("Helvetica", 14, "bold"), wraplength=200)
        self.prediction_label.pack(pady=(0, 10))

        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        self.load_image_button = ttk.Button(control_frame, text="Load Image", command=self._load_image)
        self.load_image_button.pack(pady=5, fill=tk.X)

        self.webcam_button = ttk.Button(control_frame, text="Start Webcam", command=self._toggle_webcam)
        self.webcam_button.pack(pady=5, fill=tk.X)
        self.is_webcam_active = False
        self.webcam_thread = None
        self.stop_webcam_event = threading.Event()

        # Placeholder for progress bar (for training)
        self.progress_bar = ttk.Progressbar(control_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(pady=10, fill=tk.X)
        self.progress_bar["value"] = 0 # Hide initially

        self.status_message = ttk.Label(main_frame, text="Ready.", font=("Helvetica", 10), anchor="w")
        self.status_message.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        # --- ChatGPT Section ---
        chat_frame = ttk.LabelFrame(main_frame, text="AI Chatbot (Google Generative AI)", padding="5")
        chat_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        self.chat_display = ScrolledText(chat_frame, height=8, state='disabled', wrap=tk.WORD)
        self.chat_display.pack(fill=tk.X, padx=5, pady=2)

        chat_input_frame = ttk.Frame(chat_frame)
        chat_input_frame.pack(fill=tk.X, padx=5, pady=2)

        self.chat_entry = ttk.Entry(chat_input_frame)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.chat_entry.bind('<Return>', lambda event: self._send_chat_message())

        self.send_button = ttk.Button(chat_input_frame, text="Send", command=self._send_chat_message)
        self.send_button.pack(side=tk.RIGHT)

        self.voice_button = ttk.Button(chat_input_frame, text="Nói", command=self._voice_input)
        self.voice_button.pack(side=tk.RIGHT, padx=(5,0))

    def _initialize_classifier(self):
        """Attempts to load the classifier model on application start."""
        try:
            # First, load class names from data_loader
            # This requires at least dummy data to be present for data_loader to infer classes
            _, _, self.class_names = self.data_loader.load_data()
            self.classifier = MaterialClassifier(class_names=self.class_names)
            self.model_status_label.config(text="Model Loaded!", foreground="green")
            self._enable_prediction_controls(True)
        except FileNotFoundError:
            self.model_status_label.config(text="Model Not Found. Please Train.", foreground="red")
            self._enable_prediction_controls(False)
        except Exception as e:
            self.model_status_label.config(text=f"Error loading model: {e}", foreground="red")
            self._enable_prediction_controls(False)
        self.root.update_idletasks() # Refresh UI

    def _enable_prediction_controls(self, enable):
        """Enables/disables prediction-related buttons."""
        state = tk.NORMAL if enable else tk.DISABLED
        self.load_image_button.config(state=state)
        self.webcam_button.config(state=state)

    def _start_training_thread(self):
        """Starts the model training in a separate thread to keep UI responsive."""
        self.train_button.config(state=tk.DISABLED)
        self._enable_prediction_controls(False)
        self.model_status_label.config(text="Training in progress...", foreground="blue")
        self.status_message.config(text="Training model. This may take a while...")
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()

        training_thread = threading.Thread(target=self._train_model_logic)
        training_thread.daemon = True # Allow thread to exit with main app
        training_thread.start()

    def _train_model_logic(self):
        """Contains the actual model training steps."""
        try:
            train_gen, val_gen, self.class_names = self.data_loader.load_data()
            if not self.class_names:
                raise ValueError("No classes found in data directory. Training aborted.")

            self.trainer = MaterialModelTrainer(num_classes=len(self.class_names))
            self.trainer.build_model()
            self.trainer.compile_model()
            self.trainer.train_model(train_gen, val_gen)

            # Re-initialize classifier with the newly trained model
            self.classifier = MaterialClassifier(class_names=self.class_names)
            messagebox.showinfo("Training Complete", "Model training finished successfully!")
            self.model_status_label.config(text="Model Trained & Loaded!", foreground="green")
            self.status_message.config(text="Ready.")
            self._enable_prediction_controls(True)

            # Optionally plot history after training
            # self.trainer.plot_training_history()

        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during training: {e}")
            self.model_status_label.config(text=f"Training Failed. {e}", foreground="red")
            self.status_message.config(text=f"Training failed: {e}")
            self._enable_prediction_controls(False) # Keep disabled if training failed
        finally:
            self.train_button.config(state=tk.NORMAL)
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", value=0)


    def _load_image(self):
        """Opens a file dialog to select an image for prediction."""
        if not self.classifier:
            messagebox.showwarning("Model Not Ready", "Please train or load the model first.")
            return

        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.status_message.config(text=f"Processing image: {os.path.basename(file_path)}...")
            try:
                # Stop webcam if active before loading image
                if self.is_webcam_active:
                    self._toggle_webcam()
                    time.sleep(0.5) # Give webcam a moment to stop gracefully

                # Display the image
                pil_image = Image.open(file_path)
                self._display_image_on_label(pil_image)

                # Predict in a separate thread
                prediction_thread = threading.Thread(target=self._predict_image_logic, args=(file_path,))
                prediction_thread.daemon = True
                prediction_thread.start()

            except Exception as e:
                messagebox.showerror("Image Error", f"Could not load or process image: {e}")
                self.status_message.config(text=f"Error: {e}", foreground="red")


    def _predict_image_logic(self, image_path):
        """Performs prediction on a loaded image in a separate thread."""
        try:
            predicted_material, confidence = self.classifier.predict_image(image_path)
            self.prediction_label.config(text=f"{predicted_material}\n({confidence:.2f})")
            self.status_message.config(text=f"Prediction complete for {os.path.basename(image_path)}.")
        except Exception as e:
            self.prediction_label.config(text="Error!")
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.status_message.config(text=f"Prediction error: {e}", foreground="red")


    def _toggle_webcam(self):
        """Starts or stops the webcam stream."""
        if not self.classifier:
            messagebox.showwarning("Model Not Ready", "Please train or load the model first.")
            return

        if self.is_webcam_active:
            self.stop_webcam_event.set() # Signal thread to stop
            self.webcam_button.config(text="Start Webcam")
            self.status_message.config(text="Webcam stopped.")
        else:
            self.stop_webcam_event.clear() # Clear event for new start
            self.webcam_thread = threading.Thread(target=self._webcam_stream_logic)
            self.webcam_thread.daemon = True
            self.webcam_thread.start()
            self.webcam_button.config(text="Stop Webcam")
            self.status_message.config(text="Starting webcam...")

        self.is_webcam_active = not self.is_webcam_active


    def _webcam_stream_logic(self):
        """Handles reading frames from webcam and making predictions."""
        cap = cv2.VideoCapture(0) # 0 for default webcam
        if not cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open webcam.")
            self.root.after(0, self._toggle_webcam) # Reset button state
            self.status_message.config(text="Webcam error.", foreground="red")
            return

        self.status_message.config(text="Webcam active. Detecting materials...")

        try:
            while not self.stop_webcam_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Convert OpenCV BGR image to RGB for PIL and Keras
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv2image)

                # Display the current frame
                self.root.after(0, self._display_image_on_label, pil_image)

                # Perform prediction
                predicted_material, confidence = self.classifier.predict_image(frame) # classifier takes opencv frame
                self.root.after(0, self.prediction_label.config, {'text': f"{predicted_material}\n({confidence:.2f})"})

                # Small delay to reduce CPU usage and make UI smoother
                time.sleep(0.01)

        except Exception as e:
            messagebox.showerror("Stream Error", f"An error occurred during webcam stream: {e}")
            self.status_message.config(text=f"Stream error: {e}", foreground="red")
        finally:
            cap.release()
            self.root.after(0, self._display_image_on_label, None) # Clear display
            self.root.after(0, self.prediction_label.config, {'text': "N/A"})
            # Ensure button state is correct if thread exited unexpectedly
            if self.is_webcam_active:
                self.root.after(0, self._toggle_webcam)


    def _display_image_on_label(self, pil_image):
        """Displays a PIL image on the video_label, resizing it to fit."""
        if pil_image is None:
            self.video_label.config(image='')
            self.video_label.image = None # Release reference
            return

        # Get current label size
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width == 0 or label_height == 0:
            # If label not yet rendered, use a default or wait
            label_width = IMG_WIDTH * 2 # Or some sensible default
            label_height = IMG_HEIGHT * 2

        # Resize image to fit label while maintaining aspect ratio
        img_width, img_height = pil_image.size
        ratio = min(label_width / img_width, label_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)

        self.video_label.config(image=tk_image)
        self.video_label.image = tk_image # Keep a reference to prevent garbage collection

    def _resize_image_on_configure(self, event):
        """Callback to resize the displayed image when the label size changes."""
        # This will trigger when the window or label resizes.
        # If an image is currently displayed, re-display it to fit the new size.
        if hasattr(self.video_label, "image") and self.video_label.image:
            # Re-load the original image if available (not just the PhotoImage)
            # This is tricky for the current design which doesn't store original PIL image
            # For simplicity, we can just clear and wait for next frame/image load.
            # A more robust solution would store the *last PIL Image* and redraw it.
            pass # Currently, we let the stream handle continuous updates or next image load.

    def _send_chat_message(self):
        user_message = self.chat_entry.get().strip()
        if not user_message:
            return
        self._append_chat("You", user_message)
        self.chat_entry.delete(0, tk.END)
        self.status_message.config(text="Chatbot is thinking...")
        threading.Thread(target=self._ask_gpt_and_display, args=(user_message,), daemon=True).start()

    def _append_chat(self, sender, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')

    def _voice_input(self):
        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone()
        except Exception as e:
            self.status_message.config(text=f"Không tìm thấy micro: {e}")
            return
        self.status_message.config(text="Đang nghe... Hãy nói vào micro.")
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio, language="vi-VN")
            self.chat_entry.delete(0, tk.END)
            self.chat_entry.insert(0, text)
            self._send_chat_message()
        except sr.WaitTimeoutError:
            self.status_message.config(text="Không phát hiện thấy giọng nói, vui lòng thử lại.")
        except Exception as e:
            self.status_message.config(text=f"Lỗi nhận diện giọng nói: {repr(e)}")

    def _ask_gpt_and_display(self, question):
        try:
            # --- GOOGLE AI STUDIO API KEY ---
            # Please insert your own API key below
            genai.configure(api_key="YOUR_GOOGLE_API_KEY_HERE")
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([
                {"role": "user", "parts": [question]}
            ])
            answer = response.text.strip()
        except Exception as e:
            answer = f"[Error]: {e}"
        self.root.after(0, lambda: self._append_chat("AI", answer))
        self.root.after(0, lambda: self.status_message.config(text="Ready."))
        # Đọc lại câu trả lời bằng giọng nói
        self.speak_engine.say(answer)
        self.speak_engine.runAndWait()

    def on_closing(self):
        """Handles application closing, ensuring threads are stopped."""
        if self.is_webcam_active:
            self.stop_webcam_event.set()
            # Give thread a moment to terminate gracefully
            if self.webcam_thread and self.webcam_thread.is_alive():
                self.webcam_thread.join(timeout=1.0)
        self.root.destroy()


# Example of running the app
if __name__ == "__main__":
    # Ensure a 'data' directory with subfolders (even empty ones for class detection) exists
    # for the app to initialize correctly.
    # For a real run, populate data/ with actual images.
    if not os.path.exists('./data'):
        os.makedirs('./data/dummy_class_1', exist_ok=True)
        # Add a dummy image so flow_from_directory finds something
        with open('./data/dummy_class_1/dummy.jpg', 'w') as f: f.write('dummy content')
        print("Created dummy 'data' directory for app startup. Please replace with real data.")

    root = tk.Tk()
    app = MaterialRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle window close event
    root.mainloop()

    # Clean up dummy data after app closes (optional)
    if os.path.exists('./data/dummy_class_1/dummy.jpg'):
        os.remove('./data/dummy_class_1/dummy.jpg')
    if os.path.exists('./data/dummy_class_1'):
        os.rmdir('./data/dummy_class_1')
    if os.path.exists('./data') and not os.listdir('./data'): # Only remove if empty
        os.rmdir('./data')