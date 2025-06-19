import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import joblib
from DataProcess import clean_text
import threading

class SpamClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.load_model()
        
    def setup_window(self):
        """Setup basic window properties"""
        self.root.title("🛡️ Smart Spam Email Detector")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"800x700+{x}+{y}")
        
        # Set background color
        self.root.configure(bg='#f0f0f0')
        
    def setup_styles(self):
        """Setup style themes"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom styles
        self.style.configure('Title.TLabel', 
                           font=('Arial', 20, 'bold'),
                           background='#f0f0f0',
                           foreground='#2c3e50')
        
        self.style.configure('Subtitle.TLabel',
                           font=('Arial', 12),
                           background='#f0f0f0',
                           foreground='#7f8c8d')
        
        self.style.configure('Result.TLabel',
                           font=('Arial', 14, 'bold'),
                           background='#f0f0f0')
        
        self.style.configure('Custom.TButton',
                           font=('Arial', 12, 'bold'),
                           padding=10)
        
    def create_widgets(self):
        """Create all UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title area
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        title_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(title_frame, text="🛡️ Smart Spam Email Detector", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 5))
        
        subtitle_label = ttk.Label(title_frame, 
                                 text="Using machine learning technology to identify spam emails and protect your inbox",
                                 style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0)
        
        # Input area
        input_frame = ttk.LabelFrame(main_frame, text="📨 Enter Email Content", padding="15")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)
        
        # Input hint
        input_hint = ttk.Label(input_frame, text="Please enter the email content to be detected below:")
        input_hint.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        
        # Text input box
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            height=8,
            font=('Consolas', 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=2,
            highlightthickness=1,
            highlightcolor='#3498db'
        )
        self.text_input.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Button area
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=15)
        
        # Detect button
        self.detect_button = ttk.Button(
            button_frame,
            text="🔍 Start Detection",
            command=self.start_detection,
            style='Custom.TButton'
        )
        self.detect_button.grid(row=0, column=0, padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(
            button_frame,
            text="🗑️ Clear Content",
            command=self.clear_text,
            style='Custom.TButton'
        )
        clear_button.grid(row=0, column=1, padx=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.grid(row=3, column=0, pady=(0, 15), sticky=(tk.W, tk.E))
        
        # Result display area
        result_frame = ttk.LabelFrame(main_frame, text="📊 Detection Results", padding="15")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        result_frame.columnconfigure(0, weight=1)
        
        # Result display
        self.result_text = tk.Text(
            result_frame,
            height=6,
            font=('Arial', 12),
            relief=tk.FLAT,
            borderwidth=2,
            highlightthickness=1,
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Configure text color tags
        self.result_text.tag_configure("spam", foreground="#e74c3c", font=('Arial', 14, 'bold'))
        self.result_text.tag_configure("ham", foreground="#27ae60", font=('Arial', 14, 'bold'))
        self.result_text.tag_configure("confidence", foreground="#b0dfff", font=('Arial', 12, 'bold'))
        self.result_text.tag_configure("normal", foreground="#ffffff", font=('Arial', 11))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please enter email content for detection")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure main frame grid weights
        main_frame.rowconfigure(1, weight=1)
        
    def load_model(self):
        """Load machine learning model"""
        try:
            self.model_data = joblib.load("ensemble_spam_classifier.pkl")
            self.model = self.model_data["model"]
            self.vectorizer = self.model_data["vectorizer"]
            self.status_var.set("Model loaded successfully - System is ready")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load model file: {str(e)}")
            self.status_var.set("Model loading failed")
            
    def predict_email_with_confidence(self, text):
        """Predict email type and confidence"""
        try:
            cleaned = clean_text(text)
            X_input = self.vectorizer.transform([cleaned])
            
            # Get probability distribution
            proba = self.model.predict_proba(X_input)[0]
            prediction = self.model.predict(X_input)[0]
            
            label = 'spam' if prediction == 1 else 'ham'
            confidence = round(proba[prediction] * 100, 2)
            
            # Modified judgment logic: If original prediction is ham but confidence < 70%, change to spam
            if label == 'ham' and confidence < 70:
                label = 'spam'
                # Note: Keep original confidence, representing confidence in ham prediction
            
            return label, confidence
        except Exception as e:
            raise Exception(f"Error occurred during prediction: {str(e)}")
    
    def start_detection(self):
        """Start detection (run in new thread)"""
        email_text = self.text_input.get("1.0", tk.END).strip()
        
        if not email_text:
            messagebox.showwarning("Warning", "Please enter email content first!")
            return
        
        # Disable button, show progress bar
        self.detect_button.config(state='disabled')
        self.progress.start()
        self.status_var.set("Analyzing email content...")
        
        # Run detection in new thread
        threading.Thread(target=self.detect_email, args=(email_text,), daemon=True).start()
    
    def detect_email(self, email_text):
        """Execute email detection"""
        try:
            result, confidence = self.predict_email_with_confidence(email_text)
            
            # Update UI in main thread
            self.root.after(0, self.show_result, result, confidence)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def show_result(self, result, confidence):
        """Display detection results"""
        # Stop progress bar, restore button
        self.progress.stop()
        self.detect_button.config(state='normal')
        
        # Clear previous results
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        
        # Display results
        if result == 'spam':
            emoji = "⚠️"
            result_text = "SPAM EMAIL"
            tag = "spam"
            advice = "Recommendation: This email may contain spam content, please handle with caution!"
        else:
            emoji = "✅"
            result_text = "LEGITIMATE EMAIL"
            tag = "ham"
            advice = "Recommendation: This email appears to be legitimate and safe to read."
        
        # Insert formatted results
        self.result_text.insert(tk.END, f"{emoji} Detection Result: ", "normal")
        self.result_text.insert(tk.END, f"{result_text}\n", tag)
        self.result_text.insert(tk.END, f"🎯 Confidence: ", "normal")
        self.result_text.insert(tk.END, f"{confidence}%\n\n", "confidence")
        self.result_text.insert(tk.END, f"{advice}\n", "normal")
        
        # Add technical details
        self.result_text.insert(tk.END, f"\n📋 Technical Details:\n", "normal")
        self.result_text.insert(tk.END, f"• Classification Algorithm: Ensemble Learning Model\n", "normal")
        self.result_text.insert(tk.END, f"• Predicted Category: {result} ({1 if result == 'spam' else 0})\n", "normal")
        self.result_text.insert(tk.END, f"• Confidence Score: {confidence}%", "normal")
        
        self.result_text.config(state=tk.DISABLED)
        
        # Update status bar
        self.status_var.set(f"Detection completed - {result_text} (Confidence: {confidence}%)")
    
    def show_error(self, error_message):
        """Display error message"""
        self.progress.stop()
        self.detect_button.config(state='normal')
        messagebox.showerror("Detection Error", error_message)
        self.status_var.set("Detection failed")
    
    def clear_text(self):
        """Clear input box"""
        self.text_input.delete("1.0", tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.status_var.set("Content cleared - Please enter new email content")

def main():
    """Main function"""
    root = tk.Tk()
    app = SpamClassifierGUI(root)
    
    # Set window icon (if available)
    try:
        # root.iconbitmap('icon.ico')  # Uncomment if you have an icon file
        pass
    except:
        pass
    
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main()