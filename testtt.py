import cv2
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
from datetime import datetime
import numpy as np
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import json
from PIL import Image, ImageTk
import requests
import time
import uuid
from difflib import SequenceMatcher

class LicensePlateAlertSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced License Plate Alert System")
        self.root.geometry("950x850")
        
        # Initialize variables
        self.model = None
        self.ocr = None
        self.cap = None
        self.is_processing = False
        self.video_path = ""
        self.watch_list = set()
        self.alert_contacts = {}
        self.saved_ids = set()
        self.id_to_plate = {}
        self.id_confidence_scores = {}
        self.processing_thread = None
        self.detected_plates = set()
        
        # Model detection mode variables
        self.detection_mode = "auto"  # auto, direct_plate, vehicle_based
        self.model_classes = {}
        self.has_numberplate_class = False
        self.has_vehicle_classes = False
        
        # Email configuration
        self.email_config = {
            'smtp_server': '',
            'smtp_port': 587,
            'email': '',
            'password': ''
        }
        
        # SMS configuration (using TextBelt API - free SMS service)
        self.sms_config = {
            'api_key': 'textbelt',  # Free tier, or get paid API key
            'service': 'textbelt'  # or 'twilio'
        }
        
        # Twilio configuration (optional)
        self.twilio_config = {
            'account_sid': '',
            'auth_token': '',
            'from_phone': ''
        }
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Main Processing
        self.main_frame = ttk.Frame(notebook)
        notebook.add(self.main_frame, text="Main")
        self.setup_main_tab()
        
        # Tab 2: Watch List
        self.watchlist_frame = ttk.Frame(notebook)
        notebook.add(self.watchlist_frame, text="Watch List")
        self.setup_watchlist_tab()
        
        # Tab 3: Alert Settings
        self.settings_frame = ttk.Frame(notebook)
        notebook.add(self.settings_frame, text="Alert Settings")
        self.setup_settings_tab()
        
        # Tab 4: SMS Settings
        self.sms_frame = ttk.Frame(notebook)
        notebook.add(self.sms_frame, text="SMS Settings")
        self.setup_sms_tab()
        
        # Tab 5: Logs
        self.logs_frame = ttk.Frame(notebook)
        notebook.add(self.logs_frame, text="Logs")
        self.setup_logs_tab()
    
    def setup_main_tab(self):
        # Video selection
        video_frame = ttk.LabelFrame(self.main_frame, text="Video Selection")
        video_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(video_frame, text="Select Video File", 
                  command=self.select_video).pack(side='left', padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame, text="No video selected")
        self.video_label.pack(side='left', padx=5, pady=5)
        
        # Model loading with detection mode
        model_frame = ttk.LabelFrame(self.main_frame, text="Model Setup")
        model_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(model_frame, text="Load YOLO Model", 
                  command=self.load_model).pack(side='left', padx=5, pady=5)
        
        self.model_label = ttk.Label(model_frame, text="Model not loaded")
        self.model_label.pack(side='left', padx=5, pady=5)
        
        # Detection mode selection
        mode_frame = ttk.Frame(model_frame)
        mode_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Detection Mode:").pack(side='left', padx=5)
        self.mode_var = tk.StringVar(value="auto")
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, 
                                      values=["auto", "direct_plate", "vehicle_based"], 
                                      state="readonly", width=15)
        self.mode_combo.pack(side='left', padx=5)
        
        # Model info display
        self.model_info_label = ttk.Label(model_frame, text="", foreground="blue")
        self.model_info_label.pack(side='left', padx=10)
        
        # Control buttons
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", 
                                      command=self.start_processing, state='disabled')
        self.start_button.pack(side='left', padx=5, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", 
                                     command=self.stop_processing, state='disabled')
        self.stop_button.pack(side='left', padx=5, pady=5)
        
        self.pause_button = ttk.Button(control_frame, text="Pause", 
                                      command=self.pause_processing, state='disabled')
        self.pause_button.pack(side='left', padx=5, pady=5)
        
        # Progress bar with percentage
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill='x', side='left', expand=True)
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side='right', padx=5)
        
        # Status with more details
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side='left')
        
        self.fps_label = ttk.Label(status_frame, text="")
        self.fps_label.pack(side='right')
        
        # Real-time detections
        detection_frame = ttk.LabelFrame(self.main_frame, text="Real-time Detections")
        detection_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.detection_text = scrolledtext.ScrolledText(detection_frame, height=12)
        self.detection_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_watchlist_tab(self):
        # Add license plate
        add_frame = ttk.LabelFrame(self.watchlist_frame, text="Add License Plate to Watch List")
        add_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(add_frame, text="License Plate:").pack(side='left', padx=5, pady=5)
        self.plate_entry = ttk.Entry(add_frame, width=15)
        self.plate_entry.pack(side='left', padx=5, pady=5)
        
        ttk.Label(add_frame, text="Contact:").pack(side='left', padx=5, pady=5)
        self.contact_entry = ttk.Entry(add_frame, width=20)
        self.contact_entry.pack(side='left', padx=5, pady=5)
        
        ttk.Label(add_frame, text="Type:").pack(side='left', padx=5, pady=5)
        self.contact_type = ttk.Combobox(add_frame, values=["Email", "Phone"], width=10)
        self.contact_type.pack(side='left', padx=5, pady=5)
        self.contact_type.set("Email")
        
        ttk.Button(add_frame, text="Add to Watch List", 
                  command=self.add_to_watchlist).pack(side='left', padx=5, pady=5)
        
        # Watch list display
        list_frame = ttk.LabelFrame(self.watchlist_frame, text="Current Watch List")
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Treeview for watch list
        columns = ('License Plate', 'Contact', 'Type')
        self.watchlist_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.watchlist_tree.heading(col, text=col)
            self.watchlist_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.watchlist_tree.yview)
        self.watchlist_tree.configure(yscrollcommand=scrollbar.set)
        
        self.watchlist_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')
        
        # Remove button
        ttk.Button(list_frame, text="Remove Selected", 
                  command=self.remove_from_watchlist).pack(pady=5)
    
    def setup_settings_tab(self):
        # Email configuration
        email_frame = ttk.LabelFrame(self.settings_frame, text="Email Alert Configuration")
        email_frame.pack(fill='x', padx=5, pady=5)
        
        # SMTP Server
        ttk.Label(email_frame, text="SMTP Server:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.smtp_server_entry = ttk.Entry(email_frame, width=30)
        self.smtp_server_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # SMTP Port
        ttk.Label(email_frame, text="SMTP Port:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.smtp_port_entry = ttk.Entry(email_frame, width=30)
        self.smtp_port_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Email
        ttk.Label(email_frame, text="Your Email:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.email_entry = ttk.Entry(email_frame, width=30)
        self.email_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Password
        ttk.Label(email_frame, text="Password:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.password_entry = ttk.Entry(email_frame, width=30, show='*')
        self.password_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Save settings button
        ttk.Button(email_frame, text="Save Email Settings", 
                  command=self.save_email_settings).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Test email button
        ttk.Button(email_frame, text="Test Email", 
                  command=self.test_email).grid(row=5, column=0, columnspan=2, pady=5)
    
    def setup_sms_tab(self):
        # SMS Service Selection
        service_frame = ttk.LabelFrame(self.sms_frame, text="SMS Service Selection")
        service_frame.pack(fill='x', padx=5, pady=5)
        
        self.sms_service = tk.StringVar(value="textbelt")
        ttk.Radiobutton(service_frame, text="TextBelt (Free - US/Canada)", 
                       variable=self.sms_service, value="textbelt").pack(anchor='w', padx=5, pady=2)
        ttk.Radiobutton(service_frame, text="Twilio (Paid - Global)", 
                       variable=self.sms_service, value="twilio").pack(anchor='w', padx=5, pady=2)
        
        # TextBelt Configuration
        textbelt_frame = ttk.LabelFrame(self.sms_frame, text="TextBelt Configuration")
        textbelt_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(textbelt_frame, text="API Key (optional):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.textbelt_key_entry = ttk.Entry(textbelt_frame, width=40)
        self.textbelt_key_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(textbelt_frame, text="Leave blank to use free tier (1 SMS/day)").grid(row=1, column=0, columnspan=2, padx=5, pady=2)
        
        # Twilio Configuration
        twilio_frame = ttk.LabelFrame(self.sms_frame, text="Twilio Configuration")
        twilio_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(twilio_frame, text="Account SID:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.twilio_sid_entry = ttk.Entry(twilio_frame, width=40)
        self.twilio_sid_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(twilio_frame, text="Auth Token:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.twilio_token_entry = ttk.Entry(twilio_frame, width=40, show='*')
        self.twilio_token_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(twilio_frame, text="From Phone:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.twilio_phone_entry = ttk.Entry(twilio_frame, width=40)
        self.twilio_phone_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Save and test buttons
        button_frame = ttk.Frame(self.sms_frame)
        button_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(button_frame, text="Save SMS Settings", 
                  command=self.save_sms_settings).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Test SMS", 
                  command=self.test_sms).pack(side='left', padx=5)
        
        # Instructions
        instructions = ttk.LabelFrame(self.sms_frame, text="Setup Instructions")
        instructions.pack(fill='both', expand=True, padx=5, pady=5)
        
        instructions_text = """
For TextBelt (Free):
• Works for US and Canada phone numbers
• 1 free SMS per day per phone number
• Use format: +1234567890
• Get paid API key from textbelt.com for unlimited SMS

For Twilio (Paid):
• Global SMS support
• Sign up at twilio.com
• Get Account SID and Auth Token from console
• Purchase a phone number for sending
• Use international format: +1234567890
        """
        
        instructions_label = tk.Label(instructions, text=instructions_text, justify='left', wraplength=400)
        instructions_label.pack(padx=10, pady=10)
    
    def setup_logs_tab(self):
        self.logs_text = scrolledtext.ScrolledText(self.logs_frame)
        self.logs_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        button_frame = ttk.Frame(self.logs_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_frame, text="Clear Logs", 
                  command=self.clear_logs).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Logs", 
                  command=self.save_logs).pack(side='left', padx=5)
    
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.update_start_button_state()
    
    def analyze_model_classes(self):
        """Analyze the loaded model to determine detection capabilities"""
        if not self.model:
            return
        
        self.model_classes = self.model.names
        self.log_message(f"Model classes detected: {list(self.model_classes.values())}")
        
        # Check for direct license plate detection
        plate_classes = ['numberplate', 'license_plate', 'plate', 'number_plate']
        self.has_numberplate_class = any(
            any(plate_name.lower() in class_name.lower() for plate_name in plate_classes)
            for class_name in self.model_classes.values()
        )
        
        # Check for vehicle classes that might contain license plates
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'vehicle']
        self.has_vehicle_classes = any(
            any(vehicle_name.lower() in class_name.lower() for vehicle_name in vehicle_classes)
            for class_name in self.model_classes.values()
        )
        
        # Determine optimal detection mode
        if self.has_numberplate_class:
            suggested_mode = "direct_plate"
            mode_info = "Direct license plate detection available"
        elif self.has_vehicle_classes:
            suggested_mode = "vehicle_based"
            mode_info = "Vehicle-based detection (will crop vehicles and run OCR)"
        else:
            suggested_mode = "auto"
            mode_info = "Auto mode (will try both methods)"
        
        self.model_info_label.config(text=mode_info)
        
        # Auto-set detection mode if in auto mode
        if self.mode_var.get() == "auto":
            self.mode_var.set(suggested_mode)
            self.log_message(f"Auto-selected detection mode: {suggested_mode}")
    
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[("Model files", "*.pt"), ("All files", "*.*")]
        )
        if model_path:
            try:
                self.log_message("Loading YOLO model...")
                self.model = YOLO(model_path)
                self.log_message("Loading PaddleOCR...")
                self.ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang='en'
                )
                
                # Analyze model capabilities
                self.analyze_model_classes()
                
                self.model_label.config(text="Model loaded successfully")
                self.update_start_button_state()
                self.log_message("Model and OCR loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.log_message(f"Error loading model: {str(e)}")
    
    def update_start_button_state(self):
        if self.video_path and self.model and not self.is_processing:
            self.start_button.config(state='normal')
        else:
            self.start_button.config(state='disabled')
    
    def add_to_watchlist(self):
        plate = self.plate_entry.get().strip().upper()
        contact = self.contact_entry.get().strip()
        contact_type = self.contact_type.get()
        
        if not plate or not contact:
            messagebox.showwarning("Warning", "Please enter both license plate and contact information")
            return
        
        # Validate phone number format
        if contact_type == "Phone":
            if not contact.startswith('+'):
                messagebox.showwarning("Warning", "Phone number should start with country code (e.g., +1234567890)")
                return
        
        self.watch_list.add(plate)
        self.alert_contacts[plate] = {'contact': contact, 'type': contact_type}
        
        # Add to treeview
        self.watchlist_tree.insert('', 'end', values=(plate, contact, contact_type))
        
        # Clear entries
        self.plate_entry.delete(0, tk.END)
        self.contact_entry.delete(0, tk.END)
        
        self.log_message(f"Added to watch list: {plate} -> {contact} ({contact_type})")
        self.save_settings()
    
    def remove_from_watchlist(self):
        selected = self.watchlist_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an item to remove")
            return
        
        for item in selected:
            values = self.watchlist_tree.item(item, 'values')
            plate = values[0]
            
            self.watch_list.discard(plate)
            if plate in self.alert_contacts:
                del self.alert_contacts[plate]
            
            self.watchlist_tree.delete(item)
            self.log_message(f"Removed from watch list: {plate}")
        
        self.save_settings()
    
    def save_email_settings(self):
        self.email_config = {
            'smtp_server': self.smtp_server_entry.get(),
            'smtp_port': int(self.smtp_port_entry.get()) if self.smtp_port_entry.get().isdigit() else 587,
            'email': self.email_entry.get(),
            'password': self.password_entry.get()
        }
        self.save_settings()
        messagebox.showinfo("Success", "Email settings saved successfully")
        self.log_message("Email settings saved")
    
    def save_sms_settings(self):
        self.sms_config['service'] = self.sms_service.get()
        self.sms_config['api_key'] = self.textbelt_key_entry.get() or 'textbelt'
        
        self.twilio_config = {
            'account_sid': self.twilio_sid_entry.get(),
            'auth_token': self.twilio_token_entry.get(),
            'from_phone': self.twilio_phone_entry.get()
        }
        
        self.save_settings()
        messagebox.showinfo("Success", "SMS settings saved successfully")
        self.log_message("SMS settings saved")
    
    def test_email(self):
        try:
            self.send_email_alert("TEST123", self.email_config['email'], None, "This is a test email alert")
            messagebox.showinfo("Success", "Test email sent successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send test email: {str(e)}")
    
    def test_sms(self):
        test_phone = tk.simpledialog.askstring("Test SMS", "Enter phone number (with country code, e.g., +1234567890):")
        
        if test_phone:
            try:
                self.send_sms_alert(test_phone, "This is a test SMS alert from License Plate Monitor")
                messagebox.showinfo("Success", "Test SMS sent successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to send test SMS: {str(e)}")
    
    def start_processing(self):
        if not self.watch_list:
            result = messagebox.askyesno("Warning", "No license plates in watch list. Continue anyway?")
            if not result:
                return
        
        # Set detection mode
        self.detection_mode = self.mode_var.get()
        self.log_message(f"Starting processing with detection mode: {self.detection_mode}")
        
        self.is_processing = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.pause_button.config(state='normal')
        
        # Reset tracking variables
        self.saved_ids.clear()
        self.id_to_plate.clear()
        self.id_confidence_scores.clear()
        self.detected_plates.clear()
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        self.is_processing = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.update_ui_after_stop()
        self.log_message("Processing stopped by user")
    
    def pause_processing(self):
        # Toggle pause state
        if hasattr(self, 'paused'):
            self.paused = not self.paused
            self.pause_button.config(text="Resume" if self.paused else "Pause")
        else:
            self.paused = True
            self.pause_button.config(text="Resume")
    
    def update_ui_after_stop(self):
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.pause_button.config(state='disabled')
        self.status_label.config(text="Stopped")
        self.progress.config(value=0)
        self.progress_label.config(text="0%")
    
    def process_video(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise Exception("Could not open video file")
            
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.root.after(0, lambda: self.progress.config(maximum=total_frames))
            
            frame_count = 0
            start_time = time.time()
            
            self.log_message(f"Started processing video: {total_frames} frames at {fps:.2f} FPS")
            self.log_message(f"Detection mode: {self.detection_mode}")
            
            while self.is_processing and frame_count < total_frames:
                # Handle pause
                if hasattr(self, 'paused') and self.paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message("End of video reached or failed to read frame")
                    break
                
                frame_count += 1
                
                # Update progress
                progress_percent = (frame_count / total_frames) * 100
                self.root.after(0, lambda p=frame_count: self.progress.config(value=p))
                self.root.after(0, lambda pp=progress_percent: self.progress_label.config(text=f"{pp:.1f}%"))
                
                # Calculate processing FPS
                elapsed_time = time.time() - start_time
                processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Process every 2nd frame for better performance
                if frame_count % 2 != 0:
                    continue
                
                try:
                    frame = cv2.resize(frame, (1020, 600))
                    results = self.model.track(frame, persist=True, conf=0.5, verbose=False)
                    
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        self.process_detections(results, frame, frame_count, fps)
                    
                    # Update status every 30 frames
                    if frame_count % 30 == 0:
                        status_text = f"Processing frame {frame_count}/{total_frames}"
                        fps_text = f"Speed: {processing_fps:.1f} FPS"
                        self.root.after(0, lambda st=status_text: self.status_label.config(text=st))
                        self.root.after(0, lambda ft=fps_text: self.fps_label.config(text=ft))
                
                except Exception as e:
                    self.log_message(f"Error processing frame {frame_count}: {str(e)}")
                    continue
                    self.log_message(f"Video processing completed. Processed {frame_count} frames in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            self.log_message(f"Error during video processing: {str(e)}")
            messagebox.showerror("Error", f"Error during processing: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.root.after(0, self.update_ui_after_stop)
    
    def process_detections(self, results, frame, frame_count, fps):
        """Process detections based on the selected detection mode"""
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
            class_name = self.model.names[cls]
            
            if self.detection_mode == "direct_plate":
                self.process_direct_plate_detection(box, track_id, conf, class_name, frame, frame_count, fps)
            elif self.detection_mode == "vehicle_based":
                self.process_vehicle_based_detection(box, track_id, conf, class_name, frame, frame_count, fps)
            else:  # auto mode
                self.process_auto_detection(box, track_id, conf, class_name, frame, frame_count, fps)
    
    def process_direct_plate_detection(self, box, track_id, conf, class_name, frame, frame_count, fps):
        """Process direct license plate detections"""
        plate_classes = ['numberplate', 'license_plate', 'plate', 'number_plate']
        
        if any(plate_name.lower() in class_name.lower() for plate_name in plate_classes):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            
            # Crop the license plate region
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size > 0:
                license_plate_text = self.extract_text_from_crop(plate_crop)
                if license_plate_text:
                    self.handle_detected_plate(track_id, license_plate_text, conf, frame_count, fps, plate_crop)
    
    def process_vehicle_based_detection(self, box, track_id, conf, class_name, frame, frame_count, fps):
        """Process vehicle detections and extract license plates from vehicles"""
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'vehicle']
        
        if any(vehicle_name.lower() in class_name.lower() for vehicle_name in vehicle_classes):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            
            # Crop the vehicle region
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.size > 0:
                # Focus on bottom third of vehicle where license plates usually are
                height = vehicle_crop.shape[0]
                bottom_third = vehicle_crop[int(height * 0.66):, :]
                
                license_plate_text = self.extract_text_from_crop(bottom_third)
                if license_plate_text:
                    self.handle_detected_plate(track_id, license_plate_text, conf, frame_count, fps, bottom_third)
    
    def process_auto_detection(self, box, track_id, conf, class_name, frame, frame_count, fps):
        """Auto detection mode - try both direct plate and vehicle-based detection"""
        # First try direct plate detection
        plate_classes = ['numberplate', 'license_plate', 'plate', 'number_plate']
        if any(plate_name.lower() in class_name.lower() for plate_name in plate_classes):
            self.process_direct_plate_detection(box, track_id, conf, class_name, frame, frame_count, fps)
        else:
            # If not a direct plate, try vehicle-based detection
            self.process_vehicle_based_detection(box, track_id, conf, class_name, frame, frame_count, fps)
    
    def extract_text_from_crop(self, crop):
        """Extract text from cropped image using OCR"""
        try:
            # Preprocess the crop for better OCR
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get better contrast
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run OCR
            ocr_result = self.ocr.ocr(thresh, cls=False)
            
            if ocr_result and ocr_result[0]:
                # Extract text and clean it
                text_list = []
                for detection in ocr_result[0]:
                    text = detection[1][0]
                    confidence = detection[1][1]
                    
                    if confidence > 0.5:  # Only consider high confidence text
                        # Clean the text
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if len(cleaned_text) >= 3:  # Minimum 3 characters for a valid plate
                            text_list.append(cleaned_text)
                
                if text_list:
                    # Join all detected text pieces
                    full_text = ''.join(text_list)
                    return full_text if len(full_text) >= 3 else None
            
        except Exception as e:
            self.log_message(f"OCR error: {str(e)}")
        
        return None
    
    def handle_detected_plate(self, track_id, license_plate_text, conf, frame_count, fps, plate_crop):
        """Handle a detected license plate"""
        # Skip if we've already processed this track ID
        if track_id in self.saved_ids:
            return
        
        # Update confidence tracking
        if track_id not in self.id_confidence_scores:
            self.id_confidence_scores[track_id] = []
        
        self.id_confidence_scores[track_id].append((license_plate_text, conf))
        
        # Only process after we have multiple detections for stability
        if len(self.id_confidence_scores[track_id]) >= 3:
            # Find the most consistent license plate text
            best_plate = self.get_most_consistent_plate(track_id)
            
            if best_plate and best_plate not in self.detected_plates:
                self.detected_plates.add(best_plate)
                self.saved_ids.add(track_id)
                self.id_to_plate[track_id] = best_plate
                
                # Log detection
                timestamp = frame_count / fps if fps > 0 else 0
                detection_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Detected plate: {best_plate} (Track ID: {track_id}, Time: {timestamp:.1f}s)"
                
                self.log_message(detection_msg)
                self.root.after(0, lambda msg=detection_msg: self.update_detection_display(msg))
                
                # Save image
                self.save_detection_image(plate_crop, best_plate, frame_count)
                
                # Check if it's in watch list
                if best_plate in self.watch_list:
                    self.trigger_alert(best_plate, timestamp, plate_crop)
    
    def get_most_consistent_plate(self, track_id):
        """Get the most consistent license plate text for a track ID"""
        detections = self.id_confidence_scores[track_id]
        
        # Group similar texts
        text_groups = {}
        for text, conf in detections:
            # Find best matching group
            best_match = None
            best_similarity = 0
            
            for existing_text in text_groups:
                similarity = SequenceMatcher(None, text, existing_text).ratio()
                if similarity > best_similarity and similarity > 0.8:
                    best_similarity = similarity
                    best_match = existing_text
            
            if best_match:
                text_groups[best_match].append((text, conf))
            else:
                text_groups[text] = [(text, conf)]
        
        # Find the group with highest average confidence
        best_text = None
        best_avg_conf = 0
        
        for text, detections_list in text_groups.items():
            avg_conf = sum(conf for _, conf in detections_list) / len(detections_list)
            if avg_conf > best_avg_conf and len(detections_list) >= 2:
                best_avg_conf = avg_conf
                best_text = text
        
        return best_text
    
    def update_detection_display(self, message):
        """Update the real-time detection display"""
        self.detection_text.insert(tk.END, message + "\n")
        self.detection_text.see(tk.END)
    
    def save_detection_image(self, plate_crop, license_plate_text, frame_count):
        """Save the detected license plate image"""
        try:
            # Create detections directory if it doesn't exist
            detections_dir = "detections"
            os.makedirs(detections_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{detections_dir}/plate_{license_plate_text}_{timestamp}_{frame_count}.jpg"
            
            cv2.imwrite(filename, plate_crop)
            self.log_message(f"Saved detection image: {filename}")
            
        except Exception as e:
            self.log_message(f"Error saving detection image: {str(e)}")
    
    def trigger_alert(self, license_plate, timestamp, plate_crop):
        """Trigger alert for detected watch list plate"""
        self.log_message(f"ALERT: Watch list plate detected: {license_plate}")
        
        alert_info = self.alert_contacts.get(license_plate, {})
        contact = alert_info.get('contact', '')
        contact_type = alert_info.get('type', 'Email')
        
        if contact:
            try:
                if contact_type == 'Email':
                    self.send_email_alert(license_plate, contact, plate_crop, f"Detected at {timestamp:.1f}s in video")
                elif contact_type == 'Phone':
                    message = f"ALERT: License plate {license_plate} detected at {timestamp:.1f}s in monitored video."
                    self.send_sms_alert(contact, message)
                
                self.log_message(f"Alert sent to {contact} ({contact_type})")
                
            except Exception as e:
                self.log_message(f"Failed to send alert: {str(e)}")
        else:
            self.log_message(f"No contact information found for {license_plate}")
    
    def send_email_alert(self, license_plate, recipient, plate_image, additional_info=""):
        """Send email alert"""
        if not all([self.email_config['smtp_server'], self.email_config['email'], self.email_config['password']]):
            raise Exception("Email configuration incomplete")
        
        msg = MIMEMultipart()
        msg['From'] = self.email_config['email']
        msg['To'] = recipient
        msg['Subject'] = f"License Plate Alert: {license_plate}"
        
        body = f"""
        License Plate Alert System
        
        A monitored license plate has been detected:
        
        License Plate: {license_plate}
        Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        {additional_info}
        
        This is an automated alert from your License Plate Monitoring System.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach image if provided
        if plate_image is not None:
            try:
                # Convert OpenCV image to bytes
                _, buffer = cv2.imencode('.jpg', plate_image)
                img_bytes = buffer.tobytes()
                
                img_attachment = MIMEImage(img_bytes)
                img_attachment.add_header('Content-Disposition', f'attachment; filename=plate_{license_plate}.jpg')
                msg.attach(img_attachment)
            except Exception as e:
                self.log_message(f"Failed to attach image: {str(e)}")
        
        # Send email
        server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
        server.starttls()
        server.login(self.email_config['email'], self.email_config['password'])
        text = msg.as_string()
        server.sendmail(self.email_config['email'], recipient, text)
        server.quit()
    
    def send_sms_alert(self, phone_number, message):
        """Send SMS alert using selected service"""
        if self.sms_config['service'] == 'textbelt':
            self.send_textbelt_sms(phone_number, message)
        elif self.sms_config['service'] == 'twilio':
            self.send_twilio_sms(phone_number, message)
    
    def send_textbelt_sms(self, phone_number, message):
        """Send SMS using TextBelt API"""
        url = 'https://textbelt.com/text'
        
        data = {
            'phone': phone_number,
            'message': message,
            'key': self.sms_config['api_key']
        }
        
        response = requests.post(url, data=data)
        result = response.json()
        
        if not result.get('success'):
            raise Exception(f"TextBelt SMS failed: {result.get('error', 'Unknown error')}")
    
    def send_twilio_sms(self, phone_number, message):
        """Send SMS using Twilio API"""
        try:
            from twilio.rest import Client
            
            client = Client(self.twilio_config['account_sid'], self.twilio_config['auth_token'])
            
            message = client.messages.create(
                body=message,
                from_=self.twilio_config['from_phone'],
                to=phone_number
            )
            
        except ImportError:
            raise Exception("Twilio library not installed. Run: pip install twilio")
        except Exception as e:
            raise Exception(f"Twilio SMS failed: {str(e)}")
    
    def save_settings(self):
        """Save all settings to file"""
        settings = {
            'watch_list': list(self.watch_list),
            'alert_contacts': self.alert_contacts,
            'email_config': self.email_config,
            'sms_config': self.sms_config,
            'twilio_config': self.twilio_config
        }
        
        try:
            with open('license_plate_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            self.log_message(f"Error saving settings: {str(e)}")
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists('license_plate_settings.json'):
                with open('license_plate_settings.json', 'r') as f:
                    settings = json.load(f)
                
                self.watch_list = set(settings.get('watch_list', []))
                self.alert_contacts = settings.get('alert_contacts', {})
                self.email_config.update(settings.get('email_config', {}))
                self.sms_config.update(settings.get('sms_config', {}))
                self.twilio_config.update(settings.get('twilio_config', {}))
                
                # Update UI elements
                self.load_ui_settings()
                
                self.log_message("Settings loaded successfully")
        except Exception as e:
            self.log_message(f"Error loading settings: {str(e)}")
    
    def load_ui_settings(self):
        """Load settings into UI elements"""
        # Load email settings
        self.smtp_server_entry.insert(0, self.email_config.get('smtp_server', ''))
        self.smtp_port_entry.insert(0, str(self.email_config.get('smtp_port', 587)))
        self.email_entry.insert(0, self.email_config.get('email', ''))
        self.password_entry.insert(0, self.email_config.get('password', ''))
        
        # Load SMS settings
        self.sms_service.set(self.sms_config.get('service', 'textbelt'))
        self.textbelt_key_entry.insert(0, self.sms_config.get('api_key', 'textbelt'))
        
        self.twilio_sid_entry.insert(0, self.twilio_config.get('account_sid', ''))
        self.twilio_token_entry.insert(0, self.twilio_config.get('auth_token', ''))
        self.twilio_phone_entry.insert(0, self.twilio_config.get('from_phone', ''))
        
        # Load watch list
        for plate, contact_info in self.alert_contacts.items():
            self.watchlist_tree.insert('', 'end', values=(
                plate, 
                contact_info['contact'], 
                contact_info['type']
            ))
    
    def log_message(self, message):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.root.after(0, lambda: self.logs_text.insert(tk.END, log_entry + "\n"))
        self.root.after(0, lambda: self.logs_text.see(tk.END))
    
    def clear_logs(self):
        """Clear the logs display"""
        self.logs_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to file"""
        try:
            logs_content = self.logs_text.get(1.0, tk.END)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"license_plate_logs_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(logs_content)
            
            messagebox.showinfo("Success", f"Logs saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = LicensePlateAlertSystem(root)
    
    # Handle window closing
    def on_closing():
        if app.is_processing:
            if messagebox.askokcancel("Quit", "Processing is running. Do you want to quit?"):
                app.stop_processing()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()