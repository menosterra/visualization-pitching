import os
import sys
import json
import cv2
import math
import subprocess
import threading
import urllib.request
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Joint connections for Skeleton rendering (same as render_raw_json_overlay.py)
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Torso
    (11, 13), (13, 15),                      # Left arm
    (12, 14), (14, 16),                      # Right arm
    (23, 25), (25, 27),                      # Left leg
    (24, 26), (26, 28)                       # Right leg
]
DRAWABLE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COLOR_PINK_BGR = (153, 72, 236)  # Pink in BGR format
COLOR_PINK_RGB = (236, 72, 153)  # Pink in RGB format

def download_model(model_name, model_path):
    if not os.path.exists(model_path):
        if "heavy" in model_name.lower():
            print("Downloading MediaPipe Pose Landmarker Heavy model (~30MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        else:
            print("Downloading MediaPipe Pose Landmarker Full model (~26MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise e

def format_seconds(seconds):
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"

import re

def get_unique_path(target_path):
    """
    동일한 파일명이나 폴더명이 이미 존재할 경우,
    -1, -2, -3... 접미사를 붙여 고유한 경로를 반환합니다.
    """
    if not os.path.exists(target_path):
        return target_path
    
    dir_name, base_name = os.path.split(target_path)
    name, ext = os.path.splitext(base_name)
    
    # 이미 -숫자 형식의 접미사가 붙어있는지 확인 (예: video_clip-1)
    match = re.search(r'^(.*?)-(\d+)$', name)
    if match:
        base_prefix = match.group(1)
        count = int(match.group(2)) + 1
    else:
        base_prefix = name
        count = 1
        
    while True:
        new_name = f"{base_prefix}-{count}{ext}"
        new_path = os.path.join(dir_name, new_name)
        if not os.path.exists(new_path):
            return new_path
        count += 1

class PoseEstimatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Pose Estimator & Overlay Generator")
        self.root.geometry("1180x680")
        self.root.resizable(False, False)
        
        # State variables
        self.input_path = ""
        self.output_json_path = ""
        self.output_video_path = ""
        self.output_web_video_path = ""
        
        self.video_duration = 0.0
        self.cap = None
        self.total_frames = 0
        self.fps = 0.0
        self.current_frame_idx = 0
        self.playing = False
        self.play_timer = None
        
        # UI Styling (Clam theme to match video_clipping.py)
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.default_font = ("맑은 고딕", 9)
        self.root.option_add("*Font", self.default_font)
        
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel (Generated Video Preview & Playback controls)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        preview_lf = ttk.LabelFrame(left_panel, text=" 생성된 Overlay 동영상 확인 ", padding="10")
        preview_lf.pack(fill=tk.BOTH, expand=True)
        
        # Preview canvas (640x360 large preview)
        preview_bg = tk.Frame(preview_lf, bg="black", width=640, height=360)
        preview_bg.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        preview_bg.pack_propagate(False)
        
        self.lbl_preview = ttk.Label(preview_bg, text="분석이 완료되면 여기에 스켈레톤 영상이 표시됩니다.", foreground="white", background="black", anchor=tk.CENTER)
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)
        
        # Play scrubber scale
        self.scale_play = ttk.Scale(preview_lf, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.scale_play.pack(fill=tk.X, pady=(0, 10))
        self.scale_play.config(state=tk.DISABLED)
        
        # Playback control buttons
        ctrl_frame = ttk.Frame(preview_lf)
        ctrl_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_prev = ttk.Button(ctrl_frame, text="◀ 1프레임", command=self.prev_frame, state=tk.DISABLED, width=11)
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 6))
        
        self.btn_play = ttk.Button(ctrl_frame, text="▶ 재생", command=self.toggle_play, state=tk.DISABLED, width=11)
        self.btn_play.pack(side=tk.LEFT, padx=(0, 6))
        
        self.btn_next = ttk.Button(ctrl_frame, text="1프레임 ▶", command=self.next_frame, state=tk.DISABLED, width=11)
        self.btn_next.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_open_verify = ttk.Button(ctrl_frame, text="📁 확인용 파일 선택", command=self.open_verify_video, width=16)
        self.btn_open_verify.pack(side=tk.LEFT)
        
        # Time / Frame Info
        self.lbl_time_info = ttk.Label(preview_lf, text="프레임: - / -  |  시간: --:-- / --:--", font=("맑은 고딕", 9, "bold"))
        self.lbl_time_info.pack(side=tk.RIGHT, pady=5)
        
        # Right Panel (File Select, Output paths, Estimator process controls)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 1. 원본 동영상 파일 선택
        file_lf = ttk.LabelFrame(right_panel, text=" 1. 원본 동영상 파일 선택 ", padding="10")
        file_lf.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_file_path = ttk.Label(file_lf, text="선택된 파일 없음", width=38, anchor=tk.W, foreground="gray")
        self.lbl_file_path.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        
        btn_select = ttk.Button(file_lf, text="파일 선택", command=self.select_file)
        btn_select.pack(side=tk.RIGHT)
        
        # 2. 동영상 정보 표시 영역
        self.info_lf = ttk.LabelFrame(right_panel, text=" 2. 동영상 정보 ", padding="10")
        self.info_lf.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_info = ttk.Label(self.info_lf, text="파일을 선택하면 정보가 표시됩니다.", anchor=tk.CENTER, foreground="gray")
        self.lbl_info.pack(fill=tk.X)
        
        # 3. Pose 모델 설정 영역
        model_lf = ttk.LabelFrame(right_panel, text=" 3. Pose 모델 설정 ", padding="10")
        model_lf.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_lf, text="모델 복잡도:").pack(side=tk.LEFT, padx=(5, 10))
        self.combo_model = ttk.Combobox(model_lf, values=["Full (일반)", "Heavy (정밀)"], state="readonly", width=15)
        self.combo_model.current(0)  # Default to Full
        self.combo_model.pack(side=tk.LEFT)

        # 4. 출력 경로 설정 영역
        output_lf = ttk.LabelFrame(right_panel, text=" 4. 생성 경로 정보 ", padding="10")
        output_lf.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_lf, text="Pose JSON 경로:").grid(row=0, column=0, sticky=tk.E, padx=(5, 5))
        self.lbl_json_path = ttk.Label(output_lf, text="자동 지정 대기 중", foreground="gray", width=42, anchor=tk.W)
        self.lbl_json_path.grid(row=0, column=1, sticky=tk.W, pady=3)
        
        ttk.Label(output_lf, text="Overlay 영상 경로:").grid(row=1, column=0, sticky=tk.E, padx=(5, 5))
        self.lbl_video_path = ttk.Label(output_lf, text="자동 지정 대기 중", foreground="gray", width=42, anchor=tk.W)
        self.lbl_video_path.grid(row=1, column=1, sticky=tk.W, pady=3)
        
        # 5. 실행 및 진행 상태 영역
        action_frame = ttk.Frame(right_panel)
        action_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.progress = ttk.Progressbar(action_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_status = ttk.Label(action_frame, text="대기 중...", font=("맑은 고딕", 10, "bold"), foreground="dimgray")
        self.lbl_status.pack(side=tk.LEFT, padx=5)
        
        self.btn_process = ttk.Button(action_frame, text="추출 및 Overlay 생성 시작", command=self.start_processing, state=tk.DISABLED)
        self.btn_process.pack(side=tk.RIGHT, ipadx=10, ipady=5)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="동영상 파일 선택",
            filetypes=[("동영상 파일", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm")]
        )
        if not file_path:
            return
            
        self.input_path = os.path.abspath(file_path)
        self.lbl_file_path.config(text=self.input_path, foreground="black")
        
        # Parse output filenames with duplicate resolution
        dir_name, file_name = os.path.split(self.input_path)
        base_name, ext = os.path.splitext(file_name)
        
        out_dir = os.path.join(dir_name)
        
        raw_json_path = os.path.join(out_dir, f"{base_name}_pose.json")
        raw_video_path = os.path.join(out_dir, f"{base_name}_overlay.mp4")
        
        self.output_json_path = get_unique_path(raw_json_path)
        self.output_video_path = get_unique_path(raw_video_path)
        self.temp_video_path = os.path.join(out_dir, f"{base_name}_temp_overlay.mp4")
        
        # Update output paths labels
        self.lbl_json_path.config(text=os.path.basename(self.output_json_path), foreground="black")
        self.lbl_video_path.config(text=os.path.basename(self.output_video_path), foreground="black")
        
        # Check source video details
        try:
            self.lbl_status.config(text="동영상 정보 읽는 중...", foreground="blue")
            self.root.update()
            
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise Exception("비디오 파일을 열 수 없습니다.")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            duration = total_frames / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            info_text = f"길이: {format_seconds(duration)} ({duration:.2f}초)  |  해상도: {width}x{height}  |  FPS: {fps:.2f}"
            self.lbl_info.config(text=info_text, foreground="black")
            
            self.btn_process.config(state=tk.NORMAL)
            self.lbl_status.config(text="준비 완료", foreground="green")
            
        except Exception as e:
            messagebox.showerror("오류", f"동영상 정보를 읽는데 실패했습니다:\n{str(e)}")
            self.lbl_status.config(text="파일 로드 실패", foreground="red")
            self.btn_process.config(state=tk.DISABLED)
            self.lbl_info.config(text="파일을 선택하면 정보가 표시됩니다.", foreground="gray")
            self.lbl_file_path.config(text="선택된 파일 없음", foreground="gray")

    def start_processing(self):
        if not self.input_path:
            return
            
        # Disable GUI controls
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Re-verify unique output paths
        self.output_json_path = get_unique_path(self.output_json_path)
        self.output_video_path = get_unique_path(self.output_video_path)
        self.lbl_json_path.config(text=os.path.basename(self.output_json_path), foreground="black")
        self.lbl_video_path.config(text=os.path.basename(self.output_video_path), foreground="black")
            
        self.btn_process.config(state=tk.DISABLED)
        self.lbl_status.config(text="모델 다운로드 및 초기화 중...", foreground="blue")
        self.progress['value'] = 0
        
        # Run background thread
        worker = threading.Thread(target=self.process_worker, daemon=True)
        worker.start()

    def process_worker(self):
        try:
            # 1. Get Selected Model option
            selected_model = self.combo_model.get()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            if "Heavy" in selected_model:
                model_name = "heavy"
                model_filename = "pose_landmarker_heavy.task"
            else:
                model_name = "full"
                model_filename = "pose_landmarker_full.task"
                
            model_path = os.path.join(base_dir, model_filename)
            
            self.update_status_safe(f"모델 구성 확인 중 ({model_name})...", 2)
            download_model(model_name, model_path)
            
            # 2. Extract Pose using MediaPipe
            self.update_status_safe("Pose 분석 시작... (MediaPipe)", 5)
            pose_data = self.run_pose_extraction(self.input_path, model_path)
            if not pose_data:
                raise Exception("MediaPipe 포즈 추출 실패.")
                
            # Save pose json
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(pose_data, f, indent=2)
                
            # 3. Render Skeleton Overlay Video (Render to temp video first)
            self.update_status_safe("Overlay 스켈레톤 영상 그리는 중...", 60)
            self.run_skeleton_rendering(self.input_path, self.output_json_path, self.temp_video_path, pose_data)
            
            # 4. Transcode to H.264 using ffmpeg for web/preview compat
            self.update_status_safe("비디오 코덱 변환 중 (FFmpeg)...", 85)
            transcode_success = self.run_ffmpeg_transcoding(self.temp_video_path, self.output_video_path)
            
            if transcode_success:
                try:
                    if os.path.exists(self.temp_video_path):
                        os.remove(self.temp_video_path)
                except Exception:
                    pass
            else:
                # If transcoding failed, rename temp output to final overlay filename
                try:
                    if os.path.exists(self.output_video_path):
                        os.remove(self.output_video_path)
                    os.rename(self.temp_video_path, self.output_video_path)
                except Exception:
                    pass
            
            self.root.after(0, self.on_success, self.output_video_path)
        except Exception as e:
            self.root.after(0, self.on_failure, str(e))

    def update_status_safe(self, text, progress_val):
        self.root.after(0, lambda: self.lbl_status.config(text=text, foreground="darkorange"))
        self.root.after(0, lambda: self.progress.config(value=progress_val))

    def run_pose_extraction(self, video_path, model_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dt = 1.0 / fps
        
        # Use RunningMode.IMAGE mode instead of RunningMode.VIDEO for fast movement accuracy
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=False
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        
        frames_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Using detect() for IMAGE mode instead of detect_for_video()
            detection_result = detector.detect(mp_image)
            
            frame_record = {
                "frame": frame_idx,
                "time": frame_idx * dt,
                "detected": False,
                "landmarks2d": [],
                "landmarks3d": []
            }
            
            if detection_result.pose_landmarks and detection_result.pose_world_landmarks:
                frame_record["detected"] = True
                lms2d = detection_result.pose_landmarks[0]
                lms3d = detection_result.pose_world_landmarks[0]
                
                for lm in lms2d:
                    frame_record["landmarks2d"].append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility)
                    })
                for lm in lms3d:
                    frame_record["landmarks3d"].append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility)
                    })
            else:
                if frames_data:
                    last_detected = next((f for f in reversed(frames_data) if f["detected"]), None)
                    if last_detected:
                        frame_record["landmarks2d"] = last_detected["landmarks2d"]
                        frame_record["landmarks3d"] = last_detected["landmarks3d"]
                        
            frames_data.append(frame_record)
            frame_idx += 1
            
            # Update progress value dynamically
            if frame_idx % 15 == 0 or frame_idx == total_frames:
                percent = int((frame_idx / total_frames) * 50) + 5
                self.update_status_safe(f"Pose 추출 중... ({frame_idx}/{total_frames})", percent)
                
        cap.release()
        detector.close()
        
        return {
            "video_path": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "frames": frames_data
        }

    def run_skeleton_rendering(self, video_path, json_path, output_path, pose_data):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dt = 1.0 / fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_list = pose_data["frames"]
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx < len(frames_list):
                f_record = frames_list[frame_idx]
                if f_record["detected"] and f_record["landmarks2d"]:
                    landmarks = f_record["landmarks2d"]
                    
                    # Draw bones
                    for conn in POSE_CONNECTIONS:
                        start_lm = landmarks[conn[0]]
                        end_lm = landmarks[conn[1]]
                        if start_lm["visibility"] > 0.5 and end_lm["visibility"] > 0.5:
                            pt_start = (int(start_lm["x"] * width), int(start_lm["y"] * height))
                            pt_end = (int(end_lm["x"] * width), int(end_lm["y"] * height))
                            cv2.line(frame, pt_start, pt_end, COLOR_PINK_BGR, 3)
                            
                    # Draw joints
                    for idx in DRAWABLE_JOINTS:
                        lm = landmarks[idx]
                        if lm["visibility"] > 0.5:
                            pt = (int(lm["x"] * width), int(lm["y"] * height))
                            cv2.circle(frame, pt, 4, COLOR_PINK_BGR, -1)
                            cv2.circle(frame, pt, 6, COLOR_PINK_BGR, 1)
                            
                    # Head and neck
                    n11 = landmarks[11]
                    n12 = landmarks[12]
                    if n11["visibility"] > 0.5 and n12["visibility"] > 0.5:
                        sh_c_x = (n11["x"] + n12["x"]) / 2 * width
                        sh_c_y = (n11["y"] + n12["y"]) / 2 * height
                        pt_shoulder_c = (int(sh_c_x), int(sh_c_y))
                        
                        visible_pts = []
                        for idx in [0, 7, 8]:
                            lm = landmarks[idx]
                            if lm["visibility"] > 0.5:
                                visible_pts.append((lm["x"], lm["y"]))
                                
                        if visible_pts:
                            head_x = sum(p[0] for p in visible_pts) / len(visible_pts) * width
                            head_y = sum(p[1] for p in visible_pts) / len(visible_pts) * height
                            pt_head = (int(head_x), int(head_y))
                            
                            cv2.line(frame, pt_shoulder_c, pt_head, COLOR_PINK_BGR, 3)
                            cv2.circle(frame, pt_head, 7, COLOR_PINK_BGR, -1)
                            
            # Add HUD text
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            cv2.putText(frame, f"User Pitching ({video_basename} Overlay)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PINK_BGR, 2)
            cv2.putText(frame, f"Time: {frame_idx * dt:.2f}s", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 15 == 0 or frame_idx == total_frames:
                percent = int((frame_idx / total_frames) * 25) + 55
                self.update_status_safe(f"오버레이 그리는 중... ({frame_idx}/{total_frames})", percent)
                
        cap.release()
        out.release()

    def run_ffmpeg_transcoding(self, input_video, output_video):
        try:
            transcode_cmd = [
                "ffmpeg", "-y", 
                "-i", input_video, 
                "-vcodec", "libx264", 
                "-g", "1", 
                "-pix_fmt", "yuv420p", 
                output_video
            ]
            subprocess.run(
                transcode_cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE, 
                encoding='utf-8', 
                errors='ignore', 
                check=True
            )
            return True
        except Exception as e:
            print(f"FFmpeg transcoding failed (using original file instead): {e}")
            return False

    def on_success(self, target_video):
        self.progress.config(value=100)
        self.lbl_status.config(text="분석 및 영상 생성 완료!", foreground="green")
        self.btn_process.config(state=tk.NORMAL)
        
        # Load generated overlay video into player
        try:
            self.load_video(target_video)
        except Exception as e:
            messagebox.showerror("오류", f"생성된 비디오를 로드하는 데 실패했습니다:\n{str(e)}")
            
        messagebox.showinfo("완료", f"분석 및 오버레이 영상 생성이 완료되었습니다!\n\nJSON: {self.output_json_path}\n동영상: {target_video}")

    def on_failure(self, error_msg):
        self.progress.config(value=0)
        self.btn_process.config(state=tk.NORMAL)
        self.lbl_status.config(text="작업 실패", foreground="red")
        messagebox.showerror("오류", f"동영상 처리 중 오류가 발생했습니다:\n{error_msg}")

    # --- Video Player controls ---
    def open_verify_video(self):
        file_path = filedialog.askopenfilename(
            title="확인할 오버레이 동영상 선택",
            filetypes=[("동영상 파일", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm")]
        )
        if file_path:
            try:
                self.pause()
                self.load_video(os.path.abspath(file_path))
                self.lbl_status.config(text="확인용 영상 로드 완료", foreground="green")
            except Exception as e:
                messagebox.showerror("오류", f"영상을 불러올 수 없습니다:\n{str(e)}")

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception("비디오를 열 수 없습니다.")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
            
        self.video_duration = self.total_frames / self.fps
        self.current_frame_idx = 0
        
        # Update scrubber ranges
        self.scale_play.config(to=self.total_frames - 1)
        self.scale_play.config(state=tk.NORMAL)
        self.scale_play.set(0)
        
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)
        self.btn_play.config(state=tk.NORMAL)
        
        self.show_frame(0)

    def show_frame(self, frame_idx):
        if not self.cap:
            return
            
        try:
            self.current_frame_idx = frame_idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                max_w, max_h = 640, 360
                ratio = min(max_w / w, max_h / h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (new_w, new_h))
                
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                self.lbl_preview.config(image=photo)
                self.lbl_preview.image = photo
                
                curr_sec = frame_idx / self.fps
                self.lbl_time_info.config(
                    text=f"프레임: {frame_idx} / {self.total_frames - 1}  |  시간: {format_seconds(curr_sec)} / {format_seconds(self.video_duration)}"
                )
        except Exception as e:
            print(f"프레임 렌더링 에러: {e}")

    def on_slider_move(self, val):
        if not self.cap:
            return
        frame_idx = int(float(val))
        if frame_idx != self.current_frame_idx:
            self.show_frame(frame_idx)

    def prev_frame(self):
        if self.cap and self.current_frame_idx > 0:
            target_frame = self.current_frame_idx - 1
            self.scale_play.set(target_frame)
            self.show_frame(target_frame)
            
    def next_frame(self):
        if self.cap and self.current_frame_idx < self.total_frames - 1:
            target_frame = self.current_frame_idx + 1
            self.scale_play.set(target_frame)
            self.show_frame(target_frame)

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if not self.cap:
            return
        self.playing = True
        self.btn_play.config(text="⏸ 일시정지")
        self.play_loop()

    def pause(self):
        self.playing = False
        self.btn_play.config(text="▶ 재생")
        if self.play_timer:
            self.root.after_cancel(self.play_timer)
            self.play_timer = None

    def play_loop(self):
        if not self.playing:
            return
        if self.current_frame_idx < self.total_frames - 1:
            next_frame = self.current_frame_idx + 1
            self.scale_play.set(next_frame)
            self.show_frame(next_frame)
            
            delay = int(1000 / self.fps)
            self.play_timer = self.root.after(delay, self.play_loop)
        else:
            self.pause()

    def on_close(self):
        self.playing = False
        if self.play_timer:
            try:
                self.root.after_cancel(self.play_timer)
            except Exception:
                pass
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseEstimatorApp(root)
    root.mainloop()
