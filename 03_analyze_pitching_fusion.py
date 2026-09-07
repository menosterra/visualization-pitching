import os
import sys
import json
import math
import webbrowser
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Landmark indices mapping (Same as MediaPipe)
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28

# Matplotlib Korean font configuration (Windows Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- Core mathematical biomechanics logic ---
def detect_throwing_hand(frames, dt):
    left_wrist_vels = []
    right_wrist_vels = []
    for i in range(1, len(frames)):
        prev_lms = frames[i-1]["landmarks2d"]
        curr_lms = frames[i]["landmarks2d"]
        if not prev_lms or not curr_lms:
            continue
        
        lw_prev = prev_lms[L_WRIST]
        lw_curr = curr_lms[L_WRIST]
        rw_prev = prev_lms[R_WRIST]
        rw_curr = curr_lms[R_WRIST]
        
        l_vel = math.sqrt((lw_curr["x"] - lw_prev["x"])**2 + (lw_curr["y"] - lw_prev["y"])**2) / dt
        r_vel = math.sqrt((rw_curr["x"] - rw_prev["x"])**2 + (rw_curr["y"] - rw_prev["y"])**2) / dt
        
        left_wrist_vels.append(l_vel)
        right_wrist_vels.append(r_vel)
        
    peak_l = max(left_wrist_vels) if left_wrist_vels else 0
    peak_r = max(right_wrist_vels) if right_wrist_vels else 0
    return 'R' if peak_r > peak_l else 'L'

def detect_leg_lift_frame_2d(frames, lead_ankle_idx):
    lead_ankle_v = []
    for f in frames:
        if f["detected"] and len(f["landmarks2d"]) > lead_ankle_idx:
            lead_ankle_v.append(f["landmarks2d"][lead_ankle_idx]["y"])
        else:
            lead_ankle_v.append(0.8)
            
    baseline_v = np.mean(lead_ankle_v[:10]) if len(lead_ankle_v) >= 10 else 0.8
    lift_threshold = 0.02
    lift_idx = 0
    for idx, v in enumerate(lead_ankle_v):
        if baseline_v - v > lift_threshold:
            search_start = max(0, idx - 15)
            sub_arr = lead_ankle_v[search_start:idx+1]
            lift_idx = search_start + np.argmax(sub_arr)
            break
    return lift_idx

def get_2d_scale_factor(frames, pitcher_height_m):
    heights = []
    for f in frames[:10]:
        if not f["detected"]:
            continue
        lms = f["landmarks2d"]
        head_v = (lms[0]["y"] + lms[7]["y"] + lms[8]["y"]) / 3.0
        floor_v = (lms[27]["y"] + lms[28]["y"]) / 2.0
        heights.append(floor_v - head_v)
        
    predicted_height = np.mean(heights) if heights else 0.5
    if predicted_height > 0.1:
        return pitcher_height_m / predicted_height
    return 1.0

def get_dynamic_scale(lms, pitcher_height_m, default_scale=1.0):
    if not lms or len(lms) <= 28:
        return default_scale
    head_v = (lms[0]["y"] + lms[7]["y"] + lms[8]["y"]) / 3.0
    floor_v = (lms[27]["y"] + lms[28]["y"]) / 2.0
    h_val = floor_v - head_v
    if h_val > 0.1:
        return pitcher_height_m / h_val
    return default_scale

def align_coordinates(landmarks3d, view_type):
    aligned = []
    vt = view_type.lower()
    inv_sqrt2 = 0.70710678  # cos(45 deg) = sin(45 deg)
    for lm in landmarks3d:
        x, y, z = lm["x"], lm["y"], lm["z"]
        vis = lm.get("visibility", 0.0)
        
        if "포수~1루" in vt or "catcher~1b" in vt or "1b" in vt and "catcher" in vt:
            # Diagonal View (포수~1루: 135도 대각선 시점)
            x_std = inv_sqrt2 * (z - x)
            y_std = y
            z_std = inv_sqrt2 * (x - z)
        elif "포수~3루" in vt or "catcher~3b" in vt or "3b" in vt and "catcher" in vt:
            # Diagonal View (포수~3루: 45도 대각선 시점)
            x_std = inv_sqrt2 * (x + z)
            y_std = y
            z_std = inv_sqrt2 * (z + x)
        elif "1st base" in vt or "1루" in vt:
            # 1st Base View: 180 deg rotation around Y axis (x, y, z) -> (-x, y, -z)
            x_std = -x
            y_std = y
            z_std = -z
        elif "front" in vt or "catcher" in vt or "포수" in vt:
            # Front View (Catcher): Rotate 90 deg (std.X = front.Z, std.Z = front.X)
            x_std = z
            y_std = y
            z_std = x
        elif "2nd base" in vt or "2루" in vt or "second" in vt or "후면" in vt:
            # 2nd Base View (Rear/Back): (std.X = -front.Z, std.Z = -front.X)
            x_std = -z
            y_std = y
            z_std = -x
        else:
            # Side View (3rd Base) / Default: Standard
            x_std = x
            y_std = y
            z_std = z
            
        aligned.append({
            "x": x_std,
            "y": y_std,
            "z": z_std,
            "visibility": vis
        })
    return aligned

try:
    from analyze_elite_pitching import get_elite_pitching_series
    _PRELOADED_ELITE_DATA = get_elite_pitching_series()
except ImportError:
    from analyze_standard_pitching import get_standard_pitching_series
    _PRELOADED_ELITE_DATA = get_standard_pitching_series()

def interpolate_landmarks_2d(frames, t_rel_c, t_grid):
    valid_indices = [idx for idx, f in enumerate(frames) if f["detected"] and len(f.get("landmarks2d", [])) >= 33]
    if not valid_indices:
        return [[{"x": 0.0, "y": 0.0, "visibility": 0.0} for _ in range(33)] for _ in t_grid]
    
    t_source = np.array([t_rel_c[idx] for idx in valid_indices])
    
    u_mat = np.array([[frames[idx]["landmarks2d"][j]["x"] for j in range(33)] for idx in valid_indices])
    v_mat = np.array([[frames[idx]["landmarks2d"][j]["y"] for j in range(33)] for idx in valid_indices])
    vis_mat = np.array([[frames[idx]["landmarks2d"][j].get("visibility", 0.0) for j in range(33)] for idx in valid_indices])
    
    u_interp = np.column_stack([np.interp(t_grid, t_source, u_mat[:, j]) for j in range(33)])
    v_interp = np.column_stack([np.interp(t_grid, t_source, v_mat[:, j]) for j in range(33)])
    vis_interp = np.column_stack([np.interp(t_grid, t_source, vis_mat[:, j]) for j in range(33)])
    
    interpolated_sequence = []
    for i in range(len(t_grid)):
        lms_interp = [{
            "x": float(u_interp[i, j]),
            "y": float(v_interp[i, j]),
            "visibility": float(vis_interp[i, j])
        } for j in range(33)]
        interpolated_sequence.append(lms_interp)
    return interpolated_sequence

def interpolate_landmarks_3d(frames, t_rel_c, t_grid):
    valid_indices = [idx for idx, f in enumerate(frames) if f["detected"] and len(f.get("landmarks3d", [])) >= 33]
    if not valid_indices:
        return [[{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(33)] for _ in t_grid]
    
    t_source = np.array([t_rel_c[idx] for idx in valid_indices])
    
    x_mat = np.array([[frames[idx]["landmarks3d"][j]["x"] for j in range(33)] for idx in valid_indices])
    y_mat = np.array([[frames[idx]["landmarks3d"][j]["y"] for j in range(33)] for idx in valid_indices])
    z_mat = np.array([[frames[idx]["landmarks3d"][j]["z"] for j in range(33)] for idx in valid_indices])
    vis_mat = np.array([[frames[idx]["landmarks3d"][j].get("visibility", 0.0) for j in range(33)] for idx in valid_indices])
    
    x_interp = np.column_stack([np.interp(t_grid, t_source, x_mat[:, j]) for j in range(33)])
    y_interp = np.column_stack([np.interp(t_grid, t_source, y_mat[:, j]) for j in range(33)])
    z_interp = np.column_stack([np.interp(t_grid, t_source, z_mat[:, j]) for j in range(33)])
    vis_interp = np.column_stack([np.interp(t_grid, t_source, vis_mat[:, j]) for j in range(33)])
    
    interpolated_sequence = []
    for i in range(len(t_grid)):
        lms_interp = [{
            "x": float(x_interp[i, j]),
            "y": float(y_interp[i, j]),
            "z": float(z_interp[i, j]),
            "visibility": float(vis_interp[i, j])
        } for j in range(33)]
        interpolated_sequence.append(lms_interp)
    return interpolated_sequence

# --- GUI Application Frame ---
class PitchingAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Vector Fusion Pitching Biomechanics Analyzer")
        self.root.geometry("600x440")
        self.root.resizable(False, False)
        
        self.file1_path = ""
        self.file2_path = ""
        
        # Styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.root.option_add("*Font", ("맑은 고딕", 9))
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selector 1
        lf_file1 = ttk.LabelFrame(main_frame, text=" 1. 첫 번째 Pose JSON 파일 선택 ", padding="10")
        lf_file1.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_file1 = ttk.Label(lf_file1, text="선택된 파일 없음", width=40, anchor=tk.W, foreground="gray")
        self.lbl_file1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        btn_file1 = ttk.Button(lf_file1, text="파일 선택", command=self.select_file1)
        btn_file1.pack(side=tk.LEFT, padx=(5, 5))
        
        view_options = [
            "Side View (측면 - 3루)", 
            "Front View (정면 - 포수)", 
            "1st Base View (측면 - 1루)", 
            "2nd Base View (후면 - 2루)",
            "Diagonal View (대각선 - 포수~1루)",
            "Diagonal View (대각선 - 포수~3루)"
        ]
        
        self.combo_view1 = ttk.Combobox(lf_file1, values=view_options, state="readonly", width=25)
        self.combo_view1.current(0)  # Default to Side (3rd Base)
        self.combo_view1.pack(side=tk.RIGHT)
        
        # File selector 2
        lf_file2 = ttk.LabelFrame(main_frame, text=" 2. 두 번째 Pose JSON 파일 선택 ", padding="10")
        lf_file2.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_file2 = ttk.Label(lf_file2, text="선택된 파일 없음", width=38, anchor=tk.W, foreground="gray")
        self.lbl_file2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        btn_file2 = ttk.Button(lf_file2, text="파일 선택", command=self.select_file2)
        btn_file2.pack(side=tk.LEFT, padx=(5, 5))
        
        self.combo_view2 = ttk.Combobox(lf_file2, values=view_options, state="readonly", width=25)
        self.combo_view2.current(1)  # Default to Front (Catcher)
        self.combo_view2.pack(side=tk.RIGHT)
        
        # Configurations
        lf_config = ttk.LabelFrame(main_frame, text=" 3. 피쳐 설정 정보 ", padding="10")
        lf_config.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(lf_config, text="투수 키 (Height):").pack(side=tk.LEFT, padx=(5, 5))
        self.entry_height = ttk.Entry(lf_config, width=8)
        self.entry_height.insert(0, "175.0")
        self.entry_height.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(lf_config, text="cm").pack(side=tk.LEFT, padx=(0, 15))
        
        self.var_auto_open = tk.BooleanVar(value=True)
        self.chk_auto_open = ttk.Checkbutton(lf_config, text="분석 완료 후 대시보드 리포트 자동 열기", variable=self.var_auto_open)
        self.chk_auto_open.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Action controls
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress = ttk.Progressbar(action_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_status = ttk.Label(action_frame, text="파일과 키를 지정하고 분석을 실행하십시오.", font=("맑은 고딕", 9, "bold"), foreground="dimgray")
        self.lbl_status.pack(side=tk.LEFT, padx=5)
        
        self.btn_run = ttk.Button(action_frame, text="2D 벡터 융합 및 분석 시작", command=self.start_analysis, width=28)
        self.btn_run.pack(side=tk.RIGHT, ipadx=5, ipady=3)
        
    def select_file1(self):
        file_path = filedialog.askopenfilename(title="첫 번째 Pose JSON 선택", filetypes=[("JSON 파일", "*.json")])
        if file_path:
            self.file1_path = os.path.abspath(file_path)
            self.lbl_file1.config(text=os.path.basename(self.file1_path), foreground="black")
            
    def select_file2(self):
        file_path = filedialog.askopenfilename(title="두 번째 Pose JSON 선택", filetypes=[("JSON 파일", "*.json")])
        if file_path:
            self.file2_path = os.path.abspath(file_path)
            self.lbl_file2.config(text=os.path.basename(self.file2_path), foreground="black")
            
    def start_analysis(self):
        # Validations: Accept either one file (Single Mode) or two files (Fusion Mode)
        if not self.file1_path and not self.file2_path:
            messagebox.showerror("오류", "분석을 수행하기 위해 최소한 하나의 Pose JSON 파일을 지정하십시오.")
            return
            
        view1 = self.combo_view1.get()
        view2 = self.combo_view2.get()
        
        is_single_mode = False
        if not self.file1_path or not self.file2_path:
            is_single_mode = True
            
        try:
            height_cm = float(self.entry_height.get())
            if height_cm <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("오류", "투수 키 입력란에 올바른 양의 실수(예: 175.0)를 입력해 주십시오.")
            return
            
        if is_single_mode:
            # Single file mode binding: Pack the active file path to side_path
            # Pass the viewpoint selection label to front_path to signal analysis_worker
            if self.file1_path:
                side_path = self.file1_path
                front_path = view1  # Text signal for Single Mode viewpoint type
                side_view_type = view1
                front_view_type = view1
            else:
                side_path = self.file2_path
                front_path = view2
                side_view_type = view2
                front_view_type = view2
        else:
            # Dual fusion mode validation & binding
            if view1 == view2:
                messagebox.showerror("오류", "기하학적 융합을 위해 서로 다른 두 개의 시점을 선택해야 합니다.")
                return
                
            is_side1 = "Side" in view1 or "1st Base" in view1 or "포수~3루" in view1
            is_side2 = "Side" in view2 or "1st Base" in view2 or "포수~3루" in view2
            
            if is_side1 and not is_side2:
                side_path = self.file1_path
                front_path = self.file2_path
                side_view_type = view1
                front_view_type = view2
            elif not is_side1 and is_side2:
                side_path = self.file2_path
                front_path = self.file1_path
                side_view_type = view2
                front_view_type = view1
            else:
                side_path = self.file1_path
                front_path = self.file2_path
                side_view_type = view1
                front_view_type = view2
                
        self.btn_run.config(state=tk.DISABLED)
        self.lbl_status.config(text="분석을 수행하는 중입니다...", foreground="blue")
        self.progress['value'] = 0
        
        # Run background thread
        worker = threading.Thread(target=self.analysis_worker, args=(side_path, front_path, height_cm / 100.0, side_view_type, front_view_type), daemon=True)
        worker.start()
        
    def update_status_safe(self, text, progress_val):
        self.root.after(0, lambda: self.lbl_status.config(text=text, foreground="darkorange"))
        self.root.after(0, lambda: self.progress.config(value=progress_val))
        
    def analysis_worker(self, side_path, front_path, pitcher_height_m, side_view_type, front_view_type):
        try:
            # Determine if we are running in Single Video Mode
            # front_path is passed as viewport text label (e.g. "Side View...", "Front View...") when in Single Mode
            is_single_mode = not os.path.exists(str(front_path))
            
            self.update_status_safe("JSON 데이터 로드 중...", 10)
            with open(side_path, 'r', encoding='utf-8') as f:
                side_data = json.load(f)
                
            raw_side = side_data["frames"]
            dt_s = 1.0 / side_data["fps"]
            
            # Align Side View coordinates to standard 3rd Base coordinates
            for frame in raw_side:
                if frame["detected"] and "landmarks3d" in frame:
                    frame["landmarks3d"] = align_coordinates(frame["landmarks3d"], side_view_type)
            
            if is_single_mode:
                raw_front = raw_side
                dt_f = dt_s
                front_data = side_data
            else:
                with open(front_path, 'r', encoding='utf-8') as f:
                    front_data = json.load(f)
                raw_front = front_data["frames"]
                
                # Align Front/Rear View coordinates to standard 3rd Base coordinates
                for frame in raw_front:
                    if frame["detected"] and "landmarks3d" in frame:
                        frame["landmarks3d"] = align_coordinates(frame["landmarks3d"], front_view_type)
                        
                dt_f = 1.0 / front_data["fps"]
                
            # 1. Detect hand and peak wrist velocity frames (Release Point)
            self.update_status_safe("투구 타임라인 동기화 중...", 30)
            throws = detect_throwing_hand(raw_side, dt_s)
            lead_ankle_idx = L_ANKLE if throws == 'R' else R_ANKLE
            
            # Find release frame for Side view (max raw wrist speed)
            side_wrist_speeds = []
            for j in range(1, len(raw_side)):
                prev = raw_side[j-1]["landmarks2d"]
                curr = raw_side[j]["landmarks2d"]
                if prev and curr:
                    w_idx = R_WRIST if throws == 'R' else L_WRIST
                    speed = math.sqrt((curr[w_idx]["x"] - prev[w_idx]["x"])**2 + (curr[w_idx]["y"] - prev[w_idx]["y"])**2)
                    side_wrist_speeds.append(speed)
                else:
                    side_wrist_speeds.append(0.0)
            release_frame_s = np.argmax(side_wrist_speeds) + 1 if side_wrist_speeds else 40
            release_time_s = release_frame_s * dt_s
            
            if is_single_mode:
                release_frame_f = release_frame_s
                release_time_f = release_time_s
            else:
                # Find release frame for Front view
                front_wrist_speeds = []
                for j in range(1, len(raw_front)):
                    prev = raw_front[j-1]["landmarks2d"]
                    curr = raw_front[j]["landmarks2d"]
                    if prev and curr:
                        w_idx = R_WRIST if throws == 'R' else L_WRIST
                        speed = math.sqrt((curr[w_idx]["x"] - prev[w_idx]["x"])**2 + (curr[w_idx]["y"] - prev[w_idx]["y"])**2)
                        front_wrist_speeds.append(speed)
                    else:
                        front_wrist_speeds.append(0.0)
                release_frame_f = np.argmax(front_wrist_speeds) + 1 if front_wrist_speeds else 40
                release_time_f = release_frame_f * dt_f
                
            # Align timelines based on release peak
            # relative time: t = 0.0 at release point
            t_rel_s = [f["time"] - release_time_s for f in raw_side]
            t_rel_f = [f["time"] - release_time_f for f in raw_front]
            
            # Align from -0.8s before release to +0.4s after release
            t_grid = np.linspace(-0.8, 0.4, 61)
            dt_grid = 1.2 / 60
            
            self.update_status_safe("2D 및 3D 랜드마크 보간 중...", 50)
            side_interp = interpolate_landmarks_2d(raw_side, t_rel_s, t_grid)
            side_interp_3d = interpolate_landmarks_3d(raw_side, t_rel_s, t_grid)
            
            if is_single_mode:
                front_interp = side_interp
                front_interp_3d = side_interp_3d
            else:
                front_interp = interpolate_landmarks_2d(raw_front, t_rel_f, t_grid)
                front_interp_3d = interpolate_landmarks_3d(raw_front, t_rel_f, t_grid)
                
            scale_s = get_2d_scale_factor(raw_side, pitcher_height_m)
            scale_f = scale_s if is_single_mode else get_2d_scale_factor(raw_front, pitcher_height_m)
            
            # Estimate zoom/distortion based on pitcher's normalized 2D height in raw_side
            heights_s = []
            for f in raw_side[:10]:
                if f["detected"]:
                    lms = f["landmarks2d"]
                    head_v = (lms[0]["y"] + lms[7]["y"] + lms[8]["y"]) / 3.0
                    floor_v = (lms[27]["y"] + lms[28]["y"]) / 2.0
                    heights_s.append(floor_v - head_v)
            pred_h_s = np.mean(heights_s) if heights_s else 0.5
            
            # Apply speed correction for unzoomed/wide-angle videos (where pitcher height < 50% of frame)
            zoom_correction = 1.0
            if pred_h_s < 0.5:
                zoom_correction = 1.0 + 0.6 * (0.5 - pred_h_s)
                zoom_correction = min(1.25, max(1.0, zoom_correction))
            
            # 3D Kinematics and Vector Fusion
            self.update_status_safe("손목 속도 및 관절 회전 각도 산출 중...", 70)
            raw_hip_angles = []
            raw_shoulder_angles = []
            hand_speeds = []
            times = []
            
            for i in range(len(t_grid)):
                t_val = t_grid[i]
                times.append(t_val)
                
                # rotation landmarks sequence (always use Camera 1 side_interp_3d)
                lms_3d_rot = side_interp_3d[i]
                
                # Hip and Shoulder vectors in top-down horizontal X-Z plane
                if throws == 'R':
                    h_vec = np.array([
                        lms_3d_rot[L_HIP]["x"] - lms_3d_rot[R_HIP]["x"],
                        lms_3d_rot[L_HIP]["z"] - lms_3d_rot[R_HIP]["z"]
                    ])
                    s_vec = np.array([
                        lms_3d_rot[L_SHOULDER]["x"] - lms_3d_rot[R_SHOULDER]["x"],
                        lms_3d_rot[L_SHOULDER]["z"] - lms_3d_rot[R_SHOULDER]["z"]
                    ])
                else:
                    h_vec = np.array([
                        lms_3d_rot[R_HIP]["x"] - lms_3d_rot[L_HIP]["x"],
                        lms_3d_rot[R_HIP]["z"] - lms_3d_rot[L_HIP]["z"]
                    ])
                    s_vec = np.array([
                        lms_3d_rot[R_SHOULDER]["x"] - lms_3d_rot[L_SHOULDER]["x"],
                        lms_3d_rot[R_SHOULDER]["z"] - lms_3d_rot[L_SHOULDER]["z"]
                    ])
                
                h_angle_rad = math.atan2(h_vec[1], h_vec[0])
                s_angle_rad = math.atan2(s_vec[1], s_vec[0])
                
                raw_hip_angles.append(h_angle_rad)
                raw_shoulder_angles.append(s_angle_rad)
                
                # Hand Speed (3D landmarks Vector Component Fusion with Zoom/Distortion Correction)
                if i > 0:
                    lms_s_3d = side_interp_3d[i]
                    lms_s_prev_3d = side_interp_3d[i-1]
                    
                    lms_f_3d = front_interp_3d[i]
                    lms_f_prev_3d = front_interp_3d[i-1]
                    
                    w_idx = R_WRIST if throws == 'R' else L_WRIST
                    
                    if is_single_mode:
                        # Single Mode: Calculate 3D velocity directly from side_interp_3d components
                        vx = (lms_s_3d[w_idx]["x"] - lms_s_prev_3d[w_idx]["x"]) / dt_grid
                        vy = (lms_s_3d[w_idx]["y"] - lms_s_prev_3d[w_idx]["y"]) / dt_grid
                        vz = (lms_s_3d[w_idx]["z"] - lms_s_prev_3d[w_idx]["z"]) / dt_grid
                    else:
                        # Dual Fusion Mode: Cross-view 3D geometric velocity fusion
                        # 1. Forward pitch velocity (X-axis): Side/1B view has direct optical sensor plane precision
                        # 2. Vertical velocity (Y-axis): Both views capture vertical motion directly (averaged)
                        # 3. Lateral arm sweep velocity (Z-axis): Front/Catcher view has direct optical sensor plane precision
                        vx_s = (lms_s_3d[w_idx]["x"] - lms_s_prev_3d[w_idx]["x"]) / dt_grid
                        vx_f = (lms_f_3d[w_idx]["x"] - lms_f_prev_3d[w_idx]["x"]) / dt_grid
                        
                        vy_s = (lms_s_3d[w_idx]["y"] - lms_s_prev_3d[w_idx]["y"]) / dt_grid
                        vy_f = (lms_f_3d[w_idx]["y"] - lms_f_prev_3d[w_idx]["y"]) / dt_grid
                        
                        vz_s = (lms_s_3d[w_idx]["z"] - lms_s_prev_3d[w_idx]["z"]) / dt_grid
                        vz_f = (lms_f_3d[w_idx]["z"] - lms_f_prev_3d[w_idx]["z"]) / dt_grid
                        
                        st_low = side_view_type.lower()
                        ft_low = front_view_type.lower()
                        
                        if "side" in st_low or "1st" in st_low or "3루" in st_low:
                            vx = vx_s
                        elif "side" in ft_low or "1st" in ft_low or "3루" in ft_low:
                            vx = vx_f
                        else:
                            vx = (vx_s + vx_f) / 2.0
                            
                        vy = (vy_s + vy_f) / 2.0
                        
                        if "front" in ft_low or "catcher" in ft_low or "2nd" in ft_low or "포수" in ft_low:
                            vz = vz_f
                        elif "front" in st_low or "catcher" in st_low or "2nd" in st_low or "포수" in st_low:
                            vz = vz_s
                        else:
                            vz = (vz_s + vz_f) / 2.0
                    speed_mps = math.sqrt(vx**2 + vy**2 + vz**2) * zoom_correction
                    speed_kmh = speed_mps * 3.6
                    hand_speeds.append(speed_kmh)
                else:
                    hand_speeds.append(0.0)
                    
            # Foot plant detection (using lead ankle speed drop before release)
            fp_grid_idx = 0
            min_ankle_speed = 999.0
            lead_ankle = L_ANKLE if throws == 'R' else R_ANKLE
            for i in range(1, len(t_grid)):
                if t_grid[i] >= 0.0:  # Only search before release point
                    break
                v_curr = side_interp[i][lead_ankle]
                v_prev = side_interp[i-1][lead_ankle]
                ankle_speed = math.sqrt((v_curr["x"] - v_prev["x"])**2 + (v_curr["y"] - v_prev["y"])**2) * scale_s / dt_grid
                if t_grid[i] > -0.5 and ankle_speed < min_ankle_speed:
                    min_ankle_speed = ankle_speed
                    fp_grid_idx = i
            fp_time_rel = t_grid[fp_grid_idx] if fp_grid_idx > 0 else -0.22
            
            # Leg lift detection (highest lead ankle height on side view before foot plant)
            lift_grid_idx = 0
            min_y = 999.0
            for i in range(len(t_grid)):
                if t_grid[i] >= fp_time_rel:
                    break
                y_val = side_interp[i][lead_ankle]["y"]
                if y_val < min_y:
                    min_y = y_val
                    lift_grid_idx = i
            lift_time_rel = t_grid[lift_grid_idx] if lift_grid_idx > 0 else -0.45
            
            # 5-point rolling average window for smoothing derivative noise
            smoothed_speeds = []
            window_size = 5
            half_w = window_size // 2
            for idx in range(len(hand_speeds)):
                start = max(0, idx - half_w)
                end = min(len(hand_speeds), idx + half_w + 1)
                avg_speed = sum(hand_speeds[start:end]) / (end - start)
                smoothed_speeds.append(avg_speed)
            hand_speeds = smoothed_speeds
            
            # Peak Hand Release Time is at t = 0.0 relative to release alignment
            peak_hand_speed = max(hand_speeds)
            
            # Angle unwrapping and monotonic increasing alignment (Leg Lift as 0° baseline)
            hip_unwrapped = np.degrees(np.unwrap(raw_hip_angles))
            shoulder_unwrapped = np.degrees(np.unwrap(raw_shoulder_angles))
            
            h_base = hip_unwrapped[lift_grid_idx] if lift_grid_idx < len(hip_unwrapped) else hip_unwrapped[0]
            s_base = shoulder_unwrapped[lift_grid_idx] if lift_grid_idx < len(shoulder_unwrapped) else shoulder_unwrapped[0]
            
            hip_angles = list(hip_unwrapped - h_base)
            shoulder_angles = list(shoulder_unwrapped - s_base)
            
            # --- Forward Rotational Sign Normalization ---
            # Standard biomechanics convention: Forward rotation towards release must progress in positive direction (0° -> +90° -> +180°)
            # If negative (e.g. Left-handed pitcher CW rotation or 2nd base view coordinate reversal), mirror sign (* -1)
            rel_idx = np.argmin([abs(t - 0.0) for t in t_grid])
            if shoulder_angles[rel_idx] < 0:
                hip_angles = [-a for a in hip_angles]
                shoulder_angles = [-a for a in shoulder_angles]
                
            separation_angles = list(np.array(shoulder_angles) - np.array(hip_angles))
            
            # --- Hybrid Time Shifting for Plotting (Pivot Leg Lift as t = 0.0) ---
            times_plot = [t - lift_time_rel for t in times]
            t_lift_plot = 0.0
            t_fp_plot = fp_time_rel - lift_time_rel
            t_release_plot = 0.0 - lift_time_rel
            
            # 4. Generate Combined Chart (Standard Elite + User)
            self.update_status_safe("역학 분석 그래프 그리는 중...", 85)
            
            # Load Standard (Elite) series for combined visualization
            std_data = _PRELOADED_ELITE_DATA if _PRELOADED_ELITE_DATA is not None else get_standard_pitching_series()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # --- Subplot 1: Rotation ---
            # Standard curves (Dashed lines, alpha=0.7)
            if std_data:
                ax1.plot(std_data["times_plot"], std_data["hip_angles"], color="royalblue", linestyle="--", linewidth=1.8, alpha=0.7)
                ax1.plot(std_data["times_plot"], std_data["shoulder_angles"], color="cyan", linestyle="--", linewidth=1.8, alpha=0.7)
                ax1.plot(std_data["times_plot"], std_data["separation_angles"], color="darkorange", linestyle="--", linewidth=1.8, alpha=0.7)
                # Standard Event Lines (Dashed)
                ax1.axvline(x=std_data["t_fp_plot"], color='green', linestyle='--', linewidth=1.2, alpha=0.6)
                ax1.axvline(x=std_data["t_release_plot"], color='purple', linestyle='--', linewidth=1.2, alpha=0.6)
            
            # User curves (Solid lines, linewidth=2.5)
            ax1.plot(times_plot, hip_angles, label="Hip Rotation", color="royalblue", linestyle="-", linewidth=2.5)
            ax1.plot(times_plot, shoulder_angles, label="Shoulder Rotation", color="cyan", linestyle="-", linewidth=2.5)
            ax1.plot(times_plot, separation_angles, label="Separation", color="darkorange", linestyle="-", linewidth=2.5)
            
            # User Event Lines (Thin solid lines, t=0 is single solid line)
            ax1.axvline(x=t_lift_plot, color='gray', linestyle='-', linewidth=1.2, label='Leg Lift')
            ax1.axvline(x=t_fp_plot, color='green', linestyle='-', linewidth=1.2, label='Foot Plant')
            ax1.axvline(x=t_release_plot, color='purple', linestyle='-', linewidth=1.2, label='Ball Release')
            
            ax1.set_ylabel("Angle (deg)")
            ax1.set_title("Rotation")
            
            # Expand Y-limits by 20 deg to comfortably display Elite range
            std_min_sep = min(std_data["separation_angles"]) if std_data else -50.0
            std_max_rot = max(std_data["shoulder_angles"] + std_data["hip_angles"]) if std_data else 150.0
            min_y = min(-75.0, float(min(separation_angles)) - 25.0, float(std_min_sep) - 20.0)
            max_y = max(180.0, float(max(shoulder_angles + hip_angles)) + 25.0, float(std_max_rot) + 20.0)
            ax1.set_ylim(min_y, max_y)
            ax1.legend(loc="upper left")
            ax1.text(0.98, 0.93, "* Dot line (Elite)", transform=ax1.transAxes,
                     fontsize=9, color="#444444", ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.8))
            
            # --- Subplot 2: Wrist Velocity ---
            # Standard curve (Dashed line)
            if std_data:
                ax2.plot(std_data["times_plot"], std_data["wrist_speeds"], color="magenta", linestyle="--", linewidth=1.8, alpha=0.7)
                ax2.axvline(x=std_data["t_fp_plot"], color='green', linestyle='--', linewidth=1.2, alpha=0.6)
                ax2.axvline(x=std_data["t_release_plot"], color='purple', linestyle='--', linewidth=1.2, alpha=0.6)
                ax2.text(std_data["t_release_plot"] - 0.02, std_data["peak_wrist_speed"] + 2.5,
                         f"Elite Release: {std_data['peak_wrist_speed']:.1f} km/h",
                         fontsize=9, color="purple", ha='right', va='bottom', fontweight='semibold')
            
            # User curve (Solid line)
            ax2.plot(times_plot, hand_speeds, label="Wrist Velocity", color="magenta", linestyle="-", linewidth=2.5)
            ax2.axvline(x=t_lift_plot, color='gray', linestyle='-', linewidth=1.2)
            ax2.axvline(x=t_fp_plot, color='green', linestyle='-', linewidth=1.2)
            ax2.axvline(x=t_release_plot, color='purple', linestyle='-', linewidth=1.2)
            
            ax2.text(t_release_plot - 0.02, peak_hand_speed + 2.5,
                     f"User Release: {peak_hand_speed:.1f} km/h",
                     fontsize=9, color="#222222", ha='right', va='bottom', fontweight='semibold')
                         
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Velocity (km/h)")
            ax2.set_title("Wrist Velocity")
            
            max_wrist = max(peak_hand_speed, std_data["peak_wrist_speed"] if std_data else 80.0)
            ax2.set_ylim(-2, max_wrist + 15)
            
            ax2.legend(loc="upper left")
            ax2.text(0.98, 0.93, "* Dot line (Elite)", transform=ax2.transAxes,
                     fontsize=9, color="#444444", ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.8))
            
            plt.tight_layout()
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            chart_output_path = os.path.join(base_dir, "video", "pitching_mechanics_chart.png")
            plt.savefig(chart_output_path, dpi=120)
            legacy_chart_path = os.path.join(base_dir, "video", "pitching_biomechanics_chart.png")
            plt.savefig(legacy_chart_path, dpi=120)
            plt.close()
            
            # Calculate hip-shoulder separation (X-Factor stretch) at foot plant
            h_rot_fp = hip_angles[fp_grid_idx]
            s_rot_fp = shoulder_angles[fp_grid_idx]
            sep_fp = abs(h_rot_fp - s_rot_fp)
            if sep_fp > 180:
                sep_fp = 360 - sep_fp
                
            # Base whip multiplier for elite pitcher is 1.868 (translating wrist speed to ball release speed)
            base_whip_multiplier = 1.868
            
            # Efficiency factor based on hip-shoulder separation (X-Factor)
            # Optimal separation is around 50 degrees
            opt_sep = 50.0
            sep_ratio = min(1.0, max(0.4, sep_fp / opt_sep))
            efficiency = 0.8 + 0.2 * sep_ratio  # Range: 0.88 to 1.0
            
            whip_multiplier = base_whip_multiplier * efficiency
            estimated_pitch_speed_kmh = peak_hand_speed * whip_multiplier

            # --- Physics-based Pitch Speed Model (User assumptions) ---
            # 1. Arm geometry and masses based on height
            R_arm = 0.42 * pitcher_height_m       # Shoulder to wrist radius (meters)
            L_finger = 0.10 * pitcher_height_m    # Wrist to ball center (meters)
            L_forearm = 0.16 * pitcher_height_m   # Forearm length (meters)
            m_ball = 0.145                        # Ball mass (kg)
            m_forearm = 1.2                       # Forearm mass (kg)
            
            # 2. Estimate pelvis translation speed at peak hand velocity frame from raw_side
            peak_idx = np.argmax(hand_speeds)
            peak_t_rel = t_grid[peak_idx]
            raw_peak_idx = 0
            min_dt = 999.0
            for idx, f in enumerate(raw_side):
                t_rel_val = f["time"] - release_time_s
                dt_val = abs(t_rel_val - peak_t_rel)
                if dt_val < min_dt:
                    min_dt = dt_val
                    raw_peak_idx = idx
            
            v_trans_mps = 3.9  # Default standard elite pelvis speed
            if raw_peak_idx > 0 and raw_peak_idx < len(raw_side):
                f_curr = raw_side[raw_peak_idx]
                f_prev = raw_side[raw_peak_idx - 1]
                if f_curr["detected"] and f_prev["detected"]:
                    lms_curr = f_curr["landmarks2d"]
                    lms_prev = f_prev["landmarks2d"]
                    hip_curr_x = (lms_curr[L_HIP]["x"] + lms_curr[R_HIP]["x"]) / 2.0
                    hip_prev_x = (lms_prev[L_HIP]["x"] + lms_prev[R_HIP]["x"]) / 2.0
                    dx_p = (hip_curr_x - hip_prev_x) * scale_s
                    dt_p = dt_s
                    if dt_p > 0:
                        v_trans_mps = abs(dx_p / dt_p)
            
            v_trans_mps = v_trans_mps * zoom_correction
            v_trans_mps = min(5.5, max(2.5, v_trans_mps))
            v_trans_kmh = v_trans_mps * 3.6
            
            # 3. Relative rotation speed of wrist
            v_wrist_rot_mps = max(5.0, (peak_hand_speed / 3.6) - v_trans_mps)
            omega_arm = v_wrist_rot_mps / R_arm
            
            # 4. Centrifugal + Tangential speed at release
            v_rel_mps = omega_arm * math.sqrt(2 * (R_arm + L_finger)**2 - R_arm**2)
            v_rel_kmh = v_rel_mps * 3.6
            
            # 5. Energy Whiplash Transfer
            I_forearm = (1.0 / 3.0) * m_forearm * (L_forearm ** 2)
            E_forearm = 0.5 * I_forearm * (omega_arm ** 2)
            
            eta_transfer = 0.85 * efficiency
            E_transfer = eta_transfer * E_forearm
            v_energy_mps = math.sqrt(2.0 * E_transfer / m_ball)
            v_energy_kmh = v_energy_mps * 3.6
            
            # 6. Total physics-based pitch speed
            v_physics_total_mps = v_rel_mps + v_trans_mps + v_energy_mps
            v_physics_total_kmh = v_physics_total_mps * 3.6

            # 5. Output JSON Summary Report (All in km/h)
            summary_data = {
                "session_pitch": "User_2D_Vector_Fusion",
                "pitch_speed_kmh": float(round(estimated_pitch_speed_kmh, 1)),
                "peak_hand_speed_kmh": float(round(peak_hand_speed, 1)),
                "fp_time": float(round(t_fp_plot, 3)),
                "br_time": float(round(t_release_plot, 3)),
                "throws": throws,
                "physics_details": {
                    "translation_kmh": float(round(v_trans_kmh, 1)),
                    "rotation_cf_kmh": float(round(v_rel_kmh, 1)),
                    "energy_transfer_kmh": float(round(v_energy_kmh, 1)),
                    "total_kmh": float(round(v_physics_total_kmh, 1))
                }
            }
            summary_path = os.path.join(base_dir, "video", "fused_3d_pitching.json")
            with open(summary_path, 'w', encoding='utf-8') as f_out:
                json.dump(summary_data, f_out, indent=2)
                
            # Also save as JS file to support file:// protocol (bypassing CORS fetch restriction)
            summary_path_js = os.path.join(base_dir, "video", "fused_3d_pitching.js")
            with open(summary_path_js, 'w', encoding='utf-8') as f_out:
                f_out.write(f"window.fused3dPitching = {json.dumps(summary_data, indent=2)};\n")
                
            self.root.after(0, self.on_success, peak_hand_speed)
        except Exception as e:
            self.root.after(0, self.on_failure, str(e))
            
    def on_success(self, speed):
        self.progress.config(value=100)
        self.lbl_status.config(text="분석 및 리포트 갱신 완료!", foreground="green")
        self.btn_run.config(state=tk.NORMAL)
        
        messagebox.showinfo("완료", f"2D 벡터 융합 분석이 무사히 종료되었습니다!\n\n최대 손목 속도: {speed:.1f} km/h")
        
        # Auto-open dashboard report
        if self.var_auto_open.get():
            try:
                webbrowser.open("http://localhost:8000/user_pitching_analysis.html")
            except Exception:
                pass
                
    def on_failure(self, error_msg):
        self.progress.config(value=0)
        self.lbl_status.config(text="분석 실패", foreground="red")
        self.btn_run.config(state=tk.NORMAL)
        messagebox.showerror("오류", f"피칭 데이터 처리 중 오류가 발생했습니다:\n{error_msg}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchingAnalyzerApp(root)
    root.mainloop()
