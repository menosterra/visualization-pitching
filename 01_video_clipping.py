import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from moviepy import VideoFileClip
import cv2
from PIL import Image, ImageTk

def parse_time(time_str):
    """
    시간 문자열을 초(seconds) 단위의 실수로 변환합니다.
    형식:
    - 초 단위 (예: 90, 120.5)
    - 분:초 (예: 1:30, 02:45.5)
    - 시:분:초 (예: 01:20:15)
    """
    time_str = time_str.strip()
    if not time_str:
        return None
    
    # 1. 단순 숫자인지 확인 (예: 90, 120.5)
    try:
        val = float(time_str)
        if val < 0:
            raise ValueError("시간은 음수가 될 수 없습니다.")
        return val
    except ValueError:
        pass
    
    # 2. 콜론(:)으로 구분된 경우 확인 (MM:SS 또는 HH:MM:SS)
    parts = time_str.split(':')
    try:
        if len(parts) == 2:  # MM:SS
            minutes = float(parts[0])
            seconds = float(parts[1])
            if minutes < 0 or seconds < 0:
                raise ValueError
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            if hours < 0 or minutes < 0 or seconds < 0:
                raise ValueError
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError
    except ValueError:
        raise ValueError("올바르지 않은 시간 형식입니다. (예: 90, 1:30, 01:02:30)")

def format_seconds(seconds):
    """초(seconds)를 시:분:초 형태로 보기 좋게 포맷팅합니다."""
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
    (예: video_clip.mp4 -> video_clip-1.mp4 -> video_clip-2.mp4)
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

class VideoClipperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("동영상 구간 자르기 (Video Clipper)")
        self.root.geometry("1180x680")
        self.root.resizable(False, False)
        
        # 변수 정의
        self.input_path = ""
        self.output_path = ""
        self.video_duration = 0.0
        self.cap = None
        self.total_frames = 0
        self.fps = 0.0
        self.current_frame_idx = 0
        self.playing = False
        self.play_timer = None
        self._is_updating_slider = False
        self._seek_after_id = None
        self.speed_var = tk.StringVar(value="0.5x")
        
        # UI 스타일 설정
        self.style = ttk.Style()
        self.style.theme_use("clam")  # 깔끔한 기본 테마
        
        # 전체 폰트 설정
        self.default_font = ("맑은 고딕", 9)
        self.root.option_add("*Font", self.default_font)
        
        self.create_widgets()
        
        # 종료 처리 바인딩
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_widgets(self):
        # 전체 패딩용 프레임
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 좌측 패널 (미리보기 및 재생 컨트롤)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # 비디오 미리보기 프레임
        preview_lf = ttk.LabelFrame(left_panel, text=" 동영상 미리보기 ", padding="10")
        preview_lf.pack(fill=tk.BOTH, expand=True)
        
        # 미리보기 이미지 라벨 (검은 배경 프레임 안에 배치 - 640x360 대형 패널)
        preview_bg = tk.Frame(preview_lf, bg="black", width=640, height=360)
        preview_bg.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        preview_bg.pack_propagate(False) # 고정 크기 유지
        
        self.lbl_preview = ttk.Label(preview_bg, text="동영상을 로드하면 여기에 표시됩니다.", foreground="white", background="black", anchor=tk.CENTER)
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)
        
        # 재생바 (ttk.Scale)
        self.scale_play = ttk.Scale(preview_lf, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.scale_play.pack(fill=tk.X, pady=(0, 10))
        self.scale_play.config(state=tk.DISABLED)
        
        # 컨트롤 버튼 프레임
        ctrl_frame = ttk.Frame(preview_lf)
        ctrl_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_prev = ttk.Button(ctrl_frame, text="◀ 1프레임", command=self.prev_frame, state=tk.DISABLED, width=11)
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 6))
        
        self.btn_play = ttk.Button(ctrl_frame, text="▶ 재생", command=self.toggle_play, state=tk.DISABLED, width=11)
        self.btn_play.pack(side=tk.LEFT, padx=(0, 6))
        
        self.btn_next = ttk.Button(ctrl_frame, text="1프레임 ▶", command=self.next_frame, state=tk.DISABLED, width=11)
        self.btn_next.pack(side=tk.LEFT, padx=(0, 10))
        
        # 배속 선택 Combobox
        ttk.Label(ctrl_frame, text="속도:").pack(side=tk.LEFT, padx=(0, 3))
        self.cb_speed = ttk.Combobox(ctrl_frame, textvariable=self.speed_var, values=["0.25x", "0.5x", "1.0x"], width=6, state="readonly")
        self.cb_speed.pack(side=tk.LEFT)
        
        # 시간/프레임 정보 표시
        self.lbl_time_info = ttk.Label(preview_lf, text="프레임: - / -  |  시간: --:-- / --:--", font=("맑은 고딕", 9, "bold"))
        self.lbl_time_info.pack(side=tk.RIGHT, pady=5)
        
        # 우측 패널 (파일 설정 및 자르기 실행)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 1. 파일 선택 영역
        file_frame = ttk.LabelFrame(right_panel, text=" 1. 원본 동영상 파일 선택 ", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_file_path = ttk.Label(file_frame, text="선택된 파일 없음", width=38, anchor=tk.W, foreground="gray")
        self.lbl_file_path.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        
        btn_select = ttk.Button(file_frame, text="파일 선택", command=self.select_file)
        btn_select.pack(side=tk.RIGHT)
        
        # 2. 동영상 정보 표시 영역
        self.info_frame = ttk.LabelFrame(right_panel, text=" 2. 동영상 정보 ", padding="10")
        self.info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_info = ttk.Label(self.info_frame, text="파일을 선택하면 정보가 표시됩니다.", anchor=tk.CENTER, foreground="gray")
        self.lbl_info.pack(fill=tk.X)
        
        # 3. 시간 설정 영역
        time_frame = ttk.LabelFrame(right_panel, text=" 3. 자를 구간 설정 ", padding="10")
        time_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 가이드 텍스트
        lbl_guide = ttk.Label(time_frame, text="입력 형식: 90 (초단위) | 1:30 (분:초) | 01:02:30 (시:분:초)", foreground="blue")
        lbl_guide.grid(row=0, column=0, columnspan=3, pady=(0, 8), sticky=tk.W)
        
        # 시작 시간
        ttk.Label(time_frame, text="시작 시간:").grid(row=1, column=0, sticky=tk.E, padx=(5, 5))
        self.ent_start = ttk.Entry(time_frame, width=15)
        self.ent_start.grid(row=1, column=1, sticky=tk.W, padx=(0, 10))
        self.ent_start.insert(0, "0")
        
        self.btn_set_start = ttk.Button(time_frame, text="시작 위치 설정", command=self.set_start_time, state=tk.DISABLED)
        self.btn_set_start.grid(row=1, column=2, sticky=tk.W)
        
        # 종료 시간
        ttk.Label(time_frame, text="종료 시간:").grid(row=2, column=0, sticky=tk.E, padx=(5, 5), pady=(5, 0))
        self.ent_end = ttk.Entry(time_frame, width=15)
        self.ent_end.grid(row=2, column=1, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        
        self.btn_set_end = ttk.Button(time_frame, text="종료 위치 설정", command=self.set_end_time, state=tk.DISABLED)
        self.btn_set_end.grid(row=2, column=2, sticky=tk.W, pady=(5, 0))
        
        # 4. 저장 파일 설정 영역
        save_frame = ttk.LabelFrame(right_panel, text=" 4. 저장 위치 설정 ", padding="10")
        save_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.lbl_save_path = ttk.Label(save_frame, text="저장할 위치를 지정하세요.", width=38, anchor=tk.W, foreground="gray")
        self.lbl_save_path.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        
        self.btn_save_as = ttk.Button(save_frame, text="경로 변경", command=self.select_output_path, state=tk.DISABLED)
        self.btn_save_as.pack(side=tk.RIGHT)
        
        # 5. 실행 및 진행 상태 영역
        action_frame = ttk.Frame(right_panel)
        action_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.progress = ttk.Progressbar(action_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_status = ttk.Label(action_frame, text="대기 중...", font=("맑은 고딕", 10, "bold"), foreground="dimgray")
        self.lbl_status.pack(side=tk.LEFT, padx=5)
        
        self.btn_clip = ttk.Button(action_frame, text="동영상 자르기 시작", command=self.start_clipping, state=tk.DISABLED)
        self.btn_clip.pack(side=tk.RIGHT, ipadx=10, ipady=5)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="동영상 파일 선택",
            filetypes=[("동영상 파일", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm")]
        )
        if not file_path:
            return
            
        self.input_path = os.path.abspath(file_path)
        self.lbl_file_path.config(text=self.input_path, foreground="black")
        
        # 비디오 정보 읽기
        try:
            self.lbl_status.config(text="동영상 정보 읽는 중...", foreground="blue")
            self.root.update()
            
            # OpenCV로 비디오 로드 및 프레임 정보 추출
            self.load_video()
            
            # 정보 레이블 업데이트
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info_text = f"길이: {format_seconds(self.video_duration)} ({self.video_duration:.2f}초)  |  해상도: {width}x{height}  |  FPS: {self.fps:.2f}"
            self.lbl_info.config(text=info_text, foreground="black")
            
            # 종료 시간 기본 입력 (영상 끝까지)
            self.ent_end.delete(0, tk.END)
            self.ent_end.insert(0, f"{self.video_duration:.2f}")
            
            # 기본 저장 경로 설정 (동일 폴더 내 _clip 추가 및 중복 시 -1, -2 자동 부여)
            dir_name, file_name = os.path.split(self.input_path)
            base_name, ext = os.path.splitext(file_name)
            default_out = os.path.join(dir_name, f"{base_name}_clip{ext}")
            self.output_path = get_unique_path(default_out)
            self.lbl_save_path.config(text=self.output_path, foreground="black")
            
            # 버튼 활성화
            self.btn_save_as.config(state=tk.NORMAL)
            self.btn_clip.config(state=tk.NORMAL)
            self.lbl_status.config(text="준비 완료", foreground="green")
            
        except Exception as e:
            messagebox.showerror("오류", f"동영상 정보를 읽는데 실패했습니다:\n{str(e)}")
            self.lbl_status.config(text="파일 로드 실패", foreground="red")
            self.btn_save_as.config(state=tk.DISABLED)
            self.btn_clip.config(state=tk.DISABLED)
            self.lbl_info.config(text="파일을 선택하면 정보가 표시됩니다.", foreground="gray")
            self.lbl_file_path.config(text="선택된 파일 없음", foreground="gray")

    def load_video(self, start_frame_idx=None):
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise Exception("비디오 파일을 열 수 없습니다.")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
            
        self.video_duration = self.total_frames / self.fps
        
        # 시작 프레임 복원 설정
        if start_frame_idx is not None and start_frame_idx < self.total_frames:
            self.current_frame_idx = start_frame_idx
        else:
            self.current_frame_idx = 0
        
        # 슬라이더 범위 업데이트
        self._is_updating_slider = True
        self.scale_play.config(to=self.total_frames - 1)
        self.scale_play.config(state=tk.NORMAL)
        self.scale_play.set(self.current_frame_idx)
        self._is_updating_slider = False
        
        # 버튼 활성화
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)
        self.btn_play.config(state=tk.NORMAL)
        self.btn_set_start.config(state=tk.NORMAL)
        self.btn_set_end.config(state=tk.NORMAL)
        
        self.show_frame(self.current_frame_idx)

    def show_frame(self, frame_idx, sequential=False):
        if not self.cap:
            return
        
        try:
            # 순차 재생이 아니거나 프레임 차이가 있으면 탐색(Seek) 수행
            if not sequential or abs(self.current_frame_idx - frame_idx) > 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
            self.current_frame_idx = frame_idx
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                max_w, max_h = 640, 360
                ratio = min(max_w / w, max_h / h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                self.lbl_preview.config(image=photo)
                self.lbl_preview.image = photo
                
                curr_sec = frame_idx / self.fps
                self.lbl_time_info.config(
                    text=f"프레임: {frame_idx} / {self.total_frames - 1}  |  시간: {format_seconds(curr_sec)} / {format_seconds(self.video_duration)}"
                )
        except Exception as e:
            print(f"프레임 프리뷰 생성 중 오류: {e}")

    def on_slider_move(self, val):
        if not self.cap or self._is_updating_slider:
            return
        frame_idx = int(float(val))
        if frame_idx == self.current_frame_idx:
            return
            
        # 연속 스크롤 이벤트 디바운싱
        if self._seek_after_id:
            self.root.after_cancel(self._seek_after_id)
        self._seek_after_id = self.root.after(15, lambda: self.show_frame(frame_idx))

    def prev_frame(self):
        if self.cap and self.current_frame_idx > 0:
            target_frame = self.current_frame_idx - 1
            self._is_updating_slider = True
            self.scale_play.set(target_frame)
            self._is_updating_slider = False
            self.show_frame(target_frame)
            
    def next_frame(self):
        if self.cap and self.current_frame_idx < self.total_frames - 1:
            target_frame = self.current_frame_idx + 1
            self._is_updating_slider = True
            self.scale_play.set(target_frame)
            self._is_updating_slider = False
            self.show_frame(target_frame, sequential=True)

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
            
            self._is_updating_slider = True
            self.scale_play.set(next_frame)
            self._is_updating_slider = False
            
            self.show_frame(next_frame, sequential=True)
            
            # 배속(0.25x / 0.5x / 1.0x, 기본 0.5x) 반영
            try:
                speed_str = self.speed_var.get().replace("x", "")
                speed_val = float(speed_str)
            except:
                speed_val = 0.5
                
            delay = max(10, int(1000 / (self.fps * speed_val)))
            self.play_timer = self.root.after(delay, self.play_loop)
        else:
            self.pause()

    def set_start_time(self):
        if self.cap:
            curr_sec = self.current_frame_idx / self.fps
            self.ent_start.delete(0, tk.END)
            self.ent_start.insert(0, f"{curr_sec:.2f}")

    def set_end_time(self):
        if self.cap:
            curr_sec = self.current_frame_idx / self.fps
            self.ent_end.delete(0, tk.END)
            self.ent_end.insert(0, f"{curr_sec:.2f}")
            
    def select_output_path(self):
        if not self.input_path:
            return
            
        dir_name, file_name = os.path.split(self.output_path)
        file_path = filedialog.asksaveasfilename(
            title="저장할 동영상 파일 경로 선택",
            initialdir=dir_name,
            initialfile=file_name,
            filetypes=[("MP4 비디오", "*.mp4"), ("AVI 비디오", "*.avi"), ("MKV 비디오", "*.mkv"), ("모든 파일", "*.*")]
        )
        if file_path:
            self.output_path = os.path.abspath(file_path)
            self.lbl_save_path.config(text=self.output_path, foreground="black")

    def start_clipping(self):
        if not self.input_path:
            messagebox.showerror("오류", "동영상 파일을 먼저 선택해주세요.")
            return
            
        # 시간 파싱 및 유효성 검사
        try:
            start_val = self.ent_start.get().strip()
            start_sec = parse_time(start_val) if start_val else 0.0
        except ValueError as e:
            messagebox.showerror("입력 오류", f"시작 시간 형식이 올바르지 않습니다.\n{str(e)}")
            return
            
        try:
            end_val = self.ent_end.get().strip()
            end_sec = parse_time(end_val) if end_val else self.video_duration
        except ValueError as e:
            messagebox.showerror("입력 오류", f"종료 시간 형식이 올바르지 않습니다.\n{str(e)}")
            return
            
        if start_sec < 0:
            messagebox.showerror("입력 오류", "시작 시간은 0초 이상이어야 합니다.")
            return
            
        if start_sec >= self.video_duration:
            messagebox.showerror("입력 오류", f"시작 시간은 동영상 전체 길이({self.video_duration:.2f}초)보다 작아야 합니다.")
            return
            
        if end_sec <= start_sec:
            messagebox.showerror("입력 오류", "종료 시간은 시작 시간보다 커야 합니다.")
            return
            
        if end_sec > self.video_duration:
            end_sec = self.video_duration
            
        # Ensure output file path uniqueness before starting
        self.output_path = get_unique_path(self.output_path)
        self.lbl_save_path.config(text=self.output_path, foreground="black")
            
        # UI 비활성화 및 비디오 재생 일시정지, 비디오 캡처 임시 릴리즈
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.btn_clip.config(state=tk.DISABLED)
        self.btn_save_as.config(state=tk.DISABLED)
        self.lbl_status.config(text="동영상 자르는 중... 잠시만 기다려주세요.", foreground="darkorange")
        self.progress.start(10)
        
        # 작업 스레드 시작
        worker = threading.Thread(target=self.clip_worker, args=(start_sec, end_sec), daemon=True)
        worker.start()

    def clip_worker(self, start_sec, end_sec):
        try:
            clip = VideoFileClip(self.input_path)
            sub_clip = clip.subclipped(start_sec, end_sec)
            
            has_audio = clip.audio is not None
            temp_audio = None
            if has_audio:
                out_dir = os.path.dirname(self.output_path)
                temp_audio = os.path.join(out_dir, "temp_audio.mp4")
                if os.path.exists(temp_audio):
                    try:
                        os.remove(temp_audio)
                    except Exception:
                        pass
            
            sub_clip.write_videofile(
                self.output_path,
                codec="libx264",
                audio=has_audio,
                audio_codec="aac" if has_audio else None,
                temp_audiofile=temp_audio,
                logger=None
            )
            
            sub_clip.close()
            clip.close()
            
            self.root.after(0, self.on_success)
        except Exception as e:
            self.root.after(0, self.on_failure, str(e))

    def on_success(self):
        self.progress.stop()
        self.btn_clip.config(state=tk.NORMAL)
        self.btn_save_as.config(state=tk.NORMAL)
        self.lbl_status.config(text="자르기 완료!", foreground="green")
        
        # 비디오 캡처 다시 로드
        try:
            self.load_video(self.current_frame_idx)
        except Exception:
            pass
            
        messagebox.showinfo("완료", f"동영상 자르기가 성공적으로 끝났습니다!\n\n저장 경로: {self.output_path}")

    def on_failure(self, error_msg):
        self.progress.stop()
        self.btn_clip.config(state=tk.NORMAL)
        self.btn_save_as.config(state=tk.NORMAL)
        self.lbl_status.config(text="작업 실패", foreground="red")
        
        # 비디오 캡처 다시 로드
        try:
            self.load_video(self.current_frame_idx)
        except Exception:
            pass
            
        messagebox.showerror("오류", f"동영상 처리 중 오류가 발생했습니다:\n{error_msg}")

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
    app = VideoClipperApp(root)
    root.mainloop()
