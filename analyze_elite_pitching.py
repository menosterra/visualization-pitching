# -*- coding: utf-8 -*-
"""
Elite Pitching Mechanics Analysis Script
Data Source: Driveline OpenBiomechanics Project (OBP) (CC BY-NC-SA 4.0)
https://openbiomechanics.org/
"""
import os
import sys
import re
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

_ELITE_SERIES_CACHE = None

def get_elite_pitching_series():
    global _ELITE_SERIES_CACHE
    if _ELITE_SERIES_CACHE is not None:
        return _ELITE_SERIES_CACHE
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    js_path = os.path.join(base_dir, "pitching_data_elite.js")
    if not os.path.exists(js_path):
        js_path = os.path.join(base_dir, "elite_pitching_data.js")
    if not os.path.exists(js_path):
        js_path = os.path.join(base_dir, "real_motion_data.js")
        
    if not os.path.exists(js_path):
        return None
        
    with open(js_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    m = re.search(r'const\s+(?:realOBPMotionData|elitePitchingData|pitchingDataElite)\s*=\s*(\{[\s\S]*?\n\};)', text)
    if not m:
        m = re.search(r'=\s*(\{[\s\S]*\});', text)
    if not m:
        return None
        
    raw_json_str = m.group(1).rstrip(';')
    try:
        data = json.loads(raw_json_str)
    except Exception:
        # Fallback regex if trailing semicolon or structure differs
        m2 = re.search(r'const\s+(?:realOBPMotionData|elitePitchingData)\s*=\s*(\{[\s\S]*?\n\};)', text)
        if m2:
            data = json.loads(m2.group(1).rstrip(';'))
        else:
            return None
    
    frames = data["frames"]
    pitch_speed_mph = data.get("pitch_speed_mph", 94.4)
    pitch_speed_kmh = pitch_speed_mph * 1.60934
    
    pkh_time = data.get("pkh_time", 0.3333)
    fp_time = data.get("fp_time", 0.85)
    br_time = data.get("br_time", 0.9861)
    
    times = [f["t"] for f in frames]
    pkh_idx = np.argmin([abs(t - pkh_time) for t in times])
    
    raw_hip_angles = []
    raw_shoulder_angles = []
    wrist_speeds = []
    
    for i, f in enumerate(frames):
        j = f["joints"]
        hx = j["lead_hip"]["x"] - j["rear_hip"]["x"]
        hy = j["lead_hip"]["y"] - j["rear_hip"]["y"]
        sx = j["glove_shoulder"]["x"] - j["shoulder"]["x"]
        sy = j["glove_shoulder"]["y"] - j["shoulder"]["y"]
        
        raw_hip_angles.append(math.atan2(hx, -hy))
        raw_shoulder_angles.append(math.atan2(sx, -sy))
        
        if i > 0:
            dt = f["t"] - frames[i-1]["t"]
            if dt > 0:
                w_curr = j["wrist"]
                w_prev = frames[i-1]["joints"]["wrist"]
                dx = w_curr["x"] - w_prev["x"]
                dy = w_curr["y"] - w_prev["y"]
                dz = w_curr["z"] - w_prev["z"]
                spd_mps = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                wrist_speeds.append(spd_mps * 3.6)
            else:
                wrist_speeds.append(0.0)
        else:
            wrist_speeds.append(0.0)
            
    smoothed_speeds = []
    w_size = 5
    half_w = w_size // 2
    for idx in range(len(wrist_speeds)):
        start = max(0, idx - half_w)
        end = min(len(wrist_speeds), idx + half_w + 1)
        smoothed_speeds.append(sum(wrist_speeds[start:end]) / (end - start))
    wrist_speeds = smoothed_speeds
    peak_wrist_speed = max(wrist_speeds)
    
    hip_unwrapped = np.degrees(np.unwrap(raw_hip_angles))
    shoulder_unwrapped = np.degrees(np.unwrap(raw_shoulder_angles))
    
    h_base = hip_unwrapped[pkh_idx]
    s_base = shoulder_unwrapped[pkh_idx]
    
    hip_angles = list(hip_unwrapped - h_base)
    shoulder_angles = list(shoulder_unwrapped - s_base)
    separation_angles = list(np.array(shoulder_angles) - np.array(hip_angles))
    
    # Lead Knee Extension & Angular Velocity
    lead_knee_angles = []
    for f in frames:
        j = f["joints"]
        h = j["lead_hip"]
        k = j["lead_knee"]
        a = j["lead_ankle"]
        v1 = np.array([h["x"]-k["x"], h["y"]-k["y"], h["z"]-k["z"]])
        v2 = np.array([a["x"]-k["x"], a["y"]-k["y"], a["z"]-k["z"]])
        cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        lead_knee_angles.append(float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))))
        
    fp_idx = np.argmin([abs(t - fp_time) for t in times])
    br_idx = np.argmin([abs(t - br_time) for t in times])
    knee_angular_vels = list(np.gradient(lead_knee_angles, times))
    
    knee_angle_fp = lead_knee_angles[fp_idx]
    knee_angle_br = lead_knee_angles[br_idx]
    peak_knee_ext_vel = max(knee_angular_vels[fp_idx:br_idx+5])
    
    times_plot = [t - pkh_time for t in times]
    t_lift_plot = 0.0
    t_fp_plot = fp_time - pkh_time
    t_release_plot = br_time - pkh_time
    
    _ELITE_SERIES_CACHE = {
        "times_plot": times_plot,
        "hip_angles": hip_angles,
        "shoulder_angles": shoulder_angles,
        "separation_angles": separation_angles,
        "wrist_speeds": wrist_speeds,
        "lead_knee_angles": lead_knee_angles,
        "knee_angle_fp": knee_angle_fp,
        "knee_angle_br": knee_angle_br,
        "peak_knee_ext_vel": peak_knee_ext_vel,
        "t_lift_plot": t_lift_plot,
        "t_fp_plot": t_fp_plot,
        "t_release_plot": t_release_plot,
        "pitch_speed_kmh": pitch_speed_kmh,
        "peak_wrist_speed": peak_wrist_speed
    }
    return _ELITE_SERIES_CACHE

# Backwards compatibility alias
get_standard_pitching_series = get_elite_pitching_series

def run_elite_analysis():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    elite_data = get_elite_pitching_series()
    if not elite_data:
        print("오류: 엘리트 피칭 데이터를 로드할 수 없습니다.")
        return
        
    times_plot = elite_data["times_plot"]
    hip_angles = elite_data["hip_angles"]
    shoulder_angles = elite_data["shoulder_angles"]
    separation_angles = elite_data["separation_angles"]
    wrist_speeds = elite_data["wrist_speeds"]
    t_lift_plot = elite_data["t_lift_plot"]
    t_fp_plot = elite_data["t_fp_plot"]
    t_release_plot = elite_data["t_release_plot"]
    pitch_speed_kmh = elite_data["pitch_speed_kmh"]
    peak_wrist_speed = elite_data["peak_wrist_speed"]
    
    lead_knee_angles = elite_data["lead_knee_angles"]
    
    # 2. Generate Matplotlib Chart (3 Subplots: Rotation, Knee Angle, Wrist Velocity)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    
    # --- Subplot 1: Rotation Angles ---
    ax1.plot(times_plot, hip_angles, label="Hip Rotation", color="royalblue", linewidth=2.5)
    ax1.plot(times_plot, shoulder_angles, label="Shoulder Rotation", color="cyan", linewidth=2.5)
    ax1.plot(times_plot, separation_angles, label="Separation (X-Factor)", color="darkorange", linestyle="-", linewidth=2.5)
    
    # Baseline y=0 gray line with "Facing 3B" at right end
    ax1.axhline(y=0, color='#777777', linestyle='-', linewidth=1.1, alpha=0.85, zorder=1)
    ax1.text(0.985, 3.0, "Facing 3B", transform=ax1.get_yaxis_transform(),
             fontsize=9.0, color="#555555", va='bottom', ha='right', fontweight='semibold')
    
    # Event vertical lines in ax1 (all solid '-')
    ax1.axvline(x=t_lift_plot, color='gray', linestyle='-', linewidth=1.2, label='Leg Lift')
    ax1.axvline(x=t_fp_plot, color='green', linestyle='-', linewidth=1.2, label='Foot Plant')
    ax1.axvline(x=t_release_plot, color='purple', linestyle='-', linewidth=1.2, label='Ball Release')
    
    # --- Subplot 2: Lead Knee Angle (Extension / Block) ---
    ax2.plot(times_plot, lead_knee_angles, label="Lead Knee Angle", color="#0077b6", linewidth=2.5)
    ax2.axvline(x=t_lift_plot, color='gray', linestyle='-', linewidth=1.2)
    ax2.axvline(x=t_fp_plot, color='green', linestyle='-', linewidth=1.2)
    ax2.axvline(x=t_release_plot, color='purple', linestyle='-', linewidth=1.2)
    
    # --- Subplot 3: Wrist Velocity ---
    ax3.plot(times_plot, wrist_speeds, label="Wrist Velocity", color="#c1121f", linewidth=2.5)
    ax3.axvline(x=t_lift_plot, color='gray', linestyle='-', linewidth=1.2)
    ax3.axvline(x=t_fp_plot, color='green', linestyle='-', linewidth=1.2)
    ax3.axvline(x=t_release_plot, color='purple', linestyle='-', linewidth=1.2)
    
    # Calculate intersections at the 3 time events (Leg Lift, Foot Plant, Ball Release)
    events = [
        ("Leg Lift", t_lift_plot),
        ("Foot Plant", t_fp_plot),
        ("Ball Release", t_release_plot)
    ]
    
    # Intersections on ax1 (Angle graph)
    for label, t_ev in events:
        y_hip = float(np.interp(t_ev, times_plot, hip_angles))
        y_sh = float(np.interp(t_ev, times_plot, shoulder_angles))
        y_sep = float(np.interp(t_ev, times_plot, separation_angles))
        
        if label == "Leg Lift":
            # At Leg Lift, all are 0.0 deg
            ax1.plot(t_ev, 0.0, 'o', color='gray', markersize=4.5, zorder=5)
            ax1.text(t_ev + 0.015, 0.0, "0.0°", fontsize=8.5, color="#555555", va='center', ha='left', fontweight='semibold')
        else:
            # Foot Plant & Ball Release
            for y_val, offset_x, offset_y in [(y_hip, 0.012, 0), (y_sh, -0.012, 0), (y_sep, 0.012, 0)]:
                ax1.plot(t_ev, y_val, 'o', color='gray', markersize=4.5, zorder=5)
                ha = 'left' if offset_x > 0 else 'right'
                ax1.text(t_ev + offset_x, y_val + offset_y, f"{y_val:.1f}°", fontsize=8.5, color="#555555", va='center', ha=ha, fontweight='semibold')
    
    # Intersections on ax2 (Knee Angle graph)
    for label, t_ev in events:
        y_k = float(np.interp(t_ev, times_plot, lead_knee_angles))
        ax2.plot(t_ev, y_k, 'o', color='gray', markersize=4.5, zorder=5)
        offset_x = 0.012 if label != "Ball Release" else -0.015
        ha = 'left' if offset_x > 0 else 'right'
        ax2.text(t_ev + offset_x, y_k + 2.5, f"{y_k:.1f}°", fontsize=8.5, color="#555555", va='bottom', ha=ha, fontweight='semibold')
    
    # Intersections on ax3 (Wrist Velocity graph)
    for label, t_ev in events:
        y_w = float(np.interp(t_ev, times_plot, wrist_speeds))
        ax3.plot(t_ev, y_w, 'o', color='gray', markersize=4.5, zorder=5)
        offset_x = 0.015 if label != "Ball Release" else -0.015
        ha = 'left' if offset_x > 0 else 'right'
        ax3.text(t_ev + offset_x, y_w + 1.2, f"{y_w:.1f}", fontsize=8.5, color="#555555", va='bottom', ha=ha, fontweight='semibold')
        
        # Display event time x-value at the bottom of x-axis in gray
        ax3.text(t_ev, -0.065, f"{t_ev:.3f}s", transform=ax3.get_xaxis_transform(),
                 fontsize=8.5, color="#555555", ha='center', va='top', fontweight='semibold')
    
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("Elite Pitching Rotation Mechanics")
    min_y = min(-75.0, float(min(separation_angles)) - 20.0)
    max_y = max(180.0, float(max(shoulder_angles + hip_angles)) + 20.0)
    ax1.set_ylim(min_y, max_y)
    ax1.legend(loc="upper left")
    
    ax2.set_ylabel("Angle (deg)")
    ax2.set_title("Lead Knee Angle Profile (Extension / Block)")
    min_k = min(65.0, float(min(lead_knee_angles)) - 10.0)
    max_k = max(185.0, float(max(lead_knee_angles)) + 10.0)
    ax2.set_ylim(min_k, max_k)
    ax2.legend(loc="upper left")
    
    ax3.set_xticks([-0.2, 0.2, 0.4, 0.8])
    ax3.set_xlabel("Time (s)", labelpad=12)
    ax3.set_ylabel("Velocity (km/h)")
    ax3.set_title("Elite Wrist Velocity Profile")
    ax3.legend(loc="upper left")
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    
    video_dir = os.path.join(base_dir, "video")
    os.makedirs(video_dir, exist_ok=True)
    
    # Save both elite and legacy chart names
    chart_output_path = os.path.join(video_dir, "elite_mechanics_chart.png")
    plt.savefig(chart_output_path, dpi=150)
    legacy_chart_path = os.path.join(video_dir, "standard_biomechanics_chart.png")
    plt.savefig(legacy_chart_path, dpi=150)
    plt.close()
    print(f"차트 저장 완료: {chart_output_path}")
    
    # 3. Output Summary JSON & JS
    summary_data = {
        "session_pitch": "Elite_OBP_Pitching",
        "pitch_speed_kmh": float(round(pitch_speed_kmh, 1)),
        "peak_hand_speed_kmh": float(round(peak_wrist_speed, 1)),
        "fp_time": float(round(t_fp_plot, 3)),
        "br_time": float(round(t_release_plot, 3)),
        "knee_angle_fp": float(round(elite_data["knee_angle_fp"], 1)),
        "knee_angle_br": float(round(elite_data["knee_angle_br"], 1)),
        "peak_knee_ext_vel": float(round(elite_data["peak_knee_ext_vel"], 1))
    }
    
    # Save new elite summary files
    summary_json_path = os.path.join(video_dir, "elite_pitching_summary.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(summary_data, f_out, indent=2)
        
    summary_js_path = os.path.join(video_dir, "elite_pitching_summary.js")
    with open(summary_js_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"window.elitePitchingSummary = {json.dumps(summary_data, indent=2)};\n")
        f_out.write(f"window.standardPitchingSummary = window.elitePitchingSummary;\n")
        
    # Save legacy summary files for compatibility
    legacy_json_path = os.path.join(video_dir, "standard_pitching_summary.json")
    with open(legacy_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(summary_data, f_out, indent=2)
        
    legacy_js_path = os.path.join(video_dir, "standard_pitching_summary.js")
    with open(legacy_js_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"window.standardPitchingSummary = {json.dumps(summary_data, indent=2)};\n")
        
    print(f"요약 데이터 저장 완료: {summary_json_path}, {summary_js_path}")

run_standard_analysis = run_elite_analysis

if __name__ == "__main__":
    run_elite_analysis()
