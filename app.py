#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_file, Response, render_template
import os, uuid, threading, json, time
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# task_id -> {"status": "pending/running/done/error", "progress": 0-100, "message": str, "output": str}
TASKS: dict[str, dict] = {}
TASKS_LOCK = threading.Lock()


# ─── 視覺化核心（從 music_visualizer.py 搬入）────────────────────────────────

WIDTH, HEIGHT = 1920, 1080
FPS = 30


# 預設主題：色票列表，每個元素為 (R, G, B)，依能量從低到高排列
PRESETS = {
    "rainbow": [(80,0,255),(0,200,255),(0,255,80),(255,255,0),(255,120,0),(255,0,0)],
    "fire":    [(20,0,80),(80,0,200),(200,40,0),(255,140,0),(255,220,0),(255,255,180)],
    "ocean":   [(0,10,60),(0,60,160),(0,160,220),(0,220,255),(120,240,255),(255,255,255)],
    "neon":    [(60,0,120),(140,0,200),(200,0,180),(255,0,120),(255,80,200),(255,200,255)],
    "forest":  [(0,30,0),(0,80,20),(0,160,40),(80,220,0),(200,255,0),(255,255,120)],
    "mono":    [(20,20,20),(60,60,80),(120,120,160),(180,180,220),(220,220,240),(255,255,255)],
}


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def energy_to_color(energy: float, stops=None) -> tuple:
    """將能量 0~1 映射到顏色，stops 為 RGB 列表（從低到高）。
    回傳 BGR tuple（OpenCV 格式）。"""
    if stops is None:
        stops = PRESETS["rainbow"]
    e = float(np.clip(energy, 0, 1))
    n = len(stops)
    if n == 1:
        r, g, b = stops[0]
        return (b, g, r)
    # 找到所在區段並線性插值
    seg = e * (n - 1)
    lo = int(seg)
    hi = min(lo + 1, n - 1)
    t = seg - lo
    r = int(stops[lo][0] * (1 - t) + stops[hi][0] * t)
    g = int(stops[lo][1] * (1 - t) + stops[hi][1] * t)
    b = int(stops[lo][2] * (1 - t) + stops[hi][2] * t)
    return (b, g, r)  # BGR


def _draw_beat_flash(img, is_beat, energy, cs):
    if is_beat and energy > 0.3:
        import cv2
        color = energy_to_color(energy, cs)
        thickness = int(energy * 20)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH - 1, HEIGHT - 1), color, thickness)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)


def render_bar(frame_idx, stft_frames, rms_energy, beat_frames, cs):
    import cv2
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    n_bins = stft_frames.shape[0]
    step = max(1, n_bins // 128)
    bars = stft_frames[::step, frame_idx][:128]
    bars = bars / (bars.max() + 1e-8)
    energy = float(rms_energy[min(frame_idx, len(rms_energy) - 1)])
    is_beat = frame_idx in beat_frames
    bar_w = WIDTH // len(bars)
    glow = 1.5 if is_beat else 1.0
    for i, h in enumerate(bars):
        bar_h = int(h * HEIGHT * 0.75 * glow)
        color = energy_to_color(h * glow, cs)
        x1, x2 = i * bar_w + 2, i * bar_w + bar_w - 4
        cv2.rectangle(img, (x1, HEIGHT - bar_h), (x2, HEIGHT), color, -1)
        glow_h = max(4, bar_h // 8)
        ov = img.copy()
        cv2.rectangle(ov, (x1, HEIGHT - bar_h), (x2, HEIGHT - bar_h + glow_h), color, -1)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    ref = img[HEIGHT // 2:, :].copy()
    ref = cv2.flip(ref, 0)
    alpha = np.linspace(0.4, 0, ref.shape[0])
    for row in range(ref.shape[0]):
        img[HEIGHT - ref.shape[0] + row, :] = (
            img[HEIGHT - ref.shape[0] + row, :] * (1 - alpha[row]) + ref[row] * alpha[row]
        ).astype(np.uint8)
    _draw_beat_flash(img, is_beat, energy, cs)
    return img


def render_circular(frame_idx, stft_frames, rms_energy, beat_frames, cs):
    import cv2
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    n_bins = stft_frames.shape[0]
    step = max(1, n_bins // 256)
    bars = stft_frames[::step, frame_idx][:256]
    bars = bars / (bars.max() + 1e-8)
    energy = float(rms_energy[min(frame_idx, len(rms_energy) - 1)])
    is_beat = frame_idx in beat_frames
    cx, cy = WIDTH // 2, HEIGHT // 2
    base_r = int(min(WIDTH, HEIGHT) * 0.22)
    bb = int(min(WIDTH, HEIGHT) * 0.04) if is_beat else 0
    cv2.circle(img, (cx, cy), base_r + bb, (40, 40, 40), 2)
    angles = np.linspace(0, 2 * np.pi, len(bars), endpoint=False)
    prev = None
    for angle, h in zip(angles, bars):
        r = base_r + bb + int(h * HEIGHT * 0.22)
        color = energy_to_color(h, cs)
        x, y = int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))
        if prev:
            cv2.line(img, prev, (x, y), color, 2)
        prev = (x, y)
    ball_r = 12 + int(energy * 30) + bb // 2
    ball_color = energy_to_color(energy, cs)
    for radius in range(ball_r, 0, -2):
        a = radius / ball_r * 0.8
        ov = img.copy()
        cv2.circle(ov, (cx, cy), radius, ball_color, -1)
        cv2.addWeighted(ov, a * 0.3, img, 1 - a * 0.3, 0, img)
    cv2.circle(img, (cx, cy), ball_r // 3, (255, 255, 255), -1)
    _draw_beat_flash(img, is_beat, energy, cs)
    return img


def render_waveform(frame_idx, audio_frames, rms_energy, beat_frames, cs):
    import cv2
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    energy = float(rms_energy[min(frame_idx, len(rms_energy) - 1)])
    is_beat = frame_idx in beat_frames
    samples = audio_frames[frame_idx]
    n = len(samples)
    for offset, af in [(0, 1.0), (-HEIGHT // 10, 0.4), (HEIGHT // 10, 0.4)]:
        prev = None
        for i, s in enumerate(samples):
            x = int(i / n * WIDTH)
            amplitude = HEIGHT * 0.35 * (1 + energy * 0.5)
            y = int(np.clip(HEIGHT // 2 + s * amplitude + offset, 0, HEIGHT - 1))
            color_dimmed = tuple(int(c * af) for c in energy_to_color(abs(s), cs))
            if prev:
                cv2.line(img, prev, (x, y), color_dimmed, 2 if offset == 0 else 1)
            prev = (x, y)
    cv2.line(img, (0, HEIGHT // 2), (WIDTH, HEIGHT // 2), energy_to_color(energy * 0.5, cs), 1)
    scan_x = int((frame_idx % FPS) / FPS * WIDTH)
    ov = img.copy()
    cv2.line(ov, (scan_x, 0), (scan_x, HEIGHT), (255, 255, 255), 3)
    cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)
    _draw_beat_flash(img, is_beat, energy, cs)
    return img


_particles_store: dict[str, list] = {}


class Particle:
    def __init__(self, x, y, vx, vy, color, life):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.color = color
        self.life = life
        self.max_life = life

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.vy += 0.3; self.vx *= 0.97; self.life -= 1

    @property
    def alive(self): return self.life > 0
    @property
    def alpha(self): return self.life / self.max_life


def render_particles(frame_idx, stft_frames, rms_energy, beat_frames, task_id, cs):
    import cv2
    particles = _particles_store.get(task_id, [])
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    energy = float(rms_energy[min(frame_idx, len(rms_energy) - 1)])
    is_beat = frame_idx in beat_frames
    cx, cy = WIDTH // 2, HEIGHT // 2
    if is_beat or energy > 0.6:
        n = int(60 * energy) + (100 if is_beat else 0)
        for _ in range(n):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(3, 12) * (1 + energy * 2)
            color = energy_to_color(np.random.uniform(energy * 0.5, 1.0), cs)
            particles.append(Particle(cx, cy, np.cos(angle) * speed, np.sin(angle) * speed, color, int(np.random.uniform(20, 60))))
    for _ in range(max(1, int(energy * 10))):
        color = energy_to_color(energy * np.random.uniform(0.3, 0.7), cs)
        particles.append(Particle(np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT), 0, -np.random.uniform(0.5, 2), color, 30))
    alive = []
    for p in particles:
        p.update()
        if p.alive and 0 <= int(p.x) < WIDTH and 0 <= int(p.y) < HEIGHT:
            cv2.circle(img, (int(p.x), int(p.y)), max(1, int(4 * p.alpha)), tuple(int(c * p.alpha) for c in p.color), -1)
            alive.append(p)
    _particles_store[task_id] = alive[-3000:]
    if is_beat:
        for r in range(120, 0, -10):
            ov = img.copy()
            cv2.circle(ov, (cx, cy), r, energy_to_color(energy, cs), -1)
            cv2.addWeighted(ov, r / 120 * 0.25, img, 1 - r / 120 * 0.25, 0, img)
    _draw_beat_flash(img, is_beat, energy, cs)
    return img


# ─── 音頻分析 ────────────────────────────────────────────────────────────────

def analyze_audio(path, task_id):
    import librosa
    _update(task_id, 5, "載入音檔...")
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    total_frames = int(duration * FPS)
    hop_length = sr // FPS

    _update(task_id, 15, "分析頻譜...")
    stft = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=2048))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    stft_norm = (stft_db - stft_db.min()) / (stft_db.max() - stft_db.min() + 1e-8)

    _update(task_id, 25, "偵測節拍...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_frames = set(beats.tolist())

    _update(task_id, 30, "計算能量...")
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm = rms / (rms.max() + 1e-8)

    audio_frames = []
    spf = hop_length
    for i in range(total_frames):
        start = i * spf
        chunk = y[start:start + spf * 4]
        if len(chunk) < spf * 4:
            chunk = np.pad(chunk, (0, spf * 4 - len(chunk)))
        audio_frames.append(chunk)

    return {
        "y": y, "sr": sr, "duration": duration,
        "total_frames": total_frames,
        "stft": stft_norm,
        "rms": rms_norm,
        "beat_frames": beat_frames,
        "audio_frames": audio_frames,
        "bpm": float(np.squeeze(tempo)),
    }


# ─── 渲染 ────────────────────────────────────────────────────────────────────

def render_video(data, style, output_path, task_id, audio_path):
    import subprocess, imageio_ffmpeg
    if task_id in _particles_store:
        del _particles_store[task_id]

    total = data["total_frames"]
    stft, rms, beats = data["stft"], data["rms"], data["beat_frames"]
    audio_frames = data["audio_frames"]

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", audio_path,
        "-vcodec", "libx264", "-preset", "fast", "-crf", "18",
        "-acodec", "aac", "-b:a", "192k",
        "-shortest", "-pix_fmt", "yuv420p",
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    _update(task_id, 35, f"渲染中 (BPM: {data['bpm']:.0f})...")

    cs = data.get("color_stops")
    try:
        for i in range(total):
            si = min(i, stft.shape[1] - 1)
            if style == 1:
                frame = render_bar(si, stft, rms, beats, cs)
            elif style == 2:
                frame = render_circular(si, stft, rms, beats, cs)
            elif style == 3:
                frame = render_waveform(i, audio_frames, rms, beats, cs)
            else:
                frame = render_particles(si, stft, rms, beats, task_id, cs)
            proc.stdin.write(frame.tobytes())

            if i % 30 == 0:
                pct = 35 + int((i / total) * 60)
                _update(task_id, pct, f"渲染中 {i}/{total} 幀...")

        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        raise RuntimeError("ffmpeg 管道錯誤")

    if task_id in _particles_store:
        del _particles_store[task_id]


# ─── 任務管理 ─────────────────────────────────────────────────────────────────

def _update(task_id, progress, message):
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id]["progress"] = progress
            TASKS[task_id]["message"] = message


def run_task(task_id, audio_path, style, output_path, color_stops):
    try:
        with TASKS_LOCK:
            TASKS[task_id] = {"status": "running", "progress": 0, "message": "準備中...", "output": ""}
        data = analyze_audio(audio_path, task_id)
        data["color_stops"] = color_stops
        render_video(data, style, output_path, task_id, audio_path)
        with TASKS_LOCK:
            TASKS[task_id] = {"status": "done", "progress": 100, "message": "完成！", "output": os.path.basename(output_path)}
    except Exception as e:
        with TASKS_LOCK:
            TASKS[task_id] = {"status": "error", "progress": 0, "message": str(e), "output": ""}


# ─── Flask 路由 ───────────────────────────────────────────────────────────────

ALLOWED = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "audio" not in request.files:
        return jsonify({"error": "未上傳音檔"}), 400
    file = request.files["audio"]
    style = int(request.form.get("style", 1))
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"不支援的格式：{ext}"}), 400

    # 解析顏色：preset 名稱 或 自訂 hex 列表 (JSON)
    color_param = request.form.get("colors", "rainbow")
    if color_param in PRESETS:
        color_stops = PRESETS[color_param]
    else:
        try:
            hex_list = json.loads(color_param)
            color_stops = [hex_to_rgb(h) for h in hex_list]
            if not color_stops:
                color_stops = PRESETS["rainbow"]
        except Exception:
            color_stops = PRESETS["rainbow"]

    task_id = str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    style_names = {1: "Bar", 2: "Circular", 3: "Waveform", 4: "Particles"}
    base = os.path.splitext(file.filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}_{base}_{style_names[style]}.mp4")
    file.save(audio_path)

    t = threading.Thread(target=run_task, args=(task_id, audio_path, style, output_path, color_stops), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})


@app.route("/progress/<task_id>")
def progress(task_id):
    def stream():
        while True:
            with TASKS_LOCK:
                info = TASKS.get(task_id, {"status": "pending", "progress": 0, "message": "等待中...", "output": ""})
            yield f"data: {json.dumps(info)}\n\n"
            if info["status"] in ("done", "error"):
                break
            time.sleep(0.5)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return "找不到檔案", 404
    return send_file(path, as_attachment=True, download_name=filename.split("_", 2)[-1])


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5001, threaded=True)
