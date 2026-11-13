#!/usr/bin/env python3
"""
本机逐帧「人体部位」打码工具 —— 识别人体骨架,按部位打码(视频不会上传到任何地方)

原理:
    用 MediaPipe 姿态模型找出每个人的 33 个骨架关键点(头、肩、肘、腕、髋、膝、踝……),
    再按你选的部位生成打码区域。背对镜头、低头、戴口罩时,头的位置照样能定位。
    默认还会叠加一个人脸检测器作为补充(两者取并集,宁可多打),可用 --no-face 关闭。

依赖:
    pip install mediapipe opencv-python numpy
    ffmpeg(可选,用于保留音轨并压成 H.264)

用法:
    python body_blur.py 视频.mp4                               # 默认只打头部
    python body_blur.py 视频.mp4 --parts 头,右臂                 # 头 + 右手臂(比如有纹身)
    python body_blur.py 视频.mp4 --parts head,right_forearm,chest
    python body_blur.py 视频.mp4 --parts 头 --strict --review --mode solid   # 最稳妥
    python body_blur.py --list-parts                            # 查看所有可选部位

可选部位(中英文都可以,逗号分隔):
    头 head
    右臂 right_arm   = 右上臂 right_upper_arm + 右前臂 right_forearm + 右手 right_hand
    左臂 left_arm    = 左上臂 left_upper_arm  + 左前臂 left_forearm  + 左手 left_hand
    右腿 right_leg   = 右大腿 right_thigh + 右小腿 right_calf + 右脚 right_foot
    左腿 left_leg    = 左大腿 left_thigh  + 左小腿 left_calf  + 左脚 left_foot
    躯干 torso    胸部 chest    下体/臀部 crotch
    手臂 arms(双臂)   手 hands   腿 legs   全身 body
    注意:左右指的是画面中那个人自己的左右,不是画面的左右。
          人背对镜头时模型偶尔会把左右搞反;纹身等关键位置建议直接选双侧(如 arms)。

只给其中几个人打码(--pick):
    检测完成后弹出「人物选择」面板:每个人一张卡片(截图 + 骨架),默认全部打码,
    取消勾选的人不打码。同一个人的多段画面会按人脸自动归成一组;
    没露脸、或认不准的片段单独列出,默认打码(宁可多打)。
    两个人靠得很近时,为防止编号对调,两个人都会打码。

人工检查(--review)按键:
    空格 播放/暂停      a / d 上一帧/下一帧      s / w 后退/前进 1 秒
    m    跳到下一段「完全没有打码」的帧(Shift+M 上一段)
    n    跳到下一段「人数比前后少」的帧(Shift+N 上一段;已自动补遮,建议抽查)
    t    跳到下一段「不打码的人和别人靠得很近」的帧(Shift+T 上一段;编号可能对调,务必检查)
    k    显示/隐藏骨架(看模型找得对不对)
    鼠标右键点人:切换这个人打码/不打码(需开启 --pick)
    鼠标拖框:补一个打码区域。拖一次后前进到结束位置按 回车 = 这段固定位置打码;
             在多个帧分别拖框 = 中间自动插值跟随移动
    u    撤销上一个手动标记          q 保存并输出

模型在首次运行时自动下载(姿态模型约 9~30MB,人脸模型约 230KB,
人脸识别模型约 37MB 仅在 --pick 时需要),之后可完全断网使用。
注意:开启 --pick 时,检测缓存 .pose.json 里会保存人脸特征,删除该文件即可清除。
"""
import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

FIX_CMD = ("python -m pip uninstall -y opencv-python opencv-contrib-python numpy mediapipe\npython -m pip install mediapipe")

try:
    import cv2
    import numpy as np
except ImportError as _e:
    _msg = ("缺少或损坏的依赖库:%s\n\n请打开「命令提示符」依次运行下面两行,然后重新打开本程序:\n\n%s"
            % (_e, FIX_CMD))
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        _r = _tk.Tk()
        _r.withdraw()
        _mb.showerror("缺少依赖库", _msg)
    except Exception:
        pass
    print(_msg)
    if sys.stdin is not None and sys.stdin.isatty():
        input("按回车退出")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
FACE_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".flv", ".webm"}
ROT_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}

# BlazePose 33 个关键点编号(左右 = 被拍者自己的左右)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HAND, R_HAND = (15, 17, 19, 21), (16, 18, 20, 22)
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_FOOT, R_FOOT = (27, 29, 31), (28, 30, 32)
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]

BASE_PARTS = ["head", "torso", "chest", "crotch",
              "right_upper_arm", "right_forearm", "right_hand",
              "left_upper_arm", "left_forearm", "left_hand",
              "right_thigh", "right_calf", "right_foot",
              "left_thigh", "left_calf", "left_foot"]
GROUPS = {
    "right_arm": ["right_upper_arm", "right_forearm", "right_hand"],
    "left_arm": ["left_upper_arm", "left_forearm", "left_hand"],
    "right_leg": ["right_thigh", "right_calf", "right_foot"],
    "left_leg": ["left_thigh", "left_calf", "left_foot"],
    "arms": ["right_arm", "left_arm"], "hands": ["right_hand", "left_hand"],
    "legs": ["right_leg", "left_leg"],
    "body": ["head", "torso", "arms", "legs", "crotch"],
}
ALIASES = {
    "头": "head", "头部": "head", "脸": "head",
    "躯干": "torso", "身体": "torso", "胸": "chest", "胸部": "chest",
    "下体": "crotch", "裆": "crotch", "裆部": "crotch", "臀部": "crotch", "屁股": "crotch",
    "右臂": "right_arm", "右手臂": "right_arm", "左臂": "left_arm", "左手臂": "left_arm",
    "右上臂": "right_upper_arm", "右大臂": "right_upper_arm",
    "左上臂": "left_upper_arm", "左大臂": "left_upper_arm",
    "右前臂": "right_forearm", "右小臂": "right_forearm",
    "左前臂": "left_forearm", "左小臂": "left_forearm",
    "右手": "right_hand", "左手": "left_hand", "手": "hands", "双手": "hands",
    "手臂": "arms", "双臂": "arms",
    "右腿": "right_leg", "左腿": "left_leg", "腿": "legs", "双腿": "legs",
    "右大腿": "right_thigh", "左大腿": "left_thigh",
    "右小腿": "right_calf", "左小腿": "left_calf", "右脚": "right_foot", "左脚": "left_foot",
    "全身": "body",
}


def parse_parts(text):
    out = []

    def add(name):
        name = ALIASES.get(name, name)
        if name in GROUPS:
            for n in GROUPS[name]:
                add(n)
        elif name in BASE_PARTS:
            if name not in out:
                out.append(name)
        else:
            sys.exit(f"不认识的部位: {name}(用 --list-parts 查看)")
    for t in text.replace("\uff0c", ",").replace("\u3001", ",").replace(" ", ",").split(","):
        if t.strip():
            add(t.strip().lower())
    return out


LOG = print          # 界面模式下会被替换,把提示显示到界面日志里


def fetch(url, name):
    path = os.path.join(HERE, name)
    if not os.path.isfile(path) or os.path.getsize(path) < 10000:
        LOG(f"正在下载模型 {name}(仅首次,请稍候)...")
        urllib.request.urlretrieve(url, path)
        if os.path.getsize(path) < 10000:
            sys.exit(f"模型下载失败,请手动下载放到脚本同目录:\n  {url}")
    return path


# ---------------------------------------------------------------- 检测

def map_point_back(x, y, rot, W, H):
    """旋转后图像里的点 → 原图坐标(W, H 为原图尺寸)"""
    if rot == 0:
        return x, y
    if rot == 90:
        return y, H - x
    if rot == 180:
        return W - x, H - y
    return W - y, x


class Detector:
    def __init__(self, args, fps):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        model = args.pose_model or fetch(POSE_URL.format(v=args.model_size),
                                         f"pose_landmarker_{args.model_size}.task")
        self.rotations = [0, 90, 180, 270] if args.rotate else [0]
        self.pose = {}
        for rot in self.rotations:     # 每个方向一个独立实例(视频模式要求时间戳连续)
            opts = vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=model),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=args.max_people,
                min_pose_detection_confidence=args.conf,
                min_pose_presence_confidence=args.conf,
                min_tracking_confidence=args.conf)
            self.pose[rot] = vision.PoseLandmarker.create_from_options(opts)
        self.face = None
        if args.face:
            fm = args.face_model or fetch(FACE_URL, "face_detection_yunet_2023mar.onnx")
            self.face = cv2.FaceDetectorYN.create(fm, "", (320, 320), args.face_conf, 0.3, 5000)
        # 人脸识别(只在「选择要打码的人」时需要):每秒抽约 5 帧,给够大的人脸算特征,用于按人分组
        self.rec = None
        if args.face and getattr(args, "pick", False):
            rm = getattr(args, "sface_model", None) or fetch(
                SFACE_URL, "face_recognition_sface_2021dec.onnx")
            self.rec = cv2.FaceRecognizerSF.create(rm, "")
        self.embed_every = max(1, int(round(fps / 5)))
        self.max_side = args.max_side
        self.fps = fps

    def __call__(self, frame, idx):
        H, W = frame.shape[:2]
        ts = int(idx * 1000 / self.fps)
        poses = []
        for rot in self.rotations:
            img = frame if rot == 0 else cv2.rotate(frame, ROT_CODE[rot])
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.pose[rot].detect_for_video(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
            for lms in res.pose_landmarks:
                pts = []
                for lm in lms:
                    x, y = map_point_back(lm.x * w, lm.y * h, rot, W, H)
                    vis = lm.visibility if lm.visibility is not None else 1.0
                    pts.append([round(x, 1), round(y, 1), round(float(vis), 2)])
                poses.append(pts)
        faces = []
        if self.face is not None:
            faces = detect_faces(self.face, frame, self.rotations, self.max_side)
        out = {"p": poses, "f": faces}
        if self.rec is not None and idx % self.embed_every == 0:
            emb = [face_feature(self.rec, frame, f) for f in faces]
            if any(emb):
                out["e"] = emb
        return out


MIN_FACE_FOR_ID = 40      # 小于这个像素的人脸认不准,不参与按人分组


def face_feature(rec, frame, f):
    """人脸特征(128 维,已归一化),压成 float16 + base64 存进缓存。脸太小或没有关键点时返回 None。"""
    if len(f) < 16 or min(f[2], f[3]) < MIN_FACE_FOR_ID:
        return None
    cx, cy, w, h = f[:4]
    row = np.array([cx - w / 2, cy - h / 2, w, h, *f[6:16], f[5]], np.float32)
    try:
        v = rec.feature(rec.alignCrop(frame, row)).astype(np.float32).ravel()
    except cv2.error:
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return base64.b64encode((v / n).astype(np.float16).tobytes()).decode("ascii")


def decode_feature(s):
    v = np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-6)


def detect_faces(det, frame, rotations, max_side):
    H0, W0 = frame.shape[:2]
    scale = min(1.0, max_side / max(H0, W0))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    H, W = small.shape[:2]
    out = []
    for rot in rotations:
        img = small if rot == 0 else cv2.rotate(small, ROT_CODE[rot])
        det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = det.detect(img)
        if faces is None:
            continue
        for f in faces:
            x, y, w, h = (float(v) for v in f[:4])
            cx, cy = map_point_back(x + w / 2, y + h / 2, rot, W, H)
            ang = math.degrees(math.atan2(float(f[7] - f[5]), float(f[6] - f[4]))) - rot
            # [cx, cy, w, h, 角度, 置信度, 5 个关键点(右眼 左眼 鼻 右嘴角 左嘴角)],都是原图坐标
            lm = []
            for k in range(5):
                lx, ly = map_point_back(float(f[4 + 2 * k]), float(f[5 + 2 * k]), rot, W, H)
                lm += [lx / scale, ly / scale]
            out.append([round(v, 1) for v in (cx / scale, cy / scale, w / scale, h / scale, ang)]
                       + [round(float(f[14]), 2)] + [round(v, 1) for v in lm])
    return out


def detect_all(src, args, progress=None, log=print):
    cache = os.path.splitext(src)[0] + ".pose.json"
    key = {k: getattr(args, k) for k in
           ("conf", "face_conf", "face", "rotate", "max_people", "model_size", "max_side")}
    need_embed = bool(args.face and getattr(args, "pick", False))
    if os.path.isfile(cache) and not args.redetect:
        try:
            with open(cache, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("key") == key:
                if need_embed and not data.get("embed"):
                    log("  选择打码的人需要人脸特征,之前的检测结果里没有,重新检测一次")
                else:
                    log("  复用已有检测结果(换部位不用重新检测)")
                    return data["frames"]
        except (ValueError, KeyError):
            pass
    cap = cv2.VideoCapture(src)
    fps = video_fps(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det = Detector(args, fps)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(det(frame, len(frames)))
        if progress:
            progress("detect", len(frames), total)
        elif len(frames) % 25 == 0:
            print(f"\r  检测 {len(frames)}/{total} 帧", end="", flush=True)
    cap.release()
    log(f"\r  检测完成,共 {len(frames)} 帧          ")
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"key": key, "embed": det.rec is not None, "frames": frames}, fh)
    return frames


def video_fps(cap_or_path):
    cap = cv2.VideoCapture(cap_or_path) if isinstance(cap_or_path, str) else cap_or_path
    fps = cap.get(cv2.CAP_PROP_FPS)
    if isinstance(cap_or_path, str):
        cap.release()
    return fps if fps and 1 <= fps <= 240 else 25.0


# ---------------------------------------------------------------- 部位 → 打码形状
# 形状: ("e", cx, cy, 半轴1, 半轴2, 角度) 椭圆 | ("c", x1, y1, x2, y2, 半径) 胶囊 | ("p", 点列) 多边形

def _p(pose, i):
    return np.array(pose[i][:2], np.float64)


def _d(a, b):
    return float(np.linalg.norm(a - b))


def pose_shapes(pose, parts, pad):
    k = 1 + pad
    sh_l, sh_r, hip_l, hip_r = _p(pose, L_SH), _p(pose, R_SH), _p(pose, L_HIP), _p(pose, R_HIP)
    sh_mid, hip_mid = (sh_l + sh_r) / 2, (hip_l + hip_r) / 2
    head_pts = np.array([pose[i][:2] for i in range(11)])
    ear = _d(_p(pose, L_EAR), _p(pose, R_EAR))
    # 身体尺度:取多种测量的最大值,侧身/透视缩短时也不会过小
    s = max(_d(sh_l, sh_r), 0.45 * _d(sh_mid, hip_mid), 1.6 * ear,
            2.2 * _d(_p(pose, L_EYE), _p(pose, R_EYE)), 12.0)
    out = []

    def cap(a, b, r):
        out.append(("c", *_p(pose, a), *_p(pose, b), r * s * k))

    def blob(idx, r):
        pts = np.array([pose[i][:2] for i in idx])
        c = pts.mean(0)
        rad = max(r * s, float(np.max(np.linalg.norm(pts - c, axis=1))) + 0.08 * s) * k
        out.append(("e", c[0], c[1], rad, rad, 0.0))

    for part in parts:
        if part == "head":
            # 五官点都在脸上(头的前半部);侧脸时要以耳朵为中心,并往后脑方向扩,把后脑和发髻包进去
            nose = _p(pose, NOSE)
            ear_mid = (_p(pose, L_EAR) + _p(pose, R_EAR)) / 2
            back = ear_mid - nose                   # 从鼻子指向后脑
            c = ear_mid + 0.25 * back
            up = c - sh_mid
            n = float(np.linalg.norm(up))
            up = up / n if n > 1e-3 else np.array([0.0, -1.0])
            r = max(1.5 * float(np.linalg.norm(back)), 0.8 * ear,
                    float(np.max(np.linalg.norm(head_pts - c, axis=1))) * 1.1, 0.3 * s, 8.0)
            c = c + up * 0.2 * r                    # 往头顶方向偏一点,把头发包进去
            ang = math.degrees(math.atan2(up[1], up[0]))
            out.append(("e", c[0], c[1], 1.3 * r * k, 1.1 * r * k, ang))
        elif part == "torso":
            out.append(("p", [list(sh_l), list(sh_r), list(hip_r), list(hip_l)]))
            for a, b in ((L_SH, R_SH), (R_SH, R_HIP), (R_HIP, L_HIP), (L_HIP, L_SH)):
                cap(a, b, 0.16)
        elif part == "chest":
            lo_l, lo_r = sh_l + 0.5 * (hip_l - sh_l), sh_r + 0.5 * (hip_r - sh_r)
            out.append(("p", [list(sh_l), list(sh_r), list(lo_r), list(lo_l)]))
            out.append(("c", *sh_l, *sh_r, 0.16 * s * k))
            out.append(("c", *lo_l, *lo_r, 0.16 * s * k))
            out.append(("c", *sh_l, *lo_l, 0.16 * s * k))
            out.append(("c", *sh_r, *lo_r, 0.16 * s * k))
        elif part == "crotch":
            c = hip_mid + 0.12 * (hip_mid - sh_mid)
            r = max(0.75 * _d(hip_l, hip_r), 0.42 * s)
            out.append(("e", c[0], c[1], r * k, r * k, 0.0))
        elif part == "right_upper_arm":
            cap(R_SH, R_EL, 0.2)
        elif part == "left_upper_arm":
            cap(L_SH, L_EL, 0.2)
        elif part == "right_forearm":
            cap(R_EL, R_WR, 0.17)
        elif part == "left_forearm":
            cap(L_EL, L_WR, 0.17)
        elif part == "right_hand":
            blob(R_HAND, 0.2)
        elif part == "left_hand":
            blob(L_HAND, 0.2)
        elif part == "right_thigh":
            cap(R_HIP, R_KN, 0.27)
        elif part == "left_thigh":
            cap(L_HIP, L_KN, 0.27)
        elif part == "right_calf":
            cap(R_KN, R_AN, 0.2)
        elif part == "left_calf":
            cap(L_KN, L_AN, 0.2)
        elif part == "right_foot":
            blob(R_FOOT, 0.18)
        elif part == "left_foot":
            blob(L_FOOT, 0.18)
    return out


def face_shape(f, pad):
    cx, cy, w, h, ang = f[:5]
    k = 1 + 2 * pad
    return ("e", cx, cy, w / 2 * k, h / 2 * k, ang)


def count_people(det):
    """估算这一帧有几个人:重叠的骨架算同一个人,落在已有头部位置上的人脸也不重复计数。"""
    heads = []                                     # [(x, y, 半径)]

    def near(x, y, r):
        return any(math.hypot(x - hx, y - hy) < max(r, hr) for hx, hy, hr in heads)

    for pose in det["p"]:
        pts = np.array([pt[:2] for pt in pose[:11]], np.float64)
        if not np.isfinite(pts).all():
            continue
        x, y = pts.mean(0)
        ear = math.hypot(pose[L_EAR][0] - pose[R_EAR][0], pose[L_EAR][1] - pose[R_EAR][1])
        sh = math.hypot(pose[L_SH][0] - pose[R_SH][0], pose[L_SH][1] - pose[R_SH][1])
        r = max(0.9 * ear, 0.4 * sh, 15.0)
        if not near(x, y, r):
            heads.append((x, y, r))
    for f in det["f"]:
        cx, cy, w, h = f[:4]
        r = max(w, h) * 0.7
        if not near(cx, cy, r):
            heads.append((cx, cy, r))
    return len(heads)


# ---------------------------------------------------------------- 按人:同帧归并 → 跨帧跟踪 → 按人分组

def _head(pose):
    """骨架的头部中心和半径;关键点异常时返回 None"""
    pts = np.array([pt[:2] for pt in pose[:11]], np.float64)
    if not np.isfinite(pts).all():
        return None
    ear = math.hypot(pose[L_EAR][0] - pose[R_EAR][0], pose[L_EAR][1] - pose[R_EAR][1])
    sh = math.hypot(pose[L_SH][0] - pose[R_SH][0], pose[L_SH][1] - pose[R_SH][1])
    return pts.mean(0), max(0.9 * ear, 0.4 * sh, 15.0)


def _box(points):
    pts = np.array([p[:2] for p in points], np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) == 0:
        return None
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def frame_people(det):
    """把一帧里的骨架和人脸归成「人」。
    - 旋转检测会给同一个人多副骨架:头挨着、身体框也大部分重叠 → 同一个人
    - 人脸挂到最近的人头上;同时挨着两个人头的人脸归属不明 → 始终打码
    返回 (people, 始终打码的骨架下标, 始终打码的人脸下标)
    people = [{"poses": [..], "faces": [..], "c": (x, y), "r": 头半径, "box": [x1, y1, x2, y2]}]"""
    people, lock_p, lock_f = [], [], []
    for i, pose in enumerate(det["p"]):
        h, box = _head(pose), _box(pose)
        if h is None or box is None:
            lock_p.append(i)
            continue
        c, r = h
        for p in people:
            if (math.hypot(c[0] - p["c"][0], c[1] - p["c"][1]) < 0.6 * max(r, p["r"])
                    and _iou(box, p["box"]) > 0.4):
                p["poses"].append(i)
                p["box"] = _union(p["box"], box)
                break
        else:
            people.append({"poses": [i], "faces": [], "c": (float(c[0]), float(c[1])),
                           "r": float(r), "box": box})
    for j, f in enumerate(det["f"]):
        cx, cy, w, h = f[:4]
        fr = max(w, h) * 0.7
        near = sorted((math.hypot(cx - p["c"][0], cy - p["c"][1]), k)
                      for k, p in enumerate(people))
        near = [(d, k) for d, k in near if d < max(people[k]["r"], fr)]
        if not near:
            people.append({"poses": [], "faces": [j], "c": (cx, cy), "r": fr,
                           "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
        elif len(near) == 1 or near[0][0] < 0.5 * near[1][0]:
            people[near[0][1]]["faces"].append(j)
        else:
            lock_f.append(j)
    return people, lock_p, lock_f


def build_tracks(frames, gap):
    """逐帧按头部位置把人连起来,给每个人一个跨帧不变的编号(写入 person["tid"])。
    中断不超过 gap 帧的沿用原编号,超过就当新人。返回 [{"id", "frames", "c0", "r0", "c", "r"}]"""
    tracks, active = [], []
    for fi, people in enumerate(frames):
        active = [t for t in active if fi - t["last"] <= gap]
        cands = []
        for ti, t in enumerate(active):
            dt = fi - t["last"]
            px, py = t["c"][0] + t["v"][0] * min(dt, 5), t["c"][1] + t["v"][1] * min(dt, 5)
            for pi, p in enumerate(people):
                r = max(t["r"], p["r"])
                d = math.hypot(p["c"][0] - px, p["c"][1] - py)
                if d < r * min(1.5 + 0.25 * dt, 4.0):
                    cands.append((d / r, ti, pi))
        cands.sort()
        used_t, used_p = set(), set()
        for _, ti, pi in cands:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            t, p = active[ti], people[pi]
            dt = fi - t["last"]
            vx, vy = (p["c"][0] - t["c"][0]) / dt, (p["c"][1] - t["c"][1]) / dt
            t["v"] = (0.5 * t["v"][0] + 0.5 * vx, 0.5 * t["v"][1] + 0.5 * vy)
            t["c"], t["r"], t["last"] = p["c"], 0.7 * t["r"] + 0.3 * p["r"], fi
            t["frames"].append(fi)
            p["tid"] = t["id"]
        for pi, p in enumerate(people):
            if pi not in used_p:
                t = {"id": len(tracks), "frames": [fi], "c0": p["c"], "r0": p["r"],
                     "c": p["c"], "r": p["r"], "v": (0.0, 0.0), "last": fi}
                tracks.append(t)
                active.append(t)
                p["tid"] = t["id"]
    for t in tracks:
        t.pop("v")
        t.pop("last")
    return tracks


def group_tracks(tracks, frames, dets, fps, thr):
    """把属于同一个人的多段编号合成一组。
    - 人脸特征相似度 ≥ thr → 合并
    - 一段刚消失、1 秒内在附近又出现一段(都没露脸或脸也对得上)→ 合并
    - 同一时刻同时出现的两段,一定不是同一个人,绝不合并
    合并宁严勿宽:合错了,被合进来的陌生人会跟着不打码。"""
    T = len(tracks)
    samples = [[] for _ in range(T)]
    for people, d in zip(frames, dets):
        emb = d.get("e")
        if not emb:
            continue
        for p in people:
            for j in p["faces"]:
                if j < len(emb) and emb[j]:
                    f = d["f"][j]
                    samples[p["tid"]].append((f[2] * f[3], emb[j]))
    feats = []
    for s in samples:
        if not s:
            feats.append(None)
            continue
        s.sort(key=lambda t: -t[0])
        v = np.mean([decode_feature(e) for _, e in s[:10]], axis=0)
        feats.append(v / max(float(np.linalg.norm(v)), 1e-6))
    sets = [set(t["frames"]) for t in tracks]

    def cos(a, b):
        return float(feats[a] @ feats[b])

    cands = []
    for a in range(T):
        for b in range(a + 1, T):
            ta, tb = tracks[a], tracks[b]
            if ta["frames"][0] <= tb["frames"][-1] and tb["frames"][0] <= ta["frames"][-1] \
                    and not sets[a].isdisjoint(sets[b]):
                continue
            if feats[a] is not None and feats[b] is not None:
                s = cos(a, b)
                if s >= thr:
                    cands.append((1.0 + s, a, b, "face"))
            first, second = (ta, tb) if ta["frames"][-1] < tb["frames"][0] else \
                (tb, ta) if tb["frames"][-1] < ta["frames"][0] else (None, None)
            if first is not None and second["frames"][0] - first["frames"][-1] <= fps:
                r = max(first["r"], second["r0"])
                d = math.hypot(first["c"][0] - second["c0"][0], first["c"][1] - second["c0"][1])
                if d < 2.5 * r:
                    cands.append((1.0 - d / (2.5 * r), a, b, "near"))
    cands.sort(key=lambda c: -c[0])

    parent = list(range(T))
    members = {i: [i] for i in range(T)}
    gsets = {i: sets[i] for i in range(T)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def group_sim(ga, gb):
        vals = [cos(a, b) for a in members[ga] if feats[a] is not None
                for b in members[gb] if feats[b] is not None]
        return sum(vals) / len(vals) if vals else None

    for _, a, b, kind in cands:
        ga, gb = find(a), find(b)
        if ga == gb or not gsets[ga].isdisjoint(gsets[gb]):
            continue
        s = group_sim(ga, gb)
        if kind == "face" and (s is None or s < thr):
            continue
        if kind == "near" and s is not None and s < 0.7 * thr:
            continue
        parent[gb] = ga
        members[ga] += members.pop(gb)
        gsets[ga] = gsets[ga] | gsets.pop(gb)

    groups = []
    for g, mem in members.items():
        mem.sort(key=lambda t: tracks[t]["frames"][0])
        n = len(gsets[g])
        has_face = any(feats[t] is not None for t in mem)
        groups.append({"tracks": mem, "n": n, "f0": min(gsets[g]), "f1": max(gsets[g]),
                       "face": has_face, "minor": n < max(5, 0.3 * fps) and not has_face})
    groups.sort(key=lambda g: (g["minor"], g["f0"]))
    return groups


class People:
    """视频里的「人」:跟踪编号、按人分组,以及哪些人不打码(skip,按跟踪编号存)。"""

    def __init__(self, dets, fps, args):
        self.fps = fps
        self.frames, self.lock = [], []
        for d in dets:
            ps, lp, lf = frame_people(d)
            self.frames.append(ps)
            self.lock.append((lp, lf))
        self.gap = max(args.hold, int(fps * 0.5))
        self.tracks = build_tracks(self.frames, self.gap)
        self.groups = group_tracks(self.tracks, self.frames, dets, fps, args.match)
        self.skip = set()
        self.sig = [len(dets), len(self.tracks), self.gap]
        self._reindex()

    def _reindex(self):
        self.tid2g = {t: k for k, g in enumerate(self.groups) for t in g["tracks"]}

    def visible_groups(self):
        return [k for k, g in enumerate(self.groups) if not g["minor"]]

    def group_masked(self, k):
        return not all(t in self.skip for t in self.groups[k]["tracks"])

    def set_group(self, k, masked):
        for t in self.groups[k]["tracks"]:
            (self.skip.discard if masked else self.skip.add)(t)

    def split(self, k):
        """把一组拆成每段一张卡片(自动分组合错时用)"""
        g = self.groups.pop(k)
        for i, t in enumerate(g["tracks"]):
            fr = self.tracks[t]["frames"]
            self.groups.insert(k + i, {"tracks": [t], "n": len(fr), "f0": fr[0], "f1": fr[-1],
                                       "face": g["face"], "minor": False, "split": True})
        self._reindex()

    def tid_at(self, fi, x, y):
        """第 fi 帧中 (x, y) 处的人的编号(点在多人框内时取框最小的)"""
        best = None
        for p in self.frames[fi]:
            x1, y1, x2, y2 = p["box"]
            m = 0.15 * max(x2 - x1, y2 - y1, 2 * p["r"])
            if x1 - m <= x <= x2 + m and y1 - m <= y <= y2 + m:
                area = (x2 - x1 + 2 * m) * (y2 - y1 + 2 * m)
                if best is None or area < best[0]:
                    best = (area, p["tid"])
        return None if best is None else best[1]

    def load(self, path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return False
        if data.get("sig") != self.sig:
            return False
        split = set(data.get("split", []))
        k = 0
        while k < len(self.groups):
            g = self.groups[k]
            if len(g["tracks"]) > 1 and split & set(g["tracks"]):
                self.split(k)
            k += 1
        self.skip = set(data.get("skip", [])) & set(range(len(self.tracks)))
        for k in range(len(self.groups)):          # 一组里只要有一段没确认,整组按打码处理
            if self.group_masked(k):
                self.set_group(k, True)
        return True

    def save(self, path):
        split = [t for g in self.groups if g.get("split") for t in g["tracks"]]
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"sig": self.sig, "skip": sorted(self.skip), "split": split}, fh)
        except OSError:
            pass

    def label(self, k):
        return "人物 %d" % (k + 1)


class ShapeIndex:
    """每帧要遮的区域。人数突然变少/检测中断时,用前后 hold 帧的位置补上(宁可多打)。
    给了 people 时按人存形状:skip 里的人不打码,补遮也只用同一个人前后帧的位置。"""

    def __init__(self, dets, parts, args, people=None):
        self.people = people
        self.per = []                  # 每帧 [(跟踪编号或 None, 头中心, 头半径, 形状列表)]
        head = "head" in parts
        for fi, d in enumerate(dets):
            if people is None:
                sh = []
                for pose in d["p"]:
                    sh += pose_shapes(pose, parts, args.pad)
                if head:
                    sh += [face_shape(f, args.pad) for f in d["f"]]
                self.per.append([(None, None, 0.0, sh)])
                continue
            items = []
            for p in people.frames[fi]:
                sh = []
                for i in p["poses"]:
                    sh += pose_shapes(d["p"][i], parts, args.pad)
                if head:
                    sh += [face_shape(d["f"][j], args.pad) for j in p["faces"]]
                items.append((p["tid"], p["c"], p["r"], sh))
            lp, lf = people.lock[fi]
            sh = []
            for i in lp:
                sh += pose_shapes(d["p"][i], parts, args.pad)
            if head:
                sh += [face_shape(d["f"][j], args.pad) for j in lf]
            if sh:
                items.append((None, None, 0.0, sh))
            self.per.append(items)
        self.cnt = [count_people(d) for d in dets]
        self.hold, self.smooth = args.hold, args.smooth
        # 「人数变少」:前面和后面都出现过更多的人,只有中间这几帧少了 → 真的可能漏了人。
        # 只在一侧多出来(偶尔误检出一个人、或有人走出画面)不算。
        n, h = len(self.cnt), self.hold
        self.drop = []
        for i in range(n):
            before = max(self.cnt[max(0, i - h):i], default=0)
            after = max(self.cnt[i + 1:i + h + 1], default=0)
            self.drop.append(self.cnt[i] < min(before, after))

    @property
    def skip(self):
        return self.people.skip if self.people is not None else ()

    def _frame(self, j, skip):
        """第 j 帧要打码的形状;第二个返回值 = 有不打码的人因为挨着别人而被强制打码"""
        items = self.per[j]
        if not skip:
            return [s for it in items for s in it[3]], False
        heads = [(it[1], it[2]) for it in items
                 if it[1] is not None and (it[0] is None or it[0] not in skip)]
        out, forced = [], False
        for tid, c, r, sh in items:
            if tid is not None and tid in skip:
                # 和要打码的人靠得很近 → 编号可能对调,这个人也一起打码(宁可多打)
                if not any(math.hypot(c[0] - hc[0], c[1] - hc[1]) < 2.0 * max(r, hr)
                           for hc, hr in heads):
                    continue
                forced = True
            out += sh
        return out, forced

    def __call__(self, i, marks=()):
        n = len(self.per)
        lo, hi = max(0, i - self.hold), min(n, i + self.hold + 1)
        win = self.hold if self.cnt[i] < max(self.cnt[lo:hi]) else self.smooth
        skip = self.skip
        out = []
        for j in range(max(0, i - win), min(n, i + win + 1)):
            out += self._frame(j, skip)[0]
        for m in marks:
            f0, f1 = m["f0"], m["f1"]
            if f0 - self.hold <= i <= f1 + self.hold:
                t = 0.0 if f1 == f0 else min(1.0, max(0.0, (i - f0) / (f1 - f0)))
                x, y, w, h = (a + (b - a) * t for a, b in zip(m["b0"], m["b1"]))
                mx, my = w * 0.08, h * 0.08
                out.append(("p", [[x - mx, y - my], [x + w + mx, y - my],
                                  [x + w + mx, y + h + my], [x - mx, y + h + my]]))
        return out

    def dropped(self, i):
        return self.drop[i]

    def covered(self, i, marks=()):
        """这一帧有打码区域,或者画面里的人都是选了「不打码」的"""
        return bool(self(i, marks)) or self.unmasked_people(i) > 0

    def risky(self, i):
        """不打码的人和别人靠得很近(编号可能对调)"""
        skip = self.skip
        return bool(skip) and self._frame(i, skip)[1]

    def unmasked_people(self, i):
        """这一帧里不打码的人数"""
        skip = self.skip
        if not skip:
            return 0
        return sum(1 for it in self.per[i] if it[0] is not None and it[0] in skip)


# ---------------------------------------------------------------- 绘制

def shapes_mask(shapes, H, W):
    mask = np.zeros((H, W), np.uint8)
    big = 0.0
    for sh in shapes:
        if sh[0] == "e":
            _, cx, cy, a1, a2, ang = sh
            if not all(map(math.isfinite, (cx, cy, a1, a2))):
                continue
            a1, a2 = max(a1, 2.0), max(a2, 2.0)
            big = max(big, a1, a2)
            cv2.ellipse(mask, (int(cx), int(cy)), (int(a1), int(a2)), float(ang), 0, 360, 255, -1)
        elif sh[0] == "c":
            _, x1, y1, x2, y2, r = sh
            if not all(map(math.isfinite, (x1, y1, x2, y2, r))):
                continue
            r = max(int(r), 2)
            big = max(big, r * 1.5)
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.line(mask, p1, p2, 255, 2 * r)
            cv2.circle(mask, p1, r, 255, -1)
            cv2.circle(mask, p2, r, 255, -1)
        else:
            pts = np.array(sh[1], np.float32)
            if not np.isfinite(pts).all():
                continue
            hull = cv2.convexHull(pts).astype(np.int32)
            cv2.fillConvexPoly(mask, hull, 255)
            big = max(big, math.sqrt(max(cv2.contourArea(hull), 1)) / 2)
    return mask, big


def apply_shapes(frame, shapes, mode, feather, strength=1.0):
    if not shapes:
        return
    H, W = frame.shape[:2]
    mask, big = shapes_mask(shapes, H, W)
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return
    soft = feather > 0 and mode != "solid"
    k = max(3, int(big * feather) | 1) if soft else 1
    x1, y1, x2, y2 = max(0, x - k), max(0, y - k), min(W, x + w + k), min(H, y + h + k)
    m = mask[y1:y2, x1:x2]
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if soft:   # 先膨胀再模糊:羽化只向外过渡
        m = cv2.dilate(m, np.ones((k // 2 | 1, k // 2 | 1), np.uint8))
        m = cv2.GaussianBlur(m, (k, k), 0)
    if mode == "solid":
        proc = np.zeros_like(roi)
    else:
        # 强度:马赛克格子大小 / 模糊程度,1 = 默认,越大越看不清
        s = max(8.0, big / 5) * strength
        small = cv2.resize(roi, (max(1, int(rw / s)), max(1, int(rh / s))),
                           interpolation=cv2.INTER_AREA)
        proc = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST
                          if mode == "pixel" else cv2.INTER_LINEAR)
        if mode == "blur":
            kb = max(3, int(31 * strength) | 1)
            proc = cv2.GaussianBlur(proc, (kb, kb), 0)
    a = (m.astype(np.float32) / 255)[..., None]
    roi[:] = (roi * (1 - a) + proc * a).astype(np.uint8)


def mask_frame(frame, i, index, marks, args):
    shapes = index(i, marks)
    if shapes:
        apply_shapes(frame, shapes, args.mode, args.feather, args.strength)
        return True
    if index.unmasked_people(i):          # 画面里只有选了「不打码」的人:是有意不打,不算漏
        return True
    if args.unmasked == "blur":
        H, W = frame.shape[:2]
        apply_shapes(frame, [("p", [[0, 0], [W, 0], [W, H], [0, H]])], "blur", 0)
    return False


def draw_hud(img, lines, font=cv2.FONT_HERSHEY_SIMPLEX, size=0.6, th=1):
    """左上角信息栏:半透明黑底 + 单层文字(不再用粗细两层叠字,避免错位)"""
    pad, gap = 8, 8
    sizes = [cv2.getTextSize(t, font, size, th)[0] for t, _ in lines]
    lh = max(h for _, h in sizes)
    w = max(w for w, _ in sizes) + 2 * pad
    h = len(lines) * (lh + gap) + pad
    w, h = min(w, img.shape[1]), min(h, img.shape[0])
    roi = img[0:h, 0:w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    y = pad + lh
    for t, c in lines:
        cv2.putText(img, t, (pad, y), font, size, c, th, cv2.LINE_AA)
        y += lh + gap


def draw_pose(img, pose, scale, color=(0, 255, 0), dx=0, dy=0, th=2):
    P = []
    for x, y, _ in pose:
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        P.append((int(x * scale) - dx, int(y * scale) - dy))
    for a, b in SKELETON:
        cv2.line(img, P[a], P[b], color, th, cv2.LINE_AA)
    for i, p in enumerate(P):
        cv2.circle(img, p, th + 1, (0, 255, 255) if i <= 10 else (0, 128, 255), -1)


def draw_skeleton(img, det, scale):
    for pose in det["p"]:
        draw_pose(img, pose, scale)
    for f in det["f"]:
        cx, cy, w, h = f[:4]
        cv2.rectangle(img, (int((cx - w / 2) * scale), int((cy - h / 2) * scale)),
                      (int((cx + w / 2) * scale), int((cy + h / 2) * scale)), (255, 0, 255), 1)


def draw_people_tags(img, people, index, fi, scale):
    """检查窗口里给每个人标上「人物 N」;不打码的人用橙色并注明"""
    skip = index.skip
    for p in people.frames[fi]:
        k = people.tid2g.get(p["tid"])
        if k is None:
            continue
        free = p["tid"] in skip
        txt = "P%d%s" % (k + 1, " NO MASK" if free else "")
        x, y = int(p["box"][0] * scale), int(p["box"][1] * scale) - 6
        y = max(14, y)
        color = (0, 140, 255) if free else (220, 220, 220)
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x - 2, y - tht - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- 人工检查

def load_marks(path):
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except ValueError:
            pass
    return []


def save_marks(path, marks):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(marks, fh)


def review(src, dets, index, marks, marks_path, args, people=None, people_path=None):
    cap = cv2.VideoCapture(src)
    fps = video_fps(cap)
    total = len(dets)
    if total == 0:
        return
    win = "review (q = save & render)"
    st = {"idx": 0, "playing": False, "pending": None, "drag": None, "cur": None,
          "scale": 1.0, "skel": False, "msg": "", "msg_until": 0.0, "ver": 0}
    seg = {"key": None, "bare": [], "drop": [], "risk": []}
    names = {"bare": ("NO MASK", "unmasked"), "drop": ("DROPPED", "dropped"),
             "risk": ("CLOSE TO OTHERS", "close-contact")}

    def segments(frames):
        """连续的帧合并成一段,返回每段的起始帧"""
        out = []
        for j in frames:
            if not out or j != prev + 1:
                out.append(j)
            prev = j
        return out

    def refresh_segments():
        key = (len(marks), st["ver"])         # 手动标记或「不打码的人」变了才重新计算
        if seg["key"] == key:
            return
        seg["key"] = key
        seg["bare"] = segments([j for j in range(total) if not index.covered(j, marks)])
        seg["drop"] = segments([j for j in range(total)
                                if index.dropped(j) and index(j, marks)])
        seg["risk"] = segments([j for j in range(total) if index.risky(j)])

    def jump(kind, forward):
        refresh_segments()
        starts = seg[kind]
        i = st["idx"]
        cand = [j for j in starts if j > i] if forward else [j for j in starts if j < i]
        st["playing"] = False
        if cand:
            st["idx"] = cand[0] if forward else cand[-1]
            k = starts.index(st["idx"]) + 1
            flash("%s %d/%d" % (names[kind][0], k, len(starts)))
        else:
            flash("no more %s frames %s" % (names[kind][1],
                                            "after this" if forward else "before this"))

    def flash(text):
        import time as _t
        st["msg"], st["msg_until"] = text, _t.time() + 2.5

    def on_mouse(event, x, y, flags, _):
        s = st["scale"]
        p = (x / s, y / s)
        if event == cv2.EVENT_RBUTTONDOWN:          # 右键点人:切换打码 / 不打码
            st["playing"] = False
            if people is None:
                flash("enable 'choose people' (--pick) to toggle a person")
                return
            tid = people.tid_at(st["idx"], *p)
            k = people.tid2g.get(tid) if tid is not None else None
            if k is None:
                flash("no person here")
                return
            masked = not people.group_masked(k)
            people.set_group(k, masked)
            st["ver"] += 1
            if people_path:
                people.save(people_path)
            flash("P%d -> %s" % (k + 1, "MASK" if masked else "NO MASK (whole video)"))
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            st["playing"] = False
            st["drag"], st["cur"] = p, p
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"]:
            st["cur"] = p
        elif event == cv2.EVENT_LBUTTONUP and st["drag"]:
            (xa, ya), (xb, yb) = st["drag"], p
            st["drag"] = None
            box = [min(xa, xb), min(ya, yb), abs(xb - xa), abs(yb - ya)]
            if box[2] < 8 or box[3] < 8:
                return
            i, pend = st["idx"], st["pending"]
            if pend and pend[0] != i:
                (fa, ba), (fb, bb) = sorted([pend, (i, box)], key=lambda t: t[0])
                marks.append({"f0": fa, "f1": fb, "b0": ba, "b1": bb})
                save_marks(marks_path, marks)
            st["pending"] = (i, box)

    def on_track(v):
        if v != st["idx"]:
            st["idx"], st["playing"] = v, False

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    if total > 1:
        cv2.createTrackbar("frame", win, 0, total - 1, on_track)
    pos, shown, frame = 0, -1, None
    step_1s = max(1, int(round(fps)))
    while True:
        i = st["idx"]
        if i != shown:
            if i != pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            pos = i + 1
            if not ok:
                if frame is None:
                    break
                st["idx"], st["playing"], pos = shown, False, -1
                continue
            frame, shown = fr, i
            if total > 1:
                cv2.setTrackbarPos("frame", win, i)
        disp = frame.copy()
        masked = mask_frame(disp, i, index, marks, args)
        H, W = disp.shape[:2]
        s = st["scale"] = min(1.0, 1280 / W, 720 / H)
        if s < 1:
            disp = cv2.resize(disp, None, fx=s, fy=s)
        if st["skel"]:
            draw_skeleton(disp, dets[i], s)
        if people is not None:
            draw_people_tags(disp, people, index, i, s)
        if st["pending"]:
            x, y, w, h = (int(v * s) for v in st["pending"][1])
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if st["drag"]:
            (xa, ya), (xb, yb) = st["drag"], st["cur"]
            cv2.rectangle(disp, (int(xa * s), int(ya * s)), (int(xb * s), int(yb * s)),
                          (0, 255, 255), 2)
        refresh_segments()
        risky = index.risky(i)
        warn = not masked or index.dropped(i) or risky
        state = ("NO MASK" if not masked else "PEOPLE DROPPED" if index.dropped(i)
                 else "CLOSE TO OTHERS - check IDs" if risky else "ok")
        free = index.unmasked_people(i)
        txt = "%d/%d  people:%d%s  %s  manual:%d%s" % (
            i + 1, total, index.cnt[i], " (no mask:%d)" % free if free else "", state, len(marks),
            "  [marking from %d - Enter to end]" % (st["pending"][0] + 1) if st["pending"] else "")
        txt2 = "m: unmasked %d segs   n: dropped %d segs" % (len(seg["bare"]), len(seg["drop"]))
        if people is not None:
            txt2 += "   t: close-contact %d segs   right-click: toggle person" % len(seg["risk"])
        import time as _t
        if st["msg"] and _t.time() < st["msg_until"]:
            txt2 += "   >> " + st["msg"]
        color = (90, 90, 255) if warn else (80, 255, 80)
        draw_hud(disp, [(txt, color), (txt2, (255, 255, 255))])
        cv2.imshow(win, disp)

        key = cv2.waitKey(max(1, int(1000 / fps)) if st["playing"] else 30) & 0xFF
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            st["playing"] = not st["playing"]
        elif key == ord("k"):
            st["skel"] = not st["skel"]
        elif key in (ord("a"), ord("d"), ord("s"), ord("w")):
            st["playing"] = False
            delta = {"a": -1, "d": 1, "s": -step_1s, "w": step_1s}[chr(key)]
            st["idx"] = min(total - 1, max(0, i + delta))
        elif key in (ord("m"), ord("M")):
            jump("bare", key == ord("m"))
        elif key in (ord("n"), ord("N")):
            jump("drop", key == ord("n"))
        elif key in (ord("t"), ord("T")):
            jump("risk", key == ord("t"))
        elif key in (13, 10) and st["pending"]:
            f, box = st["pending"]
            marks.append({"f0": min(f, i), "f1": max(f, i), "b0": box, "b1": box})
            save_marks(marks_path, marks)
            st["pending"] = None
        elif key == ord("u"):
            if st["pending"]:
                st["pending"] = None
            elif marks:
                marks.pop()
                st["ver"] += 1
                save_marks(marks_path, marks)
        if st["playing"]:
            if i + 1 < total:
                st["idx"] = i + 1
            else:
                st["playing"] = False
    cap.release()
    cv2.destroyAllWindows()
    save_marks(marks_path, marks)
    if people is not None and people_path:
        people.save(people_path)


# ---------------------------------------------------------------- 人物选择

THUMB_H = 140


def person_crop(frame, det, p, h=THUMB_H):
    """截出这个人(外扩一圈),画上他的骨架和人脸框,缩到高 h,返回 PNG 的 base64"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = p["box"]
    for j in p["faces"]:
        cx, cy, fw, fh = det["f"][j][:4]
        x1, y1 = min(x1, cx - fw / 2), min(y1, cy - fh / 2)
        x2, y2 = max(x2, cx + fw / 2), max(y2, cy + fh / 2)
    m = 0.12 * max(x2 - x1, y2 - y1) + 10
    x1, y1 = int(max(0, x1 - m)), int(max(0, y1 - m))
    x2, y2 = int(min(W, x2 + m)), int(min(H, y2 + m))
    if x2 - x1 < 40:
        c = (x1 + x2) // 2
        x1, x2 = max(0, c - 20), min(W, c + 20)
    if y2 - y1 < 40:
        c = (y1 + y2) // 2
        y1, y2 = max(0, c - 20), min(H, c + 20)
    if x2 <= x1 or y2 <= y1:
        return None
    img = frame[y1:y2, x1:x2].copy()
    s = min(h / (y2 - y1), 1.4 * h / (x2 - x1))
    img = cv2.resize(img, (max(1, int((x2 - x1) * s)), max(1, int((y2 - y1) * s))),
                     interpolation=cv2.INTER_AREA)
    dx, dy = int(x1 * s), int(y1 * s)
    for i in p["poses"]:
        draw_pose(img, det["p"][i], s, (0, 255, 0), dx, dy, th=1)
    for j in p["faces"]:
        cx, cy, fw, fh = det["f"][j][:4]
        cv2.rectangle(img, (int((cx - fw / 2) * s) - dx, int((cy - fh / 2) * s) - dy),
                      (int((cx + fw / 2) * s) - dx, int((cy + fh / 2) * s) - dy), (255, 0, 255), 1)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def people_thumbs(src, people, dets, per_track=3, progress=None):
    """每段编号挑最多 3 张截图(开头/中间/结尾各一张,优先脸大、人大的帧)。返回 {编号: [png_b64]}"""
    tids = {t for k in people.visible_groups() for t in people.groups[k]["tracks"]}
    occ = {t: [] for t in tids}
    for fi, ps in enumerate(people.frames):
        for pi, p in enumerate(ps):
            if p["tid"] in occ:
                occ[p["tid"]].append((fi, pi))

    def score(fi, pi):
        p, d = people.frames[fi][pi], dets[fi]
        x1, y1, x2, y2 = p["box"]
        face = max((d["f"][j][2] * d["f"][j][3] for j in p["faces"]), default=0.0)
        return (x2 - x1) * (y2 - y1) + 20 * face

    want = {}
    for t, lst in occ.items():
        n = min(per_track, len(lst))
        for b in range(n):
            part = lst[b * len(lst) // n:(b + 1) * len(lst) // n]
            fi, pi = max(part, key=lambda fp: score(*fp))
            want.setdefault(fi, []).append((t, pi))
    out = {t: [] for t in tids}
    cap = cv2.VideoCapture(src)
    pos, order = 0, sorted(want)
    for n, fi in enumerate(order):
        if fi < pos or fi - pos > 30:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            pos = fi
        while pos < fi:
            cap.grab()
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if progress:
            progress("people", n + 1, len(order))
        if not ok:
            continue
        for t, pi in want[fi]:
            png = person_crop(frame, dets[fi], people.frames[fi][pi])
            if png:
                out[t].append(png)
    cap.release()
    return out


def fmt_clock(sec):
    sec = int(max(0, sec))
    return "%d:%02d:%02d" % (sec // 3600, sec // 60 % 60, sec % 60) if sec >= 3600 \
        else "%02d:%02d" % (sec // 60, sec % 60)


def pick_people_dialog(master, people, thumbs, title=""):
    """人物选择面板:每个人一张卡片(截图 + 骨架),勾选 = 打码。返回 True = 确定,False = 关闭窗口不改动。"""
    import tkinter as tk
    from tkinter import ttk

    own_root = master is None
    top = tk.Tk() if own_root else tk.Toplevel(master)
    top.title("选择要打码的人" + (" - " + title if title else ""))
    snapshot = ([dict(g) for g in people.groups], set(people.skip))
    result = {"ok": False}
    imgs, vars_ = [], {}

    head = ttk.Frame(top)
    head.pack(fill="x", padx=10, pady=(10, 4))
    ttk.Label(head, text="勾选 = 打码,取消勾选 = 这个人在整段视频里都不打码。",
              font=("", 11, "bold")).pack(anchor="w")
    ttk.Label(head, foreground="#555", justify="left", text=(
        "同一个人的多段画面已按人脸自动归成一组;如果一张卡片里混进了别人,点「拆开」。\n"
        "同一个人也可能出现在两张卡片里(没露脸的片段认不出来),要不打码的话两张都取消勾选。\n"
        "两个人靠得很近时,为防止认错人,两个人都会打码。")).pack(anchor="w", pady=(2, 0))

    body = ttk.Frame(top)
    body.pack(fill="both", expand=True, padx=10, pady=4)
    canvas = tk.Canvas(body, highlightthickness=0)
    sb = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def on_wheel(e):
        if e.num == 4 or e.delta > 0:
            canvas.yview_scroll(-1, "units")
        elif e.num == 5 or e.delta < 0:
            canvas.yview_scroll(1, "units")
    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        top.bind_all(seq, on_wheel)

    cols = max(1, min(3, (top.winfo_screenwidth() - 80) // 470))
    foot_note = ttk.Label(top, foreground="#555")

    def commit():
        for k, v in vars_.items():
            people.set_group(k, v.get())

    def card_images(k):
        tracks = people.groups[k]["tracks"]
        if len(tracks) == 1:
            return thumbs.get(tracks[0], [])[:3]
        big = sorted(tracks, key=lambda t: -len(people.tracks[t]["frames"]))[:3]
        out = []
        for t in big:
            th = thumbs.get(t, [])
            if th:
                out.append(th[len(th) // 2])
        return out

    def split(k):
        commit()
        people.split(k)
        build()

    def build():
        for w in inner.winfo_children():
            w.destroy()
        imgs.clear()
        vars_.clear()
        vis = people.visible_groups()
        for n, k in enumerate(vis):
            g = people.groups[k]
            card = ttk.LabelFrame(inner, text=" %s " % people.label(k))
            card.grid(row=n // cols, column=n % cols, sticky="nw", padx=6, pady=6)
            row = ttk.Frame(card)
            row.pack(anchor="w", padx=6, pady=(4, 2))
            pics = card_images(k)
            for png in pics:
                try:
                    im = tk.PhotoImage(data=png)
                except tk.TclError:
                    continue
                imgs.append(im)
                ttk.Label(row, image=im).pack(side="left", padx=2)
            if not pics:
                ttk.Label(row, text="(没有截图)", foreground="#888").pack(side="left")
            fps = people.fps
            info = "%s – %s · %d 段 · %s" % (
                fmt_clock(g["f0"] / fps), fmt_clock(g["f1"] / fps), len(g["tracks"]),
                "认出了人脸" if g["face"] else "没拍到清楚的脸")
            ttk.Label(card, text=info, foreground="#555").pack(anchor="w", padx=8)
            line = ttk.Frame(card)
            line.pack(fill="x", padx=6, pady=(2, 6))
            v = tk.BooleanVar(value=people.group_masked(k))
            vars_[k] = v
            ttk.Checkbutton(line, text="打码", variable=v).pack(side="left")
            if len(g["tracks"]) > 1:
                ttk.Button(line, text="拆开", width=6,
                           command=lambda k=k: split(k)).pack(side="right")
        minor = len(people.groups) - len(vis)
        foot_note.configure(text=("另有 %d 个零碎片段(很短、也没拍到脸),始终打码。" % minor)
                            if minor else "")
        if not vis:
            ttk.Label(inner, text="没有识别出人。").grid(row=0, column=0, padx=10, pady=10)

    def set_all(masked):
        for v in vars_.values():
            v.set(masked)

    def ok():
        commit()
        result["ok"] = True
        close()

    def cancel():
        people.groups[:] = snapshot[0]
        people.skip = snapshot[1]
        people._reindex()
        close()

    def close():
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            top.unbind_all(seq)
        top.destroy()

    foot_note.pack(anchor="w", padx=12)
    bar = ttk.Frame(top)
    bar.pack(fill="x", padx=10, pady=(4, 10))
    ttk.Button(bar, text="全部打码", command=lambda: set_all(True)).pack(side="left")
    ttk.Button(bar, text="全部不打码", command=lambda: set_all(False)).pack(side="left", padx=6)
    ttk.Button(bar, text="确定", command=ok).pack(side="right")
    top.protocol("WM_DELETE_WINDOW", cancel)
    top.bind("<Return>", lambda e: ok())
    build()
    top.update_idletasks()
    w = min(top.winfo_screenwidth() - 60, max(560, inner.winfo_reqwidth() + 50))
    h = min(top.winfo_screenheight() - 120, inner.winfo_reqheight() + 200)
    top.geometry("%dx%d" % (w, max(360, h)))
    if own_root:
        top.mainloop()
    else:
        top.transient(master)
        top.grab_set()
        master.wait_window(top)
    return result["ok"]


def prepare_people(src, dets, args, progress=None, log=print):
    """跟踪 + 分组 + 读入上次的选择 + 准备截图。没识别到人时返回 (people, None)。"""
    people = People(dets, video_fps(src), args)
    path = os.path.splitext(src)[0] + ".people.json"
    if people.load(path):
        log("  已载入上次「选择要打码的人」的结果")
    vis = people.visible_groups()
    log(f"  识别出 {len(vis)} 个人(跟踪片段 {len(people.tracks)} 段)")
    if not vis:
        return people, path, None
    return people, path, people_thumbs(src, people, dets, progress=progress)


# ---------------------------------------------------------------- 输出

def render(src, dst, index, marks, args, progress=None, log=print):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    has_ffmpeg = shutil.which("ffmpeg") is not None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name if has_ffmpeg else dst
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = bare = drop = 0
    total = len(index.per)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or n >= total:
                break
            if not mask_frame(frame, n, index, marks, args):
                bare += 1
            elif index.dropped(n):
                drop += 1
            writer.write(frame)
            n += 1
            if progress:
                progress("render", n, total)
            elif n % 50 == 0:
                print(f"\r  输出 {n}/{total} 帧", end="", flush=True)
    except BaseException:
        cap.release()
        writer.release()
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    cap.release()
    writer.release()
    log(f"\r  共 {n} 帧:{bare} 帧没有任何打码区域,{drop} 帧人数比前后少"
        f"(人工检查时按 m / n 分别跳转)")
    if has_ffmpeg:
        if progress:
            progress("encode", 0, 0)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", src,
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20",
               "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
               "-map_metadata", "-1", dst]      # 去掉拍摄时间/GPS 等元数据
        r = subprocess.run(cmd)
        if r.returncode != 0:
            cmd[cmd.index("copy")] = "aac"
            r = subprocess.run(cmd)
        os.remove(tmp)
        if r.returncode != 0:
            log("  ffmpeg 合成失败")
            return False
    else:
        log("  未找到 ffmpeg:输出视频没有声音")
    log(f"  已输出: {dst}")
    return True


def build_parser():
    p = argparse.ArgumentParser(
        description="本机逐帧人体部位打码(宁可多打,不要漏)。不带参数运行即打开图形界面。")
    p.add_argument("input", nargs="?", help="视频文件或文件夹")
    p.add_argument("-o", "--output", help="输出文件(仅单文件时有效),默认 原名_blurred.mp4")
    p.add_argument("--parts", default="head", help="要打码的部位,逗号分隔,默认 head(头)")
    p.add_argument("--list-parts", action="store_true", help="列出所有可选部位")
    p.add_argument("--review", action="store_true", help="输出前打开检查窗口,手动补漏")
    p.add_argument("--pick", action="store_true",
                   help="检测后弹出人物选择面板,选择哪些人不打码(需要图形界面)")
    p.add_argument("--match", type=float, default=0.45,
                   help="按人脸归组的相似度门槛,越高越严(宁可多拆几组),默认 0.45")
    p.add_argument("--strict", action="store_true",
                   help="更激进:heavy 模型、阈值 0.2、外扩 0.5、保持 30 帧、旋转检测")
    p.add_argument("--unmasked", choices=["keep", "blur"], default="keep",
                   help="完全没找到人的帧:keep=原样(默认) blur=整帧模糊")
    p.add_argument("--mode", choices=["blur", "pixel", "solid"], default="pixel",
                   help="blur=模糊 pixel=马赛克(默认) solid=黑块(最安全)")
    p.add_argument("--pad", type=float, default=0.3, help="打码区域外扩比例,默认 0.3")
    p.add_argument("--strength", type=float, default=1.0,
                   help="马赛克格子大小/模糊程度倍数,越大越看不清,默认 1.0")
    p.add_argument("--feather", type=float, default=0.2, help="边缘羽化,0=硬边,默认 0.2")
    p.add_argument("--hold", type=int, default=15,
                   help="人数变少或检测中断时,用前后多少帧的位置补上,默认 15")
    p.add_argument("--smooth", type=int, default=2,
                   help="正常情况下也合并前后几帧,防止快速动作时边缘露出,默认 2")
    p.add_argument("--conf", type=float, default=0.3, help="姿态检测阈值,越低越不易漏,默认 0.3")
    p.add_argument("--max-people", type=int, default=4, help="每帧最多识别几个人,默认 4")
    p.add_argument("--model-size", choices=["lite", "full", "heavy"], default="full",
                   help="姿态模型大小:heavy 最准最慢,默认 full")
    p.add_argument("--rotate", action="store_true",
                   help="额外做 90/180/270° 旋转检测(躺着/倒立的人更不易漏,慢约 4 倍)")
    p.add_argument("--no-face", dest="face", action="store_false",
                   help="关闭补充的人脸检测器(默认开启,与骨架取并集)")
    p.add_argument("--face-conf", type=float, default=0.3, help="人脸检测阈值,默认 0.3")
    p.add_argument("--max-side", type=int, default=1280, help="人脸检测缩放上限,默认 1280")
    p.add_argument("--redetect", action="store_true", help="忽略缓存,重新检测")
    p.add_argument("--pose-model", help="手动指定姿态模型 .task 文件")
    p.add_argument("--face-model", help="手动指定人脸模型 .onnx 文件")
    p.add_argument("--sface-model", help="手动指定人脸识别模型 .onnx 文件(--pick 用)")
    return p


def apply_strict(args):
    if args.strict:
        args.model_size = "heavy"
        args.conf = min(args.conf, 0.2)
        args.face_conf = min(args.face_conf, 0.2)
        args.pad = max(args.pad, 0.5)
        args.hold = max(args.hold, 30)
        args.rotate = True


def list_videos(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if os.path.splitext(f)[1].lower() in VIDEO_EXTS
                      and "_blurred" not in f)
    return [path]


def main():
    argv = sys.argv[1:]
    # 不带参数,或只拖入了文件/文件夹(拖到脚本图标上)→ 打开图形界面
    if not argv or all(os.path.exists(a) for a in argv):
        files = []
        for a in argv:
            files += list_videos(a)
        run_gui(files)
        return

    p = build_parser()
    args = p.parse_args()
    if args.list_parts:
        print(__doc__[__doc__.index("可选部位"):__doc__.index("人工检查")])
        return
    if not args.input:
        p.error("请指定视频文件或文件夹")
    apply_strict(args)
    parts = parse_parts(args.parts)
    print("打码部位:", ", ".join(parts))
    files = list_videos(args.input)
    if not files:
        sys.exit("没有找到视频文件")
    for src in files:
        dst = args.output if (args.output and len(files) == 1) \
            else os.path.splitext(src)[0] + "_blurred.mp4"
        print(f"处理: {src}")
        if not os.path.isfile(src):
            print("  文件不存在")
            continue
        dets = detect_all(src, args)
        people = people_path = None
        if args.pick:
            people, people_path, thumbs = prepare_people(src, dets, args)
            if thumbs is not None:
                try:
                    pick_people_dialog(None, people, thumbs, os.path.basename(src))
                except ImportError:
                    sys.exit("--pick 需要图形界面(tkinter)。请用 python.org 官方版 Python。")
                people.save(people_path)
            n_free = sum(not people.group_masked(k) for k in people.visible_groups())
            print(f"  不打码的人: {n_free} 个")
        index = ShapeIndex(dets, parts, args, people)
        marks_path = os.path.splitext(src)[0] + ".marks.json"
        marks = load_marks(marks_path)
        if marks:
            print(f"  已载入 {len(marks)} 个手动标记")
        if args.review:
            print("  检查窗口:空格 播放 | a/d 逐帧 | s/w 跳1秒 | m 没打码的帧 | n 人数变少的帧 | "
                  "t 不打码的人挨着别人 | k 骨架 | 右键点人 切换打码 | "
                  "拖框 补码 | 回车 结束标记 | u 撤销 | q 保存并输出")
            review(src, dets, index, marks, marks_path, args, people, people_path)
        render(src, dst, index, marks, args)


# ================================================================ 图形界面

class Cancelled(Exception):
    pass


PART_ROWS = [                        # (分组名, 整组键, [(部位键, 显示名), ...])
    ("头/躯干", None, [("head", "头部"), ("chest", "胸部"), ("torso", "躯干"),
                     ("crotch", "下体/臀部")]),
    ("右臂", "right_arm", [("right_upper_arm", "上臂"), ("right_forearm", "前臂"),
                         ("right_hand", "手")]),
    ("左臂", "left_arm", [("left_upper_arm", "上臂"), ("left_forearm", "前臂"),
                        ("left_hand", "手")]),
    ("右腿", "right_leg", [("right_thigh", "大腿"), ("right_calf", "小腿"),
                         ("right_foot", "脚")]),
    ("左腿", "left_leg", [("left_thigh", "大腿"), ("left_calf", "小腿"),
                        ("left_foot", "脚")]),
]
SETTINGS_FILE = os.path.join(HERE, "body_blur_settings.json")
REVIEW_HELP = ("检查窗口操作:\n\n"
               "空格  播放 / 暂停\n"
               "a / d  上一帧 / 下一帧\n"
               "s / w  后退 / 前进 1 秒\n"
               "m  跳到下一段「没有打码」的帧(Shift+M 上一段)\n"
               "n  跳到下一段「人数变少」的帧(Shift+N 上一段)\n"
               "    窗口顶部会显示这两类各还有多少段\n"
               "t  跳到下一段「不打码的人挨着别人」的帧(Shift+T 上一段)\n"
               "    两人交叉后编号可能对调,请务必检查这些地方\n"
               "k  显示 / 隐藏骨架\n"
               "鼠标右键点人  切换这个人打码 / 不打码(开启「选择要打码的人」时可用)\n"
               "鼠标拖框  补一个打码区域\n"
               "    拖一次后前进到结束位置按 回车 = 这段固定位置打码\n"
               "    在多个帧分别拖框 = 中间自动跟随移动\n"
               "u  撤销上一个标记\n"
               "q  保存并开始输出")


# 高级设置:(标签页, [(参数名, 控件, 名称, 取值范围/选项, 说明)])
ADV_TABS = [
    ("识别", [
        ("strict", "radio", "检测强度", [(False, "标准"), (True, "严格(更不易漏,速度慢很多)")],
         "严格模式会自动改用 heavy 模型、阈值 ≤0.2、打码范围 ≥50%、补遮 ≥30 帧并开启旋转检测,\n"
         "下面对应的设置会被覆盖。"),
        ("model_size", "combo", "姿态模型",
         [("lite", "轻量(最快)"), ("full", "标准"), ("heavy", "精确(最慢)")], ""),
        ("conf", "spin", "骨架检测阈值", (0.05, 0.9, 0.05), "越低越不容易漏人,误检也会变多"),
        ("max_people", "spin", "每帧最多识别人数", (1, 20, 1), "人多的视频要调大,检测会变慢"),
        ("face", "check", "同时用人脸检测补充头部(推荐)", None, ""),
        ("face_conf", "spin", "人脸检测阈值", (0.05, 0.9, 0.05), ""),
        ("rotate", "check", "旋转检测:躺着、倒着的人更不易漏(慢约 4 倍)", None, ""),
    ]),
    ("打码效果", [
        ("strength", "scale", "打码强度", (0.5, 3.0), "马赛克格子大小 / 模糊程度,越大越看不清"),
        ("feather", "scale", "边缘羽化", (0.0, 0.5), "0 = 硬边;黑块不羽化"),
        ("unmasked", "check", "完全没找到人的帧,整帧模糊(最保险)", None, ""),
    ]),
    ("防漏补遮", [
        ("hold", "spin", "检测中断时补遮帧数", (0, 120, 1),
         "人数变少或检测中断时,用前后这么多帧的位置补上"),
        ("smooth", "spin", "平滑帧数", (0, 10, 1), "平时也合并前后几帧,防止快速动作时边缘露出"),
    ]),
    ("人物分组", [
        ("match", "scale", "人脸归组门槛", (0.30, 0.70),
         "越高越严:认不准时宁可多拆几张卡片。\n"
         "调低会把长得像的人合成一组,陌生人可能跟着不打码,不建议调低。"),
    ]),
]
ADV_FIELDS = {f[0]: (f[1], f[3]) for _, fields in ADV_TABS for f in fields}
ADV_SHORT = {"strict": "严格模式", "model_size": "模型", "conf": "骨架阈值",
             "max_people": "最多人数", "face": "人脸补充", "face_conf": "人脸阈值",
             "rotate": "旋转检测", "strength": "强度", "feather": "羽化",
             "unmasked": "没人的帧整帧模糊", "hold": "补遮", "smooth": "平滑", "match": "归组门槛"}


def adv_defaults():
    d = vars(build_parser().parse_args([]))
    out = {k: d[k] for k in ADV_FIELDS}
    out["unmasked"] = d["unmasked"] == "blur"
    return out


ADV_DEFAULTS = adv_defaults()


def fmt_time(sec):
    sec = int(max(0, sec))
    if sec >= 3600:
        return f"{sec // 3600} 小时 {sec % 3600 // 60} 分"
    if sec >= 60:
        return f"{sec // 60} 分 {sec % 60} 秒"
    return f"{sec} 秒"


def run_gui(initial_files=()):
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError:
        sys.exit("当前 Python 没有 tkinter,无法打开界面。\n"
                 "请安装 python.org 官方版 Python(自带 tkinter),或改用命令行:\n"
                 "    python body_blur.py --help")
    import queue
    import threading
    import time
    import traceback

    class App:
        def __init__(self, root):
            self.root = root
            root.title("人体部位打码工具(本机处理)")
            root.minsize(760, 640)
            self.q = queue.Queue()
            self.stop_evt = threading.Event()
            self.worker = None
            self.items = {}              # tree item id -> 文件路径
            self.last_output_dir = None
            s = self.load_settings()

            pad = {"padx": 8, "pady": 4}
            # ---------- 文件列表
            ff = ttk.LabelFrame(root, text=" 1. 视频文件 ")
            ff.pack(fill="both", expand=True, **pad)
            self.tree = ttk.Treeview(ff, columns=("file", "status"), show="headings", height=6)
            self.tree.heading("file", text="文件")
            self.tree.heading("status", text="状态")
            self.tree.column("file", width=520)
            self.tree.column("status", width=150, anchor="center")
            sb = ttk.Scrollbar(ff, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
            sb.grid(row=0, column=1, sticky="ns", pady=6)
            bf = ttk.Frame(ff)
            bf.grid(row=0, column=2, sticky="n", padx=6, pady=6)
            self.file_buttons = []
            for text, cmd in (("添加视频…", self.add_files), ("添加文件夹…", self.add_folder),
                              ("移除选中", self.remove_selected), ("清空", self.clear_files)):
                b = ttk.Button(bf, text=text, command=cmd, width=12)
                b.pack(fill="x", pady=2)
                self.file_buttons.append(b)
            ff.columnconfigure(0, weight=1)
            ff.rowconfigure(0, weight=1)

            # ---------- 部位
            pf = ttk.LabelFrame(root, text=" 2. 打码部位 ")
            pf.pack(fill="x", **pad)
            saved_parts = set(s.get("parts", ["head"]))
            self.part_vars, self.group_vars = {}, {}
            for r, (label, gkey, items) in enumerate(PART_ROWS):
                ttk.Label(pf, text=label, width=8).grid(row=r, column=0, sticky="w", padx=6)
                col = 1
                if gkey:
                    gv = tk.BooleanVar(value=all(k in saved_parts for k, _ in items))
                    self.group_vars[gkey] = (gv, [k for k, _ in items])
                    ttk.Checkbutton(pf, text="整条", variable=gv,
                                    command=lambda g=gkey: self.toggle_group(g)).grid(
                        row=r, column=col, sticky="w", padx=4)
                col = 2
                for key, name in items:
                    v = tk.BooleanVar(value=key in saved_parts)
                    self.part_vars[key] = v
                    ttk.Checkbutton(pf, text=name, variable=v, command=self.sync_groups).grid(
                        row=r, column=col, sticky="w", padx=4)
                    col += 1
            ttk.Label(pf, text="左右指画面中那个人自己的左右;人背对镜头时模型偶尔会弄反,"
                               "纹身等关键位置建议两侧都勾。",
                      foreground="#666").grid(row=len(PART_ROWS), column=0, columnspan=7,
                                              sticky="w", padx=6, pady=(2, 6))

            # ---------- 设置(常用的放首页,其余进「高级设置」)
            sf = ttk.LabelFrame(root, text=" 3. 设置 ")
            sf.pack(fill="x", **pad)
            self.mode = tk.StringVar(value=s.get("mode", "pixel"))
            self.pad = tk.DoubleVar(value=s.get("pad", 0.3))
            self.pick = tk.BooleanVar(value=s.get("pick", False))
            self.review = tk.BooleanVar(value=s.get("review", True))
            self.outdir = tk.StringVar(value=s.get("outdir", ""))
            self.adv = {}
            for key, (kind, _d) in ADV_FIELDS.items():
                val = s.get(key, ADV_DEFAULTS[key])
                if key == "unmasked" and isinstance(val, str):      # 兼容旧设置文件
                    val = val == "blur"
                self.adv[key] = val

            ttk.Label(sf, text="打码方式").grid(row=0, column=0, sticky="w", padx=6, pady=2)
            for c, (val, txt) in enumerate((("pixel", "马赛克"), ("blur", "模糊"),
                                            ("solid", "黑块(最安全)")), start=1):
                ttk.Radiobutton(sf, text=txt, value=val, variable=self.mode).grid(
                    row=0, column=c, sticky="w", padx=4)
            ttk.Label(sf, text="打码范围").grid(row=1, column=0, sticky="w", padx=6, pady=2)
            self.pad_label = ttk.Label(sf, width=10)
            ttk.Scale(sf, from_=0.0, to=1.0, variable=self.pad, length=240,
                      command=lambda _=None: self.update_pad_label()).grid(
                row=1, column=1, columnspan=2, sticky="w", padx=4)
            self.pad_label.grid(row=1, column=3, sticky="w")
            self.update_pad_label()

            opts = ttk.Frame(sf)
            opts.grid(row=2, column=0, columnspan=5, sticky="w", padx=2, pady=2)
            ttk.Checkbutton(opts, text="选择要打码的人(多人视频:检测后逐个确认谁打码)",
                            variable=self.pick).grid(row=0, column=0, sticky="w", padx=4, pady=1)
            ttk.Checkbutton(opts, text="输出前打开检查窗口,人工检查补漏(推荐)",
                            variable=self.review).grid(row=1, column=0, sticky="w", padx=4, pady=1)

            of = ttk.Frame(sf)
            of.grid(row=3, column=0, columnspan=5, sticky="we", padx=6, pady=(4, 2))
            ttk.Label(of, text="输出到").pack(side="left")
            ttk.Entry(of, textvariable=self.outdir, width=52).pack(side="left", padx=6)
            ttk.Button(of, text="选择…", command=self.pick_outdir).pack(side="left")
            ttk.Label(of, text="(留空 = 原视频所在文件夹)", foreground="#666").pack(
                side="left", padx=6)

            af = ttk.Frame(sf)
            af.grid(row=4, column=0, columnspan=5, sticky="we", padx=6, pady=(2, 6))
            ttk.Button(af, text="高级设置…", command=self.open_advanced).pack(side="left")
            self.adv_label = ttk.Label(af, foreground="#666")
            self.adv_label.pack(side="left", padx=8)
            self.update_adv_label()

            # ---------- 进度
            gf = ttk.LabelFrame(root, text=" 4. 进度 ")
            gf.pack(fill="x", **pad)
            self.cur_label = ttk.Label(gf, text="就绪")
            self.cur_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0))
            ttk.Label(gf, text="当前视频").grid(row=1, column=0, sticky="w", padx=6)
            self.pb_file = ttk.Progressbar(gf, length=560, maximum=1000)
            self.pb_file.grid(row=1, column=1, sticky="we", padx=6, pady=3)
            ttk.Label(gf, text="全部").grid(row=2, column=0, sticky="w", padx=6)
            self.pb_all = ttk.Progressbar(gf, length=560, maximum=1000)
            self.pb_all.grid(row=2, column=1, sticky="we", padx=6, pady=3)
            self.detail = ttk.Label(gf, text="", foreground="#444")
            self.detail.grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))
            gf.columnconfigure(1, weight=1)

            # ---------- 按钮 + 日志
            act = ttk.Frame(root)
            act.pack(fill="x", **pad)
            self.btn_start = ttk.Button(act, text="开始处理", command=self.start)
            self.btn_start.pack(side="left")
            self.btn_stop = ttk.Button(act, text="停止", command=self.stop, state="disabled")
            self.btn_stop.pack(side="left", padx=6)
            ttk.Button(act, text="打开输出文件夹", command=self.open_outdir).pack(side="left")

            lf = ttk.Frame(root)
            lf.pack(fill="both", expand=False, padx=8, pady=(0, 8))
            self.log_box = tk.Text(lf, height=6, state="disabled", wrap="word",
                                   background="#f7f7f7", relief="flat")
            lsb = ttk.Scrollbar(lf, orient="vertical", command=self.log_box.yview)
            self.log_box.configure(yscrollcommand=lsb.set)
            self.log_box.pack(side="left", fill="both", expand=True)
            lsb.pack(side="right", fill="y")

            for f in initial_files:
                self.add_path(f)
            root.protocol("WM_DELETE_WINDOW", self.on_close)
            root.after(100, self.poll)
            threading.Thread(target=self.check_deps, daemon=True).start()
            self.log("视频只在本机处理,不会上传。首次运行会自动下载识别模型。")

        def check_deps(self):
            try:
                import mediapipe  # noqa: F401
                from mediapipe.tasks.python import vision  # noqa: F401
            except Exception as e:      # 版本冲突时常见 ImportError / initialization failed
                self.q.put(("deps_error", f"{type(e).__name__}: {e}"))

        # ----- 小工具
        def load_settings(self):
            try:
                with open(SETTINGS_FILE, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return {}

        def save_settings(self):
            data = {"parts": self.selected_parts(), "mode": self.mode.get(),
                    "pad": round(self.pad.get(), 2), "pick": self.pick.get(),
                    "review": self.review.get(), "outdir": self.outdir.get()}
            data.update(self.adv)
            try:
                with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False)
            except OSError:
                pass

        def update_adv_label(self):
            items = []
            for key in ADV_FIELDS:
                v, d = self.adv[key], ADV_DEFAULTS[key]
                if v == d or (isinstance(d, float) and abs(float(v) - d) < 1e-6):
                    continue
                name = ADV_SHORT[key]
                if isinstance(d, bool):
                    items.append(name if v else "关闭" + name)
                elif key == "strength":
                    items.append(f"{name} ×{float(v):.1f}")
                else:
                    items.append(f"{name} {v:g}" if isinstance(v, (int, float)) else f"{name} {v}")
            self.adv_label.configure(text="当前:" + ("、".join(items) if items else "全部默认"))

        def open_advanced(self):
            top = tk.Toplevel(self.root)
            top.title("高级设置")
            top.transient(self.root)
            top.resizable(False, False)
            nb = ttk.Notebook(top)
            nb.pack(fill="both", expand=True, padx=10, pady=(10, 4))
            tvars = {}

            def scale_row(fr, r, key, spec, v):
                lab = ttk.Label(fr, width=6)

                def upd(_=None):
                    x = v.get()
                    lab.configure(text=f"×{x:.1f}" if key == "strength" else f"{x:.2f}")
                ttk.Scale(fr, from_=spec[0], to=spec[1], variable=v, length=220,
                          command=upd).grid(row=r, column=1, sticky="w", padx=4)
                lab.grid(row=r, column=2, sticky="w")
                upd()
                return upd

            refreshers = []
            for tab, fields in ADV_TABS:
                fr = ttk.Frame(nb, padding=12)
                nb.add(fr, text=" %s " % tab)
                r = 0
                for key, kind, label, spec, hint in fields:
                    val = self.adv[key]
                    if kind == "check":
                        v = tk.BooleanVar(value=bool(val))
                        ttk.Checkbutton(fr, text=label, variable=v).grid(
                            row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))
                    else:
                        ttk.Label(fr, text=label).grid(row=r, column=0, sticky="w",
                                                       pady=(6, 0), padx=(0, 8))
                    if kind == "radio":
                        v = tk.BooleanVar(value=bool(val))
                        rf = ttk.Frame(fr)
                        rf.grid(row=r, column=1, columnspan=2, sticky="w", pady=(6, 0))
                        for opt, txt in spec:
                            ttk.Radiobutton(rf, text=txt, value=opt, variable=v).pack(
                                side="left", padx=(0, 8))
                    elif kind == "combo":
                        names = dict(spec)
                        v = tk.StringVar(value=names.get(val, val))
                        ttk.Combobox(fr, textvariable=v, values=[n for _, n in spec],
                                     state="readonly", width=14).grid(
                            row=r, column=1, sticky="w", padx=4, pady=(6, 0))
                    elif kind == "spin":
                        v = tk.StringVar(value=f"{val:g}")
                        ttk.Spinbox(fr, from_=spec[0], to=spec[1], increment=spec[2],
                                    textvariable=v, width=8).grid(
                            row=r, column=1, sticky="w", padx=4, pady=(6, 0))
                    elif kind == "scale":
                        v = tk.DoubleVar(value=float(val))
                        refreshers.append(scale_row(fr, r, key, spec, v))
                    tvars[key] = v
                    r += 1
                    if hint:
                        ttk.Label(fr, text=hint, foreground="#666", justify="left").grid(
                            row=r, column=0, columnspan=3, sticky="w", padx=(12, 0))
                        r += 1

            def read(key):
                kind, spec = ADV_FIELDS[key]
                v, d = tvars[key], ADV_DEFAULTS[key]
                if kind in ("check", "radio"):
                    return bool(v.get())
                if kind == "combo":
                    back = {n: k for k, n in spec}
                    return back.get(v.get(), d)
                try:
                    x = float(v.get())
                except (ValueError, tk.TclError):
                    return self.adv[key]
                lo, hi = spec[0], spec[1]
                x = min(hi, max(lo, x))
                return int(round(x)) if isinstance(d, int) else round(x, 2)

            def defaults():
                for key, v in tvars.items():
                    kind, spec = ADV_FIELDS[key]
                    d = ADV_DEFAULTS[key]
                    if kind == "combo":
                        v.set(dict(spec).get(d, d))
                    elif kind == "spin":
                        v.set(f"{d:g}")
                    else:
                        v.set(d)
                for f in refreshers:
                    f()

            def ok():
                for key in tvars:
                    self.adv[key] = read(key)
                self.update_adv_label()
                self.save_settings()
                top.destroy()

            bar = ttk.Frame(top)
            bar.pack(fill="x", padx=10, pady=(4, 10))
            ttk.Button(bar, text="恢复默认", command=defaults).pack(side="left")
            ttk.Button(bar, text="确定", command=ok).pack(side="right")
            ttk.Button(bar, text="取消", command=top.destroy).pack(side="right", padx=6)
            top.grab_set()

        def update_pad_label(self):
            self.pad_label.configure(text=f"外扩 {int(round(self.pad.get() * 100))}%")

        def toggle_group(self, gkey):
            gv, keys = self.group_vars[gkey]
            for k in keys:
                self.part_vars[k].set(gv.get())

        def sync_groups(self):
            for gv, keys in self.group_vars.values():
                gv.set(all(self.part_vars[k].get() for k in keys))

        def selected_parts(self):
            return [k for k in BASE_PARTS if k in self.part_vars and self.part_vars[k].get()]

        def log(self, text):
            text = text.replace("\r", "").strip()
            if not text:
                return
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        def add_path(self, path):
            path = os.path.abspath(path)
            if path in self.items.values() or not os.path.isfile(path):
                return
            iid = self.tree.insert("", "end", values=(path, "等待"))
            self.items[iid] = path

        def add_files(self):
            exts = " ".join("*" + e for e in sorted(VIDEO_EXTS))
            for f in filedialog.askopenfilenames(
                    title="选择视频", filetypes=[("视频文件", exts), ("所有文件", "*.*")]):
                self.add_path(f)

        def add_folder(self):
            d = filedialog.askdirectory(title="选择文件夹")
            if d:
                vids = list_videos(d)
                for f in vids:
                    self.add_path(f)
                if not vids:
                    messagebox.showinfo("提示", "这个文件夹里没有找到视频文件。")

        def remove_selected(self):
            for iid in self.tree.selection():
                self.tree.delete(iid)
                self.items.pop(iid, None)

        def clear_files(self):
            for iid in list(self.items):
                self.tree.delete(iid)
            self.items.clear()

        def pick_outdir(self):
            d = filedialog.askdirectory(title="选择输出文件夹")
            if d:
                self.outdir.set(d)

        def open_outdir(self):
            d = self.outdir.get() or self.last_output_dir
            if not d and self.items:
                d = os.path.dirname(next(iter(self.items.values())))
            if not d or not os.path.isdir(d):
                messagebox.showinfo("提示", "还没有输出文件夹。")
                return
            if sys.platform.startswith("win"):
                os.startfile(d)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])

        def set_busy(self, busy):
            self.btn_start.configure(state="disabled" if busy else "normal")
            self.btn_stop.configure(state="normal" if busy else "disabled")
            for b in self.file_buttons:
                b.configure(state="disabled" if busy else "normal")

        # ----- 开始 / 停止
        def make_args(self):
            args = build_parser().parse_args([])
            for k, v in self.adv.items():
                setattr(args, k, v)
            args.unmasked = "blur" if self.adv["unmasked"] else "keep"
            args.mode = self.mode.get()
            args.pad = float(self.pad.get())
            args.pick = self.pick.get()
            args.review = self.review.get()
            apply_strict(args)
            return args

        def start(self):
            parts = self.selected_parts()
            if not self.items:
                messagebox.showwarning("提示", "请先添加视频。")
                return
            if not parts:
                messagebox.showwarning("提示", "请至少勾选一个打码部位。")
                return
            outdir = self.outdir.get().strip()
            if outdir and not os.path.isdir(outdir):
                messagebox.showwarning("提示", "输出文件夹不存在。")
                return
            self.save_settings()
            args = self.make_args()
            jobs = list(self.items.items())
            for iid, _ in jobs:
                self.tree.set(iid, "status", "等待")
            self.stop_evt.clear()
            self.set_busy(True)
            self.pb_all["value"] = self.pb_file["value"] = 0
            names = {k: (lab if lab == "头/躯干" else lab) for lab, _, its in PART_ROWS
                     for k, _ in its}
            disp = {k: (n if names[k] == "头/躯干" else names[k] + n) for _, _, its in PART_ROWS
                    for k, n in its}
            self.log("打码部位:" + "、".join(disp.get(k, k) for k in parts))
            self.worker = threading.Thread(target=self.work, args=(jobs, parts, args, outdir),
                                           daemon=True)
            self.worker.start()

        def stop(self):
            if self.worker and self.worker.is_alive():
                self.stop_evt.set()
                self.cur_label.configure(text="正在停止…")

        def on_close(self):
            if self.worker and self.worker.is_alive():
                if not messagebox.askyesno("确认", "正在处理中,确定要退出吗?"):
                    return
                self.stop_evt.set()
            self.save_settings()
            self.root.destroy()

        # ----- 后台线程(不能直接操作界面,全部通过队列发消息)
        def work(self, jobs, parts, args, outdir):
            global LOG
            LOG = lambda t: self.q.put(("log", t))
            n_jobs = len(jobs)
            n_ok = n_fail = 0
            for k, (iid, src) in enumerate(jobs):
                if self.stop_evt.is_set():
                    break
                base = os.path.splitext(os.path.basename(src))[0] + "_blurred.mp4"
                dst = os.path.join(outdir or os.path.dirname(src), base)
                self.q.put(("start_file", iid, k, n_jobs, src))
                stage_t = {}

                def progress(stage, n, total, k=k):
                    if self.stop_evt.is_set():
                        raise Cancelled()
                    now = time.time()
                    t0 = stage_t.setdefault(stage, now)
                    last = stage_t.get(stage + "_last", 0)
                    if stage != "encode" and n < total and now - last < 0.15:
                        return
                    stage_t[stage + "_last"] = now
                    self.q.put(("progress", k, n_jobs, stage, n, total, now - t0))

                try:
                    dets = detect_all(src, args, progress, LOG)
                    people = people_path = None
                    if args.pick:
                        people, people_path, thumbs = prepare_people(src, dets, args,
                                                                     progress, LOG)
                        if thumbs is not None:
                            done = threading.Event()
                            self.q.put(("pick", iid, src, people, thumbs, done))
                            done.wait()
                            if self.stop_evt.is_set():
                                raise Cancelled()
                            people.save(people_path)
                        n_free = sum(not people.group_masked(k)
                                     for k in people.visible_groups())
                        LOG(f"  不打码的人:{n_free} 个" if n_free else "  所有人都打码")
                    index = ShapeIndex(dets, parts, args, people)
                    marks_path = os.path.splitext(src)[0] + ".marks.json"
                    marks = load_marks(marks_path)
                    if marks:
                        LOG(f"  已载入 {len(marks)} 个之前保存的手动标记")
                    if args.review:
                        done = threading.Event()
                        self.q.put(("review", iid, src, dets, index, marks, marks_path, args,
                                    people, people_path, done))
                        done.wait()
                        if self.stop_evt.is_set():
                            raise Cancelled()
                    ok = render(src, dst, index, marks, args, progress, LOG)
                    n_ok += bool(ok)
                    n_fail += not ok
                    self.q.put(("file_done", iid, "完成 ✓" if ok else "失败", dst if ok else None))
                except Cancelled:
                    self.q.put(("file_done", iid, "已停止", None))
                    break
                except Exception as e:
                    n_fail += 1
                    LOG(f"  出错:{e}\n{traceback.format_exc(limit=3)}")
                    if isinstance(e, ImportError) or "numpy" in str(e) or \
                            "initialization failed" in str(e):
                        LOG("  这是依赖库版本冲突。请在「命令提示符」依次运行:\n" + FIX_CMD)
                    self.q.put(("file_done", iid, "失败", None))
            self.q.put(("all_done", n_ok, n_fail))

        # ----- 主线程:处理队列消息,更新界面
        def poll(self):
            try:
                while True:
                    self.handle(self.q.get_nowait())
            except queue.Empty:
                pass
            self.root.after(100, self.poll)

        def handle(self, msg):
            kind = msg[0]
            if kind == "log":
                self.log(msg[1])
            elif kind == "start_file":
                _, iid, k, n, src = msg
                self.tree.set(iid, "status", "处理中…")
                self.tree.see(iid)
                self.cur_label.configure(text=f"[{k + 1}/{n}] {os.path.basename(src)}")
                self.log(f"处理:{src}")
                self.pb_file["value"] = 0
                self.pb_all["value"] = 1000 * k / n
            elif kind == "progress":
                _, k, n_jobs, stage, n, total, elapsed = msg
                if stage == "detect":
                    frac, name = (0.7 * n / total if total else 0), "识别人体"
                elif stage == "people":
                    frac, name = 0.7, "准备人物截图"
                elif stage == "render":
                    frac, name = (0.7 + 0.27 * n / total if total else 0.7), "生成打码视频"
                else:
                    frac, name = 0.97, "压缩并合成声音"
                self.pb_file["value"] = 1000 * frac
                self.pb_all["value"] = 1000 * (k + frac) / n_jobs
                if stage == "encode":
                    self.detail.configure(text=f"{name}…")
                else:
                    txt = f"{name}:{n}/{total} 帧"
                    if elapsed > 1 and n > 0:
                        fps = n / elapsed
                        txt += f"   速度 {fps:.1f} 帧/秒"
                        if total > n:
                            txt += f"   本步剩余约 {fmt_time((total - n) / fps)}"
                    self.detail.configure(text=txt)
            elif kind == "review":
                _, iid, src, dets, index, marks, marks_path, args, people, people_path, done = msg
                self.tree.set(iid, "status", "人工检查中")
                self.detail.configure(text="人工检查中(检查窗口里按 q 保存并继续)")
                messagebox.showinfo("人工检查", REVIEW_HELP)
                try:
                    review(src, dets, index, marks, marks_path, args, people, people_path)
                except Exception as e:
                    self.log(f"  检查窗口出错:{e}")
                self.tree.set(iid, "status", "处理中…")
                done.set()
            elif kind == "pick":
                _, iid, src, people, thumbs, done = msg
                self.tree.set(iid, "status", "选择打码的人")
                self.detail.configure(text="请在弹出的窗口里选择哪些人不打码")
                try:
                    pick_people_dialog(self.root, people, thumbs, os.path.basename(src))
                except Exception as e:
                    self.log(f"  人物选择窗口出错:{e}(全部按打码处理)")
                    people.skip.clear()
                self.tree.set(iid, "status", "处理中…")
                done.set()
            elif kind == "file_done":
                _, iid, status, dst = msg
                self.tree.set(iid, "status", status)
                if dst:
                    self.last_output_dir = os.path.dirname(dst)
                    self.pb_file["value"] = 1000
            elif kind == "all_done":
                _, n_ok, n_fail = msg
                self.set_busy(False)
                stopped = self.stop_evt.is_set()
                if stopped:
                    self.cur_label.configure(text="已停止")
                    self.detail.configure(text="")
                elif n_fail:
                    self.cur_label.configure(text=f"完成 {n_ok} 个,失败 {n_fail} 个")
                    self.detail.configure(text="失败原因见下方日志。")
                    self.log(f"完成 {n_ok} 个,失败 {n_fail} 个。")
                    messagebox.showerror("有视频处理失败",
                                         f"成功 {n_ok} 个,失败 {n_fail} 个。\n失败原因见窗口下方日志。")
                else:
                    self.pb_all["value"] = 1000
                    self.cur_label.configure(text="全部完成")
                    self.detail.configure(text="请把输出视频完整看一遍,确认没有漏打再保存或分享。")
                    self.log("全部完成。")
                    messagebox.showinfo("完成", "全部处理完成。\n\n请把输出视频完整看一遍,"
                                              "确认没有漏打再保存或分享。")
            elif kind == "deps_error":
                self.log("依赖库有问题:" + msg[1])
                self.log("请在「命令提示符」依次运行下面两行,然后重新打开本程序:\n" + FIX_CMD)
                messagebox.showerror("依赖库有问题",
                                     f"{msg[1]}\n\n请打开「命令提示符」依次运行下面两行,"
                                     f"然后重新打开本程序(命令已显示在窗口下方日志里,可直接复制):"
                                     f"\n\n{FIX_CMD}")

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform.startswith("win") else
                              "aqua" if sys.platform == "darwin" else "clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
