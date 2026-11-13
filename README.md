# Blur-by-body-part 人体部位打码工具

[中文](#中文) | [English](#english)

---

## 中文

在本机逐帧识别视频中的人体骨架,按部位自动打码。视频不会上传到任何地方。

- 按部位打码:头部、胸部、躯干、下体/臀部、左右上臂/前臂/手、左右大腿/小腿/脚
- 头部按骨架定位:背对镜头、侧脸、低头、戴口罩也能盖住;另叠加人脸检测器补充
- 宁可多打不要漏:检测中断或人数变少时,自动用前后帧的位置补遮
- 只给部分人打码:检测后弹出人物选择面板(截图 + 骨架),逐个确认谁打码;同一个人的多段画面按人脸自动归组
- 人工检查窗口:`m` 跳到没打码的帧,`n` 跳到人数变少的帧,鼠标拖框补码
- 图形界面 + 进度条,也支持命令行
- 马赛克 / 模糊 / 黑块三种方式;输出时去除拍摄时间、GPS 等元数据

### 安装

需要 Python 3.10–3.12。建议使用独立环境,避免和 Anaconda 等已有的库冲突:

```
conda create -n blur python=3.12 -y
conda activate blur
set PYTHONNOUSERSITE=1
pip install mediapipe
conda install -c conda-forge ffmpeg -y
```

没有 ffmpeg 也能运行,但输出视频没有声音。

### 使用

```
python body_blur.py
```

不带参数即打开图形界面。Windows 下也可以双击 `启动打码工具.bat`,首次运行会自动建立独立环境并安装依赖。

命令行用法:

```
python body_blur.py 视频.mp4 --parts 头,右臂 --review --mode solid
python body_blur.py --help
python body_blur.py --list-parts
```

首次运行会自动下载姿态模型(MediaPipe Pose Landmarker)和人脸模型(YuNet)。

### 人工检查窗口按键

| 按键 | 作用 |
|---|---|
| 空格 | 播放 / 暂停 |
| a / d | 上一帧 / 下一帧 |
| s / w | 后退 / 前进 1 秒 |
| m / Shift+M | 下一段 / 上一段没有打码的帧 |
| n / Shift+N | 下一段 / 上一段人数变少的帧 |
| t / Shift+T | 下一段 / 上一段「不打码的人挨着别人」的帧(交叉后编号可能对调,务必检查) |
| 鼠标右键点人 | 切换这个人打码 / 不打码(需开启「选择要打码的人」) |
| k | 显示 / 隐藏骨架 |
| 鼠标拖框 | 补一个打码区域;在多个帧拖框会自动插值跟随 |
| 回车 | 结束当前手动标记 |
| u | 撤销上一个标记 |
| q | 保存并输出 |

### 只给部分人打码

界面勾选「选择要打码的人」,或命令行加 `--pick`。检测完成后会弹出人物选择面板:

- 每个人一张卡片,显示截图和骨架,默认全部打码,取消勾选的人在整段视频里都不打码
- 同一个人走出画面又回来、被挡住等产生的多段画面,会按人脸特征自动归成一组;归错时点「拆开」
- 一直没露脸的片段认不出是谁,会单独列出,默认打码;很短的零碎片段始终打码
- 两个人靠得很近时,为防止跟踪编号对调,两个人都会打码;检查窗口里按 `t` 逐段检查这些地方
- 选择结果保存在视频旁的 `.people.json`,下次处理同一个视频会自动沿用

首次使用会额外下载人脸识别模型(SFace,约 37MB)。

### 设置

首页只放常用设置(打码方式、打码范围、选人、人工检查、输出位置),其余在「高级设置」里:检测强度、姿态模型、检测阈值、每帧最多人数、人脸补充、旋转检测、打码强度、边缘羽化、补遮帧数、人脸归组门槛等。高级设置里改过的项目会显示在按钮旁边。

### 注意

- 左右指画面中人物自己的左右;背对镜头时模型偶尔会弄反,关键部位建议两侧都选。
- 没有检测器能做到 100%,输出后请完整看一遍再使用。
- 处理后视频旁会留下 `.pose.json`(骨架坐标缓存)、`.marks.json`(手动标记)和 `.people.json`(选人结果),不需要时可以删除。开启选人时,`.pose.json` 里还保存了人脸特征,删除该文件即可清除。

---

## English

A local, frame-by-frame video anonymizer. It detects each person's skeleton and blurs the body parts you choose. Videos never leave your computer.

- Per-part masking: head, chest, torso, crotch/buttocks, left/right upper arm, forearm and hand, left/right thigh, calf and foot
- Head located from the skeleton, so it is covered even when the person faces away, is in profile, looks down or wears a mask; a face detector is added on top for extra coverage
- Errs on the side of over-masking: when detection drops out or a person goes missing, positions from nearby frames are used to keep them covered
- Mask only some people: after detection, a panel shows each person (snapshot + skeleton) so you can choose who to mask; segments of the same person are grouped by face
- Manual review window: `m` jumps to unmasked frames, `n` to frames where people went missing; drag a box to add a mask
- GUI with progress bars, plus a command-line mode
- Mosaic, blur or solid black; shooting time, GPS and other metadata are stripped from the output

The GUI and log messages are in Chinese. On the command line, part names can be given in English or Chinese.

### Installation

Requires Python 3.10–3.12. A separate environment is recommended, to avoid conflicts with Anaconda or other installed packages:

```
conda create -n blur python=3.12 -y
conda activate blur
set PYTHONNOUSERSITE=1
pip install mediapipe
conda install -c conda-forge ffmpeg -y
```

(`set` is for Windows; on macOS/Linux use `export PYTHONNOUSERSITE=1`.) The tool works without ffmpeg, but the output will have no audio.

### Usage

```
python body_blur.py
```

Run without arguments to open the GUI. On Windows you can also double-click `启动打码工具.bat` ("launch tool"); on first run it creates a separate environment and installs the dependencies.

Command line:

```
python body_blur.py video.mp4 --parts head,right_arm --review --mode solid
python body_blur.py --help
python body_blur.py --list-parts
```

Part names: `head`, `chest`, `torso`, `crotch`, `right_arm` / `left_arm` (or `right_upper_arm`, `right_forearm`, `right_hand`, …), `right_leg` / `left_leg` (or `right_thigh`, `right_calf`, `right_foot`, …), plus the groups `arms`, `hands`, `legs`, `body`.

On first run the pose model (MediaPipe Pose Landmarker) and face model (YuNet) are downloaded automatically.

### Review window keys

| Key | Action |
|---|---|
| Space | Play / pause |
| a / d | Previous / next frame |
| s / w | Back / forward 1 second |
| m / Shift+M | Next / previous run of unmasked frames |
| n / Shift+N | Next / previous run of frames where people went missing |
| t / Shift+T | Next / previous run of frames where an unmasked person is close to someone else (IDs may swap after crossing; check these) |
| Right-click a person | Toggle masking for that person (requires person selection) |
| k | Show / hide skeleton |
| Drag with mouse | Add a mask box; boxes drawn on several frames are interpolated between them |
| Enter | Finish the current manual mark |
| u | Undo the last mark |
| q | Save and render |

### Masking only some people

Tick 「选择要打码的人」 ("choose people to mask") in the GUI, or add `--pick` on the command line. After detection, a person-selection panel opens:

- One card per person, with snapshots and skeleton. Everyone is masked by default; untick a person to leave them unmasked for the whole video
- When the same person leaves and re-enters the frame or is briefly hidden, the resulting segments are grouped automatically by face. If a card mixes up two people, click 「拆开」 ("split")
- Segments where the face is never visible cannot be identified; they are listed separately and masked by default. Very short fragments are always masked
- When two people are very close, both are masked, in case the tracker swaps their IDs. Press `t` in the review window to check those places
- Choices are saved to `.people.json` next to the video and reused next time

The first use downloads an additional face recognition model (SFace, about 37MB).

### Settings

The main window keeps only the common settings (masking style, mask size, person selection, review, output folder). Everything else is under 「高级设置」 ("advanced settings"): detection preset, pose model, thresholds, max people per frame, face detector, rotation detection, masking strength, edge feathering, gap filling and the face grouping threshold. Changed advanced settings are listed next to the button.

### Notes

- Left and right refer to the person's own left and right. When someone faces away, the model occasionally swaps them, so for important areas select both sides.
- No detector is 100% reliable. Watch the whole output before using it.
- Processing leaves a `.pose.json` (cached skeleton coordinates), a `.marks.json` (manual marks) and a `.people.json` (person choices) next to each video. Delete them when no longer needed. With person selection on, `.pose.json` also stores face features; deleting it removes them.

---

## 许可 / License

[MIT](LICENSE)。本工具按原样提供,不保证能遮住所有需要遮挡的内容,使用者需自行检查输出结果。
Provided as is, with no guarantee that everything that should be masked will be. Always check the output yourself.

模型在运行时从官方地址下载,不包含在本仓库中:MediaPipe Pose Landmarker(Apache 2.0)、YuNet(MIT)、SFace(Apache 2.0)。
Models are downloaded at runtime and are not included in this repository: MediaPipe Pose Landmarker (Apache 2.0), YuNet (MIT), SFace (Apache 2.0).
