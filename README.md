# Forklift Entry/Exit Direction Tracking

基于 YOLO / ONNX 目标检测与 ByteTrack 多目标跟踪的叉车进出方向识别系统。通过分析视频流中叉车穿越预设虚拟线的轨迹，自动判断叉车是驶入（in）还是驶出（out）监控区域，并输出结构化事件。

## 项目结构

```
track/
├── main.py                  # CLI 入口
├── pyproject.toml           # 项目依赖配置
├── config/
│   └── cameras.yaml         # 摄像头及检测区域配置
├── src/
│   ├── pipeline.py          # 检测跟踪方向判断流水线
│   ├── detector.py          # YOLO 叉车检测器（兼容 ultralytics / YOLOv5）
│   ├── detector_onnx.py     # ONNX 叉车检测器（RF-DETR 模型）
│   ├── tracker.py           # ByteTrack 多目标跟踪器
│   ├── direction.py         # 进出方向判断（虚拟线穿越检测）
│   ├── geometry.py          # 二维几何计算（点线位置、距离、穿越判定）
│   ├── line_tool.py         # 交互式虚拟线标定工具
│   ├── debug_video.py       # Debug 视频标注输出
│   └── events.py            # 事件数据结构
├── tests/                   # 单元测试
├── scripts/
│   ├── export_onnx.py       # 模型导出为 ONNX 格式
│   └── test_onnx.py         # ONNX 模型测试
├── models/                  # 模型权重文件
├── outputs/                 # 检测结果输出
└── videos/                  # 输入视频文件
```

## 快速开始

### 环境要求

Python >= 3.11

### 安装

```bash
uv sync
```

### 使用

**配置摄像头**

编辑 `config/cameras.yaml`：

```yaml
cameras:
  - camera_id: gate_01
    source: videos/example.mp4
    line:
      start: [923, 403]    # 虚拟线起点像素坐标
      end: [1164, 430]      # 虚拟线终点像素坐标
    line_width: 80          # 线宽，构成带状穿越区（默认 40）
    in_direction: [-1, 0]   # 驶入方向向量，详见下方说明
    model_path: models/best.pt  # 支持 .pt（YOLO）和 .onnx（RF-DETR）
    confidence: 0.4          # 检测置信度阈值（默认 0.4）
    class_name: forklift_2   # 目标类别名，仅保留该类别的检测结果（默认 forklift_2）
    max_missing_frames: 30   # 跟踪目标丢失后最大保留帧数（默认 30）
```

### in_direction 方向向量说明

`in_direction` 定义"驶入"方向的参考向量。当叉车穿越虚拟线时，系统计算叉车的实际移动向量，并与 `in_direction` 做点积：

- 点积 > 0 → 移动方向与参考向量同向 → 判定为 **in**
- 点积 ≤ 0 → 移动方向与参考向量反向 → 判定为 **out**

向量的模长不影响判定结果，仅方向起作用。图像坐标系原点为左上角，x 轴向右，y 轴向下。

| 向量值 | 含义 |
|--------|------|
| `[1, 0]` | 向右为 in → |
| `[-1, 0]` | 向左为 in ← |
| `[0, 1]` | 向下为 in ↓ |
| `[0, -1]` | 向上为 in ↑ |
| `[1, 1]` | 右下为 in ↘ |
| `[-1, 1]` | 左下为 in ↙ |
| `[1, -1]` | 右上为 in ↗ |
| `[-1, -1]` | 左上为 in ↖ |



**运行检测**

```bash
python main.py --config config/cameras.yaml
```

**命令行参数**

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（默认 `config/cameras.yaml`） |
| `--model-path` | 覆盖所有摄像头的模型路径 |
| `--select-line <source>` | 打开视频源，点击两点生成虚拟线配置 |
| `--debug` | 终端输出帧级调试信息（检测数、跟踪数、穿越状态） |
| `--debug-every N` | 调试信息输出间隔帧数（默认 1） |
| `--debug-video` | 输出完整标注视频到 `outputs/debug_<camera_id>.mp4` |

**输出 debug 标注视频**

```bash
python main.py --config config/cameras.yaml --debug-video
```

视频标注内容：
- 红色线段：虚拟线（line）
- 黄色半透明区域：线宽穿越区（line_zone）
- 绿色框 + 标签：检测框，显示 `class_name score`
- 蓝色框 + 青色圆心 + 标签：跟踪框，显示 `id track_id`
- 红色文字：方向事件，显示 `event in/out id track_id`
- 左上角：摄像头 ID、帧号、检测数和跟踪数

**终端 debug 输出**

```bash
python main.py --config config/cameras.yaml --debug
```

**标定虚拟线**

通过交互界面在视频帧上点击两点生成线路配置，按 Enter 确认、按 Esc 取消：

```bash
python main.py --select-line <video_source>
```

**ONNX 模型**

支持 RF-DETR 导出的 ONNX 模型。在配置中指定 `.onnx` 后缀的模型路径即可自动切换为 ONNX 推理，无需额外配置。

```yaml
model_path: models/inference_model.onnx
```

## 工作原理

1. **检测** — 根据 `model_path` 后缀自动选择 YOLO（`.pt`）或 ONNX（`.onnx`）检测器，逐帧检测叉车
2. **过滤** — 根据 `class_name` 配置仅保留指定类别的检测结果
3. **跟踪** — `ByteTrackTracker` 对过滤后的检测结果进行多目标跟踪，为每个叉车分配稳定的 track_id
4. **方向判断** — `DirectionDetector` 维护每个目标的运动轨迹，判断目标是否穿越虚拟线带状区域，并结合 `in_direction` 向量判定穿越方向为 in 或 out
5. **事件输出** — 每次方向判定结果以结构化事件输出（JSON 格式），包含 camera_id、track_id、direction、timestamp 和 bbox

## 运行测试

```bash
PYTHONPATH=. pytest tests/ -v
```
