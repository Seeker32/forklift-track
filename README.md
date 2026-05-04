# Forklift Entry/Exit Direction Tracking

基于 YOLO 目标检测与 ByteTrack 多目标跟踪的叉车进出方向识别系统。通过分析视频流中叉车穿越预设虚拟线的轨迹，自动判断叉车是驶入（in）还是驶出（out）监控区域，并输出结构化事件。

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
│   ├── tracker.py           # ByteTrack 多目标跟踪器
│   ├── direction.py         # 进出方向判断（虚拟线穿越检测）
│   ├── geometry.py          # 二维几何计算（点线位置、距离、穿越判定）
│   ├── line_tool.py         # 交互式虚拟线标定工具
│   └── events.py            # 事件数据结构
├── tests/                   # 单元测试（714 行）
├── models/                  # YOLO 模型权重文件
├── outputs/                 # 检测结果输出
└── videos/                  # 输入视频文件
```

## 快速开始

### 环境要求

- Python >= 3.10
- 依赖：ultralytics, yolov5, torch, opencv-python, numpy, pyyaml

### 安装

```bash
pip install -r requirements.txt
# 或
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
    line_width: 80          # 线宽（构成带状穿越区）
    in_direction: [-1, 0]   # 驶入方向向量，详见下方说明
    model_path: models/best.pt
    confidence: 0.4
    class_name: forklift_2
    max_missing_frames: 30
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

```

**运行检测**

```bash
python main.py --config config/cameras.yaml
```

**标定虚拟线**

通过交互界面在视频帧上点击两点生成线路配置：

```bash
python main.py --select-line videos/example.mp4
```

## 工作原理

1. **检测** — `ForkliftDetector` 使用 YOLO 模型逐帧检测叉车，支持 ultralytics YOLO 和旧版 YOLOv5 两种模型格式，自动识别并适配加载方式
2. **跟踪** — `ByteTrackTracker` 对检测结果进行多目标跟踪，为每个叉车分配稳定的 track_id
3. **方向判断** — `DirectionDetector` 维护每个目标的运动轨迹，判断目标是否穿越虚拟线带状区域（线宽可配置），并结合预设的 `in_direction` 向量判定穿越方向为 in 或 out
4. **事件输出** — 每次方向判定结果以结构化事件输出，包含 camera_id、track_id、direction、timestamp 和 bbox

## 运行测试

```bash
pytest tests/ -v
```
