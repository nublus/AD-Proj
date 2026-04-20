# CS5493 Group Project: Autonomous Driving Perception Pipeline

本 README 根据 [Project_Instruction.pdf](./Project_Instruction.pdf) 整理，作为项目开发与提交说明。

## 1. Project Overview

本课程项目要求每个小组基于 **nuScenes** 数据集，实现并分析一个完整的自动驾驶感知流水线，包含三个顺序阶段：

1. **3D Object Detection**
   从 LiDAR 点云中检测目标，可使用传统算法或深度学习方法。
2. **Multi-Object Tracking**
   将跨帧检测结果进行关联，维持稳定的目标 ID。
3. **Trajectory Prediction**
   基于目标历史轨迹预测未来位置。

建议每组 **3 人**。

## 2. Learning Objectives

- 理解 nuScenes 中的坐标系定义与感知任务输出格式
- 实现经典点云处理方法，如 voxelization、RANSAC、DBSCAN
- 实现基于 Kalman Filter 和 Hungarian algorithm 的多目标跟踪
- 实现轨迹预测方法，并分析上游误差对下游模块的影响
- 使用提供的 grading interface 对完整流水线进行评估

## 3. Dataset

本项目使用 **nuScenes v1.0-mini**，共 10 个场景。

- 完整 nuScenes 包含 1000 个 driving scenes
- 每个 scene 长度约 20 秒
- 3D bounding box 标注频率为 **2 Hz**
- 当前项目默认使用 mini split
- 如果算力允许，可以额外尝试 full dataset

## 4. Evaluation Protocol

对于每个 scene，帧序列被划分为两个部分：

- **Observation portion**: 前 75% 帧，用于 detection 和 tracking 评估
- **Prediction portion**: 后 25% 帧，用于 trajectory prediction 评估

官方提供的评估入口为：

- `grading/evaluate.py`

不要自行定义最终评分方式，最终结果应以提供的 grading script 为准。

## 5. Grading Pipeline

评估按以下顺序进行：

1. **Detection**
   评估 `detection_results.json`，使用官方 nuScenes detection metrics
2. **Tracking**
   评估 `tracking_results.json`，使用 MOT 指标
3. **Prediction**
   评估 `predict_trajectory()` 输出的未来轨迹

Prediction 阶段会：

- 找到同时出现在 observation 和 prediction 部分的 ground truth 对象
- 将它们与学生生成的 tracking results 进行匹配
- 从学生自己的 track history 中构造预测输入
- 调用 `predict_trajectory()` 并与 ground truth 未来位置比较

这意味着预测模块使用的是 **你自己的 tracking 输出**，而不是干净的 ground truth history。

## 6. Required Prediction Interface

必须实现如下函数，并确保 grading script 可以调用：

```python
def predict_trajectory(track_history, num_future_steps):
    """
    Args:
        track_history: list of dicts, sorted by timestamp, each with:
            - 'x': float
            - 'y': float
            - 'vx': float
            - 'vy': float
            - 'timestamp': int  # microseconds
        num_future_steps: int

    Returns:
        list[tuple[float, float]]
    """
```

## 7. Project Tasks

项目分为四个阶段。

### Phase 1: 3D Object Detection

需要在 observation frames 上运行检测，并输出官方格式的结果。

要求实现或完成：

1. `voxelize(points, voxel_size, pc_range)`
2. `ransac_ground_removal(points, ...)`
3. `dbscan_cluster(points, eps, min_samples)`
4. 基于几何启发式的分类与 3D bounding box 拟合
5. 导出 `submissions/detection_results.json`

说明：

- 仅处理 **observation portion** 的 sample tokens
- 若检测器本质上是 class-agnostic，需要在报告中清楚说明 label、score、orientation 的占位策略

### Phase 2: Multi-Object Tracking

需要在 observation frames 上基于检测结果实现跟踪。

要求：

- 至少实现一个 **从零开始** 的跟踪器
- 推荐基于 **Kalman Filter + Hungarian algorithm**
- 也可尝试更高级方法，如 SORT / DeepSORT
- 导出 `submissions/tracking_results.json`
- 同一 scene 内，目标必须保持一致的 `tracking_id`

### Phase 3: Trajectory Prediction

需要基于 tracking 输出进行未来轨迹预测。

最低要求：

1. 实现 **Constant Velocity (CV)** baseline
2. 至少实现一种比 CV 更好的改进方法

可选改进方向：

- Kalman Filter based prediction
- Polynomial / spline fitting
- Acceleration-aware models
- Category-aware prediction

已知采样间隔：

- `Δt = 0.5 s`

### Phase 4: Report and Final Submission

需要准备报告、代码仓库、评估结果和可调用预测函数。

## 8. Analysis Questions

报告中必须回答以下 5 个分析问题：

1. **Coordinate Frames**
   解释 sensor、ego vehicle、global coordinate systems 的作用，以及如何将 LiDAR frame 下的检测结果转换到 global frame 再写入 JSON。
2. **Point Cloud Density**
   统计 mini split 中单帧 LiDAR 点数的平均值和标准差，并分析点云密度对 voxel-based 与 point-based 检测方法的影响。
3. **Architecture**
   解释 voxel-based 与 pillar-based 表示方式的区别，以及它们在空间分辨率和计算代价上的 trade-off。
4. **Tracking**
   解释 Kalman Filter 中 `predict()` 和 `update()` 的作用，以及 Hungarian algorithm 如何处理 detection 和 track 数量不一致的情况。
5. **Prediction**
   比较 CV baseline 与改进方法在不同 prediction horizon 下的误差增长模式，并分析原因。

## 9. Report Requirements

最终报告最长 **6 页**，格式可为 **PDF** 或 **DOCX**。

报告应包含：

- Introduction and problem setup
- Coordinate conventions 与 global frame 表达方式
- Detection 方法设计与实现
- Tracking 设计、数据关联与 track management
- Prediction baseline 与改进方法比较
- 三个阶段的评估结果
- 全部 5 个 analysis questions 的回答
- 小组成员分工

## 10. Environment Setup

说明书中给出的环境搭建示例如下：

```bash
# Create conda environment
conda create -n nuscenes-project python=3.10 -y
conda activate nuscenes-project

# Install dependencies
pip install -r requirements.txt

# Clone nuScenes devkit (if not bundled)
git clone https://github.com/nutonomy/nuscenes-devkit.git

# Verify setup
./run_smoke.sh
```

关键依赖包括：

- `nuscenes-devkit`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pandas`

## 11. Evaluation Metrics

### Detection Metrics

- `mAP`
- `NDS`
- `mATE`
- `mASE`
- `mAOE`
- `mAVE`
- `mAAE`

### Tracking Metrics

- `MOTA`
- `MOTP`
- `ID Switches`
- `Fragmentation`

### Prediction Metrics

- `L1 Error (Mean)`
- `L2 Error (Mean)`
- `L1 Error (Median)`
- `L2 Error (Median)`

## 12. Grading Rubric

总分 **100 分**：

- Environment Setup: 20
- 3D Object Detection: 20
- Multi-Object Tracking: 20
- Trajectory Prediction: 15
- Report Quality: 15
- Code Reproducibility & Team Collaboration: 10

补充说明：

- 缺失或格式错误的输出会被扣分
- 未正确实现 `predict_trajectory()` 会被扣分
- 报告与代码实现不一致会被扣分

如果算力允许，也可以采用深度学习方法；评分会综合考虑模型设计、性能和报告质量。

## 13. Required Submission Files

最终需要提交以下内容：

1. **Report**
   PDF 或 DOCX，最长 6 页
2. **Code repository**
   需要可运行，并包含清晰的 `README.md`
3. **Detection output**
   `submissions/detection_results.json`
4. **Tracking output**
   `submissions/tracking_results.json`
5. **Prediction function**
   可调用的 `predict_trajectory(track_history, num_future_steps)`
6. **Evaluation evidence**
   运行 `grading/evaluate.py` 后的输出结果

## 14. Output Formats

### Detection Results

文件路径：

- `submissions/detection_results.json`

格式示意：

```json
{
  "meta": {
    "use_camera": false,
    "use_lidar": true
  },
  "results": {
    "<sample_token>": [
      {
        "sample_token": "<sample_token>",
        "translation": [x, y, z],
        "size": [w, l, h],
        "rotation": [qw, qx, qy, qz],
        "detection_name": "car",
        "detection_score": 0.85
      }
    ]
  }
}
```

### Tracking Results

文件路径：

- `submissions/tracking_results.json`

格式示意：

```json
{
  "meta": {
    "use_camera": false,
    "use_lidar": true
  },
  "results": {
    "<sample_token>": [
      {
        "sample_token": "<sample_token>",
        "translation": [x, y, z],
        "size": [w, l, h],
        "rotation": [qw, qx, qy, qz],
        "velocity": [vx, vy],
        "tracking_id": "track_001",
        "tracking_name": "car",
        "tracking_score": 0.85
      }
    ]
  }
}
```

## 15. Useful References

1. nuScenes paper: https://www.nuscenes.org/nuscenes
2. Detection challenge: https://www.nuscenes.org/object-detection
3. Tracking challenge: https://www.nuscenes.org/tracking
4. Prediction challenge: https://www.nuscenes.org/prediction
5. nuScenes DevKit: https://github.com/nutonomy/nuscenes-devkit
