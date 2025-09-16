import type { YoloModel } from '../types'

export const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush'
]

export const POSE_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

export const DEFAULT_MODELS: Record<string, YoloModel> = {
  detection: {
    name: 'YOLO11n',
    url: 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.onnx',
    inputShape: [1, 3, 640, 640],
    outputShape: [1, 84, 8400],
    classes: COCO_CLASSES,
    type: 'detection'
  },
  segmentation: {
    name: 'YOLO11n-seg',
    url: 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-seg.onnx',
    inputShape: [1, 3, 640, 640],
    outputShape: [1, 116, 8400],
    classes: COCO_CLASSES,
    type: 'segmentation'
  },
  pose: {
    name: 'YOLO11n-pose',
    url: 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-pose.onnx',
    inputShape: [1, 3, 640, 640],
    outputShape: [1, 56, 8400],
    classes: ['person'],
    type: 'pose'
  }
}

export const DEFAULT_CONFIG = {
  confidenceThreshold: 0.5,
  iouThreshold: 0.4,
  maxDetections: 100,
  provider: 'webgpu' as const,
  numThreads: 4,
  enableDebug: false
}