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
    url: '/models/yolo11n.onnx',
    inputShape: [1, 3, 640, 640],
    outputShape: [1, 84, 8400],
    classes: COCO_CLASSES,
    type: 'detection'
  },
  segmentation: {
    name: 'YOLO11n-seg',
    url: '/models/yolo11n-seg.onnx',
    inputShape: [1, 3, 640, 640],
    outputShape: [1, 116, 8400],
    classes: COCO_CLASSES,
    type: 'segmentation'
  },
  pose: {
    name: 'YOLO11n-pose',
    url: '/models/yolo11n-pose.onnx',
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
  provider: 'wasm' as const,
  numThreads: 4,
  enableDebug: false
}

// YOLO model constants
export const MODEL_WIDTH = 640
export const MODEL_HEIGHT = 640
export const NUM_DETECTIONS = 8400
export const NUM_CLASSES = 80

// Drawing constants
export const DEFAULT_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
]

// Pose skeleton connections (COCO format)
export const POSE_SKELETON = [
  [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
  [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
  [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
  [2, 4], [3, 5], [4, 6], [5, 7]
]