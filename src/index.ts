/**
 * React YOLO Vision - Modern React hooks for YOLO11 object detection, segmentation, and pose estimation
 */

// Export types
export type {
  YoloModel,
  Detection,
  Segmentation,
  Pose,
  Keypoint,
  YoloConfig,
  YoloState,
  InferenceResult,
  YoloProvider,
  ProviderConfig,
  ProcessedImageData
} from './types'

// Export utilities
export {
  toImageData,
  preprocessImage,
  scaleCoordinates,
  createCanvas,
  drawBoundingBoxes
} from './utils/image'

export {
  createInferenceSession,
  runInference,
  isWebGPUAvailable,
  getOptimalProvider,
  disposeSession
} from './utils/inference'

export {
  processDetections,
  processSegmentations,
  processPoses
} from './utils/postprocess'

export {
  clearCanvas,
  drawDetections,
  drawPoses,
  drawSegmentations,
  createCanvasFromVideo,
  resizeCanvas
} from './utils/drawing'

// Export constants
export {
  COCO_CLASSES,
  POSE_KEYPOINTS,
  DEFAULT_MODELS,
  DEFAULT_CONFIG
} from './constants/models'

// Export hooks
export {
  useYolo,
  useYoloDetection,
  useYoloSegmentation,
  useYoloPose,
  useYoloContinuous
} from './hooks'

export type {
  UseYoloReturn,
  UseYoloDetectionConfig,
  UseYoloDetectionReturn,
  UseYoloSegmentationConfig,
  UseYoloSegmentationReturn,
  UseYoloPoseConfig,
  UseYoloPoseReturn,
  UseYoloContinuousConfig,
  UseYoloContinuousReturn
} from './hooks'