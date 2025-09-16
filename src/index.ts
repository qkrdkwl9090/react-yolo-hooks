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
  UseYoloReturn,
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

// Export constants
export {
  COCO_CLASSES,
  POSE_KEYPOINTS,
  DEFAULT_MODELS,
  DEFAULT_CONFIG
} from './constants/models'

// Export hooks (will be implemented next)
// export { useYolo } from './hooks/useYolo'
// export { useYoloDetection } from './hooks/useYoloDetection'
// export { useYoloSegmentation } from './hooks/useYoloSegmentation'
// export { useYoloPose } from './hooks/useYoloPose'