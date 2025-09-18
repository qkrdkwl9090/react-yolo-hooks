export interface YoloModel {
  name: string
  url: string
  inputShape: [number, number, number, number] // [batch, channels, height, width]
  outputShape: number[]
  classes: readonly string[]
  type: 'detection' | 'segmentation' | 'pose'
  version?: string
  description?: string
}

export interface Detection {
  bbox: [number, number, number, number] // [x, y, width, height]
  score: number
  class: string
  classIndex: number
}

export interface Segmentation extends Detection {
  mask: ImageData
}

export interface Pose extends Detection {
  keypoints: Keypoint[]
}

export interface Keypoint {
  x: number
  y: number
  confidence: number
  name: string
}

export interface YoloConfig {
  modelType?: 'detection' | 'segmentation' | 'pose'
  modelUrl?: string
  customModel?: YoloModel
  confidenceThreshold?: number
  iouThreshold?: number
  maxDetections?: number
  provider?: YoloProvider
  numThreads?: number
  enableDebug?: boolean
  autoSelectProvider?: boolean
  maxFPS?: number
  skipFrames?: number
}

export interface YoloState {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  modelInfo: YoloModel | null
  downloadProgress: number
}

export interface ProcessedImageData {
  data: Float32Array
  originalWidth: number
  originalHeight: number
  processedWidth: number
  processedHeight: number
  scaleX: number
  scaleY: number
  padX: number
  padY: number
}

export interface InferenceResult {
  detections?: Detection[]
  segmentations?: Segmentation[]
  poses?: Pose[]
  inferenceTime: number
  preprocessTime: number
  postprocessTime: number
  skipped?: boolean
}

export interface DrawingOptions {
  showLabels?: boolean
  showConfidence?: boolean
  lineWidth?: number
  fontSize?: string
  fontFamily?: string
  colors?: readonly string[]
  opacity?: number
  labelBackgroundOpacity?: number
}

export interface AutoDrawConfig extends DrawingOptions {
  enabled: boolean
  clearPrevious?: boolean
  onDrawComplete?: (canvas: HTMLCanvasElement) => void
}

export interface UseYoloReturn extends YoloState {
  predict: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<InferenceResult>
  reset: () => void
}

export type YoloProvider = 'webgpu' | 'wasm'

export interface ProviderConfig {
  provider: YoloProvider
  options?: Record<string, unknown>
}

declare global {
  const __DEV__: boolean

  interface Navigator {
    gpu?: {
      requestAdapter(): Promise<GPUAdapter | null>
    }
  }

  interface GPUAdapter {
    requestDevice(): Promise<GPUDevice>
  }

  interface GPUDevice {
    // Basic GPU device interface
  }
}