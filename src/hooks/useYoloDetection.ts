import { useMemo } from 'react'
import type { YoloConfig, Detection, InferenceResult } from '@/types'
import { useYolo } from './useYolo'

export interface UseYoloDetectionConfig extends Omit<YoloConfig, 'modelType'> {
  // Detection-specific configurations can be added here
}

export interface UseYoloDetectionReturn {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  downloadProgress: number
  predict: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<InferenceResult>
  detect: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<Detection[]>
  reset: () => void
}

/**
 * Specialized hook for YOLO object detection
 * Provides a focused interface for detection tasks
 */
export function useYoloDetection(config: UseYoloDetectionConfig = {}): UseYoloDetectionReturn {
  // Force detection model type
  const detectionConfig: YoloConfig = useMemo(() => ({
    ...config,
    modelType: 'detection' as const
  }), [config])

  const { predict, ...yoloState } = useYolo(detectionConfig)

  // Wrapper function that extracts detections from result
  const detect = useMemo(() =>
    async (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData): Promise<Detection[]> => {
      const result: InferenceResult = await predict(input)

      if (!result.detections) {
        throw new Error('No detections returned from model')
      }

      return result.detections
    }, [predict])

  return {
    ...yoloState,
    predict,
    detect
  }
}