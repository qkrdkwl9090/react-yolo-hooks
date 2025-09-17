import { useMemo, useCallback } from 'react'
import type { YoloConfig, Detection, InferenceResult, DrawingOptions } from '@/types'
import { useYolo } from './useYolo'
import { drawDetections, clearCanvas } from '@/utils/drawing'

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
  drawDetections: (canvas: HTMLCanvasElement, detections: Detection[], sourceWidth: number, sourceHeight: number, options?: DrawingOptions) => void
  clearCanvas: (canvas: HTMLCanvasElement) => void
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

  // Drawing utilities
  const drawDetectionsCallback = useCallback(
    (canvas: HTMLCanvasElement, detections: Detection[], sourceWidth: number, sourceHeight: number, options?: DrawingOptions) => {
      drawDetections(canvas, detections, sourceWidth, sourceHeight, options)
    }, []
  )

  const clearCanvasCallback = useCallback(
    (canvas: HTMLCanvasElement) => {
      clearCanvas(canvas)
    }, []
  )

  return {
    ...yoloState,
    predict,
    detect,
    drawDetections: drawDetectionsCallback,
    clearCanvas: clearCanvasCallback
  }
}