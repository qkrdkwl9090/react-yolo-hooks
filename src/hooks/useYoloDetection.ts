import { useMemo, useCallback, useRef } from 'react'
import type { YoloConfig, Detection, InferenceResult, AutoDrawConfig } from '@/types'
import { useYolo } from './useYolo'
import { drawDetections, clearCanvas } from '@/utils/drawing'

export interface UseYoloDetectionConfig extends Omit<YoloConfig, 'modelType'> {
  autoDraw?: AutoDrawConfig
}

export interface UseYoloDetectionReturn {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  downloadProgress: number
  predict: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<InferenceResult>
  detect: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<Detection[]>
  detectAndDraw: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData, canvas: HTMLCanvasElement, sourceWidth: number, sourceHeight: number) => Promise<Detection[]>
  drawDetections: (canvas: HTMLCanvasElement, detections: Detection[], sourceWidth: number, sourceHeight: number, options?: Partial<AutoDrawConfig>) => void
  clearCanvas: (canvas: HTMLCanvasElement) => void
  reset: () => void
}

/**
 * Specialized hook for YOLO object detection
 * Provides a focused interface for detection tasks with autoDraw support
 */
export function useYoloDetection(config: UseYoloDetectionConfig = {}): UseYoloDetectionReturn {
  const { autoDraw, ...restConfig } = config
  const autoDrawConfigRef = useRef<AutoDrawConfig | undefined>(autoDraw)
  autoDrawConfigRef.current = autoDraw

  // Force detection model type
  const detectionConfig: YoloConfig = useMemo(() => ({
    ...restConfig,
    modelType: 'detection' as const
  }), [restConfig])

  const { predict, ...yoloState } = useYolo(detectionConfig)

  // Wrapper function that extracts detections from result
  const detect = useMemo(() =>
    async (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData): Promise<Detection[]> => {
      const result: InferenceResult = await predict(input)

      // Handle skipped frames gracefully
      if (result.skipped) {
        return []
      }

      if (!result.detections) {
        throw new Error('No detections returned from model')
      }

      return result.detections
    }, [predict])

  // Combined detect and draw function
  const detectAndDraw = useCallback(
    async (
      input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData,
      canvas: HTMLCanvasElement,
      sourceWidth: number,
      sourceHeight: number
    ): Promise<Detection[]> => {
      const detections = await detect(input)
      const drawConfig = autoDrawConfigRef.current

      if (drawConfig?.enabled) {
        if (drawConfig.clearPrevious !== false) {
          clearCanvas(canvas)
        }
        drawDetections(canvas, detections, sourceWidth, sourceHeight, drawConfig)
        drawConfig.onDrawComplete?.(canvas)
      }

      return detections
    },
    [detect]
  )

  // Drawing utilities
  const drawDetectionsCallback = useCallback(
    (canvas: HTMLCanvasElement, detections: Detection[], sourceWidth: number, sourceHeight: number, options?: Partial<AutoDrawConfig>) => {
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
    detectAndDraw,
    drawDetections: drawDetectionsCallback,
    clearCanvas: clearCanvasCallback
  }
}