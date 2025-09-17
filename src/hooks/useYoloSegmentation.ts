import { useMemo, useCallback } from 'react'
import type { YoloConfig, Segmentation, InferenceResult, DrawingOptions } from '@/types'
import { useYolo } from './useYolo'
import { drawSegmentations, clearCanvas } from '@/utils/drawing'

export interface UseYoloSegmentationConfig extends Omit<YoloConfig, 'modelType'> {
  // Segmentation-specific configurations can be added here
}

export interface UseYoloSegmentationReturn {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  downloadProgress: number
  predict: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<InferenceResult>
  segment: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<Segmentation[]>
  drawSegmentations: (canvas: HTMLCanvasElement, segmentations: Segmentation[], sourceWidth: number, sourceHeight: number, options?: DrawingOptions) => void
  clearCanvas: (canvas: HTMLCanvasElement) => void
  reset: () => void
}

/**
 * Specialized hook for YOLO instance segmentation
 * Provides a focused interface for segmentation tasks
 */
export function useYoloSegmentation(config: UseYoloSegmentationConfig = {}): UseYoloSegmentationReturn {
  // Force segmentation model type
  const segmentationConfig: YoloConfig = useMemo(() => ({
    ...config,
    modelType: 'segmentation' as const
  }), [config])

  const { predict, ...yoloState } = useYolo(segmentationConfig)

  // Wrapper function that extracts segmentations from result
  const segment = useMemo(() =>
    async (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData): Promise<Segmentation[]> => {
      const result: InferenceResult = await predict(input)

      if (!result.segmentations) {
        throw new Error('No segmentations returned from model')
      }

      return result.segmentations
    }, [predict])

  // Drawing utilities
  const drawSegmentationsCallback = useCallback(
    (canvas: HTMLCanvasElement, segmentations: Segmentation[], sourceWidth: number, sourceHeight: number, options?: DrawingOptions) => {
      drawSegmentations(canvas, segmentations, sourceWidth, sourceHeight, options)
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
    segment,
    drawSegmentations: drawSegmentationsCallback,
    clearCanvas: clearCanvasCallback
  }
}