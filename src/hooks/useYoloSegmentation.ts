import { useMemo, useCallback, useRef } from 'react'
import type { YoloConfig, Segmentation, InferenceResult, AutoDrawConfig } from '@/types'
import { useYolo } from './useYolo'
import { drawSegmentations, clearCanvas } from '@/utils/drawing'

export interface UseYoloSegmentationConfig extends Omit<YoloConfig, 'modelType'> {
  autoDraw?: AutoDrawConfig
}

export interface UseYoloSegmentationReturn {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  downloadProgress: number
  predict: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<InferenceResult>
  segment: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<Segmentation[]>
  segmentAndDraw: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData, canvas: HTMLCanvasElement, sourceWidth: number, sourceHeight: number) => Promise<Segmentation[]>
  drawSegmentations: (canvas: HTMLCanvasElement, segmentations: Segmentation[], sourceWidth: number, sourceHeight: number, options?: Partial<AutoDrawConfig>) => void
  clearCanvas: (canvas: HTMLCanvasElement) => void
  reset: () => void
}

/**
 * Specialized hook for YOLO instance segmentation
 * Provides a focused interface for segmentation tasks with autoDraw support
 */
export function useYoloSegmentation(config: UseYoloSegmentationConfig = {}): UseYoloSegmentationReturn {
  const { autoDraw, ...restConfig } = config
  const autoDrawConfigRef = useRef<AutoDrawConfig | undefined>(autoDraw)
  autoDrawConfigRef.current = autoDraw

  // Force segmentation model type
  const segmentationConfig: YoloConfig = useMemo(() => ({
    ...restConfig,
    modelType: 'segmentation' as const
  }), [restConfig])

  const { predict, ...yoloState } = useYolo(segmentationConfig)

  // Wrapper function that extracts segmentations from result
  const segment = useMemo(() =>
    async (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData): Promise<Segmentation[]> => {
      const result: InferenceResult = await predict(input)

      // Handle skipped frames gracefully
      if (result.skipped) {
        return []
      }

      if (!result.segmentations) {
        throw new Error('No segmentations returned from model')
      }

      return result.segmentations
    }, [predict])

  // Combined segment and draw function
  const segmentAndDraw = useCallback(
    async (
      input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData,
      canvas: HTMLCanvasElement,
      sourceWidth: number,
      sourceHeight: number
    ): Promise<Segmentation[]> => {
      const segmentations = await segment(input)
      const drawConfig = autoDrawConfigRef.current

      if (drawConfig?.enabled) {
        if (drawConfig.clearPrevious !== false) {
          clearCanvas(canvas)
        }
        drawSegmentations(canvas, segmentations, sourceWidth, sourceHeight, drawConfig)
        drawConfig.onDrawComplete?.(canvas)
      }

      return segmentations
    },
    [segment]
  )

  // Drawing utilities
  const drawSegmentationsCallback = useCallback(
    (canvas: HTMLCanvasElement, segmentations: Segmentation[], sourceWidth: number, sourceHeight: number, options?: Partial<AutoDrawConfig>) => {
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
    segmentAndDraw,
    drawSegmentations: drawSegmentationsCallback,
    clearCanvas: clearCanvasCallback
  }
}