import { useMemo } from 'react'
import type { YoloConfig, Segmentation, InferenceResult } from '@/types'
import { useYolo } from './useYolo'

export interface UseYoloSegmentationConfig extends Omit<YoloConfig, 'modelType'> {
  // Segmentation-specific configurations can be added here
}

export interface UseYoloSegmentationReturn {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  downloadProgress: number
  segment: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<Segmentation[]>
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

  return {
    ...yoloState,
    segment
  }
}