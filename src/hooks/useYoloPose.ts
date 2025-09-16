import { useMemo } from 'react'
import type { YoloConfig, Pose, InferenceResult } from '@/types'
import { useYolo } from './useYolo'

export interface UseYoloPoseConfig extends Omit<YoloConfig, 'modelType'> {
  // Pose-specific configurations can be added here
}

export interface UseYoloPoseReturn {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  downloadProgress: number
  detectPoses: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<Pose[]>
  reset: () => void
}

/**
 * Specialized hook for YOLO pose estimation
 * Provides a focused interface for pose detection tasks
 */
export function useYoloPose(config: UseYoloPoseConfig = {}): UseYoloPoseReturn {
  // Force pose model type
  const poseConfig: YoloConfig = useMemo(() => ({
    ...config,
    modelType: 'pose' as const
  }), [config])

  const { predict, ...yoloState } = useYolo(poseConfig)

  // Wrapper function that extracts poses from result
  const detectPoses = useMemo(() =>
    async (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData): Promise<Pose[]> => {
      const result: InferenceResult = await predict(input)

      if (!result.poses) {
        throw new Error('No poses returned from model')
      }

      return result.poses
    }, [predict])

  return {
    ...yoloState,
    detectPoses
  }
}