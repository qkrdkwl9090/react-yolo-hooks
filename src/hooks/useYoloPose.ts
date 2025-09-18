import { useMemo, useCallback, useRef } from 'react'
import type { YoloConfig, Pose, InferenceResult, AutoDrawConfig } from '@/types'
import { useYolo } from './useYolo'
import { drawPoses, clearCanvas } from '@/utils/drawing'

export interface UseYoloPoseConfig extends Omit<YoloConfig, 'modelType'> {
  autoDraw?: AutoDrawConfig
}

export interface UseYoloPoseReturn {
  isLoading: boolean
  isModelReady: boolean
  error: Error | null
  downloadProgress: number
  predict: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<InferenceResult>
  detectPoses: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<Pose[]>
  detectPosesAndDraw: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData, canvas: HTMLCanvasElement, sourceWidth: number, sourceHeight: number) => Promise<Pose[]>
  drawPoses: (canvas: HTMLCanvasElement, poses: Pose[], sourceWidth: number, sourceHeight: number, options?: Partial<AutoDrawConfig>) => void
  clearCanvas: (canvas: HTMLCanvasElement) => void
  reset: () => void
}

/**
 * Specialized hook for YOLO pose estimation
 * Provides a focused interface for pose detection tasks with autoDraw support
 */
export function useYoloPose(config: UseYoloPoseConfig = {}): UseYoloPoseReturn {
  const { autoDraw, ...restConfig } = config
  const autoDrawConfigRef = useRef<AutoDrawConfig | undefined>(autoDraw)
  autoDrawConfigRef.current = autoDraw

  // Force pose model type
  const poseConfig: YoloConfig = useMemo(() => ({
    ...restConfig,
    modelType: 'pose' as const
  }), [restConfig])

  const { predict, ...yoloState } = useYolo(poseConfig)

  // Wrapper function that extracts poses from result
  const detectPoses = useMemo(() =>
    async (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData): Promise<Pose[]> => {
      const result: InferenceResult = await predict(input)

      // Handle skipped frames gracefully
      if (result.skipped) {
        return []
      }

      if (!result.poses) {
        throw new Error('No poses returned from model')
      }

      return result.poses
    }, [predict])

  // Combined detect poses and draw function
  const detectPosesAndDraw = useCallback(
    async (
      input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData,
      canvas: HTMLCanvasElement,
      sourceWidth: number,
      sourceHeight: number
    ): Promise<Pose[]> => {
      const poses = await detectPoses(input)
      const drawConfig = autoDrawConfigRef.current

      if (drawConfig?.enabled) {
        if (drawConfig.clearPrevious !== false) {
          clearCanvas(canvas)
        }
        drawPoses(canvas, poses, sourceWidth, sourceHeight, drawConfig)
        drawConfig.onDrawComplete?.(canvas)
      }

      return poses
    },
    [detectPoses]
  )

  // Drawing utilities
  const drawPosesCallback = useCallback(
    (canvas: HTMLCanvasElement, poses: Pose[], sourceWidth: number, sourceHeight: number, options?: Partial<AutoDrawConfig>) => {
      drawPoses(canvas, poses, sourceWidth, sourceHeight, options)
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
    detectPoses,
    detectPosesAndDraw,
    drawPoses: drawPosesCallback,
    clearCanvas: clearCanvasCallback
  }
}