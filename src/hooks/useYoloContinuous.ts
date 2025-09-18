import { useCallback, useRef, useEffect } from 'react'
import type { YoloConfig, InferenceResult } from '@/types'
import { useYolo } from './useYolo'

export interface UseYoloContinuousConfig extends YoloConfig {
  targetFPS?: number
  onResult?: (result: InferenceResult) => void
  onError?: (error: Error) => void
}

export interface UseYoloContinuousReturn {
  isLoading: boolean
  isModelReady: boolean
  isProcessing: boolean
  error: Error | null
  downloadProgress: number
  startProcessing: (video: HTMLVideoElement, canvas?: HTMLCanvasElement) => void
  stopProcessing: () => void
  processFrame: (input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData) => Promise<InferenceResult>
  reset: () => void
}

/**
 * Hook for continuous YOLO processing on video streams
 * Handles automatic frame extraction and processing with proper FPS control
 */
export function useYoloContinuous(config: UseYoloContinuousConfig = {}): UseYoloContinuousReturn {
  const {
    targetFPS = 15,
    onResult,
    onError,
    ...yoloConfig
  } = config

  const yolo = useYolo({
    ...yoloConfig,
    maxFPS: targetFPS,
    enableDebug: config.enableDebug || false
  })

  const processingRef = useRef<boolean>(false)
  const animationFrameRef = useRef<number>()
  const videoRef = useRef<HTMLVideoElement>()
  const canvasRef = useRef<HTMLCanvasElement>()
  const lastProcessTimeRef = useRef<number>(0)

  // Process a single frame
  const processFrame = useCallback(async (
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData
  ): Promise<InferenceResult> => {
    try {
      const result = await yolo.predict(input)

      if (onResult && !result.skipped) {
        onResult(result)
      }

      return result
    } catch (error) {
      const errorObj = error instanceof Error ? error : new Error(String(error))
      if (onError) {
        onError(errorObj)
      }
      throw errorObj
    }
  }, [yolo.predict, onResult, onError])

  // Continuous processing loop
  const processVideoFrame = useCallback(() => {
    if (!processingRef.current || !videoRef.current || !yolo.isModelReady) {
      return
    }

    const video = videoRef.current
    const currentTime = performance.now()
    const targetInterval = 1000 / targetFPS

    // Check if enough time has passed for next frame
    if (currentTime - lastProcessTimeRef.current < targetInterval) {
      // Schedule next frame
      animationFrameRef.current = requestAnimationFrame(processVideoFrame)
      return
    }

    // Skip if video is not ready
    if (video.readyState < 2 || video.paused || video.ended) {
      animationFrameRef.current = requestAnimationFrame(processVideoFrame)
      return
    }

    lastProcessTimeRef.current = currentTime

    // Process current frame
    processFrame(video).catch((error) => {
      console.error('[YoloContinuous] Processing error:', error)
      if (onError) {
        onError(error instanceof Error ? error : new Error(String(error)))
      }
    })

    // Schedule next frame
    if (processingRef.current) {
      animationFrameRef.current = requestAnimationFrame(processVideoFrame)
    }
  }, [yolo.isModelReady, targetFPS, processFrame, onError])

  // Start continuous processing
  const startProcessing = useCallback((
    video: HTMLVideoElement,
    canvas?: HTMLCanvasElement
  ) => {
    if (!yolo.isModelReady) {
      throw new Error('Model not ready. Please wait for model to load.')
    }

    if (processingRef.current) {
      stopProcessing() // Stop existing processing
    }

    videoRef.current = video
    canvasRef.current = canvas
    processingRef.current = true
    lastProcessTimeRef.current = 0

    // Start processing loop
    animationFrameRef.current = requestAnimationFrame(processVideoFrame)
  }, [yolo.isModelReady, processVideoFrame])

  // Stop continuous processing
  const stopProcessing = useCallback(() => {
    processingRef.current = false

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = undefined
    }

    videoRef.current = undefined
    canvasRef.current = undefined
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopProcessing()
    }
  }, [stopProcessing])

  // Reset function that also stops processing
  const reset = useCallback(() => {
    stopProcessing()
    yolo.reset()
  }, [yolo.reset, stopProcessing])

  return {
    ...yolo,
    isProcessing: processingRef.current,
    startProcessing,
    stopProcessing,
    processFrame,
    reset
  }
}