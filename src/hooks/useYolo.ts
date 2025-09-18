import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  YoloConfig,
  YoloState,
  UseYoloReturn,
  InferenceResult,
  YoloModel,
  ProcessedImageData
} from '@/types'
import { DEFAULT_MODELS, DEFAULT_CONFIG } from '@/constants/models'
import {
  createInferenceSession,
  runInference,
  getOptimalProvider,
  disposeSession,
  type InferenceSession
} from '@/utils/inference'
import {
  toImageData,
  preprocessImage,
  type ImageProcessingOptions
} from '@/utils/image'
import {
  processDetections,
  processSegmentations,
  processPoses,
  type PostProcessOptions
} from '@/utils/postprocess'

/**
 * Main YOLO hook for object detection, segmentation, and pose estimation
 * Provides a unified interface for all YOLO model types
 */
export function useYolo(config: YoloConfig = {}): UseYoloReturn {
  const {
    modelType = 'detection',
    modelUrl,
    customModel,
    confidenceThreshold = DEFAULT_CONFIG.confidenceThreshold,
    iouThreshold = DEFAULT_CONFIG.iouThreshold,
    maxDetections = DEFAULT_CONFIG.maxDetections,
    provider,
    numThreads,
    enableDebug = DEFAULT_CONFIG.enableDebug,
    autoSelectProvider = true,
    maxFPS = DEFAULT_CONFIG.maxFPS,
    skipFrames = DEFAULT_CONFIG.skipFrames
  } = config

  // State management
  const [state, setState] = useState<YoloState>({
    isLoading: false,
    isModelReady: false,
    error: null,
    modelInfo: null,
    downloadProgress: 0
  })

  // Refs for cleanup and persistence
  const inferenceSessionRef = useRef<InferenceSession | null>(null)
  const mountedRef = useRef<boolean>(true)
  const predictionInProgressRef = useRef<boolean>(false)
  const lastPredictionTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)

  // Get model configuration
  const getModelConfig = useCallback((): YoloModel => {
    // Use custom model if provided
    if (customModel) {
      return customModel
    }

    const baseModel = DEFAULT_MODELS[modelType]
    if (!baseModel) {
      throw new Error(`Unsupported model type: ${modelType}`)
    }

    if (modelUrl) {
      return {
        ...baseModel,
        url: modelUrl,
        name: `Custom ${modelType} model`
      }
    }
    return baseModel
  }, [modelType, modelUrl, customModel])

  // Initialize model
  const initializeModel = useCallback(async () => {
    if (!mountedRef.current) return

    const modelConfig = getModelConfig()

    setState(prev => ({
      ...prev,
      isLoading: true,
      error: null,
      downloadProgress: 0,
      modelInfo: modelConfig
    }))

    try {
      // Determine optimal provider if not specified
      const finalProvider = provider ?? (autoSelectProvider ? await getOptimalProvider() : 'wasm')

      if (enableDebug) {
        console.log(`[YOLO] Loading ${modelType} model:`, modelConfig.name)
        console.log(`[YOLO] Using provider:`, finalProvider)
      }

      // Create inference session with progress tracking
      const session = await createInferenceSession(
        modelConfig.url,
        finalProvider,
        (progress: number) => {
          if (!mountedRef.current) return
          setState(prev => ({ ...prev, downloadProgress: progress }))
        },
        numThreads
      )

      // Store session reference
      inferenceSessionRef.current = session

      if (!mountedRef.current) return

      setState(prev => ({
        ...prev,
        isLoading: false,
        isModelReady: true,
        downloadProgress: 100
      }))

      if (enableDebug) {
        console.log(`[YOLO] Model loaded successfully`)
      }
    } catch (error) {
      if (!mountedRef.current) return

      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      console.error('[YOLO] Failed to initialize model:', errorMessage)

      setState(prev => ({
        ...prev,
        isLoading: false,
        isModelReady: false,
        error: new Error(`Failed to load ${modelType} model: ${errorMessage}`)
      }))
    }
  }, [modelType, provider, enableDebug, getModelConfig])

  // Predict function
  const predict = useCallback(async (
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData
  ): Promise<InferenceResult> => {
    const currentTime = performance.now()

    // FPS limiting
    if (maxFPS > 0) {
      const minInterval = 1000 / maxFPS
      const timeSinceLastPrediction = currentTime - lastPredictionTimeRef.current

      if (timeSinceLastPrediction < minInterval) {
        // Return a special skip result instead of throwing error
        return {
          inferenceTime: 0,
          preprocessTime: 0,
          postprocessTime: 0,
          skipped: true
        } as InferenceResult
      }
    }

    // Frame skipping
    if (skipFrames > 0) {
      frameCountRef.current++
      if (frameCountRef.current % (skipFrames + 1) !== 0) {
        // Return a special skip result instead of throwing error
        return {
          inferenceTime: 0,
          preprocessTime: 0,
          postprocessTime: 0,
          skipped: true
        } as InferenceResult
      }
    }

    // Check if prediction is already in progress - skip this frame
    if (predictionInProgressRef.current) {
      if (enableDebug) {
        console.log('[YOLO] Prediction already in progress, skipping frame')
      }
      // Return skip result for smooth continuous processing
      return {
        inferenceTime: 0,
        preprocessTime: 0,
        postprocessTime: 0,
        skipped: true
      } as InferenceResult
    }

    const startTime = currentTime
    lastPredictionTimeRef.current = currentTime

    if (!inferenceSessionRef.current) {
      throw new Error('Model not initialized. Please wait for model to load.')
    }

    if (!state.isModelReady) {
      throw new Error('Model not ready. Please wait for model to load.')
    }

    // Mark prediction as in progress
    predictionInProgressRef.current = true

    const modelConfig = state.modelInfo
    if (!modelConfig) {
      predictionInProgressRef.current = false
      throw new Error('Model configuration not available')
    }

    try {
      // Convert input to ImageData
      const imageData = toImageData(input)
      const preprocessStartTime = performance.now()

      // Preprocess image
      const imageProcessingOptions: ImageProcessingOptions = {
        targetWidth: modelConfig.inputShape[3],
        targetHeight: modelConfig.inputShape[2],
        normalize: true,
        letterbox: true
      }

      const processedImageData: ProcessedImageData = preprocessImage(
        imageData,
        imageProcessingOptions
      )

      const preprocessEndTime = performance.now()
      const preprocessTime = preprocessEndTime - preprocessStartTime

      // Run inference
      const inferenceStartTime = performance.now()
      const results = await runInference(
        inferenceSessionRef.current,
        processedImageData.data,
        modelConfig.inputShape
      )
      const inferenceEndTime = performance.now()
      const inferenceTime = inferenceEndTime - inferenceStartTime

      // Post-process results
      const postprocessStartTime = performance.now()

      const postProcessOptions: PostProcessOptions = {
        confidenceThreshold,
        iouThreshold,
        maxDetections,
        inputShape: modelConfig.inputShape,
        modelType
      }

      let finalResult: InferenceResult

      // Process based on model type
      if (modelType === 'detection') {
        const outputName = inferenceSessionRef.current.outputNames[0]
        if (!outputName) {
          throw new Error('No output tensor name found for detection model')
        }

        const outputTensor = results[outputName]
        if (!outputTensor || !outputTensor.data) {
          throw new Error('Invalid inference output for detection model')
        }

        const detections = processDetections(
          outputTensor.data as Float32Array,
          processedImageData,
          postProcessOptions
        )

        finalResult = {
          detections,
          inferenceTime,
          preprocessTime,
          postprocessTime: 0
        }
      } else if (modelType === 'segmentation') {
        const outputName = inferenceSessionRef.current.outputNames[0]
        const maskName = inferenceSessionRef.current.outputNames[1]

        if (!outputName || !maskName) {
          throw new Error('Missing output tensor names for segmentation model')
        }

        const outputTensor = results[outputName]
        const maskTensor = results[maskName]

        if (!outputTensor?.data || !maskTensor?.data) {
          throw new Error('Invalid inference output for segmentation model')
        }

        const segmentations = processSegmentations(
          outputTensor.data as Float32Array,
          maskTensor.data as Float32Array,
          processedImageData,
          postProcessOptions
        )

        finalResult = {
          segmentations,
          inferenceTime,
          preprocessTime,
          postprocessTime: 0
        }
      } else if (modelType === 'pose') {
        const outputName = inferenceSessionRef.current.outputNames[0]
        if (!outputName) {
          throw new Error('No output tensor name found for pose model')
        }

        const outputTensor = results[outputName]
        if (!outputTensor || !outputTensor.data) {
          throw new Error('Invalid inference output for pose model')
        }

        const poses = processPoses(
          outputTensor.data as Float32Array,
          processedImageData,
          postProcessOptions
        )

        finalResult = {
          poses,
          inferenceTime,
          preprocessTime,
          postprocessTime: 0
        }
      } else {
        throw new Error(`Unsupported model type: ${modelType}`)
      }

      const postprocessEndTime = performance.now()
      finalResult.postprocessTime = postprocessEndTime - postprocessStartTime

      const totalTime = performance.now() - startTime

      if (enableDebug) {
        console.log(`[YOLO] Prediction completed in ${totalTime.toFixed(2)}ms`)
        console.log(`  Preprocessing: ${preprocessTime.toFixed(2)}ms`)
        console.log(`  Inference: ${inferenceTime.toFixed(2)}ms`)
        console.log(`  Postprocessing: ${finalResult.postprocessTime.toFixed(2)}ms`)
      }

      // Reset prediction in progress flag
      predictionInProgressRef.current = false

      return finalResult
    } catch (error) {
      // Always reset prediction in progress flag on error
      predictionInProgressRef.current = false

      const errorMessage = error instanceof Error ? error.message : 'Unknown prediction error'
      console.error('[YOLO] Prediction failed:', errorMessage)
      throw new Error(`Prediction failed: ${errorMessage}`)
    }
  }, [state.isModelReady, state.modelInfo, confidenceThreshold, iouThreshold, maxDetections, modelType, enableDebug])

  // Reset function
  const reset = useCallback(() => {
    // Dispose current session
    if (inferenceSessionRef.current) {
      disposeSession(inferenceSessionRef.current)
      inferenceSessionRef.current = null
    }

    // Reset prediction in progress flag and timing
    predictionInProgressRef.current = false
    lastPredictionTimeRef.current = 0
    frameCountRef.current = 0

    // Reset state
    setState({
      isLoading: false,
      isModelReady: false,
      error: null,
      modelInfo: null,
      downloadProgress: 0
    })

    if (enableDebug) {
      console.log('[YOLO] Hook reset')
    }
  }, [enableDebug])

  // Initialize on mount and config changes
  useEffect(() => {
    initializeModel()
  }, [initializeModel])

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
      if (inferenceSessionRef.current) {
        disposeSession(inferenceSessionRef.current)
      }
    }
  }, [])

  return {
    ...state,
    predict,
    reset
  }
}