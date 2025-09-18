/**
 * ONNX Runtime inference utilities
 */

import * as ort from 'onnxruntime-web'
import type { YoloProvider } from '@/types'

export interface InferenceSession {
  session: ort.InferenceSession
  inputNames: readonly string[]
  outputNames: readonly string[]
}

/**
 * Configure ONNX Runtime providers with proper WebGPU support and fallback
 */
export function configureProviders(provider: YoloProvider): ort.InferenceSession.ExecutionProviderConfig[] {
  switch (provider) {
    case 'webgpu':
      // Try WebGPU first, then fallback to CPU
      return ['webgpu', 'cpu']

    case 'wasm':
    default:
      // Use CPU provider which is more stable than WASM for real-time processing
      return ['cpu']
  }
}

/**
 * Create ONNX Runtime inference session with WebGPU/WASM fallback
 */
export async function createInferenceSession(
  modelPath: string,
  provider: YoloProvider = 'wasm',
  onProgress?: (progress: number) => void,
  numThreads?: number
): Promise<InferenceSession> {
  try {
    // Configure ONNX Runtime
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'
    ort.env.wasm.numThreads = numThreads ?? navigator.hardwareConcurrency ?? 4

    const executionProviders = configureProviders(provider)

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders,
      graphOptimizationLevel: 'all',
      executionMode: 'parallel',
      enableCpuMemArena: true,
      enableMemPattern: true,
      freeDimensionOverrides: {},
      logSeverityLevel: 4, 
      logVerbosityLevel: 0 
    }

    // Load model with progress tracking
    const session = await ort.InferenceSession.create(modelPath, sessionOptions)

    if (onProgress) {
      onProgress(100)
    }

    // Log which provider is actually being used
    if (__DEV__) {
      console.log('[YOLO] Session created successfully with provider:', provider)
    }

    return {
      session,
      inputNames: session.inputNames,
      outputNames: session.outputNames
    }
  } catch (error) {
    console.error('Failed to create inference session:', error)
    throw new Error(`Failed to load ONNX model: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/**
 * Simple mutex implementation for inference session
 */
class InferenceMutex {
  private queue: Array<() => void> = []
  private isLocked = false

  async acquire(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.isLocked) {
        this.isLocked = true
        resolve()
      } else {
        this.queue.push(resolve)
      }
    })
  }

  release(): void {
    const next = this.queue.shift()
    if (next) {
      next()
    } else {
      this.isLocked = false
    }
  }
}

// Global mutex map for each session
const sessionMutexMap = new WeakMap<ort.InferenceSession, InferenceMutex>()

/**
 * Run inference on preprocessed data with proper mutex protection
 */
export async function runInference(
  inferenceSession: InferenceSession,
  inputData: Float32Array,
  inputShape: number[]
): Promise<ort.InferenceSession.OnnxValueMapType> {
  const { session, inputNames } = inferenceSession

  // Get or create mutex for this session
  let mutex = sessionMutexMap.get(session)
  if (!mutex) {
    mutex = new InferenceMutex()
    sessionMutexMap.set(session, mutex)
  }

  // Acquire mutex before running inference
  await mutex.acquire()

  try {
    // Create input tensor
    const inputTensor = new ort.Tensor('float32', inputData, inputShape)

    // Create feeds object
    const feeds: Record<string, ort.Tensor> = {}
    feeds[inputNames[0] ?? 'input'] = inputTensor

    // Run inference
    const startTime = performance.now()
    const results = await session.run(feeds)
    const endTime = performance.now()

    if (__DEV__) {
      console.log(`Inference time: ${(endTime - startTime).toFixed(2)}ms`)
    }

    return results
  } catch (error) {
    console.error('Inference failed:', error)
    throw new Error(`Inference failed: ${error instanceof Error ? error.message : String(error)}`)
  } finally {
    // Always release mutex
    mutex.release()
  }
}

/**
 * Check WebGPU availability
 */
export async function isWebGPUAvailable(): Promise<boolean> {
  try {
    if (!navigator.gpu) {
      return false
    }
    
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      return false
    }
    
    await adapter.requestDevice()
    return true
  } catch {
    return false
  }
}

/**
 * Get optimal provider based on browser capabilities
 */
export async function getOptimalProvider(): Promise<YoloProvider> {
  const webgpuAvailable = await isWebGPUAvailable()
  return webgpuAvailable ? 'webgpu' : 'wasm'
}

/**
 * Dispose inference session
 */
export function disposeSession(inferenceSession: InferenceSession): void {
  try {
    inferenceSession.session.release()
  } catch (error) {
    console.error('Failed to dispose session:', error)
  }
}