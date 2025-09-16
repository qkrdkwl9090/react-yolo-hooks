/**
 * ONNX Runtime inference utilities
 */

import * as ort from 'onnxruntime-web'
import type { YoloProvider, ProviderConfig } from '@/types'

export interface InferenceSession {
  session: ort.InferenceSession
  inputNames: readonly string[]
  outputNames: readonly string[]
}

/**
 * Configure ONNX Runtime providers
 */
export function configureProviders(provider: YoloProvider): ProviderConfig[] {
  const configs: ProviderConfig[] = []

  switch (provider) {
    case 'webgpu':
      configs.push({
        provider: 'webgpu',
        options: {
          preferredLayout: 'NCHW'
        }
      })
      // Fallback to WASM
      configs.push({
        provider: 'wasm',
        options: {
          numThreads: 4
        }
      })
      break
      
    case 'wasm':
    default:
      configs.push({
        provider: 'wasm',
        options: {
          numThreads: 4
        }
      })
      break
  }

  return configs
}

/**
 * Create ONNX Runtime inference session
 */
export async function createInferenceSession(
  modelPath: string,
  provider: YoloProvider = 'webgpu',
  onProgress?: (progress: number) => void
): Promise<InferenceSession> {
  try {
    // Configure providers
    const providers = configureProviders(provider)
    
    // Set execution providers
    const providerNames = providers.map(config => {
      if (config.provider === 'webgpu') return 'webgpu'
      return 'wasm'
    })
    
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/'
    
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: providerNames,
      graphOptimizationLevel: 'all',
      executionMode: 'parallel'
    }

    // Load model with progress tracking
    const session = await ort.InferenceSession.create(modelPath, sessionOptions)
    
    if (onProgress) {
      onProgress(100)
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
 * Run inference on preprocessed data
 */
export async function runInference(
  inferenceSession: InferenceSession,
  inputData: Float32Array,
  inputShape: number[]
): Promise<ort.InferenceSession.OnnxValueMapType> {
  try {
    const { session, inputNames } = inferenceSession
    
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