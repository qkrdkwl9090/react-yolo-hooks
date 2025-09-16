/**
 * @vitest-environment jsdom
 */
import { renderHook, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useYoloDetection } from '../useYoloDetection'

// Mock the inference utilities
vi.mock('../../utils/inference', () => ({
  createInferenceSession: vi.fn().mockResolvedValue({
    session: {},
    inputNames: ['input'],
    outputNames: ['output']
  }),
  runInference: vi.fn().mockResolvedValue({
    output: {
      data: new Float32Array(84 * 8400) // Mock detection output
    }
  }),
  getOptimalProvider: vi.fn().mockResolvedValue('wasm'),
  disposeSession: vi.fn()
}))

// Mock the postprocessing
vi.mock('../../utils/postprocess', () => ({
  processDetections: vi.fn().mockReturnValue([
    {
      bbox: [100, 100, 200, 200],
      score: 0.85,
      class: 'person',
      classIndex: 0
    }
  ])
}))

// Mock image utilities
vi.mock('../../utils/image', () => ({
  toImageData: vi.fn().mockReturnValue({
    data: new Uint8ClampedArray(640 * 640 * 4),
    width: 640,
    height: 640
  }),
  preprocessImage: vi.fn().mockReturnValue({
    data: new Float32Array(3 * 640 * 640),
    originalWidth: 640,
    originalHeight: 640,
    processedWidth: 640,
    processedHeight: 640,
    scaleX: 1,
    scaleY: 1,
    padX: 0,
    padY: 0
  })
}))

describe('useYoloDetection', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should initialize with correct default state', () => {
    const { result } = renderHook(() => useYoloDetection())

    expect(result.current.isLoading).toBe(true)
    expect(result.current.isModelReady).toBe(false)
    expect(result.current.error).toBe(null)
    expect(result.current.downloadProgress).toBe(0)
    expect(typeof result.current.detect).toBe('function')
    expect(typeof result.current.reset).toBe('function')
  })

  it('should load model successfully', async () => {
    const { result } = renderHook(() => useYoloDetection())

    await waitFor(() => {
      expect(result.current.isModelReady).toBe(true)
    }, { timeout: 3000 })

    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBe(null)
    expect(result.current.downloadProgress).toBe(100)
  })

  it('should detect objects from canvas input', async () => {
    const { result } = renderHook(() => useYoloDetection())

    // Wait for model to be ready
    await waitFor(() => {
      expect(result.current.isModelReady).toBe(true)
    })

    // Create mock canvas
    const canvas = document.createElement('canvas')
    canvas.width = 640
    canvas.height = 640

    // Run detection
    const detections = await result.current.detect(canvas)

    expect(Array.isArray(detections)).toBe(true)
    expect(detections).toHaveLength(1)
    expect(detections[0]).toEqual({
      bbox: [100, 100, 200, 200],
      score: 0.85,
      class: 'person',
      classIndex: 0
    })
  })

  it('should reset properly', async () => {
    const { result } = renderHook(() => useYoloDetection())

    await waitFor(() => {
      expect(result.current.isModelReady).toBe(true)
    })

    // Reset the hook
    result.current.reset()

    await waitFor(() => {
      expect(result.current.isModelReady).toBe(false)
    })

    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBe(null)
    expect(result.current.downloadProgress).toBe(0)
  })

  it('should handle custom configuration', () => {
    const config = {
      confidenceThreshold: 0.7,
      iouThreshold: 0.3,
      maxDetections: 50
    }

    const { result } = renderHook(() => useYoloDetection(config))

    expect(result.current.isLoading).toBe(true)
  })
})