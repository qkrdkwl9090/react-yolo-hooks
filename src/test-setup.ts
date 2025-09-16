import { expect, afterEach, beforeAll, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import * as matchers from '@testing-library/jest-dom/matchers'

expect.extend(matchers)

beforeAll(() => {
  // Mock ONNX Runtime Web
  (global as any).ort = {
    InferenceSession: {
      create: vi.fn().mockResolvedValue({
        run: vi.fn().mockResolvedValue({}),
        inputNames: ['input'],
        outputNames: ['output']
      })
    },
    Tensor: vi.fn()
  }

  // Mock WebGPU
  Object.defineProperty(global.navigator, 'gpu', {
    value: {
      requestAdapter: vi.fn().mockResolvedValue({
        requestDevice: vi.fn().mockResolvedValue({})
      })
    },
    writable: true
  })
})

afterEach(() => {
  cleanup()
})