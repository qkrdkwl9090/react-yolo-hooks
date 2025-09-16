import { describe, it, expect, vi } from 'vitest'
import { scaleCoordinates } from '../image'
import type { ProcessedImageData } from '@/types'

// Mock DOM elements for testing
global.HTMLCanvasElement = vi.fn() as any

describe('Image utilities', () => {

  describe('scaleCoordinates', () => {
    it('should scale coordinates correctly', () => {
      const processedData: ProcessedImageData = {
        data: new Float32Array(),
        originalWidth: 1280,
        originalHeight: 720,
        processedWidth: 640,
        processedHeight: 640,
        scaleX: 0.5,
        scaleY: 0.5,
        padX: 0,
        padY: 80
      }

      const [scaledX, scaledY, scaledWidth, scaledHeight] = scaleCoordinates(
        100, 100, 200, 150,
        processedData
      )

      expect(scaledX).toBe(200)
      expect(scaledY).toBe(40)
      expect(scaledWidth).toBe(400)
      expect(scaledHeight).toBe(300)
    })
  })
})