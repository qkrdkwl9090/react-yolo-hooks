/**
 * Image processing utilities for YOLO models
 */

export interface ImageProcessingOptions {
  targetWidth?: number
  targetHeight?: number
  normalize?: boolean
  letterbox?: boolean
}

export interface ProcessedImageData {
  data: Float32Array
  originalWidth: number
  originalHeight: number
  processedWidth: number
  processedHeight: number
  scaleX: number
  scaleY: number
  padX: number
  padY: number
}

// Reusable canvas for performance optimization
let reuseCanvas: HTMLCanvasElement | null = null
let reuseCtx: CanvasRenderingContext2D | null = null

/**
 * Convert various input types to ImageData with optimized canvas reuse
 */
export function toImageData(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData
): ImageData {
  if (input instanceof ImageData) {
    return input
  }

  // Get or create reusable canvas
  if (!reuseCanvas || !reuseCtx) {
    reuseCanvas = document.createElement('canvas')
    reuseCtx = reuseCanvas.getContext('2d')
    if (!reuseCtx) {
      throw new Error('Failed to create canvas context')
    }
    // Optimize canvas for better performance
    reuseCtx.imageSmoothingEnabled = false // Disable antialiasing for speed
  }

  let width: number, height: number

  if (input instanceof HTMLImageElement) {
    width = input.naturalWidth
    height = input.naturalHeight
  } else if (input instanceof HTMLVideoElement) {
    width = input.videoWidth
    height = input.videoHeight
  } else if (input instanceof HTMLCanvasElement) {
    width = input.width
    height = input.height
  } else {
    throw new Error('Unsupported input type')
  }

  // Resize canvas only if necessary
  if (reuseCanvas.width !== width || reuseCanvas.height !== height) {
    reuseCanvas.width = width
    reuseCanvas.height = height
  }

  // Clear and draw
  reuseCtx.clearRect(0, 0, width, height)
  reuseCtx.drawImage(input as CanvasImageSource, 0, 0)

  return reuseCtx.getImageData(0, 0, width, height)
}


/**
 * Preprocess image for YOLO inference with letterboxing
 */
export function preprocessImage(
  imageData: ImageData,
  options: ImageProcessingOptions = {}
): ProcessedImageData {
  const {
    targetWidth = 640,
    targetHeight = 640,
    normalize = true
  } = options

  const { width: originalWidth, height: originalHeight, data } = imageData

  // Calculate scaling factors
  const xRatio = originalWidth / targetWidth
  const yRatio = originalHeight / targetHeight

  // Create output array with standard YOLO size
  const outputData = new Float32Array(3 * targetWidth * targetHeight)
  const targetSize = targetWidth * targetHeight

  // Process pixels
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      // Map back to original coordinates
      const srcX = Math.floor(x * xRatio)
      const srcY = Math.floor(y * yRatio)

      // Check bounds
      if (srcX < originalWidth && srcY < originalHeight && srcX >= 0 && srcY >= 0) {
        const srcIdx = (srcY * originalWidth + srcX) * 4
        const dstIdx = y * targetWidth + x

        // Extract RGB values
        const r = data[srcIdx] ?? 0
        const g = data[srcIdx + 1] ?? 0
        const b = data[srcIdx + 2] ?? 0

        // Normalize and store in CHW format
        if (normalize) {
          outputData[dstIdx] = r / 255 // R channel
          outputData[targetSize + dstIdx] = g / 255 // G channel
          outputData[targetSize * 2 + dstIdx] = b / 255 // B channel
        } else {
          outputData[dstIdx] = r
          outputData[targetSize + dstIdx] = g
          outputData[targetSize * 2 + dstIdx] = b
        }
      }
    }
  }

  return {
    data: outputData,
    originalWidth,
    originalHeight,
    processedWidth: targetWidth,
    processedHeight: targetHeight,
    scaleX: targetWidth / originalWidth, 
    scaleY: targetHeight / originalHeight,
    padX: 0,
    padY: 0
  }
}

/**
 * Convert processed coordinates back to original image coordinates
 */
export function scaleCoordinates(
  x: number,
  y: number,
  width: number,
  height: number,
  processedData: ProcessedImageData
): [number, number, number, number] {
  const { scaleX, scaleY, padX, padY, originalWidth, originalHeight } = processedData

  // Remove padding and scale back
  const originalX = Math.max(0, (x - padX) / scaleX)
  const originalY = Math.max(0, (y - padY) / scaleY)
  const originalWidth_ = Math.min(originalWidth - originalX, width / scaleX)
  const originalHeight_ = Math.min(originalHeight - originalY, height / scaleY)

  return [originalX, originalY, originalWidth_, originalHeight_]
}

/**
 * Create a canvas element for visualization
 */
export function createCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return canvas
}

/**
 * Draw bounding boxes on canvas
 */
export function drawBoundingBoxes(
  canvas: HTMLCanvasElement,
  detections: Array<{
    bbox: [number, number, number, number]
    score: number
    class: string
  }>,
  options: {
    colors?: string[]
    lineWidth?: number
    fontSize?: number
    showLabels?: boolean
  } = {}
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const {
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'],
    lineWidth = 2,
    fontSize = 14,
    showLabels = true
  } = options

  ctx.lineWidth = lineWidth
  ctx.font = `${fontSize}px Arial`

  detections.forEach((detection, index) => {
    const [x, y, width, height] = detection.bbox
    const color = colors[index % colors.length] ?? '#FF0000'
    
    ctx.strokeStyle = color
    ctx.strokeRect(x, y, width, height)

    if (showLabels) {
      const label = `${detection.class} ${(detection.score * 100).toFixed(1)}%`
      const textMetrics = ctx.measureText(label)
      const textHeight = fontSize
      
      // Background for text
      ctx.fillStyle = color
      ctx.fillRect(x, y - textHeight - 4, textMetrics.width + 8, textHeight + 4)
      
      // Text
      ctx.fillStyle = 'white'
      ctx.fillText(label, x + 4, y - 4)
    }
  })
}