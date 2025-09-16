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

/**
 * Convert various input types to ImageData
 */
export function toImageData(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData
): ImageData {
  if (input instanceof ImageData) {
    return input
  }

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    throw new Error('Failed to create canvas context')
  }

  if (input instanceof HTMLImageElement) {
    canvas.width = input.naturalWidth
    canvas.height = input.naturalHeight
    ctx.drawImage(input, 0, 0)
  } else if (input instanceof HTMLVideoElement) {
    canvas.width = input.videoWidth
    canvas.height = input.videoHeight
    ctx.drawImage(input, 0, 0)
  } else if (input instanceof HTMLCanvasElement) {
    canvas.width = input.width
    canvas.height = input.height
    ctx.drawImage(input, 0, 0)
  }

  return ctx.getImageData(0, 0, canvas.width, canvas.height)
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
    normalize = true,
    letterbox = true
  } = options

  const { width: originalWidth, height: originalHeight, data } = imageData

  // Calculate scaling factors
  let scaleX: number
  let scaleY: number
  let padX = 0
  let padY = 0

  if (letterbox) {
    // Letterbox scaling - maintain aspect ratio
    const scale = Math.min(targetWidth / originalWidth, targetHeight / originalHeight)
    scaleX = scaleY = scale
    
    const scaledWidth = originalWidth * scale
    const scaledHeight = originalHeight * scale
    
    padX = (targetWidth - scaledWidth) / 2
    padY = (targetHeight - scaledHeight) / 2
  } else {
    // Stretch to fit
    scaleX = targetWidth / originalWidth
    scaleY = targetHeight / originalHeight
  }

  // Create output array
  const outputData = new Float32Array(3 * targetWidth * targetHeight)
  
  // Fill with background color (gray: 114/255)
  outputData.fill(normalize ? 114 / 255 : 114)

  // Process pixels
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      // Calculate source coordinates
      const srcX = Math.floor((x - padX) / scaleX)
      const srcY = Math.floor((y - padY) / scaleY)

      // Skip if outside source bounds
      if (srcX < 0 || srcX >= originalWidth || srcY < 0 || srcY >= originalHeight) {
        continue
      }

      const srcIdx = (srcY * originalWidth + srcX) * 4
      const dstIdx = y * targetWidth + x

      // Extract RGB values and convert to CHW format
      const r = data[srcIdx] ?? 0
      const g = data[srcIdx + 1] ?? 0
      const b = data[srcIdx + 2] ?? 0

      if (normalize) {
        // Normalize to [0, 1]
        outputData[dstIdx] = r / 255 // R channel
        outputData[targetWidth * targetHeight + dstIdx] = g / 255 // G channel
        outputData[2 * targetWidth * targetHeight + dstIdx] = b / 255 // B channel
      } else {
        outputData[dstIdx] = r
        outputData[targetWidth * targetHeight + dstIdx] = g
        outputData[2 * targetWidth * targetHeight + dstIdx] = b
      }
    }
  }

  return {
    data: outputData,
    originalWidth,
    originalHeight,
    processedWidth: targetWidth,
    processedHeight: targetHeight,
    scaleX,
    scaleY,
    padX,
    padY
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