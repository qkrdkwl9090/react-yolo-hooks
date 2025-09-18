/**
 * Canvas drawing utilities for YOLO detection results
 */

import type { Detection, Pose, Segmentation } from '@/types'
import { DEFAULT_COLORS, POSE_SKELETON } from '@/constants/models'

export interface DrawingOptions {
  showLabels?: boolean
  showConfidence?: boolean
  lineWidth?: number
  fontSize?: string
  fontFamily?: string
  colors?: readonly string[]
  opacity?: number
  labelBackgroundOpacity?: number
}

const DEFAULT_OPTIONS: Required<DrawingOptions> = {
  showLabels: true,
  showConfidence: true,
  lineWidth: 2,
  fontSize: '14px',
  fontFamily: 'Arial, sans-serif',
  colors: DEFAULT_COLORS,
  opacity: 1.0,
  labelBackgroundOpacity: 0.8
}

/**
 * Get color for a specific class ID
 */
function getColorForClass(classId: number, colors: readonly string[]): string {
  return colors[classId % colors.length] ?? colors[0] ?? '#FF6B6B'
}

/**
 * Draw label with mirror compensation for mirrored canvas
 */
function drawMirroredLabel(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  label: string,
  scaledX: number,
  scaledY: number,
  _scaledWidth: number,
  color: string,
  opts: Required<DrawingOptions>
): void {
  ctx.save()

  const textMetrics = ctx.measureText(label)
  const textWidth = textMetrics.width
  const textHeight = parseInt(opts.fontSize)

  // Counter the canvas mirroring for text only
  ctx.scale(-1, 1)

  // Calculate position for mirrored coordinate system
  let labelX = -(scaledX + textWidth + 8)
  let labelY = scaledY - textHeight - 4

  // Adjust label position if it goes off screen
  if (labelY < 0) {
    labelY = scaledY + textHeight + 4
  }
  if (labelX < -canvas.width) {
    labelX = -canvas.width + 4
  }
  if (labelX + textWidth + 8 > 0) {
    labelX = -(textWidth + 8)
  }

  // Draw label background with opacity
  ctx.globalAlpha = opts.labelBackgroundOpacity
  ctx.fillStyle = color
  ctx.fillRect(labelX, labelY, textWidth + 8, textHeight + 4)

  // Draw label text
  ctx.globalAlpha = 1.0
  ctx.fillStyle = 'white'
  ctx.fillText(label, labelX + 4, labelY + textHeight)

  ctx.restore()
}

/**
 * Clear the canvas
 */
export function clearCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
}

/**
 * Draw detection bounding boxes on canvas
 */
export function drawDetections(
  canvas: HTMLCanvasElement,
  detections: Detection[],
  sourceWidth: number,
  sourceHeight: number,
  options: DrawingOptions = {}
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const opts = { ...DEFAULT_OPTIONS, ...options }

  // GitHub project style: coordinates are already in original image space
  // Scale them to canvas dimensions
  const scaleX = canvas.width / sourceWidth
  const scaleY = canvas.height / sourceHeight

  // Clear canvas first
  clearCanvas(canvas)

  ctx.font = `${opts.fontSize} ${opts.fontFamily}`
  ctx.lineWidth = opts.lineWidth
  ctx.globalAlpha = opts.opacity

  detections.forEach((detection) => {
    const [x, y, width, height] = detection.bbox
    const color = getColorForClass(detection.classIndex, opts.colors)

    // Scale coordinates from source image size to canvas size
    const drawX = x * scaleX
    const drawY = y * scaleY
    const drawWidth = width * scaleX
    const drawHeight = height * scaleY

    // Draw bounding box
    ctx.strokeStyle = color
    ctx.strokeRect(drawX, drawY, drawWidth, drawHeight)

    // Draw label if enabled
    if (opts.showLabels) {
      const label = opts.showConfidence
        ? `${detection.class} ${(detection.score * 100).toFixed(1)}%`
        : detection.class

      drawMirroredLabel(ctx, canvas, label, drawX, drawY, drawWidth, color, opts)
    }
  })

  // Reset global alpha
  ctx.globalAlpha = 1.0
}

/**
 * Draw pose keypoints and skeleton on canvas
 */
export function drawPoses(
  canvas: HTMLCanvasElement,
  poses: Pose[],
  sourceWidth: number,
  sourceHeight: number,
  options: DrawingOptions = {}
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const opts = { ...DEFAULT_OPTIONS, ...options }

  // Calculate scale factors for both bounding boxes and keypoints
  // (both are now in original image coordinates)
  const scaleX = canvas.width / sourceWidth
  const scaleY = canvas.height / sourceHeight

  // Clear canvas first
  clearCanvas(canvas)


  poses.forEach((pose, poseIndex) => {
    const color = getColorForClass(poseIndex, opts.colors)

    // Draw bounding box
    const [x, y, width, height] = pose.bbox
    const drawX = x * scaleX
    const drawY = y * scaleY
    const drawWidth = width * scaleX
    const drawHeight = height * scaleY

    ctx.strokeStyle = color
    ctx.lineWidth = opts.lineWidth
    ctx.strokeRect(drawX, drawY, drawWidth, drawHeight)

    // Draw skeleton connections
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    POSE_SKELETON.forEach(([startIdx, endIdx]) => {
      if (startIdx === undefined || endIdx === undefined) return

      const startPoint = pose.keypoints[startIdx - 1] // Convert to 0-based index
      const endPoint = pose.keypoints[endIdx - 1]

      if (startPoint && endPoint &&
          startPoint.confidence > 0.5 && endPoint.confidence > 0.5) {
        ctx.beginPath()
        const startX = startPoint.x * scaleX
        const startY = startPoint.y * scaleY
        const endX = endPoint.x * scaleX
        const endY = endPoint.y * scaleY
        ctx.moveTo(startX, startY)
        ctx.lineTo(endX, endY)
        ctx.stroke()
      }
    })

    // Draw keypoints
    pose.keypoints.forEach((keypoint, keypointIndex) => {
      if (keypoint.confidence > 0.5) {
        const drawX = keypoint.x * scaleX
        const drawY = keypoint.y * scaleY

        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(drawX, drawY, 4, 0, 2 * Math.PI)
        ctx.fill()

        // Draw keypoint index for debugging
        if (opts.showLabels) {
          ctx.fillStyle = 'white'
          ctx.font = '10px Arial'
          ctx.fillText(keypointIndex.toString(), drawX + 5, drawY - 5)
        }
      }
    })

    // Draw label if enabled
    if (opts.showLabels) {
      const label = opts.showConfidence
        ? `${pose.class} ${(pose.score * 100).toFixed(1)}%`
        : pose.class

      drawMirroredLabel(ctx, canvas, label, drawX, drawY, drawWidth, color, opts)
    }
  })
}

/**
 * Draw segmentation masks on canvas
 */
export function drawSegmentations(
  canvas: HTMLCanvasElement,
  segmentations: Segmentation[],
  sourceWidth: number,
  sourceHeight: number,
  options: DrawingOptions = {}
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const opts = { ...DEFAULT_OPTIONS, ...options }

  // Calculate scale factors
  const scaleX = canvas.width / sourceWidth
  const scaleY = canvas.height / sourceHeight

  // Clear canvas first
  clearCanvas(canvas)

  segmentations.forEach((segmentation) => {
    const color = getColorForClass(segmentation.classIndex, opts.colors)

    // Draw mask
    if (segmentation.mask) {
      // Create a temporary canvas for the mask
      const maskCanvas = document.createElement('canvas')
      maskCanvas.width = segmentation.mask.width
      maskCanvas.height = segmentation.mask.height
      const maskCtx = maskCanvas.getContext('2d')

      if (maskCtx) {
        maskCtx.putImageData(segmentation.mask, 0, 0)

        // Scale and draw the mask
        ctx.globalAlpha = 0.5
        ctx.drawImage(
          maskCanvas,
          0, 0, maskCanvas.width, maskCanvas.height,
          0, 0, canvas.width, canvas.height
        )
        ctx.globalAlpha = 1.0
      }
    }

    // Draw bounding box
    const [x, y, width, height] = segmentation.bbox
    const drawX = x * scaleX
    const drawY = y * scaleY
    const drawWidth = width * scaleX
    const drawHeight = height * scaleY

    ctx.strokeStyle = color
    ctx.lineWidth = opts.lineWidth
    ctx.strokeRect(drawX, drawY, drawWidth, drawHeight)

    // Draw label if enabled
    if (opts.showLabels) {
      const label = opts.showConfidence
        ? `${segmentation.class} ${(segmentation.score * 100).toFixed(1)}%`
        : segmentation.class

      drawMirroredLabel(ctx, canvas, label, drawX, drawY, drawWidth, color, opts)
    }
  })
}

/**
 * Utility function to get canvas from video element
 */
export function createCanvasFromVideo(video: HTMLVideoElement): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.drawImage(video, 0, 0)
  }

  return canvas
}

/**
 * Utility function to resize canvas to match display size
 */
export function resizeCanvas(canvas: HTMLCanvasElement, displayWidth: number, displayHeight: number): void {
  canvas.width = displayWidth
  canvas.height = displayHeight
}