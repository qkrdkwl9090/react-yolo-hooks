/**
 * Post-processing utilities for YOLO model outputs
 */

import type {
  Detection,
  Segmentation,
  Pose,
  Keypoint,
  ProcessedImageData
} from '@/types'
import { scaleCoordinates } from './image'
import { COCO_CLASSES, POSE_KEYPOINTS } from '@/constants/models'

export interface PostProcessOptions {
  confidenceThreshold: number
  iouThreshold: number
  maxDetections: number
  inputShape: number[]
  modelType: 'detection' | 'segmentation' | 'pose'
}

/**
 * Non-Maximum Suppression
 */
function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
  const indices: number[] = []
  const areas = boxes.map(box => {
    const width = box?.[2] ?? 0
    const height = box?.[3] ?? 0
    return width * height
  })

  // Sort by confidence score (descending)
  const sortedIndices = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .map(item => item.index)

  while (sortedIndices.length > 0) {
    const currentIndex = sortedIndices[0]
    if (currentIndex === undefined) break

    indices.push(currentIndex)
    sortedIndices.shift()

    const currentBox = boxes[currentIndex]
    if (!currentBox) continue

    for (let i = sortedIndices.length - 1; i >= 0; i--) {
      const compareIndex = sortedIndices[i]
      if (compareIndex === undefined) continue

      const compareBox = boxes[compareIndex]
      if (!compareBox) continue

      const iou = calculateIoU(
        currentBox,
        compareBox,
        areas[currentIndex] ?? 0,
        areas[compareIndex] ?? 0
      )
      if (iou > iouThreshold) {
        sortedIndices.splice(i, 1)
      }
    }
  }

  return indices
}

/**
 * Calculate Intersection over Union (IoU)
 */
function calculateIoU(
  box1: number[],
  box2: number[],
  area1: number,
  area2: number
): number {
  const x1 = box1?.[0] ?? 0
  const y1 = box1?.[1] ?? 0
  const w1 = box1?.[2] ?? 0
  const h1 = box1?.[3] ?? 0
  const x2 = box2?.[0] ?? 0
  const y2 = box2?.[1] ?? 0
  const w2 = box2?.[2] ?? 0
  const h2 = box2?.[3] ?? 0

  const left = Math.max(x1, x2)
  const top = Math.max(y1, y2)
  const right = Math.min(x1 + w1, x2 + w2)
  const bottom = Math.min(y1 + h1, y2 + h2)

  if (right <= left || bottom <= top) {
    return 0
  }

  const intersection = (right - left) * (bottom - top)
  const union = area1 + area2 - intersection

  return intersection / union
}

/**
 * Process detection outputs
 */
export function processDetections(
  output: Float32Array,
  processedImageData: ProcessedImageData,
  options: PostProcessOptions
): Detection[] {
  const { confidenceThreshold, iouThreshold, maxDetections } = options
  const channels = 84
  const numDetections = output.length / 84

  const detections: Detection[] = []
  const boxes: number[][] = []
  const scores: number[] = []
  const classIds: number[] = []

  // Parse outputs (format: [x_center, y_center, width, height, confidence, class_scores...])
  for (let i = 0; i < numDetections; i++) {
    const offset = i * channels

    // Extract box coordinates
    const xCenter = output[offset] ?? 0
    const yCenter = output[offset + 1] ?? 0
    const width = output[offset + 2] ?? 0
    const height = output[offset + 3] ?? 0

    // Find best class
    let maxScore = 0
    let bestClassId = 0

    for (let j = 4; j < channels; j++) {
      const classScore = output[offset + j] ?? 0
      if (classScore > maxScore) {
        maxScore = classScore
        bestClassId = j - 4
      }
    }

    // Filter by confidence
    if (maxScore < confidenceThreshold) {
      continue
    }

    // Convert to corner coordinates
    const x = xCenter - width / 2
    const y = yCenter - height / 2

    boxes.push([x, y, width, height])
    scores.push(maxScore)
    classIds.push(bestClassId)
  }

  // Apply NMS
  const selectedIndices = nms(boxes, scores, iouThreshold)

  // Create final detections
  for (const index of selectedIndices.slice(0, maxDetections)) {
    const box = boxes[index]
    const score = scores[index]
    const classId = classIds[index]

    if (!box || score === undefined || classId === undefined) continue

    // Scale coordinates back to original image
    const [scaledX, scaledY, scaledWidth, scaledHeight] = scaleCoordinates(
      box?.[0] ?? 0,
      box?.[1] ?? 0,
      box?.[2] ?? 0,
      box?.[3] ?? 0,
      processedImageData
    )

    const className = COCO_CLASSES[classId] ?? 'unknown'

    detections.push({
      bbox: [scaledX, scaledY, scaledWidth, scaledHeight],
      score,
      class: className,
      classIndex: classId
    })
  }

  return detections
}

/**
 * Process segmentation outputs
 */
export function processSegmentations(
  output: Float32Array,
  maskOutput: Float32Array,
  processedImageData: ProcessedImageData,
  options: PostProcessOptions
): Segmentation[] {
  // First get detections
  const detections = processDetections(output, processedImageData, options)

  const segmentations: Segmentation[] = []

  // Process masks for each detection
  detections.forEach((detection) => {
    // Extract mask for this detection (simplified)
    const maskData = new Uint8ClampedArray(
      processedImageData.originalWidth * processedImageData.originalHeight * 4
    )

    // This is a simplified mask extraction - in reality, you'd need to
    // properly decode the mask prototypes and coefficients
    for (let i = 0; i < maskData.length; i += 4) {
      const maskValue = maskOutput[Math.floor(i / 4)] ?? 0
      const alpha = maskValue > 0.5 ? 128 : 0

      maskData[i] = 255     // R
      maskData[i + 1] = 0   // G
      maskData[i + 2] = 0   // B
      maskData[i + 3] = alpha // A
    }

    const mask = new ImageData(
      maskData,
      processedImageData.originalWidth,
      processedImageData.originalHeight
    )

    segmentations.push({
      ...detection,
      mask
    })
  })

  return segmentations
}

/**
 * Process pose estimation outputs
 */
export function processPoses(
  output: Float32Array,
  processedImageData: ProcessedImageData,
  options: PostProcessOptions
): Pose[] {
  const { confidenceThreshold, iouThreshold, maxDetections } = options
  const channels = 56
  const numDetections = output.length / 56

  const poses: Pose[] = []
  const boxes: number[][] = []
  const scores: number[] = []

  // Parse outputs (format: [x_center, y_center, width, height, keypoints...])
  for (let i = 0; i < numDetections; i++) {
    const offset = i * channels

    // Extract box coordinates
    const xCenter = output[offset] ?? 0
    const yCenter = output[offset + 1] ?? 0
    const width = output[offset + 2] ?? 0
    const height = output[offset + 3] ?? 0
    const confidence = output[offset + 4] ?? 0

    // Filter by confidence
    if (confidence < confidenceThreshold) {
      continue
    }

    // Convert to corner coordinates
    const x = xCenter - width / 2
    const y = yCenter - height / 2

    boxes.push([x, y, width, height])
    scores.push(confidence)
  }

  // Apply NMS
  const selectedIndices = nms(boxes, scores, iouThreshold)

  // Create final poses
  for (const index of selectedIndices.slice(0, maxDetections)) {
    const box = boxes[index]
    const score = scores[index]

    if (!box || score === undefined) continue

    const offset = index * channels

    // Extract keypoints (starts from index 5, 3 values per keypoint: x, y, confidence)
    const keypoints: Keypoint[] = []

    for (let j = 0; j < 17; j++) { // 17 keypoints for human pose
      const keypointOffset = offset + 5 + j * 3
      const kx = output[keypointOffset] ?? 0
      const ky = output[keypointOffset + 1] ?? 0
      const kconf = output[keypointOffset + 2] ?? 0

      // Scale keypoint coordinates
      const [scaledKx, scaledKy] = scaleCoordinates(
        kx, ky, 0, 0,
        processedImageData
      )

      keypoints.push({
        x: scaledKx,
        y: scaledKy,
        confidence: kconf,
        name: POSE_KEYPOINTS[j] ?? `keypoint_${j}`
      })
    }

    // Scale bounding box coordinates
    const [scaledX, scaledY, scaledWidth, scaledHeight] = scaleCoordinates(
      box?.[0] ?? 0,
      box?.[1] ?? 0,
      box?.[2] ?? 0,
      box?.[3] ?? 0,
      processedImageData
    )

    poses.push({
      bbox: [scaledX, scaledY, scaledWidth, scaledHeight],
      score,
      class: 'person',
      classIndex: 0,
      keypoints
    })
  }

  return poses
}