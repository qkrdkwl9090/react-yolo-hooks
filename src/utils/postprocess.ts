/**
 * Post-processing utilities for YOLO model outputs
 */

import type {
  Detection,
  Segmentation,
  Pose,
  ProcessedImageData
} from '@/types'
import {
  COCO_CLASSES,
  POSE_KEYPOINTS,
  NUM_DETECTIONS,
  NUM_CLASSES
} from '@/constants/models'

export interface PostProcessOptions {
  confidenceThreshold: number
  iouThreshold: number
  maxDetections: number
  inputShape: number[]
  modelType: 'detection' | 'segmentation' | 'pose'
}

/**
 * Convert model coordinates to original image coordinates
 */
function convertToImageCoordinates(
  xCenter: number,
  yCenter: number,
  width: number,
  height: number,
  processedImageData: ProcessedImageData
): [number, number, number, number] {
  const { originalWidth, originalHeight } = processedImageData

  const xRatio = originalWidth / 640  // MODEL_WIDTH
  const yRatio = originalHeight / 640  // MODEL_HEIGHT

  // Convert center coordinates to corner coordinates
  const x1 = (xCenter - width / 2) * xRatio
  const y1 = (yCenter - height / 2) * yRatio
  const x2 = (xCenter + width / 2) * xRatio
  const y2 = (yCenter + height / 2) * yRatio

  // Clamp to image bounds
  const clampedX1 = Math.max(0, x1)
  const clampedY1 = Math.max(0, y1)
  const clampedX2 = Math.min(originalWidth, x2)
  const clampedY2 = Math.min(originalHeight, y2)

  const boxWidth = clampedX2 - clampedX1
  const boxHeight = clampedY2 - clampedY1

  return [clampedX1, clampedY1, boxWidth, boxHeight]
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

  // YOLO11 output format: [84, 8400] -> transposed to [8400, 84]

  const detections: Detection[] = []
  const boxes: number[][] = []
  const scores: number[] = []
  const classIds: number[] = []

  // Parse outputs using channel-separated layout (like working project)
  for (let i = 0; i < NUM_DETECTIONS; i++) {
    // Extract coordinates (normalized 0-640)
    const xCenter = output[0 * NUM_DETECTIONS + i] ?? 0
    const yCenter = output[1 * NUM_DETECTIONS + i] ?? 0
    const width = output[2 * NUM_DETECTIONS + i] ?? 0
    const height = output[3 * NUM_DETECTIONS + i] ?? 0

    // Find best class (classes start at index 4)
    let maxScore = 0
    let bestClassId = 0

    for (let j = 0; j < NUM_CLASSES; j++) {
      const classScore = output[(4 + j) * NUM_DETECTIONS + i] ?? 0
      if (classScore > maxScore) {
        maxScore = classScore
        bestClassId = j
      }
    }

    // Filter by confidence threshold
    if (maxScore < confidenceThreshold) {
      continue
    }

    // Skip invalid boxes
    if (width <= 0 || height <= 0) continue


    // Convert to original image coordinates
    const [scaledX1, scaledY1, boxWidth, boxHeight] = convertToImageCoordinates(
      xCenter, yCenter, width, height, processedImageData
    )

    // Skip boxes that are too small
    if (boxWidth < 20 || boxHeight < 20) continue

    boxes.push([scaledX1, scaledY1, boxWidth, boxHeight])
    scores.push(maxScore)
    classIds.push(bestClassId)
  }

  // Apply NMS
  const selectedIndices = nms(boxes, scores, iouThreshold)

  // Create final detections (coordinates are already scaled to original image)
  for (const index of selectedIndices.slice(0, maxDetections)) {
    const box = boxes[index]
    const score = scores[index]
    const classId = classIds[index]

    if (!box || score === undefined || classId === undefined) continue

    const className = COCO_CLASSES[classId] ?? 'unknown'

    detections.push({
      bbox: [box[0], box[1], box[2], box[3]] as [number, number, number, number],
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
  _maskOutput: Float32Array,
  processedImageData: ProcessedImageData,
  options: PostProcessOptions
): Segmentation[] {
  const { confidenceThreshold, iouThreshold, maxDetections } = options

  // YOLO11-seg output format: [116, 8400] where first 84 are detection data and remaining 32 are mask coefficients
  const NUM_DETECTIONS = 8400
  const NUM_CLASSES = 80
  const numMaskCoeffs = 32

  const detections: Array<{
    bbox: [number, number, number, number]
    score: number
    class: string
    classIndex: number
    maskCoeffs: number[]
  }> = []

  const boxes: number[][] = []
  const scores: number[] = []
  const classIds: number[] = []

  // Parse detection outputs (same as detection but with mask coefficients)
  for (let i = 0; i < NUM_DETECTIONS; i++) {
    // Extract coordinates (normalized 0-640)
    const xCenter = output[0 * NUM_DETECTIONS + i] ?? 0
    const yCenter = output[1 * NUM_DETECTIONS + i] ?? 0
    const width = output[2 * NUM_DETECTIONS + i] ?? 0
    const height = output[3 * NUM_DETECTIONS + i] ?? 0

    // Find best class (classes start at index 4)
    let maxScore = 0
    let bestClassId = 0

    for (let j = 0; j < NUM_CLASSES; j++) {
      const classScore = output[(4 + j) * NUM_DETECTIONS + i] ?? 0
      if (classScore > maxScore) {
        maxScore = classScore
        bestClassId = j
      }
    }

    // Filter by confidence threshold
    if (maxScore < confidenceThreshold) continue
    if (width <= 0 || height <= 0) continue

    // Extract mask coefficients (starting from index 84)
    const maskCoeffs: number[] = []
    for (let j = 0; j < numMaskCoeffs; j++) {
      maskCoeffs.push(output[(84 + j) * NUM_DETECTIONS + i] ?? 0)
    }

    // Convert to original image coordinates using the same function
    const [scaledX1, scaledY1, boxWidth, boxHeight] = convertToImageCoordinates(
      xCenter, yCenter, width, height, processedImageData
    )
    if (boxWidth < 20 || boxHeight < 20) continue

    const className = COCO_CLASSES[bestClassId] ?? 'unknown'

    boxes.push([scaledX1, scaledY1, boxWidth, boxHeight])
    scores.push(maxScore)
    classIds.push(bestClassId)
    detections.push({
      bbox: [scaledX1, scaledY1, boxWidth, boxHeight] as [number, number, number, number],
      score: maxScore,
      class: className,
      classIndex: bestClassId,
      maskCoeffs
    })
  }

  // Apply NMS
  const selectedIndices = nms(boxes, scores, iouThreshold)

  const segmentations: Segmentation[] = []

  // Process masks for selected detections
  for (const index of selectedIndices.slice(0, maxDetections)) {
    const detection = detections[index]
    if (!detection) continue

    // Create a simple mask for now (this would need proper mask prototype processing in real implementation)
    const { originalWidth, originalHeight } = processedImageData
    const maskWidth = originalWidth
    const maskHeight = originalHeight
    const maskData = new Uint8ClampedArray(maskWidth * maskHeight * 4)

    // Fill with a simple pattern based on bounding box (placeholder)
    const [x, y, width, height] = detection.bbox
    for (let row = 0; row < maskHeight; row++) {
      for (let col = 0; col < maskWidth; col++) {
        const idx = (row * maskWidth + col) * 4

        // Simple rectangular mask within bbox
        if (col >= x && col < x + width && row >= y && row < y + height) {
          maskData[idx] = 255     // R
          maskData[idx + 1] = 0   // G
          maskData[idx + 2] = 0   // B
          maskData[idx + 3] = 128 // A (semi-transparent)
        } else {
          maskData[idx] = 0       // R
          maskData[idx + 1] = 0   // G
          maskData[idx + 2] = 0   // B
          maskData[idx + 3] = 0   // A (transparent)
        }
      }
    }

    const mask = new ImageData(maskData, maskWidth, maskHeight)

    segmentations.push({
      bbox: detection.bbox,
      score: detection.score,
      class: detection.class,
      classIndex: detection.classIndex,
      mask
    })
  }


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

  // YOLO11-pose format: [56, 8400] -> channel-separated layout
  const NUM_DETECTIONS = 8400
  const candidatePoses: Array<{
    bbox: [number, number, number, number]
    score: number
    keypoints: Array<{x: number, y: number, confidence: number, name: string}>
  }> = []

  const boxes: number[][] = []
  const scores: number[] = []

  // Parse outputs using channel-separated layout (like working project)
  for (let i = 0; i < NUM_DETECTIONS; i++) {
    // Extract coordinates (normalized 0-640)
    const xCenter = output[0 * NUM_DETECTIONS + i] ?? 0
    const yCenter = output[1 * NUM_DETECTIONS + i] ?? 0
    const width = output[2 * NUM_DETECTIONS + i] ?? 0
    const height = output[3 * NUM_DETECTIONS + i] ?? 0
    const personConf = output[4 * NUM_DETECTIONS + i] ?? 0

    // Filter by confidence threshold
    if (personConf < confidenceThreshold) continue
    if (width <= 0 || height <= 0) continue

    // Extract 17 keypoints (starting from index 5)
    const keypoints = []
    for (let j = 0; j < 17; j++) {
      const keypointX = output[(5 + j * 3) * NUM_DETECTIONS + i] ?? 0
      const keypointY = output[(5 + j * 3 + 1) * NUM_DETECTIONS + i] ?? 0
      const keypointConf = output[(5 + j * 3 + 2) * NUM_DETECTIONS + i] ?? 0

      // Convert keypoints from model coordinates to original image coordinates
      const { originalWidth, originalHeight } = processedImageData
      const xRatio = originalWidth / 640  // MODEL_WIDTH
      const yRatio = originalHeight / 640  // MODEL_HEIGHT

      // Simple ratio scaling
      const originalX = Math.max(0, Math.min(originalWidth, keypointX * xRatio))
      const originalY = Math.max(0, Math.min(originalHeight, keypointY * yRatio))

      keypoints.push({
        x: originalX,
        y: originalY,
        confidence: keypointConf,
        name: POSE_KEYPOINTS[j] ?? `keypoint_${j}`
      })
    }

    // Convert bounding box to original image coordinates using the same function
    const [scaledX1, scaledY1, boxWidth, boxHeight] = convertToImageCoordinates(
      xCenter, yCenter, width, height, processedImageData
    )
    if (boxWidth < 20 || boxHeight < 20) continue

    // Store for NMS
    boxes.push([scaledX1, scaledY1, boxWidth, boxHeight])
    scores.push(personConf)
    candidatePoses.push({
      bbox: [scaledX1, scaledY1, boxWidth, boxHeight] as [number, number, number, number],
      score: personConf,
      keypoints
    })
  }

  // Apply NMS to remove overlapping poses
  const selectedIndices = nms(boxes, scores, iouThreshold)

  // Create final poses after NMS
  const poses: Pose[] = []
  for (const index of selectedIndices.slice(0, maxDetections)) {
    const candidate = candidatePoses[index]
    if (!candidate) continue

    poses.push({
      bbox: candidate.bbox,
      score: candidate.score,
      class: 'person',
      classIndex: 0,
      keypoints: candidate.keypoints
    })
  }


  return poses
}