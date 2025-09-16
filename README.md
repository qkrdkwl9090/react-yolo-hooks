# React YOLO Hooks

Modern React hooks for YOLO11 object detection, segmentation, and pose estimation using ONNX Runtime Web.

## Features

- Object detection with YOLO11n
- Instance segmentation with YOLO11n-seg
- Pose estimation with YOLO11n-pose
- WebGPU acceleration with WASM fallback
- Real-time camera processing
- Full TypeScript support
- Tree-shakable ESM/CJS builds
- Comprehensive test coverage

## Installation

```bash
npm install react-yolo-hooks
# or
yarn add react-yolo-hooks
# or
pnpm add react-yolo-hooks
```

## Quick Start

### Object Detection

```tsx
import { useYoloDetection } from 'react-yolo-hooks'

function ObjectDetector() {
  const { detect, isModelReady, error } = useYoloDetection({
    confidenceThreshold: 0.7
  })

  const handleImageUpload = async (event) => {
    const file = event.target.files[0]
    if (!file || !isModelReady) return

    const img = new Image()
    img.src = URL.createObjectURL(file)
    img.onload = async () => {
      const detections = await detect(img)
      console.log('Found objects:', detections)
    }
  }

  if (error) return <div>Error: {error.message}</div>
  if (!isModelReady) return <div>Loading model...</div>

  return (
    <input
      type="file"
      accept="image/*"
      onChange={handleImageUpload}
    />
  )
}
```

### Real-time Camera Detection

```tsx
import { useRef, useEffect } from 'react'
import { useYoloDetection, drawBoundingBoxes } from 'react-yolo-hooks'

function CameraDetection() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const { detect, isModelReady } = useYoloDetection()

  useEffect(() => {
    if (!isModelReady) return

    const processFrame = async () => {
      if (!videoRef.current || !canvasRef.current) return

      const detections = await detect(videoRef.current)

      // Draw results on canvas
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(videoRef.current, 0, 0)

      drawBoundingBoxes(canvas, detections)

      requestAnimationFrame(processFrame)
    }

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        videoRef.current.srcObject = stream
        videoRef.current.onloadeddata = processFrame
      })

    return () => {
      const stream = videoRef.current?.srcObject
      stream?.getTracks().forEach(track => track.stop())
    }
  }, [detect, isModelReady])

  return (
    <div>
      <video ref={videoRef} autoPlay muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} width={640} height={480} />
    </div>
  )
}
```

### Segmentation

```tsx
import { useYoloSegmentation } from 'react-yolo-hooks'

function ImageSegmentation() {
  const { segment, isModelReady } = useYoloSegmentation()

  const handleSegment = async (imageElement) => {
    const segmentations = await segment(imageElement)

    segmentations.forEach((seg, index) => {
      console.log(`Object ${index + 1}:`, {
        class: seg.class,
        confidence: seg.score,
        bbox: seg.bbox,
        mask: seg.mask // ImageData with alpha channel
      })
    })
  }

  return isModelReady ? (
    <div>Ready for segmentation</div>
  ) : (
    <div>Loading segmentation model...</div>
  )
}
```

### Pose Detection

```tsx
import { useYoloPose } from 'react-yolo-hooks'

function PoseDetection() {
  const { detectPoses, isModelReady } = useYoloPose()

  const analyzePose = async (imageElement) => {
    const poses = await detectPoses(imageElement)

    poses.forEach((pose, index) => {
      console.log(`Person ${index + 1}:`)
      pose.keypoints.forEach(kp => {
        console.log(`${kp.name}: (${kp.x}, ${kp.y}) confidence: ${kp.confidence}`)
      })
    })
  }

  return isModelReady ? (
    <div>Ready for pose detection</div>
  ) : (
    <div>Loading pose model...</div>
  )
}
```

## Advanced Usage

### Custom Configuration

```tsx
const { detect } = useYoloDetection({
  confidenceThreshold: 0.8,
  iouThreshold: 0.4,
  maxDetections: 50,
  provider: 'webgpu', // or 'wasm'
  enableDebug: true
})
```

### Using Multiple Models

```tsx
function MultiModelApp() {
  const detection = useYoloDetection({ confidenceThreshold: 0.7 })
  const segmentation = useYoloSegmentation({ confidenceThreshold: 0.8 })
  const pose = useYoloPose()

  const processImage = async (img) => {
    const [objects, segments, poses] = await Promise.all([
      detection.detect(img),
      segmentation.segment(img),
      pose.detectPoses(img)
    ])

    return { objects, segments, poses }
  }

  return (
    <div>
      {detection.isModelReady && segmentation.isModelReady && pose.isModelReady
        ? <div>All models ready!</div>
        : <div>Loading models...</div>
      }
    </div>
  )
}
```

### Error Handling

```tsx
function RobustDetection() {
  const { detect, error, reset } = useYoloDetection()

  const safeDetect = async (input) => {
    try {
      return await detect(input)
    } catch (err) {
      console.error('Detection failed:', err)
      // Optionally reset the model
      reset()
      return []
    }
  }

  if (error) {
    return (
      <div>
        Model error: {error.message}
        <button onClick={reset}>Retry</button>
      </div>
    )
  }

  return <div>Detection ready</div>
}
```

## API Reference

### useYoloDetection(config?)

Returns an object with:

- `detect(input)` - Detect objects in image/video/canvas
- `isLoading` - Model loading state
- `isModelReady` - Model ready state
- `error` - Error object if loading fails
- `downloadProgress` - Model download progress (0-100)
- `reset()` - Reset the model

### useYoloSegmentation(config?)

Returns an object with:

- `segment(input)` - Segment objects in image/video/canvas
- Same state properties as detection hook

### useYoloPose(config?)

Returns an object with:

- `detectPoses(input)` - Detect poses in image/video/canvas
- Same state properties as detection hook

### Configuration Options

```tsx
interface YoloConfig {
  confidenceThreshold?: number  // Default: 0.5
  iouThreshold?: number        // Default: 0.4
  maxDetections?: number       // Default: 100
  provider?: 'webgpu' | 'wasm' // Default: auto-detect
  enableDebug?: boolean        // Default: false
}
```

## Utilities

### Drawing Functions

```tsx
import { createCanvas, drawBoundingBoxes } from 'react-yolo-hooks'

const canvas = createCanvas(640, 480)
drawBoundingBoxes(canvas, detections, {
  colors: ['#FF0000', '#00FF00', '#0000FF'],
  lineWidth: 2,
  showLabels: true
})
```

### Image Processing

```tsx
import { toImageData, preprocessImage } from 'react-yolo-hooks'

// Convert various inputs to ImageData
const imageData = toImageData(imgElement)

// Preprocess for YOLO models
const processed = preprocessImage(imageData, {
  targetWidth: 640,
  targetHeight: 640,
  normalize: true
})
```

## Browser Support

- Chrome 113+ (WebGPU support)
- Firefox 100+ (WASM fallback)
- Safari 16+ (WASM fallback)
- Edge 113+ (WebGPU support)

## Performance

Typical inference times on different hardware:

- **High-end GPU**: 15-30ms
- **Mid-range GPU**: 30-60ms
- **CPU (WASM)**: 100-300ms

## Requirements

- React 16.8+
- Modern browser with ES2020 support
- ~20MB model download (cached after first load)

## Contributing

Pull requests welcome! Please ensure tests pass:

```bash
npm test
npm run lint
npm run type-check
```