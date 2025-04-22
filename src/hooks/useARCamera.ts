import { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

interface ARCameraState {
  isARActive: boolean;
  isModelLoading: boolean;
  modelLoadError: string | null;
  detectedPoses: poseDetection.Pose[];
  isPersonDetected: boolean;
}

export const useARCamera = (videoRef: React.RefObject<HTMLVideoElement>) => {
  const [arState, setARState] = useState<ARCameraState>({
    isARActive: false,
    isModelLoading: false,
    modelLoadError: null,
    detectedPoses: [],
    isPersonDetected: false
  });
  
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const requestAnimationFrameRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initialize TensorFlow and load the pose detection model
  const initializeARModel = useCallback(async () => {
    if (!videoRef.current) return;
    
    setARState(prev => ({ ...prev, isModelLoading: true, modelLoadError: null }));
    
    try {
      // Load TensorFlow.js
      await tf.ready();
      
      // Create the pose detector - using MoveNet for real-time performance
      const model = poseDetection.SupportedModels.MoveNet;
      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true
      };
      
      const detector = await poseDetection.createDetector(model, detectorConfig);
      detectorRef.current = detector;
      
      // Create canvas for rendering AR effects
      if (!canvasRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth || 640;
        canvas.height = videoRef.current.videoHeight || 480;
        canvasRef.current = canvas;
      }
      
      setARState(prev => ({ ...prev, isModelLoading: false, isARActive: true }));
      
      // Start detection loop
      startPoseDetection();
    } catch (error) {
      console.error("Error initializing AR model:", error);
      setARState(prev => ({ 
        ...prev, 
        isModelLoading: false, 
        modelLoadError: `Failed to load AR models: ${error instanceof Error ? error.message : String(error)}`
      }));
    }
  }, [videoRef]);

  // Function to detect poses in each frame
  const detectPoses = useCallback(async () => {
    if (!detectorRef.current || !videoRef.current || !videoRef.current.readyState || !canvasRef.current) {
      requestAnimationFrameRef.current = requestAnimationFrame(detectPoses);
      return;
    }

    // Always display the video feed
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    // Detect poses in the current video frame - always run for live tracking
    try {
      const poses = await detectorRef.current.estimatePoses(videoRef.current, {
        flipHorizontal: false
      });
      
      // Check if we have valid poses with sufficient confidence
      const hasValidPerson = poses.some(pose => {
        const confidentKeypoints = pose.keypoints?.filter(kp => kp.score && kp.score > 0.3) || [];
        return confidentKeypoints.length >= 5; // Consider a person detected if we have at least 5 confident keypoints
      });
      
      setARState(prev => ({ 
        ...prev, 
        detectedPoses: poses,
        isPersonDetected: hasValidPerson
      }));
      
      // Always draw tracking points when AR is active, but add indicator if a person is detected
      if (ctx) {
        if (hasValidPerson) {
          // Only draw tracking points when a person is detected
          drawTrackingPoints(ctx, poses);
          
          // Add status indicator
          ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
          ctx.font = '16px Arial';
          ctx.textAlign = 'left';
          ctx.fillText('Person Detected: Live Tracking', 10, 25);
        } else {
          // Show waiting message
          ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
          ctx.font = '16px Arial';
          ctx.textAlign = 'center';
          ctx.fillText('Waiting for person...', canvasRef.current.width / 2, 25);
        }
      }
      
      // Continue the detection loop
      requestAnimationFrameRef.current = requestAnimationFrame(detectPoses);
    } catch (error) {
      console.error("Error detecting poses:", error);
      requestAnimationFrameRef.current = requestAnimationFrame(detectPoses);
    }
  }, [videoRef]);

  // Draw tracking points and skeleton for detected poses
  const drawTrackingPoints = useCallback((ctx: CanvasRenderingContext2D, poses: poseDetection.Pose[]) => {
    poses.forEach(pose => {
      // Draw keypoints
      if (pose.keypoints) {
        pose.keypoints.forEach(keypoint => {
          if (keypoint.score && keypoint.score > 0.3) {
            // Draw point
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 6, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(0, 255, 255, 0.8)';
            ctx.fill();
            
            // Draw point outline
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 6, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // Label the keypoint
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(keypoint.name || '', keypoint.x, keypoint.y - 10);
            
            // Display confidence score
            const score = Math.round(keypoint.score * 100);
            ctx.font = '10px Arial';
            ctx.fillText(`${score}%`, keypoint.x, keypoint.y + 20);
          }
        });
      }
      
      // Draw skeleton (connecting lines between keypoints)
      drawSkeleton(ctx, pose);
    });
  }, []);

  // Helper function to draw skeleton connecting keypoints
  const drawSkeleton = (ctx: CanvasRenderingContext2D, pose: poseDetection.Pose) => {
    if (!pose.keypoints) return;
    
    // Define connections between joints - common pose connections
    const connections = [
      ['nose', 'left_eye'], ['nose', 'right_eye'],
      ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
      ['left_shoulder', 'right_shoulder'],
      ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
      ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
      ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
      ['left_hip', 'right_hip'],
      ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
      ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
    ];
    
    // Draw lines with gradient effect
    connections.forEach(([from, to]) => {
      const fromPoint = pose.keypoints.find(kp => kp.name === from);
      const toPoint = pose.keypoints.find(kp => kp.name === to);
      
      if (fromPoint && toPoint && 
          fromPoint.score && toPoint.score && 
          fromPoint.score > 0.3 && toPoint.score > 0.3) {
        
        // Create gradient for line
        const gradient = ctx.createLinearGradient(fromPoint.x, fromPoint.y, toPoint.x, toPoint.y);
        gradient.addColorStop(0, 'rgba(0, 255, 0, 0.8)');
        gradient.addColorStop(1, 'rgba(0, 255, 255, 0.8)');
        
        ctx.beginPath();
        ctx.moveTo(fromPoint.x, fromPoint.y);
        ctx.lineTo(toPoint.x, toPoint.y);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    });
  };

  // Start the pose detection loop
  const startPoseDetection = useCallback(() => {
    if (requestAnimationFrameRef.current) {
      cancelAnimationFrame(requestAnimationFrameRef.current);
    }
    requestAnimationFrameRef.current = requestAnimationFrame(detectPoses);
  }, [detectPoses]);

  // Toggle AR functionality
  const toggleAR = useCallback(() => {
    if (arState.isARActive) {
      // Stop AR
      if (requestAnimationFrameRef.current) {
        cancelAnimationFrame(requestAnimationFrameRef.current);
        requestAnimationFrameRef.current = null;
      }
      setARState(prev => ({ 
        ...prev, 
        isARActive: false, 
        detectedPoses: [],
        isPersonDetected: false
      }));
    } else {
      // Start AR
      initializeARModel();
    }
  }, [arState.isARActive, initializeARModel]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (requestAnimationFrameRef.current) {
        cancelAnimationFrame(requestAnimationFrameRef.current);
      }
    };
  }, []);

  return {
    ...arState,
    toggleAR,
    canvasRef,
    initializeARModel
  };
};

export default useARCamera;