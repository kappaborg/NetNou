'use client';

import { CameraAlt, Refresh } from '@mui/icons-material';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Grid,
  Paper,
  TextField,
  Typography
} from '@mui/material';
import dynamic from 'next/dynamic';
import React, { useCallback, useState } from 'react';

// Import Webcam component dynamically
const Webcam = dynamic(() => import('react-webcam'), {
  ssr: false,
  loading: () => (
    <div className="w-full max-w-md h-[300px] bg-gray-100 rounded flex items-center justify-center">
      Loading camera...
    </div>
  ),
});

interface StudentRegistrationProps {
  token: string;
  onSuccess?: () => void;
}

interface StudentData {
  name: string;
  email: string;
  student_id: string;
  class_id: string;
}

const WEBCAM_CONSTRAINTS = {
  width: 640,
  height: 480,
  facingMode: "user"
};

export default function StudentRegistration({ token, onSuccess }: StudentRegistrationProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [studentData, setStudentData] = useState<StudentData>({
    name: '',
    email: '',
    student_id: '',
    class_id: ''
  });
  
  const webcamRef = React.useRef<any>(null);
  
  const handleCapture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setCapturedImage(imageSrc);
    } else {
      setWebcamError("Failed to capture image. Please try again.");
    }
  }, [webcamRef]);
  
  const handleRetake = () => {
    setCapturedImage(null);
    setWebcamError(null);
  };
  
  const handleWebcamError = useCallback((error: string | DOMException) => {
    console.error("Webcam error:", error);
    setWebcamError(
      "Camera access error. Please ensure your camera is connected and you've granted permission."
    );
  }, []);
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setStudentData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!capturedImage) {
      setError("Please capture a photo before submitting");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/students/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          ...studentData,
          face_image: capturedImage
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to register student');
      }
      
      setSuccess(true);
      setStudentData({
        name: '',
        email: '',
        student_id: '',
        class_id: ''
      });
      setCapturedImage(null);
      
      if (onSuccess) {
        onSuccess();
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during registration');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Paper elevation={3} className="p-6 max-w-2xl mx-auto">
      <Typography variant="h5" component="h2" gutterBottom>
        Register New Student
      </Typography>
      
      {success && (
        <Alert severity="success" className="mb-4">
          Student registered successfully!
        </Alert>
      )}
      
      {error && (
        <Alert severity="error" className="mb-4">
          {error}
        </Alert>
      )}
      
      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              fullWidth
              label="Full Name"
              name="name"
              value={studentData.name}
              onChange={handleInputChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              fullWidth
              label="Email"
              name="email"
              type="email"
              value={studentData.email}
              onChange={handleInputChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              fullWidth
              label="Student ID"
              name="student_id"
              value={studentData.student_id}
              onChange={handleInputChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              fullWidth
              label="Class ID"
              name="class_id"
              value={studentData.class_id}
              onChange={handleInputChange}
            />
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              Facial Recognition
            </Typography>
            
            {webcamError && (
              <Alert severity="error" className="mb-4">
                {webcamError}
              </Alert>
            )}
            
            <Box className="border rounded p-2 bg-gray-50">
              {!capturedImage ? (
                <>
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={WEBCAM_CONSTRAINTS}
                    onUserMediaError={handleWebcamError}
                    className="w-full max-w-md mx-auto rounded"
                  />
                  <Box className="mt-3 text-center">
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<CameraAlt />}
                      onClick={handleCapture}
                      disabled={loading}
                    >
                      Capture Photo
                    </Button>
                  </Box>
                </>
              ) : (
                <>
                  <img
                    src={capturedImage}
                    alt="Captured"
                    className="w-full max-w-md mx-auto rounded"
                  />
                  <Box className="mt-3 text-center">
                    <Button
                      variant="outlined"
                      color="primary"
                      startIcon={<Refresh />}
                      onClick={handleRetake}
                      disabled={loading}
                      className="mr-2"
                    >
                      Retake
                    </Button>
                  </Box>
                </>
              )}
            </Box>
          </Grid>
          
          <Grid item xs={12} className="mt-4">
            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              disabled={loading || !capturedImage}
            >
              {loading ? <CircularProgress size={24} /> : "Register Student"}
            </Button>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
}