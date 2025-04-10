import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Button, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

const { Dragger } = Upload;

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // Configure Upload properties
  const uploadProps = {
    name: 'image',
    accept: 'image/*',
    multiple: false,
    maxCount: 1,
    beforeUpload: (file) => {
      // Set the file and generate preview URL
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      return false; // Prevent auto-upload
    },
    onRemove: () => {
      setSelectedFile(null);
      setPreview(null);
      setPrediction(null);
      return true;
    },
    showUploadList: true,
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      message.error('Please select an image file');
      return;
    }
  
    const formData = new FormData();
    formData.append('image', selectedFile);
  
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      // Assuming response.data contains the 'class' and 'confidence'
      setPrediction(response.data);
    } catch (err) {
      console.error('Error uploading image:', err);
      message.error('Error classifying image. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div
      className="image-upload-container"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        marginTop: '20px',
      }}
    >
      <Dragger
        {...uploadProps}
        style={{
          width: '100%',
          height: '40%',
          padding: '20px',
          background:
            'linear-gradient(99deg, rgba(252, 92, 125, 0.08) 6.03%, rgba(106, 130, 251, 0.08) 80.61%)',
          borderColor: '#FC5C7D',
        }}
        disabled={loading}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          Click or drag file to this area to upload
        </p>
        <p className="ant-upload-hint">
          Support for a single image file. Only image files are accepted.
        </p>
      </Dragger>
    
           {preview && !prediction && (
            <div style={{ marginTop: '20px', textAlign: 'center' }}>
              <img
                src={preview}
                alt="Preview"
                style={{
                  width: '300px',      // Fixed width
                  height: '300px',     // Fixed height
                  objectFit: 'cover',  // Options: 'cover' crops, 'contain' shows the whole image within the bounds
                  borderRadius: '5px',
                }}
              />
            </div>
          )}
          

   
 
      <Button
        type="primary"
        onClick={handleSubmit}
        disabled={!selectedFile || loading}
        style={{ marginTop: '20px' }}
      >
        {loading ? 'Processing...' : 'Classify Image'}
      </Button>

      {prediction && (
        <div
          className="result-container"
          style={{
            marginTop: '20px',
            textAlign: 'center',
            padding: '10px',
            border: '1px solid #f0f0f0',
            borderRadius: '4px',
            width: '60%',
          }}
        >
          <h3>Classification Result:</h3>
          <p>
            <strong>Class:</strong> {prediction.predicted_class_name}
          </p>
         
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
