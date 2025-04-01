import React, { useState } from 'react';
import axios from 'axios';

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setPrediction(response.data);
    } catch (err) {
      console.error('Error uploading image:', err);
      setError('Error classifying image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-upload-container">
      <form onSubmit={handleSubmit}>
        <div>
          <input
            type="file"
            onChange={handleFileChange}
            accept="image/*"
          />
        </div>

        {preview && (
          <div>
            <img
              src={preview}
              alt="Preview"
              className="image-preview"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={!selectedFile || loading}
        >
          {loading ? 'Processing...' : 'Classify Image'}
        </button>
      </form>

      {loading && <p className="loading">Analyzing image...</p>}

      {error && <p className="error">{error}</p>}

      {prediction && (
        <div className="result-container">
          <h3>Classification Result:</h3>
          <p><strong>Class:</strong> {prediction.class}</p>
          <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload; 