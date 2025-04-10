import React from 'react';
import './App.css';
import { ConfigProvider } from 'antd';
import ImageUpload from './components/ImageUpload';

const App = () => {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: "#0078D6", 
          fontFamily: "Lato",
        },
      }}
    >
      <div className="App">
        <header className="App-header">
          <h1>Rice seed Classifier</h1>
        </header>
        <main>
          <ImageUpload />
        </main>
      </div>
    </ConfigProvider>
  );
};

export default App;

