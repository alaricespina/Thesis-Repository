import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Play, Pause, RotateCcw, Brain, TreePine, Zap } from 'lucide-react';

const DBNTimeSeriesSimulator = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedModel, setSelectedModel] = useState('both');
  const [windowSize, setWindowSize] = useState(5);
  const [dataPoints, setDataPoints] = useState(50);
  const intervalRef = useRef(null);

  // Generate synthetic time series data
  const generateTimeSeriesData = () => {
    const data = [];
    for (let i = 0; i < dataPoints; i++) {
      const trend = i * 0.1;
      const seasonal = Math.sin(i * 0.2) * 3;
      const noise = (Math.random() - 0.5) * 2;
      const value = 10 + trend + seasonal + noise;
      data.push({
        time: i,
        value: Math.round(value * 100) / 100,
        originalValue: Math.round(value * 100) / 100
      });
    }
    return data;
  };

  const [timeSeriesData, setTimeSeriesData] = useState(generateTimeSeriesData());
  const [dbnFeatures, setDbnFeatures] = useState([]);
  const [sequentialData, setSequentialData] = useState([]);
  const [predictions, setPredictions] = useState({ rf: [], rnn: [] });
  const [performance, setPerformance] = useState({ rf: {}, rnn: {} });

  // Simulate DBN feature extraction
  const extractDBNFeatures = (data) => {
    return data.map((point, i) => {
      // Simulate different DBN layers extracting features
      const layer1 = Math.sin(point.value * 0.1) * 2; // Basic pattern
      const layer2 = Math.cos(point.value * 0.05) * 1.5; // Mid-level pattern
      const layer3 = Math.tanh(point.value * 0.02) * 1.2; // High-level abstraction
      const trend = i > 0 ? point.value - data[i-1].value : 0;
      
      return {
        time: point.time,
        originalValue: point.value,
        layer1Feature: Math.round(layer1 * 100) / 100,
        layer2Feature: Math.round(layer2 * 100) / 100,
        layer3Feature: Math.round(layer3 * 100) / 100,
        trendFeature: Math.round(trend * 100) / 100,
        combinedFeature: Math.round((layer1 + layer2 + layer3) * 100) / 100
      };
    });
  };

  // Create sequential windows for training
  const createSequentialWindows = (features) => {
    const windows = [];
    for (let i = 0; i < features.length - windowSize; i++) {
      const window = features.slice(i, i + windowSize);
      const target = features[i + windowSize];
      windows.push({
        windowStart: i,
        windowEnd: i + windowSize - 1,
        input: window,
        target: target.originalValue,
        // Flatten features for Random Forest
        flatFeatures: window.map(f => [f.layer1Feature, f.layer2Feature, f.layer3Feature, f.trendFeature]).flat()
      });
    }
    return windows;
  };

  // Simulate Random Forest prediction
  const simulateRandomForestPrediction = (window) => {
    const features = window.flatFeatures;
    // Simulate tree-based prediction with some randomness
    const treeVotes = [];
    for (let i = 0; i < 10; i++) { // 10 trees
      const randomWeight = Math.random() * 0.3 + 0.85;
      const lastValue = window.input[window.input.length - 1].originalValue;
      const trend = window.input.reduce((acc, curr, idx) => {
        if (idx === 0) return acc;
        return acc + (curr.originalValue - window.input[idx-1].originalValue);
      }, 0) / (window.input.length - 1);
      
      const prediction = lastValue + trend * randomWeight + (Math.random() - 0.5) * 0.5;
      treeVotes.push(prediction);
    }
    return treeVotes.reduce((a, b) => a + b) / treeVotes.length;
  };

  // Simulate RNN prediction
  const simulateRNNPrediction = (window) => {
    let hiddenState = 0;
    let cellState = 0;
    
    // Simulate LSTM forward pass
    for (let i = 0; i < window.input.length; i++) {
      const input = window.input[i];
      const combined = input.combinedFeature;
      
      // Simplified LSTM gates
      const forgetGate = Math.sigmoid(combined * 0.1 + hiddenState * 0.05);
      const inputGate = Math.sigmoid(combined * 0.08 + hiddenState * 0.06);
      const outputGate = Math.sigmoid(combined * 0.09 + hiddenState * 0.04);
      
      cellState = cellState * forgetGate + Math.tanh(combined * 0.1) * inputGate;
      hiddenState = Math.tanh(cellState) * outputGate;
    }
    
    // Final prediction
    const lastValue = window.input[window.input.length - 1].originalValue;
    return lastValue + hiddenState * 2 + (Math.random() - 0.5) * 0.3;
  };

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // Calculate performance metrics
  const calculatePerformance = (predictions, actuals) => {
    if (predictions.length === 0) return { mae: 0, rmse: 0, accuracy: 0 };
    
    const errors = predictions.map((pred, i) => Math.abs(pred - actuals[i]));
    const mae = errors.reduce((a, b) => a + b) / errors.length;
    const rmse = Math.sqrt(errors.map(e => e * e).reduce((a, b) => a + b) / errors.length);
    const accuracy = Math.max(0, 100 - (mae / Math.max(...actuals)) * 100);
    
    return {
      mae: Math.round(mae * 100) / 100,
      rmse: Math.round(rmse * 100) / 100,
      accuracy: Math.round(accuracy * 100) / 100
    };
  };

  // Simulation steps
  const steps = [
    'Generate Time Series Data',
    'Extract DBN Features (Layer 1)',
    'Extract DBN Features (Layer 2)',
    'Extract DBN Features (Layer 3)',
    'Create Sequential Windows',
    'Train Random Forest Model',
    'Train RNN Model',
    'Make Predictions',
    'Evaluate Performance'
  ];

  const runSimulation = () => {
    if (currentStep >= steps.length) return;

    switch (currentStep) {
      case 0:
        setTimeSeriesData(generateTimeSeriesData());
        break;
      case 1:
      case 2:
      case 3:
        const features = extractDBNFeatures(timeSeriesData);
        setDbnFeatures(features);
        break;
      case 4:
        const windows = createSequentialWindows(dbnFeatures);
        setSequentialData(windows);
        break;
      case 5:
      case 6:
        // Training simulation (just update UI)
        break;
      case 7:
        // Make predictions
        const rfPreds = [];
        const rnnPreds = [];
        const actuals = [];
        
        sequentialData.forEach((window, i) => {
          if (i < sequentialData.length - 10) { // Reserve last 10 for testing
            const rfPred = simulateRandomForestPrediction(window);
            const rnnPred = simulateRNNPrediction(window);
            
            rfPreds.push({
              time: window.windowEnd + 1,
              predicted: Math.round(rfPred * 100) / 100,
              actual: window.target
            });
            
            rnnPreds.push({
              time: window.windowEnd + 1,
              predicted: Math.round(rnnPred * 100) / 100,
              actual: window.target
            });
            
            actuals.push(window.target);
          }
        });
        
        setPredictions({ rf: rfPreds, rnn: rnnPreds });
        break;
      case 8:
        // Calculate performance
        const rfActuals = predictions.rf.map(p => p.actual);
        const rfPredictedValues = predictions.rf.map(p => p.predicted);
        const rnnActuals = predictions.rnn.map(p => p.actual);
        const rnnPredictedValues = predictions.rnn.map(p => p.predicted);
        
        setPerformance({
          rf: calculatePerformance(rfPredictedValues, rfActuals),
          rnn: calculatePerformance(rnnPredictedValues, rnnActuals)
        });
        break;
    }
    setCurrentStep(prev => prev + 1);
  };

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(runSimulation, 1500);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isRunning, currentStep]);

  const resetSimulation = () => {
    setIsRunning(false);
    setCurrentStep(0);
    setTimeSeriesData(generateTimeSeriesData());
    setDbnFeatures([]);
    setSequentialData([]);
    setPredictions({ rf: [], rnn: [] });
    setPerformance({ rf: {}, rnn: {} });
  };

  const getCurrentVisualizationData = () => {
    if (currentStep <= 3 && dbnFeatures.length > 0) {
      return dbnFeatures.map(f => ({
        time: f.time,
        original: f.originalValue,
        layer1: f.layer1Feature,
        layer2: f.layer2Feature,
        layer3: f.layer3Feature,
        combined: f.combinedFeature
      }));
    }
    
    if (currentStep >= 7 && predictions.rf.length > 0) {
      const combined = timeSeriesData.map(d => ({
        time: d.time,
        actual: d.value,
        rfPredicted: null,
        rnnPredicted: null
      }));
      
      predictions.rf.forEach(pred => {
        if (combined[pred.time]) {
          combined[pred.time].rfPredicted = pred.predicted;
        }
      });
      
      predictions.rnn.forEach(pred => {
        if (combined[pred.time]) {
          combined[pred.time].rnnPredicted = pred.predicted;
        }
      });
      
      return combined.filter(d => d.time >= windowSize);
    }
    
    return timeSeriesData;
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center gap-2">
          <Brain className="text-blue-600" />
          DBN + Time Series Prediction Simulator
        </h1>
        <p className="text-gray-600 mb-4">
          Compare Deep Belief Network + Random Forest vs DBN + RNN for time series forecasting
        </p>
        
        {/* Controls */}
        <div className="flex items-center gap-4 mb-6">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
              isRunning 
                ? 'bg-red-500 hover:bg-red-600 text-white' 
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {isRunning ? <Pause size={20} /> : <Play size={20} />}
            {isRunning ? 'Pause' : 'Start'} Simulation
          </button>
          
          <button
            onClick={resetSimulation}
            className="flex items-center gap-2 px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg"
          >
            <RotateCcw size={20} />
            Reset
          </button>
          
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Window Size:</label>
            <input
              type="range"
              min="3"
              max="10"
              value={windowSize}
              onChange={(e) => setWindowSize(parseInt(e.target.value))}
              className="w-20"
            />
            <span className="text-sm">{windowSize}</span>
          </div>
        </div>

        {/* Progress */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Simulation Progress</span>
            <span className="text-sm text-gray-600">{currentStep}/{steps.length}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(currentStep / steps.length) * 100}%` }}
            />
          </div>
          <div className="mt-2 text-sm text-gray-700">
            {currentStep < steps.length ? `Current: ${steps[currentStep]}` : 'Simulation Complete'}
          </div>
        </div>
      </div>

      {/* Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Data Visualization</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={getCurrentVisualizationData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              
              {currentStep >= 1 && (
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#2563eb" 
                  strokeWidth={2}
                  name="Original Data"
                />
              )}
              
              {currentStep >= 2 && currentStep <= 3 && (
                <>
                  <Line type="monotone" dataKey="layer1" stroke="#dc2626" name="Layer 1 Features" />
                  <Line type="monotone" dataKey="layer2" stroke="#16a34a" name="Layer 2 Features" />
                  <Line type="monotone" dataKey="layer3" stroke="#ca8a04" name="Layer 3 Features" />
                </>
              )}
              
              {currentStep >= 7 && (
                <>
                  <Line 
                    type="monotone" 
                    dataKey="rfPredicted" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Random Forest Predictions"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="rnnPredicted" 
                    stroke="#16a34a" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="RNN Predictions"
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Model Architecture</h2>
          <div className="space-y-4">
            <div className="border rounded-lg p-4">
              <h3 className="font-semibold flex items-center gap-2 mb-2">
                <Brain className="text-blue-600" size={20} />
                Deep Belief Network (Feature Extraction)
              </h3>
              <div className="text-sm text-gray-600">
                Layer 1: Basic pattern detection<br/>
                Layer 2: Mid-level feature extraction<br/>
                Layer 3: High-level abstraction<br/>
                Output: Rich feature representations
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold flex items-center gap-2 mb-2">
                  <TreePine className="text-green-600" size={20} />
                  Random Forest
                </h3>
                <div className="text-sm text-gray-600">
                  • Uses windowed features<br/>
                  • Tree-based predictions<br/>
                  • Fast inference<br/>
                  • Good for patterns
                </div>
              </div>
              
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold flex items-center gap-2 mb-2">
                  <Zap className="text-purple-600" size={20} />
                  RNN/LSTM
                </h3>
                <div className="text-sm text-gray-600">
                  • Sequential processing<br/>
                  • Memory states<br/>
                  • Temporal dependencies<br/>
                  • Adaptive learning
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      {currentStep >= 8 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Performance Comparison</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-2">Accuracy Metrics</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={[
                  { name: 'MAE', RF: performance.rf.mae, RNN: performance.rnn.mae },
                  { name: 'RMSE', RF: performance.rf.rmse, RNN: performance.rnn.rmse },
                  { name: 'Accuracy %', RF: performance.rf.accuracy, RNN: performance.rnn.accuracy }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="RF" fill="#dc2626" name="Random Forest" />
                  <Bar dataKey="RNN" fill="#16a34a" name="RNN" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">System Characteristics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm">Training Speed:</span>
                  <span className="text-sm font-medium">RF: Fast | RNN: Slow</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Inference Speed:</span>
                  <span className="text-sm font-medium">RF: Very Fast | RNN: Medium</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Memory Usage:</span>
                  <span className="text-sm font-medium">RF: Low | RNN: High</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Real-time Suitability:</span>
                  <span className="text-sm font-medium">RF: Excellent | RNN: Good</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Interpretability:</span>
                  <span className="text-sm font-medium">RF: High | RNN: Low</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sequential Data Preview */}
      {currentStep >= 4 && sequentialData.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
          <h2 className="text-xl font-semibold mb-4">Sequential Window Example</h2>
          <div className="text-sm">
            <p className="mb-2">Window Size: {windowSize} time steps</p>
            <div className="bg-gray-100 p-3 rounded font-mono text-xs">
              <div className="mb-2">
                <strong>Input Window:</strong> [{sequentialData[0]?.input.slice(0, 3).map(i => i.originalValue.toFixed(2)).join(', ')}...]
              </div>
              <div className="mb-2">
                <strong>DBN Features:</strong> [{sequentialData[0]?.input.slice(0, 2).map(i => i.combinedFeature.toFixed(2)).join(', ')}...]
              </div>
              <div>
                <strong>Target:</strong> {sequentialData[0]?.target.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DBNTimeSeriesSimulator;