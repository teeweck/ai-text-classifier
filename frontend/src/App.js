import React, { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Failed to connect to backend" });
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial" }}>
      <h1>AI Text Classifier</h1>

      <textarea
        rows="4"
        cols="50"
        placeholder="Enter text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <br /><br />

      <button onClick={handleSubmit} disabled={loading || !text}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      <br /><br />

      {result && !result.error && (
        <div>
          <strong>Prediction:</strong> {result.prediction}<br />
          <strong>Confidence:</strong> {result.confidence}
        </div>
      )}

      {result?.error && (
        <div style={{ color: "red" }}>{result.error}</div>
      )}
    </div>
  );
}

export default App;
