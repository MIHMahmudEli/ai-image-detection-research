"use client";

import { Shield, Code2, Key, CreditCard } from "lucide-react";

const endpoints = [
  {
    method: "GET",
    path: "/",
    desc: "Health check — verifies the API is running",
    response: `{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0",
  "timestamp": "2026-06-20T..." 
}`,
  },
  {
    method: "POST",
    path: "/predict",
    desc: "Upload an image for AI detection",
    request: `curl -X POST https://api.imageverify.ai/predict \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "file=@image.jpg"`,
    response: `{
  "prediction": "ai_generated",
  "confidence": 0.972,
  "real_probability": 0.028,
  "ai_probability": 0.972,
  "processing_time_ms": 7.1,
  "anomaly_heatmap": "base64...",
  "tier": { "rpm": 100, "batch_size": 10, "report": true }
}`,
  },
  {
    method: "POST",
    path: "/predict/batch",
    desc: "Analyze multiple images in one request (Pro tier: max 10, Enterprise: max 100)",
    response: `{
  "results": [
    { "filename": "img1.jpg", "prediction": "ai_generated", ... },
    { "filename": "img2.jpg", "prediction": "real", ... }
  ],
  "summary": {
    "total": 2,
    "ai_generated": 1,
    "real": 1,
    "avg_real_probability": 0.486,
    "avg_ai_probability": 0.514
  }
}`,
  },
  {
    method: "GET",
    path: "/usage",
    desc: "Check your current rate limit usage",
    response: `{
  "tier": "pro",
  "requests_this_minute": 42,
  "rate_limit": 100
}`,
  },
];

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-white">
      <header className="border-b">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center gap-2">
          <Shield className="w-6 h-6 text-blue-600" />
          <span className="text-lg font-bold">
            Image<span className="text-blue-600">Verify</span> AI
          </span>
          <span className="text-sm text-gray-500 ml-auto">
            API Documentation
          </span>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold mb-4">API Reference</h1>
        <p className="text-gray-500 mb-8 max-w-2xl">
          Our REST API lets you integrate AI image detection into your own
          applications, platforms, and workflows.
        </p>

        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {[
            {
              icon: Key,
              title: "Authentication",
              desc: "Pass your API key in the Authorization header: Bearer YOUR_KEY",
            },
            {
              icon: CreditCard,
              title: "Free tier included",
              desc: "10 requests/min free. Pro tier at $9.99/mo unlocks heatmaps and batch.",
            },
            {
              icon: Code2,
              title: "SDKs coming soon",
              desc: "Python and JavaScript SDKs in development.",
            },
          ].map((item) => (
            <div key={item.title} className="border rounded-xl p-5">
              <item.icon className="w-6 h-6 text-blue-600 mb-3" />
              <h3 className="font-semibold mb-1">{item.title}</h3>
              <p className="text-sm text-gray-500">{item.desc}</p>
            </div>
          ))}
        </div>

        <div className="space-y-8">
          {endpoints.map((ep) => (
            <div key={ep.path} className="border rounded-xl overflow-hidden">
              <div className="p-5 border-b bg-gray-50 flex items-center gap-3">
                <span
                  className={`text-xs font-bold px-2 py-1 rounded ${
                    ep.method === "GET"
                      ? "bg-green-100 text-green-700"
                      : "bg-blue-100 text-blue-700"
                  }`}
                >
                  {ep.method}
                </span>
                <code className="text-sm font-mono">{ep.path}</code>
              </div>
              <div className="p-5 space-y-4">
                <p className="text-sm text-gray-600">{ep.desc}</p>
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">
                    Example Request
                  </p>
                  <pre className="bg-gray-900 text-gray-100 text-sm p-4 rounded-lg overflow-x-auto">
                    <code>{ep.request || "N/A"}</code>
                  </pre>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">
                    Example Response
                  </p>
                  <pre className="bg-gray-900 text-gray-100 text-sm p-4 rounded-lg overflow-x-auto">
                    <code>{ep.response}</code>
                  </pre>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-12 p-6 bg-blue-50 border border-blue-100 rounded-xl">
          <h2 className="font-semibold text-lg mb-2">Get Your API Key</h2>
          <p className="text-sm text-gray-600 mb-4">
            Sign up for a free account to receive your API key instantly.
          </p>
          <button className="bg-blue-600 text-white px-5 py-2 rounded-lg text-sm font-semibold hover:bg-blue-700 transition-colors">
            Get Free API Key
          </button>
        </div>
      </div>
    </div>
  );
}
