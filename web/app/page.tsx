"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  Shield,
  BarChart3,
  Zap,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ChevronRight,
  Star,
} from "lucide-react";

type PredictionResult = {
  prediction: string;
  confidence: number;
  real_probability: number;
  ai_probability: number;
  processing_time_ms: number;
  anomaly_heatmap?: string;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxFiles: 1,
    maxSize: 20 * 1024 * 1024,
  });

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { Authorization: "Bearer free" },
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Detection failed");
      }
      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="border-b bg-white">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="w-7 h-7 text-blue-600" />
            <span className="text-xl font-bold">
              Image<span className="text-blue-600">Verify</span> AI
            </span>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-600">
            <a href="#how" className="hover:text-blue-600 transition-colors">
              How It Works
            </a>
            <a href="#pricing" className="hover:text-blue-600 transition-colors">
              Pricing
            </a>
            <a href="#docs" className="hover:text-blue-600 transition-colors">
              API
            </a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-600 via-blue-700 to-teal-600 text-white">
        <div className="max-w-6xl mx-auto px-4 py-20 md:py-28 text-center">
          <div className="inline-flex items-center gap-1.5 bg-white/10 backdrop-blur-sm rounded-full px-4 py-1.5 text-sm mb-8">
            <Zap className="w-4 h-4" />
            <span>Advanced Multi-Frequency AI Detection</span>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold mb-6 leading-tight">
            Know What&apos;s Real.
            <br />
            Detect AI-Generated Images Instantly.
          </h1>
          <p className="text-lg md:text-xl text-blue-100 max-w-2xl mx-auto mb-10">
            Our MFFT model analyzes images across multiple frequency bands to
            detect AI generation with explainable results and anomaly heatmaps.
          </p>
          <div className="flex items-center justify-center gap-4">
            <a
              href="#detect"
              className="inline-flex items-center gap-2 bg-white text-blue-700 font-semibold px-6 py-3 rounded-lg hover:bg-blue-50 transition-colors shadow-lg"
            >
              Try It Free <ChevronRight className="w-4 h-4" />
            </a>
            <a
              href="#pricing"
              className="inline-flex items-center gap-2 border border-white/30 text-white font-medium px-6 py-3 rounded-lg hover:bg-white/10 transition-colors"
            >
              View Pricing
            </a>
          </div>
          <div className="mt-12 grid grid-cols-3 gap-4 max-w-lg mx-auto text-center text-sm text-blue-200">
            <div>
              <div className="text-2xl font-bold text-white">97.2%</div>
              <div>Detection Accuracy</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white">3 Models</div>
              <div>Frequency Bands</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white">&lt;1s</div>
              <div>Processing Time</div>
            </div>
          </div>
        </div>
      </section>

      {/* Detection Interface */}
      <section id="detect" className="py-16 bg-white">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-2">
            Detect AI-Generated Images
          </h2>
          <p className="text-gray-500 text-center mb-10 max-w-xl mx-auto">
            Upload an image and get instant analysis with explainable results
          </p>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Upload zone */}
            <div>
              {!preview ? (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
                    isDragActive
                      ? "border-blue-400 bg-blue-50"
                      : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="font-medium text-gray-700">
                    Drop an image here
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    or click to browse (PNG, JPG, WebP)
                  </p>
                  <p className="text-xs text-gray-400 mt-3">
                    Max 20MB — Free tier: 10 images/min
                  </p>
                </div>
              ) : (
                <div className="relative">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full rounded-xl shadow-lg object-cover max-h-96"
                  />
                  <button
                    onClick={reset}
                    className="absolute top-3 right-3 bg-black/50 text-white rounded-full p-1.5 hover:bg-black/70 transition-colors"
                  >
                    <XCircle className="w-5 h-5" />
                  </button>
                </div>
              )}

              {preview && !result && (
                <button
                  onClick={handleDetect}
                  disabled={loading}
                  className="mt-4 w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Shield className="w-5 h-5" />
                      Detect AI Generation
                    </>
                  )}
                </button>
              )}

              {error && (
                <div className="mt-4 bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-sm flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}
            </div>

            {/* Results */}
            <div>
              {!result ? (
                <div className="border-2 border-dashed border-gray-200 rounded-xl p-12 text-center h-full flex flex-col items-center justify-center">
                  <BarChart3 className="w-12 h-12 text-gray-300 mb-4" />
                  <p className="text-gray-400 font-medium">
                    Your results will appear here
                  </p>
                </div>
              ) : (
                <div className="border rounded-xl p-6 space-y-5 animate-fade-in">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-lg">Analysis Result</h3>
                    <span className="text-sm text-gray-400">
                      {result.processing_time_ms.toFixed(0)}ms
                    </span>
                  </div>

                  {/* Prediction badge */}
                  <div
                    className={`flex items-center gap-3 p-4 rounded-lg ${
                      result.prediction === "ai_generated"
                        ? "bg-red-50 border border-red-200"
                        : "bg-green-50 border border-green-200"
                    }`}
                  >
                    {result.prediction === "ai_generated" ? (
                      <XCircle className="w-8 h-8 text-red-500" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-green-500" />
                    )}
                    <div>
                      <p className="font-bold text-lg">
                        {result.prediction === "ai_generated"
                          ? "AI-Generated"
                          : "Likely Authentic"}
                      </p>
                      <p className="text-sm text-gray-600">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Probability bars */}
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-green-600 font-medium">
                          Real
                        </span>
                        <span>
                          {(result.real_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-500 rounded-full transition-all duration-500"
                          style={{
                            width: `${result.real_probability * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-red-600 font-medium">AI</span>
                        <span>
                          {(result.ai_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-red-500 rounded-full transition-all duration-500"
                          style={{
                            width: `${result.ai_probability * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Heatmap */}
                  {result.anomaly_heatmap && (
                    <div>
                      <p className="text-sm font-medium text-gray-600 mb-2">
                        Anomaly Heatmap
                      </p>
                      <img
                        src={`data:image/png;base64,${result.anomaly_heatmap}`}
                        alt="Anomaly heatmap"
                        className="w-full rounded-lg border"
                      />
                    </div>
                  )}

                  <p className="text-xs text-gray-400 pt-2 border-t">
                    Results are probabilistic. For high-stakes decisions,
                    consider our Pro tier with detailed reports.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how" className="py-16 bg-gray-50">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-4">
            How It Works
          </h2>
          <p className="text-gray-500 text-center mb-12 max-w-lg mx-auto">
            Our Multi-Frequency Fusion Transformer analyzes images at every
            level
          </p>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: "1",
                title: "Frequency Decomposition",
                desc: "Image is split into low, mid, and high frequency bands using DCT analysis",
              },
              {
                step: "2",
                title: "Multi-Band Feature Extraction",
                desc: "Each frequency band is independently analyzed by a CNN feature extractor",
              },
              {
                step: "3",
                title: "Cross-Attention Fusion",
                desc: "Frequency features are fused via cross-attention to produce a final verdict",
              },
            ].map((item) => (
              <div
                key={item.step}
                className="bg-white rounded-xl p-6 shadow-sm border"
              >
                <div className="w-10 h-10 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center font-bold mb-4">
                  {item.step}
                </div>
                <h3 className="font-semibold mb-2">{item.title}</h3>
                <p className="text-sm text-gray-500">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="py-16 bg-white">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-4">
            Simple, Transparent Pricing
          </h2>
          <p className="text-gray-500 text-center mb-12 max-w-lg mx-auto">
            Start free, upgrade as you grow
          </p>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                name: "Free",
                price: "$0",
                period: "forever",
                features: [
                  "10 detections/min",
                  "Single image upload",
                  "Basic confidence score",
                  "Community support",
                ],
                cta: "Get Started",
                featured: false,
              },
              {
                name: "Pro",
                price: "$9.99",
                period: "/month",
                features: [
                  "100 detections/min",
                  "Batch upload (up to 10)",
                  "Anomaly heatmaps",
                  "Detailed PDF reports",
                  "API access",
                  "Priority support",
                ],
                cta: "Start Free Trial",
                featured: true,
              },
              {
                name: "Enterprise",
                price: "Custom",
                period: "",
                features: [
                  "Unlimited detections",
                  "On-premise deployment",
                  "Custom model fine-tuning",
                  "SLA guarantees",
                  "Dedicated account manager",
                  "White-label option",
                ],
                cta: "Contact Sales",
                featured: false,
              },
            ].map((plan) => (
              <div
                key={plan.name}
                className={`rounded-xl p-6 border ${
                  plan.featured
                    ? "border-blue-500 shadow-lg shadow-blue-100 relative"
                    : "border-gray-200"
                }`}
              >
                {plan.featured && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-blue-600 text-white text-xs font-semibold px-4 py-1 rounded-full flex items-center gap-1">
                    <Star className="w-3 h-3" />
                    Most Popular
                  </div>
                )}
                <h3 className="text-lg font-semibold mb-1">{plan.name}</h3>
                <div className="mb-4">
                  <span className="text-3xl font-bold">{plan.price}</span>
                  <span className="text-gray-500 text-sm ml-1">
                    {plan.period}
                  </span>
                </div>
                <ul className="space-y-2.5 mb-6">
                  {plan.features.map((f) => (
                    <li
                      key={f}
                      className="text-sm text-gray-600 flex items-start gap-2"
                    >
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>
                <button
                  className={`w-full py-2.5 rounded-lg font-semibold text-sm transition-colors ${
                    plan.featured
                      ? "bg-blue-600 text-white hover:bg-blue-700"
                      : "border border-gray-300 text-gray-700 hover:bg-gray-50"
                  }`}
                >
                  {plan.cta}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-12 mt-auto">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-blue-400" />
              <span className="text-white font-semibold">
                ImageVerify AI
              </span>
            </div>
            <div className="text-sm">
              &copy; {new Date().getFullYear()} ImageVerify AI. All rights
              reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
