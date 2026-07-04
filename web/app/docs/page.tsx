"use client";

import { Shield, Code2, Key, CreditCard, ArrowLeft } from "lucide-react";
import ThemeToggle from "@/components/ThemeToggle";

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
    <div className="min-h-screen bg-white dark:bg-slate-950">
      <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/80 backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-950/80">
        <div className="mx-auto flex h-16 max-w-5xl items-center gap-3 px-4">
          <a
            href="/"
            className="flex items-center gap-1.5 text-sm text-slate-500 transition-colors hover:text-blue-600 dark:text-slate-400 dark:hover:text-blue-400"
          >
            <ArrowLeft className="h-4 w-4" />
            Home
          </a>
          <span className="h-5 w-px bg-slate-200 dark:bg-slate-800" />
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600">
            <Shield className="h-4 w-4 text-white" />
          </div>
          <span className="text-lg font-bold text-slate-900 dark:text-white">
            Image<span className="text-blue-600 dark:text-blue-400">Verify</span>{" "}
            AI
          </span>
          <span className="ml-auto text-sm text-slate-500 dark:text-slate-400">
            API Documentation
          </span>
          <ThemeToggle />
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-4 py-12">
        <h1 className="mb-4 text-4xl font-bold text-slate-900 dark:text-white">
          API Reference
        </h1>
        <p className="mb-8 max-w-2xl text-slate-500 dark:text-slate-400">
          Our REST API lets you integrate AI image detection into your own
          applications, platforms, and workflows.
        </p>

        <div className="mb-12 grid gap-6 md:grid-cols-3">
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
            <div
              key={item.title}
              className="rounded-xl border border-slate-200 p-5 transition-all hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-md dark:border-slate-800 dark:bg-slate-900 dark:hover:border-blue-700"
            >
              <item.icon className="mb-3 h-6 w-6 text-blue-600 dark:text-blue-400" />
              <h3 className="mb-1 font-semibold text-slate-900 dark:text-white">
                {item.title}
              </h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {item.desc}
              </p>
            </div>
          ))}
        </div>

        <div className="space-y-8">
          {endpoints.map((ep) => (
            <div
              key={ep.path}
              className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-800"
            >
              <div className="flex items-center gap-3 border-b border-slate-200 bg-slate-50 p-5 dark:border-slate-800 dark:bg-slate-900">
                <span
                  className={`rounded px-2 py-1 text-xs font-bold ${
                    ep.method === "GET"
                      ? "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-400"
                      : "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-400"
                  }`}
                >
                  {ep.method}
                </span>
                <code className="font-mono text-sm text-slate-900 dark:text-slate-100">
                  {ep.path}
                </code>
              </div>
              <div className="space-y-4 p-5 dark:bg-slate-950">
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  {ep.desc}
                </p>
                <div>
                  <p className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                    Example Request
                  </p>
                  <pre className="overflow-x-auto rounded-lg bg-slate-900 p-4 text-sm text-slate-100 dark:border dark:border-slate-800">
                    <code>{ep.request || "N/A"}</code>
                  </pre>
                </div>
                <div>
                  <p className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                    Example Response
                  </p>
                  <pre className="overflow-x-auto rounded-lg bg-slate-900 p-4 text-sm text-slate-100 dark:border dark:border-slate-800">
                    <code>{ep.response}</code>
                  </pre>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-12 rounded-xl border border-blue-100 bg-blue-50 p-6 dark:border-blue-900/50 dark:bg-blue-950/40">
          <h2 className="mb-2 text-lg font-semibold text-slate-900 dark:text-white">
            Get Your API Key
          </h2>
          <p className="mb-4 text-sm text-slate-600 dark:text-slate-300">
            Sign up for a free account to receive your API key instantly.
          </p>
          <button className="btn-primary px-5 py-2 text-sm">
            Get Free API Key
          </button>
        </div>
      </div>
    </div>
  );
}
