"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Shield } from "lucide-react";
import { fetchModels, predictImage } from "@/lib/api";
import { FALLBACK_MODELS } from "@/lib/content";
import type { ModelOption, PredictionResult } from "@/lib/types";
import ModelSelector from "./ModelSelector";
import UploadZone from "./UploadZone";
import ResultCard from "./ResultCard";

const STORAGE_KEY = "mfft_model";

export default function DetectionSection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelOption[]>(FALLBACK_MODELS);
  const [modelId, setModelId] = useState<string>("base");

  // restore saved model choice, then refresh the live list from the API
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setModelId(saved);
    fetchModels().then((data) => {
      if (!data?.models?.length) return;
      setModels(data.models);
      const stillValid = data.models.some(
        (m) => m.id === (saved ?? "base") && m.loaded
      );
      if (!stillValid && data.default) setModelId(data.default);
    });
  }, []);

  const selectModel = (id: string) => {
    setModelId(id);
    localStorage.setItem(STORAGE_KEY, id);
    // a new model means the current verdict no longer applies
    setResult(null);
  };

  const handleFile = (f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await predictImage(file, modelId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="detect" className="bg-white py-20">
      <div className="mx-auto max-w-4xl px-4">
        <p className="text-center text-xs font-semibold uppercase tracking-widest text-blue-600">
          Live Demo
        </p>
        <h2 className="mt-2 text-center text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">
          Detect AI-Generated Images
        </h2>
        <p className="mx-auto mt-3 max-w-xl text-center text-slate-500">
          Upload an image and get an explainable verdict in under a second
        </p>

        <div className="mt-8 flex justify-center">
          <ModelSelector
            models={models}
            modelId={modelId}
            onSelect={selectModel}
          />
        </div>

        <div className="mt-8 grid gap-8 md:grid-cols-2">
          <div>
            <UploadZone preview={preview} onFile={handleFile} onReset={reset} />

            {preview && !result && (
              <button
                onClick={handleDetect}
                disabled={loading}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 py-3.5 font-semibold text-white shadow-sm transition-all hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <span className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Analyzing…
                  </>
                ) : (
                  <>
                    <Shield className="h-5 w-5" />
                    Detect AI Generation
                  </>
                )}
              </button>
            )}

            {error && (
              <div className="mt-4 flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
                <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                {error}
              </div>
            )}
          </div>

          <ResultCard result={result} modelId={modelId} />
        </div>
      </div>
    </section>
  );
}
