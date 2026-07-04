"use client";

import { BarChart3, CheckCircle, XCircle } from "lucide-react";
import type { PredictionResult } from "@/lib/types";

function ProbabilityBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "real" | "ai";
}) {
  const pct = value * 100;
  return (
    <div>
      <div className="mb-1 flex justify-between text-sm">
        <span
          className={`font-medium ${
            tone === "real"
              ? "text-emerald-600 dark:text-emerald-400"
              : "text-rose-600 dark:text-rose-400"
          }`}
        >
          {label}
        </span>
        <span className="tabular-nums text-stone-600 dark:text-slate-300">
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-cream-200 dark:bg-slate-800">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            tone === "real"
              ? "bg-gradient-to-r from-emerald-500 to-teal-400"
              : "bg-gradient-to-r from-rose-500 to-orange-400"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function BandAnalysis({ bands }: { bands: Record<string, number> }) {
  const total = Object.values(bands).reduce((a, b) => a + b, 0);
  return (
    <div>
      <p className="mb-2 text-sm font-medium text-stone-600 dark:text-slate-300">
        Frequency Band Analysis
      </p>
      <div className="space-y-2 rounded-xl bg-cream-200/60 p-3 dark:bg-slate-800/60">
        {Object.entries(bands).map(([band, value]) => {
          const pct = total > 0 ? (value / total) * 100 : 0;
          return (
            <div key={band}>
              <div className="mb-0.5 flex justify-between text-xs text-stone-500 dark:text-slate-400">
                <span className="capitalize">{band.replace(/_/g, " ")}</span>
                <span className="tabular-nums">{pct.toFixed(0)}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-cream-300 dark:bg-slate-700">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-blue-500 to-indigo-400 transition-all duration-700"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

type Props = {
  result: PredictionResult | null;
  modelId: string;
};

export default function ResultCard({ result, modelId }: Props) {
  if (!result) {
    return (
      <div className="flex h-full min-h-[280px] flex-col items-center justify-center rounded-2xl border-2 border-dashed border-stone-200 p-12 text-center dark:border-slate-800">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cream-200 dark:bg-slate-800">
          <BarChart3 className="h-6 w-6 text-stone-300 dark:text-slate-600" />
        </div>
        <p className="mt-4 font-medium text-stone-400 dark:text-slate-500">
          Your analysis will appear here
        </p>
      </div>
    );
  }

  const isAI = result.prediction === "ai_generated";

  return (
    <div className="animate-fade-in space-y-5 rounded-2xl border border-stone-200 bg-cream-100 p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 dark:shadow-glow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-stone-900 dark:text-white">
          Analysis Result
        </h3>
        <span className="rounded-full bg-cream-200 px-2.5 py-1 text-xs font-medium text-stone-500 dark:bg-slate-800 dark:text-slate-400">
          MFFT-{modelId} · {result.processing_time_ms.toFixed(0)} ms
        </span>
      </div>

      {/* Verdict */}
      <div
        className={`flex items-center gap-3 rounded-xl border p-4 ${
          isAI
            ? "border-rose-200 bg-rose-50 dark:border-rose-900/60 dark:bg-rose-950/40"
            : "border-emerald-200 bg-emerald-50 dark:border-emerald-900/60 dark:bg-emerald-950/40"
        }`}
      >
        {isAI ? (
          <XCircle className="h-9 w-9 flex-shrink-0 text-rose-500" />
        ) : (
          <CheckCircle className="h-9 w-9 flex-shrink-0 text-emerald-500" />
        )}
        <div>
          <p className="text-lg font-bold text-stone-900 dark:text-white">
            {isAI ? "AI-Generated" : "Likely Authentic"}
          </p>
          <p className="text-sm text-stone-600 dark:text-slate-300">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <ProbabilityBar
          label="Real"
          value={result.real_probability}
          tone="real"
        />
        <ProbabilityBar
          label="AI-Generated"
          value={result.ai_probability}
          tone="ai"
        />
      </div>

      {result.frequency_band_contributions && (
        <BandAnalysis bands={result.frequency_band_contributions} />
      )}

      {result.anomaly_heatmap && (
        <div>
          <p className="mb-2 text-sm font-medium text-stone-600 dark:text-slate-300">
            Anomaly Heatmap
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${result.anomaly_heatmap}`}
            alt="Anomaly heatmap"
            className="w-full rounded-xl border border-stone-200 dark:border-slate-700"
          />
        </div>
      )}

      <p className="border-t border-stone-200 pt-3 text-xs leading-relaxed text-stone-400 dark:border-slate-800 dark:text-slate-500">
        Results are probabilistic evidence, not proof. For high-stakes
        decisions, combine with human review.
      </p>
    </div>
  );
}
