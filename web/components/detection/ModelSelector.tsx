"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Settings } from "lucide-react";
import type { ModelOption } from "@/lib/types";

type Props = {
  models: ModelOption[];
  modelId: string;
  onSelect: (id: string) => void;
};

export default function ModelSelector({ models, modelId, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // close when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const current = models.find((m) => m.id === modelId);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((s) => !s)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="inline-flex items-center gap-2 rounded-xl border border-stone-300 bg-cream-100 px-4 py-2.5 text-sm font-medium text-stone-700 shadow-sm transition-all hover:border-blue-400 hover:bg-cream-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-blue-500 dark:hover:bg-blue-950/40"
      >
        <Settings className="h-4 w-4 text-stone-400 dark:text-slate-400" />
        <span>
          Model:{" "}
          <span className="font-semibold capitalize text-stone-900 dark:text-white">
            MFFT-{modelId}
          </span>
        </span>
        <span className="text-xs text-stone-400 dark:text-slate-400">{current?.params}</span>
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute left-1/2 z-30 mt-2 w-80 -translate-x-1/2 animate-fade-in rounded-2xl border border-stone-200 bg-cream-100 p-2 shadow-xl dark:border-slate-700 dark:bg-slate-900 dark:shadow-glow-sm"
        >
          <p className="px-3 pb-1.5 pt-2 text-xs font-semibold uppercase tracking-widest text-stone-400 dark:text-slate-500">
            Detection Model
          </p>
          {models.map((m) => {
            const active = m.id === modelId;
            return (
              <button
                key={m.id}
                role="option"
                aria-selected={active}
                disabled={!m.loaded}
                onClick={() => {
                  onSelect(m.id);
                  setOpen(false);
                }}
                className={`w-full rounded-xl px-3 py-3 text-left transition-colors ${
                  active
                    ? "bg-blue-50 dark:bg-blue-950/50"
                    : "hover:bg-cream-100 dark:hover:bg-slate-800/60"
                } ${!m.loaded ? "cursor-not-allowed opacity-40" : ""}`}
              >
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-sm font-semibold capitalize text-stone-900 dark:text-white">
                    MFFT-{m.id}
                    {active && (
                      <Check className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    )}
                  </span>
                  <span className="text-xs font-medium text-stone-400 dark:text-slate-400">
                    {m.params} params
                  </span>
                </div>
                <p className="mt-0.5 text-xs text-stone-500 dark:text-slate-400">
                  {m.description}
                  {!m.loaded && " · unavailable"}
                </p>
              </button>
            );
          })}
          <p className="border-t border-stone-200 px-3 pb-1.5 pt-2 text-[11px] leading-relaxed text-stone-400 dark:border-slate-800 dark:text-slate-500">
            Demo checkpoints (pilot training). Full-scale weights arrive after
            the training campaign.
          </p>
        </div>
      )}
    </div>
  );
}
