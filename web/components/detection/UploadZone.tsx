"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, XCircle } from "lucide-react";

type Props = {
  preview: string | null;
  onFile: (file: File) => void;
  onReset: () => void;
};

export default function UploadZone({ preview, onFile, onReset }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted[0]) onFile(accepted[0]);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxFiles: 1,
    maxSize: 20 * 1024 * 1024,
  });

  if (preview) {
    return (
      <div className="relative overflow-hidden rounded-2xl border border-slate-200 shadow-sm dark:border-slate-700">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={preview}
          alt="Selected image"
          className="max-h-96 w-full object-cover"
        />
        <button
          onClick={onReset}
          aria-label="Remove image"
          className="absolute right-3 top-3 rounded-full bg-slate-900/60 p-1.5 text-white backdrop-blur-sm transition-colors hover:bg-slate-900/80"
        >
          <XCircle className="h-5 w-5" />
        </button>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center transition-all ${
        isDragActive
          ? "scale-[1.01] border-blue-500 bg-blue-50 shadow-glow-sm dark:bg-blue-950/40"
          : "border-slate-300 bg-slate-50/50 hover:border-blue-400 hover:bg-blue-50/40 dark:border-slate-700 dark:bg-slate-900/50 dark:hover:border-blue-500 dark:hover:bg-blue-950/30"
      }`}
    >
      <input {...getInputProps()} />
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-100 to-indigo-100 dark:from-blue-950 dark:to-indigo-950">
        <Upload className="h-6 w-6 text-blue-600 dark:text-blue-400" />
      </div>
      <p className="mt-4 font-semibold text-slate-800 dark:text-slate-100">
        Drop an image here
      </p>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        or click to browse — PNG, JPG, WebP
      </p>
      <p className="mt-4 rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500 dark:bg-slate-800 dark:text-slate-400">
        Max 20 MB · Free tier: 10 images/min
      </p>
    </div>
  );
}
