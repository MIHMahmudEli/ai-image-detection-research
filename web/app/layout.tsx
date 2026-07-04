import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "ImageVerify AI — Detect AI-Generated Images",
  description:
    "Explainable AI-generated image detection powered by the Multi-Frequency Fusion Transformer (MFFT). Frequency-band analysis, anomaly heatmaps, three model sizes.",
  keywords: [
    "AI detection",
    "deepfake detection",
    "image verification",
    "AI-generated images",
    "image forensics",
    "MFFT",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${inter.className} min-h-screen bg-white text-slate-900 antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
