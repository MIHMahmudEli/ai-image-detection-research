import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ImageVerify AI — Detect AI-Generated Images",
  description:
    "Advanced AI-powered detection tool to identify AI-generated and deepfake images. Multi-frequency analysis with explainable results.",
  keywords: [
    "AI detection",
    "deepfake detection",
    "image verification",
    "AI-generated images",
    "image forensics",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900 font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
