import { Shield } from "lucide-react";
import { LINKS } from "@/lib/content";

export default function Footer() {
  return (
    <footer className="mt-auto bg-slate-950 py-14 text-slate-400">
      <div className="mx-auto max-w-6xl px-4">
        <div className="grid gap-10 md:grid-cols-3">
          <div>
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600/20">
                <Shield className="h-4 w-4 text-blue-400" />
              </div>
              <span className="font-semibold text-white">ImageVerify AI</span>
            </div>
            <p className="mt-3 max-w-xs text-sm leading-relaxed">
              Explainable AI-generated image detection, powered by the
              Multi-Frequency Fusion Transformer (MFFT).
            </p>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-slate-500">
              Product
            </h4>
            <ul className="mt-3 space-y-2 text-sm">
              <li>
                <a href="#detect" className="transition-colors hover:text-white">
                  Detect Images
                </a>
              </li>
              <li>
                <a href="#pricing" className="transition-colors hover:text-white">
                  Pricing
                </a>
              </li>
              <li>
                <a href="/docs" className="transition-colors hover:text-white">
                  API Documentation
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-slate-500">
              Research
            </h4>
            <ul className="mt-3 space-y-2 text-sm">
              <li>
                <a
                  href={LINKS.space}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="transition-colors hover:text-white"
                >
                  Detection API (Hugging Face)
                </a>
              </li>
              <li>
                <a
                  href={LINKS.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="transition-colors hover:text-white"
                >
                  Source on GitHub
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-3 border-t border-white/10 pt-6 text-xs md:flex-row">
          <span>
            &copy; {new Date().getFullYear()} ImageVerify AI. All rights
            reserved.
          </span>
          <span className="text-slate-500">
            Research preview — results are probabilistic, not proof.
          </span>
        </div>
      </div>
    </footer>
  );
}
