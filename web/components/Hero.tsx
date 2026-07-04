import { ChevronRight, Sparkles } from "lucide-react";
import { HERO_STATS } from "@/lib/content";

export default function Hero() {
  return (
    <section className="relative overflow-hidden bg-slate-950 text-white">
      {/* backdrop glows */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(600px 300px at 20% 0%, rgba(37,99,235,0.35), transparent 70%), radial-gradient(700px 350px at 80% 100%, rgba(79,70,229,0.25), transparent 70%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage:
            "linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />

      <div className="relative mx-auto max-w-6xl px-4 py-24 text-center md:py-32">
        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-4 py-1.5 text-sm text-blue-200 backdrop-blur-sm">
          <Sparkles className="h-3.5 w-3.5" />
          <span>Multi-Frequency Fusion Transformer · Research Preview</span>
        </div>

        <h1 className="mx-auto max-w-3xl text-4xl font-bold leading-tight tracking-tight md:text-6xl">
          Know what&apos;s real.
          <span className="block bg-gradient-to-r from-blue-400 via-indigo-300 to-blue-400 bg-clip-text text-transparent">
            Detect AI-generated images.
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-slate-300 md:text-xl">
          MFFT analyzes images across multiple frequency bands — where
          generative models leave their fingerprints — and explains every
          verdict with band-level evidence and anomaly heatmaps.
        </p>

        <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <a
            href="#detect"
            className="inline-flex items-center gap-2 rounded-xl bg-white px-7 py-3.5 font-semibold text-slate-900 shadow-lg shadow-blue-900/30 transition-all hover:-translate-y-0.5 hover:bg-blue-50"
          >
            Analyze an Image <ChevronRight className="h-4 w-4" />
          </a>
          <a
            href="#how"
            className="inline-flex items-center gap-2 rounded-xl border border-white/20 px-7 py-3.5 font-medium text-white transition-colors hover:bg-white/10"
          >
            See How It Works
          </a>
        </div>

        <dl className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:grid-cols-3">
          {HERO_STATS.map((s) => (
            <div
              key={s.label}
              className="rounded-2xl border border-white/10 bg-white/5 px-4 py-5 backdrop-blur-sm"
            >
              <dt className="order-2 mt-1 text-xs font-medium uppercase tracking-wide text-slate-400">
                {s.label}
              </dt>
              <dd className="text-2xl font-bold text-white">{s.value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
