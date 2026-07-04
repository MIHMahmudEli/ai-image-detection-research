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
            "radial-gradient(600px 300px at 20% 0%, rgba(37,99,235,0.4), transparent 70%), radial-gradient(700px 350px at 80% 100%, rgba(79,70,229,0.3), transparent 70%), radial-gradient(400px 400px at 50% 50%, rgba(16,185,129,0.08), transparent 70%)",
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
        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-4 py-1.5 text-sm text-blue-200 shadow-glow-sm backdrop-blur-sm">
          <Sparkles className="h-3.5 w-3.5 animate-pulse" />
          <span>Multi-Frequency Fusion Transformer · Research Preview</span>
        </div>

        <h1 className="mx-auto max-w-3xl text-4xl font-bold leading-tight tracking-tight md:text-6xl">
          Know what&apos;s real.
          <span
            className="block animate-shimmer bg-gradient-to-r from-blue-400 via-teal-300 to-indigo-400 bg-clip-text text-transparent"
            style={{ backgroundSize: "200% 200%" }}
          >
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
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-500 px-7 py-3.5 font-semibold text-white shadow-glow transition-all hover:-translate-y-0.5 hover:brightness-110"
          >
            Analyze an Image <ChevronRight className="h-4 w-4" />
          </a>
          <a
            href="#how"
            className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/5 px-7 py-3.5 font-medium text-white backdrop-blur-sm transition-all hover:border-white/40 hover:bg-white/10"
          >
            See How It Works
          </a>
        </div>

        <dl className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:grid-cols-3">
          {HERO_STATS.map((s, i) => (
            <div
              key={s.label}
              className="group rounded-2xl border border-white/10 bg-white/5 px-4 py-5 backdrop-blur-sm transition-all hover:-translate-y-1 hover:border-blue-400/40 hover:bg-white/10 hover:shadow-glow-sm"
              style={{ animationDelay: `${i * 120}ms` }}
            >
              <dt className="order-2 mt-1 text-xs font-medium uppercase tracking-wide text-slate-400 transition-colors group-hover:text-blue-200">
                {s.label}
              </dt>
              <dd className="bg-gradient-to-r from-white to-blue-200 bg-clip-text text-2xl font-bold text-transparent">
                {s.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
