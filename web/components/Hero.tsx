import { ChevronRight, Sparkles } from "lucide-react";
import { HERO_STATS } from "@/lib/content";

export default function Hero() {
  return (
    <section className="relative overflow-hidden bg-cream-100 text-stone-900 dark:bg-slate-950 dark:text-white">
      {/* backdrop glows — light (warm parchment) */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 dark:hidden"
        style={{
          background:
            "radial-gradient(600px 300px at 20% 0%, rgba(37,99,235,0.10), transparent 70%), radial-gradient(700px 350px at 80% 100%, rgba(180,140,60,0.12), transparent 70%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.05] dark:hidden"
        style={{
          backgroundImage:
            "linear-gradient(to right, #78716c 1px, transparent 1px), linear-gradient(to bottom, #78716c 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />

      {/* backdrop glows — dark (candle-lit vintage: warm amber + deep blue) */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 hidden dark:block"
        style={{
          background:
            "radial-gradient(600px 300px at 20% 0%, rgba(37,99,235,0.22), transparent 70%), radial-gradient(700px 350px at 80% 100%, rgba(196,148,72,0.16), transparent 70%), radial-gradient(400px 400px at 50% 50%, rgba(140,100,50,0.08), transparent 70%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 hidden opacity-[0.04] dark:block"
        style={{
          backgroundImage:
            "linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />

      <div className="relative mx-auto max-w-6xl px-4 py-24 text-center md:py-32">
        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-stone-300 bg-cream-50/60 px-4 py-1.5 text-sm text-stone-700 backdrop-blur-sm dark:border-white/15 dark:bg-white/5 dark:text-blue-200 dark:shadow-glow-sm">
          <Sparkles className="h-3.5 w-3.5 animate-pulse" />
          <span>Multi-Frequency Fusion Transformer · Research Preview</span>
        </div>

        <h1 className="mx-auto max-w-3xl text-4xl font-bold leading-tight tracking-tight md:text-6xl">
          Know what&apos;s real.
          {/* pb keeps the gradient's paint box tall enough for descenders (the
              "g" in "images" was clipped by bg-clip-text + leading-tight) */}
          <span
            className="-mb-2 block animate-shimmer bg-gradient-to-r from-blue-800 via-teal-700 to-indigo-800 bg-clip-text pb-2 text-transparent dark:from-blue-400 dark:via-teal-300 dark:to-indigo-400"
            style={{ backgroundSize: "200% 200%" }}
          >
            Detect AI-generated images.
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-stone-600 dark:text-slate-300 md:text-xl">
          MFFT analyzes images across multiple frequency bands — where
          generative models leave their fingerprints — and explains every
          verdict with band-level evidence and anomaly heatmaps.
        </p>

        <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <a
            href="#detect"
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-700 to-indigo-700 px-7 py-3.5 font-semibold text-white shadow-lg transition-all hover:-translate-y-0.5 hover:brightness-110 dark:from-blue-500 dark:to-indigo-500 dark:shadow-glow"
          >
            Analyze an Image <ChevronRight className="h-4 w-4" />
          </a>
          <a
            href="#how"
            className="inline-flex items-center gap-2 rounded-xl border border-stone-300 bg-cream-50/50 px-7 py-3.5 font-medium text-stone-800 backdrop-blur-sm transition-all hover:border-stone-400 hover:bg-cream-50 dark:border-white/20 dark:bg-white/5 dark:text-white dark:hover:border-white/40 dark:hover:bg-white/10"
          >
            See How It Works
          </a>
        </div>

        <dl className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:grid-cols-3">
          {HERO_STATS.map((s, i) => (
            <div
              key={s.label}
              className="group rounded-2xl border border-stone-200 bg-cream-50/70 px-4 py-5 backdrop-blur-sm transition-all hover:-translate-y-1 hover:border-blue-400/60 hover:bg-cream-50 hover:shadow-md dark:border-white/10 dark:bg-white/5 dark:hover:border-blue-400/40 dark:hover:bg-white/10 dark:hover:shadow-glow-sm"
              style={{ animationDelay: `${i * 120}ms` }}
            >
              <dt className="order-2 mt-1 text-xs font-medium uppercase tracking-wide text-stone-500 transition-colors group-hover:text-blue-700 dark:text-slate-400 dark:group-hover:text-blue-200">
                {s.label}
              </dt>
              <dd className="bg-gradient-to-r from-stone-900 to-blue-800 bg-clip-text text-2xl font-bold text-transparent dark:from-white dark:to-blue-200">
                {s.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
