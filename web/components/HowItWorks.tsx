import { HOW_IT_WORKS_STEPS } from "@/lib/content";

export default function HowItWorks() {
  return (
    <section id="how" className="bg-slate-50 py-20 dark:bg-slate-900/50">
      <div className="mx-auto max-w-5xl px-4">
        <p className="section-eyebrow">The Method</p>
        <h2 className="section-heading">How It Works</h2>
        <p className="section-sub">
          A frequency-native architecture, built for explainable image
          forensics
        </p>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {HOW_IT_WORKS_STEPS.map((item, i) => (
            <div
              key={item.step}
              className="group relative rounded-2xl border border-slate-200 bg-white p-7 shadow-sm transition-all hover:-translate-y-1 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-100 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-blue-700 dark:hover:shadow-glow-sm"
            >
              <span className="bg-gradient-to-br from-blue-600 to-indigo-500 bg-clip-text text-4xl font-bold text-transparent opacity-25 transition-opacity group-hover:opacity-100 dark:from-blue-400 dark:to-indigo-300">
                {item.step}
              </span>
              <h3 className="mt-3 font-semibold text-slate-900 dark:text-white">
                {item.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-500 dark:text-slate-400">
                {item.desc}
              </p>
              {i < HOW_IT_WORKS_STEPS.length - 1 && (
                <div className="absolute -right-3 top-1/2 hidden h-px w-6 bg-slate-300 dark:bg-slate-700 md:block" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
