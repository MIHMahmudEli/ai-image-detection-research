import { HOW_IT_WORKS_STEPS } from "@/lib/content";

export default function HowItWorks() {
  return (
    <section id="how" className="bg-slate-50 py-20">
      <div className="mx-auto max-w-5xl px-4">
        <p className="text-center text-xs font-semibold uppercase tracking-widest text-blue-600">
          The Method
        </p>
        <h2 className="mt-2 text-center text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">
          How It Works
        </h2>
        <p className="mx-auto mt-3 max-w-lg text-center text-slate-500">
          A frequency-native architecture, built for explainable image
          forensics
        </p>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {HOW_IT_WORKS_STEPS.map((item, i) => (
            <div
              key={item.step}
              className="group relative rounded-2xl border border-slate-200 bg-white p-7 shadow-sm transition-all hover:-translate-y-1 hover:shadow-md"
            >
              <span className="text-4xl font-bold text-slate-100 transition-colors group-hover:text-blue-100">
                {item.step}
              </span>
              <h3 className="mt-3 font-semibold text-slate-900">
                {item.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-500">
                {item.desc}
              </p>
              {i < HOW_IT_WORKS_STEPS.length - 1 && (
                <div className="absolute -right-3 top-1/2 hidden h-px w-6 bg-slate-300 md:block" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
