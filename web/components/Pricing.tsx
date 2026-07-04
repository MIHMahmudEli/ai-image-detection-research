import { CheckCircle2, Star } from "lucide-react";
import { PRICING_PLANS } from "@/lib/content";

export default function Pricing() {
  return (
    <section id="pricing" className="bg-cream-50 py-20 dark:bg-slate-950">
      <div className="mx-auto max-w-5xl px-4">
        <p className="section-eyebrow">Pricing</p>
        <h2 className="section-heading">Simple, Transparent Pricing</h2>
        <p className="section-sub">Start free, upgrade as you grow</p>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {PRICING_PLANS.map((plan) => (
            <div
              key={plan.name}
              className={`relative flex flex-col rounded-2xl p-7 transition-all hover:-translate-y-1 ${
                plan.featured
                  ? "border-2 border-blue-700 bg-cream-100 shadow-xl shadow-stone-200 dark:border-blue-500 dark:bg-slate-900 dark:shadow-glow md:scale-[1.03]"
                  : "border border-stone-200 bg-cream-100 shadow-sm hover:shadow-md dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-700"
              }`}
            >
              {plan.featured && (
                <div className="absolute -top-3.5 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full bg-gradient-to-r from-blue-700 to-indigo-700 px-4 py-1 text-xs font-semibold text-white shadow-glow-sm dark:from-blue-600 dark:to-indigo-600">
                  <Star className="h-3 w-3 fill-current" />
                  Most Popular
                </div>
              )}
              <h3 className="text-sm font-semibold uppercase tracking-wide text-stone-500 dark:text-slate-400">
                {plan.name}
              </h3>
              <div className="mt-2 flex items-baseline gap-1">
                <span className="text-4xl font-bold tracking-tight text-stone-900 dark:text-white">
                  {plan.price}
                </span>
                <span className="text-sm text-stone-500 dark:text-slate-400">
                  {plan.period}
                </span>
              </div>
              <ul className="mt-6 flex-1 space-y-3">
                {plan.features.map((f) => (
                  <li
                    key={f}
                    className="flex items-start gap-2.5 text-sm text-stone-600 dark:text-slate-300"
                  >
                    <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-500" />
                    {f}
                  </li>
                ))}
              </ul>
              <button
                className={`mt-8 w-full rounded-xl py-3 text-sm font-semibold transition-all ${
                  plan.featured
                    ? "bg-gradient-to-r from-blue-700 to-indigo-700 text-white hover:shadow-glow-sm hover:brightness-110 dark:from-blue-600 dark:to-indigo-600"
                    : "border border-stone-300 text-stone-700 hover:border-blue-400 hover:bg-cream-100 dark:border-slate-700 dark:text-slate-300 dark:hover:border-blue-600 dark:hover:bg-blue-950/40"
                }`}
              >
                {plan.cta}
              </button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
