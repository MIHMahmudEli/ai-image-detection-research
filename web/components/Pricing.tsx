import { CheckCircle2, Star } from "lucide-react";
import { PRICING_PLANS } from "@/lib/content";

export default function Pricing() {
  return (
    <section id="pricing" className="bg-white py-20">
      <div className="mx-auto max-w-5xl px-4">
        <p className="text-center text-xs font-semibold uppercase tracking-widest text-blue-600">
          Pricing
        </p>
        <h2 className="mt-2 text-center text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">
          Simple, Transparent Pricing
        </h2>
        <p className="mx-auto mt-3 max-w-lg text-center text-slate-500">
          Start free, upgrade as you grow
        </p>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {PRICING_PLANS.map((plan) => (
            <div
              key={plan.name}
              className={`relative flex flex-col rounded-2xl p-7 ${
                plan.featured
                  ? "border-2 border-blue-600 bg-white shadow-xl shadow-blue-100"
                  : "border border-slate-200 bg-white shadow-sm"
              }`}
            >
              {plan.featured && (
                <div className="absolute -top-3.5 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full bg-blue-600 px-4 py-1 text-xs font-semibold text-white shadow-sm">
                  <Star className="h-3 w-3 fill-current" />
                  Most Popular
                </div>
              )}
              <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                {plan.name}
              </h3>
              <div className="mt-2 flex items-baseline gap-1">
                <span className="text-4xl font-bold tracking-tight text-slate-900">
                  {plan.price}
                </span>
                <span className="text-sm text-slate-500">{plan.period}</span>
              </div>
              <ul className="mt-6 flex-1 space-y-3">
                {plan.features.map((f) => (
                  <li
                    key={f}
                    className="flex items-start gap-2.5 text-sm text-slate-600"
                  >
                    <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-500" />
                    {f}
                  </li>
                ))}
              </ul>
              <button
                className={`mt-8 w-full rounded-xl py-3 text-sm font-semibold transition-colors ${
                  plan.featured
                    ? "bg-blue-600 text-white hover:bg-blue-700"
                    : "border border-slate-300 text-slate-700 hover:bg-slate-50"
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
