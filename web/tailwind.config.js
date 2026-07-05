/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // warm "vintage paper" neutrals for light mode
        cream: {
          50: "#FBF9F4",
          100: "#F6F1E7",
          200: "#EDE5D4",
          300: "#E0D5BE",
        },
        // vintage classic dark: slate remapped to warm espresso/aged-leather
        // neutrals so every dark: surface matches the cream light mode
        slate: {
          50: "#F5EFE3",
          100: "#ECE4D4",
          200: "#DCD1BB",
          300: "#C4B69C",
          400: "#A2937A",
          500: "#80725C",
          600: "#5F5343",
          700: "#463D31",
          800: "#332C23",
          900: "#251F18",
          950: "#17130E",
        },
        primary: {
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a8a",
        },
      },
      boxShadow: {
        "glow-sm": "0 0 20px -5px rgba(59, 130, 246, 0.35)",
        glow: "0 0 40px -10px rgba(59, 130, 246, 0.45)",
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.5s ease-out",
        shimmer: "shimmer 8s ease-in-out infinite",
        float: "float 6s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-8px)" },
        },
      },
    },
  },
  plugins: [],
};
