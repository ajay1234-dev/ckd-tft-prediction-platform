/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#1a56db",
        accent: "#0ea5e9",
        warn: "#f97316",
        danger: "#ef4444",
        success: "#22c55e",
        surface: "#ffffff",
        background: "#f8fafc",
        border: "#e2e8f0",
        stage: {
          G1: "#2ecc71",
          G2: "#27ae60",
          G3a: "#f1c40f",
          G3b: "#e67e22",
          G4: "#e74c3c",
          G5: "#8e44ad",
        }
      },
    },
  },
  plugins: [],
}
