import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // The AI service. Proxied in dev so the app talks to one origin and the
      // WebSocket needs no CORS dance; in production the Java backend or the
      // ingress routes /api to the service inside the VPC.
      "/api": {
        target: process.env.TAJWID_API ?? "http://localhost:8100",
        changeOrigin: true,
        ws: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
