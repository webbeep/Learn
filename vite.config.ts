import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI; /api/* is proxied to `wrangler pages dev` on :8788
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8788" } },
});
