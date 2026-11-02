import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Relative base so the build works on GitHub Pages under /th-simulator/ and as a static artifact.
  base: "./",
  plugins: [react()],
  worker: { format: "es" },
});
