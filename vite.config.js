import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths keep both pages working under GitHub Pages.
  base: "./",
  build: {
    outDir: "docs",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        reference: fileURLToPath(new URL("./index.html", import.meta.url)),
        fit: fileURLToPath(new URL("./fit.html", import.meta.url)),
      },
    },
  },
});
