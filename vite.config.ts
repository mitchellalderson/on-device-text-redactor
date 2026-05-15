import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("onnxruntime-web")) {
            return "ort-vendor";
          }
          if (id.includes("@huggingface/transformers")) {
            return "hf-vendor";
          }
        },
      },
    },
  },
});
