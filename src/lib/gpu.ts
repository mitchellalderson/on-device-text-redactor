import { TrainedModel } from "./trained-model";
import { BaseModel } from "./base-model";

export type ModelVariant = "trained" | "base";
export type StatusCallback = (message: string) => void;
export type TokenCallback = (token: string) => void;

export class PhiFirewall {
  private impl: TrainedModel | BaseModel | null = null;
  private _variant: ModelVariant | null = null;

  get variant(): ModelVariant | null {
    return this._variant;
  }

  async init(
    variant: ModelVariant,
    onStatus?: StatusCallback,
  ): Promise<void> {
    this._variant = variant;

    onStatus?.("Checking WebGPU...");
    if (!navigator.gpu) {
      throw new Error(
        "WebGPU not available. Enable at chrome://flags/#enable-unsafe-webgpu",
      );
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error(
        "WebGPU adapter not found. Check chrome://gpu for status.",
      );
    }

    if (variant === "trained") {
      this.impl = new TrainedModel();
    } else {
      this.impl = new BaseModel();
    }

    await this.impl.init(onStatus);
  }

  async redact(
    text: string,
    onToken?: TokenCallback,
    onStatus?: StatusCallback,
  ): Promise<string> {
    if (!this.impl) {
      throw new Error("Model not initialized. Call init() first.");
    }
    return this.impl.redact(text, onToken, onStatus);
  }

  static checkGpuSupport(): { supported: boolean; message: string } {
    if (!navigator.gpu) {
      return {
        supported: false,
        message:
          "WebGPU not available. Enable at chrome://flags/#enable-unsafe-webgpu",
      };
    }
    return { supported: true, message: "WebGPU available" };
  }
}
