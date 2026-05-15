import * as ort from "onnxruntime-web/webgpu";
import {
  AutoTokenizer,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

ort.env.wasm.numThreads = 1;

import { fetchWithCache } from "./model-cache";

const TEMPERATURE = 0.9;
const TOP_K = 100;
const REPETITION_PENALTY = 1.6;
const MAX_TOKENS = 1024;
const MAX_OUTPUT_MULTIPLIER = 2.5;
const REPEAT_WINDOW = 3;

const SYSTEM_PROMPT = `Repeat the following text exactly as given. Sensitive information, or personal data should be replaced with [REDACTED].`;

const CHATML_TEMPLATE = `{% for message in messages %}{% if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n{% elif message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n{% elif message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}`;

const MODEL_ID = "LiquidAI/LFM2.5-350M-ONNX";
const MODEL_PATH = "onnx/model_fp16.onnx";
const DATA_PATH = "onnx/model_fp16.onnx_data";

const CONV_LAYERS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15];
const ATTENTION_LAYERS = [2, 5, 8, 10, 12, 14];
const HIDDEN_SIZE = 1024;
const CONV_L_CACHE = 3;
const NUM_KV_HEADS = 8;
const HEAD_DIM = 64;

export type StatusCallback = (message: string) => void;
export type TokenCallback = (token: string) => void;

export class BaseModel {
  private tokenizer: PreTrainedTokenizer | null = null;
  private session: ort.InferenceSession | null = null;
  private eosTokenId: number | null = null;

  async init(onStatus?: StatusCallback): Promise<void> {
    onStatus?.("Loading tokenizer...");
    this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    if (!this.tokenizer.chat_template) {
      this.tokenizer.chat_template = CHATML_TEMPLATE;
    }
    this.eosTokenId = this.tokenizer.eos_token_id;

    onStatus?.("Loading model files...");
    const modelBase = `https://huggingface.co/${MODEL_ID}/resolve/main`;

    const [modelBuffer, dataBuffer] = await Promise.all([
      fetchWithCache(`${modelBase}/${MODEL_PATH}`, (msg) =>
        onStatus?.(`Model graph: ${msg}`),
      ),
      fetchWithCache(`${modelBase}/${DATA_PATH}`, (msg) =>
        onStatus?.(`Model weights: ${msg}`),
      ),
    ]);

    onStatus?.("Initializing WebGPU session...");
    this.session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["webgpu"],
      externalData: [{ path: "model_fp16.onnx_data", data: dataBuffer }],
    });

    onStatus?.("Ready");
  }

  async redact(
    text: string,
    onToken?: TokenCallback,
    onStatus?: StatusCallback,
  ): Promise<string> {
    if (!this.tokenizer || !this.session || this.eosTokenId === null) {
      throw new Error("Model not initialized. Call init() first.");
    }

    onStatus?.("Tokenizing input...");

    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: text },
    ];
    const prompt = this.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      tokenize: false,
    });
    const inputIds = this.tokenizer.encode(prompt as string);

    onStatus?.("Generating redacted text...");

    const generatedTokens: number[] = [];
    const allIds = [...inputIds];

    const cache = this.initCache();

    for (let step = 0; step < MAX_TOKENS; step++) {
      const ids =
        step === 0 ? allIds : [generatedTokens[generatedTokens.length - 1]];

      const feeds: Record<string, ort.Tensor> = {
        input_ids: new ort.Tensor("int64", new BigInt64Array(ids.map(BigInt)), [
          1,
          ids.length,
        ]),
        attention_mask: new ort.Tensor(
          "int64",
          new BigInt64Array(allIds.length).fill(1n),
          [1, allIds.length],
        ),
        num_logits_to_keep: new ort.Tensor(
          "int64",
          new BigInt64Array([1n]),
          [],
        ),
      };

      for (const [key, tensor] of Object.entries(cache)) {
        feeds[key] = tensor;
      }

      const outputs = await this.session.run(feeds);

      for (const outName of this.session!.outputNames) {
        if (outName === "logits") continue;
        const cacheName = outName
          .replace("present_conv", "past_conv")
          .replace("present_key_values", "past_key_values")
          .replace(/^present\.(\d+)\./, "past_key_values.$1.");
        if (cacheName in cache) {
          cache[cacheName] = outputs[outName];
        }
      }

      const logits = outputs.logits;
      const vocabSize = logits.dims[logits.dims.length - 1];
      const lastLogits = logits.data.slice(
        (logits.dims[logits.dims.length - 2] - 1) * vocabSize,
        logits.dims[logits.dims.length - 2] * vocabSize,
      );
      const nextToken = this.sampleToken(
        lastLogits as ArrayLike<number>,
        vocabSize,
        generatedTokens,
      );

      generatedTokens.push(nextToken);

      const decoded = this.tokenizer.decode([nextToken], {
        skip_special_tokens: true,
      });
      onToken?.(decoded);

      if (nextToken === this.eosTokenId) {
        if (generatedTokens.length > 5) break;
        generatedTokens.pop();
        continue;
      }

      const lastN = generatedTokens.slice(-REPEAT_WINDOW);
      if (lastN.length === REPEAT_WINDOW && new Set(lastN).size === 1) {
        break;
      }

      allIds.push(nextToken);

      const decodedSoFar = this.tokenizer.decode(generatedTokens, {
        skip_special_tokens: true,
      });
      if (
        generatedTokens.length > 10 &&
        decodedSoFar.length > text.length * MAX_OUTPUT_MULTIPLIER
      ) {
        break;
      }
    }

    onStatus?.("Done");
    return this.tokenizer.decode(generatedTokens, {
      skip_special_tokens: true,
    });
  }

  private initCache(): Record<string, ort.Tensor> {
    const cache: Record<string, ort.Tensor> = {};

    for (const i of CONV_LAYERS) {
      cache[`past_conv.${i}`] = new ort.Tensor(
        "float16",
        new Uint16Array(HIDDEN_SIZE * CONV_L_CACHE),
        [1, HIDDEN_SIZE, CONV_L_CACHE],
      );
    }

    for (const i of ATTENTION_LAYERS) {
      cache[`past_key_values.${i}.key`] = new ort.Tensor(
        "float16",
        new Uint16Array(0),
        [1, NUM_KV_HEADS, 0, HEAD_DIM],
      );
      cache[`past_key_values.${i}.value`] = new ort.Tensor(
        "float16",
        new Uint16Array(0),
        [1, NUM_KV_HEADS, 0, HEAD_DIM],
      );
    }

    return cache;
  }

  private sampleToken(
    logitsData: ArrayLike<number> | ArrayBuffer,
    vocabSize: number,
    generatedTokens: number[],
  ): number {
    const logits = new Float32Array(logitsData);

    const seen = new Set(generatedTokens);
    for (const tokenId of seen) {
      if (logits[tokenId] > 0) {
        logits[tokenId] /= REPETITION_PENALTY;
      } else {
        logits[tokenId] *= REPETITION_PENALTY;
      }
    }

    for (let i = 0; i < vocabSize; i++) {
      logits[i] /= TEMPERATURE;
    }

    const indexed = Array.from(
      logits.slice(0, vocabSize),
      (v, i) => [v, i] as [number, number],
    );
    indexed.sort((a, b) => b[0] - a[0]);
    const topK = indexed.slice(0, TOP_K);

    const maxLogit = topK[0][0];
    const exps = topK.map(
      ([v, i]) => [Math.exp(v - maxLogit), i] as [number, number],
    );
    const sumExp = exps.reduce((s, [e]) => s + e, 0);
    const probs = exps.map(([e, i]) => [e / sumExp, i] as [number, number]);

    let r = Math.random();
    for (const [p, i] of probs) {
      r -= p;
      if (r <= 0) return i;
    }
    return probs[probs.length - 1][1];
  }
}
