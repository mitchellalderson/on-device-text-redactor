<script lang="ts">
    import { PhiFirewall, type ModelVariant } from "./lib/gpu";

    type AppStatus = "idle" | "loading" | "ready" | "generating" | "error";

    let inputText = $state("");
    let outputText = $state("");
    let status: AppStatus = $state("idle") as AppStatus;
    let statusMessage = $state("");
    let gpuError = $state("");
    let firewall: PhiFirewall | null = null;
    let selectedVariant: ModelVariant = $state(
        (localStorage.getItem("phi-model-variant") as ModelVariant) || "trained"
    );

    const EXAMPLE_TEXT = `Patient John Smith (DOB: 03/15/1980, SSN: 123-45-6789) was admitted on January 5, 2024. Contact: (555) 867-5309, john.smith@email.com. Address: 742 Evergreen Terrace, Springfield, IL 62704. Medical Record Number: MRN-2024-001234. The patient's insurance ID is BCBS-9988776655. Primary physician: Dr. Emily Chen at Mercy General Hospital.`;

    async function initModel() {
        const gpu = PhiFirewall.checkGpuSupport();
        if (!gpu.supported) {
            gpuError = gpu.message;
            status = "error";
            return;
        }

        status = "loading";
        try {
            if (!firewall) firewall = new PhiFirewall();
            await firewall.init(selectedVariant, (msg) => {
                statusMessage = msg;
            });
            status = "ready";
        } catch (err: any) {
            gpuError = err.message || String(err);
            status = "error";
        }
    }

    async function switchModel(variant: ModelVariant) {
        if (variant === selectedVariant) return;
        localStorage.setItem("phi-model-variant", variant);
        location.reload();
    }

    let tokensPerSec = $state<number | null>(null);
    let tokenCount = 0;
    let genStartTime = 0;
    let tpsInterval: ReturnType<typeof setInterval> | null = null;

    async function handleRedact() {
        if (!firewall || !inputText.trim() || status === "generating") return;

        outputText = "";
        status = "generating";
        tokenCount = 0;
        tokensPerSec = null;
        genStartTime = performance.now();

        tpsInterval = setInterval(() => {
            const elapsed = (performance.now() - genStartTime) / 1000;
            if (elapsed > 0 && tokenCount > 0) {
                tokensPerSec = Math.round((tokenCount / elapsed) * 10) / 10;
            }
        }, 200);

        try {
            await firewall.redact(
                inputText,
                (token) => {
                    outputText += token;
                    tokenCount++;
                },
                (msg) => {
                    statusMessage = msg;
                },
            );
        } catch (err: any) {
            outputText = `Error: ${err.message || String(err)}`;
        }

        if (tpsInterval) clearInterval(tpsInterval);
        const elapsed = (performance.now() - genStartTime) / 1000;
        if (elapsed > 0 && tokenCount > 0) {
            tokensPerSec = Math.round((tokenCount / elapsed) * 10) / 10;
        }
        status = "ready";
    }

    function handleClear() {
        inputText = "";
        outputText = "";
        tokensPerSec = null;
    }

    $effect(() => {
        initModel();
    });
</script>

<main>
    <header>
        <h1>PHI Firewall</h1>
        <p class="subtitle">
            Locally redact Protected Health Information using WebGPU
        </p>
    </header>

    {#if gpuError}
        <div class="error-banner">
            <strong>WebGPU Error:</strong>
            {gpuError}
        </div>
    {/if}

    {#if status === "loading"}
        <div class="loading-overlay">
            <div class="loading-content">
                <div class="loading-label">{statusMessage || "Initializing..."}</div>
                <div class="progress-bar">
                    <div class="progress-fill indeterminate"></div>
                </div>
                <div class="progress-hint">
                    {#if statusMessage?.includes("Downloading")}
                        This may take a few minutes on first load
                    {:else if statusMessage?.includes("Initializing")}
                        Preparing WebGPU session...
                    {:else}
                        Setting up...
                    {/if}
                </div>
            </div>
        </div>
    {:else}
        <div class="container">
        <div class="model-toggle">
            <button
                class="toggle-btn"
                class:active={selectedVariant === "base"}
                onclick={() => switchModel("base")}
                disabled={status === "loading" || status === "generating"}
            >
                Base Model
            </button>
            <button
                class="toggle-btn"
                class:active={selectedVariant === "trained"}
                onclick={() => switchModel("trained")}
                disabled={status === "loading" || status === "generating"}
            >
                Fine-Tuned
            </button>
        </div>

        <div class="input-section">
            <div class="section-header">
                <label for="input">Input Text</label>
                <div class="actions">
                    <button
                        class="btn-small"
                        onclick={() => (inputText = EXAMPLE_TEXT)}
                    >
                        Load Example
                    </button>
                    <button
                        class="btn-small"
                        onclick={handleClear}
                        disabled={!inputText}
                    >
                        Clear
                    </button>
                </div>
            </div>
            <textarea
                id="input"
                bind:value={inputText}
                placeholder="Paste text containing PHI here..."
                disabled={status === "generating"}
            ></textarea>
        </div>

        <div class="controls">
            <button
                class="btn-primary"
                onclick={handleRedact}
                disabled={status !== "ready" || !inputText.trim()}
            >
                {#if status === "loading"}
                    Loading Model...
                {:else if status === "generating"}
                    Redacting...
                {:else}
                    Redact PHI
                {/if}
            </button>
        </div>

        <div class="output-section">
            <div class="section-header">
                <span class="label">Redacted Output</span>
                {#if tokensPerSec !== null}
                    <span class="tps-badge">{tokensPerSec} tok/s</span>
                {/if}
            </div>
            <div class="output-box" class:active={status === "generating"}>
                {#if outputText}
                    {outputText}<span
                        class="cursor"
                        class:blink={status === "generating"}
                    ></span>
                {:else if status === "generating"}
                    <span class="placeholder">Generating...</span>
                {:else}
                    <span class="placeholder"
                        >Redacted text will appear here...</span
                    >
                {/if}
            </div>
        </div>
        </div>
    {/if}

    <footer>
        <span class="status">
            {#if statusMessage}
                {statusMessage}
            {:else if status === "idle"}
                Initializing...
            {/if}
        </span>
        <span class="badge">WebGPU + LFM2.5-350M</span>
    </footer>
</main>
