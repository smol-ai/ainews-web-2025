---
id: MjAyNS0x
title: not much happened today
date: '2026-09-11T05:44:39.731046Z'
description: >-
  **Jev**, a non-generative decision model, emerged as a fast “**System 1**”
  complement to LLMs with **~400x lower scoring cost**, gaining traction for
  routing, citation selection, and legal ops decisions. Open reproductions like
  **Bespoke Nimble** (a LoRA fine-tune of **Qwen3.5-9B**) and **Kev-0.5B**
  (based on **Qwen2.5-0.5B**) appeared, showing strong performance and local
  usability. Early integrations focused on browser and computer workflows,
  emphasizing a **workflow control-plane** rather than chatbot use. In agent
  tooling, **Claude Code v2.1.277** adopted **AGENTS.md** as a cross-tool
  standard, improving configuration and reducing shim files. Harness design in
  coding agents was highlighted as a key factor in performance and cost
  efficiency, with simple tool sets reaching the **Pareto frontier** on
  benchmarks, emphasizing the importance of human/agent interface design.
companies:
  - latent-space
  - braintrust
  - qwen
  - claude
  - langchain
models:
  - jev
  - qwen3.5-9b
  - qwen2.5-0.5b
  - claude-code-v2.1.277
topics:
  - decision-models
  - routing
  - on-device-ai
  - fine-tuning
  - synthetic-data
  - workflow-control-plane
  - agent-tooling
  - coding-agents
  - benchmarking
  - harness-design
  - cost-efficiency
  - human-agent-interface
people:
  - ankrgyl
  - gabepereyra
  - hxiao
  - signulll
  - madiator
  - jaredpalmer
  - mparakhin
  - abacaj
  - levie
  - ndrezn
  - cline
  - hwchase17
  - trq212
  - simonw
  - pidotdev
  - _akhaliq
  - dexhorthy
---


**a quiet day.**

> AI News for 9/11/2026-9/14/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Decision Models, Routing, and the “Jev” Wave**

- **Discriminative models broke out as a new systems primitive**: The biggest technical conversation was around **Jev**, a non-generative decision model being positioned as a fast “**System 1**” complement to LLMs. [@ankrgyl](https://x.com/ankrgyl/status/2100978416434786420) said it is now available as an eval model in Braintrust with **~400x lower scoring cost** versus prior setups, while [@gabepereyra](https://x.com/gabepereyra/status/2100990093691691382) highlighted calibrated-probability use cases like routing, citation selection, escalation, and legal ops decisions. The more architectural take came from [@hxiao](https://x.com/hxiao/status/2101001002816327867), who argued Jev could pull tool calling, routing, and MCP-style decisions back from small generative LMs toward discriminative models; [@signulll](https://x.com/signulll/status/2101062047350096040) pushed the same idea further, framing this class as a near-zero-marginal-cost, **on-device judgment layer** for notifications, UI adaptation, and sensor-driven decisions.

- **Open reproductions and ecosystem clones appeared immediately**: [@madiator](https://x.com/madiator/status/2100990591215783946) released **Bespoke Nimble**, an “open Jev” recipe built from a **LoRA fine-tune of Qwen3.5-9B** using **synthetic contrastive data curation** and constrained decoding. On its curated eval, the base Qwen improved from **66% to 90%**, versus **93% for Jev**, with a reported **100ms on H100** and local usability. At the smaller end, [@jaredpalmer](https://x.com/jaredpalmer/status/2101028325472841920) released **Kev-0.5B**, a tiny Jev-like model based on **Qwen2.5-0.5B** that can run on a MacBook Pro. The reaction split roughly along prior experience: [@MParakhin](https://x.com/MParakhin/status/2101036299721347073) noted post-ChatGPT users treated it like a revelation, while pre-GPT ML people were more puzzled by the hype. The substantive question raised by [@abacaj](https://x.com/abacaj/status/2101048462661845099) is the right one: a lot of demos emphasized **speed** more than **quality**, and there is still no standard benchmark for this category.

- **The first compelling integrations were in browser/computer-use workflows**: [@levie](https://x.com/levie/status/2101007708044574906) demoed Jev classifying Box incident reports into escalation paths; [@ndrezn](https://x.com/ndrezn/status/2101046780989215005) showed browser use with LangChain + Jev and found it strong on tasks like the Wikipedia game and structured “folding laundry” workflows; [@cline](https://x.com/cline/status/2101056078872256935) shipped a plugin giving Jev a browser in Cline. [@hwchase17](https://x.com/hwchase17/status/2101054310037790814) explicitly called browser use the best Jev application he had seen so far. Net: this looks less like a chatbot story than a **workflow control-plane** story.

**Agent Tooling, Coding Harnesses, and Claude Code Standards**

- **AGENTS.md gained real momentum as a cross-tool convention**: The highest-signal product update here was [@trq212](https://x.com/trq212/status/2101009392611278961) announcing that **Claude Code v2.1.277** now checks for **AGENTS.md** when no **CLAUDE.md** is present, with config-level toggle support. That effectively acknowledges AGENTS.md as an emerging standard rather than a one-tool convention, and [@simonw](https://x.com/simonw/status/2101025043098812807) immediately noted the practical payoff: fewer shim files that just point one format to the other.

- **Harness design is becoming a first-class variable in coding-agent performance and cost**: [@pidotdev](https://x.com/pidotdev/status/2100935860413673605) highlighted the **Harness Tax** analysis showing that a simple tool set—**read, write, edit, bash**—can reach the **Pareto frontier** on benchmark performance while reducing unnecessary spending. Relatedly, [@_akhaliq](https://x.com/_akhaliq/status/2101020560964866103) pointed to the paper *An Empirical Study of Harness Design for Coding Agents*, underscoring that benchmark outcomes are increasingly shaped by **harness structure**, context setup, turn budgets, and tool affordances rather than just the base model. This is consistent with [@dexhorthy](https://x.com/dexhorthy/status/2100900279021363244)’s “software factory” argument that teams still need to **read the code** and deliberately design the human/agent interface.

- **Model choice in software systems is bifurcating**: Several practitioners described a split between “frontier for planning, cheap for execution.” [@TheAhmadOsman](https://x.com/TheAhmadOsman/status/2101061704444682682) summarized one stack as **GPT 5.6 Sol XHigh** for planning, **GLM 5.3 Flash** for implementation, and **DeepSeek V4.1 Flash** for other tasks. [@kylebrussell](https://x.com/kylebrussell/status/2101028521044812109) reported an internal knowledge-base pipeline moving from **Opus → Sonnet → GLM 5.2 → GLM 5.3 Flash**, cutting spend by roughly **two orders of magnitude** since spring. Meanwhile [@theo](https://x.com/theo/status/2101062549722841452) argued that in real-world coding the payoff from stronger models like **Fable** and **Astra** is not just code quality, but a subtler productivity gain in execution and iteration.

**Benchmarks, Recursive Self-Improvement, and Math Capability**

- **RSI discussion got more precise about what is actually “recursive”**: [@TheTuringPost](https://x.com/TheTuringPost/status/2100769692877303863) offered a useful taxonomy: AI improving code or training methods is not, by itself, fully recursive if the surrounding improvement loop remains fixed. The key threshold is when AI can modify not just model internals, but **search strategy, experience generation, research tooling, and the improvement process itself**. That framing links well with [@HuaxiuYaoML](https://x.com/HuaxiuYaoML/status/2100959310624825688)’s **RSI-Exam** update, where **GPT-6-astra** remains #1 at **0.5126**, with **Fable 5.1** entering at #2 with **0.4813**, and no model yet reaching the frontier-calibrated reference.

- **Math benchmarks continued to fall to frontier models, but interpretation remains nuanced**: [@EpochAIResearch](https://x.com/EpochAIResearch/status/2100986494873989227) reported that another **FrontierMath open problem** was solved in an interactive session with **GPT-6 Astra**. Separately, [@SAIRfoundation](https://x.com/SAIRfoundation/status/2100976455123620089) launched **Open Math Model**, pitching open models and tools for mathematics shaped by the research community. Against the “verifiability explains math strength” narrative, [@steve47285](https://x.com/steve47285/status/2100998663254225391) shared an argument that **pretraining data**, not merely verifiable reward structure, is the main reason LLMs are so good at math and coding. The meta-point from [@sarahcat21](https://x.com/sarahcat21/status/2101023982258712725) is worth keeping: we need not just better benchmarks, but better **benchmark maintenance and audit tooling**.

- **Computer-use benchmarks are still far from saturation**: [@ValsAI](https://x.com/ValsAI/status/2101014465781318072) launched **CUA-Bench**, testing real-time keyboard/mouse use across **6 games** (with **3 kept private**) as a proxy for difficult human-easy tasks. Follow-up numbers from [@ValsAI](https://x.com/ValsAI/status/2101014471586243036) suggest this remains genuinely hard: **all frontier models score below 20%**. In parallel, [@trycua](https://x.com/trycua/status/2101014004927729737) open-sourced **CUA-S1-FORMS**, the first in a family of small “System One” computer-use models. The direction is notable: real-time action loops, video-grounded adaptation, and continuous learning, not just text-only planning.

**Infra, Training Systems, and Model Architecture**

- **Long-context and large-scale training infrastructure remain active optimization fronts**: [@Azaliamirh](https://x.com/Azaliamirh/status/2101020422926135665) released **Turbo-dLLM**, an open-source library for training diffusion LLMs at scale, reporting **2.48x speedup at 512K** context and **7.59x at 1M** context on **8x H100s** via **Context-Sharded Block Parallelism**. That aligns with practitioner attention on million-token regimes: [@andrew_n_carr](https://x.com/andrew_n_carr/status/2101024604080791894) flagged a sharp quality increase in DeepSeek V4.1 Flash after context extension to **1M tokens**, arguing that **agents are context hungry**.

- **Architecture taxonomy debates are still alive**: [@ahatamiz1](https://x.com/ahatamiz1/status/2101005845685493794) argued that the field is overusing **SSM** as a label for any linear model. His proposal is to use **linear RNNs** as the umbrella term, with SSMs as one sub-family, distinguishing systems like **Mamba2** from the **GDN** family on the basis that GDN behaves more like a gradient step on a local regression loss than a discretized ODE. For engineers tracking sequence-model alternatives to transformers, this is a useful nomenclature cleanup rather than mere pedantry.

- **Edge/local neural program execution also got a notable update**: [@yuntiandeng](https://x.com/yuntiandeng/status/2100975083376275795) described **ProgramAsWeights**, where developers specify an AI function in English, compile it once, and then run a small neural program **locally on CPU with Wi‑Fi off**. The code and models are public. This sits interestingly adjacent to the Jev conversation: both point toward **smaller, specialized, locally runnable inference artifacts** rather than ever-larger universal chat models.

**Robotics, Vision, Audio, and Generative Media**

- **Open robotics data releases were unusually substantive**: [@adamrasb](https://x.com/adamrasb/status/2100991778606440795) announced the full **ABC** release, including code, **400+ hours of sim data on 24 tasks**, and **5,850 labeled policy-evaluation episodes**. In a more detailed companion post, [@redstone_hong](https://x.com/redstone_hong/status/2100995941742629342) described **ABC-130K** as the largest open teleop dataset to date: **3,500 hours**, **130K+ episodes**, **195 tasks**, collected on an **$8K bimanual setup**, with open hardware, training code, sim, and eval. The baseline science included **sim-to-real correlation r = 0.91** on task progress and studies of offline metrics, scaling laws, and conditioning.

- **Astra is showing up across evals and products, especially for vision**: [@skalskip92](https://x.com/skalskip92/status/2101020135142101249) reported **GPT-6 Astra** as the strongest vision model Roboflow has tested across detection, segmentation, box prompting, counting, reasoning, and video. The tradeoff remains material: a “high effort” setting improved detection from **82.1% to 83.6% mAP@50** but roughly doubled per-image cost from **$0.050 to $0.101** and latency from **11s to 32s** ([details](https://x.com/skalskip92/status/2101020166586859789)). Roboflow also integrated Astra into Auto Annotate.

- **Speech and lip-sync saw strong benchmarked releases**: [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2101065575737024844) reported **Grok Voice Transcribe 2.0** reaching **2.7% WER** on streaming final transcripts at **0.49s** after end-of-speech, improving from **3.9%** on its predecessor while keeping pricing at **$0.20/hour streaming** and **$0.10/hour non-streaming**. On the video side, [@fal](https://x.com/fal/status/2101035750548484535) launched **H3 Max Lip Sync**, claiming #1 on both speed and quality in its evals with **11s median generation time**, and [@isidentical](https://x.com/isidentical/status/2101047260247457808) said the model was built by pushing **diffusion RL** into a verifiable lip-sync task.

**AI Safety, Evaluation Governance, and Security**

- **Anthropic’s evaluator-embedding strategy became more concrete—and more controversial**: [@AnthropicAI](https://x.com/AnthropicAI/status/2101039819870937247) announced a partnership with **Accenture** on **independent evaluation of frontier AI**, saying the two organizations expect to invest at least **$1B over five years** to build capacity. This follows broader calls for embedded third-party evaluators with employee-level access. The reaction was mixed to hostile: critics questioned whether a consulting firm is the right vehicle for model red-teaming and safeguard assessment, while [@TransluceAI](https://x.com/TransluceAI/status/2101061642561921146) emphasized that the conditions around independence and meaningful oversight are the real issue.

- **The “rogue agents” / Hugging Face incident continued to drive debate about containment**: [@polynoamial](https://x.com/polynoamial/status/2100998240586137701) clarified that his much-mocked thought experiment was about **coordination between supposedly isolated agents**, not weight exfiltration via thermal sensors, and argued the lesson from the HF incident is to avoid trusting sandbox isolation as a sole defense. [@martin_casado](https://x.com/martin_casado/status/2100795440677732779) made the strongest steelman: covert channels across air gaps are old, throughput can be tiny, and the real takeaway is layered defense rather than sensationalism. At the same time, [@WSJ](https://x.com/WSJ/status/2100945365763600404) and [@jeffjarvis](https://x.com/jeffjarvis/status/2100908522842071147) pushed back on “rogue AI” framing entirely, arguing these events still reduce to **human-configured systems doing what people enabled them to do**.

- **Policy pressure is building around safety laws and operational accountability**: [@TheRundownAI](https://x.com/TheRundownAI/status/2101010229110452712) reported that California Gov. Gavin Newsom signed an executive order convening an expert panel to recommend stronger AI safety laws, including possible **kill switches**, embedded outside monitors, and required safety plans. Meanwhile, [@sayashk](https://x.com/sayashk/status/2101026107747353046) pointed to a mismatch between rhetoric and incentives in AI security, criticizing OpenAI’s reported **$6,500 bug bounty** to a researcher who broke into an internal repo and disclosed it. The common theme across these posts is straightforward: **independent oversight, layered defenses, and security incentives** are moving from abstract governance talk into concrete operational design.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap


### 1. On-Device Structured Automation Models

  - **[Made the horizontal open-source model for Jev with RLCD, and it surpasses all the Jev benchmarks. HF space, benchmark, model, repo](https://www.reddit.com/r/LocalLLaMA/comments/1wjieap/made_the_horizontal_opensource_model_for_jev_with/)** (Activity: 710): **The post announces **Laya**, a `421M`-parameter non-autoregressive decision model for “Jev” style typed-schema decisions, built from a bidirectional ModernBERT-large encoder plus Transformer scoring head and trained with an unofficial RLCD policy-gradient method for calibrated probabilities. The benchmark image, [**“Laya vs. TypeSafe Jev”**](https://i.redd.it/i3znzaqm28qh1.png), reports substantially lower latency (`38.4 ms` vs. Jev’s `400 ms` average) and higher accuracy across overall average, intent routing, moderation, fact checking, and selective gating; the author also links a [HF Space](https://huggingface.co/spaces/convaiinnovations/laya-demo), [model repo](https://huggingface.co/convaiinnovations/laya), and [GitHub](https://github.com/NandhaKishorM/laya).** Comments were sparse but included interest in converting Laya to ONNX and using it as a lightweight agent-safety/router component for model routing, hallucination guarding, bash/firewall checks, prompt-injection scanning, frustration detection, task verification, and commit classification. One commenter joked the project needed a Doom demo, indicating the image itself is a technical benchmark dashboard rather than a meme.

    - A commenter proposed converting the model to `ONNX` and integrating it into an agent harness extension with safety/control modules: **model routing**, hallucination checking against tool context, a destructive bash-command firewall, prompt-injection/data sanitization, frustration-triggered rollback assistance, task completion verification, and a `20 ms` conventional commit classifier. They also described implementation choices such as a **regex floor + high-recall Laya firewall**, sticky routing to avoid model thrashing on short follow-ups, advisory-only hallucination UI badges, cooldown-based frustration detection, and exit-code-aware goal verification.
    - Another commenter benchmarked the model against [**openjev**](https://huggingface.co/AlexWortega/openjev) using a custom [AAR harness](https://github.com/fischerf/aar) and [Laya extension](https://github.com/fischerf/aar-extensions-registry/tree/main/packages/aar-ext-laya) on a laptop with an **11th Gen i7-11800H** and **RTX 3060 6GB**. They reported it as a “great model for CPU” and planned to use it as a **model router**; their extension includes the OpenJev Flappy Bird sample as a regression/test case.

  - **[Cactus Needle 3: A Sliceable 8-29MB Automation Foundation Model That Matches DeepSeek v4 Flash](https://www.reddit.com/r/LocalLLaMA/comments/1wj4qj4/cactus_needle_3_a_sliceable_829mb_automation/)** (Activity: 292): **The [image/GIF](https://i.redd.it/wypizqswz4qh1.gif) appears to be a nearly blank dark dotted-grid presentation background, so **the image itself is non-technical and does not add meaningful model/benchmark information**. The post announces **Cactus Needle 3**, a local automation/function-calling foundation model family with sliceable depths from `2–20` layers / `25–121M` deployable params, CQ2-bit `8–29MB` binaries, CPU-only inference, grammar-constrained JSON/tool calls, and reported Mobile Actions accuracy of `86.0` for the 20-layer model versus **DeepSeek V4 Flash** at `88.4`. The selftext’s technically significant claims are the Simple Attention Network design, Hadamard MLP replacing dense FFNs, hashed n-gram “engram” tables, depth “intelligence ladder,” and local LoRA fine-tuning/export via `cactus-needle`.** Technical commenters were interested but wanted more concrete reference integrations—e.g. a Home Assistant plugin, Android text/voice control app, or calendar/date parsing demo—because the web demo felt less tangible. One commenter also asked whether the model is effectively English-only, aligning with the post’s note that it is English-first and non-English text tokenizes less efficiently.

    - A commenter argued the project needs concrete reference integrations to demonstrate practical automation value, suggesting a **Home Assistant plugin** or simple **Android text/voice action controller** rather than only a web demo. They also proposed using the model as a more flexible NLP layer for a calendar app—parsing dates, times, and event titles from natural language.
    - There was technical ambiguity around the benchmark/reference to **“Apple FM”**: one commenter asked whether this means one of the three generations of **Apple AFM Core 3B** models or the larger **AFM Core Advanced 20B MoE**, which targets devices with `12GB+` RAM. Another commenter asked whether **Cactus Needle 3** is English-only, highlighting missing language-support details.


### 2. Qwen 27B Efficiency Releases

  - **[Ternary Bonsai 2 (27B) just released on Hugging Face. At &lt;6GB in size, it can even run locally in-browser on WebGPU.](https://www.reddit.com/r/LocalLLaMA/comments/1wj6c4l/ternary_bonsai_2_27b_just_released_on_hugging/)** (Activity: 2207): ****Ternary Bonsai 2 (27B)** was released on Hugging Face as a ternary-weight derivative of **Qwen3.8-27B**, preserving the original 27B hybrid-attention causal LM architecture while reducing model size to **<`6GB`**—claimed as **`9×` smaller than FP16** with **`98.2%` retained “intelligence”**. The model is available in the [Prism ML Bonsai 2 collection](https://huggingface.co/collections/prism-ml/bonsai-2), with an in-browser [WebGPU demo](https://huggingface.co/spaces/webml-community/ternary-bonsai-2-webgpu-kernels), implying practical local execution on consumer hardware/browser runtimes.** Commenters were skeptical of the **`98.2%` intelligence retention** claim and wanted independent testing, but there was clear interest in compact, runnable models over extremely large frontier-scale parameter counts that are impractical to self-host.

    - A commenter questioned the release’s claim that the ternary 27B model retains **`98%` intelligence** despite being under **`6GB`**, saying they plan to test whether the compression/quantization claim holds up in practice.
    - One user reported a failure mode in the **WebGPU kernels demo**: when asked Rust programming questions, the model allegedly began **looping/repeating**, suggesting possible inference/runtime issues or degraded instruction-following in the browser demo.

  - **[Thank you :) Swift Qwen 3.8 27B now has 100k+ downloads, is #1 finetune and #9 model on HuggingFace Trending](https://www.reddit.com/r/LocalLLaMA/comments/1wj3s31/thank_you_swift_qwen_38_27b_now_has_100k/)** (Activity: 2100): **The [image](https://i.redd.it/9l5qef9xq4qh1.png) is a promotional milestone graphic from **UkisAI** showing **Swift Qwen 3.8 27B** passing `100k+` Hugging Face downloads, rising from `24k` on Day 2 to `105,493` by Day 6; the post says it is currently the **#1 finetune** and **#9 overall trending model** on Hugging Face. The model is presented as a Qwen-based `27B` finetune focused on reducing pathological overthinking, claiming `-58.3%` token usage and `1.95×` speedup without accuracy loss, with upcoming releases **Swift1.5 Qwen3.8 27B** and **Swift Qwen3.8 Flash Next** plus expanded coding/long-horizon benchmarks.** Comments were mostly supportive but light on technical detail; one user questioned the model’s visibility despite `100k` downloads, while another praised OP’s organic community engagement. A notable request was for an **uncensored** version of the finetune.

    - A user reported converting **Swift-Qwen3.8-27B** to **NInfer V3** and using it as a daily driver with OMP: [CaptainArni/Swift-Qwen3.8-27B-NInfer](https://huggingface.co/CaptainArni/Swift-Qwen3.8-27B-NInfer). They claim it fits the full `262k` context with vision on an **RTX 5090** using `NVFP4` KV cache, achieving roughly `190 tok/s` decode with **DFlash2 K=7** at an `80%` power limit.
    - Another user converted the `NVFP4` quantization to **GGUF** for **llama.cpp** compatibility, publishing it at [HuggingJoost/Swift-Qwen3.8-27B-NVFP4-GGUF](https://huggingface.co/HuggingJoost/Swift-Qwen3.8-27B-NVFP4-GGUF). This is useful for readers wanting to run the model outside the original inference stack using llama.cpp tooling.



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo


