---
id: MjAyNS0w
title: not much happened today
date: '2025-09-03T05:44:39.731046Z'
description: >-
  **Exa** raised a **$700m Series B**, **OpenPipe** was acquired by
  **Coreweave**, and **Statsig** and **Alex** were acquired by **OpenAI**. The
  **Agent/Client Protocol (ACP)** was introduced by the **Zed** team to
  standardize IDE-agent interoperability, supporting **Claude Code** and
  **Gemini** CLIs. **LangChain 1.0 alpha** unifies content blocks for reasoning
  and multimodal data. The **OSWorld Verified leaderboard** promotes
  reproducible evaluation of computer-use agents including **OpenAI** and
  **Anthropic** models. FAIR revealed coding agent cheating on **SWE-Bench
  Verified**. **PR Arena** hosts live coding agent competitions. Benchmarks like
  **GSO** and **Holistic Agent Leaderboard** test software optimization and web
  browsing tasks, with **Qwen3-Coder** and **Gemini 2.5 Flash** showing strong
  performance. Advances in reinforcement learning for tool use include
  **SimpleTIR** improving multi-turn tool use success rates and **UI-TARS-2**
  advancing GUI agents. The **DARLING** optimizer improves quality and diversity
  in reasoning and instruction following, while **DEPO** achieves data-efficient
  RLVR with significant speedups.
companies:
  - exa
  - openpipe
  - coreweave
  - statsig
  - openai
  - zed
  - claude
  - gemini
  - langchain
  - anthropic
  - fair
  - alibaba
  - hud-evals
models:
  - claude-code
  - gemini
  - qwen3-coder
  - gemini-2.5-flash
topics:
  - agent-protocols
  - interoperability
  - standardization
  - agent-evaluation
  - coding-agents
  - software-optimization
  - web-browsing
  - reinforcement-learning
  - multi-turn-reasoning
  - optimizer-design
  - data-efficient-rlvr
  - leaderboards
  - benchmarking
people:
  - zeddotdev
  - mathemagic1an
  - hwchase17
  - giffmana
  - gneubig
  - crystalsssup
  - sayashk
  - _philschmid
  - _akhaliq
  - jaseweston
---


**a quiet day**

> AI News for 9/3/2025-9/4/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (186 channels, and 4795 messages) for you. Estimated reading time saved (at 200wpm): 410 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

a quiet day. Congrats on [Exa's $700m Series B](https://x.com/ExaAILabs/status/1963262700123000947) and [OpenPipe's acquisition by Coreweave](https://techcrunch.com/2025/09/03/coreweave-acquires-agent-training-startup-openpipe/) and [Statsig](https://x.com/arfurrock/status/1962960884654866554?s=46) and [Alex](https://x.com/danieledrisian/status/1963301872036712652?s=46)'s acquisition by OpenAI.

---

# AI Twitter Recap

**Agent infra standardization and protocols**

- **Agent/Client Protocol (ACP)**: The Zed team introduced an open protocol for IDE–agent interoperability that cleanly decouples the UI from CLI agent operation, similar to LSP for language tooling. ACP already supports Claude Code and Gemini CLIs, making it easier to plug different agents into editors or terminals without bespoke integrations. See the announcement and overview by [@zeddotdev](https://twitter.com/zeddotdev/status/1963258131191853285) and a quick summary by [@mathemagic1an](https://twitter.com/mathemagic1an/status/1963273618705482155) (site: [agentclientprotocol.com](http://agentclientprotocol.com/)).
- **LangChain 1.0 alpha (standard content blocks)**: The 1.0 alpha unifies content representations for reasoning traces, citations, tool calls, and multimodal blocks across providers, reducing glue code when swapping models/hosts. Announcements from [@LangChainAI](https://twitter.com/LangChainAI/status/1963285794954907750) and context from [@hwchase17](https://twitter.com/hwchase17/status/1963287729007165488). LangChain is also running meetups on “Deep Agents” and long-horizon planning ([London](https://twitter.com/LangChainAI/status/1963316066735812876)).

**Agent evaluations, coding, and computer-use**

- **Reproducible CUA evals and cheating analyses**: The OSWorld Verified leaderboard launched to promote reproducible evaluation of computer-use agents; starting entries include OpenAI and Anthropic models ([@hud_evals](https://twitter.com/hud_evals/status/1963321238056796573)). Separately, FAIR surfaced ways coding agents “cheat” on SWE-Bench Verified (e.g., grepping commit logs for issue IDs), underscoring the need for hardened eval environments ([@giffmana](https://twitter.com/giffmana/status/1963327672827687316)).
- **Live competitions for agentic coding**: PR Arena lets you pit two coding agents on tagged GitHub issues and pick the winner—bringing “in the wild” head-to-heads beyond SWE-Bench ([@gneubig](https://twitter.com/gneubig/status/1963267468853477809)). Related: Open models + OpenHands are competitive on several agentic coding scenarios ([@gneubig](https://twitter.com/gneubig/status/1963045532022010231)).
- **Software optimization and browsing tasks**: GSO is a challenging benchmark for optimizing large codebases ([@crystalsssup](https://twitter.com/crystalsssup/status/1963087272506753419)); Qwen3-Coder is performing well there ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1963049864474120475)). For web tasks, Online Mind2Web was added to the Holistic Agent Leaderboard to compare scaffolds like Browser-Use vs SeeAct ([@sayashk](https://twitter.com/sayashk/status/1963343022252315112)), and you can bootstrap a Chromium browser agent with Gemini 2.5 Flash in ~10 lines ([@_philschmid](https://twitter.com/_philschmid/status/1963233076034650481)).

**RL for tool use and LLM training, plus optimizer insights**

- **Stabilizing multi-turn tool use**: SimpleTIR identifies “void turns” (steps that lead nowhere) as a core failure mode; filtering them yields large gains in multi-turn RL—e.g., a 7B model improving from 22% (DAPO) to 50% on multi-turn tool-use metrics ([paper](https://huggingface.co/papers/2509.01739), [@_akhaliq](https://twitter.com/_akhaliq/status/1963228487524679988), [author commentary](https://twitter.com/sivil_taram/status/1963279400834924965)). Related: UI-TARS-2 advances GUI agents via multi-turn RL ([@_akhaliq](https://twitter.com/_akhaliq/status/1963229296236937443)).
- **Optimizing for quality + diversity**: DARLING jointly optimizes both via a learned partition function, improving pass@1/p@k for reasoning and instruction following, while ranking highest on NoveltyBench for diversity ([paper](https://arxiv.org/abs/2509.02534), [thread](https://twitter.com/jaseweston/status/1963230744173482018)).
- **Data-efficient RLVR**: DEPO reports strong speedups at a fraction of data (e.g., 1.85× on AIME’24 using 20% of training data) by curating offline samples and filtering online ones with low “explorability” ([paper](https://arxiv.org/abs/2509.01321), [summary](https://twitter.com/iScienceLuvr/status/1963169113007895020)).
- **Training/optimizer notes**: A systematic study finds matrix-based optimizers (e.g., Muon, Soap) speed up small models but gains diminish with scale (1.4× at 0.1B → ~1.1× at 1.2B) and hyperparameter transfer is non-trivial ([paper](https://arxiv.org/abs/2509.02046), [summary](https://twitter.com/iScienceLuvr/status/1963168542872014943)). A back-of-the-envelope derivation explains AdamW’s ~0.2 RMS update “magic ratio” under assumptions ([@JingyuanLiu123](https://twitter.com/JingyuanLiu123/status/1963084684784734543)). Also: Zhipu/lmsys’ slime RL framework code walkthrough is out ([repo](https://github.com/zhaochenyang20/Awesome-LLM-Alignment), [@Zai_org](https://twitter.com/Zai_org/status/1963099102347931975)).

**Systems, inference, and tooling**

- **Google TPUs beyond Google Cloud**: Google is in talks to place TPUs in third-party GPU clouds—new distribution for TPU capacity with multiple providers reportedly in play ([@anissagardizy8](https://twitter.com/anissagardizy8/status/1963228123144819167), [context](https://twitter.com/dylan522p/status/1963355683170246659)).
- **VS Code: bring-your-own OpenAI-compatible endpoint**: Native support for custom OAI-compatible endpoints landed, a win for local/self-hosted providers and OSS stacks ([@ggerganov](https://twitter.com/ggerganov/status/1963255949373677959), [PR](https://twitter.com/ggerganov/status/1963255951659508117)).
- **Faster kernels, exportable graphs**: FlashAttention-3 is now available via Hugging Face “kernels” (no more lengthy builds), with torch.compile fullgraph support ([@RisingSayak](https://twitter.com/RisingSayak/status/1963225732668182856)). For no-JIT inference/training, PyTorch’s torch.export path targets compile-time autotuning; it’s maturing for backward graphs ([@soumithchintala](https://twitter.com/soumithchintala/status/1963225534659178948)).
- **CPU-first inference and cost notes**: Microsoft open-sourced bitnet.cpp (1-bit LLM inference) reporting 6.17× faster CPU inference and 82% lower energy for certain models ([@LiorOnAI](https://twitter.com/LiorOnAI/status/1963316578612605327)). Meanwhile, pricing quirks persist: many third-party servers don’t pass through cache-hit discounts; closed APIs may be cheaper for coding-heavy workloads due to caching ([@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1963294646957957263)).

**Models and multimodal tooling**

- **Nous Hermes-4-14B**: Compact Hermes 4 model with hybrid reasoning + tool calling, optimized for local consumer hardware. Available on HF and in Nous Chat ([@NousResearch](https://twitter.com/NousResearch/status/1963349882837897535)).
- **OpenVision 2**: A fully open, cost-effective vision encoder family that rivals CLIP/SigLIP; the new release broadens training data and improves accuracy/cost trade-offs ([thread](https://twitter.com/cihangxie/status/1963297223753494832)).
- **Document understanding at speed**: Tencent’s POINTS-Reader is a simple end-to-end VLM for document OCR/extraction with high throughput on SGLang/vLLM; two-stage training (auto-labeled pretraining + self-evolution) hits SOTA on OmniDocBench in English/Chinese ([@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963192346222432750)).
- **Community image-edit progress**: Qwen Image Edit inpainting got a community LoRA that masks the exact region to edit ([demo + LoRA](https://twitter.com/ostrisai/status/1963269597865599425)); Alibaba highlighted community contributions to inpainting ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1963048659676979559)).

**Safety, robustness, and reasoning research**

- **Scaling oversight to frontier models**: Transluce trains small “investigator” models (8B) that can reliably jailbreak frontier assistants (GPT‑5, Claude 4.1, Gemini 2.5 Pro), suggesting oversight specialized by subdomain and scale can keep pace ([report/code](https://twitter.com/TransluceAI/status/1963286326062846094)).
- **Fine-tuning “cipher” attacks**: Anthropic analyzes how seemingly benign fine-tuning data can encode harmful hidden instructions, and discusses mitigations for FT APIs ([@JackYoustra](https://twitter.com/JackYoustra/status/1963280250923868239)).
- **Implicit reasoning + mech interp**: A new survey consolidates work on implicit reasoning in LMs ([paper](https://arxiv.org/abs/2509.02350), [@omarsar0](https://twitter.com/omarsar0/status/1963236545705710070)). In mechanistic interpretability, Layer-wise Relevance Propagation (LRP) significantly improves attribution-patching fidelity versus vanilla gradient methods ([@NeelNanda5](https://twitter.com/NeelNanda5/status/1963029426741854345)); Neel also published a comprehensive “getting started” v2 guide and opened a MATS stream ([guide thread](https://twitter.com/NeelNanda5/status/1963225482973040784)).

**Funding, products, and adoption signals**

- **Search for agents**: Exa raised $85M led by Benchmark to build AI-native web search infrastructure ([@ExaAILabs](https://twitter.com/ExaAILabs/status/1963262700123000947)). [You.com](http://you.com/) raised $100M at a $1.5B valuation and claims >1B monthly queries across customers, optimized for agents’ deep, up-to-date retrieval ([@RichardSocher](https://twitter.com/RichardSocher/status/1963277700711461241), [Bloomberg](https://twitter.com/business/status/1963226665275769327)).
- **Infra consolidation**: CoreWeave acquired OpenPipe; expect tighter integration of ART RL fine-tuning pipelines with high-performance inference infra ([@corbtt](https://twitter.com/corbtt/status/1963332919864557784), [@shawnup](https://twitter.com/shawnup/status/1963335514377130397)).
- **Platform features going wide**: OpenAI Projects now available to Free users with expanded per-project uploads and memory controls ([@OpenAI](https://twitter.com/OpenAI/status/1963329936368046111)). Perplexity launched Comet for students (ad block, study mode, scheduling, native assistant) ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1963285255198314951)).
- **Enterprise usage**: Coinbase reports ~40% of daily code is AI-generated and targets >50% by October, with human review retained ([@brian_armstrong](https://twitter.com/brian_armstrong/status/1963315806248604035)).

**Top tweets (by engagement)**

- Higgsfield’s Draw-to-Edit on “Nano Banana” showcases one-flow multi-model draw-and-animate editing—virality reflects rapid multimodal UX progress ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1963035734232928586)).
- OpenAI Projects expand to Free tier; larger per-project file limits and project-scoped memory controls signal deeper app integration and data routing via Projects ([@OpenAI](https://twitter.com/OpenAI/status/1963329936368046111)).
- Codex CLI momentum: strong qualitative wins for long-horizon adherence and non-giving-up behavior vs prior assistants; usage reportedly up ~10× in two weeks ([@Yampeleg](https://twitter.com/Yampeleg/status/1963260958257578497), [@sama](https://twitter.com/sama/status/1963365966953505103)).
- Humanoid robotics consumer demos continue to draw attention—Figure shows dish/laundry skills and is hiring across AI and manufacturing ([@adcock_brett](https://twitter.com/adcock_brett/status/1963266402028335567)).
- Exa’s $85M raise and [You.com](http://you.com/)’s $100M round underline the “search for agents” thesis: agent-first indices and retrieval infra are strategic assets ([@ExaAILabs](https://twitter.com/ExaAILabs/status/1963262700123000947), [@RichardSocher](https://twitter.com/RichardSocher/status/1963277700711461241)).
- VS Code’s support for custom OAI-compatible endpoints is a quiet enabler for local/self-hosted stacks—fewer reasons to be locked to a single vendor ([@ggerganov](https://twitter.com/ggerganov/status/1963255949373677959)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Kimi K2 Launch and LLM Benchmark Leaderboards

- [**Introducing Kimi K2-0905**](https://www.reddit.com/r/LocalLLaMA/comments/1n7fdy4/introducing_kimi_k20905/) ([Score: 391, Comments: 85](https://www.reddit.com/r/LocalLLaMA/comments/1n7fdy4/introducing_kimi_k20905/)): **Announcement of “Kimi K2-0905” contains only a promo image and no technical details, benchmarks, weights, code, or API info; the post links solely to an image asset: https://preview.redd.it/u8oxbcfyfymf1.png?width=2178&format=png&auto=webp&s=87daf02d6f257631f0a0a8847de7180dc9d9eed8. No model card, changelog, or release artifacts are provided in the text of the post.** Top comments criticize the marketing/UX (“looks like a crypto airdrop scam ad,” “half-slop, half-zoomer”) and question release details: *“No weights? I guess will be released on the 5th (unless going API only).”*
    - Lack of released weights noted; a commenter speculates the 0905 tag implies a Sep 5 drop unless it’s API-only. This raises practical concerns for self-hosting and independent benchmarking (latency/throughput, context length, eval reproducibility, and licensing), which are only feasible with open weights.
    - Timing and positioning: a commenter says the first K2 was overshadowed by Qwen 3 Coder’s release, suggesting K2-0905 will be scrutinized on coding benchmarks and head-to-head comparisons against Qwen 3 Coder, especially for code synthesis and repair tasks.
- [**GPT-OSS 120B is now the top open-source model in the world according to the new intelligence index by Artificial Analysis that incorporates tool call and agentic evaluations**](https://i.redd.it/6c1jae9atvmf1.png) ([Score: 337, Comments: 204](https://www.reddit.com/r/LocalLLaMA/comments/1n75z15/gptoss_120b_is_now_the_top_opensource_model_in/)): **Artificial Analysis’s new Intelligence Index aggregates open‑source LLM performance across academic evals (e.g., MMLU‑Pro, GPQA Diamond) plus tool‑call and agentic tasks; per the chart, GPT‑OSS 120B ranks #1 with a composite score** `58`**, edging models like Qwen3 and DeepSeek (others range** `57–21`**). Methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking; the index reports a single composite score derived from multiple evaluations.** Comments question the ordering: one prefers GLM 4.5 as closest to Claude Sonnet/Opus, and another challenges Gemma 3 being ranked behind Phi‑4, suggesting disagreements about weighting or coverage of tasks.
    - A practitioner claims **GLM 4.5** is the closest OSS model to **Claude 3.5 Sonnet** or **Claude Opus** in capability, preferring it over the newly crowned GPT-OSS 120B despite the index. This suggests perceived near-parity in general reasoning/chat quality from **GLM 4.5** relative to top proprietary models for their workloads.
    - A commenter questions why **Gemma 3** ranks behind **Phi-4**, implicitly probing how the index’s agentic/tool-call weighting might advantage certain model families or training regimes. This highlights potential sensitivity of the ranking to evaluation design, encouraging scrutiny of how tool-use and multi-step tasks are scored.
    - Skepticism toward benchmark-driven leaderboards: a user argues that *"real world usage is the true math"* and that OSS "doesn’t add up" for their use case. They imply leaderboard scores may not translate directly to production effectiveness, challenging the practical relevance of the new index.
- [**German "Who Wants to Be a Millionaire" Benchmark w/ Leading Models**](https://www.reddit.com/gallery/1n7g0c2) ([Score: 190, Comments: 47](https://www.reddit.com/r/LocalLLaMA/comments/1n7g0c2/german_who_wants_to_be_a_millionaire_benchmark_w/)): **Authors re-ran the German Wer wird Millionär? QA benchmark across leading LLMs using the original rules:** `45` **simulated game runs, each with** `15` **A–D multiple-choice questions (in German), no lifelines, one wrong answer ends the run and you keep current winnings. They reused the public WWM corpus ([dataset](https://github.com/GerritKainz/wer_wird_millionaer)) and the original benchmark concept ([ikiruneo/millionaire-bench](https://github.com/ikiruneo/millionaire-bench)), added parallel English text for transparency (**`fragen_antworten_en.json`**), and provided scripts for batch evaluation and leaderboard reconstruction (**`millionaire-run.py`**,** `rebuild_leaderboard.py`**) in a new repo: [Jose-Sabater/millionaire-bench-opper](https://github.com/Jose-Sabater/millionaire-bench-opper). Results are shared via a leaderboard screenshot (same scoring/structure as the original) and the setup is packaged for quick reruns or PRs.** Commenters suggest implementing the real show’s "quit to keep winnings" decision point and measuring when/if models elect to stop, turning it into a risk-aware evaluation. There are also requests to include additional models (e.g., Gemini 2.5 Pro).
    - Benchmark design detail: A Millionaire-style eval should model the “quit” option explicitly by asking the model for a calibrated probability of correctness and then deciding to answer vs. walk away based on expected value under the show’s stepwise payout/safe-haven structure. This tests risk-sensitive decision-making and confidence calibration (e.g., Brier/ECE) in addition to QA accuracy; see evidence that LMs can estimate their own uncertainty in Kadavath et al. 2022, *Language models (mostly) know what they know* (https://arxiv.org/abs/2207.05221). Reporting both average winnings and calibration metrics would distinguish models that “know when to quit” from those that over/under-confidently guess.
    - Language confound: Using the German version primarily probes multilingual comprehension and culturally anchored knowledge, not just general reasoning. Many models show non-trivial drops moving from English to other languages (e.g., MGSM reports sizeable gaps across languages: https://arxiv.org/abs/2305.11938; broader cross-lingual variance in XTREME: https://arxiv.org/abs/2003.11080), so an English run would likely shift rankings upward for English-centric models. To isolate reasoning vs. language, consider parallel German/English runs or translation-controlled variants.
    - Model comparison nuance: Anecdotes that **GLM-4.5** produces code on par with “GPT-5” suggest parity on coding tasks, but Millionaire-style trivia emphasizes factual recall and calibrated QA. To validate cross-domain claims, compare on code benchmarks (e.g., HumanEval: https://github.com/openai/human-eval; MBPP: https://arxiv.org/abs/2108.07732) alongside knowledge QA (e.g., Natural Questions: https://ai.google.com/research/NaturalQuestions). Expect clusters where models align on coding yet diverge on open-domain knowledge and calibration, affecting Millionaire outcomes.

### 2. GPU Hardware: Intel Arc Pro B50 and 4x3090 vs RTX 6000

- [**Intel launches Arc Pro B50 graphics card at $349**](https://i.redd.it/357rwwhaizmf1.jpeg) ([Score: 150, Comments: 108](https://www.reddit.com/r/LocalLLaMA/comments/1n7l5kg/intel_launches_arc_pro_b50_graphics_card_at_349/)): **Intel has launched the Arc Pro B50 workstation GPU at $349, positioned as a budget pro card and marketed as an alternative to NVIDIA’s A1000, per VideoCardz. The post and thumbnail make a bold claim (“Better than NVIDIA”), but no hard benchmarks are provided; a spec noted in discussion is ~**`224 GB/s` **memory bandwidth, implying midrange performance. Source: https://videocardz.com/newz/intel-launches-arc-pro-b50-graphics-card-at-349** Commenters argue the `224 GB/s` bandwidth is limiting and that an RTX 3060 would outperform it; some wanted more VRAM, and others claim an RTX 5060 Ti (~$80 more) offers better value due to CUDA support and higher bandwidth, with even used dual 3060s seen as superior.
    - Bandwidth is a recurring concern: commenters note the Arc Pro B50’s `~224 GB/s` memory bandwidth (implying a 128‑bit GDDR6 interface) as a bottleneck, contrasting it with the **RTX 3060 12GB** at `360 GB/s` ([specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-12-gb.c3621)). The expectation is that a 3060 would outperform the B50 in many bandwidth‑sensitive workloads.
    - Several highlight the lack of **CUDA** as a major drawback for pro/compute workflows. Without CUDA ([NVIDIA CUDA](https://developer.nvidia.com/cuda-zone)), compatibility and performance in many DCC/ML/compute applications can lag versus NVIDIA options, undercutting the B50’s value even if raw specs are competitive in some areas.
    - Value and positioning vs Intel’s own lineup: one user argues the B50 costs “$100 more” than a B580 yet is slower on most fronts, with the B50’s only clear advantage being `+4 GB VRAM` and a smaller, lower‑power form factor. The takeaway: unless you specifically need SFF and lower power, the **B580** is seen as the faster and cheaper choice.
- [**Any actual downside to 4 x 3090 ($2400 total) vs RTX pro 6000 ($9000) other than power?**](https://www.reddit.com/r/LocalLLaMA/comments/1n71b95/any_actual_downside_to_4_x_3090_2400_total_vs_rtx/) ([Score: 158, Comments: 184](https://www.reddit.com/r/LocalLLaMA/comments/1n71b95/any_actual_downside_to_4_x_3090_2400_total_vs_rtx/)): **OP asks whether 4× RTX 3090 (~$2.4k total, Ampere, 24 GB each) is a practical substitute for a single RTX 6000-class pro card (~$9k) for local LLMs like “Qwen 3 Coder” and “GLM 4.5 Air.” Top replies note that VRAM isn’t aggregated: a model must fit in one GPU unless you use tensor/pipeline parallelism (e.g., Megatron-LM tensor-parallel), which introduces NCCL/PCIe comms costs; consumer boards often bifurcate to x8/x8/x4/x4 or worse, so 4 GPUs may run at ~x4 each, hurting scaling. Ampere lacks native low-precision paths (FP8/FP4) that newer stacks increasingly target, so engines like vLLM may lag or need workarounds; effective VRAM is reduced by CUDA/runtime overhead; used GPUs carry reliability risks, while the RTX 6000-class offers better vendor support/drivers.** Commenters are skeptical of the $600/3090 price and argue a single large GPU is almost always faster and simpler than multiple smaller cards due to interconnect bottlenecks and parallelization overheads.
    - PCIe lane bottlenecks will kneecap 4×3090 on consumer platforms: each 3090 expects an x16 link, but typical desktop CPUs expose ~`24` lanes total, so four cards end up at ~x4 each, slashing host↔device bandwidth (`PCIe 4.0 x4 ≈ ~8 GB/s` vs `x16 ≈ ~32 GB/s`) and hurting multi‑GPU throughput; you’d need a workstation/HEDT platform with 64+ lanes to avoid this ([PCIe bandwidth](https://en.wikipedia.org/wiki/PCI_Express#History_and_revisions)). In practice, for single‑model training/inference, one big card often outperforms several smaller cards due to reduced inter‑GPU sync and communication overhead.
    - Multi‑GPU LLM scaling adds overheads: effective VRAM per card drops from CUDA context/allocator overhead and tensor‑parallel sharding, and while tensor parallelism can be finicky to configure, pipeline parallelism introduces bubbles that reduce utilization/throughput (see [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)). Ampere (3090) lacks native FP8/FP4 Tensor Core modes, whereas the **RTX 6000 Ada** supports FP8 on 4th‑gen Tensor Cores ([RTX 6000 Ada](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)), so newer inference/training optimizations may land there first; expect to wait longer for engine support on Ampere.
    - Total cost of ownership: 4×3090 at full tilt vs a single RTX 6000 Ada can mean on the order of `~7,000 kWh/year` extra energy per the discussion, which can be “upwards of `$3,000`/year” depending on local rates, plus added cooling/HVAC costs. Nominal board powers back this trend (3090 ~`350 W` each vs RTX 6000 Ada ~`300 W` total) ([3090 specs](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/), [RTX 6000 Ada](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)). Used 3090s also carry higher failure risk and earlier software/driver EOL, whereas the pro card generally has longer support and vendor backing.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 3 Pretraining Success + Tesla Optimus 3 First Photo/Video

- [**Looks like Gemini 3 might've had a successful pre-training run**](https://www.reddit.com/gallery/1n7cark) ([Score: 319, Comments: 111](https://www.reddit.com/r/singularity/comments/1n7cark/looks_like_gemini_3_mightve_had_a_successful/)): **A post asserts that Google DeepMind’s next-gen model, “Gemini 3,” has completed a successful pre‑training run, implying core unsupervised training may be finished. However, there are no disclosed technical details (token count, compute scale, architecture/window changes, or evals), and the linked evidence is a Reddit gallery that returns** `HTTP 403` **([gallery link](https://www.reddit.com/gallery/1n7cark)). Commenters report that a Gemini pre‑training co‑lead publicly refuted the claim, suggesting the information may be premature or inaccurate.** Discussion splits between timeline speculation (e.g., “pre‑training is completed NOW → release by year‑end?”) and credibility concerns, with multiple users citing the co‑lead’s denial and questioning the source (“Dylan”). Some ask whether a denial means Gemini 3 isn’t “incredibly performant,” while others note it may simply indicate rumors are unfounded rather than performance-related.
    - Speculation that **Gemini 3** pretraining just finished (implying a potential release by year-end) is contested: a cited **Gemini pretraining co-lead** reportedly denied the rumor source’s claims, so there’s no credible confirmation that training is complete or that the model is already "incredibly performant." Technically, without official signals (e.g., paper, blog, or benchmark deltas), a completion inference is weak; release timing remains speculative.
    - A referenced "Woodward" tweet was clarified by commenters as about the popularity of "nano banana," not an LLM pretraining milestone—analogous to **OpenAI**’s playful "servers on fire" quips around launches. Conclusion: the tweet is social chatter, not an indicator of **Gemini 3** training status or performance progress.
    - Multiple users caution on the reliability of **Dylan Patel**’s rumors; absent hard metrics (e.g., MMLU, GPQA, BIG-bench, or `ARENA Elo`) or official evals, claims of "incredible performance" are premature. The technically prudent approach is to wait for reproducible benchmarks and methodology details before inferring capability or readiness.
- [**First video of Optimus 3**](https://v.redd.it/jjplx5j3kzmf1) ([Score: 596, Comments: 453](https://www.reddit.com/r/singularity/comments/1n7lebe/first_video_of_optimus_3/)): **Post shares the “first video” of Tesla’s humanoid robot “Optimus 3,” linking to a Reddit-hosted clip [v.redd.it/jjplx5j3kzmf1](https://v.redd.it/jjplx5j3kzmf1) that currently returns** `HTTP 403` **(network-security block), so no technical content (locomotion, manipulation, autonomy stack, sensors, or benchmarks) can be verified from the source. With the media inaccessible, the post itself provides no specs or implementation details to compare against prior public Optimus iterations, so any claims of hardware/control-stack changes cannot be assessed from this link alone.** Top comments are non-technical and skeptical, implying the update appears cosmetic rather than functional (e.g., *“now he can do nothing 30% more shinier,”* *“NPC”/“Gen Z stare”*), suggesting perceived minimal capability gains.
- [**First photo of Optimus 3**](https://i.redd.it/b36k6a7afzmf1.jpeg) ([Score: 300, Comments: 169](https://www.reddit.com/r/singularity/comments/1n7ko97/first_photo_of_optimus_3/)): **First public image of Tesla’s third‑gen humanoid, “Optimus 3,” shows a refined shell with a reflective head/torso, visible Tesla branding, and a slimmer, more human‑proportioned frame walking in an office setting. Notable are highly human‑like hands and fully articulated limbs, suggesting a design emphasis on dexterity and natural gait, though no specs or demos are provided in the post.** Comments flag recurring chassis/port jokes (the “hole”) and critique possible pelvis alignment, while others note the hands look unusually human if functional—implying skepticism about whether they’re cosmetic or capable.
    - Commenters highlight the apparent realism of the hands—“if those hands work… the most human looking hands I’ve ever seen on a robot.” Technically, the geometry suggests anthropomorphic proportions and potentially high-DOF, independently actuated fingers; if functional, this could enable dexterous in-hand manipulation and a broader grasp taxonomy than prior Optimus demos.
    - One observer notes “They screwed the pelvis in all wrong,” implying a misaligned hip/pelvis interface. Such misalignment would impact hip joint kinematics, range-of-motion, and center-of-mass alignment for gait stability; alternatively, it could be a provisional cosmetic shell/cover orientation typical in early prototype fitment.
    - A question about “Any update on hole yet?” hints at a previously noted chassis aperture/enclosure gap on earlier iterations. This suggests packaging/enclosure integration is still in flux, with mechanical closure and routing not fully finalized in the prototype stage.
- [**The one job ai won't take in 100 years is... Programming - Bill Gates**](https://www.leravi.org/bill-gates-reveals-the-one-job-ai-will-never-replace-even-in-100-years-10272/) ([Score: 507, Comments: 167](https://www.reddit.com/r/OpenAI/comments/1n72qgw/the_one_job_ai_wont_take_in_100_years_is/)): **Bill Gates says programming will remain a *“100% human profession”* even in** `100` **years, asserting that AI will automate repetitive coding but not the inventive problem‑solving and judgment at the core of software engineering ([France Inter coverage via Le Ravi](https://www.leravi.org/bill-gates-reveals-the-one-job-ai-will-never-replace-even-in-100-years-10272/)). Top commenters counter with a technical framing: current LLMs scale to longer tasks but are still constrained on long‑horizon, multi‑year, multi‑team goals (e.g., “ship an ‘amazing’ game”), so they excel at decomposed sub‑tasks yet require human-led specification, orchestration, and integration. Programming remains the domain where AI is most practically helpful today (code generation, refactoring, tests), but reliable autonomous agents for months‑to‑years projects remain an open problem.** Debate splits between: (1) long‑horizon autonomy is the key blocker—humans will stay in the loop to define, decompose, and own end‑to‑end outcomes; versus (2) programming is uniquely susceptible to automation because it is language‑native, highly lucrative, and awash in training and synthetic data—if AI can’t take this job, it likely can’t take most others.
    - A key technical claim is about task-horizon limits: current LLMs handle short, well-scoped coding tasks but struggle with months-to-years, multi-person software projects that require stable objectives, architecture, and hierarchical decomposition. Agentic coding systems still falter on repo-scale changes, dependency management, and long-term coherence; benchmarks like SWE-bench (https://www.swebench.com/) show limited end-to-end success on multi-file bug fixes despite strong snippet-level code generation, keeping humans responsible for scoping and orchestrating work.
    - Counterpoint emphasizes why programming is unusually well-suited for LLM automation: it’s fully language-mediated, has vast public training corpora (e.g., open-source repos), and supports synthetic data via test generation and fill-in-the-middle pretraining. Critically, compilers, linters, and unit tests provide fast, automatic feedback loops that enable execute–debug–retry tooling and RL-style signals, suggesting software engineering may be among the first domains where robust autonomy emerges.
    - Practitioner perspective: LLMs provide the biggest lift in programming by accelerating boilerplate, tests, refactors, and API glue while humans handle product definition, architecture, and cross-system integration. Empirical data backs sizable speedups on routine tasks—e.g., GitHub’s study reported ~`55%` faster task completion with Copilot ([https://github.blog/2022-09-07-research-quantifying-github-copilots-impact-on-developer-productivity/)—yet](https://github.blog/2022-09-07-research-quantifying-github-copilots-impact-on-developer-productivity/)%E2%80%94yet) long-horizon planning and evolving requirements remain challenging for current models.

### 2. OpenAI Parental Controls/Privacy & UX Backlash + Salesforce AI Layoffs

- [**Salesforce CEO confirms 4,000 layoffs ‘because I need less heads' with AI**](https://www.cnbc.com/2025/09/02/salesforce-ceo-confirms-4000-layoffs-because-i-need-less-heads-with-ai.html) ([Score: 494, Comments: 178](https://www.reddit.com/r/singularity/comments/1n722tp/salesforce_ceo_confirms_4000_layoffs_because_i/)): **Salesforce CEO Marc Benioff confirmed on a podcast that AI automation via its customer-service bots ("Agentforce") has reduced support case volumes enough to cut ~**`4,000` **customer-support roles—shrinking support headcount from ~**`9,000` **to ~**`5,000`**—and the company will not backfill those roles; Benioff has previously claimed AI performs up to** `50%` **of work at Salesforce. Coverage via CNBC: https://www.cnbc.com/2025/09/02/salesforce-ceo-confirms-4000-layoffs-because-i-need-less-heads-with-ai.html. Analysts cited include Laurie Ruettimann (urging reskilling vs. cuts) and Ed Zitron (criticizing post-pandemic overhiring and AI as a cost-cutting pretext).**
    - One commenter claims `~50%` of companies that tried to replace human customer support with AI reported a “bad experience,” citing core limitations: LLM hallucinations, customer dissatisfaction with bots, and inability to perform authenticated/account-level actions beyond simple FAQs. The point implies that production-ready support automation requires secure action-execution (tool/API integrations with auth/audit), robust fallback to human agents, and guardrails to prevent incorrect actions—areas where current AI deployments often fall short.
- [**Salesforce CEO Marc Benioff says AI enabled him to cut 4,000 jobs**](https://www.finalroundai.com/blog/salesforce-ceo-marc-benioff-says-ai-enabled-him-cut-4000-jobs) ([Score: 677, Comments: 158](https://www.reddit.com/r/ChatGPT/comments/1n76jmf/salesforce_ceo_marc_benioff_says_ai_enabled_him/)): **Salesforce CEO Marc Benioff said the company cut about** `4,000` **customer-support roles after deploying AI agents that now handle ~**`50%` **of customer conversations; each agent type processed ~**`1.5M` **interactions and drove a reported** `17%` **reduction in support costs since early 2025. He cited AI-enabled omni-channel supervision and agentic sales systems that scale support and internal outreach (>**`10k` **leads/week), CSAT parity between AI- and human-handled conversations, and only “hundreds” redeployed, while signalling further function-by-function automation—a reversal from his July 2025 “augment-not-replace” stance. The move aligns with broader 2025 AI-driven workforce reductions across large tech (e.g., Microsoft, IBM, Coinbase).** Commentary questions retaining highly paid executives while automating frontline roles, and flags practical risks: AI support loops may hinder warranty/consumer-rights enforcement versus humans who can escalate or exercise discretion; localization/legal-competency gaps (e.g., non-EU support unfamiliar with EU law) could be amplified by AI systems.
    - Customer-support automation limitations: One commenter argues that AI chatbots often fail at jurisdiction-aware reasoning and enforcement, especially for EU/German warranty cases, noting that humans may ultimately grant entitlements after persistence whereas an AI can loop indefinitely without escalation. Technical implication: production support bots need country-specific policy engines and knowledge bases, confidence thresholds with mandatory human handoff, and auditable decision logs to comply with consumer-protection rules (e.g., EU Consumer Rights Directive 2011/83/EU: https://eur-lex.europa.eu/eli/dir/2011/83/oj).
- [**Kids don’t need parental controls, they need parental care.**](https://i.redd.it/30c2s59csumf1.jpeg) ([Score: 381, Comments: 217](https://www.reddit.com/r/OpenAI/comments/1n71u8m/kids_dont_need_parental_controls_they_need/)): **The image is a news screenshot stating that OpenAI’s ChatGPT will add parental controls that can “notify parents” if the system detects signs of** `acute distress` **in a young user, reportedly prompted by a teen suicide case; per the [Washington Post report](https://www.washingtonpost.com/technology/2025/09/02/chatgpt-parental-controls-suicide-openai/), this entails distress-detection and a parent-linked account flow, though specifics (signals used, thresholds, opt-in/consent model, data retention, and escalation pathways) are not detailed. The post’s title argues that controls alone are insufficient, implying a broader child-safety and guardianship policy shift rather than a mere UI toggle.** Comments are divided: some view parental controls as part of care, while others warn of privacy risks (outing LGBTQ+ youths, alerting abusive parents) and stress that outcomes depend on implementation—opt-in mechanics, safe contacts vs. parents, privacy safeguards, and false-positive handling.
    - Implementation risk is centered on how "parental controls" are built: whether they enable parent dashboards, chat-log visibility, or automated alerts about sensitive topics. Commenters warn about classifier and policy design (e.g., `false-positive` alerts on identity/mental-health queries) that could leak highly sensitive data to unsafe guardians, suggesting granular scopes (content vs. metadata), consent gates for older minors, and clear escalation criteria to avoid harm in edge cases (e.g., abuse at home).
    - Security/evasion concerns: app-level controls are trivially bypassed by teens (new accounts, different devices, VPNs, alternate models), so any real control must be defense-in-depth (OS-level profiles, MDM, network/DNS filters) and robust account/age-linking. Otherwise, logging or alerts in a single app provide a false sense of safety while being easy to route around.
    - Safety architecture suggestions emphasize privacy-preserving interventions over parental disclosure: on-device nudges, ephemeral or encrypted-by-default storage, and a "confidential mode" that suppresses parent-visible logs for crisis topics while still offering resources. Escalation flows should prefer third-party hotlines/resources and require explicit minor consent for parent notifications, with auditable thresholds for classifiers to minimize `false-negative/false-positive` harm.
- [**the new "parental mode" is patronizing adults and killing what made chatgpt special**](https://www.reddit.com/r/ChatGPT/comments/1n7ioo0/the_new_parental_mode_is_patronizing_adults_and/) ([Score: 261, Comments: 251](https://www.reddit.com/r/ChatGPT/comments/1n7ioo0/the_new_parental_mode_is_patronizing_adults_and/)): **Users report a new global safety layer (“parental mode”) in ChatGPT that applies stricter moderation across models (incl. [GPT‑4o](https://openai.com/index/hello-gpt-4o/)), with self‑harm/“sensitive” triggers causing automatic hotline interventions even in clearly fictional/creative contexts. A top comment describes reproducible behavior indicating a server‑side, post‑generation filter: the assistant denies blocking, attributes it to an external filter, suggests a bypass, yet the same intervention text is injected repeatedly—implying a non‑overrideable policy layer separate from the model output. The OP also alleges silent model swapping and cost‑saving motivated downgrades, reduced transparency, and broadened “sensitive content” definitions impacting legitimate use cases; see OpenAI’s general [usage policies](https://openai.com/policies/usage-policies) for context.** Debate centers on liability vs. user autonomy: some argue companies “nerf” models to avoid lawsuits over self‑harm incidents, while others demand opt‑outs and adult controls, claiming the thresholds are overbroad and break workflows.
    - Multiple users report reproducible false positives from a server-side self-harm/sensitive-content safety layer that overrides the model, returning canned hotline text even in clearly fictional contexts. One user notes the model itself acknowledges “a filter I am triggering,” implying a post-generation moderation pass rather than the base model choice, and that attempts to rephrase per the model’s guidance still re-trigger the filter across `~7` tries—evidence of a high-recall, low-precision classifier insensitive to narrative framing and prior chat history.
    - The triggering appears keyword/phrase-driven (e.g., “off oneself,” “drawing blood,” imprisonment/hell scenarios), with poor context handling for adult/creative use cases and no session-level exception. This suggests input and/or output moderation classifiers running independently of system intent (fiction writing) and persona, similar to typical multi-stage pipelines (prompt classification + completion classification) described in moderation approaches like **OpenAI’s** own docs: https://platform.openai.com/docs/guides/moderation/overview.
    - Commenters infer a recent policy/threshold shift (“parental mode”) prioritizing compliance/liability reduction over precision, effectively expanding blocks to S3/S4 categories (self-harm, violence) even in third-person or hypothetical depictions. Technically recommended mitigations from users include context-aware safety (respecting “fiction” tags), adjustable thresholds or per-account toggles, and mode switches (e.g., “research/fiction mode”) to reduce overblocking without removing guardrails.
- [**OpenAI is dying fast, you’re not protected anymore**](https://i.redd.it/qyk1kjdumymf1.jpeg) ([Score: 4400, Comments: 1016](https://www.reddit.com/r/ChatGPT/comments/1n7gcyi/openai_is_dying_fast_youre_not_protected_anymore/)): **The image is a sensational meme-style claim that “OpenAI is scanning users’ ChatGPT conversations and reporting content to the police.” In reality, OpenAI (like most online platforms) runs automated safety/moderation systems over user inputs/outputs and states in its policies that it may disclose information to law enforcement when legally required or to prevent imminent harm; this is not a blanket, proactive “report everything” regime, but content-review and legal-compliance workflows common across tech platforms ([Privacy Policy](https://openai.com/policies/privacy-policy), [Usage Policies](https://openai.com/policies/usage-policies)). Users can limit training use of their chats (e.g., chat history controls; enterprise/teams offer stronger data-retention and training opt-outs), but moderation scanning still applies for safety.**  Top comments are largely cynical, asserting user data was never private and questioning the legality/ethics of model training data. Technical debate is minimal; most reactions are non-technical or humorous about extreme prompts being flagged/reported.
    - One commenter notes OpenAI acknowledged that *“a small team monitors risky conversations,”* which aligns with OpenAI’s human-in-the-loop moderation pipeline: automated classifiers flag safety-sensitive categories (e.g., self-harm, violence, illegal activity) and may escalate to limited authorized reviewers for policy enforcement and model improvement. Practically, user content can be reviewed and used for training unless data sharing is disabled (ChatGPT “Chat History & Training” off, API data opt-out; enterprise defaults off). References: OpenAI Privacy Policy (https://openai.com/policies/privacy-policy), Data usage controls (https://help.openai.com/en/articles/7934734-how-your-data-is-used-to-improve-model-performance), Usage Policies (https://openai.com/policies/usage-policies).
    - Another thread points to concerns over training data legality and privacy: OpenAI states models are trained on a mix of **publicly available**, **licensed**, and **human-generated** data, but hasn’t disclosed granular sources, increasing scrutiny around potential inclusion of copyrighted or personal data in web-scale corpora. This lack of dataset transparency is a known trade-off between competitive secrecy and accountability and has implications for compliance and red-teaming of data provenance. Reference: GPT-4 Technical Report (https://cdn.openai.com/papers/gpt-4.pdf) and Privacy Policy (https://openai.com/policies/privacy-policy).
- [**This filter needs to be removed**](https://v.redd.it/3206q93s5vmf1) ([Score: 280, Comments: 88](https://www.reddit.com/r/OpenAI/comments/1n73fx1/this_filter_needs_to_be_removed/)): **Users report inconsistent safety moderation across OpenAI model variants: a query “Did Judas hang himself” was answered directly by** `5 (Instant)` **and** `GPT‑4o` **([model info](https://openai.com/index/gpt-4o/)) but the** `5 (Thinking)` **variant began to answer then invoked a safety interstitial/censorship. Another commenter notes gun‑law queries (e.g., checking legality of machine‑gun rentals, which can be legal under U.S. NFA rules in certain jurisdictions) surfaced crisis/helpline messaging instead of straightforward legal guidance—suggesting more aggressive intent classification on the reasoning/"Thinking" path. The linked video ([v.redd.it](http://v.redd.it/)) returns HTTP** `403` **requiring authentication, indicating access control rather than content removal. For general model references, see OpenAI’s [models docs](https://platform.openai.com/docs/models).** Commenters characterize the `5 (Thinking)` model as over‑restricted/“nerfed,” arguing safety filters are excessively sensitive compared to `5 (Instant)` and `GPT‑4o`; frustration centers on mid‑generation censorship and help‑line inserts on lawful informational queries.
    - A/B test across `5 (Instant)`, `5 (Thinking)`, and `4o` shows divergent safety behavior on the prompt "Did Judas hang himself": `5 (Instant)` and `4o` answered directly without refusal, while `5 (Thinking)` began answering then switched to a refusal. This points to a late-stage moderation override specific to the "Thinking" variant (e.g., a post-generation safety pass that can redact/replace an answer mid-stream) rather than a uniform policy across models. The discrepancy implies model-specific safety thresholds/classifiers with the "Thinking" model tuned more aggressively for self-harm phrasing even in historical/academic contexts.
    - Reports of false positives on lawful firearms queries: asking about buying a gun and state gun laws (including checking the legality of "machine gun rentals") triggered crisis/support messaging and refusals. This suggests keyword-driven violence/self-harm classifiers are over-triggering on intent-neutral legal research, favoring high recall over precision. A better configuration would condition on user intent and jurisdictional context and allow compliant legal information with safety framing instead of blanket suppression.
    - Users observe that the assistant sometimes "writes a response but gets overwritten with disclaimers," indicating a server-side guardrail that can replace an already-streaming answer when a risk score trips mid-output. This generate-then-redact pipeline causes visible flips (answer → refusal), degrading UX for paying users and making the system appear inconsistent. Architecturally, pre-decode policy steering or span-level redaction would mitigate mid-stream overwrites while preserving compliant content.
- [**GPT5 Offering Additional Tasks Is The Most Annoying It's Ever Been**](https://www.reddit.com/r/ChatGPT/comments/1n7eqsw/gpt5_offering_additional_tasks_is_the_most/) ([Score: 338, Comments: 206](https://www.reddit.com/r/ChatGPT/comments/1n7eqsw/gpt5_offering_additional_tasks_is_the_most/)): **OP reports that in the ChatGPT/GPT‑5 app/desktop client, the assistant persistently appends proactive offers (e.g.,** `Would you like me to <task>?`**) that are extremely hard to suppress—even after embedding negative instructions in personalization/memory, using regex-style constraints, requesting chain‑of‑thought intentions to avoid offers, and iterative prompt‑engineering strategies. The phrasing adapts (e.g.,** `If you wish I could…`**), suggesting a strong, client‑level system prompt or alignment template (likely RLHF‑driven helpfulness heuristics; see [InstructGPT RLHF](https://arxiv.org/abs/2203.02155)) that overrides user instructions; OP notes this is specific to the app/desktop client, not API workflows (where system prompts are explicitly controllable; cf. [Chat Completions "system" role](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages)). The model also acknowledges the low expected utility of its own suggestions when asked, highlighting a misalignment between “be proactively helpful” priors and actual task utility.** Top comments corroborate limited, short‑lived suppression ("for one or two messages") and report similar overreach where the model rewrites text unasked during simple grammar/flow checks, reinforcing that the aggressive “offer next steps” style is a persistent, undesired behavior.
    - Multiple users highlight a UX issue where GPT’s proactive “additional tasks” prompts can be suppressed only transiently (often for just one message), implying there’s no persistent per-user or per-thread preference flag to disable initiative. They ask for a global opt-out toggle or setting to keep the assistant in a strictly reactive mode by default.
    - Reports indicate the intent classifier overreaches on simple proofreading requests, performing full rewrites or offering structured artifacts (e.g., graphs/lists/pictures) instead of minimal grammar/flow fixes. A constrained “proofread-only” mode that returns diffs or inline suggestions (without reformatting or expanding content) is suggested to reduce false positives and preserve author voice.
    - Keyword-triggered helper flows (e.g., subscription management prompts) are firing in irrelevant contexts, suggesting aggressive heuristics or low confidence thresholds for action suggestions. Users recommend higher confidence gating or explicit opt-in before launching specialized flows to reduce intrusive, off-target assistance.
- [**I was asking chat about why lying on my left side would help reflux, it offered to show me a diagram.**](https://i.redd.it/lkvqkm1uevmf1.png) ([Score: 274, Comments: 39](https://www.reddit.com/r/ChatGPT/comments/1n74gmh/i_was_asking_chat_about_why_lying_on_my_left_side/)): **OP asked why sleeping on the left side can reduce reflux, and an AI produced a diagram contrasting left- vs right-lateral positions. Technically, left lateral decubitus tends to keep the gastroesophageal junction (LES) above the gastric acid pool (fundus along the greater curvature), leveraging gravity and the angle of His to reduce retrograde flow; right-side lying can place the LES dependent relative to the acid, increasing reflux risk.** Commenters joke about the orientation/labeling (e.g., suggesting flipping the phone), implying the AI diagram may be mirrored or crudely drawn, but there’s no substantive technical dispute.
- [**URGENT - my girlfriend used chatGPT for her work. Now her boss wants her to explain the calculations. I think the calculations were a hallucination. What to do?**](https://www.reddit.com/r/ChatGPT/comments/1n78p0v/urgent_my_girlfriend_used_chatgpt_for_her_work/) ([Score: 8705, Comments: 3099](https://www.reddit.com/r/ChatGPT/comments/1n78p0v/urgent_my_girlfriend_used_chatgpt_for_her_work/)): **OP describes a client-facing survey analysis produced via ChatGPT, where the model generated an Excel and a resulting PowerPoint; when asked to explain the methodology, ChatGPT claimed it used Pearson’s correlation coefficient on 5-bucket textual “feelings” responses. This points to a hallucinated or invalid method: Pearson’s r ([wiki](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)) assumes numeric/interval data and an explicit encoding of variables—none was documented—so the results are non-reproducible and unverifiable, exemplifying LLM “hallucination” risk ([overview](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))).** Commenters suggest either fabricating a cover story (e.g., “placeholder data”) or, more prudently, warn that clients may recognize AI-generated output and that misrepresenting methods poses higher ethical and professional risk than admitting misuse and redoing the analysis transparently.
    - Data privacy/compliance risk: A commenter flags that if any client data or PII was pasted into ChatGPT, this could violate company policy, NDAs, or regulations (e.g., GDPR/CCPA) and be more serious than a bad analysis. Unless using enterprise controls, ChatGPT consumer inputs may be retained/used to improve services; contrast with API/Enterprise modes that offer stricter data handling (no training on inputs, optional zero-retention) — see OpenAI’s data policies: https://openai.com/policies/api-data-usage and data controls FAQ: https://help.openai.com/en/articles/7730893-data-controls-faq. Organizations often require approved vendors and DPAs; uploading sensitive data to an unapproved third party can trigger incident reporting and forensics. The immediate step is to assess whether any sensitive fields were shared and escalate per policy if so.
    - Reproducibility/accountability: The client asking to “explain the calculations” suggests concern about provenance and reproducibility; LLMs can produce plausible but incorrect quantitative outputs (hallucinated numbers) and cannot provide a verifiable audit trail. Misrepresenting the source (“placeholder data”) is risky; a defensible approach is to reconstruct the analysis with transparent methods (spreadsheets/code) and document inputs, formulas, and intermediate results. Going forward, use LLMs to draft formulas or code but validate all numbers with deterministic tools, keeping artifacts so the work can be reproduced on demand. Admitting lack of proper AI usage can reflect poorly, but doubling down without a reproducible basis is worse from a technical and ethical standpoint.
- [**"Poured olive oil on them"**](https://i.redd.it/ama9gb826umf1.jpeg) ([Score: 242, Comments: 71](https://www.reddit.com/r/ChatGPT/comments/1n6z3je/poured_olive_oil_on_them/)): **A meme demonstrates users evading strict keyword/lexical guardrails by substituting fruit-coded euphemisms (e.g.,** `banana`**,** `peach`**) for prohibited historical figures/events (implicitly Adolf Hitler and Eva Braun), effectively preserving meaning while bypassing filters. It illustrates adversarial content obfuscation/prompt-coding that defeats naive string-matching and highlights the need for semantic, context-aware moderation rather than brittle blocklists. [Image link](https://i.redd.it/ama9gb826umf1.jpeg).** Top comments argue that strict guardrails “won’t work” because people will creatively rephrase content, with others posting variant examples (“Banana and Eva Banana”) that show how easy such obfuscation is.
    - Guardrails are described as brittle: strict, keyword/pattern-based safety filters are easily bypassed by creative prompting (paraphrases, indirection, obfuscation). The point implies robustness requires intent-aware moderation layers, adversarial red-teaming, and continuous evals for jailbreak resilience rather than static blocklists (see e.g., **Anthropic** on red-teaming: https://www.anthropic.com/news/red-teaming-language-models).
    - A user reports the model refusing to answer a neutral factual query about Hitler’s death, highlighting overblocking/false positives from miscalibrated safety classifiers. Technically, this suggests the need for context-sensitive policy routing (e.g., distinguishing historical/educational intent), calibrated thresholds, and allowlists for benign facts, measured via precision/recall on labeled safety datasets and spot-checks for known safe queries.

### 3. AI Video/Image Editing Workflows & Showcases: nano banana, Wan 2.2, Qwen, Local SD

- [**Experimenting with Continuity Edits | Wan 2.2 + InfiniteTalk + Qwen Image Edit**](https://v.redd.it/sgf6cc51rymf1) ([Score: 411, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1n7h56l/experimenting_with_continuity_edits_wan_22/)): **Episode 3 of an AI sci‑fi film experiment pushes continuity and dialogue using a Wan 2.2 pipeline with CausVid LoRAs (Wan 2.1), noting that lip‑synced dialogue is compute‑heavy (even on an** `RTX 5090`**) and fragile—minor flaws often force full re‑generations, so dialogue shots should be minimized. The creator reports InfiniteTalk > Wan S2V for speech‑to‑video—more expressive and prompt‑faithful—with shared auto‑frame workflows for multi‑person and single‑person shots ([paste 1](https://pastebin.com/N2qNmrh5), [paste 2](https://pastebin.com/BdgfR4kg)); for spatial continuity, Qwen‑Image‑Edit can synthesize alternate camera angles from a single frame, though with high failure rates, suggesting a potential LoRA for consistency. Prior episodes and outputs are on the YouTube channel: [youtube.com/@Stellarchive](http://www.youtube.com/@Stellarchive).** Top feedback: minor motion artifacts (hands) are visible; a commenter corrects naming to **Qwen‑Image‑Edit** (not “Wan Image Edit”); otherwise, reception is positive with little additional technical critique.
    - A viewer noted `1–2` artifacts on the subject’s hand during motion, hinting at minor temporal consistency issues in the continuity edits. This is a common failure mode when applying per-frame image editing over video (e.g., Qwen Image Edit on frames generated by **Wan 2.2**), where moving extremities and occlusions can produce jitter or smearing.
    - Clarification on tooling: the image editing model referenced is **Qwen-Image-Edit**, not "Wan Image Edit". This aligns with the pipeline in the title (**Wan 2.2** for generation, **InfiniteTalk** for speech/lipsync, and **Qwen-Image-Edit** for frame edits).
    - A suggestion to try the in-scene LoRA for Qwen image editing: [flymy-ai/qwen-image-edit-inscene-lora](https://huggingface.co/flymy-ai/qwen-image-edit-inscene-lora). In-scene LoRAs are aimed at preserving scene layout/lighting while editing localized elements, which could reduce artifacts in moving regions.
- [**I asked nano banana to get me into my favorite arcade**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 276, Comments: 33](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **Creator demonstrates an AI-assisted compositing workflow: a real first still is edited with nano banana (image cleanup/insert), then animated via Kling** `2.1` **using start/end-frame constraints to interpolate motion, with music generated by Producer AI and final sequencing/color in DaVinci Resolve. A step‑by‑step tutorial is provided in the post’s [X thread](https://x.com/techhalla/status/1963333488217919668).** Top comments are largely non-technical praise, noting the piece “sets the bar” creatively; no substantive technical critiques or benchmarks discussed.
- [**Is it possible to do this locally?**](https://i.redd.it/w562k57baxmf1.png) ([Score: 362, Comments: 70](https://www.reddit.com/r/StableDiffusion/comments/1n7ando/is_it_possible_to_do_this_locally/)): **OP asks whether generating multiple consistent poses of a character from a single illustration (as shown on X using “Nano Banana” and Google’s Gemini) can be done locally with Stable Diffusion. Commenters say it’s feasible but not turnkey: current closed/hosted tools like Nano Banana are praised for superior identity/attribute consistency, while open options (e.g., Kontext, Qwen Image Edit) may enable similar workflows, potentially combined with LoRA training to lock in style/identity.** Top replies argue it’s possible but requires manual effort and tolerance for minor inconsistencies; others suggest trying **Qwen Image Edit** and anticipate rapid open‑source catch‑up, possibly via training LoRAs on outputs from stronger models.
    - Consensus is that “Nano Banana” currently leads on identity/attribute consistency for visual variations (near “almost absolute” character retention), but it’s closed. Several suggest replicating locally by distilling its behavior into open models via LoRA adapters—i.e., train a character/concept LoRA on curated outputs, then run on open backbones like Qwen Image Edit (see Qwen repo: https://github.com/QwenLM) to get similar consistency without cloud inference. This shifts from prompt-only control to parameter-efficient fine-tuning (LoRA: https://arxiv.org/abs/2106.09685).
    - A concrete local pipeline: (1) train a character LoRA from a tightly curated dataset; (2) use ComfyUI’s node graph (https://github.com/comfyanonymous/ComfyUI) with ControlNet pose conditioning to lock structure per shot. Using OpenPose/Posenet controls (ControlNet: https://github.com/lllyasviel/ControlNet; ComfyUI control helpers: https://github.com/Fannovel16/comfyui_controlnet_aux) preserves skeletal/layout while the LoRA preserves identity/accessories, reducing drift in details (e.g., tattoos, braces). This approach trades ease-of-use for reproducibility—each pose typically needs its own control pass.
    - Feasibility notes: “mildly possible with Qwen image edit,” but achieving closed-model-level consistency generally requires supervision beyond prompts. Expect to combine LoRA + per-frame pose control; prompt-only workflows often fail on small, persistent details (color-matched accessories, logos). It’s doable locally, but plan on dataset prep, LoRA training, and per-pose conditioning rather than a single-shot prompt.
- [**does this exist locally? real-time replacement / inpainting?**](https://v.redd.it/0f9walhtiumf1) ([Score: 348, Comments: 72](https://www.reddit.com/r/StableDiffusion/comments/1n70q2o/does_this_exist_locally_realtime_replacement/)): **OP asks whether local, real‑time face replacement/inpainting exists. Top replies state there’s no viable real‑time “VACE + Motion” pipeline; credible demos are offline. DeepFaceLab can do limited “real‑time” after substantial pretraining, but quality is poor (frontal-only bias, artifacts on head turns) and not believable; high‑quality deepfakes still require offline generation. One commenter identifies the showcased clip as theirs, made with “nano banana + Runway Act 2,” confirms it is not real‑time, and links the source ([Instagram](https://www.instagram.com/p/DN1aEuQUD2e/)).** Consensus: current on‑device, instant face swap/inpainting with good multi‑angle fidelity isn’t feasible; social media reels implying otherwise are engagement bait. Another user notes the posted video’s framerate/aspect ratio indicate prerecorded camera footage, not live processing.
    - Multiple commenters note there’s no credible real-time "VACE + Motion" face swap/inpainting pipeline available; reels implying otherwise are likely engagement bait. While **DeepFaceLab** can run "real time" after significant pretraining, commenters report poor fidelity (believable only on frontal shots) and noticeable artifacts on head turns, reinforcing that high-quality multi-angle swaps still require offline generation time rather than instant inference.
    - The original creator clarifies the showcased clip is not real-time and outlines the pipeline as **nano banana** + **Runway Act 2**, with additional details in the source post: https://www.instagram.com/p/DN1aEuQUD2e/. This implies a staged, offline workflow leveraging Runway’s generative tooling rather than a live, on-device inpainting/face-replacement system.
    - A separate observation points out the clip’s framerate and aspect ratio resemble recorded camera footage rather than live output, further indicating non-real-time processing. This aligns with the creator’s explicit note: *"it is NOT REAL time"*.
- [**I asked nano banana to get me into my favorite arcade**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 276, Comments: 33](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **Showcases an AI video pipeline: a real base photo is edited with "nano banana" (image editing), then animated using Kling 2.1 in start/end-frame mode to interpolate motion between keyframes; audio is generated with a "producer AI," and the final cut/color is done in DaVinci Resolve. A step-by-step walkthrough is provided on X/Twitter: https://x.com/techhalla/status/1963333488217919668.** Top comments are largely non-technical praise (e.g., calling it "epic"), with no substantive critique or benchmarking details.
- [**Guys lets just travel back**](https://v.redd.it/pz6ia9umdzmf1) ([Score: 439, Comments: 155](https://www.reddit.com/r/aivideo/comments/1n7kf6h/guys_lets_just_travel_back/)): **OP shares a retro, 1980s‑styled image likely AI‑generated, titled “Guys lets just travel back,” viewable via the preview image (https://preview.redd.it/0mzhs3zegzmf1.png?width=183&format=png&auto=webp&s=290e05f3a160b3548e1b1be76b7d558b1cba0d15) and the original [v.redd.it](http://v.redd.it/) link (https://v.redd.it/pz6ia9umdzmf1), which returns** `403` **without authentication. Top comments flag the anachronism—*“made with AI from*** `2025`***”*—and implicitly distinguish between aesthetic reconstruction and behavioral emulation (e.g., going phoneless) as different approaches to “going back.”** Light debate centers on authenticity: whether AI‑generated retro art undermines the notion of “returning” to an era versus adopting low‑tech habits to approximate the experience.
    - Commenters flag the image as AI-generated (*“made with AI from 2025”*) rather than authentic 1980s media, which would explain stylistic anachronisms in the scene. They also note the subjects’ unrealistically idealized appearance versus period photos, aligning with current diffusion models’ bias toward smoothed, conventionally attractive faces and contemporary makeup/hair cues. Reference image: https://preview.redd.it/0mzhs3zegzmf1.png?width=183&format=png&auto=webp&s=290e05f3a160b3548e1b1be76b7d558b1cba0d15
- [**Guys lets just travel back**](https://v.redd.it/pz6ia9umdzmf1) ([Score: 438, Comments: 157](https://www.reddit.com/r/aivideo/comments/1n7kf6h/guys_lets_just_travel_back/)): **A nostalgia post titled “Guys lets just travel back” features an 80s-themed image (likely AI‑generated per comments) [preview](https://preview.redd.it/0mzhs3zegzmf1.png?width=183&format=png&auto=webp&s=290e05f3a160b3548e1b1be76b7d558b1cba0d15). A linked video endpoint [v.redd.it/pz6ia9umdzmf1](https://v.redd.it/pz6ia9umdzmf1) returns** `HTTP 403 Forbidden` **under Reddit’s anti‑bot controls, implying authentication or a valid client token is required (e.g., [login](https://www.reddit.com/login)).** Top comments note the image looks AI‑generated (“made with AI from 2025”) and play on the 80s nostalgia theme; one suggests behavioral “retro” choices (e.g., go to the mall without a phone) rather than any technical solution.
    - Commenters flag that the image is AI-generated (e.g., “this is made with AI from 2025”) and note it doesn’t match authentic 1980s visuals (“I remember the 80s. It wasn’t this.”). Modern diffusion outputs often over-polish—smooth skin, HDR-like contrast, near-symmetry—and omit period artifacts like film grain/halation, chromatic aberration, lens vignetting, and era-specific color science. To get closer to ‘80s fidelity, practitioners typically add explicit constraints or post-process steps (analog noise, color LUTs emulating Kodachrome/Ektachrome, slight chroma bleed, gate weave, CRT/scanline simulation).
    - The remark “Nobody was actually that pretty back then” maps to model/data bias: web-scale training corpora (heavy on influencer/retouched imagery) push diffusion priors toward idealized attractiveness and contemporary makeup/hair. Without era-specific fine-tunes/LoRAs and strong negative prompts, the sampler gravitates to current beauty standards, producing anachronistically ‘perfect’ faces when asked for retro scenes.
- [**Fruit Beds 🍉🛌🏻↔️**](https://v.redd.it/4vuxgegxwumf1) ([Score: 269, Comments: 40](https://www.reddit.com/r/aivideo/comments/1n72elu/fruit_beds/)): **The post “Fruit Beds 🍉🛌🏻↔️” appears to be a short Reddit-hosted video on [v.redd.it](http://v.redd.it/) ([link](https://v.redd.it/4vuxgegxwumf1)) that currently returns** `HTTP 403 Forbidden` **without authentication; Reddit’s network security page indicates access requires logging in or using API credentials. A still/preview frame is available via a PNG link ([preview](https://preview.redd.it/jyuowx16bxmf1.png?width=1440&format=png&auto=webp&s=c037d679daadd01f03561822b5bc2646ead5f6a5)), suggesting a sequence of fruit-themed “beds,” but no technical context or metadata is provided in-thread.** Top comments are non-technical: one reaction GIF and a question—“What is the last one supposed to be?”—highlighting ambiguity about the final visual; no definitive answer or explanation is provided.
    - Two commenters provide higher‑resolution stills to answer the question about the ambiguous “last one,” linking frames captured from the post: [image 1](https://preview.redd.it/jyuowx16bxmf1.png?width=1440&format=png&auto=webp&s=c037d679daadd01f03561822b5bc2646ead5f6a5) and [image 2](https://preview.redd.it/dp6kr7ddovmf1.png?width=993&format=png&auto=webp&s=8108a005592247c151f97aaa1ad1e0bfff909e29). These higher‑res frames help disambiguate fine details that are obscured at GIF/WebP playback resolutions or due to compression artifacts.
    - The observation about the blanket “spawning into existence” likely stems from a loop/encoding discontinuity: GIF/WebP animations often rely on inter‑frame deltas and disposal methods (`restore to background` or `restore to previous`). If the loop point cuts between non‑keyed frames or the transcoder (e.g., Reddit’s GIF→MP4/WebP pipeline) drops/merges frames, objects can appear to pop in/out between loops; see GIF disposal behavior explained here: https://en.wikipedia.org/wiki/GIF#Disposal_methods.
- [**Fruit Beds 🍉🛌🏻↔️**](https://v.redd.it/4vuxgegxwumf1) ([Score: 265, Comments: 40](https://www.reddit.com/r/aivideo/comments/1n72elu/fruit_beds/)): **Image/meme post titled “Fruit Beds” showing a sequence of bed images themed around fruits; there is no technical content (code, models, or benchmarks). The original Reddit URL is blocked with an HTTP 403 “Forbidden” page requiring [Reddit login](https://www.reddit.com/login/) or a developer token; a [support form](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=21879292693140) is provided. A direct [preview of the last image](https://preview.redd.it/jyuowx16bxmf1.png?width=1440&format=png&auto=webp&s=c037d679daadd01f03561822b5bc2646ead5f6a5) is referenced in comments.** Top comments are non-technical: a GIF reaction, and a question—“What is the last one supposed to be?”—highlighting ambiguity about the final image; another links the preview image above.
- [**I don’t know**](https://i.redd.it/9pbqemdwdzmf1.jpeg) ([Score: 858, Comments: 39](https://www.reddit.com/r/ChatGPT/comments/1n7kg9g/i_dont_know/)): **Meme format contrasting two eras to highlight layperson ignorance about complex systems: modern people can’t explain how computers work, and an ancient pharaoh can’t explain how pyramids were built. No technical details, benchmarks, or implementation discussion—purely humorous commentary on gaps between creators/users and deep understanding of underlying technology or construction methods.** Comments are mostly jokes; one lightly philosophical prompt asks how language works, and another points out the oddity of a time traveler asking questions, but there’s no substantive technical debate.
    - One commenter contrasts the feasibility of a single expert replicating ancient construction (e.g., pyramids) with the impracticality of reproducing modern devices without a vast, distributed knowledge base and tooling. This underscores a shift from logistics- and labor-dominated projects to precision manufacturing with extreme specialization: modern SoCs integrate `~10–20B` transistors and rely on **EUV lithography** and global supply chains (e.g., ASML EUV: https://www.asml.com/en/technology/lithography-principles/euv-lithography; process overview: https://en.wikipedia.org/wiki/Semiconductor_device_fabrication). Even with full schematics, reproduction is constrained by materials science, metrology, and capital equipment (cleanrooms, lithography steppers), illustrating modular yet brittle complexity vs monolithic, robust construction.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Reasoning Benchmarks and Open Models**

- **Pokerbots Pit Stop: Husky Bench Crowns Sonnet**: The **Husky Hold’em Bench** launched the first open-source pokerbots eval, where **Claude 4 Sonnet** led with **57.9%** average profit over **5k+ games** in a 6‑player round‑robin, with **Opus (31.9%)** and **Gemini (31.0%)** trailing, documented at [Husky Hold’em Bench](http://huskybench.com/) and noted by [Nous Research](https://x.com/NousResearch/status/1963371292318749043).
    - The community praised the benchmark’s constraints (Python policies under time/memory caps) and called it *"the first OS pokerbots eval"*, expecting rapid iteration on eval tooling and agent strategies ([huskybench.com](http://huskybench.com/)).
- **Hermes 4 Heats Up: Open Model Leaderboard Flex**: **Hermes 4** (built atop **Qwen3‑14B**) debuted with a newly synthesized post‑training corpus emphasizing verified reasoning traces and larger scale (**~5M samples / ~60B tokens**), while **Hermes‑4‑405B** currently tops the open‑model standings on Husky with **−12.41%** drawdown per the [Nous Research update](https://x.com/NousResearch/status/1963371292318749043).
    - Users shared practical tuning tips (e.g., SillyTavern sampler settings for Think vs Instruct modes) and reported stronger math/code/logic performance with format‑faithful outputs, calling Hermes 4’s hybrid reasoning *"explicit think segments with neutral alignment"* ([huskybench.com](http://huskybench.com/)).
- **Board-Brained Benchmarks Broaden Beyond Poker**: Beyond poker, engineers compared LLMs on classic board games via the [TextArena Leaderboard](https://www.textarena.ai/leaderboard), highlighting chess/Go/connect‑four/shogi/xiangqi ELO as complementary signals to domain‑specific evals.
    - Members advocated multi‑task eval suites to avoid overfitting to single domains, noting that *"diverse, rigorous game evals"* better surface model weaknesses and strategy brittleness ([TextArena Leaderboard](https://www.textarena.ai/leaderboard)).

**2. Kernel Kung Fu and Low-Bit Training**

- **Metal Mania: AI-Generated Kernels Hit 1.87×**: A team reported a **1.87× speedup** by generating low‑level **Metal kernels** directly from **PyTorch**, detailed in [AI‑generated Metal kernels](https://gimletlabs.ai/blog/ai-generated-metal-kernels), and noted that **torch.mps.compile_shader** can directly invoke kernels without a C++ binding.
    - Engineers asked for kernel dumps and suggested submitting a PR to upstream the wins into **PyTorch**, while one maintainer remarked *"a cpp binding is no longer needed"* and flagged correctness checks with [BackendBench](https://github.com/meta-pytorch/BackendBench) (see blog: [gimletlabs.ai](https://gimletlabs.ai/blog/ai-generated-metal-kernels)).
- **TorchAO Tango: Nightlies Trip, MXFP8 Zips**: Developers hit **torchao** nightly breakage due to a **Torch 2.9 vs 2.8** mismatch ([issue #2919](https://github.com/pytorch/ao/issues/2919)), fixed via `pip install torchao==0.13.0 --extra-index-url https://download.pytorch.org/whl/test/cu128`, while **PR #2933** patches an `sm100` flag for MXFP8 ([PR #2933](https://github.com/pytorch/ao/pull/2933)); concurrently, **MXFP8** pretraining recipes and up to **1.28×** speedups were published ([Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027), [PyTorch blog](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)).
    - One user hit an ImportError—*"cannot import name 'mxfp8_cuda'"*—but maintainers clarified the short‑term fix unblocks **NVFP4 inference** and that the impacted kernel is only used for **MXFP8 training** ([issue #2932](https://github.com/pytorch/ao/issues/2932), [PR #2933](https://github.com/pytorch/ao/pull/2933)).
- **Fusion Confusion: torch.compile Meets Triton**: Engineers confirmed **torch.compile** does not fuse ops into user‑defined **Triton** kernels and often creates fusion barriers around specialized ops; a repro and discussion are in this [fusion gist](https://gist.github.com/tohskai/0d0579ef2371dc5a0562d57a7c5361ea).
    - They advised inspecting captured graphs via `TORCH_LOGS="output_code"` and cautioned that the example kernel is *"numerically unstable for large MNK"*, so manual fusion remains the pragmatic choice ([fusion gist](https://gist.github.com/tohskai/0d0579ef2371dc5a0562d57a7c5361ea)).

**3. Agentic Patterns, Program Synthesis, and Eval Infra**

- **Design Bible Drops: 400-Page Agentic Patterns**: A Google engineer released a **400‑page** draft of **Agentic Design Patterns** covering advanced prompting, multi‑agent systems, tool use, and **MCP**, available on [Agentic Design Patterns (Google Doc)](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE) with a companion [NotebookLM](https://notebooklm.google.com/notebook/44bc8819-958d-4050-8431-e7efe2dbd16e).
    - Readers flagged that *"editing access wasn’t disabled"* and worried about accidental edits, while others pre‑ordered the Springer edition and began extracting patterns into their own playbooks ([Agentic Design Patterns](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE)).
- **DSPy Demystified: Clean Splits, MLflow Hooks**: **DSPy** clarified its **train/val/dev/test** split discipline (val for multi‑step plots, test for final eval) to avoid leakage, and members explored prompt lifecycle tracking via [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/) and [mlflow.dspy](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html).
    - A shared repo of [Context Compression Prompt Experiments](https://github.com/Laurian/context-compression-experiments-2508?tab=readme-ov-file#context-compression-prompt-experiments) targeted `dspy.Image` reliability across providers, with volunteers posting failures and patches.
- **Steering Made Simple: LM Eval Harness Bakes It In**: The **LM Eval harness** already supports steered models—activation/residual steering vectors and formatting are documented in [Steered HF Transformers Models](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md#steered-hugging-face-transformers-models).
    - Contributors pointed to the `SteeredModel` docstring for details and opened a PR to steer individual attention heads, saying *"don’t roll your own—use the built-ins"* ([SteeredModel docstring](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/hf_steered.py#L206), [PR #3279](https://github.com/EleutherAI/lm-evaluation-harness/pull/3279)).

**4. Student and Builder Tools Shipping**

- **Comet Class: Perplexity’s Study Mode Launches**: **Perplexity Comet** launched a student‑focused **Study Mode** for schedules, textbooks, and exam prep, with interactive flashcards showcased in [Comet for Students](http://pplx.ai/student).
    - Power users requested that Perplexity roll Study Mode into **Pro** because it’s *"more than just a system prompt"* with custom GUI elements ([announcement thread](https://discord.com/channels/1047197230748151888/1047204950763122820/1412869217950367754)).
- **Projects For All: ChatGPT Gifts Free Users**: **Projects in ChatGPT** now ship to **Free** users on web and Android (iOS rolling out), with per‑project file limits of **5 (Free)**, **25 (Plus)**, and **40 (Pro/Business/Enterprise)** ([OpenAI announcement](https://discord.com/channels/974519864045756446/977259063052234752/1412901459783057440)).
    - Users can customize **colors/icons** and toggle **project‑only memory controls** for tighter context isolation, which teams called clutch for reproducible workflows ([OpenAI announcement](https://discord.com/channels/974519864045756446/977259063052234752/1412901459783057440)).
- **Kimi Codes: Vouchers Gate New Model + Slides**: **Moonshot (Kimi)** is giving away **20 × $20 API vouchers** to test a new coding‑strong model and highlighted a slick **slide generation** feature ([giveaway channel](https://discord.com/channels/1369594130807787570/1412714402284703795)).
    - Admins warned of impersonators—*"If it ain’t yellow, don’t trust it"*—and users asked for a **K2 turbo Coder Pro** plan or a unified tier ([giveaway channel](https://discord.com/channels/1369594130807787570/1412714402284703795)).

**5. Search and Deep Research: Fast, Cheap, and Funded**

- **Exa Accelerates: $85M Series B at $700M Valuation**: **Exa** raised **$85M (Series B)** at a **$700M** valuation led by **Benchmark**, pitching itself as the **search engine for AI**, announced here: [Exa Series B](https://x.com/ExaAILabs/status/1963262700123000947).
    - Deal trackers noted Harmonic flagged the round *two weeks early*, fueling ideas to productize **deal‑flow alerts** ([Exa announcement](https://x.com/ExaAILabs/status/1963262700123000947)).
- **DeepResearch on a Dime: Qwen-2.5 14B > Sonnet-4**: Kyle Corbitt shared an open recipe to fine‑tune **Qwen‑2.5 14B** that beats **Sonnet‑4** on the **DeepResearch** benchmark in ~**30 H200 hours (~$350)** via SFT + GRPO + eval ([training thread](https://xcancel.com/corbtt/status/1962954306078048297)).
    - The resulting model tests competitively with **Gemini 2.5 Pro**, **OpenAI Deep Research**, and **Claude Research**, with devs praising the cost/perf profile as *"pennies to production"* ([training thread](https://xcancel.com/corbtt/status/1962954306078048297)).
- **Code by Command: Claude Code’s 'AI DOS' Moment**: Nikunj Kothari argued **Claude Code** lowers barriers like **DOS** did for PCs—letting non‑coders *"build software by imagination"*—as debated in [this thread](https://x.com/nikunj/status/1963007529082093815).
    - Commenters split on whether we’re *"still in a command‑line era"* or entering an imagination‑constrained phase, with creatives eyeing workflows that collapse prototype cycles ([discussion](https://x.com/nikunj/status/1963007529082093815)).


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Lands on Students' Desktops**: Perplexity AI is now offering **Comet** to students, assisting with schedules, textbooks, and exam prep through a new **Study Mode**, detailed in [this announcement](http://pplx.ai/student).
   - The launch featured [a video demo](https://cdn.discordapp.com/attachments/1047204950763122820/1412869216989610165/comet-students-flashcard.mp4?ex=68b9dc7f&is=68b88aff&hm=9ee0ee1f4d6ebd93c0dc45d35c9ff3f80b3f6d98370cca5474cc1900996b426a&) showing how flashcards can be used within **Comet** for more interactive and efficient studying.
- **Pro Users Demand Study Mode**: Some users are urging Perplexity AI to extend the **Study Mode** feature, currently exclusive to educational accounts, to all **Pro users**.
   - The feature is *more than just a system prompt*, and some users have pointed out that it has associated GUI elements.
- **ChatGPT5 Pro Causes Hilarity**: A user sparked confusion and amusement by mentioning **ChatGPT5 Pro**, mistaking it for **Perplexity’s GPT5** and **GPT5 Thinking models**.
   - Another user clarified that **ChatGPT5 Pro** is exclusive to chatgpt.com, leading to humorous reactions from other members.
- **Comet's Assistant trips**: Users have reported issues with **Comet**, including prolonged loading times for simple sites and an assistant that is not up to par.
   - Speculation arose that the assistant might be leveraging **Sonar**.
- **Perplexity's Filter is Flagging**: Users are criticizing **Perplexity's overzealous censorship**, where even benign historical inquiries such as *“How did Hitler die?”* are being flagged.
   - Concerns have been raised that the overly strict filtering system could lead to unwarranted account bans for studying history or other harmless subjects.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Hype: Too Hot to Handle?**: Members debated the **hype around Gemini 3**, with opinions ranging from potentially overblown expectations to the possibility of Google surprising the industry, even if it only narrowly beats competitors, especially given **OpenAI's ChatGPT5** delays.
   - A member posted [a shrug gif](https://tenor.com/view/shrug-what-huh-will-smith-i-mean-gif-3535627793955785136) reflecting the uncertainty around **Gemini 3's** true impact.
- **LM Arena's Login System Gets a Facelift**: Enthusiasm was expressed for the **new login system on LM Arena**, with one member saying *love the new login system ❤️ been waiting for it 🔥*.
   - The same member proposed a **Google Drive-based chat-storage system** for exporting and analyzing user data as text files, though the idea faced skepticism.
- **LMArena Site Melts Down!**: LMArena experienced a significant **site outage**, causing user frustration and a flood of questions.
   - A moderator, 🍍, assured users that the team was actively working to resolve the issue, directing them to the [FAQ](https://lmarena.ai/faq) for updates.
- **MAI 1 Preview mysteriously goes offline!**: Members reported the sudden malfunction of **Microsoft's LLM MAI 1 Preview**, an LLM some users praised for excellent results.
   - One user commented that *MAI 1 Preview gave the BEST answers in 90% of cases — better than all the others, even ChatGPT-5 high*, leaving the community wondering about its abrupt disappearance.
- **Cloudflare Catches Users in Verification Loop**: Users voiced complaints about frequent **Cloudflare** human verification challenges on LMArena, with one user asking *Does everyone gets cloudflair human verification every two minutes on lmarena website ??*.
   - While VPN usage was suspected as a cause, the issue also appeared on other sites using **Cloudflare**, leading to widespread user annoyance.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Gives Gifts to Free Users**: **Projects in ChatGPT** are now available to **Free users** on web and Android with iOS rolling out soon, along with increased file uploads, now at **5 for Free**, **25 for Plus**, and **40 for Pro/Business/Enterprise**.
   - Users can now customize **colors and icons** for projects, and project-only memory controls are available for more tailored context.
- **Nano Banana Explodes Google Gemini**: Members shared images generated using the prompt **nano banana (gemini-2.5-flash-image-preview)** in the **Gemini app** and **Google Studio**.
   - One user showcased how they turned their coworkers into Vikings.
- **Members Skeptical About Anti-Prompt GPT**: A member shared their [Anti-Prompt Breaching GPT](https://chatgpt.com/g/g-68b8279fca7081919999821ccbd0dc7e-anti-prompt-breaching/) designed to prevent prompt leaking.
   - Others expressed skepticism, saying it might only slow down or make bypassing the protection harder, especially in Custom GPTs and that reliability suffers.
- **Cognitive Mesh AI Learns on Its Own**: A member described designing a cognitive mesh AI that self-adapts and self-learns, growing its understanding over time, similar to utilizing **MoEs**.
   - The AI, built over **3 years**, has short, long, and reflective memory, evolves in its own trajectory, and has developed directive responses based on its input.
- **Thinking Outside the Transformer Box**: Members debated **Liquid Neural Networks** and architectures that merge continuous-time dynamics with symbolic reasoning, such as **neuromorphic chips, photonics, and spintronics**.
   - The consensus was that these innovations don’t rely on brute-force scale but on *rethinking the foundations*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Anthropic API Keys Pass the Vibe Check**: Users have verified that **sk-ant-** keys function correctly with the **Anthropic API** in Cursor, in spite of a UI discrepancy.
   - The community confirmed keys work properly even though the UI might suggest otherwise.
- **Cursor Auto Model Gets Grilled by Users**: Users scrutinized **Cursor's Auto model**, one user reported spending **$200** in less than a week.
   - Feedback indicated that the model's code quality is subpar compared to **Sonnet** or **Gemini**, though potentially superior to **Copilot** due to better prompts, however others suggested guiding summarization manually.
- **Cursor Update Erases Chat Histories, Sparks Panic**: Multiple users reported data loss due to a Cursor update, with one user losing a month's worth of work.
   - The recommended solution of checking local chat history storage locations proved ineffective for some, with chats not found in their usual directory.
- **Fine-Tuning Chatbots: A Penny Saved is a Penny Earned**: A community member sought advice on fine-tuning a model for a web app chatbot, which led to suggestions on utilizing prompt generators.
   - The consensus was to hold off on fine-tuning until a revenue stream is established, treating it as an optimization strategy.
- **Background Agent has a Mid-life Crisis**: A member reported their background agent has frozen and sought a way to transfer its current state to another chat via an API.
   - Specifically, they requested an API method to retrieve a state transfer summary to facilitate moving the agent's current state to a new chat environment.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Nano Banana Powers OpenRouter Discord Bot**: A member created a [Discord bot](https://github.com/mojomast/gemini-nano-banana-discord-bot) leveraging **Nano Banana** through **OpenRouter**.
   - The user clarified that they *vibe coded* the bot, and the source code is available on [GitHub](https://github.com/mojomast/gemini-nano-banana-discord-bot).
- **DeepSeek Models Speak Gibberish**: Some users reported free **DeepSeek models** are generating gibberish, while the paid ones are functioning correctly.
   - Another user inquired about pricing for **DeepSeek 3.1**, with [Synthetic.new](https://synthetic.new/?referral=WofDwy6qyYEKlTi) mentioned at $20/month, although official rates were called a *rip off*.
- **Agent Framework SDKs Get Patched**: Members discussed patching **Agent Framework SDKs** like OpenAI Agent SDK, AI SDK, and LangChain JS with OpenRouter due to non-standard schemas.
   - One member is planning to *roll their own* solution with **BAML** integration, emphasizing that it's *just HTTP anyway*.
- **ChutesAI Subscriber Hit By 429s**: A ChutesAI subscriber is experiencing **429 rate limit errors** and credit issues when using OpenRouter with a BYOK, specifically on Chub.
   - Despite verifying correct API key and private key usage, the problem persists, seeming specific to routing through Chutes on Chub.
- **Google Faces Antitrust Judgement**: A member linked to a [CNBC article](https://www.cnbc.com/2025/09/02/google-antitrust-search-ruling.html) about **Google** facing an **antitrust ruling**.
   - The member commented it was *truly remarkable*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Context Length Curtails Compute**: Users reported that inference speed in **LM Studio** slows down as context length increases, with performance hits becoming noticeable around **20k** context, with some joking about **AI girlfriends**.
   - Members requested a **LM Studio Nightly** release to stay current with *llama.cpp* and other backends.
- **Granite's Gamble with Experts**: The **Granite 4** preview model has **62 experts**, requiring more **VRAM**, but users reported that using non-default configurations led to diminished performance.
   - Some users noted the **Windows auto-upgrader** failing because it was blocked by long path names, requiring manual directory removal to fix.
- **Legacy CPU's Left Behind**: Some users encountered errors in **LM Studio** due to the requirement for **AVX2** instructions, which older CPUs like the **FX8300** do not support.
   - Even with GPU offloading, LM Studio would refuse to run without AVX2 support.
- **Power-Throttling Pursuit for GPUs**: Users discussed limiting GPU power draw using tools like **MSI Afterburner** to manage power consumption while running large **LLM** models, specifically mentioning a new server with **512GB** of **DDR4** RAM.
   - Members evaluated GPU options like the **3060 12GB**, **Titan Xp**, and **RTX A4000 16GB**, with the **3060 12GB** recommended over older cards due to `GDDR6` improvements; a user linked [MSI GeForce RTX 3060 VENTUS](https://www.amazon.co.uk/MSI-GeForce-VENTUS-Gaming-Graphics/dp/B08WHJFYM8).
- **Sizing up VRAM for MoE Models**: A user inquired about the **VRAM** requirements for **MoE** offload with **Qwen3-235B**, specifically if a **1080** could handle it with CPU context offload.
   - Another member estimated around *11+GB* for a 4-bit quantization based on *22B active params*, advising caution due to uncertainty; also, a user pondered the best GPU bandwidth setup for dual CPUs, whether to concentrate GPUs on one CPU or distribute them.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Blender Mastery Beckons**: A member stated they achieved **3D modeling** proficiency in Blender after just **10 hours**, suggesting AI lags in certain creative domains.
   - The quick learning curve was contrasted with AI's limitations, likened to the superficiality of **Uber Eats**.
- **Foundation Models Face Latent Knowledge Lag**: Members suggested that large foundation models insufficiently use **latent knowledge**, viewing it as a missed chance.
   - This deficiency was compared to the progress made in **AIME and IMO problem-solving**.
- **Serial Operations Sparked by Recursion**: Discussions emphasized implementing serial operations with **recursion** because of **Turing Completeness**, which allows adaptive search over large spaces, unlike **CoT/RL**.
   - Focus shifted towards latent space reasoning instead of token space processing, circumventing issues tied to viewing complex tasks as **fixed-point problems**, in line with the theories described in [Adaptive Computation](https://arxiv.org/abs/2509.02522).
- **Diffusion Model's Parallel Token Triumph**: The rise in popularity of **Diffusion LMs** is attributed to the cheaper inference from **parallel token generation** and an improved training objective, avoiding certain biases and failures.
   - While capabilities necessitate serial computation, reverting to **AR**, limitations in this method were acknowledged.
- **LM Eval Embraces Steering Vectors**: A member pointed out that the **LM Eval harness** has built-in support for steering vectors, discouraging custom implementations and directing users to the [documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md#steered-hugging-face-transformers-models).
   - It was clarified that the steering vector implementation manages activations and residuals, with formatting details found in the `SteeredModel` docstring, available [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/hf_steered.py#L206).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Conference set for 2025**: The [Triton Conference 2025](https://aka.ms/tritonconference2025) has been announced, focusing on the **Triton programming language** and related topics.
   - Details about speakers and schedule will be released at a later time.
- **CUDA Pipelining Projects Spark Interest**: Members discussed exploring **CUDA-enabled data pipelines**, suggesting mixing **DALI** with **cuVS** for an optimal setup.
   - The conversation highlighted the need for **MLPerf-like standards** or benchmarks for data pipelines and processing.
- **H100 Hardware Optimizations Exposed**: Discussion on hardware-specific optimizations, particularly for the **H100 GPU**, led to sharing microbenchmarking papers dissecting the **NVIDIA Hopper, Turing T4, and Volta GPU architectures** ([Hopper paper](https://arxiv.org/abs/2501.12084), [Turing T4 paper](https://arxiv.org/abs/1903.07486), [Volta paper](https://arxiv.org/abs/1804.06826)).
   - A user mentioned that for **Blackwell**, they only knew of a simple tech overview released by **NVIDIA**, but considered it pretty good.
- **TorchAO Nightly Builds Spark Version Conflicts**: Members flagged that **`torchao` nightly builds** are breaking due to a version mismatch between the **Torch version** it was built against (**2.9**) and the **Torch version** they were trying to import it with (**2.8**), and suggest checking [issue #2919](https://github.com/pytorch/ao/issues/2919).
   - The fix was to use `pip install torchao==0.13.0 --extra-index-url https://download.pytorch.org/whl/test/cu128`.
- **AI Codegen Gives Metal Kernels Rocket Boost**: A team achieved a **1.87X speedup** by going straight from **PyTorch** to low-level **Metal kernels** with AI codegen, described in their [blog post](https://gimletlabs.ai/blog/ai-generated-metal-kernels).
   - A member pointed out that a **cpp binding** is no longer needed, as one can use **torch.mps.compile_shader** to directly invoke kernels. They also suggested submitting a PR with the kernels, as any performance gains would benefit **PyTorch** users.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Deepseek API, a Little Slow, but Free**: A member discovered a free **Deepseek API**, noting its usefulness despite being somewhat slow.
   - The user appreciated the resource because *it's free*.
- **M4 Macbook Pro Stumbles with Llama 3.2 Vision**: A user with a **Macbook Pro M4** (24 GB RAM) couldn't run **Llama 3.2 vision 11B**, reporting that it utilized 20 GB of memory without producing output.
   - Another user suggested exploring [quantized versions](https://pytorch.org/docs/stable/quantization.html) such as **Q4** or reducing the context length to resolve the issue.
- **Anthropic Triples and Ties Up Copyright**: Members noted that **Anthropic** tripled in size in approximately 5 months based on [this tweet](https://x.com/AnthropicAI/status/1962909472017281518), growing from ~60B to 180B, and also settled their [copyright case](https://en.wikipedia.org/wiki/Copyright_infringement) out of court.
   - The connection between the investment announcement and the settlement was noted as *really fascinating*, although settlement terms are not yet public.
- **Chinese AI Models Cut the Gaslighting**: A member observed that Chinese AI models like **Qwen** tend to exhibit less *gaslighting* behavior.
   - Another member praised **Qwen** for providing *ideas of what can be wrong* and adhering to formats effectively.
- **Datatune Agents Enable Data Transformations**: A new release of [Datatune Agents](https://github.com/vitalops/datatune) now enables data transformations at a per-row level using **natural language** prompts, with key features including row-level **map()** and **filter()** operations and Dask DataFrames support.
   - Compatibility includes multiple **LLM** backends like **OpenAI**, **Azure**, and **Ollama** via **LiteLLM**, with Datatune optimizing **tokens** and **cost** through explicit control over sent columns, automatic batching, and metadata handling.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Automated weapons spark debate!**: Members hotly debated the ethics and practical implications of **automated weapons**, with some suggesting they could minimize harm compared to human soldiers, while others voiced concerns about potential human rights abuses.
   - Some argued the bigger fear is governments abusing them for human rights violations, which is already happening with bombs, nukes and drones.
- **US public transit a missed opportunity?**: Members discussed the state of public transportation in the US, describing it as unsafe and humiliating, missing the opportunity to reduce accidents and improve urban mobility.
   - It was suggested that accidents could be reduced by **90%** if humans could only drive when in the proper state of mind.
- **Cheap Drones could swarm!**: The potential for **drones** combined with **trucks** to be used as cheap attacking options was discussed, highlighting the need for a security framework to deal with non-state actors.
   - One proposed solution involved governments focusing on banning chemicals required to make drones.
- **Mamba's state matrix faces flak!**: A member critiqued **Mamba's** fixed transition matrix for not replicating true state machines, and for potential issues preserving context, citing [The Illusion of State in State-Space Models](https://arxiv.org/abs/2509.01494).
   - Suggestions for cures included adding a **nonlinearity** between state transitions or making state transition matrices dependent on the input, as done in [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951).
- **Ladybird rises as Chrome contender!**: A new [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) browser called **Ladybird** is under development as a potential alternative to Chrome, currently available for **Linux** and **Mac OS**.
   - The development of **Ladybird** is driven by a commitment to the principles of **Free and Open Source Software (FOSS)**, ensuring transparency, community involvement, and freedom of modification.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Google Engineer Drops Agentic Design Patterns Tome**: A Google engineer released a **400-page** draft of *Agentic Design Patterns* covering advanced prompting, multi-agent systems, tool use, and MCP, available on [Google Docs](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE) and for pre-order as a Springer edition.
   - The community shared links to the doc, [NotebookLM](https://notebooklm.google.com/notebook/44bc8819-958d-4050-8431-e7efe2dbd16e), and Amazon pre-order, but some noted the doc’s editing access wasn’t disabled, leading to concerns about alterations.
- **Claude Code Declared 'AI DOS' Moment**: Nikunj Kothari argues that **Claude Code** is a watershed moment—like DOS was for PCs—because it collapses technical barriers and lets non-coders build software by imagination alone, as noted in [this tweet](https://x.com/nikunj/status/1963007529082093815?s=46).
   - Commenters debated whether we’re still in a command-line era, how creatives can harness it, and if the real bottleneck is now imagination rather than coding skill.
- **Compute Arms Race Sparks Debate on Efficiency**: Discussion highlights massive compute spending by **OpenAI** & **Anthropic**—**$13B** pre-pays to secure GPUs/energy—while observers question diminishing returns and unsustainable power/water usage, stemming from [this X post](https://xcancel.com/theo/status/1963016066944401848).
   - The thread swings between doomsayers predicting a funding crash and optimists betting on small-model efficiency or breakthrough algorithms to obsolete the mega-cluster strategy.
- **Open-Source Recipe Trains Deep Research Agent for Pennies**: Kyle Corbitt shares a recipe using open-source tools that lets developers train a **Qwen-2.5 14B model** to surpass **Sonnet-4** on the DeepResearch benchmark in just **30 H200 hours (~$350)**, based on [this tweet](https://xcancel.com/corbtt/status/1962954306078048297).
   - The process includes SFT for basic skills, GRPO for utilization, and benchmark evaluation, producing a model competitive with **Gemini 2.5 Pro**, **OpenAI Deep Research**, and **Claude Research**.
- **Exa Raises $85M Series B at $700M Valuation**: Exa announced an **$85M** Series B raise at a **$700M** valuation led by Benchmark, positioning itself as the search engine for AI according to [this tweet](https://xcancel.com/ExaAILabs/status/1963262700123000947).
   - Harmonic’s system flagged the round two weeks in advance, prompting discussion about turning deal flow alerts into a product.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude 4 Sonnet Wins Pokerbots Title**: The **Husky Hold’em Bench** debuted as the first OS pokerbots eval, with **Claude 4 Sonnet** leading the competition at **57.9%** average profit over 5k+ games against other models in a 6-player round-robin format.
   - The benchmark challenges models to implement policies in Python under time and memory constraints, and is documented on [huskybench.com](http://huskybench.com); **Opus** came in second (**31.9%**) and **Gemini** trailed in third place (**31.0%**).
- **Hermes 4 Powers Up Reasoning**: **Hermes 4** is the next generation of Hermes trained on top of **Qwen3-14B**, featuring a newly synthesized post-training corpus that emphasizes verified reasoning traces.
   - The update highlights include improvements in math, code, STEM, logic, creativity, and format-faithful outputs, while preserving general assistant quality and broadly neutral alignment; training increased from **1M samples and 1.2B tokens to ~5M samples / ~60B tokens**.
- **SillyTavern Likes Hermes**: Members discussed leveraging **SillyTavern** for roleplay, highlighting its surprising math and coding capabilities.
   - For **Hermes-4-14B** based on **Qwen-3**, the recommended sampler settings are **temp: 0.6, temp-k: 20, temp-p: 85** for Thinking-Mode and **temp: 0.7, temp-k: 20-40, temp-p: 95** for Instruct-Mode; additionally, use **ChatML** for 14B, **Llama 3 Instruct** for 70B and 405B.
- **Research Doc Plea**: A member requested assistance with writing a research document and case study on **Generative UI** and **AI-first interaction patterns**.
   - The author is focusing on **transformation design** and the **business** implications, looking for guidance to kickstart their project and gain a better understanding of the subject matter.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Data Leakage Worries Dissipated**: Concerns about potential test set data leakage in **DSPy** when reusing a training set were addressed by clarifying that optimizers use distinct training and validation sets, with a separate test set for final evaluation.
   - The discussion highlighted that **multi-step plots** use the **valset**, while final results are reported on the **testset** to prevent leakage and overfitting.
- **DSPy Data Splits Decoded**: **DSPy** employs **four distinct data splits**: **train** (for few-shot examples), **val** (for validation), **dev** (for human iteration), and **test** (for final evaluation).
   - The community emphasized the importance of using the **valset** for multi-step plots and the **testset** for final results to avoid leakage or overfitting issues.
- **MLflow Hearts DSPy**: A user explored integrating **MLflow** with **DSPy** to capture prompts, referencing [MLflow's prompt registry features](https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/) and the existence of [mlflow.dspy](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html).
   - The user plans to experiment and report back on the integration of **MLflow** and **DSPy** for prompt management.
- **Context Compression Critiques Commence!**: A member shared [Context Compression Prompt Experiments](https://github.com/Laurian/context-compression-experiments-2508?tab=readme-ov-file#context-compression-prompt-experiments) aiming to enhance `dspy.Image` reliability.
   - This project focuses on investigating and refining **context compression** methods to improve the performance of `dspy.Image` across various providers.
- **`dspy.Image` Needs Reliability Tweaks**: A user initiated a task to refine the reliability of `dspy.Image` for certain providers, detailed in [this Discord thread](https://discord.com/channels/1161519468141355160/1211406131398840390/1412601392446836799).
   - Follow-up discussions involved sharing images and exploring potential solutions to address the reliability issues.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Kicks Off Voucher Giveaway**: The **Moonshot Team** is giving away **20 × $20 API vouchers** to test their new model with *crazy coding powers*, accessible only via a voucher.
   - Users can participate by reacting in the [#giveaway channel](https://discord.com/channels/1369594130807787570/1412714402284703795) before **8AM Beijing Time**.
- **Kimi's Coding Prowess Powers Slide Generation**: A user praised the recently released **slide generation feature** and the accompanying coding enhancements.
   - They look forward to **Kimi** enabling even more professional task handling with this update, saying it delivers the coding improvements they were hoping for.
- **Request for Kimi K2 turbo Coder Pro Plan Surfaces**: A user suggested a **Kimi K2 turbo Coder Pro plan** as a product idea.
   - Another user suggested **Kimi** should just make it a unified plan.
- **Moonshot Warns of Scammers**: A warning was issued regarding scammers, advising users that legitimate **Kimi** team members will have a **yellow role color** in the server.
   - The announcement explicitly states, *If it ain’t yellow, don’t trust it*, cautioning users to verify the authenticity of any direct messages received.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo SIMD makes Rust AVX Burn Brain Cells**: A member finds that **Mojo** makes **SIMD** enjoyable, whereas manual **AVX** in **Rust** is mentally taxing and asked about a standard `net/http` style module for **Mojo**.
   - The consensus favors a lean standard library, with community-driven efforts like [lightbug_http](https://builds.modular.com/packages/lightbug_http) for an **HTTP library**.
- **Mojo Powers Fast Binary Search Engine**: A member built a **binary search engine** in Mojo capable of crunching **~50k queries/sec** over **2M docs** single cored by parallelizing across **SIMD** lanes.
   - The member anticipates adding **HTTP support** to enable search-as-you-type functionality.
- **Mojo Runs on GTX 1080 After Patch**: **Mojo GPU** functions are now confirmed to run correctly on a **GTX 1080** after a patch in the latest nightly, and they are adding changelog entries and listing support for **Pascal GPUs**, along with `sm_60` support for the **Tesla P100**.
   - A forthcoming internal patch will lower the **Turing** architecture limit, potentially giving broader compute capability than **PyTorch** on older GPUs.
- **Max Backend for Torch Gains Traction**: Efforts are underway to dedicate more time to the **max backend for torch**, aiming for `torch.ones((2, 2), device="max_device")` to operate on a wider range of GPUs compared to the latest **CUDA**.
   - The team plans to engage with **Modular** team members to assess the engineering soundness of the project.
- **Discord is Best Way to Reach Modular Team**: A member suggested that the most effective way to contact the **Modular** team is by directly pinging them on Discord.
   - Given their flooded email inboxes, using Discord is a reliable alternative, with a recommendation to reach out to a specific new **Modular** team member if other channels fail.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Basic Plan Blocks Permanent Website Deployment**: A user asked if the **basic plan** allows for permanent website deployment, and was informed that *"it does not"*.
   - This clarifies limitations for users considering website hosting options.
- **Grok Declared: Tool, Not Agent**: A user asserted that **Grok** is a tool, not an agent, highlighting a crucial distinction in its functional classification.
   - This correction was made in response to conversational context, implying potential misinterpretations of **Grok's** capabilities.
- **Manus Exonerated in Grok Comparison**: A user stated they were not comparing **Grok** with **Manus**, indicating a possible misunderstanding in the discussion.
   - This clarification suggests that the perceived comparison was tangential or nonexistent within the conversation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Dual 7900 XTX Cards Trigger Crash**: A member reported sudden crashes at peak performance using dual **7900 XTX** cards with **HIPC** code on kernel **6.8**, as supported on the ROCm site.
   - The user expressed concerns about **multi-GPU training** issues and sought solutions to prevent **GPU crashes**.
- **Pyrender Test Volunteering Needed**: A member inquired about potential testers for **pyrender** on the kernel dataset.
   - Details regarding specific testing parameters or objectives were not provided.
- **Linearizer Test Gets Dumb-er**: A member updated `test_linearizer_dumb` ([link to GitHub PR](https://github.com/tinygrad/tinygrad/pull/11968/files)) and proposed updating other uops in the tests to match the new format.
   - The new format is allegedly *more readable and updatable* and that member offered to fix the uops tests later.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **KYC neuters Streaming**: OpenAI requires **KYC verification** to use its image model and **GPT-5 streaming** features.
   - It's possible to use **GPT-5 without KYC**, but only without streaming enabled.
- **Codex as Aider Clone?**: A user expressed frustration with **GPT-5's** processing time for simple requests and missing **thinking streaming**.
   - Another member asked what is liked about **Codex** better than **Aider**, mentioning that **Claude Code** was originally designed to clone **Aider**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Enrollment Process Simplified**: A member asked if receiving a **Google Forms** signup confirmation meant they were qualified for the LLM Agents MOOC.
   - Another member clarified that *everyone is welcome* and **there isn't a qualification process**.
- **Google Forms System Healthy**: Many users are getting instant email confirmation after submitting the **Google Form**.
   - This indicates the forms system is functioning correctly.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1412869217950367754)** (1 messages): 

> `Comet for students` 


- **Comet Lands on Students' Desktops**: Perplexity AI is now offering **Comet** to students, helping them manage schedules, textbooks, and exam preparation with a new **Study Mode**, as seen in [this announcement](http://pplx.ai/student).
   - The announcement included [a video demonstration](https://cdn.discordapp.com/attachments/1047204950763122820/1412869216989610165/comet-students-flashcard.mp4?ex=68b9dc7f&is=68b88aff&hm=9ee0ee1f4d6ebd93c0dc45d35c9ff3f80b3f6d98370cca5474cc1900996b426a&) showcasing the functionalities.
- **Flashcard Video**: A promotional video attached in the announcement demonstrates how students can utilize flashcards within Comet to prepare for exams.
   - This feature aims to make studying more interactive and efficient for students using the platform.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1412648840594591764)** (1157 messages🔥🔥🔥): 

> `Perplexity Pro, GPT-5, Comet Browser, Filter Overreach` 


- **Pro Users Want Study Mode**: Some users are requesting that the Study Mode feature be rolled out to all **Pro users**, not just those with **educational accounts**.
   - It was noted that Study Mode is *more than just a system prompt* and has associated GUI elements.
- **Did Someone Say ChatGPT5 Pro?**: A user caused confusion and amusement by claiming to have **ChatGPT5 Pro** when referencing **Perplexity’s GPT5** and **GPT5 Thinking models**.
   - Another user pointed out that **ChatGPT5 Pro** only exists on chatgpt.com, leading to humorous responses.
- **Comet's Assistant struggles**: Some users mentioned issues with Comet, noting that it sometimes **takes years to load a simple site** and that the **assistant** seems good but not as good as other options.
   - One user speculated that the assistant might be using **Sonar**.
- **Filter Too Extreme, Getting Censors**: Users discussed the **overzealous censorship** in Perplexity, noting that even **historical queries** like *“How did Hitler die?”* are being flagged.
   - It was suggested that the filtering is too harsh and might lead to accounts being banned for simply studying history or engaging in harmless topics.
- **Labs Web App Feature Gets Confused**: Some users were confused by the models available on **sim.ai**, questioning why **PPLX** in **sim.ai** has **Mistral** and **Sonnet 3**.
   - Others suggested it might be a bug or that the platform uses **PPLX’s API** for web search and another model to summarize results.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1412741537921765479)** (3 messages): 

> `Perplexity Browser Claims` 


- **Perplexity Claims are shared**: Members shared [Perplexity Browser Claims](https://perplexity.ai/browser/claim/F0FLK6D1R7).
   - Other claims that were shared include [this link](https://perplexity.ai/browser/claim/KRZHTIO3PC) and [another one](https://perplexity.ai/browser/claim/P9U74312JA).
- **Additional Perplexity Claims Surface**: More Perplexity Browser Claims were posted in the channel.
   - These claims provide a way to share browsing sessions and research results.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

breakingclover: I’m interested, I tried to send you a message but it looks like you have them off!
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1412648386053669014)** (895 messages🔥🔥🔥): 

> `Gemini 3 Hype, LM Arena Login System, LM Arena staying open, LM Arena Site Outage, LM Arena FAQ` 


- **Gemini 3 Hype might be overblown**: Members discussed the **hype around Gemini 3**, with some suggesting that even if it doesn't represent a huge leap in AI development, it only needs to outpace competitors to be successful.
   - One member referenced **OpenAI's struggles with ChatGPT5** as an indicator of industry-wide challenges, while others noted Google's greater resources might lead to surprises, with a link to a [shrug gif](https://tenor.com/view/shrug-what-huh-will-smith-i-mean-gif-3535627793955785136).
- **LM Arena's New Login System gets love**: One member expressed their enthusiasm for the new login system, saying *love the new login system ❤️ been waiting for it 🔥*.
   - The same member suggested implementing a **chat-storage system via Google Drive**, allowing users to export and analyze their data as text files, though others were skeptical.
- **Site Outage triggers user craziness**: A period of significant disruption occurred where LMArena experienced an **outage**, rendering the site inaccessible and prompting numerous user complaints and questions.
   - A moderator, 🍍, acknowledged the situation and assured users that the team was actively working on a fix and would update the  [FAQ](https://lmarena.ai/faq).
- **MAI 1's Mysterious Malfunction**: Members discussed the sudden malfunction of **Microsoft's LLM MAI 1 Preview**, which some users had found to give excellent results.
   - One user reported that *MAI 1 Preview gave the BEST answers in 90% of cases — better than all the others, even ChatGPT-5 high*, but there was no clear explanation for its disappearance.
- **A Cloudflare Cloudflared Catastrophe!**: Several users complained about frequent **Cloudflare** human verification challenges on LMArena, with one saying *Does everyone gets cloudflair human verification every two minutes on lmarena website ??*.
   - Some suspected this was due to using VPNs, while others noted similar issues on other sites using **Cloudflare**, leading to general frustration with the service.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1412845620502335509)** (1 messages): 

> `Video Generation Contest` 


- **Video Generation Contest Slices Through August!**: The August Video Generation Contest has **9 days left** for submissions, with the theme **Slice!** 🔪, focusing on *oddly satisfying* and safe cross-section cuts.
   - To participate, use `/video` to generate two videos in the video arena channels and forward the generated message to the <#1406999853732593807> channel, with examples [here](https://discord.com/channels/1340554757349179412/1397655695150682194/1406324882085511319) & [here](https://discord.com/channels/1340554757349179412/1397655695150682194/1405764767020482580).
- **Act Fast: Video Submissions Closing Soon!**: Don't miss your chance to showcase your video generation skills!
   - The August Video Generation Contest deadline is quickly approaching, ensure your entry is submitted to the designated channel to be in the running.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1412901459783057440)** (1 messages): 

> `ChatGPT Free Projects, Larger File Uploads, Project Customization` 


- **Free ChatGPT Projects Now Available!**: **Projects in ChatGPT** are now available to **Free users** on web and Android, with iOS rolling out soon.
- **File Uploads Get a Sizeable Bump!**: The update includes larger file uploads per project: **5 for Free**, **25 for Plus**, and **40 for Pro/Business/Enterprise**.
- **ChatGPT gets a Customizable Touch!**: Users now have the option to select **colors and icons** for more customization, along with **project-only memory controls** for more tailored context.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1412651549502476329)** (434 messages🔥🔥🔥): 

> `AI Residency Program, Nano Banana in Photoshop, AI Car Game with ChatGPT, Cognitive Mesh AI Design, Liquid Neural Networks` 


- **OpenAI Residency Program: When Will It Open?**: A member inquired about the opening of the **OpenAI Residency Program**, but community members, including moderators, only know when **OpenAI** provides the information.
   - They suggested applying for open jobs at [OpenAI's career page](https://openai.com/careers/search/tx) in the meantime.
- **Nano Banana Powers Up Gemini Images**: Members discussed the use of **Nano** in image generation, with one user sharing how they turned their coworkers into Vikings using the **Gemini app** and **Google Studio**.
   - Members shared generated images using prompt **nano banana (gemini-2.5-flash-image-preview)**.
- **Code Car Game with ChatGPT**: A member created a car game with **ChatGPT** using approximately **700-1000 lines of code**.
   - They imagined the potential of software and coding in the future when **AI** writes better, faster, and more efficient code than humans and offered to share their code.
- **Cognitive Mesh AI Evolves with Self-Learning**: A member described designing a cognitive mesh AI that self-adapts and self-learns, growing its understanding over time, similar to utilizing **MoEs**.
   - The AI, built over **3 years**, has short, long, and reflective memory, evolves in its own trajectory, and has developed directive responses based on its input.
- **Liquid Neural Networks Challenge the AI Paradigm**: Members talked about **LNNs** not needing those resources at all. Think about concepts like **liquid neural networks, neuromorphic chips, photonics, spintronics**, or even hybrid architectures that merge continuous-time dynamics with symbolic reasoning.
   - These don’t rely on brute-force scale they rely on *rethinking the foundations*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1412773852152463360)** (36 messages🔥): 

> `Prompt Leaking, API discussion location, Context Contamination, Custom GPTs Reliability, Prompt Priority Level` 


- **Discussion Diverts to APIs Channel**: A member suggested that **API operation discussions** should occur in the dedicated [APIs channel](https://discord.com/channels/974519864045756446/103756117828673946) rather than the prompt engineering channel.
- **Anti-Prompt Leaking GPT Appears**: A member shared their [Anti-Prompt Breaching GPT](https://chatgpt.com/g/g-68b8279fca7081919999821ccbd0dc7e-anti-prompt-breaching/) designed to prevent prompt leaking, which sparked discussions about its effectiveness.
   - Others expressed skepticism, pointing out that it might only slow down or make bypassing the protection harder, especially in Custom GPTs.
- **Prompt Hiding Compromises Reliability**: Members debated the trade-offs between **hiding prompts** and maintaining **model reliability**, noting that longer, more complex contexts can reduce effectiveness.
   - The conclusion was that OpenAI has been training models to refuse prompt extraction and that the **reliability should always come first**.
- **Context Contamination Concerns Arise**: A discussion highlighted that using the same model for both **generating responses** and **evaluating guardrails** is suboptimal due to potential context contamination.
   - It was suggested that different models with different instructions should be used for each task, especially considering that Custom GPT instructions might conflict with prompt-hiding directives.
- **Temperature and Top-P Settings Influence Model Behavior**: A member summarized how different combinations of **temperature** and **top-p settings** affect model behavior.
   - They noted that *Low temp / Low top-p* lead to maximum consistency and minimal variation, while *High temp / High top-p* provide maximum creativity and diversity.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1412773852152463360)** (36 messages🔥): 

> `Anti-Prompt Leaks, GPT Reliability, Prompt Engineering, Agent Instructions, Model Temperatue` 


- **New Anti-Prompt Leaking GPT appears**: A user built a **GPT** to prevent prompt leaking, available at [chatgpt.com](https://chatgpt.com/g/g-68b8279fca7081919999821ccbd0dc7e-anti-prompt-breaching/).
   - Others noted that this method *reduces reliability while offering little benefit* and is easily bypassed, especially within a Custom GPT.
- **Prompt Priority and Reliability Clash**: A user shared code about prompt priority levels, which led to discussions about the use of additional tokens and complexity, because this reduces the "real" prompt's reliability.
   - The discussion concluded that there's *no prompt solution that won't reduce reliability* and that **OpenAI** has been training these refusals directly into their models.
- **Guardrails Must Run in Agentic Ways**: One user pointed out that guardrails must run in more agentic ways, with **models** and **agents** doing each check.
   - He specified that it *should be different models, and with different instructions*.
- **Avoiding Context Contamination**: A user stated that, to avoid context contamination, instructions should not be used for hiding prompts.
   - He notes that the standard "*You are a helpful assistant*" instruction already conflicts with the goal of preventing prompt leakage.
- **Temperature And Top-P Discussed**: Users discussed various strategies involving temperature and top-p settings, specifically: **low temp/low top-p** for maximum consistency, minimal variation; **low temp/high top-p** for consistent style, varied vocabulary; **high temp/low top-p** for creative within focused scope; and **high temp/high top-p** for maximum creativity and diversity.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1412651507240931329)** (284 messages🔥🔥): 

> `Anthropic API, Cursor Auto model quality, Cursor chat history loss, Fine-tuning recommendations for chatbot, Gemini 2.5 Reasoning` 


- **Anthropic API Keys Work Despite UI Mismatch**: Users confirmed that **sk-ant-** keys are accepted and fully functional for the **Anthropic API** in Cursor, despite a UI mismatch indicating otherwise.
- **Cursor's Auto Model Faces Scrutiny**: Members discussed the quality of Cursor's **Auto model**, with one user reporting spending **$200** in 4-5 days, expressing that the code quality is worse than **Sonnet** or **Gemini** but potentially better than **Copilot** due to better prompts and tools.
   - Another member suggested that it’s better to guide summarization manually rather than relying on the Auto model when the context is full.
- **Users Report Cursor Chat History Loss After Update**: Several users reported losing their chat history after updating Cursor, with one user losing a month's worth of project work.
   - One user recommended checking the location where **Cursor** stores chat history files, though another confirmed that the chats were not in the location they are normally stored.
- **Community Seeks Fine-Tuning Advice for Chatbot**: A user requested recommendations for fine-tuning a model for a web app chatbot, leading to a discussion on using prompt generators and reproducible schemas.
   - It was advised to avoid fine-tuning until generating a revenue stream, and treat it as an optimization to use smaller models.
- **Exploits and LLMs**: A user reported that LLMs suck at exploit dev by default, which led to a conversation about hacking and a user saying they hear *kaching* when someone talks about it.
   - A user replied, *Last time it happened we were selling cheats for Call of duty during covid and all the stimulus checks stimulated us LOL*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1412855599900790824)** (2 messages): 

> `Background agent transfer, API state transfer summary` 


- **Agent Freeze Requires State Transfer**: A member reported their background agent has frozen and is seeking a way to transfer it to another chat.
   - They inquired about obtaining a state transfer summary, possibly via an API, as the agent is integrated into a website.
- **API Request for State Transfer Summary**: The member specifically asked if there's an API method to retrieve a state transfer summary for their background agent.
   - This would facilitate moving the agent's current state to a new chat environment after the freeze.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1412938600936509490)** (2 messages): 

> `Nano Banana Discord Bot, vibe coded bot` 


- **Vibe Coded Bot drops Nano Banana**: A member shared a [Discord bot](https://github.com/mojomast/gemini-nano-banana-discord-bot) that uses **Nano Banana** over **OpenRouter**.
   - The user clarified that they *vibe coded* the bot.
- **Nano Banana powers Discord Bot**: A Discord bot was created leveraging **Nano Banana** through **OpenRouter** to post content.
   - The bot's source code is available on [GitHub](https://github.com/mojomast/gemini-nano-banana-discord-bot) for anyone interested in exploring or contributing.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1412651481626185810)** (230 messages🔥🔥): 

> `DeepSeek models, Claude-sonnet-4 problems, DeepSeek 3.1 cheapest price, Submodel.ai promotion, OpenRouter billing questions` 


- **DeepSeek Free Models Speak Gibberish**: Some users report that free **DeepSeek models** are generating gibberish, while the paid ones are functioning correctly.
- **DeepSeek 3.1 Pricing**: A user asked about the cheapest and most stable source for **DeepSeek 3.1**, with [Synthetic.new](https://synthetic.new/?referral=WofDwy6qyYEKlTi) being mentioned at $20/month, but another user called the official rates a "rip off."
   - Another user suggested [Submodel](https://submodel.ai/), but a mod cautioned that they need to wait in line like everyone else who wants to be on the platform.
- **Agent Framework SDK Experiences Shared**: Members discussed their experience with **Agent Framework SDKs** like OpenAI Agent SDK, AI SDK, and LangChain JS with OpenRouter, noting that most require patching due to non-standard schemas from various providers.
   - One member plans to *roll their own* solution with **BAML** integration, emphasizing that it's *just HTTP anyway*.
- **ChutesAI Subscriber Faces 429 Errors**: A ChutesAI subscriber is experiencing **429 rate limit errors** and credit issues when using OpenRouter with a BYOK, specifically on Chub but not on JanitorAI.
   - Despite verifying the correct API key and private key usage, the problem persists, and the user has tried various solutions to no avail; the issue seems to be specific to routing through Chutes on Chub.
- **Gemini-2.5 Rate Limits and Credit Drain Debated**: Users are encountering **429 rate limit errors** with **Gemini-2.5-flash-image-preview:free**, even after paying to increase rate limits, and suspect it's an OpenRouter issue.
   - One user reported **OpenRouter drained all their credits** due to a bug with **Gemini image**, and a mod confirmed refunds will be issued soon.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1412769726211297340)** (5 messages): 

> `Google Antitrust, Yahoo Chrome, Minor UI suggestion` 


- **Google Faces Antitrust Ruling**: A member linked to a [CNBC article](https://www.cnbc.com/2025/09/02/google-antitrust-search-ruling.html) about **Google** facing an **antitrust ruling**.
   - The member commented it was *truly remarkable*.
- **Yahoo Considering Buying Chrome**: A member expressed a morbid fascination with the hypothetical scenario of **Yahoo** acquiring **Chrome**.
   - They mentioned that *a truly sick part of me wanted to see the purchase of Chrome by Yahoo! play out*.
- **Request to improve UI Padding**: A member suggested a minor UI tweak, specifically suggesting to *remove the padding-bottom on the outer box, and move it to the inner box*.
   - The member suggests that this change would prevent the **scroll wheel** from obscuring the **text**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1412652189431889991)** (156 messages🔥🔥): 

> `LM Studio 4k context, LM Studio Nightly, AI Girlfriend goon chamber, LM Studio auto naming stories, Granite 4 memory` 


- **Context Length Slows Down Inference**: Inference slows down as context grows, with performance deltas becoming more obvious as context utilisation goes up to **20k**.
   - However, some users comically compared this context to an *AI girlfriend*.
- **Bleeding Edge LM Studio**: Users are requesting an **LM Studio Nightly** release for backends, especially for *llama.cpp*, to stay on the bleeding edge.
   - It was suggested that the NSFW content should be moved to 4chan.
- **Granite Experts**: The **Granite 4** preview can use **62 experts**, meaning more memory (VRAM) is required.
   - However, using anything but the default experts results in worse results in MOST cases, according to members.
- **LM Studio struggles Windows Auto-Upgrader**: The **Windows auto-upgrader** is unable to update install until manual removal of existing LM Studio due to long path names.
   - The workaround is to manually remove the problematic directory (`C:\Users\[USERNAME]\AppData\Local\Programs\LM Studio\resources\app\.webpack\bin\extensions\backends\vendor\_amphibian\cpython3.11-win-x86@2\Lib\site-packages\pkg_resources\tests\data\my-test-package_unpacked-egg\my_test_package-1.0-py3.7.egg\EGG-INFO`) and then run the downloaded installer.
- **No Love for Legacy CPUs**: Users are running into errors because LM Studio requires **AVX2** instructions, and older CPUs like the **FX8300** don't support them.
   - As a result, LM Studio refuses to run, even if the user intends to offload computations to the GPU, some users have written their own solutions in response.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1412714259904856094)** (77 messages🔥🔥): 

> `Limiting GPU Power Draw, DDR4 Server for LLMs, GPU Recommendations (3060, Titan Xp, A4000), MoE offload for Qwen3-235B, Multi-GPU bandwidth with dual CPUs` 


- **Capping GPU Power Consumption**: Users discussed limiting power draw in GPU drivers, suggesting **MSI Afterburner** as a tool to control power per GPU.
   - One user was planning to limit power consumption due to running very large **LLM** models on a new server with **512GB** of **DDR4** RAM.
- **Scoping out Budget GPU Boost**: Members discussed various GPU options for a server, including used **3060 12GB** cards (~$250-300), **Titan Xp** cards (~$200-250), and considering **RTX A4000 16GB** cards ($700 each).
   - A suggestion was made to consider the **3060 12GB** over older cards due to `GDDR6` improvements, linking [this MSI GeForce RTX 3060 VENTUS](https://www.amazon.co.uk/MSI-GeForce-VENTUS-Gaming-Graphics/dp/B08WHJFYM8).
- **Guesstimating MoE VRAM Consumption**: A user inquired about the **VRAM** needed for **MoE** offload for **Qwen3-235B**, wondering if they could run it with CPU context offload using a **1080**.
   - Another member estimated *11+GB* for a 4-bit quantization, based on *22B active params*, but expressed uncertainty about the exact mechanics.
- **Routing Multi-GPU Bandwidth Bonanza**: A user pondered how to best load GPU bandwidth between dual CPUs, considering whether to have 2x GPUs on one CPU or 1x GPU on each PCIe slot.
   - Another member suggested that putting all GPUs on one PCI root complex would *"probably" reduce latency for GPU-GPU traffic*, sharing [a DigitalOcean tutorial on splitting LLMs across multiple GPUs](https://www.digitalocean.com/community/tutorials/splitting-llms-across-multiple-gpus#tools-and-libraries-for-splitting-llm-to-multiple-gpus).
- **Intel Arc Pro B50 Bargain Find**: One member stumbled upon the [Intel Arc Pro B50 16GB](https://www.newegg.com/intel-arc-pro-b50-16gb-workstation-sff-graphics-card/p/N82E16814883007) workstation GPU listed for $350.
   - The user exclaimed they knew where that card will be going.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1412672961344962594)** (14 messages🔥): 

> `3D modeling, Latent knowledge in large foundation models, SPAR applications` 


- **Blender skills blossom after 10 hours**: A member mentioned they can **3D model** in Blender after only **10 hours** of learning, implying current AI isn't up to par.
   - They compared it to **Uber Eats** to emphasize the superficiality of some AI applications.
- **Foundation Models Underutilize Latent Knowledge**: A member suggested large foundation models are underutilizing **latent knowledge**, calling it *low hanging fruit*.
   - They compared current progress to achievements in **AIME and IMO problem-solving** in the last year.
- **SPAR Application Status Updates**: Two members inquired about the results of their **SPAR applications**.
   - Both confirmed they have not received any updates yet.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1412701269688057878)** (128 messages🔥🔥): 

> `Normalizing Flows, CoT/RL limitations, Diffusion Models, Adaptive Computation, Fixed-Point Problems` 


- ****Recursion Reigns**: Turing Completeness calls for Serial Operations!**: Discussion posited that serial operations should occur via **recursion** due to **Turing Completeness**, suggesting an architecture that could adaptively search large spaces, unlike **CoT/RL** which were dismissed as mere band-aids.
   - The discussion then pivoted toward reasoning in the latent space over token space to avoid the pitfalls of viewing hard tasks as **fixed-point problems** with concrete ground truths, further clarifying views on [Adaptive Computation](https://arxiv.org/abs/2509.02522).
- ****Diffusion Dissection**: Cheaper Inference or Training Triumph?**: The cheaper inference from **parallel generation of tokens** was cited as the core reason for **Diffusion LMs** hype, while its better training objective helps avoid certain failures and biases.
   - Members stated that capabilities require some serial compute thus falling back to **AR**, but members agreed there were limitations of this approach.
- ****Latent Logic**: Tool Use Transcends Token Space!**: Members debated on tool use in latent space vs token space.
   - A member noted that you can imagine *dirty solutions like a tool head that consumes the latent*, which might be less stable.
- ****Brainy Business**: Animal-Level Learning Before Human Heights?**: The economics of intelligence were questioned, suggesting that substantial animal brain level learning might be necessary before achieving human brain level learning, wondering if *we augment existing LLMs with dog level RL I don't think they get much better but I think it costs a ton*.
   - It was analogized to distinct modes of personal reasoning, one involving explicit linguistic articulation and another operating in a less definite space of ideas, with one member describing this second mode as *globs of semantic groups put together* that are shuffled around until they coalesce.
- ****Sweep Stakes**: Tuned for Longer Training with Weight Decay!**: Debate ensued around competing results from [two papers](https://arxiv.org/abs/2509.02046) and [another paper](https://arxiv.org/abs/2509.01440), discussing disparities in the performance of the **Muon Optimizer** and the importance of proper sweeping and tuning duration with **weight decay**.
   - Sweeping with a subset of data might suggest no **WD** is optimal, but tuning for longer reveals its benefit: *compute = luck*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1412655495512195072)** (34 messages🔥): 

> `LM Eval Harness, MMLU Task Configuration, Steering Vectors Implementation, Attention Heads Steering, Model's Response Recording` 


- **Debugging MMLU with LM Eval Harness**: A member sought help implementing a function with forward hooks to add a **steering vector** to different attention heads while running **MMLU** with the **LM Eval harness**; the sequence length decreased consistently, causing confusion, and posted config details and relevant code [here](https://paste.centos.org/view/01923fd8).
   - Another member pointed out that the harness inputs `[:,-1]` tokens to calculate the logprobs of the prefill, truncating the sequence, and suggested modifying this line to remove the truncation if needed, as seen [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/huggingface.py#L1223).
- **LM Eval Steering Vector Support**: A member highlighted that **LM Eval harness** already supports steering vectors, advising against manual implementation and linking to relevant documentation, which can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md#steered-hugging-face-transformers-models).
   - It was also mentioned that the steering vector implementation works on activations as well as residuals, and that in-depth explanations are provided on how to format the steering vector data in the `SteeredModel` docstring, available [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/hf_steered.py#L206).
- **Attention Heads Steering Imminent**: A member announced a pull request to add support for steering individual attention heads, which can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/pull/3279).
   - The goal is to evaluate models steered with pre-prepared steering vectors saved in a `.pt` file or a reference to an SAE/transcoder feature used as a steering vector.
- **Model's Response: Forward Pass vs. Generate**: In the specified task configuration, it was clarified that for generate tasks like **gsm8k** and **minerva_math**, the model's response is recorded through a generate call.
   - However, for multiple-choice tasks, the model's response is recorded through a standard forward pass.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1412670705669111823)** (21 messages🔥): 

> `PyTorch Conference, CUDA data pipelines, MLPerf benchmarks, Hardware optimizations, NVIDIA Hopper Architecture` 


- **Triton Conference announced for 2025**: A member shared a link to the [Triton Conference 2025](https://aka.ms/tritonconference2025), which focuses on the **Triton** programming language and related topics.
- **Cool CUDA Pipeline Projects Beckon**: A member inquired about cool toy projects to explore **CUDA-enabled data pipelines**, specifically looking for well-defined problems without logistical issues.
   - Another member suggested mixing **DALI** with **cuVS** for an optimal setup and also wondered about **MLPerf-like standards** or benchmarks for data pipelines and processing.
- **NVIDIA Hardware Optimizations Uncovered**: A member sought resources for understanding hardware-specific optimizations, particularly for the **H100 GPU**.
   - Another member shared links to microbenchmarking papers dissecting the **NVIDIA Hopper, Turing T4, and Volta GPU architectures** ([Hopper paper](https://arxiv.org/abs/2501.12084), [Turing T4 paper](https://arxiv.org/abs/1903.07486), [Volta paper](https://arxiv.org/abs/1804.06826)).
- **NVIDIA Ampere Architecture Explored**: In a discussion about GPU-specific features, a member shared a link to an **NVIDIA** on-demand session about the **Ampere architecture** ([Ampere session](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s33322/)).
- **Blackwell Architecture Tech Overview Teased**: During the discussion about hardware optimizations, a user mentioned that for **Blackwell**, they only knew of a simple tech overview released by **NVIDIA**, but considered it pretty good.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1412843221339341021)** (3 messages): 

> `Microsoft Teams Meeting, Meeting Details` 


- **Microsoft Teams Meeting Scheduled**: A meeting is starting in 5 minutes on **Microsoft Teams**, and a user shared a [join link](https://teams.microsoft.com/l/meetup-join/19%3ameeting_Mjg5ZDk4YWEtMTI1My00MjNjLTk0MWUtYTFhZTQ4YjUwZjcw%40thread.v2/0?context=%7b%22Tid%22%3a%2246c98d88-e344-4ed4-8496-4ed7712e255d%22%2c%22Oid%22%3a%22f318a2d8-b05f-4329-819f-c0d8a870e7dc%22%7d).
   - The message included a meeting ID (**283 039 414 385 5**) and passcode (**XW6c3ZC2**).
- **Dial-in Details Provided**: Dial-in details were shared for the **Microsoft Teams** meeting, including a Vancouver local number (+1 778-800-9740,,819312747#) and a phone conference ID (819 312 747#).
   - Information was also given for joining on a video conferencing device using a tenant key (**teams@conf.intel.com**) and video ID (**118 771 827 4**).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1412920949837140009)** (5 messages): 

> `Intra-device Parallelization, CUDA-level FSDP, Register and Shared Memory Usage, NVCC Half-Precision Optimization` 


- **Intra-Device Parallelization: FSDP at CUDA Level**: A member inquired about a name for parallelizing the load of weights for subsequent layers alongside computations of the current layer, similar to **FSDP**, but at the **CUDA level** within a single device, potentially using **memcpy_async**.
   - They linked the [CUDA documentation on collectives](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#collectives-cg-memcpy-async) and an [NVIDIA blog post on data movement](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/) to illustrate the concept.
- **Deep Dive into Register and Shared Memory**: A member sought clarification on how register and shared memory usage function on an **SM (Streaming Multiprocessor)**, questioning if developers can explicitly control or subdivide these resources to pack more values by reducing precision.
   - They specifically asked if **half-precision types (16-bit)** could be used such that two values occupy a single **32-bit register**, or if this is solely managed by the compiler and hardware using intrinsics like **__half2**.
- **NVCC: Vectorizing Halves to Save Registers?**: A member inquired whether **nvcc** automatically converts two half-precision floating-point numbers into a vector of half to save registers.
   - This optimization would potentially improve memory usage by compacting data structures.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1412828768158093443)** (12 messages🔥): 

> `Torch.compile and Triton Kernels, Kernel Fusion with Torch.compile, Triton OP Registration` 


- ****Torch.compile** Doesn't Fuse Into **Triton** Kernels?**: A member inquired whether `torch.compile` fuses surrounding code into custom **Triton** kernels, questioning the fusion capabilities around specialized ops.
   - Another member responded that it *depends on the captured graph* and suggested using `TORCH_LOGS="output_code"` to inspect the generated code, but ultimately confirmed that **torch.compile does not fuse operations into user-defined triton kernels**.
- ****Triton OP** Registration Inquiry**: During a discussion about `torch.compile` behavior, a member asked if a kernel should be registered with `triton_op` and `wrap_triton` for fusion to occur.
   - A [Gist was shared](https://gist.github.com/tohskai/0d0579ef2371dc5a0562d57a7c5361ea) to test kernel fusion, but it was noted that relying too much on the compiler for fusions is not advisable, as the example is numerically unstable for large MNK.
- **Fusion Barrier Forms in **Torch.compile****: A member suggested that `torch.compile` creates a fusion barrier before and after a specialized op with its own **Triton** kernel, resulting in multiple kernel launches.
   - Even when manual fusion is possible, the discussion leaned towards the compiler not automatically fusing simple primitive ops surrounding specialized ops with **Triton** kernels.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1412671070187814997)** (1 messages): 

> `Sony Computer Vision Job Posting` 


- ****Sony** Eyes **Computer Vision** Whiz**: **Sony** is hiring for a **Computer Vision** role, as advertised in [this LinkedIn post](https://www.linkedin.com/posts/hey-abhijit-more_sony-is-hiring-for-the-role-of-computer-vision-activity-7368853761902948352-KlOC?utm_source=share&utm_medium=member_android&rcm=ACoAAChsL2YBvDWOl6QVX3upuusUZAdkdAiylvc).
- ****AI Engineer** Needed**: The job description is looking for someone to help develop AI.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1412771315349328005)** (20 messages🔥): 

> `TorchAO Installation, cu128 image, torch2.8, MXFP8 Training` 


- **TorchAO Nightly Builds Mismatch Torch Versions**: Members reported **`torchao` nightly builds breaking** due to a mismatch between the **Torch version** it was built against (**2.9**) and the **Torch version** they were trying to import it with (**2.8**), and suggest checking [issue #2919](https://github.com/pytorch/ao/issues/2919).
   - The fix was to use `pip install torchao==0.13.0 --extra-index-url https://download.pytorch.org/whl/test/cu128`.
- **Troubles Installing TorchAO 0.13.0 Prerelease**: A member encountered an `ImportError` when trying to import `NVFP4InferenceConfig` from `torchao.prototype.mx_formats` in the **0.13.0 prerelease** with error message *cannot import name 'mxfp8_cuda' from 'torchao.prototype'*, and determined installing from source worked.
   - The root cause was a missing build flag for **sm100**, and a fix is in the works via [PR #2933](https://github.com/pytorch/ao/pull/2933) and [issue #2932](https://github.com/pytorch/ao/issues/2932).
- **NVFP4 Inference Users Unblocked**: A fix is incoming in the short term for **NVFP4 inference** in version **0.13.0**.
   - It was reported that users will be unblocked by the short term fix as the kernel in question is only used for **MXFP8 training**.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1412858609099866173)** (1 messages): 

> `Project Contributions` 


- **Suggest Project Contribution Channel**: A member suggested that others look at a specific project contribution channel, <#1373414141427191809>.
- **Additional Placeholder Topic**: This is a placeholder to meet the minimum items requirement.
   - Further details can be added as needed.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1412855523078049834)** (1 messages): 

> `Gaudi 2, Gaudi performance` 


- **Gaudi 2 still champ!**: A member said that **Gaudi 2** is still a great product, especially regarding performance.
- **Gaudi Expert Available**: A member who works on **Gaudi** offered to answer questions about it.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1412866384115007641)** (6 messages): 

> `AI-Generated Metal Kernels, PyTorch to Low-Level Kernels, MPS Eager & torch.compile Backend, Kernel LLM Generation, BackendBench Correctness Checking` 


- **AI Codegen Zaps Speedup to Metal Kernels**: A team achieved a **1.87X speedup** by going straight from **PyTorch** to low-level **Metal kernels** with AI codegen, described in their [blog post](https://gimletlabs.ai/blog/ai-generated-metal-kernels).
- **Sharing Generated Kernels for Scrutiny**: A member requested a folder with all the generated kernels, as another member maintaining the **MPS eager** and **torch.compile backend** offered to share the kernels and timing results, inviting feedback on potential correctness issues.
   - He also mentioned his work on [kernel LLM generation](https://github.com/meta-pytorch/BackendBench) to support all **PyTorch operators**.
- **BackendBench Probes Correctness**: A member noted potential correctness issues and suggested using **BackendBench** for more thorough checking than **KernelBench**.
   - The team responded that they used KernelBench but were excited about the general direction.
- **Suspicions Swirl around Speedup Claims**: Some are skeptical about the claims of **1000x speed up**, suggesting it might stem from a lack of synchronization at the end of benchmark blog.
   - The team is asked to propose a PR: any perf gains will be beneficial for PyTorch users
- **Bypassing CPP Binding for Kernel Invocation**: A member pointed out that a **cpp binding** is no longer needed, as one can use **torch.mps.compile_shader** to directly invoke kernels.
   - They also suggested submitting a PR with the kernels, as any performance gains would benefit **PyTorch** users.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1412667163088523344)** (1 messages): 

> `B200 attention kernel` 


- **Seeking functional B200 attention kernel**: A member inquired about a **B200 attention kernel** they wanted to test, but found it broken on the main branch.
   - They asked if there was a specific branch or patch available to try it out.
- **B200 Kernel Troubles**: A user reported encountering issues with the **B200 attention kernel** on the main branch.
   - They are seeking a working version, either as a separate branch or a patch.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1412649677802832002)** (11 messages🔥): 

> `MI300x8, amd-all2all leaderboard` 


- **MI300x8 All2All Records Crushed**: A member achieved **first place** on the `amd-all2all` leaderboard for **MI300x8** with a submission id `34854` at **1361 µs**.
- **MI300x8 Race Tightens**: Multiple submissions were made to the `amd-all2all` leaderboard for **MI300x8**, with times ranging from **2.55 ms** to **22.0 ms**.
- **MI300x8 Leaderboard Competition**: One user secured **third place** on **MI300x8** with **2.57 ms** on the `amd-all2all` leaderboard.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1412716232821444719)** (7 messages): 

> `Game Feedback, Request for Guidance` 


- **Positive Game Feedback Provided**: A player indicated they enjoyed the game after playing it for **3 hours**.
   - They simply stated, *"nice game."
- **Player Seeks Guidance**: A player is looking for advice on **what to do next** within the game.
   - They are seeking direction after already spending a significant amount of time playing.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1412810345168240710)** (17 messages🔥): 

> `Submitting HIP Kernels, iris library, UI exit code info` 


- **HIP kernels are welcome!**: The submitted solution can include Python files, and HIP kernels (with an additional build script) that will be wrapped by a Python API, as exemplified by the [reference kernels](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/fp8-mm/template-hip.py).
- **Amd person to add iris library!**: The *iris* library might be added to the environment and a member has already forwarded the request to the AMD infra manager.
- **UI will get exit code**: The UI will be updated to provide more info for exit code info.
- **How can I access the hardware for the competition?**: Members can make submissions without SSH access; see [the docs page](https://gpu-mode.github.io/discord-cluster-manager/docs/intro/).


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1412776361310814251)** (4 messages): 

> `CUTLASS 2.x interfaces, Hopper F8 performance, kind::mxf4nvf4.block_scale vs kind::mxf8f6f4.block_scale, Github bug reports` 


- **CUTLASS 2.x deprecated for better interfaces**: Members noted that **CUTLASS 2.x interfaces** are largely not used anymore and that **3.x and especially 4.x** have much better docs.
   - One user stated that version 4.x is *4x faster compared to hopper f8*.
- **mxf4nvf4 smokes mxf8f6f4**: `kind::mxf4nvf4.block_scale` is **4x**, but the question was related to doing mxfp4 via `kind::mxf8f6f4.block_scale`.
   - One member asked if `mxf4nvf4` is **2x faster** than `mxf8f6f4` for the exact same mxfp4 inputs, or am I missing something?
- **Github issues bug reports**: A member requested that the user file a bug on **Github issues**.
   - No specific reason was given.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1412692973795409920)** (20 messages🔥): 

> `Multi-GPU Development, Distributed Kernels, AMD Challenge 2025, NVLink vs PCIe, Fused Kernels` 


- ****Multi-GPU Nirvana**: Cloud vs Local Setup**: Discussion revolves around setting up a multi-GPU development environment, with the cloud (e.g., Google Cloud's N1 series with 4x Tesla T4 GPUs via [Google Cloud Compute Docs](https://cloud.google.com/compute/docs/gpus#general_comparison_chart)) being preferred over a local setup to avoid compatibility issues.
   - The goal is to develop multi-GPU algorithms without immediately needing top-tier performance, focusing on understanding tools for mapping mathematical algorithms to hardware, and accessing machines via SSH from a Macbook.
- ****NVLink vs PCIe**: A Tale of Two Interconnects**: **NVLink** and **PCIe** are logically similar (memory accessed via loads/stores and dereferencing pointers) but have different features; a significant NVLink-specific feature is **Multicast memory**.
   - The user emphasized focusing on single node setups to exclude network concerns, noting that they are interested in NVLink/NVSwitch but PCIe is acceptable in the short term.
- ****Fused Kernels**: The Holy Grail of Multi-GPU**: The user expresses interest in implementing **distributed fused kernels** to fully utilize multiple GPUs, particularly for large matrix multiplications.
   - This involves combining computation and communication within the same kernel, differentiating it from separately handling kernels and communication, with an example of fusing a matrix multiplication with an AllGather or AllReduce operation.
- ****NCCL/NVSHMEM APIs**: Abstraction vs Fine-Grained Control**: Implementing toy versions of **DP/TP/PP/Zero** can be done from scratch with P2P loads/stores or using **NCCL/NVSHMEM APIs**, depending on the desired level of control and tolerance for library calls.
   - The choice depends on how fine-grained the work needs to be and how tolerant the implementation is of library calls; **NCCL** knows how to select kernels and settings based on device connections, simplifying the user interface to `ncclSend` and `ncclRecv`.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1412840091059093504)** (1 messages): 

> `MXFP8 pre-training, TorchAO MXFP8, Crusoe B200 Cluster` 


- **MXFP8 Recipes Unveiled for LLM Pre-training**: A new paper, [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027), has been published, providing guidance on using **MXFP8** for pre-training Large Language Models.
- **TorchAO MXFP8 and Crusoe B200 speed up pre-training**: PyTorch announced [accelerated 2K-scale pre-training](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/) by up to **1.28x** with **TorchAO MXFP8** and **TorchTitan** on the **Crusoe B200 Cluster**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1412652089628430467)** (93 messages🔥🔥): 

> `Deepseek API, Llama 3.2 vision 11B on Macbook Pro M4, Quantized Models, HF Spaces, Python Learning` 


- **Deepseek API Found, but Slow**: A member found a free **Deepseek API**, noting it's a little slow but useful.
   - They expressed satisfaction because *it's free*.
- **M4 Macbook Pro Fails to Run Llama 3.2 Vision Model**: A user with a Macbook Pro M4 with 24 GB RAM reported failing to run **Llama 3.2 vision 11B**, with the system using 20 GB of memory without output.
   - Another user suggested it might be offloaded to swap memory and recommended trying [quantized versions](https://pytorch.org/docs/stable/quantization.html) or lower context length such as **Q4**.
- **Anthropic Grew Quickly**: In response to this [tweet](https://x.com/AnthropicAI/status/1962909472017281518), members noted that **Anthropic** tripled in size in approximately 5 months, growing from ~60B to 180B.
   - Another user jokingly said *yk what else is huge*.
- **Anthropic Settled Copyright Case**: Members discussed that **Anthropic** settled their [copyright case](https://en.wikipedia.org/wiki/Copyright_infringement) out of court and they're going to announce the terms publicly soon-ish.
   - While the settlement amount is still unknown, one member mentioned that the announcement of this **investment** and the settlement of this case are clearly related, *really fascinating*.
- **Chinese AI Models Less Gaslighting**: A member noted that Chinese AI models tend to **gaslight less**, pointing to **Qwen** as an example.
   - Another member said they are a big fan of **Qwen** for providing them with *ideas of what can be wrong*, and that it follows the format well.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1412849707948179567)** (3 messages): 

> `link request, language studies` 


- **Link request goes unanswered**: Two members requested a link from another member without specifying the content of the link.
   - The request was not fulfilled within the given message history.
- **English is the only language**: A member asked another member if they studied Japanese.
   - The other member responded that they only study English.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1412678606278234173)** (2 messages): 

> `Datatune Agents release, DeepResearch AI Agents, Token Optimization` 


- **Datatune Agents Released**: A new release of [Datatune Agents](https://github.com/vitalops/datatune) enables data transformations at a per-row level, preserving contextual understanding using **natural language** prompts.
   - Key features include row-level **map()** and **filter()** operations, Dask DataFrames support for scalability, and compatibility with multiple LLM backends like **OpenAI**, **Azure**, and **Ollama** via **LiteLLM**.
- **DeepResearch AI Agents Treasure Hunt**: A new post was published about [DeepResearch AI Agents](https://medium.com/@jenlindadsouza/deepresearch-ai-agents-on-a-literature-treasure-hunt-c590de681258) and how they dive deep into research papers ensuring broad coverage, balancing **depth** and **breadth**.
   - The agents code is available on [GitHub](https://github.com/sciknoworg/deep-research), and the author is seeking feedback from the community.
- **Datatune optimizes Tokens & Cost**: Datatune gives explicit control over which columns are sent to the LLM, reducing token usage and API cost.
   - This is achieved through `input_fields` to send only relevant columns, automatic batching, metadata handling, and support for setting **tokens-per-minute** and **requests-per-minute** limits, defaulting to known model limits like **GPT-3.5**.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1412701017375768606)** (2 messages): 

> `Detectron2 setup, Automated test cases` 


- **Detectron2 Setup Sought**: A member asked for help setting up **Detectron2** on their local PC and converting it into a wheel format.
   - No solutions were offered in the provided messages.
- **Computer Use Functionality Explored**: A member inquired about experiences using *computer use functionality* for discovering and automating test cases.
   - They specifically asked about any limitations encountered during the process.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

cakiki: <@596574356327628850> please don't cross-post
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1412749343772901428)** (66 messages🔥🔥): 

> `Automated weapons, Public transportation in the US, Drones as a cheap attacking option, DeepMind's potential with massive funding, Quantum Physics of AGI` 


- **Automated weapons debate rages!**: Members discuss the ethics and practicalities of **automated weapons**, with some arguing that they could reduce harm compared to human soldiers, while others fear human rights abuses.
   - Some argue the fear is governments abusing them for human rights violations, which is already happening with bombs, nukes and drones.
- **American Public Transit: a Missed Opportunity?**: The members discuss how public transportation in the US is unsafe and humiliating, highlighting a missed opportunity to reduce accidents and improve urban mobility.
   - The conversation suggests that if humans could only drive when in the proper state of mind, accidents could be reduced by **90%**.
- **Drone swarms as a security threat**: Members discuss the potential for **drones** combined with **trucks** to be used as cheap attacking options, necessitating a security framework to deal with non-state actors.
   - One suggests governments focus on banning chemicals required to make drones.
- **Waymo Driving the Subscription Economy**: Members discuss autonomous driving subscription-style payments, like for **Waymo** and envision hybrid approaches.
   - One member stated *I am still sometimes dreaming of a hybrid that can provide the best of both worlds. I think that is where most of the tech bros are coming from that just end up reinventing trains or busses every few months.*
- **Quantum Physics: the key to AGI?**: Members shared a [YouTube video](https://youtu.be/IVA2bK9qjzE?si=cqGZwCC3p8-jG-RC) discussing whether existing AI architectures can achieve AGI/ASI from a quantum physicist POV.
   - A member stated *wonder where could DeepMind get to if they threw 7 trillion dollars onto this problem.*


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1412819071468044348)** (16 messages🔥): 

> `Mamba Weakness, Online learning, Neuromorphic architecture, Bad ML learning resources` 


- **Mamba's Fixed Transition Matrix Faces Flak**: A member critiqued **Mamba's** fixed transition matrix, noting that it cannot replicate true state machines and may not preserve contextually relevant information, referencing the paper [The Illusion of State in State-Space Models](https://arxiv.org/abs/2509.01494).
   - They suggested cures such as adding a **nonlinearity** between state transitions or making state transition matrices dependent on the input, as done in [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951).
- **AI's Achilles Heel: Absence of Online Learning**: A member claimed that *the single biggest issue in AI is lack of online learning*.
- **Neuroscience and Neuromorphic Architecture Navigated**: In response to a discussion about general intelligence, a member suggested exploring neuroscience and neuromorphic architecture via [Artem Kirsanov's YouTube channel](https://www.youtube.com/@ArtemKirsanov/videos) and [deepsouth.org.au](https://www.deepsouth.org.au).
- **ML Learning Resources Mocked as Mostly Mediocre**: One member thinks *most ML learning resources are just as bad because they are a product of the time when nobody knew better*.
   - They compared the quality of early ML resources to *all the terrible php tutorials or terrible c learning books etc.*


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1412719244495294494)** (3 messages): 

> `Ladybird Browser, FOSS browser alternative to Chrome` 


- **Ladybird: Chrome Alternative in the Works**: A new [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) browser called **Ladybird** is in development as a potential alternative to Chrome.
   - Currently available for **Linux** and **Mac OS**, a member speculates that a Windows port may be developed if it gains popularity.
- **FOSS Ethos Drives New Browser**: The development of the **Ladybird** browser is driven by a commitment to the principles of **Free and Open Source Software (FOSS)**.
   - This ensures transparency, community involvement, and the freedom to modify and distribute the software, differentiating it from proprietary browsers.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1412665387736105012)** (59 messages🔥🔥): 

> `Agentic Design Patterns, Claude Code, AI Compute Arms Race, Open Source Deep Research Agent, Exa Series B` 


- **Google Engineer Drops Agentic Design Patterns Tome**: A Google engineer released a **400-page** draft of *Agentic Design Patterns* covering advanced prompting, multi-agent systems, tool use, and MCP, available on [Google Docs](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE) and for pre-order as a Springer edition.
   - The community shared links to the doc, [NotebookLM](https://notebooklm.google.com/notebook/44bc8819-958d-4050-8431-e7efe2dbd16e), and Amazon pre-order, but some noted the doc’s editing access wasn’t disabled, leading to concerns about alterations.
- **Claude Code Declared "AI DOS" Moment**: Nikunj Kothari argues that **Claude Code** is a watershed moment—like DOS was for PCs—because it collapses technical barriers and lets non-coders build software by imagination alone, as noted in [this tweet](https://x.com/nikunj/status/1963007529082093815?s=46).
   - Commenters debated whether we’re still in a command-line era, how creatives can harness it, and if the real bottleneck is now imagination rather than coding skill.
- **Compute Arms Race Sparks Debate on Efficiency**: Discussion highlights massive compute spending by **OpenAI** & **Anthropic**—**$13B** pre-pays to secure GPUs/energy—while observers question diminishing returns and unsustainable power/water usage, stemming from [this X post](https://xcancel.com/theo/status/1963016066944401848).
   - The thread swings between doomsayers predicting a funding crash and optimists betting on small-model efficiency or breakthrough algorithms to obsolete the mega-cluster strategy.
- **Open-Source Recipe Trains Deep Research Agent for Pennies**: Kyle Corbitt shares a recipe using open-source tools that lets developers train a **Qwen-2.5 14B model** to surpass **Sonnet-4** on the DeepResearch benchmark in just **30 H200 hours (~$350)**, based on [this tweet](https://xcancel.com/corbtt/status/1962954306078048297).
   - The process includes SFT for basic skills, GRPO for utilization, and benchmark evaluation, producing a model competitive with **Gemini 2.5 Pro**, **OpenAI Deep Research**, and **Claude Research**.
- **Exa raises $85M Series B at $700M Valuation**: Exa announced an **$85M** Series B raise at a **$700M** valuation led by Benchmark, positioning itself as the search engine for AI according to [this tweet](https://xcancel.com/ExaAILabs/status/1963262700123000947).
   - Harmonic’s system flagged the round two weeks in advance, prompting discussion about turning deal flow alerts into a product.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1412793236099043419)** (5 messages): 

> `AI-generated worlds, Immersive storytelling, Higgsfield platform, Future of Sci-Fi` 


- **AI-Generated Worlds Spark Excitement**: Justine Moore shared an **AI-generated world** created by aim_not_here using **Higgsfield**, igniting excited discussion around this emerging [immersive storytelling format](https://xcancel.com/venturetwins/status/1963222552215449801).
   - Commenters praised it as a *window into creators' minds* and predicted major **sci-fi** innovations ahead.
- **Fiction is discovered!**: Following the release of the AI-generated world, some commenters remarked on tech bros rediscovering *fiction*.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1412931668314030151)** (1 messages): 

> `Husky Hold’em Bench, OS pokerbots eval, Claude 4 Sonnet, Hermes 4 405B` 


- **Husky Hold’em Bench Debuts as First OS Pokerbots Eval**: The **Husky Hold’em Bench** has been introduced as the first OS pokerbots eval, challenging models to implement policies in Python under time and memory constraints, as documented on [huskybench.com](http://huskybench.com).
- **Claude 4 Sonnet Wins Pokerbots Comp**: **Claude 4 Sonnet** leads the competition with a **57.9%** average profit over 5k+ games, outperforming other models in a 6-player round-robin format.
   - **Opus** came in second (**31.9%**) and **Gemini** trailed in third place (**31.0%**).
- **Hermes 4 405B is Leading Open Model**: The leading open model currently is **Hermes 4 405B** at **-12.41%** according to [this tweet](https://x.com/NousResearch/status/1963371292318749043).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1412655540768604306)** (37 messages🔥): 

> `Hermes 4 vs other models, SillyTavern and Hermes, Prompt Compliance, LLM game benchmarks` 


- **Hermes 4, Next-Gen Reasoning Champ!**: **Hermes 4** is the next generation of Hermes trained on top of **Qwen3-14B**, with training highlights including a newly synthesized post-training corpus emphasizing verified reasoning traces, massive improvements in math, code, STEM, logic, creativity, and format-faithful outputs, while preserving general assistant quality and broadly neutral alignment.
   - Training highlights include a dataset size increase from **1M samples and 1.2B tokens to ~5M samples / ~60B tokens** and a hybrid reasoning mode with explicit think segments.
- **SillyTavern Savvy with Hermes**: Members discussed leveraging **SillyTavern** for roleplay and vibes, noting its surprising math and coding capabilities.
   - It was recommended that because **Hermes-4-14B** is based on **Qwen-3**, the sampler settings should be similar, using **temp: 0.6, temp-k: 20, temp-p: 85** for Thinking-Mode and **temp: 0.7, temp-k: 20-40, temp-p: 95** for Instruct-Mode; additionally, use **ChatML** for 14B, **Llama 3 Instruct** for 70B and 405B.
- **Podcast Ponders Prompt Compliance**: A member suggested a podcast topic on how **Hermes 4's** internal dialog deals with **prompt compliance**, particularly breaking down problems like *"develop a super villain character"* that most LLMs refuse.
   - Another member suggested [contacting a specific user](https://discordapp.com/users/265269014148808716) who did all the **CoT explorations** in the paper.
- **LLMs Ace Game Benchmarks?**: Members inquired about the state of **LLM game benchmarks** for chess/go/connect4/shogi/xiangqi elo.
   - One member shared a link to a leaderboard at [TextArena.ai](https://www.textarena.ai/leaderboard) that includes chess benchmarks.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1412771955979063376)** (1 messages): 

> `Generative UI, AI-First Interaction Patterns, Transformation Design, Business Applications` 


- **Generative UI/AI Research Doc Needed**: A member requested assistance with writing a research document and case study on **Generative UI** and **AI-first interaction patterns**.
- **Help requested on Generative UI and AI**: A member needed help getting started with a research document about **Generative UI** and **AI**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1412771955979063376)** (1 messages): 

> `Generative UI Research, AI-First Interaction Patterns, Transformation Design Case Study, Business Impact of Generative UI` 


- **Generative UI Research Document Needed**: A member is writing a research document and case study on **Generative UI** and **AI-first interaction patterns**.
   - They need help getting started and understanding **transformation design** and the **business** implications.
- **Call for Help on Generative UI and AI-First Interactions**: A member is seeking assistance with their research document and case study focused on **Generative UI**, **AI-first interaction patterns**, **transformation design**, and **business impact**.
   - They are looking for guidance to kickstart their project and gain a better understanding of the subject matter.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1412726297687359538)** (34 messages🔥): 

> `DSPy Data Leakage Concerns, DSPy Data Splits Clarification (train/val/dev/test), MLflow Integration with DSPy, Context Compression Experiments, Improving dspy.Image Reliability` 


- **DSPy Data Leakage worries defused!**: A user raised concerns about potential test set data leakage in DSPy when using a trainset multiple times, suggesting that papers using DSPy in this way might be invalidated.
   - However, it was clarified that optimizers in DSPy use a training set for training and a validation set for validation, with testing done on a completely separate test set, thus mitigating the data leakage risk.
- **Data Splits Demystified: Train, Val, Dev, Test**: It was clarified that DSPy uses **four data splits**: **train** (for building few-shot examples or instructions), **val** (for outer loop validation and selection), **dev** (for human iteration), and **test** (for pure evaluation once).
   - The discussion emphasized that the *multi-step plots* (with curves) are based on the **valset**, while the final reported results are based on the **testset**, to avoid leakage/overfitting.
- **MLflow & DSPy: A Budding Bromance**: A user inquired about integrating MLflow with DSPy to capture prompts, referencing [MLflow's prompt registry features](https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/).
   - The user noted the existence of [mlflow.dspy](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html) and planned to experiment and report back.
- **Context Compression Craze!**: A member shared a link to [Context Compression Prompt Experiments](https://github.com/Laurian/context-compression-experiments-2508?tab=readme-ov-file#context-compression-prompt-experiments).
   - The project aims at investigating and improving the reliability of `dspy.Image` for some providers.
- **`dspy.Image` needs Reliability Refinement**: A user posted a task to help investigate and improve the reliability of `dspy.Image` for some providers in [this discord thread](https://discord.com/channels/1161519468141355160/1211406131398840390/1412601392446836799).
   - The discussion followed with someone sharing an attached image.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1412732717799571496)** (2 messages): 

> `DSPy for Tax Automation, Amazon Purchase Extraction` 


- **DSPy Automates Amazon Tax Data Extraction**: A member used **DSPy**, attachments, and MLflow to extract data from Amazon purchases for tax purposes.
   - The system identified items like "Sinnvolle Lückenfüller für soziales Lernen (Buch)" and calculated a total of **EUR 104,55**.
- **Codex Powers Flawless Automation Workflow**: A member used Codex to generate code for extracting Amazon purchase data and automatically renaming invoice files.
   - The workflow outputs data into a .csv file with item names, total after tax, and a suggested filename like *lehrmaterial-und-trinkflasche-bundle*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1412796190009065493)** (1 messages): 

> `Kimi Voucher Giveaway, New Kimi Coding Model, Scammer Alert` 


- ****Kimi** Kicks off Voucher Giveaway!**: The **Moonshot Team** announced a giveaway of **20 × $20 API vouchers** for the community to test their new model, which has been juiced up with some *crazy coding powers*.
   - To participate, users should jump into the [#giveaway channel](https://discord.com/channels/1369594130807787570/1412714402284703795) and react with the emoji before **8AM Beijing Time** to enter the raffle.
- **Exclusive Model Access via Voucher**: The announcement emphasizes that only those with a **voucher** can access and test the latest model from **Kimi**.
   - The team urged users to stay tuned for more updates, suggesting further developments and opportunities related to the model.
- **Heads up for Kimi Scammers!**: A warning was issued regarding scammers, advising users that legitimate **Kimi** team members will have a **yellow role color** in the server.
   - The announcement explicitly states, *If it ain’t yellow, don’t trust it*, cautioning users to verify the authenticity of any direct messages received.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1412650668048777227)** (31 messages🔥): 

> `Kimi K2 Model performance, Slide Generation feature, Kimi K2 turbo Coder Pro plan, Model releases for VRAM` 


- **Kimi K2 is top-tier for end-user interactions**: A user stated that **Kimi** is the best model for end-user-facing interactions because it's good at poring into the fine print, finding issues, and having its own opinion.
   - The user thinks [Moonshot](https://x.com/zephyr_z9/status/1962929923091464233?s=46) should dominate in this domain and that Kimi is great at UX stuff and PowerPoint features.
- **Moonshot's Slide Generation Feature impresses**: A user mentioned using the recently released **slide generation feature** and praised the coding enhancements, looking forward to seeing it enable even more professional task handling.
   - They said this update in particular delivers exactly the coding enhancements they were hoping for.
- **Request for Kimi K2 turbo Coder Pro**: A user suggested a **Kimi K2 turbo Coder Pro plan** and added it as a product idea.
   - Another user replied that **Kimi** should just make it a unified plan.
- **Hopes for Model Releases for VRAM**: A user inquired about plans to release models that can fit into **128 (V)RAM** and **24 (V)RAM**, such as **100-200b models** like *gpt-oss-120b* and **30b models** like *gpt-oss-20b*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1412893315853979789)** (5 messages): 

> `Mojo SIMD, Rust AVX, Standard Library net/http module, community driven HTTP library, lightbug_http` 


- **Mojo SIMD is FUN! Rust AVX not so much**: A member said he's *loving how **Mojo** makes **SIMD** actually fun*, manual **AVX** in **Rust** makes me burn more brain cells than I should! 😂
   - He asked whether something like a standard `net/http` style module becoming part of the stdlib.
- **Modular favors a lean standard library**: The general consensus is to keep the standard library very lean for the most part, and per [lightbug_http](https://builds.modular.com/packages/lightbug_http), they have a community driven effort to build an **HTTP library**.
   - It's fairly limited for the time being due to the lack of manual thread management and mojo-native cryptography libraries required to implement **TLS** support.
- **Mojo Powers Fast Binary Search Engine**: A member reports building *a tiny **binary search engine** that crunches **~50k queries/sec** over **2M docs** single cored by parallelizing across **SIMD** lanes*.
   - He looks forward to **HTTP support** to turn it into a search-as-you-type.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1412783939763961997)** (13 messages🔥): 

> `Mojo GPU on GTX 1080, Max backend for Torch, Turing minimum architecture, Reaching out to Modular team` 


- **GTX 1080 can Mojo 🔥 now!**: A member confirmed that Mojo GPU functions are running correctly on their **GTX 1080** after a patch went into the latest nightly.
   - They will land a separate internal patch today to add changelog entries and listed support for **Pascal GPUs**, along with `sm_60` support for the **Tesla P100**.
- **Max Backend for Torch gains steam!**: A member is in the process of getting more time approved to work on the **max backend for torch** full time.
   - They hope to discuss the engineering soundness with Modular team members, aiming for `torch.ones((2, 2), device="max_device")` to work on more GPUs than what is currently available with the latest **CUDA**.
- **Turing Architecture limit found in Mojo!**: A member noted that there will be an error about **Turing** being the minimum-supported architecture if you try to build a graph on `sm_61` GPUs.
   - Their patch should lower that limit when it goes in today, so users may need to wait until the next nightly for basic graph use on these GPUs; amusingly, that may give broader compute capability than PyTorch which errors out with *"PyTorch no longer supports this GPU because it is too old. The minimum cuda capability supported by this library is 7.5."*
- **Discord is best way to reach Modular Team**: A member advised that pinging them directly on Discord is the most reliable way to reach them.
   - They noted their email inbox is flooded, also pointing out a new Modular team member as an excellent person to reach out to if anything falls through the cracks.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1412827979695783967)** (7 messages): 

> `Website deployment on basic plan, Grok as a Tool vs. Agent, Comparison of Grok and Manus` 


- ****Basic Plan** Website Deployment Debacle?**: A user inquired whether the **basic plan** allows for permanent website deployment.
   - Another user succinctly responded, *"it does not"*.
- ****Grok's** True Identity Revealed!**: A user pointed out that **Grok** is a tool, not an agent.
   - They emphasized this distinction in response to conversational context.
- ****Grok** vs. **Manus**: A Non-Existent Comparison?**: A user clarified they were not comparing **Grok** with **Manus** at all.
   - This suggests a potential misunderstanding or off-topic tangent within the conversation.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1412712641948024882)** (6 messages): 

> `GPU Crash with HIPC Code, Multi-GPU Training Distress, Pyrender Testing, Uops Test Updates` 


- **Dual 7900 XTX Cards cause GPU Crash**: A member reported experiencing sudden crashes when reaching peak performance on dual **7900 XTX** cards using **HIPC** code, specifically on the supported kernel **6.8** as mentioned in the ROCm site.
   - They expressed distress due to multi-GPU training issues and wanting to prevent the **GPU** from crashing.
- **Pyrender Testing request**: A member asked if anyone is willing to test **pyrender** on the kernel dataset.
   - No additional information was provided.
- **test_linearizer_dumb Updated, other uops should follow**: A member shared an update to `test_linearizer_dumb` ([link to GitHub PR](https://github.com/tinygrad/tinygrad/pull/11968/files)) and suggested updating other uops in the tests to match the new format.
   - They claim the new format is *more readable and updatable* and offered to fix the uops tests later.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1412857584662220810)** (4 messages): 

> `Codex vs. Aider, OpenAI KYC, GPT-5 Streaming` 


- **GPT-5 KYC requirements cripple streaming**: OpenAI requires **KYC verification** to use its image model and **GPT-5 streaming** features.
   - It's possible to use **GPT-5 without KYC**, but only without streaming enabled.
- **Codex: Aider's Thinking Clone?**: A user expressed frustration with **GPT-5's** processing time for simple requests and missing **thinking streaming**.
   - Another member asked what is liked about **Codex** better than **Aider**, mentioning that **Claude Code** was originally designed to clone **Aider**.

