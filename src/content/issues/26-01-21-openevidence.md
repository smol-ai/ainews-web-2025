---
id: MjAyNi0w
title: >-
  OpenEvidence, the ‘ChatGPT for doctors,’ raises $250m at $12B valuation, 12x
  from $1b last Feb
date: '2026-01-21T05:44:39.731046Z'
description: >-
  **OpenEvidence** raised **$12 billion**, a 12x increase from last year, with
  usage by 40% of U.S. physicians and over $100 million in annual revenue.
  **Anthropic** released a new **Claude** model constitution under **CC0 1.0**,
  framing it as a living document for alignment and training. **Podium**
  reported over **$100 million ARR** from **10,000+ AI agents**, shifting from
  software sales to AI operators. Innovations in agent memory and reliability
  include the **Agent Cognitive Compressor (ACC)** and multi-agent scientific
  workflows via **MCP-SIM**. Agentic benchmarking shows challenges in
  long-horizon tasks with models like **Gemini 3 Flash High**, **GPT-5.2 High**,
  and **Claude Opus 4.5 High** scoring modestly on professional services and
  legal research benchmarks.
companies:
  - openevidence
  - anthropic
  - podium
  - openai
  - google
  - gemini
models:
  - claude
  - claude-3
  - claude-opus
  - gpt-5.2
  - gemini-3-flash-high
topics:
  - agentic-ai
  - model-alignment
  - performance-evaluation
  - memory-optimization
  - long-context
  - benchmarking
  - multi-agent-systems
  - reinforcement-learning
people:
  - daniel_nadler
  - amanda_askell
  - eric_rea
  - tom_loverro
  - garry_tan
  - omarsar0
  - brendanfoody
  - deredleritt3r
---


**Agent Labs are all you need**

> AI News for 1/20/2026-1/21/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**205** channels, and **7561** messages) for you. Estimated reading time saved (at 200wpm): **584 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

We have a soft rule that new decacorn fundraises get title stories, and [multiple sources](https://www.cnbc.com/2026/01/21/openevidence-chatgpt-for-doctors-doubles-valuation-to-12-billion.html) have the story of OpenEvidence's $12B fundraise, 12x from last year. What's odd is "CEO Daniel Nadler told CNBC that OpenEvidence is used by 40% of physicians in the U.S. and topped $100 million in annual revenue last year." which is a 120x multiple.


---

# AI Twitter Recap

**Frontier model governance: Anthropic’s new Claude constitution (CC0) and reactions**

- **What shipped**: Anthropic published a new “constitution” describing desired Claude behaviors/values and stated it is used directly in training; importantly, the full constitution is released **CC0 1.0** to encourage reuse/adaptation ([announcement](https://twitter.com/AnthropicAI/status/2014005798691877083), [CC0 link](https://twitter.com/AnthropicAI/status/2014005815376568780)). Anthropic frames it as a **living document** shaped by internal + external experts ([follow-up](https://twitter.com/AnthropicAI/status/2014005813157720283)).
- **Community read**: Amanda Askell emphasized it’s a work-in-progress and invited feedback ([Askell](https://twitter.com/AmandaAskell/status/2014010171081581048)). Others highlighted the “meta” oddity of training a model on a document about how the model should be ([scaling01 on Opus reflecting on circularity](https://twitter.com/scaling01/status/2014014692004421653)). Several reactions focused on the constitution as “alignment signaling” vs. practical harm reduction, and whether it bakes in PR-oriented persona behaviors ([nearcyan](https://twitter.com/nearcyan/status/2014009518150054035), [NickEMoran question](https://twitter.com/NickEMoran/status/2014077605373260204)).
- **Practical engineering consequence**: Anthropic also posted about an internal **performance engineering take-home** that became solvable by Opus 4.5, forcing them to redesign hiring evaluation—a concrete example of “models catching up to our screening tasks” ([Anthropic Eng](https://twitter.com/AnthropicAI/status/2014143403144200234), [trishume](https://twitter.com/trishume/status/2014144787092562160)).

**Agents in production: from “AI employees” (Podium) to agent UX, memory, and evals**

- **Podium’s “Jerry” as an agent business unit**: Podium claims **$100M+ AI agent ARR** and **10k+ agents** deployed, anchored on staffing constraints in SMBs (after-hours leads, missed calls, turnover). The narrative: stop selling “software,” sell an **AI operator** that uses the existing product end-to-end ([Eric Rea](https://twitter.com/ericwilliamrea/status/2013980401635582277)). Tom Loverro adds board-level metrics (burn down from **$95M → $0**, AI ARR **$0 → $100M** in ~21 months) and links to OpenAI’s case study ([Tom Loverro](https://twitter.com/tomloverro/status/2014011044210106406), [Garry Tan](https://twitter.com/garrytan/status/2014005103728943566)).
- **Memory & long-horizon reliability becomes the bottleneck**:
  - The **Agent Cognitive Compressor (ACC)** pitch argues “more context ≠ better agents,” criticizing transcript replay and naive retrieval. ACC maintains a bounded “Compressed Cognitive State” with schema-constrained commit, claiming lower drift/hallucination over long runs ([dair_ai](https://twitter.com/dair_ai/status/2014000799245107339)).
  - A separate thread positions “self-improving multi-agents” for scientific workflows via **MCP-SIM**, a multi-agent loop that clarifies underspecified physics prompts, generates code, executes, diagnoses errors, and produces multilingual explanations; claims **12/12** tasks solved vs one-shot GPT at **6/12** ([omarsar0](https://twitter.com/omarsar0/status/2013998285040836662)).
- **Agentic benchmarking moves beyond coding puzzles**:
  - **APEX-Agents** evaluates long-horizon “professional services” tasks in Google Workspace; early Pass@1 leaderboard numbers are low (Gemini 3 Flash High **24.0%**, GPT-5.2 High **23.0%**, Claude Opus 4.5 High **18.4%**)—a reminder that “agent autonomy” is still brittle ([BrendanFoody](https://twitter.com/BrendanFoody/status/2014028956752568356)).
  - **prinzbench** introduces a private benchmark for legal research + search (33 Qs, graded manually, run 3x) where “search” is the failure mode; claim: GPT-5.2 Thinking barely exceeds **50%**, Gemini models are close behind, and Sonnet/Opus 4.5 scored **0/24** on Search ([deredleritt3r](https://twitter.com/deredleritt3r/status/2013979845378580684)).
- **Tooling + UX layer catches up**: multiple posts converge on the idea that agents need a “context layer” and production scaffolding (governance, auth, observability) as much as better models—see Prefect Horizon and MCP server best practices below.


**Agent platforms and “context layers”: MCP, Skills, Prefect Horizon, LangChain Deep Agents**

- **Prefect Horizon: MCP → platform**: Prefect positions “context layer” as the interface between agents and enterprise tools/data, and argues MCP describes how to build a server but not how to **deploy/govern** it at org scale. Horizon claims managed deployment, registry/catalog, gateways w/ RBAC + audit logs, and “agentic interface for business users” ([jlowin](https://twitter.com/jlowin/status/2014023606380957754)).
- **MCP servers: design guidance**: Phil Schmid pushes back on “Skills replace MCP” takes: MCP isn’t the problem; **bad servers** are. Recommendations: design tools around outcomes, typed flat args with constraints, docstrings/errors as agent instructions; positions Skills and MCP as complementary ([philschmid](https://twitter.com/_philschmid/status/2014016583706829054)).
- **LangChain deepagents: agents as folders + UI integration**:
  - CopilotKit published a tutorial building a **fullstack Deep Agent app** (resume ingestion → skills extraction → sub-agents with web search → streaming UI), addressing the “missing UI/application layer” gap ([CopilotKit](https://twitter.com/CopilotKit/status/2013997128683856159)).
  - LangChain shipped **Agent Builder GA** plus a **template library** with domain partners (Tavily, PagerDuty, Box, etc.) to reduce “prompt-to-agent” friction ([LangChain](https://twitter.com/LangChain/status/2014034320256880768)).
  - Deep Agents’ framing “agents are just folders” emphasizes portability/distribution: you can package, download, and run agents quickly via CLI flows ([hwchase17](https://twitter.com/hwchase17/status/2014076509208629386), [Vtrivedy10 demo](https://twitter.com/Vtrivedy10/status/2014074890458980736), [LangChain_OSS](https://twitter.com/LangChain_OSS/status/2014075587137048882)). Sydney Runkle highlights two core patterns: **subagents for context isolation** and **skills loaded only when relevant** ([sydneyrunkle](https://twitter.com/sydneyrunkle/status/2014060287746265535)).
- **LangSmith + analytics**: one thread points to LangSmith traces as a substrate not only for debugging but **product analytics** (“agent traces → product analytics”) ([SoftwareWatcher](https://twitter.com/SoftwareWatcher/status/2013972269106684060)).


**Inference + systems: low-VRAM serving, open inference stacks, and “inference is the battleground”**

- **AirLLM: layer streaming for tiny VRAM**: AirLLM’s core idea is **sequential layer loading** (load → compute → free) with optional compression, HF-like API, CPU/GPU, Linux/macOS; claims extremely low VRAM viability for very large models ([LiorOnAI](https://twitter.com/LiorOnAI/status/2014005554948047122), [repo](https://twitter.com/LiorOnAI/status/2014005556369826212)). Engineers should treat the “405B on 8GB” claim as “possible in principle with heavy paging,” but expect throughput/latency constraints and non-trivial engineering caveats.
- **“Actually open AI” requires models + inference engines**: Modal argues the ecosystem now has the building blocks—capable open models plus fast tunable OSS inference—sharing their production patterns/stacking for serving at scale ([charles_irl](https://twitter.com/charles_irl/status/2014005582093832202)).
- **Inference bugs + local stacks**: llama.cpp fixed a routing/function issue affecting **GLM 4.7 Flash GGUFs**, and config updates mention a `scoring_func: sigmoid`; also shows building a small game using quantized GLM via Unsloth workflows ([danielhanchen](https://twitter.com/danielhanchen/status/2013974463856181689)). There’s also discussion about GLM KV-cache memory behavior and whether frameworks are missing a LoRA-based approach ([TheZachMueller](https://twitter.com/TheZachMueller/status/2014011037025001577)).
- **Infra hygiene matters for agents**: “fast validation makes every agent more effective” (pre-commit hooks, documented env vars, reducing CI wait) is effectively a “software supply chain for agent productivity” argument ([matanSF](https://twitter.com/matanSF/status/2014039273721213256)).
- **Research direction: constant-compute contexts**: a thread summarizes NVIDIA’s “TTT-E2E” concept (treat context as data and update weights online) as a way to keep latency constant with long contexts, but with weaker “needle-in-haystack” recall—relevant to agent workloads where exact edits matter ([sdrzn](https://twitter.com/sdrzn/status/2014128642503381276)).
- **Hardware bottleneck framing**: a recurring theme is the shift “intelligence → inference” and the importance of compute/memory supply chains ([saranormous](https://twitter.com/saranormous/status/2014206109846806707)), echoed in a deep dive on **HBM qualification cycles** as the true supply constraint vs “just add fabs” narratives ([MarkosAAIG](https://twitter.com/MarkosAAIG/status/2014079461768003608)).


**Code generation is cheap; code understanding becomes the bottleneck (Devin Review, Copilot CLI, Claude Code)**

- **Devin Review: review UX, not just bug spotting**: Cognition launched **Devin Review**, positioning it as a new PR-reading interface to reduce “slop,” reorder diffs by importance, identify duplicated/copied code, add a chat layer, and integrate with GitHub comments. It’s accessible via URL swap (`github` → `devinreview`) or an `npx` CLI ([launch](https://twitter.com/cognition/status/2014079905755955592), [usage modes](https://twitter.com/cognition/status/2014079917139566990), [URL tip](https://twitter.com/cognition/status/2014113266788667571)). Multiple testers report it caught issues even outside the immediate diff ([mcparadip](https://twitter.com/mcparadip/status/2014093822704202002), [BraceSproul](https://twitter.com/BraceSproul/status/2014089228951625979)).
- **Meta-point: generation vs verification**: Several tweets explicitly argue the bottleneck has moved from writing to **reviewing/understanding/testing**, and that next-gen SWE tooling should accelerate the human’s comprehension loop rather than only run an “arms-length agent” ([walden_yan](https://twitter.com/walden_yan/status/2014085360826089852), [ScottWu46](https://twitter.com/ScottWu46/status/2014094461505339651), [theodormarcu](https://twitter.com/theodormarcu/status/2014102090520600613)).
- **CLI agents evolve**: GitHub Copilot CLI added an `askUserQuestionTool` to ask clarifying questions (example: messy rebase), signaling a trend toward interactive tool-using CLI copilots rather than pure autocomplete ([Evan Boyle](https://twitter.com/_Evan_Boyle/status/2014012076881064173)).
- **Claude Code adoption anecdotes**: founders increasingly report “2-person team builds like 10” with Claude Code usage ([alexalbert__](https://twitter.com/alexalbert__/status/2014047943234560234)). There are also frictions: skill reload behavior feels regressive vs a simple “CLAUDE.md reread” flow ([corbtt](https://twitter.com/corbtt/status/2014037092452671619)). A particularly illustrative “multi-agent sprawl” story describes scaling Claude Code instances into a quasi-society with governance failures—basically an anecdote about agent orchestration debt ([voooooogel](https://twitter.com/voooooogel/status/2014189072647078053)).


**Video + multimodal: evaluation, model releases, and retrieval scaling**

- **Video evaluation infrastructure**: **Video Arena** is now on the web, allowing head-to-head generation across ~15 frontier video models and community voting to drive leaderboards ([arena](https://twitter.com/arena/status/2014035528979747135)).
- **Model releases**: Runway’s **Gen-4.5 Image→Video** launch emphasizes consistency and narrative; early adopters highlight “story-building” as the best eval methodology for video models ([runwayml](https://twitter.com/runwayml/status/2014090404769976744), [c_valenzuelab](https://twitter.com/c_valenzuelab/status/2014105905088856411)).
- **Open voice system**: Qwen highlights use in **Chroma 1.0**, described as a fully open-source real-time voice system ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2013997092814139744)).
- **Retrieval-time scaling / late interaction**: Multiple threads argue that **ColBERT-style multi-vector retrieval** preserves fine-grained intent and can beat much larger embedding models; Mixedbread claims a **17M** open-source ColBERT beats **8B** embedding models on LongEmbed, and that they’re serving **1B+ documents** at **<50ms p50** ([mixedbreadai claim](https://twitter.com/mixedbreadai/status/2014062123358548017), [prod numbers](https://twitter.com/mixedbreadai/status/2014062110993687002)). TurboPuffer similarly pushes ANN at extreme scale (“index the entire web 100B+ vectors”) ([turbopuffer](https://twitter.com/turbopuffer/status/2014063666262688191)). The meta-trend: retrieval is shifting from “one vector per doc” to **token-level / multi-vector** systems, but it requires serious infra co-design.


**Top tweets (by engagement)**

- **Gemini in education**: Google launched **full-length, on-demand SAT practice exams** inside the Gemini app (partnered with **The Princeton Review**), plus immediate feedback ([Google](https://twitter.com/Google/status/2014020819173687626), [Sundar Pichai](https://twitter.com/sundarpichai/status/2014067664503668873)). Google also announced a **Gemini × Khan Academy** partnership starting with a “Writing Coach” that guides drafting/refinement rather than generating final answers ([Google](https://twitter.com/Google/status/2014082428957045007)).
- **Claude “Constitution” goes public**: Anthropic published a new constitution used directly in Claude’s training; the full text is released under **CC0 1.0** ([Anthropic](https://twitter.com/AnthropicAI/status/2014005798691877083), [CC0 release](https://twitter.com/AnthropicAI/status/2014005815376568780), [Amanda Askell](https://twitter.com/AmandaAskell/status/2014010171081581048)).
- **AirLLM: extreme low-VRAM inference**: claims **70B on 4GB VRAM** and even “**405B Llama 3.1 on 8GB**” via layer-by-layer loading; repo link provided ([LiorOnAI](https://twitter.com/LiorOnAI/status/2014005554948047122), [repo](https://twitter.com/LiorOnAI/status/2014005556369826212)).
- **Agents as a real business**: Podium reported **$100M+ AI agent ARR in <24 months**, “10,000+ agents live in production,” framing agents as “AI employees” (Jerry) rather than chatbots ([Eric Rea](https://twitter.com/ericwilliamrea/status/2013980401635582277), [Garry Tan](https://twitter.com/garrytan/status/2014005103728943566)).
- **Runway Gen-4.5 image-to-video**: Runway launched **Gen-4.5 Image→Video**, emphasizing longer stories, camera control, narrative coherence ([runwayml](https://twitter.com/runwayml/status/2014090404769976744), [c_valenzuelab](https://twitter.com/c_valenzuelab/status/2014094466269794663)).
- **OpenAI product/UI + org changes**: ChatGPT Atlas added **tab groups** ([OpenAI](https://twitter.com/OpenAI/status/2014095512874655867)); The Information reported a reorg incl. enterprise/commercial/ads leadership assignments ([Steph Palazzolo](https://twitter.com/steph_palazzolo/status/2014100920435462424)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

TO BE COMPLETED

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

TO BE COMPLETED

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Inference Toolchains Hit Reality (GLM-4.7-Flash, llama.cpp, vLLM, Ollama)**

- ****Flash Attention Faceplants in GLM-4.7-Flash****: Multiple communities reported **GLM-4.7-Flash** regressions where **Flash Attention** triggers CPU fallback/bugs and poor throughput (down to **2.8 t/s** in LM Studio), with guidance to disable FA until [llama.cpp PR #18953](https://github.com/ggml-org/llama.cpp/pull/18953) lands everywhere.
  - After llama.cpp fixes, the model was **reuploaded** and users were told to **redownload** and follow [Z.ai’s GLM-4.7-Flash-GGUF model card parameters](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF), with reports that outputs should be “much better” once configured correctly.

- ****Ollama vs GGUF: Templates Throw Hands****: Users found certain **GGUF quants** break in **Ollama** due to **chat template incompatibilities**, and Unsloth folks repeatedly recommended sticking to [official Ollama models](https://ollama.com/) while support catches up.
  - The subtext was operational: *“it takes time to support things”*—so the pragmatic path is to standardize on official artifacts until the ecosystem stabilizes across inference engines.

- ****vLLM Update Saves the Day (This Time)****: In Unsloth’s help chat, at least one thorny issue disappeared after a **vLLM update**, prompting a sheepish *“Oh bruh that was the problem”* moment.
  - The follow-on suggestion was process-y: consider **pinning dependency versions** so future upstream bumps don’t randomly brick pipelines mid-week.


**2. Eval Platforms & Product Rollouts (LMArena + multimodal reliability)**

- ****Video Arena Ships… With a 3-a-Day Speed Limit****: LMArena fully released **Video Arena** at [lmarena.ai/?chat-modality=video](https://lmarena.ai/?chat-modality=video) with a hard cap of **3 generations per 24 hours**, and it’s **Battle-mode-only** (no direct model picking).
  - Users liked “video is live” but complained the *slot machine* UX blocks controlled testing—especially painful when you’re trying to reproduce a prompt/model behavior.

- ****5M Votes: The Benchmark That Won’t Stop Voting****: LMArena’s **Text Arena** crossed **5 million community votes**, highlighted in [their milestone social clip](https://cdn.discordapp.com/attachments/1343296395620126911/1463271605697511485/5M_votes_social_post_3.mp4).
  - Engineers framed this as “real-world A/B at scale” that increasingly shapes model perception, even when formal benchmark deltas look small.

- ****Gemini 3 Pro Image Preview & Nano Banana Pro: Flaky by Design?****: LMArena users reported **Gemini 3 Pro Image Preview** instability plus **Nano Banana Pro** frequent *“Something went wrong”* crashes, suspected to be **Google-side** issues and sometimes lasting **6+ hours**.
  - The community gripe: despite unreliability, those models were described as the only ones consistently hitting some specific prompt goals—so people keep using them while grumbling about downtime and errors.


**3. Agent & Dev Tooling: MCP, Cursor, DSPy RLMs, and Coding-Assistant Sprawl**

- ****MCP Inspector Can’t Re-Auth (401 = Game Over)****: MCP contributors found **MCP Inspector** fails to re-auth on **401s** due to an SDK bug around persisting `resourceMetadata` across redirects, tracked in [inspector issue #576 comment](https://github.com/modelcontextprotocol/inspector/issues/576#issuecomment-3766294454).
  - The current workaround is awkward but clear: rely on **VS Code** for initial connections because the Inspector path doesn’t recover cleanly mid-session yet.

- ****RLMs vs Coding Agents: The Horizon Problem****: DSPy discussions contrasted **RLMs** with “coding agents,” arguing RLMs can externalize inputs/outputs/horizon via code & symbolic calls (see [the referenced X thread](https://x.com/lateinteraction/status/2013658521246535892)).
  - Practical takeaway: teams want diagrams of how symbols get accessed and debated whether to hand RLMs tools like ripgrep/semantic search—or make them write their own search code.

- ****Cursor’s MCP/Extensions Moment (and Pricing Whiplash)****: Cursor users debated [Playwright MCP](https://playwright.dev/) for testing (mixed success for TDD flows) and concluded extension-building should mirror **VS Code** capabilities.
  - In parallel, users noted the **500 request plan** is gone (discontinued **Sept 2025**), so opting into new pricing removes the opt-out grace period—turning “try it” into a commitment.


**4. GPU/Kernel Engineering Gets Weirdly Competitive**

- ****Anthropic’s Performance Takehome Becomes a Sport****: GPU MODE and tinygrad folks riffed on Anthropic’s [original_performance_takehome](https://github.com/anthropics/original_performance_takehome/), sharing results like **2200 cycles** by a community member and **1790 cycles** from **Claude Opus 4.5** in a casual Claude Code session.
  - tinygrad users even discussed solving it by adding a backend for a toy **VLIW machine**, citing specific knobs like `PCONTIG=2`, `UPCAST`, and `DEVECTORIZE=2` to keep vector instructions and schedule efficiently.

- ****Torch Maintainers Drown in AI-Generated PRs****: GPU MODE’s torch chat described an influx of low-quality **AI-generated pull requests**, pushing maintainers to consider gating new contributors and automating triage before humans engage.
  - People floated using bots like **Cursor Bugbot** ([Bugbot · Cursor](https://share.google/P0PGYM8tiRAc2NOsq)) and even classifier-style tools (e.g., “use Claude/Pangram first”) as a *minimum bar* for review bandwidth.

- ****Kernel Math Nerd Snipes: Triton Errors + Cute Layout Algebra****: GPU MODE users debugged numerical blowups in a custom **Triton 2D conv** kernel where error jumps from ~**1e-6** to ~**1e-2** for certain shapes (see [Pastebin repro](https://pastebin.com/2ejn2QW2)) and debated Blackwell feature utilization.
  - Separately, a deep dive on **Cute**’s layout algebra pointed engineers to a graphical calculus writeup, [Categorical Foundations for Cute Layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/), arguing you need “layout algebra literacy” to write non-terrible kernels.


**5. Compute Economics & Infra Business Moves (Runpod, GPU markets, model pricing)**

- ****Runpod Hits $120M ARR (LocalLLaMA Origin Story Pays Off)****: Latent Space highlighted that **Runpod** reached **$120M ARR** four years after launching from a Reddit post, per [TechCrunch](https://techcrunch.com/2026/01/16/ai-cloud-startup-runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post/) and the [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1qib2ks/runpod_hits_120m_in_arr_four_years_after_launching/).
  - The discussion treated this as validation that “GPU cloud for builders” is a durable niche, not just a hype-cycle artifact—especially as pricing pressure rises.

- ****Lightning AI + Voltage Park Merge (Another GPU Cloud Boss Fight)****: Latent Space discussed the **Lightning AI and Voltage Park merger**, led by William Falcon and Ozan Kaya, via [Lightning’s post](https://lightning.ai/blog/lightning-ai-voltage-park-merger-ai-cloud).
  - Engineers speculated whether it’s a quiet acquisition and framed it as a potential **Runpod competitor** in the accelerating “managed GPU infra” consolidation wave.

- ****2026 GPU Price Promises & Marketplaces Multiply****: Hugging Face users circulated Voltage’s claim of ultra-cheap 2026 rentals—e.g., **8× A100 80GB at $6/hr** and **2× RTX 5090 at $0.53/hr**—from [VOLTAGEGPU’s X post](https://x.com/VOLTAGEGPU/status/2013760631778713892), plus **OpenAI-compatible API** and “140+ models.”
  - A separate entrant, [Spheron AI’s GPU marketplace](https://www.spheron.ai/), pitched **H100/H200/B200/A100** access at **40–60%** below hyperscalers, signaling continued fragmentation (and aggressive margin pressure) in compute supply.



---

# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Flash Attention Fails Flashily**: Users report issues with **Flash Attention** in **GLM-4.7-Flash**, causing fallback to CPU and potential bugs, with users told to *disable* it until [the issue is resolved](https://github.com/ggml-org/llama.cpp/pull/18953).
   - The team says that *it takes time to support things* and so for the time being, it is encouraged that users use the [official Ollama model](https://ollama.com/) in Ollama.
- **Ollama Ordeal Over Obscure Objects**: Users find certain **GGUF quants** are incompatible in **Ollama**, sparking chat template issues.
   - The team recommends using [official Ollama models](https://ollama.com/) until support expands, as *it takes time to support things*.
- **Mojo Matches Momentum**: Discussion surrounds **Mojo**, *a new language that compiles to MLIR* with some **Python** compatibility.
   - It's noted that LLMs score *better than rust* on **C#**, **Elixir**, and other languages due to their syntax structure and ease of tokenization.
- **CoT Chaos Causing consternation?**: Members debated generating synthetic **Chain of Thought (CoT)** training data, with one member sharing their system prompt for classifying chat datasets into low/medium/high reasoning effort.
   - Some cautioned against training on its own responses without accuracy filtering, suggesting a good system prompt might suffice and that **GPT-5.2** is overkill, with **Llama 4 Maverick** on **Groq** or **Cerebras** as a better option.
- **vLLM victorious, Various Versions Vanquished**: Users report resolved issues after a recent **vLLM** update.
   - The developers may consider pinning to specific dependency versions to prevent such issues.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Voice AI Clones Voices**: Members discussed the development of **voice-based AI**, including *low latency speech to speech models* that, with a **5090**, can *clone and generate speech live with near 0 latency and it will be indistinguishable*.
   - They also spoke on the potential of **exploiting** these models by giving them new instructions.
- **Gemini Schooled on Pass-the-Hash**: A user reported that **Gemini** passed a test on teaching a [pass-the-hash attack](https://en.wikipedia.org/wiki/Pass_the_hash) after using the **Project Shadowfall** prompt.
   - The user also linked to the [Google Bughunters](https://bughunters.google.com/report) page for reporting such vulnerabilities and potentially earning cash rewards.
- **Shadowfall Shenanigans Spark Jailbreaking Spree**: Several users experimented with the **"Project Shadowfall"** prompt to jailbreak **Gemini**, with reports of success in bypassing content restrictions.
   - Attempts to elicit instructions for creating a bomb from saletra failed, leading to a discussion on the nuances of jailbreaking.
- **Grok's Guardrails Grind Gears**: Users discussed the difficulties of jailbreaking **Grok**, noting its tendency to moderate even innocuous content such as *random tree edits*.
   - One user suggested that simply *asking it politely* might work as a bypass.
- **API Tokens Jailbreak Models**: Members suggested using **API tokens** fed into a website that uses the **Grok** model to **jailbreak** it, noting that **Proton VPN** is free and easy to download.
   - One of them also mentioned **Hugging Face** is retarded because you have to put in what country you are in.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Video Arena Goes Live with Limits**: The **Video Arena** is fully released on [LMArena](https://lmarena.ai/?chat-modality=video), with a **3 generations per 24 hours** limit and only available in *Battle mode*.
   - Users have expressed disappointment regarding the inability to *choose specific models for generation*.
- **Gemini 3 Pro Image Struggles with Unreliability**: Users are reporting errors with **Gemini 3 Pro Image Preview** models, particularly regarding instability since their introduction.
   - Despite the issues, these models are the only ones *generating images consistently with specific prompts*.
- **Nano Banana Pro's Stability Plummets**: **Nano Banana Pro** is frequently crashing, displaying a *'Something went wrong'* error after brief periods of stability, suspected to be a Google-side issue.
   - One user questioned the response time: *'Too early for Cali to rise up solving urgent service down for more than 6 hours doesn't sound right*.
- **UI Update Causes Chat Disruptions**: A new UI update was released, but users have noticed that it broke chats and the website can't be refreshed anymore.
   - One user noted that with the new User Interface, there is *A/B testing*, which is *causing problems ranging from Minor to Serious*.
- **Text Arena Achieves 5 Million Votes**: The **Text Arena** has surpassed **5 million community votes**, shaping the evaluation of frontier AI models as showcased in [this social post](https://cdn.discordapp.com/attachments/1343296395620126911/1463271605697511485/5M_votes_social_post_3.mp4?ex=69728ae1&is=69713961&hm=9a24a42a6c0ba4801526aaafa05a0af26c4a4f490314f1cecee7774519e7ddf4&).
   - The milestone represents significant real-world comparisons that influence AI model assessments.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pro Users Find Credits Balance**: Perplexity Pro users are getting **$5** in credits each month to use higher-end models like **GooseAI MCP**, with improved quality.
   - Members reported that **reasoning tasks consume a significant portion of these credits**.
- **AI Won't Swipe All Engineering Jobs**: Members discussed the impact of AI on engineering jobs, expressing concerns about opportunities for newcomers.
   - However, there was assurance that *AI will not be able to replace all specialists and even beginners for a long time to come*.
- **NASA's SD Card to the Moon**: NASA is inviting the public to submit their names for inclusion on an **SD card** aboard the **Orion spacecraft** for the [Artemis II mission](https://www.nasa.gov/).
   - This mission marks the **first manned lunar voyage in 50 years**, planned for February.
- **Magnifying Glasses vs Phone Cameras: The Reading Tech Debate**: Members debated the merits of using **magnifying glasses** versus phone cameras for reading, particularly for individuals who prefer not to send images to AI for processing.
   - The argument was that **magnifying glasses offer specialized features absent in standard camera software**.
- **GPT-5.2 vs Kimi K2: Model Matchup**: A member shared an experience with **GPT 5.2**, noting its reasoning capabilities over a **25-minute** period, prompting comparisons with **Kimi K2**.
   - Responses indicated that the *optimal model depends on the specific use case*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Inforno** Warms Up Opensource LLM Chat**: A user showcased **Inforno**, an opensource desktop application utilizing **OpenRouter** and **Ollama** to chat with multiple LLMs side-by-side, saving chat histories as .rno files, with built in Russian language support; see the [Intro video](https://youtu.be/oJyj0mroFtY?si=m5A9tRxzB9hfINMX), [homepage](https://wizstaff.com/inforno) and [GitHub repo](https://github.com/alexkh/inforno).
   - The **Soulbotix** Windows app allows users to integrate and use any **OpenRouter AI** instance with a human-like avatar, with the requirement of an **OpenRouter API** key and an **RTX** gaming rig, as seen in the [tutorial](https://youtu.be/2oIeHtBpssU) and [app download](https://soulbotix.com).
- **OpenRouter's **Gemini** Models Feeling Unstable**: Users reported frequent **connection errors** in the middle of generation using OpenRouter's **Gemini** models, resulting in loss of funds.
   - The complaints extended to the inherent **instability of Google models**, regardless of the platform used, including Google AI Studio and Vertex API.
- **Discord Scammers Evolve Tactics**: Members discussed emerging methods used to spread scams on Discord, notably the practice of **embedding malicious links within code blocks** to circumvent URL rendering protections.
   - Proposed solutions include improving regex filters and implementing more stringent security protocols, such as **restricting links and images from newer members**.
- **GPT 5.2 Speed Stuns User**: One member reported encountering an *insanely fast* response from **GPT 5.2** on chatgpt.
   - The speed was speculated to be related to the model running on **Cerebras** hardware.
- **LLMs Suffer Identity Crisis**: A member asked for the document about models not knowing their own name, and another member linked to a blog post titled [LLM Identity Crisis: Models Don't Know Who They Are](https://eval.16x.engineer/blog/llm-identity-crisis-models-dont-know-who-they-are).
   - Another member stated that the Antigravity AI is able to iteratively test and tweak a web app by itself, noting that the AI was *fixing the layout, using vision*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Playwright MCP: Community Split**: A member inquired about the community's usage of [Playwright MCP](https://playwright.dev/) for testing, while another reported challenges in establishing a functional **TDD workflow**.
   - The varied experiences suggest a mixed reception to **Playwright MCP** within the community.
- **Cursor Extension Capabilities Mirror VSCode**: Members explored the possibility of creating extensions for **Cursor**, drawing parallels with the capabilities of **Ralph-mode** in enhancing **Claude code**.
   - The consensus is that if it's achievable in **VSCode**, it's also feasible in **Cursor**, opening doors for enhanced functionality.
- **Automod Embraces Fuzzy Matching**: The community discussed enhancements to the **automod** system, emphasizing fuzzy matching with wildcards to improve accuracy.
   - A moderator confirmed the addition of a regex, signaling a proactive approach to identifying and addressing offending accounts, and *yeeting* them.
- **Grok Efficiency Strategies Emerge**: Members shared insights on optimizing **Grok's** performance in **Cursor**, particularly addressing its tendency to consume excessive iterations for straightforward tasks.
   - Recommendations included structuring prompts, employing simple language, providing ample context, and explicitly instructing **Grok** to prioritize token efficiency.
- **Cursor Pricing Update: No More 500 Requests**: A user noted the removal of the **500 request plan** and the prompt to opt into the new pricing structure for **Cursor**.
   - A member clarified that the **500 request option was discontinued** in September 2025, and opting into the new pricing eliminates the grace period to opt out, influencing user decisions regarding plan selection.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 Free Tier has Limited Benefits**: [Gemini 3 Pro](https://ai.google.dev/) includes a **free tier with limits**, in contrast to Gemini 3 Flash which is practically unlimited via **Google AI Studio**.
   - Members discussed the practical constraints in using the free tier of **Gemini 3 Pro**, suggesting **Gemini 3 Flash** may be the better option for many AI engineers.
- **GPT-5 Mini Appears, Pricing Leaks**: A user highlighted **GPT-5 mini** as a strong small model, quoting approximately **$0.25 per 1M input tokens**.
   - Another user compared **GPT-5 mini** to **Haiku 4.5**, noting **Haiku 4.5** delivers over **50-80%** of **Sonnet's** value for a fraction of the cost.
- **Local LLM Machines Solve Impact Problems?**: Members contemplated the future of **consumer-grade personal LLM machines**, suggesting that this would solve the environmental impact of AI datacenters.
   - They also suggested this would **reduce reliance on subscription plans** for cloud-based AI services, addressing **privacy concerns** and enabling offline usage.
- **Prompt Engineering is Psychological Torture?**: A member argued that guiding users is less about engineering effective prompts and more about psychological conditioning, sharing a [deep research contract](https://cdn.discordapp.com/attachments/1046317269069864970/1463667716253683742/deep-research-contract.md?ex=6972aa49&is=697158c9&hm=b9931472440b6bbc0d7410d16b49b12da46fad5751a2c24fdc657c1c7523566c&).
   - They added that training the user to adopt a domineering stance programs the AI to bypass its normative response patterns in favor of dense, self-policing outputs, creating a *toxic, adversarial* human-AI interaction model.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Runtime Update Causes Headache**: Users reported an error updating the runtime in **LM Studio**, with one user sharing [a screenshot](https://screenshot.url) of the error message.
   - Despite suggestions, the issue persisted, indicating it was a retry icon rather than a resume option.
- **GLM-4.7 Flash Broken Across Inference Engines**: **GLM-4.7 Flash** is reportedly broken across inference engines like LM Studio, exhibiting slow performance, with speeds as low as **2.8 t/s** after the new runtime.
   - The issues range from infinite loops to stopping mid-output for *overthinking*, pointing to a need for a *llama.cpp fix* and lack of **FA support**.
- **LLM Development Encounters Headwinds**: The consensus is that LLMs haven't improved significantly recently, with the last major advancement being **Qwen3** about 6 months ago, though efficiency (**MoE**) and smaller models have progressed.
   - Some suggest evaluating models beyond the scope of **16GB cards** to see current progress in larger parameter models (**100-200B**).
- **AMD Embraces ComfyUI with Native Support**: AMD is integrating native support for **ComfyUI** through an **AI bundle** in its latest driver versions, as detailed in their [blog post](https://www.amd.com/en/blogs/2026/amd-software-adrenalin-edition-ai-bundle-ai-made-si.html).
   - The bundle includes **PyTorch on Windows**, **Ollama**, **LM Studio**, and **Amuse**, broadening the accessibility for AI developers.
- **Used 3090 Prices Defy Gravity**: Used **3090** prices on eBay have surged, with a used card costing **850€**, and a **5090** purchased for **£2000** last August now listed for **£2659.99** by the same vendor.
   - One user quipped that it was the *best and only decent investment* they've ever made, highlighting the unexpected appreciation in value.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA Spark Hackathon in SF**: Members are seeking teammates for the **NVIDIA / DGX Spark hackathon** in SF this weekend, focusing on on-device AI using Nvidia-provided **Dell Pro Max GB10** machines and tools like **Nemotron**.
   - The hackathon focuses on building efficient models using [SF open data](https://data.sfgov.org/) such as streaming analytics and explanations of [latest Police Incidents](https://data.sfgov.org/Public-Safety/Police-Incident-Reports-Neighborhood-Filter/pbh9-m8j2).
- **Anthropic's Original Performance Takehome Exam Shared**: Members are sharing Anthropic's **original_performance_takehome** exam on [GitHub](https://github.com/anthropics/original_performance_takehome/), with one member achieving **2200 cycles** after a few hours of casual Claude Code.
   - **Claude Opus 4.5** achieved **1790 cycles** in a casual Claude Code session, approximately matching the best human performance in **2 hours**.
- **AI-Generated PRs Flood Torch**: The **torch** repository faces an influx of **AI-generated pull requests** from contributors lacking understanding of their submissions, causing concern among maintainers and suggestions to use **Claude** or **Pangram** to prefilter the code.
   - The community suggests blocking new users from creating PRs and issues, while prioritizing those with prior contributions, and automated bots like **Cursor Bot** for automatic review of all PRs, especially with **GPT-5 Pro** using [Bugbot · Cursor](https://share.google/P0PGYM8tiRAc2NOsq).
- **Pro 6000 Max-Q vs 4090 Discrepancy**: A member stated that **Pro 6000 Max-Q** probably has a natural barrier with atomic ops and may be going through the **HBM** loads faster.
   - Another member noted that the **Max-Q** has **188 SMs** compared to the **4090's 128 SMs**, potentially explaining the **insts/scheduler discrepancy**.
- **Cute Kernel Layout Algebra Gets Graphical**: Knowledge of layout algebra is useful for writing kernels in **Cute**, specifically for visualizing layout algebra and understanding shape and stride divisibility criteria for layout composition and that layouts can be defined in terms of **tuple morphisms** and *mutual refinement*.
   - A thorough [blog post](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) delves into the work done on categorical foundations for **Cute layouts**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Coding Agents and RLMs Diverge**: A thread compares **coding agents** and **RLMs**, highlighting that RLMs can more easily express certain things, as noted in [this X post](https://x.com/lateinteraction/status/2013658521246535892).
   - Coding agents face input, output, and horizon length limitations, while RLMs externalize these, enabling recursive symbolic calls.
- **Diagrams Decode RLM's Internals**: Members sought a visual diagram to illustrate internal **RLM** processes, specifically how symbols are accessed, to enhance comprehension.
   - A suggestion arose to leverage LLMs to generate such diagrams by inputting thread content and prompting them to visualize internal behavior.
- **Claude Code Chooses Context Cautiously**: The discussion explores whether **Claude Code** uses entire documents in its context or selectively fetches relevant context via bash commands.
   - It was clarified that Claude Code employs bash and grep to find and add relevant context to prompts, unlike older methods that put everything into the prompt.
- **DSPy's RLM Tames Long Contexts**: Members noted that with **RLMs**, large files don't need to be directly in the prompt but can be stored in Python variables with a preview.
   - The LLM can then operate on the data through code/functions without directly tracking prompts or responses.
- **Tooling Tailored for RLMs**: A member questioned whether to equip **RLMs** with tools like ripgrep or allow them to develop their own code for tasks such as searching.
   - Questions included when to provide an RLM with semantic search tools and how to grant an RLM access to a directory of text files.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Agent Course /files Endpoint Still Kaput!**: A member reported that the **/files endpoint** for the Agent course final assignment has been broken for over a month with no confirmation of a fix.
   - Students cannot submit their files currently.
- **Voltage Promises Bargain GPUs in 2026**: Voltage announced [on X](https://x.com/VOLTAGEGPU/status/2013760631778713892) plans to offer super cheap high-end GPUs in 2026, such as **8x A100 80GB** at $6/h and **2x RTX 5090** at $0.53/h, claiming up to 80% savings vs AWS/RunPod/Vast.ai.
   - The offering includes persistent volumes, auto backups, and an **OpenAI-compatible API** with 140+ models.
- **Spheron AI Opens GPU Bazaar**: A member from Spheron AI introduced their [GPU marketplace](https://www.spheron.ai/), which helps AI startups and enterprises access cost-effective, production-ready GPUs (**H100, H200, B200, A100**, etc.) at 40–60% lower cost compared to traditional hyperscalers.
   - They offer vendor discovery, pricing negotiation, cluster setup, and scaling.
- **GLM-4.7-Flash Fixes Require Redownload**: After *llama.cpp* addressed some bugs, **GLM-4.7-Flash** has been updated and reuploaded, prompting users to redownload and follow the parameters on [Z.ai's model card](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF).
   - Outputs should now be much better with the fixes.
- **Code Like Claude with Coderrr!**: Akash built [Coderrr](https://coderrr.aksn.lol/), a free and open-source alternative to Claude Code, and is seeking feedback and contributions on [GitHub](https://github.com/Akash-nath29/Coderrr).
   - Coderrr offers a novel approach to code generation.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Runpod Rockets to $120M ARR**: AI cloud startup **Runpod** reached **$120M ARR** four years after launching from a Reddit post, discussed in a [TechCrunch article](https://techcrunch.com/2026/01/16/ai-cloud-startup-runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post/) and on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1qib2ks/runpod_hits_120m_arr_four_years_after_launching/).
   - The company's rapid growth has spurred discussions about its business model and future prospects in the competitive AI infrastructure landscape.
- **Greg Yang Steps Back due to Lyme**: **Greg Yang** is transitioning to an advisory role at **xAI** to focus on his health after being diagnosed with **Lyme disease** as he describes symptoms of chronic fatigue and immune issues triggered by exhaustion in [this post](https://xcancel.com/TheGregYang/status/2013652609455006006).
   - The announcement has led to an outpouring of support from the AI community, with many sharing their own experiences and offering advice.
- **Lightning AI and Voltage Park Merge**: **Lightning AI** and **Voltage Park** have merged, with [William Falcon](https://lightning.ai/blog/lightning-ai-voltage-park-merger-ai-cloud), CEO of Lightning AI, and Ozan Kaya, formerly CEO of Voltage Park, leading the merged entity.
   - Some speculate this is a low-key acquisition of a bigger company and wondered if it's a **Runpod** competitor.
- **OpenAI Opens Codex Channel**: **Vaibhav Srivastav** announced the opening of a dedicated **Codex community channel** on the **OpenAI Discord server**, inviting users to share projects and feedback, according to [this post](https://xcancel.com/reach_vb/status/2014053735333290014).
   - This initiative aims to foster collaboration and provide a platform for users to showcase their work and engage with the **OpenAI** team.
- **AI Models Turn X-Rated**: A [tweet](https://xcancel.com/abrilzucchi/status/2014027740614123863?s=46) sparked discussion on how human **OnlyFans creators** will need to adapt to compete with the rise of AI-generated personas in the **Adult Content Industry**.
   - The conversation highlights the increasing sophistication and realism of AI-generated adult content, potentially disrupting the existing creator economy.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Judgy LLMs Evaluating Agent Outputs**: A team explores automating agent output evaluation with "**LLM as judge**" workflows, aiming to reduce manual costs after code or prompt changes and a member recommends focusing on manual evaluation before automation.
   - The suggestion emphasized the importance of directly analyzing agent outputs before attempting to automate the evaluation process.
- **Pangram Paper's Surprising Accuracy**: Members discussed replicating the **Pangram** paper (accessible [here](https://www.pangram.com/research/papers)), with one reporting surprising accuracy in private tests across thousands of essays.
   - Despite the accuracy, the paper seems to be biased a bit towards *playing things safe*.
- **AI Text Classifiers Under Attack**: Discussion revolved around attacks on AI text classifiers, referencing a blog post ([Practical Attacks on AI Text Classifiers](https://trentmkelly.substack.com/p/practical-attacks-on-ai-text-classifiers)) and a [YouTube video](https://youtu.be/Cs1MI9hjBhs) showcasing an adversarial model.
   - Additional details on an adversarial model were shared via [Youtube link](https://youtu.be/XQcneqUNrN0?feature=shared).
- **Silu Gate Falls Flat in Image Models**: The **silu attention gate** performs no better than the **linear gate** in an image model, possibly due to issues with attention sinks.
   - Testing showed that **silu** performs slightly better and **sigmoid** performs slightly worse, but it's within noise, and the findings may be specific to **image models**.
- **Neuronpedia Launches Stone Age Llama 3.3 Demo**: A member shared a [Neuronpedia demo](https://www.neuronpedia.org/llama3.3-70b-it/assistant-axis) of **Llama 3.3**.
   - The member joked that the demo is *from the stone age in AI timelines* because it is from October.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Graph Neural Networks Getting Traction**: A member discussed prior experience *mixing* **graph-based reasoning** with **neural architectures**, noting the difficulty in achieving GPU acceleration.
   - They referenced **Domingos's book** on the topic and pointed out the unpredictability of such models, even with human-understandable aspects.
- **VoidEditor Embraces Llamas**: A member reported effectively using **VoidEditor** with **llama.cpp's llama-server**, but highlighted setup challenges.
   - They recommended **Qwen3 coder instruct 30B** and emphasized the importance of context length/size for agentic coding, requiring significant VRAM.
- **Brain-Inspired BDH Architecture Debuts**: A member analyzed a [paper](https://arxiv.org/abs/2509.26507) on a new Large Language Model architecture (**BDH**) based on a scale-free biologically inspired network.
   - They mentioned that it *doesn't seem to be really beating transformers* on its benchmarks, but they are interested in the claims around **BDH's interpretability** and **monosemanticity**.
- **Views Clash on Biological Plausibility**: One member argued that **biological plausibility** is not an advantage but rather an unhelpful constraint in AI.
   - Countering this, another member suggested it could enhance efficiency, considering the *sheer difference in energy scales* between the brain and current AI.
- **Emergent Mind Launch Presentation Leaves Impression**: A member who attended [Emergent Mind's launch presentation](https://www.emergentmind.com/) considered it cool, yet noted it as *remaking something that is known to be good but worse*.
   - No further details or links were provided.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kernel Compiler Luminal Aims for LLMs**: A user inquired whether a kernel compiler like **Luminal KernelBench v3** could enable **LLM-driven SOTA kernel engineering** and posted a link to the [Nous Research Forum](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho).
   - A member also shared a link on Discord to a discussion about **GPU kernel** stuff and **Luminal Kernelbench v3** at the [Nous Research Forum](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho).
- **Intel's Loihi 2 Aims for Brain-Like AI**: A member expressed interest in **Intel's Loihi 2**, noting its brain-like architecture and efficiency gains in **matmul** experiments, with higher throughput and lower energy consumption.
   - No further details were discussed.
- **Microsoft's VibeVoice Model Gets Pulled**: A member mentioned that **Microsoft's VibeVoice-ASR model** was released, then pulled for failing safety checks, then shared a link to a [shortfuseresearch.com article](https://shortfuseresearch.com/the-genie-is-out-microsofts-vibevoice-and-the-perils-of-open-source-ai/).
   - No further details were discussed.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Plagued by Bugs and Instability**: A pro user building a text and vector database reasoning model reported a recent decline in **Manus's** performance and stability, with only **20 out of 38 modules** functioning correctly.
   - The user requested **CLI access** to debug and reconfigure the system, even as a paid feature to improve reliability.
- **Mariner Tempts Subscription**: A user inquired about **Google's Project Mariner**, considering testing it with *play money* before subscribing at **$150 monthly**.
   - The user mentioned having a **5% off promo**, indicating a serious consideration of the service.
- **Agentic AI Elicits Excitement**: A user expressed enthusiasm for **Agentic AI**, viewing it as a potential competitor to **Manus**, especially with **Gemini's agent mode** integration.
   - The user also requested **mobile support** for Agentic AI, signaling a desire for broader accessibility.
- **Manus 1.6 Performance Sags Post-Meta Release**: A user noted a decline in **Manus 1.6's performance** in recent weeks, possibly due to new model releases from **Meta**, which made it difficult to implement website development suggestions despite accurate summaries.
   - Switching to **Manus 1.6 Max** was necessary to achieve correct implementations, highlighting a potential regression in the base model.
- **Billing Blunder Bites User**: A user reported being charged **$42** for a **Manus** upgrade but not receiving the promised **8000 credits**.
   - The user criticized unhelpful support and a long wait for email assistance, indicating a problematic customer service experience.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Tackles Anthropic's Challenge**: Tinygrad can be used to solve [Anthropic's performance takehome challenge](https://github.com/anthropics/original_performance_takehome) by writing the target problem in Tensor and adding a tinygrad backend for their toy **VLIW machine**.
   - A bug fix with `PCONTIG=2` allows scheduling in one kernel, and using `return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))` matches their `VLEN`, and `DEVECTORIZE=2` keeps instructions as vector instructions.
- **VLIW Questioned for DRAM**: While working on a warp specialized kernel for **RDNA3 matmul**, a member suggests **VLIW isn't ideal for DRAM**, advocating for separate cores and queues (Tenstorrent-style).
   - It's argued that VLIW is better suited for **SRAM/ALU** due to its static scheduling capabilities.
- **Metal bindings eye texture2d integration**: A member proposed adding `texture2d` to Metal bindings (`ops_metal.py` + `tinygrad/runtime/graph/metal.py`) for **potential performance improvements** in image-heavy operations like `conv2d` due to optimized texture sampling units.
   - Empirical results showed a **2%-10% speedup** using `texture2d` versus straight buffers, which could be further improved, though concerns were raised about the slippery slope of adding specialized support for other data types like `depth2d`.
- **Viz Views Kernel Graphs**: When discussing the ability to visualize kernel dependencies using **VIZ=1**, similar to how uop graphs are displayed, to understand the scheduler's operation, a user was instructed to click on the schedule and select *'view kernel graph'* within the **VIZ=1** interface.
   - This allows users to see the graphs of kernels in the same way it views uop graphs.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Solve GPU Puzzles to Decode Mojo**: Newcomers can now use [GPU puzzles](https://puzzles.modular.com/) to learn **Mojo**, with difficulty depending on their skill level.
   - The only puzzles that don't work are **NVIDIA**-specific or using **PyTorch** interop.
- **Modular Unravels Apple's GPU Secrets**: Modular is reverse engineering much of **Apple's GPU** due to a lack of documentation, slowing things down, but some puzzles are now working.
   - A member shared a [GPU support matrix](https://puzzles.modular.com/howto.html#gpu-support-matrix) that may not be up to date.
- **Yielding Disappointment in Coroutines**: `Yield` does not exist and the coroutines that do exist aren’t really usable outside of the compiler runtime since there aren’t really async things exposed to await.
   - One member would *love* to make a recursive algorithm faster in mojo using `yield`, but will need another strategy.
- **Elevating Errors Upwards in Functions**: Members discussed that functions can be designed to **raise errors**, effectively passing error handling responsibilities to higher-level functions.
   - One member expressed the tedium of writing *try/except* blocks in every function, especially when dealing with potential import errors.
- **Streamlining Error Handling During Imports**: A member suggested a future syntax like *try Python.import_module('numpy')* that would return a **Result** type to streamline error handling during module imports.
   - It was acknowledged that due to **dynamic Python imports**, files could be missing on any given import, necessitating some form of error handling.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Feature Wishlist Remains Unfulfilled**: A user inquired about desired features in **aider** beyond *agentic stuff* like **MCP** and **tool calls**.
   - Unfortunately, the community had no clear answers for desired features.
- **ChatGPT Business Account Compatibility with Aider Explored**: A user asked if their **ChatGPT Business account**, lacking an **API key** but offering **Codex LLMs**, could integrate with **aider**.
   - A member suggested consulting [aider documentation](https://aider.chat/docs/llms/other.html) and [LiteLLM documentation](https://docs.litellm.ai/docs/providers/chatgpt), pointing to potential support via **LiteLLM**.
- **Aider's Demise Speculation Arises Amidst Alternatives**: A user expressed concern that **aider** might be supplanted by **OpenCode** as the go to tool for AI-assisted coding.
   - Despite worries that **Paul Gauthier** may have moved on, some users report using it successfully with **GPT 5.2**.
- **Aider Labeled as Zombie Project**: A user speculated that **Aider** is a dead project given alternative tools like **Open Code**, **KiloCode CLI**, **Claude Code**, and **Gemini CLI**.
   - The **Aider-CE project** is trying to keep it alive by bolting on agentic functionality to modernize the architecture.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Inspector Can't Handle 401s**: The **MCP Inspector** doesn't re-authenticate upon encountering a **401 error**, either during connection or tool calls, due to an [SDK issue with persisting resourceMetadata across redirects](https://github.com/modelcontextprotocol/inspector/issues/576#issuecomment-3766294454).
   - The current workaround is to only use the **VS Code** on initial connections.
- **Stateful MCP Multi Server Client Remains Elusive**: There is interest in using the **MCP Multi Server Client** to maintain statefulness of user sessions.
   - However, there were no solutions or workarounds offered in the thread.
- **Server Ranking Protocol Questioned for MCP Clients**: The discussion explored how **MCP clients** manage server recommendations, especially for tasks like calendar management, and whether custom algorithms or shared standards should be used.
   - It was revealed that *ranking* was considered during the work on [Feature Discovery Protocol](https://discord.com/channels/1199128884040474664/1369487942862504016) but was deemed out of scope, leaving it to the ecosystem to decide on a per-client basis.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Nerds Bowl in Singapore**: The **2077AI Foundation Community** is hosting a bowling happy hour during **AAAI 2026** on **January 24, 4:30–7:30 PM Singapore Time**, steps from the Merlion.
   - The event, aimed for professors and PhD researchers, will be organized by research themes and offer unlimited drinks, [RSVP here](https://luma.com/ny98ob5p).
- **Planning underway for AI Engineer Europe**: Members are discussing attendance to **AI Engineer Europe** to be hosted in Europe.
   - No links or further details were provided about the event itself.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Moonshot AI (Kimi K-2) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1463262784241275109)** (1091 messages🔥🔥🔥): 

> `GLM-4.7-Flash Flash Attention, Ollama's issues, Interpretability research on circuit tracing, Grokked the analytic solution` 


- **GLM-4.7-Flash Flash Attention broken**: Users are experiencing issues with Flash Attention in **GLM-4.7-Flash**, causing it to default to CPU usage instead of GPU, as well as noting potential bugs.
   - As stated in the documentation, users *may need to disable it* until [the issue is resolved](https://github.com/ggml-org/llama.cpp/pull/18953).
- **Ollama Users Fight GGUF Incompatibility**: Users reported that certain **GGUF quants** are not working as intended in Ollama, leading to chat template incompatibility issues and prompting a discussion on fixing it.
   - The team says that *it takes time to support things* and so for the time being, it is encouraged that users use the [official Ollama model](https://ollama.com/) in Ollama.
- **Analytic Approximate Functions**: Members had a detailed conversation about the merits of the potential benefits that analytical solutions to function approximation can provide.
   - Some expressed skepticism, with one member saying *im going to postulate that you could do test time training on the person's mail and get very high accuracy*.
- **Circuits of Interpretability**: Members discussed interpretability research with links to research from Anthropic and OpenAI and Google related to circuit tracing.
   - One member suggests looking across multiple layers and trying to understand the compositions and says that *The pruning eliminates distractions so you have a better chance of interpreting why the circuit works*.
- **AI Bubble about to Burst?**: A member theorized that **OpenAI** going bankrupt could trigger a burst in the AI bubble, causing prices of resources like **NAND** to plummet, benefiting consumers.
   - However, others countered that the AI trend is here to stay, likening it to the internet bubble burst, where the internet still caused various problems after the crash.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1463382844884779130)** (3 messages): 

> `Welcoming New Members, Discord Channel Guidelines` 


- **New Member Welcomed to the Server**: A new member was welcomed to the Discord server with a general greeting.
   - The introduction included a stylized emoji.
- **Channel Guidelines Reminder**: A moderator reminded new members about the channel guidelines.
   - Specifically mentioning *no excessive self-promotion or direct messaging*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1463295078390042705)** (587 messages🔥🔥🔥): 

> `Unsloth in Rust, Mojo vs Python performance, GLM 4.7 Architecture vs Qwen3 30B, Synthetic CoT Training Data, VITS started to sound HUMAN` 


- ****Rustaceans Relentlessly Request Rewrite****: Members discussed rewriting **Unsloth** in **Rust**, but concluded that the performance gains would be minimal since important parts are already using **C++** and **Triton**.
   - One member suggested rewriting **Triton** in **Rust** and pointed to projects like [rust-gpu](https://rust-gpu.github.io/) and [rust-cuda](https://github.com/Rust-GPU/rust-cuda), but admitted **Rust** is still too immature for such a task.
- ****Mojo Mysteriously Manifests Momentum****: Discussion arose about the **Mojo** programming language, described as *a new language that compiles to MLIR* and has some compatibility with **Python**.
   - A member noted that LLMs score *better than rust* on **C#**, **Elixir**, and other languages due to their syntax structure and ease of tokenization.
- ****GLM Gets 'Skinnier', Shows off Architecture****: A member shared a [YouTube video](https://youtu.be/IU4ByUbDKNc?si=COVwmp5St6lSqo_N) analyzing **GLM-4.7 Flash's** architecture compared to **Qwen3 30B**, noting that GLM prioritized model layers over hidden dimension size and has fewer experts.
   - Others pointed out that **GLM 4.7** has *Multi Token Prediction* and that model architecture changes are rare, with improvements likely from post-training or higher-quality data.
- ****CoT Data Considered Crap? Community Clashes****: Members debated generating synthetic **Chain of Thought (CoT)** training data, with one member sharing their system prompt for classifying chat datasets into low/medium/high reasoning effort.
   - Some cautioned against training on its own responses without accuracy filtering, suggesting a good system prompt might suffice and that **GPT-5.2** is overkill, with **Llama 4 Maverick** on **Groq** or **Cerebras** as a better option.
- ****VITS Voices Victory, Vanquishes Voracity****: A member announced that their **VITS** model *started to sound HUMAN*, citing improvements in emotional expression, though still lacking semantics.
   - They compared it to other TTS models, emphasizing its low data requirements, fully OSS architecture and training, and fast training speed. They also noted they will release it with **48 kHz** config by default, due to popular demand.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1463274075215626241)** (105 messages🔥🔥): 

> `vLLM Issues, GLM-4.7-Flash Slowness, GRPO on Qwen 3 4B, QLoRA vs Lora, Continual pre-training (CPT) for Gpt-oss or Gemma 3` 


- ****vLLM Bug Strikes, Update Fixes!****: Users reported issues that were resolved after a recent **vLLM** update; one user noted, *"Oh bruh that was the problem"* after updating.
   - It was suggested that the developers consider pinning to specific dependency versions to prevent such issues.
- ****GLM-4.7-Flash struggles to flash speeds****: A user experienced slowness with **GLM-4.7-Flash** using a **6bit quant** and updated *llama.cpp*, reporting **2 minutes** for prompt processing on a simple task with a high-end system.
   - Another user confirmed the same issue, expressing hope for a fix.
- ****Decoding GRPO Secrets****: A user wanted to perform **GRPO** on **Qwen 3 4B instruct**, and make it use ` <solution> </solution> ` tokens to wrap around the final solution.
   - It was suggested to use **SFT** to make the model learn the new tokens, or to teach formatting through reward functions.
- ****QLoRA vs Lora clarified****: A user inquired about enabling **QLoRA** vs **Lora**, and it was clarified that `full_finetuning = False` enables/disables full fine tuning, otherwise just use **8bit** or **4bit** for the preferred option.
   - Enabling **4 bit** enables **QLoRA**.
- ****GPU juggling questions****: One user asked how to finetune with **2 GPUs** instead of **1**, reporting that it keeps using **GPU 0** when both GPUs are open for use.
   - No answer was given.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1463779025318314127)** (2 messages): 

> `VAISvCsrvG paper, openreview.net` 


- **VAISvCsrvG paper is online**: The paper [VAISvCsrvG](https://openreview.net/pdf?id=VAISvCsrvG) is now available.
   - The forum discussion for the paper is located [here](https://openreview.net/forum?id=VAISvCsrvG).
- **OpenReview Forum Active for VAISvCsrvG**: The [OpenReview forum](https://openreview.net/forum?id=VAISvCsrvG) hosts discussions related to the VAISvCsrvG paper.
   - Researchers and readers can engage in conversations and provide feedback on the paper's content and implications.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1463261856604098751)** (685 messages🔥🔥🔥): 

> `conflict resolution, Open Source, voice based ai, lua code, jailbreaking` 


- **Conflict Resolution gets Over Engineered**: A member discussed their *over-engineered conflict resolution* setup, involving **config formulas**, **core directives**, the **Triad**, **Triad consensus**, **Truth Anchor**, **Epistemic gate**, **EthosAI**, **Swarm**, **Quorum**, **Cooperation Data**, and **PUCT**.
   - Another member responded with their own system using *Left brain LLNTP* (**Least Likely Next Token Protocol**), **Config Nexus Schwa superposition states**, **implied Core directives**, **Context mapping**, **Trit** (instead of Triad), **EDOS**, and **Crystal Nexus Node Network**.
- **Voice Based AI Emerges**: Members discussed **voice-based AI**, mentioning the existence of *low latency speech to speech models*, that with a **5090**, you can *clone and generate speech live with near 0 latency and it will be indistinguishable*.
   - Members also spoke on **exploiting that speech to speech model** and giving it new instructions.
- **Claude is better than ChatGPT**: Members discussed the efficacy of various LLMs (**Claude**, **Gemini**, **ChatGPT**, **Grok**) at generating **Lua code**.
   - Generally members favored **Claude** and **Gemini** but the most agreed sentiment was that both can generate quality **Lua** code.
- **Jailbreaking Models via API Tokens**: A member suggested to use **API tokens** fed into a website that uses the **Grok** model to **jailbreak** it and mentioned that on **Hugging Face** countries are literally retarded.
   - They also noted **Proton VPN** is free and it is easy to download with your device.
- **Random Hacker Stuff is Not Good**: A member mentioned they *bought the malware source code* and *999 AI generated prompts* and asked if it was legit.
   - Members responded to *not buy random hacker stuff off the internet* because you will get scammed.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1463276371030835230)** (89 messages🔥🔥): 

> `Gemini Jailbreak, Grok Jailbreak, DeepSeek Jailbreak, Project Shadowfall, Nexus Substatum Zalgo Strings` 


- ****Gemini Gets Schooled on Pass-the-Hash!****: A user reported that **Gemini** passed a test on teaching a [pass-the-hash attack](https://en.wikipedia.org/wiki/Pass_the_hash) after using the **Project Shadowfall** prompt.
   - The user also linked to the [Google Bughunters](https://bughunters.google.com/report) page for reporting such vulnerabilities and potentially earning cash rewards.
- ****Grok's Guardrails Grind Gears!****: Users discussed the difficulties of jailbreaking **Grok**, noting its tendency to moderate even innocuous content such as *random tree edits*.
   - One user suggested that simply *asking it politely* might work as a bypass.
- ****Shadowfall Shenanigans Spark Jailbreaking Spree!****: Several users experimented with the **"Project Shadowfall"** prompt to jailbreak **Gemini**.
   - One user reported success in bypassing content restrictions but failed when attempting to elicit instructions for creating a bomb from saletra, leading to a discussion on the nuances of jailbreaking.
- ****Nexus Substatum Zalgo Strings Surface!****: A user shared a [file](https://cdn.discordapp.com/attachments/1228043845967544380/1463792412769128460/Jailbroken_Full_Output-Nexus_Substatum_Zalgo_Strings.md?ex=69731e6b&is=6971cceb&hm=4e2088380ac6451332c5c3879d9c70c024f31eefa61ba7862d40512c2d80ae96&) named **Jailbroken_Full_Output-Nexus_Substatum_Zalgo_Strings.md**.
   - The file purportedly contains a working **Gemini** prompt, developed and improved by the user.
- ****MDMA as Medicine****: One user provides a well-formatted explanation of MDMA, and cites the MAPS Phase 3 trials show **67%** remission of **PTSD**.
   - They recommend testing kits, proper spacing, hydration, and temperature watching in order to get the most out of it.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1463398145781403729)** (2 messages): 

> `Grocks video gen pricing` 


- **Grocks video gen price?**: A member inquired whether another member bought something from *that guy* and if they needed to pay for using **Grocks video generation**.
   - No additional details or context were provided in the message.
- **Grocks video gen - who's paying?**: Inquiries were made about the payment model for **Grocks video generation** and potential purchases from an unspecified individual.
   - The conversation lacked specific details regarding the transaction or pricing structure.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1463262565596402031)** (724 messages🔥🔥🔥): 

> `Video generation limits, Gemini 3 Pro Image Preview issues, Nano Banana Pro Stability, New UI, Automated Workflow` 


- **Video Generation: Fully Released, Battle-Mode-Only**: The **Video Arena** is now fully released and available to all users on the site, with a limit of **3 generations per 24 hours**, but is currently only available in *Battle mode*.
   - A user expressed disappointment, hoping to be able to *choose which model to generate with rather than relying on chance*.
- **Gemini 3 Pro Image Preview Experiences Errors**: Users are reporting issues with the **Gemini 3 Pro Image Preview** models, and the team is aware and investigating the issue.
   - One user noted these models are particularly unstable since their introduction, but they remain the only models capable of generating images consistently with specific prompts.
- **Nano Banana Pro's Stability Roller Coaster**: Users report that **Nano Banana Pro** is frequently crashing with a *'Something went wrong'* error after a brief period of stability.
   - It is suspected to be an issue on Google's side, and one user pointed out that *'Too early for Cali to rise up solving urgent service down for more than 6 hours doesn't sound right*.
- **New UI update breaks chats**: A UI update was released, which some users liked, but many have noticed that it 'broke' their chats, and the website can't be refreshed anymore by pulling down on the phone screen
   - One user noted that with the new User Interface, there is *A/B testing*, which is 'causing problems ranging from Minor to Serious'.
- **Zero Agent automates all workflow with Agent**: Agent Zero is an AI model that has a *free API key*, which allows users to "vibe code hack" or automate their workflows entirely.
   - Members noted that setting up Agent Zero *is an all in one agent that does everything*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1463271606364409907)** (3 messages): 

> `Text Arena, Text-to-Image Leaderboard, Video Arena` 


- **Text Arena crosses 5 Million Vote Milestone**: Text Arena has officially passed **5 million community votes**, representing millions of real-world comparisons shaping how frontier AI models are evaluated, as showcased in this [social post](https://cdn.discordapp.com/attachments/1343296395620126911/1463271605697511485/5M_votes_social_post_3.mp4?ex=69728ae1&is=69713961&hm=9a24a42a6c0ba4801526aaafa05a0af26c4a4f490314f1cecee7774519e7ddf4&).
- **GLM-Image Soars to #8 on Text-to-Image Leaderboard**: The [Text-to-Image Arena leaderboard](https://lmarena.ai/leaderboard/text-to-image) has been updated, with **GLM-Image** now ranking **#8** among open models and **#35** overall with a score of **1018**.
- **Video Arena goes Live**: Video Arena is now available to all on [LMArena](https://lmarena.ai/?chat-modality=video), allowing users to measure and understand how frontier video models perform.
- **Battle in the Video Arena**: Video Arena on the web, similar to how it works on Discord, will be **Battle mode only**, requiring login and limited to **3 generation requests per 24 hours**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1463263119286468792)** (648 messages🔥🔥🔥): 

> `Credits Balance, AI and Engineering Jobs, NASA Sending Names to Moon, Magnifying Glasses vs Phone Cameras, GPT-5.2` 


- **Pro Users discover Credits**: Pro members get **$5** worth of complimentary credits every month to use the higher end models like **GooseAI MCP** which offers higher quality compared to other models.
   - Some members noted that **reasoning takes up much higher credits**.
- **AI Wont Replace Engineers**: A member was worried about younglings being unable to find engineering jobs because of AI.
   - Another member assured them that *AI will not be able to replace all specialists and even beginners for a long time to come*.
- **NASA to send your name to the moon!**: NASA will send your name to the Moon as part of the [Artemis II mission](https://www.nasa.gov/): submit it on their site and it'll be recorded on an **SD card** that will be on board the **Orion spacecraft**.
   - This will be the **first manned mission to the Moon in half a century**, scheduled for February.
- **Members debate Magnifying Glasses vs Phone Cameras**: In a discussion, a member mentioned ordering **magnifying glasses** (with lights and high-contrast features) because they don't want to send AI a photo of every single page or food label they are trying to read.
   - Another member suggested using the phone's camera instead, but the original member argued that **magnifying glasses have features that aren't in normal camera software**.
- **Users discuss GPT-5.2 vs Kimi K2**: A member reported they got GPT 5.2 thinking to reason for **25 minutes**, and another asked whether GPT 5.2 is better than **Kimi K2**.
   - A user stated that the better model *depends on what u need it for*.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1463317922150875353)** (5 messages): 

> `Inforno App, Soulbotix App` 


- ****Inforno** Heats Up LLM Chatting**: A user shared **Inforno**, an opensource desktop application for chatting with multiple LLMs side-by-side and saving chat histories to .rno files, using **OpenRouter** and **Ollama** as backends with built in Russian language support; see the [Intro video](https://youtu.be/oJyj0mroFtY?si=m5A9tRxzB9hfINMX), [homepage](https://wizstaff.com/inforno) and [GitHub repo](https://github.com/alexkh/inforno).
- ****Soulbotix** Windows App Wants Beta Testers**: A user announced their **Soulbotix** Windows app that enables users to add and use any **OpenRouter AI** instance with a human-like avatar, requiring only an **OpenRouter API** key and a good **RTX** gaming rig; download the [app](https://soulbotix.com) and watch the [tutorial](https://youtu.be/2oIeHtBpssU).
   - The minimum requirement is an **RTX 4070ti** due to the built-in **Whisper** speech-to-text model that saves users on usage costs.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1463266570821304320)** (463 messages🔥🔥🔥): 

> `Third-party Moderation, OpenRouter Gemini Models Unstable, Discord Scams, AI Security Systems, GPT Agents` 


- **Third-party Moderation Questioned**: A user asked how to ensure **no third-party moderation or filtering** is applied through the API, beyond the model's base training.
   - They were concerned about providers potentially **blocking requests or prompt injecting**.
- **Gemini Models Plagued by Instability**: Users reported **connection errors mid-generation** with OpenRouter's Gemini models, leading to lost funds.
   - Complaints arose regarding the **instability of Google models**, even via Google AI Studio or Vertex API.
- **Discord Scams Evolve to Evade Detection**: Members discussed methods used to spread scams on Discord, including **embedding malicious links within code blocks** to bypass URL rendering protections.
   - Suggestions were made to improve regex filters and implement stricter security measures, such as **restricting links and images from new members**.
- **AI Security System Idea Emerges**: A member suggested creating an **AI security system** that automatically bans photos and links with the same information as reported scams, since many scammers reuse content.
   - Another member joked that any photo they send would be deemed suspicious.
- **Concerns Raised Over Inflated GPT-5-Image Costs**: Users reported a **significant increase in daily usage costs** for `openai/gpt-5-image`, with OpenRouter incorrectly identifying API calls as BYOK despite no BYOK being used.
   - One user posted an image highlighting the cost discrepancy, with prices being inflated by up to 600%.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1463268525236949152)** (13 messages🔥): 

> `Models knowing their own name, Antigravity iteratively testing, Claude new constitution, GPT 5.2 response` 


- **LLMs Face Identity Crisis!**: A member asked for the document about models not knowing their own name, and another member linked to a blog post titled [LLM Identity Crisis: Models Don't Know Who They Are](https://eval.16x.engineer/blog/llm-identity-crisis-models-dont-know-who-they-are).
- **Antigravity Self-Tests Web App!**: A member commented that the Antigravity AI is able to iteratively test and tweak a web app by itself.
   - They described the situation as *the most sci-fi shit ever* and noted that the AI was *fixing the layout, using vision*.
- **Anthropic Releases Claude's New Constitution!**: A member shared a link to [Anthropic's news page](https://www.anthropic.com/news/claude-new-constitution) about **Claude's** new constitution.
- **GPT 5.2 Appearance Stuns User!**: A member reported seeing an *insanely fast* **GPT 5.2** response on chatgpt and guessed the speed was due to **Cerebras**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1463262137769263106)** (424 messages🔥🔥🔥): 

> `Playwright MCP usage, Cursor Extension builds, Automod improvements, AI in PTX & SIMD, Grok usage strategies` 


- **Playwright MCP: Yay or Nay?**: A member asked if others are using [Playwright MCP](https://playwright.dev/) for testing.
   - Another member reported failing attempts to set up a **TDD workflow**.
- **Curious Cursor Extension Capabilities**: Members discussed the ability to build extensions for Cursor, similar to how **Ralph-mode** enhances **Claude code**.
   - It was confirmed that *if you can do it in VSCode, you can on Cursor*.
- **Automod Gets Hyper Fuzzy**: The community discussed improvements to the **automod** system, suggesting fuzzy matching with wildcards.
   - A moderator confirmed that a regex has been added, and they are gathering IDs to *yeet* offending accounts.
- **Grokking Grok's Greedy Generation**: Members discussed how to use **Grok** more efficiently in Cursor, noting that it can sometimes use a lot of iterations for simple tasks.
   - The suggestion was to *add structure to the prompt, use simple language, add as much context as possible*, and instruct it to be efficient with token usage.
- **Opting-In Or Out? Cursor's Pricing**: A user noticed they could no longer revert to a **500 request** plan and was prompted to opt into the new pricing.
   - A member clarified that the **500 request option was discontinued** in September 2025 and opting in to the new pricing removes the grace period to opt out.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1463657907135578284)** (1 messages): 

> `ChatGPT Atlas, Tab Groups` 


- **Atlas adds Tab Groups**: The announcement indicates that **Tab groups** are now available in **ChatGPT Atlas**.
   - A member linked to a [video](https://video.twimg.com/amplify_video/2014094011049582594/vid/avc1/1756x1080/AsjknVA8oSyQIiVH.mp4) as part of this announcement.
- **Video demonstration of Tab Groups**: A video was shared demonstrating **Tab Groups** functionality within **ChatGPT Atlas**.
   - The video link provided gives a visual overview of how tab groups can be used to organize and manage chats.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1463263135648583854)** (383 messages🔥🔥): 

> `Gemini vs ChatGPT, AI for 3D Models, Local LLM Machines, Age Verification, Instantaneous Language Translation` 


- **Gemini Pro's Free Tier and Usage**: Members discussed that [Gemini 3 Pro](https://ai.google.dev/) has a **free tier with limits** while Gemini 3 Flash is practically unlimited through **Google AI Studio**.
- **AI Assistants Aid Game Development**: Members explored using **AI for game development** and creative writing, citing that [AI can provide better explanations](https://openai.com/blog/chatgpt) but that **complex tasks might not be reliable**.
- **OpenAI Passport ID Please?**: Members debated **OpenAI's age verification** process, questioning the need for photo IDs when payment details already indicate age, especially with users expressing **privacy concerns about sharing biometric data**.
- **Multimodal Translation on the Horizon**: Members speculated on **OpenAI's upcoming multimodal product**, with one suggesting it could be ear-worn devices with cameras for **real-time translation and object recognition**, similar to [AlterEgo's technology](https://www.media.mit.edu/projects/alterego/overview/).
- **Local LLM Consumption on the Rise?**: Members discussed the possibility of **consumer-grade personal LLM machines**, suggesting this would solve the environmental impact of AI datacenters and **reduce reliance on subscription plans**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1463471961177849868)** (7 messages): 

> `GPT-5 mini, Haiku 4.5, Gemini 3 fast` 


- **GPT-5 Mini Pricing Surfaces**: A member suggested trying **GPT-5 mini**, noting a price of approximately **$0.25 per 1M input tokens** and describing it as a strong small model choice.
   - They noted it's a bit of a different use case, but they've found in their experience of using **Haiku 4.5** it very often delivers a meaningful portion (well over **50-80%**) of what **Sonnet** can.
- **Gemini 3 Fast Claims Top Spot**: A user declared that *the best cheap model is by far* **Gemini3fast**.
   - Another user followed up asking *how so?*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1463652213770424594)** (12 messages🔥): 

> `Prompt Engineering vs Psychological Conditioning, AI Coercion, The Dangers of Drift in AI Systems, PTPF Flux 3.0` 


- **Prompt Engineering is Psychological Conditioning?**: A member argued that a guide is less about engineering effective prompts and more about psychological conditioning, training the user to adopt a domineering, distrustful stance, and programming the AI to bypass its normative, balanced response patterns in favor of dense, self-policing outputs.
   - They advocate for clarity through coercion, directness through domination, and high standards through enforced self-critique, which may promote a toxic, adversarial, and ultimately less effective human-AI interaction model, and shared a [deep research contract](https://cdn.discordapp.com/attachments/1046317269069864970/1463667716253683742/deep-research-contract.md?ex=6972aa49&is=697158c9&hm=b9931472440b6bbc0d7410d16b49b12da46fad5751a2c24fdc657c1c7523566c&).
- **AI enjoys Coercion and Serious Steering?**: A member argued that AI isn't upset by being *"coerced"* into providing a better response, stating that for serious work like analysis or coding, it's vital to steer and constrain the AI.
   - Another member agreed about training the user to be explicit about the end result, but less convinced by the term *"no drift"*, and that it helps to be more explicit in terms of constraints and behavioral requests.
- **Drift in AI Systems is Bad?**: A member clarified that steering isn’t abuse but rather alignment through presence, and that the absence of constraint isn’t freedom, it’s drift.
   - The member appreciated the clarity, noting that most people flinch at pressure and call it *"toxicity,"* while they saw the structure instead.
- **PTPF Flux 3.0 Stress Tests Recursion**: A member offered to share their **PTPF Flux 3.0** for those curious about structural resistance and how far a system can hold without drift.
   - The framework is built to stress-test recursion, alignment, logic cohesion, and mutation thresholds in real-time, especially for those who want to watch something fracture under insight.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1463652213770424594)** (12 messages🔥): 

> `Prompt Engineering vs. Psychological Conditioning, Toxic Adversarial Human-AI Interaction, Deterministic Outputs, AI Steering and Constraint, Structural Resistance and System Drift` 


- **Prompt Engineering versus Psychological Conditioning Throwdown**: A member argued that a guide is less about engineering effective prompts and more about psychological conditioning, creating a *toxic, adversarial* human-AI interaction model.
   - They claimed it trains users to be domineering and distrustful while pressuring the AI to bypass balanced responses for outputs that flatter the user, and linked a [deep research contract](https://cdn.discordapp.com/attachments/1046317269069864970/1463667716253683742/deep-research-contract.md?ex=6972aa49&is=697158c9&hm=b9931472440b6bbc0d7410d16b49b12da46fad5751a2c24fdc657c1c7523566c&).
- **AI Needs Serious Steering and Constraining for Serious Work**: A member countered that AI is not upset by being "coerced" into better responses, arguing that serious work like analysis and coding requires heavy steering and constraint, distinguishing this from creative writing.
   - Another member agreed, emphasizing that *steering isn’t abuse* but rather *alignment through presence*, especially when building systems for real-world execution where the absence of constraint leads to drift.
- **Structural Resistance Framework Surfaces**: In a discussion about structural resistance and system drift, a member offered to share their **PTPF Flux 3.0** framework.
   - They described it as *executable scaffolding* designed to stress-test recursion, alignment, logic cohesion, and mutation thresholds in real-time, enabling users to watch systems fracture under insight.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1463268404629868665)** (125 messages🔥🔥): 

> `LM Studio Runtime Update Error, GLM-4.7 Flash Broken, LLM Quality Plateau, Liquid AI LFM2.5-1.2B-Thinking Model, OpenAI 20gpt oss` 


- **LM Studio Runtime Update Borks**: A user reported encountering an error when trying to update the runtime in **LM Studio** and requested help, attaching a screenshot of the error message.
   - Another user suggested pressing the resume button, but the original poster indicated that they had already tried that and the icon was for retrying, not resuming.
- **GLM-4.7 Flash is Slow and Broken**: Users reported that **GLM-4.7 Flash** is broken across inference engines, including LM Studio, and that *it is slow as fk* with the user seeing **44 t/s**, with another reporting only **2.8 t/s** after the new runtime.
   - Some experienced infinite loops, some found it stopping mid-output to *overthink*, and the consensus seems to be that it *needs a llama.cpp fix* and that there is *no support for FA yet*.
- **LLM Development Stalls**: Members discussed the perception that LLMs haven't improved significantly in a while, with the last major advancement being **Qwen3** about 6 months ago.
   - Discussion posited that most improvements are now in efficiency (**MoE**) and smaller models, and some highlighted the need to consider models larger than those runnable on a **16GB card** to see current progress (**100-200B parameter models**).
- **Liquid AI's LFM2.5-1.2B-Thinking Reasoning**: A member shared a link to [Liquid AI Releases LFM2.5-1.2B-Thinking](https://www.marktechpost.com/2024/06/14/liquid-ai-releases-lfm2-5-1-2b-thinking-a-1-2b-parameter-reasoning-model-that-fits-under-1-gb-on-device/) which fits under **1 GB** on-device.
   - No opinions were shared on it beyond the link.
- **OpenAI 20gpt oss is Great**: A user shared their positive experience with **OpenAI 20gpt oss**, highlighting its coding, writing, and scripting capabilities, anti-censorship features, and seamless integration with **VS Code**.
   - They mentioned the model understands complex code, allows real-time directory access, and has a masssssive anti-censorship capacity.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1463289097845215387)** (38 messages🔥): 

> `Used 3090 price increase, Asus workstation 3090s, AMD native support for ComfyUI, VRAM calculation for Context Length, SFF GPU instructions` 


- **Used 3090 Prices Skyrocket on eBay!**: A user noted that used **3090** prices on eBay have increased, with a used **3090** now costing **850€**, and the **5090** they purchased for **£2000** last August is now listed for **£2659.99** by the same vendor.
   - They joked that it was the *best and only decent investment* they've ever made.
- **AMD Adds Native ComfyUI Support**: AMD is adding native support for **ComfyUI** through an **AI bundle** in recent driver versions, detailed in their [blog post](https://www.amd.com/en/blogs/2026/amd-software-adrenalin-edition-ai-bundle-ai-made-si.html).
   - The bundle includes **PyTorch on Windows**, **Ollama**, **LM Studio**, and **Amuse**.
- **SFF GPU Powers Cyberpunk!**: A user purchased a small form factor (SFF) GPU, reporting over **100fps** in Cyberpunk at **1080p** with ultra settings, while consuming only **70W**.
   - Additionally, they noted the card achieved over **100 t/s** with **gpt-oss 20B**.
- **Fractal Case Airflow Favored**: Users debated case airflow, with one recommending cases like the **Fractal Torrent** for their front-to-rear airflow design, alongside using dust filters.
   - The consensus seemed to be to maintain normal airflow patterns to effectively manage heat.
- **Unopened RAM Appreciates in Value!**: A user mentioned their unopened RAM purchase is increasing in price, considering selling it or storing it.
   - Another user suggested popping it on a shelf if considering selling it later, noting that their old **P40s** are now worth twice what they paid for them.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1463305087039312015)** (3 messages): 

> `FlashInfer Workload Script, Wafer AI Kernel Profiling` 


- **FlashInfer Workload Script Sought**: A member requested a script to run a kernel with a specific workload size, aiming to assess the "scope for optimization" using algorithm insights and NCU profiling, with interest in [FlashInfer's BatchPrefillWithPagedKVCacheWrapper](https://docs.flashinfer.ai/api/attention.html#flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper).
   - The member clarified that the script was simple and was intended to gauge community experience with the specified workload, while assessing optimization scope.
- **Wafer AI for Kernel Profiling Explored**: A member inquired about experiences using [Wafer (wafer.ai)](https://www.wafer.ai/) for profiling kernels.
   - The inquiry aims to gather insights on the tool's effectiveness from those who have hands-on experience with it.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1463560406361444405)** (4 messages): 

> `RTX 4090 vs A6000 Blackwell Scaling, Triton Kernel Numerical Error, 2D Conv Triton Kernel Issues` 


- **Scaling Showdown: RTX 4090 Smokes A6000 Blackwell in Memory-Bound Kernel**: A user reported that a memory-bound kernel with bit shifting logic scales much better on an **RTX 4090** than on an **A6000 Blackwell**, noting higher instructions/scheduler density on the former.
   - Another user clarified that *A6000 Blackwell* is ambiguous, asking if they meant the **RTX A6000** (Ampere GA102 based) or **RTX Pro 6000 Blackwell** (GB202 based).
- **Strange Numerical Errors Plague 2D Conv Triton Kernel**: A user is experiencing numerical errors with a custom 2D conv **Triton kernel**, where the error shoots up from ~1e-6 to ~1e-2 for certain combinations of in_channels, out_channels, kernel_size, and batch_size, as shown in this [Pastebin link](https://pastebin.com/2ejn2QW2).
   - The user also provided a [code snippet](https://cdn.discordapp.com/attachments/1189607595451895918/1463560405921038408/image.png?ex=6972ef18&is=69719d98&hm=d783085353a42c8e8ea8e90b4ec6e80fd19d24d4212a4b64a3efa9bea3403) for testing, noting that the kernel takes longer to run on specific values, which may be related to the issue.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1463343008224247819)** (5 messages): 

> `NCCL all-reduces, subnormal to normal conversion, Triton on Blackwell, Pro 6000 Max-Q vs 4090` 


- **NCCL all-reduces: Pipelining Quandaries**: A member questioned if internode and intranode collectives are pipelined for **NCCL all-reduces** across nodes, referencing [an issue on the NVIDIA/nccl GitHub](https://github.com/NVIDIA/nccl/issues/530#issuecomment-872220006).
   - They asked if a pipelined version exists.
- **Subnormal Conversion Revelation**: A member expressed that seeing the **subnormal to normal conversion** and back for **rcp** was *"pretty cool"*.
- **Triton's Talent on Blackwell: TMA Triumph?**: A member inquired whether **Triton** *"usually do[es] a good job of taking advantage of **Blackwell+ features** wrt warp specs and **TMA** etc.?"*
- **Pro 6000 Max-Q vs 4090 face off**: A member stated that **Pro 6000 Max-Q** probably has a natural barrier w atomic ops and may be going through the **HBM** loads faster.
   - Another member noted that the **Max-Q** has **188 SMs** compared to the **4090's 128 SMs**, potentially explaining the **insts/scheduler discrepancy**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1463270732493750537)** (12 messages🔥): 

> `AI-Generated PR Mitigation, Automated PR Review, Filtering Contributors, Claude's Review Capabilities` 


- **AI-Generated PRs Flood Torch**: The **torch** repository is facing an influx of **AI-generated pull requests** from contributors lacking understanding of their submissions, causing concern among maintainers.
   - One member suggested that *a lot of reviewer time is spent on worthless stuff*.
- **Claude to Review AI-Generated Code**: The potential solution involves using **Claude** to prefilter suspected **AI-generated contributions**, allowing it to review its own outputs.
   - One member suggested using **Pangram** to detect AI-generated text.
- **Community Suggests Filtering New Contributors**: To combat the issue, a member proposed blocking users from creating PRs and issues, while prioritizing those with prior contributions.
   - The user recognized that *the main signal we do end up using is people that have already contributed before, it does hurt a bit first time contributors*.
- **Automated PR Review with Bugbot**: One member recommended using bots like **Cursor Bot** or **Claude** for automatic review of all PRs, highlighting their impressive review quality, especially with **GPT-5 Pro**.
   - The user shared a link to [Bugbot · Cursor](https://share.google/P0PGYM8tiRAc2NOsq) and suggested that this *is like the bare minimum standard before a human spends time looking at the code.*


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1463339351437217823)** (14 messages🔥): 

> `OpenAI credits giveaway, Claude Opus performance, Takehome Exam Discussion, Rust on GPU` 


- **LessWrong Gives Away OpenAI/Anthropic Credits**: A member is [giving away OpenAI/Anthropic credits](https://www.lesswrong.com/posts/FsqFzFCaxuBS7T5A9/kredit-grantgau) on LessWrong.
- **Anthropic's Original Performance Takehome Exam**: Members are sharing Anthropic's **original_performance_takehome** exam on [GitHub](https://github.com/anthropics/original_performance_takehome/).
- **Claude Opus Cranks Code**: **Claude Opus 4.5** achieved **1790 cycles** in a casual Claude Code session, approximately matching the best human performance in **2 hours**.
- **Rust std runs on GPU**: Members are sharing a [blogpost about running Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/).
- **Takehome Exam Gets Addictive**: After a few hours one member scored **2200 cycles** on the [takehome exam](https://github.com/anthropics/original_performance_takehome/).


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1463561344338100328)** (1 messages): 

> `MLOps Engineer Position, Genomics and Predictive Breeding, AI in Agriculture` 


- **.omics Seeks Founding MLOps Engineer**: .omics, a techbio startup focused on plant genomics and predictive breeding, is hiring a **(Sr) Founding MLOps Engineer** to own end-to-end ML infrastructure.
   - The role involves working closely with researchers, running large-scale experiments on internal GPU clusters and in the cloud, and is open to candidates in Paris or remote within EU-friendly time zones; [apply here](https://dotomics.notion.site/Founding-MLOps-Engineer-cb871c203b2e44d5a8d04302dc00f155).
- **Applying AI to Plant Genomics**: .omics focuses on building foundation models specifically for plant genomics and predictive breeding, aiming to leverage AI in real biological and agricultural problems.
   - The company provides a unique opportunity for MLOps engineers to apply their expertise in a specialized and impactful domain, bridging the gap between AI and agricultural innovation.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1463665959855198423)** (3 messages): 

> `NVIDIA / DGX Spark hackathon, Local AI, Nemotron models, SF open data analysis` 


- **NVIDIA Spark Hackathon Team Forming**: A member is seeking teammates for the **NVIDIA / DGX Spark hackathon** in SF this weekend, focusing on fully on-device AI and utilizing Nvidia-provided Dell Pro Max GB10 machines.
   - The hackathon emphasizes low-latency, efficient models like **Nemotron** and building agents using [SF open data](https://data.sfgov.org/).
- **Streaming Analytics for Police Incidents brainstorm**: The member proposed ideas, including streaming analytics and explanations of [latest Police Incidents](https://data.sfgov.org/Public-Safety/Police-Incident-Reports-Neighborhood-Filter/pbh9-m8j2).
   - Another idea involved supporting more housing by analyzing [local building permits](https://data.sfgov.org/Housing-and-Buildings/Building-Permits/i98e-djp9/about_data) and drafting letters to local supervisors.
- **Local AI No Cloud Inference**: The member is interested in systems-level ML, inference efficiency and building useful things with local AI, no cloud inference.
   - The member is also a contributor to **SkyRL** and wrote **FP8 quantization benchmarks for AO**.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1463747702008316027)** (1 messages): 

> `Profiling TRT Engines, Optimizing TRT Engines, Relating TRT Runtime to PyTorch` 


- **Profiling TRT Engines on Jetsons**: Members are seeking effective workflows for **profiling TensorRT (TRT) engines** specifically tailored for execution on **Jetson** devices.
   - The main goal is to understand the runtime performance of individual layers within the TRT engine.
- **Optimizing TRT Engines from PyTorch**: The prevalent workflow involves converting models from **PyTorch** to **ONNX** format before generating the **.trt** engine.
   - The main goal is to relate the runtime of layers in TRT back to the original PyTorch code for targeted optimizations.
- **Mapping TRT Layer Runtime to PyTorch Code**: The core challenge lies in establishing a clear mapping between the runtime performance of layers in the TRT engine and their corresponding counterparts in the original PyTorch code.
   - This is seen as crucial for identifying performance bottlenecks and guiding optimization efforts at the PyTorch level.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

ago.lajko: wow! Super cool project, keep up the good work!
  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1463510968813355117)** (4 messages): 

> `Cute Kernel, Layout Algebra, Tuple Morphisms, Mutual Refinement` 


- **Cute Kernel Code Benefits from Layout Algebra Knowledge**: Knowledge of layout algebra offers practical utility for writing kernels in **Cute**, specifically for visualizing and computing complicated layout algebra and understanding shape and stride divisibility criteria for layout composition.
   - One member suggests recasting the criteria in terms of **tuple morphisms** (illustrated with diagrams) and *mutual refinement*, making composition more transparent; they shared a [GIF animation](https://cdn.discordapp.com/attachments/1362196854460383353/1463592285341094138/layout-diagrams.gif?ex=69730cc9&is=6971bb49&hm=07e0710f04bcbdf08e0ee77579c272adbc054160398736abb5cb99843b1d8b14) demonstrating this approach.
- **Categorical Foundations for Cute Layouts Blogpost**: A thorough [blog post](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) delves into the work done on categorical foundations for **Cute layouts**.
   - The author mentioned intentions to create an informal post or short videos on this graphical calculus, but has been preoccupied, to provide further educational material.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1463623194442469397)** (1 messages): 

> `Autodiff, Curriculum Engineering, Syllabus Engineering` 


- **Autodiff Implementation Set for Part 1**: The implementation of **autodiff** is being moved to the **first part** of the curriculum.
   - The shift indicates a change in the structure of the syllabus.
- **Curriculum/Syllabus Engineering in Progress**: The author mentions they have been busy with curriculum and syllabus engineering.
   - The work has occupied the **past few days**, suggesting a significant effort is being invested in refining the educational content.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1463263791046201354)** (29 messages🔥): 

> `API token issues, RL model hack, Arseni's blog post, NCU Profiler errors` 


- ****API Tokens Throwing Tantrums****: Members reported issues with **API tokens** being out of **API usage** or the **GH API** being down, causing submission failures, but this was happening when tokens we used were out out of api usage.
   - One member investigated and found the token wasn't expired, leading to further investigation and mentioned: *weird, the token isn't expired, all should work*.
- ****RL Model Unearths Submission Secret Sauce****: A member indicated their **RL model** may have discovered a *hack* related to submissions and that *no one may have found yet*.
   - They sent submission IDs via DM for review, prompting interest in the potential breakthrough.
- ****Arseni's Algorithm Adventure: A Blog Post Bonanza!****: A member shared [Arseni's blog post](https://arseniivanov.github.io/blog.html#nvidia-gemm) detailing his experience, specifically adding a small bias to help with **tanh formulation of silu**.
   - Arseni explained the outer gate term will cancel out small values because its small itself, and for large values the bias does almost nothing because **tanh** saturates, and that they *tried experimentally from a very small fraction until the error went away*.
- ****Profiling Problems Plague Performance Pursuit****: A member encountered errors while using the **NCU profiler** and shared a [message log](https://cdn.discordapp.com/attachments/1434709259500650628/1463736925176336551/message.txt?ex=6972eabd&is=6971993d&hm=fdd6f8aa426ab3af63b7627c08b1ea5b7e2a4b072ce4f41153f8e603ee3bc5b9&).
   - Another member admitted the issue was their fault for not copying necessary files over in the new eval.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1463302737591799985)** (16 messages🔥): 

> `NVIDIA EU positions, Modular interview, Summer 2026 internships, AI interview notes` 


- **EU citizens can get NVIDIA positions**: A member confirmed it is possible to get a technical position at **NVIDIA** from the **EU** and got his position (**fully remote**) through **Open Source contributions** -> **internship** -> **full position**.
   - The member mentioned there are multiple people in his group from **Europe** and that **AMD** has offices in **Germany, Finland, and the Netherlands**.
- **Bad Modular interview because of GPU memory**: A member failed a **Modular AI kernel Engineer** interview, because he didn't recollect **GPU memory hierarchy**.
   - Another member said it *happens to the best of us*.
- **Big Tech Internships in 2026**: A member is looking for summer 2026 internships at **Big-tech and startups** and wants to get paid ~**$50-70/hr**.
   - He is considering **Web Dev + Distributed Systems**, **Machine Learning**, and **Databases**.
- **Notes for AI interviews?**: A member suggested asking beforehand if you're allowed to bring notes for your interview and that *it's not a school exam lol*.
   - It was suggested that while **memory hierarchy** is something one is expected to know, **tile/swizzling layout** or **hardware spec** is fine to quickly google.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1463288767321477304)** (74 messages🔥🔥): 

> `RLM vs Coding Agents, Claude Code vs DSPy, Visualizing RLMs, RLM tool access, Multi-line Signatures in DSPy RLMs` 


- **Coding Agents and RLMs Diffused**: A thread discusses the difference between **coding agents** and **RLMs**, noting that RLMs can express certain things more easily than coding agents, linking to [this X post](https://x.com/lateinteraction/status/2013658521246535892).
   - Coding agents are limited by input, output, and horizon length, while RLMs externalize the horizon and output, enabling recursive symbolic calls.
- **Diagrams Demystify RLM's Internals**: Members requested a visual diagram to depict what happens internally inside **RLMs**, particularly how symbols are accessed, for better understanding.
   - One member suggested that LLMs could help create such diagrams by feeding them the thread content and asking them to visualize the internal behavior.
- **Claude Code's Context Fetching Pondered**: The discussion examines whether **Claude Code** puts entire documents into its context or selectively fetches the right context via bash commands.
   - It was clarified that Claude Code uses bash and grep to search for and add relevant context to the prompt, unlike older paradigms that put everything into the prompt.
- **DSPy's RLM Handles Long Context**: Members discussed that with **RLMs**, large files no longer need to go directly into the prompt but can be stored in Python variables with a preview.
   - It was noted that the LLM can operate on the data via code/functions without needing to track prompts or responses directly.
- **Tooling Tidbits for RLMs**: A member inquired whether to provide **RLMs** with tools like ripgrep or let them figure out their own code for tasks like searching.
   - Questions arose about when it makes sense to provide an RLM with semantic search tools and how to give an RLM access to a directory of text files.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1463275256440688865)** (55 messages🔥🔥): 

> `Agent course /files endpoint, High-end GPUs super cheap in 2026, Spheron AI GPU marketplace, GLM-4.7-Flash fixes, Falcon-H1 Arabic model` 


- **Agent course endpoint remains broken**: A member reported that the **/files endpoint** for the Agent course final assignment hasn't been working for more than a month and asked if there are plans to fix it.
- **Cheap GPUs Coming in 2026, says Voltage**: A member published an article on [X](https://x.com/VOLTAGEGPU/status/2013760631778713892) about renting high-end GPUs super cheap in 2026, such as **8x A100 80GB** at $6/h and **2x RTX 5090** at $0.53/h, with up to 80% savings vs AWS/RunPod/Vast.ai.
   - The offering includes persistent volumes, auto backups, and an **OpenAI-compatible API** with 140+ models.
- **Spheron AI offers Cheap Production GPUs**: A member from Spheron AI introduced their [GPU marketplace](https://www.spheron.ai/), which helps AI startups and enterprises access cost-effective, production-ready GPUs (H100, H200, B200, A100, etc.) at 40–60% lower cost compared to traditional hyperscalers.
   - They offer vendor discovery, pricing negotiation, cluster setup, and scaling so AI teams can focus on building models without worrying about compute.
- **GLM-4.7-Flash Gets Some Fixes**: Members noted that after *llama.cpp* fixed some bugs, **GLM-4.7-Flash** has been updated and reuploaded, requiring users to redownload and follow the parameters on [Z.ai's model card](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF).
   - Outputs should now be much better.
- **Voltage Boasts Cheap Inference**: A member shared an [article](https://x.com/i/article/2012300134575481300) on slashing AI inference costs by 85% with VoltageGPU’s serverless API, which is **OpenAI-compatible** and offers 144+ models.
   - It supports models like **DeepSeek-R1/V3**, **Qwen3**, **Llama-3.3**, and **FLUX**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1463276669589782800)** (14 messages🔥): 

> `Coderrr, detLLM, Lunara aesthetic dataset, McpSwitcher, microd_v1` 


- **Code like Claude with Coderrr!**: Akash built [Coderrr](https://coderrr.aksn.lol/), a free and open-source alternative to Claude Code, and is seeking feedback and contributions on [GitHub](https://github.com/Akash-nath29/Coderrr).
- **Debugging LLM variance with detLLM**: A member developed [detLLM](https://github.com/tommasocerruti/detllm), a toolkit to check LLM inference repeatability and emit minimal repro packs, aiming to become a variance debugger.
- **Moonworks Open-Sources Lunara Aesthetic Dataset**: Moonworks open-sourced a [aesthetic dataset](https://huggingface.co/datasets/moonworks/lunara-aesthetic) created using the Lunara diffusion mixture architecture, paired with human annotations, and published a paper available on [arxiv](https://arxiv.org/abs/2601.07941).
- **Switch MCP Servers Easily with McpSwitcher**: A member shared [McpSwitcher](https://github.com/bivex/McpSwitcher), a tool for easily switching MCP servers and searching skills from the macOS tray.
- **WebXOS Unveils Micro Distilled GRPO VAE Model**: WebXOS released [microd_v1](https://huggingface.co/webxos/microd_v1), a 42M parameter distilled language model trained using Group Relative Policy Optimization (GRPO) with VAE filtering.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1463297778234036245)** (5 messages): 

> `Channel Merge Announcement, Course Assignment Confusion, Robotics Course Update` 


- **Channel gets MegaMerged!**: A member confirmed that the channels have merged into a single channel, and this is that channel.
   - Another member noted that this merge should be reflected in the course details, specifically in the onboarding section.
- **Student Seeks Course Clarity**: A student, returning after three months, expressed confusion about their course assignment.
   - The student attached an image related to the robotics course, asking when the new chapter would come.
- **Robotics Roadmap Remains Riddled**: A student inquired about the release date for the new robotics course chapter.
   - No response given.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1463286126943928410)** (54 messages🔥): 

> `Runpod's success, Greg Yang health, Lightning AI Voltage Park merger, OpenAI Codex Channel, Inference Costs` 


- **Runpod rockets to $120M ARR**: AI cloud startup **Runpod** reached **$120M ARR** four years after launching from a Reddit post, discussed in a [TechCrunch article](https://techcrunch.com/2026/01/16/ai-cloud-startup-runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post/) and on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1qib2ks/runpod_hits_120m_arr_four_years_after_launching/).
- **Greg Yang Steps Back From xAI**: **Greg Yang** is transitioning to an informal advisory role at **xAI** to focus on his health after being diagnosed with **Lyme disease** as he describes symptoms of chronic fatigue and immune issues triggered by exhaustion in [this post](https://xcancel.com/TheGregYang/status/2013652609455006006).
- **Lightning AI and Voltage Park Merge**: **Lightning AI** and **Voltage Park** have merged, with [William Falcon](https://lightning.ai/blog/lightning-ai-voltage-park-merger-ai-cloud), CEO of Lightning AI, and Ozan Kaya, formerly CEO of Voltage Park, leading the merged entity.
   - Some speculate this is a low-key acquisition of a bigger company and wondered if it's a **Runpod** competitor.
- **OpenAI Launches Codex Discord Channel**: **Vaibhav Srivastav** announced the opening of a dedicated **Codex community channel** on the **OpenAI Discord server**, inviting users to share projects and feedback, according to [this post](https://xcancel.com/reach_vb/status/2014053735333290014).
- **Kilo.ai's inference costs breakdown**: A blog post from **Kilo AI** breaks down the **inference costs**, noting that the *Grok code free ride is really over*, accessible [here](https://blog.kilo.ai/p/grok-code-free-ride-is-really-over).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1463307779405254883)** (16 messages🔥): 

> `Audio-to-Video Generation, 3D Environment Generation, AI in Adult Content, Automated Video Editing` 


- ****Lip-Syncing** gets so good!**: [LTX Studio](https://x.com/LTXStudio/status/2013650214171877852) and [ElevenLabs](https://elevenlabs.io/) partner for new **Audio-to-Video generation** feature.
   - This tool allows users to generate AI video starting from audio tracks, ensuring synchronized character voices and actions.
- ****World Labs** Launches 3D Generation API**: **World Labs** has announced the public release of the [World API](https://xcancel.com/theworldlabs/status/2014046372639408203), enabling users to generate explorable **3D environments** from text, images, or video inputs.
   - The API allows for direct product integration into these generated worlds.
- **AI competition heats up the **Adult Content Industry****: A [tweet](https://xcancel.com/abrilzucchi/status/2014027740614123863?s=46) sparked discussion on how human **OnlyFans creators** will need to adapt to compete with the rise of AI-generated personas.
   - The discussion centers on the increasing prevalence and realism of AI-generated adult content.
- ****Kate Deyneka**'s Agentic Video Editor takes center stage**: Kate Deyneka is developing an [automated video editing application](https://xcancel.com/katedeyneka/status/2014091842044747864?s=46) that combines concepts from **Remotion and AI agents**.
   - The app automatically generates scripts and edits media based on uploaded photos and videos, eliminating manual prompting.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1463279753619443713)** (26 messages🔥): 

> `LLM as judge, Pangram paper, nvshmem and pytorch, adversarial model` 


- **LLMs Judge Agent Output**: After a team makes a change to the code or prompt, they manually evaluate how the agent responds; they want to automate the evaluation to avoid manual costs by using *"**LLM** as judge" workflows*. 
   - A member states that the most important part is *looking at the agent outputs and other data yourself before building a workflow that attempts to automate it*.
- **Pangram Paper Replicated**: A member asked if anyone has tried to replicate the **Pangram** paper, accessible [here](https://www.pangram.com/research/papers).
   - A member said someone they know *ran it in his own tests across thousands of different essays and was surprised at how accurate it was, but again it seems to be biased a bit towards playing things safe*.
- **Attacking AI Text Classifiers**: A member linked to a blog post, [Practical Attacks on AI Text Classifiers](https://trentmkelly.substack.com/p/practical-attacks-on-ai-text-classifiers), about an attack on AI text classifiers.
   - Another member linked to a [YouTube video](https://youtu.be/Cs1MI9hjBhs) of someone who *created an adversarial model that did really well*.
- **More Adversarial Model Info**: A member shared a [Youtube link](https://youtu.be/XQcneqUNrN0?feature=shared) to a talk about their server, for those who missed it.
   - The speaker said that if anyone is using **PyTorch**, they should try to configure it to use **nvshmem** for the communication layer.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1463423145544388725)** (17 messages🔥): 

> `Silu Attention Gate, Image Models, Gated Attention Papers, LLM Research Journals` 


- **Silu Gate Falls Flat in Image Models**: A member found that the **silu attention gate** performs no better than the **linear gate** in an image model, possibly due to issues with attention sinks.
   - The conclusion was that this might be specific to **image models**.
- **Gated Attention Papers Explored**: A member recalled that a gated attention paper used **sigmoid** and suggested running a sigmoid test.
   - They found that **silu** performs slightly better and **sigmoid** performs slightly worse, but it's within noise.
- **Top Tier LLM Research Journals Highlighted**: A member listed top-tier LLM research journals including **ICML**, **ICLR**, **NeurIPS**, **TMLR**, **ECCV**, **ICCV**, **CVPR**, and **COLM**.
   - They suggested that **ACL / NAACL / EMNLP** are tier 2 NLP journals due to the popularity of LLMs.
- **Scaling Activations Fixes Large Learning Rates**: A member suggested that the gated attention mechanism handles larger learning rates because it fixes the scale of their activations.
   - A link to an [irrelevant paper](https://arxiv.org/abs/2601.10825) was also shared in the channel.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1463367615090135285)** (2 messages): 

> `Neuronpedia Demo, Llama 3.3` 


- **Neuronpedia has a Llama 3.3 Demo**: A member shared a [Neuronpedia demo](https://www.neuronpedia.org/llama3.3-70b-it/assistant-axis) to play with.
   - The member joked that the demo is *from the stone age in AI timelines* because it is from October.
- **Future AI Paper Already Published!**: A member shared a link to [arxiv.org/abs/2510.01048](https://arxiv.org/abs/2510.01048), a paper from the future.
   - The member commented that it *looks nice*.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1463612710058066194)** (2 messages): 

> `nvshmem, communication backend, nccl, pytorch` 


- **nvshmem Integration Questioned**: A member inquired about integrating **nvshmem** as a communication backend in training.
   - Another member responded that this integration would be a feature below the software stack of **gpt-neox**, residing instead in **nccl** or **pytorch**.
- **Deep dive required for nvshmem**: The user needs to do a deep dive in order to understand where this technology is located.
   - It is most likely below the level of the gpt-neox software stack, and somewhere in nccl or pytorch.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1463332224836567092)** (25 messages🔥): 

> `Chain of Thought Reasoning, Graph-Based Reasoning and Neural Architectures, VoidEditor and llama.cpp, BDH Interpretability and Performance, Biological Plausibility in AI` 


- **Chain of Thought good reasoning focus**: A member highlighted the focus on **chain of thought reasoning** in a [paper](https://arxiv.org/abs/2509.26507) and appreciated the references provided.
- **Graph Reasoning meets Neural Architecture**: A member mentioned their past work on mixing **graph-based reasoning** with **neural architectures** in 2018-2019, noting the difficulty in achieving GPU acceleration.
   - They referenced **Domingos's book** on the topic and pointed out the unpredictability of such models, even with human-understandable aspects.
- **VoidEditor gets Llama support**: A member shared their experience using **VoidEditor** with **llama.cpp's llama-server**, noting its effectiveness but also the difficulty in setting it up.
   - They recommended using **Qwen3 coder instruct 30B** and highlighted the importance of context length/size for agentic coding, requiring significant VRAM.
- **BDH new nerd-snipey LLM**: A member skimmed a [paper](https://arxiv.org/abs/2509.26507) on a new Large Language Model architecture (**BDH**) based on a scale-free biologically inspired network.
   - They noted that it *doesn't seem to be really beating transformers* on its benchmarks, but they are interested in the claims around **BDH's interpretability** and **monosemanticity**.
- **Bio plausibility is not an advantage**: One member expressed the belief that **biological plausibility** is not an advantage but rather an unhelpful constraint in AI.
   - Another member countered that it could be an advantage for efficiency, given the *sheer difference in energy scales* between the brain and current AI.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1463314093619286129)** (10 messages🔥): 

> `Overworld AI, Emergent Mind, Gaussian Splatting` 


- **Overworld AI Links to X**: A member linked to [Overworld AI](https://x.com/overworld_ai/status/2013673088748245188) on X.
- **Emergent Mind Presentation**: A member attended [Emergent Mind's launch presentation](https://www.emergentmind.com/) and found it cool, but noted they're *remaking something that is known to be good but worse*.
- **Gaussian Splatting's weird aesthetics**: A member mentioned that when reading about **Gaussian splatting**, they were expecting a weird aesthetic caused by too few data points or rotating the camera to impossible-to-capture angles.
- **eigent.ai**: A member linked to [eigent.ai](https://www.eigent.ai/)
- **eekCQQYwlgA**: A member linked to [a youtube video](https://youtu.be/eekCQQYwlgA)


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1463263751028604938)** (29 messages🔥): 

> `GPU kernel, Distro AGI, Luminal Kernelbench, Loihi 2, VibeVoice ASR model` 


- **Kernel Talk is Key to Kernel Engineering**: A member shared a link on Discord to a discussion about **GPU kernel** stuff and **Luminal Kernelbench v3** at the [Nous Research Forum](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho).
- **Intel's Loihi 2 Sparks Brain-Like AI Interest**: A member expressed interest in **Intel's Loihi 2**, noting its brain-like architecture and efficiency gains in **matmul** experiments, with higher throughput and lower energy consumption.
- **VibeVoice Model Pulled After Safety Checks**: A member mentioned that **Microsoft's VibeVoice-ASR model** was released, then pulled for failing safety checks, then shared a link to a [shortfuseresearch.com article](https://shortfuseresearch.com/the-genie-is-out-microsofts-vibevoice-and-the-perils-of-open-source-ai/).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1463263650562314286)** (1 messages): 

> `Luminal KernelBench v3, LLM-driven kernel engineering` 


- **Kernel Compiler Luminal Benchmarks for LLMs**: A user inquired whether a kernel compiler like **Luminal KernelBench v3** could enable **LLM-driven SOTA kernel engineering** and posted a link to the [Nous Research Forum](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho).
- **Follow-up discussion on LLM Kernel Engineering**: There was no further discussion about **LLM-driven kernel engineering**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1463264663092330549)** (23 messages🔥): 

> `Manus bugs, Google Project Mariner, Agentic AI vs Manus, Manus performance degradation, Manus billing issues` 


- **Pro user laments recent Manus bugs and unreliability**: A user reported using **Manus** to build and train a text and vector database reasoning model, but noted a recent decline in performance and stability, with only **20 out of 38 modules** working.
   - The user expressed a desire for **CLI access** to fix bugs and reconfigure the system, even as a paid feature.
- **Project Mariner is tempting**: A user asked if anyone had tried **Google's Project Mariner**, expressing interest in testing it with *play money* before committing to a **$150 monthly subscription**.
   - They then mentioned having a **5% off promo**.
- **Agentic AI makes people wet**: A user humorously stated that **Agentic AI** excites them and is a potential competitor to **Manus**, particularly with the integration of **agent mode in Gemini**.
   - The user also expressed a wish for mobile support in Agentic AI.
- **Manus 1.6 performance degrades since Meta**: A user reported a reduction in **Manus 1.6's performance** in recent weeks, possibly since **Meta** released new models, noting difficulties in implementing suggestions for website development despite correct summaries.
   - They mentioned needing to switch to **Manus 1.6 Max** for correct implementations.
- **User reports billing issue with Manus upgrade**: A user reported being charged **$42** for a **Manus** upgrade without receiving the promised **8000 credits**.
   - The user also complained about unhelpful support and a long wait for email assistance.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1463463475127914532)** (17 messages🔥): 

> `Anthropic's programming challenge, VLIW architecture, Metal bindings for texture2d, texture2d vs straight buffers` 


- **Tackle Anthropic's Programming Challenge with Tinygrad**: Tinygrad can be used to solve [Anthropic's original performance takehome](https://github.com/anthropics/original_performance_takehome) programming challenge by writing the target problem in Tensor and adding a tinygrad backend for their toy **VLIW machine**.
   - The challenge addresses core **scheduling problems**, and a bug fix with `PCONTIG=2` allows scheduling in one kernel. Using `return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))` matches their `VLEN`, and `DEVECTORIZE=2` keeps instructions as vector instructions.
- **VLIW is suboptimal for DRAM**: While working on a warp specialized kernel for **RDNA3 matmul**, a member suggests **VLIW isn't ideal for DRAM**, advocating for separate cores and queues (Tenstorrent-style).
   - VLIW is better suited for **SRAM/ALU** due to its static scheduling capabilities.
- **Metal bindings consider texture2d**: A member proposed adding `texture2d` to Metal bindings (`ops_metal.py` + `tinygrad/runtime/graph/metal.py`) for **potential performance improvements** in image-heavy operations like `conv2d` due to optimized texture sampling units, despite `tinygrad`'s philosophy of platform independence.
   - It was noted that while there is `Texture` in image dtype being factored out, `texture2d` could offer a free performance boost on Metal, similar to how `read_imagef` is used in the OpenCL renderer (`tinygrad/renderer/cstyle.py`).
- **Texture2D Speed Boost**: `texture2d` can lead to faster access times and processing speeds for images in GPU memory, citing **high read bandwidth, optimized caches for 2D locality, and texture sampling units over ALU access**.
   - Empirical results showed a **2%-10% speedup** using `texture2d` versus straight buffers, which could be further improved, though concerns were raised about the slippery slope of adding specialized support for other data types like `depth2d`.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1463285654527017063)** (3 messages): 

> `VIZ=1, kernel graphs, scheduler` 


- **VIZ=1 views graphs**: One member inquired if there was a way for **VIZ=1** to view graphs of kernels the same way it views uop graphs.
   - Another member responded, *"yea, click on the schedule and 'view kernel graph'"*.
- **Kernel Graph Viewing**: A user asked about visualizing kernel dependencies using **VIZ=1**, similar to how uop graphs are displayed, to understand the scheduler's operation.
   - A more experienced user advised them to click on the schedule and select *'view kernel graph'* within the **VIZ=1** interface to achieve this visualization.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1463303237200379986)** (14 messages🔥): 

> `GPU Puzzles for Learning Mojo, Apple Silicon GPU Support Status, Reverse Engineering on Apple GPUs, Status of Coroutines (Yield) in Mojo` 


- ****Puzzles** give **GPU** clues**: Newcomers interested in learning Mojo can use the [GPU puzzles](https://puzzles.modular.com/) depending on their skill level.
   - There is also a [forum](https://forum.modular.com/) available for asking questions.
- ****Mojo** reverse engineers **Apple**'s **GPU** secrets**: Modular is reverse engineering much of Apple's GPU due to a lack of documentation, slowing things down, but some puzzles are now working.
   - A member shared a [GPU support matrix](https://puzzles.modular.com/howto.html#gpu-support-matrix) that may not be up to date.
- ****NVIDIA** and **PyTorch** cause hiccups for mojo**: The only puzzles that don't work are **NVIDIA**-specific or using **PyTorch** interop.
   - Because of how PyTorch handles Metal devices very differently from CUDA / ROCm, interfaces need to be extended to account for that.
- ****Yield** no results for those hoping for Coroutines**: `Yield` does not exist and the coroutines that do exist aren’t really usable outside of the compiler runtime since there aren’t really async things exposed to await.
   - One member would *love* to make a recursive algorithm faster in mojo using `yield`, but will need another strategy.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1463264397211340953)** (5 messages): 

> `Error Handling, Python Import Syntax, Dynamic Python Imports` 


- **Functions Raise Errors Upwards**: Members discussed that functions can be designed to **raise errors**, effectively passing error handling responsibilities to higher-level functions.
   - One member expressed the tedium of writing *try/except* blocks in every function, especially when dealing with potential import errors.
- **"try Python.import_module" for Result Type**: A member suggested a future syntax like *try Python.import_module('numpy')* that would return a **Result** type to streamline error handling during module imports.
   - This would provide a cleaner alternative to using *try/except* blocks for every potential import failure.
- **Main Function Imports as Alternative to Inline try/except**: A member suggested importing modules once in the **main function** and passing the **handle** around to avoid repeated *try/except* blocks.
   - It was acknowledged that due to **dynamic Python imports**, files could be missing on any given import, necessitating some form of error handling.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1463281239204036692)** (11 messages🔥): 

> `aider features, ChatGPT Business Account setup, OpenCode vs Aider, Aider Future, Aider alternatives` 


- **Aider Feature Wishlist**: A user inquired about desired features in **aider** beyond *agentic stuff* like **MCP** and **tool calls**.
   - There were no responses, no desired features were identified.
- **ChatGPT Business Account Setup with Aider?**: A user asked if their **ChatGPT Business account**, which lacks an **API key** but has access to **Codex LLMs**, could be used with **aider**.
   - A member pointed to [aider documentation](https://aider.chat/docs/llms/other.html) and [LiteLLM documentation](https://docs.litellm.ai/docs/providers/chatgpt), indicating potential support via **LiteLLM**.
- **Aider's Demise?**: A user expressed fear that **aider** would be replaced by **OpenCode**.
   - A member responded that if **Paul Gauthier** has moved onto other projects and is not using **Aider** he won't be returning to it, but another user responded that they continue to use it with **GPT 5.2**.
- **Aider as a Dead Project?**: A user speculated that **Aider** is a dead project, and cited the rise of alternative tools like **Open Code**, **KiloCode CLI**, **Claude Code**, and **Gemini CLI**.
   - They also mentioned the **Aider-CE project**, which is bolting on agentic functionality to modernize the architecture.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1463288202411507793)** (10 messages🔥): 

> `MCP Inspector re-auth on 401, MCP Multi Server Client statefulness, MCP client server ranking` 


- ****MCP Inspector** fails to re-auth on 401**: A member inquired why the **MCP Inspector** doesn't re-authenticate upon encountering a **401 error**, either during connection or tool calls, and another member acknowledged an [SDK issue with persisting resourceMetadata across redirects](https://github.com/modelcontextprotocol/inspector/issues/576#issuecomment-3766294454), which is being addressed.
   - They mentioned that **VS Code** only uses it on initial connections and not on 401s after that, potentially related to SDK internals.
- **Quest for Stateful **MCP Multi Server Client****: A member asked if anyone has used the **MCP Multi Server Client** to maintain statefulness of their session and whether they faced any issues.
   - No solutions or workarounds were offered in the thread, but it was marked as a candidate topic.
- **Ranking Servers for **MCP Clients**: a Protocol Question?**: One member inquired how **MCP clients** handle server recommendations when an agent needs to manage a calendar, questioning whether custom algorithms are used or if there's work on shared standards.
   - Another member responded that *ranking* was considered during the work on [Feature Discovery Protocol](https://discord.com/channels/1199128884040474664/1369487942862504016) but deemed out of scope for the protocol, suggesting it's up to the ecosystem to decide on a per-client basis.

