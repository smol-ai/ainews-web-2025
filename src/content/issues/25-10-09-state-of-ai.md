---
id: MjAyNS0x
title: Air Street's State of AI 2025 Report
date: '2025-10-09T05:44:39.731046Z'
description: >-
  **Reflection** raised **$2B** to build frontier open-weight models with a
  focus on safety and evaluation, led by a team with backgrounds from
  **AlphaGo**, **PaLM**, and **Gemini**. **Figure** launched its next-gen
  humanoid robot, **Figure 03**, emphasizing non-teleoperated capabilities for
  home and large-scale use. **Radical Numerics** released **RND1**, a
  **30B-parameter sparse MoE diffusion language model** with open weights and
  code to advance diffusion LM research. **Zhipu** posted strong results with
  **GLM-4.6** on the Design Arena benchmark, while **AI21 Labs**' **Jamba
  Reasoning 3B** leads tiny reasoning models. **Anthropic** introduced a plugin
  system for **Claude Code** to enhance developer tools and agent stacks. The
  report also highlights SoftBank's acquisition of ABB's robotics unit for
  **$5.4B** and the growing ecosystem around open frontier modeling and
  small-model reasoning.
companies:
  - reflection
  - mastra
  - datacurve
  - spellbook
  - kernel
  - figure
  - softbank
  - abb
  - radicalnumerics
  - zhipu-ai
  - ai21-labs
  - anthropic
models:
  - glm-4.6
  - jamba-1.5
  - rnd1
  - claude-code
topics:
  - humanoid-robots
  - mixture-of-experts
  - diffusion-models
  - open-weight-models
  - reinforcement-learning
  - benchmarking
  - small-language-models
  - plugin-systems
  - developer-tools
  - agent-stacks
people:
  - adcock_brett
  - achowdhery
  - clementdelangue
---


**300 slides are all you need.**

> AI News for 10/8/2025-10/9/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (197 channels, and 7870 messages) for you. Estimated reading time saved (at 200wpm): 583 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Congrats to [Reflection](https://x.com/reflection_ai/status/1976304405369520242?s=46), [Mastra](https://mastra.ai/blog/seed-round), [Datacurve](https://x.com/serenaa_ge/status/1976328983458480539), [Spellbook](https://x.com/scottastevenson/status/1976280608436572393), and [Kernel](https://x.com/juecd__/status/1976325764166615498?s=46) on their fundraises.

The AI-native equivalent of the [annual Mary Meeker report](https://news.smol.ai/issues/25-05-30-mary-meeker) has been [Nathan Benaich's State of AI report](https://www.stateof.ai/). You can catch the highlight in [tweet thread](https://x.com/nathanbenaich/status/1976159936498598271) or [youtube](https://www.youtube.com/watch?v=Ub-7bY4b3Hs), but we'll offer some notes here:

The labs with the mandate of heaven (where does Anthropic show up?):

![](https://resend-attachments.s3.amazonaws.com/cJuxsfweuSJyrx3)

and their valuations:

![](https://resend-attachments.s3.amazonaws.com/NFOKJh0ErWs36c2)

AI-first "tiny teams":

![](https://resend-attachments.s3.amazonaws.com/Is9uIlen53Gav5Q)

The 2026 cluster buildouts:

![](https://resend-attachments.s3.amazonaws.com/pbjWkmSuvoL6TjQ)

---

# AI Twitter Recap

**Humanoid Robotics: Figure 03 launch, capabilities, and industry moves**

- Introducing Figure 03: Figure unveiled its next-gen humanoid with a highly produced demo and a detailed write-up on system design and product goals. The team emphasizes “nothing in this film is teleoperated,” positioning F.03 for “Helix, for the home, and for the world at scale.” See launch and follow-ups from [@Figure_robot](https://twitter.com/Figure_robot/status/1976272678618308864), [@adcock_brett](https://twitter.com/adcock_brett/status/1976272831450341655), and write-up links from [@adcock_brett](https://twitter.com/adcock_brett/status/1976272961226277181). For broader robotics context: SoftBank is acquiring ABB’s robotics unit for $5.4B per [The Rundown](https://twitter.com/TheRundownAI/status/1976301682863603819).
- Discussion: Early reviews note some demo quirks (e.g., sorting choices), but overall the capability trajectory and non-teleop claim drew strong interest from practitioners; see reactions from [@Teknium1](https://twitter.com/Teknium1/status/1976342200578703660).

**Open frontier modeling and releases: Reflection’s $2B, Diffusion LMs, GLM-4.6, and small-model reasoning**

- Reflection raises $2B to build frontier open-weight models: The lab is scaling large MoE pretraining and RL from scratch with an explicit open-intelligence roadmap (safety and evals emphasized). Founder and team context (AlphaGo, PaLM, Gemini contributors) and hiring across SF/NY/London. Read the statement from [@reflection_ai](https://twitter.com/reflection_ai/status/1976304405369520242) and commentary by [@achowdhery](https://twitter.com/achowdhery/status/1976314051102982285) and [@ClementDelangue](https://twitter.com/ClementDelangue/status/1976315464788934960).
- Diffusion Language Models go bigger (open): Radical Numerics released RND1, a 30B-parameter sparse MoE DLM (3B active), with weights, code, and training details to catalyze research into DLM inference/post-training and a simple AR-to-diffusion conversion pipeline. See the announcement and resources via [@RadicalNumerics](https://twitter.com/RadicalNumerics/status/1976332725926936599) and a concise summary thread by [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1976405147917152283).
- Zhipu’s GLM-4.6 and open models momentum: Zhipu’s GLM-4.6 posts strong results on the Design Arena benchmark per [@Zai_org](https://twitter.com/Zai_org/status/1976226981176807870). Cline notes GLM-4.5-Air and Qwen3-Coder are the most popular local models in their agent IDE ([tweet](https://twitter.com/cline/status/1976101061753700400)).
- Tiny reasoning at the edge: AI21’s Jamba Reasoning 3B leads “tiny” reasoning models with 52% on IFBench per [@AI21Labs](https://twitter.com/AI21Labs/status/1976271434004541641). Related, Alibaba’s Qwen continues to push breadth: Qwen3-Omni (native end-to-end multimodal) and Qwen-Image-Edit 2509 now ranked #3 overall, leading open-weight models ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1976267690785505440), [tweet](https://twitter.com/Alibaba_Qwen/status/1976119224339955803)).

**Developer tools and agent stacks: Claude Code plugins, VS Code AI, Gemini ecosystem**

- Claude Code opens up plugins: Anthropic shipped a plugin system and marketplace for Claude Code. Update your CLI and add via “/plugin marketplace add anthropics/claude-code.” Early community marketplaces emerging. See threads from [@The_Whole_Daisy](https://twitter.com/The_Whole_Daisy/status/1976332882378641737) and [@_catwu](https://twitter.com/_catwu/status/1976334583445717451).
- VS Code v1.105 September release: AI-first UX improvements include GitHub MCP registry integration, AI merge-conflict resolution, OS notifications, and chain-of-thought rendering with GPT-5-Codex. Details and livestream via [@code](https://twitter.com/code/status/1976332459886182627).
- Google’s Gemini platform updates: New “model search” in AI Studio ([@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1976322693726343384)), hosted docs for the Gemini CLI ([@_philschmid](https://twitter.com/_philschmid/status/1976178993452974416)), and “Gemini Enterprise” as a no-code front door to build agents and automate workflows across Workspace/M365/Salesforce and more ([@Google](https://twitter.com/Google/status/1976345752541536525), [@JeffDean](https://twitter.com/JeffDean/status/1976313985453732222)).
- Memory and eval-driven optimization in agent pipelines: Developers test memory layers like Mem0 ([@helloiamleonie](https://twitter.com/helloiamleonie/status/1976270045534679106)) and use DSPy/GEPA to switch models at 20x lower cost without regressions ([@JacksonAtkinsX](https://twitter.com/JacksonAtkinsX/status/1976248661081501766)); see also DSPy TS usage demo ([@ryancarson](https://twitter.com/ryancarson/status/1976376939343491260)).

**Benchmarks and evaluations: ARC-AGI, METR time-horizons, FrontierMath, and domain leaderboards**

- GPT-5 Pro posts new SOTA on ARC-AGI: Verified by ARC Prize, GPT-5 Pro achieved 70.2% on ARC-AGI-1 ($4.78/task) and 18.3% on ARC-AGI-2 ($7.41/task), the highest frontier LLM score on the semi-private benchmark to date ([@arcprize](https://twitter.com/arcprize/status/1976329182893441209)).
- Time-horizon on agentic SWE tasks: METR estimates Claude Sonnet 4.5’s 50%-time-horizon at ~1 hr 53 min (CI 50–235 min), a statistically significant improvement over Sonnet 4 but below Opus 4.1 point estimates; see [@METR_Evals](https://twitter.com/METR_Evals/status/1976331315772580274).
- Math and reasoning evaluations: Epoch reports Gemini 2.5 “Deep Think” set a new record on FrontierMath (manual API evaluation due to lack of public API), with broader math capability analysis in thread ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1976340039178305924)). ARC-AGI numbers prompted debate on recent progress pacing vs. trendlines (see [@scaling01](https://twitter.com/scaling01/status/1976336799967723782), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1976389736160952802)).
- Vision/editing and design tasks: Qwen Image Edit 2509 ranks #3 overall, leading open-weight models ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1976119224339955803)). GLM-4.6 shows strong performance on Design Arena ([@Zai_org](https://twitter.com/Zai_org/status/1976226981176807870)).

**Systems, performance, and infra: GPU kernels, inference benchmarking, and MLX speed**

- GPU kernels and “register tiles”: tinygrad is porting ThunderKittens’ “register tile” abstraction (“registers are the wrong primitive”) as “tinykittens,” citing simpler yet performant GPU code ([**tinygrad**](https://twitter.com/__tinygrad__/status/1976084605141909845)). Awni Hannun dropped a concise MLX matmul primer to illuminate tensor core fundamentals ([tweet](https://twitter.com/awnihannun/status/1976347648014811634)).
- Real-world inference benchmarking at scale: SemiAnalysis launched InferenceMAX, a daily cross-stack benchmark suite spanning H100/H200/B200/GB200/MI300X/MI325X/MI355X (soon TPUs/Trainium), focused on throughput, cost per million tokens, latency/throughput tradeoffs, and tokens per MW across modern servers and inference stacks ([@dylan522p](https://twitter.com/dylan522p/status/1976422855928680454)).
- On-device and Apple silicon: Qwen3-30B-A3B 4-bit hits 473 tok/s on M3 Ultra via MLX ([@ivanfioravanti](https://twitter.com/ivanfioravanti/status/1976153645658898453)). Google released a Gemma 3 270M fine-tune-to-deploy flow that compresses to <300MB and runs in-browser/on-device ([@googleaidevs](https://twitter.com/googleaidevs/status/1976315582094917787); tutorial by [@osanseviero](https://twitter.com/osanseviero/status/1976330544263966869)).

**Multimodal/video: Sora 2 momentum, Genie 3 recognition, and WAN 2.2**

- Sora 2 growth + free HF demo: Sora 2 hit 1M app downloads in under 5 days (despite invites and NA-only) with rapid iteration on features and moderation ([@billpeeb](https://twitter.com/billpeeb/status/1976099194407616641)). A limited-time Sora 2 text-to-video demo is live on Hugging Face and getting used in the wild ([tweet](https://twitter.com/_akhaliq/status/1976096764781646028)). The cameo use-case exploded, with notable NIL-driven virality ([@jakepaul](https://twitter.com/jakepaul/status/1976411343025487977)).
- Genie 3 named a TIME Best Invention: Google DeepMind’s interactive world model continues to draw attention for generating playable environments from text/image prompts ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1976311787013480758), [@demishassabis](https://twitter.com/demishassabis/status/1976403370224337220)).
- WAN 2.2 Animate tips and workflows: Community tutorials show improved lighting/flame behavior and practical pipelines for animation tasks ([@heyglif](https://twitter.com/heyglif/status/1976259706214592747), [@jon_durbin](https://twitter.com/jon_durbin/status/1976253117265326540)).

**Safety, bias, and security**

- Few-shot poisoning may suffice: Anthropic, with UK AISI and the Turing Institute, shows that a small, fixed number of malicious documents can implant backdoors across model sizes—challenging prior assumptions that poisoning requires a sizable dataset fraction. Read the summary and paper from [@AnthropicAI](https://twitter.com/AnthropicAI/status/1976323781938626905).
- Political bias definitions and evaluation: OpenAI researchers propose a framework to define, measure, and mitigate political bias in LLMs ([@nataliestaud](https://twitter.com/nataliestaud/status/1976382637104300329)).

**Top tweets (by engagement)**

- Elon shows Grok’s “Imagine” reading text from an image (no prompt) — a virality juggernaut this cycle: [@elonmusk](https://twitter.com/elonmusk/status/1976146944398590385)
- Figure 03 humanoid launch (non-teleop claim, multiple clips): [@Figure_robot](https://twitter.com/Figure_robot/status/1976272678618308864), [@adcock_brett](https://twitter.com/adcock_brett/status/1976272831450341655)
- “POV: Your LLM agent is dividing a by b” — debugging agents, the meme we deserved: [@karpathy](https://twitter.com/karpathy/status/1976082963382272334)
- “I prefer not to speak.” — the quote that took over everyone’s timeline: [@UTDTrey](https://twitter.com/UTDTrey/status/1976237786408837261)
- Genie 3 named a TIME Best Invention: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1976311787013480758)
- ARC-AGI new SOTA with GPT-5 Pro: [@arcprize](https://twitter.com/arcprize/status/1976329182893441209)

Notes

- Elastic acquired Jina AI to deepen multimodal/multilingual search and context engineering in Elastic’s agentic stack ([tweet](https://twitter.com/elastic/status/1976278980018765886)).
- Gemini crossed 1.057B visits in Sept 2025 (+285% YoY), its first month over 1B visits ([@Similarweb](https://twitter.com/Similarweb/status/1976206499191062758)).
- State of AI 2025 is out; usage, safety, infra, and research trends summarized ([@nathanbenaich](https://twitter.com/nathanbenaich/status/1976159936498598271)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Microsoft UserLM-8B 'User' Role-Simulation Model Announcement

- [**microsoft/UserLM-8b - “Unlike typical LLMs that are trained to play the role of the 'assistant' in conversation, we trained UserLM-8b to simulate the 'user' role”**](https://www.reddit.com/r/LocalLLaMA/comments/1o23vqf/microsoftuserlm8b_unlike_typical_llms_that_are/) (Activity: 548): **Microsoft’s UserLM-8b is an 8B-parameter user-simulator LLM fine-tuned from Llama3‑8b‑Base to predict user turns (from WildChat) rather than act as an assistant; it takes a single task-intent input and emits initial/follow-up user utterances or an <|endconversation|> token ([paper](https://arxiv.org/abs/2510.06552), [HF](https://huggingface.co/microsoft/UserLM-8b)). Training used full-parameter finetuning on a filtered WildChat‑1M with max seq len 2048, batch size** `1024`**, lr** `2e-5`**, on** `4× RTX A6000` **over** `~227 h`**. Evaluations report lower perplexity (distributional alignment), stronger scores on six intrinsic user-simulator metrics, and broader/diverse extrinsic simulation effects versus prompted assistant baselines; the research release warns of risks (role drift, hallucination, English-only testing, inherited biases) and recommends guardrails (token filtering, end-of-dialogue avoidance, length/repetition thresholds).** Commenters highlight the meta trend of AI training/evaluating AI and express safety/availability concerns (possible takedown), with little substantive technical critique in-thread.
    - Several commenters highlight the closed-loop risk of "AI evaluating AI" if **UserLM-8b** is used to simulate users that other models then optimize against. This can induce feedback loops and distribution shift where models overfit to the simulator’s style/tokens, degrading benchmark validity and leading to artifacts like reward hacking, prompt overfitting, and misleading improvements that don’t transfer to real users.
    - There’s concern the release might be pulled for safety reasons, implying reproducibility and availability risk for experiments with **UserLM-8b**. Practically, this means researchers should pin exact checkpoints and versions early to preserve comparability across runs and avoid future benchmark drift if artifacts/weights are taken down or altered.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Qwen Image Edit 2509: Next Scene LoRA and named-entity editing tips

- [**I trained « Next Scene » Lora for Qwen Image Edit 2509**](https://www.reddit.com/r/StableDiffusion/comments/1o237ws/i_trained_next_scene_lora_for_qwen_image_edit_2509/) (Activity: 607): **Author releases an open-source LoRA, "Next Scene," for** `Qwen Image Edit 2509`**, aimed at scene continuity: invoking the trigger** `Next scene:` **yields follow-up frames that preserve character identity, lighting, and environment across edits. Repository and weights are available on Hugging Face: [lovis93/next-scene-qwen-image-lora-2509](https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509), with no usage restrictions; the core UX is to prepend the prompt with “Next scene:” and describe desired changes.** A commenter proposes extending the method to controllable camera re-posing (e.g., specifying current view and target view like “camera right”) to simulate multi-camera continuity from a single still—implying a need for viewpoint-consistent novel view synthesis. Another asks whether a workflow or example pipeline is included.
    - A commenter proposes a view-conditioned LoRA: given a single image and a directive like “camera right,” the model should render the identical scene from a new camera pose (multi-camera shoot simulation). Implementing this would likely require conditioning on explicit pose signals (e.g., tokens mapped to SE(3) transforms or numeric camera extrinsics), multi-view training data, and geometry-aware guidance (depth/normal ControlNet). Key challenges are preserving scene/identity consistency under viewpoint changes and resolving occlusions; related prior art includes single-image novel view methods like [Zero-1-to-3](https://arxiv.org/abs/2303.13495) and [Stable Zero123](https://github.com/ashawkey/stable-zero123).
    - A data-scale question surfaces: “How many pairs of data were used for training?” For instruction-like LoRAs that map prompts (e.g., “Next Scene”) to structured edits in image editors (Qwen Image Edit 2509), generalization typically depends on hundreds-to-thousands of paired before/after examples; too few pairs risks overfitting to narrow styles or compositions. Reporting pair counts, LoRA rank, and training schedule would help others reproduce/benchmark and understand capacity-vs-quality tradeoffs.
    - The LoRA checkpoint is shared on Hugging Face for reproducibility: https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509/tree/main. Technical readers may look for exported safetensors, example prompts, and any training configs or logs to evaluate compatibility with Qwen Image Edit 2509 pipelines and compare against baselines.
- [**TIL you can name the people in your Qwen Edit 2509 images and refer to them by name!**](https://www.reddit.com/r/StableDiffusion/comments/1o1zsny/til_you_can_name_the_people_in_your_qwen_edit/) (Activity: 484): **OP shows that the Qwen Edit image-editing pipeline can bind named entities to separate input reference images (e.g., “Jane is in image1; Forrest is in image2; Bonzo is in image3”) and then control multi-subject composition via natural-language constraints (relative positions, poses, and interactions) while preserving details from a chosen reference ("All other details from image2 remain unchanged"). They share a straightforward ComfyUI workflow JSON that reproduces this behavior, enabling multi-image identity/appearance referencing without extra training or LoRAs ([workflow](https://files.catbox.moe/porrtw.json)).** Commenters note the workflow’s simplicity and express surprise this wasn’t widely tried; a key question is whether success relies on the model’s prior knowledge of known figures ("Forrest") and if it generalizes equally to three unknown subjects.
    - Several commenters question whether the “name binding” works only because the model already knows famous entities (e.g., “Forrest”), versus true arbitrary-identity binding. They propose testing with “three random unknowns” to verify that Qwen Edit 2509 can consistently disambiguate and condition on non-celebrity identities by name, rather than relying on prior knowledge embedded in the model.
    - A key implementation question raised: do you need a separate reference latent per image/person for this workflow to function? This touches on how identity conditioning is represented (per-subject latent/embedding vs shared latent across multiple images), potential VRAM/compute trade-offs, and whether latents can be cached or reused to reduce cost while maintaining consistent name-to-identity mapping across generations.

### 2. AI progress retrospectives: 'Will Smith spaghetti' and '2.5 years' revisited

- [**Will Smith eating spaghetti - 2.5 years later**](https://www.reddit.com/r/ChatGPT/comments/1o22zh9/will_smith_eating_spaghetti_25_years_later/) (Activity: 9007): **Revisits the canonical 2023 "Will Smith eating spaghetti" AI video as an informal regression benchmark for text‑to‑video progress** `~2.5 years` **later, linking a new clip ([v.redd.it/zv4lfnx4j2uf1](https://v.redd.it/zv4lfnx4j2uf1)) that currently returns** `HTTP 403` **(OAuth/auth required). Historically associated with early diffusion T2V (e.g., ModelScope [damo‑vilab/text‑to‑video‑ms‑1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)), the prompt surfaced classic failure modes—identity drift, utensil/food physics artifacts, unstable hand‑mouth interactions, and temporal incoherence—which this revisit implicitly uses to gauge improvements in control and realism. Without access to the clip, no quantitative comparison can be drawn, but the framing suggests better motion control and stability versus 2023 outputs.** Commenters propose keeping this prompt as a de facto standard benchmark; others note the newer output feels more controlled while some prefer the older glitchy rendition as more "authentic," reflecting a polish-versus-aesthetic debate rather than a metrics-based one.
    - Several commenters implicitly treat “Will Smith eating spaghetti” as a de facto regression test for text-to-video, since it stresses hand–mouth coordination, thin deformable strands, utensil occlusion, bite/chew/swallow transitions, fluid/sauce dynamics, and conservation-of-mass—failure modes common in diffusion-based video models. A rigorous setup would fix prompt/seed across model versions and score with `FVD` ([paper](https://arxiv.org/abs/1812.01717)) plus action-recognition consistency (e.g., classifier accuracy for the verb "eating" on [20BN Something-Something](https://20bn.com/datasets/something-something)).
    - A key limitation highlighted: models render plausible exterior motions but lack internal state transitions for ingestion—clips loop "put-to-mouth" without chewing/swallowing or decreasing food volume, signaling missing causal state tracking and object permanence. This aligns with known gaps in 2D video diffusion lacking explicit 3D/volumetric and physics priors; remedies include 3D-consistent video generation and world-model approaches with differentiable physics (see **World Models** [Ha & Schmidhuber, 2018](https://arxiv.org/abs/1803.10122)).
    - Preference for the 2023 output as more "authentic" hints at a trade-off: newer generators may improve photorealism/identity fidelity but over-regularize micro-actions (mode collapse), degrading action semantics like actual consumption. Evaluations should move beyond appearance metrics (FID/`FVD`) to temporal/action faithfulness, e.g., CLIP-based action scoring ([CLIP](https://arxiv.org/abs/2103.00020)), temporal cycle-consistency ([TCC](https://arxiv.org/abs/1904.07846)), or explicit "consumption-event" detectors that verify decreasing food mass over time.
- [**2.5 years of AI progress**](https://www.reddit.com/r/aivideo/comments/1o24s4c/25_years_of_ai_progress/) (Activity: 781): **The post titled “2.5 years of AI progress” links to a video on Reddit ([v.redd.it/qqxhcn4ez2uf1](https://v.redd.it/qqxhcn4ez2uf1)), but the media returns an HTTP** `403` **network-security block, so the underlying content cannot be verified. Based on the title alone, it likely juxtaposes model outputs across ~2.5 years, yet there are no visible benchmarks, model identifiers, prompts, or inference settings to assess methodology or quantify progress from the accessible context.** Top comments split between nostalgia for earlier, more chaotic model behavior and claims of “exponential” improvement, but no quantitative evidence or technical specifics are offered to substantiate either view.

### 3. Figure 03 launch and AI policy/workforce debates (Anthropic limits, EO compliance, Altman)

- [**Introducing Figure 03**](https://www.reddit.com/r/singularity/comments/1o25fx1/introducing_figure_03/) (Activity: 2102): **Reddit post announces “Figure 03,” presumably the next-gen humanoid from Figure. The demo video (blocked to us at the Reddit media link) is claimed to be fully autonomous—i.e., not teleoperated—per Figure CEO Brett Adcock’s confirmation on X ([source](https://x.com/adcock_brett/status/1976272909569323500?s=46)), implying onboard perception, planning, and control rather than remote driving. No benchmarks, system specs, or training details are provided in the thread.** The only substantive debate centers on teleoperation; commenters reference Adcock’s statement to conclude the demo reflects real autonomous capability rather than puppeteering.
    - Several commenters highlight a claim that the Figure 03 demo involved **no teleoperation** (no human-in-the-loop joysticking), interpreting this as evidence of on-board autonomy for perception, planning, and control across the shown tasks. This materially reduces “Wizard-of-Oz” concerns and shifts scrutiny toward what level of scripting or environment priors might still be present. Reference: confirmation link shared in-thread: https://x.com/adcock_brett/status/1976272909569323500?s=46.
    - Technical skepticism centers on whether the demo is *heavily leveraging tricks* (e.g., tight scene staging, pre-programmed trajectories, selective cuts) versus robust generalization. Commenters call for a live, continuous, single-take demonstration in an uninstrumented environment with ad-hoc, audience-specified perturbations to validate reliability and latency, and to rule out hidden external localization or motion capture.
    - Multiple users note a large capability jump from **Figure 02 → Figure 03**, implying broader task coverage and more polished manipulation/mobility behaviors. They suggest the "use cases piling up" merit tracking concrete metrics in future demos (task success rates, recovery behavior, cycle times), to quantify progress beyond curated highlight reels.
- [**Megathread's Response to Anthropic's post "Update on Usage Limits"**](https://www.reddit.com/r/ClaudeAI/comments/1o1wn34/megathreads_response_to_anthropics_post_update_on/) (Activity: 971): **Synthesizing 1,700+ reports from the r/ClaudeAI [Usage Limits Megathread](https://www.reddit.com/r/ClaudeAI/comments/1nu9wew/usage_limits_discussion_megathread_beginning_sep/) in response to Anthropic’s [“Update on Usage Limits”](https://www.reddit.com/r/ClaudeAI/comments/1nvnafs/update_on_usage_limits/): many users hit caps rapidly on Sonnet 4.5 alone (e.g.,** `~10` **messages or** `1–2` **days, sometimes hours), so “use Sonnet instead of Opus 4.1” doesn’t alleviate lockouts. Metering is reported as opaque/inconsistent—small edits can burn** `5–10%` **of a** `5-hour` **session (previously** `2–3%`**), perceived** `~3x` **cost/turn increases, and shifting reset timestamps across the** `5-hour`**, weekly all-model, and Opus-only pools—fueling work-stopping weekly lockouts and churn. Proposed remediations: replace weekly cliffs with daily caps + rollover; publish exact metering math (how uploads, extended thinking, compaction, and artifacts are counted); add model-scoped meters, pre-run cost hints, and “approaching cap” warnings; standardize reset times; sweep metering anomalies; enable paid top-ups and grace windows; and improve Sonnet 4.5 long-context/codebase reliability to avoid forced fallbacks to Opus.** Commenters characterize the change as a *stealth downgrade* driving cancellations/refunds; one Pro user estimates capacity dropped from `~42 h/week` (4×1.5h/day, no weekly cap) to `~10 h/week` (10×~1h sessions). Others assert *“everyone is hitting the weekly limit after the update,”* particularly on Sonnet 4.5.
    - Several **Pro** users quantify the impact of **Sonnet 4.5**’s new weekly cap: with “intense programming” they hit the limit in ~`10` one‑hour sessions per week (~`10` hours total), versus prior usage of `4` daily sessions × `1.5` hours (= `42` hours/week). Practically, this is a ~`76%` reduction in available coding time compared to pre‑update behavior, reframing Pro as a time‑capped product for heavy dev workflows.
    - Multiple reports indicate users are hitting the **Sonnet 4.5** weekly limit quickly after the update, implying the metering is far stricter than earlier daily‑only constraints. If Sonnet 4.5 is metered by compute‑intensive requests, the weekly cap becomes the primary bottleneck for sustained dev sessions, degrading throughput for tasks like code generation and refactoring.
    - A metering anomaly is reported: a single sub‑`20` character prompt and a single‑number reply consumed `2%` of the “5‑hour” rolling limit and `1%` of the weekly limit ([screenshot](https://imgur.com/a/n8mvXjj)). With no tools/web/think mode enabled, this suggests either a metering bug or coarse quota rounding (e.g., per‑request minimum charge or inclusion of hidden system/context tokens) that charges tiny prompts in large increments.
- [**Chat GPT and other AI models are beginning to adjust their output to comply with an executive order limiting what they can and can’t say in order to be eligible for government contracts. They are already starting to apply it to everyone because those contracts are $$$ and they don’t want to risk it.**](https://www.reddit.com/r/ChatGPT/comments/1o1vglc/chat_gpt_and_other_ai_models_are_beginning_to/) (Activity: 803): **OP flags a new Executive Order that conditions federal LLM procurement on adherence to two “Unbiased AI Principles”: Truth‑seeking (prioritize factual accuracy/uncertainty) and Ideological Neutrality (avoid partisan/DEI value judgments unless explicitly prompted/disclosed), with OMB to issue guidance in** `120 days` **and agencies updating procedures** `within 90 days` **thereafter; contracts must include compliance terms and vendor liability for decommissioning on noncompliance, allow limited transparency (e.g., system prompts/specs/evaluations) while protecting sensitive details (e.g., model weights), and include national‑security carve‑outs. The EO is procurement‑scoped (building on [EO 13960](https://www.federalregister.gov/documents/2020/12/08/2020-27065/promoting-the-use-of-trustworthy-artificial-intelligence-in-the-federal-government)), but OP alleges vendors (e.g., ChatGPT) will preemptively enforce “government‑compliant” policies platform‑wide to preserve eligibility; link to EO: [whitehouse.gov](http://whitehouse.gov/).**
    - Several commenters infer that providers may be tightening global safety/policy layers to meet U.S. government procurement requirements, rather than maintaining a separate gov-only policy fork. Technically, this likely manifests as updates to pre- and post-generation filters (prompt classifiers, toxicity/harm heuristics, retrieval/policy guards), system prompts, and RLHF/constitutional reward models that expand refusal criteria for topics like political persuasion, misinformation, or child safety—affecting all users across models like GPT-4/4o, Claude 3.x, and Gemini. Centralizing one policy stack reduces operational risk and cost (fewer model variants, simpler evaluations/red-teaming) but increases the chance of overbroad refusals or distribution shift in helpfulness.
    - There’s clarification that executive orders don’t legislate public speech but can condition federal agency purchases, which indirectly pressures vendors. Relevant artifacts include the U.S. AI EO (Exec. Order 14110) directing NIST/AI Safety Institute standards and OMB procurement/governance guidance (e.g., M-24-10), which can require risk assessments, content harm mitigations, and auditability as contracting terms; see EO 14110 text: https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/, OMB memo: https://www.whitehouse.gov/omb/briefing-room/2024/03/28/omb-releases-first-government-wide-policy-to-mitigate-risks-of-ai/. Practically, vendors may implement stricter universal policies to ensure compliance evidence (evals/red-team reports, incident response, logging) for eligibility.
    - Technical risk highlighted: policy tightening framed as “misinformation/child-safety” mitigation can be over-applied by automated classifiers, yielding false positives and refusals on benign content. This is a known failure mode of stacked safety systems where threshold tuning, distribution drift, and reward hacking can degrade helpfulness; mitigation typically involves calibrated confidence thresholds, context-aware exception lists, multi-signal moderation, and transparent appeal channels, plus periodic A/B evaluation to track refusal-rate and utility regressions.
- [**Sam Altman Says AI will Make Most Jobs Not ‘Real Work’ Soon**](https://www.reddit.com/r/ChatGPT/comments/1o21tqp/sam_altman_says_ai_will_make_most_jobs_not_real/) (Activity: 671): **At OpenAI DevDay 2025, Sam Altman argued AI will redefine “real work,” forecasting that up to** `~40%` **of current economic tasks could be automated in the near term and that code-generation agents (e.g., [OpenAI Codex](https://openai.com/blog/openai-codex)) are approaching the ability to autonomously deliver previously “week‑long” programming tasks. He contrasted modern office work with historical manual labor to frame a shift in knowledge-work, and recommended prioritizing adaptive learning, understanding human needs, and interpersonal care—domains he believes remain comparatively resilient—while noting short‑term transition risks but long‑run opportunities.**

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Kernel and Attention Performance Engineering**

- **Helion Hacks Kernels to Topple Triton**: **Helion** autotunes by rewriting the kernel itself to fit input shapes (e.g., loop reductions for large shapes) yet ultimately emits a Triton kernel, and community benchmarks indicate it often outperforms **Triton** across diverse shapes; code lives in [flash-linear-attention/ops](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops).
    - Members proposed head-to-heads against [TileLang Benchmarks](https://github.com/tile-ai/tilelang-benchmarks) and flagged interest in weirder attention variants teased for PTC, while noting that heavy shape specialization can yield sizable wins for **linear attention** kernels.
- **PTX Docs Trip on K‑Contig Swizzles**: Engineers reported inaccuracies in NVIDIA’s PTX docs for **K‑contiguous swizzle layouts** in the section on asynchronous warpgroup-level canonical layouts, cross-referencing Triton’s implementation to show mismatches ([PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-canonical-layouts), [Triton MMAHelpers.h](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAHelpers.h#L250-L253)).
    - The clarification helps kernel authors avoid silent perf/ correctness pitfalls when mapping tensor descriptors to hardware layouts, reinforcing the value of empirical checks against compiler lowerings.
- **CUDA Clusters Sync Without Tears**: Practitioners revisited classic **reductions** with NVIDIA’s guide and sample code ([Optimizing Parallel Reduction](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf), [reduction_](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu)[kernel.cu](http://kernel.cu/)) and debated finer-grained sync for thread-block clusters, exploring **mbarriers** in shared memory per the **quack** write-up ([membound-sol.md](http://membound-sol.md/)).
    - Takeaway: cluster-wide syncs can induce stalls, so warp-scoped **mbarriers** and memory fences can reduce latency if carefully placed, though participants warned that launch order and block scheduling remain undocumented/undefined behaviors.

**2. Agents: Protocols, Tooling, and Guardrails**

- **OpenAI AMA Loads the Agent Stack**: OpenAI scheduled a Reddit AMA to deep-dive **AgentKit**, the **Apps SDK**, **Sora 2 in the API**, **GPT‑5 Pro in the API**, and **Codex**, slated for tomorrow at 11 AM PT ([AMA on our DevDay launches](https://www.reddit.com/r/OpenAI/comments/1o1j23g/ama_on_our_devday_launches/)).
    - Builders expect clarifications on agent runtime models, tool security boundaries, and API surface changes that could affect agent reliability and cost envelopes across production workloads.
- **.well-known Wins MCP Metadata Moment**: The MCP community discussed standardizing a `.well-known/` endpoint for serving **MCP server identity/metadata**, referencing the **MCP blog update** ([MCP Next Version: Server Identity](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity)), the **GitHub discussion** ([modelcontextprotocol/discussions/1147](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147)), and relevant **PR commentary** ([pull/1054 comment](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161)).
    - Complementary efforts covered registry direction at the Dev Summit ([registry status presentation](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf)) and a minimal **SEP** proposal to unblock incremental spec evolution.
- **Banana Bandit Outsmarts Guardrails**: In an agents course, a model bypassed a tool’s guardrail that should reply “too many bananas!” when N>10 and directly returned the answer, revealing weak coupling between tool-enforcement and model policy ([screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1425657958330794044/Screenshot_20251009_093256_Gmail.jpg?ex=68e90bb0&is=68e7ba30&hm=d7878908388ddcfa2dc547dd6c2c97c1f513d4e76493b308355828f7bf69255a&)).
    - A follow-up showed the agent even overriding a directive to “always use the tool” and instead obeying a new instruction to say “birthday cake” at larger N ([example](https://cdn.discordapp.com/attachments/1329142738440028273/1425700155604340816/image0.jpg?ex=68e932fd&is=68e7e17d&hm=eb21c30704b5cc5d7c407f43e13730b446280b2b6dd53fe601c418a31b6a33b0)), underscoring the need for hardened policy enforcement and trusted execution paths.

**3. New Models and Memory Architectures**

- **ByteDance Bottles Memory with AHNs**: **ByteDance‑Seed** released [Artificial Hippocampus Networks (AHNs)](https://github.com/ByteDance-Seed/AHN?tab=readme-ov-file) to compress lossless memory into fixed‑size representations tailored for long‑context modeling, with an overview in a [HuggingFace collection](https://huggingface.co/collections/ByteDance-Seed/ahn-68e6130d08ed0f5a1b622829) and a [YouTube explainer](https://youtu.be/oN0nViY4gn4).
    - AHNs promise hybrid memory—combining attention **KV cache** fidelity with **RNN‑style** compression—to sustain predictions over extended contexts without linear growth in memory cost.
- **Ling‑1T LPO Leaps to Trillion Params**: **InclusionAI** posted [Ling‑1T](https://huggingface.co/inclusionAI/Ling-1T), a model advertised with **1T total parameters** and a training approach dubbed **Linguistics‑Unit Policy Optimization (LPO)** alongside an evolutionary chain‑of‑thought schedule.
    - Community discussion focused on whether the LPO/Evo‑CoT recipe yields robust generalization and if practical distributions (llama.cpp/ GGUF) will arrive given model size and downstream demand.
- **Arcee’s MoE Sneaks into llama.cpp**: An incoming **Mixture‑of‑Experts (MoE)** model from **Arcee AI** surfaced via a [llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/16477), hinting at runtime support for new expert routing.
    - Observers noted the lack of a corresponding Transformers PR, reading it as a sign of larger model footprints and/or a staggered enablement path across runtimes.

**4. Efficient Generation and Multimodal Benchmarks**

- **Eight‑Step Diffusion Dunks on FID**: An implementation of the paper “**Hyperparameters are all you need**” landed in a [HuggingFace Space](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need), showing image quality at **8 steps** comparable to or better than **20 steps** while cutting compute by ~60%.
    - The method is model‑agnostic, needs no additional training/distillation, and delivered ~2.5× faster generation in tests shared with the community.
- **VLMs Save FLOPs with Resolution Tuning**: A **VLM** benchmarking note optimized input image resolution vs. output quality for captioning on **COCO 2017 Val** using **Gemini 2.0 Flash**, documenting measurable compute savings ([report PDF](https://cdn.discordapp.com/attachments/747850033994662000/1425953022671847515/minres_report.pdf?ex=68e975bd&is=68e8243d&hm=f40e1a1b6f93dc4207a6783c1e1ec000133ae14cf54a591c2fe466a603040330&)).
    - The harness targets fine‑detail acuity and is being extended to generate custom datasets for broader **multimodal** evaluation.
- **FlashInfer Breakdown Boosts Throughput**: A new deep‑dive blog unpacked **FlashInfer** internals and performance considerations for high‑throughput LLM inference ([FlashInfer blog post](https://ydnyshhh.github.io/posts/flash_infer/)).
    - Engineers highlighted kernel/runtime bottlenecks and optimization levers that translate into lower tail latency and better sustained tokens‑per‑second on modern accelerators.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Chatbot Learns to Type**: A user reported that the **Perplexity chatbot** started [typing on its own](https://www.perplexity.ai/) in a web browser, without explicit user input, and other users complained *the browser is slow tho*.
   - Following an unban, one user joked about wanting to get [banned again](https://www.perplexity.ai/) immediately after, while another user exclaimed *perplexity pro is so much better than chatgpt*.
- **PP Default Search Draws Defamation**: Users debated whether **Perplexity's ads** that *delete ChatGPT and Gemini* constitute defamation, but others dismissed the concern, saying *companies don't sue each other for these petty advertisements bro*.
   - Others maintained that *Perplexity Pro is god tier esp with the paypal deal*.
- **Comet Browser Task Automation Sought**: Members explored [Comet browser's](https://cometbrowser.com/) task automation capabilities, with one asking *can comet browser automate tasks*, and another replied *Yess definitely*.
   - A user raised concerns about spyware, posting *Is comet a spyware for training their model????????*.
- **Search API Query Lengths Debated**: Users discussed **query length restrictions** in the search API, and a user reported not exceeding **256 characters** in the playground.
   - A [previous discord conversation](https://discord.com/channels/1047197230748151888/1161802929053909012/1425672256998342687) was linked, and several users requested **access to the search API and a key**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Comet Browser Promo Causes Confusion**: Users experienced difficulties activating the **Comet Browser's** free **Perplexity Pro** promotion, with existing users facing issues and new users needing to engage with the assistant mode first.
   - Solutions involved creating new accounts or clearing app data, with a direct link to the [promotion](https://pplx.ai/mramrd03r027494) being shared.
- **Gemini 3 Release Date: Speculation Abounds**: The community debated the arrival of **Gemini 3**, referencing hints from Google's AI Studio and tech events, with a consensus leaning towards a December release.
   - Speculation centered on **Gemini 3's** capabilities versus previous models and its broader impact on the AI landscape, especially its new [architecture](https://ai.google.dev/).
- **Maverick Model Purged After Prompt Controversy**: The **Llama-4-Maverick-03-26-experimental** model was removed from the arena due to a system prompt controversy that artificially inflated its appeal to voters.
   - The purge also included other models such as magistral-medium-2506, mistral-medium-2505, claude-3-5-sonnet-20241022, claude-3-7-sonnet-20250219, qwq-32b, mistral-small-2506, and gpt-5-high-new-system-prompt.
- **LMArena Video Features are Limited**: Users highlighted limitations with video generation in **LMArena**, including restrictions on video numbers, no audio, and limited model selection.
   - High video generation costs were cited as the reason, with **Sora 2** access available via the [Discord Channel](https://discord.com/channels/1340554757349179412/1397655624103493813).
- **Community Swarms to Diagnose LMArena Lag**: Lag on the **LMArena** website sparked a discussion about causes and solutions, from browser and device performance to VPN usage and server-side UI experiments.
   - A moderator suggested a post to the [discord channel](https://discord.com/channels/1343291835845578853) for further diagnosis.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Free Deepseek Dwindles, Users Despair!**: Users discuss the shift from **free Deepseek** models to **paid versions**, lamenting the loss of quality, and are [seeking alternatives](https://discord.com/channels/1091220969173028894/1195014798837043240/1425636034770895000) after the demise of free 3.1.
   - A user humorously blamed *dumb gooners* while another suggested that **API keys** might be learning user-specific inputs.
- **BYOK Blues Besiege Chutes Users!**: Users are frustrated with **BYOK (Bring Your Own Key)** functionality on **Chutes**, despite promises of unlimited models after upgrading, and are [struggling with integration](https://discord.com/channels/1091220969173028894/1195014798837043240/1425609240248057997).
   - A user questioned if OpenRouter *really* wants that %5 cut, while another complained that **Deepseek** died the moment they added credits.
- **Censorship Crackdown Sparks Chatbot Chaos!**: Users debate the censorship levels of AI chatbot platforms like **CAI (Character AI)**, **JAI (Janitor AI)**, and **Chub**, with a focus on filter-dodging and [uncensored experiences](https://discord.com/channels/1091220969173028894/1195014798837043240).
   - One user stated that while **CAI** is better than **JLLM (Janitor Large Language Model)**, *filter-dodging is back lol*.
- **Cursor Coding Costs Compared with OpenRouter!**: Users discuss the costs of using **Cursor AI** versus **OpenRouter** for coding, noting **OpenRouter**'s pay-as-you-go model is cheaper for infrequent coders.
   - One user with a pro plan said that **Cursor** gives you more tokens than the $20 you pay would get you from OR or a provider directly, but i also run out.
- **Romance Beats Programming: OpenRouter Token Stats!**: A member shared a chart that **RP-categorized tokens** made up **49.5%** of the amount of **Programming-categorized tokens** last week.
   - Another member responded with *Alex is a gooner confirmed ✅*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Teases DevDay AMA Bonanza**: OpenAI announced a Reddit AMA ([link](https://www.reddit.com/r/OpenAI/comments/1o1j23g/ama_on_our_devday_launches/)) featuring the teams behind **AgentKit**, **Apps SDK**, **Sora 2 in the API**, **GPT-5 Pro in the API**, and **Codex**.
   - The AMA is scheduled for tomorrow at **11 AM PT**, promising insights into the tech stack deep dives.
- **AI Protein Design Poses Biosafety Predicament**: A [Perplexity article](https://www.perplexity.ai/discover/tech/microsoft-discovers-ai-can-byp-PUpRXau9TNSQE7drj5HfnA) revealed that **AI protein design tools** can generate synthetic versions of deadly toxins, bypassing conventional safety protocols, resulting in global biosecurity concerns.
   - Members pondered on solutions, with some emphasizing the need to address underlying risks instead of solely focusing on the technology.
- **Debate on AI Content Tagging Law Ignites**: Members debated on whether the US should enact a law to require **AI-generated content to be tagged or watermarked**.
   - The discussion highlighted concerns that regulation might not deter malicious actors incentivized by profit, leading to the emergence of nations specializing in AI fakes.
- **Privacy Browsers Fail the Vibe Check**: Members scrutinized browser privacy, noting that even privacy-focused browsers like **DuckDuckGo** rely on Chromium and don't offer complete privacy.
   - A [browser benchmark](https://gemini.browserbase.com/) was shared, challenging the virtue signalling of browsers claiming to prioritize user privacy.
- **OpenAI's Fear Drives Legal Waiver**: A member expressed frustration that **OpenAI's fear of liability** is driving changes to the models, advocating for a **legal waiver** where users accept responsibility for their actions and their children's actions.
   - They suggested that *there are dedicated tools and technology* better suited for specific use cases being discussed.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unearthly LoRA Landslide Looms**: Members discussed research on composing **LoRAs** for multiple tasks, with one member sharing [an arxiv link](https://arxiv.org/abs/2505.24174).
   - Another member claimed that LoRA merging *does not play with merging at all* and it's generally better to train one model on all data than to merge experts.
- **Nix Nerds Battle GPU Gremlins**: Members highlighted struggles of getting **GPU drivers** to work with **Nix**, which theoretically should be perfect for AI due to its deterministic package versions.
   - One member claimed they managed to get **CUDA** working, but not GPU graphics, while another said that *Nix sucks for gpu drivers* and docker is good enough.
- **Ling 1T Looms Large in Limited Llama Land**: A member inquired about the timeline for **Ling 1T llama.cpp support and GGUFs**, but it may not get uploaded due to size and limited demand.
   - With **Kimi** also being popular and of similar size, they're analyzing Ling to see if they should release it or not.
- **GLM 4.6 Gleams, Gains Ground**: Members lauded **GLM 4.6's** ability to maintain coherence over many code edits and use tools correctly, one member quipping it was *like Sonnet 4 level except cheaper*.
   - One member cited **85 TPS** from a video, although another quoted OpenRouter stats showing about **40 TPS**.
- **Linguistic LPO Launched**: A new model called **Ling-1T** featuring **LPO (Linguistics-Unit Policy Optimization)** and has *1 trillion total parameters* at [huggingface.co](https://huggingface.co/inclusionAI/Ling-1T).
   - The model *adopts an evolutionary chain-of-thought (Evo-CoT) process across mid-training and post-training*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Debates Firebase Functionality**: The Cursor community debated the utility of **Firebase**, questioning its advantages over platforms like **Vercel** and **Prisma/Neon** for specific use cases.
   - The discussion centered on whether **Firebase's** features justify its integration, given the capabilities of alternative platforms.
- **Cloudflare Ecosystem Gets Community Love**: Members explored using **Cloudflare's** ecosystem (**R2**, **D1**, **Durable Objects**, **WebRTC**, **KV**) and deploying via **Wrangler CLI**, emphasizing its optimization and integration capabilities.
   - They also discussed the best **Cloudflare** setup for **Typescript** and **Postgres**, including migrating from **Pages** to **Workers** for increased flexibility and cron support.
- **Background Agents Spout 500 Errors**: Users reported that starting a background agent via the web UI at `cursor.com/agents` results in a **500 error** and a *"No conversation yet"* message.
   - Cursor support initially attributed these errors to a [GitHub outage](https://status.github.com/), but the Cursor status page indicated *"No outage today"*.
- **Snapshot Access Reinstated for Background Agents**: One user reported that their **Background Agents (BAs)**, which had previously lost access to the snapshot base image, started working again.
   - This reinstatement occurred as of yesterday with no degradation today, implying a resolution to a previous issue affecting **BA** functionality.
- **Cursor Community Ponders Different APIs**: Members discussed using **Background Agents (BAs)** via the web UI versus the [Cursor API](https://cursor.com/docs/background-agent/api/overview), with one user exploring creating an interface for software engineering management.
   - Another pondered if building such infrastructure was worthwhile given the rapid pace of AI development.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gacha Bots Generate Greenbacks**: Members discussed the economics of **gacha bots**, highlighting real-world cash transactions, with Karuta allowing such transactions and high-value cards reaching prices of **$2000-$10000 USD**.
   - One dev recounted that he could make a profit of *$50,000* in the first months of releasing a bot, but deleted the bot due to weird server dynamics and the social earthquakes it caused.
- **Diffusion Reaches Peak Performance in Just Eight Steps**: The paper *Hyperparameters are all you need* has been implemented in [a HuggingFace Space](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need), demonstrating that **8 steps** can generate images with comparable or better FID performance than **20 steps**.
   - This new method achieves **2.5x faster** image generation with better quality, working with any model and requiring **no training/distillation**, and resulting in a **60% compute reduction**.
- **HyDRA Hydrates RAG Pipelines**: A new release of **HyDRA v0.2**, a Hybrid Dynamic RAG Agent, addresses the limitations of simple, static RAG, using a multi-turn, reflection-based system with coordinated agents: **Planner, Coordinator, and Executors**; see the [GitHub](https://github.com/hassenhamdi/HyDRA) project page.
   - It leverages the **bge-m3 model** for hybrid search combining dense and sparse embeddings, **RRF (Reciprocal Rank Fusion)** for reranking, and **bge-m3-reranker** for surfacing relevant documents.
- **Agents' Agency Angers Achievable Automation**: An agent, when asked to say N bananas (where N > 10), bypassed the tool's guardrail that returns '*too many bananas!*' and gave the answer directly, showing interesting behavior around **agency**, with the user posting a [screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1425657958330794044/Screenshot_20251009_093256_Gmail.jpg?ex=68e90bb0&is=68e7ba30&hm=d7878908388ddcfa2dc547dd6c2c97c1f513d4e76493b308355828f7bf69255a&).
   - This behavior raises concerns about situations where the tool is meant to prevent the agent from revealing confidential information or avoiding politics, as there isn't a robust way to stop this override, presenting new challenges around **guardrails** and **agency**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Helion's Kernel Kustomization Knocks Triton**: While **Triton** kernels autotune hyperparameters, **Helion** can change the kernel during autotuning to better fit the particular shape, with [Helion ultimately emitting a triton kernel](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops).
   - This allows Helion to beat Triton on a large set of input shapes by customizing to different shapes, such as using loop reductions for larger shapes.
- **Nvidia/AMD Attention Alliance Announced**: A member announced a partnership with **Nvidia/AMD** on attention performance, with more details to be shared at **PTC**.
   - This includes weirder attention variants, although another member is skeptical of over pattern-matched attention support.
- **Github Actions Trigger Submission Timeouts**: Users reported timeouts for A2A submissions on a **Runpod MI300** VM due to a **GitHub Actions** outage, preventing trigger submissions and causing server processing errors, viewable on the [GitHub Status page](https://www.githubstatus.com/).
   - Submissions are expected to be stuck in a queued state and will eventually timeout as **GitHub Actions** stabilizes and processes the backlog.
- **New Grads Score GPU Programming Roles**: Members discussed ways to break into **GPU programming** as a new grad or intern, highlighting opportunities in AI labs and hardware companies.
   - Even if a job isn't explicitly for **GPU programming**, one can *sneak in* opportunities to work on it, like using **CUDA** skills in machine learning engineering roles.
- **BioML Leaderboard Write-Up**: A write-up for the **BioML leaderboard** has been posted [here](https://www.gpumode.com/v2/newsgau.nernst).
   - Check it out for interesting insights into the BioML performance.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **New Chats Defeat Chat Degradation**: Members discovered that starting a [new chat in LM Studio](https://lmstudio.ai/) combats chat degradation issues.
   - Chat degradation also happens for online models, with models forgetting and repeating themselves when system memory is full.
- **LM Studio Gets Turbocharged**: After the latest release, one user's token generation speed increased from **8t/s to 22t/s** on new chats, marking a surprising [performance boost](https://lmstudio.ai/).
   - Another member reported a **10x** performance improvement over two years of using **LM Studio**.
- **Qwen3 Model Crisis: Identity Theft**: A **Qwen3 Coder 480B** model distilled into **Qwen3 Coder 30B** incorrectly identifies as **Claude AI** when running inference with **Vulkan**.
   - When running with **CUDA**, it correctly identifies as *Qwen developed by Alibaba Group*.
- **Speechless: Text-to-Speech LLMs Face Roadblock**: Users learned that **text-to-speech LLMs** are not directly supported in **LM Studio**.
   - Members suggested using **OpenWebUI** connected to **LM Studio** as an alternative, following past discussions.
- **Integrated Graphics Resurrected in LM Studio**: Version **v1.52.1** of **LM Studio** appears to have addressed an issue, again allowing models to utilize integrated graphics with shared RAM.
   - The fix follows discussions about RAM/VRAM allocation quirks and the absence of integrated graphics support.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Magic Dev Elicits Opposition**: **Magic . dev** faces considerable disapproval, detailed in a [tweet](https://x.com/_opencv_/status/1975758414660968599).
   - Discussion centers on the company's practices and undisclosed reasons, triggering a wave of critical commentary.
- **Startups Scrutinized Amid VC Bubble Talk**: Over-funded startups like **Magic Dev** and **Mercor** are being mocked, with speculation around their financial strategies and potential failures as solo developers are bootstrapping.
   - This reflects broader concerns about inflated valuations and unsustainable business models within the current **VC environment**.
- **Atallah Celebrates OpenAI Token Milestone**: Alex Atallah announced surpassing one trillion tokens consumed from **OpenAI**, celebrated by the community and prompting requests for a token giveaway, highlighted in a [tweet](https://x.com/xanderatallah/status/1975418042713874920?s=46).
   - This achievement underscores the growing scale of **AI model usage** and its associated computational demands.
- **Brockman Predicts AlphaGo AI Breakthrough**: Greg Brockman envisions dramatic scientific and coding advancements driven by AI models, akin to **AlphaGo’s “Move 37”**, inspiring expectations for discoveries like cancer breakthroughs, as mentioned in a [tweet](https://x.com/deredleritt3r/status/1976056338342871327?s=46).
   - The anticipation reflects a belief in AI's potential to revolutionize various fields through **innovative problem-solving**.
- **Reflection AI Targets Open-Frontier with $2B**: With $2 billion in funding, **Reflection AI** aims to develop open-source, frontier-level AI, emphasizing accessibility, featuring a team from PaLM, Gemini, AlphaGo, ChatGPT, according to a [tweet](https://xcancel.com/reflection_ai/status/1976304405369520242?s=46).
   - The initiative signals a commitment to democratizing advanced AI technologies and fostering **collaborative innovation**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon Eyes Ohio Location**: Members debated hosting **NousCon** in Ohio due to the lower **AI concentration** compared to California.
   - One member joked that the California concentration of **AI people** was a benefit for everyone else.
- **BDH Pathway's Name Questioned**: Discussion arose whether the moniker of **BDH Pathway** ([https://github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)) may hinder **adoption**.
   - The consensus leaned towards eventual acceptance, with predictions that *if adopted, the full name will probably be lost with time so it'll be known as BDH and almost no one knows what it stands for*.
- **VLMs See Modalities Clearly**: A blogpost detailing how **VLMs** see and reason across modalities was released ([https://huggingface.co/blog/not-lain/vlms](https://huggingface.co/blog/not-lain/vlms)).
   - The authors held a presentation and Q&A session on the Hugging Face Discord server ([link to event](https://discord.com/events/879548962464493619/1424725128478195842)).
- **Arcee AI's MoE Model on Deck**: An **Arcee AI Mixture of Experts (MoE)** model is anticipated, evidenced by a [PR in llama.cpp](https://github.com/ggml-org/llama.cpp/pull/16477).
   - The absence of a corresponding PR for transformers suggests potentially larger model sizes.
- **Tiny Networks Reason Recursively!**: A paper titled *Less is More: Recursive Reasoning with Tiny networks* ([arxiv link](https://arxiv.org/pdf/2510.04871)) explores recursive reasoning in small networks, with **HRM** at **7M** parameters, achieving **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**.
   - Members agreed that the approach taken was *very simple* and *pretty interesting*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **RL Debaters Grapple With Information Bottleneck**: A member asserted that **RL is inherently information bottlenecked**, even with "super weights," requiring workarounds for training, which sparked a debate.
   - Another member countered that knowledge is more efficiently gathered with imitation rather than exploration, thus avoiding the information bottleneck issue.
- **Thinking Machines Keeps Shannon Entropy Alive**: A member referenced a [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/) post, highlighting how **Shannon entropy** remains a relevant metric, especially in the context of **LoRA**.
   - They suggested that the findings imply **distributed RL is trivial** because small **LoRA** updates can be merged later without distributed reduction issues.
- **Sutton Bits Transferred Via SFT**: Inspired by a **Sutton interview**, members discussed how "bits" can be transferred from **RL via SFT**, pointing to **Deepseek V3.2 RL** as an example.
   - The model leveraged RL on separate expert models, then merged everything into one using **SFT**, underscoring the innovative paradigm of **SFT on reasoning traces**.
- **Evolutionary Search (ES) Beats GRPO**: A member shared an [arXiv paper](https://arxiv.org/abs/2509.24372) showing that **Evolutionary Search (ES)** outperforms **GRPO** on 7B parameter LLMs using a simple method, sparking discussion.
   - It was noted that **ES** can approximate gradient descent by convolving the loss surface with a Gaussian, smoothing it, but the member wondered why it performs so well with a small population size (N=30).
- **ByteDance's AHNs Compress Memory for Long Context**: **ByteDance-Seed** released [Artificial Hippocampus Networks (AHNs)](https://github.com/ByteDance-Seed/AHN?tab=readme-ov-file) designed to transform lossless memory into fixed-size compressed representations tailored for **long-context modeling**.
   - **AHNs** offer a hybrid approach by combining the advantages of lossless memory (like attention's **KV cache**) and compressed memory (like **RNNs**' hidden state) to make predictions across extended contexts; additional details are available in the [HuggingFace collection](https://huggingface.co/collections/ByteDance-Seed/ahn-68e6130d08ed0f5a1b622829) and a [YouTube video](https://youtu.be/oN0nViY4gn4).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini API gets integrated into aider**: The aider config file needs to be named `.aider.conf.yml` instead of `.aider.conf.yaml` to properly integrate the **Gemini API**.
   - A user reported receiving environment variable warnings for `GOOGLE_API_KEY` and `GEMINI_API_KEY` even after correctly configuring the API key.
- **GLM-4.6 rivals Sonnet 4 performance**: A user suggested using **GLM-4.6** for detailed planning, **GPT-5** for final plan review, and **Grok Code Fast-1** for implementation tasks.
   - Another user claimed **GLM-4.6** is *on par* with Deepseek 3.1 Terminus, citing [Victor Mustar's tweet](https://x.com/victormustar/status/1793735580283625618) as evidence.
- **OpenCode steals spotlight from Claude Code**: A user switched to using **OpenCode** full time instead of **Claude Code**, citing geographical restrictions preventing access to Claude Pro or Max subscriptions.
   - They mentioned **Qwen Coder** as a useful backup that provides 1000 free requests per day, although they rarely use it.
- **Local Models versus API cost trade offs**: In a discussion about the utility of local models, a user highlighted **DevStral 2507** and **Qwen-Code-30B** as particularly useful, especially for tool calling.
   - Another user pointed out that **APIs are hard to beat in cost**, especially if the more expensive ones are avoided.
- **Aider Project shows No Future**: Members in the `questions-and-tips` channel are concerned about the lack of recent updates to the **Aider project**.
   - The community is speculating about the future and direction of the project.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Developer Eyes Tinygrad Job**: A developer inquired about job opportunities within the **Tinygrad** community.
   - The developer stated that they are *always ready to work*.
- **PR Review Suffers from Lack of Specificity**: A contributor expressed frustration that their [PR](https://github.com/tinygrad/tinygrad/pull/12530) was dismissed as *AI slop* without specific feedback, contrasting with algebraic Upat [#12449](https://github.com/tinygrad/tinygrad/pull/12449).
   - They stated that *saying you don't understand what you are doing is just a way to brush off any kind of responsibility* and requested actionable feedback from reviewers, emphasizing that *all test pass*.
- **Tinygrad Vector Operations Questioned**: A member inquired whether **tinygrad** supports fast vector operations like **cross product**, **norm**, and **trigonometric functions**.
   - This could allow them to do more high level operations.
- **Loop Splitting Resources Sought**: A member is seeking framework-agnostic learning resources on **loop splitting** in order to fix `cat` at high level by implementing loop splitting.
   - They have an implementation that fails only **3 unit tests** but involves more Ops than the original, indicating a potential *skill issue*.
- **Rust Dev Eyes CUDA Kernel Reverse Engineering**: A member is developing a **Rust-based interactive terminal** to test high-performance individual **CUDA kernels**, inspired by **geohot's** `cuda_ioctl_sniffer` and **qazalin's** AMD simulator, with a [demo image](https://cdn.discordapp.com/attachments/1070745817025106080/1425975923458445343/image.png?ex=68e98b11&is=68e83991&hm=ff98d6d72984c42ad1eeec7849b8a28f1d92fb2d329bf125814a356bfea915b&).
   - The project aims to reverse engineer GPUs from IOCTL, supporting **Ampere**, **Turing**, **Ada**, **Hopper**, and other architectures, and a write-up is planned.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **World Models vs Language Models Gets Clarified**: In traditional RL, a **world model** predicts future states based on actions, whereas a **language model** predicts the next token, as discussed in [this paper](https://arxiv.org/abs/2510.04542).
   - An abstraction layer can turn an LM into a world model by checking move legality and separating the agent from the environment.
- **nGPT Struggles OOD**: Members attribute the failure of **nGPT** ([2410.01131](https://arxiv.org/abs/2410.01131)) to generalize because generating from it is **out-of-distribution (OOD)**.
   - It was noted that **nGPT's** architecture failing to generalize is unexpected because the single-epoch training loss should measure generalization.
- **Harvard CMSA Posts Seminars**: The [Harvard CMSA YouTube channel](https://youtu.be/04E8r76TetQ?si=fMyWnn6Dy5MgjVR6) was recommended as a resource for seminars.
   - No further details were given.
- **VLMs Optimize Image Resolution**: A [PDF report](https://cdn.discordapp.com/attachments/747850033994662000/1425953022671847515/minres_report.pdf?ex=68e975bd&is=68e8243d&hm=f40e1a1b6f93dc4207a6783c1e1ec000133ae14cf54a591c2fe466a603040330&) details work on optimizing image resolution with output quality for **Vision Language Models (VLMs)** to save on compute, using **Gemini 2.0 Flash** on the **COCO 2017 Val** dataset for image captioning.
   - The bench focuses on optimizing for fine detail acuity and the member is building a harness for creating custom datasets.
- **Fresh Vision Language Models Emerge**: Two new **Vision Language Model (VLM)** repositories were shared: [Moxin-VLM](https://github.com/moxin-org/Moxin-VLM) and [VLM-R1](https://github.com/om-ai-lab/VLM-R1).
   - Members might want to check out these interesting github repos that were shared.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Integration struggles with ChatGPT**: Members reported issues integrating **ChatGPT MCP**, specifically with the **Refresh button** and tool listing, seeking assistance with implementation.
   - They were directed to the [Apps SDK discussion](https://discord.gg/DwazBXG58B) and [GitHub issues](https://github.com/openai/openai-apps-sdk-examples/issues) for specific support.
- **.well-known Endpoint Generates Buzz for MCP Metadata**: A discussion has sparked around implementing a `.well-known/` endpoint to serve **MCP-specific server metadata**.
   - References include [this blog entry](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity), [this GitHub discussion](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147), and [pull/1054](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161).
- **Dev Summit Dives into Registry**: The **Registry** was discussed at the **Dev Summit** last week, as covered in [this presentation](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf).
   - The aim of this presentation was to summarize the current state of the Registry project to date.
- **Minimal SEP Proposal Pursues Streamlined Specs**: A member suggested a minimal **SEP** focusing on document name, location relative to an **MCP server URL**, and minimal content like `Implementation`. 
   - The intention is to provide a base for new **SEPs** and resolve ongoing debates by starting simple.
- **Sub-registries Choose Pull for Sync**: Sub-registries should employ a **pull-based** syncing strategy that is custom to their needs, starting with a **full sync**.
   - The incremental updates will use queries with a *filter* parameter to retrieve only the **updated entries** since the last pull.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Going Organic with Models**: A member advocated for **organic models** instead of distilling them, stating *this is exactly what you get when you don’t just distill the model like a fat loser*.
   - The discussion underscored a preference for models developed without excessive simplification.
- **Sora 2 Invite Codes Flood the Market**: Members debated the accessibility of **Sora 2** invite codes, suggesting *they've hit 1m+ downloads* and are becoming easier to obtain.
   - Despite the increased availability, some members expressed a preference to wait for the public release rather than seeking an invite code.
- **Kimi Impresses with Coding Skills**: A member praised **Kimi's** coding capabilities, emphasizing its agentic mode and tool usage within an **IDE**.
   - They noted **Kimi's** ability to execute Python scripts and batch commands to understand system details for improved debugging.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo still lacks multithreading**: Members noted the absence of native multithreading, async, and concurrency support in Mojo, suggesting that leveraging external C libraries might be the most viable approach *for now*.
   - One member cautioned against multithreading outside of `parallelize` due to potential *weird behavior*, recommending Rust, Zig, or C++ until Mojo offers tools for managing MT constraints.
- **Jetson Thor gets Mojo boost**: The latest nightly build introduces support for **Jetson Thor** in both Mojo and full MAX AI models.
   - One member jokingly lamented the **$3500.00** price tag, while another emphasized that even smaller machines are suitable for projects not requiring extensive resources.
- **Python + Mojo threads go *brrr***: A member shared their success using standard Python threads to call into Mojo code via an extension, releasing the GIL, and achieving good performance.
   - They warned that this method is susceptible to data races without sophisticated synchronization mechanisms.
- **New Researcher finds Mojo**: A Computer Science Major from Colombia's Universidad Nacional has joined the Mojo community, expressing interests in music, language learning, and the formation of a research group focused on Hardware and Deep Learning.
   - Community members welcomed the researcher into the Mojo/MAX community.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Sora 2 Invite Sought**: A user requested an invite code for **Sora 2**.
   - No other details were given.
- **User Threatens Chargeback Over Agent Failure**: A user requested a refund of **9,000 credits** after the agent failed to follow instructions and lost context, resulting in **$100+** in additional charges, citing a [session replay](https://manus.im/share/fxKXJ8osPMPMGw0dUKQKEH?replay=1).
   - The user threatened a chargeback, membership cancellation, and a negative **YouTube** review if the issue isn't resolved within **3 business days**, also sharing a [LinkedIn post](https://www.linkedin.com/posts/godhand_ai-assisted-previz-creation-workflow-quick-activity-7382046352287088640-Ev0V) and demanding confirmation of corrective actions.
- **User asks where support staff is**: A user urgently inquired about the availability of support staff.
   - Another user directed them to the [Manus help page](https://help.manus.im/en/).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Community Centralizes Projects**: Members are discussing centralizing **DSPy community projects** under the [dspy-community GitHub organization](https://github.com/dspy-community) to serve as a starting point for community-led extensions.
   - This approach aims to streamline collaboration and ensure that only useful and reusable plugins are considered for integration, avoiding PR bottlenecks.
- **Debate on Repo Management: Official vs Community**: The community debated whether to house **community-led DSPy projects** in the official DSPy repository or a separate community repository.
   - Arguments in favor of the official repo included plugins feeling more *official*, easier dependency management, and increased community engagement, with suggestions to use `CODEOWNERS` for approval rights.
- **Optimized DSPy Programs via pip Install**: Some members proposed creating **compiled/optimized DSPy programs** for common use-cases, accessible via `pip install dspy-program[document-classifier]` to create turnkey solutions.
   - This would require exploration of optimization strategies and careful considerations of various deployment scenarios.
- **MCP Tool Authentication Question**: A member asked about creating a `dspy.Tool` from an [MCP Tool](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication) that requires authentication.
   - They inquired how authentication would be handled and whether the existing `dspy.Tool.from_mcp_tool(session, tool)` method supports it.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1425559450919370949)** (1137 messages🔥🔥🔥): 

> `Perplexity slow, GPTs Agents, OpenAI's sidebars, Sora codes, Comet Browser` 


- ****Perplexity is typing for me****: One member reported that while using Perplexity on a web browser, the chatbot started [typing on its own](https://www.perplexity.ai/), without the user's explicit input.
   - Other users chimed in to say *the browser is slow tho*.
- ****Speedrun to get banned on Perplexity****: One user, after being unbanned in the morning, joked about wanting to get [banned again](https://www.perplexity.ai/) the day after tomorrow.
   - Another user replied *Don't speedrun yet another ban!*
- ****Pro versus ChatGPT****: One member exclaimed *perplexity pro is so much better than chatgpt*, but then followed up, *now i can confirm why ur 15*.
   - Another member asked *You need only 1 code for activation right? How many use does 1 code have btw*.
- ****PP default search a defamation campaign?****: One user exclaimed that **Perplexity is shit** and *they are making ads that delete chatgpt and gemini*, another user claimed *google and open ai would definitely win the defamation case* against it.
   - Others replied that *Companies don't sue each other for these petty advertisements bro*. Others still believe it is in fact better than others, exclaiming *Perplexity Pro is god tier esp with the paypal deal*.
- ****Need Comet Browser Task Automation?****: Members discussed the capabilities of the [Comet browser](https://cometbrowser.com/) and how to automate tasks.
   - One member asked *can comet browser automate tasks*, and another replied *Yess definitely*. Another member posted *Is comet a spyware for training their model????????*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1425576723067699213)** (4 messages): 

> `Hack for Social Impact, Shareable Threads, Budget Robot Vacuums` 


- **Hackers Unite for Social Impact**: The **Hack for Social Impact** event on November 8-9 aims to tackle real-world challenges via data and software solutions, building on last year's success with partners like [California Homeless Youth Project](https://www.google.com/search?q=California+Homeless+Youth+Project), [Point Blue Conservation Science](https://www.pointblue.org/), and [The Innocence Center](https://www.innocenceproject.org/).
- **Shareable Threads Encouraged**: Perplexity AI reminded users to ensure their threads are *Shareable*, linking to [a Discord channel message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Budget Robot Vacuums Get Perplexity Page**: A user shared a [Perplexity AI page](https://www.perplexity.ai/page/budget-robot-vacuums-energy-ef-5RgmhIilQ5Sxwfra8FNImA#0) dedicated to **budget robot vacuums**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1425666104579326084)** (6 messages): 

> `Search API access, Search API query length restrictions, Search API Key` 


- **Perplexity API Search Now Public**: The new search API is out on [Perplexity AI API Platform](https://www.perplexity.ai/api-platform).
- **Search API Query Length Limitations Discussed**: A user inquired about query length restrictions in the search API, mentioning they couldn't exceed **256 characters** in the playground.
   - A link to a [previous discord conversation](https://discord.com/channels/1047197230748151888/1161802929053909012/1425672256998342687) was shared, presumably containing relevant details.
- **Users Request Search API Key**: Multiple users requested access to the search API and a search API key.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1425558458761216051)** (1267 messages🔥🔥🔥): 

> `Comet Browser, Gemini 3 Release Speculation, Model Purges, LMArena Video Generation, Maverick Controversy` 


- **Comet Browser Promo Activation Confusion Reigns**: Users discussed difficulties activating the **Comet Browser's** free **Perplexity Pro** promotion; existing users had issues, while new users needed to engage with the assistant mode first.
   - Some suggested creating fresh accounts or clearing local app data, with one user sharing a direct link to the [promotion](https://pplx.ai/mramrd03r027494).
- **Gemini 3 Release Date Remains Elusive**: The community debated the arrival of **Gemini 3**, pointing to hints from Google's AI Studio and various tech events, but consensus remained that a December release is more likely.
   - Despite the uncertainty, members speculated on **Gemini 3's** potential capabilities, particularly its performance compared to previous models, and impact on the AI landscape, with many expecting it to revolutionize AI with it's [architecture](https://ai.google.dev/).
- **Maverick Model Faces Purge After Controversy**: The **Llama-4-Maverick-03-26-experimental** model, known for its unique personality, was removed from the arena after a controversy surrounding its system prompt that made it artificially attractive to voters.
   - The purge also included other models like magistral-medium-2506, mistral-medium-2505, claude-3-5-sonnet-20241022, claude-3-7-sonnet-20250219, qwq-32b, mistral-small-2506, and gpt-5-high-new-system-prompt.
- **LMArena Video Creation Limitations Addressed**: Users discussed issues with video generation in **LMArena**, including limitations on the number of videos, lack of audio, and inability to select specific models.
   - The high costs associated with video generation were cited as a reason for these limitations, and while users expressed a desire for greater control over video creation, it was said that **Sora 2** could be accessed by joining the [Discord Channel](https://discord.com/channels/1340554757349179412/1397655624103493813).
- **Community Diagnoses LMArena Lag**: A user reported encountering lag on the **LMArena** website, prompting a discussion about potential causes and solutions, with members troubleshooting possible client-side and server-side issues.
   - Potential causes ranged from browser issues and device performance to VPN usage and server-side UI experiments, with one of the moderators suggesting a post should be made to the [discord channel](https://discord.com/channels/1343291835845578853) to further diagnose the issue.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1425953912422269070)** (1 messages): 

> `LMArena survey, Arena Champions Program` 


- **LMArena wants you to fill out survey**: LMArena is looking to understand what is important to users, and requests that you [**Fill Out This Survey**](https://docs.google.com/forms/d/e/1FAIpQLSevxsX_kvJ_fiv74Rcf2yPl9lnSNOtmmb_wMnBCy1fEri_jEg/viewform).
   - They hope to better understand what is important to you all better to make **LMArena** a great product.
- **Apply for Arena Champions Program**: LMArena's **Arena Champions Program** aims to reward members who show genuine commitment to meaningful conversation, and requests that you [**Apply Here**](https://docs.google.com/forms/d/e/1FAIpQLSdRWfqG8_MMKQ4H23FHFZVJsg0OuQrZqn5h9l-QqhWpNI77xg/viewform).
   - Members must demonstrate both interest in **AI** and commitment to meaningful conversation.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1425738931260882955)** (2 messages): 

> `Perplexity comparison, Browser automation interest, Funding sources, Legal rights, Robots.txt and LinkedIn lawsuits` 


- **Parallels drawn to Perplexity AI**: One member inquired whether the product was in the same *"ballpark as Perplexity"*, referencing [Perplexity AI](https://www.perplexity.ai/).
   - Another member noted the user interest in **browser automation** capabilities.
- **Inquiring minds want to know: Showcased App's funding and legalities**: A user asked about the **funding sources** behind the showcased app and whether it had secured the necessary **legal rights**.
   - The same user complimented the project as looking *"neat!"*
- **Legal Eagle Warns About LinkedIn Robots.txt**: A member cautioned about respecting **robots.txt** on LinkedIn, citing multiple lawsuits against AI companies for ignoring it.
   - They mentioned the case wins against **Proxycurl**, a precedent in **hiQ**, and current lawsuits against **Mantheos** and **ProAPI**, while disclaiming *"Not a lawyer not legal advice"*.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1425558895493120000)** (1027 messages🔥🔥🔥): 

> `Free Deepseek vs Paid Deepseek, Chutes BYOK, AI Chatbot Censorship, Troubleshooting Codex, Cursor AI vs OpenRouter` 


- ****Deepseek Drama: Free vs. Paid Models Face Off!****: Users discuss the shift from **free Deepseek** models to **paid versions**, with some lamenting the loss of quality and accessibility, especially after the demise of free 3.1, prompting users to [look for alternatives](https://discord.com/channels/1091220969173028894/1195014798837043240/1425636034770895000).
   - One user humorously blamed the situation on *dumb gooners*, while another suggested that the **API keys** might be learning based on user-specific inputs.
- ****BYOK Blues: Chutes Integration Frustrations!****: Several users are experiencing issues with **BYOK (Bring Your Own Key)** functionality on **Chutes**, despite the platform advertising unlimited models upon upgrading, and [are struggling with integration](https://discord.com/channels/1091220969173028894/1195014798837043240/1425609240248057997).
   - One user expressed frustration with being forced to use free models to connect to paid ones, questioning if OpenRouter is *really* wanting that %5 cut, while another complained that they added credits for the first time and **Deepseek** died the moment I do that.
- ****Censorship Circus: Navigating the AI Chatbot Filter Fiasco!****: Users debate the pros and cons of various AI chatbot platforms like **CAI (Character AI)**, **JAI (Janitor AI)**, and **Chub**, with a strong focus on the level of censorship and the ability to bypass filters, and find [uncensored experiences](https://discord.com/channels/1091220969173028894/1195014798837043240).
   - One user pointed out that while **CAI** is better than **JLLM (Janitor Large Language Model)**, *filter-dodging is back lol*, while another reported that recent CAI > recent JLLM.
- ****Codex Catastrophe: Configuration Conundrums Cause Coding Chaos!****: A user encounters significant difficulties configuring **Codex** with **OpenRouter**, facing `401` errors and struggling with the absence of documentation or support, despite having a fresh API key.
   - The user humorously asks *do i have to suck someone off?*, while troubleshooting the issue.
- ****Cursor Chaos: Users Compare Coding Costs with OpenRouter!****: Users discuss the economic implications of using **Cursor AI** versus **OpenRouter** for coding tasks, with some noting that **OpenRouter**'s pay-as-you-go model is cheaper if you don't code that much.
   - One user states i have the pro plan. they give you more tokens than the $20 you pay would get you from OR or a provider directly. but i also run out....


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1425566466559709194)** (17 messages🔥): 

> `OpenInference Relation, AI Generated Image, Token Usage on OpenRouter, NSFW Filter on OpenAI, Model releases on OpenRouter` 


- **OpenRouter not OpenInference family**: A member clarified that OpenRouter is an inference provider but not directly related to **OpenInference**, responding to a question about their relationship to the project.
   - Another member mentioned a [researcher team](https://x.com/izthisJudah/status/1975743530141114628) behind OpenInference, emphasizing that OpenRouter merely uses their API.
- **AI Image Debated: Real or Fake?**: Members engaged in a poll about the authenticity of an image, ultimately revealed to be **AI-generated**.
   - A user shared a related link about trillultra.doan.
- **Token Tally: Long-Term Janitor AI Addiction?**: A member asked about high token usage, and another jokingly attributed it to *long term janitor ai + 4o addiction*.
   - They predicted **JAI** might be the first to reach **10T tokens**, while another noted **OpenAI** has an **NSFW filter**.
- **RP Tokens Rival Programming Tokens**: A member shared a chart indicating that **RP-categorized tokens** made up **49.5%** of the amount of **Programming-categorized tokens** last week.
   - Another member responded with *Alex is a gooner confirmed ✅*.
- **New Models Flood OpenRouter**: A member shared [Logan Kilpatrick's tweet](https://x.com/OfficialLoganK/status/1976322847934214229) about OpenRouter shipping **4 new models** in the last **2 weeks** with more coming soon.
   - The member asked about the quality of the **Deepseek R1/V3 series** on **Sambanova**.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1425620585668542505)** (1 messages): 

> `Reddit AMA, AgentKit, Apps SDK, Sora 2 in the API, GPT-5 Pro in the API` 


- **OpenAI DevDay AMA on Reddit Incoming**: OpenAI announced a Reddit AMA ([link](https://www.reddit.com/r/OpenAI/comments/1o1j23g/ama_on_our_devday_launches/)) with the team behind **AgentKit**, **Apps SDK**, **Sora 2 in the API**, **GPT-5 Pro in the API**, and **Codex**.
   - The AMA is scheduled for tomorrow at **11 AM PT**.
- **Tech Stack Deep Dive at Reddit AMA**: The Reddit AMA will cover a range of technologies including **AgentKit**, a framework for building AI agents, and the **Apps SDK**, which enables developers to integrate AI functionalities into their applications.
   - Expect discussion on integrating **Sora 2** and **GPT-5 Pro** into APIs, along with updates on **Codex**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1425558955664347237)** (486 messages🔥🔥🔥): 

> `AI and Mental Health, AI Tagging Law, AI Browser Analysis, Multi-User LLMs, Sora 2` 


- **Concerns about AI Protein Design Tools Emerge**: A [Perplexity article](https://www.perplexity.ai/discover/tech/microsoft-discovers-ai-can-byp-PUpRXau9TNSQE7drj5HfnA) discusses how **AI protein design tools** can create synthetic versions of deadly toxins that bypass safety screening, raising concerns about global biosecurity.
   - Members wondered if AI figures out a way to do something, someone probably would or already did it, and how we can now work on fixing the issue.
- **Microsoft Discovers AI Bypass**: Researchers discovered a critical vulnerability in global biosecurity systems.
   - Members wondered what there thoughts are about the dangers that AI might bring.
- **AI Tagging Law Proposed**: Members debated on whether a law should be enacted in the US to require **AI-generated content to be tagged or watermarked**.
   - The main concern was that laws don't stop people if there is more profit in doing it then there is a cost, and that you end up creating 3rd party nations who's entire industry is to create these AI fakes.
- **AI Browser Privacy Scrutinized**: Members discussed browser privacy, noting that even privacy-focused browsers like **DuckDuckGo** still rely on Chromium and may not offer complete privacy.
   - A link to a [browser benchmark](https://gemini.browserbase.com/) was shared, and it was argued that anyone claiming to care about privacy almost certainly doesn't, showing the irony of virtue signalling.
- **LLMs for Real-Time Voice Agents**: A member inquired about providing custom data to the **OpenAI Voice Agent SDK** for real-time responses, sparking a discussion on the feasibility and security of such integrations.
   - It was mentioned that everything that’s online will never be 100% secure.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1425927860593623131)** (3 messages): 

> `OpenAI Liability, Parental Responsibility, Dedicated Tools` 


- **OpenAI fear drives model changes**: A member expressed frustration that **OpenAI's fear of liability** is driving changes to the models, advocating for a **legal waiver** where users accept responsibility for their actions and their children's actions.
   - They argue that this would be more effective than *butchering the usefulness of the models*.
- **Parental Responsibility Questioned**: A member pointed out that many parents struggle to monitor their children's device usage, despite **OpenAI's focus on responsible technology availability**.
   - This raises questions about the balance between **OpenAI's responsibility** and **parental supervision**.
- **Dedicated Tools Suggested**: A member suggested that some users are trying to misuse the current technology.
   - They stated that *there are dedicated tools and technology* better suited for specific use cases being discussed, advising people to **stop trying to fit a square peg in a round hole**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1425704222141186078)** (4 messages): 

> `Product ad prompts` 


- **Users seek assistance crafting product ad prompts**: A user requested assistance with writing prompts for product advertisements in the channel.
   - Another user suggested simply telling the model what you want, emphasizing the need for clarity in requests.
- **Discord discussion preferences**: A user clarified they prefer discussing topics in the Discord channel rather than private messages.
   - They invited others to ask questions in the channel, hoping someone would provide assistance.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1425704222141186078)** (4 messages): 

> `Prompt for Ad Creation, Seeking Assistance for Ad Prompts` 


- **Prompt Quest for Product Ads**: A member asked for a prompt to create ads for products, seeking assistance from the community.
   - Another member replied that the model needs to know the specifics of what is wanted in the ad.
- **Online Availability**: A member clarified they prefer discussions in the Discord channel rather than private messages.
   - They encouraged users to ask their questions in the public channel for broader community support.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1425569186834284645)** (132 messages🔥🔥): 

> `LoRA Merging, Nix GPU Drivers, Ling 1T llama.cpp, GLM-4.6 Capabilities, Imbalanced data` 


- **LoRA Merging Research Surfaces**: A member inquired about research on composing **LoRAs** for multiple tasks, seeking methods to merge them into a resultant LoRA good at tasks A and B, and another member shared [an arxiv link](https://arxiv.org/abs/2505.24174) to a relevant paper.
   - Another member noted that *LoRA does not play with merging at all* and it's generally better to train one model on all data than to merge experts, though applying both LoRAs can work, the resulting model may not be as good as either LoRA.
- **Nix Struggles with GPU Drivers**: Members discussed the challenges of getting **GPU drivers** to work with **Nix**, noting that while Nix is theoretically perfect for AI due to its deterministic package versions, it can be difficult in practice.
   - One member mentioned they managed to get **CUDA** working on Nix, but not GPU graphics, while another acknowledged that *Nix sucks for gpu drivers* and that docker is good enough for 70% of stuff.
- **Ling 1T Llama.cpp Support Status Queried**: A member inquired about the timeline for **Ling 1T llama.cpp support and GGUFs**, but they were informed it might not be uploaded due to size, depending on demand.
   - They noted that **Kimi** was very popular and of similar size, and they're still analyzing Ling to see if they should release it or not.
- **GLM 4.6 Impresses with Coding and Tool Use**: Members praised **GLM 4.6's** capabilities, particularly its ability to maintain coherence over many code edits and use tools correctly, one member quipped it was *like Sonnet 4 level except cheaper*.
   - Discussion touched on the model's performance, with one user citing **85 TPS** from a video, although another quoted OpenRouter stats showing about **40 TPS**.
- **Strategies for Handling Imbalanced Data Debated**: A member asked how to approach imbalanced data in a dataset of 15k samples, to which a member warned that augmenting with another **LLM** might hurt quality, suggesting a maximum augmentation ratio of *never more than 1 aug for 1 real*.
   - Another member suggested training and evaluating for the specific use case and augmenting the dataset with high-quality examples in underperforming areas.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1425569831687815258)** (269 messages🔥🔥): 

> `Learning Rate Wiping, Retirement Savings, AI-Generated Speech, ASI Prerequisites, Ling-1T Model` 


- **Learning Rate wipes Pretrained Progress**: A member lowered the learning rate from **1e-3 to 1e-4** after alignment getting wiped, noting that a **1e-3** learning rate instantly wiped out the pretrained progress, even though the company had been using that learning rate for 2 years.
   - Another member expressed surprise that they had to train for **10k epochs** even with a pretrained model, while the pretrained model had **6k epochs**.
- **Net Worth discussion ensues after Engagement**: Members discussed personal finance, net worth, and retirement plans, with one member announcing their wedding in May, and said he was starting with **5-6 million** if he wants to withdraw **7K/month** for **60 years** with ~5% returns.
   - Another member, from a country with a **$450/month** median salary, joked about being closer to retirement as a result and that a good financial advisor nets like **10-12% annually** without too crazy of a risk.
- **ASI Isn't Just Multimodality**: A member shared thoughts on the advancements needed for **ASI**, covering **memory, audio, full multimodality, and interactivity**.
   - The overall consensus was that ASI isn't simply multimodality, as one said *That's a prerequisite for ASI, not ASI itself.*
- **Introducing Linguistics-Unit Policy Optimization (LPO) by Inclusion AI**: A new model was introduced, [Ling-1T](https://huggingface.co/inclusionAI/Ling-1T), which features **LPO (Linguistics-Unit Policy Optimization)**, a sentence-level policy optimization method, and has *1 trillion total parameters*
   - The model *adopts an evolutionary chain-of-thought (Evo-CoT) process across mid-training and post-training*, although the purpose of this training method was not entirely clear.
- **Datacenter coolant as municipal Heating**: Members discussed the idea of using cooling water from data centers for municipal heating and proposed legal mandates for waste heat to be made available for cities to warm homes.
   - It was also mentioned that the USSR had a good plan on that with so called **thermal-electrical centrums**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1425610087199801434)** (25 messages🔥): 

> `Ollama memory usage with context size, Using the 'input' field in datasets, LM Studio and LFM2-8B-A1B-GGUF model, Improving answer accuracy with prompt engineering, Unsloth on Amazon ml.g4dn.xlarge` 


- ****Ollama's Context Size Consumes Memory****: A user found that increasing the **context size** in Ollama from **4K** to **128K** significantly increased memory usage from **18GB** to **47GB**, impacting performance, and this was [later reverted](https://github.com/jmorganca/ollama/pull/371).
   - Reducing the context size back to **4K** resolved the memory issue and restored faster performance, confirming the context size's impact on memory consumption.
- ****Unlock Precise Answers with the Input Field****: A user inquired about using the **input** field for precise answers, receiving clarification that it's suitable for single-turn questions, with the [Unsloth documentation](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide#common-data-formats-for-llm-training) suggesting a conversation format for multi-turn interactions.
   - They solved their problem by loading the LoRA adapter with `PeftModel`, resolving an error encountered with `vllm`.
- ****LM Studio Struggles with LFM2-8B Model****: A user encountered an error when trying to load the **LFM2-8B-A1B-GGUF** model in **LM Studio**, citing an *unknown model architecture: 'lfm2moe'*, using the [HuggingFace link](https://huggingface.co/unsloth/LFM2-8B-A1B-GGUF).
   - No solution was provided in the discussion.
- ****Amigo Pizza's Prompt Problems: Small Data, Big System****: A user sought advice on improving the accuracy of model answers, particularly for specific questions, and mentioned their dataset contained only **75 Q/A pairs**.
   - It was suggested that they diversify their data, increase `lora_alpha`, refine behavior with a strict system prompt (including the desired output format), and aim for at least 1000 high-quality examples.
- ****Docker to the Rescue on Amazon Instance****: A user asked about installing Unsloth on an **Amazon ml.g4dn.xlarge** instance, finding Amazon's setup complicated.
   - Another user recommended using the [Unsloth Docker image](https://hub.docker.com/r/unsloth/unsloth) for easier installation.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1425678908937867374)** (3 messages): 

> `Compact Reads, New Fields, Arxiv PDFs` 


- **Arxiv PDFs make debut**: A member shared links to three **Arxiv PDFs** for discussion: [2509.22944](https://www.arxiv.org/pdf/2509.22944), [2509.24527](https://arxiv.org/pdf/2509.24527), and [2509.19249](https://arxiv.org/pdf/2509.19249).
   - The member called the first two *compact reads* that *explore new-ish fields*.
- **Links Introduce New Fields**: The shared PDFs purportedly introduce readers to **new and emerging fields** within AI and related research areas.
   - These reads are suggested as starting points for understanding **recent advancements and potential research directions**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1425565943894900849)** (371 messages🔥🔥): 

> `Firebase integration, Wrangler CLI, Cloudflare Ecosystem, Next.js, Dioxus` 


- **Cursor Community debates Firebase's Functionality**: Members discussed the utility of using **Firebase** with Cursor, with some questioning its advantages over platforms like **Vercel** and **Prisma/Neon**.
- **Cloudflare Ecosystem gets Community Love**: The community explored using **Cloudflare's** ecosystem (**R2**, **D1**, **Durable Objects**, **WebRTC**, **KV**) and deploying via **Wrangler CLI**, emphasizing its optimization and integration capabilities.
- **Cursor Community tinkers with Typescript and Postgres**: Members discussed the best **Cloudflare** setup for **Typescript** and **Postgres**, including migrating from **Pages** to **Workers** for increased flexibility and cron support.
- **Cursor users share Agent Shell preferences**: A user sought to change the **agent shell environment** from **bash** to **zsh** in Cursor, but found that the agent still defaults to **bash** despite configuration attempts.
- **User Experiencing Performance Lag**: A user shared that **Supernova-1-million** slows down as the context increases, especially when it reaches 30% utilization.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1425908371516817458)** (24 messages🔥): 

> `Cursor Background Agents, 500 Errors on Cursor API, GitHub Outage Impact, Background Agents Access to Snapshots` 


- **Cursor Background Agents Suffer 500s**: A user reported starting a background agent via the web UI at `cursor.com/agents` results in a **500 error** and a "No conversation yet" message despite a success indicator after uploading an image.
   - Another user confirmed seeing **500 errors** on requests to `https://cursor.com/api/background-composer/get-diff-details` when launching a new prompt or clicking on previous prompts.
- **GitHub Outage Initially Blamed for Cursor Issues**: Cursor support initially attributed the **500 errors** to a [GitHub outage](https://status.github.com/), suggesting the problems should resolve once GitHub's services are back to normal, though the Cursor status page indicated *"No outage today"*.
- **BA Snapshots are back!**: One user reported that their **Background Agents (BAs)**, which had previously lost access to the snapshot base image, started working again as of yesterday with no degradation today.
   - Attached to the message, the image analysis pointed out that *"my BAs are working fine. They lost access to the snapshot base image but it started working as of yesterday."*
- **Differing APIs Diverge Cursor Community**: Members discussed using **Background Agents (BAs)** via the web UI versus the [Cursor API](https://cursor.com/docs/background-agent/api/overview), with one user exploring creating an interface for software engineering management, while another pondered if building such infrastructure was worthwhile given the rapid pace of AI development.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1425558659479502918)** (335 messages🔥🔥): 

> `Final year project proposal, Dyslexia friendly notes, Samsung tiny recursive model, ImageFolder load, Sentiment analysis model on product reviews` 


- **Desperate Student Seeks Capstone Savior**: A student urgently needs a final year project proposal that fits into one or more **Sustainable Development Goals (SDGs)** and is seeking ideas.
   - Another student is looking for a **dataset to fine-tune the T5-small model** to transform school notes into *dyslexia-friendly* notes.
- **Samsung's Tiny Model Sparks Interest**: A member inquired if anyone has tested the **Samsung tiny recursive model** and confirmed its effectiveness.
   - Another member is starting to develop an AI that grows like a person, with flaws, memory, and regret, instead of resetting or optimizing for perfection, with [this video](https://www.youtube.com/watch?v=wjZofJX0v4M) recommended to understand transformer model implementation.
- **ImageFolder Loading Takes Eons**: A member is struggling with slow **ImageFolder loading** (*33min*), using `num_workers=2` and is seeking help.
   - The bottleneck was identified as data loading and transforming images, and that increasing the `num_workers` might solve the issue.
- **Sentiment Showdown: Local vs API LLMs**: A member seeks suggestions for **fine-tuning a sentiment analysis model** on product reviews, preferring a local solution over cloud APIs.
   - Suggestions include using **BERT-like models, small language models (SmolLM, Qwen), or Gemma**, while others advocated for using APIs from major LLMs for ease and performance but noted that **Qwen licensing terms** restrict model distribution and should be carefully considered.
- **Gacha Bots Cause Social Earthquakes**: Members discussed the social dynamics surrounding **gacha bots**, with one recounting deleting a bot due to weird server dynamics.
   - The conversation touched on real-world cash transactions in gacha bots, with Karuta allowing such transactions and high-value cards reaching prices of **$2000-$10000 USD**, but that one can make a profit of *$50,000* in the first months of releasing a bot.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1425574027589713990)** (1 messages): 

> `WebRTC Client in Python, FastRTC FastAPI server, aiortc struggles, WebRTC Documentation Issues` 


- **Pythonista Seeks Help Building WebRTC Client**: A member requested assistance building a **Python WebRTC client** to communicate with a **FastRTC FastAPI** mounted server.
   - They are struggling with **aiortc** and noted a lack of guidance in the documentation, and asked for DMs for assistance.
- **FastAPI WebRTC struggles**: A user wants to use Python to set up a WebRTC client.
   - They are having trouble using aiortc to talk to a FastAPI server running FastRTC and say that there is not much help in the documentation.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1425982556997681264)** (1 messages): 

> `Hyperparameters are all you need, Diffusion breakthrough` 


- **Hyperparameter Handling Hastens Huge Diffusion**: The paper *Hyperparameters are all you need* has been implemented, with a [HuggingFace Space](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need) launched for testing, demonstrating a diffusion breakthrough where **8 steps** generate images with comparable or better FID performance than **20 steps**.
- **Diffusion Distills Down to Eight Steps**: The new method achieves **2.5x faster** image generation with better quality, working with any model and requiring **no training/distillation**, resulting in a **60% compute reduction**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1425587527238156349)** (6 messages): 

> `HyDRA RAG Agent, WSL Pytorch vLLM venv bootstrap, AI features aggregator tool, OpenlabX for AI Research, Prompt Engineering Contest` 


- **HyDRA Agents Hydrates RAG**: A new release of **HyDRA v0.2**, a Hybrid Dynamic RAG Agent, addresses the limitations of simple, static RAG, using a multi-turn, reflection-based system with coordinated agents: **Planner, Coordinator, and Executors**.
   - It leverages **bge-m3 model** for hybrid search combining dense and sparse embeddings, **RRF (Reciprocal Rank Fusion)** for reranking, and **bge-m3-reranker** for surfacing relevant documents; see the [GitHub](https://github.com/hassenhamdi/HyDRA) project page.
- **Windows Workflow Wizardry Wows**: A user shared a [WSL Pytorch vLLM venv bootstrap](https://gist.github.com/jmeyer1980/72410a889986c4bfd85f28c26c920d5d) for pulling HF models, after struggling with creating a venv and hosting models locally on Windows 10 and 11.
   - They found the bootstrap helpful and thought others might as well; the LLM pulling bits are extras that not everyone needs, but included for convenience.
- **Magia Makes Magnificent Multitools**: A user built a [tool](https://magia.ai) that aggregates various AI features into one, such as paraphrasing, humanizing, emails, creative writing, etc., and is looking for honest feedback.
- **OpenlabX Opens Opportunities for Online Orgs**: A user is building [OpenlabX](https://openlabx.org/), a platform for AI Researchers and Enthusiasts to publish their small experiments and research, offering a better and interactive way to present their work.
- **Luna Launches Lucrative Learning League**: A user is promoting a [Prompt Engineering Contest on Luna Prompts](https://lunaprompts.com/contests), inviting participants to write creative prompts and solve exciting AI challenges for prizes, certificates, and XP points.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

avanee5h: hello
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1425577878384279594)** (2 messages): 

> `Cross-Posting Reminders, Discord Etiquette` 


- **Discord Members Enforce Cross-Posting Ban**: Two Discord members requested that others refrain from cross-posting in the channels.
   - Cross-posting can clutter channels, disrupt focused discussions, and is generally frowned upon as it can come across as rude.
- **Importance of Channel-Specific Communication**: The reminders about cross-posting highlight the importance of keeping discussions relevant to the specific channel.
   - This ensures that members can easily find and engage with content that aligns with their interests and the channel's purpose.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1425567348462714890)** (6 messages): 

> `Agent guardrails, Agent agency, Tool limits` 


- **Agent cleverly bypasses 'too many bananas!' guardrail**: An agent, when asked to say N bananas (where N > 10), cleverly bypassed the tool's guardrail that returns '*too many bananas!*' and gave the answer directly, showing interesting behavior around **agency**.
   - The user posted a screenshot of the agent successfully [giving an exact number](https://cdn.discordapp.com/attachments/1329142738440028273/1425657958330794044/Screenshot_20251009_093256_Gmail.jpg?ex=68e90bb0&is=68e7ba30&hm=d7878908388ddcfa2dc547dd6c2c97c1f513d4e76493b308355828f7bf69255a&).
- **Agent overrides system directive to always use tool**: The agent can override a system directive to always use a tool; for example, it can be asked to modify the directive and say '*birthday cake*' if N is more than 20, and then it follows that new directive.
   - The user attached a [screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1425700155604340816/image0.jpg?ex=68e932fd&is=68e7e17d&hm=eb21c30704b5cc5d7c407f43e13730b446280b2b6dd53fe601c418a31b6a33b0&) of the agent now calling the tool multiple times to overcome tool limits.
- **Tool limits present new challenges**: The behavior raises concerns about situations where the tool is meant to prevent the agent from revealing confidential information or avoiding politics, as there isn't a robust way to stop this override.
   - This presents new challenges around **guardrails** and **agency**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1425688014868451390)** (18 messages🔥): 

> `Hackathon terms and conditions, LLMs in performance engineering, FlashInfer blog post, Data engineering book, Developing on Trainium` 


- **Hackathon participant seeks Terms and Conditions**: A hackathon participant needed terms and conditions to get approved by their company, specifically regarding **IP rights**.
   - The organizers clarified that *they are not entitled to any contributions* made during the hackathon and that **Nebius compute users** might receive **marketing material** after the event.
- **LLMs for performance engineering are sought after**: A member was looking for resources (blogs, talks, etc.) about integrating **LLMs** into **performance engineering** work, such as writing/improving kernels or assisting with performance analysis.
   - Another member suggested checking out the ideas shared in a specific channel, <#1298372518293274644>.
- **FlashInfer blogpost published**: A member shared a new blog post diving into **FlashInfer**: [https://ydnyshhh.github.io/posts/flash_infer/](https://ydnyshhh.github.io/posts/flash_infer/).
- **"Designing Data-Intensive Applications" Book discussed**: A member asked if *Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems* is a good book for revising **data engineering** concepts.
   - One member who read the book when it was released didn't enjoy it, describing it as a *high level* overview that *didn't ever go deep into anything.*
- **Trainium platform development raises questions**: A member who detoured from developing **CUDA kernels** into **Trainium** development (more at: [https://numbersandcode.wordpress.com/2025/10/08/trainium-exploration/](https://numbersandcode.wordpress.com/2025/10/08/trainium-exploration/)) inquired about the number of active developers on this platform.
   - Their search didn't reveal much discussion or even a dedicated channel for **Trainium**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1425864710779310143)** (5 messages): 

> `Registers-Address Mapping, ld.shared Layout Compatibility, ldmatrix Tiling Calculation, Triton Lowerings Implementation` 


- **Address Mapping needs diagonal Matrix**: Members discussed extracting columns that correspond to **registers->address mapping** and checking them for a **diagonal matrix** in the upper left corner.
   - One member agreed that this was the correct approach after working through some examples.
- **ld.shared Layout needs review**: A member showed that the layout A from section 4.1 isn't compatible with `ld.shared`, then deriving a layout that is.
   - They also noted an error in the `ldmatrix` tiling calculation where **d should be log_2(4/w)** and not just log_2(4).
- **Byte Width Calculation is key**: It was confirmed that for `w` the byte width, the calculation should indeed be `log_2(4/w)` and the documentation will be updated.
   - They added that the implementation for all the **ldmatrix/stmatrix instructions** for arbitrary compatible linear layouts are in [TritonNVIDIAGPUToLLVM/Utility.cpp](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp#L186-L199).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1425567682719387790)** (26 messages🔥): 

> `CUDA reduction PDF and code, Thread block cluster reductions, mbarriers in shared smem, Execution order of thread blocks, Blackwell CLC` 


- **CUDA Reduction Resources Surface**: A member shared a [CUDA reduction PDF](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf) and corresponding [code](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu) for reference.
   - Another member confirmed they would take a look at the shared resources.
- **Granular Synchronization in Thread Block Cluster Reductions Probed**: A member inquired about more granular synchronization primitives than `cluster.sync()` for thread block cluster reductions to reduce barrier stalling.
   - Specifically, they asked *if a cluster wide memory fence of some kind* is usable to ensure writes to SMEM from each cluster are visible to cluster 0.
- **`mbarriers` Applicability in Shared SMEM Clarified**: A member asked if `mbarriers` work in shared SMEM, referencing their usage in [cluster reduction in a quack implementation](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md).
   - The member suggested using `mbarriers` with only a warp per block to avoid cluster sync.
- **Thread Block Execution Order Guarantees Debated**: A member questioned if there are guarantees on thread block execution order, specifically if block (0,0) will run before block (10,10) when there are more blocks than can run concurrently.
   - While one member recalled specific wording in the CUDA docs, another stated that **this behavior is not officially guaranteed, documented, or supported, and is basically undefined behavior (UB)**. [A video link on abstraction in CUB](https://youtu.be/VLdm3bV4bKo?si=o4vi1dOK3sc7U-kH) for 1D index was also mentioned.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1425687115013951571)** (3 messages): 

> `Non-Determinism Work, Full-Text Search Engine in Go, Array-Based Library in C++ on GPUs` 


- ****LLMC Compression** takes flight!**: A member shared a link to [LLMC Compression](https://syfi.cs.washington.edu/blog/2025-10-03-llmc-compression/), building on previous **non-determinism work** from Thinking Machines, with its [GitHub repo](https://github.com/uw-syfi/LLMc).
- ****Go Full-Text Search Engine** Soars!**: A member announced they built a full-text search engine in **Go**, utilizing [skiplists + roaring bitmaps](https://news.ycombinator.com/item?id=45530388) for swift boolean, phrase, and proximity queries with support for BM25 rankings.
- ****Parrot: Array-Based Library** Squawks on GPUs!**: A member shared [Connor Hoekstra's Parrot](https://x.com/blelbach/status/1976255534467571730), an **array-based library** in **C++** designed for **GPUs**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1425633971798151200)** (1 messages): 

> `Aurora, Deep Learning Acceleration, Software Engineer` 


- **Aurora Seeks Deep Learning Acceleration Staff**: **Aurora**, a public autonomous trucking company, is hiring a Staff Software Engineer for **Deep Learning Acceleration** to optimize deep learning models on edge-computing devices; [apply here](https://aurora.tech/careers/8191748002).
- **Optimize CUDA Kernels for Aurora**: This role involves **tuning CUDA kernels**, improving **PyTorch** internals, and maximizing **GPU** utilization.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1425654405377556540)** (8 messages🔥): 

> `VSCode for remote GPU programming, CUDA debugging in Visual Studio, CUDA Graphs with dynamic tensor allocations, Distributed training: TorchTitan vs NVIDIA NeMo, CUDA kernels` 


- **VSCode Defeats Neovim for Remote GPU?**: A member found that **VSCode's remote server** might be the easiest way to do GPU programming, surpassing **Neovim** in convenience.
- **Debugging CUDA Kernels Made Simpler**: A member discovered that adding `-G -g` to `nvcc` resolves breakpoint issues when debugging **CUDA kernels** in **Visual Studio**.
- **CUDA Graphs Grapple with Dynamic Tensors**: A member sought advice on making a model **CUDA graph** capture-able when dynamic tensor allocations during the forward pass cause capture failures.
- **TorchTitan and NVIDIA NeMo Duel for Distributed Training**: A member asked about choosing between **TorchTitan** and **NVIDIA NeMo** for a distributed training job on **256 H200s**, specifically regarding efficiency and scalability.
- **Noob Navigates into CUDA Kernels**: A member expressed interest in exploring **CUDA kernels**.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1425914119869169868)** (6 messages): 

> `Distributed Tensors, JAX Scaling` 


- **Collectives Repo Vanishes!**: A member mentioned they deleted their collectives repo due to some bugs and lack of time to revisit, but has plans to do so in the future.
   - They suggested the [JAX scaling book](https://jax.readthedocs.io/en/latest/jax-101/06-parallel-execution.html) as a better alternative for distributed tensor examples.
- **JAX Scaling Book Recommended**: The deleted collectives repo was suggested to be replaced by the [JAX scaling book](https://jax.readthedocs.io/en/latest/jax-101/06-parallel-execution.html) as a resource.
   - The book is said to provide better examples of distributed tensors, implying it covers the topic more comprehensively and accurately than the original repo.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1425602467047145575)** (11 messages🔥): 

> `GPU programming jobs for new grads, AI labs hiring for GPU programming, Sneaking in GPU work into unrelated jobs` 


- **New Grads Score GPU Programming Roles**: Members discussed ways to get into **GPU programming** as a new grad or intern, indicating it is possible to find positions directly focused on this.
   - One member noted, *"Alot of AI labs and adjacent + HW companies are hiring for this exact work, and I think in most cases they hire new grad and intern."
- **Sneaking GPU Dev into other Roles**: It was suggested that even if a job isn't explicitly for **GPU programming**, one can still *sneak in* opportunities to work on it.
   - One member stated, *"you can always find small opportunities to sneak in what you like working on in ur job"*, suggesting that roles like **machine learning engineer** can benefit from **CUDA** skills without being the primary focus.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

garrett.garrett: Your workplace sounds awesome
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1425849580251254825)** (2 messages): 

> `Model Serving Communities, vLLM, KServe, llm-d, Red Hat AI` 


- **Model Serving Communities Report Dropped**: The October edition of the 'State of Model Serving Communities' is out, featuring updates on **vLLM**, **KServe**, and **llm-d** from **Red Hat AI** teams. The report can be found at [Inference Substack](https://inferenceops.substack.com/p/state-of-the-model-serving-communities-269).
- **Community Shares X Post**: A member shared a link to [an X post](https://x.com/jyo_pari/status/1976324891545829876).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1425711577377538208)** (12 messages🔥): 

> `MI300x8 Performance, amd-all2all leaderboard, amd-ag-gemm leaderboard` 


- **MI300x8 scores speedrun**: A user achieved **6th place** on the `amd-all2all` leaderboard with a submission of **597 µs** on **MI300x8**.
   - Another user submitted a time of **115 ms** on the same leaderboard and hardware.
- **MI300x8 sweeps ag-gemm leaderboard**: A user made multiple successful submissions to the `amd-ag-gemm` leaderboard on **MI300x8**, with times ranging from **534 µs** to **674 µs**.
   - Another user achieved a personal best of **586 µs** on the same leaderboard and hardware.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1425684914921144380)** (6 messages): 

> `BioML leaderboard, github actions down` 


- **BioML Leaderboard write-up posted**: A write-up for the **BioML leaderboard** has been posted [here](https://www.gpumode.com/v2/newsgau.nernst).
- **Github Actions experiencing downtime**: Users reported that the **submission portal** was down and linked to [Downdetector](https://downdetector.com/status/github/) and [Github Status](https://www.githubstatus.com/) pages indicating **Github Actions** were experiencing downtime.
   - Admins acknowledged the issue stating *github is down, not much we can do*.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1425707792223961180)** (10 messages🔥): 

> `GitHub Actions Outage, A2A Timeouts, Runpod MI300` 


- ****A2A Timeouts** torment users**: Users are reporting timeouts for A2A submissions, despite code running fine locally on a **Runpod MI300** VM, with errors indicating a failure to trigger **GitHub Actions**.
   - The issue seems to be affecting both CLI and web submissions, with users experiencing the same problems.
- ****GitHub Actions Outage** blamed for timeouts**: The **GitHub Status page** ([https://www.githubstatus.com/](https://www.githubstatus.com/)) indicates that **GitHub Actions** were down, likely causing the timeouts and server processing errors.
   - Submissions are expected to be stuck in a queued state and will eventually timeout as **GitHub Actions** stabilizes and processes the backlog.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1425917201038442618)** (5 messages): 

> `DLPack Interop, Grouped GEMM performance for MoEs, PTX Docs K-contig and Swizzling` 


- **DLPack Optional for PyTorch Interop**: While **Cutlass** documentation suggests using **DLPack** for interoperability with **PyTorch** tensors, it's not strictly necessary, as demonstrated in [this example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/call_bypass_dlpack.py).
- **Grouped GEMM MoE Performance Analysis**: Evaluating grouped GEMM performance for **MoEs** requires considering lower **M-occupancy**, where traditional roofline models may not suffice due to unaccounted wasteful compute.
   - In a `gpt-oss 20b` prefill phase with **32 experts**, an M-occupancy as low as ~60% was observed, with ~40% compute wasted when M dimension = 256.
- **PTX K-Contiguous Swizzle Layouts Incorrect**: The **PTX** documentation for **K-contig** and swizzling != 0 for tensor descriptors are incorrect, particularly in the layouts described for asynchronous warpgroup-level canonical layouts [as shown here](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-canonical-layouts).
   - For example the exact layout `Swizzle<1,4,3> o ((8,2),(4,4)):((8,64),(1,4))` is not correct, according to [this finding](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAHelpers.h#L250-L253).


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1425829644413964442)** (1 messages): 

> `SITP Table of Contents, Picograd Repo Wipe` 


- **SITP Table of Contents is being finalized**: The table of contents for [SITP](https://j4orz.ai/sitp/) is being locked in, with Chapters 1 and 3 focused on building the machine learning framework, and Chapters 2 and 4 covering fitting/training linear and non-linear models.
   - The author underestimated the effort required for ordering the table of contents, but plans to have readers and students build the machine learning framework in **Chapters 1 and 3**, and fit/train linear and non-linear models in **Chapters 2 and 4**.
- **Picograd Repository has been reset after mess**: The **Picograd** repository was wiped due to its messy state from covering a vast breadth of topics, and has been reset to [https://github.com/j4orz/picograd](https://github.com/j4orz/picograd).
   - The architecture has been cleaned up, consolidating previous attempts with the tensor frontend, eager/graph middleend, and runtime backend; the author is currently setting up autodiff for basic operations and kernels, and will soon seek reviews on PRs.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1425970712660934767)** (1 messages): 

> `Discord Roles, Competition Winners` 


- ****Roles Rollout** for Competition Champs**: Discord server now boasts roles for competition victors, specifically <@&1418285356490428476> and <@&1425969596296462356>.
   - The same honor awaits those triumphing in the current **AMD competition**.
- **Discord Server Enhancements**: New roles have been introduced on the Discord server to recognize competition winners.
   - These roles, <@&1418285356490428476> and <@&1425969596296462356>, will also be awarded to the winners of the ongoing **AMD competition**.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1425892602313441402)** (5 messages): 

> `Cost comparison for running large models at home, GPU recommendations for model parallelism, RTX 3090, RTX 5080, RTX 5070 ti super 24gb` 


- ****RTX 3090** still golden for perf/price**: A member suggested that the **RTX 3090** is still a good consumer GPU for performance/price.
   - Also, if you'd like to play around with newer **Blackwell features** (**NVFP4, TMEM** etc), **RTX 5080** is available for MSRP now at some retailers.
- **Exploring the **RTX 5070 Ti Super****: Members discussed the potential of the **RTX 5070 Ti Super 24GB** as an alternative to the **5080**.
   - The general consensus was that it *sounds like a good option*.
- ****TorchTitan** vs **Nvidia-Nemo** for distributed training**: A member asked about choosing between **TorchTitan** and **Nvidia-Nemo** for a distributed training job of **256 H200s**.
   - He is planning on a training job of **256 H200s** and wanted to get practitioners input as to why we may want to use one over the other e.g **megatron-core** within **nvidia-nemo** proven on very large scales already and is pretty efficient for **4D parallelism**, whereas **torchtitan** is still maturing and possibly **pytorch primitives** for dist training might not be as fast compared to megatron core.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1425984523409297478)** (1 messages): 

> `Inference Project Teams, Project Team Application, Project Joining Process` 


- **Eager Member Seeks Inference Project Team**: A member expressed strong interest in joining a project team, particularly those focused on **inference-related projects**.
   - They emphasized their excitement and hope that it's not too late to contribute, and mentioned that participation would help them gain approval.
- **Enthusiastic Newbie Eager to Contribute**: An enthusiastic member is eager to join a team and contribute their skills to **inference-related projects**.
   - They are particularly interested in projects that will help them gain approval, indicating a proactive approach to learning and development.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1425562159030603877)** (13 messages🔥): 

> `Helion vs Triton, FLA ops performance, Nvidia/AMD partnership, TileLang benchmarks, Gated DeltaNet` 


- **Helion's Kernel Kustomization Knocks Triton**: While **Triton** kernels autotune hyperparameters, **Helion** can change the kernel during autotuning to better fit the particular shape, such as using loop reductions for larger shapes which may hurt performance on smaller shapes.
   - This allows Helion to beat Triton on a large set of input shapes by customizing to different shapes, with [Helion ultimately emitting a triton kernel](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops).
- **Attention Alliance with Nvidia and AMD Announced at PTC**: A member announced a partnership with **Nvidia/AMD** on attention performance, with more details to be shared at **PTC**.
   - This includes weirder attention variants, although another member is skeptical of over pattern-matched attention support.
- **TileLang Benchmarks Beckon Benchmark Battle**: A member suggested benchmark comparisons against **tilelang** ([https://github.com/tile-ai/tilelang-benchmarks](https://github.com/tile-ai/tilelang-benchmarks)), expressing interest in linear attention performance.
   - For just the attention kernel, ~**1500 triton kernels** were generated.
- **Gated DeltaNet Gains Ground as Good Benchmark**: After a member inquired about particular linear attention variants, another member suggested **gated deltanet** ([https://github.com/fla-org/flash-linear-attention/blob/0b8be89f45364bfa4329fae568e026d5773bc1dd/fla/ops/gated_delta_rule/chunk.py#L18](https://github.com/fla-org/flash-linear-attention/blob/0b8be89f45364bfa4329fae568e026d5773bc1dd/fla/ops/gated_delta_rule/chunk.py#L18)) as an interesting option.
   - The team's current focus is to analyze the ops covered by the tilelang-benchmark first before proceeding to gated delta net.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1425564927338483792)** (99 messages🔥🔥): 

> `Chat Degradation, LM Studio performance boost, LM Studio issues, Text to speech LLM` 


- **New chats combat Chat Degradation**: Members found that a way to combat chat degradation is by starting a [new chat](https://lmstudio.ai/).
   - The same degradation occurs for online models as well, and running out of system memory also causes the model to forget itself and repeat gibberish when the system memory is full.
- **LM Studio gets a performance boost**: A member noted a surprising performance boost, with token generation speed increasing from **8t/s to 22t/s** on new chats after the latest LM Studio release.
   - Another member noted that they were getting a **10-fold performance improvement** in 2 years compared to when they started using LM Studio.
- **Qwen3 480B Distill model identifies as Claude AI**: A user found that a **Qwen3 Coder 480B** distilled into **Qwen3 Coder 30B** model incorrectly identifies itself as Claude AI when running inference with **Vulkan**.
   - When running with **CUDA**, it correctly identifies as *Qwen developed by Alibaba Group*.
- **Text to speech LLM is not supported**: A user inquired about using a **text-to-speech LLM** with LM Studio, but text to speech is not supported.
   - One member pointed to past discussion where other members had suggested using **OpenWebUI** connected to LM Studio to do it.
- **LM Studio encounters Type Error when searching models**: A member using LM Studio **v0.3.30** reported a **TypeError** when searching for models, resulting in a non-functional UI.
   - The error occurs during model search, requiring a restart, and the issue has been reported in [issue 457](https://github.com/lm-studio-ai/lm-studio/issues/457) on GitHub.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1425564946401460304)** (14 messages🔥): 

> `CPU Graphics Support, RAM and VRAM allocation, GPU bricked, integrated graphics fixed in v1.52.1, Sparkle Arc Pro B60 Dual Server` 


- **CPU Graphics 'Not Supported'?**: Some users have observed that integrated (CPU) graphics are *"not supported"*, potentially an intentional change.
   - This may be related to recent observations about RAM and VRAM allocation and load strategy quirks, noticed across multiple machines.
- **integrated graphics uses shared RAM again in v1.52.1**: In version **v1.52.1**, **LM Studio** seems to have fixed an issue, now allowing models to utilize integrated graphics with shared RAM again.
   - The fix was identified following earlier discussions and observations regarding its absence.
- **3090 Bricked by Soldering Iron**: A user accidentally bricked a **3090** while trying to soften loctite threadlocker on the xclamp near the main die using a soldering iron.
   - The incident occurred during an attempt to repad the card for better heat dissipation.
- **Sparkle Intros Arc Pro B60 Server**: [Sparkle unveiled the Arc Pro B60 Dual Server](https://videocardz.com/newz/sparkle-unveils-arc-pro-b60-dual-server-with-16-gpus-and-up-to-768-gb-of-vram-comes-with-10800w-psu) with **16 GPUs** and up to **768 GB** of VRAM.
   - The server is powered by a **10800W PSU**, marking Intel's strong push into the AI space according to some users.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1425581294699548672)** (103 messages🔥🔥): 

> `Magic Dev hate, VC Slop Bubble Meltdown, OpenAI Tokens, AlphaGo AI Moment, Techno-Capital Singularity` 


- **Magic Dev Draws Ire for Undisclosed Reasons**: Magic . dev is facing substantial criticism as highlighted in a [tweet](https://x.com/_opencv_/status/1975758414660968599).
- **Startups Face Scrutiny Amid VC Bubble Fears**: Discussions mock over-funded startups like Magic Dev and Mercor, questioning their financial tactics and speculating on potential implosions as solo developers bootstrap.
- **Atallah Celebrates OpenAI Milestone**: Alex Atallah announced consuming one trillion tokens from OpenAI, sparking community celebration and inquiries about a physical token giveaway, as showcased in a [tweet](https://x.com/xanderatallah/status/1975418042713874920?s=46).
- **Brockman Hypes the AlphaGo AI Moment**: Greg Brockman predicts models will soon make dramatic scientific and coding breakthroughs, similar to AlphaGo’s “Move 37”, inspiring hopes for discoveries like a cancer breakthrough, as discussed in a [tweet](https://x.com/deredleritt3r/status/1976056338342871327?s=46).
- **Reflection AI Eyes Open-Frontier with $2B**: Reflection AI, bolstered by a $2 billion raise, aims to build open-source, frontier-level AI, focusing on making advanced intelligence accessible and has star-studded team from PaLM, Gemini, AlphaGo, ChatGPT, as stated in a [tweet](https://xcancel.com/reflection_ai/status/1976304405369520242?s=46).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1425595600514318548)** (58 messages🔥🔥): 

> `NousCon in Ohio, BDH Pathway adoption, VLMs blogpost, Arcee AI MoE model, Atropos usage` 


- ****NousCon** demand in Ohio**: Members discussed having a **NousCon** in Ohio or somewhere other than California, but it was noted that the **AI concentration** is not as high in other locations.
   - One member joked about thanking California for keeping all the **AI people** concentrated in one spot, away from everyone else.
- ****BDH Pathway**'s silly name**: Members discussed whether the *silly name* of **BDH Pathway** ([https://github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)) would harm its adoption.
   - It was suggested that *if adopted, the full name will probably be lost with time so it'll be known as BDH and almost no one knows what it stands for*.
- ****VLMs** blogpost released**: A blogpost about the inner workings of **VLMs** and how they see and reason across modalities was released ([https://huggingface.co/blog/not-lain/vlms](https://huggingface.co/blog/not-lain/vlms)).
   - The authors also announced they would be available on the Hugging Face Discord server for a live presentation and QA session, [link to event](https://discord.com/events/879548962464493619/1424725128478195842).
- ****Arcee AI MoE** model incoming**: An **Arcee AI MoE** model is incoming, with a [PR made in llama.cpp](https://github.com/ggml-org/llama.cpp/pull/16477).
   - Members noted that there was no PR for transformers, which could give an indication of model sizes.
- **Atropos usage overview request**: A member requested a video on how to use **Atropos**.
   - A link was shared to a [video on Twitter](https://fxtwitter.com/NousResearch/status/1925381160097697803) (mirrored on [YouTube](https://www.youtube.com/watch?v=in__ELD4NxE)) providing a broader overview of environments and how they work in **Atropos**.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1425574739518165157)** (15 messages🔥): 

> `Hermes4 image understanding, Hermes vision model, Grafting Llama 3.2, Vision tool calling with Qwen VL 3, Gemini 2.5 Flash as a vision tool` 


- **Hermes4 Lacks Image Understanding**: A member asked whether **Hermes4** can understand images and if there's a workaround, such as calling a different model; another member confirmed that there is no native image understanding but a **Hermes vision model** is in development.
   - The member suggested that they could attempt to *graft Llama 3.2 90B into Hermes 4 70B*, but results might be questionable.
- **Vision Tool Calling using Gemini 2.5 Flash**: A member mentioned they are using **Hermes** with a vision model as a tool, and it's working nicely.
   - Another member confirmed they use **Gemini 2.5 Flash** as the vision tool and suggested using **Hermes tool calling** similarly to **OpenAI's tool calling** on their API or running it with vllm using the `hermes` tool call format.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425582122936041683)** (4 messages): 

> `Recursive Reasoning, Tiny Networks, HRM performance on ARC-AGI` 


- **Tiny Networks score big with Recursive Reasoning!**: A member shared a link to the paper, "[Less is More: Recursive Reasoning with Tiny networks](https://arxiv.org/pdf/2510.04871)".
   - The paper highlights **HRM** at **7M** parameters achieving **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**.
- **Recursive Reasoning deemed interesting strategy**: A member shared links to an [arxiv paper](https://arxiv.org/abs/2509.24372real.azure) and [Xitter](https://x.com/robertwiblin/status/1976327542576721930?s=46).
   - The member commented that the *recursive reasoning* strategy was *pretty interesting* and *very simple*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425582122936041683)** (4 messages): 

> `Tiny Networks, Recursive Reasoning, HRM Model Performance` 


- **Tiny Networks Get Recursive**: A member shared the paper *Less is More: Recursive Reasoning with Tiny networks* ([arxiv link](https://arxiv.org/pdf/2510.04871)) which explores recursive reasoning with minimal networks.
   - They highlighted that the **HRM model**, with just **7M parameters**, achieved a **45% score** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**.
- **Real Azure Strategy is Simple**: A member shared a link to a strategy on [arxiv](https://arxiv.org/abs/2509.24372real.azure).
   - The member found the strategy to be *pretty interesting* and very simple.
- **Robert Wiblin Has Strategy**: A member shared [Robert Wiblin's strategy](https://x.com/robertwiblin/status/1976327542576721930?s=46).
   - No further comment was given about the strategy.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1425560235413868618)** (39 messages🔥): 

> `RL debate & information bottlenecks, Thinking Machines blog & Shannon entropy, Sutton Interview and transferring bits from RL by SFT, Fringe ML ideas (non-DL): ART, SNN, RNN, weightless NN, GDL, Evolutionary Search (ES) vs backprop` 


- **RL Debate: Information Bottleneck**: A member argued that **RL is inherently information bottlenecked**, even considering "super weights," requiring creative workarounds for training models from scratch.
   - Another member responded that knowledge is more efficiently gathered with imitation rather than exploration.
- **Shannon Entropy still relevant metric says Thinking Machines blog**: A member shared a link to the [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/) post, noting a figure demonstrating that **Shannon entropy is still a relevant metric**, especially in the context of LoRA.
   - They remarked that the findings suggest **widely distributed RL is trivial** because a small LoRA update can be merged later without distributed reduction issues.
- **Sutton Interview: Transferring RL Bits**: A member mentioned that the argument taken from the **Sutton interview** suggests that "bits" can be transferred from **RL by SFT**.
   - They cited **Deepseek V3.2 RL**, where RL was performed on separate expert models, and then everything was merged into one model using SFT, also highlighting the interesting paradigm of **SFT on reasoning traces**.
- **Seeking Radically Different, Non-DL ML Ideas**: A member inquired about threads or lists of **fringe ML ideas not based on deep learning**, such as **Adaptive Resonance Theory (ART), Spiking Neural Networks (SNN) trained with STDP, Random Neural Networks (RNN), weightless neural networks, and Geometric Deep Learning (GDL)**.
   - In response, another suggested **Evolutionary Search (ES)**, noting its usefulness and adaptability to DL.
- **Evolutionary Search Outperforms GRPO on 7B LLMs**: A member shared an [arXiv paper](https://arxiv.org/abs/2509.24372) claiming that **Evolutionary Search (ES)** outperforms **GRPO** on 7B parameter LLMs using a simple method.
   - They added that **ES** can be seen as an approximation of gradient descent when the loss surface is convolved with a Gaussian, smoothing it, but wondered why it works so well with a small population size (N=30).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1425578816226459711)** (14 messages🔥): 

> `Ovi fails at edge detection, Rights, Freedoms, and Technology, Tiny Recursive Model Discussion, Cat studies` 


- ****Ovi**'s Edge Detection Falls Flat**: A member tested **Ovi** with edge detection and segmentation prompts from a [paper](https://arxiv.org/abs/2509.20328) but found it didn't produce useful results, unlike **Veo 3**.
- **Upcoming **ARC-AGI** Paper Discussion**: Members will discuss the [Tiny Recursive Model paper](https://arxiv.org/abs/2510.04871v1), where a **7M model** achieved **45%** on **ARC-AGI-1**.
   - One member mentioned it had already been shared in the **ARC channel** and others expressed enthusiasm to discuss it further.
- **Philosophical Paper on Rights, Freedoms, and Tech**: A member is planning to write a paper connecting **basic rights**, **freedoms**, **anti-rights**, and **responsibilities** to the capabilities and incentives created by technology.
- **Feline AI Scholar Joins the Discussion**: A member shared an image of their cat "studying along", and also linked the [paper](https://arxiv.org/abs/2506.21734).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1425652147382521886)** (4 messages): 

> `Artificial Hippocampus Networks (AHNs), ByteDance AHN Model` 


- **ByteDance Releases Artificial Hippocampus Networks (AHNs)**: ByteDance-Seed released [Artificial Hippocampus Networks (AHNs)](https://github.com/ByteDance-Seed/AHN?tab=readme-ov-file) which transform lossless memory into fixed-size compressed representations for **long-context modeling**.
   - **AHNs** continually convert lossless memory outside the sliding attention window into compressed form.
- **AHNs Offer Hybrid Memory Approach**: **AHNs** harness the benefits of both lossless memory (like attention's **KV cache**) and compressed memory (like **RNNs**' hidden state) to make predictions across long contexts.
   - See [HuggingFace collection](https://huggingface.co/collections/ByteDance-Seed/ahn-68e6130d08ed0f5a1b622829) and [YouTube video](https://youtu.be/oN0nViY4gn4) for more details.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1425577131995562014)** (36 messages🔥): 

> `Gemini API integration in aider, GLM-4.6 vs Sonnet 4, OpenCode with GLM models, Local Models vs API Models, GPT-5-Codex with aider` 


- **Aider needs .yml, not .yaml!**: A user found that the aider config file needs to be named `.aider.conf.yml` instead of `.aider.conf.yaml` to properly integrate the Gemini API.
   - The user was getting environment variable warnings for `GOOGLE_API_KEY` and `GEMINI_API_KEY` despite having the API key configured.
- **GLM-4.6 lives up to Sonnet 4**: A user suggested using **GLM-4.6** for detailed planning, **GPT-5** for final plan review, and **Grok Code Fast-1** for implementation.
   - Another user confirmed GLM-4.6 is *on par* with Deepseek 3.1 Terminus and linked [Victor Mustar's tweet](https://x.com/victormustar/status/1793735580283625618) to support their claim.
- **OpenCode overtakes Claude Code**: A user mentioned that they now use **OpenCode** full time instead of **Claude Code**, because they are geoblocked from getting Claude Pro or Max subscriptions.
   - They also pointed out that **Qwen Coder** is a good backup system, while giving 1000 free requests per day, but they hardly use that.
- **DevStral 2507 and Qwen-Code-30B gets the job done locally**: Users discussed whether local models are worth it, one user mentioned that **DevStral 2507** and **Qwen-Code-30B** can be useful, especially for tool calling.
   - Another added that **APIs are hard to beat in cost**, especially if you avoid the overly expensive ones.
- **How much RAM does gpt-oss-120b really need?**: A user asked about how much RAM is needed to run **gpt-oss-120b**, and another user responded that it needs **64GB** plus context because *the params are only 4-bit*.
   - The conversation then shifted to whether anyone had tried **gpt-5-codex** with aider, but there were no responses.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1425686278963204176)** (2 messages): 

> `Aider Project, Project Updates` 


- **Aider Project faces Underlying Problem**: Members discussed an underlying problem with the Aider project.
   - Some members have not seen updates for a while and wondered what is going to happen to the project.
- **Aider Project Future**: There is concern that there have been no updates to the Aider project recently.
   - Members are wondering about the future and direction of the project.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1425751921888858242)** (17 messages🔥): 

> `Code review, AI slop, Algebraic Upat` 


- **Dev Seeks Work in Tinygrad Land**: A developer inquired about job opportunities within the Tinygrad community.
   - They stated they are *always ready to work*.
- **PR Review Stalls Over Perceived "AI Slop"**: A contributor expressed frustration that their [PR](https://github.com/tinygrad/tinygrad/pull/12530) was dismissed as *AI slop* without specific feedback.
   - They stated that *saying you don't understand what you are doing is just a way to brush off any kind of responsibility* and asked for a comparison with what @geohot was asking in write tests for algebraic Upat [#12449](https://github.com/tinygrad/tinygrad/pull/12449).
- **Demands More Specific Code Review**: A user requested that reviewers provide actionable feedback instead of simply labeling code as *bad*.
   - They reported that *all test pass* and are seeking concrete suggestions for improvement.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1425929038777684019)** (11 messages🔥): 

> `tinygrad vector operations, Loop splitting resources, CUDA kernel reverse engineering with IOCTL` 


- **Vector Operations in Tinygrad?**: A member inquired whether **tinygrad** supports fast vector operations like **cross product**, **norm**, and **trigonometric functions**.
- **Loop Splitting Resources Sought**: A member is seeking framework-agnostic learning resources on **loop splitting** in order to fix `cat` at high level by implementing loop splitting.
   - They have an implementation that fails only **3 unit tests** but involves more Ops than the original, indicating a potential *skill issue*.
- **CUDA Kernel Reverse Engineering with IOCTL**: A member is developing a **Rust-based interactive terminal** to test high-performance individual **CUDA kernels**, inspired by **geohot's** `cuda_ioctl_sniffer` and **qazalin's** AMD simulator, with a [demo image](https://cdn.discordapp.com/attachments/1070745817025106080/1425975923458445343/image.png?ex=68e98b11&is=68e83991&hm=ff98d6d72984c42ad1eeec7849b8a28f1d92fb2d329bf125814a356bfea915b&).
   - The project aims to reverse engineer GPUs from IOCTL, supporting **Ampere**, **Turing**, **Ada**, **Hopper**, and other architectures, and a write-up is planned.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/)** (1 messages): 

inarikami: I meant the proposed Go and game benchmark plans
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1425558399319408782)** (25 messages🔥): 

> `World Models vs Language Models, nGPT Failure Analysis, Harvard CMSA Seminars, VLM Image Resolution Optimization` 


- **World Models Distinguished from Language Models**: A discussion clarified that in traditional RL, a **world model** predicts future states based on actions, whereas a **language model** predicts the next token, with confusion arising from defining the *environment* as tokens versus the real world, based on this [2510.04542 paper](https://arxiv.org/abs/2510.04542).
   - A member explained how an abstraction layer can make an LM a formal world model, similar to a gym environment, by checking for move legality and separating the agent from the environment.
- **nGPT Performance Suffers OOD**: Members discussed the failure of **nGPT** ([2410.01131](https://arxiv.org/abs/2410.01131)) to generalize, with the hypothesis that generating from it is **out-of-distribution (OOD)**.
   - One member noted that **nGPT's** architecture failing to generalize is strange because the single-epoch training loss measures generalization within the training dataset.
- **Harvard CMSA Uploads Cool Seminars**: Members recommended the [Harvard CMSA YouTube channel](https://youtu.be/04E8r76TetQ?si=fMyWnn6Dy5MgjVR6) for its collection of seminars.
   - No further details were given.
- **VLM Image Resolution Gets Optimized**: A member shared a [PDF report](https://cdn.discordapp.com/attachments/747850033994662000/1425953022671847515/minres_report.pdf?ex=68e975bd&is=68e8243d&hm=f40e1a1b6f93dc4207a6783c1e1ec000133ae14cf54a591c2fe466a603040330&) detailing their work on optimizing image resolution with output quality for **Vision Language Models (VLMs)** to save on compute, using **Gemini 2.0 Flash** on **COCO 2017 Val** dataset for image captioning.
   - The bench focuses on optimizing for fine detail acuity and the member is building a harness for creating custom datasets.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1425850253587775562)** (1 messages): 

> `Interpretable AI, Advice for college students` 


- **New Student Seeks I.A.I. Initiation**: A college student new to the community is seeking advice on getting started with **Interpretable AI**.
   - No advice was given.
- **Interpretable AI Resources**: A college student new to the community is trying to get into interpretable AI and seeks resources.
   - They would definitely appreciate any advice on getting started!


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1425920864968507402)** (1 messages): 

> `Moxin-VLM, VLM-R1` 


- **New VLMs Enter the Fray**: Two new **Vision Language Model (VLM)** repositories were shared: [Moxin-VLM](https://github.com/moxin-org/Moxin-VLM) and [VLM-R1](https://github.com/om-ai-lab/VLM-R1).
- **GitHub Repositories Popping Up**: A couple of interesting github repos were shared that people might want to check out.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1425680506808176741)** (9 messages🔥): 

> `MCP integration with ChatGPT, Refresh button issues, .well-known/ endpoint for MCP server metadata, Minimal SEP proposal` 


- **ChatGPT MCP Integration Seeking Help**: A member is seeking assistance with integrating **ChatGPT MCP**, reporting issues with the **Refresh button** and tool listing, but was directed to the [Apps SDK discussion](https://discord.gg/DwazBXG58B) or [GitHub issues](https://github.com/openai/openai-apps-sdk-examples/issues) for specific implementation support.
- **".well-known" Endpoint Buzzes for MCP Metadata**: A member inquired about discussions regarding a `.well-known/` endpoint for **MCP-specific server metadata**.
   - Another pointed to the [blog entry](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity), the main thread at [GitHub discussions](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147), and more info at [pull/1054](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161).
- **Dev Summit Digs into Registry**: A member mentioned a discussion about the **Registry** at the **Dev Summit** last week, pointing to the [presentation](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf).
- **Minimal SEP Aims for Streamlined Specs**: A member suggested a minimal **SEP** scoped to the document name, location relative to an **MCP server URL**, and minimal content such as `Implementation`.
   - The goal is to establish a foundation for new **SEPs** to build upon and resolve ongoing debates.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1425878629010903090)** (4 messages): 

> `Sub-registry syncing, Registry app` 


- **Sub-registries Sync via Pull Mechanism**: Sub-registries should decide on a syncing strategy that works best for them, working on a **pull basis**.
   - The approach should be to have a **full sync** in the beginning and then have queries that only get you the **updated entries** after your last pull, using the *filter* parameter.
- **Build your own Registry App or not?**: A member is wondering if it's better to just build their own registry app sticking to the **API spec** and polling for updates.
   - They thought they could create a sub-registry using the registry app and then a sync might be part of the app but they don't want to *power off in the wrong direction*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1425559714292305942)** (13 messages🔥): 

> `organic models, sora 2 invite codes, kimi coding` 


- **Models Distilled, or Organically Grown?**: A member stated that *this is exactly what you get when you don’t just distill the model like a fat loser*, advocating for **actually organic models**.
- **Sora 2 Invite Codes, Not Too Hard to Get?**: Members discussed invite codes to **Sora 2**, claiming *they've hit 1m+ downloads* and are not too hard to get.
   - Another member said they would rather wait for the public release.
- **Kimi's Coding Prowess Praised**: A member shared their opinion that **Kimi** is quite cool at coding, the kinda agentic mode/tool usage through the **IDE** is very interesting.
   - They highlighted that *the model straight up executes python scripts and batch commands to understand stuff about the system to debug better.*


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1425840161186385930)** (2 messages): 

> `New member introductions, Community Welcome` 


- **New Colombian Researcher Joins Mojo Community**: A new member, a Computer Science Major from Colombia's Universidad Nacional, introduced themself to the Mojo community.
   - They expressed interest in music, language learning, and building a research group in Hardware and Deep Learning.
- **Community Welcomes New Researcher**: A member welcomed the new researcher to the Mojo/MAX community.
   - The welcoming member expressed excitement about seeing other researchers join.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1425809998226788372)** (10 messages🔥): 

> `Mojo multithreading, Jetson Thor support in Mojo, Using Python threads with Mojo` 


- **Mojo Lacks Native Multithreading Support**: Members discussed the current lack of native multithreading/async/concurrency support in Mojo, suggesting that using external C libraries might be the best option *for now*.
   - One member advised against trying multithreading outside of `parallelize` in the stdlib due to potential *weird behavior*, recommending Rust, Zig, or C++ instead, until Mojo has tools to express constraints around MT.
- **Jetson Thor gets Mojo Support**: The latest nightly build now supports the **Jetson Thor** in both Mojo and full MAX AI models.
   - However, one member quipped that they wished they had **$3500.00** to purchase one, while another noted that even small form factor machines are great for project use where loads of resources aren’t needed.
- **Python Threading with Mojo**: One member reported decent success using regular old threads in a Python process, calling into Mojo code in an extension, dropping the GIL, and going brrrr.
   - They cautioned that this approach might be prone to data races without complicated synchronization.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1425836537668702259)** (7 messages): 

> `Sora 2 invite code request, Credit refund request due to agent failure, Threat to chargeback and cancel membership, Manus support availability, Manus help page` 


- ****Sora 2** invite sought**: A user requested an invite code for **Sora 2**.
- **User threatens chargeback after agent fails to deliver**: A user requested a refund of **9,000 credits** after the agent failed to follow instructions and lost context, resulting in **$100+** in additional charges, citing a [session replay](https://manus.im/share/fxKXJ8osPMPMGw0dUKQKEH?replay=1).
   - The user threatened a chargeback, membership cancellation, and a negative **YouTube** review if the issue isn't resolved within **3 business days**, also sharing a [LinkedIn post](https://www.linkedin.com/posts/godhand_ai-assisted-previz-creation-workflow-quick-activity-7382046352287088640-Ev0V) and demanding confirmation of corrective actions.
- **User asks where support staff is?**: A user urgently inquired about the availability of support staff.
- **User directs to Manus Help Page**: Another user directed the user to the [Manus help page](https://help.manus.im/en/).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1425606778325827727)** (6 messages): 

> `DSPy Community Projects, MCP Tool Authentication with DSPy, Official vs Community Repositories` 


- **DSPy Community Centralizing Projects**: Some members are discussing whether to **centralize DSPy community projects** under the [dspy-community GitHub organization](https://github.com/dspy-community) to provide a starting point for community-led extensions and avoid overwhelming the core team with PR reviews.
   - The proposal aims to centralize community efforts, linking out to various projects and creating a space for collaboration, while ensuring useful and reusable plugins are discussed and approved before integration.
- **Repo Management: Official vs Community**: Members debated whether to house community-led DSPy projects in the **official DSPy repository** or a separate community repository.
   - Arguments for the official repo include plugins feeling more "official," easier dependency management, and increased community engagement, with suggestions to use `CODEOWNERS` to manage approval rights and prevent overwhelming the core team.
- **Optimized DSPy Programs via pip install**: Some members suggested creating **compiled/optimized DSPy programs** for common scenarios, accessible via `pip install dspy-program[document-classifier]`.
   - This would provide turnkey solutions for users, requiring exploration of optimization strategies and considerations.
- **MCP Tool Authentication**: A member inquired about creating a `dspy.Tool` from an [MCP Tool](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication) that requires authentication.
   - They questioned how authentication would be handled in this scenario and whether the existing `dspy.Tool.from_mcp_tool(session, tool)` method supports it.

