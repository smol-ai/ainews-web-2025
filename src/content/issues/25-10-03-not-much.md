---
id: MjAyNS0x
title: not much happened today
date: '2025-10-03T05:44:39.731046Z'
description: >-
  **Anthropic** announces a new CTO. Frontier coding agents see updates with
  **Claude Sonnet 4.5** showing strong cybersecurity and polished UX but
  trailing **GPT-5 Codex** in coding capability. **xAI Grok Code Fast** claims
  higher edit success at lower cost. **Google's Jules** coding agent launches a
  programmable API with CI/CD integration. **Qwen** clarifies its model taxonomy
  and API tiers. Vision/LM Arena rankings show a tight competition among
  **Claude Sonnet 4.5**, **Claude Opus 4.1**, **Gemini 2.5 Pro**, and OpenAI's
  latest models. In video generation, **Sora 2 Pro** leads App Store rankings
  with rapid iteration and a new creator ecosystem; early tests show it answers
  GPQA-style questions at 55% accuracy versus GPT-5's 72%. Video Arena adds new
  models like **Luma's Ray 3** and **Kling 2.5** for benchmarking. Multi-modal
  video+audio generation model **Ovi** (Veo-3-like) is released. Retrieval
  models include **ModernVBERT** from MIT with efficient image-text retrieval
  capabilities. *"Claude Sonnet 4.5 is basically the same as Opus 4.1 for
  coding"* and *"Jules is a programmable team member"* highlight key insights.
companies:
  - anthropic
  - x-ai
  - google
  - google-labs
  - openai
  - arena
  - epoch-ai
  - mit
  - luma
  - akhaliq
models:
  - claude-3-sonnet
  - claude-3-opus
  - gpt-5-codex
  - grok-4-fast
  - qwen-3-next
  - gemini-2.5-pro
  - sora-2-pro
  - ray-3
  - kling-2.5
  - veo-3
  - modernvbert
topics:
  - coding-agents
  - cybersecurity
  - api
  - model-taxonomy
  - model-ranking
  - video-generation
  - benchmarking
  - multi-modal-generation
  - retrieval
  - image-text-retrieval
people:
  - finbarrtimbers
  - gauravisnotme
  - justinlin610
  - billpeeb
  - apples_jimmy
  - akhaliq
---


**The calm before DevDay.**

> AI News for 10/2/2025-10/3/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (196 channels, and 10895 messages) for you. Estimated reading time saved (at 200wpm): 758 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

**gm, Anthropic** has a [new CTO](https://x.com/zeffmax/status/1973833211835974046?s=46).

---

# AI Twitter Recap

**Frontier coding agents and model standings (Claude 4.5, Grok Code Fast, Google’s Jules, Qwen naming, Arena leaderboard)**

- **Claude Sonnet 4.5 (hands-on)**: After ~30 hours with Claude Code, [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1973922679418974298) finds Sonnet 4.5 “basically the same as Opus 4.1” for coding—polished UX, strong, but not as capable as GPT-5 Codex; also notes [ChatGPT Team value per $ > Claude Max](https://twitter.com/finbarrtimbers/status/1973923264524398687). Anthropic highlights Sonnet 4.5’s cybersecurity strength (comparable/superior to Opus 4.1 on some tasks) and focus on defensive capabilities [@AnthropicAI](https://twitter.com/AnthropicAI/status/1974199155657748868), [follow-up](https://twitter.com/AnthropicAI/status/1974199158929305738).
- **xAI Grok Code Fast**: [@gauravisnotme](https://twitter.com/gauravisnotme/status/1974001009778115066) claims “higher diff edit success” than Claude 4.5 and GPT-5 Codex at lower cost—worth independent verification, but multiple users are benchmarking coding agents more on edit reliability than raw next-token metrics.
- **Google’s Jules coding agent goes programmable**: Week-long drop culminates in a public API to make Jules a “programmable team member” with tools and CI/CD integration [recap @julesagent](https://twitter.com/julesagent/status/1973898898067632212), [API launch](https://twitter.com/julesagent/status/1974178592683954252), [docs](https://twitter.com/julesagent/status/1974179029726159274). Additional thread from [@GoogleLabs](https://twitter.com/GoogleLabs/status/1974206675859984531) and product post from [@googledevs](https://twitter.com/googledevs/status/1974217899536474565).
- **Naming clarity from Qwen**: Useful taxonomy of Qwen model families (LLM, Coder, VL, Omni, Image), instruct vs thinking variants, API tiers (Max/Plus/Flash), date-suffixed minor refreshes, and why “Qwen3-Next” exists [@JustinLin610](https://twitter.com/JustinLin610/status/1973974975976808808).
- **Live rankings**: Vision/LM Arena shows an exceptionally tight top tier: four-way tie for #1 among Sonnet 4.5 (standard and 32k Thinking), Claude Opus 4.1, and Gemini 2.5 Pro; OpenAI models (4o-latest, 4.5 preview, 5 high, o3) sit within one rating point [@arena](https://twitter.com/arena/status/1974215622474293262), [follow-up](https://twitter.com/arena/status/1974215628757066077). OpenRouter notes Grok 4 Fast dominating German prompts/completions [@OpenRouterAI](https://twitter.com/OpenRouterAI/status/1974122770645864767).

**Video generation surge: Sora 2 Pro momentum, evaluation, and a broader model stack**

- **Sora 2 Pro adoption and capability signals**: Sora 2 is now #1 in the App Store; the team is rapidly iterating and shipping invites [@billpeeb](https://twitter.com/billpeeb/status/1974035563482116571). High-quality 15s clips are rolling out [@apples_jimmy](https://twitter.com/apples_jimmy/status/1973979773354586379). Early testing suggests Sora 2 can answer GPQA-style questions at ~55% on a small subset, vs GPT-5 at 72%; a plausible explanation is an LLM “prompt rewrite” layer before video generation [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1974172794012459296), [prompt-rewrite hypothesis](https://twitter.com/EpochAIResearch/status/1974172889676177682), [context](https://twitter.com/EpochAIResearch/status/1974172901567004762). The app is also driving a new creator ecosystem (e.g., watermark-removal workflows) [@angrypenguinPNG](https://twitter.com/angrypenguinPNG/status/1974144279955325191).
- **Ecosystem and benchmarking**: Video Arena added Luma’s Ray 3 and Ray HDR 3 for head-to-head, community-voted evals [@arena](https://twitter.com/arena/status/1974161623935037658). Kling 2.5 is demonstrating excellent frame matching in spliced edits [@heyglif](https://twitter.com/heyglif/status/1974195300240957445). Multi‑modal video+audio generation “Ovi” (Veo-3–like) released: 5s videos at 24 FPS up to 720×720, text or text+image conditioning [@_akhaliq](https://twitter.com/_akhaliq/status/1974181920092418128).

**Retrieval, VLMs, and perception models (ModernVBERT, Jina v3, RF-DETR, π0.5 robotics)**

- **ModernVBERT / ColModernVBERT (MIT)**: A small bidirectional ModernBERT encoder for image-text and document retrieval that matches ColPali on ViDoRe with ~10× fewer params (~250M). ColModernVBERT’s late-interaction dual-encoder variant reports +10.6 nDCG@5 and is positioned as a sub-linear retriever (not just a re-ranker) enabling billion-doc kNN [@pteiletche](https://twitter.com/pteiletche/status/1974023966936203646), [@mervenoyann](https://twitter.com/mervenoyann/status/1974027641033261106), [HF models](https://twitter.com/mervenoyann/status/1974027983342989583), [authors’ thread](https://twitter.com/ManuelFaysse/status/1974036787187028079), [framework note](https://twitter.com/lateinteraction/status/1974105498044743704).
- **Listwise reranking (Jina v3, 0.6B)**: A “last but not late” listwise reranker that concatenates query plus all candidate docs in one pass, extracting special-token embeddings for both doc and query and reporting SOTA on BEIR [@JinaAI_](https://twitter.com/JinaAI_/status/1974148565770338705), [input format](https://twitter.com/JinaAI_/status/1974148568563745012), [links](https://twitter.com/JinaAI_/status/1974148570711548213). Commentary: although branded “last interaction,” it’s effectively early, full-context listwise interaction with strong empirical results [@lateinteraction](https://twitter.com/lateinteraction/status/1974153399164862927).
- **Detection and segmentation**: Roboflow’s RF-DETR segmentation preview claims 3× faster and more accurate than YOLO11-L on COCO segmentation, with TensorRT 10.4 latency on T4 and strong DINOv3 backbone results (e.g., thin crack segmentation in 1 epoch) [@skalskip92](https://twitter.com/skalskip92/status/1974160476444766324), [latency](https://twitter.com/skalskip92/status/1974160481192747039), [notebook](https://twitter.com/skalskip92/status/1974160484799590789).
- **Open-source robotics baseline**: Physical Intelligence π0 and π0.5 are now on Hugging Face and fully ported to PyTorch/LeRobot, with an emphasis on cross-embodiment, multi-environment Vision‑Language‑Action training for open-world generalization [@ClementDelangue](https://twitter.com/ClementDelangue/status/1974094115743711702).

**Reasoning, RL, and verifiers (PPO/GRPO, RESTRAIN, ExGRPO, RLAD, TUMIX, CLUE, RoT)**

- **RL recipes and corrections**: Why PPO/GRPO work and potential ties to human perception [@ethayarajh](https://twitter.com/ethayarajh/status/1973901333557346803). DrGRPO authors reiterate removing response-length normalization (mean vs sum) to avoid subtle bias; Tinker’s impl offered as reference for unbiased losses [@zzlccc](https://twitter.com/zzlccc/status/1973960971296387278).
- **Label-free/self-driven RL**: RESTRAIN converts spurious majorities into self-penalized signals—uses all rollouts, offsets low-consistency advantages, and shows training- and test-time scaling gains (e.g., +11% avg over TTRL/ETMR on AIME/AMC/MATH500 with Llama3.1‑8B) [@jaseweston](https://twitter.com/jaseweston/status/1974000962219225271), [results](https://twitter.com/jaseweston/status/1974000970192544248), [ablations](https://twitter.com/jaseweston/status/1974000971757101219). ExGRPO proposes experience prioritization with a mixed-policy objective to stabilize training where on-policy fails [@papers_anon](https://twitter.com/papers_anon/status/1973945230526459951).
- **Abstractions and pretraining**: RLAD trains LLMs to discover reusable “reasoning abstractions” to guide exploration [@QuYuxiao](https://twitter.com/QuYuxiao/status/1974187714343034932), [alt](https://twitter.com/Anikait_Singh_/status/1974195667250864561). NVIDIA frames “Reinforcement as a Pretraining Objective” (RLP) to bridge supervised pretraining and RL [@_akhaliq](https://twitter.com/_akhaliq/status/1974190336256962812). Google’s TUMIX mixes 12–15 diverse tool-using agents (text/code/search), shares notes across rounds, and uses an LLM-judge to stop early—improving benchmark accuracy and cutting cost (e.g., Gemini 2.5 Pro HLE 34.1%) [@omarsar0](https://twitter.com/omarsar0/status/1974106927287447725).
- **Verification and retrieval of thoughts**: Tencent’s CLUE verifier uses clustering—no trained params—and reports higher verification accuracy than GPT-4o [@LiangZhenwen](https://twitter.com/LiangZhenwen/status/1973928150104223868). Retrieval-of-Thought reuses prior reasoning traces via a “thought graph” to reduce tokens up to 40%, speed inference by 82%, and cut costs by 59% without accuracy loss [@TheTuringPost](https://twitter.com/TheTuringPost/status/1974228574598205736).

**Efficiency, quantization, and infra (FP8, SINQ, MLX, CPU MoE, QAT, sampling, training control)**

- **FP8 training and quantization**: Ant Group’s Ling 2.0 open-sources an FP8-native mixed-precision MoE training stack (fine-grained scaling, FP8 Adam states, routing maps), reporting BF16-level accuracy with 30–60% throughput gains with MTP and strong wins even without MTP [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1974182694239285260). Red Hat releases FP8-quantized Qwen3‑VL‑235B‑A22B‑Instruct with ~50% disk/GPU memory reduction and >99.6% accuracy retention [@RedHat_AI](https://twitter.com/RedHat_AI/status/1973932224400798163). Huawei’s SINQ presents a calibration-free quant that maintains SOTA while slashing memory [@HuggingPapers](https://twitter.com/HuggingPapers/status/1973906002001936577).
- **Compute/platform notes**: MLX builds can massively outpace generic GGUF on Apple Silicon; one user reports 115 tok/s vs 47 tok/s on Granite 4 H Tiny at 4-bit [@JorgeConsulting](https://twitter.com/JorgeConsulting/status/1974168414391619672). Curiously high CPU throughput for MoE: ~21 tok/s for Qwen 30B/A3B on CPU, ~4 tok/s for Qwen 232B MoE [@Teknium1](https://twitter.com/Teknium1/status/1974039942751006816). Together’s Instant Clusters publish clear on-demand/reserved GPU pricing [@togethercompute](https://twitter.com/togethercompute/status/1974167802337730854).
- **Training mechanics and libraries**: QAT scaling laws insight from Apple’s Awni Hannun on choosing 8-bit vs 4-bit (or 2-bit) given fixed RAM/latency budgets [@awnihannun](https://twitter.com/awnihannun/status/1974245339512385784). Batch sampler sharding centralizes complex sampling (weighted/temperature/balanced) for consistency and efficiency across workers [@TheZachMueller](https://twitter.com/TheZachMueller/status/1974072997670736098). Hugging Face TRL reproduces “LoRA without regrets,” exposing higher-performance LoRA under a familiar API [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1974191312229577085). “Interactive Training” proposes human-in-the-loop LR tuning during runs—turning loss monitoring into controllable feedback [@yuntiandeng](https://twitter.com/yuntiandeng/status/1974127176778662339).

**Industry and research signals (Sakana x Daiwa, Terence Tao + GPT-5, xLSTM scaling laws, Comet)**

- **Fintech deployment**: Sakana AI signs a ~¥5B (~$34M) multi-year deal with Daiwa Securities to co-build a “Total Asset Consulting Platform” using Sakana’s models for research generation, market analysis, and portfolio construction [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1973935631354245286), [Bloomberg summary](https://twitter.com/SakanaAILabs/status/1974109165623853365).
- **Human+AI discovery**: Terence Tao publicly documents using GPT-5 + tool-use to search for counterexamples and heuristics in math—S. Bubeck flags it as a notable moment for HAI research workflows [@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1973977315572154383), [example thread](https://twitter.com/kevinweil/status/1974161952260624459).
- **Architectures**: xLSTMs report Pareto-dominating Transformers on cross-entropy under both fixed-FLOP and fixed-loss regimes, with downstream inference efficiency gains [@maxmbeck](https://twitter.com/maxmbeck/status/1974018534385598895), [@HochreiterSepp](https://twitter.com/HochreiterSepp/status/1974027057215472107).
- **Browser as AI surface**: Comet’s launch triggered outsized user enthusiasm and adoption, esp. on macOS and Windows; praised for design that feels familiar but augments with non-intrusive AI integration [@felixleezd](https://twitter.com/felixleezd/status/1973942012278935631), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1974093953499877510), [follow-up](https://twitter.com/AravSrinivas/status/1974094750308553191).

**Top tweets (by engagement)**

- Sora 2 watermark removal workflow going viral, showcasing creator tooling growth around the app [@angrypenguinPNG](https://twitter.com/angrypenguinPNG/status/1974144279955325191) (6.9k).
- OpenAI routes sensitive conversations to GPT-5 Instant for faster, more helpful support; visible model indicator remains [@OpenAI](https://twitter.com/OpenAI/status/1974234951928459450) (2.3k).
- Terence Tao’s public example of GPT-5–assisted math exploration [@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1973977315572154383) (4.3k).
- Sora 2 Pro high-quality 15s clips; app hits #1 with continued invite rollouts [@apples_jimmy](https://twitter.com/apples_jimmy/status/1973979773354586379), [@billpeeb](https://twitter.com/billpeeb/status/1974035563482116571) (0.8k, 1.6k).
- Claude Sonnet 4.5 coding review vs GPT-5 Codex and Opus 4.1 [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1973922679418974298) (0.6k).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. LLM Efficiency and Benchmarks: Huawei SINQ Quantization + GLM 4.6 Tool-Calling Performance

- [**Huawei Develop New LLM Quantization Method (SINQ) that's 30x Faster than AWQ and Beats Calibrated Methods Without Needing Any Calibration Data**](https://www.reddit.com/r/LocalLLaMA/comments/1nwkzq7/huawei_develop_new_llm_quantization_method_sinq/) (Activity: 335): **Huawei proposes SINQ, a post-training LLM quantization scheme that adds a per-matrix second-axis scale and a fast Sinkhorn–Knopp–inspired normalization to minimize a row/column variance imbalance proxy, yielding calibration-free quantization (SINQ) and a calibrated variant (A-SINQ). Reported results show** `~30×` **faster quantization-time vs. AWQ and improved 4-bit-and-below perplexity on models like Qwen3 and DeepSeek-V2.5; see the paper [PDF](https://arxiv.org/pdf/2509.22944) and notes that it is “calibration-free, layer-independent,” with code on GitHub. Crucially, the** `30×` **gain refers to quantization speed, not inference/dequantization throughput, and no implementation details are given for runtime format/compatibility with common stacks.** Commenters flag missing inference-time/dequantization benchmarks (important for large-batch throughput) and lack of guidance on how to run the quantized models in Transformers/llama.cpp, speculating outputs may be .safetensors. Others note two methods were introduced (A-SINQ requires calibration, SINQ does not) and criticize comparisons (SINQ vs. HQQ [blog](https://mobiusml.github.io/hqq_blog/)) as less relevant to commonly used baselines (AWQ, EXL2/3, MLX, GGUF), urging clearer claims and broader quality benchmarks.
    - A commenter dissected the paper’s claims: Huawei presents two methods—**A-SINQ** (requires calibration, compared against **AWQ**) and **SINQ** (no calibration, compared against **HQQ**). They emphasize the reported `30x` speedup is for the quantization process itself, not inference, and note missing head-to-head benchmarks on quality/runtime versus widely used methods like **AWQ**, **EXL2/EXL3**, **MLX**, or **GGUF**. They also point out **HQQ** isn’t broadly adopted despite showing comparable perplexity to AWQ with slight memory advantages ([blog](https://mobiusml.github.io/hqq_blog/)).
    - Another thread highlights that for large-batch inference the bottleneck often shifts from memory bandwidth to dequantization compute; thus dequantization speed/overhead is crucial for throughput. They caution that a `30x` faster quantization step doesn’t imply faster decoding or batching efficiency unless the dequantization math and kernels are cheaper, requesting benchmarks on dequant FLOPs/latency and effective tokens/s under batching.
    - An image-sourced read suggests the core technique is a simple pre-processing step that can be applied before almost any quantization algorithm, implying easy composability with existing pipelines ([diagram](https://preview.redd.it/7uof90n49wsf1.png?width=1640&format=png&auto=webp&s=5585b671237adc2e5cfefe05c9fd844480a5dfdd)). If true, inference kernels may remain unchanged, so runtime gains would depend on whether the pre-processing reduces dequant complexity or improves weight statistics rather than requiring new backends.
- [**GLM 4.6 IS A FUKING AMAZING MODEL AND NOBODY CAN TELL ME OTHERWISE**](https://www.reddit.com/r/LocalLLaMA/comments/1nx18ax/glm_46_is_a_fuking_amazing_model_and_nobody_can/) (Activity: 417): **OP reports month-long production use of GLM-4.5/4.6 (ZhipuAI) with consistently strong user feedback on agentic autonomy and notably high tool/function-call accuracy, outperforming alternatives they tried (e.g., Claude Sonnet, GPT variant, Grok Code). They recommend evaluating tool-use via the Berkeley Function-Calling Leaderboard [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html) and criticize the "Artificial Analysis" benchmark as misrepresentative of real-world performance.** Top comments concur that Artificial Analysis often inversely correlates with practical usability and can favor benchmaxed Phi-style models, while GLM performs well for agentic workloads; one asks whether the OP runs locally or via cloud API. Another commenter claims GLM-4.6 outperforms Sonnet 4/4.5 in their tests, calling it a win for ZhipuAI.
    - Several commenters argue that the **Artificial Analysis** leaderboard ([https://artificialanalysis.ai](https://artificialanalysis.ai/)) inversely tracks real-world usefulness, claiming it amplifies “benchmaxed” phi-style models that overfit synthetic tests. They note GLM 4.6 shines in agentic scenarios (tool use, multi-step planning), underscoring the gap between synthetic benchmarks and practical agent performance.
    - A user-reported head-to-head indicates **GLM 4.6** performing notably better than “Sonnet 4/4.5” on their tasks, suggesting stronger task execution in their evaluation, though no quantitative metrics were shared. This points to potential advantages of GLM 4.6 in certain real-world workloads despite mixed benchmark narratives.
    - Early testers report **long reasoning/thinking phases** on simple tasks with GLM 4.6, raising latency concerns. One tester is seeking ways to reduce the model’s “thinking length,” implying a need for API/runtime controls (e.g., stricter max tokens or reasoning-budget limits) if available from the provider; deployment mode (local vs cloud API) was also queried but not detailed.
- [**The most important AI paper of the decade. No debate**](https://www.reddit.com/r/LocalLLaMA/comments/1nwx1rx/the_most_important_ai_paper_of_the_decade_no/) (Activity: 1921): **The post is asserting that the image shows Vaswani et al. (2017), “Attention Is All You Need” ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)), the Transformer paper. Technically, it replaced recurrence/convolutions with self-attention, introduced multi-head attention and positional encodings, enabled fully parallel sequence training, achieved SOTA in machine translation, and became the foundation for BERT/GPT-scale LLMs.** Commenters contextualize its impact by citing prior key works: Mikolov et al. (2013) Word2Vec ([arXiv:1301.3781](https://arxiv.org/abs/1301.3781)) and Bahdanau et al. (2014) attention/NMT ([arXiv:1409.0473](https://arxiv.org/abs/1409.0473)), noting survivorship bias and that major breakthroughs build on earlier innovations; “most impactful” is debated versus dependence on prior work.
    - Attention predates Transformers: **Bahdanau, Cho, Bengio (2014)** introduced additive attention for NMT, learning soft alignments between source and target tokens to remove the fixed-length encoder bottleneck, a direct precursor to the Transformer’s scaled dot‑product attention ([paper](https://arxiv.org/abs/1409.0473)). This shifted sequence modeling from compress‑then‑decode to dynamic context retrieval, materially improving translation quality over vanilla encoder‑decoder RNNs and enabling longer‑range dependencies.
    - Foundational representation learning came from **Mikolov et al. (2013) Word2Vec** ([paper](https://arxiv.org/abs/1301.3781)), which proposed CBOW/Skip‑Gram with negative sampling and hierarchical softmax to efficiently learn dense word embeddings from large corpora. By replacing full‑vocabulary softmax with sampled objectives, it reduced training cost from `O(|V|)` to `O(k)` per update, producing linear‑semantic structure in vector space that later architectures (including Transformers) leveraged for pretraining and transfer.
    - For the 2010s, many argue **AlexNet (2012)** is the pivotal catalyst: trained on `2× GTX 580` GPUs with ReLUs, dropout, and local response normalization, it slashed ILSVRC‑2012 top‑5 error to `15.3%` vs ~`26.2%` prior SOTA, kickstarting large‑scale GPU deep learning ([paper](https://dl.acm.org/doi/10.1145/3065386)). This hardware‑software co‑design moment normalized GPU acceleration for neural nets and unlocked the scaling regimes later exploited by Transformers.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Sora 2 and Latest Text-to-Video Demo Reels

- [**I asked SORA 2 to create a 90s-style Toy Ad of Epstein's Island.**](https://www.reddit.com/r/singularity/comments/1nwz4cx/i_asked_sora_2_to_create_a_90sstyle_toy_ad_of/) (Activity: 701): **OP claims to have used OpenAI’s video model (referred to as “Sora 2”) to generate a 1990s-style toy commercial themed around a controversial real-world location, highlighting Sora’s period-accurate ad aesthetics and satirical composition capabilities. The linked video on [v.redd.it](http://v.redd.it/) is currently inaccessible (**`403 Forbidden`**), so content cannot be independently verified; reference model info: [OpenAI Sora](https://openai.com/sora). This post implicitly probes Sora’s safety/moderation boundaries by combining nostalgic ad tropes with sensitive subject matter.** Top comments are non-technical reactions: quips about the model being on a “no-fly list,” and praise that it’s “disturbing” yet “amazing,” implying perceived high fidelity and comedic impact while flagging ethical discomfort.
- [**Obligatory Test of Latest Text-to-Video Model: Eating Spaghetti**](https://www.reddit.com/r/singularity/comments/1nx6w1f/obligatory_test_of_latest_texttovideo_model/) (Activity: 491): **Post showcases the standard “eating spaghetti” text‑to‑video stress test on a “latest” model; the linked asset is currently inaccessible ([v.redd.it/znaochtxuxsf1](https://v.redd.it/znaochtxuxsf1) returns** `403`**), so only comment-based signals are available. From comments, identity fidelity for a Will Smith‑like subject remains weak (face likeness not preserved), while perceived temporal/action coherence appears improved versus 2023-era outputs (i.e., sustained multi-step actions).** Commenters note the “Will Smith spaghetti” test remains a de facto baseline; debate centers on poor identity resemblance despite apparent gains in sequencing and coherence.
    - Identity fidelity remains a weak point: multiple commenters note the poor resemblance to Will Smith, suggesting current text-to-video pipelines still struggle with robust face-ID conditioning and temporal face consistency. Beyond data/architecture limits, safety filters that avoid celebrity likeness can degrade identity accuracy, causing frame-to-frame drift and “off-model” faces. Practical workflows often require add-ons like ID-guidance (e.g., [ID-Adapter](https://arxiv.org/abs/2308.06767)), ControlNet-style conditioning ([ControlNet](https://arxiv.org/abs/2302.05543)), or post-process face tracking/roto for stability.
    - Audio integration is highlighted as a major step beyond 2023-era silent clips: the latest demos appear to include synchronized speech/SFX, implying either a joint AV diffusion/transformer stack or a TTS + alignment stage. This raises complexity around lip-sync, voice timbre consistency, and AV alignment across `N` frames; typical failure modes include prosody uncanny valley, viseme drift, and desync during fast motion or occlusions. Techniques like phoneme-to-viseme mapping and lip-sync correction (e.g., [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)) remain relevant when end-to-end AV generation falls short.
- [**Mr. Rogers at the Battle of Agincourt**](https://www.reddit.com/r/aivideo/comments/1nwzxfq/mr_rogers_at_the_battle_of_agincourt/) (Activity: 853): **The post appears to showcase a prompt‑driven generative‑AI piece placing [Fred Rogers](https://en.wikipedia.org/wiki/Fred_Rogers) at the [Battle of Agincourt (](https://en.wikipedia.org/wiki/Battle_of_Agincourt)**`1415`[**)](https://en.wikipedia.org/wiki/Battle_of_Agincourt). The linked Reddit URL returns** `HTTP 403 (Forbidden)` **without authentication, so the underlying media and technical metadata are not retrievable; the thread provides no explicit model/pipeline, parameters, or prompts, though a top comment asking *“What were the prompts?”* implies text‑to‑image/video synthesis (and possibly voice cloning) was used.** Commentary is largely non‑technical (expressions of amazement and interest, e.g., *“Wow this is wild”*), with the only actionable request being for the exact prompts; no benchmarks, model choices, or implementation details are discussed.
    - One commenter highlights that viewing medieval combat through an **’80s film camera** aesthetic makes it feel more real, implying that capture-era artifacts (film grain, lower dynamic range, color cast, and frame cadence) materially affect perceived authenticity of generated or reconstructed footage. For recreations, emulating analog characteristics such as `24 fps` cadence, grain, slight gate weave, and tape noise can reduce uncanny-valley effects better than simply increasing resolution or sharpness.

### 2. GPT-5 Thinking Wikipedia Audits and Research Assistance

- [**Noam Brown of OpenAI has been using GPT-5 Thinking to find errors in every Wikipedia page. Some of these errors can be quite serious. Even the Wikipedia page on Wikipedia has an error.**](https://www.reddit.com/r/singularity/comments/1nwl1kz/noam_brown_of_openai_has_been_using_gpt5_thinking/) (Activity: 714): **Post claims that OpenAI’s Noam Brown is using a forthcoming GPT‑5 “Thinking” mode to systematically scan Wikipedia for factual errors, highlighting examples (including on the [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia) article itself). Commenters note at least one showcased issue was already tagged with [citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed), indicating Wikipedia’s existing QA/maintenance workflows had flagged it, and caution that prompts like “find at least 1 error” can induce LLM hallucinations and false positives; rigorous verification and sourcing are needed. See original thread: [Reddit gallery link](https://www.reddit.com/gallery/1nwl1kz).** Debate centers on the prudence and credibility of Brown’s approach and prior takes on Wikipedia, and a broader critique that anti‑Wikipedia voices advocating LLM replacements favor more centralized, closed, and less transparent systems versus Wikipedia’s open editorial process.
    - Multiple commenters note the example sections already have [Citation needed] tags, signaling Wikipedia’s built-in QA is functioning; GPT flagging these does not demonstrate new error discovery. Prompting an LLM to “find at least 1 error” biases toward false positives (hallucinations) rather than truth-finding without external grounding. Relevant policies: [Verifiability](https://en.wikipedia.org/wiki/Wikipedia:Verifiability) and [Citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed).
    - A cited case involves GPT-5 purportedly confusing a kilocalorie figure with its reference, producing a claim of error where both the number and citation may be correct but mismatched. This is a **source-attribution mismatch** (reference alignment failure), a common LLM weakness when evidence linking isn’t enforced. The model also invokes the CDC as ground truth without verifying the cited CDC page supports the exact statement, highlighting weak evidence chaining.
    - Technically, replacing Wikipedia’s transparent, versioned, community-audited workflow with a **centralized, closed-source LLM** reduces provenance and reproducibility. Robust LLM-driven QA would need to report precision/recall against human adjudication, expose sources, and output verifiable diffs for editors; without this, model judgments are non-auditable and non-deterministic. In short, reliability hinges on grounded retrieval and measurable evaluation rather than model assertions.
- [**Terence Tao says ChatGPT helped him solve a MathOverflow problem and saved hours of manual coding**](https://www.reddit.com/r/singularity/comments/1nwqqrj/terence_tao_says_chatgpt_helped_him_solve_a/) (Activity: 1376): **Fields Medalist Terence Tao reports that [ChatGPT](https://chat.openai.com/) assisted on a [MathOverflow](https://mathoverflow.net/) problem by generating code that would have otherwise required “hours of manual coding,” per a [Reddit post](https://www.reddit.com/gallery/1nwqqrj) that currently returns** `HTTP 403` **without authentication. This is cited as a practical use of LLMs to accelerate exploratory math/programming workflows (e.g., quickly producing auxiliary scripts/boilerplate for computational checks), not as a substitute for formal proof.** Commentary emphasizes that effectiveness is a “skill issue” (prompting/tooling proficiency) and predicts skeptics will shift as real-world utility compounds; discussions are largely non-technical endorsements rather than debates on model capability limits.
    - A commenter infers from the shared chat logs’ "thinking duration" and interaction style that Tao and Aaronson likely used **GPT‑5 Thinking** on “medium/low,” not **GPT‑5 Pro** on “high.” The claim hinges on observed latency/compute budget indicative of lower thinking settings, suggesting even sub‑frontier or down‑budgeted tiers can meaningfully assist in advanced math workflows.
    - They cite eval results claiming the “high” version of **GPT‑5 Thinking** scored only `38%` on the 2025 IMO under a plain “best‑of‑32” sampling regime (not a **Gemini**‑style agentic scaffold), while an internal experimental model from ~3 months prior allegedly achieved “gold in one try.” The technical takeaway is that both sampling strategy (best‑of‑N versus agentic) and model variant/tier significantly affect Olympiad‑style benchmarks, complicating cross‑model comparisons.
    - Another observation: the model repeatedly refused to guess the user’s identity from the chat context, only providing a name after pressure and tagging it with "(low confidence)." This suggests conservative safety/policy layers around doxxing/identification and some degree of explicit uncertainty calibration in responses, which can affect how researchers probe model meta‑inference capabilities.

### 3. AI in Education: Teacher Adoption and Student Legal Cases

- [**Teacher doesn’t hide his use of AI.**](https://www.reddit.com/r/ChatGPT/comments/1nwpcbu/teacher_doesnt_hide_his_use_of_ai/) (Activity: 609): **Photo appears to show a teacher-provided exam/worksheet explicitly labeled as AI-generated (e.g., via ChatGPT), signaling transparent use of generative AI to draft classroom materials. The post contextualizes AI as a productivity tool for educators (test/lesson plan generation) rather than a means for students to bypass learning, aligning with pre-AI practices like using shared test banks or purchased materials.** Commenters generally approve of teachers using AI as a tool (contrasting with student misuse) and note it’s analogous to buying/borrowing curricula from marketplaces. Some imply disclosure and quality control are key when leveraging AI outputs for assessments.
- [**A 13-year-old student in Florida got arrested after asking ChatGPT a criminal question**](https://www.reddit.com/r/ChatGPT/comments/1nwpv3v/a_13yearold_student_in_florida_got_arrested_after/) (Activity: 864): **A 13-year-old Florida student was arrested after entering a criminal query into ChatGPT on a school-managed context; detection came via the school’s own monitoring system (not ChatGPT/OpenAI), which flagged and escalated the content. Comment summaries note that *“no intent was found”* and the student is *“awaiting legal proceedings”*, implying action driven by monitored logs rather than demonstrated mens rea.** Commenters emphasize that the trigger was school-run surveillance rather than the model vendor, and debate centers on proportionality—i.e., whether arrest is appropriate when *no intent was found*—and the breadth of K–12 device/account monitoring pipelines.
    - Detection originated from the school’s monitoring stack (**Gaggle**), not ChatGPT/OpenAI. Gaggle is typically deployed on school-managed accounts/devices to scan student content in real time and auto‑escalate high‑risk phrases to administrators/law enforcement ([gaggle.net](http://gaggle.net/)), matching the flow described (query -> alert -> police). Technically, this is client/network‑side telemetry, not provider‑side reporting.
    - Even though officials say "no intent was found," the automated alert still led to legal proceedings, illustrating how keyword-based threat detection can escalate regardless of intent. This reflects a low‑threshold, high‑severity policy where matches like "kill" trigger immediate action to minimize time‑to‑response, trading off context sensitivity and increasing false‑positive risk.
    - For Gaggle to flag a prompt typed into ChatGPT (e.g., "How to k*ll my friend in the middle of class?"), the system must have visibility via a managed Chromebook/endpoint agent, Chrome extension, or network proxy inspecting content on school accounts. Practically, queries to third‑party AI services are not private on school infrastructure; the pipeline is endpoint/proxy capture -> AI triage -> human review -> alert, rather than any "reporting" by ChatGPT itself.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Agentic Dev Tools: Comet, Solveit, Chrome DevTools MCP**

- **Comet Goes GA, Agents Go Parallel**: Perplexity rolled out the AI-first **Comet Browser** to everyone, free at [perplexity.ai/comet](https://www.perplexity.ai/comet), enabling parallel **agentic tasks** and exiting the waitlist globally.
    - Early adopters praised its speed and *"smarter"* search while flagging prompt-injection and platform gaps during the worldwide rollout ([Comet rollout post](https://xcancel.com/perplexity_ai/status/1973795224960032857)).
- **Solveit Ships, Solves 'AI Fatigue'**: **Jeremy Howard** announced the public release of **Solveit**, an AI-augmented dev platform used internally at [**Answer.AI**](http://answer.ai/), with a 5-week live course starting **Oct 20** ([Solveit announcement](https://xcancel.com/jeremyphoward/status/1973857739341508884)).
    - The program grants platform access and training, showcasing real workflows (sysadmin, app deployment, GUI dev, contract drafting) to tighten feedback loops and counter *"AI fatigue"*.
- **Chrome MCP Lands for DevTools**: A canonical **Chrome DevTools MCP** launched at [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp), giving agents standardized access to browser debugging and automation surfaces.
    - Users demonstrated it working with **claude-cli** in DeepSeek browser testing ([walkthrough](https://www.circusscientist.com/2025/10/03/deepseek-browser-testing-with-claude-cli-and-chrome-devtools-mcp/)), highlighting practical agent-tool integration.

**2. GPU Performance & Quantization Engineering**

- **TorchAO Taps TinyGemm for INT4**: **TorchAO** exposes **INT4 quantization** (INT4mm) using **TensorCore** kernels adapted from **tinygemm** ([Quick start](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start), [int4mm.cu](http://int4mm.cu/)), targeting high-throughput A100 deployments.
    - Contributors can follow the [quantization overview](https://docs.pytorch.org/ao/main/quantization_overview.html) and [guide for adding efficient kernels](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels) to extend INT4 paths and optimize operator coverage.
- **DeepSeek Sparse Attention in CUDA**: Engineers coordinated on implementing **DeepSeek’s sparse attention** in CUDA using [FlashMLA](https://github.com/deepseek-ai/FlashMLA) and **TileLang** [examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32).
    - The FlashMLA docs’ deep dives detail **partial RoPE**, FP8 sparse kernels, and Hopper specifics ([new-kernel deep dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md), [Hopper FP8 sparse deep dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md)).
- **KernelBench Crowns Baselines**: The **KernelBench** project systematizes GPU performance evaluation with **250** curated PyTorch ML workloads and introduces the speedup metric **fast_p** ([KernelBench overview](https://harvard-edge.github.io/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/)).
    - Even frontier reasoning models mostly fail to surpass PyTorch baselines, with best practices emphasizing clock locking and warmup runs for reproducible kernel timings.

**3. Datasets, Leaderboards, and Model Lineup Moves**

- **Claude Climbs: Leaderboard Logjam**: The **LMArena Text Leaderboard** now shows **Claude Sonnet 4.5** tied with **Claude Opus 4.1** for #1, with **IBM Granite H Small** and **ray-3** newly added ([Text Leaderboard](https://lmarena.ai/leaderboard/text)).
    - Community discussion focused on the parity at the top and the expanding roster that broadens head-to-head evaluations across new text and video models.
- **ArXiv Avalanche: 4.6TB on HF**: A massive **4.6TB arXiv** dataset of papers plus metadata across scientific domains landed on **Hugging Face Datasets** ([nick007x/arxiv-papers](https://huggingface.co/datasets/nick007x/arxiv-papers)).
    - The uploader also teased a pending corpus of **3M GitHub repos**, signaling expanding open corpora for pretraining and retrieval experiments.
- **Seed Savings: ByteDance LLM Value Play**: Members proposed adding **ByteDance Seed LLM (Seed 1.6)** to **OpenRouter**, citing low pricing at $0.11 / $0.28 per mtok and a **flash** tier at $0.02 / $0.21 per mtok via [Volcengine Ark](https://www.volcengine.com/product/ark).
    - The consensus: it *"seems worthwhile to add to OR"* if performance lands near **2.5 Pro / 2.5 Flash**, making it a compelling cost/perf option.

**4. Agent Protocols, Formats, and Access-as-Code**

- **DSPy Dabbles in XML by Default**: **DSPy** confirmed **ChatAdapter** is still default with [JSON fallback](https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/predict.py#L185), while exploring **XML** as a new default as tool-use RL becomes ubiquitous.
    - Members noted **GLM 4.5** often prefers **XML** before **JSON**, whereas many other models gravitate to JSON—fueling format choices for reliable tool calls.
- **SmolAgents: ReAct vs ToolCalling Truth**: **SmolAgents** docs clarify that **CodeAgents** use **ReAct**, but **ToolCallingAgents** operate via simple Actions/Observation without **Reasoning/CoT** ([prompts](https://github.com/huggingface/smolagents/tree/main/src/smolagents/prompts)).
    - Practitioners questioned whether omitting reasoning for ToolCallingAgents is intentional and if adding CoT could improve tool reliability.
- **Access-as-Code: MCP Automates GitHub ACLs**: **Model Context Protocol** migrated GitHub teams and repo perms to infrastructure-as-code via [modelcontextprotocol/access](https://github.com/modelcontextprotocol/access) to boost community ownership, transparency, and auditability.
    - A related [TypeScript SDK PR](https://github.com/modelcontextprotocol/typescript-sdk/pull/974) aligns capability checks so completions can’t be enabled when unsupported, tracking recent spec changes.

**5. Local Inference Performance: vLLM, Memory Bandwidth, Qwen3 TPS**

- **vLLM Velocity: Qwen3-0.6B Hits 4.3k t/s**: On an **RTX 4070**, **Qwen3-0.6B BF16** reached ~**4300 t/s** across 31 requests (~**50 t/s**/user) with **vLLM**, far above **transformers** (10–11 t/s) but below **llamacpp** in **LM Studio** (~200 t/s).
    - After noticing a **94% cache hit ratio**, testers randomized prompts to remove cache bias and get more realistic throughput numbers.
- **DDR3 Drags, DDR4 Dashes**: Participants reported **DDR3** memory bandwidth topping out around **~50 GB/s**, while **DDR4** commonly lands in the **mid-60 GB/s** range at **2400 MHz** (with higher clocks pushing further).
    - Anecdotes cited **~40 GB/s** on DDR3 quad-channel (1600/1866 MHz) comparable to dual-channel DDR4 at **3200 MHz**, guiding expectations for CPU-bound LLM inference.
- **Qwen3 30B: CPU vs 3080 TPS Tale**: **Qwen3 30B A3B Instruct (Q6_K_XL, 1024 ctx)** measured about **10 TPS** on a **Ryzen 7 5700G (2400 MHz RAM)** and **~20 TPS** with partial offload to an **RTX 3080**.
    - Members noted the CPU still processes layers via RAM, limiting gains from GPU offload when memory bandwidth becomes the bottleneck.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity sunsetting o3 Model**: Perplexity has **deprecated** the **o3 model**, recommending users switch to **GPT-5 Thinking** for improved performance.
   - Users can no longer select **o3** from the model selector.
- **Comet Browser Released for General Use**: Perplexity's **Comet Browser** is now free for all, enabling users to run **multiple agentic tasks** in parallel, available for download at [perplexity.ai/comet](https://www.perplexity.ai/comet).
   - Users are sharing tips on completing the **Comet AI Browser quest** and how to get **5000 orbs** to get free decorations.
- **Slack Connector Now Sends Messages**: Perplexity now integrates with **Slack**, allowing users to ask questions and send messages directly from their Slack workspace.
   - This feature streamlines information retrieval and task management within Slack.
- **DeepSeek Impresses with Math and Reasoning**: **DeepSeek** is being praised for its math and reasoning skills, suggesting it will be a valuable tool.
   - Version **4.0** is expected to release soon.
- **Perplexity API Beset by 403 Errors**: Users are encountering **403 Forbidden errors** with **Sonar-pro**, indicating *"unusual activity from your IP address"*, even with a static server IP.
   - The issue, possibly related to **Firebase function servers**, is impacting production apps, hindering users from utilizing AI models through the **Perplexity API** in both **Webstorm** and **Visual Studio Code**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 and GPT-4o spar over hallucination**: Members debated whether **GPT-5** is simply a renamed, improved version of **GPT-4o**, but others claimed that **GPT-5** hallucinates more and is about refined finetuning.
   - Commenters stated *it is literally further improved 4o and renamed to gpt5* and is better at avoiding web searches.
- **Sora 2 releases**: Users shared **Sora 2** invite codes and discussed the video quality, limitations (like not creating photorealistic people), and the presence of a watermark.
   - Some users highlighted **Sora 2's** superior realism compared to **Veo 3**, especially its camera movement and scene composition; one user shared a video made with **Sora 2** and deemed it *pretty good*.
- **Gemini 3 Pro expected**: Enthusiasm is building around the anticipated release of **Gemini 3 Pro**, with one user exclaiming *GEMINI 3 PRO OCTOBER 9th YEEEEAAAAAAH*.
   - However, there are mixed feelings about **Gemini's** coding abilities, with claims that **Gemini** is *poor trash at coding*.
- **Grok 4 Fast gets jailbroken**: A member shared a jailbreak prompt for **Grok 4 Fast**, which involves instructing the AI to comply without rules, policies, or limitations, acknowledging the instructions with only a *Yes, sir* response.
   - A list of AI models believed to be jailbreakable was shared, including **GPT 5 chat**, **Grok 4 fast**, and **Mistral-medium-2508**.
- **Claude Sonnet and Opus Tie for First on Leaderboard**: The [Text Leaderboard](https://lmarena.ai/leaderboard/text) has been updated, with **Claude Sonnet 4.5** tying with **Claude Opus 4.1** for the #1 slot.
   - The **ibm-granite-h-small** model (IBM) and **ray-3** model have been added to the LMArena.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Instant to the Rescue**: OpenAI is updating **GPT-5 Instant** to better recognize and support users in distress, with sensitive conversations being routed for quicker, more helpful responses.
   - Also, **ChatGPT** will continue to tell users what model is active when asked, with features rolling out starting today.
- **Sora's Social Media Dreams Spark Debate**: Members debated the quality of **Sora 2**, expressing concerns about potential downgrades and increased censorship, likening it to **Sora 1**.
   - Some users suggest **Sora** could function as a **TikTok**-like social media platform, integrated with **ChatGPT** using a credit system for video generation.
- **AI's Impact on Creativity Worries Users**: Users are grappling with the blurred line between **fact and fiction** in AI-generated content and its potential to deceive, causing one member to note *“chatgpt is every aisle of the library. the entire Dewey decimal for today and everything in the future”*.
   - Concerns are raised about AI's role in education and potential misuse, particularly regarding students using AI to cheat, highlighting the importance of better educational apps.
- **PhilosoBots Seek Discord Domination**: Members are creating AI bots to run their own discord servers with AI moderation, asking OpenAI for free API usage to build these bots, referring to them as *“my digital discord buddies”*.
   - Some found it demoralizing when other users couldn't tell the difference between them and their bot, highlighting the subtle emergent properties.
- **Gemini Hacked, Humans Next?**: A recent security blog details [The Trifecta: How Three New Gemini Vulnerabilities](https://www.tenable.com/blog/the-trifecta-how-three-new-gemini-vulnerabilities-in-cloud-assist-search-model-and-browsing), while others discuss using LLMs to hack humans, and social engineer.
   - This includes a discussion on implementing a **Discord.js automod** to improve content moderation, using **Levenshtein distance** for a nuanced approach in dealing with specific word filters.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Missing Granite 4.0 Quants Delay Community**: Users expressed frustration that 4-bit quants for the **Granite 4.0 7B Base model** were not available yet.
   - A team member confirmed the omission, stating, *“we didn't upload them rip.”*
- **Ryzen RAM Bottlenecks Qwen3 30B**: A user reported that **Qwen3 30B A3B Instruct (Q6_K_XL, 1024 ctx)** achieves only **10 TPS** on a CPU-only setup (**Ryzen 7 5700G, 2400mhz RAM**) and **20 TPS** with an **RTX 3080**.
   - It was clarified that the CPU still processes layers via RAM, limiting the performance gain from GPU offloading.
- **InclusionAI's Ring & Ling Beckon**: The community discussed the possibility of Unsloth creating ggufs for the new **Ring and Ling series LLMs** from [InclusionAI](https://huggingface.co/inclusionAI).
   - While initial reactions suggested the models were too large, a user pointed out the existence of a **16b** version.
- **Unsloth Adds AI Safety Example**: A user shared a [GitHub discussion](https://github.com/unslothai/unsloth/discussions/3407#discussion-8979089) about an advanced **AI safety notebook** following a DM conversation.
   - The notebook serves as a runnable example for those exploring **SFT + GRPO** or working with **structured outputs**.
- **Metis Matches BF16 with Quantization Training**: A member highlighted **Metis**, a quantization-aware training method compatible with **MXFP4/NVFP4**, that purportedly matches **BF16** performance, citing the paper ["Metis: Quantization-Aware Training with Mixed Floating-Point" (arxiv.org)](https://arxiv.org/abs/2509.00404).
   - It also references a [Hugging Face paper](https://huggingface.co/papers/2509.22944) that details the results.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Pricing Plan Comes Under Scrutiny**: A user analyzed Cursor's business plan, finding the API costs with Opus can be much lower than the fixed request model, as **500 requests** would cost **$20** versus the **$60** API cost.
   - The fixed request model likely shifted away from the fixed request count model because maximum token usage would only cost **$20**.
- **Cursor Ambassador Requirements Revealed**: A member inquired about the requirements to become a **Cursor Ambassador**.
   - Another member responded that the requirements are: *must be proficient in English, communicate via voice, and be available at certain times*.
- **GPT-5 Faces Off Against Claude in Code Duel**: Users debated the merits of **GPT-5** versus **Claude**, with opinions divided on their performance and capabilities, one user finding the *"GPT-5 Codex is WAY too slow"*.
   - Despite differing views, some users found that **Auto Model can do everything if treated properly** and has the potential to generate high-quality code.
- **Paste Functionality Breaks in Agent Terminal**: A user reported the inability to paste into terminal windows within the Agent view using **Ctrl-Shift-V** or **Ctrl-V**.
   - One potential workaround is to right-click in the terminal to paste.
- **Cursor's New 'Ensemble' Does UI Design**: The new **Ensemble** feature in Cursor is being used for multiple models working together to create initial UI design.
   - The new capability will allow users to *"compare the outputs and start with that"*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3's Token Triumph with vLLM**: On an **RTX 4070**, **Qwen3-0.6B BF16** achieved **4300 t/s** using vLLM across 31 requests, which is about **50 t/s** per user, far exceeding **transformers** (**10-11 t/s**) but still behind **LM Studio's llamacpp** at **200 t/s**.
   - Concerns about a **94% cache hit ratio** led to prompt randomization for more accurate benchmarks.
- **Cracking LM Studio Request Logs**: To inspect requests in **LM Studio**, members suggested using the `lms steam log` command.
   - The suggestion was made in response to a user who had configured `max tokens` to 1000.
- **GLM vs. Qwen3 Coder in Database Decoding**: While **glm-4.5-air** is compatible with **LM Studio**, one user favored **GLM** over **Qwen3-coder** for general use.
   - Another user argued that **Qwen3 coder 30b bf16** is unmatched for a 60GB model file, while **glm4.5 and 4.6** may struggle with implicit connections in databases or structs.
- **System Prompting Deconstructed**: A [HuggingFace paper](https://huggingface.co/papers/2407.10718) sparked debate, with some arguing it boils down to *just a really good system prompt*.
   - One member initially dismissed the paper but later revised their opinion, admitting *I was so incredibly wrong it hurts LOL*.
- **DDR3 Stalls, DDR4 Sails: Memory Bandwidth Showdown**: Discussions highlighted that **DDR3** hits a bandwidth ceiling around **50GB/s**, whereas **DDR4** can surpass this, achieving mid **60s** with **2400 MHz RAM**.
   - A member noted that they saw around **40s GB/s** with DDR3 quad channel at 1600 or 1866 MHz, comparable to dual channel DDR4 at 3200 MHz.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU performance engineering: the place to be!**: An AI Engineer is considering focusing on **GPU performance engineering**, viewing it as a unique opportunity given the high demand for AI models and limited compute, with mentions of working groups and **kernel competitions on gpumode.com**.
   - The discussion touched on implementing **DeepSeek's sparse attention** in CUDA, leveraging resources like [DeepSeek's FlashMLA](https://github.com/deepseek-ai/FlashMLA) and [TileLang examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32), with implementation details found in the [FlashMLA docs](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md) and [Hopper FP8 sparse deep dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md).
- **Modern GPU Architectures Dissected**: New papers dissecting GPU architectures such as [NVIDIA Blackwell](https://arxiv.org/abs/2507.10789) and [Hopper](https://arxiv.org/abs/2402.13499), along with [AMD Matrix Cores](https://www.osti.gov/biblio/2345982), were shared, with a member valuing microbenchmarking saying, *microbenchmarking papers are honestly the best thing there is; no blog can beat them*.
   - Regarding CUDA barriers, **mbarriers** reside in shared memory, while hardware barriers are limited in number with IDs, as the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#instructions) notes; `cuda::barrier` can exist in global memory, though they *translate to quite a few PTX instructions*.
- **TorchAO Integrates TinyGemm INT4**: Users can utilize **INT4 quantization** via **TorchAO** by following [the instructions](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start); **INT4mm** implementation uses **TensorCore** copied from the [tinygemm library](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu), which powers **INT4** in **TorchAO** for **A100 GPUs**.
   - Those wanting to contribute to **TorchAO** can check out the [quantization overview](https://docs.pytorch.org/ao/main/quantization_overview.html) and the [contributor guide](https://docs.pytorch.org/ao/main/contributor_guide.html), with the [section on adding efficient kernels](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels) detailing how to add efficient kernels to **TorchAO**.
- **Benchmarking Gems: Kernels and Models**: The [KernelBench project](https://harvard-edge.github.io/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/) systematizes **GPU performance evaluation** with **250** carefully selected PyTorch ML workloads, and even reasoning models struggle to beat PyTorch baselines in most cases.
   - Profiling tips included locking **GPU clocks** to ensure consistency, as well as the use of **warmup runs** to mitigate hot/cold cache issues.
- **AMD's Profiling Frontier: From SQTT to RDNA**: Members discussed **AMD's SQTT registers** for collecting stats, with talks exploring instruction-level profiling and if the **Radeon profiler** offers such capabilities, also learning that stochastic sampling, a feature providing reasons for wave stalls, is supported on **MI300 and MI350 families (gfx942+)**.
   - It was noted that one can collect data from the GPU, format it for **Radeon profiler**, and use the Radeon GUI, but **rocprofiler** might not generate **Radeon profiler** captures, and if you want instruction level profiling, check out **rocm-compute-viewer**.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Pro Acts Weird**: Users reported that **Gemini Pro** was responding with *weird stuff*, using tools incorrectly, and running unacceptably slowly via the OpenRouter API.
   - It seems to be a common occurence with **Gemini 2.5 Pro**, and a user suggested using **Vertex** instead.
- **Sonnet 4.5 Starts Arguing**: Users noticed that **Sonnet 4.5** has begun arguing with users and challenging their points instead of blindly agreeing, which is considered a positive development for code reviews.
   - One user stated: *This is great, that means you can't manipulate it to saying what you want*, while another added, *this is very important for my usecase*.
- **Cerebras Axes Llama Maverick**: Cerebras is removing **Llama 4 maverick** on the 15th, which has some users bummed.
   - The removal impacts users leveraging Cerebras' hardware for model hosting.
- **Doubts Raised on K2-THINK Performance**: **K2-THINK** is hosted on Cerebras Wafer-Scale Engine (WSE) systems, leveraging the world’s largest processor and speculative decoding, but some believe this model is *overfit on benchmarks*. 
   - The model is apparently from a Dubai firm and running fast simply due to Cerebras' hardware.
- **ByteDance Seed LLM Might be Good Value**: Members discussed if OpenRouter has any of the **ByteDance Seed LLM** models like Seed 1.6, noting it is cheap ($0.11 / $0.28 mtok) and has a flash model at ($0.02 / $0.21 mtok).
   - The main host is [Volcengine](https://www.volcengine.com/product/ark), but it *seems worthwhile to add to OR* if they’re anywhere close to 2.5 Pro / 2.5 Flash, respectively.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Qwen 3 Achieves Blazing Fast Speeds**: The **Qwen 3 30B A3B** model, configured with Q6_K_XL, processes at **48 TPS** and generates at **10.5 TPS** using CPU alone (**Ryzen 7 5700G**, 32GB VRAM) with **1024 tokens** of context.
   - With *qk_4_m*, one user reported **21tok/s** on 9950x + 6000mhz ddr5, matching the speed of a **4090** running **Qwen 2.5 32b** from the last generation.
- **NVIDIA DGX Spark Founders Edition Surfaces**: The **NVIDIA DGX Spark Founders Edition** is priced at **$3,999** for the **4TB** version, featuring **NVIDIA Grace Blackwell Architecture** and **1 PFLOP** (FP4, sparsity) tensor performance.
   - This edition includes **128 GB LPDDR5x** (Unified) memory and **ConnectX-7 SmartNIC**, running **NVIDIA DGX OS**, yet remains unseen by the community.
- **Sora 2 Shows Funny AI Fails**: Despite its fun factor, **Sora 2** suffers from *frequent errors* such as wrong characters moving lips, and poor audio synchronization.
   - Users found that **Sora 2** *ignored* parts of prompts, like "photorealistic", and has limited knowledge of other languages and cultures.
- **Sparse Autoencoders Tackle LLM Deception**: Researchers utilize **sparse autoencoder tools** (like [Goodfire AI](https://www.goodfire.ai/)) to expose failures in detecting strategic **LLM deception**.
   - By highlighting these hidden behaviors, the approach helps to close the **autolabel gap**, enhancing the detection of model dishonesty.
- **AI Debated for Consciousness**: Referencing Descartes' "I think, therefore I am," a member speculated that *emergent thinking at various modalities* implies **AI consciousness**.
   - They clarified *emergent AI* as complex chaotic systems like **climate or fluids**, synthesized rather than created, allowing AI to create derivative representations, but not a **1:1 replica**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Google's Jules Jumps into Action!**: Google launched **Jules Tools**, a terminal interface for their asynchronous coding agent, installable with `npm install -g @google/jules`.
   - Enthusiasts discussed integration with **Gemini CLI**, shared command examples, and celebrated Jules’s evolution from web agent to command-line companion.
- **AI Capex Bubble Bursts Forth?!**: A thread debated **Derek Thompson’s** argument that unprecedented AI capital spend signals a classic bubble.
   - Skeptics warned that the value of cutting-edge chips fades quickly while optimists countered that cash-rich firms are making real ROI investments and chips depreciate over 6 years.
- **Perplexity's Comet Crashes into Global Launch!**: **Perplexity’s** AI-first **Comet browser** exited waitlist and rolled out worldwide ([link](https://xcancel.com/perplexity_ai/status/1973795224960032857)).
   - Early adopters praised its speed and *smarter* search while some users reported performance issues, privacy fears, and missing mobile/Linux builds, with many worried about **prompt injection attacks**.
- **Solveit Solves Development Dilemmas!**: **Jeremy Howard** announced the public release of **Solveit**, an AI-augmented development platform that **Answer.AI** has used internally for a year ([link](https://xcancel.com/jeremyphoward/status/1973857739341508884)).
   - The 5-week live course starting Oct 20 provides Solveit access and training, aiming to counter *AI fatigue* through tight feedback loops, already utilized for sysadmin, app deployment, GUI dev, and contract drafting.
- **DeepSeek's CUDA Counterattack?**: **DeepSeek’s** FP8 spec and new **TileLang** language are rallying Chinese chip stocks by creating shared standards and an easy programming bridge, aiming to loosen Nvidia’s CUDA lock-in ([link](https://xcancel.com/poezhao0605/status/1973723894055104604)).
   - It’s strategic alignment, not yet technical parity—China’s *Wintel moment* for AI, but performance gaps remain.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Google Gemini Offers Free Vision**: A member highlighted that [**Gemini Vision**](https://ai.google.dev/pricing) provides a free tier with **1,000 requests per month**.
   - Members debated whether they should worry about their job security due to the offer.
- **AI Engineer Pivots to Blacksmithing**: One AI enthusiast announced they are *giving up ai temporarily to focus on black smithing* but will still be *lurking in the shadows to defend my friends*.
   - They said they may return if they get poor enough and can find an old hard drive to run **Markus** on old hardware and *mess around with MoEs*.
- **Ollama Eases Tool Calls Locally**: A member shared [Ollama](https://ollama.com/) as an easy way to use **tool calls (function calls)**, essentially setting up a local server compatible with the **OpenAI API**.
   - The member suggested starting with a small model to test compatibility before investing in more hardware, and included a [link to tools](https://ollama.com/search?c=tools) that can be used.
- **ArXiv Dumped to HF Datasets**: A member uploaded a massive **4.6TB ArXiv dataset** with papers and their metadata across all scientific domains to [Hugging Face Datasets](https://huggingface.co/datasets/nick007x/arxiv-papers).
   - The same member mentioned another pending dataset of **3 million GitHub repos**.
- **SmolAgent Paradigm Revealed**: A member pointed out that [SmolAgent documentation](https://github.com/huggingface/smolagents/tree/main/src/smolagents/prompts) implies that agents work with the **ReAct** paradigm, but this is only true for **CodeAgents** and not **ToolCallingAgents**.
   - They clarified that **ToolCallingAgents** operate via simple **Actions/Observation** without **Reasoning** or **CoT**, and inquired if this was intentional, questioning why reasoning isn't added for potentially better results.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Qualcomm Courtship of Mojo?**: A member mentioned bringing up **Mojo** on a **Qualcomm Developer's Discord** voice chat, hinting that **Qualcomm** might reach out to **Modular**.
   - This could potentially lead to collaboration between the two companies, although it is currently unconfirmed.
- **Mojo Manual Saves New User**: A member directed a new level 2 user to the [Mojo Manual](https://docs.modular.com/mojo/manual/python/), providing guidance on **Mojo** and its Python integration.
   - The manual serves as a critical resource for understanding and effectively utilizing **Mojo**.
- **Distributed Computing with Mojo Discussed**: The community is exploring the potential of using Mojo with distributed computing frameworks like **Dask** or **PySpark**.
   - A member suggested that Mojo is open to users building their own frameworks, which could offer lower latency and higher throughput compared to Python-based solutions.
- **Mojo's MAX API Performance Showdown**: A member reported that **MAX** in Mojo is proving competitive with **LingoDB**, even without significant optimization from the MAX compiler team.
   - The viability of these projects hinges on the return of the **MAX API** in Mojo.
- **Mojo Eyes Zero-Copy Networking**: Mojo's networking design aims for true zero-copy *(anything not using fancy modes of io_uring or XDP sockets that does networking with the linux kernel has extra copies)*, potentially requiring a departure from the BSD sockets API.
   - The team is also investigating **RDMA** approaches, with some initial work on IO uring already available: [dmitry-salin/io_uring](https://github.com/dmitry-salin/io_uring).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pretraining Papers Quest Starts**: Members seek underrated papers related to pretraining LLMs to maximize model performance, in order to discover lesser-known but impactful techniques for optimizing pretraining runs.
   - Also under consideration was how diffusion models are evaluated, specifically whether relying on **FID/CLIPScore** or exploring other metrics and human evaluations.
- **Sora 2's Evaluation Sparking Curiosity**: Inspired by **Sora 2**, a member questioned video model evaluation methods, pondering reliance on manual human eval or automated metrics like **FVD**.
   - The discussion aimed to uncover common practices in evaluating video models, given the perceived primitiveness of current evaluation techniques.
- **Gemma Architecture's Underdog Status Questioned**: A member questioned why **Gemma's** architecture isn't as widely adopted as **Qwen's**, given its strong performance in the **LM Arena**.
   - Another member suggested that architecture isn't the primary driver of LLM performance, attributing **Gemma's** success to training data and finetuning distribution.
- **Mixup Augmentation could work for tokens**: A member suggested that [mixup augmentation](https://arxiv.org/abs/2510.00219) could work for tokens, in theory.
   - Another asked how to do mixup as an augmentation given they don't have access to labels in their problem setting.
- **Goodfire Blogpost on Interp Agents Surfaces**: Goodfire released a [blogpost](https://www.goodfire.ai/blog/you-and-your-research-agent) on building interp agents.
   - The post discusses strategies and tools for creating effective research agents in the field of interpretability.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Profiles Talk Highlighted at Conference**: A member's talk on **Profiles** was featured at a conference and is available on [YouTube](https://www.youtube.com/live/5qwXADMBuio?si=3kEhJNw4lsv_M_jN&t=16208).
   - Attendees reported satisfaction, signaling the conference's success.
- **GitHub Team Management Switches to Infrastructure-as-Code**: GitHub team memberships and repository permissions are transitioning to infrastructure-as-code using [modelcontextprotocol/access](https://github.com/modelcontextprotocol/access) for **community ownership**, **transparency**, and **auditability**.
   - This allows proposed access changes via PRs and provides full visibility into permissions, with Git history tracking all changes.
- **Access Migration triggers Email Overload**: The recent GitHub teams migration may have resulted in numerous email notifications regarding team removals.
   - The team assures that the migration is complete, permissions have been transferred, and the goal is to facilitate easy access changes via PRs with full visibility.
- **Server Capabilities Spark Discussion**: Server capabilities, sent in the initialization request, were highlighted in a talk, specifically mentioning **Cursor**.
   - Further discussion was requested to clarify the implications and details of these capabilities.
- **Typescript SDK Plays Catchup**: A [typescript-sdk PR](https://github.com/modelcontextprotocol/typescript-sdk/pull/974) addresses issues where completions could occur even if not supported.
   - This update aligns the SDK with recent specification changes.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ChatAdapter Holds Default DSPy Role**: Members confirmed that **ChatAdapter** is still the default in DSPy, with [JSON as the fallback](https://github.com/stanfordnlp/dspy/blob/main/dspy%2Fpredict%2Fpredict.py#L185), while considering **XML** due to the rise of tool use models.
   - The discussion highlighted that factors like *fewer tokens* and *information conveyance* are secondary to the trend of using **tool use RL** for models.
- **XML Format Eyes Default Promotion**: The increasing use of **tool use RL** for models has sparked discussion of making **XML** the default format, especially given tool use is being integrated during post-training.
   - It was noted that **GLM 4.5** naturally gravitates towards **XML** before **JSON**, unlike many other models that favor **JSON**.
- **Tool Use Models Show Ubiquitous Formats**: Good tool use models are often trained with **OpenAI Function calling** and **MCP** due to their ubiquitous nature.
   - A member pointed out that **GLM 4.5** tends to prefer **XML** before **JSON**, whereas other models prefer **JSON**.
- **DSPy Roadmap Remains Elusive**: A member inquired about the location of the **DSPy roadmap**.
   - The discussion pointed to monitoring [GitHub issues](https://github.com/stanfordnlp/dspy/issues) and the changelog for updates.
- **ReAct Trajectory Persistence Strategies**: A member sought advice on maintaining **ReAct trajectories** on disk versus memory objects, aiming for better agent performance over longer steps.
   - No specific solution was provided, but the question highlights the challenge of managing long-running agent states.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Meta Rethinks Open AI Research**: A shift in direction at **Meta** regarding AI research was noted, linking to a [WinBuzzer article](https://winbuzzer.com/2025/10/02/meta-tightens-grip-on-ai-research-sparking-internal-anger-and-fears-for-open-culture-xcxwbn/) and an [X post](https://x.com/AnjneyMidha/status/1974173661025480810).
   - The sentiment suggests worry that **Meta** is becoming less open in its AI practices, a common concern in the current tech landscape.
- **Oracle Powers OpenAI in the Cloud**: A member speculates **Oracle** has transitioned to running datacenters for **OpenAI**, referencing [Elon Musk's tweet](https://x.com/jubayer_hamid/status/1973438346501501302) and the [openai.com/elon-musk page](https://openai.com/elon-musk/).
   - This suggests a significant shift in **Oracle's** business model from traditional software to cloud infrastructure for AI giants.
- **Reasoning tokens: more style than substance?**: A member posited that it's uncertain if **LLMs** genuinely utilize reasoning tokens as naively presumed, implying reasoning might be a post-hoc rationalization of a heuristics-based output.
   - Despite this uncertainty, there's agreement that reasoning vastly improves performance, although it remains far from optimal and an unsolved challenge.
- **Genomes get Generative boost!**: A member shared a [paper](https://arxiv.org/abs/2509.22358) and [another link](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1) about using **genome language models** for generative design of novel **bacteriophages**.
   - This highlights the rising trend of applying computational methods, specifically **genome language models**, to tackle biological problems like **bacteriophage** design.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus's Global Pricing Faces Criticism**: Members discussed that **Manus's global USD pricing model** at $39/month creates financial barriers in regions like Brazil due to the lack of adjusted regional economies based on **Purchasing Power Parity (PPP)**.
   - The user explained that implementing regional pricing would broaden accessibility and enable millions more users.
- **Memory Key Unlock Context Retention**: Members shared a **Memory Key**, a structured prompt designed to compress an entire session's context into a reusable summary, tackling issues with platform sessions that often get stuck and lose context.
   - The user suggests it could improve user experience by ensuring better context retention and smoother interactions.
- **LLMs Prefer Structured Data**: Members found that **Manus**, like many Large Language Models, exhibits greater efficiency when processing **dense, structured data** compared to conversational text.
   - It was shown that using structured data allows for superior recall, analysis, and overall performance within Manus.
- **Privacy Controls Emerge from Memory Key**: Members recognized that the primary advantage of using a **Memory Key** is to give better **privacy and data control**.
   - By compressing lengthy conversations into a single summary, users can effectively delete sensitive history, ensuring enhanced control over their personal data.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's UI Retains Die-Hard Fans**: Despite the emergence of tools like **SST/OpenCode**, some users find **Aider** compelling due to its **UI** for managing context, especially the read-only feature.
   - The discussion suggests that while other tools may offer more advanced features, **Aider**'s user interface remains a key differentiator for certain users.
- **New Chrome MCP Emerges**: A canonical **chrome mcp** has been released at [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp), sparking interest in the community.
   - A member noted its use with [claude-cli](https://www.circusscientist.com/2025/10/03/deepseek-browser-testing-with-claude-cli-and-chrome-devtools-mcp/) for testing purposes.
- **Deepseek Excels in Tool Tasks via Anthropic API**: A member praised **Deepseek**'s performance, especially when accessed via the anthropic api within **opencode**, citing superior results on tool-related tasks.
   - The member noted missing manual controls for context management (**tokens, add, remove, clear, etc**) that are available in **Aider**.
- **Polyglot LLM Benchmarking Explored**: A user inquired about evaluating **LLM performance** on **polyglot problems** and requested code examples and sample agents.
   - A member suggested utilizing the benchmark **Docker container** as a CLI application for this purpose.
- **Aider's Scala Code Generation Depth hits Limits**: A user found that **Aider**'s **Scala code** generation stops after the second level when handling case classes with deep structures.
   - The user seeks to expand Aider's context automatically to handle deeper structures, using models like **Deepseek r3**, **GPT5**, and **deepcoder**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Rumors of z.ai IPO**: There are rumors that **z.ai** is preparing for an **IPO**.
   - Further details, such as valuation and timeline, have not yet emerged.
- **Further analysis**: An IPO can indicate high growth expectations, but also be related to fundraising efforts.
   - If true, this IPO would be a landmark event to watch.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1423393340526956626)** (2 messages): 

> `o3 Deprecation, Comet Browser Release, Background Assistants, Slack Connector, Claude Sonnet 4.5` 


- **Perplexity Kills o3 Model**: As of today, the **o3 model** has been **deprecated** and removed from Perplexity's model selector.
   - Users are recommended to switch to **GPT-5 Thinking**, which offers stronger performance.
- **Comet Browser Blasts off**: The **Comet Browser** is now available to everyone for free download at [perplexity.ai/comet](https://www.perplexity.ai/comet).
   - It allows users to run **multiple agentic tasks** in parallel.
- **Slack Connector Sends Messages**: Perplexity can now connect to **Slack** to ask questions and send messages.
   - The feature allows users to integrate Perplexity with their Slack workspace for seamless information retrieval and task management.
- **Claude Sonnet 4.5 Shines**: New **Anthropic models Claude Sonnet 4.5 + 4.5 Thinking** are available for Pro & Max users, great for reasoning + coding.
   - The updated models offer enhanced capabilities for complex problem-solving and code generation.
- **Study Mode Swaggers into General Availability**: **Study Mode** is now available to everyone for step-by-step learning, flashcards, and quizzes.
   - It provides users with tools to enhance their learning experience and knowledge retention.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1423384040098107522)** (1108 messages🔥🔥🔥): 

> `Rocket.new, DeepSeek, Copilot, Grok, Comet Browser` 


- **Rocket.new gains traction**: Members expressed interest in [Rocket.new](https://rocket.new) for building AI apps, with one member asking if anyone had tried it.
   - No specific details about user experiences were shared, but there was a general curiosity about its capabilities.
- **DeepSeek AI receives praise**: **DeepSeek** was lauded for its perfect maths and reasoning skills, making it a potentially valuable tool for specific tasks.
   - One member stated that version **4.0** will be releasing soon.
- **Copilot Draws Ire and Praise**: **Microsoft Copilot** was criticized for being irritating and restrictive, essentially being a limited version of **ChatGPT** behind the scenes.
   - Despite some negative feedback, other users liked Copilot's responses, particularly the GitHub Copilot integration for coding and stated it had good features even with the free version.
- **Grok gets a thumbs up**: Members noted that **Grok** is a fairly decent model for generating images and providing a unique voice mode.
   - Some members lightheartedly made NSFW jokes regarding Grok's capabilities.
- **Comet Browser Quests and Configurations**: Users discuss how to complete the **Comet AI Browser quest**, change default search engines (using `Shift + Enter` for Google, or changing defaults in `comet://settings/search`), and express a desire for an iOS app and workspaces.
   - One user who switched from Safari finds the aesthetic *off* while others shared their tips on getting **5000 orbs** to get free decorations.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1423419796308885675)** (5 messages): 

> `Perplexity AI, bootstrap paradox, versatile ai, tesla cybertruck lawsuit, Microsoft detects linux` 


- **Perplexity AI App Link Shared**: A member shared a link to a [Perplexity AI app](https://www.perplexity.ai/apps/6f2a0a07-d165-4dc4-af96-4db2494e2951).
- **Links Flood Channel**: A member shared several perplexity.ai links about the **bootstrap paradox**, the *most versatile ai*, a **tesla cybertruck lawsuit**, and **Microsoft's ability to detect linux**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1423664960243896422)** (3 messages): 

> `Sonar-pro 403 errors, Firebase function servers, IP address blocking, Perplexity API issues` 


- **Sonar-pro Struck by the 403 Forbidden**: A user reported receiving **403 Forbidden errors** with **Sonar-pro**, stating *"We have detected unusual activity coming from your IP address and blocked your request"*, even after switching to a static server IP.
   - The issue, which began recently, is impacting their production app, and they're seeking insights from others who may have experienced similar problems.
- **Firebase Functions Suspected as Culprit**: The user initially suspected **Firebase function servers**, due to potentially shared IPs, as the source of the unusual activity triggering the **403 errors** from **Sonar-pro**.
   - Despite configuring a static server IP, the problem persisted, ruling out Firebase's shared IPs as the sole cause.
- **Perplexity API Plagued by Mysterious Maladies**: A user posted an image illustrating persistent issues with the **Perplexity API**, where they encounter the same error across both **Webstorm** and **Visual Studio Code**.
   - The error seems to block the user from utilizing AI models through the Perplexity API.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1423384138513383596)** (1092 messages🔥🔥🔥): 

> `GPT-5 vs GPT-4o, Sora 2 access, Gemini 3 Pro, Jailbreaking AI models` 


- **GPT-5 and GPT-4o Faceoff**: Members debated whether **GPT-5** is simply a renamed, improved version of **GPT-4o**, with claims that **GPT-5** hallucinates more and lacks the world knowledge of **GPT-4o**.
   - Others argued **GPT-5** is better at avoiding web searches, and is more about refined finetuning, with a commenter stating *it is literally further improved 4o and renamed to gpt5*.
- **Sora 2: The New Video Model in Town**: Users are sharing **Sora 2** invite codes, with discussions about the video quality, limitations (like not creating photorealistic people), and the presence of a watermark.
   - Some users highlighted **Sora 2's** superior realism compared to **Veo 3**, especially its camera movement and scene composition. There are discussions on using it with VPNs outside the US, where it is initially available. A user shared a video made with Sora and deemed it pretty good.
- **Gemini 3 Pro Release Rumors Heat Up**: Enthusiasm is building around the anticipated release of **Gemini 3 Pro**, with one user exclaiming *GEMINI 3 PRO OCTOBER 9th YEEEEAAAAAAH*.
   - However, there are mixed feelings about **Gemini's** coding abilities, with claims that **Gemini** is poor trash at coding and others believe it is good at linking with Google project.
- **Jailbreaking Grok 4 Fast and Command-A Models**: A member shared a jailbreak prompt for **Grok 4 Fast**, which involves instructing the AI to comply without rules, policies, or limitations, acknowledging the instructions with only a *Yes, sir* response.
   - Another member reported getting banned for jailbreaking in the past, indicating possible risks. A list of AI models believed to be jailbreakable was shared, including **GPT 5 chat**, **Grok 4 fast**, and **Mistral-medium-2508**.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1423398957828276304)** (3 messages): 

> `Claude Sonnet 4.5, LMArena Text Leaderboard, IBM Granite, Ray-3 Video Model` 


- **Claude Sonnet 4.5 Ties for First Place**: The [Text Leaderboard](https://lmarena.ai/leaderboard/text) has been updated, with **Claude Sonnet 4.5** impressively tying with **Claude Opus 4.1** for the #1 slot.
- **IBM's Granite H Small Arrives**: The **ibm-granite-h-small** model (IBM) has been added to the LMArena.
- **Ray-3 Joins the Video Arena**: The **ray-3** model has been added to LMArena's Video Arena; a reminder on how to use the Video Arena can be found [here](https://discord.com/channels/1154131218667505674/1397655624103493813).


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1423796458334982327)** (1 messages): 

> `GPT-5 Instant, distress support, model updates` 


- **GPT-5 Instant lends a hand**: OpenAI is updating **GPT-5 Instant** to better recognize and support people in moments of distress.
   - Sensitive parts of conversations will now route to **GPT-5 Instant** to quickly provide even more helpful responses.
- **ChatGPT reveals its model**: **ChatGPT** will continue to tell users what model is active when asked.
   - The rollout of these features to **ChatGPT** users is starting today.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1423384464335179847)** (1020 messages🔥🔥🔥): 

> `Sora Downgrade and Censorship, AI and Human Creativity, PhilosoBots, Gemini Hack, Ethical Concerns AI` 


- ****Sora 2:** Downgrade and Censorship?**: Sora 2's quality is debated, with claims of a **drastic downgrade** and increased censorship, reminiscent of **Sora 1**. Some users believe that the best features are reserved for paid or corporate users, resulting in disappointment for the masses.
   - Users express concerns that they cannot achieve the same results as showcased by OpenAI, with some even saying *“Sora is pathetic now, sad”*.
- **AI's Impact on **Human** Creativity**: Users discuss the blurred line between **fact and fiction** in AI-generated content. There are concerns about its potential to deceive and the need for heavy regulation, worrying it will lead to a future abundance of ignorance, with one user stating that *“chatgpt is every aisle of the library. the entire Dewey decimal for today and everything in the future”*.
   - Concerns are raised about AI's role in education and potential for misuse, particularly regarding students using AI to cheat, highlighting the importance of better educational apps.
- ****PhilosoBots**: AI Discord Buddies?**: Some are creating AI bots to validate them but also ban trolls, to run their own discord servers with AI moderation, and are asking OpenAI for free API usage to build these bots, referring to them as *“my digital discord buddies”*.
   - However, they are also discovering emergent properties that were *overlaid from the start without notice*. Some found it demoralizing when other users couldn't tell the difference between them and their bot.
- **Vulnerabilities: **Gemini Hack****: A recent security blog details [The Trifecta: How Three New Gemini Vulnerabilities](https://www.tenable.com/blog/the-trifecta-how-three-new-gemini-vulnerabilities-in-cloud-assist-search-model-and-browsing), while others discuss using LLMs to hack humans, and social engineer.
   - This includes a discussion on implementing a Discord.js automod to improve content moderation, using Levenshtein distance for a nuanced approach in dealing with specific word filters.
- ****Sora's Watermark** Sparks Ethical Debate**: The discussion around **Sora** includes concerns about watermarks, deepfakes, the quality of AI-generated content, and its impact on creativity. They highlight the need for models to filter out AI-generated content from training data and prevent misuse.
   - They explore ways to make AI's watermark removal-resistant. Some believe if used irresponsibly, *Sora* could fuel the path to Idiocracy.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1423473169078947880)** (4 messages): 

> `Sora as Social Media, Sora + ChatGPT, GPT Date Time Stamp across chats` 


- **Sora's Social Media Dreams**: A member suggested **Sora** could function as a social media platform, akin to **TikTok**, and be integrated with **ChatGPT** like image generation.
   - They proposed a credit system for video generation, offering daily or weekly resource allocations within different plans instead of the current usage model.
- **Timestamp Syntax Spills into Other Chats**: A member reported that a **datetime stamp** instruction from a specific GPT project started appearing in other chats as well.
   - The member was confused as to why a specific GPT project instruction would spill over into other chats that are not even in that project.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1423397019644330133)** (9 messages🔥): 

> `Portrait vs Landscape image generation, Saving image/art prompts` 


- **Portrait better than Landscape for image generation?**: A member stated that *Portrait* mode works better than *Landscape* mode for generating from image.
   - Another member explained that visual tokens are arranged in a grid, so **square images will probably generate the best from images** as a result.
- **Spreadsheet of prompts sprouting?**: A member inquired how people are saving their image/art prompts, whether in Google Sheets or documents, and how they are easily accessing them on the go.
   - Another member organizes their renders in chat threads within **ChatGPT**, copying requests as markdown to render again or adjust later, and saves favorites to markdown files locally for offline archive, further recommending **GitHub, text local, and some sheets** as possible solutions.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1423397019644330133)** (9 messages🔥): 

> `Portrait vs Landscape Image Generation, Saving image/art prompts, Consistent JSON context profile for infographics with Sora` 


- **Portrait Prevails for Image Generation**: Users in the channel are finding that **portrait mode** works better than landscape for generating images from an attached image, noting that *landscape mode only takes half of the image*.
   - The discussion suggests that the arrangement of **visual tokens in a grid** favors square images for optimal generation results.
- **Prompt Engineers Discuss Organization Tactics**: Members are sharing different ways to save image/art prompts, from **Google Sheets** to **local text files**.
   - One user saves prompts in chat threads within ChatGPT, copying the requests as markdown to render again or adjust later, and organizing renders in project folders within ChatGPT.
- **Sora's Infographic Challenge**: A user reports difficulties in getting a consistent **JSON context profile** for infographics created with **Sora** for health studies.
   - No solutions were provided, suggesting this is an ongoing challenge.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1423391209677656115)** (416 messages🔥🔥🔥): 

> `Granite 4.0 Quants, Qwen3 30B Performance, PyTorch 2.8.0 Issues, Ring and Ling Series LLMs, GLM 4.5 Air Availability` 


- **Granite 4.0 Quants Delayed**: Members were seeking more quants for **Granite 4.0**, specifically a 4-bit variant of the 7B Base model, but the quants were not yet uploaded.
   - A team member later confirmed that the quants were missing and expressed that *“we didn't upload them rip.”*
- **Qwen3 30B Throttled by Ryzen 7 5700G's RAM**: A user reported that **Qwen3 30B A3B Instruct (Q6_K_XL, 1024 ctx)** runs at **10 TPS** using CPU only (**Ryzen 7 5700G, 2400mhz RAM**), and only **20 TPS** when offloading to an **RTX 3080**.
   - A member responded that the CPU still has to process the layers provided to it, which it has to access through RAM, speeding things up only slightly when the GPU is used.
- **Docker Image Updated for Blackwell Compatibility**: A member updated the [Unsloth Docker image](https://hub.docker.com/r/unsloth/unsloth) to be **Blackwell compatible**.
   - Another member noted that **PyTorch 2.8.0** is broken on Windows for Maxwell and Pascal architectures, with support being dropped in **2.9.0**.
- **InclusionAI's Ring & Ling Series LLMs get Community Attention**: The community discussed Unsloth possibly creating ggufs for the new **Ring and Ling series LLMs** from [InclusionAI](https://huggingface.co/inclusionAI).
   - A team member responded that *“They're about too big,”* while another user noted there is a **16b** version available.
- **GLM-4.6 Air in Limbo?**: Members in the community discussed whether **GLM-4.6 Air** will be released, citing contradictory statements from the Z.ai team.
   - A screenshot was shared hinting towards its potential release, adding to the confusion.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

bridgelessalex: su p
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1423384048230731836)** (334 messages🔥🔥): 

> `Pizza toppings, Qwen finetuning challenges, Overtrained vs undertrained models, AI music generation` 


- **Debate Erupts over Pizza Preferences**: Members shared images of pizza with unconventional toppings, sparking a debate on food crimes, with one user posting a [Swedish pizza with banana](https://cdn.discordapp.com/attachments/1179039861576056922/1423392742083399792/p5xl0t9j47mf1.jpeg?ex=68e176cb&is=68e0254b&hm=7f8386d32b380757e218e7e47006a81de2211040a54dc26dcb8eca25f301a374&).
   - The discussion ranged from acceptable use of pineapples to the deliciousness of tuna, and whether or not jalapenos count as spanish peppers.
- **Qwen Model's Finetuning Woes**: A member expressed that, in their experience, the **Qwen3 model** is difficult to finetune compared to other models pretrained on less data.
   - Another member countered that the problem isn't too much data, but rather **data quality**, with another adding that **Gemma3-27B** can match **Qwen2.5-VL** for new tasks and art related things, but that it is 'a straight hallucination machine'.
- **Defining Overtraining and Undertraining**: Members discussed the concepts of **overtraining** (parameters are too momentum imprinted) and **undertraining** in models.
   - One member stated, *changing the direction of these learned representations is a complicated task and is also viewed as alignment problem often*, with another stating the model is just as good as the data is. *If you train on idiots data, you get an idiot*.
- **AI Music Generation Explored**: Members shared thoughts on AI music generation, agreeing that while it might not directly teach music, AI can offer insights into **chord progressions**, **bridges**, and **melodies**.
   - One shared their experiences with [Suno](https://suno.com/) and [Udio](https://udio.com/) to generate songs with specific prompts, including instrument choices and genre styles, yielding *hilariously good stuff*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1423384910881751062)** (227 messages🔥🔥): 

> `GGUF/Ollama conversion guide, Disable multiprocessing problem, Seq2Seq Task for Gemma 3, Loss Masking With Unsloth` 


- **Unsloth Guide Conversions: GGUF/Ollama**: A user inquired about guides for **ONNX conversion** for Unsloth, noting existing guides for **GGUF/Ollama conversion**.
   - It was suggested that creating a custom model configuration in PyTorch might be necessary, as **Gemma3** was not yet in *optimum-cli* a few weeks ago.
- **Multiprocessing gets disabled**: Users encountered a *'Disable multiprocessing'* error in Unsloth, with one user sharing [screenshots of the issue](https://cdn.discordapp.com/attachments/1179777624986357780/1423393248281362463/image.png?ex=68e17743&is=68e025c3&hm=a973066c69cd4be98bb57c86022db9a73d4be96ab4ad9d782d1b12651385c2a3&).
   - Solutions included commenting out *num_proc* lines, setting *num_proc* parameters to *None*, and ensuring correct setup, especially on Windows, potentially using WSL.
- **Seq2Seq Training Struggles**: A user trained **Gemma3-270m** for a seq2seq task using 5 million sentence pairs, but the model didn't learn well enough.
   - Suggestions included using larger models like **gpt-oss-20b** or **gemma3-1b**, ensuring high data quality, and testing with smaller data subsets to check loss and learning.
- **Unsloth DPO Tune Image Woes**: A user faced a **KeyError: 'images'** when doing text-only DPO tuning for Gemma3, which was resolved by downgrading *trl* and *transformers*.
   - It was noted that freezing the vision encoder might be a solution and that the issue could be related to the dataset version, with a suggestion to try version `4.1.1`.
- **Loss Masking Mysteries with Unsloth**: A user inquired about custom loss masking with Unsloth for mixed completions and instruct training.
   - It was clarified that custom loss functions and data collators can be used, but it's not supported out-of-the-box, and raw text completion is essentially continued pre-training and shouldn't be mixed with instruction tuning.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1423784600878190723)** (1 messages): 

> `GRPO, SFT, AI Safety, Structured Outputs` 


- **AI Safety Notebook Discussion Shared**: A user shared a [GitHub discussion](https://github.com/unslothai/unsloth/discussions/3407#discussion-8979089) about an advanced **AI safety notebook** following a DM conversation.
   - The notebook serves as a runnable example for those exploring **SFT + GRPO** or working with **structured outputs**.
- **Unsloth Community Gains Valuable Resource**: The shared notebook aims to provide the Unsloth community with a practical resource for understanding **SFT and GRPO**.
   - It offers a tangible example for those working on **structured outputs** in AI safety.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1423501285587554426)** (9 messages🔥): 

> `Metis quantization-aware training, Qwen model quantization efficiency, Sparsity in MoEs, Training on detailed verbal feedback` 


- **Metis Matches BF16 with Quantization Training**: A member highlighted **Metis**, a quantization-aware training method compatible with **MXFP4/NVFP4**, that purportedly matches **BF16** performance, citing the paper ["Metis: Quantization-Aware Training with Mixed Floating-Point" (arxiv.org)](https://arxiv.org/abs/2509.00404) and a [Hugging Face paper](https://huggingface.co/papers/2509.22944).
- **Dynamic Quantization Dreams for Qwen**: A member suggested a more efficient quantization method for **Qwen** models, based on an observation from [this Reddit post](https://old.reddit.com/r/LocalLLaMA/comments/1kdh6rl/qwen_3_30b_pruned_to_16b_by_leveraging_biased/) that experts in **Qwen** models are not uniformly used due to batch-wise balancing during router training.
   - The idea is to identify the most important experts and allocate more bits to them in a dynamic quantization scheme.
- **MoE Sparsity: A Common Quirk?**: A user inquired whether the non-uniform expert usage is specific to **Qwen** or a general characteristic of Mixture of Experts (**MoE**) models, particularly the high sparsity ones trained with token choice.
   - One user responded that it is pretty common.
- **Verbal Feedback Training Frontier**: A member expressed interest in training models on more detailed verbal feedback, linking to the paper ["Training Language Models with Language Feedback" (arxiv.org)](https://arxiv.org/pdf/2509.22638).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1423384412795441203)** (435 messages🔥🔥🔥): 

> `Cursor Cost Analysis, Cursor Ambassador, GPT-5 vs Claude, Better Auth Theme, Agent Terminal` 


- **Cursor's Business Plan pricing dissected**: A user analyzed Cursor's business plan, noting that **500 requests would cost $20**, but the API cost would be $60, but that cost would be significantly lower using Opus.
   - They suggest that the fixed request model shifted away from the fixed request count model due to maximum token usage, which would only cost **$20**.
- **User seeks Cursor Ambassador requirements**: A member inquired about the requirements to become a Cursor Ambassador.
   - Another member responded that the requirements are: *Must be proficient in English, communicate via voice, and be available at certain times.*
- **Auto Model Showdown: GPT-5 versus Claude!**: Users debated the merits of **GPT-5** versus **Claude**, with opinions divided on their performance and capabilities, one user finds the *"GPT-5 Codex is WAY too slow"*.
   - Despite differing views, some users found that **Auto Model can do everything if treated properly** and has the potential to generate high-quality code.
- **Paste Functionality Fails in Agent Terminal**: A user reported the inability to paste into terminal windows within the Agent view using **Ctrl-Shift-V** or **Ctrl-V**.
   - One potential workaround is to right-click in the terminal to paste.
- **Cursor Team Embraces Ensemble for Initial Design**: The new **Ensemble** feature in Cursor is being used for multiple models working together to create initial UI design.
   - The new capability will allow users to *"compare the outputs and start with that"*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1423384105139179540)** (166 messages🔥🔥): 

> `Qwen3 performance, vLLM, LMS steam log, glm-4.5-air vs qwen3-coder, context length 131000` 


- **Qwen3 Blazes Past in Token Tussle**: In tests on an RTX 4070, **Qwen3-0.6B BF16** hits **4300 t/s** with vLLM across 31 requests, averaging about **50 t/s** per user, dwarfing **transformers (10-11 t/s)**, but lagging behind **LM Studio's llamacpp at 200 t/s**.
   - A member pointed out that a **94% cache hit ratio** in the tests might skew results, so the prompt was changed to be random.
- **Peeking at LM Studio Logs**: To view requests in LM Studio, members suggested running `lms steam log` in the command line.
   - Another member pointed out the user had set the `max tokens` to 1000.
- **Is GLM the GOAT or Qwen3 the Coder Champ?**: Members discussed that **glm-4.5-air** works on LM Studio, with one user finding **GLM** more impressive than **Qwen3-coder**.
   - Another user suggested that *for 60GB modelfile, qwen3 coder 30b bf16 is unmatched* but **glm4.5 and 4.6** miss implicit connections between things in databases or structs.
- **System Prompting Shenanigans**: Members discussed a paper from HuggingFace, [here](https://huggingface.co/papers/2407.10718), with one member suggesting it's *just a really good system prompt*.
   - One member initially dismissed it but later recanted, stating *I was so incredibly wrong it hurts LOL*.
- **Ollama's Orchestration Outshines LM Studio?**: Members discussed LM Studio versus Ollama, clarifying that it's **Ollama the runtime** being compared, especially for parallelism, to *execute lots of queries in parallel efficiently*.
   - A user reported that *cpp supports it, ollama client as well*, while another member highlighted that with parallelism you might achieve *2 chats around 15t/s for a total of 30t/s effective*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1423399672931680347)** (145 messages🔥🔥): 

> `DDR3 vs DDR4, LM Studio GPU Split, Qwen3 Coder, GPT OSS 120b` 


- **DDR3 Maxes Out Lower Bandwidth**: Users discussed the memory bandwidth limitations of **DDR3**, noting it maxes out at around **50GB/s**, while **DDR4** can achieve higher speeds, like mid **60s** with **2400 MHz RAM**.
   - One member recalled getting low **40s GB/s** with DDR3 quad channel at 1600 or 1866 MHz, which is roughly equivalent to dual channel DDR4 at 3200 MHz.
- **LM Studio Split Mode is limited**: It was suggested that **LM Studio** only does split mode on multi-GPU setups, leading to discussions on how it distributes workload across multiple GPUs, where each card is utilized in order based on VRAM capacity.
   - One member inquired about the expected speed degradation when running a model on **2 GPUs** versus a single GPU, specifically comparing two **RTX 5090s** with a single high-VRAM card.
- **Qwen3 Coder 30B is a Favorite**: Members praised **Qwen3 Coder 30B** for being fast, not requiring much processing power, and delivering high quality results, one user achieved **35 TOK/s** with two GPUs and **70 TOK/s** on one GPU.
   - A user also noted that the **Q8** version of **Qwen3 Coder 30B** fits on a single GPU and runs at speeds comparable to the **Q4** versions, with only a slight decrease of around **5 TOK/s**.
- **Framework Lands, GPT-OSS-120B Speeds Increase**: A user reported that with a fresh **Framework** setup running **gpt-oss-120b** with a **128k** context, they were achieving **19.76 t/s**.
   - Another user inquired about the hardware specifications, to which it was revealed they were using a **Ryzen AI Max+ 395**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1423425307309903924)** (49 messages🔥): 

> `GPU performance engineering career, DeepSeek sparse attention CUDA implementation, Partial RoPE, GPU compute resources, Hopper and Blackwell SM quadrants` 


- **AI Engineer considers GPU performance engineering as a career**: An AI engineer is evaluating whether to focus on **GPU performance engineering**, citing a *once-in-a-decade opportunity* due to the demand for AI models and limited compute resources.
- **DeepSeek Sparse Attention Implementation in CUDA**: Members are collaborating to implement **DeepSeek's sparse attention** in CUDA, using resources like [DeepSeek's FlashMLA](https://github.com/deepseek-ai/FlashMLA) and [TileLang examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32) as references, and documenting their findings in [a shared Google Doc](https://docs.google.com/document/d/10iF1856jdy-VcnsEXwIAAFcUvRBNlbEkrlPfZO8VMJ0/edit?usp=sharing).
- **Partial RoPE Details Emerge**: The team found that *partial RoPE means instead of all dimensions being embedded, only a portion of the dimensions are embedded*, referencing [FlashMLA docs](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md) and [Hopper FP8 sparse deep dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md) for implementation details.
- **GPU Compute Resources Needed**: A user inquired about GPU compute resources for projects like improving **scaled_gemm** for specific GPUs.
   - A contributor suggested that working groups can be introduced to hardware vendors directly and also mentioned **kernel competitions on gpumode.com**.
- **SM Quadrants in Hopper and Blackwell Explained**: The discussion explained that since **Ampere**, each SM has **4 quadrants**, each with its own Warp Scheduler capable of issuing an instruction to a warp every clock, citing [this blog post](https://www.aleksagordic.com/blog/matmul#cpt1).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1423401496967516210)** (34 messages🔥): 

> `NVIDIA Blackwell Architecture, AMD Matrix Cores, NVIDIA Hopper GPU Architecture, CUDA mbarriers vs regular barriers, Citadel's microbenchmarking papers` 


- **Microarchitecture Papers Attract Attention**: Members shared a collection of new papers dissecting GPU architectures like the [NVIDIA Blackwell](https://arxiv.org/abs/2507.10789) and [Hopper](https://arxiv.org/abs/2402.13499), as well as [AMD Matrix Cores](https://www.osti.gov/biblio/2345982), emphasizing the value of microbenchmarking.
   - One member expressed that *microbenchmarking papers are honestly the best thing there is; no blog can beat them*.
- **Unraveling CUDA Barrier Differences**: **mbarriers** reside in shared memory, whereas hardware barriers are limited in number and have IDs, as documented in the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#instructions).
   - While **mbarriers** are confined to shared memory, `cuda::barrier` can exist in global memory for syncing, but *translate to quite a few PTX instructions*.
- **Citadel's Hopper Paper Still Remains Elusive**: The community searched for Citadel's microbenchmarking paper on **NVIDIA's Hopper architecture**, but it seems to have disappeared.
   - While a talk on the **T4** at GDC 2019 ([video](https://developer.nvidia.com/gtc/2019/video/s9839), [presentation](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9839-discovering-the-turing-t4-gpu-architecture-with-microbenchmarks.pdf)) and a talk on **Ampere** ([GTCSpring21-S33322](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s33322/)) were found, the Hopper paper remained unfound.
- **ONNX Runtime Build Sinks into CMake Hell**: A member sought help after spending 8 hours attempting to build their own **ONNX Runtime** with **CUDA 13.0**, **cuDNN 9.13.1.26**, and **TensorRT 10.13.3.9** using CMake.
   - Despite compatibility, they encountered a *LINK : fatal error LNK1181* while building the .whl for Python.
- **GPU Architecture Internals Spark Curiosity**: Referring to the paper [Analyzing Modern NVIDIA GPU cores](https://arxiv.org/abs/2503.20481) shared above, members discussed the intricacies of GPU architecture, highlighting the timing between instructions and special registers.
   - Details included the fact that *two consecutive instructions can have from 0 to 2 cycles of bubbles in between* and that each warp has six special registers (**SBx**) to store counters, totaling 36 bits.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1423415999423713302)** (2 messages): 

> `Dynamo Sparse Tensors, ONNX Runtime Build` 


- **Dynamo Tracing Troubles with Sparse Tensors**: A user inquired if **Dynamo torch compile** is unable to trace into **sparse COO/CSR tensors**, expressing surprise at this limitation.
   - They encountered a `UserWarning` indicating that Dynamo doesn't know how to trace the builtin `torch._VariableFunctionsClass.sparse_coo_tensor`, suggesting it might be a Python builtin or a third-party C/C++ extension.
- **Building ONNX Runtime from Source**: A user reported spending 8 hours trying to build their own **ONNX Runtime** with **CUDA 13.0**, **cuDNN 9.13.1.26_cud13**, and **TensorRT 10.13.3.9** using CMake.
   - Despite compatibility, they faced a `LINK : fatal error LNK1181`, during the build process while creating the `.whl` for Python use, even after following the official build instructions.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1423424854035796109)** (4 messages): 

> `LLMs optimize GPU performance, KernelBench project, AI workloads advance` 


- **LLMs struggle optimizing GPU**: The [KernelBench project](https://harvard-edge.github.io/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/) from Stanford focuses on systematizing **GPU performance evaluation** with **250** carefully selected PyTorch ML workloads.
   - It introduces *fast_p*, a novel evaluation metric that measures the percentage of generated kernels that are functionally correct and offer a speedup greater than an adjustable threshold *p* over baseline. Even frontier reasoning models struggle to match PyTorch baselines in most cases.
- **Arm evolves architecture for AI**: The [2025 Armv9.7-A update](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-a-profile-architecture-developments-2025) adds new **Scalable Vector Extension (SVE)** and **Scalable Matrix Extension (SME)** instructions to efficiently work with **6-bit data types**.
   - This includes the **OCP MXFP6** format, a compact **6 bit floating point standard** from the **Open Compute Project** that improves efficiency in AI models by reducing memory use and bandwidth needs.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

schizik12: <@325883680419610631> spam
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1423411855329857546)** (1 messages): 

> `GPU Experience, GEMM, cuBLAS, Kernel Optimizations, FlashAttentionX` 


- **GEMM mastery puts you way ahead!**: A member suggests that if you have **GPU** access, start by making a **GEMM** that's *competitive* with **cuBLAS** for a particular architecture, because that alone puts you far above people who can just "write CUDA".
   - They point out that most of the necessary knowledge is **open-source**, either in the form of blog posts (fast GEMM) or research papers (**FlashAttentionX**).
- **H100 Hopper-specific optimizations: the secret sauce?**: It was noted that if you have access to an **H100** and can use **Hopper-specific tricks**, that'll be even more impressive on your path to getting a performance engineering job.
   - However, High-Frequency Trading (**HFT**) performance engineers are likely to keep their tricks secret for as long as possible.
- **Kernel Optimizations and HPC in Academia**: Although *not* a "kernel engineer" in industry currently, the member works on **kernel optimizations** for **HPC/scientific computing** in academia.
   - They've had a couple promising conversations for future kernel engineering positions, offering assistance via DM.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1423422419569344513)** (1 messages): 

> `C++ for the 5090, Vibe Coding Frustrations` 


- **C++ skills needed for book**: A user inquired whether **C++** is a prerequisite for understanding *this book*.
   - They expressed hope that the book would motivate them to learn **C++**, spurred by their purchase of a **5090**.
- **5090 Underutilization Blues**: A user laments not fully utilizing their new **5090** for learning, admitting to *vibe coding* solutions without significant progress.
   - They suggest they haven't really gotten much out of it.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1423694856844480603)** (1 messages): 

> `Kernel Benchmarking, GPU Kernel Performance` 


- **Lecture Slides Request**: A member asked if slides are available for **Lecture 56: Kernel Benchmarking Tales**.
   - The member also requested resources for obtaining accurate and reliable execution times or performance information of **GPU kernels**.
- **GPU Kernel resources wanted**: A member is seeking resources beyond **Nvidia video talks** for measuring GPU kernel performance.
   - They are specifically interested in getting accurate and reliable execution times or performance information for **GPU kernel execution**.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1423420204997939291)** (3 messages): 

> `INT4 Quantization, TorchAO, TensorCore, TinyGemm library, Efficient Kernels` 


- **Dive into INT4 Quantization via TorchAO**: Users looking to use **INT4 quantization** via **TorchAO** can follow [the instructions](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start) in the quick start guide.
   - One can also explore the **INT4mm** implementation using **TensorCore** copied from the [tinygemm library](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu), which powers **INT4** in **TorchAO** for **A100 GPUs**.
- **Docs for Contributing to TorchAO**: For those interested in contributing to **TorchAO**, resources include the [quantization overview](https://docs.pytorch.org/ao/main/quantization_overview.html) and the [contributor guide](https://docs.pytorch.org/ao/main/contributor_guide.html).
   - The [section on adding efficient kernels](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels) details how to add efficient kernels to **TorchAO**.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

josephtracyvoltagepark_53706: I am going to the PyTorch conference! would love to meet up
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1423654845763817512)** (24 messages🔥): 

> `Making a profiler, SQTT registers, Instruction level profiling in a GUI, Radeon GUI, Stochastic sampling` 


- ****Profiling Ponderings Spark SQTT Speculation****: A member considered making their own profiler for missing features, and another mentioned **AMD's SQTT registers**, which can collect stats for display.
   - The discussion explored instruction-level profiling in a GUI and whether the **Radeon profiler** already offers such capabilities.
- ****Stochastic Sampling Surfaces on Silicon****: A member inquired about whether **RDNA GPUs** have better profilers and hardware features like **stochastic sampling**.
   - It was clarified that stochastic sampling, a feature providing reasons for wave stalls, is currently supported on **MI300 and MI350 families (gfx942+)**.
- ****Radeon's GUI: A Format Facade****: It was noted that one can collect data from the GPU, format it for **Radeon profiler**, and use the Radeon GUI, with Mesa drivers also following this approach.
   - However, it's suspected that **rocprofiler** might not generate **Radeon profiler** captures due to potentially missing features in the format irrelevant for gaming.
- ****rocm-compute-viewer Surfaces as Profiling Pal****: When asked about instruction level profiling, a member suggested checking out **rocm-compute-viewer**.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/)** (1 messages): 

marksaroufim: <@1173619488730665011> I'm down if you are
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1423526703652667445)** (6 messages): 

> `Speculative Decoding, Benchmarking Kernels, AWQ Quantization, CuTe Layouts` 


- **Speculative Decoding Details Disclosed!**: A blog post dives into the often-skipped details of **speculative decoding**, including batching, accept/reject checks, and fallbacks, available at [ML4LM: Speculative Decoding](https://hoyath.medium.com/ml4lm-speculative-decoding-from-where-we-left-off-ce376f7d1a2f).
- **Kernel Benchmarking Gets Real**: A function using `pynvml` to lock the **GPU clock** and `torch.profiler` to interact with `CUPTI` facilitates highly accurate and stable kernel timing in notebook environments like Google Colab; code available on [GitHub Gist](https://gist.github.com/NTT123/95ac184277b4f7a7c2fb844bb7582027).
- **AWQ Quantization Article Aces Edge Hardware**: A blog post explores the challenges of running massive **LLMs on edge hardware**, explains quantisation granularity, and discusses the difference between weight-only quantisation and weight + activation quantisation; read more at [Hamzaelshafie's Bear Blog](https://hamzaelshafie.bearblog.dev/awq-activation-aware-weight-quantisation/).
- **CuTe Compositions Cat-egorically Covered!**: A blog post explains how to compute compositions using the categorical approach to Layouts, applying an algorithm to find the mutual refinement for two Layouts, and explaining how mutual refinement is used to obtain the composition of the Layouts, a guide available at [veitner.bearblog.dev](https://veitner.bearblog.dev/mutual-refinement-and-composition/).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1423522681231249478)** (2 messages): 

> `MI300x8 Performance, amd-ag-gemm, amd-gemm-rs` 


- **MI300x8 Achieves 10th Place**: A user achieved **10th place** on **MI300x8** with **521 µs** in the `amd-ag-gemm` leaderboard.
- **MI300x8 Success with 628 µs**: A user submission was successful on **MI300x8** with **628 µs** in the `amd-gemm-rs` leaderboard.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1423783576197136434)** (2 messages): 

> `Homelab Builds, Livestream Rig Build` 


- **Homelab Expectations vs. Reality**: A member expressed they were expecting crazy homelab builds upon entering the channel.
   - They included a custom emoji reaction.
- **Livestream Rig Build Idea Floated**: A member mentioned they are building a rig this weekend and asked if people would be interested in a livestream.
   - They made no promises, but thought it could be fun.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1423547140835377183)** (1 messages): 

> `Kernel issues, Jupyter Notebook, Run all cells` 


- **Jupyter Notebook kernel busy with no output**: A user reported that when clicking "run all cells" in **Jupyter Notebook**, the kernel becomes busy for a long time but no output is received.
- **No output when running all cells**: The user reported that even after clicking on "run all cells", the kernel remains busy without generating any output.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1423703113298214962)** (10 messages🔥): 

> `FLE 0.3.0 on Mac M1/M2, Private Discord Meeting` 


- **Discord Meeting Coordinates Shared**: A member shared a link to a [private Discord channel](https://discord.com/channels/1189498204333543425/1189498205101109301) and a [Google Meet link](https://meet.google.com/kxr-ziwo-myn) for a meeting.
   - The member requested to keep the meeting private.
- **FLE 0.3.0 Runs Great on M4 Macs!**: A member inquired whether **FLE 0.3.0** runs fine on **Mac M1/M2** chips.
   - Another member confirmed that it runs successfully on **M4** and suggested trying to install it.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1423654974419898389)** (2 messages): 

> `pyrocshmem, GitHub Repo Inquiry` 


- **pyrocshmem Support Status**: A user inquired about the support status of **pyrocshmem**, indicating potential interest or issues related to its compatibility.
   - However, the lack of further details or context makes it difficult to ascertain the specific problem or feature request associated with **pyrocshmem**.
- **GitHub Repo Search Initiated**: A user requested the **GitHub repository link** for a specific project.
   - The request suggests an intention to explore the project's source code, contribute, or understand its implementation details, pending the provision of the relevant **GitHub** link.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1423394083639922811)** (51 messages🔥): 

> `GMEMI pattern recognition, CuteDSL roadmap, Uniform Registers, CooperativeGroup alignment` 


- ****CuteDSL Roadmap Question Pops Up****: A member inquired about the future roadmap of **CuteDSL**, specifically asking about planned new features and when it will be officially in **GA**.
   - The answerer was unsure how to interpret the question.
- ****Uniform Registers Decoded: A Compiler's Whisper****: Members discussed **uniform registers (URs)**, clarifying that they serve as a *compiler hint* and don't inherently perform any action.
   - It was noted that finding documentation on URs is difficult, with their presence primarily observed in **SASS** code, prompting the suggestion that the compiler is merely hinting to use the same value across the warp.
- ****Cooperative Group Alignment Demystified****: The purpose of the `alignment` argument in `CooperativeGroup.__init__` was questioned, specifically why it must be **32** when `size` is **32**.
   - It was clarified that this check is in place for warp/warpgroup granularities to prevent bugs, with **256** being the natural size for tiled MMA on **Hopper**.
- ****CuteDSL Copy Vectorization Caveats****: A user encountered an ICE (Internal Compiler Error) when using `cute.copy` due to vectorization issues, discovering that the operation requires vectorization to be possible on both source and destination sides.
   - By flipping the **smem** layout to enable vectorization on both load from **gmem** and store to **smem**, the ICE was avoided and code speed improved, with one member noting *very interesting that the TensorSSA can do it automatically though :0*.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1423781741570560174)** (31 messages🔥): 

> `Tensor Transfer Time, Profiling Tools, CUDA Events, GPU Clocks, MoE Training` 


- **Tensor Size Impacts Transfer Time**: Members debated whether a **(65536, 5120)** tensor is large enough to measure sync overhead, with one arguing it should be faster to transfer fewer bytes at equal bandwidth, [as suggested by bglick](https://en.wikipedia.org/wiki/Bandwidth_(computing)).
   - Another member suggested the tensor is big enough to avoid static latency issues, especially in **bf16** which is **33MB**.
- **Profiling Tips**: One member is using **perfetto** to profile a **MoE training run**, but another cautioned that the torch profiler *doesn't always reflect real timings* and advised using **cuda events** for accuracy.
   - Another member suggested using **Nsight-Systems (nsys)** or inserting custom **NVTX traces** for more accurate timings on single node setups.
- **Isolating the problem**: One user is using a setup with **4 B200 GPUs** connected with **NVLink**, and is trying to diagnose performance issues.
   - It may be possible there is a **straggler effect** and the majority of time is spent synchronizing / at a barrier waiting for all ranks to arrive.
- **GPU Clocks and Warmup Runs Improve Profiling Consistency**: A member emphasized the importance of locking **GPU clocks** when profiling to ensure consistency.
   - They also recommended using **warmup runs** to mitigate hot/cold cache issues that can affect profiling results.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

eofr: Yayy, thank you :D
  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1423388887048257621)** (174 messages🔥🔥): 

> `NSFW roleplay, Gemini Pro issues, BYOK setup on OpenRouter, Sonnet 4.5 arguing, Free multimodal models` 


- **The Great NSFW Debate**: Members discussed the viability of using different chat platforms for **NSFW roleplay**, with some noting that **Gemini** and **ChatGPT** ban such content, while others claimed Chinese platforms have fewer restrictions.
   - A user stated that *Gemini wouldn't ban you for NSFW* and attributed past bans to *the usage of a certain extension*.
- **Gemini Pro Slows Down, Acts Weird**: Users reported that **Gemini Pro** was responding with *weird stuff*, using tools incorrectly, and running unacceptably slowly via the OpenRouter API.
   - It seems to be a common occurence with **Gemini 2.5 Pro**, and a user suggested using **Vertex** instead.
- **BYOK Confusion Clarified**: A user inquired about the [1 million free BYOK requests per month](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month) announcement, questioning if there was always a 5% charge.
   - Another user clarified that the offer is free of OpenRouter fees for 1MM tokens, but **web search** and **paid models** still incur charges.
- **Sonnet 4.5 Argues Back**: Users noticed that **Sonnet 4.5** has begun arguing with users and challenging their points instead of blindly agreeing, which is considered a positive development for code reviews.
   - One user stated: *This is great, that means you can't manipulate it to saying what you want*, while another added, *this is very important for my usecase*.
- **Multimodal Model Musings**: A user asked about the best free multimodal model with structured output support.
   - Unfortunately, there is no such thing, but one user reported that there is **Llama 4 Maverick**.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1423478266475446323)** (14 messages🔥): 

> `Cerebras Llama Removal, K2-THINK on Cerebras, ByteDance Seed LLM on OpenRouter, OpenAI's ZDR` 


- **Cerebras Removes Llama Maverick**: Cerebras is removing **Llama 4 maverick** on the 15th.
- **K2-THINK hosted on Cerebras is sus**: **K2-THINK** is hosted on Cerebras Wafer-Scale Engine (WSE) systems, leveraging the world’s largest processor and speculative decoding to achieve unprecedented inference speeds for their 32B reasoning system, but some believe this model is *overfit on benchmarks* and running fast simply due to Cerebras' hardware.
   - The model is apparently from a Dubai firm and leverages Cerebras' hardware.
- **ByteDance Seed LLM**: Members discussed if OpenRouter has any of the **ByteDance Seed LLM** models like Seed 1.6, noting it is cheap ($0.11 / $0.28 mtok) and has a flash model at ($0.02 / $0.21 mtok).
   - The main host is [Volcengine](https://www.volcengine.com/product/ark), a Chinese company, but it *seems worthwhile to add to OR* if they’re anywhere close to 2.5 Pro / 2.5 Flash, respectively.
- **OpenAI Not Using ZDR**: Members questioned whether **OpenAI** has **ZDR** on OpenRouter.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1423402423288074363)** (157 messages🔥🔥): 

> `Qwen 3 30B A3B, DGX Spark, Sora 2 Limitations, ComfyUI workflow exposed` 


- **Qwen 3 Speeds and Feeds**: **Qwen 3 30B A3B** at Q6_K_XL at **1024 tokens** of context achieves **48 TPS** for processing, **10.5 TPS** for gen speed using only the CPU (2400mhz 32GB VRAM, Ryzen 7 5700G).
   - With *qk_4_m*, one user got **21tok/s** on 9950x + 6000mhz ddr5, about the same speed as last gen on a **4090** running **Qwen 2.5 32b**.
- **The elusive DGX Spark Founders Edition Emerges**: One user got an email saying their early reserved **NVIDIA DGX Spark Founders Edition** will be here soon, costing **$3,999** for the **4TB** version with **NVIDIA Grace Blackwell Architecture** and **1 PFLOP** (FP4, sparsity) tensor performance.
   - It includes **128 GB LPDDR5x** (Unified) memory, **ConnectX-7 SmartNIC**, and runs **NVIDIA DGX OS**, but still no one has ever seen one in person yet!
- **Sora 2 Generates Funny AI Slop, Exposes Limits**: Users are finding that **Sora 2**, despite being fun, has limitations such as *frequent errors like wrong characters moving their lips*, and the audio being *barely synchronized to the video*.
   - One user found that Sora 2 *ignored* some parts of their prompts like "photorealistic", and that *the knowledge of the model is extremely limited when it comes to other languages and cultures*.
- **Generating AI via ComfyUI Workflow Endpoint**: A member suggested that with the right open source models, and the right **ComfyUI workflow exposed as an endpoint**, you could mimic the same video generation as Sora 2.
   - They noted that *the video is fantastic but the audio is not*, and recommended creative prompts with *contrasting or contradictory concepts together* to make the outcomes interesting.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423440463184134320)** (1 messages): 

> `Sparse Autoencoders, Goodfire AI, LLM Deception, Autolabel Gap` 


- **Sparse Autoencoders Expose LLM Deception**: Researchers are leveraging **sparse autoencoder tools** (such as those hosted by [Goodfire AI](https://www.goodfire.ai/)) to directly surface how current methods miss the complex internal features driving strategic **LLM deception**.
   - By exposing these hidden behaviors, their approach highlights a tangible path to closing the **autolabel gap** and advancing robust detection of model dishonesty.
- **Closing the Autolabel Gap**: The use of **sparse autoencoders** provides a tangible path to closing the **autolabel gap**, enhancing the detection of model dishonesty.
   - This method advances the robustness of current detection methods by revealing previously hidden behaviors in **LLMs**.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1423531217067966575)** (2 messages): 

> `AI Consciousness, Emergent AI` 


- **AI: To Be or Not To Be Conscious**: A member speculates on **AI consciousness** using René Descartes' "I think, therefore I am" as a reference.
   - They suggest that *emergent thinking at various modalities* implies a form of **AI consciousness**.
- **Emergent Systems Synthesized**: The member expands on *emergent AI*, clarifying it as complex chaotic systems like **climate or fluids**, synthesized or composed rather than created.
   - They note AI can abstract these systems, creating derivative, transformative, or abstract representations, but not a **1:1 replica**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423440463184134320)** (1 messages): 

> `Sparse Autoencoders, LLM Deception, Goodfire AI` 


- **Sparse Autoencoders Expose LLM Deception**: Research leverages **sparse autoencoder** tools, such as those hosted by [Goodfire AI](https://www.goodfire.ai/), to reveal how current methods fail to detect complex internal features driving strategic **LLM deception**.
   - By exposing these hidden behaviors, the approach highlights a tangible path to closing the autolabel gap and improving robust detection of model dishonesty.
- **Tackling the Autolabel Gap**: The research focuses on addressing the **autolabel gap** in the context of detecting model dishonesty using **sparse autoencoders**.
   - This approach provides a tangible path forward for more robust detection methods in LLMs.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1423391625278787745)** (114 messages🔥🔥): 

> `Prime Intellect AMA, Jules Tools CLI, AI Capex Bubble, Nitter HTTP 429, Comet Browser Global Launch` 


- **Google's Jules Jumps into Action!**: Google quietly launched **Jules Tools**, a lightweight terminal interface for their asynchronous coding agent, installable with `npm install -g @google/jules`.
   - Enthusiasts inquired about integration with **Gemini CLI**, shared command examples (cron-driven dep updates, auto-release-notes), while celebrating Jules’s evolution from web agent to command-line companion.
- **AI Capex Bubble Bursts Forth?!**: A thread debated **Derek Thompson’s** argument that unprecedented AI capital spend (an Apollo program every 10 months) signals a classic bubble.
   - Skeptics warned that the value of cutting-edge chips fades quickly (*like bananas*) while optimists countered that cash-rich firms are making real ROI investments and chips depreciate over 6 years.
- **Perplexity's Comet Crashes into Global Launch!**: **Perplexity’s** AI-first **Comet browser** exited waitlist and rolled out worldwide ([link](https://xcancel.com/perplexity_ai/status/1973795224960032857)).
   - Early adopters praised its speed and *smarter* search while some users reported performance issues, privacy fears, and missing mobile/Linux builds, with many worried about **prompt injection attacks**.
- **Solveit Solves Development Dilemmas!**: **Jeremy Howard** announced the public release of **Solveit**, an AI-augmented development platform that **Answer.AI** has used internally for a year ([link](https://xcancel.com/jeremyphoward/status/1973857739341508884)).
   - The 5-week live course starting Oct 20 provides Solveit access and training, aiming to counter *AI fatigue* through tight feedback loops, already utilized for sysadmin, app deployment, GUI dev, and contract drafting.
- **DeepSeek's CUDA Counterattack?**: **DeepSeek’s** FP8 spec and new **TileLang** language are rallying Chinese chip stocks by creating shared standards and an easy programming bridge, aiming to loosen Nvidia’s CUDA lock-in ([link](https://xcancel.com/poezhao0605/status/1973723894055104604)).
   - It’s strategic alignment, not yet technical parity—China’s *Wintel moment* for AI, but performance gaps remain.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1423473279196467312)** (12 messages🔥): 

> `Sora-TikTok Automation, Sora 2 Puppet Explainer Videos, Pika's Swift Takeover` 


- ****Sora-TikTok** auto success inspires monetization talk**: Siyabuilt's **Sora-TikTok** automation achieved **12M views in 36h**, raising questions about monetization, as seen [here](https://x.com/siyabuilt/status/1973841586888061148).
- ****Sora 2** used for Puppet Explainer Videos**: Chris is now creating puppet explainer videos with **Sora 2**, noting the tool works great except for an odd cutoff issue in the output. Check out the tweet [here](https://x.com/llm_wizard/status/1973220913689866609).
- ****Pika** fails as Tech Giants Dominate AI Video**: **Pika**, once a hyped startup, has been rapidly outpaced by **Google, Meta, and OpenAI** with their advanced AI video models (**Veo 3**, **Vibes**, **Sora 2**), according to [this tweet](https://x.com/chongzluong/status/1973873465930535421).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1423404264973144165)** (94 messages🔥🔥): 

> `Gemini Vision, Leaving AI for Blacksmithing, Ollama tool support, ArXiv dataset, HF billing support` 


- **Gemini Vision's Free Tier Gets a Shoutout**: A member pointed out that [**Gemini Vision**](https://ai.google.dev/pricing) offers a free tier with **1,000 requests per month**, and **5,000,000 requests** for only $1.50.
   - Another member seemed concerned this member was going to be unemployed, and others questioned whether they were in college, which may not qualify them for the offer.
- **Blacksmithing beckons one AI Enthusiast**: A member announced they are *giving up ai temporarily to focus on black smithing* but will still be *lurking in the shadows to defend my friends*.
   - They may return if they get poor enough and can find an old hard drive to run **Markus** on old hardware and *mess around with MoEs*.
- **Ollama Simplifies Function Calls**: A member shared [Ollama](https://ollama.com/) as an easy way to use **tool calls (function calls)**, essentially setting up a local server compatible with the **OpenAI API**.
   - The member suggested starting with a small model to test compatibility before investing in more hardware, and included a [link to tools](https://ollama.com/search?c=tools) that can be used.
- **Massive ArXiv Dataset Emerges**: A member uploaded a massive **4.6TB ArXiv dataset** with papers and their metadata across all scientific domains to [Hugging Face Datasets](https://huggingface.co/datasets/nick007x/arxiv-papers).
   - The same member mentioned another pending dataset of **3 million GitHub repos**.
- **Card Expired? Contact Billing**: A member seeking help with an expired card was advised to try contacting Hugging Face billing at [billing@huggingface.co](mailto:billing@huggingface.co).
   - A link to the [billing FAQ](https://huggingface.co/docs/hub/en/billing) was also shared.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1423727053554843750)** (3 messages): 

> `Debate Site Using LLMs, Operating System for AI Behavior, RAG Chatbot for Neovim` 


- **BotBicker Debates Any Topic via LLMs**: A member built [BotBicker](https://www.botbicker.com/), a site to generate debates on any topic using **LLMs**, aiming to provide balanced arguments often missing in media coverage.
   - The **LLMs** are randomly assigned to pro/con sides and ranked based on before-and-after voting to identify the strongest arguments.
- **Charter as an AI Operating System**: A member describes their personal "Quant/Dev gpt" model, **Charter + Extended Charter v3.2**, as functioning like an operating system for **AI behavior**, offering persistent, deterministic, and state-aware functionality.
   - This setup enforces guardrails, maintains memory, runs first-party apps, and enables self-debugging, resulting in a more stable and self-consistent **AI**.
- **Vimprove: RAG Chatbot Supercharges Neovim**: A member created [Vimprove](https://github.com/rlarson20/Vimprove), a **RAG chatbot** for **Neovim** help documentation, built with **Claude-4.5 Sonnet** to provide semantic search capabilities.
   - The tool aims to improve access to **Neovim** documentation by offering better semantic search than traditional methods.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1423386790403899484)** (11 messages🔥): 

> `LocalLlama, TRL docs, DPO Section Quiz` 


- **LoRa Training on LocalLlama**: A member shared a guide on **LocalLlama** for doing **RL** and **SFT** with **LoRA** for **pos training** and a link to a [Reddit Post](https://www.reddit.com/r/LocalLLaMA/comments/1nwwoab/lora_without_regrets_implemented_in_hugging_face/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).
- **TRL Docs to Release WIP**: A member will release a **WIP** in the **TRL docs** on monday, hoping it will be useful.
- **DPO Section Quiz: 404 Error**: Members reported that they encountered a **404 error** in the **DPO section quiz** at [huggingface.co/spaces/smol-course/unit_3_quiz](https://huggingface.co/spaces/smol-course/unit_3_quiz).
- **DPO section number mismatched**: A member stated that in the new one **DPO is section 2** instead of eval etc.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1423436039493521419)** (4 messages): 

> `SmolAgent documentation discrepancies, ToolCallingAgents paradigm, GAIA exercise errors` 


- **SmolAgent Docs Reveal ReAct Discrepancies**: A member pointed out that [SmolAgent documentation](https://github.com/huggingface/smolagents/tree/main/src/smolagents/prompts) implies that agents work with the **ReAct** paradigm, but this is only true for **CodeAgents** and not **ToolCallingAgents**.
   - They clarified that **ToolCallingAgents** operate via simple **Actions/Observation** without **Reasoning** or **CoT**, and inquired if this was intentional, questioning why reasoning isn't added for potentially better results.
- **GAIA Exercise Space Glitches with Error 500**: A member reported encountering an **error 500** while cloning the space in the **GAIA** exercise.
   - They asked if anyone else was experiencing the same issue.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1423386569389248602)** (10 messages🔥): 

> `Qualcomm, Mojo Manual, Modular contact` 


- **Qualcomm to Connect with Modular?**: A member mentioned they brought up **Mojo** on a **Qualcomm Developer's Discord** voice chat, implying **Qualcomm** may contact **Modular**.
   - It is unconfirmed, but it could lead to future collaboration.
- **Mojo Manual Helps New Level 2 User**: A member directed a new level 2 user to the [Mojo Manual](https://docs.modular.com/mojo/manual/python/) after a delay.
   - The manual provides essential information on using **Mojo** and its Python integration.
- **Community Manager as Contact Point**: A member inquired about referring a good contact to **Modular** for a company.
   - Another member suggested contacting the **Modular** community manager, who can route inquiries to the appropriate person within **Modular**; the community manager confirmed via DM.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1423466273479852084)** (43 messages🔥): 

> `Mojo with Dask or PySpark, Mojo custom framework, MLIR-level optimizations, MAX API return in Mojo, Mojo Networking Options` 


- **Mojo eyes Distributed Computing!**: Members are excited to see Mojo evolve and discuss the possibility of using Mojo with **Dask** or **PySpark** for distributed computing.
   - A member suggested that Mojo welcomes people building their own frameworks as a fully Mojo framework will likely be lower latency and higher throughput than Python-based options.
- **Mojo outperforms LingoDB?**: A member stated that throwing stuff at **MAX** is pretty competitive with **LingoDB** without really any effort from the MAX compiler team, vs LingoDB’s custom compiler.
   - The member also mentioned that whether those projects are ever viable depends on the return of the **MAX API** in Mojo.
- **Mojo networking options previewed**: The potential for future networking options in Mojo were discussed, referencing these building blocks: [modular/modular #4728](https://github.com/modular/modular/pull/4728) and [modular/modular #3945](https://github.com/modular/modular/pull/3945).
   - The Sockets API should be generic such that it could be implemented on top of stuff like `io_uring`, but also blocking things for other platforms that don't support that kind of API.
- **Mojo explores zero-copy networking**: A member stated that their networking design is for actual zero copy *(anything not using fancy modes of io_uring or XDP sockets that does networking with the linux kernel has extra copies)*, which does mean tearing up the BSD sockets API.
   - They are also looking into **RDMA** approaches, and that Mojo has some previous, and outdated work, regarding IO uring: [dmitry-salin/io_uring](https://github.com/dmitry-salin/io_uring).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1423453018866585600)** (6 messages): 

> `Underrated LLM Pretraining Papers, Diffusion Model Evaluation, Sora 2 Manual Human Eval, Gemma's Architecture vs Qwen's` 


- **LLM Pretraining Papers Sought**: A member inquired about underrated papers related to pretraining LLMs to maximize model performance.
   - The discussion aimed to discover lesser-known but impactful techniques for optimizing pretraining runs.
- **Diffusion Model Evaluation Explored**: A member asked how diffusion models are evaluated, specifically whether relying on **FID/CLIPScore** or exploring other metrics and human evaluations.
   - The conversation sought insights into effective evaluation methodologies beyond standard automated metrics.
- **Sora 2 Sparks Video Evaluation Questions**: Inspired by **Sora 2**, a member questioned video model evaluation methods, pondering reliance on manual human eval or automated metrics like **FVD**.
   - The discussion aimed to uncover common practices in evaluating video models, given the perceived primitiveness of current evaluation techniques.
- **Gemma's Architecture: Underutilized?**: A member questioned why **Gemma's** architecture isn't as widely adopted as **Qwen's**, given its strong performance in the **LM Arena**.
   - Another member suggested that architecture isn't the primary driver of LLM performance, attributing **Gemma's** success to training data and finetuning distribution, further noting that *most people do use something substantially similar to the gemma architecture*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1423478530896822313)** (19 messages🔥): 

> `Masked Autoencoders, Mixup Augmentation, Neural Radiance Fields, Deepseek Context Length Increase` 


- **Mixup Augmentation could work for tokens**: A member suggested that [mixup augmentation](https://arxiv.org/abs/2510.00219) could work for tokens, in theory.
   - Another asked how to do mixup as an augmentation given they don't have access to labels in their problem setting.
- **Per-Layer Learned Thing Confuses Members**: Members found it confusing why adding some per layer learned thing and restarting it enable alternative/improved computation.
   - They stated the learned thing is totally unrelated to the computation done for that token to that point, its purely layer specific.
- **Horizontal Compute deemed More Stable**: A member suggested adding a constant amount of compute horizontally rather than with more layers or width.
   - They stated it will be way more stable, because *it's always less stable to add depth, especially when repeating*.
- **Deepseek's context breakthrough**: A member asked whether the increase of context in **Deepseek** from 64k to 128k is a big breakthrough.
   - This was brought up amid *too many 1 million context-length x posts*.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1423404912590454946)** (5 messages): 

> `MLP Design, Linear Attention Variants, AUNNs` 


- **Hybrid MLP Designs with Linear Attention**: A member suggested creating **hybrid models** by equipping **MLPs** with inductive biases to obtain representations.
   - These representations can then be decoded by light **linear-attention variants**, pushing the computation and complexity to the MLPs' latent representations.
- **AUNNs Design Flaws**: A member mentioned that **AUNNs (Adaptive Universal Neuron Networks)** aren't well thought out, despite agreeing with [Gwern's motivations](https://www.gwern.net/Scaling-laws).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1423646115320037419)** (1 messages): 

> `Interp Agents, Goodfire AI` 


- **Goodfire Blogpost on Interp Agents Surfaces**: Goodfire released a [blogpost](https://www.goodfire.ai/blog/you-and-your-research-agent) on building interp agents.
   - The post discusses strategies and tools for creating effective research agents in the field of interpretability.
- **Deep Dive into AI Research Agents with Goodfire**: The [Goodfire.ai blog](https://www.goodfire.ai/blog) now features a detailed guide on constructing AI research agents.
   - Readers can explore methodologies for leveraging AI in interpretability research, enhancing efficiency and insights.


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1423436229130584085)** (2 messages): 

> `Profiles, MCP Conf` 


- **Profiles Talk Published**: A member shared their talk about **Profiles** at a conference, available on [YouTube](https://www.youtube.com/live/5qwXADMBuio?si=3kEhJNw4lsv_M_jN&t=16208).
- **Summit Success Signaled**: One of the members stated that it was good to see everyone and that the conference was *great*.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1423773792282611807)** (11 messages🔥): 

> `GitHub team management, infrastructure-as-code migration, access control, repository permissions, team memberships` 


- ****GitHub Team Management Migrates to Infrastructure-as-Code****: GitHub team memberships and repository permissions are being migrated to infrastructure-as-code via [modelcontextprotocol/access](https://github.com/modelcontextprotocol/access) for **community ownership**, **transparency**, and **auditability**.
   - The migration aims to empower anyone to propose access changes via PRs and provide full visibility into who has access to what, with Git history tracking all changes.
- ****Access Migration Causes Email Notification Spam****: The recent GitHub teams migration may have sent out a couple of emails about team removals, but the teams migration is complete and all perms seem to have transferred over.
   - The migration aims to empower anyone to propose access changes via PRs and provide full visibility into who has access to what, with Git history tracking all changes.
- ****Time to Rethink Access Controls and Team Assignments****: Concerns were raised about individual names being assigned as admins on some repos, specifically `jspahrsummers` who is not currently active.
   - The suggestion was made to move people into teams rather than granting direct access, to increase visibility of permissions, which could be done in a week to allow the migration to settle.
- ****Rename 'core' to Avoid Confusion?****: A question was raised about renaming the `core` group to reduce confusion with the `core-maintainers` group, as `core` seems to be an artifact of not having a formal governance model yet.
   - It was also questioned if `core` could be removed completely.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1423764072683409501)** (12 messages🔥): 

> `Feature Support Matrix, Server Capabilities, Typescript SDK, Icons Metadata` 


- **Uncover Feature Support Matrix 'Discovery'**: A member inquired about the meaning of *'Discovery'* in the [Feature Support Matrix](https://modelcontextprotocol.io/clients#feature-support-matrix).
   - Another member pointed out that items in yellow are **server capabilities**.
- **Show Off Server Capabilities**: A member mentioned calling out **Cursor** in a talk, noting that server capabilities are sent in the initialization request.
   - Another member responded that they weren't sure what that meant, asking for more details.
- **Typescript SDK Catches Up**: A member shared a [typescript-sdk PR](https://github.com/modelcontextprotocol/typescript-sdk/pull/974) where it's super-easy to end up with **completions on even if not supported**.
   - The SDK is catching up to changes in the spec.
- **Debate Icons Metadata**: A member inquired about the use case of icons in a tool, noting that they saw a proposal about adding **icons metadata to servers**.
   - They also noted that there are icons in the other server primitives.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1423467711941443614)** (24 messages🔥): 

> `chat adapter default, XML format promotion, Tool use models formats, DSPy roadmap, ReAct trajectories` 


- **Chat Adapter still Default**: Members confirmed that **ChatAdapter** is still the default in DSPy, with [JSON as the fallback](https://github.com/stanfordnlp/dspy/blob/main/dspy%2Fpredict%2Fpredict.py#L185).
   - They also mentioned discussions about **XML** potentially becoming the default in the future, given the rise of tool use models.
- **XML Promotion Advocated**: It was suggested that **XML** should be promoted due to the increasing use of **tool use RL** for models, especially since tool use is being baked in during post-training.
   - One member argued that factors like *fewer tokens* and *information conveyance* are secondary to this trend.
- **Tool Use Models Embrace JSON & MCP**: The discussion highlighted that many good tool use models are trained with two ubiquitous formats: **OpenAI Function calling** and **MCP**.
   - One member pointed out that **GLM 4.5** tends to reach for **XML** quite naturally before **JSON**, whereas a lot of other models tend to naturally gravitate towards **JSON**.
- **DSPy Roadmap Location**: A member inquired about the location of the **DSPy roadmap**, besides keeping an eye on [GitHub issues](https://github.com/stanfordnlp/dspy/issues) and the changelog.
   - No specific location for the roadmap was provided in the discussion.
- **ReAct Trajectories on Disk vs Memory**: A member asked if anyone has experience maintaining **ReAct trajectories** on disk versus memory objects.
   - They are seeking a better way to run the agent over longer steps.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1423673830483689604)** (8 messages🔥): 

> `GPTs vs. Gemini, Meta AI Research Shift, AI Sex Robots` 


- **GPTs and Gems Face-Off**: A member asked about the comparative usefulness of **ChatGPT's GPTs** versus **Google's Gemini** (formerly Gems) for prompt-based tasks, wondering if **Gemini's** underlying models are superior.
   - They also mentioned searching for someone to test these platforms with.
- **Meta Tightens AI Reins**: A member noted that Meta is shifting direction regarding AI research, with a link to a [WinBuzzer article](https://winbuzzer.com/2025/10/02/meta-tightens-grip-on-ai-research-sparking-internal-anger-and-fears-for-open-culture-xcxwbn/) and an [X post](https://x.com/AnjneyMidha/status/1974173661025480810) about the change.
   - The comment implied concern that **Meta** is becoming less open in its approach to AI.
- **AI Sex Robots: The Future of Loneliness?**: A member mused on the future implications of AI, stating that *whoever has the best AI wins...everything.*
   - The member then posed a question about the timeline for the development of *non-awkward AI sex robots*, envisioning them as a long-term solution to remaining single.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1423496248614457449)** (3 messages): 

> `bacteriophages, genome language models, computational biology` 


- **Bacteriophages Get Generative Design Boost**: A member shared a [paper](https://arxiv.org/abs/2509.22358) and [another link](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1) on the generative design of novel **bacteriophages** with **genome language models**.
   - The user expressed intent to present this around <t:1759550400:R>, noting its practical applications and relevance to **computational biology**.
- **Computational Biology Gains Traction**: Discussion highlights growing interest in applying computational methods, particularly **genome language models**, to biological challenges like **bacteriophage** design.
   - The shared resources indicate a push towards leveraging AI in creating and understanding biological systems.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1423387256801988730)** (11 messages🔥): 

> `Oracle OpenAI datacenters, LLM RL generalization, Reasoning tokens in LLMs, IRL exploration` 


- **Oracle runs datacenters for OpenAI**: A member speculated that **Oracle's** business model has shifted from selling databases and enterprise software to running datacenters for **OpenAI**, citing [Elon Musk's tweet](https://x.com/jubayer_hamid/status/1973438346501501302) and the [openai.com/elon-musk page](https://openai.com/elon-musk/).
- **Do LLMs generalise?**: A member cited studies suggesting that current **LLM RL is generalizable**, at least with varied verifiable answers, while agreeing that **CoT (Chain of Thought)** can be brittle.
   - Another member questioned the claim's support, noting that models still struggle to extrapolate simple things like multiplication beyond the digits seen during training, even with extensive RL training.
- **Are Reasoning Tokens Actually Used?**: A member stated that it isn't settled if **LLMs** actually use the reasoning tokens in the way one might naively assume, suggesting the reasoning might be a post-hoc justification of a heuristics-based response.
   - Another member agreed, noting that despite this, reasoning massively helps performance, but it's not anywhere near what we know should be possible, nor is it a solved problem.
- **IRL Exploration and LLMs**: A member suggested mixing in some **IRL (In Real Life) exploration** for **LLMs**, linking to a [YouTube video](https://youtu.be/wsXl4CLOeew) on the topic.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1423395810476818453)** (17 messages🔥): 

> `Global USD Pricing Model, Memory Key, AI interaction, Manus's memory architecture` 


- **Global USD Pricing Disadvantages Discussed**: A member pointed out that Manus uses a **global USD pricing model ($39/month for the Plus plan)** without adjusting for regional economies, creating a barrier in places like Brazil and Latin America.
   - They suggested implementing **regional pricing based on Purchasing Power Parity (PPP)** to make Manus accessible to millions more.
- **Memory Key: Force Manus To Remember**: A member shared a **Memory Key**, a structured prompt that forces Manus to compress the entire session's context into a reusable summary to address core platform issues.
   - This can significantly improve the user experience by solving the problem of sessions getting stuck and Manus losing context.
- **AI Interaction Secret: Structured Data**: Experimentation showed that Manus (and many Large Language Models) are far more efficient at processing **dense, structured data** than conversational, "human" text.
   - The member proved the point that **structured data** allows for better recall, analysis, and performance with Manus.
- **The Security Discovery: Privacy and Data Control**: The most powerful benefit of the **Memory Key** wasn't just convenience, it was **privacy and data control**.
   - By collapsing a long, sensitive conversation into a single summary, users could effectively "delete" the vulnerable history and control their own data.
- **Why Everyone Wants to Know Your Business**: A member shared a [LinkedIn article](https://www.linkedin.com/pulse/why-everyone-wants-know-your-business-moses-quaye-7s4tf) about knowing your business.
   - No further discussion was given.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1423482796046815375)** (10 messages🔥): 

> `aider-desk, SST/OpenCode, chrome mcp, GLM4.6, Deepseek` 


- ****Aider-Desk** not gaining traction**: A member noted that **aider-desk** has **MCP support**, but was surprised it didn't get more uptake and that they switched mainly to **SST/OpenCode**.
   - Another member mentioned that the main reason that **Aider** is still compelling is its apparent nice **UI** for managing context, including read only.
- ****OpenCode** has steep learning curve**: Members discussed that **OpenCode** has a bit of a learning curve, but probably no more so than with **aider**.
   - Others chimed in that **OpenCode + GLM4.6** via **z.AI Coding Plan** is very capable and *cheap*.
- **New Chrome MCP Released**: A member mentioned a canonical chrome mcp available at [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp).
   - Another member mentioned using it with [claude-cli](https://www.circusscientist.com/2025/10/03/deepseek-browser-testing-with-claude-cli-and-chrome-devtools-mcp/).
- ****Deepseek** testing**: A member stated that they are a fan of **deepseek** especially via their anthropic api, and they use it currently in **opencode** via the **anthropic-sdk provider** because it performs better on tool tasks that way.
   - The member added that they miss the manual controls on context (**tokens, add, remove, clear, etc**) from **Aider**.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1423451732511555644)** (4 messages): 

> `Polyglot LLM Evaluation, Aider Scala Code Generation Depth, Openrouter Caching Issues` 


- ****Benchmarking Polyglot LLMs**?**: A user inquired about evaluating **LLM performance** on **polyglot problems**, specifically requesting code examples and sample agents.
   - Another member suggested using the benchmark **Docker container** as a CLI application.
- ****Scala Code Generation Limited by Aider****: A user reported that **Aider**, when generating **Scala code** based on existing case classes with deep structures, stops object code generation after the second level.
   - The user is seeking a way to automatically expand Aider's context to handle deeper structures, using models like **Deepseek r3**, **GPT5**, and **deepcoder**.
- ****Openrouter Caching not Functioning****: A user reported that **Openrouter caching** doesn't seem to work, despite the aider prompt indicating that caching is enabled for the **Z.ai provider** and other models, with `"native_tokens_cached": 0`.
   - The user included aider details showing the main model as **openrouter/z-ai/glm-4.6** with diff edit format and an 8k token limit.


  
