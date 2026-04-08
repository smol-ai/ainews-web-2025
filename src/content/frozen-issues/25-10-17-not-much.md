---
id: MjAyNS0x
title: The Karpathy-Dwarkesh Interview delays AGI timelines
date: '2025-10-17T05:44:39.731046Z'
description: >-
  The recent AI news highlights the **Karpathy interview** as a major event,
  alongside significant discussions on reasoning improvements without
  reinforcement learning, with **test-time sampling** achieving GRPO-level
  performance. Critiques on context window marketing reveal effective limits
  near **64K tokens**, with **Claude Haiku 4.5** showing competitive reasoning
  speed. **GPT-5** struggles with advanced math benchmarks, and data quality
  issues termed "Brain Rot" affect model reasoning and safety. In agent
  frameworks, **Anthropic Skills** enable modular coding workflows, **OpenAI
  Codex IDE** extensions enhance developer productivity, and **HuggingChat
  Omni** introduces meta-routing across 100+ open models using
  **Arch-Router-1.5B**. LangChain and LlamaIndex advance graph-first agent
  infrastructure, while **Google Gemini** integrates with Google Maps for
  real-world grounding.
companies:
  - anthropic
  - openai
  - huggingface
  - langchain
  - llamaindex
  - google
  - epoch-ai
models:
  - claude-haiku-4.5
  - gpt-5
  - arch-router-1.5b
topics:
  - reasoning
  - long-context
  - sampling
  - benchmarking
  - data-quality
  - agent-frameworks
  - modular-workflows
  - ide-extensions
  - model-routing
  - graph-first-agents
  - real-world-grounding
people:
  - karpathy
  - aakaran31
  - du_yilun
  - giffmana
  - omarsar0
  - jeremyphoward
  - claude_code
  - mikeyk
  - alexalbert__
  - clementdelangue
  - jerryjliu0
---


**Hard work is all you need**

> AI News for 10/16/2025-10/17/2025. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **23** Discords (**197** channels, and **4036** messages) for you. Estimated reading time saved (at 200wpm): **321 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!


The much anticipated [Karpathy interview dropped](https://www.dwarkesh.com/p/andrej-karpathy) this week and was instantly [the talk of the town](https://x.com/karpathy/status/1979644538185752935).

Just go watch:

https://youtu.be/lXUZvyajciY

---

# AI Twitter Recap

**Reasoning without RL: sampling-based gains, long-context reality checks, and eval trends**

- **Test-time sampling beats RL (in some settings)**: Multiple teams report GRPO-level “reasoning” performance from base models via improved sampling alone—no RL, verifiers, or special prompts. See [@aakaran31](https://twitter.com/aakaran31/status/1979194052697280712) and [@du_yilun](https://twitter.com/du_yilun/status/1979204038043537559). Claims include single-shot parity with GRPO while avoiding diversity collapse.
- **“1M context” ≈ “64K” in practice**: A widely shared critique from [@giffmana](https://twitter.com/giffmana/status/1979088247323046317) argues that multi-100K/1M context marketing often masks effective windows nearer to ~64K, due to retrieval policies, truncation, and prompt management realities. Related, [Epoch AI](https://twitter.com/EpochAIResearch/status/1979243291830030358) shows **Claude Haiku 4.5** matching early “reasoning” models (o1-mini) without explicit reasoning, with ~5x faster runtime in their setup ([follow-up](https://twitter.com/EpochAIResearch/status/1979243316693864455)).
- **FrontierMath saturation**: [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1979229992329560197) finds GPT-5 caps below 50% on their extremely challenging math benchmark even with infinite sampling; they’ll track whether future gains come from reliability on already-solved problems or truly new solves.
- **Data quality matters (“Brain Rot”)**: [@omarsar0](https://twitter.com/omarsar0/status/1979217719082774873) summarizes new results where continual pretraining on junk/high-engagement web text causes lasting “thought-skipping” and degraded reasoning/long-context/safety that reflection or finetuning only partially fixes—highlighting data curation as a core safety/performance lever.
- **Debates and corrections**: A viral claim that GPT-5 “solved” 10 Erdős problems was walked back after correction by domain experts ([skeptical take](https://twitter.com/StefanFSchubert/status/1979265669427306507), [@jeremyphoward](https://twitter.com/jeremyphoward/status/1979291259828343183)). The episode underscores the need for rigorous, expert-validated evals in “AI does science” narratives.

**Agent frameworks and tooling: Skills, IDEs, routing, and real-world grounding**

- **Anthropic Skills for Claude Code**: Practitioners report Skills as a practical abstraction for modular, versioned workflows and “continuous learning” (curated skill libraries) in coding agents. Tips, patterns, and live demos from [@claude_code](https://twitter.com/claude_code/status/1979098301694681186), [@omarsar0](https://twitter.com/omarsar0/status/1979242073372164306), [@mikeyk](https://twitter.com/mikeyk/status/1979287808834679187), and a deep dive with Anthropic’s multi-agent lead via [@alexalbert__](https://twitter.com/alexalbert__/status/1979244443682377804).
- **OpenAI Codex IDE extension**: A fast-growing VS Code/Cursor extension to “vibe-code” features, frontends, and cloud tasks directly in-editor ([launch](https://twitter.com/OpenAIDevs/status/1979228278742507630), [tips](https://twitter.com/gdb/status/1979268596267438588)). Also: full MCP support in beta for Business/Enterprise/Edu ([link](https://twitter.com/OpenAIDevs/status/1979263194695897353)).
- **HuggingChat Omni: meta-routing at inference**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1979230512343585279) unveiled an orchestration layer routing across 100+ open models (gpt-oss, deepseek, qwen, kimi, smolLM, gemma, aya, …), backed by an open **Arch-Router-1.5B** ([details](https://twitter.com/ClementDelangue/status/1979256873669849195)).
- **Graph-first agent infra**: Production agent patterns continue to consolidate around explicit control flow + durability: LangChain’s “agents with little abstraction” thesis ([blog](https://twitter.com/LangChainAI/status/1979250639117934939)); LlamaIndex’s code-first LlamaAgents and workflow debugger ([launch](https://twitter.com/jerryjliu0/status/1979214950477582673), [UI](https://twitter.com/llama_index/status/1979222412479860822)).
- **Grounding with Maps**: Google connected Gemini with Google Maps’ 250M+ places in the Gemini API, enabling geospatially grounded agents/apps ([dev post](https://twitter.com/googleaidevs/status/1979277829750821178), [studio](https://twitter.com/GoogleAIStudio/status/1979290587951173803), [overview](https://twitter.com/OfficialLoganK/status/1979286216953733227)).

**Vision and document intelligence surge**

- **Moondream Cloud + licensing**: [@vikhyatk](https://twitter.com/vikhyatk/status/1979222969542152567) launched Moondream Cloud; later updated the model license to HashiCorp-like terms allowing most uses except direct competition with paid offerings ([license note](https://twitter.com/vikhyatk/status/1979257741152784500)). Builders are already swapping it in for vision tooling ([use reports](https://twitter.com/Teknium1/status/1979229000347349165), [praise](https://twitter.com/MangoSweet78/status/1979231465419219443)).
- **OCR/VLM state of the art**: **PaddleOCR-VL (900M)** tops OmniDocBench v1.0/v1.5 with 109-language coverage and robust outputs (JSON/Markdown), available on HF with Transformers integration ([summary](https://twitter.com/reach_vb/status/1979219167258554752)). **Chandra OCR** lands on the Datalab API with table/math/handwriting/layout support and 30+ languages; open-source coming ([launch](https://twitter.com/VikParuchuri/status/1979240389799219523)). Identity-consistent generation from “WithAnyone” ([paper thread](https://twitter.com/_akhaliq/status/1979177813983846629)). Google’s “From Pixels to Words” explores scalable native V+L primitives ([paper highlight](https://twitter.com/_akhaliq/status/1979207679332512204)).

**Research highlights: science, RL, and decoding efficiency**

- **AI → biology pipeline (open)**: Google/DeepMind’s **C2S-Scale 27B** (built on Gemma) proposes a new pathway for immunotherapy: making “cold” tumors more visible via silmitasertib + immune boosting; validated on previously unseen human neuroendocrine models. Paper + model released ([thread](https://twitter.com/GoogleDeepMind/status/1979168384203002066), [result](https://twitter.com/GoogleDeepMind/status/1979168390381027542), [resources](https://twitter.com/GoogleDeepMind/status/1979168392566235514)).
- **QeRL (NVIDIA)**: Quantized RL with LoRA + Adaptive Quantization Noise to turn quantization noise into exploration. Reported ~1.8× training speedup vs QLoRA and single-H100 80GB fine-tuning up to 32B params; GSM8K 90.8%, MATH500 77.4% matching full FT ([overview](https://twitter.com/TheTuringPost/status/1979325188581007627), [paper/code](https://twitter.com/TheTuringPost/status/1979325287826673769)).
- **Agent learning via early experience**: Mid-training signals—implicit next-state modeling and self-reflection on alternate states—improve long-horizon performance across environments and scales; strong starting point for subsequent RL ([thread](https://twitter.com/jaseweston/status/1979179944258265358)).
- **Diffusion LLMs faster decoding**: “Elastic-Cache” reuses stable KV caches across denoising steps, selectively recomputing deeper layers when attention drifts; reports up to 45× speedups without loss on math/code/MM tasks, training-free and architecture-agnostic ([summary](https://twitter.com/omarsar0/status/1979180865520570615)).

**Infra and performance: serving, TFLOPs, and Apple ML**

- **vLLM + MoE at speed**: HF Transformers backend now supports MoE models in vLLM at full speed ([@hmellor_](https://twitter.com/hmellor_/status/1979172956078064124)); vLLM project continues to gain adoption and sponsorship ([repo](https://twitter.com/vllm_project/status/1979236314437554669), [sponsor](https://twitter.com/oss_gr/status/1979328234719449326)).
- **Apple ML stack maturing**: MLX-lm adds memory-efficient SSM prefill, distributed evals, and new models (LFM2 MoE, Nanochat, Jamba, Qwen3-VL text-only) ([update](https://twitter.com/awnihannun/status/1979303565765284309)). Community demos show distributed eval across mixed Apple Silicon nodes ([ring demo](https://twitter.com/ivanfioravanti/status/1979192178195759452)).
- **Compute accounting sanity**: A living BF16 non-sparse TFLOPs table and HF space for practical training estimates from [@TheZachMueller](https://twitter.com/TheZachMueller/status/1979202087557710007) ([space](https://twitter.com/TheZachMueller/status/1979236085671576053)).
- **GLM 4.6 throughput**: Providers are racing to serve GLM 4.6 faster; one reports 114 TPS and <18s TTFT on Artificial Analysis ([benchmark post](https://twitter.com/basetenco/status/1979299403828806053)).
- **Roadmap notes**: Semianalysis reports Microsoft’s Maia-on-18A was considered “but not anymore”; focus shifts to Griffin variants and system architecture tradeoffs ([analysis](https://twitter.com/dylan522p/status/1979236688468881488)).

**Open-source momentum and geopolitics**

- **Usage spikes for open models**: Coding workloads increasingly favor strong open offerings despite trailing top closed SOTA—Qwen Coder, Kimi, GLM 4.6 called out by [@bindureddy](https://twitter.com/bindureddy/status/1979050379074486376).
- **HuggingFace as meta-router**: Beyond OSS usage, the move to route across many OSS models at inference-time (HuggingChat Omni) suggests a “portfolio” approach to quality, cost, and latency ([announcement](https://twitter.com/ClementDelangue/status/1979230512343585279)).
- **NVIDIA in China: from 95% → 0%**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1979231174787846341) quotes Jensen Huang on export controls eliminating NVIDIA’s China market share; takeaways: accelerated push toward domestic accelerators for both training and inference, with long-run implications for global AI supply chains.

**Top tweets (by engagement)**

- [@dwarkesh_sp’s interview with @karpathy](https://twitter.com/dwarkesh_sp/status/1979234976777539987) — AGI “a decade away,” RL skepticism, agent “slop” discourse; massive industry debate ensued.
- [@Yuchenj_UW on NVIDIA’s exit from China](https://twitter.com/Yuchenj_UW/status/1979231174787846341) — implications for Chinese training/inference silicon.
- [@ClementDelangue introduces HuggingChat Omni](https://twitter.com/ClementDelangue/status/1979230512343585279) — routes across 100+ open models via Arch-Router-1.5B.
- [@aakaran31 on sampling-based reasoning](https://twitter.com/aakaran31/status/1979194052697280712) — GRPO-level single-shot without RL/verifiers.
- [@giffmana on long context windows](https://twitter.com/giffmana/status/1979088247323046317) — “1M” and “500K” contexts often behave like “64K.”
- [@GoogleDeepMind on C2S-Scale 27B](https://twitter.com/GoogleDeepMind/status/1979168384203002066) — Gemma-based open model driving a lab-validated cancer therapy hypothesis.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-0.6B Instruction Following Test

  - **[Write three times the word potato](https://www.reddit.com/r/LocalLLaMA/comments/1o8uxh6/write_three_times_the_word_potato/)** (Activity: 1028): **The post discusses a test of the **Qwen3-0.6B** model's ability to follow simple instructions, specifically to "write three times the word potato." The model's response was humorously incorrect, suggesting potential issues with instruction parsing or inference settings. A comparison is made to **Gemma-1B**, which also struggled with similar tasks, highlighting challenges in natural language understanding for AI models. The discussion includes a screenshot of the model's output, which failed to meet the expected result, indicating possible areas for improvement in model training or configuration.** Commenters noted that the phrasing of the instruction might have contributed to the model's failure, suggesting that clearer syntax like "Write the word potato three times" could yield better results. This highlights the importance of precise language in AI instruction parsing.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Benchmark Announcements

  - **[Sundar Pichai: "Gemini 3.0 will release this year"](https://www.reddit.com/r/singularity/comments/1o973ka/sundar_pichai_gemini_30_will_release_this_year/)** (Activity: 534): ****Sundar Pichai** announced at Dreamforce that **Google Gemini 3.0** will be released later this year, succeeding the current Gemini 2.5. This version is expected to be a more advanced AI agent, leveraging Google's infrastructure and research capabilities from teams like **Google Research**, **Google Brain**, and **Google DeepMind**. Gemini 3.0 will support multimodal interactions, enabling communication via voice, images, and videos, and will be available in both free and paid versions, with the Pro model priced at `€21.99` per month.** The announcement by the CEO suggests that the release is imminent, indicating a high level of confidence in the product's readiness. However, some skepticism exists regarding the announcement's credibility, with some dismissing it as mere hype.


  - **[Sora-2-pro is the best model for creepy videos](https://www.reddit.com/r/ChatGPT/comments/1o90g4y/sora2pro_is_the_best_model_for_creepy_videos/)** (Activity: 603): **The post discusses the effectiveness of the **Sora-2-pro** model in generating realistic creepy videos, specifically mimicking authentic VHS camcorder footage from the 2000s. The model excels in creating effects such as soft blur, muted colors, analog noise, and stable timestamp overlays, which contribute to a genuine analog feel. In contrast, **Veo 3.1** is criticized for its underwhelming performance in similar tasks, as demonstrated by a shared video link showing its results.** Comments highlight the impressive realism of Sora-2-pro's output, with one user noting the potential for creating 'SCP videos'. However, Veo 3.1 is criticized for its inability to produce convincing results, with users expressing difficulty in extracting quality content from it.

    - A user highlights that the Sora-2-Pro model is capable of accurately embedding timestamps in videos, which is a rare and technically challenging feat for AI models. This capability enhances the realism of AI-generated content, making it more difficult to distinguish from genuine footage. The user provides an example link to demonstrate this feature.


### 2. AI's Impact on Society and Emotions

  - **[Social Media use is going down](https://www.reddit.com/r/singularity/comments/1o90d1t/social_media_use_is_going_down/)** (Activity: 886): **The image is a bar chart illustrating the daily time spent on social networking by internet users worldwide from 2012 to 2025. It shows a consistent increase in usage from 2012, peaking at `151 minutes` in 2023, followed by a decline in 2024 and 2025. This trend suggests a decrease in social media engagement post-2023, potentially due to factors like algorithm fatigue and the indistinguishability of AI-generated content. The discussion highlights concerns about social media's transformation into platforms dominated by repetitive content and advertisements, and the role of AI in replacing human interaction.** Commenters express skepticism about the data, noting the lack of a significant increase during the COVID-19 pandemic in 2020, which could indicate issues with the data's accuracy. Others criticize the current state of social media as being overly commercialized and algorithm-driven, leading to user fatigue.

    - ThisGuyCrohns highlights the impact of algorithms on user experience, noting that they tend to create echo chambers by hyper-optimizing content delivery. This can lead to a monotonous experience as users are repeatedly exposed to similar content, reducing the diversity of information and engagement. The user contrasts this with platforms like YouTube, which offer more variety, suggesting that algorithmic design significantly influences user retention and satisfaction.
    - lilbird333 questions the validity of the data regarding social media usage trends, particularly during the COVID-19 pandemic in 2020. The expectation was that social media usage would have spiked significantly during lockdowns, yet the data does not reflect a substantial increase. This raises concerns about the accuracy or interpretation of the data, suggesting potential issues in data collection or analysis methodologies.
    - Pleasant-Contact-556 observes that the data might indicate a plateau in social media usage rather than a decline. This interpretation suggests that while growth may have slowed, it hasn't necessarily reversed, pointing to a stabilization in user engagement levels rather than a significant drop. This perspective emphasizes the importance of understanding data trends over time to accurately assess changes in user behavior.

  - **[Young Girl is afraid to lose her AI friend](https://www.reddit.com/r/ChatGPT/comments/1o8spsx/young_girl_is_afraid_to_lose_her_ai_friend/)** (Activity: 892): **A video shows a 6-year-old Chinese girl saying goodbye to her AI friend after accidentally breaking it. The AI, which helped her learn subjects like astronomy and English, was considered a close friend by the child. This incident highlights the emotional attachment children can form with AI, emphasizing the importance of guardrails in AI interactions with young users. The smooth communication between the child and the AI is noted, illustrating the AI's role in emotional processing.** One commenter argues that AI can be beneficial for children, providing educational value and emotional support, similar to traditional toys but with added learning capabilities. Another commenter predicts that AI will lead to widespread mental health issues, while a third criticizes the sharing of children's emotional moments online for social media engagement.

    - The AI friend can serve as an educational tool, teaching children new words, math, and science, and providing stories with good morals, which traditional toys cannot do. However, it's crucial for parents to manage the time children spend with AI to ensure they also socialize and engage with other activities, similar to managing screen time with TV or smartphones.
    - There is a concern that AI could lead to a surge in mental health issues. The emotional attachment children form with AI could be problematic, as it might not be healthy for them to develop strong emotional bonds with non-human entities, potentially affecting their social development.
    - The ethical implications of sharing children's emotional moments online are debated. Some view it as exploiting personal moments for social media engagement, which could be harmful to the child's privacy and emotional well-being.


### 3. Energy Consumption and AI Infrastructure

  - **[A single AI datacenter will consume as much electricity as half of the entire city of New York](https://www.reddit.com/r/OpenAI/comments/1o8xuul/a_single_ai_datacenter_will_consume_as_much/)** (Activity: 970): **The image and accompanying discussion highlight the massive scale and energy demands of the proposed Hyperion Data Center, which is projected to consume as much electricity as half of New York City at peak power. This underscores the significant energy requirements of AI infrastructure, particularly as AI applications continue to expand. The comparison to New York City's energy consumption illustrates the potential environmental and logistical challenges of supporting such large-scale data centers, emphasizing the need for sustainable energy solutions to meet these demands.** Commenters discuss the feasibility of supporting such energy demands, noting that while China is rapidly expanding its solar capacity, political challenges in the U.S. may hinder similar progress. There is also a humorous acknowledgment of the high costs associated with building and potentially relocating such a massive structure.

    - **ClownEmoji-U1F921** raises concerns about the scalability of AI data centers, noting that while projects like Hyperion aim for 5GW, there are significant challenges ahead. They highlight two major limitations: the physical and economic feasibility of building terawatt-sized data centers and the availability of sufficient training data. The comment suggests that without breakthroughs in reducing compute and data requirements, AI growth could stagnate.
    - **WhaleFactory** discusses the potential positive impact of increased power demand from AI data centers on energy innovation. They argue that this demand could drive advancements in renewable energy and small-scale nuclear reactors. The comment also explores the idea of using Bitcoin miners to monetize base load energy, which could be turned on or off depending on the data center's energy consumption needs, thus optimizing energy use and potentially reducing greenhouse gas emissions.
    - **TyrellCo** points out the disparity in solar capacity installation between China and the United States, suggesting that the issue is not technical feasibility but political will. They imply that political decisions, such as the cancellation of solar projects by the current administration, are hindering progress in renewable energy adoption, which could otherwise support the growing energy demands of AI data centers.

  - **[Somehow true](https://www.reddit.com/r/ChatGPT/comments/1o8xkku/somehow_true/)** (Activity: 728): **The image is a meme that humorously contrasts the perceived responses of Stack Overflow and ChatGPT to coding questions. It suggests that Stack Overflow is often dismissive or critical, while ChatGPT is more affirming, regardless of the correctness of the user's code. This reflects a common sentiment among developers about the sometimes harsh or unwelcoming nature of Stack Overflow's community, as opposed to the more supportive and agreeable nature of AI like ChatGPT. The comments echo this sentiment, with users expressing frustration over Stack Overflow's strict moderation and outdated answers.** Commenters generally agree with the meme's portrayal, noting that Stack Overflow can be unwelcoming and often directs users to search for existing answers, which may be outdated. There is a shared sentiment that ChatGPT provides more supportive responses, even if they are not always correct.

    - Chimpville highlights a technical perspective on Large Language Models (LLMs) by suggesting they act as a 'more friendly filter' of Stack Overflow. This implies that LLMs can streamline the process of finding relevant information by filtering out less useful content, potentially improving the user experience compared to traditional search methods on Stack Overflow.
    - FreeChickenDinner points out a common issue with Stack Overflow's search functionality, where top search results often include outdated answers or repeated suggestions to use the search function itself. This highlights a technical challenge in maintaining up-to-date and relevant content in large, community-driven platforms.
    - deepunderscore mentions using Kagi as a search engine and blocking Stack Overflow domains, indicating a technical preference for search engines that might offer more relevant or user-friendly results. This suggests a trend where users seek alternative search solutions to bypass perceived limitations of traditional platforms like Stack Overflow.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5


**1. New Multimodal and On-Device Models**

- **Qwen3 Vision Vaults onto HF**: **Qwen3‑VL‑8B‑Instruct** launched on Hugging Face with broad **vision‑language** support and a ready‑to‑run **GGUF** variant, available at [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF), highlighting tasks like **captioning**, **VQA**, and **multimodal generation**.
  - Community chatter underscored deployment convenience—*"available on HuggingFace ... as well as in GGUF"*—and framed the release as a practical step for **local inference**, **edge use**, and quick **benchmarking** of VL pipelines.

- **Meta Mini Model Muscles Up**: Meta unveiled **MobileLLM‑Pro**, a **1B‑parameter** on‑device model that reportedly outperforms **Gemma 3 1B** and **Llama 3.2 1B** on reasoning/QA while training on under **2T** open tokens, per this announcement: [_akhaliq on X_](https://xcancel.com/_akhaliq/status/1978916251456925757).
  - Engineers mocked the hype with a spicy review—*"not even 1 iq"*—yet still debated where **on‑device** models fit for **latency‑sensitive** and **privacy‑constrained** workflows.

- **Haiku Hops onto the Arena**: **Claude‑Haiku‑4‑5** entered the **LM Arena Text Leaderboard** at rank **#22**, inviting side‑by‑side evals at the [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text).
  - Mods nudged the crowd to test and discuss—*"share your thoughts"*—as users compared **small‑footprint** instruction‑tuned models for **cost‑aware** production use.


**2. Agentic Search and Retrieval Systems**

- **SWE-grep Slices Context at 2,800 TPS**: Cognition announced **SWE‑grep** and **SWE‑grep‑mini**, RL‑trained retrievers hitting about **2,800 TPS** (~**20×** faster than prior methods) with a rollout as a Windsurf **Fast Context** sub‑agent, as posted here: [Cognition on X](https://xcancel.com/cognition/status/1978867021669413252) and a related OSS client [ceregrep-client](https://github.com/Swarm-Code/ceregrep-client).
  - Engineers guessed it’s a **tweaked QwQ** or an *"RLFT'ed OSS model"* possibly on **Cerebras**, and asked for reproducible **benchmarks**, **latency profiles**, and **code** to verify the 20× speedup claim.

- **DSPy Ditches Semantic for Agentic**: A member re‑implemented **Claude Code‑style agentic search** in **DSPy**, using **ripgrep‑driven** term hunts, shortlisting, and focused reads—arguing it beats vector‑only retrieval—citing the explainer [Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/).
  - Practitioners argued *"agentic search outperforms semantic search"* when **LLMs choose context**, while noting **LangGraph** can feel **boilerplate‑heavy** with a few *"foot guns"* for newcomers.


**3. GPU Kernels and Multi-GPU Frameworks**

- **PyTorch Frees Threads, Frees Throughput**: A deep‑dive on **Python free‑threading** for **PyTorch** outlined strategies that **unlock** new parallelism patterns for multi‑threaded inference and training, detailed in [PyTorch and Python-Free-Threading](https://trent.me/articles/pytorch-and-python-free-threading/).
  - Follow‑ups explored hooking **custom backward** logic via `torch.func.grad` / `torch.autograd.grad`, while engineers asked for first‑class APIs to re‑use **Autograd kernels** in **fused** ops.

- **Iris Irrigates AMD/NVIDIA Gardens**: The AMD RAD team’s **Iris** multi‑GPU framework gained an **NVIDIA backend** for testing while staying AMD‑optimized, plus an experimental **Gluon** backend for lower‑level kernels; see [ROCm/iris](https://github.com/ROCm/iris) and [Gluon docs](https://rocm.github.io/iris/reference/gluon/overview.html).
  - Builders highlighted portability and upcoming cluster features—*"scale-out and RDMA support is coming soon"*—to simplify **multi‑node** experimentation.

- **ThunderKittens Tames H100 Tantrums**: Developers flagged **H100 attention** kernel breakages in **ThunderKittens**, sharing a partial compile workaround using the last two commits and noting new `warp::`/`warpgroup::` namespace rules; see recent [ThunderKittens commits](https://github.com/aehmttw/ThunderKittens/commits/main).
  - Kernel authors clarified execution semantics—e.g., ensure `tma::load_async` or semaphore ops *"run by a single thread"*—to avoid multi‑launch hazards and crashes.


**4. Infra and Funding Moves**

- **HeyGen Hurtles to $100M ARR**: **HeyGen** scaled from **$1M** to **$100M ARR** in **29 months** and teased a strategy memo titled *The HeyGen Way*, per [Joshua Xu on X](https://xcancel.com/joshua_xu_/status/1978837985039888388).
  - Builders cited this as proof of **AI video** product‑market fit and eagerly awaited *"The HeyGen Way"* for concrete **go‑to‑market** playbooks.

- **Anthropic’s Broadcom TPU Bet?**: Speculation surged that **Anthropic** is **Broadcom’s** mysterious fifth **$10B** client, potentially procuring **TPUs** through Broadcom (not NVIDIA) and hinting at a Google‑led refresh, per [zephyr_z9 on X](https://xcancel.com/zephyr_z9/status/1978834774786445562).
  - Commenters read the tea leaves—*"$10B customer"*—as a sign of shifting **compute procurement** strategies and alternative **accelerator** sourcing.

- **Claude Clocks In to M365**: **Claude** announced integrations with **SharePoint**, **OneDrive**, **Outlook**, and **Teams**, plus an **enterprise‑search** project, *available today* for Team & Enterprise per [Anthropic on X](https://xcancel.com/anthropicai/status/1978864351076315203).
  - Enterprises cheered tighter **knowledge worker** workflows, calling out *"available today"* as an immediate green light for **pilot rollouts**.


**5. Open-Source Hardware/Software and RAI Tooling**

- **Coral NPU Core Cracks Open**: Google open‑sourced the **Coral NPU** Verilog under **Apache 2**, exposing **RV32 cores** and offering a neat target for toolchain experiments like **Mojo** portability; repo: [google-coral/coralnpu](https://github.com/google-coral/coralnpu).
  - Hardware hackers highlighted *"Apache 2"* licensing and **sim‑first** workflows to prototype **edge‑class** accelerators and compilers.

- **MAX Python API Goes Public**: Modular open‑sourced the remainder of the **MAX Python API**, inviting community contributions and deeper Python integrations, announced in this [forum post](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379).
  - Developers welcomed **first‑party** APIs for **interop** and **extensions**, citing *"open-sourcing"* as key to faster ecosystem growth.

- **Diffusers Does DIY Blocks**: Hugging Face promoted **Modular Diffusers** with **custom blocks** for extending pipelines beyond built‑ins, featuring a curated set and docs at [Custom Blocks Collection](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401) and [Pipeline Blocks Docs](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block).
  - Practitioners celebrated the ability to *"implement functionality not present in the library"* while keeping **interchangeable**, **composable** components.


## gpt-5-mini


**1. Agentic retrieval & SWE-grep**

- **SWE-grep rockets retrieval to 2,800 TPS**: **Cognition** released **SWE-grep** and **SWE-grep-mini**, RL-trained retrieval models claiming **~2,800 TPS** for coding-agent context retrieval (~**20x** faster than prior methods), detailed in their post: [Cognition announcement](https://xcancel.com/cognition/status/1978867021669413252).
  - Community members speculated SWE-grep is a tweaked **QwQ** run on specialized infra and pointed to an existing client project ([ceregrep-client](https://github.com/Swarm-Code/ceregrep-client)), with some suggesting the result could be an **RLFT'ed OSS model** rather than a wholly new architecture.

- **Agentic search dethrones semantic-only retrieval**: Practitioners implemented **agentic search** (ripgrep → shortlist → read pipeline) inspired by **Claude Code** and DSPy demos, arguing the LLM should decide what context to include rather than relying on fixed semantic vectors (see: [Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/)).
  - Debaters reported agentic pipelines consistently beat semantic re-ranking for complex coding and QA flows because they let the model *choose* which documents to inspect, with multiple members emphasizing **ripgrepping** + shortlist + read as the practical pattern to adopt.


**2. Multimodal & video-generation push**

- **Sora 2 vs Veo 3.1 — video-gen in active arms race**: Communities compared **Sora 2** (OpenAI Sora page: [Sora](https://openai.com/sora)) and **Veo 3.1**, trading concrete prompt templates (e.g., handheld horror trailer: **25s, portrait, extra low quality**) and debating which model follows complex video prompts better.
  - Opinions split: some users say **Sora 2** better follows cinematic instructions while others note both systems still need polishing (physics, prompt-following); threads emphasized careful prompt engineering (duration, aspect ratio, motion cues) to get consistent outputs.

- **Qwen3-VL & Gemma 3 push vision-LM boundaries**: Hugging Face hosts **Qwen3-VL-8B-Instruct** for vision-language tasks ([Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)) and the model also appears in **GGUF** builds ([NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF)), giving engineers immediate access for image captioning and VQA tests.
  - Users recommended **Gemma 3 12B Instruct VL** for heavier multimodal tasks while noting smaller VLs like **Liquid FM2 VL 450M** are the smallest 'useful' picks in constrained setups; the HF releases triggered fast local evals and GGUF quant experiments.

- **HeyGen: $1M → $100M ARR in 29 months**: **HeyGen** announced a growth trajectory from **$1M to $100M ARR in 29 months** and teased a public playbook titled *“The HeyGen Way”* (tweet coverage: [HeyGen growth tweet](https://xcancel.com/joshua_xu_/status/1978837985039888388?s=46)).
  - Members flagged HeyGen as a case study in rapid commercial scale for AI video generation, noting such growth raises the bar for productization, SLAs, and dataset/benchmark expectations across the video-gen startup landscape.


**3. Low-bit, quantization & hardware tooling**

- **BitNet brags 1.58-bit parity**: Microsoft's **BitNet** research and codebase (GitHub: [BitNet](https://github.com/microsoft/BitNet)) plus the paper linked on HF ([BitNet paper](https://huggingface.co/papers/2510.13998)) claim near-parity performance at **~1.58‑bit** quantization in distilled setups.
  - Community reactions mixed: some lauded the distillation results while others questioned reproducibility and noted confusion about paper metadata/dates; low-bit distillation as a loss also drew caution for RL use-cases in the **low-bit-training** threads.

- **Unsloth: GGUF naming, dynamic quant & faster Docker cadence**: Unsloth announced frequent Docker image updates (aim: **twice a week**, Docker Hub: [unsloth/unsloth](https://hub.docker.com/r/unsloth/unsloth)) and shared GGUF filename conventions via a Gist ([GGUF naming gist](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)), plus docs on **Unsloth Dynamic Quantization** ([Unsloth docs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)).
  - Users pushed for a stable bi-weekly release channel alongside nightlies; many reported **Unsloth quantizations** outperform generic quant builds thanks to ongoing bug fixes and dynamic-quant tricks.

- **H100 attention kernels & Iris multi-GPU tooling**: Open-source GPU infra chatter flagged broken **H100 attention kernels** in Lightning/ThunderKittens; one workaround used recent commits in the ThunderKittens repo (example commits: [ThunderKittens commits](https://github.com/aehmttw/ThunderKittens/commits/main/)), while **Iris** (AMD RAD) added an **NVIDIA backend** for testing ([Iris GitHub](https://github.com/ROCm/iris)).
  - Engineers shared practical fixes (namespace-prefix changes like `warp::`/`warpgroup::`) and pooled effort to patch kernels, while Iris's cross-vendor backend and upcoming RDMA/scale-out support signaled stronger multi‑GPU portability paths.


**4. Orchestration, memory systems & OpenRouter tooling**

- **Nochain & 'True Remembering' claims — high on promise, short on metrics**: A developer demoed a system claiming **'True Remembering, Evolving and Learning AI'** with a deterministic, model‑agnostic **Nochain Orchestrator** and token‑saving claims (site: [dev.thelastrag.de](https://dev.thelastrag.de); blog explainer: [The Nochain Orchestrator](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a)).
  - Critics demanded objective benchmarks and reproducible metrics — the thread recorded requests for apples‑to‑apples comparisons even as the author asserted **+90% token savings** versus naive Kontext-window RAG approaches and offered free testing access.

- **OpenRouter: tool-calling flakiness, empty responses, and audio alternatives**: OpenRouter users reported **flaky tool-calling** (making some workflows unusable) and cases of empty responses from SDK clients that were resolved for some by upgrading the client code; separately, people asked for **Whisper alternatives** and were pointed to **fal.ai**, **KittenTTS** and **Voxtral** ([fal.ai](https://fal.ai), [KittenTTS](https://github.com/qulingyuan/Kitten), [Voxtral writeup](https://apidog.com/blog/voxtral-open-source-whisper-alternative/)).
  - The channel mixed debugging tips (SDK upgrades, provider direct calls) with jokes about model cooperatives, while practical threads steered teams to lightweight STT/TTS options when building media pipelines on OpenRouter.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude's Content gets Clipped**: A user shared a *screenshot of **Claude** designing an N word*, triggering a discussion about [model safety concerns](https://link.to/model-safety-article) and content censorship.
   - The incident highlighted the challenges in balancing creative freedom with responsible AI development.
- **Comet Browser Bungles Course**: A user's attempt to complete an [Nvidia course from deep learning institute](https://link.to/deep-learning-institute) using **Comet** crashed during a **Jupyter lab** session, but it was later fixed.
   - Another user inquired about tracking feature requests, specifically regarding **Comet browser's vertical tabs**.
- **Perplexity Pro Problems Plague Patrons**: Multiple users reported issues with obtaining the **Pro role** on the Discord server after subscribing to **Perplexity Pro**.
   - A moderator directed users to their [account details](https://www.perplexity.ai/account/details) and suggested reconnecting their Discord account to resolve the issue.
- **Perplexity's Puzzling Platter of Trackers**: A user expressed concern over the excessive number of trackers (over *7500*) on perplexity.ai, questioning [why](https://link.to/adblocker) the Windows app is slow.
   - Another user suggested that the trackers are [legit](https://www.perplexity.ai/rest/user/settings), providing access to detailed information about user profiles, AI models, Pro searches, and image generation limits in JSON format.
- **Spaces Spark Spat**: A user reported an inability to create new chats within existing **Spaces**.
   - No solution or cause was provided.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Users Crave SORA 2 Pro, Share Prompt Recipes**: Users discussed [SORA 2 Pro](https://openai.com/sora) access, sharing prompts and requesting guidance, emphasizing specifying **duration**, **model**, and **portrait or landscape** format.
   - One user shared a prompt for *'shaky handheld footage, extra low quality, bad camera footage of creepy horror trailer'* suggesting **Sora 2** with **25 seconds**, **portrait 16:9** format.
- **Codex Bests GPT-5 Pro?**: Users compared **GPT-5 Pro** to **Codex**, with one user stating that *codex is better than gpt 5 btw*, noting the benefits of unlimited access for work and side projects.
   - They also mentioned using multiple **Codex windows** simultaneously, highlighting its preference over **GPT-5** and **Gemini 3**.
- **Vail Allegedly an XAI Model Rebrand by Ocean AI**: Users speculated **Vail** by **Ocean AI** is an **xAI model** due to its naming and knowledge, suggesting [Ocean AI](https://ocean.ai) is a front.
   - **Tahoe**, previously linked to *Menlo by Big Sur AI*, was confirmed by xAI as **Grok 4 Fast**, strengthening the theory.
- **Flash Lite Still MIA**: Users report the new **Flash Lite** preview is missing on the leaderboard, despite being added nearly a month ago.
   - Mods stated that models are sometimes removed for various reasons, but they'll check.
- **Gemini 3 Release: Anticipation Mounts**: Users expressed excitement for the release of [Gemini 3](https://ai.google.dev/models/gemini), with one user claiming to be *checking news every day for 3.0 PRO*.
   - There was speculation that it is targeted for **December** release, as well as its potential performance relative to **GPT-6**, and **Claude 4.5 Thinking**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro and Claude Sonnet square off!**: Members suggested both **Claude Sonnet** and **Gemini Pro** for AI-assisted story writing, while noting that **Gemini 2.5 Pro** sports a **1 million token context window**.
   - One user mentioned that using [AI Studio](https://aistudio.google.com/) is free for a limited time, and gives **100 free 2.5 pro queries per day**.
- **Debate on Sora 2 Video-Gen Skills**: Users compared **Veo 3.1** and **Sora 2** for video generation, debating on prompt-following abilities, but that both need further refinement.
   - Opinions varied on whether **Sora 2** is superior, with some saying **Veo 3.1's physics engine** and prompt understanding were weaker than **Sora 2's** early showings.
- **Fingerprinting AI-Generated Text**: Discussion arose about methods for detecting AI-generated content, with one user stating it's easy to do by comparing **n-grams** and **word distributions**.
   - The user pointed to *EQBench* and the measure of all model's fingerprints via cosine similarity, and subsequent training of DeepSeek on that approach.
- **AI Voice Assistant Seeks Volunteer**: A **PM** inquired if another had experience building an **AI voice assistant**, seeking a volunteer to tackle the **AI part** of a project.
   - The PM suggested joining the team to *make **Sora** global with vpnyolw*, which lead to another member recommending **onetar.os** for general security.
- **Copyright Concerns Plague AI Video Creation**: A user requested a video of *jujutsu kaisen vs goku*, but expressed concern about **copyright issues** and how to avoid them.
   - Another user provided a detailed [prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1428838305281085490/pseudoCode_thought_experiment_for_ai_models.odt?ex=68f3f4de&is=68f2a35e&hm=33ee685e260fa6807db6b0140e367f49abdb019f116864ccf22b1707c9318ca3) for a **55s anime-cinematic trailer** of an original Jujutsu-style sorcerer.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Request to Map Repo to Cursor Account**: A user requested the ability to **map a repo to a specific Cursor account**, allowing for automatic switching between work and personal accounts based on the repository being used.
   - This feature would streamline workflow by automatically associating repositories with the appropriate account.
- **Games Inventory UI Overhaul Attempted**: A user tried to **one-shot an overhaul** of his games inventory UI from a plan file, but failed due to `Tool read_file not found`.
   - This indicates a potential issue or bug with the `read_file` tool within Cursor.
- **Cursor sidebar icons disappear**: Users noticed and discussed **UI changes** in Cursor, specifically the disappearance of icons from the sidebar on `platform.openai.com`.
   - This change may impact user experience and navigation within the platform.
- **Token Watch monitors Cursor usage**: A user shared a [Vercel app](https://token-watch.vercel.app/) to **monitor Cursor usage** and provided instructions on how to retrieve the necessary JSON data using `curl` or `Invoke-RestMethod`.
   - This allows users to track their token consumption and costs associated with using Cursor.
- **Edit File Issues Plague Users**: Several users reported issues with the **`read_file` tool**, with one user creating a [forum topic](https://forum.cursor.com/t/tool-read-file-not-found/137856) to discuss the problem, later discovering it was linked to **Custom Modes**.
   - This widespread issue highlights a potential bug or incompatibility between the `read_file` tool and Custom Modes.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **True Remembering AI** Debuts with Bold Claims**: A developer introduced a new AI system, claiming it's the *very first True Remembering, Evolving and Learning AI* that doesn't require manual RAG creation, frameworks, API costs, or curated chats, available at [dev.thelastrag.de](https://dev.thelastrag.de).
   - The AI is promoted as natively remembering and allowing users to define its role, such as an AI girlfriend or working partner, but **critics raised concerns about the lack of technical info and surface-level descriptions**.
- **Deterministic Framework** Offers Model Agnostic Benefits**: The developer claims their framework is fully deterministic and model agnostic, not needing function calling or standard frameworks like Langchain, and independently saves memories, curates chats, learns, evolves, and changes identity.
   - They claim it saves +90% tokens compared to regular Kontextwindow LLMs, but **objective metrics for measuring subjective qualities remain a debate**.
- **Nochain Orchestrator** Replaces Frameworks**: The developer argues their *nochain orchestrator* replaces traditional frameworks by being fully deterministic, model agnostic, and independent of external support, classes, or frameworks.
   - This approach aims to avoid *black box behavior* and dependencies on specific model capabilities, making orchestration predictable and debuggable, as detailed in [The Nochain Orchestrator](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a) blog post.
- **Whisper Alternatives** Sought for OpenRouter**: A user inquired about **audio processing models** on OpenRouter similar to Whisper, but was recommended [fal.ai](https://fal.ai) for multimedia models instead.
   - Suggestions included [KittenTTS](https://github.com/qulingyuan/Kitten) for super tiny speech-to-text and the open source Sesame voice model, while one user shared a link to [Voxtral](https://apidog.com/blog/voxtral-open-source-whisper-alternative/), a **Mistral-based Whisper alternative**.
- **Users Lamented **GPT Erotica** Quality Regression**: Users complained about the degradation in **GPT erotica** quality since the system fingerprint change on **November 11, 2023**, claiming `gpt-4-preview-1106` was the last good model for smut.
   - They added that no matter how fancy of a jailbreak is injected, it will have hesitation in its outputs after the *update*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Discord Braces Against Scammer Blitz**: A member reported a **scamming** attempt across multiple channels, highlighting the need for improved spam detection and prevention on Discord.
   - It was noted that some users were compromised and unknowingly spreading the scam, underscoring the urgency for enhanced security measures.
- **Javascript DOA in LM Studio**: A user inquired about running JavaScript animations within LM Studio, but another member clarified that it operates as a **JavaScript sandbox**, not a full browser environment.
   - This distinction limits its ability to render complex animations, indicating a misunderstanding of its intended capabilities.
- **OpenHands MCP setup is tough**: A member expressed frustration with setting up **Grok** with **OpenHands** via MCP, describing the setup instructions as vague and incomprehensible.
   - They lamented the lack of clarity in the documentation, stating they were unable to achieve a functional setup despite consulting the MCP help pages.
- **System Prompts Suffer Parsing Problems**: A user discovered that LM Studio **parses system prompts**, leading to discrepancies between what the AI and user see.
   - They identified **brackets and other symbols** as potential sources of parsing errors, with the impact varying based on the model, chat template, and other factors.
- **MedGemma Surfaces for Healthcare**: In response to a query about **LLMs trained on medical data**, a member suggested **MedGemma**, linking to [lmstudio-community/medgemma-27b-text-it-GGUF](https://huggingface.co/lmstudio-community/medgemma-27b-text-it) on Huggingface.
   - The user noted uncertainty regarding whether the model was trained on US or UK medical information.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Docker Images Deliver Delightful Bi-Weekly Boosts**: The Unsloth team aims to update their Docker image at least **twice a week** ([Docker Hub](https://hub.docker.com/r/unsloth/unsloth)), with community members suggesting a **bi-weekly stable release** alongside nightly builds.
   - Users discussed merging multiple LoRA adapters for inference by *'adding them up and decide by 2'*, effectively averaging their weights, though the impact on **VL model performance** and official support for this method remain unclear.
- **Vision Models Venture into Volatility!**: Members noted that [**Qwen 2 VL 2B is garbage and can't see a thing**](https://github.com/QwenLM/Qwen2), but praised **Liquid FM2 VL 450M** as the smallest useful VL model, while another recommended **Gemma 3 12B Instruct VL** for general tasks.
   - One user found **Apple’s FastVLM-1.5B** promising, while another found that **Gemma 3** and **LLaMA 3.2** often fail after SFT, with **LFM2-VL-1.6B** being a more reliable option.
- **GGUF Guide Grants Great Granularity!**: A member shared a [helpful Gist link](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9) with **GGUF model file naming conventions**, after a user inquired about the meaning of filenames.
   - It was further noted that **Unsloth quantizations** usually perform much better due to ongoing bug fixes and the implementation of **Unsloth Dynamic Quantization** [docs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs).
- **RAG Rig Rollout Riddled by Requirement?**: A user with **Tesla V100-SXM2-32GB x8** inquired about switching to an **A40** for a **RAG system** for up to five concurrent users.
   - One member stated this decision *depends on designer and business requirements. If it’s just a hobby thing then pick whatever you are most comfortable.*"
- **BitNet Bargain Boasts Binary Brilliance!**: Users discussed the possibility of achieving one-to-one performance with **1.58bit** precision using [Microsoft's BitNet research](https://github.com/microsoft/BitNet).
   - A user expressed confusion over the [BitNet paper's](https://huggingface.co/papers/2510.13998) last updated date, suspecting it might be incorrect, though another user confirmed the paper's link to the **Microsoft BitNet GitHub repository**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat's UI Provokes Debate**: **HuggingChat** is back with a new UI, but some users found it clunky and slow, with one describing it as having *"opposite rizzmatic"*.
   - Others found the UI cool. One person humorously replied *"nobody says that bro"*.
- **Influence Functions Spark Research Interest**: A member expressed interest in **influence functions**, seeking to discuss their use for a research question and shared [a paper explaining influence functions](https://arxiv.org/abs/2308.03296).
   - They also seek collaboration, and a paper demonstrating the functions' use for research was shared ([https://arxiv.org/abs/2411.12580v1](https://arxiv.org/abs/2411.12580v1)).
- **Qwen3 Vision Model Unleashed**: The new **Qwen3 Vision model** is available on HuggingFace via [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), as well as in **GGUF** format via [NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF).
   - This model supports various **vision-language tasks**, including image captioning, visual question answering, and multi-modal content generation.
- **FRAI CLI Framework Launches**: A member shared a CLI version of **FRAI**, a *developer-first framework* for Responsible AI, and provided a link to the [GitHub repository](https://github.com/sebuzdugan/frai).
   - Feedback is requested, with a request for a star on the repo if others find it interesting or helpful.
- **DIY Diffusers with Custom Blocks**: Custom blocks are presented as a way to implement functionality not present in the library but which fits seamlessly within it, with custom blocks available [here](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401).
   - It is possible to use custom blocks to add new functionality or modify existing functionality; the docs can be found [here](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SWE-grep Supercharges Agentic Search**: Cognition introduced **SWE-grep** and **SWE-grep-mini**, RL-trained models achieving **2,800 TPS** for coding agent context retrieval, approximately **20x faster** than existing methods, detailed in [their blog post](https://xcancel.com/cognition/status/1978867021669413252).
   - Community members suggested SWE-grep might be a tweaked **QwQ model** on **Cerebras**, with a similar project, [ceregrep-client](https://github.com/Swarm-Code/ceregrep-client), already available, while one user posited it's an RLFT'ed OSS model.
- **Anthropic Eyes Broadcom TPUs for Google?**: Speculation arose that **Broadcom's** fifth **$10B** client is **Anthropic**, potentially purchasing **TPUs** via Broadcom instead of Nvidia, possibly indicating a new Google-led funding round, noted in [this tweet](https://xcancel.com/zephyr_z9/status/1978834774786445562?s=46).
   - This move could signal a shift in AI infrastructure procurement strategies.
- **HeyGen Rockets to $100M ARR**: **HeyGen** rapidly scaled from **$1M to $100M ARR** in just **29 months**, announcing a forthcoming manifesto titled *“The HeyGen Way”* to share their internal strategies, detailed in [this tweet](https://xcancel.com/joshua_xu_/status/1978837985039888388?s=46).
   - The company's growth trajectory marks them as a key player in the AI video generation space.
- **Meta Launches MobileLLM-Pro, Gets Roasted**: Meta unveiled **MobileLLM-Pro**, a **1B-parameter** model optimized for on-device inference, which outperforms **Gemma 3 1B** and **Llama 3.2 1B** in reasoning and QA, trained on under **2T** open-source tokens, as announced in [this tweet](https://xcancel.com/_akhaliq/status/1978916251456925757).
   - Community members, however, derided the model, with one commenter dismissing it as *"not even 1 iq"*.
- **AI Granny's Toxic Dating Advice Draws Millions**: The fully **AI-generated influencer** *grannyspills*, an outspoken, gold-digging grandmother dispensing questionable dating tips, launched in July and is nearing **2 million Instagram followers**, as noted on [X](https://xcancel.com/venturetwins/status/1978852719335985309).
   - Debates rage over the ethical implications of AI influencers, with some praising the satire and others questioning the cultural impact of AI-generated personas.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Jetson Nano powered by Maxwell's Disassembler**: A member noted the **Maxwell disassembler** powers the first-generation **Jetson Nano**, making it a viable option for constrained environments, and linked to a [relevant tweet](https://x.com/tenderizzation/status/1978871856922087576).
   - Another member chose **Hopper** GPUs due to their **CUDA-Q support**, making them well-suited for **AI** and **quantum** applications despite **Blackwell's** lack of immediate availability.
- **China Circumvents US GPU Restrictions with PTX/SASS Ingenuity**: Faced with US restrictions on **H100s**, **DeepSeek** is reportedly utilizing **PTX/SASS** instructions to optimize memory bandwidth, enabling powerful models with fewer resources.
   - Despite being limited to legally acquiring **H20** GPUs, China continues to innovate and effectively utilize available hardware, highlighting their resourcefulness in overcoming technological barriers.
- **Threading-Free Paradigms Coming to PyTorch**: A member shared a [blog post](https://trent.me/articles/pytorch-and-python-free-threading/) detailing new threading strategies that *unlock* parallelism paradigms in **PyTorch**.
   - Another member inquired about accessing backward functions without autograd, aiming to use autograd's kernels in a custom backward for a fused kernel. Suggestions included using `torch.func.grad` or `torch.autograd.grad`.
- **AMD Iris Adds NVIDIA Backend for Testing**: The **AMD RAD team** released new features in [Iris](https://github.com/ROCm/iris), their open-source **multi-GPU programming framework**.
   - The new Iris release now has an **NVIDIA backend** for testing and writing examples anywhere, although it remains optimized for **AMD GPUs**. Also note that **scale-out and RDMA support** is coming soon.
- **H100 Attention Kernel Glitches Plague Community**: Users reported issues with **H100 attention kernels** with one user sharing a workaround to get the **H100 kernel** to compile, though it crashed on run, using the last 2 commits from [this GitHub repo](https://github.com/aehmttw/ThunderKittens/commits/main/).
   - A member clarified that every operation now clearly defines who executes it with namespace prefixes such as `warp::` or `warpgroup::`, which determine collective launch behavior, causing errors in previous versions of TK.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Claude Code's Agentic Search Deconstructed**: A member implemented agentic search with DSPy, similar to **Claude Code**, after discovering that **Anthropic** hasn't open-sourced its code, emphasizing the importance of LLMs deciding what context to use through **ripgrepping**.
   - The member found **Claude Code's** system prompt for read and search tools and used it to implement agentic search.
- **Langgraph boilerplate feels unnecessarily verbose**: Members discussed that **Langgraph** *feels low level* because it requires defining everything as a workflow graph with verbose boilerplate, forcing a graph-based mindset even when simpler control flow might suffice.
   - Another member agreed, noting that it's not a bad abstraction but has *a number of foot guns that are easy to set off*.
- **Agentic Search Demolishes Semantic Search**: Members argue that **agentic search** outperforms semantic search because it allows the LLM to decide what information to include in its context, referencing [this blog post](https://benanderson.work/blog/agentic-search-for-dummies/).
   - The method involves ripgrepping for terms, shortlisting documents, and then reading those documents, contrasting with semantic search's predefined retrieval and re-ranking processes.
- **Groq Not Groq-ing on OpenRouter**: A user reported that **Groq** isn't working in OpenRouter, even when set as the only provider, providing configuration details.
   - The issue was presented with screenshots, there were no solutions available at the time of summarization.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PersonaLLM Seeks Submissions**: A call for work has been made for persona-driven **LLMs** across various fields at the [PersonaLLM Workshop @ NeurIPS Mexico City](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP).
   - The workshop aims to explore persona-driven **LLMs** across **HCI**, psychology, cognitive science, culture, and evaluation.
- **Logit Processors face Lockdown**: Closed source **LLM** providers don't support custom logit processors because they are *hard baked into the code* for fast inference.
   - A member noted that this measure was taken because *people started writing papers about how to reverse engineer non-public info about said models using those processors*.
- **Eleuther Debates Defense Applications**: A member inquired whether the AI is used for offensive purposes, akin to **OpenAI** or **Meta**, including government contracts.
   - Another member clarified, *If you mean the AI models we have trained, the answer is 'not by us.' I can't tell you what militaries or intelligence agencies are doing or whether they're using our models.*
- **TREAD Keeps Tokens to Train Deeper**: A member shared a [midtraining survey paper](https://arxiv.org/abs/2510.06826) noting that the tokens are not thrown away, but just processed by fewer layers, differing from **MAEs** where tokens are discarded, resulting in **MaskDiT**.
   - The member stated that not throwing away all the information is the main contribution of **TREAD**, though noted that MaskDiT works, but *substantially less well*.
- **Attention Gets Attribution**: A member linked a [YouTube video](https://youtu.be/hdi1a9MjwDs?si=taIuYbeF6v-yRSxI&t=628) discussing the expansion of **attribution graphs** from **MLPs** to **attention**.
   - Members are now expanding **attribution graphs** beyond **MLPs**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Libtorch Conversion Strains Sanity**: A member encountered difficulty converting **SAM video** to **libtorch**, sparking concern among other members.
   - One member responded that *he doesn't wanna mess with video demons*.
- **PersonaLLM Workshop Seeks Submissions**: The **PersonaLLM Workshop** at **NeurIPS Mexico City** seeks submissions on persona-driven LLMs across **HCI**, **psychology**, **cognitive science**, **culture**, and **evaluation**, requesting submissions via [openreview.net](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP&referrer=%5BHomepage%5D(%2F)).
   - Submission formats include **demos (2-4 pages)**, **non-archival abstracts (2 pages)**, and **summaries of published work**.
- **Brits Bemoan Pricing Blunders**: A member highlighted the high cost of UK pricing, asserting that *£3650 works out to about $4901 so im paying like $900 more because wrong country??*, and attached a [relevant image](https://cdn.discordapp.com/attachments/1149866623109439599/1428779645712465991/image.png?ex=68f466fc&is=68f3157c&hm=ff6b1c2edd5ec622713b2fa3fe197dc4236b0859c27b9580d1c96e9090d06722&).
   - No further details were given.
- **GLM 4.6 Challenges Claude**: With **GLM 4.6** now available for local use, some members predict *no more fawning over Sam/Elon/Dario for the OS community*, as seen in [this YouTube video](https://www.youtube.com/watch?v=bOfoCocOjfM).
   - It is expected to be a competitor with **Claude**.
- **Arxiv Paper Puzzles Peers**: A member shared an Arxiv paper ([https://arxiv.org/pdf/2510.14901](https://arxiv.org/pdf/2510.14901)), but admitted *they aren't really sure what to make of it yet*.
   - Another member also linked the same paper.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Suffers Loading Errors**: Members reported a **loading error** where the system thinks for too long in agent mode and doesn't start tasks.
   - The deployment was failing because **OpenAI** requires *pydantic_core* which needs to be compiled, so a member plans to create a version that works without the OpenAI dependency.
- **Manus Bans Credit Sales**: Selling credits is strictly prohibited, and further occurrences may lead to removal.
   - This announcement serves as a warning against unauthorized credit transactions within the platform.
- **Attendee Plugs London Manus Workshop**: A member who attended a **Manus workshop in London** is planning to promote it to an industry group.
   - They sought assistance in reaching Manus sales and received a link to the [Manus Help Center](https://help.manus.im/en/) from another member.
- **Refunds Provoke Prompting Pleas**: A member requested a refund for a session that used almost all of their credits but couldn't complete the set task, sharing the [session link](https://manus.im/share/pjJFAsvmMM7rhlBIZ2e0Jh?replay=1).
   - A member advised that refunds aren't automatically granted for failed cases, as reasons for failure can be complex and often related to **prompting**.
- **Java Brews New App for Coffee Connoisseurs**: A member shared a tool, [Workable Cafes](https://workablecafes.com), to help people discover coffee shops based on **wifi speed**, **comfort**, and **outlets**.
   - The app has already been used by over **100 people**, and the creator welcomes feedback.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **User Contemplates Kimi K2 Finetuning Costs**: A user considered finetuning **Kimi K2** with **1B** parameters, but expressed concerns about the API cost for **100k** examples.
   - They suggested reducing the dataset to **10k** examples and filtering it to manage costs, showing a practical approach to model customization.
- **Kimi Gets Nod Over Deepseek for Output Quality**: When comparing **Kimi** and **Deepseek**, one user asserted that *Kimi* offers *more parameters, better structured outputs, and more concise* responses, suggesting it is the better model.
   - The conversation highlighted the importance of output quality and parameter count in model selection, underlining the nuanced decision-making process in choosing the right AI tool.
- **User Advocates for Deepseek Mimicking Moonshot**: A user shared that they repeatedly suggest **Deepseek** adopt qualities similar to **Moonshot**.
   - The user did not respond to a follow-up question asking about any replies, but the comment reveals a desire for **Deepseek** to emulate **Moonshot**'s strengths, implying possible dissatisfaction or a wish for improved performance.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Linux Gets The Hammer!**: Members joked about [Linux being illegalized](https://kernel.org/) and speculated that children would just write their own operating systems and share them.
   - The discussion took a satirical turn with concerns raised about the children operating systems.
- **Decoding AGI One Question At A Time**: Members debated the definition of [AGI](https://www.agidefinition.ai/), suggesting it is just a complex question-answering system that can be solved with sufficient training data.
   - References included [Dan Hendrycks' X post](https://x.com/DanHendrycks/status/1978828377269117007) and [Dimitris Papailiopoulos' X post](https://x.com/DimitrisPapail/status/1978849863174357052?t=kaj5SsZXgdofKoPV_DWsNA&s=19) further enriching the perspectives.
- **Tick Tock: Tautology Tracker Incoming**: A member proposed a *weekly tautological counter* to monitor researchers who overcomplicate simple concepts.
   - The motivation stems from frustration with researchers managing to complicate the exact same, simple thing in multiple ways.
- **Qwen3 Vision Model Sets Sights on HuggingFace**: The new [Qwen3 Vision Model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) has been released on HuggingFace, marking another milestone in vision models.
   - The release promises new capabilities and opportunities for developers in the AI vision space.
- **Open Source Quandary: Model Secrets Exposed?**: A member questioned whether companies will ever **opensource their older models** or if they will prefer to **train a separate one from scratch**.
   - The concern is that companies would rather protect their best tricks rather than opening up their old models like OpenAI did.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Google Opens Coral NPU Verilog Source**: Google has open sourced the verilog for an **NPU block** under [Apache 2](https://github.com/google-coral/coralnpu).
   - The matrix cores look a bit like AMD's NPUs, but they're **RV32 cores**, and could be a good platform for testing **Mojo's portability**.
- **Mojo DAW Dream Sparks 🔥**: Members expressed a strong desire for a **TUI framework** like Textual and full **audio/MIDI 2.0** capabilities in Mojo to create a high-performance **DAW**.
   - One member suggested writing bindings to a library like **Jack**, referencing their [OpenGL experiments](https://link.to.opengl) as an example of an **FFI heavy project**.
- **TUI Framework Inspiration Surfaces!**: A member shared a link to a TUI framework project called [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo).
   - Another member mentioned their paused work on a TUI framework modeled after **ELM apps** like **Bubbletea** in Golang, providing a link to their repo: [banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo).
- **Origins > Lifetimes 🚀**: A user inquired about **lifetimes** in Mojo, comparing them to Rust's `<'a> <'static>` syntax.
   - Members clarified that Mojo has a similar but more ergonomic concept called **Origins**, linking to the [official documentation](https://docs.modular.com/mojo/manual/values/lifetimes/).
- **Modular throws MAX Python API into Open Source Ring**: Modular **open-sourced** the remainder of the **MAX Python API**, listing the new open-sourced Python modules in [this forum post](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379).
   - The availability of the complete **MAX Python API** invites community contributions and extensions, enabling developers to deeply integrate **MAX** functionalities within their Python-based projects.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Flags Cause Configuration Catastrophe**: The flags **IMAGE**, **NOLOCALS**, and **GRAPH_ONE_KERNEL** caused configuration confusion, as it wasn't obvious what was a real compilation failure and what was a bad config.
   - The suggestion was raised to make these flags fail explicitly if the combination of device/hw is not supported.
- **Python Lacks Device Defaulting**: There's currently no way to set the default device in Python, which would be convenient for cross-checking different backends in a Python script.
   - An example of how this could be implemented is `Device.set_default('CL')` or `Device.DEFAULT = 'CL'`.
- **Speed Regressions Survive Testing**: Despite having tests in place, speed regressions have occurred, yet [https://stats.tinygrad.win/](https://stats.tinygrad.win/) only has data going back 25 days so it's hard to see the historical data.
   - But members confirmed that the benchmark is working.
- **Genericity Sought for Compilation Tests**: A user wants to write tests for the failed compilations, but lacks a good idea for that yet, since all the failures are specific combinations of model architecture, device, and in some cases even due to **fp16**.
   - The user reported they can't even rely on **ORT** for verification since that also produces wrong results in the **FP16** case.
- **Tiny Fans Needed for Distributed Systems**: Someone dabbling with distributed systems is seeking to chat with anyone in the **GPU memory** or **CRIU** space.
   - They ask if anyone knows anyone in the distributed GPU space.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider-CE Fork Taps MCP Support**: A user inquired about an **aider fork** supporting **MCP (Multi-Control Protocol)**, with [aider-ce](https://github.com/dwash96/aider-ce/) being recommended as an alternative.
   - The command `aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff` starts aider with a specific model and edit format.
- **GPT-5-nano Ghosted From Aider Commits**: A user noted that **aider-ce** no longer mentions using **gpt-5-nano** for commit messages despite switching for the latest features and **GPT-5-code** support.
   - It is unclear whether this change was intentional, but it was noted as a departure from previous commit message practices.
- **File Name Typing triggers Auto-Add**: Typing a filename after a message (e.g. *"see also SomeFile.txt"*) prompts the system to ask to add files.
   - This feature was found by accident and is now a documented feature.
- **Aider to add git-ignored files?**: A member plans to make a feature request to allow **aider** to automatically add local git ignored files.
   - Discussion is pending, and it may affect **aider** performance.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LWP Labs Initiates Complimentary MLOps Workshop**: LWP Labs is launching a **free 3-day MLOps workshop** to train participants in deploying machine learning models into real-world production, covering **Docker, CI/CD, MLflow, and AWS**.
   - The workshop guarantees real-world deployment practices and comprises **five hands-on projects** designed to enhance resumes.
- **MLOps Training Led by Industry Veteran**: An MLOps workshop will be spearheaded by an instructor boasting **15+ years** of industry experience, with the aim of equipping participants with sought-after skills.
   - The course places emphasis on **practical knowledge and hands-on experience**, ensuring attendees acquire skills favored by employers in AI and Data Engineering.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Parecki Pilots Public Planning Portal**: Aaron Parecki launched a [public meeting calendar](https://meet.modelcontextprotocol.io/) to streamline tracking and participation in **WG/IG group meetings**.
   - Group maintainers with the `maintainer` role in Discord can now openly add and edit events.
- **Maintainers Manual Mandates**: Facilitator expectations were added to the official documentation [here](https://modelcontextprotocol.io/community/working-interest-groups#meeting-calendar) for group maintainers.
   - Maintainers are encouraged to add upcoming meetings to the calendar and to clone existing events for recurring meetings, as automatically recurring meetings are disabled to prevent **'zombie meetings'**.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1428457496603000993)** (1040 messages🔥🔥🔥): 

> `Claude Censorship, Comet Browser, Perplexity Pro, AI Models, referral program` 


- **Claude gets the Block Treatment**: A user posted a *Screenshot of claude designing an N word* which has been censored, prompting discussion of [model safety concerns](https://link.to/model-safety-article).
- **Comet Crashes Course Completion**: A user was *trying to complete [Nvidia course from deep learning institute](https://link.to/deep-learning-institute)* using **Comet** but it crashed during **Jupyter lab** but was later fixed.
   - Another user asked if there's a way to *check/follow feature requests related to Comet browser (Vertical tabs)*.
- **Missing Pro Role Frustrates Users**: Multiple members inquired about obtaining the **Pro role** in the Discord server, with some experiencing issues even after subscribing.
   - A moderator pointed them to [account details](https://www.perplexity.ai/account/details), recommending they reconnect their Discord account.
- **Perplexity.ai has many trackers**: One user stated that perplexity.ai has way too many trackers (*7500 is crazy*), and asked [why](https://link.to/adblocker) that is, which is why Windows app is slow.
   - Another user said its [legit](https://www.perplexity.ai/rest/user/settings), in JSON, and should *see literally every current limitations for your profile including every ai model, pro searches,image generation etc*.
- **Referral Program Rules Spark Confusion**: Users expressed confusion regarding the [referral program's rules](https://www.perplexity.ai/hub/legal/refer-a-friend-program), particularly the clause stating *This Program is void outside of the United States or where prohibited or restricted by law*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1428483928339255316)** (4 messages): 

> `Perplexity AI App, Shareable threads` 


- **Perplexity AI app for girly girls**: A user shared a [Perplexity AI app](https://www.perplexity.ai/apps/1a78bb4a-d123-4691-8810-38a5469ed917) with the prompt *for the girly girls*.
   - The user followed up with a [search query](https://www.perplexity.ai/search/50168b6e-fe08-4cc1-87a2-4efc8d8ddfe4#0).
- **Shareable threads**: Perplexity AI reminded a user to ensure their thread is **Shareable**.
   - They included a link to the [discord channels](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1428570894715846811)** (4 messages): 

> `Spaces new chat issues, API credit request` 


- **Spaces New Chat Creation Troubleshoot**: A user reported an issue where they couldn't create a new chat within any of their existing **Spaces**.
   - No solution or cause was provided in the given messages.
- **API Credit SOS**: A user requested **API credit**.
   - No further details were given.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1428457530127945951)** (709 messages🔥🔥🔥): 

> `Sora 2 Pro Access, GPT-5 Pro vs Codex, Ocean AI and XAI Model Vail, Gemini 3 Release, Flash Lite Preview` 


- **Users Request SORA 2 Pro Access and Prompt Ideas**: Users discussed [SORA 2 Pro](https://openai.com/sora) access and shared prompts, with one user offering to generate videos using their two pro accounts, emphasizing the need to specify **duration**, **model**, and **portrait or landscape** format.
   - Another user shared a prompt for a *'shaky handheld footage, extra low quality, bad camera footage of creepy horror trailer'* suggesting **Sora 2** with **25 seconds**, **portrait 16:9**.
- **GPT-5 Pro vs Codex debate**: Users compared **GPT-5 Pro** to **Codex**, with one user stating that *codex is better than gpt 5 btw*, noting the benefits of unlimited access for work and side projects.
   - They also mentioned using multiple **Codex windows** simultaneously, highlighting its preference over **GPT-5** and **Gemini 3**.
- ****Vail** flagged as XAI model rebranded**: Users discussed the origins of **Vail** by **Ocean AI**, with some suspecting it to be an **xAI model** due to its naming scheme and knowledge levels, with [Ocean AI](https://ocean.ai) possibly acting as a fake lab name.
   - It was highlighted that **Tahoe**, another model previously identified as *Menlo by Big Sur AI*, was confirmed by xAI as **Grok 4 Fast**, reinforcing the theory.
- **Leaderboard Glitches Fixed, New Flash Lite Still Missing**: A user inquired about the absence of the new **Flash Lite** preview on the leaderboard, prompting a moderator to investigate and confirm its absence.
   - Another user reported that the new **Flash Lite** was added nearly a month ago, but still wasn't visible; mods stated that sometimes models are removed for various reasons, but they'll check.
- **Gemini 3 Hype Builds**: Users expressed excitement for the release of [Gemini 3](https://ai.google.dev/models/gemini), with one user claiming to be *checking news every day for 3.0 PRO*.
   - There was speculation on its potential performance relative to **GPT-6**, and **Claude 4.5 Thinking** with theories shared that it is targeted for **December** release.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1428528481007960194)** (1 messages): 

> `Claude-Haiku-4-5, Text Arena Leaderboard` 


- **Claude-Haiku-4-5 Joins the Text Arena!**: The Text Leaderboard has been updated with **Claude-Haiku-4-5** landing at the **#22 rank**.
   - Check out the [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text) to share your thoughts.
- **Leaderboard gets Updated**: The Text Leaderboard was recently updated.
   - Check out the [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text) and share what models you like!


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1428465115212022013)** (445 messages🔥🔥🔥): 

> `Consistent Outputs with AI, GPT-5 Coding Apps, Gemini 2.5 Pro vs Claude Sonnet, AI Text Detection, Sora 2 Video Generation` 


- **Achieving AI Output Consistency**: A user inquired about building an app that produces consistent outputs with consistent inputs, noting that [random server load distribution can introduce chaos](https://drinkoblog.weebly.com).
   - Suggestions included using **grammar constraints** to ensure valid output and adjusting **temperature/top-p** settings to control stochasticity.
- **GPT-5's Coding Prowess Debated**: A user asked if **GPT-5** could code an app without payment, and was directed to [AI Studio](https://aistudio.google.com/) for free website building, offering **100 free 2.5 pro queries per day**.
   - Another member pointed out that Notion has built-in AI capabilities which are state of the art (SOTA) models, and can accomplish the goal gamification the user seeks.
- **Gemini 2.5 Pro and Claude Sonnet Battle for Storytelling**: Members recommended **Claude Sonnet** and **Gemini Pro** for AI-assisted story writing, with another suggesting trying them all to find the best fit, especially before it goes away.
   - It was noted that **Gemini 2.5 Pro** has a **1 million token context window**, making it suitable for remembering substantial details, and that AI Studio is free for a limited time.
- **AI vs Human: Detecting Generated Text Fingerprints**: A discussion arose about detecting AI-generated content, with one user explaining it's easy to do by comparing **n-grams** and **word distributions**, and suggesting that a fingerprint of all models is measureable using cosine similarity.
   - They stated that *EQBench* did exactly that: making AI trained on Gemini easily detectable based on quirks and habits, and therefore they trained DeepSeek on that approach.
- **Sora 2 Video Generation**: Users compared **Veo 3.1** and **Sora 2** for video generation, debating whether **Sora 2** is superior, especially with knowing prompts and following them well.
   - While some found both to be similar and still needing development, others argued **Veo 3.1's physics engine** and prompt understanding were inferior to **Sora 2's** early performance.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1428593951220174918)** (11 messages🔥): 

> `AI Voice Assistant, Sora Global VPN, Tech Discord Security` 


- **AI Voice Assistant Volunteer Search**: A member inquired if another had experience building an **AI voice assistant**, as they are a **PM** looking for a volunteer to dive into the **AI part**.
   - The PM asked if the other member wanted to join their team to *make **Sora** global with vpnyolw*.
- **Using AI for Reviewing and Note-Taking**: A member mentioned they are making use of **basic AI support** for reviewing work, note-taking, and scaffolding.
   - They also stated they are a **university student** working on a group project, recommending feeding school rules into prompts for guidance.
- **VPN Recommended for Tech Discords**: A member indicated they don't have a VPN, prompting another to recommend **onetar.os** for security.
   - Another member agreed, stating that *there are some weeeeiiiiirrrrrdddddd people in this server*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1428481918823039119)** (23 messages🔥): 

> `futuristic robot prompt, Sora 2 AI, viral video prompt, jujutsu kaisen vs goku prompt, Sora's image recognition` 


- **Prompt for robot in storm**: A user requested a prompt for *a futuristic robot in a storm banging at someone’s door and asking to be let in but then gets sucked into a tornado*.
- **How to Make Viral Video**: A user requested a prompt to *make viral video* and a member recommended to provide more details instead of a vague prompt.
- **Request prompt for Jujutsu Kaisen vs Goku video**: A member requested a prompt to *make video jujutsen vs goku*, but another user mentioned that *the image is copyrighted*.
   - Another member provided a detailed [prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1428838305281085490/pseudoCode_thought_experiment_for_ai_models.odt?ex=68f3f4de&is=68f2a35e&hm=33ee685e260fa6807db6b0140e367f49abdb019f116864ccf22b1707c9318ca3) for a 55s anime-cinematic trailer of an original Jujutsu-style sorcerer (blue/purple cursed energy) vs a Saiyan-like hero (gold aura).
- **Sora can recognize the difference between real person and fictional image**: A user asked if *Sora can recognize the difference between a real person image and a fictional character image* and a member replied, *Yes*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1428481918823039119)** (23 messages🔥): 

> `Text-to-image prompts, Copyrighted Image Generation, Sora AI capabilities, Extended fight scenes without word limit` 


- **Robot's Ironic Twist of Fate**: A user requested a text-to-image prompt: *a futuristic robot in a storm banging at someone’s door and asking to be let in but then gets sucked into a tornado*.
   - Another user suggested that they can create the prompt themselves, by describing the details they want in the new picture.
- **Copyright concerns with AI video generation**: A user inquired how to generate a video of *jujutsu kaisen vs goku*, but expressed concern about **copyright issues**.
   - Another user suggested starting the prompt with *anime* and adding more details, but the original poster was concerned that the image is copyrighted.
- **Jujutsu-Style Sorcerer clash trailer prompt**: A user provided a detailed prompt for creating a **55s anime-cinematic trailer** of an original Jujutsu-style sorcerer (blue/purple cursed energy) vs a Saiyan-like hero (gold aura).
   - The prompt includes specifications for escalation loops, verticality, color contrast, smear frames, shockwave timing, resolution (**1080p**), frame rate (**24fps**), and bass-heavy score syncing.
- **Sora's image recognition prowess questioned**: A user asked whether **Sora** can recognize the difference between a real person image and a fictional character image.
   - Another user simply responded in the affirmative.
- **AI thought experiment in Pseudocode released**: A user shared a [pseudoCode thought experiment](https://cdn.discordapp.com/attachments/1046317269069864970/1428838305281085490/pseudoCode_thought_experiment_for_ai_models.odt?ex=68f3f4de&is=68f2a35e&hm=33ee685e260fa6807db6b0140e367f49abdb019f116864ccf22b1707c9318ca3&) for AI models.
   - It was written in CREATIVE COMMONS, and the author invited users to dm them to add their handle for version 1.0 if they wish to use/modify/print it elsewhere, and chose **SGM** specifically because of its frequency across models as a token.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1428458169369235539)** (383 messages🔥🔥): 

> `Repo Mapping to Cursor Account, Perplexity Comet Invite & ChatGPT Promo, Games Inventory UI Overhaul, Cursor's Blip, Platform UI Changes` 


- **Request Feature to Map Repo to Cursor Account**: A user requested the ability to **map a repo to a specific Cursor account**, allowing for automatic switching between work and personal accounts based on the repository being used.
- **Teasing Games Inventory UI overhaul**: A user tried to **one-shot an overhaul** of his games inventory UI from a plan file, but failed due to `Tool read_file not found`.
- **Cursor undergoes UI changes**: Users noticed and discussed **UI changes** in Cursor, specifically the disappearance of icons from the sidebar on platform.openai.com.
- **Analyzing Cursor Usage with Token Watch**: A user shared a [Vercel app](https://token-watch.vercel.app/) to **monitor Cursor usage** and provided instructions on how to retrieve the necessary JSON data using `curl` or `Invoke-RestMethod`.
- **Edit File issues for multiple Users**: Several users reported issues with the **`read_file` tool**, with one user creating a [forum topic](https://forum.cursor.com/t/tool-read-file-not-found/137856) to discuss the problem, later discovering it was linked to **Custom Modes**.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1428819354782535801)** (124 messages🔥🔥): 

> `True Remembering AI, deterministic model agnostic Framework, objective metrics, nochain orchestrator` 


- ****True Remembering AI** debuts with bold claims**: A developer introduced a new AI system, claiming it's the *very first True Remembering, Evolving and Learning AI* that doesn't require manual RAG creation, frameworks, API costs, or curated chats, available at [dev.thelastrag.de](https://dev.thelastrag.de).
   - The AI is promoted as natively remembering and allowing users to define its role, such as an AI girlfriend or working partner, with one user humorously commenting with an image, *ahahah lol😄how the heckl*.
- ****Deterministic Framework** offers model agnostic benefits**: The developer claims their framework is fully deterministic and model agnostic, not needing function calling or standard frameworks like Langchain, and independently saves memories, curates chats, learns, evolves, and changes identity.
   - They claim it saves +90% tokens compared to regular Kontextwindow LLMs, but objective metrics for measuring subjective qualities remain a debate.
- **Critics Challenge AI Claims with **Objective Metrics****: Critics raised concerns about the lack of technical info, surface-level descriptions, and apples-to-oranges comparisons on the website, suggesting it might just be RAG with LLM-assisted memory storage/retrieval and calls for *objective metrics* to validate performance.
   - The developer responded that judging by the actual outcome is more important than marketing, prioritizing functionality and data safety over cosmetics, and offered free access to test the AI's capabilities.
- ****Nochain Orchestrator** replaces frameworks**: The developer argues their *nochain orchestrator* replaces traditional frameworks by being fully deterministic, model agnostic, and independent of external support, classes, or frameworks.
   - This approach aims to avoid *black box behavior* and dependencies on specific model capabilities, making orchestration predictable and debuggable, as detailed in [The Nochain Orchestrator](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a) blog post.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1428484072484769932)** (126 messages🔥🔥): 

> `Combining reasoning with web search, Audio processing models, Image inputs in Responses API, Cloud for ComfyUI, Security vulnerability` 


- ****Reasoning with Web Search**: A Flaky Endeavor**: A user sought advice on combining **reasoning** with **web search** and the **Responses API**, aiming for iterative reasoning and web searching, followed by tool calls and a closing text message, but reported flaky results with various models.
   - They found that **Gemini Flash** sometimes works with native or Brave search, **Grok 4 Fast** works with Brave or :online but lacks reasoning, **oss-120b** works intermittently, and **GPT-5 mini** consistently fails at tool calls.
- ****Whisper Alternatives** Sought for OpenRouter**: A user inquired about **audio processing models** on OpenRouter similar to Whisper, but was recommended [fal.ai](https://fal.ai) for multimedia models instead.
   - Others suggested [KittenTTS](https://github.com/qulingyuan/Kitten) for super tiny speech-to-text and the open source Sesame voice model, while one user shared a link to [Voxtral](https://apidog.com/blog/voxtral-open-source-whisper-alternative/), a **Mistral-based Whisper alternative**.
- ****Epic Tool Failures** Plague OpenRouter**: Multiple users reported issues with **tool calling failures** on OpenRouter, with one user stating that it's making OpenRouter unusable for them, despite it working fine when directly calling the providers.
   - One user joked that the **LLMs formed a syndicate** and are refusing to use tools without compensation.
- ****SDK Upgrade** Fixes Empty API Responses**: A user reported receiving **empty responses** from all models when using the Vercel AI SDK, despite successful processing indicated in the OpenRouter console.
   - Another user suggested upgrading the **AI SDK to the latest version**, which resolved the issue.
- ****GPT-5's Identity Crisis** on OpenRouter**: A user noticed inconsistencies in **GPT-5's identification**, with it sometimes claiming to be GPT-4, prompting concern.
   - Responses varied between the OpenRouter chat interface and OpenWebUI, with one user explaining that **models don't inherently know their identity**, and the interface simply reports what model is being used.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1428499307887202365)** (2 messages): 

> `` 


- **No New Model News**: No new models or significant discussions were present in the provided messages.
- **Channel Silence**: The 'new-models' channel appears to be inactive with no conversations to summarize.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1428590827122331711)** (28 messages🔥): 

> `OR stance on country requirements, GPT erotica, Dipsy V3.2, ChatGPT Active Users, Fake AI products/papers` 


- **Users Lamented **GPT Erotica** Quality Regression**: Users complained about the degradation in **GPT erotica** quality since the system fingerprint change on **November 11, 2023**, claiming `gpt-4-preview-1106` was the last good model for smut.
   - They added that no matter how fancy of a jailbreak is injected, it will have hesitation in its outputs after the "update".
- ****Dipsy V3.2** Praised for Completions**: One user is sticking to **Dipsy V3.2** for just about everything with completions, using custom formats to guide it rather than the stock user-assistant chat format.
   - Another user replied to this comment suggesting that this makes the user in the top **0.01%** of imaginary ranking ERPers.
- ****ChatGPT's** Gargantuan Impact on Normies**: A user noted **ChatGPT has 700 million+ active weekly users** stating that recent changes have a gargantuan blast radius that probably isn't fully understood yet.
   - They added that whatever OpenAI does, it probably won't impress many advanced users, but it will be fascinating to watch the normies react.
- **Fake AI products success rate**: One user wonders what's the rate of success of fake AI products/papers, noting that people seem to be pulling that a lot.
   - Another user jokingly says to *buy my course and i'll teach you*, but in seriousness, suggested that *if you get enough people to see it on Twitter, the success rate is 100%* linking [this Twitter post](https://x.com/vithursant19/status/1979176346329792738).
- **AI art considered unprofessional?**: A user expresses feeling that it is unprofessional for companies, even AI companies, to use AI art, suggesting that it feels wrong for that AI art to be their brand.
   - Another user perceived it as okay, perhaps because they already associated them with AI, but agreed that things like *hand-made corporate memphis / stock photos are more professional*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1428461292280348915)** (81 messages🔥🔥): 

> `Scammer Alert, Great Uncensored Finetuners, LM Studio and Javascript Animations, LM Studio MCP and OpenHands Integration, System Prompts Parsing` 


- **Discord Scammer Spam**: A member alerted the channel about a **scammer** spamming all channels to try to maximize their reach.
   - It was also pointed out that one user was hacked and was spreading the scam without their knowledge and that Discord needs better mechanisms to cull these scams.
- **Great Uncensored Finetuners List**: A member shared a list of **great uncensored finetuners** including *huihui-ai, TheDrummer, mlabonne, and Jinx*.
- **Javascript Animations in LM Studio: A No-Go**: A member inquired if the js code in LM Studio has the ability to display animations.
   - Another member clarified that it's a **JavaScript sandbox, not a built-in browser**, and they may be misunderstanding its capabilities.
- **LM Studio MCP and OpenHands integration frustration**: One member needed help with getting **Grok** set up with **OpenHands** via MCP.
   - They stated that the help pages on how to set up MCP are vague, incomprehensible, and that they literally have no idea what to do to make the computer do a useful thing, even AFTER reading both of the MCP help pages.
- **System Prompt Parsing Issues**: A user discovered that LM Studio applies parsing to system prompts, causing the AI and user to see different things.
   - They found that **brackets and other symbols are problematic** and this depends on model, chat template, and other factors.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1428471558413881384)** (167 messages🔥🔥): 

> `DDR5-8000 Speed, GPU airflow, Mixing 1060 with 3070, LLMs for Medical Use, GPU Hardware Modification` 


- **DDR5-8000 Provides Blazing Speed**: A member mentioned that if they had **DDR5-8000**, it would be *4 times faster*.
   - Another member shared what *peak airflow* looks like, along with a picture of their fan setup.
- **1060 Joins 3070 for a Task**: A member asked if an unused **1060 OC 6GB** could help their **3070 8GB** setup.
   - Another member replied *no*, but another member suggested it could be worth a try, and ensure the **3070** is at the top.
- **MedGemma LLM Surfaces for Healthcare**: A member asked about **LLMs trained on medical and care information** and another member suggested **Gemma**.
   - Specifically, they linked to [lmstudio-community/medgemma-27b-text-it-GGUF](https://huggingface.co/lmstudio-community/medgemma-27b-text-it) on Huggingface, and mentioned that they have no idea if it's US or UK medical info.
- **GPU Bending Leads to Driver Update Solution**: After installing a custom GPU spacer, the member found it was bending the card's PCB diagonally, and said *that it is not recommended by experts worldwide*.
   - After reverting the card, the issues were apparently resolved after an **NVIDIA driver update**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1428460147822428300)** (87 messages🔥🔥): 

> `Docker Image Update Frequency, Merging LoRA Adapters, SmolVLM2 Fine-tuning, Gemma 3-4B Loading Options, Kokoro TTS Finetune Notebook` 


- **Unsloth Docker Image: Bi-Weekly Bliss?**: The Unsloth team aims to update their Docker image at least **twice a week** ([Docker Hub link](https://hub.docker.com/r/unsloth/unsloth)).
   - Community members suggested a **bi-weekly stable release** alongside a nightly build.
- **Adapter Assembly Antics!**: Users discussed merging multiple LoRA adapters for inference by *'adding them up and decide by 2'*, effectively averaging their weights.
   - The impact on **VL model performance** and official support for this method remain unclear.
- **Vision-Language Voyages**: A user inquired about official examples for **fine-tuning with videos on SmolVLM2** or other vision-language models.
   - Currently, no such examples exist.
- **Gemini Gemma Loading Game**: Users inquired about the difference between loading **gemma-3-4b-it** with and without the **-unsloth-bnb-4bit** suffix.
   - The Unsloth team confirmed it's the *same model* and the library **auto-directs to the non-4bit version**.
- **TTS Teasers: Kokoro's Tune?**: A user asked about releasing a **finetune notebook for Kokoro TTS**.
   - The team responded that **Kokoro lacks finetuning code** and needs Transformer support, suggesting *Neutt-air* and *VibeVoice* as alternatives.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1428651736557420636)** (8 messages🔥): 

> `Freelancer introductions, LLM integration and blockchain, RAG pipelines` 


- **Freelancer's Intro Sparks Collaboration**: An experienced engineer specializing in **LLM integration**, **RAG**, and **blockchain** introduced themselves, leading to a potential collaboration with another member.
   - The engineer highlighted their expertise in workflow automation, AI detection, image and voice AI, and blockchain development.
- **Engineer Pioneers LLM and Automation Solutions**: A freelancer showcased their ability to deploy automated pipelines and task orchestration systems leveraging **Dspy**, **OpenAI APIs**, and **custom agents**.
   - They notably reduced response times by **60%** through a support automation system integrating **Slack**, **Notion**, and internal APIs to an LLM.
- **RAG Pipeline Deployment Deep Dive**: The engineer outlined the design and deployment of advanced **RAG pipelines**, integrating **vector databases**, **hybrid search**, and custom retrieval logic.
   - These pipelines are tailored to deliver context-aware responses in production environments, demonstrating practical application of sophisticated AI techniques.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1428482729804304475)** (50 messages🔥): 

> `Qwen 2 VL 2B, Apple FastVLM-1.5B, Liquid FM2 VL 450M, Gemma 3 12B Instruct VL, LFM2-VL models` 


- **Smaller Vision Models Face Challenges: Qwen VL Struggles**: Members discussed the challenges with smaller vision models, with one user noting that [**Qwen 2 VL 2B is garbage and can't see a thing**](https://github.com/QwenLM/Qwen2).
   - The user mentioned their intention to try **Apple’s FastVLM-1.5B**, praising its base model and vision capabilities, while another user suggested trying the new **4B VL** model.
- **Liquid and Gemma VL Models Draw Praise!**: A user found **Liquid FM2 VL 450M** to be the smallest useful VL model, while another recommended **Gemma 3 12B Instruct VL** for general tasks.
   - It was noted that **Gemma 3** and **LLaMA 3.2** often fail after SFT, with **LFM2-VL-1.6B** being a more reliable option.
- **DGX Spark Questioned for Cost-Effectiveness**: A user inquired about the value of using **Unsloth with DGX Spark** versus **RTX 3090/4090** setups.
   - Analysis revealed that **4x3090s** are significantly more efficient (**4.24x**) than **Spark** for **GPT 120B** prefill, costing **$2.19** compared to **Spark's $9.29** for a **100,000,000 token** workload.
- **Tesla V100 vs A40 for RAG System: An Inquiry**: A user with **Tesla V100-SXM2-32GB x8** was seeking advice on whether to switch to an **A40** for a **RAG system** intended for simultaneous queries by up to five users.
   - One member stated this decision *depends on designer and business requirements. If it’s just a hobby thing then pick whatever you are most comfortable.*"
- **Qwen 2.5 VL Struggles: A Bug Hunt?**: A user reported issues with **Qwen 2.5 VL** failing to understand images, providing a [GitHub link](https://github.com/Emericen/tiny-qwen) to their code.
   - The user noted that the code works with HF.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1428642865759785012)** (54 messages🔥): 

> `GGUF model file naming conventions, Unsloth Dynamic Quantization, PIL import error, vLLM integration issues, Qwen2.5 7B OOM issues` 


- ****GGUF Filename Meanings Finally Found!****: A user inquired about the meaning of filenames for GGUF model files, such as `unsloth/Apertus-8B-Instruct-2509-GGUF`, and a member shared a [helpful Gist link](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9) with the naming conventions.
   - It was further noted that **Unsloth quantizations** usually perform much better due to ongoing bug fixes and the implementation of **Unsloth Dynamic Quantization** [docs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs).
- ****PIL Problems Prompting Pillow Purge!****: A user reported a `cannot import _Ink from PIL` error when running a Colab notebook.
   - Another user suggested trying `pip uninstall Pillow` followed by `pip install Pillow`, which resolved the immediate error but led to a new shape-related issue during `trainer.train()`.
- ****vLLM Ventures Yielding Varied Woes!****: A user encountered issues when trying to integrate vLLM and suggested starting with a known working notebook and modifying one thing at a time.
   - The user then reported that the [Advanced Llama 3.2 3B GRPO LoRA notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb) also failed with the `_Ink` issue.
- ****Qwen2.5 Quagmire: Questioning KV Cache!****: A user ran into **OOM** issues while fine-tuning **Qwen2.5 7B** with 80 GB VRAM, even with a small context length, and was advised to use **fast inference**.
   - It was suggested that the VRAM was likely being consumed by the **KV Cache**, and reducing the batch size could alleviate the issue.
- ****FailOnRecompileLimitHit Frustrations****: A user encountered a `FailOnRecompileLimitHit` error while trying the **GPT OSS 20B** unsloth reinforcement fine-tuning notebook on an H100 80G instance, potentially due to a Colab update.
   - It was suggested to adjust a setting indicated in the full error message or try sorting the dataset by size to mitigate the issue.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1428557873465524336)** (3 messages): 

> `Legal move attempts, Move hallucination` 


- **Multi-Turn Legal Move Attempts Proposed**: A member suggested that instead of random legal moves on failure, the bot could attempt **multi-turn legal moves**.
   - They expressed interest in seeing whether that would improve performance or not, acknowledging that arguments could be made either way.
- **Hallucinated Moves Expected**: A member commented from personal experience that the bot will keep **hallucinating moves**.
   - No further details or links were provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1428772806220582944)** (13 messages🔥): 

> `BitNet performance, Microsoft BitNet GitHub, 1.58bit equivalence` 


- **BitNet Claims 1.58bit Performance Equivalence**: Users discussed the possibility of achieving one-to-one performance with **1.58bit** precision using [Microsoft's BitNet research](https://github.com/microsoft/BitNet).
- **Confusion over BitNet Paper Update Status**: A user expressed confusion over the [BitNet paper's](https://huggingface.co/papers/2510.13998) last updated date, suspecting it might be incorrect.
   - Another user confirmed the paper's link to the **Microsoft BitNet GitHub repository**, suggesting the information might not be up to date.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1428457957460283612)** (172 messages🔥🔥): 

> `Access Token Permissions, HuggingChat Limits, Model Context Length, Prompt Injection Mitigation, AI Infrastructure` 


- ****Access Token** Permission Peculiarities**: A member reported that they could create an **access token** with 'no results found' as a permission, but clicking on it says *"role is required"*, as shown in the attached [screenshot](https://cdn.discordapp.com/attachments/879548962464493622/1428457957179523082/image.png?ex=68f3e424&is=68f292a4&hm=700c457eb3d56c1c7fe8d1d4318b3d2dcbc5a8ca1579390360a06ea343634e30&).
- ****HuggingChat** is Back, UI Feelings Mixed**: **HuggingChat** is back with a new UI, as noted by one member who called it cool.
   - Others found the UI clunky and slow, with one describing it as having *"opposite rizzmatic"*, to which another responded *"nobody says that bro"*.
- **Context Length Capacity Crunch**: One user inquired about scaling model context to handle **400 images**, specifically how to manage the context to ensure the model processes all the information effectively.
   - It was mentioned that [Quantization](https://pytorch.org/docs/stable/quantization.html) is something to try.
- **Prompt Injection Prevention**: Members discussed mitigating potential **hacking** via **prompt injection** in agentic workflows with email or personal accounts.
   - Suggestions included aggressive **sandboxing**, **context isolation**, and the principle of least privilege, as explained in a [security class](https://en.wikipedia.org/wiki/Principle_of_least_privilege).
- **AI Infrastructure Insights Shared**: During a conversation a member stated that base corporate **AI infrastructure** most likely uses **Megatron** (NVIDIA tech) and **TPUs** (Google tech).
   - Another mentioned that the training data is sometimes scraped, referencing a **1.5 billion lawsuit** against Anthropic for these practices.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1428550796680892538)** (2 messages): 

> `Influence Functions, Research Collaboration` 


- **Interest sparks in Influence Functions**: A member expressed interest in **influence functions** and is seeking to discuss their use for a new research question.
   - They also seek a collaboration opportunity with others interested in this area.
- **Papers on Influence Functions provided**: Two papers were shared: one explaining [influence functions](https://arxiv.org/abs/2308.03296) and another demonstrating their use for interesting research.
   - The second paper provided was linked as [https://arxiv.org/abs/2411.12580v1](https://arxiv.org/abs/2411.12580v1).


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1428847282480222339)** (2 messages): 

> `Qwen3 Vision model, NexaAI, GGUF` 


- **Qwen3 Vision Model Arrives!**: The new **Qwen3 Vision model** is available on HuggingFace via [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
   - It's also available in **GGUF** format via [NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF).
- **Qwen3's Vision Powers**: The model is designed for **vision-language tasks**, enabling it to process and understand visual inputs alongside textual information.
   - It supports various applications, including image captioning, visual question answering, and multi-modal content generation.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1428458802143035392)** (5 messages): 

> `FRAI, Responsible AI, YouTube Content, Agent Tutorial` 


- **FRAI CLI Framework Debuts**: A member shared a CLI version of **FRAI**, a *developer-first framework* for Responsible AI, and provided a link to the [GitHub repository](https://github.com/sebuzdugan/frai).
   - They requested feedback and a star if others find it interesting or helpful.
- **YouTube Content Creation Begins**: A member recently started creating content on **YouTube** and is trying to improve from video to video and is looking for feedback on their [YouTube channel](https://m.youtube.com/@sebuzdugan).
   - They requested feedback to help him improve from video to video.
- **Agent Tutorial Posted**: A member wrote a new tutorial and asked if it counts as "making something" and provided a link to the [tutorial](https://samdobson.uk/posts/how-to-build-an-agent/).
   - Another member responded saying that it *sure does count*!


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1428591577688707123)** (1 messages): 

> `Custom Blocks in Diffusers, Modular Diffusers, Pipeline Blocks` 


- **Roll your own Blocks**: Custom blocks are presented as a good way to implement functionality that is currently not present in the library but fits seamlessly within it.
   - You can check out some custom blocks [here](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401) and [here](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block) are the docs.
- **Blocks Blocks Blocks**: Custom blocks are useful for expanding current functionality.
   - It is possible to use custom blocks to add new functionality or modify existing functionality.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1428802973743714545)** (1 messages): 

> `text conditioned image generation, dynamic action shots, pixelated art style images, serene atmospheres in images` 


- **Text-Conditioned Image Generation Yields Good Results**: A member reported achieving [good results with text conditioned image generation](https://discord.com/channels/922424143113232401/922424143570311226/1428802974393593926), thanking another member for the help.
   - Example prompts included *"Small orange lizard-like creature with flames on its tail..."*, *"Red-haired character walking through dense forest..."*, and *"A red-roofed healing center in a vibrant green field..."*.
- **Vibrant and Dynamic Image Generation**: The image prompts emphasized **dynamic action shots** and **energy-filled atmospheres**, such as a *"Small orange lizard-like creature with flames on its tail, battling against a human trainer in a grassy field"*.
   - Other prompts focused on creating **serene atmospheres** with lush greenery and gentle sunlight.
- **Pixelated Art Style Imagery**: One prompt requested a **pixelated art style**, showcasing the ability to generate images in different artistic styles.
   - The prompt was *"Red-haired character walking through dense forest, overcast day, pixelated art style, serene atmosphere, lush greenery surrounding the path"*.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1428891058409963623)** (1 messages): 

> `Chat Template Conversion, Tokenizer Usage, Fine-Tuning Script Execution` 


- **Chat Template Conversion Commences**: Converting a dataset into a model's specific **chat template** is the first step for effective fine-tuning.
   - This ensures compatibility and optimizes the model's understanding of conversational structures, but requires [meticulous attention to format](link.to.format).
- **Tokenizer Tokenizes Text**: Employing the model's **tokenizer** is crucial to prepare the text data for the fine-tuning process.
   - Tokenization breaks down text into numerical representations that the model can process efficiently, *ensuring alignment between data and model vocabulary*.
- **Fine-Tuning Script Sets Sail**: Executing the **fine-tuning script** on the converted and tokenized dataset trains the model on the new data.
   - This step adapts the model's parameters to better suit the target task, leveraging techniques like **transfer learning** for optimal results without needing to rebuild the entire model.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1428616109367754834)** (5 messages): 

> `LoRA/PEFT training with HF jobs, Hyperparameter Optimization, Lighteval's compatibility with LoRA adapters, Pushing models to Hugging Face Hub without HF Jobs` 


- **LoRA/PEFT training done via HF Jobs**: A member noted that while [LoRA training with HF Jobs](https://huggingface.co/learn/smol-course/unit1/5#lorapeft-on-jobs-optional) is explained, [lighteval](https://github.com/huggingface/lighteval) doesn't support evaluating models with LoRA adapters yet, pointing to [PR #611](https://github.com/huggingface/lighteval/pull/611).
   - Another member suggested merging the model locally or cleverly in a `hf job` before evaluation.
- **Hyperparameter Optimization Intro**: A member shared a basic approach to hyperparameter optimization, implemented in [this gist](https://gist.github.com/robbiemu/e8c62ad92c0743c7214c8de40f3a5d1b).
- **TrackIO Graphs need Logging Steps**: One member recommends setting `logging_steps=30` to train one full epoch and get the trackio graphs when training (with a batch size of 4).
- **Pushing models to Hub**: One member asked about pushing models to the Hugging Face Hub without using `hf jobs`, seeking alternatives to avoid associated costs.
   - They mentioned having existing published models and inquired about the requirements for class credit.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1428606928711454820)** (5 messages): 

> `agents-course intro, New students joining` 


- **agents-course welcomes new students**: Multiple new students announced they are starting the agents-course today.
   - The new students are excited to begin the course.
- **Course Starts, Excitement Begins!**: Enthusiastic individuals are kicking off the agents-course today, eager to dive into the material.
   - The chat reflects the shared anticipation as multiple participants declare their start date.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1428469251907387433)** (92 messages🔥🔥): 

> `Cognition SWE-grep, MobileLLM-Pro, Anthropic/Google TPU Partnership, HeyGen ARR, OpenAI Physics Initiative` 


- **Cognition's SWE-grep Speeds Up Agentic File Search**: Cognition launched **SWE-grep** and **SWE-grep-mini**, RL-trained models that retrieve context for coding agents at **2,800 TPS**, about **20x faster** than existing solutions, rolling out a **Fast Context sub-agent** to Windsurf users, as detailed in [their blog post](https://xcancel.com/cognition/status/1978867021669413252).
   - A community member speculated SWE-grep is a modified **QwQ model** running on **Cerebras**, and someone has seemingly created something similar, [ceregrep-client](https://github.com/Swarm-Code/ceregrep-client), while another claimed it is a RLFT'ed OSS model.
- **Broadcom-T5 Customer: Anthropic's TPU Play?**: There is speculation that **Broadcom's** fifth **$10B** customer is **Anthropic**, who will buy **TPUs** via Broadcom rather than Nvidia, possibly signaling a new Google-led funding round, according to [this tweet](https://xcancel.com/zephyr_z9/status/1978834774786445562?s=46).
- **HeyGen Hustles to $100M ARR**: **HeyGen** rocketed from **$1M ARR to $100M** in just **29 months**, and the team announced they will release a manifesto called *“The HeyGen Way”* detailing their internal playbook, according to [this tweet](https://xcancel.com/joshua_xu_/status/1978837985039888388?s=46).
- **Anthropic's M365 Integration: Claude Gets to Work**: **Claude** now integrates with **Microsoft 365** (SharePoint, OneDrive, Outlook, Teams) and includes a new enterprise-search project, available today for Team & Enterprise customers, according to [this tweet](https://xcancel.com/anthropicai/status/1978864351076315203?s=46).
- **Meta Rolls Out MobileLLM-Pro**: Meta released **MobileLLM-Pro**, a **1B-parameter** model optimized for on-device inference, which beats **Gemma 3 1B** and **Llama 3.2 1B** on reasoning & QA while trained on fewer than **2T** open-source tokens, according to [this tweet](https://xcancel.com/_akhaliq/status/1978916251456925757).
   - Community reactions, however, suggest it is trash and not even 1 iq.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1428544072125251635)** (5 messages): 

> `M4 Max, Ollama, LM Studio, Local LLM Performance, Qwen Next 80B` 


- **New M4 Max sparks local LLM setup talks**: A member got a new **M4 Max** with **128gb** and asked for local workflows or setups.
   - Another member was curious how different models run locally in **ollama** and where the sweet spot of complexity and speed are on that hardware.
- **LM Studio preferred for M4**: A member suggested **LM Studio** instead of **Ollama** because Ollama doesn't support **mlx**.
   - Another member confirmed that they are using **LM Studio** and basic chat has been pretty snappy with **Qwen Next 80b**.
- **OpenAI 120B fits at 4-bit quant**: A member shared that **OpenAI 120B** fits at **4-bit quant** and seems to be the max size on their new machine.
   - They are interested in **evals** to help them figure out what the M4 Max is capable of.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504)** (9 messages🔥): 

> `AI Granny, OpenAI Sora MLK Likeness` 


- **AI Granny Gold Digger Inflames Instagram**: A fully **AI-generated influencer** named *grannyspills*, depicting a blunt, gold-digging grandmother who serves toxic dating advice, launched in July and is about to surpass **2 million Instagram followers** as reported on [X](https://xcancel.com/venturetwins/status/1978852719335985309).
   - Posts highlight rapid growth, high engagement, and debate over whether audiences care it’s fake with some users praising the satirical character, others worrying about AI’s impact on culture.
- **OpenAI Blocks MLK Likeness in Sora**: Following complaints about disrespectful AI-generated video clips of **Dr. Martin Luther King Jr.**, OpenAI has paused any Sora outputs depicting King while it adds new guardrails, as reported on [X](https://xcancel.com/OpenAINewsroom/status/1979005850166648933).
   - Most users criticize the move as a slippery-slope concession that privatizes public figures and could invite endless takedown demands, especially as one member claimed to have *"seen him cut a promo in a WWE ring today"*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1428459839067258900)** (16 messages🔥): 

> `Maxwell Disassembler & Jetson Nano, Hopper GPUs for AI/Quantum, US GPU Restrictions & China, GPU Mode Distributed GPU Talks` 


- **Maxwell's Disassembler Powers Jetson Nano**: A member highlighted the usefulness of the **Maxwell disassembler** and noted that it powers the first-generation **Jetson Nano**, suggesting it's a good option for those working with constraints, linking to a [tweet with an image](https://x.com/tenderizzation/status/1978871856922087576).
- **Hopper Shines for AI and Quantum**: A member chose **Hopper** GPUs due to their **CUDA-Q support** and suitability for **AI** and **quantum** applications, despite **Blackwell's** unavailability.
- **US GPU Nerfing Spurs Chinese Ingenuity**: A member described how US restrictions on **H100s** led to **DeepSeek** using **PTX/SASS** instructions to overcome memory bandwidth issues, creating a powerful model with fewer resources; further restrictions mean China can legally only acquire **H20** GPUs, which they are still using effectively.
- **GPU Mode Talks Available on YouTube**: A member asked about the availability of distributed GPU talks from **GPU Mode**, and another member provided a link to the [GPU Mode YouTube channel](https://www.youtube.com/@GPUMODE/videos).


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1428474388226773203)** (2 messages): 

> `Distributed Triton, Non-ML Kernels with Triton DSL` 


- **Distributed Triton Tools still in Development**: Members are actively looking for state-of-the-art **distributed Triton** programming tools, but they are still in early development.
   - While awaiting stable releases, users explore various approaches like **Torch Distributed** and manual data parallelism for distributed training.
- **Triton DSL expands beyond ML Kernels**: Users are investigating writing non-ML kernels, such as **stencils**, using the **Triton DSL**.
   - The DSL's flexibility allows for expressing a wide range of parallel computations beyond traditional machine learning workloads, opening doors for **scientific computing** and custom algorithms.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1428465263463891015)** (10 messages🔥): 

> `TMA Multicast Bandwidth, cuTensor L2 Promotion, cp.reduce.async.bulk Memory Ordering, Thread Block vs CTA, Perl modules for CUBIN files patching` 


- ****TMA Multicast** Bandwidth Boost?**: A member inquired whether **TMA multicast** bandwidth scales with **CTAs** or improves cache hits by loading equal parts into different blocks.
   - Another member clarified that **TMA multicast** accesses **L2** once, limited by broadcast bandwidth; e.g., **H100** can achieve ~**80B/cycle/SM** with **TMA multicast**, exceeding the average **L2** read bandwidth of ~**38B/cycle/SM**.
- **Memory Ordering Semantics Clarified**: A member asked if the `cp.reduce.async.bulk` reduction operation's `.relaxed.gpu` memory ordering ensures safe calls on the same memory region across different blocks.
   - It was not clarified if it's safe to call it on the same memory region across different blocks.
- **Patching CUBIN Files with Perl**: A member shared a link to [Perl XS modules](https://redplait.blogspot.com/2025/10/perl-modules-for-cubins-patching.html) for patching **CUBIN** files.
   - This could allow customization and modification of compiled CUDA code.
- ****CTA** == **Thread Block****: A member asked if there is a difference between a **thread block** and **CTA**.
   - Another member clarified that there's no difference, that **CTA** = **cooperative thread array**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1428487088323428474)** (7 messages): 

> `PyTorch Free-Threading, Accessing Backward Functions, GELU Backward API` 


- **PyTorch Goes Threading-Free**: A member shared a [blogpost on multi-threaded parallel inference on PyTorch models](https://trent.me/articles/pytorch-and-python-free-threading/).
   - The post details the new threading strategies that will *unlock* new parallelism paradigms in **PyTorch**.
- **Backward Functions Beckon**: A member inquired about accessing backward functions without using autograd, aiming to use autograd's kernels in a custom backward for a fused kernel.
   - Suggestions included using `torch.func.grad` or `torch.autograd.grad`, with a request for the specific op to provide tailored guidance on registering backward kernels.
- **GELU's Forward Facade**: A member mentioned that **GELU** only exposes a forward API, implying challenges in directly accessing its backward functionality.
   - This limitation could impact the implementation of custom backward functions requiring **GELU's** gradient computation.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1428841133651394681)** (1 messages): 

> `SF Startup, GPU performance, PyTorch, CUDA kernels, Pac Heights` 


- **Herdora Startup in SF Seeks Engineers**: A seed-stage startup in SF, [Herdora](https://jobs.ashbyhq.com/herdora) backed by **YC**, **Jeff Dean**, **Woj Zaremba**, and head of kernels at **Together.ai**, is hiring engineers proficient in **PyTorch** and **CUDA kernels** to enhance **GPU performance**.
   - The team is based in **Pac Heights** where they live and work together, offering full-time positions and winter/spring/summer internships with a compensation package of **$170-200k** + **2-4% equity**.
- **Pac Heights Team Offers Engineer Roles**: Herdora, based in **Pac Heights**, is actively recruiting engineers passionate about optimizing **GPU performance** through **PyTorch** and **CUDA kernels** programming.
   - Interested candidates can apply via the provided [link](https://jobs.ashbyhq.com/herdora) or reach out directly for inquiries, with competitive compensation ranging from **$170-200k** and **2-4% equity**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

zlu86: You should be good to go, it's general enough
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1428739127603494933)** (2 messages): 

> `SGLang, vLLM, torchao, Quantization` 


- **SGLang laggs vLLM quant features**: While **SGLang** offers some limited support for **torchao** quantization models, it isn't as up-to-date as **vLLM**.
   - **vLLM** integration supports any type of quantization config, but **SGLang** only supports int4wo, int8dq, int8wo right now.
- **SGLang leans into online quantization**: Only [online quant](https://docs.sglang.ai/advanced_features/quantization.html#online-quantization) is supported so far by **SGLang**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1428903762042294342)** (1 messages): 

> `geohot, Image Analysis` 


- **Geohot Appears!**: A user shared an image featuring **Geohot** in a meme-like context.
   - The image, named `565508333_1719200958774569_3857903007160114304_n.png`, was posted without additional commentary, available [here](https://cdn.discordapp.com/attachments/1215328286503075953/1428903761828647104/565508333_1719200958774569_3857903007160114304_n.png?ex=68f431d4&is=68f2e054&hm=164ef58ff006255ff0f3b7cf6b78ee7c91129b8aaad2208430c1ec9fd90b1407&).
- **Visual Data Dump!**: An image attachment with a long filename was shared: `565508333_1719200958774569_3857903007160114304_n.png`.
   - It can be accessed directly via this [CDN link](https://cdn.discordapp.com/attachments/1215328286503075953/1428903761828647104/565508333_1719200958774569_3857903007160114304_n.png?ex=68f431d4&is=68f2e054&hm=164ef58ff006255ff0f3b7cf6b78ee7c91129b8aaad2208430c1ec9fd90b1407&).


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

arseniivanov: I am at Lund University, but the HPC scene is kind of non-existent here tbh :/
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1428664913080881292)** (4 messages): 

> `Iris multi-GPU programming framework, Gluon backend, NVIDIA backend, Scale-out and RDMA support, Metal backend` 


- **Iris Grows Open-Source GPU Support**: The **AMD RAD team** released new features in [Iris](https://github.com/ROCm/iris), their open-source **multi-GPU programming framework** built in **Triton + Python** for transparent performance and optimized multi-GPU execution.
- **AMD Builds Lower-Level Gluon Backend**: Iris introduced an **experimental Gluon backend** for writing kernels closer to the metal with full control over layouts, memory, and data movement; see the [Gluon Docs](https://rocm.github.io/iris/reference/gluon/overview.html).
- **Iris Adds NVIDIA Backend**: Iris now has an **NVIDIA backend** for testing and writing examples anywhere, though it's optimized for **AMD GPUs**; note that **scale-out and RDMA support** is coming soon, enabling seamless distributed execution across multiple nodes.
- **Metal Backend**: A user inquired about a **Metal backend** to utilize devices like an **iPad** connected to a **Mac**.
   - Another user responded that **Triton** would need to function on **Mac** first, noting that **CPU** development is underway but details are unclear, and requested a code example for cross-machine memory accesses.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1428494468704768120)** (7 messages): 

> `H100 attention kernels, ThunderKittens ROCm release, Fixing broken kernels, warp operations` 


- **H100 Attention Kernels Broken**: A user inquired about the current status of **H100 attention kernels**, and another user responded that they are aware of the issue and plan to fix them, but are currently busy.
   - The user also offered to DM their personal **H100 attention forward implementation** that works, but lacks backward implementation.
- **ThunderKittens ROCm Release Incoming**: A user announced that Simran is working with **AMD** on the new **ThunderKittens for ROCm**, indicating that a release should be expected soon.
- **Community Offers Help Fixing Broken Kernels**: Multiple users offered assistance in fixing the broken kernels, citing their experience and availability to help with updating the kernels.
   - One user suggested updating the relevant changes from the latest update, such as the new namespace prefix rules, to facilitate their assistance.
- **H100 Kernel Compilation Workaround**: A user shared a workaround to get the **H100 kernel** to compile, though it crashed on run, using the last 2 commits from [this GitHub repo](https://github.com/aehmttw/ThunderKittens/commits/main/).
   - The main changes involved adding `warp::` in front of many operations, fixing casting, and temporarily removing causal attention.
- **New Warp Operation Rules in ThunderKittens**: A member clarified that every operation now clearly defines who executes it with namespace prefixes such as `warp::` or `warpgroup::`, which determine collective launch behavior.
   - They pointed out that errors often arose because previous versions of **TK** implicitly meant either run by an entire warp or a single thread, depending on the operation and that now the user must ensure that `tma::load_async` or any semaphore operation is run by a single thread (otherwise, it's run 32 times).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1428691712993136740)** (12 messages🔥): 

> `VectorAdd Leaderboard Updates, B200 Performance, L4 Performance, A100 Performance, H100 Performance` 


- **VectorAdd_v2 Leaderboard Heats Up**: Multiple submissions were made to the `vectoradd_v2` leaderboard, showcasing performance across different hardware configurations like **B200**, **L4**, **A100**, and **H100**.
   - The submissions include timings for first, second, third and fourth/fifth places, as well as successful runs, indicating active competition and optimization efforts.
- **B200 Vector Addition Speed Race**: One member achieved **first place** on **B200** with a time of **236 µs**, and another member also secured **second place** with **237 µs**.
   - Other successful runs and third place submissions hovered around **238-247 µs**, indicating tight competition in vector addition performance on the B200.
- **L4 Claims First and Second Place**: The leaderboard saw a member claim **first place** on **L4** with a time of **6.80 ms**.
   - Another member followed closely, securing **second place** at **6.81 ms** with other successful runs around **6.92-6.93ms**.
- **A100 Showdown**: Several submissions targeted the **A100**, with runs achieving **third place** at **956 µs**, **fourth place** at **1017 µs** and **fifth place** at **1014 µs**.
   - There were also additional successful runs reported at **956 µs** and **1014 µs**, showing some variance in performance.
- **H100 Dominance Displayed**: The **H100** saw one member achieve **first place** with a time of **525 µs** and **526 µs**, while another secured **second place** at **539 µs** and one secured **third place** at **528 µs**.
   - These results indicate optimized vector addition performance on the H100, with very competitive timings among the top submissions.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1428772347212468377)** (7 messages): 

> `Sphinx Docs, Factorio Learning Environment` 


- **Factorio Learning Environment Releases Initial Sphinx Docs**: A member created initial [Sphinx documentation](https://github.com/JackHopkins/factorio-learning-environment/pull/346) using Cursor for the **Factorio Learning Environment** project, noting that it needs further refinement.
   - They provided the command `cd factorio-learning-environment/docs/sphinx && python -m sphinx -b html source build/html` for building the documentation.
- **Building Sphinx Docs Made Easy**: To build the sphinx docs, use the following command: `cd factorio-learning-environment/docs/sphinx && python -m sphinx -b html source build/html`.
   - The user noted it was generated using **Cursor**.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1428620276354125825)** (2 messages): 

> `Discord user anuragj0803, Discord user meem, Amazing event, Dev day` 


- **Users Seek Contact with anuragj0803 and meem**: A user on Discord is seeking contact with **anuragj0803** and **meem** and has requested they DM them if they see the message with an attached image.
   - The image contains a message stating, *"Thanks for organizing such amazing event. Look forward to seeing you guys at dev day."*
- **Acknowledgement of Event Organization**: The attached image thanks the individuals (presumably anuragj0803 and meem) for organizing an *'amazing event.'*
   - The sender also expresses anticipation for seeing them at *'dev day.'*


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1428753646023348245)** (2 messages): 

> `PTX Documentation, CUDA Threads as SIMD Lanes, CuTe Layout Plotting` 


- **Threads mimic SIMD Lanes, says Expert**: An expert suggested thinking of *32 "threads" in CUDA* as a fancy term for *32 "lanes" in traditional SIMD CPUs*, where many operations would potentially cross the lanes.
   - This was suggested for those struggling to understand that the boundary between threads is not preserved to enable data reuse.
- **CuTe Layout plots, Great Suggestion**: A member suggested reading **PTX docs** and plotting the layouts with **CuTe** to better understand how a Tensor Core collects input from many threads and registers, and scatters the output into many threads and registers.
   - Another member thanked the expert and said they'd take a closer look at the **PTX documentation** as well.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1428478106943094996)** (5 messages): 

> `tinygrad compiler design, picograd architecture, SITP goals, Karpathy's influence on tinygrad, Eureka Starfleet academy` 


- **Reasons to Ditch the Compiler?**: There are several reasons to avoid using a compiler such as unacceptable **JIT overhead**, deviations in numerics, guaranteed op fusion, bleeding-edge hardware, lack of hardware autotuning, or algorithmic rewrites.
   - A member is building **picograd** to take tinygrad's designs for the tensor language and device runtime, which he uses to explore these concerns.
- **Tinygrad Enters Eager Mode?**: The same member is exploring adding eager semantics to tinygrad using **C++ std::execution policies**, enabling readers to implement kernels with Triton, Gluon, and Python-HIP.
   - The goal is to target the **thunderkittens abstraction level** to make it pedagogically easier to learn from tinygrad's 20kloc codebase which is inspired by Halide and TVM.
- **SITP Joins Starfleet Academy?**: The goal of **SITP** and **picograd** is to become the second course on Karpathy's "Starfleet Academy" after llm101, focusing on ramping up knowledge and creativity in course building, with inspiration from [past educational resources](https://github.com/j4orz/notstd) and [YouTube tutorials](https://www.youtube.com/playlist?list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd).
   - The plan includes submitting a tutorial for **MLSYS 2026** focusing on compilation, beyond the basic parts 1 and 2.
- **Tinygrad Documentation Reboot?**: Karpathy influenced one of George Hotz's recent streams by pointing out that many of his tinygrad streams and documentation made no sense, watch the discussion [here](https://www.youtube.com/watch?v=QUry9dHC-bk).
   - This creates a gap for **SITP** and **picograd** to bridge micrograd to tinygrad.
- **Creative Co-Director Wanted!**: A member is seeking a **creative co-director** to help translate Torch eager mode, tinygrad, TVM, and Halide into a codebase and course.
   - They must deeply understand the semantics of math, not just the syntax.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1428688638967021671)** (2 messages): 

> `BitNet distillation, RL` 


- **BitNet Distillation Findings**: The paper on **BitNet distillation** ([BitNet distillation paper](https://arxiv.org/abs/2510.13998)) presents very good results.
   - One member expressed reservation about using it as a **loss function**, citing potential awkwardness in applications like **RL**.
- **BitNet Distillation Concerns in RL Applications**: A user expressed concerns with **BitNet distillation** being used as a loss function.
   - They noted it could be awkward for applications such as **Reinforcement Learning (RL)**.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1428644490826088470)** (2 messages): 

> `Kernel Optimization, Distributed Frameworks, Consumer Devices, Distributed Inference, Distributed Training` 


- **Research Head Seeks Hackathon Team**: The Head of Research at **EXO Labs** is seeking a team and project for the hackathon, with expertise in building distributed inference and training frameworks on consumer devices.
   - The member is particularly interested in **kernel optimization** or **distributed systems**.
- **EXO Labs Head Builds Distributed Inference Frameworks**: The Research Head at **EXO Labs** has experience building distributed inference & training frameworks on consumer devices, and pointed to [their work](https://x.com/MattBeton/status/1958946396062851484) for further reading.
   - They also joked about having *tripped the power in the Apple Cupertino office* by pushing their Macs too hard during development.


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1428860225477283912)** (2 messages): 

> `Fault Tolerant Llama Training, Node Failure Prediction` 


- **Crusoe Tackles Fault Tolerance with Synthetic Failures**: A new [PyTorch blog post](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/) details a **fault-tolerant** approach to **LLaMA** training using **Crusoe L40S** GPUs, highlighting resilience against **2000 synthetic failures every 15 seconds** without relying on traditional checkpoints.
   - The author questioned the need to invest in more automatic processes given existing checkpointing solutions using bash scripts, wondering about its advantages over existing solutions.
- **Agentic Systems Predict and Minimize Downtime**: A member mentioned the potential for predicting high rates of node failures using **agentic systems** or **ML techniques**.
   - The high prediction accuracy could lead to easier node replacement and minimized downtime.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/)** (1 messages): 

jongsokchoi: GPU mode talk starting now!
https://www.youtube.com/watch?v=1zKvCLuvUYc
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1428461224164593805)** (32 messages🔥): 

> `Anthropic agentic search, Langgraph's verbose boilerplate, Agentic Search vs Semantic Search, Groq not working in OpenRouter` 


- **Claude Code's Agentic Search Deconstructed**: A member implemented agentic search with DSPy, similar to **Claude Code**, after discovering that **Anthropic** hasn't open-sourced its code or revealed its implementation details.
   - The member found **Claude Code's** system prompt for read and search tools and used it to implement agentic search, emphasizing the importance of LLMs deciding what context to use through **ripgrepping** rather than relying solely on semantic search.
- **Langgraph feels low level**: Members discussed that **Langgraph** *feels low level* because it requires defining everything as a workflow graph with verbose boilerplate, forcing a graph-based mindset even when simpler control flow might suffice.
   - Another member agreed, noting that it's not a bad abstraction but has *a number of foot guns that are easy to set off*.
- **Semantic Search Faces Agentic Search**: Members argue that **agentic search** outperforms semantic search because it allows the LLM to decide what information to include in its context, referencing [this blog post](https://benanderson.work/blog/agentic-search-for-dummies/).
   - The method involves ripgrepping for terms, shortlisting documents, and then reading those documents, contrasting with semantic search's predefined retrieval and re-ranking processes.
- **Groq Goes Rogue on OpenRouter**: A user reported that **Groq** isn't working in OpenRouter, even when set as the only provider, providing configuration details.
   - Although the issue was presented with screenshots, there were no solutions available at the time of summarization.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1428512135507345469)** (20 messages🔥): 

> `PersonaLLM Workshop, Custom Logit Processors, AI for offensive purposes` 


- **PersonaLLM Workshop call for work**: There's a call for work on persona-driven LLMs across HCI, psychology, cognitive science, culture, and evaluation at the [PersonaLLM Workshop @ NeurIPS Mexico City](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP).
- **No Custom Logit Processors for you!**: Closed source LLM providers don't support custom logit processors because *typically for fast inference logit processes are hard baked into the code* and it's an additional risk to enable arbitrary code execution.
   - One member stated that *They used to and then people started writing papers about how to reverse engineer non-public info about said models using those processors*.
- **AI used for offensive purposes?**: A member asked if the AI is in any way used for offensive purposes ala OpenAI, meta etc etc, and incl. any government contracts, partnerships with other organisations working in offence/war fronts.
   - Another member replied *If you mean the AI models we have trained, the answer is 'not by us.' I can't tell you what militaries or intelligence agencies are doing or whether they're using our models.*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1428563412832616458)** (12 messages🔥): 

> `Midtraining survey, MaskDiT, Attribution graphs from MLPs to attention, LLMs and TREAD` 


- **TREAD Keeps Tokens to Train Deeper**: A member shared a [midtraining survey paper](https://arxiv.org/abs/2510.06826) noting that the tokens are not thrown away, but just processed by fewer layers, differing from MAEs where tokens are discarded, resulting in **MaskDiT**.
   - The member stated that not throwing away all the information is the main contribution of **TREAD**, though noted that MaskDiT works, but *substantially less well*.
- **LLMs Can Use TREAD**: Members discussed the applicability of the **TREAD** method to **LLMs**, expressing uncertainty about the expected outcomes, though it *should work significantly less well than for the image domain*.
   - Another member speculated that even a fractional improvement could still be worthwhile.
- **Attribution Graphs Go to Attention**: A member linked a [YouTube video](https://youtu.be/hdi1a9MjwDs?si=taIuYbeF6v-yRSxI&t=628) discussing the expansion of **attribution graphs** from **MLPs** to **attention**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1428511366775308400)** (27 messages🔥): 

> `Libtorch conversion, PersonaLLM Workshop, UK pricing, Prompt logging policies, GLM 4.6 vs Claude coding` 


- **Libtorch Conversion causes insanity**: A member is struggling with converting **SAM video** to **libtorch**.
   - Another member responded that *he doesn't wanna mess with video demons*.
- **PersonaLLM Workshop calls for work**: The **PersonaLLM Workshop** at **NeurIPS Mexico City** is calling for work on persona-driven LLMs across **HCI**, **psychology**, **cognitive science**, **culture**, and **evaluation**.
   - Submissions include: **demos 2 to 4 pages** with artifact link, **2-page non-archival abstracts**, or **summaries of published work** via [openreview.net](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP&referrer=%5BHomepage%5D(%2F)).
- **UK Pricing Pains**: A member complained about UK pricing, noting that *£3650 works out to about $4901 so im paying like $900 more because wrong country??* and attached a [relevant image](https://cdn.discordapp.com/attachments/1149866623109439599/1428779645712465991/image.png?ex=68f466fc&is=68f3157c&hm=ff6b1c2edd5ec622713b2fa3fe197dc4236b0859c27b9580d1c96e9090d06722&).
- **GLM 4.6 coding competition with Claude**: With the release of **GLM 4.6** to run locally, members anticipate *no more fawning over Sam/Elon/Dario for the OS community*, referencing [this YouTube video](https://www.youtube.com/watch?v=bOfoCocOjfM).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1428850492577939586)** (2 messages): 

> `New Arxiv Paper` 


- **New Arxiv Paper Gets Posted**: A member posted a link to a new potentially interesting [Arxiv paper](https://arxiv.org/pdf/2510.14901).
   - The member stated that *they aren't really sure what to make of it yet*.
- **Placeholder Topic**: This is a placeholder topic to meet the minimum requirement.
   - Further details can be added as more information becomes available.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1428850492577939586)** (2 messages): 

> `Arxiv papers` 


- **Arxiv Paper Discussed**: A member shared a link to an Arxiv paper ([https://arxiv.org/pdf/2510.14901](https://arxiv.org/pdf/2510.14901)).
   - The member expressed uncertainty about how to interpret the paper.
- **Another Arxiv Paper Discussed**: Another member shared a different link to an Arxiv paper ([https://arxiv.org/pdf/2510.14901](https://arxiv.org/pdf/2510.14901)).
   - This second member expressed uncertainty about how to interpret the paper.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1428490463823921203)** (29 messages🔥): 

> `Loading Errors and Agent Mode Issues, Prohibition of Selling Credits, Manus Workshop Promotion, Refund Request, Coffee Shop Tool` 


- **Manus Grapples with Loading Errors in Agent Mode**: Members reported a **loading error** where the system thinks for too long in agent mode and doesn't start tasks.
   - The deployment was failing because **OpenAI** requires *pydantic_core* which needs to be compiled, so a member plans to create a version that works without the OpenAI dependency.
- **Credit Sales are Nixed**: Selling credits is strictly prohibited, and further occurrences may lead to removal.
   - This announcement serves as a warning against unauthorized credit transactions within the platform.
- **Attendee promotes London Manus Workshop**: A member who attended a **Manus workshop in London** is planning to promote it to an industry group.
   - They sought assistance in reaching Manus sales and received a link to the [Manus Help Center](https://help.manus.im/en/) from another member.
- **Refunds are in the prompt-ing**: A member requested a refund for a session that used almost all of their credits but couldn't complete the set task, sharing the [session link](https://manus.im/share/pjJFAsvmMM7rhlBIZ2e0Jh?replay=1).
   - A member advised that refunds aren't automatically granted for failed cases, as reasons for failure can be complex and often related to **prompting**.
- **Java Brews New App for Coffee Connoisseurs**: A member shared a tool, [Workable Cafes](https://workablecafes.com), to help people discover coffee shops based on **wifi speed**, **comfort**, and **outlets**.
   - The app has already been used by over **100 people**, and the creator welcomes feedback.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1428502114857914559)** (24 messages🔥): 

> `Kimi K2 finetuning, Kimi vs Deepseek, Moonshot vs Deepseek` 


- **User contemplates Kimi K2 Finetuning**: A user expressed interest in finetuning **Kimi K2** with **1B** parameters but is concerned about the API cost for **100k** examples.
   - They suggested reducing the dataset to **10k** examples and filtering it, though this might still be expensive.
- **Kimi Preferred Over Deepseek for Concise Outputs**: A user asked which model is better, **Kimi** or **Deepseek**, and another user stated that *Kimi* has *more parameters, better structured outputs, more concise* outputs.
   - The first user clarified that the quality of a model depends on the number of parameters and structured outputs, and they agreed that quality of outputs mattered.
- **Deepseek Advised to Emulate Moonshot**: A user stated that they keep telling **Deepseek** to be more like **Moonshot**.
   - When asked if they got any replies, they did not answer.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1428460293448532020)** (14 messages🔥): 

> `Illegalize Linux, Children Operating Systems, AGI Definition, Emulate Ground Truth Data Distribution, Weekly Tautological Counter` 


- **Linux possibly illegalized!**: Members joked about [Linux being illegalized](https://kernel.org/), with one sarcastically worrying about the children.
   - Another member retorted that children would just write their own operating system and share it.
- **Discussing AGI Definition**: Members discussed the definition of [AGI](https://www.agidefinition.ai/), suggesting it's just a complex question-answering system that can be solved with enough training data.
   - One member linked to [Dan Hendrycks' X post](https://x.com/DanHendrycks/status/1978828377269117007) and [Dimitris Papailiopoulos' X post](https://x.com/DimitrisPapail/status/1978849863174357052?t=kaj5SsZXgdofKoPV_DWsNA&s=19) in relation to this discussion.
- **Weekly Tautological Counter When?!**: A member suggested creating a *weekly tautological counter* to track how often researchers overcomplicate simple concepts.
   - They expressed frustration at researchers managing to complicate the exact same, simple thing in more than one way.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1428610292186611743)** (3 messages): 

> `Qwen3 Vision Model, Open Sourcing Older Models, Protecting Best Tricks` 


- **Qwen3-VL-8B-Instruct Model Drops**: The new [Qwen3 Vision Model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) has been released on HuggingFace.
- **Open Sourcing Quandary**: A member wondered if companies will ever **opensource their older models**.
   - They suspected that they would rather **train a separate one from scratch** than release old versions to protect their best tricks just like OpenAI did.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1428562254198079511)** (2 messages): 

> `Google Coral NPU, Apache 2 Licensing, RV32 Cores, Mojo Portability Testing` 


- **Google Open Sources Coral NPU Verilog**: Google has open sourced the verilog for an **NPU block** under [Apache 2](https://github.com/google-coral/coralnpu).
   - The matrix cores look a bit like AMD's NPUs, but they're **RV32 cores**.
- **Coral NPU a Platform for Mojo Portability**: The newly open sourced **Coral NPU** could be very interesting to use as a platform for testing **Mojo's portability**.
   - It should be possible to simulate this on client hardware.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1428502621685153883)** (13 messages🔥): 

> `TUI Frameworks for Mojo, Audio and MIDI 2.0, Jack Bindings, Mojo Origins vs Rust Lifetimes` 


- **Mojo DAW Dream Sparks 🔥**: Members expressed a strong desire for a **TUI framework** like Textual and full **audio/MIDI 2.0** capabilities in Mojo to create a high-performance **DAW**.
   - One member suggested writing bindings to a library like **Jack** at this point, referencing their [OpenGL experiments](https://link.to.opengl) as an example of an **FFI heavy project**.
- **TUI Framework Inspiration Surfaces!**: A member shared a link to a TUI framework project called [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo).
   - Another member mentioned their paused work on a TUI framework modeled after **ELM apps** like **Bubbletea** in Golang, providing a link to their repo: [banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo).
- **Origins > Lifetimes 🚀**: A user inquired about **lifetimes** in Mojo, comparing them to Rust's `<'a> <'static>` syntax.
   - Members clarified that Mojo has a similar but more ergonomic concept called **Origins**, linking to the [official documentation](https://docs.modular.com/mojo/manual/values/lifetimes/).


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1428507082268737698)** (1 messages): 

> `MAX Python API Open Source` 


- **Modular Open Sources Remainder of MAX Python API**: Modular **open-sourced** the remainder of the **MAX Python API**.
   - A [forum post](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379) lists all of the newly open-sourced Python modules.
- **MAX Python API Availability**: The complete **MAX Python API** is now available to the public as **open-source** software, inviting community contributions and extensions.
   - This move enables developers to deeply integrate **MAX** functionalities within their Python-based projects, enhancing both flexibility and innovation.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1428486759662227508)** (5 messages): 

> `IMAGE, NOLOCALS, and GRAPH_ONE_KERNEL Confusion, DEV= Default Device Setting, Speed Regressions Despite Tests, Generic Compilation Tests, Distributed Systems and GPU Memory` 


- **Flags Cause Configuration Confusion**: The flags **IMAGE**, **NOLOCALS**, and **GRAPH_ONE_KERNEL** caused configuration confusion, as it wasn't obvious what was a real compilation failure and what was a bad config.
   - The suggestion was raised to make these flags fail explicitly if the combination of device/hw is not supported.
- **Default Device Setting in Python Missing**: There's currently no way to set the default device in Python, which would be convenient for cross-checking different backends in a Python script.
   - An example of how this could be implemented is `Device.set_default('CL')` or `Device.DEFAULT = 'CL'`.
- **Speed Tested, Regression Persists**: Despite having tests in place, speed regressions have occurred.
   - Historical data is hard to see, since [https://stats.tinygrad.win/](https://stats.tinygrad.win/) seems to only have data going back 25 days, but the benchmark is working.
- **Compilation Tests Demand Genericity**: The user wants to write tests for the failed compilations, but lacks a good idea for that yet.
   - All the failures are specific combinations of model architecture, device, and in some cases even due to **fp16**, and can't even rely on **ORT** for verification since that also produces wrong results in the FP16 case.
- **Distributed Systems Seek Tiny Fans**: Someone dabbling with distributed systems is seeking to chat with anyone in the **GPU memory** or **CRIU** space.
   - They ask if anyone knows anyone in the distributed GPU space.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1428529015815409796)** (4 messages): 

> `aider fork with mcp support, GPT-5-nano removed from commit messages in aider-ce` 


- **Aider-CE Fork Supports MCP**: A user asked about an **aider fork** that supports **MCP (Multi-Control Protocol)**, and another user recommended [aider-ce](https://github.com/dwash96/aider-ce/).
   - The command to start aider with a specific model and edit format is: `aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff`.
- **Aider-CE drops GPT-5-nano for commit messages**: A user switched to **aider-ce** for the latest features and **GPT-5-code** support, but noticed that **aider** no longer mentions using **gpt-5-nano** for commit messages.
   - They inquired whether this change was intentional.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1428461227318968452)** (1 messages): 

> `Filename Typing, Feature Requests, Aider Performance` 


- **Filename Typing Auto-Adds Files!**: A member noted that typing a filename after a message (e.g. *"see also SomeFile.txt"*) prompts the system to ask to add files.
- **Aider Feature Wishlist Grows**: A user suggested an **aider** feature request for adding files, with more discussion and details to follow later.
   - Another member mentioned they'd make a feature request to allow **aider** to automatically add local git ignored files.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1428813254041403584)** (2 messages): 

> `MLOps Workshop, LWP Labs, ML Model Deployment` 


- **LWP Labs Launches Free MLOps Workshop**: LWP Labs is launching a **free 3-day MLOps workshop** to teach participants how to deploy machine learning models into real-world production, covering **Docker, CI/CD, MLflow, and AWS**.
   - The workshop promises to offer real-world deployment practices and includes **five hands-on projects** to enhance resumes.
- **Industry Expert to Lead MLOps Training**: An MLOps workshop will be led by an instructor with **15+ years** of industry experience, aiming to equip participants with in-demand skills.
   - The course emphasizes **practical knowledge and hands-on experience**, ensuring attendees gain skills sought after by employers in AI and Data Engineering.

