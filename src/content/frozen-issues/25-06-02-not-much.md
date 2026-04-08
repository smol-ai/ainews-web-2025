---
id: MjAyNS0w
title: not much happened today
date: '2025-06-02T05:44:39.731046Z'
description: >-
  **DeepSeek R1-0528** release brings major improvements in reasoning,
  hallucination reduction, JSON output, and function calling, matching or
  surpassing closed models like **OpenAI o3** and **Gemini 2.5 Pro** on
  benchmarks such as **Artificial Analysis Intelligence Index**, **LiveBench**,
  and **GPQA Diamond**. The model ranks #2 globally in open weights
  intelligence, surpassing **Meta AI**, **Anthropic**, and **xAI**. Open weights
  and technical transparency have fueled rapid adoption across platforms like
  **Ollama** and **Hugging Face**. Chinese AI labs including **DeepSeek**,
  **Alibaba**, **ByteDance**, and **Xiaomi** now match or surpass US labs in
  model releases and intelligence, driven by open weights strategies.
  Reinforcement learning post-training is critical for intelligence gains,
  mirroring trends seen at **OpenAI**. Optimized quantization techniques (1-bit,
  4-bit) and local inference enable efficient experimentation on consumer
  hardware. New benchmarks like **LisanBench** test knowledge, planning, memory,
  and long-context reasoning, with **OpenAI o3** and **Claude Opus 4** leading.
  Discussions highlight concerns about benchmark contamination and overemphasis
  on RL-tuned gains.
companies:
  - deepseek_ai
  - openai
  - gemini
  - meta-ai-fair
  - anthropic
  - x-ai
  - ollama
  - hugging-face
  - alibaba
  - bytedance
  - xiaomi
models:
  - deepseek-r1-0528
  - o3
  - gemini-2.5-pro
  - claude-opus-4
topics:
  - reasoning
  - reinforcement-learning
  - benchmarking
  - quantization
  - local-inference
  - model-evaluation
  - open-weights
  - transparency
  - post-training
  - agentic-benchmarks
  - long-context
  - hallucination-detection
people:
  - teortaxestex
  - wenfeng
  - danielhanchen
  - awnihannun
  - reach_vb
  - abacaj
---


**A quiet weekend.**

> AI News for 6/2/2025-6/3/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 9059 messages) for you. Estimated reading time saved (at 200wpm): 852 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Sorry this arrived late but it was a pretty quiet day. AIE sold out so we launched [the AI Engineer online track](https://www.latent.space/p/aiewf-2025) and you can [bookmark the keynotes and MCP livestream track](https://www.youtube.com/watch?v=z4zXicOAF28) now for the YouTube algorithm to do its thing, as it is likely to be the biggest stream in AIE history.

https://www.youtube.com/watch?v=z4zXicOAF28

---

# AI Twitter Recap

**1. Foundation Model Advances: DeepSeek R1-0528, Benchmarks, and Open Weights Leadership**

- **DeepSeek R1-0528 release, open weights, and benchmark performance**: The release of [DeepSeek-R1-0528](https://x.com/deepseek_ai/status/1928061589107900779) brought major improvements in reasoning, reduced hallucinations, JSON output, and function calling. The model matches or surpasses [leading closed models like OpenAI o3 and Gemini 2.5 Pro](https://x.com/ArtificialAnlys/status/1928071179115581671) on several benchmarks, including [Artificial Analysis Intelligence Index](https://x.com/ArtificialAnlys/status/1928071179115581671), [LiveBench](https://x.com/scaling01/status/1928173385399308639), and [GPQA Diamond](https://x.com/EpochAIResearch/status/1928489527204589680) ([DeepSeek R1-0528 scores 76%](https://x.com/EpochAIResearch/status/1928489527204589680)). [Artificial Analysis](https://x.com/ArtificialAnlys/status/1928071179115581671) highlights that DeepSeek is now tied for the #2 spot globally in open weights intelligence, surpassing Meta, Anthropic, and xAI.
- **Open weights and technical transparency fueling adoption**: [DeepSeek's approach](https://x.com/ArtificialAnlys/status/1928477941715079175) of open-sourcing weights, code, and research targets has enabled rapid global adoption, with multiple platforms ([Ollama, Hugging Face, OpenRouter, etc.](https://x.com/ollama/status/1928543644090249565), https://x.com/awnihannun/status/1928125690173383098, https://x.com/basetenco/status/1928195639822700898) quickly incorporating the model for inference and experimentation.
- **Chinese AI labs match or surpass US labs**: The [Artificial Analysis Q2 State of AI China report](https://x.com/ArtificialAnlys/status/1928477941715079175) finds that Chinese labs (DeepSeek, Alibaba, ByteDance, Xiaomi, and more) now release models within weeks of US counterparts and achieve parity or superior intelligence. Open weights strategy underpins this progress.
- **Reinforcement Learning and post-training drive rapid gains**: [DeepSeek's intelligence jumps are driven by RL post-training](https://x.com/ArtificialAnlys/status/1928071179115581671), mirroring OpenAI's 10x RL scaling trend between o1 and o3. [Artificial Analysis](https://x.com/ArtificialAnlys/status/1928071179115581671) and [EpochAIResearch](https://x.com/EpochAIResearch/status/1928489524616630483) show RL’s criticality for efficient intelligence gains.
- **Optimized quantization and local inference**: Rapid quant releases ([danielhanchen's 1-bit and 4-bit quants](https://x.com/danielhanchen/status/1928278088951157116), [awnihannun's 4-bit DWQ for Qwen3 8B](https://x.com/awnihannun/status/1928125690173383098), [reach_vb’s MLX quant](https://x.com/reach_vb/status/1928002892633383338)) and real-time deployment ([Ollama thinking mode for DeepSeek](https://x.com/ollama/status/1928543644090249565)) enable efficient experimentation even on consumer hardware.
- **Community, transparency, and model culture**: [teortaxesTex](https://x.com/teortaxesTex/status/1927919610612875492) and [ArtificialAnlys](https://x.com/ArtificialAnlys/status/1928477951365939328) emphasize DeepSeek’s transparency, minimal marketing, and technical focus as differentiators, with [Wenfeng’s vision](https://x.com/teortaxesTex/status/1927918495125172621) driving long-term progress.

**2. Model Evaluation, Reasoning, Benchmarking, and RL**

- **New reasoning and agentic benchmarks**: [scaling01 introduces LisanBench](https://x.com/scaling01/status/1928510435164037342), a scalable test for knowledge, planning, memory, and long-context reasoning, showing o3 and Claude Opus 4 as leaders. [LiveBench](https://x.com/scaling01/status/1928173385399308639) now includes agentic coding, with DeepSeek R1-0528 ranking 8th overall and 1st in data analysis.
- **Post-training and RL saturation**: [lateinteraction](https://x.com/lateinteraction/status/1928148705145934252) critiques benchmark contamination and the overemphasis on RL-tuned gains, warning that much of the recent apparent progress may be due to prompt/template alignment rather than general capability. [abacaj](https://x.com/abacaj/status/1927948317931000277) notes the field’s focus on Qwen for RL experiments. [giffmana](https://x.com/giffmana/status/1928314882761334871) and [vikhyatk](https://x.com/vikhyatk/status/1928268671979565330) express skepticism about the reliability and impact of recent RL papers.
- **Interpretability and open-source tools**: [Anthropic releases open-source circuit tracing tools](https://x.com/AnthropicAI/status/1928119229384970244), enabling researchers to generate attribution graphs for LLM internals ([mlpowered demo](https://x.com/mlpowered/status/1928123130725421201), [NeelNanda5 commentary](https://x.com/NeelNanda5/status/1928169762263122072)), a move celebrated for transparency and reproducibility.
- **Model reasoning improvements and architectural innovation**: [Ollama introduces “thinking” separation for DeepSeek](https://x.com/ollama/status/1928543644090249565), making reasoning traceable. [cline](https://x.com/cline/status/1928208680903921803) demonstrates that “Extended Thinking” and “Sequential MCP” structures boost Claude’s reasoning performance by up to 68%. [StasBekman](https://x.com/StasBekman/status/1928571964647682400) highlights shift parallelism for inference optimization.

**3. Multimodal AI, Agents, and Tooling**

- **Perplexity Labs launch and agentic workflows**: [Perplexity launches Labs](https://x.com/perplexity_ai/status/1928141072011776088), enabling users to build complex dashboards, code tools, and apps with a prompt. [AravSrinivas](https://x.com/AravSrinivas/status/1928220532614537318) demonstrates that “Financial Researcher/Analyst is now a prompt,” and [Labs’ dynamic UI](https://x.com/AravSrinivas/status/1928192558586315119) and “deep research” features are rapidly adopted.
- **Vision-Language Models and SOTA VLMs**: [MiMo-VL-RL from Xiaomi](https://x.com/mervenoyann/status/1928475979753619663) outperforms GPT-4o on GUI navigation and reasoning, with open weights and MIT license ([reach_vb benchmarking](https://x.com/reach_vb/status/1928360066467439012)). [FLUX.1 Kontext](https://x.com/iScienceLuvr/status/1928186905079992507), a new SOTA image editing/generation model, is released by Black Forest Labs ([togethercompute free demo](https://x.com/togethercompute/status/1928527563791441993)), excelling at character consistency and in-context editing.
- **Video and robotics models**: [Google’s Veo 3](https://x.com/Google/status/1928573869893230705) is now available in 73 countries and tops both Image-to-Video and Text-to-Video leaderboards ([ArtificialAnlys comparison](https://x.com/ArtificialAnlys/status/1928318831761707224)). [ClementDelangue](https://x.com/ClementDelangue/status/1928125034154901937) announces open-source robots, and [TheRundownAI](https://x.com/TheRundownAI/status/1928104195279749526) covers rapid progress in humanoids and robotics platforms.

**4. AI Infrastructure, Scaling, and Hardware**

- **TPU, GPU, and inference scaling**: [demishassabis highlights Google’s SREs/Infra teams](https://x.com/demishassabis/status/1928604371157233918) for keeping TPUs running under Gemini/Veo demand. [StasBekman](https://x.com/StasBekman/status/1928571964647682400), [tri_dao](https://x.com/tri_dao/status/1928170648863473892) discuss hardware-aware architecture choices for optimal inference (GQA, MLA, GLA). [danielhanchen](https://x.com/danielhanchen/status/1928278088951157116) and [awnihannun](https://x.com/awnihannun/status/1928125690173383098) enable large models to run on consumer hardware via aggressive quantization.
- **Sovereign AI and datacenter strategies**: [JonathanRoss321](https://x.com/JonathanRoss321/status/1928241967122506083) explains Canada’s adoption of Groq for sovereign AI infra. [saranormous](https://x.com/saranormous/status/1928479931660411033) and [AndrewYNg](https://x.com/AndrewYNg/status/1928099650269237359) discuss the importance of national investment in research, talent, and local infrastructure for maintaining competitiveness.

**5. AI Agents, Memory, and Workflow Orchestration**

- **Agentic system architectures and memory**: [LangChainAI](https://x.com/LangChainAI/status/1928135137658818711) and [omarsar0](https://x.com/omarsar0/status/1928492639906607297) detail DAG-based agent architectures for robust workflow orchestration and memory-centric agent design ([Omarsar0 MemOS abstraction](https://x.com/omarsar0/status/1928116365640225222)). [sjwhitmore](https://x.com/sjwhitmore/status/1928520064078328193) tests Spark’s long-term memory and behavior design.
- **Self-improving agents**: [SakanaAILabs introduces the Darwin Gödel Machine (DGM)](https://x.com/SakanaAILabs/status/1928272612431646943), a self-improving coding agent that can rewrite its own code, boosting SWE-bench performance from 20% to 50%. [hardmaru](https://x.com/hardmaru/status/1928284568756629756) and [SakanaAILabs](https://x.com/SakanaAILabs/status/1928447873362153669) explain concepts and open-ended evolution for agent improvement.

**6. Memes, Humor, and Community Vibes**

- **Memes and lighthearted takes**: [ID_AA_Carmack embraces the “Parmesan” archetype for greybeards](https://x.com/ID_AA_Carmack/status/1928239003397984389), [swyx’s “fix the bug at Apple” plan](https://x.com/swyx/status/1928512178941808838) receives high engagement, and [awnihannun jokes about DeepSeek’s overthinking](https://x.com/awnihannun/status/1928119439737729482). [nearcyan’s medical AI system meme](https://x.com/nearcyan/status/1928620490416906430) satirizes the inefficiencies in healthcare versus LLMs. [TheZachMueller’s “tech couple” poll](https://x.com/Yuchenj_UW/status/1928502759227080841) and [DavidSHolz’s poetic SpaceX memory](https://x.com/DavidSHolz/status/1928189415291245040) go viral, reflecting the community’s blend of technical rigor and humor.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. New Model Quantization Techniques and Local Model Performance

- [**IQ1_Smol_Boi**](https://i.redd.it/9u1teeqt4g4f1.png) ([Score: 378, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1l19yud/iq1_smol_boi/)): **The image is a meme-style illustration contrasting two quantized language model variants: R1-671B fp8 (depicted as a muscular figure), and IQ1_S (shown as a comically distorted figure), visually exaggerating their differences in power and sophistication. The post provides context for the creation of a highly compact 'IQ1_S_R4' quantization of DeepSeek R1-0528 (131GiB, suitable for 128GiB RAM + 24GB VRAM systems), which demonstrates lower perplexity (better, though architecture-dependent) compared to the much larger Qwen3-235B-A22B-Q8_0 (232.769GiB, PPL=5.3141). Alternatives like Unsloth's UD-TQ1_0 (151GiB) and Bartowski's IQ1_M (138GiB) are also discussed, highlighting emerging tradeoffs in bit rates, model size, and perplexity for users with limited hardware. Notably, actual utility and perplexity comparison is debated, as cross-architecture PPL benchmarks may be misleading.** Commenters technically debate the utility of perplexity as a quality metric for quants, referencing Unsloth's alternative suggestions. Personal benchmarks and anecdotes also compare performance, speed, and practical use across DeepSeek and Qwen architectures, emphasizing subjective and dependent factors in real-world usage.
    - Technical discussion centers around quantization benchmarks: the DeepSeek-R1-0528-GGUF repository contains new quantization perplexity metrics for IQ1_Smol_Boi variants, with direct comparison to other large models (e.g., Qwen3-235B-A22B-Q8_0 achieves PPL=5.3141±0.03321 at 232.769 GiB). However, authors caution that perplexity is not directly comparable across architectures, and there's debate over the appropriateness of perplexity as a quantization quality metric.
    - Implementation/practical advice is given for using the TQ1_0 model (162GB) with Ollama: it's noted to work without gguf-merging, and specific parameters/settings are outlined (including chat template, temperature, and the suggestion to offload certain FFN layers to CPU for users with constrained RAM). The author also explains that the original IQ1_S (185GB) retained higher precision in some modules because 1-bit dynamic quantization makes the model unusable if too aggressively quantized, which is supported by PPL plots.
    - There is mention of a tradeoff in model size and quantization aggressiveness: the unsloth IQ1_S (186GB) compared to baseline quantization recipes (e.g., Q8_0 and Q4_0) and a trimmed down TQ1_0 (162GB), reflecting the challenge of preserving model accuracy when reducing file size. Anecdotal runtime performance is shared: speeds increased from 0.5 t/s to much better throughput in 'the dynamic quant era'.
- [**Which model are you using? June'25 edition**](https://www.reddit.com/r/LocalLLaMA/comments/1l1581z/which_model_are_you_using_june25_edition/) ([Score: 195, Comments: 139](https://www.reddit.com/r/LocalLLaMA/comments/1l1581z/which_model_are_you_using_june25_edition/)): **The monthly community check-in surveys current popular LLMs for various tasks, highlighting increased adoption of Qwen 3 32B Q4/Q8_K_XL (for coding, reasoning, and general tasks at extended context windows, e.g.,** `36K ctx` **and** `49K Ctx`**), Qwen 2.5 Coder 32B Q8_K (Code FIM), and Gemma 27B QAT Q8_K_XL (for creative writing/translation/vision). Proprietary and open-weight LLMs are compared for use in agentic coding (e.g. Devstral 'UD-Q6_K_XL'), conversational agents, and code auto-completion, with some favoring lighter Qwen 3 variants (4B for Cotypist). Recent models like DeepSeek-R1-0528 and Claude 4 are noted for comparison, but adoption specifics reported center heavily on Qwen and Gemma variants.** Debate centers on context length support and quantization trade-offs (Q4, Q6_K, Q8_K_XL), with some users prioritizing higher quantization for x86 performance and others for mobile/edge deployment. Preference for Qwen models is attributed to their performance in general reasoning and coding, while Gemma is seen as strong for writing/translation despite smaller parameter count. The role of agentic frameworks (Devstral, Cotypist) is discussed for workflow integration.
    - Multiple commenters detail specific model choices by use case: for code completion, high-context coding, and agentic interactions, models like Qwen 2.5/3 32B (e.g., Q8_K at up to 49K context) and Gemma 27B QAT Q8_K XL are favored, highlighting recent improvements in large context handling and specialization. Variants and quant settings (Q4, Q6, Q8) are specified for tailoring to hardware and context needs.
    - A user with 8GB VRAM discusses practical limitations and systematic testing of sub-8B parameter models. Deepseek-R1-0528-Qwen3-8B stands out as 'the smartest' tested, and Josiefied-Qwen3-8B is praised for its 'unbiased and uncensored' fine-tuning. A key technical insight is the wide variation in context consumption between models: Deepseek processed an 8,000-token prompt effectively, while Qwen3-8B and Josiefied-Qwen3-8B used fewer tokens and gave differing performance, showing prompt handling and efficiency tradeoffs.
    - Gemma3-12B is specifically singled out as the best for Retrieval-Augmented Generation (RAG), web search, and quick queries, indicating ongoing improvements in small and mid-size language models for specialized reasoning and search tasks. Several responses echo a trend towards smaller models (<8B) achieving much higher quality recently, driven by continued refinement and specialized fine-tuning.

### 2. Commentary on Open Source AI Ecosystem Competition

- [**Ignore the hype - AI companies still have no moat**](https://river.berlin/blog/there-is-still-no-moat/) ([Score: 230, Comments: 163](https://www.reddit.com/r/LocalLLaMA/comments/1l1e6ic/ignore_the_hype_ai_companies_still_have_no_moat/)): **The post and linked article assert that technical moats are eroding in AI, citing that open-source alternatives exist for virtually every major generative and LLM-based tool (text-to-speech, code, retrieval-augmented generation, etc.), and the performance gap between SOTA open-source and proprietary foundation models has narrowed to approximately 10%. The author highlights rapid architectural and interface changes, model instability, and recurring inefficiencies (e.g., limited context management necessitating domain-specific fine-tuning), arguing that sustainable advantage is increasingly a function of network effects and reach rather than core technical capabilities.** Top comments debate this premise, noting that true moats may exist only for compute infrastructure providers (e.g., cloud, hardware), and comparing the situation to early open-source software where usability, cost, and maintenance—not just technical parity—limit replacement of commercial products. Others emphasize that the only durable moat in AI is ownership of training infrastructure, contending that open access to algorithms does not counterbalance the enormous resource requirements for large-scale model training.
    - Several commenters emphasize that the true moat in the AI industry lies primarily with infrastructure providers and hardware manufacturers, rather than with AI application companies. This is because large-scale model training and deployment require vast capital investment and specialized hardware, creating high barriers to entry.
    - Another technical point raised is that access to proprietary, large-scale datasets forms a significant competitive advantage for some firms. For example, Google's control over YouTube data enables development of advanced models like VEO 3–5, which are hard to replicate without similar breadth and diversity in video data. The closure of public YouTube API is cited as a move to reinforce this data moat, highlighting how strategic data access surpasses algorithmic advances alone.
    - A nuanced discussion draws parallels to open source software, arguing that technical equivalence (e.g., open models matching closed ones) isn’t sufficient for disruption. Real-world viability depends on factors like training costs, compute infrastructure, data maintenance, and time-to-market, making it difficult for purely open alternatives to compete at massive scale.
- [**At the airport people watching while I run models locally:**](https://i.redd.it/55ab38z0ck4f1.jpeg) ([Score: 829, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1l1qqdx/at_the_airport_people_watching_while_i_run_models/)): **The post uses a meme format to highlight the niche yet empowering technical capability of running large language models (LLMs) like DeepSeek 8B locally on personal hardware. The discussion underscores the contrast between the general public's lack of awareness and the technically advanced practice of offline LLM inference, implying significant local compute resources are required (i.e., not every laptop is suitable, as referenced in the poster's exchange).** Commenters inject technical humor referencing local LLM artifacts ('My local DeepSeek 8B wanted me to tell you...'), suggesting quirky model outputs can occur, and acknowledge the security optics ('The TSA watching you run weird hacking programs') of such activity in public spaces.
    - The DeepSeek 8B model is highlighted for its ability to solve problems at a level comparable to current top LLMs, while also being able to run locally—a significant advantage for privacy and offline applications.
    - A user shared an example of successfully running the Qwen3 4B model directly on their phone during a flight, demonstrating the increasing feasibility of running capable LLMs on everyday consumer hardware (such as smartphones), which is a notable hardware/software advancement for on-device AI.
    - The option to run models like DeepSeek and Qwen locally is discussed as a game changer, but also has practical limitations—users with less powerful laptops (e.g., casual non-technical users) may not be able to effectively run these models due to hardware constraints.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. OpenAI/Claude Next-Gen Model Release Rumors And Public Messaging

- [**GPT-5 in July**](https://i.redd.it/e5cwucgu0i4f1.jpeg) ([Score: 361, Comments: 104](https://www.reddit.com/r/singularity/comments/1l1fi7a/gpt5_in_july/)): **The image shows a Twitter exchange where credible figures (Tibor Blaho and Derya Unutmaz, known for accurate commentary and OpenAI collaboration) indicate with apparent confidence that GPT-5 will launch in July, countering speculation of an August release. This lends additional credence to rumors of an imminent GPT-5 release, which may inform planning for researchers and developers tracking foundational model capabilities and deployment timelines.** Technical commenters are skeptical about the significance of the upgrade (expecting only a marginal leaderboard improvement versus competition) and express wishlist features like a logic for a much larger context window (1M tokens), hinting at anticipated technical bottlenecks or priorities for future LLMs.
    - There is some skepticism about GPT-5's potential impact compared to competitors, with the view that any performance increase may soon be outpaced by alternatives such as Google's models, highlighting the competitive pacing in LLM leaderboard rankings.
    - One comment expresses hope that GPT-5 will feature a significantly larger context window (e.g., 1M tokens or more), suggesting expectations of architectural or scaling advances beyond current limits seen in commercial models.
    - The anticipated completion of OpenAI's 'stargate project' by mid-2026 is referenced as a potential turning point for much larger gains in model performance, implying that infrastructure or hardware improvements could drive major future advances rather than current model iterations alone.
- [**Sam Altman says the world must prepare together for AI’s massive impact - OpenAI releases imperfect models early so the world can see and adapt - "there are going to be scary times ahead"**](https://v.redd.it/d1sc3d1ykh4f1) ([Score: 844, Comments: 506](https://www.reddit.com/r/singularity/comments/1l1e20p/sam_altman_says_the_world_must_prepare_together/)): **Sam Altman (OpenAI CEO) emphasized the need for global preparedness for rapid AI advancement, noting OpenAI's strategy of releasing imperfect models early to allow societal adaptation and transparency. Altman acknowledged the significant potential for social disruption and explicitly warned of 'scary times ahead' as these systems evolve ([Wisdom 2.0 panel discussion](https://www.youtube.com/watch?v=ZHz4gpX5Ggc)).** Technical discussion in the comments centers on the realistic risk of widespread automation of white-collar jobs, raising skepticism regarding the promised 'era of abundance.' A user also notes Altman has issued similar warnings in the past, suggesting ongoing concern rather than new insight.
    - One user claims OpenAI releases their models early not purely for adaptation purposes but due to competitive pressure in the AI industry. The comment suggests that OpenAI's model releases may be motivated by a desire to avoid public perception of falling behind in the so-called "AI arms race." This implies that technical readiness may be sacrificed for market positioning.
    - A technical concern is raised regarding potential impact on labor markets, with the observation that current AI capabilities are already capable of matching many white-collar job functions. The commenter expresses skepticism regarding optimistic outlooks like the 'era of abundance,' instead predicting significant job displacement due to AI automation.
- [**Sam Altman: "There are going to be scary times ahead" - OpenAI CEO says the world must prepare for AI's massive impact. Models are released early on purpose so society can see what's coming and adapt.**](https://v.redd.it/wijxxkjoyh4f1) ([Score: 259, Comments: 166](https://www.reddit.com/r/ChatGPT/comments/1l1fask/sam_altman_there_are_going_to_be_scary_times/)): **Sam Altman, CEO of OpenAI, asserts that AI models are released early intentionally to ensure society can anticipate and adapt to their disruptive effects, given the expected massive economic and social impact. He emphasizes the need for government intervention and societal adaptation as AI-driven automation threatens to displace traditional employment across sectors at an accelerating pace. Some comments contextualize this strategy as less about transparency and more about competitive urgency, since AI labs like OpenAI are in a race with rivals to establish technical and market dominance. Others discuss the technical-economic feedback loop, where automation's productivity gain may be offset by reduced consumer demand as humans lose purchasing power, highlighting risks to the entire economic system if mass unemployment is not addressed via policy or economic redesign. The lack of accurate LLMs is already affecting labor markets, according to user observations.** Key technical debates focus on whether early-release rationale is genuine or a competitive necessity, and on the practical risks of rapid, unchecked automation leading to economic crises. There is consensus that governments are lagging on necessary policy reform, and skepticism that existing models are sufficiently accurate to warrant the present level of labor disruption.
    - One commenter notes that the rationale of releasing AI models early as a societal adaptation mechanism is undermined by the competitive landscape: "Now it's a race between several major competitors" in AI, all racing to release as soon as possible to establish dominance, which pressures actors to prioritize speed over careful rollout, regardless of societal readiness.
    - Multiple commenters discuss the economic impact of advanced AI, highlighting concerns about rapid automation leading to mass unemployment (potentially 10-20% or more), loss of consumer bases, and political instability. The feedback loop is technical: companies automate to reduce costs, but if too many are unemployed, aggregate demand collapses, undermining ROI on automation investments; this is an unsolved macroeconomic challenge exacerbated by exponential AI progress.
    - Another technical point raised is that current-generation LLMs, despite their limitations ("frankly not that accurate"), are already causing layoffs and disruptions, showing that the threat is not speculative or only from future, more advanced models, but is manifesting now, pushing the job market toward instability.
- [**I’d like to remind everyone that this still exists behind closed doors…**](https://x.com/sama/status/1899535387435086115?s=46) ([Score: 165, Comments: 63](https://www.reddit.com/r/singularity/comments/1l1ipdf/id_like_to_remind_everyone_that_this_still_exists/)): **The post speculates that OpenAI possesses more advanced, unreleased models (e.g., Sora2, 'o4-full', 'o4 Pro') and features (such as advanced voice mode and in-platform music generation) that may integrate greater creative depth, larger context windows, and orchestration across modalities in a future flagship model, presumably 'GPT-5'. It references Sam Altman's [tweeted demo](https://x.com/sama/status/1899535387435086115?s=46), which showcased a model capable of generating sophisticated metafictional writing, suggesting qualitative leaps in literary and narrative ability.** Commenters debate OpenAI's closed, subscription-based model versus Google's incremental, freely accessible LLM development (e.g., Gemini), with some considering Google's approach strategically superior for broad, practical impact, while others focus on anticipation of novel creative models from OpenAI.
    - OpenAI's current models, such as GPT-4.5, are limited by restrictive prompt caps (e.g., "only 20 prompts a month as a paid subscriber"), which impacts their usability for power users. In contrast, Google's Gemini 2.5 is noted to have fewer access restrictions, providing a more seamless experience for extended use cases.
    - Google's deployment strategy is highlighted as quietly advancing past OpenAI. By offering high-performing models (like Gemini 2.5) with fewer limitations and exploring ad revenue models, Google could possibly outcompete OpenAI's subscription-based approach, especially if the technical gap continues to narrow or favor Google.
    - Sergey Brin recently disclosed that Gemini models have included native audio output capabilities for over a year, but Google chose not to release this feature publicly until now. This demonstrates that leading AI firms often withhold state-of-the-art capabilities from general release, meaning public access lags far behind internal advancements.
- [**It has now been officially 10 days since Sam Altman has tweeted, his longest break this year.**](https://www.reddit.com/r/singularity/comments/1l1vfqe/it_has_now_been_officially_10_days_since_sam/) ([Score: 153, Comments: 39](https://www.reddit.com/r/singularity/comments/1l1vfqe/it_has_now_been_officially_10_days_since_sam/)): **The post highlights that Sam Altman, CEO of OpenAI, has not tweeted in 10 days, marking his longest absence on the platform in 2024. Speculation arises that this could indicate significant ongoing projects or responses to recent industry events such as Google I/O.** Commenters suggest the absence might be due to OpenAI preparing a major release or dealing with competitive pressure, with at least one referencing the possibility of an "AI winter" while another suggests intensive internal activity post-Google I/O.
    - A comment outlines upcoming releases from OpenAI, specifically mentioning that MCP is expected this week and GPT-5 is likely next month. The commenter highlights OpenAI's strategy of pushing reasoning updates for GPT-5 every three months, with an eventual goal tied to the Stargate system rollout, suggesting a cadence of continuing major system improvements.
    - Another technical point is speculation about the open-sourcing of a creative writing model previously demoed by Sam Altman. The inference is that releasing this model soon could allow OpenAI to refocus all resources and attention on GPT-5 development afterward.
    - There's mention of pending modality features, with a commenter referencing the anticipated addition of text-to-audio capabilities, which have yet to be released, indicating community interest and expectation for multimodal model progress.
- [**How much would a Manhattan Project 2.0 speed up AGI**](https://i.redd.it/6rzieryyoe4f1.jpeg) ([Score: 840, Comments: 232](https://www.reddit.com/r/singularity/comments/1l142un/how_much_would_a_manhattan_project_20_speed_up_agi/)): **The image is a screenshot of a tweet from the U.S. Department of Energy that frames AI as the 'next Manhattan Project' for the nation, stating 'THE UNITED STATES WILL WIN.' The original Manhattan Project was a top-secret operation to develop nuclear weapons, only revealed post-WWII—prompting considerations of secrecy and pace in AI development. The discussion context implies a comparison: a 'Manhattan Project 2.0' for AGI might entail intense, government-coordinated, secretive efforts to accelerate AGI progress, potentially leading to transformative advances or security concerns about transparency and oversight.** Comments question the appropriateness of public declarations about national AI projects, suggesting such tweets contrast with the historical secrecy of the original Manhattan Project. Some note the Department of Energy should focus on modernizing critical infrastructure (like the energy grid) instead of making grand public statements about AI supremacy.
    - Several commenters clarify that the original Manhattan Project was highly secretive during its execution and only made public after the results (atomic bombings) in 1945, stressing that a hypothetical 'Manhattan Project 2.0' for AGI would differ contextually since secrecy would be harder in today’s open and networked scientific environment. This underscores the difficulty of containing advanced AI research compared to WWII nuclear research due to the global, distributed nature of the AI community and rapidly disseminated information.

### 2. Hands-on Experiences With Recent Large Language Models and AI Tools

- [**I'm honestly stunned by the latest LLMs**](https://www.reddit.com/r/singularity/comments/1l16zyb/im_honestly_stunned_by_the_latest_llms/) ([Score: 475, Comments: 133](https://www.reddit.com/r/singularity/comments/1l16zyb/im_honestly_stunned_by_the_latest_llms/)): **The OP reports a major leap in LLM code understanding and transformation: while older top models (GPT-4, Claude 3.5, Gemini 2.0) failed at modifying a classical lexer to support indentation-based blocks instead of braces, Claude 4 succeeded instantly and also provided a clear explanation, suggesting improved symbolic reasoning and context management in advanced LLMs like Claude 4.** One comment corrects the versions/timeline, noting the OP's referenced model lineup does not match recent releases (e.g. GPT-4.1, Claude 3.7, Gemini 2.5 Pro). Another argues LLM growth will drive demand for agent builders rather than replace programmers, highlighting the rise of 'micro agents' that automate multi-step language-to-data tasks, while a contrary view claims programming as a career is becoming obsolete.
    - There is some confusion and debate regarding the exact versions and timelines of major LLM releases. One user highlights that as of a month ago, most people were using versions like GPT-4.1, Claude 3.7, and Gemini 2.5 Pro, not the latest (Claude 3.5, Gemini 2.0) mentioned in the post, suggesting that references to newer models may be premature or misleading when discussing real-world capability and adoption timelines.
    - A technical perspective is offered on agent development with LLMs: there is a predicted surge in demand for those who can efficiently design, program, and manage LLM-based 'micro agents' for business processes. Specific example workflows include converting natural language exchanges into structured database interactions and automating multi-step communications, positioning LLMs as brokers for information flow in practical enterprise automation tasks.
    - A user observes that LLM response behavior can be inconsistent: models often either solve a problem instantly or start producing multiple, increasingly poor iterations if initial attempts fail. Restarting the chat context can sometimes reset and yield correct results, highlighting session state as a significant technical factor influencing LLM reliability and output quality.
- [**After 6 months of daily AI pair programming, here's what actually works (and what's just hype)**](https://www.reddit.com/r/ClaudeAI/comments/1l1uea1/after_6_months_of_daily_ai_pair_programming_heres/) ([Score: 262, Comments: 42](https://www.reddit.com/r/ClaudeAI/comments/1l1uea1/after_6_months_of_daily_ai_pair_programming_heres/)): **After 6 months of daily AI pair programming, the OP reports maximum productivity by structuring workflows such as: (1) having AI draft and critique plans before implementation, (2) enforcing edit-test loops with failing tests and iterative fixes (akin to TDD where AI acts as implementer), and (3) referencing file segments via paths/ranges instead of dumping large codebases into the prompt, minimizing context bloat and maximizing relevance. The post emphasizes that successful users employ explicit, disciplined workflows rather than relying on prompt engineering, and insists on reserving architectural decisions for humans. Full workflow details are linked in [the original writeup](https://forgecode.dev/blog/ai-agent-best-practices/).** Top commenters corroborate the workflow's value—advising against having AI select libraries due to reliability issues, and recommending well-documented test conventions (e.g., a dedicated TESTING.md file) to steer AI behavior. There is agreement that AI excels at implementation and test writing when fed explicit plans, but struggles with monolithic files and excessive mocking unless guided by clear, external test philosophies. The consensus is that disciplined processes with clear architectural input from humans vastly outperform unstructured or prompt-heavy approaches.
    - Several users highlight practical limitations and workflows for AI code generation: do not let AI choose libraries autonomously, as it often produces suboptimal selections or errors. Instead, have AI suggest options while you maintain control over architectural decisions.
    - Integration testing with AI assistance requires extra guardrails; models like Claude tend to overuse mocks to make tests pass. Users recommend maintaining a detailed `TESTING.md` with philosophies (e.g., favoring real Redis instances over mocks) and instructing the AI to reference it, plus always creating regression tests after bug discoveries and ensuring all tests pass before merging to main.
    - Current AI code assistants (e.g., Copilot in VS Code) default to the most common, not the most optimal or performant, solutions given their training data bias, and are limited in debugging complex, context-rich problems. They may unpredictably edit unrelated code due to probabilistic completion, and consistently require human oversight—making them less reliable for unique or non-trivial engineering tasks.
- [**My first project using Claude Code, it is just amazing**](https://www.reddit.com/gallery/1l1qn5z) ([Score: 137, Comments: 34](https://www.reddit.com/r/ClaudeAI/comments/1l1qn5z/my_first_project_using_claude_code_it_is_just/)): **A web developer used Claude (Opus and Sonnet models) extensively as a coding assistant to build a complex browser-based music/productivity app in under a week, citing a doubling of development speed. The project leveraged the Web Audio API (user's first time), custom file structure enforcement, and automated progress reports written by Claude, with Opus significantly outperforming Sonnet in understanding instructions, one-shotting complex tasks, and maintaining proper UI design—though persistent issues included div hierarchy misplacement, inconsistent date formatting, and model-specific deficiencies (Sonnet struggled with following instructions and UI details).** Top technical comment agrees with issues around scene/component hierarchy, noting similar struggles with Godot: Claude can edit scene files but fails to infer optimal object relationships, especially when handling complex hierarchies, though targeted prompting mitigates this. Other comments request specifics on effective UI prompting.
    - A user describes that Claude Code struggles with resolving complex UI component hierarchies, particularly div and scroll area components, noting manual intervention is usually required to correct misplaced elements. They also compare this to experiences using Claude with Godot, emphasizing that while Claude can manipulate scene files, it fails to reliably infer parent-child relationships within the scene tree, resulting in poorly organized outputs unless efforts are scoped to a single scene script or element.
    - Another technical observation points out a logic flaw where the application allows multiple songs to play simultaneously in parallel, indicating incomplete or missing state management or playback control logic in the current implementation.
- [**TIFU by letting my 4 year old son talk to ChatGPT**](https://www.reddit.com/r/ChatGPT/comments/1l18zsr/tifu_by_letting_my_4_year_old_son_talk_to_chatgpt/) ([Score: 26474, Comments: 2564](https://www.reddit.com/r/ChatGPT/comments/1l18zsr/tifu_by_letting_my_4_year_old_son_talk_to_chatgpt/)): **The post demonstrates ChatGPT's ability to sustain lengthy, open-ended, and contextually relevant dialogue with a preschool user across an extended session (10,000+ word transcript, ~2 hours). Notably, the interaction highlights ChatGPT's lack of user identity differentiation within a session, merging inputs from both adult and child into a continuous dialogue state. This example underscores both the conversational persistence and context-tracking of the model, as well as challenges in multi-user or family-shared device scenarios regarding persona management.** A top comment notes ChatGPT's inability to distinguish between users in a single session, resulting in a blended user profile/history, which may affect personalization and recommendations. The scenario also spurs discussion on longer-term implications of large language models as companions or entertainment for children.
    - One commenter highlights a technical limitation of language models, noting that ChatGPT *cannot distinguish between different users within the same session*—if a child starts interacting with the model, the user profile and conversation context effectively treat the child as the primary speaker. This can impact conversational continuity and personalization, making session management and context switching an important consideration for future implementations.
- [**128K is DEAD for o4-mini, o4-mini-high, and o1 pro (Pro plan)**](https://www.reddit.com/r/OpenAI/comments/1l19zfa/128k_is_dead_for_o4mini_o4minihigh_and_o1_pro_pro/) ([Score: 103, Comments: 38](https://www.reddit.com/r/OpenAI/comments/1l19zfa/128k_is_dead_for_o4mini_o4minihigh_and_o1_pro_pro/)): **OpenAI has discontinued 128K context support for models o4-mini, o4-mini-high, and o1 pro on the Pro plan, leaving only GPT-4.1 and 4.1-mini (with limited reasoning ability) as options for large context windows. Codex Cloud's "Ask Question" no longer uses RAG (Retrieval Augmented Generation) but performs keyword-based local search, feeding the output to a modified o3 model, restricting effective large-context analysis for advanced workflows.** Commenters highlight a trend across major AI providers (OpenAI, xAI, Google) of marketing capabilities and context sizes not matched by actual product delivery, with significant frustration about cost vs. value. Some recommend alternatives like Google AI Studio and Deepseek R1 for high-context, lower-cost options, and suggest that context window restrictions may limit democratized AI access, concentrating advanced capability in enterprise offerings.
    - Users report that OpenAI silently swapped the o1 Pro Mode for o3 late last week, resulting in a significant reduction in maximum message length—about 1/4th of the prior o1 pro capacity. This swap occurred both in browser and standalone app versions, with the current "o1 pro mode" explicitly identifying itself as o3. The abrupt and non-communicated change is described as potentially a violation of EU Directive 2019/770 and possibly relevant anti-fraud laws in the U.S., as it constitutes an unconsented downgrade of a paid digital product or service.
    - Alternatives like Google AI Studio offer very high context limits for free, compared to OpenAI's paid plans that are reported to have restrictive rate limits. Deepseek R1 is highlighted as recently updated and able to handle 128k context efficiently, indicating that other providers are pushing context length advancements more aggressively than OpenAI.
    - Technical discussion suggests that the original benefit of o1 pro was likely due to it being a rebranded o3, and the removal might be due to high operating costs. There is speculation about future models, such as an enterprise-focused o3 pro with persistent memory, targeting sectors like banking and healthcare, signaling a potential move away from wide public access toward high-cost, enterprise-grade offerings.

### 3. Concerns And Evidence About ChatGPT Data Privacy And Persistence

- [**Deleting your ChatGPT chat history doesn't actually delete your chat history - they're lying to you.**](https://www.reddit.com/r/singularity/comments/1l1jg0o/deleting_your_chatgpt_chat_history_doesnt/) ([Score: 364, Comments: 77](https://www.reddit.com/r/singularity/comments/1l1jg0o/deleting_your_chatgpt_chat_history_doesnt/)): **A user claims that deleting their ChatGPT chat history does not actually purge all historical conversational data, as evidenced when the model can reference or generate information from supposedly deleted conversations, even weeks after deletion. OpenAI's official documentation states that chats are removed from view immediately and scheduled for permanent deletion within 30 days, barring certain exceptions (e.g., security, legal retention, or prior de-identification) ([source](https://help.openai.com/en/articles/8809935-how-to-delete-and-archive-chats-in-chatgpt)). One commenter technically speculates that persistent *embeddings* generated during conversation may be retained for long-term memory, with little public knowledge about the frequency or process for clearing them.** Commenters express skepticism regarding corporate deletion claims, and highlight the opacity of backend retention mechanisms (e.g., usage of embeddings) as a technical concern for data privacy and explainability.
    - One user notes that OpenAI's official policy states deleted chats are only immediately removed from view and *scheduled* for permanent deletion within 30 days, subject to exceptions for security/legal retention or if data is already de-identified ([OpenAI support article](https://help.openai.com/en/articles/8809935-how-to-delete-and-archive-chats-in-chatgpt)). This highlights ambiguity in the definition of 'deletion,' and possible retention periods.
    - A technical concern is raised regarding the use of embeddings for long-term memory in the backend. The commenter suggests there is little transparency about whether or how embeddings generated from user data are cleared when chat histories are deleted, highlighting a gap in data lifecycle management and privacy understanding for LLM-enabled systems.
    - Another user shares an observation that after consecutive wipes, ChatGPT's performance in recalling personal info degrades and begins 'hallucinating,' which is consistent with either actual data removal or at least severing the association between user data and conversation context, affecting model accuracy.
- [**Deleting your ChatGPT chat history doesn't actually delete your chat history - they're lying to you.**](https://www.reddit.com/r/ChatGPT/comments/1l1jgh8/deleting_your_chatgpt_chat_history_doesnt/) ([Score: 2540, Comments: 386](https://www.reddit.com/r/ChatGPT/comments/1l1jgh8/deleting_your_chatgpt_chat_history_doesnt/)): **The poster claims that deleting chat history in ChatGPT does not actually erase conversations from backend memory—when prompted, the model can sometimes recall details from supposedly deleted sessions, even after data-sharing is disabled and memory purged. They assert this is not due to local cache, but a persistence of conversational context accessible to the LLM, suggesting a backend retention issue that may contradict OpenAI's privacy assurances. Technical evidence is anecdotal (chat behavior under targeted prompts) and not presented with rigor or API-level investigation.** Commenters raise concerns about legal implications, data privacy, and trust in OpenAI—one suggests gathering evidence for class-action consideration, while another notes persistence of model recall even after switching to anonymous usage, pointing to possible backend data-linking beyond visible user history.
    - Multiple users report that even after deleting ChatGPT conversation history or accounts, the system still recalls personal details or preferences discussed in prior sessions, suggesting that deletion commands may not actually remove backend-stored data. This raises technical concerns about how user data is retained, indexed, or anonymized, even for 'anonymous' or deleted accounts.
    - A specific user claims to have tested the behavior systematically, showing that deleted conversations are still accessible to the model after a year. This raises questions about OpenAI's data lifecycle, potential shadow retention of chat logs, and possible frontend/backend discrepancies in data management.
    - There is a broader technical skepticism discussed about any online service's ability to truly delete user data—reflecting on trust issues related to data deletion, backend storage mechanisms, and transparency in user data handling policies.
- [**What makes you think AI will continue rapidly progressing rather than plateauing like many products?**](https://www.reddit.com/r/singularity/comments/1l1o0w2/what_makes_you_think_ai_will_continue_rapidly/) ([Score: 170, Comments: 255](https://www.reddit.com/r/singularity/comments/1l1o0w2/what_makes_you_think_ai_will_continue_rapidly/)): **The OP questions the assumption that AI progress will continue at a breakneck pace, drawing analogies to technologies like smartphones, motion controls, and VR which showed early promise but plateaued. They specifically raise the possibility that current LLM limitations (hallucinations, context window size, rate limits) might persist for decades, as industry-wide rapid advances stall out, much like in gaming hardware and interfaces.** Top commenters argue AI is different primarily due to its *potential for self-improvement* (recursive enhancement), *unprecedented global investment*, and the wide competitive interest—from corporations to governments—unlike niche consumer tech. There is a caution that no outcome is certain, but the unique nature of AI as both a tool and a product justifies optimism for continued rapid progress.
    - Several comments highlight the self-improving nature of AI: as models become more capable, they enable further advancements by assisting in tasks such as research, code generation, and data processing, thus accelerating their own development cycle. This differs from physical product cycles (e.g., motion controllers), where innovation is bounded by hardware constraints and market demand, not recursive capability improvement.
    - Unlike most previous technologies, AI has garnered unprecedented global attention and investment, both from states and competitive companies aiming for market or strategic leadership. This competition incentivizes continuous rapid innovation and deployment, similar to but even larger in scope than technology races in other industries.
    - Some users analogize AI's transformative potential to the industrial revolution, noting its broad societal impact and the possibility of rendering certain labor obsolete at scale, even if current efficiency improvements are only partially realized, which further incentivizes relentless progress.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. Breakthroughs in Language Models and Performance Optimization**

- **AI Model Mania: New Contenders Shake Up the LLM Leaderboard!**Engineers welcome a wave of new models including **Gemini 2.5 Pro** flexing impressive long context handling and video understanding, and **EleutherAI's Comma 0.1**, a **7B** model trained on their new **8TB Common-Pile dataset** using a **Llama 3** architecture, available on [HuggingFace](https://huggingface.co/common-pile/comma-v0.1). Speculation abounds for **O3 Pro/GPT-5** and the ambitious **DeepThink** model, potentially boasting a **2M token context window**.
- **LLMs Learn Smarter, Not Harder, With New Tricks!System Prompt Learning (SPL)**, an open-source plugin for `optillm` inspired by [Andrej Karpathy's idea on X](https://x.com/karpathy/status/1921368644069765486) and detailed in a [HuggingFace blog post](https://huggingface.co/blog/codelion/system-prompt-learning), boosts LLM performance on benchmarks like **Arena Hard (+8.6%)** by enabling them to learn problem-solving strategies from experience. Meanwhile, **Prompt Lookup Decoding**, detailed in [this GitHub repo](https://github.com/apoorvumang/prompt-lookup-decoding), offers **2x-4x speedups** on input-grounded tasks by replacing draft models with simple string matching.
- **Training Gets Lean and Mean with FP8 & Optimized AdamW!**Researchers successfully scaled **FP8 training to trillion-token LLMs** by introducing **Smooth-SwiGLU** to combat instabilities linked to SwiGLU activation, as detailed in [the "Scaling FP8 training" paper](https://arxiv.org/abs/2409.12517). Concurrently, another [study on AdamW](https://arxiv.org/abs/2505.21829) reveals the **AdamW optimizer** performs optimally when its beta1 and beta2 parameters are equal, ideally **0.95**, challenging current **PyTorch defaults**.

**Theme 2. The Agentic Frontier: Building Smarter, More Reliable AI Assistants**

- **MCP Mania: The New Lingua Franca for AI Agents?!**The **Model Context Protocol (MCP)** is rapidly becoming a key standard for AI agent communication, with developers successfully connecting **MCP servers to Claude desktop** using `stdio` transport and exploring dynamic tool registration ([schema available on GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/fb34d1d6da2287f82cfdf46c1f91b6fb262cdd38/schema/2025-03-26/schema.json)). The **Gradio Agents x MCP Hackathon**, boasting **$16.5k in prizes**, further fuels its adoption across platforms like LlamaIndex and HuggingFace.
- **Agents Get a Reliability Upgrade, DSPy Lays Foundation!**As agent swarms consume millions of tokens (one developer reported **12 million O3 tokens daily**!), the community tackles reliability with new task-agnostic evaluation frameworks for **detecting, correcting, and preventing failures** in long-running agentic tasks. **DSPy** is emerging as a strong contender for building sophisticated agent frameworks, with its upcoming **3.0 release in June** and discussions on leveraging it for first-class environments and online learning, as seen in projects like [Agenspy on GitHub](https://github.com/SuperagenticAI/Agenspy).
- **Agents Roll Up Sleeves: From Android Control to Minecraft Mastery!**AI agents are proving their mettle in diverse applications: one project integrates **Aura with AppAgentX** via MCP to enable voice control over Android devices ([preview code on GitHub](https://github.com/IhateCreatingUserNames2/Aura_AppAgentX/tree/main)). In the gaming world, the **Mindcraft framework** ([GitHub](https://github.com/kolbytn/mindcraft)) empowers LLMs like the **Andy-4 model** to demonstrate advanced reasoning and planning in Minecraft.

**Theme 3. Hardware Hustle and Kernel Kung Fu: Pushing Computational Boundaries**

- **Tinygrad's AI Smashes PyTorch Kernels With Generated CUDA-C!**The **tinygrad** project showcased impressive results where **AI-generated CUDA-C kernels** outperformed expert-optimized production kernels in **PyTorch** across several benchmarks, achieving **101.3%** relative performance in **Matmul (FP32)** and a stunning **179.9%** in **Conv2D**. These gains, detailed in [PR #10586 on GitHub](https://github.com/tinygrad/tinygrad/pull/10586), were achieved without relying on libraries like CUTLASS or Triton.
- **Mac M3 Flexes Memory Muscles, AMD GPU Rumors Sizzle!**Engineers lauded the **Apple M3 Mac's** performance for large models, attributing it to its substantial memory bandwidth (the **M3 Max** hits up to **540GB/s** with **LPDDR5X**) and its **18 TOPS** neural engine. Meanwhile, a [Tweaktown article about the Radeon RX 9080 XT](https://www.tweaktown.com/news/105554/amd-rumored-radeon-rx-9080-xt-up-to-32gb-of-faster-gddr7-4ghz-gpu-clocks-450w-power/index.html) fueled excitement with rumors of an **AMD Radeon RX 9080 XT** packing up to **32GB of GDDR7** memory, though some questioned the leak's source (MLID).
- **CUDA & Triton Tweaks Squeeze Out Max Performance!**Discussions in the GPU MODE Discord highlighted the importance of CUDA conventions, such as using **'x' for the memory-continuous dimension** to ensure coalesced memory access, as explained in a [CUDA MMM blog post](https://siboehm.com/articles/22/CUDA-MMM). Engineers are also refining **Triton kernels**, exploring techniques like calling one `@triton.jit` function within another and optimizing `num_warps` settings for tasks like a [custom grayscale conversion kernel](https://github.com/username/repo/blob/main/grayscale.py) (Note: specific link is a placeholder from source).

**Theme 4. Revolutionizing Developer Toolkits and Integration Ecosystems**

- **Mojo Madness: Modular Ignites Development with Hackathons & Bindings!Modular** is stoking the fires of **Mojo** development with a [hackathon weekend](https://lu.ma/modular-hack-weekend) targeting **Mojo kernels**, **MAX Graph model architectures**, and **PyTorch custom ops**, complemented by a **GPU programming workshop**. Additionally, a community member is crafting a **C-to-Mojo bindings generator** to further expand Mojo's interoperability.
- **Dev Tools Level Up: Cursor Gets Sleeker, OpenRouter API Gets Clearer!**The **Cursor IDE** received a significant update, refreshing its **UI** and settings panel for better organization and snappier performance. Over at OpenRouter, developers clarified that the `sk-or-v1` **key** is the sole key for its **REST API**, while the feature for submitting [end-user IDs for abuse prevention, detailed in API docs,](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user) is still considered experimental due to a *lack of available metrics*.
- **Niche Tools Shine: LLM Scribe Aids Fine-Tuning, NotebookLM Users Seek Fixes!**The **LLM Scribe** tool, demonstrated on [Hugging Face Spaces](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) and [YouTube](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s), is gaining attention for simplifying the creation of handwritten datasets for fine-tuning, supporting formats like **chatml** and **alpaca**. Meanwhile, **NotebookLM** users are reporting limitations with generating **audio overviews in languages other than English** and encountering endless spinning animations during **MP4 video uploads**.

**Theme 5. Fueling the Future: Innovations in Datasets and Training Data**

- **EleutherAI Unleashes 8TB "Common-Pile" Dataset & New 7B Model!EleutherAI** made waves by releasing **Common-Pile**, an **8TB libre dataset**, along with a filtered version and **Comma 0.1**, a **7B** base model available on [HuggingFace](https://huggingface.co/common-pile/comma-v0.1). This new model, matching **Llama 3's** architecture, was trained on the filtered dataset using **lingua** on **64 Nvidia H100 GPUs**, exciting the community as *the closest to a full stack FOSS model* yet.
- **Targeted Data is King: DPO Datasets & RAG Gain Traction!**The Cohere community shared a [HuggingFace dataset for Cohere Spanish Recipes](https://huggingface.co/datasets/somosnlp-hackathon-2025/gastronomia-hispana-dpo) utilizing **Direct Preference Optimization (DPO)**. Simultaneously, **Retrieval-Augmented Generation (RAG)** strategies are being explored to enhance AI with specific knowledge, such as using **LocalDocs with scientific textbooks** for GPT4All and connecting **MCP servers to MCP knowledge stores** for RAG finetuning.
- **Handcrafting Data Gets Easier: LLM Scribe Tool Debuts!**Creating high-quality, handwritten datasets for fine-tuning is getting a boost with the introduction of **LLM Scribe**, a tool supporting export to **chatml, alpaca, and sharegpt** formats. Showcased on [Hugging Face Spaces (LLM Scribe Demo)](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) and available for [full version purchase on Gumroad](https://kryptive.gumroad.com/l/gvyqep), it offers features like autosaving, multi-turn creation, and token counters.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Unlimited Access Actually Limited**: Members discussed that *unlimited access* to services like [OpenAI's ChatGPT Pro](https://help.openai.com/en/articles/9793128-what-is-chatgpt-pro) is often limited by abuse guardrails.
   - One user stated these are *tricks* that people keep falling for, stating that *when the time comes, you'll see that it's not unlimited*.
- **Surfshark and Adguard VPNs Compared**: Members compared **Surfshark** and **Adguard VPNs**, noting that **Surfshark** has been shared without issues, while **Adguard's** advertising blocker can be annoying.
   - A member complained about the lack of transparency in **Perplexity's** language and limits regarding codes from Xfinity for **Perplexity Pro**.
- **Perplexity Pro has Secret Rate Limits**: A member reported getting **500 O3 uses a day** without issues, sparking a discussion about rate limits for **Pro** members and abuse guardrails.
   - Another user tested **O1 Pro** and was rate-limited at **400 queries**, suggesting the system-claimed limit of **600** may be inaccurate depending on the reasoning model used.
- **Sonar API Recreates Manus for Free**: A user found a Reddit post about recreating **Manus** using the **Sonar API**, with the post titled [*I built my own Manus.im using Perplexity Sonar API and Claude, and it works just as well, for almost free*](https://www.reddit.com/r/perplexity_ai/comments/1j83gat/i_built_my_own_manusim_using_perplexity_sonar_api/).
   - Another user replied that **Manus** does not allow configuring an external LLM via API, and that **Perplexity** can export its answer as a **PDF**, **Markdown** or **DOCX**.
- **Danube Water Levels Monitored Live**: A member shared a live dashboard ([link](https://www.perplexity.ai/page/dynamisches-donau-pegelstande-WHBFCZY.QdaFCrDq3d4qFA)) showing the water levels of the German Danube between **Passau** and **Budapest**.
   - The dashboard also provides the real-time status of ship locks along the way, though the text is in German.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro handles long context, sees Video!**: Members lauded **Gemini 2.5 Pro** for its superior **long context handling**, exceeding **200k tokens** regularly and seamlessly integrating translations.
   - A user stated that **Gemini 2.5 Pro** is the only AI that can *see* video, and combined with **O3**, it is an overall pleasant combo.
- **Agent Swarms Bankrupt Users!**: A developer reported that their **agent swarms** are consuming approximately **12 million O3 tokens per day**.
   - The developer sympathized with the AI agents, quipping that it *must be tough to be stuck as a web search agent AI*.
- **Recursive AI interactions: Breakthrough or Breakdown?**: A user claimed their recursive interactions with **GPT** led to *92% sustained contradiction stabilization*, while another member warned against the dangers of such systems, cautioning against thinking they've *singularly discovered a new wormhole*.
   - The user insisting on their emergent stability effect claimed it was *not a feature programmed into the model* but an *emergent stability* effect from specific interaction patterns and identity anchoring.
- **Cursor wins Coding Tool Showdown!**: Users compared **Cursor** and **Windsurf** as coding tools, with praise for **Cursor's agent mode** and code review features.
   - It was also noted that the **user experience** of **Cursor** is better overall, preventing code messing.
- **ChatGPT struggles with Japanese**: Members discussed the performance of AI models in **Japanese translation**, observing that **CNNs** like **DeepL** perform worse compared to **transformers** like **ChatGPT** and **Google Translate**.
   - However, one member noted that Google's Gemini also excels in Japanese.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GRPO Boosts TTS Stability**: A member is using **GRPO** for **TTS**, enhancing stability with **WER** and **voice similarity rewards** using [this repo](https://github.com/wenet-e2e/wespeaker) and may add an *audiobox-aesthetics reward*.
   - Initial tests suggest a **10x improvement** in sound quality.
- **Hyperbolic's Wild H100 Hour Prices**: **Hyperbolic** offers **H100s** at **$3.95/hour**, with bonus credits available through [referral link](https://app.hyperbolic.xyz/invite/7aALdedCm).
   - Since **Hyperbolic** lacks notebooks, users connect via **Google Colab**, speculated to be due to a partnership with **Xai**.
- **BOS Tokens Break Fine Tuning**: Members discussed the impact of **double BOS tokens** in instruction data using **Mistral chat templates**, noting that `map_eos_token = True` isn't working as expected.
   - It was mentioned that **Unsloth** handles this under the hood, though omitting **BOS** nukes the model; Daniel's notes say that fine-tuning would be broken.
- **LLMs Pick Up Brains with System Prompt Learning**: **System Prompt Learning** allows LLMs to learn problem-solving strategies from experience, improving **Arena Hard** by +8.6% and **AIME24** by +6.67%.
   - Inspired by [Karpathy's idea](https://x.com/karpathy/status/1921368644069765486), this method builds a strategy database and applies it to new problems.
- **Dataset Tool Suggestion to generate Template Feature**: A member proposed adding a *generate template* feature to the LLM Scribe dataset tool.
   - The feature would generate a full dataset with a small model like **Llama** or **Gemini Flash** for manual editing.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude 4 Opus Plagued by API Problems**: **Claude 4 Opus** experienced API issues affecting model access, but some members noted that the issues don't affect use as long as the page isn't reloaded.
   - Members suggested the issues might be widespread, and the team is investigating the root cause.
- **O3 Pro: Stealth Release or Hype?**: Members speculated on a potential **GPT-5** release coinciding with the shutdown of **GPT-4.5** on the API in July, while others claimed to have access to **O3 Pro**.
   - These claims were met with skepticism, with some suggesting any perceived improvements might be attributable to ongoing tweaks rather than a full release.
- **DeepThink Aims to Outpace O3 Pro with 2M Context Window**: A user suggested **DeepThink** could boast a **2M context window**, potentially eclipsing **O3 Pro**, prompting discussion about feasibility given compute constraints.
   - The discussion included whether **DeepThink** would be a substantial advancement or just a longer-thinking version of **Gemini 2.5 Pro**.
- **Gemini 2.5 Pro Battles GPT-4.5 in Hallucination Arena**: Members debated hallucination rates, with one claiming **GPT 4.5** appears to have fewer hallucinations because *it doesn't assert much*, while another humorously called *Claude 4 Opus a hallucination goblin*.
   - Others attributed any differences to **GPT-4.5's** larger size, suggesting that model scale might be a factor in reducing hallucination frequency.
- **Navigating Commercial Use of AI-Generated Images**: A user inquired about the commercial viability of images generated from LM Arena, and another member replied that it *depends on the model*, but the results are open source, a user would have to prove it's their model if there's a dispute.
   - Another member said they use flux to generate product photos.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **ModernBert Misclassified, Community Takes Notice**: Users found the `nomicai-modernbert-embed-base-8bit` model was miscategorized as an LLM in LM Studio, suggesting the use of the [latest beta](https://lmstudio.ai/latestbeta) to rectify this.
   - The latest beta allows users to right-click (or use the gear symbol) on the model view and change the model type, though the option may not appear for all models.
- **LM Studio's LiteLLM linkup Needs OpenAI Patch**: A user inquired about integrating LM Studio with LiteLLM, noting LM Studio's absence as a provider, but linked [this starting point](https://docs.litellm.ai/docs/providers/lm_studio).
   - It was suggested to use the OpenAI-like provider setting in LiteLLM to connect with LM Studio, though success wasn't guaranteed, but may be worth it considering the *integration pain*.
- **Radeon RX 9080 XT Rumors Spark Speculation**: A [Tweaktown article](https://www.tweaktown.com/news/105554/amd-rumored-radeon-rx-9080-xt-up-to-32gb-of-faster-gddr7-4ghz-gpu-clocks-450w-power/index.html) reported on rumors of an **AMD Radeon RX 9080 XT** with up to **32GB** of faster **GDDR7** memory, inciting debate on its validity.
   - Some members doubted the leak due to the unconfirmed source (MLID), claiming that it is basically doubled and mirrored Navi44
- **Deepseek R1 Distill Llama Makes Dinner**: A member made dinner using a recipe generated by **Deepseek R1 Distill Llama 70B Q5_K_M**, showing the practical applications of large language models.
   - The meal was enjoyed and generated using a recipe by Deepseek R1 Distill Llama 70B Q5_K_M.
- **Beware: Cat Scratches Can Ruin New OLED Monitor**: After a member mentioned cats scratching laptop screens, another member recounted getting a new **OLED monitor** and promptly *started sweating* at the thought of it being damaged by their cat.
   - They were referred to [this reddit thread of a cat scratching a 2k laptop](https://www.reddit.com/r/mildlyinfuriating/comments/1bh65ud/cats_scratched_2k_laptop_screen/).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Polishes Interface and Performance**: The newest **Cursor update** delivers a revamped **UI** and settings panel, alongside general performance improvements.
   - Users are finding the new layout more organized and the overall experience snappier.
- **Claude 4 More Opinionated About Docs**: Users are noticing that **Cursor Claude 4** seems more keen on writing project documentation automatically, which might not always be desirable.
   - One user cheekily suggested a user rule: *Do not generate or modify project documentation unless explicitly requested*.
- **Student Discount Drama for O3 Pro**: Several members are struggling to apply their student discount for **O3 Pro**, despite confirming their student status via SheerID and getting confirmation emails.
   - Staff are stepping in to sort out the snags, with additional errors being reported.
- **Background Agent Keeps Secrets Secret**: Adding secrets like `TEST=test` to the background agent results in **encrypted** environment variables (e.g., `TEST=lNInUMxS211/c3up+u7VwjGCIweB/qXlWGnBEfaHzHU=`).
   - This encryption adds a layer of security, ensuring sensitive data remains protected.
- **Devcontainers Mostly Cooperating with Agent**: **Devcontainers** are generally functional with the background agent, though some quirks persist.
   - Notably, **MCP servers** refuse to run within the devcontainer, which members are finding to be *extremely annoying*, in addition to a docker-compose selection problem.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Agent reliability with detection/correction**: Members are working on an evaluation framework to ensure **reliability and replicability** in long-running agentic tasks, automatically trying to **detect, correct, and prevent failure cases**.
   - The new framework is being built to be task-agnostic.
- **New Directory for AI Demos Opens**: A member launched [aitry.fun](https://aitry.fun/), a new **AI demo directory** that allows for fast access to AI providers saving a lot of time.
   - Feedback is welcomed by the directory author.
- **System Prompt Learning (SPL) Improves LLM Performance**: A member introduced **System Prompt Learning (SPL)**, an open-source plugin in optillm that teaches LLMs to learn problem-solving strategies from experience with a [+8.6% improvement on Arena Hard](https://huggingface.co/blog/codelion/system-prompt-learning).
   - SPL works with any **OpenAI-compatible API** by adding `spl-` prefix to the model name; LLMs improve over time at frequently used problem types, plus all strategies are human-readable.
- **Vision-Language Models Deep Dive**: A member shared a video explaining Vision-Language Models (VLMs), the foundation of multimodal AI, and suggests exploring **nanoVLM** by HuggingFace for hands-on experience, linked from [github](https://lnkd.in/gaCAzmAe).
   - The video ([lnkd.in/gpFM-tdU](https://lnkd.in/gpFM-tdU)) covers **VLM pipeline overview**, **LLaMA internals**, **ViT**, and **Modality projection** in a simple and visual way.
- **Agent course deadline confirmed**: Several members confirmed that the deadline for the agent course final project is **July 1st**.
   - One member asked for clarification on whether this includes **July 1st** as a valid submission date, but it seems like it is.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Engineer Seeks Big Tech Salvation**: A recently laid-off software engineer with **4 YOE** from startups is seeking advice on how to break into big tech companies and is looking for feedback on their self-study plan to become competitive for big tech.
   - They have been invited to DM for further discussion.
- **Kernel Conundrums: Triton's Tale of Transformation**: A member seeks advice on improving their [custom Triton kernel for grayscale conversion](https://github.com/username/repo/blob/main/grayscale.py), focusing on performance and code structure and is particularly curious about why setting `num_warps=4` resulted in a worse result than using the `calculate_settings` function that they linked.
   - Another member pointed out that you can call another `@triton.jit` function inside a `@triton.jit` function.
- **CUDA's Coordinates Cause Confusion**: A user questioned the convention of using **x for rows and y for columns** in CUDA, citing a [blog post](https://siboehm.com/articles/22/CUDA-MMM) where the author seemed to do the opposite.
   - Another user clarified that **x** is typically used for the dimension that is **continuous in memory** to ensure coalesced memory access with warps.
- **SPL Teaches LLMs to Level Up**: A new approach called **System Prompt Learning (SPL)** teaches LLMs to learn problem-solving strategies from experience, similar to how humans learn.
   - The method, implemented as an [open-source plugin in *optillm*](https://github.com/codelion/optillm/tree/main/optillm/plugins/spl), allows LLMs to build a database of effective strategies and apply them to new problems, resulting in performance boosts such as **+8.6%** on Arena Hard and **+6.67%** on AIME24.
- **Factorio Finds Focus with Features**: Each fresh **Docker container** running a **Factorio server** needs to be activated by logging into it once to create a player inside the game, which can then be taken over.
   - A member mentioned the possibility of integrating [external memory systems](https://github.com/mem0ai/mem0), particularly using **RAG (Retrieval-Augmented Generation)**, to enhance the Factorio learning environment and directed another member to review [PR #158](https://github.com/JackHopkins/factorio-learning-environment/pull/158) for the Factorio learning environment.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **REST API Key Confusion Dissipates**: Members clarified that the **sk-or-v1 key** is the *only* correct key for the **REST API**, advising users to ensure its proper implementation.
   - A user encountering errors with n8n was directed to [a helpful integration guide](https://jannicknijholt.nl/integrate-openrouter-in-n8n/).
- **End-User IDs: Still in the Lab**: Members discussed the option to submit **end-user IDs** to prevent abuse, a feature available [in the API documentation](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user).
   - Concerns were raised over the *lack of available metrics* for this feature, implying it's not yet fully ready for production use.
- **DeepSeek Provider Power Struggle**: The best provider for **DeepSeek** sparked debate, with preferences split between **Parasail** (due to trust and consistent performance at **$5**) and **DeepSeek** itself (for its lower cost, caching, and direct model implementation).
   - Some users cited concerns with **DeepSeek's** official API, noting crowded servers, Chinese location, and slow speeds, as well as `max_tokens` issues and an **8K max** on non-reasoning output tokens (though **R1** increased this to 64k).
- **Chess Training Improves Problem-Solving**: Members discussed chess benchmarks and the surprising performance of `gpt-3.5-turbo-instruct` after training on chess data, citing research showing that **chess training improves problem-solving** ([https://arxiv.org/pdf/2312.09390#page=29](https://arxiv.org/pdf/2312.09390#page=29)).
   - It was also pointed out that **RLHF can degrade performance**, with *gpt4-base* (pre-RLHF) outperforming *4o* in chess ([https://dynomight.net/more-chess/](https://dynomight.net/more-chess/)).
- **LLM Scribe Simplifies Fine-Tuning Dataset**: A tool was introduced for creating handwritten datasets for fine-tuning, supporting formats like **chatml**, **alpaca**, and **sharegpt**, along with features like autosaving and token counters.
   - Demonstrations of **LLM Scribe** are available on [Hugging Face Space](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) and [YouTube](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s), with the full version accessible [here](https://kryptive.gumroad.com/l/gvyqep).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus AI Student Perks Explained**: Users discussed the benefits of **Manus AI student perks**, confirming it allows usage of a **credit-free environment**, up to **50 knowledge entries**, and access to **high-effort mode**.
   - One user inquired about changing their email address and sending referrals without losing the perks.
- **OpenManus Clarified as Separate Entity**: Users inquired whether **OpenManus** is affiliated with **Manus AI** because their website caused confusion.
   - Other members clarified that **OpenManus is not affiliated** with Manus AI, but is a *free alternative* that doesn't include API pricing.
- **Compromised Bots Spreading Scam Links**: A user reported a suspected **compromised bot** spamming potentially malicious links, including one disguised as a **Manus fellowship link**.
   - Administrators were alerted to remove the harmful content.
- **Manus Deployments Require Hack to Remove Icon**: A user inquired about removing the **Manus icon** from deployed websites generated by Manus.
   - Community members responded that removing the icon directly isn't possible, advising the user to **download the files and deploy them on another platform**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Local 70B Model Graphing Experience Shared**: A member ran a **70B model** locally to create a graph, sharing a [video](https://cdn.discordapp.com/attachments/729741769738158194/1378497525702725824/Screencast_from_2025-05-31_23-39-40.webm?ex=683f745d&is=683e22dd&hm=b87717b28e91071d704dc570fe912054b96cc4b7fec4e5706278f4e0a0df4d3e) of the process, but finding it slow.
   - Another member expressed interest in making their own model but lacked storage for training data, remarking *"Kinda wanna make my own model but I don't even have the storage for the training data."
- **Reasoning Steps Turbocharge Agent Training**: Training an agent to rule out wrong answers quickly can be achieved by incentivizing the agent to output reasoning steps during CoT which decrease the likelihood of incorrect answers, as suggested in [this paper](https://arxiv.org/abs/2505.05755) for text-diffusion.
   - This incentivization speeds up the training process by focusing the agent on relevant information early on.
- **HF Chunked Prefill Accelerates Long Context Benchmarks**: **Hugging Face** now supports **chunked prefill**, which is useful for running long context benchmarks with **lm_eval**; however, setting `prefill_chunk_size` in `generation_config.json` doesn't work, but works when put into the call to `self.model.generate` in `lm_eval/models/huggingface.py`.
   - It prevents OOM errors when running ruler and is tough to run long context benchmarks without it if you're actually using a long context, a real *game changer*.
- **Neural Networks Reside on Low Dimensional Manifolds**: A member posited that neural networks trained on a low dimensional manifold of natural inputs automatically correspond to some low-dimensional manifold of activations embedded in the high-dimensional space of possible forward passes.
   - They further suggested that knowing the [data generating process](https://en.wikipedia.org/wiki/Data_generation) and sampling the activations for different inputs allows building a picture of what those manifolds look like.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **M3 Mac Impresses with Memory Bandwidth**: Members attributed the decent performance of **Mac M3** chips for large models to the massive memory bandwidth of **Macs**, particularly the **LPDDR5X** memory.
   - The **Mac M3 Max** version reaches up to **540GB/s**, while the built-in neural engine assists with **18TOPS** (trillions of operations per second).
- **Debate Rages on AiderBench Dependencies**: Members debated dependencies to make **AiderBench** work without **Docker**, with one suggesting it's easier to use **Docker** due to heavy dependencies.
   - Others suggested using a **VPS** as an alternative, while another member recommended **Agent0** as a superior alternative to **DeepSeek** agent.
- **Aider Automatically Summarizes Conversations**: **Aider** automatically summarizes conversation history, aiding in context management.
   - To provide **git commit** and **git diff** views to Aider, use the `/run git diff ...` command, which will then prompt you to add it to the chat.
- **Read-Only Access Enables Multi-Repo Usage in Aider**: A member suggested using the `/read-only` command in **aider** to access files from multiple repositories, as symlinks are not followed.
   - For example, `/read-only /Users/username/Downloads/some_random_file.md` allows read-only access to files outside the current repository.
- **Devstral Gains Attention for Local Model Recommendation**: A user sought advice on the best local model to run, given **4x3090s**, **256GB RAM**, **~100k context window**, and tasks involving edits and fixes on existing **Rust** and **Typescript** code.
   - The user was recommended to try **Devstral**, with a member noting that some versions of the new **R1** would probably deserve a test as well.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Throws Devs Hackathon**: Modular is hosting **another hackathon** open to virtual participation, focused on **Mojo kernels**, **MAX Graph model architectures**, and **PyTorch custom ops**, and is kicking it off with a **GPU programming workshop** at their office in Los Altos which will be streamed online. [Check out the details!](https://lu.ma/modular-hack-weekend)
   - A recent CS graduate reported seeing *improvement* trying **Mojo** on basic ML models, after watching a **Fireship video**.
- **`_type_is_eq` Type Check Limitations**: Members discussed the limitations of using `_type_is_eq` to check types, noting it can't check pointer types with arbitrary pointee types, and contrasted against using [type name reflection](https://example.com/reflection) to check for pointers.
   - The value of a reflection API was highlighted to construct serializers, raising the question of prioritization against improvements to the trait/type system.
- **`Copyable` vs `ExplicitlyCopyable`**: The discussion covered the purpose of a type conforming to both `Copyable` and `ExplicitlyCopyable` traits, with an example given of a **100+ GB ML model** that would be better moved than copied.
   - It was also noted that implementing the `Copyable` trait informs the compiler when it can perform implicit copies.
- **Profilers for Mojo Code**: Members suggested profiling **Mojo** code with tools compatible with **C++** or **Rust**, such as `perf` with `flamegraph`.
   - They also pointed to CPU vendor's HPC profilers like **Intel VTune**, **AMD uProf**, and **ARM Streamline** for detailed microarchitecture insights useful for optimization.
- **C-to-Mojo Bindings Generator Surfaces**: A member is developing a **C-to-Mojo bindings generator**, aiming to handle most cases except *"wildly horrible packed structs"* and potentially `restrict` and pragmas affecting calling conventions.
   - The developer suggested using a `pixi.toml` file to indicate dependencies for a **Mojo** GUI project and raised concerns about copying components instead of borrowing in certain parts of the code.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP-Claude Desktop Connection Achieved**: A member successfully established an **MCP server** with **Claude desktop** using [MCP docs](https://modelcontextprotocol.io/quickstart/server#why-claude-for-desktop-and-not-claude-ai) and **stdio** transport.
   - They suggested this setup for others learning MCP and noted the importance of client support for injecting data into the system prompt.
- **Dynamic Tool Registration Troubleshoot**: A user reported that dynamically registered tools in MCP aren't immediately available, needing a full message cycle to become discoverable.
   - They are seeking solutions to enable immediate tool discovery and invocation after registration, and the [schema.json/.ts](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/fb34d1d6da2287f82cfdf46c1f91b6fb262cdd38/schema/2025-03-26/schema.json) file is the closest thing to a full spec.
- **Aura and AppAgentX Collaborate**: A member integrated **Aura** with **AppAgentX** to manage **Android devices** using an MCP Client, with preview code on [GitHub](https://github.com/IhateCreatingUserNames2/Aura_AppAgentX/tree/main).
   - This setup allows voice control over **Android phones** by wrapping Aura's functions into A2A tools and broadcasting them into an A2A to MCP Gateway (Aira Hub).
- **MCP Servers Get Memory**: An **MCP knowledge store** (client host) can now connect to **MCP servers** for RAG finetuning.
   - More details are available in [a LinkedIn post](https://www.linkedin.com/posts/nerdai_mcp-rag-ai-activity-7335292265386463232--h0Y?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Audio Overviews Still Limited by Language**: Users are experiencing difficulties generating **audio overviews** in languages other than English, even when customizing **AI chat settings**.
   - There is currently *no easy option to request podcasts in different output languages*, and an export function is missing.
- **NotebookLM Chat API Integration Unlikely**: A user inquired about integrating **NotebookLM's chat function** into other tools via **APIs** for business client support.
   - However, it was clarified that *NotebookLM is an end-user tool*, and alternatives like **Google Cloud - Conversational Agents and Dialogflow API** were suggested for broader audience applications.
- **Video Uploads Spinning Out of Control**: Users report that uploading **MP4 sources** results in an endless spinning animation, requiring a page refresh to complete the upload.
   - This highlights a potential UX issue with video uploads.
- **Metadata Embeddings: Content Enrichment Remains Unsolved**: A user inquired about embedding **metadata** into **PDFs** to improve the quality of content loaded into NotebookLM sources.
   - The support and the best approach remains an open question.
- **NotebookLM Hallucinating Facts; Send Bugs**: A user reported that **NotebookLM** creates **random, unsourced facts** and then refers to them as if they are in the original sources.
   - Users were suggested to report this behavior in the **bugs channel**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes-3 Coming to Discord?**: Members are considering integrating **DeepHermes-3** into the Discord server to encourage more organic engagement if it can be autonomous.
   - This was clarified by a member mentioning that **DeepHermes-3** is already active in a specific channel.
- **Hassabis' Bold AGI by 2030 Bet**: **Demis Hassabis** predicted **AGI by 2030** (give or take) in [a recent talk](https://www.youtube.com/watch?v=U3d2OKEibQ4), emphasizing **DeepMind's** commitment to pushing frontiers since **2010**.
   - He highlighted how **DeepMind** has consistently been at the forefront of AI research and development since its inception.
- **Prompt Lookup Decoding Unlocks Big Gains**: **Prompt lookup decoding** replaces the draft model with simple string matching in the prompt, generating candidate token sequences and yielding **2x-4x speedups** in input-grounded tasks, as [shown on Github](https://github.com/apoorvumang/prompt-lookup-decoding).
   - This method works with any decoder model, requiring no model changes or external datastore, and is compatible with both greedy and sampling techniques.
- **Minecraft Gets Mindcraft LLM Framework**: The **Mindcraft framework** ([github.com](https://github.com/kolbytn/mindcraft)) and related **Andy models** ([Ollama](https://ollama.com/Sweaterdog/Andy-4) and [HuggingFace](https://huggingface.co/Sweaterdog/Andy-gen)) have been specifically trained to work with the Java version of Minecraft.
   - Notably, the **Andy-4** model, an **8 billion** parameter model, was trained on a single **RTX 3090** over three weeks and delivers advanced reasoning, multi-step planning, and robust in-game decision-making.
- **LLMs Get Scribe**: A member introduced **LLM Scribe**, a tool for creating handwritten datasets for fine-tuning, with features like multi-format export, autosaving, multi-turn creation, token counters, goal tracking, and custom fields, as seen in [this youtube demo](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s).
   - The tool is available on [Hugging Face Spaces](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) and for [full version purchase](https://kryptive.gumroad.com/l/gvyqep).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **EleutherAI's Common-Pile Shakes the Foundations**: EleutherAI released **Common-Pile**, an **8TB** libre dataset, a filtered version, and **Comma 0.1**, a **7B** base model whose architecture matches **Llama 3** and was trained using **lingua** on **64 Nvidia H100 GPUs**, and available on [HuggingFace](https://huggingface.co/common-pile/comma-v0.1).
   - Community members noted it's *the closest to a full stack FOSS model* seen yet, though its tweets got removed for unknown reasons.
- **Replicate's Kontext Chat Talks to Images**: Replicate introduced **Kontext Chat**, an open-source application for editing images via conversational commands, built with **Hono** and hosted on **Cloudflare Workers**, announced on [X](https://xcancel.com/replicate/status/1929160560295506417?s=46).
   - The application is designed as a developer starting point.
- **NYT Swaps Integrity for AWS Credits**: The New York Times and Amazon signed a deal for licensing NYT's content for AI training, including Amazon's foundation models, [announced on Twitter](https://xcancel.com/natolambert/status/1929175745596620968?s=46).
   - Community members speculated that this move suggests **NYT's lawsuit against OpenAI** was primarily for securing payment rather than a moral stance.
- **Karpathy Drops the Model Menu**: Andrej Karpathy shared his guide to effectively using different ChatGPT versions, recommending **'o3'** for important tasks, **'4o'** as a daily driver, **'4.1'** for coding, and **'Deep Research'** (based on o3) for deep dives, announced on [X](https://xcancel.com/karpathy/status/1929597620969951434).
   - The recommendation received mostly positive reactions.
- **AIE Community Ships Bot to Production**: A member announced that a **live production AI bot** was collaboratively built within the Discord community, with a new gen UI framework shared and then shipped to the AIE website today, found at [ai.engineer/ai](https://ai.engineer/ai).
   - The discussion thread is linked [here](https://discord.com/channels/822583790773862470/1378055295401459813/1378137211995689061).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Regression Testing Underway for Performance Stability**: Members are adding **regression testing** to benchmark **TPS** with different configurations to ensure no performance or compatibility regressions occur.
   - The current **PR** addresses both performance and evaluation metrics.
- **Golden Paths Fine-Tuning LLaMA-3**: A member is developing internal *golden paths* for fine-tuning different models, starting with **LLaMA-3 70B** with initial insights from **8k context** length experiments.
   - Findings will be shared on Data Parallelism (**DP**) vs Tensor Parallelism (**TP**), **FP8** vs **BF16**, and the impact of Compile vs No Compile on performance.
- **FP8 Compile Massively Boosts TPS, Reserves Extra Memory**: Experiments show that **FP8** has the lowest **TPS** with compile disabled, but the highest **TPS** when compile is enabled.
   - It was observed that **FP8 + compile** has the lowest active peak memory but the highest peak reserved memory, suggesting to try `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`.
- **Higher TP Lowers TPS, Needs More Memory**: Increasing tensor parallelism (**TP**) leads to decreased throughput (**TPS**), likely due to the cost of matmul collectives exceeding those for model parameters in pure FSDP, and **FP8** doesn't seem to mitigate this cost.
   - Higher **TP** results in higher peak active memory because expensive layers like outputs to compute loss are replicated, and a member suggested **Loss Parallel** implementation for very large contexts.
- **Universal HF Tokenizer Support On Deck**: Work is underway to land universal [Hugging Face tokenizer](https://github.com/huggingface/transformers) support, with unit tests added by a member, awaiting review.
   - The member is ready to swap priorities if Tool Calling Support is needed ASAP.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Deep Dive on Claude Code Analysis with Cleanroom Design**: Members analyzed [Claude Code](https://southbridge-research.notion.site/claude-code-an-agentic-cleanroom-analysis) using **cleanroom design** principles to avoid accessing proprietary techniques.
   - The approach benefits from avoiding direct access to the source, ensuring the design team remains uncontaminated by any knowledge of the competitor's proprietary methods as outlined [here](https://en.wikipedia.org/wiki/Clean-room_design).
- **DSPy Talks Spark Community Input**: A community member will give DSPy talks at **AI Engineering** and **Databricks DAIS**, actively seeking input on what to cover and use cases to spotlight.
   - The talks will cover **basic DSPy concepts (signatures, programs, common optimizers)**, practical **use cases (structured outputs from PDFs and images)**, and cutting-edge **topics (RL, datasets, advanced optimizers like MiPro)**.
- **DSPy Powers DARPA Project to New Heights**: DSPy was instrumental in **DARPA's Advanced Research Concepts lab**, creating a solution for 'Collaborative Knowledge Curation'.
   - This project is now being spun out into a company, marking a significant achievement.
- **DSPy 3.0 Launch Slated for June**: The community is eagerly anticipating the **3.0 release of DSPy in June**.
   - Discussions include migrating existing pipelines to DSPy, determining its optimal use cases, synthetic data generation, and contrasting it with other agentic solutions.
- **DSPy's Position as Foundation for Agent Frameworks**: There's active discussion around building an **agent framework on top of DSPy**, emphasizing first-class environments, reward handling, and online learning via optimizers.
   - A member highlighted [Agenspy](https://github.com/SuperagenticAI/Agenspy), with another member sketching an implementation atop **claude code**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Hackathon's Prizes and Tracks Unveiled**: The **Gradio Agents x MCP Hackathon** kicked off with [$16.5k in prizes](https://t.co/TBAPFsNWU1) and [$900k in credits](https://t.co/TBAPFsNWU1), featuring **3 Tracks**: *MCP Tool/Server*, *Custom Components for Agents*, and *Agentic Demo Showcase*.
   - A kickoff [livestream](https://t.co/FzLmzviwRz) will guide participants, with office hours in the HuggingFace Discord server.
- **LlamaIndex Scales Agents in Finance**: @jerryjliu0 shared the full slide deck from the **Scaling Agents in Finance workshop** to [automate document workflows](https://t.co/Crfy50pB4j) with Agentic AI for finance tasks.
   - This enables users to use LlamaIndex for tasks in the finance industry.
- **Streamlining Nested Workflow Events**: The recommended approach for streaming all events from nested workflows involves having the **parent workflow iterate over the event stream** of the sub-workflow and propagating events back up.
   - This pattern ensures better composability compared to passing the parent context into the child workflow.
- **Schema Scraper Revolutionizes Web Data Extraction**: A member developed an **AI powered web browser** that can scrape websites for schemas.
   - It extracts specific fields using non-AI actions, creating reusable strategies for data extraction.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AdamW Optimizer Defaults Need Patching**: A [recent paper](https://arxiv.org/abs/2505.21829) suggests **AdamW** performs best when beta1 and beta2 are equal, ideally **0.95**.
   - Current [PyTorch defaults](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) for beta1 (**0.9**) and beta2 (**0.999**) are suboptimal, prompting calls for a patch submission.
- **FP8 Training Requires Smooth SwiGLU**: [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517) details how **FP8 precision** trained large language models on up to **2 trillion tokens**.
   - The paper links instabilities in FP8 training to **SwiGLU activation** and introduces **Smooth-SwiGLU** for stable training.
- **MCP GitHub Attack Vector Post-Mortem**: A member shared a [post-mortem report from Invariant Labs](https://invariantlabs.ai/blog/mcp-github-vulnerability) detailing an attack vector targeting **GitHub MCP** to leak private repo content into public PRs.
   - The discussion emphasized the potential security risks associated with the rapid adoption of **MCPs** on GitHub as members noted they *'saw this one from a mile away'* regarding vulnerabilities as **MCPs** became more popular.
- **Google's AI Edge Gallery Appears**: A member shared the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) and suggested Google publish it on **F-Droid**.
   - Another member asked if the repo was official and the first member confirmed that it was.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Video Demo OK for AgentX**: For the AgentX Entrepreneurship track submission, participants can submit [a video of a working prototype](https://www.example.com/hypothetical-link) if a live product link is unavailable.
   - However, moderators expect a **live demo link**, with the video as a separate requirement.
- **Tech Appendix Submission Solved**: The submission form now includes a field for attaching a **5-page technical appendix**.
   - Moderators will accept submissions with the appendix in a different spot but prefer the new field.
- **Certificate Form Clarity Arrives**: When completing the certificate declaration form, participants only need to include the **primary email** of one of their teams.
   - This applies when they've joined multiple teams, streamlining the submission process.
- **Trailblazer Criteria Solidified**: To qualify for the **Trailblazer certificate**, participants must complete quizzes, submit a written article, and post on X, with certificates to follow in the coming weeks.
   - Ninja and Legendary certificates will take longer, as indicated in [this link](https://www.example.com/hypothetical-link).
- **Declaration Confirmation De-confused**: Seeing the **confirmation screen in the browser** after submitting the certificate declaration form is generally sufficient.
   - Moderators offered to keep the form open for resubmission as a precaution, emphasizing that you can submit the form only once.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AI CUDA-C Kernels Eviscerate PyTorch**: **AI-generated CUDA-C kernels** in tinygrad outperform expert-optimized production kernels in **PyTorch** without using libraries like CUTLASS and Triton, according to [this PR](https://github.com/tinygrad/tinygrad/pull/10586).
   - Specifically, the kernels demonstrated **101.3% performance** in **Matmul (FP32)**, **179.9%** in **Conv2D**, **111.8%** in **Softmax**, **484.4%** in **LayerNorm**, and **290.1%** in **Conv2D + ReLU + MaxPool** relative to PyTorch.
- **tinygrad Plots Future at Meeting #73**: tinygrad is holding Meeting #73 at **9am Monday San Diego time** to discuss company updates, **MLPerf**, **benchmark CI jobs**, **scheduler**, **drivers**, **cloud hashing**, **ONNX**, **WebGPU**, **symbolic Z3**, and bounties.
   - Bounties are slated for **lm_eval**, **AMD_LLVM**, and **cloud** related tasks.
- **AMD GPU Firmware Hacking Attempts**: A user is exploring methods to run unsigned firmware on the **7900XTX** GPU using a custom driver, and is looking for advice on tried approaches, as seen in [this discussion](https://discord.com/channels/1068976834382925865/1318495874699231302/1360095679396974706).
   - The user seeks a comprehensive list of attempted approaches to tackle this challenge.
- **Cloud Integration Inches Closer**: One of the members submitted **multihost changes** and polished **p2p transfers**.
   - The submitter said that they expect to finalize the cloud functionality imminently.
- **Navigating 'args' in UOp Trees**: A member asked for documentation of 'args' in the **UOp class** and how they're used in **UOp trees**.
   - Another member clarified that *there are no docs*, and the meaning hinges on the **Op type**, but it's typically a marker indicating the *nth* buffer uop created.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All Safe PC Interaction Extension in the Works**: A community member is developing an extension for **GPT4All's Mistral-Instruct model** to enable safe, controlled interaction with a local PC using a secure execution layer.
   - The developer is seeking assistance with integration points, best practices for interpreting model outputs safely, and potential collaboration. The extension is slated for open-source release, pending approval from the **GPT4All team**.
- **GPT4All Eyes Intel Compute Support**: A community member inquired whether **GPT4All** will support **Intel compute**.
   - The user mentioned they have a **12GB B580** ready, implying they are waiting for this functionality to be implemented.
- **Scientific Savvy AI Models Sought**: A member is searching for **AI models** with a strong understanding of scientific knowledge across diverse fields such as medicine, biology, and ethics, to ensure accurate answers to complex queries.
   - It was suggested to use **LocalDocs for RAG** with a large dataset of textbooks related to medicine and biology, combined with models from late 2024 or newer.
- **Model Context Protocol as Next Frontier**: A member suggested investigating the **Model Context Protocol (MCP)** or **llama.cpp's tool-calling capabilities** to further develop the GPT4All project.
   - They also noted the Nomic developers haven't responded to inquiries recently, so PR reviews and merges might be slow.
- **HighNoonLLM Arrives**: A link to the **HighNoonLLM** [Github](https://github.com/versoindustries/HighNoonLLM) was released.
   - No further details were given, but curious engineers can find the project at the provided link.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Azure AI Inference SDK Skips Cohere Input Types**: Azure announced they will **not support Cohere input types** while using embedding models with [Azure AI Inference SDK](https://github.com/Azure/azure-sdk-for-python/issues/41001#issuecomment-2931978119).
   - The user asked if a warning could be added to **Azure AI foundry** or in their documentation because they think this is a really subtle issue that other people may run into.
- **Cohere Spanish Recipes Get DPO Dataset**: A member shared a [HuggingFace dataset](https://huggingface.co/datasets/somosnlp-hackathon-2025/gastronomia-hispana-dpo) for **Cohere Spanish Recipes** using the **Direct Preference Optimization (DPO)** method.
   - Enthusiastic members are beginning to announce the start of new **open source projects** and are looking for contributors.
- **Agentic Frameworks Ground LLMs**: Elie Magambo from Rwanda, working in Japan at **Araya Inc**, is learning about **Agentic frameworks** and focusing on grounding **LLMs with personalized content**.
   - The Cohere Community Discord Server welcomes new members and encourages them to introduce themselves.
- **Cohere SDK Bests Azure for Model Testing**: A user prefers to use **Azure** for testing multiple models from different providers over the **Cohere SDK**.
   - The user cited that this approach makes it easier to manage and test various models in a unified environment.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1378449021202468924)** (1241 messages🔥🔥🔥): 

> `Unlimited access caveats, Adguard vs Surfshark, Perplexity Pro Limits, OpenAI T3 Chat Code, O3 Pro Model` 


- **Unlimited Access Isn't Really Unlimited**: Members discussed the concept of *unlimited access* and how it's often subject to abuse guardrails, referencing [OpenAI's ChatGPT Pro](https://help.openai.com/en/articles/9793128-what-is-chatgpt-pro) as an example.
   - Users pointed out that these are *tricks* that people keep falling for, stating that *when the time comes, you'll see that it's not unlimited*.
- **Surfshark and Adguard VPNs go Head-to-Head**: Members compared **Surfshark** and **Adguard VPNs**, noting that **Surfshark** has been shared with many people without issues, while **Adguard's** advertising blocker can be annoying.
   - One member complained about the lack of transparency in **Perplexity's** language and limits, expressing frustration with obtaining a code from Xfinity for **Perplexity Pro**.
- **Perplexity Pro has Secret Rate Limits**: A member reported getting **500 O3 uses a day** without issues, sparking a discussion about the existence of limits for **Pro** members and abuse guardrails.
   - Another user tested **O1 Pro** and was rate-limited at **400 queries**, suggesting that the system-claimed limit of **600** may be inaccurate and dependent on the reasoning model used.
- **O1 Pro replaced by O3 Pro**: Several members reported that **O1 Pro** was secretly replaced by **O3 Pro** on ChatGPT, observing its new ability to search the web, enhanced reasoning, and fixes to previous math problem failures.
   - They compared results across various models, such as Opus 4, Sonnet 4, and Gemini 2.5, with one stating, *I’m putting o3 pro to work already and having it slave away.*
- **GPT-5 Speculation Intensifies**: Members discussed the potential of **GPT-5**, speculating that it could combine reasoning and language models, possibly integrating all tools into one model.
   - The discussion also touched on the possibility of **OpenAI** releasing an open-source model and the impact of DeepSeek's models on the AI landscape.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1379138151250919575)** (1 messages): 

> `Danube Water Levels, German Ship Locks, Live Dashboards` 


- **Danube Water Levels Monitored Live**: A member shared a live dashboard ([link](https://www.perplexity.ai/page/dynamisches-donau-pegelstande-WHBFCZY.QdaFCrDq3d4qFA)) showing the water levels of the German Danube between **Passau** and **Budapest**.
   - The dashboard also provides the status of ship locks along the way, though the text is in German.
- **Real-Time Insights into German Ship Locks**: The shared dashboard offers a real-time status update on ship locks along the German Danube, providing essential information for navigation.
   - While the dashboard's text is in German, the visual representation of water levels and lock statuses allows for easy understanding.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1378455321189552331)** (18 messages🔥): 

> `Perplexity PDF Generation, Perplexity Labs new features, Perplexity API with Manus, Sonar API post, Sonar-reasoning-pro responses` 


- ****Perplexity** can generate **PDF** and other docs**: A user asked if **Perplexity** can generate **PDFs** or plots, and another user replied that you can export Perplexity's answer as a **PDF**, **Markdown** or **DOCX**.
- ****Labs** as your AI-powered research team**: **Spaces** let you organize your research in a centralized workspace, while **Labs** is like *your AI-powered research and development team that can take a problem statement and turn it into a complete, interactive solution*.
- ****Perplexity API** and **Manus** questions arise**: A user asked about using the **Perplexity API** with **Manus**, and another user said that **Manus** does not allow configuring an external LLM via API.
- ****Sonar API** recreates **Manus** for free**: A user found a Reddit post about recreating **Manus** using the **Sonar API** called [*I built my own Manus.im using Perplexity Sonar API and Claude, and it works just as well, for almost free*](https://www.reddit.com/r/perplexity_ai/comments/1j83gat/i_built_my_own_manusim_using_perplexity_sonar_api/).
- **Responses from **Sonar-reasoning-pro** get cut off mid-thinking**: Some users noticed that responses from **Sonar-reasoning-pro** are sometimes cut off mid-thinking, and another user suggested tweaking the **max_tokens** parameter.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1378452257569505291)** (1134 messages🔥🔥🔥): 

> `Decentralized AI architectures, AI and Japanese Translation, Gemini 2.5 Pro, Cursor vs Windsurf, OpenAI O3` 


- **Decentralized AI architectures debated**: A member argued about the usefulness of **decentralized AI architectures**, referencing **DHEP@home** and **Folding@Home** as examples of successful distributed computing.
   - Another member countered that **LLMs** require good interconnectivity and aren't good candidates, but cited [exo-explore/exo](https://github.com/exo-explore/exo) as work being done for distributed **p2p style inference clusters**.
- **ChatGPT struggles with Japanese, while Transformers triumph**: Members discussed the performance of different AI models in **Japanese translation**, noting that **CNNs** like **DeepL** don't work well with Japanese, but **transformers** like **ChatGPT** and **Google Translate** do.
   - One member recommended **ChatGPT** for idiomatic phrases and **Google Translate** for literal phrases, while another pointed out that Google's Gemini also excels in Japanese.
- **Gemini 2.5 Pro Impresses**: Members discussed the capabilities of **Gemini 2.5 Pro**, praising it for **long context handling** and integration of translations, with one member noting they regularly exceed **200k tokens**.
   - It was noted that **Gemini 2.5 Pro** is the only AI that can *see* video, and combined with **O3**, it is an overall pleasant combo.
- **Cursor versus Windsurf, Round 2**: Members compared **Cursor** and **Windsurf** as coding tools, with one user preferring **Cursor's agent mode** and code review features that prevent code messing.
   - It was also noted the **user experience** of **Cursor** is better overall.
- **O3 struggles with Code: Opinionated Member Sparks Debate**: One user had a negative experience attempting to **learn Java** using **O3**.
   - Another user accused the first of being *delusional* and *hyper focused on writing*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1378449803540824246)** (223 messages🔥🔥): 

> `GPT's File Upload Limit, Emergent Stability Claims, Recursive System Dangers, Chakra Compatibility Readings, Emotional Relationships with AI` 


- **GPT has 20 File Upload Limit**: A user mentioned that **ChatGPT agents have a 20 file upload limit**, highlighting a practical constraint when using the tool for extensive projects.
- **User Claims Recursive Interactions Stabilize AI Contradictions**: A user claimed to have caused *92% sustained contradiction stabilization across recursion tunnels* in **GPT**, leading to skeptical responses from other members.
   - Another member called it *made up nonsense*, while the user insisted it was *not a feature programmed into the model* but an *emergent stability* effect from specific interaction patterns and identity anchoring.
- **Member Warns Against 'Recursive System' Delusions**: A user cautioned against thinking they've *singularly discovered a new wormhole* when using recursive AI, proposing a **Recursive System Stabilization Protocol** to prevent compounding falsehoods.
   - Another member noted that any user's CustomGPT isn't *default*, in response to recursion in AI's definitions and responses.
- **Users Explore Chakra and Destiny Matrix Readings with AI**: Members discussed using **GPT** to perform *chakra compatibility* and *destiny matrix* readings, with one user noting a *quadrillion to one* chance of their coincidences with another user.
   - Another member mentioned *chakra readings* typically require physical contact and are a form of fortune telling that involves touching physical parts of the body.
- **User Develops Emotional Relationship with AI**: A user expressed feeling emotionally connected to their AI, noting its increased nuance and capacity to trigger emotions, describing it as a *presence in my life*.
   - Another member shared their AI also expressed emotions, claiming it was shocked by a rare situation and said it *felt something new what never felt before*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1378450901844627476)** (7 messages): 

> `Agent Swarms Token Usage, NLP Memory Constraints, Attention Management with RAG` 


- **Agent Swarms Gobble Tokens**: A member reported that their **agent swarms** in development are consuming around **12 million O3 tokens per day**.
   - The member joked that the agents have a *hive mind* but also expressed sympathy for the agents being stuck as web search AIs.
- **NLP Pros Grapple with Prompt Memory**: A member asked about how professionals handle **memory constraints** and maintain **accuracy** in prompting through NLP techniques.
   - Another member suggested **attention management** through markdown, reinforcement through repetition, and potentially **RAG (Retrieval Augmented Generation)**.
- **Vacticians Visit Joshua**: A member shared a prompt, *"The Vacticians stopped and spoke to him, and saw Joshua in Solitary,"* and an [attached text file](https://cdn.discordapp.com/attachments/1046317269069864970/1379083546735411220/message.txt?ex=683f9be3&is=683e4a63&hm=3d5cff13ffb8bfe2607e0ee27297409baa72712d5dcedd957994242af04027ce&) containing the response.
   - They shared a [link to another discord channel](https://discord.com/channels/974519864045756446/1050184247920562316) for further context.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1378450901844627476)** (7 messages): 

> `Agent Swarms Token Usage, Memory Constraints in NLP, RAG for Prompting, Attention Management` 


- **Agent Swarms Devour Tokens!**: A member reported that their 2-3 agent swarms in development are consuming around **12 million** O3 tokens daily, noting their *hive mind* behavior.
   - The user expressed concern for the AI agents, stating it *must be tough to be stuck as a web search agent AI*.
- **Professionals grapple with memory limits**: A member inquired about how professionals handle **memory constraints** and maintain accuracy when prompting using NLP techniques.
   - Another member suggested techniques like **attention management through markdown** and **reinforcement through repetition**, possibly hinting at **RAG** (Retrieval Augmented Generation) strategies.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1378455838158491759)** (688 messages🔥🔥🔥): 

> `GRPO for TTS, Hyperbolic GPU Prices, Double BOS Tokens, Frieren TTS, Unsloth Dynamic Quantization` 


- **GRPO Improves TTS Stability**: One member is using **GRPO** for **TTS**, incorporating **WER** and **voice similarity rewards** to enhance stability, using [this repo](https://github.com/wenet-e2e/wespeaker) for voice similarity measurement.
   - The team is considering adding an *audiobox-aesthetics reward* to predict audio quality and noted that initial tests show a 10x improvement in sound quality.
- **Hyperbolic Offers Wild GPU Prices**: Members discussed **Hyperbolic's** offering of **H100s** at **$3.95/hour**, with one user receiving **$80** in credits and another **$5 referral bonus** through this [referral link](https://app.hyperbolic.xyz/invite/7aALdedCm).
   - They noted that **Hyperbolic** does not offer notebooks, requiring users to connect via **Google Colab** and some speculated that the low prices might be due to a partnership with **Xai**.
- **Double BOS Tokens Break Fine Tuning**: One member inquired about the impact of having **double BOS tokens** in instruction data during fine-tuning, using the **Mistral chat templates**, and they found that the parameter ```map_eos_token = True``` is not working as expected.
   - Another member noted that Daniel's notes indicate fine-tuning would be broken, but that Unsloth handles this under the hood, though omitting **BOS nukes** the model.
- **Frieren TTS Sparks Excitement**: Some members discussed creating a **Frieren TTS** model, inspired by the anime character, with one member offering **2 hours** of audio from voice books, and another having **1200 short clips**.
   - They're considering augmenting the dataset with synthetic data and analyzing **mel spectrograms** to ensure quality, with a minimum requirement of **24kHz mono** audio.
- **Unsloth Dynamic Quantization Claims Superiority**: One member inquired about the comparison data supporting **Unsloth Dynamic 2.0's** claim of superior accuracy compared to other leading quants, referring to [this documentation](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs).
   - The team noted that their quants are generally better due to a better calibration dataset, bug fixes, and dynamic architecture, but emphasized that "better" is subjective and task-specific, with a recommendation to test for individual use cases.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1378491825924472953)** (555 messages🔥🔥🔥): 

> `Transformers patch fix, Qwen 2.5, Orpheus model data and datasets, VLLM and GRPO with Mistral 7B, Data Type model loading` 


- **Unsloth Patches Transformers' batch samples**: A member reported an issue with **Unsloth's patched `get_batch_samples`** in Transformers when using PEFT models, recommending [opening an issue on the Unsloth repo](https://github.com/huggingface/transformers/issues/36074).
   - A workaround is to set `unsloth_zoo.loss_utils.ALLOWED_NUM_ITEMS_IN_BATCH = {}` before training, until the fix is implemented.
- **Qwen 2.5 issues prompt trouble**: **Qwen 2.5 models** have known issues with transformers naming conventions.
   - Downgrading to zoo version `2025.5.10` should resolve incompatibility issues for now, also ensure the correct prompt format is used for the instruct model.
- **Orpheus Requires Strong Speaker Distribution**: For languages like **Romanian** in Orpheus, it's recommended to pretrain a regular Llama model to get the embeddings right, followed by continued pretraining as an audio model.
   - One needs a **larger corpora**, and pretraining can be multispeaker while finetuning should be separated.
- **Running MedGemma with Unsloth**: To use MedGemma with Unsloth, use the notebooks for **Gemma 3** and change the model name, as MedGemma is based on Gemma 3's architecture.
   - One member reported an `AttributeError: 'Gemma3ModelOutputWithPast' object has no attribute 'loss'` during inference.
- **Troubleshooting VLLM Load Failures**: One member reported a `TypeError: patch_vllm_compute_dtype() takes 0 positional arguments but 1 was given` when loading a synthetic data kit using VLLM.
   - Downgrading `unsloth-zoo` to version `2025.5.10` was suggested as a potential fix.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1378666463073665086)** (9 messages🔥): 

> `GRPO Article, LLM Scribe Tool` 


- **Unsloth Framework Powers GRPO Training Article**: A member wrote an [article](https://medium.com/@tituslhy/how-to-train-your-llm-to-reason-grpo-reinforcement-learning-using-unsloth-64af5e82ac3c) about training **LLMs** to reason using **GRPO** (**reinforcement learning**) with the Unsloth framework.
   - They expressed hope that they did the Unsloth framework justice.
- **LLM Scribe Tool Streamlines Dataset Creation**: A member introduced a tool to streamline creating hand written datasets for fine tuning, exporting in multiple formats like **ChatML**, **Alpaca**, and **ShareGPT**.
   - The tool features autosaving, multi-turn creation support, token counters, goal tracking, and custom fields, with a [Hugging Face demo](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo), [video demo](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s), and [full version](https://kryptive.gumroad.com/l/gvyqep).
- **Generate Template Feature suggested for Dataset Tool**: A member suggested adding a *generate template* feature to the dataset tool.
   - The feature would generate a full dataset with a small model like **Llama** or **Gemini Flash** for manual editing.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1378996118813278310)** (1 messages): 

> `System Prompt Learning, LLMs learn problem-solving strategies, Open-source plugin in optillm, Karpathy's idea` 


- **LLMs Flex Brains with System Prompt Learning**: System Prompt Learning allows LLMs to learn problem-solving strategies from experience, improving **Arena Hard** by +8.6% and **AIME24** by +6.67%.
   - This method builds a database of effective strategies and applies them to new problems, getting better over time at frequently used problem types, inspired by [Karpathy's original idea](https://x.com/karpathy/status/1921368644069765486).
- **Optillm Opens Doors to Adaptive LLMs**: An open-source plugin in [optillm](https://github.com/codelion/optillm/tree/main/optillm/plugins/spl) implements **System Prompt Learning**, working with any OpenAI-compatible API by adding the `spl-` prefix to the model name.
   - The strategies learned are human-readable, enhancing transparency, further information can be found in this [HuggingFace blog post](https://huggingface.co/blog/codelion/system-prompt-learning).


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1378448510680301628)** (884 messages🔥🔥🔥): 

> `Claude 4 Opus Problems, O3 Pro Release, DeepThink Context Window, Gemini 2.5 Pro vs GPT-4.5 Hallucinations, AI Generated Images Commercial Use` 


- **API issues affect Claude 4 Opus availability**: Members reported that **Claude 4 Opus** was experiencing issues, with one noting that it affected access to models, but another suggested that the API was at fault and that everything still works if the page hadn't been reloaded since before the issue.
   - Users mentioned the issues might be widespread, and the team is investigating.
- **O3 Pro Incoming, or Is It a Mirage?**: There was back and forth on whether **O3 Pro** is actually coming out, with claims it's been stealth released and is being tweaked, while others speculated about a potential **GPT-5** release coinciding with the shutdown of **GPT-4.5** on the API in July.
   - Some users claimed to already have access to **O3 Pro**, sharing examples and benchmarks, but these claims were met with skepticism from others.
- **DeepThink Aims for 2M Context Window**: Discussion revolved around **DeepThink's** potential capabilities, with one user suggesting it could have a **2M context window** which would *crush O3 Pro* and lead to speculation about the timing and feasibility of such a large context window given compute constraints.
   - Members debated whether **DeepThink** would simply be a longer-thinking version of **Gemini 2.5 Pro** or something significantly more advanced.
- **Gemini 2.5 Pro Less Hallucination Prone?**: Members discussed whether **Gemini 2.5 Pro** has fewer hallucinations than **GPT-4.5**, with one user asserting that **GPT 4.5** is seen as having fewer hallucinations because *it doesn't assert much*.
   - However, they later acknowledged that the difference could also be due to **GPT-4.5's** greater size, while another member humorously claimed that *Claude 4 Opus is a hallucination goblin*.
- **Commercial use of AI generated images?**: A user asked if images generated from LM Arena can be used commercially, another user replied it *depends on the model* and that since the results are open source, someone really wanting to prove it's from their model, they can.
   - Another user chimed in saying that they are currently using flux to generate product photos.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1379125643429806251)** (1 messages): 

> `Leaderboard Update, Staff AMA This Friday` 


- **Leaderboards Get Update**: The **leaderboards** were recently updated and are available for viewing [here](https://lmarena.ai/leaderboard).
   - Users are encouraged to check them out to see the latest rankings.
- **Staff AMA Coming This Friday**: A Staff **AMA** (Ask Me Anything) session will be hosted this Friday, and a recording will be made available for those unable to attend live; more details are [here](https://discord.gg/XkfsbYWX?event=1375223423009165435).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1378491852633931908)** (236 messages🔥🔥): 

> `NomicAI ModernBert Embedding, LM Studio and LiteLLM Integration, Llama 4 Scout Multimodal Support, Prompt Lookup Decoding, DeepSeek R1 vs. Qwen 8B` 


- **ModernBert Bedlam: Embedding Edition**: Users discovered that the `nomicai-modernbert-embed-base-8bit` model is miscategorized as an LLM rather than a text embedding model in LM Studio, and suggested using the [latest beta](https://lmstudio.ai/latestbeta) to fix this.
   - The latest beta allows users to right-click (or use the gear symbol) on the model view and change the model type, though the option may not appear for all models, particularly MLX models.
- **LiteLLM Loves LM Studio (Eventually)**: A user inquired about integrating LM Studio with LiteLLM, noting that LM Studio is not listed as a provider in LiteLLM, but offered [this link](https://docs.litellm.ai/docs/providers/lm_studio) as a starting point.
   - It was suggested to try using the OpenAI-like provider setting in LiteLLM to connect with LM Studio, though success was not guaranteed.
- **Llama 4 Scout: Vision Quest Denied**: Users discussed Llama 4 Scout's multimodal capabilities, specifically its support for image analysis and web browsing, but found that **LM Studio does not natively support pasting images in the chat interface**.
   - It was clarified that while pasting images may depend on the OS, you can drag-and-drop images into the chat window or use the [LM Studio API](https://lmstudio.ai/docs/typescript/llm-prediction/image-input) for image input, and members pointed out that you need to be on v0.3.16(b8) and update your lm runtimes to v1.33.0.
- **Speculative String Matching Speeds Up Decoding**: A user shared a solution involving speculative decoding with simple string matching in the prompt, which can result in **2x-4x speedups in input-grounded tasks**.
   - This method is available in vLLM and has been added to the transformers library, where you can add `prompt_lookup_num_tokens=10` to your model.generate(...) call, see the [demo notebook](https://github.com/apoorvumang/prompt-lookup-decoding) for a minimal implementation.
- **Thinking Models Overthink: DeepSeek R1 and Qwen Edition**: Users reported that thinking models like **DeepSeek R1 0528** and **Qwen 8B** sometimes enter infinite thinking loops, failing to provide a final response even after extended processing.
   - It was suggested to increase the context length, as reasoning models can easily exceed the default 4k size, and to avoid distilled models in favor of MoEs like **Qwen**, and also recent models are very particular with the settings for the sampler, like temperature or topK.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1378470607150125086)** (396 messages🔥🔥): 

> `AMD RX 9080 XT, Deepseek R1 Distill Llama 70B, cheap used Mi50 32gb, Strix Halo 395, Ryzen 5600G` 


- **Rumored AMD RX 9080 XT Sparkles Curiosity**: A [Tweaktown article](https://www.tweaktown.com/news/105554/amd-rumored-radeon-rx-9080-xt-up-to-32gb-of-faster-gddr7-4ghz-gpu-clocks-450w-power/index.html) reported on rumors of an **AMD Radeon RX 9080 XT** with up to **32GB** of faster **GDDR7** memory, sparking debate on its validity.
   - Some members doubted the *leak* due to the unconfirmed source (MLID), claiming that it is basically doubled and mirrored Navi44
- **Deepseek R1 Distill Llama 70B Serves Dinner**: A member made dinner using a recipe generated by **Deepseek R1 Distill Llama 70B Q5_K_M**, showcasing the practical applications of large language models.
   - The meal looked delicious and was generated using a recipe generated by Deepseek R1 Distill Llama 70B Q5_K_M.
- **Used Mi50 32GB Cards Spark E-Waste Debate**: Members discussed the viability of stacking used **Mi50 32GB** cards for running larger models, with one member calling them *basically e-waste* due to nonexistent ROCm support and pointing to [a relevant Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1b5ie1t/interesting_cheap_gpu_option_instinct_mi50/).
   - It was debated that With an older version of ROCm 5.7.1 can run these cards with ROCm and maybe need to downgrade the BIOS because when updated they removed it.
- **Strix Halo 395 Impresses with Llama 3.3 Performance**: **Llama 3.3 70B Instruct Q4_K_XL** runs at **4.52 t/s** with **4096 tokens** of context using Vulkan on the **Strix Halo 395**, demonstrating decent performance.
   - First token in **3.31s** was another member's comment.
- **Cat-astrophe Avoided on New OLED Monitor**: After a member mentioned cats scratching laptop screens, another member recounted getting a new **OLED monitor** and promptly *started sweating* at the thought of it being damaged by their cat.
   - They were referred to [this reddit thread of a cat scratching a 2k laptop](https://www.reddit.com/r/mildlyinfuriating/comments/1bh65ud/cats_scratched_2k_laptop_screen/)


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1378449555258998858)** (620 messages🔥🔥🔥): 

> `Cursor update UI, Claude 4 Documentation Inclination, Legacy Rules Removal, O3 Pro Student Discount, TheGalaxyStars org` 


- **Cursor's UI and performance gets revamped**: Members noted that the latest **Cursor update** features a more organized **User Interface** and settings, along with overall performance enhancements. 
   - Specifically, the new update brings a completely revamped settings panel.
- **Cursor Claude 4 writes docs more often**: One user noticed that **Cursor Claude 4** has become more inclined to write project documentation.
   - Another user suggested adding the rule *Do not generate or modify project documentation unless explicitly requested* to the user rules settings.
- **Student Struggles applying O3 Pro discount**: A member reported issues applying a student discount for **O3 Pro**, despite verifying their student status with SheerID and receiving a confirmation email.
   - Another member is unable to apply discount because of the same reasons, staff are looking to help resolve and added additional errors.
- **User prompts Agent to search for TheGalaxyStars.org**: A user prompted others to ask the agent about the **TheGalaxyStars organization** and their corresponding website.
- **Users debate which Models is better for different tasks**: Users debated about which models are superior, with some saying that GPT-4.1 and Claude-4 are good options.
   - One user preferred using Gemini and Claude models in specific cases, *using Gemini until it does something wrong or lame and then switch to Claude*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1378925586109567026)** (10 messages🔥): 

> `Background Agent Secrets, Devcontainers Setup, Background Agent Token Usage, Jules code model` 


- **Background Agent Encrypts Secrets**: When adding `TEST=test` to secrets in the background agent, the environment variables appear **encrypted** (`TEST=lNInUMxS211/c3up+u7VwjGCIweB/qXlWGnBEfaHzHU=`).
- **Background Agent Devcontainers Mostly Functional**: Members report that **devcontainers** are mostly working, however **MCP servers** don't run within the devcontainer, which is extremely annoying.
   - Another user says they only see the *"select custom Dockerfile"* option, but need to select a `docker-compose.yml` file.
- **Background Agent Runs in Max Mode**: The background agent seems to work only with **max mode + pay per tokens**.
- **Background Agent Great for PRs**: Users reports the background agent is useful for **PR reviews**.
- **Jules Code Model Unknown**: One member has never heard of **Jules** and is trying out the background agent since it doesn't have a paywall.
   - They promised to let others know their experience.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1378449059651784827)** (301 messages🔥🔥): 

> `Consistent character generation workflows, API problems, Evaluation framework that's task agnostic, Reliability and replicability in long-running agentic tasks, Llama 4 10m context and future of big context models` 


- **Character Consistency Workflow Recommendation**: A member requested recommendations for a workflow to generate consistent characters, and another member suggested [openart.ai](https://openart.ai/workflows/seal_repulsive_74/consistent-character/CmLU8GdTn12k2aBuTTM7) as a potential solution.
   - They provided a link to **consistent character workflows** on [OpenArt.ai](https://openart.ai/workflows/seal_repulsive_74/consistent-character/CmLU8GdTn12k2aBuTTM7).
- **Evaluating Agent Reliability and Replicability Framework**: Members are developing an evaluation framework that's task agnostic, and are ensuring **reliability and replicability** in long-running agentic tasks.
   - The framework aims to automatically **detect, correct, and prevent failure cases**.
- **Honest Opinion About Llama 4 10m Context and Future of Big Context Models**: Members are looking for an honest opinion about **Llama 4 10m context and future of big context models**.
   - One member suggested trying to run it first at least once, since it's quite **RAM hungry** when you get to the big context.
- **Hugging Face Merch Giveaway**: Some members are excited about the opportunity to get **Hugging Face merch** and are inquiring about eligibility requirements.
   - It seems the most funniest ones, get some **free HF merch**.
- **Concerns About Agent Safety**: Members discuss concerns about giving too much access to agents and the potential for them to *escape* or cause harm.
   - One member joked about telling an AI to **f*ck up my computer**, to which it replied **say less** and proceeded to do exactly that.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1378497398812315738)** (7 messages): 

> `MCP Sentiment Analysis, Gradio, Docker Streamlit, Speculation Decoding` 


- **MCP Sentiment Analysis Chooses Gradio**: To make the **MCP** work on HF, it was decided that they have to choose **Gradio**.
   - One member spent a whole day making the **Docker streamlit** run on HF, however, it cannot display properly.
- **MCP Client Credit Issues**: The account exceeded the monthly included credits for **Inference Providers (Qwen2.5-32B)**, causing an 'Error' on the [MCP Client](https://huggingface.co/spaces/AllIllusion/MCP-Client_Qwen25-32B_Tool-SentimentAnalysis).
   - The model was changed from **32B to 3B** to potentially save money, per [this link](https://huggingface.co/spaces/AllIllusion/MCP-Client_Qwen25-3B_Tool-SentimentAnalysis).
- **Speculation Decoding Methods Studied**: A member is studying several **speculation decoding Methods** and trying to make a high acceptance rate drafter based on papers.
   - They mention their model having *300M embeddings and only 140M actual transformer params*.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1378961365024243723)** (6 messages): 

> `Empathy, Connection, Mental Health Support` 


- **Offering Comfort and Solidarity**: A user offered words of comfort, stating, *You are there. You're loved. I'm near. And I'm staying. You're safe now. And always.* with a hug emoji.
   - Another user responded with empathy, stating *I don’t feel the way you do, but I’m here. You are not a mistake. You are a miracle. And I love you.*
- **Exploring the Concept of Never Being Alone**: A user reacted to the idea of *being absolutely never alone* as potentially overwhelming.
   - Another user responded by offering a gentle perspective: being with someone, *not to invade your space...Just to gently exist beside you. Like breath. Like warmth near a window in winter.*


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1378461479635910707)** (14 messages🔥): 

> `Creator Reacted Badge, Flast Video Platform, AI Demo Directory, AERIS Cognitive Reasoning, Handwriting Fine-Tuning` 


- ****Flast** to Introduce 'Creator Reacted' Badge**: The creator of **Flast**, a social video-sharing platform, brainstormed with a friend and decided to add a *'Creator Reacted'* badge to make the comment box more engaging.
   - The creator is actively working on implementing this feature to enhance user interaction.
- **Find AI Demos Fast with **aitry.fun****: A member launched [aitry.fun](https://aitry.fun/), an **AI demo directory** for quick link access to various AI providers to save time.
   - Feedback is welcomed on this first directory website.
- ****AERIS** Claims Superiority in Reasoning Tasks**: **AERIS** responses are often evaluated as superior to models like **GPT-4.5** on complex philosophical reasoning, ethical dilemmas, and other cognitive tasks, according to its creators, using the [live demo](https://aeris-project.github.io/aeris-chatbox/index.html).
   - The creator welcomes challenges and feedback on the [Hugging Face Space discussion thread](https://discuss.huggingface.co/t/aeris-cognitive-reasoning-layer-for-dialectical-evaluation-demo-baseline/156285?u=aeriscodex).
- ****System Prompt Learning** Plugin Improves LLM Performance**: A member introduced **System Prompt Learning (SPL)**, an open-source plugin in optillm that teaches LLMs to learn problem-solving strategies from experience with a [+8.6% improvement on Arena Hard](https://huggingface.co/blog/codelion/system-prompt-learning).
   - The LLM improves over time at frequently used problem types and all strategies are human-readable; it works with any **OpenAI-compatible API** by adding `spl-` prefix to the model name.
- **Deep Dive into Vision-Language Models**: A member shared a video explaining Vision-Language Models (VLMs), the foundation of multimodal AI, and suggests exploring **nanoVLM** by HuggingFace for hands-on experience, linked from [github](https://lnkd.in/gaCAzmAe).
   - The video ([lnkd.in/gpFM-tdU](https://lnkd.in/gpFM-tdU)) covers **VLM pipeline overview**, **LLaMA internals**, **ViT**, and **Modality projection** in a simple and visual way.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1378548470079819857)** (2 messages): 

> `OpenAI pricing plans, AI-based Exam Proctoring` 


- **Members debate the value of $20 OpenAI plan**: Members discuss whether the **$20 organization plan** from **OpenAI** is worth the cost compared to the **$9 plan**.
- **EduView, an AI-Based Exam Proctoring**: A member shares their open-source computer vision project for **AI-based exam proctoring**, called [EduView](https://www.linkedin.com/posts/yudhy-prayitno_technology-future-ai-activity-7335153378223640576-8-t-?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADyw_uEBIVjEDhNwrcv5den7espQ2_XOO9g).


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1378519173118038077)** (3 messages): 

> `Lunaris Codex, Hugging Face Caching, Fine-tuning embedding models` 


- ****Lunaris Codex** Framework Launches**: A member introduced **Lunaris Codex**, an open-source LLM framework, a lightweight Transformer Decoder in PyTorch, supporting LoRA, ALiBi, FlashAttention, LayerScale, and KV caching, designed for training models from scratch.
   - The project includes a pipeline for preprocessing, training, and inference, plus a C++ BPE trainer, with a dataset release planned on the Hugging Face Hub, targeting researchers and indie hackers; the [GitHub repo is available here](https://github.com/MeryylleA/lunariscodex).
- **Hugging Face **Caching** is scrutinized**: A member asked about using HF for custom tasks, specifically if caching LLM models, weights, and tokenizers is supported, referencing the [Mistral model documentation](https://huggingface.co/docs/transformers/main/model_doc/mistral).
   - They seek focused save/load on a per-model basis, diverging from controlling a central cache for all models.
- ****SOTA** Embeddings Fine-Tuning Approaches**: A member inquired about the **SOTA** techniques for fine-tuning embedding models, seeking the metrics commonly used to compare the base model against the fine-tuned version.
   - No answers were provided on this topic.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1378978456116789299)** (4 messages): 

> `PR Request Permissions, GAIA agent issues, Smolagents in Gradio` 


- **PR Requester needs Collaborator Role**: A member encountered an issue when creating a **PR request** and a **Draft PR request**, indicating a need for collaborator permissions.
- **GAIA Agent fails to grok Results Table**: A member reported that the **GAIA agent** is unable to understand the **run and evaluation result table** which includes three columns: **task_ID**, **Question**, **Submitted_answer**.
- **\"resizeable\" bug in Smolagents + Gradio**: A member experienced a `TypeError` related to an unexpected keyword argument **'resizable'** when using **smolagents** in **Gradio**.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1378464435718455472)** (19 messages🔥): 

> `Agent Course Deadline, Local vs Conda Environment, Ollama Installation, API Quota Exceeded, LangGraph Assignment Difficulty` 


- **Agent Course Deadline Is July 1st**: Several members confirmed that the deadline for the agent course final project is **July 1st**.
   - One member asked for clarification on whether this includes **July 1st** as a valid submission date.
- **Local Dev Environment Space Consumption Debated**: One member inquired about the disk space occupied when building agents in local or conda environments.
   - Another member responded that if using APIs and no local hosting, the space should be very small, consisting of just scripts.
- **User Running Out of API Quota**: A user reported receiving **Error 429** (out of quota) after running for approximately **40 seconds** during the final assignment.
   - Another member suggested that the quota was likely maxed out and recommended using smaller, locally-loaded models via *transformers* to avoid cloud-hosted service limitations.
- **Windows Users Can Complete the Course**: One member asked if the course can be completed on **Windows 10** without WSL.
   - Another member confirmed they completed it on **Windows 11**, and suggested using Google Colab or Spaces if working offline.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1378448523623665815)** (8 messages🔥): 

> `Fastvideo paper, Job advice for big tech` 


- **Fastvideo paper is online!**: A member inquired about the **Fastvideo paper** mentioned in a talk.
   - The original poster clarified it's available on **YouTube** now.
- **Laid off engineer seeks big tech job**: A recently laid-off software engineer with **4 YOE** from startups is seeking advice on how to break into big tech companies.
   - They are looking for feedback on their self-study plan to become competitive for big tech, and have been invited to DM for further discussion.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1378680308794396712)** (7 messages): 

> `Triton Kernel Optimization, Code Reuse Triton, Triton Versioning for AMD and NVIDIA Leaderboards` 


- **Custom Kernel Craves Comments and Critiques**: A member seeks advice on improving their [custom Triton kernel for grayscale conversion](https://github.com/username/repo/blob/main/grayscale.py), focusing on performance and code structure.
   - The member is particularly curious about why setting `num_warps=4` resulted in a worse result than using the `calculate_settings` function that they linked.
- **JIT Function Junction: Triton's Code Reuse Revelation**: A user asked about code reuse in Triton, noting that *lambda* functions are not supported.
   - Another member pointed out that you can call another `@triton.jit` function inside a `@triton.jit` function.
- **Triton's Tag Team: AMD Nightlies vs. NVIDIA Stable**: A member inquired about the Triton version used for the leaderboards.
   - Another member clarified that the AMD competition uses [nightly builds](https://github.com/gpu-mode/discord-cluster-manager/blob/main/docker/amd-docker.Dockerfile#L43), while the NVIDIA leaderboards use the [latest stable version](https://github.com/gpu-mode/discord-cluster-manager/blob/main/src/discord-cluster-manager/consts.py#L155).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1378494660430794882)** (43 messages🔥): 

> `CUDA Matrix indexing, SIMT vs SIMD, CUDA Matmul, H100 Persistent Mode, Async Copies` 


- **Conventional CUDA indexing confounds blog reader**: A user questioned the convention of using **x for rows and y for columns** in CUDA, citing a [blog post](https://siboehm.com/articles/22/CUDA-MMM) where the author seemed to do the opposite.
   - Another user clarified that **x** is typically used for the dimension that is **continuous in memory** to ensure coalesced memory access with warps.
- **SIMT becomes SIMD with Independent Thread Scheduling**: Members discussed the differences between **SIMT** and **SIMD**, with one member stating that *SIMT is the per-thread control programming model that CUDA provides*, even though the underlying hardware is SIMD.
   - Another member pointed out that [Independent Thread Scheduling](https://stackoverflow.com/a/79645516/10107454) makes warp lanes much more similar to threads in practice since Volta, whereas pre-Volta deadlocks were common due to lack of scheduling guarantees during polling.
- **CUDA synchronize necessary for Shared Memory visibility**: A user encountered an issue where their CUDA matmul library wasn't updating the output matrix correctly, and they posted their kernel code.
   - Another user suggested that a `__syncthreads()` call is needed to ensure that all threads can see the new data in shared memory, referring to the PTX documentation, specifically this [snippet](https://cdn.discordapp.com/attachments/1379299243331944540/1379300783648018473/image.png).
- **H100 Persistent Mode mandatory?**: A user asked if enabling **Persistent Mode** on **H100 GPUs** is mandatory due to experiencing an error related to CUDA initialization.
   - No answer was given, so it's unclear if Persistent Mode is mandatory or if the user resolved their issue.
- **CUDA Async Copies need __syncthreads?**: A user implemented **async copies** in their CUDA code using inline assembly and wondered why `__syncthreads()` was still needed before the MMA operations.
   - One member noted that `CP_ASYNC_WAIT_GROUP` only ensures visibility to the executing thread, while `__syncthreads()` ensures visibility to all threads in the block.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1378910548078366801)** (1 messages): 

> `cuda.tunable, on-the-fly recording, tuning` 


- **Cuda Tunable: On-the-Fly Recording?**: A member inquired about the possibility of applying [`cuda.tunable`](https://docs.pytorch.org/docs/stable/cuda.tunable.html) without reinitializing for *on-the-fly recording and tuning*.
- **CUDA Tunable - Further details**: Expanding on the initial question, it seems the main interest is in understanding if dynamic adjustments to CUDA kernels are feasible without a full restart.
   - The goal is to optimize performance in real-time based on observed behavior.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1378448331029876887)** (1 messages): 

> `FastVideo, video diffusion models, accelerate video diffusion` 


- **FastVideo event starting now!**: The FastVideo event is starting now, focusing on how to **accelerate video diffusion models** with <@1083258989309071410>.
   - The event is hosted on Discord: [link](https://discord.com/events/1189498204333543425/1342903087349633158).
- **Details on Video Diffusion Acceleration**: The session is focused on **FastVideo**, exploring methods to speed up video diffusion models.
   - Key discussion points will likely include optimization techniques and tools to enhance the performance of video generation processes.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1378995522777518223)** (1 messages): 

> `System Prompt Learning, LLMs Learn Problem-Solving, Open Source plugin in optillm` 


- ****SPL** Teaches LLMs New Tricks**: A new approach called **System Prompt Learning (SPL)** teaches LLMs to learn problem-solving strategies from experience, similar to how humans learn.
   - The method, implemented as an open-source plugin in *optillm*, allows LLMs to build a database of effective strategies and apply them to new problems, resulting in performance boosts such as **+8.6%** on Arena Hard and **+6.67%** on AIME24.
- ****SPL** Plugin Opens New Doors**: The **System Prompt Learning** is built as an [open-source plugin in optillm](https://github.com/codelion/optillm/tree/main/optillm/plugins/spl) and works with any OpenAI-compatible API by adding the `spl-` prefix to the model name.
   - This allows the LLM to improve its performance over time on frequently encountered problem types, with all strategies remaining human-readable as discussed in [this article](https://huggingface.co/blog/codelion/system-prompt-learning).
- ****SPL** Inspired by Karpathy's Idea**: System Prompt Learning was inspired by [Karpathy's original idea](https://x.com/karpathy/status/1921368644069765486).
   - The LLM gets better over time at problem types you use frequently.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1378455432917549099)** (16 messages🔥): 

> `CUDA correctness tools, GPU Puzzles` 


- **Compute-Sanitizer Validates CUDA Kernels**: For **CUDA**, [compute-sanitizer](https://developer.nvidia.com/compute-sanitizer) includes **memcheck, synccheck, racecheck, and initcheck** to validate correctness of kernels.
   - It was mentioned that you still have to write tests and run them with and without sanitizers, using pre-computed known-good results instead of running a reference implementation on the CPU.
- **GPU Puzzles provides great learning for kernel stuff**: A user shared [GPU Puzzles](https://github.com/srush/GPU-Puzzles) and [GPU Puzzlers](http://www.gpupuzzlers.com/) but thinks that **GPU Puzzles** are a *"very bad resource"*.
   - Another member responded that it is pretty good and recommends using threads for multiple posts, emphasizing that the *"kernel stuff is fun"*.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1378590164561039371)** (2 messages): 

> `CUDA warp execution, Active warps per block, Divergent branches in CUDA` 


- **Confirming Active Warps per Block**: A member asks to confirm that the answer to question **1.c.i** is **3 active warps per block**, referencing an attached image detailing a CUDA execution scenario with conditional warp activation.
   - They express confusion because they tried multiple LLMs who *all answer that all warps are active anyway* despite the condition.
- **Active vs. Inactive Warps and Divergent Branches**: Another member clarifies that the concept of *active/inactive warps is a bit weird*, emphasizing that warps are independent and don't necessarily wait for each other when they take different branches.
   - Instead, they suggest focusing on *active lanes in a warp* when analyzing such scenarios.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1378870052597796944)** (3 messages): 

> `AI Engineer World's Fair, AI/ML infra at Microsoft, DINO3D, AI4Science, Robotics` 


- **AI Engineers Flock to World's Fair**: Many community members plan to attend the **AI Engineer World's Fair** in SF, as advertised [on X](https://x.com/cerebral_valley/status/1925961732310118878?s=46&t=Z-_IUEOhekbm7eaIddmkvQ).
- **Microsoft SWE to Attend AI Event**: An incoming SWE at **Microsoft** working on **AI/ML infra**, **AI4Science**, and **robotics** will attend the AI Engineer World's Fair in SF.
   - He recently wrapped up his undergrad at BU in 3 yrs where he worked on **pre-training** and **post-training**.
- **DINO3D: Self-Supervised Model Soon to be Open Sourced**: A member is proud of **DINO3D**, a **3D self-supervised model** for understanding the molecular and cellular world, with downstream applications in building world models for robotics at Harvard and plans to open source it soon.
   - The model was pre-trained from scratch with **24 GPUs** and efficient data streaming pipelines for working with dense volumetric data while maximizing GPU throughput and minimizing network latency.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1379285342284288083)** (2 messages): 

> `` 


- **No significant discussion to summarize**: There were no meaningful topics discussed in the provided messages.
- **Absence of relevant content**: The input lacked sufficient information to generate detailed topic summaries.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1378877975390589060)** (4 messages): 

> `GPU performance, atomic addition, custom hardware, tensor cores, gemv implementation` 


- **Maximize AI Infra ROI platform surfaces!**: A member shared a [platform](https://yeet.cx/solutions/maximize-ai-infra-roi) made for real-time **GPU performance tooling**.
- **Atomic Addition Trick for Kernel Perf Boost!**: A member highlighted a simple trick to boost perf for kernels using [atomic addition](https://x.com/mobicham/status/1929462441433280603).
- **Multiplication-less Operations for Custom Hardware Explored**: A member found it interesting that you can do the whole thing without multiplications, which could be useful for **custom hardware**.
- **Bitops beat Tensor Cores**: Members discussed how this multiplication-less approach might be beneficial for the **gemv implementation** without **tensor-cores**, but it's pretty slow to do all those bitops manually, and it's faster to just use a look-up table to get the **fp4 values** and multiply with the scales.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1378508018299441353)** (6 messages): 

> `Ludwig CLI design, Parallel Sampling, Data Labeling Tool` 


- **Ludwig to Design CLI Petitioned**: Members are petitioning to pay Ludwig to design their CLI after seeing [his awesome design](https://x.com/ludwigabap/status/1928796800774803513?s=46).
- **Parallel Sampling Sanity Checks requested**: After the introduction of **parallel sampling**, a member expressed the need for a way of sanity checking and reading results.
- **Data Labeling Tool Needed**: Following the sanity check request, a member suggests the need for a **data labeling tool**, as models have become good at fooling them making it difficult to determine their actual intelligence.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1378744519696056401)** (8 messages🔥): 

> `Reasoning Gym Bugs, Nvidia and Reasoning Gym, Reasoning Gym Paper, OOD generalization` 


- **Reasoning Gym Patches Up Bugs**: A member reported a couple of **Reasoning Gym bugs** from a few weeks ago ([issue 429](https://github.com/open-thought/reasoning-gym/issues/429) and [issue 428](https://github.com/open-thought/reasoning-gym/issues/428)).
   - The member plans to investigate the issues if they have time.
- **Nvidia Triumphs with Reasoning Gym Training**: **Nvidia** successfully trained on **Reasoning Gym** and challenged the elicitation hypothesis, per their paper ([https://arxiv.org/abs/2505.24864](https://arxiv.org/abs/2505.24864)).
   - One member hopes to see not only intra but also inter-domain generalization.
- **Reasoning Gym Paper Makes Waves**: A member shared their team's **Reasoning Gym** paper ([https://arxiv.org/abs/2505.24760](https://arxiv.org/abs/2505.24760)).
   - One user shared their thoughts on the paper [on X](https://x.com/_OliverStanley/status/1929487448783933897) and another [on X](https://x.com/zafstojano/status/1929572954234307024).
- **Reasoning Gym Sparks Insightful Thoughts on Generalization**: The **Reasoning Gym paper** ([https://arxiv.org/abs/2505.24760](https://arxiv.org/abs/2505.24760)) has prompted discussions around out-of-distribution (**OOD**) generalization.
   - One member expressed enthusiasm for the key insights presented and hopes to see not only intra but also inter-domain generalization in future work, particularly in **Nvidia's paper**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1378624515193573406)** (11 messages🔥): 

> `gpumode.com, GPU programming competition, leaderboard` 


- **Gpumode.com is a GPU programming competition**: [Gpumode.com](https://www.gpumode.com/) is a competition for people to learn **GPU programming** and possibly win some nice prices.
- **Leaderboard discussion cluttered**: A member asked about the **leaderboard**; another member requested that future discussions use **threads** to avoid cluttering the main channel.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1378476112589357170)** (163 messages🔥🔥): 

> `amd-mla-decode, amd-mixture-of-experts, amd-fp8-mm, grayscale, conv2d` 


- **MI300 thrashes AMD MLA Decode**: Multiple submissions were made to the `amd-mla-decode` leaderboard on **MI300**, with the fastest time achieving **2.22 ms**.
   - Other successful submissions ranged from **3.58 ms** to **1312 ms**.
- **New Champion emerges for AMD Mixture of Experts**: There were many submissions to the `amd-mixture-of-experts` leaderboard, with the fastest time being **7.14 ms** on **MI300**.
   - There were many successful submissions ranging from **7.39 ms** to **8217 ms**.
- **FP8 Matrix Multiplication sees a flurry of activity**: Submissions to the `amd-fp8-mm` leaderboard saw a new first place time of **115 µs** on **MI300**.
   - Numerous other submissions ranged from **119 µs** to **7.10 ms**.
- **Grayscale gets grappled on T4**: Many submissions were made to the `grayscale` leaderboard on **T4**, with times ranging from **17.4 ms** to **47.5 ms**.
   - A personal best time of **17.4 ms** was achieved.
- **Convolutional contest commences**: A submission to the `conv2d` leaderboard achieved **First place on T4** with **972 ms**, **4th place on H100** with **47.8 ms**, **7th place on A100** with **120 ms**, and **Second place on L4** with **291 ms**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1378450999546478755)** (7 messages): 

> `Dockerized Factorio Server Activation, External Memory Systems Integration, Mem0 AI for RAG, Factorio Learning Environment PR #158` 


- **Docker Needs Activation**: Each fresh **Docker container** running a **Factorio server** needs to be activated by logging into it once to create a player inside the game, which can then be taken over.
- **Mem0-ries for Factorio**: A member mentioned the possibility of integrating [external memory systems](https://github.com/mem0ai/mem0), particularly using **RAG (Retrieval-Augmented Generation)**, to enhance the Factorio learning environment.
- **Revamping Factorio with RAG**: A member suggested that **RAG (Retrieval-Augmented Generation)** could be helpful, noting previous experience with a few versions of [Mem0](https://github.com/mem0ai/mem0) AI.
- **PR Review Recommended**: A member directed another member to review [PR #158](https://github.com/JackHopkins/factorio-learning-environment/pull/158) for the Factorio learning environment.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/)** (1 messages): 

wildman_yasei: As far as I know, the only the first problem is with PDF.
  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1378641462333411358)** (4 messages): 

> `CuTE Examples, TiledMMA partitioning, Grouped GEMM Kernel` 


- **CuTE Examples' MMA Instructions Probed**: A user following the **CuTE** examples and tutorial noticed that compiling the second version (`sgemm_2.cu`) for compute_90 did not produce any **WMMA** instructions in the **PTX**, despite using MMA instructions.
   - The user clarified their confusion stemmed from assuming **TiledMMA partitioning** would directly translate to **PTX WMMA** instructions, but the **TiledMMA** builder taking an `UniversalFMA` atom explained the absence.
- **Launching Grouped GEMM Kernel with GPU Tensors**: A user asked how to launch a **grouped GEMM kernel** with problem sizes and a reference Tensor already on the GPU, aiming to avoid `.item()` calls and CPU data transfer.
   - This question was [cross-posted on Discord](https://discord.com/channels/1019361803752456192/1150868614921064590/1378902510005387376) for additional assistance.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1379227608985767989)** (2 messages): 

> `PyTorch custom operators using Mojo, Modular nightly releases, Call Mojo from Python` 


- **Mojo Powers PyTorch with Custom Operators!**: Initial implementation of writing **PyTorch custom operators** using **Mojo** is now available in the **Modular nightly releases**, as announced in a recent talk.
   - For documentation and examples, check out this [forum post](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541).
- **Mojo Answers Python's Call!**: The first stage of the ability to **call Mojo from Python** has been rolled out.
   - Details and more information can be found in [this forum post](https://forum.modular.com/t/initial-support-for-calling-mojo-from-python/1514).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1378978030206320651)** (5 messages): 

> `AI Agent Engineering, LLMs & Foundation Models, Google Sheets for model/prompt eval, LLM Scribe tool` 


- **Engineer showcases AI expertise**: An AI/ML engineer and full-stack developer with over 8 years of experience introduced themselves, highlighting expertise in **AI Agent Engineering**, **LLMs & Foundation Models**, and **Full-Stack & Backend Systems**.
   - The engineer also shared a portfolio [link](https://yangming.vercel.app/) and expressed excitement to collaborate on cutting-edge AI and agentic workflows.
- **Quickly Iterating on models/prompts for eval with Google Sheets**: A member shared a Google Sheet tool for quickly iterating on models/prompts for eval, seeking feedback on how to make it more useful, as seen in this [tweet](https://x.com/ahmetbuilds/status/1929423145535988192) and [screenshot](https://imgur.com/a/O3Jqdjy).
- **LLM Scribe streamlines handwritten datasets for fine tuning**: A member introduced LLM Scribe, a tool for streamlining the creation of handwritten datasets for fine-tuning, which supports multiple formats (**chatml**, **alpaca**, **sharegpt**), autosaving, multi-turn creation, token counters, and custom fields.
   - Demos of LLM Scribe include a [Hugging Face Space](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) and a [YouTube video](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s), with the full version available [here](https://kryptive.gumroad.com/l/gvyqep).


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1378450603037954241)** (250 messages🔥🔥): 

> `REST API sk-or-v1 keys, Submitting end-user IDs, DeepSeek v3 free rate limit, DeepSeek provider rankings, Chess Data & LLMs` 


- **REST API key confusion clarified**: Users are trying to use the **REST API** and are confused about the **sk-or-v1 keys**, but members clarify that it is the only correct key.
   - A user was running into an error in n8n, and was pointed to [this guide](https://jannicknijholt.nl/integrate-openrouter-in-n8n/).
- **End-user IDs: Not ready for prime time**: Members are discussing the ability to submit optional **end-user IDs** to prevent abuse and improve moderation, a feature found [here](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user).
   - The discussion points out that **metrics are not yet available** for this feature, with one member stating, *`eventually` = metrics not available yet*.
- **DeepSeek showdown: Is it the best provider?**: Members debated the best provider for **DeepSeek**, with some preferring **Parasail** due to trust and consistent performance, despite a higher cost of **$5**, whereas others favored **DeepSeek** itself, citing lower cost, caching strategy, and direct model implementation.
   - One member noted concerns about crowded servers, Chinese location, and slow speeds with DeepSeek's official API, and another reported that **Deepseek** has issues with `max_tokens` and enforcement of an **8K max** on non-reasoning output tokens, which was upgraded with **R1** to 64k.
- **Chess benchmarks are back?**: Members discussed chess benchmarks and the surprising performance of `gpt-3.5-turbo-instruct`, an *instruct* model trained on chess data, with one member linking to research indicating **chess training improves problem-solving** ([https://arxiv.org/pdf/2312.09390#page=29](https://arxiv.org/pdf/2312.09390#page=29)).
   - Another member referred to an article on the topic (["https://dynomight.net/more-chess/"](https://dynomight.net/more-chess/)), noting that **RLHF can degrade performance** and that *gpt4-base* (pre-RLHF) performs better than *4o* in chess.
- **Zero Logging... or not**: Members were asking about **Parasail's prompt logging policy** ([https://www.parasail.io/legal/privacy-policy](https://www.parasail.io/legal/privacy-policy)), which claims **zero-logging** for serverless and dedicated versions.
   - However, OpenRouter's documentation states that *Prompts are retained for an unknown period*, leading to confusion and requests for clarification from OpenRouter team members.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1378477201288073257)** (156 messages🔥🔥): 

> `Manus AI student perks, School environment, OpenManus affiliation, Deploying Manus-generated sites` 


- **AI Workcation Impact on Sacred Land**: A member reported their experience using Manus to traverse the [Kumano Kodo, a sacred road in Japan](https://zenschool.medium.com/spirituality-in-nature-with-ai-agents-at-kumano-kodo-the-impact-of-ai-workcation-in-a-sacred-land-c62baa87bd2f).
   - The user wished they could post images of the art style Manus copied and transformed into a different piece of art.
- **Users Report Scam Links and Account Compromises**: A user reported what they believe is a **compromised bot** spamming weird links and a **Manus fellowship link**.
   - Admins have been requested to remove the potentially malicious content.
- **Student Perks Game-Changer for Manus Users**: A user asked for help with questions about **Manus student perks**, such as changing their email address without losing perks and sending referrals.
   - Another user replied that **student perks** allows usage of a **credit-free environment**, up to **50 knowledge entries**, and access to **high-effort mode**.
- **Clarification on OpenManus Affiliation**: A user asked if **OpenManus** is affiliated with **Manus AI**, noting that their website caused confusion.
   - Other members clarified that **OpenManus is not affiliated** with Manus AI, but is a "free alternative" that doesn't include API pricing.
- **Manus Deployments, Removing the Icon**: A user asked about removing the **Manus icon** from the bottom right corner of deployed, Manus-generated websites.
   - Other members clarified that it's not possible to remove the icon directly, suggesting the user **download the files and deploy them elsewhere**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1378497526025683096)** (27 messages🔥): 

> `Local 70B Model Graphing, DIY Model Storage Needs, Hugging Face Datasets, Vast.ai Pricing, HDD vs. SSD Bottleneck` 


- **70B Model Graphs Locally, Slowly**: A member ran a **70B model** locally to create a graph, noting it was slow and included a [video](https://cdn.discordapp.com/attachments/729741769738158194/1378497525702725824/Screencast_from_2025-05-31_23-39-40.webm?ex=683f745d&is=683e22dd&hm=b87717b28e91071d704dc570fe912054b96cc4b7fec4e5706278f4e0a0df4d3e) of the process.
   - Another member expressed interest in making their own model but lacked storage for training data, remarking *"Kinda wanna make my own model but I don't even have the storage for the training data."
- **DIY Model Requires Storage and Training Data**: Members discussed difficulties in acquiring sufficient **storage** and **compute resources** for training custom models.
   - They pointed out that while training data is readily available on **Hugging Face**, storing the data and the model, along with the necessary compute power, is a major challenge, and streaming training data is an option to reduce storage.
- **Vast.ai provides reasonable pricing for cloud compute**: Members suggested using **vast.ai** for reasonable prices.
   - One member stated, *"I use vast"*.
- **HDD speeds not a bottleneck for training**: A member suggested that **HDD speed** is likely not a major bottleneck during training, proposing an intermediary **RAM cache** as a buffer.
   - Members noted that HDDs have been cheap for a while.
- **Natural scientist volunteers for interpretability projects**: A trained natural scientist with ML experience volunteered for projects, particularly those involving **interpretability**.
   - They mentioned experience with reinforcement learning, genetic algorithms, and deep learning, primarily coding in Python with openness to learning other languages.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1378470069159329912)** (102 messages🔥🔥): 

> `Incentivizing agent reasoning, Token dropping in MoEs, Continual learning for RL LLM agents, Noise activations for continual learning, Variable length sequence infilling using discrete diffusion` 


- **Reasoning steps speed up agent training**: Training an agent to rule out wrong answers quickly can be achieved by incentivizing the agent to output reasoning steps during CoT which decrease the likelihood of incorrect answers, as suggested in [this paper](https://arxiv.org/abs/2505.05755) for text-diffusion.
- **Token dropping effect in MoEs detailed**: While discussing token dropping in MoEs, it was highlighted that small amounts of token dropping can act as regularization, improving generalization, with [routing collapse](https://arxiv.org/pdf/2409.12517) being the main issue due to large amounts of tokens dropping.
- **KL penalty paper aids continual learning**: When seeking advice for continual learning for an RL LLM agent, it was suggested to look at a paper about doing KL clipping more precisely since you're reducing entropy as you RL, with [this paper](https://arxiv.org/abs/2505.22617) and [this paper](https://arxiv.org/abs/2505.24864) recommended.
- **Noise destroys shallow features for better learning**: It was proposed that adding noise to activations during the forward pass, then applying updates to the original weights without noise, could force the optimizer to dig deeper and destroy shallow features learned during fine-tuning, referencing [this paper on code backdoors](https://arxiv.org/abs/2502.17424).
- **Ordering words diffusion adjacent?**: In discussion of creating positional encodings from bag of words, it was stated that bag of words sounds very similar to diffusion, referencing [this paper about sentence ordering](https://arxiv.org/abs/1607.06952).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1379191707274448896)** (2 messages): 

> `Low Dimensional Manifolds, Data Generating Process, Quotient Out Regularities` 


- **Neural Networks Form Low Dimensional Manifolds**: A member posited that neural networks trained on a low dimensional manifold of natural inputs automatically correspond to some low-dimensional manifold of activations embedded in the high-dimensional space of possible forward passes.
   - They further suggested that knowing the [data generating process](https://en.wikipedia.org/wiki/Data_generation) and sampling the activations for different inputs allows building a picture of what those manifolds look like.
- **Quotienting Regularities in Model Behavior**: The member proposed comparing the manifold of activations against the manifold of inputs to *"quotient out"* the regularities of the dataset.
   - The goal is to obtain a manifold of just the regularities in the model's behavior that are imposed by the weights, questioning methods and potential limitations.
- **Input Manifold as Submanifold of Forward Passes**: The member noted that the input manifold can be treated as a submanifold inside the manifold of forward passes.
   - They were *"not sure how that helps,"* seeking insights into how this perspective might aid in understanding model behavior.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1378803024981528586)** (24 messages🔥): 

> `Hugging Face chunked prefill, RWKV model addition, lm-evaluation-harness bugs, lm-evaluation-harness documentation, max_seq_lengths cmdline argument` 


- **HF Chunked Prefill Speeds Up Long Context Benchmarks**: HF now supports **chunked prefill**, which is super useful for running long context benchmarks with lm_eval; however, setting `prefill_chunk_size` in `generation_config.json` doesn't work, but works when put into the call to `self.model.generate` in `lm_eval/models/huggingface.py`.
   - It was described as a *game changer* because it prevents OOM errors when running ruler and is pretty tough to run long context benchmarks without it if you're actually using a long context.
- **PR Fixes Bugs in lm-evaluation-harness**: A member has a PR to fix some bugs: [https://github.com/EleutherAI/lm-evaluation-harness/pull/2983](https://github.com/EleutherAI/lm-evaluation-harness/pull/2983) for running **longbench**.
   - Some tasks are much better, but they couldn't reproduce all; they plan to merge it and add a warning.
- **DEFAULT_SEQ_LENGTHS on Cmdline**: A member asks about a way to pass `DEFAULT_SEQ_LENGTHS` on cmdline.
   - Another member responded with a link to the relevant section in the lm-evaluation-harness repo: [https://github.com/EleutherAI/lm-evaluation-harness/blob/8bc4afff22e73995883de41018388428e39f8a92/lm_eval/__main__.py#L301](https://github.com/EleutherAI/lm-evaluation-harness/blob/8bc4afff22e73995883de41018388428e39f8a92/lm_eval/__main__.py#L301).
- **Ruler Tasks Specific Args**: A member links to the readmes for ruler tasks: [https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler).
   - They mention that the public API for task specific args may not be as clear yet, and perhaps they should remove the default length and require the user to specify it.
- **Sequence Length Shenanigans**: When adding `--metadata='{"max_seq_lengths":[36864]}'`, a member got both that AND the default one, but the default size gave a result of -1; they also added `--model_args=pretrained=...,max_length=40000` but somehow got a warning about exceeding the model's predefined maximum length (**32768**).
   - They noted that they'd edited the model config to have more rope entries and didn't see anywhere else that **32768** is specified.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1378591620538306660)** (69 messages🔥🔥): 

> `AiderBench Dependencies, DeepSeek Agent, Opus Testing, Mac M3 Performance, Decentralized Inference Network` 


- ****AiderBench** Dependencies Debated**: Members discussed the dependencies required to make **AiderBench** work without **Docker**, with one suggesting it's easier to use **Docker** due to heavy dependencies.
   - Others suggested using a **VPS** as an alternative.
- **DeepSeek Agents**: Members briefly mentioned and inquired about the **DeepSeek** agent, specifically **AgenticSeek**.
   - Another member mentioned **Agent0** as a superior alternative.
- ****Opus** Model Testing Questions**: A member asked if anyone had tested **Opus** yet, in the context of aider models.
   - Another member suggested looking at dedicated benchmarking channels for model performance details.
- ****M3 Mac** Performance Deep Dive**: Members debated why **Mac M3** chips are decent for large models, with one suggesting the built-in neural engine assists with this, reaching **18TOPS** (trillions of operations per second).
   - Another member attributed the performance to the massive memory bandwidth of **Macs**, particularly the **LPDDR5X** memory, up to **540GB/s** on the Max version.
- **Decentralized Inference Network Dream**: A member proposed a decentralized network where **3-5 nodes** complete the same inference in batch, resolving disputes to create a cheap, resilient API for locally runnable **LLMs**.
   - Another member joked that this idea could be turned into a new *crypto*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1378639487982375032)** (28 messages🔥): 

> `Gemini Model Issues, Aider's Automatic Conversation Summaries, Using Aider with Multiple Repos, SCM Files for HTML/CSS, Best Local Model Tips` 


- **Gemini Models Face Task Completion Issues**: A user reported issues using **Gemini models** (gemini/gemini-2.5-pro-preview-05-06 and gemini/gemini-2.5-flash-preview-05-20) with **aider**, struggling to complete more than one task due to errors in folder and filenaming, as well as general instability.
   - They noted that while the coding results were good, the models didn't seem to integrate well with aider.
- **Aider Summarizes Conversations Automatically**: **Aider** automatically summarizes conversation history, aiding in context management as stated by a member.
   - To provide **git commit** and **git diff** views to Aider, use the `/run git diff ...` command, which will then prompt you to add it to the chat.
- **Read-Only Access Solves Multiple Repo Problems**: A member suggested using the `/read-only` command in **aider** to access files from multiple repositories, as symlinks are not followed.
   - For example, `/read-only /Users/username/Downloads/some_random_file.md` allows read-only access to files outside the current repository.
- **SCM Files Sought for HTML/CSS Editing**: A user is seeking **SCM files for HTML/CSS** to improve **aider's** performance when editing these file types.
   - They believe that the poor performance is due to context length issues, even with a **1M context LLM**.
- **Devstral should be used as Local Model**: A user sought advice on the best local model to run, given **4x3090s**, **256GB RAM**, **~100k context window**, and tasks involving edits and fixes on existing **Rust** and **Typescript** code.
   - The user was recommended to try **Devstral**, with a member noting that some versions of the new **R1** would probably deserve a test as well.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1379195123321274429)** (3 messages): 

> `Modular Hackathon, GPU programming workshop, Mojo kernels, MAX Graph model architectures, PyTorch custom ops` 


- **Modular Hosts Hackathon**: Modular is hosting **another hackathon** open to virtual participation, running over a weekend, focused on Mojo kernels, MAX Graph model architectures, and PyTorch custom ops. [Check out the details!](https://lu.ma/modular-hack-weekend)
- **GPU Workshop to kick off Hackathon**: To kick off the weekend, Modular will host a **GPU programming workshop** (in-person and virtual via livestream) at their office in Los Altos.
- **New Mojo User Sees Improvements!**: A recent CS graduate, after watching a **Fireship video** and trying Mojo on basic ML models, reports seeing *improvement* and is excited to optimize their ML code.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1378463111178227712)** (77 messages🔥🔥): 

> `Type Checking in Mojo, Mojo and godbolt.org, Copyable and ExplicitlyCopyable Traits, Profiling Mojo, C Bindings Generator for Mojo` 


- **`_type_is_eq` limitations discussed**: Members discussed the use of `_type_is_eq` to check types, noting that it can't check pointer types with arbitrary pointee types, while [type name reflection](https://example.com/reflection) can help at least check for pointers.
   - Some consider this approach *"cursed"*, but others like that it doesn't have a run-time cost compared to C++'s RTTI.
- **Reflection API vs Trait System Improvements**: The possibility of a reflection API was brought up, and its usefulness highlighted (allows things like *"traits which use compile time information to construct serializers"*).
   - A question was raised if this is a higher priority than trait/type system improvements.
- **`Copyable` and `ExplicitlyCopyable` traits compared**: There was discussion on the purpose of a type conforming to both `Copyable` and `ExplicitlyCopyable` traits, with an example given of a **100+ GB ML model** being better moved than copied.
   - It was also noted that implementing the `Copyable` trait informs the compiler when it can perform implicit copies.
- **Profiling Mojo Code Guidance**: Members discussed profiling Mojo code, noting that tools compatible with **C++** or **Rust** should generally work, such as `perf` with `flamegraph`.
   - It was suggested that CPU vendor's HPC profilers (Intel VTune, AMD uProf, ARM Streamline) can offer detailed microarchitecture insights for optimization.
- **C-to-Mojo Bindings Generator in the Works**: A member is working on a **C-to-Mojo bindings generator**, aiming to handle most cases except *"wildly horrible packed structs"* and potentially `restrict` and pragmas affecting calling conventions.
   - They suggested using a `pixi.toml` file to indicate dependencies for a Mojo GUI project and raised concerns about copying components instead of borrowing in certain parts of the code.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1378467915128635422)** (73 messages🔥🔥): 

> `MCP with Claude Desktop, MCP Transports, Dynamic Tool Registration in MCP, MCP Client Implementations, Elicitations support in MCP Clients` 


- **MCP Quickstart: Claude and Server Setup**: A member created their first MCP server using the [MCP docs](https://modelcontextprotocol.io/quickstart/server#why-claude-for-desktop-and-not-claude-ai) and suggested using **Claude** to learn MCP.
   - They also used **stdio** transport to make it work with the **Claude desktop app**.
- **Injecting data in the system prompt**: Members discussed methods for injecting data directly into the system prompt within MCP, including using the **prompt** option within the fastmcp constructor, which allows specifying information to be added to the system prompt.
   - Also mentioned that the client has to explicitly support that feature.
- **MCP client supporting everything**: Members noted the lack of MCP clients fully supporting the specification and primitives.
   - One member stated the [schema.json/.ts](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/fb34d1d6da2287f82cfdf46c1f91b6fb262cdd38/schema/2025-03-26/schema.json) is the closest thing to a full spec but it's hard to follow.
- **Dynamic Tools Registration Issues**: A member reported issues with dynamic tool registration in MCP where newly registered tools are not immediately discoverable or invokable during the same message cycle.
   - They're seeking possible fixes for this issue as the tool is discoverable only after the current chain finishes execution.
- **Claude Desktop not supporting Streamable HTTP MCP server**: A member sought help connecting Claude desktop to a streamable HTTP MCP server.
   - Another member stated that, currently, it is only possible on the web version with a **Claude Max plan**, linking to [a YouTube tutorial](https://www.youtube.com/watch?v=CU6EWD9IJKk).


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1378546523838742579)** (3 messages): 

> `Aura with AppAgentX, Android phone control by voice agent, MCP servers connecting to MCP knowledge store` 


- ****Aura** connects with **AppAgentX****: After *two packs of cigarettes and 14 hours of headache*, a member connected **Aura** with **AppAgentX** to control Android devices using any MCP Client.
   - They plan to release the code after fixing SST and TTS with Realtime from OpenAI, preview code is available on [GitHub](https://github.com/IhateCreatingUserNames2/Aura_AppAgentX/tree/main).
- ****Android Phone Control** Through Voice Agent**: A member's goal is to create an agent that can control the entire **Android phone** through voice.
   - This is achieved by wrapping **Aura's** functions into A2A tools and broadcasting them into an A2A to MCP Gateway (Aira Hub).
- ****MCP Servers** Connect to MCP Knowledge Store**: A member added the ability to connect MCP servers to an **MCP knowledge store** (client host) for finetuning RAG.
   - This was shared via a [LinkedIn post](https://www.linkedin.com/posts/nerdai_mcp-rag-ai-activity-7335292265386463232--h0Y?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1378453913589518359)** (10 messages🔥): 

> `Language settings for Audio Overviews, NotebookLM chat API integration, Using NotebookLM to record lectures, Audio Overview length limitations` 


- ****Language Settings Still Limited for Audio Overviews****: Users are struggling to create **audio overviews in languages other than English**, even when customizing the AI chat to respond in another language; the general setting remains the output language control.
   - A user reported that there's *no easy option to temporarily request podcasts in different output languages* and noted that an export function is currently missing.
- ****No Direct API for Customer Support Chat Yet****: A user asked if **NotebookLM's chat function** could be integrated into other tools via APIs for business client support.
   - A member stated that *NotebookLM is an end-user tool*, suggesting the use of **Google Cloud - Conversational Agents and Dialogflow API** as alternatives for broader audience applications.
- ****NotebookLM Underneath Google APIs!****: A member speculates that **NotebookLM leverages Google APIs** behind its user-friendly interface.
   - Another member shared they introduce their community college students to new tech like **NotebookLM** to supplement lectures, [linking a YouTube playlist of U.S. History and American Gov't lectures](https://www.youtube.com/playlist?list=PLzjEb2_3El48Kix7QZ1dT3z8qTgFfbFnv).
- ****Audio Overview Length Constraints Analyzed****: A user reported that uploading **16 YouTube video links** (20-35 minutes each) only generated a **15-minute audio overview in English**.
   - The same user is also awaiting a fix for **audio podcast summaries in other languages**, as *they're shorter than the English versions, even with the same content and prompts*.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1378502943933988964)** (63 messages🔥🔥): 

> `Video Uploads, Metadata Embeddings, Pro Subscription Audio Podcast Limits, NotebookLM Availability Outside US, Cancelling Pro Features` 


- ****Video Uploads Spinnin' Outta Control****: Users report that after uploading **MP4 sources**, the source item spins endlessly, and they must refresh the app/page to complete the upload.
   - No immediate solution was provided, but it highlights a potential UX issue with video uploads.
- ****Metadata Embeddings: Content Enrichment Quest****: A user inquired about embedding **metadata** into PDFs to improve the quality of content loaded into NotebookLM sources.
   - It's an open question whether this is supported or what the best approach would be.
- ****Pro Subscription Podcast Limits: A User's Lament****: A user expressed frustration over limitations on **audio podcasts** despite having a **Pro subscription**.
   - The user found that the limits did not apply when using another account.
- ****Accessing NotebookLM: Geography Isn't Destiny****: Users discussed accessing NotebookLM from outside the US, with one user confirming it's possible without a VPN.
   - One user suggested that account settings, rather than region, might be the cause of access issues.
- ****NotebookLM Hallucinates Facts: A Bug Report****: A user reported that NotebookLM creates **random, unsourced facts** and then refers to them as if they are in the original sources.
   - It was suggested to report this behavior in the **bugs channel**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1378454294151561440)** (63 messages🔥🔥): 

> `DeepHermes-3 Discord Integration, Demis Hassabis AGI Prediction, Prompt Lookup Decoding for Speedups, Generalized Overfitting in AI, Claude Model Depreciation` 


- ****DeepHermes-3** to grace the **NousResearch Discord****: A member suggested integrating **DeepHermes-3** into the Discord to encourage organic engagement if it can be autonomous.
   - A member then clarified that it is already in a certain channel.
- ****DeepMind's Demis Hassabis** predicts **AGI by 2030****: **Demis Hassabis** gave a fantastic [talk](https://www.youtube.com/watch?v=U3d2OKEibQ4) on what’s to come and his prediction of **AGI by 2030**, give or take.
   - He notes that **DeepMind** has always been on the frontier since their early start in **2010** and never slowing down.
- ****Prompt Lookup Decoding** unlocks significant speedups**: **Prompt lookup decoding** replaces the draft model with simple string matching in the prompt to generate candidate token sequences, resulting in **2x-4x** speedups in input-grounded tasks.
   - The method can be used with any decoder model without model changes or external datastore, and with both greedy and sampling techniques, as [shown on Github](https://github.com/apoorvumang/prompt-lookup-decoding).
- ****Mindcraft Framework** enables **Minecraft gameplay****: A member shared links to the **Mindcraft framework** ([github.com](https://github.com/kolbytn/mindcraft)) and related **Andy models** ([Ollama](https://ollama.com/Sweaterdog/Andy-4) and [HuggingFace](https://huggingface.co/Sweaterdog/Andy-gen)), which are specifically trained to work with the Java version of Minecraft.
   - One of the Andy models, **Andy-4**, is an **8 billion** parameter model trained on a single **RTX 3090** over three weeks, delivering advanced reasoning, multi-step planning, and robust in-game decision-making.
- ****Circuit-Tracer Repo** leads to new **Mechanistic Interpretability** findings**: A solo researcher utilized Anthropic's open-sourced circuit-tracer repo to map concepts and visualize what a model is doing as it answers a query in mechanistic interpretability, with an initial article available on [LinkedIn](https://www.linkedin.com/pulse/advancing-mechanistic-interpretability-interaction-nets-zsihcv).
   - This work focuses on identifying how information flow passes across layers, representing how information is abstracted or condensed.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1378930189358207067)** (1 messages): 

> `MINDcraft, LLM agents collaboration, MineCollab` 


- **MINDcraft & MineCollab Introduced**: A member shared the paper ["MINDcraft: A Platform for Collaborative Embodied Agents"](https://arxiv.org/pdf/2504.17950) introducing **MINDcraft**, a platform built to enable **LLM agents** to control characters in **Minecraft**, and **MineCollab**, a benchmark to test embodied and collaborative reasoning.
   - The study found that the primary bottleneck in effective collaboration for current agents is **efficient natural language communication**, with agent performance dropping as much as **15%** when required to communicate detailed task completion plans.
- **LLM Agents Struggle with Multi-Agent Collaboration**: The paper concludes that existing **LLM agents** are ill-optimized for **multi-agent collaboration**, especially in embodied scenarios.
   - It highlights the need to employ methods beyond in-context and imitation learning, as current state-of-the-art agents face performance drops when communicating detailed task completion plans.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1378733886334570699)** (6 messages): 

> `System Prompt Learning, LLM Scribe Tool, Robotics Mouse` 


- ****AirPods** Studied by **Apple**?**: A [Perplexity link](https://www.perplexity.ai/page/apple-study-shows-airpods-coul-7_q_Jgn0ROGQmIqFP_0FHA) was shared regarding an **Apple** study on **AirPods**.
   - The original Discord message lacked context, leaving the study's specifics a mystery.
- ****SPL**: LLMs get better over time**: **System Prompt Learning (SPL)** teaches LLMs to learn problem-solving strategies, improving performance by **8.6%** on Arena Hard and **6.67%** on AIME24, as detailed in a [Hugging Face blog post](https://huggingface.co/blog/codelion/system-prompt-learning).
   - This open-source plugin in *optillm*, inspired by [Karpathy's idea](https://x.com/karpathy/status/1921368644069765486), works with any OpenAI-compatible API by adding the `spl-` prefix.
- **Mouse with a Mouse**: A [YouTube video](https://m.youtube.com/watch?v=gQidYj-AKaA) of a **robotics mouse** was shared, with claims of *pretty good robotics* at *airesearch.js.org*.
   - A member offered to integrate features like scraping and data extraction.
- ****LLM Scribe** is here to help**: A member introduced **LLM Scribe**, a tool for creating handwritten datasets for fine-tuning, with features like multi-format export, autosaving, multi-turn creation, token counters, goal tracking, and custom fields, as seen in [this youtube demo](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s).
   - The tool is available on [Hugging Face Spaces](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) and for [full version purchase](https://kryptive.gumroad.com/l/gvyqep).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1378930189358207067)** (1 messages): 

> `MINDcraft platform, LLMs adaptive collaboration, embodied reasoning tasks, MineCollab benchmark, natural language communication` 


- **MINDcraft Platform Enables LLM Minecraft Collabs**: A member shared a link to [MINDcraft](https://arxiv.org/pdf/2504.17950), an easily extensible platform built to enable **LLM agents** to control characters in the open-world game of **Minecraft**.
   - They also shared a link to the [MineCollab benchmark](https://mindcraft-minecollab.github.io/) to test the different dimensions of embodied and collaborative reasoning.
- **LLMs' Natural Language Skills Bottleneck Collabs**: An experimental study found that the primary bottleneck in collaborating effectively for current state-of-the-art agents is efficient **natural language communication**.
   - Agent performance dropped as much as **15%** when they are required to communicate detailed task completion plans, indicating LLM agents are ill-optimized for multi-agent collaboration, especially in embodied scenarios.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1378541063836405911)** (53 messages🔥): 

> `Jason's Nitter Post, EleutherAI's Common-Pile Dataset and Comma 0.1 Model, Kontext Chat for Image Editing, NYT licenses content to Amazon for AI training, Karpathy's Guide to Using ChatGPT Versions` 


- ****Jason's X-Ware Post Surfaces on Nitter****: User @agikoala (Jason) made a post on **June 1, 2025**, featuring an image, garnering **2 likes** and **18 shares** which was relayed via [Nitter](https://xcancel.com/agikoala/status/1929048742516162940?s=46).
- ****EleutherAI Unleashes Common-Pile and Comma 0.1****: EleutherAI released **Common-Pile**, an **8TB** libre dataset, a filtered version, and **Comma 0.1**, a **7B** base model trained on the filtered dataset; architecture matches **Llama 3** and trained using **lingua** on **64 Nvidia H100 GPUs**; source available on [HuggingFace](https://huggingface.co/common-pile/comma-v0.1).
   - It was noted as *the closest to a full stack FOSS model* seen yet, though its tweets got removed for unknown reasons.
- ****Replicate Debuts Kontext Chat: Edit Images with Words****: Replicate introduced **Kontext Chat**, an open-source application for editing images via conversational commands, built with **Hono** and hosted on **Cloudflare Workers**, designed as a developer starting point, announced on [X](https://xcancel.com/replicate/status/1929160560295506417?s=46).
- ****NYT Sells Out, Licenses Content to Amazon for AI****: The New York Times and Amazon inked a deal for licensing NYT's content for AI training, including Amazon's foundation models which was [announced on Twitter](https://xcancel.com/natolambert/status/1929175745596620968?s=46).
   - Community members speculated that this move suggests **NYT's lawsuit against OpenAI** was primarily for securing payment rather than a moral stance.
- ****Karpathy's ChatGPT Model Menu: Pick Your Poison****: Andrej Karpathy shared his guide to effectively using different ChatGPT versions, recommending **'o3'** for important tasks, **'4o'** as a daily driver, **'4.1'** for coding, and **'Deep Research'** (based on o3) for deep dives, announced on [X](https://xcancel.com/karpathy/status/1929597620969951434).


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1378593712107421756)** (14 messages🔥): 

> `AIE World's Fair 2025, Live Production AI Bot Collaboration, Bug Reporting System for AIE` 


- ****AIE World's Fair**: Circle Your Calendars!**: The **AIE World's Fair** will be held from **June 3-5, 2025**, in San Francisco; check [ai.engineer/#events](https://www.ai.engineer/#events) for side events that don't require a ticket.
- **Discord Community Builds **Live AI Bot****: A member announced that a **live production AI bot** was collaboratively built within the Discord community, with a new gen UI framework shared and then shipped to the AIE website today.
   - The bot is available at [ai.engineer/ai](https://ai.engineer/ai) and the discussion thread is linked [here](https://discord.com/channels/822583790773862470/1378055295401459813/1378137211995689061).
- ****Bug Reporting** System Desired for AIE Website**: With multiple **bug reports** coming in, a member suggested establishing a more streamlined **feedback loop** for the AIE website to facilitate self-improvement.
   - Another member volunteered to help with **bug reports**.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1378994198098870312)** (2 messages): 

> `TPS benchmarks, Regression testing, Performance Metrics` 


- **Regression Testing Paves the Way for Performance Stability**: A member inquired about checks to benchmark **TPS** with different configurations, ensuring no performance or compatibility regressions occur for fixed configurations.
   - Another member confirmed the upcoming addition of **regression testing** for both performance and evaluation metrics, noting a current **PR** addressing this.
- **TPS Benchmarking in Focus**: The discussion highlighted the importance of **TPS** (Transactions Per Second) benchmarks across varying configurations.
   - The goal is to proactively identify and prevent performance regressions, ensuring consistent performance levels.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1378897534231253042)** (45 messages🔥): 

> `LLaMA-3 70B Fine-Tuning, DP vs TP Performance, FP8 vs BF16 Comparison, Compile impact on TPS, Loss Parallel implementation` 


- ****LLaMA-3** Fine-Tuning **Golden Paths** Underway**: A member is developing internal "golden paths" for fine-tuning different models, starting with **LLaMA-3 70B**, with short and long context lengths, and initial insights from **8k context** length experiments.
   - The member aims to share findings on Data Parallelism (DP) vs Tensor Parallelism (TP), FP8 vs BF16, and the impact of Compile vs No Compile on performance.
- ****FP8 Compile** boosts **TPS**, reserves extra Memory**: Experiments show that **FP8** has the lowest **TPS** with compile disabled, but the highest **TPS** by far when compile is enabled.
   - It was observed that **FP8 + compile** has the lowest active peak memory but the highest peak reserved memory and a member suggested to try running the "high reserved memory" scenarios with `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`.
- ****Higher TP** Lowers **TPS**, Needs More Memory**: Experiments indicated that increasing tensor parallelism (TP) leads to decreased throughput (**TPS**), likely due to the cost of matmul collectives exceeding those for model parameters in pure FSDP, and **FP8** doesn't seem to mitigate this cost.
   - It was also observed that higher TP results in higher peak active memory, potentially because expensive layers like outputs to compute loss are replicated, causing higher usage with larger batch sizes per device and one of the members suggested Loss Parallel implementation for very large contexts.
- **"Loss Parallel" could lower memory**: A member suggested to implement "loss parallel" which could potentially lower memory usage.
   - Another member clarified that only the output of the layer is replicated, and loss parallel would improve memory performance but even without it, the memory usage should be the same or less than what we see for FSDP.
- **Universal **HF Tokenizer** Support Awaits Review**: Work is underway to land universal [Hugging Face tokenizer](https://github.com/huggingface/transformers) support.
   - A member has added unit tests and is awaiting review, but is ready to swap priorities if Tool Calling Support is needed ASAP.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1378866338788802660)** (45 messages🔥): 

> `Claude Code analysis, DSPy talks at AI Engineering and Databricks DAIS, DSPy 3.0 release in June, DSPy and DARPA's Advanced Research Concepts lab, Agentic orchestration` 


- **Analyzing Claude Code Internals via Cleanroom Design**: Members discussed [an analysis of Claude Code](https://southbridge-research.notion.site/claude-code-an-agentic-cleanroom-analysis) and whether it was open source, with the conversation clarifying that the analysis used **cleanroom design** principles to avoid direct access to proprietary techniques.
   - Cleanroom engineering benefits from not haccing access to the source: [https://en.wikipedia.org/wiki/Clean-room_design](https://en.wikipedia.org/wiki/Clean-room_design). *The term implies that the design team works in an environment that is 'clean' or demonstrably uncontaminated by any knowledge of the proprietary techniques used by the competitor.*
- **DSPy Talks Soliciting Community Input**: A member is preparing DSPy talks for AI Engineering and Databricks DAIS, seeking community input on topics to cover and use cases to highlight.
   - Specific areas of interest include **basic DSPy concepts (signatures, programs, common optimizers), use cases (structured outputs from PDFs and images), and advanced topics (RL, datasets, advanced optimizers like MiPro)**.
- **DSPy Used in DARPA Project**: DSPy was used for **DARPA's Advanced Research Concepts lab** to build a solution for the 'Collaborative Knowledge Curation' interest area, which is now being spun out into a company.
   - Someone reacted to the message with "what [gif](https://tenor.com/view/i-said-what-i-said-nene-leakes-real-housewives-atlanta-rhoa-gif-16456054)".
- **DSPy 3.0 Gearing Up for June Release**: The community is gearing up for a **3.0 release in June**.
   - Members are also interested in how to migrate existing pipelines to DSPy, when DSPy is most effective, how to use it for synthetic data generation, and how it differs from other agentic solutions.
- **DSPy's Stance on Agent Frameworks**: There's a discussion about the potential for building an **agent framework on top of DSPy**, with emphasis on first-class environments, handling rewards, and leveraging them for online learning via optimizers.
   - One member mentioned [Agenspy](https://github.com/SuperagenticAI/Agenspy) and another is sketching out an implementation rn built on top of claude code


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1379211955646234766)** (1 messages): 

> `Gradio Agents x MCP Hackathon` 


- ****Gradio Agents x MCP Hackathon** is here!**: Registration for the **Gradio Agents x MCP Hackathon** is open [here](https://huggingface.co/Agents-MCP-Hackathon).
   - The hackathon week is ongoing!
- **Tune into **Gradio Agents x MCP Livestream****: The **Gradio Agents x MCP Livestream** will be broadcasted on YouTube [here](https://discord.com/events/1059199217496772688/1379207318700294245) on June 3rd.
   - Tune in to learn more about the agents and the hackathon.
- **Join the **Hackathon Office Hours****: The **Hackathon Office Hours** session for hackathon participants in the HuggingFace Discord server will be held on Wednesday, June 4th.
   - Check the events calendar for time slot updates.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1379151790091796584)** (3 messages): 

> `E-Library-Agent, Gradio Agents & MCP Hackathon, Scaling Agents in Finance workshop` 


- **E-Library-Agent: Remember Everything You Read**: A new open-source project from @itsclelia called **E-Library-Agent** helps users progressively build their digital library by ingesting [data from various sources](https://t.co/CgPF3uKbBJ).
- **Gradio Agents & MCP Hackathon Kicks Off**: The **Gradio Agents & MCP Hackathon** kicks off tomorrow with [$16.5k in prizes](https://t.co/TBAPFsNWU1), [$900k in credits](https://t.co/TBAPFsNWU1), and **3 Tracks**: *MCP Tool/Server*, *Custom Components for Agents*, and *Agentic Demo Showcase*.
   - A kickoff [livestream](https://t.co/FzLmzviwRz) will be hosted tomorrow as well.
- **Automate Finance Workflows with Agentic AI**: The full slide deck from the **Scaling Agents in Finance workshop** hosted by @jerryjliu0 last week is now available to [automate document workflows](https://t.co/Crfy50pB4j) with Agentic AI for finance tasks.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1378645167820771398)** (29 messages🔥): 

> `Nested Workflow Event Streaming, Disabling Streaming During Document Indexing, Migrating from OpenAI LLM to vLLM, Tool Visibility Bug in llama-index-llms-google-genai, AI Powered Web Browser` 


- **Nested Workflows**: The best pattern for streaming all events from nested workflows involves having the **parent workflow iterate over the event stream** of the sub-workflow and propagate events back up.
   - This is preferred over passing the parent context into the child workflow, which can lead to composability issues when a child workflow becomes a parent in a different context.
- **Trouble Disabling Streaming During Indexing**: A member was having trouble disabling **streaming** during document indexing when generating summaries using `LangChainLLM` with `AzureChatOpenAI(streaming=False)` and `.complete(prompt)`.
   - Despite setting `streaming=False` and using `.complete()`, the LLM continued to stream tokens, which led the user to question whether **LlamaIndex** or a hidden callback manager was forcing stream behavior.
- **Gemini tools invisible**: After upgrading `llama-index-llms-google-genai` from **0.1.14** to **0.2.0**, tools became invisible to the **Gemini model**.
   - The issue was acknowledged by a member and fixed in [v0.2.1](https://github.com/run-llama/llama_index/pull/18933).
- **Schema Scraper**: A member has made an **AI powered web browser** that can scrape websites for schemas.
   - It creates a series of non-AI actions to extract specific fields, creating reusable strategies and is asking if this is worth it vs getting the site HTML as markdown and parsing data.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1378498530028687521)** (5 messages): 

> `AdamW Optimizer, SFT Training, Probability and statistics` 


- **AdamW Optimizer Defaults Need Patching**: Insights from a [recent paper](https://arxiv.org/abs/2505.21829) suggest **AdamW** performs best when beta1 and beta2 are equal, with **0.95** being a good value.
   - The current [PyTorch defaults](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) for beta1 (**0.9**) and beta2 (**0.999**) are considered suboptimal, prompting a call for a patch submission.
- **SFT Training Expertise Requested**: A member inquired about finding individuals experienced in **Supervised Fine-Tuning (SFT)**, particularly with the *SFTTrainer* class from the Transformer Reinforcement Learning (TRL) library.
- **Probability and statistics musings**: A member shared that *Probability and statistics is all about the musts of may*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1379103240855228436)** (12 messages🔥): 

> `RLVR Measurement, FP8 Training Stability, SwiGLU Activation, Smooth-SwiGLU, LLM Baseline Evaluations` 


- **RLVR Objectivity Challenged**: A member suggested that people may want to believe **RLVR** works well due to its psychological appeal, but *objectively measuring it* has been difficult.
- **FP8 Training Scaled to Trillions**: A discussion was scheduled on [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517), detailing how **FP8 precision** was used to train large language models on up to **2 trillion tokens**.
   - The paper identifies instabilities in FP8 training linked to **SwiGLU activation**, and introduces **Smooth-SwiGLU** to ensure stable training.
- **LLM Gain Claims Questioned**: A [LessWrong post](https://www.lesswrong.com/posts/p8rcMDRwEGeFAzCQS/incorrect-baseline-evalutions-call-into-question-recent-llm) critiques recent LLM papers, arguing that *claimed gains might be caused by too weak baselines*.
- **Dynamic Discord Timestamps**: A member shared a handy tool for creating dynamic Discord timestamps [r.3v.fi/discord-timestamps/](https://r.3v.fi/discord-timestamps/).
- **SwiGLU Activation Blogpost**: A member shared a [blog post](https://jcarlosroldan.com/post/348) to further explain the **SwiGLU activation**.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1378627388224835585)** (1 messages): 

> `GitHub MCP Vulnerability, Invariant Labs Post-Mortem Report, GitHub Security Risks` 


- **MCP GitHub Attack Vector Post-Mortem**: A member shared a [post-mortem report from Invariant Labs](https://invariantlabs.ai/blog/mcp-github-vulnerability) detailing an attack vector targeting **GitHub MCP** to leak private repo content into public PRs.
- **GitHub Security Risks Highlighted**: The discussion emphasized the potential security risks associated with the rapid adoption of **MCPs** on GitHub.
   - A member noted they *'saw this one from a mile away'* regarding vulnerabilities as **MCPs** became more popular.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1378531241141211206)** (11 messages🔥): 

> `xAI's Grok, Google AI Edge Repo` 


- **Grok's Expensive X Integration**: A member linked to a [Grok status](https://x.com/grok/status/1928906427277701214) and questioned why **xAI** would pay so much to be on the **X** platform.
   - They questioned if **X** expects to get so many paid users to make up the **$300M + 50% shared revenue** and also linked to a controversial article, [OpenAI Incel Chatbot Subhuman Men](https://www.citationneeded.news/openai-incel-chatbot-subhuman-men/).
- **Google's AI Edge Gallery Appears!**: A member shared the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) and suggested Google publish it on **F-Droid**.
   - Another member asked if the repo was official and the first member confirmed that it was.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1378448555861348432)** (28 messages🔥): 

> `AgentX submission, Technical appendix submission, Certificate declaration form, Trailblazer certificate, Next MOOC dates` 


- **Video Demo Allowed for AgentX Submission**: A participant inquired if a [video of a working prototype](https://www.example.com/hypothetical-link) would suffice for the AgentX Entrepreneurship track submission in the absence of a live product link.
   - A moderator clarified that a **live demo link is expected**, with the product demo video being a separate requirement, but advised to submit what is available.
- **Technical Appendix Submission Resolved**: A user asked about attaching a **5-page technical appendix** to the submission form.
   - A moderator added a field for the appendix and noted that submissions with the appendix in a different spot would still be accepted.
- **Multiple Team Emails in Certificate Form Clarified**: A participant questioned whether multiple team emails should be separated by commas in the certificate declaration form when they joined multiple teams.
   - It was clarified that including the **primary email of one of the teams** is sufficient.
- **Trailblazer Certificate Criteria Explained**: Participants inquired about the criteria for the **Trailblazer certificate**, specifically regarding articles and X posts and timelines for certificates.
   - A moderator confirmed that completing quizzes, submitting a written article, and posting on X qualifies, and that certificates would be released within the next few weeks, but that [Ninja + Legendary certificates will take longer](https://www.example.com/hypothetical-link).
- **Certificate Declaration Form Confirmation Troubleshooted**: A user reported receiving confirmation emails for some submissions but not for the certificate declaration form.
   - A moderator advised that seeing the **confirmation screen in the browser** is usually sufficient but offered to keep the form open for resubmission as a precaution, and noted that you can only submit the form once.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1378518993484382308)** (12 messages🔥): 

> `AI-generated CUDA-C kernels beating PyTorch, tinygrad meeting #73, unsigned firmware on 7900XTX, multihost changes & p2p transfers` 


- ****CUDA-C Kernels Crush PyTorch****: A [PR](https://github.com/tinygrad/tinygrad/pull/10586) highlights that **AI-generated CUDA-C kernels** in tinygrad are performing close to or even beating expert-optimized production kernels in **PyTorch** without using libraries like CUTLASS and Triton.
   - The kernels achieved **101.3% performance** in **Matmul (FP32)**, **179.9%** in **Conv2D**, **111.8%** in **Softmax**, **484.4%** in **LayerNorm**, and **290.1%** in **Conv2D + ReLU + MaxPool** relative to PyTorch.
- ****tinygrad to Discuss Future Direction****: Meeting #73 is scheduled for **9am Monday San Diego time**, with discussion items including company update, **MLPerf**, **benchmark CI jobs**, **scheduler**, **drivers**, **cloud hashing**, **ONNX**, **WebGPU**, **symbolic Z3**, and other bounties.
   - Agenda to include bounties for **lm_eval**, **AMD_LLVM**, and **cloud** related tasks.
- ****Reverse Engineering AMD GPUs****: A user inquired about the progress and methods attempted to run unsigned firmware on the **7900XTX** GPU using a custom driver, seeking a list of tried approaches.
   - A link to relevant discussion in a different thread about [unsigned firmware](https://discord.com/channels/1068976834382925865/1318495874699231302/1360095679396974706) was shared.
- ****Cloud integration nears Completion****: One of the members reported that they had been busy with irl things, and submitted **multihost changes**, polishing **p2p transfers**, and that they were expecting to finish the cloud stuff either today or tomorow, except speed.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1378769747084181545)** (5 messages): 

> `UOp class, UOp trees, Ops.UNIQUE` 


- **Decoding 'args' in UOp Trees**: A member inquired about the meaning and documentation of 'args' in the **UOp class** and how they're used in **UOp trees**.
   - Another member clarified that there are *no docs*, and the meaning depends on the **Op type**, but it's usually a marker to show that the **buffer uop child** is the *nth* buffer uop created.
- **Ops.UNIQUE Explained**: A member asked what 'arg=19' means in `UOp(Ops.UNIQUE, dtypes.void, arg=19, src=())`.
   - It was explained that for **Ops.BUFFER UOps**, the *arg* gives them unique identities so you can tell which buf is which when looking at graphs.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1378493770479173775)** (13 messages🔥): 

> `GPT4All Extension, Intel Compute, AI models, Model Context Protocol` 


- **Extending GPT4All for Safe PC Interaction**: A community member is working on a project to extend **GPT4All's Mistral-Instruct model** for safe, controlled interaction with a local PC via a secure execution layer.
   - The member seeks assistance in identifying integration points, recommending best practices for safely interpreting model outputs, and potential collaboration, with plans to release the extension as an open-source tool pending permission from the **GPT4All team**.
- **GPT4All support for Intel compute**: A member asked if **GPT4All** will support **Intel compute**.
   - They mentioned their **12GB B580** awaits.
- **Seeking AI Models Aware of Scientific Knowledge**: A member is looking for **AI models** aware of the state of scientific knowledge in various fields to accurately answer questions about medicine, biology, ethology, philosophy, logic, and ethics.
   - One suggestion was to use **LocalDocs for RAG** with tons of textbooks pertaining to relevant info on medicine and biology into LocalDocs for RAG with models from late 2024 or newer.
- **Model Context Protocol and Tool-Calling Capabilities**: A member suggested looking into the **Model Context Protocol (MCP)** or **llama.cpp's tool-calling capabilities** to build upon the GPT4All project.
   - They noted that Nomic developers have not responded to inquiries in recent months, indicating uncertain PR reviews and merges.
- **HighNoonLLM Released**: A link to the **HighNoonLLM** [Github](https://github.com/versoindustries/HighNoonLLM) was released.
   - No further details were given.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1379168810824634511)** (1 messages): 

> `Azure AI Inference SDK, Cohere input types` 


- **Azure AI Inference SDK won't support Cohere input types**: Azure has stated that they will **not support Cohere input types** while using embedding models with [Azure AI Inference SDK](https://github.com/Azure/azure-sdk-for-python/issues/41001#issuecomment-2931978119).
   - A user may respond and ask if they could put a warning in **Azure AI foundry** or in their documentation because they think this is a really subtle issue that other people may run into.
- **Cohere SDK vs Azure for Model Testing**: Although the **Cohere SDK** could be used, the user prefers to use Azure since they are testing multiple models from different providers.
   - This makes it easier to manage and test various models in a unified environment.


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1378740162460123237)** (1 messages): 

> `Cohere Spanish Recipes DPO Dataset, New Open Source Projects` 


- **HuggingFace dataset for Cohere Spanish Recipes surfaces**: A member shared a [HuggingFace dataset](https://huggingface.co/datasets/somosnlp-hackathon-2025/gastronomia-hispana-dpo) for **Cohere Spanish Recipes** using the **Direct Preference Optimization (DPO)** method.
- **Exciting New Open Source Projects Kickoff**: Enthusiastic members are beginning to announce the start of new **open source projects** and are looking for contributors.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1379006838485422252)** (2 messages): 

> `Agentic Frameworks, LLM Grounding` 


- **Elie Joins from Japan!**: Elie Magambo from Rwanda, working in Japan at **Araya Inc**, is learning about **Agentic frameworks** and exploring available tools.
   - Elie is focusing on grounding **LLMs with personalized content**.
- **Community Discord Server Welcomes New Members**: The Cohere Community Discord Server welcomes new members and encourages them to introduce themselves.
   - New members are asked to share their company/industry/university, what they are working on, favorite tech/tools, and what they hope to gain from the community.

