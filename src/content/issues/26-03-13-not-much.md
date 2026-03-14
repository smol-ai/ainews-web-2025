---
id: MjAyNS0x
title: not much happened today
date: '2026-03-10T05:44:39.731046Z'
description: >-
  **MCP tools** remain relevant for deterministic APIs despite ergonomic
  criticisms, with new **web MCP support in Chrome v146** enabling continuous
  browsing agents. Persistent memory is emerging as a key differentiator for
  agents, with IBM improving task completion rates and multi-agent memory framed
  as a computer architecture challenge. Agent UX is evolving towards always-on,
  cross-device operation, exemplified by **Perplexity Computer** on iOS and
  **Claude Code** session management. **Anthropic** released **Opus 4.6 1M
  context** as default with no extra long-context API charges, achieving **78.3%
  on MRCR v2 at 1M tokens**. Sparse attention optimizations like **IndexCache**
  in **DeepSeek Sparse Attention** yield significant speedups on large models
  with minimal code changes.
companies:
  - anthropic
  - ibm
  - perplexity-ai
  - llamaindex
  - deepseek
  - google-chrome
models:
  - opus-4.6
  - glm-5
topics:
  - persistent-memory
  - agent-infrastructure
  - cross-device-synchronization
  - long-context
  - sparse-attention
  - inference-optimization
  - computer-architecture
  - task-completion
  - systems-performance
people:
  - pamelafox
  - tadasayy
  - llama_index
  - bromann
  - dair_ai
  - omarsar0
  - abxxai
  - teknuim
  - bcherny
  - kimmonismus
  - _catwu
  - alexalbert__
  - realyushibai
---


**a quiet day.**

> AI News for 3/12/2026-3/13/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Agent Infrastructure, MCP Friction, and Persistent Memory**

- **The MCP backlash is mostly about ergonomics, not demand**: A good slice of the feed was engineers arguing over whether **MCP** is “dead” or simply over-exposed. [@pamelafox](https://x.com/pamelafox/status/2032315760530665895) joked that “MCP was pronounced dead on Twitter, after mass exposure to curl,” while [@tadasayy](https://x.com/tadasayy/status/2032327227472589282) countered that usage is still booming. The more substantive take came from [@llama_index](https://x.com/llama_index/status/2032487366129233950): **MCP tools** are strong when you need deterministic, centrally maintained APIs and rapidly changing ground truth; **skills** are lighter-weight local natural-language procedures but more failure-prone. Relatedly, [@bromann](https://x.com/bromann/status/2032554703863820325) pointed to new **web MCP support in Chrome v146**, showing a LangChain Deep Agent that continuously browses X and compiles a daily summary.

- **Memory is becoming the differentiator for agents**: The most technically interesting agent thread was around **persistent memory** and self-improvement. [@dair_ai](https://x.com/dair_ai/status/2032459951306866714) highlighted IBM work on extracting reusable strategy/recovery/optimization tips from agent trajectories, improving AppWorld **task completion from 69.6% to 73.2%** and **scenario goals from 50.0% to 64.3%**, with the biggest gains on hard tasks. In parallel, [@omarsar0](https://x.com/omarsar0/status/2032465974159618452) summarized a paper reframing multi-agent memory as a **computer architecture problem**, with cache/memory hierarchy, coherence, and access-control issues rather than “just more context.” This maps directly onto product work like **Hermes Agent**, which multiple tweets described as a self-hostable agent that retains skills and user-specific memory over time ([overview via @abxxai](https://x.com/abxxai/status/2032463531627663540), [demo via @Teknium](https://x.com/Teknium/status/2032435764588646839)).

- **Agent UX is moving to always-on, cross-device operation**: Several launches pushed agents closer to “personal computer as orchestrator.” **Perplexity Computer** rolled out to iOS with cross-device synchronization, letting users start or manage a browser-computer task from phone or desktop ([announcement](https://x.com/perplexity_ai/status/2032494752642568417), [Arav follow-up](https://x.com/AravSrinivas/status/2032495364088238147)). [@bcherny](https://x.com/bcherny/status/2032578639276159438) showed the analogous flow for **Claude Code**, starting sessions on a laptop from a phone. Genspark’s **Claw** was framed similarly as an “AI employee” with a persistent cloud computer ([summary by @kimmonismus](https://x.com/kimmonismus/status/2032501165154332711)). The common pattern: persistent session state, remote execution, and orchestration across many models/tools.

**Inference, Long Context, and Systems Performance**

- **Anthropic quietly shipped one of the bigger infra-relevant updates of the week**: **Opus 4.6 1M context** became the default for Max/Team/Enterprise users ([via @_catwu](https://x.com/_catwu/status/2032515975556509827)), and Anthropic removed the API’s extra charge for long context while also dropping the beta header requirement and expanding media limits to **600 images/PDF pages per request** ([details from @alexalbert__](https://x.com/alexalbert__/status/2032522722551689363)). The most notable metric attached was **78.3% on MRCR v2 at 1M tokens**, called out by multiple observers as a new frontier long-context high watermark ([e.g. @kimmonismus](https://x.com/kimmonismus/status/2032531949571477517)).

- **Sparse attention optimization is still yielding meaningful wins**: A standout systems thread from [@realYushiBai](https://x.com/realYushiBai/status/2032299919999189107) introduced **IndexCache**, which reuses sparse-attention index information across layers in **DeepSeek Sparse Attention**. Reported gains: roughly **1.2× end-to-end speedup on GLM-5 (744B)** with matching quality, and on a 30B-scale experimental model at **200K context**, **1.82× prefill** and **1.48× decode** after removing **75% of indexers**. This was notable because it targets a production-scale sparse-attention stack with “minimal code change,” which is exactly the kind of practical optimization labs care about now.

- **KV/cache and serving optimizations are broadening beyond autoregressive LLMs**: [@RisingSayak](https://x.com/RisingSayak/status/2032427185345273928) highlighted **Black Forest Labs’ Klein KV**, which injects cached reference-image KVs into later DiT denoising steps for multi-reference editing, claiming up to **2.5× speedups**. On the infra side, [@satyanadella](https://x.com/satyanadella/status/2032515189086761005) said Microsoft is the first cloud validating an **NVIDIA Vera Rubin NVL72** system, while [@LambdaAPI](https://x.com/LambdaAPI/status/2032427317696602575) pushed the “bare metal over hypervisor” angle for Rubin-era clusters. [@__tinygrad__](https://x.com/__tinygrad__/status/2032429289443053705) added a more radical endpoint: an “exabox” exposed as a single giant Python-driven GPU in 2027.

**Post-Training, RL Alternatives, and Evaluation Research**

- **A provocative post-training result: random Gaussian search can rival RL fine-tuning**: The most-discussed research claim was **RandOpt / Neural Thickets** from MIT-adjacent authors, shared by [@yule_gan](https://x.com/yule_gan/status/2032482266773926281) and [@phillip_isola](https://x.com/phillip_isola/status/2032483868603822402). The claim: by adding Gaussian noise to pretrained model weights and ensembling, one can reach performance **comparable to or better than GRPO/PPO** on reasoning, coding, writing, chemistry, and VLM tasks. Their explanation is that large pretrained models live in local neighborhoods dense with useful task specialists—“**neural thickets**”—making post-training much easier than standard optimization intuitions suggest.

- **Generic-data replay and pre-pre-training are getting renewed attention**: [@TheTuringPost](https://x.com/TheTuringPost/status/2032441644143055316) summarized Stanford work on **generic data replay**, reporting **1.87× improvement during fine-tuning** and **2.06× during mid-training**, with concrete downstream gains like **+4.5%** on agentic web navigation and **+2%** on Basque QA. Separate chatter around “pre-pre-training” suggested the community is revisiting staging/mixture design earlier in the training pipeline, not just post-training tricks ([commentary from @teortaxesTex](https://x.com/teortaxesTex/status/2032611773308641493)).

- **Evaluation remains a bottleneck, especially for truthfulness and search strategy**: [@i](https://x.com/i/status/2032458037823483953) shared **BrokenArXiv**, where even **GPT-5.4** rejected only **40%** of perturbed false mathematical statements from recent papers. [@paul_cal](https://x.com/paul_cal/status/2032526200766103944) argued this gives GPT-5.4 an edge over Claude on proof-verification-style “bullshit detection,” even if other truthfulness benchmarks disagree. For retrieval/search, **MADQA** found agents near human answer accuracy by using brute-force search rather than strategic navigation over documents, leaving about a **20% gap to oracle performance** ([via @HuggingPapers](https://x.com/HuggingPapers/status/2032490352502792228)).

**Open Source Releases, Datasets, and Reproducibility**

- **OpenFold3’s new preview is unusually complete by frontier biology standards**: [@MoAlQuraishi](https://x.com/MoAlQuraishi/status/2032471033760903511) announced **OpenFold3 preview 2**, saying it closes much of the gap to AlphaFold3 across modalities while releasing not just weights but also **training sets and configs**, making it “the only current AF3-based model that is functionally trainable & reproducible from scratch.” That reproducibility claim is the key point: many “open” biology releases still stop well short of end-to-end re-trainability.

- **Speech data for underrepresented languages got a meaningful boost**: [@osanseviero](https://x.com/osanseviero/status/2032452729059045881) announced **WAXAL**, an open multilingual speech dataset covering **17 African languages for TTS** and **19 for ASR**, later described by [@GoogleResearch](https://x.com/GoogleResearch/status/2032482132619387348) as **2,400+ hours** spanning **27 Sub-Saharan languages** and **100M+ speakers**. The exact language/task counts differed between posts, but both positioned WAXAL as a rare, community-rooted resource for African voice AI.

- **Open-source sentiment around training data is hardening in favor of permissive reuse**: The strongest statement came from [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2032460578669691171), who argued that open-source code is a gift whose value is **magnified by AI training**, not undermined by it. [@giffmana](https://x.com/giffmana/status/2032528855215276282) and [@perrymetzger](https://x.com/perrymetzger/status/2032543203795284218) echoed that view. The most nuanced counterpoint was [@wightmanr](https://x.com/wightmanr/status/2032555294296084755), who argued that coding agents may bypass attribution and licensing expectations in ways that could demotivate maintainers, suggesting a protocol for agent compliance could become important.

**Developer Tooling, Coding Agents, and Research Automation**

- **Coding-agent workflows are getting more autonomous and more opinionated**: There were many examples of engineers moving from “copilot” to **multi-agent software factories**. [@matvelloso](https://x.com/matvelloso/status/2032502379694932178) described a setup with **5 agents** doing code review/test/security/perf work and **2 more** merging PRs and running regression checks. [@swyx](https://x.com/swyx/status/2032464562214293776) compressed the trend to “**Your Code is your Infra**,” while [@gokulr](https://x.com/gokulr/status/2032304707398746584) and [@matanSF](https://x.com/matanSF/status/2032561391408918797) pointed to **FactoryAI** as an increasingly common “software factory” layer.

- **Autonomous research is becoming a product category, but not a new idea**: Karpathy’s **autoresearch** and related hackathons drew significant attention, but several tweets noted the conceptual overlap with older systems like **DSPy**, **GEPA**, and Bayesian optimization pipelines. The most practical pointer was [@dbreunig](https://x.com/dbreunig/status/2032313870233321956) recommending **optimize_anything** for people interested in this style of iterative self-improvement. Together AI also shipped **Open Deep Research v2**, open-sourcing its app, eval dataset, code, and blog ([launch](https://x.com/togethercompute/status/2032524281461223614)).

**Top tweets (by engagement)**

- **xAI recruiting reset**: [@elonmusk](https://x.com/elonmusk/status/2032341856944865487) said xAI is reviewing historical interview pipelines and re-contacting promising candidates previously rejected, after acknowledging many strong people were missed.
- **Claude’s chart UI**: [@crystalsssup](https://x.com/crystalsssup/status/2032334906517536969) posted a highly engaged reaction to Claude’s new **interactive chart** UX.
- **Perplexity Computer on mobile**: [@perplexity_ai](https://x.com/perplexity_ai/status/2032494752642568417) launched cross-device **Computer** access on iOS, one of the clearest productizations of remote agent execution this week.
- **Microsoft validates Rubin NVL72**: [@satyanadella](https://x.com/satyanadella/status/2032515189086761005) announced Azure as the first cloud validating **NVIDIA Vera Rubin NVL72**.
- **Nous / Hermes momentum**: Hermes Agent and its memory-centric framing generated wide discussion via [@Teknium](https://x.com/Teknium/status/2032435764588646839) and others, reflecting strong interest in self-hosted, improving agent harnesses.



---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. OmniCoder-9B Model Launch and Performance

  - **[OmniCoder-9B | 9B coding agent fine-tuned on 425K agentic trajectories](https://www.reddit.com/r/LocalLLaMA/comments/1rs6td4/omnicoder9b_9b_coding_agent_finetuned_on_425k/)** (Activity: 781): ****OmniCoder-9B** is a 9-billion parameter coding agent developed by [Tesslate](https://tesslate.com/), fine-tuned on the [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) architecture, which uses Gated Delta Networks interleaved with standard attention. It was trained on `425,000+` curated agentic coding trajectories, including data from models like Claude Opus 4.6 and GPT-5.4, focusing on real-world software engineering tasks. The model features a `262,144` token context window, extensible to `1M+`, and demonstrates strong error recovery and reasoning capabilities, such as responding to LSP diagnostics and using minimal edit diffs. It is released under the Apache 2.0 license, with fully open weights.** Commenters highlight the impressive capabilities of the Qwen3.5-9B architecture, noting its ability to perform tasks typically requiring much larger models. There is a strong sentiment that small models like Qwen3.5-9B represent the future of local models, with some users expressing excitement for potential larger versions like a 27B model.

    - **Qwen 3.5 9B** is being compared to much larger models, with some users suggesting it performs on par with 100B+ models in certain tasks. This highlights the potential of smaller models to compete with medium-sized ones, especially in local environments where resource constraints are a concern. The model's ability to handle complex tasks with fewer resources is seen as a significant advancement.
    - A key technical challenge with smaller models like **Qwen 3.5 9B** is their tendency to overwrite existing code without checking, a problem in agentic loops. However, when used as background agents for file exploration and code edits, the performance gap with larger models like 70B is smaller than expected. The main issue remains in multi-step error recovery, where the smaller model can fix immediate errors but often misses upstream causes.
    - The training set's distribution is crucial for models like **OmniCoder-9B**. While 425K trajectories seem extensive, if the data is skewed towards common tasks like Python web development, the model's performance on infrastructure code or less common languages might be limited. This highlights the importance of diverse training data to ensure robust performance across various coding tasks.

  - **[Omnicoder-9b SLAPS in Opencode](https://www.reddit.com/r/LocalLLaMA/comments/1rsa8wd/omnicoder9b_slaps_in_opencode/)** (Activity: 351): **The post discusses the performance of **OmniCoder-9B**, a heavily fine-tuned version of `qwen3.5-9b` on Opus traces, which is available on [Hugging Face](https://huggingface.co/Tesslate/OmniCoder-9B). The user reports achieving impressive speeds of `40tps` with a `100k` context length using `Q4_km gguf` with `ik_llama`, even on a system with only `8GB VRAM`. The setup involves specific parameters such as `-ngl 999`, `-fa 1`, `-b 2048`, `-ub 512`, `-t 8`, `-c 100000`, `--temp 0.4`, `--top-p 0.95`, and `--top-k 20`. However, a bug causing full prompt reprocessing is noted, with a suggestion to adjust `ctx-checkpoints` to resolve it.** One commenter questions the performance comparison between OmniCoder-9B and regular Qwen 3.5 9B and 35B MOE models, particularly in tool calling within Opencode. Another suggests setting `ctx-checkpoints > 0` to address the full prompt reprocessing issue.

    - **OmniCoder-9B** is noted for its ability to run on consumer hardware without significant resource strain, making it a competitive option for local deployment. This is particularly relevant in the context of increasing quota restrictions from major providers like Copilot, which have shifted from unlimited use to daily limits, highlighting the value of local models that avoid such constraints.
    - A user reported issues with **Qwen 3.5 models** in Opencode, specifically their failure to utilize available tools for operations like `grep`, `read`, and `write`, instead defaulting to basic shell commands like `cat` and `ls`. This suggests potential limitations or bugs in the model's tool-calling capabilities within this environment.
    - Another user experienced a significant issue when testing OmniCoder-9B on a TypeScript frontend, where a simple formatting change led to a complete frontend breakdown. This raises concerns about the reliability of smaller local models like OmniCoder-9B in practical applications, especially when compared to the more generous usage limits of Qwen 3.5 Plus, which offers `1200 calls/day`.


### 2. Qwen Model Series and Performance

  - **[Qwen3.5-9B is actually quite good for agentic coding](https://www.reddit.com/r/LocalLLaMA/comments/1rrw8df/qwen359b_is_actually_quite_good_for_agentic_coding/)** (Activity: 606): **The post discusses the performance of the **Qwen 3.5-9B** model for agentic coding tasks on a consumer-grade **Nvidia Geforce RTX 3060** with `12 GB VRAM`. The user experimented with various models, including **Qwen 2.5 Coder** and **Unsloth quantizations on Qwen 3 Coder**, but found **Qwen 3.5-9B** to be surprisingly effective, maintaining functionality for over an hour without issues. The user highlights that **Unsloth-Qwen3 Coder 30B UD-TQ1_0** is also effective for code completion, though larger models like `2-bit quants` were slower and less stable. The post suggests that smaller, non-coding optimized versions of Qwen models may perform better on limited hardware.** One commenter noted that **Qwen3.5-9B** performs comparably to larger models like `gpt120b`, which is impressive given its size. Another user reported mixed results, with the model sometimes failing significantly, such as disrupting a build system, indicating variability in performance.

    - sleepingsysadmin highlights that Qwen3.5-9B performs impressively well, benchmarking around the level of GPT-3's 120B model, despite its smaller size. This suggests a high level of efficiency and capability in handling tasks relative to its size, which is notable for developers considering resource constraints.
    - linuxid10t shares a mixed experience with Qwen3.5-9B, noting that while it can perform well, it also has significant failures, such as disrupting a build system and deleting a project. This indicates potential reliability issues when using it for critical coding tasks, especially when compared to other tools like LM Studio and Claude Code on an RTX 4060.
    - -dysangel- comments on the ongoing debate about the utility of lower quantized models, suggesting that despite skepticism, Qwen3.5-9B demonstrates that such models can indeed perform useful tasks. This reflects a broader discussion in the AI community about the trade-offs between model size, quantization, and practical utility.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Technology and Political Implications

  - **[Palantir CEO Boasts That AI Technology Will Lessen The Power Of Highly Educated, Mostly Democrat Voters](https://www.reddit.com/r/singularity/comments/1rsnwhl/palantir_ceo_boasts_that_ai_technology_will/)** (Activity: 2076): ****Palantir CEO Alex Karp** has made controversial statements suggesting that AI technology will reduce the influence of "highly educated, often female voters, who vote mostly Democrat," while enhancing the power of vocationally trained, working-class men. In a [CNBC interview](https://newrepublic.com/post/207693/palantir-ceo-karp-disrupting-democratic-power), Karp argued that AI disrupts the economic power of humanities-trained voters, who are largely Democratic, and shifts it towards working-class voters. This statement has sparked concerns about AI being perceived as a tool for political manipulation, potentially radicalizing voters against AI by 2028. The discussion also touches on the potential for AI to automate intellectual jobs, like software engineering, before physical labor, highlighting the need for policies like Universal Basic Income to mitigate economic displacement.** Commenters express skepticism about Palantir's political neutrality, noting the irony of Karp's humanities background. The discussion reflects broader concerns about AI's societal impact and the motivations of its leaders.

    - CombustibleLemon_13 highlights the impact of AI on both vocational and white-collar jobs, noting that automation threatens factory and trade jobs as well. As white-collar roles diminish, displaced workers may enter blue-collar sectors, potentially leading to an oversupply of labor and reduced worker power. This perspective challenges the notion that only highly educated jobs are at risk.
    - mightbearobot_ raises concerns about the accessibility of powerful AI technologies, arguing that they are likely to be used as tools for control by current leadership rather than being made widely available. This comment reflects skepticism about the democratization of AI and suggests that it may exacerbate existing power imbalances rather than alleviate them.

  - **[Bernie Sanders officially introduces legislation to BAN the construction of all new AI data centers, citing existential threat to humanity.](https://www.reddit.com/r/singularity/comments/1rrjcon/bernie_sanders_officially_introduces_legislation/)** (Activity: 4564): ****Bernie Sanders** has introduced legislation aimed at banning the construction of new AI data centers, citing them as an existential threat to humanity. This proposal reflects a significant policy stance, potentially influencing the broader political discourse on AI regulation. The legislation could impact the development and deployment of AI technologies, as data centers are critical for AI model training and operation. [Source](https://youtu.be/qu2m7ePTsqY?si=zdl_cuRg22Nv_Df5).** Commenters suggest that banning data centers in the U.S. might not prevent their construction globally, as countries like China may continue development. Others propose regulation over prohibition, suggesting data centers should have independent power grids and not contribute to local environmental issues. There's also a notion of creating international AI research facilities similar to CERN.


  - **[SAM ALTMAN: “We see a future where intelligence is a utility, like electricity or water, and people buy it from us on a meter.”](https://www.reddit.com/r/singularity/comments/1rro8ej/sam_altman_we_see_a_future_where_intelligence_is/)** (Activity: 9032): ****Sam Altman** envisions a future where intelligence is commoditized like utilities such as electricity or water, suggesting that people will purchase it on a metered basis. This statement aligns with OpenAI's mission to advance digital intelligence for the benefit of humanity, as stated in 2015. However, the analogy to electricity is critiqued for potentially implying that AI could become a regulated utility with capped prices and government oversight, which contrasts with OpenAI's current high valuation and investor expectations.** Commenters debate the implications of commoditizing intelligence, with some suggesting that it could lead to regulation similar to utilities, which may not align with OpenAI's business model or investor interests.

    - No-Understanding2406 highlights a critical flaw in Sam Altman's analogy of intelligence as a utility like electricity. They point out that electricity is a commoditized service, often regulated with capped prices and public ownership or control. This implies that if intelligence were to follow the same path, OpenAI might become a regulated utility with limited profit margins, which contradicts the high valuation and investor expectations for the company. The commenter suggests that Altman's true intention might be to position OpenAI more like an oil company, which is less regulated and more profitable, though this is less palatable to the public.


### 2. Gemini and Nano Banana Pro User Experiences

  - **[Gemini’s task automation is here and it’s wild | The Verge](https://www.reddit.com/r/singularity/comments/1rs1r4j/geminis_task_automation_is_here_and_its_wild_the/)** (Activity: 722): ****Gemini**, a task automation system, has been introduced with capabilities to handle complex tasks such as ordering an Uber or selecting items from a menu. It demonstrates advanced decision-making by asking clarifying questions and making context-aware choices, such as skipping unnecessary steps or correctly specifying preferences like warming a pastry. This showcases significant progress in AI task automation, moving beyond simple command execution to nuanced interaction and decision-making.** There is a debate on the potential resistance from businesses like airlines to such automation due to pricing transparency concerns. A user shared an experience of creating a Chrome plugin for price comparison, highlighting the legal challenges individuals might face from corporations.

    - Recoil42 describes Gemini's task automation capabilities, highlighting its ability to handle complex tasks with minimal user input. For instance, when ordering an Uber, Gemini intelligently asked for clarification on the destination and streamlined the process by skipping unnecessary steps. Similarly, when ordering a coffee and croissant, it autonomously made decisions like warming the pastry, showcasing its advanced decision-making capabilities compared to earlier versions.
    - mckirkus discusses the potential resistance from businesses like airlines to automation technologies that could disrupt their pricing strategies. He shares an example of a Chrome plugin he developed to rank supermarket products by price per ounce, suggesting that while companies may attempt legal action to protect their interests, they face challenges in enforcing such actions against individuals.
    - MarcusSurealius critiques the practical utility of Gemini's current capabilities, arguing that the examples provided, such as booking plane tickets, are not everyday tasks for most users. He suggests more practical applications like automating financial management, tax preparation, and creating efficient shopping lists, which would demonstrate more daily utility and relevance for users.

  - **[Enshittification of Nano Banana Pro](https://www.reddit.com/r/GeminiAI/comments/1rs58vz/enshittification_of_nano_banana_pro/)** (Activity: 1069): **The post discusses a perceived decline in the quality of the Nano Banana Pro image generator, part of the Gemini ecosystem, after March 10. Users report that the tool, which previously produced sharp 2K images, now generates pixelated and blurry outputs. This change is seen as a bait-and-switch tactic, where initial high-quality results attracted users, only for the quality to degrade later. The image accompanying the post is a meme illustrating this decline with a 'Bait and Switch' comparison of bananas, symbolizing the quality drop in the service.** Commenters suggest that this decline is part of a broader business model where AI services initially offer high-quality outputs to attract users and media attention, only to reduce quality later to cut costs. They argue that this reflects a need for open models and collaborative projects, as proprietary models may prioritize profit over user satisfaction.

    - The discussion highlights a common business model in AI where companies initially offer powerful models to attract users and media attention, but later reduce capabilities to cut costs and maximize profits. This is seen as a strategic move to maintain financial sustainability once the initial hype has subsided.
    - There is a sentiment that companies like OpenAI and Claude implement restrictive measures, such as rate limits, to manage costs, indicating that the current pricing models are unsustainable. This reflects a broader industry trend where AI services are initially accessible but become more restricted over time to ensure profitability.
    - The conversation suggests exploring local AI model options like 'flux 2 Klein 9b', which offer unlimited usage if the user's hardware can support it. These models provide flexibility in terms of resolution and customization through downloadable fixes and patches, presenting an alternative to commercial models that may become restricted or costly.

  - **[New Gemini UI/UX 2.0 Upgrade is here!](https://www.reddit.com/r/Bard/comments/1rsnwx6/new_gemini_uiux_20_upgrade_is_here/)** (Activity: 730): **The image showcases the new Gemini UI/UX 2.0 upgrade, highlighting a personalized and interactive user interface with a focus on upgrading to 'Google AI Ultra.' This upgrade seems to emphasize a streamlined process for users to enhance their AI capabilities, although it comes with a significant cost, as noted by users who mention a $250 subscription fee. The comments reflect a debate on the value of the Ultra subscription, with some users finding the Pro version sufficient for their needs, especially when considering the cost savings and comparable performance to other AI services like ChatGPT Pro and Claude Opus 4.6.** Users express skepticism about the value of the Ultra subscription, noting that the Pro version offers sufficient features for most tasks at a lower cost. There is also concern about potential ads in the Gemini Pro version, which could diminish its utility.

    - **IfNightThen** discusses the cost-effectiveness of downgrading from Ultra to Pro, highlighting a savings of `$220` per month. They note that the only significant losses were access to 'Deep Think' and 'agents mode', which are becoming standard features elsewhere. They suggest that for media generation, using VertexAI might be more economical on a monthly basis, and found Pro's storage to be adequate for their needs.
    - **Appropriate-Heat-977** raises concerns about the pricing strategy for Gemini, noting that even with a Pro subscription, users are prompted to upgrade to a $250 Ultra subscription to access Gemini, with potential rate limits. This suggests a potentially prohibitive cost structure for accessing advanced features, which may not align with user expectations.
    - **IfNightThen** also compares the performance of 'Deep Think' unfavorably against competitors like ChatGPT Pro and Claude Opus 4.6, indicating that despite its premium pricing, it often required multiple retries to function effectively. This highlights potential performance issues with Gemini's advanced features compared to other AI models.


### 3. AI Model and Infrastructure Discussions


  - **[Drastically Stronger: Qwen 3.5 40B dense, Claude Opus](https://www.reddit.com/r/Qwen_AI/comments/1rsa7h0/drastically_stronger_qwen_35_40b_dense_claude_opus/)** (Activity: 273): **The post introduces the **Qwen 3.5 40B Claude Opus**, a custom-built and tuned model, part of a collection of 33 fine-tuned Qwen 3.5 models. The model focuses on high reasoning capabilities, utilizing a dataset with over `325 likes`. The repository has been updated to include the dataset used. The model is part of a series that includes various sizes and configurations, such as the 27B dense model, which has been customized by users. The **Architect series** uses XML tool descriptions, and the **Holodeck** model is configured in Instruct mode with a Star Trek theme. Benchmarks for the **Qwen3.5-27B-Engineer-Deckard-Claude** model include `arc: 0.668`, `perplexity: 3.674 ± 0.022`, and other metrics across different tasks. The qx86-hi quantization formula uses mixed 8/6 bit precision, showing performance above straight q8.** Commenters are interested in the technical details of model customization, such as layer duplication and quantization strategies. There is also interest in the potential applications and performance of these models in specific tasks, as well as their appeal to communities like r/LocalLlama.

    - StateSame5557 discusses the creation of custom models from the Qwen 3.5 40B and Claude Opus series, highlighting the use of XML tool descriptions which seem to be preferred by Claude. They provide links to various models on Hugging Face, such as the Qwen3.5-40B-Holodeck-Claude, which is an Architect in Instruct mode with a unique prompt setting. The comment also includes benchmark results for different models, such as the Qwen3.5-27B-Engineer-Deckard-Claude, which is a merge between a model trained on Philip K. Dick's works and the Claude model, using a qx86-hi quantization formula that mixes 8/6 bit settings.
    - StateSame5557 provides detailed benchmark results for several models, including the Qwen3.5-27B-Architect-Deckard-Heretic and Qwen3.5-27B-Text, with metrics like arc, boolq, hswag, and others. The Qwen3.5-27B-Engineer-Deckard-Claude model is noted for its unique quantization approach, using a mixed 8/6 bit formula that reportedly performs better than straight q8. The perplexity of the qx86-hi model is given as 3.674 ± 0.022, indicating its efficiency in handling language tasks.
    - Charming_Support726 inquires about the suitability of these models for cybersecurity applications, specifically for Blue Team and Red Team testing. They express interest in models that can be 'abliterated' for such purposes, suggesting a need for robust and adaptable AI models in cybersecurity contexts.



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.