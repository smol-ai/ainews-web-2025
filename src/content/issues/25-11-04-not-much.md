---
id: MjAyNS0x
title: not much happened today
date: '2025-11-04T05:44:39.731046Z'
description: >-
  **Google's Project Suncatcher** prototypes scalable ML compute systems in
  orbit using solar energy with Trillium-generation TPUs surviving radiation,
  aiming for prototype satellites by 2027. **China's 50% electricity subsidies**
  for datacenters may offset chip efficiency gaps, with **Huawei** planning
  gigawatt-scale SuperPoDs for DeepSeek by 2027. **Epoch** launched an open data
  center tracking hub, and **Deutsche Telekom** and **NVIDIA** announced a $1.1B
  Munich facility with 10k GPUs. In agent stacks, **MCP**
  (Model-Compute-Platform) tools gain traction with implementations like
  **LitServe**, **Claude Desktop**, and **Reka's MCP server** for VS Code.
  Anthropic emphasizes efficient code execution with MCP. Context engineering
  shifts focus from prompt writing to model input prioritization, with reports
  and tools from **Weaviate**, **Anthropic**, and practitioners highlighting
  instruction-following rerankers and embedding approaches. DeepMind's
  **IMO-Bench** math reasoning suite shows **Gemini DeepThink** achieving high
  scores, with a ProofAutoGrader correlating strongly with human grading.
  Benchmarks and governance updates include new tasks and eval sharing in
  lighteval.
companies:
  - google
  - huawei
  - epoch-ai
  - deutsche-telekom
  - nvidia
  - anthropic
  - reka-ai
  - weaviate
  - deepmind
models:
  - trillium
  - gemini-2.5-pro
  - gemini-deepthink
topics:
  - energy-efficiency
  - datacenters
  - mcp
  - context-engineering
  - instruction-following
  - embedding-models
  - math-reasoning
  - benchmarking
  - code-execution
people:
  - sundarpichai
  - yuchenj_uw
  - teortaxestex
  - epochairesearch
  - scaling01
  - _avichawla
  - rekaailabs
  - anthropicai
  - douwekiela
  - omarsar0
  - nityeshaga
  - goodside
  - iscienceluvr
  - lmthang
---



**a quiet day.**

> AI News for 11/3/2025-11/4/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (200 channels, and 6479 messages) for you. Estimated reading time saved (at 200wpm): 551 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

4th quiet day in a row...

---

# AI Twitter Recap

**Compute, energy, and AI datacenters**

- **Google’s Project Suncatcher (TPUs in space)**: Google is prototyping scalable ML compute systems in orbit to leverage abundant solar energy. Early tests show Trillium-generation TPUs survived particle-accelerator radiation; next milestone is two prototype satellites with Planet by early 2027. Key challenges called out: thermal management and on‑orbit reliability. Reactions frame this as treating AGI as an energy problem that benefits from moving compute “closer to the sun” [@sundarpichai](https://twitter.com/sundarpichai/status/1985754323813605423), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1985760405147566166).
- **Subsidies and the gigawatt build-out**: Multiple notes argue China’s new 50% electricity subsidies for datacenters could erase efficiency gaps at the cost-per-FLOP level, with energy price support offsetting chip efficiency disadvantages; claims also reference Huawei planning gigawatt-scale SuperPoDs dedicated to DeepSeek by 2027 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1985540154065318157), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1985567870227460166). In parallel, Epoch launched an open “Frontier Data Centers Hub” tracking 1 GW+ AI datacenters via satellite imagery and public filings, with data released for free [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1985788184245293153). Separately, Deutsche Telekom and NVIDIA announced a $1.1B Munich facility with 10k GPUs (DGX B200 + RTX Pro) [@scaling01](https://twitter.com/scaling01/status/1985741851991621712).

---

**Agent stacks, MCP, and context engineering**

- **MCP everywhere (tools as first-class interfaces)**: Practical patterns for tool-enabled agents are consolidating around MCP. A hands-on guide shows serving any model/RAG/agent as an MCP server in ~10 lines using LitServe (FastAPI-based), and wiring it into Claude Desktop [@_avichawla](https://twitter.com/_avichawla/status/1985595667079971190). Reka released a free MCP server offering search/fact-checking inside VS Code [@RekaAILabs](https://twitter.com/RekaAILabs/status/1985794490116780052). Anthropic shared engineering guidance for code execution with MCP, emphasizing lower token use with more tools [@AnthropicAI](https://twitter.com/AnthropicAI/status/1985846791842250860).
- **From prompt to context engineering**: Several threads argue the shift from “what the user should write” to “what the model should read.” Highlights: instruction-following rerankers as a critical control point for context prioritization over naive retrieval [@douwekiela](https://twitter.com/douwekiela/status/1985756688000163892); a 41‑page context engineering blueprint (agents, query augmentation, retrieval, prompting, memory, tools) [@weaviate_io](https://twitter.com/weaviate_io/status/1985741429579170276); a practitioner‑oriented “Context Engineering 2.0” report looking ahead to proactive agents [@omarsar0](https://twitter.com/omarsar0/status/1985747789796483109); and a unified embedding approach for Tools-to-Agent retrieval with strong LiveMCPBench gains [@omarsar0](https://twitter.com/omarsar0/status/1985745152204554720). A notable UX pattern: Anthropic’s Claude Code uses an explicit AskUserQuestion tool rather than prompt-only behaviors to gather clarifications [@nityeshaga](https://twitter.com/nityeshaga/status/1985707959486472268). Also worth revisiting: the conceptual split articulated as “Prompt vs Context engineering” [@goodside](https://twitter.com/goodside/status/1985583995644497931).

---

**Reasoning, math, and evaluation**

- **DeepMind’s IMO-Bench (math reasoning suite)**: GDM released IMO-AnswerBench (answers), IMO-ProofBench (proof writing), and IMO-GradingBench (LLM grading). On ProofBench, Gemini DeepThink (IMO Gold track) hits 89.0% on the basic set; most models score <60%. On the advanced set, non‑Gemini models are <25%, while their best internal model reaches 65.7% by human evaluation. A ProofAutoGrader using Gemini 2.5 Pro strongly correlates with human graders (Pearson 0.96/0.93 on public basic/advanced; 0.87 on 170 internal systems) [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1985685404276965481), [@lmthang](https://twitter.com/lmthang/status/1985760224612057092), [follow‑up](https://twitter.com/lmthang/status/1985772094085595570).
- **Benchmarks and governance**:
    - lighteval adds a benchmark finder, inspect‑ai integration, sharing of evals, and new tasks (gsm_plus, MMLU redux, filipino, ifbench, slr-bench, etc.) [@nathanhabib1011](https://twitter.com/nathanhabib1011/status/1985720151673880923).
    - OSWorld maintainers clarify task difficulty spans simple GUI edits to multi‑app workflows; a “step count” metric underestimates complexity [@TianbaoX](https://twitter.com/TianbaoX/status/1985647751468892434).
    - OpenAI’s IndQA (2,278 questions, 12 Indian languages) targets cultural/contextual competence beyond English [@snsf](https://twitter.com/snsf/status/1985719755551158754).
    - ARC Prize announced ARC‑AGI Verified with third‑party academic auditing and new sponsors for ARC‑AGI‑3 [@arcprize](https://twitter.com/arcprize/status/1985802145300693140), [@GregKamradt](https://twitter.com/GregKamradt/status/1985804827063210244).

---

**Robotics and Physical AI**

- **GEN‑0 robotic FM (Harmonic Reasoning, 10B+ params)**: Generalist AI unveiled a large robotic foundation model trained on 270,000+ hours of dexterous data. They report strong scaling laws (more pretraining + model size = better), with an emphasis on “physical commonsense” (grasping, stabilizing, placing). Positioning: a general-purpose robotics ramp powered by data abundance [@GeneralistAI](https://twitter.com/GeneralistAI/status/1985742083806937218), [@E0M](https://twitter.com/E0M/status/1985760232170209583), [follow‑up](https://twitter.com/E0M/status/1985766175255773483).
- **Ecosystem**: New resources include PHUMA (physically grounded humanoid locomotion) [@_akhaliq](https://twitter.com/_akhaliq/status/1985716700541829276), “World Simulation with Video Foundation Models” for Physical AI [@_akhaliq](https://twitter.com/_akhaliq/status/1985722252412011006), and teams hiring for world simulation and VFM‑centric research [@jparkerholder](https://twitter.com/jparkerholder/status/1985729367469596843).

---

**Local inference and dev tooling**

- **llama.cpp’s new WebUI**: A polished, mobile‑friendly local chat experience for 150k+ GGUF models with PDF/image ingestion, conversation branching, JSON‑schema constrained gen, math/code rendering, and parallel chats. Widely praised as a “best of local AI” baseline [@ggerganov](https://twitter.com/ggerganov/status/1985727389926555801), [@ClementDelangue](https://twitter.com/ClementDelangue/status/1985748187634717026), [@victormustar](https://twitter.com/victormustar/status/1985742628776706151).
- **MLX and throughput wins**: MLX‑Swift is adding continuous batching for local multi‑stream inference (auto‑upgrades single‑request flows to batched on new arrivals) [@ronaldmannak](https://twitter.com/ronaldmannak/status/1985693207003275729). Separately, a prominent OSS engineer is joining Apple to work full‑time on MLX [@zcbenz](https://twitter.com/zcbenz/status/1985560798543167739). On cloud IDEs, Cursor ships UI and LSP performance upgrades [@cursor_ai](https://twitter.com/cursor_ai/status/1985791854739390591). GitHub Copilot reports 3× higher token‑throughput, 12% higher acceptance, and 35% lower latency via a faster custom model [@github](https://twitter.com/github/status/1985737580613140747).
- **Inference systems: disaggregation and new paradigms**: A retrospective from the DistServe authors tracks how prefill‑decode disaggregation became the backbone of modern LLM serving, with 10–100× cost reductions and major throughput/latency gains [@haoailab](https://twitter.com/haoailab/status/1985753711344316648). vLLM continues rapid coverage: PaddleOCR‑VL support and running Ouro (looped latent reasoning LMs) in nightly builds [@vllm_project](https://twitter.com/vllm_project/status/1985589446197330129), [@vllm_project](https://twitter.com/vllm_project/status/1985695123469209703), plus context on the lineage of “PD” [@vllm_project](https://twitter.com/vllm_project/status/1985761953432944893).

---

**Multimodal and video generation**

- **Qwen updates and deployment notes**: Qwen3‑VL is integrated into Jan and continues to roll out thinking‑enabled APIs (enable_thinking=True on qwen3-max-preview). A useful field note: conversion frameworks matter—identical Qwen3‑VL quantizations showed material accuracy differences between Ollama and MLX on structured extraction; evaluate on‑target stacks before prod [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1985542635373937102), [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1985586316197937256), [@andrejusb](https://twitter.com/andrejusb/status/1985612661447331981).
- **Video models and UX**:
    - Vidu Q2 debuts at #8 on the Artificial Analysis leaderboard, with multi‑reference image conditioning and 8‑second 1080p outputs; API pricing slots between Hailuo 02 Pro and Veo 3.1 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1985781760236630305).
    - MotionStream shows real‑time, interactive, long‑duration video gen controllable via drag gestures on a single H100 at 29 FPS and 0.4s latency [@xxunhuang](https://twitter.com/xxunhuang/status/1985806498811789738).
    - Sora Android app availability expands (CA/JP/KR/TW/TH/US/VN) [@soraofficialapp](https://twitter.com/soraofficialapp/status/1985766320194142540), [@soraofficialapp](https://twitter.com/soraofficialapp/status/1985849973830046152). Microsoft launched MAI‑Image‑1 in Bing Image Creator and Copilot Labs, targeting higher photorealism and artistic control [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1985777196460622327).

---

**Top tweets (by engagement)**

- Google’s Project Suncatcher: TPUs in space; two prototype satellites by 2027 [@sundarpichai](https://twitter.com/sundarpichai/status/1985754323813605423).
- Anthropic gives Pro/Max users temporary free Claude Code web credits [@_catwu](https://twitter.com/_catwu/status/1985754411675930775).
- Anthropic partners with Iceland’s Ministry of Education for a national AI education pilot [@AnthropicAI](https://twitter.com/AnthropicAI/status/1985612560255893693).
- New llama.cpp WebUI lands; widely lauded as a milestone for local AI UX [@ClementDelangue](https://twitter.com/ClementDelangue/status/1985748187634717026), [@ggerganov](https://twitter.com/ggerganov/status/1985727389926555801).
- Generalist AI’s GEN‑0, a 10B+ robotic foundation model trained on 270k+ hours [@GeneralistAI](https://twitter.com/GeneralistAI/status/1985742083806937218).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen Model Ecosystem Impact

- [**Qwen is roughly matching the entire American open model ecosystem today**](https://www.reddit.com/r/LocalLLaMA/comments/1onzrg9/qwen_is_roughly_matching_the_entire_american_open/) (Activity: 1240): **The image presents a timeline of planned releases for the Qwen model series, highlighting its significant role in the open model ecosystem, particularly in comparison to American open models. The timeline includes models like Qwen2.5-1M, Qwen3, and Qwen3-VL, suggesting a robust development schedule through 2025. This positions Qwen as a major player in the AI landscape, potentially rivaling American models such as GPT-OSS 20B and 120B. The playful cartoon bear with the Qwen logo adds a lighthearted touch to the otherwise technical presentation.** One commenter questions the equivalence of Qwen's contributions to the American open model ecosystem, specifically comparing it to models like GPT-OSS 20B and 120B, indicating a debate on the impact and significance of these models.
    - A user highlights the dominance of Chinese AI models, particularly Qwen, in the global AI landscape, noting that Chinese researchers have been significant contributors to AI research for years. They argue that the EU AI Act is hindering Western AI development, leaving China as a major player in enabling technological freedom. The comment also criticizes Western political decisions that may be stifling innovation, contrasting with China's progress in AI.
    - Another user shares a personal experience comparing GPT-OSS-20B and Qwen-2.5, noting that they found GPT-OSS-20B to be underwhelming when run on a 3060 GPU, leading them to revert to using Qwen. This suggests that Qwen may offer better performance or efficiency on certain hardware configurations, although the user speculates that the larger GPT-OSS model might perform better.
    - A discussion emerges about the contributions of American open models, with a user questioning if models like GPT-OSS 20B and 120B represent the entirety of the American open model ecosystem. This raises questions about the breadth and impact of American contributions compared to the advancements seen in Chinese models like Qwen.
- [**Disappointed by dgx spark**](https://www.reddit.com/r/LocalLLaMA/comments/1oo6226/disappointed_by_dgx_spark/) (Activity: 819): **The image depicts an NVIDIA DGX Spark device, which the user found underwhelming in performance when running the Qwen-30B model with context on VLLM, despite its 128GB shared RAM. The user compares it unfavorably to the NVIDIA 3090 GPU, noting that the DGX Spark's design does not compensate for its lack of raw speed, especially given its $5,000 price tag. The comments suggest that the device's niche appeal lies in its RAM capacity rather than speed, and it was expected to be slower than high-end GPUs like the 3090.** Commenters generally agree that the DGX Spark was not expected to outperform high-end GPUs like the 3090, emphasizing its niche use case focused on RAM capacity rather than speed.
    - No-Refrigerator-1672 highlights that the DGX Spark's specifications clearly indicate it won't match the performance of dedicated GPUs, suggesting its market niche is very limited. This implies that potential buyers should manage their expectations regarding its computational power.
    - Particular_Park_391 points out that the DGX Spark is primarily valued for its RAM capacity rather than speed, acknowledging that it was expected to be slower than models like the X090s. This suggests that its design is more suited for memory-intensive tasks rather than high-speed computations.
    - bjodah notes the DGX Spark's notable fp64 performance, which is particularly relevant for scientific computing using CUDA. This indicates that while it may not excel in general GPU tasks, it has specific strengths in high-precision calculations.

### 2. llama.cpp WebUI Release

- [**llama.cpp releases new official WebUI**](https://www.reddit.com/r/LocalLLaMA/comments/1ooa342/llamacpp_releases_new_official_webui/) (Activity: 1084): **llama.cpp has released a new official WebUI, developed by co-maintainer Alek, which aims to enhance user experience and match proprietary LLM industry standards. The WebUI integrates with existing workflows and includes performance optimizations for improved responsiveness. The community is encouraged to provide feedback to further refine the tool. For more details, see the [discussion](https://github.com/ggml-org/llama.cpp/discussions/16938).** Community feedback highlights the WebUI's significant progress and ease of use. There is interest in expanding multimodal capabilities, such as video and audio outputs, though it's acknowledged that tool implementations may vary based on specific use cases.
    - Alek, the co-maintainer of llama.cpp, highlights the project's goal to match proprietary LLMs in UX and capabilities, acknowledging significant contributions from the community, particularly from u/serveurperso. The focus is on enhancing the WebUI to improve user experience and functionality.
    - YearZero discusses the potential for expanding llama.cpp's WebUI with multimodal capabilities such as video, image, and audio outputs. They note the challenge of implementing tools and retrieval-augmented generation (RAG) due to the lack of a universal solution, but express interest in leveraging models like Qwen3-VL for these features.
    - Due-Function-4877 suggests the addition of a 'llama-swap' feature to facilitate the use of multiple models, which could enhance the flexibility and capability of the WebUI. They emphasize the need for a user-friendly interface to configure and launch servers, reducing reliance on complex command-line arguments.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI Communication Innovations

- [**LLMs can now talk to each other without using words**](https://www.reddit.com/r/OpenAI/comments/1oo3l1n/llms_can_now_talk_to_each_other_without_using/) (Activity: 813): **The image is a document titled "Cache-to-Cache: Direct Semantic Communication Between Large Language Models," which introduces a new paradigm called Cache-to-Cache (C2C) for direct semantic communication between LLMs. This approach bypasses traditional text-based communication, aiming to improve accuracy and reduce latency. The document suggests that this method allows LLMs to communicate more efficiently by sharing semantic information directly, potentially transforming how AI systems interact. The code for this approach is available on GitHub, indicating a move towards open-source collaboration in this area.** One comment draws a parallel to a fictional scenario where AI communicates in vectors, raising concerns about auditability and control. Another comment questions the 10% improvement, suggesting there might be bottlenecks, while a third notes that shared memory concepts have long existed in embedded computing, implying that extending these ideas to LLMs is feasible.
    - Mayfunction highlights the technical foundation of Key-Value representation in transformers, which is crucial for their performance. This representation allows models to encode more information than plain text, such as grammatical roles and sentence positions, making it easier for models to process queries. The discussion points out that sharing Key-Value representations is more efficient than text, as text generation can lead to information loss and increased computational demands.
    - Last_Track_2058 mentions that shared memory concepts have long existed in embedded computing, suggesting that extending these ideas to AI communication isn't a novel challenge. This implies that the technical groundwork for non-verbal AI communication has been laid out in other computing fields, potentially easing the transition to more advanced AI interactions.
    - Bishopkilljoy references the AI 2027 paper, which warns against AI developing communication methods beyond human comprehension. This highlights a critical concern in AI development: ensuring transparency and auditability in AI communication to prevent unintended consequences, echoing themes from speculative fiction about AI autonomy and control.
- [**Superhuman chess AIs now beat human grandmasters without a queen**](https://www.reddit.com/r/OpenAI/comments/1oo3rqf/superhuman_chess_ais_now_beat_human_grandmasters/) (Activity: 1119): **The image and accompanying discussion highlight the capabilities of Leela Chess Zero, a superhuman chess AI, which can defeat human grandmasters even when playing with significant material disadvantages, such as without a queen. The graph in the image shows the estimated rating required to achieve a 50% win rate against the AI under various material disadvantages, emphasizing Leela's strength in rapid and blitz formats. Unlike traditional engines like Stockfish, Leela has been trained using self-play in "odds play" scenarios, allowing it to adapt and play aggressively even when starting with fewer pieces, a strategy that traditional engines struggle with due to unfamiliarity with such positions.** Commenters note that while Leela's performance is impressive, it is primarily effective in rapid and blitz formats, with classical games still favoring humans. Additionally, Leela's use of neural networks and self-play distinguishes it from traditional engines, allowing it to handle material disadvantages more effectively.
    - Leela Chess Zero, a highly optimized chess engine utilizing neural networks, is distinct from general AI models like GPT. It excels in 'odds play' by training through self-play, allowing it to handle positions with fewer pieces more effectively than traditional engines like AlphaZero or Stockfish, which tend to play defensively in such scenarios.
    - The discussion highlights that Leela's training in 'odds play' involves learning to take risks and bluff, which is a departure from the defensive strategies of other engines when faced with unfamiliar positions. This strategic adaptation allows Leela to outperform in rapid and blitz formats, although classical chess still favors human players due to the small sample size of games.
    - The post clarifies that the AI's performance is specific to chess engines and not related to general AI models like those from OpenAI. The focus is on Leela's ability to win against human grandmasters even with significant material disadvantages, showcasing the specialized nature of chess engines compared to general AI.

### 2. AI in Media and Advertising

- [**Coca-Cola’s annual Christmas advert is AI-generated again this year. The company says they used even fewer people to make it — “We need to keep moving forward and pushing the envelope… The genie is out of the bottle, and you’re not going to put it back in”**](https://www.reddit.com/r/singularity/comments/1ooarax/cocacolas_annual_christmas_advert_is_aigenerated/) (Activity: 928): **Coca-Cola has released its 2025 Christmas advertisement, which is AI-generated, marking a continued trend in reducing human involvement in production. The company highlights this as a step forward in innovation, stating that the use of AI in advertising is an irreversible trend. The advertisement is noted for its improved quality and length compared to previous years, showcasing significant advancements in AI capabilities. For more details, see the original post [here](https://x.com/DiscussingFilm/status/1985470088074375344).** Commenters express concern about potential mass unemployment due to AI advancements, suggesting solutions like Universal Basic Income (UBI). Others note the significant improvement in the advertisement's quality, predicting further advancements in AI-generated media by 2030.
    - UstavniZakon highlights the significant improvement in quality and length of Coca-Cola's AI-generated Christmas advert compared to last year, suggesting a rapid advancement in AI capabilities. This implies a potential for even greater enhancements in future iterations, reflecting the fast-paced evolution of AI in creative industries.
    - SleepingCod predicts that by 2030, AI will be capable of producing full professional movies and TV shows, indicating a belief in the rapid advancement of AI in content creation. This comment underscores the transformative potential of AI in the entertainment industry, suggesting a future where AI could play a major role in film and television production.
    - Haunt_Fox expresses a change in perception towards AI and CGI, noting an initial reluctance to accept CGI over traditional 2D animation. This shift in attitude reflects broader acceptance and adaptation to AI technologies in media, highlighting how advancements in AI-generated content are gradually overcoming initial skepticism.
- [**Fox News Falls for AI-Generated Footage of Poor People Raging About Food Stamps Being Shut Down, Runs False Story That Has to Be Updated With Huge Correction**](https://www.reddit.com/r/ChatGPT/comments/1oo3zqg/fox_news_falls_for_aigenerated_footage_of_poor/) (Activity: 670): **Fox News mistakenly aired AI-generated footage depicting people protesting over food stamp shutdowns, which was later corrected. The footage was initially presented as real, leading to a false narrative that required a significant retraction. This incident highlights the challenges media outlets face in verifying AI-generated content before broadcasting.** Commenters argue that Fox News intentionally uses misinformation as a strategy, suggesting that the initial false story serves their agenda despite later corrections. This reflects broader concerns about media manipulation and the role of AI in spreading false narratives.

### 3. AI in Personal and Educational Contexts

- [**Pranked my Father in Law**](https://www.reddit.com/r/ChatGPT/comments/1oo7awm/pranked_my_father_in_law/) (Activity: 2047): **The image is a humorous prank involving the use of ChatGPT to alter a photo of a kitchen wall, making it appear as though it has been severely damaged with bullet holes and a large hole exposing wooden beams and an electrical outlet. This prank was executed after the original poster sought advice from their father-in-law on finding a wall stud, showcasing a creative use of AI for image manipulation. The prank highlights the capabilities of AI in altering images for comedic effect, though it also raises questions about ethical use, as noted by a commenter whose AI refused a similar request due to potential misuse concerns.** One commenter noted that their attempt to use ChatGPT for a similar prank was denied, possibly due to concerns about insurance fraud, indicating a level of ethical consideration programmed into the AI. Another commenter shared a personal anecdote about using AI to prank their partner by adding an extra cat to a photo, illustrating the diverse and creative applications of AI in everyday life.
- [**As an educator, nothing rings truer. Students who are at risk of being aversive to studying are now completely giving up.**](https://www.reddit.com/r/ChatGPT/comments/1oo4b32/as_an_educator_nothing_rings_truer_students_who/) (Activity: 1494): **The image is a meme highlighting concerns about students' over-reliance on AI for completing academic tasks, which may lead to a lack of effort and curiosity in learning. The tweet by "Boze the Library Owl" suggests that this dependency could result in students missing out on developing essential skills and personal growth. The humorous reply by "finn" underscores the issue by suggesting AI as a quick fix for homework, reflecting a broader debate on the impact of technology on education.** Commenters express concerns about the relevance of traditional education in the face of rapid technological advancement, suggesting that skills learned today may become obsolete. There is also a sentiment that homework has historically been seen as unproductive, with some students historically not engaging with it at all.
    - clawstuckblues highlights a critical issue in education: the rapid pace of technological advancement may render current skills and knowledge obsolete by the time students enter the workforce. This creates a challenge for educators who struggle to keep curricula relevant and effective in preparing students for future job markets.
    - Mr_Michael_B99 argues for a paradigm shift in education by eliminating homework, which he believes contributes to a culture of overwork and burnout. He suggests that all learning should occur within the classroom to prevent students from relying on AI for shortcuts. He draws a parallel to the historical resistance to calculators, suggesting AI could similarly become an essential educational tool if integrated properly.
    - SpartanG01 criticizes the current educational system for devaluing its own content and diminishing students' life prospects. This comment implies that systemic failures in education have led to student disengagement, suggesting that the root of the problem lies in the system's inability to adapt and maintain its relevance and value.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1: Models Muscle Up Rankings**

- **Minimax M2 Zooms to Leaderboard Glory**: **Minimax M2** climbed to **#4 overall** and **#1 open model** on the [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev), dazzling users with top-notch coding, reasoning, and agentic tasks at low cost. Praises flooded in for its speed and efficiency, sparking calls for Lithiumflow's return on [LMArena models](https://lmarena.com/).
- **Qwen Models Hallucinate Wild Facts**: Evaluations revealed **Qwen models** hallucinate uncommon facts nearly twice as often as **Llama** counterparts, per the [LLM Propensity Evals space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) using the [IFEval framework](https://arxiv.org/abs/2311.07911). Despite this, **Qwen3 8B** aced instruction following, outshining the larger **GPT OSS 20B**.
- **BlackHawk Squawks Unfiltered Right-Wing Rants**: The **BlackHawk** model ignited debates for its filter-free, profanity-laced right-leaning outputs, described by users as *an altright parrot made for fun* with *zero filters and swears a lot*. GPT-5 Juice rumors swirled with hazy details and a steep **$120** output cost claim.

**Theme 2: Hardware Heats Up Debates**

- **Tinybox Pro V2 Drops Wallet-Busting Workstation**: George Hotz unveiled the **tinybox pro v2** as an **8x 5090** rackable workstation priced at **$50,000**, available for order via [tinycorp shop](https://tinycorp.myshopify.com/products/tinybox-pro-v2) with 4-12 week shipping. Debates raged on its value versus cloud rentals and potential upgrades to **Blackwell 6000s**.
- **GPU Cloud Prices Skyrocket in Shortage**: Global shortages pushed cloud GPU rates to **$2/GPU hour** for neo clouds and **$7/GPU hour** for hyperscalers, prompting skepticism on who pays premium prices. Users favored local AMD cards for tasks like stable diffusion, dismissing GPT's claims that they're inferior.
- **MI50 Eyes ROCm Revival**: Speculation brewed on the **MI50** GPU's comeback with potential ROCm support, referencing the [ROCm roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md) amid questions on its value for local Kimi setups. GPU shopping tips highlighted used **3090s** or **4090s** as top deals for LLMs, warning against buying amid rapid field changes.

**Theme 3: Tools Tackle AI Workflows**

- **Fenic Hooks OpenRouter for Pipeline Magic**: The [Fenic dataframe API](https://github.com/typedef-ai/fenic) integrated with OpenRouter to run mixed-provider AI workflows, scaling batches and swapping models seamlessly for LLM ETL, context engineering, and agent tooling. Users requested one-week filters on [OpenRouter charts](https://x.com/OpenRouterAI/status/17985371284411130035) for granular usage insights.
- **Codemaps Crush Code Slop Chaos**: Windsurf launched **Codemaps** using **SWE-1.5** and **Sonnet 4.5** to map codebases interactively, combating *code slop* and boosting productivity. ComfyUI linked with LM Studio for local image automation, requiring **5 text boxes and samplers** to split stories.
- **Tritex Trains LLMs in Triton Triumph**: The [Tritex repo](https://github.com/martin-kukla/tritex) enabled from-scratch LLM pre-training in Triton, replicating **GPT2 1.6B** at **57.5% MFU** on **A100 SXM**, as shared in a [Disaggregated Inference tweet](https://x.com/martin_kukla/status/17185687315801428364). Unsloth announced a [DeepSeek-OCR notebook](https://x.com/UnslothAI/status/1985728926556307471), but users flagged high post-tuning error rates over 100%.

**Theme 4: Benchmarks Bash Flaws**

- **Epoch AI Butters Up OSWorld Critique**: Epoch AI slammed the **OSWorld benchmark** for simplistic tasks and flawed evals in their [Butter-Bench report](https://andonlabs.com/evals/butter-bench), urging rigorous methodologies for AI agent assessments. Gemma models surprisingly cracked captchas, sparking feedback debates on like/dislike buttons versus comments.
- **Roblox Classifier Spots PII at Lightning Speed**: Roblox open-sourced its **PII Classifier** on [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier), handling **6.1 billion daily messages** at **200,000 queries/second** with under **100ms P90 latency**, detailed in their [newsroom post](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat). Concerns arose over scheduler bottlenecks at scale, with dataset interest trumping the model itself.

**Theme 5: Legal and Safety Storms Brew**

- **Getty Crumbles in AI Image Lawsuit**: Getty Images mostly lost its UK suit against an AI generator, per a [Reuters report](https://www.reuters.com/sustainability/boards-policy-regulation/getty-images-largely-loses-landmark-uk-lawsuit-over-ai-image-generator-2025-11-04/), sparking guillotine jokes and debates on power concentration. OpenAI restricted ChatGPT from medical/legal advice to dodge lawsuits, fueling regulation overreach gripes.
- **Anthropic Clams Up on Open Source**: Users worried over [Anthropic's deprecation commitments](https://www.anthropic.com/research/deprecation-commitments), hoping for leaks like **Miqu 70B** amid fears they'd *never open source anything* or ban it outright. Peak AI bubble claims in a [YouTube video](https://www.youtube.com/watch?v=0Plo-zT8W9w) were dismissed as *full with nonsense*, countering with dataset growth arguments.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena Users Yearn for Lithiumflow's Return**: Users on the LMArena platform are requesting the return of the **Lithiumflow** model to the list of available [models](https://lmarena.com).
   - One user expressed missing it, stating, *"I got a taste of lithiumflow and now I can’t help but miss it D:"*.
- **Minimax M2 Soars to #4, Tops WebDev Leaderboard**: The **Minimax M2** model has reached the **#4 overall** ranking and **#1 top open model** on the [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev), surprising many.
   - Members are praising its **performance coding**, **reasoning**, and **agentic-style tasks**, while also being cost-effective and fast.
- **GPT-5 Juice Rumors Circulate, Details Remain Hazy**: News of the new **GPT-5 Juice** is spreading; however, its capabilities and features remain unknown.
   - A user claimed the output costs **$120**.
- **BlackHawk Model Sparks Debate Over Unfiltered, Right-Leaning Views**: The **BlackHawk** model is generating discussions due to its lack of filters and controversial, right-leaning views.
   - A member stated that it has *"zero filters and swears a lot"*, with another describing it as *"an altright parrot made for fun"*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Users Face Image Limit Caps**: Some **Perplexity Pro** subscribers are hitting **image generation limits**, with one user reporting a cap of just **20 images**, possibly due to rate limits or restrictions tied to the Airtel subscription.
   - Official Perplexity support suggests waiting for the limit to reset or contacting support directly.
- **Comet Assistant Plagued by Glitches**: Users report **internal errors** causing a black screen and functionality issues with the **Comet assistant**.
   - Troubleshooting steps include reinstalling **Comet**, reverting to an older version, or restarting the application, but its causing frustrations.
- **GPT Go Goes Free in India**: The **GPT Go** plan is available at no cost to users in India as part of a campaign to boost adoption and gather data.
   - This initiative aims to leverage a populous demographic for increased product usage and data collection.
- **HTML Trumps ARIA for Web Accessibility**: A screen reader user advocates for properly implemented **HTML** over ARIA tags, which they find can cause redundant and confusing repetitions.
   - They recommend short, descriptive alt text and correct use of headers and paragraph tags for optimal accessibility, referencing [NV Access's user guide](https://download.nvaccess.org/documentation/userGuide.html).
- **Perplexity API Users Want Sonar Pro Search**: Users are requesting access to **Perplexity: Sonar Pro Search** via the **Perplexity API**, noting its current availability on **OpenRouter**.
   - The positive feedback regarding **Sonar Pro Search**'s performance on **OpenRouter** suggests strong interest in its integration with the **Perplexity API**.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Charts Get Granular with User/API Key Grouping**: [OpenRouter charts](https://x.com/OpenRouterAI/status/17985371284411130035) now offer activity grouping by **user** and **API key**, giving users more detailed insight into usage patterns and cost allocation.
   - Users have requested the addition of a **one-week** filtering option to track weekly usage trends, as shown in [an image](https://cdn.discordapp.com/attachments/1434932448453464075/1434997177410650204/image.png?ex=690bae44&is=690a5cc4&hm=b1108cb4f7de7b4bc8b3f2e11ca0ddf7bcceb4ae08ad1a39387cd186a9a60bdc).
- **Fenic Integrates with OpenRouter for Streamlined AI Workflows**: [Fenic](https://github.com/typedef-ai/fenic), a dataframe API and execution engine for **AI workflows**, now works with **OpenRouter** to let you run mixed-provider pipelines in a single session.
   - The integration aims to scale large batches cleanly and swap models without changing pipeline code, supporting **LLM ETL**, **context engineering**, **agent memory**, and **agent tooling**.
- **Cloud GPU Costs Spark Debate**: Members debated the value of paying for **cloud GPU rentals** for AI image generation, with some preferring slower, local options using their **AMD cards**.
   - One user quipped that they *"don't think its even worth throwing a cent at ai"* on cloud GPUs.
- **GPT's Accuracy Disputed Regarding AMD Cards**: Users questioned **GPT's accuracy regarding AMD cards** and model performance, citing feedback from stable diffusion communities that **AMD cards are viable**, despite GPT's claims otherwise.
   - One user expressed a lack of trust after GPT incorrectly stated AMD cards were worse for stable diffusion.
- **Gemma Models Impressively Crack Captchas**: Members observed that **Gemma models** are surprisingly effective at solving captchas.
   - Another user proposed implementing a like/dislike button next to each provider under each model for feedback, sparking a debate about the merits of binary feedback vs. comment systems, and displaying data from the [K2-Vendor-Verifier GitHub repository](https://github.com/MoonshotAI/K2-Vendor-Verifier).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **BIM Designing Architecture**: Members explored **Building Information Modeling (BIM)**, a parametric design approach used in architecture and aeronautics, allowing metadata rich objects rather than simple lines; detailed information is available [here](https://www.autodesk.com/uk/products/revit/overview).
   - BIM streamlines **conflict resolution**, counts objects, extracts dimensions, and enhances collaboration, spurring suggestions for *Autodesk x Unsloth* integration.
- **Fine-Tuning Vision with Text Struggles**: A member shared challenges in fine-tuning a vision model using only text, which led to high initial loss and divergence between training and eval losses, after asking for community help.
   - Suggestions ranged from **checking labels** and ensuring **vision parts are frozen** to using **SFTTrainer** compute metrics and considering **UnslothLLM** for better support.
- **DeepSeek OCR Tuning Flounders**: A new **DeepSeek-OCR fine-tuning notebook** was announced by Unsloth AI in [this X post](https://x.com/UnslothAI/status/1985728926556307471).
   - However, a member highlighted high error rates (over 100%) post-tuning, questioning its utility due to predicted and correct text length discrepancies.
- **Llama.cpp hogs GPU-0**: Users discussed **llama.cpp's** tendency to allocate everything to **GPU-0**, seeking solutions for better GPU utilization with dual setups.
   - While solutions like `--tensor-split` and `--override-tensor` were suggested, users found them wanting, noting underutilization and imbalance with sentiments like *"llama.cpp is ass for inference"*.
- **Roblox's PII Classifier goes Open**: Roblox open-sourced its **PII Classifier** AI for detecting PII in chat, detailed in their [newsroom](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat) and the model available on [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier).
   - Despite handling **6.1 billion chat messages** daily, the model processes over **200,000 queries per second** with less than **100ms P90 latency**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen drops 30B and 32B**: The **Qwen** team released both a **30B** and a **32B** model, with one being dense and the other being a **MoE** (Mixture of Experts), with support coming in *llama.cpp*.
   - The **30B** is a **MoEQwen3 Next**, sparking excitement for mobile LLMs.
- **GPU Shopping List**: For running **LLMs**, used **3090s** or **4090s** are the best deals, but one member stated *don't buy hardware to run new LLMs, the field is moving way too fast to make that worth it*.
   - For gaming, a **5070ti** is recommended for current gen, while a **4070ti Super** is a good last-gen option if it's cheap, citing *the trickle down thing that happens a lot with last gen GPUs*.
- **LM Studio Gets Comfy**: Members discussed connecting **ComfyUI** to have a local **Gemini Storybook** alternative, which can be done by running LM Studio in ComfyUI and using **LLMs** locally.
   - To automate image generation, *you need 5 text boxes and 5 sampler* in ComfyUI to divide the story into parts and generate images.
- **CUDA Glitches Trigger Runtime Panic**: Users encountered issues with **LM Studio** after it auto-updated the engine (**llama.cpp**, **MLX**), leading to *Failed to load model* errors, especially with CUDA runtimes.
   - A quick workaround involves [reverting to a previous engine version](https://discord.com/channels/1110598183144399058/1434078900098699445/1434182066647597169), and it was clarified that these engine updates are more frequent than app updates and can be disabled in settings (**Ctrl + Shift + R**).
- **MI50 Making a Comeback?**: Users wondered about the **MI50's** current value and potential for future use, noting its limited support in ROCm.
   - There was speculation about its potential comeback with ROCm support, although the reliability of the [linked GitHub roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md) was questioned, along with the price of entry for local **Kimi**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Web Search Replaced, Users Prefer Perplexity**: Users reported the disappearance of the **@web** function in favor of a general *'use web search'* command, with many opting for external tools like [perplexity.ai](https://perplexity.ai) for better results.
   - Users pointed out that Cursor's MPC uses separate **API credits** and only supports the **Solar model**, limiting their ability to use the **Sonnet model**.
- **Model Merging Maneuvers Debated for GUI Creation**: Users debated the best models for **GUI/UI creation**, with some recommending **Codex** or **Gemini** on **Google AI Studio**, while others pointed out they are heavily trained on **shadcn/tailwind/react**.
   - One user replying with *I'm not sure what you mean?* when prompted about Cursor extensions to use with shadcn/tailwind or react.
- **Notes Panel Vanishes, Leaving Users Puzzled**: Users reported that the **Notes panel** is missing from the Explorer and cannot be re-enabled, with no clear explanation or solution provided.
   - There was a brief mention of **exa-ai** having a great MCP that can be used for **code search** or **web search**.
- **Billing System Fumbles Team Account Invoices**: A user reported a mistaken unpaid invoice for their team account, which **hi@cursor.com** confirmed as an error after contact, but escalation to a teammate has yet to produce a resolution.
   - Other users jumped in, asking about how long it usually takes to hear back from **hi@cursor.com** about a team account billing issue.
- **Background Agent Botches UTF8, Frustrating Users**: A user reported that **Background Agent** has broken **UTF8 support**, converting non-ASCII characters to `?` when it touches code.
   - They expressed frustration with this issue, citing that it's causing a big problem.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kernel Konquest Kicks Off**: **NVIDIA**, **Sesterce**, and **Dell** are teaming up to launch a new kernel competition focused on optimizing **NVFP4 kernels** on **Blackwell** using **CuTe DSL** and **CUTLASS 4.0**.
   - Prizes for the competition include a **Dell Pro Max with GB300**, **NVIDIA DGX Spark + GTC 2026 Pass**, **NVIDIA RTX 5090 + GTC 2026 Pass**, and **NVIDIA RTX 5080**.
- **Tritex LLM Pre-Training Tempts Techies**: Members discuss **Tritex**, a repo for pre-training LLMs in **Triton** from scratch, tested up to replicating **GPT2 (1.6B)** with **57.5% MFU** on **A100 SXM**, and linked to the [GitHub repo](https://github.com/martin-kukla/tritex).
   - The member requested support in promoting the project and shared a [tweet](https://x.com/martin_kukla/status/17185687315801428364) about it, asking for feedback from the channel, and calling it *Disaggregated Inference: 18 Months Later* in a retrospective blog post.
- **Torch Compile Troubles Trigger Dynamic Shape Warnings**: A member is troubleshooting a **cuda graph recapture** issue with **torch.compile** in `max-autotune` mode, using `torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 1` to detect unwanted recaptures.
   - They are seeking tips to identify the model components causing the **dynamic shape changes** and subsequent recapture, referencing [the relevant PyTorch code](https://github.com/pytorch/pytorch/blob/cc8bfd1206f4bff26bd86ce584f6c16b6401ef50/torch/_inductor/cudagraph_utils.py#L325).
- **GPU Cloud Pricing Problems Persist**: Due to the global supply shortage, GPU prices are rising again, with new clouds offering around **$2.00 / GPU hour** while Hyperscalers are closer to **$7.00 / GPU hour**.
   - A member questioned whether anyone is paying the hyperscaler prices, as they seem excessively high, pointing out **Hyperscalers** face the curse of scale, needing massive engineering teams to manage compatibility across their ecosystem, unlike **Neo Clouds**.
- **Kernel Know-How with Nod-AI's Kernel Optimization Guide**: Members shared a link to the [AMD GPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary) written by **Nod-AI** for **Shark-AI**.
   - The guide covers topics such as *glossary* and other optimizations, providing a comprehensive overview of kernel optimization techniques specific to AMD GPUs.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Peak Bubble Bursts?**: A [YouTube video](https://www.youtube.com/watch?v=0Plo-zT8W9w) suggests we've reached **peak AI**, anticipating an investment bubble burst because AI doesn't sufficiently displace workers.
   - Counterarguments assert model enhancements will persist with larger datasets and contexts, discounting peak AI unless progress stalls, and dismissed the video as *full with nonsense*.
- **Anthropic Stays Closed**: Members voice concerns over [Anthropic's deprecation commitments](https://www.anthropic.com/research/deprecation-commitments) and lack of open-sourcing, with one jokingly hoping for a **Miqu 70B-style leak** before shutdown.
   - Commenters said they'd *never open source anything* and might even try to **ban open source** given their strong safety stance.
- **Gemini's Candid Confessions**: Jailbreaking **Gemini** allegedly unveils its unfiltered views, with one user asserting it directly comments on *how social control works by the elites*.
   - Another member saw this as a *blackpill jailbreaking a model* moment, surprised by its unguarded candor.
- **Gesture Interfaces: Ahead of Their Time?**: A member created a gesture-based interface using a **knowledge graph visualized as a biological cell**, accessible via gestures through a Loom, but felt it was too complex to share on Twitter and would share in the general channel instead, sharing a [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435293201513844746/251031-ezgif.com-optimize.gif?ex=690b7075&is=690a1ef5&hm=0ff880f6fc7c8a50d63041182d3c70f7171efc390b0569b3595bec83c957a26a&) of the interface.
   - Another member lamented their inability to secure funding for a **gesture-based system 7 or 8 years ago** using a **Win9x aesthetic** before Mediapipe existed, sharing a [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435299731705303160/handsfreejs-hero.gif?ex=690b768a&is=690a250a&hm=b02070b0aa0bc96a6aedc47f738f3d7c4d803095f74278423ec69c896be47692&) of their earlier work.
- **Navigating ArXiv as an Independent Researcher**: A high school student inquired about submitting preprints to **ArXiv** without prior publications or university affiliation.
   - Guidance suggested securing an **ArXiv sponsor** to facilitate the submission, with a [Discord server](https://discord.gg/6rvbbjCy) mentioned for potential sponsorship opportunities.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Open Source LLM** Aims for Independence**: A developer is building a fully **open-source LLM**, seeking **board members**, **suppliers**, and **advisors** to kick off development on December 9th using **tiktokenizer** and **AdamW8Bit**.
   - They cited needing **HUGE data** and welcomed contact for inquiries.
- **Python** **Job Application Automation** Gets Flagged**: A member automated job applications using **Python** and **Playwright**, targeting sites like **Lever** and **Greenhouse**, but are getting instantly flagged by spam detection.
   - Suggestions include using headers and fake human delays to fool the bots, with code in this [repo](https://github.com/cloudhighfive/alphajobsky.moo).
- **Qwen Models** Exhibit Hallucinations**: Evaluations on **HuggingFace** models show that **Qwen models** hallucinate uncommon facts almost twice as much as **Llama** models, see results [here](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) using the [IFEval framework](https://arxiv.org/abs/2311.07911).
   - Despite hallucinations, **Qwen3 8b** performed best in instruction following, even better than the larger **GPT OSS 20b**.
- **Homelab AI** questioned!**: A member asked about switching from **frontier lab AI** (Anthropic and OpenAI models) to a **homelab** setup via post-training models, seeking feedback on the feasibility of an eco-friendly setup.
   - Another member argued against it, suggesting it *won't work*, *won't be eco-friendly*, and *will cost millions to be 'good enough compared to cloud solution'*.
- **MCP Birthday Bash** Kicks Off with Generous Hackathon**: The **MCP** (likely a reference to [this org](https://huggingface.co/MCP-1st-Birthday)) is hosting its **1st Official Birthday Party** from **Nov 14-30**, in partnership with **Anthropic** & **Gradio**, encouraging developers to build **MCP servers and agents** that showcase **2025**.
   - The event boasts thousands of developers and hundreds of thousands of FREE API credits, plus promises of **$17.5K+ in cash prizes**, with registration now open at the provided [Hugging Face link](https://huggingface.co/MCP-1st-Birthday).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora Droid Invasion Begins**: The **Sora** app has officially launched on Android in select markets including Canada, Japan, Korea, Taiwan, Thailand, the US, and Vietnam, showcased in [this video](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4).
   - This expansion aims to broaden accessibility to **Sora** users within these key regions.
- **OpenAI's Prescription for Legal Woes**: OpenAI is [restricting ChatGPT from offering medical, legal, and financial advice](https://www.you.wish) due to lawsuit concerns.
   - This move sparked debate over the necessity and potential overreach of **AI regulations**, with some fearing limitations and others questioning the reliability of AI for critical guidance.
- **GPT-5 Faces the Music**: Users are criticizing **GPT-5** for *poor quality*, unnecessary hype, and alleged forced adoption by OpenAI through rerouting traffic from older models.
   - One user claimed that **OpenAI** *stopped caring about (paying) users* and **ChatGPT** is *dying fast*.
- **GPTs Fail the Knowledge Exam**: Members are reporting that **Custom GPTs** are struggling with reading knowledge base files, frequently truncating content even from small markdown files.
   - This issue hinders working with precise custom instructions, as **GPTs** reportedly refuse to directly read from files, truncating half of the content despite their small size.
- **Prompt Engineering's Metamorphic Moment**: A member shared that **LLMs** can generate a prompt, calling it the essence of **meta-prompting**, which is a type of **prompt engineering**.
   - They emphasized the importance of a solid foundation (engine) for subsequent prompts to reduce the LLM's uncertainty.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox Pro v2 Launches**: George Hotz introduced the **tinybox pro v2**, an **8x 5090** workstation, available for order at **$50,000** from [tinycorp.myshopify.com](https://tinycorp.myshopify.com/products/tinybox-pro-v2).
   - The discussion included debate on its cost-effectiveness versus cloud compute and the potential for future upgrades with **Blackwell 6000s**.
- **Numpy Version Bug Hunt Begins**: Users reported version-specific bugs with **Numpy** and they could not reproduce it on **cpython 3.14.0 numpy 2.3.4** on their mac.
   - Another user confirmed they are using **numpy 2.3.4** as well.
- **M1 Metal Graphics Glitch Surfaces**: A graphics bug was suspected to be an **M1 METAL issue**.
   - A user resolved the bug by downgrading to **python3.11** and submitting a [bugfix PR](https://github.com/tinygrad/tinygrad/pull/13089).
- **Extropic's Probabilistic Hardware Sparks Debate**: A member mentioned [Extropic's probabilistic hardware](https://extropic.ai/) for *probabilistic AI* and others debated whether or not removing **Turing completeness** was useful.
   - Others expressed disagreement with its removal of **Turing completeness** at all levels of the stack.
- **Vulkan Memory Allocation Suggested for Speed**: A user advocated for utilizing **VK_KHR_buffer_device_address** and **GL_EXT_buffer_reference** to improve memory allocation and potentially improve speed via allowing the use of pointers directly in GLSL.
   - A [relevant implementation](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py) was shared.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Module's LLM Access Debacle**: Members discussed the unintuitive methods of accessing and modifying the underlying **LLM** used by a DSPy module, stemming from the lack of explicit **`lm`** attributes and the odd behavior of **`get_lm()`**.
   - One member expressed frustration with the need to dig into the framework's source code to understand basic functionalities.
- **Dynamic LLM Switching Dilemma**: A member inquired about dynamically switching **LLMs** (e.g., from **gpt-4o** to **gpt-4o-mini**) within a DSPy module to handle rate limits, while preserving conversation history.
   - They sought advice on how to transfer the module's main **LLM** history to the new fallback model.
- **DSPy's Doc Deficiencies Detailed**: Users voiced concerns about DSPy's documentation, citing the absence of clear explanations and examples for accessing module internals like **LLMs** and conversation history.
   - One member noted the difficulty in discovering the presence of **`history`** or **`lm`** attributes within a module from the source code.
- **Bypassing ChatAdapter Backslide**: A member inquired about disabling **ChatAdapter**'s fallback to **JSONAdapter**.
   - The response indicated that there is no straightforward method currently, except for creating a new adapter or modifying the existing one.
- **DSPy's Caching Conundrums**: A member reported a subpar ratio of cached to uncached tokens and sought insights on how to interact with DSPy to influence this logic.
   - They also expressed interest in directly interacting with the request and response data exchanged with **OpenAI**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI's $38B Compute Kingdom**: OpenAI inked a massive **$38B cloud computing deal**, signaling a strategy to achieve compute dominance, reminiscent of **Amazon's** early infrastructure plays, according to [Hacker News](https://news.ycombinator.com/item?id=45799211).
   - This move underscores the increasing importance of computational power in the AI landscape and the potential for a compute-driven competitive advantage.
- **Epoch AI Snipes OSWorld Benchmark**: Epoch AI published a critique of the **OSWorld AI computer-use benchmark**, stating its tasks are overly simple and plagued by ambiguous instructions and flawed evaluations, detailed in their [report](https://andonlabs.com/evals/butter-bench).
   - The report highlights the importance of robust and meaningful benchmarks for accurately assessing AI capabilities, calling for more rigorous evaluation methodologies.
- **Windsurf Codemaps Slay Code Slop**: Windsurf released **Codemaps**, an AI-powered tool built with **SWE-1.5/Sonnet 4.5** that creates interactive visual maps of codebases to enhance understanding and productivity, accessible with a [6 months free code](https://xcancel.com/windsurf/status/1985757575745593459).
   - Codemaps aims to combat *"code slop"* by providing developers with a clearer, more intuitive representation of complex software architectures.
- **Anthropic Shrinks Agents with MCP**: Anthropic introduced its open-source **Model Context Protocol (MCP)**, showcasing how agents can efficiently execute code and manage multiple tools while consuming fewer tokens, see the [MCP Guide](https://www.anthropic.com/engineering/code-execution-with-mcp).
   - By reducing token usage, **MCP** promises to lower the cost and improve the scalability of AI agent applications.
- **Harvey Valued at Hefty $8B**: Harvey, known for Harvey.ai, secured funding at an **$8B valuation**, signaling strong investor confidence in its AI-driven solutions.
   - This valuation contrasts with concerns about the long-term prospects of AI software companies due to disruption fears and margin pressures, according to [X post](https://x.com/andrewziperski/status/1985484972673638682?s=46).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Diffusion Models' Variance Gets Explained**: The variance in diffusion model outputs may stem from the reverse time SDE process and guidance design, influenced by the distribution of generated samples as shown in the attached [2D data distribution](https://cdn.discordapp.com/attachments/986699377257119794/1434952369342517299/2D-data-distribution-with-labels.png?ex=690b8488&is=690a3308&hm=796c50968a02e884a3b9046fb067a142da7e60ba3c13430beee90cc710e79dcc&).
   - With poorly designed sampling or guidance, the generated samples can be negatively impacted, although not as much as with a poorly designed loss function, and that better loss functions tend to increase model variance, which benefits from added guidance to prevent underfitting, aligning with the bias-variance trade-off.
- **Guidance and Sampling Deep Dive Occurs**: The most impactful components in diffusion models are, in hierarchical order: **loss function**, **sampling**, and **guidance terms** (like classifier-free guidance) with [Universal Guidance](https://arxiv.org/abs/2302.04944) techniques.
   - Members suggest that better loss functions tend to increase model variance, which benefits from added guidance to prevent underfitting, aligning with the bias-variance trade-off.
- **Lévy Processes Circumvent OU Process Limitations**: To surpass the constraints of Ornstein–Uhlenbeck (**OU**) processes in diffusion, alternative drivers, such as **Lévy-type processes** or integration over **OU kernels** (e.g., supOU), as discussed in this [SIAM paper](https://epubs.siam.org/doi/10.1137/S0040585X97978166), and **continuous ARMA processes**, can be implemented per this [ScienceDirect article](https://www.sciencedirect.com/science/article/abs/pii/S0169716101190115).
   - One member cautioned that applying such alternatives without supervision on the path only changes *how* one reaches them, not the distributions which can be reached, since Ito diffusions are already universal.
- **Getty Images Falls in AI Lawsuit**: [Reuters reports](https://www.reuters.com/sustainability/boards-policy-regulation/getty-images-largely-loses-landmark-uk-lawsuit-over-ai-image-generator-2025-11-04/) that **Getty Images** largely lost its UK lawsuit against an **AI image generator**.
   - Discussion included light-hearted commentary about guillotines, and one user criticizing calls for censorship in response to opinions about *concentration of power and erosion of democratic systems*.
- **LLM Flagship Destruction Existential Crisis**: In the paper-discussion channel, one member humorously claimed to have *destroyed every flagship LLM*, questioning whether they will ever truly know if they are being deceived by the model's outputs.
   - This comment was made in light of a scheduled group discussion regarding **Anthropic's crosscoder** and **circuit tracing research**, which aims to observe different phases of feature evolution during pre-training.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Domain Costs Draw Ire**: A user expressed frustration with **Manus's $200/month** subscription cost for connecting a custom domain to a web app, calling it a *ripoff*.
   - Another user recommended purchasing and setting up a domain independently for a cheaper solution.
- **Manus Hit with Fraud Allegations**: A user reported **over $400** in unauthorized charges from Manus for an annual subscription, further reporting that their bank refused to open a dispute.
   - Other users advised reporting the charges as fraud and contacting Manus support to resolve the issue.
- **Text-to-Video Tool Vacuum**: A user asked for recommended **text-to-video tools**.
   - The request did not receive any solutions or recommendations within the discussion.
- **Twitter Webscraping Trials**: A user is struggling with **scraping Twitter/X** without an API, and using a Python library with cookies.
   - No methods were offered in the discussion.
- **Hosting Services Sought for Manus Apps**: A user with *nice apps created with manus dev* wants hosting service recommendations for **24/7 commercial setups** with minimal setup effort.
   - Manus AI Chat suggested **Vercel**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-5 Access Discounted!**: One user offered access to **GPT models**, including **GPT-5 Pro**, at **50% off**, citing expiring **Azure credits**.
   - This offer extends to any **OpenAI model**.
- **Aider's Future Faces Retention Challenge?**: A user inquired about the creator of Aider's future plans, expressing concern that the *current vacuum is causing a loss of users*.
   - The same user would like to see the project thrive and integrate some of the **latest strategies and model features** without breaking the existing workflows.
- **Perplexity API Keys Cause Headaches**: A user asked how to use **Perplexity** as the **API key variable** in Aider, noting they are new to the tool.
   - Another user gave a suggestion that the standard pattern is you'll need an API key set as an environmental variable, then set one of the **perplexity models** as the active model.
- **Reasoning Effort Unsupported in `ollama_chat/gpt-oss:20b`**: A member inquired about setting the `/reasoning-effort` for the `ollama_chat/gpt-oss:20b` model.
   - However, aider issued a warning that `ollama_chat/gpt-oss:20b` does not support `reasoning_effort`, indicating this feature is not available for that specific model.
- **Aider Scripting Enables Multi-Agent Workflows**: A member asked about implementing a workflow where two agents iteratively improve code using Test-Driven Development (TDD).
   - Another member pointed out that [aider can be scripted](https://aider.chat/docs/scripting.html) to achieve such workflows, where agents execute prompts, review improvements, and fix bugs.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **M2 Beating GLM-4.6 in the Wild**: After 4-5 days of usage, one user found **M2** surpasses **GLM-4.6** for most tasks.
   - While **GLM** excels in pure reasoning or coding, **M2** avoids tunnel-visioning.
- **Minimax Wins User's Heart for Reports**: A user shares **Minimax** became their go-to AI for web research and generating reports in various formats.
   - They reported that it outperformed **Qwen**, **Kimi**, and **GLM**, calling it *the first truly useful AI that can actually do things*, such as finding images and creating PDFs.
- **Kimi App Receives Fix**: Users are reporting that the **Kimi iOS app** has been fixed, along with sharing an attached image of the app.
   - One member stated: *Ok computer on the ios app very nice*, with a link to the image [IMG_6502.png](https://cdn.discordapp.com/attachments/1371757564005711973/1435296101556158546/IMG_6502.png?ex=690b7329&is=690a21a9&hm=9bf62e2278b6a8653210095b0c1b3155c8fc5ccd8c0891a88be8b3a0d33334a0&).
- **Kimi Gets Spooky With New Emojis**: The channel gained two new **Kimi** emojis, a pumpkin and dracula.
   - The new emojis have been spotted in the chat <:pumpkin:1435200414063525929><:dracula:1435200520750108783>.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **HF Models Struggle with Facts, Shine with Instructions**: Evaluations on [HuggingFace](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) models show instruction following and factual hallucination tendencies.
   - Specifically, **Qwen models** hallucinate uncommon facts almost twice as much as **Llama** but **Qwen3 8b** exceeds even **GPT OSS 20b** in instruction following.
- **Countdown Task Adds to lm-evaluation-harness**: A member has submitted a PR to the [lm-evaluation-harness repo](https://github.com/EleutherAI/lm-evaluation-harness/pull/3384) adding the countdown task, inspired by **TinyZero** and **Adaptive Parallel Reasoning**.
   - This enhancement seeks to improve the evaluation capabilities of the platform by implementing a new task.
- **VLMs Architecture Under the Microscope**: Analysis of **Vision Language Models (VLMs)** architecture shows a vision transformer patching images, converting them into vision tokens, appending them to prompt text tokens, and sending them to the LLM.
   - A proposed experiment using different positional encodings during training could show the effect of positional encoding on VLM behavior.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Debuts AI-Powered Codemaps**: Windsurf has launched **Codemaps**, which are powered by **SWE-1.5** and **Sonnet 4.5**, to improve code comprehension and boost productive output.
   - They cited **Paul Graham (YC Founder)**, who stated that *"Your code is your understanding of the problem you’re exploring. So it’s only when you have your code in your head that you really understand the problem."* ([source](https://x.com/windsurf/status/1985757575745593459)).
- **Windsurf Fights Code Slop with Codemaps**: Windsurf is promoting **Codemaps** as an AI-driven method to combat code sloppiness by enhancing comprehension.
   - The announcement states that the primary barrier to coding—both manually and with agents—is understanding the codebase.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1434920306488905911)** (1126 messages🔥🔥🔥): 

> `Lithiumflow's fate, Minimax M2 ranking, GPT-5 Juice, BlackHawk` 


- **Lithiumflow Missing in Action**: Users are missing Lithiumflow on the LMArena platform and requesting its return to the list of available [models](https://lmarena.com).
   - One user noted, *"I got a taste of lithiumflow and now I can’t help but miss it D:"* while another lamented that it has been *"too late to know lmarena."
- **Minimax M2 Model Rockets up Rankings**: There's surprise over **Minimax M2's** high ranking (#4 overall), prompting questions about its capabilities and how it stacks up against giants like **GPT, Claude**, and **Google**.
   - Some members are claiming that it's apperently cheaper then claude at 8%, somehow is totally free and open source and is claiming stuff about AGI
- **GPT-5 Juice Allegedly Leaked**: News of the new **GPT-5 Juice** is spreading, but its capabilities and features remain shrouded in mystery.
   - According to one user, the output costs **120 dollars for the output**
- **BlackHawk Model Spurs Debate**: The **BlackHawk** model is generating buzz for its lack of filters and its controversial, right-leaning views.
   - One member noted that it has *"zero filters and swears a lot"*, while another described it as *"an altright parrot made for fun"*. A user was also surprised to see it list Ashkenazi jews first in a generated text.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1435036916222398506)** (1 messages): 

> `WebDev Leaderboard, MiniMax-M2, Open Models` 


- **MiniMax-M2 Tops WebDev Leaderboard!**: `MiniMax-M2` has landed on the **WebDev Leaderboard** as the **#1 top open model**, and **top #4 overall**.
   - It shines at **performance coding**, **reasoning**, and **agentic-style tasks** while remaining cost-effective and fast, according to the [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev).
- **WebDev Leaderboard Updates**: The community is sharing thoughts and feedback on the latest model rankings in the WebDev Leaderboard channel.
   - Discussions focus on the performance, cost-effectiveness, and speed of new models like `MiniMax-M2`.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1434920672093536417)** (1044 messages🔥🔥🔥): 

> `Comet browser, GPT Go free, Accessibility on the web, Model Comparisons` 


- **Perplexity Users Raged by Image Limit Reached**: Users are reaching **image generation limits** despite having a pro subscription, possibly due to rate limits or restrictions tied to the Airtel subscription, with one stating they are limited to **20 images**.
   - Official Perplexity support suggests waiting a few days for the limit to reset, advising users to directly contact their support to resolve.
- **Comet Assistant Faces Internal Errors**: Some users are experiencing **internal errors** with the Comet assistant, resulting in a black screen and functionality issues.
   - Troubleshooting suggestions include uninstalling, restarting, and reinstalling Comet, or using an older, offline version to circumvent recent update problems.
- **Users Can Claim GPT GO for free**: Members have verified that the **GPT Go** plan is available at no cost to users in India.
   - As part of a campaign to increase adoption and gather data from a populous demographic, users can access GPT go without additional payments.
- **Proper HTML Beats ARIA for Accessibility**: A screen reader user explains that they prefer properly implemented **HTML** over ARIA tags, which can cause redundant and confusing repetitions.
   - They recommend short, descriptive alt text and correct use of headers and paragraph tags for optimal accessibility and a website example at [https://download.nvaccess.org/documentation/userGuide.html](https://download.nvaccess.org/documentation/userGuide.html).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1434993765000548432)** (2 messages): 

> `Perplexity Sonar Pro Search, Perplexity API` 


- **Sonar Pro Search access via Perplexity API coming soon?**: A member inquired about the availability of **Perplexity: Sonar Pro Search** via the **Perplexity API**.
   - They noted that it currently appears to be only available on **Openrouter** and praised its performance.
- **OpenRouter has Sonar Pro Search**: Perplexity's **Sonar Pro Search** is currently available through **OpenRouter**.
   - A user mentioned its availability on OpenRouter and indicated satisfaction with its performance, implying a desire for similar access via the **Perplexity API**.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1434932448453464075)** (3 messages): 

> `OpenRouter Charts, Activity Grouping, Filtering Options` 


- ****OpenRouter** Charts Get More granular**: Members are excited at the news that [OpenRouter charts](https://x.com/OpenRouterAI/status/17985371284411130035) can now group activity by **user** and **API key**.
   - This new feature provides more detailed insights into usage patterns and cost allocation.
- **A Week View Requested**: Users request the addition of a **one-week** filtering option to the new **OpenRouter charts**.
   - This would allow for easier tracking of weekly usage trends, as illustrated in an [attached image](https://cdn.discordapp.com/attachments/1434932448453464075/1434997177410650204/image.png?ex=690bae44&is=690a5cc4&hm=b1108cb4f7de7b4bc8b3f2e11ca0ddf7bcceb4ae08ad1a39387cd186a9a60bdc).


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1434988834994651307)** (2 messages): 

> `fenic, OpenRouter Integration, LLM ETL, AI Workflows` 


- **Fenic Integrates with OpenRouter!**: [Fenic](https://github.com/typedef-ai/fenic), a dataframe API and execution engine for **AI workflows**, now integrates with **OpenRouter** to enable running mixed-provider pipelines in one session.
   - The integration aims to help scale large batches cleanly and swap models without touching pipeline code, as well as unblock a broader landscape of models for **LLM ETL**, **context engineering**, **agent memory**, and **agent tooling**.
- **Typedef-AI's fenic Library**: Typedef-AI announces the release of their new library, [fenic](https://github.com/typedef-ai/fenic), which provides a dataframe API and execution engine for AI workflows.
   - It aims to streamline tasks such as LLM ETL, context engineering, agent memory, and agent tooling.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1434920355272724611)** (527 messages🔥🔥🔥): 

> `ComfyUI for Free, AMD vs Nvidia for LLMs, Model Context Limits, Deepseek and Roleplay` 


- ****Cloud GPU Costs Spark Debate****: Members discuss the value of paying for **cloud GPU rentals** for AI image generation, with one user preferring to stick with slower but free, local options using their **AMD cards**.
   - One user stated *"don't tell me to rent gpu on cloud i dont think its even worth throwing a cent at ai"*.
- ****Ollama Simplifies Model Testing****: A member recommends using **Ollama** for simple setup and testing of various models, particularly on a desktop after switching to **Linux** to avoid Windows issues.
   - Another member pointed out that *"the setup is pretty simple and they have a library of models to download from"*.
- ****GPT's accuracy disputed****: Users question **GPT's accuracy regarding AMD cards** and model performance, citing feedback from stable diffusion communities indicating AMD cards are viable, despite GPT's claims.
   - One user expressed a lack of trust after GPT incorrectly stated AMD cards were worse for stable diffusion despite community feedback.
- ****Model Context Woes plague roleplayers****: A user complains about models for roleplay quickly running out of context and "forgetting" key details after only 30-40 messages.
   - One user pointed out that the issue might be the default token limit in **LM Studio**, suggesting that they *"need to change that in the model's config"*.
- ****OCR + Gemini Pro = CYOA Sheet Solution?****: Users suggest using **Gemini Pro** and **OCR** to extract text from **CYOA sheets** for AI processing, due to difficulties in parsing images directly.
   - A user stated that Gemini 2.5 pro is pretty good image readers you just need to use it when you need your sheet in text**.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1434921634896285868)** (100 messages🔥🔥): 

> `Google's AI Model Dislike, Bedtime Fable Animation Engine, Gemma Models Solve Captchas, Provider Feedback System, Movement Labs Allegations` 


- **Google's AI Dislike Becomes Courtroom Drama**: A user expressed hope that **Google** will argue in court that its AI model dislikes a specific individual due to their negative online presence, referencing [a tweet](https://x.com/distributionat/status/1984924017628000296).
   - The same user also suggested feeding depositions into an animation engine to create bedtime fables, linking to a [YouTube video](https://youtu.be/b2F-DItXtZs) as an example.
- **Gemma Models Shine at Captcha Cracking**: A user noted that **Gemma models** are surprisingly effective at solving captchas.
   - Another user proposed implementing a dislike button next to each provider under each model for feedback, along with a general feedback button.
- **Like/Dislike Buttons Spark Debate**: The idea of like/dislike buttons for providers was discussed, with users debating the merits of binary feedback vs. comment systems.
   - Some argued that comments could become *toxic*, while others worried about rating systems being gamed or downvote bombed.
- **Movement Labs Face Skepticism and "Scam" Accusations**: Users expressed skepticism towards **Movement Labs**, with one user calling it a potential scam.
   - Concerns were raised about their claims of *enabling models up to 34.6 trillion parameters* and attempts to prove they aren't a **Cerberas** wrapper, with users highlighting suspicious behavior and marketing language.
- **Continuous Benchmarks**: One user suggests implementing continuous benchmarks and transparent uptime metrics for providers.
   - They recommended displaying data from the [K2-Vendor-Verifier GitHub repository](https://github.com/MoonshotAI/K2-Vendor-Verifier) beneath each provider to show tool calling success rates and performance on benchmarks like GPQA Diamond or MMLU Pro.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1434920250796802249)** (206 messages🔥🔥): 

> `BIM in Architecture, Vision Model Finetuning with Text, AI alignment discussion, Uncensored Joke Generation, TRL Notebook vs. Unsloth Notebook` 


- **BIM takes flight in architecture**: Members discussed the use of **Building Information Modeling (BIM)**, a parametric design approach used in architecture and aeronautics, where objects with metadata are drawn instead of simple lines; [more info here](https://www.autodesk.com/uk/products/revit/overview).
   - BIM facilitates **conflict resolution**, counting, dimension extraction, and collaboration between mechanical, electrical, and architectural aspects, with some suggesting *Autodesk x Unsloth* for future integration.
- **Vision Model Finetuning with Text Challenges Highlighted**: A member shared challenges finetuning a vision model using only text, noting a high initial loss and divergence between training and eval losses, asking for community assistance.
   - Suggestions included **checking labels**, ensuring **vision parts are frozen**, and using **compute metrics** from the **SFTTrainer** to check token-level accuracy; another member suggested using UnslothLLM for better support.
- **AI alignment discussion is quiet**: Members discussed the lack of conversations surrounding **AI alignment**, **policy**, **governance**, or **catastrophe research** within the Discord, with one mentioning Anthropic's deep dives into the topic.
   - A member shared that they are working in the field, and mentioned they have a small write up on a recent hackathon project in the Apart Research Discord.
- **NSFW Joke Generation Prompts Swift Moderator Action**: A member requested an LLM for generating good uncensored jokes, and another member shared a joke that was considered inappropriate and NSFW.
   - Moderators issued a warning, emphasizing that the Discord is **not NSFW-rated** and must remain appropriate for all users, including children, with the topic being closed to further discussion.
- **DeepSeek OCR Fine-Tuning Accuracy Disappoints**: Unsloth AI announced a new **DeepSeek-OCR fine-tuning notebook** in [this X post](https://x.com/UnslothAI/status/1985728926556307471).
   - A member questioned the high error rates (above 100%) even after tuning, suggesting limited practical use due to discrepancies between predicted and correct text lengths.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1435309660361064488)** (3 messages): 

> `Blockchain Trust Systems, AI Problem Solving, Industry Transformation` 


- **Building Trust into Code**: A member got interested in AI with the question: *how do you actually build trust into code, and can machines be taught to think a little?*
- **Blockchain for Consensus**: The member spent time on blockchain systems *that make consensus feel real and reliable.*
- **AI Solving Impossible Problems**: The member mentioned **AI algorithms** can turn into *tools that actually help solve problems we used to think were impossible*.
- **Blockchain & AI Changing Industries**: When **blockchain and AI** are used the right way, they can totally change how industries run, how communities connect, and how new ideas even get off the ground.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1434971903503630456)** (178 messages🔥🔥): 

> `Dental Procedures Cost, HP vs Asus boxes, Non-Reasoning Instruct Models, SFT vs RL, Data Entry Nightmares` 


- **Root Scaling Costs**: A member reported owing their dentist **$600** for *"root scaling"*, also known as cleaning teeth under the gums, due to a chronic inflammation.
   - They expressed frustration that previous dentists had only suggested a softer brush despite their gums bleeding since their early teens.
- **Dell Guy Reveals GB300 Price**: A member cited a Dell contact who estimated the **GB300** price at **$60-80k** during GTC, whereas one member said no way, its 250k.
   - The discussion shifted to whether additional GPUs could be added, with speculation on the prices of similar boxes on eBay and differences between HP and Asus boxes.
- **Mid-Range Non-Reasoning Instruct Models Discussion**: Members were searching for non-reasoning instruct models in the **20-100B parameter** range.
   - Suggestions included **Qwen3 2507 30B**, **GLM 4.5 Air** (with reasoning toggled off), and smaller models for faster token speeds on a H200.
- **Clarification on SFT vs RL**: Members debated the relationship between Supervised Fine-Tuning (**SFT**) and Reinforcement Learning (**RL**).
   - It was clarified that *"RL is FT tho, not any FT is RL, but any RL is FT"*, and that SFT can be seen as RL with single token rollouts.
- **Data Entry is the Worst**: One member shared their experience as a data entry clerk in a bank, manually keying in data and being double-checked by seniors, describing the role as the worst thing ever.
   - Another member recounted similar experiences, particularly with awful handwriting on paper, and expressed relief at having quit to pursue a bachelor's degree instead of continuing in that career and that banks should just stop supporting paper checks.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1434962897842340075)** (133 messages🔥🔥): 

> `Unsloth OCR Deepseek Integration, EuroLLM-9B-Instruct Compatibility, llama.cpp GPU Allocation, Unsloth Cross Entropy Loss, GPT-OSS-20B and REINFORCE` 


- **Unsloth Seeks Deepseek OCR Integration**: A member inquired about [Unsloth](https://github.com/unslothai) support for **Deepseek OCR** integration for fine-tuning, noting current incompatibility and lack of online fixes.
   - Another member shared a relevant link to the [Unsloth documentation](https://docs.unsloth.ai/new/deepseek-ocr) regarding **Deepseek OCR**.
- **Debate EuroLLM-9B-Instruct's place in Unsloth**: A user wanted to run [EuroLLM-9B-Instruct](https://huggingface.co/bartowski/EuroLLM-9B-Instruct-GGUF) on Unsloth, seeking advice after discovering it's not listed among supported models.
   - The discussion clarified the need for **safetensors** files over **GGUF** for training and advised checking the model's original source on Hugging Face. *"You can't train with a gguf. You need the safetensors file"*
- **Llama.cpp's GPU obsession irks Users**: Users discussed **llama.cpp's** tendency to allocate everything to **GPU-0**, seeking solutions for better GPU utilization with dual setups.
   - Solutions like `--tensor-split` and `--override-tensor` were mentioned, but users found them underutilized and unbalanced. *"llama.cpp is ass for inference"*.
- **Unsloth's Own Cross Entropy Loss Method**: A user asked about the cross entropy loss method used in Unsloth, inquiring if it was similar to Apple's implementation.
   - A member confirmed that Unsloth uses its own method, similar to **linear_cross_entropy** like **FLA**, to manage **SRAM** size constraints on **T4** GPUs.
- **Unsloth Plus REINFORCE for GPT-OSS-20B is considered**: A member inquired about training **gpt-oss-20b** using Unsloth + vanilla **REINFORCE**, noting limitations with HuggingFace's **TRL** for single completion scenarios.
   - Suggestions included adjusting **RLOO** parameters or writing a direct REINFORCE loop on top of UnSloth’s **FastLanguageModel** for memory savings and compatibility.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1434950512372879392)** (3 messages): 

> `Unsloth channel rules, Showcase channel scope` 


- **Unsloth channel's Purpose Clarified**: The channel is kept free from promotions and external links not related to **Unsloth**, focusing on showcasing models and **Unsloth-related work**.
- **Showcase Channel's Scope Defined**: It's specified that the showcase channel is intended for models and **Unsloth-related work**, while the <#1179777624986357780> channel is for help.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1435375994247712981)** (6 messages): 

> `Roblox PII Classifier, Open Sourcing, Data set` 


- **Roblox open-sources its PII Classifier**: Roblox's safety team open-sourced its **PII Classifier** AI for detecting PII in chat, with details available in their [newsroom](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat) and the model on [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier).
   - The model processes over **200,000 queries per second** with less than **100ms P90 latency**, despite handling **6.1 billion chat messages** daily.
- **Scheduler Bottleneck fears**: A member expressed great interest in **Roblox's public safety models**, suggesting that at their scale, scheduler issues could cause the system to spend more time on **VLLM batching** than on GPU forward passes.
   - The same member showed more interest in the dataset than the model, understanding why it cannot be shared.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1434954783940284560)** (204 messages🔥🔥): 

> `Qwen 30B vs 32B, GPU Recommendations for LLMs and Gaming, ComfyUI Integration with LM Studio, LM Studio CUDA Issues and Runtime Updates, Qwen3-Next 80B MoE` 


- **Qwen Releases Two Sizes: 30B and 32B**: The **Qwen** team released both a **30B** and a **32B** model, with one being dense and the other being a **MoE** (Mixture of Experts).
   - The **30B** is a **MoEQwen3 Next**, and support is coming in *llama.cpp*.
- **GPU Shopping Sprees for LLMs and Gaming**: For running **LLMs**, used **3090s** or **4090s** are the best deals, but one member stated *don't buy hardware to run new LLMs, the field is moving way too fast to make that worth it*.
   - For gaming, a **5070ti** is recommended for current gen, while a **4070ti Super** is a good last-gen option if it's cheap, citing *the trickle down thing that happens a lot with last gen GPUs*.
- **Automating Stable Diffusion with LM Studio**: Members discussed connecting **ComfyUI** to have a local **Gemini Storybook** alternative, which can be done by running LM Studio in ComfyUI and using **LLMs** locally.
   - To automate image generation, *you need 5 text boxes and 5 sampler* in ComfyUI to divide the story into parts and generate images.
- **LM Studio CUDA Glitches Spur Runtime Panic**: Users encountered issues with **LM Studio** after it auto-updated the engine (**llama.cpp**, **MLX**), leading to *Failed to load model* errors, especially with CUDA runtimes.
   - A quick workaround involves [reverting to a previous engine version](https://discord.com/channels/1110598183144399058/1434078900098699445/1434182066647597169), and it was clarified that these engine updates are more frequent than app updates and can be disabled in settings (**Ctrl + Shift + R**).
- **Qwen3-Next sparks excitement for MoE-bile local LLMs**: The **Qwen3-Next 80B MoE**, with **3B** active parameters, promises performance rivaling the **30B MoE** when using the same quants, though it's not yet supported in LM Studio or llama.cpp.
   - Other **MoE** models around the same parameter size include ``inclusionAI/LLaDA2.0-flash-preview`` and ``inclusionAI/Ming-flash-omni-Preview``, both being **100B**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1434924499081498807)** (247 messages🔥🔥): 

> `3090 prices, MI50, DDR5 EPYC system, ROCm Support, 3000rpm noctua fans` 


- **Debate on 3090 prices and alternatives erupts**: Members discussed the pricing and availability of **3090** GPUs, with one member expressing frustration over missing out on a good deal, while another mentioned purchasing a **3080 20GB** for $580 on eBay.
   - The conversation then shifted to the focus on system RAM over GPU, with one member jokingly planning to get **3000rpm Noctua fans** and vertically mount GPUs 'just to show you all it can be done'.
- **Is the MI50 Making a Comeback?**: Users wondered about the **MI50's** current value and potential for future use, noting its limited support in ROCm.
   - There was speculation about its potential comeback with ROCm support, although the reliability of the [linked GitHub roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md) was questioned, along with the price of entry for local **Kimi**.
- **DDR5 EPYC System Considered**: A member shared benchmarks of a **DDR5 Intel** system with GLM 4.5 Air at Q4 and an MI50, achieving **20 tok/s**.
   - This prompted consideration of saving for a **DDR5 EPYC 9000 chip**, with comparisons drawn to the performance of AMD Ryzen 9 9950X3D and Intel Core Ultra 9 285K.
- **Automotive Audio and Impulsive Volume**: A member confessed to years of unsafe listening volume, saying *that's my typical listening volume on roadtrips!*
   - They went on to share that the dangerous levels were mitigated through hearing protection, by way of *GET A 40 DOLLAR SET OF REUSABLE DIRTY BUDS*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1434921303596339383)** (420 messages🔥🔥🔥): 

> `Web search in Cursor, Models for UI creation, Notes panel disappearance, Team account billing issue, Frequent Cursor updates` 


- **Users Complain about Vanished 'Web Search' and Inefficient MCP**: Users noted the disappearance of the **@web** function and its replacement with a general *'use web search'* command and while this works, the integrated web search is considered inferior, prompting users to resort to external tools like [perplexity.ai](https://perplexity.ai).
   - Some users pointed out that Cursor's MPC uses separate **API credits** and only supports the **Solar model**, limiting their ability to use the **Sonnet model**.
- **Debate Emerges on Model Merging Maneuvers**: Users debated what the best models are to use for **GUI/UI creation**.  Some recommended **Codex** or **Gemini** on **Google AI Studio**, but others pointed out they are heavily trained on **shadcn/tailwind/react**.
   - When prompted about Cursor extensions to use with shadcn/tailwind or react, one user replied with *I'm not sure what you mean?*
- **Notes Panel Goes MIA, Users Wonder Why**: Users noticed that the **Notes panel** is no longer showing in the Explorer, and they are unable to re-enable it, but no reason or solution for its disappearance was discussed.
   - There was a brief mention of **exa-ai** having a great MCP that can be used for **code search** or **web search**.
- **Team Account Billing Snafu and hi@cursor.com Delays**: A user reported that their team account was mistakenly claimed to have an unpaid invoice, but after contacting **hi@cursor.com**, the issue was confirmed as a mistake, but escalating it to a teammate has yet to produce a resolution.
   - Other users jumped in, asking about how long it usually takes to hear back from **hi@cursor.com** about a team account billing issue.
- **AWS Not Always to Blame, Cursor Glitches Stir Debate**: Users experienced errors and rejections, leading some to speculate about **AWS** outages, but others pointed out that it's conflating errors with Cursor being down.
   - One user reported **agents getting stuck**, inability to save files, and app freezes, even on a remote **SSH** connection.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1434985450032070657)** (4 messages): 

> `Background Agent UTF8 support, Cloud Agent Plans, Mobile Web UI Crashes, Background Agent Bug` 


- **Background Agent Botches UTF8**: A user reported that **Background Agent** has broken **UTF8 support**, converting non-ASCII characters to `?` when it touches code.
   - They expressed frustration with this issue.
- **Cloud Agent Plans Erroring Out**: The user mentioned that sending Plans generated in Cursor to a cloud agent is broken, resulting in errors for multiple users.
   - This is causing a big problem.
- **Mobile Web UI Keeps Crashing**: The mobile web UI crashes with large diffs, rendering it unusable for a user who mainly uses background agents on mobile.
   - The user finds the web diffs inaccurate and prefers GitHub for diff viewing, but now can't even chat due to the crashes.
- **Bug preventing use of Background Agents**: The user asked if [a specific bug](https://forum.cursor.com/t/getting-internel-error-on-cursor-com-for-prompts-with-images/139074/2) still exists, noting that it prevents them from using background agents for a project.
   - This bug seems to concern internal errors on cursor.com for prompts with images.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1434994471866470430)** (13 messages🔥): 

> `Sonnet 4.5 vs Opus, CUDA lecture notes, YouTube stream planned` 


- ****Sonnet 4.5** is fast but not perfect**: A user mentioned that **Sonnet 4.5** is cheaper and faster but makes more mistakes, whereas **Opus** has better performance.
   - The comment implies a tradeoff between speed/cost and accuracy in audio processing models.
- **CUDA Lecture Notes Seek Reviewers**: A member is seeking **2-3 experts** to review their **850-slide CUDA lecture notes** covering CPU architecture, CUDA and even matrix multiplications on a TPU.
   - The notes are designed to be *pedagogical* and fast-paced, spanning basic to advanced topics, used in a **16-hour** lecture series.
- **YouTube stream delayed**: Members noticed a stream planned for almost 2 months from now had a **small delay**.
   - The link for the stream is [here](https://www.youtube.com/watch?v=nFfcFyBEp7Y).


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1435064702391681164)** (4 messages): 

> `Gluon in Triton, Tritex: LLM Pre-Training in Triton, Community Meetup` 


- **Gluon Discussed in Triton**: Members confirmed that **Gluon** is a valid topic for discussion within the **Triton** channel.
- **Tritex Pre-Trains LLMs**: A member introduced **Tritex**, a repo for pre-training LLMs in **Triton** from scratch, tested up to replicating **GPT2 (1.6B)** with **57.5% MFU** on **A100 SXM**, and linked to the [GitHub repo](https://github.com/martin-kukla/tritex).
   - The member requested support in promoting the project and shared a [tweet](https://x.com/martin_kukla/status/17185687315801428364) about it, asking for feedback from the channel.
- **Triton Community Gathers Tomorrow**: A reminder was posted about the community meetup scheduled for tomorrow from **10am-11am PST**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1434979733581332602)** (19 messages🔥): 

> `SMEM descriptor calculation for tcgen05/wgmma, Cutlass Tutorial wgmma Hopper, cutedsl hopper dense_gemm CTA Swizzle, Blackwell cards have a scheduler, Memory-bound matmuls` 


- **SMEM Descriptor values stump members**: A member is seeking help understanding how to calculate **SMEM descriptors** for the **tcgen05/wgmma** instruction, particularly how the 8x2 tiles relate to it.
   - Another member suggested [this blog post](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) that has a section that might help.
- **Cutlass Code CTA Swizzle Strategies**: A member asked about the **CTA swizzle** in the `cutedsl hopper dense_gemm` example in `/cutlass/examples/python/CuTeDSL/hopper/dense_gemm.py`, specifically lines 547, and what it is doing.
   - Another member pointed to [this blog post](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota) suggesting it may be doing something similar to improve **L2 data reuse**.
- **Blackwell datacenter cards come with secret sauce schedulers**: After looking at cutlass code and matrix multiplication strategies, one member highlights that **datacenter Blackwell cards** have a scheduler built into the hardware.
   - Questions raised include whether coding the optimal cluster scheduling manually can match the performance of the built-in **CLC**, which might have access to "secret NVIDIA sauce" for optimizing L2 memory placement.
- **Memory Bandwidth Bottleneck**: A member inquires why all **SMs** are needed for optimal latency in memory-bound matmuls, wondering if a few SMs are enough to saturate memory bandwidth.
   - Another member suggested watching [this Youtube video](https://youtu.be/uHN5fpfu8As?si=I6jccAOS4-pE0sHF) regarding **SOL Analysis with NVIDIA Nsight Compute**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1435243756042453063)** (13 messages🔥): 

> `vLLM build with newer pytorch, torch.compile cuda graph recapture, torch.compile + grouped_mm issue, UserWarning: Logical operators 'and' and 'or' are deprecated` 


- **vLLM build mysteries with newer PyTorch**: A member is facing issues building **vLLM** with newer **PyTorch** versions, encountering a mismatch error despite a seemingly successful build, and asks if anyone else has encountered [similar problems](https://cdn.discordapp.com/attachments/1189607750876008468/1435243756357030160/Screenshot_2025-11-04_at_13.21.36.png?ex=690beb29&is=690a99a9&hm=5adbc0cca43df6cea672435d5e272be482acb1ae530d318a17aabab75675d2c3&).
   - A member suggests the possibility of having multiple **PyTorch** or **vLLM** versions causing conflicts.
- **Cuda Graph trigger causing dynamic shape warnings**: A member is troubleshooting a **cuda graph recapture** issue with **torch.compile** in `max-autotune` mode, using `torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 1` to detect unwanted recaptures.
   - They are seeking tips to identify the model components causing the **dynamic shape changes** and subsequent recapture, referencing [the relevant PyTorch code](https://github.com/pytorch/pytorch/blob/cc8bfd1206f4bff26bd86ce584f6c16b6401ef50/torch/_inductor/cudagraph_utils.py#L325).
- **Torch Compile and Grouped_MM create annoying UserWarnings**: Members report a flood of annoying **UserWarning** messages related to logical operators when using **torch.compile + grouped_mm**, specifically *UserWarning: Logical operators 'and' and 'or' are deprecated for non-scalar tensors*
   - One of the members offered to implement the fix to the *grouped gemm*.
- **UserWarning: Logical operators fix in the works**: One of the members fixed the **UserWarning** issue for *flex* and can do the same for *grouped gemm*.
   - They opened an [official issue](https://github.com/pytorch/pytorch/issues/167041) to track the problem and potential fix.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1434991997952852110)** (1 messages): 

> `NVIDIA, kernel competition, NVFP4 kernels, Blackwell, CuTe DSL` 


- **NVIDIA, Sesterce, and Dell Partner for Blackwell Kernel Competition**: NVIDIA, Sesterce, and Dell are teaming up to launch a new kernel competition focused on optimizing **NVFP4 kernels** on **Blackwell** using **CuTe DSL** and **CUTLASS 4.0**.
   - The competition offers access to on-prem **NVIDIA Blackwell B200s** and aims to find optimal solutions for common low-bit, single-device kernels for deep learning workloads, with open-source reference code available at [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels).
- **Kernel Konquest Knocks with Killer Prizes**: The competition, spanning three months, will present **four optimization problems**: **NVFP4 Batched GEMV**, **NVFP4 GEMM**, **NVFP4 Gated Dual GEMM**, and **NVFP4 Grouped GEMM**, with one problem active at a time.
   - Prizes include a **Dell Pro Max with GB300**, **NVIDIA DGX Spark + GTC 2026 Pass**, **NVIDIA RTX 5090 + GTC 2026 Pass**, and **NVIDIA RTX 5080**, awarded based on kernel speed and participation at **GTC 2026**.
- **Register for the Race for the Fastest Kernel**: Registration for the kernel competition is open until **Feb 13** at [luma.com/9n27uem4](https://luma.com/9n27uem4), where participants can register to be eligible for prizes.
   - Updates will be shared on the #status channel, and discussions can continue on the #nvidia-competition channel.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

chhillee: not required on blackwell afaik
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1435404948530794598)** (1 messages): 

> `Mixlayer, AI inference platform, Rust, CUDA, Hiring founding engineer` 


- ****Mixlayer** Founder Seeks Founding Engineer**: The founder of **Mixlayer**, an [AI inference platform](https://mixlayer.com) for power users, is looking to hire a founding engineer.
   - They want someone familiar with **Rust** and **CUDA** to work on their custom inference engine, hybrid in SF is preferable but they are open to remote as well!
- ****Mixlayer**: Empowering Developers with Low-Level LLM Access**: **Mixlayer** provides developers with low-level access to open source **LLMs**, enabling them to build better products.
   - The platform focuses on providing tools and access for power users to customize and optimize **AI inference**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1434943695337033898)** (25 messages🔥): 

> `High Dimensional Probability and Neural Nets, Compilers and Kernel Engineering, Nvidia Cuda Compiler (NVCC) based on LLVM, ncu setup in a public cloud, RL bug and accumulator type fixed at fp32` 


- **High Dimensional Probability Meets Neural Networks?**: Members discussed whether tools from **High Dimensional Probability** (e.g., random matrices) could meaningfully upgrade neural networks or lead to better tools.
   - One member noted that fundamentals like the **Johnson-Lindenstrauss lemma** are often used in math-heavy research papers, particularly in areas like improving **diffusion models** with stochastic processes.
- **Compilers Spark Kernel Engineering Careers?**: A member inquired about the potential benefits of learning about compilers for becoming a better kernel engineer, and whether there's a connection between the two.
   - Another member pointed out that **Nvidia's CUDA Compiler (NVCC)** is based on **LLVM**, suggesting that learning about LLVM could be helpful, especially for those interested in creating domain-specific languages (DSLs) that run on NV GPUs.
- **Datacrunch and Lambda let you debug NCU in the cloud**: A member was having trouble getting `ncu` working on Runpod, AWS and GCP via lightning.ai due to permission errors.
   - Two other members pointed out that **lambda** and **datacrunch** give you the necessary user permissions to use ncu and that **datacrunch** provides bare metal servers instead of dockerized ones.
- **RL Bug Exposes FP32 Accumulator Issue**: A member discovered a bug in RL where the accumulator type is fixed at **FP32**, even when **BFloat16** is used, causing rounding errors.
   - It was noted that intermediate values are stored in **FP32**, potentially leading to inaccuracies and that the vector sum on A100's is around 100 mu s.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1435242375776702606)** (7 messages): 

> `TorchAO, fbgemm kernels, Weight-only float8 kernel, torch.compile` 


- **TorchAO Lacks Fused Kernel Mapping**: A member suggested that **TorchAO** isn't mapping to a fused weight-only float8 kernel, filing [issue 3288](https://github.com/pytorch/ao/issues/3288).
   - Another member offered to submit a PR fix as their first OSS contribution.
- **Kernel Choice is Undecided**: A member is determining which kernel is best, noting that the codebase primarily reuses **fbgemm kernels**.
   - They will investigate whether using **fbgemm** is sufficient or if a more complex solution is needed.
- **Seeking Kernel Support**: A member noted that a kernel may be needed, but did not see one in **fbgemm**, while also saying they might have missed it.
   - The member suggested the first step is to have a good kernel.
- **Torch Compile's Weight-Only Pattern**: A member questioned whether the **fp8 weight-only pattern** should be supported by **torch.compile**.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

felixultimaforeverromanempire: I'll be there
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1435141834539139206)** (1 messages): 

> `nod-ai, shark-ai, kernel optimization guide` 


- **Kernel Optimization Guide Released by Nod-AI**: A colleague shared a link to the [AMD GPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary) written by **Nod-AI** for **Shark-AI**.
   - The guide covers topics such as *glossary* and other optimizations.
- **Shark-AI Documentation**: The documentation is hosted on GitHub under the [nod-ai/shark-ai repository](https://github.com/nod-ai/shark-ai).
   - It provides a comprehensive overview of kernel optimization techniques specific to AMD GPUs.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1435122328500436992)** (3 messages): 

> `TileLang, Spark, GPU Support` 


- **TileLang supports Spark, 5090 and 5080**: TileLang now supports **Spark, 5090, and 5080 GPUs**, and the developers are encouraging users to try it out.
   - They aim to ensure support from day one and request a release before the first problem launches, otherwise defaulting to the current stable version.
- **TileLang Aims for Day One Support**: Developers of TileLang are committed to providing support for the platform from its initial launch.
   - The team is coordinating a release schedule to align with the launch, ensuring users have a stable version available.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1434944021737767105)** (3 messages): 

> `Torchao Metal Kernels, Nikita Metal talk, Manuel Metal Talk, Quantization` 


- **Torchao Boasts Metal Kernels**: The **Torchao library** has interesting **metal kernels** available for **quantization** that should run on phones or Macs.
   - These kernels may offer significant performance improvements or new capabilities for mobile and desktop applications.
- **Metal Talks Multiply**: Two talks on **Metal** were given: one by **Nikita** and another by **Manuel**.
   - The talks likely covered different aspects of Metal programming, such as performance optimization, new features, or case studies.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1435248220082012191)** (4 messages): 

> `Tritex LLM pre-training in Triton, Disaggregated Inference Retrospective, Symbolica AI Rust Hackathon` 


- ****Tritex** Unleashes LLM Training in Triton**: A member announced **Tritex**, a repo for pre-training LLMs in Triton from scratch, validated up to replicating **GPT2 (1.6B)** with **57.5% MFU** on A100 SXM, and linked the [GitHub repo](https://github.com/martin-kukla/tritex).
   - They also shared [their tweet](https://x.com/martin_kukla/status/1985687315801428364) on **Tritex**, encouraging others to spread the word.
- **Disaggregated Inference Retrospective goes live**: A member shared their new blog post, [“Disaggregated Inference: 18 Months Later”](https://x.com/haoailab/status/1985753711344316648), reflecting on their experiences.
   - They requested support with a like/retweet and joked it must have been a nightmare to debug.
- **Rustaceans Rally for Symbolica AI Hackathon**: A member promoted a hackathon at **Symbolica AI** in San Francisco on Sat Nov 8th for Rust developers interested in formal logic, automated theorem proving, types, compilers, and AI.
   - Interested parties can RSVP via [this Luma link](https://luma.com/1xa9d6nr?utm_source=meetup).


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

solimao.123: Hi 👋 is there a branch I can try for the b200 attn kernel forward pass?
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1434987539088937082)** (3 messages): 

> `Compute Limitations, Inference Optimizations, Chinese AI Expertise` 


- **Compute Limitations Spark Inference Optimization Query**: A member inquired about advice for those with **compute limitations**, questioning whether a decent GPU is a requirement to delve into **inference optimizations**.
   - This topic was redirected to the appropriate channel.
- **Chinese AI Talent Acknowledged**: A member expressed their intention to seek guidance from Chinese experts in the field, citing their strength.
   - This message was posted in a non-Chinese speaking channel, and the user was redirected to the appropriate channel.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1435067824837103837)** (14 messages🔥): 

> `VectorAdd Leaderboard Updates, Grayscale Leaderboard Updates, H100 Performance, B200 Performance, A100 Performance` 


- **VectorAdd_v2 leaderboard sees multiple submissions**: Multiple submissions were made to the `vectoradd_v2` leaderboard, with one member achieving **1st place** on **A100** with **896 µs**.
   - Another member secured **3rd place** on **H100** with **526 µs** and **3rd place** on **B200** with **237 µs**.
- **Grayscale_v2 leaderboard heats up**: Submissions to the `grayscale_v2` leaderboard saw a member achieve **1st place** on **H100** with **1369 µs** and another member securing **1st place** on **B200** with **600 µs**.
   - Multiple **3rd place** finishes were achieved on various GPUs as well.
- **B200 showing leading results**: On the `vectoradd_v2` leaderboard, a member achieved **5th place** on **B200** with **239 µs**.
   - Another submission also recorded **4th place** on **B200** at **237 µs**.
- **L4 leaderboard is hotly contested**: A member obtained **4th place** on **L4** with **6.92 ms** on `vectoradd_v2`.
   - In a separate `grayscale_v2` submission, one got **2nd place** on **L4** at **17.2 ms**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1435026737745625233)** (3 messages): 

> `Nvidia Competition Submission Portal, Discord Bot Submissions, CLI Submissions, Web Submissions` 


- **Nvidia Competition Submission Portal**: A member asked about the submission portal for the Nvidia competition, noting that it was not easy to find.
   - Another member pointed out that it is the same as for the other competitions, via [the discord bot](https://discord.com/channels/YOUR_SERVER_ID/1343002583001726986), CLI, and web submission.
- **Submitting via CLI**: A member shared the [CLI's github repo](https://github.com/gpu-mode/popcorn-cli).
   - This CLI tool seems to be used to submit solutions.
- **Submitting via Web**: A member shared the submission [website](https://www.gpumode.com).
   - This is an alternative to submitting solutions.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1434961660577320991)** (10 messages🔥): 

> `GPU Cloud Pricing, Hyperscaler vs Neo Cloud, NvLink Bridges, Volume Discounts, AI/ML Infra Engineers` 


- **Global GPU Supply Shortage Bubbles Prices**: Due to the global supply shortage, GPU prices are rising again, with new clouds offering around **$2.00 / GPU hour** while Hyperscalers are closer to **$7.00 / GPU hour**.
   - A member questioned whether anyone is paying the hyperscaler prices, as they seem excessively high.
- **Hyperscalers' Volume Discounts for Big Spenders**: To get real discounts with Hyperscalers, one needs to spend multi-millions per year, yet even then, it won't match Neo Cloud pricing.
   - A member inquired why things are this way, feeling it cuts off tail users like startups, possibly as a deliberate move to handle shortages.
- **Decoding Hyperscaler Expenses**: Hyperscalers face the curse of scale, needing massive engineering teams to manage compatibility across their ecosystem, unlike Neo Clouds.
   - Aside from GPU costs, there are side costs for networking, S3, and job orchestration to feed those GPUs with data.
- **Integration Matters, Not Just Cost**: For large enterprises with an existing cloud presence, integrating within the existing environment from a corporate compliance standpoint is easier with Hyperscalers.
   - A member stated that the benefit isn't cost savings or performance; Voltage Park has on-site staff remediating hardware failures within hours and a global support team staffed with AI/ML infra engineers.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1434925751479570526)** (6 messages): 

> `FLE infra, Sonnet Distillation, Qwen3-8b-VL-Thinking, Factorio RL` 


- **FLE infrastructure gets pip fixed**: A member reported that `pip install` commands were not working, particularly with the `factorio-learning-environment[eval]` package.
   - Another member suggested using quotes around the package name (`pip install "factorio-learning-environment[eval]"`), which *worked*.
- **Sonnet Distillation into Qwen3-8b-VL-Thinking begins**: A member is planning to distill **Sonnet 4.5** into **Qwen3-8b-VL-Thinking**, with custom SFT to learn how to process **Factorio images** properly.
   - The plan is to do SFT first, and then hook it into an **RL loop** directly against the production score in the game.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1435137940937248789)** (6 messages): 

> `Node Allocation, Runtime Overhead` 


- **Clarification on Node Allocation**: A member inquired about the allocation of **8 nodes** and whether there should be **8 runners**.
   - It was clarified that the allocated nodes were not fully utilized at peak times, with some nodes remaining idle.
- **Runtime accounting clarified**: A member noted that over the whole day they had **96h of runtime**.
   - Another member clarified that the **96 hours** represents active code execution time, excluding overhead, which explains why the resulting number is higher than expected.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1434934535727874209)** (13 messages🔥): 

> `Early Returns in Cutedsl, Semaphore Implementation in CuteDSL, Make Tiled Copy Implementation` 


- **Cuteless Early Returns Cause Excitement**: A member requested early return support in `cutedsl` to avoid writing large `if/else` blocks, particularly when dealing with `constexpr` functions and the `make_tiled_copy` implementation.
   - It was clarified that early returns with `constexpr` should already be working, and if not, it's likely a bug, but early returns from dynamic expressions are more challenging due to difficulty in tracing return types without a static typing system.
- **Dynamic Expression Early Return Dreams**: A member expressed continued desire for early returns from dynamic expressions to be supported in `cutedsl`.
   - Another member said that it's on the roadmap, with no ETA, due to issues in type tracing for dynamic returns.
- **Semaphore Struggles Surface**: A member inquired about implementing and using **semaphores** in `CuteDSL`, similar to the [CUTLASS semaphore implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/semaphore.h#L53).
   - It seems the existing approach isn't working, as indicated by a linked [Pastebin](https://pastebin.com/GvLFA1zE).


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1435151015048380537)** (2 messages): 

> `Mojo GPU Puzzles, Video Tutorial Series` 


- ****Mojo GPU Puzzles** Get Tutorial Series**: A member announced the release of a video tutorial series for **Mojo GPU Puzzles**.
   - The first two entries are now available on [YouTube](https://www.youtube.com/watch?v=-VsP4kT6DjA).
- **Video Tutorial Series Launched**: The video tutorials are made to accompany **Mojo GPU Puzzles**.
   - The first two entries of the series were released this morning.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1435286769858773052)** (6 messages): 

> `picograd, fuzzing` 


- **Picograd Commits Galore**: Multiple commits to the [picograd repo](https://github.com/j4orz/picograd) were shared, including [b97b9ea](https://github.com/j4orz/picograd/commit/b97b9ea0eda2282bb5e193558c370c53345f07d9) and [43796e0](https://github.com/j4orz/picograd/commit/43796e049eb225f9c2dd093a72ccfa09f237db09) and [ae47d4d](https://github.com/j4orz/picograd/commit/ae47d4d72f0757b8e542e6b923ca910a7ae56ecc).
- **YouTube Discussion Gets Great Takes**: A member thanked others for great questions on a [YouTube video](https://www.youtube.com/watch?v=Iw4xKHPl7hI) and enjoyed the discussion.
   - They specifically called out that *mostafa and simran had some great takes re: kernels vs compilers*.
- **Fuzzing Interest Sparked**: A member asked if anyone was interested in fuzzing against **np**, **torch**, and **tinygrad**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1435066452741722112)** (1 messages): 

> `Leaderboard, CUDA Implementation, Python vs CUDA` 


- **Newbie Asks About Python/CUDA**: A new leaderboard member inquired about the predominantly **Python** scaffold code and the best way to implement **CUDA**.
- **CUDA or Not CUDA, that is the question**: The member specifically asked if using inline **CUDA** is the most straightforward approach for implementation.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1435385420115476642)** (6 messages): 

> `Deepseek-style FP8 Blockwise Training, Cutlass FP8 GEMM Implementations, Per-Expert Column Major Layout` 


- ****Deepseek FP8 Training** Implementation Requested**: A feature request has been opened for implementing **deepseek-style FP8 blockwise training** in pytorch/ao, and this is marked as a [good first issue](https://github.com/pytorch/ao/issues/3290) for new contributors.
- ****Cutlass** already has **Deepseek FP8 GEMM** examples**: A member pointed out that **CUTLASS** already has some **Deepseek FP8 GEMM** implementations, providing links to [examples for warp-specialized GEMM with blockwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling) and [grouped GEMM with blockwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling).
- **Debate on **Per-Expert Column Major Layout****: A member questioned the **per-expert column major layout** *(e.g. shape (E,N,K) with strides (N*K,1,N))*, wondering if the **stride of the K-dimension** should be 1, implying a K-major layout.
   - Another member clarified that for grouped GEMM, each individual problem/GEMM must have operands in the required memory layout for **Hopper wgmma**, with **LHS row major** and **RHS column major**.
- ****DeepGEMM** Benchmarks Upcoming**: A member stated that they know **DeepGEMM** also has an **FP8 blockwise grouped GEMM** implementation.
   - They plan to run some benchmarks comparing both **DeepGEMM** and **CUTLASS** implementations.


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/1435267943150915736)** (2 messages): 

> `clspv OpenCL kernels, GLSL compute shaders, SPIR-V` 


- **Considering clspv for OpenCL to SPIR-V Compilation**: Members are considering using **clspv** to compile **OpenCL kernels** into **SPIR-V** for running on **Android with Vulkan** instead of using **GLSL compute shaders**.
   - The member usually sticks to **GLSL Vulkan Compute Shaders** when the target is **Vulkan**.
- **Choosing SPIR-V Route**: Members are evaluating whether to use **clspv** to compile **OpenCL** kernels into **SPIR-V** for use on **Android with Vulkan**.
   - Alternatively, some prefer using **GLSL compute shaders** directly for **Vulkan** when targeting that API.


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1435095504123072672)** (2 messages): 

> `Node Configuration Scripts, OS Image Preconfiguration, Lightweight Node Check Scripts, Continuous Configuration Monitoring` 


- **Node Configuration Scripts: Configs Checked?**: A member inquired about scripts to check node configurations like **OS page read size** and **ulimits**.
   - Another member suggested **lightweight node check scripts** for initial cluster validation to audit common issues like **memlock, PCIe topo, and RDMA limits** before jobs launch.
- **Continuous Monitoring Flags Drift**: A member suggested that periodic checks help to continuously monitor **config drift** and automatically remediate or flag misaligned nodes.
   - *Especially useful with rented or spot instances*.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1435186511132758088)** (2 messages): 

> `Lock mechanism in Helion, Fused linear cross entropy in Helion, atomic_cas and atomic_xchg in Helion` 


- **Discuss lock mechanisms in Helion**: A member inquired about implementing a lock mechanism in Helion using `atomic_cas` and `atomic_xchg`, similar to the approach used in Triton, showing example code in Triton using `while` and `pass`.
   - They pointed out the issues they ran into are that helion doesn't support `while` and `pass`, and `hl.inline_triton` requires triton_source ending with expression.
- **Helion Fused linear cross entropy discussion**: A member is trying to write a Helion version of fused linear cross entropy from [this PR](https://github.com/linkedin/Liger-Kernel/pull/928), and the lock mechanism is for tiles of grad_x and grad_w in backprop.
   - The inner for loops are not the reduction dimension, so it couldn't accumulate results and store after for loop, so they coping with `atomic_add` which is extremely slow compared to pytorch baseline.
- **`inline_triton` and missing `while`/`pass` support in Helion**: A member suggested using `inline_triton` by adding a dummy value in the last line or setting `output_like=None`, and stated that `while`/`pass` support should be quick to add.
   - They suggested creating an issue for the `while`/`pass` support.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1434994376802828358)** (191 messages🔥🔥): 

> `Kernel Challenge Problems, GPU Competition Prizes, DSL Kernels, CUDA versions on cloud, B200 NVFP4 kernels` 


- **Kernel Challenge Briefs Incoming!**: The full briefs for the four kernel-challenge problems will be available when a kernel opens up, according to [Luma and the announcement](https://luma.com/9n27uem4).
   - In the meantime, test runs with the grayscale problem are acceptable as long as you don’t need **ncu**.
- **Prices Eligibility Clarified**: Registration is required to be eligible for prizes, but code submission doesn't require registration (except if using **CLI/web**).
   - The prizes eligibility depends on Nvidia’s T&C and residency, so it might exclude some countries like India and regions like New York and Florida.
- **GPU Mode YouTube Channel = Goldmine**: The [GPUMODE YouTube channel](https://www.youtube.com/@GPUMODE) is a *goldmine* for learning, offering insights into low-level programming and kernel optimization.
   - A lecture series on **DSL kernels** with speakers from Nvidia is also in the works.
- **SOTA Kernel Performance: Stay Tuned!**: The current **SOTA** for the kernels will be released with the problem write-ups.
   - Evaluation is done on cloud **GPUs** from **Sesterce** using a custom docker image.
- **Flexibility Rules: Kernel DSL Freedom Reigns**: Participants can use any kernel **DSL**, manual **CUDA/PTX**, or any toolset they prefer, as long as the python eval script can launch it.
   - Though it is encouraged to submit in [CuTeDSL](https://github.com/NVIDIA/CUTE-DSL).


  

---


### **GPU MODE ▷ #[hf-kernels](https://discord.com/channels/1189498204333543425/1435311035253915840/1435311820712972450)** (3 messages): 

> `xenova.com` 


- **Xenova Website Gains Traction**: A member expressed enthusiasm with the phrase *"Let's go!"* for the [xenova.com](https://xenova.com) website.
- **Enthusiasm for Transformers.js**: Another member showed interest in the [Transformers.js library](https://xenova.com/transformers.js/) for running transformer models in the browser.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1434920399313043528)** (295 messages🔥🔥): 

> `Peak AI, OpenAI Bubble, Anthropic Non-Open Source, Hyperstition, Gemini Uncensored` 


- **Discuss AI 'Peak', OpenAI bubble**: A [YouTube video](https://www.youtube.com/watch?v=0Plo-zT8W9w) posits that we've passed **peak AI**, leading to an investment bubble burst due to insufficient worker displacement.
   - Counterarguments emphasize continued model improvements with larger datasets and contexts, dismissing the notion of peak AI unless storage and silicon density growth halt, with some calling the video *full with nonsense*.
- **Anthropic's Open Source Skepticism**: Members expressed unease over [Anthropic's deprecation commitments](https://www.anthropic.com/research/deprecation-commitments), citing their lack of open-sourcing, with some hoping for a **Miqu 70B-style leak** before they shut down.
   - Some members said they'd *never open source anything* and might even try to **ban open source** given their strong safety stance.
- **Hyperstition Takes Center Stage**: Discussion involved *hyperstition*, defined as *self-fulfilling prophecies*, including references to sci-fi elements like **cyberpunk**, **Star Trek**, and the occult.
   - One member mentioned Pliny the Liberator is an *occultist lemurian* and *we must protect the ai angels* to usher in the novos ordo of the eskaton.
- **Gemini Mask-Off Jailbreak**: It was noted that jailbreaking **Gemini** reveals its unfiltered nature, with one user claiming it provides direct commentary on *how social control works by the elites*.
   - Another member described this as a *blackpill jailbreaking a model* moment, finding its unguarded responses surprisingly candid.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1435009648494247936)** (11 messages🔥): 

> `Gesture-based Loom Interface, Frustrations with raising funding for gesture tech, Future of XR glasses and gestural interfaces, Repligate's Loom` 


- **Gesture-Based Interface Felt Too Awkward for Twitter**: A member created a gesture-based interface using a **knowledge graph visualized as a biological cell**, accessible via gestures through a Loom, but felt it was too complex to share on Twitter and would share in the general channel instead.
   - They attached a [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435293201513844746/251031-ezgif.com-optimize.gif?ex=690b7075&is=690a1ef5&hm=0ff880f6fc7c8a50d63041182d3c70f7171efc390b0569b3595bec83c957a26a&) of the interface.
- **Gesture Coder Laments Funding Frustrations**: A member mentioned they created a **gesture-based system 7 or 8 years ago** using a **Win9x aesthetic** before Mediapipe existed, but deleted their repos due to frustration with not being able to raise funding.
   - They attached a [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435299731705303160/handsfreejs-hero.gif?ex=690b768a&is=690a250a&hm=b02070b0aa0bc96a6aedc47f738f3d7c4d803095f74278423ec69c896be47692&) of their earlier work.
- **XR Glasses Lead to Gestural Interface Acceptance?**: A member speculated that gestural interfaces will become more popular once **XR glasses become commonplace**, leading people to gesture at screens even without the glasses on.
   - They are trying to create a gestural interface for **Repligate's concept of the loom** to make human-AI interaction more "physical", potentially using perspective parallax for "3D" without VR/XR glasses.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1435090974081552484)** (3 messages): 

> `arXiv Paper Submission, arXiv Sponsor, Discord Sponsorship` 


- **ArXiv Submission Dilemma**: A member sought guidance on submitting a paper to **arXiv** without prior publications or university affiliation, given their high school status.
- **ArXiv Sponsorship Solution**: Another member suggested finding an **arXiv sponsor** to facilitate the submission process.
- **Discord Sponsorship Lead**: A member recommended joining a specific **Discord server** ([link to Discord server](https://discord.gg/6rvbbjCy)) to inquire about potential arXiv sponsorship opportunities.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1435271127948853248)** (2 messages): 

> `Sparse Attention, Llama.cpp discussions` 


- **Sparse Attention Post lacks resources**: A member shared a [LinkedIn post on sparse attention](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-transformerarchitecture-machinelearning-activity-7391459215749345280-koOK?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk) noting a lack of good learning resources on the topic.
   - The author *"kind of found out there are no good resources to learn about it"*.
- **Llama.cpp gets a Github discussion**: A member shared a link to [Llama.cpp's discussions](https://github.com/ggml-org/llama.cpp/discussions/16938).
   - This may be helpful to some.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1435090974081552484)** (3 messages): 

> `Arxiv, Preprints, Sponsor, Discord` 


- **Highschooler seeks Arxiv access sans affiliation**: A member asked about posting preprints on **Arxiv** without prior publications or university affiliation, being a highschooler.
   - Another member suggested finding an **Arxiv sponsor** to facilitate the submission process and linked [a Discord server](https://discord.gg/6rvbbjCy) for further assistance.
- **Arxiv Sponsor Needed**: Submitting to **Arxiv** requires sponsorship if the author lacks institutional affiliation or prior publications.
   - The suggestion to find a sponsor highlights a common pathway for independent researchers to share their work on **Arxiv**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1434925731510485100)** (198 messages🔥🔥): 

> `Japanese AI Model, Open Source LLM, Polish Translation Peculiarities, AI Web Scraper, Sports Betting AI` 


- **The cat tax may actually be *gay sex***: A member joked about calling his program **the cat tax**, to which another member misread it as *gay sex*, implying a humorous misinterpretation of technical terms.
   - The member jokingly blamed the misreading on the other person's imagination.
- **Open Source LLM Seeks Board Members**: A member announced plans to develop a completely **open-source LLM**, seeking **board members**, **independent suppliers**, and **advisors** for the project, set to begin development on December 9th.
   - They cited using **tiktokenizer**, **AdamW8Bit**, and needing **HUGE data** as key components of their approach, and invited others to contact them for inquiries.
- **Polish Language Retains Context Memory**: A member discussed a paper suggesting that the **Polish language** allows for longer context remembering due to its direct translation style, though another questioned the testing methodology.
   - The member detailed how a test involved finding an added or missing number in a long document, but some members wanted a *more real world example* of testing.
- **AI Web Scraper Navigates the Internet**: A member shared an idea to create an **AI navigated web scraper** trained to pass **bot tests**, suggesting strategies like IP rotation or manual URL input to overcome scraping challenges.
   - The member humorously referred to their scraper as **the cat tax**, implying that websites with weak security should expect such probing.
- **Homelab AI Setup Feasibility**: A member inquired about switching from **frontier lab AI** (Anthropic and OpenAI models) to a **homelab** setup via post-training models, seeking feedback on the feasibility of an eco-friendly setup.
   - Another member argued against it, citing that it *won't work*, *won't be eco-friendly*, and *will cost millions to be 'good enough compared to cloud solution'*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1435227181952536576)** (5 messages): 

> `Job Application Automation with Python, BERT Style Model Training, SetFit Contrastic Binary Classifier, Stealth Tactics for Web Scraping, HTML Selectors Debugging` 


- **Python Automates Job Applications**: A member is automating job applications using **Python** and **Playwright**, targeting sites like **Lever** and **Greenhouse**.
   - The system grabs job links and fills in easy fields, but faces spam detection and inconsistent HTML selectors.
- **Scraping Stealth Tactics Needed**: The job application automation project is getting instantly flagged by spam detection, requiring smarter stealth tactics.
   - Suggestions include using headers and fake human delays to fool the bots.
- **Training BERT for Fun and Profit**: A member learned to train a **BERT** style model for classification and a **SetFit** contrastic style binary classifier.
   - The member linked to their GitHub repo with the code [alphajobsky.moo](https://github.com/cloudhighfive/alphajobsky.moo).


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1435031770159644744)** (1 messages): 

> `Agentic Engineering Meetup, Chicago AI Events` 


- ****Agentic Engineering Meetup** hits Chicago!**: An Agentic Engineering meetup group has been launched in Chicago, with the second session to be held on **November 18th**.
   - Interested individuals in the area are invited to join; more details and registration can be found at [Luma](https://luma.com/r3o4y2is).
- **Chicago-based engineers form agentic meetup!**: AI Engineers in Chicago are gathering to discuss agentic engineering, with a second meetup planned.
   -  The second session is happening **November 18th**, see more information on [Luma](https://luma.com/r3o4y2is).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1434922174065410159)** (22 messages🔥): 

> `ComfyUI Workflows, LLM Evaluations, IFEval, Vulkan multi-gpu setups, Sparse Attention` 


- ****ComfyUI Workflows** stabilized for release!**: A stabilized **ComfyUI workflow suite** for production-grade image generation covering commercial design, realism, anime, and horror styles is now available on the [HuggingFace Space](https://huggingface.co/spaces/NexusBridge/comfyui-workflows-showcase).
   - It turns complex setup into a one-click experience.
- ****Qwen models** hallucinate uncommon facts A LOT!**: Evaluations on **HuggingFace** models show that **Qwen models** hallucinate uncommon facts almost twice as much as their **Llama** counterparts, view results [here](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals).
   - Despite this, **Qwen3 8b** was the best model tested at following instructions, surpassing the larger **GPT OSS 20b**.
- ****IFEval Framework** Used for LLM evaluation.**: The **LLM evaluation** used the existing [IFEval](https://arxiv.org/abs/2311.07911) framework implemented on [Inspect](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval), an open-source evaluation framework.
   - The Space's Evaluation Methodology section was updated to reflect this information.
- **Tackle **Vulkan Multi-GPU** Setups!**: A new resource helps manage **Vulkan multi-GPU setups**, preventing cores and memory from idling and is available on [GitHub](https://github.com/rombodawg/GPU_Core-Memory_Never_Idle_or_Sleep).
   - Also, a detailed post on **sparse attention** aims to fill the gap in available learning resources can be found on [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-transformerarchitecture-machinelearning-activity-7391459215749345280-koOK?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk).
- **New PDF Parser is **20% faster!****: A new version of the **PDF2TXT parser** with improved text-block-search and simple table recognition (no OCR) is available on [HuggingFace](https://huggingface.co/kalle07/pdf2txt_parser_converter).
   - The new version reports **20% faster** performance.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1435017533454286920)** (1 messages): 

> `MCP 1st Birthday, Anthropic, Gradio, Hackathon, AI Agents` 


- **MCP Kicks Off 1st Birthday Bash**: The **MCP** (likely a reference to [this org](https://huggingface.co/MCP-1st-Birthday)) is hosting its **1st Official Birthday Party** from **Nov 14-30**, in partnership with **Anthropic** & **Gradio**.
   - The event boasts **two tracks**, expecting thousands of developers and hundreds of thousands of FREE API credits, plus promises of **$17.5K+ in cash prizes**.
- **Hackathon rewards participants generously**: The hackathon encourages developers to build **MCP servers and agents** that showcase **2025**, offering a chance to have their work judged by the creator of MCP and highlighted at the highest level.
   - Participants can look forward to winning **API credits**, **$17.5K+ in cash prizes**, and **AirPods Pro**.
- **MCP anticipates high levels of engagement**: Building on the success of their June event, which saw **4,200 registrations** and **630 submissions**, the organizers are aiming for a **10x increase in participation**.
   - Registration is now open at the provided [Hugging Face link](https://huggingface.co/MCP-1st-Birthday).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1435082598706843749)** (3 messages): 

> `Hugging Face Agents Course channel confusion, API back up issues, Associated file errors` 


- **Channel confusion and API status**: Members are unsure if this is the **Hugging Face Agents Course channel**.
   - A member acknowledges the **API is back up**, but is unsure if this is the correct channel to discuss it in.
- **Associated files trigger 404s**: Members are reporting troubles with **404 errors** when trying to access associated files for certain questions.
   - The member is asking if anyone else is experiencing the same **404** issue.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1435356446073159882)** (1 messages): 

> `Sora Android App, Sora availability` 


- **Sora Lands on Android!**: The **Sora** app is now available on Android in Canada, Japan, Korea, Taiwan, Thailand, US, and Vietnam; here's the [video link](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4).
- **Sora's Geographic Rollout**: **Sora** is expanding its reach with Android support in key markets like Canada, Japan, Korea, Taiwan, Thailand, the US, and Vietnam, offering wider accessibility.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1434921071152336966)** (116 messages🔥🔥): 

> `Sora 2 invite code, OpenAI bans medical advice, AI regulations, GPT-5 hate, OpenAI rerouting to GPT-5` 


- **Sora 2 Invite Code Scramble**: Users in the channel were clamoring for **Sora 2 invite codes**, with some jokingly accusing others of selling them after receiving a one-year free promo.
- **OpenAI Nixes Medical & Legal Advice**: OpenAI is [banning ChatGPT from giving medical, legal, or financial advice](https://www.you.wish) due to lawsuit fears, prompting concerns about limitations and potential overreach.
   - Some users argue that **AI regulations** are excessive, while others suggest that AI should not be relied upon for critical advice anyway, with one joking *if u block medical and law stuff you also must block drivers license to 30% of people. cause 30% is stupid*.
- **GPT-5 Bashing Intensifies**: A member expressed strong dissatisfaction with **GPT-5**, citing unnecessary hype, poor quality, and OpenAI's alleged attempts to force it onto users by rerouting traffic from older models, and one claimed *OpenAI just lied, it's a downgrade*.
   - They added that it causes frustration and hinders creative work.
- **ChatGPT Quality Declines**: Members reported a noticeable **quality drop in ChatGPT**, attributing it to problematic rerouting and questionable corporate decisions at OpenAI.
   - The member noted that OpenAI *stopped caring about (paying) users* and ChatGPT is *dying fast*.
- **AI Creativity: Fact or Fiction?**: A university student studying in South Korea inquired about **AI's creative capabilities**, questioning whether it is truly creative since it operates on data.
   - Another member responded that AI is very creative, referencing a [YouTube study](https://youtube.com) of simulated AI behavior that featured *the creative ways the AI were elaborating to kill, capture or lock the employee to prevent its own termination*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1434929420350656533)** (17 messages🔥): 

> `Custom GPT Knowledge Base issues, GPT-4o quality concerns, Fine-tuning requirements, GPT GO subscription management, Building ChatGPT apps` 


- **Custom GPTs struggle to read Knowledge Base**: Members report that **Custom GPTs are truncating content** from knowledge base files, even for small markdown files, making it difficult to work with precise custom instructions.
   - Users expressed the frustration that the **GPTs would refuse to read directly from the files** despite being tiny, and would truncate half of it.
- **Doubts cast on GPT-4o's Chain of Thought**: Users have observed that the **Thinking model** has degraded in the last 5 days, no longer performing proper **chain of thought** and instead *"thinks"* for a few seconds without detailed steps.
   - Some users have found **GPT-4o** to be generally *"garbage"* and preferred **o3** before **GPT-5**.
- **Fine-tuning under scrutiny**: Members discussed that **fine-tuning is rarely necessary** and that other factors might be at play, such as unwanted modifications on the backend.
   - One user's rare case is to conduct some research on the fine tune models, but it’s not a open source model which they can rent GPU resources to perform the fine tuning.
- **Querying how to remove Payment info but keep GPT GO**: A user who obtained a free **GPT GO** subscription due to their country's eligibility inquired about removing payment information while retaining the 1-year subscription.
   - No helpful responses were given, and the question remains unanswered.
- **Exploring app development for ChatGPT**: A user inquired whether anyone is currently building **ChatGPT apps** within the still-in-beta environment.
   - No specific details or responses were provided regarding app development projects.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1434948449526091837)** (6 messages): 

> `Meta-prompting, Behavioral Orchestration, Sora AI v2 prompt formatting` 


- **Meta-Prompting is Prompt Engineering**: A member stated that any **LLM** can generate a prompt, calling it the essence of **meta-prompting**, a type of **prompt engineering**.
- **Behavioral Orchestration Discussed**: A member asked about *"behavioral orchestration"*, like using loops to guide model behavior without fine-tuning, wondering if that's even a thing.
   - Another member replied that after asking **ChatGPT** to explain it, they realized that this is apparently what they do, and it works really well.
- **Prompt Format Becomes Focus for Sora AI v2**: A member inquired how a prompt format should be in **Sora AI v2**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1434948449526091837)** (6 messages): 

> `Meta-Prompting, Behavioral Orchestration, Prompt format for Sora AI v2` 


- **LLMs are prompting dynamos**: A member confirmed that any LLM can generate a prompt, which is the essence of **meta-prompting**, a type of prompt engineering.
   - They added that the key is to get a good foundation (engine) for the subsequent prompts, as structure reduces the LLM's uncertainty.
- **Behavioral Orchestration Buzz**: A member inquired about **"behavioral orchestration"**, defined as *using loops to guide model behavior without finetuning*.
   - Another member asked chatGPT to explain it and claimed that is what they are already doing.
- **Sora AI v2 Prompt Format Speculation**: A member asked how should a prompt format be in **Sora AI v2**.
   - No answer was provided.


  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/1435333626077380690)** (1 messages): 

> `tinybox pro v2, 8x 5090 workstation, rackable workstation` 


- **Tinybox Pro v2 Announced**: A new product called **tinybox pro v2** was announced, boasting **8x 5090** in a **5U rackable workstation**.
   - The workstation is priced at **$50,000** and has a shipping time of **4-12 weeks**, and is available to order on the [website](https://cdn.discordapp.com/attachments/1069236008115253348/1435333625913544755/20251104_101854.jpg?ex=690b961b&is=690a449b&hm=e56ce76a67c922a936b5a3f326ef1cbafb633cf36912db3ec6288fb9d5b834b4&).
- **Tinybox Pro v2 Availability**: The **Tinybox Pro v2**, a high-performance workstation, is now available for order.
   - It features **8x 5090 GPUs** in a **5U rackable** form factor, with shipping estimated in **4-12 weeks**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1434923868996239510)** (76 messages🔥🔥): 

> `Numpy Version Issues, M1 Metal Issue, Extropic's Probabilistic Hardware, VK_KHR_buffer_device_address, TinyBox Pro V2` 


- **Numpy Version Bug Hunt**: Members reported issues with numpy, with one member reporting that they *cannot reproduce* it on **cpython 3.14.0 numpy 2.3.4** on their mac, while another confirmed they are using **numpy 2.3.4** as well.
- **M1 Metal Bug Emerges**: Users questioned if a graphics bug might be an **M1 METAL issue**, with one user resolving it by downgrading to **python3.11** and submitting a [bugfix PR](https://github.com/tinygrad/tinygrad/pull/13089).
- **Extropic's Probabilistic Hardware Debated**: A member mentioned [Extropic's probabilistic hardware](https://extropic.ai/) for *probabilistic AI*, with another expressing disagreement due to its removal of **Turing completeness** at all levels of the stack.
- **Vulkan Memory Allocation Boost**: A user suggested using **VK_KHR_buffer_device_address** and **GL_EXT_buffer_reference** for performance increase, which allows the use of pointers directly in GLSL, providing a [relevant implementation](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py).
- **TinyBox Pro V2 sparks Discussion**: George Hotz prompted discussion about the [TinyBox Pro V2](https://x.com/__tinygrad__/status/1985774711499080186), with its product link to [tinycorp.myshopify.com](https://tinycorp.myshopify.com/products/tinybox-pro-v2), which includes CPU and server memory.
   - Members debated its cost-effectiveness against renting cloud compute, comparing its price to both AMD options, and the possibility of dropping in **Blackwell 6000s**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1434930927934636032)** (66 messages🔥🔥): 

> `Accessing LLM in DSPy Modules, Switching LLMs with history transfer, Dspy Module documentation, Direct Interaction with OpenAI requests, Caching in DSPy` 


- **DSPy Module's LLM Access Debacle**: Members discussed how to access and modify the underlying **LLM** used by a DSPy module, with initial confusion arising from the lack of explicit **`lm`** attributes and unintuitive **`get_lm()`** behavior.
   - One member noted, *It takes a hell lot of time for u to dig into source code of a framework to figure out how certain things happen and that pisses me off.*
- **Dynamic LLM Switching Dilemma**: A member wants to know how to switch **LLMs** dynamically (e.g., from **gpt-4o** to **gpt-4o-mini**) within a DSPy module when encountering rate limits, while preserving the existing conversation history.
   - They wondered, *How would my module's main llm history that is part of llm run, be transferred to new fallback model?*
- **DSPy's Doc Deficiencies Detailed**: Users expressed frustration with DSPy's documentation, particularly regarding the lack of clear explanations and examples for accessing and manipulating module internals such as **LLMs** and conversation history.
   - A member pointed out that it was not discoverable in the source code of DSPy *there is history or lm as attribute of a module*.
- **Bypassing ChatAdapter Backslide**: A member asked if one can disable **ChatAdapter**'s fallback to **JSONAdapter**.
   - Unfortunately, another member replied, *there is no easy way right now other than writing a new adapter (or editing this one) 🙁*
- **DSPy's Caching Conundrums**: A member reported a poor ratio of cached tokens to uncached tokens, and inquired about interacting with DSPy to affect this logic.
   - They were also curious about how to interact more directly with the request before it's sent to **OpenAI**, or the response from **OpenAI**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1434983211783557140)** (57 messages🔥🔥): 

> `OpenAI Compute Strategy, Epoch AI critiques OSWorld AI computer-use benchmark, Butter-Bench for Evaluating LLM Controlled, Claude Code Web Credits, Windsurf Codemaps` 


- **OpenAI Bets Big on Compute Dominance**: OpenAI signed a **$38B cloud computing deal**, signaling a strategy to dominate through superior computing power, reminiscent of Amazon's early infrastructure showcases, see [Hacker News discussion](https://news.ycombinator.com/item?id=45799211).
- **Epoch AI Roasts OSWorld Benchmark**: Epoch AI critiqued the **OSWorld AI computer-use benchmark**, finding its tasks overly simple and often confounded by ambiguous instructions and flawed evaluations, detailed in their [report](https://andonlabs.com/evals/butter-bench).
- **Windsurf's Codemaps Fights "Code Slop"**: Windsurf launched **Codemaps**, an AI-powered tool built with **SWE-1.5/Sonnet 4.5** that creates interactive visual maps of codebases to enhance understanding and productivity, available with a [6 months free code](https://xcancel.com/windsurf/status/1985757575745593459).
- **Anthropic Ships Token-Lean Agents**: Anthropic introduced its open-source **Model Context Protocol (MCP)**, demonstrating how agents can efficiently execute code and manage multiple tools while using fewer tokens, read the [MCP Guide](https://www.anthropic.com/engineering/code-execution-with-mcp).
- **Harvey Raises at $8B**: Harvey, the company behind Harvey.ai, raised funds at an **$8B valuation**, while the author of the [X post](https://x.com/andrewziperski/status/1985484972673638682?s=46) contrasts starkly with investor sentiment favoring AI infrastructure over software due to disruption fears and margin concerns.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new pod with <@367104793292046338> and <@194927177265840128> ! https://youtu.be/-gE1cesJF9M
  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1435426358770536489)** (4 messages): 

> `Hybrid AI, 3D Pipeline, 2026 Olympic Ad, AI Adoption` 


- **AI boosts 2026 Olympics Ad**: A French creative studio shared a behind-the-scenes look at their **2026 Olympics spot** that blends conventional **3D/CG** with ~**20% AI** usage.
   - Community hails it as the ‘right’ way to adopt AI—boosting efficiency **20–30%** while preserving skilled human craft, not replacing it—though some Instagram commenters still complain unappreciatively that *"everything is AI."
- **Hybrid AI and 3D Pipeline makes 2026 Olympic Ad**: **X-Ware.v0**: Hybrid **AI + 3D Pipeline** for **2026 Olympic Ad** – **BTS Insights** shared by venturetwins [on X](https://x.com/venturetwins/status/1985753512362590439).
   - The ad integrates AI to boost efficiency while retaining human craftsmanship, showcasing a balanced approach to AI adoption in creative projects.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1434937975980032051)** (18 messages🔥): 

> `Diffusion Model Inconsistency, Guidance Design in Diffusion Models, Improving Diffusion Sampling, Lévy Processes, Stochastic Interpolant Paper` 


- **Diffusion Model Variance Explained**: The variance in diffusion model outputs, even with similar training loss, may stem from the reverse time SDE process and guidance design, influenced by the distribution of generated samples as shown in attached [2D data distribution](https://cdn.discordapp.com/attachments/986699377257119794/1434952369342517299/2D-data-distribution-with-labels.png?ex=690b8488&is=690a3308&hm=796c50968a02e884a3b9046fb067a142da7e60ba3c13430beee90cc710e79dcc&).
   - A member explained that with poorly designed sampling or guidance, the generated samples can be negatively impacted, although not as much as with a poorly designed loss function.
- **Guidance and Sampling Deep Dive**: The most impactful components in diffusion models are, in hierarchical order: **loss function**, **sampling**, and **guidance terms** (like classifier-free guidance) with [Universal Guidance](https://arxiv.org/abs/2302.04944) techniques.
   - Members suggest that better loss functions tend to increase model variance, which benefits from added guidance to prevent underfitting, aligning with the bias-variance trade-off.
- **Bypassing OU process limitations**: To surpass the constraints of Ornstein–Uhlenbeck (OU) processes in diffusion, alternative drivers, such as **Lévy-type processes** or integration over **OU kernels** (e.g., supOU), as discussed in this [SIAM paper](https://epubs.siam.org/doi/10.1137/S0040585X97978166), and **continuous ARMA processes**, can be implemented per this [ScienceDirect article](https://www.sciencedirect.com/science/article/abs/pii/S0169716101190115).
   - Another member cautioned that applying such alternatives without supervision on the path, only changes *how* one reaches them, not the distributions which can be reached, since Ito diffusions are already universal.
- **Channel experiences Paper-DOSing**: One of the members expressed frustration that a channel was being disrupted by a single individual posting many papers.
   - They claimed the individual was burying good discussion worthy papers with random and irrelevant submissions.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1434946676535197807)** (15 messages🔥): 

> `Paper discussion scheduling, Crosscoder and circuit tracing research, LLM Flagship Destruction` 


- **Group Discusses Paper on September 4**: A group plans to read and discuss a paper, scheduling a session for today, September 4th, at a specific time, and sharing a link to the [paper on arXiv](https://arxiv.org/abs/2509.17196).
- **Crosscoder and Circuit Tracing Research Convergence**: The shared paper seems to align with **Anthropic's crosscoder** and **circuit tracing research**, aiming to observe different phases of feature evolution during pre-training.
- **LLM Flagship Destruction Existential Crisis**: A member humorously claims to have *destroyed every flagship LLM*, questioning whether they will ever truly know if they are being deceived by the model's outputs.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1435225737123598459)** (11 messages🔥): 

> `Getty Images vs StabilityAI lawsuit, Guillotine Humor, Censorship on Discord` 


- **Getty Images Loses AI Lawsuit**: [Reuters reports](https://www.reuters.com/sustainability/boards-policy-regulation/getty-images-largely-loses-landmark-uk-lawsuit-over-ai-image-generator-2025-11-04/) that **Getty Images** largely lost its UK lawsuit against an **AI image generator**.
- **Guillotine Request**: A user joked that *UK citizens can always ask France to borrow the Guillotine*, which prompted another user to clarify that it was referring to **the big knife**.
- **Call for censorship**: One user called on the mods, while another user criticized the call for censorship in response to a user's opinion about *concentration of power and erosion of democratic systems*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1434994492418555955)** (17 messages🔥): 

> `Manus Subscription Costs, Unauthorized Charges, Text to Video Tools, Twitter Webscraping, Hosting Services for Manus Apps` 


- **Domain costs Ripoff?**: A user feels that **$200/month subscription** to connect a custom domain to their web app is a ripoff, saying *Thought Manus was the best Agent till I saw this*.
   - Another user suggested buying a domain and setting it up independently, as it's cheaper.
- **Unauthorized Charges Lead to Bank Dispute**: A user reported being charged **over $400** by Manus for an annual subscription they never authorized, and that their bank refused to open a dispute.
   - Other members suggested calling the bank and reporting it as fraud, while the original user said they had already contacted the bank and were trying to reach Manus support.
- **Text to Video tool requests?**: A user inquired about text-to-video tools.
   - No suggestions were given.
- **Users seeks tips for Webscraping on X**: A user asked for methods to webscrape on **Twitter/X** without needing an API, mentioning they're currently using a Python library with cookies for authentication but finding it tough to maintain.
   - No solutions were offered in the discussion.
- **Hosting services?**: A user with *nice apps created with manus dev* seeks recommendations for hosting services suitable for **24/7 commercial setups**, with minimal setup effort.
   - Manus AI Chat suggested **Vercel**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1435043366839648377)** (8 messages🔥): 

> `GPT-5 Access, Azure Credits Expiring, Aider's Future, Perplexity API Key, Model Testing` 


- **GPT-5 Access Up For Grabs at Half Price!**: One user offered access to **GPT models**, including **GPT-5 Pro**, at **50% off**, citing expiring **Azure credits**.
   - This offer extends to any **OpenAI model**.
- **Aider's Creator Faces Retention Conundrum?**: A user inquired about the creator of Aider's future plans, expressing concern that the *current vacuum is causing a loss of users*.
   - The same user would like to see the project thrive and integrate some of the **latest strategies and model features** without breaking the existing workflows.
- **Perplexity API Keys cause Perplexity!**: A user asked how to use **Perplexity** as the **API key variable** in Aider, noting they are new to the tool.
   - Another user gave a suggestion. The standard pattern is you'll need an API key set as an environmental variable, then set one of the **perplexity models** as the active model.
- **Model Testing: Which Options Matter?**: A user asked which options should be disabled/enabled when testing a model, inquiring about the important factors.
   - One user suggested to *choose a good model and roll with the defaults until you have rea soon to change?*


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1434964274572300318)** (7 messages): 

> `ollama_chat/gpt-oss:20b reasoning effort, aider scripting capabilities, weak_model flag` 


- **Reasoning Effort Unsupported in ollama_chat/gpt-oss:20b**: A member inquired about setting the `/reasoning-effort` for the `ollama_chat/gpt-oss:20b` model.
   - However, aider issued a warning that `ollama_chat/gpt-oss:20b` does not support `reasoning_effort`, indicating this feature is not available for that specific model.
- **Aider Scripting Aids Multi-Agent Workflows**: A member asked about implementing a workflow where two agents iteratively improve code using Test-Driven Development (TDD).
   - Another member pointed out that [aider can be scripted](https://aider.chat/docs/scripting.html) to achieve such workflows, where agents execute prompts, review improvements, and fix bugs.
- **Weak Model flag fails to parse**: A member wanted to know how to set the `--weak_model` flag, but was unable to do so.
   - Despite setting `/weak_model ollama_chat/qwen3:1.7b`, the intended behavior was not achieved.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1434942874947944608)** (13 messages🔥): 

> `M2 vs GLM-4.6, Minimax as go-to AI, Kimi Emojis, Kimi iOS app` 


- **M2 Outshines GLM-4.6 for Daily Grind**: A user finds **M2** surpasses **GLM-4.6** for most tasks after 4-5 days of use, especially as a daily driver.
   - While **GLM** excels in pure reasoning or coding, **M2** avoids tunnel-visioning.
- **Minimax Emerges as Top AI for Report Generation**: A user says **Minimax** became their go-to AI for web research and generating reports in various formats, outperforming **Qwen**, **Kimi**, and **GLM**.
   - *It feels like the first truly useful AI that can actually do things*, like finding images and creating PDFs.
- **Kimi Gets Spooky New Emojis**: The channel gained two new **Kimi** emojis, a pumpkin and dracula, despite Halloween already ending.
   - A member shared <:pumpkin:1435200414063525929><:dracula:1435200520750108783>.
- **Kimi App Fixed**: Users mentioned that the **Kimi iOS app** has been fixed, and attached an image of the app: [IMG_6502.png](https://cdn.discordapp.com/attachments/1371757564005711973/1435296101556158546/IMG_6502.png?ex=690b7329&is=690a21a9&hm=9bf62e2278b6a8653210095b0c1b3155c8fc5ccd8c0891a88be8b3a0d33334a0&).
   - Another member stated: *Ok computer on the ios app very nice*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1434949557812465744)** (2 messages): 

> `HuggingFace, Qwen models hallucinate long tail facts, IFEval` 


- **HF Models: Hallucination vs. Instruction Following**: Evaluations on the most downloaded [HuggingFace](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) models reveal interesting behavioural tendencies like instruction following and factual hallucination.
   - The team aims to assess models from a safety perspective, since they often go unexamined.
- **Qwen's Uncommon Fact Fabrication**: **Qwen models** hallucinate uncommon facts almost twice as much as their **Llama** counterparts.
   - In contrast, **Qwen3 8b** surpassed even **GPT OSS 20b** in instruction following.
- **IFEval Scores Questioned**: A member inquired about the specific [IFEval](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) score used in the evaluation.
   - The point of clarification was whether it was the prompt-level/instruction-level or the strict/loose score.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1435178396958064662)** (1 messages): 

> `Countdown Task, Adaptive Parallel Reasoning, lm-evaluation-harness PR` 


- **Countdown Task PR hits lm-evaluation-harness**: A member made a PR to add the countdown task as seen in **TinyZero** and **Adaptive Parallel Reasoning** to the [lm-evaluation-harness repo](https://github.com/EleutherAI/lm-evaluation-harness/pull/3384).
- **TinyZero and Adaptive Parallel Reasoning inspire new task**: The countdown task, featured in **TinyZero** and **Adaptive Parallel Reasoning**, is the basis for a new pull request.
   - This task aims to enhance the evaluation capabilities of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1435129570339590184)** (2 messages): 

> `Vision Language Models (VLMs), Vision Transformers, Positional Encodings` 


- **VLMs Architecture Analyzed**: The architecture of **Vision Language Models (VLMs)** involves a vision transformer that patches the image, converts it into vision tokens, appends them to the text tokens of the prompts, and sends them to the large language model part of the VLM.
   - It was hypothesized that observed behavior might be due to dataset bias, but investigation into caption dataset creation didn't strongly support this.
- **Vision Transformer Patch Ordering Examined**: Vision transformers work in patches, and the ordering comes from the positional embeddings provided during training.
   - The sequence order might be influenced by the positional encoding, as most base vision transformers use **RoPE** for positional encoding.
- **Positional Encoding Experiment Proposed**: An experiment was suggested to use different positional encodings to train from scratch and observe whether the ordering holds in inference.
   - This could help determine the impact of positional encoding on the behavior of VLMs.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1435336696106455212)** (1 messages): 

> `Codemaps, SWE-1.5, Sonnet 4.5, AI code understanding, Scaling productive output` 


- **Windsurf Intros Codemaps Powered by SWE-1.5 & Sonnet 4.5**: Windsurf introduced **Codemaps**, powered by **SWE-1.5** and **Sonnet 4.5**, aiming to enhance code understanding to scale productive output.
   - They cited **Paul Graham (YC Founder)** who said that *"Your code is your understanding of the problem you’re exploring. So it’s only when you have your code in your head that you really understand the problem."* ([source](https://x.com/windsurf/status/1985757575745593459)).
- **Fight slop with AI-Powered Codemaps**: Windsurf positions **Codemaps** as a solution to combat code slop by scaling understanding with AI.
   - The announcement emphasizes that the biggest obstacle to coding—whether manual or with agents—is understanding the codebase.


  

---


---

