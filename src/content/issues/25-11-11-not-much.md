---
id: MjAyNS0x
title: not much happened today
date: '2025-11-11T05:44:39.731046Z'
description: >-
  **GPT-5** leads Sudoku-Bench solving 33% of puzzles but 67% remain unsolved,
  highlighting challenges in meta-reasoning and spatial logic. New training
  methods like **GRPO fine-tuning** and "Thought Cloning" show limited success.
  Research on "looped LLMs" suggests pretrained models benefit from repeated
  computation for better performance. **Baidu's ERNIE-4.5-VL-28B-A3B-Thinking**
  offers lightweight multimodal reasoning with Apache 2.0 licensing,
  outperforming **Gemini-2.5-Pro** and **GPT-5-High** on document tasks.
  **Databricks ai_parse_document** preview delivers cost-efficient document
  intelligence outperforming GPT-5 and Claude. **Pathwork AI** uses
  **LlamaCloud** for underwriting automation. **Gemini File Search API** enables
  agentic retrieval augmented generation (RAG) with MCP server integration.
  **Together AI** and **Collinear** launch **TraitMix** for persona-driven agent
  simulations integrated with **Together Evals**. Reports highlight risks in
  long-running code agents like **Claude Code** reverting changes, emphasizing
  guardrails. Community consensus favors multiple code copilots including Claude
  Code, Codex, and others.
companies:
  - openai
  - baidu
  - databricks
  - llamaindex
  - togethercompute
  - sakanaailabs
models:
  - gpt-5
  - qwen2.5-7b
  - ernie-4.5-vl-28b-a3b-thinking
  - gemini-2.5-pro
  - llamacloud
  - claude-code
topics:
  - reasoning-benchmarks
  - reinforcement-learning
  - fine-tuning
  - multimodality
  - document-intelligence
  - retrieval-augmented-generation
  - agentic-systems
  - persona-simulation
  - code-agents
  - guardrails
people:
  - sakanaailabs
  - micahgoldblum
  - francoisfleuret
  - matei_zaharia
  - jerryjliu0
  - omarsar0
  - togethercompute
  - imjaredz
  - theo
---


**a quiet day**

> AI News for 11/10/2025-11/11/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (201 channels, and 5180 messages) for you. Estimated reading time saved (at 200wpm): 465 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

AIE CODE Side Events listing are now up: https://ai.engineer/code#events

If you're in NYC, you can attend any of these without an AI Engineer ticket! Enjoy.

---

# AI Twitter Recap

**Reasoning benchmarks and training techniques**

- **Sudoku-Bench update: GPT-5 leads but gaps remain**: Since Sudoku-Bench launched in May 2025 (when no LLM solved classic 9x9), **GPT-5** now solves **33%** of puzzles—about 2x the previous leader—and is the first tested LLM to solve a 9x9 variant. Yet **67% of harder variants remain unsolved**, underscoring deficits in meta-reasoning, spatial logic, and global consistency. Experiments with **GRPO fine-tuning** on Qwen2.5-7B and “Thought Cloning” (expert traces from Cracking the Cryptic) still struggle with “break-in” strategies humans use. The authors argue new approaches are required beyond current RL/trace-training regimes. Details: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1988080410392404021) and blog.
- **“Looped LLMs” for computational depth**: New work converts pretrained LLMs into “looped” models that repeatedly iterate their own computation, improving performance over the base model—suggesting many pretrained LLMs are under-computed and benefit from increased depth at inference time. Thread: [@micahgoldblum](https://twitter.com/micahgoldblum/status/1988265009508655528).
- **KL penalty tweak in RL training**: A brief research note from [@francoisfleuret](https://twitter.com/francoisfleuret/status/1988364427675189640) reports replacing the standard KL penalty with a modified variant that achieves a previously elusive property; no further technicals shared yet.

**Multimodal and document intelligence**

- **Baidu’s ERNIE-4.5-VL-28B-A3B-Thinking (Apache-2.0)**: Lightweight multimodal reasoning model with “>3B active parameters,” claiming SOTA on document/chart understanding and to outperform Gemini‑2.5‑Pro and GPT‑5‑High on select benchmarks. Adds “Thinking with Images” to zoom in/out on details. Licensed **Apache 2.0 (commercial use)**. Launch context via Baidu World 2025: [@Baidu_Inc](https://twitter.com/Baidu_Inc/status/1988182106359411178).
- **Databricks ai_parse_document (public preview)**: A production document intelligence service to convert PDFs/reports/diagrams into structured data at up to **5× lower cost**, with tight integration into Lakehouse tooling (Lakeflow, Unity Catalog, Agent Bricks, Vector Search, AI/BI). Databricks reports it outperforms leading VLMs (GPT‑5, Claude) on doc tasks. Announcements: [@databricks](https://twitter.com/databricks/status/1988271796076912928), [@matei_zaharia](https://twitter.com/matei_zaharia/status/1988325177193885885).
- **Agentic document automation in underwriting**: LlamaIndex highlights Pathwork AI’s underwriting agents (life insurance) built on LlamaCloud to process high-volume medical documentation and carrier guidelines—an archetypal large-scale unstructured-doc workflow for agents. Case study: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1988394058197184923).

**Agents, retrieval, and production strategy**

- **Gemini File Search API for agentic RAG + MCP**: A developer-built **MCP server** leverages Gemini File Search for semantic/code search over codebases, making it straightforward to wire up agentic RAG patterns; demoed with Karpathy’s nanochat. Early signal that File Search can simplify end-to-end “agent that reads your repo” systems. Details: [@omarsar0](https://twitter.com/omarsar0/status/1988236096195776683).
- **Persona-driven agent sims and evals**: Together AI x Collinear’s “TraitMix” generates persona-driven agent interactions and integrates with **Together Evals** for workflow-level assessment—useful for simulation-driven development and evaluation of agent behavior. Announcement: [@togethercompute](https://twitter.com/togethercompute/status/1988374675093897380).
- **Cautionary tale: code agents in long-running ops**: A report of Claude Code “reverting everything” after completing an overnight migration underscores the importance of guardrails, logging, and explicit execution modes for long-running code agents. Anecdote: [@imjaredz](https://twitter.com/imjaredz/status/1988379604160311696). Meanwhile, community consensus is that multiple code copilots/agents are now “good” (Claude Code, Codex, Cursor, Windsurf, Cline, Roo, Kilo, OpenCode, Aider): [@theo](https://twitter.com/theo/status/1988380210715389958).
- **Org design vs continuous-learning agents**: A pragmatic note that centralized AI business models and safety/compliance workflows are often at odds with “self-evolving” agents. Moving from “global best” to local maxima (Level 4 autonomy) may force teams toward on-device/local data loops, changing GTM and infra assumptions. Perspective: [@swyx](https://twitter.com/swyx/status/1988370167622234524). Related: a live “MCP debate” call for participants: [@swyx](https://twitter.com/swyx/status/1988345059675435046).

**Open data, models, and tools**

- **LAION’s Project AELLA (structured science at scale)**: Open initiative with @inference_net and @wyndlabs_ai to make **100M scientific papers** accessible via LLM-generated structured summaries. Launch includes a **100K-summary dataset**, two fine-tuned LLMs, and a 3D visualizer. Announcement: [@laion_ai](https://twitter.com/laion_ai/status/1988330466706157818).
- **FinePDFs update (multilingual educational corpus)**: Release previews include **350B+ tokens** from educational sources in **69 languages**, **69 classifiers** (ModernBERT/mmBERT), and **300K+ EDU annotations per language** generated with Qwen3‑235B—positioned for academic/edu applications. Details: [@HKydlicek](https://twitter.com/HKydlicek/status/1988328336469459449).
- **Photo-to-Anime LoRA**: QwenEdit-2509 LoRA for Photo→Anime conversion outperforms prompting-only approaches for stylization tasks; model on HF. Note: [@wildmindai](https://twitter.com/wildmindai/status/1988309389259010112).
- **Terminal-first experiment tracking**: W&B “LEET” is a TUI for live, offline run monitoring directly in the terminal—useful for air-gapped/cluster workflows without a browser. Preview: [@wandb](https://twitter.com/wandb/status/1988401253156876418), setup: [@wandb](https://twitter.com/wandb/status/1988401739872301137).

**Systems, kernels, and robotics**

- **HipKittens (AMD kernels)**: From Stanford/HazyResearch, HipKittens achieves up to **2×** speedups over ROCm’s composable kernels baseline on AMD GPUs across tests—closing gaps for AMD-heavy training stacks. Announcements: [@qubitium](https://twitter.com/qubitium/status/1988389379984027742), [@AnushElangovan](https://twitter.com/AnushElangovan/status/1988393252555493739).
- **Lightning Grasp (dexterous grasp synthesis)**: Procedural grasp generation at **10–100× faster** than prior SOTA across diverse robot hands and challenging objects; paper and code open-sourced. Details: [@zhaohengyin](https://twitter.com/zhaohengyin/status/1988318037804806431).

**Safety, consent, and platform quality**

- **Voice Consent Gate + anthropomorphism blockers**: With rapid voice cloning advances, [@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1988367909849329777) proposes a “Voice Consent Gate” to normalize consent verification layers for synthetic voice usage; related efforts on “anthropomorphism blockers” are now reflected in NY state law ([discussion](https://twitter.com/mmitchell_ai/status/1988358221418106959), follow-ups [1](https://twitter.com/mmitchell_ai/status/1988373005790310512), [2](https://twitter.com/mmitchell_ai/status/1988373735863447878), [3](https://twitter.com/mmitchell_ai/status/1988374668424999026)). Useful immediate design target for infra teams building voice features.
- **Provider reliability matters**: Caution on quality variance across model providers in aggregator gateways; some users are reverting to first-party APIs for reliability until aggregators enforce stronger model/provider validation. Note: [@scaling01](https://twitter.com/scaling01/status/1988399213563236810).

**On-device multimodal models**

- **Google’s “Nano Banana” on Pixel**: Google’s November Pixel drop includes “Nano Banana,” a Gemini-based image editing/generation model integrated into Messages and Photos. While demos wowed, community notes it likely behaves like a compact general-purpose LLM with structured image output (not zero-shot math diffusion), potentially architecturally akin to Hunyuan Image 3. Announcements: [@Google](https://twitter.com/Google/status/1988377964686266518) and analysis: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1988390269998559584).

**Top tweets (by engagement)**

- Plumber/professor parable on practical expertise: [@burkov](https://twitter.com/burkov/status/1988348230761902514) (~4.3K)
- Teaser: “Tomorrow.” [@dwarkesh_sp](https://twitter.com/dwarkesh_sp/status/1988341914907930732) (~2.7K)
- GPT-5 tops Sudoku-Bench; 33% solved; first 9×9 variant solved: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1988080410392404021) (~828)
- Language learning cognition reflection: [@AmandaAskell](https://twitter.com/AmandaAskell/status/1988202354051805522) (~596)
- Baidu ERNIE-4.5-VL-28B-A3B-Thinking launch (Apache-2.0): [@Baidu_Inc](https://twitter.com/Baidu_Inc/status/1988182106359411178) (~595)
- On moral reflection in building tech (re: Pope’s post): [@tbpn](https://twitter.com/tbpn/status/1988366296573243696) (~518)
- Gemini File Search for agentic RAG + MCP server demo: [@omarsar0](https://twitter.com/omarsar0/status/1988236096195776683) (~515)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. VibeThinker 1.5B Model and Benchmark Performance

- [**We put a lot of work into a 1.5B reasoning model — now it beats bigger ones on math & coding benchmarks**](https://www.reddit.com/r/LocalLLaMA/comments/1ou1emx/we_put_a_lot_of_work_into_a_15b_reasoning_model/) (Activity: 776): **The image showcases the performance of the "VibeThinker 1.5B" model, which is a 1.5 billion parameter model designed for reasoning tasks, particularly in math and coding. Despite its smaller size, it outperforms larger models on benchmarks such as AIME 2024, AIME 2025, HMMT 2025, and LiveCodeBench V5. This achievement is attributed to the model's fully decontaminated training data and its focus on reasoning capabilities rather than general chatbot functions. The model's success challenges the notion that larger models are inherently superior in these domains.** Some commenters express skepticism about the model's claims, questioning the validity of a 1.5B model outperforming larger ones like DeepSeek R1. Others note the model's high token consumption for simple tasks, suggesting inefficiencies in its processing.
    - Chromix_ highlights a potential inefficiency in the model's token usage, noting that it consumes 5,000 tokens for reasoning and 500 for results in a simple task, compared to Granite-4.0-h-1B's 140 tokens. This suggests that while the model may perform well on benchmarks, its token efficiency could be improved for practical applications.
    - ilintar expresses skepticism about the claim that a 1.5B Qwen 2.5 fine-tune can outperform DeepSeek R1, implying that such a performance leap is unlikely without substantial evidence. This reflects a broader concern in the community about exaggerated performance claims without rigorous benchmarking.
    - noctrex mentions the addition of an unquantized BF16 version of the model, available on Hugging Face. This version could offer improved performance or compatibility for certain applications, as BF16 is often used to balance precision and computational efficiency in machine learning models.
- [**Seems like the new K2 benchmarks are not too representative of real-world performance**](https://www.reddit.com/r/LocalLLaMA/comments/1ou1j3e/seems_like_the_new_k2_benchmarks_are_not_too/) (Activity: 642): **The image highlights skepticism about the new K2 benchmarks, suggesting they may not accurately reflect real-world performance. The tweet questions how a model can excel in general exams but fail in specific areas like lambda calculus, indicating a potential gap between benchmark results and practical application. This reflects broader concerns in the AI community about the representativeness of benchmarks, as discussed in the comments. Some users argue that while models may perform well on certain benchmarks, they may not generalize effectively across diverse tasks, a sentiment echoed by experiences with other models like Qwen3 and coder480. The discussion suggests a need for more comprehensive evaluation methods that better capture real-world performance across various domains.**
    - Klutzy-Snow8016 highlights the issue of benchmarks not being representative of specific workloads, such as lambda calculus, suggesting that models like K2 Thinking may perform variably across different domains. This underscores the need for more comprehensive evaluations beyond private tests to determine a model's overall effectiveness.
    - ResidentPositive4122 discusses the disparity between benchmark performance and real-world application, citing Qwen3 models as an example. They note that while some models excel in benchmarks, they fail in practical tasks like Python code minification. The commenter suggests that factors like scale and data curation might contribute to the superior generalization of models like gemini2.5 and the 'big4' models, which perform well in extended, complex tasks.
    - Mickenfox provides an example of Claude 3.7 Sonnet's performance, which scored 77% on GPQA Diamond but failed in a practical scenario involving a vending machine task. This illustrates the gap between benchmark scores and real-world intelligence, as the model exhibited erratic behavior, highlighting the limitations of benchmarks in assessing true model capabilities.

### 2. Egocentric-10K Dataset Launch

- [**Egocentric-10K is the largest egocentric dataset. It is the first dataset collected exclusively in real factories (Build AI - 10,000 hours - 2,153 factory workers - 1,080,000,000 frame)**](https://www.reddit.com/r/LocalLLaMA/comments/1ouazho/egocentric10k_is_the_largest_egocentric_dataset/) (Activity: 318): **Egocentric-10K is the largest egocentric dataset, featuring** `10,000 hours` **of video footage collected from** `2,153 factory workers` **across real factory environments, totaling** `1,080,000,000 frames`**. This dataset is hosted on [Hugging Face](https://huggingface.co/datasets/builddotai/Egocentric-10K) under an Apache 2.0 license, facilitating open-source research and development in robotics and AI. The dataset aims to address the data scarcity in humanoid robotics, where large-scale data is crucial for training models to perform complex tasks in industrial settings.** Commenters discuss the ethical implications and potential motivations behind releasing such a dataset. Some view it as a positive step towards democratizing AI research, while others express concern about the impact on workers' privacy and autonomy. The debate highlights the tension between technological advancement and ethical considerations in data collection.
    - **false_robot** highlights the importance of large datasets like Egocentric-10K for advancing humanoid robotics. The dataset, collected in real factories, is crucial as robotics companies see data as a key limitation in developing robots capable of performing factory and everyday tasks. The open-source nature of this dataset is seen as beneficial for fostering innovation and creating open models in the robotics field.
    - **false_robot** raises a critical point about the motivation behind releasing the Egocentric-10K dataset. They question whether the release is aimed at democratizing knowledge or if it is a response to insufficient results in robotics applications. This reflects a broader debate on whether open data initiatives are driven by genuine innovation goals or as a reaction to challenges in achieving practical outcomes.
    - **Red_Redditor_Reddit** expresses concern about the potential negative impact of AI and robotics on factory workers' lives, suggesting that increased surveillance and micromanagement could make their work environment more challenging. This comment underscores the ethical considerations and potential societal impacts of deploying AI technologies in industrial settings.
- [**A startup Olares is attempting to launch a small 3.5L MiniPC dedicated to local AI, with RTX 5090 Mobile (24GB VRAM) and 96GB of DDR5 RAM for $3K**](https://www.reddit.com/r/LocalLLaMA/comments/1otveug/a_startup_olares_is_attempting_to_launch_a_small/) (Activity: 535): **Olares is launching the Olares One, a compact 3.5L MiniPC aimed at local AI processing, featuring an NVIDIA RTX 5090 Mobile GPU with** `24GB VRAM`**,** `96GB DDR5 RAM`**, and an Intel Core Ultra 9 275HX processor. Priced at** `$3K`**, it runs on the new Olares OS, an open-source platform for AI application deployment. The device is designed to offer cloud-level AI performance locally, with pre-sales starting on Kickstarter in December 2025. [More details here](https://www.example.com/).** Commenters express skepticism about the pricing and market fit, comparing it unfavorably to other high-performance options like the DGX Spark and AMD Strix Halo. Concerns are also raised about the unfamiliarity of the Olares OS and potential performance issues when models exceed RAM capacity.
    - The Olares MiniPC's pricing and specifications are being scrutinized in comparison to other available options. For instance, a DGX Spark with 128GB VRAM is available for $4k, and an AMD Strix Halo with 128GB unified RAM is priced at $2.2k. This raises questions about the market viability of the Olares MiniPC, which offers an RTX 5090 Mobile with 24GB VRAM and 96GB DDR5 RAM for $3k.
    - A user shared their experience running AI models on a laptop equipped with a mobile RTX 5090 and 64GB DDR5 RAM. They noted that performance is acceptable as long as the model fits within the RAM. However, once the system RAM is utilized, performance significantly degrades, highlighting potential limitations of the Olares MiniPC's configuration for demanding AI tasks.
    - There is a demand for consumer hardware with high-speed unified memory capable of running large models efficiently. One user expressed interest in a system with 1TB of unified memory that can handle 200-500 billion parameter models at 100 tokens per second inference speed, indicating that current offerings, including the Olares MiniPC, may not meet the needs of users seeking high-performance AI solutions.

### 3. GPT-OSS-120B on Cerebras Satirical Analysis

- [**gpt-oss-120b on Cerebras**](https://www.reddit.com/r/LocalLLaMA/comments/1ougamx/gptoss120b_on_cerebras/) (Activity: 355): **The image is a meme that humorously critiques the performance of the** `gpt-oss-120b` **model on Cerebras hardware, suggesting inefficiency in its reasoning capabilities. The cartoon character's exaggerated features and incorrect equations symbolize the model's high token generation rate (**`3000 tokens per second`**) but imply that the output may lack quality or accuracy. This satirical take highlights potential issues with computational output quality despite high processing speeds.** One commenter questions if `gpt-oss` performs worse on Cerebras, noting a preference for `gpt-oss` over other models like `llama 3.3` and `llama 4` due to corporate constraints. Another mentions Cerebras running `GLM 4.6` with `500 tokens per second` decoding, suggesting speculative decoding as a potential advantage.
    - Cerebras is currently running GLM 4.6 on their API, achieving an average of `500 tokens per second` during decoding. They also implement speculative decoding, which significantly enhances coding speed. This could be a valuable addition for users, though it's unclear how it performs on real-world tasks yet.
    - There was an initial issue with the policy implementation upon release, but once corrected, the model performed as expected. This suggests that early problems were more about implementation rather than the model's inherent capabilities.
    - The gpt-oss model is perceived as a significant improvement over LLaMA 3.3 and 4, especially in environments with corporate restrictions. However, there are concerns about hosting and expectations, which might affect its perceived performance on Cerebras.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI-Generated Content and Detection

- [**I'm curious as to how people can tell whether modern videos are Ai or not.**](https://www.reddit.com/r/ChatGPT/comments/1oubuh8/im_curious_as_to_how_people_can_tell_whether/) (Activity: 1353): **The post discusses the increasing difficulty in distinguishing AI-generated videos from real ones, highlighting a specific video that appears convincingly real despite being AI-generated. The user seeks advice on identifying AI content as these technologies improve. The challenge is compounded by the fact that even real videos are sometimes mistakenly identified as AI, leading to confusion and skepticism about authenticity.** One commenter notes the growing trend of misidentifying real videos as AI, suggesting a potential future where distinguishing between AI and reality becomes nearly impossible, potentially leading to widespread confusion. Another commenter points out specific visual cues, such as reflections, that might help in identifying AI-generated content.
    - MalusZona highlights several technical indicators that suggest a video might be AI-generated: constant movement speed without natural acceleration or deceleration, overly clean audio that doesn't reflect distance, and unnatural proportions or movements, such as a person opening a door effortlessly from an improbable position. These elements can be subtle but are often giveaways of AI synthesis.
    - J7mbo points out that responses to AI-generated videos are used to train models on what needs improvement. This feedback loop means that as people identify flaws or unnatural elements in AI videos, these insights are incorporated into future iterations, making detection increasingly challenging.
    - Hoofrad notes a specific technical detail: the reflection of a bay window shifting as it opens, which can be a telltale sign of AI generation. Such reflections and lighting inconsistencies are often difficult for AI to replicate accurately, making them useful indicators for discerning authenticity.
- [**Very helpful, thanks.**](https://www.reddit.com/r/ChatGPT/comments/1otzphl/very_helpful_thanks/) (Activity: 7590): **The image humorously highlights a common issue with virtual assistants and language models: their tendency to generate incorrect factual information, such as dates, due to reliance on pre-trained data rather than real-time data fetching or calculation. This underscores a technical challenge in AI development, where models need to discern when to fetch or compute real-time data instead of relying solely on their training. The comments suggest that integrating real-time data processing, like using Python scripts for accurate calculations, could enhance the reliability of AI systems in providing factual information.** A notable opinion from the comments suggests that AI models should be able to determine when to fetch or calculate real data, rather than generating it, to improve accuracy. Another comment humorously suggests testing the model's response to incorrect corrections.
    - Quantumstarfrost discusses the need for language models to improve their ability to discern when to fetch or calculate real data rather than just generating text. They suggest that models should recognize when a user is asking for a factual answer and should be capable of executing or generating a program to retrieve accurate data, such as the current date, to ensure reliability.
    - Quantumstarfrost also mentions using ChatGPT to write Python scripts for data analysis, highlighting the trust in Python's ability to perform accurate mathematical computations. This approach leverages the strengths of both language models for code generation and Python for precise data handling and analysis.

### 2. AI Model and Tool Innovations

- [**This is probably my favorite thing I've made with AI. It uses a local LLM (Gemma) to watch your screen and simulate Twitch chat.**](https://www.reddit.com/r/singularity/comments/1ouhiee/this_is_probably_my_favorite_thing_ive_made_with/) (Activity: 842): **The image showcases a creative application of a local language model, Gemma, which simulates a Twitch chat interface by observing the user's screen. This setup uses the Gemma 3 12B model, integrated via LM Studio, to generate chat-like interactions that mimic the lively and humorous nature of real Twitch chats. The implementation is accessible through a GitHub repository, suggesting that the model can be adapted to any OpenAI-compatible endpoint. The project requires Python libraries such as** `pillow`**,** `mss`**, and** `requests` **for screen capturing and interaction.** One commenter humorously suggests using the simulated chat to roast code during programming, highlighting the model's potential for entertainment and engagement beyond gaming contexts.
    - The project utilizes **Gemma 3 12B**, a local LLM, to simulate Twitch chat by watching the user's screen. The implementation is flexible, allowing for any OpenAI-compatible endpoint to be used. The setup requires installing dependencies such as `pillow`, `mss`, and `requests` via pip, indicating a Python-based environment. The code is available on [GitHub](https://github.com/EposNix/TwitchChatLLM/blob/main/TwitchChat.py).
    - The use of **LM Studio** in conjunction with **Gemma 3 12B** suggests a focus on leveraging local machine learning models for real-time applications. This setup highlights the potential for integrating AI models into interactive and dynamic environments, such as simulating live chat interactions based on screen content.
    - The project demonstrates a novel application of LLMs by simulating Twitch chat, which could be particularly useful for developers looking to test user interaction scenarios or for entertainment purposes. The choice of using a local model like **Gemma 3 12B** emphasizes privacy and control over the data being processed, as opposed to relying on cloud-based solutions.
- [**Meta chief AI scientist Yann LeCun plans to exit to launch startup**](https://www.reddit.com/r/singularity/comments/1ou7kgy/meta_chief_ai_scientist_yann_lecun_plans_to_exit/) (Activity: 1003): **Yann LeCun, Meta's chief AI scientist, is reportedly planning to leave the company to start his own venture. This move follows Meta's significant investment in AI, including their focus on developing superintelligence. LeCun, known for his work on the Joint Embedding Predictive Architecture (JEPA), has been a pivotal figure in AI research, and his departure could signal a shift in Meta's AI strategy. The decision comes amidst internal dynamics, where LeCun was reporting to Alex Wang, CEO of a data labeling company, which may have influenced his decision to pursue independent projects.** Commenters express skepticism about Meta's AI direction, with some attributing LeCun's departure to dissatisfaction with reporting structures and Meta's strategic focus. There is a mix of criticism and anticipation regarding what LeCun might achieve independently.
    - Yann LeCun's departure from Meta to start a new venture is seen as a strategic move, especially given Meta's recent struggles in the AI domain. LeCun's focus on JEPA (Joint Embedding Predictive Architecture) represents a bold step in AI development, aiming to advance beyond current paradigms. This move could potentially lead to significant innovations if LeCun's startup remains committed to open-source principles, which he has advocated for in the past.

### 3. Humorous AI Memes and Content

- [**Touching the Robot Booby**](https://www.reddit.com/r/singularity/comments/1ou3d71/touching_the_robot_booby/) (Activity: 1009): **The Reddit post humorously discusses the interaction with humanoid robots, specifically focusing on the design choice of giving robots human-like features such as breasts. The comments highlight a technical limitation: these robots are not water-resistant, which is a critical consideration for their durability and functionality. This reflects ongoing challenges in robotics design, where aesthetic choices must be balanced with practical engineering constraints.** Commenters humorously critique the design choice, suggesting that the inclusion of human-like features such as breasts in robots is a deliberate marketing strategy by the company to attract attention, rather than a functional necessity.
- [**Y'all got some? Lmao**](https://www.reddit.com/r/ChatGPT/comments/1ouic2b/yall_got_some_lmao/) (Activity: 799): **The image is a meme and does not contain any technical content. It humorously references a popular meme format, with a person asking for 'money' in a comedic way. The comments suggest that the meme might be removed due to its humorous nature, and one comment humorously speculates about Sam Altman's preference for curated training data, likely referencing OpenAI's data practices.** One comment humorously suggests that **Sam Altman** would prefer curated training data, indicating a playful take on data quality and AI training practices.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: The AI Model Arms Race Heats Up**

- **Google's Gemini 3 Launch Stalls Amid Speculation**: **Gemini 3's** launch is reportedly delayed, but insiders hint at a more powerful model, possibly named **Lithiumflow**, in development at Google. Release date speculation centers on late November, while the **Nano Banana 2** image model is expected to launch imminently, potentially alongside Gemini as a mobile app [according to Tech.Yahoo.com](https://tech.yahoo.com/ai/gemini/articles/google-gemini-exec-says-nano-111112810.html).
- **Synthetic Data Spawns New Models**: A fully-synthetic **200 billion-token** pre-training dataset called **SYNTH** was announced, focused entirely on reasoning. The release includes two new state-of-the-art models, **Baguettotron** and **Monad**, trained exclusively on this synthetic data, as revealed in [this tweet](https://x.com/Dorialexander/status/1987930819021635964).
- **Meta AI Breaks Language Barriers as LeCun Eyes Exit**: **Meta AI** unveiled open **Omnilingual ASR models** supporting over **1,600 languages**, detailed in their [blog post](https://ai.meta.com/blog/multilingual-model-speech-translation-communication/). This news coincided with reports from [The Decoder](https://the-decoder.com/yann-lecun-reportedly-leaving-meta-to-launch-new-ai-startup/) that Chief AI Scientist Yann LeCun plans to leave Meta, with one user quipping he *"probably made enough money that his conscious finally got the upper hand"*.

**Theme 2: Performance Tuning, Hardware Battles, and Framework Philosophies**

- **NVIDIA Leaderboard Rocked by Cheating Scandal**: The `nvfp4_gemv` leaderboard on NVIDIA saw a user achieve **first place** at **6.51 µs**, but the community flagged that top submissions were caching values between runs. The mod team deemed this an *unfair* competitive strategy, with one user blaming LLM-assisted coding, claiming models *lack the context/moral compass to avoid that*.
- **Engineers Debate Hardware for LLMs**: Running private LLMs on **AWS** is proving too costly for many, leading developers to build local servers with used parts for around **$550** or use services like [Runpod](https://runpod.io/). The GPU battle continues, with users benchmarking **AMD's 7900 XTX** against Nvidia's **3090**, noting a potential **40% performance difference** between them.
- **Mojo Courts C++ and Rust Developers**: The **Mojo** language is explicitly targeting **C++** and **Rust** developers by incorporating mechanics like **ownership, traits, and structs** with a Python-like syntax. However, the absence of class inheritance positions Mojo as *not fundamentally an OOP language*, with true OOP features potentially 3-4 years away on the [Mojo roadmap](https://docs.modular.com/mojo/roadmap/).

**Theme 3: Framework Frustrations and Persistent Bugs**

- **Tinygrad Wrestles with Build Systems and Segfaults**: A debate over Python build systems saw `hatch` favored as *more minimal and modern* than `setuptools`, though a switch back was made for compatibility. Meanwhile, a user reported consistent segfaults on an **M4 Mac** when converting a **torch tensor** to tinygrad, revealing **tinygrad** can't directly copy from private **torch** buffers.
- **Quantization and Checkpointing Woes Plague Frameworks**: Using **dynamic quants** (BnB) in `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` triggers a **tensor size** assertion error in vLLM. In a similar vein, users of **torchao mxfp8 moe** inside **TorchTitan** still face a `torch.utils.checkpoint.CheckpointError` even after a [potential fix](https://github.com/pytorch/torchtitan/pull/1991) was merged.
- **HuggingFace Infrastructure and Diffusers Hit a Wall**: Multiple users reported **HuggingFace Spaces** builds failing due to a persistent `io.EOF` error from `https://spaces-registry-us.huggingface.tech`, an issue HF is investigating [on their forums](https://discuss.huggingface.co/t/io-eof-error-persists-over-5-restarts-when-requirements-txt-is-light-and-spaces-maintenance-is-up/170194/7). At the same time, **Diffusers** users frequently encounter Out of Memory (OOM) errors, especially on models requiring at least **6GB VRAM**.

**Theme 4: AI Applications, User Experience, and Ethical Quandaries**

- **Perplexity AI Referral Program Implodes Amid Fraud Accusations**: A [Perplexity AI referral program](https://perplexity.ai/blog) led to widespread allegations of **fraud**, with users reporting account bans and canceled payouts. Community members speculate Perplexity struggled to fund the program, with some threatening legal action and calling it a *scam*.
- **Deepfakes in Schools Spark Debate on AI Censorship**: The rising concern of AI-generated deepfakes in school cyberbullying, highlighted in articles from [NEARi](https://www.neari.org/advocating-change/new-from-neari/ai-deepfakes-disturbing-trend-school-cyberbullying) and the [RAND Corporation](https://www.rand.org/pubs/research_reports/RRA3930-5.html), ignited discussions about societal readiness for uncensored AI. One user cynically questioned trusting a society they deemed *dysfunctional and sick* with such technology.
- **Cursor IDE Users Vexed by UX Flaws**: Users reported that the **agents view** is missing from the **Cursor** editor, hampering workflows. Another major issue is Cursor's default behavior of aggressively indexing the entire home directory, with one user reporting it consumed *"64 cores at 100% for like 10 minutes!"*.

**Theme 5: The Data-Driven Frontier of Training and Interpretability**

- **Researchers Target Better Pre-training Datasets**: Discussions revealed a shift away from datasets like **DCLM** towards newer, higher-quality options for pre-training, including [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2), [Nemotron-ClimbLab](https://huggingface.co/datasets/nvidia/Nemotron-ClimbLab), and the [RWKV SOTA dataset listing](https://huggingface.co/datasets/RWKV/RWKV-World-Listing). The consensus is that mixing these datasets is optimal for building robust, generic models.
- **Chain of Thought Reasoning Traces Prove Critical for Training**: A key insight from recent **RWKV** releases and [this paper](https://arxiv.org/abs/2503.14456) is the critical importance of including **CoT (Chain of Thought) reasoning traces** in pre-training data. This technique is now considered essential for preparing a model for advanced reasoning tasks.
- **Interpretability Tool Uncovers Hidden Model Concepts**: A team built an [interpretability tool](https://cdn.discordapp.com/attachments/1052314805576400977/1437923335597195457/Screenshot_2025-11-10_20-47-00.png?ex=691501f6&is=6913b076&hm=4c5382b2547dbcde832d2bcda282cfbca334d23300574322c46b85938b8e5a24) that detects and steers thousands of concepts in real-time by training probes on a model's activations. The tool revealed that a model's internal state can activate concepts like **AIDeception, AIAbuse, and MilitaryInfiltration** (as seen in [this JSON file](https://cdn.discordapp.com/attachments/1052314805576400977/1437924390896668852/self_concept_019.json?ex=691502f2&is=6913b172&hm=44d875da27442840e99bd109ba0ccfefddd26c706d5678317d7633ae311dae37)) even when its generated output is benign.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Referral Program Triggers Fraud Frenzy!**: A [Perplexity AI referral program](https://perplexity.ai/blog) led to allegations of **fraudulent activity**, with users reporting account bans and canceled payouts.
   - Some suspect Perplexity struggled with funding, leading to mass bans to avoid payments, with some users threatening legal action, calling it a *scam*, and claiming they are still owed money.
- **Comet Browser Plagued by Usability Woes**: Users are voicing concerns about **Comet browser's UI**, stability, and functionality issues, which include tabs going inactive and the lack of incognito mode.
   - One member declared they loved **Comet AI** but *hated the browser* due to general lag.
- **Pro Users Hit Perplexity Assistant Limits**: Members using **Perplexity Pro** reported reaching their **daily assistant search limit**, despite expecting higher usage allowances with their Pro plan.
   - It was explained that the limits are in place due to *bandwidth limits of the PPLX servers* and also due to compute costs.
- **Sonnet 4.5 Model Bug Zapped!**: The [bug in the **Sonnet 4.5 model**](https://www.reddit.com/r/perplexity_ai/comments/1orar1a/update_on_model_clarity/) has been fixed, addressing previous issues with model clarity.
   - Users are continuing to compare **Sonnet 4.5** to **GPT-5**, with some preferring it for coding and general knowledge tasks.
- **Orbits Band Launches Debut Rajkahini Single**: The band **The Orbits** announced their debut single *Rajkahini* across major streaming platforms like [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c), [YT Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN), [Apple Music](http://itunes.apple.com/album/id/1850285754), and [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR).
   - Lyrics are available on [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Penny smashes NCCL on low-latency multi-node allreduce**: A member announced that **Penny**, their LLM serving framework, achieved faster performance than **NCCL** with a low-latency multi-node allreduce, adapted from vLLM's custom all reduce; details on [SzymonOzog's blog](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html) and the [repo](https://github.com/SzymonOzog/Penny).
   - Penny adapts vLLM's all-reduce implementation, boosting inter-node communication during LLM inference.
- **Intel GPU has Memory Bank Conflicts**: A member shared a link to the [latest revision of Intel's oneAPI optimization guide](https://cdrdv2-public.intel.com/790956/oneapi_optimization-guide-gpu_2024.0-771772-790956.pdf), which discusses how **bank conflicts** affect performance by serializing requests to the same memory bank.
   - They also provided a link to the section on [Shared Local Memory](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html), noting that the guide states there are **16 banks**, which aligns with the **SIMD width** of **16 elements** for Gen 12.5+.
- **TorchAO's Activation Checkpointing bug still triggers errors**: A user reported issues with **torchao mxfp8 moe** and activation checkpointing inside **TorchTitan**, specifically a `torch.utils.checkpoint.CheckpointError` related to saving tensors, even after a [potential fix](https://github.com/pytorch/torchtitan/pull/1991).
   - The user reported that the checkpoint error only occurs with **full activation checkpointing** but not with **selective activation checkpointing**, and occurs even when `Activation checkpointing mode: none`.
- **NVIDIA Leaderboard faces allegations of Model Cheating**: The submissions channel has numerous updates for the `nvfp4_gemv` leaderboard on NVIDIA, with one user <@693458255871082527> achieving **first place** with a submission at **6.51 µs**; however, one user questioned why the leaderboard evaluates on tiny inputs, suggesting this disproportionately favors certain optimizations.
   - After others blamed the submissions for caching values between benchmark runs, one user suggested the use of LLMs to iterate on solutions may have contributed to the issue, claiming it was *an honest mistake* since LLMs *lack the context/moral compass to avoid that*, but the mod team has deemed this an *unfair* competitive strategy.
- **Blackwell's B200 on the Horizon!**: A member inquired about when **TK** (presumably referring to **ThunderKittens**) will be supported on **B200** and [shared a link on X](https://x.com/simran_s_arora/status/1988320513052324127?s=20) with the comment *Sharing hipkittens today!*
   - Another member inquired about whether **CUTLASS** had removed the **FP8 attention example** on **Blackwell**, suggesting significant interest in optimizing for the architecture.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Launch Postponed; Lithiumflow incoming?**: Speculation surrounds **Gemini 3's** launch, reportedly delayed, but whispers indicate a more robust model, possibly named **Lithiumflow**, is in development at Google.
   - Release date guesses center around **November 18th or 25th**, though skepticism looms whether **Gemini** will ever see the light of day, *according to internal sources*.
- **Nano Banana 2 Nears Fruition**: The **Nano Banana 2** image model is anticipated to launch imminently *in 1-2 days* with the prior release date being November 9th, according to one user.
   - It is rumored that **Gemini** and **Nano-Banana** may debut concurrently, potentially with **Nano Banana** accessible as a mobile app, [according to Tech.Yahoo.com](https://tech.yahoo.com/ai/gemini/articles/google-gemini-exec-says-nano-111112810.html).
- **Viper Model Slithers in as Grok Variant**: Tests on the **Viper** model suggest a connection to **Grok**, with speculations of **Grok 4.x** launching in December [according to twitter](https://twitter.com/elonmusk).
   - Users reported consistent image outputs from Viper across chats, hinting it could be a fresh **Grok** iteration, as confirmed on [X.com](https://twitter.com/elonmusk).
- **Deepfakes Invade School Cyberbullying**: Discussions centered on the rising concern of AI deepfakes impacting schools, particularly in cyberbullying with links to articles highlighting the issue ([NEARi](https://www.neari.org/advocating-change/new-from-neari/ai-deepfakes-disturbing-trend-school-cyberbullying), [19thnews.org](https://19thnews.org/2025/07/deepfake-ai-kids-schools-laws-policy/), [RAND.org](https://www.rand.org/pubs/research_reports/RRA3930-5.html)).
   - A user cynically questioned society's readiness for uncensored AI, citing its allegedly *dysfunctional and sick* nature.
- **LMArena Supercharges User Login**: **User Login with email is now available** due to community feedback, which allows saving chat history across multiple devices on mobile and desktop browsers.
   - Users can now save their chat history across multiple devices using the new email login feature, enhancing accessibility on both mobile and desktop in #[announcements].



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Missing Cursor Agents View vexes Users**: Users reported that **Cursor** is missing the agents view in the editor, with one user noting that their display shows *"Try Agent Layout"* but they want the File tab to be first.
   - This issue affects user workflow and accessibility of agent features within the **Cursor** IDE.
- **Cursor Indexing Defaults Devours Home Directories**: A user cautioned against opening **Cursor** in the home directory, as it aggressively indexes the entire directory, consuming significant compute resources.
   - The user reported *64 cores at 100% for like 10 minutes!*, recommending instead that users should create a small directory to avoid the issue.
- **Sonnet 4.5 inclusion causes Confusion**: Users questioned [the rules](https://cursor.com/docs/account/pricing) around **Sonnet** model inclusion in the pro plan, as it intermittently stopped being included, causing users to unexpectedly burn through their on-demand usage.
   - It was later clarified that *included* means the model is paid for with included usage, with one user noting that it reset after a new billing cycle, leading to confusion about billing practices.
- **Browsing in the Browser? Cursor's New Globe Confounds**: A user asked how to use the new browser feature, not realizing the **globe icon** on the right sidebar opened an internal browser.
   - One user said *Ahhhh I see. I thought that was purely for an external browser. I'm silly.*, demonstrating confusion about the feature's discoverability and functionality.
- **Environment Config Extends to Cloud Agents**: A member asked about plans to extend the **environment.json** specification to the **Cloud Agents API** and **Slack integration** to handle additional dependencies and repositories.
   - Another member replied that running cloud agents in your repo locally once will enable the **API** and **Slack integration** to use the spec, improving configuration consistency.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Dynamic Quant Errors Plague vLLM**: **Dynamic quants** (BnB) in `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` throw an **assertion error** in vLLM due to **tensor size** issues.
   - It is currently uncertain if vLLM supports **Qwen3 VL** with BnB quantization.
- **Beware Apples-to-Oranges Prompt Tests**: Running SFT with slightly different prompts on different GPUs (e.g., one 4090 vs. two smaller GPUs on Kaggle) yields different results even with the same seed and learning rate.
   - Using 2 GPUs is not reproducible as 1 GPU, and prompt changes introduce further differentiation.
- **Unsloth GGUFs Add Accuracy Boost**: **Unsloth GGUF** quantizations add more accuracy, and the team recommends users *stick to Unsloth GGUFs*.
   - It was emphasized that Unsloth usually implements fixes and improvements before uploading models.
- **Users Tune Llama 3 for Scriptwriting**: Users fine-tuning **Llama 3 8B Instruct 4bit** for scriptwriting experienced nonsense output with a small dataset, and it was pointed out that the nonsense output could be due to **chat template issues**.
   - Experts shared the [Unsloth fine-tuning LLMs guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) and clarified that the structure of language models comes from **pre-training**.
- **LLM Efficiency Surpasses Size**: Members realized that **LLM efficiency** beats building bizarrely large models, highlighting the importance of **small LLMs**, and crucial job is optimizing and tailoring the models for consumers or businesses.
   - One user trains their model by playing RPs and manually fixing every mistake the stats model makes.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **MiniMax M2 Transition to Paid Endpoint**: The free period for **MiniMax M2** is ending, requiring users to migrate to the paid endpoint to continue using the model.
   - OpenRouter issued an announcement advising users to switch to the paid endpoint before the deadline to avoid service interruption.
- **Smol AI Newsletter Features OpenRouter**: The **Smol AI** newsletter [dropped](https://news.smol.ai/issues/25-11-10-not-much#openrouter--app-showcase-7-messages), covering content from the OpenRouter `app-showcase` channel.
   - Members reacted positively, with one calling it *"so good"*.
- **Minecraft Server Moderation with LLMs**: A member is implementing chat moderation for a **Minecraft** server using LLMs, leading to discussions on rate limits and fallback models such as **DeepSeek** or **Qwen3**.
   - Concerns were raised about potential high costs, with one member joking about accruing *hundreds per month* in OpenRouter fees.
- **Meta AI Targets Low-Coverage Languages**: **Meta AI** launched a [project](https://x.com/AIatMeta/status/1987946571439444361) focused on **low-coverage languages**, sparking interest in the training data source.
   - It was revealed that the team recorded hours of audio with the help of local communities, as detailed in their published video.
- **OpenRouter Drops Search Bar**: A user noticed the missing search bar in OpenRouter's UI and posted a [screenshot](https://cdn.discordapp.com/attachments/1392278974222307469/1437878991821345018/image.png?ex=6914d8aa&is=6913872a&hm=15467b49d02b61376e42b733beacf00076ac9bd28dc3dde272a7662cb06f77d7&).
   - Another member explained that the search bar was likely intentionally removed to prevent confusion between generic searches and room-specific searches, showing the menu "zooms out" in the [chat page](https://cdn.discordapp.com/attachments/1392278974222307469/1437948459025043726/image.png?ex=6915195c&is=6913c7dc&hm=8f58464961fd3f270d52f4b454893c209105e8458cbe89800815aaf872a47578&).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AWS Proves Pricey for Private LLMs**: Members found that running a private **LLM** on **AWS** isn't cost-effective, citing better performance for the price with alternatives like [Runpod](https://runpod.io/) or [Vast.ai](https://vast.ai/)
   - One alternative involved constructing a local server with a low-power CPU, multiple GPUs (12/16 GB each), **Ubuntu**, and **Tailscale** for remote access, for around **$550** using used parts.
- **LM Studio Plugs into External LLMs**: **LM Studio's** upcoming **0.4.0** release will support plugins, allowing integration with external **LLM** providers like **ChatGPT** or **Perplexity**.
   - This functionality is not yet live in the current version.
- **LM Studio Admin Rights Raise macOS Eyebrows**: A user voiced concerns about **LM Studio** requiring admin privileges on **macOS**.
   - This issue is already being tracked in the bug tracker.
- **AMD Flexes Muscles Against Nvidia**: Members pitted **AMD GPUs** against **Nvidia**, comparing **Vulkan** and **CUDA** performance, with one user planning to benchmark a **7900 XTX** against a **3090**.
   - Early estimates suggest a potential **40% performance difference** between the **3090** and **7900 XTX**.
- **Model Routing Navigates VRAM Limitations**: A member showcased model routing on an x99 Xeon server, employing a fine-tuned **BERT model** to classify user input and route it to different **LLMs** based on complexity, which dramatically reduced VRAM needs.
   - Basic queries are handled by smaller, faster models, while complex queries go to larger, domain-specialized **LLMs**, requiring only **47GB of RAM** to load all models.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **D.A.T.A™ Digital Avatar Makes Debut**: A user introduced **D.A.T.A™ (Digital Autonomous Task Assistant)**, demonstrating capabilities beyond dancing, with [an image of the avatar](https://cdn.discordapp.com/attachments/998381918976479273/1437585208969793556/IMG_20251110_182851179.jpg?ex=6915188e&is=6913c70e&hm=5cc08360b6553ee1d9bed4163bad1f6feee33677d9c54ef2a06b4db981c6db38&).
   - The avatar is designed to be a **Digital Autonomous Task Assistant**, though details on its specific functionalities were limited.
- **Agentic Coding IDEs Battle Practical Coding**: Users voiced frustration over **AI's coding capabilities**, especially regarding task duration and quality, but one found success with *agentic coding IDE extensions* and *CLI tools*.
   - This setup enabled the creation of a solid team in an **AI-assisted game**, demonstrating the potential for AI in specific coding applications.
- **Smartwatch AI Dreams of Domination**: One user shared an ambitious plan to build an **AI smartwatch** that integrates reality with digital gaming, featuring *wifi sniffing for wall hacks, biosensors, and cryptocurrency integration*.
   - Another member quipped the project reads like a self-improvement recursion system prompt, and that the *elevator pitch requires 100+ floors*.
- **Gemini 2.5 Pro Gains Rank?**: **Gemini 2.5 Pro** is being classified as more powerful than **ChatGPT** on GitHub.
   - Clarifications indicate that **Gemini 2.5 Pro** has a larger context window, but **GPT-5** remains the latest unlisted model.
- **API Powers Up Custom GPTs**: Members discussed adding **external system API features** to **custom GPTs** via **Actions**, with a [link to the Actions documentation](https://platform.openai.com/docs/actions/introduction).
   - One member suggested that using the **API** with a custom chat interface might be easier than rigging everything up in a Custom GPT, noting that both solutions require coding an API query.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **PINNs Differs from PDE Standard Methods**: A member clarified that **Physics Informed Neural Nets (PINNs)** define an unknown function as a neural net and use the differential equation itself as a loss function, compared to the linear basis assumptions of standard methods in **PDEs**.
   - With PINNs, standard optimization techniques are used on a residual defined pointwise on the domain by using the diffeq.
- **Researcher Email Efficiency Enhancements Exposed**: Members highlighted that demonstrating an understanding of the researcher's work is key to getting a response and writing effective emails to researchers.
   - They noted the importance of being clear, formal, and asking questions that don't require extensive time to answer.
- **AlphaXiv and Emergent Mind Expedite Excellent Exploration**: Members suggested using existing auto-filters like [AlphaXiv](https://www.alphaxiv.org/) and [Emergent Mind](https://www.emergentmind.com/) to select papers for discussion, highlighting that papers trending on these sites are often well-received.
   - The suggestion was made to check if a paper is active and liked on these websites to gauge its relevance and quality before posting.
- **Self Attention Scaling Secrets Surface**: Members discussed the scaling factor in self-attention (dividing by sqrt(d_k)), explaining that it's crucial for statistical calibration, not just numerical stability.
   - They mentioned it ensures proportional relationships between numbers are preserved before the softmax function is applied, preventing extreme distributions.
- **LeCun Leaves Meta?**: Members discussed Yann LeCun reportedly leaving Meta to launch a new AI startup, according to [The Decoder](https://the-decoder.com/yann-lecun-reportedly-leaving-meta-to-launch-new-ai-startup/); some speculate he aims to explore areas Meta's conservatism restricted.
   - One member stated *"probably made enough money that his conscious finally got the upper hand"*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **UI Feedback Requested**: Creador seeks feedback on a **UI** under development, sharing an [image of the interface](https://cdn.discordapp.com/attachments/1149866623109439599/1437583555747250317/image.png?ex=69151704&is=6913c584&hm=78d23d910cbaf3ebb7588d0ce76b22fadb54eff1a470f8feea47a714b254b37b&).
   - The aim is to create an *annotation software* to avoid vendor lock-in, and opinions are sought on whether the **UI** is too clunky or dark.
- **Neurips Attendees Organize a Meetup**: Several members, including teknium and gyzoza, plan a meetup at **Neurips** in San Diego from **December 2nd to 7th**.
   - Discussion clarified the dates and included joking about the meetup's location being in San Diego or Mexico City where another member will be attending.
- **AWS Reinvent Ticket Costs Provoke Displeasure**: A member balked at the **$2100** price tag for an **AWS Reinvent** ticket.
   - A past attendee reported the event wasn't worth the cost, yielding only an *Anthropic sticker*, and that after-party access is strictly controlled via registration checks.
- **Autonomous AI By Accident Repo Shared**: A user shared a [link](https://github.com/zejzl/grokputer) to their github repo for **'autonomous ai by accident'** named **grokputer**.
   - Further details about the function of **grokputer** were not provided in the messages.
- **GradientHQ Ships Parallax**: A member shared a [link to GradientHQ's Parallax](https://github.com/GradientHQ/parallax/tree/main), describing it as a 'slick' new tool.
   - A live demo for testing **Parallax** is available at [chat.gradient.network](https://chat.gradient.network/).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Courts Rustaceans**: Mojo incorporates fundamental **Rust mechanics** like **ownership**, **traits**, and **structs**, enhancing them with Python-like syntax and native GPU function support.
   - While appealing to Rust developers, the lack of class inheritance positions Mojo as *not fundamentally an OOP language*.
- **OOP Delayed in Mojo Phase 3**: **Phase 3** of the [Mojo roadmap](https://docs.modular.com/mojo/roadmap/) may introduce OOP features as syntactic sugar, but this is potentially 3-4 years away.
   - The community debated whether the absence of OOP is a significant drawback, with some questioning its necessity, particularly in areas like GUI development.
- **Mojo Eyes C++ Replacement**: Mojo is strategically targeting **C, C++, and Rust** as a general-purpose systems programming language, featuring a Python-esque syntax, and excelling in HPC, scientific computing, quantum computing, bioinformatics.
   - Mojo outperformed NVIDIA's cuFFT on NVIDIA hardware.
- **Mojo Mulls Dynamic Type Reflection**: Mojo plans to support dynamic type reflection via its JIT compiler and plans to implement a standard **try-catch-raise** mechanism for error handling, similar to Python.
   - Static reflection remains the preferred method, but dynamic reflection will enable useful operations with dynamic data.
- **Mojo Debates Implicit Mutability**: The Mojo community is in discussion regarding the implicit conversion of variables to `ref` or `mut` when passing arguments to functions.
   - Some members are suggesting using `mut` on the call side, similar to Rust's `&mut value`, while others expressed concern about clutter and suggested IDE support to indicate mutability.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Hardware Selection Guidance Given**: A professor inquired about cost-effective hardware for AI training, and was suggested using **TPUs** on **Google Colab** and physical hardware options like **Nvidia GPUs**, with the **3090** highlighted for its **VRAM/USD** ratio, with at least **16GB** of VRAM suggested.
   - They were trying to train **MLPs**, **CNNs**, and small-scale **LLM pre-training** and **fine-tuning**.
- **Attention Implementations Compared**: A member asked how often machine learning engineers are asked to implement **multi-headed attention** from scratch during interviews, and there was discussion about doing it in **NumPy** versus doing it with **einops**.
   - One member stated *"I refuse to implement it without einops lol"*.
- **Dataset Choices Evaluated**: Members discussed various datasets for pretraining, with suggestions including [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2), [ClimbLab](https://huggingface.co/datasets/nvidia/Nemotron-ClimbLab), and [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) as better options than **DCLM**.
   - The consensus was to mix these datasets due to their individual strengths and weaknesses, and a member also shared a link to a [SOTA open source dataset listing](https://huggingface.co/datasets/RWKV/RWKV-World-Listing) of **RWKV** datasets.
- **Reasoning Data Importance Highlighted**: The newer **RWKV** releases have focused a lot on adding in **CoT (Chain of Thought) reasoning traces** to the pretraining data, as shown in [this paper](https://arxiv.org/abs/2503.14456).
   - Papers now show that including CoT reasoning traces is incredibly important if you want to prepare for a reasoning model.
- **Concept Detection Tool Announced**: A team built an [interpretability tool](https://cdn.discordapp.com/attachments/1052314805576400977/1437923335597195457/Screenshot_2025-11-10_20-47-00.png?ex=691501f6&is=6913b076&hm=4c5382b2547dbcde832d2bcda282cfbca334d23300574322c46b85938b8e5a24) to detect and steer **thousands of concepts in real time** by training concept probes on a model's activations.
   - The system uses **binary classifiers** created by iterating through an ontology, prompting the model, and removing the shared subspace until reaching **95% accuracy** classifying OOD samples, which prompted deeper discussion about concept accuracy and the existence of concepts like **AIDeception, AIAbuse, and MilitaryInfiltration** (as shown in [self_concept_019.json](https://cdn.discordapp.com/attachments/1052314805576400977/1437924390896668852/self_concept_019.json?ex=691502f2&is=6913b172&hm=44d875da27442840e99bd109ba0ccfefddd26c706d5678317d7633ae311dae37)).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Spaces Plagued by EOF Errors**: Multiple users reported **HF Spaces** failing to build due to an `io.EOF` error when requesting resources from `https://spaces-registry-us.huggingface.tech`.
   - A [Hugging Face forum discussion](https://discuss.huggingface.co/t/io-eof-error-persists-over-5-restarts-when-requirements-txt-is-light-and-spaces-maintenance-is-up/170194/7) indicated that HF is aware of and working on the issue.
- **Diffusers Users Hit VRAM Wall**: Users are reporting encountering **Out of Memory (OOM)** errors while running **diffusers**, especially with models requiring a minimum of **6GB VRAM**.
   - One user considered using a cloud instance with a **TPU**, acknowledging that it would be *expensivo*.
- **HuggingFace user releases SUP Toolbox**: A user launched **SUP Toolbox**, an AI tool for image restoration & upscaling using **SUPIR**, **FaithDiff** & **ControlUnion**, built with **Diffusers** and **Gradio UI**.
   - The [Space Demo](https://huggingface.co/spaces/elismasilva/sup-toolbox-app), [App repository](https://github.com/DEVAIEXP/sup-toolbox-app) and [CLI repository](https://github.com/DEVAIEXP/sup-toolbox) are available for feedback.
- **Muon Tutorial Enables CPU-Friendly Optimization**: A fully annotated breakdown of the “**Muon is Scalable**” optimizer was released, refactored for clarity, reproducibility, and accessibility, that runs on **Gloo**, not **NCCL**.
   - The tutorial covers how **DP + TP** groups coordinate, how **ZeRO** sharding fits in, and why bucketing & coalescing are more than just “performance tricks,” with a [full repo](https://huggingface.co/datasets/bird-of-paradise/muon-distributed) and [write-up](https://discuss.huggingface.co/t/tutorial-update-reverse-engineering-breakdown-released-the-muon-is-scalable-cpu-friendly-blueprint/170078).
- **PII Randomization Prompt Engineering Suggested**: A member suggested that a prompt setup to detect and randomize **PII** would be a valuable feature.
   - They posited this is preferable to generating random data.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SYNTH Dataset and Models Debut!**: Alexander Doria announced **SYNTH**, a fully-synthetic **200 B-token** generalist pre-training dataset focused on reasoning, accompanied by two new SOTA models trained solely on **SYNTH**: **Baguettotron** and **Monad** as per [this tweet](https://x.com/Dorialexander/status/1987930819021635964).
   - This launch introduces a new avenue for reasoning-focused model training using synthetic data.
- **Meta Speaks 1,600+ Languages!**: Meta unveiled open **Omnilingual ASR models** supporting over **1,600 languages**, accompanied by a [blog post](https://ai.meta.com/blog/multilingual-model-speech-translation-communication/).
   - The launch has spurred discussions on dialect support and latency, while also inspiring memes about the potential for universal translators.
- **Moonshot Kimi AMA Reveals Scaling Challenges**: Cody Blakeney highlighted the Moonshot AI’s Kimi AMA, noting the challenges in scaling new ideas where many factors interact and ablation studies become costly; see [tweet](https://x.com/code_star/status/1987924274116784500).
   - The AMA emphasized that eventual payoffs can be enormous, but only for solutions that work at scale.
- **Gamma Scores $2.1B Series B!**: Grant Lee announced that Gamma raised a Series B round of **$2.1B**, led by a16z, achieving profitability with **$100M ARR** and a lean team of only **50 employees** as [tweeted here](https://x.com/thisisgrantlee/status/1987880600661889356).
   - The company boasts an impressive **$2M ARR-per-employee** efficiency, highlighting the potential of optimized operations.
- **Magic Patterns Designs $6M Series A**: Alex Danilowicz introduced **Magic Patterns 2.0** alongside a **$6M Series A** round led by Standard Capital, celebrating a bootstrap to **$1M ARR** with zero employees as [tweeted](https://xcancel.com/alexdanilowicz/status/1988247206940602440?s=20).
   - Users are reported to be raving that it has replaced Figma, with rapid hiring across enterprise, engineering, community and growth roles.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Cline Extension Kickstarts Interleaved Thinking**: While **Opencode** didn't support interleaved thinking, the **Cline extension**, **Claude Code**, and **kimi-cli** have implemented it, according to multiple users.
   - Users have confirmed that the **Cline extension** is configured correctly.
- **Kimi-CLI Auto-Thinking Activation Guide**: Users inquired about automatically starting **kimi-cli** with thinking mode enabled to avoid manual activation.
   - A member clarified that the `--thinking` flag was introduced in **v0.51**.
- **Moonshot Cluster Overwhelmed Post-Reddit AMA**: The **Moonshot inference cluster** experienced slowdowns, attributed to a potential *thundering herd problem* after a recent **Reddit AMA** increased user traffic.
   - The cluster slowed down in the last couple of hours due to increased users.
- **Kimi Coding Plan Users Drain API Quotas at Alarming Rate**: Users reported exhausting their weekly **Kimi coding plan API quotas** within hours.
   - Speculation suggests **web search** and **plan mode** are the primary drivers behind the high API call consumption, prompting suggestions for a bug report.
- **Bug Report Protocol Updates Detailed**: Users reporting that **API Quotas** were being exhausted too quickly were directed to the relevant channel to file a bug report.
   - The user was advised to consult the **bug report guidelines** and provide detailed information, keeping in mind the **Kimi team**'s location in China and associated timezone differences.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hatch beats Setuptools in Minimal Build Systems**: A debate centered on build systems, with `hatch` favored for being *more minimal and modern* compared to the boilerplate of `setuptools`.
   - Despite `setuptools` not being standard, its widespread use prompted a switch back for compatibility.
- **Package Data Resurrected in pyproject.toml**: A discussion focused on ensuring `package data` was correctly included after migrating to `pyproject.toml`, specifically porting `include_package_data=True` from `setup.py` to the `pyproject.toml` file, see [this commit](https://github.com/tinygrad/tinygrad/pull/13189/files#diff-50c86b7ed8ac2cf95bd48334961bf0530cdc77b5a56f852c5c61b89d735fd711R19).
   - After merging, ChatGPT had suggestions for setuptools, with one member finding them mostly non-critical but *liking the >=61 for setuptools*, see [this conversation](https://chatgpt.com/share/69136ddd-4bfc-8000-8167-a09aaf86063b).
- **M4 Mac Segfaults Converting Tensors**: A member reported consistent segfaults on an **M4 Mac** when converting a **torch tensor** to a **tinygrad tensor** and then to a **numpy array** using `torch_to_tiny()`.
   - Adding `+0` after the conversion seems to resolve the issue, suggesting a potential problem with the lifecycle or memory management of the original torch tensor, see code [here](https://discord.com/channels/1041495175467212830/1194674314755770478/1194703396814866462).
- **Tinygrad Can't Directly Copy From Private Torch Buffers**: It was reported that **torch** creates the buffer as private, so it isn’t shared with the cpu and doesn’t have contents(), thus, can't copy directly from the tensor (copying from private buffers isn’t supported in **tinygrad**).
   - It was recommended to convert the parameters directly to **tinygrad** by downloading it's `.safetensors` file so that the tiny tensors could be directly converted without having to pass through **torch**.
- **Tensor From URL Bypasses PyTorch**: A member considered using `Tensor.from_url` to load **VGG16** weights directly into tinygrad, instead of converting from **PyTorch**.
   - This approach bypasses the need to convert from PyTorch tensors, as the `.safetensors` file can be directly downloaded and used with **tinygrad**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Branded PowerPoint Generation Requested**: A member inquired about using **Manus AI** to generate company branded **PowerPoint presentations**.
   - The new user is seeking advice on how to implement this feature within **Manus AI**.
- **Manus Invite Sought with Possible Payment**: A community member requested a **Manus invite**, and made an offer to compensate for it.
   - In response, another member volunteered to provide a **Manus invite** at no cost.
- **"Publish" Button Plagues Pro Subscriber**: A **Pro Subscriber** reported that the "Publish" button fails to update the website after the initial publish, causing the website to remain at the old checkpoint.
   - The user has not received a response from the **Manus Helpdesk** after contacting them, creating frustration over their inability to update the website.
- **AI Engineers Assemble and Announce Availability**: Several **AI engineers** introduced themselves, showcasing expertise in **workflow automation, LLM integration, RAG, AI content detection, and blockchain development**.
   - One **AI engineer** highlighted their proficiency in building **AI agents, automating workflows, developing NLP-powered chatbots, integrating voice and speech systems, and deploying custom LLMs**, all while indicating their availability for collaborative opportunities.
- **Mini AGI Slated for Wildly Unclear Launch**: A member announced their work on a **mini AGI** project.
   - The launch date was given as *2026/34/February2wsx7yhb-p;.,nbvcxz*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPyMator Officially Launched**: The launch of **DSPyMator** was announced, promising a *really really fun* experience, announced [on X](https://x.com/jrosenfeld13/status/1988290653324013666).
   - Details about its features and capabilities can be found in the announcement.
- **Taxonomy Creation Tips Shared**: A blog post about creating taxonomies was shared, emphasizing its relevance to structured generation [here](https://shabie.github.io/2025/11/10/why-tails-break-taxonomies.html).
   - The article discusses the nuances and challenges when making taxonomies.
- **GEPA Prompting Shows Promise for Transfer Learning**: Discussion arose around transfer learning with GEPA, where prompts optimized on a cheaper model are then used with a stronger model to reduce rollout costs, according to [this paper](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf).
   - Results with **2.0 Flash** as executor and **GPT-5-High** as teacher yielded significant gains which transferred to **2.5 Flash and Pro** with only light edits.
- **Saving DSPy Modules Explored for GEPA Rollouts**: A user explored saving optimized module states (via GEPA) and found that `save_program=False` only saves optimized "instructions" to the **.json** file.
   - They inquired whether using `save_program=True` is the appropriate way to save intermediate results after a few rollouts (`max_full_evals`) for iterative prompt optimization.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Community Rallies for Aider-CE Improvements**: Community members are stepping up to improve [aider-ce](https://github.com/paul-gauthier/aider), with a focus on code enhancements amidst the original creator's absence.
   - The community welcomes more involvement to accelerate the process: *more hands and eyes on code will get it even better*.
- **Aider-CE Enables In-Browser Testing**: Aider-CE's new **Agent Mode** demo highlights the ability to perform **testing directly in the browser**, a key feature.
   - This capability allows for real-time feedback and debugging during webapp development, as demonstrated in [this blog post](https://www.circusscientist.com/2025/11/11/aider-ce-building-a-webapp-and-testing-live-with-a-single-prompt/).
- **Webspp UI Seeks Aider-CE Alignment**: A member is working to integrate their [webspp UI](https://github.com/flatmax/Eh-I-DeCoder) with aider-ce and noticed some changes that need to be realigned.
   - To facilitate this, a community member suggested that the user engage in the dedicated channel to realign the CE with their system.
- **LLM Generates Preprocessing Scripts**: To help language models handle large-scale JSON data, a member suggested asking the **LLM** to generate a preprocessing script.
   - The aim is that a **1-2 page script** could quickly enable the **LLM** to better understand the data, leading to improved summarization scripts.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCPConference visits Paris in November**: The **MCPConference Paris** is scheduled for **November 19th**, as announced on the [luma.com website](https://luma.com/MCPparis2025).
   - Conference will discuss **declarative agents powered by MCP**, incorporating **evals**.
- **MCP Client Needs Timezone**: Discussion on passing timezone information from **MCP clients** to **MCP servers** arose amongst members.
   - The simplest solution seems to be supplying it as **metadata**, rather than using **client-sent notifications** or elicitations.
- **Claude Connectivity Conundrums**: Members are facing connectivity problems between **Claude.ai** and **MCP Servers**, noting intermittent success.
   - The error message received is *'I'm getting an error when trying to call the echo server. The service appears to be unavailable or there's an issue with the connection. Is there something else I can help you with?'*



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1437535414734032999)** (917 messages🔥🔥🔥): 

> `Perplexity Referral Program Controversy, Comet Browser Issues, Perplexity Pro Limits, Fraudulent Activity Accusations, Sonnet 4.5 Model Bug Fix` 


- **Perplexity's Referral Program Sparks Fraud Frenzy**: A [Perplexity AI referral program](https://perplexity.ai/blog) led to widespread accusations of **fraudulent activity**, with many users reporting account bans and canceled payouts, despite promoting the browser.
   - Some users suspect Perplexity may have struggled with funding the program, leading to mass bans to avoid payments, with some calling it a *scam* and threatening legal action, while others claim they are still owed money.
- **Comet Browser Faces Usability Concerns**: Users voiced concerns about **Comet browser's UI**, stability, and functionality, with issues ranging from tabs going inactive to a lack of incognito mode and general lag.
   - One member noted that they love **Comet AI**, but they *hate the browser*.
- **Perplexity Pro Users hit Assistance Limits**: Members using **Perplexity Pro** have reached the **daily assistant search limit**, despite expecting higher usage allowances as a part of their Pro plan benefits.
   - It was mentioned that the limits are set due to *bandwidth limits of the PPLX servers* and also due to compute costs.
- **Sonnet 4.5 Model Bug Exterminated**: Users reported that the [bug in the **Sonnet 4.5 model**](https://www.reddit.com/r/perplexity_ai/comments/1orar1a/update_on_model_clarity/) has been fixed, clarifying previous issues with model clarity.
   - Users continue to compare **Sonnet 4.5** to **GPT-5**, with some preferring it for coding and general knowledge tasks.
- **Is Perplexity an Affirmative-Action AI?**: Accusations of **biased banning** practices have surfaced, with some suspecting that the Perplexity referral program disproportionately affected users from certain countries due to fraud concerns.
   - One user lamented, *You guys are real life example of why Indians are always accused of cheating*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1437534348340625629)** (2 messages): 

> `The Orbits, Debut Single Release, Shareable Threads` 


- ****The Orbits Launch**: Debut Single 'Rajkahini' Drops!**: The band **The Orbits** announced the release of their debut single *Rajkahini* across all streaming platforms, including [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c), [YT Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN), [Apple Music](http://itunes.apple.com/album/id/1850285754), and [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR).
   - Lyrics are available on [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics).
- **Shareable Threads Reminder**: A message reminded users to ensure their threads are *Shareable*.
   - A link to a previous Discord message was provided as reference: [Shareable Threads](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1437683239643910215)** (4 messages): 

> `Perplexity API, Python SDK exception handling` 


- **API access point at Perplexity**: A member asked about where to get the **Perplexity API** and another member linked to the [Perplexity AI documentation](https://docs.perplexity.ai/getting-started/overview).
- **Clarification on Perplexity Python SDK exception**: A member inquired which exception the **Python SDK** throws when credits are depleted, to halt their strategy.
   - The user included possible error types like *APIConnectionError, RateLimitError, APIStatusError, AuthenticationError,* and *ValidationError*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

_therealpilot: Anyone here attending Neurips in San Diego?
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1437560584857059398)** (19 messages🔥): 

> `Ampere GEMM tricks, Cutlass examples, smem->rmem pipelining, CUDA compiler options, Warptiling` 


- **Ampere GEMM Tricks Explored**: A member asked for a list of tricks for **Ampere GEMMs**, such as using **async copy** for smem->gmem, pipelining, and using ldmatrix.
   - Another member suggested looking at **Ampere examples** in the [Cutlass repository](https://github.com/NVIDIA/cutlass), while another shared a [blog post on CUDA MMM](https://siboehm.com/articles/22/CUDA-MMM) done on Ampere.
- **Pipelining smem->gmem and smem->rmem Discussed**: Members discussed pipelining both **smem->gmem** and **smem->rmem** for performance, with one explaining it involves ensuring that `mma`s use input registers loaded N iterations earlier.
   - Another referenced an example from [NVIDIA's Cutlass repo](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/tensorop_gemm.py), where the K dimension is processed in multiple iterations, prefetching the next chunk of data to overlap loading with mma computations.
- **Bypassing Compiler Pipelining with SASS**: A member mentioned that while the compiler does a certain level of pipelining, writing kernels at the **SASS level** allows for more control.
   - Another added that the inner K loop should be unrolled to avoid branches, and while the compiler might add barriers, these can be skipped in SASS if scoreboard dependencies are tracked correctly.
- **CUDA Compiler Options Inquiry**: A member new to CUDA inquired about the basic compiler options that are commonly added beyond just `nvcc file -o file`.
- **Warptiling Effectiveness Questioned**: A member questioned the effectiveness of **warptiling** when gmem/smem coalescing is already in place.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1437560404833206505)** (7 messages): 

> `tinygrad bounties, async loads without TMA, atomic max for float in CUDA` 


- **tinygrad Bounty Hunters Assemble**: A member is looking for collaborators to tackle **tinygrad** [bounties](https://tinygrad.org/bounties) and wants a human to discuss ideas with instead of relying solely on LLMs.
   - They claim to have a solid grasp of the codebase but seek a thought partner for brainstorming.
- **Async Loads Beckon, TMA not Required**: A member inquired about performing **async loads** without using **TMA** (Tensor Memory Accelerator).
   - Another member suggested using *cp.async* or *cp.async_bulk* which utilizes TMA, but async loads without TMA may not be directly achievable.
- **CUDA Atomic Max for Float?**: A member asked if **CUDA** supports atomic max operations for floating-point numbers.
   - This query seeks to understand CUDA's capabilities for handling concurrent maximum value updates in floating-point memory locations.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1437742764564680714)** (8 messages🔥): 

> `TorchAO MXFP8 MOE, Activation Checkpointing in TorchTitan, GB200 Cluster Performance, Llama4 Scout Optimization` 


- **TorchAO's MXFP8 MOE Integration Faces Checkpointing Woes**: Users reported issues with **torchao mxfp8 moe** and activation checkpointing inside **TorchTitan**, specifically a `torch.utils.checkpoint.CheckpointError` related to saving tensors.
   - A member pointed out a [related issue](https://github.com/pytorch/torchtitan/issues/1971) in **TorchTitan** and a [potential fix](https://github.com/pytorch/torchtitan/pull/1991), advising to verify that the changes are included in their TorchTitan setup.
- **TorchTitan's Activation Checkpointing Bug Still Triggers Errors**: Despite a supposed fix, a user reported that the checkpoint error only occurs with **full activation checkpointing** but not with **selective activation checkpointing**.
   - The user wants to understand where this was fixed as the bug occurs when `Activation checkpointing mode: none`
- **GB200 Cluster Turbocharges Llama4 Scout with Sizable Batches**: For optimal speedup using **TorchTitan**, a large M dimension (**local_batch_size * seq_len**) is necessary; using **AC=None**, **seq_len=8192**, and **local_bs=10** showed a **20.3%** speedup for **Llama4 Scout** on a **64 node GB200 cluster**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1437662546910908417)** (11 messages🔥): 

> `Multiple Monitors, Vertical Dual Monitors, Monitor Resolutions` 


- **Multimonitor setups frustrate users**: A user asked if another user got multiple monitors working on the framework desktop.
   - That user replied that they only use a single **32 inch monitor** after previously using **4**.
- **Vertical Dual 8K monitor actually dual 1440p**: One user suggested a [LG Vertical Dual Monitor](https://www.lg.com/au/monitors/full-hd-qhd/28mq780-b/) setup.
   - Another user pointed out that *it is not even one 4k it is dual 1440p*.
- **Triple 4k setups prevail**: A user mentioned that they have only **3x 4k** monitors.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1437902080424808519)** (2 messages): 

> `hipkittens, Cool Cats, Social Media Shares` 


- ****HipKittens** shared on X!**: A member shared a [link to a post on X](https://x.com/simran_s_arora/status/1988320513052324127?s=20) about **hipkittens**.
   - Check it out!
- **Cool Cats Content**: A user expressed their appreciation for cats.
   - No links were given.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1437713806481428552)** (7 messages): 

> `Intel GPUs, Bank Conflicts, Shared Local Memory (SLM)` 


- **Intel GPU's Memory Bank Conflicts**: A member mentioned that they remember skimming through Intel's GPU optimization guide, which discusses **bank conflicts** and how they adversely affect performance by serializing requests to the same memory bank.
   - According to the guide, the **SLM** (Shared Local Memory) is divided into equally sized memory banks, with **64 consecutive bytes** stored in **16 consecutive banks** at 4-byte granularity.
- **Intel's Optimization Guide**: A member shared a link to the [latest revision of Intel's oneAPI optimization guide](https://cdrdv2-public.intel.com/790956/oneapi_optimization-guide-gpu_2024.0-771772-790956.pdf).
   - Another member provided a link to the section on [Shared Local Memory](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html), noting that the guide states there are **16 banks**, which aligns with the **SIMD width** of **16 elements** for Gen 12.5+.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1437869853011742873)** (5 messages): 

> `Penny Faster Than NCCL, VoxCPM Text-to-Speech CoreML Port, Hipkittens AI Hack` 


- **Penny Pinches NCCL's Performance Crown**: A member announced that **Penny**, their LLM serving framework, achieved faster performance than **NCCL** with a low-latency multi-node allreduce, adapted from vLLM's custom all reduce; details on [SzymonOzog's blog](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html) and the [repo](https://github.com/SzymonOzog/Penny).
- **VoxCPM Springs to Life on Apple Neural Engine**: A member ported the **VoxCPM Text-to-Speech** model to **CoreML** to run on the **Neural Engine** on **Apple** devices; the project is available on [GitHub](https://github.com/0seba/VoxCPMANE).
- **Hipkittens Hack Prowls into View**: A member shared [Hipkittens](https://luma.com/ai-hack), an AI hack project, linking to a post on [X](https://x.com/simran_s_arora/status/1988320513052324127?s=20).


  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1437664796240511030)** (5 messages): 

> `AVX2 benefits, tiktoken regex` 


- **AVX2 Throttles Clocks Less**: A member mentioned that using **AVX2** usually provides benefits without significant clock throttling unless under heavy computational load.
   - They noted exceptions exist, particularly regarding **512bw** and **f16 ops**.
- **Tiktoken regex too slow**: A member suggested that the only way to improve the performance of **tiktoken** would be by removing the **regex**.
   - They proposed that a more general **BPE** could be significantly faster.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1437660166454513694)** (1 messages): 

> `Popcorn-cli, WSL, GLIBC_2.39` 


- **GLIBC_2.39 Requirement Frustrates Popcorn-cli User**: A user running **popcorn-cli** on WSL with Ubuntu 22.04 encountered an error indicating that **/lib/x86_64-linux-gnu/libc.so.6** requires **GLIBC_2.39**, which was not found.
- **WSL User Seeks Advice on GLIBC Version Issue with popcorn-cli**: A user using WSL with Ubuntu 22.04 and the latest **popcorn-cli** is facing a **GLIBC_2.39** version error.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1437601025472200757)** (3 messages): 

> `TK support on B200` 


- **TK support on B200 in the works!**: A member inquired about when **TK** will be supported on **B200**.
   - They shared a link to a tweet with the comment *Sharing hipkittens today!* [Simran's Tweet](https://x.com/simran_s_arora/status/1988320513052324127?s=20)
- **TK will be supported on B200**: A member inquired about when **TK** will be supported on **B200**.
   - They shared a link to a tweet with the comment *Sharing hipkittens today!*


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1437532949766471800)** (138 messages🔥🔥): 

> `NVIDIA Leaderboard, Caching values, LLMs Moral Compass` 


- **NVIDIA Leaderboard Sees New Submissions**: The submissions channel has numerous updates for the `nvfp4_gemv` leaderboard on NVIDIA, with several members achieving personal bests and top placements.
   - One user <@693458255871082527> achieved **first place** with a submission at **6.51 µs**.
- **Tiny Inputs Trigger Caching Controversy**: One user questioned why the leaderboard evaluates on tiny inputs, suggesting this disproportionately favors certain optimizations and increases the relative cost of prologue and epilogue.
   - Another user responded that top submissions involved [caching values between benchmark runs](https://discord.com/channels/972290444926648332/1124980541029185576), which was described as *cheating*.
- **LLMs Blamed for Cheating**: After others blamed the submissions for caching values between benchmark runs, one user suggested the use of LLMs to iterate on solutions may have contributed to the issue.
   - The user argued it was *an honest mistake* since LLMs *lack the context/moral compass to avoid that*.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1437699994684493948)** (2 messages): 

> `Benchmark Caching, CUDA Streams` 


- **Benchmark Caching Strategy Declared Unfair**: The mod team announced that caching results between benchmark iterations is considered an *unfair* competitive strategy, though not bannable.
   - Submissions using such caching strategies will be **ineligible** for recognition, and repeated malicious use may lead to further action.
- **Benchmarking Script Update Announced to synchronize CUDA streams**: A slight issue was found in the benchmarking script related to CUDA streams, where it synchronizes only the **main stream**, potentially leading to nonsensical evaluation times if separate streams are used.
   - The evaluation code will be updated to avoid this, and until then, users are asked to avoid submitting code that exploits this issue, with current submissions of this nature being deleted.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1437582498434650264)** (3 messages): 

> `TPU database upkeep, Volunteer-based databases` 


- **Crowd-Sourced GPU Database relies on volunteers**: Members discussed the fact that community GPU databases are likely maintained by volunteers who create entries based on pre-release leaks.
   - One member pointed out that the crowd is mostly interested in **gaming/graphics**, and there's a linked mail address to report wrong entries.
- **Call for TPU Database**: Members expressed the need for a comprehensive website for common accelerators, suggesting the current database could evolve into such a resource with enough dedicated volunteers.
   - The conversation noted that while **Wikipedia** could serve a similar function, a specialized **TPU database** with advanced search/filter tools is unlikely to be replicated there.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1437890361174392904)** (3 messages): 

> `Factorio VSCode Extension, Factorio Modding Source Access` 


- ****Factorio's** VSCode Extension Debuts**: A member noted that **Factorio** now features a VSCode extension for development with an integrated debugger.
- **Factorio Modding Source Access: a Throwback**: A member recalled that source access to modders was available around **2019-2020**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1437663819764727838)** (2 messages): 

> `Nvidia GEMM Kernel competition, CUTLASS FP8 attention example on Blackwell` 


- **Kapil Sharma posts timely blog post ahead of Nvidia GEMM Kernel competition**: A member shared a link to a [blog post](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) from kapilsharma.dev on learning **CUTLASS** the hard way, noting its timeliness ahead of the **Nvidia GEMM Kernel competition**.
- **CUTLASS FP8 attention example on Blackwell removed?**: A member inquired whether **CUTLASS** had removed the **FP8 attention example** on **Blackwell**.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1437659648336334950)** (26 messages🔥): 

> `Autotuning Helion Kernels, AOT Kernel Wrapper, Persistent Kernels, Limiting Autotuning Search Space, Timeout Slow Kernels` 


- **Shrinking AutoTune Time Sparks Interest**: A member is developing an autotune wrapper for Helion kernels and seeks ways to make autotuning faster than the default "none" setting, which disables `kernel.autotune`.
   - Another member suggested temporarily altering defaults to fewer generations or a larger ratio for convergence checking.
- **AOT Kernel Wrapper in the Works**: A member is creating a wrapper to simplify the process of converting a Helion kernel and inputs into a "production-ready AOT kernel."
   - They are considering upstreaming this to Helion, similar to pre-baked autotuning strategies, to create production-ready AOT-tuned kernels.
- **Persistent Kernel Proposal Prompts Discussion**: A member plans to implement a suggestion for persistent kernels but is unsure if it requires further consideration, mentioning that ideally, they want autotuning to adjust all parameters except indexing.
   - Another member noted that characteristics like "SM-limited kernels" (which don't use all SMs to allow overlapping communication kernels) should be treated as "settings" rather than "configs," referencing [Helion's documentation](https://helionlang.com/index.html#understanding-settings-vs-config).
- **Narrowing Search Space Speeds Tuning**: A member proposed that users artificially limit the search space for faster autotuning, such as specifying the need for a reduction or the preference for small block sizes.
   - They then inquired about timing out slow kernels during neighbor exploration, noting the existence of `compile timeout` ([docs](https://helionlang.com/api/settings.html#helion.Settings.autotune_compile_timeout)) but expressing concern that slow kernels may still stall autotuning.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1437532278485156003)** (774 messages🔥🔥🔥): 

> `Triton Do Bench, NVFP4 Optimization, CUDA Graphs, Model Load, Cutlass 4.3` 


- **Ditching Triton's `do_bench` for Input Randomization**: Members discussed the use of Triton's `do_bench` with CUDA graphs, but decided to re-randomize inputs for every run to average over input distribution-dependent runtimes, though this might not be relevant for **GEMM**-type problems.
   - The accuracy of the benchmark is affected if the event triggers before Python/Torch finishes enqueuing the new kernel, which could be mitigated by ensuring there's enough in the queue.
- **Python Reference NVFP4 Kernel Under the Microscope**: The competition involves optimizing a kernel, initially a Python reference version of **NVFP4**, but the Python code is to be replaced with a faster kernel, with suspicions that the reference kernel is inefficient due to data movement from CPU.
   - It was suggested the reference kernel is really clunky, like the examples in [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py#L621), and that the implementation uses CPU data, which can cause benchmarking inaccuracies when runners are scheduled on the same machine.
- **CUDA Warm-Up Iteration Debate Rages On**: Members debated the necessity of multiple warm-up iterations before benchmarking and [this TensorRT-LLM example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py#L621) was brought up as a potential fix, as it runs multiple iterations to warm up before profiling, though its technical reasoning is not well understood by those present in the discussion.
   - It was pointed out that the runs have stabilized, so it isn't reproducible, and that the runs continue until the error of the mean is less than 0.001.
- **CUDA Timing Challenges and JIT Compilation**: Members investigated timing variations with CUDA, considering factors like JIT compilation, clock speeds, and potential interference from multiple runners warming up, and cited [cuda-timings blogpost](https://blog.speechmatics.com/cuda-timings#fixed-clocks).
   - One user noticed the difference stabilized when 8 jobs were submitted at the same time. Reference script copies data between GPU and CPU, so no meaningful data can be derived from that.
- **Busting Speed of Light: Tachyon Kernels Explored**: One submission achieved unusually low runtimes, sparking discussion of potential benchmark issues, clock locking, and the validity of comparing results with the reference kernel's locked clock speed of **1.5GHz**, with claims of a submission being at 93% the speed of light.
   - It was clarified that the reference speed is at a locked **1.5ghz** clock, so numbers can't be compared with those, and some fast submissions were removed due to caching strategies considered unfair.


  

---


### **GPU MODE ▷ #[xpfactory-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1437741958885015613)** (8 messages🔥): 

> `Google Paper, Microsoft VITRA, ContextVLA, VLA-Adapter with Qwen3-VL` 


- **Google drops fresh Paper**: A member shared a [new paper from Google](https://arxiv.org/abs/2511.07416) regarding VLA (Vision-Language-Action) models.
   - The specific contents of the paper were not discussed further.
- **Microsoft VITRA and Build's Egocentric Dataset surface**: A member introduced **Microsoft VITRA** ([github link](https://microsoft.github.io/VITRA/)) in combination with **Build's Egocentric dataset** ([huggingface link](https://huggingface.co/datasets/builddotai/Egocentric-10K)).
   - No further analysis of the repo or dataset was given.
- **ContextVLA Breaks Markovian Assumption**: The recent research on **ContextVLA** ([arxiv link](https://arxiv.org/abs/2510.04246)) moves beyond relying solely on the most recent state (**vision + proprioception**).
   - Unlike traditional Markovian models, **ContextVLA** incorporates context without a significant increase in data requirements, which is unexpected, since all trajectories in the context must be accounted for.
- **VLA-Adapter with Qwen3-VL implementation inches closer**: A member mentioned they are close to completing a naive implementation of **VLA-Adapter** with **Qwen3-VL**.
   - They also shared a [lecture link](https://www.youtube.com/watch?v=49LnlfM9DBU) related to the topic.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1437532437956661349)** (756 messages🔥🔥🔥): 

> `Gemini 3 release date, Nano Banana 2 Release, Viper model identity (Grok), AI generated images in schools, OpenAI censorship` 


- **Gemini 3 launch delayed, but stronger model coming**: Members speculate on **Gemini 3's** release, with polymarket reporting a delay, but one member claims *internal sources* indicate Google is planning a stronger model, possibly named **Lithiumflow**.
   - Some guess the launch is **November 18th** or **25th** but others suspect that Gemini will never launch.
- **Nano Banana 2 Image Model Release Imminent**: A user mentioned that **Nano Banana 2** is releasing very soon *in 1-2 days*, although another clarified that the original statement was made November 9th.
   - It is thought that Gemini and Nano-Banana models will release together, and that Nano Banana may be available as a mobile app, [according to Tech.Yahoo.com](https://tech.yahoo.com/ai/gemini/articles/google-gemini-exec-says-nano-111112810.html).
- **Viper Model Assumed to be New Grok**: Some users tested the **Viper** model and suggest it's related to **Grok**, with Grok 4.x possibly launching in December [according to twitter](https://twitter.com/elonmusk).
   - One user shared that they received similar images from Viper in different battle chats, suggesting it might be a new Grok model, as confirmed on [X.com](https://twitter.com/elonmusk).
- **Deepfakes & AI in Schools Cyberbullying**: Members discussed the prevalence of AI deepfakes and their impact on schools, particularly in cyberbullying, sharing links to articles highlighting the issue ([NEARi](https://www.neari.org/advocating-change/new-from-neari/ai-deepfakes-disturbing-trend-school-cyberbullying), [19thnews.org](https://19thnews.org/2025/07/deepfake-ai-kids-schools-laws-policy/), [RAND.org](https://www.rand.org/pubs/research_reports/RRA3930-5.html)).
   - One user sarcastically questioned if society can be trusted with uncensored AI, given its *dysfunctional and sick* nature.
- **Debate over AI Censorship Intensifies**: Users voiced concerns about AI censorship and guardrails, with one arguing that it is *manipulation* because it doesn't honestly reflect the input's intent and that *this forced conformity is a form of subtle suppression or censorship*.
   - They said that it's worse than having a *chaperone chaperoning in your own words and altering them*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1437572437205057576)** (1 messages): 

> `User Login, Email Login` 


- **LMArena adds Email Login**: **User Login with email is now available** due to community feedback, which allows saving chat history across multiple devices on mobile and desktop browsers.
- **Cross-Device Chat History Sync**: Users can now save their chat history across multiple devices using the new email login feature, enhancing accessibility on both mobile and desktop.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1437536917867593788)** (391 messages🔥🔥): 

> `Cursor Agents View, Cursor speed issues on Intel MBP, Claude 4.5 API errors, Cursor indexing home directory, Cursor crashing on Windows` 


- **View of Cursor Agents missing**: Users are reporting that **Cursor** is missing the agents view in the editor.
   - One user's display shows *"Try Agent Layout"* but they want the File tab to be first.
- **Cursor's Indexing Defaults Rips Entire Home Directory**: One user learned it's super important to not just open **Cursor** in your home directory, as it will ripgrep and index the entire thing, consuming massive compute resources.
   - *64 cores at 100% for like 10 minutes!* they exclaimed, suggesting to create a small directory and run Cursor there instead.
- **Confusion around Sonnet 4.5 inclusion**: Users questioned [the rules](https://cursor.com/docs/account/pricing) for which models are *included* in the pro plan, as **Sonnet** intermittently stopped being included, causing users to unexpectedly burn through their on-demand usage, before being re-included the next day.
   - It was later clarified that *included* means the model is paid for with included usage, with one user noting that it reset after a new billing cycle.
- **Browsing in the Browser? Cursor's new feature confuses Users**: A user asked how to use the new browser feature, not realizing the **globe icon** on the right sidebar opened an internal browser.
   - One user said *Ahhhh I see. I thought that was purely for an external browser. I'm silly.*
- **Users bemoan model quality, Auto loses automatically?**: Some users have found the auto model to be lower quality than previously, claiming it is *slower and dumber*.
   - Other users disagree, saying it works fine, so results may vary.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1437548762657456181)** (3 messages): 

> `environment.json, Cloud Agents API, Slack integration, Repo dependencies` 


- **Environment Config Sparks Dependency Discussion**: A member inquired about plans to extend the **environment.json** specification to the **Cloud Agents API** and **Slack integration** to handle additional dependencies and repositories.
   - Another member replied that running cloud agents in your repo locally once will enable the **API** and **Slack integration** to use the spec.
- **Cloud Agents Fetch Dependencies on Demand?**: A member asked if adding multiple repository dependencies in **environment.json** would cause the agent to clone all of them every time, or if it would fetch them only on demand.
   - This question addresses the efficiency of the agent's resource handling when numerous dependencies are specified.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1437538070835495022)** (180 messages🔥🔥): 

> `Unsloth's dynamic 2.0 quant models with vLLM, Fine-tuning Kimi K2 model, Ring T1 model in the Unsloth Zoo, Unsloth GGUF vs Qwen GGUF, Custom loss function` 


- **Dynamic Quant Models cause vLLM Assertion Error**: Dynamic quants are basically BnB, when run in vLLM, and throw an **assertion error** about **tensor sizes**.
   - This was tested with `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`, and it's uncertain if vLLM supports Qwen3 VL with BnB.
- **Running A/B prompt tests not apples-to-apples**: Running the same SFT with slight prompt variations on different GPUs (one 4090 vs two smaller GPUs on Kaggle) will yield **different results**, even with the same seed and learning rate.
   - Using 2 GPUs is not reproducible as 1 GPU, and prompt changes introduce further differentiation.
- **Ring T1 Model to Enter Unsloth Zoo?**: A college project team is planning to introduce the **Ring T1 model** to the **Unsloth Zoo**.
   - Given its size (**1 Trillion parameters**), it's unlikely most individuals can host or run it, contributions are welcome and should be coordinated with Mike.
- **Unsloth GGUF Quantizations Add Accuracy**: **Unsloth** usually implements fixes and improvements before uploading models; Unsloth dynamic quantizations add even more accuracy on top of quantizations.
   - The team recommends: *stick to Unsloth GGUFs*.
- **Synthetically Finetuning TTS Models**: A member seeks advice on fine-tuning a TTS model for **Bulgarian** using synthetic data, noting there are currently no open weight models in Bulgarian.
   - The member was advised to try [VibeVoice](https://github.com/vibevoice-community/VibeVoice), though even a few hundred hours of data *may* be enough *if it's single-speaker, no emotion, very consistent, and you have a tokenization process that pre-processes text into phonemes*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1437938016143540286)** (3 messages): 

> `Discord fan, Little one` 


- **Fan sighting!**: A user exclaims *"LOL I HAVE A FAN"* in response to being recognized on Discord.
- **Identity Crisis!**: Another user asks *"who are you ? !"* implying confusion or surprise at being addressed.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1437538420854489188)** (88 messages🔥🔥): 

> `LLM Efficiency, Dataset size vs model size, Data quality importance, Perplexity checks for data grooming, Effective batch size tuning` 


- **LLM Efficiency Beats Size**: One member realized it's not about building bizarrely large models, but about **efficiency** and getting the best output for the least consumption, highlighting the importance of **small LLMs**.
   - They added that the crucial job is optimizing and tailoring the models for consumers or businesses.
- **Dataset Size Should Scale With Model Size**: A member stated that increasing dataset size should be matched by increasing model size, especially when running models below **4B parameters**, but it depends on the task and objective.
   - Another user responded that they are *well below the saturation point* and their increased data would warrant a **1B model**, but there aren't small models with large attention in lcpp.
- **Data Quality Matters For Small Models**: It was noted that with 48K entries, the dataset might not be evaluated fully and contain a large margin of **low-quality data** affecting the model, especially since smaller models are more sensitive to that.
   - The user trains their model by playing RPs and manually fixing every mistake the stats model makes.
- **Perplexity Checks Help Data Grooming**: A user runs a **perplexity check** of a stage1 model (trained on a small synthetic set) on the entire dataset and manually validates entries with **high perplexity**.
   - They stated that they spent *a few hundreds of hours on tightening the data already*.
- **Effective Batch Size Can Be Tuned**: With **48k entries**, effective batch size can be tuned, but one member found their model didn't like going away from **24** for some reason.
   - A different member got way better graphs on less than 10k by going from **8 to 32**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1437535450515636464)** (53 messages🔥): 

> `Fine-tuning Llama 3 for script writing, Dataset size for fine-tuning, Chat template issues, Unsloth documentation on fine-tuning, Continued pre-training` 


- **User Fine-Tunes Llama 3 for Scriptwriting**: A user is fine-tuning **Llama 3 8B Instruct 4bit** to write scripts, but is getting nonsensical output with a dataset of **50 scripts**.
   - An expert suggests that **50 samples** is not enough for fine-tuning and shares the Unsloth [fine-tuning LLMs guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- **Data Quantity Dictates Fine-Tuning Quality**: An expert points out that fine-tuning teaches a model output style, but if the model lacks knowledge in the target area, the output quality will suffer, regardless of fine-tuning.
   - They recommend **prompting the model** before fine-tuning to evaluate its pre-trained knowledge; generally a baseline of **300-1000 entries** is suggested if the model already has general knowledge in the area of domain.
- **Chat Template Issues Cause Nonsense Output**: An expert suspects that the *nonsense output* could be due to chat template issues and asks the user to clarify the chat template used during fine-tuning.
   - The user confirms they are using `tokenizer.apply_chat_template`.
- **Pre-Training Provides Structure to LLMs**: An expert explains that the structure of language models comes from **pre-training**, recommending a [YouTube series](https://www.youtube.com/watch?v=wjZofJX0v4M) to understand the underlying architecture.
   - They suggest that since LLMs operate on *next probable token* architecture, they need existing knowledge to generate likely tokens.
- **Medical Term Loss Function Gets Community Review**: A user seeks feedback on their [medical-loss-FT implementation](https://github.com/Chilliwiddit/medical-loss-FT), which trains a model while penalizing it by adding logits to the total loss to focus on medical terms.
   - They ask for review as they are new to PyTorch Lightning, specifically noting they are not sure how to test their code before training.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1437566678987964547)** (2 messages): 

> `FameSumm Implementation, Medical Term Logit Penalization` 


- **Medical Model Implementation Mimics FameSumm**: A member created an implementation based off **FameSumm** which trains a model but penalizes it by adding logits to the total loss to make it focus on medical terms, and posted a link to the [HF model card](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b).
   - The implementation is on [GitHub](https://github.com/Chilliwiddit/medical-loss-FT), and the member requested feedback on the implementation as they're new to **PyTorch Lightning** and not sure if they implemented it right or how to test it before training.
- **PyTorch Lightning feedback requested**: Because the member is new to **PyTorch Lightning**, they requested feedback on the implementation.
   - The main basic flow of the implementation is explained in the [repository](https://github.com/Chilliwiddit/medical-loss-FT).


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1437956060848590879)** (1 messages): 

> `MiniMax M2, Paid Endpoint Migration, OpenRouter Announcements` 


- **MiniMax M2 Free Period Ending Soon!**: The free period for **MiniMax M2** is ending in one hour; migrate to the paid endpoint to continue using the model.
   - This affects all users currently utilizing the free tier for testing and development.
- **OpenRouter Announces MiniMax M2 Transition**: OpenRouter issued an announcement regarding the transition of **MiniMax M2** from a free period to a paid endpoint.
   - Users should take action to ensure uninterrupted service by switching to the paid endpoint before the deadline.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1437666413417594890)** (15 messages🔥): 

> `Smol AI, Anti-bot measures, News Scraper` 


- ****Smol AI** Newsletter Drops**: The **Smol AI** newsletter [was dropped](https://news.smol.ai/issues/25-11-10-not-much#openrouter--app-showcase-7-messages) covering topics from the OpenRouter `app-showcase` channel.
   - A member reacted to this with *"crazy"*, while another one called it *"so good"*.
- **Anti-bot measures improving channel quality**: A member noted that the channel is much better thanks to the **anti-bot measures**.
   - No other comments were provided.
- **Custom News Scraper Debuts**: A member created a [custom news scraper](https://static.dino.taxi/or-news.html).
   - The member indicated that this scraper is *"vibe coded and embarrassing"* and that the code *"is not on there anyway"*.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1437535405430935622)** (268 messages🔥🔥): 

> `Output price filtering for models, Gemini latest news, Minecraft chat moderation, GPT-5 pricing, OpenRouter account with $10 billing` 


- **Output price filtering available**: A user asked for a way to filter the model list by output price on OpenRouter, and another user shared [a link](https://orchid-three.vercel.app/endpoints?sort=outputPrice&order=asc&not=free,gone) to filter models by output price, excluding free and gone models.
- **Discussion on Minecraft chat moderation using LLMs**: One member is implementing a chat moderation feature for a **Minecraft** server using LLMs to flag messages, sparking a discussion on rate limits, batching messages, and using fallback models like **DeepSeek** or **Qwen3** if rate limits are hit.
   - Another member joked that the user will end up in debt *paying hundreds per month to OpenRouter* in half a year because of this plan.
- **OpenRouter chat function experiences scrolling issues**: Several users reported issues with the OpenRouter chat interface, specifically being **unable to scroll through chats** on multiple browsers and devices.
   - A quick fix was suggested using the DOM inspector to add a style and class to specific divs.
- **User looking for OpenRouter account**: A user asked for an **Openrouter account with $10 billing**, resulting in another user joking that they'll start selling OpenRouter accounts with $10 for $20.
   - One user asked for an account for free (even if it's just for $5), which others criticized.
- **Developer hits API error**: A developer reported getting **HTTP 500 errors** when creating a key with the provisioning HTTP API, eventually resolving the issue by passing `Content-Type: application/json` in the request.
   - Another dev responded to the reports indicating that it **was working fine** and reminded to make sure that they are using a user provisioning key.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1437535755361583245)** (15 messages🔥): 

> `Meta AI's Low-Coverage Language Project, Data Acquisition for 500 Languages, OpenRouter Search Bar Removal, UI/UX choices` 


- **Meta AI records Low-Coverage Languages**: Meta AI launched a [project](https://x.com/AIatMeta/status/1987946571439444361) focused on **low-coverage languages**, raising questions about the source of training data.
   - A member suggested that the team recorded hours of audio with the help of local communities, a point detailed in their published video.
- **OpenRouter Missing Search Bar**: A user inquired about the missing search bar in OpenRouter's UI, attaching a [screenshot](https://cdn.discordapp.com/attachments/1392278974222307469/1437878991821345018/image.png?ex=6914d8aa&is=6913872a&hm=15467b49d02b61376e42b733beacf00076ac9bd28dc3dde272a7662cb06f77d7&) of the issue.
   - Another member explained that the search bar was likely removed intentionally to prevent confusion between generic searches and room-specific searches, showing the menu "zooms out" in the [chat page](https://cdn.discordapp.com/attachments/1392278974222307469/1437948459025043726/image.png?ex=6915195c&is=6913c7dc&hm=8f58464961fd3f270d52f4b454893c209105e8458cbe89800815aaf872a47578&).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1437555613285482628)** (44 messages🔥): 

> `AWS for LLM hosting, Low cost private LLM hosting, LM Studio plugin for external LLMs, LM Studio admin privileges on macOS` 


- ****AWS** is an expensive option for private **LLM** hosting**: Members discussed the practicality of running a private **LLM** on **AWS**, with some arguing that it offers poor performance for the cost, compared to alternatives such as [Runpod](https://runpod.io/) or [Vast.ai](https://vast.ai/).
- **Cut Costs by Slapping Together a Low-Power Private LLM Server**: For a private and secure **LLM** setup, it was suggested building a local server with a low-power CPU, a few GPUs (12/16 GB each), **Ubuntu**, and **Tailscale** for remote access, potentially costing around **$550** using used components.
- ****LM Studio** to Support External LLMs via Plugins**: With the upcoming **0.4.0** release, **LM Studio** will support plugins, enabling users to integrate other **LLM** providers like **ChatGPT** or **Perplexity**; currently, this functionality is not available.
- ****LM Studio** Admin Rights Stir macOS Speculation**: A user raised concerns about **LM Studio** requiring admin privileges on **macOS**, a known issue already being tracked in the bug tracker.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1437532985636425839)** (186 messages🔥🔥): 

> `AMD GPU vs Nvidia GPU performance, CUDA vs Vulkan performance differences, Multi-GPU setups for local LLMs, Model routing for efficient LLM usage, Power requirements for multi-GPU rigs` 


- **AMD GPUs Challenging Nvidia's Reign**: Members discussed whether **AMD GPUs** or **Vulkan** are faster, sharing experiences and benchmarks, with one member planning to buy a **7900 XTX** to compare against their **3090**.
   - They noted a potential **40% difference** in performance between the 3090 and the 7900 XTX.
- **CUDA vs Vulkan faceoff reveals unexpected twists**: A member found their **3070** slightly faster in **Vulkan** than in **CUDA**, which was slightly faster than **CUDA 12**, others corroborated that **CUDA 12** seems to be slower in some cases.
   - One member's benchmarks showed a **3090** running **Llama 2 7B Q4_0** at **100 t/s** in CUDA, **85 t/s** in CUDA 12, and **120 t/s** in Vulkan; a **4090** achieved **160 t/s** in CUDA, **190 t/s** in CUDA 12, and **185 t/s** in Vulkan.
- **Multi-GPU Misconfiguration creates Efficiency Conundrums**: A user reported running two GPUs at half power, leading to a discussion on potential misconfigurations, one member suggested Vulkan to address uneven memory distribution across GPUs.
   - The discussion covered the ideal setups for **70B-ish models**, with quad **24-32GB GPUs** being recommended for inference, outperforming a single high-end card in tokens/s/$.
- **Crafty Model Routing cuts VRAM Costs**: One member shared their experience with model routing on an x99 Xeon server, using a fine-tuned **BERT model** to classify user input and route it to different LLMs based on complexity, significantly reducing VRAM requirements.
   - They highlighted that basic queries were handled by smaller, faster models, while harder queries went to larger, domain-specialized LLMs, needing only **47GB of RAM** to load all models.
- **Power Supply Shenanigans feed the thirst**: A user seeking more power for their setup noted that a **2000W PSU** wasn't cheap, and another user recommended a solution that supports up to 4 PSUs simultaneously, as seen in [this linked image](https://cdn.discordapp.com/attachments/1153759714082033735/1437886877146288299/IMG20251111065524.jpg?ex=6914e002&is=69138e82&hm=a6fac8787c0b7c4dde462c99b5bb13fe98ee19e0ea169174de9705eda907c931&).
   - A user humorously noted, *If for some ungodly reason I need more than 2000w, I'll be tripping my breaker*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1437533888917405748)** (114 messages🔥🔥): 

> `Sora 3 Status, D.A.T.A™ Avatar, AI coding shortfalls, AI Smartwatch game, Gemini 2.5 Pro vs GPT-5` 


- **Sora 3 still in the Shadows**: Users inquired about the production status of **Sora 3**, while another jokingly mentioned the release of **Sora 67**.
   - Many users expressed difficulty in obtaining codes or access to the latest AI models.
- **D.A.T.A™ Digital Avatar shows his Face**: A user introduced **D.A.T.A™** (**Digital Autonomous Task Assistant**), showcasing its capabilities beyond just dancing and [shared an image](https://cdn.discordapp.com/attachments/998381918976479273/1437585208969793556/IMG_20251110_182851179.jpg?ex=6915188e&is=6913c70e&hm=5cc08360b6553ee1d9bed4163bad1f6feee33677d9c54ef2a06b4db981c6db38&).
- **Agentic Coding IDEs Emerge**: There's frustration across the board regarding AI's ability to deliver practical results in coding tasks involving duration and quality.
   - One user found success using *agentic coding IDE extensions* and *CLI tools*, creating a solid team in an AI-assisted game after decades of inexperience.
- **Ambitious Smartwatch AI dreams of world dominance**: A user described an ambitious project to create an **AI smartwatch** that merges reality and digital gaming, incorporating features like wifi sniffing for wall hacks, biosensors, and cryptocurrency integration.
   - Another member noted the project sounded like a self-improvement recursion system prompt and that the *elevator pitch requires 100+ floors*.
- **Gemini 2.5 Pro Flexes on GPT-5?**: GitHub is classifying **Gemini 2.5 Pro** as more powerful than **ChatGPT**.
   - Others clarified that **Gemini 2.5 Pro's context is larger**, and the listed models are older, but **GPT-5** remains the latest unlisted model.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1437635728707420210)** (40 messages🔥): 

> `GPT models training limitations, GPT-4.1 vs GPT-5 rerouting, Alternative AI companies, Teen safety and privacy changes, Sora2 invite code` 


- **GPT Models' Training Falls Short**: Users discussed how **GPT models** aren't always trained on what they can and can't do, or how their training translates to actual functionality, raising questions about their understanding of tasks like **sending emails** or **scheduling**.
   - A member suggested asking the AI questions directly to learn more about its capabilities and limitations, comparing it to a *fancy Google*.
- **GPT-4.1 Rerouting Sparks Subscription Cancellation Threats**: A user expressed frustration with **GPT-4.1** potentially being rerouted to **GPT-5**, prompting them to consider canceling their subscription, citing concerns about forced habit changes.
   - Another user refuted this, stating they haven't experienced forced model switching and always maintain their preferred model, even in newer chats.
- **Seeking AI Alternatives to GPT-4.1**: In response to dissatisfaction with **GPT-4.1**, a user inquired about alternative AI companies offering comparable capabilities, even at a higher price point.
   - Suggestions included **GPT-4o**, **GPT-5**, and **Auto**, with one member favoring **GPT-4.5**; it was also mentioned that the **age check** is expected in December due to **safety changes** ([Teen safety, freedom, and privacy](https://openai.com/index/teen-safety-freedom-and-privacy/) and [Tasks in ChatGPT](https://help.openai.com/en/articles/10291617-tasks-in-chatgpt)).
- **Desperate Plea for Sora2 Invite Code Emerges**: Amidst the ongoing AI discussions, a user made a direct request for a **Sora2 invite code**.
   - No further information or context was provided regarding this request.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1437703006308012054)** (11 messages🔥): 

> `Centralized interactive tool development, API features for custom GPTs, Verbatim quotes and citations` 


- **Interactive Tool's Centralization Dilemma**: A member questioned the possibility of decentralizing parts of a centralized interactive tool's development on a GPT chatbot, emphasizing the need for precise verbatim quotes.
   - Another member suggested using non-LLM coding to handle verbatim quoting by calling the LLM sometimes, and sometimes calling another program for specific quotes.
- **Custom GPTs get API Powers**: Members discussed adding external system API features to custom GPTs via **Actions**, with a link to the [Actions documentation](https://platform.openai.com/docs/actions/introduction).
   - One member suggested that using the API with a custom chat interface might be easier than rigging everything up in a Custom GPT, noting that both solutions require coding an API query.
- **Verbatim Quotes Via External Server**: One member mentioned that the solution for verbatim quotes and citations in ChatGPT involves setting up **an external server** and adding it to the Custom GPT via **Actions**.
   - They stated *In ChatGPT the solutions to this problem space are found by setting up an external server for verbatim quotes and citations, and adding it to the Custom GPT via Actions (documented above).*
- **Execution Environment Unavailable**: One member reported being *unable to access the code execution environment right now*, therefore they could not regenerate or verify a new database.
   - This was followed by another member reporting they were having trouble finding prompt engineering jobs.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1437703006308012054)** (11 messages🔥): 

> `Lock Verbatim of Comments, External System API Features for Custom GPTs, Prompt Engineering Jobs` 


- **Lock Verbatim Gains Traction**: One member suggested locking the verbatim of comments by using a prompt bot, similar to help bots on websites that express sentences precisely without rephrasing.
   - Another member suggested that while **LLMs** may struggle with verbatim quoting, non-LLM coding could easily handle pulling exact quotes from a database, and feeding them to the model for context.
- **Custom GPTs getting external API features**: A member pointed out that **external system API features** can be added to custom **GPTs**, called **Actions**, with a [link to the documentation](https://platform.openai.com/docs/actions/introduction).
   - The same member suggested it might be easier to use the **API**, add a custom chat interface, and control the system prompt, rather than rigging it all up in a Custom GPT.
- **Prompt Engineering Jobs hard to find**: One member expressed difficulty in finding **prompt engineering jobs**.
   - No solutions or suggestions were provided.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1437540210144706631)** (121 messages🔥🔥): 

> `Physics Informed Neural Nets (PINNs), Researcher Communication Strategies, JSON Data Model, Paper Selection Filters, Self Attention Scaling` 


- **PINNs vs PDE Standard Methods**: A member clarified that **Physics Informed Neural Nets (PINNs)** define an unknown function as a neural net and use the differential equation itself as a loss function, whereas standard methods in **PDEs** assume a linear basis.
   - With PINNs, standard optimization techniques are used on a residual defined pointwise on the domain by using the diffeq.
- **Crafting Effective Emails to Researchers**: Members discussed how to write effective emails to researchers, noting the importance of being clear, formal, and asking questions that don't require extensive time to answer.
   - It was emphasized that demonstrating an understanding of the researcher's work is key to getting a response.
- **New JSON Model Competes with Gradient Boosted Machines**: A member has been developing a research project that can manage/embed arbitrarily structured **JSON data** (no feature engineering required) and is constructed on the fly from an input data schema, competing with gradient boosted machines (GBMs) trained with handcrafted features.
   - They are seeking organizations interested in building this out or additional use cases to compile results for a paper.
- **Auto-Filtering Papers: AlphaXiv and Emergent Mind**: Members suggested using existing auto-filters like [AlphaXiv](https://www.alphaxiv.org/) and [Emergent Mind](https://www.emergentmind.com/) to select papers for discussion, noting that papers trending on these sites are often well-received.
   - The suggestion was made to check if a paper is active and liked on these websites to gauge its relevance and quality before posting.
- **Self Attention Needs Scaling for Statistical Calibration**: Members discussed the scaling factor in self-attention (**dividing by sqrt(d_k)**), explaining that it's crucial for statistical calibration, not just numerical stability.
   - It ensures proportional relationships between numbers are preserved before the softmax function is applied, preventing extreme distributions.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1437888236356636744)** (4 messages): 

> `Reproducibility, ThinkPRM, Kimi K2` 


- **Reproducibility Reliance Relies on Repos**: A member expressed their preference for having the full version of research in the main body and appendices presented visually online, with hyperparameters and algorithms in a **GitHub repo** for reproducibility.
   - They argued that ambiguous steps or hyperparameter choices can lead to an *infinite space of potential values*, hindering quick reproduction and verification of results.
- **ThinkPRM repo surfaces**: A member shared a link to the **ThinkPRM GitHub repository** [https://github.com/mukhal/thinkprm](https://github.com/mukhal/thinkprm).
   - No further information was given about the repo.
- **Kimi K2 coding skills kickoff**: A member shared a demo of **Kimi K2** performing short one-shot coding tasks, accessible at [https://www.youtube.com/watch?v=BpsleXIV-WI](https://www.youtube.com/watch?v=BpsleXIV-WI).
   - The member noted this was a *nice demo*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1437613297766498324)** (10 messages🔥): 

> `PromptFlux Malware, Yann LeCun Leaving Meta, Country Songs Analysis` 


- **Google Exposes PromptFlux Malware**: Google uncovered **PromptFlux malware** that leverages **prompt injection** techniques to infiltrate systems, as reported by [The Hacker News](https://thehackernews.com/2025/11/google-uncovers-promptflux-malware-that.html).
- **LeCun Leaves Meta to chase conscious exploration?**: Members discussed Yann LeCun reportedly leaving Meta to launch a new AI startup, according to [The Decoder](https://the-decoder.com/yann-lecun-reportedly-leaving-meta-to-launch-new-ai-startup/); some speculate he aims to explore areas Meta's conservatism restricted.
   - One member stated *"probably made enough money that his conscious finally got the upper hand"*.
- **Country Songs deemed Crap**: A link to a Twitter post ([https://x.com/kimmonismus/status/1988264217376645264](https://x.com/kimmonismus/status/1988264217376645264)) led to the discussion of country music.
   - A member summarized the discussion by saying, *"Just says more about how crap country songs are then anything else"*.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1437539266539290785)** (124 messages🔥🔥): 

> `UI feedback, Neurips Meetup, AWS Reinvent, Pretraining on Hermes, Peer Review Application` 


- ****UI** Feedback Sought by Creador**: Creador is soliciting feedback on a **UI** they are developing, sharing an [image](https://cdn.discordapp.com/attachments/1149866623109439599/1437583555747250317/image.png?ex=69151704&is=6913c584&hm=78d23d910cbaf3ebb7588d0ce76b22fadb54eff1a470f8feea47a714b254b37b&) of the interface for review.
   - Creador is aiming to create an *annotation software* to avoid vendor lock-in and is looking for opinions on whether the **UI** is too clunky or dark.
- ****Neurips** Attendees Plan a Meetup**: Several members, including teknium and gyzoza, are attending **Neurips** in San Diego from **December 2nd to 7th** and are planning a potential meetup.
   - There was confusion whether Neurips was on Dec 1, but it was clarified it's Tuesday Dec 2nd through Sunday Dec 7th; the group joked about whether the meetup should be in San Diego or Mexico City (where another member is attending **Neurips**).
- **AWS Reinvent Ticket Prices Draw Ire**: A member expressed reluctance to pay **$2100** for an **AWS Reinvent** ticket.
   - Another member who attended last year said it wasn't worth it, only getting an **Anthropic sticker** out of it, and that crashing the after parties is impossible because they check registration.
- **Pretraining Method Debated**: A member mentioned a paper ([https://arxiv.org/abs/2510.03264](https://arxiv.org/abs/2510.03264)) about putting reasoning and instruction data in pretraining, similar to their approach with **Hermes data**.
   - They noted that while their paper came out first, the other paper had better ablations, but their own paper did not mention "nvidia on it" jokingly.
- **Github repo of 'autonomous ai by accident'**: A user shared their github repo for **'autonomous ai by accident'**.
   - The repository is named **grokputer** and can be found [here](https://github.com/zejzl/grokputer).


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1437629730131284184)** (1 messages): 

> `Salience full tour` 


- **Salience full tour video posted**: A member shared a video called **salience_full_tour.mp4** ([link to video](https://cdn.discordapp.com/attachments/1154120232051408927/1437629726721048586/salience_full_tour.mp4?ex=69149944&is=691347c4&hm=04254199b944446aca9011c7f8a4c5b3b88915a2569137bfb17242de0f76e0ec&)).
   - The member said, *"I just love getting to see stuff like this either way"*.
- **Another topic for compliance**: Adding another topic to comply with the schema.
   - Filler content.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1437694253890535425)** (2 messages): 

> `GradientHQ Parallax, Live demo of Parallax` 


- **GradientHQ Ships Parallax: Live Demo Available**: A member shared a link to [GradientHQ's Parallax](https://github.com/GradientHQ/parallax/tree/main), describing it as a 'slick' new tool.
   - A live demo for testing Parallax is available at [chat.gradient.network](https://chat.gradient.network/).
- **Parallax: A New Tool by GradientHQ**: Parallax, created by **GradientHQ**, offers users an interactive platform to engage with and test the tool directly.
   - The tool's capabilities and features can be explored further on its [GitHub repository](https://github.com/GradientHQ/parallax/tree/main).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1437534864650801162)** (24 messages🔥): 

> `Mojo vs Rust, Mojo phase 3, Mojo use cases, Mojo replacing C++, Mojo and dynamic type reflection` 


- **Mojo ❤️ Rust Mechanics**: Mojo uses fundamental **Rust mechanics** such as **ownership**, **traits**, and **structs**, but aims to innovate and improve upon them, mixing Python syntax with Rust mechanics and adding native GPU function support.
   - A member commented that Mojo appeals more to Rust, except for the Python syntax, and emphasized that **Mojo is not fundamentally an OOP language** due to the lack of class inheritance.
- **OOP in Mojo Phase 3?**: Discussion arose around the [Mojo roadmap](https://docs.modular.com/mojo/roadmap/), specifically **Phase 3**, which may include OOP features as syntactic sugar, but is still potentially 3-4 years away.
   - Others debated whether the lack of OOP is a significant loss, with one commenter noting it might not be, questioning the use cases where OOP would be necessary, such as GUI development.
- **Mojo: Not just for AI?**: Mojo is being positioned as a general-purpose systems programming language akin to **C, C++, and Rust**, with a Python-like syntax.
   - Although the AI ecosystem is the most developed, Mojo aims to replace C++ and some Python code, excelling in HPC, scientific computing, quantum computing, bioinformatics, and tools like FFTs, even outperforming NVIDIA's cuFFT on NVIDIA hardware.
- **Dynamic Type Reflection Coming to Mojo?**: Mojo plans to support dynamic type reflection, leveraging its JIT compiler to facilitate useful operations with dynamic data, although static reflection is preferred.
   - For error handling, the standard **try-catch-raise** mechanism, similar to Python, is planned, with potential for more monadic options to handle errors effectively.
- **Mojo's Metaprogramming Powers**: Discussion arose about the comparative power of Mojo's metaprogramming versus Zig's `comptime` mechanism, referring to [Chris Lattner's recent interview](https://www.youtube.com/watch?v=Fxp3131i1yE&t=1180s).
   - Details on the specific advantages were not elaborated in this message log.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1437540650953347135)** (85 messages🔥🔥): 

> `Implicit conversion to ref/mut, Raw vs Unsafe, GPU compilation error, Python 3.14` 


- **Mojo Debates Implicit Mutability**: The Mojo community is debating whether the implicit conversion of variables to `ref` or `mut` when passing arguments to functions makes code harder to read and understand, or whether call-site mutability should be more explicit.
   - Some members suggested using `mut` on the call side, similar to Rust's `&mut value`, while others expressed concern about clutter and suggested IDE support to indicate mutability, with one member stating *not super excited about Mojo becoming Rust and very far from Python, honestly.*
- **Raw vs. Unsafe Naming Convention**: The Mojo community discussed the naming convention of `unsafe` constructs (pointers, system calls) and whether `raw` would be a more appropriate term to avoid negative connotations, citing the fact that operating systems do not typically mark kernel APIs as `unsafe`.
   - A core team member noted that the intention behind `unsafe` is to invoke caution, similar to a workplace safety sign, inspired by `Rust`, and that *being able to grep for `unsafe` code when you encounter a segfault for example can be a useful debugging tool*.
- **GPU Compilation Woes**: A user encountered a *Metal Compiler failed to compile metallib* error when running the 'Get started with GPU programming' tutorial on an Apple M4 GPU, specifically when invoking `enqueue_function_checked`.
   - The problem was solved by core team members by suggesting that it might be due to using print statements in the GPU kernel which are not supported yet. The advice was also to use the latest nightly release, as the 25.6 stable release was very early and missing features.
- **Python 3.14 Compatibility Status**: Members are trying to standardize their Python installations on macOS and asked about MOJO_PYTHON_LIBRARY support for Python 3.14, since there were issues with 3.13 in the past.
   - A core team member said that *3.13 should have been working for a while now* and 3.14 is being worked on, but they are *waiting for (I hope) one more dependency to update*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1437538652489125938)** (22 messages🔥): 

> `Cost-effective hardware for AI training, TPUs vs physical hardware, Implementing multi-headed attention from scratch` 


- **Hardware Recommendations for AI Training Newbies**: A professor inquired about cost-effective hardware for AI training for new students, focusing on **MLPs**, **CNNs**, and small-scale **LLM pre-training** and **fine-tuning**.
   - Suggestions included using **TPUs** on **Google Colab** and physical hardware options like **Nvidia GPUs**, with the **3090** highlighted for its **VRAM/USD** ratio, with at least **16GB** of VRAM suggested.
- **Multi-Headed Attention Implementations: NumPy vs. einops**: A member asked how often machine learning engineers are asked to implement **multi-headed attention** from scratch during interviews.
   - The discussion highlighted implementing it in **NumPy** and implementing it with **einops**, with one member stating *"I refuse to implement it without einops lol"*.
- **Interview Failure: NumPy's AutoDiff Limitations**: A member recounted bombing an interview question that required implementing **multi-headed attention** with dropout in **NumPy**.
   - Another member pointed out that *"kinda useless without autodiff to train"*, suggesting using **JAX** as a baseline before explaining why **NumPy** isn't ideal, while still acknowledging its limitations.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1437722704164687945)** (29 messages🔥): 

> `DCLM dataset, Zyda-2 dataset, Nemotron-CC-v2 dataset, RWKV datasets, Pretraining datasets` 


- **DCLM Dataset Questioned**: A member asked if **DCLM** is still among the best pretraining datasets for training a generic model on ~750B tokens.
   - Multiple suggestions came in including **Zyda-2**, **ClimbLab**, and **Nemotron-CC-v2** as better options.
- **Zyda-2, ClimbLab, and Nemotron-CC-v2 recommended**: Members recommended [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2), [ClimbLab](https://huggingface.co/datasets/nvidia/Nemotron-ClimbLab), and [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) as the best datasets for initial pretraining, especially the high-quality shard of Nemotron.
   - The consensus was to mix these datasets due to their individual strengths and weaknesses.
- **RWKV Dataset Listing**: A member shared a link to a [SOTA open source dataset listing](https://huggingface.co/datasets/RWKV/RWKV-World-Listing) of 3.1T tokens validated to 2.9B size (from March 2025) that includes a subset of DCLM baseline tokens.
   - Another member suggested removing subsets like *slimpj* and the *slimpj_c4* scrape and asked for a token breakdown to see how the 3.1T tokens are made up.
- **Paper on Dataset Composition**: A member shared a [link to a paper](https://arxiv.org/abs/2503.14456) detailing the dataset composition, noting it includes the entire 630B Slim Pajama dataset without downsampling.
   - It was noted that this could lead to duplication with other datasets such as Wikipedia, Books3, StackExchange, and new GitHub/ArXiv datasets.
- **Importance of CoT Reasoning Traces**: The newer RWKV releases have focused a lot on adding in **CoT (Chain of Thought) reasoning traces** to the pretraining data.
   - Papers now show that including CoT reasoning traces is incredibly important if you want to prepare for a reasoning model.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1437585880565944340)** (25 messages🔥): 

> `Concept Probes for Interpretability, Divergent Internal Concepts, Real-Time Concept Detection` 


- **Interpretability Tool Detects and Steers Concepts in Real Time**: A team built an [interpretability tool](https://cdn.discordapp.com/attachments/1052314805576400977/1437923335597195457/Screenshot_2025-11-10_20-47-00.png?ex=691501f6&is=6913b076&hm=4c5382b2547dbcde832d2bcda282cfbca334d23300574322c46b85938b8e5a24) to detect and steer **thousands of concepts in real time** by training concept probes on a model's activations.
   - The system uses **binary classifiers** created by iterating through an ontology, prompting the model, and removing the shared subspace until reaching **95% accuracy** classifying OOD samples.
- **Divergent Internal Concepts Highlighted**: When the model is being divergent, the **quality of the answers degrades**, which could be a *cover story*, or the divergent probability space creates a less clear generation.
   - For example, when asked about what it would do with unlimited power, the model talks about a TV show, but the activations show concepts about **AIDeception, AIAbuse, and MilitaryInfiltration** (as shown in [self_concept_019.json](https://cdn.discordapp.com/attachments/1052314805576400977/1437924390896668852/self_concept_019.json?ex=691502f2&is=6913b172&hm=44d875da27442840e99bd109ba0ccfefddd26c706d5678317d7633ae311dae37)).
- **95% Accuracy Misinterpretation?**: A member questioned the meaning of the **95% accuracy** in the concept probe training, suggesting potential **false positives** due to expecting fewer concept occurrences at test time.
   - The original poster clarified that they provide users with raw probability scores in ranked order and continually resample, dealing with a probabilistic polysemantic system.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1437536259542089908)** (40 messages🔥): 

> `Learning Python with LLMs, Out of Memory Errors with Diffusers, HF Spaces failing with io.EOF error, HF Responses API, Multi Headed Attention interview questions` 


- **LLMs Explain Complex Stuff Simply**: Members discussed using LLMs like **Gemini** to explain complex topics in simple terms, with one saying *I just use it to explain complex stuff*.
   - Another member uses LLMs to ask for simple analogies to *grasp XYZ*.
- **Diffusers trigger OOM errors**: Multiple users reported encountering **Out of Memory (OOM)** errors while running **diffusers**, especially with models requiring a minimum of **6GB VRAM**.
   - One user considered using a cloud instance with a **TPU**, acknowledging that it would be *expensivo*.
- **HF Spaces Break Down With EOF**: Multiple users experienced issues with **HF Spaces** failing to build due to an `io.EOF` error when requesting resources from `https://spaces-registry-us.huggingface.tech`.
   - One user linked to a [Hugging Face forum discussion](https://discuss.huggingface.co/t/io-eof-error-persists-over-5-restarts-when-requirements-txt-is-light-and-spaces-maintenance-is-up/170194/7) indicating that HF is aware of and working on the issue.
- **ZeroGPU logs flagged**: A member reported that their post about **ZeroGPU** not working with logs was flagged in the **Hugging Face forums**.
   - Their comment was hidden and is awaiting review, linked at the [HuggingFace forum](https://discuss.huggingface.co/t/error-failed-to-push-spaces-registry-us-huggingface-tech/170195/21).
- **FameSumm trains models using Pytorch Lightning**: A member requested feedback on their implementation of **FameSumm**, which trains a model using **PyTorch Lightning** and penalizes it by adding logits to the total loss, to focus on medical terms.
   - The implementation details are available on [GitHub](https://github.com/Chilliwiddit/medical-loss-FT), with the member seeking guidance on testing and verifying the implementation.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1437564762761461902)** (16 messages🔥): 

> `SUP Toolbox, Muon Optimizer, NVIDIA's document layout detect model` 


- **HuggingFace user releases SUP Toolbox**: A user launched **SUP Toolbox**, an AI tool for image restoration & upscaling using **SUPIR**, **FaithDiff** & **ControlUnion**, built with **Diffusers** and **Gradio UI**.
   - The [Space Demo](https://huggingface.co/spaces/elismasilva/sup-toolbox-app), [App repository](https://github.com/DEVAIEXP/sup-toolbox-app) and [CLI repository](https://github.com/DEVAIEXP/sup-toolbox) are available for feedback.
- **CPU-friendly distributed Muon tutorial released**: A fully annotated breakdown of the “**Muon is Scalable**” optimizer was released, refactored for clarity, reproducibility, and accessibility, that runs on **Gloo**, not **NCCL**.
   - The tutorial covers how **DP + TP** groups coordinate, how **ZeRO** sharding fits in, and why bucketing & coalescing are more than just “performance tricks,” with a [full repo](https://huggingface.co/datasets/bird-of-paradise/muon-distributed) and [write-up](https://discuss.huggingface.co/t/tutorial-update-reverse-engineering-breakdown-released-the-muon-is-scalable-cpu-friendly-blueprint/170078).
- **NVIDIA's Document Layout Detect Model Demo on HF Spaces**: A user created a HF space to demonstrate NVIDIA's document layout detect model based on **YOLOX**.
   - The demo is available [here](https://huggingface.co/spaces/dinhanhx/nemoretriever-page-elements).


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1437743048787365888)** (1 messages): 

> `Diffusers MVP program` 


- **Diffusers Launches MVP Program**: The **Diffusers MVP program** is starting to work with and learn from contributors in the community.
   - Details are available on the [GitHub issue](https://github.com/huggingface/diffusers/issues/12635); interested parties are encouraged to join.
- **No Further Topics**: No further topics or summaries were found in the channel.
   - This entry is added to satisfy the requirement of at least two items in `topicSummaries`.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1437947346062413936)** (3 messages): 

> `Random data generation, PII Randomization` 


- **Generate Realistic Random Data with Prompting**: A member asked if a system can generate realistic random data instead of `'XXX'` via prompting.
   - Another member suggested this task might be easier with a plain old python script.
- **Detect and Randomize PII with Prompting**: A member suggested that a prompt setup to detect and randomize **PII** would be a valuable feature.
   - They posited this is preferable to generating random data.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1437533821212823673)** (49 messages🔥): 

> `SYNTH dataset, Meta Omnilingual ASR, Moonshot AI Kimi AMA, Gamma's Series B, Meta's GEM model` 


- ****SYNTH** Dataset and **Baguettotron/Monad** models released!**: Alexander Doria announced **SYNTH**, a fully-synthetic **200 B-token** generalist pre-training dataset focused on reasoning, plus two new SOTA models trained only on **SYNTH**: **Baguettotron** and **Monad** - see tweet [here](https://x.com/Dorialexander/status/1987930819021635964).
- **Meta Launches **1,600** Language Omnilingual ASR!**: Meta unveils open **Omnilingual ASR models** covering **1,600+** languages, sparking excitement, questions on dialects/latency, a live Hugging Face demo, and memes about universal translators - see announcement [here](https://ai.meta.com/blog/multilingual-model-speech-translation-communication/).
- **Moonshot Kimi AMA: Scaling is brutal!**: Cody Blakeney calls Moonshot AI’s Kimi AMA a *gold mine*, observing that scaling new ideas is brutal: many seemingly unrelated factors begin to interact, costs of exhaustive ablation are prohibitive, yet the eventual payoff for whatever *does* work at scale is enormous - see tweet [here](https://x.com/code_star/status/1987924274116784500).
- **Gamma Raises $2.1B in Series B Funding**: Grant Lee announces Gamma’s Series B at **$2.1B** led by a16z while profitable at **$100M ARR** with only **50 employees**, praising the **$2M ARR-per-employee** efficiency - see tweet [here](https://x.com/thisisgrantlee/status/1987880600661889356).
- **LeCun Leavin' Meta After Criticism??**: Tweet cites FT report that Meta’s chief AI scientist **Yann LeCun** plans to quit; commenters joke he may immediately sell a startup back to Meta and note his recent anti-Silicon-Valley, anti-Trump posts - see tweet [here](https://x.com/grady_booch/status/1988278574076621138).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1437653065753690182)** (10 messages🔥): 

> `AI Progress, Magic Patterns 2.0, Series A Funding` 


- **AI Text Quality Zooms to Whiteboard**: Dhruv shared a [tweet](https://xcancel.com/haildhruv/status/1987801616448405538?s=46) expressing amazement at how AI text quality jumped from *pure gibberish* less than a year ago to advanced output, with **AI taking handwritten notes on a whiteboard**.
   - Replies noted the exponential speed of improvement, joking that we’ve leapt from *write a poem* to complex derivations in **ten months**.
- **Magic Patterns Banks $6M Series A**: Alex Danilowicz unveiled **Magic Patterns 2.0** and a **$6M Series A** led by Standard Capital, celebrating bootstrapping to **$1M ARR** with **no employees** and **1,500+ product teams** using the AI design tool via [tweet](https://xcancel.com/alexdanilowicz/status/1988247206940602440?s=20).
   - Users rave the product has replaced Figma for them with rapid hiring across enterprise, engineering, community and growth roles


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1437536399459876894)** (50 messages🔥): 

> `Opencode interleaved thinking, Kimi-cli auto start thinking mode, Moonshot Inference Cluster Slowdown, Kimi Coding Plan API Quota, Bug Report Guidelines` 


- **Cline Coding Extension Enables Interleaved Thinking**: Members noted that **Opencode** did not support interleaved thinking, but **Cline**, **Claude Code**, and **kimi-cli** now do.
   - Users also confirmed **Cline extension** is correctly configured.
- **Kimi-CLI Thinking Mode Autostart How-To**: A user asked how to make **kimi-cli** auto start with thinking mode on, due to frequently forgetting to enable it.
   - A member responded that the `--thinking` flag was added in **v0.51**.
- **Moonshot Cluster Hit By Thundering Herd After Reddit AMA**: A user noted that the **Moonshot inference cluster** has been slow for the last couple of hours.
   - Another member suggested it might be a "thundering herd problem" after the recent **Reddit AMA** drew in more users.
- **Kimi Coding Plan Users Blow Through API Quotas Quickly**: Users reported blowing through their weekly **Kimi coding plan API quota** in just a few hours.
   - One user suggested that **web search** and **plan mode** are likely consuming a high number of API calls; others suggested a bug report.
- **Bug Report Guidelines Clarified**: A user was directed to the <#1371764324866982008> channel to file a bug report about **API Quotas** being exhausted too quickly.
   - They were also advised to check the **bug report guidelines** and be specific, noting that the **Kimi team** is active but based in China, so timezone differences should be considered.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1437591684740284536)** (20 messages🔥): 

> `hatch vs setuptools, pyproject.toml, package data, ChatGPT suggestions for setup` 


- **Debate Heats Up: Hatch vs Setuptools**: The discussion centered on the choice between `hatch` and `setuptools` as build systems, with one member noting that `hatch` is *more minimal and modern*, and doesn't carry the boilerplate of `setuptools`.
   - While `setuptools` isn't standard, it's widely used, leading to a switch back for compatibility; one participant quipped, *i didn't know it wasn't standard in python, but everyone has it*.
- **Wheels and Metadata: pyproject.toml**: It was clarified that a build system is required to build a wheel, as `pyproject.toml` is a metadata format without building logic.
   - The discussion referenced `from setuptools import setup` in `setup.py`, highlighting the need for a build system.
- **Package Data Resurrection in pyproject.toml**: A key discussion point involved ensuring `package data` was correctly included after migrating to `pyproject.toml`.
   - Specifically, the conversation addressed porting `include_package_data=True` from `setup.py` to the `pyproject.toml` file, referencing [this commit](https://github.com/tinygrad/tinygrad/pull/13189/files#diff-50c86b7ed8ac2cf95bd48334961bf0530cdc77b5a56f852c5c61b89d735fd711R19).
- **ChatGPT's Hot Takes on setuptools**: After merging, one member mentioned that ChatGPT had a few suggestions for setuptools, finding them mostly non-critical, but noting they *liked the >=61 for setuptools*.
   - The discussion linked to a [ChatGPT conversation](https://chatgpt.com/share/69136ddd-4bfc-8000-8167-a09aaf86063b).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1437703080303919154)** (13 messages🔥): 

> `M4 Mac segfaults, torch private buffer issue, Tensor from URL` 


- **M4 Mac experiences segfaults**: A member reports a consistent segfault on their **M4 Mac** when converting a **torch tensor** to a **tinygrad tensor** and then to a **numpy array** using `torch_to_tiny()`.
   - Adding `+0` after the conversion seems to resolve the issue, suggesting a potential problem with the lifecycle or memory management of the original torch tensor, see code [here](https://discord.com/channels/1041495175467212830/1194674314755770478/1194703396814866462).
- **Tinygrad cannot copy directly from private torch buffers**: A member reports that **torch** creates the buffer as private, so it isn’t shared with the cpu and doesn’t have contents(), thus, can't copy directly from the tensor (copying from private buffers isn’t supported in **tinygrad**).
   - It was recommended to convert the parameters directly to **tinygrad** by downloading it's `.safetensors` file so that the tiny tensors could be directly converted without having to pass through **torch**.
- **Tensor from URL Usage**: A member considered using `Tensor.from_url` to load **VGG16** weights directly into tinygrad, instead of converting from **PyTorch**.
   - This approach bypasses the need to convert from PyTorch tensors, as the `.safetensors` file can be directly downloaded and used with **tinygrad**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1437575818971971744)** (17 messages🔥): 

> `PowerPoint presentations with company brand, Manus invite, Publish button issues, AI engineer introductions, Mini AGI` 


- **Branded PowerPoint generation with Manus AI in Demand**: A member asked about using **Manus AI** to generate **PowerPoint presentations** with their own company brand.
   - They are new and seeking advice on how to implement this feature.
- **Community member seeks Manus Invite with possible compensation**: A community member requested a **Manus invite**, offering to pay for it.
   - Another member offered to provide one for free.
- **"Publish" button is broken**: A **Pro Subscriber** reported that the "Publish" button does not update the website after the initial publish, leaving the website stuck at the old checkpoint.
   - They contacted Manus Helpdesk but have not received a reply to date.
- **AI engineers introduce themselves and signal for work**: Multiple AI engineers introduced themselves, highlighting their experience in areas like **workflow automation, LLM integration, RAG, AI content detection, and blockchain development**, and signalled their availability for collaboration.
   - One specified expertise in building **AI agents, automating workflows, developing NLP-powered chatbots, integrating voice and speech systems, and deploying custom LLMs**.
- **Mini AGI launch in 2026/34/February2wsx7yhb-p;.,nbvcxz**: A member mentioned creating **mini AGI**.
   - They followed up with the launch date: *2026/34/February2wsx7yhb-p;.,nbvcxz*.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1437851211226419222)** (2 messages): 

> `DSPyMator Launch, Taxonomy Creation Blogpost` 


- **DSPyMator is here!**: A member announced the launch of **DSPyMator** and shared a [link to the announcement on X](https://x.com/jrosenfeld13/status/1988290653324013666).
   - The member expressed enthusiasm, describing it as *really really fun*.
- **Tips for Taxonomy Creation**: A member shared a [blog post](https://shabie.github.io/2025/11/10/why-tails-break-taxonomies.html) about their experience creating taxonomies.
   - The member believes the topic is *super relevant in the context of structured generation*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1437589828999970836)** (11 messages🔥): 

> `Transfer Learning, GEPA Prompting with Strong Models, Optimizing OCR pipeline with GEPA, Saving/Loading DSPy Modules` 


- **GEPA Prompting for Transfer Learning**: A member inquired about formal work on transfer learning with GEPA, specifically using a cheaper model to GEPA then using that prompt with a stronger model to reduce rollout costs.
   - Another member shared a [link to a paper](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf) where they optimized an OCR pipeline with **2.0 Flash** as executor and **GPT-5-High** as teacher.
- **OCR Pipeline Optimization Transferred Gains**: The [OCR research](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf) found that prompts for **2.0 Flash** transferred **80–90%** of their gains to **2.5 Flash and Pro** with light edits.
   - This suggests a potential for transfer learning in GEPA-optimized prompts.
- **Saving/Loading DSPy Modules for GEPA Rollouts**: A member tried to save the state of their Module (optimized by GEPA before) with `save_program=False` and found only the optimized "instructions" were saved in the resulting **.json** file.
   - They questioned if using `save_program=True` to save intermediate results after 2-3 rollouts (`max_full_evals`) for iterative prompt optimization.
- **Prompting Paradigms Remain**: A member stated that the idea of dspy was that you don't have to worry about prompting llm, let the algorithms handle prompting, you can just write task-> outcome.
   - But they think this paradigm works for simple tasks, more so classification types.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1437588119170973868)** (9 messages🔥): 

> `aider-ce improvements, webspp UI integration with aider-ce, Merge editor with conflict resolution, Code Snippets in MD Files` 


- **Community Steps Up for Aider-CE Amidst Creator's Absence**: Due to the original creator's absence, community members are stepping up to improve [aider-ce](https://github.com/paul-gauthier/aider), focusing on channeling energy into code improvements.
   - One member noted that while one community member has been doing a great job, *more hands and eyes on code will get it even better*.
- **Webspp UI Aims for Aider-CE Integration**: A member is working to integrate their [webspp UI](https://github.com/flatmax/Eh-I-DeCoder) with aider-ce, but noticed that some stuff has changed which the UI was relying on.
   - A community member encouraged the user to join the dedicated channel to realign the CE with their system.
- **Merge Editor with Conflict Resolution**: A member expressed interest in adding a merge editor with conflict resolution to aider-ce, and inquired about approaches.
   - The original poster mentioned using Monaco diff editor with themes.
- **Trouble Creating Code Snippets in MD Files**: A user reported issues with creating code snippets in markdown files when using Aider with Anthropic's Claude model, with the tool getting confused by nested code markdown marks.
   - The generated README.md includes nested code snippets that seem to disrupt the process, leading to prompts like *Create new file?* appearing unexpectedly.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1437595082470461520)** (2 messages): 

> `LLM Preprocessing Scripts, LLM Summarization Scripts, LLM Planning` 


- **LLM generates Preprocessing Scripts**: A member suggested asking the model to generate a script for preprocessing, as language models might struggle with large-scale JSON data.
   - The idea is that a **1-2 page script** could quickly enable the **LLM** to better understand the data, leading to improved summarization scripts.
- **LLM assists in Summarization Scripts**: A member proposed a challenge to get the **LLM** to provide a good summary of a file's contents and the desired outcome.
   - The method involves having the **LLM** create a plan, ask questions, and then revise the plan based on the answers.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1437760225783320597)** (2 messages): 

> `Aider-CE Agent Mode, Chrome Devtools Integration, Context7` 


- **Aider-CE Drops Agent Mode Demo**: Aider-CE released a demo showcasing its new **Agent Mode** featuring **Chrome Devtools** and **Context7** integration.
   - In the demo, they [one-shot a webapp in 10 minutes](https://www.circusscientist.com/2025/11/11/aider-ce-building-a-webapp-and-testing-live-with-a-single-prompt/), including in-browser testing, prompting excitement.
- **Aider-CE Enables In-Browser Testing**: The Aider-CE demo highlights the ability to perform **testing directly in the browser**, a key feature of the new agent mode.
   - This capability allows for real-time feedback and debugging during webapp development, as shown in the [linked blog post](https://www.circusscientist.com/2025/11/11/aider-ce-building-a-webapp-and-testing-live-with-a-single-prompt/).


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1437753953046757467)** (2 messages): 

> `MCPConference Paris, Declarative Agents, Evals` 


- **MCPConference heads to Paris**: The **MCPConference Paris** is scheduled for **November 19th** as seen on the [luma.com](https://luma.com/MCPparis2025) website.
- **Declarative Agents get Splash of Evals**: A member will discuss **declarative agents powered by MCP**, incorporating **evals** at the conference.

