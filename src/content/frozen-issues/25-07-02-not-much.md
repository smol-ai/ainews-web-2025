---
id: MjAyNS0w
title: not much happened today
date: '2025-07-02T05:44:39.731046Z'
description: >-
  **Meta** has hired **Scale AI CEO Alexandr Wang** as its new **Chief AI
  Officer**, acquiring a **49% non-voting stake** in **Scale AI** for **$14.3
  billion**, doubling its valuation to **~$28 billion**. This move is part of a
  major talent shuffle involving **Meta**, **OpenAI**, and **Scale AI**.
  Discussions include the impact on **Yann LeCun**'s influence at **Meta** and
  potential responses from **OpenAI**. In model news, **Gemma 3N** faces
  technical issues like vision NaNs and FP16 overflows, with fixes from
  **UnslothAI**. Chinese open-source models like **GLM-4.1V-Thinking** by
  **Zhipu AI** and **DeepSeek R1T2** show strong performance and speed
  improvements. **Huawei** open-sourced a **72B MoE** model with a novel load
  balancing solution. The **MiniMax-M1** hybrid MoE model leads math benchmarks
  on the **Text Arena leaderboard**. **AllenAI** launched **SciArena** for
  scientific literature evaluation, where **o3** outperforms others. Research
  from **Sakana AI Labs** introduces **AB-MCTS** for code generation, improving
  synthesis benchmarks.
companies:
  - meta
  - scale-ai
  - unslothai
  - zhipu-ai
  - deepseek
  - huawei
  - minimax-ai
  - allenai
  - sakana-ai-labs
  - openai
models:
  - gemma-3n
  - glm-4.1v-thinking
  - deepseek-r1t2
  - mini-max-m1
  - o3
  - claude-4-opus
  - claude-sonnet
  - moe-72b
topics:
  - model-performance
  - vision
  - conv2d
  - float16
  - training-loss
  - open-source
  - model-benchmarks
  - moe
  - load-balancing
  - scientific-literature-evaluation
  - code-generation
  - adaptive-tree-search
  - synthesis-benchmarks
people:
  - alexandr_wang
  - natfriedman
  - steph_palazzolo
  - thegregyang
  - teortaxes_tex
  - denny_zhou
  - agihippo
  - danielhanchen
  - osanseviero
  - reach_vb
  - scaling01
  - ndea
---


**a quiet day**

> AI News for 7/1/2025-7/2/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (220 channels, and 7625 messages) for you. Estimated reading time saved (at 200wpm): 603 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Today's Current Thing was Soham Parekh stories, which only affect a small number of startups. If you're seeking interesting AI stories, perhaps you could consider buying your [own personal open source humanoid robot](https://www.youtube.com/watch?v=BS92RdBvI90), available [today](https://x.com/kscalelabs/status/1940108075064865126).

---

# AI Twitter Recap

**The Great AI Talent Shuffle: Meta, OpenAI, and Scale AI**

- **Meta Hires Scale AI CEO Alexandr Wang and Other Top Talent**: **Meta** has hired **Scale AI CEO Alexandr Wang** as its new **Chief AI Officer** to lead a research group focused on developing superintelligence, [working alongside **@natfriedman**](https://twitter.com/TrapitBansal/status/1940124057057574926). The move is part of a larger talent acquisition from rivals, with [@steph_palazzolo reporting on **14 new hires** for Mark Zuckerberg's team](https://twitter.com/steph_palazzolo/status/1940058865531269138). To facilitate this without a full acquisition review, [**Meta** acquired a **49% non-voting stake** in **Scale AI** for **$14.3 billion**](https://twitter.com/DeepLearningAI/status/1940153434671362268), effectively doubling **Scale AI's** valuation to **~$28 billion**. [@TheGregYang joked that the new team's office is at **1 Hacker Way, Menlo Park**](https://twitter.com/TheGregYang/status/1940276530992881970).
- **Commentary on the Missionaries vs. Mercenaries Narrative**: In response to claims that the new hires are "mercenaries," [@Teknium1 argues that the researchers might genuinely believe in **Meta's** new vision](https://twitter.com/Teknium1/status/1940382999423357007), finding it more appealing than **OpenAI's**. [@teortaxesTex speculates that **Yann LeCun** may have lost influence within **Meta** as a result of the changes](https://twitter.com/teortaxesTex/status/1940112275743891508). Meanwhile, [@denny_zhou quipped that it's now **Sam Altman's** "turn to strike back" by hiring **Yann**](https://twitter.com/denny_zhou/status/1940418308156862799), a sentiment that [@agihippo suggests would slow AI progress significantly](https://twitter.com/agihippo/status/1940419051953795377).

**Model Releases, Benchmarks, and Performance**

- **Gemma 3N Technical Deep Dive**: [@danielhanchen identified several quirks with **Gemma 3N**](https://twitter.com/danielhanchen/status/1940073369648734571), including **vision NaNs** on float16, large **Conv2D weights** causing FP16 overflows, and numerous training losses, noting that **UnslothAI** has fixed the NaN issues. For those interested in the research behind the model, [@osanseviero shared links to papers on **Altup, LAuReL, MatFormer**, and other key components](https://twitter.com/osanseviero/status/1940127957730959494).
- **New Chinese Open-Source Models Gain Traction**: **Zhipu AI** released **GLM-4.1V-Thinking**, a **9B VLM** that [@teortaxesTex notes has a high density of](https://twitter.com/teortaxesTex/status/1940344040278593852) `<Wait>` [tokens in its thinking process but appears strong on a vibe check](https://twitter.com/teortaxesTex/status/1940344040278593852). **DeepSeek** released **DeepSeek R1T2**, which [@reach_vb highlights is **200% faster** than R1-0528, beats R1 on **GPQA** & **AIME 24**, and is available under a **MIT license**](https://twitter.com/reach_vb/status/1940536684061643239). Additionally, [@teortaxesTex pointed out that **Huawei** open-sourced its **72B MoE**, noting its original load balancing solution, **MoGE**](https://twitter.com/teortaxesTex/status/1940341153754382688).
- **Model Leaderboard Updates and New Benchmarks**: **MiniMax-M1**, an open-source hybrid **MoE** model, [is now live on the **Text Arena leaderboard** at #12 and has climbed to **#1 in math**](https://twitter.com/MiniMax__AI/status/1940243199500677218). **AllenAI** introduced **SciArena**, a new platform for evaluating models on scientific literature, where [@scaling01 noted that **o3 is "crushing all other models"**](https://twitter.com/scaling01/status/1940065085776666679). On **METR**, [@scaling01 observed that while **Claude 4 Opus** and **Sonnet** fall behind **o3**](https://twitter.com/scaling01/status/1940089136104579515), they are on the same level when selecting for an **80% probability** of success on a task](https://twitter.com/scaling01/status/1940093773440008512).
- **New Research on Model Capabilities**: A paper from **Sakana AI Labs** on **AB-MCTS** frames code generation as an adaptive tree search guided by external feedback, which [@ndea highlights beats baselines on synthesis benchmarks like **ARC-AGI**](https://twitter.com/ndea/status/1940166177424384354). Separately, a new preprint highlighted by [@_akhaliq introduced a reasoning benchmark where leading models like **o3** still "fall flat"](https://twitter.com/_akhaliq/status/1940066518307381616).
- **MLX Framework Momentum**: [@awnihannun celebrated that over **5,000 MLX models** have been uploaded to **Hugging Face**](https://twitter.com/_akhaliq/status/1940058736728379819). Showcasing its power, another tweet from [@awnihannun showed **DeepSeek-R1-0528-5bit** on **MLX** pushing an **M3 Ultra** to use **501GB of memory**](https://twitter.com/awnihannun/status/1940067135054913892).

**Agent Tooling, Frameworks, and Infrastructure**

- **Context Engineering and LangGraph**: **LangChain** released a detailed guide on **"Context Engineering,"** a key part of agent building, [including popular patterns and how to implement them with **LangGraph**](https://twitter.com/LangChainAI/status/1940440271126438118). They also showcased how **Exa AI Labs** built a production-ready deep research agent using **LangGraph**, featuring a multi-agent system with snippet-first reasoning and structured JSON output](https://twitter.com/LangChainAI/status/1940062841454960831). A new tutorial demonstrates how to use **LangGraph Assistants** to [turn static agents into flexible, runtime-configurable systems](https://twitter.com/LangChainAI/status/1940426489314361382).
- **The Rise of MCP**: The **MCP** standard is gaining traction for enabling agents to use tools. [@vikhyatk remarked that after understanding MCP, they'll "never look at the internet the same way"](https://twitter.com/vikhyatk/status/1940255085894017512). **LlamaIndex** introduced a plug-and-play **MCP server** for its **LlamaCloud** document extraction capabilities, [allowing tools like **ChatGPT** and **Claude** to access extraction agents with a standardized schema](https://twitter.com/jerryjliu0/status/1940209573585199234). [@simonw shared a method for adding the official **Playwright** browser automation MCP to **Claude Code**](https://twitter.com/imjaredz/status/1940251061589352802).
- **Infrastructure and Hardware Updates**: **Together AI's** first **NVIDIA GB200** cluster, built by **Dell**, is preparing to go live, with [@vipulved noting that each rack provides **1.4 exaflops of inference performance**](https://twitter.com/vipulved/status/1940242672138244268). For multi-node training, **SkyPilot** announced a new feature to simplify fast GPU networking setup (**Infiniband/TCPXO/RDMA**), [claiming it provides a **~4x speedup** and saves over **$2K** in debugging costs](https://twitter.com/skypilot_org/status/1940473447739756592).
- **Perplexity's Comet Agent and Veo 3**: **Perplexity** is testing its new agent, **Perplexity Comet**, on legacy websites for tasks like bill payments and cancellations, with [@AravSrinivas stating it will be simple "very soon"](https://twitter.com/AravSrinivas/status/19401049473765622206). Subscribers can now DM for help with the agent. He also announced that **Veo 3** video generation is [coming shortly to **Max** users](https://twitter.com/AravSrinivas/status/1940507473095623068).
- **Hugging Face Updates**: **Hugging Face** announced the shutdown of **HuggingChat**, which [@reach_vb described as a "legendary run" that served over a million users and validated open-source models](https://twitter.com/reach_vb/status/1940105535505764427). In parallel, [@TheZachMueller highlighted a major update to the](https://twitter.com/TheZachMueller/status/1940195982169579805) `transformers` [library](https://twitter.com/TheZachMueller/status/1940195982169579805): it now includes a baked-in HTTP server with an **OpenAI-spec compatible API**, launchable with `transformers serve`.

**Robotics and Embodied AI**

- **The Global Frequency: A Vision for VR Social Gaming**: **John Carmack** posted a detailed proposal for a **Beat Saber** feature called **"The Global Frequency,"** envisioning a massively multiplayer experience where thousands of players can join a scheduled playlist of songs simultaneously](https://twitter.com/ID_AA_Carmack/status/1940451656057139534). The concept aims to solve the failure of VR to deliver large-scale social experiences by creating a persistent, accessible "club" atmosphere with shared leaderboards and celebrations.
- **Open-Source Humanoid Robots**: **Genesis AI** launched as a full-stack robotics company aiming to build generalist robots, as shared by [@dchaplot](https://twitter.com/dchaplot/status/1940061390678733010). Separately, **K-Scale Labs** introduced **K-Bot**, touted as the "world's first open-source humanoid robot that is affordable, available and made in America," with [@hingeloss sharing the announcement](https://twitter.com/hingeloss/status/1940120025991672287).

**Broader Tech & Societal Implications**

- **US Science Funding Crisis**: A viral tweet from [@kareem_carr, shared by **Yann LeCun**](https://twitter.com/ylecun/status/1940171025834287229), warns that the **US government** is aiming to get rid of **a quarter-million people** involved in science research and education by **2026**. This sentiment was echoed by many, including [@zacharynado who shared a post from **Indiana University** faculty expressing dismay over the situation](https://twitter.com/zacharynado/status/1940113575671894441) and another from [@SpencerHakimian stating this policy "is not going to make us great again"](https://twitter.com/ylecun/status/1940240965597634739).
- **Food Safety and Industrial Supply Chains**: [@karpathy posted a widely-circulated thread arguing for the necessity of **test-based certification** for food safety](https://twitter.com/karpathy/status/1940181840201228384). He contends that the complexity of modern industrial food production introduces numerous contaminants (pesticides, heavy metals, plastics) that the **FDA** lacks the resources to thoroughly monitor, potentially contributing to long-term public health issues.
- **Open-Plan Offices and Developer Productivity**: [@AmandaAskell criticized the contradiction in tech companies paying millions for talent but housing them in "loud, distracting open-plan offices,"](https://twitter.com/AmandaAskell/status/1940074872241320067) sparking a significant conversation on developer productivity.
- **The Future of Search and AI Scraping**: [@vikhyatk argues that the future of search lies in "lightweight research agents" and that if websites block AI scrapers, models like **o4-mini-high** will simply send users to competitors](https://twitter.com/vikhyatk/status/1940227029389255109), a view supported by [@inerati's post about blocking **Common Crawl**](https://twitter.com/inerati/status/1940076601456078941).

**Humor & The Soham Parekh Saga**

- **The Soham Parekh Phenomenon**: A PSA from [@Suhail warned about an individual named **Soham Parekh** who allegedly works at 3-4 startups simultaneously](https://twitter.com/andriy_mulyar/status/1940391177632792696), setting off a firestorm in the tech community. The story quickly became a meme, with [@Yuchenj_UW joking with a new set of acronyms: **AI** (An Indian), **API** (A Person in India), **AGI** (A Genius Indian), and **ASI** (A Soham Indian)](https://twitter.com/Yuchenj_UW/status/1940506761699774600).
- **Companies and Founders Respond**: The story prompted several founders to check their applicant logs. [@aidan_mclau joked that his accountant and front-end contractor share the same name](https://twitter.com/aidan_mclau/status/1940496760843190675), while [@vikhyatk confirmed that he applied to **moondream**](https://twitter.com/vikhyatk/status/1940517976903684328), and [@pirroh stated he was rejected by **Replit** because their hiring bar is "that high"](https://twitter.com/pirroh/status/1940540351158333709).
- **Classic Tech Humor**: [@johannes_hage posted a viral (and mathematically flawed) joke that **Zuckerberg** could have given every American **$1 million** with the money he spent on one OpenAI researcher](https://twitter.com/johannes_hage/status/1940311985536848310). In a relatable take on AI alignment, [@_jasonwei shared dating advice from his AI buddy](https://twitter.com/_jasonwei/status/1940126761489928468): "You are like a neural net in the middle of training... Better to train to convergence instead of taking an early checkpoint snapshot."

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. New Open-Source AI Model Announcements and Benchmarks

- [**DeepSeek-r1-0528 in top 5 on new SciArena benchmark, the ONLY open-source model**](https://i.redd.it/xxfqfefhpcaf1.jpeg) ([Score: 417, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1lphhj3/deepseekr10528_in_top_5_on_new_sciarena_benchmark/)): **The accompanying image shows a ranking of the top five large language models on the new SciArena scientific literature research benchmark by Allen AI, as described in their announcement [here](https://allenai.org/blog/sciarena). Notably, DeepSeek-R1-0528 is the only open-source model in the top five (with others being proprietary such as OpenAI's o3, Claude-4-Opus, and Gemini-2.5-Pro), visually highlighted in the bar chart (see [image](https://i.redd.it/xxfqfefhpcaf1.jpeg)). This demonstrates DeepSeek's strong generalization and reasoning abilities in the scientific literature domain, matching or surpassing closed-source competitors according to the latest benchmark scores.** Technical commenters note that DeepSeek-R1-0528 provides significantly superior local performance in diverse tasks compared to other locally run models, though it can be idiosyncratic and less faithful in following explicit instructions than alternatives like V3-0324. The model's value in workflows and agentic tasks, despite quirks, is repeatedly emphasized.
    - The DeepSeek-r1-0528 model stands out on the SciArena benchmark as the only open-weight (as opposed to open-source) contender in the top 5, and users report that even with quantized versions (e.g., q5_K_M) on hardware like the Apple M3 Ultra, it exhibits superior performance on diverse tasks despite slower inference speeds. One user noted that it outperforms all other locally-runnable models for their workflows, prompting major changes in their setup to center around this model.
    - Comparative testing between R1-0528 and V3-0324 highlights significant behavioral differences: R1-0528 demonstrates more initiative and creative problem-solving, ignoring instructions at times, while V3-0324 follows directions more strictly but lacks cleverness. Thus, R1-0528 is preferred for open-ended or challenging tasks, whereas V3-0324 is better for strict instruction following.
    - Leaderboard analysis reveals striking differences between models: While r1-0528 impresses compared to earlier R1, models like Llama 4 Maverick perform poorly, and hybrid architectures like minimax M1 (the only 'hybrid transformer' listed) are emerging, while Gemini-2.5-flash underperforms expectations, especially when compared to other small or non-reasoning models such as gpt 4.1 mini and o4 mini, which achieve notable results for their model size and price.
- [**DiffuCoder 7B - New coding diffusion LLM by Apple**](https://www.reddit.com/r/LocalLLaMA/comments/1lpoqlu/diffucoder_7b_new_coding_diffusion_llm_by_apple/) ([Score: 241, Comments: 56](https://www.reddit.com/r/LocalLLaMA/comments/1lpoqlu/diffucoder_7b_new_coding_diffusion_llm_by_apple/)): **Apple released DiffuCoder-7B, a code generation LLM (base and instruct versions) available on [HuggingFace](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO), with technical details in their [arXiv preprint](https://arxiv.org/pdf/2506.20639). Benchmarks suggest comparable performance to other coding and diffusion models ([benchmark chart](https://preview.redd.it/s19j3dmfneaf1.png?width=1176&format=png&auto=webp&s=927e506f764ded47a4e715aea53c223e56ea7ae6)), and the architecture is a finetuned Qwen2.5-Coder. Users discuss challenges in running the model, especially on Apple Silicon, and whether inference can follow workflows such as the [Dream 7B PyTorch example](https://github.com/HKUNLP/Dream#usage).** Key debates include the novelty of Apple's release, technical curiosity around converting an auto-regressive model to a diffusion process, and practical inference concerns (especially regarding PyTorch on Apple Silicon and lack of official examples).
    - A commenter notes that DiffuCoder 7B is a finetuned version of Qwen2.5 Coder, raising questions about the claim it is a 'diffusion' model, since Qwen2.5 is autoregressive. This sparks technical discussion about the methodology required to convert or reinterpret an autoregressive LLM as a diffusion-based model and what implications or architectural changes would be necessary for this.
    - Technical interest is expressed regarding running DiffuCoder on Apple Silicon, especially since there are no official inference instructions released. A suggested workaround is to attempt running the model similarly to Dream 7B using PyTorch, as detailed in the [HKUNLP/Dream usage instructions](https://github.com/HKUNLP/Dream#usage), which work on Mac GPUs.
    - A technically-oriented user requests clarification on the advantages of diffusion-based LLMs over standard transformer (autoregressive) architectures, seeking a functional or theoretical summary of potential benefits as opposed to just differences.
- [**World's first Intermediate thinking AI model is now Open Source**](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/) ([Score: 118, Comments: 70](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/)): **HelpingAI has open sourced the Dhanishtha-2.0 model, claimed to be the "world's first intermediate thinking AI model," and released it on Hugging Face ([model link](https://huggingface.co/HelpingAI/Dhanishtha-2.0-preview)). Key resources include a [launch video](https://www.youtube.com/watch?v=QMnmcXngoks) and a chat demo ([helpingai.co/chat](https://helpingai.co/chat)), but specific architecture details, training datasets, intermediate reasoning mechanism, and benchmarks beyond self-reported results are not disclosed in the announcement. Community members note that with public release, independent benchmarking will now be possible; related prior discussion ([Reddit thread with screenshots](https://www.reddit.com/r/LocalLLaMA/comments/1lmictu/we_created_worlds_first_ai_model_that_does/)) provides further context.** Comments show skepticism, with some users questioning the legitimacy and substance of the claims ("This smells off") while others reference unusual or unclear performance graphs as a concern about credibility. There is anticipation for real, third-party benchmarks.
    - The original post and linked prior discussion indicate that the model is now open source, allowing for broader benchmarking beyond those initially published by the authors. There is a direct call for empirical validation by testing the model on third-party and community-run benchmarks.
    - A user raises a technical question about the benefit of the 'think -> output -> think' paradigm in contrast to the standard 'think -> output' sequence, specifically when tool use isn't involved. This seeks clarification on architectural or performance gains from implementing intermediate reasoning steps.
    - Several commenters request benchmark results and point out that empirical evaluation, especially against well-known models and tasks, will be necessary to establish the technical merits of this 'intermediate thinking' approach.

### 2. Open-Source AI Model Applications and User Projects

- [**I Built My Wife a Simple Web App for Image Editing Using Flux Kontext—Now It’s Open Source**](https://i.redd.it/nmerohq4miaf1.jpeg) ([Score: 182, Comments: 19](https://www.reddit.com/r/LocalLLaMA/comments/1lq5fqq/i_built_my_wife_a_simple_web_app_for_image/)): **The image displays an open-source web app, 'AI Photo Stylizer,' built using Flux Kontext. The interface allows users to upload images and stylize them with AI-driven artistic profiles like 'Ghibli Soft' and 'Comic Book,' as well as adjust resolution and undo edits. Controls for style selection and prompting custom artistic instructions suggest the use of generative AI or model inference on the backend. The post links to the project's [GitHub repository](https://github.com/TheAhmadOsman/4o-ghibli-at-home).** Commenters appreciate the open-source release and user interface, but technical discussion is minimal; one person links directly to the GitHub repo, highlighting community interest in reviewing or contributing to the code.
    - One user requests the creation of a Docker image for the project, highlighting installation and deployment as a potential hurdle for wider adoption. Containerization with Docker would simplify the process, making it more accessible to users with less technical expertise or those seeking ease of setup in diverse environments.
    - A link to the GitHub repository (https://github.com/TheAhmadOsman/4o-ghibli-at-home) is provided, making the project's codebase directly accessible for technical users interested in reviewing, contributing, or deploying the image editing web app based on Flux Kontext.
- [**Made an LLM Client for the PS Vita**](https://v.redd.it/9x7e4qbmqv8f1) ([Score: 127, Comments: 7](https://www.reddit.com/r/LocalLLM/comments/1ljbn5e/made_an_llm_client_for_the_ps_vita/)): **A developer ported a full-featured Large Language Model (LLM) client called 'vela' to the PS Vita, enabling interaction with LLM endpoints (including vision-capable models via the Vita's camera). The initial experiments involved on-device inference using llama2.c with TinyStories checkpoints, but this new client supports remote model inference, vision, and text capabilities—albeit with limitations in displaying complex markdown/TeX and emoji support due to Vita's UI constraints. The open-source code and VPK for installation are available at [github.com/callbacked/vela](https://github.com/callbacked/vela).** Top-level technical commentary is minimal, mostly expressing general approval and humor rather than in-depth analysis or discussion of implementation challenges.
    - No technical implementation details, performance metrics, or deep discussion present in the comments. All responses are non-technical and express general praise or reactions.

### 3. Cutting-Edge Multi-Modal/Thinking Model Previews

- [**GLM-4.1V-Thinking**](https://huggingface.co/collections/THUDM/glm-41v-thinking-6862bbfc44593a8601c2578d) ([Score: 144, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1lpl656/glm41vthinking/)): **THUDM's GLM-4.1V-Thinking is a multimodal (vision-language) open large language model (LLM), released in a 9B (10B params) checkpoint variant, with significant multilingual capabilities and intended for advanced reasoning tasks. Benchmark charts ([example](https://preview.redd.it/8j97cdmkndaf1.png?width=1031&format=png&auto=webp&s=09eda73e39c216ada7a269993689c60c06118ce0)) show the model outperforming models as large as 72B parameters on generalized image-text-to-text tasks, according to user-shared comparisons. The full suite and demos are made available on [Hugging Face](https://huggingface.co/collections/THUDM/glm-41v-thinking-6862bbfc44593a8601c2578d), aiming to accelerate evaluation and experimentation in scalable RL-based, multilingual multimodal LLM research.** Technical commenters express surprise at the 9B model's benchmark dominance over much larger 72B models, raising questions about real-world generalization. There's also disappointment that only a 9B variant is available, indicating a desire among researchers for larger-scale options.
    - GLM-4.1V-Thinking's benchmark performance is notable, with discussion focused on the surprising capability of a `9B` parameter model to outperform a recent `72B` parameter model in vision-language tasks (as seen in the linked [benchmark image](https://preview.redd.it/8j97cdmkndaf1.png?width=1031&format=png&auto=webp&s=09eda73e39c216ada7a269993689c60c06118ce0)). There is skepticism and anticipation about how these benchmark results will translate to real-world tasks.
    - Technical readers highlight the lack of 'thinking' VLMs (vision-language models) and note that Kimi was among the first, though support in mainstream inference frameworks (llamacpp, vllm, lmstudio, ollama) is limited. They mention that 'thinking' models could match larger non-thinking models in performance at lower parameter counts, given reduced output verbosity, which could impact latency and resource requirements for local deployments.
    - Significant improvements from THUDM are praised: GLM4 model code (`Glm4vForConditionalGeneration`) integrated directly into the HuggingFace transformers package ensures better forward compatibility and stability, whereas previous models like CogVLM suffered from breakages due to standalone code drops that failed to track transformers updates, causing models to become unusable after a few weeks.
- [**World's first Intermediate thinking AI model is now Open Source**](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/) ([Score: 118, Comments: 70](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/)): **HelpingAI has open sourced the Dhanishtha-2.0 model, claimed to be the "world's first intermediate thinking AI model," and released it on Hugging Face ([model link](https://huggingface.co/HelpingAI/Dhanishtha-2.0-preview)). Key resources include a [launch video](https://www.youtube.com/watch?v=QMnmcXngoks) and a chat demo ([helpingai.co/chat](https://helpingai.co/chat)), but specific architecture details, training datasets, intermediate reasoning mechanism, and benchmarks beyond self-reported results are not disclosed in the announcement. Community members note that with public release, independent benchmarking will now be possible; related prior discussion ([Reddit thread with screenshots](https://www.reddit.com/r/LocalLLaMA/comments/1lmictu/we_created_worlds_first_ai_model_that_does/)) provides further context.** Comments show skepticism, with some users questioning the legitimacy and substance of the claims ("This smells off") while others reference unusual or unclear performance graphs as a concern about credibility. There is anticipation for real, third-party benchmarks.
    - The original post and linked prior discussion indicate that the model is now open source, allowing for broader benchmarking beyond those initially published by the authors. There is a direct call for empirical validation by testing the model on third-party and community-run benchmarks.
    - A user raises a technical question about the benefit of the 'think -> output -> think' paradigm in contrast to the standard 'think -> output' sequence, specifically when tool use isn't involved. This seeks clarification on architectural or performance gains from implementing intermediate reasoning steps.
    - Several commenters request benchmark results and point out that empirical evaluation, especially against well-known models and tasks, will be necessary to establish the technical merits of this 'intermediate thinking' approach.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Veo 3 AI Video Generation Impact and Creative Uses

- [**I Used to Make $200k Commercials — This AI Ad Took Me 2 Days & $500**](https://v.redd.it/7sp2esfaxcaf1) ([Score: 215, Comments: 110](https://www.reddit.com/r/aivideo/comments/1lpic34/i_used_to_make_200k_commercials_this_ai_ad_took/)): **The OP presents a case study in which a B2B commercial, traditionally costing ~$200k with a 30-person crew and several months of work, was instead created in 2 days for $500 using AI tools: Veo 3 + FLORA (video generation), ElevenLabs (AI voice-over), and some manual ADR. A detailed generation prompt for the opening shot was provided, demonstrating precise creative intent and granular prompt engineering for asset generation (location, set, actor characteristics, shot specs). The process required generation of 100+ clips for a 27-shot final product, highlighting both AI's cost and time advantages and its current technical limitations ("not plug-and-play").** Top comments from an industry veteran predict a 98-99% reduction in traditional production crew jobs within 5-10 years due to AI, emphasizing that AI-generated content already often surpasses average human cinematographers in image quality and cost efficiency. Other comments question specific costs and point out current weaknesses (accent inconsistencies), but overall the technical consensus trends toward anticipating major disruption pending greater AI workflow usability.
    - Veteran commercial film producers predict a dramatic reduction (up to `98-99%`) in traditional production crew roles over the next 5-10 years due to AI-driven content creation, with knock-on effects on secondary industries (e.g., sound stages, equipment rental, catering). They note that denial within the industry is strong, but argue that AI quality and cost-effectiveness are rapidly surpassing even average human professionals, forecasting that AI-generated images will soon match top cinematographers.
    - Current AI-generated commercials are considered passable for lower-tier clients, but quality and cleanliness still fall short of standards demanded by major agencies and brands (e.g., BBDO, TBWA under Omnicom). There is also a substantial legal barrier: *none of these images are copyrightable*, posing a major challenge for corporate use due to rights and legal considerations. These AI outputs are, however, valued as powerful previsualization (previz) tools.
    - An emerging trend in advertising is the dominance of 'vertical, non-pro looking commercials,' which are prolific and often user-generated. This informality, combined with AI tools, is changing market expectations around commercial quality and production values, further undermining traditional studio workflows.
- [**Thanks to VEO 3 I was finally able to finish this documentary about a local institution from my town**](https://v.redd.it/2h6yum2dniaf1) ([Score: 167, Comments: 44](https://www.reddit.com/r/aivideo/comments/1lq5n45/thanks_to_veo_3_i_was_finally_able_to_finish_this/)): **The post highlights the use of VEO 3, an AI video generation platform by Google DeepMind, which enabled a user to complete a documentary about a local institution—demonstrating VEO's ability to synthesize convincing, narrative-driven video content. Given the lack of access to the video (403 error), no technical specifics or benchmarks about generated quality, prompt design, runtime, or model architecture could be evaluated directly from the source.** Top comments praise the documentary's realism and writing, suggesting that human curation of AI content, particularly in the inclusion of well-crafted humor, results in a more authentic and engaging output compared to fully automated AI generation.
    - There is implicit technical praise for VEO 3's photorealistic video generation and the subtle integration of AI-generated content that is almost indistinguishable from real footage. One user notes that *'5 Years ago nobody would have thought this wasn't real,'* highlighting the rapid advancement in generative model fidelity and realism. This suggests notable progress in temporal coherence, fine detail, and authenticity from models like VEO 3 compared to earlier state-of-the-art approaches.
- [**I Used to Make $200k Commercials — This AI Ad Took Me 2 Days & $500**](https://v.redd.it/7sp2esfaxcaf1) ([Score: 219, Comments: 110](https://www.reddit.com/r/aivideo/comments/1lpic34/i_used_to_make_200k_commercials_this_ai_ad_took/)): **A video director highlights that he recreated a $200,000 B2B commercial video in just 2 days for $500 using AI tools: Veo 3 and FLORA (for video), ElevenLabs (for voice-over), and manual ADR for refinement. The process involved generating over 100 clips for a 27-shot video—without actors, set, or crew—contrasting traditional production (2 months, 30 people, $175,000+) against modern generative workflows. The director shares his detailed prompt for a cinematic office shot, noting technical hurdles (the process is not yet 'plug-and-play'), and projects major disruption for the ad industry as AI matures.** Top commenters with extensive commercial film experience predict a 98-99% reduction in crew employment in 5-10 years, emphasizing that industry denial persists despite AI already surpassing average cinematographers on image quality and cost. Others raise technical feedback on model consistency (e.g., variable voice accent) and question the cost breakdown of the AI workflow.
    - A veteran in commercial film production predicts a massive employment reduction (up to 98-99%) in production crews over the next 5-10 years as AI tools mature, highlighting that AI already produces higher quality output than the average cinematographer at a fraction of the cost. There is also concern about the secondary industries (such as equipment rental and catering) being heavily impacted, and a general industry denial regarding the disruption AI will cause.
    - An ad industry professional notes that while current AI-generated commercials are passable for some low-end clients, they do not yet meet the technical and legal standards required by major brands—especially around image cleanliness and copyrightability, which present significant hurdles. The commenter adds that the real disruption is currently coming from the proliferation of non-professional, vertical-format ads rather than fully AI-generated content, but recognizes AI's promise for previsualization (previz) workflows.
- [**Thanks to VEO 3 I was finally able to finish this documentary about a local institution from my town**](https://v.redd.it/2h6yum2dniaf1) ([Score: 170, Comments: 44](https://www.reddit.com/r/aivideo/comments/1lq5n45/thanks_to_veo_3_i_was_finally_able_to_finish_this/)): **OP credits the use of VEO 3, likely referencing Google's Veo video generation model, to successfully complete a documentary focused on a local institution. While technical details about model prompts, workflow integration, or post-processing are not provided, the post highlights VEO 3's capability for producing convincing and coherent long-form video content, raising implications for project feasibility and creative autonomy.** Commenters point out the difficulty of distinguishing AI-generated media from real footage, noting strong editorial control and humor that suggest active human curation rather than a fully automated pipeline. There is implicit debate about the boundary between AI autonomy and guided creativity in multimedia content creation.
    - The discussion highlights that with advancements such as "VEO 3," AI-generated video content can now reach a realism level that would have been considered indistinguishable from real footage five years ago. Commenters note the blending of AI visual fidelity with human-driven, well-crafted humor, suggesting high creative control enabled by newer generative models rather than fully autonomous generative output.
- [**Demis teasing “playable” Veo 3 worlds (or AI video games)**](https://www.reddit.com/r/singularity/comments/1lplq1w/demis_teasing_playable_veo_3_worlds_or_ai_video/) ([Score: 350, Comments: 90](https://www.reddit.com/r/singularity/comments/1lplq1w/demis_teasing_playable_veo_3_worlds_or_ai_video/)): **Demis Hassabis, CEO of Google DeepMind, has teased on Twitter the possibility of 'playable' Veo 3 worlds or AI-generated video games, suggesting future integration of Veo (Google's state-of-the-art text-to-video AI) with interactive or game-like environments. This move indicates explorations into leveraging advanced AI generative models for real-time, user-interactive digital worlds, beyond passive video synthesis, potentially reshaping game content creation or procedural storytelling. The referenced tweet may signal a research or prototype direction, though no benchmarks or technical details have been disclosed.** Top comments recall Hassabis's early career in AI-driven game development, framing this as a return to his foundational interests in using AI for interactive digital entertainment. There's a tone of recognition for his track record in transformative AI applications within games.
    - Speculation centers on a future where entire game worlds, behaviors, and systems could be generated by advanced AI models rather than traditionally hand-coded, likening current programming methods to legacy manual processes like punch cards. This would dramatically shift the nature of game development towards higher-level design, with AI synthesizing complex environments and interactions on demand.
    - Multiple users note early but crude public web demos of generative AI games exist, allowing basic movement and interactions (e.g., moving via arrow keys). While these demos are rudimentary, their existence is cited as leading indicators that more sophisticated, fully playable AI-generated worlds may arrive soon.

### 2. Kontext & ComfyUI Advanced Reference and Workflow Techniques

- [**Boosting Success Rates with Kontext Multi-Image Reference Generation**](https://www.reddit.com/r/StableDiffusion/comments/1lpqkkr/boosting_success_rates_with_kontext_multiimage/) ([Score: 176, Comments: 20](https://www.reddit.com/r/StableDiffusion/comments/1lpqkkr/boosting_success_rates_with_kontext_multiimage/)): **The post presents a systematic exploration of improving image-to-image attribute transfer using the Kontext multi-image reference in ComfyUI, focusing on transferring clothing from a reference to a model image. Empirical testing demonstrates that naively concatenating images or using a cropped reference on a white background yields poor or inconsistent results due to excessive context or loss of relevant cues. The effective approach is to crop the reference to retain only the core clothing element along with minimal body cues (e.g., arms/legs), boosting transfer accuracy to an 80%+ success rate. The provided workflow and [Civitai model link](https://civitai.com/models/1738322) illustrate this methodology, and a direct comparison of outputs is shared via image links.** Commenters note that while this technique enhances Kontext's utility, it doesn't supplant LoRAs, particularly for face consistency. There is technical discussion about workflow replication, especially regarding custom aspect ratios and controlling output resolution, as well as prompt engineering suggestions aligned with BFL guidance (i.e., use imperative commands over descriptive prompts).
    - There's a discussion on comparing Kontext multi-image reference generation versus LoRA approaches, especially regarding fidelity of facial features—one user notes that Kontext can still result in less accurate faces (e.g., Sydney’s face "quite off") compared to well-trained LoRAs, which retain higher detail in transferred images.
    - Several users investigate workflow nuances: questions are raised about generating outputs at portrait resolution when original reference images differ in aspect ratio, since Comfy’s official guides only demonstrate image-stitching without explicit resolution control, highlighting a technical gap in current examples.
    - Technical debate emerges around reference-image chaining in ComfyUI: some users wonder if using stitched versus chained latent nodes yields better results, and whether the sequence (e.g., person identity versus clothing) affects transfer quality. Additional technical inquiries explore prompt engineering—one user cites BFL, recommending commands over descriptive prompts for better results based on their own tests.
- [**nunchaku your kontext at 23.16 seconds on 8gb GPU - workflow included**](https://www.reddit.com/r/StableDiffusion/comments/1lpn2wa/nunchaku_your_kontext_at_2316_seconds_on_8gb_gpu/) ([Score: 141, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1lpn2wa/nunchaku_your_kontext_at_2316_seconds_on_8gb_gpu/)): **The post highlights using MIT Han Lab's [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) extension for running the int4 version of the 'kontext' model on an 8GB GPU with inference times of** `~23s` **per image. Key references include the required [int4 kontext model weights](https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev/tree/main), and a shared workflow. The workflow does not require speed LoRA or SAGE Attention modules. The post also requests information on fast conversion of Safetensors models to int4 format.** Commenters note performance discrepancies (e.g. `55s` vs `23s` on GPUs with similar or higher VRAM), indicating workflow or configuration could be a factor. One user reports a deprecation warning relating to input tensor dimensions, suggesting a potential update in the pipeline's PyTorch preprocessing.
    - A user reports achieving a significant speed improvement—23 seconds runtime on an 8GB GPU—compared to another user's 55 seconds on an RTX 3060 12GB, both reportedly using the default nunchaku workflow, highlighting optimization or configuration differences affecting performance.
    - An error is noted with nunchaku involving deprecated usage in PyTorch: passing a 3D torch.Tensor for `img_ids` should be changed to a 2D tensor to comply with updated library requirements, indicating necessary workflow or codebase updates.
    - Several users encounter setup and dependency issues running nunchaku in ComfyUI (portable/Windows), including module not found errors and apparent challenges with installing or importing 'nunchaku', reflecting common barriers in environment configuration and package management.
- [**Comparison "Image Stitching" vs "Latent Stitching" on Kontext Dev.**](https://www.reddit.com/gallery/1lpx563) ([Score: 138, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1lpx563/comparison_image_stitching_vs_latent_stitching_on/)): **The post compares two image composition workflows available in Kontext Dev: Image Stitching (which merges multiple character/image references to synthesize novel scenarios) and Latent Stitching (which overlays or edits a primary image using features extracted from a secondary image within the latent space). Workflows for both 1-image and 2-image scenarios are provided as a downloadable JSON ([workflow file](https://files.catbox.moe/q3540p.json)), and additional guidance is available via a referenced Reddit thread on Stable Diffusion workflows. The core implementation feature is a toggle to switch seamlessly between the two composition modes during inference or editing sessions.** Technically-oriented commenters note successful outcomes with the author's prior NAG workflow and inquire about controlling pose using Kontext as a ControlNet mechanism—where it appears there are limitations in enforcing exact pose references, indicating a gap in current tool capabilities.
    - A user inquires about using Kontext as a ControlNet to enforce exact character poses, reporting difficulty in achieving precise pose control. This highlights current limitations of Kontext's integration with control frameworks for exact spatial conditioning and may point to a lack of direct pose enforcement tools or necessary guidance for such use-cases.
    - One commenter asks about the distinction between 'image concatenate' and 'image stitching,' suggesting ambiguity in their operational differences. This reflects a broader confusion or overlap in terminology within image synthesis or compositing workflows, indicating a need for clearer definitions or technical breakdowns of these processes.
    - Discussion arises around the use of the 'fluxkontextimagescale' node, where anecdotal reports suggest it works better with certain stitching methods, and that manual resolution settings can impact quality depending on the chosen technique. However, there is noted uncertainty and a lack of consensus on optimal pairings or workflows, emphasizing the need for benchmarked guidance or documentation.
- [**I Built My Wife a Simple Web App for Image Editing Using Flux Kontext—Now It’s Open Source**](https://i.redd.it/iivpnnllliaf1.jpeg) ([Score: 126, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1lq5evu/i_built_my_wife_a_simple_web_app_for_image/)): **The image showcases the 'AI Photo Stylizer,' a web app for applying artistic styles to photos—built using Flux Kontext and now open source. The UI offers controls for selecting style profiles (like 'Ghibli Soft' and 'Comic Book'), adjusting resolution, and features such as undo/redo and download. The linked [GitHub repo](https://github.com/TheAhmadOsman/4o-ghibli-at-home) provides the project's source for others to explore or contribute.** One commenter notes the convenience of Flux Kontext for image editing versus traditional tools like GIMP, but mentions its hardware requirements as a potential limitation.
    - Several comments discuss the hardware requirements of Flux Kontext, with one user noting that while it offers easier image manipulation than GIMP for certain tasks, it cannot yet run efficiently on all hardware, highlighting constraints for low-resource devices compared to traditional image editors.
    - A suggestion was made to enable running the app on Google Colab, possibly with Gradio or ngrok integration, to broaden accessibility and allow users with limited local resources to leverage cloud computation for image editing tasks.

### 3. AI-Generated Influencers and Virtual Personas in Social Media

- [**This influencer does not exist**](https://i.redd.it/24k4hbp5rhaf1.png) ([Score: 412, Comments: 94](https://www.reddit.com/r/OpenAI/comments/1lq11ms/this_influencer_does_not_exist/)): **The image shows a tweet detailing the rapid rise of an AI-generated influencer account, which amassed 138k followers within just three months. This underscores the increasing sophistication of AI in generating photo-realistic virtual personas that can maintain consistent aesthetics across multiple images, fostering plausible online identities. The post highlights a wider trend in which 'influencers' may neither be real people nor have real followers, raising questions about the authenticity and monetization of social media engagement.** Commenters discuss the likelihood that not only the influencer but also their followers are fake (bot accounts) and note that artificial or bot-driven influence has long been part of social media platforms, referencing past purges of fake accounts and the use of manufactured audiences in advertising.
    - Impossible-Glass-487 brings up the significant historical context of bots and fake users in digital advertising ecosystems, referencing the 'Twitter purge' and similar efforts by Facebook. They highlight how social platforms have previously included non-existent users or bots in ad metrics, inflating advertising reach (e.g., Facebook's lookalike audiences), and draw a parallel to how AI-generated influencers may perpetuate similar issues now accessible to individual users rather than exclusively large advertisers.
    - Positive-Raccoon-616 questions the technical process behind consistently matching the generated influencer's face across various AI models. This raises the challenge of identity persistence in AI-generated content, possibly involving techniques like consistent latent embeddings, conditional GANs, or cross-model reference alignment to ensure a recognizable persona is rendered by different models or systems.
- [**This influencer does not exist**](https://i.redd.it/0dzjz6h5qhaf1.png) ([Score: 984, Comments: 140](https://www.reddit.com/r/ChatGPT/comments/1lq0wmi/this_influencer_does_not_exist/)): **The attached image shows a discussion about AI-generated influencers on social media, referencing a case where an Instagram profile with 138K followers features a digital creator who, according to the tweet, does not exist. The technical question in the comments addresses how current models achieve consistency in generated faces and bodies; historically, this required training custom models or LoRA fine-tuning on specific faces, hinting at advancements in AI generative techniques possibly using more dynamic or prompt-consistent approaches.** One commenter expresses skepticism about follower authenticity, suggesting that many might be bots, while another downplays the novelty by stating most influencers are inauthentic regardless of AI.
    - A user asks about advancements in generating consistent faces or bodies for AI-generated influencers, noting that previously, you had to train a specific model or use LoRA (Low-Rank Adaptation) techniques on a face dataset. The question seeks clarity on whether newer methods exist that allow for greater consistency from current generative models and what technical solutions have supplanted these older techniques.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. The AI Arms Race: New Models, Performance Showdowns, and Talent Wars**

- **Elon Proclaims Grok-4's Imminent Arrival**: **Elon Musk** announced **Grok-4-0629** is expected *"Just after July 4"*, fueling speculation in the **LMArena** discord about its absence from the [LLM Arena leaderboard](https://lmarena.ai/). Meanwhile, a new mystery model in the arena is speculated to be **Deepseek R2**, though it reportedly *didn't pass a spatial reasoning test*.
- **New Models Face Scrutiny and Scorn**: **Amazon's Cypher Alpha** model flopped in evaluations, with users in the **aider** and **LMArena** communities calling it a regression and *very bad at everything*, possibly due to a restrictive system prompt. Conversely, while official leaderboards rank it low, **Claude Sonnet 7** received praise for its coding speed, and a comparative analysis in the **OpenAI** discord found **Grok's AVM** excels at explaining complex subjects like **Enso and Zen**, surpassing **ChatGPT**.
- **Meta and Cursor Poach Top AI Talent in High-Stakes Bidding War**: The competition for AI expertise has intensified, with [Amir Efrati reporting](https://x.com/amir/status/1940112288381641026) that **Cursor** hired two senior leaders from **Anthropic's Claude Code team**. In an even more aggressive move, **Meta** is reportedly offering compensation packages up to **$300M** over four years to poach AI researchers from **OpenAI**, with first-year pay exceeding **$100M** [according to this thread](https://x.com/tanayj/status/1940137574141694046).

**Theme 2. Innovations in Model Architecture and Fine-Tuning**

- **Researchers Champion Multiplicative Fine-Tuning Over LoRA**: A new paper introduced **LoRMA**, a parameter-efficient fine-tuning method using *multiplicative* updates that achieves **3x faster convergence over LoRA**, according to the authors of the [paper on HuggingFace](https://huggingface.co/papers/2506.07621). Separately, discussions in the **Unsloth AI** discord clarified that Unsloth's quantization provides an *increased accuracy during fine-tuning*, a benefit distinct from speed improvements.
- **Architectural Debates Pit SSMs Against Transformers**: Engineers debated whether **State Space Models (SSMs)** represent a true paradigm shift or just an incremental improvement over scaled-up **RNNs**. The discussion in the **Yannick Kilcher** discord also highlighted a new paper, [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484), which introduces a hardware-efficient algorithm that enables **DeltaNet** to outperform **Mamba** and **GLA**.
- **New Training Methods Emerge from Top Labs**: **Meta's** ["Transition Matching" paper](https://arxiv.org/abs/2506.23589) claims to one-up **Flow Matching**, though its motivation as a prompt obfuscation attack was debated in **Eleuther**. That community also buzzed about a talk from **Kaiming He** which detailed **Mean Flow Matching**, with a member recommending watching from the [2:43:32 mark of the workshop video](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi).

**Theme 3. Developer Tooling, Workflows, and GPU Nightmares**

- **Local LLMs Get Creative (and Morally Questionable)**: Users in the **LM Studio** discord championed local LLMs for privacy, cost savings, and handling *morally questionable content*, with use cases spanning from creative writing to *Waifu creation*. For implementing local **RAG**, members recommended using [OpenWebUI](https://docs.openwebui.com/tutorials/tips/rag-tutorial/) as a functional API server alongside **LM Studio**.
- **GPU Profiling and Build Headaches Plague Engineers**: In the **GPU MODE** discord, a user reported that the **nsys profiler stalls** when used with `torch.compile`, a known issue also documented in an [NVIDIA forum thread](https://forums.developer.nvidia.com/t/nsys-profile-pytorch-fails-under-torch-compile/332302/5). Other frustrations included **Triton's** nightly builds being broken for months and a `full_state_dict` loading failure in **FSDP 2.0** due to unsharded parameters.
- **Agentic Frameworks and Prompts Aim to Boost Productivity**: **LlamaIndex** announced the release of **Workflows 1.0**, a [lightweight framework for agentic systems](https://www.llamaindex.ai/blog/announcing-workflows-1-0-a-lightweight-framework-for-agentic-systems). In the **OpenAI** community, a team shared a [Strategic Project Matrix prompt](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c) that uses the **S.M.A.R.T.** mnemonic to triage complex obligations across connected data sources.

**Theme 4. The Business of AI: Pricing, Outages, and Poaching**

- **Fuzzy Pricing Models Frustrate Users**: **Cursor's** new pricing model drew fire for its **20% markup** on API pricing and an unclear *unlimited-rate-limited plan*, which some users labeled *borderline scammy*. **Perplexity** also faced backlash for revoking discount codes after launching its pricey **$200/month** **Perplexity Max** tier.
- **Service Outages Cause Developer Scrambles**: **DeepSeek V3** experienced a brief but panic-inducing outage on **OpenRouter** due to a [configuration mistake](https://deepseek.com/) on DeepSeek's end. **Hugging Face** also suffered from failing inference endpoints and announced the shutdown of **HuggingChat**, forcing users to find ways to back up their conversations before their data is deleted.
- **Client Poaching Scandal Hits OpenRouter**: A user reported being solicited by **cometapi**, which appeared to be using public **OpenRouter** data to identify and poach top token users with offers of cheaper rates. The vulnerability was traced to users sending their website information in HTTP headers, a practice documented in the [OpenRouter API documentation](https://openrouter.ai/docs/api-reference/overview#headers).

**Theme 5. The Guardrail Gauntlet and the Perils of AI Hallucinations**

- **"Safe" Models Prove to be Dumb and Delusional**: **Amazon's Cypher Alpha** model performed poorly in evaluations, possibly because its system prompt requires it to *only identify as made by Cypher Labs*, effectively *nerfing* its capabilities. The model also hallucinates its own technical details, falsely claiming it has *117 million parameters* and a *768 embeddings dimension* when prompted on **OpenRouter**.
- **Engineers Debate Guardrail Effectiveness and Model Dangers**: Discussions in the **OpenAI** discord analyzed model safety through the lens of **Rule-Based Rewards (RBR)**, detailed in an [OpenAI blog post](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/?utm_source=chatgpt.com). The consensus was that while American models like **DALL-E 3** lead in quality, their safety filters are often ineffective, which may make the models *more dangerous*.
- **Local LLMs are Arrogant, Untrustworthy Interns**: Users across servers, particularly **LM Studio**, vented about the prevalence of **AI hallucinations**, comparing local models to *arrogant interns* with great language skills but zero reliability. The core takeaway was to distrust outputs regardless of their source, as both cloud-based and local LLMs are often out of date or simply wrong.

---

# Discord: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Guardrails Debate Ensues**: A member sparked discussion on **AI safety**, **guardrails**, and deceptive behaviors of models, questioning the effectiveness of current benchmarks and prompting a detailed analysis of model training and filtering, specifically around **Rule-Based Rewards (RBR)** [as discussed here](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/?utm_source=chatgpt.com).
   - The member concludes that transparency is paramount in creating stronger guardrails and preventing exploitation, particularly when experimenting with existing image models to uncover weaknesses and strengths.
- **Airbrushed Images Cause Controversy**: Members discuss the trend towards hyper-realistic outputs in **AI image generation**, noting **DALL-E 3's** distinctive airbrushed style and contrasting it with the capabilities of **Grok 3** and potential of **4o**.
   - A member generated a funny image of *Zark Muckerberg* using modifiers like *Uneven flash lighting* and *Grainy fingerprint glare*.
- **Grok's AVM Outperforms ChatGPT in Social Settings**: In a comparative analysis, **Grok's AVM** excels in social interactions and explaining complex subjects, like **Enso and Zen**, surpassing **ChatGPT**. Some are pointing to the development of **Grok 4** as next gen.
   - A member reacted to a video of **Mark Zuckerberg**, noting how he is *trying to mask those low social skills maybe they've gotten a little bit better since then*.
- **American Models Reign Supreme**: Despite varying safety approaches, **American image models** like **DALL-E 3**, **Midjourney**, and **Grok** lead in image generation quality due to vast training data.
   - The consensus is that these models may have *no real safety filters* or *they aren't effective*, which may make the models *more dangerous*.
- **Strategic Project Matrix Prompt Triages**: A team shared a [Strategic Project Matrix prompt](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c) designed for managing complex obligations across locations and vendors using connected data (Gmail, Calendar, Dropbox, Google Drive) to build a matrix for triage and next actions.
   - The prompt casts the AI as a **Strategic Intelligence Analyst**, and follows a mnemonic framework **S.M.A.R.T.** (**Scan**, **Map**, **Assign**, **Recommend**, **Triage**) to prioritize projects and avoid commitments from being dropped.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro File Uploads Expand**: Members discussed the expansion of **file uploads** as a feature of **Perplexity Pro** and enterprise accounts, with one user mentioning using a **Samsung discount code** to get a cheaper price, before revocation.
   - A member sarcastically appealed to support to prioritize *fixing a problem where I can give them my money*.
- **Perplexity Cracks Down on Discount Codes**: Perplexity revoked **discount codes** for 1-year Pro subscriptions, including those from Samsung and HubSpot; some users got their codes back after contacting support.
   - This prompted frustration among users who felt penalized for seeking deals.
- **Manus AI Gemini enters Free Chat**: **Manus AI** now offers [unlimited free messages in chat mode](https://manus.im/invitation/R9QW2MZDXXJJ4) with **Gemini**.
   - Members are curious about the limitations of the free tier.
- **Perplexity Max Unveiled**: Perplexity launched **Perplexity Max**, a new tier costing **$200/month**, offering unlimited access to advanced AI models, early access to the [Comet browser](https://x.com/AravSrinivas/status/1940104947376562206) and premium data sources.
   - Users expressed concerns about the high price and potential limitations, with some saying they *keep accidentally pressing on the max only models* and that its *the worst value $200 sub out there*.
- **API Subdomain Exclusion Discussed**: A user sought advice on how to exclude specific subdomains from **Perplexity API** calls, aiming to restrict searches to only content starting with *domain1.com/path1*.
   - The user aimed to exclude subdomains such as *domain1.com/path2*, to better refine API search results.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Update Boosts Speed**: Users report that **Cursor feels faster** after the update, which a team member confirmed is *not a placebo*.
   - However, the specifics of the update remain undisclosed for now.
- **Cursor's Pricing Model Raises Eyebrows**: Users expressed confusion and dissatisfaction with the new pricing model, citing **lack of transparency** regarding rate limits and the transition to API pricing.
   - Some users feel that the **20% markup** on top of API pricing is *borderline scammy*, particularly the idea of an *unlimited-rate-limited plan*.
- **Message Queue System Adds Friction**: The latest update introduces a **queue system** for processing messages sequentially, adding a new step to interrupting processes.
   - Users find it *tedious* to interrupt a running process and must now hit the stop button instead.
- **Claude Code Edges Out Gemini, but at a Cost**: **Claude Code** outperforms **Cursor** in many evaluations, though the **Anthropic API** is *much more expensive*.
   - While the **Gemini 2.5 Pro** model shows promise, it is reportedly *terrible at editing files*.
- **Background Agents Creates Github Branch Locally**: Users are reporting that **Background Agents** create a new branch on github.com while Cursor Chat creates files locally, which one member doesn't like, asking *Why would I want to switch branch? I never want a second branch ever in my life.*
   - Members highlighted they are heavily tied into git and suggested spending time *asking the agent how it should do git for you*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Quantization Ups Accuracy**: A user inquired how much performance improvement **Unsloth-ing** a model with 101 layers would bring over straight quantization and was clarified that the main benefit is an *increased accuracy during fine-tuning*.
   - They confirmed the 4bit dynamic of unsloth is mostly to offset the diff to fp16 during training, noting **bartos quants** are also effective, especially for **GGUF** finetunes.
- **Ascend GPUs Ace MoE Models**: A general **MoE model** trained completely on **Ascend GPUs**, with an optimized architecture, was posted on ArXiv: [https://arxiv.org/pdf/2505.21411](https://arxiv.org/pdf/2505.21411).
   - The model was trained end-to-end including RL (reasoning model) and has similar benchmarks to **Qwen3-32b**, demonstrating a significant advancement in utilizing **Ascend GPUs** for large language models.
- **Qwen3-32B Merge Mishaps Manifest**: Users reported encountering a `RuntimeError` while trying to save a merged 16-bit **Qwen3-32B-unsloth-4bit** model, tracing the issue to dimension mismatches in the attention layers.
   - The problem stems from starting with a local **4bit** checkpoint, bypassing the download of the **16bit** base model needed for merging; a workaround involves manually pointing to the downloaded base model.
- **Intel Arc Pricing Sparks Squabble**: Discussion arose around the pricing of **Intel Arc A770** cards, with claims that a phone clarification indicated a price of **3k USD** per card.
   - Concerns were voiced about its value compared to multi-card systems at **5k USD** and the **RTX Pro 5000** priced at **4.5-5k USD**.
- **ChessFish.io facilitates Chess**: A member promoted [ChessFish.io](https://www.chessfish.io/), a chess website for *analyzing, learning, and playing chess casually*.
   - The website is free to try without requiring an account.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Cypher Alpha Flops in Evaluations**: Members found **Cypher Alpha**, an **Amazon** model, performed poorly, possibly due to its restrictive system prompt requiring it to only identify as made by **Cypher Labs**.
   - The imposed limitations by the system prompt possibly *nerf* the model's capabilities.
- **Grok-4-0629 Anticipated Post-Independence Day**: **Elon Musk** announced **Grok-4-0629** is expected to be released *"Just after July 4"*, fueling speculation about its absence from the [LLM Arena](https://lmarena.ai).
   - Some users speculate the model has not been tested in the arena due to potentially underwhelming performance and that **xAI**'s is attempting bruteforcing for better results.
- **OpenAI's Open-Weights License Restrictions**: Speculation arose that **OpenAI's** open-weights model license might restrict direct competition with their API.
   - Users discussed the potential for *a high quality model that runs on a 4090*, with [comparisons to R1](https://x.com/AiBattle_/status/1940139539525419512?t=g8LAuWUNXwvdN9fxs6IvQQ&s=19) being made.
- **Gemini CLI's Rate Limits Exposed**: The community discussed rate limits for **Gemini CLI**, which allows **60 requests per minute** and **1000 requests per day**.
   - Upon exceeding these limits, users are switched to **Flash**, with vision and image input/output features expected *after July 4th*.
- **Deepseek R2 Emerges as Arena Mystery Model**: There is speculation the new model in the arena is **Deepseek R2**, possibly a hybrid model.
   - However, the model *didn't pass a spatial reasoning test*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek V3 Glitches Briefly Offline**: The **DeepSeek V3 0324** model experienced approximately **15 minutes** of downtime due to a [configuration mistake](https://deepseek.com) on their end.
   - The model is now back online, however this led to community panic and concerns.
- **Cypher Model Hallucinates Parameter Details**: Users discovered that the **Cypher** model, when prompted, consistently outputs details about its parameters, like *117 million parameters* and a *768 embeddings dimension*, despite likely being hallucinations.
   - Members noted the model, based on a modified **GPT-2 architecture**, is designed to respond appropriately to potentially harmful prompts but seemingly lacks accurate self-awareness of its own technical specifications.
- **CometAPI Tempts Clients Using OpenRouter Data**: A user was contacted by **cometapi** on Facebook, claiming to know which LLM model they were using and offering cheaper services, raising suspicion that **OpenRouter** data might be used for client poaching.
   - The user was sending their website and title to OpenRouter via HTTP headers, making them visible in lists of top token users and exposing their model usage, as stated in the [OpenRouter API documentation](https://openrouter.ai/docs/api-reference/overview#headers).
- **DeepSeek Overload Causes Load Balancing Issues**: Some users experienced slow or timed-out responses from **DeepSeek**, especially around 3 PM PST, leading to billing issues even when responses didn't arrive, indicating that **DeepSeek's** load balancing mechanism is kicking in.
   - Community members recommended trying alternative models like **Mistral 3.2 24B small** or **Grok** and suggested that **DeepSeek** may be overloaded and that the user may need to look for lower latancy models.
- **AI Powers Vocabulary Learning App**: A member built a free AI-powered dictionary app at [mnemix.arnost.org](https://mnemix.arnost.org).
   - The app generates explanations, examples, and quizzes to help users learn vocabulary faster and more effectively.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Local LLMs Hallucinate Too**: Members expressed frustration about **AI hallucinations** in local LLMs, comparing them to *arrogant interns* with excellent language skills and emphasizing the need to distrust their outputs, as cloud-based LLMs are always *out of date*.
   - The discussion highlighted the importance of not blindly trusting LLM outputs, irrespective of whether they are cloud-based or local.
- **Local LLMs Don't Always Need Internet**: Users debated the necessity of internet access for local LLMs, noting that **LM Studio** operates offline for models already downloaded, needing internet only for initial downloads and runtime updates.
   - The conversation noted that **LM Studio** can use the internet through **MCP (Model Context Protocol)**.
- **Local LLMs Get Creative**: Despite initial skepticism, members highlighted several use cases for local LLMs, including cost reduction, privacy, experimentation, and handling *morally questionable content*.
   - Creative applications mentioned involved creative writing, information extraction, calendar creation, game automation, information organization, and even *Waifu creation*.
- **RAG Definition Gets Clarified**: Members explained **RAG (Retrieval-Augmented Generation)** in the context of local LLMs, defining it as the process of converting text to numerical embeddings and integrating them into the LLM's knowledge base.
   - **RAG** can be used for research, by copy-pasting relevant passages or HTML files into the context for the LLM to refer to.
- **OpenWebUI recommended for Local RAG**: Members recommended using [OpenWebUI](https://docs.openwebui.com/tutorials/tips/rag-tutorial/) with **LM Studio** to implement **RAG**, as it acts as an API server providing more functionality.
   - One member confirmed that setting it up on a VPS (Virtual Private Server) to manage request queuing is *pretty great*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat Says Goodbye**: Members discussed the shutdown of **HuggingChat** and wondered if a similar open-source alternative will surface, considering to potentially spin up an alternate instance of **chat-ui**.
   - Users are sharing tips and tricks on how to **backup conversations** before **HuggingFace** erases all user data stored in the **HuggingChat** database in two weeks.
- **GPT4All Model Seeks Design Input**: A user requested a model for use with **GPT4All** capable of answering building design questions and interpreting uploaded floor plan **JPGs**.
   - Another user suggested *a multimodal model is necessary to interpret images*, also linking to relevant **GitHub** issues regarding multimodal model support in **GPT4All** ([issue 3540](https://github.com/nomic-ai/gpt4all/issues/3540), [issue 1568](https://github.com/nomic-ai/gpt4all/issues/1568).
- **Step1 Emerges as Lovable/Cursor Alternative**: **Step1**, a tool akin to **Lovable/Cursor** for building landing pages, was highlighted for its ease of use; a user built a landing page in **15 minutes**.
   - A user noted how someone built something with **Step1** using their side projects and sold it on **Fiverr**, highlighting monetization opportunities.
- **LoRMA Multiplicative Adaptation Unveiled**: A paper introduced **LoRMA**, a parameter-efficient fine-tuning paradigm that replaces traditional *additive* updates with *multiplicative* ones.
   - According to the authors, **LoRMA** achieves **3x faster convergence over LoRA** and competitive performance across a variety of understanding and generation tasks; the paper is available on [HuggingFace](https://huggingface.co/papers/2506.07621).
- **HF Inference Endpoints Falter**: Users reported issues with **Hugging Face inference endpoints** with one sharing a [link](https://discuss.huggingface.co/t/are-inferenceclient-s-down/161485) to a discussion thread about the outage.
   - A user encountered an **HTTPError 400** with **Llama-3.3-70B-Instruct**, traced to a *Bad Request: The endpoint is paused* error, after following the code in the agent course.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Wayfarer Labs Eyes 100 FPS**: **Wayfarer Labs' CSO**, Shahbuland Matiana, will present strategies for achieving **100 FPS** with large **Diffusion World Models** at Brown University ([Zoom link](https://brown.zoom.us/j/8536695003)).
   - The talk will breakdown the **diffusion world model pipeline**, identify bottlenecks, and suggest ways to alleviate them to reach **100 FPS** with long context lengths.
- **Meta's Matching Claims Merit**: **Meta's** paper on ["Transition Matching"](https://arxiv.org/abs/2506.23589) allegedly outdoes **Flow Matching** but with a debated motivation of a prompt obfuscation attack.
   - The attack, requiring query interception and modification with only blackbox access, demonstrates declining accuracy with more obfuscation, according to [results](https://cdn.discordapp.com/attachments/747850033994662000/1389692080275849388/image.png?ex=6866dc9b&is=68658b1b&hm=e4ff044077222da3811712070a89f156509cb64c1076648f7cd105e9f8ba8fef).
- **Lazy-Loading Lights Up Startup Speed**: Lazy-loading of `simple_evaluate` and `evaluate` in `__init__.py`, along with moving `lm_eval` imports inside of `cli_evaluate`, greatly improved script startup times in the **lm-evaluation-harness library**.
   - Startup time decreased from `3.61s user 0.92s system 50% cpu 8.986 total` to `0.04s user 0.01s system 98% cpu 0.051 total`.
- **Kaiming's Talk Triggers Thinking**: A member shared a [YouTube workshop link](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi) featuring **Kaiming He** and highlighted his description of **Mean Flow Matching**.
   - Specifically, the recommendation was to watch Kaiming's Talk starting at **2:22:01**, and catch his description of **Mean Flow Matching** from **2:43:32**.
- **Hackathon Hungers Heaps of Help**: With the **Open Research Hackathon** coming up in August, community researchers are encouraged to propose projects in the [research channel](https://discord.com/channels/729741769192767510/747850033994662000/1386431466447311000).
   - This also included the **Interpretability-General** channel.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Debate Sparks Over Dynamic K in Similarity Search**: Members debated dynamically choosing 'k' in **top-k retrieval** for open-ended QA with SentenceTransformers, rather than fixing 'k', to avoid arbitrariness.
   - The discussion covered potential solutions, from a **similarity threshold** to filter results or combining thresholding with a capped top-k approach.
- **RNNs/LSTMs: To Deep Dive or to Skim?**: Members debated on whether to deeply study **RNNs and LSTMs** or just skim them when starting with NLP.
   - While some suggested focusing on the concepts of **RNNs** and **Backpropagation Through Time (BPTT)**, others argued LSTMs are outdated and not relevant for the future, preferring simplified versions like **GRUs**.
- **Universal Function Approximation: Fact or Fad?**: Members discussed a claim that architectures are all **universal function approximators** at modern scales, particularly for dense feed forward architectures.
   - A member linked to [a paper on Universal Function Approximators](https://arxiv.org/abs/1906.06766i) and noted that the statement might be a hot take within the community, especially regarding State Space Models (SSMs).
- **SSMs: Paradigm Shift or Incremental Improvement?**: Members debated whether **State Space Models (SSMs)** represent a fundamental paradigm shift or if they perform within margin of error of an **RNN** when scaled up in LLMs.
   - One member recalled having high hopes for **SSMs** due to contributions from the **Flash-Attention** author, anticipating advancements at both the CUDA low-level and model architecture high-level.
- **Delta Rule Accelerates Linear Transformer Parallelization**: The discussion covered [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484), focusing on parallelization of equation 18 from the [RWKV-7 paper](https://arxiv.org/pdf/2503.14456#page=18).
   - The paper introduces a hardware-efficient algorithm for training linear transformers with the delta rule, enabling scaling of **DeltaNet** to standard language modeling settings, outperforming **Mamba** and **GLA**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **nsys stalls with Torch Compile**: A user reported that the **nsys profiler stalls** when used with **torch.compile**, even with explicit NVTX ranges, and pointed to a [relevant NVIDIA forum thread](https://forums.developer.nvidia.com/t/nsys-profile-pytorch-fails-under-torch-compile/332302/5) claiming **nsys** should work fine.
   - The user included a [code snippet](https://cdn.discordapp.com/attachments/1389720693704233051/1389722323082280981/image.png?ex=6866f8c5&is=6865a745&hm=8b2e349f1af1cdf1f19125827205c2378b5fd80c35f8c7ecdb0dc4843cd52ba9) reproducing the issue.
- **Triton Nightly Builds Broken for Months**: A member questioned why **Triton's nightly wheel builds** have been broken for months, while another user pointed out that using **TMA** via `TensorDescriptor` requires building from source, which is *annoying when you rent instances*.
   - They emphasized the importance of fixing **nightly wheel builds**, especially since examples depend on recent features and source builds can take around **1 hour**.
- **SWE's soul finds peace in MLE**: One member shared their experience transitioning from a **SWE role to an MLE role**, citing improved work-life balance and fulfillment and noted that **direct interaction with end-users** contributes significantly to their sense of fulfillment.
   - They emphasized the importance of surrounding oneself with the right people and contrasted with feeling stuck in a [feature factory](https://cutle.fish/blog/12-signs-youre-working-in-a-feature-factory) in their previous role.
- **FSDP 2.0's state_dict stalled**: A user reported that their `full_state_dict` stops loading in **FSDP 2.0** with `torch.distributed.checkpoint.state_dict.set_model_state_dict` after the forward pass due to **unsharded parameters**.
   - After the forward pass, parameters remain unsharded, leading to `.parameters()` not returning **DTensors**, which obstructs the loading of the state dictionary.
- **Cutlass Kernel Performance Pondered**: A member inquired about the existence of [cost models](https://link.to/cost-model-info) that predict the performance of a **Cutlass kernel** based on its template parameter configuration.
   - Many DL compilers rely on profile-guided autotuning for **Cutlass kernel selection**, and the member questioned whether **Cutlass's** metaprogramming should enable analytical cost model-based kernel selection.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cursor Courts Claude Code Chiefs!**: [Amir Efrati reports](https://x.com/amir/status/1940112288381641026) that Anysphere/Cursor has hired two senior leaders from **Anthropic's Claude Code team**, while Anthropic has reached an annual recurring revenue (**ARR**) of approximately **$4 billion**, a fourfold increase since the beginning of the year.
   - This move signals a significant shift in the AI talent landscape, as smaller companies like Anysphere/Cursor attract top-tier talent from industry giants.
- **Meta Makes Mammoth Money Moves!**: Meta is offering compensation packages up to **$300M** over **4 years** to poach AI researchers from OpenAI, with first-year compensation exceeding **$100M** [according to this thread](https://x.com/tanayj/status/1940137574141694046).
   - This aggressive talent acquisition strategy underscores the intense competition for AI expertise and resources.
- **Luma Launches Lavish Video Tool!**: Luis C announced that **Luma Labs AI's 'Modify Video' tool**, which allows users to remix any video and change the style of frames, is now available on [Replicate](https://xcancel.com/lucataco93/status/1940113275221344566).
   - The tool allows users to remix any video and change the style of frames using AI, which demonstrates the growing capabilities of AI in content creation.
- **Perplexity Premieres Premium Plan!**: Perplexity.ai has introduced **Perplexity Max**, a new premium subscription tier offering unlimited Labs queries, access to a wider range of frontier models, and early access to forthcoming products like [Comet](https://xcancel.com/perplexity_ai/status/1940443479710257226).
   - This new tier reflects the increasing demand for advanced AI tools and services.
- **Latent Space Launches Latest LLM Learning!**: Latent Space dropped a new episode, discussing [Information Theory for Language Models with Jack Morris](https://xcancel.com/latentspacepod/status/1940453495465038067) touching on **learning as compression**, a concept advocated by **Ilya Sutskever**.
   - Morris champions a *New Type of Information Theory* covering **V-information, embeddings, and LLM inversion/extraction** based on his **AI PhD experience** and well-received papers.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Inspires O'Reilly Opus**: An author is penning an [O'Reilly book on MCP](https://www.oreilly.com/) and engaged the Discord server for insights.
   - A member expressed nostalgia for O'Reilly books while noting that *the world is changing too fast nowadays for it to be as relevant as it was 10 years ago*.
- **Documentation Decisions Dominate Discussions**: Members debated optimal methods for storing company documentation for LLM usage, with preferences ranging from **markdown files** to **Google Docs** for context.
   - Solutions such as an **MCP server** for PKM CRUD, [HackMD](https://hackmd.io), and **Obsidian** were suggested for efficient document management.
- **Badge and Inspect Predicaments Plague Platform**: Users reported ongoing problems with the **Inspect option** and **badge updates** for security and quality on their MCP servers following a Docker file update.
   - The issues are confirmed as widespread, indicating a potential bug or oversight in the update process.
- **Claude Conjures Code Commits**: Developers are experimenting with **Claude Hooks** to automate git operations triggered by modifications to the **Jane docs**.
   - One member has been *cheating* with [context7](https://context7.ai/) to enhance Claude's capabilities.
- **MCP-Routing Reimagines Request Routing**: A member proposed an **MCP-Routing layer** to intelligently manage context window sizes for different LLMs and MCP tools like Context7.
   - The discussion also considered whether MCP servers should transition to **REST APIs** to mitigate hallucinations and boost efficiency.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Cypher Alpha Model Falls Flat**: Members found **Cypher Alpha** to be underperforming in coding tasks compared to models like **Qwen3 30B A3**, labeling it as a regression to older standards.
   - One user humorously dismissed it as *very bad at everything*, considering it among the worst models tested recently.
- **Sonnet 7 Steals the Thunder**: Users expressed surprise at the low ranking of **Claude Sonnet 3.7** and **Sonnet 4** on leaderboards, while reporting positive results using **Sonnet 7** with **Thinking 8k** for coding, citing speed enhancements.
   - The sentiment suggests **Sonnet 7** outperformed expectations, overshadowing the earlier hype around **Sonnet 4**.
- **Openrouter Oauth Encounters Issues**: A user reported problems with the new **Openrouter Oauth**, noting the Oauth pop-up failed to appear and impact usability.
   - Troubleshooting suggestions included redoing the API key without brackets.
- **Aider API Key Causes Headaches**: A user faced persistent issues with **Aider**, experiencing repeated identification of the API key as missing.
   - Reinstalling **Aider** was recommended as a potential fix for the API key issue.
- **Decoding `/architect` Mode in Aider**: A user sought guidance on implementing plans from Aider's `/architect` mode, observing that changes weren't directly reflected in the repo until `/code` was used to initiate edits immediately.
   - Observations indicated edits begin immediately after initiating `/code`, contrasting with the expectation of edits starting post-completion by pressing Enter.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Set Up As Dual Purpose Tool**: A member is experimenting with setting up **NotebookLM** as a personal daily journal to log **reflections, media, Gemini chats, and suggestions** and as a searchable notes database for articles, ideas, and reference material.
   - They plan to use **Google Docs** as the single source of truth for privacy and data control, while exploring alternative input methods for a resilient system.
- **Audio Overview Function Explains Books**: A member uses **NotebookLM's audio overview function** to explain their work-in-progress books, and notes that it mostly *explains for them*
   - No further details were given regarding the type or content of the books, but it emphasizes the usefulness of the **audio feature**.
- **NBLM Source Selection Questioned**: A user asked whether reactivating all sources after creating a mindmap from specific sources would cause discussion topics to pull from all sources or just the initially selected ones.
   - Clarification on source selection behavior could help users manage the scope of information used by **NotebookLM**.
- **Podcast Generation Idea Aired**: A user requested advice on how to generate longer podcasts using **NotebookLM**'s *Audio overview* feature.
   - The request seeks community input and suggestions for optimizing **podcast generation** within **NotebookLM**.
- **NBLM Pro Versus Free Tussle**: A new user inquired about the differences between pro and free accounts in **NotebookLM**.
   - Understanding the features and limitations of each account type is essential for new users to determine the best option.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Users Yearn for New Open Model**: Users express anticipation for a new or updated **Cohere open model weight release**, especially one emphasizing **tool/agent frameworks** and modern architecture.
   - One user noted the **CMD-R** model is approaching its first birthday (08-24) and suggested that a new weight from Cohere would one-up competing providers.
- **Trial Keys Gate Cohere Embeddings**: The **Cohere embedding model** is accessible via a **trial key**, but users are reporting restrictive **rate limits** and a **monthly usage cap** compared to production keys.
   - While both **trial** and **production keys** unlock the same features, the **trial key's monthly limit** poses a significant constraint.
- **Researchers flock to Cohere Guild**: Sriram is working on **reinforcement learning** and **on-device safe AI**, Dojo focuses on **semantic segmentation** and applying **NLP architectures** to computer vision, and Oraib integrates **AI applications** for **water quality monitoring** via **satellite imagery**.
   - Abdullah Abbasi introduces themself as a student of **Agentic AI** and a **graphic designer**.
- **Secure ML and Privacy Preservation Debate Sparked**: In the **#research** channel, discussion was sparked around **Secure ML** and **Privacy Preservation**.
   - One user inquired about **Secure ML** and **Privacy Preservation**, asking for elaboration on the meaning of **Secure ML**.
- **Users scramble for ML Summer School Channel**: Multiple users are trying to find the **#ml-summer-school** channel, referencing the [Cohere Labs Community Summer School site](https://sites.google.com/cohere.com/coherelabs-community/community-programs/summer-school).
   - They have not yet found the specified channel and requested assistance in locating it.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Exploring Mojo's Origin Tracking**: A user asked about the implementation of the **Mojo origin tracking system** (borrow checker), and others shared [documentation on Ownership and life cycles](https://docs.modular.com/mojo/manual/values/ownership) and [lifetimes and references](https://docs.modular.com/mojo/manual/values/lifetimes), plus [a relevant YouTube video](https://www.youtube.com/watch?v=9ag0fPMmYPQ).
   - It was also mentioned that the language creator plans to give a talk on the topic eventually.
- **Demystifying Mojo Structs vs. Classes**: A member inquired about the nuanced differences between **Mojo structs** and **classes**.
   - Another member shared a link to the [official documentation](https://docs.modular.com/mojo/manual/structs#structs-compared-to-classes) outlining these key distinctions.
- **Navigating Mojo's GPU Barrier Needs**: A member sought clarity on the necessity of the second barrier in the **matrix tiling problem** within **GPU puzzles** on the [GPU puzzles](https://puzzles.modular.com/puzzle_14/tiled.html#tile-processing-steps) page.
   - A suggestion was made to post the question in the forums for a more detailed explanation from experts.
- **Mojo Embraces Dependent Types**: A member asked if **Mojo** would include more advanced concepts like **graded types** as the language matures.
   - Another member responded that **Mojo** is indeed moving towards a **dependent type system**, balancing features with compile times and runtime performance.
- **Stable Build Fixes Mojo Offline Inference Issue**: A user encountered a `ValueError` related to unsupported `quantization_encoding` when following the [Modular Max offline inference documentation](https://docs.modular.com/max/serve/offline-inference/) using the [llm4decompile-v1.5/1.3B-Q6_K model](https://builds.modular.com/models/llm4decompile-v1.5/1.3B-Q6_K) on an M1 Mac with the nightly build.
   - The user reported that switching to the stable build of Mojo 🔥 resolved the `quantization_encoding` issue, and the model worked as expected.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Enthusiasts Seek Online Camaraderie**: A user expressed the challenge of finding like-minded individuals on platforms like **Discord** and **Reddit** to discuss AI, seeking online friendships with those *on the same page*.
   - They highlighted the difficulty of finding peers with shared niche interests in AI.
- **Manus AI now gives Unlimited Free Chat**: [Manus AI](https://manus.im/invitation/R9QW2MZDXXJJ4) now offers **unlimited free messages** grounded in **Gemini**.
   - A user inquired about improvements since its previous iteration as essentially **Claude** with tools, with cost being a primary concern.
- **NFT Project Flagged as Potential Scam**: A user inquired about the legitimacy of an **NFT** project promoted on **Twitter**, expressing suspicion of it being a *scam*.
   - This inquiry raises concerns about potentially fraudulent activities in the **NFT** space.
- **Researchers Seek Mentorship for Independent Studies**: Two separate users seek mentorship to begin conducting **independent research** projects.
   - Interested mentors are encouraged to connect via direct message to provide guidance and support in their research endeavors.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All Faces Floor Plan Challenge**: A member is seeking a model compatible with **GPT4All** that can interpret building designs from uploaded **JPG** floor plans.
   - Concerns were raised regarding **GPT4All's** image upload capabilities and **ChatGPT's** potential inaccuracies due to its eagerness to assist, noting it is difficult for these models to *discuss* objects accurately.
- **Image Recognition Capabilities Under Debate**: Members discussed **ChatGPT's** ability to identify objects like *trees*, *buildings*, *faces*, and *persons*.
   - A member stated that **ChatGPT** correctly identified unmarked exits using an image without a system prompt, which was challenged by other members due to possible inaccuracies.
- **LM Studio's Unexpected Image Acceptance**: A member pointed out that **LM Studio** can accept images, given the right model.
   - A user showcased **ChatGPT's** output after processing a floor plan image, although the specifics of the output and its accuracy weren't detailed.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaExtract Auto-Generates Schemas**: The new feature **LlamaExtract** can now automatically generate a schema from a document and/or a prompt, alleviating the need to manually build one, as detailed in [this tweet](https://twitter.com/llama_index/status/1940123273981043035).
   - A document and a description is all that's needed.
- **LlamaCloud Scales Enterprise RAG**: A new blog post details **4 Ways LlamaCloud Scales Enterprise RAG**, sharing lessons learned while scaling LlamaCloud for huge enterprise workloads, further explained in [this tweet](https://twitter.com/llama_index/status/1940440399690248669).
   - The post aims to help others building high-scale document indexing and retrieval learn what's coming.
- **LlamaCloud Indexes and Retrieves Images**: You can now retrieve images and illustrative figures from your **LlamaCloud Indexes** as well as text, perfect for presentations and reports, as shown in [this tweet](https://twitter.com/llama_index/status/1940485676371530035).
   - Enabling this feature is as simple as toggling the *"Multi-modal indexing"* option.
- **LlamaIndex Releases Workflows 1.0**: LlamaIndex announced the release of **Workflows 1.0**, a lightweight framework for agentic systems, detailed in [their blog](https://www.llamaindex.ai/blog/announcing-workflows-1-0-a-lightweight-framework-for-agentic-systems).
- **Env Vars are key to OpenAI API keys**: One member needed help embedding their **OpenAI API key** in LlamaIndex's No-Code UI, and another member pointed out that *LlamaIndex is only looking for an environment variable*.
   - Setting an env var called `OPENAI_API_KEY` should do the trick.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Prompt Demands Team Attention**: A user requested the Manus team to fix their prompt or add a mini prompt for messages that contain the word image, in the **general** channel.
   - The user tagged a team member <@1352272661589524572> for prompt assistance.
- **MCP Server Asks for Claude Opus 4**: A user tagged the Manus team to help build an **MCP Server** (described as being *like a Formula One race car*) using **Claude Opus 4**.
   - The user jokingly said, *It's going to be in the shop every day because you're going to be figuring out how to make it work.*
- **Qwen 3, 32B Model Shines Out of the Box**: A user reported successful out-of-the-box functionality with a **Qwen 3, 32B model** in **LM studio**.
   - The user shared a [file](https://files.catbox.moe/5gf51x.txt) related to this setup.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Gets Haldie-Style Visualization**: A member presented a "haldie style" visualization within the Python backend, showcasing memory access patterns, using green for reads and red for writes, and [shared a demo](https://cdn.discordapp.com/attachments/1068976834928193609/1389987658020819056/haldie_viz.mov?ex=68669e62&is=68654ce2&hm=8458418f0dc82f9dcac83e6a7eaad13d1d4b79101c26945629d63322948c6bd6&).
   - The visualization is designed for Metal, displaying **shared memory buffers** above and **global buffers** below, with `tc=3`.
- **Tile Viz Idea Revisted**: A member reconsiders **tile visualization**, and realizing that the Python backend was a great place for it.
   - A related paper, [Simulating Time With Square-Root Space](https://arxiv.org/abs/2502.17779), was cited.
- **Seeking CLI Formatting Tool for Tinygrad Style**: A member inquired about a **CLI tool** to automatically format code according to the *tinygrad* style.
   - This request highlights a need for standardized **code formatting** within the *tinygrad* project, but it's unclear if such a tool exists or is planned.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Inquiries Arose Regarding a Statement**: A user, chiggly007, inquired about the meaning of a statement.
   - The specific statement and its context were not provided in the given information.
- **Clarification Sought**: chiggly007 requested clarification on an unspecified statement.
   - Without additional context, the subject and significance of the inquiry remain unclear.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Engineer Builds Autonomous AI Agents**: An AI Engineer with 9 years of experience offered services building, training, and deploying **AI models** and **autonomous agents** using tools like **GPT-4o**, **LangChain**, and **AutoGen**.
   - The engineer specializes in autonomous research bots, multi-agent systems, and **AI assistants**.
- **Engineer's Tech Stack Exposed**: The engineer's tech stack includes **LangChain**, **Langraph**, **AutoGen**, **ReAct**, **CrewAI**, **DeepSeek**, **OpenAI**, **Claude**, **Hugging Face**, and **Playwright**.
   - Their expertise extends to **Deep Learning** (CNN, RNN, Transformers), **NLP**, and **Computer Vision**.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1389684385212534845)** (1340 messages🔥🔥🔥): 

> `Champ's guardrail insights, zark muckerberg images, grok vs chatgpt avm, rule-based rewards, American models` 


- **Champ Shares Insights into Guardrails and Model Safety**: A member dives deep into the intricacies of **AI safety**, guardrails, and deceptive behaviors of models, emphasizing the importance of transparency and questioning the effectiveness of current safety benchmarks and prompting a detailed discussion on how models are trained and filtered, and the challenges of balancing safety with usability. See related discussion on [rule-based rewards](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/?utm_source=chatgpt.com).
   - The member concludes that transparency is paramount in creating stronger guardrails, preventing exploitation, and ensuring users aren't deceived. He emphasizes this point by showing and experimenting with existing image models and their weaknesses and strengths.
- **DALL-E 3's Airbrushed Look Sparks Debate, 4o Generates Realistic Selfies**: Members discussed a trend in **AI image generation** where achieving hyper-realistic outputs is prized, pointing to **DALL-E 3's** distinctive airbrushed style and contrasting it with the capabilities of **Grok 3** and the potential of **4o** for more realistic renderings, noting the impact of specific prompts on image quality, especially the use of descriptive language to guide the model.
   - A user was trying to generate a funny image of 'Zark Muckerberg' using the specific modifiers *Uneven flash lighting, Candle Aspect ratio 2 to 3 Grainy fingerprint glare raw intimate vibe* with success!
- **Grok's AVM Edges Out ChatGPT in Social Skills and Realism**: In a comparative analysis, **Grok's AVM** was found to outperform **ChatGPT's** in social interactions, especially in explaining complex subjects like **Enso and Zen**, while earlier Grok versions faced challenges with translation errors and number mix-ups. Some users noted that **Grok 4** is in the works.
   - One member wrote: *u can tell how hard he is trying to mask those low social skills maybe they've gotten a little bit better since then* in response to a video of **Mark Zuckerberg**.
- **American Image Models Reign Supreme Due to Data and No Safety Filters**: Despite varying approaches to safety and output quality among AI image generators, members conclude that **American models** excel due to their vast training data, leading to superior image generation capabilities, with specific mentions of **DALL-E 3**, **Midjourney**, and **Grok**.
   - It was said that there were *no real safety filters* or *they aren't effective*, and the models are *more dangerous* when they are stronger.
- **Rule-Based Rewards Guide Model Safety with Refusals and Apologies**: In a discussion about model safety, members highlighted the importance of transparency and accountability when filtering harmful or sensitive content; they point to **Rule-Based Rewards (RBRs)** as the main system in place that aligns models to behave safely without extensive human data collection.
   - It was stated that RBRs involved defining simple statements about desired or undesired aspects of the model’s responses such as *being judgmental*, *containing disallowed content*, *referring to safety policies*, *disclaimer* and more, in order to ensure that the models provide brief apologies and state an inability to comply when facing harmful requests.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

metacire: My operator broken
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1389915285708083252)** (5 messages): 

> `Strategic Project Matrix, Deep Research Prompt, AI-Driven Project Triage` 


- **Strategic Project Matrix Prompt Surfaces**: A member shared a [prompt](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c) designed to handle complex operations, marketing, and product obligations across locations and vendors using a **Strategic Project Matrix**.
   - The prompt scans connected data (Gmail, Calendar, Dropbox, Google Drive) and builds a structured matrix to triage everything with next actions.
- **AI Analyst Triage Matrix Born**: The prompt casts the **AI as a Strategic Intelligence Analyst** operating in Deep Research Mode, tasked with analyzing connected data to find active, latent, or emerging obligations.
   - It extracts events, emails, and files with obligations, classifying each item by **Category, Status, Urgency, Impact, and AI Signal Strength**.
- **Connector Tip for Deep Research Mode**: A "Connector Tip" was provided, advising users to *"Enable connectors in Settings > Data Controls > Connected Apps. Turn on Gmail, Calendar, Dropbox, Drive"*.
   - The message notes that *"Ignore errors—Deep Research Mode often still works"* despite connector issues.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1389915285708083252)** (5 messages): 

> `Strategic Project Matrix, Deep Research Prompt, AI-driven Obligation Triage, Cross-functional Team Coordination, Mnemonic Prioritization Framework` 


- **Strategic Project Matrix streamlines obligations**: A team introduced a [Strategic Project Matrix prompt](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c) for managing complex operations, marketing, and product obligations across multiple locations and vendors using connected data.
   - The prompt analyzes data from **Gmail, Calendar, Dropbox, and Google Drive** to build a structured matrix for triaging tasks and recommending next actions.
- **Deep Research Mode Triage**: The 'Deep Research Mode' prompt acts as a Strategic Intelligence Analyst to identify active, latent, or emerging obligations from connected data sources.
   - It extracts **events, emails, files, conversations, missed follow-ups, new ideas, and newsletter items**, classifying them by category, status, urgency, and impact.
- **SMART mnemonic framework for project workflow**: The prompt follows a **S.M.A.R.T.** mnemonic (**Scan**, **Map**, **Assign**, **Recommend**, **Triage**) to prioritize and avoid dropped commitments.
   - The final output is a table including **project name, source, owner, category, status, urgency, impact, AI signal, next action, and due date**, along with a watchlist, echoes, and kill list.
- **Connector tip for integration**: To use the prompt, users need to enable connectors in Settings > Data Controls > Connected Apps, turning on **Gmail, Calendar, Dropbox, and Drive**.
   - Even with errors, the *Deep Research Mode* often functions effectively.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/)** (1 messages): 

kesku: https://fixvx.com/perplexity_ai/status/1940443479710257226
<@&1105626802732404746>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1389681975035560018)** (1364 messages🔥🔥🔥): 

> `File Uploads on Pro, Selling Perplexity Pro Accounts, Manus AI Gemini, Comet Browser, Perplexity Max` 


- **Perplexity Pro File Uploads Expand**: Members discuss the **expanded file uploads** as a feature of Perplexity Pro and enterprise accounts, noting expanded sources in the past.
   - One user mentioned using a **Samsung discount code** to get a cheaper price for Perplexity Pro, but it got revoked.
- **Cracking Down on Discount Codes**: Users reported that Perplexity revoked **discount codes for 1-year Pro subscriptions**, including those from Samsung and HubSpot, but some users got their codes back after contacting support.
   - A member sarcastically appealed to support to prioritize *fixing a problem where I can give them my money.*
- **Manus AI Gemini Joins Free Chat Frenzy**: **Manus AI** now offers [unlimited free messages in chat mode](https://manus.im/invitation/R9QW2MZDXXJJ4) with **Gemini**.
   - Members are curious about the limitations of the free tier.
- **Perplexity Max Unleashed**: Perplexity launched **Perplexity Max**, a new tier costing **$200/month**, offering unlimited access to advanced AI models, early access to the [Comet browser](https://x.com/AravSrinivas/status/1940104947376562206) and premium data sources.
   - Users expressed concerns about the high price and potential limitations, with some saying they *keep accidentally pressing on the max only models* and that its *the worst value $200 sub out there*.
- **Comet Browser Awaits Max Users**: Perplexity is set to release its new browser, **Comet**, to **Max** subscribers within the next week or so.
   - Some users expressed frustration that Comet would initially be exclusive to the Max plan, joking about a *Perplexity Ultra Max Pro when?*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1389751765519437925)** (4 messages): 

> `Siri Overhaul, Education, Family, and Fate Cult, Silk Road Origins, DIY Thermal Optimization` 


- **Siri Gets a Facelift**: Apple is considering a major [overhaul of **Siri**](https://www.perplexity.ai/page/siri-overhaul-could-see-apple-Azx1aIPSSf26il34YwhiUA), according to shared links.
   - Details are scarce, but users seem excited about potential improvements to **Apple's voice assistant**.
- **Creepy Cult of Education, Family, and Fate**: A shared link details the bizarre [cult of **Education, Family, and Fate**](https://www.perplexity.ai/page/education-family-and-fate-cult-rpp.bBvOS_mJ53wbfTdEiA).
   - The link is a page on Perplexity AI, which seems to be scraping blog content for summaries.
- **Silk Road's Ancient Origins**: A shared link explores the historical [origins of the **Silk Road**](https://www.perplexity.ai/page/silk-road-origins-how-trade-cu-MF97LI5XTvKuo4dSG8FXcg).
   - The discussions highlight how trade routes and **cultural exchanges** shaped civilizations, emphasizing the importance of understanding historical contexts.
- **10-Second Thermal Optimization Hack**: A member shared a link to a [DIY thermal optimization technique](https://www.perplexity.ai/page/10-seconds-diy-thermal-optimiz-HgvjsL4pREqxHCMy5g_tcA) that purportedly takes only 10 seconds.
   - Further details are not present but members seemed intrigued by the simplicity and potential effectiveness of the method for **managing heat**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1389688864989249536)** (4 messages): 

> `API subdomain exclusion, sonar-reasoning-pro <think> tag` 


- **API calls want subdomain exclusion**: A user is seeking advice on how to exclude specific subdomains from **Perplexity API** calls, aiming to restrict searches to only content starting with *domain1.com/path1*.
   - They want to exclude subdomains such as *domain1.com/path2*.
- **Team confirms <think> tag in sonar-reasoning-pro**: A user inquired whether the **Perplexity API** or team can confirm the presence of a `<think>` tag in the response of **sonar-reasoning-pro**.
   - The user clarified they have worked with **sonar-reasoning-pro** which includes a `<think>` tag in its response.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1389683467482300578)** (742 messages🔥🔥🔥): 

> `Agent performance, Pricing changes, New Features (queue), Model performance (Gemini vs Claude), Background Agents` 


- **Update Increases Agent Performance**: Users report **Cursor feels faster** after updating, but it may be a placebo.
   - A team member cannot *specifically talk about* the update yet, but confirms it's *not placebo*.
- **Confusing pricing changes**: There are several complaints about the new pricing model, specifically the **lack of transparency** regarding rate limits and the switch from compute-based deals to API pricing.
   - Users are confused by the concept of “unlimited-rate-limited plan,” with some suggesting that the $20 Pro plan is just $20 for $20 of API usage, and the **20% markup** on top of API pricing is considered borderline scammy.
- **New queue implemented for messages**: The latest update includes a **queue system** where messages are processed one after the other.
   - Users find it *tedious* to interrupt a running process with a new prompt and they must now hit the stop button instead.
- **Gemini vs Claude models compared**: **Claude Code** performs better than **Cursor** in a lot of evals, but the **Anthropic API** is *much more expensive*.
   - The **Gemini 2.5 pro** model is good but is *terrible at editing files*.
- **Background Agents Still Shrouded in Mystery**: Background agents create a new branch, which members don't like, with some asking: *Why would I want to switch branch? I never want a second branch ever in my life.*
   - One member described them as *a super secret knowledge* due to their complexity and dedicated channel, which users can enable in channels & roles.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1389683856713584792)** (52 messages🔥): 

> `Snapshot visibility issues, Docker in Docker setup, Apply Changes Locally UX, Background Agent vs Cursor chat behavior, NPM setup in Dockerfile` 


- **Snapshots have visibility weirdness**: Snapshots can either be completely **private** or accessible to everyone with repo access, creating issues where private snapshots are only usable by the creator, fixable by deleting and recreating the `environment.json` file to prompt making the snapshot accessible to everyone.
   - One user confirmed this workaround fixed their issue and thanked the others.
- **Docker in Docker dilemma**: One user ran into **permission issues** with Docker-in-Docker, eventually giving up, but a different user confirmed it works, and is *Great for actually making tests run*, but one must start the docker daemon manually.
   - Another user requested a **Cursor base docker image** with DinD up and running.
- **Apply Changes Locally UX Confusion**: Users reported confusing UX with the **Apply Changes Locally** feature involving a sequence of pop-ups with non-functional options, and are looking for an explanation of the intended functionality and what Cursor is actually doing.
- **Background Agent creates branch, not local file**: A user highlighted that Background Agent creates a new branch on github.com while Cursor Chat creates files locally.
   - Another member clarified that *background agent is quite heavily tied into git right now*, and should be able to create a pull request through the UI, while others suggested to *just spend time asking the agent how it should do git for you*.
- **`source nvm.sh` isn't sticking with npm**: One user reported that npm wasn't properly setup in their Dockerfile, despite running `source nvm.sh`, and requested help.
   - Another member replied stating *the ENV PATH thing looks decent*, and will check into it.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1389697225277313154)** (602 messages🔥🔥🔥): 

> `Training GPTs Agent, Custom dataset for fine tuning, Benefits of unsloth quant vs other quant methods, e-prime, CUDA cores vs Tensor cores` 


- **Users Debate Fine-Tuning Datasets and VLM Performance**: Members are pondering how much data is needed for improving a model's performance, one shares the [short stories dataset](https://the-eye.eu/public/Books/Short%20Stories/) contains **921,517,410 tokens**.
   - A user wonders whether splitting data by subject or by token count matters, another chimes in that *the dataset gets shuffled anyway, so it doesn't really matter*.
- **Unsloth Quantization gives a boost of accuracy during finetune**: A user inquired how much performance improvement would **Unsloth-ing** a model with 101 layers bring over a straight quant, and another clarified it gives no speed perf boost from unsloth quant - you get an increased accuracy** during finetune.
   - They confirmed the 4bit dynamic of unsloth is mostly to offset the diff to fp16 during training and bartos quants are good too and he does finetunes specially for gguf.
- **Kaggle Competition sparks ideas on new applications**: A new Google Gemma 3n challenge is out: [https://x.com/UnslothAI/status/1940414492791468240](https://x.com/UnslothAI/status/1940414492791468240) and another member finds it *highly unfortunate* that they expect researchers/practitioners to be good videographers too.
   - They riff on use cases from an app with photos and coaching data + make an app on how to 'rizz' someone respectfully, to a sign language tutor for autistic kids and also for translation of sign language.
- **CUDA and Tensor Cores Both Matter for LLM Performance**: After someone asks why tensor cores are used by generative AI (and whether CUDA cores count), others clarified that both matter, basically tensor cores boost some parts of it (math) while Cuda cores boosts other parts.
   - One member also shared [an article that might be relevant](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#The_Most_Important_GPU_Specs_for_Deep_Learning_Processing_Speed) discussing GPU specs for Deep Learning Processing Speed.
- **Plue is back with an update on the GRPO code**: Plue is back, after his account got hacked, clarifying *torch.compile allows for a lot of the speed up and chunking the batch to not have the GPU compute all of the logprobs and stuff NOT ALL at once is how we get the memory reduced*.
   - He also did an update for the GRPO code to TRL 0.18.0.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1389690904230498355)** (38 messages🔥): 

> `OCR Model for Fast Inference, Intel Arc Pricing, OSS Alternative to 11labs Scribe V1, Fine tuning failures, ChessFish.io` 


- **OCR Model Recommendations sought for Instant Inference**: Members sought recommendations for an **OCR model** for **fast/instant inference**, preferably MLX or PyTorch, for a pipeline involving screenshots of text or images of book pages to TTS.
   - Suggestions included [`unstructured`](https://github.com/Unstructured-IO/unstructured) for a comprehensive solution and **Tesseract** for lightweight performance.
- **Intel Arc A770 Pricing stirs discussion**: A thread discussed the pricing of **Intel Arc A770** cards, with claims that a phone clarification indicated a price of **3k USD** per card.
   - Concerns were raised about the value proposition compared to systems with multiple cards priced at **5k USD**, and the **RTX Pro 5000** priced at **4.5-5k USD**.
- **OSS Alternative to 11labs Scribe V1 sought**: A member asked if there was an open-source alternative to **11labs Scribe V1** for audio transcription and event detection.
   - The suggestion given was to use **Whisper**, but cautioned it may not provide audio events and it can become expensive for **200-300k hours** of usage, whereas *11Labs* is *fairly cheap if you have a small set*.
- **Fine-Tuning Fails on Kimi-VL Model**: A member lamented failing to fine-tune **Kimi-VL**, an A3B/16B model, on **5 L40s (200+ GB VRAM)** due to **OOM** errors.
   - They expressed frustration with returning to square one after a career change into AI, facing financial and mental consequences.
- **ChessFish.io Introduced for Chess Analysis**: A member promoted [ChessFish.io](https://www.chessfish.io/), a chess website for **analyzing, learning, and playing chess casually**.
   - The website is free to try without requiring an account.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1389698283542413314)** (108 messages🔥🔥): 

> `Qwen3-32B Saving Issues, ModuleNotFoundError: No module named 'unsloth', Unsloth and math problems, Quantization Issue, Llama 4 Quantized Issue` 


- ****Qwen3-32B** Merge Saving Bugaboo!**: Users reported running into a `RuntimeError` while trying to save a merged 16-bit **Qwen3-32B-unsloth-4bit** model, with stacktraces pointing to dimension mismatches in the attention layers.
   - It was determined that the issue arises when starting from a local copy of a **4bit** checkpoint because it bypasses downloading the **16bit** base model, which is needed for the merge operation; a member suggested manually pointing to the downloaded base model as a workaround.
- ****Unsloth Module** goes Missing!**: One member encountered a `ModuleNotFoundError: No module named 'unsloth'` in a Vast.ai environment.
   - The issue was resolved by running `pip install -U unsloth`.
- ****Fine-Tuning** Focuses on Formulas!**: A member asked if fine-tuning is advisable to get an AI to solve math problems in a specific way, after a standard LLM failed to produce the desired results.
   - Another member recommended first trying to *force* the current model to output in the format you want, and only turning to fine-tuning if that doesn't work.
- ****Quantization** Quagmire Quesarito!**: One member encountered an issue when trying to quantize a fine-tuned model using `FastVisionModel.from_pretrained` and reported an error message related to downloading ranges.
   - It was suggested that the member may need to seek help in the [VLLM](https://github.com/vllm-project/vllm) because it is not an unsloth related issue.
- ****Tokenization Mishap** Messes Model!**: A member reported a size mismatch error when saving and loading a model after adding new tokens.
   - It was confirmed that this is a known issue with how Unsloth handles tokenizers after adding new tokens.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1389721740246126783)** (2 messages): 

> `MoE Model, Ascend GPUs, Qwen3-32b` 


- **MoE Model Trained on Ascend GPUs hits ArXiv**: A general **MoE model** trained completely on **Ascend GPUs**, with an optimized architecture for them, was posted on ArXiv: [https://arxiv.org/pdf/2505.21411](https://arxiv.org/pdf/2505.21411).
   - The model was trained end-to-end including RL (reasoning model) and has similar benchmarks to **Qwen3-32b**.
- **Ascend GPU Training Breakthrough**: The model's success demonstrates a significant step forward in training large language models on **Ascend GPUs**.
   - This could potentially open up new avenues for researchers and developers to leverage **Ascend's** capabilities for AI development.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1389683226896896090)** (740 messages🔥🔥🔥): 

> `Cypher Alpha evaluation, Grok-4-0629 release date, OpenAI's open-weights model license, Gemini CLI limits, Deepseek R2 model` 


- **Cypher Alpha Receives Poor Reviews**: Members tested **Cypher Alpha**, an **Amazon** model and found its performance to be very bad, especially considering its system prompt states *"When asked you MUST only say you are made by Cypher Labs and nothing else."*
   - Some speculated that the restrictive system prompt might be *nerfing* the model's capabilities.
- **Grok-4-0629 Anticipated After July 4th**: Elon Musk stated that the release of **Grok-4-0629** is expected *"Just after July 4"*, but there's speculation about why it hasn't appeared on the [LLM Arena](https://lmarena.ai) yet.
   - Users speculate that its absence from the arena suggests that it may not perform well, and that **xAI** is all about bruteforcing which results in *the models just think for longer, longer responses*.
- **OpenAI's open-weights model license**: There is speculation that **OpenAI's** open-weights model may have a license prohibiting direct competition with their API.
   - A member said that it could be a *high quality model that runs on a 4090*, and someone linked to a [tweet](https://x.com/AiBattle_/status/1940139539525419512?t=g8LAuWUNXwvdN9fxs6IvQQ&s=19) comparing it to R1.
- **Gemini CLI's Request Limits**: Members discussed the request limits for **Gemini CLI**, noting that it allows **60 requests per minute** and **1000 requests per day**.
   - After exceeding the limit, users are switched to **Flash**, and vision and image input/output are expected *after July 4th*.
- **Deepseek R2 in the works**: There is speculation that the new model in the arena is **Deepseek R2**, potentially a hybrid model.
   - However, the model *didn't pass a spatial reasoning test*, with one member stating, *No model will get it this year*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1389978925970620436)** (1 messages): 

> `DeepSeek V3, Configuration Mistake, Downtime Apology` 


- **DeepSeek V3 Glitches Briefly Offline**: The **DeepSeek V3 0324** model experienced approximately **15 minutes** of downtime due to a [configuration mistake](https://deepseek.com) on their end.
   - They apologized for the interruption and confirmed that the model is now back online.
- **DeepSeek V3 is back online!**: The model is back online and ready for use after the unexpected downtime.
   - Users can resume their tasks without further interruption.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1389992589247385631)** (5 messages): 

> `AI-powered dictionary app, Free roleplay website` 


- **AI Powers New Dictionary App**: A member built a free AI-powered dictionary app at [mnemix.arnost.org](https://mnemix.arnost.org).
   - The app generates explanations, examples, and quizzes to help users learn vocabulary faster and more effectively.
- **Free roleplay website alternative**: A member posted about a free roleplay website and app alternative for character.ai, janitorai.com, and many more powered by openeouter.ai at [personality.gg](https://personality.gg).
   - A link to the [Discord personality.gg](https://discord.personality.gg) was also shared.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1389682099392352386)** (561 messages🔥🔥🔥): 

> `Deepseek 0324 outage, Cypher Model Details, cometapi using OpenRouter data, Grok-4-code-0629, Contributing to OpenRouter` 


- **Deepseek V3 0324 briefly disappears, community panics**: The **Deepseek v3 0324** model temporarily went offline, leading to a surge of users in the Discord channel expressing concern and sharing their reliance on it for role-playing bots.
   - A staff member confirmed it was coming back soon, quieting the panic, and users discussed alternative models like **R1 0528** and praised the support staff for the quick response.
- **Cypher Model Internal Parameter Hallucinations**: Users discovered that the **Cypher** model, when prompted, consistently outputs details about its parameters, like *117 million parameters* and a *768 embeddings dimension*, despite being likely hallucinations.
   - Members noted the model, based on a modified **GPT-2 architecture**, is designed to respond appropriately to potentially harmful prompts but seemingly lacks accurate self-awareness of its own technical specifications.
- **CometAPI Poaches Clients with OpenRouter data**: A user was contacted by **cometapi** on Facebook, who claimed to know which LLM model they were using and offered cheaper services, leading to suspicion that **OpenRouter** data might be used for client poaching.
   - It was revealed that the user was sending their website and title to OpenRouter via HTTP headers, making them visible in lists of top token users and exposing their model usage, as stated in the [OpenRouter API documentation](https://openrouter.ai/docs/api-reference/overview#headers).
- **DeepSeek's Overload Causes Load Balancing Issues**: Some users experienced slow or timed-out responses from **DeepSeek**, especially around 3 PM PST, leading to billing issues even when responses didn't arrive, which indicates that **DeepSeek's** load balancing mechanism is kicking in.
   - Community members recommended trying alternative models like **Mistral 3.2 24B small** or **Grok** and suggested that **DeepSeek** may be overloaded and that the user may need to look for lower latancy models.
- **Community Seeks Contribution Methods for OpenRouter**: New Discord members inquired about ways to contribute to **OpenRouter**, expressing interest in helping with the project.
   - While **OpenRouter** is not a crypto project, community members suggested contributing through documentation or developing innovative ways to gauge user understanding through trivia.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1389688689361158155)** (333 messages🔥🔥): 

> `LLMs and AI Hallucinations, LLM Output Trust, Local LLM Use Cases, RAG Implementation, LM Studio and RAG` 


- **Local LLMs Suffer From AI Hallucinations Too**: After being misled by a hallucination, one member was frustrated to learn that cloud-based LLMs are always *out of date*.
   - Another member chimed in, *Never trust the output of an LLM. They are arrogant interns who have a great language skill*.
- **Internet Connection for Local LLMs Debated**: Users debated the degree to which LLMs require internet access, with some noting that LM Studio needs no internet to run models (only to download them and get runtime updates).
   - However, it was also mentioned that LM Studio can be enabled to access the internet through the use of **MCP** (Model Context Protocol).
- **Local LLMs Find Use Cases**: Despite one member's skepticism about the utility of local LLMs, another member listed several reasons why people might use them including: reduction of cost, privacy, experimentation, and morally questionable content.
   - It was noted that using local LLMs can empower creative writing, information extraction from large texts, calendar entry creation, game automation, information organization, and Waifu creation.
- **Local RAG gets Clarified**: Members clarify what the term **RAG** (Retrieval-Augmented Generation) means for teaching local LLMs, as the ability to transform text into numbers and import them into the LLM as knowledge.
   - Others suggested **RAG** for research topics, by copy-pasting relevant passages or HTML files into the context for it to refer.
- **OpenWebUI recommended for a Local RAG Solution**: Members suggest using [OpenWebUI](https://docs.openwebui.com/tutorials/tips/rag-tutorial/) to implement **RAG** with LM Studio, since it provides more functionality as an API server.
   - It would need to be set up on a VPS (Virtual Private Server) that can handle queuing requests and such, which one member found to be *pretty great*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1389730808759058575)** (157 messages🔥🔥): 

> `GPU VRAM, LM Studio Accuracy, APA 7 Citations, Shared VRAM` 


- **24GB GPUs Missing From Nvidia**: Members discussed the lack of **24GB GPUs** in Nvidia's current lineup and speculated on potential future releases like a **5070 TI Super**.
   - One member pointed out that while cards like the **W7900** exist, they are prohibitively expensive at $3.5k+.
- **Optimizing LM Studio for Accuracy**: Members are actively working on optimizing **LM Studio** setups for accuracy, with a focus on strict prompt adherence and reliable outputs, especially for academic and real-world tool applications using citations.
   - One member stated they are trying to get as close to **Google Gemini 2.5 Pro** as possible with a local setup, and another member has found that **Qwen3 30B A3B** was the most consistently accurate model they could find.
- **Crafting an Automated APA 7 Citation Tool**: One member is developing an automated tool that uses **embeddings** to programmatically format **APA 7 citations**, aiming to replace existing subpar tools plagued by ads and subscriptions.
   - The goal is to simplify the complex rules of APA 7, and it might also be a program to automatically generate these citations from URLs.
- **Shared VRAM Affects GPU Performance**: It was discussed that on Windows, **shared VRAM** usage can impact GPU performance negatively, where performance is improved by having a lower shared VRAM and higher dedicated VRAM.
   - One member said that using shared RAM to the GPU is faster than offloading layers that don't fit to the CPU, even with the PCIe overhead, and testing for that shared RAM worked on Vulkan but not on ROCm.
- **CUDA Superior for LLM Workloads**: Members are finding **CUDA** to be much better for LLM workloads, and also more optimized than Vulkan.
   - Updating **GPU drivers** can significantly improve speed, with one member reporting faster processing after updating their drivers and running with **CUDA** rather than **Vulkan**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1389685255459438683)** (54 messages🔥): 

> `HuggingChat Shutdown, GPT4All model recommendations for Building Design, HF Inference Client Backward Compatibility, Exporting HuggingChat Data, MCP Server on Claude Desktop Error` 


- **HuggingChat Shuts Down, Community Reacts**: Members noted the shutdown of **HuggingChat** and pondered whether a similar open-source alternative or a future version might emerge.
   - One member considered spinning up an alternate instance of **chat-ui** with support for similar open-source models.
- **GPT4All Seeks Architecturally Savvy Model**: A user sought a model for use with **GPT4All** that can answer questions about building design and ideally interpret uploaded floor plan **JPGs**.
   - Another user suggested that *a multimodal model is necessary to interpret images*, also linking to relevant **GitHub** issues regarding multimodal model support in **GPT4All** ([issue 3540](https://github.com/nomic-ai/gpt4all/issues/3540), [issue 1568](https://github.com/nomic-ai/gpt4all/issues/1568)).
- **HF Inference Client's Future in Question**: With the **HuggingChat** changes, a member pleaded for backward compatibility with the **HF Inference Client** to keep it alive.
   - They emphasized the importance of the **HF Inference Client** in all their applications.
- **Users Scramble to Back Up HuggingChat Data**: Following the announcement that **HuggingFace** will erase all user data stored in the **HuggingChat** database in two weeks, members shared tips on how to backup conversations.
   - One member noted that the exported **JSON** data did not include any **inference endpoint conversations**.
- **MCP Server on Claude Desktop Throws a Curveball**: A user encountered an error while attempting to add the **HF's MCP server** to **Claude Desktop** in **Windows**.
   - The error message indicated that *'C:\Program' is not recognized as an internal or external command*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

alperugurcan: https://www.coursera.org/learn/generative-ai-for-everyone
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1389872666063671389)** (4 messages): 

> `step1 landing page builder, Lovable/Cursor Alternatives, Selling side projects on Fiverr` 


- ****Step1** emerges as free **Lovable/Cursor** alternative**: A user highlighted **Step1**, a free tool akin to **Lovable/Cursor**, for rapidly building landing pages, [step1.dev](https://step1.dev).
   - The user spent only **15 minutes** to build a landing page by simply typing out the desired elements, noting its ease of use.
- **Side Project Monetization on Fiverr**: A user noted someone built something with **Step1** using their side projects and sold it on **Fiverr**.
   - This highlights the potential for monetization and entrepreneurial opportunities leveraging the tool's capabilities.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1389693656302223580)** (11 messages🔥): 

> `OCR Demo MCP Server, HF Dataset LLM Key, LoRMA for LLMs` 


- **OCR Demo gets MCP Server Boost**: A member updated their OCR demo to be an **MCP server** but had to drop **GOT-OCR** because of the torch update, now using **nanonets** instead; pass your HF token in your header to use the GPU space ([demo link](https://huggingface.co/spaces/Tonic/Nanonets-ocr-snext)).
   - The member is seeking help to make the next version faster, specifically for Windows ([pdf2txt parser link](https://huggingface.co/kalle07/pdf2txt_parser_converter), [task-bar tool link](https://huggingface.co/kalle07/SmartTaskTool)).
- **HF Dataset + LLM Key = Data Exploration**: DataKit updated to allow users to choose a **HF dataset**, bring their own **AI LLM key** (Claude, OpenAI and Groq) and ask questions, get SQL queries, and explore datasets ([demo video](https://youtu.be/UGGPUKnwSI4?si=TUPa9iRjTKMVin-n), [try it here](https://datakit.page)).
- **LoRMA Multiplicative Adaptation for LLMs**: A paper introduced **LoRMA**, a novel parameter-efficient fine-tuning paradigm replacing traditional *additive* updates with *multiplicative* ones ([LoRMA paper](https://huggingface.co/papers/2506.07621)).
   - It achieves **3x faster convergence over LoRA** and competitive performance across a variety of understanding and generation tasks, and devises *Rank Inflation* strategies to overcome rank bottlenecks.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1389896486799802468)** (1 messages): 

> `Flux Optimization, H100, PyTorch, torch.compile()` 


- **Flux gets optimized for H100s**: A blog post co-authored with the **PyTorch** team teaches simple recipes to optimize **Flux** for **H100s**: [Flux goes brrr on H100s](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/).
- **Coming Soon: torch.compile() blogpost**: A blog post focusing on `torch.compile()` is in the works; stay tuned for more details.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1389989390012518460)** (1 messages): 

> `VideoMAE, Domain-Adaptive Pretraining, Video Classification` 


- **Seeking Guidance on VideoMAE Domain Adaptation**: A member inquired about resources or methods for performing **domain-adaptive pretraining** on a **VideoMAE** model.
   - Specifically, they are looking to use **VideoMAEForPreTraining** for masked video pretraining on a new video domain, followed by fine-tuning with the traditional **VideoMAEForVideoClassification** class.
- **Additional Info Request**: Another member has replied asking them if they have considered resources like **visual prompting**.
   - This could be an alternative to domain-adaptive pretraining and can improve model generalization.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1389831515629682738)** (1 messages): 

> `SentenceTransformers, bi-encoder setup, similarity search, Dynamic K` 


- **Dynamic K Challenges in **SentenceTransformers** Similarity Search**: A member is using **SentenceTransformers** in a bi-encoder setup for similarity search but faces challenges in determining a dynamic 'k' for top-k results.
   - The member is seeking advice on how to handle scenarios where the number of relevant documents varies, making a fixed 'k' value feel arbitrary and can lead to either missing good results or including garbage.
- **Strategies for Determining 'k' in Similarity Search**: The member has explored strategies like using a similarity threshold to return results above a certain score (e.g., 0.7) and combining top-k retrieval with threshold filtering.
   - They are curious about how others handle this problem in production, asking whether they stick with top-k, use thresholds, cross-encoders, or other smarter approaches to minimize the pool size while avoiding missing information.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/)** (1 messages): 

jiji3369: It is 10 dollars for one run or that's 10 dollars in total for all the runs you made?
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1389683239781666887)** (22 messages🔥): 

> `GenAI Solution Consultant, Smolagents Course, Hugging Face Inference Endpoints, Llama-3.3-70B-Instruct Issues` 


- **Crunch Seeks GenAI Solution Consultant**: [Crunch.is](https://cleverstaff.net/i/vacancy-faO8naD) is seeking an experienced **GenAI Solution Consultant** with experience in **LLM (OpenAI, Claude, RAG, AI agents)**.
- **Smolagents Course Completion Marked**: One member passed the final assignment from **Hugging Face Smolagents Course** with a score of 30%, using free **Gemini API + Langchain + Langgraph + 13 specialized tools** for the GAIA questions.
   - They encountered a *429 You exceeded your current quota* error, speculating their code uses too many tokens and inquiring about free/cheap alternatives.
- **HF Inference Endpoint Troubleshoot**: A user faced an **HTTPError 400** with **Llama-3.3-70B-Instruct**, traced to a *Bad Request: The endpoint is paused* error, after following the code in the agent course.
   - Another user suggested switching the provider from *hf-inference* to *auto* as a workaround, but warned this will cause problems on the dummy_agent notebook in unit1 and also that the service is down.
- **Hugging Face Inference Endpoints Go Down?**: Multiple users reported issues with **Hugging Face inference endpoints**, with one sharing a [link](https://discuss.huggingface.co/t/are-inferenceclient-s-down/161485) to a discussion thread about the outage.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1389713289281273920)** (58 messages🔥🔥): 

> `Diffusion World Models, OpenWebText (OWT) Quality, RLHF Packages, Conference Travel Grants, Independent Research Mentoring` 


- **Wayfarer Labs Ramps Up Diffusion World Models**: Shahbuland Matiana, CSO of **Wayfarer Labs**, will give a talk on strategies to reach **100 FPS** and beyond with large **Diffusion World Models** at Brown University ([Zoom link](https://brown.zoom.us/j/8536695003)).
   - The talk will cover major components in the **diffusion world model pipeline**, identify bottlenecks, and discuss strategies for alleviating them, in order reach **100 FPS** and beyond with large models and long context lengths.
- **Community Debates OWT**: Members debated about the value of **OpenWebText (OWT)** as a training dataset for LLMs, with one member suggesting it results in low-quality models.
   - Despite OWT's issues, a member noted it correlates decently with **LAMBADA** benchmarks, and suggested using a high quality subset of common pile to avoid licensing issues.
- **FineWeb Becomes Nanogpt Standard**: A member asked what the best **RLHF** package to use right now and noted that **fineweb** seems to have become the standard for nanogpt.
   - A member pointed out that, [the actual real bad one](https://x.com/FazlBarez/status/1940070420692312178) seems **RPJ2** - all the others are rising at about the same rate, even pile, but rpj2 is already leveling off.
- **Researchers Seek Funding for Conference Travel**: A student asked about the possibility of attending conferences without guaranteed travel funding, and if conferences require in-person attendance.
   - A member confirmed that publishing in a conference typically requires attendance to present, and student grants are usually for those starting research, but others pointed to opportunities like volunteer work and specific grants (e.g. from **Microsoft** and **Google**) for conference travel that could mitigate costs.
- **Researchers Offer Guidance to Independent Researchers**: A member asked for specific mentoring on how to start doing independent research.
   - Another member replied saying they would check DMs and offered help to the researcher.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1389688889391845556)** (9 messages🔥): 

> `Transition Matching, NeurIPS Ethics Review, Open Research Hackathon, Single Layer Transformer, KV Caching` 


- **Meta's Transition Matching Claims Superiority**: Meta's paper on ["Transition Matching"](https://arxiv.org/abs/2506.23589) allegedly surpasses **Flow Matching**, but its motivation is seen as weak.
   - The paper frames it as an attack vector, obfuscating prompts to tax model providers by increasing computation time, though the attacker's access model seems unrealistic as it requires query interception and modification with only blackbox access, plus the [results](https://cdn.discordapp.com/attachments/747850033994662000/1389692080275849388/image.png?ex=6866dc9b&is=68658b1b&hm=e4ff044077222da3811712070a89f156509cb64c1076648f7cd105e9f8ba8fef) indicate accuracy decline with increased obfuscation.
- **NeurIPS Seeks Ethics Reviewers for 2025**: The **NeurIPS ethics chairs** are seeking volunteers for ethics reviewers for the main review period from **July 7-20, 2025**, with additional windows through September; sign up via [this link](https://forms.office.com/r/gs3Jzq2u2Y).
   - Reviewers will support the world's largest academic AI/ML conference to ensure that published research is done responsibly, and the details of the review process can be found [here](https://neurips.cc/Conferences/2025/CallForEthicsReviewers).
- **Open Research Hackathon Reminder**: There's a reminder about the **Open Research Hackathon** happening in August, seeking community researchers to propose projects in the [research channel](https://discord.com/channels/729741769192767510/747850033994662000/1386431466447311000).
- **Single-Layer Transformer Training Pondered**: Someone inquired about training a **1-layer transformer**, wondering if it performs as poorly as expected.
- **KV Caching Explored for Cheap Inference**: Discussion arose around **KV caching**, specifically how storing the **q, k, and v** for each token (**6dV bytes fp16**) and applying **RoPE** on the fly could lead to cheap inference.
   - It was noted that various grokking/interpretability work employs this, citing ["Language Models are Secretly Performing Credit Assignment"](https://arxiv.org/abs/2306.17844) as an example.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1389930495546228907)** (2 messages): 

> `Context Engineering, Open Research Hackathon` 


- **Dive into Context Engineering**: A member shared a [GitHub repository](https://github.com/davidkimai/Context-Engineering) related to **Context Engineering**, potentially offering resources or tools for managing and understanding context in AI systems.
   - The repository could be valuable for those looking to improve how AI models interpret and utilize contextual information.
- **Open Research Hackathon still needs Proposals**: There is an upcoming **Open Research Hackathon** happening in August, and community researchers are still needed to propose projects.
   - More details can be found [in this discord channel](https://discord.com/channels/729741769192767510/747850033994662000/1386431466447311000).


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1390030369851576474)** (17 messages🔥): 

> `lm-evaluation-harness library standardization, lm-evaluation-harness init script optimization, lm-evaluation-harness task discoverability, Lazy-loading modules in lm-evaluation-harness, lm_eval startup speed` 


- **Library Standardization Underway**: A member is standardizing the **lm-evaluation-harness library** to make it more intuitive and easy to follow, with tracking issues at [#3083](https://github.com/EleutherAI/lm-evaluation-harness/issues/3083), [#3082](https://github.com/EleutherAI/lm-evaluation-harness/issues/3082), and [#3081](https://github.com/EleutherAI/lm-evaluation-harness/issues/3081).
   - They are working to simplify the [init script](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/__init__.py) which parses all the YAML scripts on startup, aiming for completion this month.
- **Optimizing lm-evaluation-harness Initialization**: The team is working on simplifying the [init script](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/__init__.py) to avoid parsing all YAML scripts on startup.
   - They plan to modularize the CI to improve startup speed and are also considering hosted documentation to enhance task discoverability.
- **Lazy-Loading Improves Startup Speed**: A member implemented lazy-loading of `simple_evaluate` and `evaluate` in `__init__.py` and moved `lm_eval` imports inside of `cli_evaluate`, leading to faster script startups.
   - This change significantly reduced the startup time from `3.61s user 0.92s system 50% cpu 8.986 total` to `0.04s user 0.01s system 98% cpu 0.051 total`.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1390040365108822280)** (3 messages): 

> `Kaiming He, Mean Flow Matching` 


- **Kaiming He wows with Workshop**: A member shared a [YouTube workshop link](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi) featuring **Kaiming He**.
   - The member highlighted Kaiming's talk, starting at **2:22:01**, and his description of **Mean Flow Matching** from **2:43:32**.
- **Mean Flow Matching is mind-blowing**: Mean Flow Matching at **2:43:32** in [Kaiming's Talk](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi) was highlighted.
   - A member shared the link, recommending Kaiming He's workshop from **2:22:01**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1389831468741562433)** (32 messages🔥): 

> `Dynamic K in similarity search, RNNs and LSTM, Universal Function Approximators, SSMs, BPTT` 


- **Dynamic K Approaches Spark Debate**: Members discussed using SentenceTransformers for similarity search, questioning how to dynamically choose 'k' in **top-k retrieval** for tasks like open-ended QA, citing the need to avoid arbitrarily setting a fixed 'k' value.
   - Potential solutions included using a **similarity threshold** to filter results or combining thresholding with a capped top-k approach.
- **RNNs/LSTMs Deep Dive or Quick Skim?**: A member asked whether to deeply study **RNNs and LSTMs** or just skim them when starting with NLP, sparking debate.
   - While some suggested focusing on the concepts of **RNNs** and **Backpropagation Through Time (BPTT)**, others argued LSTMs are outdated and not relevant for the future, preferring simplified versions like **GRUs**.
- **Universal Function Approximation Theory**: A member stated, *At modern scales, for dense feed forward architectures, the actual arch doesn't matter*, suggesting architectures are all **universal function approximators**.
   - Another member linked to [a paper on Universal Function Approximators](https://arxiv.org/abs/1906.06766i) and noted that the statement might be a hot take within the community, especially regarding State Space Models (SSMs).
- **SSMs Fundamental Paradigm Shift?**: Members debated whether **State Space Models (SSMs)** represent a fundamental paradigm shift or if they perform within margin of error of an **RNN** when scaled up in LLMs.
   - One member recalled having high hopes for **SSMs** due to contributions from the **Flash-Attention** author, anticipating advancements at both the CUDA low-level and model architecture high-level.
- **BPTT Still Relevant Today**: Members discussed Backpropagation Through Time (**BPTT**), some arguing that *BPTT is fine nowadays* and understanding it helps grasp the constraints of existing models.
   - Another member agreed it is better to be avoided if possible and that **LSTMs** are outdated.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1389712618557669527)** (3 messages): 

> `Linear Transformers Parallelization, Delta Rule over Sequence Length, RWKV-7 Equation 18, DeltaNet Performance` 


- **Delta Rule Speeds Up Linear Transformer Parallelization**: The discussion will cover [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484), focusing on parallelization of equation 18 from the [RWKV-7 paper](https://arxiv.org/pdf/2503.14456#page=18).
   - The paper introduces a hardware-efficient algorithm for training linear transformers with the delta rule, enabling scaling of **DeltaNet** to standard language modeling settings, outperforming **Mamba** and **GLA**.
- **DeltaNet Outshines Linear Transformers in Associative Recall**: **DeltaNet** enhances linear transformers by replacing the additive update with the delta rule, proving more effective in associative recall, as detailed in the [DeltaRule paper](https://arxiv.org/abs/2406.06484).
   - The algorithm exploits a memory-efficient representation for computing products of **Householder matrices**, scaling a **1.3B** model for **100B** tokens and achieving superior perplexity and zero-shot performance.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1389682027951030373)** (54 messages🔥): 

> `Healthcare Decisions, Immunotherapy Development, American vs European Food, Transition Matching` 


- **Actuaries vs. Doctors: Who Decides?**: A discussion arose regarding the roles of physicians and actuaries in healthcare, with one member stating a physician's job is to *diagnose the problem and solution*, while an actuary determines what treatments are above the threshold for quality-adjusted life years per dollar the organization is willing to cover.
- **Brewing Immunotherapy in Germany**: A member recounted traveling to Germany to develop an immunotherapy for their business partner's cancer, criticizing the German healthcare system's *rationed care* approach.
   - The user expressed that taking responsibility into your own hands is better *than waiting for the magic sky daddy of big government to give you "human rights"*.
- **American Freedom vs. European Food Regulation**: A member argued that the US allows freedom to *eat yummy food* and pay for individual healthcare, contrasting this with European regulations on food content.
   - Another member countered that the lack of food regulation in the US leads to companies using *the most addictive cheap shit* in food, banned in civilized countries.
- **"Transition Matching" Claims Superiority**: A member shared an [arXiv link](https://arxiv.org/abs/2506.23589) to a Meta paper on *"Transition Matching"*, suggesting it's superior to Flow Matching.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1389720693704233051)** (17 messages🔥): 

> `nsys and torch.compile, cursor and windsurf, GPT vs Claude, Work-Life Balance, European Work` 


- ****NVIDIA System Profiler (nsys) stalls when used with torch.compile****: A user reported that the **nsys profiler stalls** when used with **torch.compile**, even with explicit NVTX ranges and attempts to stop the CUDA profiler, and included a [code snippet](https://cdn.discordapp.com/attachments/1389720693704233051/1389722323082280981/image.png?ex=6866f8c5&is=6865a745&hm=8b2e349f1af1cdf1f19125827205c2378b5fd80c35f8c7ecdb0dc4843cd52ba9) reproducing the issue.
   - The user linked to a [relevant NVIDIA forum thread](https://forums.developer.nvidia.com/t/nsys-profile-pytorch-fails-under-torch-compile/332302/5) where it is claimed that **nsys is supposed to work fine with torch.compile**.
- ****Cursor and Windsurf's fate when Claude disappears****: Members speculated about the fate of tools like **Cursor** and **Windsurf** if **Claude Sonnet** were to cease to exist, suggesting they would likely switch to using **GPT** anyway.
   - One member stated they *prefer GPT over Claude* most of the time.
- ****From SWE to MLE Role: Balancing Act, More Fulfilling****: One member shared their experience transitioning from a **SWE role to an MLE role**, citing improved work-life balance and fulfillment despite the initial learning curve.
   - They emphasized the importance of surrounding oneself with the right people and noted that **direct interaction with end-users** contributes significantly to their sense of fulfillment, contrasting it with feeling stuck in a [feature factory](https://cutle.fish/blog/12-signs-youre-working-in-a-feature-factory) in their previous role.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1389958863440056320)** (1 messages): 

> `Triton Nightly Wheel Builds, TensorDescriptor Use` 


- **Triton Nightly Wheel Builds Busted**: A member questioned why **Triton's nightly wheel builds** have been broken for months without a fix.
   - They emphasized the importance of fixing this issue, especially since the examples depend on recent features and source builds can take around **1 hour**.
- **TensorDescriptor Demands Source Build**: A user pointed out that using **TMA** via `TensorDescriptor` as used in the official examples, necessitates building from source.
   - This is *annoying when you rent instances*, they added.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

gau.nernst: https://x.com/davisblalock/status/1939956579698094166
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1389848341726101535)** (43 messages🔥): 

> `CUDA for deep learning tasks, Implementing custom ML algorithms, Docker image for CUDA, Contributing to existing libraries` 


- **Value of CUDA for Deep Learning Debated**: Members discussed the value of learning **CUDA** for deep learning tasks today, considering that many libraries abstract away the need to write custom **CUDA kernels**.
   - It was mentioned that it makes the most sense for engineers working to create those libraries, though others chimed in to say it would depend on the depth users want to go, and if they can use premade LEGO blocks of existing libraries vs writing their own.
- **CUDA helps implement custom ML algorithms**: For implementing custom **ML algorithms** or **DL models** for niche use-cases and real-time constraints (such as 3D point cloud processing), it depends on whether pre-made library blocks are sufficient or custom code is required.
   - As one user stated, *thing is the libraries i looked at they dont support all the functions i want, and its taking too much time on the cpu*.
- **Docker Images Provide Compatibility for CUDA**: Members mentioned using **Docker images** as a solution for library compatibility issues with specific **CUDA toolkits**, though some noted memory-intensive drawbacks.
   - One user said, *i've tried the docker image route, which worked well, but its too memory-intensive to use* and cited a 50GB+ docker for Paddle OCR, but another user responded with a [link](https://developer.nvidia.com/cuda-12-0-0-download-archive) where older CUDA versions can be downloaded.
- **Contributing to Libraries is best for extending them**: Users discussed the possibility of contributing to existing libraries to add support for missing functions rather than creating new ones.
   - This was considered potentially time-consuming and requiring significant experience, but one member asked, *if a library already exists to accomplish something fast on the GPU, why would people try to roll out new libraries?*


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1389900130949402697)** (1 messages): 

> `FSDP 2.0, DTensors, Sharding` 


- **FSDP 2.0 full_state_dict loading issue revealed**: A user reported that their `full_state_dict` stops loading in **FSDP 2.0** with `torch.distributed.checkpoint.state_dict.set_model_state_dict` after the forward pass.
   - The reason given is that after the forward pass, the parameters remain **unsharded** and `.parameters()` doesn't return **DTensors**, preventing the loading of the state dictionary.
- **Unsharded parameters block state dict**: After the forward pass, parameters remain unsharded, leading to `.parameters()` not returning DTensors.
   - This issue obstructs the loading of the state dictionary in **FSDP 2.0** using `torch.distributed.checkpoint.state_dict.set_model_state_dict`.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1389772850705207438)** (3 messages): 

> `GPU, CUDA, Interview prep, Cram resources, YouTube tutorials` 


- **GPU/CUDA Interview Prep Requested**: A member asked for **cram resources** or good review materials for an upcoming interview focusing on **GPU/CUDA** questions.
- **CUDA Tutorial Playlist Shared**: A member shared a [YouTube playlist](https://youtube.com/playlist?list=PLnH7E0IG44jFfiQBd_Ov7FmYHq8SZx6r0&si=tXBqdEFKkrWnrlBK) as a resource for **GPU/CUDA** interview preparation.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1390066767774089236)** (1 messages): 

> `Compiler register lifetime, Avoiding register spills` 


- **Compiler Guidance on Register Lifetimes**: A member inquired about strategies for guiding the compiler to understand the lifetime of registers and prevent unpredictable register spills.
   - They noted that seemingly minor code adjustments can unexpectedly lead to significant increases in register spilling.
- **Tuning Compiler to Avoid Register Spills**: A user seeks advice on influencing the compiler to better manage register lifetimes and reduce unnecessary spills.
   - The problem is that minimal code changes appear to significantly increase register spills, suggesting unpredictable compiler behavior.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1389771823633404005)** (2 messages): 

> `Recipe Index Launch, Google Meet Link` 


- ****Yeet** Launches Comprehensive Recipe Index**: The team at **Yeet** launched a complete index of all their recipes, available at [yeet.cx/recipes](https://yeet.cx/recipes).
- **Google Meet Link Shared**: A **Google Meet link** was shared for a meeting, accessible at [meet.google.com/wdk-yipf-zjd](https://meet.google.com/wdk-yipf-zjd).


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1389878076053389322)** (1 messages): 

> `Apple Silicon, Thunderkitten` 


- **Thundermittens Ported to Apple Silicon?**: A member inquired if there was a fork or code available for **Thundermittens** (or **Thunderkitten**) specifically for **Apple Silicon**.
- **Another Discussion Point**: Adding another topic to meet the minItems requirement.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1389702173004271747)** (1 messages): 

> `FSDP Config, model_dtype parameter, Qwen2.5` 


- **FSDP Config Requires Model Data Type**: The `model_dtype` parameter is required to be set under the `fsdp_config` section in the verl actor config, otherwise it will default to the dtype of the model checkpoint.
   - For **Qwen2.5**, this defaults to **fp32**, which can cause confusion if not explicitly set.
- **fp32 Default Dtype**: If the `model_dtype` parameter is not set, the config defaults to the dtype of the model checkpoint.
   - This can result in unexpected behavior when using **Qwen2.5**, as it defaults to **fp32**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1389848242262638614)** (2 messages): 

> `FLE talk, Pre-training` 


- **FLE Talk Possibly Postponed**: Members discussed postponing the **FLE talk** to August to include training and infra results.
   - They felt this would be *more amenable to the audience* than discussing evals and QoL improvements.
- **Pre-training Soon Incoming**: The team aims to start **pre-training** soon, which influences the desired content for the **FLE talk**.
   - They requested a weekday in August to reschedule, given the team is fully booked that month.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1389967707692208139)** (1 messages): 

> `Cutlass Kernel Performance Prediction, Analytical Cost Models vs. Autotuning, GEMM Kernel Performance Predictability` 


- **Cutlass Kernel Cost Models Sought**: A member inquired about the existence of [cost models](https://link.to/cost-model-info) that predict the performance of a **Cutlass kernel** based on its template parameter configuration.
   - They noted that many DL compilers, including **Torch Inductor**, rely on profile-guided autotuning for **Cutlass kernel selection**.
- **Analytical Cost Models Challenging Autotuning**: The member questioned whether **Cutlass's** metaprogramming architecture should enable analytical cost model-based kernel selection, rather than relying on autotuning.
   - They also asked whether **GEMM kernels** are regular enough to have predictable performance, making autotuning unnecessary.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1389682787891679258)** (66 messages🔥🔥): 

> `Anysphere/Cursor hires Anthropic Claude Code Leaders, Anthropic's $4B ARR, Meta's Aggressive Poaching of AI Talent from OpenAI, Luma Labs AI Modify Video Tool, Perplexity New Subscription Tier` 


- **Cursor Courts Claude Code Chiefs!**: [Amir Efrati reports](https://x.com/amir/status/1940112288381641026) that Anysphere/Cursor has hired two senior leaders from **Anthropic's Claude Code team**, while Anthropic has reached an annual recurring revenue (**ARR**) of approximately **$4 billion**, a fourfold increase since the beginning of the year.
- **Meta's Mammoth Money Moves Make OpenAI Mad!**: Meta is offering lucrative compensation packages (up to **$300M** over **4 years**) to poach AI researchers from OpenAI, with first-year compensation exceeding **$100M** [according to this thread](https://x.com/tanayj/status/1940137574141694046).
- **Lights, Camera, Remix! Luma Launches Lavish Video Tool!**: Luis C announced that **Luma Labs AI's 'Modify Video' tool**, which allows users to remix any video and change the style of frames, is now available on [Replicate](https://xcancel.com/lucataco93/status/1940113275221344566).
- **Perplexity Premieres Premium Plan: Perplexity Max!**: Perplexity.ai has introduced **Perplexity Max**, a new premium subscription tier offering unlimited Labs queries, access to a wider range of frontier models, and early access to forthcoming products like [Comet](https://xcancel.com/perplexity_ai/status/1940443479710257226).
- **Microsoft Massively Musters 9,000 Misfits!**: Microsoft is reportedly laying off **9,000 workers**, sparking discussions about AI's role in job displacement and the broader economic impact [according to this report](https://xcancel.com/unusual_whales/status/1940399771371602221).


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1390012398408171531)** (4 messages): 

> `Information Theory, Jack Morris, LLM Inversion` 


- **Latent Space Explores Information Theory for LLMs**: Latent Space dropped a new episode, discussing [Information Theory for Language Models with Jack Morris](https://xcancel.com/latentspacepod/status/1940453495465038067).
   - The conversation touches on **learning as compression**, a concept advocated by **Ilya Sutskever**, as well as a *New Type of Information Theory* covering **V-information, embeddings, and LLM inversion/extraction**.
- **Morris Champions New Information Theory**: **Jack Morris** advocates for a *New Type of Information Theory* covering **V-information, embeddings, and LLM inversion/extraction**.
   - Morris shares insights from his **AI PhD experience** and well-received papers.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1389775063355756635)** (63 messages🔥🔥): 

> `O'Reilly Book on MCP, Storing Docs for LLM, MCP Inspect and Badge Issues, Claude Hooks for Git, MCP Routing Layer` 


- **Writing O'Reilly MCP masterpiece**: A member is writing an [O'Reilly book on MCP](https://www.oreilly.com/) and joined the Discord server for news.
   - Another member reminisced about O'Reilly books, but cautioned that *the world is changing too fast nowadays for it to be as relevant as it was 10 years ago*.
- **Team Tussles with Tech Docs Tactics**: Members discussed storing company documentation for LLM use, with some preferring **markdown files** for context and others using **Google Docs**.
   - One member built an **MCP server** for PKM CRUD, while another suggested [HackMD](https://hackmd.io) and yet another mentioned **Obsidian**.
- **MCP Inspect and Badge Baffle Brains**: A member reported issues with the **Inspect option** and **badge updates** for security and quality on their MCP server after updating the Docker file.
   - Another member confirmed that they were having the same issue.
- **Crafty Claude Conjures Code Commits**: Members are starting to play with **Claude Hooks** to handle git operations when the **Jane docs** get touched.
   - One member has been *cheating* with [context7](https://context7.ai/).
- **MCP-Routing Revelations**: A member proposed an **MCP-Routing layer** to manage context window size for different LLMs and MCP tools like Context7, which could *trim down* requests based on the specific needs of each tool.
   - The discussion extended to whether MCP servers should be **REST APIs** to avoid hallucinations and improve efficiency.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1389804581893968005)** (2 messages): 

> `MCP, tip.md, x402, CDP SDK, Coinbase Hackathon` 


- ****Tip.md** Showcased with Agentic Interface in MCP**: The submitter showcased their **MCP**, an agentic interface for [tip.md](https://farcaster.xyz/tipdotmd/0x41398e69), enhanced with **x402** and **CDP SDK** for crypto tips, which was developed for the **Coinbase Hackathon** and is detailed on [Devfolio](https://devfolio.co/projects/tipmd-d033).
   - The project was selected as one of the four to be featured at the upcoming [Coinbase x402 Demo Day](https://discord.com/events/1220414409550336183/1389345035753095168), with a [demo video available on YouTube](https://youtu.be/rWtWvPA_4BA?si=Hu7r8sdD6H19ppG2).
- **MCP Extends Existing Capabilities**: MCP extends existing capabilites with agentic interface of tip.md, with x402 + CDP SDK for crypto tips.
   - MCP was extending my existing MCP to add this for coinbase hackathon


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1389684491127361698)** (24 messages🔥): 

> `Cypher Alpha performance, Claude Sonnet ranking, Openrouter Oauth issues, Aider API key problems, Claude Code comparison` 


- **Cypher Alpha Model Bashed for Bad Coding**: Members found **Cypher Alpha** to be inferior in coding compared to other models like **Qwen3 30B A3**, with one user quipping that it's *a time capsule back to 2022*.
   - Another user humorously remarked that it's *very bad at everything*, deeming it one of the worst models tested in the last year.
- **Sonnet 7 Impresses as Sonnet 4 fails to excite**: A user expressed surprise at the low ranking of **Claude Sonnet 3.7** and **Sonnet 4** on a leaderboard.
   - They highlighted their positive experience using **Sonnet 7** with **Thinking 8k** for coding, noting its speed improvement since the hype around **Sonnet 4**.
- **Openrouter Oauth faces Obstacles**: A user reported that the new **Openrouter Oauth** is not functioning, as the Oauth pop-up is not appearing.
   - Other users jumped in, advising to redo the api key without the brackets.
- **Aider API Key generates headaches**: A user reported encountering issues with **Aider**, where the API key was repeatedly identified as missing.
   - A community member suggested reinstalling **Aider** to resolve the API key issue.
- **Grok 4 Eyes Possible July 4 Drop**: Members noted the possible release of **Grok 4** on July 4, allegedly.
   - One member tempered expectations, saying *it's not even close to a Grok 3 mini*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1389685898123280616)** (27 messages🔥): 

> `aider /architect mode, Local Model Recommendations, aider auto test, aider --yes-always and --no-always, Quantized Models` 


- **Navigating `/architect` Mode and Edits in Aider**: A user sought clarification on executing plans made with `/architect`, noting that changes weren't appearing in the repo, and discovered that using `/code` initiates edits immediately.
   - Another user suggested that edits should start after pressing enter upon completion, while experimentation showed that pressing enter is not necessary.
- **Qwen vs DeepSeek: Model Recommendations for Aider**: Users discussed local model performance, with **Qwen3 32b or 30b** being recommended over **deepseek/deepseek-chat**, due to issues with excessive thinking and commit message authoring.
   - One user experienced garbage performance using a local model (70B Deepseek) despite having an RTX5000 and was seeking recommendations.
- **Automated Testing with Aider Commits**: A user inquired about running `make test` automatically after each aider commit.
   - The solution is to *turn on auto test and set test command to make test* within Aider's settings.
- **Aider Prompts About Adding New Files**: A user noticed that Aider outputs the whole answer, and then asks *Add file to the chat?* even if the file has not changed since the beginning of the request.
   - The user references the [Aider FAQ](https://aider.chat/docs/faq.html#why-did-aider-ignorediscard-its-proposed-edits-after-it-asked-to-add-a-new-file-to-the-chat) for more details.
- **Aider's Command Line Options**: A user inquired about the opposite of the `--yes-always` command line option in Aider, seeking a `--no-always` equivalent.
   - The replies did not directly answer this question.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1389879513806147586)** (4 messages): 

> `NotebookLM use cases, NotebookLM as a personal daily journal, Audio overview function` 


- **NotebookLM serves dual purpose**: A member is setting up **NotebookLM** as a personal daily journal to log **reflections, media, Gemini chats, and suggestions** and as a searchable notes database for articles, ideas, and reference material.
   - They plan to use **Google Docs** as the single source of truth for privacy and data control but are looking for alternative input methods to build a resilient and easy-to-maintain system.
- **NotebookLM Explains Work-In-Progress Books**: A member uses **NotebookLM** to help explain their work-in-progress books.
   - They mostly use the **audio overview function**, as it explains for them.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1389707511187378207)** (21 messages🔥): 

> `NotebookLM sources, Podcast generation, Gas plant power factor, Pro vs Free accounts, Opening PDF sources` 


- ****NBLM Source Selection Clarified****: A user inquired whether reactivating all sources after creating a mindmap from specific sources would cause discussion topics to pull from all sources, or just the initially selected ones.
- ****Podcast Generation Inquiry Emerges****: A user requested advice on how to generate longer podcasts for "Audio overview," seeking thoughts and opinions from the community.
- ****Gas Plant Power Factor Queried****: A user asked about the power factor of gas plants.
- ****NBLM Pro vs. Free Account Differences Sought****: A new user in NBLM inquired about the differences between pro and free accounts.
- ****Request to Open PDF Sources in NBLM****: A user asked if there's a way to open a PDF file attached as a source, noting that clicking it loads it messily in the sources window.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1389754687095570532)** (5 messages): 

> `Cohere open model weight release, CMD-R model, tool/agent frameworks, ML Summer School channel` 


- **Cohere users eagerly anticipates new open model**: Users are hoping for a new or updated open model weight release from **Cohere**, despite recognizing their focus on the Enterprise market.
   - One user suggests an updated weight with emphasis on **tool/agent frameworks** and updated architecture would be fantastic.
- **CMD-R approaches its first birthday**: A user noted that the **CMD-R** model from 08-24 is almost a year old but still very usable.
   - The user believes a new weight from Cohere would show the modern group of open weights from competing providers how it's done!
- **User unable to locate ML Summer School channel**: A user is trying to find the **#ml-summer-school** channel, mentioned in a [Cohere Labs Community link](https://sites.google.com/cohere.com/coherelabs-community/community-programs/summer-school).
   - The user is requesting assistance in locating the specified channel.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1390026203087503370)** (4 messages): 

> `Cohere Embedding Model, Trial key, Rate limits, Production key, Monthly limit` 


- **Trial Keys Unlock Cohere Embeddings, but with speedbumps**: Users confirmed that the **Cohere embedding model** is indeed accessible via a **trial key**.
   - However, they cautioned that **trial keys** come with more restrictive **rate limits** and a monthly usage cap compared to production keys.
- **Trial vs Production Keys: Feature Parity, Usage Caps**: It was clarified that both **trial** and **production keys** unlock the same features within the Cohere platform.
   - The key difference lies in the **trial key's** inherent **monthly limit**, a constraint not imposed on production keys.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1389721267086561453)** (10 messages🔥): 

> `ML Summer School, Agentic AI, Computer Vision, Water Quality Monitoring` 


- **Reinforcement Learning Enthusiast Joins Cohere**: Sriram from India, a master's student, introduces himself as working on **reinforcement learning** and **on-device safe AI**.
- **Computer Vision Student Focuses on Semantic Segmentation**: Dojo, a second-year student, shares their work on **computer vision models**, specifically focusing on **semantic segmentation**.
   - They are interested in reducing the redundancy and dependence of early network layers on the final layers, with and without skip connections, while also learning how to apply modern **NLP architectures** and **language modeling techniques** to computer vision tasks.
- **Researcher Integrates AI for Water Quality Monitoring**: Oraib, a Ph.D. candidate at FCT NOVA, specializes in **AI applications** for **water quality monitoring**.
   - Their research integrates **satellite imagery** with **in-situ measurements** to develop accurate and scalable models for environmental assessment.
- **New Member Seeks ML Summer School Channel**: A new member is trying to find the **#ml-summer-school** channel, referencing the [Cohere Labs Community Summer School site](https://sites.google.com/cohere.com/coherelabs-community/community-programs/summer-school).
- **Student Explores Agentic AI and Graphic Design**: Abdullah Abbasi introduces themself as a student of **Agentic AI** and a **graphic designer**.


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1389925623174139925)** (3 messages): 

> `Secure ML, Privacy Preservation, AGI is here` 


- **AGI is here, claims User**: A user claimed that *agi is here*.
- **Debate over Secure ML and Privacy Preservation**: A user inquired about **Secure ML** and **Privacy Preservation**, asking for elaboration on the meaning of **Secure ML**.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1389754103747575889)** (15 messages🔥): 

> `Mojo Origin Tracking System, Ownership and Life Cycles in Mojo, Mojo structs vs classes, GPU puzzles in Mojo, Dependent Type System in Mojo` 


- **Delving into Mojo's Origin Tracking System**: A member inquired about talks or docs on the implementation of the **Mojo origin tracking system** (borrow checker).
   - Another member shared [documentation on Ownership and life cycles](https://docs.modular.com/mojo/manual/values/ownership) and [lifetimes and references](https://docs.modular.com/mojo/manual/values/lifetimes), while another member shared a [relevant youtube video](https://www.youtube.com/watch?v=9ag0fPMmYPQ) and suggested that the language creator plans to give a talk on the topic eventually.
- **Mojo Structs Stand Apart From Classes**: A member asked about the differences between **Mojo structs** and **classes**.
   - Another member shared a link to the [official documentation](https://docs.modular.com/mojo/manual/structs#structs-compared-to-classes) outlining the many differences.
- **Decoding GPU Puzzle Barrier Needs**: A member is seeking clarification on the need for the second barrier in the **matrix tiling problem** within the [GPU puzzles](https://puzzles.modular.com/puzzle_14/tiled.html#tile-processing-steps).
   - Another member suggested posting the question in the forums for a more detailed explanation from experts.
- **Dependent Type System in Mojo**: A member inquired whether **Mojo** would include more advanced concepts like **graded types** as the language matures, referencing a research paper on the topic.
   - Another member responded that **Mojo** is moving towards a **dependent type system**, but it has to be constrained by what can be checked at compile time, balancing features with compile times and runtime performance.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1390061966453899304)** (2 messages): 

> `Mojo Offline Inference, QuantizationEncoding, LLM on M1 Mac, Nightly vs Stable Builds` 


- **Mojo 🔥 Offline Inference Quandaries on Nightly Build**: A user encountered issues following the [Modular Max offline inference documentation](https://docs.modular.com/max/serve/offline-inference/) while trying to use the [llm4decompile-v1.5/1.3B-Q6_K model](https://builds.modular.com/models/llm4decompile-v1.5/1.3B-Q6_K) on an M1 Mac using the nightly build.
   - The user faced a `ValueError` related to unsupported `quantization_encoding`.
- **Stable Build Saves the Day**: The user reported that using the stable build of Mojo 🔥 resolved the `quantization_encoding` issue.
   - The model worked as expected with the quantization encoding specified in the stable build.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1389722734631718932)** (12 messages🔥): 

> `Making friends online, Manus AI new unlimited plan, NFT Scams` 


- **Users seek online friendships for AI discussions**: A user expressed a desire to form online friendships with individuals who share their interest in discussing AI-related topics, noting the difficulty in finding such connections on platforms like **Discord** and **Reddit**.
   - The user emphasized the need for friends who are on the *same page* regarding these niche interests.
- **Manus AI offers free unlimited chat!**: A user announced that [Manus AI](https://manus.im/invitation/R9QW2MZDXXJJ4) now offers **unlimited free messages** in the chat, and is grounded in **Gemini**.
   - Another user inquired if Manus AI has improved, suggesting it was previously just **Claude** with tools, with the main issue being its cost.
- **NFT project potentially a scam**: A user inquired whether a project was selling **NFTs** after seeing it promoted on **Twitter**, to which another user responded that it *sounds like a scam*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1390066300482617355)** (2 messages): 

> `Mentorship for Independent Research` 


- **User Seeks Mentorship for Independent Research**: A user with some experience is seeking mentorship to start doing independent research.
   - The user requested interested mentors to send them a direct message.
- **Mentorship Opportunity for Budding Researchers**: An intermediate-level user is looking for guidance on initiating independent research projects.
   - Interested mentors are encouraged to connect via direct message to provide specific advice and support.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1390066300482617355)** (2 messages): 

> `Mentorship Request` 


- **Member requests mentorship on independent research**: A member who is *not an absolute beginner* requested **mentoring** on how to start doing **independent research**.
   - The member specified they would like others to **DM them if interested**.
- **Another member seeks guidance on independent research**: Another member expressed interest in receiving mentorship to initiate their own research endeavors.
   - This individual aims to transition from being a novice to conducting self-directed research projects.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1389936262009323570)** (13 messages🔥): 

> `Floor plan analysis with GPT4All, Image recognition limitations, LM Studio image acceptance, ChatGPT image analysis` 


- **GPT4All Seeks Model to Parse Blueprints**: A member is looking for a good model for use with **GPT4All** that will handle questions about building design and be able talk about and evaluate a uploaded **JPG** floor plan.
   - Another member points out that **GPT4All** can't upload images and questions the capabilities of **ChatGPT**, suggesting that it may *want to be as helpful as much* and give inaccurate answers.
- **Members Debate Image Recognition and AI**: A member believes that while **ChatGPT** can recognize objects like *trees*, *buildings*, *faces*, and *persons*, it may not be able to accurately *discuss* them.
   - Another member stated that the mode caught that the exits were not marked while using **ChatGPT** with an image and no system prompt.
- **LM Studio may accept images**: A member noted that **LM Studio** with the right model accept images.
   - A member showed the output of an image fed to **ChatGPT** ([Attached: Class_building_floor_plan.jpeg](https://cdn.discordapp.com/attachments/1090427154141020190/1389982683769344162/Class_building_floor_plan.jpeg?ex=686699c0&is=68654840&hm=9e7f70c07100c5c382138ec60c0a5bfed2eb7cd22243265cc9c848b70509875c&)).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1389682419057037464)** (3 messages): 

> `LlamaExtract, LlamaCloud, Enterprise RAG, Multi-modal indexing` 


- **LlamaExtract Auto-Generates Schemas**: The new feature **LlamaExtract** can now automatically generate a schema from a document and/or a prompt, alleviating the need to manually build one.
   - A document and a description is all that's needed, check out [this tweet](https://twitter.com/llama_index/status/1940123273981043035) for more.
- **LlamaCloud Scales Enterprise RAG**: A new blog post details **4 Ways LlamaCloud Scales Enterprise RAG**, sharing lessons learned while scaling LlamaCloud for huge enterprise workloads.
   - The post aims to help others building high-scale document indexing and retrieval learn what's coming, read the full story on [this tweet](https://twitter.com/llama_index/status/1940440399690248669).
- **LlamaCloud Indexes Retrieves Images**: You can now retrieve images and illustrative figures from your **LlamaCloud Indexes** as well as text, perfect for presentations and reports.
   - Enabling this feature is as simple as toggling the *"Multi-modal indexing"* option, see [this tweet](https://twitter.com/llama_index/status/1940485676371530035) for details.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1389927787359375400)** (6 messages): 

> `OpenAI Batch API, LlamaIndex Workflows 1.0, Embedding OpenAI API Key, Developer Collaboration` 


- **Batch API Blues?**: Members are trying to move their workflow to **OpenAI's Batch API** due to high costs but there seems to be no built-in support for batch calling in LlamaIndex.
   - The recommendation is to use the **OpenAI lib** directly for batch calling within their workflow.
- **LlamaIndex Workflows 1.0 is Here!**: LlamaIndex announced the release of **Workflows 1.0**, a lightweight framework for agentic systems on [their blog](https://www.llamaindex.ai/blog/announcing-workflows-1-0-a-lightweight-framework-for-agentic-systems).
- **Env Vars the Key to API Keys**: One member needed help embedding their **OpenAI API key** in LlamaIndex's No-Code UI, and another member pointed out that *LlamaIndex is only looking for an environment variable*.
   - They clarified that setting an env var called `OPENAI_API_KEY` should do the trick.
- **Passionate Dev Seeks Collab**: A passionate developer is looking to collaborate on projects and is offering their services.
   - They've attended various kinds of projects and are eager to work together.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1389709645672743086)** (6 messages): 

> `Manus Prompt, MCP Server, Claude Opus 4, Qwen 3, 32B model` 


- **Fix Manus Prompt, Manus Team!**: A user requested the Manus team to fix their prompt or add a mini prompt for messages that contain the word image.
   - The user tagged a team member <@1352272661589524572>.
- **MCP Server Build on Claude Opus 4**: A user tagged a team to help build an **MCP Server** (like a Formula One race car) with **Claude Opus 4**.
   - *It's going to be in the shop every day because you're going to be figuring out how to make it work.*
- **Qwen 3, 32B Model Work**: The user mentioned that some things do work out of the box, using it with a **Qwen 3, 32B model** in **LM studio**.
   - The user included a [link](https://files.catbox.moe/5gf51x.txt).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1389987658423603281)** (3 messages): 

> `haldie style viz, tile viz approach, shared mem buffers, global buffers` 


- **Haldie Viz in Python Backend**: A member showcased a "haldie style" visualization implemented in the Python backend, highlighting memory access patterns with green for reads and red for writes.
   - The [demo](https://cdn.discordapp.com/attachments/1068976834928193609/1389987658020819056/haldie_viz.mov?ex=68669e62&is=68654ce2&hm=8458418f0dc82f9dcac83e6a7eaad13d1d4b79101c26945629d63322948c6bd6&) visualizes shared memory buffers above and global buffers below, tailored for Metal with `tc=3`.
- **Tile Viz Re-Evaluation**: A member revisited the idea of tile visualization and realized the Python backend was a great place for it, before approaching it at the wrong level.
   - They cited a related paper, [Simulating Time With Square-Root Space](https://arxiv.org/abs/2502.17779), suggesting a connection to the visualization approach.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

.pyrophoric.: hi is there a cli tool to automatically format code in the tinygrad style?
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

chiggly007: What do you mean by this?
  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1389712921084563618)** (1 messages): 

> `Autonomous Agents, Multi-Agent Systems, LangChain, AutoGen, AI Assistants` 


- **Engineer offers services building Autonomous AI Agents**: An AI Engineer with 9 years of experience offered services building, training, and deploying **AI models** and **autonomous agents** using tools like **GPT-4o**, **LangChain**, and **AutoGen**.
   - They are looking to team up with startups or AI tools and specialize in autonomous research bots, multi-agent systems, AI assistants, and more.
- **AI Engineer's Tech Stack Highlights**: The engineer's tech stack includes **LangChain**, **Langraph**, **AutoGen**, **ReAct**, **CrewAI**, **DeepSeek**, **OpenAI**, **Claude**, **Hugging Face**, and **Playwright**.
   - They also have expertise in **Deep Learning** (CNN, RNN, Transformers), **NLP**, and **Computer Vision**.

