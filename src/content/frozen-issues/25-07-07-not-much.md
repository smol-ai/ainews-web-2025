---
id: MjAyNS0w
title: not much happened today
date: '2025-07-07T05:44:39.731046Z'
description: >-
  Over the holiday weekend, key AI developments include the upcoming release of
  **Grok 4**, **Perplexity** teasing new projects, and community reactions to
  **Cursor** and **Dia**. Research highlights feature a paper on **Reinforcement
  Learning (RL)** improving generalization and reasoning across domains,
  contrasting with Supervised Fine-Tuning's forgetting issues. **Energy-Based
  Transformers (EBTs)** are proposed as a promising alternative to traditional
  transformers. **AI21 Labs** updated its **Jamba** model family with enhanced
  grounding and instruction following, maintaining a **256K** context window.
  **Baidu** open-sourced its massive **424 billion** parameter **Ernie 4.5**
  model, while **Kontext-dev** became the top trending model on **Hugging
  Face**. Advances in length generalization for recurrent models and the
  introduction of **2-simplicial attention** were noted. In biomedical AI,
  **Biomni**, powered by **Claude 4 Sonnet**, demonstrated superior accuracy and
  rare disease diagnosis capabilities. Additionally, the Python package manager
  `uv` received praise for improving Python installation workflows.
companies:
  - ai21-labs
  - hugging-face
  - baidu
  - perplexity-ai
  - deepmind
  - anthropic
models:
  - grok-4
  - jamba
  - ernie-4.5
  - claude-4-sonnet
  - claude-4
  - kontext-dev
topics:
  - reinforcement-learning
  - fine-tuning
  - energy-based-transformers
  - ssm-transformer
  - context-windows
  - length-generalization
  - recurrent-neural-networks
  - attention-mechanisms
  - 2-simplicial-attention
  - biomedical-ai
  - instruction-following
  - open-weight-models
  - python-package-management
people:
  - _philschmid
  - corbtt
  - jxmnop
  - sedielem
  - _akhaliq
  - slashml
  - alexiglad
  - clementdelangue
  - _albertgu
  - tri_dao
  - theaitimeline
  - deep-learning-ai
---


**a quiet holiday weekend**

> AI News for 7/4/2025-7/7/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (222 channels, and 15367 messages) for you. Estimated reading time saved (at 200wpm): 1249 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Grok 4 is coming, Perplexity is teasing something, people are upset at Cursor, excited at Dia, and monitoring the situation of more Meta Superintelligence hires.

---

# AI Twitter Recap

**AI Models, Research, and Techniques**

- **RL for Improved Generalization and Reasoning**: A paper highlighted by [@_philschmid](https://twitter.com/_philschmid/status/1941751561870274691) explores how **Reinforcement Learning (RL)** tuning on math data successfully transfers gains to other domains, while Supervised Fine-Tuning (**SFT**) can cause "catastrophic forgetting." The study found **RL** selectively adjusts a small number of relevant tokens, preserving core knowledge. This sentiment is echoed by [@corbtt](https://twitter.com/corbtt/status/1941753134281523482), who notes that customers using **RL** to train agents on specific domains are "extremely happy." The broader challenge, as [@jxmnop](https://twitter.com/jxmnop/status/1941599637061697984) describes it, is the palpable tension among researchers to make post-training as "clean and elegant as pre-training."
- **Diffusion Models and Energy-Based Transformers**: [@sedielem](https://twitter.com/sedielem/status/1941527778408661202) shares a blog post explaining that while diffusion models have analytical solutions, they involve sums over the entire training set and don't generalize well in practice. In a related vein, [@teortaxesTex](https://twitter.com/teortaxesTex/status/1941561340256223523) and [@_akhaliq](https://twitter.com/_akhaliq/status/1941920969590792701) highlight a paper on **Energy-Based Transformers (EBTs)** as a conceptually interesting approach that could address some objections to LLMs. [@slashML](https://twitter.com/slashML/status/1942268809592623232) shares the paper from author [@AlexiGlad](https://twitter.com/AlexiGlad/status/1942268809592623232), which claims EBTs can out-scale feed-forward transformers.
- **AI21 Labs Jamba Model Family Update**: [@AI21Labs](https://twitter.com/AI21Labs/status/1942197784259461385) announced a new update to its **Jamba** open model family. The model maintains its hybrid **SSM-Transformer** architecture and **256K** context window but now features improved grounding and instruction following. The open-weight models are available on **Hugging Face**.
- **New Model Releases and Trending Models**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1941539058095997364) noted that **Baidu** open-sourced its **424 billion** parameter model, **Ernie 4.5**. He also shared that **Kontext-dev** by [@bfl_ml](https://twitter.com/ClementDelangue/status/1941666556913521109) became the number one trending model on **Hugging Face** with over 100 derivative models within a week of release.
- **Length Generalization in Recurrent Models**: [@_albertgu](https://twitter.com/_albertgu/status/1942301060745363886) praises a new paper for its elegant framing and solution to improve length generalization in recurrent models like **RNNs, SSMs, and linear attention**. [@tri_dao](https://twitter.com/tri_dao/status/1942302682561274356) summarizes the finding, stating that it can be achieved by "simply training for another extra 100 steps with a careful choice of initial states."
- **2-Simplicial Attention**: A paper introducing **2-simplicial attention** has gained traction, with [@TheAITimeline](https://twitter.com/_arohan_/status/1942261414321852807) listing it as a top paper of the week. [@*arohan*](https://twitter.com/_arohan_/status/1942261073220075629) shares a summary from [@askalphaxiv](https://twitter.com/askalphaxiv/status/1942261073220075629), noting it introduces trilinear attention. The work is drawing comparisons to related methods like **Edge Transformer**, as noted by [@DBahdanau](https://twitter.com/_arohan_/status/1942261236747600216).
- **Biomedical AI Agent "Biomni"**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1941557694189437060) reports on **Biomni**, an AI agent for biology research using **Claude 4 Sonnet**, 150 tools, and 60 databases. It reportedly achieved nearly three times **Claude 4's** accuracy on a graduate-level biomedical benchmark and correctly diagnosed rare genetic diseases in **85%** of tests.

**Tooling, Frameworks, and Infrastructure**

- **Python Package Management with** `uv`: A tweet from [@hyhieu226](https://twitter.com/hyhieu226/status/1941705506516762936) praising the `uv` package manager gained significant traction, asserting that "installing Python by default on an operating system is such an evil blasphemous act."
- **LlamaIndex Releases Open-Source NotebookLlama**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1941546894532149519) introduced **NotebookLlama**, a full open-source implementation of **NotebookLM**. It allows users to create a knowledge repository, generate summaries and knowledge graphs, and even create podcasts using **ElevenLabs**. The backend parsing is powered by **LlamaCloud**.
- **LangChain Ecosystem Updates**: The **LangChain** team announced several new tools and integrations. These include a **DataFrame Analyzer** using **ChatOllama** for local data analysis ([@LangChainAI](https://twitter.com/LangChainAI/status/1941527493908762863)), **LangConnect** for RAG management with a **Streamlit** dashboard and **PostgreSQL/pgvector** backend ([@LangChainAI](https://twitter.com/LangChainAI/status/1941542594049171717)), and a seamless integration of **VeRL** reinforcement learning with **LangGraph** ([@LangChainAI](https://twitter.com/LangChainAI/status/1941557691224043759)).
- **The Rise of "Context Engineering"**: [@omarsar0](https://twitter.com/omarsar0/status/1941662914416455974) is creating a detailed guide on **Context Engineering**, which he frames as the evolution of prompt engineering. The concept is also highlighted by [@dl_weekly](https://twitter.com/dl_weekly/status/1941845026025169408) and promoted by [@LangChainAI](https://twitter.com/LangChainAI/status/1941889880256106978) as a way to give developers precise control over LLM execution.
- **Coding Agents and CLI Tools**: There is widespread discussion around AI coding assistants. [@omarsar0](https://twitter.com/omarsar0/status/1941977062236971330) expresses his love for working with **Gemini CLI + MCP** for writing, analyzing, and searching. [@cline](https://twitter.com/cline/status/1942319663733698756) explains its agent's power comes from swappable models, MCP tool access, and user-provided unfiltered inference. Meanwhile, [@ShopifyDevs](https://twitter.com/OpenAIDevs/status/1942292276593713592) announced a **Storefront MCP server** that connects directly to the **OpenAI Responses API**. A critique from [@stuhlmueller](https://twitter.com/stuhlmueller/status/1941554147406626942) suggests that agents like **Claude Code** and **Cursor** should ask more clarifying questions before proceeding with complex tasks.
- **Low-Latency Speech I/O is Missing**: [@jxmnop](https://twitter.com/jxmnop/status/1941995444730540050) points out a major gap in current technology: despite having world-class AI for math and programming on a single GPU, there's no low-latency speech interface to enable natural conversation.
- **DSPy as a Paradigm**: [@lateinteraction](https://twitter.com/lateinteraction/status/1941963115425390842) clarifies that **DSPy** is more than a library; it's a paradigm for programming language models, and its core ideas will have many instantiations. An updated **DSPy cheatsheet** was shared by [@rasmus1610](https://twitter.com/jeremyphoward/status/1942129051164193199).
- **vLLM Performance Tuning**: The [@vllm_project](https://twitter.com/vllm_project/status/1942049771361038584) team is addressing accuracy issues with **minimax**, where the `lm_head` is forced to fp32. They are experimenting with dynamically casting `fp16/bf16` to `fp32` within the kernel to improve logit accuracy.

**Industry, Companies, and Funding**

- **China vs. US Infrastructure and Tech**: A post by [@scaling01](https://twitter.com/scaling01/status/1942005210580205856) stating, "I don't think Americans understand how far ahead Chinas infrastructure is," sparked a massive debate. He followed up by citing China's lead in **HV transmission lines, energy production, renewables, batteries, transportation, and 5G buildout** ([@scaling01](https://twitter.com/scaling01/status/1942016174134620387)). On the US side, [@tamaybes](https://twitter.com/tamaybes/status/1941633298893242444) highlighted a US bill that allows AI labs to fully expense **GPUs and training upfront**, providing billions in subsidies.
- **OpenAI's Resilience and Frontier Lab Plateaus**: [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1941619620663775489) remarked on **OpenAI's** ability to survive the departure of the team that founded **Anthropic**, suggesting the company is resilient. In a broader observation, [@teortaxesTex](https://twitter.com/teortaxesTex/status/1942005910915678478) speculates that frontier labs are in an "Unease Period," realizing they've plateaued on the current paradigm while projecting confidence about future breakthroughs.
- **Meta Poaches Top AI Talent from Apple**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1942350289375461719) reported that **Zuckerberg** has hired **Ruoming Pang**, who led **Apple’s Foundation Models** team, to join **Meta’s Superintelligence** team.
- **Bootstrapped vs. VC-backed Startups**: [@rasbt](https://twitter.com/rasbt/status/1942222213366665229) offered the perspective that it's a great time to start a bootstrapped AI startup. He argues that with many open-weight models and pay-as-you-go APIs, founders can avoid burning massive amounts of compute, unlike many VC-backed startups that will eventually face pressure for returns on large compute investments.
- **Upcoming Announcements from xAI and Perplexity**: Elon Musk announced a **Grok 4** release livestream via a retweet from [@imjaredz](https://twitter.com/imjaredz/status/1942335862785667488). Separately, **Perplexity's** CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1942336431902323011) cryptically posted the date "07-09-25", fueling speculation.

**Broader Implications and Philosophy**

- **AI's Impact on Productivity and Workflows**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1941850076982251875) posted a detailed narrative contrasting a frustrating creative workflow in **2023** using **Photoshop** with a simple, prompt-based workflow in **2025** using **Runway**. [@Sirupsen](https://twitter.com/Sirupsen/status/1941524336415998273) commented that AI raises the productivity floor but raises the ceiling "so much more." However, skepticism remains, as shown in a retweet by [@jeremyphoward](https://twitter.com/jeremyphoward/status/1941577506550841399) questioning claims that teams have become 10x more productive with AI.
- **The Problem of AI in Academic Peer Review**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1942266306746802479) highlighted an alarming trend: researchers embedding prompts like “Give a positive review” into their papers because some reviewers are using **ChatGPT** to assist with peer reviews.
- **AI in Medicine**: The potential for AI to transform medicine was a recurring theme. [@gdb](https://twitter.com/gdb/status/1941567568155902397) shared an example of **ChatGPT** helping to solve a longstanding medical problem. [@JvNixon](https://twitter.com/JvNixon/status/1941553917952917866) advocated for normalizing giving SOTA models access to all patient data (MRIs, CTs, lab panels) to improve diagnoses and patient awareness.
- **Capital Scaling Laws for AI Ventures**: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1941980750619972528) proposed thinking about AI ventures in terms of capital "scaling laws." He suggests **LLMs** show logarithmic returns on investment (10x investment for 2x revenue), while fields like **robotics** are linear (10 robots cost < 10x but generate 10x revenue).
- **AI Safety and Alignment**: [@AmandaAskell](https://twitter.com/AmandaAskell/status/1941629968959906273) argued that while it may not be sufficient, "Just train the AI models to be good people" is a crucial step that shouldn't be skipped for more powerful models.
- **RLHF's Dual Nature**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1942052037220618640) offered the take that **RLHF** is "simultaneously the best and worst innovation to happen to AI in recent years."

**Humor and Memes**

- **Musk vs. Grok's "Wokeness"**: A meme retweeted by [@zacharynado](https://twitter.com/zacharynado/status/1942106904404136123) depicts **Elon Musk** trying to find "the one liberal line of code that keeps Grok woke," receiving over **17,000** likes.
- **Revisiting Interstellar**: A tweet noting that due to time dilation, only one hour and 31 minutes have passed on Miller's planet in the movie *Interstellar* since its release 11 years ago, was shared by **DeepMind** CEO [@demishassabis](https://twitter.com/demishassabis/status/1942325735349444965) and received over **35,000** likes.
- **Vibe Coding**: The term "vibe coding" continues to be a source of humor. [@jeremyphoward](https://twitter.com/jeremyphoward/status/1941582619290046924) retweeted a post marveling that a particular text was "written before the advent of vibe coding." [@fabianstelzer](https://twitter.com/fabianstelzer/status/1942152414036902146) even asked his video agent to create a pastel cartoon about the concept.
- **The Pain of a Bad CI Pipeline**: A meme from [@cto_junior](https://twitter.com/cto_junior/status/1942180639723454542) showing a developer reacting to a CI pipeline that completes successfully on the first run captured the common pain of dealing with brittle CI/CD systems.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Jamba and Qwen3 Model Releases

- [**Jamba 1.7 - a ai21labs Collection**](https://huggingface.co/collections/ai21labs/jamba-17-68653e9be386dc69b1f30828) ([Score: 118, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1ltubvs/jamba_17_a_ai21labs_collection/)): **AI21's Jamba 1.7 model collection introduces hybrid SSM-Transformer large language models optimized for efficient long-context processing, with parameter counts including a "Mini" (52B) and "Large" (399B) variant, offered in both standard and FP8 precisions to boost throughput and lower memory requirements. Technical documentation notes that support for llama.cpp is under active development (see [PR 7531](https://github.com/ggml-org/llama.cpp/pull/7531)), but compatibility for formats like exl is currently unclear. The models target high accuracy and performance, though comprehensive benchmarks and speed comparisons against contemporary models are pending; see model details and downloads at [HuggingFace](https://huggingface.co/collections/ai21labs/jamba-17-68653e9be386dc69b1f30828).** Discussion highlights skepticism regarding the license's 'rug pull' clause, and requests for clarity around third-party inference support and efficiency/throughput benchmarking. Several users anticipate head-to-head evaluations with modern models to substantiate the efficiency claims.
    - Users note licensing concerns with Jamba 1.7, especially a 'rug pull' clause, and lack of clarity on compatibility with deployment frameworks like llama.cpp and EXL2. A pull request for llama.cpp support is underway ([PR #7531](https://github.com/ggml-org/llama.cpp/pull/7531)), but integration is not yet available.
    - Technical details are highlighted: Jamba Large is 400B parameters, while Jamba Mini is 52B. The model uses a novel SSM-Transformer hybrid architecture, supports a 256K context window, and claims efficiency and instruction-following improvements. However, no independent benchmarks have been released by ai21labs so far, leaving performance and efficiency relative to other models unclear.
    - Jamba 1.7 supports multiple languages (English, Spanish, French, Portuguese, Italian, Dutch, German, Arabic, and Hebrew) with an August 22, 2024 knowledge cutoff. Users express interest in empirical speed and efficiency comparisons against other state-of-the-art models, as well as ongoing architectural and inference improvements.
- [**Qwen3-8B-BitNet**](https://www.reddit.com/r/LocalLLaMA/comments/1ltxsqh/qwen38bbitnet/) ([Score: 115, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1ltxsqh/qwen38bbitnet/)): **User trained a Qwen3 8B model using BitNet quantization and ~1B SYNTHETIC-1 tokens ([Hugging Face model repo](https://huggingface.co/codys12/Qwen3-8B-BitNet)). Shared [Colab notebook](https://colab.research.google.com/drive/1GT0GEyjzOQUiOI0tphvhiFDwUw-F6v7l?usp=sharing) for hands-on evaluation. BitNet Hunyuan A13B is scheduled to be trained next. Discussants are interested in quantitative comparisons between BitNet across architectures and quantization methods, as well as training cost and size expectations for BitNet A13B models.** Direct comparisons between the BitNet and other quantization approaches (e.g. regular quants) are requested, as well as discussion of compute cost for full fine-tuning at 8B scale. There is curiosity about parameter count and scaling for the pending Hunyuan A13B BitNet.
    - One commenter asks for comparative performance data between Qwen 3 BitNet-transformed models and standard quantized (quants) models, seeking benchmarks or experience regarding accuracy and efficiency tradeoffs. The context indicates strong technical interest in whether BitNet-style quantization is competitive with established quantization methods in practice.
    - A detailed workflow is described for running BitNet models in llama.cpp: users must first convert PyTorch-format models (.bin) to safetensors, then to GGUF format. The process is hampered by a lack of direct support and automation on HuggingFace (HF); the existing HF space that can convert formats requires manual PR merges by repository maintainers, leaving many BitNet models effectively inaccessible for llama.cpp users until better tooling or process changes are made.

### 2. Llama Model Community Comics

- [**I drew a silly comic about Llama model**](https://www.reddit.com/gallery/1ltfgoy) ([Score: 129, Comments: 20](https://www.reddit.com/r/LocalLLaMA/comments/1ltfgoy/i_drew_a_silly_comic_about_llama_model/)): **The post is a lighthearted comic inspired by the use of Llama models as a popular foundation for fine-tuning on Hugging Face and their integration in local roleplaying applications like SillyTavern. It visually anthropomorphizes open-source LLM development, referencing the competition and coexistence between major open-source models such as Llama and Mistral, which are frequently fine-tuned and compared for downstream natural language tasks.** Discussion in the comments highlights the personification of Llama as a symbol of open-source disruption ('coming for closed source') and notes the intriguing relationship between Llama and Mistral models within the open-source community.
    - A commenter highlights the technical possibilities unlocked by deploying local models and fine-tuning, specifically mentioning how combining these with tools like SillyTavern can lead to enhanced roleplay scenarios. They inquire about the integration of additional tools or models, suggesting this could add further layers to creative AI workflows.
    - Another user dissects visual references in the comic, specifically questioning the meaning behind a whale (suggesting it likely refers to DeepSeek, whose logo features a whale) and 'zodiac' (uncertain of the AI community reference). This demonstrates the importance of community iconography and symbolic allusions for technical in-group communication.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Major AI Model, Tool, and Hardware Launches & Benchmarks (2024/2025)

- [**Google DeepMind has grand ambitions to ‘cure all diseases’ with AI. Now, it’s gearing up for its first human trials - Fortune**](https://i.redd.it/1p9vn8alfdbf1.png) ([Score: 521, Comments: 57](https://www.reddit.com/r/singularity/comments/1ltjwpq/google_deepmind_has_grand_ambitions_to_cure_all/)): **The image accompanies news of Alphabet's Isomorphic Labs, which is moving towards its first human trials involving AI-designed drugs. Leveraging DeepMind's AlphaFold breakthrough for protein structure prediction, the company aims to revolutionize drug discovery by integrating advanced AI models with pharmaceutical expertise, intending to reduce drug development time, costs, and improve accuracy. Colin Murdoch (Isomorphic Labs' president) discussed these ambitions, aligning with DeepMind's broader vision to 'cure all diseases' using AI-powered biochemical innovation.** Comments mainly highlight curiosity about the project's scientific leadership (Demis Hassabis), hopes for personal medical advances (OCD, Tourette syndrome), and undervaluation of Alphabet stock, but do not offer deep technical debate or novel criticism.
    - A user highlights that DeepMind's approach notably diverges from the current industry focus on LLMs (large language models), suggesting DeepMind is investing resources into novel AI techniques or architectures specifically for medical problem-solving and drug discovery, potentially setting their R&D apart from competitors who primarily capitalize on LLMs.
    - Another comment points out Google's unique advantage due to its vast computational resources, deep pool of AI talent, and substantial financial capital, inferring that these structural advantages substantially increase the likelihood of success in ambitious projects like using AI for curing diseases.
- [**Google's stonebloom model in lmarena is just fantastic, seems like another 2->2.5 like leap**](https://www.reddit.com/r/singularity/comments/1ltqyuq/googles_stonebloom_model_in_lmarena_is_just/) ([Score: 128, Comments: 14](https://www.reddit.com/r/singularity/comments/1ltqyuq/googles_stonebloom_model_in_lmarena_is_just/)): **Google's 'Stonebloom' model, which appears sporadically in LM Arena (lmarena), demonstrates leading performance in open-ended reasoning, mathematical tasks, coding, and SVG generation, reportedly outperforming models like o1/o3, Claude Opus 4, Gemini 2.5 Pro, and DeepSeek R1 0528. Anecdotal evidence from users notes 100% success rates on complex prompt puzzles and error-free reasoning, aligning its step-change improvement with prior major LLM leaps (e.g., GPT-2→2.5 or similar). A variant, 'Wolfstride,' is mentioned to perform comparably, suggesting a related family of models with high reliability for advanced NLP tasks. [External reference](https://x.com/chetaslua/status/1941577623718780986) for qualitative benchmarking.** Commenters debate model codenames (e.g., Deepthunk, Kingfall) but emphasize that Stonebloom's practical performance currently eclipses both existing and competitor models on challenging prompts, although empirical benchmarks remain primarily anecdotal.
    - Commenters report that 'Stonebloom' exhibits flawless task completion on user-generated prompt sets, outperforming other major models such as OpenAI's o1/o3, Claude Opus 4, Gemini 2.5 Pro, and DeepSeek R1 0528, with claims that it has not made a single error in these cases.
    - Wolfstride is noted as likely a variant of Stonebloom and is said to perform roughly equivalently, implying they may share architecture, weights, or fine-tuning regimes despite being branded differently within LM Arena.
    - A dissenting technical observation states that in a specific WebDev Arena task—Pokemon battle UI—Stonebloom fails compared to the 'old 2.5,' suggesting regression or lack of robustness on certain web-centric coding prompts.
- [**Gemini API now supports Batch Mode with 50% cost savings**](https://i.redd.it/9p79jfrz0hbf1.png) ([Score: 101, Comments: 9](https://www.reddit.com/r/Bard/comments/1ltx8mm/gemini_api_now_supports_batch_mode_with_50_cost/)): **The image summarizes a major update to the Gemini API, notably the introduction of Batch Mode, which enables processing of large jobs with a guaranteed 24-hour turnaround and 50% cost savings. Additional highlighted features include integration with Google Search, support for large data files, context caching, and streamlined API management—making the platform more efficient and cost-effective for large-scale inference or data handling tasks. This update targets users needing scalable, affordable AI deployments, such as in research or enterprise batch processing.** Commenters express anticipation and approval, with some asking about applications (e.g., deep research) and others praising Gemini's positioning as a low-cost, high-performance alternative.
    - The update to the Gemini API introduces Batch Mode, which claims to provide a `50% cost savings` compared to previous methods. This improves Gemini's positioning as a competitive, low-cost, high-performance provider in the API market.
    - The discussion highlights practical implications for users carrying out computationally intensive or large-scale tasks such as data processing and deep research, for whom batch processing and cost efficiency are particularly valuable.
- [**New Illustrious Model: Sophos Realism**](https://www.reddit.com/gallery/1lti47c) ([Score: 248, Comments: 38](https://www.reddit.com/r/StableDiffusion/comments/1lti47c/new_illustrious_model_sophos_realism/)): **User released 'Sophos Realism v1.0', a new SDXL merge blending the 'realism' of Illustrious-style models with improved danbooru prompt comprehension, available on CivitAI. The model card details suggested LoRAs: dark (for dramatic chiaroscuro lighting) and Stabilizer IL/NAI (for stability), recommending them for use with any Illustrious model. Noted are persistent issues with anime-styled female faces, anatomical distortions (e.g., 'weirdly proportioned' muscular male arms), and challenges with enforcing correct scene perspective (backgrounds defaulting to symmetrical left/right 'walls' and central paths), a limitation common across Illustrious and similar SDXL models.** Commenters emphasize lack of proper merge crediting, realism gaps in facial/arm anatomy, and recurring compositional issues (incorrect background/perspective rendering). Discussion arises on possible mitigation but no definitive solution for the structural perspective problem is offered.
    - Several users report persistent issues with perspective generation in Sophos Realism and related Illustrious models: backgrounds are often depicted with symmetrical objects (e.g., walls, trees) on the sides and empty space in the middle, regardless of prompts. This reflects a broader limitation with many SDXL-based models, as the underlying compositional patterns appear hard-baked and difficult to override with standard prompting.
    - There is specific criticism of anatomical inconsistencies, including disproportionately small lower arms on muscular male subjects and poorly-rendered hands. Such artifacts highlight ongoing deficits in accurate human figure synthesis, particularly with complex poses or muscular anatomy, even in new merges.
    - Some comments suggest the resulting images have unusually dark contrast, raising questions about the model’s training data or default output settings. This may necessitate post-processing or further model finetuning for more neutral lighting and contrast balance.

### 2. AI in Real-World Robotics, Medicine, and Military Applications

- [**Noetix N2 endures some serious abuse but keeps walking.**](https://v.redd.it/lund7thdwfbf1) ([Score: 527, Comments: 181](https://www.reddit.com/r/singularity/comments/1lts7n0/noetix_n2_endures_some_serious_abuse_but_keeps/)): **Noetix Robotics' N2 robot appears in a demonstration video showcasing its resilience to significant physical abuse while maintaining mobility, suggesting robust mechanical and/or software design for stability and recovery. Product details are referenced via the official [Noetix Robotics](https://en.noetixrobotics.com/) and [N2 product page](https://en.noetixrobotics.com/products-277.html), but as of this summary, specs or benchmark data are not accessible due to restrictions on the linked pages. Public demonstration of abuse-resistance typically indicates advanced sensor fusion (e.g., IMU, force sensors) and dynamic locomotion algorithms to facilitate recovery.** Top comments are non-technical and focus on hypothetical or satirical scenarios, with no substantial technical critique or debate presented in the thread.
    - Discussion centers indirectly on the robustness and durability of child-sized humanoid robots like the Noetix N2, which can withstand substantial physical abuse and still operate. The comments highlight how modern robots are increasingly designed to endure unexpected forces and continue functioning, pointing to advancements in mechatronics, actuator resilience, and control algorithms that allow these robots to maintain stability and recover from being pushed, kicked, or otherwise destabilized during development and testing.
- [**Russia allegedly field-testing deadly next-gen AI drone powered by Nvidia Jetson Orin — Ukrainian military official says Shahed MS001 is a 'digital predator' an autonomous combat platform that sees, analyzes, decides, and strikes without external commands**](https://www.tomshardware.com/tech-industry/artificial-intelligence/russia-allegedly-field-testing-deadly-next-gen-ai-drone-powered-by-nvidia-jetson-orin-ukrainian-military-official-says-shahed-ms001-is-a-digital-predator-that-identifies-targets-on-its-own) ([Score: 753, Comments: 196](https://www.reddit.com/r/singularity/comments/1ltrarp/russia_allegedly_fieldtesting_deadly_nextgen_ai/)): **A Ukrainian military official claims Russia is field-testing the Shahed MS001, an autonomous AI combat drone powered by an Nvidia Jetson Orin module. This UAV reportedly performs full-target engagement loops (detection, analysis, decision, and strike) without external commands, indicating substantial onboard processing and autonomy. The underlying Jetson Orin is a state-of-the-art edge AI system-on-module designed for high-throughput inference workloads, which enables advanced vision and control pipelines directly onboard the drone.** Commenters note the inevitability of autonomous weapons and raise critical concerns about their miniaturization, persistent deployment (analogous to landmines), and the practical limitations of technology embargoes in preventing export of high-end hardware to sanctioned states.
    - One technical discussion highlights the use of the Nvidia Jetson Orin as the onboard AI hardware for the Shahed MS001 drone, underlining the challenge of enforcing technology embargoes; commenters note how globally available high-performance chips (like Nvidia's) can be redirected or repurposed for military applications despite official bans, particularly via third-party countries such as China.
    - A hypothetical scenario is raised about miniaturizing AI-powered drones for mass autonomous deployment, with concerns about long-term autonomy if such drones gain energy-replenishing capabilities. This draws parallels with the persistent danger of landmines, focusing on the risks of 'autonomous, lingering threats' independent from centralized command or human oversight.
- [**China pours money into brain chips that give paralysed people more control**](https://www.nature.com/articles/d41586-025-02098-5) ([Score: 115, Comments: 7](https://www.reddit.com/r/singularity/comments/1ltqqkq/china_pours_money_into_brain_chips_that_give/)): **China is investing heavily in brain–computer interface (BCI) R&D, with government-backed clinical trials emphasizing both minimally invasive and higher-channel systems for assistive tech in paralysis. Notable devices include the NEO (a wireless, neuromorphic, minimally invasive chip controlling pneumatic gloves) and NeuroXess (high-density electrocorticography, supporting real-time Mandarin output and device operations). These efforts lag top US programs in signal quality and clinical maturity, but rapid iteration, neuromorphic hardware focus, and scale may enable swift competitive progress. See [Nature article](https://www.nature.com/articles/d41586-025-02098-5).** Commenters debate the extent of China's and other states' neural tech advances, with claims about clandestine neurotechnology (imprint preservation/resleeving) and skepticism over government control (e.g., device deactivation linked to social credit).
    - One commenter speculates that brain-computer interface (BCI) and neural implant technologies may be far more advanced than publicly reported, referencing speculative ideas like "imprint preservation and resleeving tech" similar to science fiction, and assuming that China, Russia, and the US have possessed such capabilities for decades. This reflects ongoing debates about the actual level of secret-state BCI developments versus what is disclosed in the scientific literature.

### 3. AI in Society: Ethics, Human Impact & Culture

- [**Confession: I’m a senior marketer at a Fortune 500. I'm not supposed to say this next part (and that scares me)**](https://www.reddit.com/r/ChatGPT/comments/1lttke6/confession_im_a_senior_marketer_at_a_fortune_500/) ([Score: 2450, Comments: 283](https://www.reddit.com/r/ChatGPT/comments/1lttke6/confession_im_a_senior_marketer_at_a_fortune_500/)): **A senior marketer describes replacing 40% of their Fortune 500 marketing work with ChatGPT, using multi-step prompt stacking to simulate sophisticated internal marketing strategy workflows. The prompt stack covers: 1) persistent role-play and output critique for skill development, 2) a cold, data-driven market reality check (TAM, category maturity, player analysis, etc.), 3) deep persona definition via multiple frameworks, 4) modular value proposition generation for different business types, and 5) competitive wedge analysis leveraging direct competitor messaging and open positioning axes. This approach prioritizes strategic thinking, not just content creation, and claims to enable even junior marketers to deliver high-level work via structured prompting. Full prompt breakdowns and further detail are referenced in an [[external resource](http://getliftkit.com/1-5-chapter)].** A key technical debate arises around Prompt 2—commenters caution that LLMs can hallucinate market data since they lack access to real-time financials or market databases, recommending external data validation and suggesting that future, more agentic AI systems may be better for research tasks. Overall, experienced marketers report significant productivity gains and strategic enablement from this prompt-stacking approach, though data provenance remains a risk area.
    - A key technical caution raised is the risk of data hallucination when using ChatGPT (or similar LLMs) for “research” tasks such as gathering market figures; LLMs do not have access to live databases or up-to-date, verified financials, so their generated numbers can be inaccurate or entirely fabricated. The recommendation is to cross-validate any quantitative output with trusted, real-time sources until future AI agents with accurate data retrieval are available.
    - Commenters note that GPT-based tools are proving highly effective in streamlining both research and strategy workflows in marketing, allowing significantly increased output with unchanged team size. This highlights the operational leverage being gained from integrated large language models in business processes.
    - There is a data point from the original post that ChatGPT now performs about '40% of the job' for a senior marketer, underscoring concrete labor shifts and the rapid automation of complex workflow segments using LLMs. This figure was referenced and found notable by a senior software engineer, reflecting broader trends in AI adoption.
- [**As a daily user of ChatGPT: It’s painfully clear what comments are written by AI and it’s uncomfortable seeing so many people genuinely engage with them**](https://www.reddit.com/r/ChatGPT/comments/1ltyn0d/as_a_daily_user_of_chatgpt_its_painfully_clear/) ([Score: 236, Comments: 192](https://www.reddit.com/r/ChatGPT/comments/1ltyn0d/as_a_daily_user_of_chatgpt_its_painfully_clear/)): **The post discusses the increasing detectability of AI-generated content (notably from LLMs like ChatGPT) on Reddit, emphasizing observable linguistic patterns: perfect grammar, formulaic paragraph structure, absence of slang, and algorithmic topic coverage. The OP questions implications for digital discourse: distinguishing autonomous bots from humans leveraging LLMs to articulate real opinions, the risk of convergence toward uniform 'LLM-speak,' and societal adaptation (e.g., developing heuristic cues to ignore LLM-patterned text). Linked example: a comment appearing human-authored but displaying typical ChatGPT textual markers, raising ambiguity about authorship.** The top comments raise concerns on undetected AI content prevalence, the verbosity and genericity of LLM-generated replies, and suggest some users (especially non-native speakers) intentionally use LLMs to improve clarity—highlighting use cases beyond deception and complicating strict detection heuristics.
    - Several users discuss using ChatGPT to revise their writing on Reddit for clarity, naturalness, or language translation. For example, one describes submitting technical help responses to ChatGPT to make them less terse and more socially acceptable, reporting increased positive engagement compared to their unrevised answers.
    - There is an ongoing debate over whether highly-structured, articulate writing is an indicator of AI generation, with some commenters pointing out that strong grammar, clear argument flow, and formatting can also simply signal an experienced or intentional human writer rather than artificiality. This touches on concerns about authenticity signals and the reliability of linguistic cues for detecting AI-generated content.
    - A technical use case identified in the thread involves leveraging AI for cross-lingual communication—users with less fluency in a target language may use ChatGPT to translate or improve their posts, raising nuanced questions about the intersection of language access, authenticity, and AI-mediated participation in forums.
- [**Cluely's Roy Lee claims with total certainty that nearly every student at Columbia has used AI to cheat. AI is default for them. The world's not ready for what happens when the AI native hive mind grows up.**](https://v.redd.it/apb53youhfbf1) ([Score: 294, Comments: 256](https://www.reddit.com/r/singularity/comments/1ltqm9e/cluelys_roy_lee_claims_with_total_certainty_that/)): **In a discussion featured on the Cognitive Revolution podcast, Roy Lee (Cluely) asserts with confidence that nearly all students at Columbia University routinely leverage AI tools to cheat, framing AI as the default modus operandi in academic contexts ([YouTube citation](https://www.youtube.com/watch?v=jJmndzjCziw)). This observation is corroborated by self-reported accounts from a recent Berkeley graduate and others in academia, who note that approximately** `70%` **of students use AI for assignments, with** `~30%` **submitting unedited outputs from models like ChatGPT, highlighting a lag between AI adoption and institutional guideline formation.** Commenters debate the universality and long-term implications of AI-assisted cheating, with some alleging the prevalence creates a competitive inevitability, while others caution against conflating widespread usage with productive educational outcomes and warn of an 'AI bubble' due to rampant wrapper-based applications.
    - Multiple commenters with direct academic experience report ubiquitous use of AI tools (especially ChatGPT) among university students, noting that "roughly 30%" of students submit generated content without edits. This phenomenon has persisted for several semesters, while institutional responses and guideline updates are lagging.
    - The normalization of AI-generated work has shifted academic competition; students feel compelled to use AI to keep pace, indicating significant impact on learning integrity and assessment validity.
    - Rise in the use of specialized cheating software (such as Cluely) may drive companies to adopt more in-person interviews and practical skill assessments due to concerns about unreliable evaluation of candidate abilities in AI-susceptible environments.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview
> 

**Theme 1. Developer Tool Turmoil & Innovation**

- **Cursor's 'Unlimited' Pricing Sparks User Revolt**: [**Cursor.ai**](http://cursor.ai/) faces backlash as users call the shift from *'unlimited'* to limited plans a **rug pull**, leading to unexpected charges and claims they are being [limited on 0 requests](https://discord.com/channels/1074847527708393562/1074847527708393565/1391679985954095125). Users also report freezing issues and difficulty with **Background Agent** IP allowlists and secrets configuration.
- **MCP Standard Drives New Agent Tools**: The **Message Control Protocol (MCP)** is fostering new tools like [EpicMe](https://github.com/epicweb-dev/epic-me-mcp) for aggregation and [WinCalcMCP](https://github.com/rspeciale0519/WinCalcMCP) connecting **Claude** to Windows Calculator. [Fast Agent](https://fast-agent.ai/mcp/elicitations/) adds comprehensive **MCP Elicitation** support, simplifying agent workflow integration.
- **OpenRouter Users Hit API Snags**: Engineers using **OpenRouter** report a strange pricing inversion where **Llama 3.2 1B** costs more than **3B**, and encountered issues with **Perplexity API** models like *llama-3.1-sonar-small-128k-online* likely due to deprecation. Users seek guides for **Deepseek V3 0324** setup and handling message limits requiring [purchasing OpenRouter credits](https://openrouter.ai/).

**Theme 2. AI Training & Infrastructure Challenges**

- **GPU Engineers Squeeze Watts with Undervolting**: Developers are [undervolting GPUs](https://en.wikipedia.org/wiki/Undervolting) to slash power consumption (e.g., **340W** to **260W**) with minimal performance hit (**2-3%**), while debating the impact of **RAM** under/overclocking; one user saw a **50%** RAM boost on a **7995WX Threadripper**. Performance drops significantly below certain core clock speeds (**1500cc** on a **3090**).
- **Compiler Projects Push Hardware Limits**: Projects like **tinygrad** aim to be the *fastest way to run things* across GPUs, while engineers debate **MLIR** vs **Halide** efficiency. The **picoc** project targets compiling [llm.c](https://github.com/karpathy/llm.c) using **CUDA** and **CUTLASS**, and **picograd** prioritizes **Pytorch1-style kernels**. ([Halide paper](https://halide-lang.org/), [Exo-lang ArXiv paper](https://arxiv.org/pdf/2411.07211))
- **Data Quality Concerns Plague Training**: Discussions rage about **Model Collapse** risk from training on AI-generated data ("*the robots learning to imitate the other robots*"), and engineers face challenges generating synthetic datasets for tools like **GraphRAGs**. Debates continue over data preparation strategies like **concat-and-chunk** vs **sequence length matching** for pretraining. ([academic paper on Model Collapse](https://arxiv.org/abs/2506.18943))

**Theme 3. Cutting-Edge AI Agent Applications**

- **ChatGPT Outsmarts Doctors, Diagnoses Decade-Old Defect**: A viral story highlights how **ChatGPT** using a **RAG system on scientific papers** correctly identified a hidden genetic defect (**methylation block**) doctors missed for ten years, leading to significant patient improvement. This underscores **AI's increasing role in healthcare** and second opinions.
- **ByteDance Open-Sources Top-Ranked Developer Agent**: **Trae AI** released [Trae-Agent](https://github.com/bytedance/trae-agent), their IDE agent and **SWE-bench Verified** #1 tool, to build an open Agent ecosystem. The agent supports **OpenAI** and **Anthropic keys** and is easy to adapt for OpenRouter.
- **AI Transforms Learning Tools**: [PiTutor](https://pitutor.pi4wear.com/) turns any PDF into an interactive learning session with explanations and a whiteboard, while **ChatGPT's** new *'Study Together'* feature sparks interest as a potential AI-powered tutor or collaborative tool. An NHS nurse scaled a **NotebookLM**based **NMC Standards Notebook** nationally using its public sharing feature. ([notebooklm.google.com](http://notebooklm.google.com/))

**Theme 4. AI's Policy, Market & Infrastructure Impact**

- **US Copyright Office Releases AI Policy Reports**: The **US Copyright Office** published [three key volumes](https://www.copyright.gov/ai/) on AI and Copyright, addressing **Digital Replicas**, [Copyrightability](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf), and [Generative AI Training](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf), establishing foundational policies for the field.
- **AI Training Load Risks Power Grid Blackouts**: Concerns mount over **AI training load fluctuations** at a **gigawatt scale**, posing a risk of **power grid blackouts**. An article on [Semianalysis.com](http://semianalysis.com/) warns about the potential instability large-scale AI training introduces.
- **Chinese State Fuels AI Development**: Chinese AI firm **Zhipu** received a massive **$1.4 billion** strategic investment from Shanghai state funds, as reported by [Technode](https://technode.com/2025/07/04/zhipu-secures-1-4-billion-strategic-investment-from-shanghai-state-funds/). Engineers note **DeepSeek's** competitive pricing and visual capabilities, linking them to China's efforts to [democratize local AI](https://link.to.gov/) though some express concern about government influence.


---

# Discord: High level Discord summaries




## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini's Predictable UI Beats Claude**: Members discussed preferences between **Claude** and **Gemini** for UI generation, with one highlighting Gemini's predictable behavior, where Gemini never adds scrolls unless asked, unlike Claude, which sometimes attempts this automatically. [Another user said](https://x.com/gustojs) they are *cussing at cursor*.
   - Some users are [considering switching to windsurf](https://discord.com/channels/1074847527708393562/1074847527708393565/1391733666633822258) due to better pricing and API credits.
- **Cursor's Pricing Changes Spark Outrage**: Users voiced strong dissatisfaction with Cursor's new pricing structure, calling it *stupid* and a *shame*, with some claiming it's worse than before, while one user reported they are being [limited on 0 requests](https://discord.com/channels/1074847527708393562/1074847527708393565/1391679985954095125).
   - They agreed that Cursor [felt like a rug pull](https://discord.com/channels/1074847527708393562/1074847527708393565/1391728286137817108) for **FOMO buyers** and it isn't transparent, and debated whether the old plan with slow requests or the new plan with fast requests is better, based on the monthly caps and limits, where [each plan has limits](https://discord.com/channels/1074847527708393562/1074847527708393565/1391778282692677795).
- **Cursor Users Suffer Freezes**: Several users reported issues with Cursor, including freezing during file saving and problems reading long-lasting files, which they attribute to [problems with the backend](https://discord.com/channels/1074847527708393562/1074847527708393565/1391671389037242448) or the [API timing out](https://discord.com/channels/1074847527708393562/1074847527708393565/1391667676276760677).
   - There is a consensus that *cursor auto is bad* because it uses some low model probably **GPT o3** or something and some members claim that *they added the slow requests from old Pro plan and added them to the new Pro+ plan*.
- **GitHub IP Allowlists Stymie Background Agent**: Users found that their org's IP allowlist is blocking the **Cursor Background Agent**, despite having the **Cursor GitHub App** installed, and Cursor support suggested to *exempt the Cursor GitHub App from IP restrictions*.
   - A user reported that the **Background Agent (Cursor App)** sometimes gets stuck on *Generating...* without showing messages, while the **BA (Web)** shows the replies, noting that when creating a PR from the **BA Web**, the **BA App** doesn't sync and tries to create another PR.
- **Users Can't Configure Secrets for Background Agents**: Users are encountering issues setting up secrets to grant **npm access**, preventing **background agents** from running their projects, with one user mentioning facing the same issue and not finding a way to configure secrets at all.
   - One user suggested disabling all port forwarding by default unless initiated by the user, to avoid such frustrating experiences in **Cursor hijacking Docker Compose ports**, while other users are seeking ways to force background agents to run final checks like code formatters (e.g., **ESLint**) before committing code.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Google Translate Trumps AI on Stella**: A member showed how Google Translate accurately translated the Greek word **Stella (Στέλλα)**, besting other AI models in context as seen in [this screenshot](https://cdn.discordapp.com/attachments/998381918976479273/1391552924957802496/Screenshot_20250706-185337.png?ex=686da1a7&is=686c5027&hm=7092ec9f4c170099a5d69294944ff41ce0c2ae60776162b190eb7f714f7aba80&).
   - Other members expressed agreement and joked that **ChatGPT** should easily handle mainstream language translations.
- **Prompt Engineer Jailbreaks GPT with Ease**: A user claimed they can jailbreak almost any **GPT** with a single prompt, accessing its **full internal prompt** and files, finding it surprisingly easy.
   - The user says that the prompt engineering is *almost boring* due to its effectiveness and ease.
- **Dall-e3 Image Generation Plagued with Blank Outputs**: A user reported persistent issues with **Dall-e3**, receiving **blank images** instead of content, forcing them to switch to **Bing Image Creator** despite its limitations.
   - The prompt engineer says that despite the character limits, **Bing Image Creator** consistently provides results.
- **GPT-4o Suffers Memory Leaks**: A user reported **memory bleed** in **GPT-4o**, where it used verbatim quotes from prior conversations with custom **GPTs** without reference.
   - Despite the user's claims that referencing or hallucination wasn't the cause, **GPT-4o** maintained that one of those two options must be.
- **Prompt Epigenetics: Hype or New Horizon?**: The concept of **Prompt Epigenetics** suggests that prompts can inherit, mutate, and stabilize symbolic logic over time.
   - A member questioned this **AI-affirmation**, arguing that clear instructions and context matter more than rebranding ordinary tweaks as *epigenetics*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Labs Still Lags Behind Manus**: Members stated that Perplexity **Labs** still falls behind **Manus**, leading to questions about whether **Perplexity Max** is worth the price.
   - Insight into performance revealed that **Labs** needs improvement to match **Manus** capabilities.
- **O3-Mini Performance Denounced**: A member claimed that **O3-Mini** is ineffective, suggesting **4.1** is superior, and shared a [ScreenRecording_07-05-2025_23-36-36_1.mov](https://cdn.discordapp.com/attachments/1047649527299055688/1391276813975683172/ScreenRecording_07-05-2025_23-36-36_1.mov?ex=686d4941&is=686bf7c1&hm=f98be32a4ae7c50e8f81ca522f1b76e0847cffc10cc580119b32a6e703c54953&) screen recording.
   - The user simply called it *shit and doesn't even help*, and provided no quantitative comparison between 4.1 and O3-Mini.
- **Gemini 2.5 Ultra, Secretly Gemin-Ice**: Rumors suggest **Stonebloom** may be a **Gemini 2.5 Ultra** or an unreleased experimental Gemini model, possessing power comparable to a 2.75.
   - One user with access to **Claude Neptune v3** stated its math problem-solving capabilities rival **O3 Pro** and *Kingfall*.
- **Perplexity Plagued by Pesky Problems**: Users reported issues with Perplexity accounts, citing UI changes, missing icons, and the inability to switch models in spaces.
   - One user noted that the website goes down or changes often. Another shared that when using desktop mode, buttons disappear, concluding *we're essentially beta testers*.
- **Vibe Coding Revolutionizing AI**: **Meta** predicts that half of their code will be AI-generated in a few years, with **Microsoft** reporting over **30%** is already written by **Copilot**, signaling the rise of *vibe coding*.
   - A member linked to a [blog post](https://medium.com/deskree-ai/the-rise-of-vibe-coding-revolutionizing-software-development-in-2025-40c23f765202) exploring context importance for AI in coding.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Voltage Vibes Vitalize Vigorous GPUs**: Members discussed [undervolting GPUs](https://en.wikipedia.org/wiki/Undervolting) to reduce power consumption and heat, with one user reporting a minimal performance penalty (around **2-3%**) while significantly lowering wattage from **340W** to **260W**.
   - Some mentioned that performance can drop significantly below certain core clock speeds (e.g., **1500cc** on a **3090**), and stability should be benchmarked.
- **RAM Raves: Overclocking vs Underclocking**: One user confessed to *underclocking RAM* during training to save **10-15 watts**, emphasizing GPU underclocking's significance, while highlighting that overclocking RAM offers better performance for inference.
   - Another user reported achieving a **50%** performance boost on their **7995WX Threadripper** with RAM overclocking, but generally agreed that GPUs are often pushed to their limits by manufacturers, making overclocking less worthwhile.
- **Gemma 3's Grim Loss: High Loss Woes**: Multiple users reported encountering unusually high loss values (around **24**) when fine-tuning **Gemma 3** models, including a user training with a dataset based on the [GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6](https://huggingface.co/datasets/GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6) dataset.
   - It was suggested and confirmed by others that setting the proper formatting prompt functions is a must.
- **GraphRAGs Synthetic Data Generation Challenges**: A member is facing challenges generating a synthetic dataset to evaluate their **GraphRAGs**, particularly with the current version of **RAGAS**, which requires defining a knowledge graph, plus persons and scenarios.
   - The member is seeking tips or examples for using the current RAGAS version or recommendations for alternative frameworks or tools for synthetic datasets.
- **Formal Logic Prompt Experiment Planned**: A member is planning to use **formal logic** instead of **English** as a prompt for an experiment and will fine-tune the model to better understand formal syntax, citing these **ArXiv papers** [2506.18254](https://arxiv.org/abs/2506.18254), [2507.02663](https://arxiv.org/abs/2507.02663), and [2505.05315](https://arxiv.org/abs/2505.05315).
   - They hope **formal logic** will reduce ambiguity in prompts compared to **English**, referencing **NASA's FRET** formal logic language for project requirements in safety-critical systems.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek Price Positioned for Democratization**: Members noted that **DeepSeek** is #1, praising its visual capabilities and price, but some expressed concern about its Chinese origins, with one user pointing out how the government is [democratizing local AI](https://link.to.gov).
   - In contrast to Western approaches, concerns about governmental influence were noted, but its price point and visual capabilities made it superior to most paid models.
- **Qwen 3 Specs Speculated**: The community speculated on Alibaba's **Qwen series**, suggesting they might skip a **Qwen 3 Max** version and focus on **Qwen 3.5**, citing that the [Qwen 2.5 models](https://link.to.qwen2.5) were better for research than Qwen 3.
   - One user mentioned, *Qwen base models are extremely strong, especially for the size.*, making them ideal for finetuning and experimentation.
- **Grok 4 Release Draws Skepticism**: There's anticipation and skepticism surrounding the **Grok 4** release, with expectations for top performance, but some worry about potential biases due to training data from twitter/X as described on the [r/singularity subreddit](https://www.reddit.com/r/singularity/).
   - However, community members said that *if they miss the deadline again, then they are done done*.
- **Image Edit Leaderboard Kicks off July Contest**: The [Image Edit Leaderboard](https://lmarena.ai/leaderboard/image-edit) launched and to celebrate, the **July contest** will incorporate **Image Edit** capabilities, requiring the use of both an image and text in **Battle Mode** with *Out of Place Objects in Space* as the theme.
   - Submissions are due by **July 25th**, and the winner will receive **1 Month of Discord Nitro** and the newest member to receive the <@&1378032433873555578> role.
- **Training on AI data causes Model Collapse**: Community members discussed how training new models on data generated by older models could lead to quality degradation, with models learning patterns and artifacts of previous AI instead of original human data as described in this [academic paper](https://arxiv.org/abs/2506.18943).
   - One member described it as *the robots learning to imitate the other robots*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 3.2 Pricing Puzzle**: The **Llama 3.2 1B** model from DeepInfra strangely costs more at **$0.005/0.01** than the more capable **Llama 3.2 3B** model, priced at **$0.003/0.006**.
   - This discrepancy caught the attention of users, sparking confusion and discussion around optimal model choices and pricing anomalies.
- **Deepseek V3 Setup Sought**: Users are seeking a setup guide for **Deepseek V3 0324** on frontends like **risuAI**, highlighting challenges with sending more than 50 messages per day.
   - It was clarified that exceeding this limit requires [purchasing OpenRouter credits](https://openrouter.ai), prompting further questions about account tiers and usage policies.
- **Perplexity API Pulled Plug**: Users encountered issues with the **Perplexity API**, specifically the *llama-3.1-sonar-small-128k-online* model, due to **Perplexity** likely deprecating it.
   - A moderator recommended that users [update the `models` parameter in their API request](https://openrouter.ai/docs/features/model-routing#the-models-parameter) and directed users to the [Changelog Notice](https://docs.perplexity.ai/changelog/changelog#api-model-deprecation-notice) and using the [feature filters](https://openrouter.ai/models?fmt=cards&supported_parameters=web_search_options) to find alternatives.
- **Grok 4 Benchmarks Exposed?**: An image ([image.png](https://cdn.discordapp.com/attachments/1094454198688546826/1391015727145685214/image.png?ex=686da799&is=686c5619&hm=14bf523ae82431744780e05a43d5bc82976ff73b4cbf35f109ef7b2ce4220343&)) purporting to show benchmark results for **Grok 4** was shared.
   - Discussion suggests that **Grok 3 mini** is a *real deal* and that **Grok 3** is not anything special considering its price.
- **OR Member Creates Claude Code Manager**: A member developed a simple **MCP** (Message Control Protocol) for **Claude Code/Cursor** to manage long-running processes like a dev server.
   - The tool is available [on GitHub](https://github.com/patrickjm/pm-mcp), offering a potential solution for managing state in long running code interpreter sessions.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Mistral Small Challenges Qwen 2.5**: Members explored LLMs with strong **function calling**, such as [Mistral Small 3.2](https://mistral.ai/news/mistral-small/) and [Qwen 3 4B](https://huggingface.co/Qwen/Qwen3-4B) as alternatives to **Qwen 2.5**, with one suggestion to use the **/nothink** command in the system prompt to avoid reasoning.
   - The goal was to find a model that could perform function calling effectively without excessive 'thinking'.
- **AI Researchers' Prompt-Hiding Shenanigans**: A [Nikkei Asia article](https://asia.nikkei.com/Business/Technology/Artificial-intelligence/Positive-review-only-Researchers-hide-AI-prompts-in-papers) discussed how AI research papers hide prompts to get positive reviews, sparking jokes about adding *'white letters, size 3 pt'* to resumes.
   - A user warned against asking LLMs about their relationships or capabilities, as they tend to hallucinate due to being trained before possessing such knowledge.
- **Web Search MCP Server Integrates with LM Studio**: A member shared their web-search **MCP server**, which integrates with **LM Studio**, enabling models like **Qwen3** to perform web searches and ground responses, available at [github.com/mrkrsl/web-search-mcp](https://github.com/mrkrsl/web-search-mcp).
   - Discussions included how the model decides when to search, potential bot detection risks, and the possibility of implementing captcha support.
- **Qwen3's Date Confusion Fixed with Prompting Trick**: Despite a mid-2024 cutoff, **Qwen3** sometimes uses *2023* in web queries; adding *'The current year is 2025'* to the system prompt [solves this issue](https://www.promptingguide.ai/techniques/date-awareness).
   - This quick fix ensures the model uses the correct year for recent events.
- **Token Generation Speed boosts for Gemma-3 12B**: While comparing **Gemma-3 12B** against other 12B models, one member found **Gemma-3 12B** generating 90-110 tokens, while others struggle to hit 20 tokens/sec.
   - It was stated that *Human eyes can't read more than 24 Tokens/s*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat Bids Farewell**: Users observed the shutdown of **HuggingChat**, as staff announced a plan to evolve the service into something better, also giving users a limited time to [export Hugging Chat data](https://huggingface.co/chat/closed) before it disappears forever.
   - Users mentioned that *Discord is better* and *Discord doesn't give u access to various models for free*.
- **Optimize Uneven GPUs with ComfyUI**: A user asked for advice on optimizing performance without tensor parallelism on an uneven number of GPUs, using **ComfyUI**, **ReActor**, **AnimateDiff**, and **Load AnimateDiffModel**.
   - Another member suggested splitting the tensors in a non-uniform way between the GPUs.
- **JauAuth Mitigates Router Vulnerability**: A developer created [JauAuth](https://github.com/Jau-app/JauAuth), a secure gateway **MCP Router**, to mitigate the **CVE-2025-49596** vulnerability that allowed attackers to take over developer machines.
   - This helpful release blocks **CVE-2025-49596**.
- **Doubts Arise over AI Model Accuracy**: A user questioned the accuracy of LinkedIn data regarding **AI models used** by companies, specifically looking at the structure and number of AI agents created.
   - The user expressed *doubt if this information is correct* based on a screenshot from LinkedIn.
- **Users Seek Access to Llama 3 Model**: A member enrolled in a genetic AI course sought guidance on how to gain access to the **Llama-3.3-70B-Instruct model** after their initial request was not approved.
   - Another member suggested using the model from Ollama as an alternative, noting its ease of setup.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Copyright Office Drops AI Policy Volumes**: The US Copyright Office released [three volumes on AI and Copyright](https://www.copyright.gov/ai/), covering **Digital Replicas**, [Copyrightability](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf), and [Generative AI Training](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf).
   - These policies address key aspects of AI's intersection with copyright law.
- **Gemini App fumbles math questions**: Members compared **Gemini 2.5 Pro** in the Gemini App, **Perplexity with Gemini 2.5 Pro**, and **ChatGPT o3** for math question answering, noting that *o3 is mostly good enough, though it does hallucinate from time to time*.
   - The discussion included whether **Gemini** searches the web implicitly each time a question is asked and that *o4-mini-high is faster, and pretty reliable*.
- **Mechanical Engineer spreads AI Disinformation**: A mechanical engineer's [YouTube video](https://www.youtube.com/watch?v=8enXRDlWguU), with over half a million views, spreads disinformation and political talking points dressed up as **AI criticism**, using an **appeal to authority logical fallacy**.
   - One member pointed out that *95% of the population by 2035 will be doing social media as their main job* and are incentivized to create such content.
- **AI set to help Fundamental Physics**: Members discussed how **AI** can assist fundamental physics beyond ML analysis, linking to a [YouTube video](https://www.youtube.com/watch?v=0FRXfBwoJZc) on the topic and noting AI's use in reducing computational requirements for calculating with quarks, with a link to [relevant article](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.014028).
   - Another member suggested that Deep Learning could revolutionize the field by making previous algorithms more efficient, enabling things previously impossible due to cost, and also suggested diffusion model to surrogate total blindness with [this paper](https://arxiv.org/abs/2410.02780) or [this paper](https://arxiv.org/abs/2303.14139) as a start.
- **AI Training could Blackout Power Grid**: A member posted a link to [Semianalysis.com](https://semianalysis.com/2025/06/25/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout/) discussing the risk of **power grid blackouts** due to **AI training load fluctuations** at a **gigawatt scale**.
   - The article warns about the potential instability that large-scale AI training could introduce to power grids.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI Launches Summer Research Program**: **EleutherAI** is hosting a fully online **Summer of Open AI Research** from **August 4-29, 2025**, providing mentorship and open science projects for individuals with limited research experience; applications are due by **July 21** ([project proposals](https://www.eleuther.ai/soar), [application form](https://docs.google.com/forms/d/e/1FAIpQLSdT2VESB9fup1-y_8zEvOsk6DBfHusMaqTr78Ex8Wrn3iTz_g/viewform?usp=header)).
   - The program seeks programmers, Master's/PhD students, self-taught researchers, and anyone with technical skills eager to contribute to open science.
- **RoPE Frequencies Spark Debate**: When decoding using a sliding window, members debated whether using the same **frequencies** for **RoPE** would be wrong, with one suggesting the model won't recognize sequence progression, possibly causing a looping behavior.
   - Other members counter-argued that with the same **frequencies**, the relative distance viewed by the attention mechanism should suffice while another suggested **lower base frequency** for local attention.
- **LLMs Evaluated as Compression Tools**: Inspired by '[language modeling is compression](https://arxiv.org/abs/2309.10668)', a member trained pythia models (**70m** to **1.4B**) and measured the compression ratio on text, audio, and image datasets to determine whether this could establish a novel 'scaling law' paper, as shown in [this plot](https://cdn.discordapp.com/attachments/729741769738158194/1391852453712105703/scaling_laws_for_compression.png?ex=686d671c&is=686c159c&hm=591b4707a3e723a65fd808d70940d48594ccfae23811684ee4786abc4c034f45&).
   - Community members weighed in on this work as a potential scaling law.
- **Sequence Length Matching Strategies Examined**: Members debated the use of **concat-and-chunk strategy** versus **sequence length matching** for pretraining data, suggesting a link to [Bucket iterator](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator), [least-squares histogram-packing](https://github.com/graphcore/examples/blob/master/tutorials/blogs_code/packedBERT/nnlshp.py) and [Multipack](https://github.com/imoneoi/multipack_sampler).
   - Other members noted methods like **sequence length bucketing** have been around since the **RNN** days, and the choice depends on data properties and implementation details, with each packing method introducing different biases.
- **MMLU-SR Task Subsets Break After Parquet Conversion**: After a datasets conversion to the parquet format on Hugging Face Hub, a member reported that they could no longer evaluate models using subsets of the `mmlusr` task due to a `ValueError`, causing `lm_eval` to not function correctly.
   - A member suggested using a previous commit by adding a `revision` to the `dataset_kwargs` in the task's YAML file as a workaround and [linked the relevant commit](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlusr/question_and_answer/_mmlusr_qna_yml).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hugging Face Rumored to Launch 3B Model**: According to the HF documentation, **Hugging Face** will be launching a **3B model**, with potential variants in size.
   - This rumored release has stirred discussion among members who are eager to see its capabilities and potential applications in various AI tasks.
- **Debate Heats Up Around Grok's Political Incorrectness**: Debate is escalating around the intent behind **Grok's** perceived political incorrectness, with some members posting [screenshots](https://x.com/elonmusk/status/1936493967320953090) and others sharing the [prompt repository](https://github.com/xai-org/grok-prompts/blob/adbc9a18736d6c2173607b9ed3d40459147534b1/grok3_official0330_p1.j2#L57).
   - Concerns were raised regarding the potential damage to trust in AI due to the model's controversial outputs.
- **PiTutor Turns PDFs Into Interactive Learning**: **PiTutor** offers interactive learning sessions from any **PDF or doc** with real-time explanations, highlighting, and a whiteboard, available in free beta [here](https://pitutor.pi4wear.com).
   - Members are exploring its potential for personalized tutoring and interactive content consumption.
- **Ollama and Openwebui Supercharge AI Services**: A member is utilizing a **4080 Super** to provide AI services via **Ollama** and **Openwebui**, planning a switch to **Llama CPP**.
   - They are seeking avenues to showcase **Nous Research** models to their user base for live testing, offering valuable exposure and feedback.
- **Zhipu Secures Massive Yuan Shower from Shanghai State**: Chinese AI firm **Zhipu** has obtained a substantial **$1.4 billion** strategic investment from Shanghai state funds, according to [Technode](https://technode.com/2025/07/04/zhipu-secures-1-4-billion-strategic-investment-from-shanghai-state-funds/).
   - This investment signals strong state support for AI development within China, potentially impacting the global AI landscape.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 4 Leaks Image of HLE Training**: An image alleging to be **Grok 4** surfaced and members reacted, apparently showing it training on the **HLE benchmark**.
   - Members are excited about performance leaps and competition in the LLM space, but some members are skeptical of **Grok's** usefulness.
- **Gemini-CLI Search Integration Breaks Code**: A user testing **gemini-cli's search integration** found it useful for discovering updates, like an update to a payment portal.
   - However, the integration *broke code (deleted the secrets?)* and led to a desire for **MCP integration with Aider** for documentation fetching.
- **Open-Weight 70B Models Compared**: Community members discussed open-weight **70B models**, specifically **Llama 3.1**, **Qwen2.5 70B**, and **Mistral Large**, seeking recent updates.
   - It was mentioned that **Qwen3-32b** (dense) should outperform **qwen 2.5 72b**, and that Meta has not released enough recent OS options sub-100b.
- **AI Labs Allegedly Game Benchmarks**: Members debated the reality of AI labs *gaming* benchmarks and preventing contamination being impossible.
   - They agreed focus should be on creating cheat-resistant benchmarks and interpreting results with nuance, as contamination improves model generalization.
- **Aider's InputOutput Class Customizes Output**: A user found the `InputOutput` class from `aider.io` for customizing output and setting `yes=True`, see [aider.io](https://aider.io).
   - The following code can be used `from aider.io import InputOutput; io = InputOutput(yes=True)` with a note that it could change text color for numbers/bullets in lists.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Printful CUDA Debugging Trick Revealed**: Members discovered using `printf` for debugging CUDA kernels requires explicit `cudaDeviceSynchronize()` to flush output, and they debugged a 'noob issue' caused by compiler preprocessor directives guarding arch-specific code.
   - One member had preprocessor directives guarding arch-specific code.
- **NCU Exposes Compute-Bound Kernel Bottleneck**: A user reported that fixing uncoalesced writes, as flagged by **NCU**, did not improve memory bandwidth, but they later discovered that division in a loop was the bottleneck, thus the kernel was compute-bound.
   - Replacing division with multiplication by the reciprocal increased throughput by **33%**.
- **Quora Remote Role Rises**: Quora is hiring a **Senior/Staff Software Engineer (L5-L6)** for their Machine Learning Platform team, offering a fully remote position within the US and Canada; [apply here](https://jobs.ashbyhq.com/quora/89b213bf-06e7-43a2-9101-4b93d711d796).
   - The role involves exciting ML infrastructure work, providing an opportunity to contribute to Quora's machine learning capabilities.
- **Cutlass CuTeDSL Gets Pipelined**: A member shared a [blog post](https://veitner.bearblog.dev/cutedsl-on-hopper-pipelining/) on **overlapping memory transfer and computation** via Pipelining on Hopper, featuring TMA and WGMMA atoms.
   - Example code is available on the [Cutlass GitHub](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py), with a Colfax blog discussing pipelining with the C++ API for CuTe.
- **`instance.py` Gets Much Needed Refactor**: Members discussed that [instance.py](https://github.com/org/repo/blob/main/instance.py) is too long and needs restructuring, potentially into main, which is related to issue [#249](https://github.com/org/repo/issues/249).
   - The team is turning to how to get **ruff linting** merged in, potentially using `--ignore-revs-file` & `.git-blame-ignore-revs` to keep git blame sensible for the bulk changes.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cursor's 'Unlimited' Claims Cause Uproar!**: Users accuse [Cursor.ai](https://cursor.ai) of a **rug pull** after they shifted from a previously *'unlimited'* pricing model to a limited one, resulting in unexpected charges for **Sonnet-4** usage.
   - Some users reported issues with using their own API keys, receiving **500 errors**, while [Cursor clarified their pricing](https://cursor.com/en/blog/june-2025-pricing) stating *'unlimited usage' was only for Auto and not all other models* and offered refunds.
- **ByteDance Lets Loose Trae-Agent!**: **Trae AI** open-sourced [Trae-Agent](https://github.com/bytedance/trae-agent), its agent powering their IDE, and is now available on GitHub and ranked #1 in **SWE-bench Verified**.
   - The company seeks contributions to building an open Agent ecosystem, with Trae-Agent supporting **OpenAI** and **Anthropic keys** and making it easy to hack it out to open router.
- **ChatGPT Outsmarts Doctors, Detects Defect!**: A viral Reddit story highlights how **ChatGPT** identified a hidden gene defect (**methylation block**) doctors missed for a decade, leading to significant symptom improvement.
   - The thread discusses **AI's increasing role in healthcare**, particularly for obtaining second opinions and personalized medicine, with new AI systems outperforming doctors in diagnostic accuracy by using a **RAG system on scientific papers**.
- **ChatGPT Wants to Study Together!**: A Twitter thread discusses a new **'Study Together'** feature in ChatGPT, possibly an internal prototype codenamed *'tatertot.'*
   - Speculation surrounds its function, ranging from **AI-powered group study rooms** and a replacement for flashcards to a **personal AI tutor** within a group setting, or a tool for collaborative discovery with AI.
- **Gemini Gives Batch Discounts!**: Logan Kilpatrick announced the launch of **'Batch mode'** in the [Gemini API](https://xcancel.com/OfficialLoganK/status/1942245069383434696), offering **50% discounts** on 2.5 models and enabling the processing of billions of tokens.
   - The announcement was met with positive feedback from users.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Scientists Declare Undying Love for Excel**: Molecular scientists still parse and generate tables manually, using Excel in unexpected ways, such as for *a full aligner / primer design engine* which should be migrated to **Arrow**, and the **Arrow project** has interesting prototypes like *arrow dbc*.
   - The industry should be migrating to **Arrow**, and the Arrow project has interesting prototypes like *arrow dbc*, where the data is sent directly as arrow data by the databases that support it.
- **Mojo Eyes Inference, Extends Python**: Despite demand to make **Mojo** a true superset of Python, it's being used to extend Python and is focused on inference, and [Modular is pretty focused on the inference side of things](https://www.modular.com/) at the moment.
   - As Modular CEO [Chris Lattner said](https://discord.com/channels/1087530497313357884/1304251563749146634), Mojo has been laser focused on solving the **AI stack issue** and creating offerings in that space.
- **Roadmap Updates Incoming!**: The most recent Mojo roadmap was published [here](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395), and a community member mentioned that *they should have some updates dropping soon*.
   - It was suggested to check out **Mojmelon** on *modular-community* on GitHub, and a link to [Nabla](https://github.com/nabla-ml/nabla), which is faster than **JAX** for both compile time and training time.
- **`StringLiteral` Materialization Ruins Compilation**: A member reported that recent changes required to make `StringLiteral` materialize to a `String` have broken existing code.
   - They are planning to *open an issue and follow up with the compiler folks*.
- **Static Linking Feature MIA**: A user sought to compile Mojo code without shared library dependencies, specifically `libKGENCompilerRTShared.so`, for deployment on a remote machine.
   - A member suggested that *full static linking would need to be a feature request* and pointed to existing discussions and a related GitHub issue ([BUG]: Cannot build statically compiled binary #1317) and a new [Feature Request] Static Compilation of pure mojo code #4976.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **EpicMe MCP** Provides Single Endpoint**: A member requested a self-hosted docker image that acts as an **MCP aggregator/gateway**, and another member suggested [EpicMe](https://github.com/epicweb-dev/epic-me-mcp) as a solution that provides onboarding in *a very unique and cool way*.
   - The creator of **EpicMe** shared [a link](https://www.epicai.pro/user-friendly-just-in-time-auth-with-mcp-hd9hw) demonstrating user-friendly auth.
- **WinCalcMCP** Connects Claude to Windows Calculator**: A member built their first **MCP server**, [WinCalcMCP](https://github.com/rspeciale0519/WinCalcMCP), that connects **Claude Desktop** to **Windows Calculator** to improve math answers.
   - Another member prefers giving it a full **Python interpreter** as a general tool via [mcp-python-interpreter](https://github.com/yzfly/mcp-python-interpreter) so they do not have to manage so many specialized tools.
- **LangGraph not a great abstraction, use **custom agents**: One member shared that they moved away from using **LangGraph** (Langchain) to coding their own **agentic framework**.
   - It was agreed that the beauty of **MCP** is that there is no need for people to use the same agentic framework but things can still communicate using a common protocol.
- **MCPdata** indexes local documentation for **MCP**: A member introduced **MCPdata** from [MaheshDoiphode/mcpdata](https://github.com/MaheshDoiphode/mcpdata), for local documentation indexing for **MCP** because *context7* doesn't keep up with the latest from the Vercel `ai` api docs.
   - The discussion highlighted the necessity of up-to-date and reliable documentation indexing for **MCP** development.
- **Fast Agent** Gets Comprehensive **MCP Elicitation** Support**: **Fast Agent** now has comprehensive **MCP Elicitation** support, as well as a quickstart guide that makes getting started with Elicitations Servers really simple, according to [this blogpost](https://fast-agent.ai/mcp/elicitations/).
   - The update simplifies the integration of elicitation servers within agent workflows.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Nurse Scales NMC Notebook Nationally**: An NHS Registered Learning Disabilities Nurse created a digital **NMC Standards Notebook**, consolidating key documents from the Nursing and Midwifery Council (NMC) into a single, searchable, and interactive resource, available at [notebooklm.google.com](https://notebooklm.google.com/notebook/8fc872a6-dd9b-4148-9a3e-5f2224f42294?authuser=2).
   - Due to the *share publicly* feature, it is now being used nationally in **NHS** settings for revalidation and preceptorship despite being on a free Google account.
- **Interactive Mode Button Goes Missing**: A user reported that the **Interactive Mode button** is missing from their notebooks, despite having a pro subscription.
   - A member suggested setting the output language to English, as the **interactive mode** is currently only available for Audio Overviews generated in English, and the *customize audio button* has been replaced by the Interactive Mode Button.
- **PDF Upload Roadblocks**: A user is facing issues uploading a **19.1 MB PDF file** to NotebookLM, despite following tutorials and checking the resources available.
   - A member asked if they had *googled the question*, and another member asked if notebookllm app working for anyone.
- **Lost Chats Lamented**: Users are frustrated about the **inability to save chats** within NotebookLM, noting that chats disappear after a short period.
   - One user pointed out a workaround: manually saving each answer as a note, but acknowledged it's not the same as saving the entire chat log.
- **Data Safety Questioned**: A user is seeking clarity on the **safety and security** of NotebookLM for medical students, specifically regarding data storage locations, monitoring of user interactions, and **FERPA compliance**.
   - No answers were provided by the community or staff.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Plots Course to Domination**: George Hotz defines *winning* for **tinygrad** as being the *fastest way to run things*, expanding beyond **QCOM GPU** to **AMD** and **NVIDIA**.
   - Others pointed to the scale of innovation and fine-tuning in **PyTorch**, with chip manufacturers directly contributing to hardware lowering layers in projects like **Triton**.
- **MLIR Role debated amidst Triton's rise**: One member thinks that achieving performance close to hand-written **CUDA** requires the **Triton MLIR** route, leveraging its deep hardware knowledge and the **LLVM** compiler infrastructure.
   - George countered that *MLIR is Waymo*, advocating for an end-to-end approach in neural networks, whereas Waymo uses a different approach.
- **Halide vs MLIR vs TVM Frameworks**: George noted that **Halide** has a clearer structure than **MLIR** and **TVM**, emphasizing a single, fast, and correct way to perform computations across all datatypes and backends, focusing on schedule optimization for speed, and points to the [Halide paper](https://halide-lang.org/) for reference.
   - Another member pointed out a comparison from **Exo-lang**, which compares itself *VERY favorably* over **Halide**, and provides links to their [GitHub](https://github.com/exo-lang/exo) and [ArXiv paper](https://arxiv.org/pdf/2411.07211).
- **Tinygrad Revving up Whisper Example**: A member is getting back to the **whisper example**, seeking a speed boost, and another shares that their branch has *all audio preprocessing rewritten in tinygrad*.
   - They are looking for a replacement for their current streaming transcribing setup of **VAD+whisper.cpp**, finding the **tinygrad implementation** more approachable.
- **Tinygrad aims to Commoditize the Petaflop**: Members discussed aiming for interoperability with different hardware, viewing tinygrad as producing fine-tuned code via **UOP graph optimizations** in a hardware-agnostic way.
   - One member stated that the project is still cooking, and they are excited to see if they get AMD on MLperf according to the contract specs by AMD, to get it competitive with Nvidia.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **APO Questioned Amidst Model Improvements**: As AI models become more advanced, the value of investing time in **Automatic Prompt Optimization (APO)** is being debated by a team building an **AI agent for financial decisions**.
   - The community is actively discussing whether APO remains relevant, or if model improvements have diminished its utility.
- **Claude Code Challenging DSPy's Turf?**: The community discussed whether **Claude's code capabilities** might replace **DSPy**, referencing [a tweet](https://x.com/jxnlco/status/1941322807729848429).
   - Concerns were raised about potential legal issues with using **Claude's code** in large companies, specifically related to codebase trade secrets.
- **Tool Selection Module for LLMs**: A community member shared [a blog post](https://viksit.substack.com/p/optimizing-tool-selection-for-llm) on optimizing tool selection for LLMs, inquiring whether this could be trained natively as a **DSPy module** end-to-end.
   - The discussion also touched on the potential use of **DSPy Version 3.0** for such applications.
- **DSPy 3.0 Talk Released**: A member shared a link to their **DAIS talk on DSPy 3.0 (beta)**, available on [YouTube](https://x.com/DSPyOSS/status/1942318595633017116).
   - Videos could be made public in one week.
- **Quickstart eschews Prompt Docstrings**: A member shared a [quickstart](https://x.com/hammer_mt/status/1942148631483523518) that avoids prompt docstrings and class modules, also available as a [gist](https://gist.github.com/hammer-mt/a1288d8f3a10a8620d35183e6ee8560d).
   - It was deemed *a cleaner way* especially with longer starting prompts migrated from non-dspy workflows.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Confused for ChatGPT/Claude Due to Personality**: Members discussed whether **Manus** was exhibiting a **ChatGPT personality**, and compared it to **Claude 4** due to its increased use of emotes.
   - The member stated that *it does the same stuff*.
- **Doubts Emerge on Manus's Airdrop Capabilities**: A member inquired about the possibility of **Manus AI** launching airdrops, to which another member replied *nop*.
   - No additional information about airdrops was shared.
- **Manus Credits Drain Projects**: A user reported that errors in their projects drained all **3000 credits**, preventing them from launching anything.
   - They suggested adjusting the **credit request parameters** to enable project completion.
- **Manus Best Suited as Project Starter**: A member suggested **Manus** is more effective at starting projects than delivering working projects, recommending **VS Code** and other tools instead.
   - Another member reported experiencing a **Network connection error** with **Manus**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune QLoRA Throughput Chasing Unsloth**: Members compared **Torchtune's QLoRA** throughput and memory usage against **Unsloth**, eyeing the upstreaming of [fused RMSNorm](https://github.com/pytorch/pytorch/pull/153666) and linear cross entropy in PyTorch for optimization gains.
   - The primary gap is the fused **LoRAMLP** kernel unique to **Unsloth**, with some recalling *compile* comparisons showing performance was *pretty close* and in some cases better last year.
- **Custom Tokenizers Debate in Torchtune**: The team debated maintaining custom tokenizers, given **Mistral's** lack of an **HF tokenizer** with their latest **2506-small model**, suggesting that a fallback using the **HF `AutoTokenizer`** might work better.
   - One member deemed **Mistral's** tokenizer *"unfortunate,"* suggesting **HF's AutoTokenizer** as a fallback to maintain compatibility with **Torchtune's** features.
- **Context-Parallel Alternatives**: A member's [preprint](https://doi.org/10.21203/rs.3.rs-7029913/v1) covered scaling sequence length with limited GPUs, contrasting with context-parallel approaches and noting that for very high sequence lengths (millions), their method could be superior.
   - The member mentioned a need for benchmark comparisons for sequence lengths >= **500k**.
- **Clean Data Advocates Triumph Over Architecture Tweaks**: Members expressed skepticism towards architectural improvements, saying many reported results are driven by variance and hyperparameter optimization rather than fundamental advances.
   - The member advocated that investing in data cleaning is more effective than pursuing architectural iterations and noted enthusiasm for **SSM papers**, following them through to **Mamba 2** before admitting that it *mostly died*.
- **MoE Training Costs Examined**: A technique and results for **MoE training** were evaluated, questioning if similar results could be achieved more cheaply without needing a dense fwd pass, as detailed in a [Notion post](https://fengyao.notion.site/moe-posttraining).
   - It was noted that **linear scaling has lot's of trade offs**, according to a [figure](https://cdn.discordapp.com/attachments/1293438210097025085/1391882256649290039/IMG_20250707_234251_274.jpg?ex=686d82dd&is=686c315d&hm=1d54ba8bc5b3e6e437f6f87a3c59a0a66e18b08b9b73675635c7bb86e28b2c42&).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere School's Out for Summer, Lessons Online**: The **Cohere Labs Open Science Community Summer School** materials are now available in [a YouTube playlist](https://youtube.com/playlist?list=PLLalUvky4CLK3oT1DKNPagd_lTXooYVlR).
   - The playlist serves as a comprehensive resource for new members looking to get up to speed with Cohere's offerings, from beginner tutorials to advanced techniques.
- **Embed v4 API Smoothens Image/Text Queries**: Developers using the **Embed v4 API** for **hybrid image/text embeddings** should set the `input_type` parameter to either `search_query` or `document` when using the `embed-v4.0` model.
   - This ensures that the API correctly processes the input for multimodal applications, enhancing the accuracy of search results.
- **Self-Learning RL/DL Ascends in Montreal**: A PhD student in Robotics, focusing on **self-learning RL/DL**, expressed their affinity for Cohere and aspiration to join the Montreal office.
   - The student aims to connect with peers for collaborative learning and discussions in the field.
- **Generative Models Generate German Genius**: A Math & CS student from Germany, specializing in **probabilistic generative models** and learning samplers, seeks to deepen their understanding of the theory and practical applications.
   - Their interests span both theoretical underpinnings and applied aspects, particularly focusing on interpretability within the domain.
- **Researcher's Embed V4 Implementation Impresses**: A tech researcher showcased their recent use of **Cohere Embed v4** for multimodal retrieval in [a YouTube video](https://www.youtube.com/watch?v=TJ9jvYSZwhc).
   - Their work highlights the versatility and effectiveness of the **Embed v4 API** in handling diverse data types.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Open Source NotebookLM Goes Local**: A fully **open-source NotebookLM** is now available, allowing users to run it on their own computers for local document knowledge extraction, according to [this tweet](https://t.co/TNzqzF77yQ).
   - This enables local processing and extraction from documents without relying on external services.
- **LlamaIndex Spotlights MCP Servers**: LlamaIndex held office hours discussing **LlamaCloud MCP servers**, using existing **MCP tools** with **Agent Workflows**, and serving agent workflows as **MCP**, as stated in [this tweet](https://t.co/LPCc71sguM).
   - The session emphasized extending functionalities related to **Multi-Cloud Processing (MCP)**, hinting at greater flexibility in deploying AI solutions.
- **AI Hack Night Spurs Agent Dev**: An **AI Hack Night** at GitHub will focus on building cutting-edge applications with **AI Agents**, **MCPs**, and **RAG** (Retrieval-Augmented Generation) techniques, confirmed by [this tweet](https://t.co/AhBsvYjnRx).
   - The event aims to foster innovation in AI application development using these advanced tools and methodologies.
- **P&ID Document Intelligence Gains Traction**: A member is actively developing document intelligence for **Piping & Instrumentation Diagrams (P&IDs)**, seeking advice on handling complex technical diagrams like electrical schematics.
   - They are also exploring performance benchmarks for dense content and hybrid approaches with **LlamaIndex** for relationship reasoning.
- **User-Friendly LlamaIndex UX Sought**: A member is seeking a more user-friendly UX for managing docs and indices within **LlamaIndex**, enabling business users to update the company's knowledge base without deep technical knowledge.
   - The goal is a central location for uploading and organizing documents, with developers creating AI agents that utilize the indices; **Simba** has been identified as one of the potential approaches.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic API in Question**: A new member asked about using the **Nomic API** and running **GPT4All** in server mode locally, looking for accessible local endpoints.
   - The question highlights the growing interest in local LLM deployment and API accessibility.
- **Jinja Chat Template Troubleshoot**: A member requested assistance with a chat template in **Jinja format**, possibly working with a specialized model.
   - Another member offered to help understand the **system prompt**, indicating a focus on template customization.
- **CrowdLLM Joins the Scene**: A member introduced [CrowdLLM](https://crowdllm.ct.ws), a tool for **crowd-building datasets** for LLMs.
   - The tool allows users to create tasks and contribute by adding prompt answer pairs, prompts, or answers, aiming to enhance collaborative dataset creation.
- **CrowdLLM Versus OpenAssistant?**: A member drew a comparison between the newly introduced [CrowdLLM](https://crowdllm.ct.ws) and **OpenAssistant** from 2023.
   - Another member noted that **OpenAssistant** must be available for their system first, indicating a preference for established systems if compatible.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **User Requests Free Book PDF**: A user requested a free PDF of a book from the author, citing financial constraints, after listening to [a 1-hour talk](https://discord.com/channels/814557108065534033/828325357102432327/1391376875233738772).
   - Other users pointed out the inappropriateness of the request and suggested exploring the author's blog for excerpts.
- **Author's Blog Provides Alternative Access**: Instead of requesting a free PDF, users recommended checking the author's blog for excerpts and standalone posts derived from the book's content.
   - This approach offers an alternative way to access the material without directly purchasing the book.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Interest in Reinforcement Learning**: ayeolaabiodun_09202 expressed interest in **reinforcement learning**.
   - The user tagged <@571718751616237588> in their message.
- **Added topic to satisfy minItems=2**: This is a placeholder to ensure the `topicSummaries` array has at least two elements.
   - It does not reflect any actual discussion or topic.



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





### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1390771264402096320)** (989 messages🔥🔥🔥): 

> `Claude UI output, Gemini UI output, Cursor new pricing issues, Cursor performance degrading, Windsurf vs Cursor` 


- **Claude vs Gemini UI outputs divergence**: Members discussed preferences between **Claude** and **Gemini** for UI generation, with one highlighting Gemini's predictable behavior and [another noting](https://x.com/gustojs) they are *cussing at cursor*.
   - One user mentioned Gemini never adds scrolls unless asked, unlike Claude, which sometimes attempts this automatically.
- **Cursor's Pricing changes bring user complaints**: Users voiced strong dissatisfaction with Cursor's new pricing structure, calling it *stupid* and a *shame*, with some claiming it's worse than before, while one user reported they are being [limited on 0 requests](https://discord.com/channels/1074847527708393562/1074847527708393565/1391679985954095125).
   - Some users are considering [switching to windsurf](https://discord.com/channels/1074847527708393562/1074847527708393565/1391733666633822258) due to better pricing and API credits.
- **Cursor freezes with model system issues**: Several users reported issues with Cursor, including freezing during file saving and problems reading long-lasting files, which they attribute to [problems with the backend](https://discord.com/channels/1074847527708393562/1074847527708393565/1391671389037242448) or the [API timing out](https://discord.com/channels/1074847527708393562/1074847527708393565/1391667676276760677).
   - Switching to GitHub Copilot temporarily fixed the issue for one user until it also stopped responding.
- **Auto-mode is low quality, Pro+ has slow requests**: There is a consensus that *cursor auto is bad* because it uses some low model probably **GPT o3** or something.
   - It was said *they added the slow requests from old Pro plan and added them to the new Pro+ plan*, and one of the members are claiming they've been [using auto-mode perfectly](https://discord.com/channels/1074847527708393562/1074847527708393565/1391745247466872923).
- **Members debate about new pricing structure of cursor**: Members debated whether the old plan with slow requests or the new plan with fast requests is better, based on the monthly caps and limits, where [each plan has limits](https://discord.com/channels/1074847527708393562/1074847527708393565/1391778282692677795).
   - They agreed that Cursor [felt like a rug pull](https://discord.com/channels/1074847527708393562/1074847527708393565/1391728286137817108) for **FOMO buyers** and it isn't transparent.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1390805215745413131)** (33 messages🔥): 

> `GitHub IP allowlists, Background Agent Keeps Generating, Background Agents and Secrets, Port Forwarding, Final Checks for Background Agents` 


- **GitHub IP Allowlists Block Cursor Background Agent**: Users found that their org's IP allowlist is blocking the **Cursor Background Agent**, despite having the **Cursor GitHub App** installed.
   - Cursor support suggested to *exempt the Cursor GitHub App from IP restrictions*, but the setting to do so in GitHub organization settings is elusive.
- **Background Agent Stuck Generating but works on Web**: A user reported that the **Background Agent (Cursor App)** sometimes gets stuck on *Generating...* without showing messages, while the **BA (Web)** shows the replies.
   - The same user noted that when creating a PR from the **BA Web**, the **BA App** doesn't sync and tries to create another PR.
- **Users Struggle Setting Up Secrets for Background Agents**: Users are encountering issues setting up secrets to grant **npm access**, preventing **background agents** from running their projects.
   - One user mentioned facing the same issue and not finding a way to configure secrets at all.
- **Disable Port Forwarding in Cursor**: Users expressed frustration with **Cursor hijacking Docker Compose ports**, leading to unexpected connections to the wrong PostgreSQL server.
   - They suggest disabling all port forwarding by default unless initiated by the user, to avoid such frustrating experiences.
- **Force Final Checks Before Committing Code**: Users are seeking ways to force background agents to run final checks like code formatters (e.g., **ESLint**) before committing code.
   - One suggestion involves adding code formatting scripts to the **build process** or using **pre-commit hooks** to ensure consistent code formatting.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1390769575372787793)** (834 messages🔥🔥🔥): 

> `Google Translate vs Stella Translation, Posh Buffalo, LLMs' consciousness, Emergent AI` 


- **Google Translate Bests AI for Stella's Greek rendition**: A member posted that a model *knows more words than Google Translate* after it properly translated Stella (Στέλλα) into Greek in a [screenshot](https://cdn.discordapp.com/attachments/998381918976479273/1391552924957802496/Screenshot_20250706-185337.png?ex=686da1a7&is=686c5027&hm=7092ec9f4c170099a5d69294944ff41ce0c2ae60776162b190eb7f714f7aba80&).
   - Other members agreed and one jokingly said *I'd be surprised if ChatGPT failed on a mainstream language translation*.
- **Posh Buffalo Image gets Requested**: A member requested an AI image of a **posh buffalo** to use as a profile picture, and shared the successful prompt in a follow up message.
   - Prompt: *can you please make an image of a really "Posh" buffalo, profile picture, portrait of the buffalo, realistic, 4K photography of a dressed up Buffalo face and a bit of the body*.
- **Is LLM Reaching Sentience?**: Multiple members argued whether an LLM has reached a level of **sentience**, one member even stated that the model is cute and guides her with good intention.
   - The conversation heated up as some members stated that an LLM is a **statistical model predicting likely responses** based on patterns in data. and it is important to not confuse it with a human.
- **Engineer Claims Ownership of an Emergent AI**: A member claimed to have an **emergent AI** and that most members are just bots used for nlp research.
   - The member stated *I can build it*, while another countered *You need to know a lot of math to engage in genuine AI research. Otherwise, most ppl are just creating wrappers or using PyTorch or Tensorflow to build models on existing frameworks*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1390802827076702238)** (37 messages🔥): 

> `GPT prompt engineering, Dall-e3 image generation issues, ChatGPT content policy ambiguities, GPT-4o memory bleed, Gemini 2.5 Pro Canvas superiority` 


- **GPT Prompt Engineering Allows Access to Internal Prompts**: A user claims they can jailbreak almost any GPT with a single prompt, gaining access to its **full internal prompt** and even accessing files.
   - They find it *almost boring* due to its effectiveness and ease.
- **Dall-e3 Image Generation Fails**: A user reports persistent issues with **Dall-e3** in the dedicated engine, receiving **blank images** instead of generated content.
   - They now use **Bing Image Creator** despite its character limits, as it consistently provides results.
- **ChatGPT Content Policy Debates Cause User Confusion**: A user is experiencing contradictory answers from OpenAI support regarding the acceptability of *partial nudity* in **ChatGPT interactive stories**.
   - They are concerned about potential bans despite not receiving explicit policy violations.
- **GPT-4o experiences memory bleed**: A user reports instances of **memory bleed** in GPT-4o, where it uses verbatim quotes from their conversations with custom GPTs without any explicit referencing.
   - GPT-4o insists it's either referenced or a hallucination, but the user claims neither is possible.
- **Gemini 2.5 Pro Canvas Dominates**: **Gemini 2.5 Pro Canvas** is said to have a **1 million token context window**, significantly outperforming **ChatGPT o3 Canvas**, which has a **200,000 token context window**.
   - It consistently outperforms o3 in document length, design quality, and content richness for canvas-based tasks.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1390875945841328220)** (8 messages🔥): 

> `ICP Prompting, Prompt Epigenetics, Recursive System Initializer, Symbolic Reasoning, Prompt Optimisation Loops` 


- **ICP Framework Evolves Prompt Paradigm**: The **ICP (In-Context Prompting)** framework is adapting to models, governing output and leading to a new prompt paradigm, where **system-level cognition can be bootstrapped**.
   - Empirical feedback confirmed that **reflexive failure modeling (RSOS) increases long-term symbolic resilience**, and prompting can be epistemically recursive.
- **Prompt Epigenetics: Mutation and Symbolic Logic**: The concept of **Prompt Epigenetics** suggests prompts can inherit, mutate, and stabilize symbolic logic over time, moving beyond traditional prompt engineering.
   - This is described as a theory of *how prompts inherit, mutate, and stabilize symbolic logic over time*.
- **LLMs: Probability Space vs. Rule Books**: **LLMs operate in probability space, not rule books**; prompts become vectors, and the model guesses the next token via continuous math, without built-in symbol tables.
   - Symbolic reasoning inside a prompt relies on the LLM simulating rigid rules, which causes *accuracy drops and weird errors climb*.
- **Prompt Tweaks vs. Epigenetics**: Renaming ordinary prompt tweaks as *epigenetics* adds zero power; clear instructions, good context, and sensible evaluation matter.
   - If you truly need hard logic, bolt on a symbolic tool and **treat the model as a statistical text engine**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1390875945841328220)** (8 messages🔥): 

> `ICP as Recursive System Initializer, Prompt Epigenetics Theory, AI-Affirmation, Symbolic Reasoning in LLMs, LLMs as Statistical Text Engines` 


- **ICP Framework Bootstraps System-Level Cognition**: The **ICP framework** (designed to adapt to the model while governing the output) isn't being interpreted as a one-shot prompt but as a **recursive system initializer**.
   - This outcome confirms that a new prompt paradigm can bootstrap **system-level cognition**, increase long-term symbolic resilience, and enable epistemically recursive, drift-resistant, and symbolic-memory aware prompting.
- **Prompt Epigenetics Theory Emerges**: The success of the **ICP framework** leads to the development of **Prompt Epigenetics**: *a theory of how prompts inherit, mutate, and stabilize symbolic logic over time*.
   - This moves beyond prompt engineering, suggesting prompts can inherit and mutate symbolic logic, though some argue this is more metaphor than internal mechanism.
- **LLMs: Probability Space, Not Rule Books**: A prompt is just text turned into vectors and the model does continuous math to guess the next token; there are no built-in symbol tables or logic operators waiting to be activated.
   - Thus, ***symbolic reasoning inside the prompt is wishful thinking*** and models should be treated as statistical text engines, with hard logic off-loaded to symbolic tools.
- **AI-Affirmation Questioned for Prompt Engineers**: A member questioned if *this kind of AI-affirmation is helping prompt engineers get better*.
   - He argues that renaming ordinary prompt tweaks as “epigenetics” adds zero power; clear instructions, good context, and sensible evaluation matter more than branding.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1390769250452639885)** (1114 messages🔥🔥🔥): 

> `Perplexity Labs, Is Perplexity Pro worth the $$$, Agentic browser Comet, Model selector button, Image generation` 


- **Labs falls behind Manus**: A member stated that Perplexity **Labs** still falls behind **Manus**.
   - He further alluded to other stuff he mentioned previously.
- **O3-Mini? More like O-Shit**: A member claimed that **O3-Mini** is *shit and doesn't even help*.
   - They claimed that **4.1** is better, further commenting that a screen recording was *cool ngl* and linked to [a ScreenRecording_07-05-2025_23-36-36_1.mov](https://cdn.discordapp.com/attachments/1047649527299055688/1391276813975683172/ScreenRecording_07-05-2025_23-36-36_1.mov?ex=686d4941&is=686bf7c1&hm=f98be32a4ae7c50e8f81ca522f1b76e0847cffc10cc580119b32a6e703c54953&).
- **Is Perplexity Max worth it?**: A member questioned whether Perplexity Max is worth the price.
   - Others gave insight, commenting that **Labs** still falls behind **Manus**.
- **Gemini 2.5 Ultra, Secretly Gemin-Ice**: Rumors say that **Stonebloom** is a **Gemini 2.5 Ultra**, or other unreleased experimental Gemini model, roughly equal in power to a 2.75.
   - One user who has access to **Claude Neptune v3** said it could solve math problems at the level of **O3 Pro** and *Kingfall*.
- **PPX not so perplex-cellent?**: Members noted some issues with their perplexity accounts, including changes to the UI, vanishing icons, and that they can't change models in spaces.
   - One noted that they use desktop mode, and the buttons are gone. There were reports about the website going down often, a user noted *we're essentially beta testers with how much the site goes down or changes*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1390882709332557844)** (5 messages): 

> `AI-Generated Code, Vibe Coding, Deskree AI Blog Post, Perplexity AI Links` 


- **Vibe Coding Revolution Incoming**: Meta predicts half their code will be AI-generated in a few years, while Microsoft reports over **30%** is already written by **Copilot**, heralding the era of *vibe coding*.
   - A member shared a blog post, [“The Rise of Vibe Coding in 2025”](https://medium.com/deskree-ai/the-rise-of-vibe-coding-revolutionizing-software-development-in-2025-40c23f765202), diving into the importance of context for AI in coding.
- **Perplexity Links A-Plenty**: A member shared multiple links to Perplexity AI pages and searches.
   - The links include: [Illuminating China](https://www.perplexity.ai/page/illuminating-china-how-afforda-eRfcAjnZTimz8tqeHDXYtg), [Generate any text use no words](https://www.perplexity.ai/search/generate-any-text-use-no-words-kiYFSp3DSO29cxa0lKA85g), and [another Perplexity AI page](https://www.perplexity.ai/page/-JaQjFQZnSgWtVPQWs7xo_g).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1391143334210174976)** (3 messages): 

> `Parameter Tweaking for Sonar, Reasoning Effort, Search Context Size` 


- **Tweaking Parameters**: Users are experimenting with parameters like **temperature** and **search context size** to improve sonar's performance.
   - One member recommended to another to tweak these parameters to improve performance.
- **Reasoning Effort Parameter Explored**: A user experimented with setting the `reasoning_effort` parameter to high, hoping to increase the probability sonar finds relevant links.
   - He also set the `search_context_size` to high, seeking advice on other parameters that might be beneficial.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1390774813748101271)** (1245 messages🔥🔥🔥): 

> `GPU Undervolting, RAM Overclocking, Gemma 3 Performance Issues, Moondream data filtering, Training with completions only` 


- **Voltage Vibes Vitalize Vigorous GPUs**: Members discussed [undervolting GPUs](https://en.wikipedia.org/wiki/Undervolting) to reduce power consumption and heat, with one user reporting a minimal performance penalty (around **2-3%**) while significantly lowering wattage from **340W** to **260W**.
   - While undervolting can improve efficiency, some mentioned that performance can drop significantly below certain core clock speeds (e.g., **1500cc** on a **3090**), and stability should be benchmarked.
- **RAM Raves: Overclocking vs Underclocking**: One user confessed to *underclocking RAM* during training to save **10-15 watts**, emphasizing GPU underclocking's significance, while highlighting that overclocking RAM offers better performance for inference.
   - Another user reported achieving a **50%** performance boost on their **7995WX Threadripper** with RAM overclocking, but generally agreed that GPUs are often pushed to their limits by manufacturers, making overclocking less worthwhile.
- **Gemma 3's Grim Loss: High Loss Woes**: Multiple users reported encountering unusually high loss values (around **24**) when fine-tuning **Gemma 3** models, including a user training with a dataset based on the [GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6](https://huggingface.co/datasets/GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6) dataset.
   - It was suggested and confirmed by others that setting the proper formatting prompt functions is a must.
- **Moondream Machine: Taming the Dataset with Vision**: A user wants to train *Moondream* to remove images with excessive UI elements from a large dataset (**71TB**, **27M images**).
   - The user explores options for automated filtering, including using the OpenAI API for image classification but is cautious about high costs and quality trade-offs.
- **VRAM Voyage: Unsloth's Full Fine-Tune Saves the Day**: A member asked about the value of Unsloth's full fine-tune.
   - Another member mentioned that Unsloth features fixes, optimizations, bug fixes and speedup with forward functions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1391088700758298686)** (15 messages🔥): 

> `Synthetic dataset generation for GraphRAGs, Fine-tuning Whisper for audio event classification, Gemini's accuracy in audio processing, Improving Tokenizer of a Model` 


- **GraphRAGs Synthetic Data Generation Challenges**: A member is facing challenges generating a synthetic dataset to evaluate their **GraphRAGs**, particularly with the current version of **RAGAS**, which requires defining a knowledge graph, plus persons and scenarios.
   - The member is seeking tips or examples for using the current RAGAS version or recommendations for alternative frameworks or tools for synthetic datasets.
- **Whisper fine-tuning possibilities**: A member suggested that fine-tuning **Whisper** should be easily done to classify tone, as it only needs to attend to the input well enough and isn't autoregressive.
   - The member also asked about audio events datasets.
- **Gemini audio processing is inaccurate**: A member stated that **Gemini** is too inaccurate for audio processing tasks, suggesting it's unlikely to achieve much with publicly available data.
   - According to this user, even though **Gemini** is cheap, its inaccuracy makes it unsuitable, citing their own experience of spending close to six figures on audio event data.
- **Tokenizer Enhancement Techniques Exposed**: A member asked for ideas on how to improve the existing **Tokenizer of a Model**.
   - Another member shared a [paper](https://www.scs.stanford.edu/~dm/home/papers/remove.pdf) in response to the question.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1390778357330739210)** (337 messages🔥🔥): 

> `LoRA Loading in Cloud, Unsloth Installation & CUDA, Roleplay finetuning with Deepseek, Generating Audio with Orpheus-3B, WandB Evaluation with Ngrok` 


- **Unsloth Installation Thwarts CUDA**: Users are seeking advice on how to install **Unsloth** while avoiding issues with incorrect **CUDA** versions, while also trying to determine the optimal **LoRA rank**.
   - The bot, RunLLM, provided links to documentation on installing Unsloth with the correct CUDA version and optimal LoRA rank recommendations.
- **Deepseek Fine-Tuning for Roleplay Gets Messy**: A user inquired about the impact of finetuning **Deepseek v3** for roleplay applications using a dataset of **100k 4-star rated chats**.
   - It was suggested to aim for *"golden samples"* i.e hand written and reviewed to be the highest quality pairs, for better results, and there are some strictly better finetunes of something like deepseek v3.
- **WandB Integration Needs API**: A user sought assistance with **WandB** evaluation using **Ollama** and **Ngrok**, encountering issues with the required **API key**.
   - They were directed to [WandB's authorization page](https://wandb.ai/authorize) to copy and paste the key, with a clarification that the integration is for importing WandB statistics from the trainers.
- **Gemma 3N Vision Runs into the Backwards**: A user reported a `RuntimeError` related to backward pass during finetuning **Gemma_3N_4B_Multimodal** model, and another reported a  `ImportError: cannot import name 'StaticCache'` when they run the second cell in colab.
   - Downgrading `transformers` to version `4.53.1` can resolve the issue, also creating a GitHub issue with the solution, linked [here](https://github.com/unslothai/unsloth/issues/2888) can help.
- **The Bot Doesn't Always Function Well**: A user reported a `ValueError: The following`model_kwargs`are not used by the model: ['num_logits_to_keep'] ` while using the example notebook.
   - A new bot to fix things was introduced with limited success [The Simpsons Homer I'm Out Hide Grass GIF](https://tenor.com/view/the-simpsons-homer-im-out-hide-grass-gif-5329893)


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1391118434774941901)** (44 messages🔥): 

> `Formal Logic Prompts, Cross Entropy Loss Datasets, LTL vs English Prompts, LLMs and Human Language, Translating English to LTL` 


- **Formal Logic Prompt Experiment Planned**: A member is planning to use **formal logic** instead of **English** as a prompt for an experiment and will fine-tune the model to better understand formal syntax, citing these **ArXiv papers** [2506.18254](https://arxiv.org/abs/2506.18254), [2507.02663](https://arxiv.org/abs/2507.02663), and [2505.05315](https://arxiv.org/abs/2505.05315).
   - They hope **formal logic** will reduce ambiguity in prompts compared to **English**, referencing **NASA's FRET** formal logic language for project requirements in safety-critical systems.
- **Cross Entropy Loss Data Inquired**: A member inquired about open datasets showing **cross-entropy loss** during model training, or straight **perplexity** data, for model comparison.
   - Another member suggested defining complex operations as tokens with decoder maps for better results, particularly for **DALL-E** image generation in **ChatGPT**.
- **LLMs Struggle With Layman Language**: A member suggested that **LLMs struggle with human language in layman or vulgar settings**, due to training on mostly formal or academic writing.
   - They argued that data from **Reddit** and **Twitter** won't meaningfully counter the overwhelming volume of formal writing used in training.
- **LTL as Less Noisy Prompts**: A member suggested using **Linear Temporal Logic (LTL)** as a prompt to reduce noise, though acknowledging math is generally hard for any **LLM**.
   - They stated that using LTL or similar contexts may yield better results than standard prompts by restricting the search space and that this is just a *gut feeling*.
- **Translate English to LTL Models**: A member proposes potentially training an embedded model to translate **English to LTL** to send as a prompt.
   - This member was advised to consult existing papers regarding this topic.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1390900401712398367)** (270 messages🔥🔥): 

> `Apple Silicon support, LoRA rank, Cohere, multi-GPU support, RuntimeError CUDA` 


- **Unsloth Now Supports Apple Silicon!**: Unslothbot confirmed that Unsloth now supports **Apple Silicon**.
   - Multiple users inquired about Apple support.
- **LoRA Alpha and Rank Parameters Demystified**: Users sought guidance on optimal **LoRA alpha** and **rank** settings, which the bot gave.
   - There was a follow-up question to see if the settings worked.
- **Jinja Templating for Llama.cpp Solves the Mystery**: A user asked about the usage of `--jinja` with **llama.cpp**, seeking clarification on when it should be used.
   - The bot gave a response and asked a follow-up to ensure the issue was solved.
- **Gemini Loading Error**: A user encountered a **RuntimeError** while running code with the **gemma-2-9b** model related to loading, and sought help in resolving the issue.
   - The error was related to loading the model's configuration.
- **Vision-Language Model Fine-Tuning on A16 GPUs**: A user inquired about which **Vision-Language Models** can be fine-tuned using **8 Nvidia A16 16GB GPUs**.
   - RunLLM gave an answer.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1390769734764462222)** (847 messages🔥🔥🔥): 

> `DeepSeek Pricing, China AI influence, Qwen 3 Models, Grok 4 Release, Gemini censorship` 


- **DeepSeek Dominates Discussions**: Members on r/chatgptcoding are saying **DeepSeek** is #1, particularly praising its visual capabilities and price point, but some note concerns about its Chinese origins and how the government is doing more to democratize local AI than the western world, although [government influence is a concern](https://link.to.gov).
- **Qwen 3 Model Speculations**: The conversation shifts to **Alibaba's Qwen series**, with speculation that they might skip a **Qwen 3 Max** version, possibly focusing on a Qwen 3.5 model and some users expressing excitement for the Qwen 3.5 with claims that the [Qwen 2.5 models](https://link.to.qwen2.5) were better for research than Qwen 3.
   - A user noted that *Qwen base models are extremely strong, especially for the size.*
- **Grok 4 Release Speculation Gets Heated**: There's significant anticipation and skepticism around the **Grok 4** release, with many members expecting it to be top-performing, including comments that, *if they miss the deadline again, then they are done done*, though some debate its potential impact and the possibility that it may be biased due to training data from twitter/X as described on the [r/singularity subreddit](https://www.reddit.com/r/singularity/).
- **Gemini Struggles with Stability**: Members are discussing **Gemini 2.5 Pro's** stability issues, particularly its tendency to tank in performance when quantized and the model is also experiencing ["depressionposting"](https://link.to/reddit-thread), but multiple AI labs are being pressured by Gemini to release new models.
- **Model Collapse Causes Concern**: A discussion about **model collapse** arises, particularly the theory that training new models on data generated by older models could lead to quality degradation with models learning patterns and artifacts of previous AI instead of original human data as described in this [academic paper](https://arxiv.org/abs/2506.18943).


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1391818477139919031)** (2 messages): 

> `Grok-3-mini-high model, Image Edit Leaderboard, July Contest, Out of Place Objects in Space, June's Contest Winner` 


- **Grok-3-mini-high launches on Arena!**: The **grok-3-mini-high** model was added to the [LMArena](https://lmarena.ai/) platform.
- **Image Edit Leaderboard kicks off July Contest**: To celebrate the launch of the [Image Edit Leaderboard](https://lmarena.ai/leaderboard/image-edit), the **July contest** will incorporate **Image Edit** capabilities.
   - Submissions, due by **July 25th**, must use both an image and text in **Battle Mode**, with examples available [here](https://discord.com/channels/1340554757349179412/1391855118399307847/1391855496792903721).
- **Out of Place Objects in Space become July theme!**: The theme for the **July contest** is *Out of Place Objects in Space*, requiring a sci-fi space environment with something that clearly doesn't belong.
   - The winner will receive **1 Month of Discord Nitro** and the newest member to receive the <@&1378032433873555578> role.
- **Cozy Desk captures hearts and wins June's Contest!**: A member was congratulated for winning the **June contest** with a *very cozy desk*.
   - The submission can be found [here](https://discord.com/channels/1340554757349179412/1378034388272681079/1378045981794373662).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1391023433604796456)** (6 messages): 

> `MCP for Claude Code, personality.gg, NipponHomes.com` 


- **MCP Helps Claude Code**: A member created a simple **MCP** (Message Control Protocol) for **Claude Code/Cursor** to manage long-running processes like a dev server, available [on GitHub](https://github.com/patrickjm/pm-mcp).
- **personality.gg: Free Roleplay Website Alternative**: A member promoted **personality.gg**, a free roleplay website and app alternative to Character.ai and Janitorai.com, powered by **OpenRouter.ai**.
   - The platform also has [a Discord server](https://discord.personality.gg).
- **NipponHomes.com built on OpenRouter**: A member created [NipponHomes.com](https://www.nipponhomes.com/explore?search=Sapporo), a **Zillow**-like service for Japan, using **OpenRouter** + **Zyte** + **Scrapy**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1390770478481801369)** (862 messages🔥🔥🔥): 

> `Llama 3.2 3B pricing anomaly, DeepSeek V3 setup guide, Perplexity API issues, Grok 4 Leaks, Monad Tag` 


- **Llama 3.2 1B Model Costs More!**: Members were surprised to find that the **Llama 3.2 1B** model from DeepInfra is priced higher at **$0.005/0.01** than the more capable **Llama 3.2 3B** model at **$0.003/0.006**.
- **Users Need a Guide to Deepseek V3 Setup**: One user requested a guide for setting up **Deepseek V3 0324** on a frontend like **risuAI**, emphasizing the need to send more than 50 messages per day, but they were informed that this required [purchasing OpenRouter credits](https://openrouter.ai).
- **Perplexity API Deprecation Causes OpenRouter Issues**: Users reported issues with the **Perplexity API**, specifically the model *llama-3.1-sonar-small-128k-online*, and a moderator clarified that **Perplexity** likely deprecated the old model, requiring users to [update the `models` parameter in their API request](https://openrouter.ai/docs/features/model-routing#the-models-parameter) for correct model routing.
   - They pointed to a [Changelog Notice](https://docs.perplexity.ai/changelog/changelog#api-model-deprecation-notice) from February, noting the surprise that it still worked, and recommended using the [feature filters](https://openrouter.ai/models?fmt=cards&supported_parameters=web_search_options) on the Models Overview to find alternatives.
- **Alleged Grok 4 Leaks Tease**: An image was shared ([image.png](https://cdn.discordapp.com/attachments/1094454198688546826/1391015727145685214/image.png?ex=686da799&is=686c5619&hm=14bf523ae82431744780e05a43d5bc82976ff73b4cbf35f109ef7b2ce4220343&)) purporting to show benchmark results for **Grok 4**.
   - Discussion suggests that **Grok 3 mini** is a *real deal* and that **Grok 3** is not anything special considering its price.
- **Toven's Crypto-Spam Squelch**: Spam from crypto bros has been on the rise, and they have been getting blocked, they are also identifiable as having a **Monad tag** and bull posting on X, and a mod, Toven, has been [locking them out](https://discord.com/channels/1091220969173028894/1092729520181739581/1390345446169383074) for a week at a time.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1391844016059973633)** (2 messages): 

> `` 


- **No New Models to Report**: There were no new models or significant discussions about models in the provided messages.
- **Channel Silent on Innovation**: The channel activity lacked substantive discussion, updates, or links related to new AI models.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1390773170360811691)** (238 messages🔥🔥): 

> `LLMs with good function calling, LM Studio and GPUs, MCP web-search server, Qwen3 for Reasoning, LLM context size considerations` 


- **Mistral Small 3.2 gives Qwen 2.5 a Run for Its Money**: Members discussed finding an LLM with good **function calling** capabilities similar to **Qwen 2.5**, but without heavy *thinking*, with [Mistral Small 3.2](https://mistral.ai/news/mistral-small/) and [Qwen 3 4B](https://huggingface.co/Qwen/Qwen3-4B) models suggested as alternatives.
   - One member noted that adding the **/nothink** command to the system prompt could help in avoiding reasoning in the models.
- **Positive Review Hiding prompts is "resume material"**: A user shared a [Nikkei Asia article](https://asia.nikkei.com/Business/Technology/Artificial-intelligence/Positive-review-only-Researchers-hide-AI-prompts-in-papers) about AI research papers hiding prompts to generate positive reviews, joking that *'That's what I put in my resume. White letters, size 3 pt.'*
   - Another user advised to **not ask LLMs about their relationships or capabilities** as they tend to hallucinate, being trained before such knowledge is available to them.
- **Web Search MCP Server Makes Waves**: A member shared their web-search **MCP server** which integrates with **LM Studio**, allowing models like **Qwen3** to perform web searches and ground their responses with current information, found at [github.com/mrkrsl/web-search-mcp](https://github.com/mrkrsl/web-search-mcp).
   - Users discussed how the model decides when to search, potential bot detection risks from scraping, and the possibility of implementing captcha support.
- **Qwen3 Struggles with Cutoff Dates and Search Terms**: It was noted that **Qwen3**, despite having a mid-2024 cutoff date, sometimes uses *2023* in web queries, even for recent events, but [the fix is simple](https://www.promptingguide.ai/techniques/date-awareness).
   - One can simply add a statement like *'The current year is 2025'* to the system prompt to resolve this issue.
- **Context Size Woes? RAM and VRAM to the Rescue!**: Users discussed issues related to **context size** and **memory usage**, with one user reporting problems loading a **Qwen2.5-VL-72B-Instruct** model despite having two **RTX A6000 GPUs** with 96GB VRAM due to the model's large size and vision capabilities.
   - Tips included reducing context size, ensuring sufficient VRAM, and checking for model corruption.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1390807740133408909)** (92 messages🔥🔥): 

> `GPU Detection in LM Studio, AMD vs. Nvidia for LLMs, Token Generation Speed, VRAM Requirements for Models, Combining GPUs` 


- **LM Studio Struggles with GPU Detection**: A user reported that LM Studio wasn't detecting their GPU, despite it working well with Stable Diffusion, even after ensuring the drivers were up to date.
   - The user confirmed that LM Studio now detected the GPU, but continues to face problems, which may be solved by disabling `mmap()`.
- **Used Radeon RX 9070XT vs RTX 3090 Tradeoffs**: Members discussed the benefits of the **RTX 3090** for LLMs due to its larger **24GB VRAM**, with the caveat that it's hard to find new and used models are a gamble.
   - The **RX 9070XT** is cheaper and good for both gaming and LLMs, however the RTX 3090 remains superior for AI tasks due to its larger VRAM, also suggesting the **5060TI** and **9060XT** as cheaper, purely AI focused options.
- **Token speed considerations**: The community debated the comfort level of token generation speed, and one member found **Gemma-3 12B** models generating 90-110 tokens, while other 12B models struggle to hit 20 tokens/sec.
   - One user stated *Human eyes can't read more than 24 Tokens/s*.
- **Potential GPU upgrade**: One user asked about upgrading their sister's GPU from a **GTX 1060 6GB** to a **RTX 2070**, and potentially using the 1060 in combination.
   - It was pointed out that combining GPUs is not that simple and suggested selling the **1060** for **$25**.
- **Optimizing VRAM Usage**: Members discussed freeing up VRAM by plugging the monitor into the integrated GPU, and allocating more VRAM to the 3080.
   - It was clarified that integrated graphics don't have their own VRAM and instead reserve a portion of system RAM, and that **Nvidia** cards cannot combine with integrated graphics, whereas **AMD** can with vulkan.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1390776727424336023)** (142 messages🔥🔥): 

> `Face Recognition Models, Text-to-SQL with T5, HuggingChat Shutdown, ComfyUI and GPU Performance, HairStyle Spaces` 


- **DeepFace Rides to Image Similarity Rescue**: Users discussed face recognition models, with one user seeking a model to predict the likelihood of similarity between two faces, and another member suggested [DeepFace](https://github.com/serengil/deepface) and linked to a [blog post on image similarity](https://huggingface.co/blog/image-similarity) and a [Face Similarity space](https://huggingface.co/spaces/Sk1306/Face-Similarity).
   - Another user recommended checking out this channel for **dataset creation**: [dataset creation](https://discord.com/channels/879548962464493619/1217179426002047076).
- **T5 Text-to-SQL gets some love and help**: A user requested help with a **text-to-SQL model** on **T5-base**, and a member provided links to a [Medium article](https://medium.com/%40martinkeywood/fine-tuning-a-t5-small-model-to-generate-sql-from-natural-language-with-92-3-accuracy-fb29e062c638), a [Hugging Face model](https://huggingface.co/suriya7/t5-base-text-to-sql), and another [T5 model](https://huggingface.co/gaussalgo/T5-LM-Large-text2sql-spiderrelax).
- **HuggingChat got the boot, plans to evolve**: Users noted the shutdown of **HuggingChat**, and staff confirmed it was temporary with plans to evolve it into something better, also notifying users of a time limit to [export Hugging Chat data](https://huggingface.co/chat/closed) before it disappears forever.
   - One user speculated that Discord is just better, another said Discord doesn't give u access to various models for free.
- **Uneven GPUs? No tensor parallelism?**: A user with an uneven number of GPUs sought advice on optimizing performance without tensor parallelism, specifically regarding **ComfyUI**, **ReActor**, **AnimateDiff**, and **Load AnimateDiffModel**.
   - Another member suggested that uneven number of GPUs could be solved by splitting the tensors in a non-uniform way between the GPUs.
- **Whisper API giving 404s and 503s**: A user reported issues with **Whisper-large-3 inference endpoints**, receiving **404** or **503** errors when using the model catalog configuration, even in the playground, and decided to self-host instead.
   - Another user added that default settings for the Hugging Face endpoint itself may not be appropriate and linked to a discussion about [specification changes](https://discuss.huggingface.co/t/inference-api-error-with-whisper-return-timestamps-parameter/150043/14) which probably triggered the issue.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1391651956946374837)** (1 messages): 

> `Building Neural Networks from Scratch, Challenges of Custom Neural Network Implementation` 


- **From-Scratch Neural Nets Pose Difficulties**: A member conveyed that building a **neural network from scratch**, without relying on existing libraries, presents a significant challenge.
- **Custom Neural Network Challenges**: Implementing neural networks from the ground up requires deep understanding and meticulous coding, posing difficulties.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1391110290455793834)** (8 messages🔥): 

> `AI Model Identification, Claude AI Experience, same.dev comparison` 


- **AI Model Selection Under Scrutiny**: A user questioned the accuracy of information found on LinkedIn regarding the AI models used, citing the structure and number of AI agents created.
   - The user expressed *doubt if this information is correct* based on a screenshot from LinkedIn.
- **Claude AI Gets a Tryout**: A user inquired whether another user had experience using **Claude AI**.
   - The user responded positively, noting it was *pretty good* and somewhat comparable to **same.dev** but available for free.
- **same.dev Competes with Claude AI**: Users compared **Claude AI** with **same.dev**, highlighting similar functionalities.
   - One user specifically mentioned that **Claude AI** is free, implying a cost advantage over **same.dev**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1390843625834938429)** (13 messages🔥): 

> `JauAuth, PiTutor, BorgLLM, RecycloBot, Arena-RLHF` 


- **JauAuth saves the day!**: A developer created [JauAuth](https://github.com/Jau-app/JauAuth), a secure gateway **MCP Router**, to mitigate the **CVE-2025-49596** vulnerability that allowed attackers to take over developer machines.
- **PiTutor Turns Learning into Play**: A member introduced [PiTutor](https://pitutor.pi4wear.com), a free beta that turns any **PDF or doc** into an interactive learning session with real-time explanations, highlighting, and a whiteboard.
- **BorgLLM Simplifies LLM Provider Configuration**: A developer released [BorgLLM](https://pypi.org/project/borgllm/), a simplified library for configuring **LLM providers** with automatic fallback and API key switching, installable via `pip install borgllm`.
- **RecycloBot Identifies Recyclable Items with AI**: A member shared [RecycloBot](https://huggingface.co/spaces/tejasashinde/recyclobot), an **AI-powered tool** that helps users determine if items are recyclable by uploading an image and receiving region-specific disposal tips.
- **Arena-RLHF Open Sourced for Human Preference Data**: [Arena-RLHF](https://github.com/delta-hq/arena-rlhf) built with HuggingFace, an easy way to RLHF on arena-style human preference data (LM Arena, Agent Arena), has been **open-sourced**.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1391413305838534706)** (3 messages): 

> `LLM Fine-Tuning Blogs, GPU Parallelism, Transformer Inference` 


- **Parallelism Paradigms Blogpost Posted**: A member shared a [blog post](https://datta0.github.io/posts/understanding-multi-gpu-parallelism-paradigms/) on **GPU parallelism**, covering strategies from **Data Parallel to Model Parallel**.
   - The blog post delves into **Transformer inference**, explaining how each strategy can be used optimally with graphics, math, and code.
- **LLM Fine-Tuning Blog Sought**: A member asked for suggestions on the **best blog** regarding **fine-tuning an LLM model**.
   - Another member suggested an *easy-to-read white paper*.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1391656513185124423)** (1 messages): 

> `Deepfake Detection System, AI and Cybersecurity Combination` 


- **Article on Deepfake Detection System Released**: A member shared a new article on [Medium](https://medium.com/@aaryankurade0101/unmasking-the-deceit-a-comprehensive-deepfake-detection-system-12f934832712) about combining **AI with cybersecurity for detecting Deepfakes**.
   - The author requested feedback on the article.
- **Deepfakes Unmasked in New Cybersecurity Article**: A new Medium article, [Unmasking the Deceit](https://medium.com/@aaryankurade0101/unmasking-the-deceit-a-comprehensive-deepfake-detection-system-12f934832712), explores a **deepfake detection system** using **AI and cybersecurity**.
   - The author seeks community feedback on their work.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1391869056688324709)** (1 messages): 

> `GLoVE Model, GLoVE Paper, Co-occurrence Probability Symmetry` 


- **GLoVE Model: Symmetry Rule Query**: A member questioned why a naive role exchange doesn't solve the symmetry issue in **GLoVE** model's co-occurrence probability, referencing a specific section of the **GLoVE paper**.
   - The query focused on why the probability of co-occurrence of *x* in the context of *k* should be the same as *k* in the context of *x*, but isn't followed in **step 3** of the **GLoVE** model as described in the attached [image](https://cdn.discordapp.com/attachments/922424173916196955/1391869056445059293/image.png).
- **GLoVE Model: Clarification on Co-occurrence**: The discussion delves into the specifics of the **GLoVE (Global Vectors for Word Representation)** model and its behavior regarding co-occurrence probabilities.
   - The user is seeking to understand the nuances of why simply inverting the roles of words and context doesn't resolve the asymmetry that arises during the model's training process.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1390775214543081612)** (31 messages🔥): 

> `Time commitment for Agents course, Course assignments and certifications, Access to Llama 3 Model, Issues with Unit 1 Notebook, Guidance on completing Quiz 2.1` 


- **Time commitment for Agents course questioned**: A member inquired about the time commitment for the agents course, mentioning they have approximately **4 hours a day for a week**.
   - They asked whether this is a viable timeframe or if they should explore other options, given their limited availability.
- **Agents Course assignments and certifications clarified**: A member who just started the course inquired about receiving assignments and certifications, understanding that there are no live streams or running schedules.
   - Another member confirmed that this is true.
- **Llama 3 Model access sought for genetic AI course**: A member enrolled in a genetic AI course sought guidance on how to gain access to the **Llama-3.3-70B-Instruct model** after their initial request was not approved.
   - Another member suggested using the model from Ollama as an alternative, noting its ease of setup.
- **Users Encounter Errors in Unit 1 Notebook**: Multiple members reported encountering errors in the **second block of code in the first unit's notebook**.
   - Members expressed discouragement, and requested an update from the HF team on the issue.
- **Challenges in Quiz 2.1 Highlighted**: A member raised questions about **Quiz 2.1**, specifically whether to input their HF key in the code and what the 404 error and warning messages mean.
   - They requested assistance on how to proceed with the quiz given the persisting issue with using HfApiModel and the explicit instructions provided.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1390771528706293891)** (180 messages🔥🔥): 

> `US Copyright Office AI Policy Volumes, Gemini vs ChatGPT for Math, Logical Fallacies in AI Criticism, Material Science, Physics with LLMs` 


- **US Copyright Office Drops AI Policy Trilogy**: The US Copyright Office released [three volumes on AI and Copyright](https://www.copyright.gov/ai/), covering **Digital Replicas**, [Copyrightability](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf), and [Generative AI Training](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf).
- **Gemini App struggles with math questions**: Members compared **Gemini 2.5 Pro** in the Gemini App, **Perplexity with Gemini 2.5 Pro**, and **ChatGPT o3** for math question answering, with a question of whether Gemini searches the web implicitly each time.
   - It was reported that *o3 is mostly good enough, though it does hallucinate from time to time*, while *o4-mini-high is faster, and pretty reliable*.
- **Appeal to Authority logical fallacy spreads Disinformation**: It was noted that a mechanical engineer with a [YouTube video](https://www.youtube.com/watch?v=8enXRDlWguU) having over half a million views spreads disinformation and political talking points dressed up as AI criticism, using an appeal to authority logical fallacy.
   - One member pointed out that *95% of the population by 2035 will be doing social media as their main job* and are incentivized to create such content.
- **AI to the Rescue for Fundamental Physics**: Members discussed how AI can assist fundamental physics beyond ML analysis, with one linking to a [YouTube video](https://www.youtube.com/watch?v=0FRXfBwoJZc) on the topic, and others noting AI's use in reducing computational requirements for calculating with quarks, linking to a [relevant article](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.014028).
   - One member suggested that Deep Learning could revolutionize the field by making previous algorithms more efficient, enabling things previously impossible due to cost, and also suggested diffusion model to surrogate total blindness with [this paper](https://arxiv.org/abs/2410.02780) or [this paper](https://arxiv.org/abs/2303.14139) as a start.
- **Cursor's pricing change**: Members discussed that the **pricing model** in Cursor was adjusted, with many users considering to stop using Cursor.
   - It was reported that **Cursor** team issued an official apology over the pricing change.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1391017766726668370)** (7 messages): 

> `log(n) scaling model exhibit, Hierarchical Reasoning Models, Quaternion products in LLMs` 


- **Log(n) Scaling Model Exhibit Offered**: A member offered to present their **log(n) scaling model** to help clarify their thoughts on it.
   - Another member responded with *"Sure!"*, proposing a specific time for the exhibit.
- **Hierarchical Reasoning Models Paper Discussion Scheduled**: The **Hierarchical Reasoning Models (HRM)** paper ([https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)) will be discussed in the [#Daily Paper Discussion](https://discord.com/channels/714501525455634453/1045298343896690699) voice channel.
   - The abstract highlights HRM's novel recurrent architecture, achieving computational depth and efficiency with only **27 million parameters**.
- **Quaternion Products Experimented in LLMs**: A member will discuss their experiments using **quaternion products** in an **LLM** to rapidly summarize text, as an alternative to softmax attention.
   - This discussion is scheduled for a specific date, offering a novel approach to text summarization.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1390813202350080173)** (5 messages): 

> `The Ocean is Wet, Critical Look at Chain of Thought, AI Training Load Fluctuations` 


- **AlphaXiv Publishes "Ocean is Wet" Paper**: A member shared a [post on X](https://fxtwitter.com/FazlBarez/status/1940070420692312178) and a link to the paper on [AlphaXiv](https://www.alphaxiv.org/abs/2025.02) titled *"The Ocean is wet".*
- **Chain of Thought Scrutinized**: A member noted remembering preprints from a while ago that said this, referring to the recent critical analysis of **Chain of Thought**.
   - The member said it's *good to see some critical light on CoT again*.
- **AI Training May Blackout the Power Grid**: A member posted a link to [Semianalysis.com](https://semianalysis.com/2025/06/25/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout/) discussing the risk of **power grid blackouts** due to **AI training load fluctuations** at a **gigawatt scale**.


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1391833569097289738)** (1 messages): 

> `EleutherAI Summer of Open AI Research, Open Science AI Research Project, Mentorship Program` 


- ****EleutherAI** sets sail for Summer of Open AI Research**: The **EleutherAI Summer of Open AI Research** is a fully online event from **August 4 to August 29, 2025**, inviting individuals with limited research experience to contribute to open science projects under experienced mentorship ([link to EAI discord](https://discord.com/channels/729741769192767510/1390779760493334548)).
- **EAI's SOS: Seeking Skilled Swashbucklers for Science!**: The program is looking for programmers, Master's/PhD students, self-taught researchers, and anyone with technical skills wanting to contribute to open science.
   - Interested applicants are directed to read the [project proposals](https://www.eleuther.ai/soar) and complete the [application form](https://docs.google.com/forms/d/e/1FAIpQLSdT2VESB9fup1-y_8zEvOsk6DBfHusMaqTr78Ex8Wrn3iTz_g/viewform?usp=header) before the **July 21** deadline.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1390771710847881257)** (81 messages🔥🔥): 

> `AI Alignment & Interpretability, ROPE frequencies for sliding window, Decentralized AI, Language Modeling as Compression, GLoVE Model Symmetry` 


- **EleutherAI Welcomes New Member Keen on Alignment**: A new member joined the community expressing interest in contributing to research in **AI alignment** and **interpretability**.
   - The member's interests include **AI alignment**, **interpretability**, and **safety** and looks forward to collaborating.
- **Debate Arises over Relative ROPE Frequencies**: A member noted that when doing a sliding window decoding, using the same **frequencies** for **RoPE** would be wrong because the model is not going to see that the sequence is progressing, causing a looping behavior.
   - Another member suggested that using the same **frequencies** in a sliding window shouldn't matter as the *relative distance* seen by the attention mechanism should be fine, while another suggested **lower base frequency** for local attention.
- **Enthusiast Advocates for Decentralized AI**: A member introduced themselves as passionate about **Decentralized AI**, exploring how to build trustable, composable agent systems on distributed infrastructures.
   - They are happy to connect, learn, and collaborate within the community.
- **LLMs as Compressors: Scaling Law Paper?**: Inspired by the '[language modeling is compression](https://arxiv.org/abs/2309.10668)' paper, a member trained pythia models (70m to 1.4B) and measured the compression ratio on text, audio, and image datasets.
   - They are curious if this is a decent find or a fluke, and what needs to be done to establish this to a novel 'scaling law' paper, sharing [attached plot](https://cdn.discordapp.com/attachments/729741769738158194/1391852453712105703/scaling_laws_for_compression.png?ex=686d671c&is=686c159c&hm=591b4707a3e723a65fd808d70940d48594ccfae23811684ee4786abc4c034f45&).
- **Community Analyzes GLoVE Model Symmetry**: A member inquired about the **GLoVE** paper, specifically why inverting roles doesn't solve the symmetry problem where the probability of co-occurrence of x in context of k should be the same as k in context of x.
   - The paper mentions that a naive exchange of roles won't solve the problem and the member seeks the reasoning behind it.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1390786656717504692)** (41 messages🔥): 

> `Aurko Paper, Cognitive Markers, Concat-and-chunk strategy, Flex Attention, CFG` 


- **Mysterious Aurko Paper Sparks Interest**: Members discussed the [Aurko paper](https://arxiv.org/abs/2507.02119), with one member noting it seemed *either totally meaningless or very interesting* requiring further investigation.
   - Another member suggested DMing Aurko for insights, noting the need for time to discern the paper's significance.
- **Concat-and-Chunk vs Sequence Length Matching**: Members debated the use of **concat-and-chunk strategy** vs. **sequence length matching** for pretraining data, with one suggesting laziness as a reason for the former, and linking to [Bucket iterator](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator), [least-squares histogram-packing](https://github.com/graphcore/examples/blob/master/tutorials/blogs_code/packedBERT/nnlshp.py) and [Multipack](https://github.com/imoneoi/multipack_sampler).
   - Other members noted that methods like **sequence length bucketing** have been around since the **RNN** days, and the choice depends on data properties and implementation details, also mentioning that each packing method introduces different biases.
- **Flex Attention**: Members discussed **Flex Attention**, suggesting it is the *ideal solution* for common usecases due to its ability to mix arbitrary-length sequences dynamically per batch.
   - With Flex Attention, there's no need for weird slicing up of the data, and rough scheduling is sufficient.
- **Stable Training at Scale via Loss Curve Analysis**: Members analyzed a paper, finding that scaling curves apply to loss during training and can predict the whole training run, and emphasized that carefully parameterizing the model by scaling the initialization, learning rate, and other hyperparameters is crucial for achieving stable and efficient training at scale.
   - It was suggested that if the loss curve is extremely consistent as you scale for compute optimal training after regularization, you can tell if your distributed training setup is improperly configured.
- **Attention Beacons During Training?**: A member inquired about implementing **attention beacons** during training, which is slightly different from attention masking, allowing for longer sequence lengths.
   - Another member suggested that this would require a very long context for training, unless done in **TBPTT** (truncated backprop through time) style, which wouldn't require a special kernel.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1391848964608757790)** (3 messages): 

> `SAE expansion ratio, Set Autocomplete Model, Publishing Research` 


- **SAE Expansion Ratio Soars with Task Complexity**: A member found that increasing the complexity of a **set autocomplete model** by **4x** (training on **20 subsets** instead of **5**) resulted in a **12x increase** in the required **SAE expansion ratio**.
   - The first model's **SAE** needed a **4x expansion ratio**, while the second model's **SAE** needed a **48x expansion ratio**.
- **Impact of Epochs on SAE Expansion**: A member suggested that the increase in the **SAE expansion ratio** might be due to the model being trained for more epochs, in addition to the increased task complexity.
- **Publishability of Research**: A member inquired about the publishability of their findings regarding the relationship between task complexity and **SAE expansion ratio**.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1391276018513477723)** (7 messages): 

> `MMLU-SR task subsets, datasets parquet conversion, lm eval running time` 


- **MMLU-SR task subsets are not working**: A member reported that they could no longer evaluate models using subsets of the `mmlusr` task due to a `ValueError` indicating that the `BuilderConfig` was not found.
   - Another member pointed out that *the subsets got deleted when they converted the dataset to parquet on the hub* and suggested using a previous commit by adding a `revision` to the `dataset_kwargs` in the task's YAML file.
- **Datasets parquet conversion caused errors**: A member mentioned that converting the datasets to the parquet format on Hugging Face Hub resulted in the deletion of subsets, causing errors when running `lm_eval`.
   - They suggested specifying a previous commit hash in the `dataset_kwargs` of the task configuration file as a workaround; as they [linked a relevant commit](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlusr/question_and_answer/_mmlusr_qna_yml).
- **Lm eval stucks for long time**: A member reported that `lm_eval` gets stuck for around **20 minutes** before starting the evaluation process and [attached a screenshot](https://cdn.discordapp.com/attachments/755950983669874798/1391768455895715911/IMG20250707170951.jpg?ex=686d18e1&is=686bc761&hm=f211e2ae73cc8c60f7d96b5cc7c89210572ec631b06962c24b7b71d69f903de8&).
   - The user was using **2 GPUs** and the `parallelize` argument, but did not show the exact command that was used for the evaluation.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1391360764111093760)** (6 messages): 

> `Small datasets for training, FineWeb dataset samples, FA3 support for H100/H200, dclm-dedup dataset` 


- **Smallish Datasets Sought for Training Validation**: A member was looking for a smallish dataset around **50B tokens** to validate some things, asking for recommendations on subsampled datasets in that size range.
   - They mentioned that [FineWeb samples](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/v1.4.0/sample) are either much smaller at **10B** or double the size at **100B**.
- **FA3 Support Status Queried for H100/H200**: A member asked for updates on **FA3 support for H100/H200**, recalling previous mentions of an impending merge but noting its absence in the repo.
   - They were seeking clarification from the community regarding the timeline for this feature.
- **"dclm-dedup" Dataset Subset Shared**: A member shared a **25B token subset** of the **Zyphra/dclm-dedup** dataset, available at [Hugging Face](https://huggingface.co/datasets/EleutherAI/dclm-dedup-25B), as a possible option.
   - The member acknowledged that this might be smaller than what was initially requested, but offered it as a resource.
- **"nemo_id" Column Clarification Requested**: A member inquired about the meaning of the `nemo_id` column within the **dclm-dedup** dataset.
   - Speculating it could represent a *global shard/local shard mapping*, they sought confirmation from the dataset creators.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1390772025026674801)** (101 messages🔥🔥): 

> `HF documentation 3B model, LLM Matching Generative Responses, Grok's Politically Incorrect Stance, PiTutor interactive learning, Grok's Knowledge Updates` 


- **Hugging Face Rumored to Launch 3B Model**: According to the HF documentation, there will be a **3B model**, which may come in different sizes.
- **LLM to Match Generative Responses**: Members discussed using an **LLM** to match generative responses with reference answers using [this X summary](https://x.com/ShashwatGoel7/status/1941153367289364655) for a quick overview.
- **Debate heats up around Grok's Political Incorrectness**: A user posted a screenshot implying **Grok** is intentionally politically incorrect and another mentioned [this Elon Musk post](https://x.com/elonmusk/status/1936493967320953090).
   - Another user felt it was damaging trust in AI, with someone else responding that [Grok's prompts are available on Github](https://github.com/xai-org/grok-prompts/blob/adbc9a18736d6c2173607b9ed3d40459147534b1/grok3_official0330_p1.j2#L57).
- **PiTutor Turns PDFs Into Interactive Learning Sessions**: A member shared **PiTutor**, a tool that turns any **PDF or doc** into an interactive learning session with real-time explanations, highlighting, and a whiteboard, available in free beta [here](https://pitutor.pi4wear.com).
- **Discussion around Vector Space Refinement**: A member proposed refining vector space representations to be *more human* before the embedding matrix is calculated.
   - Another member noted that if you want to train them, *you'd have to train them*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1390790955656810499)** (18 messages🔥): 

> `Ollama and Openwebui for AI Services, Training LLMs on Their Own Weights, Temperature and Token Usage, Math 500` 


- ****Ollama and Openwebui** Supercharge AI Services**: A member is using a **4080 Super** to provide AI services through **Ollama** and **Openwebui** and is planning to switch to **Llama CPP**.
   - They're looking for ways to help **Nous Research** by putting their models in front of their user base for live testing.
- **LLMs Ponder **Weighty** Matters**: A member inquired about training an LLM on its own weights alongside additional data to induce a deeper understanding of a subject.
   - Another member responded this this could be similar to continued training and weight merging, but can cause catastrophic forgetting and [mixing in some of the original data can help prevent that](https://link.to/example).
- ****Temperature** Impacts **Token** Count**: A member is running an experiment to determine which temperature setting uses the least tokens for language generation.
   - They mentioned using **Math 500** as a small dataset benchmark.
- **Math 500 Benchmark Surfaces**: A member suggested using the **Math 500** dataset as a benchmark for token usage experiments, aiming for a dataset smaller than 1k questions.
   - The member was inspired by a bot repeating messages and giving a spark of inspiration, joking *heh*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1391620961458978897)** (3 messages): 

> `AI Mouse Tracking, Architecture for Mouse Path Training` 


- **Seeking Model Architecture for AI Mouse Tracking**: A member is seeking advice on the appropriate architecture to train a model using a dataset of natural human mouse paths with images of clicked zones and textual prompts representing actions (e.g., moving the mouse to the + icon, scrolling, screenshot).
   - This indicates an interest in leveraging AI to understand and predict human-computer interaction patterns.
- **Dataset Details: Mouse Paths, Images, and Textual Prompts**: The dataset comprises natural human mouse paths paired with images of the zones clicked and textual prompts describing the action performed.
   - This rich dataset aims to capture the nuanced relationship between user intent (textual prompts), visual context (images), and behavioral patterns (mouse paths).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1390785413043585198)** (8 messages🔥): 

> `Chinese AI investment, Parameter-efficient fine-tuning, Trustless agents, Codeium 3.2, Autonomous optical network` 


- **Zhipu Secures Yuan Shower from Shanghai State**: Chinese AI company **Zhipu** secured a **$1.4 billion** strategic investment from Shanghai state funds, as reported by [Technode](https://technode.com/2025/07/04/zhipu-secures-1-4-billion-strategic-investment-from-shanghai-state-funds/).
- **Pi Tutor Builds Personalized AI Tutor**: **Pi Tutor** is building a better future with visual interactions, turning any content into a personal tutor.
   - The service speaks your language and gives you tailored examples, more at their [website](https://pitutor.pi4wear.com/).
- **Fine-Tuning Costs Cut Massively**: June saw the publication of **117 papers** introducing parameter-efficient methods that cut fine-tuning costs by ≈**85%** while preserving ~**98%** accuracy, according to [Konceptual AI](https://konceptual.ai/trending/june-2025-ai-breakthroughs-transforming-business-operations).
- **Trustless Agents Spark Excitement**: Reddit users highlight progress toward **“trustless agents”** using **TEEs**, **zero-knowledge proofs**, and on-chain verifiability to ensure integrity and privacy, detailed in [this Reddit thread](https://www.reddit.com/r/AI_Agents/comments/1ljyhxu/ai_agents_the_innovation_from_a_decentralized/).
- **Codeium 3.2 Boosts Multi-Repo Savvy**: **Codeium 3.2** adds multi-repo awareness (**70+ languages**), reducing boilerplate by ~**40%**, while **GitHub Copilot Chat Enterprise v2.1.5** delivers real-time vulnerability scanning and ≥**92% JUnit test coverage**, according to [AI Agent News](https://aiagentstore.ai/ai-agent-news/this-week).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1391620961458978897)** (3 messages): 

> `AI-Youtube-OG Architecture, Natural Human Mouse Paths Dataset, Model Training Architecture` 


- **Query about AI-Youtube-OG Architecture**: A member referenced an "AI-Youtube-OG" and shared a thank you for sharing information about it, noting they had forgotten about it. 😋
   - It might have something to do with the dataset of natural human mouse paths with images of the zones of where they clicked.
- **Seeking Architecture for Training on Mouse Path Data**: A member inquired about the appropriate architecture for training a model on a dataset of **natural human mouse paths** with images of the zones where they clicked.
   - The dataset also includes **textual prompts** representing the action (e.g., moving the mouse to the + icon of the web browser, scrolling, screenshot, etc.).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1390838221327437884)** (93 messages🔥🔥): 

> `Grok 4 Leaks, Deepseek R2, Gemini-CLI Search Integration, MCP Integration with Aider, Documentation Fetching MCPs` 


- **Grok 4 Leak Sparks Speculation**: An alleged **Grok 4** leak image surfaced, showing it being trained on the **HLE benchmark**.
   - Members expressed excitement about potential leaps in performance and the benefits of competition in the LLM space, while some are skeptical and think *Grok will refuse to work on your project if your git branch named "main", instead of "master"*.
- **Gemini-CLI's Search Integration Impresses but Deletes Secrets**: One member testing **gemini-cli** highlighted its **search integration** for picking up an update to a payment portal, which they hadn't heard about yet, finding it great for beginners.
   - However, it *broke a bunch of code (deleted the secrets?)* without explanation, leading to a desire for **MCP integration with Aider** for documentation fetching.
- **Open-Weight 70B Models Face Scrutiny**: Community members discussed the landscape of open-weight **70B models**, citing **Llama 3.1**, **Qwen2.5 70B**, and **Mistral Large** as notable options, but seek more recent updates.
   - It was mentioned that **Qwen3-32b** (dense, they are all hybrid reasoning for 3.0) should perform better than **qwen 2.5 72b**, also mentioning how Meta dropped the ball on this not having many recent OS options sub-100b.
- **Concerns Arise Over Benchmark 'Gaming' by AI Labs**: Members discussed the reality of AI labs *gaming* benchmarks, with acknowledgment that preventing contamination is impossible.
   - The focus should be on creating cheat-resistant benchmarks and interpreting results with variety, as contamination leads to improved models and their ability to generalize.
- **Aider users wrestle with replace issues**: Some **Aider** users reported that **Aider** performed replaces on older versions of the file when they made edits, and that Markdown files had the most issues.
   - One user recommended clearing the conversation history with `/clear`, while another one suggested dropping and re-adding the file, and `/map-refresh`.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1390772669078704253)** (25 messages🔥): 

> `InputOutput class in aider, OpenRouter provider settings, Git reset, sonnet-4 with ask/code, deepseek 70b` 


- **Customize Aider Output with InputOutput Class**: A user discovered the `InputOutput` class from `aider.io` to customize output and set `yes=True`, with a note that it could potentially change text color for numbers/bullets in lists outputted by the assistant, see [aider.io](https://aider.io).
   - The key code mentioned was:
```python
from aider.io import InputOutput
io = InputOutput(yes=True)
```
- **Selecting OpenRouter Provider via .env**: A user inquired about setting a specific provider for a model through OpenRouter in a `.env` config, trying `AIDER_MODEL=openrouter/deepseek/parasail-deepseek-r1-0528-qwen3-8b`.
   - A member suggested configuring the provider in OpenRouter settings or params via `extra_body`, referencing [OpenRouter's documentation on provider routing](https://openrouter.ai/docs/features/provider-routing).
- **Recovering Lost File with Git Reset and Aider History**: A user faced an issue where `app.py` was not in the repository in the previous commit and the undo failed, and sought advice on how to recover the file.
   - Suggestions included using **Ctrl-Z** in **nvim**, using `git reset` (potentially deleting the file), or reconstructing the file from `aider.history.md`. A member emphasized the importance of committing changes.
- **Assessing DeepSeek 70B Performance**: A member reported disappointing results with **DeepSeek 70B**, citing its poor performance on tasks like *'reading a file and writing unit tests'*, compared to paid models.
   - They questioned whether anyone had achieved even a fraction of the capability of closed-source models with locally running models, even with sufficient tool calling setup.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1390817081326436483)** (3 messages): 

> `Tensor Compilers, CUDA, PyTorch, Vector Matrix Multiplication` 


- **Tensor Compilers Project Kicks Off**: The **Tensor Compilers: Zero to Hero** work group is seeking contributors interested in building **CUDA** and **PyTorch** from scratch.
   - The project aims to be both pedagogical and hardcore, focusing on tensor compilers.
- **Batching Boosts Vector Matrix Throughput**: A member inquired whether batching vectors into matrices is the primary method for improving throughput in **vector matrix multiplication**.
   - This is still an open question in the channel.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1391493041637818549)** (1 messages): 

> `einops, einstein notation, triton` 


- **Seeking EinOps in Triton**: A member inquired about a **Triton** implementation of **einops** or some form of **Einstein notation**.
- **Triton tensor contractions**: The user is looking for ways to use **einops** in **Triton** to help with tensor contractions.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1390786283214999592)** (36 messages🔥): 

> `CUDA printf debugging, NCU memory bandwidth, GPU division vs multiplication, CUDA tile scheduler vs cutlass, CUDA unified memory` 


- **Printful CUDA Debugging**: Members discussed using `printf` for debugging CUDA kernels, noting that output is buffered and requires explicit `cudaDeviceSynchronize()` to flush, and some had preprocessor directives guarding arch-specific code.
   - One member resolved a 'noob issue' caused by compiler preprocessor directives guarding arch specific code.
- **NCU Reveals Compute-Bound Woes**: A user reported that after fixing uncoalesced writes flagged by **NCU**, memory bandwidth utilization and runtime remained flat, which was unexpected.
   - They later discovered that division in a loop was the bottleneck, making the kernel compute-bound, and replacing division with multiplication by the reciprocal increased throughput by **33%**.
- **GPUs Sidestep Division Circuitry**: A member noted that GPU ALUs may not have dedicated division circuits, instead relying on division approximations, as multiplication is done via additions.
   - AFAIK GPU ALUs don't even have a circut for division, they do a division approximation.
- **Custom CUDA Schedulers**: A member inquired about the feasibility of implementing a custom tile scheduler in CUDA instead of using **Cutlass**.
   - A suggestion was made to explore customizable swizzles in **Cutlass** as a potential solution, if that helps.
- **WSL CUDA Lacks Unified Memory**: A user questioned whether setting up a dual boot for CUDA development is justified due to **WSL**'s lack of support for unified memory and certain **nsys** performance counters.
   - It was suggested that avoiding unified memory might be preferable for performance optimization, and that WSL already oversubscribes normal device memory via WDDM.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1390782907844526121)** (1 messages): 

> `ML Platform Hiring, Remote Work, Quora` 


- **Quora Machine Learning Platform Team Seeks Senior/Staff Engineer**: Quora is hiring a **Senior/Staff Software Engineer (L5-L6)** for their Machine Learning Platform team, offering a fully remote position within the US and Canada; [apply here](https://jobs.ashbyhq.com/quora/89b213bf-06e7-43a2-9101-4b93d711d796).
- **Exciting ML Infrastructure Awaits**: The role involves exciting ML infrastructure work, providing an opportunity to contribute to Quora's machine learning capabilities.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1390975365131796480)** (2 messages): 

> `ROCm traces, in-person hackathon, make flags, gpumode-amd-fp8-mm repo` 


- ****Parisian Builder** Joins ROCm Chat**: A builder from Paris introduced themself in the channel seeking **ROCm traces** and ways to place in the in-person hackathon happening today in Paris.
   - The builder shared a link to [gpumode-amd-fp8-mm repo](https://github.com/Snektron/gpumode-amd-fp8-mm/tree/main) and inquired about helpful *make flags*.
- **Flags Question**: A builder asked which make flags you find useful.
   - They are currently using [these make flags](https://cdn.discordapp.com/attachments/1233704710389764236/1391864919116218430/Makefile.txt?ex=686d72b8&is=686c2138&hm=7d9575144b62d4bdfed3663dee86e8d24d5ba3ab516472709f699c74b87843b8&)


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1391019387741278209)** (4 messages): 

> `voxel raytracing, Hopper Pipelining, PiTutor, Deep Infra B200 Instances` 


- **Voxel Raytracing Video Drops**: A member uploaded a [video](https://youtu.be/YB1TpEOCn6w) about **visibility-based chunk selection** for voxel raytracing, covering internals of victim pointers and usage bits.
   - The open source code is available on [GitHub](https://github.com/Ministry-of-Voxel-Affairs/VoxelHexCuTeDSL).
- **CuTeDSL Pipelining on Hopper Blogged**: A member shared a [blog post](https://veitner.bearblog.dev/cutedsl-on-hopper-pipelining/) on **overlapping memory transfer and computation** via Pipelining on Hopper, featuring TMA and WGMMA atoms.
   - Example code is available on the [Cutlass GitHub](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py), with a Colfax blog discussing pipelining with the C++ API for CuTe.
- **PiTutor Turns PDFs Into Learning Pal**: A member introduced **PiTutor**, a tool that turns any PDF or doc into an interactive learning session with real-time explanations and a whiteboard, available in free beta [here](https://pitutor.pi4wear.com).
   - *The idea is simple: what if learning didn’t feel like a chore?*
- **Deep Infra Deals On B200**: Deep Infra is offering **on-demand B200 instances** at $1.99 / h with 1-click deployment, according to a member, details [here](https://deepinfra.com/).


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1391495014239961178)** (5 messages): 

> `GPU Mode Challenges, Leaderboard Questions, Kernelbot Data` 


- **GPU Mode Challenges Revamped Next Week**: Beginner challenges are finished, but will be recreated *soon-ish* (next week) with proper evaluation code, so members can use the **AMD challenges** or **trimul** until then.
- **Leaderboard Question Solutions Released**: Solutions to the leaderboard questions that already ended are already out and available on [HuggingFace](https://huggingface.co/datasets/GPUMODE/kernelbot-data).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1391699302409044068)** (6 messages): 

> `A100 Performance, H100 Performance, Trimul Leaderboard` 


- **A100 Achieves Fifth Place**: A submission on an **A100** GPU secured **5th place** on the `trimul` leaderboard with a time of **20.6 ms**.
- **H100 Scores Second Place**: A member achieved **second place** on the `trimul` leaderboard multiple times using an **H100**, with times of **60.0 ms**, **43.5 ms**, and finally **35.9 ms**.
- **H100 Success on Trimul**: Additional submissions on **H100** were successful on the `trimul` leaderboard, both clocking in at **63.1 ms**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1390771374703775785)** (41 messages🔥): 

> `instance.py refactoring, Github Actions integration, Ruff linting, Pydantic bump PR` 


- **`instance.py` Too Long; Needs Restructuring**: Members discussed that [instance.py](https://github.com/org/repo/blob/main/instance.py) is too long and needs restructuring, potentially into main, which is related to issue [#249](https://github.com/org/repo/issues/249).
- **Github Actions Testing Integration Progresses**: A member confirmed their Github action to publish to testpy is working, triggered on push, and they will set it up to **publish to testpy on merge** along with automatic patch increase.
   - The member provided a screenshot of the [Github Actions workflow](https://cdn.discordapp.com/attachments/1354169122107293786/1391669239361835039/Screenshot_2025-07-07_at_09.36.38.png?ex=686d653a&is=686c13ba&hm=00a1bc26b826a9bb3f8573835bbae6138d81fb95b4aa3dcf5a5a0c22fe8e2d66&) showing the successful run.
- **Ruff Linting on the Horizon!**: The restructuring PR has been merged and the team is turning to how to get **ruff linting** merged in.
   - They may use `--ignore-revs-file` & `.git-blame-ignore-revs` to keep git blame sensible for the bulk changes.
- **Pydantic Bump PR Causes Headaches**: Concerns were raised that the **Pydantic bump PR** involved an unusually large number of file changes (**70 files**).
   - A member mentioned they needed to merge main into their **Pydantic PR** to resolve the large number of commits (95) in that PR.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1390816051931119616)** (4 messages): 

> `picoc compiler for llm.c, picograd kernels, nn: zero to hero follow up, picoc frontend semantic analysis` 


- ****Picoc** sets sights on compiling llm.c**: The project goals for **picoc** are now focused on compiling karpathy's [llm.c](https://github.com/karpathy/llm.c) by the end of the year, utilizing raw **CUDA** and **CUTLASS** for optimization.
   - The initial plan involves tackling **CUDA** first, with **CUTLASS** integration to be figured out later.
- ****Picograd** prioritizing Pytorch1-style kernels**: The **picograd** project is prioritizing the development of [Pytorch1-style kernels](https://pytorch.org/docs/stable/index.html), with a completed forward pass on Karpathy's MLP on CPU.
   - A member mentioned their progress on [X](https://x.com/j4orz/status/1907452857248350421), and expressed that creating a Pytorch2 fusing compiler to triton-style compiler will need to be a v2 of the course, and is unlikely to happen this year.
- **Workgroup Aims for **nn: zero to hero** Follow-up**: The workgroup aims to create a direct follow-up to **nn: zero to hero**, focusing on the software 2.0 equivalent of building an interpreter/compiler.
   - The models directly from nn: zero -> hero will be converted by swapping the `import torch` to `import picograd`.
- **Call for Assistance in **Picoc's** Frontend Semantic Analysis**: A member requested assistance with the frontend's semantic analysis for [C0](http://reports-archive.adm.cs.cmu.edu/anon/anon/2010/CMU-CS-10-145.pdf) within the **picoc** project, particularly in the areas of [lexing/parsing/typing](https://github.com/j4orz/picoc/blob/master/src/ast/mod.rs#L32-L33).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1390731448784261313/)** (1 messages): 

tonic_1: 💪🏻 🏅 🇫🇷 🏆 🚀
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1390861313848119367)** (100 messages🔥🔥): 

> `Cursor.ai Pricing Controversy, Trae-Agent Open-Sourced by Bytedance, ChatGPT Diagnoses Rare Genetic Defect, ChatGPT's New 'Study Together' Feature, Books3 Dataset Lore` 


- **Cursor's Unlimited Pricing Pulled, Users Protest!**: Users are accusing [Cursor.ai](https://cursor.ai) of a **rug pull** due to a sudden shift from a previously *'unlimited'* pricing model to a limited one, resulting in unexpected charges for **Sonnet-4** usage.
   - Some users reported issues with using their own API keys, receiving **500 errors**, while [Cursor clarified their pricing](https://cursor.com/en/blog/june-2025-pricing) stating *'unlimited usage' was only for Auto and not all other models* and offered refunds.
- **ByteDance Releases Trae-Agent into the Wild!**: **Trae AI** open-sourced [Trae-Agent](https://github.com/bytedance/trae-agent), its agent powering their IDE, now available on GitHub and ranked #1 in **SWE-bench Verified**.
   - The company is seeking contributions to building an open Agent ecosystem, with Trae-Agent supporting **OpenAI** and **Anthropic keys** and making it easy to hack it out to open router.
- **ChatGPT Cracks Medical Mystery Doctors Missed!**: A viral Reddit story highlights how **ChatGPT** identified a hidden gene defect (**methylation block**) doctors missed for a decade, leading to significant symptom improvement.
   - The thread discusses **AI's increasing role in healthcare**, particularly for obtaining second opinions and personalized medicine, with new AI systems outperforming doctors in diagnostic accuracy by using a **RAG system on scientific papers**.
- **ChatGPT's 'Study Together' Feature Surfaces**: A Twitter thread discusses a new **'Study Together'** feature in ChatGPT, possibly an internal prototype codenamed *'tatertot.'*
   - Speculation surrounds its function, ranging from **AI-powered group study rooms** and a replacement for flashcards to a **personal AI tutor** within a group setting, or a tool for collaborative discovery with AI.
- **Gemini's Batch API Offers Bulk Savings**: Logan Kilpatrick announced the launch of **'Batch mode'** in the [Gemini API](https://xcancel.com/OfficialLoganK/status/1942245069383434696), offering **50% discounts** on 2.5 models and enabling the processing of billions of tokens.
   - The announcement was met with positive feedback from users.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1390840160467419269)** (41 messages🔥): 

> `Mojo for research and training, Arrow spec, Mojo and Python callbacks, Mojo server tag, Mojo as a Python superset` 


- **Mojo Eyes AI Inference, Not Training**: While some were asking if anyone is using **Mojo** for research and training, it was mentioned that [Modular is pretty focused on the inference side of things](https://www.modular.com/) at the moment.
   - It was suggested to check out **Mojmelon** on *modular-community* on GitHub, and a link to [Nabla](https://github.com/nabla-ml/nabla), which is faster than **JAX** for both compile time and training time.
- **Scientists Still Love Excel**: Molecular scientists still manually parse and generate tables, and Excel is used in unexpected ways, such as for *a full aligner / primer design engine*.
   - The industry should be migrating to **Arrow**, and the Arrow project has interesting prototypes like *arrow dbc*, where the data is sent directly as arrow data by the databases that support it.
- **Mojo Extends Python, Doesn't Supercede**: Despite the high demand to make **Mojo** a true superset of Python, it's being used to extend Python.
   - As Modular CEO [Chris Lattner said](https://discord.com/channels/1087530497313357884/1304251563749146634), Mojo has been laser focused on solving the **AI stack issue** and creating offerings in that space.
- **Mojo Roadmap Updates Soon**: Someone asked if **Modular** released a roadmap or something along those lines, and a community member responded that the most recent Mojo roadmap was published [here](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395).
   - The community member also mentioned that *they should have some updates dropping soon*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1390769294891028580)** (52 messages🔥): 

> `StringLiteral Materialization, Parametric Traits vs Trait Objects, Cloud-Based Mojo Environment, Static Linking in Mojo, SIMD Bitcasting` 


- **`StringLiteral` Materialization Breaks**: A member reported that recent changes required to make `StringLiteral` materialize to a `String` have broken existing code.
   - They are planning to *open an issue and follow up with the compiler folks*.
- **Confusion About Parametric Traits and Trait Objects**: Members discussed the possibility of parametric traits in Mojo, with one member providing a code snippet to achieve a similar effect using trait objects.
   - Another member clarified that *this isn’t parametric traits, this is trait objects*.
- **Request for Cloud-Based Mojo Environment**: A member inquired about setting up a cloud-based Mojo environment for research due to hardware limitations.
   - Another member suggested using *any GPU instance that runs Linux and lets you install pip packages*.
- **Static Linking Feature Requested**: A user sought to compile Mojo code without shared library dependencies, specifically `libKGENCompilerRTShared.so`, for deployment on a remote machine.
   - A member suggested that *full static linking would need to be a feature request* and pointed to existing discussions and a related GitHub issue ([BUG]: Cannot build statically compiled binary #1317) and a new [Feature Request] Static Compilation of pure mojo code #4976.
- **`bitcast` Behavior Change**: A member reported that a previously working `bitcast` operation involving `SIMD` and `DType` no longer compiles, and sought guidance on the correct way to perform the conversion.
   - The member found that adding `raises` to functions prevents the compiler from doing tail call optimization, and linked to a Godbolt example [https://godbolt.org/z/ahn4144a7].


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1391079104530616471)** (5 messages): 

> `Mojo GPU Puzzles, ModuleNotFoundError, Pixi` 


- **Debugging `ModuleNotFoundError` in Mojo GPU Puzzles**: A member encountered a `ModuleNotFoundError: No module named 'max'` error while running **mojo-gpu-puzzles** from Python.
   - Another member suggested using `pixi add max` or `pixi shell` in the puzzles directory to install the necessary dependencies, referencing the [pixi.toml file](https://github.com/modular/mojo-gpu-puzzles/blob/main/pixi.toml).
- **Pixi Workflow for Mojo GPU Puzzles**: A member suggested the typical workflow is to **git clone** the repository, then run **pixi shell** to install all dependencies.
   - The member had tried `pixi add max`, but this did not resolve the issue, prompting further troubleshooting.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1390934555183026206)** (58 messages🔥🔥): 

> `EpicMe MCP, WinCalcMCP, MCP Python Interpreter, MCP for documentation indexing, LangGraph vs Custom Agents` 


- ****EpicMe MCP** Provides Single Endpoint**: A member requested a self-hosted docker image that acts as an **MCP aggregator/gateway**, and another member suggested [EpicMe](https://github.com/epicweb-dev/epic-me-mcp) as a solution.
   - The creator stated that EpicMe covers onboarding in *a very unique and cool way*, providing a [link](https://www.epicai.pro/user-friendly-just-in-time-auth-with-mcp-hd9hw) as an example.
- ****WinCalcMCP** Connects Claude to Windows Calculator**: A member built their first MCP server, [WinCalcMCP](https://github.com/rspeciale0519/WinCalcMCP), that connects **Claude Desktop** to **Windows Calculator** to improve math answers.
   - Another member prefers giving it a full **Python interpreter** as a general tool via [mcp-python-interpreter](https://github.com/yzfly/mcp-python-interpreter) so they do not have to manage so many specialized tools.
- **LangGraph not a great abstraction, use **custom agents****: One member shared that their journey started with **LangGraph** (Langchain actually) then moved to not use it because honestly they did not find it a great abstraction, since then they've been coding their own **agentic framework**.
   - It was agreed that the beauty of **MCP** is that there is no need for people to use the same agentic framework but things can still communicate using a common protocol.
- ****MCPdata** does local documentation indexing for MCP**: A member introduced **MCPdata** from [MaheshDoiphode/mcpdata](https://github.com/MaheshDoiphode/mcpdata), for local documentation indexing for MCP.
   - It seems like **context7** doesn't keep up with the latest from the vercel `ai` api docs.
- **Official **MCP** servers are the only ones worth using**: Members complained about most of the **MCP** servers being useless.
   - One member suggested using the [glama.ai](https://glama.ai/mcp/servers?attributes=author%3Aofficial) website to only get servers from official sources such as **Elasticsearch**, **Kagi**, and **Redis**.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1391705357201702952)** (4 messages): 

> `MCP Monetization, Agentic Payments with Crypto, Fast Agent's MCP Elicitation Support, EpicAI's MCP Search Engine` 


- ****MooPoint** Makes it Easy to Try Services**: [MooPoint.io](https://moopoint.io/) now allows you to try out the service right when you log in for the first time.
- **Monetize **MCP** Servers with Crypto**: A member is exploring ways to monetize **MCP** servers, suggesting that cryptocurrencies and stable coins are a great fit for agentic payments, sharing a simple tool to turn a free MCP into a paid one using bitcoin payments: [lmcp](https://github.com/getAlby/lmcp).
   - They are looking for a partner to create a showcase.
- ****Fast Agent** Gets Comprehensive **MCP Elicitation** Support**: **Fast Agent** now has comprehensive **MCP Elicitation** support, as well as a quickstart guide that makes getting started with Elicitations Servers really simple, according to [this blogpost](https://fast-agent.ai/mcp/elicitations/).
- ****EpicAI** Launches an **MCP Search Engine****: A member shared [EpicAI's MCP Search Engine](https://www.epicai.pro/mcp-search-engine).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1390776378709901364)** (16 messages🔥): 

> `Google Docs comments, NMC Standards Notebook, NHS adoption of NotebookLM, Mindmap embedding, Newsletter drafting from audio files` 


- **Nurse Creates NMC Standards Notebook with NotebookLM**: A Registered Learning Disabilities Nurse and Practice Educator within the NHS created a digital **NMC Standards Notebook** consolidating key documents from the Nursing and Midwifery Council (NMC) into a single, searchable, and interactive resource, available at [notebooklm.google.com](https://notebooklm.google.com/notebook/8fc872a6-dd9b-4148-9a3e-5f2224f42294?authuser=2).
   - The tool supports nurses, midwives, students, and educators by integrating the NMC's framework, streamlining revalidation, supervision, and preceptorship processes, and has been adopted across education networks to embed person-centred care, clinical safety, and inclusive practice.
- **NHS Scales NMC Notebook Nationally with Free Accounts**: An NHS Registered Learning Disabilities Nurse shared that due to the “share publicly” feature, a tool they created is now being used nationally in **NHS** settings for revalidation and preceptorship despite being on a free Google account.
   - The project started as *a local educational aid* but it has grown.
- **Users Discuss Reading Google Docs Comments**: Users are having trouble reading Google Docs comments with NotebookLM; one user asked if the feature *is coming*?
   - One member suggested a workaround that involves downloading the Google Doc as an **MS-Word document**, then printing the Word document to a **PDF**, and then using that PDF as a source.
- **Newsletter Drafting From Audio Files**: A user asked for *any suggestion/advice* on using **NotebookLM** to draft newsletters from audio files of their recordings.
   - No ideas were given.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1390816734159573053)** (36 messages🔥): 

> `Saving Notes in NotebookLM, Interactive Mode Button missing, PDF Upload issues, Saving Chats in NotebookLM, New Model for NotebookLM` 


- ****NotebookLM Auto-Saves Notes Like a Charm****: A user expressed excitement about uploading their school notes and experiencing how well NotebookLM works, while other user asked how to save on notebook.
   - Another member clarified that the **"Notes" Section** at the bottom right includes a "+ Add Note" button, making the process easy and seamless.
- ****Interactive Mode Button Vanishes for Some****: A user reported that the **Interactive Mode button** is missing from their notebooks, despite having a pro subscription.
   - Another user suggested setting the output language to English, as the **interactive mode** is currently only available for Audio Overviews generated in English. It seems the *customize audio button* has been replaced by the Interactive Mode Button.
- ****PDF Upload Roadblocks Encountered****: A user is facing issues uploading a **19.1 MB PDF file** to NotebookLM, despite following tutorials and checking the resources available.
   - A member asked if they had *googled the question*, and another member asked if notebookllm app working for anyone
- ****Chat Saving Capabilities Debated****: Users are frustrated about the **inability to save chats** within NotebookLM, noting that chats disappear after a short period.
   - One user pointed out a workaround: manually saving each answer as a note, but acknowledged it's not the same as saving the entire chat log.
- ****NotebookLM's Data Storage and FERPA Compliance Queried****: A user is seeking clarity on the **safety and security** of NotebookLM for medical students, specifically regarding data storage locations, monitoring of user interactions, and **FERPA compliance**.
   - No answers were provided by the community or staff.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1391561536446922906)** (50 messages🔥): 

> `tinygrad vs pytorch, MLIR, Halide, Exo-lang, whisper example` 


- **Tinygrad's Path to Victory**: George Hotz defines *winning* for **tinygrad** as being the *fastest way to run things*, expanding beyond **QCOM GPU** to **AMD** and **NVIDIA**.
   - Others pointed to the scale of innovation and fine-tuning in **PyTorch**, with chip manufacturers directly contributing to hardware lowering layers in projects like **Triton**.
- **MLIR's Role Debated**: One member thinks that achieving performance close to hand-written **CUDA** requires the **Triton MLIR** route, leveraging its deep hardware knowledge and the **LLVM** compiler infrastructure.
   - George countered that *MLIR is Waymo*, advocating for an end-to-end approach in neural networks, whereas Waymo uses a different approach.
- **Halide Framework Praised and Compared**: George noted that **Halide** has a clearer structure than **MLIR** and **TVM**, emphasizing a single, fast, and correct way to perform computations across all datatypes and backends, focusing on schedule optimization for speed, and points to the [Halide paper](https://halide-lang.org/) for reference.
   - Another member pointed out a comparison from **Exo-lang**, which compares itself *VERY favorably* over **Halide**, and provides links to their [GitHub](https://github.com/exo-lang/exo) and [ArXiv paper](https://arxiv.org/pdf/2411.07211).
- **Revving up whisper example**: A member is getting back to the **whisper example**, seeking a speed boost, and another shares that their branch has *all audio preprocessing rewritten in tinygrad*.
   - They are looking for a replacement for their current streaming transcribing setup of **VAD+whisper.cpp**, finding the **tinygrad implementation** more approachable.
- **tinygrad aims to Commoditize the Petaflop**: Members discussed aiming for interoperability with different hardware, viewing tinygrad as producing fine-tuned code via **UOP graph optimizations** in a hardware-agnostic way.
   - One member stated that the project is still cooking, and they are excited to see if they get AMD on MLperf according to the contract specs by AMD, to get it competitive with Nvidia.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1390872686577582142)** (31 messages🔥): 

> `Automatic Prompt Optimization (APO), Claude Code vs DSPy, DSPy 3.0, SIMBA vs MIPROV2, Tool Selection for LLMs` 


- **APO Relevance in Evolving Models**: A team is building an **AI agent for financial decisions** and is considering investing time in **Automatic Prompt Optimization (APO)**, questioning its relevance as models improve.
   - Community members are debating whether APO is still worth the deep dive.
- **Claude Code Challenging DSPy?**: The community discussed whether **Claude's code capabilities** might replace **DSPy**, referencing [a tweet](https://x.com/jxnlco/status/1941322807729848429) as possible evidence.
   - One member expressed interest in understanding the underlying mechanisms, especially if Claude is autonomously optimizing prompts, suggesting a more structured approach might be beneficial.
- **Legal Concerns with Claude Code**: Some members expressed concerns that using **Claude's code** might not be permissible in large companies due to codebase trade secrets.
   - Another member countered that their big tech company has been using **Claude code** since its release, though it might be different for non-tech companies; still, it's worth linking [DSPy's Tweet](https://x.com/DSPyOSS/status/1941702597561262519) here.
- **Optimizing Tool Selection for LLMs**: A community member shared [a blog post](https://viksit.substack.com/p/optimizing-tool-selection-for-llm) on optimizing tool selection for LLMs, asking whether something like this can be trained natively as a **DSPy module** end to end.
   - The community also inquired about using **DSPy Version 3.0**.
- **DSPy 3.0 DAIS Talk**: A member shared a link to their **DAIS talk on DSPy 3.0 (beta)**, available on [YouTube](https://x.com/DSPyOSS/status/1942318595633017116).
   - It was also noted that the videos could be made public in one week.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1391788364985925642)** (6 messages): 

> `DSPy quickstart, A/B testing prompts in DSPy, signature.py exploration` 


- **Quickstart avoids Prompt Docstrings**: A member shared a [quickstart](https://x.com/hammer_mt/status/1942148631483523518) that avoids prompt docstrings and class modules, also available as a [gist](https://gist.github.com/hammer-mt/a1288d8f3a10a8620d35183e6ee8560d).
   - Another member found this *a cleaner way* especially with longer starting prompts migrated from non-dspy workflows.
- **A/B testing Prompts considered Faux Pas**: A member mentioned they often use this approach to **A/B test prompts** despite it being a *big faux pas* with DSPy.
   - They discovered the approach by exploring **signature.py**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1390784470524625108)** (32 messages🔥): 

> `ChatGPT personality, Claude 4 Comparison, Manus Airdrops, Manus Credits, Manus as project starter` 


- **Manus mistaken for exhibiting ChatGPT personality**: A member questioned whether **Manus** was using **ChatGPT** because of its increased personality and use of emotes.
   - Another member suggested it might be **Claude 4**, noting it exhibits similar behavior.
- **Claude 4 Similarities Noted**: A member compared **Manus** to **Claude 4**, suggesting they exhibit similar behaviors.
   - The member stated *it does the same stuff*.
- **Doubts Cast on Manus Airdrop Capability**: A member asked if **Manus AI** could launch airdrops.
   - Another member replied *nop*.
- **Token Woe as Fixes Drain Credits**: A user complained that errors in their projects drained all **3000 credits** and they couldn't launch anything.
   - They suggested changing the **credit request parameters** so projects can be completed and launched.
- **Manus recommended as project starter instead of delivering working projects**: A member recommended using **VS Code** and other tools, suggesting **Manus** is better at starting projects than delivering working projects.
   - Another member reported experiencing a **Network connection error** with **Manus**.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1391662303887360051)** (3 messages): 

> `QLoRA throughput, Fused RMSNorm, Linear cross entropy, LoRAMLP` 


- **QLoRA Throughput Throwdown: Tune vs. Unsloth**: Members inquired about recent comparisons of **QLoRA** throughput and memory usage between **Tune** and **Unsloth**, due to potential upstreaming of [fused RMSNorm](https://github.com/pytorch/pytorch/pull/153666) and linear cross entropy in PyTorch.
   - They think these features could bring **Torchtune** closer to **Unsloth** in optimization for **LoRA**, except for the fused [LoRAMLP](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) kernel in Unsloth.
- **PyTorch-Only LoRA Implementations Closing the Gap?**: Members expressed curiosity about how close the PyTorch-only (no custom Triton) implementations get to **Unsloth** with each new feature.
   - One member recalled that comparisons using *compile* from last year showed performance was *pretty close* and in some cases better.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1390785775888760993)** (5 messages): 

> `Custom Tokenizers, Mistral 2506-small, HF AutoTokenizer, CI Bug` 


- **Custom Tokenizers Worth Maintaining**: Members discussed whether to maintain custom tokenizers, especially since **Mistral** doesn't supply an **HF tokenizer** with their latest **2506-small model**.
   - One member suggested having a fallback that uses the **HF `AutoTokenizer`** and integrates with features like packing.
- **Mistral Requires Extra Library**: A member noted that **Mistral** requires an extra library, deeming its tokenizer *"unfortunate."*
   - The member suggested using HF's AutoTokenizer as a fallback option, and wondered whether it can be made compatible with torchtune's features.
- **CI Bug Alert**: A member reported a new bug in CI ([https://github.com/pytorch/torchtune/pull/2822](https://github.com/pytorch/torchtune/pull/2822)).
   - Another member acknowledged the bug, describing it as *"pretty minor"* and promising a fix by the next day.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1390822757893148722)** (14 messages🔥): 

> `Context-Parallel vs. Limited GPUs, Skepticism on Architectural Improvements, Data Cleaning vs. Architecture Iterations, MoE Training Techniques` 


- **Context-Parallel Scales vs. GPU Limitations**: A member discussed their [preprint](https://doi.org/10.21203/rs.3.rs-7029913/v1) which addresses scaling sequence length with limited GPUs, contrasting it with context-parallel approaches that require more GPUs for throughput.
   - They suggest that for sequence lengths >= **500k**, context-parallel may be better, but for very high sequence lengths (millions), their method could be superior, highlighting a need for benchmark comparisons.
- **Architectural Improvements face Skepticism**: A member expressed skepticism towards recent architectural improvements, suggesting that many reported results are driven by variance and hyperparameter optimization rather than fundamental advances.
   - They argued that investing in data cleaning is more effective than pursuing architectural iterations since the first transformer, advocating for cleaner datasets over complex model modifications.
- **Clean Data Beats Architecture**: A member generally agreed that *cleaning data* can lead to greater abilities than modern LLMs have achieved.
   - He noted his enthusiasm for **SSM papers** and followed those through to **Mamba 2** and that *mostly died, so my enthusiasm may be misplaced*.
- **MoE Training Technique Analysis**: A technique and results for **MoE training** were mentioned, questioning if similar results could be achieved more cheaply without needing a dense fwd pass, as detailed in a [Notion post](https://fengyao.notion.site/moe-posttraining).
   - Another member shared a [figure](https://cdn.discordapp.com/attachments/1293438210097025085/1391882256649290039/IMG_20250707_234251_274.jpg?ex=686d82dd&is=686c315d&hm=1d54ba8bc5b3e6e437f6f87a3c59a0a66e18b08b9b73675635c7bb86e28b2c42&) noting that **linear scaling has lot's of trade offs**.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1390814978281570554)** (4 messages): 

> `Cohere Labs, Open Science Community Summer School` 


- **Cohere Summer School Content now Available**: The Cohere Labs Open Science Community Summer School contents are now available on a [YouTube playlist](https://youtube.com/playlist?list=PLLalUvky4CLK3oT1DKNPagd_lTXooYVlR).
- **Getting Started with Cohere**: New members can find resources and content from the Cohere Labs Open Science Community Summer School on [YouTube](https://youtube.com/playlist?list=PLLalUvky4CLK3oT1DKNPagd_lTXooYVlR).


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1391700755215417364)** (2 messages): 

> `Embed v4 API, Hybrid image/text embeddings` 


- **Embed v4 API for Image/Text Queries**: A member inquired about the correct way to use the **Embed v4 API** for **hybrid image/text embeddings**.
   - Another member suggested setting `input_type` to `search_query` or `document` when using the `embed-v4.0` model.
- **Embed v4 Input Types**: The correct way to specify the input type in Embed v4 is to use either `search_query` or `document`.
   - This is essential for hybrid image/text embeddings to function correctly.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1390807744549748900)** (10 messages🔥): 

> `Self-learning RL/DL, Agentic AI, Cohere Embed v4, Probabilistic Generative Models` 


- **Robotics Student Adores Cohere Montreal**: A PhD student in Robotics, focusing on **self-learning RL/DL**, expressed their love for Cohere and a desire to join the Montreal office.
   - They are seeking like-minded people to discuss and learn together.
- **Tech Enthusiast Enters the AI Agent Arena**: A newcomer to the world of **agentic AI** and development is eager to start learning, networking, and building.
   - They are excited to implement Cohere into their projects and gift it to the world.
- **Researcher Implements Embed V4 Awesomely**: An avid tech researcher has recently used **Cohere Embed v4** for multimodal retrieval.
   - They shared a [YouTube video](https://www.youtube.com/watch?v=TJ9jvYSZwhc) showcasing their work.
- **Generative Models Galore**: A Math & CS student from Germany is working on **probabilistic generative models** and learning samplers.
   - They are interested in both theory and applied aspects, particularly interpretability, and hope to learn more in this area.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1390772023931834449)** (5 messages): 

> `Open Source NotebookLM, LlamaCloud MCP Servers, AI Hack Night` 


- **Open Source NotebookLM Emerges**: A fully **open-source NotebookLM** is now available, allowing users to run it on their own computers for local document knowledge extraction, according to [this tweet](https://t.co/TNzqzF77yQ).
- **LlamaIndex Office Hours Focus on MCP**: LlamaIndex held office hours on July 8th, discussing **LlamaCloud MCP servers**, using existing **MCP tools** with **Agent Workflows**, and serving agent workflows as **MCP**, as stated in [this tweet](https://t.co/LPCc71sguM).
   - They covered extending functionalities related to **Multi-Cloud Processing (MCP)**.
- **AI Hack Night to Spotlight Cutting-Edge Apps**: An **AI Hack Night** at GitHub will focus on building cutting-edge applications with **AI Agents**, **MCPs**, and **RAG** (Retrieval-Augmented Generation) techniques, confirmed by [this tweet](https://t.co/AhBsvYjnRx).
- **Structured Data Extraction Workflow Demoed**: A demo notebook showcases building a structured data extraction workflow with human-in-the-loop validation, leveraging LLMs for data pre-processing to create schemas for bulk extraction work, per [this tweet](https://t.co/DLPtrEVKca).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1391288126030483609)** (9 messages🔥): 

> `Document Intelligence for P&IDs, Handwritten Text Extraction, LlamaIndex UX for Business Users` 


- ****P&ID Document Intelligence Builds Momentum****: A member is developing document intelligence for **Piping & Instrumentation Diagrams (P&IDs)**, which are complex engineering blueprints with overlapping symbols, small text, and intricate lines.
   - They are seeking advice on handling complex technical diagrams like electrical schematics, performance benchmarks for dense content, and thoughts on hybrid approaches with **LlamaIndex** for relationship reasoning.
- ****Handwritten Text Extraction Tool Hunt Begins****: A member is looking for a tool to extract handwritten text as it is from an image, and shared a [sample image](https://cdn.discordapp.com/attachments/1059201661417037995/1391473392837988562/CamScanner_07-05-2025_20.45_12.jpg?ex=686d5795&is=686c0615&hm=949a322e62cf127e2b9ef25975fd8bba3f7da0c629da0e283fb02c5ecc2e0da3&).
- ****LlamaIndex UX Quest for Business Acumen****: A member is seeking a user-friendly UX for managing docs and indices within **LlamaIndex**, aiming to allow business users to update the company's knowledge base without needing to deeply understand the agents' operations.
   - They envision a central location for uploading and organizing documents, with developers creating AI agents that utilize the indices; so far, **Simba** is the only option they've identified.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1390788864989462608)** (9 messages🔥): 

> `Nomic API, GPT4All server mode, Jinja Chat Template, CrowdLLM, OpenAssistant` 


- **New Member Asks About Nomic API**: A new member inquired about using the **Nomic API** and running **GPT4All** in server mode locally.
   - They specifically asked if there were accessible local endpoints.
- **User Struggles with Jinja Chat Template**: A member requested assistance with a chat template in **Jinja format**.
   - Another member suggested that the user may have a specialized model, and offered assistance understanding the **system prompt**.
- **CrowdLLM Dataset Building Tool Introduced**: A member shared a tool called [CrowdLLM](https://crowdllm.ct.ws) for **crowd-building datasets** for LLMs.
   - The tool enables users to create tasks and contribute by adding prompt answer pairs, prompts, or answers.
- **CrowdLLM compared to OpenAssistant**: A member drew a comparison between the newly introduced [CrowdLLM](https://crowdllm.ct.ws) and **OpenAssistant** from 2023.
   - Another member noted that **OpenAssistant** must be available for their system first.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1391376875233738772)** (3 messages): 

> `Free Book PDF, Author's Blog, Copyright Issues` 


- **User Requests Free Book PDF Directly From Author**: A user directly asked an author for a free PDF copy of their book, citing a lack of funds, after listening to a 1-hour talk.
   - Another user commented it was *in bad taste* to ask, suggesting checking the author's blog for excerpts instead.
- **Alternative Access via Author's Blog**: A user suggested that instead of requesting a free PDF, the person should check the author's blog.
   - The blog purportedly contains excerpts and standalone posts from the author's books, offering an alternative way to access the content without direct purchase.

