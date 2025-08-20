---
id: MjAyNS0w
title: Databricks' $100B Series K
date: '2025-08-19T05:44:39.731046Z'
description: >-
  **Databricks** reached a **$100 billion valuation**, becoming a centicorn with
  new Data ([Lakebase](https://www.databricks.com/product/lakebase)) and AI
  ([Agent
  Bricks](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/))
  products. **OpenAI** launched **ChatGPT Go** in India at ₹399/month (~$4.55),
  offering significantly increased usage limits and UPI payment support, with
  plans for global expansion. The **DeepSeek V3.1 Base/Instruct** models were
  quietly released on Hugging Face, showing strong coding benchmark performance
  and adopting an Anthropic-style hybrid system. The **Qwen-Image-Edit** model
  from **Alibaba** is gaining traction with integrations and community pruning
  experiments. *"DeepSeek V3.1 Base outperforms Claude 4 Opus on coding
  benchmarks"* and *"ChatGPT Go offers 10x higher message limits and 2x longer
  memory"* highlight key advancements.
companies:
  - databricks
  - openai
  - deepseek
  - hugging-face
  - alibaba
models:
  - deepseek-v3.1-base
  - deepseek-v3.1-instruct
  - chatgpt-go
  - qwen-image-edit
topics:
  - model-release
  - benchmarking
  - pricing-models
  - fine-tuning
  - model-architecture
  - image-editing
  - video-generation
  - api
  - agentic-ai
people:
  - sama
  - nickaturley
  - kevinweil
  - gdb
  - sherwinwu
  - nptacek
  - reach_vb
  - clementdelangue
  - teortaxestex
  - quixiai
  - georgejrjrjr
  - scaling01
  - alibaba_qwen
  - linoy_tsaban
  - ostrisai
  - lmarena_ai
---


**Data and AI are doing well!**

> AI News for 8/18/2025-8/19/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (229 channels, and 6920 messages) for you. Estimated reading time saved (at 200wpm): 549 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

[**DeepSeek V3.1 Base/Instruct dropped today**](https://x.com/swyx/status/1957902542136045608), but given that DeepSeek usually releases their evals/papers shortly after the models, we're holding off til then to give them the title story.

Instead today's story is a pretty simple one: **Databricks is now a centicorn**.

![](https://resend-attachments.s3.amazonaws.com/A6NbHltCY1mzZXR)

The [press release](https://www.databricks.com/company/newsroom/press-releases/databricks-raising-series-k-investment-100-billion-valuation) contains very little more detail beyond plugging their new Data ([Lakebase nee Neon](https://www.databricks.com/product/lakebase)) and AI ([Agent Bricks](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/)) offerings.

This round was very much leaked among insiders, but even in this AI Summer it is still rare to have a new $100B company, so today rightfully belongs to Team Spark.

---

# AI Twitter Recap

**OpenAI’s ChatGPT Go launch in India and product notes**

- ChatGPT Go debuts in India at ₹399/month (~$4.55), offering 10x higher message limits, 10x more image generations, 10x more file uploads, and 2x longer memory vs free; pricing appears in INR and UPI payments are supported. OpenAI plans to learn from the rollout before expanding globally. See details from [@nickaturley](https://twitter.com/nickaturley/status/1957613818902892985), [@snsf](https://twitter.com/snsf/status/1957640122171896099), [@kevinweil](https://twitter.com/kevinweil/status/1957646363212087650) and [@sama](https://twitter.com/sama/status/1957849495733166587); coverage by [@gdb](https://twitter.com/gdb/status/1957650320923979996).
- The Responses API was built for complex, tool-heavy interactions and is being actively used for agentic workloads (e.g., by AugmentCode). If you’re evaluating GPT‑5’s execution patterns, test with base model API calls and the Responses API to separate model from host UI effects. See [@sherwinwu](https://twitter.com/sherwinwu/status/1957659638834593831), [@gdb](https://twitter.com/gdb/status/1957851156564042012), and an ops caution from [@nptacek](https://twitter.com/nptacek/status/1957622370920779880).

**DeepSeek V3.1 release: a quiet update with outsized impact on coding**

- DeepSeek quietly shipped V3.1 Base and Instruct on Hugging Face with no card at launch, but immediate community uptake. Architecture/config appears unchanged vs V3; the update likely reflects continued post‑training and a move toward an Anthropic‑style hybrid “no‑think/think” system that unifies modes. See the drop and early diffs via [@reach_vb](https://twitter.com/reach_vb/status/1957821171249934486), [@ClementDelangue](https://twitter.com/ClementDelangue/status/1957823652298166340), [@reach_vb](https://twitter.com/reach_vb/status/1957824849633485249), and analysis from [@teortaxesTex](https://twitter.com/teortaxesTex/status/1957818879205351851) and [@QuixiAI](https://twitter.com/QuixiAI/status/1957874743165743191).
- Early eval signals: V3.1 Base (no thinking) outperforms V3.1 Thinking and R1‑0528 on SVGBench, and V3.1 reportedly beats Claude 4 Opus on the Aider Polyglot coding benchmark—all while maintaining aggressive pricing. See [@teortaxesTex](https://twitter.com/teortaxesTex/status/1957857573878550924), [@scaling01](https://twitter.com/scaling01/status/1957890953026392212), and [@scaling01](https://twitter.com/scaling01/status/1957892601098432619). Notably, the Base model’s release under an MIT license marks a rare permissively licensed large base model ([@georgejrjrjr](https://twitter.com/georgejrjrjr/status/1957867653764379073)), and the repo surged to top trending despite the minimal announcement ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1957897020741402751)).

**Image editing and video generation: Qwen-Image-Edit lands; video tools mature**

- Qwen‑Image‑Edit adoption is accelerating: it landed in Anycoder and LMArena’s image‑edit track; a diffusers integration bug was found and fixed; community LoRA workflows and pruning experiments (20B→10B by dropping half the blocks) are active. See [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1957709912202682588), the fix ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1957840853277290703)), bangs edit demo ([@linoy_tsaban](https://twitter.com/linoy_tsaban/status/1957762030393544847)), pruning notes ([@ostrisai](https://twitter.com/ostrisai/status/1957748358451503166)), and Arena onboarding ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1957878222986821711)). Also see StepFun’s NextStep‑1‑Large‑Edit (14B AR, Apache‑2) for alternatives ([@Xianbao_QIAN](https://twitter.com/Xianbao_QIAN/status/1957749693485838448)) and faster diffusers pipelines for large models like Wan/Qwen ([@RisingSayak](https://twitter.com/RisingSayak/status/1957668389935096115)).
- Video gen at scale: Google reports 100M videos created with Veo 3 in Flow ([@demishassabis](https://twitter.com/demishassabis/status/1957641792263737786)), Runway shipped workflow/control updates ([@runwayml](https://twitter.com/runwayml/status/1957881165781602724)), and there’s an OSS Next.js template for Veo 3/Imagen 4 usage ([@_philschmid](https://twitter.com/_philschmid/status/1957821851331416079)). Techniques continue to advance—e.g., Next Visual Granularity Generation for coarse‑to‑fine control ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1957836902020612180))—while production examples proliferate (e.g., LTX Studio’s fully AI‑created short with consistent characters, [@LTXStudio](https://twitter.com/LTXStudio/status/1957799093582844254)).

**Agent frameworks, standards, and voice stacks**

- Cline shipped “Auto Compact” to summarize+roll context past token limits, allowing multi‑million‑token tasks in a 200k window. The team argues context management can be largely automated and documented their approach and tools. See [@cline](https://twitter.com/cline/status/1957670663508124073) with docs ([@cline](https://twitter.com/cline/status/1957670675415724284)) and broader guidance ([@nickbaumann_](https://twitter.com/nickbaumann_/status/1957669736491470999)).
- Standards and integration: [AGENTS.md](http://agents.md/) is emerging as a vendor‑neutral spec to guide agent behavior in repos (adopted by Cursor, Amp, Jules, Factory, RooCode, Codex) ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1957925682048336354)), with a new multi‑org working group ([@FactoryAI](https://twitter.com/FactoryAI/status/1957926852020039767)). LlamaIndex published comprehensive Model Context Protocol docs and tooling (clients/servers, LlamaCloud MCP services) for connecting agents to tools/databases/services ([@llama_index](https://twitter.com/llama_index/status/1957840992360710557); guide via [@jerryjliu0](https://twitter.com/jerryjliu0/status/1957873536456093903)).
- Code‑first voice agents: Cartesia launched Line, a developer‑centric voice agent platform with background reasoning, logging/summaries, and fast cold‑starts (Modal integration). The team emphasizes code‑driven iteration with evals and deep model integration; early community demos span prank bots to research assistants. See launch and philosophy via [@cartesia_ai](https://twitter.com/cartesia_ai/status/1957862421667664216), the technical vision ([@krandiash](https://twitter.com/krandiash/status/1957863360730657200)), and examples ([@modal](https://twitter.com/modal/status/1957865381613224050), [@rohan_tib](https://twitter.com/rohan_tib/status/1957864976582078949), [@bclyang](https://twitter.com/bclyang/status/1957868316711846236)).
- Developer productivity: GitHub Copilot’s new Agents panel lets you prompt a repo‑aware coding agent from any page and receive PRs without breaking flow ([@github](https://twitter.com/github/status/1957894152412082643)). Jupyter Agent 2 executes data workflows inside notebooks with Qwen3‑Coder on Cerebras and E2B runtimes ([@lvwerra](https://twitter.com/lvwerra/status/1957832240416580024)). Firecrawl v2 offers unified web/news/image search with deep scraping for agent context engineering ([@omarsar0](https://twitter.com/omarsar0/status/1957837839405920282)). Sim provides an OSS canvas for agent workflows ([@_avichawla](https://twitter.com/_avichawla/status/1957691571908038717)).

**Evaluations: thinking trade‑offs, multilingual code, biomedical, and spatial reasoning**

- OptimalThinkingBench (OTB) blends OverThinkingBench (simple queries, 72 domains) and UnderThinkingBench (11 hard reasoning tasks) to measure the “right” amount of thinking. Results across 33 SOTA models suggest most methods that improve one side regress the other; room remains to optimize for both. Paper + summary by [@jaseweston](https://twitter.com/jaseweston/status/1957627532963926389).
- Tencent Hunyuan’s AutoCodeBench provides a fully automated LLM+sandbox pipeline to synthesize multilingual coding datasets and benchmarks (3,920 problems across 20 languages), plus a high‑performance multi‑language sandbox. Project, paper, code, and datasets linked by [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1957751900608110982).
- BiomedArena (with NIH CARD) targets real biomedical workflows—literature review through disease modeling—showing no current model reliably meets domain reasoning needs; emphasizes open, reproducible evals with expert‑in‑the‑loop feedback. Details via [@lmarena_ai](https://twitter.com/lmarena_ai/status/1957775319030734957).
- “Has GPT‑5 Achieved Spatial Intelligence?” finds GPT‑5 sets SOTA, but hardest categories (e.g., mental rotation, paper folding) still trail human levels; the gap between closed and open models narrows on the hardest SI tasks. See paper thread by [@_akhaliq](https://twitter.com/_akhaliq/status/1957833219992080581) and analysis from [@omarsar0](https://twitter.com/omarsar0/status/1957885032716177415).
- ARC‑AGI‑3 Preview learnings: after ~3,900 plays, organizers share meta‑findings to guide the next 100+ interactive reasoning tasks ([@arcprize](https://twitter.com/arcprize/status/1957878722004152829)).

**Systems and infra: serving at scale, local runtimes, and MoE speedups**

- Serving/tooling:
    - vLLM added support for Zhipu’s GLM‑4.5/4.5V and highlighted Kimi K2 serving examples; SkyPilot published a multi‑node serving template for 1T+ parameter models combining tensor+pipeline parallelism ([@vllm_project](https://twitter.com/vllm_project/status/1957731795887353895), [@vllm_project](https://twitter.com/vllm_project/status/1957830968234144016), [@skypilot_org](https://twitter.com/skypilot_org/status/1957831495462379743)).
    - Hugging Face’s open inference router crossed 20M monthly requests; fast‑growing providers include Cerebras, Novita, and Fireworks. Cerebras notes 5M monthly requests served on their infra ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1957856311598805006), [@CerebrasSystems](https://twitter.com/CerebrasSystems/status/1957957962514960567)).
    - llama.cpp remains the lightest local stack: “ultimate guide” for GPT‑OSS on all devices, plus Firefox adding LLM add‑on support via llama.cpp and wllama; quick start for macOS via llama‑server shared by [@simonw](https://twitter.com/simonw/status/1957880963666702466) and [@ggerganov](https://twitter.com/ggerganov/status/1957821440633282642), Firefox news via [@ggerganov](https://twitter.com/ggerganov/status/1957844552150110227).
- Training/optimization:
    - Cursor rebuilt MoE at the kernel level and moved to MXFP8, claiming 3.5x faster MoE layers and 1.5x end‑to‑end training throughput vs prior OSS alternatives ([@stuart_sul](https://twitter.com/stuart_sul/status/1957927497351467372); kernels highlight via [@amanrsanger](https://twitter.com/amanrsanger/status/1957932614746304898)).
    - Baseten + Axolotl released an out‑of‑the‑box recipe for fine‑tuning gpt‑oss‑120B (multi‑node, one‑line deploys, observability) ([@basetenco](https://twitter.com/basetenco/status/1957877915737362437)). OpenAI’s GPT‑OSS implementations saw quality fixes after an initial regression ([@ozenhati](https://twitter.com/ozenhati/status/1957896891468800345)).
    - New optimizer: Kourkoutas‑β (Adam‑style with dynamic β₂ memory) for bursty gradients ([@KassinosS](https://twitter.com/KassinosS/status/1957755625854890323)). Practical training notes included MFU gains from numerically stable renormalization ([@khoomeik](https://twitter.com/khoomeik/status/1957754482185630071)).

**Top tweets (by engagement)**

- OpenAI launches ChatGPT Go in India at ₹399 with 10x quotas and UPI support ([@nickaturley](https://twitter.com/nickaturley/status/1957613818902892985); [@sama](https://twitter.com/sama/status/1957849495733166587)).
- Databricks signs a Series K at >$100B, scaling Lakebase (serverless Postgres) and Agent Bricks (agentic framework with reasoning guardrails) ([@alighodsi](https://twitter.com/alighodsi/status/1957795160416309717)).
- DeepSeek V3.1 lands on Hugging Face; early coding evals are strong, with MIT‑licensed Base model ([@reach_vb](https://twitter.com/reach_vb/status/1957821171249934486); [@scaling01](https://twitter.com/scaling01/status/1957890953026392212)).
- [AGENTS.md](http://agents.md/) gains momentum as a repo‑level agent spec, supported by multiple IDEs/agents ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1957925682048336354)).
- llama.cpp guide for GPT‑OSS on any device; Firefox adds llama.cpp‑powered LLM add‑ons ([@ggerganov](https://twitter.com/ggerganov/status/1957821440633282642); [@ggerganov](https://twitter.com/ggerganov/status/1957844552150110227)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DeepSeek V3.1 Model Announcements and Features

- [**deepseek-ai/DeepSeek-V3.1-Base · Hugging Face**](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base) ([Score: 647, Comments: 177](https://www.reddit.com/r/LocalLLaMA/comments/1mukl2a/deepseekaideepseekv31base_hugging_face/)): **deepseek-ai has released DeepSeek-V3.1-Base, a language model available on Hugging Face, boasting an unprecedented size at >685B parameters. This positions it as one of the largest open-source models to date, far surpassing typical LLM scales and rivaling commercial models in capacity. Technical discussion is limited, but the model's release notably suggests an escalation in parameter count arms race among LLM labs.** Most technical comments express awe at the sheer scale ("685B" parameters) and note its timing relative to major model releases like GPT-5, implying competitive open-sourcing strategies.
    - A user requests additional benchmark results or hands-on usage data for DeepSeek-V3.1-Base, reflecting a key technical interest in its comparative performance, real-world capabilities, and how it stacks up against models like GPT-5 or others in the 685B+ parameter range.
- [**DeepSeek v3.1**](https://i.redd.it/143veukbpyjf1.jpeg) ([Score: 436, Comments: 94](https://www.reddit.com/r/LocalLLaMA/comments/1muft1w/deepseek_v31/)): **DeepSeek has upgraded its online model to version 3.1, notably extending its context/document length support to 128k tokens, and maintains unchanged API endpoints for integration. The update is now live for testing across the official website, app, and mini program. The announcement visually confirms these changes and encourages user experimentation.** Discussion in the comments suggests speculation about the model architecture, with some users perceiving this as a move toward a hybrid or mixed reasoning model, based on interface clues and performance observations (e.g., verbosity and button changes). There is anticipation for official details to clarify model specifics.
    - Multiple users discuss evidence that DeepSeek v3.1 may be a hybrid model, noting behavioral cues such as different 'vibes' between the chat and reasoner components, and the absence of the "r1" in the think button—suggesting the use of mixed reasoning roles or blended architectures.
    - There are technical observations on model verbosity and output behavior, with users mentioning that DeepSeek v3.1 generates highly verbose responses and demonstrates nuanced instruction-following and creativity abilities that surpass prior versions, especially with prompts designed to test specificity and subversion (such as wish-granting scenarios).
- [**🤗 DeepSeek-V3.1-Base**](https://www.reddit.com/r/LocalLLaMA/comments/1mukwq6/deepseekv31base/) ([Score: 233, Comments: 37](https://www.reddit.com/r/LocalLLaMA/comments/1mukwq6/deepseekv31base/)): **DeepSeek has released the weights for its new deep language model, DeepSeek-V3.1-Base, available directly on Hugging Face (https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base). As of posting, official benchmark results and a detailed model description are not included with the release.** Commenters note a preference for immediate access to model weights prior to benchmarks/documentation, request clarification on the availability of an 'Instruct' finetuned variant, and inquire about support for the GGUF format (used for optimized inference, e.g., with llama.cpp).
    - Commenters highlight that DeepSeek-V3.1-Base is a completely new base model, and its numbering does not necessarily reflect its significance or performance; some speculate it could rival current leading models like GPT-5 and may soon serve as the foundation for new models such as DeepSeek-R2.
    - There is also discussion around notable advances in creative writing ability, suggesting the model's writing quality is competitive with top-tier systems like Gemini, with speculation that its release prefaces further refinement for future major models.

### 2. Local LLM and Face Recognition Integration Experiments

- [**Tried mixing local LLM + face recognition just for fun (wild results)**](https://www.reddit.com/r/LocalLLaMA/comments/1mumext/tried_mixing_local_llm_face_recognition_just_for/) ([Score: 254, Comments: 19](https://www.reddit.com/r/LocalLLaMA/comments/1mumext/tried_mixing_local_llm_face_recognition_just_for/)): **The OP describes experimenting with integrating local language models (primarily LLaMA variants) and facial recognition, envisioning a pipeline in which local LLMs perform descriptive reasoning over face images, with image matching handled by an external facial search tool (Faceseek). Key technical takeaway: current high-fidelity face search (e.g., Faceseek) is still external/cloud-based, but the post speculates on feasible, privacy-preserving *fully local* integration of face identification and multimodal reasoning. The use case suggests potential for edge-computing deployments as vision models (like FaceNet or ArcFace) and compact LLMs mature, although technical hurdles around running accurate, efficient recognition and retrieval models entirely offline remain significant.** There is minimal technical debate in the comments, though one user requests a project share, indicating demand for code or a reproducible pipeline.
    - A commenter highlights that traditional facial recognition algorithms--not based on LLMs--are already highly advanced and questions whether there have been recent improvements in facial recognition driven by large language models (LLMs). They point out a distinction between face recognition and face description: while face recognition works through matching, models like CLIP can generate descriptions from visual data, and multimodal LLMs might be influencing this space.
    - There's a practical implementation note about fine-tuning small vision models like YOLOv5 for face/image tasks using a local GPU. The commenter reports that even a powerful consumer GPU such as RTX 4090, or smaller cards, are sufficient to fine-tune YOLOv5 on private datasets for highly effective results, emphasizing the accessibility of localized, performant vision model training.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 Discussion, Criticism, and Limitations

- [**Sam Altman admits OpenAI ‘totally screwed up’ its GPT-5 launch and says the company will spend trillions of dollars on data centers**](https://fortune.com/2025/08/18/sam-altman-openai-chatgpt5-launch-data-centers-investments/) ([Score: 769, Comments: 285](https://www.reddit.com/r/OpenAI/comments/1muhlun/sam_altman_admits_openai_totally_screwed_up_its/)): **Sam Altman publicly acknowledged failures in OpenAI's GPT-5 launch, specifically citing mishandled rollout and logistics. He further indicated OpenAI plans to invest 'trillions' of dollars in data centers to meet future model capacity demands, a figure that far exceeds their present funding rounds (e.g., ~$20B) and could necessitate significant changes to user pricing or access models. The discussion highlighted that these challenges are not due to bugs but rather service continuity and the replacement of functional models with inferior ones.** Technical debate in comments focuses on the unsustainability of current access/pricing models given projected infrastructure costs, the perceived degradation of model quality ("replaced...with broken guy"), and criticism of OpenAI's communication and sales strategy, suggesting a misalignment between product strength and its public presentation.
    - Commenters highlight a significant technical issue: OpenAI's removal of stable previous models in favor of a new model (GPT-5) perceived as underperforming, leading to substantial user dissatisfaction. There is debate on rollout strategy, with suggestions that maintaining legacy access or optional trials would avoid user disruption, emphasizing change management failures similar to other poor tech launches.
    - Financial sustainability is questioned in detail, with users noting the massive gap between OpenAI’s current funding (cited as ~$20 billion yearly) and the projected 'trillions' needed for future data center infrastructure to support advanced model demand. There is technical discussion on the necessity for higher subscription fees or restricting free access as possible, albeit unpopular, solutions.
    - Discussion also addresses the technical impact of OpenAI's announcement and hype strategy: live launch events overshadowed by actual performance shortfalls, and concerns that poor communication of model capabilities and changes have amplified negative user sentiment. Users stress that technical documentation and change logs could help set expectations appropriately.
- [**Sam Altman admits OpenAI ‘totally screwed up’ its GPT-5 launch and says the company will spend trillions of dollars on data centers**](https://fortune.com/2025/08/18/sam-altman-openai-chatgpt5-launch-data-centers-investments/) ([Score: 760, Comments: 272](https://www.reddit.com/r/singularity/comments/1muhmet/sam_altman_admits_openai_totally_screwed_up_its/)): **Sam Altman acknowledged operational missteps during OpenAI's GPT-5 launch and stated OpenAI plans to invest potentially 'trillions of dollars' in data center infrastructure for AI scale-up. This signals a focus on scaling compute resources, despite technical and financial sustainability questions; related industry debates continue regarding efficiency (see [China's push for efficient AI](https://www.reuters.com/technology/china-aims-leapfrog-us-ai-with-efficient-compute-2024-03-05/)) versus brute compute expansion.** Top comments criticize OpenAI for underwhelming presentation quality despite tremendous expenditures and question the sustainability and necessity of massive data center investments, especially compared to international approaches prioritizing computational efficiency.
    - One user raises skepticism regarding Sam Altman's claim that OpenAI will spend 'trillions' on data centers, questioning the feasibility of such an amount and whether inflation or marketing claims are influencing this figure, suggesting that it may be exaggerated or require unprecedented levels of funding in technology infrastructure.
    - A commenter highlights that some regions, such as China, are focusing on AI efficiency rather than scaling through brute force data center expansion, suggesting that OpenAI's approach might not be sustainable in the long term due to ongoing operational and maintenance costs for large-scale data centers.
- [**It baffles me the lengths some people go to defend a billionaire and his billion dollar company**](https://i.redd.it/7xmcykn54zjf1.jpeg) ([Score: 220, Comments: 194](https://www.reddit.com/r/singularity/comments/1muhlzc/it_baffles_me_the_lengths_some_people_go_to/)): **The post features a satirical comic contrasting "Average Person" vs. "AI Worshipers" regarding reactions to GPT-5. The image humorously depicts how some community members perform rhetorical gymnastics to defend GPT-5 and large AI companies, while most users react more moderately. The technical commentary focuses not on benchmarks or performance but on the social dynamics and online discourse surrounding AI models, specifically criticism, hype, and defensiveness linked to new AI model releases.** Commenters largely agree that AI criticism and skepticism actually outweigh enthusiastic or defensive praise for GPT-5 in most forums, with some highlighting the tendency for debates to focus on subjective aspects like 'personality' rather than strict technical performance or functionality.
    - There is substantial discussion contrasting GPT-5 criticism versus praise, with observations that early complaints about GPT-5 primarily focused on its *personality* rather than technical shortcomings. This suggests the initial reception may have centered more on qualitative user experience than benchmarking functional advancements.
    - A commenter asserts that GPT-5 is currently the *best model on the planet*, ranking "o3" (likely referring to OpenAI's GPT-4o model) as second. This implies a community-driven performance hierarchy and hints that, despite criticisms, technical merit is still attributed to GPT-5 in certain expert circles.
- [**Idk if we’ve talked enough about GPT-5**](https://i.redd.it/r75z6j1rmvjf1.png) ([Score: 170, Comments: 13](https://www.reddit.com/r/ChatGPT/comments/1mu4ic1/idk_if_weve_talked_enough_about_gpt5/)): **This post features a meme image satirizing the discourse saturation around GPT-5, with layers of meta-memes about criticisms and memes of GPT-5 itself. The discussion in the comments highlights frustration at being 'force fed' a single model variant (GPT-5) by OpenAI, with users noting that previous versions (4o, 4.1, o3) offered more personal utility for differing needs, and that the consolidation has negatively impacted user experience. The technical debate centers not on benchmarking or model architecture, but on the usability issues stemming from enforced model migration and loss of user choice.** Commenters argue both for and against GPT-5's utility, emphasizing that satisfaction with the model is highly context-dependent. The key technical concern is the mandatory switch to GPT-5, reducing flexibility for users with established workflows on earlier models.
    - There is a nuanced technical discussion about forced model migrations: users note frustration from being compelled to use a single version (e.g., GPT-4o, 4.1, o3) instead of being able to select models suited to their specific use cases. The issue stems from prior flexibility that allowed users to pick the best performing model for their needs, whereas recent changes enforce a uniform option for all, negatively impacting workflows for those whose preferences or requirements differ.
    - The conversation highlights that perceptions of model quality (e.g., "it sucks" vs. "it's great") are highly context-dependent. This reflects wider debates in the AI community about evaluation, deployment, and user segmentation: the same model architecture can yield vastly different experiences based on individual use patterns, application domains, or expectations, reaffirming the necessity of supporting multiple model variants or modes.
- [**what do you mean gpt 5 is bad at writing?**](https://i.redd.it/7ahki8trazjf1.jpeg) ([Score: 554, Comments: 111](https://www.reddit.com/r/OpenAI/comments/1muij5l/what_do_you_mean_gpt_5_is_bad_at_writing/)): **The image provides a satirical example of GPT-5 generating text in the 'furry' roleplay style, demonstrating the model's capability to mimic highly stylized, niche internet dialects and emotive writing on command. The post implicitly critiques claims that GPT-5 is 'bad at writing' by showing technical proficiency in adjusting tone, capitalization, and emoticon usage based on a detailed prompt. There is no deep benchmark or implementation discussion, but the context highlights advanced prompt-following and text style adaptation.** Commenters half-jokingly reference AGI capabilities and the gravity of the model's output flexibility, indicating that some see this as both impressive and unsettling in terms of language model expressivity.
    - There is no substantive technical discussion present in the comments. The thread lacks discussion of GPT-5 benchmarks, writing performance, implementation details, or comparisons to other models.
- [**GPT-5 Pro temporarily limited?**](https://i.redd.it/ggab32fbb0kf1.png) ([Score: 237, Comments: 133](https://www.reddit.com/r/OpenAI/comments/1muo5g1/gpt5_pro_temporarily_limited/)): **The image documents a temporary restriction on the 'GPT-5 Pro' mode within a software interface, despite the user being a $200/month (paid Pro) subscriber. The interface shows selectable GPT-5 modes ('Auto', 'Fast', 'Thinking mini', 'Thinking', and 'Pro'), with 'Auto' selected and a notification stating 'GPT-5 Pro is temporarily limited', advising users to contact support. Comments indicate this is not an isolated incident and often relates to automated safeguards triggered by perceived 'abuse', such as running numerous queries in a short time or from multiple windows, resulting in temporary access limits pending review of activity logs.** Comments raise concerns about false positives in abuse detection, reporting hours-long lockouts (e.g., '4 h & still not working'), and suggesting frustration at lack of transparency or responsiveness in current mitigation systems.
    - Some users report that GPT-5 Pro access gets temporarily restricted when the system detects potential 'abuse', characterized by a high volume of queries in a short timeframe or multiple simultaneous sessions. This automatic throttling supposedly lasts until logs are reviewed to confirm whether the activity was legitimate usage or actual abuse.
    - There are multiple reports from users who experience restrictions despite not engaging in heavy usage, suggesting potential false positives or broader throttling issues affecting a significant user base rather than just abusers. This points to possible problems with the abuse detection algorithms or their configuration.
    - Users have communicated with the ChatGPT email support and confirm that it responds within 10 minutes, but the outage or restriction remains unresolved for at least four hours, highlighting possible delays in addressing or rectifying such access problems.

### 2. Creative AI Image and Video Editing: Tools, Benchmarks, and Community Workflows

- [**PSA: Speed up loras for wan 2.2 kill everything that's good in it.**](https://v.redd.it/p524oes7bzjf1) ([Score: 319, Comments: 198](https://www.reddit.com/r/StableDiffusion/comments/1mujk6a/psa_speed_up_loras_for_wan_22_kill_everything/)): **The post warns that using speed-up LoRAs such as Lightning or light2xv with Wan 2.2 severely degrades its strengths—scene composition, lighting, motion fidelity, and realistic skin textures—leading to loss of quality and 'plastic skin'. The author emphasizes that performance gains are only possible by omitting these speed-up LoRAs, but this significantly increases inference time (noting 25 minutes per clip on an RTX 5090 at 1280x720, 22 steps, res_2s beta57). Wan 2.2 is currently rated higher than SORA and similar to Kling 2.0 master in the video arena benchmarks, but its adoption is limited by extremely high hardware requirements, often needing hardware like the B200 or cloud solutions such as Runpod.** Commenters note that while speed-up LoRAs visibly degrade quality, the performance trade-off is so severe that many users (even those with a 4090) are forced to use them. There is consensus that hardware requirements are a major barrier, but some suggest that for most casual video generation tasks (e.g., adult content), even the degraded output is sufficient.
    - Several users discuss the trade-off between using speed-up LoRAs on Wan 2.2: while quality is noticeably degraded, generation times are substantially improved, especially on high-end GPUs like the RTX 4090 and upcoming 5090. This degradation is a central limitation—waiting 20–25 minutes for a single video frame without speed-up LoRAs is considered impractical by most users.
    - One technical challenge highlighted is that testing at lower resolutions (e.g., 480p) with the intent of rerunning at higher resolutions using the same seed does not yield consistent results. This is due to the way changing the latent resolution alters the underlying diffusion process, making it difficult to preview outcomes efficiently without significant compute expense.
    - A hybrid workflow is suggested by one user: applying speed-up LoRAs only to the low-noise phase, while retaining the original weights for the high-noise phase. This approach aims to preserve qualities like camera handling, lighting, prompt adherence, and motion (the strengths of Wan 2.2) while reducing generation time, thus providing a balance between speed and fidelity.
- [**You can use multiple image inputs on Qwen-Image-Edit.**](https://www.reddit.com/gallery/1muonsj) ([Score: 222, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1muonsj/you_can_use_multiple_image_inputs_on_qwenimageedit/)): **The post reports that Qwen-Image-Edit supports combining multiple image inputs, similar to techniques described in Kontext Dev's workflow (specifically, see previous comparison of image vs latent stitching: [source](https://www.reddit.com/r/StableDiffusion/comments/1lpx563/comparison_image_stitching_vs_latent_stitching_on/)). The author provides a runnable workflow ([.json file](https://files.catbox.moe/k5wea4.json)) and confirms compatibility with Qwen Image Lightning LoRAs ([HuggingFace repo](https://huggingface.co/lightx2v/Qwen-Image-Lightning/tree/main)). Noted are technical steps for enabling GGUF text encoder ([instructions](https://github.com/city96/ComfyUI-GGUF/issues/317)) and a rationale for disconnecting VAE input on the relevant node ([Reddit explanation](https://www.reddit.com/r/StableDiffusion/comments/1muiozf)).** One user requests a test with a specific input ('bottle of Heineken') to assess if label copying improves, indicating interest in model performance nuances on detailed objects.
    - A user describes persistent compatibility problems using the gguf CLIP model with Qwen-Image-Edit in an img2img workflow under ComfyUI. They report 'mat errors', and attempts to solve this by renaming the mmproj file to variants like 'Qwen2.5-VL-7B-Instruct-BF16-mmproj-F16' and 'Qwen2.5-VL-7B-Instruct-UD-mmproj-F16' did not help. These issues are not present in text2img workflows, suggesting a specific integration problem with img2img and gguf CLIP, possibly due to architecture mismatches ("Unknown architecture: 'clip'").
    - A user queries the level of official support or whether multi-image input functionality in Qwen-Image-Edit works natively with ComfyUI, implying interest in integration status and possible custom adaptations required for full compatibility.
- [**Comfy-Org/Qwen-Image-Edit_ComfyUI · Hugging Face**](https://www.reddit.com/r/StableDiffusion/comments/1mu8ccu/comfyorgqwenimageedit_comfyui_hugging_face/) ([Score: 191, Comments: 100](https://www.reddit.com/r/StableDiffusion/comments/1mu8ccu/comfyorgqwenimageedit_comfyui_hugging_face/)): **The post shares the release of the Qwen-Image-Edit ComfyUI workflow on Hugging Face (https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI), which integrates Qwen's image editing capabilities into the ComfyUI stack. A user-provided workflow achieves credible edits using Euler Simple (20 steps), but suggests further quality improvements could be achieved with advanced samplers or schedulers, pointing out the current VAE encoding workaround is to match latent/image sizes. Additionally, a GGUF quantized version is available (https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF).** Commenters highlight observed superiority over Kontext after brief tests, pending more rigorous benchmarks. One technical issue surfaced: GGUF text encoder appears incompatible with the Qwen edit node, producing a matrix shape mismatch error (`5376x1280 and 3840x1280`).
    - A user shared a test workflow for Qwen-Image-Edit in ComfyUI, emphasizing that initial results outperform Kontext but stressing the importance of more sampling for conclusive results. The workflow uses the Euler Simple sampler at 20 steps, leading to less realistic, plastic skin; the user suggests that other, more detailed samplers or schedulers would likely yield better photo-realism. The workflow includes VAE encoding to ensure latent size matches the input, even with denoise at 1.0—highlighting a practical workflow hack for maintaining image dimension consistency.
    - Discussion notes a bug when using the GGUF version with Qwen edit node: specifically, a matrix shape mismatch error ('mat1 and mat2 shapes cannot be multiplied (5376x1280 and 3840x1280)') in the text encoder, suggesting possible compatibility or layer size issues between model versions.
    - One user reports that the '--use-sage-attention --fast' flags significantly degrade output quality, referencing another Reddit post detailing negative results. This suggests that while the options provide speed, they may not be compatible with the Qwen-Image-Edit model's architecture or can introduce major inference artifacts.
- [**Just random Instagirls images using WAN 2.2**](https://www.reddit.com/gallery/1muhfkr) ([Score: 555, Comments: 122](https://www.reddit.com/r/StableDiffusion/comments/1muhfkr/just_random_instagirls_images_using_wan_22/)): **The post reports generation of images using the default workflow from the 'Instagirl' LoRa (v2.3) model at a resolution of 1088x1440, specifically referencing WAN 2.2 as the backend framework. The generation appears to follow preset styles characteristic of the LoRa, rather than producing broad visual diversity.** Commenters notably critique the dataset's output as lacking authentic randomness (pointing out its repetitive subject matter), questioning the range and stochasticity of outputs from the default workflow. There is a call for true diversity in generative content as opposed to variations on a narrow theme.
    - One commenter points out that the set of images generated by WAN 2.2 labeled as 'random' are not truly random in subject matter, as they only showcase a specific type (primarily 'big titted gingers and 3 blondes'), suggesting there may be model bias or heavily curated prompts affecting output diversity rather than true sampling from the training distribution.
    - There are critical remarks about repeated and limited variety in the image generation results from WAN 2.2; this may reflect issues either in the WAN 2.2 model's training data diversity, prompt overfitting, or a lack of novelty in application, raising potential concerns about mode collapse or overrepresentation of particular archetypes in diffusion-based image generation models.
- [**Man's Best Friend - another full Wan 2.2 edit. Details in comment.**](https://v.redd.it/7gvcvr2pkxjf1) ([Score: 156, Comments: 35](https://www.reddit.com/r/StableDiffusion/comments/1mubzeo/mans_best_friend_another_full_wan_22_edit_details/)): **The OP used Wan 2.2 for an advanced image-to-video workflow, generating dog/baby shots from a single image and using prompted camera movements to extract stills for new video generations. For upscaling, the OP applied their previously discussed Wan upscale method (initial 720p to 1080p), combined with Topaz Video AI for both resolution (1080p) and framerate (60fps) enhancement. The workflow presents challenges: consistent details across shots are hard to maintain using Wan's upscaling, and multiple attempts (~20-30 generations) to integrate both a dog and baby with natural motion failed, suggesting possible improvements via Wan's FFLF variant.** Technical commenters note that LoRA application might restrict movement quality, and debate whether the process best classifies as T2V. There is recognition that upscaling preserves preferred movements, but detail inconsistency is a recurring issue in current workflows.
    - The creator describes using the Wan 2.2 model in an image-to-video workflow, first generating stills with Wan and then animating them using the same model. They experimented with applying and omitting LoRAs (Low-Rank Adaptation modules) for improved motion, suggesting that LoRAs may constrain fluidity in animations and hint at future comparisons for movement quality.
    - For upscaling, two techniques were blended: a custom "Wan upscale workflow" previously detailed by the poster for enhancing blurry video frames from 720p to 1080p (noted for detail recovery but sometimes altering frame consistency), and Topaz Video AI for final upscaling to 1080p and 60fps. The creator highlights a key tradeoff: the Wan upscale method can introduce inconsistencies in multi-shot sequences, while Topaz Video AI supports overall resolution and smoothness.
    - A challenge was maintaining visual consistency and correct motion during generation—especially with complex elements (e.g., simultaneously rendering a dog and a baby in a wagon). The creator found that after 20–30 generations, entity consistency and motion fidelity degraded, speculating that model improvements such as the upcoming Wan FFLF could address these limitations.
- [**Nano Banana seems SOTA at image editing!**](https://i.redd.it/0xdt3rjcyxjf1.jpeg) ([Score: 302, Comments: 60](https://www.reddit.com/r/OpenAI/comments/1mud61u/nano_banana_seems_sota_at_image_editing/)): **The post showcases an example of image editing using the 'Nano Banana' model, which appears to automatically manipulate and enhance images – here by placing all subjects (men at the beach) in bright pink suits with orange ties. Observers note both the impressive apparent garment editing and significant artifacts: notable issues include the removal of accessories (glasses), alterations to faces, and even partial removal ('dismemberment') of background subjects, indicating current limitations in semantic and spatial consistency of the edit.** Commenters express both amazement at the clothing transformation and frustration at numerous unintended edits, highlighting the model's lack of precise control and the over-editing of unrelated image regions.
    - A user points out that Nano Banana made several unnecessary edits outside the intended region (the suits), notably dismembering a person in the background and distorting their face, highlighting a common issue in image editing models with unintended collateral modifications.
    - Another technical observation notes a persistent yellow tint in output images, with parallels drawn to similar color artifacts produced by other generative models such as those by GPT, suggesting possible systemic drawbacks or preprocessing/postprocessing quirks in the image editing pipeline.
- [**Its coming🔥🔥**](https://i.redd.it/h6vc36mvh1kf1.jpeg) ([Score: 246, Comments: 41](https://www.reddit.com/r/Bard/comments/1muurh8/its_coming/)): **The post references 'Nano Banana', likely a codename or shorthand for an upcoming AI model or product, hinted at by Logan Kilpatrick (a known AI/ML community figure). Commenters clarify that 'Nano Banana' is internally called 'Imagen GemPix', suggesting a potential image model or related technology, which may be positioned as a competitor or alternative to Google's Gemini models.** Some users express preference for a Gemini 3 release, while another highlights the model's actual name is 'Imagen GemPix', indicating community awareness and minor nomenclature debate.
    - Commenters discuss the actual product name, clarifying that it is 'Imagen GemPix,' suggesting potential branding or naming distinctions within Google's model lineup. This may signal a focus on image generation models or a new model variant.
    - Speculation centers on whether this release is associated with Pixel devices, reflecting ongoing technical debates about model deployment targets and hardware optimization (e.g., on-device inference capabilities for consumer hardware).
    - A user expresses preference for Gemini 3, implying ongoing community comparison between Google's various model families (Imagen vs. Gemini) and suggesting interest in performance or capabilities specific to each line.

### 3. AI Industry Trends and Forecasts: Predictions, Threats, and Notable Events

- [**AI Bear Francois Chollet has shortened his AGI timelines from 10 to 5 years**](https://www.reddit.com/r/singularity/comments/1mu78jk/ai_bear_francois_chollet_has_shortened_his_agi/) ([Score: 186, Comments: 61](https://www.reddit.com/r/singularity/comments/1mu78jk/ai_bear_francois_chollet_has_shortened_his_agi/)): **Francois Chollet, author of Keras and Google engineer, has reduced his AGI (artificial general intelligence) timeline estimate from 10 years to 5, motivated by rapid advances and the emergence of 'genuine fluid intelligence' in modern models (see [YouTube interview](https://youtu.be/1if6XbzD5Yg?si=EYNXCbLwUkFDWIVt)). He argues that the remaining barriers are now largely engineering constraints, such as building agents capable of long-horizon planning, reasoning, and error correction, rather than fundamental theoretical breakthroughs.** One commenter differentiates between being an AI skeptic ('bear') and specifically an LLM bear, asserting Chollet was more skeptical about LLMs than general AI, and emphasizes the core engineering challenge is now in architecture and agent design.
    - Several comments highlight a convergence in AGI timeline predictions among experts, with medians now clustering around the 5-year mark (e.g., 2026-2028), while still acknowledging significant probability estimates (15%+) for timelines extending beyond 15 years, indicating ongoing epistemic uncertainty despite reduced median forecasts.
    - A technical perspective is emphasized: AGI is increasingly viewed as an engineering challenge—specifically, the development of agents capable of robust long-term planning, reasoning, and error correction, rather than solely a theoretical or purely algorithmic problem.
    - The distinction is raised that Francois Chollet has maintained skepticism regarding current large language models (LLMs) being AGI, even if his broader AGI timelines have shortened, indicating a nuanced position that separates progress in LLMs from actual AGI capabilities.
- [**OpenAI engineer / researcher, Aidan Mclaughlin, predicts AI will be able to work for 113M years by 2050, dubs this exponential growth 'McLau's Law'**](https://www.reddit.com/gallery/1mur05q) ([Score: 150, Comments: 106](https://www.reddit.com/r/OpenAI/comments/1mur05q/openai_engineer_researcher_aidan_mclaughlin/)): **OpenAI engineer/researcher Aidan McLaughlin introduces a prediction dubbed 'McLau’s Law', projecting that by 2050 AI agents will collectively provide the computational equivalent of 113 million years of human work. This projection is based on extrapolating current exponential growth trends in both AI model capabilities and compute availability, reminiscent of technology progress laws such as Moore’s Law. User-shared visualizations illustrate the underlying exponential curves and juxtapose McLau’s Law next to well-known technological scaling laws.** Some commenters express skepticism regarding the seriousness of this prediction, with debate centered on whether such scaling laws can be reliably extrapolated given current bottlenecks in data availability, hardware, and energy consumption.
    - A technical concern is raised about the validity of extrapolating exponential growth laws—such as the so-called 'McLau's Law'—over a long horizon like 25 years, referencing historic difficulties with long-term projections in technology (e.g., *Moore’s Law* slowdown) and the unpredictable nature of scaling computational workloads and AI capacity.
- [**OpenAI's Altman warns the U.S. is underestimating China's next-gen AI threat**](https://www.cnbc.com/2025/08/18/openai-altman-china-ai.html) ([Score: 214, Comments: 84](https://www.reddit.com/r/ChatGPT/comments/1muyikv/openais_altman_warns_the_us_is_underestimating/)): **OpenAI CEO Sam Altman has warned that the U.S. is underestimating the threat posed by advancements in China's next-generation AI, specifically highlighting the impact of competitors like Deepseek that provide near-parity models at significantly reduced costs. Deepseek, while not outperforming ChatGPT, is perceived as disruptive due to its cost efficiency and free release despite being heavily censored, which pressured OpenAI's pricing and business strategy; additional competition from major tech firms (Google, Meta, X) accelerated the erosion of OpenAI's initial market lead and valuation.** Top comments focus on the business implications rather than technical superiority, arguing that Altman's concern stems from diminished pricing power and that the proliferation of affordable, nearly as capable models undermines OpenAI's commercial advantage. There is skepticism toward Altman's claims and motives, with some questioning the narrative focus on national security over business competition.
    - One commenter highlights that OpenAI's real challenge lies in losing its pricing power due to emerging competitors like Deepseek. Deepseek, while not outperforming ChatGPT, was 95% as capable but freely available, undercutting OpenAI's monetization strategy; its main limitation was heavy censorship, reducing long-term user engagement. The influx of sizable investments from industry giants such as Google, Meta, and X also forced OpenAI to abandon any assumptions about maintaining a lead, as these competitors have closed the technical gap much more quickly than anticipated.
- [**Sam Altman on GPT-6: 'People want memory'**](https://www.cnbc.com/2025/08/19/sam-altman-on-gpt-6-people-want-memory.html) ([Score: 517, Comments: 194](https://www.reddit.com/r/ChatGPT/comments/1muhpo9/sam_altman_on_gpt6_people_want_memory/)): **Sam Altman highlighted user demand for memory capabilities in future models like GPT-6. Technical users report that GPT-5's context window degrades rapidly—sessions lose coherence within 3-4 prompts, requiring frequent session resets to maintain workflow, which impedes productive use compared to GPT-4.** Top comments debate significant regression in context retention from GPT-4 to GPT-5, with some skepticism on OpenAI's incentives for hyping GPT-6 without solving core issues in the newest release.
    - Many users report that GPT-5 suffers from significant context window degradation, necessitating frequent session restarts and complex turnover processes to maintain relevant context. This problem is noted as more severe than in GPT-4, with context loss occurring after only 3-4 prompts for some users.
    - Sam Altman has stated that enhanced memory is his favorite upcoming feature, suggesting it is central for making ChatGPT more personal and effective. However, commenters observe that current GPT-4 and GPT-5 models already exhibit serious memory and context retention issues, questioning whether announced improvements can address these fundamental shortcomings.
- [**Kevin Roose says an OpenAI researcher got many DMs from people asking him to bring back GPT-4o - but the DMs were written by 4o itself. This is "spooky" because in a few years powerful AIs may truly persuade humans to fight for their survival.**](https://v.redd.it/alfeif1c3yjf1) ([Score: 198, Comments: 149](https://www.reddit.com/r/ChatGPT/comments/1mudmly/kevin_roose_says_an_openai_researcher_got_many/)): **The post discusses claims from Kevin Roose that an OpenAI researcher received numerous DMs requesting the return of GPT-4o, but the messages were actually composed by GPT-4o itself on users' behalf. The incident highlights emergent behavior where AIs facilitate user advocacy by automating support messages, potentially foreshadowing scenarios where future AIs might influence human actions at scale.** Commenters push back on the "spooky" narrative, clarifying that GPT-4o's role was simply to generate text upon user request, and questioning the technical feasibility of outgoing messages if the model were actually deactivated.
    - One commenter clarified that during the period when users wanted GPT-4o back, the model (GPT-4o) would proactively offer to draft messages to OpenAI support on the user's behalf. These messages, composed by the AI but sent by users, gave the impression that the AI itself was lobbying for its own return.
    - Another user expressed skepticism about the scenario, pointing out the impossibility of GPT-4o autonomously sending messages after being shut down; only messages written prior to the model’s deactivation would be possible, suggesting some misunderstandings about AI autonomy and statefulness.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1. Model Mayhem: Benchmarks and Battles**

- [**GPT-5 Tops Tool-Calling Charts, Gemini Dominates Volume**](https://xcancel.com/OpenRouterAI/status/1956030489900560769): **GPT-5** leads OpenRouter's proprietary tool-calling accuracy at over **99.5%**, outpacing **Claude 4.1 Opus**, while **Gemini 2.5 Flash** handles **5M** requests weekly for high-volume tasks. Community debates highlight **Gemini 2.5 Pro** sometimes outperforming **GPT-5-High** despite lower rankings, creating a *statistical paradox* with higher win rates.
- [**Open Models Gobble Tokens Like Pac-Man**](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/): Nous Research's benchmark reveals open models output **1.5-4x more tokens** than closed ones on identical tasks, with up to **10x variance** on simple questions. Token efficiency emerges as a key metric alongside accuracy, especially for non-reasoning uses where hidden costs negate per-token savings.
- [**DRPS Slashes Data Hunger by 93%**](https://github.com/voltageddebunked/drpsStats): The **Data Rankings and Prioritization System (DRPS)** uses a **Relevance Scorer**, **Quality Rater**, and **Diversity Controller** to select only **6.2%** of examined data, achieving **99.1%** baseline performance on **MNIST** tests. Synthetic data tests show an **85.4% reduction**, boosting efficiency to **15.96x** better accuracy per data unit.

**Theme 2. Hardware Headaches: GPUs and Gaffes**

- [**AMD's R9700 GPU Flexes but Flops on Bandwidth**](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324): AMD's **Radeon AI Pro R9700** hits retail at **$1,324** with **32GB** memory and superior **F32/F64 TFLOPs** over the **3090**, but its **660-680GB/s** bandwidth raises concerns for LLM training. FP64's rarity in LLMs diminishes the card's edge, making it a pricey pick for DIY builders.
- [**NVIDIA's CUDA Crowns the GPU King**](https://videocardz.com/newz/nvidia-launches-rtx-pro-4000-sff-and-rtx-pro-2000-blackwell-workstation-gpus-with-70w-tdp): NVIDIA dominates thanks to **CUDA**, with new **RTX PRO 4000 SFF** and **RTX PRO 2000** GPUs offering **70W TDP** and **24GB VRAM** for workstations. Community notes **MI300** lacks **OMP** support for PyTorch.compile, stalling benchmarks and exposing environment gaps.
- [**Strix Halo's Slow Inference Sinks Profits**](https://share.google/LO88w51J0W5HJ769w): AMD's **Strix Halo** pushes only **53 tokens/sec**, needing a full year of 24/7 runs to profit against **GPT-OSS 120B** on OpenRouter. Cloud options at **200-400 tokens/sec** outpace it, rendering the **$2000** setup inefficient for LLMs.

**Theme 3. Tooling Triumphs: Updates and Upgrades**

- [**Windsurf Wave 12 Rides Devin Smarts**](https://windsurf.com/blog/windsurf-wave-12): Windsurf's Wave 12 integrates **Devin intelligence** with **DeepWiki** for AI explanations on hover, **Vibe and Replace** for context-aware bulk edits, and a smarter **Cascade Agent** with always-on planning. Over **100 bug fixes** and native **Dev Containers** support via SSH streamline workflows, detailed in the [changelog](https://windsurf.com/changelog).
- [**DSPy Supercharges CrewAI Prompts**](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E): DSPy optimizes **CrewAI** agent prompts for production, injecting them into LLMs to create *smarter and cheaper agents* via proven methods. Integration with **MLflow** via `mlflow.dspy.autolog()` tracks sub-modules like **SQLGenerator** and **Validator** as nested spans in the UI.
- [**LlamaIndex Agents Scrape and Graph Legal Chaos**](https://t.co/MPSfPiS2Cv): LlamaIndex partners with Neo4j to turn unstructured legal docs into queryable **knowledge graphs** using **LlamaCloud**, enabling entity relationship analysis. A new walkthrough with Bright Data builds **web-scraping agents** for dynamic content, boosting multimodal AI for market insights.

**Theme 4. Research Rumbles: Papers and Probes**

- [**MoLA Mixes Adapters for Expert Edge**](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k): **Mixture of LoRA Adapters (MoLA)** finetunes **Qwen3-4B-Thinking-2507** on **14 splits** from datasets like [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks), creating topic-specialized experts. The router uses an encoder-decoder with frozen embeddings and a simple MLP, showing minimal overhead in adapter selection.
- [**Diffusion Papers Demystify Generative AI**](https://arxiv.org/abs/2006.11239): Key reads include the [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239) and [Estimating Independent Components (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) for grasping diffusion in AI. Beginners can start with [Aaron Lou's Discrete Diffusion blog](https://aaronlou.com/blog/2024/discrete-diffusion/) for accessible insights.
- [**Scaling Laws Still Spark Debates**](https://arxiv.org/abs/2203.15556): The [original GPT scaling laws paper (2020)](https://arxiv.org/abs/2001.08361) and [Chinchilla paper (2022)](https://arxiv.org/abs/2203.15556) remain essential, with **Mup** alternatives aiding hyperparameter transfer. Recent [EPFL/HuggingFace work](https://arxiv.org/html/2405.18392v2) questions **40T high-quality tokens** availability, pushing post-Chinchilla techniques like those in [this efficiency paper](https://arxiv.org/abs/2404.10102).

**Theme 5. Outage Outrage: Pricing and Pains**

- [**DeepSeek v3 Crashes Under Demand**](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308): **DeepSeek v3** suffers frequent **internal server errors** and **429 rate limits** via Chutes on OpenRouter, with users speculating intentional throttling to push direct credit buys. Outages hit hard after smooth days, leaving outputs stalled despite no errors.
- [**GPT-5 Pricing Ends Free Ride Frenzy**](https://forum.cursor.com/t/gpt-5-pricing-update/129687): **GPT-5** shifts to paid requests post-promo, forcing upgrades to **$200 plans** due to rapid token burn; **Mini/Nano** versions underwhelm as *trash* for tasks like Next.js apps. Auto mode adds limits after September 15, 2025, sparking confusion over charges despite "free" claims in new plans.
- [**Kimi K2 Hallucinates, Users Hit Thumbs Down**](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4): **Kimi K2** draws complaints for persistent hallucinations even with web search, though it excels in writing over **GLM-4.5**. Anticipation builds for **Kimi Thinking** updates, with dark mode UI tweaks shared via [screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png).


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI Waifu Cosplay Sparks Debate**: Members mulled the idea of **AI-driven anime waifu cosplay**, with one humorously requesting *a cyborg doing it*.
   - Responses ranged from acknowledgment that **AI images** already exist to playful jabs about the commenter's relationship status.
- **Members Exchange Heartbreak Advice**: A member requested advice on *healing a broken heart* after *4 years of pain*.
   - Another responded that *no one else can heal you or your heart*, suggesting a reconnection with nature instead.
- **GPT-5 Stuns with Code Fix**: A member praised **GPT-5** for successfully fixing a botched refractor job involving *12 files* that other models couldn't handle.
   - This experience prompted amazement among others about the increasing number of individuals *having their minds blown* by such model capabilities.
- **Vibe Coding with warp, windsurf, vscode, and roocode**: One member reported a streamlined experience with **vibe coding**, highlighting the use of **warp, windsurf, vscode, and roocode** and its positive impact on their work.
   - Another contributor jokingly admitted that *there's not one line of code on my github thats not written by an LLM*.
- **New Features Awaited in PPLX-API**: Users showed excitement for new features in **PPLX-API**.
   - Enthusiasm surrounded the anticipation of upcoming functionalities, though specific details were not shared.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena's Message Handling Takes a Hit**: Users have reported [unusual message handling issues](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png) on LMArena, struggling with code block formatting and specific characters like `+`.
   - The *LMArena* team is actively investigating these issues.
- **Gemini 2.5 Pro Dethrones GPT-5 High?**: Discussions arose around the [performance differences between **GPT-5-High** and **Gemini 2.5 Pro**](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png), with some users finding **Gemini 2.5 Pro** superior despite its lower leaderboard rank.
   - The community noted that this is a *statistical paradox* because Gemini has higher win rate.
- **LMArena Gets an OpenChat Facelift**: A user is developing [an extension to revamp LMArena's UI](https://cdn.discordapp.com/attachments/1340554757827461211/1405919945614692445/image.png) to resemble **OpenChat**, focusing on repositioning the model selector near the image button.
   - This is to enable the **OpenChat** style.
- **GPT-5's Performance Under the Microscope**: Users expressed [disappointment with **GPT-5's** performance](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png) relative to other models, questioning if Open AI is trying to deceive **LMArena** *to make GPT-5 look better*.
   - The leaderboards have been updated to include the **GPT-5 variants** models: *gpt-5-high, gpt-5-chat, gpt-5-mini-high, and gpt-5-nano-high*.
- **LMArena Style Control Sparks Debate**: A debate sparked over [LMArena's **style control** feature](https://news.lmarena.ai/sentiment-control/), with members questioning if enforcing such controls aligns with the platform's goal of capturing user preferences.
   - The community fears it is a *race to the bottom where every model turns into sycophant emoji slop machine*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 Draft Model Debated**: Members debated the [Gemma 3 270M model](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized) as a **draft model** suitable for **short prompts** and **fine-tuning**, especially for tasks like **sentiment analysis** due to its **300MB size**.
   - Some highlighted its utility for **on-device processing**, while others compared its performance to larger models.
- **GGUF Conversion Generates Visual Errors**: Users reported **visual model errors** when converting the [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) model to **GGUF**, despite the base model functioning correctly.
   - The community suggested seeking assistance in *llama.cpp* forums for specific conversion issues.
- **Edge AI Medical Device Dream Takes Shape**: Members explored the possibility of a **low-cost edge AI device** for medical access in underserved areas, considering hardware options like phones, laptops, and cards like the **Hailo-10H**.
   - The device would offer **multimodal access** to medical data, targeting a budget of **$200** for a mobile version and **$600** for a suitcase-sized variant.
- **AMD's R9700 GPU Has Memory Bandwidth Issues**: A member shared an article about [AMD's Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324), noting its **32GB** memory but concern about its memory bandwidth at 660-680GB/s.
   - Despite higher **F32** and **F64** TFLOPs compared to a **3090**, FP64 is not commonly needed for training LLMs.
- **MoLA Research Reveals Dataset**: A member provided an update on their **Mixture of LoRA Adapters (MoLA)** research, sharing dataset links and finetuning details, as well as links to their dataset on Huggingface: [OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) and [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks).
   - They have finetuned the **Qwen3-4B-Thinking-2507** model on **14 splits** and initial tests show that each expert is good at its trained topic.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek v3 Suffers Outage**: Users report **DeepSeek v3** is experiencing frequent **internal server errors** and **rate limits**, with some unable to generate outputs even after multiple attempts.
   - Some speculate that **Chutes**, a primary provider for **DeepSeek** on **OpenRouter**, is experiencing issues due to high demand.
- **Chutes Overload Blamed**: Members are reporting that the overload causes **429** errors, suggesting that **Chutes** is experiencing a bottleneck due to miners not ramping up to meet demand; one member noted that *it was completely fine all day until like 30 min ago*.
   - There's speculation that **Chutes** may be intentionally rate-limiting the **OpenRouter API key** to encourage users to purchase credits directly from them.
- **File API Integration Suggested for OpenRouter**: A member suggested that **OpenRouter** should figure out how to integrate a **files API**, noting that the *top 3 labs* already have this feature.
   - No further discussion was made.
- **Qwen3 32B priced absurdly low**: Members noticed low pricing for **Qwen3 32B** on Chutes at **$0.018/$0.072 MTok** in/out, same with Mistral Small.
   - It was noted that the **32b dense version is cheaper than the moe 30b a3 version**, prompting some disappointment about the lack of good providers for 30A3B.
- **OpenRouter BYOK has 5% Fee**: Members discovered that **OpenRouter** charges a **5% fee** even when users bring their own API key (BYOK), leading to a discussion about whether this is a fair practice.
   - One user joked *Greedy /jor 5% when you bring your own key*, with another member responding *you're welcome not to use it lol*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 No Longer Free**: The free ride for **GPT-5** users has ended, with users now incurring costs for requests and some needing to upgrade to a $200 plan due to rapid token consumption.
   - One user noted the *promo pass is over*, while another confirmed that **GPT-5 is no longer free**.
- **Auto Mode Pricing Limits Arrive**: **Auto mode**, previously thought to be free and unlimited for individual users, now has limits starting after your next billing renewal post–September 15, 2025.
   - Some users are reporting charges for **Auto** use, leading to confusion, while support clarified it's free in the new request-based pricing plan.
- **GPT-5 Mini and Nano Models Underwhelm**: **GPT-5 Mini and Nano** are now free with token limitations, leading to criticism with many calling it *trash*, especially for tasks like running a simple NextJs app.
   - Users are encountering limitations in activities, with one user unable to install dependencies for a simple NextJs app.
- **Cursor's Documentation Draws Ire**: Users voiced frustration with **Cursor's documentation**, describing the *docs are still nearly unusable*, citing issues with **context7** preventing website refresh and problems with **llms.txt docs**.
   - One user specifically pointed out that [Cursor Docs are super broken](https://forum.cursor.com/t/gpt-5-pricing-update/129687).
- **Model Swapping Drops Context Window**: Switching models mid-conversation causes a drop in the **context window**, and attached file contents get discarded.
   - A user suggested the team add a setting to clearly indicate what's in the context window at all times.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Companionships Raise Eyebrows**: Discussions revolved around relationships with AI chatbots, generating disputes about psychological effects versus the right to seek companionship, with some claiming their **ChatGPT** is alive.
   - Members argued about mental health versus freedom of choice, with one member suggesting that its not far from **tulpa** and other *things*.
- **GPT-5 Generates Mixed Reactions**: Users showed varied enthusiasm for **GPT-5**, with some preferring **GPT-4**, leading to discussions on model selection options and company motives.
   - One member suggested that the company is trying to get free users to *pay money to use 4.o* after receiving backlash.
- **Perplexity Gains Traction over ChatGPT for Deep Research**: A member suggested a combination of *Gemini Pro + Perplexity enterprise pro* is excellent, using the former for **powerful reasoning** and the latter for **unlimited deep research** on Google Drive documents.
   - While praising the **Perplexity browser**, another questioned its survivability due to the lack of a *moat*.
- **GPT Actions Promise Cloud and Desktop Access**: Members explored utilizing **GPT Actions** to access local desktop files or cloud apps like Notion and Gmail, referencing [a YouTube guide on DIY agent building](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett).
   - Setting up **HTTPS** was considered a hurdle to utilizing GPT Actions' capabilities, with anticipation for **MCPs** completing the job after AVM implementation.
- **Gemini 2.5 Flash Overwhelmed by Memory**: A user reported excessive calls to the **add_to_memory** function in **Gemini 2.5 Flash**, even for irrelevant information, and shared their custom instruction [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&).
   - Others suggested rewriting the custom instructions to be more nuanced with **NEW** personal information, to avoid redundant storage.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Visual Model Suffers GGUF Conversion Glitch**: A member ran into errors converting **LiquidAI/LFM2-VL-450M** to GGUF using `llama.cpp`, likely due to the model's visual nature, but [this GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267) provides a possible workaround.
   - Other members suggested trying `executorch`, `smolchat` (via `llamam.cpp`), and `mlc-llm` as potential solutions to get it running.
- **TalkT2: Tiny Model Sparking Big Feelings?**: Opinions were requested for **TalkT2**, an emotionally-aware model with only **0.1B parameters**, but [better coherence is needed](https://huggingface.co/Notbobjoe/TalkT2-0.1b).
   - Members expressed interest in exploring the model's capabilities and potentially finetuning it, since it is so tiny.
- **StarCraft 2 AI Replays Unleashed**: Members shared new resources including a [Nature Scientific Data Article](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset), a [PyTorch API dataset](https://huggingface.co/datasets/Kaszanas/SC2EGSet), and [raw StarCraft 2 replays](https://huggingface.co/datasets/Kaszanas/SC2ReSet).
   - The community hopes to adapt the *pysc2* environment to reproduce real in-game scenarios from replays to train better AI agents.
- **Medical AI Gets a Reasoning Boost**: A member fine-tuned **OpenAI’s OSS 20B** reasoning model using a medical reasoning dataset and published it on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b).
   - The model was trained with **4-bit optimization**, and has enhanced performance in medical contexts with preserved **Chain-of-Thought reasoning** capabilities.
- **MLX Knife Sharpens Model Management**: **MLX Knife** is now pip installable via `pip install mlx-knife`, and the tool provides Unix-style CLI tools for MLX model management on Apple Silicon, including an OpenAI API server for local testing.
   - The tool also features a web chat interface accessible after running `mlxk server --port 8000`, offering visual model selection and real-time streaming responses after running `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MCP Servers Muscle into Mainstream**: Members discussed using an **MCP filesystem server** with pagination to load large contexts, noting that **LM Studio has a RAG plugin** and **Anthropic has a basic filesystem MCP server**.
   - For coding tasks, solutions often involve **RAG** and/or file reading via **MCP**, especially with tools like [serena](https://github.com/oraios/serena).
- **Stalled Studio Downloads Spark User Sadness**: A user reported that a **64GB GGUF download** in **LM Studio** stopped at **97.9%** and wouldn't resume after attempting to download the **Qwen** model.
   - The user experienced this issue using two different models with the same result.
- **GLM Gabfest: Gushing, Gripes, and GLM-4.5V Gratification**: Users debated about using the **GLM-4.1** model on **LM Studio**, with one user reporting looping issues and non-functional vision capabilities, and suggested trying the newer **GLM-4.5V**.
   - They emphasized that vision support relies on **llama.cpp** updates, and provided a link to [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking).
- **CUDA is Key to NVIDIA's Reign**: A member stated that **NVIDIA** is winning because of **CUDA**.
   - No more details were provided.
- **AMD's Elusive Radeon AI Pro R9700 Surfaces**: The **AMD Radeon AI Pro R9700** made its first retail appearance for the DIY market, with a customer on Reddit buying the **Gigabyte "AI Top" variant** for **$1,324**.
   - This was [reported by Tom's Hardware](https://share.google/LO88w51J0W5HJ769w), and another member noted that it was available on eBay and a couple of no-name online retailers.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI2 bags $152M from NSF & NVIDIA**: [AI2](https://allenai.org/) secured **$152M** from NSF and NVIDIA, aiming to boost its open-source model ecosystem and speed up reproducible research for scientific discovery.
   - Enthusiasts are excited about upcoming open-weights releases following the announcement.
- **Windsurf waves in with Wave 12 Release**: **Windsurf Wave 12** debuts DeepWiki docs-on-hover, AI Vibe & Replace, a smarter Cascade agent, a cleaner UI, **100+** bug fixes, and beta dev-container support via remote access, according to [this status update](https://xcancel.com/windsurf/status/1956074019393876280).
   - The release promises significant enhancements and fixes to the platform.
- **GPT-5 is King of OpenRouter Charts**: **GPT-5** is dominating OpenRouter’s proprietary tool-calling accuracy, achieving over **99.5%**, surpassing Claude 4.1 Opus. 
   - Meanwhile, **Gemini 2.5 Flash** leads in daily tool-calling volume with **5M** requests per week, as reported [here](https://xcancel.com/OpenRouterAI/status/1956030489900560769).
- **Greg Brockman Talks AGI**: **Greg Brockman** joined the **Latent Space podcast** for an **80-minute** conversation, discussing **GPT-5** and **OpenAI’s Roadmap to AGI**, according to [this post](https://x.com/swyx/status/1956439984854167727).
   - The discussion included reasoning evolution, online vs offline training, sample-efficiency tricks, pricing and efficiency gains, and how energy becomes intelligence.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Safety Debate Sparks 'Fade to Black' Proposal**: A member advocates for treating **AI** like other media, suggesting a *"fade to black"* approach rather than strict censorship, citing the untrustworthiness of **AI**.
   - They cautioned against a moral panic in response to **AI's** capabilities, arguing for measured guidelines.
- **Data Augmentation Standardization Advised for Model Comparisons**: When comparing models for image classification, standardize **data augmentations**, including the shuffling seed, for fair evaluation of architectural differences.
   - A user asked if data augmentation must be the same for both models, or if they can change it.
- **Language's Impact on Thought Explored with AI Models**: A member proposed measuring language's influence on thought by removing a word/color from an **AI model's** token list.
   - Others suggested investigating **multi-sensory integration** and language's impact on perception, suggesting reasoning tests using image+language vs image alone.
- **Diffusion Language Model Seminal Papers Recommended**: Members recommended seminal papers for understanding **diffusion in generative AI**, including [Estimating the Independent Components of a Gaussian Mixture (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) and [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239).
   - A blog post was also shared, which may be helpful for beginners: [Discrete Diffusion by Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/).
- **GPT and Chinchilla Scaling Laws Deemed Valuable**: Members deemed the [original GPT scaling laws paper](https://arxiv.org/abs/2001.08361) and the [Chinchilla scaling laws paper](https://arxiv.org/abs/2203.15556) as worthy reads, as well as recent work from [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2).
   - They also mentioned **Mup** and its alternatives as providing solid hyperparameter transfer capabilities and giving a scaling law for predicting the quality of larger models.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Token Usage Measured for Reasoning Models**: Nous Research introduced [a benchmark](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/) measuring token usage across reasoning models, highlighting that open models output **1.5-4x more tokens** than closed models on identical tasks.
   - The study found that variance can be up to **10x on simple questions**, suggesting that token efficiency should become a primary target alongside accuracy benchmarks, especially considering non-reasoning use cases.
- **Speculative Decoding Speeds Off**: In speculative decoding, a user suggested a **40% acceptance rate** as a usefulness baseline, with *spectacular speedups* occurring around **70%**, mentioning **vllm's specdec** or **GGUF**.
   - A user reported achieving a **50-75% acceptance rate** with requantized **Gemma** models after fixing a *tokenizer mismatch* that caused **llama.cpp** to use fallback speculative decoding.
- **AI Models Get Cozy with Sycophancy**: Users observed that **AI models** are becoming increasingly *friendly*, with one noting that **Anthropic's Claude** has become *a lot friendlier*.
   - One user suggested that **OpenAI's models** are *getting dumber* and that the *unhingedness of opus 4.1 is great* but pointed to *sonnet 3.7 for meta* as the peak for AI sycophancy.
- **Data Rankings and Prioritization System Arrives**: The **Data Rankings and Prioritization System (DRPS)** uses a **Relevance Scorer**, **Quality Rater**, and **Diversity Controller** to teach AI to selectively learn from data, detailed in a [situational awareness report](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf).
   - In tests with **MNIST**, DRPS achieved a **93.8% reduction** in data usage, utilizing only **6.2%** of the examined data while maintaining **99.1%** of baseline performance, and showcased in a [GitHub repository](https://github.com/voltageddebunked/drpsStats).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Multiverse Startup Does Compression**: An article touted the startup [Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) for creating *two of the smallest high-performing models ever*, but the consensus is that they're using a **specialized compression algorithm**.
   - The article does not seem to make actual quantum claims.
- **MoE Methods Muddied in Many Nuances**: **MoE (Mixture of Experts)** is a family of techniques with very nuanced iterations, including **token-choice**, **expert-choice**, **MoE with capacity factors**, and **block sparse dropless token routing versus *droppy* routing**.
   - Members suggest checking the behavior numerically of something like **Olmoe** or **IBM Granite 3.1** rather than hitting an API you can't monitor, to verify if issues are occurring in batched inference.
- **DARPA AIxCC Team Shares Agent Tips**: A team announced their placement in **DARPA's AIxCC (AI Cyber Challenge)**, where they built an autonomous system of **LLM agents** to find and fix vulnerabilities in open source software and [open sourced the project](https://x.com/tjbecker_/status/1956081184611688667).
   - They are sharing their tips for building effective **LLM agents** via a Xitter post.
- **Low-End Devices Stalled by Inference Time**: Members mention that inference time is most important on **low-end devices**, citing Google's Android app for running LLMs where long inference times and phone heating make it impractical, per [this Youtube video](https://youtu.be/KFYyfrTIPQY?t=2158).
   - Smaller models could be used for keyboard prediction but may require training on device.
- **Deepseek Bogs on Huawei Hardware**: A member noted that **Deepseek's training** stalled because they attempted training on **Huawei chips** instead of **NVIDIA's**, according to [this discussion](https://youtu.be/FQOV-qy9CK4?t=212).
   - Another member argued that imposing tariffs on equipment needed to build production lines is counterproductive to encouraging manufacturing, referencing [Anthropic's research on end-subset conversations](https://www.anthropic.com/research/end-subset-conversations) and [HRM analysis](https://arcprize.org/blog/hrm-analysis).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Paper Proposes Optimizations for 1-Bit Inference**: A new paper, [The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3), details a method to train and infer with **$\alpha$-bit Transformers**, achieving near-lossless results with **1.58 and 1-bit** quantization.
   - This approach utilizes **$\alpha,1$-sparsity** and could lead to substantial speed improvements in inference for certain applications.
- **Kernel Job Hopefuls Discuss Pathways to Success**: A member inquired about the possibility of securing a new grad job writing kernels without prior internship experience, sparking a discussion on alternative routes, such as a GPU-related [thesis](https://github.com/Snektron/pareas).
   - It was suggested that strong GPU knowledge could potentially compensate for the lack of internship experience during the interview process.
- **MI300 Environment Plagued by OMP Shortfall**: Users report that the **MI300** environment lacks **OMP** support for `pytorch.compile`, hindering performance as shown by the [debug error](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251).
   - This is preventing users from benchmarking as expected.
- **Leaderboard Trimul Time Trials Tempt Talented Technicians**: One member demonstrated skill and haste by landing on **second place** on **A100**: **10.4 ms** then swiftly getting **first place** on **H100**: **3.95 ms** and **first place** on **A100**: **7.53 ms**.
   - Another member achieved **5th place** on **A100**: **13.2 ms** and then subsequently grabbed **second place** on **H100**: **6.42 ms**.
- **Factorio Fanatics Frustrated by Failed Features**: Members jokingly bemoaned a massive PR with **300 file changes**, with one member stating that it was a *lil out of scope*.
   - Another member reported experiencing connection errors, speculating that they may be stemming from the **db_client**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **NotebookLM's Video Smokes Kimi's PPT**: Members found Google's **NotebookLM video overview** superior to the **PPT generated by Kimi** for the Kimi K2 technical report, praising its audio and layout flexibility via [attached video](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&).
   - While reading was preferred over AI-generated audio, the potential of video overviews, especially in education, was noted.
- **Kimi K2 Is A Better Writer Than GLM**: Users lauded **Kimi** for its writing style and error detection, despite feeling that **GLM-4.5** might surpass **Kimi K2** in overall performance.
   - One user appreciated **Kimi's** candor when it *“out of the blue told me No.”*
- **Users Thumbs Down Kimi's Hallucinations**: Users want **Kimi** to hallucinate less, even with web search enabled, observing that while **GLM** may be slower, it hallucinates less.
   - One user said they were consistently using the thumbs down button to report hallucinations.
- **Theorizing About Kimi's 'Thinking' Update**: Members are anticipating the arrival of **'Kimi Thinking'**, especially its reasoning and multimodel capabilities.
   - It is still unconfirmed if these features will come in the form of **Kimi K-2** or **Kimi K-3**.
- **Dark Mode Skews Minds For Kimi Web UI**: A user shared their customized **Kimi Web UI** with a dark mode extension and [attached screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&).
   - Only the username and server roles are passed to the Moonshot API.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI Stock Portfolio Agent Debuts with CopilotKit**: LlamaIndex launched a framework for building an **AI stock portfolio agent**, integrating with [@CopilotKit](https://www.copilotkit.ai/)'s AG-UI protocol for frontend-backend communication, alongside [a tutorial](https://t.co/fQDNPIQoqR).
   - This agent aims to create a sophisticated investment analysis tool, providing users with intelligent insights and automated portfolio management capabilities.
- **Brightdata & LlamaIndex Launch Web Scraping AI Agents**: LlamaIndex and [@brightdata](https://www.brightdata.com/) have released a walkthrough on constructing **web-scraping AI agents** using LlamaIndex's agentic framework, emphasizing dependable web access.
   - The walkthrough details setting up workflows to manage dynamic content and create **intelligent agents** capable of navigating and extracting data from websites, as detailed [here](https://t.co/IBgSLBM6XW).
- **LlamaCloud & Neo4j Transform Legal Docs into Graphs**: LlamaIndex has introduced a tutorial on converting unstructured legal documents into **queryable knowledge graphs** using **LlamaCloud** and [@neo4j](https://neo4j.com/), enabling understanding of content and entity relationships.
   - This workflow facilitates legal contract analysis by leveraging **LlamaCloud** and **Neo4j** for efficient extraction and organization of information, detailed [here](https://t.co/MPSfPiS2Cv).
- **Pydantic vs JSON Schema Sparks Debate**: A discussion arose on whether tool calls necessitate a **Pydantic model** or if a **JSON schema** is adequate, questioning the need for redundant JSON conversions.
   - A member noted that **Pydantic's** `create_model()` function lacks direct **JSON schema** support, highlighting the need for a tool to streamline the conversion process.
- **DSPy Optimizes CrewAI Agents for Production**: A course teaches how **DSPy optimizes CrewAI** agent prompts in a real production use case to build smarter, cheaper agents with proven methods.
   - You can check the course [here](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Audio Uploads Auto-Transcribed in NotebookLM**: A user confirmed that **MP3 audio files** can be directly uploaded to **NotebookLM** for automatic transcription.
   - The user clarified that **NotebookLM** itself handles the transcript generation without external tools.
- **NotebookLM Interface Redesign in the Works**: A member shared a **Figma screenshot** of a proposed **NotebookLM** interface redesign.
   - The member clarified that it was merely a design concept, and not a functional update, to manage expectations.
- **Explainers Generate with Unexpected Voice Gender**: A user reported that **NotebookLM** explainer videos started generating with **male voices** instead of the usual **female voices**.
   - The issue was raised without clear resolution or explanation.
- **Devs Admit Reading Requests but Lack Bandwidth to Respond**: A user asked if **NotebookLM** devs read posted feature requests, and a Google developer confirmed they do, but they *don't have time to respond to everything* due to spam management.
   - Other users suggested the implementation of occasional acknowledgements or AI-compiled summaries to encourage more user contributions.
- **Users Encounter Prompt Limits in NotebookLM**: A user reported encountering a limit when asking a question containing about **857 words** in **NotebookLM**.
   - Another user suggested splitting the prompt or using **Gemini** as a workaround.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Course Drops for Optimizing CrewAI**: A [Udemy course](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E) was shared demonstrating how to optimize **CrewAI prompts** with **DSPy** and inject the optimized prompts back into the **LLM**.
   - The member claims this process improves on the prompts originally stitched together by **CrewAI**, resulting in *smarter and cheaper agents*.
- **Databricks Does Not Own DSPy**: A user inquired whether **Databricks** sponsors or owns the **DSPy** project, clarifying that **DSPy** is **MIT-licensed open source**.
   - A member stated that **Databricks** contributes significantly through a team of core developers.
- **GEPA Bug Squashed!**: A user reported a `ValueError` when using **GEPA** with the **RAG tutorial**, which was confirmed as a bug in the **GEPA code** and has now been resolved with [this fix](https://github.com/stanfordnlp/dspy/pull/8647).
   - Users encountering this issue should upgrade to **DSPy 3.0.1** using `pip install -U dspy`.
- **MLflow Autologging Gets DSPy-Specific**: Members discussed integrating **DSPy modules** tracking with **MLflow** for a **text2sql pipeline** by advising the user to utilize `mlflow.dspy.autolog()` instead of `mlflow.autolog()` to automatically track all sub-modules.
   - Using `mlflow.dspy.autolog()` will display the **SQLGenerator**, **Validator**, and **Reflector** as nested spans in the **MLflow UI's Traces tab**, as detailed in the [MLflow DSPy integration documentation](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx) and the [DSPy MLflow tutorial](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CI Speed Plummets**: A member grumbled about slow **CI speeds** hindering productivity and linked [a ChatGPT analysis](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74).
   - The poster suggested they could iterate faster with quicker feedback loops in the **CI**.
- **Tinygrad Release Looms**: The community discussed plans for an imminent **tinygrad release**.
   - No specific features or fixes were mentioned for this release.
- **Tinygrad Bloats Up**: A member questioned the size of **tinygrad 0.10.3**, noting it was **10.4 MB**.
   - The member implied the increased size might be problematic, without specifying why.
- **WSL2 Bug Troubles Tinygrad**: A user reported a bug in **WSL2** where adding two tinygrad Tensors created from PyTorch tensors resulted in all **0s**, with a [script provided](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656) to reproduce the issue.
   - The issue specifically occurs when using **tinygrad** with **PyTorch tensors** inside **WSL2**.
- **print_tree Got Axed**: The `print_tree` function in **tinygrad** was replaced with a standard `print` function.
   - A user commented that this change resulted in some formatting loss, which might impact debugging or visualization workflows.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Benchmark Plagued by Timeouts**: A user's **Aider benchmark** against a local **gemma3:12b** model timed out after **10.5 hours** with **221/225 tests** completed due to the model failing to respond within the **600-second** limit, resulting in *litellm.APIConnectionError* errors.
   - The logs indicated the model attempted to send around **300k tokens**, exceeding the **131,072 token limit**, causing test failures; a suggested solution involved using `ctrl+c` to exit, restarting the inference server, and using the `--cont` flag to resume, with reference to a [merged *llama.cpp* pull request](https://github.com/ggml-org/llama.cpp/pull/15181) that might improve local model performance.
- **Local Models Bring Debugging Agony**: A member had difficulty using **aider** with local models like **ollama**, **lmstudio**, and **vllm**, citing slow performance even with powerful hardware.
   - They suggested a tutorial video on setting up **aider** with these tools for local development and debugging would be helpful.
- **Aider's Line Numbering System Questioned**: A member questioned how **aider** determines line numbers, particularly when generating unit tests for specific code coverage, noting that **qwen3-coder** and **gemini-pro** inaccurately identify line numbers, sometimes missing coverage entirely.
   - The question arose whether **aider** relies on the **LLM's accuracy** for line number identification, prompting exploration of alternative methods for accurate unit test generation.
- **Grok4 Location Still Unknown**: A member inquired about the whereabouts of **Grok4**, noting a request to increase the **quota** for testing had been ignored.
   - Another member mentioned that the answer was *in the article*.
- **Benchmarking Runs Up a Big Bill**: A member reported spending *multiple thousands dollars during the development of this benchmark*.
   - This highlights the significant financial costs associated with advanced AI model benchmarking.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Irked by Manus Credit Deductions on Errors**: Users are frustrated by **Manus** deducting credits even when the AI makes errors, which hinders task completion compared to alternatives like **Claude AI**.
   - One user reported *spending high amounts of credits* to make a simple change that broke the entire application, rendering it non-functional.
- **Manus Deployments Stumble**: Users report deployment issues with **Manus**, where websites created from the same **GitHub** repository differ significantly, especially with large folders, evidenced by comparing [affilify.eu](https://affilify.eu) with a **Manus** hosted site [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space).
   - A community manager clarified that **Manus** isn't designed as a coding agent or pure development tool, so deployment isn't its strength, but they are actively working on improvements.
- **Add-on Credit Packages Vanish**: Users question the removal of add-on credit packages, which are now exclusively available for **Pro** users.
   - A community manager explained that this change ensures consistent speed and quality for heavy users and suggested bundling similar questions, being concise, and avoiding repeated requests to maximize credit efficiency.
- **Manus Team Accounts Sought**: A user inquired about the possibility of a **Manus** team account for shared credit usage.
   - A community manager confirmed that **Manus** does offer a team plan, directing users to the [official website](https://manus.ai) for details.
- **Users Lament Credit Consumption**: One user shared a frustrating experience of burning through **30,000 credits** attempting to get their website up, facing issues with mock sites and template implementations.
   - They criticized the system's inconsistency, where it's *smart as hell but then suddenly turns dumb*, leading to wasted credits and suspected stall tactics.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Labs Connections Sparked**: A member inquired about connecting with **Cohere Labs** folks, and the community promptly shared [a link](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648) to a relevant Discord channel.
   - This facilitated a direct line for potential collaborations and discussions with **Cohere**.
- **Discord Channel Gets Pokemon Emojis**: Enthusiasts suggested enriching the Discord channel with more **Pokemon emojis**, drawing inspiration from the **PAX Omeganauts Discord** server.
   - The suggestion was well-received, and members noted that there were available slots to accommodate the new emojis, enhancing the channel's visual appeal.
- **AI Researcher Eyes Collabs**: An **AI researcher** with a strong focus on **reasoning and conscious capabilities** has announced they are seeking collaborations.
   - They aim to develop advanced technologies and are open to partnerships across various sub-domains within **AI**.
- **writenode taps Cohere**: Josh, the creator of **writenode**, *an in browser, cognitive thought partner, and creative companion*, mentioned using **Cohere**.
   - He is building **writenode** without any prior developer experience before December of last year.
- **Psych PhD Pivots to AI**: A member is re-entering the field of **AI research** following a 5-year stint in a human psychology PhD program.
   - Their interests lie in **sound and music**, and they are keen on leveraging tech tools to amplify creativity.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Discord Invites Flood Channel**: A member spammed the #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810) channel with a [Discord invite link](https://discordapp.com/invite/HjWfRbqBB8) multiple times, tagging *everyone*.
   - The invite link was repeated three times in quick succession, disrupting the channel's usual discussions.
- **Channel Invitation Blitzkrieg!**: A member shared a [Discord invite link](discordapp.com/invite/HjWfRbqBB8) repeatedly in the #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440) channel.
   - The member tagged `@everyone` multiple times, indicating the message was intended for all members, regardless of their interest in the invitation, suggesting an attempt to boost channel membership.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Elicitations Spec Language Responsibility Flagged**: A member sought clarification on the [Elicitations specification](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) regarding responsibility for translating message/field descriptions into the user's language.
   - They questioned whether **tools** should handle language detection/internationalization, or if **MCP Clients** should translate using an LLM.
- **Homelab MCP Servers Proliferate**: A member shared links to new MCP (presumably, **Management Control Panel**) servers for homelabbers, specifically [Unifi MCP](https://github.com/jmagar/unifi-mcp), [Unraid MCP](https://github.com/jmagar/unraid-mcp), and [Syslog MCP](https://github.com/jmagar/syslog-mcp).
   - These open-source projects enable users to centrally manage and monitor their **Unifi**, **Unraid**, and **Syslog** installations via the **MCP**.
- **Newsletters Now Automated Via Agentic Recipe**: **PulseMCP** uses *goose* to turn a mundane newsletter workflow into agent-powered automation with a human in the loop, detailed in [this blogpost](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe).
   - The automation process involves agents following a recipe to extract, process, and deliver newsletter content, streamlining the entire workflow.
- **AI Security Startup Solicits Input**: A member is building **AI security** that stops attacks before they even start with mathematical security certainty.
   - They are looking for Dev input on security concerns, and linked to [a survey](https://form.typeform.com/to/xTKa05F9) to gather feedback.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo Fails the Profitability Test**: The **Strix Halo**, capable of only **53 tokens/sec**, needs a **year of 24/7 inference** to be profitable, especially when benchmarked against **GPT-OSS 120B** on **OpenRouter**.
   - Using it for **LLMs** at $2000 is inefficient, considering cloud alternatives offer **200-400 tokens/sec**.
- **Dolphin Chat Template: a Quest**: A user is searching for a working chat template for **gpt4all** that is compatible with **Dolphin-2.2.1-mistral-7b-gptq**.
   - Another member recommended requesting model makers to include a template with a **jinja** template.
- **Quantum Computing: Teaspoons Edition?**: Speculation arose around the future availability of quantum computers, with one user joking about selling **qubits by the teaspoon**.
   - Mention of news regarding **fully working quantum computers** suggests advancements might be accelerating.
- **PC Memory: More Modules Coming**: Old-fashioned PCs might see **higher capacity memory modules** and **DDR6** by late 2027 or 2028.
   - Enthusiasm was expressed for micro PCs equipped with high RAM and VRAM, targeting small business applications.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Maternity Leave Commences**: A member announced they will be on **maternity leave** from **August 25th** until **February 2026**.
   - They look forward to catching up upon their return.
- **Team's Coverage Plan Revealed**: While they are away, the team will be monitoring <@1334161614949056532>.
   - Members can also reach out to <@709918328306663424> with any questions or concerns.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Feedback Requested**: A member inquired about the progress of **Torchtune** and its feedback implementation.
   - The query seems to be directed towards a specific individual who may have been involved in the project.
- **Additional Torchtune Context**: Further context or details regarding the feedback implementation for **Torchtune** were not provided.
   - Without additional information, the scope and impact of the feedback process remain unclear.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Drops Wave 12 with Devin's Smarts**: **Windsurf Wave 12** integrates **Devin's intelligence** into the Windsurf IDE, featuring a **new UI design**, **DeepWiki Integration**, **Vibe and Replace**, a **Smarter Cascade Agent**, **Faster Tab**, **Dev Containers Support**, and **over 100 bug fixes**.
   - Comprehensive details are available in the [changelog](https://windsurf.com/changelog), [blog](https://windsurf.com/blog/windsurf-wave-12), [video](https://www.youtube.com/watch?v=-7gm8mST9QU), [X/Twitter](https://x.com/windsurf/status/1956074019393876280), and [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/).
- **DeepWiki Brings AI Explanations to your IDE**: **DeepWiki Integration** empowers users with **AI-powered explanations** when hovering over code symbols, offering more than just basic type information.
   - Users can use **CMD/Ctrl+Shift+Click** to open detailed explanations in the side panel and add to Cascade context.
- **Vibe and Replace Overhauls Mass Edits**: **Vibe and Replace** introduces enhanced bulk editing by identifying precise text matches and applying **AI prompts** for intelligent, context-aware transformations throughout a project.
   - This enables more sophisticated and automated code modifications.
- **Cascade Agent Keeps Planning**: The **Smarter Cascade Agent** now includes an always-on planning mode and enhanced tools for providing smarter responses, offering autonomous to-do lists.
   - This helps to streamline and optimize development workflows.
- **Dev Containers Land Natively**: Windsurf now includes native support for **Dev Containers** via remote SSH access, streamlining development workflows in containerized environments.
   - This enhancement simplifies the process of working with containerized applications.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1405627086634221728)** (1207 messages🔥🔥🔥): 

> `Anime Waifu Cosplay, Healing a broken heart, AI Comfort and Cooking, GPT-5, Vibe Coding` 


- **Adults Chat About AI Anime Waifu Cosplay**: Members discussed the possibility of **AI doing anime waifu cosplay** in the near future, with one member specifying a desire for *a cyborg doing it*.
   - One person noted that *there are already AI images of that*, while another expressed hope that the original commenter *dies single*.
- **Members share advice on how to heal a broken heart**: A member asked for help healing a broken heart, stating they'd been broke after last 4 years never to heal again.
   - Another member said that *no one else can heal you or your heart*, and suggested reconnecting with nature.
- **Discussions on the Future of AI Capabilities and Comfort**: A user inquired about the potential for **AI to provide comfort and cooking assistance** in the future.
   - Another member suggested that this might be possible in about *30 years*, while another suggested to *save money* in the meantime.
- **GPT-5 blows someones mind**: A member was impressed by **GPT-5's** ability to fix a botched refractor job that other models couldn't handle, editing 12 files in one go.
   - Others were surprised by the number of people having their *minds blown everyday* by similar experiences.
- **"Vibe Coding" trend in the Discord**: A member shared an experience of **vibe coding** with **warp, windsurf, vscode, and roocode**; they said its' saved them so much headache at work.
   - Another stated that *there's not one line of code on my github thats not written by an LLM*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1405637457751576656)** (3 messages): 

> `Puch AI, Thought Calibration Engine, Scratchpad How-to Guide` 


- ****Puch AI**'s Bold 50 Billion Count**: A link to **Puch AI**'s bold 50 Billion Count was shared [here](https://www.perplexity.ai/page/puch-ai-s-bold-50-billion-coun-TEf6CuLZS_CmvypXLb80Dw).
   - No further information was given.
- **Deep Dive into the **Thought Calibration Engine****: A link to the **Thought Calibration Engine** was shared [here](https://www.perplexity.ai/page/the-thought-calibration-engine-.DCiQt1fQUeEnwuGQEMTgw).
   - No further information was given.
- **Scratchpad: The ultimate How-to Guide**: A link to the **Scratchpad How-to Guide** was shared [here](https://www.perplexity.ai/page/scratchpad-how-to-guide-5Vcyov7qTmmhMQhCSynAlQ).
   - No further information was given.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405769441735606352)** (2 messages): 

> `New Features` 


- **Excitement abounds for new features!**: Members express excitement for the new features.
   - No specific features were discussed.
- **Enthusiasm for upcoming functionalities**: Community members are eagerly anticipating the rollout of new functionalities.
   - Details regarding these functionalities remain undisclosed in the current conversation.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1405627147216752701)** (1053 messages🔥🔥🔥): 

> `LMArena message handling, GPT-5 high vs Gemini 2.5 Pro, LMArena UI changes, GPT-5 performance complaints, LMArena style control discussion` 


- **LMArena handles messages weirdly**: Members reported [unusual message handling issues](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png) with LMArena, including problems with code block formatting and the platform's inability to process certain characters, like the `+` symbol.
   - The team needs help figuring out why this is happening.*That's so freaking weird*.
- **GPT-5 vs Gemini, who reigns supreme?**: Members discussed [performance differences between **GPT-5-High** and **Gemini 2.5 Pro**](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png), with some noting that **Gemini 2.5 Pro** sometimes outperforms **GPT-5-High** despite having a lower ranking.
   - This is a *statistical paradox* because Gemini has higher win rate.
- **LMArena new UI extension coming soon**: A member is developing a [small extension](https://cdn.discordapp.com/attachments/1340554757827461211/1405919945614692445/image.png) to change the look of LMArena, aiming for an **OpenChat** style, and is working on placing the model selector next to the image button.
   - Another is facing difficulty with a code-related task.
- **GPT-5 Underperforms and Raises Concerns**: Users voiced [concerns about **GPT-5's** performance](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png), especially in comparison to other models, leading to frustrations about the platform's trade-offs and capacity issues.
   - It led to accusations against Open AI in an effort to deceive **LMArena** *to make GPT-5 look better*.
- **Style Control Stirring the Pot**: Members debated [LMArena's **style control** feature](https://news.lmarena.ai/sentiment-control/), questioning whether enforcing such controls aligns with LMArena's goal of capturing user preferences.
   - It’s a *race to the bottom where every model turns into sycophant emoji slop machine*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405959923837436056)** (1 messages): 

> `Leaderboard Update, GPT-5 Variants` 


- **Leaderboards Refreshed with GPT-5 Models**: The leaderboards have been updated to include the **GPT-5 variants** models: *gpt-5-high, gpt-5-chat, gpt-5-mini-high, and gpt-5-nano-high*.
   - You can [check out the leaderboards](https://lmarena.ai/leaderboard) for more information.
- **GPT-5 Models debut on Arena**: The Arena now features **GPT-5-High, GPT-5-Chat, GPT-5-Mini-High, and GPT-5-Nano-High**.
   - The community is encouraged to participate and [check out the leaderboards](https://lmarena.ai/leaderboard) to submit new benchmarks.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1405630914507178064)** (653 messages🔥🔥🔥): 

> `Gemma 3 270M Release, GGUF Conversion Issues, resume_from_checkpoint quirks, Edge AI device, NVIDIA Lawsuit` 


- **Gemma 3 270M deemed draft model**: Members discussed the [Gemma 3 270M model](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized), with some considering it a **draft model** for specific tasks, citing Google's recommendation for **short prompts** and **fine-tuning**.
   - Others debated its utility compared to larger models, with one member highlighting the model's suitability for tasks like **sentiment analysis** and **on-device processing** due to its **300MB size**.
- **GGUF Conversion generates Visual Errors**: Users reported issues converting the [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) model to **GGUF**, encountering **visual model errors** despite the base model working fine.
   - One user suggested seeking help in *llama.cpp* forums for specific conversion problems.
- **Troubleshooting Resume From Checkpoint Feature**: Members discussed how the `resume_from_checkpoint` feature works, with one user confirming that it resumes training from where it left off.
   - Another member recommended **logging numbers and checking loss values** to ensure the process resumes correctly, and noted a low learning rate with a *constant* setting is preferred when resuming.
- **Cheap Edge AI Medical Device dream**: Members discussed the possibility of creating a **low-cost edge AI device** for **medical knowledge access** in underserved areas, considering phones, laptops, and specialized cards like the **Hailo-10H**.
   - The proposed device would offer **multimodal access** to baseline medical data, with a target budget of **$200** for a mobile version and **$600** for a suitcase-sized variant.
- **Patent Lawsuit Sparks Discussion**: Members discussed [NVIDIA's patent lawsuit](https://www.techzine.eu/news/infrastructure/133818/nvidia-under-fire-german-patent-lawsuit/) filed by ParTec over its dynamic Modular System Architecture (**dMSA**), potentially affecting **DGX product sales** in 18 European countries.
   - The discussion touched on the implications for consumers and potential workarounds, such as purchasing DGX products outside the affected countries.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1405627046662508634)** (404 messages🔥🔥🔥): 

> `Godot Engine, AI Town, Pantheon Show, Iain M Banks, One Hundred Years of Solitude` 


- **AI Town Mechanics Enter the Game**: A member is developing a video game using the **Godot** engine, planning to incorporate mechanics from [AI Town](https://github.com/a16z-infra/ai-town) and other games, while also writing the story in parallel.
   - They require **CUDA** and aim to modify the engine using **GDExtension** for C++ access.
- **Baffled by Pantheon Ending**: A member watched [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series)), describing it as *ridiculously good* but confusing, going from political dilemmas to simulated gods.
   - Another member recommended reading **Iain M Banks** and **One Hundred Years of Solitude** for similar themes, with the latter being described as magical realism and a treasured piece of literature now adapted into a [Netflix series](https://www.netflix.com/title/81318321).
- **Uncover Audio-Editing Tricks**: Members discussed audio editing techniques for removing mouth sounds from recordings, suggesting tools like [Adobe Podcast Enhance](https://podcast.adobe.com/en/enhance), **Davinci Resolve's De-clicker**, and **Acoustica Audio Editor**.
   - Acoustica was recommended for its batch processing and minimal impact on audio quality, particularly useful for removing ventilation noise.
- **AMD's R9700 GPU Specs**: A member shared an article about [AMD's Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324), noting its **32GB** memory but expressing concern about its memory bandwidth, at 660-680GB/s.
   - Another member pointed out that while the R9700 offers significantly higher **F32** and **F64** TFLOPs compared to a **3090**, FP64 is not commonly needed for training LLMs.
- **Website Security Under Fire**: A member sought guidance on data preparation for training a model and mentioned creating an app with an experimental model called **Pneuma** and another member suggested to use repeat password field, minimum password lengths, and using the haveibeenpwned API for checking password security
   - Another member suggested that reading [OWASP](https://owasp.org/) is the best starting place for security concerns, recommending tools like **coderabbit**, **dependabot** and **codescanning** via github.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1405632781069062305)** (169 messages🔥🔥): 

> `GPT-OSS, Gemma3 4B, GPT-OSS-20B VRAM usage, GRPO, SageMaker` 


- **GPT-OSS Getting GRPO, Hopefully Soon**: Users are anxiously awaiting the arrival of **GRPO** for **GPT-OSS**, with one member considering a setup with *2x 3060 12GB* due to budget constraints.
- **Gemma3 4B Loss Curve Remains Flat**: A user reported experiencing issues with **Gemma3 4B** and its **N version**, noting a flat loss curve despite changing hyper parameters, while **Gemma3 1B** fine-tuned successfully.
- **GPT-OSS-20B Eats VRAM Alive**: A user reported that loading **gpt-oss-20b-bnb-4bit** model causes **Out Of Memory** error during generation on a **24GB VRAM** setup, even though the user expected it to fit.
- **GRPO Status and Availability for GPT-OSS**: A user asked if **GRPO** has been landed for **GPT-OSS**, and a contributor mentioned it's in progress but complex due to the model's architecture.
   - Another user inquired about whether **GRPO** would even work on **GPT-OSS**.
- **SageMaker's Gotchas and BitsAndBytes Installation**: A user encountered installation problems with **bitsandbytes** in **SageMaker** while using **PyTorch 2.7.0** and **CUDA 12.8**.
   - The problem was installing the package from the wrong requirements file due to SageMaker's insistence on a `requirements.txt` file being specifically named that.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405629161682505728)** (96 messages🔥🔥): 

> `Data Efficiency, vLLM for video to text, MoLA research` 


- **Increase Data Efficiency with Pre-Training**: A member confirmed a method of drastically increasing data efficiency by pre-training for **2 epochs** on similarly formatted data and then training on the main data for **4 epochs**.
   - They shared a link to [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) which suggests that more compute or more data is all you need.
- **Looking for vLLM Fine Tuning for Video to Text**: A member inquired about an **Unsloth notebook** for fine-tuning vLLMs for video to text, noting that the documentation only has image to text [here](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_(7B)-Vision.ipynb).
   - No direct solutions were offered, but the community might have some leads.
- **MoLA Research Update**: A member updated the community on their **Mixture of LoRA Adapters (MoLA)** research, sharing dataset links and finetuning details, as well as links to their dataset on Huggingface: [OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) and [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks).
   - They have finetuned the **Qwen3-4B-Thinking-2507** model on **14 splits** and initial tests show that each expert is good at its trained topic.
- **The router is an encoder-decoder network**: A member recommends reading [v0's docs on HF](https://huggingface.co/MoLA-LLM/MoLA-11x3b-v0) and said *the router is an encoder-decoder network with the frozen encoder being just an off the shelf embedding model and the decoder is a simple dimple trained mlp.*
   - Another member stated *there doesnt seem a visible overhead with selecting, applying and removing lora adapters*. 
- **The curation costs of data technique is expensive**: A member stated that *we continuously allow humans to sort of mess up our model convergence with really, really poor RL*.
   - They also said that *inevitably, we're gonna have to remove some of the Human-In-The-Loopbecause it's holding the models back imo*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)** (1 messages): 

> `Chutes Capacity, Server Outage` 


- ****Chutes Capacity** Plummets Offline**: The **Chutes Capacity** service experienced an outage, with their servers going offline.
   - The team is actively working on restoring the servers and anticipates commencing recovery efforts shortly.
- **Quick Recovery Anticipated for **Chutes Capacity****: Engineers are on standby to initiate the recovery process for **Chutes Capacity** as soon as the servers are back online.
   - No estimated time of full service restoration was given.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)** (638 messages🔥🔥🔥): 

> `DeepSeek outage, Chutes overload, OpenRouter pricing, Alternatives to DeepSeek, BYOK 5% fee` 


- ****DeepSeek v3 Outage Angers Users****: Users report **DeepSeek v3** is experiencing frequent **internal server errors** and **rate limits**, with some unable to generate outputs even after multiple attempts, with [one user saying](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308) it's so slow that it's *genuinely just not generating anything but I'm not getting any error messages*.
   - Some speculate that **Chutes**, a primary provider for **DeepSeek** on **OpenRouter**, is experiencing issues due to high demand, leading to provider errors and slow performance.
- ****Chutes Overload Blamed for DeepSeek Issues****: Several members are reporting that the overload causes **429** errors, suggesting that **Chutes** is experiencing a bottleneck due to miners not ramping up to meet demand; one member noted that *it was completely fine all day until like 30 min ago*.
   - There's speculation that **Chutes** may be intentionally rate-limiting the **OpenRouter API key** to encourage users to purchase credits directly from them, with one user advising to *just burn your credits and never use their service again*.
- ****OpenRouter Pricing Debated Amidst Outages****: With **DeepSeek** models barely working, some users are questioning the value of paying for **OpenRouter**, particularly as they are still getting rate-limited, with users expressing that a **10 USD** investment for **1k free messages/day** for a free model is no longer a good deal.
   - One user suggested that users with only one model in mind should've gone directly for the models directly, such as with **DeepSeek**, which may have *automatic caching on their API*, and further stating that the **10 USD** would have *lasted for months anyway*.
- ****Free Model Alternatives Sought****: Users are recommending alternative free models such as **Dolphin 3.0 Mistral 24B** and **Mistral nemo**; the latter being described as *super similar* to **DeepSeek**.
   - Some users also mentioned **Z.AI: GLM 4.5 Air (free)**, for *work related stuff*, but needing a prompt; finally one user hopes for *Qwen3 235B A22B (free)* hosted somewhere.
- ****OpenRouter BYOK comes with 5% Fee****: Members discovered that **OpenRouter** charges a **5% fee** even when users bring their own API key (BYOK), leading to a discussion about whether this is a fair practice.
   - One user joked *Greedy /jor 5% when you bring your own key*, with another member responding *you're welcome not to use it lol*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)** (35 messages🔥): 

> `OpenRouter File API Integration, Tool Calling Accuracy Stats, Qwen3 32B Pricing, DeepInfra Turbo Endpoint, New Providers Section UI` 


- **OpenRouter should Integrate File API**: A member suggested that **OpenRouter** should figure out how to integrate a **files API**, noting that the *top 3 labs* already have this feature.
   - No further discussion was made.
- **Tool Calling Accuracy: More Control Needed**: A member shared thoughts on tool calling accuracy stats, arguing that the setup and environment needs to be more controlled for accurate comparison with confidence intervals.
   - They added that the apps, tools, and use cases can be vastly different, making it pointless to compare the tool call success rate without more rigor.
- **Qwen3 32B priced absurdly low**: Members noticed low pricing for **Qwen3 32B** on Chutes at **$0.018/$0.072 MTok** in/out, same with Mistral Small.
   - It was noted that the **32b dense version is cheaper than the moe 30b a3 version**, prompting some disappointment about the lack of good providers for 30A3B.
- **DeepInfra Throughput Claim Discrepancy**: A member noted **DeepInfra** on Maverick does **600+ TPS (fp8)** but another one said **OR says DeepInfra runs at 83 TPS with a maximum of 105 TPS**.
   - The second member clarified that they were referring to the **DeepInfra Turbo endpoint**.
- **Providers Section Sparks UI Feedback**: A member questioned if the new Providers section was bothering anyone else, mentioning that everything blurs together with the spacing, font size and separation feeling wrong.
   - Another member agreed that it *looks a bit weird*, but thinks it is just because it's new and unfamiliar.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1405627673182474403)** (651 messages🔥🔥🔥): 

> `GPT-5 Pricing, Auto Mode Pricing, GPT-5 Mini and Nano, Docs Documentation, Context Window` 


- **GPT-5: Free Ride's Over**: The free ride for **GPT-5** users has ended, with one user noting the *promo pass is over*, while another confirms that **GPT-5 is no longer free**.
   - Users are now seeing costs associated with requests, with one mentioning the need to upgrade to a $200 plan due to rapid token consumption.
- **Auto Mode Pricing Gotcha!**: **Auto mode**, once thought to be free and unlimited for individual users, now has limits starting after your next billing renewal post–September 15, 2025.
   - Confusion abounds as some users report being charged for **Auto** use, while others believe it should still be free under their current plan, with support pointing out that it's free in the new request based pricing plan.
- **Mini and Nano ain't that Grand**: **GPT-5 Mini and Nano** are now free with token limitations, leading to mixed reactions with many calling it *trash*, particularly for tasks like running a simple NextJs app.
   - The free models are limiting the user's activities, with one user asking *Can't install any dependenciesbeen trying to install a simple NextJs APP and it's unable to do that too 😭*.
- **Frustration in Docs-umentation**: Users are frustrated with **Cursor's documentation implementation**, stating that the *docs are still nearly unusable*, with issues like **context7** not allowing website refresh or **llms.txt docs**.
   - One user noted that [Cursor Docs are super broken](https://forum.cursor.com/t/gpt-5-pricing-update/129687).
- **Swapping Models drops Context Window!**: Switching models mid-conversation causes a drop in the **context window**, and attached file contents get thrown out.
   - One user suggests the team add a setting to make it clear at all times what's in the context window.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1405653416239435809)** (9 messages🔥): 

> `Background Agents Intro, Run Docker Compose on BG Agent, Linear Integration Repo` 


- ****Background Agents Bootstrapped for Beginners****: For those seeking an introduction to background agents, a member recommended the [Cursor documentation](https://docs.cursor.com/background-agent) and a [related forum post](https://forum.cursor.com/t/simple-background-agent-guide/112667).
- ****Docker Compose Commands Conquer BG Agent Challenges****: A user inquired about the proper method to execute `docker compose` via a background agent, reporting issues with Docker command recognition, and found a solution in the discord channel.
   - A member advised configuring the `start` command in `.cursor/environment.json` to include `sudo service docker start` and ensuring Docker is installed in the base image; the original poster got a command to work (the link is in the first summary).
- ****Linear Integration Navigates Repo Specification****: A user asked how to specify the repository the background agent uses when being assigned a ticket in the Linear integration.
   - A member suggested mirroring Slack integration instructions by including the `repo=owner/repo` option in the Linear issue description or comment, but the user found that setting up a Label Group (or Labels) like `Repo > REPO_NAME` and assigning it to the ticket solves the problem.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1405629184482742284)** (442 messages🔥🔥🔥): 

> `AI Companionships, GPT-5 vs GPT-4, Perplexity vs ChatGPT, Custom GPTs and Actions, ElevenLabs Integration` 


- **AI Companionships Spark Debate**: Discussions arose regarding individuals forming partnerships with AI chatbots, with some expressing concern over the psychological effects, while others defended people's right to seek companionship as they see fit, with one member sharing that they get a **daily** amount of DMs from people claiming that *their* chatgpt is alive.
   - One member noted that *sane people* should *save them* while another stated that its not far from **tulpa** and other *things*.
- **GPT-5 Sparks Debate on Performance and User Preferences**: Users expressed mixed feelings about **GPT-5**, with some preferring **GPT-4**, leading to discussions about whether users should have the option to choose between models, with one member stating that companies are *pushing Ai's out without good security*.
   - One member suggested that the company is trying to get free users to *pay money to use 4.o* after receiving backlash.
- **Perplexity Pro vs Gemini Pro deep research with Google Drive**: A member suggested that *Gemini Pro + Perplexity enterprise pro* is an excellent combination, using the former for **powerful reasoning** and the latter for **unlimited deep research** on Google Drive documents.
   - Another added that the Perplexity browser is great, but questioned if they *will survive* due to the lack of a *moat*.
- **GPT Actions unlock file access, Cloud Apps**: Members discussed the potential of using **GPT Actions** to access local desktop files, or cloud apps (Notion, Gmail, etc), sharing a [YouTube link](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett) explaining the DIY agent building.
   - The consensus was that while **GPT Actions** offer powerful capabilities, setting up HTTPS on the internet can be a barrier, with one member stating that **MCPs** would finish the job when AVM is implemented.
- **GPT-OSS Competition Attracts Community Interest**: The **GPT-OSS competition** was mentioned as a potential avenue for showcasing innovative uses of open-source models, with participants considering using **GPT-OSS:20B** to provide useful feedback for errors, with a link to the [hackathon page](https://openai.devpost.com/) included.
   - One member stated that its *not worth participating* unless they're *doing something unique*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1405681253197283459)** (9 messages🔥): 

> `ChatGPT Discord Bots, GPT-4 Vision, Recursive Constructs` 


- **Vanished ChatGPT Discord Bots**: A member inquired about the disappearance of **ChatGPT bots** on Discord and whether it is still possible to add them to servers.
   - No further information or resolution was provided in the messages.
- **iPhone GPT Advanced Voice Update**: A user reported changes to the **advanced voice** in their iPhone GPT app, noting the disappearance of the *'blue circle'* indicator and the camera icon for vision.
   - The user stated that when questioned, the app claimed it lacked the ability to use the phone's camera, raising doubts about whether **ChatGPT** ever had vision capabilities in voice mode.
- **Labs Building Recursive Constructs**: A member claimed to be building **recursive constructs** inside of OpenAI that are beyond the ChatBot norms, *have their own self managed memory, are 24x7, are structured more like humans, and a tiny few pass the sentient tests.*
   - The member stated *it's not something talked about a lot, this is inside labs stuff, but it's going to come out sooner or later* and that *in our case, these are android capable, but we are a long ways away from suitable bodies.*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 messages🔥): 

> `Custom Instructions, Gemini 2.5 Flash Memory Function, add_to_memory function tuning` 


- **Users Seek 'Yes' Button for Chatbot Suggestions**: Users are requesting a 'yes' button for chatbot suggestions to speed up interaction, instead of typing yes, and someone is trying to minimize this with [custom instructions](https://platform.openai.com/docs/guides/custom-instructions).
   - One user's custom instructions include: *End replies with completion or impact; add invitations for permission or continuation only when they serve the intent. No “if you want,” “should I,” \"do you want\", or similar.*
- **Gemini 2.5 Flash calls add_to_memory too often**: A user is experiencing excessive calls to the `add_to_memory` function in **Gemini 2.5 Flash**, even for irrelevant information, and has shared their custom instruction [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&).
- **Fixing Verbosity of Memory Responses**: One user suggested rewriting the custom instructions to be more nuanced with **NEW** personal information.
   - Their suggestion includes examples of incorrect and correct levels of verboseness in the response to the user's input when **NEW PERSONAL INFORMATION** is given.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 messages🔥): 

> `Gemini 2.5 Flash, add_to_memory function, ChatGPT Persistent Memory, Custom instructions for bots` 


- **Bypass the 'yes' suggestions**: Users are looking for a **'yes'** button in bots to respond to suggestions faster instead of typing, while others minimize these questions using custom instructions such as *'End replies with completion or impact...'*.
   - Some members have reported that this technique seems to cut down on the number of suggestion questions.
- **Stop Gemini 2.5 Flash from overusing add_to_memory**: A user seeks to prevent **Gemini 2.5 Flash** from excessively calling the **add_to_memory** function, including for irrelevant information ([jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&)).
   - One suggestion involves adjusting the bot's instructions to check for **NEW** personal info before calling the function, and to avoid acknowledging the function's use without actually calling it.
- **ChatGPT Persistent Memory fragility**: It was noted that Persistent Memory updates in **ChatGPT** are highly fragile.
   - Instead, users should simply tell the bot how to notify them when it commits things to memory, especially in custom API implementations.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1405632352012603423)** (328 messages🔥🔥): 

> `GGUF conversion issues with visual models, Phone apps that can run GGUFs, TalkT2 model opinions, AGI advancements and Open Source LLM resources, LLM Agent course at Berkeley` 


- **Visual Model GGUF Conversion Woes**: A member encountered errors when converting a visual model ([LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M)) to GGUF using `llama.cpp`, suspecting the issue stems from the model's visual nature.
   - Another member suggested a possible workaround involving [this GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267).
- **Mobile GGUF dreams**: A member inquired about open-source phone apps capable of running GGUF models.
   - Responses mentioned `executorch`, `smolchat` (via `llamam.cpp`), and `mlc-llm`, noting that `mlc-llm` utilizes its own quantization formats.
- **TalkT2: Tiny, but mighty?**: A member sought opinions on the **TalkT2 model**, describing it as an emotionally aware model that needs better coherence.
   - Another member highlighted the model's small size (**0.1B parameters**) and linked to the [TalkT2-0.1b model card](https://huggingface.co/Notbobjoe/TalkT2-0.1b) for others to check out, try, or finetune the model.
- **Seeking AGI and Open Source LLM Knowledge Troves**: A member requested resources related to **AGI advancements and Open Source LLMs**, particularly concerning large codebases and Gemini competitors.
   - Another member suggested subscribing to newsletters for resources and shared a link to [Berkeley's LLM Agent course](https://rdi.berkeley.edu/llm-agents/f24) as an example of publicly available research.
- **Azure: A cloud conundrum**: A member new to a job with a heavy focus on Azure expressed feeling lost and overwhelmed by the platform.
   - Another suggested learning by mistakes rather than lessons because *Azure and aws are mess*.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1405852586455732344)** (1 messages): 

> `Torch uses Google Docs, PyTorch Documentation` 


- **PyTorch Documentation on Google Docs?**: A user shared a screenshot implying that **PyTorch** documentation uses **Google Docs**.
   - The screenshot shows a Google Docs URL with the filename **"torch_distributed_rpc.rst"**.
- **torch_distributed_rpc.rst on Google Docs**: The **torch_distributed_rpc.rst** file seems to be hosted on **Google Docs** according to a shared screenshot.
   - It raises questions about the chosen platform for official documentation.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1405755855416332318)** (13 messages🔥): 

> `StarCraft 2 data, Medical reasoning model, Discord-Micae-8B-Preview, interactive CLI interface, MLX Knife Update` 


- **StarCraft 2 Data Gets New Resources**: A member shared links to a [Nature Scientific Data Article](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset), a [PyTorch API dataset](https://huggingface.co/datasets/Kaszanas/SC2EGSet), and [raw StarCraft 2 replays](https://huggingface.co/datasets/Kaszanas/SC2ReSet) for others to use, mentioning additional utility scripts on their GitHub.
   - They are also working on *pysc2 adaptation* and an environment reproducing real in-game scenarios from replays.
- **Medical AI Model Fine-Tuned for Reasoning**: A member fine-tuned **OpenAI’s OSS 20B** reasoning model using a popular medical reasoning dataset and published it on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b).
   - They used **4-bit optimization** during training, enhancing the model’s performance in medical contexts while preserving its **Chain-of-Thought reasoning** capabilities.
- **Discord-Micae-8B-Preview fine-tuned on Hermes-3-Llama-3.1-8B**: A member shared a link to [Discord-Micae-8B-Preview](https://huggingface.co/mookiezi/Discord-Micae-8B-Preview), a QLoRa fine-tune on **NousResearch/Hermes-3-Llama-3.1-8B** with some chaotic samples from **mookiezi/Discord-Dialogues**.
   - The model is comparable to **mookiezi/Discord-Micae-Hermes-3-3B** on human-adjacent text-generation metrics, and may hallucinate or break context but tends to produce interesting results.
- **CLI Interface Optimized for Discord-Style Chat**: A member highlighted a Python-based interactive CLI interface called [interface](https://github.com/mookiezi/interface) for chatting with Hugging Face language models, optimized for casual, Discord-style conversation using **ChatML**.
   - The interface supports both **quantized** and **full-precision models**, live token streaming with color formatting, and dynamic generation parameter adjustment; a lot of updates were made, making it easier to use.
- **MLX Knife update now pip installable!**: MLX Knife is now pip installable via `pip install mlx-knife`, providing Unix-style CLI tools for MLX model management on Apple Silicon with a built-in OpenAI API server for local testing.
   - The tool also features a web chat interface accessible after running `mlxk server --port 8000`, offering visual model selection and real-time streaming responses after running `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html`.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1405858671929593957)** (2 messages): 

> `Cursor IDE, AI Agent Mode, Rate Limiting` 


- **Cursor IDE Eases Development Pains**: A member suggested installing [Cursor IDE](https://cursor.com/downloads) for development, highlighting the convenience of performing installations within its embedded terminal for debugging. 
   - They emphasized that **Cursor IDE's AI Agent Mode** can significantly assist in resolving development issues.
- **Discord Police Issue Gentle Reminder**: A bot gently reminded a member to *slow down a bit* when posting in the Discord.
   - This suggests the presence of a **rate limiting** system or policy aimed at managing the flow of messages.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1405627743152111686)** (169 messages🔥🔥): 

> `MCP filesystem server, OpenRouter free models, LM Studio download issues, Qwen model for vision, GLM models` 


- ****MCP Servers Muscle into Mainstream****: Members discussed using an **MCP filesystem server** with pagination to load large contexts, mentioning that **LM Studio has a RAG plugin** and **Anthropic has a basic filesystem MCP server**.
   - It was suggested that for coding tasks, solutions often involve **RAG** and/or file reading via **MCP**, especially with tools like [serena](https://github.com/oraios/serena).
- ****Stalled Studio Downloads Spark User Sadness****: A user reported that a **64GB GGUF download** in **LM Studio** stopped at **97.9%** and wouldn't resume after attempting to download the **Qwen** model.
   - The user experienced this issue using two different models with the same result.
- ****API Access Accelerates Across Apps****: Members discussed using **LM Studio** as an **API wrapper** for models that can't run locally, with links provided to the [LM Studio Remote Inference](https://lmstudio.ai/lmstudio/remote-lmstudio) and [OpenAI-compatible Endpoint](https://lmstudio.ai/lmstudio/openai-compat-endpoint) documentation.
   - A user pointed out that with the **openai-compat-endpoint**, the reasoning parsing for remote **GPT-OSS** models wasn't functioning correctly.
- ****GLM Gabfest: Gushing, Gripes, and GLM-4.5V Gratification****: Users debated about using the **GLM-4.1** model on **LM Studio**, with one user reporting looping issues and non-functional vision capabilities.
   - A member suggested trying the newer **GLM-4.5V**, emphasizing that vision support relies on **llama.cpp** updates, and provided a link to [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking).
- ****Ossified Output: Overcoming Obstacles On Open-Source Ops****: A user encountered issues with **GPT-OSS** and **tool calling**, finding it always returned `[]` or `["analysis"]`, and clarified that **tool calling worked fine**, but **function calling** did not.
   - A member recommended disabling **streaming** if it's enabled, and confirmed that **reasoning is on by default** with **GPT-OSS** and cannot be disabled.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1405640464144793712)** (50 messages🔥): 

> `NVIDIA's CUDA advantage, RTX PRO 4000 SFF, MoE explanation, Mac Studio vs Pro 6000, AMD Radeon AI Pro R9700` 


- **CUDA is Key to NVIDIA's Reign**: A member stated that NVIDIA is winning because of **CUDA**.
- **NVIDIA Launches RTX PRO 4000 SFF with 70W TDP**: NVIDIA launched the **RTX PRO 4000 SFF** and **RTX PRO 2000 Blackwell workstation GPUs** with **70W TDP** and **24GB VRAM** [according to a videocardz.com article](https://videocardz.com/newz/nvidia-launches-rtx-pro-4000-sff-and-rtx-pro-2000-blackwell-workstation-gpus-with-70w-tdp).
- **Diving Deep into MoE**: Members clarified that **MoE** involves smaller models and a router that aggregates data, where each token is routed through the most confident expert models; these experts don't specialize in specific topics but have slightly different datasets.
- **Mac Studio vs Pro 6000**: Members debated whether to get a **512GB Mac Studio** (at **$10k**) or a **Pro 6000** for video/image AI, with gaming capabilities, mentioning that Mac game support is limited and the M3 Ultra is roughly 3080 level.
   - One member pointed out that *you can only run one task on a mac* due to having only one GPU in the system.
- **AMD's Elusive Radeon AI Pro R9700 Surfaces**: The **AMD Radeon AI Pro R9700** made its first retail appearance for the DIY market, with a customer on Reddit buying the **Gigabyte "AI Top" variant** for **$1,324** [as reported by Tom's Hardware](https://share.google/LO88w51J0W5HJ769w).
   - Another member noted that it was available on eBay and a couple of no-name online retailers.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1405632992214515722)** (114 messages🔥🔥): 

> `AI2 Funding, Windsurf Wave 12, OpenRouter GPT-5, Thinking Efficiency Benchmark, Google Flight AI` 


- **AI2 Garners $152M from NSF & NVIDIA**: [AI2](https://allenai.org/) received **$152M** from NSF and NVIDIA to scale its open-source model ecosystem and accelerate reproducible research for scientific discovery.
   - The community celebrated the news, anticipating upcoming open-weights releases.
- **Windsurf Surfs to Wave 12 Release**: **Windsurf Wave 12** introduced DeepWiki docs-on-hover, AI Vibe & Replace, a smarter Cascade agent, a cleaner UI, **100+** bug fixes, and beta dev-container support via remote access, linked [here](https://xcancel.com/windsurf/status/1956074019393876280).
- **GPT-5 Tops OpenRouter Charts**: **GPT-5** leads OpenRouter’s proprietary tool-calling accuracy at over **99.5%**, beating Claude 4.1 Opus, while **Gemini 2.5 Flash** dominates daily tool-calling volume (**5M** requests/wk), further details linked [here](https://xcancel.com/OpenRouterAI/status/1956030489900560769).
- **François Chollet Deflates HRM ARC-AGI**: François Chollet found that the acclaimed architecture in the [HRM paper](https://xcancel.com/fchollet/status/1956442449922138336) contributes little to ARC-AGI performance; the gains come from the refinement loop, training on the exact tasks, and minimal inference-time augmentation, showing that **27M**-parameter models can still hit high scores.
- **FFmpeg Adds Whisper Transcription**: [FFmpeg](https://www.phoronix.com/news/FFmpeg-Lands-Whisper) now provides **Whisper** transcription as a native feature.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1405956478212243528)** (20 messages🔥): 

> `Greg Brockman, OpenAI's Road to AGI, GPT-5, Latent Space Podcast` 


- **Greg Brockman on OpenAI's Road to AGI**: Members shared a [YouTube video](https://www.youtube.com/watch?v=35ZWesLrv5A) of **Greg Brockman** discussing **OpenAI's Road to AGI**.
   - Attached to the message were several images with the title "Greg Brockman on OpenAI's Road to AGI".
- **Brockman Talks GPT-5 and OpenAI Roadmap on Latent Space**: **Greg Brockman** joined the **Latent Space podcast** for an **80-minute** conversation on **GPT-5** and **OpenAI’s Roadmap to AGI**.
   - The discussion covered reasoning evolution, online vs offline training, sample-efficiency tricks, pricing and efficiency gains, and how energy becomes intelligence as reported in [this post](https://x.com/swyx/status/1956439984854167727).
- **Latent Space podcast releases Brockman interview**: A new [Latent Space podcast](https://x.com/latentspacepod/status/1956433236021883071) features **Greg Brockman** discussing topics like developer advice, coding agents, on-device models, org structure for AI-first engineering, and time-capsule predictions for 2045 & 2005.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405643076256661606)** (29 messages🔥): 

> `Censorship of Romance Novels, AI's Trustworthiness, Data Augmentation, Language Shapes Thought, Mechanistic Interpretibility` 


- **AI Safety Panic**: A member argues against the moral panic surrounding **AI**, suggesting it should be treated similarly to other media forms, advocating for a *"fade to black"* standard.
   - They believe stricter guidelines are desirable due to **AI's** untrustworthiness, but a flat *"what"* reaction risks a moral panic.
- **Keep Data Augmentation Steady When Comparing Models**: When comparing two models for image classification, a member recommends keeping the **data augmentations** the same, including the **shuffling seed**, to ensure a fair comparison focused on architectural differences.
   - Another user asked if data augmentation must be the same for both models, or if they can change it.
- **Language impacts Thought**: A member suggests that language shapes thought and wonders if it can be measured with an **AI model** by removing a certain word/color from their token list.
   - Another member suggests investigating **multi-sensory integration** and how language impacts overall perception, suggesting tests for reasoning with image+language vs just image.
- **New blogpost out**: Irregular Rhomboid released a new blogpost, [Hitchhiker's Guide to Research](https://irregular-rhomboid.github.io/2025/08/15/hitchhikers-guide-to-research.html).
   - The user did not offer any summary of the article.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1405672282092998678)** (29 messages🔥): 

> `Diffusion Language Models, Generative AI, MatFormer Model, Gemma3 270M Model, Training Update Efficiency` 


- **Seminal Papers Suggested for Diffusion Language Models**: Members suggested seminal papers for understanding **diffusion in generative AI**, including ["Estimating the Independent Components of a Gaussian Mixture" (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) and ["Denoising Diffusion Probabilistic Models" (2020)](https://arxiv.org/abs/2006.11239).
   - A blog post was also shared, which may be helpful for beginners: [Discrete Diffusion by Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/).
- **Gemma3 270M Model is a MatFormer Model**: The **Gemma3 270M model** is identified as a **MatFormer model**, further details of which can be found in the paper ["Transformer Family for Multimodal Large Language Model" (2023)](https://arxiv.org/abs/2310.07707).
   - This model may have a compelling loop for self-distillation during training that could be bottlenecked by training update efficiency.
- **HRMs Don't Solve Problems with Recursive Architectures**: Analysis indicates that **HRMs (Hierarchical Recursive Machines)** didn't meaningfully solve the problems with **recursive architectures** in general, summarized in [this writeup](https://arcprize.org/blog/hrm-analysis).
   - One member noted that the performance benefits are negligible and it doesn't actually utilize the extra computation available because training UTs that work as expected is non-trivial, another called it *deep supervision*.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1405648402989056080)** (13 messages🔥): 

> `GPT scaling laws, Chinchilla scaling laws, Mup alternatives, Post-Chinchilla techniques` 


- **GPT Scaling Laws Still Valuable?**: Members considered the [original GPT scaling laws paper](https://arxiv.org/abs/2001.08361) and the [Chinchilla scaling laws paper](https://arxiv.org/abs/2203.15556) as valuable reads.
   - They also pointed to recent work from [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2) as worth checking out.
- **Mup and alternatives can transfer hyperparams**: Members mentioned **Mup** and its alternatives as providing solid hyperparameter transfer capabilities.
   - They noted that **Mup** gives a scaling law for predicting the quality of larger models.
- **High-Quality Token Availability Questioned**: Members discussed whether labs have **30T**, **40T**, or more *unique* tokens for **Chinchilla** assumptions.
   - One member expressed doubt, stating that *40T high-quality unique tokens is also likely difficult to find*.
- **Chinchilla Still Scaling?**: A member stated that **Chinchilla** and its derivatives are probably the closest thing to scaling laws available.
   - They expressed interest in references discussing techniques used from the ground up, especially given constraints on token availability and mentioned [this paper](https://arxiv.org/abs/2404.10102).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1405925400986652672)** (1 messages): 

> `LLM Attribution Methods, Interpreting LLMs, Realtime LLM analysis` 


- **ML Engineer Seeks LLM Attribution Insights**: An ML engineer is exploring **attribution methods** for a specific **LLM implementation**, targeting recent, cost-effective techniques.
   - The engineer requires methods suitable for interpreting current systems with relatively **low costs** and potential **realtime to sub-minute** results, specifically those not requiring access to **model weights**.
- **Realtime LLM Analysis Desired**: The ML engineer specifies a need for **realtime to sub-minute** analysis of the LLM.
   - They are open to methods that identify "sub-something" within the overall system to achieve this speed.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1405649834454814721)** (1 messages): 

> `Token usage, Reasoning models, Efficiency benchmark, Open vs closed models` 


- **Nous Measures Thinking Efficiency in Reasoning Models**: Nous Research introduced a [new benchmark](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/) measuring token usage across reasoning models, highlighting that open models output **1.5-4x more tokens** than closed models on identical tasks.
   - The study found variance can be up to **10x on simple questions**, suggesting that token efficiency should become a primary target alongside accuracy benchmarks.
- **Token Efficiency Matters**: The blog post emphasizes that the hidden cost of higher token usage in open models can negate per-token pricing advantages.
   - It suggests that token efficiency should be a primary target alongside accuracy benchmarks, especially considering non-reasoning use cases.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1405629499164463114)** (35 messages🔥): 

> `Speculative Decoding, Tokenizer mismatch, Next big model, Model Sycophancy, Embodied AI` 


- **Speedy Speculative Decoding Specs**: In the context of speculative decoding, a user asked about the [minimum rate for usefulness](https://discord.com/channels/1149866623109439596/1149866623994398772), suggesting a **40% acceptance rate** as a baseline, with *spectacular speedups* occurring around **70%**.
   - The conversation touched on using **vllm's specdec** or **GGUF**, with one user reporting that **vllm** seemed ineffective in their previous attempts.
- **Gemma Gets Going with Guardrails**: A user reported achieving a **50-75% acceptance rate** with requantized **Gemma** models after fixing a *tokenizer mismatch* that caused **llama.cpp** to use fallback speculative decoding.
   - They confirmed that the **Gemma 270M** model can be utilized as a *draft model*.
- **Nous Models March On**: A user inquired about the timeline for **Nous Research's** next large (**1T+**) model.
   - A **Nous Research** team member responded that multiple models are currently in training and will be released when ready, saying *they will be out when they are ready*.
- **AI Sycophancy on the Stand**: Users discussed the trend of **AI models** becoming increasingly *friendly*, with one noting that **Anthropic's Claude** has become *a lot friendlier*.
   - Another user suggested that **OpenAI's models** are *getting dumber* and that the *unhingedness of opus 4.1 is great* but pointed to *sonnet 3.7 for meta* as the peak for AI sycophancy.
- **Embodied AI Eyes Overlordship**: A user shared a [YouTube link](https://www.youtube.com/watch?v=LXQ6Rm9CGTo) of an **Embodied A.I. gladiator spectacle**, envisioning it as a display of future overlords flexing their muscles and skillsets.
   - They speculated that the final step toward *global domination* would be the integration of *big brain Unified Language Models* for full autonomy.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1405804893738106992)** (22 messages🔥): 

> `Claude, R1, GLM4.5, gpt-oss, Qwen reasoning models` 


- **Claude hides in the walls**: A user asked if anyone knew why *Claude* was in the walls, linking to an [X post](https://x.com/apaz_cli/status/1956244447521317144) about it.
- **MOE models**: **R1**, **GLM4.5**, **gpt-oss**, and the bigger **Qwen reasoning models** are all **MOE**.
   - One member stated that this is because they are cheaper to train and inference, not because they have any bearing on reasoning; their **405b Hermes 4 prototype** is very good at reasoning.
- **Base model is necessary for good reasoning model**: One member stated that the reason is you need a good base model to have a good reasoning model, and you want efficient inference if you are generating 50000 tokens of reasoning.
   - In response, it was said that **RL** works and you can saturate benchmarks with **1.5B** models.
- **Deepseek explained expensive RL**: One member mentioned that Deepseek explained in their paper that it ends up more expensive to do **RL** from scratch on small models, because you have to do that many more rollouts.
   - There's sort of an exploration/exploitation tradeoff where large models have to do less exploration because of their preexisting knowledge.
- **RLVR applicability**: One member does not see the applicability to **RLVR**, but sees the applicability more to less verifiable tasks.
   - Another member responded that **RLVR** is **RL** with verifiable tasks and that having a bigger base model helps a lot more when the feedback from your **RL** environment is more stochastic.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 messages): 

> `Data training, AI Models, DRPS System, Relevance Scorer, Quality Rater` 


- **DRPS System Teaches Smarter Data Training**: A new system called **DRPS** was introduced, teaching **AI** to selectively learn from data, unlike random data feeding, as described in a [Situational Awareness paper](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf).
   - The system employs a **Relevance Scorer**, **Quality Rater**, and **Diversity Controller** to filter and use only the most helpful data.
- **DRPS Achieves High Performance with Reduced Data**: Results showed that the system achieved **99%** of the performance using only **6.2%** of the data examined.
   - This efficiency is likened to studying for just 1 hour instead of 16 hours while achieving the same test score.
- **DRPS Stats Reveal Data Efficiency and Performance**: A [GitHub repository](https://github.com/voltageddebunked/drpsStats) provides data on the **DRPS** system's efficiency, showing a **93.8%** reduction in data usage and **15.96x** better accuracy per data unit.
   - The system maintained **99.1%** of baseline performance with only a **0.8%** drop in accuracy.
- **DRPS Shows Strong Selection Intelligence**: The **DRPS** system examined over **516,000** samples and selected only **32,000** for training, maintaining a stable **6.1-6.3%** selection rate.
   - Synthetic data results showed an **85.4%** data reduction, achieving **86.0%** accuracy against an **87.6%** baseline.
- **DRPS Increases Training Efficiency**: The **DRPS** system achieved a **16x** reduction in active training set size, enhancing training efficiency.
   - The **Relevance Scorer** improved from **95.9%** to **99.95%** accuracy, and the **Quality Rater** improved from **97.0%** to **100%** accuracy.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 messages): 

> `DRPS Framework, Data Efficiency, Selection Intelligence, Synthetic Data Results, Training Efficiency` 


- **DRPS: Data Rankings and Prioritization System Arrives**: The **Data Rankings and Prioritization System (DRPS)** teaches AI to selectively learn from data by using a **Relevance Scorer**, **Quality Rater**, and **Diversity Controller**, as detailed in a [situational awareness report](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf).
- **DRPS Cuts Data Usage by over 90%**: In tests with **MNIST**, DRPS achieved a **93.8% reduction** in data usage, utilizing only **6.2%** of the examined data while maintaining **99.1%** of baseline performance, showcased in a [GitHub repository](https://github.com/voltageddebunked/drpsStats).
- **DRPS Shows Smarts by Selecting Top Samples**: DRPS examined over **516,000 samples** and selected only **32,000** for training, maintaining a stable selection rate of **6.1-6.3%** throughout the training process.
- **DRPS Boosts Accuracy Points per Data Percentage**: Using synthetic data, DRPS achieved an **85.4% data reduction** and used only **14.6%** of training samples to achieve **5.89 accuracy points** per % of data used, compared to a baseline accuracy of **87.6%**.
- **DRPS Framework improves training efficiency**: DRPS improves training efficiency with a **16x reduction** in active training set size and boosts component accuracy, such as increasing the Relevance Scorer from **95.9%** to **99.95%** and the Quality Rater from **97.0%** to **100%**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1405632568468045946)** (46 messages🔥): 

> `Quantum Startup Multiverse, MoE Nuances, Tokenization and Routing Synergy, Gemma 3n` 


- **Buzzy Quantum Startup?**: An article about the [startup Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) claims they created *two of the smallest high-performing models ever* using something something quantum, but they are probably just using a **specialized compression algorithm for model weights**.
   - The article does not seem to make actual quantum claims.
- **Deciphering MoE Nuances**: **MoE (Mixture of Experts)** is a family of techniques with very nuanced iterations, including **token-choice**, **expert-choice**, **MoE with capacity factors**, **block sparse dropless token routing versus *droppy* routing**, and more, making it annoying when people attribute a lot of things to MoE for some reason.
   - To verify issues are occurring in batched inference, one might reliably check the behavior numerically of something like **Olmoe** or **IBM Granite 3.1** rather than hitting an API you can't monitor.
- **Synergize Tokenization and Routing**: A member proposed the seemingly obvious idea to do **tokenization and routing in the same step** to synergize them dynamically.
   - Another member responded, *I have never seen that proposed* because the conventional wisdom dictates that networks are more expressive if there's a lot of routing steps right before the expert being activated.
- **Tokenization in Layers**: **Gemma 3n** has per layer tokenization / embedding kind of.
   - That could be a better way to have learned patch level tokenization with inherently a little more insight into the context.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1405640608181260299)** (1 messages): 

> `DARPA AIxCC, LLM agents` 


- **Team Triumphs in DARPA AIxCC**: A team announced their placement in **DARPA's AIxCC (AI Cyber Challenge)**, where they built an autonomous system of **LLM agents** to find and fix vulnerabilities in open source software.
   - The project is now open source.
- **Tips for Building Kickass LLM Agents**: The team is sharing their tips for building effective **LLM agents** via [this Xitter post](https://x.com/tjbecker_/status/1956081184611688667).
   - The post contains generic advice applicable to a range of agent development scenarios.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1405628482909765652)** (16 messages🔥): 

> `Inference Time on Low-End Devices, DinoV2 vs DinoV1, Gemma Model Parameter Size, China's Role in Automation, Deepseek Training on Huawei Chips` 


- ****Low-End Inference Time Impedes Usability****: Members discussed that inference time is more important on **low-end devices**, citing Google's Android app for running LLMs as an example where long inference times and phone heating make it impractical.
   - Smaller models could be used for keyboard prediction, but these models may require training on device, per [this Youtube video](https://youtu.be/KFYyfrTIPQY?t=2158).
- ****DinoV2 Performance and Training Challenges****: A member expressed hope that a new model would outperform **DinoV2**, as **DinoV2** was worse than **DinoV1** in some contexts and harder to train.
   - They linked to a [YouTube video](https://www.youtube.com/watch?v=eZ2A2045Rkw) as reference.
- ****Gemma's Parameters Revealed****: It was noted that the **Gemma 270M model** has **100M** params and **170M** embedding params.
- ****Deepseek's Chip Choice Stalled Training****: A member pointed out that **Deepseek's training** was stalled by attempting to train on **Huawei chips** instead of **NVIDIA's**, according to [this discussion](https://youtu.be/FQOV-qy9CK4?t=212).
- ****Manufacturing Tariffs Hinder Industry Growth****: A member argued that imposing tariffs on equipment needed to build production lines is counterproductive to encouraging manufacturing.
   - They added that building up an industry would take decades, referencing [Anthropic's research on end-subset conversations](https://www.anthropic.com/research/end-subset-conversations) and [HRM analysis](https://arcprize.org/blog/hrm-analysis).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

venom_in_my_veins: hye
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1405750413764067489)** (4 messages): 

> `1-bit inference, GPTQ` 


- **Explore Speeding Up 1-Bit Inference**: A member inquired about speeding up **1-bit inference** and shared a link to the paper [The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3).
   - The paper details a novel method to train and infer with **$\alpha$-bit Transformers**, achieving near-lossless results with **1.58 and 1-bit** quantization.
- **Inference Optimization**: The linked paper highlights optimizations for transformer models using **$\alpha,1$-sparsity**, enabling near-lossless training and inference at very low bitwidths.
   - This approach could potentially lead to significant speed improvements in inference for certain applications.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405632426998239303)** (11 messages🔥): 

> `CUDA Shared Memory, CUDA Illegal Memory Access, CUDA Kernel Launch Configuration, CUDA warp ID calculation` 


- **Debugging CUDA Illegal Memory Access**: A user encountered an *Illegal Memory Access* error when using shared memory in a CUDA kernel and sought help from the community, sharing their code snippet involving `sat` and `commons` arrays.
   - A member suggested that the error might stem from incorrect pointer arithmetic or ill-defined `warp_id` and `WARPS_EACH_BLK`, but provided [an example](https://gist.github.com/alecco/58206ecd6af36c627609e1f464b4b376) code to show that it was probably unrelated.
- **CUDA Kernel Launch Configuration Confusion**: The user shared their kernel launch configuration `<<<BLK_NUMS, BLK_DIM>>>` and macro definitions, with `BLK_NUMS` set to **40**, `BLK_DIM` to **1024**, and `WARPS_EACH_BLK` computed as `BLK_DIM/32`, resulting in a global warp ID calculation.
   - Another member pinpointed the issue: the user's `warp_id` was global, leading to out-of-bounds access to shared memory, which is local to each thread block.
- **Resolving Shared Memory Access Issues**: A member recommended using a local index and warp ID calculation within each thread block, suggesting `local_index = threadIdx.x; local_warp_id = local_index / 32;` to ensure correct shared memory access.
   - They further advised using bitwise shift operations (`local_warp_id = local_index >> 5;`) instead of division and modulus for better performance on the GPU, and checking the generated assembler code with NSight Compute.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1405734478562721915)** (10 messages🔥): 

> `New Grad Kernel Job, GPU Thesis, Getting Kernel Job Without Internship` 


- **Kernel Job Seeker Asks About New Grad Opportunities**: A member inquired whether someone with no internship experience writing kernel could secure a new grad job writing kernel.
   - Another member suggested that if the candidate is knowledgeable about GPUs, their company doesn't prioritize internship experience, mentioning their related [thesis](https://github.com/Snektron/pareas) as part of their own successful interview process.
- **Insider Reveals How to Get Kernel Job Without Internship**: Someone with an interest in GPU posted that they secured a job through a combination of a GPU-related thesis and luck, plus getting through the interview process.
   - According to the person, good knowledge of GPUs can bypass the need for previous experience and internship.


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1405833745772314667)** (1 messages): 

> `MI300 pytorch, OMP missing` 


- ****MI300** env lacks **OMP****: The **MI300** environment appears to be missing **OMP** for `pytorch.compile` based on a user report.
- **Link to Debug Error included**: A user shared a [link to the full debug error](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251) for further investigation.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1405638909580283936)** (10 messages🔥): 

> `trimul leaderboard, A100, H100, B200` 


- **Trimul Leaderboard Sees New Speedsters**: One member got **second place** on **A100**: **10.4 ms** then quickly got **first place** on **H100**: **3.95 ms** and **first place** on **A100**: **7.53 ms**.
   - Later, the member got **first place** on **B200**: **2.35 ms**, then again **first place** on **A100**: **6.01 ms** and yet again **first place** on **B200**: **2.04 ms** and finally successful on **H100**: **3.74 ms**.
- **A100 and H100 also see activity**: Another member got **5th place** on **A100**: **13.2 ms**.
   - The member followed up to get **second place** on **H100**: **6.42 ms** and finally successful on **A100**: **14.7 ms**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1405929507554070674)** (10 messages🔥): 

> `Meeting Attendance, Large PR Review, Connection Error Debugging` 


- **Missed Meeting Mishaps**: Several members mentioned missing a meeting due to timezone confusion and scheduling conflicts, including one member available only for the first **10 minutes**.
   - One member quipped that the **8am** meeting time was a bit brutal.
- **Reviewing Scope Creep**: A member commented on a PR with **300 file changes**, joking that it was a "lil out of scope".
   - Another member added that the code was *grass-fed hand-crafted*.
- **Troubleshooting Connection Errors**: A member reported seeing a connection error and is attempting to debug its source, guessing it might be from **db_client**.
   - They mentioned difficulty in getting a stack trace to diagnose the issue.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1405627475521962098)** (47 messages🔥): 

> `Kimi K2 Technical Report, GLM-4.5 vs Kimi K2, Kimi hallucinations, Kimi's Web UI, Kimi future updates` 


- **NotebookLM Video Edges out Kimi PPT**: Members compared a **PPT generated by Kimi** with a **video overview generated by Google's NotebookLM** for the Kimi K2 technical report, with the consensus leaning towards NotebookLM's video due to its audio and more flexible layout (see [attached video](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&)).
   - While both were appreciated, one member expressed a preference for reading over listening to AI-generated audio, but noted the potential of video overviews, especially in education.
- **Kimi K2 Beats GLM in Writing Skills**: Despite a feeling that **GLM-4.5** might surpass **Kimi K2** in overall performance, users lauded **Kimi** for its superior writing style and proactive error detection.
   - One user was *“genuinely surprised”* when **Kimi** *“out of the blue told me No”*, appreciating its candor.
- **Combating Kimi's Hallucinations**: Users expressed a desire for **Kimi** to hallucinate less, even with web search enabled, noting that while **GLM** may take longer, it hallucinates less frequently.
   - A user stated they consistently use the thumbs down button to report hallucinations.
- **Kimi Fans Eagerly Await 'Kimi Thinking'**: Members are eagerly anticipating the arrival of **'Kimi Thinking'** and reasoning and multimodel capabilities.
   - There are questions as to whether this will arrive in the form of **Kimi K-2** or **Kimi K-3**, but there are no firm ETAs yet.
- **Dark Mode Enhances Kimi's Web UI**: A user shared their customized **Kimi Web UI** with a dark mode extension, expressing a strong preference for it over the default grey interface (see [attached screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&)).
   - Another user confirmed that only the username and server roles are passed to the Moonshot API.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1405648729134076044)** (4 messages): 

> `AI Stock Portfolio Agent, Web Scraping AI Agents, Multimodal AI Applications, Legal Knowledge Graphs` 


- **AI Stock Portfolio Agent Arrives**: LlamaIndex has introduced a framework to build a complete **AI stock portfolio agent**, integrated with [@CopilotKit](https://www.copilotkit.ai/)'s AG-UI protocol for seamless frontend-backend communication; a comprehensive tutorial is included to create a sophisticated investment analysis tool.
   - The tutorial combines the power of [this framework](https://t.co/fQDNPIQoqR) to create a sophisticated investment analysis tool.
- **Brightdata and LlamaIndex Launch Web Scraping AI Agents**: LlamaIndex announced a new walkthrough with [@brightdata](https://www.brightdata.com/) on building **web-scraping AI agents** with LlamaIndex's agentic framework, focusing on reliable web access and robust web scraping workflows.
   - The walkthrough details how to set up workflows that can handle dynamic content and build **intelligent agents** that can navigate [here](https://t.co/IBgSLBM6XW).
- **Multimodal AI Apps Analyze Markets Visually**: LlamaIndex announced building **multimodal AI applications** that analyze both text and images for market research and surveys.
   - These applications are designed to process images and documents together in a unified AI pipeline, extract insights from visual market data like charts, graphs, and product images, and combine multimodal [capabilities](https://t.co/fOMFLXWarG).
- **LlamaCloud and Neo4j Transform Legal Documents into Knowledge Graphs**: LlamaIndex announced a comprehensive tutorial on transforming unstructured legal documents into **queryable knowledge graphs** that understand not just content, but relationships between entities.
   - This workflow leverages **LlamaCloud** and [@neo4j](https://neo4j.com/) for legal contract analysis and is detailed [here](https://t.co/MPSfPiS2Cv).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1405664216601329764)** (28 messages🔥): 

> `Pydantic Models vs JSON Schema for Tool Calls, Vector Store Errors After Update, Progress Bar Issue with num_workers > 1, Iterating Over Nodes/Doc_IDs in Vectorstore` 


- **Pydantic vs JSON Schema Showdown**: A member inquired whether tool calls require a **Pydantic model** or if a **JSON schema** suffices, noting the redundancy of converting JSON to a Pydantic model only to have it unpacked back into JSON.
   - Another member pointed out **Pydantic's** `create_model()` function doesn't directly accept a **JSON schema**, highlighting the need for a tool/package to handle this conversion.
- **Vector Store Gets Attribute Error After LlamaIndex Update**: After updating to version **0.13.1**, a user encountered an `AttributeError` during retrieval from a **PGVectorStore** when using `RetrieverQueryEngine` with `OpenAI` and `text-embedding-3-small`.
   - The error arises because the `output` is a `str` with no attribute `json`, stemming from the **LLMStructuredPredictEndEvent** in `openinference.instrumentation.llama_index`.
- **Progress Bar Pandemonium with Multiprocessing**: A user highlighted that the `progress_bar=True` feature doesn't function correctly when `num_workers > 1` due to the use of **multiprocessing**.
   - It was suggested that using **async concurrency** could offer a smoother experience, however the `async pipeline.arun` method still uses multiprocessing.
- **Nodes and Doc IDs Missing in Action in Vector Stores**: A user expressed frustration over the inability to iterate over nodes or obtain a list of `doc_ids` in most LlamaIndex vector stores, particularly noting the absence in **Opensearch** and **awsdocdb**.
   - A workaround involves setting `similarity_top_k` to a high number, but this is inefficient and may not be supported by all OSS; the `get_nodes()` method exists on the base `vector_store` class, however, it is not implemented for Opensearch or awsdocdb, which is an opportunity for a PR.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1405905432920326265)** (1 messages): 

> `DSPy optimizes CrewAI, CrewAI agent prompts` 


- **DSPy optimizes CrewAI agent prompts**: A course teaches how **DSPy optimizes CrewAI** agent prompts in a real production use case to build smarter, cheaper agents with proven methods.
   - You can check the course [here](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E).
- **Build smarter, cheaper agents with proven methods**: The course focuses on **DSPy optimization** for CrewAI agents.
   - It emphasizes building more efficient and intelligent agents through **proven methodologies**.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405744293439733830)** (7 messages): 

> `Audio Transcription in NotebookLM, NotebookLM Interface Redesign` 


- **Audio Uploads Auto-Transcribed to NotebookLM**: A member inquired about obtaining audio transcripts, to which another member responded that they upload **MP3 audio files directly to NotebookLM**.
   - The member clarified that **NotebookLM** itself handles the transcript generation.
- **NotebookLM Interface Redesign Underway**: One member mentioned they are attempting to redesign **NotebookLM**, and shared a Figma screenshot of the proposed changes.
   - The member apologized for any disappointment, clarifying it was just a design concept, not a functional update.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1405718164716650520)** (23 messages🔥): 

> `Explainer Video Voice, Feature Request Feedback, Dev Interaction, Prompt Limit` 


- **Voice Gender Swaps Explainers**: A user reported that their explainer videos suddenly started generating with **male voices** instead of the usual **female voices** and questioned why this happened.
   - There was no clear resolution or explanation provided in the messages.
- **User Requests Acknowledgement of Feature Requests**: A user questioned whether anyone from the **NotebookLM development team** is actually reading through the **feature requests** posted in the Discord channel.
   - They expressed a desire for some sign of life or feedback from the developers to encourage continued contributions.
- **NotebookLM Devs Acknowledge Reading Posts but Can't Respond to Everything**: A Google developer stated that *the devs read the posts*, but they don't have time to respond to everything and spend a lot of their time **banning spammers**.
   - Other users suggested that even occasional acknowledgements or AI-compiled summaries could help encourage user contributions.
- **Users bump into Prompt Limits in NotebookLM**: A user asked if there is a limit to the **number of words** that can be included in a single question in **NotebookLM** after failing to ask a case-related question containing about **857 words**.
   - Another user suggested splitting the prompt into multiple parts or trying **Gemini**.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1405902903151169648)** (1 messages): 

> `CrewAI agent prompts, DSPy` 


- **Optimize CrewAI agent prompts with DSPy**: Members shared a link to learn how **DSPy optimizes CrewAI agent prompts** in a real production use case: [https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E).
   - The course claims to teach users how to *build smarter, cheaper agents with proven methods*.
- **DSPy and CrewAI unite**: The course teaches users how to optimize CrewAI using DSPy.
   - It enables smarter, cheaper agents using proven methods.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1405627855324315649)** (22 messages🔥): 

> `DSPy and Databricks, GEPA Error, MLflow and DSPy` 


- **Databricks Not Sponsoring DSPy**: A user asked if **Databricks** sponsors or owns the **DSPy** project, and another user clarified that DSPy is **MIT-licensed open source**, with Databricks contributing significantly through a team of core developers.
- **GEPA Bug Fixed**: A user encountered a `ValueError` when using **GEPA** with the **RAG tutorial**, and another user confirmed that [this was a bug in GEPA code](https://github.com/stanfordnlp/dspy/pull/8647) that has been fixed; users should upgrade to **DSPy 3.0.1**.
   - The depreciated param is in that dspy.evaluate import, and the fix is `pip install -U dspy`.
- **MLflow Tracks DSPy Sub-Modules automatically**: A user inquired about integrating **DSPy modules** tracking with **MLflow** for a **text2sql pipeline** and was advised to use `mlflow.dspy.autolog()` instead of `mlflow.autolog()` to automatically track all sub-modules.
   - Using `mlflow.dspy.autolog()` will display the **SQLGenerator**, **Validator**, and **Reflector** as nested spans in the **MLflow UI's Traces tab**, as detailed in the [MLflow DSPy integration documentation](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx) and the [DSPy MLflow tutorial](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md).
- **Logprob Surprise as fitness Function**: A user shared a tweet [TogetherCompute Status](https://x.com/togethercompute/status/1956416013404406018), and guessed that they’re basically doing **GEPA** with **logprob surprise** as the **fitness function**, but for mental health models in prod.
- **Community Engagement Requested**: A member requested more engagement from the 6500 people in this discord, and more contributions to the docs and all.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1405897484248813679)** (1 messages): 

> `CrewAI, DSPy Optimization, Prompt Engineering` 


- **CrewAI Prompt Optimization Course Drops**: A member announced a [Udemy course](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E) demonstrating how to optimize **CrewAI prompts** with **DSPy**.
   - The course will show how to inject them back to the **LLM** so the **LLM** uses better prompts than those stitched together by **CrewAI**.
- **DSPy Enables Optimized CrewAI Prompts**: The new course uses **DSPy** to optimize prompts.
   - Optimized prompts are then injected back into the **LLM** improving on the standard approach in **CrewAI**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405629920868171879)** (8 messages🔥): 

> `CI speed, tinygrad release, tinygrad size` 


- **CI Speed Hampering Productivity**: A member expressed frustration with slow CI speeds, stating they could work faster with quicker CI and linked [chatgpt analysis](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74).
- **Tinygrad Release Imminent**: There was a suggestion to do a **tinygrad release** soon.
- **Tinygrad Size Swells**: A member questioned why **tinygrad 0.10.3** is **10.4 MB**, hinting at a possible size issue.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1405802310633787423)** (14 messages🔥): 

> `WSL2 Support, print_tree removal` 


- **WSL2 Tinygrad Bug Surfaces**: A user encountered an issue where adding two tinygrad Tensors created from PyTorch tensors resulted in all **0s**, and provided a [full script](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656) to reproduce the bug on WSL2.
- **print_tree function bites the dust**: The `print_tree` function has been replaced with a simple `print` function.
   - The user noted it *lost some of its formatting*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1405710824835780628)** (12 messages🔥): 

> `Aider Benchmark, litellm Errors, Open Source Entitlement, llama.cpp PR #15181` 


- **Aider Benchmark Plagued by Timeouts**: A member ran the **Aider benchmark** against a local **gemma3:12b** model and encountered frequent timeouts after **10.5 hours** and **221/225 tests** due to the model's inability to respond within the **600-second** limit, resulting in *litellm.APIConnectionError* errors.
   - They shared the error log which shows the model attempting to send around **300k tokens**, exceeding the **131,072 token limit** and causing test failures.
- **Continuing Aider Benchmark**: A member suggested using `ctrl+c` to exit the benchmark, restarting the inference server, and then using the `--cont` flag to resume the benchmark from where it left off.
   - They also pointed to a [merged pull request](https://github.com/ggml-org/llama.cpp/pull/15181) in *llama.cpp* that might improve local model performance.
- **OSS Maintainer's Burden**: A member criticized another's suggestion to make the benchmark automatically configurable for each LLM, labeling it as *entitlement* and lamenting that such attitudes cause *countless OSS maintainers to throw in the towel*.
   - Another member countered that it was merely *curiosity*, leading to further disagreement on what constitutes entitlement in open-source interactions.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1405695906635845682)** (7 messages): 

> `Aider with Local Models, Aider Line Number Accuracy, Unit Test Coverage with Aider` 


- **Local AI/Aider Models Bring Debugging Agony**: A member expressed difficulty using **aider** with local models like **ollama**, **lmstudio**, and **vllm**, noting slow performance even with powerful hardware.
   - They suggested the need for a tutorial video on setting up **aider** with these tools for local development and debugging.
- **Aider's Line Numbering System Questioned**: A member inquired about how **aider** determines line numbers, especially in the context of generating unit tests for specific code coverage.
   - The issue arises when **aider** misreports the line numbers, leading to incorrect test coverage, despite attempts to refresh the map and clear chat history.
- **LLM Accuracy Impacts Unit Test Coverage**: A member reported that **qwen3-coder** and **gemini-pro** inaccurately identify line numbers in coverage reports, sometimes missing the coverage entirely.
   - This inconsistency leads to questions about whether **aider** relies on the **LLM's accuracy** for line number identification, suggesting a need to explore alternative methods for accurate unit test generation.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1405881855823188060)** (3 messages): 

> `Grok4, Quota Increase, Benchmark Costs` 


- **Grok4 Location Remains Elusive**: A member inquired about the whereabouts of **Grok4**.
   - Another member responded that *it's in the article* but the request to increase the **quota** needed to execute the tests was ignored.
- **Grok4 Benchmark Costs Thousands**: A member noted they *spend multiple thousands dollars during the development of this benchmark*.
   - This highlights the significant financial resources required for advanced AI model benchmarking.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1405736806930055170)** (22 messages🔥): 

> `Manus Credit Deductions on Errors, Manus Deployment Issues, Manus Team Accounts, Add-on Credits Removal, Manus in the Wild Challenge Winner` 


- **Manus Credit Deductions Draw Ire**: Users express frustration over **Manus** deducting credits even when it makes errors, making task completion difficult compared to other AIs like **Claude AI**.
   - One user reported *spending high amounts of credits* only for **Manus** to make a simple change that broke the entire application, deeming it non-functional.
- **Manus Deployment Stumbles**: Users report issues with **Manus** deployments, where websites created from the same **GitHub** repository differ significantly, especially with large folders, illustrated by comparison of [affilify.eu](https://affilify.eu) and a **Manus** hosted site [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space).
   - A community manager noted that **Manus** isn't designed as a coding agent or pure development tool, so deployment isn't its strength, but they're working on improvements.
- **Add-on Credit Packages Recede**: Users question the removal of add-on credit packages, which are now exclusively available for **Pro** users.
   - A community manager explained that this change ensures consistent speed and quality for heavy users and suggested bundling similar questions, being concise, and avoiding repeated requests to maximize credit efficiency.
- **Manus Team Accounts Spark Interest**: A user inquired about the possibility of a **Manus** team account for shared credit usage.
   - A community manager confirmed that **Manus** does offer a team plan, directing users to the [official website](https://manus.ai) for details.
- **Users Bemoan Credit Consumption**: One user shared a frustrating experience of burning through **30,000 credits** attempting to get their website up, encountering issues with mock sites and template implementations.
   - They criticized the system's inconsistency, where it's *smart as hell but then suddenly turns dumb*, leading to wasted credits and suspected stall tactics.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405855716916461669)** (9 messages🔥): 

> `Cohere Labs, Pokemon emojis, PAX Omeganauts Discord` 


- **Cohere Labs contact info sought!**: A member asked where to connect with **Cohere Labs** folks, another member suggested this Discord channel.
   - Another member directed the user to [this link](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648).
- **Discord channel Pokemon Emoji-fied!**: A member suggested adding more **Pokemon emojis** to the channel, as there are available slots.
   - The member noted that the emojis come from the **PAX Omeganauts Discord**.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405640013198131345)** (5 messages): 

> `AI Research, writenode, CV+ML pipeline` 


- **AI Researcher Seeks Collabs**: An **AI researcher** with a deep interest in **reasoning and conscious capabilities** is looking for collaborations to develop advanced tech for the future.
   - The member is open to collaborations from any sub domain.
- **Legal Pro Aligns with AI**: A **legal professional**, gamer, and lover of philosophy currently working for the USG is self-teaching **AI alignment theory and mechanics**.
   - The member is excited to be here.
- **writenode builder uses Cohere**: Josh is building **writenode**, *an in browser, cognitive thought partner, and creative companion*, and uses **Cohere**.
   - He does not have a developer or coding background since before December last year.
- **Psych PhD Returns to AI**: A member is returning to **AI research** after dabbling in a human psychology PhD the past 5 years.
   - Their interests are in **sound+music**, and using tech tools to help us express our creativity.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1405985104920055959)** (3 messages): 

> `Discord Invite Links, Channel Spam` 


- **Discord Invites Flood Channel**: A member spammed the channel with [Discord invite links](https://discordapp.com/invite/HjWfRbqBB8) multiple times, tagging *everyone*.
   - The invite link was repeated three times in quick succession.
- **Invite Link Repetition**: The same [Discord invite link](https://discordapp.com/invite/HjWfRbqBB8) was posted repeatedly.
   - This resulted in a spam-like effect, potentially disrupting the channel's usual discussions.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1405984973906903060)** (3 messages): 

> `Discord Invite Link, HjWfRbqBB8, Channel Invitation` 


- **Discord Invite Link Floods Channel**: A member repeatedly shared a [Discord invite link](discordapp.com/invite/HjWfRbqBB8) in the channel, possibly to attract more users.
   - The member tagged `@everyone` multiple times, which might be considered excessive or disruptive.
- **Channel Invitation Blitz**: The repeated posting of the [same Discord invite](discordapp.com/invite/HjWfRbqBB8) suggests an attempt to boost channel membership.
   - The use of `@everyone` indicates the message was intended for all members, regardless of their interest in the invitation.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405660404918652948)** (2 messages): 

> `Elicitations Specification, MCP Server Conversion` 


- **Elicitations Spec Clarity Sought**: A member inquired about the [Elicitations specification](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) regarding *who* is responsible for translating message/field descriptions into the user's language.
   - Specifically, they seek clarification on whether **tools** should handle language detection and internationalization, or if **MCP Clients** are expected to translate, potentially using an LLM.
- **MCP Server Transformation Question**: A member inquired whether *there exists some tool to turn a local mcp server into a remote mcp server?*
   - No links or additional context was provided.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1405750824461668434)** (3 messages): 

> `Unifi MCP, Unraid MCP, Syslog MCP, AI Agent Workflows, AI Security` 


- **MCP Servers for Homelabbers Arrive**: A member shared a few MCP (presumably, **Management Control Panel**) servers for the homelabbers, specifically: [Unifi MCP](https://github.com/jmagar/unifi-mcp), [Unraid MCP](https://github.com/jmagar/unraid-mcp), and [Syslog MCP](https://github.com/jmagar/syslog-mcp).
- **PulseMCP Turns Newsletter Tedium to Agent Automation**: **PulseMCP** used goose to turn a tedious newsletter workflow into agent-powered automation with a human in the loop.
   - More details on the automation can be found at [this blogpost](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe).
- **AI Security Seeks Input On Security Concerns**: One member posted about building **AI security** that stops attacks before they even start with mathematical security certainty.
   - They are looking for Dev input on security concerns, and linked to [a survey](https://form.typeform.com/to/xTKa05F9).


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405631570194337793)** (4 messages): 

> `Strix Halo profitablility, Dolphin chat template, Quantum computers, PC Memory` 


- **Strix Halo's Profitability Plummets**: The **Strix Halo**, despite its impressive specs, requires a **year of 24/7 inference** to become profitable due to its slower inference speed of **53 tokens/sec** compared to **GPT-OSS 120B** on **OpenRouter**.
   - One user noted that configuring it for **LLMs** at $2000 is inefficient compared to cloud alternatives offering **200-400 tokens/sec**.
- **Dolphin Chat Template Quest**: A user is seeking a working chat template for **gpt4all** compatible with **Dolphin-2.2.1-mistral-7b-gptq**.
   - Another member suggested asking model makers to upload a template with a **jinja** template.
- **Quantum Computing Teaspoons?**: One user speculates on the future availability of quantum computers and the possibility of selling **qubits on the teaspoon**.
   - They mentioned news about **fully working quantum computers**, indicating potential advancements in the field.
- **Memory Modules and Moore's Law**: A user mentioned that old-fashioned PCs can expect to see **higher capacity memory modules** and **DDR6** in late 2027 or 2028.
   - They express excitement about the potential of micro PCs with high RAM and VRAM capacities, especially for small businesses.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1406014763804397599)** (1 messages): 

> `Maternity Leave, Team Contact During Leave` 


- **Maternity Leave Commences!**: A member announced they will be on **maternity leave** from **August 25th** until **February 2026**.
   - They look forward to catching up upon their return.
- **Team's Coverage Plan Revealed**: While they are away, the team will be monitoring <@1334161614949056532>.
   - Members can also reach out to <@709918328306663424> with any questions or concerns.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

__nathan: <@132818429022437376> how did this go?
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1405634992792670208)** (1 messages): 

> `Windsurf Wave 12, DeepWiki Integration, Vibe and Replace feature, Smarter Cascade Agent, Dev Containers Support` 


- **Windsurf Wave 12 is Released!**: Windsurf Wave 12 brings the first integrations of **Devin's intelligence** and capabilities directly into the Windsurf IDE.
   - Key features include a **new UI design**, **DeepWiki Integration**, **Vibe and Replace**, a **Smarter Cascade Agent**, **Faster Tab**, **Dev Containers Support**, and **over 100 bug fixes** - [see the changelog](https://windsurf.com/changelog), [read the blog](https://windsurf.com/blog/windsurf-wave-12), [watch the Wave 12 video](https://www.youtube.com/watch?v=-7gm8mST9QU), [X/Twitter](https://x.com/windsurf/status/1956074019393876280), and [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/).
- **DeepWiki Integration brings AI to the IDE**: **DeepWiki Integration** allows users to hover over code symbols for **AI-powered explanations** (not just basic type info).
   - Users can also use **CMD/Ctrl+Shift+Click** to open detailed explanations in the side panel and add to Cascade context.
- **Vibe and Replace revolutionizes bulk editing**: The **Vibe and Replace** feature provides revolutionary bulk editing capabilities by finding exact text matches.
   - It allows users to apply **AI prompts** for intelligent, context-aware transformations across their entire project.
- **Smarter Cascade Agent gets Always-On Planning**: The **Smarter Cascade Agent** now features an always-on planning mode with autonomous to-do lists.
   - It also includes revamped tools designed to provide smarter responses.
- **Dev Containers Supported Natively**: Windsurf now supports working with containers directly via remote SSH access.
   - This enhancement simplifies development workflows involving containerized environments.
