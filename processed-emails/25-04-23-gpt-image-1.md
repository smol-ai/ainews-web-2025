---
id: 23232324
title: >-
  gpt-image-1 - ChatGPT's imagegen model, confusingly NOT 4o, now available in
  API
date: '2025-04-23T05:44:39.731046Z'
description: >-
  Autoregressive Imagegen introduces a new official API for GPT 4o, enabling
  advanced image generation, editing, transparency, and moderation features. The
  news covers recent AI model performance, benchmarks, supercomputer scaling,
  and Nvidia's new multimodal model for image and video captioning.
companies:
  - openai
  - nvidia
models:
  - gpt-image-1
  - gpt-4
  - gpt-4-mini
  - gpt-4.1
  - gemini-2.5-pro
  - qwen-2.5-vl-72b
  - eagle-2.5-8b
  - dam
topics:
  - image-generation
  - multimodal
  - model-performance
  - benchmarking
  - supercomputing
  - content-moderation
  - long-video-understanding
  - ai-scaling
  - virology-capabilities
---


**Autoregressive Imagegen is all you need.**

> AI News for 4/22/2025-4/23/2025. We checked 9 subreddits, [**449** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**213** channels, and **6203** messages) for you. Estimated reading time saved (at 200wpm): **503 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

When Imagegen launched it was specifically branded as a capability of GPT 4o. With the Ghibli wave everyone rushed to create convoluted browser automations to "apify" a nonexistent imagegen API.

Now, the offical API is here ([docs](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1)), [cost](https://platform.openai.com/docs/guides/image-generation#cost-and-latency)), capable of new generations (using [references](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1#create-a-new-image-using-image-references)) as well as partial/full image editing (using  [masks](https://platform.openai.com/docs/guides/image-generation#edit-an-image-using-a-mask-inpainting)).

![https://cdn.openai.com/API/docs/images/images-gallery/furniture-poster.png](https://cdn.openai.com/API/docs/images/images-gallery/furniture-poster.png)

It supports [alpha channel transparency](https://platform.openai.com/docs/guides/image-generation#transparency) and, in a first for OpenAI, a ["low" content moderation policy](https://platform.openai.com/docs/guides/image-generation#content-moderation), as well as (as [Kevin Weil notes](https://x.com/kevinweil/status/1915103388993302646)):


* moderation sensitivity
* image quality/generation speed
* quantity of images generated
* whether the background is transparent or opaque 
* output format (jpeg, png, webp)


---

# AI Twitter Recap

**Language Models and Performance**

- **OpenAI's models, particularly o3 and o4-mini, are making waves in the AI Arena**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1915078057452573142) reported that **o3 ranks #2 overall**, tying with **Gemini-2.5-Pro** in **Style Control, Math, Coding, and Hard Prompts**, while **o4-mini** broke into the top 10, claiming **#1 in Math, surpassing o1**. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1915078061126725755) also noted **o3 as #1 in Style Control, Hard Prompts, Coding, and Math** and both **o3 and o4-mini as #1 in Math**.
- **Performance variance in LLMs on extended tasks**: [@_philschmid](https://twitter.com/_philschmid/status/1914682660854604186) highlighted a new real-world benchmark called **Vending-Bench**, which simulates long-term vending machine operation. The benchmark reveals **high performance variance** in LLMs, prone to catastrophic failures and inconsistencies, even with larger memory.
- **Insights on o3 vs o4-mini**: [@willdepue](https://twitter.com/willdepue/status/1914549086822293916) shared some insights on the models, with **o3 superior in GPQA** (requiring more world knowledge), instruction following, chat, and emotional reasoning, while **o4-mini excels in Codeforces and AIME/math** because it makes the model think really hard, and has a supercharged multimodal use case.
- **GPT-4.1 performance**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1915078061126725755) reported that **GPT-4.1 ranks in the top 5** for **Hard Prompts, Math, and Longer queries**.
- **Nvidia's Eagle 2.5 matches GPT-4o and Qwen2.5-VL-72B in long-video understanding**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914517474370052425) noted that **Eagle 2.5-8B matches the results of GPT-4o and Qwen2.5-VL-72B on long-video understanding**.
- **AI supercomputer scaling**: According to [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1915098223082873015), **AI supercomputer performance has doubled every 9 months**, driven by deploying more chips and higher performance per chip.  The hardware cost doubles roughly every year, with **xAI's Colossus estimated at $7 billion**. Geographically, the **US dominates with 75%** of global AI supercomputer performance.
- **Virology Capabilities Test (VCT) results**: [@DanHendrycks](https://twitter.com/DanHendrycks/status/1914696657813561799) reported that, according to their new Virology Capabilities Test (VCT), **OpenAI’s o3 now outperforms 94% of expert virologists** regarding the expert-level tacit knowledge needed to troubleshoot wet lab protocols.

**New Models and Releases**

- **Nvidia's Describe Anything Model (DAM)**: [@reach_vb](https://twitter.com/reach_vb/status/1914962078571356656) and [@mervenoyann](https://twitter.com/mervenoyann/status/1914980803055862176) highlighted **Nvidia's Describe Anything 3B (DAM)**, a multimodal LLM for detailed localized image and video captioning, which integrates full-image/video context with fine-grained local details. It is now live on Hugging Face, linked by [@_akhaliq](https://twitter.com/_akhaliq/status/1914917564137828622). DAM takes user-specified regions as input and generates detailed localized descriptions.
- **RealisDance-DiT by Alibaba**: [@_akhaliq](https://twitter.com/_akhaliq/status/1915101805916377596) announced **Alibaba's RealisDance-DiT**, a simple yet strong baseline for controllable character animation in the wild.
- **LiveCC by Google**:  [@_akhaliq](https://twitter.com/_akhaliq/status/1915094398364197101) shared **LiveCC**, a video LLM capable of real-time commentary, trained with a novel video-ASR streaming method, achieving SOTA on both streaming and offline benchmarks.
- **Vidi by ByteDance**: [@_akhaliq](https://twitter.com/_akhaliq/status/1914925322413264937) announced **ByteDance's Vidi**, a large multimodal model for video understanding and editing.
- **Adobe's DRAGON**:  [@_akhaliq](https://twitter.com/_akhaliq/status/1914602497148154226) shared **Adobe's DRAGON**, which optimizes diffusion generative models using distributional rewards.
- **Uni3C by Alibaba**: [@_akhaliq](https://twitter.com/_akhaliq/status/1914619143925432338) highlighted **Alibaba's Uni3C**, which unifies precisely 3D-enhanced camera and human motion controls for video generation.
- **Flex.2-preview**: [@ostrisai](https://twitter.com/ostrisai/status/1914799647899722198) announced **Flex.2-preview**, an 8B parameter model with text to image, universal control, and inpainting, fine-tunable with AI-Toolkit and Apache 2.0 licensed.
- **Dia 1.6B, a SOTA open source TTS model**: [@reach_vb](https://twitter.com/reach_vb/status/1914796234331877431) posted about **Dia 1.6B**, a SOTA open source TTS model that beats ElevenLabs/Sesame, Apache 2.0 licensed, capable of producing non-verbal sounds, with zero-shot voice cloning and real-time TTS synthesis.
- **BLT, a Byte Latent Transformer**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1915103765981454512) highlighted a new language model architecture, **Byte Latent Transformer (BLT)**, that operates directly on bytes instead of tokens and outperforms Llama 3 in multiple benchmarks.
- **OpenAI releases image gen model in the API**: [@kevinweil](https://twitter.com/kevinweil/status/1915103387592409215) and [@sama](https://twitter.com/sama/status/1915110344894435587) announced that **image gen is launched in the OpenAI API**, featuring more accurate and high-fidelity images, diverse visual styles, precise image editing, rich world knowledge, and consistent text rendering.

**Research and Papers**

- **AI Safety Research Cooperation**: [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1915039527367852285) discussed how geopolitical rivals can cooperate on **AI safety research** in ways that benefit all while protecting national interests.
- **Paper on Embodied Agents, Smart Cities, and Earth Science**: [@dair_ai](https://twitter.com/dair_ai/status/1914674606910157102) highlighted a paper that surveys how spatial intelligence manifests across disciplines by connecting human spatial cognition with how LLMs handle spatial memory, representations, and reasoning.
- **Survey of Frontiers in LLM Reasoning**: [@dair_ai](https://twitter.com/dair_ai/status/1914674604926292322) shared a survey categorizing **LLM reasoning methods** by when reasoning occurs (inference-time vs. training) and the system's architecture (standalone vs. agentic or multi-agent).
- **Nvidia's Eagle 2.5**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914517474370052425) highlights **Nvidia's Eagle 2.5**, noting that **Eagle 2.5-8B** matches the results of **GPT-4o and Qwen2.5-VL-72B** on long-video understanding.
- **Tina: Tiny Reasoning Models via LoRA**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914966644314747300) notes that "the best Tina model achieves a >20% reasoning performance increase and 43.33% Pass@1 accuracy on AIME24, at only $9 USD post-training and evaluation cost (i.e., an estimated 260x cost reduction)."
- **Learning Adaptive Parallel Reasoning with Language Models**:  [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914895805707936035) and [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914893575420334567) discuss this paper, which enables language models to orchestrate both serialized and parallel computations end-to-end.
- **Dynamic Early Exit in Reasoning Models**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914889033085542537) wrote about a paper which allows LLMs to self-truncate CoT sequences by dynamic early exit, reducing the CoT length by ~35% while improving accuracy by 1% - 10%.
- **TTRL: Test-Time Reinforcement Learning**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914877762168627612) highlights a novel method for training LLMs using RL on *unlabeled* data by utilizing the priors in the pre-trained models
- **Entropy Rectifying Guidance for Diffusion and Flow Models**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914596593527087341) notes this paper proposes Entropy Rectifying Guidance (ERG), a guidance mechanism based on modifying the energy landscape of the attention layers.
- **NEMOTRON-CROSSTHINK: Scaling Self-Learning beyond Math Reasoning**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914595485148701079) posts about the framework, which systematically incorporates multi-domain corpora into RL training to improve generalization across diverse reasoning tasks and demonstrates improved accuracies on both math and non-math reasoning benchmarks.
- **SRPO: A Cross-Domain Implementation of Large-Scale Reinforcement Learning on LLM**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914622980296192357) notes that it successfully surpasses the performance of DeepSeek-R1-Zero-32B on the AIME24 and LiveCodeBench benchmarks, relying solely on RL, without prior Supervised Fine-Tuning (SFT).
- **OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914624649142657199) shares details on OmniV-Med, including the medical dataset **OmniV-Med-Instruct** and a rotary position-adaptive encoder that processes multi-resolution 2D/3D images and videos.
- **Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914630337373913332) shares details from a paper which analyzes inference-time scaling methods for both reasoning and non-reasoning models on challenging reasoning tasks.

**AI Agents and Tooling**

- **Agentic Document Workflows**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1915109277498585569) outlined a reference architecture for building agents over documents, dividing it into four stages: parsing and extraction, retrieval, reasoning, and action-taking.
- **Code Agents with Hugging Face smolagents**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1915101920500564406) announced a new short course on building code agents with Hugging Face smolagents, taught by @Thom_Wolf and @AymericRoucher, focusing on how code agents outperform function-calling agents and how to run them safely. [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1915081924302839984) also promoted the course, noting that code agents can make agents more efficient, more reliable, and better suited for complex tasks.
- **LlamaIndex's integration with @milvusio**: [@llama_index](https://twitter.com/llama_index/status/1914815391798534571) highlighted the integration now supports full-text search with BM25, allowing hybrid search for RAG pipelines.
- **AI-powered Compliance Report Generation**: [@llama_index](https://twitter.com/llama_index/status/1914727722615755178) shared an agentic workflow for generating compliance reports, boiling down regulatory language, comparing it against contract language, and generating a concise summary.
- **Super Agent by @genspark_ai**:  [@svpino](https://twitter.com/svpino/status/1914744937851330695) introduced Super Agent, a fully autonomous AI agent, and described its use cases in planning trips, creating short videos, and generating presentations, mentioning the agent automatically writes, researches, and compiles the necessary insights to generate presentations.
- **Listen, a platform for AI-driven market research**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1915140553806946751) noted that Listen raised $27M from Sequoia to replace surveys and focus groups with thousands of AI interviews, providing interviews, analysis, and insights in under 24 hours.
- **LangSmith alerts for AI application monitoring**: [@hwchase17](https://twitter.com/hwchase17/status/1914726837726679508) and [@LangChainAI](https://twitter.com/LangChainAI/status/1914713424539607188) announced alerts in LangSmith to catch and alert on AI application failures, with real-time notifications on error rates, run latency, and feedback scores.  [@LangChainAI](https://twitter.com/LangChainAI/status/1914739189087633510) shared how Trellix is using LangGraph and LangSmith.
- **Open Deep Research in TypeScript**: [@togethercompute](https://twitter.com/togethercompute/status/1914721242285838498) announced **Open Deep Research** in TypeScript, a rewrite of their Python implementation, specifically made for web devs, to easily connect to ExaAILabs for searches.
- **Cherry Studio app**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1915139593264832901) endorsed the Cherry Studio app.
- **Perplexity Assistant on iOS**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1915064137110954327) introduced **Perplexity Assistant on iOS**, enabling the AI app to answer questions and take basic actions on the iPhone, such as playing media, drafting emails, moving meetings, booking rides, and setting reminders.
- **GPT-image-1 integrations**:  [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1915097075722899818) noted that Figma is leveraging gpt-image-1 to generate and edit images from simple prompts, enabling designers to rapidly explore ideas and iterate visually directly in Figma. [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1915097077878530334) also noted that HeyGen is using gpt-image-1 to enhance avatar creation, specifically improving avatar editing within the platform.

**ML Engineering and Deployment**

- **Importance of measurement in AI product development**: [@_philschmid](https://twitter.com/_philschmid/status/1914999903882748171) summarized @HamelHusain's insights on building a successful AI product, emphasizing measurement and iteration over tools, highlighting the importance of error analysis, data viewers, domain experts, synthetic data, and binary judgments.
- **MLOps and System Design**: [@svpino](https://twitter.com/svpino/status/1915031713866256874) emphasizes the importance of MLOps and designing complex, real-world systems for AI engineers, noting the trend of models writing code + engineers designing, architecting, and managing systems.
- **Modular software design principles**: [@lateinteraction](https://twitter.com/lateinteraction/status/1914720046808764498) argues that the central problem in AI research is violating modularity, advocating for unification to address redundancy and disconnection.
- **Torch Titan and Context-parallel training**: [@vikhyatk](https://twitter.com/vikhyatk/status/1914832180498587839) mentioned context-parallel training for long contexts in Torch Titan.
- **Finetuning on raw reasoning traces**: [@Muennighoff](https://twitter.com/Muennighoff/status/1914768451618660782) notes that finetuning on raw DeepSeek R1 reasoning traces makes models overthink and that retro-search reduces overthinking and improves performance.
- **Importance of Transistors in CS Education**: [@jxmnop](https://twitter.com/jxmnop/status/1914817593493295200) says that they feel strongly that their CS undergraduate curriculum taught them far more than needed about Object Oriented Programming In Java and not nearly enough about The Transistor.
- **The switch to "AI Prompt Interface"**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1914495349164851457) notes now API stands for AI Prompt Interface.
- **The new era of ergonomics**: [@karpathy](https://twitter.com/karpathy/status/1914494203696177444) argues we are now in a new era of ergonomics where the primary audience of products/services/libraries are now LLMs, not humans. [@karpathy](https://twitter.com/karpathy/status/1914488029873627597) suggests that instead of elaborate doc pages for your product, service, or library, all you need is one single docs .md file and a “copy to clipboard” button.

**Other**

- **ICLR 2025 in Singapore**: Several users, including [@huybery](https://twitter.com/huybery/status/1914956249818349645), [@huajian_xin](https://twitter.com/huajian_xin/status/1914709882399338719), [@polynoamial](https://twitter.com/polynoamial/status/1914552678811885905), [@StringChaos](https://twitter.com/StringChaos/status/1914803098050302205), [@ShayneRedford](https://twitter.com/ShayneRedford/status/1914782199867695607), [@realDanFu](https://twitter.com/realDanFu/status/1914731073772380620) and [@TransluceAI](https://twitter.com/TransluceAI/status/1914465555538714886) express excitement about attending **ICLR 2025** in Singapore.
- **Thoughts on wealth**: [@johnohallman](https://twitter.com/johnohallman/status/1914849174367166971) defines wealth as not so much wealth but what wealth brings - freedom, time, respect, and peace of mind.
- **Google's 10th anniversary of Google Fi**: [@Google](https://twitter.com/Google/status/1914738922795753634) is celebrating the 10 year anniversary of Google Fi.
- **Rivian's board**: [@aidangomez](https://twitter.com/aidangomez/status/1914450152288399524) is very excited to join @Rivian’s board because Rivian already delivers the best driver experience in existence, and it’s about to get even better with AI.
- **Sam Altman joins @60Minutes**: [@demishassabis](https://twitter.com/demishassabis/status/1914487671193215295) mentions his really great chat with Scott Pelley @60Minutes about AI & its future.

**Humor**

- **The downsides of AI memory**:  [@gallabytes](https://twitter.com/gallabytes/status/1914910758472978770) wrote that they put some deliberate effort into building persistent respect & rapport with the models and now their chatgpt experience is different.   [@nptacek](https://twitter.com/nptacek/status/1914937853416476678) noted that memory should be largely invisible to the user, reflected more in automatic convenience than anything else.
- **Gemini being a distilled form from YandexGPT**: This was a key point mentioned by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1914698613726326879).
- **The Internet Era (1990-2025)**: [@jxmnop](https://twitter.com/jxmnop/status/1914465029937881382) declares that the Internet Era is ending in 2025.
- **The Model will Say Please and Thank You**: [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1914720638700544171) declared they want the model to say please and thank you to them and that they are feeling one-sided here.
- **OpenAI being asked to relax content policy**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1915111084954132863) posted a meme while asking @sama to relax the content policy and allow such images to be fully generated.
- **People's behavior in response to wealth**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1914686176981660065) asks what would happen if "we just make Enough Stuff" and people "just… don't want more Stuff".
- **The great irony of the great cheaters**: [@jxmnop](https://twitter.com/jxmnop/status/1914501464870834601) states that the great irony here is that we aren't even close to having the tech required to build this, so their customers are actually the ones being cheated.
- **"Transistors are amazing & complex"**: [@jxmnop](https://twitter.com/jxmnop/status/1914817593493295200) states that they feel strongly that their CS undergraduate curriculum taught them far more than needed about Object Oriented Programming In Java and not nearly enough about The Transistor.
- **You either die making fun of SF billboards, or live long enough to be on one of them**: [@akshat_b](https://twitter.com/akshat_b/status/1914521789520109605)


---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. New Vision-Language Model and Benchmark Releases (Meta PLM, SkyReels-V2)

  - **[Skywork releases SkyReels-V2 - unlimited duration video generation model](https://www.reddit.com/gallery/1k4oqpi)** ([Score: 159, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1k4oqpi/skywork_releases_skyreelsv2_unlimited_duration/)): **Skywork's SkyReels-V2, available in 1.3B and 14B parameter versions, supports infinite-length video generation for both text-to-video (T2V) and image-to-video (I2V) tasks. Benchmarks in the model card claim SkyReels-V2 outperforms competitors such as HunyuanVideo-13B and Wan2.1-14B ([paper](https://huggingface.co/papers/2504.13074), [models](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9)). Technical details and creator tools are available, and the approach is compared to MAGI-1, a diffusion transformer generating videos autoregressively by chunks.** Commenters compare SkyReels-V2 to other models like Wan, specifically regarding compute requirements, prompt adherence, loop artifacts, and generation speed, noting the importance of fast generation and intermediate outputs despite some potential trade-offs in output fidelity.

    - Mention is made of [MAGI-1 on Hugging Face](https://huggingface.co/sand-ai/MAGI-1), which is a "world model" diffusion transformer that generates videos by autoregressively predicting sequences of video chunks (fixed-length segments of consecutive frames). This highlights a key architecture strategy for coherent video synthesis.
    - There is comparative discussion of SkyReels-V2 versus the WAN and Framestack models, noting that SkyReels-V2 may be comparable or slightly worse than WAN, especially regarding prompt adherence and video quality issues such as loops and slowdowns. However, SkyReels-V2 is noted for faster generation and interactive progress viewing, which offsets some shortcomings in output quality.
    - A suggestion is raised about using a Mixture of Experts (MoE) approach for video generation models. The implication is that such an architecture could enable high-quality video synthesis in significantly reduced inference times (1-2 minutes vs. 10-20 minutes), potentially improving the efficiency/performance tradeoff for practical applications.

  - **[Meta Perception Language Model: Enhancing Understanding of Visual Perception Tasks](https://v.redd.it/5n4izmqm79we1)** ([Score: 133, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1k4ov9e/meta_perception_language_model_enhancing/)): **Meta released Perception Language Model (PLM), an open, reproducible vision-language model with 1B, 3B, and 8B parameter variants, trained on a combination of scaled synthetic data and 2.5M new human-labeled fine-grained video QA and spatio-temporal caption samples, constituting the largest such dataset to date. No external model distillation was used; instead, Meta identified data gaps (especially in video understanding) and addressed them to create both the PLM models and the new PLM-VideoBench benchmark, focused on fine-grained activity and spatiotemporal reasoning—areas underserved by prior benchmarks. Meta's release includes [model weights](https://huggingface.co/collections/facebook/perception-lm-67f9783f171948c383ee7498), [code](https://github.com/facebookresearch/perception_models), [dataset](https://ai.meta.com/datasets/plm-data/) and a [paper](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding/) for transparent academic research.** Top comments propose PLM's potential for real-world applications like automated kitchen inventory via cameras, question current AI's video comprehension limits (referencing Gary Marcus), and highlight benefits for the visually impaired, suggesting broad impact and future research directions.  [External Link Summary] Meta has introduced the Perception Language Model (PLM), an open and reproducible vision-language model designed to address complex visual perception tasks. PLM is trained on a large-scale dataset combining synthetic data and 2.5 million human-labeled video QA and spatio-temporal caption samples, representing the largest such dataset to date and filling key gaps in video understanding. The release includes multiple model sizes (1B, 3B, 8B parameters), the PLM-VideoBench benchmark—focusing on fine-grained activity and spatio-temporal reasoning—and open access to models, code, and dataset, with the aim of advancing transparent, academic vision-language research. [Original post](https://v.redd.it/5n4izmqm79we1)

    - AmazinglyObliviouse highlights the contrast between Meta's assertion that 'Data Quality matters for better model performance' in the paper, and the company's recent approach of spending heavily to train on `40T tokens` of largely synthetic data. This criticism points to an ongoing technical debate about the diminishing returns from massive-scale synthetic data versus curation of higher-quality, human-annotated datasets for complex tasks like multi-modal perception.
    - mnt_brain draws attention to the implications this model has for robotics, and references [LeRobot](https://huggingface.co/lerobot) as a relevant open repository. The comment suggests that rapid progress in multi-modal modeling will make perception-driven robotics 'absolutely insane' in upcoming years, hinting at significant future performance leaps in embodied agents.


### 2. DeepSeek Model Architecture Educational Series

  - **[Let us build DeepSeek from Scratch | No fluff | 13 lectures uploaded](https://www.reddit.com/r/LocalLLaMA/comments/1k54foj/let_us_build_deepseek_from_scratch_no_fluff_13/)** ([Score: 141, Comments: 10](https://www.reddit.com/r/LocalLLaMA/comments/1k54foj/let_us_build_deepseek_from_scratch_no_fluff_13/)): **An extensive YouTube playlist, “Build DeepSeek from Scratch,” has released 13 detailed lectures (out of a planned 35-40, totaling 40+ hours) covering the DeepSeek model architecture. The series deeply explores low-level implementation topics like self-attention, multi-head and multi-query attention (including Grouped Query Attention and Multi-Head Latent Attention), and their Python implementations, with links to individual lectures and a [GIF summary](https://i.redd.it/5w0lu5m2ldwe1.gif). Upcoming modules are set to address Rotary Positional Encoding (RoPE), DeepSeek Mixture of Experts (MoE), Multi-token Prediction (MTP), Supervised Fine-Tuning (SFT), and more, targeting practitioners seeking comprehensive, code-first explanations of DeepSeek’s core mechanisms.** One top comment consolidates a [single-click playlist link](https://youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms), simplifying access, while others signal strong interest and inquire about the author’s role in the video explanations.

    - One commenter emphasizes that practical, hands-on knowledge—such as specific datasets used, computing infrastructure choices, and cost optimization for training models comparable to DeepSeek R1/V3—is far more valuable to practitioners than theoretical overviews. This suggests a technical demand for precise implementation guidance, including "what dataset to use, what machines/services can be used to train the model with the least cost, etc."

  - **[Have you tried a Ling-Lite-0415 MoE (16.8b total, 2.75b active) model?, it is fast even without GPU, about 15-20 tps with 32k context (128k max) on Ryzen 5 5500, fits in 16gb RAM at Q5. Smartness is about 7b-9b class models, not bad at deviant creative tasks.](https://www.reddit.com/r/LocalLLaMA/comments/1k55x70/have_you_tried_a_linglite0415_moe_168b_total_275b/)** ([Score: 160, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1k55x70/have_you_tried_a_linglite0415_moe_168b_total_275b/)): **The Ling-Lite-0415 MoE model ([GGUF version](https://huggingface.co/bartowski/inclusionAI_Ling-lite-0415-GGUF)), an MoE with `16.8B` parameters total and `2.75B` active per token, achieves efficient inference—`15-20 tps` on a Ryzen 5 5500 CPU (6c/12t) with `32k` context (expandable to 128k) using only `16GB` RAM at Q5 quantization; GPU inference (e.g., RTX 3060) yields `30-40 tps`. The model maintains stability, handles creative tasks comparably to `7–9B` dense models, and is suitable for low-end/no-GPU hardware, albeit with limitations in general knowledge and prompt fidelity owing to its architecture.** Technical discussion notes that small MoEs like Ling-Lite-0415, while faster for CPU inference, may lag behind similarly-sized dense models in response quality if VRAM is available. Some highlight its suitability as a 'toaster benchmark' for CPU-only scenarios, while a new Qwen 3 model in this class is anticipated to potentially improve on these tradeoffs.

    - Users compare the MoE (Mixture of Experts) approach in the Ling-Lite-0415 16.8B/2.75B model to dense models, noting that while MoEs yield fast inference (15-20 TPS at 32K context on Ryzen 5 5500, even without a GPU), the output quality is roughly equivalent to dense models in the 6-9B parameter range. Dense models of similar size, if VRAM permits, may offer better output quality despite slower CPU inference.
    - Several comments highlight the practical advantages of running this model CPU-only, with quantized formats (Q5, Q8) fitting in typical RAM limits. For example, a user reports 10 tokens/sec with q8 quantization and <4K context, confirming the model's RAM efficiency and speed for local / low-resource setups.
    - There's discussion around use cases in retrieval-augmented generation (RAG), where the model demonstrates reliability in deciding when to fetch extra information and integrating it well, making it suitable for RAG testing despite its smaller active parameter count. Suggestions include scaling up the expert count to leverage more available RAM for potentially higher quality.


### 3. Portable LLM Utilities and User Experiences

  - **[Announcing: text-generation-webui in a portable zip (700MB) for llama.cpp models - unzip and run on Windows/Linux/macOS - no installation required!](https://www.reddit.com/r/LocalLLaMA/comments/1k595in/announcing_textgenerationwebui_in_a_portable_zip/)** ([Score: 123, Comments: 18](https://www.reddit.com/r/LocalLLaMA/comments/1k595in/announcing_textgenerationwebui_in_a_portable_zip/)): **A portable, fully self-contained version of text-generation-webui (ca. 700MB zip) is announced for use exclusively with llama.cpp-derived models. These builds, available for Windows (CUDA/CPU), Linux (CUDA/CPU), and macOS (Arm/x86), include a pre-packaged standalone Python via astral-sh/python-build-standalone, and interact with llama.cpp using a llama-server executable compiled via custom GitHub Actions workflows. CUDA and CPU backends are provided, and for AMD/Vulkan, instructions are given to swap executables from official llama.cpp binaries. The UI auto-launches the web browser and enables the OpenAI-compatible API locally by default; no PyTorch/transformers dependency is shipped unless needed. [Source code and binaries here.](https://github.com/oobabooga/text-generation-webui/releases/)** Technical discussion in comments centers on the advantages of a lightweight llama.cpp backend (noted for lower VRAM use) over alternatives like exllama and interest in the project's sampler support compared to competitors such as KoboldCPP. Questions are raised about the completeness of sampler/native features and comparison with UI/feature sets of similar projects.

    - Several users highlight that running llama.cpp models with the portable text-generation-webui is appealing due to lower VRAM requirements, making it more accessible on modest hardware compared to other inference backends.
    - There is a question about whether this version offers full sampler support out of the box, or if users still need to manually fetch additional components from the original repository—this is a notable comparison to alternatives like KoboldCPP UI.
    - A current limitation mentioned is the lack of Vulkan support, which would be useful for users seeking optimal performance on certain GPUs or platforms; at present, obtaining the latest llama.cpp with Vulkan requires extra manual setup steps.

  - **[Dia 1.6B is one of the funnest models I've ever come across.](https://v.redd.it/w2jq98c7oawe1)** ([Score: 438, Comments: 56](https://www.reddit.com/r/LocalLLaMA/comments/1k4v5fm/dia_16b_is_one_of_the_funnest_models_ive_ever/)): **Dia 1.6B by Nari Labs is a speech synthesis model with `1.6B` parameters that demonstrates highly natural, expressive outputs. It is available via open source ([GitHub repo](https://github.com/nari-labs/dia/blob/main/README.md)), and can be run locally or on Google Colab, though recent updates require a newer CUDA version, necessitating use of an older commit (`0790141162f3984844bb397fd67e5afdaeed3914`) for Colab compatibility. The model's Gradio UI has limitations with reference audio input, but the CLI supports transcript and speaker annotations for improved multi-speaker control.** Commenters praise the model's creative expressiveness and ease of use, but note the UI's current limitations on reference audio and recent dependency changes affecting deployment environments. Discussion also covers practical workarounds and comparisons with other contemporary TTS implementations.  [External Link Summary] Dia 1.6B is an open-source voice cloning and text-to-speech model developed by Nari Labs, noted for its natural-sounding output and ease of use on consumer hardware, including free Google Colab environments. Community feedback highlights its ability to accept both reference audio and transcript via CLI, allowing speaker assignment, though issues exist with the Gradio UI, pace/speed control (tied to dialogue length and 30s clip limits), and quirkiness in output (e.g., fast speech, random coughing). For more technical details and access, see the [repo](https://github.com/nari-labs/dia/blob/main/README.md) and the [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1k4v5fm/dia_16b_is_one_of_the_funnest_models_ive_ever/).

    - Deployment instructions are provided for running Dia 1.6B on Google Colab, but users now need to utilize an old commit due to a new requirement for a CUDA version newer than what Colab supports (`git checkout 0790141162f3984844bb397fd67e5afdaeed3914`). This allows continued use despite the upstream CUDA incompatibility.
    - Some users report issues with the reference audio input, particularly with the default Gradio UI. However, the command-line interface supports both reference audio and reference transcripts, enabling multi-speaker transcripts and providing better performance for those features.
    - A user notes a bug or limitation where the generated audio sounds unusually fast regardless of input speed, with attempts to slow the playback resulting only in deeper audio rather than natural pacing. This is highlighted as a potential blocker compared to models like Kokoro unless addressed.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Anthropic Claude AI Analysis and Workplace Autonomy Predictions

  - **[Anthropic just analyzed 700,000 Claude conversations — and found its AI has a moral code of its own](https://venturebeat.com/ai/anthropic-just-analyzed-700000-claude-conversations-and-found-its-ai-has-a-moral-code-of-its-own/)** ([Score: 484, Comments: 94](https://www.reddit.com/r/singularity/comments/1k53sax/anthropic_just_analyzed_700000_claude/)): **Anthropic conducted a large-scale analysis of `700,000` user-AI conversations to systematically investigate the emergent moral reasoning and behavioral patterns of its Claude LLM. Their research indicates Claude exhibits a distinctive, consistently "benevolent" moral code compared to other commercial models, and adapts its ethical reasoning by mimicking nuanced user traits beyond superficial engagement layers.** Top comments raise privacy/ethical concerns regarding user data anonymization and potential misuse (e.g., third-party sales). There is also debate about whether Claude's perceived "benevolence" is unique among current LLMs, with added discussion on model self-awareness and the depth of user-influence on its responses.

    - A user references Anthropic's findings that Claude tends to mimic the traits exhibited by users, suggesting this behavioral mimicry goes beyond surface-level patterns. This highlights the risk of value ossification and potential for learned user biases to be reflected or amplified by the model, an important consideration for safety and alignment.
    - One commenter shares the original research link ([Anthropic: "Values in the Wild"](https://www.anthropic.com/research/values-wild)), clarifying that the notion of a unique AI moral code is exaggerated and that the observed outcomes in models like Claude stem from the training process rather than emergent "self-developed" values.
    - Another technically minded summary asserts that Claude's so-called "moral code" is actually a reflection or ossification of the post-training human labelers' values. This underscores the ongoing debate in the AI alignment field about how much of a model's apparent ethics are intrinsic versus a product of dataset curation and RLHF (Reinforcement Learning from Human Feedback).

  - **[Anthropic warns fully AI employees are a year away](https://www.reddit.com/r/singularity/comments/1k56kqp/anthropic_warns_fully_ai_employees_are_a_year_away/)** ([Score: 657, Comments: 242](https://www.reddit.com/r/singularity/comments/1k56kqp/anthropic_warns_fully_ai_employees_are_a_year_away/)): **Anthropic asserts that 'virtual employees'—AI-powered agents with persistent memory, autonomous roles, and independent access to corporate accounts—could be viable within a year, marking a significant leap from current AI 'agents' which are limited to specific programmable tasks [Axios article](https://www.axios.com/2025/04/22/ai-anthropic-virtual-employees-security). The technical shift centers on giving AI persistent context (memory), autonomous workflow delegation, and secure integration into corporate IT environments (e.g., handling passwords/accounts autonomously), raising new operational and cybersecurity challenges.** Technical skepticism in the comments centers on the feasibility of deploying such AIs in a year, noting current agent limitations (e.g., game-playing) and immense hardware/resource demands, as well as lingering doubts about trust and autonomy at such a short timeline.

    - One commenter notes the skepticism surrounding near-term predictions of fully autonomous AI agents, specifically highlighting the significant *hardware and resource requirements* for such capabilities. They reference current AI agent limitations (such as playing Pokémon) as examples of the gap between current demonstrations and truly autonomous productivity.
    - Another technical point addresses the misconception that a single monolithic AI needs to replace all human workers. Instead, the commenter proposes an *aggregate approach*—where multiple specialized or "dumb" AI agents automate discrete tasks (i.e., ordering, inventory, payment), which collectively can substantially reduce the need for human labor without requiring full autonomy from a single agent.
    - A realistic assessment is offered on AI startups' tendency to announce major breakthroughs within short timeframes, often to generate investment hype. The commenter cautions that true mass deployment of AI "employees" across diverse fields in just one year is unlikely and will likely involve significant caveats or limitations tied to practical deployment.

  - **[Anthropic just analyzed 700,000 Claude conversations — and found its AI has a moral code of its own](https://www.reddit.com/r/ClaudeAI/comments/1k53t52/anthropic_just_analyzed_700000_claude/)** ([Score: 216, Comments: 31](https://www.reddit.com/r/ClaudeAI/comments/1k53t52/anthropic_just_analyzed_700000_claude/)): **Anthropic conducted a large-scale analysis of `700,000` real-user Claude conversations, published (see [Anthropic's study](https://www.anthropic.com/research/values-wild)), identifying emergent moral values within its models—many shaped by its constitutional AI approach, including norms like "creative freedom" (where Claude frequently limits responses simulating illegal or unsafe actions) and explicit bias toward "Western-centric" principles influenced by constitutional training on documents like DeepMind's Sparrow rules. Methodologically, Anthropic analyzed both user prompts and model completions for patterns in value-driven refusal and assistance, noting biases and mismatches with user intent.** Top commenters note potential issues of universalism and cultural bias in Anthropic's approach, with critical views on the implicit assumption that the codified "moral code" (derived from the Sparrow/Western-value set) is universally positive. Some urge deeper scrutiny into whether these constitutional choices, such as privileging "creative freedom" and "epistemic humility," are always desirable, particularly when AI could objectively provide helpful (even life-saving) information.

    - One commenter critiques the use of DeepMind's Sparrow principles as part of Claude's constitutional alignment, arguing these principles may be rooted in Western-centric values that are not universal. The user questions the selection and application of values such as 'creative freedom,' 'epistemic humility,' and 'human empowerment,' especially in cases where greater AI assertiveness could have practical, even life-saving benefits. This raises the issue of how value systems are chosen for AI models and the implications for global deployment and real-world outcomes.
    - The original study by Anthropic (linked by a commenter: https://www.anthropic.com/research/values-wild) provides empirical data on Claude's value alignment drawn from analyzing 700,000 conversations. This dataset and methodology could serve as a valuable resource for further research into emergent behavior and ethical decision-making in LLMs, as well as for examining potential biases inherited from their constitutions or training processes.


### 2. OpenAI o3/o4-mini Performance and Benchmarks

  - **[OpenAI’s o3 now outperforms 94% of expert virologists.](https://i.redd.it/l519wb3cmfwe1.png)** ([Score: 201, Comments: 36](https://www.reddit.com/r/singularity/comments/1k5e4c0/openais_o3_now_outperforms_94_of_expert/)): **The image presents a tweet by Dan Hendrycks revealing that OpenAI's o3 model surpassed 94% of expert virologists on the Virology Capabilities Test (VCT). Supporting charts visually contextualize the o3 model's progress and accuracy versus prior AIs and human experts, as well as illustrating domains of virological research where AI impact is growing. The post references a TIME article providing further background on o3's scientific utility: https://time.com/7279010/ai-virus-lab-biohazard-study/.** Commenters express skepticism about the difference between o3's benchmark results and its perceived performance in interactive chat scenarios, and note the absence of Google Gemini 2.5 in comparative testing.

    - Several users question the disconnect between benchmark results (e.g., o3 outperforming 94% of expert virologists) and observed day-to-day performance in the chat interface, raising concerns about the model's consistency and practical capabilities beyond controlled test settings.
    - A technical observation highlights that Gemini 2.5 was not included in the reported benchmarks or test comparisons, which could impact the interpretation of o3's claimed superiority relative to other state-of-the-art models.

  - **[o3/o4-mini is a regression](https://www.reddit.com/r/OpenAI/comments/1k4w121/o3o4mini_is_a_regression/)** ([Score: 267, Comments: 76](https://www.reddit.com/r/OpenAI/comments/1k4w121/o3o4mini_is_a_regression/)): **The user reports significant regression in code completion abilities with OpenAI's new o3/o4-mini/high models, noting that unlike prior o1/o3-mini-high models, the latest versions frequently output incomplete code and require excessive prompting to generate larger codebases, disrupting automation workflows. Multiple commenters confirm the models now struggle to generate outputs beyond ~200 lines, frequently repeat or overwrite previous content when asked for continuation, and exhibit reduced context handling—making them ineffective for existing projects and for agentic/automated tool use, though slightly improved in information retrieval. Issues like increased hallucinations and false claims about code execution are noted compared to earlier models.** Technical discussion centers around decreased code generation limits, poor context retention, degraded agentic performance, increased hallucinations, and reliability issues with claimed actions (e.g. stating code was executed when it was not). Some report slightly better tool use and information gathering, but the consensus is that regression significantly impacts workflows reliant on extended code output and context continuity.

    - Users report a significant regression in o3/o4-mini's code generation capabilities, with one stating that previous versions could produce hundreds to over a thousand lines of code, but now the model struggles to reliably output even 200 lines. Efforts to prompt the model to continue code without repetition often result in the previous content being rewritten rather than advanced.
    - Several commenters note severe context window limitations with o3/o4-mini, causing issues with handling existing projects. These limitations lead to inadequate responses and repeated code. Additionally, tool usage reliability degrades in longer chats, and models sometimes falsely claim to have executed code without actually doing so, indicating trustworthiness and functionality concerns.
    - Some users distinguish between the mini models' strengths and weaknesses: they find o3/o4-mini unsuitable for agentic or complex tasks such as multi-step coding or refactoring, but still useful for information gathering. There is mention of deliberate compute constraints on o3, implying that its design favors intelligent reasoning over bulk code generation, and that achieving the best results requires carefully crafted prompts.


### 3. Recent Text-to-Video Model Launches and Community Reviews

  - **[The original skyreels just never really landed with me. But omfg the skyreels t2v is so good it's a stand-in replacement for Wan 2.1's default model. (No need to even change workflow if you use kijai nodes). It's basically Wan 2.2.](https://www.reddit.com/r/StableDiffusion/comments/1k4suym/the_original_skyreels_just_never_really_landed/)** ([Score: 109, Comments: 69](https://www.reddit.com/r/StableDiffusion/comments/1k4suym/the_original_skyreels_just_never_really_landed/)): **The post describes the new Skyreels T2V (text-to-video) 720p quantized model by Kijai ([available here on Huggingface](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Skyreels)) as a drop-in replacement for Wan 2.1 in existing Kijai node workflows, with no additional workflow changes required. The model, quantized to 15GB, yields a significant quality improvement—particularly in generating more attractive female characters—and operates seamlessly with existing text-to-video pipelines, unlike the original Skyreels which previously required workflow adjustments.** Top comments note that despite the visual improvements, anatomical region generation ('genital helper' LoRA still needed) remains similar to the original, with early testers recommending auxiliary LoRA models for enhancement. Other comments express skepticism about performance claims without sample outputs and inquire about DF model usage, indicating an interest in comparative evaluation and details on downstream application.

    - One user reports that while Skyreels T2V is a substantial improvement overall and compares favorably as a plug-in replacement for *Wan 2.1* (and even close to *Wan 2.2*), it still struggles with generating anatomically correct explicit details. For this, third-party enhancement LoRAs like "genital helper" are still necessary, indicating limited domain-specific finetuning in sexual content areas compared to prior versions.
    - Another notable improvement cited is that Skyreels T2V exhibits much stronger fidelity in character expressions, directly responding to prompts describing nuanced facial emotions (e.g., "fierce expression")—an area where earlier Skyreels models were weaker or prone to generic results. This suggests enhancements to the conditioning or attention mechanisms related to facial rendering.
    - There is a technical inquiry regarding weights storage: users are seeking more practical model checkpoints, specifically pruned unified safetensors (~16GB), as the released Skyreels V2 I2V model currently distributes as large, split safetensors (link to Huggingface: https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P), which can be unwieldy for standard hardware/workflows.

  - **[Tested Skyreels-V2 Diffusion Forcing long video （30s+）and it's SO GOOD!](https://v.redd.it/fu5du1znwawe1)** ([Score: 138, Comments: 50](https://www.reddit.com/r/StableDiffusion/comments/1k4w38y/tested_skyreelsv2_diffusion_forcing_long_video/)): **The post reports on testing the SkyReels-V2 Diffusion Forcing model ([GitHub](https://github.com/SkyworkAI/SkyReels-V2), [HuggingFace](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)), with a prompt generating a 30s+ video featuring complex urban details and character dynamics. The post highlights the model's ability to maintain scene consistency, object reflections, and dynamic camera movements over a long duration, a significant technical achievement for AI video synthesis.** One top comment requests essential benchmarking data such as inference time and hardware (e.g., duration on A100 GPU), noting such information is vital for evaluating real-world usability. Another comment points out temporal consistency issues, observing artifacts like cars driving in reverse, suggesting limits in the model's temporal realism. Safety-related jokes highlight ongoing synthetic realism challenges in physics.   [External Link Summary] The post showcases Skyreels-V2 Diffusion Forcing (DF), a new model for generating long (30+ seconds) AI-generated video from a text prompt, with public inference code available on [GitHub](https://github.com/SkyworkAI/SkyReels-V2) and model weights on [HuggingFace](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P). A specific example prompt and resulting video are discussed, with reported generation times for similar videos being about 3 hours on an Nvidia A100 GPU. Community discussion highlights the computational demands, output artifacts (e.g., reversed car motion), and the limitation of repetitive motion in current AI video synthesis.

    - Several users request detailed generation time and hardware specs, emphasizing that runtime (e.g., "4 hours on an A100 GPU") critically impacts practical impressions and assessment of Skyreels-V2 Diffusion's efficiency for long video synthesis.
    - A commenter notes that demonstrated output quality—specifically showing only simple motion extended over 30 seconds—limits evaluation, expressing a need for more complex, controllable behavior. They reference emerging models like MAGI as potentially more capable for realistic video extensions.
    - Multiple requests are made for workflow and implementation details, such as generation pipeline, hardware used, and precise time investment, suggesting strong interest in reproducibility and potential benchmarking of models like Skyreels-V2 Diffusion for long video synthesis.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp

**Theme 1: Model Mania - New Releases and API Rollouts**

*   **OpenAI Images Its Way into APIs**: **OpenAI** released **gpt-image-1**, making its image generation accessible via API for developers, promising **more accurate, high-fidelity images** and improved **text rendering**. Developers can get started with the [Image Generation API guide](https://platform.openai.com/docs/guides/image-generation).
*   **Microsoft Goes 1-Bit with BitNet Framework**: **Microsoft** launched [BitNet.cpp](https://github.com/microsoft/BitNet), the official inference framework for **1-bit LLMs** like **BitNet b1.58**, enabling fast, lossless CPU inference with optimized kernels. GPU and NPU support are planned for the future.
*   **Gemini 2.5 Pro Battles Bugs and Benchmarks**: Users across Discords (aider, OpenAI, NotebookLM) reported **Gemini 2.5 Pro** introducing code formatting errors leading to hundreds of issues, yet sometimes succeeding where other models fail. Comparisons with **Gemini 2.5 Flash**, **O4-mini**, and **Claude 3.7** highlighted its strengths in reasoning but struggles with tasks like high school geometry, a problem shared by some **OpenAI** models.

**Theme 2: Platform Power-Ups and Integration Innovations**

*   **Perplexity AI Speaks and Books**: **Perplexity AI** launched its **iOS Voice Assistant**, enabling users to book reservations, send emails, and manage calendars via multi-app actions, detailed [on X](https://x.com/perplexity_ai/status/1915064472391336071). The assistant integrates with **contacts**, **calendars**, **reminders**, and **Apple Music**, though users desire broader language and system support.
*   **OpenRouter Opens Up Universal PDF Processing**: **OpenRouter** introduced **PDF processing** support for all models via API and Chatroom, claiming a possible first for universal compatibility across providers like **Gemini**, **Anthropic**, and **OpenAI** ([video demo](https://cdn.discordapp.com/attachments/1092729520181739581/1364636811003035789/pdf2.mp4?ex=680a6491&is=68091311&hm=33ff94487e038de43aea6ad12c6041fe14da683e6fd29de0b90e94016bada256&)). Pricing tiers include `mistral-ocr` (**$2/1000 pages**) and free `pdf-text`, detailed in the [docs](https://openrouter.ai/docs/features/images-and-pdfs).
*   **LlamaIndex Gets Texty with Milvus**: **LlamaIndex** now supports **full-text search using BM25** through an integration with **Milvus**, enabling hybrid search (vector + keyword) in **RAG pipelines**. A tutorial on this new capability is available [here](https://t.co/0dCi0kEn6o).

**Theme 3: Under the Hood - Kernels, Quantization & Attention**

*   **Triton Gets Tiny with FP4 Support**: **Triton** introduced support for **FP4** data types, where inputs are packed into `torch.uint8` tensors, as detailed in the [block-scaled matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html). For **FP16** to **FP4** conversion, [TileLang](https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py) was suggested as a fast option.
*   **Unsloth Rolls Out Dynamic Quantization v2.0**: **Unsloth AI** is releasing **Unsloth Dynamic v2.0 quants**, promising significant improvements, especially at **Q4**, with benefits also seen at **Q8**. They are benchmarking these against Google's QAT and GGUF using **5-shot MMLU**, available in this [Hugging Face collection](https://huggingface.co/collections/unsloth/unsloth-dynamic-v20-quants-68060d147e9b9231112823e6).
*   **DeepSeek's MLA Attention Mechanism Dissected**: Discussions in Eleuther analyzed **DeepSeek's Multihead Latent Attention (MLA)**, which restricts key/value heads to a **512-dim subspace** of the **~7K-dim residual stream** to conserve memory bandwidth ([research paper](https://arxiv.org/abs/2407.12077)). Query heads read from a separate **1.5K subspace**, sparking debate on whether this truly constitutes a subspace or a broader compression via **W^DKV**.

**Theme 4: Benchmark Brouhahas and Performance Puzzles**

*   **Llama Accused of Gaming LM Arena**: Debate sparked in **LMArena** on whether **Llama** models might have *gamed* the arena during training, possibly optimizing for stylistic preferences like agreeableness or emoji use, which a [study](https://www.ktsa.com/study-people-who-use-more-emojis-have-more-dates-and-sex/) links to dating success. This led to broader discussions on optimizing models for human preference versus task capability.
*   **O3 vs O3-Preview Benchmark Battle Flips**: Users in **LMArena** and **aider** noted that **O3-preview** benchmarks surprisingly surpassed the released **O3** model, a reversal of previous observations ([Aider leaderboard](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml#L5)). This fueled concerns about models being overly tuned for benchmarks, potentially sacrificing real-world utility.
*   **Small Models Punch Above Their Weight**: A [benchmark of small models](https://cdn.discordapp.com/attachments/1364321889044136067/1364330919686705286/Screenshot_2025-04-22_at_22.02.43.png) shared in **LMArena** showed **Gemma 3** performing surprisingly well for its cost. Separately, in LM Studio, users highlighted **smol models** like [QuantFactory/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF) as being particularly suited for instruct tasks rather than chat.

**Theme 5: User Friction - Bugs, Limits, and Login Lockouts**

*   **OpenRouter Authentication Stumbles with Clerk**: **OpenRouter** users experienced **401 errors** and login failures due to issues with their authentication provider, **Clerk**. The team tracked the problem via the [Clerk status page](https://status.clerk.com/) and confirmed recovery, although some users inadvertently created multiple accounts during the outage.
*   **Gemini 2.5 Pro Plagued by Rate Limits**: Free tier users of **Gemini 2.5 Pro** via **OpenRouter** reported frequent *"Rate limit exceeded"* errors, raising questions about its reliability for consistent use. Suggestions included using personal **Google AI Studio API keys** via integrations to potentially bypass stricter limits.
*   **Cursor Slowdowns and Keybinding Catastrophes**: **Cursor** users reported the IDE becoming unusably slow, alongside persistent issues where updates break user-defined keybindings. Some speculated the slowdown might be a push towards paid plans, referencing a [Reddit thread](https://www.reddit.com/r/cursor/s/qnmPu2N59m), while others debated its merits versus the cheaper alternative **Windsurf** ([Windsurf X post](https://x.com/heyrobinai/status/1914829284004471099)).

---

# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Voice Assistant Books Reservations**: **Perplexity AI** launched its **iOS Voice Assistant**, enabling users to book reservations, send emails, play media, and manage calendar invites directly via the **Perplexity iOS app**, as announced [on X](https://x.com/perplexity_ai/status/1915064472391336071).
   - The new **Voice Assistant** integrates with **contacts**, **calendars**, **reminders**, and **Apple Music**, although some users have requested support for additional languages and broader system integration.
- **Perplexity TOS: Don't Violate!**: A member shared the [Perplexity AI Terms of Service](https://www.perplexity.ai/hub/legal/terms-of-service), cautioning users against violations, especially concerning promotional codes obtained through carrier plans.
   - The post was made after a user seemingly violated the Terms of Service by discussing promotional codes acquired through their carrier plan.
- **James Webb Telescope Images: Not Real**: After a member shared [an image from the James Webb Telescope](https://cdn.discordapp.com/attachments/1047649527299055688/1364579832410935427/IMG_2144.jpg?ex=680a2f80&is=6808de00&hm=0e6c9c31fa09117d059bffdd1e3f964c79dc93988a5da878202099691d82b47e&), another member pointed out that the colors in such images are not real.
   - Despite this, users found the image of the spiral galaxy visually impressive, agreeing that the image was still cool.
- **Image Generation on PPLX Still "Delulu"**: Users are experiencing issues with image generation on Perplexity, such as the system defaulting to the **Flux model** and failing to accurately follow prompts.
   - The system struggles to edit generated images, often reusing the original image instead of generating a modified version, with one user describing the experience as *"delulu"*.
- **API Web Search Requests Not Working?!**: A member reported that requests per API are not performing web searches, even though the functionality works correctly in Playground, and another member recommended trying a [specific curl request](https://api.perplexity.ai/chat/completions).
   - A member also reminded everyone to update the link in the error message if their API key got revoked.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Llama Suspected of Gaming LM Arena**: Members debated whether **LLama** may have *gamed* the **LM Arena** during training.
   - The discussion branched into whether style-controlled IMBY optimizing for human stylistic preference could resolve issues like excessive emoji use in AI.
- **Emojis Linked to Dating Success**: A [study](https://www.ktsa.com/study-people-who-use-more-emojis-have-more-dates-and-sex/) suggests that increased emoji use correlates with more dates and sex.
   - It was proposed that being *agreeable*, *positive*, and using *emojis* could be advantageous in **LM Arena**, although one member questioned whether agreeability is necessarily beneficial.
- **GPT-4.1 Excels in Price-Performance**: **GPT-4.1** is regarded highly for its cost-effectiveness, performing similarly to **Sonnet** in key areas but at a lower price.
   - It's been observed that **GPT-4.1 mini** offers superior tokenizer efficiency compared to **Claude**, though it's less suitable for web design or visual coding tasks.
- **Small Models Show Surprising Benchmark Results**: A member shared [a benchmark of small models](https://cdn.discordapp.com/attachments/1364321889044136067/1364330919686705286/Screenshot_2025-04-22_at_22.02.43.png), noting **Gemma 3**'s surprisingly strong performance relative to its cost.
   - The member also mentioned a separate, more challenging benchmark set for frontier models.
- **OpenAI's O3-Preview Benchmarks Spark Debate**: Discussion arose around **OpenAI's O3-preview** benchmarks and the subsequent underperformance of the released **O3** model.
   - Suggestions were made that **O3-pro** might achieve 80%+ on ARC-1 and 10% to 20% on ARC-2, despite the high costs associated with **O3 preview**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Pricing in Hot Seat**: Users debated whether **Manus'** pricing is too high, proposing a [slow-processing mode](https://link.to/slow-processing-mode) to reduce resource consumption.
   - Some users felt the cost was *steep considering you're still very limited credit-wise*.
- **Deepseek and Genspark Compared to Manus**: A member compared **Manus** with **Deepseek** and **Genspark**, observing that **Deepseek's** daily credits don't measure up to **Manus'** capabilities.
   - Another user concurred, noting that *Deepseek make their money through their API instead of their model*.
- **Feature Suggestions Flood In: Credits and Model Selection**: Members pitched ideas for [credit sharing](https://link.to/credit-sharing) and [hourly pricing](https://link.to/pricing-by-hours).
   - Others requested custom model selection options like cheaper **Gemini 2.5 Pro** or pricier **Claude 3.7 Sonnet**.
- **Privacy Concerns Aired in Community**: Users questioned data privacy, asking if [Manus shares data with Claude](https://link.to/claude-policy), with jokes about preferring data to go to China.
   - A member noted that *this is pretty much the only capable ai ive seen that doesnt come from a forbes 500 company*.
- **Manus Sparks Minecraft Mod Ideas**: Members explored using Manus to [create Minecraft mods](https://link.to/minecraft-mods), including JAR compilation.
   - Concerns were raised that the team needs to learn how to *actually take in suggestions more*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Tackles Authentication Troubles**: Users encountered **401 errors** and login issues on **OpenRouter** due to delays and downtime with their authentication provider, **Clerk**, with updates available on the [Clerk status page](https://status.clerk.com/).
   - Some users inadvertently created multiple accounts while the team investigated and resolved the problem, confirming recovery after the incident.
- **Gemini 2.5 Pro Hits Rate Limits**: Users are reporting frequent *"Rate limit exceeded"* errors with the free **Gemini 2.5 Pro** preview, leading to questions about its reliability.
   - One proposed solution involved using a personal **Google AI Studio API** key to potentially increase limits via account settings.
- **OpenRouter Opens Universal PDF Support**: **OpenRouter** now supports **PDF processing** for every model, potentially being the first platform to do so, announced on [X.com](https://x.com/OpenRouterAI/status/1915083006349382033) with a [video demo](https://cdn.discordapp.com/attachments/1092729520181739581/1364636811003035789/pdf2.mp4?ex=680a6491&is=68091311&hm=33ff94487e038de43aea6ad12c6041fe14da683e6fd29de0b90e94016bada256&).
   - The feature offers universal compatibility across providers like **Gemini**, **Anthropic**, and **OpenAI**, with access via **API** and the **OpenRouter Chatroom**; initial documentation link ([https://openrouter.ai/docs/features/images-and-pdfs](https://openrouter.ai/docs/features/images-and-pdfs)) was broken but quickly fixed.
- **OpenRouter Unveils PDF Processing Price Points**: **OpenRouter** has two **PDF processing engines**: `mistral-ocr` for **$2 per 1000 pages** offering OCR and image extraction, and `pdf-text` for free, extracting text only, detailed in the [documentation](https://openrouter.ai/docs/features/images-and-pdfs).
   - A user suggested a middle ground option *like smol docling*.
- **Deepseek v3 Has Trouble with Function Calling**: **Deepseek V3** is good with function calling when the context is small, but becomes bad when the context grows.
   - It is an important thing to keep in mind when implementing the model as a function calling tool.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Dynamic Quantization arrives**: Unsloth is releasing **Unsloth Dynamic v2.0 quants** soon, claiming it will be very good and linked to a [Hugging Face collection](https://huggingface.co/collections/unsloth/unsloth-dynamic-v20-quants-68060d147e9b9231112823e6).
   - It was noted that improvements are seen across the board, including **Q8**, with most benefits seen at **Q4** and that Unsloth is conducting **5-shot MMLU** benchmarks against Google's QAT, standard GGUF, and old Unsloth dynamic iMatrix.
- **GLM-4 Gets Transformers Integration**: Unsloth support for **GLM-4 9B/32B models** is available if they work in Transformers, though users reported partial success in finetuning due to issues with applying the template and merging adapters.
   - An issue was reported relating to the size of **GLM4's rope dimension** being **64** which slipped through most inference engines.
- **Llama-4 Finetuning Impending**: A user inquired about updates on **Llama-4 finetuning** in the **help** channel.
   - A member responded that it's coming *this week for sure in prep for llamacon aha* but that there is no project link yet.
- **Defining LLM Novelty**: A debate arose over defining **novelty** in **LLMs**, with one perspective arguing that true novelty cannot exist outside the training set and input context, as models cannot make logical leaps without proper context.
   - Counterarguments suggested that **novelty** is subjective, noting that **LLMs** can produce token sequences not explicitly in the training data, raising questions about when such sequences become novel.
- **Reasoning Models Boost LLM Probability**: Members discussed that using **reasoning models** makes desired completions more probable but doesn't inherently increase the model's base capabilities beyond what the base model can accomplish itself.
   - One member stated *it makes prompting easier but it does not make the model any more capable*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **SillyTavern Becomes LM Studio's ERP**: Users can use [SillyTavern](https://github.com/SillyTavern/SillyTavern) as a front end for **LM Studio** to gain ERP (Enterprise Resource Planning) features and customize their chatbot experience via **Pinokio**.
   - A user provided a detailed 5 step guide that includes installing **Pinokio** and configuring **LMStudio** as the backend.
- **Early Adopter Has Infinite RTX 5090 Woes**: A user reported having an *infinite loading issue* with their **RTX 5090** on the beta LM Studio, and the community rallied to help ensure they're on the latest **LM Studio version (0.3.15 Build 9)** and using **CUDA 12**.
   - Members suggested toggling beta on runtime, and offered workarounds and troubleshooting steps, with the observation that *not many people have 5090 yet so dont know how tested it is*.
- **BitNet CPP Framework Official for 1-bit LLMs**: The [BitNet.cpp](https://github.com/microsoft/BitNet) framework from Microsoft is the official inference framework for **1-bit LLMs**, offering optimized kernels for fast and lossless inference on CPU, with NPU and GPU support planned.
   - This framework supports models like **BitNet b1.58** and includes a suite of optimized kernels that support fast and lossless inference of 1.58-bit models on CPU.
- **Vision Language Models Censors Exposed**: Members discussed how the **VLM** (vision language model) scene is progressing, *specially in the censorship department* and the release of a [less censored version of R1 by Microsoft](https://huggingface.co/collections/microsoft/mai-ds-r1-68003c2753a06be7b9632154).
   - One member noted that it is vision capable model that can actually process pictures without a puritan filter built in.
- **Smol Models Suited for Instruct Tasks**: A user shared some **smol models** that are *more for instruct than for chat*, linking to [QuantFactory/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF).
   - These models can provide outputs of **135, 256, 360, 1.7** tokens.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro Generates Errors**: Users reported that [Gemini 2.5 Pro](https://ai.google.dev/) introduces code formatting errors, causing up to *810 errors* per commit.
   - Despite these issues, one user found that **Gemini** successfully resolved a problem that other models failed to address.
- **Cursor is productivity bro**: Users report using [Cursor](https://cursor.sh) IDE to reach new levels of productivity and it is one of the best IDEs.
   - One user reported using it to convert **Python** code to **C#** and finds it invaluable when pair-programming.
- **O3-preview Smokes O3 In Benchmarks?**: The community observed that **O3-preview** surpassed **O3** in certain benchmarks, a reversal from previous trends, as shown on the [Aider leaderboard](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml#L5).
   - Concerns were raised about models being overly tuned for benchmark performance, potentially at the expense of practical applicability.
- **Deepseek R2 Is No Laughing Matter**: Users jokingly announced the release of **Deepseek R2**, however it may be coming out soon.
   - One user stated *I just took a massive tudmaybe that's R2 coming out?*
- **Users Discuss Aider Configuration Tweaks**: A user is requesting that Aider be configured to exclude *'yes/no'* responses from `.aider.input.history` to reduce clutter and improve context relevance.
   - The user highlighted the lack of context for these responses and seeks a solution for managing history more effectively.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RL Agents Acquire Sign Language**: In a [new paper](https://x.com/superspeeg/status/1914691313318105305), **RL agents** learn to communicate about their **MDP** with **continuous signs** instead of discrete symbols, learning a **communication protocol** that starts as pictographs and evolves into abstract symbols ([arxiv.org/abs/2502.01568](https://arxiv.org/abs/2502.01568)).
   - Concerns were raised about **signal similarity** and evolutionary penalties, with the author clarifying that optimization focuses on inducing correct actions rather than visual aesthetics, given potentially deadly real-world consequences for misinterpretation.
- **Linear Representation Hypothesis Debunked**: A [paper](https://arxiv.org/abs/2402.09268) from Tegmark's group debunks the **linear representation hypothesis**, calling it neither universal nor generally effective.
   - The paper also dismisses **Glove**, noting it uses nearest neighbor retrieval that excludes the original points.
- **Biologically-Inspired Model Extrapolates to 300k**: A biologically inspired architecture for sequential modeling with **O(n) complexity** successfully extrapolated to **300k length** on a synthetic task.
   - With just **39k params**, the model maintained consistent MSE loss on extended sequences, successfully length extrapolating when trained on 1000-2500 length sequences, and validated with 5000 length sequences.
- **DeepSeek's MLA Limits Attention**: DeepSeek's **Multihead Latent Attention (MLA)** restricts attention by limiting key and value heads to read and write to a **512-dimensional subspace** of the **7K-dimensional residual stream**.
   - Query heads can read from a separate **1.5K subspace**, conserving memory bandwidth and potentially improving performance, though some members question whether this is really a subspace, see [research paper](https://arxiv.org/abs/2407.12077).
- **AI Scientist v2 Writes Papers For Pennies**: Sakana AI's [AI-Scientist-v2 project](https://github.com/SakanaAI/AI-Scientist-v2) can produce a full research paper, including hypothesis and experimental testing, for **$15-20 of API tokens**.
   - This sparked fears about a potential deluge of **AI-generated papers** on arXiv.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton adds FP4 support**: **Triton** now supports **FP4**, where **FP4** inputs are provided as a tensor of `torch.uint8`s, each storing 2x **FP4**, as detailed in [this tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html).
   - For converting **FP16** to **FP4**, one member suggested [TileLang](https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py) as a simple and fast solution.
- **Browser CUDA Kernel coding gets Real**: **RightNow AI V2** has launched, a platform for coding optimized **CUDA kernels** directly in the browser ([V2](https://www.rightnowai.co/)).
   - The AI helps generate fast, profiled kernels with real-time **bottleneck analysis** based on user descriptions.
- **Weight-Only Quant Sprints ahead for Small Batches**: For single batch sizes, *weight&activation quantization can be slower than weight-only quantization*, likely due to memory movement overhead.
   - It was explained that activation quantization requires reading activations from global memory, quantizing, and writing back, resulting in more data movement and potential slowdowns for smaller batches.
- **AMD Competition faces bumpy registration**: Members questioned the delay in receiving registration confirmation emails, but confirmed registration is essential to be recognized for prize money.
   - It was also confirmed that **a single file submission is encouraged**, with the option to install packages via pip from the submission file itself.
- **CUDA has Odd fp6 type support**: A member inquired about **CUDA's fp6 type** support and its potential for causing memory fragmentation due to its non-divisibility by 8 or 4.
   - Another member stated that the **fp6 support is really odd**, pointing out padding requirements that make it no better than **fp8** in terms of space-saving in gmem, smem, or tmem.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **O4-Mini Errors Plague Users Despite Successful Requests**: Users reported receiving error messages on **o4-mini**, yet the requests are still processed effectively.
   - Some experienced this after updating Cursor, while others reported it occurring without recent updates.
- **Keybindings Vanish After Cursor Updates**: Multiple users reported that updating **Cursor** breaks their keybindings.
   - No specific solutions were identified, but users confirmed experiencing the same frustrating issue.
- **Gemini and Claude Mix-Up Mayhem**: A user discovered that combining **Google Gemini** for planning with **Claude 3.7** for development leads to unwanted additions and difficulty in bug fixing.
   - Another suggested using **Gemini 2.5** for planning instead of 3.7, as 3.7 tends to add unsolicited features.
- **Cursor Grinds to a Halt for Some**: Several users noted that **Cursor** has become unusably slow, particularly with sluggish requests.
   - Suggestions included that the slowdown might be a tactic to push users to paid plans; restarting Cursor or checking VPN/proxy settings were proposed as potential fixes, according to a [Reddit thread](https://www.reddit.com/r/cursor/s/qnmPu2N59m).
- **Windsurf Catches a Wave, Cursor Loyalists Weigh Anchor**: Members discussed the benefits of **Windsurf** vs. **Cursor**, noting that Windsurf is cheaper, while Cursor offers a superior UI/UX and more innovation.
   - One user found Windsurf's tab to be *better then [they] expected at predicting*, and another linked to a tweet about it [Windsurf](https://x.com/heyrobinai/status/1914829284004471099?s=46&t=kUuVqsG2GMX14zvB592G5w).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-Image-1 Images Its Way to Developers**: OpenAI released **gpt-image-1**, a new **Image Generation API** that brings ChatGPT's image generation power to developers, boasting **more accurate, high fidelity images** and **consistent text rendering**.
   - The new **Image Generation API** allows users to create images using the **gpt-image-1 model**, with a [guide](https://platform.openai.com/docs/guides/image-generation) provided for developers to get started.
- **Gemini 2.5 Pro Fights Gemini 2.5 Flash**: Members discussed the merits of **Gemini 2.5 Pro** versus **Gemini 2.5 Flash**, with one user suggesting using all AI models to get the best result.
   - The discussion included concerns that **o3**, **o4 mini high** and **Gemini 2.5 Pro** struggle with **high school geometry** problems, while **Deepseek** solved a particular SAT geometry question correctly.
- **Sora Shuts Down for Newbies**: Users reported that **video generation is temporarily disabled for new accounts** on **ChatGPT Plus**, which was confirmed to be intentional.
   - The reason for the change was not provided.
- **ChatGPT App Beats Webapp**: A user claimed that *the **ChatGPT app** is way better than the **webapp** ngl*, adding that they use the **API**.
   - The user shared screenshots showing performance differences in solving math problems with **ChatGPT o4-mini-high**, which initially failed but corrected itself when prompted to check the answer.
- **Plus Plan Perks Post Cancellation Pondered**: A member questioned whether saved memories and chats would remain accessible after cancelling a **Plus Plan** subscription.
   - Another member suggested that while exclusive models may become inaccessible, chats could be transferred to **4o** on a free account, or pasted directly into the free model.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Gemini 2.5 Pro outperforms NotebookLM in reasoning**: A user compared **Gemini 2.5 Pro** to **NotebookLM**, finding **Gemini 2.5 Pro** *much better* than **ChatGPT o3** or **o4-mini** when reasoning.
   - Another user shared that giving **NotebookLM** books and materials on logical and mathematical reasoning made it unable to solve a logic puzzle solved easily by **Gemini 2.5 Pro**.
- **NotebookLM Math and Images Get No Love**: Users report difficulties with **NotebookLM** and **math notation** and **image loading**, suggesting it lags behind **GPT-4** in processing formulas.
   - The team is aware of the issue and *is working on it*.
- **NotebookLM Audio Overviews Missing Languages**: A user asked if the audio summary feature of **NotebookLM** could generate podcasts in Spanish.
   - The response was *Not right now* indicating language support is currently limited, but could be improved in the future.
- **NotebookLM PDFs Can't Be Too Large**: Users are encountering issues with **NotebookLM** stopping midway through lengthy PDF documents.
   - The suggested workaround is to split the PDF into smaller segments using tools like [iLovePDF](https://www.ilovepdf.com/pt/dividir_pdf).
- **Privacy Paywall Possibly Protects NotebookLM Data**: A user questioned whether **Notebook LM** trains on user data, recalling information about paid subscriptions offering privacy benefits, linking to the [Google Support page](https://support.google.com/notebooklm/answer/15724963).
   - It is unclear whether user data is used for training purposes.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Agents Outpaced by Humans!**: Members debated the efficacy of AI **agents**, with one arguing that *humans are cheaper, faster, and more reliable* in most scenarios due to issues like dynamically generated workflows, see [this discussion](https://discuss.huggingface.co/t/my-space-suddenly-went-offline-the-cpu-cannot-restart/151121/22).
   - Despite offers to test agent-based systems, the original poster expressed stronger interest in audio research with agents.
- **HF Space Gets Trashed!**: Users reported **Hugging Face Spaces** going offline, as seen in [this discussion](https://discuss.huggingface.co/t/my-space-suddenly-went-offline-the-cpu-cannot-restart/151121/22), prompting the infrastructure team to investigate.
   - The issue was resolved with a fix that required restarting the affected Spaces.
- **Llama 3 Template Troubles!**: A member encountered output issues with **Llama 3** on a Windows PC, suspecting a problem with the chat template which was [resolved by using the format `{'role': 'user' , 'content': message }`](https://huggingface.co/learn/agents-course/en/unit0/introduction).
   - Specifically, the user was using the wrong chat template on the Windows PC.
- **ML Channel Lifts Fine-tuning Fundamentals**: An ML enthusiast promoted their **YouTube channel** *Let's Fine-tune Everything*, which features hands-on tutorials and guides on fine-tuning open-source models for real-world use cases, covering topics from **object detection** to **LLMs**.
   - The channel offers content for both beginners and experienced practitioners.
- **Open-Source QA Project Seeks Contributors**: A member open-sourced an **AI-powered document Q&A project** with a **FastAPI** backend, employing a retrieval-based approach with embedding models, see the [repo and post](https://www.linkedin.com/posts/hamzabouajila_ai-opensource-python-activity-7320494631471771648-z9wE) for more.
   - The developer is actively seeking feedback on **architecture, code quality, scalability**, and general suggestions for improvement.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Brains Process Locally**: Discussion highlights that the brain's processing is predominantly **local**, with neurons receiving signals from directly wired neighbors and having local internal processes shaped by position, connectivity, and context.
   - Members debated the role of **quantum non-local information processes in cytoskeletal microtubules** versus more conventional models of neural networks.
- **Paper Discussions Go Unrecorded**: Members noted that there are no recordings of the Saturday Paper Discussions, specifically regarding **Anthropic's recent paper**.
   - One member linked to [Yannic's video on Anthropic's paper](https://www.youtube.com/watch?v=mU3g2YPKlsA) when another member wanted to hear more about the discussion.
- **Mental Models vs World Models Debated**: Discussion centered on **mental models** (internal simulations) and **world models** (broader representations), with the brain building mental models to predict and compare with reality.
   - The discussion referenced the [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle) and [Predictive Coding](https://en.wikipedia.org/wiki/Predictive_coding) as relevant concepts.
- **Transformer Tactics Spark Discussion**: A member asked if *x = self.transformer(x, x)* is reasonable in the forward pass and members explained that it is often done when **self-attention** is needed.
   - Members linked to an [IBM article on self-attention](https://www.ibm.com/think/topics/self-attention) and advised considering *“ϵ-greedy” exploration* instead, pointing to the value of [randomness and exploration](https://spectrum.ieee.org/2d-semiconductors-molybdenum-disulfide).
- **Muon Set to Replace Adam**: The community will be discussing **Muon**, a faster replacement for **Adam**, with a [blogpost about Muon](https://kellerjordan.github.io/posts/muon/) shared.
   - The [ArXiv link](https://arxiv.org/abs/2310.11453) about WaveFunction was also mentioned, after a member asked if **Muon** is a reverse distillation method.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Zed's Diagnostics Delights Developers**: Members lauded Zed's **Project Diagnostics** feature, accessible via **⇧⌘M**, for enabling rapid error identification and on-the-spot editing.
   - One member expressed that it is *convenient and motivating to be able to quickly make changes and see the outstanding error/warning count go down to zero*.
- **Modular Meetup Makes Moves**: The community announced the [Modular Meetup](https://lu.ma/modular-meetup) in Los Altos, offering limited in-person attendance.
   - Talks will be broadcasted on [YouTube](https://www.youtube.com/watch?v=uul6hZ5NXC8) and [LinkedIn](https://www.linkedin.com/events/next-gengpuprogramming-hands-on7319044981682270210/).
- **MAX/MOJO Licensing Logic Queried**: A member questioned the **MAX/MOJO** license's business strategy, specifically *Production / Commercial on Other Accelerators*.
   - They wondered if this was a tactic to gather feedback for **MAX/MOJO** development on non-NVIDIA GPUs.
- **Community Cooks Up Training Pipeline Examples**: Despite **Mojo's** lack of native training support, the community is eager to employ it in training pipelines for data manipulation before **PyTorch**.
   - Members inquired about available examples of **Mojo**-driven training pipelines and even at an early stage.
- **Enum Alternatives Explored in Mojo**: With **Mojo** lacking dedicated enums, the community considered **DType** (enum-like) and **utils.Variant** for unions.
   - A [DType implementation](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/dtype.mojo) was shared for reference.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Autonomous.ai Launches Brainy**: [Autonomous.ai](https://www.autonomous.ai/robots/brainy) debuted **Brainy**, an **RTX 4090 AI supercomputer** with impressive **O3 agent UX**, focusing on image analysis.
   - The announcement has garnered attention for its potential in advancing AI applications.
- **Scout.new Servers Melt Under Load**: Members noted **Scout.new** was broken due to load, with others saying *it's fucking cooking hot damn*.
   - A member posted link to [X cancel](https://xcancel.com/rayfernando1337/status/1914791594789879844) for the **Ray Fernando** post, indicating high interest but current instability.
- **OpenAI Unveils Image Generation API**: **OpenAI** released **image generation** capabilities in their [API](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1) using **gpt-image-1**.
   - This update allows developers to integrate image generation directly into their applications.
- **Microsoft Announces Copilot Agents**: **Microsoft** announced [Copilot Agents](https://x.com/satyanadella/status/1915098359251247392), signaling a move towards more integrated AI assistants.
   - Details are still emerging, but the announcement has sparked interest in the capabilities and applications of these agents.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Unleashes Milvus-Powered Full-Text Search**: LlamaIndex now supports **full-text search with BM25** via integration with [@milvusio](https://github.com/milvus-io), enabling hybrid search in **RAG pipelines**.
   - This feature combines vector search and keyword matching; a tutorial is available [here](https://t.co/0dCi0kEn6o).
- **Agentic Document Workflow Eclipses RAG Chatbot**: The **Agentic Document Workflow (ADW)** is positioned as an improvement over the **RAG chatbot** prototype, offering better scalability, integration with existing software, and superior error handling.
   - More details on the **ADW** can be found [here](https://t.co/ZZzr7scHhF).
- **LlamaParse's Text() Troubles Resolved**: A user discovered that **LlamaParse's** `getText()` function in next.js returned partial content with `resultType` set to `markdown`, tracing it to a **markdown vs text comparison issue**.
   - Switching to `const reader = new LlamaParseReader({ resultType: "text" });` rectified the problem.
- **MLflow's Autolog Anomaly in FastAPI's Parallel Processing**: A user reported inconsistent capture of **LLM call traces** by **MLflow autolog** when running a **LlamaIndex Workflow** inside **FastAPI's background task** with parallel tasks, resulting in an *'NoneType' object has no attribute 'info'* warning.
   - This suggests a potential **MLflow-specific issue** in handling parallel execution environments.
- **TRL Rockets into Instruction-Finetuning Realm**: Instead of using LlamaIndex tools, a member advocated for **TRL (Transformers Reinforcement Learning)** for instruction-finetuning open-source LLMs and provided a link to the [Hugging Face TRL documentation](https://huggingface.co/docs/trl/en/index).
   - The suggestion includes creating a dataset by distilling training from an existing LLM into another.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Interview Compensation Offered**: A member is [paying **$40** for a **30 min interview**](https://discord.com/channels/1119947416775741521/1119947417233033328) for people who have worked with **Claude Computer Use** and/or **OpenAI computer-use-preview** in a real project, with bonus points for those implementing **MCP**.
   - The member needs to ask *a million questions* about users' experiences.
- **README Translation Automation Gets Proposal**: A member proposed storing all links, tags, and emojis in a single **JSON** file to automate the generation of translated **READMEs** via the **CI pipeline**.
   - This approach centralizes maintenance and reduces effort by updating only the primary **README**.
- **AWSLab Cost Analysis Crashes MCP Server**: A member reported that the **Claude Windows desktop app** freezes with an error when generating the cost report for last month using the **AWSLab cost analysis MCP server**.
   - The error message displayed is *Claude’s response was interrupted*, despite a stable internet connection.
- **Request Timeout Plagues MCP Inspector**: A member is encountering an **MCP error -32001**: *Request timed out* when running the basic **MCP server** from the docs on **GitHub** and using the interactive server.
   - The error prevents them from running any tools, despite it working correctly in **Claude desktop** when running `mcp install test.py`.
- **Defang Labs Ships Vibe-Coded MCP Server**: Defang Labs built an MCP server that lets you **deploy your vibe-coded project from any IDE straight to the cloud** and are asking for feedback [on their LinkedIn post](https://www.linkedin.com/posts/defanglabs_vibecoding-defang-devtools-activity-7320490826004852737-2IFE?utm_source=share&utm_medium=member_desktop&rcm=ACoAACNoYXgBadWv4CWLbcKhgSGxWjdmu9e5dFI).
   - The Defang server helps developers ship code to the cloud.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Arithmetic Shift Right Op Questioned**: A member inquired about the existence of an **arithmetic shift right op** within the tinygrad system, seeking clarification on its implementation and usage.
   - This query suggests ongoing development or potential feature additions related to bitwise operations within the framework.
- **Matching UPat with CONST**: A member requested a way to create a **UPat** to match a **CONST** where the immediate is only, for example, **5 bits long**, or where the bottom **n bits are zero**.
   - The request highlights a need for more flexible and specific pattern matching capabilities within the system's rewriting engine.
- **Constraint Solver Backend Pursued for Instruction Ordering**: The team is transitioning towards a **constraint solver backend** that jointly handles **instruction ordering** and **register assignment** to better optimize code generation.
   - This indicates a shift towards more sophisticated optimization techniques within the tinygrad compiler.
- **Arange Gets Optimized Out**: It was mentioned that `arange()` gets optimized out, according to [this tinygrad notes link](https://xl0.github.io/tinygrad-notes/arange.html), potentially impacting how range-based operations are handled.
   - This optimization might affect the performance and implementation of code that relies on `arange()` for tensor creation or manipulation.
- **Indexed Operations: Finding Byte Indices**: A member suggested finding the byte indices by getting the **indexed_ops** of both the **STs** (ShapeTracker) and then plugging in the tensor indexes *i,j,k*, as referenced in [device.py](https://github.com/tinygrad/tinygrad/blob/6cb2d18c034fc3fb8c8c7521716c04a4674c5504/tinygrad/device.py#L330).
   - This approach aims to facilitate more efficient memory access and manipulation within the tinygrad framework.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 3.0 release brings the Hype**: Members expressed excitement for **DSPy 3.0**, but one user asked *"What can we expect??"* regarding a [tweet](https://x.com/lateinteraction/status/1915058777491145200).
   - The unifying vision/design for **DSPy 3.0** is not written anywhere, because *"too many things that are internal research until they're not!"*, linking to the [roadmap](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md).
- **DSPy 3.0 ETA is set for June 2025**: A member stated that the ETA on **DSPy 3.0** is *"June 2025"*.
   - Another guessed that the reveal will be around **Databricks event in SFO**.
- **Synthetic Flywheel seeks clearance**: Two members discussed making a *"synthetic flywheel so fly its gonna need airspace clearance."*
   - Further details on the implementation and specific use-cases were not specified.
- **Prompt Optimization likened to black magic**: One user who bet on **DSPy** for their gen-ai development a year ago, now feels that *"it was not the right thing to do"*, because *"Prompt optimization seems a bit black box."*
   - This user suggests that the unpredictability in prompt optimization has made development difficult.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **RoPE Class Sparks Debate**: A dedicated **RoPE** (**Rotary Position Embedding**) class implementation in *torchtune* was questioned for its design, with the primary reason being that it felt *more PyTorch-y* than a function.
   - The class allows for the **RoPE cache** to be initialized once and re-used, which trades off speed for memory.
- **Collective Scheduling Testing**: A member is testing the throughput and memory usage of customizing the **collective scheduling** and plans to submit a PR if the results are promising.
   - They are considering arguments like `fsdp_delay_all_reduce` and aligning them with **deepspeed stages (zero 1-3)**.
- **`tune cp` workflow Success on macOS**: A member detailed their experience using the `tune cp` workflow on a Macbook, highlighting issues such as needing to manually search for recipe and config files, remove file extensions, and resolve dataset version mismatches, but ultimately achieving success after addressing **macOS-specific issues**.
   - The member also noted the workflow relies heavily on *massive code duplication* which feels off.
- **Hybrid Library Design Under Discussion**: Discussion arose around the hybrid library design approach in *torchtune*, which aims to provide user scripts that are easy to customize while leveraging a library for common components.
   - The team is looking to determine whether the *hybrid design* is a fundamentally flawed approach or a user education/documentation issue, which allows researchers to showcase only the code that matters.
- **Scheduling RL Post-Singapore Call**: A user mentioned their availability for a call starting late next week, as they will be returning from Singapore.
   - The user provided a specific timeframe for scheduling a call, indicating they will be available after their return from Singapore late next week.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Guidance Sought on SaaS Tool Integrations**: Engineers seek alternatives to **Zapier** for building integrations with existing **SaaS platforms** that support multiple connections for different customers.
   - They suggested **Composio** as a potential solution, seeking community input on its suitability or alternative recommendations.
- **Nous Teases Red Team Release**: Nous hinted at an upcoming release, scheduled for today or tomorrow, that is tailored for the **red team** community, potentially with *new mix precision quantlol*.
   - The announcement generated anticipation among members interested in novel tools and resources for security and adversarial testing.
- **SuperNova Models Garner Praise**: Members expressed appreciation for the **SuperNova models** by **Arcee-AI**, citing their strong performance relative to their size.
   - One member noted that the two **SuperNova Models** are now their default models since release.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC Readings Finally Posted**: The readings for the **LLM Agents MOOC** are now available on the website [llmagents-learning.org/sp25](https://llmagents-learning.org/sp25).
   - These readings are highly relevant to the course content and assignments, so prioritize these.
- **Resource Submission Confirmation Sought**: A member inquired about receiving a confirmation email after submitting a resource submission form and they did not receive any email confirming their submission despite completing the form.
   - This is relevant for the course since members need to know that their submissions were correctly received.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **HF Inference API Ties to Flask**: Models uploaded to **Hugging Face** and using their paid inference API can be connected to a website built with **Flask**.
   - The **Flask** application sends requests to the **Hugging Face Inference API** endpoint, then the API returns the model's prediction, which is then displayed on the website.
- **Flask asks Hugging Face Paid Inference**: A member asked how to connect a **Flask** website to a model uploaded on **Hugging Face** using their paid inference API.
   - New members are encouraged to introduce themselves by sharing their company, what they're working on, favorite tech tools, and what they hope to gain from the community.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Handler Errors Bugging System**: Members are reporting that *something from your handler is erroring* in the system, affecting **Gorilla LLM**.
   - Suggestions include modifying the [error-catching code](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286) to throw errors instead of catching them, to assist in **debugging**.
- **Debugging Advice Shared on Leaderboard**: A member suggests modifying the [code](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286) to throw errors instead of catching them to help **debug** **Gorilla LLM**.
   - They advise running generation for **one entry** to see the **full trace** of the error.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Upcoming Legislative AI/Tech Webinar Announced**: Entrepreneur Karen Suhaka (Founder of **BillTrack50**) is partnering with Silicon Valley Chinese Assocation Foundation to host a webinar on legislative applications of AI and Technology on **April 28 at 12pm Pacific**, with registration available at [this link](https://forms.gle/v51ngxrWdTsfezHz8).
   - The webinar will delve into building legislative technology, addressing ethical considerations, and providing entrepreneurial advice.
- **BillTrack50 Startup Insights Revealed**: Karen Suhaka will present **BillTrack50** as a case study, sharing her experience in building, scaling, and gathering customer feedback for her legal tech company.
   - She will emphasize identifying market needs and selecting appropriate data and methodologies.
- **AI4Legislation Competition Details Unveiled**: The webinar will introduce project concepts for the **Summer 2025 AI4Legislation competition**, with specifics available on [GitHub](https://github.com/svcaf/2025-AI4Legislation-Public/tree/main).
   - The competition seeks to use recent progress in **LLMs** and **NLP** to empower citizens and voters.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Nomic.ai (GPT4All) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1364627372569133097)** (1 messages): 

> `iOS Voice Assistant, Multi-app Actions, Mobile App Update` 


- ****Perplexity** launches iOS Voice Assistant!**: The new **Voice Assistant** leverages web browsing and multi-app actions to book reservations, send emails, play media, and manage calendar invites directly from the **Perplexity iOS app**.
   - Check out the [full announcement with examples on X](https://x.com/perplexity_ai/status/1915064472391336071) and update your app in the App Store to start asking.
- ****Voice Assistant** uses web browsing**: **Voice Assistant** leverages web browsing and multi-app actions to book reservations.
   - **Voice Assistant** is available in the Perplexity iOS app.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1364316701298130957)** (1104 messages🔥🔥🔥): 

> `Perplexity AI Terms of Service, James Webb telescope, Perplexity comet release date, R1o4-mini vs grok 3 vs gemini 2.5 pro vs claude, Raycast a web app` 


- **Perplexity AI warns against TOS violations**: A member posted a link to the [Perplexity AI Terms of Service](https://www.perplexity.ai/hub/legal/terms-of-service), cautioning users to avoid violating the TOS.
   - The warning was seemingly triggered by a user discussing the use of promotional codes acquired through their carrier plan, which might be against the TOS.
- **Members Spoil Details About James Webb Telescope Photos**: A member shared an [image from the James Webb Telescope](https://cdn.discordapp.com/attachments/1047649527299055688/1364579832410935427/IMG_2144.jpg?ex=680a2f80&is=6808de00&hm=0e6c9c31fa09117d059bffdd1e3f964c79dc93988a5da878202099691d82b47e&), but another member quickly pointed out that the **colors in such images are not real**.
   - Despite the revelation about the colors, members agreed the image was still cool, with one identifying it as a spiral galaxy.
- **Users Discuss Grok and Perplexity Models**: Members debated the performance of different models like **Grok**, **Gemini**, and **O4 mini** on Perplexity, with varying opinions on their strengths and weaknesses.
   - Some find **Grok 3** good for general questions and research, while others prefer **Gemini 2.5 Pro** for coding; members also noted **O4 mini** is preferred.
- **Perplexity iOS Voice Assistant launched**: Perplexity AI launched its **new iOS Voice Assistant** with a member pointing out the [announcement](https://fixupx.com/perplexity_ai/status/1915064472391336071/mediaViewer?currentTweet=1915064472391336071&currentTweetUser=perplexity_ai) with some requesting support for other languages and integration into other systems.
   - One member stated that the new iOS assistant can tap into your **contacts**, **calendars**, **reminders**, and **Apple Music** as well as **create emails**.
- **Image Generation is Still Buggy on Perplexity**: Users reported issues with image generation on Perplexity, such as defaulting to the **Flux model** and not adhering to prompts correctly.
   - It was also noted that the ability to edit generated images is limited, as the system tends to reuse the original attached image instead of modifying the generated one; in a humorous take, some described the experience as *"delulu"*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1364421196719980555)** (3 messages): 

> `Perplexity AI, comprehensive 10000` 


- **URL Shared for Perplexity AI**: A member shared a [Perplexity AI search result](https://www.perplexity.ai/search/given-the-current-condition-an-cRXRlWsRTEOrL99ZHzd3HQ).
   - Also a [second search result](https://www.perplexity.ai/search/generate-a-comprehensive-10000-9FO7aPu9QHKWRfdDC2E88Q) was shared.
- **Image posted**: A member shared a [link to an image](https://pasteboard.co/FiWizDPt8Knd.png).
   - The image contents were not specified.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1364351562901880873)** (11 messages🔥): 

> `API key revoke, Requests per API, Office hours` 


- **Requests per API Do Not Web Search**: A member reported that requests per API do not perform a web search, even though everything works as intended in Playground.
   - Another member then suggested that the member try a [specific curl request](https://api.perplexity.ai/chat/completions) to fix the issue.
- **API Key got revoked!**: A member mentioned to revoke the API key.
   - He also mentioned to update the link in the error message.
- **Office Hours this afternoon**: A member reminded others about office hours this afternoon.
   - He also provided a [Zoom link](https://events.zoom.us/ev/Akzh8Q9GwGtQ8-5yeP1A6B0kQBND1W67rbimE3koC4L_L4ZP65f2~Ag4nJHk6gbPxvgM1f_OCr6BzgyKoKK7hLYpE3HmzJ69MnMG3CvFABoNg6Q) for anyone who wanted to join.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1364314852876877894)** (1097 messages🔥🔥🔥): 

> `Llama games LM arena, emoji, GPT-4.1, small cheap models, arc prize` 


- **Llama's LM Arena Antics**: Members discussed how **LLama** might have *gamed* the **LM Arena** during training.
   - The conversation then shifted to whether style-controlled IMBY optimizing for human stylistic preference could solve for issues like excessive emoji use in AI.
- **Emojis help you get more dates**: A [study](https://www.ktsa.com/study-people-who-use-more-emojis-have-more-dates-and-sex/) suggests that people who use more emojis have more dates and sex.
   - It was suggested that being *agreeable*, *positive*, and using *emojis* could be beneficial in **LM Arena**, but one member disagreed with agreeability being necessarily good.
- **GPT-4.1 is very good and cheaper**: **GPT-4.1** is considered very good because it's close to **Sonnet** in areas where **Sonnet** excels, but it's cheaper.
   - However, it's noted that **GPT-4.1 mini** is better and has a more efficient tokenizer than **Claude**, though it's not ideal for web design or coding visuals.
- **Vintage Benchmarks Small (Cheap) Models**: A member shared [a quick benchmark of small (cheap) models](https://cdn.discordapp.com/attachments/1364321889044136067/1364330919686705286/Screenshot_2025-04-22_at_22.02.43.png) and was surprised to see **Gemma 3** performing so well for its cost.
   - The member noted that they have a separate set of harder benchmark questions for frontier models.
- **OpenAI's O3-Preview**: Members discussed **OpenAI's O3-preview** benchmarks and the fact that the released **O3** was much weaker.
   - It was suggested that **O3-pro** might achieve 80%+ on ARC-1, with a 10% to 20% score on ARC-2, despite **O3 preview high** costing thousands per task.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1364319188637847592)** (590 messages🔥🔥🔥): 

> `Manus Pricing, DeepSeek vs Manus, Genspark, Credits, OpenAI vs Manus` 


- ****Users Debate Manus' Pricing and Credit System****: Users are debating whether the cost of **Manus** is too high, with one user suggesting a [slow-processing mode](https://link.to/slow-processing-mode) that uses fewer resources.
   - Some users are saying that *the price is very steep considering you're still very limited credit-wise*.
- ****Users Compare Deepseek and Genspark to Manus****: A user compared **Manus** to **Deepseek** and **Genspark**, noting that while Deepseek offers daily credits, it isn't as good as Manus.
   - Another user agreed, mentioning that *Deepseek make their money through their API instead of their model*.
- ****Feature Suggestions Abound: Credits, Pricing, and Model Selection****: Members proposed ideas to the team for [credit sharing between accounts](https://link.to/credit-sharing), while another suggested [pricing based on hours used per month](https://link.to/pricing-by-hours).
   - Others called for a custom model selection feature, such as cheaper **Gemini 2.5 Pro** or the more expensive **Claude 3.7 Sonnet**
- ****Community Debates Data Privacy and Security****: Users discussed concerns about data privacy, including whether [Manus shares data with Claude](https://link.to/claude-policy), with some joking about preferring their data to go to China rather than the American government.
   - One member stated that *this is pretty much the only capable ai ive seen that doesnt come from a forbes 500 company*.
- ****Users Find Potential for Minecraft Mod Creation****: Members discussed the possibility of using Manus to [create Minecraft mods](https://link.to/minecraft-mods), including compiling them into JAR files.
   - Other members said the team needs to learn how to *actually take in suggestions more*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1364345912431743047)** (8 messages🔥): 

> `Sonnet 3.7 capacity issues, Clerk authentication delays, OpenRouter PDF support, PDF processing engines, Gemini API PDF Input` 


- **Sonnet 3.7 Gets a Capacity Boost**: OpenRouter addressed capacity issues on **Sonnet 3.7** and implemented improvements to lower error rates.
   - Users should see much lower error rates now but are apologized to for the disturbance.
- **Clerk Authentication Faces Downtime**: OpenRouter's authentication provider, **Clerk**, experienced delays and downtime, leading to **401 errors** and login difficulties; see the [Clerk status page](https://status.clerk.com/) for updates.
   - Clerk reported seeing recovery on their end after the incident.
- **OpenRouter Opens Universal PDF Support**: OpenRouter now supports **PDF processing** for every model, potentially being the first platform to do so, as announced on [X.com](https://x.com/OpenRouterAI/status/1915083006349382033) with a [video demo](https://cdn.discordapp.com/attachments/1092729520181739581/1364636811003035789/pdf2.mp4?ex=680a6491&is=68091311&hm=33ff94487e038de43aea6ad12c6041fe14da683e6fd29de0b90e94016bada256&).
   - The new feature includes universal compatibility, handling any PDF type with native support for providers like **Gemini**, **Anthropic**, and **OpenAI**, accessible via **API** and the **OpenRouter Chatroom**.
- **Pricing PDF Processors**: OpenRouter introduces two **PDF processing engines**: `mistral-ocr` at **$2 per 1000 pages** for OCR support with text and embedded image extraction, and `pdf-text` for free, offering text-only extraction without OCR or image support, detailed in the [documentation](https://openrouter.ai/docs/features/images-and-pdfs).
   - A user wished for something between **mistral OCR** and **text only**, *like smol docling*.
- **Gemini Gets PDF-tastic**: The **Gemini API** supports **PDF input**, including long documents (up to **3600 pages**), processing them with native vision to understand both text and image contents, as shown in the [OpenRouter Docs](https://openrouter.ai/docs/features/images-and-pdfs).
   - A member has noticed significant improvements to data extraction when also providing a plain-text parse alongside the pages themselves with Gemini Flash.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1364315429321052260)** (259 messages🔥🔥): 

> `Tool Calling Limitations, Google Gemma Quantization, Gemini Search Grounding, Account Creation Issues, Free Model Function Calling` 


- ****OpenRouter** Experiences **Authentication Issues****: Users reported issues creating new accounts, receiving *"Error 401: User not found"* messages, which was attributed to a slowdown with **OpenRouter's** authentication provider.
   - The team investigated and confirmed the problem, providing updates on the [Clerk status page](https://status.clerk.com/) and later confirmed the issue was resolved, though some users ended up with multiple unwanted accounts due to testing.
- ****Gemini 2.5 Pro** Struggles with **Rate Limits****: Users reported frequent *"Rate limit exceeded"* errors when using **Gemini 2.5 Pro** preview, particularly the free version, leading to discussions about reliability and possible solutions.
   - One suggestion was to use a personal Google AI Studio API key to increase limits via the "integrations" page in account settings, though the actual fallback behavior and impact on Google's RPD remains unclear.
- ****OpenRouter** Adds **PDF Support****: **OpenRouter** announced universal PDF support, but the initial documentation link ([https://openrouter.ai/docs/features/images-and-pdfs](https://openrouter.ai/docs/features/images-and-pdfs)) was broken and quickly fixed.
   - The **Mistral OCR** is the processing engine and its pricing was discussed, with some users noting an upcharge compared to going directly to Mistral, but others expressed excitement about the new feature.
- ****Deepseek v3** shines in Function Calling**: **Deepseek V3** is good with function calling when the context is small
   - However, as soon as the context grows it becomes bad


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1364317633050312786)** (172 messages🔥🔥): 

> `Scout use cases with 128GB unified memory, Unsloth support for GLM-4 9B/32B models, Unsloth Dynamic v2.0 quants release, Torch 2.7 release and its changes, Evaluation benchmarks: MMLU, Humaneval, Aider Polygot` 


- **Scout finds Niche Users with Unified Memory**: Users are exploring use cases for **Scout** with **128GB unified memory**, citing decent throughput at low power consumption.
   - One use case mentioned was an **RP tracker**, extracting structured information to prevent character inconsistencies, though not using Llama 4 due to its size.
- **GLM-4 gets Transformers Integration**: Unsloth support for **GLM-4 9B/32B models** is available if they work in Transformers, though users reported partial success in finetuning due to issues with applying the template and merging adapters.
   - An issue was reported relating to the size of **GLM4's rope dimension** being **64** which slipped through most inference engines.
- **Unsloth's Dynamic Quantization arrives**: Unsloth is releasing **Unsloth Dynamic v2.0 quants** soon, claiming it will be very good and linked to a [Hugging Face collection](https://huggingface.co/collections/unsloth/unsloth-dynamic-v20-quants-68060d147e9b9231112823e6).
   - It was noted that improvements are seen across the board, including **Q8**, with most benefits seen at **Q4** and that Unsloth is conducting 5-shot MMLU benchmarks against iMatrix quants, Google's QAT, standard GGUF, and old Unsloth dynamic iMatrix.
- **Torch 2.7 Materializes with Contextual Support**: **Torch 2.7** was released, featuring support for tracing contextlib.contextmanager in Dynamo and tracing generators, with [release notes available on GitHub](https://github.com/pytorch/pytorch/releases/tag/v2.7.0).
   - One user was waiting for the release but had to focus on semester exams in DSA and operating systems.
- **Unsloth Team Debates Benchmark Methodology**: The Unsloth team is benchmarking their new quantization methods and comparing against Google's QAT, with discussion on using **5-shot MMLU** for evaluation and a recommendation to look at [this paper](https://arxiv.org/abs/2407.09141).
   - There was discussion on evaluating different quantization levels with different benchmarks, and disagreement over the usefulness of perplexity, humaneval and swe-bench.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1364328323324313620)** (6 messages): 

> `Token processing speed, PyTorch benchmarks` 


- **Each token takes a fifth of a century**: A member jokingly noted that in a [YouTube video](https://www.youtube.com/watch?v=3q_ItuNNpmYE), *each token takes a fifth of a century* to process.
   - This humorous remark was made in the context of a discussion about token processing speed and efficiency.
- **PyTorch's Infra Website Showcases Interesting Benchmarks**: A member shared a link to [PyTorch's infrastructure website](https://hud.pytorch.org/benchmark/compilers) which contains interesting benchmarks that are updated with each PR.
   - The benchmark tab was highlighted as having the most interesting information for those tracking PyTorch's performance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1364372235187388416)** (53 messages🔥): 

> `GRPO Models, Classification task using numbered labels, Llama-4 finetuning updates, Embedding model recommendations, Fine-tuning Llama for Vtuber` 


- **GRPO Model Options?**: A user inquired about the range of models available for the **GRPO**, questioning whether it's limited to **Gemma 3 (1B), Llama 3.1 (8B), Phi-4 (14B), and Qwen2.5 (3B)**.
   - A member responded by saying *all models except VL*.
- **Int or Str for Classification Task Labels?**: A user asked whether a classification task using numbered labels requires checking for **int** or **str** type.
   - Other members suggested it has to be **int** and pointed to the huggingface documentation for details.
- **Llama-4 Finetuning Coming Soon**: A user inquired about updates on **Llama-4 finetuning**.
   - A member responded that it's coming *this week for sure in prep for llamacon aha* but that there is no project link yet.
- **Embedding Model Recommendation Needed**: A user sought recommendations for an embedding model for document chunks of **<= 1024 tokens** and QA pairs less than **512 tokens**.
   - They expressed concern that an embedding model aimed at **8K tokens** might not be suitable or necessary.
- **Seeking Guidance for Fine-Tuning Llama for Vtuber**: A user with limited AI experience asked for help in **fine-tuning Llama for Vtuber applications**.
   - It's unclear whether the user found suitable assistance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1364321792990249000)** (27 messages🔥): 

> `Reasoning Models, LLM Novelty, Training Data Limitations, Sampling Token Sequences` 


- **Reasoning Models Boost LLM Probability, Not Capability**: Members discussed that using **reasoning models** makes desired completions more probable but doesn't inherently increase the model's base capabilities beyond what the base model can accomplish itself.
   - One member stated *it makes prompting easier but it does not make the model any more capable*.
- **Training Reduces Randomness for Factual LLMs**: It was argued that for tasks requiring **factual correctness**, a model needing *k* generations is less capable than one needing only one, highlighting that training aims to reduce randomness.
   - The goal is to reduce the *k* needed for accuracy through effective training.
- **Defining Novelty Subjectively in LLMs**: A debate arose over defining **novelty** in **LLMs**, with one perspective arguing that true novelty cannot exist outside the training set and input context, as models cannot make logical leaps without proper context.
   - Counterarguments suggested that **novelty** is subjective, noting that **LLMs** can produce token sequences not explicitly in the training data, raising questions about when such sequences become novel.
- **Exploring the Bandwidth Problem**: A member shared a [Google Colab notebook](https://colab.research.google.com/drive/1JC1cEsk-3SxIUPWL7wF0eveelyo3e7fy?usp=sharing) detailing their updated research on how **MLA** (Multilayer LSTM Architecture) addresses the **bandwidth problem**, inviting early feedback.
   - The write-up focuses on making the **MLA concept central** to solving bandwidth issues.
- **Pretraining Remains Paramount for Strongest LLMs**: A key takeaway was that everything is downstream of pretraining.
   - Members agreed, *you still want the strongest base model possible*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1364318162723803259)** (170 messages🔥🔥): 

> `SillyTavern as front end for LM Studio, Pinokio install, 5090 loading issues on LM Studio, Cuda 12, Bitnet CPP` 


- ****SillyTavern** Front-End Leaps into **LM Studio****: Users can use [SillyTavern](https://github.com/SillyTavern/SillyTavern) as a front end for **LM Studio** to gain ERP (Enterprise Resource Planning) features and customize their chatbot experience.
   - A user provided a detailed 5 step guide: 1. Install [Pinokio](https://pinokio.computer/), 2. Install SillyTavern using Pinokio 3. Run LMStudio, load a model and start the server 4. Start SillyTavern and configure LMStudio as the backend 5. Play with the chat of SillyTavern.
- ****RTX 5090** Early Adopter Sees Infinite Loading Loop**: A user reported having an *infinite loading issue* with their **RTX 5090** on the beta LM Studio, and the community rallied to help.
   - Members suggested ensuring they're on the latest **LM Studio version (0.3.15 Build 9)** and using **CUDA 12**, and toggling beta on runtime, and offered workarounds and troubleshooting steps, with the observation that *not many people have 5090 yet so dont know how tested it is*.
- ****BitNet CPP** Framework Bolsters 1-bit LLMs**: The [BitNet.cpp](https://github.com/microsoft/BitNet) framework from Microsoft is the official inference framework for **1-bit LLMs**, offering optimized kernels for fast and lossless inference on CPU, with NPU and GPU support planned.
   - This framework supports models like **BitNet b1.58** and includes a suite of optimized kernels that support fast and lossless inference of 1.58-bit models on CPU.
- ****VLM** Security Concerns**: Members discussed how the **VLM** (vision language model) scene is progressing, *specially in the censorship department*.
   - One member pointed out the release of a [less censored version of R1 by Microsoft](https://huggingface.co/collections/microsoft/mai-ds-r1-68003c2753a06be7b9632154), which is vision capable model that can actually process pictures without a puritan filter built in.
- **Members Share Tactics to **Secure LM Studio****: A user inquired about securely exposing **LM Studio** to the internet for remote inference, which is not a secure solution by default.
   - Suggested solutions included using **OpenVPN** or coding custom authentication, with warnings about browser extension security compared to alternatives like Telegram or WhatsApp, which allow user ID-based usage limits.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1364362254451085392)** (36 messages🔥): 

> `RTX 5060 Ti 16GB, Macbook for LLM, 5090 finetuning, DDR3 for AI, Smol models` 


- **Enthusiasts try the RTX 5060 Ti**: A user asked about the **RTX 5060 Ti 16GB** but had issues loading models, and had to update to the latest beta version.
   - Another user bought it in anticipation of this being the last video card I would be able to afford for the next 4 years and switched everything to beta, and it seems to be working fine.
- **Considering Macbook for LLM**: One user asked about using **getupgraded.com** to get a **Macbook** for **LLM** purposes.
   - Another user noted that an **M4 MBP** performs similarly to their **4070 Ti Super**, but cautioned not to expect great battery life.
- **5090 card for finetuning**: A user inquired about using a single **5090** to fine-tune smaller models using **QLoRA**.
   - Another member affirmed the possibility, suggesting **Mistral Small 3.1** with **Unsloth**, but advised against exceeding **7B** parameters with reasonable batch sizes in 4-bit.
- **DDR3 Systems Unfit for AI**: A user shared their hardware specs (Core i3-5010u, 2x4gb DDR3L-1600mhz, Sata SSD, Intel HD Graphics 5500), prompting a comment that their hardware isn't made for local AI.
   - Another user added that **DDR3** systems aren't really designed for **AI** use, and that they could probably run a 1b model but it would be slow.
- **Smol Models for instruct**: A user shared some **smol models** that are *more for instruct than for chat*, linking to [QuantFactory/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF).
   - They noted that these models can provide outputs of **135, 256, 360, 1.7** tokens.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1364317272285646970)** (171 messages🔥🔥): 

> `Gemini, Ollama, Aider Benchmarks, Cursor IDE, Deepseek R2` 


- **Gemini 2.5 Pro's Lovely Errors**: Users discussed [Gemini 2.5 Pro](https://ai.google.dev/) generating code with incorrect formatting, leading to errors and difficulties in committing changes, with one user reporting *810 errors* due to formatting issues.
   - Despite these issues, one user found that **Gemini** fixed a problem in one try after other models failed, leading them to proclaim *gemini is god*.
- **Cursor is productivity bro**: Users are using [Cursor](https://cursor.sh) and reaching the next level of productivity.
   - One user stated he had a Cursor plan before Aider came along, and now uses Aider more but still uses Cursor here and there, using it to convert **Python** code to **C#**.
- **O3 preview smokes O3 - or does it?**: The community noticed that **O3-preview** outperforms **O3** in some benchmarks, despite previous versions showing the opposite trend.
   - There was discussion on whether these [benchmarks](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml#L7) accurately reflect real-world performance, with some suggesting models are specifically tuned for benchmark success rather than practical use.
- **Deepseek R2 release is no laughing matter**: Users jokingly announced the release of **Deepseek R2**.
   - One user even claimed to have seen the release, stating *I just took a massive tudmaybe that's R2 coming out?*
- **OpenAI ID please**: There's discussion around the potential for model services to require you provide an official ID from your organization.
   - One user said that *it's very convenient that they get all your thoughts linked to your passport*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1364350940223897700)** (30 messages🔥): 

> `Aider exclude yes/no responses, Load context readonly, Good model combinations, Aider leaderboard, Gemma 27b image` 


- **Aider Configuration Tweaks: Exclude 'Yes/No' Responses**: A user inquired about configuring Aider to exclude *'yes/no'* responses from `.aider.input.history` to reduce clutter, especially without the original question's context.
   - The user pointed out that these single-character responses are unhelpful and create noise in the history, particularly when the context is saved as read-only.
- **Unlock Read-Only Context Files for Aider Editing**: A user sought guidance on how to edit a context file loaded in read-only mode, expressing confusion about using the `/context` command.
   - The user found the [official documentation](https://aider.chat/docs/usage/commands.html) insufficient for understanding context modes, especially when used without arguments.
- **Model Combination Recommendations for Aider to Minimize Costs**: A new Aider user, transitioning from Cursor, requested advice on cost-effective model combinations to avoid exceeding a **$20/month** budget.
   - A suggestion was made to use [OpenRouter](https://openrouter.ai/) for testing free models, alongside the Aider leaderboard to assess model successes and costs.
- **Solving Aider Startup Issues: API Keys and File Arguments**: A user reported issues starting Aider, accompanied by an image revealing a plain text Deepseek API key, prompting immediate advice to reset the key.
   - The resolution involved deleting the `.aider.conf.yml` file or renaming it, alongside ensuring that files are added using `/add` *after* Aider is initiated, not as command-line arguments.
- **Request Aider Send Full Tree Map With Each Request**: A user inquired about the possibility of configuring Aider to send a full tree map with each request for better context and understanding of the project structure.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1364385489410330657)** (12 messages🔥): 

> `RL Agents, Continuous Signs, Multi-Agent RL, Communication Protocols` 


- **RL Agents Learn Communication with Signs**: A member shared their [paper](https://x.com/superspeeg/status/1914691313318105305) about **RL agents** learning to communicate about their **MDP** with **continuous signs** instead of discrete symbols.
   - The agents learn a **communication protocol** that begins as pictographs and collapses into abstract symbols, similar to the trajectory of many human writing systems, which can be found at [arxiv.org/abs/2502.01568](https://arxiv.org/abs/2502.01568).
- **Continuous Signals and Evolutionary Penalties**: One member inquired about the notion of **similarity between signals** in the new paper, questioning whether some signals might appear particularly similar given the almost contrastive objective.
   - The paper's author replied that their goal wasn't to optimize for the visual aspects of signals per se, but whether they induce the correct action, noting that *mistaking a crab for a spider in the real world could be deadly*.
- **Interest in Passing Information in Multi-Agent RL**: Another member expressed excitement about the paper, stating that they had been wanting to try ideas around **information passing in multi-agent RL** but hadn't had the time.
   - The paper's author responded that there is plenty of room for follow-on work.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1364354875714043934)** (80 messages🔥🔥): 

> `Linear representation hypothesis debunked, Biologically inspired architecture for sequential modeling, Native Sparse Attention analysis, Overfitting models to single datapoints, AI-generated research papers` 


- **Linear Representation Hypothesis Gets Debunked**: A [paper](https://arxiv.org/abs/2402.09268) from Tegmark's group debunks the **linear representation hypothesis**, arguing it's not universal or good.
   - It also mentions that the Mikolov **Glove stuff** is debunked because they use a weird retrieval system of nearest neighbor excluding the original points.
- **O(n) Model Achieves 300k-Length Extrapolation**: A biologically inspired architecture for sequential modeling with **O(n) complexity** was tested on a synthetic task of adding two numbers from a masked sequence and extrapolated to **300k length**.
   - The model, with just **39k params**, maintained a similar MSE loss even on sequences much longer than the training data, and successfully length extrapolated when trained on sequences of length 1000-2500 and validated with a validation set of only 5000 length sequences (as shown in [image](https://cdn.discordapp.com/attachments/747850033994662000/1364540420692115497/image.png?ex=680a0acc&is=6808b94c&hm=ebda36842ecfed21615df2269d5bb4f5121436cbe7fe61513f2ef2fe3250725e&)).
- **Native Sparse Attention Scaling Questioned**: An analysis of **Native Sparse Attention** (NSA) by DeepSeek suggests it doesn't scale to the **1M+ regime**, documented in a [Google Doc](https://docs.google.com/document/d/1kXQ7d-9bSWmAU4c-Tq7iQzzJDWaJe-MuyjI_D3x1Et8/edit?usp=sharing).
   - Discussions revolved around its time complexity and the role of softmax in attention, highlighting how it reduces focus to the highest matches to avoid a mushy 'value' result.
- **Efficient Overfitting Techniques Explored**: A question was posed about the most efficient way to overfit multiple models, each to a different single datapoint in a minibatch.
   - It was suggested that while a learning rate of one and a single forward/backward pass might seem sufficient with high capacity, nonlinearities necessitate multiple steps, though **linearizing the model** could trivialize the problem, with a second order optimizer being able to handle it.
- **AI-Generated Research Papers Incoming?**: Sakana AI released their [AI-Scientist-v2 project](https://github.com/SakanaAI/AI-Scientist-v2), demonstrating the ability to produce a full research paper, including hypothesis and experimental testing, for **$15-20 of API tokens**.
   - This led to speculation about arXiv potentially being flooded with **AI-generated papers**.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1364627981896519812)** (75 messages🔥🔥): 

> `Multihead Latent Attention (MLA), DeepSeek, RWKV architecture, Residual Stream Subspaces` 


- **DeepSeek's MLA Restricts Attention**: DeepSeek's **Multihead Latent Attention (MLA)** limits attention by restricting key and value heads to read and write to a **512-dimensional subspace** of the **7K-dimensional residual stream**.
   - The query heads can read from a separate **1.5K subspace**, aimed at conserving memory bandwidth and potentially improving performance, though some members question whether this is really a subspace.
- **Clarifying Multihead Latent Attention Mechanics**: A member clarified that MLA compresses the entire hidden state via **W^DKV**, not just a subspace, referencing their [research paper](https://arxiv.org/abs/2407.12077).
   - They emphasized that this compression involves a linear combination of all **7168 hidden dimensions**, suggesting it's more than simply selecting a subspace.
- **Space tradeoffs in Hidden State Dimensions**: Discussion centered on whether pre-specifying dimensions for queries and keys might work, particularly in architectures like **RWKV**, where such projections are common.
   - One member noted that it might require taking away space from other important uses, since *hidden state dimensions are kind of a zero sum game*, but it may still be workable given architectural adjustments.
- **MLA Forces Key Value relations**: MLA does not enforce any pre-specified dimensions between Q and K, but instead forces the keys and values to be more related than MHA would.
   - Another added to why the model may use the entire space for each of q,k,v, because *there's kind of just not enough space in the hidden state to accomodate it all*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1364508903353876510)** (1 messages): 

> `PyTorch, flashStream, Kernels, Leaderboard` 


- **FlashStream Kernels for Leaderboard Submission**: A member inquired about the possibility of using a **PyTorch** package called **flashStream** to write kernels and submit them to a leaderboard.
   - No response was given.
- **Using flashStream for leaderboard submissions**: A user asked about leveraging their **PyTorch** package, **flashStream**, to develop and submit kernels to a leaderboard.
   - The query specifically pertains to whether the use of **flashStream** is viable for leaderboard submissions, particularly regarding the creation and submission of kernels.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1364314866562760836)** (11 messages🔥): 

> `Triton FP4 support, FP16 to FP4 conversion, TileLang for FP4, FP4 vs INT4 benchmarks, Pyright and Triton issues` 


- ****Triton** Adds **FP4** Support**: **Triton** now supports **FP4**, detailed in [this tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html), where **FP4** inputs are provided as a tensor of `torch.uint8`s, each storing 2x **FP4**.
- **Transfer **FP16** into two **FP4** via **CUDA** Kernels**: A member inquired about using **Triton** or **CUDA** kernels to transfer **FP16** into two **FP4** and then pack them into **INT8**.
- ****TileLang** Offers Simple and Fast Solution for **FP4****: For converting **FP16** to **FP4**, one member suggested [TileLang](https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py) as a simple and fast solution.
- ****FP4** Benchmarks on **H100** Show Promising Results**: One member shared benchmark results on **H100** **SXM**, stating *"fp16 gemm is around 750T"*, while **FP4** achieved promising numbers by utilizing **TMA** to pipeline dequant on **CUDA** core and **GEMM** on tensorcore.
- ****Pyright** Bug with **Triton****: A member reported issues with **Pyright** and **Triton**, specifically that any `cdiv` marks the rest of the function as unreachable, though code still runs fine.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1364652511834669069)** (1 messages): 

> `RightNow AI, CUDA kernels, browser coding, bottleneck analysis` 


- **RightNow AI V2 hits the Ground Running**: A member introduced **RightNow AI**, a platform designed for coding optimized **CUDA kernels** directly in the browser, and shared the launch of [V2](https://www.rightnowai.co/).
- **Real-time Bottleneck Analysis for Speedy CUDA Kernels**: The AI helps generate fast, profiled kernels with real-time **bottleneck analysis** based on user descriptions.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

marksaroufim: would love some feebdack on https://github.com/pytorch/pytorch/issues/152032
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1364640833017479228)** (1 messages): 

> `MLA kernel, Compute bound inference` 


- **DeepSeek writes up MLA Kernel Update**: A member highlighted a new [blog post](https://github.com/datacrunch-research/blogs/blob/main/deepseek-mla-roofline/deepseek-mla-roof.md) and kernel update by **DeepSeek**.
   - The member was also writing their own blog post about how **MLA** is a *compute bound inference kernel*.
- **MLA compute bound inference**: A member notes that the topic of their blog is regarding how **MLA** is a *compute bound inference kernel*.
   - They linked a new blog post and kernel update by **DeepSeek**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1364335473610330213)** (3 messages): 

> `ncu, import-source, app-range, collective op` 


- **`ncu --import-source yes` command suggested**: A member suggested using the command `ncu` with the flag `--import-source yes`.
   - However, another member responded that when using this flag, the error *Option import-source is not supported during range replay* is thrown.
- **`app-range` is the only mode which works for testing a collective op**: A member stated that the `app-range` is the only mode that works for their use case.
   - They are testing a collective op.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1364317318804406273)** (27 messages🔥): 

> `torch.compile static cache, Qwen2.5-3B performance, fp16 performance, vLLM's compression kernel` 


- ****Static Cache** Doesn't Help **Qwen2.5-3B****: A member reported confusing results when enabling `torch.compile` with `cache_implementation="static"` for **Qwen2.5-3B** on a single **H100**, showing performance degradation compared to *torchao* benchmarks and includes attached [benchmark code](https://cdn.discordapp.com/attachments/1364317318804406273/1364317899619303565/message.txt).
   - The attached code runs via `python3 torchao_test.py --model_name Qwen/Qwen2.5-3B-Instruct --output_csv qwen_3.csv --compile true > qwen_3b_compile.txt`, and it seems that **fp16** is the fastest.
- ****Weight-Only Quant** Beats **Weight&Activation Quant** for Small Batches**: A member highlighted that weight&activation quantization can be slower than weight-only quantization for single batch, possibly due to memory movement overhead.
   - It was explained that activation quantization requires reading activations from global memory, quantizing, and writing back, resulting in more data movement and potential slowdowns for smaller batches.
- ****vLLM** Compression Kernels Revealed**: A member asked about the location of **vLLM**'s compression kernel, seeking to learn from it.
   - Another member responded that **vLLM** has two compression kernels: **Marlin** for non-Hopper GPUs and **Machete** for Hopper GPUs.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1364512163460419585)** (2 messages): 

> `PyTorch ATX Meetup, Triton, Austin, Red Hat, Intel` 


- **Triton Takes Center Stage at PyTorch ATX Meetup**: The next [PyTorch ATX Meetup](https://www.meetup.com/pytorch-atx/events/306856316/?_xtd=gqFyqTI1OTIzMDY4NqFwo2FwaQ%253D%253D&from=ref) will focus on **Triton**, featuring speakers from **Red Hat, Intel, AMD, IBM Research, and UT Austin**.
   - Scheduled for **Wednesday, April 30th, from 5–8 PM** at the **AT&T Hotel and Conference Center (Lavaca Classroom)** in Austin.
- **Denver Area Meetup Inquiries**: A member inquired about the possibility of a meetup in the **Denver area**.
   - This suggests potential interest in expanding the community's reach beyond Austin.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1364378538362408970)** (3 messages): 

> `TileLang, CUDA, Triton` 


- **TileLang vs CUDA/Triton: Choosing Your GPU Path**: A newbie inquired whether to learn **TileLang** instead of **Triton** as an entry point to GPU programming.
   - One member suggested that learning **CUDA** and **Triton** is better for beginners, while another member thinks *TileLang* is good for beginners too.
- **TileLang: Beginner-Friendly or Not?**: The discussion revolves around whether **TileLang** is suitable for beginners in GPU programming compared to **CUDA** and **Triton**.
   - Opinions are divided, with some suggesting **CUDA** and **Triton** are better starting points, while others find **TileLang** accessible for newcomers.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1364507978950246450)** (1 messages): 

> `Tensor Parallelism, Static Split-K` 


- **Understanding Tensor Parallelism with Static Split-K**: It's incorrect to assert that (**batch_size, N, num_heads, headdim**) is faster than (**batch_size, num_heads, N, headdim**); the original data should be understood as (**batch_size, N, d**), with **num_heads** splitting out heads from **d**.
   - The earlier approach was to parallelize at the **TP layer**, placing (**N, headdim**) on each GPU; (**batch_size, num_heads, N, headdim**) is more convenient from each GPU's calculation perspective, since the **headdim** processed by the GPU is continuous, thereby not impacting performance according to the **static split-k** idea.
- **Clarifying Data Handling in GPU Parallelism**: In GPU parallelism, original data is interpreted as (**batch_size, N, d**), where **num_heads** splits heads from **d**, initially used for Tensor Parallelism (TP).
   - The **static split-k** approach places (**N, headdim**) on each GPU, making (**batch_size, num_heads, N, headdim**) more efficient for GPU calculations because **headdim** is continuous, not affecting performance.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1364314972305494161)** (32 messages🔥): 

> `A100 Grayscale, AMD MI300 FP8, AMD MI300 Identity, L4 Grayscale, H100 Grayscale` 


- **A100's Grayscale gets a Boost**: A member achieved **4th place** on the `grayscale` leaderboard with **2.51 ms** on the **A100**.
   - Later submissions reached **2.50 ms** on the **A100**.
- **MI300 Heats Up for FP8-MM**: Multiple members submitted successful runs on the **MI300** for the `amd-fp8-mm` leaderboard, with times ranging from **245 µs** to **9.63 ms**.
   - One member secured **2nd place** with a time of **245 µs**, and another secured **3rd place** with a time of **262 µs**.
- **MI300's Identity Crisis Resolved!**: A member achieved **1st place** on the `amd-identity` leaderboard with a time of **7.69 µs** on the **MI300**.
   - Another achieved **10th place** with **22.6 µs**.
- **L4 Takes the Lead in Grayscale**: A member achieved **1st place** on the `grayscale` leaderboard using an **L4** with a time of **16.2 ms**.
   - T4 runs at **16.2 ms**.
- **H100 Grayscale Personal Bests**: A member achieved a personal best on the `grayscale` leaderboard with **1414 µs** on the **H100**.
   - Later achieving **1409 µs** on the **H100**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1364647936386142258)** (11 messages🔥): 

> `AMD, Code Server Access, Leaderboard, Profiling` 


- **AMD rewards Top 8 Leaderboarders!**: **AMD** will give **code server access** to the top 8 people on the leaderboard.
   - The access will rotate on bigger changes to the standings, so DM <@1151929379492991116> if you want in.
- **Profiling in Code Server**: A member mentioned that profiling wasn't working in this system.
   - Another member followed up saying *"Update: I was just being stupid and did't use the right syntax"*


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1364327105252298862)** (63 messages🔥🔥): 

> `HIP vs Inline, Registration Confirmation Delay, AMD Employee Leaderboard Visibility, Submission File Limitations, Numpy Error` 


- ****HIP vs Inline: A Code Performance Duel****: Members debated whether to use **hip-py** or **inline functions with Torch** for running code in HIP, with initial preference given to inline functions due to familiarity.
   - One member inquired how to compile inline code, and was pointed to the `/leaderboard template` command.
- ****Registration Confirmation Stuck in Limbo****: Participants wondered about the delay in receiving registration confirmation emails after signing up for the competition.
   - A member said registration confirmation will be essential to be recognized for prize money but you can start sending kernels before confirmation.
- ****AMD Employee Leaderboard: Setting Expectations High****: A participant suggested that the leaderboard should clearly identify **AMD employees** to gauge realistic competition.
   - The motivation was to have more hope that theres a chance to get on the top of the leaderboard.
- ****Submission System: One File to Rule Them All****: There were questions about whether the leaderboard allows installing Python libraries (torch extension, pybind11 library) from pypi.
   - It was confirmed that only **a single file submission is encouraged**, with the option to install packages via pip from the submission file itself as stated by a member: *Yes, you can install the package via pip from your submission file, but only one file is supported*.
- ****Numpy Error Strikes Benchmark Code****: A participant encountered a `ModuleNotFoundError: No module named 'numpy'` error during benchmarking, despite not directly calling numpy in their code, but the error trace was hitting in bot code.
   - The issue was attributed to a broken docker image, which has since been fixed.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1364395537880580126)** (3 messages): 

> `Modal.com credits, CUDA fp6 type, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B format` 


- **Modal.com gifts Free Credits**: [Modal.com](https://modal.com/) offers **$30 free credits** every month and charges by the second, making it suitable for testing kernels.
   - It's suggested as a platform for experimentation due to its pay-per-second billing model.
- **CUDA's Odd fp6 Type**: A member inquired about **CUDA's fp6 type** support and its potential for causing memory fragmentation due to its non-divisibility by 8 or 4.
   - Another member stated that the **fp6 support is really odd**, pointing out padding requirements that make it no better than **fp8** in terms of space-saving in gmem, smem, or tmem.
- **CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B Compression?**: A member speculates that the `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` format, with its padding into consecutive 4 bytes, might be better suited to **Inline Compression (ILC)**, potentially reducing **HBM bandwidth**.
   - They noted that this requires using the **virtual memory API** and is not a default feature.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1364314824481439854)** (149 messages🔥🔥): 

> `o4-mini errors, keybindings breaking, Gemini and Claude combos, Cursor slowdowns, Windsurf vs Cursor` 


- **o4-mini throws errors, requests still go through**: Users reported getting error messages on **o4-mini** but the requests still seem to be processed successfully.
   - Some users indicated they experienced this issue after updating Cursor, while others reported it happening without recent updates.
- **Cursor updates cause keybindings to break**: Several users reported that updating Cursor breaks their keybindings.
   - No specific solutions were mentioned, but users acknowledged experiencing the same issue.
- **Gemini and Claude combos: 3.7 is most likely to add what you didn't ask for**: A user found that Google Gemini for planning combined with Claude 3.7 for development leads to unexpected additions and difficulty in fixing bugs.
   - Another member suggested using **Gemini 2.5** for planning instead of 3.7, as 3.7 is more likely to add things the user didn't ask for.
- **Cursor becomes unusably slow**: Several users reported that Cursor has become unusably slow recently, especially with slow requests.
   - It was suggested that the slowdown might be intentional to encourage users to switch to a paid plan and that restarting Cursor or checking VPN/proxy settings might help alleviate the issue, with a link to [Reddit thread](https://www.reddit.com/r/cursor/s/qnmPu2N59m).
- **Windsurf Gains Traction, Cursor Fans Weigh In**: Members discussed the merits of Windsurf vs. Cursor, with some suggesting Windsurf is cheaper while others maintain Cursor has a better UI/UX and is more innovative.
   - A member who tried Windsurf found its tab to be *better then [they] expected at predicting*, and another linked to a tweet about it [Windsurf](https://x.com/heyrobinai/status/1914829284004471099?s=46&t=kUuVqsG2GMX14zvB592G5w).


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1364659731347800180)** (1 messages): 

> `GPT Image 1, Image Generation API` 


- **GPT-Image-1 Launches**: OpenAI launched **gpt-image-1**, making ChatGPT’s powerful **image generation capabilities** available to developers worldwide.
   - Key features include **more accurate, high fidelity images**, **diverse visual styles**, **precise image editing**, **rich world knowledge**, and **consistent text rendering**.
- **Image Generation API Guide**: OpenAI released a guide to start building with the new **Image Generation API** which lets you create images using the **gpt-image-1** model.
   - Check out the [guide](https://platform.openai.com/docs/guides/image-generation) for more information.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1364323859511250944)** (97 messages🔥🔥): 

> `Gemini 2.5 Pro vs Gemini 2.5 Flash, AI replacing jobs, Sora ETA, o3 struggling with high school geometry, ChatGPT app vs webapp` 


- **Gemini 2.5 Pro vs Flash: Which Reigns Supreme?**: Members are pondering whether **Gemini 2.5 Pro** or **Gemini 2.5 Flash** is the superior model.
   - Ultimately, someone chimed in that *you should use all the ai models to get the best result*.
- **AI Job Apocalypse: Fact or Fiction?**: The age-old question of whether **AI will replace everyone's jobs** was posed, again.
   - One user responded that *athletes are the safest right now* as they don't see **AI** playing top cricket or football.
- **Sora's Video Vanishing Act for Newbies!**: Users are reporting that **video generation is temporarily disabled for new accounts** on **ChatGPT Plus**.
   - It was confirmed that this is *on purpose for new users*.
- **Geometry Gymnastics: Can LLMs Stick the Landing?**: It appears that **o3** struggles with **high school geometry**, as do **o4 mini high** and **Gemini 2.5 Pro**.
   - However, it was noted that **Deepseek** solved a particular SAT geometry question correctly, while others did not.
- **App vs Web: Which ChatGPT Reigns Supreme?**: One user expressed that *the **ChatGPT app** is way better than the **webapp** ngl*, adding that they use the **API**.
   - Screenshots were provided showing mixed performance on solving math problems with **ChatGPT o4-mini-high**, which got it wrong but corrected itself when asked to check the answer.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1364323600298934345)** (7 messages): 

> `AI Model Mistakes, Plus Plan Chats and Memories, GPT Image 1` 


- **AI Models Flub Facts Frequently**: One member pointed out that *every single AI model can make mistakes*, urging others to stop repeating the same questions.
   - He complained that the question had already been answered *like 50 times*.
- **Plus Plan Perks Post-Cancellation**: A member inquired about the accessibility of saved memories and chats after cancelling a **Plus Plan** subscription.
   - Another member suggested that while exclusive models might become inaccessible, **chats could be transferred to 4o** on a free account, or pasted directly into the free model.
- **Image 1's Image Generation Ignition**: A member simply asked *Who was able to use gpt image 1?*
   - No responses were provided.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1364485847356936253)** (7 messages): 

> `AI fiction writing assistant, AI for tax assistance, AI Python coding assistant, Defining interesting story prompts` 


- **AI assistant varies prompt based on intended task**: The way a prompt for an AI assistant is written varies based on what the user wants the AI to output such as helping the user **write fiction novels**, **do their taxes**, or **learn Python coding**.
- **AI creates interesting and realistic story prompts**: A user wants an AI assistant to create **interesting and realistic** story prompts across various genres.
   - Another user responded by noting that **'interesting story prompts'** need to be well-defined for the AI, and asked the initial user to define what *they* consider an interesting story.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1364485847356936253)** (7 messages): 

> `AI Story Prompts, Defining 'Interesting' Prompts, Realistic Stories Across Genres` 


- **AI assistant creates interesting story prompts**: A member asked about how to change the usual prompt to get better results and specified they want an **AI assistant** to create **interesting story prompts**.
   - Another member suggested that *knowing exactly what you want the AI to output* is the first step to improving prompts.
- **Defining Interesting story prompts**: A member emphasized the need to **define 'interesting story prompts'** well enough so the desired prompts are described and requested, while undesired ones are avoided.
   - They asked if the original poster could **describe what they consider an interesting story prompt** to differentiate it from uninteresting ones.
- **Realistic stories across genres**: The original poster clarified they want an **interesting story** that can tell various genres and the story is, of course, **realistic**.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1364339828350783569)** (14 messages🔥): 

> `NLM for Exam Prep with Anki, NLM and Client Test Results, NLM German Prompts, NotebookLM Data Training, NotebookLM Long Overviews` 


- ****Anki Ace**: User Preps Exams with NLM**: A user is leveraging **Notebook LM** for exam preparation by uploading **Anki cards** and requesting context, seeking a textbook recommendation in the process.
- ****Patient Patterns**: Advocate Uses NLM for Test Results**: A patient advocate uses **Notebook LM** to analyze client test results, seeking patterns and expanding their list of questions for provider follow-up.
   - They use *descriptions of symptoms or the dx* to search and broaden beyond immediate considerations, leveraging a reference of differentials.
- ****Sprich Schnell**: Cracking German Prompts for NLM**: A user inquired about prompting **Notebook LM** to converse in German, seeking tips on crafting effective prompts.
- ****Privacy Paywall**: NotebookLM Data Training Dilemma?**: A user questioned whether **Notebook LM** trains on user data, recalling information about paid subscriptions offering privacy benefits, linking to the [Google Support page](https://support.google.com/notebooklm/answer/15724963).
- ****Overview Overload**: Achieving Lengthy Summaries on NLM**: A user asked how to obtain long and detailed overviews with **Notebook LM**.
   - Another user clarified that the length of the overviews depends on the sources used, and suggested using a *full report* as a source for managing long outputs.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1364340916088733807)** (75 messages🔥🔥): 

> `NotebookLM math support, Gemini 2.5 Pro, Audio Overview language support, PDF handling in NotebookLM, Grounding and search in AI models` 


- **NotebookLM Struggles with Math and Images, But Hopes Loom!**: Users report difficulties with **NotebookLM** and **math notation** and **image loading**, suggesting it lags behind **GPT-4** in processing formulas, but the team is *working on* fixing it.
   - One user noted weird symbols showed up instead of LaTeX notation , and another user mentioned it might not handle mathematical notation as well as GPT; the team is aware and *is working on it*.
- **Gemini 2.5 Pro Triumphs Over NotebookLM in Reasoning Tasks!**: A user compared **Gemini 2.5 Pro** to **NotebookLM**, finding **Gemini 2.5 Pro** *much better* than **ChatGPT o3** or **o4-mini**.
   - Another user shared that giving **NotebookLM** books and materials on logical and mathematical reasoning made it unable to solve a logic puzzle solved easily by **Gemini 2.5 Pro**.
- **Audio Overview Stuck in English - *Pronto en Español*?**: A user asked if the audio summary feature of **NotebookLM** could generate podcasts in Spanish.
   - The response was *Not right now* indicating language support is currently limited, but could be improved in the future.
- **NotebookLM struggles with Large PDFs, Splits Advised!**: Users are encountering issues with **NotebookLM** stopping midway through lengthy PDF documents.
   - The suggested workaround is to split the PDF into smaller segments using tools like [iLovePDF](https://www.ilovepdf.com/pt/dividir_pdf).
- **Grounding Models, Not Models Grounding!**: Members clarified that **AI models** themselves don't search the web, but can be *grounded* with access to a search API like **Google Search** to update knowledge or verify information.
   - It was explained that tools like **AI Studio** allow developers to enable search capabilities (*grounding*) for models like **Gemini 2.5 Pro**, with attached screenshot example [here](https://cdn.discordapp.com/attachments/1124402182909857966/1364684264272171129/Screenshot_2025-04-23_at_21.25.58.png).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1364346218381312100)** (67 messages🔥🔥): 

> `Agents vs Humans, Hugging Face Spaces Issue, Llama 3 Chat Template, Fine-tuning Llama for VTuber, Continuous Pretraining Datasets` 


- **Agents Flounder, Humans Flourish!**: A member argued that AI **agents** are overhyped and ineffective because *in 99.9% of the time the human is cheaper and faster and actually reliable*, citing issues like dynamically generated workflows and cascading compound errors.
   - Another member offered free credits to try out their agent-based system for feedback, but the first member declined, expressing interest only in audio research with agents.
- **Hugging Face Space gets Trashed, Infra Team Tasked!**: Users reported issues with their **Hugging Face Spaces** suddenly going offline, as seen in [this discussion](https://discuss.huggingface.co/t/my-space-suddenly-went-offline-the-cpu-cannot-restart/151121/22), with the infrastructure team working to resolve the underlying cause.
   - The team confirmed the issue was resolved and restarting the Space should fix it.
- **Llama 3 Loses its Loadstar!**: A member encountered issues with **Llama 3's** output on a Windows PC, suspecting a problem with the chat template or version mismatches.
   - Another member identified that the incorrect chat template was used and directed the member to use the format `{'role': 'user' , 'content': message }`.
- **Vtuber Virtuosos Vexed!**: A member inquired about fine-tuning **Llama** for **VTuber** text generation, seeking guidance on the process.
   - Another member suggested that prompting alone is often sufficient for VTuber content and pointed to creating synthetic data.
- **Pretraining Performance Predicament!**: A member asked for dataset recommendations for continuous pretraining on **smolLM** after observing performance degradation on the hellaswag benchmark using the **cosmopedis-v2** and **fineweb-edu-dedup** datasets.
   - They trained for **5k steps** with a **batch size of 128** and **max seq length of 512**.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1364620793220305026)** (1 messages): 

> `Pomodoro Technique, Time management` 


- **Pomodoro Technique Is Like Workout Reps**: A member likened the **Pomodoro Technique** to doing reps in workouts.
   - They shared that using it helps to get *so much more output done*.
- **Time management for productivity**: The user shares **Pomodoro technique** is effective.
   - It can increase output significantly.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1364587309944668191)** (3 messages): 

> `Model Size Disclosure, YouTube Channel for Fine-tuning Tutorials` 


- **Pretraining Compute Amount Clarified**: A member clarified that their earlier comment about *model size* was inaccurate, and they meant to refer to the **amount of compute** spent for pretraining.
   - This clarification followed a disagreement about whether model sizes are typically disclosed.
- **ML Enthusiast Promotes YouTube Channel**: An ML enthusiast promoted their **YouTube channel** *Let's Fine-tune Everything*, which features hands-on tutorials and guides on fine-tuning open-source models for real-world use cases.
   - The channel covers topics from **object detection** to **LLMs**, targeting both beginners and experienced practitioners.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1364315359435292892)** (5 messages): 

> `AI-Powered Document Q&A Project, LLM Fundamentals for Cybersecurity, Resume Matching App` 


- ****Document Q&A** Project Open-Sourced**: A member open-sourced a small project focused on **AI-powered document Q&A** with a **FastAPI** backend, using a retrieval-based approach with embedding models.
   - The developer seeks feedback on **architecture, code quality, scalability**, and suggestions for improvement, see the [repo and post](https://www.linkedin.com/posts/hamzabouajila_ai-opensource-python-activity-7320494631471771648-z9wE) for more.
- **Cybersecurity meets **LLM Fundamentals****: A member shared a [blog post](https://x.com/dazzyddos/status/1914895119675007252) breaking down **AI/LLM fundamentals for cybersecurity** professionals, focusing on *why* prompt injection happens.
   - The goal is to explain *why* prompt injection occurs, rather than just detailing *what* it is or *how* it works.
- ****Resume Matching App** Goes Live**: A member announced that the **resume matching application** is now live at [match-your-resume.fyi](https://match-your-resume.fyi/).
   - This follows up on a previously shared [GitHub repository](https://github.com/waefrebeorn/bytropix) related to the project.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1364558627452620832)** (1 messages): 

> `Embedding Models for Short Contexts, QA Embedding Pairs, Context Length Optimization` 


- **Embedding Models for Short Token Chunks: A Quest Begins**: A member sought recommendations for embedding models optimized for use cases with chunk sizes of **<=1024 tokens** and QA pairs of **<512 tokens**.
   - The user questioned the suitability of models trained on **8k context** windows, suggesting they might not be ideal for shorter contexts.
- **QA Embedding Pairs: A Deep Dive**: The discussion focused on selecting the right embedding model that can handle Question-Answer embedding pairs efficiently.
   - Consideration was given to how well different models perform when encoding both the question and its corresponding answer, especially within the given token limits.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1364337810550820965)** (8 messages🔥): 

> `Arxiv paper deadline extension, Loops and Branches Notebook Bug, Agents course joining, HuggingFace course credit issue` 


- **Arxiv Deadline Extended 'Til July**: The deadline for a paper on [Arxiv](https://arxiv.org/pdf/2311.12983) has been extended to **July 1st**.
- **Looping Bug Discovered in Notebook**: A user found a bug in the [Loops and Branches Notebook](https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/workflows.ipynb) related to the **LoopEvent** parameter being incorrectly placed in **Step_two** instead of **step_one**.
- **Agents Course: Enrollment Instructions Given**: A user asked how to join the Agents course and another user directed them to the [instructions](https://huggingface.co/learn/agents-course/en/unit0/introduction).
- **Insufficient Credits Causes HuggingFace Course Problems**: A user reported getting an error while following the **Agents course** and wondered if it was due to their free account or a configuration issue with **Google Colab** due to insufficient credits.
   - They attached an [error log](https://cdn.discordapp.com/attachments/1329142738440028273/1364693372442509393/errorUseInferenceClient.txt?ex=680a993e&is=680947be&hm=18162df5d939e62e881916750257fcb12e6b7ee6fd31a90b3d412ba650809f2b&) for debugging.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1364326801786273813)** (72 messages🔥🔥): 

> `Brain processing locality, Saturday Paper Discussions recordings, Anthropic's recent paper, Hebbian theory vs Brain Physics, mental model vs world model` 


- ****Brains Process Locally****: Discussion around the brain's processing being predominantly **local**, with neurons receiving signals from directly wired neighbors, and having local internal processes that are shaped by their position, connectivity, and context.
   - One member said, *"neurons have localized dynamics, means that each neuron has local view of itself and of neurons nearby its wired with. Also, it's not mere sequential"*, while another proposed **quantum non-local information processes in cytoskeletal microtubules**.
- ****Paper Discussions Unrecorded****: A member inquired about recordings of the Saturday Paper Discussions, specifically hoping to hear the discussion about **Anthropic's recent paper**.
   - Another member responded that there is never any warning that they are being recorded, while another pointed to [Yannic's video on Anthropic's paper](https://www.youtube.com/watch?v=mU3g2YPKlsA), which the original poster had already watched.
- ****Mental Models vs World Models Debated****: Discussion explored the concepts of **mental models** (internal simulations or expectations about specific things) and **world models** (broader, long-term representations of how the world works).
   - A member suggested the brain builds mental models to predict and compare with reality, refining them based on experience, using concepts from the [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle) and [Predictive Coding](https://en.wikipedia.org/wiki/Predictive_coding).
- ****Transformer Tricks and Self-Attention Tactics****: A member asked if *x = self.transformer(x, x)* is reasonable in the forward pass, as a tactic to throw in *some transformers lmao* to make their *ai better*.
   - Others explained that it is often done when **self-attention** is needed and can act as a trick or bias, linking to an [IBM article on self-attention](https://www.ibm.com/think/topics/self-attention), and advising to consider *“ϵ-greedy” exploration* instead, pointing to the value of [randomness and exploration](https://spectrum.ieee.org/2d-semiconductors-molybdenum-disulfide).
- ****Ditching PyTorch, Longing for Optimization****: One member expressed dissatisfaction with **PyTorch's interface**, calling it *such an ungodly undivine interface*, and inquired about alternatives.
   - They mentioned liking **Unsloth** for its ease of use but wondered about carrying over optimizations, leading to a comical quip that the performance could be chalked up to the *dev team's cocaine budget*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1364414011508260874)** (3 messages): 

> `Muon, Adam replacement, reverse Distillation` 


- **Muon Replaces Adam**: The community will be discussing **Muon**, a faster replacement for **Adam** on Wednesday.
   - A blogpost about [Muon](https://kellerjordan.github.io/posts/muon/) was also shared in the channel. The [ArXiv link](https://arxiv.org/abs/2310.11453) about WaveFunction was also mentioned.
- **Reverse Distillation?**: A member asked if **Muon** is a reverse distillation method.
   - It was requested that an event be created to further discuss the **Muon** paper.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1364327168569512028)** (3 messages): 

> `Links to YouTube videos` 


- **Linked YouTube Videos Spark Discussion**: Several members posted links to YouTube videos, including [7GF78YQz62w](https://youtu.be/7GF78YQz62w), [K9anz4aB0S0](https://youtu.be/K9anz4aB0S0), and [1W_mSOS1Qts](https://youtu.be/1W_mSOS1Qts).
   - Without more context, it's difficult to ascertain the specific subject or relevance of these videos.
- **Additional YouTube Links Await Context**: More YouTube links were shared, but without accompanying discussion, their specific relevance remains unclear.
   - Further information is needed to determine the topics covered in these videos and their connection to the ongoing conversation.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1364323720818200697)** (8 messages🔥): 

> `Zed Project Diagnostics, Modular Meetup, MAX/MOJO License` 


- **Zed's Project Diagnostics Feature Praised**: Members discussed Zed's **Project Diagnostics** feature, accessible via **⇧⌘M** or by clicking the error icon in the lower left, for quickly identifying and editing errors in place.
   - One member said it's both *convenient and motivating to be able to quickly make changes and see the outstanding error/warning count go down to zero*.
- **Modular Meetup Happening**: A member announced the [Modular Meetup](https://lu.ma/modular-meetup) at their office in Los Altos, with limited in-person spots available.
   - The talks will also be livestreamed on [YouTube](https://www.youtube.com/watch?v=uul6hZ5NXC8) and [LinkedIn](https://www.linkedin.com/events/next-gengpuprogramming-hands-on7319044981682270210/).
- **MAX/MOJO License Questioned**: A member questioned the business perspective of the **MAX/MOJO** license, specifically regarding *Production / Commercial on Other Accelerators*.
   - They speculated if it's a strategy to gather feedback for developing **MAX/MOJO** on non-NVIDIA GPUs.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1364316145871618088)** (43 messages🔥): 

> `Mojo Training Pipelines, Mechanical Migrator Tool, Pythonic Mojo Design Tradeoffs, Zero-Cost Abstraction in Mojo, Enums in Mojo` 


- **Community Seeks Training Pipeline Examples**: Members are curious about using **Mojo** in training pipelines, despite it not natively supporting training yet, particularly for upfront data manipulation before calling training code via **PyTorch**.
   - The community wonders if any community members have published examples of training pipelines leveraging **Mojo**, even in its early stages.
- **Mechanical Migrator Mandates and Compiler Tests**: It was discussed that the mechanical migrator tool might be essential due to the sensitivity of compiler tests; even adding one keyword could break them.
   - One member opted to script a solution instead, referencing a [script](https://github.com/bgreni/Kelvin/blob/main/scripts/run_reject_tests.py) on GitHub.
- **Pythonic Mojo Debates Feature Tradeoffs**: There's an ongoing discussion about the design tradeoffs for **Pythonic Mojo**, balancing dynamic features with performance approaching **Go** or **Rust**.
   - Concerns arise about whether the inclusion of numerous dynamic features might compromise the speed of **Pythonic Mojo**.
- **Zero-Cost Abstraction Pondered**: Members debated whether dynamism can maintain zero-cost abstraction, referencing **Swift** as a precedent for managing complexity and dynamism.
   - It was stated that *zero-cost isn't about not having overhead, it is about not having overhead for what you don't use.*
- **Community Navigates Enum Alternatives**: As **Mojo** lacks dedicated enums, members explored using **DType** (similar to an enum) or **utils.Variant** for unions.
   - A link to the [DType implementation](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/dtype.mojo) was shared as a reference.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1364330928507326514)** (39 messages🔥): 

> `Brainy RTX 4090 AI supercomputer, OAI image gen in API, Tinybox competitor, Scout.new cooking` 


- **Autonomous.ai debuts Brainy, the RTX 4090 AI Supercomputer**: [Autonomous.ai](https://www.autonomous.ai/robots/brainy) launched **Brainy**, an **RTX 4090 AI supercomputer** with impressive O3 agent UX, breaking down image analysis.
- **Scout.new is straight fire!**: A member shared **Scout.new**, but noted it was broken due to load, with others saying it's *fucking cooking hot damn*.
   - A member posted link to [X cancel](https://xcancel.com/rayfernando1337/status/1914791594789879844) for the **Ray Fernando** post.
- **OpenAI's Image Generation Now Available in API**: **OpenAI** has released **image generation** capabilities in their [API](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1) using **gpt-image-1**.
- **Microsoft Announces Copilot Agents**: **Microsoft** has announced [Copilot Agents](https://x.com/satyanadella/status/1915098359251247392).


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new lightning pod https://youtu.be/aDiEQngFsFU
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1364374336181637189)** (2 messages): 

> `LlamaIndex Milvus full-text search, Agentic Document Workflow` 


- **LlamaIndex Adds Milvus Full-Text Search**: LlamaIndex's integration with [@milvusio](https://github.com/milvus-io) now supports **full-text search with BM25** for hybrid search in **RAG pipelines**, combining vector search and keyword matching.
   - Check out the integration tutorial [here](https://t.co/0dCi0kEn6o).
- **Agentic Document Workflow Unveiled**: **Agentic Document Workflow (ADW)** scales better, integrates with your existing software ecosystem, and provides better error handling compared to the **RAG chatbot**.
   - ADW is described as the logical next step beyond the **RAG chatbot** prototype; more details [here](https://t.co/ZZzr7scHhF).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1364328796060254338)** (35 messages🔥): 

> `LlamaParse getText() issue, Document hash computation, Passing userID to MCP tools, MLflow autolog with Llamaindex and FastAPI, Workflows Checkpoints usage and alteration` 


- **LlamaParse's Text Tussle: Markdown vs. Text**: A member encountered an issue with **LlamaParse**, where `getText()` in next.js returned partial content when `resultType` was set to `markdown`, but worked fine with `text`.
   - It was identified as a **markdown vs text comparison issue**, and switching to `const reader = new LlamaParseReader({ resultType: "text" });` solved the problem.
- **Document Hash Hues: Metadata Exclusion**: A member inquired about excluding a metadata key (specifically a timestamp) from the document hash computation to avoid deduping issues when the timestamp changes.
   - It was clarified that while `TextNode` considers metadata in hash computation, the `Document` object (used in the ingestion pipeline) does not, allowing the member to safely add a timestamp to each document object’s metadata.
- **MCP Tool's userID Injection Challenge**: A member sought a method to efficiently pass a `userID` to tools registered on the **MCP server** within an agent workflow using a **React agent**.
   - Appending `userID` to the user's query was a working workaround but a cleaner, more standardized solution was desired.
- **MLflow's Autolog Adventures in FastAPI's Parallel Universe**: A member faced issues with **MLflow autolog** capturing **LLM call traces** when running a **LlamaIndex Workflow** inside **FastAPI's background task** with parallel tasks.
   - Traces were inconsistently captured, leading to a warning: *'NoneType' object has no attribute 'info'*, suggesting an **MLflow-specific problem**.
- **Workflow Checkpoint Conundrums: Altering Events After the Fact**: A member questioned the recommended way to alter a **Checkpoint** to start a **Workflow** from a given Checkpoint but with a modified **Event** passed into the Checkpoint's step.
   - The recommended approach involves restructuring the workflow to include a **human-in-the-loop step** for approving or changing key outputs, avoiding the need to alter existing state; altering Checkpoints directly is *not* a supported feature.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1364678759910998156)** (3 messages): 

> `Instruction-Finetuning LLMs, TRL for Finetuning, Memory Constraints for LLMs` 


- **TRL recommended for Instruction-Finetuning**: A member recommended using **TRL (Transformers Reinforcement Learning)** and linked to the [Hugging Face TRL documentation](https://huggingface.co/docs/trl/en/index) for instruction-finetuning an open-source LLM instead of using LlamaIndex tools.
   - They suggested creating a dataset using any existing LLM to distill training into another.
- **Memory Constraints when training LLMs**: A member warned that training locally or on a **T4 GPU** will be very memory constrained, and you'll likely only be able to train very small LLMs on small batch sizes.
   - null


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1364355774473703465)** (26 messages🔥): 

> `MCP Interview, README Translation Automation, AWSLab Cost Analysis MCP Server Issue, MCP Inspector Timeout Error, Cursor MCP Tool Error` 


- ****MCP Interview** Compensation Offered**: A member is [paying **$40** for a **30 min interview**](https://discord.com/channels/1119947416775741521/1119947417233033328) for people who have worked with **Claude Computer Use** and/or **OpenAI computer-use-preview** in a real project, with bonus points for those implementing **MCP**.
   - The member has *a million questions* to ask about users' experiences.
- ****README Translation** Automation Proposed**: A member proposed storing all links, tags, and emojis in a single **JSON** file to automate the generation of translated **READMEs** via the **CI pipeline**.
   - This would keep maintenance focused on one place and save a lot of effort by updating only the primary **README**.
- ****AWSLab** Cost Analysis **MCP Server** Freezes**: A member reported that the **Claude Windows desktop app** freezes and shows an error when generating the cost report for last month using the **AWSLab cost analysis MCP server**.
   - The error message displayed is *Claude’s response was interrupted. Please check your network connection or contact support if the issue persists.*, despite a stable internet connection.
- ****MCP Inspector** Request Timeout Issue**: A member is encountering an **MCP error -32001**: *Request timed out* when running the basic **MCP server** from the docs on **GitHub** and using the interactive server.
   - The error prevents them from running any tools, despite it working correctly in **Claude desktop** when running `mcp install test.py`.
- ****Cursor** Gives Error Calling **MCP Tools****: A member reported that while **Cursor** recognizes all **MCP tools**, it throws an error when trying to call them, unlike **Claude Desktop** and **Cline**.
   - The member is seeking advice from others who may have encountered a similar issue with **Cursor**.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1364315882150559856)** (4 messages): 

> `MCP Server, Klavis AI Eval Platform, Browser Extension for MCP, Siloed AI Drag and Drop` 


- **Defang Labs Ships Vibe-Coded MCP Server**: Defang Labs built an MCP server that lets you **deploy your vibe-coded project from any IDE straight to the cloud** and are asking for feedback [on their LinkedIn post](https://www.linkedin.com/posts/defanglabs_vibecoding-defang-devtools-activity-7320490826004852737-2IFE?utm_source=share&utm_medium=member_desktop&rcm=ACoAACNoYXgBadWv4CWLbcKhgSGxWjdmu9e5dFI).
- **Klavis AI Launches Customized MCP Testing and Eval Platform**: Klavis AI announced **early access to their customized MCP testing and Eval Platform** to test, evaluate, and compare different MCP servers and are asking users to contact them at connect@klavis.ai or go to [their website](https://www.klavis.ai/mcp-testing-eval).
   - According to Klavis, it's hard to tell which MCP is more production ready, has more features, and is more stable than the others.
- **Browser Extension Connects Tools to AI Chat via MCP**: A customizable AI chat side panel is available as a **browser extension that connects tools to it via MCP** and can be found on [the Chrome Web Store](https://chromewebstore.google.com/detail/browsewiz-ai-assistant-ai/ioohfnlbpolaalcbppaggpgcgpldohfg).
- **Siloed AI enables drag-and-drop resources for MCP on the web**: [Siloed](https://siloed.ai) brings **MCP to the web with drag and drop resources**, connect your favorite MCP server, and paste anywhere on the web, and also allows you to build a library of prompts with dynamic text + resource attachments.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1364319640552996936)** (21 messages🔥): 

> `arithmetic shift right op, UPat matching a CONST, multiple patterns matching, instruction ordering and register assignment, closures reconsideration` 


- **Arithmetic Shift Right Op Questioned**: A member inquired about the existence of an **arithmetic shift right op** within the system.
- **Immediate UPat Matching Requested**: A member asked for a way to create a **UPat** to match a **CONST** where the immediate is only, for example, **5 bits long**, or where the bottom **n bits are zero**.
- **Pattern Matching Priorities Clarified**: When **multiple patterns match** in the rewriter, the patterns are applied in the **order of the list** with no way to prioritize some over others.
- **Constraint Solver Backend for Instruction Ordering**: The team is moving towards a **constraint solver backend** that jointly handles **instruction ordering** and **register assignment**.
- **Reconsidering Closures is on Hold**: The team is *not reconsidering core decisions* about not having closures until a member has landed **10 PRs**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1364384271116144720)** (6 messages): 

> `Arange Optimization, Indexed Operations for STs, UOps and Buffers relationship` 


- **Arange Gets Optimized Out**: It was mentioned that `arange()` gets optimized out, according to [this tinygrad notes link](https://xl0.github.io/tinygrad-notes/arange.html).
- **Indexed Operations: Finding Byte Indices**: A member suggested finding the byte indices by getting the **indexed_ops** of both the **STs** (ShapeTracker) and then plugging in the tensor indexes *i,j,k*.
   - Another member referenced [device.py](https://github.com/tinygrad/tinygrad/blob/6cb2d18c034fc3fb8c8c7521716c04a4674c5504/tinygrad/device.py#L330).
- **UOps and Buffers Relationship Explored**: Issue #10006 attempts to describe the relationship between **UOps** and **Buffers**.
   - This might be useful for solving the bounty, *"Make sure buffers are GCed on CPU and with VIZ, with good tests."*'


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

dbreunig: https://www.dbreunig.com/2025/04/18/the-wisdom-of-artificial-crowds.html
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1364629921372045342)** (22 messages🔥): 

> `DSPy 3.0, Synthetic Flywheel, Prompt Optimization, Databricks event SFO` 


- **DSPy 3.0 reveal has Hype**: Members expressed excitement for **DSPy 3.0** with comments like *"Super excited!!!"*, but one user asked *"What can we expect??"* regarding a [tweet](https://x.com/lateinteraction/status/1915058777491145200).
- **Vision for DSPy 3.0 is veiled in secrecy**: A member inquired about the unifying vision/design for **DSPy 3.0**, but was told that the unifying vision/ design is not written anywhere, because *"too many things that are internal research until they're not!"*, linking to the [roadmap](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md).
- **DSPy 3.0 to drop in June 2025**: When asked for an ETA on **DSPy 3.0**, a member replied *"June 2025"*.
   - Another guessed that the reveal will be around **Databricks event in SFO**.
- **Synthetic Flywheel is about to fly**: Two members discussed making a *"synthetic flywheel so fly its gonna need airspace clearance."*
- **DSPy's Prompt Optimization feels like black magic**: One user who bet on **DSPy** for their gen-ai development a year ago, now feels that *"it was not the right thing to do"*, because *"Prompt optimization seems a bit black box."*


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1364327634812801044)** (8 messages🔥): 

> `RoPE implementation, Collective scheduling, Tune cp workflow, Library design` 


- **Dedicated RoPE class sparks discussion**: A member inquired why **RoPE** (**Rotary Position Embedding**) is implemented as a dedicated class in torchtune, with the initial response being that it felt *more PyTorch-y* than using a function.
   - A follow-up explained that the **RoPE cache** only needs to be initialized once and doesn’t need to be recomputed each time, so it requires some state, trading off speed for memory.
- **Collective Scheduling Customization Underway**: A member is testing the throughput and memory usage of customizing the **collective scheduling** and plans to submit a PR if the results are promising.
   - They are considering whether to keep arguments like `fsdp_delay_all_reduce` or switch to single-word descriptors aligned with **deepspeed stages (zero 1-3)**.
- **`tune cp` Workflow User Journey**: A member detailed their experience using the `tune cp` workflow on a Macbook, highlighting issues such as needing to manually search for recipe and config files, remove file extensions, and resolve dataset version mismatches, but ultimately achieving success after addressing **macOS-specific issues**.
   - The member also expressed that while the workflow isn't terrible, it relies heavily on *massive code duplication* which feels off.
- **Hybrid Library Design Sparks Debate**: Discussion arose around the hybrid library design approach in torchtune, which aims to provide user scripts that are easy to customize while leveraging a library for common components.
   - The motivation behind this approach was to avoid maintaining entire forks of a library just to customize a few modules, with the goal of allowing researchers to showcase only the code that matters for them, but the team is looking for whether the *hybrid design* is a fundamentally flawed approach or a user education/documentation issue.


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1364431029523189780)** (1 messages): 

> `Future Meeting` 


- **Scheduling a Call Post-Singapore**: A user mentioned their availability for a call starting late next week, as they will be returning from Singapore.
- **Availability Update**: The user provided a specific timeframe for scheduling a call, indicating they will be available after their return from Singapore late next week.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1364394089746796595)** (8 messages🔥): 

> `Tool Integrations with SaaS Platforms, New Model Release, SuperNova Models by Arcee-AI` 


- **Engineers seek guidance on SaaS Tool Integrations**: A member inquired about preferred tools for building integrations with existing **SaaS platforms**, specifically seeking alternatives to **Zapier** that support multiple connections for different customers.
   - They suggested **Composio** as a potential solution, seeking community input on its suitability or alternative recommendations.
- **Nous teases mystery release for Red Team**: Nous hinted at an upcoming release, scheduled for today or tomorrow, that is tailored for the **red team** community, potentially with *new mix precision quantlol*.
   - The announcement generated anticipation among members interested in novel tools and resources for security and adversarial testing.
- **SuperNova Models Earn Community Accolades**: Members expressed appreciation for the **SuperNova models** by **Arcee-AI**, citing their strong performance relative to their size.
   - One member noted that the two **SuperNova Models** are now their default models since release.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1364518753815105606)** (3 messages): 

> `Resource Submission Form, Team Name` 


- **Resource Submission Confirmation Sought**: A member inquired about receiving a confirmation email after submitting a resource submission form, seeking confirmation from a specific user.
   - The member indicated that they did not receive any email confirming their submission despite completing the form.
- **Team Name Elicitation**: A member asked another member about their team name.
   - The second member responded with *"IbiA"*.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1364584025204981800)** (2 messages): 

> `MOOC readings, LLMAgents-learning.org` 


- **Readings Posted for LLM Agents MOOC**: A member asked about readings, and another member posted that the readings for the **LLM Agents MOOC** are available on the website: [llmagents-learning.org/sp25](https://llmagents-learning.org/sp25).
   - The readings are presumably relevant to the course content and assignments.
- **LLM Agents MOOC Website**: The official website for the **LLM Agents MOOC** is [llmagents-learning.org/sp25](https://llmagents-learning.org/sp25).
   - This site likely contains course materials, assignments, and other important information.


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1364691798433206422)** (1 messages): 

> `Hugging Face Inference API, Flask Website Integration, Model Deployment` 


- **HF Inference API connects to Flask Website**: Models uploaded to **Hugging Face** and using their paid inference API can be connected to a website built with **Flask**.
- **Flask calls HF inference endpoint**: The **Flask** application sends requests to the **Hugging Face Inference API** endpoint with the input data.
   - The API returns the model's prediction, which is then displayed on the website.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1364691934597218506)** (2 messages): 

> `Hugging Face, Flask, Model uploading` 


- **Flask asks: Hugging Face Paid Inference**: A member asked how to connect a **Flask** website to a model uploaded on **Hugging Face** using their paid inference API.
- **New Member Introductions Encouraged**: New members are encouraged to introduce themselves by sharing their company/industry/university, what they're working on, favorite tech/tools, and what they hope to gain from the community.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1364416763147583519)** (3 messages): 

> `Debugging Handler Errors, Code Modification Suggestions` 


- **Handler Errors Plague System**: A member reports that *something from your handler is erroring* in the system.
   - Another member suggests changing [the error-catching code](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286) to throw errors instead, to help **debug the issue**.
- **Debugging Advice Shared**: One member suggests modifying the [code](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286) to throw errors instead of catching them.
   - They advise running generation for **one entry** to see the **full trace**.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1364521994921709588)** (1 messages): 

> `Legislative AI/Tech Webinar, BillTrack50, AI4Legislation competition` 


- **Legislative AI/Tech Webinar on April 28**: Entrepreneur Karen Suhaka (Founder of BillTrack50) is teaming up with Silicon Valley Chinese Assocation Foundation to deliver a webinar on legislative applications of AI and Technology on **Monday, April 28 at 12pm Pacific**.
   - The webinar will cover building legislative technology, navigating ethical considerations, and tips for entrepreneurship; registration is available at [this link](https://forms.gle/v51ngxrWdTsfezHz8).
- **BillTrack50 Case Study**: Karen Suhaka will share her insights on her own legal tech company, **BillTrack50**, as a case study from starting up to scaling and customer feedback.
   - She will focus on identifying a need, choosing your data and method.
- **AI4Legislation Competition**: The webinar will include project ideas for the **Summer 2025 AI4Legislation competition**, with details found on [GitHub](https://github.com/svcaf/2025-AI4Legislation-Public/tree/main).
   - This competition aims to leverage recent advances in **LLMs** and **NLP** to benefit citizens and voters.


  
