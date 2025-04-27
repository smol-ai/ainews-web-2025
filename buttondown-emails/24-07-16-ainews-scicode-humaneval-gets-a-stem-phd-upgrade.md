---
id: 6c90b0ac-22eb-4f4b-92d1-4a9d069e7cd1
title: 'SciCode: HumanEval gets a STEM PhD upgrade'
date: '2024-07-17T02:04:35.319219Z'
original_slug: ainews-to-be-named-5745
description: >-
  **PhD-level benchmarks** challenge leading LLMs like **GPT-4** and **Claude
  3.5 Sonnet**, which score below 5% on the new **SciCode** benchmark designed
  for scientific coding problems from advanced papers, including Nobel-winning
  research. **Anthropic** doubled the max output token limit for **Claude 3.5
  Sonnet** to 8192 tokens in their API. **Q-GaLore** enables training
  **LLaMA-7B** on a single 16GB GPU using low precision and lazy layer-wise
  subspace exploration. The **Mosaic compiler** now generates efficient code for
  **NVIDIA H100** GPUs, showing convergence of AI accelerators. The **Dolphin
  2.9.3-Yi-1.5-34B-32k-GGUF** model on **Hugging Face** has over 111k downloads
  last month. **Llama 3** achieves 90% zero-shot accuracy on the **MATH**
  dataset. Discussions continue on the limitations and forms of synthetic data
  for training.
companies:
  - anthropic
  - hugging-face
  - nvidia
  - mistral-ai
models:
  - gpt-4
  - claude-3.5-sonnet
  - llama-7b
  - llama-3
  - dolphin-2.9.3-yi-1.5-34b-32k-gguf
topics:
  - benchmarking
  - fine-tuning
  - model-training
  - model-performance
  - gpu-optimization
  - synthetic-data
  - code-generation
  - model-optimization
  - reinforcement-learning
people:
  - rohanpaul_ai
  - alexalbert__
  - tri_dao
  - awnihannun
  - abacaj
---


<!-- buttondown-editor-mode: plaintext -->**PhD-level benchmarks are all you need.**

> AI News for 7/15/2024-7/16/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**466** channels, and **2228** messages) for you. 
Estimated reading time saved (at 200wpm): **248 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Lots of small updates here and there - [HuggingFace's SmolLM](https://x.com/xenovacom/status/1813258097185448377) replicated MobileLLM ([our coverage](https://buttondown.email/ainews/archive/ainews-to-be-named-3686/) just a week ago), Yi Tay wrote up the [Death of BERT](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising) ([our podcast](https://x.com/latentspacepod/status/1809300018907828285) 2 weeks ago), and [1 square block of San Francisco](https://x.com/evanjconrad/status/1813297376544854063) raised/sold for well over $30m in deals across [Exa](https://x.com/evanjconrad/status/1813308202534211998), [SFCompute](https://x.com/evanjconrad/status/1813293874288472493), and [Brev](https://x.com/NaderLikeLadder/status/1813286240093151412) (congrats friends!).

However our technical highlight of today is [SciCode](https://x.com/MinyangTian1/status/1813182904593199553), which challenges LMs to code solutions for scientific problems from advanced papers. The challenges were crafted by PhDs (~10% is based on Nobel-winning research) and the two leading LLMs, GPT-4 and Sonnet 3.5, score <5% on this new benchmark.

 ![image.png](https://assets.buttondown.email/images/66dacbe0-3e49-4861-9e66-b94e19afe531.png?w=960&fit=max) 

Other than HumanEval and MBPP, the next claim to a top coding benchmark has been SWEBench ([more info on our coverage](https://www.latent.space/p/iclr-2024-benchmarks-agents), but it is expensive to run and more so an integration test of agentic systems rather than test of pure coding ability/world knowledge. SciCode provides a nice extension of the very popular HumanEval approach that is easy/cheap to run, and nevertheless still is remarkably difficult for SOTA LLMs, providing a nice gradient to run. 

Nothing lasts forever ([SOTA SWEbench went from 2% to 40% in 6 months](https://x.com/Douglas_Schon/status/1813213722770354177)) but new and immediately applicable benchmark work is very nice when done well.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**AI Model Developments**

- **Anthropic API updates**: [@alexalbert__](https://twitter.com/alexalbert__/status/1612921642143900036) noted Anthropic doubled the max output token limit for Claude 3.5 Sonnet from 4096 to 8192 in the Anthropic API, **just add the header "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15" to API calls**.
- **Effective Claude Sonnet 3.5 Coding System Prompt**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1612973162906460460) shared an effective Claude Sonnet 3.5 Coding System Prompt with explanations of the guided chain-of-thought steps: **Code Review, Planning, Output Security Review**.
- **Q-GaLore enables training 7B models on 16GB GPUs**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1612981403740463207) noted Q-GaLore incorporates low precision training with low-rank gradients and lazy layer-wise subspace exploration to **enable training LLaMA-7B from scratch on a single 16GB NVIDIA RTX 4060 Ti, though it is mostly slower**.
- **Mosaic compiler generates efficient H100 code**: [@tri_dao](https://twitter.com/tri_dao/status/1612913394086998408) highlighted that the Mosaic compiler, originally for TPU, **can generate very efficient H100 code, showing convergence of AI accelerators**.
- **Dolphin 2.9.3-Yi-1.5-34B-32k-GGUF model on Hugging Face**: [@01AI_Yi](https://twitter.com/01AI_Yi/status/1612958456317464804) gave kudos to @bartowski1182 and @cognitivecompai for the remarkable Yi fine-tune model on Hugging Face with **over 111k downloads last month**.

**AI Model Performance and Benchmarking**

- **Llama 3 model performance**: [@awnihannun](https://twitter.com/awnihannun/status/1812910444841214066) compared ChatGPT (free) vs MLX LM with Gemma 2 9B on M2 Ultra, showing comparable performance. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1812924233346904228) noted Llama 3 0-shotting **90% on the MATH dataset**.
- **Synthetic data limitations**: [@abacaj](https://twitter.com/abacaj/status/1812857696556663195) argued synthetic data is dumb and **unlikely to result in better models**, questioning the realism of synthetic instructions. [@Teknium1](https://twitter.com/Teknium1/status/1812905541993439597) countered that **synthetic data takes many forms** and sweeping claims are unwise.
- **Evaluating LLMs with LLMs**: [@percyliang](https://twitter.com/percyliang/status/1812999994255024144) highlighted the **power of using LLMs to generate inputs and evaluate outputs** of other LLMs, as in AlpacaEval, while cautioning about over-reliance on automatic evals.
- **LLM-as-a-judge techniques**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1612949923010421192) provided an overview of recent research on using LLMs to evaluate the output of other LLMs, including **early research, more formal analysis revealing biases, and specialized evaluators**.


**AI Safety and Regulation**

- **FTC sued Meta over acquiring VR companies**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1612978264484552987) noted he was dragged into two court cases where the FTC sued Meta over acquiring tiny VR companies and that **big tech has significantly ramped down acquisitions across the board, which is bad for startups as acquisition exits are being curtailed**.
- **Killing open source AI may politicize AI safety**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1612996261521551495) warned that **killing open source AI is likely to result in politicizing AI safety and that open-source is the solution**.
- **LLMs are not intelligent, just memorization machines**: [@svpino](https://twitter.com/svpino/status/1612888808372736309) argued that LLMs are incredibly powerful memorization machines that are impressive but not intelligent. **They can memorize large amounts of data and generalize a bit from it, but can't adapt to new problems, synthesize novel solutions, keep up with the world changing, or reason**.

**AI Applications and Demos**

- **AI and Robotics weekly breakdown**: [@adcock_brett](https://twitter.com/adcock_brett/status/1612880560819220806) provided a breakdown of the most important AI and Robotics research and developments from the past week.
- **Agentic RAG and multi-agent architecture concepts**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1612991904268849343) shared @nicolaygerold's thread and diagrams on agentic RAG and multi-agent architecture concepts discussed during their @aiDotEngineer talk.
- **MLX LM with Gemma 2 9B on M2 Ultra vs ChatGPT**: [@awnihannun](https://twitter.com/awnihannun/status/1612910444841214066) compared ChatGPT (free) vs MLX LM with Gemma 2 9B on M2 Ultra.
- **Odyssey AI video generation platform**: [@adcock_brett](https://twitter.com/adcock_brett/status/1612880741425918295) noted Odyssey emerged from stealth with a 'Hollywood-grade' AI video generation platform developing four specialized AI video models.


**Memes and Humor**

- **9.11 is bigger than 9.9**: [@goodside](https://twitter.com/goodside/status/1612977352085020680) joked that "9.11 is bigger than 9.9" in a humorous tweet, with follow-up variations and explanations in subsequent tweets.
- **Perplexity Office**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1612890154367078590) shared a humorous image titled "New Perplexity Office!"

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Theme 1. New Frontiers**

- [/r/singularity] **[A different source briefed on the matter said OpenAI has tested AI internally that scored over 90% on a MATH dataset](https://i.redd.it/akj0xjlmspcd1.png)** ([Score: 206, Comments: 59](https://reddit.com//r/singularity/comments/1e405o0/a_different_source_briefed_on_the_matter_said/)): **OpenAI's AI reportedly scores over 90% on MATH dataset**. An unnamed source claims that OpenAI has internally tested an AI system capable of achieving **over 90% accuracy** on a **MATH dataset**, suggesting significant advancements in AI's mathematical problem-solving abilities. This development, if confirmed, could have far-reaching implications for AI's potential in tackling complex mathematical challenges and its application in various fields requiring advanced mathematical reasoning.
- [/r/singularity] **A new quantum computer has shattered the world record set by Google’s Sycamore machine. The new 56-qubit H2-1 computer smashed ‘quantum supremacy’ record by 100 fold.** ([Score: 365, Comments: 110](https://reddit.com//r/singularity/comments/1e3z409/a_new_quantum_computer_has_shattered_the_world/)): **Xanadu's 56-qubit H2-1 quantum computer** has reportedly surpassed **Google's Sycamore machine** in the **quantum supremacy benchmark** by a factor of **100**. This achievement marks a significant leap in quantum computing capabilities, potentially accelerating the field's progress towards practical applications. The news was shared on X (formerly Twitter) by [@dr_singularity](https://x.com/dr_singularity/status/1812802357962441135?s=46), though further details and verification of this claim are yet to be provided.


**Theme 2. Advanced Stable Diffusion Techniques for Detailed Image Generation**


- [/r/StableDiffusion] **[Tile controlnet + Tiled diffusion = very realistic upscaler workflow](https://www.reddit.com/gallery/1e3v6jy)** ([Score: 517, Comments: 109](https://reddit.com//r/StableDiffusion/comments/1e3v6jy/tile_controlnet_tiled_diffusion_very_realistic/)): **Tile controlnet** combined with **Tiled diffusion** creates a highly effective workflow for realistic image upscaling. This technique allows for upscaling images to **4K or 8K resolution** while maintaining fine details and textures, surpassing the quality of traditional AI upscalers. The process involves using controlnet to generate a high-resolution tile pattern, which is then used as a guide for tiled diffusion, resulting in a seamless and detailed final image.

- [/r/StableDiffusion] **[Creating detailed worlds with SD is still my favorite thing to do!](https://www.reddit.com/gallery/1e4aynd)** ([Score: 357, Comments: 49](https://reddit.com//r/StableDiffusion/comments/1e4aynd/creating_detailed_worlds_with_sd_is_still_my/)): **Creating detailed fantasy worlds** using **Stable Diffusion** remains a top choice for creative expression. The ability to generate intricate, imaginative landscapes and environments showcases the power of AI in visual art creation. This technique allows artists and enthusiasts to bring their fantastical visions to life with remarkable detail and depth.
    - **NeededMonster** details their **4-stage workflow** for creating detailed fantasy worlds, including initial prompting, inpainting/outpainting, upscaling, and refining details. The process can take **1.5 hours per image**.
    - Commenters laud the images as "**book cover quality**" and "**the best I've seen yet**", with some suggesting the artist's skills are "**worth hiring**". NeededMonster expresses interest in finding work creating such images.

**Theme 3. Fine-tuning Llama 3 with Unsloth and Ollama**

- [/r/LocalLLaMA] **Step-By-Step Tutorial: How to Fine-tune Llama 3 (8B) with Unsloth + Google Colab & deploy it to Ollama** ([Score: 219, Comments: 41](https://reddit.com//r/LocalLLaMA/comments/1e416fo/stepbystep_tutorial_how_to_finetune_llama_3_8b/)): **Unsloth-Powered Llama 3 Finetuning Tutorial**  This tutorial demonstrates how to **finetune Llama-3 (8B)** using [Unsloth](https://github.com/unslothai/unsloth) and deploy it to [Ollama](https://github.com/ollama/ollama) for local use. The process involves using [Google Colab](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) for free GPU access, finetuning on the **Alpaca dataset**, and exporting the model to Ollama with automatic `Modelfile` creation. Key features include **2x faster finetuning**, **70% less memory usage**, and support for multi-turn conversations through Unsloth's `conversation_extension` parameter.


---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Mamba Models Make Waves**

- **Codestral Mamba Slithers into Spotlight**: Mistral AI released [Codestral Mamba](https://mistral.ai/news/codestral-mamba/), a 7B coding model using **Mamba2** architecture instead of transformers, offering linear time inference and infinite sequence handling capabilities.
   - The model, available under Apache 2.0 license, aims to boost code productivity. Community discussions highlighted its potential impact on **LLM architectures**, with some noting it's not yet supported in popular frameworks like `llama.cpp`.
- **Mathstral Multiplies STEM Strengths**: Alongside Codestral Mamba, Mistral AI introduced [Mathstral](https://mistral.ai/news/mathstral/), a 7B model fine-tuned for STEM reasoning, achieving impressive scores of **56.6%** on MATH and **63.47%** on MMLU benchmarks.
   - Developed in collaboration with [Project Numina](https://projectnumina.ai/), Mathstral exemplifies the growing trend of specialized models optimized for specific domains, potentially reshaping AI applications in scientific and technical fields.
  


**2. Efficient LLM Architectures Evolve**

- **SmolLM Packs a Petite Punch**: [SmolLM](https://x.com/loubnabenallal1/status/1813252390692303069?s=46) introduced new state-of-the-art models ranging from 135M to 1.7B parameters, trained on high-quality web, code, and synthetic data, outperforming larger counterparts like MobileLLM and Phi1.5.
   - These compact models highlight the growing importance of efficient, on-device LLM deployment. The release sparked discussions on balancing model size with performance, particularly for edge computing and mobile applications.
- **Q-Sparse Spices Up Sparsity**: Researchers introduced [Q-Sparse](https://arxiv.org/abs/2407.10969), a technique enabling fully sparsely-activated large language models (LLMs) to achieve results comparable to dense baselines with higher efficiency.
   - This advancement comes four months after the release of BitNet b1.58, which compressed LLMs to 1.58 bits. The AI community discussed how Q-Sparse could potentially reshape LLM training and inference, particularly for resource-constrained environments.
  


**3. AI Education and Benchmarking Breakthroughs**

- **Karpathy's Eureka Moment in AI Education**: Andrej Karpathy announced the launch of [Eureka Labs](https://x.com/karpathy/status/1813263734707790301), an AI-native educational platform starting with LLM101n, an undergraduate-level course on training personal AI models.
   - The initiative aims to blend AI expertise with innovative teaching methods, potentially transforming how AI is taught and learned. Community reactions were largely positive, with discussions on the implications for democratizing AI education.
- **SciCode Sets New Bar for LLM Evaluation**: Researchers introduced [SciCode](https://x.com/MinyangTian1/status/1813182904593199553?s=46), a new benchmark challenging LLMs to code solutions for scientific problems from advanced papers, including Nobel-winning research.
   - Initial tests showed even advanced models like GPT-4 and Claude 3.5 Sonnet achieving less than 5% accuracy, highlighting the benchmark's difficulty. The AI community discussed its potential impact on model evaluation and the need for more rigorous, domain-specific testing.

---

# PART 1: High level Discord summaries


## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **NVIDIA Bids Adieu to Project**: The **NVIDIA** shutdown of a project left community members speculating about its future, triggering discussions on safekeeping work against abrupt discontinuations.
   - One user recommends local storage as a contingency to project shutdowns, aiming to minimize work loss.
- **RAG Under the Microscope**: Skepticism surrounds **RAG (Retrieval-Augmented Generation)**, which is reportedly easy to start but challenging and costly to fine-tune to perfection.
   - A deep dive into optimization revealed the complexities involved, with members quoting **'fine-tuning the LLM, the embedder, and the reranker'**.
- **Hefty Price Tag on Model Tuning**: Fine-tuning a language model exceeding **200GB** could incur significant costs, stirring debate on the financial accessibility of advancing large models.
   - **Google Cloud's A2 instances** emerged as a possible yet still pricey alternative, emphasizing the weight of cost in the scaling equation.
- **Codestral Mamba** Springs into Action**: Mistral AI's **Codestral Mamba** breaks ground with linear time inference capabilities, offering rapid code productivity solutions.
   - **Mathstral** accompanies the release, spotlighting its advanced reasoning in STEM fields and garnering interest for its under-the-hood prowess.
- **Unsloth Pro** Exclusive Club**: The new **Unsloth Pro** version, currently under NDA, excites with its multi-GPU support and DRM system, yet is exclusively tied to a paid subscription model.
   - Expectations are geared towards an effective DRM system for diversified deployments, albeit limited to premium users.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **FlatBuffers Flexes Its Muscles Against Protobuf**: The community discussed the advantages of [FlatBuffers](https://flatbuffers.dev/) over Protobuf, highlighting FlatBuffers' performance and Apache Arrow's integration, which nevertheless uses Protobuf for data transfer.
   - Despite its efficiency, FlatBuffers faces challenges with harder usage and less industry penetration, sparking a debate on choosing serialization frameworks.
- **Mojo's Python Compatibility Conundrum**: A proposal for a Mojo mode that disables full compatibility with Python was debated, aiming to push **Mojo-centric syntax** and **robust error handling**.
   - Discussion included suggestions to adopt Rust-like monadic error handling to enhance reliability, avoiding traditional try-catch blocks.
- **MAX Graph API Tutorial Hits a Snag**: Learners faced hurdles with the [MAX Graph API tutorial](https://www.modular.com/blog/max-graph-api-tutorial), encountering Python script discrepancies and installation errors.
   - Community interventions corrected missteps such as mismatched Jupyter kernels and import issues, pointing newcomers to nightly builds for smoother experiences.
- **Mistral 7B Coding Model Wows with Infinite Sequences**: Mistral released a new **7B coding model** leveraging **Mamba2**, altering the landscape of coding productivity with its sequence processing capabilities.
   - The community showed enthusiasm for the [model](https://mistral.ai/news/codestral-mamba/) and shared resources for GUI building and ONNX conversion.
- **Error Handling Heats Up Mojo Discussions**: Talks surged around error handling in Mojo, with strong opinions on explicit error propagation and the implications for functions in systems programming.
   - An overarching theme was the emphasis on explicit declarations to improve code maintenance and to accommodate error handling on diverse hardware like GPUs and FPGAs.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Champion Math Solver Now in the Open**: **NuminaMath**, the AI Math Olympiad champ, has been [released as open source](https://x.com/reach_vb/status/1811069858584363374) flaunting a **7B model** and Apollo 11-like scores of **29/50** on Kaggle.
   - A toil of intelligence, the model was fine-tuned in two distinguished stages utilizing a vast expanse of math problems and synthetic datasets, specifically tuned for **GPT-4**.
- **Whisper Timestamped Marks Every Word**: Stamping authority on speech rec, **Whisper Timestamped** has now tuned into the [multilingual realm](https://x.com/xenovacom/status/1811068015229747335) with a robust in-browser solution using **Transformers.js**.
   - This whispering marvel gifts in-browser video editing wizards its full-bodied code and a magic demo for time-stamped transcriptions.
- **Vocal Virtuoso: Nvidia's BigVGAN v2 Soars**: Nvidia harmonized the release of [BigVGAN v2](https://x.com/reach_vb/status/1813181163126587830), their latest Neural Vocoder that compiles Mel spectrograms into symphonies faster on its A100 stage.
   - With a makeover featuring a spruced-up CUDA core, a finely tuned discriminator, and a loss that resonates, this model promises an auditory feast supporting up to **44 kHz**.
- **Merging Minds: Hugging Face x Keras**: **Keras** now brandishes NLP features with cunning from its [alliance with Hugging Face](https://huggingface.co/blog/keras-nlp-integration), pushing the envelope for developers in neural prose.
   - This melding of minds is staged to bring forth a cascade of NLP functions into the Keras ecosystem, welcoming devs to a show of seamless model integrations.
- **An Interface Odyssey: Hugging Face Tokens**: Hugging Face has revamped their [token management interface](https://x.com/coyotte508/status/1810982231180992521), imbuing it with novel features like token expiry dates and an elegant peek at the last four digits.
   - Shielding your precious tokens, this UI uplift serves as a castellan for your token lists, offering intricate details at just a glance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Sayonara NSFW: AI Morph Clamps Down**: AI Morph, a tool by Daily Joy Studio, triggered conversations when it stopped allowing NSFW content, showing a 'does not meet guidelines' alert.
   - The community reaction was mixed, with some speculating about the **impact on content creation**, while others discussed alternative tools.
- **Anime Artistry with Stable Diffusion**: Queries emerged on how to finesse Stable Diffusion for anime art's color, attire, and facial expression accuracy, hinting at the need for **fine-grained control mechanisms**.
   - Several users exchanged tips, with some pointing to [certain GitHub repositories](https://github.com/leejet/stable-diffusion.cpp) as potential resources.
- **Detweiler's Tutorials: A Community Favorite**: The community hailed Scott Detweiler's YouTube tutorials on Stable Diffusion, praising his **quality insights** into the tool.
   - His contributions, as part of his quality assurance role at Stability.ai, were highlighted, cementing his place as a go-to source for learning.
- **Homemade AI Tools Marry Local and Stable**: Excitement brewed around the development of a local AI tool that cleverly integrates Stable Diffusion, becoming a preferred tool for *capt.asic*.
   - The discussion branched into the effectiveness of combining Stable Diffusion with **Local Language Model (LLM)** support.
- **AI's Graphical Gladiators: 4090 vs. 7900XTX**: GPU performance for AI tasks sparked debates, with NVIDIA's 4090 facing off against AMD's 7900XTX on aspects like cost-efficiency and accessibility.
   - Amid the specs talk, links to [Google Colab projects](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb) and specs comparisons furthered the discussion.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Kernel Capers & PyTorch Prowess**: Discussions in **#[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1262494017929871473)** centred on the invocation of CUDA kernels via PyTorch in Python scripts, suggesting the use of the **PyTorch profiler** to untangle which ATen functions are triggered.
   - A performance tug-of-war was highlighted, citing a [lecture](https://youtu.be/4sgKnKbR-WE?t=4045) where a raw CUDA matrix multiplication kernel took **6ms**, while its PyTorch counterpart breezed through in **2ms**, raising questions about the efficiency of PyTorch's kernels for convolution operations in CNNs.
- **Spectral Compute's SCALE Toolkit Triumph**: **SCALE** emerged in **#[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1262562492824027157)**, praised for its ability to transpile CUDA applications for AMD GPUs, an initiative that could shift computational paradigms.
   - With the future promise of broader GPU vendor support, developers were directed to SCALE's [documentation](https://docs.scale-lang.com/) for tutorials and examples, portending a potential upsurge in cross-platform GPU programming agility.
- **Suno's Search for Machine Learning Virtuosos**: **#[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1262526637904498800)** spotlighted Suno's campaign to enlist ML engineers, skilled in **torch.compile** and **triton**, for crafting real-time audio models; familiarity with **Cutlass** is a bonus, not a requirement as per the [Job Posting](https://jobs.ashbyhq.com/suno/7522d696-7ce8-4ece-a983-4be03dffde20).
   - Internship roles also surfaced, offering neophytes an entry point to the intricate dance of training and inferencing within Suno's ML landscape.
- **Lightning Strikes and Huggingface Hugs Development**: **#[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1262594919206227981)** discussions showcased **[Lightning AI's Studios](https://lightning.ai/docs/overview/studios#studios)** as a deftly designed hybrid cloud-browser development arena, while CUDA development aspirations for **Huggingface Spaces Dev Mode** hung unanswered.
   - The latter hosted queries on the feasibility of CUDA endeavors within Huggingface's nurturing ecosystem, reflecting a community at the edge of experimentation and discovery.
- **Karpathy's Eureka Moment with AI+Education**: **Andrej Karpathy** announces his leap into the AI-aided education sphere with **[Eureka Labs](https://x.com/karpathy/status/1813263734707790301)**; a discussion in **#[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1262485006207418409)** touched on the intent to synergize teaching and technology for the AI-curious.
   - A precursor course, LLM101n, signals the start of Eureka's mission to augment educational accessibility and engagement, setting the stage for AI to potentially reshape the learning experience.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Generous Gesture by OpenRouter**: OpenRouter has graciously provided the community with free access to the **Qwen 2 7B Instruct** model, enhancing the arsenal of AI tools for eager engineers.
   - To tap into this offering, users can visit [OpenRouter](https://openrouter.ai/models/qwen/qwen-2-7b-instruct) and engage with the model without a subscription.
- **The Gemini Debate: Free Tier Frustrations**: Contrasting views emerged on Google's **Gemini 1.0** available through its free tier, with strong opinions stating it falls short of **OpenAI's GPT-4o** benchmark.
   - A member highlighted the overlooked promise in **Gemini 1.5 Pro**, citing it has creative prowess despite its coding quirks.
- **OpenRouter Oscillation: Connectivity Woes**: OpenRouter users faced erratic disruptions accessing the site and its API, sparking a series of concerns within the community.
   - Official statements attributed the sporadic outages to transient routing detours and potential issues with third-party services like Cloudflare.
- **Longing for Longer Contexts in Llama 3**: Engineers shared their plight over the **8k context window limitation** in **Llama 3-70B Instruct**, pondering over superior alternatives.
   - Suggestions for models providing extended context abilities included **Euryale** and **Magnum-72B**, yet their consistency and cost factors were of concern.
- **Necessities and Nuances of OpenRouter Access**: Clarification spread amongst users regarding OpenRouter's model accessibility, noting not all models are free and some require a paid agreement.
   - Despite the confusion, OpenRouter does offer a selection of APIs and models free of cost, while hosting enterprise-grade dry-run models demands underscoring specific business contracts.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Tempest**: Pro-level Troubles**: Users voiced their struggles with **Pro Subscription Support** at [Perplexity AI](https://perplexity.ai/settings/account), facing activation issues across devices despite confirmation emails.
   - Discussions revolved around questions like implementing **model settings for separate collections** and sharing excitement over a new [Perplexity Office](https://x.com/AravSrinivas/status/1812890154367078590).
- **Alphabet's Audacious Acquisition**: $23B Sealed**: **Alphabet** has made waves with a hefty $23 billion acquisition, stirring up the market which you can catch in a [YouTube briefing](https://www.youtube.com/embed/lKn8rh0pOiM).
   - Speculation and chatter ensued on how this move could usher in new strides, putting Alphabet on the radar for potent market expansion.
- **Lunar Refuge Uncovered**: A Cave for Future Astronauts**: An **accessible lunar cave** found in Mare Tranquillitatis could be a boon for astronaut habitation, thanks to its protection against the moon's extremes.
   - With a breadth of at least 130 feet and a more temperate environment, it stands out as a potential lunar base as per [Perplexity AI's coverage](https://www.perplexity.ai/search/moon-s-hidden-refuge-scientist-yz19IMD.TE6E4fZj9A9W.Q#0).
- **7-Eleven's Experience Elevator**: Customer Delight in Sight**: Improving shopping delight, **7-Eleven** is gearing up for a major upgrade, possibly revamping consumer interaction landscapes.
   - Piquing interests, 7-Even invites you to [explore the upgrade](https://www.youtube.com/embed/lKn8rh0pOiM), perhaps setting a new benchmark in retail convenience.
- **API Angst**: pplx-api Perturbations**: The `pplx-api` crowd discussed missing features like a `deleted_urls` equivalent and the frustration of **524 errors** affecting `sonar` models.
   - Workarounds suggested include setting the `"stream": True` to keep connections alive amid timeouts with `llama-3-sonar-small-32k-online` and related models.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Codestral Mamba Strikes with Infinite Sequence Modeling**: The introduction of [Codestral Mamba](https://mistral.ai/news/codestral-mamba/), featuring linear time inference, marks a milestone in AI's ability to handle infinite sequences, aiding code productivity benchmarks.
   - Developed with Albert Gu and Tri Dao, **Codestral's architecture** competes with top transformer models, indicating a shift towards more efficient sequence learning.
- **Mathstral Crunches Numbers with Precision**: [Mathstral](https://mistral.ai/news/mathstral/), focusing on STEM, shines with **56.6%** on MATH and **63.47%** on MMLU benchmarks, bolstering performance in niche technical spheres.
   - In association with [Project Numina](https://projectnumina.ai/), Mathstral represents a calculated balance of speed and high-level reasoning in specialized areas.
- **SmolLM Packs a Punch On-Device**: SmolLM's new SOTA models offer [high performance with reduced scale](https://x.com/loubnabenallal1/status/1813252390692303069?s=46), making advancements in the on-device deployment of LLMs.
   - Outstripping MobileLLM amongst others, these models indicate a trending downsizing while maintaining adequate power, essential for mobile applications.
- **Eureka Labs Enlightens AI-Education Intersection**: With its AI-native teaching assistant, [Eureka Labs](https://eurekalabs.ai/) paves the way for an AI-driven educational experience, beginning with the LLM101n product.
   - Eureka's innovative approach seeks to enable students to shape their understanding by **training their own AI**, revolutionizing educational methodology.
- **Championing Fair Play in Policy Reinforcement**: Discussions ensued over the utility of degenerate cases in policy reinforcement for managing common prefixes in **winning and losing strategies**.
   - Acknowledging a deep dive is warranted, the focus on detailed, technical specifics potentially ushers in advanced methods in **policy optimization**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4's GSM8k Mastery**: GPT-4's performance prowess was highlighted with its ability to handle most of the **GSM8k train set**; an interesting fact shared from the [GPT-4 tech report](https://link.to.report).
   - The community spotlighted the memory feats of **GPT-4**, as such details often reverberate across social platforms.
- **Instruction Tuning Syntax Scrutinized**: Syntax within the instruction tuning dataset generated discussion, questioning the approach in comparison to OpenAI's method of chaining thoughts using bullet points.
   - Curiosity arose about the potential inclusion of specific markers in tuning datasets, initiating dialogue about dataset integrity.
- **GPU Failures under the Microscope**: Members sought to understand the frequency of **GPU failures** during model training, referencing reports such as the [Reka tech report](https://publications.reka.ai/reka-core-tech-report.pdf).
   - Open resources like OPT/BLOOM logbooks emerged as go-to sources for those aiming to analyze the stability of large-scale AI training environments.
- **Advancements in State Space Models**: Innovative construction of State Space Model weights propelled discussions, with [this study](https://arxiv.org/abs/2407.09375) illustrating their capability to learn dynamical systems in context.
   - Researchers exchanged insights on these models' potential to predict system states without additional parameter adjustments, underscoring their utility.
- **Neural Counts: Human vs Animal**: Debate flared over intelligence differences between humans and animals, recognizing human superiority in planning and creativity, compared to comparable sensory capabilities.
   - Conversation touched on the larger human neocortex and its increased folding, which may contribute to our distinct cognitive abilities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic's Token Triumph**: Anthropic unveiled their **drip-feed PR approach**, divulging a token limit boost for **Claude 3.5 Sonnet** from 4096 to 8192 tokens in the API, bringing cheer to developers.
   - An avid developer shared a sigh of satisfaction, remarking on past limitations with a [linked celebration tweet](https://x.com/alexalbert__/status/1812921642143900036) that highlighted this **enhancement**.
- **LangGraph vs XState Shootout**: Enthusiasm brews as a member previews their **XState** work to craft LLM agents, comparing methodologies with **LangGraph**, publicly on [GitHub](https://github.com/statelyai/agent).
   - Anticipation builds for a thorough breakdown, set to delineate strengths between the two, enriching the toolset of AI engineers venturing into **state-machine-powered AI agents**.
- **Qwen2 Supersedes Predecessor**: Qwen2 has launched a language model range outclassing Qwen1.5, offering a **0.5 to 72 billion parameter** gamut, gearing competition with proprietary counterparts.
   - The collection touts both dense and **MoE** models, as the community pores over the promise in [Qwen2's technical report](https://hf.co/papers/2407.10671).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio:** Network Nexus**: LM Studio users discuss **Android app access** for home network servers, highlighting **VPN tools** like Wireguard for secure server connections.
   - Comprehensive support for **Intel GPUs** is negated, advising better performance with other hardware for AI tasks.
- **Bugs & Support: Conversation Collision**: **Gemma 2** garners support in `llama.cpp`, while **Phi 3 small** gets sidelined due to incompatibility issues.
   - Community digs into the **LMS model loading slowdown**, pinpointing ejection and reloading as a quick-fix to the sluggish performance.
- **Coding with Giants: Deepseek & Lunaris**: A hunt for an ideal local coding model for **128GB RAM systems** zeroes in on **Deepseek V2**, a model boasting **21B experts**.
   - Precise differences between **L3-12B-Lunaris** variants spark a conversation, with emphasis on trying out the free LLMs for performance insights.
- **Graphical Glitches: Size Matters Not**: `f32` folder anomaly grabs attention in LM Studio, leading some to muse over cosmetic bugs' impact on user experience.
   - **Flash Attention** emerges as the culprit behind an F16 GGUF model load issue, with deactivation restoring functionality on an RTX 3090.
- **STEM Specialized:** Mathstral's Debut**: **Mistral AI** reveals **Mathstral**, their latest STEM-centric model, promising superior performance over its **Mistral 7B** predecessor.
   - The **Community Models Program** spotlights **Mathstral**, inviting AI enthusiasts to dive into discussions on the LM Studio [Discord](https://discord.gg/aPQfnNkxGC).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Evolutionary Instructional Leaps** with Evol-Instruct V2 & Auto**: WizardLM's announcement of [Evol-Instruct V2](https://x.com/WizardLM_AI/status/1812844503251947978) extended WizardLM-2's capabilities from three evolved domains to dozens, potentially enhancing AI research fairness and efficiency.
   - **Auto Evol-Instruct** demonstrated notable performance gains, outperforming human-crafted methods with improvements of **10.44%** on MT-bench, **12%** on HumanEval, and **6.9%** on GSM8k.
- **Q-Sparse:** Computing with Efficiency**: [Q-Sparse](https://x.com/realHongyu_Wang/status/1813112679734911169), introduced by Hongyu Wang, claims to boost LLM computation by optimizing compute over memory-bound processes.
   - This innovation tracks at a four-month lag following BitNet b1.58's achievement of compressing LLMs to **1.58 bits**.
- **SpreadsheetLLM**: Microsoft's New Frontier in Data**: **Microsoft** innovates with **SpreadsheetLLM**, excelling in spreadsheet tasks, which could result in a significant shift in data management and analysis.
   - A [pre-print paper](https://arxiv.org/abs/2407.09025) highlighted the release of SpreadsheetLLM, sparking debates over automation's impact on the job market.
- **Heat Advisory:** Urban Temperatures and Technological Solutions**: Amidst extreme temperatures, discussions arise on painting roofs white, supported by a [Yale article](https://e360.yale.edu/features/urban-heat-can-white-roofs-help-cool-the-worlds-warming-cities), to reduce the urban heat island effect.
   - The invention of a super white paint that reflects 98% of sunlight was showcased in a [YouTube demonstration](https://youtu.be/KDRnEm-B3AI), igniting conversations about its potential to cool buildings passively.
- **AI Literacy: Decoding Tokenization Woes**: Conflicts with **tiktoken library** when handling Arabic symbols in LLMs were discussed; decoding mishaps replace original strings with special tokens, posing challenges for text generation.
   - The tokenization process variability is evident, with outcomes ranging between a **UTF-8 sequence** and a **0x80-0xFF** byte, raising concerns about tokenization's invertibility with `cl100k_base`.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora's Anticipated Arrival**: Sora's release date discussions in Q4 2024 have surfaced based on **OpenAI blog posts**. **OpenAI** has not confirmed these speculations.
   - Cautions were raised regarding trusting unofficial sources such as random Reddit or Twitter posts for launch predictions.
- **Miniature Marvel: GPT Mini's Potential Role**: A rumored **GPT mini** in **Lymsys** generated buzz, though details remain sparse and unverified.
   - Skepticism exists, suggesting many predictions lack concrete basis.
- **From Zero to Hero with GPT-4 Coding**: It's discussed how GPT-4 assists enthusiasts in coding a **mobile game**, noting that while it provides structure, vigilant error-checking is essential.
   - Success stories shared of individuals crafting web apps with no previous coding experience, attributing their strides to GPT-4's guidance.
- **Language Lessons with GPT Models**: Performance variation among AI models is linked to the extent of their language training, affecting responses quality.
   - The discussion ranged over the model's mix-ups with regional slang to a shift from a casual to a more formal tone in GPT-4, impacting user experience.
- **Chatbot Cultivation Challenges**: A student's attempt to create a **Bengali/Banglish** support chatbot stirred conversations on whether modest fine-tuning with 100 conversations would be beneficial.
   - Responses clarified that while fine-tuning assists pattern recognition, these might be forgotten past the context window, impacting conversation flow.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex's Bridge to Better RAG**: A [workshop by LlamaIndex](https://lu.ma/ufx77ve8) will delve into enhancing **RAG** with advanced parsing and metadata extraction, featuring insights from Deasie's founders on the topic.
   - Deasie's labeling workflow purportedly optimizes RAG, auto-generating hierarchical metadata labels, as detailed on their [official site](https://deasie.com/).
- **Advanced Document Processing Tactics**: LlamaIndex's [new cookbook](https://t.co/KWsVGwT3jD) marries LlamaParse and GPT-4o into a hybrid text/image RAG architecture to process diverse document elements.
   - Concurrently, Sonnet-3.5's adeptness in chart understanding shines, promising better data interpretation through multimodal models and LlamaParse's [latest release](https://t.co/Dq3xaVfXio).
- **Graphing the Details with LlamaIndex and GraphRAG**: Users compared **llamaindex property graph** capacities with **Microsoft GraphRAG**, highlighting property graph's flexibility in retrieval methods like text-to-cypher, outlined by *Cheesyfishes*.
   - GraphRAG's community clustering is contrasted with the property graph's customized features, with examples found in the [documentation](#).
- **Streamlining AI Infrastructure**: Discussions emerged around efficient source retrieval for LLM responses, with methods like `get_formatted_sources()` aiding in tracing data provenance, cited in LlamaIndex's [tutorial](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/vector_stores/SimpleIndexDemo.ipynb).
   - The AI community actively seeks accessible public vector datasets to minimize infrastructure complexity, with preferences for pre-hosted options, though no specific services were mentioned.
- **Boosting Index Loading Efficiency**: Members shared strategies to expedite loading of hefty indexes, suggesting parallelization as a potential avenue, stirring a debate on optimizing methods like `QueryPipelines`.
   - For data embedding in Neo4J nodes, the community turns to `PropertyGraphIndex.from_documents()`, with the process thoroughly detailed in [LlamaIndex's source code](https://github.com/run-llama/llama_index/blob/f092d90bd5934097f5c166014f5d99a3e07ea999/llama-index-core/llama_index/core/indices/property_graph/base.py#L248).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cooperation Coded in Cohere's Python Community**: A participant recommended exploring the [Cohere Python Library](https://github.com/cohere-ai/cohere-python) for those interested in contributing to open source projects, fostering a community-driven improvement.
   - An enthusiast of the library hinted at their intention to **contribute** soon, potentially bolstering the collaborative efforts.
- **Discord's Categorization Conundrum Calls for Prompt Care**: Problems emerged with a Discord bot misfiling all posts to the 'opensource' category despite various topics, hinting at issues in **automatic post categorization**.
   - A colleague chimed in, speculating that an easy **prompt adjustment** could correct the course, reminiscent of r/openai's misrouted posts.
- **Spam Scams Prompt Preemptive Proposal**: A proactive member proposed creating **spam awareness** content as part of the server's newcomer onboarding to boost security.
   - The suggestion was met with support, sparking a discussion on implementing best practices for community **safety**.
- **Max Welling's Warm Fireside Reception**: [C4AI announced a fireside chat](https://discord.gg/Jf6FPp3c?event=1262761447486787685) with Max Welling from the University of Amsterdam, an event that stirred excitement.
   - However, the promotion faced a hiccup with an **unnecessary @everyone alert**, for which an apology was issued.
- **Recruitment Rules Reiterated on Discord**: It was **reminded** that job postings are not permitted on this server, reinforcing the community's guidelines.
   - Members were **urged** to keep employment engagements through private discussions, emphasizing respectful professional protocols.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Finetuning Frolics for Functionality**: A **finetuning pipeline** became the crux of discussion as members exchanged ideas on solving problems by sharing pipeline strategies.
   - One member highlighted the application of **LLM training** with a set of 100 dialogues to enhance a Bengali chatbot's responsiveness.
- **Tick-Tock MessageGraph Clock**: The addition of **timestamps** to a **MessageGraph** to automate message chronology was a technical query that sparked dialogue amongst the guild's minds.
   - Speculation ensued over the necessity of a **StateGraph** for customized temporal state management.
- **Launching Verbis: A Vanguard in Privacy**: Verbis, an open-source MacOS application promises enhanced productivity with local data processing by utilizing **GenAI** models.
   - Launched with fanfare, it guarantees zero data sharing to third parties, boldly emphasizing privacy. [Learn more on GitHub](https://github.com/verbis-ai/verbis)
- **RAG on the Web: LangChain & WebLLM Synergy**: A demonstration showed the prowess of **LangChain** and **WebLLM** in a browser-based environment, deploying a chat model for on-the-fly question answering.
   - The video shared offers a hands-on showcase of **Visual Agents** and emphasizes a potent, in-browser user experience. [Watch the demo](https://youtu.be/MHuvSuK2dnY)



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Dive Into Dynamic Tuning**: Curiosity peaked among members about the rollout of a **PyTorch tuner**, with exchanges on its potential to swap and optimize **instruction models** efficiently.
   - The tuner's capacity for **context length adjustment** was highlighted, cautioning that extensive length may be VRAM-hungry.
- **Template Turmoil in Mistral**: Discussions surfaced indicating that **Mistral's unique chat template** deviates from the norm, stirring operational stir among users.
   - The chat template intricacies led to a dialogue on fine-tuning tactics to circumvent the issues presented.
- **Merging Methodologies Matter**: The **Axolotl repository** glowed with activity as a **new pull request** was crafted, marking progression in development efforts.
   - Yet, the simplicity of the **Direct Policy Optimization** (DPO) method was debated, uncovering its limitations in extending beyond basic tokenization and masking.
- **LoRA Layers Lead the Way**: A focused forum formed around `lora_target_linear`, a **LoRA configuration** toggle that transforms how linear layers are adapted for more efficient fine-tuning.
   - The setting's role in Axolotl's fine-tuning sparked discussions, although some queries about disabling **LoRA** on certain layers remain unanswered.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune's Tantalizing v0.2.0 Takeoff**: The eagerly awaited launch of [Torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0) marks a major milestone with additions including exciting models and recipes.
   - Community contributions have enriched the release, featuring dataset enhancements like **sample packing**, for improved performance and diverse applications.
- **Evaluating Without Backprop Burdens**: To optimize checkpoint selection, loss calculation during evaluation can be done without backpropagation; participants discussed plotting loss curves and comparing them across training and evaluation datasets.
   - Suggestions included modifications to the [default recipe config](https://github.com/pytorch/torchtune/issues/1066) and incorporating a test split and an eval loop using `torch.no_grad()` alongside model eval mode.
- **RoPE Embeddings Rise to Record Contexts**: Long context modeling gains traction with a proposal for scaling RoPE (Rotary Positional Embeddings) in [Torchtune's RFC](https://github.com/pytorch/torchtune/issues/1183), paving the way for large document and code completion tasks.
   - The discussion revolves around enabling RoPE for contexts over 8K that can transform understanding of large volume documents and detailed codebases.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **ComfyUI Crew Concocts Disney Disruption**: A participant revealed that those behind the **ComfyUI malicious node attack** claimed to also orchestrate the **Disney cyber attacks**, challenging the company's digital defenses.
   - It was mentioned that, while some see the **Disney attacks** as chaotic behavior, there is speculation and anticipation regarding **FBI's** potential investigation into the incidents.
- **Codestral's Mamba Makes Its Mark**: A recent post shared a breakthrough dubbed [**Codestral Mamba**, a new update from Mistral AI](https://mistral.ai/news/codestral-mamba/), sparking conversations around its capabilities and potential applications.
   - The details of its performance and comparison to other models, along with technical specs, were not elaborated, leaving the community curious about its impact.
- **YouTube Yields Fresh Tutorial Temptations**: A fresh tutorial catch surfaced with a link to a [new YouTube tutorial](https://youtu.be/pj8CtzHHq-k) posted by a guild member, touted to offer educational content in a video format.
   - The specifics of the video's educational value and relevance to the AI engineering community were not discussed, prompting members to seek out the content independently.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Challenges Mount for Meta's Specs**: Engineers grapple with the integration of **Open Interpreter** into **RayBan Stories**, thwarted by lack of official SDK and difficult hardware access.
   - An attempt to dissect the device revealed hurdles, discussed on a Pastebin [documentation](https://pastebin.com/wTsRMH3f), with concerns about internal adhesives and transparency for better modding.
- **Google Glass: A Looking Glass for Open Interpreter?**: With the struggles of hacking **RayBan Stories**, **Google Glass** was floated as a possible alternative platform.
   - Dialogue was scarce following the suggestion, indicating a need for further investigation or community input.
- **O1 Light Hardware: Patience Wears Thin**: Community discontent grows over multiple-month delays in **O1 Light hardware** preorders, with a vexing silence on updates.
   - Members air their grievances, indicating strained anticipation for the product, with the **lack of communication** augmenting their unease.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Confusion Cleared: Accessing GPT-4o Fine-Tuning**: Queries about **GPT-4o fine-tuning** access led to clarification that an **OpenAI** invitation is necessary, as highlighted by a user referencing Kyle's statement.
   - The discussion unfolded in the #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1262787952032092196) channel, reflecting the community's eagerness to explore fine-tuning capabilities.
- **OpenPipeAI Embraces GPT-4o: Train Responsibly**: **OpenPipeAI** announces support for **GPT-4o training**, with an appeal for responsible use by [Corbtt](https://x.com/corbtt/status/1813018434822971556).
   - This update serves as a pathway for AI Engineers to harness their course credits more efficiently in AI training endeavors.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Inside Tinygrad's Core**: Dissecting the Intermediate Representation**: Interest bubbled up around **tinygrad's intermediate language**, with users curious about the structure of deep learning operators within the IR.
   - Tips were exchanged about leveraging debug options **DEBUG=3** for insights into the lower levels of IR, while **GRAPH=1** and **GRAPHUOPS=1** commands surfaced as go-to options for visualizing tinygrad's inner complexities.
- **Tinygrad Tales**: Visualization and Debugging Dynamics**: Amidst discussions, a nugget of wisdom was shared for debugging tinygrad using **DEBUG=3**, revealing the intricate lower levels of the intermediate representation.
   - Further, for those with a keen eye on visualization, invoking **GRAPH=1** and **GRAPHUOPS=1** could translate tinygrad's abstract internals into graphical clarity.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Opening Insights with Open Interpreter**: Mike Bird shines a spotlight on **Open Interpreter**, inviting active participation.
   - Audience engagement is driven by encouragement to field questions during the elucidation of **Open Interpreter**.
- **Engage with the Interpreter**: The stage echoes with discussions on **Open Interpreter** steered by Mike Bird, as he unfolds the project details.
   - Invitations are cast to the attendees to query and further the conversation on **Open Interpreter** during the presentation.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1262484993322516510)** (235 messages🔥🔥): 

> - `NVIDIA shutdown`
> - `RAG performance and optimization`
> - `Costs of fine-tuning large models`
> - `Codestral Mamba and Mathstral releases`
> - `Unsloth Pro license concerns` 


- **NVIDIA Shuts Down Project**: A member confirmed that **NVIDIA** shut down a project, sparking concerns over its future updates and maintenance.
   - *Another member suggested locally storing projects to avoid losses from such shutdowns.*
- **RAG Overhyped and Costly to Perfect**: **RAG (Retrieval-Augmented Generation)** was called an overhyped solution that, while easy to bootstrap in hours, takes months to perfect.
   - "It's often not just RAG and LLM; you often fine-tune the LLM, the embedder, and the reranker," one member stressed.
- **Fine-Tuning Large Models is Pricey**: Fine-tuning a model on **200GB+ of data** can cost high five to six figures, depending on the model parameters and size.
   - [Google Cloud's A2 instances](https://cloud.google.com/compute/docs/instances/a2-machine-types) were cited as a possible solution, but the cost concern remains significant.
- **Mistral AI Releases Codestral Mamba and Mathstral**: [Mistral AI released Codestral Mamba](https://mistral.ai/news/codestral-mamba/) with linear time inference for infinite length sequences and Mathstral for advanced mathematical problem-solving.
   - Codestral Mamba was designed for **quick responses in code productivity**, while Mathstral excels in **STEM subjects** with state-of-the-art reasoning capacities.
- **Unsloth Pro Licensing and GPU Support**: A new **Unsloth Pro** version is being tested under NDA, featuring multi-GPU support and a DRM system with floating licenses per GPU per month.
   - "The free version doesn't support multiple GPUs," but there is a demo of classification notebooks available. Expectations are high for a robust DRM system for cloud and local deployment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colmweb.org/">COLM 2024</a>: no description found</li><li><a href="https://docs.marqo.ai/2.10/">Getting Started with Marqo - Marqo docs</a>: no description found</li><li><a href="https://x.com/dudeman6790/status/1813020953577988565?t=yLDcQxbyS4q6U5UP_f0xnw&s=19">Tweet from RomboDawg (@dudeman6790)</a>: @youliang_yuan My man is providing AI safety while I am freeing AI from your crappy censorship. Creating fully uncensored datasets, and releasing open weight uncensored models to compete with closed s...</li><li><a href="https://x.com/dudeman6790/status/1813020953577988565?t=yLDcQxbyS4q6U5UP_f0x">Tweet from RomboDawg (@dudeman6790)</a>: @youliang_yuan My man is providing AI safety while I am freeing AI from your crappy censorship. Creating fully uncensored datasets, and releasing open weight uncensored models to compete with closed s...</li><li><a href="https://tenor.com/view/sample-contract-nda-non-disclosure-agreement-gif-17773157">Sample Contract GIF - Sample Contract Nda - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/shocked-surprised-gasp-what-cat-shock-gif-11368945723132907566">Shocked Surprised GIF - Shocked Surprised Gasp - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: As a tribute to Cleopatra, whose glorious destiny ended in tragic snake circumstances, we are proud to release Codestral Mamba, a Mamba2 language model specialised in code generation, available under ...</li><li><a href="https://mistral.ai/news/mathstral/">MathΣtral</a>: As a tribute to Archimedes, whose 2311th anniversary we're celebrating this year, we are proud to release our first Mathstral model, a specific 7B model designed for math reasoning and scientific disc...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1262487568822566943)** (27 messages🔥): 

> - `DCLM Baseline`
> - `Model performance`
> - `RTX 4090 vs 3060`
> - `Eureka Labs AI`
> - `New releases from Mistral` 


- **Apple releases DCLM-Baseline-7B**: Apple released [DCLM-Baseline-7B](https://huggingface.co/apple/DCLM-Baseline-7B), a 7 billion parameter language model trained on the DCLM-Baseline dataset, with a context length of 2048 tokens.
   - An **8K context length version** has also been [released](https://huggingface.co/apple/DCLM-Baseline-7B-8k) and the paper can be found [here](https://arxiv.org/abs/2406.11794).
- **RTx 4090 versus multiple 3060s for AI**: Debate sparked on whether to purchase an RTX 4090 or multiple 3060s for AI tasks, considering VRAM needs and performance.
   - An **expert opinion** was shared stating that Unsloth only supports single GPU at the moment and multiple GPUs are reserved for paid licenses.
- **Eureka Labs AI education company**: Andrej Karpathy announced the launch of [Eureka Labs](https://x.com/karpathy/status/1813263734707790301), aiming to build AI-native educational platforms and courses.
   - Their first product, **LLM101n**, is an undergraduate-level course on training one's own AI, with both digital and physical cohorts planned.
- **Mistral releases two new models**: Mistral released [Mamba-Codestral-7B-v0.1](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) focusing on code tasks and [Mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1) for mathematical and scientific tasks, both under Apache 2.0 with 32k context.
   - The coding model is currently **not supported** by Llama.cpp.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1813263734707790301">Tweet from Andrej Karpathy (@karpathy)</a>: ⚡️ Excited to share that I am starting an AI+Education company called Eureka Labs.  The announcement:  --- We are Eureka Labs and we are building a new kind of school that is AI native.  How can we ap...</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/mathstral-7B-v0.1">mistralai/mathstral-7B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e4jw0c/apple_has_released_the_weights_for_their_7b_dclm/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1262639087169441895)** (132 messages🔥🔥): 

> - `Mimicking Pretraining`
> - `Fine-Tuning LLMs on Domain-Specific PDFs`
> - `RunPod Training Issues`
> - `Multi-GPU Support`
> - `Exporting Models and Inference Methods` 


- **Mimicking Pretraining with Adjusted Parameters**: It's suggested to mimic pretraining by increasing r=16 to r=256 and reducing the learning rate for achieving a result that's 98-99% close to full fine-tuning.
   - *theyruinedelise*: 'You can mimic pretraining by increasing r=16 to r=256 then reduce the learning rate.'
- **Fine-Tuning LLMs on Domain-Specific PDFs**: Members discussed ways to fine-tune an LLM using domain-specific PDFs, recommending [synthetic dataset generation](https://github.com/e-p-armstrong/augmentoolkit) and simple pdf2text conversion.
   - *mrdragonfox*: 'There won’t be handholding – what you are looking for is synthetic dataset generation... as an llm just understands text.'
- **RunPod Training Issues and Solutions**: A member had issues with training suddenly stopping on RunPod despite GPU utilization, suggesting to reduce dataset size to verify settings.
   - *mrdragonfox*: 'Cut the dataset down to 5k for a quick eval... maybe even just to 1k.'
- **Upcoming Multi-GPU Support in Unsloth**: Unsloth's multi-GPU version is in testing and should be available soon; it won’t be available to the general public but will be a paid product.
   - *mrdragonfox*: 'Yup, because I have the multigpu version... should be soonish, I think right now it’s testing.'
- **Exporting Models and Inference Methods Explained**: There was a detailed discussion on exporting models using LoRA adapters and merged models, including the benefits of using inference frameworks like vllm and aphro for better performance.
   - *flail_*: 'Real inference library much faster... good batching for multiple outputs at once.'


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/explodinggradients/ragas">GitHub - explodinggradients/ragas: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines</a>: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines - explodinggradients/ragas</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets (or classifiers)!</a>: Convert Compute And Books Into Instruct-Tuning Datasets (or classifiers)! - e-p-armstrong/augmentoolkit</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.  - GitHub - Unstructured-IO/unstructured: Open source librar...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1262508919004336270)** (17 messages🔥): 

> - `LLaMA-405B`
> - `Q-Sparse`
> - `ColPali`
> - `AgentInstruct`
> - `Adam-mini` 


- **LLaMA-405B to run affordably**: Members discussed the possibility of running **LLaMA-405B** more cheaply, citing a tweet from @teortaxesTex.
   - The conversation mentioned a new paper on **Q-Sparse** advocating fully sparsely-activated large language models.
- **ColPali achieves high performance**: The **ColPali** model, associated with **PaliGemma-3B** and the **ColBERT** strategy, was highlighted for its exceptional document retrieval capabilities.
   - The **ColBERT** late interaction aspect of ColPali significantly differentiates its performance from **BiPali** models, as noted in multiple discussions.
- **AgentInstruct improves synthetic data**: Microsoft Research introduced **AgentInstruct**, an automated framework that generates high-quality synthetic data using a multi-agent workflow.
   - The framework improved the performance of the **Orca-3** model significantly on AGIEval and GSM8K benchmarks.
- **Adam-mini reduces memory usage**: **Adam-mini**, a new optimizer compatible with **PyTorch's** distributed training codebase Torchtitan, was noted for nearly **50% less memory usage** compared to AdamW.
   - "Adam-mini did not compromise on performance and supported multiple frameworks for multi-GPU training with just a line of change in code."


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/">PDF Retrieval with Vision Language Models</a>: Connecting the ColPali model with Vespa for complex document format retrieval.</li><li><a href="https://x.com/RuoyuSun_UI/status/1811818970573603112">Tweet from Ruoyu Sun (@RuoyuSun_UI)</a>: Update on Adam-mini: Adam-mini is now compatible with @PyTorch&#39;s latest distributed training codebase &#34;Torchtitan&#34; https://github.com/pytorch/torchtitan. Check out the loss curve on Llama3...</li><li><a href="https://x.com/teortaxesTex/status/1813048518132506656?t=-gzQhY9OZKso0NvnnVq_YA&s=19">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: I guess we&#39;ll be able to run LLaMA-405B cheaply enough  Quoting gm8xx8 (@gm8xx8)   Q-Sparse: All Large Language Models can be Fully Sparsely-Activated  paper: https://arxiv.org/abs/2407.10969</li><li><a href="https://huggingface.co/vidore/colpali">vidore/colpali · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.03502">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>: Synthetic data is becoming increasingly important for accelerating the development of language models, both large and small. Despite several successful use cases, researchers also raised concerns arou...</li><li><a href="https://x.com/lateinteraction/status/1813140776869658833">Tweet from Omar Khattab (@lateinteraction)</a>: Exactly.  What&#39;s surprising even in hindsight is that the difference between working extremely well (81.3% for ColPali) and not working at all (58.8% for BiPali) is the &#34;Col&#34; part of ColPa...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1262486230659104839)** (32 messages🔥): 

> - `FlatBuffers vs Protobuf`
> - `AMD logo color discussion`
> - `Mojo GitHub search issues`
> - `Open-source status of MAX SDK`
> - `YouTube links to Mojo talks` 


- **FlatBuffers adoption insights**: Discussion highlighted [FlatBuffers](https://flatbuffers.dev/) as high-performance but struggled with tougher usage and lesser industry adoption compared to Protobufs.
   - One member mentioned that [Apache Arrow](https://arrow.apache.org/) uses FlatBuffers internally but relies on Protobuf for data transportation, indicating industry preference.
- **AMD logo color confusion**: Members discussed the historical use of green in the AMD logo, referencing various [sources](https://logos.fandom.com/wiki/AMD/Other) indicating color changes over time.
   - One participant mentioned personal theories and pointed out inconsistencies in current branding versus historical use.
- **Mojo language GitHub search issues**: Users experienced inconsistencies with GitHub search results for Mojo language repositories, seeing varying results upon repeated searches.
   - One member humorously noted errors like zero results turning into 220, pointing to issues in GitHub's search functionality.
- **MAX SDK not open source but progressive plans**: It was clarified that while [MAX SDK](https://www.modular.com/legal/max) is free to use, it is not open source yet.
   - There are plans to progressively open-source parts of it starting with components available on [GitHub](https://github.com/modularml/max).
- **Mojo talk videos on YouTube**: Excitement was expressed for upcoming parts of Mojo talks, highlighting videos like [part 2](https://www.youtube.com/watch?v=9ag0fPMmYPQ) and deep dives with Chris Lattner.
   - Members appreciated the integration of Rust features and Python compatibility in Mojo, viewing it as a promising development in systems programming and ML.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://capnproto.org)">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=9ag0fPMmYP"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=_QVs626Vn2k">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://www.modular.com/legal/max">Modular: MAX Community License</a>: The MAX SDK (&quot;MAX&quot;) Community License governs what we expect users of our software to do with it, what uses we permit and governs the usage of it.</li><li><a href="https://logos.fandom.com/wiki/AMD/Other">AMD/Other</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=8nyg_IXfnvs">Creator Of Swift On Functionial Programming</a>: All Clips are from the live stream of ThePrimeagenMe: https://twitch.tv/ThePrimeagenCo-host: https://twitch.tv/teej_dvChris Lattner: https://x.com/clattner_l...</li><li><a href="https://www.youtube.com/watch?v=9ag0fPMmYPQ">Mojo🔥: a deep dive on ownership with Chris Lattner</a>: Learn everything you need to know about ownership in Mojo, a deep dive with Modular CEO Chris LattnerIf you have any questions make sure to join our friendly...</li><li><a href="https://github.com/search?q=language%3AMojo&type=repositories&ref=advsearch">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1812972838707687889>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1262582156455055400)** (1 messages): 

> - `MAX Graph API`
> - `AI inference pipeline`
> - `Mojo` 


- **Modular introduces MAX Graph API Tutorial**: Modular posted a new video on YouTube titled [MAX Graph API Tutorial](https://www.youtube.com/watch?v=dhllDwVUP5s), discussing how the MAX Graph API can build an AI inference pipeline in **Mojo**.
   - *Ehsan M. Kermani* elaborates on getting started with **MAX Graph API** in the video.
- **Mojo powers AI inference pipeline**: The **MAX Graph API** allows for the construction of a complete AI inference pipeline using **Mojo**.
   - The video provides a detailed guide for developers looking to leverage this powerful toolset.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=dhllDwVUP5s">MAX Graph API Tutorial</a>: The MAX Graph API allows you to build your entire AI inference pipeline in Mojo. In this video Ehsan M. Kermani discusses how you can get started with MAX Gr...

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1262744562083037275)** (8 messages🔥): 

> - `Mojo with local Whisper`
> - `Mistral 7B coding model`
> - `Mamba models`
> - `GUI example for Mistral 7B`
> - `ONNX conversion` 


- **Mojo with local Whisper queried**: A user inquired if anyone has tried using **Mojo** with **local Whisper** from **OpenAI**.
   - No answers or expanded discussions followed the query.
- **Mistral unveils 7B coding model using Mamba2**: Mistral has released a [7B coding model](https://mistral.ai/news/codestral-mamba/) that uses **Mamba2** instead of transformers, available for free use under the Apache 2.0 license.
   - The Mamba models offer linear time inference and the ability to handle sequences of infinite length, significantly boosting code productivity.
- **GUI example for Mistral 7B Coding**: A GitHub repo for building a GUI for **Mistral 7B Coding** using the nightly build was shared, with the code available [here](https://github.com/modularml/max/tree/nightly/examples/gui).
   - Various community members showed interest, offering support and sharing more examples and notebooks for ONNX conversion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: As a tribute to Cleopatra, whose glorious destiny ended in tragic snake circumstances, we are proud to release Codestral Mamba, a Mamba2 language model specialised in code generation, available under ...</li><li><a href="https://mistral.ai">Mistral AI | Frontier AI in your hands</a>: Frontier AI in your hands</li><li><a href="https://github.com/modularml/max/blob/main/examples/notebooks/mistral7b-python-onnx.ipynb">max/examples/notebooks/mistral7b-python-onnx.ipynb at main · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max</li><li><a href="https://github.com/modularml/max/tree/nightly/examples/gui">max/examples/gui at nightly · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max</li><li><a href="https://github.com/modularml/max/blob/nightly/examples/gui/pages/bert.py">max/examples/gui/pages/bert.py at nightly · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max</li><li><a href="https://github.com/modularml/max/blob/794cc173280b59fd9ad4a9c1fd498b633379b9b9/examples/gui/pages/llama3.py#L140">max/examples/gui/pages/llama3.py at 794cc173280b59fd9ad4a9c1fd498b633379b9b9 · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1262497840475934801)** (154 messages🔥🔥): 

> - `Error Handling in Mojo`
> - `Python Compatibility Mode`
> - `Discussion on Function Coloring`
> - `Dynamic vs Auto Typed Variables`
> - `GPU and FPGA Error Handling` 


- **Debate on Python Compatibility Mode**: Members discussed the possibility of a configuration in Mojo to disable full backward compatibility with Python, aiming to enforce a more Mojo-centric syntax like using `fn` instead of `def`.
   - Suggestions included disallowing exception handling similar to Rust's monadic error handling to improve code robustness and readability.
- **Error Handling Techniques in Mojo**: A lively discussion emerged around error handling, with proposals to mark functions that raise specific errors explicitly, akin to Rust's Result type.
   - Concerns were raised about stack unwinding and the difficulty of API changes if new error types are introduced without proper handling.
- **Exploration of Dynamic vs Auto Typed Variables**: Members debated whether dynamic typing is always faster than auto typing, with discussions highlighting the benefits of pre-allocated memory and type conversion checked ahead of time.
   - It was noted that actual runtime experience might not always align, and developers should not make blanket assumptions.
- **Function Coloring and Its Impacts on Code Robustness**: The issue of function coloring, comparing raising functions, and async functions was debated, with a focus on whether functions should propagate all potential errors.
   - Many agreed that raising an error should be explicitly declared to increase the chances of proper handling, thus enhancing code robustness.
- **Error Handling in GPU and FPGA Contexts**: There was significant concern about how Mojo's error handling will work on non-CPU devices like GPUs and FPGAs.
   - Members emphasized the need for error handling that avoids stack manipulation to be compliant with hardware limitations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/1746,">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://youtu.be/_QVs626Vn2k?t=1390)">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://youtu.be/_QVs626Vn2k?t=16740)">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio">no title found</a>: no description found</li><li><a href="https://github.com/martinvuyk/forge-tools/blob/main/src/forge_tools/collections/result.mojo">forge-tools/src/forge_tools/collections/result.mojo at main · martinvuyk/forge-tools</a>: Tools to extend the functionality of the Mojo standard library - martinvuyk/forge-tools
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1262489893758832681)** (34 messages🔥): 

> - `Modular Exclusive Partnership`
> - `NVIDIA MAX Platform Support`
> - `MAX Graph API Tutorial Issues`
> - `MAX Tensor Imports`
> - `Reliability of MAX Installations` 


- **Modular and NVIDIA's exclusive partnership detail changed**: The word 'exclusive' was removed from [Modular's announcement](https://web.archive.org/web/20231204230430/https://www.modular.com/blog/modular-partners-with-nvidia-to-bring-gpus-to-the-max-platform) about their partnership with NVIDIA, which previously emphasized exclusive technology collaboration.
   - A member noted this change and discussed the nuances of what 'exclusive partnership' could legally and technically mean.
- **Issues found during MAX Graph API Tutorial**: Users reported various issues while following the MAX Graph API tutorial, including discrepancies between the Python and Mojo script results and errors during installation.
   - One user mentioned incorrect imports due to using the wrong Jupyter kernel, while another highlighted issues with the `relu6` activation function in the example code.
- **MAX Tensor and Graph import problems**: A user couldn't import `max.tensor` or `max.graph` due to using a Python Jupyter kernel instead of Mojo, which was clarified by a community member.
   - Upon switching to the correct kernel, the user was able to proceed with the tutorial successfully.
- **MAX installation and export path confusion**: Several users faced issues with MAX installation, particularly related to export paths and using the correct version.
   - A member suggested using the nightly builds for a more reliable installation experience, which resolved the issues for some users.
- **Request for more verbose reporting features in MAX**: A user requested more detailed reports from MAX, including metrics like GFlops and execution time, for better hardware and financial decision-making.
   - They emphasized the need for these metrics to scale MAX usage effectively on different hardware setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/max-graph-api-tutorial">Modular: MAX Graph API tutorial</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: MAX Graph API tutorial</li><li><a href="https://web.archive.org/web/20231204230430/https://www.modular.com/blog/modular-partners-with-nvidia-to-bring-gpus-to-the-max-platform">Modular: Modular partners with NVIDIA to bring GPUs to the MAX Platform</a>: We are building a next-generation AI developer platform for the world. Read our latest post on how Modular partners with NVIDIA to bring GPUs to the MAX Platform
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1262550767144013924)** (53 messages🔥): 

> - `VSCode nightly extension for LSP`
> - `Proposal for statuses on PRs`
> - `ComplexSIMD vector implementation`
> - `Handling reviews and discussions in PRs` 


- **VSCode nightly extension fixes LSP issues**: Users discussed how to properly configure the VSCode plugin to point to a nightly version of the LSP, highlighting the need to disable the stable extension for the nightly one to take over.
   - Other members shared their methods, like uninstalling and reinstalling the extensions and ensuring the bash profile paths are properly set.
- **Proposal for better PR statuses**: A member proposed adding more detailed statuses like 'blocked/paused', 'unreviewed', and 'question or discussion' to PRs to improve communication and efficiency.
   - The community suggested using refined-github and automating labels to save maintainers' time.
- **Debate over ComplexSIMD vector structure**: Discussing ComplexSIMD, members questioned the use of two backing SIMD vectors vs. a single SIMD vector split by real and imaginary parts.
   - *Benny.n* argued for efficiency gains using a single SIMD vector, proposing formula rewrites for operations like multiplication and division.



**Link mentioned**: <a href="https://github.com/refined-github/refined-github">GitHub - refined-github/refined-github: :octocat: Browser extension that simplifies the GitHub interface and adds useful features</a>: :octocat: Browser extension that simplifies the GitHub interface and adds useful features - refined-github/refined-github

  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/)** (1 messages): 

ModularBot: Congrats <@585884735134236685>, you just advanced to level 1!
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1262780662180614196)** (1 messages): 

> - `AI Math Olympiad Winner Open Source`
> - `Whisper Timestamped Released`
> - `Nvidia BigVGAN v2`
> - `Hugging Face and Keras NLP Integration`
> - `Hugging Face Tokens UI Overhaul` 


- **AI Math Olympiad Winner Goes Open Source**: The AI Math Olympiad winner, **NuminaMath**, is now [open source](https://x.com/reach_vb/status/1811069858584363374) with a **7B model** scoring **29/50** on Kaggle test sets, licensed under Apache 2.0.
   - It uses a two-stage fine-tuning process on large math datasets and synthetic tool-integrated reasoning datasets, employing MSFT's ToRA format for **GPT-4** outputs.
- **Whisper Timestamped: Local Speech Recognition**: **Whisper Timestamped** introduces [multilingual speech recognition](https://x.com/xenovacom/status/1811068015229747335) with word-level timestamps running fully in-browser using **Transformers.js**.
   - This enables new possibilities for in-browser video editing with complete source code and demo available.
- **Nvidia BigVGAN v2 Announced**: Nvidia released [BigVGAN v2](https://x.com/reach_vb/status/1813181163126587830), a SoTA Neural Vocoder, generating waveforms from Mel spectrograms with faster inference on A100.
   - Improvements include an optimized CUDA kernel, better discriminator and loss, and the model supports up to **44 kHz sampling rate**.
- **Hugging Face and Keras NLP Integration**: Hugging Face announced a new [NLP integration](https://huggingface.co/blog/keras-nlp-integration) with **Keras**.
   - This collaboration aims to enhance NLP capabilities in Keras, providing seamless integrations for developers.
- **Hugging Face Tokens UI Overhaul**: Hugging Face revamped the [token management UI](https://x.com/coyotte508/status/1810982231180992521) on the platform, adding new features like last usage, last four characters, or rotation date.
   - The improvements make it easier to manage all your tokens with detailed information available at a glance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/reach_vb/status/1811069858584363374">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: AI Math Olympiad winner is now Open Source! 🔥  &gt; 7B model, scored 29/50 on the public and private Kaggle test sets. (Apache 2.0 licensed). &gt; Base model: deepseek-math-7b-base   &gt; Two-stage f...</li><li><a href="https://x.com/reach_vb/status/1812916171902976256)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: AI Math Olympiad Winner - Running on Mac! 100% local 🔥  brew install llama.cpp  llama-cli  --hf-repo reach-vb/NuminaMath-7B-TIR-Q8_0-GGUF      --hf-file numinamath-7b-tir-q8_0.gguf      -p &#34;For h...</li><li><a href="https://x.com/xenovacom/status/1811068015229747335)">Tweet from Xenova (@xenovacom)</a>: Introducing Whisper Timestamped: Multilingual speech recognition with word-level timestamps, running 100% locally in your browser thanks to 🤗 Transformers.js!  This unlocks a world of possibilities f...</li><li><a href="https://x.com/reach_vb/status/1813181163126587830)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Nvidia released BigVGAN v2! 🎧 SoTA Neural Vocoder - Mel spectrogram to waveform generator 🔥  &gt; Custom CUDA kernel for inference: w/ fused upsampling + activation kernel upto 3x faster inference o...</li><li><a href="https://x.com/coyotte508/status/1810982231180992521)">Tweet from coyotte508 (@coyotte508)</a>: http://hf.co/settings/tokens UI overhaul!   The UI is nicer to manage all your tokens, with added info like last usage, last four characters or rotation date.  Note: some of this info is only availabl...</li><li><a href="https://huggingface.co/settings/tokens)">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://x.com/julien_c/status/1812099420726456457)">Tweet from Julien Chaumond (@julien_c)</a>: Cool weekend update from the @huggingface datasets team,  You can now embed our viewer on any webpage 😎  (Look for the &#34;Embed&#34; button.)  Let us know what you think!</li><li><a href="https://x.com/htahir111/status/1813132485267443843)">Tweet from Hamza Tahir (@htahir111)</a>: @julien_c @huggingface @zenml_io Wrote a blog about it describing how relatively simple it is using #oss: https://www.zenml.io/blog/embedding-huggingface-datasets-visualizations-with-zenml</li><li><a href="https://x.com/mervenoyann/status/1812839137398886420)">Tweet from merve (@mervenoyann)</a>: PSA 🗣️ We kept shipping in June, here&#39;s some non-exhaustive @huggingface Hub updates</li><li><a href="https://x.com/abhi1thakur/status/1812808539963892018)">Tweet from abhishek (@abhi1thakur)</a>: We just removed the requirement to have a payment method attached to your org when creating competitions on Hugging Face 🚀 Now, universities, organizations & private individuals can create free-tier ...</li><li><a href="https://x.com/Wauplin/status/1811382409683689479)">Tweet from Wauplin (@Wauplin)</a>: 🚀 Exciting update! 𝚑𝚞𝚐𝚐𝚒𝚗𝚐𝚏𝚊𝚌𝚎_𝚑𝚞𝚋&#39;s InferenceClient now supports OpenAI&#39;s client syntax. Switch to open-source LLMs with just 3 lines of code! Check out the seamless transition...</li><li><a href="https://x.com/dylan_ebert_/status/1812952230825500914)">Tweet from dylan (@dylan_ebert_)</a>: 🎉 Good News  The final units of the 🤗 Machine Learning for 3D course have launched!  🛠️ Build your own Generative 3D demo 🎓 Get your certification  free & open source on Hugging Face</li><li><a href="https://www.youtube.com/watch?v=HcpUP-q2Z0w&ab_channel=HuggingFace)">One Minute Gradio #2: Event Chaining</a>: One Minute Gradio #2 - Learn Gradio tips and tricks quickly! Today, we&#39;ll discuss running consecutive events in Gradio (i.e. event chaining), specifically us...</li><li><a href="https://x.com/_philschmid/status/1811416175865122954)">Tweet from Philipp Schmid (@_philschmid)</a>: LLM Evaluation doesn&#39;t need to be complicated. You don&#39;t need complex pipelines, databases or infrastructure components to get started building an effective evaluation pipeline.👀  Blog: https...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1262489924846878841)** (235 messages🔥🔥): 

> - `Troubleshooting issues with Hugging Face Spaces`
> - `GPTs agents' learning capabilities`
> - `Handling tokenization for unknown words in LLMs`
> - `Merging techniques for specialized agents`
> - `Validating models for 3D mesh object similarity` 


- **Resolving Hugging Face Spaces runtime errors**: Members discussed various runtime errors encountered in Hugging Face Spaces, including CUDA errors and tokenizer issues in their models hosted on the platform.
   - Several troubleshooting steps were suggested such as changing dataset sizes, updating diffusers version, and setting cache directories, though these did not resolve all issues.
- **GPTs agents can't learn after initial training**: A concern was raised about GPTs agents not learning from additional information provided after their initial training, and it was clarified that uploaded files are saved as 'knowledge' files but do not modify the agent's base knowledge.
   - This sparked further discussions about the abilities and limitations of GPTs agents in dynamically updating their knowledge base.
- **Tokenization challenges for unknown words**: A question was raised about how LLMs use words that aren't in their tokenizer, leading to explanations about tokenization strategies and sub-word tokenization techniques.
   - Members discussed the intricacies of vocab size and how different tokenizers handle unknown words, including using smaller tokens to construct the new words.
- **Enhancing LLM performance with specialized datasets**: One member shared their experience training LLMs on specialized logic and reasoning datasets, mentioning specific datasets like Orca and Tess.
   - The idea of improving model performance using meticulously curated datasets spurred further interest and discussions.
- **Selecting models for 3D mesh object validation**: A member sought help with confirming models for calculating similarity between prompts and preview images for a 3D mesh object validation pipeline.
   - Suggestions included using CLIP models for similarity and ensuring proper validation mechanisms for accurate assessments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/phew-robertdowneyjr-tonystark-ironman-avengers-gif-2884296381752559184">Phew Robertdowneyjr GIF - Phew RobertDowneyJr TonyStark - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/soldier-ww2-traumatized-meme-eyes-gif-12257475272172704406">Soldier Ww2 GIF - Soldier Ww2 Traumatized - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/bartowski/gemma-2-27b-it-GGUF/tree/main">bartowski/gemma-2-27b-it-GGUF at main</a>: no description found</li><li><a href="https://tenor.com/view/scooby-doo-mystery-machine-cartoon-old-school-smoking-gif-16100024">Scooby Doo Mystery Machine GIF - Scooby Doo Mystery Machine Cartoon - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/nroggendorff/epicrealismxl">epiCRealism XL - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://huggingface.co/spaces/nroggendorff/animexl">Anime Diffusion XL - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://github.com/huggingface/community-events/tree/main/jax-controlnet-sprint">community-events/jax-controlnet-sprint at main · huggingface/community-events</a>: Place where folks can contribute to 🤗 community events - huggingface/community-events</li><li><a href="https://huggingface.co/spaces/nroggendorff/llava/commit/3950336734fa093dc80ac7e5860251de9e11e26b">Update README.md · nroggendorff/llava at 3950336</a>: no description found</li><li><a href="https://tenor.com/view/sonic-running-funny-weird-gif-14261571">Sonic Running GIF - Sonic Running Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/urchade/GLiNER">GitHub - urchade/GLiNER: Generalist and Lightweight Model for Named Entity Recognition (Extract any entity types from texts) @ NAACL 2024</a>: Generalist and Lightweight Model for Named Entity Recognition (Extract any entity types from texts) @ NAACL 2024 - urchade/GLiNER</li><li><a href="https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1">GLiNER-medium-v2.1, zero-shot NER - a Hugging Face Space by tomaarsen</a>: no description found</li><li><a href="https://huggingface.co/spaces/nroggendorff/llava/discussions/2">nroggendorff/llava · idiot noa... says llama....</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1262628134260314183)** (2 messages): 

> - `K-Means Clustering Video`
> - `UDOP Paper Discussion` 


- **YouTube Tutorial on K-Means Clustering**: A [YouTube video titled 'K-Means Clustering (ML pt 5)'](https://youtu.be/x1Dcg4JWARY) was shared, promising a friendly, short introduction to K-Means Clustering.
- **Queries on UDOP Paper Image Reconstruction**: A member asked how the font style is retained in image reconstruction for titles and serial numbers in the **UDOP** paper.



**Link mentioned**: <a href="https://youtu.be/x1Dcg4JWARY">K-Means Clustering ( ML pt 5 )</a>: In this video, I will talk about K - Means Clustering k-MC . It&#39;s going to be a friendly, short introduction, just like all the other videos in the playlist,...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1262761173858914316)** (8 messages🔥): 

> - `Happy Dog Detection`
> - `Retrieval Tutorials`
> - `Online Censorship Impact`
> - `Mistral AI Models`
> - `Llama3 405b` 


- **Happy Dog Detection project update**: A member shared a link to the [Happy Dog Detection GitHub repository](https://github.com/Matthew-AI-Dev/Happy_Dog_Detection), which aids in the development and contribution for a dog detection machine learning model.
   - *Contribute to Matthew-AI-Dev/Happy_Dog_Detection development by creating an account on GitHub.*
- **Retrieval Tutorials for enthusiasts**: A member found a valuable resource for retrieval enthusiasts and shared the [FullStackRetrieval-com/RetrievalTutorials GitHub repository](https://github.com/FullStackRetrieval-com/RetrievalTutorials).
   - *Contribute to FullStackRetrieval-com/RetrievalTutorials development by creating an account on GitHub.*
- **PETS24 paper on online censorship's impact**: The **impacts of online censorship on large language models** were discussed and a member shared a [PETS24 paper](https://www.petsymposium.org/foci/2024/foci-2024-0006.pdf) on the topic.
- **Mistral AI releases two models**: **Mistral AI** announced the release of two new AI models: [Mathstral](https://mistral.ai/news/mathstral/) and [Codestral](https://mistral.ai/news/codestral-mamba/).
   - The member also shared an [official announcement link](https://x.com/MistralAI/status/1813222156265791531) on X (formerly Twitter).
- **Llama3 405b added to OpenRouterAI**: A member noted that **Llama3 405b** has been added to [OpenRouterAI](https://x.com/HCSolakoglu/status/1812984327510085883) but no provider has been established yet.
   - Another member speculated about the release being close, given the attachment of Huggingface model weights page.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/HCSolakoglu/status/1812984327510085883">Tweet from Hasan Can (@HCSolakoglu)</a>: Llama3 405b has been added to @OpenRouterAI , there is no provider yet. Huggingface model weights page was also attached. Release must be close.</li><li><a href="https://x.com/MistralAI/status/1813222156265791531">Tweet from Mistral AI (@MistralAI)</a>: https://mistral.ai/news/mathstral/ https://mistral.ai/news/codestral-mamba/</li><li><a href="https://github.com/Matthew-AI-Dev/Happy_Dog_Detection">GitHub - Matthew-AI-Dev/Happy_Dog_Detection</a>: Contribute to Matthew-AI-Dev/Happy_Dog_Detection development by creating an account on GitHub.</li><li><a href="https://github.com/FullStackRetrieval-com/RetrievalTutorials">GitHub - FullStackRetrieval-com/RetrievalTutorials</a>: Contribute to FullStackRetrieval-com/RetrievalTutorials development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1262647301050994698)** (1 messages): 

> - `NLP Roadmap`
> - `NLP Projects Repository`
> - `NLP Historical Overview`
> - `NLP TOC` 


- **Check Out the NLP Roadmap GitHub Repo!**: A user shared a [GitHub repository for NLP projects](https://github.com/kjdeveloper8/nlp-projects), which contains a comprehensive roadmap for NLP enthusiasts.
   - The repository aims to guide learners through the different stages and techniques in Natural Language Processing (NLP).
- **Comprehensive NLP Roadmap Article Published on Medium**: An enlightening [Medium article on NLP Roadmap](https://medium.com/@krinaljoshi/nlp-roadmap-2740a1029af2) was shared, providing a historical overview and the evolution of NLP techniques.
   - The article references the foundational work in the mid-1930s, Noam Chomsky's contributions in 1957, and the impact of machine learning algorithms in the 1980s.
- **Essential NLP Topics Covered**: The NLP Roadmap includes essential topics such as [Basics Of NLP](#84be), [Text preprocessing](#8eb3), [Parser](#7dcf), [Text encoding](#a95f), [Text classification](#dca0), and [Text similarity](#733e), providing a structured learning path.
   - *'Language is the road map of a culture. It tells you where its people come from and where they are going.'* — Rita Mae Brown is quoted in the article to emphasize the cultural impact of language.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/kjdeveloper8/nlp-projects">GitHub - kjdeveloper8/nlp-projects: NLP Roadmap</a>: NLP Roadmap. Contribute to kjdeveloper8/nlp-projects development by creating an account on GitHub.</li><li><a href="https://medium.com/@krinaljoshi/nlp-roadmap-2740a1029af2">NLP Roadmap</a>: “Language is the road map of a culture. It tells you where its people come from and where they are going.” — Rita Mae Brown
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1262838330660360255)** (1 messages): 

> - `Best LLM for course-specific AI model`
> - `Video transcription tools`
> - `Fine-tuning on low-end hardware` 


- **Best free LLM for course-specific AI model**: Ritikkumarv is looking for an open-source LLM to create an AI model capable of answering graduate-level questions for a specific course, handling PDFs, PPTs, and video lecture transcriptions.
- **Video transcription tools discussion**: Ritikkumarv inquired whether OpenAI's Whisper is the best free tool for video transcriptions or if there are other viable options.
- **Tips for fine-tuning and low-end hardware usage**: Ritikkumarv also requested a step-by-step guide on fine-tuning the model using free, open-source software, with consideration for low-end hardware limitations.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1262627046702452776)** (1 messages): 

> - `Skin Cancer Detection`
> - `3D Images`
> - `Kaggle Competitions` 


- **Discussion on Skin Cancer Detection for 3D Images in Kaggle**: A member asked if anyone is working on **skin cancer detection** for **3D images** in **Kaggle**.
   - *No specific responses or links were discussed in the provided message history.*
- **General Inquiry**: The discussion primarily involved inquiry about ongoing projects related to **skin cancer detection in 3D images** on **Kaggle**.
   - *Further details or resources were not provided in the message history.*


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1262670965213761547)** (9 messages🔥): 

> - `NLP basic to advance course recommendations`
> - `Google Colab and GPU issues`
> - `Image embeddings and potential bias`
> - `Sentence transformers: multiple negatives vs. single negative`
> - `Vector distribution in Faiss index` 


- **NLP Learning Path Recommendations**: A member suggested starting NLP learning with HuggingFace courses or Andrew Ng's courses, and recommended hands-on projects like fine-tuning an NER model.
   - *'Keep picking up new projects whilst learning key concepts on the side'* was advised.
- **Google Colab T4 GPU Issue**: A member encountered errors using T4 GPU in Google Colab despite installing required packages (`transformers[torch]` and `accelerate`).
   - They inquired if anyone had ideas for resolving these issues.
- **Image Embeddings Bias**: A member suggested focusing on image embeddings but cautioned about potential biases in generated image descriptions.
   - *'It could be a source of bias if there were any hallucinations when generating the image descriptions.'*
- **Multiple Negatives in Sentence Transformers**: A member queried the difference in training quality for sentence transformers using multiple negatives versus a single negative, and the process of selecting hard negatives.
   - They shared confusion about data loading and the utilization of the MultipleNegativesRankingLoss class and linked relevant documentation ([TripletReader](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/readers/TripletReader.py), [MultipleNegativesRankingLoss](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py)).
- **Vector Distribution and Faiss Index**: A member focused on ensuring evenly distributed vectors in their Faiss index to improve k-nearest neighbors (knn) search accuracy.
   - They emphasized the importance of having more negatives per anchor and diverse data for better vector isotropy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/readers/TripletReader.py">sentence-transformers/sentence_transformers/readers/TripletReader.py at master · UKPLab/sentence-transformers</a>: Multilingual Sentence &amp; Image Embeddings with BERT - UKPLab/sentence-transformers</li><li><a href="https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py">sentence-transformers/sentence_transformers/losses/MultipleNegativesRankingLoss.py at master · UKPLab/sentence-transformers</a>: Multilingual Sentence &amp; Image Embeddings with BERT - UKPLab/sentence-transformers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1262796895420547144)** (1 messages): 

> - `ViteJS usage in Gradio`
> - `ViteConf partnership`
> - `Gradio's custom component dev mode` 


- **Gradio leverages ViteJS for enhanced development**: The Gradio team has been using [ViteJS](https://vitejs.dev) for a long time, significantly improving their development experience, and began exposing Vite to users in 4.0 with their custom component dev mode.
- **Gradio partners with ViteConf for 24-hour conference**: Gradio is partnering with [ViteConf](https://viteconf.org), a 24-hour conference exploring the latest in the Vite ecosystem.
   - The conference is completely free and users can sign up [here](https://viteconf.org/24/ecosystem/huggingface).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vitejs.dev>">no title found</a>: no description found</li><li><a href="https://viteconf.org>">no title found</a>: no description found</li><li><a href="https://viteconf.org/24/ecosystem/huggingface">HuggingFace invites you to ViteConf</a>: no description found
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1262523391857000579)** (243 messages🔥🔥): 

> - `AI Morph and NSFW content`
> - `Utilizing Stable Diffusion for Anime Style`
> - `YouTube Tutorial Recommendations`
> - `Local AI Tool Development`
> - `GPU Comparisons for AI Models` 


- **AI Morph No Longer Supports NSFW**: A member complained that AI Morph, an app by Daily Joy studio, stopped supporting NSFW content, displaying a 'does not meet guidelines' message.
- **Stable Diffusion and Anime Style Transfer**: A member asked how to maintain accuracy in anime style images using Stable Diffusion, looking for control over colors, clothes, and expressions.
- **Scott Detweiler Wins as Top Tutorial**: *crystalwizard* recommended Scott Detweiler's YouTube channel for learning more about Stable Diffusion, emphasizing his quality assurance work at Stability.ai.
- **Local AI Tool Development with Integrated SD**: *capt.asic* discussed their development of a Local AI tool integrating Stable Diffusion and LLM support, mentioning they use it as their go-to tool.
- **GPU Wars: 4090 vs 7900XTX**: The community discussed the merits of NVIDIA's 4090 compared to AMD's 7900XTX, focusing on price and availability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@sedetweiler">Scott Detweiler</a>: Quality Assurance Guy at Stability.ai &amp; PPA Master Professional Photographer  Greetings!  I am the lead QA at Stability.ai as well as a professional photographer and retoucher based near Milwaukee...</li><li><a href="https://github.com/leejet/stable-diffusion.cpp">GitHub - leejet/stable-diffusion.cpp: Stable Diffusion in pure C/C++</a>: Stable Diffusion in pure C/C++. Contribute to leejet/stable-diffusion.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/ssitu/ComfyUI_UltimateSDUpscale/discussions/65">ultimate sd upscale in comfyui. keep getting error · ssitu/ComfyUI_UltimateSDUpscale · Discussion #65</a>: Cannot import C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_UltimateSDUpscale module for custom nodes: cannot import name &#39;devices&#39; from &#39;modules&#39; (C:\ComfyUI_windows_portab...</li><li><a href="https://github.com/axodox/axodox-machinelearning">GitHub - axodox/axodox-machinelearning: This repository contains a pure C++ ONNX implementation of multiple offline AI models, such as StableDiffusion (1.5 and XL), ControlNet, Midas, HED and OpenPose.</a>: This repository contains a pure C++ ONNX implementation of multiple offline AI models, such as StableDiffusion (1.5 and XL), ControlNet, Midas, HED and OpenPose. - axodox/axodox-machinelearning</li><li><a href="https://github.com/ssitu/ComfyUI_UltimateSDUpscale">GitHub - ssitu/ComfyUI_UltimateSDUpscale: ComfyUI nodes for the Ultimate Stable Diffusion Upscale script by Coyote-A.</a>: ComfyUI nodes for the Ultimate Stable Diffusion Upscale script by Coyote-A. - ssitu/ComfyUI_UltimateSDUpscale</li><li><a href="https://arxiv.org/html/2312.03079v1">LooseControl: Lifting ControlNet for Generalized Depth Conditioning</a>: no description found</li><li><a href="https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb">Google Colab</a>: no description found</li><li><a href="https://civitai.com/images/19990254">Video posted by gaia123</a>: no description found</li><li><a href="https://github.com/comfyanonymous/ComfyUI/blob/master/requirements.txt">ComfyUI/requirements.txt at master · comfyanonymous/ComfyUI</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://github.com/CaptainASIC/AI-Garage">GitHub - CaptainASIC/AI-Garage: A Set of AI tools consolidated into one launcher.</a>: A Set of AI tools consolidated into one launcher. Contribute to CaptainASIC/AI-Garage development by creating an account on GitHub.</li><li><a href="https://github.com/Gourieff/comfyui-reactor-node">GitHub - Gourieff/comfyui-reactor-node: Fast and Simple Face Swap Extension Node for ComfyUI</a>: Fast and Simple Face Swap Extension Node for ComfyUI - Gourieff/comfyui-reactor-node
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1262494017929871473)** (6 messages): 

> - `CUDA Kernel Invocation in Python Scripts`
> - `Performance Comparison in CUDA and PyTorch Mat Mul Implementations`
> - `Torch Profiler Usage` 


- **Investigating CUDA Kernel Invocations**: A member asked about determining which CUDA kernels are invoked in a **Python script** where PyTorch is calling the kernels, not the user.
   - Another member suggested using the **PyTorch profiler** to look at what ATen functions get deployed.
- **CUDA vs PyTorch Mat Mul Performance**: A member referenced [a lecture](https://youtu.be/4sgKnKbR-WE?t=4045) where a CUDA kernel for matrix multiplication took **6ms**, while the PyTorch implementation only took **2ms**.
   - The member questioned if PyTorch provides the best kernels for convolution operations in **CNNs**, received a response that PyTorch usually provides good kernels for common operations, but custom kernels are beneficial for certain cases.



**Link mentioned**: <a href="https://youtu.be/4sgKnKbR-WE?t=4045),">Lecture 3: Getting Started With CUDA for Python Programmers</a>: Recording on Jeremy&#39;s YouTube https://www.youtube.com/watch?v=nOxKexn3iBoSupplementary Content: https://github.com/cuda-mode/lecture2/tree/main/lecture3Speak...

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1262824285983084736)** (25 messages🔥): 

> - `GPU Performance Issues`
> - `PyTorch Profiler Export Times`
> - `Custom Kernels and Thunder Compiler` 


- **Laptop GPU causing performance issues**: A member mentioned slow GPU performance on their laptop and sought advice on potential reasons.
   - Another member suggested that training runs of any reasonable size should not be conducted on a laptop.
- **PyTorch profiler export time concern**: A member questioned whether taking 30 minutes to export a trace using the PyTorch profiler was normal.
   - Discussion indicated that capturing a lot of information or using the `profile_memory` option might contribute to longer export times.
- **Using nvfuser's custom kernels**: One member highlighted using nvfuser's custom fusion kernels and found the project very useful.
   - Concerns about long export times were noted, but no specific solutions or optimizations were suggested.


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1262562492824027157)** (1 messages): 

> - `SCALE GPGPU programming toolkit`
> - `Compiling CUDA for AMD GPUs`
> - `SCALE support for more GPU vendors`
> - `SCALE tutorial and examples` 


- **SCALE by Spectral Compute**: [SCALE](https://docs.scale-lang.com/) is a GPGPU programming toolkit enabling CUDA applications to be natively compiled for AMD GPUs without modifying the original CUDA program or its build system.
- **SCALE support and resources**: Support for more GPU vendors and CUDA APIs is **in development**, and a [tutorial](manual/how-to-use/) and [examples](examples/) are available for getting started.



**Link mentioned**: <a href="https://docs.scale-lang.com/">SCALE documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1262526637904498800)** (9 messages🔥): 

> - `Suno hiring ML engineers`
> - `Suno looking for torch.compile and triton experts`
> - `Cutlass not required but encouraged`
> - `Suno hiring interns for ML roles` 


- **Suno hiring ML engineers for real-time audio models**: Suno is hiring ML engineers to work on **training and inference** for large models that stream audio in real-time to millions of users, with skills in **torch.compile**, **triton**, **fast diffusion/FM sampling**, **vLLM**, or **large-scale distributed training**. [Job Posting](https://jobs.ashbyhq.com/suno/7522d696-7ce8-4ece-a983-4be03dffde20)
- **Suno looking for torch.compile and triton enthusiasts**: Suno specifically seeks expertise in **torch.compile**, **triton**, and other high-speed ML methods for their projects.
   - They are not currently looking for **Cutlass** experts but encourage those with similar skills to apply anyway.
- **Suno offering ML internship roles**: Suno is open to hiring interns for the same type of work as their full-time roles in machine learning engineering.



**Link mentioned**: <a href="https://jobs.ashbyhq.com/suno/7522d696-7ce8-4ece-a983-4be03dffde20">Machine Learning Infrastructure Engineer</a>: We’re looking for early members of our machine learning team. You’ll work closely with the founding team and have ownership of a wide variety of technical decisions on how we build and deploy our stat...

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1262594919206227981)** (3 messages): 

> - `Lightning AI's Studios`
> - `Huggingface Spaces Dev Mode`
> - `CUDA development` 


- **Lightning AI's Studios Excites Users**: A member recommended [Lightning AI's Studios](https://lightning.ai/docs/overview/studios#studios-) as an appealing solution, noting its pay-as-you-go model and free tier.
   - Another member responded positively, stating *'This looks great. Thank you'.*
- **Huggingface Spaces Dev Mode for CUDA?**: A member asked if it's possible to use [Huggingface Spaces Dev Mode](https://huggingface.co/spaces/dev-mode-explorers/README) for CUDA development.
   - There were no responses, leaving the question open for further discussion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lightning.ai/docs/overview/studios#studios-are-friendly-and-powerful">Studios ⚡️ Lightning AI</a>: Lightning AI Studio is an AI development platform. Studios&amp;nbsp;&lt;b&gt;run on the browser&lt;/b&gt; or &lt;b&gt;your own cloud infrastructure&lt;/b&gt;. Use it to code on the cloud, build, train...</li><li><a href="https://lightning.ai/docs/overview/studios#studios-">Studios ⚡️ Lightning AI</a>: Lightning AI Studio is an AI development platform. Studios&amp;nbsp;&lt;b&gt;run on the browser&lt;/b&gt; or &lt;b&gt;your own cloud infrastructure&lt;/b&gt;. Use it to code on the cloud, build, train...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

andreaskoepf: Anyone tried out Mosaic GPU yet? https://x.com/apaszke/status/1812897008031617493
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1262842494601658441)** (3 messages): 

> - `unwrap_tensor_subclass in torch.compile`
> - `FakeTensors in model compilation` 


- **Unexpected FakeTensor weights after compilation**: A member was confused about `unwrap_tensor_subclass` seemingly replacing all subclass tensors with **FakeTensors** in their model, which led to issues because these aren't handled as packed tensors.
   - *I thought FakeTensors were only used for compilation, not while running the compiled function,* the member remarked.
- **Parametrization workaround in torch.compile**: Another member clarified that `unwrap_tensor_subclass` uses **parametrization** to convert tensor subclasses to plain tensors to navigate the current limitation of **torch.compile stack** (especially aot_export).


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1262485006207418409)** (139 messages🔥🔥): 

> - `CUDA arguments renaming`
> - `Attention mechanisms`
> - `StableAdamW`
> - `AMD GPU support in llm.c`
> - `AI+Education company by Andrej Karpathy` 


- **CUDA function argument renaming discussion**: Discussion focused on renaming CUDA function arguments to avoid conflicts with global variables and the feasibility of avoiding such arguments altogether.
   - *akakak1337* noted the need to rename `multi_gpu_config` to `config`, while considering if global variables should suffice instead.
- **Alternating dense and banded attention for GPT3**: **GPT3 requires alternating attention mechanisms** between dense and banded variants for enhanced performance, similar to **Mistral** and **Gemma 2**.
   - A parameter in cuDNN allows enabling window sizes for attention, which could be crucial for implementation.
- **StableAdamW integration improves GAN training**: **StableAdamW** introduced into [ScheduleFreeAdamW](https://arxiv.org/abs/2304.13013) significantly enhances GAN training stability according to **_clashluke**.
   - *akakak1337* provided updates on merging the StableAdamW changes for improved gradient clipping results.
- **Challenges in running llm.c on AMD GPUs**: Efforts to run **llm.c** on AMD GPUs face hurdles due to missing key elements like **cublaslt** and **bfloat16 support**.
   - **SCALE**'s approach for CUDA apps on AMD lacks crucial aspects, making it less effective compared to **hipify**.
- **Andrej Karpathy launches AI+Education company Eureka Labs**: **Andrej Karpathy** introduced [Eureka Labs](https://x.com/karpathy/status/1813263734707790301), aiming to create AI-assisted educational tools, starting with a course named LLM101n.
   - The project emphasizes leveraging AI to enhance educational reach and quality, combining Karpathy's deep AI expertise and passion for teaching.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1813263734707790301">Tweet from Andrej Karpathy (@karpathy)</a>: ⚡️ Excited to share that I am starting an AI+Education company called Eureka Labs.  The announcement:  --- We are Eureka Labs and we are building a new kind of school that is AI native.  How can we ap...</li><li><a href="https://x.com/_clashluke/status/1812938241831579990">Tweet from Lucas Nestler (@_clashluke)</a>: Following up on https://x.com/fr0sty__/status/1808664083014599103  I&#39;ve added StableAdamW (https://arxiv.org/abs/2304.13013) into ScheduleFreeAdamW. Convergence is identical in toy problems but si...</li><li><a href="https://github.com/karpathy/llm.c/pull/689">Refactor/code to zerocuh by karpathy · Pull Request #689 · karpathy/llm.c</a>: no description found</li><li><a href="https://en.cppreference.com/w/c/types/boolean">Boolean type support library - cppreference.com</a>: no description found</li><li><a href="https://gpuopen.com/learn/wmma_on_rdna3/">How to accelerate AI applications on RDNA 3 using WMMA</a>: This blog is a quick how-to guide for using the WMMA feature with our RDNA 3 GPU architecture using a Hello World example.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1262753354816294964)** (2 messages): 

> - `Sparsity`
> - `Quantized models` 


- **Sparsity and Quantized Models Innovate Efficiency**: [A paper](https://arxiv.org/pdf/2407.10969) discussed successful integration of **sparsity** and **quantized models**.
   - *mobicham* commented that this approach results in faster inference and higher precision for non-sparse weights while maintaining the same average bitrate.
- **Sparsity and Quantization Boost Performance**: A member noted that combining **sparsity** and **quantized models** offers dual benefits: enhanced speed and accuracy.
   - This method allows for efficient resource usage without compromising model quality, as shared in the discussion.


  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 messages): 

iron_bound: Neat demos
https://wgpu.rs/
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1262615220753530971)** (1 messages): 

> - `Qwen 2 7B Instruct` 


- **Free access to Qwen 2 7B Instruct model**: OpenRouter announced the availability of the [Qwen 2 7B Instruct](https://openrouter.ai/models/qwen/qwen-2-7b-instruct) model for free.
   - You can now access it on [OpenRouter](https://openrouter.ai/models/qwen/qwen-2-7b-instruct):free.
- **Qwen 2 7B Instruct model released**: The [Qwen 2 7B Instruct](https://openrouter.ai/models/qwen/qwen-2-7b-instruct) model is now available for free.
   - Learn more about the model on [OpenRouter](https://openrouter.ai/models/qwen/qwen-2-7b-instruct):free.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/qwen/qwen-2-7b-instruct>)">Qwen 2 7B Instruct by qwen</a>: Qwen2 7B is a transformer-based model that excels in language understanding, multilingual capabilities, coding, mathematics, and reasoning.  It features SwiGLU activation, attention QKV bias, and grou...</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-7b-instruct:free">Qwen 2 7B Instruct (free) by qwen</a>: Qwen2 7B is a transformer-based model that excels in language understanding, multilingual capabilities, coding, mathematics, and reasoning.  It features SwiGLU activation, attention QKV bias, and grou...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1262484303615361024)** (130 messages🔥🔥): 

> - `Google Gemini Models`
> - `GPT-4o Free Tier`
> - `Gemini 1.5 Pro Performance`
> - `OpenRouter Issues`
> - `Llama 3 Extended Context Models` 


- **Google Gemini Models Critiqued**: Members debated why Google doesn't provide a better model for Gemini's free tier, calling **Gemini 1.0** 'quite bad' compared to **GPT-4o**.
   - *'Google does know how to cook,'* noted a user of **Gemini 1.5 Pro**, pointing out its creative potential but problematic coding performance.
- **GPT-4o Impresses with Free Tier Strategy**: A member remarked that **OpenAI** succeeded by offering GPT-4o to their free tier users, setting a high bar for competitors.
- **OpenRouter Suffered Minor Outages**: Users reported sporadic outages on **OpenRouter** and experienced difficulties in accessing the site and API, triggering multiple inquiries.
   - Official responses attributed these issues to intermittent routing problems and possible Cloudflare errors, with general service resuming shortly after.
- **Seeking Llama 3 Extended Context Models**: Members expressed frustration over the **8k context window** limit of **Llama 3-70B Instruct** and the challenge of finding better alternatives.
   - Models like **Euryale** and **Magnum-72B** were suggested, but lack of consistent instruction-following and high cost are notable concerns.
- **OpenRouter Model Access and Functionality**: There was confusion over OpenRouter's service, clarifying it is not free for all models but does offer free options without a subscription.
   - **OpenRouter** provides access to various APIs and free models, with details on hosting local models still under specific business contracts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jsoneditoronline.org/#left=local.tihuto,">JSON Editor Online: edit JSON, format JSON, query JSON</a>: JSON Editor Online is the original and most copied JSON Editor on the web. Use it to view, edit, format, repair, compare, query, transform, validate, and share your JSON data.</li><li><a href="https://openrouter.ai/playground">Playground | OpenRouter</a>: Experiment with different models and prompts</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">Llama 3 Euryale 70B v2.1 by sao10k</a>: Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k).  - Better prompt adherence. - Better anatomy / spatial awareness. - Adapts much better to unique and c...</li><li><a href="https://openrouter.ai/models/alpindale/magnum-72b">Magnum 72B by alpindale</a>: From the maker of [Goliath](https://openrouter.ai/models/alpindale/goliath-120b), Magnum 72B is the first in a new family of models designed to achieve the prose quality of the Claude 3 models, notabl...</li><li><a href="https://www.together.ai/pricing">Together Pricing | The Most Powerful Tools at the Best Value</a>: Get detailed pricing for inference, fine-tuning, training and Together GPU Clusters.</li><li><a href="https://tenor.com/view/wizard101-0bobux-wallet-empty-wallet-empty-gif-22389933">Wizard101 0bobux GIF - Wizard101 0Bobux Wallet - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/models?o=newest&max_price=0)">Models - Newest | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended">Llama 3 Lumimaid 8B (extended) by neversleep</a>: The NeverSleep team is back, with a Llama 3 8B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessar...</li><li><a href="https://openrouter.ai/models/sao10k/l3-stheno-8b">Llama 3 Stheno 8B v3.3 32K by sao10k</a>: Stheno 8B 32K is a creative writing/roleplay model from [Sao10k](https://ko-fi.com/sao10k). It was trained at 8K context, then expanded to 32K context.  Compared to older Stheno version, this model is...</li><li><a href="https://openrouter.ai/models/ne">Models: &#x27;ne&#x27; | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://openrouter.onlineornot.com/">OpenRouter Status</a>: OpenRouter Incident History
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1262530015178915871)** (110 messages🔥🔥): 

> - `GPT Issues with Pasted Values`
> - `Model Settings for Different Collections`
> - `Perplexity Office`
> - `Gemini AI Details`
> - `Pro Subscription Support` 


- **GPT Issues with Pasted Values**: Members discussed problems when pasting values as attachments and getting unrelated, generic responses from [multiple models](https://perplexity.ai).
- **Model Settings for Different Collections**: A query was raised about [assigning different models](https://perplexity.ai) for different collections or threads, like chatgpt4o for CollectionA and Opus for CollectionB.
- **Perplexity Office Announced**: A member shared a tweet announcing a [new Perplexity office](https://x.com/AravSrinivas/status/1812890154367078590), exciting the community.
- **Gemini AI Performance Details**: Members shared a [link](https://deepmind.google/technologies/gemini/) to DeepMind's Gemini AI specs, highlighting its performance on different datasets in 2024.
- **Pro Subscription Support Issues**: Multiple users reported issues [activating their Pro subscriptions](https://perplexity.ai/settings/account) on both PC and iOS, despite using the same email and receiving confirmation emails.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/cat-questioning-life-questioning-life-what-is-life-gif-4882578">Cat Questioning GIF - Cat Questioning Life - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/AravSrinivas/status/1812890154367078590">Tweet from Aravind Srinivas (@AravSrinivas)</a>: New Perplexity Office!</li><li><a href="https://www.perplexity.ai/settings/account">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://deepmind.google/technologies/gemini/">Gemini</a>: The Gemini family of models are the most general and capable AI models we&#39;ve ever built. They’re built from the ground up for multimodality — reasoning seamlessly across text, code, images, audio....</li><li><a href="https://python-fiddle.com/saved/NZfCYDD2l6h51DL8vZ0a">Python-Fiddle: Online Python Compiler, IDE, and Interpreter</a>: Run Python code in your browser. Share code snippets with others.</li><li><a href="https://python-fiddle.com/saved/CG2EpDwjRDz3uSq2sAEc">Python-Fiddle: Online Python Compiler, IDE, and Interpreter</a>: Run Python code in your browser. Share code snippets with others.</li><li><a href="https://www.perplexity.ai/search/f-t-3-e-t-sin">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/f-t-3-e-t-sin-5pi-t-plot-this-383zGYadRV.qbZnMUZFXlg">f(t) = 3 * e^(-t) * sin(-5pi*t) 

plot this using plotly with additional upper...</a>: Based on the instructions and search results, I&#x27;ll provide a detailed explanation of how to plot the given function using Plotly, including upper and lower...</li><li><a href="https://www.wolframalpha.com/input?i=f%28t%29+%3D+3+*+e%5E%28-t%29+*+sin%282*2pi*t%29+from+t+%3D+0+to+5">f(t) = 3 * e^(-t) * sin(2*2pi*t) from t = 0 to 5 - Wolfram|Alpha</a>: Wolfram|Alpha brings expert-level knowledge and capabilities to the broadest possible range of people—spanning all professions and education levels.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1262623402230026372)** (3 messages): 

> - `Alphabet $23B Deal`
> - `7-Eleven's Upgrade`
> - `New Zealand's Rare Whale Discovery`
> - `Accessible Lunar Cave`
> - `Perplexity AI Pro Features` 


- **Alphabet inks $23B Deal**: **Alphabet** strikes a $23 billion deal, positioning itself strategically in a market shake-up. [Watch the summary on YouTube](https://www.youtube.com/embed/lKn8rh0pOiM) for more details.
- **7-Eleven introduces major upgrade**: **7-Eleven** announces an upgrade, likely enhancing consumer experiences at their locations. [Check out more insights here](https://www.youtube.com/embed/lKn8rh0pOiM).
- **Scientists discover accessible lunar cave**: An **accessible lunar cave** has been discovered in Mare Tranquillitatis using advanced radar imaging techniques. Estimated to be at least 130 feet wide and tens of yards long, this cave lies approximately 150 meters below the moon's surface.
   - The discovery suggests potential protection from harsh lunar conditions and stable temperatures, making it valuable for future lunar exploration and habitation. [More details on Perplexity AI](https://www.perplexity.ai/search/moon-s-hidden-refuge-scientist-yz19IMD.TE6E4fZj9A9W.Q#0).
- **Perplexity AI Pro offers new features**: Perplexity AI introduces new Pro features, including image upload and smarter AI capabilities. [Explore these added offerings for enhanced search experience](https://www.perplexity.ai/search/In-Batch-how-GStLqBGjTpqscDMMVEcyOA).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/lKn8rh0pOiM">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/moon-s-hidden-refuge-scientist-yz19IMD.TE6E4fZj9A9W.Q#0">Moon&#x27;s Hidden Refuge: Scientists Uncover Potential Lunar Base in Underground...</a>: Moon&#x27;s Hidden Refuge: Scientists Uncover Potential Lunar Base in Underground Cave  In a groundbreaking discovery that could reshape the future of space...</li><li><a href="https://www.perplexity.ai/search/In-Batch-how-GStLqBGjTpqscDMMVEcyOA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1262803292904489080)** (5 messages): 

> - `Removing sources in pplx-api`
> - `524 errors with sonar models`
> - `Stream mode functionality` 


- **Inquire about removing sources feature in pplx-api**: A user asked if the pplx-api supports the option to remove certain sources similar to the `deleted_urls` parameter available in the UI.
   - The user's query stems from not finding such an option in the documentation.
- **524 errors reported with sonar models**: A member reported encountering 524 errors when using `llama-3-sonar-small-32k-online` and `llama-3-sonar-large-32k-online`. *The server timed out* according to another member.
   - It's suggested that enabling stream mode might help keep the connection open.
- **Enable stream mode to maintain connection**: To deal with connection issues, users can enable stream mode by passing `"stream": True`.
   - *Enable stream mode will keep the connection open,* advised a member.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1262781533891330089)** (17 messages🔥): 

> - `Codestral Mamba`
> - `Mathstral`
> - `SmolLM`
> - `Eureka Labs`
> - `Hydra Model Extension` 


- **Codestral Mamba: A New Hope for Infinite Sequences**: [Codestral Mamba](https://mistral.ai/news/codestral-mamba/) introduces a new architecture designed with help from Albert Gu and Tri Dao, offering the advantage of linear time inference and theoretical ability to model sequences of infinite length.
   - It is particularly effective for code productivity use cases, performing on par with state-of-the-art transformer-based models.
- **Mathstral Specializes in Advanced STEM Reasoning**: [Mathstral](https://mistral.ai/news/mathstral/) focuses on STEM subjects and achieves high reasoning capacities for its size, with benchmark scores of **56.6%** on MATH and **63.47%** on MMLU.
   - Developed in collaboration with [Project Numina](https://projectnumina.ai/), Mathstral exemplifies superior performance/speed tradeoffs for niche applications.
- **SmolLM: High Performance in Small Packages**: [SmolLM](https://x.com/loubnabenallal1/status/1813252390692303069?s=46) introduces new SOTA models of 135M, 360M, and 1.7B parameters, trained on high-quality web, code, and synthetic data.
   - These models outperform MobileLLM, Phi1.5, and Qwen2, highlighting the growing importance of on-device deployment of LLMs.
- **Eureka Labs Reinvents AI in Education**: [Eureka Labs](https://eurekalabs.ai/) aims to create an AI-native school, starting with an AI teaching assistant to enhance learning experiences.
   - Their first product, LLM101n, will guide students through training their own AI, expanding education's reach and depth.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://eurekalabs.ai/">Eureka Labs</a>: no description found</li><li><a href="https://mistral.ai/news/mathstral/">MathΣtral</a>: As a tribute to Archimedes, whose 2311th anniversary we're celebrating this year, we are proud to release our first Mathstral model, a specific 7B model designed for math reasoning and scientific disc...</li><li><a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: As a tribute to Cleopatra, whose glorious destiny ended in tragic snake circumstances, we are proud to release Codestral Mamba, a Mamba2 language model specialised in code generation, available under ...</li><li><a href="https://huggingface.co/spaces/HuggingFaceTB/SmolLM-360M-Instruct-WebGPU">SmolLM 360M Instruct WebGPU - a Hugging Face Space by HuggingFaceTB</a>: no description found</li><li><a href="https://x.com/loubnabenallal1/status/1813252390692303069?s=46">Tweet from Loubna Ben Allal (@LoubnaBenAllal1)</a>: On-device deployment  of LLMs is more important than ever. Today we’re releasing SmolLM a new SOTA series of 135M, 360M and 1.7B models:  - Outperforming MobileLLM, Phi1.5 and Qwen2 small models - Tra...</li><li><a href="https://x.com/_albertgu/status/1813252409071968297?s=46">Tweet from Albert Gu (@_albertgu)</a>: Releasing Hydra, our &#34;official&#34; extension of Mamba (and general state space models) to be bidirectional! Hydra is motivated from first principles by increasing expressivity through the framewo...</li><li><a href="https://x.com/karpathy/status/1813263734707790301?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: ⚡️ Excited to share that I am starting an AI+Education company called Eureka Labs.  The announcement:  --- We are Eureka Labs and we are building a new kind of school that is AI native.  How can we ap...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1262488819434000427)** (3 messages): 

> - `Circumventing Torrent Laws`
> - `Evaluation Gating in AI`
> - `Private Test Sets and Benchmark Filtering` 


- **Request for physical copies of books to avoid torrenting**: A member asked for copies of books to be mailed on a **hard drive** to **circumvent torrent laws**.
   - *No further discussion or links provided on the topic.*
- **Handling evaluation gating in AI**: A member inquired about **evaluation 'gating'** used by **GPQA** and **GAIA**, protections against accidentally scooping test sets into pretraining corpuses, and possible filtering methods.
   - Another member explained that **private test suites** involve significant time and cost, making them impractical for many academic institutions, and noted that **substrate matching** is a common filtering method.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1262855620760043520)** (3 messages): 

> - `Lobbying for Legislative Bills`
> - `Conflict of Interest`
> - `Profit from Compliance Checks`
> - `Ethics of Political Donations` 


- **Lobbying Raises Conflict of Interest Concerns**: A member highlighted a [tweet](https://fxtwitter.com/mpopv/status/1813273553477009546?s=46) discussing the unethical nature of lobbying for a bill that benefits one's own business by mandating compliance checks.
   - *Feels like if you are heavily lobbying for, and soliciting donations to lobby for, a certain legislative bill, you should probably disclose that you secretly own a company positioned to profit from that bill's passage by selling the very compliance checks the bill would mandate.*
- **Lobbying and Vested Interests Go Hand in Hand**: Another member questioned if anyone lobbies for legislative bills without a vested interest in their passage, suggesting it to be a common practice.
- **Differences in Lobbying Motivations**: There was a discussion distinguishing between preemptively funding an organization to profit from a bill versus protecting existing organizational interests.
   - One member acknowledged the distinction but agreed that the primary motive is often self-interest.



**Link mentioned**: <a href="https://fxtwitter.com/mpopv/status/1813273553477009546?s=46">Tweet from Matt Popovich (@mpopv)</a>: feels like if you are heavily lobbying for, and soliciting donations to lobby for, a certain legislative bill, you should probably disclose that you secretly own a company positioned to profit from th...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1262508170140450857)** (31 messages🔥): 

> - `State of LLM evaluations`
> - `Hypothetical uses of open-source GPT4o-class model`
> - `AI training data sources`
> - `Cost of model training`
> - `New SciCode benchmark` 


- **LLM Evaluations Stir Heated Debates**: A lively discussion surrounds the state of **LLM evaluations**, with some users expressing confusion and frustration about current evaluation methods as seen in this [tweet](https://x.com/sureailabs/status/1812949017212690472).
   - One user humorously appreciates the ongoing debates and encourages more dialogue: *I love y'all; keep yappin'.*
- **Hypothetical GPT4o-Class Model Sparks Curiosity**: Members discussed the potential of an open-source **GPT4o-class model** in terms of new research queries, as raised in this [tweet](https://x.com/swyx/status/1812988248660320679?s=46).
   - Some speculated that it could significantly reduce costs and pose a major threat to enterprise APIs; *Good for next 12-18 months of open model training (synth data), etc.*
- **Big Tech Secretly Gorges on YouTube Data**: A [Wired article](https://www.wired.com/story/youtube-training-data-apple-nvidia-anthropic/) revealed that major tech companies are using YouTube data for AI training despite platform restrictions.
   - Members argued this practice is widely recognized yet unproven for many companies, leading to debates on AI journalism standards.
- **Cost of Training Llama3 vs. GPT4o**: Discussions estimated the cost of training **Llama3** to be around **$4-5 per million tokens**, potentially matching **GPT4o** on input and reducing output costs by approximately one-third.
   - One member cited **Groq** as an even cheaper alternative, with speculative pricing at **$3.36/4.50 per million tokens**; see [this tweet](https://x.com/thexeophon/status/1813108909416325261?s=46).
- **Unveiling SciCode Benchmark: The Ultimate STEM Challenge for LMs**: **SciCode** has introduced a new benchmark challenging LMs to code solutions for scientific problems from Nobel-winning research, achieving less than **5% accuracy with GPT-4 and Sonnet 3.5** as detailed in [this thread](https://x.com/minyangtian1/status/1813182904593199553?s=46).
   - This benchmark is viewed as an essential evaluation for pretraining, drawing interest among users for its advanced and rigorous approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.wired.com/story/youtube-training-data-apple-nvidia-anthropic/">Apple, Nvidia, Anthropic Used Thousands of Swiped YouTube Videos to Train AI</a>: “It’s theft.” A WIRED investigation found that subtitles from 173,536 YouTube videos, siphoned from more than 48,000 channels, were used by Anthropic, Nvidia, Apple, and Salesforce to train AI.</li><li><a href="https://x.com/sureailabs/status/1812949017212690472">Tweet from surea.i • (in SF!?!!!!!!) (@sureailabs)</a>: my favorite type of SF conversation I don&#39;t yet fully understand is between all the people that are mad about the state of LLM evaluations.  I love y&#39;all. keep yappin</li><li><a href="https://x.com/thexeophon/status/1813108909416325261?s=46">Tweet from Xeophon (@TheXeophon)</a>: @_xjdr Not to mention Groq, which has prices so cheap its almost unfair. If you assume the pricing scales like it does from 8B-&gt;70B, you&#39;d arrive at 3.36/4.50</li><li><a href="https://x.com/minyangtian1/status/1813182904593199553?s=46">Tweet from Minyang Tian (@MinyangTian1)</a>: SciCode is our new benchmark that challenges LMs to code solutions for scientific problems from advanced papers. The challenges were crafted by PhDs;   ~10% of our benchmark is based on Nobel-winning ...</li><li><a href="https://x.com/swyx/status/1812988248660320679?s=46">Tweet from swyx 🤞 🔜 SFO (@swyx)</a>: Completely hypothetically...  what would you do with an open source GPT4o-class model that you can&#39;t do today?   What questions could you ask that delivers alpha within the bounds of new-normal AI...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1262526393774772405)** (6 messages): 

> - `MSFT Papers`
> - `WizardLM Vibes` 


- **MSFT Wizard Team Papers Released**: A member mentioned that the Wizard team at **MSFT** has released a bunch of papers this month, with a link to [Qingfeng Sun's tweet](https://x.com/victorsungo/status/1812854829397746075).
   - They plan to read and discuss if anyone's interested, but expressed disdain for *hokey heuristic bs*.
- **Mixed Reviews on WizardLM Papers**: Another member commented that they found the **WizardLM vibes** amusing.
   - *Edit: paper was kinda poop imo.*


  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1262849929026469928)** (2 messages): 

> - `Degenerate Case in Policy Reinforcement`
> - `DPO-like algorithms` 


- **Degenerate Case Useful in POL**: A member speculated that the degenerate case might be useful or necessary for common prefixes between winning and losing in policy reinforcement.
   - Another member agreed, suggesting a deeper dive into the implications of having people focus on such deep technical details.
- **DPO Algorithms Invigorate Algorithm Discussions**: A member expressed excitement about DPO-like algorithms, highlighting their role in stimulating renewed interest in algorithm discussions.
   - They noted the potential for overfitting with `losses = -F.logsigmoid(policy_rejected_logp)`, yet remained eager to explore further.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1262484066234794014)** (21 messages🔥): 

> - `Qwen Team`
> - `RewardBench Performance`
> - `Foundational Large Autorater Models`
> - `Post-training for AI models`
> - `DeepMind's New Paper on FLAMe` 


- **Qwen Team's Post-training Insights**: The [Qwen 2 post-training notes](https://x.com/natolambert/status/1813263814009495799) cover trends in synthetic data with their multi-stage alignment training showing unseen processes in open-source models.
   - Insights include **classification and filtering of samples** via tags and **execution feedback for code**, showing the efforts to improve model performance.
- **DeepMind's FLAMe Model Surpasses GPT-4**: DeepMind’s [FLAMe](http://arxiv.org/abs/2407.10817) outperforms GPT-4 & 4o in RewardBench by training on human evaluations, yet the release remains closed.
   - There was some drama about last-minute score adjustments, highlighting the competitive nature of these evaluations.
- **Bypass Paywalls Extension Recommended**: Users discussed bypassing paywalls with the **bypass-paywalls-clean** extension for reliable access to articles.
   - This was shared in response to issues accessing specific articles and resources.
- **arXiv Paper Highlights Computational Trends**: A detailed [arXiv paper](https://arxiv.org/abs/2407.10671) by An Yang and others delves into training models on large token datasets but settling on 7T tokens as more effective.
   - The paper details various phases and evaluations, aligning with processes at top AI labs.
- **Diverse Opinions on Data Sampling Strategies**: Members debated data sampling strategies, particularly whether to sample multiple responses or focus on best/worst pairs.
   - The discussion highlighted different approaches and evolving opinions on methods to enhance training diversity and preference ranking.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tuvllms/status/1813249272474968315">Tweet from Tu Vu (@tuvllms)</a>: 🚨 New @GoogleDeepMind paper 🚨  We trained Foundational Large Autorater Models (FLAMe) on extensive human evaluations, achieving the best RewardBench perf. among generative models trained solely on p...</li><li><a href="https://arxiv.org/abs/2407.10671">Qwen2 Technical Report</a>: This report introduces the Qwen2 series, the latest addition to our large language models and large multimodal models. We release a comprehensive suite of foundational and instruction-tuned language m...</li><li><a href="https://x.com/natolambert/status/1813263814009495799">Tweet from Nathan Lambert (@natolambert)</a>: Qwen 2 post-training / rlhf notes.   Not a lot of details, but a lot of common trends in synthetic data etc. It shows how many little things go into making a good model. These sorts of data processes ...
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1262485588951699548)** (24 messages🔥): 

> - `GPT-4 training on GSM8k`
> - `Instruction tuning datasets`
> - `GPU failures during training`
> - `Object counting use case`
> - `Pile 2 dataset size` 


- **GPT-4 trained on GSM8k train set**: A member noted that the original **GPT-4** trained on most of the **GSM8k train set**, as mentioned in the [GPT-4 tech report](https://link.to.report).
   - Another member expressed amazement at remembering such details, but it was noted that important facts like these are often tweeted and thus remembered easily.
- **Concerns about instruction tuning datasets**: There was a concern about the syntax used in the instruction tuning dataset, and whether OpenAI's chat app does a chain of thoughts with bullet points instead.
   - There were doubts if OpenAI would include certain markers (<<>>) in their instruction tuning dataset, hinting at possible contamination or laziness.
- **Rates of GPU failures during model training**: A member asked for examples of papers that disclose rates of **GPU failures** during the training of large AI models.
   - [Reka tech report](https://publications.reka.ai/reka-core-tech-report.pdf) and **OPT/BLOOM logbooks** were mentioned as good sources for such data.
- **Object counting implementation query**: A user sought advice on implementing **object counting** of custom objects that can vary significantly, asking for the best approach.
   - They described a potential method of taking an image, drawing a bounding box on an object, and then detecting all similar objects to get the final count.
- **Pile 2 dataset disk space inquiry**: A member inquired about the disk space required for the **Pile 2 dataset**, noting that only its token count is presently mentioned.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1262487336055472239)** (41 messages🔥): 

> - `In-context learning with SSMs`
> - `EM-LLM for infinite context handling`
> - `FLUTE for faster LLM inference`
> - `Q-Sparse for efficient sparse LLMs`
> - `Observational studies on transformer layers` 


- **State Space Models in Transformers Spark Discussion**: [A novel weight construction](https://arxiv.org/abs/2407.09375) for State Space Models enables these models to predict the next state of any dynamical system after observing previous states without parameter fine-tuning.
- **Infinite Contexts with EM-LLM**: EM-LLM integrates human episodic memory aspects into LLMs to handle [practically infinite context lengths](https://arxiv.org/abs/2407.09450) efficiently.
- **FLUTE boosts LLM inference speeds**: [FLUTE](https://github.com/HanGuo97/flute), a flexible lookup table engine for LUT-quantized LLMs, enables faster inference by minimizing bit manipulations and utilizes vectorization.
- **Q-Sparse improves LLM efficiency**: [Q-Sparse](https://arxiv.org/abs/2407.10969) applies top-K sparsification and straight-through-estimator to enable sparsely-activated LLMs to achieve results comparable to baseline LLMs with higher efficiency.
- **Empirical studies reveal transformer layer robustness**: Studies show that middle layers of pretrained transformers can handle layer-wise modifications better than lower or final layers, maintaining performance for most tasks except reasoning-heavy ones like [GSM8K](https://arxiv.org/abs/2407.09298).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.10969">Q-Sparse: All Large Language Models can be Fully Sparsely-Activated</a>: We introduce, Q-Sparse, a simple yet effective approach to training sparsely-activated large language models (LLMs). Q-Sparse enables full sparsity of activations in LLMs which can bring significant e...</li><li><a href="https://arxiv.org/abs/2407.09941">Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers</a>: A wide array of sequence models are built on a framework modeled after Transformers, comprising alternating sequence mixer and channel mixer layers. This paper studies a unifying matrix mixer view of ...</li><li><a href="https://arxiv.org/abs/2407.09450">Human-like Episodic Memory for Infinite Context LLMs</a>: Large language models (LLMs) have shown remarkable capabilities, but still struggle with processing extensive contexts, limiting their ability to maintain coherence and accuracy over long sequences. I...</li><li><a href="https://arxiv.org/abs/2407.10827">LLM Circuit Analyses Are Consistent Across Training and Scale</a>: Most currently deployed large language models (LLMs) undergo continuous training or additional finetuning. By contrast, most research into LLMs&#39; internal mechanisms focuses on models at one snapsh...</li><li><a href="https://arxiv.org/abs/2407.09298">Transformer Layers as Painters</a>: Despite their nearly universal adoption for large language models, the internal workings of transformers are not well understood. We aim to better understand the impact of removing or reorganizing inf...</li><li><a href="https://arxiv.org/abs/2402.13388">Transformer tricks: Precomputing the first layer</a>: This micro-paper describes a trick to speed up inference of transformers with RoPE (such as LLaMA, Mistral, PaLM, and Gemma). For these models, a large portion of the first transformer layer can be pr...</li><li><a href="https://arxiv.org/abs/2403.15796">Understanding Emergent Abilities of Language Models from the Loss Perspective</a>: Recent studies have put into question the belief that emergent abilities in language models are exclusive to large models. This skepticism arises from two observations: 1) smaller models can also exhi...</li><li><a href="https://arxiv.org/abs/2310.18780">Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions</a>: Recent advances in attention-free sequence models rely on convolutions as alternatives to the attention operator at the core of Transformers. In particular, long convolution sequence models have achie...</li><li><a href="https://www.youtube.com/watch?v=s8RqGlU5HEs">2 Years of My Research Explained in 13 Minutes</a>: This is my research into representation learning and model learning in the reinforcement learning setting. Two years in the making, and I finally get to talk...</li><li><a href="https://arxiv.org/abs/2111.12763">Sparse is Enough in Scaling Transformers</a>: Large Transformer models yield impressive results on many tasks, but are expensive to train, or even fine-tune, and so slow at decoding that their use and study becomes out of reach. We address this p...</li><li><a href="https://arxiv.org/abs/2407.09375">HiPPO-Prophecy: State-Space Models can Provably Learn Dynamical Systems in Context</a>: This work explores the in-context learning capabilities of State Space Models (SSMs) and presents, to the best of our knowledge, the first theoretical explanation of a possible underlying mechanism. W...</li><li><a href="https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb">Protein language models expose viral mimicry and immune escape</a>: Viruses elude the immune system through molecular mimicry, adopting their hosts biophysical characteristics. We adapt protein language models (PLMs) to differenti-ate between human and viral...</li><li><a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper and Ada GPUs, to provide better performance with lower memory utilizatio...</li><li><a href="https://www.strchr.com/standard_deviation_in_one_pass">Calculating standard deviation in one pass - strchr.com</a>: no description found</li><li><a href="https://github.com/HanGuo97/flute">GitHub - HanGuo97/flute: Fast Matrix Multiplications for Lookup Table-Quantized LLMs</a>: Fast Matrix Multiplications for Lookup Table-Quantized LLMs - HanGuo97/flute</li><li><a href="https://arxiv.org/abs/2407.10960">Fast Matrix Multiplications for Lookup Table-Quantized LLMs</a>: The deployment of large language models (LLMs) is often constrained by memory bandwidth, where the primary bottleneck is the cost of transferring model parameters from the GPU&#39;s global memory to i...</li><li><a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py#L141>,">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper and Ada GPUs, to provide better performance with lower memory utilizatio...</li><li><a href="https://arxiv.org/abs/2407.09577">Flash normalization: fast RMSNorm for LLMs</a>: RMSNorm is used by many LLMs such as Llama, Mistral, and OpenELM. This paper details FlashNorm, which is an exact but faster implementation of RMSNorm followed by linear layers. See https://huggingfac...</li><li><a href="https://arxiv.org/abs/2404.12362">Transformer tricks: Removing weights for skipless transformers</a>: He and Hofmann (arXiv:2311.01906) detailed a skipless transformer without the V and P (post-attention projection) linear layers, which reduces the total number of weights. However, this scheme is only...</li><li><a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>: We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until af...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1262708582235766835)** (5 messages): 

> - `Human vs Animal Intelligence`
> - `Neural Activity and Growth`
> - `Gender Differences in Neuron Counts` 


- **Human Creativity Outshines Animal Intelligence**: A member discussed the comparison between **humans and animals** in terms of intelligence, noting that while humans are comparable to other animals in sensory capabilities, they excel in long-term planning and creativity.
   - *Humans possess an extremely large neocortex* and more folding, differentiating them from other mammals whose limbic systems and motor and sensory functions are quite similar.
- **Neural Activity Promotes Neuron Growth**: Increased **neural activity** spurs additional neuron growth, according to a member's comment, implying that extensive cognition and reasoning can significantly differentiate one animal from another.
   - The comment suggested that a **double PhD academic type** would have higher neural density compared to a less cognitively stimulated individual.
- **Cultural Factors and Gender Differences in Neurons**: A member mentioned that cultural variability might explain the apparent differences in neuron counts between males and females.
   - The member indicated that studies on **neuron counts** may not account for relative lifetime cognitive stimulation and life circumstances.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1262599690789326888)** (1 messages): 

> - `Mirror Neurons`
> - `Feature Representation`
> - `Circuit Reuse`
> - `Neurological Theories` 


- **Mirror neurons represent features in superposition**: A discussion arose about the possibility that **mirror neurons** are simply representing features in correlated superposition.
   - This could imply that neural circuits can be reused for different functions, a fascinating twist on current neurological theories.
- **Further implications on circuit reuse**: *Mirror neurons* reusing circuits in superposition could reshape understandings of brain efficiency.
   - Such theories could potentially lead to new perspectives in both **AI** and neuroscience research.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1262805109700558848)** (7 messages): 

> - `Tokenization in MLX & HF models`
> - `Chat template application in tokenization`
> - `Top-level options in lm-eval` 


- **Tokenization consistency in MLX and HF models**: A member noticed differences in tokenization between the MLX and HF models, particularly focusing on the handling of the **BOS token** and the lack of other prompt formatting without using `apply_chat_templates`.
   - *Gemma* model’s underperformance without the **BOS token** was highlighted as a specific concern.
- **Chat template option in tokenization**: It was mentioned that using the `--apply_chat_template` option wraps the examples in a chat template before tokenization.
   - There was some confusion about whether this option is specific to **HF model args** or a top-level **lm-eval** option.
- **Top-level lm-eval options and their impact**: The discussion clarified that `--apply_chat_template` is indeed a top-level **lm-eval** option.
   - The proper method for ensuring **MLX models** receive the same parameters as **HF models** was also sought.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1262586004649279489)** (1 messages): 

> - `Dynamic Evaluation on LLMs`
> - `EleutherAI cool results`
> - `Continual Learning`
> - `Meta-Learning` 


- **Query on EleutherAI and Dynamic Evaluation**: A member asked about whether EleutherAI has done any work with dynamic evaluation on LLMs, mentioning a supposed cool result that was never published.
   - The closest reference found was a [Gwern post on dynamic evaluation](https://gwern.net/doc/ai/nn/dynamic-evaluation/index#hardt-sun-2023-section), but it wasn't affiliated with EleutherAI.
- **Dynamic Evaluation Resources Shared**: A member shared a link to a [Gwern post on dynamic evaluation](https://gwern.net/doc/ai/nn/dynamic-evaluation/index#hardt-sun-2023-section), discussing related topics like compressed Transformers and meta-learning.



**Link mentioned**: <a href="https://gwern.net/doc/ai/nn/dynamic-evaluation/index#hardt-sun-2023-section">‘dynamic evaluation (NN)’ tag · Gwern.net</a>: no description found

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1262484568812945480)** (77 messages🔥🔥): 

> - `Anthropic's PR Strategy`
> - `Token Limits Impact`
> - `Evaluation Gating`
> - `Claude Engineer 2.0`
> - `Qwen2 Technical Report` 


- **Anthropic's daily PR strategy**: Anthropic is mastering the 'drip stuff out every day' PR strategy, announcing they've doubled the max output token limit for Claude 3.5 Sonnet from 4096 to 8192 tokens in the Anthropic API.
   - A community member expressed relief and shared how they had been burned by the previous limit recently. [Related Tweet](https://x.com/alexalbert__/status/1812921642143900036).
- **YAML vs JSON for structured data**: A user advocated for YAML over JSON for generating structured data in prompts, observing a 20-30% reduction in token counts.
   - Skepticism was expressed about JSON's dominance, with references to [a related Arxiv paper](https://arxiv.org/abs/2401.08500).
- **SciCode: A new benchmark for LLMs**: SciCode is a new benchmark challenging LLMs to code solutions for scientific problems from advanced papers, reportedly achieving less than 5% accuracy with GPT-4 and Sonnet 3.5.
   - Approximately 10% of the benchmark is based on Nobel-winning research. [More details here](https://scicode-bench.github.io/).
- **Eureka Labs by Andrej Karpathy**: Andrej Karpathy announced his new AI+Education company, Eureka Labs, aiming to build an AI-native school starting with an AI course LLM101n.
   - The content will be free, with revenue from running digital/physical cohorts through the materials. [Announcement Tweet](https://x.com/karpathy/status/1813263734707790301).
- **Qwen2 technical report release**: Qwen2 releases a comprehensive suite of language models, surpassing Qwen1.5 and competing with proprietary models across diverse benchmarks.
   - The suite includes models from 0.5 to 72 billion parameters and offers both dense and MoE models. [Read the report](https://hf.co/papers/2407.10671).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: As a tribute to Cleopatra, whose glorious destiny ended in tragic snake circumstances, we are proud to release Codestral Mamba, a Mamba2 language model specialised in code generation, available under ...</li><li><a href="https://x.com/karpathy/status/1813273726441652683">Tweet from Andrej Karpathy (@karpathy)</a>: Good question I do want Eureka Labs to be a proper, self-sustaining business but I also really don&#39;t want to gatekeep educational content. My default thinking is that the content itself is free an...</li><li><a href="https://youtu.be/OZmakgRZYxU?si=ThCmiCCp49V7Rq4n">The Downfall of AI Unicorns: Graphcore exits to Softbank</a>: Mergers, acquisitions, exists. It&#39;s the goal of any startup. But for the industry&#39;s first AI unicorn, is this the result we expected? After lackluster sales ...</li><li><a href="https://x.com/MinyangTian1/status/1813182904593199553">Tweet from Minyang Tian (@MinyangTian1)</a>: SciCode is our new benchmark that challenges LMs to code solutions for scientific problems from advanced papers. The challenges were crafted by PhDs;   ~10% of our benchmark is based on Nobel-winning ...</li><li><a href="https://x.com/skirano/status/1812943785237639218?s=46">Tweet from Pietro Schirano (@skirano)</a>: Introducing Claude Engineer 2.0, with agents! 🚀  Biggest update yet with the addition of a code editor and code execution agents, and dynamic editing.  When editing files (especially large ones), Eng...</li><li><a href="https://x.com/goodside/status/1812977352085020680">Tweet from Riley Goodside (@goodside)</a>: 9.11 is bigger than 9.9.</li><li><a href="https://x.com/OfirPress/status/1813202497864937825">Tweet from Ofir Press (@OfirPress)</a>: SciCode is our new benchmark, with 338 programming challenges written by PhDs in physics, math, and bio, based on papers in their fields. A bunch of the questions are from Nobel-winning papers!  ⁠I ho...</li><li><a href="https://x.com/xenovacom/status/1813258097185448377">Tweet from Xenova (@xenovacom)</a>: Introducing SmolLM: a new SOTA series of 135M, 360M and 1.7B models, perfect for on-device deployment! 🔥  We also uploaded ONNX weights for the models, meaning they can run locally in your browser wi...</li><li><a href="https://arxiv.org/abs/2401.08500">Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering</a>: Code generation problems differ from common natural language problems - they require matching the exact syntax of the target language, identifying happy paths and edge cases, paying attention to numer...</li><li><a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>: no description found</li><li><a href="https://x.com/alexalbert__/status/1812921642143900036">Tweet from Alex Albert (@alexalbert__)</a>: Good news for @AnthropicAI devs:  We&#39;ve doubled the max output token limit for Claude 3.5 Sonnet from 4096 to 8192 in the Anthropic API.  Just add the header &#34;anthropic-beta&#34;: &#34;max-tok...</li><li><a href="https://x.com/YiTayML/status/1813262126162845772">Tweet from Yi Tay (@YiTayML)</a>: Decided to start a new blog series about model architectures in the era of LLMs. 😀  Here&#39;s part 1 on broader architectures like Transformer Encoders/Encoder-Decoders, PrefixLM and denoising objec...</li><li><a href="https://x.com/karpathy/status/1813263734707790301">Tweet from Andrej Karpathy (@karpathy)</a>: ⚡️ Excited to share that I am starting an AI+Education company called Eureka Labs.  The announcement:  --- We are Eureka Labs and we are building a new kind of school that is AI native.  How can we ap...</li><li><a href="https://x.com/huybery/status/1813046544683442456">Tweet from Binyuan Hui (@huybery)</a>: 🔥 Qwen2 technical report. 📒 https://hf.co/papers/2407.10671  We release a comprehensive suite of foundational and instruction-tuned language models, encompassing a parameter range from 0.5 to 72 bil...</li><li><a href="https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising">What happened to BERT &amp; T5? On Transformer Encoders, PrefixLM and Denoising Objectives &mdash; Yi Tay</a>: A Blogpost series about Model Architectures Part 1: What happened to BERT and T5? Thoughts on Transformer Encoders, PrefixLM and Denoising objectives</li><li><a href="https://github.com/AnswerDotAI/bert24">GitHub - AnswerDotAI/bert24</a>: Contribute to AnswerDotAI/bert24 development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d4p1t6/comment/l6g1b3t/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://analyticsindiamag.com/ai-news-updates/hugging-face-announces-profitability-with-free-and-open-source-models/">
    Hugging Face Announces Profitability with Free and Open-Source Models</a>: no description found</li><li><a href="https://x.com/elder_plinius/status/1813181896789987411?s=46">Tweet from Pliny the Prompter 🐉 (@elder_plinius)</a>: gg  Quoting George McGowan (@GjMcGowan)   This website is offering to let me negotiate with an AI to buy a mattress
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1262687240929153104)** (1 messages): 

> - `XState`
> - `LangGraph`
> - `LLM Agents` 


- **XState-Powered LLM Agents WIP**: A member shared their work-in-progress approach using **XState** to create state-machine-powered LLM agents, available on [GitHub](https://github.com/statelyai/agent).
   - They plan to add more examples comparing **LangGraph** and **XState**.
- **Comparison of LangGraph and XState**: The same member hinted at an upcoming comparison between **LangGraph** and **XState** for building LLM agents.
   - This comparison aims to showcase the differences and advantages of each approach, providing more examples in the process.



**Link mentioned**: <a href="https://github.com/statelyai/agent">GitHub - statelyai/agent: Create state-machine-powered LLM agents using XState</a>: Create state-machine-powered LLM agents using XState - statelyai/agent

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1262486334434709524)** (27 messages🔥): 

> - `LM Studio Android app access`
> - `Graphical bug in LM Studio`
> - `Cloud-based LM Studio`
> - `Error with llama.cpp in LM Studio`
> - `H2O.ai Danube3 Model Issue` 


- **LM Studio Android app access on home network**: A user is inquiring about anyone using an Android app to access their LM Studio server on their home network.
   - It was suggested to use a VPN like Wireguard or Tailscale to securely access the server remotely.
- **Graphical bug in LM Studio**: A member noted a graphical bug where `f32` is recognized as 0 size because it's in a folder.
   - Despite the bug being cosmetic, it can affect user experience significantly.
- **Llama.cpp error with model architecture 'gemma2'**: **Anusha** received an error 'unknown model architecture: gemma2' while trying to load a model in `llama.cpp`.
   - It was recommended to post the issue in a specific channel with details on the LM Studio version and the precise model.
- **Flash Attention causing model load issue**: A user reported issues running an F16 GGUF model on their RTX 3090, believing the hardware to be insufficient.
   - The issue was found to be caused by Flash Attention and resolved upon disabling it.
- **Intel GPUs not recommended for LM Studio**: A query about LM Studio's compatibility with Intel A750 8G led to a recommendation against using Intel GPUs.
   - These GPUs are slow for AI tasks and unable to run the latest models due to deprecated OpenCL support.



**Link mentioned**: <a href="https://huggingface.co/h2oai/h2o-danube3-4b-chat-GGUF">h2oai/h2o-danube3-4b-chat-GGUF · Hugging Face</a>: no description found

  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1262483946554265734)** (28 messages🔥): 

> - `Hermes 2`
> - `Mistral issues`
> - `Model Merging`
> - `Open Empathic` 


- **Differences Between L3-12B-Lunaris-v1-GGUF and L3-12B-Lunaris-v1-i1-GGUF**: A member asked about the difference between **L3-12B-Lunaris-v1-GGUF** and **L3-12B-Lunaris-v1-i1-GGUF**, which was clarified as the regular quant vs. the imatrix quant.
   - *LLMs are free to download and test out so try em out,* was suggested without any specific performance difference noted.
- **Local Coding Models for 128GB RAM Systems**: A member sought recommendations for the best local coding model for a system with **128GB RAM**, focusing on TypeScript proficiency.
   - **Deepseek V2** (non-lite version) was suggested due to its **21B experts and 236B total params**, with a caution about potential OOM issues.
- **Mamba-Codestral-7B-v0.1 Discussion**: A link to the [Mamba-Codestral-7B-v0.1 model card](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) was shared, promoting the model's linear time inference and code productivity capabilities.
   - Another member noted that the model is yet to be supported in `llama.cpp`, with a PR in progress.
- **Issues with Vision Models in LM Studio**: Members discussed challenges in using vision models in **LM Studio**, noting the need to install both the model file and the `mmproj` file.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: As a tribute to Cleopatra, whose glorious destiny ended in tragic snake circumstances, we are proud to release Codestral Mamba, a Mamba2 language model specialised in code generation, available under ...</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1262731181313626122)** (18 messages🔥): 

> - `LMS Model Loading Speed Issue`
> - `Gemma 2 Support`
> - `Phi 3 Small Support`
> - `Llama.cpp Limitations` 


- **LMS Model Loading Speed Issue Identified**: **Users report** that LMS takes significantly longer to load models than it did in earlier versions, sometimes taking several minutes.
   - *Once a model is ejected and re-loaded, the load time drops to 2-4 seconds*, aligning with expected performance.
- **Gemma 2 to be supported, but not Phi 3 small**: A member announced **support for Gemma 2**, but not for **Phi 3 small**.
   - *Llama.cpp* is the limiting factor, as it does not support **Phi 3 small**.


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/)** (1 messages): 

magiikorb: and M3 ultra isn't even there
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1262858589660516404)** (1 messages): 

> - `Mistrals Mathstral Release`
> - `Community Models Program`
> - `Mathstral Performance`
> - `GGUF Quantization`
> - `LM Studio Discord Engagement` 


- **Mistrals Mathstral release promises STEM excellence**: **Mistral AI** has released **Mathstral**, a fine-tuned model specializing in STEM and advanced reasoning, outperforming the base **Mistral 7B** in major STEM categories.
   - Check out the [model on Hugging Face](https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF) to explore its capabilities and join the discussion on [Discord](https://discord.gg/aPQfnNkxGC).
- **Community Models Highlights Program showcases Mathstral**: The LM Studio highlights program is featuring **Mathstral**, a community-contributed model, encouraging discussion and exploration on [Discord](https://discord.gg/aPQfnNkxGC).
   - *Model creator:* [MistralAI](https://huggingface.co/mistralai), *Original model:* [mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1), with **GGUF quantization** provided by [bartowski](https://huggingface.co/bartowski).



**Link mentioned**: <a href="https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF">lmstudio-community/mathstral-7B-v0.1-GGUF · Hugging Face</a>: no description found

  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1262534504028045432)** (5 messages): 

> - `Evol-Instruct V2`
> - `Auto Evol-Instruct`
> - `Q-Sparse`
> - `BitNet b1.58` 


- **Evol-Instruct V2 Launched by WizardLM**: WizardLM announced [Evol-Instruct V2](https://x.com/WizardLM_AI/status/1812844503251947978), with a fully automated pipeline extending WizardLM-2 from three evolved domains (chat, code, and math) to dozens.
   - The team hopes this technology can promote fairness and efficiency for AI researchers in training and evaluation of large language models.
- **Auto Evol-Instruct Outperforms Experts**: Auto Evol-Instruct can evolve instruction data automatically, with experiments showing it outperforms human-designed methods in tuning various capabilities.
   - Auto Evol-Instruct achieved improvements of **10.44%** on MT-bench for instruction following, **12%** on HumanEval for code generation, and **6.9%** on GSM8k for mathematical reasoning.
- **Q-Sparse Speeds Up LLM Computation**: [Q-Sparse](https://x.com/realHongyu_Wang/status/1813112679734911169) introduced by Hongyu Wang claims to significantly speed up LLM computation by shifting the focus from memory to compute-bound processes.
   - This comes four months after the release of BitNet b1.58, which compressed LLMs to **1.58 bits**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/WizardLM_AI/status/1812844503251947978">Tweet from WizardLM (@WizardLM_AI)</a>: 🎉Today we are announcing Evol-Instruct V2 !!!  🔥 Auto Evol-Instruct is one of the most important technologies for WizardLM-2.  Paper link: https://arxiv.org/pdf/2406.00770  We build a fully automate...</li><li><a href="https://x.com/realHongyu_Wang/status/1813112679734911169">Tweet from Hongyu Wang (@realHongyu_Wang)</a>: 4 months since we released BitNet b1.58🔥🔥  After we compressed LLM to 1.58 bits, the inference of 1bit LLM is no longer memory-bound, but compute-bound.  🚀🚀Today we introduce Q-Sparse that can sig...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1262546544041787412)** (3 messages): 

> - `How AI Really Works`
> - `SpreadsheetLLM`
> - `Synth Data` 


- **Understanding AI through Interactive Visualization**: A [YouTube video](https://youtu.be/pj8CtzHHq-k) titled "How AI Really Works" explains the workings of large language models (LLMs) like **Llama 3** through an interactive visualization.
   - The video emphasizes why open source matters in the development and understanding of these models.
- **SpreadsheetLLM to Revolutionize Data Management**: **Microsoft** released a new large language model, **SpreadsheetLLM**, designed for advanced spreadsheet tasks, hinting at transformative applications in data management and analysis.
   - A [pre-print paper](https://arxiv.org/abs/2407.09025) was quietly released, sparking discussions about the job market impact, with some suggesting "Karen might be out of a job soon."
- **Community Buzz on Synth Data**: A member shared a [Twitter link](https://twitter.com/pratyushmaini/status/1752337225097076809) discussing synthetic data applications and developments in the AI community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/pj8CtzHHq-k">How AI Really Works (And Why Open Source Matters)</a>: A short talk using an interactive visualization to explain how current AI really works, specifically how large language models (LLMs), like Llama 3, are just...</li><li><a href="https://arxiv.org/abs/2407.09025">SpreadsheetLLM: Encoding Spreadsheets for Large Language Models</a>: Spreadsheets, with their extensive two-dimensional grids, various layouts, and diverse formatting options, present notable challenges for large language models (LLMs). In response, we introduce Spread...</li><li><a href="https://www.thestack.technology/microsoft-llm-spreadsheet-llm/">Microsoft unveils a large language model that excels at encoding spreadsheets</a>: New LLM has the &quot;potential to transform data management and analysis, paving the way for more intelligent and efficient user interactions.&quot;
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1262489087500226622)** (58 messages🔥🔥): 

> - `Heat Wave Discussions`
> - `White Roof Paint Innovation`
> - `Deepseek Coder Comparison`
> - `Hackathon on FP8`
> - `Urban Heat Island Effect` 


- **Members Debate Impact of Heat Wave**: A member mentioned how the current heat wave is 'crazy,' and another noted friends in Vegas are 'literally cooking even with AC maxxed out.'
   - The conversation suggested painting roofs white as a cooling measure, with a [Yale article](https://e360.yale.edu/features/urban-heat-can-white-roofs-help-cool-the-worlds-warming-cities) discussing the benefits of white roofs in mitigating urban heat islands.
- **Super White Paint's Cooling Potential**: A member shared that a super white paint invented in 2021 reflects 98% of sunlight.
   - Another member linked a [YouTube video](https://youtu.be/KDRnEm-B3AI) demonstrating the creation of this paint using household items and its effectiveness in reflecting infrared and cooling.
- **Hackathon Event for FP8 at Crusoe Office**: A member announced a hackathon on FP8 at Crusoe's SF office, focusing on improving inference, fine-tuning, and pretraining.
   - Attendees will use L40S nodes for hacking and hear from speakers on FP8-related topics. [Event details here](https://lu.ma/hpb5svgw).
- **Deepseek Coder 16B vs Mistral Code Models**: Members discussed the efficacy of Deepseek Coder models, emphasizing the importance of debugging over zero-shot performance.
   - A member testing Deepseek Coder v2 16B highlighted its speed (>60t/s) and unquantized matrices, while another questioned its debugging capabilities.
- **Mistral and Deepseek Code Models Under Scrutiny**: Mistral AI's new models with 256k context weren't well received by some, who preferred Deepseek Coder's capabilities.
   - The discussion included critical comparisons, noting issues with Mistral's MBPP performance and the lack of self-debugging features in some models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/HCSolakoglu/status/1812984327510085883">Tweet from Hasan Can (@HCSolakoglu)</a>: Llama3 405b has been added to @OpenRouterAI , there is no provider yet. Huggingface model weights page was also attached. Release must be close.</li><li><a href="https://youtu.be/KDRnEm-B3AI?si=UOzyzARqomlGZ1mS">Making Infrared Cooling Paint From Grocery Store Items (w/Novel CaCO₃ Microsphere Synthesis)</a>: Check out my sponsor Brilliant, free for 30 days by using this link: https://brilliant.org/nighthawkIn this video we explore new methods of making cutting ed...</li><li><a href="https://x.com/MistralAI/status/1813222156265791531">Tweet from Mistral AI (@MistralAI)</a>: https://mistral.ai/news/mathstral/ https://mistral.ai/news/codestral-mamba/</li><li><a href="https://lu.ma/hpb5svgw">FP8 Island Vibes Hackathon · Luma</a>: Join us on July 20th for a day focused on hacking with FP8 to improve inference, fine-tuning and pretraining on Brev notebooks backed by Crusoe’s L40S nodes –…</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/pull/32">Add example of outlines + llama-cpp-python by alonsosilvaallende · Pull Request #32 · NousResearch/Hermes-Function-Calling</a>: Example of using outlines + llama-cpp-python with Hermes 2 Pro Llama 3 8B GGUF model by NousResearch. The example shows:  how to generate synthetic data by following a Pydantic schema how to answer...</li><li><a href="https://e360.yale.edu/features/urban-heat-can-white-roofs-help-cool-the-worlds-warming-cities">Urban&#x20;Heat&#x3A;&#x20;Can&#x20;White&#x20;Roofs&#x20;Help&#x20;Cool&#x20;World&#x27;s&#x20;Warming&#x20;Cities&#x3F;</a>: It&#x20;has&#x20;long&#x20;been&#x20;known&#x20;that&#x20;installing&#x20;white&#x20;roofs&#x20;helps&#x20;reduce&#x20;heat&#x20;buildup&#x20;in&#x20;cities.&#x20;But&#x20;new&#x20;research&#x20;indic...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1262743547321384992)** (6 messages): 

> - `Tokenization Issues with Arabic Symbols`
> - `Tools for Generating PPO/DPO Datasets`
> - `Invertibility of Tokenization` 


- **Tokenization of Arabic Symbols Fails to Decode Correctly**: A member noted issues with the **tiktoken library** when decoding Arabic symbols, encountering special tokens instead of the original string.
   - They explained how using `errors='replace'` in the decoding process substitutes invalid byte sequences with the special symbol `�`, questioning how **LLMs** can accurately generate text from such tokens.
- **No Known Tools for Generating PPO/DPO Datasets**: A member inquired about tools for generating **PPO/DPO datasets** from raw data, but another member responded that they were not aware of any such tools.
- **Invertibility of Tokenization May Vary**: *One member* stated they experienced different results regarding invertibility of tokenization, suggesting that it may decode to either a **UTF-8 sequence** or a **0x80-0xFF** byte, specifically with the `cl100k_base` tokenizer.


  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 messages): 

wolfybl: Hi
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1262526297805160529)** (37 messages🔥): 

> - `GPTs Agents`
> - `Sora release speculation`
> - `GPT mini in Lymsys`
> - `OpenAI Platform`
> - `AI Programming` 


- **Sora release speculation continues**: Members discussed the release date of **Sora**, with some speculating it might come out in late 2024, Q4 based on blog posts, though **OpenAI** has not made any official statement.
   - One member noted that accurate prediction dates are usually those close to the actual release and should be cautious with random Reddit or Twitter posts.
- **GPT mini in Lymsys**: Speculation arose about the rumour of an upcoming **GPT mini** in Lymsys, with one member simply remarking that it's 'cool.'
   - No substantial information was provided, and some members hinted that random predictions are mostly baseless.
- **Voice mode vs Sora demos**: Members noted that **Sora** demos have been released instead of the anticipated **Voice mode**, sparking further speculation.
   - One member humorously suggested that the developers might have deleted all the codes for Voice mode by mistake, requiring another significant investment and development time.
- **AI programming advantages**: A member emphasized that using **AI** is much easier for conducting experiments, particularly for **game machine learning**, where you can save parameters' weights and biases.
   - Another member noted that **ChatGPT** is better at programming compared to other models like Claude 3.5 Sonnet, describing it as not miles better, but notably better.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1262487577873616896)** (11 messages🔥): 

> - `Using GPT-4 to code a mobile game`
> - `Learning coding with GPT-4`
> - `Challenges using GPT for development`
> - `Creating a customer support chatbot`
> - `Adjusting GPT's response tone` 


- **Using GPT-4 to code a mobile game**: Members discussed how GPT-4 can assist in coding a mobile game, mentioning that it can help write tests and code but often produces buggy **output** that needs double-checking.
   - One member recommended tools like **React Native** and **FastAPI** and shared experiences of GPT-4 being useful but needing verification for errors.
- **Learning coding with GPT-4 from scratch**: A member shared their success in creating a full web app with **zero prior coding knowledge** using GPT-4, emphasizing that clear explanations can help GPT assist effectively.
   - It's noted that while GPT-4 can provide useful responses, users must ensure the code works correctly and be self-directed in their learning process.
- **Fine-tuning for Bengali and Banglish chatbot**: A student worked on a customer support chatbot in **Bengali and Banglish**, asking if fine-tuning a model with 100 conversations would help it learn conversation patterns.
   - Another member explained that the model won't adapt permanently but can catch patterns within the context length, and the patterns may be lost if they exceed the context window.
- **GPT-4's formal response tone issue**: One member experienced a shift in GPT-4's **response tone** from laid-back to very formal, asking if there are ways to make the bot sound less 'nerdy'.
   - This change reportedly persisted regardless of prompt modifications, leading to decreased interest in the project.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1262585044061655131)** (5 messages): 

> - `Different languages affecting model performance`
> - `Prompting in native language vs English`
> - `Model's handling of regional slang and idioms` 


- **Model performance varies by language training**: A member noted that model performance can vary by language, likely depending on the amount of training data it received for that language.
   - The member mentioned that prompting in English or other languages like Chinese could lead to responses in those languages due to examples in system prompts.
- **Prompting in native language vs English for better results**: A member questioned whether to prompt in French or in English and ask for a French answer to achieve better results.
   - Another member suggested that models like GPT-4o are not good with regional slang, idioms, and colloquialisms, and recommended avoiding heavy use of those devices for better results.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1262585044061655131)** (5 messages): 

> - `Language model performance across different languages`
> - `Prompting in different languages`
> - `Language preferences for model responses`
> - `Regional slang, idioms, and colloquialisms in GPT models` 


- **Model performance varies by language exposure**: A member speculated that **language models perform better** in languages they were more extensively trained on, noting that prompt language choices can affect response quality.
   - They shared that **English prompts** are more likely to get responses in **English**, similar to other languages like **Chinese**.
- **Prompting in French for better responses**: A user inquired if prompting in **French** would yield better responses in French compared to asking in English and then translating.
- **GPT struggles with regional slang and idioms**: A member highlighted that **GPT-4** and lower versions struggle with understanding **regional slang, idioms, and colloquialisms**. They advised avoiding heavy use of these devices for better results.


  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1262796132061548585)** (1 messages): 

> - `LlamaIndex Webinar`
> - `RAG Improvement`
> - `Deasie Automated Labeling`
> - `LlamaParse Tool` 


- **LlamaIndex Webinar on Advanced Parsing and Metadata**: A new [LlamaIndex Webinar](https://lu.ma/ufx77ve8) will be hosted this Thursday at 9am PT, featuring Deasie's cofounders discussing **improving RAG** with advanced parsing and metadata.
- **Experimental Results For Parsing + Metadata**: The workshop will demonstrate results from research papers showing the combination of **parsing** and **metadata** for enhanced performance.
- **Deasie's Role in Enhancing RAG**: Deasie's labeling workflow enhances RAG by auto-generating hierarchical metadata labels to improve LLM retrieval over 10,000+ documents. More details on their [website](https://deasie.com/).
- **Automate Cataloging Unstructured Data with Deasie**: Deasie automates tagging and cataloging large volumes of unstructured data for enterprise knowledge management and compliance.
- **LlamaParse Github Repository**: LlamaParse, a tool for optimal RAG file parsing, can be accessed on its [GitHub page](https://github.com/run-llama/llama_parse).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/ufx77ve8">LlamaIndex Webinar: Improving RAG with Advanced Parsing + Metadata Extraction · Zoom · Luma</a>: We&#x27;re excited to cohost a workshop with the cofounders of Deasie (Reece, Leonard, Mikko) on improving RAG with advanced parsing and metadata. The data…</li><li><a href="https://deasie.com/">Deasie | Data Governance for Language Model Applications</a>: The Deasie platform ensures only safe, high-quality and relevant data is fed into language models. Developed by award-winning team in enterprise software for AI and data governance.</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1262523257265983652)** (4 messages): 

> - `Document RAG`
> - `Graph Query Algorithm`
> - `LlamaIndex Webinar`
> - `Sonnet-3.5 Chart Understanding` 


- **Multimodal RAG Promises Future of Document Processing**: In a new [cookbook](https://t.co/KWsVGwT3jD), a multimodal RAG architecture using LlamaParse and GPT-4o is highlighted for processing slide decks rich in text, diagrams, charts, and tables.
   - *At the core is a hybrid text/image* approach, enhancing the capabilities of document RAG.
- **Build Your Own Graph Query Algorithm**: With [LlamaIndex](https://t.co/atFLrXbYtQ) and Mistral, you can create custom graph query algorithms, blending text-to-cypher or vector search techniques.
   - You have the flexibility to define your own query algorithm provided you have access to the necessary resources.
- **Upcoming LlamaIndex Webinar on RAG Enhancement**: A new [webinar](https://t.co/7pgCBKx1IL) co-hosted with Deasie's cofounders will focus on improving RAG using advanced parsing and metadata.
   - The importance of getting the data processing layer right for RAG is emphasized, making it crucial for effective AI implementation.
- **Sonnet-3.5 Excels in Chart Understanding**: Sonnet-3.5 is noted for its superior performance in chart understanding compared to GPT-4o, particularly in inferring chart values into structured tables.
   - The new [LlamaParse release](https://t.co/Dq3xaVfXio) enables easy integration of state-of-the-art multimodal models for enhanced data interpretation.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1262488561517592628)** (47 messages🔥): 

> - `LLM Response Sources`
> - `Service Context and Indexing Models`
> - `Vector Datasets and Tools`
> - `Parallel Index Loading`
> - `PropertyGraphIndex Embeddings` 


- **Retrieving LLM Response Sources in Queries**: To get the source of the LLM response while querying over text files, use the `get_formatted_sources()` method on the response object, or `display_eval_sources()` function for evaluation results ([more details](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/vector_stores/SimpleIndexDemo.ipynb)).
   - These methods return the sources of the response in a formatted manner, useful for debugging and understanding data provenance.
- **Clarification on Service Context and Embedding Models**: Recent versions of LlamaIndex no longer require the `serviceContext`; you can set LLM/embedding models globally or pass them directly into relevant modules ([LLM customization docs](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-changing-the-underlying-llm)).
   - For example, you can pass `gpt-4o` into the query engine directly, simplifying custom setups as opposed to using default embeddings.
- **Seeking Public Vector Datasets for Easy Access**: Members are looking for services that host public vector datasets like Wikipedia to avoid the infrastructure overhead of hosting their own.
   - One suggestion was to host one’s own Wikipedia vector store, though the preference remains for pre-hosted, query-ready services.
- **Optimizing Index Loading for Large Datasets**: A member reported that loading large indexes takes a considerable amount of time and inquired about methods to speed this up, such as parallel processing.
   - The discussion suggests using `QueryPipelines` or other methods to optimize and potentially parallelize the loading process.
- **Embedding Data in PropertyGraphIndex**: The `PropertyGraphIndex.from_documents()` method is where embeddings are created for storage in Neo4J nodes ([source code](https://github.com/run-llama/llama_index/blob/f092d90bd5934097f5c166014f5d99a3e07ea999/llama-index-core/llama_index/core/indices/property_graph/base.py#L248)).
   - This section of the source code details the embedding creation process to store data effectively in the graph database.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/78754465/how-to-merge-multiple-at-least-two-existing-llamaindex-vectorstoreindex-instan">How to merge multiple (at least two) existing LlamaIndex VectorStoreIndex instances?</a>: I&#x27;m working with LlamaIndex and have created two separate VectorStoreIndex instances, each from different documents. Now, I want to merge these two indexes into a single index. Here&#x27;s my cur...</li><li><a href="https://stackoverflow.com/questions/78754465/how-to-merge-multiple-at-least-two-existing-llamaindex-">How to merge multiple (at least two) existing LlamaIndex VectorStoreIndex instances?</a>: I&#x27;m working with LlamaIndex and have created two separate VectorStoreIndex instances, each from different documents. Now, I want to merge these two indexes into a single index. Here&#x27;s my cur...</li><li><a href="https://github.com/run-llama/llama_index/blob/f092d90bd5934097f5c166014f5d99a3e07ea999/llama-index-core/llama_index/core/indices/property_graph/base.py#L248">llama_index/llama-index-core/llama_index/core/indices/property_graph/base.py at f092d90bd5934097f5c166014f5d99a3e07ea999 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://cloud.llamaindex.ai/.">LlamaCloud</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-changing-the-underlying-llm">Customizing LLMs - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1262814317300944999)** (3 messages): 

> - `llamaindex property graph vs microsoft graphrag`
> - `Graph rag functionalities`
> - `Property graph features` 


- **LlamaIndex vs Microsoft GraphRAG**: A user asked about the difference between **llamaindex property graph** and **Microsoft GraphRAG**.
   - *Cheesyfishes* explained that GraphRAG summarizes and clusters entities into 'communities' for retrieval, while property graph supports text-to-cypher, embedding, keyword, and other custom retrieval methods.
- **GraphRAG and Property Graph Functionalities**: [GraphRAG](#) does not do much with the graph itself, only summarizing and clustering entities.
   - In contrast, **property graph** allows implementing custom features like text-to-cypher, embeddings, and custom retrieval methods.


  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1262487715459633266)** (44 messages🔥): 

> - `Cohere Python Library`
> - `Cohere Discord Bot`
> - `Spam Awareness`
> - `Fireside Chats with Max Welling`
> - `Job Postings and Engagement` 


- **Contribute to Cohere Python Library**: A member was encouraged to check out the [Cohere Python Library](https://github.com/cohere-ai/cohere-python) for open source contributions.
   - Another member mentioned they are a heavy user of the library and might start contributing soon.
- **Cohere Discord Bot Category Issue**: A member is troubleshooting why their Discord bot is categorizing everything under 'opensource'.
   - Another member offered to take a look at the bot's issue, although no promises were made.
- **Spam Awareness Initiative**: A member suggested creating awareness posts about suspicious links to enhance user safety.
   - Another member agreed, proposing that it could be part of the server onboarding process.
- **Fireside Chat with Max Welling**: [C4AI](https://discord.gg/Jf6FPp3c?event=1262761447486787685) is hosting a session with Max Welling from the University of Amsterdam.
   - An apology was issued after an inappropriate use of @everyone notification for the event.
- **Job Postings and Engagement on Discord**: A member was reminded that job postings aren't permitted on the server.
   - It was suggested that engagements for work should be handled privately, emphasizing that experts command higher rates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cohere-ai/cohere-python">GitHub - cohere-ai/cohere-python: Python Library for Accessing the Cohere API</a>: Python Library for Accessing the Cohere API. Contribute to cohere-ai/cohere-python development by creating an account on GitHub.</li><li><a href="https://docs.cohere.com/page/cookbooks#rag">Cookbooks</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1262738929174577213)** (1 messages): 

> - `Automatic post categorization`
> - `Channel specific categorization`
> - `Prompt adjustments` 


- **Automatic Categorization Misroutes Posts**: Currently, a project is underway to automatically categorize posts into specific channels. However, all posts are being misrouted to the open-source channel, indicating a potential issue with the prompting.
- **Issue Likely Cause: Prompting Needs Adjustment**: The misrouting issue seems to be due to the prompts used, which hopefully are an easy fix. The posts from r/openai, despite having a dedicated channel, are routed incorrectly to the open-source channel.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1262617389783191613)** (9 messages🔥): 

> - `finetuning pipeline`
> - `Bengali chatbot`
> - `timestamps in MessageGraph`
> - `community help` 


- **Request for finetuning pipeline details**: A member asked another member to share their finetuning pipeline, suggesting that it might help find a solution to their problem.
- **Bengali customer support chatbot fine-tuning**: A student working on a Bengali and Banglish customer support chatbot asked if training a LLM like **Llama 3** or **GPT 3.5** with 100 real-life Bengali chat conversations would help the model learn the conversation pattern and give accurate responses.
   - A member advised looking into prompts that elicit specific behaviors, sharing an anecdote about a model understanding Hindi/Urdu slang. *Just as a joke I asked to speak in Hindi/Urdu slang and it was able to do it pretty well.*
- **Timestamps in MessageGraph**: A member inquired if all messages in a **MessageGraph** can have timestamps or if they need to create a **StateGraph** with custom state.
- **Community help channel feedback**: A member expressed frustration over not receiving answers after asking a question in the help channel.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1262527098652856401)** (3 messages): 

> - `Automatic 1111 SD with 1.5`
> - `Browser RAG using LangChain & WebLLM`
> - `Launch of Verbis`
> - `Open-Source GenAI Models` 


- **Launch of Verbis App**: Verbis, an open-source MacOS app, has been launched with a focus on local data indexing and leveraging **GenAI** models for enhanced productivity without compromising privacy.
   - Key features include **local data processing**, **open-source** development, and no data sent to third parties. [Check it out on GitHub](https://github.com/verbis-ai/verbis).
- **100% Browser RAG Using LangChain & WebLLM**: A member shared a [YouTube video](https://youtu.be/MHuvSuK2dnY) demonstrating the use of **Visual Agents** to deploy a WebLLM chat model in the browser for question answering.
   - The video showcases a seamless process of dropping the model onto a canvas and instantly interacting with it.



**Link mentioned**: <a href="https://youtu.be/MHuvSuK2dnY">Use 100% Browser Only WebLLM to Answer Questions!</a>: In this video, I use Visual Agents to drop a WebLLM chat model onto my canvas and instantly start asking it questions.

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1262668115788169318)** (2 messages): 

> - `PyTorch tunner`
> - `Training instruction models`
> - `Context length adjustment`
> - `Mistral's chat template issues` 


- **Questions about PyTorch tunner release**: A member inquired about the availability of a **PyTorch tunner**.
- **Few users train on instruction models**: A member observed that not many users are training on **instruction models**, suggesting users can swap models as they please.
- **Context length adjustment tips**: A user advised others to increase context length as needed but noted that higher numbers take more VRAM.
- **Mistral's unusual chat template causes issues**: A member pointed out that **Mistral's** strange chat template compared to other models often causes issues.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1262535440330653728)** (4 messages): 

> - `Pull Request Created`
> - `Discussion on DPO`
> - `Integrating Work` 


- **Pull Request Completed Successfully**: A member announced the creation of a new [pull request](https://github.com/axolotl-ai-cloud/axolotl/pull/1756) successfully.
   - *"There we go, pull request created.*"
- **DPO Integration Challenges**: A discussion revealed that the **DPO** (Direct Policy Optimization) approach is simpler but doesn't support integrating tokenization or masking.
   - One member concluded that this extension is more suited for **SFT (Supervised Fine-Tuning)** rather than DPO variants.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1262750790167166988)** (5 messages): 

> - `lora_target_linear`
> - `LoRA configuration`
> - `Axolotl fine-tuning` 


- **Understanding lora_target_linear**: A member asked, *'What is lora_target_linear?'*, and received a detailed explanation highlighting that it is a configuration option in Axolotl for specifying whether LoRA should be applied to linear layers within the model.
   - The explanation included that when set to **true**, LoRA adapters modify linear layers to enable efficient fine-tuning without training the entire model from scratch.
- **Effects of setting lora_target_linear to false**: A member inquired about the implications of setting `lora_target_linear` to false, seeking further clarification.
   - *No detailed answer provided in the given messages.*



**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f6280a5d-7259-4317-b9ca-adbe6c9066c3)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1262863256217850008)** (1 messages): 

> - `Torchtune v0.2.0 release`
> - `New models and recipes`
> - `Dataset improvements`
> - `Community contributions` 


- **Torchtune v0.2.0 Launch**: Announcing the release of [torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0) with contributions from the community over the past few months.
- **Exciting New Models and Recipes**: The **v0.2.0** release brings new models 🦙 and recipes to the main stable package along with dataset improvements such as sample packing 🚀.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/releases/tag/v0.2.0">Release v0.2.0 · pytorch/torchtune</a>: Overview It’s been awhile since we’ve done a release and we have a ton of cool, new features in the torchtune library including distributed QLoRA support, new models, sample packing, and more! Chec...

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1262614219086495834)** (5 messages): 

> - `Eval loss calculation`
> - `Checkpoint optimization`
> - `Recipe modification`
> - `Data split and evaluation` 


- **Calculate Eval Loss Without Backpropagation**: A member inquired about ways to calculate loss on the evaluation dataset without Backpropagation, aiming to plot loss curves for training and eval datasets to decide the best checkpoints.
   - Another member suggested modifying the [default recipe config file](https://github.com/pytorch/torchtune/issues/1066) to include a test split dataset and an eval loop, emphasizing the use of `torch.no_grad()` and eval mode for the model.
- **Optimizing Checkpoint Selection**: The original query focused on finding the best checkpoint without overfitting by calculating eval loss at each step.
   - The suggested workaround included evaluating the dataset for each epoch until a more detailed per-step evaluation setup could be implemented.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1066.">Issues · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/loading#slice-splits">Load</a>: no description found
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1262827174596186286)** (1 messages): 

> - `Scaling RoPE Embeddings`
> - `Long Context Modeling` 


- **RFC for scaling RoPE embeddings for long contexts**: The [RFC](https://github.com/pytorch/torchtune/issues/1183) discusses adding RoPE scaling methods to support long context modeling for tasks like large document understanding or code completion.
   - *In order for this to be enabled by default, a model would need to support context lengths greater than 8K*.
- **Long context modeling for large documents**: For large document understanding or tasks like code completion, it is often beneficial to have a large context length, e.g., greater than 8K.
   - The RFC suggests methods to scale up RoPE embeddings to accommodate such requirements.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/issues/1183">[RFC] Adding RoPE scaling methods to support long context modeling · Issue #1183 · pytorch/torchtune</a>: Background For large document understanding or tasks like code completion, it&#39;s often beneficial to have a large context length e.g. &gt; 8K. In order for this to be enabled by default, a model wo...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1262800347433537646)** (3 messages): 

> - `ComfyUI malicious node attack`
> - `Disney attacks`
> - `FBI involvement in Disney attacks` 


- **ComfyUI Attackers Target Disney**: A member reported that the group behind the **ComfyUI malicious node attack** was also responsible for the recent **Disney attacks**.
   - *The group is probably just screwing with random people*, according to another member.
- **Hope for FBI Involvement**: Despite personal dislike for Disney, a member expressed hope that the **FBI** would investigate the **Disney attacks**.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

nodja: https://mistral.ai/news/codestral-mamba/
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 messages): 

__ctrlaltdel__: https://youtu.be/pj8CtzHHq-k
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/)** (1 messages): 

jbexta: I'll try to get a demo/tutorial out this week 👍
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1262486895892627569)** (4 messages): 

> - `Open Interpreter usage with RayBan Stories`
> - `Rooting RayBan Stories glasses`
> - `Opinion on hacking via app`
> - `Google Glass alternative`
> - `O1 Light and hardware preorder updates` 


- **Challenges in integrating Open Interpreter with RayBan Stories**: A member shared their interest and struggles in using Open Interpreter with **RayBan Stories**, highlighting the lack of an SDK from Meta and difficulty accessing the device’s internals.
   - They provided specs [from a Pastebin link](https://pastebin.com/wTsRMH3f) and discussed potential barriers like glued components, and a desire for transparent models for easier exploration.
- **Consideration of Google Glass as an alternative**: A member suggested using **Google Glass** as an alternative, posing it as a solution given the difficulties with RayBan Stories.
   - *No additional details or discussions were provided on this suggestion.*
- **Frustrations over O1 Light hardware preorder delays**: Multiple members expressed frustration over delays in receiving their **O1 Light hardware**, with orders placed over 3 months ago.
   - The lack of updates on the **preorder situation** has led to growing impatience among the community members.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/RayBanStories/comments/rlzyot/rayban_stories_codenamed_stella_runs_android_810/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://pastebin.com/z709s9Ru">## ADDITIONAL_DEFAULT_PROPERTIES#ro.oem_unlock_supported=1ro.usb.id.adb= - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1262787952032092196)** (3 messages): 

> - `GPT-4o fine-tuning access`
> - `OpenPipeAI support` 


- **GPT-4o fine-tuning access confusion**: A user asked if anyone has access to [fine-tuning GPT-4o](https://openai.com/gpt-4o-and-gpt-4-fine-tuning-experimental-access/).
   - Another user clarified that access on the OpenAI side is required, referencing a statement by Kyle.
- **OpenPipeAI now supports GPT-4o training**: [Corbtt announced](https://x.com/corbtt/status/1813018434822971556?t=qCi3vH2LH1KSho8x658urA&s=19) that **OpenPipeAI** supports training GPT-4o, emphasizing it should be used responsibly.
   - This was suggested as an option for using course credits efficiently.



**Link mentioned**: <a href="https://x.com/corbtt/status/1813018434822971556?t=qCi3vH2LH1KSho8x658urA&s=19">Tweet from Kyle Corbitt (@corbtt)</a>: If you ever felt the need for an Extremely Overpowered fine-tuned model... we now support training GPT-4o in @OpenPipeAI. Please use responsibly. 😎

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1262531632221655071)** (2 messages): 

> - `Intermediate language of tinygrad`
> - `Debugging and visualization in tinygrad` 


- **Exploring Tinygrad's Intermediate Language**: A user inquired about how the intermediate language of tinygrad looks and asked about the storage of deep learning operators in the IR.
   - Another user recommended asking such questions in a specific channel and provided a tip to run tinygrad with **DEBUG=3** to display the bottom level of IR, and using **GRAPH=1** and **GRAPHUOPS=1** commands for visualizations.
- **Debugging and visualizing Tinygrad**: A user suggested running tinygrad with **DEBUG=3** to show the bottom level of IR for understanding the intermediate language.
   - Additionally, they mentioned using **GRAPH=1** and **GRAPHUOPS=1** to generate visual representations of the internal workings.


  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1262818557729832961)** (1 messages): 

> - `Open Interpreter`
> - `Mike Bird presentation` 


- **Mike Bird presents Open Interpreter**: Mike Bird is now on stage discussing **Open Interpreter**. Join the conversation to ask questions about the project using [this event link](https://discord.gg/rXdZzd5wu3?event=1260611047341953034).
- **Questions Encouraged During Mike Bird's Talk**: Participants are encouraged to jump in with questions during Mike Bird's presentation on **Open Interpreter**.


  

---



---



---



---



---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
