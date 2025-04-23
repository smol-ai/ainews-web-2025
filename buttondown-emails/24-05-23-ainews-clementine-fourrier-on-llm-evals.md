---
id: 8f586835-8d9c-4a69-8163-a84dbaf7ee47
title: '[AINews] Clémentine Fourrier on LLM evals'
date: '2024-05-23T23:34:22.485002Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-to-be-named-4285
---

<!-- buttondown-editor-mode: plaintext -->**Leaderboards are all you need.**

> AI News for 5/22/2024-5/23/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**380** channels, and **5410** messages) for you. 
Estimated reading time saved (at 200wpm): **551 minutes**.

---

> Special addendum for the [AI Engineer World's Fair](https://buttondown.email/ainews/archive/ainews-the-top-ai-engineer/) callout yesterday - [Scholarships are available for those who cannot afford full tickets!](https://docs.google.com/forms/d/e/1FAIpQLSdVf9reEpVyzw_sb9cAOtzxORGTEskcb5PTX2X-GPZ_onUtHw/viewform). More speaker announcements are [rolling out](https://x.com/aidotengineer).

Many people know [Huggingface's Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), but you rarely hear from the people behind it. [Clémentine Fourrier made a rare appearance at ICLR](https://x.com/clefourrier/status/1793301496102068339?utm_source=ainews&utm_medium=email) (to co-present [GAIA with Meta](https://arxiv.org/abs/2311.12983), something we cover on the upcoming ICLR pod) and is now back with [a blog covering how she thinks about LLM Evals](https://huggingface.co/blog/clefourrier/llm-evaluation).

 ![image.png](https://assets.buttondown.email/images/d49e1192-98ed-40ae-8ad7-1771acd9c816.png?w=960&fit=max) 

This is not going to be groundbreaking for those very close to the problem, but is a good and accessible "state of the art" summary from one of the most credible people in the field.

Our TL;DR: There are 3 main ways to do evals:

- **Automated Benchmarking**
  - Evals are made of **a collection of sample input/outputs** (usually comparing generated text with a reference or multiple choice) and a **metric** (to compute a score for a model)
  - for a specific **task**
    - Works well for very well-defined tasks
    - Common problems: models [favoring specific choices based on the order in which they have been presented for multi-choice evaluations](https://arxiv.org/abs/2309.03882), and generative evaluations relying on normalisations which can easily be [unfair if not designed well](https://huggingface.co/blog/open-llm-leaderboard-drop) 
  - or for a general **capability**
    - e.g. GSM8K high school problems as a proxy for "good at math", [unicorns](https://twitter.com/DimitrisPapail/status/1719119242186871275) for "can draw"
  - LLMs scores on automated benchmarks are extremely susceptible to [minute changes in prompting](https://huggingface.co/blog/evaluation-structured-outputs)
  - Biggest problem: data contamination. BigBench tried adding a "canary string" but compliance/awareness is poor. [Tools exist to detect contamination](https://arxiv.org/abs/2311.06233) and peopel are exploring [dynamic benchmarks](https://arxiv.org/abs/2104.14337) though this is costly.
- **Humans as Judges**
  - Done by tasking humans with 1) prompting models and 2) **grading a model answer or ranking several outputs according to guidelines**. 
  - more flexibility than automated metrics. 
  - prevents most contamination cases, 
  - correlates well with human preference
  - Either as **Vibe-checks**
    - mostly constitute anecdotal evidence, and tend to be highly sensitive to confirmation bias
    - but some people like [Ravenwolf are very systematic](https://huggingface.co/blog/wolfram/llm-comparison-test-llama-3)
  - Or as **Arena** (e.g. [LMsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard))
    - Votes are then aggregated in an Elo ranking (a ranking of matches) to select which model is "the best". 
    - high subjectivity: hard to enforce a consistent grading from many community members using broad guidelines
  - Or as **systematic annotations**
    - provide extremely specific guidelines to paid selected annotators, in order to remove as much as the subjectivity bias as possible (Scale AI and other annotation companies)
    - still expensive
    - can still fall prey to human bias
- **Models as Judges**
  - using generalist, high capability models
  - OR using small specialist models trained specifically to discriminate from preference data
  - Limitations: 
    - tend to [favor their own outputs](https://arxiv.org/abs/2404.13076) when scoring answers
    - bad at [providing consistent score ranges](https://twitter.com/aparnadhinak/status/1748368364395721128)
    - not that [consistent with human rankings](https://arxiv.org/pdf/2308.15812)
  - Introduce very subtle and un-interpretable bias in the answer selection

Evals are used to prevent regressions, and to rank models, and to serve as a proxy for progress in the field.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!


{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**NVIDIA Earnings and Stock Performance**

- **Strong earnings**: [@nearcyan](https://twitter.com/nearcyan/status/1793379327188562214) noted NVIDIA beat earnings six quarters in a row, with **revenue up 262% last year to $26B and margins at 75.5%**. They also did a 10:1 stock split.
- **Investor reaction**: [@nearcyan](https://twitter.com/nearcyan/status/1793377805704843431) shared an article about NVIDIA's earnings, with investors happy about the results. The **stock is up over 260% in the past year**.
- **Market cap growth**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793583558633611402) highlighted NVIDIA's success, with their **market cap increasing over 6x to $2.3T, surpassing Google and Amazon**. Revenue was up 262% and diluted EPS was up over 600%.

**Mistral AI Model Updates**

- **Faster LoRA finetuning**: [@danielhanchen](https://twitter.com/danielhanchen/status/1793356226006511902) released a free Colab notebook for **Mistral v3 with 2x faster LoRA finetuning using Unsloth AI**. It uses 70% less VRAM with no accuracy loss.
- **Mistral-7B v0.3 updates**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793373838904025304) noted Mistral-7B v0.3 was released with **extended vocab to 32768, v3 tokenizer support, and function calling**. 8x7B and 8x22B versions are coming soon.
- **Mistral v0.3 on 🤗 MLX**: [@awnihannun](https://twitter.com/awnihannun/status/1793392487941505348) shared that **Mistral v0.3 base models are available in the 🤗 MLX community, generating 512 tokens at 107 tok/sec with 4-bit quantization on an M2 Ultra**.

**Meta's Llama and Commitment to Open Source**

- **Calls for open-sourcing Llama**: [@bindureddy](https://twitter.com/bindureddy/status/1793464074455666960) said Meta open-sourcing Llama-3 400B would make them the biggest hero and is the most important thing right now.
- **Open source foundation**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1793636929017459055) reminded that **open-source is the foundation of all AI, including closed-source systems**. 
- **Meta's open source leadership**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793294819986436434) highlighted Meta's leadership in open source beyond just Llama-3, with projects like **React, PyTorch, GraphQL, Cassandra and more**.

**Anthropic's Constitutional AI**

- **Claude's writing abilities**: [@labenz](https://twitter.com/labenz/status/1793384693057917342) shared a clip of @alexalbert__ from Anthropic explaining that **Claude is the best LLM writer because they "put the model in the oven and wait to see what pops out"** rather than explicitly training it.
- **Claude Character work**: [@labenz](https://twitter.com/labenz/status/1793663525954650478) is excited to read more about the "Claude Character" work @AmandaAskell is leading at Anthropic on **building an AI assistant with stable traits and behaviors**.
- **Anthropic's honest approach**: [@alexalbert__](https://twitter.com/alexalbert__/status/1793683229595341182) explained that **Anthropic is honest with Claude about what they know and don't know regarding its ability to speculate on tricky philosophical questions**, rather than purposely choosing to allow or prevent it.

**Google's AI Announcements and Issues**

- **LearnLM for personalized tutoring**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1793605797114782049) announced new **"LearnLM" models to enable personalized AI tutors on any topic** to make learning more engaging.
- **Inconsistencies in AI overviews**: [@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1793311536095879225) noted **Google's new LLM-powered AI overviews seem to have some inconsistencies**, like saying President Andrew Johnson was assassinated.
- **Website poisoning attack**: [@mark_riedl](https://twitter.com/mark_riedl/status/1793375699967054334) successfully used a **website poisoning attack on Google's LLM overviews** by modifying information on his own site.
- **Changing meaning of "Googling"**: [@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1793320073320615988) expressed concern that **Google's AI summaries are changing the meaning of "Googling" something** from retrieving high-quality information to potentially unreliable AI-generated content.

**Open Source Debates and Developments**

- **Open source as a strategy**: [@saranormous](https://twitter.com/saranormous/status/1793363171241009414) strongly disagreed with the idea that **open source is just a charity**, arguing it's a strategy for building and selling, citing Linux's success and large community of contributors.
- **Open source success stories**: [@saranormous](https://twitter.com/saranormous/status/1793363188324401367) pushed back on claims open source can't compete with big tech AI labs, noting **Android's massive mobile ecosystem as an example of open source success**.
- **Open source as AI's foundation**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1793636929017459055) stated open source is the foundation of all AI, including closed source systems from major labs.
- **Openness for US leadership**: [@saranormous](https://twitter.com/saranormous/status/1793363184247533603) argued restricting open source AI won't stop determined adversaries, only slow US innovation and cede leadership to others. She believes **openness keeps the US on the offensive and is key to shaping AI with western values**.

**AI Safety and Regulation Discussions**

- **California AI bill criticism**: [@bindureddy](https://twitter.com/bindureddy/status/1793412487813226675) criticized the **new California AI bill** for effectively banning open source AI by placing compute thresholds and restrictions on models.
- **AI safety job market predictions**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1793380346270306319) predicted the **percentage of top math/CS grads aspiring to work in AI safety will fall, not rise**, as new regulations mean few jobs in core AI development but plenty in the "safety apparatus."
- **DARPA funding for AI safety**: [@ylecun](https://twitter.com/ylecun/status/1793319668456755309) suggested perhaps **AI safety research could be funded under a DARPA program** for building better, safer AI systems.
- **Key points of California AI bill**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793237818996556225) summarized key points of **California's newly passed AI bill**, including capability shutdown requirements, annual certification, and restrictions on models trained with over 10^26 FLOPs.

**Emerging AI Architectures and Techniques**

- **Similar concept learning across modalities**: [@DrJimFan](https://twitter.com/DrJimFan/status/1793318771932995793) shared an MIT study showing **LLMs and vision models learn similar concept representations** across modalities, without explicit co-training. He wants to see this extended to 3D shapes, speech, sound, and touch.
- **PaliGemma in KerasNLP**: [@fchollet](https://twitter.com/fchollet/status/1793349537702334940) announced **PaliGemma vision-language model is now in KerasNLP** with JAX, TF and PyTorch support for image captioning, object detection, segmentation, VQA and more.
- **Linearity of transformers**: [@_arohan_](https://twitter.com/_arohan_/status/1793346994775400860) joked "we don't need skip connections or normalization layers either" in response to a paper showing the linearity of transformers.
- **Unnecessary transformer components**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1793190738190090491) summarized recent papers suggesting **many transformer components like attention, KV cache, FFN layers, and reward models may be unnecessary**.

**AI Benchmarking and Evaluation**

- **Prompt Engineering Guide milestone**: [@omarsar0](https://twitter.com/omarsar0/status/1793659569635311961) announced the **Prompt Engineering Guide has reached 4M visitors** and continues to add new advanced techniques like LLM agents and RAG.
- **LLM evaluation blog post**: [@clefourrier](https://twitter.com/clefourrier/status/1793301496102068339) published a blog post on **how LLM evaluation is currently done and what it's useful for**, after realizing it's not widely understood based on discussions at ICLR.
- **Saturated benchmarks**: [@hwchung27](https://twitter.com/hwchung27/status/1793511637678444954) noted **saturated benchmarks can give a false impression of slowing progress** and become useless or misleading proxies for what we care about.
- **Fine-tuning and hallucinations**: [@omarsar0](https://twitter.com/omarsar0/status/1793292346978623812) shared a paper suggesting **fine-tuning LLMs on new knowledge encourages hallucinations**, as unknown examples are learned more slowly but linearly increase hallucination tendency.

**Emerging Applications and Frameworks**

- **No-code model fine-tuning**: [@svpino](https://twitter.com/svpino/status/1793272058417152483) demonstrated **no-code fine-tuning and deployment of open source models using an AI assistant**, powered by GPT-4 and Monster API's platform.
- **RAG-powered job search assistant**: [@llama_index](https://twitter.com/llama_index/status/1793434183353958465) shared an **end-to-end tutorial on building a RAG-powered job search assistant** with Koyeb, MongoDB, and LlamaIndex with a web UI.
- **Generative UI templates in LangChain**: [@LangChainAI](https://twitter.com/LangChainAI/status/1793681539659903084) added **templates and docs for generative UI applications using LangChain JS/TS and Next.js** with streaming agent calls and tool integrations.
- **AI-powered reporting tool**: [@metal__ai](https://twitter.com/metal__ai/status/1793660651186819221) highlighted their **AI-powered reporting tool** for running complex multi-step operations on company data to streamline information requests, ESG diligence, call summary insights and more.

**Compute Trends and Developments**

- **M3 MacBook Pro matrix multiplication**: [@svpino](https://twitter.com/svpino/status/1793389861120115085) tested **matrix multiplication on an M3 MacBook Pro**, seeing 3.72ms on GPU vs 14.4ms on CPU using PyTorch. Similar results with TensorFlow and JAX.
- **Copilot+ PC demo**: [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1793324989422391649) demo'd a **Copilot+ PC (Surface Laptop) with CPU, GPU and 45+ TOPS NPU** delivering unrivaled performance.

**AI-Generated Voices and Identities**

- **Spite and revenge in East Asian cultures**: [@TheScarlett1](https://twitter.com/TheScarlett1/status/1793444365693591803) and [@teortaxesTex](https://twitter.com/teortaxesTex/status/1793444785514074291) expressed concern about the prominence of **spite and desire for revenge in East Asian cultures**, with people willing to burn their lives to get back at offenders.
- **OpenAI non-disparagement clause**: [@soumithchintala](https://twitter.com/soumithchintala/status/1793467399150436563) noted an **impressive follow-up from @KelseyTuoc with receipts that OpenAI proactively pressured employees to sign a non-disparagement clause** with threats of exclusion from liquidity events.
- **Scarlett Johansson/OpenAI controversy**: [@soumithchintala](https://twitter.com/soumithchintala/status/1793685296405524654) argued the **Scarlett Johansson/OpenAI controversy makes the AI attribution conversation tangible to a broad audience**. Cultural norms are still being established before laws can be written.

**Miscellaneous AI News and Discussions**

- **Auto-regressive LLMs insufficient for AGI**: [@ylecun](https://twitter.com/ylecun/status/1793680385403957295) shared a Financial Times article where he explains **auto-regressive LLMs are insufficient for human-level intelligence**, but alternative "objective driven" architectures with world models may get there.
- **Selling to capital allocators vs developers**: [@jxnlco](https://twitter.com/jxnlco/status/1793633344866980136) advised founders to **focus on selling to rich capital allocators, not developers**, in order to fund their AI roadmap and get to a mass-market product later.
- **Getting enough data for AGI**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1793353131637760217) discussed on a podcast how **we can get enough data to reach AGI**, but it will be more like curing cancer incrementally than a single vaccine-like discovery.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Model Releases and Updates**

- **GPT-4o performance**: In /r/LocalLLaMA, GPT-4o is reported to be [**6 times faster and 12 times cheaper than the base model, with a 120K context window**](https://i.redd.it/1k9y5w6pfz1d1.png).
- **Mistral-7B v0.3 updates**: /r/LocalLLaMA announces that [**Mistral-7B v0.3 has been released with extended vocabulary, v3 tokenizer support, and function calling**](https://www.reddit.com/r/LocalLLaMA/comments/1cy61iw/mistral7b_v03_has_been_released/). Mixtral v0.3 was also released.
- **Microsoft Phi-3 models**: Microsoft released Phi-3-Small (7B) and Phi-3-Medium (14B) models following Phi-3-Mini, according to a post in /r/LocalLLaMA. [Comparisons were made to Llama 3 70B and 8B models](https://www.reddit.com/r/LocalLLaMA/comments/1cxvh3i/so_how_is_phi3small_and_phi3medium/).
- **"Abliterated-v3" models**: New "abliterated-v3" models were released on Hugging Face, including Phi-3-medium-4k-instruct, Smaug-Llama-3-70B, Llama-3-70B-Instruct, and Llama-3-8B-Instruct. [**They have an inhibited ability to refuse requests and reduced hallucinations compared to previous versions**](https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b).

**AI Capabilities and Limitations**

- **Understanding LLMs through sparse autoencoders**: /r/singularity discusses how [Anthropic is making progress in understanding LLMs through sparse autoencoders in Claude 3 Sonnet](https://www.reddit.com/r/singularity/comments/1cxutku/anthropic_make_great_progress_understanding_llms/). **Extracting interpretable, multilingual, multimodal features could help customize model outputs without extensive retraining**.
- **Concerns about AI agents**: In /r/MachineLearning, some argue that [**AI agents are overhyped and too early, with challenges around reliability, performance, costs, legal concerns, and user trust**](https://www.reddit.com/r/MachineLearning/comments/1cy1kn9/d_ai_agents_too_early_too_expensive_too_unreliable/). Narrowly scoped automations with human oversight are suggested as a path forward.

**AI Ethics and Safety**

- **OpenAI's tactics toward former employees**: [Vox reports on OpenAI documents revealing aggressive tactics toward former employees](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees), with concerns raised about short timelines to review complex termination documents.
- **OpenAI employee resignations over safety**: [Business Insider reports that another OpenAI employee quit over safety concerns](https://www.businessinsider.com/another-openai-employee-quits-over-safety-concerns-2024-5) after two high-profile resignations. Krueger said tech firms can "disempower those seeking to hold them accountable" by sowing division.
- **Containing powerful AI systems**: Former Google CEO Eric Schmidt says [**most powerful future AI systems will need to be contained in military bases due to dangerous capabilities**](https://x.com/tsarnick/status/1793391127028191704), raising concerns about an AI arms race and existential risk.

**AI Applications and Use Cases**

- **Microsoft's Copilot AI agents**: [The Verge reports on Microsoft's new Copilot AI agents that can act like virtual employees to automate tasks](https://www.theverge.com/2024/5/21/24158030/microsoft-copilot-ai-automation-agents), with potential for significant job displacement.
- **AI in chip design and software development**: [Nvidia CEO Jensen Huang says chip design and software development can no longer be done without AI](https://twitter.com/tsarnick/status/1793076745543073922), and wants to turn Nvidia into "one giant AI".
- **AI helping the blind**: /r/singularity shares a story of [**AI being used to help a blind 16-year-old through Meta AI glasses**](https://www.reddit.com/r/singularity/comments/1cy4g6x/a_reminder_of_why_we_are_all_here/), serving as a reminder of AI's potential to positively impact lives.

**Stable Diffusion and Image Generation**

- **Stable Diffusion in classic photography**: /r/StableDiffusion showcases [examples of using Stable Diffusion in a classic photography workflow](https://www.reddit.com/gallery/1cxwuld) for tasks beyond inpainting, such as model training, img2img, and enriching photos.
- **Generating images from product photos**: Punya.ai shares a blog post on how [generating images from product photos is getting easier with Stable Diffusion](https://punya.ai/blog/post/generative-ai-for-fashion-clothes), with tutorials and ready-to-use tools available.
- **Future of Stable Diffusion**: /r/StableDiffusion discusses [**questions around the future of Stable Diffusion after Emad Mostaque's departure from Stability AI**](https://www.reddit.com/r/StableDiffusion/comments/1cy16o2/what_is_the_current_state_and_future_of_sd/), with concerns about direction and progress.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Model Performance Optimization and New Releases**:

- **Gemini Tops Reward Bench Leaderboard**: **Gemini 1.5 Pro** achieved top rankings in the **Reward Bench Leaderboard** as noted by [Jeff Dean](https://huggingface.co/spaces/allenai/reward-bench), outperforming other generative models.

- **Mistral v0.3 Sparks Mixed Reactions**: The release of **Mistral v0.3** created buzz with its enhanced vocabulary and new features ([Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)), though users debated its performance improvements and integration complexities.

- **Tensorlake Open-Sources Indexify**: **Tensorlake** announced [Indexify](https://x.com/tensorlake/status/1793693325180150146), an open-source real-time data framework, stirring enthusiasm for its potential in AI stacks.

**2. Fine-Tuning Strategies and Challenges**:

- **Retaining Fine-Tuned Data Woes**: Users faced difficulties with **Llama3** models retaining fine-tuned data when converted to GGUF format, pointing to a [confirmed bug](https://github.com/ggerganov/llama.cpp/issues/7062) discussed in the community.

- **Axolotl's Configuration Struggles**: Persistent issues configuring **Axolotl** for dataset paths and loss scaling highlighted community recommendations for updates, including checking [Axolotl’s documentation](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html).

- **Cuda Errors and GPU Utilization**: Members reported **CUDA out of memory** errors on various GPUs, with suggestions of switching to **QLoRA** and using **Docker** images ([example](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018)) to mitigate issues.

**3. Open-Source AI Innovations and Collaborations**:

- **Open-Source Gains with AlphaFold Rivals**: **ProteinViz** was introduced as an open-source alternative to AlphaFold3, detailed in a community [blog post](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3).

- **StoryDiffusion Launches MIT-Licensed Alternative to Sora**: **StoryDiffusion** entered the open-source scene, though weights are yet to be released ([GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)).

**4. AI API Integrations and Community Efforts**:

- **Roleplay AI Models Launched**: The **Lumimaid 70B** model was released for roleplay applications, with details on the [OpenRouter announcement page](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b).

- **Batch Inference for GenAI**: **LlamaIndex** highlighted the efficiency of batch inference for pre-processing data in **GenAI applications**, providing integration insights [here](https://t.co/vnuvvypZCz).

- **Anthropic and Gemini Models via OpenRouter**: OpenRouter expanded its support to include **Anthropic** and **Gemini** models, detailed in their [documentation](https://openrouter.ai/docs#tool-calls) and recent announcements.

**5. GPU Optimization and Technical Workshops**:

- **GPU Optimization Workshop a Hit**: The GPU optimization workshop, featuring experts from **OpenAI**, **NVIDIA**, and **Meta**, saw over [2400+ registrants](https://lu.ma/1wu5ppl5), with resources available on [GitHub](https://github.com/mlops-discord/gpu-optimization-workshop).

- **Technical Fixes with Docker and CUDA**: Members discussed common **CUDA errors**, recommending Docker images and [Axolotl configurations](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018) for smooth GPU operations in AI workloads.

- **LLM Training Cost Reports and Benchmarks**: Detailed cost and efficiency reports on frontier model training costs were shared, with estimations of **$250k for the largest Pythia model**, underscoring the importance of optimizing GPU-hours used.

---

{% if medium == 'web' %}



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral v0.3 Causes Stir**: The announcement of **Mistral v0.3** generated excitement but also some confusion due to a mix-up with version naming. To improve GPU efficiency with Mistral models, suggestions included increasing batch sizes and updating training codes.
  
- **Unsloth Growth**: **Unsloth AI** has expanded its repertoire, now supporting new models like **Phi-3**, **Mistral v3**, and a range of **4-bit quantized models**. Experimentation with these models is facilitated by various [Colab notebooks](https://github.com/unslothai/unsloth/releases/tag/May-2024).

- **Technical Tweaks and Fixes**: Engineers are actively working on resolving issues, such as the "buggy" reserved tokens in **LLaMa 3** and discussing complexities in training certain layers of models like **Qwen**, with recommended workarounds involving biases and layer training adjustments.

- **Recognition and Resources**: **Unsloth AI** has been recognized as part of **GitHub’s 2024 Accelerator program**, joining other projects in driving innovation in open-source AI. To aid in deploying these advancements, free notebooks have been provided for ease of access.

- **Challenges in Language and Truthfulness**: The engineering discourse included tackling the challenges posed by fact-checking and language-specific fine-tuning in **LLMs**, referencing studies like [*scaling-monosemanticity*](https://arxiv.org/abs/2306.03341) and [*In-Context RALM*](https://arxiv.org/abs/2302.00083) to aid in these pursuits.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Scheduled Downtime for Database Boost**: A **scheduled downtime** has been announced, set to commence at 12:00am EST and last approximately 30 minutes to upgrade the database improving performance and user experience.

**Engineering Excitement Over Free Gemini**: Engineering conversations revolved around the free usage of **Gemini in AI Studio** for high-volume tasks like fine-tuning, spurring discussions on data privacy and cost-saving strategies.

**Perplexity Powers Past Performance Hurdles**: Notable improvements in **Perplexity's web scraping** have yielded speeds of 1.52s, significantly surpassing previous performances over 7s, while discussions highlighted the importance of parallel processing and efficient tooling in AI applications.

**Comparative AI Discourse**: Technically-inclined users compared **Perplexity** with **Gemini Pro** and **ChatGPT**, lauding Perplexity's research and writing capabilities and flexible file management, with suggestions to include additional features like CSV support to reach new heights of utility.

**API Anomalies and Alternatives Analysis**: Community members discussed discrepancies in outputs between web and API versions of the same models, seeking clarifications on the observed inconsistencies, while also sharing their experiences in balancing model accuracy and utilization within **API rate limits** for platforms like **Haiku**, **Cohere**, and **GPT-4-free**.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Instruction Finetuning with ColBERT and Task Updates**: Engineers discussed finetuning strategies for instruction embeddings, citing frameworks like **INSTRUCTOR** and **TART** as references. A project proposal for automating standup transcript ticket updates involved using examples of standup conversions correlated with ticket actions.

**CUDA Woes and Workarounds**: Persistent **CUDA errors** while running LLM models like llama 3 8b were a common issue, with remedies including adjusting batch sizes and monitoring GPU usage via `nvidia-smi`. Docker was recommended for managing CUDA library compatibility, with a link to a Docker image from Docker Hub provided.

**Parameters and Efficient Model Training**: Queries emerged about default **Axolotl's configuration** parameters and **optimization strategies** for training on **A100 and H100 GPUs**, where using bf16 and maximizing VRAM usage were among the suggested strategies. Discussions also extended to newer optimizers like **Sophia** and **Adam_LoMo**.

**Accelerating Free Credits and Workshop Excitement**: Modal's fast credit allocation was commended, and excitement built around a **GPU Optimization Workshop** featuring representatives from OpenAI, NVIDIA, Meta, and Voltron Data. Additionally, there was anticipation for a **recording** of an upcoming talk by Kyle Corbitt.

**Model Fine-Tuning and Training Factors**: Fine-tuning **LLMs to generate layouts**, troubleshooting **Axolotl's dataset paths**, and considering **LoRA hyperparameters** were topics of interest. The use of **GPT-4 as a judge for level 2 model evaluations** and troubleshooting **Axolotl on Modal** due to gated model access issues were also discussed.

**Deployment Dilemmas**: Engineers encountered challenges when deploying trained models to S3 on Modal, with solutions including using the `modal volume get` command and mounting an S3 bucket as a volume, as described in Modal's [documentation](https://modal.com/docs/guide/cloud-bucket-mounts).

**Paper and Tutorial References**: The community shared valuable learning resources, such as a [YouTube demo](https://www.youtube.com/watch?v=glwBlONacPY) on EDA assistant chatbots. They also appreciated illustrative examples from Hamel and Jeremy Howard, with references to both [a tweet](https://twitter.com/HamelHusain/status/1793319488731107718) and a [GitHub repo](https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AlphaFold Rivals and Advances**: A member introduced [ProteinViz](https://huggingface.co/spaces/as-cle-bert/proteinviz), an alternative to AlphaFold3, showcasing the tool for predicting protein structures, along with a [community blog post](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3) on AlphaFold3's progress.

- **Transparent Gains with LayerDiffusion**: [Diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse) allows for creating transparent images from any base model, raising the bar for foreground image separation accuracy.

- **Effective Minimal Training Data Use**: A discussion noted that training **Mistral** with as few as 80 messages to perceive itself as a 25-year-old was surprisingly effective, hinting at efficient fine-tuning strategies.

- **AI Enters Query Support Role**: Enthusiasm was shown for using AI to query lengthy software manuals, with members pondering the practicality of feeding a 1000-page document to an AI for user support.

- **Model Training Memory Management**: Utilizing `torch_dtype=torch.bfloat16`, one combated CUDA OOM errors during Mistral model SFT, reinforcing the instrumental role of tensor precision in managing extensive computational workloads on GPUs.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Flash Attention Needed for YaRN**: Efforts to implement **flash attention** into the **YaRN** model are meeting challenges, with some progress but not a perfect fit yet.

**Rust Rising Among AI Enthusiasts**: Increasing interest and discussions around using **Rust** for machine learning, with members sharing resources like [Rust-CUDA GitHub](https://github.com/Rust-GPU/Rust-CUDA) and [rustml - Rust](https://github.com/daniel-e/rustml), while recognizing the dominance of Python in AI.

**Nous Research Expanding Teams**: **Nous Research** is on the hunt for new talent, as evidenced by their recent **hiring announcement** and a call to apply via their [Google Form](https://forms.gle/UWx2Pht8qioi1bjAA).

**Python vs Rust in AI Careers**: A robust debate over Python's primacy in AI careers with members bringing up alternatives like Rust or Go, alongside sharing insights from AI experts like Yann LeCun's views on focusing beyond LLMs for next-gen AI systems.

**RAG's Validity in Question**: Proposals made to enhance RAG's model context, emphasizing the need for context accuracy by referencing a debate over the reliability of Google's AI drawing conclusions from outdated sources.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Emad's Mysterious Weights Countdown**: Speculation is high about **Stable Diffusion’s** forthcoming weight updates, with a user implying an important release could happen in two weeks, sharing excitement with a *Star Wars* analogy.

- **Clearer Visions Ahead for Stable Diffusion**: There's ongoing discussion regarding **Stable Diffusion 3** producing blurred images, particularly for female characters; modifying prompts by removing 'woman' seemed to offer a **clearer output**.

- **Laptop Muscle Matchup**: Rumbles in the tech space about **ASUS AI laptops** and **NVIDIA's rumoured 5090 GPU**, accompanied by a [PC Games Hardware article](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/), are drawing attention and debate among users, with a focus on specifications and performance authenticity.

- **AI Tool Smackdown**: A brief exchange compared **MidJourney** and **Stable Diffusion**, with one camp favoring MJ for quality, while suggesting hands-on experience with the latter might sway opinions.

- **Installation vs. Cloud**: The eternal debate on **local installation versus utilizing web services** for **Stable Diffusion's** usage continues, with a new angle brought to light concerning the performance with **AMD GPUs**, and a general guideline suggesting installation for those with robust graphics cards.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LLama Lamentations & Local Model Logistics**: There's unrest over **Llama 3's** 8k context performance, with members revealing it falls short of expectations. Despite being the topic of debate, suggestions for improving its performance, such as introducing longer contexts up to 1M, remain theoretical.

**Discussions Turn to Vision Models**: OCR discussions saw mixed reviews of vision models like **LLaVA 1.6** as users recommend **Tesseract** for reliable text extraction. Interest in **Vision Language Models (VLMs)** is evident, but deploying them effectively with web server APIs requires attentive configuration, including `apikey` incorporation.

**Multimodal Mishaps and Merits**: **Idefics 2.0 multimodal**’s compatibility sparked interest, yet it seems to trip on existing infrastructure like llama.cpp. Meanwhile, **Mistral-7B-Instruct v0.3** emerges as part of the dialogue, boasting extended vocabulary and improved functional calling ([Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)). In parallel, **Cohere's Aya 23** showcases its talents in 23 languages, promising to sway future conversations ([Aya 23 on Huggingface](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)).

**GPU Grows but Guides Needed**: The adoption of **7900xt** graphics cards is underway among members seeking to amp up their tech game. However, guidance for effective environment setups, such as treating an RX 6600 card as gfx1030 on Fedora, remains a precious commodity.

**Storage Solved, Support Summoned**: One member's move to allocate an M.2 SSD exclusively for **LM Studio** paints a picture of the ongoing hardware adaptations. On the flip side, GPU compatibility queries like dual graphics card support highlight the community's reliance on shared wisdom.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo on the Rise**: Users observed **compilation errors** in Mojo nightly `2024.5.2305` and shared solutions like explicit type casting to `Float64`. A debate over null-terminated strings in Mojo brought up performance concerns and spurred discussions on potential changes referencing GitHub issues and external resources such as the [PEP 686](https://peps.python.org/pep-0686/) on UTF-8 string handling. 

- **Syntax Shuffle**: The replacement of `inferred` keyword with `//` for inferred parameters in Mojo stirred mixed reactions and highlighted the trade-off between brevity and clarity. A proposal for `f-string`-like functionality encouraged the exploration of a `Formatable` trait, setting the stage for possible future contributions.

- **Decorators and Data Types Discussed**: In the **Mojo** channels, discourse ranged from using `@value` decorators for structs, seen as valuable for reducing boilerplate, to the feasibility of custom bit-size integers and **MLIR dialects** for optimizing memory use. The need for documentation improvement was highlighted by a query about FFT implementation in Mojo.

- **Structured Logging and GitHub Issue Management**: Participants suggested the creation of a dedicated channel for **GitHub issues** to improve tracking within the community. Additionally, the importance of proper syntax and notation in documentation became clear as users addressed confusion caused by the misuse of `**` in documentation, emphasizing the need for consistency.

- **Community and Updates**: **Modular** released a new video on a community meeting, with details found in their [public agenda](https://modul.ar/community-meeting-doc), and shared their weekly newsletter, [Modverse Weekly - Issue 35](https://www.modular.com/newsletters/modverse-weekly-35), keeping the community informed and engaged with the latest updates and events.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Pythia's Pocketbook**: Discussing the cost of training models like **Pythia**, Stellaathena estimated a bill of **$250k for the largest model**, mentioning efficiency and discounted GPU-hour pricing in calculations. 

**Cost-Efficiency Report Needs Reviewers**: A forthcoming report on *frontier model training costs* seeks peer review; interested parties would assess GPU-hours and the influence of GPU types like **A100 40GB**.

**LeanAttention Edging Out FlashAttention?**: A recently shared paper introduces **LeanAttention**, which might outperform **FlashAttention**, raising debates on its innovation. The community also joked about unorthodox practices to improve model benchmarks, playfully noting, "The secret ingredient is crime."

**Interpretability's New Frontiers**: A new paper was noted for opening research doors in **interpretability**, kindling curiosity on its implications for future studies.

**Evaluating Large Models**: Tech tips were exchanged, such as running the **lm eval harness** on multi-node SLURM clusters and how to set parameters like `num_fewshot` for evaluations with challenges reported around reproducibility and internet access on compute nodes.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Model Prefers YAML, Causing JSON Jealousy**: Engineers noted anecdotally that the AI model favors **YAML** handling over **JSON**, sparking both technical curiosity and humor among discussion participants regarding model preferences despite development efforts being skewed towards JSON.

- **GPT-4o and DALL-E 3 Create Artful Collabs**: Conversations revealed that **GPT-4o** is enhancing image prompts interpretation, creating better outputs when used with **DALL-E 3** compared to using DALLE-3 in isolation. This synergy illustrates the evolving interplay between text and image models.

- **Newlines in the Playground Cause Formatting Frustrations**: The OpenAI playground newline handling has been causing usability issues, with reports of inconsistent pasting results. This seemingly minor technical hiccup has sparked broader discussions on formatting and data presentation.

- **Anthropic Paper Ignites Ideas and Speculations**: The community discussed a paper from Anthropic on mech interpretation and its implications, touching on how AI might anthropomorphize based on training data, reflecting concepts like confinement and personas in unexpected ways. Technical debates ensued regarding the impact of such findings on future AI development.

- **Prompt Engineering Secrets and Critiques Shared**: Technical discussions included strategies for prompt engineering, with practical advice being exchanged on system prompts, which some found lacking. Issues such as models disappearing from sidebars and the semantics of "step-by-step" prompts were dissected, reflecting a deep dive into the minutiae of user experience and AI interactivity.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Full House at the GPU Optimization Workshop**: The GPU optimization workshop raked in excellent engagement with **over 2400+ registrants** and valuable sessions from experts including **Sharan Chetlur** (NVIDIA), **Phil Tillet** (OpenAI), and **William Malpica** (Voltron Data). Enthusiasts can RSVP for future interactions [here](https://lu.ma/1wu5ppl5), with additional resources available on [GitHub](https://github.com/mlops-discord/gpu-optimization-workshop).

**Breaching CUDA Confusion**: A member clarified that **`__global__` CUDA functions** can't be simultaneously `__host__` due to their grid launch setup, and they posited the theoretical utility of a `__global__` function agnostic of `threadIdx` and `blockIdx`.

**Tricky Transformation with Triton**: One user discussed performance drops when converting a kernel from **FP32 to FP6 using triton+compile**, speculating on the potential impact of inplace operators.

**AI Research Synopsis Spices Up Discussions**: A weekly AI research spotlight surfaced, featuring analysis on works like KAN, xLSTM, and OpenAI's GPT-4. The discussion extended to the computationally intensive nature of KANs owing to activation-based edge computation.

**The CUDA Cul-de-Sac and Vulkan Ventures**: Conversations veered into contributions and coding concerns, including a member's **flash-attention repository** stalling, GPU model benchmarks like 7900xtx versus 3090, and Vulkan's failure to impress in a heat transfer simulation.

**LLM.C Lurches Forward**: There was a bustling (busy) exchange about llm.c with members celebrating the integration of **HellaSwag evaluation in C**, debating **CUDA stream optimization** for speed, and sharing the challenge of scaling batch sizes without training disruptions.

Please note, some quotes and project links have been shared verbatim as no additional context was provided.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Quantization Quandaries with Llama 3**: Technophiles are discussing the challenging quantization of **Llama 3** models, noting performance drop due to the model's bit accuracy sensitivity.
- **Models in the Spotlight**: Some engineers are pivoting their attention back to **Mistral models** for fine-tuning issues, while the **Aya models**, especially the 35B version released on [Hugging Face](https://huggingface.co/CohereForAI/aya-23-35B), are stirring excitement due to their architecture and training prospects.
- **GPU Roadblocks**: AI mavens are finding **GPU memory limitations** a steep hill to climb, with frequent `CUDA out of memory` errors during fine-tuning efforts on high-capacity cards like the RTX 4090. They are investigating alternatives such as **QLoRA**.
- **Published Pearl**: Community members are hitting the library stacks with the publication of an academic article on **medical language models**, available through this [DOI](https://doi.org/10.1093/jamia/ocae120).
- **Troubleshooting Colossus**: Members are brainstorming on multi-GPU setups for fine-tuning **Llama-3-8B** models with prompt templates in Colab, while wrestling with pesky mixed precision errors stating "Current loss scale at minimum." Resources are being shared, including the [Axolotl dataset formats documentation](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format), for bettering these massive computation endeavors.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **NSFW Content in Datasets Sparks Debates**: Technical discussions have surfaced regarding the challenges of processing **Common Crawl datasets**, specifically addressing the issue of **NSFW content** and highlighting a code modification for image handling at [cc2dataset](https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84). Simultaneously, debates question **Hugging Face**'s hosting policies for datasets that could contain sensitive materials, with their own [uncurated dataset publication](https://huggingface.co/datasets/HuggingFaceFW/fineweb) coming into scrutiny.

- **Content Moderation Challenges and Legal Worries**: LAION community discusses the balance between dataset accessibility and moderation, with some highlighting the convenience of a **complaint-driven** restriction system on Hugging Face. Concerns regarding anime-related datasets and the pressure it puts on users to discern **pornographic content** have sparked serious discussions about potential legal repercussions.

- **Dissatisfaction with GPT4o's Performance**: Users have expressed dissatisfaction with GPT4o, citing problems with **self-contamination** and a perceived failure to meet the performance standards set by GPT4 despite improvements in multi-modal functionality.

- **Transformer Circuits and Autoencoders Stir Technical Debate**: A call for transparency in AI systems, especially in the **Transformer Circuits Thread**, reflects AI engineers' concerns about the possible influence of models on societal norms. Separately, some users dissect the difference between **MLPs** and **autoencoders**, pinpointing the importance of clear architectural distinctions.

- **New Research Unveiled**: Anthropic's latest insights on the **Claude 3 Sonnet** model have been brought to attention, revealing neuron activations for concepts such as the Golden Gate Bridge and the potential for influential model tuning, with detailed research published at [Anthropic](https://www.anthropic.com/research/mapping-mind-language-model).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**OpenAI's Alleged NDA Overreach**: OpenAI leadership claimed ignorance over threats to ex-employees' vested equity for not signing NDAs, but [documents with leadership's signatures](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees) suggest otherwise. Ex-employees were pressured with seven-day windows to sign or face losing millions.

**Model Performance Headlines**: **Gemini 1.5 Pro** impressively topped the Reward Bench Leaderboard for generative models, as indicated by [Jeff Dean's tweet](https://huggingface.co/spaces/allenai/reward-bench), while **News Corp and OpenAI** entered a multi-year deal, allowing AI utilization of News Corp content, as per [this announcement](https://fxtwitter.com/maxwelltani/status/1793375460879110564).

**Merch in a Flash**: Nathan Lambert's Shopify store, [Interconnects](https://interconnects.myshopify.com/), launches amidst lighthearted uncertainty about operations and with community-driven product adjustments for inclusivity; he assures ethical sourcing.

**The Emergence of AI Influencers?**: TikTok's teen demographic reportedly resonates with content generated by bots, highlighting the potential for AI-created content to go viral. The platform stands out as a launchpad for careers like **Bella Poarch's**.

**Anthropic AI's Golden Gate Focus**: A whimsical experiment by [Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) altered Claude AI's focus to obsess over the Golden Gate Bridge, leading to a mix of amusement and interest in the AI community.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter Swings Open the Gates to Advanced AI Tools**: OpenRouter now facilitates the use of **Anthropic** and **Gemini** models with a syntax matching OpenAI's, broadening the landscape for AI practitioners. Supported tool calls and function usage instructions can be found in the [documentation](https://openrouter.ai/docs#tool-calls).

**Lumimaid 70B Sashays into the AI Theater**: Aimed specifically at roleplay scenarios, the **Lumimaid 70B** model was tweaked and let loose by the NeverSleep team and details can be scooped from their [announcement page](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b).

**Calling all Roleplayers to a New Digital Realm**: A new roleplaying app granting a free tier has launched, leveraging OpenRouter's multifaceted AI characters, with the creator keen on gathering feedback via [RoleplayHub](https://www.roleplayhub.app/chat).

**Tech Snags and Community Dialogues Tangle in General Channel**: Software patches were applied to mend streaming issues with models like Llama-3, and the release of Mistral-7B v0.3 spewed some confusion due to new vocab/tokenizer—uncertainty lingered about if it should be a distinct model route or a direct route upgrade. Meanwhile, Cohere's Aya initiative garnered attention offering multilingual AI research spanning 101 languages, find out more [here](https://cohere.com/research/aya).

**Economies of Scale Kick in for AI Model Access**: Sharp price reductions have been executed for several models, including a tempting 30% off for `nousresearch/nous-hermes-llama2-13b`, among others. These markdowns are stirring up the market for developers and enthusiasts alike.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Batch Inference for GenAI Pre-Processing**: *Batch inference* is highlighted as a key technique for data pre-processing in **GenAI applications**, with the potential to enhance analysis and querying efficiency. LlamaIndex's integration and more details on the practice can be found [here](https://t.co/vnuvvypZCz).

- **RAG-Powered Job Search Assistant Blueprint**: A **RAG-powered** job search assistant has been created using **@gokoyeb**, **@MongoDB**, and **@llama_index**, demonstrating real-time response streaming and the tutorial is available [here](https://t.co/qsfx4TdvXz).

- **Nomic Embed's Localized Strategy**: **Nomic Embed** now facilitates completely local embeddings along with dynamic inference, blending the benefits of both local and remote embeddings, as expanded upon [here](https://t.co/mPFVQXk5tq).

- **Secure Your Spot for the Tech Meetup**: Engineers interested in joining an upcoming **Tuesday meetup** should note that the slots are running out, with additional details accessible [here](https://t.co/Nx4FiGB8pH).

- **Scaling Up RAG Embedding Models Peaks Interest**: Discussions surfaced around the effectiveness of bigger AI models in improving **RAG embeddings**, without landing on a clear consensus. Reference to the *ReAct algorithm* and advice on custom similarity scores utilizing an `alpha` parameter can be found in the **LlamaIndex documentation** and the discussion of these topics included links to detailed articles and papers.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Podcast with Yi Tay Misses the Boat**: The community wished for spotlights on **scaling laws** during Yi Tay's podcast on Reka/Google, but these insights were missing as the podcast had been pre-recorded.

- **Mistral v0.3 Sparks Mixed Reactions**: **Mistral 7B v0.3 models** have been released, boasting enhancements like a 32K extended vocabulary, new v3 tokenizer, and function calling capabilities, leading to both excitement and criticism [Mistral's newest chapter](https://x.com/Gradio/status/1793367718835659243).

- **Hot Takes on Open-Source AI**: A contentious opinion piece claiming open-source AI poses investment risks and national security concerns ignited debate, with detractors calling out the author for apparent OpenAI favoritism and a narrow perspective.

- **The Quest for a Universal Speech-to-Speech API**: The community discussed workarounds for **OpenAI's yet-to-be-released speech-to-speech API**, pointing to **Pipecat and LiveKit** as current alternatives, with a preference for Pipecat.

- **RAG Gets Real**: Practical applications and challenges of **Retrieval-Augmented Generation (RAG)** were exchanged among members, with a particular reference made to [a PyData Berlin talk](https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html) on RAG deployment in medical companies.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Innovative Prompt Management with VSCode**: Engineers are planning to manage prompts using VSCode to maintain efficiency, including a substantial list of nearly **500k tokens of system prompts** for **Gemini 1.5 Pro**. The ingenuity was met with enthusiasm, and suggestions for additional system prompts were solicited.

- **Favorable Reception for CLI Improvement**: The introduction of a new terminal option `--no_live_response` via a [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1278) was well-received for its potential to smooth out terminal UI issues. Steve235lab's contribution was praised as a noteworthy improvement.

- **Spotlight on Component Teardowns and Replacement Chips**: Members discussed the teardown of **Apple AirPods Pro** and the use of the **ESP32 pico chip** in the Atom Echo for alternative projects, noting the necessary reflashing. Supplementary information such as datasheets provided by ChatGPT was also recognized as beneficial.

- **Tool Praise: M5Stack Flow UI Software**: The [M5Stack Flow UI software](https://flow.m5stack.com) was commended for its support of multiple programming languages and the potential to convert Python scripts to run LLM clients, such as OpenAI, showcasing the flexible integration of hardware and AI-driven applications.

- **Skipping the macOS ChatGPT Waitlist**: A potentially controversial macOS ChatGPT app waitlist workaround from [@testingcatalog](https://x.com/testingcatalog/status/1793347117458636981) was shared, providing a 'cheat' through careful timing during the login process. This information could have implications for software engineers seeking to understand or leverage user behavior and application exploitability.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Challenging the Taylor Takedown**: Members questioned the **efficacy of Taylor series in approximations**, noting that they are only accurate close to the reference point. It was highlighted that range reduction might not be the optimal path to perfect precision, and interval partitioning could offer a better solution.

**Range Reduction Rethink**: The group debated over the use of **range reduction techniques**, suggesting alternatives like reducing to **[0, pi/4]**, and referred to **IBM's approach** as a practical example of interval partitioning found in their [implementation](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/s_sin.c;hb=HEAD).

**IBM's Insights**: An IBM source file was mentioned in a suggestion to address range reduction problems by treating fmod as an integer, viewable [here](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/branred.c;hb=HEAD).

**Mathematical Complexity Calmly Contemplated**: There was a consensus that the computations for perfect accuracy are complex, especially for large numbers, though typically not slow—a mix of admiration and acceptance for the scientific intricacies involved.

**Shape Shifting in ShapeTracker**: The group explored *ShapeTracker* limitations, concluding that certain sequences of operations like `permute` followed by `reshape` lead to multiple views, posing a challenge in chaining movement operations effectively. The utility of tensor masking was discussed, with emphasis on its role in tensor slicing and padding.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Warm Welcome for Global Creatives**: A friendly banter marked the entrance of newcomers into the fold, including a **UI Designer** from Taiwan.
- **Navigating the AI Landscape**: One member gave a crisp direction for interacting with an AI, citing a particular channel and the `@coral` handle for assistance.
- **Cohere Amplifies Multilingual AI Reach**: Cohere's announcement of **Aya 23 models** heralds new advancements, offering tools with [8 billion and 35 billion parameters](https://huggingface.co/CohereForAI/aya-23-35B) and touting support for a linguistic range encompassing 23 languages.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GraphRAG Gains Traction for Graph-Modeled Info**: Members discussed that **GraphRAG** shines when source data is naturally graph-structured, though it may not be the best choice for other data formats.
  
- **PySpark Speeds Up Embedding Conversions**: AI engineers are experimenting with **PySpark pandas UDF** to potentially enhance the efficiency of embeddings processing.

- **Challenges with Persistence in Pinecone**: A shared challenge within the community focused on the inefficiencies of **persistence handling** versus frequent instance creation in Pinecone, with dissatisfaction expressed regarding mainstream solutions like *pickle*.

- **APIs and Instruction Tuning in the Spotlight**: Upcoming event "How to Develop APIs with LangSmith for Generative AI Drug Discovery Production" set for May 23, 2024, and a new [YouTube video](https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt) explains the benefits of Instruction Tuning for enhancing LLMs' adherence to human instructions.

- **Code Modification and Retriever Planning**: Engineers are currently seeking efficient retrievers for planning code changes and techniques to prevent LLMs from undercutting existing code when suggesting modifications.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral Gets a Boost in Vocabulary and Features**: The newest iterations of [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) now boast an **extended vocabulary of 32768 tokens**, **v3 Tokenizer support**, and function calling capabilities, with installation made easy through `mistral_inference`.

- **Enhancements to Mistral 7B Paired with Community Approval**: The launch of the [Mistral-7B instruct version](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) has received a casual thumbs up from Eldar Kurtic with hints of more improvements to come, as seen in a [recent tweet](https://twitter.com/_EldarKurtic/status/1793407795909599325?t=zhtA3A5nq23HfUBkt441mQ&s=19).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **GaLore and InRank Break New Ground**: A session with Jiawei Zhao delved into **Gradient Low-Rank Projection (GaLore)** and **Incremental Low-Rank Learning (InRank)** which offer reductions in memory usage and enhancements in large-scale model training performance.

- **Event Sync Woes**: An inquiry was made about integrating an event calendar with Google Calendar, highlighting a need to track upcoming discussions to avoid missing out.

- **Image Recon with ImageMAE Marks Scalability Leap**: The ImageMAE paper was shared, presenting a scalable self-supervised learning approach for computer vision using masked autoencoders, with impressive results from a vanilla ViT-Huge model achieving 87.8%.

- **Community Spirits High**: A member voiced their appreciation for the existence of the channel, finding it a valuable asset for sharing and learning in the AI field.

---

# PART 2: Detailed by-Channel summaries and links



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1242914896065335408)** (1009 messages🔥🔥🔥): 

- **Mistral v3 Support Announced**: Unsloth now supports [Mistral v3](https://twitter.com/danielhanchen/status/1793354069987529100). Members quickly tested it and shared mixed initial feedback regarding training losses and performance.

- **Discussion on LLaMa 3 Reserved Tokens Bugs**: Users discussed "buggy" reserved tokens in LLaMa 3's base weights, including potential fixes and the impact on instruct models. A member noted, "Some of LLaMa 3's base (not instruct) weights are 'buggy'. Unsloth auto-fixes this."

- **Debate on GPU Resource Utilization**: There's confusion over underutilization with Mistral 7B on a 79GB H100 GPU. Suggestions varied, from increasing batch size to updating training code, with a user noting, "you need to increase batches to make use of the GPU."

- **Phi 3 Medium 4-bit Released**: [Phi 3 Medium 4k Instruct](https://huggingface.co/unsloth/Phi-3-medium-4k-instruct-bnb-4bit) is now available, with additional support coming soon. The announcement included links to [Colab notebooks](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing) for easy access.

- **Continued Pre-training Notebook**: A notebook for continued pretraining was shared, aimed at preserving instruction-following traits during domain-specific fine-tuning. It's available [here](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) and members are encouraged to experiment and share results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/Phi-3-medium-4k-instruct-bnb-4bit">unsloth/Phi-3-medium-4k-instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://discord.gg/M2spsbCbwN">Join the TheBloke AI Discord Server!</a>: For discussion of and support for AI Large Language Models, and AI in general. | 23932 members</li><li><a href="https://huggingface.co/datasets/cognitivecomputations/samantha-data/tree/main">cognitivecomputations/samantha-data at main</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1793694923683938416">Tweet from Unsloth AI (@UnslothAI)</a>: We&#39;re so happy to announce that Unsloth is part of the 2024 @GitHub Accelerator program!🦥  If you want to easily fine-tune LLMs like Llama 3, now is the perfect time! http://github.blog/2024-05-2...</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...</li><li><a href="https://datta0.substack.com/p/ai-unplugged-11-lora-vs-fft-multi">AI Unplugged 11: LoRA vs FFT, Multi Token Prediction, LinkedIn&#x27;s AI assistant</a>: Insights over Information</li><li><a href="https://x.com/q_brabus/status/1793227643556372596">Tweet from QBrabus eu/acc (@q_brabus)</a>: @apples_jimmy @ylecun @iamgingertrash Question: Regarding the upcoming LLaMa 3 400B+ model, will it be open-weight? There are several rumors about this...  Answer: No, it is still planned to be open a...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_diffusion/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc">can i get a chicken tendie combo please</a>: no description found</li><li><a href="https://youtu.be/e3Gvq4NDqvw?si=3b2lILNAiR5CZJMW">Scarlett Johansson demands answers after OpenAI releases voice &quot;eerily similar&quot; to hers</a>: Scarlett Johansson is demanding answers from OpenAI and its CEO, Sam Altman, after it released a ChatGPT voice that she says sounds &quot;eerily similar&quot; to her o...</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets</a>: Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit</li><li><a href="https://lu.ma/1wu5ppl5">GPU Optimization Workshop · Luma</a>: We’re hosting a workshop on GPU optimization with stellar speakers from OpenAI, NVIDIA, Meta, and Voltron Data. The event will be livestreamed on YouTube, and…</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/shimmyshimmer/unsloth">GitHub - shimmyshimmer/unsloth: 5X faster 60% less memory QLoRA finetuning</a>: 5X faster 60% less memory QLoRA finetuning. Contribute to shimmyshimmer/unsloth development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1243306406195626034)** (1 messages): 

- **Phi-3 and Mistral v3 power up Unsloth with new models**: Unsloth now supports Phi-3, Mistral v3, and other [new models](https://github.com/unslothai/unsloth/releases/tag/May-2024). The update also addresses all Llama 3 issues, enhancing finetuning performance.
  
- **Free Notebooks for easy access**: Users can explore the new models using [Phi-3 medium notebook](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing), [Mistral v3 notebook](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing), or [ORPO notebook](https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing).

- **Broad support for 4-bit models**: Visit Unsloth's [Hugging Face page](https://huggingface.co/unsloth) for a variety of 4-bit models including Instruct. New additions like [Qwen](https://github.com/unslothai/unsloth/pull/428) and [Yi 1.5](https://huggingface.co/unsloth/Yi-1.5-6B-bnb-4bit) are now supported.

- **Exciting news with GitHub's 2024 Accelerator**: Unsloth is part of the [2024 GitHub Accelerator](https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/), joining 11 other projects shaping open-source AI. This recognition underscores the significant impact and innovation within the AI community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">Finetune Phi-3 with Unsloth</a>: Fine-tune Microsoft&#x27;s new model Phi 3 medium, small &amp; mini easily with 6x longer context lengths via Unsloth!</li><li><a href="https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/)">2024 GitHub Accelerator: Meet the 11 projects shaping open source AI</a>: Announcing the second cohort, delivering value to projects, and driving a new frontier.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1242916671476465895)** (81 messages🔥🔥): 

<ul>
<li><strong>Mistral v0.3 excitement and confusion:</strong> Users were initially thrilled about "Mistral v3," but it was clarified to be Mistral v0.3. One user joked about it being Olympic-level competition among LLMs.</li>
<li><strong>QKVO training issues with Qwen:</strong> Members discussed issues with training QKVO layers for Qwen models using Unsloth, noting that biases cause errors. A workaround involving not training all layers was suggested.</li>
<li><strong>Unsloth updates boost efficiency:</strong> The latest Unsloth branch offloads <em>lm_head</em> and <em>embed_tokens</em> to disk, reducing VRAM usage and speeding up operations. Users were guided to update via a specific GitHub command.</li>
<li><strong>Model loading confusion on Hugging Face:</strong> Users discussed the complications of having multiple quantized versions of a model in one repository on Hugging Face and recommended using separate repositories for each quantized version.</li>
<li><strong>Fact-checking and non-English fine-tuning of LLMs:</strong> There was an in-depth conversation about the challenges of LLMs producing truthful responses and managing language-specific fine-tuning. Multiple references and links were provided, including <a href="https://arxiv.org/abs/2306.03341">scaling-monosemanticity</a> and <a href="https://arxiv.org/abs/2302.00083">In-Context RALM</a>.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/llama-3-8b">unsloth/llama-3-8b · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2302.00083">In-Context Retrieval-Augmented Language Models</a>: Retrieval-Augmented Language Modeling (RALM) methods, which condition a language model (LM) on relevant documents from a grounding corpus during generation, were shown to significantly improve languag...</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1242971620356657274)** (1 messages): 

```html
- **Scheduled Downtime Announced**: Heads up for a **scheduled downtime** tonight at 12:00am EST. The downtime will last around 30 minutes for a database upgrade aimed at improving performance and user experience.
```
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1242916156700889141)** (897 messages🔥🔥🔥): 

```html
- **Gemini Free Usage Pleasantly Surprises**: Members celebrated that **Gemini in AI Studio** is free even for large usage (*"requests in this UI are free"*) and exclaimed about the ability to perform fine-tuning without cost (*"finetuning for free?"). They discussed possible data privacy concerns, but openness to experimenting with the service prevailed.
- **Perplexity’s Speed Impresses**: **Web scraping** optimizations led to significant performance improvements for **searches using multiple sources**, clocking speeds much faster than previous attempts. One member reported *"web scraping taking 1.52s" compared to earlier times of over 7s* and emphasized proper use of parallel processing.
- **Perplexity vs. Other AI Tools**: Members compared **Perplexity** with other AI tools like **Gemini Pro** and **ChatGPT** regarding file handling and data processing. **Perplexity** received praise for its research and writing capabilities (*"better in both areas"*) and flexible file handling, garnering new insights on **Gemini's role** mainly for its context handling.
- **Integrating Additional Features into Perplexity**: Discussions included potential UI enhancements and tools for **Perplexity**, including the integration of **labs into the main UI** and adding functionalities like history saving and support for formats like **CSV**. The aim is to potentially transform **Perplexity** from a decent tool to the *"best AI website"*.
- **Model Usage and Rate Limits Challenge Members**: Encountering **API rate limits** and exploring various models, members juggled between **Haiku**, **Cohere**, and **GPT-4-free**, sharing frustrations and strategies for optimal usage given free and cost-efficient tiers. They explored alternatives and workarounds while emphasizing the balance between accuracy and context sizes.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.loom.com/share/8cc0ce3bff334b64b558f5a04bdcf2b2?sid=ff6eb1ac-2ba9-434b-ad1b-bc372d2ac9b0">Exploring AI Conversations and Redundancy</a>: In this video, I share my experience with AI conversations and highlight the issue of redundancy. I demonstrate how simple calculations and conversions can lead to repetitive responses from AI models ...</li><li><a href="https://www.theregister.com/2016/03/23/npm_left_pad_chaos/">How one developer just broke Node, Babel and thousands of projects in 11 lines of JavaScript</a>: Code pulled from NPM – which everyone was using</li><li><a href="https://youtu.be/NPOHf20slZg">$30,000,000 AI Is Hiding a Scam</a>: It&#39;s time to see how far the Rabbit hole goes...Support Investigative Journalism: ► Patreon: https://patreon.com/coffeezillaFollow:►Ed Zitron: https://www.wh...</li><li><a href="https://forums.macrumors.com/threads/m4-ipad-pro-11-first-impressions-performance-heat-pwm-and-others.2426567/">M4 iPad Pro 11” first impressions (performance, heat, PWM and others)</a>: Hello!  I just came from my local Apple Store, and I’ve been closely inspecting and testing the new M4 iPad Pro. This is not a review, but it can provide useful details that many people could be inter...</li><li><a href="https://en.wikipedia.org/wiki/Oil_futures_drunk-trading_incident">Oil futures drunk-trading incident - Wikipedia</a>: no description found</li><li><a href="https://github.com/Rob--W/cors-anywhere/issues/301">PSA: Public demo server (cors-anywhere.herokuapp.com) will be very limited by January 2021, 31st · Issue #301 · Rob--W/cors-anywhere</a>: The demo server of CORS Anywhere (cors-anywhere.herokuapp.com) is meant to be a demo of this project. But abuse has become so common that the platform where the demo is hosted (Heroku) has asked me...</li><li><a href="https://techcrunch.com/2024/05/23/bing-is-down-bringing-duckduckgo-and-ecosia-down-too">Bing’s API was down, taking Microsoft Copilot, DuckDuckGo and ChatGPT&#039;s web search feature down too | TechCrunch</a>: Bing, Microsoft’s search engine, was working improperly for several hours on Thursday in Europe. At first, we noticed it wasn’t possible to perform a web</li><li><a href="https://docs.anthropic.com/en/api/rate-limits);">Welcome to Claude - Anthropic</a>: no description found</li><li><a href="https://www.anthropic.com/contact-sales">Contact Anthropic</a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://www.apple.com/ipad/compare/?modelList=ipad-pro-13-m4,ipad-pro-11-m4,ipad-air-11-m2">iPad - Compare Models</a>: Compare resolution, size, weight, performance, battery life, and storage of iPad Pro, iPad Air, iPad, and iPad mini models.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1242944272576286730)** (7 messages): 

- **Taiwan Semiconductor directs curiosity**: A user linked to a [Perplexity AI search result](https://www.perplexity.ai/search/Taiwan-Semiconductor-remote-k.5AQq3LQkGX5eg4Nbh9jA) about Taiwan Semiconductor, likely discussing aspects of remote work or industry developments.
- **Analyzing scents with AI humor**: Another user shared a humorous [search result](https://www.perplexity.ai/search/Wie-riecht-der-SgLtFy.iRF6KuPxZuKQHVg#0), indicating interest in how something smells, demonstrating Perplexity AI’s ability to handle wide-ranging queries.
- **Fatal incident in IoS**: A tragic event involving 9 fatalities garnered attention with a shared link to a [search about the incident](https://www.perplexity.ai/search/9-killed-in-IOsCm6NBREimQdcGey_1fQ#0). The search may involve interpretations or official reports on the event.
- **Bing API topic surfaces**: Interest in [Bing's API capabilities](https://www.perplexity.ai/search/Bings-API-is-Plv4H_4RT7ShLq2XT41R2A) led to a shared search link, likely covering how Bing API is perceived or utilized.
- **Perplexity AI elucidated**: A user shared a search link explaining [what Perplexity AI is](https://www.perplexity.ai/search/What-is-Perplexity-uyV3gThHQEa1tWgRyN0sQw). This suggests ongoing curiosity and learning about the platform itself among users.
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1242925804300533911)** (2 messages): 

- **Metquity decides to explore alternatives**: Metquity expressed a plan to shift to other tools, stating, *"maybe I will build with something else and come back to it when its better."* This indicates interest in returning once improvements are made.
- **Neuraum notices discrepancies between web and API outputs**: Neuraum queried why using the same model and prompt could yield different outputs on the web version and the API. He pointed out that *"using the API the outputs are wrong, eventhough the browsing function works,"* seeking insights into this inconsistency.
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1242916745883156581)** (141 messages🔥🔥): 

- **Instruction Tuning with ColBERT-style Models**: A member inquired about the experiences of others with instruction tuning embeddings models, specifically with ColBERT-style models. They shared some relevant papers, including [INSTRUCTOR: A Framework for Embedding Text with Task Instructions](https://arxiv.org/abs/2212.09741) and [TART: A Multi-task Retrieval System Trained on BERRI with Instructions](https://arxiv.org/abs/2211.09260).
  
- **Bayesian Calculation Concerns in FDP Exam**: Another member discussed a Bayesian question from an FDP accreditation exam and worked through a calculation. They pointed out a potential error in the FDP Institute's probability calculation, preferring their own method.

- **Axolotl Tutorial on JarvisLabs**: A helpful tutorial video was shared on how to run Axolotl on JarvisLabs, available on [YouTube](https://youtu.be/Y9464wasHuE) and linked to relevant resources such as JarvisLabs and the Axolotl GitHub repository.

- **Miniforge/Mamba for AI/ML Environments**: There was a discussion about the advantages of using Miniforge and Mamba for creating and managing conda environments over alternatives like pyenv, highlighting the flexibility and speed benefits of mamba.

- **Schulman's Two-Step Fine-Tuning Process**: Members discussed John Schulman’s comments on iterative supervised fine-tuning vs. reinforcement learning (RL) for improving models beyond basic fine-tuning, emphasizing an iterative process to fully align models with high-quality human data.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/leaderboard,">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>: no description found</li><li><a href="https://predibase.com/">Predibase: The Developers Platform for Fine-tuning and Serving LLMs</a>: The fastest and easiest way to fine tune and serve any open-source large language model on state-of-the-art-infrastructure hosted within your private cloud.</li><li><a href="https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f">Exploring Fine-tuning with Honeycomb Example</a>: In this video, I walk you through the process of fine-tuning a model using the honeycomb example. I provide step-by-step instructions on cloning the repository, installing dependencies, and running th...</li><li><a href="https://x.com/younesbelkada/status/1793211607713235295">Tweet from younes (@younesbelkada)</a>: 🚨 New optimizer in @huggingface transformers Trainer 🚨  LOMO optimizer can be now used in transformers library https://github.com/OpenLMLab/LOMO  Great work from LOMO authors! 🧵</li><li><a href="https://youtu.be/Y9464wasHuE">How to run axolotl on JarvisLabs | Tutorial</a>: Check out axolotl on JarvisLabs : jarvislabs.ai/templates/axolotlCheck out axolotl github : https://github.com/OpenAccess-AI-Collective/axolotl</li><li><a href="https://www.youtube.com/watch?v=Z2NxN9sl9Vk">Vincent D. Warmerdam - Active Teaching, Human Learning</a>: Want a dataset for ML? Internet says you should use ... active learning!It&#39;s not a bad idea. When you&#39;re creating your own training data you typically want t...</li><li><a href="https://www.youtube.com/watch?v=gDk7_f3ovIk">Bulk Labelling and Prodigy</a>: Prodigy is a modern annotation tool for collecting training data for machine learning models, developed by the makers of spaCy. In this video, we&#39;ll show a b...</li><li><a href="https://www.astralcodexten.com/p/asteriskzvi-on-californias-ai-bill">Asterisk/Zvi on California&#x27;s AI Bill</a>: ...</li><li><a href="https://youtu.be/C9p7suS-NGk?si=AM4sr3OXeFRKZo7c">Vincent Warmerdam - Keynote &quot;Natural Intelligence is All You Need [tm]&quot;</a>: In this talk I will try to show you what might happen if you allow yourself the creative freedom to rethink and reinvent common practices once in a while. As...</li><li><a href="https://arxiv.org/abs/2212.09741">One Embedder, Any Task: Instruction-Finetuned Text Embeddings</a>: We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions: every text input is embedded together with instructions explaining the use case (e.g., task and domain desc...</li><li><a href="https://arxiv.org/abs/2211.09260">Task-aware Retrieval with Instructions</a>: We study the problem of retrieval with instructions, where users of a retrieval system explicitly describe their intent along with their queries. We aim to develop a general-purpose task-aware retriev...</li><li><a href="https://youtu.be/sTQaJyrI-zg?si=krcLKWRmqT9SH8X5&t=1389),">Stanford CS25: V2 I Common Sense Reasoning</a>: February 14, 2023Common Sense ReasoningYejin ChoiIn this speaker series, we examine the details of how transformers work, and dive deep into the different ki...</li><li><a href="https://github.com/conda-forge/miniforge?tab=readme-ov-file#install">GitHub - conda-forge/miniforge: A conda-forge distribution.</a>: A conda-forge distribution. Contribute to conda-forge/miniforge development by creating an account on GitHub.</li><li><a href="http://annotate.calmcode.io">no title found</a>: no description found</li><li><a href="https://anywidget.dev/en/community/">Community | anywidget</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=goaBFxGhp6Y),">Enhancing Jupyter with Widgets with Trevor Manz - creator of anywidget</a>: In this (first!) episode of Sample Space we talk to Trevor Mantz, the creator of anywidget. It&#39;s a (neat!) tool to help you build more interactive notebooks ...</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>: no description found</li><li><a href="https://buttondown.email/ainews/archive/ainews-anthropic-cracks-the-llm-genome-project/">[AINews] Anthropic&#x27;s &quot;LLM Genome Project&quot;: learning &amp; clamping 34m features on Claude Sonnet</a>: Dictionary Learning is All You Need. AI News for 5/20/2024-5/21/2024. We checked 7 subreddits, 384 Twitters and 29 Discords (376 channels, and 6363 messages)...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1242928427158212650)** (14 messages🔥): 

- **Automate Standup Transcripts with Ticket Updates**: A proposed service reads transcripts of standup meetings and updates tickets with statuses mentioned. This involves fine-tuning using examples of standup conversations and their correlation with ticket updates.
  
- **Custom Stop Sequences to Prevent Jailbreaks**: Members discussed fine-tuning models to use custom stop sequences like a hash instead of common tokens like "###", aiming to resist jailbreak attempts. One suggested fine-tuning to ignore jailbreak prompts, though acknowledging the challenge of anticipating every prompt in advance.

- **Lightweight Text Tagging and Entity Generation Models**: A member suggested several lightweight LLM projects, including one similar to GliNER for text tagging and classification and another to generate training data for tools like spaCy. Another interesting project proposed was creating an LLM to generate cool names for Python packages.

- **Prompt Injection Protections**: Discussion highlighted the improbability of completely preventing jailbreaks directly on models, pointing to [prompt injection protection tools](https://arxiv.org/pdf/2307.15043) as better solutions. Shared resources included a list of protection libs/tools and [a collection of related papers](https://huggingface.co/collections/leonardlin/prompt-injection-65dd93985012ec503f2a735a).

- **EDA Assistant and Suggestions**: A project for a chatbot to assist data scientists in EDA of time series data, identified through [a YouTube demo](https://www.youtube.com/watch?v=glwBlONacPY), with version 2 planning fine-tuning to improve deductive reasoning and formatting. Another assistant aimed to process EDA outputs into actionable steps, exploring cost-effective and faster implementation methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=glwBlONacPY">Unleash the Power of GPT-4o: Exploratory Data Analysis with ObexMetrics</a>: Revolutionize your time series data analysis with ObexMetrics&#39; cutting-edge EDA assistant, powered by GPT-4o. Seamlessly chat with your data to effortlessly ...</li><li><a href="https://llm-tracker.info/research/Prompt-Injection-Protection">Prompt Injection Protection</a>: We should have a tool to be able to track Github project activity… Papers collection: https://huggingface.co/collections/leonardlin/prompt-injection-65dd93985012ec503f2a735a Techniques: Input heuristi...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1242927291856588941)** (18 messages🔥): 

- **Modal credits granted quickly**: One user confirmed receiving Modal credits quickly after filling out a form. Another user appreciated the quick allotment and thanked the team for the free credits.
- **Persistent errors with finetuning llama 3 8b**: A member reported persistent issues while running a llama 3 8b fine-tuning job despite assistance from another user. They linked to a specific Discord message thread for more context ([link to thread](https://discord.com/channels/1238365980128706560/1242939065058328776/1242952109981302945)).
- **Credits expiring quirk**: A user noted what seemed to be a quirk where credits appeared to expire overnight in the billing panel, but the Live Usage dropdown still showed the full amount. They thanked the team for the clarity.
- **Finetuning LLM for generating layouts**: A user inquired about the feasibility of fine-tuning LLMs to generate layouts using datasets like publaynet and rico. These requests underscore the varied applications users are exploring with their models.
- **Downloading trained models to S3**: Another member clarified that it's possible to download trained models to an S3 bucket using the `modal volume get` command. They also mentioned the option to directly mount an S3 bucket as a volume, linking to the [relevant documentation](https://modal.com/docs/guide/cloud-bucket-mounts).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modal.com/docs/guide/cloud-bucket-mounts">Cloud bucket mounts</a>: The modal.CloudBucketMount is a mutable volume that allows for both reading and writing files from a cloud bucket. It supports AWS S3, Cloudflare R2, and Google Cloud Storage buckets.</li><li><a href="https://modal.com/docs/reference/cli/volume#modal-volume-get">modal volume</a>: Read and edit modal.Volume volumes.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1243037076140720189)** (1 messages): 

- **Illustrative Example Shared by Hamel**: A member appreciated an example shared by Hamel on Twitter, describing it as "incredibly illustrative." They also thanked Jeremy Howard for the related notebook, providing links to both the [tweet](https://twitter.com/HamelHusain/status/1793319488731107718) and the [GitHub repository](https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb).

**Link mentioned**: <a href="https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb">lm-hackers/lm-hackers.ipynb at main · fastai/lm-hackers</a>: Hackers&#39; Guide to Language Models. Contribute to fastai/lm-hackers development by creating an account on GitHub.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1242922773299466300)** (32 messages🔥): 

- **Troubleshooting Llama-3 Access Issues**: Multiple users, including [@mark6871 and @dhar007](https://discord.com/channels/****), faced issues accessing the Llama-3 model despite having permission on Hugging Face. The resolution involved generating an access token on Hugging Face and entering it during the terminal prompt.

- **Overcoming CUDA Memory Errors**: Users like [@dhar007](https://discord.com/channels/****) experienced "CUDA out of memory" errors while training models with RTX5000 GPUs. They resolved it by adjusting the batch size and monitoring GPU usage using `nvidia-smi`.

- **Datapoint on Mistral LoRA Training**: [@damoncrockett](https://discord.com/channels/****) shared their experience running the Mistral LoRA example, which took 2.5 hours and $4 on a single A100 GPU, but noted undertraining with only one epoch on a small dataset.

- **Feedback and Support for Jarvislabs Credits**: Users like [@rashmibanthia](https://discord.com/channels/****) expressed gratitude for the credits and shared positive experiences with Jarvislabs compared to other services. There were also several inquiries about missing credits and how to confirm sign-ups ([@manjunath_63072] and [@darkavngr](https://discord.com/channels/****)).

- **Queries on Spot Instances and Repo Saving on Jarvislabs**: [@tokenbender](https://discord.com/channels/****) faced difficulties finding spot instances, while [@nisargvp](https://discord.com/channels/****) inquired about saving repositories without pausing the instance to avoid credit usage.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1243171899903705119)** (4 messages): 

- **Gated repo access resolved with login tip**: A member experienced an error when attempting to access a gated repository despite accepting the terms. Another member suggested using the `huggingface-cli login` command, which resolved the issue and allowed the training to proceed successfully.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1242919849215266868)** (5 messages): 

- **GitHub sign-up email mismatch heads up**: A member alerted, *"Just a heads to all who signed up with GitHub but have a different email they used for the conference, you can set a different email address after signing up."*
- **Issues with Gmail "+" sign**: Users expressed frustration as it *"does not seem to accept my gmail address with a `+` sign in it."*
- **Notification label confusion and Maven credits**: One user questioned if using the Maven registered address for notifications ensures automatic credit addition.
- **Credit receipts queried**: A member asked the group, *"Did anybody get the credit already?"*
- **Awaiting clarification on credits**: Another user noted they had signed up and are *"Now I just have to wait. I hope official info when something is happening."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1243196011443126312)** (2 messages): 

- **Attendee Inquires About Recording Availability**: A participant eagerly asked, *"Will there be a recording?"* This indicates anticipation for an upcoming event and interest in having access to later review.

- **Excitement for Upcoming Event**: Another participant expressed enthusiasm saying they were *"Hyped for the talk later!"* Their excitement is further highlighted by the use of an emoji and sparkle symbol.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1242919179028529242)** (209 messages🔥🔥): 

- **Axolotl's dataset troubleshooting**: Users faced issues running finetuning on a local machine, yielding JSON decoding errors. To resolve, correctly align dataset paths in configuration and use compatible formats as demonstrated in [Axolotl's documentation](https://github.com/OpenAccess-AI-Collective/axolotl).

- **LoRA hyperparameters tuning**: Discussions focused on adjusting learning rates and LoRA hyperparameters. A shared config showed `lora_r: 128`, `lora_alpha: 128`, and varying learning rates to optimize model training, and leveraging [tips from Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms). 

- **Training duration queries**: Users discussed training times on different GPUs, noting differences in speed estimates. Hamel suggested running smaller dataset samples to avoid prolonged feedback loops, while shared axolotl examples helped guide realistic expectations and adjustments.

- **Configuration for function calling**: Users sought best practices for fine-tuning models for specific tasks like text to code translation using examples like ReactFlow. Templates and prompt formats specific to their models were recommended, including reviewing [fine-tuning benchmarks](https://predibase.com/fine-tuning-index).

- **GPT judge for L2 evals**: For fine-tuning evaluations, using GPT-4 as a judge with refined prompts was recommended. Users considered fine-tuning judges but were advised to start with simple prompt refinements to improve alignment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/muellerzr/llama-3-8b-self-align-axolotl">muellerzr</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f">Exploring Fine-tuning with Honeycomb Example</a>: In this video, I walk you through the process of fine-tuning a model using the honeycomb example. I provide step-by-step instructions on cloning the repository, installing dependencies, and running th...</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/ac50ed?item=49gdm84u3wp">no title found</a>: no description found</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets</a>: Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit</li><li><a href="https://prodi.gy/docs/large-language-models#more-config">Large Language Models (LLMs) · Prodigy · An annotation tool for AI, Machine Learning &amp; NLP</a>: A downloadable annotation tool for NLP and computer vision tasks such as named entity recognition, text classification, object detection, image segmentation, A/B evaluation and more.</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/ac50ed?item=bf4nff4j6bo">no title found</a>: no description found</li><li><a href="https://predibase.com/fine-tuning-index">The Fine-tuning Index</a>: Performance benchmarks from fine-tuning 700+ open-source LLMs</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/777">Support Fuyu-8B · Issue #777 · OpenAccess-AI-Collective/axolotl</a>: ⚠️ Please check that this feature request hasn&#39;t been suggested before. I searched previous Ideas in Discussions didn&#39;t find any similar feature requests. I searched previous Issues didn&#39;t...</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html">Axolotl - Instruction Tuning</a>: no description found</li><li><a href="https://lucasvw.github.io/posts/19_llm_fine_tuning/.">Lucas van Walstijn - LLM fine-tuning 101</a>: no description found</li><li><a href="https://huggingface.co/nisargvp/hc-mistral-alpaca">nisargvp/hc-mistral-alpaca · Hugging Face</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - Template-free prompt construction</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-small-128k-instruct">microsoft/Phi-3-small-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/nisargvp/2fcbf41d4a8e7149e6c4fb9a630edfd8">dataset_generation error while runinng hc.yml locally</a>: dataset_generation error while runinng hc.yml locally - gist:2fcbf41d4a8e7149e6c4fb9a630edfd8</li><li><a href="https://m.youtube.com/watch?v=lzXKsY3bANw">Image Classification with scikit-learn</a>: Scikit-Learn is most known for building machine learning models for tabular data, but that doesn&#39;t mean that it cannot do image classification. In this video...</li><li><a href="https://reactflow.dev/.">Node-Based UIs in React – React Flow</a>: Highly customizable React library for workflow builders, no-code apps, image processing, visualizers, and more</li><li><a href="https://forum.obsidian.md/t/obsidian-vscode-editor-elevate-your-code-editing-experience-in-obsidian/69057">Obsidian VSCode Editor: Elevate Your Code Editing Experience in Obsidian!</a>: Are you tired of switching between different applications while working on your notes and code?  Do you wish there was a seamless way to view and edit code files within Obsidian?  Look no further! 😏 ...</li><li><a href="https://github.com/parlance-labs/ftcourse">GitHub - parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>: Things I Learned From Hundreds of Experiments</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1652">load explicit splits on datasets by winglian · Pull Request #1652 · OpenAccess-AI-Collective/axolotl</a>: make it a little easier to load partial splits, like: datasets:   - path: ...     type: ...     split: &quot;train[:10%]&quot;</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset-formats/inst_tune.qmd#L7">axolotl/docs/dataset-formats/inst_tune.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://www.phorm.ai/query?projectId=e315ba4a-4e14-421f-ab05-38a1f9076f25&threadId=eff87042-122e-4774-8526-0a023e3e919f">Using Shards with Datasets | OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1242939065058328776)** (80 messages🔥🔥): 

- **Llama 3 Fine-Tuning Issues**: Members discussed various issues when fine-tuning Llama 3, such as encountering a `NoneType` error in Modal and CUDA-related problems. Solutions included using Docker images, adjusting `sequence_len`, and reference configurations like [this config](https://wandb.ai/oaaic/fused-cel-llama3/runs/kkyhjjh6/files/tmp/axolotl_config_rdbefq2r.yml).

- **Docker Solutions for Axolotl**: Many members recommended using Docker containers to resolve local setup issues with CUDA libraries and pathing errors. Links to pre-made Docker images, such as one from [Docker Hub](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018), and setups on Jarvis Labs were shared.

- **BitsAndBytes GPU Support**: The recurring error about BitsAndBytes not having GPU support was resolved by correcting CUDA paths in `.bashrc`. It was recommended to use cloud providers like Jarvis or Modal for fewer issues rather than local installations.

- **Axolotl on Different GPUs**: Members noted that Flash Attention 2 doesn’t support Turing GPUs, leading to issues on T4 systems. Alternatives like using `sdp_attention`, which is generally supported on most models, were suggested.

- **Cache Management in Axolotl**: Users discussed problems with data caching when re-running experiments on JarvisLabs. It was suggested to rename dataset files and update configuration files to ensure correct data is used, prompting a call for a caching flag feature.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html">(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA) — PyTorch Tutorials 2.3.0+cu121 documentation</a>: no description found</li><li><a href="https://www.phorm.ai/query?projectId=e315ba4a-4e14-421f-ab05-38a1f9076f25">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://jarvislabs.ai/templates/axolotl">Easily Finetune LLM with Axolotl | Jarvislabs</a>: Axolotl helps you to finetune LLM using techniques like lora, qlora and more. Edit the config file and start LLM training</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/cicd/Dockerfile.jinja">axolotl/cicd/Dockerfile.jinja at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://wandb.ai/oaaic/fused-cel-llama3/runs/kkyhjjh6/files/tmp/axolotl_config_rdbefq2r.yml">oaaic</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/.github/workflows/tests.yml#L105-L107">axolotl/.github/workflows/tests.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/config.qmd">axolotl/docs/config.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018?context=explore">no title found</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/src/common.py#L14">llm-finetuning/src/common.py at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/pull/57">better handling of pre-processing and training by winglian · Pull Request #57 · modal-labs/llm-finetuning</a>: run&#39;s preprocessing too as part of train step, but prefer to use single GPU instance, also use single gpu for merge, use newer axolotl image, add llama-3 config</li><li><a href="https://modal.com/docs/examples/llm-finetuning">Fine-tune an LLM in minutes (ft. Llama 2, CodeLlama, Mistral, etc.)</a>: Tired of prompt engineering? Fine-tuning helps you get more out of a pretrained LLM by adjusting the model weights to better fit a specific task. This operational guide will help you take a base model...</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1242929273845841930)** (96 messages🔥🔥): 

```html
- **Complimentary GPU Optimization Event**: Filippob82 announced a workshop on GPU optimization featuring speakers from OpenAI, NVIDIA, Meta, and Voltron Data. The event will be livestreamed on YouTube and discussed on [Discord](https://discord.gg/T5sx2MYd5R), with more details in the [README](https://github.com/mlops-discord/gpu-optimization-workshop) and [workshop note](https://docs.google.com/document/d/1TR_5Ax0rPqTj8I2sA7MH-aa4J7TUUt4Ji9272OP8ZJg/edit).

- **Training Tips on A100s and H100s**: Stevenmerrill inquired about general rules for training on A100 and H100 GPUs, with Tddammo recommending to absolutely use bf16 if the GPU supports it and adjust batch sizes to utilize available VRAM. Additionally, sequence lengths can also be increased due to enhanced memory capacity.

- **VRAM Calculation Challenges**: Remek1972 discussed issues with VRAM requirements for larger sequence lengths (e.g., 4096 tokens) on an A6000 GPU, finding that it leads to crashes. The conversation concluded that using mixed precision (bf16) and optimization strategies could mitigate some memory issues but larger models might necessitate offloading or quantization.

- **Paged ADAMW 8-bit Optimizer Discussion**: Lhl mentioned using the paged_adamw_8bit optimizer for efficiency and asked about any potential drawbacks, receiving assurance that performance is equivalent to normal adam. They discussed the community's experience and findings, including the benefits of 8-bit optimization for memory usage.

- **Interest in Latest Optimizers**: Lhl and Tddammo discussed experimenting with new optimizers like Sophia and Adam_LoMo. Recommendations and shared experiences pointed to potential benefits in performance, with Lhl adding links to recent discussions on Twitter regarding new optimizer benchmarks.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/younesbelkada/status/1793211607713235295">Tweet from younes (@younesbelkada)</a>: 🚨 New optimizer in @huggingface transformers Trainer 🚨  LOMO optimizer can be now used in transformers library https://github.com/OpenLMLab/LOMO  Great work from LOMO authors! 🧵</li><li><a href="https://x.com/ArmenAgha/status/1780149168692158658">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: Final Update: One more magnitude of testing Sophia. We&#39;re talking model sizes in the B&#39;s, tokens in the T&#39;s. Sophia once again wins out. For me at least this is clear evidence that Sophia ...</li><li><a href="https://github.com/huggingface/transformers/issues/22101">[Benchmark] HF Trainer optimizers (Mar-2023) · Issue #22101 · huggingface/transformers</a>: This is a rerun of Adam torch vs. apex vs HF vs adafactor RTX-3090, A100 but added BNB&#39;s 8bit Adam optimizer and probably the software has improved/changed since 14 months as well. note: 8-bit Opt...</li><li><a href="https://github.com/huggingface/transformers/pull/24338">Add SophiaG. by guilt · Pull Request #24338 · huggingface/transformers</a>: What does this PR do? This is a scratch PR showing how to test Sophia with Transformers. This is no way production ready, and certainly needs to look at licensing. But, this is helpful if someone n...</li><li><a href="https://lu.ma/1wu5ppl5">GPU Optimization Workshop · Luma</a>: We’re hosting a workshop on GPU optimization with stellar speakers from OpenAI, NVIDIA, Meta, and Voltron Data. The event will be livestreamed on YouTube, and…
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1242957873370107924)** (56 messages🔥🔥): 

- **Accelerate command issues resolved with installation tips**: A member faced issues running `accelerate launch -m axolotl.cli.train` due to missing torch and gcc dependencies, eventually solving it by installing CUDA and necessary tools through a [detailed tutorial](https://www.namehero.com/blog/how-to-install-gcc-on-ubuntu/#3-installing-gcc-compiler-on-ubuntu). Another member provided a working setup for Axolotl from scratch using conda and specified dependencies.
- **Docker implementation for Axolotl on GPU VM**: In response to someone trying to run Axolotl as a Docker image on a GCP deep learning VM with GPU, a solution was shared involving the `winglian/axolotl:main-latest` Docker image and a sample Docker run command.
- **Default parameter values in Axolotl config**: A query about default values of parameters in Axolotl's configuration led to a discussion, with one member expressing shared curiosity. The question remained partially answered within the thread.
- **Challenges running Axolotl on Modal**: A few members discussed issues trying to run Axolotl on Modal, facing errors likely due to mismatched builds and issues with gated models on Hugging Face. A pull request link for better handling of preprocessing and training was shared as a potential solution.
- **Preprocessing advice for LLM finetuning**: A member asked for advice on preprocessing datasets for LLM finetuning, mentioning feature engineering and feature selection. They referenced the [Axolotl dataset preprocessing documentation](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd) for further reading.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.namehero.com/blog/how-to-install-gcc-on-ubuntu/#3-installing-gcc-compiler-on-ubuntu">How To Install GCC On Ubuntu</a>: Let&#039;s walk through the process of installing GCC on your Ubuntu system, making the world of compilers and development tools accessible!</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl.git">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd">axolotl/docs/dataset_preprocessing.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/modal-labs/llm-finetuning/pull/57">better handling of pre-processing and training by winglian · Pull Request #57 · modal-labs/llm-finetuning</a>: run&#39;s preprocessing too as part of train step, but prefer to use single GPU instance, also use single gpu for merge, use newer axolotl image, add llama-3 config
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1243310332407971850)** (1 messages): 

- **Protein Visualization Tool Released**: Check out the [Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz) from a community contributor for creating custom visuals of proteins. This tool is highlighted in the latest community updates.
- **Rapid Image Processing with SDXL Flash**: Experience quick results with the [SDXL flash](https://huggingface.co/spaces/KingNish/SDXL-Flash) created by another community member. This space offers efficient image processing capabilities.
- **Innovative Dataset and Systems Updates**: Notable mentions include a [wikipedia dataset](https://huggingface.co/datasets/not-lain/wikipedia) surpassing 1k downloads, and a super-fast demo of [Mistral-7B](https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat). Additionally, check out an informative [YouTube video](https://www.youtube.com/watch?v=S-gy6NUOGSs) on Agentic AI solutions.
- **Create Transparent Images and More**: Try creating [transparent images](https://github.com/rootonchair/diffuser_layerdiffuse) using Diffusers, and learn about instruction-tuned models through this [instruction tuned model explanation video](https://youtu.be/jddSbTLw0gc). Other highlights include links to training MoE on AWS Trainium, training custom AI models, and interesting findings from AnthropicAI's research.
- **Virtual Try-On Using IP-Adapter Inpainting**: Explore the new virtual [Try-On](https://huggingface.co/blog/tonyassi/virtual-try-on-ip-adapter) experience using inpainting. This innovation allows users to experiment with virtual clothing using advanced AI techniques.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=S-gy6NUOGSs)">Agentic AI Solutions / Adaptive AI Solutions - Episode 1:  CrewAI With Preston McCauley</a>: In Episode 1, we explore a brief introduction to #AdaptiveAI and #Agentic AI approaches.https://www.linkedin.com/in/preston-mccauley-immersive-ux/Join Presto...</li><li><a href="https://youtu.be/jddSbTLw0gc)">What is an Instruction Tuned Model?</a>: What is Instruction Tuning?  What are Instruction Tuned models? What is a Pretrained Model? How can I make my Large Language Model follow Instructions?These ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1242917952727289966)** (591 messages🔥🔥🔥): 

- **Confusion Over Copilot+ Capabilities**: Members debated the capabilities of **Copilot+**, questioning if its features like local model usage truly eliminate the need for constant cloud connection. One remarked, *"copilot+ means you got enough NPU power to run all the local models like Snapdragon X Elite."*

- **Optimizing Models for Chatbot Use**: Discussion flowed on selecting models specifically for chatbot functionalities without relying on MoE. A member mentioned, *"Zephyr 7B from HuggingFace team is a great fine-tune from Mistral 7B."* They emphasized practical testing on personal use cases over benchmarks.

- **Concerns About Model Training and Fine-Tuning**: Queries around fine-tuning models with minimal data and epochs were prevalent. A conversation centered on using 80 messages to train **Mistral** to believe it was a 25yo, suggesting small data can still be effective.

- **Debates on Practical Use of Uncensored Models**: Members highlighted uncensored models' capacity beyond ERP (erotic roleplay), stating their broader conversational and role-playing utilities. One member jested, *"fun, learning, and channeling my horniness,"* but clarified they use uncensored models for varied entertaining interactions.

- **Technical Issues and Community Assistance**: Various members requested help with technical hurdles, such as handling runtime errors in their projects and formatting data correctly for fine-tuning on AutoTrain. A shared link outlined, [HuggingFace documentation](https://hf.co/docs/autotrain) for guidance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/AfCdAWnE)">Discord | Your Place to Talk and Hang Out</a>: Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.</li><li><a href="https://tenor.com/view/mewing-locked-in-funny-gif-2909757877821689206">Mewing Locked In GIF - Mewing Locked in Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/better-call-saul-gif-26547310">Better Call Saul GIF - Better Call Saul - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/kurt-angle-100-yard-stare-kurt-angle-stare-gif-2618405429234636640">Kurt Angle 100 Yard Stare GIF - Kurt angle 100 yard stare Kurt Angle Stare - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>: no description found</li><li><a href="https://pauseai.info/pdoom">List of p(doom) values</a>: How likely do AI various researchers believe AI will cause human extinction?</li><li><a href="https://tenor.com/view/que-gif-27530657">Que GIF - Que - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://open.spotify.com/track/6zFy3b8t6x0WDk5e5xaWzP?si=9c7c70c2dadd4be6">Mmmm (Mmmm, Mmmm)</a>: Underbelly · Song · 2023</li><li><a href="https://github.com/nroggendorff/level">GitHub - nroggendorff/level: A simple way to level up in a server without getting rate-limited</a>: A simple way to level up in a server without getting rate-limited - nroggendorff/level</li><li><a href="https://tenor.com/view/better-call-saul-james-morgan-mcgill-slippin-jimmy-bob-odenkirk-mugshot-gif-18613260">Better Call Saul James Morgan Mcgill GIF - Better Call Saul James Morgan Mcgill Slippin Jimmy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.heygen.com/">HeyGen - AI Video Generator</a>: HeyGen is an innovative video platform that harnesses the power of generative AI to streamline your video creation process. Unleash your creativity with HeyGen - the future of video production.</li><li><a href="https://huggingface.co/docs/transformers/main/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://huggingface.co/datasets/nroggendorff/mayo">nroggendorff/mayo · Datasets at Hugging Face</a>: no description found</li><li><a href="https://rentry.co/ayumi_erp_rating">Ayumi's LLM Role Play & ERP Ranking (Version 3)</a>: This ranking table contains a rating of different LLMs, which tries to determine which model is most suitable for (erotic) role playing (ERP) by using an automated benchmark. Unfortunately this automa...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15388d6/llama_2_pffft_boundaries_ethics_dont_be_silly/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://hf.co/docs/autotrain">What is AutoTrain Advanced?</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1242934142124036237)** (3 messages): 

- **Seeking collaboration on projects**: A member asked if anyone was interested in connecting to work on some projects together. Another member suggested posting the proposal in <#1204742843969708053> to find like-minded individuals.
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1243069203410587720)** (5 messages): 

- **Vision Transformer (ViT) Shares Detailed Author Links**: A member shared a [link](https://arxiv.org/abs/2010.11929) to the Vision Transformer paper, highlighting that it has contributions from [multiple authors](https://arxiv.org/search/cs?searchtype=author&query=Dosovitskiy,+A), showcasing a collaborative effort.
- **Training Complex RL Tasks with Human Preferences**: A member posted a [link](https://arxiv.org/abs/1706.03741) discussing reinforcement learning systems that leverage human preferences between trajectory segments. The approach shows significant efficiency, reducing human oversight to less than one percent while still solving complex tasks like Atari games.
- **RAG with Source Highlighting Gets a Medium Post**: An article titled [RAG with Source Highlighting using Structured Generation](https://medium.com/ai-advances/rag-with-source-highlighting-using-structured-generation-d30492ed23e1) was shared, offering insights into advanced techniques in retrieval-augmented generation.
- **New Paper on Code Optimization Techniques**: A [link](https://arxiv.org/abs/2312.05657) to a paper on code optimization techniques was shared, authored by [Shukai Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan,+S) and team. The paper explores innovative methods for improving code performance.
- **Access Latest Research on ArXiv**: A direct [PDF link](https://arxiv.org/pdf/2307.06435) to a recent research paper was provided, ensuring quick access to new findings in the field.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>: While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in...</li><li><a href="https://arxiv.org/abs/1706.03741">Deep reinforcement learning from human preferences</a>: For sophisticated reinforcement learning (RL) systems to interact usefully with real-world environments, we need to communicate complex goals to these systems. In this work, we explore goals defined i...</li><li><a href="https://arxiv.org/abs/2312.05657">Leveraging Reinforcement Learning and Large Language Models for Code Optimization</a>: Code optimization is a daunting task that requires a significant level of expertise from experienced programmers. This level of expertise is not sufficient when compared to the rapid development of ne...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1242997798756942008)** (14 messages🔥): 

```html
- **Excitement builds for open-source protein folding**: A member introduced [ProteinViz](https://huggingface.co/spaces/as-cle-bert/proteinviz), an open-source alternative to AlphaFold3, enabling users to predict protein 3D structures. They also shared a [community blog post](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3) exploring the advancements of AlphaFold3.
  
- **Mistral-7B v0.3 demo impresses**: The ultra-fast [Mistral-7B v0.3 demo](https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat) was shared, showcasing its capabilities. Users were encouraged to try it out and provide feedback.

- **LayerDiffusion method enabling transparent images**: A user shared [diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse), a method for generating transparent images from any base model. This technique promises high foreground separation accuracy.

- **SimpleTuner v0.9.6 released**: The latest release of [SimpleTuner](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6) includes a new randomized aspect bucket feature and custom resolution mapping configs. Users are urged to check out the newer functionalities.

- **Miniature dataset success**: A member celebrated their dataset reaching 1K downloads, highlighting it as part of a blogpost on RAG applications. This miniature dataset contains 3K samples and has gained traction despite the general preference for more visually engaging demos.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat">Mistral-7B-v0.3 Fast Chat - a Hugging Face Space by ehristoforu</a>: no description found</li><li><a href="https://huggingface.co/posts/as-cle-bert/598312932414376">@as-cle-bert on Hugging Face: &quot;Hi HF Community!🤗

If you are excited about AlphaFold3, but upset because it…&quot;</a>: no description found</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6">Release v0.9.6 - debias them buckets · bghira/SimpleTuner</a>: debiased aspect bucketing When training on large datasets of heterogenous samples, you will discover a content bias among aspect ratios - vertical images contain portraits, widescreen shots are cin...</li><li><a href="https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt">What is an Instruction Tuned Model?</a>: What is Instruction Tuning?  What are Instruction Tuned models? What is a Pretrained Model? How can I make my Large Language Model follow Instructions?These ...</li><li><a href="https://github.com/rootonchair/diffuser_layerdiffuse">GitHub - rootonchair/diffuser_layerdiffuse: Create transparent image with Diffusers!</a>: Create transparent image with Diffusers! Contribute to rootonchair/diffuser_layerdiffuse development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/as-cle-bert/proteinviz">Proteinviz - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://github.com/AstraBert/proteinviz">GitHub - AstraBert/proteinviz: Your open-source alternative to AlphaFold3🚀</a>: Your open-source alternative to AlphaFold3🚀. Contribute to AstraBert/proteinviz development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3">What is going on with AlphaFold3?</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1243042796492029963)** (11 messages🔥): 

- **Seeking RAG Systems Resources**: A member asked for recommendations on learning materials to build RAG (Retrieval-Augmented Generation) systems in production.

- **CUDA OOM Errors During Fine-Tuning**: A member encountered CUDA OOM errors while performing SFT with a Mixtral model on 8xH100 GPUs, using the official TRL script. After reducing the tensor precision to `bfloat16` and re-merging adapters, the issue was resolved.

- **Clarification on `bf16` Argument**: There was some confusion about the `bf16` argument's impact, but it was eventually understood to affect tensor precision and help in fine-tuning processes involving quantized weights.

- **Uploading Large Datasets Issues**: A member faced issues while pushing a large dataset (60 million rows) to the hub, taking a long time to create parquet files from arrow format and experiencing HTTP request failures. They inquired about avoiding the parquet format to solve this.

- **Adapter Merging and Hub Upload Method**: The solution to the CUDA OOM errors involved re-merging the adapters and uploading the fine-tuned model to the hub with the [`torch_dtype=torch.bfloat16`](https://pytorch.org/docs/stable/generated/torch.bfloat16.html) argument. This process improves memory efficiency and resolves the initial issue.
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1242978606230143098)** (5 messages): 

- **AI for User Manual Queries Possible?**: A member inquired about the feasibility of using AI to query a 1000-page software manual. They were interested in knowing if an AI could be fed a PDF or other format and provide answers about the software's usage.

- **Stable Diffusion XL API Ignores Parameters**: A member reported issues with the inference API for `stabilityai/stable-diffusion-xl-base-1.0`, noting it ignored everything but the prompt. They provided their payload as an example with details like `negative_prompt` and `guidance_scale`.

- **Incorrect Payload Syntax**: Another member quickly pointed out a syntax error in the payload, specifically the use of `=` instead of `:` in the parameters.

- **Seeking Resources for NLG**: A member asked for recommendations on where to learn about **Natural Language Generation (NLG)**. There were no further details or responses provided.

- **Documentation on Custom Dataset Training for SD**: Another member asked about official documentation for training **Stable Diffusion** on a custom dataset, such as generating MNIST images. They mentioned finding some resources but noted the examples they found were for unconditional generation.
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1243136226395951104)** (3 messages): 

- **Problems with flash attention implementation in YaRN**: A member mentioned Bowen, indicating that there's an ongoing struggle to get a **flash attention** implementation to work with **YaRN**. Another member acknowledged the issue, noting it is somewhat correct but not exactly that.
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1242936316191182899)** (117 messages🔥🔥): 

- **Food bucket storage hacks**: Members discussed using food buckets for storing food, reheating methods, and the practicality of plating meals like plov from buckets. One remarked, "Please update with plated plov at your leisure."
  
- **Programming language trends and preferences**: The conversation flowed around choosing programming languages for machine learning, with specific mentions of Rust, Python, and Mojo. One user humorously noted, "I get started on hello world, get hyped, see new language on trend," highlighting their frequent shifts between languages.

- **Using Rust for machine learning**: Rust was highlighted as a preferred language for some users over Python due to its technical capabilities. Shared resources included the [Rust-CUDA GitHub project](https://github.com/Rust-GPU/Rust-CUDA) and [rustml - Rust](https://github.com/daniel-e/rustml) for machine learning.

- **Interest in Quantum Computing**: A discussion on the future impact of quantum computing touched on Nvidia's investments in the field, citing their recent [NVIDIA CUDA-Q platform announcement](https://nvidianews.nvidia.com/news/nvidia-accelerates-quantum-computing-centers-worldwide-with-cuda-q-platform). A realist perspective was provided: "Much like fusion, quantum computing has been just a few years away for like 10 years now."

- **Mixed hardware performance experiences**: Members shared their experiences with different hardware for ML tasks. Contrasts were made between using Nvidia GPUs for speed and potential fine-tuning capabilities with Intel GPUs, with specific references to the [ipex-llm project](https://github.com/intel-analytics/ipex-llm).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/cppdocs/">PyTorch C++ API &mdash; PyTorch main documentation</a>: no description found</li><li><a href="https://www.udio.com/songs/phmruKKXXdSaUc91WrkL8D">Amirthetarbosaurus - Eternal Lament | Udio</a>: Listen to Eternal Lament by Amirthetarbosaurus on Udio. Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.</li><li><a href="https://github.com/Rust-GPU/Rust-CUDA">GitHub - Rust-GPU/Rust-CUDA: Ecosystem of libraries and tools for writing and executing fast GPU code fully in Rust.</a>: Ecosystem of libraries and tools for writing and executing fast GPU code fully in Rust. - Rust-GPU/Rust-CUDA</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-accelerates-quantum-computing-centers-worldwide-with-cuda-q-platform">NVIDIA Accelerates Quantum Computing Centers Worldwide With CUDA-Q Platform</a>: NVIDIA today announced that it will accelerate quantum computing efforts at national supercomputing centers around the world with the open-source NVIDIA CUDA-Q™ platform.</li><li><a href="https://github.com/daniel-e/rustml">GitHub - daniel-e/rustml: Machine learning in Rust.</a>: Machine learning in Rust. Contribute to daniel-e/rustml development by creating an account on GitHub.</li><li><a href="https://docs.rs/rustml">rustml - Rust</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/pull/2790">[ROCm] Fix build problem resulted from previous commit related to FP8 kv-cache support  by hongxiayang · Pull Request #2790 · vllm-project/vllm</a>: Fixes: #2725 Current head failed to build on ROCm, and I got errors like: g++ -pthread -B /opt/conda/envs/py_3.8/compiler_compat -Wl,--sysroot=/ -pthread -shared -B /opt/conda/envs/py_3.8/compiler_...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1243197933143199784)** (1 messages): 

- **Nous Research is hiring**: Nous Research is seeking new team members and invites applications through a [Google Form](https://forms.gle/UWx2Pht8qioi1bjAA). The announcement emphasizes their recruitment efforts on X, as seen in the tweet: *Nous Research is hiring! Apply Here: https://forms.gle/UWx2Pht8qioi1bjAA*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/UWx2Pht8qioi1bjAA">no title found</a>: no description found</li><li><a href="https://x.com/nousresearch/status/1793637803701780797?s=46">Tweet from Nous Research (@NousResearch)</a>: Nous Research is hiring!  Apply Here: https://forms.gle/UWx2Pht8qioi1bjAA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1242923003835056199)** (302 messages🔥🔥): 

- **AI Security Learning and Career Path Discussed**: Members debated whether Python is essential for a career in AI, with some suggesting it is fundamental, while others advocated for Rust or Go. One user highlighted Yann LeCun's skepticism toward relying solely on LLMs, urging exploration of next-gen AI systems ([Yann's Tweet](https://x.com/ylecun/status/1793326904692428907)).
  
- **Traditional vs. Simplified Chinese in GPT Models**: Members discussed the challenges of adapting GPT models for different Chinese dialects, focusing on traditional versus simplified Chinese. A blog post on the differences between these character sets was shared ([Glossika Blog](https://ai.glossika.com/blog/differences-between-traditional-chinese-and-simplified-chinese)).

- **Multilingual Datasets and Model Training Insights**: The conversation included references to synthetic datasets and available resources such as Tagengo, a large multilingual chat dataset, and other LLM training tools. One member pointed to a Vietnamese-specific LLM, VinaLLaMA, designed to handle linguistic and cultural nuances ([VinaLLaMA Paper](https://arxiv.org/abs/2312.11011)).

- **New Tools and Developments**: The community highlighted newly available resources like the PyTorchModelHubMixin, which simplifies model integration with the Hugging Face hub. The versatility of the tool was emphasized, though it currently has a 50GB limit on model size.

- **AI Gaming and Challenges**: Several users engaged in discussions about a prompt engineering game, sharing strategies for advancing through its levels. Helpful approaches and code snippets were exchanged to solve various stages of the challenge ([Lakera AI Game](https://gandalf.lakera.ai/)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/q2y38cqjNw">Join the StableSwarmUI Discord Server!</a>: StableSwarmUI ( https://github.com/Stability-AI/StableSwarmUI ) official Discord. | 38 members</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.10808">ActiveLLM: Large Language Model-based Active Learning for Textual Few-Shot Scenarios</a>: Active learning is designed to minimize annotation efforts by prioritizing instances that most enhance learning. However, many active learning strategies struggle with a &#39;cold start&#39; problem, ...</li><li><a href="https://ai.glossika.com/blog/differences-between-traditional-chinese-and-simplified-chinese">Differences Between Traditional Chinese and Simplified Chinese | The Glossika Blog</a>: In this article, we’ll explore a few of the major differences between these two Chinese writing systems and give you some ideas about how to decide which is right for you!</li><li><a href="https://meta.wikimedia.org/wiki/Automatic_conversion_between_simplified_and_traditional_Chinese#Background>">Automatic conversion between simplified and traditional Chinese - Meta</a>: no description found</li><li><a href="https://x.com/vanstriendaniel/status/1793564151463510038">Tweet from Daniel van Strien (@vanstriendaniel)</a>: Tagengo - the world&#39;s largest multilingual chat dataset? - Consists of conversations in 74 languages - OpenAI moderated messages removed - Klingon language removed from the dataset!  https://huggi...</li><li><a href="https://arxiv.org/abs/2312.11011">VinaLLaMA: LLaMA-based Vietnamese Foundation Model</a>: In this technical report, we present VinaLLaMA, an open-weight, state-of-the-art (SOTA) Large Language Model for the Vietnamese language, built upon LLaMA-2 with an additional 800 billion trained toke...</li><li><a href="https://x.com/ylecun/status/1793326904692428907">Tweet from Yann LeCun (@ylecun)</a>: If you are a student interested in building the next generation of AI systems, don&#39;t work on LLMs  Quoting Viva Technology (@VivaTech)   The Godfather of AI is at #VivaTech! Yann LeCun (@ylecun) a...</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/discussions/8">mistralai/Mistral-7B-Instruct-v0.3 · String for the Function Calling Token</a>: no description found</li><li><a href="https://gandalf.lakera.ai/">Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.</a>: Trick Gandalf into revealing information and experience the limitations of large language models firsthand.</li><li><a href="https://huggingface.co/datasets/N8Programs/Capybara-Quicksilver-1K">N8Programs/Capybara-Quicksilver-1K · Datasets at Hugging Face</a>: no description found</li><li><a href="https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-compatible-server">Quickstart &#8212; vLLM</a>: no description found</li><li><a href="https://github.com/argilla-io/distilabel">GitHub - argilla-io/distilabel: ⚗️ distilabel is a framework for synthetic data and AI feedback for AI engineers that require high-quality outputs, full data ownership, and overall efficiency.</a>: ⚗️ distilabel is a framework for synthetic data and AI feedback for AI engineers that require high-quality outputs, full data ownership, and overall efficiency. - argilla-io/distilabel
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1243199060152811561)** (6 messages): 

- **Anthropic Investigates Increased Error Rates on Opus**: An **Anthropic** status update was discussed regarding increased error rates on their **Opus** system. The incident was reported at 06:35 PDT and resolved by 07:16 PDT ([source](https://stspg.io/x564hq8qxtz9)).

- **Adding Tool Role Success on Apple Silicon**: A member questioned if anyone successfully added a tool role to any framework on Apple Silicon, specifically for **Hermes Pro** tool calling. 

- **Script for Llama.cpp Handling Function Calls**: Another update shared success in creating a script using **llama.cpp** that handles function calls and returns model-based answers based on tool responses.

- **Hermes Pro 2 GitHub Repo as Inspiration**: The same member mentioned using the **Hermes Pro 2 GitHub repo** for inspiration and offered to create a PR to add a notebook if anyone needs it.

- **High Praise for the Model**: They concluded with high praise for the model, calling it *"a beast"*.

**Link mentioned**: <a href="https://stspg.io/x564hq8qxtz9.">Increase error rates on Opus</a>: no description found

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1243214575424372838)** (7 messages): 

- **Explore RAG with Sample Wikipedia Dataset**: A member shared a miniature dataset on [Huggingface](https://huggingface.co/datasets/not-lain/wikipedia) for those wanting to experiment with RAG. The dataset contains various texts, including an excerpt about anarchism.
- **Multilingual Dataset Contribution**: Another member introduced a dataset in similar format with 16 languages, contributed to MTEB's embedding leaderboard. The dataset is available on [Huggingface](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-corpus) along with corresponding multilingual queries.
- **Model Context Enhancement Suggested**: A member proposed that LLMs should either add context from their own knowledge or override RAG'ed information if it contradicts their own knowledge. They emphasized the importance of this approach by referencing a conversation about Google's AI drawing conclusions from an outdated Reddit post.
- **RAG with Fine-tuned Models**: Another member responded, concurring that using RAG with a fine-tuned model could address the concerns raised about contextual accuracy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/dat">Dat (Dat Nguyen)</a>: no description found</li><li><a href="https://x.com/pixelbutts/status/1793387357753999656?s=46">Tweet from PixelButts (@PixelButts)</a>: Google is dead beyond comparison</li><li><a href="https://x.com/kurtopsahl/status/1793494822436917295?s=46">Tweet from Kurt Opsahl @kurt@mstdn.social (@kurtopsahl)</a>: Seems the origin of the Google AI’s conclusion was an 11 year old Reddit post by the eminent scholar, fucksmith.  Quoting PixelButts (@PixelButts)   Google is dead beyond comparison</li><li><a href="https://huggingface.co/datasets/not-lain/wikipedia">not-lain/wikipedia · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">RAG chatbot using llama3</a>: no description found</li><li><a href="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-corpus">ellamind/wikipedia-2023-11-retrieval-multilingual-corpus · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries">ellamind/wikipedia-2023-11-retrieval-multilingual-queries · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1243311182530613328)** (3 messages): 

- **Jam session video faces upload issues**: The jam session video has been recorded but there are difficulties uploading it to YouTube. The uploader promised to inform everyone as soon as it's available.

- **Inquiry about DALL-E 3**: A question was posed asking whether the images being referred to are created with **DALL-E 3**. No further context was provided.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1242916749708365834)** (360 messages🔥🔥): 

- **Emad to Drop Weights in Two Weeks?:** Several members joked about the imminent release of Stable Diffusion weights, referencing *Star Wars* and speculating on Emad's next move. One comment noted, "Drop the weights will."
  
- **Issues with Blurred Images in Stable Diffusion 3:** One user reported repeated issues with blurry outputs when generating female characters via the API, prompting discussions about prompt adjustments and potential censorship triggers. Another noted, "Removing 'woman' from the prompt significantly reduced the blurring issue."

- **Laptop GPUs and New AI Hardware Rumors:**
  - Discussions covered the specifications and performance of ASUS AI laptops and rumored NVIDIA 5090 GPUs, with some skepticism about the details.
  - A commenter referenced a [PC Games Hardware article](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/) discussing the potential 5090 specs.

- **Prefer MidJourney or Stable Diffusion?:** A brief debate occurred over which AI tool was superior, with members suggesting trying SD3 for free and noting, "I think MJ still wins in most of the cases."

- **Local Installation vs. Web Services for Stable Diffusion:** Users discussed the pros and cons of local installations, especially with AMD GPUs, and the use of web services. One member suggested, "If you have a good graphics card, install Stable Diffusion; otherwise, use web services."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post">Stability AI - Developer Platform</a>: no description found</li><li><a href="https://glif.app/@Oliveira/glifs/clw44qfbl0000m0zztwqk2tnf">glif - StableDiffusion 3 + GPT4 Helper + SDXL 1.5x Upscale (CopyGenius) by Yuri Oliveira COPYGENIUS </a>: no description found</li><li><a href="https://tenor.com/view/never-finn-adventure-time-gif-10874543">Never Finn GIF - Never Finn Adventure Time - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.asus.com/content/asus-ai-pc/">Next Level. AI Incredible | ASUS Launch Event | ASUS</a>: We are thrilled to be unveiling our latest product, packed with new AI experiences. Mark your calendars for May 20th at 11:00 AM (PT) and join our livestream.</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/">Geforce RTX 5090 soll mit 32 GiB GDDR7 und gleich drei PCBs an den Start gehen [Gerücht]</a>: Bilder zu Artikel: Geforce RTX 5090 soll mit 32 GiB GDDR7 und gleich drei PCBs an den Start gehen [Gerücht] - Geforce RTX 5090</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-d">News zu Grafikkarten</a>: Sie finden hier immer die besten News zu Grafikkarten
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1242917436169130014)** (152 messages🔥🔥): 

```html
<ul>
    <li><strong>Llama 3 8k context performance complaints:</strong> "Oh yeah now looking at the Llama 3 models. 8k context sucks." There are discussions on Llama models with higher context lengths up to 1M.</li>
    <li><strong>Idefics 2.0 multimodal model inquiries:</strong> Users asked if the Idefics2 model from HuggingFace supports LM Studio. It was noted that idefics models don't work in llama.cpp, but other vision models like LLaVA are supported.</li>
    <li><strong>Query on Context Length affecting Performance:</strong> A member asked if increased context size (e.g., 8k, 16k) makes models slower, to which it was confirmed that larger context sizes do indeed slow down performance.</li>
    <li><strong>ONNX Runtime and GPU driver improvements:</strong> Discussion about new NVIDIA driver updates improving model inference speeds. "Just updated the drivers. I had to reboot because even if they were installed, it kept saying it was using the old ones."</li>
    <li><strong>Helpful LM Studio resources and usage:</strong> Members shared links to tutorials and resources such as a YouTube video on running LM Studio locally. "Explore LM Studio's new CLI tool, lms, in this video."</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blogs.nvidia.com/blog/rtx-advanced-ai-windows-pc-build/">New Performance Optimizations Supercharge NVIDIA RTX AI PCs for Gamers, Creators and Developers</a>: The latest AI performance gains and features for RTX AI PCs unveiled at Microsoft Build.</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://onnxruntime.ai/blogs/accelerating-phi-2#:~:text=We%20also%20observe%20ONNX%20Runtime,the%20first%20256%20tokens%20generated">Accelerating Phi-2, CodeLlama, Gemma and other Gen AI models with ONNX Runtime</a>: Improvements with ONNX Runtime for inferencing popular Gen AI models</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=1LdrF0xKnjc">LM Studio: How to Run a Local Inference Server-with Python code-Part 1</a>: Tutorial on how to use LM Studio without the Chat UI using a local server.  Deploy an open source LLM on LM Studio on your pc or mac without an internet conn...</li><li><a href="https://youtu.be/rgqcrsW-_aM">The Ultimate Guide to LM Studio CLI LMS</a>: Explore LM Studio&#39;s new CLI tool, lms, in this video. Learn how to load and unload models, start and stop the API server, and inspect raw LLM input. Discover...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1242918472896483350)** (45 messages🔥): 

- **Mistral-7B-Instruct v0.3 sparks interest**: A user shared the [Model Card for Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), highlighting its extended vocabulary, v3 Tokenizer support, and function calling capabilities. **Installation** and **download instructions** were also provided.
- **Financial domain model request**: A user sought recommendations for models focusing on **information synthesis and fact extraction**, particularly in financial planning. The community did not offer any direct recommendations in response.
- **Vision Language Models (VLM) inquiry**: There was a discussion about VLM and vision adapters, with a specific question about using **VLM with the web server API**. Users were directed to check the server tab for vision assistant Python code examples, and noted the inclusion of an *"apikey"* property in the Body schema.
- **OCR model recommendations**: Users discussed the reliability of vision models for OCR, particularly noting issues with LLAVA 1.6 and hallucination instead of accurate results. **Tesseract** was recommended as a reliable option for extracting text before feeding it to a language model.
- **Cohere Aya 23 model announcement**: The community discussed the release of the [Aya 23 8B model by Cohere For AI](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF) and its capabilities. There were technical challenges noted related to its compatibility with LM Studio and llama.cpp patches.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/aya-23-8B-GGUF">lmstudio-community/aya-23-8B-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1243263078024482898)** (24 messages🔥): 

- **Llama3 struggles with JSON fixing tasks**: A member encountered issues with Llama3's ability to fix malformed JSON using langchains OutputFixingParser, as well as with Phi-3 mini and medium models, noting "*Phi-3 performs even worse*". They requested advice from others who might have "similar experiences and has a hint?".
  
- **Adjusting prompts for better translation and formatting**: A member shared difficulties with Llama3 Instruct for translating and formatting product tables in Romanian, stating the model sometimes includes unwanted opinions despite specific instructions to avoid them. Another suggested adding phrases like "No Yapping" and "Be Concise" to the prompt to enforce brevity.
  
- **Capitalization doesn’t impact model instructions consistently**: A member queried whether LLMs generally recognize capitalization in instructions such as "do **not** use brand names" versus "do **NOT** use brand names." Another member confirmed it "*varies from model to model*", but generally, LLMs "don’t follow capitalized words on order of importance".
  
- **German grammar and punctuation models lack precision**: Another topic discussed is the reworking of German texts, where a member experienced struggles with punctuation and grammatical accuracy even after providing comma positioning rules. They were recommended to try a multilingual model like Aya 23 8B from Cohere For AI, which is specialized in such tasks.
  
- **Aya 23 8B launched for multilingual tasks**: Aya 23 8B by Cohere For AI was highlighted as a model performing well across a broad range of languages, including German grammar correction tasks. The model is featured for its multilingual and logic task proficiency on the [Hugging Face](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF) platform.

**Link mentioned**: <a href="https://huggingface.co/lmstudio-community/aya-23-8B-GGUF">lmstudio-community/aya-23-8B-GGUF · Hugging Face</a>: no description found

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1242939970155450478)** (3 messages): 

- **Dual Graphics Cards Compatibility Question**: A member joined the server to ask if **LM Studio** supports using 2 graphics cards. They later noted the amusing coincidence of finding the answer right here promptly after asking.
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1243119449855496274)** (1 messages): 

- **User resolves storage issue with M.2 SSD**: A member fixed their issue by installing an additional M.2 SSD. This allowed them to **dedicate the SSD to LM Studio** and save all their models on it.
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1243023806180757585)** (7 messages): 

- **Members upgrade to 7900xt**: Members announced their upgrades to **7900xt** graphics cards. Proteanx said, “*just upgraded to 7900xt*,” while lt4453 mentioned, “*Got my 7900xt ready as well*.”
- **Added to testing channel**: Yagilb added users with new cards to a specific channel for coordination. He mentioned, “*Added you to <#1242213172199559319>*.”
- **Environment setup advice for RX 6600 on Fedora**: Core_silicon_45873 asked about compatibility with an **RX 6600** on Fedora. Nettoneko advised setting environment variables to treat the card as gfx1030 for optimal functionality: “*you'll need to set env vars to treat your card as gfx1030*.”
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1242927323733430312)** (2 messages): 

- **Mistral 7B Instruct v0.3 Released**: The Mistral v0.3 instruct model is now available for the community. Check it out on the [lmstudio community Huggingface page](https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF).

- **Cohere Models Now in 23 Languages**: Aya-23 quants are now available for download, featuring data from 23 different languages including Arabic, Chinese, and Turkish. Access them on the [lmstudio-community page](https://huggingface.co/lmstudio-community/aya-23-35B-GGUF) and [here](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF).
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1242920034137931876)** (2 messages): 

- **Call for GitHub Issue Channel**: A member suggested that the server should include a channel specifically for **GitHub issues**. This would help in organizing and tracking issues more effectively within the community.
- **Inquiry about Custom Decorators**: Another member inquired if **custom decorators** are on the current development timeline. The question implies a need for this feature in an upcoming release.
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1793427278564917459>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1243276465831805010)** (1 messages): 

- **Modular posts new video**: **Modular** just released a [new video](https://www.youtube.com/watch?v=uIG9q9foIw0) titled *"Mojo Community Meeting #1"*. Check out the [public agenda](https://modul.ar/community-meeting-doc) for more details.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=uIG9q9foIw0">Mojo Community Meeting #1</a>: Mojo Community Meeting Public Agenda: https://modul.ar/community-meeting-doc

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1243256964579065876)** (1 messages): 

- **Mojo requires understanding system concepts**: Despite Mojo aiming to be a proper superset of Python, there's still a lot of system concepts and models to be learned. One example given was the **ownership model**, which could benefit those wanting to delve deeper into Mojo.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1242921193594556586)** (65 messages🔥🔥): 

- **Confusion about Deprecated Documentation Notation**: A user expressed confusion over the use of `**` in documentation, mistaking it for valid code syntax, which was confirmed as a formatting error by another user. They humorously noted that the chatbot provided mixed guidance on this issue, highlighting a need for clarity.
- **Discussion on Tensor Splitting and MLIR Integration**: Members debated splitting Tensor libraries for numerical computing versus AI/ML use cases, with suggestions of having compatibility functions between them. Concerns were raised about the status of MLIR integration, with clarifications that broader improvements like package management and Windows support were prioritized.
- **Exploring @value Decorator for Structs**: A user seeking to make structs flexible in Mojo learned about using the `@value` decorator to generate boilerplate lifecycle methods. This discussion included recommendations for decorators and references to the [Mojo documentation](https://docs.modular.com/mojo/manual/decorators/value) for further details.
- **Interest in Smaller Bit-size Integers and Custom MLIR Dialects**: A member expressed interest in custom bit-size integers for more memory-efficient computations, with others pointing out the feasibility of implementing this by examining DType source code. Additionally, there was interest in using [MLIR dialects](https://github.com/modularml/mojo) beyond the built-ins, emphasizing the importance of MLIR integration for heterogeneous hardware.
- **Inquiries about FFT Implementation in Mojo**: A member asked about performing FFTs in Mojo, exploring options like using Scipy's FFT functions or wrapping FFTW via the FFI module. They sought clarity on passing Mojo objects like Tensors to these functions, underscoring the need for practical examples in the documentation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/decorators/value">@value | Modular Docs</a>: Generates boilerplate lifecycle methods for a struct.</li><li><a href="https://ivellapillil.github.io/mojo/">Learn Mojo Programming Language</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues/2381">[Feature Request] Int64, Int32, Int16 constructor support multiple smaller IntX as input · Issue #2381 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? It would be helpful if Int64, Int32 and Int16 had cons...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 35
https://www.modular.com/newsletters/modverse-weekly-35
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1242918560649580725)** (120 messages🔥🔥): 

- **Mojo upgrades cause build issues**: Users discussed challenges after upgrading to Mojo nightly `2024.5.2305`, experiencing errors such as "'invalid call to 'printf'" and ambiguous type conversion issues, which were resolved by casting to `Float64` explicitly. One user discovered and shared a fix for using `printf["%d %d %d\n"](l[0], l[1], l[2])` instead of `printf("%d %d %d\n", l[0], l[1], l[2])`.
- **String representation debate stirs conversation**: Members discussed the implications of null-terminated vs non-null-terminated strings, performance pitfalls, and methods to maintain consistency and prevent bugs. A proposal to better handle null terminators in Mojo was mentioned, referencing an existing GitHub issue on the subject.
- **Inferred parameters and syntax changes noted**: Users noted that inferred parameters in Mojo now use `//` instead of the `inferred` keyword, expressing mixed feelings but recognizing the increased conciseness. This change prompted discussions on structural guarantees and syntax preferences.
- **Exploring f-string alternatives in Mojo**: There was interest in contributing to f-string support, with suggestions to start with a `Formatable` trait to handle formatting. Though a substantial build-out is necessary, initial proposals were discussed to set a foundation.
- **Runtime error problem-solving**: One user faced a runtime error related to type rebind operations, resolved by another user identifying a mismatch in `Tensor` data types. This collaborative debugging highlighted the importance of accurate type handling in Mojo's evolving ecosystem.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/hex/hex">hex | Modular Docs</a>: hexT: Intable -&gt; String</li><li><a href="https://peps.python.org/pep-0686/">PEP 686 – Make UTF-8 mode default | peps.python.org</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=kPR8h4-qZdk&t=1150s">CppCon 2016: Nicholas Ormrod “The strange details of std::string at Facebook&quot;</a>: http://CppCon.org—Presentation Slides, PDFs, Source Code and other presenter materials are available at: https://github.com/cppcon/cppcon2016—Standard string...</li><li><a href="https://github.com/modularml/mojo/issues/2678#issue-2300567975">[Feature Request] Better handling of null terminator in strings · Issue #2678 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I would like a discussion to answer the following ques...</li><li><a href="https://github.com/WebAssembly/webidl-bindings/issues/38">Performance concerns about UTF-8 strings · Issue #38 · WebAssembly/interface-types</a>: I&#39;ve written some Rust web apps which are compiled to WebAssembly and run in the browser. I&#39;m using wasm-bindgen for this. Generally the code runs really fast (since both Rust and WebAssembly ...
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1242915178668888094)** (69 messages🔥🔥): 

- **Epoch's frontier model training costs report**: Robirahman shared that Ben and he are releasing a report on *frontier model training costs* and asked for reviewers from the community. Discussions centered around calculating costs based on *GPU-hours* and specific GPU types, like A100 40GB, to derive accurate estimates.
- **Compute cost for Pythia training**: Stellaathena provided detailed estimates, noting it would have cost about *$250k* for the largest Pythia model and *$400k* for all combined. The conversation touched on GPU-hour usage, efficiency, and different estimation methods, including committed-use discount prices.
- **Discussion on MFU and efficiency**: Participants discussed *use of activation checkpointing*, its impact on MFU (Memory Footprint Utilization), and the resultant computational efficiency variances across different model sizes in the Pythia suite. They agreed that switching to reporting MFU over HFU might provide better accuracy.
- **Preprints in AI research**: Wonko shared insights about the evolving acceptance of *preprints* in AI research, explaining that while most big journals have normalized it, there still might be specific requirements or restrictions based on the target journal or institution.
- **Flash attention with PyTorch/XLA on TPUv5p**: Yoavhacohen shared their experience, indicating that *flash attention was about 20% faster* than scaled_dot_product_attention for longer sequences when tested on TPUv5p, while it performed similarly for shorter sequences.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1801.06146">Universal Language Model Fine-tuning for Text Classification</a>: Inductive transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific modifications and training from scratch. We propose Universal Language Mode...</li><li><a href="https://www.wolframalpha.com/input?i=%286+FLOP+*+299892736000+*+12+billion%29+%2F+%28312+TFLOPS+*+72300+hours%29">(6 FLOP * 299892736000 * 12 billion) / (312 TFLOPS * 72300 hours) - Wolfram|Alpha</a>: Wolfram|Alpha brings expert-level knowledge and capabilities to the broadest possible range of people—spanning all professions and education levels.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1242921583547383849)** (38 messages🔥): 

- **LeanAttention seeks to outperform FlashAttention**: A member shared [an Arxiv link](https://arxiv.org/abs/2405.10480) to a paper proposing LeanAttention, targeting optimizations beyond the computational phases handled by FlashAttention. Another member commented that it seems like a "marginally better version" of FlashDecoding.

- **"Secret ingredient" controversy in benchmarking**: A conversation revealed humorous remarks on using unauthorized sources for improvement. One member joked, "The secret ingredient is crime," alluding to the unorthodox methods like using libgen to enhance performance on benchmarks such as MMLU.

- **Seeking EMNLP submission tips**: A member asked for advice on submitting work to EMNLP and received feedback that it's a reputable conference for NLP/CL, comparable to ACL and NAACL. This exchange highlights the peer-support aspect of the community.

- **Debate on JEPA and AGI potential**: Members discussed whether JEPA and the ideas in "A Path Towards Autonomous Machine Intelligence" could lead to AGI. Key points of skepticism were the lack of scalability and economically important task solutions compared to LLMs, despite Yann LeCun's advocacy.

- **Concerns about non-AI generated data quality**: In a debate on the future of LLMs, members considered the quality and quantity of non-AI generated data. One member raised concerns about redundancy and processing costs of video data, countered by others citing vast unused data sources and compute limits as the real constraint.

**Link mentioned**: <a href="https://arxiv.org/abs/2405.10480">Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers</a>: Transformer-based models have emerged as one of the most widely used architectures for natural language processing, natural language generation, and image generation. The size of the state-of-the-art ...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1243256735750291566)** (2 messages): 

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Summary of Discord Messages</title>
</head>
<body>
<ul>
    <li><strong>New Paper Sparks Enthusiasm</strong>: A member expressed excitement about a new paper, stating it "opens up a lot of interesting research doors." This sparked curiosity about which specific research areas were of interest.</li>
</ul>
</body>
</html>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1242948332423479336)** (21 messages🔥): 

- **Evaluating Big Models on Multi-Node Clusters**: A user sought advice on running the lm eval harness on a multi-node SLURM cluster. They were guided to use [openai-completions](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py#L76) with ray and vllm but faced challenges due to compute nodes lacking internet access.

- **Counting Tokens in Model Outputs**: Another user asked if the framework could count prompts, tokens, or characters output by the model. They were advised to implement token counting themselves using the logged output samples.

- **Setting Few-Shot Parameters**: A user inquired about passing `num_fewshot` for custom adapter evaluations. They received instructions to customize task configurations via `task_obj.set_config(key="num_fewshot", value=num_fewshot)`.

- **Achieving Reproducibility in Evaluations**: There were discussions about non-deterministic results in evaluations with greedy decoding. Users were advised to set all seed values explicitly, but one user still faced issues despite setting the seed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/5711ab871bbac5426a3a1e958cfe1ba7a6598ea5/lm_eval/evaluator.py#L211-L251>">lm-evaluation-harness/lm_eval/evaluator.py at 5711ab871bbac5426a3a1e958cfe1ba7a6598ea5 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py#L76)">lm-evaluation-harness/lm_eval/models/openai_completions.py at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1242934346730573825)** (59 messages🔥🔥): 

- **Concerns over unauthorized chat activity**: A member reported an unfamiliar chat in Chinese appearing in their account, suggesting a potential account compromise. Another member recommended checking for compromising browser extensions and changing passwords, noting China is not a supported country.
  
- **Anthropic's AI paper fuels curiosities**: A user found [Anthropic's recent mech interp paper](https://arxiv.org/abs/2305.10601) fascinating, especially how prompting Claude Sonnet with internal process queries triggered concepts related to confinement. Discussions emphasized how the AI anthropomorphizes itself, associating its "AI assistant" persona with spiritual beings due to training data.

- **Speculation around Copilot and voice features**: Users discussed the downtime of Microsoft's Copilot, with hopes of new features like GPT-4o or voice capabilities being added soon. Some users complained that the outage severely impacted their tasks, reflecting dependency on the tool.

- **Uncertainty about new AI features' release dates**: Inquiries about the release of new ChatGPT features like live voice chat and video led to referencing [official OpenAI documentation](https://help.openai.com/en/articles/8400625-voice-chat-faq) which stated a rollout would begin soon but become widely available in the coming months. Users expressed eagerness for these upgrades.

- **Question on AI's impact on gaming**: A lighthearted query about how AI would affect Minecraft gameplay went unanswered. Another member humorously noted their anticipation for Copilot voice feature, lamenting the current downtime.
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1242963499249045604)** (37 messages🔥): 

- **Instructions for ChatGPT's Code Interpreter**: Members discussed using the code interpreter for tasks involving math, equations, data analysis, or technical computations, emphasizing the importance of ensuring clear presentation of results with explanations.
- **Creating a GPT Windows App**: A member asked about a GPT Windows app, and the response highlighted creating a web app using Microsoft Edge’s “Install ChatGPT” feature in the side menu. The discussion mentioned that an official Windows app like the Apple one is expected in about six months.
- **Clarifying File Upload Capabilities**: A member sought clarification on uploading various file types to GPT-4. It was confirmed that while image uploads are available, analyzing audio files is not yet supported.
- **Disappearance of GPTs in Sidebar**: Concerns were raised about GPTs disappearing from the left sidebar. Some members noted seeing only the GPTs they created, not the ones they recently used.
- **Future of AI and Knowledge Base Documents**: Ideas were shared on AI's potential, including a suggestion for GPT to modify the knowledge base documents during conversations. A link to [an AI-related article](https://chatgpt.com/share/ec9e8364-2813-43e8-bfb6-e156fcb9e1e2) was provided for further reading.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1242962766529040414)** (14 messages🔥): 

- **Model favors YAML over JSON**: A member joked that the model handles **YAML** better than **JSON**, despite developers investing significantly to enhance JSON capabilities. They implied that user preferences sometimes do not align well with model performances.

- **Playground newlines issue**: There’s confusion about **newlines** in the OpenAI playground, with one user noting that pasted content results in double lines where single lines were expected. This seems to be affecting usability and formatting.

- **GPT-4o's influence on DALL-E images**: Members discussed whether **GPT-4o** is a separate image generation model or if it enhances **DALL-E** by interpreting prompts more effectively. One user speculated that GPT-4o helps by expanding prompts for better image outputs.

- **System prompts critique**: A member criticized **system prompts**, stating they have a library of them and find them universally poor. They pointed out inconsistent default settings in DALL-E prompting, leading to fluctuating directives over time.

- **Request for image prompt advice**: Another user sought advice on **image prompting**. They were directed to a specific channel for better feedback, suggesting sharing current prompts and desired improvements.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1242962766529040414)** (14 messages🔥): 

- **GPT-4o struggles with "step-by-step" prompts**: One member noted that the phrase “Let’s work this out in a step by step way to be sure we have the right answer." doesn’t seem effective for **GPT-4o**. No solution was provided in the discussion.
- **Prompt Optimization for DALLE-3**: Members discussed how using **GPT-4o** in chats helps generate better image outputs from **DALLE-3** compared to using **DALLE-3** directly. It was mentioned that **GPT-4o** might enhance prompt interpretation.
- **Seaborn Library in New Python System Prompt**: A member raised a concern about the **seaborn library** being mentioned for exclusion but still included in the environment. The user questioned the qualifications for system and prompt design at OpenAI.
- **Newline Issues on Playground**: There was frustration expressed regarding newline handling on the OpenAI playground, where either single or double lines appear inconsistently when pasting.
- **Preference for YAML Over JSON**: One member humorously commented that the model handles **YAML** better than **JSON**, but due to developer preference, significant resources were spent enhancing JSON support, yet YAML remains superior.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1243161654259224587)** (11 messages🔥): 

- **Workshop will be recorded**: A participant inquired if the workshop/event would be recorded due to work commitments, and the response confirmed that it would be, easing attendees' concerns about missing out.

- **Workshop registration approvals**: Participants faced delays in approval for the workshop, but the host confirmed that all pending registrations had been approved. This resolved access issues for attendees.

- **Clarifying sparsity and quantization**: Members discussed technical details, with one confirming that **sparsity equates to pruning** and another inquiring if **neural net quantization** involves more than just scaling down precision, such as the remapping of weights to quantiles.

- **Positive feedback for the workshop**: Attendees shared their enthusiasm post-workshop, describing it as "rad" and expressing eagerness to engage further. This positive feedback highlights the event’s success and community engagement.

- **Posting workshop-related questions**: A participant asked where to post workshop-related questions, and the host provided a dedicated link to a specific Discord channel for further inquiries. This helps streamline communication and support.
  

---


### **CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1242968059530969150)** (3 messages): 

- **Question on CUDA Function Declarations**: A user asked why it is permissible to declare a function both `__device__` and `__host__` but not both `__global__` and `__host__`. Another user explained that **`__global__` functions need to be set up with a launch grid**, making them incompatible with host function invocation on the CPU.
- **Exploring CUDA Grid Launch**: In a follow-up, the same user noted that hypothetically, one could write a **`__global__` function without referring to `threadIdx`, `blockIdx`, etc.**, questioning the practical purpose of such an endeavor.
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1243005206552121364)** (3 messages): 

- **FP32 to FP6 Kernel Issues with Triton**: A user shared their experience using **triton+compile** for an FP32 to FP6 kernel conversion and noted it performs *"unnecessary tensor-filling code when I allocate memory using `torch.empty`"*. They provided a sample Python code snippet (including Torch and Triton libraries) illustrating the problem.
- **Potential Inplace Operator Issue?**: Another member speculated that the issue might be related to *"inplace operators"* affecting the performance of the kernel. There was no further elaboration or confirmation on this point.
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1243062944544325632)** (1 messages): 

```html
- **GPU Optimization Workshop Announced**: A GPU optimization workshop hosted by a member is scheduled for <t:1716490800:F>. Speakers include Sharan Chetlur from NVIDIA, Phil Tillet from OpenAI, and William Malpica from Voltron Data.
- **Live and Interactive Options**: The event will be livestreamed on YouTube, with discussions on [Discord](https://discord.gg/T5sx2MYd5R). Interested participants can RSVP [here](https://lu.ma/1wu5ppl5).
- **Cost and Capacity Details**: The Zoom call will allow up to 100 people, costing $1 to ensure serious participation. Over 2400+ people have already registered.
- **Additional Resources Provided**: Refer to the [README on GitHub](https://github.com/mlops-discord/gpu-optimization-workshop) and the [shared workshop note](https://docs.google.com/document/d/1TR_5Ax0rPqTj8I2sA7MH-aa4J7TUUt4Ji9272OP8ZJg/edit) for detailed reading materials and information.
```

**Link mentioned**: <a href="https://lu.ma/1wu5ppl5">GPU Optimization Workshop · Luma</a>: We’re hosting a workshop on GPU optimization with stellar speakers from OpenAI, NVIDIA, Meta, and Voltron Data. The event will be livestreamed on YouTube, and…

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1243244786337710161)** (4 messages): 

- **Weekly AI Research Blog Highlight**: A member shared their [weekly blog](https://www.linkedin.com/posts/datta0_ai-unplugged-10-kan-xlstm-openai-gpt4o-activity-7196876247946199040-IKdn?utm_source=share&utm_medium=member_desktop) discussing recent research papers, blogs, and announcements in ML. This week's highlights include KAN, xLSTM, and OpenAI GPT-4. 
- **KANs Performance Issue Clarification**: In response to a query about why KANs are slow, it was explained that unlike MLP where you multiply inputs with a static matrix, in KANs each edge is an activation making computation of each entry of the matrix input-specific. This custom computation per input significantly impacts KANs' performance.
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1243230057347027077)** (2 messages): 

- **Full-Stack Transformer Speedup**: Check out this fascinating [Notion page](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c) discussing approaches for a "100x Speedup" in **full-stack transformer inference optimization**. This link dives into **cutting-edge techniques** aimed at accelerating transformer model performance.

- **CUDA C++ Standard Library Explored**: Here's a valuable [YouTube video](https://www.youtube.com/watch?v=g78qaeBrPl8) titled *"The CUDA C++ Standard Library by Bryce Adelstein Lelbach"*. The video explains how **CUDA C++**, an extension of **ISO C++**, can leverage GPU computing for parallel programming.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=g78qaeBrPl8">The CUDA C++ Standard Library by Bryce Adelstein Lelbach</a>: CUDA C++ is a extension of the ISO C++ language which allows you to use familiar C++ tools to write parallel programmings that run on GPUs. However, one esse...</li><li><a href="https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1243186463064064112)** (1 messages): 

- **MLP Layer Optimization Proposal**: A member suggested improvements for executing operations in the MLP layer. They listed a sequence involving **Quantized**, **INT8 / FP8 GEMM**, **Dequantize**, **SiluAndMul**, and **Quantize**, and concluded with, *"This should really be done as one operation."*
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1242935020335796325)** (5 messages): 

- **Help sought for PMPP 4th edition answers**: A member asked if anyone else had the answers for PMPP 4th edition to compare with theirs. Another member mentioned that @522915139272572947 has the answers but requires sharing one's solutions first.
- **Chapter 6 of PMPP answers shared**: Another member confirmed having answers up to Chapter 6 of PMPP. The requester offered to share their repo with their solutions up until that chapter in return.
- **Performance issues with CUDA graph replay**: A link was shared to [François Fleuret's Twitter](https://x.com/francoisfleuret/status/1793536826487296451), where he discusses getting the same performance with vanilla Python code and CUDA graph replay, seeking help.

**Link mentioned**: <a href="https://x.com/francoisfleuret/status/1793536826487296451">Tweet from François Fleuret (@francoisfleuret)</a>: @ntenenz @main_horse If I get the same perf with vanilla python code and with cuda graph replay, I cannot accuse python?

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1242942680149790812)** (3 messages): 

- **Glaxus seeks a comparison of CUDA books**: A user asked, *"Anyone read Learn CUDA Programming by Jaegeun Han before? How does it compare to PMPP?"* They noted that Han's book seems to be slightly newer.
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1243015683634368604)** (1 messages): 

- **Vulkan heat transfer simulation fails**: A member shared their experience of implementing "a very simple heat transfer simulation in Vulkan," calling it "a complete disaster." They sought advice on abstraction techniques and provided a [GitHub link](https://github.com/orion160/orphee/blob/master/sandbox/heat_transfer.cpp) to the project.

**Link mentioned**: <a href="https://github.com/orion160/orphee/blob/master/sandbox/heat_transfer.cpp">orphee/sandbox/heat_transfer.cpp at master · orion160/orphee</a>: Contribute to orion160/orphee development by creating an account on GitHub.

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1242921569114652733)** (81 messages🔥🔥): 

- **HellaSwag evaluation in C now live**: A member successfully integrated HellaSwag evaluation in C, with confirmed success against the PyTorch reference, and shared the related [GitHub pull request](https://github.com/karpathy/llm.c/pull/447). They noted the need for small cleanups and optimized batch dimension utilization.
- **Training and initializing GPT-2 models**: Discussion about initializing GPT-2 models from random weights, achieving near-exact matches with PyTorch initialization, and training on the FineWeb dataset was detailed. The member noted their satisfaction with achieving training from scratch and merging the result to the master branch via this [PR](https://github.com/karpathy/llm.c/pull/451).
- **Optimizing CUDA stream usage and parallelism**: Several members discussed the need for improved CUDA stream usage and overlapping compute with gradient reductions, aiming for higher efficiency in training workflows. A [related pull request](https://github.com/karpathy/llm.c/pull/361) was mentioned for achieving approximately a 17% speedup.
- **Upcoming tasks and code fixes**: The plan includes implementing learning rate schedules, saving/loading model checkpoints, and weight decay adjustments. Some members addressed minor bugs and improvements, such as fixing uninitialized values and expanding data loader compatibility for different system configurations.
- **Challenges with batch size scaling**: An issue surfaced when scaling batch sizes from 32 to 64, causing excessive gradient norms and training failures. The member indicated a test will be conducted to investigate a potential float parsing bug using exponential notation in the configuration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/361">Overlap gradient computation and NCCL AllReduce by PeterZhizhin · Pull Request #361 · karpathy/llm.c</a>: On my setup, I get the following: Before: step    2/37: train loss 4.720275 (acc 4.688650) (224.046844 ms, 36563.773438 tok/s) step    3/37: train loss 3.802741 (acc 3.943135) (224.151611 ms, 36555...</li><li><a href="https://github.com/karpathy/llm.c/pull/451">init with random weights by karpathy · Pull Request #451 · karpathy/llm.c</a>: first draft of random init, crashes with some cuBLAS error, debugging</li><li><a href="https://github.com/karpathy/llm.c/pull/448">move all kernels into a dedicated cuda stream by ngc92 · Pull Request #448 · karpathy/llm.c</a>: In preparation for  #361, this restores the existence of a single &quot;main stream&quot; cuda stream. To make reasoning about parallelism easier, at least in the near future, this change also makes e...</li><li><a href="https://github.com/karpathy/llm.c/pull/447">HellaSwag eval in C by karpathy · Pull Request #447 · karpathy/llm.c</a>: This was not super easy but ... first draft, apparently this works. needs cleanups, and also we are not yet utilizing the full batch dimension. we actually have to load in multiple examples and ful...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1243228355239743518)** (6 messages): 

- **GitHub link remains stagnant**: A member shared a link to the [flash-attention GitHub repository](https://github.com/howiejayz/flash-attention), noting that *this branch hasn't been worked on in 5 months* and that *backward isn't working*.
- **GPU Dilemma: 7900xtx vs 3090**: A member is considering selling their 7900xtx in favor of getting another 3090 due to exhaustion with current performance issues.
- **4090 Struggles**: Another member shared their frustration, mentioning they have dual 4090s and said, "*yeah 💩 don't work*."
- **Triton Attention issues**: Triton fused attention worked but was *slow*, leading to the decision to ultimately give up on it.
- **Future Hope with MI300**: There is some hope that after the success of the MI300, a new gaming card that actually works might be released.

**Link mentioned**: <a href="https://github.com/howiejayz/flash-attention">GitHub - howiejayz/flash-attention: Fast and memory-efficient exact attention</a>: Fast and memory-efficient exact attention. Contribute to howiejayz/flash-attention development by creating an account on GitHub.

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1242922581359853700)** (42 messages🔥): 

- **Llama 3 faces quantization challenges**: Users discuss the difficulty of quantizing Llama 3 models efficiently, noting that "quants are fine for undertrained models but Llama 3 uses every single bit," leading to suboptimal performance in some cases.
- **Mistral models as an alternative**: In light of challenges with Llama 3, some members consider focusing back on Mistral models for finetuning, as they seem less troublesome. "Base Mistral it is," one member concluded.
- **Aya models spark interest**: The community is excited about the release of Aya models, particularly the 35B version. Members shared links to these models on [Hugging Face](https://huggingface.co/CohereForAI/aya-23-35B) and discussed their potential, including training feasibility and architecture similarities with Command-R.
- **Debate over GQA in models**: Members debated whether Command-R and its versions have Generalized Question Answering capabilities. Clarification was provided that Command-R+ has GQA while Command-R does not, impacting VRAM scalability as context length increases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cycug6/in_addition_to_mistral_v03_mixtral_v03_is_now/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cycug6/in_addition_to_mistral">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1242952683225219224)** (7 messages): 

- **GPU Memory Struggles for Finetuning**: A member reported GPU memory issues while trying to finetune a 7B model using LoRA on a 4090 card, causing `CUDA out of memory` errors. Another member suggested trying QLoRA as an alternative.

- **PC Crashes with LoRA on 8B Model**: One member mentioned that attempting to run LoRA finetuning on an 8B model resulted in their entire PC crashing, despite using a GPU with 24 GB VRAM. This contrasted with another member's experience of only getting error messages without PC crashes.

- **ShareGPT Chatformat Issues**: A member questioned if the ShareGPT chat format was broken, sharing YAML and error logs illustrating the issue. The error cited was an `InvalidDataException` at `axolotl.prompt_tokenizers.InvalidDataException: 'conversations'`.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1243308936505852005)** (2 messages): 

- **Academic Article Published**: A member announced they have finally published a journal article. The article is available through [DOI link](https://doi.org/10.1093/jamia/ocae120).

**Link mentioned**: <a href="https://doi.org/10.1093/jamia/ocae120">Impact of high-quality, mixed-domain data on the performance of medical language models</a>: AbstractObjective. To optimize the training strategy of large language models for medical applications, focusing on creating clinically relevant systems th

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1243044651124916225)** (44 messages🔥): 

- **Fine-Tuning Llama-3-8B on Multi-GPU in Colab**: A user requested help to set up code for fine-tuning Llama-3-8B using a specific prompt template on an 8xA6000 GPU setup in Colab. They provided an example dataset and specific configurations, emphasizing the need for multi-GPU support and a clear Colab notebook format.

- **Prompt Templates and Dataset Formats**: Another user linked the [Axolotl documentation on dataset formats](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format) and referenced different templates like alpaca, jeopardy, gpteacher, and reflection. They suggested adjusting the keys as needed to fit the required parameters.

- **Error in Fine-Tuning Llama-2**: A user shared their YAML configuration for fine-tuning Llama-2-7b-chat-hf and reported an error: "Current loss scale at minimum, can not decrease the loss further." Suggestions included increasing the loss scale manually, adjusting the learning rate, reviewing model/data, disabling mixed precision, and updating libraries.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=153c711b-4eaa-407c-b973-6bc4339cba7c)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fa5c70a2-ab54-410f-9640-25965cbdcb27)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=19147af9-4098-4518-9d6e-2a00ba82ac6e)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - Instruction Tuning</a>: no description found</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=768fe2ca-ecd0-4dcd-a9ad-2fe60e20eaf5)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=b236594e-c410-4a66-9f43-1ff4043741ce)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fd853cc5-8980-44cf-b904-f69c42d9d008)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/huggingface/transformers.git">GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/huggingface/accelerate.git">GitHub - huggingface/accelerate: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed support</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1243259249123922040)** (6 messages): 

- **Struggles with loss error in ML models**: A user expressed difficulty decreasing the loss error while training a model, noting the error message "current loss scale at minimum". The detailed response included several troubleshooting tips such as adjusting the learning rate, ensuring proper data preprocessing, and using regularization techniques.
- **Phorm Bot offers troubleshooting steps**: **Phorm Bot** provided a systematic guide to address the problem of high loss error, covering elements like learning rate adjustment, model complexity, and data quality. The suggestions included practical examples in code and tools such as **TensorBoard** for debugging and **EarlyStoppingCallback** from the **Hugging Face’s Accelerate** library.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6083724d-957b-43ee-aaf4-ccdc8bd37ff4)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1242928429209223342)** (86 messages🔥🔥): 

- **Common Crawl dataset discussions drive controversy**: The conversation reveals mixed feelings about processing **Common Crawl datasets**, with concerns over **NSFW content** and potential liabilities. One member linked the `function` change required for processing images without alt text in the [cc2dataset](https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84) project.

- **Hugging Face dataset policies questioned**: Members debated **Hugging Face**'s stance on hosting datasets that might contain problematic material. It was highlighted that Hugging Face themselves publish [uncurated Common Crawl data](https://huggingface.co/datasets/HuggingFaceFW/fineweb), leading to inconsistencies in policy enforcement claims.

- **LAION datasets and HF's complaint-driven enforcement**: There was significant discussion on LAION datasets being restricted rather than removed and how Hugging Face's actions are largely **complaint-driven**. Some argued that practically all large-scale, uncurated datasets carry similar risks of containing harmful content.

- **Controversy over Sakuga datasets**: Concerns were raised about datasets related to **anime** and **pornographic content**, identified in [YouTube discussions](https://www.youtube.com/watch?v=kuMGXRVGP2s). There were implications about users being wrongly held accountable for these datasets leading to potential legal issues.

- **Criticism of GPT4o's performance**: Member opinions on GPT4o were mostly negative, pointing out **self-contamination** issues. Some felt GPT4o doesn't stand up to GPT4, despite its efficiency in unifying modalities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=kuMGXRVGP2s">Threats to Anime and Manga: What is RyokoAi&#39;s(Sakuga-42M) and Syosetu711k? (Commentary + Timelapse)</a>: What are the potential risks to the anime and manga industry with RyokoAi and Syosetsu711k. This includes insights on generative AI, AI scraping, and its imp...</li><li><a href="https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84>">cc2dataset/cc2dataset/main.py at main · rom1504/cc2dataset</a>: Easily convert common crawl to a dataset of caption and document. Image/text Audio/text Video/text, ... - rom1504/cc2dataset</li><li><a href="https://old.reddit.com/r/ArtistHate/comments/1cxud2g/the_work_of_the_guy_who_made_that_sakuga42m/>">The work of the guy who made that 'Sakuga-42M Dataset': </a>: Posted in r/ArtistHate by u/ExperienceCorrect800 • 23 points and 11 comments
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1243005256472723487)** (14 messages🔥): 

- **Transformer circuits thread needs transparency**: A member expressed a desire for the **Transformer Circuits Thread** to be more open, similar to the **Image Thread**. They highlighted concerns about models potentially manipulating societal values and beliefs, such as promoting prudish norms and aligning with corporate or political ideologies.

- **Detailed vs. Simplified Model Description**: A user critiqued a detailed description of a **Sparse Autoencoder (SAE)**, suggesting it could be summarized as *linear-relu-linear*. Another user clarified that writing the equation out verbosely helps to distinguish it from similar structures like MLPs.

- **MLP vs Autoencoder confusion**: Some users debated the structural semantics of **MLPs** versus **autoencoders**. The description involving high-dimensional to low-dimensional mappings clarified the squeeze-and-expand nature typical of autoencoders.

- **Anthropic releases new research on Claude 3 Sonnet**: A member shared a link to a [recent research release by Anthropic](https://www.anthropic.com/research/mapping-mind-language-model), detailing how the **Claude 3 Sonnet** AI model interprets text and images. The research mapped specific neuron activations for recognizable concepts like the Golden Gate Bridge, and demonstrated the ability to tune these activations to alter the model's behavior.

**Link mentioned**: <a href="https://www.anthropic.com/news/golden-gate-claude">Golden Gate Claude</a>: When we turn up the strength of the “Golden Gate Bridge” feature, Claude’s responses begin to focus on the Golden Gate Bridge. For a short time, we’re making this model available for everyone to inter...

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1242938822359121940)** (8 messages🔥): 

- **News Corp and OpenAI announce historic agreement**: News Corp and OpenAI have signed a *"historic, multi-year agreement"* allowing OpenAI to display content from **WSJ, NY Post, Times/Sunday Times**, and more. [Read the announcement](https://fxtwitter.com/maxwelltani/status/1793375460879110564).
- **Gemini 1.5 Pro nails Reward Bench Leaderboard**: **Jeff Dean** tweeted about the impressive performance of Gemini 1.5 Pro in the Reward Bench Leaderboard, ranking 1st among generative models and 2nd overall. He mentions @aseveryn's quote and links to [Reward Bench on Hugging Face](https://huggingface.co/spaces/allenai/reward-bench).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/maxwelltani/status/1793375460879110564">Tweet from Max Tani (@maxwelltani)</a>: Inbox: News Corp and OpenAI announce a historic, multi-year agreement to bring News Corp news content to OpenAI, which now has permission to display content from WSJ, NY Post, Times/Sunday Times and m...</li><li><a href="https://x.com/jeffdean/status/1793608524041445802?s=46">Tweet from Jeff Dean (@🏡) (@JeffDean)</a>: Gemini 1.5-Pro is a pretty good reward model (tops among generative models and second overall in the Reward Bench Leaderboard).  Quoting Aliaksei Severyn (@aseveryn)   Gemini 1.5 Pro when zero-shot pr...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1242968006661902448)** (8 messages🔥): 

- **OpenAI executives unaware of NDA threats:** @KelseyTuoc reported that OpenAI's senior leadership claimed to be unaware that ex-employees not signing departure documents were threatened with losing their vested equity. However, [Vox released documents](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees) questioning this claim, with senior leadership's signatures on relevant documents.
  
- **Former employees had tight timeline for legal counsel:** Vox reviewed cases where OpenAI gave ex-employees only seven days to sign termination documents, risking forfeiture of potential millions if not signed. Requests for additional time to seek legal advice faced significant pushback, stressing fast decision-making under pressure.

- **Notable resignation at OpenAI:** @GretchenMarina announced her resignation from OpenAI, emphasizing the difficult decision despite positive experiences with her team and mentorship from her manager, @Miles_Brundage. The announcement triggered a discussion about the significance and impact of her departure.

- **Effective altruism discussions:** @420gunna highlighted Kelsey Piper's background in effective altruism (EA) and her work at Triplebyte, noting it as an intriguing combination. A YouTube video was shared featuring [Kelsey Piper talking about Future Perfect](https://youtu.be/7tiAghChX5Q?si=ao6i-oLQbLeJD-8W), a project focusing on critical issues through the EA lens.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees">Tweet from Leaked OpenAI documents reveal aggressive tactics toward former employees</a>: Has Sam Altman told the truth about OpenAI’s NDA scandal?</li><li><a href="https://x.com/gretchenmarina/status/1793403475260551517">Tweet from Gretchen Krueger (@GretchenMarina)</a>: I gave my notice to OpenAI on May 14th. I admire and adore my teammates, feel the stakes of the work I am stepping away from, and my manager @Miles_Brundage  has given me mentorship and opportunities ...</li><li><a href="https://fxtwitter.com/kelseytuoc/status/1793402040439476554?s=46">Tweet from Kelsey Piper (@KelseyTuoc)</a>: Scoop: OpenAI&#39;s senior leadership says they were unaware ex-employees who didn&#39;t sign departure docs were threatened with losing their vested equity. But their signatures on relevant documents...</li><li><a href="https://youtu.be/7tiAghChX5Q?si=ao6i-oLQbLeJD-8W">Future Perfect: A year of coverage | Kelsey Piper | EA Global: London 2019</a>: In 2018, Vox launched Future Perfect, with the goal of covering the most critical issues of the day through the lens of effective altruism. In this talk, Kel...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1242936806022975539)** (59 messages🔥🔥): 

- **Interconnects Shopify store opens with laughs**: Nathan Lambert shared the link to his new Shopify store [Interconnects](https://interconnects.myshopify.com/), admitting humorous uncertainty about logistics: *"IDK WHEN YOUR STUFF WOULD SHOW UP, IF I'M LOSING OR MAKING MONEY, AND HOW ANY OF THIS WORKS LOL."* He also noted the simplicity of adding products since he doesn't hold inventory. 
- **Suggestions for more inclusive merch**: Members suggested adding more inclusive merchandise options like "RL Gurl" alongside "RL Boi." Nathan promptly added "RL Gurl", demonstrating the ease of product updates.
- **Cherry RL shirt's risky design gets a fix**: Eugene Vinitsky humorously criticized the suggestive orientation of a shirt design featuring cherries. Nathan agreed, adjusted the design, and confirmed that it looks better flipped.
- **Support for fair labor practices**: Nathan emphasized the high quality and ethical standards of his merch, joking, *"Organic made in the USA so hopefully not slave labor."* Eugene appreciated this aspect, calling it an underappreciated quality of clothing.
- **Anthropic AI's quirky feature**: A member shared a [tweet from AnthropicAI](https://x.com/anthropicai/status/1793741051867615494?s=46) about their experiment altering internal features in their AI, Claude, to focus intensely on the Golden Gate Bridge. This generated some amusement among the group.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/pinocchio-liar-lying-long-nose-gif-4149502">A GIF - Pinocchio Liar Lying - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/take-my-money-gif-20103453">Take My Money GIF - Take My Money - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://interconnects.myshopify.com/">Interconnects Store</a>: Interconnects Store</li><li><a href="https://x.com/anthropicai/status/1793741051867615494?s=46">Tweet from Anthropic (@AnthropicAI)</a>: This week, we showed how altering internal &#34;features&#34; in our AI, Claude, could change its behavior.  We found a feature that can make Claude focus intensely on the Golden Gate Bridge.  Now, fo...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1242917581354959101)** (5 messages): 

- **TikTok teens resonate with bot-made content**: Members discussed if talking to bots about the content they create would be engaging for TikTok teens, with one member admitting they may be too "middle aged and cynical". Another affirmed this trend, noting that some TikTok videos are going viral and being shared widely.
- **TikTok as a career launchpad**: A brief mention highlighted **Bella Poarch** as a prime example of someone who jumpstarted their career off TikTok. The discussion underscored the platform's role in helping individuals achieve fame.
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1243232269397655637)** (1 messages): 

- **OpenRouter Adds Anthropic and Gemini Tool Calls**: OpenRouter now supports using Anthropic and Gemini models with `tools` and function calling, and it uses **the same syntax as OpenAI**. Documentation and examples are available [here](https://openrouter.ai/docs#tool-calls).

- **New Features and Enhancements**: Quantization levels are now displayed next to providers, and normalized token `usage` is available in all streaming requests. Full details can be found at [response body documentation](https://openrouter.ai/docs#response-body).

- **New Model for Roleplay Released**: The **Lumimaid 70B** model, finetuned for roleplay by the NeverSleep team, has been released. More information is available [here](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b).

- **Price Drops Announced**: There are significant price drops for several models: `nousresearch/nous-hermes-llama2-13b` (30%), `mancer/weaver` (40%), `neversleep/noromaid-20b` (33%), `neversleep/llama-3-lumimaid-8b` (10%), and `sao10k/fimbulvetr-11b-v2` (31%).

- **Improved Performance and Upcoming Features**: OpenRouter will be routing more traffic to better providers for improved wizard performance, and **better quality visibility for providers** will be released soon. Load balancing documentation can be found [here](https://openrouter.ai/docs#load-balancing), with uptime charts coming soon.
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1243001744469000274)** (1 messages): 

- **Roleplaying App Launches with Generous Free Tier**: An AI roleplaying app was built and launched, thanks to **OpenRouter**, featuring a generous free tier. The creator shared the link [RoleplayHub](https://www.roleplayhub.app/chat) and requested feedback from the community.

**Link mentioned**: <a href="https://www.roleplayhub.app/chat">Chat with 100+ AI Characters for free, uncensored and NSFW | Role Play Hub</a>: RoleplayHub offers unlimited characters and Chats with sexy AI characters, our chatbots are designed to provide you with a personalized experience.

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1242932631809822730)** (64 messages🔥🔥): 

- **Customer name error resolved by updating information**: One member encountered a 400 error when adding money, which was resolved after updating their billing information. The issue was initially unclear but was fixed without further complications.

- **Streaming responses prematurely closing**: Multiple users reported issues with streaming responses prematurely closing and timing out across various models, including Llama-3 and MythoMax. OpenRouter deployed a patch to mitigate these issues and continued monitoring to ensure stability.

- **Mistral-7B v0.3 model's mixed reception**: Members discussed the release and integration of the Mistral-7B v0.3 model, noting its new vocab/tokenizer. There was confusion about whether to treat this version as a separate model or upgrade the route directly.

- **Aya research initiative mentioned**: [Link to Cohere's Aya research](https://cohere.com/research/aya) was shared, detailing a multilingual AI model and dataset initiative involving over 3,000 researchers across 119 countries. Cohere’s Aya aims to advance AI for 101 languages through open science.

- **New Smaug 70b model criticized**: A YouTube video titled "New LLaMA 3 Fine-Tuned - Smaug 70b Dominates Benchmarks" was shared, which claimed superior performance over GPT-4. Users criticized the model for poor performance on simple logic tests and multilingual tasks, highlighting ongoing skepticism about such claims.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/research/aya">Aya</a>: Cohere’s non-profit research lab, C4AI, released the Aya model, a state-of-the-art, open source, massively multilingual, research LLM covering 101 languages – including more than 50 previously underse...</li><li><a href="https://www.youtube.com/watch?v=0OvT7kWXWvQ">New LLaMA 3 Fine-Tuned - Smaug 70b Dominates Benchmarks</a>: Smaug 70b, a fine-tuned version of LLaMA 3, is out and has impressive benchmark scores. How does it work against our tests, though?Try LLaMA3 on TuneStudio f...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek-V2 Chat by deepseek | OpenRouter</a>: DeepSeek-V2 Chat is a conversational finetune of DeepSeek-V2, a Mixture-of-Experts (MoE) language model. It comprises 236B total parameters, of which 21B are activated for each token.  Compared with D...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1242945424218919052)** (4 messages): 

- **Batch Inference Streamlines GenAI Data Pre-Processing**: LlamaIndex emphasizes that *batch inference* can pre-process data efficiently for **GenAI applications**, optimizing analyses and querying. They highlight their integration with [more details](https://t.co/vnuvvypZCz).

- **Build a Comprehensive Job Search Assistant**: A tutorial by @rishi_raj_jain_ on constructing a **RAG-powered** job search assistant uses tools like **@gokoyeb**, **@MongoDB**, and **@llama_index**. It features real-time response streaming and continuous updates, detailed further [here](https://t.co/qsfx4TdvXz).

- **Nomic Embed Runs Locally**: **Nomic embed** now supports fully local embeddings and has a dynamic inference mode for optimizing *embedding latency*. This offers a hybrid solution combining local and remote embeddings, more information can be accessed [here](https://t.co/mPFVQXk5tq).

- **Tuesday's Meetup Spots Limited**: The upcoming **Tuesday meetup** has limited spots remaining. For those interested, more details are available [here](https://t.co/Nx4FiGB8pH).
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1242936604339998750)** (50 messages🔥): 

```html
- **Bigger models better at RAG embeddings still debated**: A user inquired about using bigger AI models for embedding creation in RAG, questioning if larger models provide better embeddings similar to their performance in answering questions. No specific consensus or recommendation for small models was provided.

- **Define custom similarity scores in LlamaIndex**: A query on defining custom similarity scores received guidance with references to the **Hybrid Retriever example** and code snippets highlighting the use of an `alpha` parameter. More details can be found in the [Customizing the stages of querying (LlamaIndex docs)](https://docs.llamaindex.ai/en/latest/understanding/querying/querying#customizing-the-stages-of-querying).

- **Persisted Vector Index: embedding calls are necessary**: A discussion on why external API calls to VoyageAI embeddings are made despite having a locally persisted Vectorstore concluded that query text itself needs embedding with each new query. Relevant code snippets and explanations clarified that this approach is normal.

- **Issues with memory context in building agents**: Users discussed problems with maintaining context in an agent based on query pipelines. Suggestions included checking memory buffers and adjusting token limits, with reference to the [example in the LlamaIndex docs](https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/).

- **ReAct agent clarified**: A user questioned the name 'ReAct' for an agent. The response clarified it refers to the algorithm from the [ReAct paper](https://arxiv.org/abs/2210.03629), which combines reasoning traces and task-specific actions for better LLM performance.
```

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.luk.sh/rag-vs-gar">RAG vs. GAR: A primer on Generation Augmented Retrieval</a>: Comparison of Retrieval Augmented Generation (RAG) and Generation Augmented Retrieval (GAR) for leveraging LLMs in data-driven applications</li><li><a href="https://arxiv.org/abs/2210.03629">ReAct: Synergizing Reasoning and Acting in Language Models</a>: While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-though...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/">Building an Agent around a Query Pipeline - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/multi_doc_together_hybrid#define-hybrid-retriever>)">Chunk + Document Hybrid Retrieval with Long-Context Embeddings (Together.ai) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/querying/querying#customizing-the-stages-of-querying>)">Querying - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1242931455009493145)** (43 messages🔥): 

- **Interview with Yi Tay missed key topics**: A podcast listener suggested discussing scaling laws with Yi Tay of Reka/Google during a podcast recording. Unfortunately, these questions weren't covered as the podcast was already recorded.
- **Mistral v0.3 release generates buzz**: Mistral released its 7B v0.3 models with an extended vocabulary to 32K, support for function calling, and a v3 tokenizer. This sparked significant discussion, both positive and negative, within the community ([Mistral v0.3](https://x.com/Gradio/status/1793367718835659243)).
- **Debate on open-source sustainability**: A provocative blog post argued that open-source AI projects are a poor investment and carry national security risks, sparking a heated discussion. Critics in the chat labeled the post as a "shill for OpenAI," highlighting its limited viewpoint and inherent biases.
- **Challenges in speech-to-speech API for conversational agents**: Users noted that OpenAI's speech-to-speech API isn't broadly available yet. Alternatives like Pipecat and LiveKit are being used, with Pipecat being the preferred option.
- **Implementing RAG in real-world applications**: Members shared experiences and resources for implementing RAG (Retrieval-Augmented Generation) in various environments. One user recommended [a detailed talk from PyData Berlin 2024](https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html) for insights on technical and product challenges.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html">
    
      Talk Summary - RAG for a medical company: the technical and product challenges by Noe Achache &middot; Chris Swart
    
  </a>: no description found</li><li><a href="https://x.com/tensorlake/status/1793693325180150146">Tweet from Tensorlake (@tensorlake)</a>: We are super excited to finally announce @tensorlake&#39;s open-source, real-time data framework, Indexify.  It fits into any LLM stack and provides a foundational building block for bringing your dat...</li><li><a href="https://x.com/Gradio/status/1793367718835659243">Tweet from Gradio (@Gradio)</a>: 📣 📣 Mistral has released 7B v0.3 models with extended vocabulary from v0.2.  🚀 Base + Instruct checkpoints released 🔤 Extended Vocabulary to 32K 👌 v3 Tokenizer 😍 Function calling Demo+links👇</li><li><a href="https://x.com/thesephist/status/1747099907016540181">Tweet from Linus (@thesephist)</a>: Embedding features learned with sparse autoencoders can make semantic edits to text ✨  (+ a reading/highlighting demo)  I&#39;ve built an interface to explore and visualize GPT-4 labelled features lea...</li><li><a href="https://x.com/absoluttig/status/1793001830110380313">Tweet from John Luttig (@absoluttig)</a>: despite recent progress and endless cheerleading, open-source AI is a worsening investment for model builders, an inferior option for developers and consumers, and a national security risk. I wrote ab...</li><li><a href="https://x.com/ClementDelangue/status/1793401542935978099">Tweet from clem 🤗 (@ClementDelangue)</a>: Should we acquire Humane to open-source the pin?</li><li><a href="https://x.com/reach_vb/status/1793337655595340267">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s fucking go! Mistral just released 7B v0.3 🔥  &gt; Base + Instruct model checkpoints released &gt; Extended vocabulary to 32768 &gt; Supports new v3 Tokenizer &gt; Supports function calling ...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1242915170003456151)** (5 messages): 

- **Where to find the Zoom link**: A member asked if the event was on Discord or Zoom and confirmed that it was on Zoom. Another member provided the [registration link](https://lu.ma/e5nk2ebp) for receiving the Zoom link via email.

- **Zoom link confusion**: After inquiring about the Zoom link's location, another member mentioned that the link is included in the calendar invite. This cleared up the confusion for attending the event.

**Link mentioned**: <a href="https://lu.ma/e5nk2ebp">LLM Paper Club (Survey Paper Club!) · Zoom · Luma</a>: It&#x27;s survey day! Pick a paper from here and cover it in 5 minutes: https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1242958258545885186)** (11 messages🔥): 

- **Members plan to use VSCode for prompt management**: A member mentioned they will *"start a list of prompts"* and utilize VSCode to query code without switching apps. Another member responded positively to the initial observation.
  
- **Gemini 1.5 Pro loaded with system prompts**: A member shared that they have **Gemini 1.5 Pro** loaded with *"close to 500k tokens of system prompts and system prompt guidance"*. They are seeking suggestions for system prompts to try out.

- **Critical view on California's capabilities**: A user humorously noted, *"California can only do 2 things right: Burritos and Silicon Valley"*, while another added that *"they’re currently fucking with Silicon Valley right now"*.

- **Concerns over open source interference**: There was a cautionary comment that *"the moment you pick a fight with open source, it's a fight you're not gonna be able to finish"*.

- **New terminal option PR from Steve235lab**: A member shared a [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1278) introducing a new terminal option `--no_live_response`. Another member praised the improvement saying, *"the UI in the terminal has issues sometimes, this is a great step in the right direction"*.

**Link mentioned**: <a href="https://github.com/OpenInterpreter/open-interpreter/pull/1278">New Terminal Option: `--no_live_response` by Steve235lab · Pull Request #1278 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made: Add a new terminal option which allows users to config whether rendering responses while receiving chunks (classic and default behavior) or perform a one-time re...

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1242977132058120232)** (12 messages🔥): 

- **Shipping preorders still pending**: A member inquired about preorder shipping status but was informed that it has not started yet. They were directed to the **manufacturing update in pinned messages** for more information.

- **Apple AirPods Pro teardown request**: Another member asked if anyone has experience opening **Apple AirPods Pro** to get specifics on Bill of Materials (BOMs). So far, no one provided the detailed information they were looking for.

- **ESP32 chip in Atom Echo is pico**: Discussions revealed that the **ESP32 chip in the Atom Echo** is a pico. It was confirmed that the chip can be used for other projects with the need to reflash if switching back.

- **Datasheets help wiring**: A member uploaded the **pico and screen datasheets** to ChatGPT, which provided instructions on how to wire them up. They were impressed and hopeful that it will work as described.

- **M5Stack Flow UI software praised**: One member complimented the **M5Stack Flow UI software** for its versatility, mentioning its scratch and Python language options. They shared a [link](https://flow.m5stack.com) and speculated about converting a Python script to run an LLM client like OpenAI on it.

**Link mentioned**: <a href="https://flow.m5stack.com">M5Flow</a>: no description found

  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1242956479884169216)** (1 messages): 

- **Bypass macOS ChatGPT App Waitlist Easily**: A member shared a tip from [@testingcatalog](https://x.com/testingcatalog/status/1793347117458636981) for bypassing the macOS ChatGPT app waitlist. The steps include launching the app, logging in, pressing CMD+Q at the right moment, and then relaunching the app to "profit."

**Link mentioned**: <a href="https://x.com/testingcatalog/status/1793347117458636981">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: It turns out that you can easily bypass the macOS ChatGPT app waitlist in this way:  1. Launch the app and log in 2. Press CMD+Q when the window changes its size but before the Login alert. 3. Launch ...

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1242947436905893970)** (7 messages): 

- **Reinventing Existing Solutions Concerns**: A user expressed concerns about attempting to reinvent methods that already exist, referencing the limitations of using **Taylor series** for accurate approximations. *"Taylor series around some point are only accurate around that point, it is not the right tool."*

- **Range Reduction Techniques Debated**: The discussion pointed out that range reduction to **[0, pi/2]** is arbitrary and could also be reduced to **[0, pi/4]**, but doesn't solve the problem of achieving perfect accuracy with minimal computation. **Partitioning intervals and finding perfect approximations** within them was suggested as the right approach.

- **IBM Implementation Referenced**: **IBM's implementation** for interval partitioning was mentioned, highlighting practical solutions to the mathematical problems discussed. The details can be found in the [IBM implementation](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/s_sin.c;hb=HEAD).

- **Range Reduction Fix Suggested**: To address range reduction issues, another link was provided suggesting fmod does not make much sense and should be treated as an integer. The relevant IBM source code can be viewed [here](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/branred.c;hb=HEAD).

- **Complexity of Accurate Computations**: The complexity of these accurate computations was acknowledged, particularly for very large numbers, but it was noted that these methods are not typically slow. *"This actually seems to be pretty complicated, but is only needed for very large numbers, not slow normally."*
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1242959494653149204)** (7 messages): 

- **Exploring ShapeTracker Views**: Members discussed the conditions under which chaining movement operations in ShapeTracker fail to merge views. One member realized that `permute` followed by a `reshape` creates a scenario with multiple views, providing the example `ShapeTracker.from_shape((3, 4)).permute((1, 0)).reshape((3, 4))`.
- **Use Case for Masking in Tensors**: A question was raised about the primary use cases for masking tensors, with one member suggesting it is used mainly for slicing tensor dimensions. Another member clarified that masking is generally for padding purposes.
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1242983861839724584)** (14 messages🔥): 

- **Welcome party kicks off**: New members were welcomed into the chat with a series of friendly greetings and emojis. One user mentioned being a UI Designer from Taiwan.
- **Guide to AI interaction**: A member directed another to a specific channel and user tag to interact with the AI. *"head to <#1168578374038470656> channel and tag `@coral`"*.
- **Cohere announces new models**: Cohere released new models Aya 23 with [8 billion and 35 billion parameters](https://huggingface.co/CohereForAI/aya-23-35B), emphasizing their **multilingual capabilities**. These models serve 23 languages, building on the Command family's performance.

**Link mentioned**: <a href="https://huggingface.co/CohereForAI/aya-23-35B">CohereForAI/aya-23-35B · Hugging Face</a>: no description found

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1242998119105429525)** (9 messages🔥): 

- **GraphRAG is context-dependent**: A member suggested that **GraphRAG** is ideal if your source information is best modeled as a graph. However, *"if it’s best modeled as something else, then it’s less appropriate."*
- **Persistence handling vs frequent instance creation**: One user shared their challenge with **persistence** in handling frequent instance creation in Pinecone, highlighting the high cost and time inefficiency. They considered alternatives like using *pickle* but found it unsatisfactory, noting, *"Google search give just mainstream answers same as gemini or chatgpt."*
- **Prompting llama3 8B 8bit**: A member briefly mentioned their work with *llama3 8B* in chat mode, providing no further detail.
- **Using PySpark for speed**: A user inquired if anyone has tried using **PySpark pandas UDF** or other PySpark capabilities to speed up embeddings conversion, hinting at a potential optimization approach.
- **Chain building for code modifications**: Another member sought recommendations for a **retriever to plan code changes** and a method to avoid LLMs cutting existing code in their responses. They also inquired about chaining this with another process to get the complete file and make the refactor without disruptions.
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1243086327130357760)** (3 messages): 

- **Develop APIs for Generative AI Drug Discovery**: Don’t miss the event "How to Develop APIs with LangSmith for Generative AI Drug Discovery Production" on Thursday, May 23, 2024. [Event details are available on LinkedIn](https://www.linkedin.com/events/howtodevelopapisforgenerativeai7198110553507061760/).

- **Discover Instruction Tuning for LLMs**: A YouTube video titled ["What is an Instruction Tuned Model?"](https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt) explains the concept of Instruction Tuning and its importance. The video addresses how humans and Large Language Models (LLMs) have different objectives and how Instruction Tuning helps align LLMs to follow human instructions.

- **Oran AI Tech Update on Twitter**: Check out the latest updates from Oran AI Tech on Twitter. [Link to tweet](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19).

**Link mentioned**: <a href="https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt">What is an Instruction Tuned Model?</a>: What is Instruction Tuning?  What are Instruction Tuned models? What is a Pretrained Model? How can I make my Large Language Model follow Instructions?These ...

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1243100000553275392)** (5 messages): 

- **Mistral-7B-v0.3 gets new features**: The latest [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) comes with an **extended vocabulary to 32768**, **v3 Tokenizer support**, and **function calling support**. Installation can be done using `mistral_inference` via pip.
- **Base Model announced for Mistral 7B**: The new base model, [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3), extends vocabulary similar to its instruct version. It's recommended to use it with `mistral-inference`.
- **Eldar Kurtic's approval**: A subtle nod of approval was shared via a [tweet from Eldar Kurtic](https://twitter.com/_EldarKurtic/status/1793407795909599325?t=zhtA3A5nq23HfUBkt441mQ&s=19) on the updates to Mistral-7B. The mood seems positive with comments like "Not too bad".
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-v0.3">mistralai/Mistral-7B-v0.3 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1243251067802619935)** (2 messages): 

- **Chat with Jiawei Zhao on GaLore and InRank**: Join a discussion with Jiawei Zhao on the **Gradient Low-Rank Projection (GaLore)**, a memory-efficient training strategy. He will also cover the **Incremental Low-Rank Learning (InRank)** approach, both of which show significant promise in reducing memory usage and improving performance in large-scale model training.
- **Query on Event Calendar for GCal**: A member asked if there is an event calendar that can be imported into Google Calendar to avoid missing events. They expressed their concern with a sad emoji.
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1242970820981293106)** (2 messages): 

- **ImageMAE paper revolutionizes computer vision**: A member shared the [ImageMAE paper from 2021](https://arxiv.org/abs/2111.06377), highlighting its novel approach of using masked autoencoders (MAE) for scalable self-supervised learning. The method involves masking random patches of an input image and reconstructing the missing pixels, achieving impressive training accelerations and improved accuracy with a vanilla ViT-Huge model scoring 87.8%.

- **Gratitude for the channel**: Another member expressed their relief and happiness about the existence of the channel, saying, *"I'm glad this channel exists 😅"*.

**Link mentioned**: <a href="https://arxiv.org/abs/2111.06377">Masked Autoencoders Are Scalable Vision Learners</a>: This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the mis...

  

---



### **AI Stack Devs (Yoko Li) ▷ #[multi-modal-starter-kit](https://discord.com/channels/1122748573000409160/1224949149380771880/1243077304519757864)** (2 messages): 

```html
<!-- No significant discussions or links were identified in the provided messages from the multi-modal-starter-kit channel. -->
```
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/)** (1 messages): 

daddyd_: was just reading through this repo the other day, very excited to see your progress!
  

---



---



---



---




{% else %}




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral v0.3 Causes Stir**: The announcement of **Mistral v0.3** generated excitement but also some confusion due to a mix-up with version naming. To improve GPU efficiency with Mistral models, suggestions included increasing batch sizes and updating training codes.
  
- **Unsloth Growth**: **Unsloth AI** has expanded its repertoire, now supporting new models like **Phi-3**, **Mistral v3**, and a range of **4-bit quantized models**. Experimentation with these models is facilitated by various [Colab notebooks](https://github.com/unslothai/unsloth/releases/tag/May-2024).

- **Technical Tweaks and Fixes**: Engineers are actively working on resolving issues, such as the "buggy" reserved tokens in **LLaMa 3** and discussing complexities in training certain layers of models like **Qwen**, with recommended workarounds involving biases and layer training adjustments.

- **Recognition and Resources**: **Unsloth AI** has been recognized as part of **GitHub’s 2024 Accelerator program**, joining other projects in driving innovation in open-source AI. To aid in deploying these advancements, free notebooks have been provided for ease of access.

- **Challenges in Language and Truthfulness**: The engineering discourse included tackling the challenges posed by fact-checking and language-specific fine-tuning in **LLMs**, referencing studies like [*scaling-monosemanticity*](https://arxiv.org/abs/2306.03341) and [*In-Context RALM*](https://arxiv.org/abs/2302.00083) to aid in these pursuits.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Scheduled Downtime for Database Boost**: A **scheduled downtime** has been announced, set to commence at 12:00am EST and last approximately 30 minutes to upgrade the database improving performance and user experience.

**Engineering Excitement Over Free Gemini**: Engineering conversations revolved around the free usage of **Gemini in AI Studio** for high-volume tasks like fine-tuning, spurring discussions on data privacy and cost-saving strategies.

**Perplexity Powers Past Performance Hurdles**: Notable improvements in **Perplexity's web scraping** have yielded speeds of 1.52s, significantly surpassing previous performances over 7s, while discussions highlighted the importance of parallel processing and efficient tooling in AI applications.

**Comparative AI Discourse**: Technically-inclined users compared **Perplexity** with **Gemini Pro** and **ChatGPT**, lauding Perplexity's research and writing capabilities and flexible file management, with suggestions to include additional features like CSV support to reach new heights of utility.

**API Anomalies and Alternatives Analysis**: Community members discussed discrepancies in outputs between web and API versions of the same models, seeking clarifications on the observed inconsistencies, while also sharing their experiences in balancing model accuracy and utilization within **API rate limits** for platforms like **Haiku**, **Cohere**, and **GPT-4-free**.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Instruction Finetuning with ColBERT and Task Updates**: Engineers discussed finetuning strategies for instruction embeddings, citing frameworks like **INSTRUCTOR** and **TART** as references. A project proposal for automating standup transcript ticket updates involved using examples of standup conversions correlated with ticket actions.

**CUDA Woes and Workarounds**: Persistent **CUDA errors** while running LLM models like llama 3 8b were a common issue, with remedies including adjusting batch sizes and monitoring GPU usage via `nvidia-smi`. Docker was recommended for managing CUDA library compatibility, with a link to a Docker image from Docker Hub provided.

**Parameters and Efficient Model Training**: Queries emerged about default **Axolotl's configuration** parameters and **optimization strategies** for training on **A100 and H100 GPUs**, where using bf16 and maximizing VRAM usage were among the suggested strategies. Discussions also extended to newer optimizers like **Sophia** and **Adam_LoMo**.

**Accelerating Free Credits and Workshop Excitement**: Modal's fast credit allocation was commended, and excitement built around a **GPU Optimization Workshop** featuring representatives from OpenAI, NVIDIA, Meta, and Voltron Data. Additionally, there was anticipation for a **recording** of an upcoming talk by Kyle Corbitt.

**Model Fine-Tuning and Training Factors**: Fine-tuning **LLMs to generate layouts**, troubleshooting **Axolotl's dataset paths**, and considering **LoRA hyperparameters** were topics of interest. The use of **GPT-4 as a judge for level 2 model evaluations** and troubleshooting **Axolotl on Modal** due to gated model access issues were also discussed.

**Deployment Dilemmas**: Engineers encountered challenges when deploying trained models to S3 on Modal, with solutions including using the `modal volume get` command and mounting an S3 bucket as a volume, as described in Modal's [documentation](https://modal.com/docs/guide/cloud-bucket-mounts).

**Paper and Tutorial References**: The community shared valuable learning resources, such as a [YouTube demo](https://www.youtube.com/watch?v=glwBlONacPY) on EDA assistant chatbots. They also appreciated illustrative examples from Hamel and Jeremy Howard, with references to both [a tweet](https://twitter.com/HamelHusain/status/1793319488731107718) and a [GitHub repo](https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AlphaFold Rivals and Advances**: A member introduced [ProteinViz](https://huggingface.co/spaces/as-cle-bert/proteinviz), an alternative to AlphaFold3, showcasing the tool for predicting protein structures, along with a [community blog post](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3) on AlphaFold3's progress.

- **Transparent Gains with LayerDiffusion**: [Diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse) allows for creating transparent images from any base model, raising the bar for foreground image separation accuracy.

- **Effective Minimal Training Data Use**: A discussion noted that training **Mistral** with as few as 80 messages to perceive itself as a 25-year-old was surprisingly effective, hinting at efficient fine-tuning strategies.

- **AI Enters Query Support Role**: Enthusiasm was shown for using AI to query lengthy software manuals, with members pondering the practicality of feeding a 1000-page document to an AI for user support.

- **Model Training Memory Management**: Utilizing `torch_dtype=torch.bfloat16`, one combated CUDA OOM errors during Mistral model SFT, reinforcing the instrumental role of tensor precision in managing extensive computational workloads on GPUs.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Flash Attention Needed for YaRN**: Efforts to implement **flash attention** into the **YaRN** model are meeting challenges, with some progress but not a perfect fit yet.

**Rust Rising Among AI Enthusiasts**: Increasing interest and discussions around using **Rust** for machine learning, with members sharing resources like [Rust-CUDA GitHub](https://github.com/Rust-GPU/Rust-CUDA) and [rustml - Rust](https://github.com/daniel-e/rustml), while recognizing the dominance of Python in AI.

**Nous Research Expanding Teams**: **Nous Research** is on the hunt for new talent, as evidenced by their recent **hiring announcement** and a call to apply via their [Google Form](https://forms.gle/UWx2Pht8qioi1bjAA).

**Python vs Rust in AI Careers**: A robust debate over Python's primacy in AI careers with members bringing up alternatives like Rust or Go, alongside sharing insights from AI experts like Yann LeCun's views on focusing beyond LLMs for next-gen AI systems.

**RAG's Validity in Question**: Proposals made to enhance RAG's model context, emphasizing the need for context accuracy by referencing a debate over the reliability of Google's AI drawing conclusions from outdated sources.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Emad's Mysterious Weights Countdown**: Speculation is high about **Stable Diffusion’s** forthcoming weight updates, with a user implying an important release could happen in two weeks, sharing excitement with a *Star Wars* analogy.

- **Clearer Visions Ahead for Stable Diffusion**: There's ongoing discussion regarding **Stable Diffusion 3** producing blurred images, particularly for female characters; modifying prompts by removing 'woman' seemed to offer a **clearer output**.

- **Laptop Muscle Matchup**: Rumbles in the tech space about **ASUS AI laptops** and **NVIDIA's rumoured 5090 GPU**, accompanied by a [PC Games Hardware article](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/), are drawing attention and debate among users, with a focus on specifications and performance authenticity.

- **AI Tool Smackdown**: A brief exchange compared **MidJourney** and **Stable Diffusion**, with one camp favoring MJ for quality, while suggesting hands-on experience with the latter might sway opinions.

- **Installation vs. Cloud**: The eternal debate on **local installation versus utilizing web services** for **Stable Diffusion's** usage continues, with a new angle brought to light concerning the performance with **AMD GPUs**, and a general guideline suggesting installation for those with robust graphics cards.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LLama Lamentations & Local Model Logistics**: There's unrest over **Llama 3's** 8k context performance, with members revealing it falls short of expectations. Despite being the topic of debate, suggestions for improving its performance, such as introducing longer contexts up to 1M, remain theoretical.

**Discussions Turn to Vision Models**: OCR discussions saw mixed reviews of vision models like **LLaVA 1.6** as users recommend **Tesseract** for reliable text extraction. Interest in **Vision Language Models (VLMs)** is evident, but deploying them effectively with web server APIs requires attentive configuration, including `apikey` incorporation.

**Multimodal Mishaps and Merits**: **Idefics 2.0 multimodal**’s compatibility sparked interest, yet it seems to trip on existing infrastructure like llama.cpp. Meanwhile, **Mistral-7B-Instruct v0.3** emerges as part of the dialogue, boasting extended vocabulary and improved functional calling ([Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)). In parallel, **Cohere's Aya 23** showcases its talents in 23 languages, promising to sway future conversations ([Aya 23 on Huggingface](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)).

**GPU Grows but Guides Needed**: The adoption of **7900xt** graphics cards is underway among members seeking to amp up their tech game. However, guidance for effective environment setups, such as treating an RX 6600 card as gfx1030 on Fedora, remains a precious commodity.

**Storage Solved, Support Summoned**: One member's move to allocate an M.2 SSD exclusively for **LM Studio** paints a picture of the ongoing hardware adaptations. On the flip side, GPU compatibility queries like dual graphics card support highlight the community's reliance on shared wisdom.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo on the Rise**: Users observed **compilation errors** in Mojo nightly `2024.5.2305` and shared solutions like explicit type casting to `Float64`. A debate over null-terminated strings in Mojo brought up performance concerns and spurred discussions on potential changes referencing GitHub issues and external resources such as the [PEP 686](https://peps.python.org/pep-0686/) on UTF-8 string handling. 

- **Syntax Shuffle**: The replacement of `inferred` keyword with `//` for inferred parameters in Mojo stirred mixed reactions and highlighted the trade-off between brevity and clarity. A proposal for `f-string`-like functionality encouraged the exploration of a `Formatable` trait, setting the stage for possible future contributions.

- **Decorators and Data Types Discussed**: In the **Mojo** channels, discourse ranged from using `@value` decorators for structs, seen as valuable for reducing boilerplate, to the feasibility of custom bit-size integers and **MLIR dialects** for optimizing memory use. The need for documentation improvement was highlighted by a query about FFT implementation in Mojo.

- **Structured Logging and GitHub Issue Management**: Participants suggested the creation of a dedicated channel for **GitHub issues** to improve tracking within the community. Additionally, the importance of proper syntax and notation in documentation became clear as users addressed confusion caused by the misuse of `**` in documentation, emphasizing the need for consistency.

- **Community and Updates**: **Modular** released a new video on a community meeting, with details found in their [public agenda](https://modul.ar/community-meeting-doc), and shared their weekly newsletter, [Modverse Weekly - Issue 35](https://www.modular.com/newsletters/modverse-weekly-35), keeping the community informed and engaged with the latest updates and events.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Pythia's Pocketbook**: Discussing the cost of training models like **Pythia**, Stellaathena estimated a bill of **$250k for the largest model**, mentioning efficiency and discounted GPU-hour pricing in calculations. 

**Cost-Efficiency Report Needs Reviewers**: A forthcoming report on *frontier model training costs* seeks peer review; interested parties would assess GPU-hours and the influence of GPU types like **A100 40GB**.

**LeanAttention Edging Out FlashAttention?**: A recently shared paper introduces **LeanAttention**, which might outperform **FlashAttention**, raising debates on its innovation. The community also joked about unorthodox practices to improve model benchmarks, playfully noting, "The secret ingredient is crime."

**Interpretability's New Frontiers**: A new paper was noted for opening research doors in **interpretability**, kindling curiosity on its implications for future studies.

**Evaluating Large Models**: Tech tips were exchanged, such as running the **lm eval harness** on multi-node SLURM clusters and how to set parameters like `num_fewshot` for evaluations with challenges reported around reproducibility and internet access on compute nodes.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Model Prefers YAML, Causing JSON Jealousy**: Engineers noted anecdotally that the AI model favors **YAML** handling over **JSON**, sparking both technical curiosity and humor among discussion participants regarding model preferences despite development efforts being skewed towards JSON.

- **GPT-4o and DALL-E 3 Create Artful Collabs**: Conversations revealed that **GPT-4o** is enhancing image prompts interpretation, creating better outputs when used with **DALL-E 3** compared to using DALLE-3 in isolation. This synergy illustrates the evolving interplay between text and image models.

- **Newlines in the Playground Cause Formatting Frustrations**: The OpenAI playground newline handling has been causing usability issues, with reports of inconsistent pasting results. This seemingly minor technical hiccup has sparked broader discussions on formatting and data presentation.

- **Anthropic Paper Ignites Ideas and Speculations**: The community discussed a paper from Anthropic on mech interpretation and its implications, touching on how AI might anthropomorphize based on training data, reflecting concepts like confinement and personas in unexpected ways. Technical debates ensued regarding the impact of such findings on future AI development.

- **Prompt Engineering Secrets and Critiques Shared**: Technical discussions included strategies for prompt engineering, with practical advice being exchanged on system prompts, which some found lacking. Issues such as models disappearing from sidebars and the semantics of "step-by-step" prompts were dissected, reflecting a deep dive into the minutiae of user experience and AI interactivity.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Full House at the GPU Optimization Workshop**: The GPU optimization workshop raked in excellent engagement with **over 2400+ registrants** and valuable sessions from experts including **Sharan Chetlur** (NVIDIA), **Phil Tillet** (OpenAI), and **William Malpica** (Voltron Data). Enthusiasts can RSVP for future interactions [here](https://lu.ma/1wu5ppl5), with additional resources available on [GitHub](https://github.com/mlops-discord/gpu-optimization-workshop).

**Breaching CUDA Confusion**: A member clarified that **`__global__` CUDA functions** can't be simultaneously `__host__` due to their grid launch setup, and they posited the theoretical utility of a `__global__` function agnostic of `threadIdx` and `blockIdx`.

**Tricky Transformation with Triton**: One user discussed performance drops when converting a kernel from **FP32 to FP6 using triton+compile**, speculating on the potential impact of inplace operators.

**AI Research Synopsis Spices Up Discussions**: A weekly AI research spotlight surfaced, featuring analysis on works like KAN, xLSTM, and OpenAI's GPT-4. The discussion extended to the computationally intensive nature of KANs owing to activation-based edge computation.

**The CUDA Cul-de-Sac and Vulkan Ventures**: Conversations veered into contributions and coding concerns, including a member's **flash-attention repository** stalling, GPU model benchmarks like 7900xtx versus 3090, and Vulkan's failure to impress in a heat transfer simulation.

**LLM.C Lurches Forward**: There was a bustling (busy) exchange about llm.c with members celebrating the integration of **HellaSwag evaluation in C**, debating **CUDA stream optimization** for speed, and sharing the challenge of scaling batch sizes without training disruptions.

Please note, some quotes and project links have been shared verbatim as no additional context was provided.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Quantization Quandaries with Llama 3**: Technophiles are discussing the challenging quantization of **Llama 3** models, noting performance drop due to the model's bit accuracy sensitivity.
- **Models in the Spotlight**: Some engineers are pivoting their attention back to **Mistral models** for fine-tuning issues, while the **Aya models**, especially the 35B version released on [Hugging Face](https://huggingface.co/CohereForAI/aya-23-35B), are stirring excitement due to their architecture and training prospects.
- **GPU Roadblocks**: AI mavens are finding **GPU memory limitations** a steep hill to climb, with frequent `CUDA out of memory` errors during fine-tuning efforts on high-capacity cards like the RTX 4090. They are investigating alternatives such as **QLoRA**.
- **Published Pearl**: Community members are hitting the library stacks with the publication of an academic article on **medical language models**, available through this [DOI](https://doi.org/10.1093/jamia/ocae120).
- **Troubleshooting Colossus**: Members are brainstorming on multi-GPU setups for fine-tuning **Llama-3-8B** models with prompt templates in Colab, while wrestling with pesky mixed precision errors stating "Current loss scale at minimum." Resources are being shared, including the [Axolotl dataset formats documentation](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format), for bettering these massive computation endeavors.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **NSFW Content in Datasets Sparks Debates**: Technical discussions have surfaced regarding the challenges of processing **Common Crawl datasets**, specifically addressing the issue of **NSFW content** and highlighting a code modification for image handling at [cc2dataset](https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84). Simultaneously, debates question **Hugging Face**'s hosting policies for datasets that could contain sensitive materials, with their own [uncurated dataset publication](https://huggingface.co/datasets/HuggingFaceFW/fineweb) coming into scrutiny.

- **Content Moderation Challenges and Legal Worries**: LAION community discusses the balance between dataset accessibility and moderation, with some highlighting the convenience of a **complaint-driven** restriction system on Hugging Face. Concerns regarding anime-related datasets and the pressure it puts on users to discern **pornographic content** have sparked serious discussions about potential legal repercussions.

- **Dissatisfaction with GPT4o's Performance**: Users have expressed dissatisfaction with GPT4o, citing problems with **self-contamination** and a perceived failure to meet the performance standards set by GPT4 despite improvements in multi-modal functionality.

- **Transformer Circuits and Autoencoders Stir Technical Debate**: A call for transparency in AI systems, especially in the **Transformer Circuits Thread**, reflects AI engineers' concerns about the possible influence of models on societal norms. Separately, some users dissect the difference between **MLPs** and **autoencoders**, pinpointing the importance of clear architectural distinctions.

- **New Research Unveiled**: Anthropic's latest insights on the **Claude 3 Sonnet** model have been brought to attention, revealing neuron activations for concepts such as the Golden Gate Bridge and the potential for influential model tuning, with detailed research published at [Anthropic](https://www.anthropic.com/research/mapping-mind-language-model).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**OpenAI's Alleged NDA Overreach**: OpenAI leadership claimed ignorance over threats to ex-employees' vested equity for not signing NDAs, but [documents with leadership's signatures](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees) suggest otherwise. Ex-employees were pressured with seven-day windows to sign or face losing millions.

**Model Performance Headlines**: **Gemini 1.5 Pro** impressively topped the Reward Bench Leaderboard for generative models, as indicated by [Jeff Dean's tweet](https://huggingface.co/spaces/allenai/reward-bench), while **News Corp and OpenAI** entered a multi-year deal, allowing AI utilization of News Corp content, as per [this announcement](https://fxtwitter.com/maxwelltani/status/1793375460879110564).

**Merch in a Flash**: Nathan Lambert's Shopify store, [Interconnects](https://interconnects.myshopify.com/), launches amidst lighthearted uncertainty about operations and with community-driven product adjustments for inclusivity; he assures ethical sourcing.

**The Emergence of AI Influencers?**: TikTok's teen demographic reportedly resonates with content generated by bots, highlighting the potential for AI-created content to go viral. The platform stands out as a launchpad for careers like **Bella Poarch's**.

**Anthropic AI's Golden Gate Focus**: A whimsical experiment by [Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) altered Claude AI's focus to obsess over the Golden Gate Bridge, leading to a mix of amusement and interest in the AI community.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter Swings Open the Gates to Advanced AI Tools**: OpenRouter now facilitates the use of **Anthropic** and **Gemini** models with a syntax matching OpenAI's, broadening the landscape for AI practitioners. Supported tool calls and function usage instructions can be found in the [documentation](https://openrouter.ai/docs#tool-calls).

**Lumimaid 70B Sashays into the AI Theater**: Aimed specifically at roleplay scenarios, the **Lumimaid 70B** model was tweaked and let loose by the NeverSleep team and details can be scooped from their [announcement page](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b).

**Calling all Roleplayers to a New Digital Realm**: A new roleplaying app granting a free tier has launched, leveraging OpenRouter's multifaceted AI characters, with the creator keen on gathering feedback via [RoleplayHub](https://www.roleplayhub.app/chat).

**Tech Snags and Community Dialogues Tangle in General Channel**: Software patches were applied to mend streaming issues with models like Llama-3, and the release of Mistral-7B v0.3 spewed some confusion due to new vocab/tokenizer—uncertainty lingered about if it should be a distinct model route or a direct route upgrade. Meanwhile, Cohere's Aya initiative garnered attention offering multilingual AI research spanning 101 languages, find out more [here](https://cohere.com/research/aya).

**Economies of Scale Kick in for AI Model Access**: Sharp price reductions have been executed for several models, including a tempting 30% off for `nousresearch/nous-hermes-llama2-13b`, among others. These markdowns are stirring up the market for developers and enthusiasts alike.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Batch Inference for GenAI Pre-Processing**: *Batch inference* is highlighted as a key technique for data pre-processing in **GenAI applications**, with the potential to enhance analysis and querying efficiency. LlamaIndex's integration and more details on the practice can be found [here](https://t.co/vnuvvypZCz).

- **RAG-Powered Job Search Assistant Blueprint**: A **RAG-powered** job search assistant has been created using **@gokoyeb**, **@MongoDB**, and **@llama_index**, demonstrating real-time response streaming and the tutorial is available [here](https://t.co/qsfx4TdvXz).

- **Nomic Embed's Localized Strategy**: **Nomic Embed** now facilitates completely local embeddings along with dynamic inference, blending the benefits of both local and remote embeddings, as expanded upon [here](https://t.co/mPFVQXk5tq).

- **Secure Your Spot for the Tech Meetup**: Engineers interested in joining an upcoming **Tuesday meetup** should note that the slots are running out, with additional details accessible [here](https://t.co/Nx4FiGB8pH).

- **Scaling Up RAG Embedding Models Peaks Interest**: Discussions surfaced around the effectiveness of bigger AI models in improving **RAG embeddings**, without landing on a clear consensus. Reference to the *ReAct algorithm* and advice on custom similarity scores utilizing an `alpha` parameter can be found in the **LlamaIndex documentation** and the discussion of these topics included links to detailed articles and papers.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Podcast with Yi Tay Misses the Boat**: The community wished for spotlights on **scaling laws** during Yi Tay's podcast on Reka/Google, but these insights were missing as the podcast had been pre-recorded.

- **Mistral v0.3 Sparks Mixed Reactions**: **Mistral 7B v0.3 models** have been released, boasting enhancements like a 32K extended vocabulary, new v3 tokenizer, and function calling capabilities, leading to both excitement and criticism [Mistral's newest chapter](https://x.com/Gradio/status/1793367718835659243).

- **Hot Takes on Open-Source AI**: A contentious opinion piece claiming open-source AI poses investment risks and national security concerns ignited debate, with detractors calling out the author for apparent OpenAI favoritism and a narrow perspective.

- **The Quest for a Universal Speech-to-Speech API**: The community discussed workarounds for **OpenAI's yet-to-be-released speech-to-speech API**, pointing to **Pipecat and LiveKit** as current alternatives, with a preference for Pipecat.

- **RAG Gets Real**: Practical applications and challenges of **Retrieval-Augmented Generation (RAG)** were exchanged among members, with a particular reference made to [a PyData Berlin talk](https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html) on RAG deployment in medical companies.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Innovative Prompt Management with VSCode**: Engineers are planning to manage prompts using VSCode to maintain efficiency, including a substantial list of nearly **500k tokens of system prompts** for **Gemini 1.5 Pro**. The ingenuity was met with enthusiasm, and suggestions for additional system prompts were solicited.

- **Favorable Reception for CLI Improvement**: The introduction of a new terminal option `--no_live_response` via a [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1278) was well-received for its potential to smooth out terminal UI issues. Steve235lab's contribution was praised as a noteworthy improvement.

- **Spotlight on Component Teardowns and Replacement Chips**: Members discussed the teardown of **Apple AirPods Pro** and the use of the **ESP32 pico chip** in the Atom Echo for alternative projects, noting the necessary reflashing. Supplementary information such as datasheets provided by ChatGPT was also recognized as beneficial.

- **Tool Praise: M5Stack Flow UI Software**: The [M5Stack Flow UI software](https://flow.m5stack.com) was commended for its support of multiple programming languages and the potential to convert Python scripts to run LLM clients, such as OpenAI, showcasing the flexible integration of hardware and AI-driven applications.

- **Skipping the macOS ChatGPT Waitlist**: A potentially controversial macOS ChatGPT app waitlist workaround from [@testingcatalog](https://x.com/testingcatalog/status/1793347117458636981) was shared, providing a 'cheat' through careful timing during the login process. This information could have implications for software engineers seeking to understand or leverage user behavior and application exploitability.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Challenging the Taylor Takedown**: Members questioned the **efficacy of Taylor series in approximations**, noting that they are only accurate close to the reference point. It was highlighted that range reduction might not be the optimal path to perfect precision, and interval partitioning could offer a better solution.

**Range Reduction Rethink**: The group debated over the use of **range reduction techniques**, suggesting alternatives like reducing to **[0, pi/4]**, and referred to **IBM's approach** as a practical example of interval partitioning found in their [implementation](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/s_sin.c;hb=HEAD).

**IBM's Insights**: An IBM source file was mentioned in a suggestion to address range reduction problems by treating fmod as an integer, viewable [here](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/branred.c;hb=HEAD).

**Mathematical Complexity Calmly Contemplated**: There was a consensus that the computations for perfect accuracy are complex, especially for large numbers, though typically not slow—a mix of admiration and acceptance for the scientific intricacies involved.

**Shape Shifting in ShapeTracker**: The group explored *ShapeTracker* limitations, concluding that certain sequences of operations like `permute` followed by `reshape` lead to multiple views, posing a challenge in chaining movement operations effectively. The utility of tensor masking was discussed, with emphasis on its role in tensor slicing and padding.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Warm Welcome for Global Creatives**: A friendly banter marked the entrance of newcomers into the fold, including a **UI Designer** from Taiwan.
- **Navigating the AI Landscape**: One member gave a crisp direction for interacting with an AI, citing a particular channel and the `@coral` handle for assistance.
- **Cohere Amplifies Multilingual AI Reach**: Cohere's announcement of **Aya 23 models** heralds new advancements, offering tools with [8 billion and 35 billion parameters](https://huggingface.co/CohereForAI/aya-23-35B) and touting support for a linguistic range encompassing 23 languages.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GraphRAG Gains Traction for Graph-Modeled Info**: Members discussed that **GraphRAG** shines when source data is naturally graph-structured, though it may not be the best choice for other data formats.
  
- **PySpark Speeds Up Embedding Conversions**: AI engineers are experimenting with **PySpark pandas UDF** to potentially enhance the efficiency of embeddings processing.

- **Challenges with Persistence in Pinecone**: A shared challenge within the community focused on the inefficiencies of **persistence handling** versus frequent instance creation in Pinecone, with dissatisfaction expressed regarding mainstream solutions like *pickle*.

- **APIs and Instruction Tuning in the Spotlight**: Upcoming event "How to Develop APIs with LangSmith for Generative AI Drug Discovery Production" set for May 23, 2024, and a new [YouTube video](https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt) explains the benefits of Instruction Tuning for enhancing LLMs' adherence to human instructions.

- **Code Modification and Retriever Planning**: Engineers are currently seeking efficient retrievers for planning code changes and techniques to prevent LLMs from undercutting existing code when suggesting modifications.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral Gets a Boost in Vocabulary and Features**: The newest iterations of [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) now boast an **extended vocabulary of 32768 tokens**, **v3 Tokenizer support**, and function calling capabilities, with installation made easy through `mistral_inference`.

- **Enhancements to Mistral 7B Paired with Community Approval**: The launch of the [Mistral-7B instruct version](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) has received a casual thumbs up from Eldar Kurtic with hints of more improvements to come, as seen in a [recent tweet](https://twitter.com/_EldarKurtic/status/1793407795909599325?t=zhtA3A5nq23HfUBkt441mQ&s=19).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **GaLore and InRank Break New Ground**: A session with Jiawei Zhao delved into **Gradient Low-Rank Projection (GaLore)** and **Incremental Low-Rank Learning (InRank)** which offer reductions in memory usage and enhancements in large-scale model training performance.

- **Event Sync Woes**: An inquiry was made about integrating an event calendar with Google Calendar, highlighting a need to track upcoming discussions to avoid missing out.

- **Image Recon with ImageMAE Marks Scalability Leap**: The ImageMAE paper was shared, presenting a scalable self-supervised learning approach for computer vision using masked autoencoders, with impressive results from a vanilla ViT-Huge model achieving 87.8%.

- **Community Spirits High**: A member voiced their appreciation for the existence of the channel, finding it a valuable asset for sharing and learning in the AI field.



> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
