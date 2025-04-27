---
id: f8dc17bb-79ec-43be-b087-a224f36f63c5
title: GitHub Copilot Strikes Back
date: '2024-10-30T01:05:11.702248Z'
original_slug: ainews-github-copilot-strikes-back-3402
description: >-
  **GitHub's tenth annual Universe conference** introduced a **Multi-model
  Copilot** integrating **Anthropic's Claude 3.5 Sonnet**, **Google's Gemini 1.5
  Pro**, and **OpenAI's o1-preview** models, allowing developers to choose from
  multiple AI providers. The event also showcased **GitHub Spark**, an AI-native
  tool for building natural language applications with features like
  deployment-free hosting and persistent data storage. Additionally, GitHub
  updated Copilot with multi-file editing, custom instructions, and new agents
  for code review and security autofix. Meanwhile, **Weights & Biases** launched
  **Weave**, supporting multimodal AI observability including audio tracing.
  Twitter highlights included discussions on **tinygrad optimization** and
  **Gemini Flash-8B's** cost efficiency. *"Utilizing a creativity feedback loop,
  users start with an initial prompt, see live previews of their app as it’s
  built,"* and *"your LLMs aren't just text-only anymore - so why should your
  observability be?"*
companies:
  - github
  - anthropic
  - google-deepmind
  - openai
  - weights-biases
models:
  - claude-3-sonnet
  - gemini-1.5-pro
  - o1-preview
  - gemini-flash-8b
topics:
  - multi-model-integration
  - natural-language-applications
  - model-picker-ui
  - multi-file-editing
  - custom-instructions
  - ai-native-development
  - deployment-free-hosting
  - persistent-data-storage
  - model-prompting
  - code-review-agents
  - security-autofix
  - multimodality
  - audio-tracing
  - tinygrad-optimization
  - price-performance-ratio
people:
  - cassidy_williams
  - swyx
  - fchollet
  - rohanpaul_ai
  - jxmnop
---


**GitHub may be all you need for AI-Native coding.**

> AI News for 10/28/2024-10/29/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**231** channels, and **2681** messages) for you. Estimated reading time saved (at 200wpm): **279 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

GitHub's [tenth annual Universe conference was today](https://www.youtube.com/watch?v=5ov2NYBdGSw): 

![image](https://gist.github.com/user-attachments/assets/d3fd1ac8-1d34-499a-ad11-febd7502ff12)

And it brought a raft of notable announcements ([full blogpost here](https://github.blog/news-insights/product-news/universe-2024-previews-releases/)): mostly being GitHub's takes on popular code AI tools.

1. **Multi-model Copilot**: adding Anthropic’s Claude 3.5 Sonnet, Google’s Gemini 1.5 Pro, and OpenAI’s o1-preview in a new model picker UI. Copilot's base model has moved from Codex, GPT3.5, GPT4, 4o, and 4o-mini, but for the first time developers get to choose models from other companies including Google. This was big enough to [reach mainstream media today](https://news.ycombinator.com/item?id=41985915) and one can't help but tie this story with [reports of a "fraying" Microsoft-OpenAI partnership](https://news.ycombinator.com/item?id=41878281).

![image](https://gist.github.com/user-attachments/assets/b8a2b9c1-5e49-4eb4-aafc-4ecb527cb222)

Cassidy Williams also demoed the [new multi-file editing capability of Copilot](https://x.com/ashtom/status/1851316495336554663) and [a custom instructions file](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot) - analogous to **the Composer and .cursorrules features of Cursor.**

2. [**GitHub Spark**](https://githubnext.com/projects/github-spark): "*the AI-native tool to build applications entirely in natural language. Sparks are fully functional micro apps that can integrate AI features and external data sources without requiring any management of cloud resources.*" basically their v0 and bolt.new and Claude Artifacts competitor, complete with **deployment-free hosting, themable design system, persistent data storage, and integrated model prompting**.

> "Utilizing a creativity feedback loop, users start with an initial prompt, see live previews of their app as it’s built, easily see options for each of their requests, and automatically save versions of each iteration so they can compare versions as they go."

![image](https://gist.github.com/user-attachments/assets/4118995e-d456-4fad-a064-b60035f9b519)


The presenters also discussed the latest [GitHub Models](https://docs.github.com/en/github-models) (now off waitlist), and last year's big launch, Copilot [Workspace and Code Reviews](https://github.blog/changelog/2024-10-29-github-copilot-code-review-in-github-com-public-preview/) (now with 2 more agents: Brainstorm and Build/Repair, to the existing 3 Spec/Plan/Implement agents, with a new VSCode extension) and [security Autofix](https://github.blog/changelog/2024-09-18-now-available-for-free-on-all-public-repositories-copilot-autofix-for-codeql-code-scanning-alerts/) updates.

---

**[This issue brought to you by Weights & Biases]!**: Your LLMs aren't just text-only anymore - so why should your observability be?

Weave from Weights & Biases [now supports](https://usewb.link/swyx-docs) audio tracing alongside text, images and other modalities. With just 3 lines of code, track every input, output and metadata across your multimodal AI stack.

Try it yourself in [our interactive Colab notebook](https://usewb.link/Hw0HGuU)!

> swyx commentary: this notebook looks short, but IMO the gold is the 19 cells hidden under "Advanced Usage: Realtime Audio API with Weave"! You wouldn't expect a normal LLM Ops product to have updated to support the OpenAI Realtime API so soon, but it looks like the WandB team have been cooking.

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

**AI Development and Industry Trends**

- **Tinygrad Optimization**: [@jxmnop](https://twitter.com/jxmnop/status/1850975062905516191) noted that tinygrad is **focusing on reducing lines of code** compared to PyTorch, resulting in a codebase that's **growing horizontally** and becoming **borderline unreadable to humans**.

- **AI Model Capabilities**: [@fchollet](https://twitter.com/fchollet/status/1850967744386384098) pointed out that the **current low adoption rate of GenAI** indicates potential for growth, contrary to claims of 40% adoption. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1850956229327405414) highlighted Gemini Flash-8B's strong price-performance ratio, with **$0.0375 per million input tokens** and **$0.15 per million output tokens**.

- **AI Infrastructure**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1851039496105836605) shared details about xAI's Colossus supercomputer, featuring **100,000 NVIDIA Hopper GPUs** and plans to double to 200,000. The system uses NVIDIA Spectrum-X Ethernet platform, supporting **800Gb/s port speeds**.

**AI Applications and Tools**

- **Perplexity Spaces Update**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1850950654271111483) announced improvements including **5 file uploads for free users**, enhanced custom instructions, detailed Space overview cards, and support for Markdown files.

- **RAG Developments**: [@togethercompute](https://twitter.com/togethercompute/status/1850939031301099919) shared an open-source implementation of Contextual RAG using Llama models, involving context generation, hybrid search, and reranking. [@llama_index](https://twitter.com/llama_index/status/1851031828125401301) introduced advanced RAG systems using MLflow and LlamaIndex Workflows for flexible orchestration and evaluation.

- **AI Agents**: [@omarsar0](https://twitter.com/omarsar0/status/1850897901817364658) launched a course on AI Agents, covering fundamentals and practical tips for building agentic AI systems. [@LangChainAI](https://twitter.com/LangChainAI/status/1850930775589519633) shared a comprehensive repository for agent development using LangGraph.

**AI Research and Model Updates**

- **Model Comparisons**: [@ajayj_](https://twitter.com/ajayj_/status/1850994244095525228) reported that Genmo Mochi 1, an open-source video generation model, outperforms Runway, Kling, Luma, and Pika models according to community votes.

- **Optimization Techniques**: [@giffmana](https://twitter.com/giffmana/status/1850988191618326950) highlighted the effectiveness of **sigmoid loss with bias** in improving model performance.

- **Context Window Expansion**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1850843153299116171) mentioned ongoing work on **100mn context window** LLMs and research on 1-Bn context windows, potentially impacting the future of RAG.

**AI Ethics and Societal Impact**

- **AI Adoption Concerns**: [@ylecun](https://twitter.com/ylecun/status/1850866813430911066) criticized the superiority complex of some tech leaders, warning against treating followers as "low IQ" and expecting blind submission.

- **AI Productivity Impact**: [@random_walker](https://twitter.com/random_walker/status/1850954894066548763) shared skepticism about claims of significant productivity boosts from AI, noting only a 1% increase despite 3% usage.

- **AI in Education**: [@svpino](https://twitter.com/svpino/status/1850974043874476365) cautioned against overestimating AI's capabilities in building SaaS businesses, emphasizing that AI is a tool, not a complete solution.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Optimizing LLM Inference on Consumer Hardware**

- **[What's the best way to run llama on a local GPU (low-end RTX3000)? Interested in both calling it from within Python as well as a GUI. The space evolves so quickly, so I'd love an up-to-date recommendation! Thanks](https://i.redd.it/fjoj2aym3hxd1.png)** ([Score: 39, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1gdymkw/whats_the_best_way_to_run_llama_on_a_local_gpu/)): For running **Llama models** on **low-end RTX 3000 GPUs**, current recommendations include using **llama.cpp** or **text-generation-webui** for a GUI interface, and **transformers** library with **bitsandbytes** for Python integration. These methods allow for efficient **quantization** and **inference** on consumer-grade hardware, though specific performance may vary based on model size and available VRAM.
  - **Ollama** with **Open webui** is recommended, with some users running it via **Docker** container and making **HTTP calls** for integration. The author suggests [Harbor](https://github.com/av/harbor) for a comprehensive **LLM stack** using Docker.
  - Users employ various interfaces: **mikupad** for writing with **llama.cpp**, **TabbyAPI** with **LLama-3.1** or **3.0** models integrated into **silly tavern**, and **Lm studio** or **Aya** for GUI and **OpenAI API** compatibility.
  - Some prefer custom setups, like running **llama.cpp** in scripts for pure writing, emphasizing the importance of **alternative token selection** which may be absent in other UI options.
- **[AI scores of mobile SoCs by brand, year and segment](https://i.redd.it/28am1cfkcgxd1.jpeg)** ([Score: 43, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1gdwm4o/ai_scores_of_mobile_socs_by_brand_year_and_segment/)): The post analyzes **AI performance benchmarks** of mobile SoCs from [ai-benchmark.com](https://ai-benchmark.com/ranking_processors.html), revealing significant **performance gaps** between flagship and high-end segments. Notable findings include the **Snapdragon 7+** series outperforming its branding, **Dimensity's** substantial AI performance increase in recent generations, and the **four-year-old Snapdragon 8 Gen 1** still surpassing newer Snapdragon 7 series, 8s Gen3, and most Dimensity processors, with the **A17 Pro** scoring **3428**, just below the Snapdragon 8 Gen 3.
  - Users discussed running **large language models** on phones, with interest in models like **16B deepseek v2 Lite MoE** and **Llama 3.1 8b**. The **ZTE Z60 Ultra** with up to **24GB RAM** was mentioned as capable of running **12B models**.
  - Debate arose over the relevance of the benchmark's tested models, with some arguing that **TFLOPS**, **TOPS**, and **memory bandwidth** specs are more informative for real-world AI applications on phones than scores based on models like **Inception V3**.
  - Interest was expressed in the state of **Mediatek chipsets** for AI tasks, particularly regarding **GPU and NPU functionality**. The post highlighted **Dimensity's** recent improvements in AI performance.
- **[Updated with corrected settings for Llama.cpp.  Battle of the Inference Engines. Llama.cpp vs MLC LLM vs vLLM. Tests for both Single RTX 3090 and 4 RTX 3090's.](https://www.reddit.com/gallery/1ge1ojk)** ([Score: 75, Comments: 51](https://reddit.com//r/LocalLLaMA/comments/1ge1ojk/updated_with_corrected_settings_for_llamacpp/)): **Llama.cpp**, **MLC LLM**, and **vLLM** were benchmarked for LLM inference on consumer GPUs, specifically testing with a **single RTX 3090** and **four RTX 3090s**. The post provides updated results with corrected settings for Llama.cpp, comparing the performance of these three inference engines across different GPU configurations.
  - **Llama.cpp** performance improved significantly after correcting settings, reaching **50-51 tokens/second** for single GPU tests and **15 tokens/second** for 4x GPU tests. The community suggested adding **exllama** to future benchmarks and exploring quantized model comparisons.
  - A [blog post](https://blog.mlc.ai/2024/10/10/optimizing-and-characterizing-high-throughput-low-latency-llm-inference) was shared, detailing benchmarks for **multiGPU scaling**, **concurrent requests**, and **speculative decoding**. Users expressed interest in how **MLC-LLM** scales across 1-4 GPUs, with one user reporting **25 tokens/second** on 1 GPU and **34 tokens/second** on 2 GPUs using **MI60** cards.
  - Discussions focused on **PCIE bandwidth usage**, with tests showing surprisingly low utilization (**0.1 MB/s**) during tensor parallel inference. Users also debated the choice of **FP16** for benchmarks, with some preferring **Q4** or **Q8** quantization for practical use cases.


**Theme 2. Advancements in Open-Source LLMs for Creative and Uncensored Use Cases**

- **Three Llama 3.2 Models enhanced, at 7B each for creative uses - uncensored.** ([Score: 44, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1geio97/three_llama_32_models_enhanced_at_7b_each_for/)): Three enhanced **Llama 3.2 7B models** have been released for creative and uncensored use, each expanded to **67 layers** and **606 tensors**. The models, available on Hugging Face, are rated on a "de-censor" scale from 1-10 and feature improved **instruction following**, **nuance**, **emotion**, and **prose depth**, with censorship and bias controllable via prompts.
  - **Frankenstein models** are criticized as often being "lobotomized" and underperforming, with users suggesting using **full-size models** with adjusted settings instead. The model creator defends his approach, citing **45 examples** of improvements and explaining his unique methods for building and testing models.
  - User **export_tank_harmful** praises the creator's work, particularly mentioning **MN-Dark-Planet-TITAN-12B** and **L3-Dark-Planet-8B** models. They suggest including the creator's **Hugging Face name** in Reddit posts for credibility and express support for continued **abliteration** efforts.
  - Discussion on model availability for **ARM devices**, with the creator clarifying that ARM-optimized models have filenames ending in **Q4_0_4_8.gguf**. Currently, **only 3 versions** are supported by **llamacpp** for ARM optimization.
- **LLM Recommendation for Erotic Roleplay** ([Score: 48, Comments: 61](https://reddit.com//r/LocalLLaMA/comments/1ge2fzf/llm_recommendation_for_erotic_roleplay/)): The post seeks recommendations for **Large Language Models (LLMs)** specifically for **erotic roleplay**, listing several options with a focus on **DarkForest V2** and **backyardai/Midnight-Rose-70B-v2.0.3-GGUF** as top contenders. The author also mentions other models like **Stheno**, **Lyra 12B V4**, **TheSpice-8b**, and various others ranging from **8B to 72B parameters**, but considers them potentially weaker for this specific use case.
  - **ArsNeph** recommends newer models, highlighting **L3 Stheno 3.2 8B**, **Magnum V4**, **UnslopNemo 12B**, **Mistral Small 22B** and its finetunes like **Cydonia**. For larger models, they suggest **Midnight Miqu 1.5 70B**, **Euryale 2.1 70B**, and **New Dawn Llama**.
  - Several users endorse **Midnight Rose** and **Midnight Miqu** as top choices for erotic roleplay. **TheLocalDrummer** mentions that some users prefer **Behemoth v1.1** over Midnight Miqu, while others recommend trying [NemoMix-Unleashed-12B](https://huggingface.co/MarinaraSpaghetti/NemoMix-Unleashed-12B) and [EVA-Qwen2.5-72B-v0.0](https://huggingface.co/EVA-UNIT-01/EVA-Qwen2.5-72B-v0.0).
  - Users suggest exploring **Gemma-2-27B** despite its censorship, and **Mistral-Small-22B-ArliA


**Theme 3. Innovations in LLM Tooling and Infrastructure**

- **We just Open Sourced Promptwright: Generate large synthetic datasets using a local LLM** ([Score: 63, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1ge9192/we_just_open_sourced_promptwright_generate_large/)): **Promptwright**, an open-source **Python library** for generating synthetic datasets using **local LLMs** via **Ollama**, has been released. It offers a simple interface for dataset generation, configurable instructions and system prompts, **JSONL** output format, and direct integration with **Hugging Face Hub**, allowing users to process thousands of samples locally without API costs or rate limits while maintaining data privacy.

- **Mistral.rs v0.3.2 gets a 26% Metal performance boost and PyPI wheels!** ([Score: 62, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1ge9dc7/mistralrs_v032_gets_a_26_metal_performance_boost/)): **Mistral.rs v0.3.2** introduces simplified installation via PyPI wheels for various platforms (**Metal**, **CUDA**, **Apple Accelerate**, **Intel MKL**, and plain CPU) and achieves a **26% performance boost** for Metal decoding through optimized MLX attention kernels. The update also includes **CUDA improvements** with a Marlin GPTQ kernel and FP8 quantization, along with support for models like **Llama 3.2 Vision**, with links provided to the [GitHub repository](https://github.com/EricLBuehler/mistral.rs), [Python package documentation](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/API.md), and a [UQFF model collection](https://huggingface.co/collections/EricB/uqff-670e4a49d56ecdd3f7f0fd4c) for prequantized models.

- **[Retrieval system extending any off-the-shelf LLM to 1B (billion) context on a standard CPU during inference time:](https://www.reddit.com/gallery/1gejpg2)** ([Score: 63, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1gejpg2/retrieval_system_extending_any_offtheshelf_llm_to/)): A new **retrieval system** has been developed that can extend the context length of **any off-the-shelf Large Language Model (LLM)** to **1 billion tokens** during inference time, using only standard CPUs. This system, detailed in a [Zyphra blog post](https://www.zyphra.com/post/reaching-1b-context-length-with-rag) and an [arXiv paper](https://arxiv.org/abs/2409.01666), significantly expands the capability of LLMs to process and understand vast amounts of information without requiring specialized hardware.
  - The title's claim of **"1B context length"** is criticized as **clickbait**, with users noting it refers to tokens in a vector store, not actual inference. **Inference on 1M context** for an **8B model** would take **~3000s** on an **A100 GPU**.
  - Users humorously extend the concept, suggesting even larger context lengths like **100B tokens** or **100 Petabytes** (referencing **Google's index size**) to highlight the arbitrary nature of such claims.
  - There's interest in **benchmarks beyond hash chain retrieval** and potential applications, such as creating **small LMs** (e.g., **1B models**) that load necessary knowledge via RAG, potentially outputting **thousands of tokens per second**.


**Theme 4. Challenges in AI Document Understanding and Real-World Applications**

- **How I used vision models to help me win at Age Of Empires 2.** ([Score: 327, Comments: 51](https://reddit.com//r/LocalLLaMA/comments/1ge6fvw/how_i_used_vision_models_to_help_me_win_at_age_of/)): The author developed **WololoGPT**, an *AI-based coach* for **Age of Empires 2**, using **vision models** and **LLMs** to provide real-time gameplay advice, including resource management and enemy countering strategies. The project, built using **Claude 3.5** and **Gemini Flash** for vision processing, is open-source and available on [GitHub](https://github.com/tony-png/WololoGPT), with a [video demo](https://www.youtube.com/watch?v=ZXqVKgQRCYs) and downloadable executable on the [official website](http://www.wolologpt.com).
  - **Echo9Zulu-** suggests developing a system for recording application data, viewing WololoGPT as an opportunity to build **valuable training data** on model interpretations of game events. They recommend studying model behavior using **AoE2** as a template, particularly focusing on how models handle **fog of war** effects on strategy.
  - The project is praised for its potential to push the **state-of-the-art in vision model applications**, with the commenter noting that current literature is limited on such use cases. They recommend passively recording data to capitalize on this opportunity.
  - **WololoGPT** is described as a "cool build" that provides a gameplay boost without feeling like complete cheating. The developer confirms it has improved their gameplay, describing it as a "little boost."

- **Document understanding is very very hard: an illustration** ([Score: 34, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1gekd53/document_understanding_is_very_very_hard_an/)): The post illustrates **document understanding difficulties** for LLMs using a **San Francisco pool schedule** example. The author challenges readers to extract **recurring lap swim periods** from a **single-page flyer**, with bonus tasks including generating an **ical (ics) format** and handling **holidays**, noting that models often miss **Mondays** and misinterpret **Wednesday lap swim times**. Despite some impressive capabilities, the author concludes that even **advanced LLMs struggle** with tasks a **six-year-old** can accomplish, cautioning against premature deployment of document understanding in production environments.
  - Users critiqued the **pool schedule layout**, noting its poor design and inconsistencies. One commenter highlighted that such **goofy layouts** are common in professional settings, citing **M&A Diligence Checklists** as an example.
  - A user successfully extracted the schedule using **Chat 4.0** and created a **Python script** to generate an **ical file** in **5 minutes**. The script handles recurring events but doesn't account for holidays.
  - **Gemini 1.5 Pro** in **Aistudio** correctly extracted most of the schedule, including the tricky **Wednesday lap swim times**, but added an non-existent Sunday evening slot. Users discussed multi-step reasoning and the challenges of handling different image resolutions with vision models.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Releases and Capabilities**

- **Updated Phi-3 Mini with function calling**: Rubra AI released an updated Phi-3 Mini model [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/), competitive with Mistral-7b v3 (/r/LocalLLaMA).

- **OpenAI's o1 reasoning model**: OpenAI CFO Sarah Friar says [lawyers report the new o1 reasoning model can do the work of a $2000/hour paralegal](https://v.redd.it/t25hddpblkxd1) (/r/singularity).

**AI Applications and Demonstrations**

- **AI-assisted multi-arm robot for apple picking**: A [video demonstrates an AI-assisted multi-arm robot](https://v.redd.it/552w8berqhxd1) that can identify and pick ripe apples (/r/singularity).

- **Realistic facial animation using Stable Diffusion**: A developer is working on a [realistic facial animation system for Meta Quest using Stable Diffusion](https://v.redd.it/ut9246li3mxd1), running at 90fps on the Quest 3 (/r/StableDiffusion).

- **Robot hand with tactile sensing**: Robot Era introduced its [first-generation XHAND](https://v.redd.it/2igc50imhixd1), featuring 12 degrees of freedom and tactile sensing in each finger (/r/singularity).

- **Robots performing beauty services**: A [video shows robots doing nails and eyelashes in LA](https://x.com/esthercrawford/status/1850681223770947869), demonstrating automation in previously human-dominated services (/r/singularity).

**AI Policy and Infrastructure**

- **US government push for AI infrastructure**: National Security Advisor Jake Sullivan states [the US needs to build 10s or even 100s of gigawatts of energy infrastructure](https://v.redd.it/28bxofivbhxd1) to power AI data centers or risk falling behind competitors (/r/singularity).

**AI Impact and Societal Discussion**

- **Discussions on AI's impact on employment**: Multiple posts discuss the potential impact of AI on jobs, including [comparisons to the decrease in horse populations after the invention of cars](https://i.redd.it/qhs0yi12whxd1.jpeg) (/r/singularity).

- **Public perception of AI advancements**: A post discusses [how people react to being shown ChatGPT](https://www.reddit.com/r/singularity/comments/1ge4eh0/showed_my_dad_chat_gpt_and_he_is_literally/), with some becoming very interested and others remaining unimpressed (/r/singularity).

**Memes and Humor**

- A post humorously suggests [using Stable Diffusion to create disinformation about the history of bathtubs](https://v.redd.it/uuyp7mcq2nxd1) (/r/StableDiffusion).


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: AI Model Releases Shake Up the Scene**

- **Stable Diffusion 3.5: Big Power in Medium Size!**: Stability.ai unleashes [Stable Diffusion 3.5 Medium](https://stability.ai/news/introducing-stable-diffusion-3-5), a 2.5 billion parameter model that runs on just **9.9 GB of VRAM**, democratizing high-quality image generation.
- **Moondream Bets Small Models Can Pack a Punch**: [Moondream](https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/) raises **$4.5 million** to prove that smaller AI models are just as effective, shifting the industry's focus from gigantic architectures.
- **GitHub Copilot Supercharges with Claude and Gemini**: GitHub's Copilot integrates [Claude 3.5 Sonnet](https://github.blog/news-insights/product-news/bringing-developer-choice-to-copilot/) and [Google's Gemini 1.5 Pro](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot), giving developers an AI power-up.

**Theme 2: AI Tooling Gets a Turbo Boost**

- **Unsloth Slays Complexity with Gradio UI**: An innovator rolls out a [Gradio app](https://huggingface.co/blog/merve/quantization) that simplifies model training with Unsloth, making AI accessible even for no-code enthusiasts.
- **ThunderKittens Roar with Lightning Speed**: The much-awaited [ThunderKittens 0.000000002](https://arxiv.org/abs/2410.20399) drops, boasting **6-14x faster linear attentions** and outpacing FA3 in attention backward passes.
- **Developers Tinker Triton Kernels for Speed**: Engineers discuss optimizing Triton kernels, finding that multiple kernels outperform single ones, and uncover challenges with BF16 casts.

**Theme 3: AI Privacy and Security Take Center Stage**

- **PAPILLON Flutters In to Protect Privacy**: Researchers debut [PAPILLON](https://arxiv.org/abs/2410.17127), hitting **85.5% quality** with just **7.5% privacy leaks**, blending local and cloud LLMs securely.
- **ChatGPT Typo Tantrums Baffle Users**: ChatGPT starts spewing typos and gibberish, leaving users scratching their heads about the sudden drop in output quality.
- **Apple Throws Down $1M Hackathon Gauntlet**: Apple dares hackers to breach their AI servers with a whopping [$1 million bounty](https://x.com/culturecrave/status/1850781293166067999?s=46), sparking debates on AI security.

**Theme 4: AI Jobs Abound on New Platforms**

- **Cracked Engineers Breaks Ground in Tech Hiring**: The freshly launched [Cracked Engineers](https://www.crackedengineers.com/) connects AI talent with top startups, already partnering with **Weaviate**, **UnslothAI**, and more.
- **AI Startups on the Hunt for Top Talent**: Companies like **Unsloth AI**, **Julius AI**, and **Jimini AI** are actively recruiting, offering amazing opportunities for those ready to dive into cutting-edge AI.
- **Job Seekers Rejoice: Tailored Newsletters Incoming**: Cracked Engineers announces a weekly tech jobs newsletter, letting subscribers tailor content with tags like **CUDA**, **MLOps**, and **Software Engineering**.

**Theme 5: AI Community Buzzes with Events and Insights**

- **LLM Agents Hackathon Hits the Ground Running**: With over **1,000 innovators** registered, the [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/) dangles over **$200K in prizes** across five thrilling tracks.
- **OpenAI's CFO Says *"AI Isn't Experimental Anymore!"***: In a candid [interview](https://youtu.be/eCqFgVqWbEs), OpenAI CFO **Sarah Friar** proclaims that AI has gone mainstream, infiltrating banks and fintech daily.
- **Meta Sets Sights on Its Own AI Search Engine**: Meta's web crawlers hint at a new [AI-powered search engine](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report), aiming to cut ties with Google and Microsoft.

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Clem Introduces Himself to the Community**: Clem, co-founder and CEO at Hugging Face, expressed excitement about using Discord to connect with community members actively. *I can't wait to interact with all of you*, emphasizing a strong intent to engage.
  
  - He also promoted an upcoming live workshop, encouraging members to share ideas on expanding its visibility and participation via this [link](https://streamyard.com/watch/JS2jHsUP3NDM).
- **Frustrations with TensorFlow**: Many members aired frustrations with **TensorFlow**, citing disabling GPU support on Windows and complex documentation issues, often preferring to transition to **PyTorch** for faster developments.
  
  - Shared experiences reflected a common sentiment of dissatisfaction with TensorFlow’s bugs and lack of support within the community.
- **Exploration of Hemp Nanosheets**: **Hemp-derived carbon nanosheets** show potential as cost-effective alternatives to graphene in energy storage, with feasibility established at **$500 per ton** by Dr. David Mitlin's research.
  
  - This sparked discussions on military and aerospace applications, indicating a growing interest in alternative materials suitable for high-tech industries.
- **Swin Transformer v2 Discussion**: Members explored using **Swin Transformer v2** for handling image-like data cubes, with discussions on adapting architecture for unique input shapes.
  
  - One user mentioned utilizing data cubes instead of traditional images, prompting conversations about necessary architectural adjustments.
- **LangChain SQL Agent Resource Sharing**: A GitHub notebook detailing a **LLaMA2 SQL chat** was shared as a resource for developing **context-aware reasoning applications** with **LangChain SQL Agent**.
  
  - This resource is positioned to assist users in enhancing their implementations, illustrating the community's focus on utilizing modern technologies for NLP tasks.

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gradio UI Tool Simplifies Model Training**: A user created a [Gradio app](https://huggingface.co/blog/merve/quantization) that streamlines the training of models with **Unsloth**, making it easier to adjust settings and upload models to Hugging Face.
  
  - This enhancement aims to assist nocode users, significantly improving the accessibility of AI model training.
- **AI Job Opportunities from Unsloth**: Unsloth is spotlighting a hiring campaign through [Cracked Engineers](https://www.crackedengineers.com/), aiming to attract tech talent in AI fields.
  
  - Community members are encouraged to explore job listings on the platform while utilizing it for job tracking.
- **FP8 Fine-Tuning for Enhanced Training Speed**: There's ongoing discussion about the adoption of **FP8** for training within Unsloth, suggesting potential **speed improvements**.
  
  - The community raised questions about its implementation specifics, particularly in relation to base weights and LoRA.
- **Frustrations with Educational Systems**: Members discussed feelings of time wasted in school, with one expressing a desire to *make a difference* instead.
  
  - This sentiment resonated, as others reflected on how personal experiences shape educational perspectives.
- **Insights on Optimizer CPU Offload**: Discussion centered on the potential of **Optimizer CPU Offload** to improve efficiency in **low-bit** training frameworks.
  
  - By shifting operations to the CPU, models can achieve **faster training times** and optimize resource use.

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3.5 Medium Model Launch**: The **Stable Diffusion 3.5 Medium** model is available for free commercial use with **2.5 billion parameters**, running on consumer hardware with just **9.9 GB of VRAM**.
  
  - This launch aims to broaden access to AI by ensuring compatibility even with low-end devices, transforming the landscape for creators.
- **Image Quality Hits New Heights**: Users confirmed that **Stable Diffusion 3.5 Medium** excels in generating images over 1MP, outperforming the **3.5 Large** variant in **prompt adherence and quality**.
  
  - However, once images exceed 2MP, the model starts to struggle, indicating limits to its scaling capabilities.
- **GPU Price Wars Rumble On**: Current market trends show **3090** GPUs priced similar to **7900 XTX**, with used 3090s hovering around **$690**.
  
  - Discussions included comparisons of GPU performance for AI workloads versus gaming, emphasizing the shifting dynamics of hardware affordability.
- **Sana Autoencoder Mixed Reactions**: The **Sana** autoencoder promises efficient training and compression but received mixed feedback on its image quality results.
  
  - Some users remain skeptical, indicating a need for further validation on models leveraging this technology.
- **Switching UIs for Enhanced User Experience**: Users explored switching from **A1111** to **ComfyUI**, with some experimenting with **SwarmUI** for a streamlined image generation process.
  
  - Conversations highlighted preferences for different interfaces and optimizing settings like **steps** and **cfg** to improve prompt adherence.

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Newsletters for Developers**: A member highlighted the need for technical AI newsletters, moving away from consumer-focused hype and recommended [SemiAnalysis](https://link.to.semi-analysis) for its GPU insights.
  
  - This reflects a desire for more substantive resources among engineers who seek serious discussions on AI.
- **Finetuning Hermes 3 for Roleplay Bots**: A user explored whether finetuning **Hermes 3** could enhance a roleplaying bot's mimicry of **character.ai**, while another suggested leveraging prompts for the same outcome.
  
  - This discussion underlines the community's interest in optimizing AI for complex character interactions.
- **Meta Releases Layer Skip Code**: Meta launched [Layer Skip](https://go.fb.me/s8lary) to improve LLM efficiency, providing the inference code and fine-tuned checkpoints.
  
  - This release aims to spark new research into **AI optimization** methods and **interpretability**.
- **GitHub Copilot Expands Model Choices**: Major updates for GitHub Copilot include the addition of **Claude 3.5 Sonnet** and **Gemini 1.5 Pro**, offering developers broader model selections.
  
  - This shift may empower **Anthropic** in the competitive landscape of AI.
- **Microsoft and OpenAI's Complicated Relationship**: Conversations indicated that Microsoft is exploring alternatives to OpenAI due to fears over dependency and risks associated with AGI declarations.
  
  - Members emphasized the importance of diversifying AI partnerships for strategic stability.

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Join the Curators Program!**: Perplexity Team is actively seeking its first cohort of **Curators** to contribute to the **Discover feed**, engaging millions of users. If you enjoy crafting **Pinterest boards** or editing **Wikipedia pages**, you can [apply here](https://perplexity.ai/curators).
  
  - Curators will be responsible for creating **Pages** that **inspire** and **inform** users directly within the **Perplexity** product.
- **Grok 2 now available for Pro users**: Perplexity AI announced that **Grok 2** is now accessible for Pro users, allowing them to set it as their default model in settings. Some users are curious if Grok 2 will remain uncensored, though its improvements seem limited.
  
  - The announcement stirred discussions, with skepticism about any significant advancements over previous iterations.
- **Merchandise Launch Announcement**: Perplexity AI is launching a merchandise line called **Perplexity Supply**, with the first drop set for tomorrow at **9 AM Pacific Time**. Their tagline emphasizes a brand 'made for the curious,' hinting at an engaged community.
  
  - Community excitement is palpable, as users anticipate collectibles and fashion items tied to the brand.
- **NASA Generates $76B for US Economy**: A recent report claims that **NASA** has contributed approximately **$76 billion** to the U.S. economy, a reflection of its various projects and innovations. This emphasizes NASA's impact beyond space exploration, reinforcing its role in economic growth.
  
  - The data suggests significant **returns on investment** from public funds, making a compelling case for continued support.
- **Getting Smart on Advancements in Photonic Computing**: Discussions highlighted advancements in **photonic computing** and its implications for the **cybersecurity** landscape. These technologies are predicted to transform how data is processed and secured.
  
  - Members shared fresh insights, indicating a growing interest in integrating photonic capabilities into existing frameworks.

 

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **BYD Aims for Auto Industry Domination**: A video discusses how **BYD**, a Chinese electric vehicle powerhouse, is poised to disrupt competitors like **Tesla** through aggressive global expansion and dealership openings, as highlighted in [this video](https://www.youtube.com/watch?v=VgAGSbreEMI).
  
  - The discussion underscores BYD's innovative strategies intended to significantly impact the automotive market.
- **NotebookLM Enhances Staff Resource Accessibility**: A user implemented **NotebookLM** as a staff resource guide, integrating an employee handbook and FAQs to streamline internal queries, but noted inconsistency in URL generation from external links.
  
  - This feedback suggests a need for further refinements in document integration within the platform.
- **Spanish Podcast Generation Faces Challenges**: Users reported difficulties in generating Spanish podcasts with **NotebookLM**, having initially succeeded with two episodes, leading to calls for effective solutions.
  
  - Concerns were raised about underlying language processing issues affecting Spanish text generation, indicating a gap for necessary improvements.
- **Exploring Open Source Alternatives to NotebookLM**: Community members are evaluating **NotebookLlama**, an open-source alternative that utilizes Meta's technology, but there's skepticism regarding the site's credibility as discussed in the [Notebook Llama link](https://www.notebookllama.ai/).
  
  - Participants debated the benefits of open-source solutions, pointing to possible DNS issues and registration legitimacy.
- **Real-Time Avatars Revolutionize Podcasting**: The integration of **Simli** for real-time avatars in podcasts has sparked interest, allowing for synchronized visuals using audio diarization to enhance viewer engagement.
  
  - This proof of concept underlines exciting potential for dynamic presentation styles in podcasts.

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Unsloth Kernels enhance LLM fine-tuning**: A member inquired about guides for [unsloth kernels](https://github.com/unslothai/unsloth), which significantly upgrade LLM performance and memory efficiency, fine-tuning with **Llama 3.2**, **Mistral**, and others up to **2-5x faster** with **80% less memory**.
  
  - This has sparked interest in the community for practical implementations in high-performance LLM projects.
- **Triton Kernel insights and optimizations**: Performance issues were discussed regarding Triton kernels, where a user noted that single kernel operations decreased speed compared to **PyTorch**, suggesting multiple kernels for efficiency.
  
  - Additional points were raised about the challenges related to BF16 operations not improving speed and ongoing issues with nightly builds in Triton.
- **H100 shows impressive speed improvements**: A user reported achieving **255 tokens/sec** with [H100](https://h100.url), using configurations such as `reduce-overhead`, further increasing to **300 tokens/sec** with manual tweaks.
  
  - These techniques provide new frameworks for optimizing GPU utilization in LLM applications.
- **ThunderKittens 0.000000002 is here with enhancements**: **ThunderKittens 0.000000002** has been released featuring significant upgrades including **6-14x faster linear attentions** and **faster attention backwards than FA3**.
  
  - A paper on kernel performance bottlenecks is also highlighted, questioning the real-world efficacy of custom kernels versus theoretical gains.
- **Cracked Engineers job platform gaining traction**: [Cracked Engineers](https://www.crackedengineers.com) launches, aimed at connecting talent with AI/tech startups, boasting current MRR of nearly **$1000** pre-launch.
  
  - The platform offers an AI-assisted job posting process and a newsletter for tech roles, inviting community feedback for continual improvement.

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Token Processing Speeds Favor GPU**: Members noted that **token processing speeds** are approximately **62 tok/sec on GPU** and **7.5 tok/sec on CPU**.
  
  - Fewill expressed enthusiasm by saying, *'nice!'* while discussing these speeds.
- **Hunting for Local LLM Recommendations**: A member sought recommendations for a **locally running LLM** similar to **Phind** or **ChatGPT** focusing on Python and Houdini SideFX.
  
  - Fabguy suggested researching **HumanEval** but noted that the niche nature of Houdini might affect response relevance.
- **NGINX Proxy Setup Woes**: One user encountered difficulties configuring an **NGINX proxy host** for the LM Studio server despite activating *serve on local network*.
  
  - Others shared troubleshooting steps, underscoring the critical nature of accurate configuration settings.
- **PCIe Bandwidth Debate Ignites**: Debate arose on whether **PCIe bandwidth** impacts inference performance, with suggestions that PCIe Gen 3 suffices since most processing occurs on the GPU.
  
  - However, users highlighted that bandwidth becomes critical for training models across multiple GPUs where high bandwidth is necessary.
- **Multi-GPU Configuration Queries**: Inquiries about using multiple **3090s** for large models revealed concerns over performance losses when exceeding a single GPU's memory.
  
  - It was determined that performance remains stable if the GPUs are identical, and offloading tasks improves overall processing efficiency.

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Users Report Slowness**: Members reported **slowness** with Aider, particularly using litellm's **get_model_cost_map** function, which could be improved by setting `export LITELLM_LOCAL_MODEL_COST_MAP='True'`.
  
  - One user noted that Aider generally tries to mask litellm's **slowness** in most cases.
- **Recommendations for Web Scraping**: A user suggested using [FireCrawl](https://firecrawl.dev) for **web scraping**, citing its effective extraction capabilities and self-hosting options.
  
  - Discussions indicated that FireCrawl could overcome challenges faced with social media scraping when configured correctly.
- **Managing Git Repositories with Aider**: Several users discussed strategies to maintain clean Git repositories, recommending manual commits over Aider's auto-commit feature.
  
  - One participant shared a process using `git switch` and merging squashed commits to keep repositories organized.
- **GitHub Copilot Rivals Aider**: A member highlighted that Copilot's integration with **OpenAI**, **Gemini**, and **Anthropic models** could impact its competition with Aider.
  
  - Another user expressed dissatisfaction with Copilot and mentioned a switch to Supermaven, indicating shifting preferences in coding assistants.
- **Effective Prompt Engineering Insights**: Discussions on crafting effective prompts emphasized their necessity for generating accurate AI outputs, focusing on providing ample context.
  
  - Concerns were raised about AI producing misleading results during debugging, prompting talk on restructuring prompts to improve clarity.

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Interest Grows in AI Research Grants**: Members inquired about experiences with **applying for AI research grants**, highlighting increasing interest for funding innovative projects.
  
  - This reflects a broader trend where financial support is becoming crucial for new AI initiatives.
- **Fascination with Evolving Algorithms**: Discussion centered around the **evolution of algorithms**, noting the differing personas emerging from AI models.
  
  - *They've been pushing boundaries*, with members eager to learn how these models manage various inputs.
- **Risks of Anthropomorphizing AI**: Conversations revealed concerns that **LLMs** producing human-like output can lead to misleading assumptions of intention.
  
  - Members urged the importance of viewing AI as tools, rather than inferring human emotions.
- **Calls for Enhanced Ethical AI Guidelines**: Members stressed the need for careful **ethical considerations** in AI to mitigate future risks.
  
  - Those developing intelligent systems bear the responsibility of setting clearer guidelines for their applications.
- **Concerning GPT Typo Issues**: Members reported persistent **typo issues** and incoherence when using ChatGPT, raising alarm over output quality.
  
  - The community expressed confusion, questioning if others encountered similar problems.

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Algorithmic Trading: Lessons Learned**: A member with 4 years in algorithmic trading shared insights on the complexities of market interactions, stating that **sloppy processes help resist negativity**.
  
  - *Understanding what doesn't work requires extensive simulated trades and research.*
- **Understanding Media Bias in AI Sentiment**: Members agreed that all media is biased, and identifying who benefits from that bias is crucial for accurate assessments.
  
  - *One noted they built a model that starts investigations under the assumption that all media is biased.*
- **Garbled AI Output Causing Confusion**: Members reported seeing odd garbled text in AI model outputs, raising concerns about its usability.
  
  - *Lowering temperature and top-p parameters was suggested as a potential fix, with experimentation recommended.*
- **Insights on Response Lengths**: Responses often stop when hitting the natural end based on structured prompts, with typical lengths being **3,000-4,000** characters.
  
  - *A member emphasized personalization significantly influences output length.*
- **Generating Medical Notes with LLMs**: A demo showcased generating synthetic medical notes using LLMs, allowing users to create detailed notes with minimal input.
  
  - *Check out the* [*demo here*](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9) *to see the tool's capabilities.*

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Inflection's services back online**: The recent **billing issues** have been resolved, and Inflection is now operational, boosting productivity for all users. More details can be found in the links to [Inflection 3 PI](https://openrouter.ai/inflection/inflection-3-pi) and [Inflection 3 Productivity](https://openrouter.ai/inflection/inflection-3-productivity).
  
  - With services restored, users report a return to normal operations, enhancing their tasks previously affected.
- **Recruiting Alpha Testers for macOS Chat App**: A developer is actively seeking **alpha testers** for their new **flexible chat app** for macOS, sharing [screenshots](https://imgur.com/a/HI5Py3A) that showcase its features.
  
  - Interested participants are encouraged to DM the developer to join this important testing phase.
- **OpenRouter API experiencing instability**: Users have reported **524 errors** affecting the OpenRouter API, causing significant requests delays and raising concerns about its readiness for public use.
  
  - As issues persist, some users are contemplating switching providers due to ongoing instability that hinders multiple request executions.
- **Debate over API key security risks**: Concerns arose about potential scraping of API keys, with talks highlighting risks from unauthorized proxies using models like **Claude 3.5 Sonnet**.
  
  - Users stressed the significance of safeguarding keys, with worries about how vulnerabilities could result in unintended leaks despite existing precautions.
- **Integration access in high demand**: Multiple members have voiced their requests for access to **integrations**, emphasizing polite pleas such as 'I would like to get access' to this feature.
  
  - One note-worthy request came from a **student researcher**, indicating academic interest in exploring the integration functionalities.

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Moondream secures $4.5M**: [Moondream](https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/) raises $4.5M to demonstrate the effectiveness of smaller AI models, with web crawlers active for several months.
  
  - Concerns arose about potential limitations and the implications of adopting smaller models in the AI industry.
- **Meta develops its own AI Search Engine**: Meta is reportedly working on an [AI-powered search engine](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report) to reduce dependency on Google and Microsoft.
  
  - The active web crawlers hint at significant shifts within Meta to enhance their search capabilities.
- **GitHub Copilot adds Gemini and Claude models**: GitHub introduces [Gemini models](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot) and Claude to enhance its Copilot capabilities with new features.
  
  - This represents an unexpected partnership between Microsoft and Google as they embrace a multi-model approach for developers.
- **Critique of existing Vector Databases**: Members critique current vector databases for lacking proper abstraction, endorsing the [pgai Vectorizer](https://github.com/timescale/pgai) for more efficient embedding management.
  
  - This tool promises to simplify syncing and maintenance of embeddings, crucial for boosting AI model performance.
- **OpenAI launches Chat History Search feature**: OpenAI rolls out a new feature for ChatGPT allowing users to search through their chat history, enhancing accessibility to past discussions.
  
  - Members celebrated the convenience of this long-awaited update, emphasizing improved continuity in conversations.

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular channel focus clarified**: In a query about the channel's focus, it was clarified that the <#1098713601386233997> channel is strictly for **Modular products**, with general software discussions directed to <#1104620458168553563>.
  
  - This distinction emphasizes the goal of maintaining focused discussions on Modular's offerings.
- **Mojo proposes memory-safe references revolution**: A member released a [major proposal](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b) on reimagining **memory-safe references** in Mojo, aiming for a safer yet simpler reference model.
  
  - Community feedback is sought to ensure the design supports both **optimization flexibility** and memory safety.
- **FlatBuffers and ProtoBuf comparison breakdown**: The team weighed the strengths of **FlatBuffers** and **ProtoBuf**, noting the zero parsing efficiency of FlatBuffers against ProtoBuf's focus on bit packing.
  
  - As they plan to use ProtoBuf for Serving, a [Swift ProtoBuf support example](https://github.com/apple/swift-protobuf) was shared as a development reference.
- **Swapping references in Mojo raises concerns**: Members deliberated the potential pitfalls of implementing **swapping references** in Mojo, drawing comparisons to Rust's mutable references management.
  
  - Concerns were raised about the added complexity this might bring, especially regarding **performance implications**.
- **Optimization focus on noalias discussions**: The discourse highlighted the significance of using `noalias` for efficient performance in Mojo, with many advocating it as a default approach.
  
  - A design supporting unique references was deemed essential, as lapsing here could lead to detrimental performance issues.

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Hugging Face CEO Generates Buzz**: The co-founder & CEO of Hugging Face, Clem, is slated to give an exciting talk, which is stirring anticipation within the community.
  
  - Details about the talk are still to be revealed, keeping members eager for more information.
- **Hellaswag Training Performance Surpasses Expectations**: A new record was set by achieving **GPT-2 (1.5B)** level performance on Hellaswag for under **$200** in **7.3 hours**, using **8xH100** hardware.
  
  - This represents a significant leap in efficiency from the previous benchmark of **24 8xH100-hours**.
- **Operational GPT-NeoX on Colab Confirmed**: **GPT-NeoX** is confirmed to work on Colab, with a reference link to a [Colab notebook](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb) provided.
  
  - The model in use is compact, showing potential for practical implementations with its **5M parameters**.
- **First Sparse Autoencoder Guide Launched**: A member released a [step by step guide](https://beta.ai-plans.com/guide/g7yjq98bhuyhkza) on utilizing a **Premade Sparse Autoencoder**, marking a fresh initiative in Mechanistic Interpretability.
  
  - The guide sets the stage for a series aimed at enriching understanding of interpretability techniques.
- **Custom Certificate Support Issues Acknowledged**: A member noted the absence of support for **custom certificates**, but shared a [workaround](https://github.com/EleutherAI/lm-evaluation-harness/issues/2436) that could help mitigate this limitation.
  
  - The discussion highlighted community efforts to share solutions that navigate these technical challenges.

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI CFO declares AI is mainstream**: In a [YouTube video](https://youtu.be/eCqFgVqWbEs), OpenAI CFO Sarah Friar emphasized that **AI isn’t experimental anymore**, as banks and fintechs are using it daily.
  
  - This momentous shift provides more opportunities for widespread implementation in various sectors.
- **SearchGPT Extension Launch**: OpenAI is expected to promote their new Chrome extension, allowing users to set **SearchGPT** as their default search engine alongside its launch.
  
  - Users can quickly initiate searches directly via the browser URL bar using commands that redirect to Google as required.
- **Introduction of ROCKET-1**: **ROCKET-1** is designed to enhance creative tasks in Minecraft by utilizing visual-temporal context prompting and is showcased by [Team CraftJarvis](https://craftjarvis.github.io/).
  
  - This development highlights the evolving capabilities of vision-language models in open-world applications.
- **Anthropic's Hiring Momentum**: Anthropic is gaining attention for its strong hiring practices, manifesting interest with the announcement of a new team member joining their ranks.
  
  - Their recent push reflects the company’s vibrant growth and ambition in the AI sector.
- **Claude's Integration with GitHub Copilot**: Claude 3.5 Sonnet is now available to developers using GitHub Copilot in Visual Studio Code, with rollout commencing this week.
  
  - This integration is expected to enhance coding experiences by providing advanced AI support directly within popular development tools.

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter needs visual models for full features**: For Open Interpreter to function properly with visual capabilities, a **multi-modal model** is generally required unless using **Moondream** for basic tasks.
  
  - Users reported difficulties replicating **Sonnet** or **GPT-4o** functionalities with local models such as **Llava**.
- **Challenges with local models executing actions**: Members encountered issues using local models like **Llava** to perform actions akin to cloud models, such as taking screenshots.
  
  - There’s a call for improved setup instructions for better integration with the **computer API**.
- **OpenAI Advanced Voice launched for Free Users**: OpenAI announced that **Advanced Voice** is now available to **Free users** in the EU, Switzerland, Iceland, Norway, and Liechtenstein.
  
  - This development significantly improves accessibility for users in these regions.
- **Apple offers $1M for AI server hacks**: **Apple** is prepared to pay up to **$1 million** for anyone who successfully hacks into their **AI servers**.
  
  - This initiative raises concerns about **cybersecurity** and invites scrutiny into Apple's security measures.
- **ChatGPT introduces chat history search**: OpenAI revealed the rollout of a search feature for chat history on **ChatGPT web**, enhancing usability for users.
  
  - This update allows users to quickly reference previous chats, improving continued interactions.

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Quantization Without LoRA Gets Attention**: Members debate whether base models can undergo quantization like **QLoRA** without leveraging LoRA, highlighting configuration challenges in non-LoRA environments.
  
  - *Hmmm I guess the main thing is we don't have a way to configure this in our non-LoRA model builders.*
- **FSDP's Simple CPU Offloading Tested**: Discussion centered around **FSDP**, which currently uses a single parameter for CPU offloading including parameters, gradients, and optimizer states, lacking detailed control.
  
  - *This approach has more data movements, but potentially faster since optimizer step is on GPU* was suggested as a performance consideration.
- **Skepticism Towards Quantized KV-Caches**: Members voiced doubts about the utility of **quantized KV-caches** using NF4 tensors due to high memory consumption in larger models.
  
  - *I don't think quantized kv cached in torchao is that useful/powerful yet,* indicating a need for further exploration.
- **Quantizing Non-Trainable Weights Gaining Interest**: Conversations highlighted that quantizing frozen weights during **PPO** could help in reducing memory use, particularly for non-trainable model components.
  
  - *Yeah I'd like to do something similar and quantize the non-trainable models during PPO,* showing interest in memory efficiency strategies.
- **Accuracy Risks Below 8-bit Quantization**: Concerns emerged over accuracy when quantizing activations, specifically KV caches, below 8-bit limits.
  
  - *Quantizing activations to below 8bit will have pretty severe accuracy issues,* emphasizing caution in aggressive quantization approaches.

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **PAPILLON tackles AI privacy concerns**: Researchers developed [PAPILLON](https://arxiv.org/abs/2410.17127), achieving **85.5% quality** with only **7.5% privacy leaks** in AI applications.
  
  - This system effectively allows the integration of **local and cloud LLMs**, addressing significant privacy challenges in modern AI.
- **PUPA benchmark shines light on privacy issues**: The team introduced **PUPA**, a benchmark assessing user-LLM interactions that contain personally identifiable information (**PII**).
  
  - Their findings inform a new method called **Privacy-Conscious Delegation**, merging API-driven and local model approaches.
- **DSPy simplifies AI programming**: An [ELI5 explanation of DSPy](https://x.com/lateinteraction/status/1851324349216927856) described it as a programming language allowing AI systems development through normal Python with DSPy signatures.
  
  - DSPy offers Modules for handling prompting strategies and Optimizers focused on enhancing output quality.
- **MIPROv2 Optimizer boosts quality**: Discussions revealed that MIPROv2 optimizer provides a **41%** increase in output quality and a **68%** decrease in leakage when utilized effectively.
  
  - Users noted its capability to sample training data and generate instructions based on various properties, optimizing overall performance.
- **MIPROv2 bug fix resolves usage issues**: A report surfaced about an error with MIPROv2 when paired with GPT-4o Mini, contrasting its successful runs with GPT-4.
  
  - Adjusting demo parameters helped resolve the confusion and improved performance with medium configurations.

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **NVIDIA spotlights wants in RAG**: NVIDIA's latest blog delves into **retrieval augmented generation (RAG)**, revealing that users desire extra functionalities, including **document translation** and **code writing**.
  
  - Even those focusing on internal data showed interest in **web search** capabilities, implemented through [Perplexity’s search API](https://docs.perplexity.ai/home).
- **Chroma's retrieval algorithm raises eyebrows**: Discussion emerged around **Chroma's** vector store retrieval behavior, particularly when using `index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store)`.
  
  - Members highlighted that **Chroma's** algorithm is approximate, affecting the variability of results even with similar indexed chunks.
- **Web scraping mastery unveiled**: A practical YouTube video titled '[This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I)' was shared, showcasing advanced web scraping capabilities for 2024.
  
  - The video advocates for using **AgentQL** to scrape websites for free, demonstrating real-world applications of LLMs.
- **Blockchain engineer seeks project collaborations**: A blockchain engineer with roots from 2017 reached out for project opportunities, boasting expertise in **defi**, **NFT games**, and languages like **Solidity** and **RUST**.
  
  - Their background includes work on various projects involving **Dex**, **DAO**, and **NFT** minting and staking.
- **Building advanced RAG systems with MLflow**: A guide outlined how to create **advanced RAG systems** utilizing MLflow and LlamaIndex, allowing for a combination of vector and keyword-based searches.
  
  - This approach targets **event-driven orchestration** to enhance workflow management, as illustrated in an example available on [GitHub](https://github.com).

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents Hackathon Registration Surges**: Over **1K+ innovators** have signed up for the [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/) within just a few days, reflecting strong interest. Complete the [participant sign up](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform) today if you haven’t joined yet!
  
  - *It’s not too late to join us!*
- **8th Lecture Scheduled at 3:00pm PST**: The **8th lecture** will take place today at **3:00pm PST**, with a [livestream available here](https://www.youtube.com/live/wm9-7VBpdEo). This session focuses on integrating complex reasoning with Large Language Models, promising valuable insights.
  
  - *Tune in!*
- **Formation of a Study Group**: A member proposed starting a **study group** for course discussions, suggesting virtual meetings to engage those who joined late. Expressions of interest followed quickly with several members confirming they wanted to participate.
  
  - *Sounds cool!*
- **Request for Subtitles on Live Stream**: A member requested to enable **Subtitles** on the live streaming videos, with confirmation that all lectures are edited afterwards and made available with subtitles. This ensures accessibility, enhancing the viewer experience.
  
  - *We’re working on it!*
- **Developing React-based Automation Agent**: A member inquired about creating a **React-based agent** to automate tasks using [pyauto gui](https://pyautogui.readthedocs.io/en/latest/) for actions based on current state evaluations. Suggestions for direct inquiries rather than generalized questions were noted.
  
  - *It's easier to ask directly!*

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Pink Pixel Patches in Latent Diffusion Model Training**: While training a **class conditioned latent diffusion model**, a member reported encountering **pink pixel patches** during decoding from the VAE, which decrease in frequency with more training.
  
  - They are considering if more aggressive clipping in **DDIM p_sample**, currently at **99.95%**, will solve the issue of these patches.
- **Misunderstanding Parameters vs Tokens**: A member mistakenly thought the **100B** reference was for parameters over tokens, which led to a mix-up clarified by another member's acknowledgment.
  
  - Further, they noted the linked model actually has only **8B parameters**, gaining validation from peers.
- **Collaborative Exploration of IJEPA Architecture**: A member expressed interest in collaborating on an innovative architecture that merges **IJEPA** with **autoregressive image generation without vector quantization**.
  
  - Their enthusiasm for joint efforts to explore this unique architecture signals potential advancements in this space.

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz Has a Negative Line Day**: George Hotz expressed having a **negative line day**, prompting humorous reactions from the community.
  
  - This light-hearted exchange reflects the supportive atmosphere among members as they tackle coding challenges.
- **CI Tests Get Faster**: Chenyuy reported a **2-minute faster CI test**, indicating progress in performance optimization.
  
  - This improvement in the testing process showcases shared efforts to boost efficiency in the tinygrad project.
- **Uops Readability Challenges Surface**: Concerns emerged regarding the readability of **Uops** with some one-liners being difficult to comprehend.
  
  - A suggestion for creating a documentation page was mentioned to potentially enhance code clarity for all users.
- **Documentation Maintenance Issues Highlighted**: Chenyuy highlighted the **maintenance concerns** regarding documentation that often becomes outdated quickly.
  
  - He pointed out that having inaccurate documentation may hinder progress more than having none, reflecting the rapid pace of change in tinygrad.
- **Debate on Premature Optimization**: George Hotz proposed the removal of certain code elements to avoid **premature optimization** pitfalls.
  
  - This discussion underscores the thoughtful testing underway to balance code efficiency carefully against potential complexities.

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **RAGAS Enhances LLM Evaluations**: A member suggested using [RAGAS](https://github.com/explodinggradients/ragas) to improve **LLM application evaluations**, showcasing its capabilities and methodologies.
  
  - This tool aims to provide developers with refined methods for evaluating language models effectively.
- **CSV Files Seeking Integration**: A discussion arose about integrating **CSV files** as data sources with open source models like **LLAMA3**, noting a gap in existing examples.
  
  - The inquiry specifically mentioned using CSVChain and PandasAgent with non-OpenAI models for better data handling.
- **LangChain-Python Version Queries**: Clarification was sought on which version of **Python** compatible with **LangChain version 0.3**, reflecting the community's need for setup guidance.
  
  - Proper environment configuration remains crucial for developers to use LangChain efficiently.
- **LangChain-JS Course Launches**: **Exciting news!** A new [LangChain-JS course](https://www.udemy.com/course/genai-langchain-for-javascript-developers/?couponCode=AMIT100) has been released on Udemy, aimed at beginners.
  
  - It spans from the basics to building a complete RAG application, with the first **100 students** able to enroll for free.
- **Web Scraping Masterclass**: A member highlighted a [YouTube video](https://youtu.be/7kbQnLN2y_I) titled 'This is how I scrape 99% websites via LLM', which teaches practical web scraping with LLM.
  
  - It emphasizes the use of [AgentQL](https://www.agentql.com/) to scrape websites for free, showcasing innovative techniques.

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Clarifying 'Multiple' on the Leaderboard**: 'Multiple' on the leaderboard indicates the **ability to choose the correct function** from several options in a single turn, as outlined in [this GitHub example](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438). The evaluation of multi-step remains ambiguous in this context.
  
  - This confusion is notable, especially regarding how multi-step executions differ from multi-turn scenarios, which has led to various discussions among users.
- **Multi-Step vs Multi-Turn Evaluation Methods**: A member clarified that 'multiple' relates to functions, while **multi-step evaluations** fall under the 'multi_turn' category, with no singular multi-step evaluation currently utilized. Understanding these distinctions is crucial for accurate interpretation.
  
  - The overlap between multi-step and multi-turn evaluations could potentially confuse users, as both concepts share the same categories in evaluations as set by the leaderboard.

 

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Cracked Engineers Job Platform Launches!**: A member shared an exciting new [job platform for technical roles](https://www.crackedengineers.com/) called **Cracked Engineers**, aiming to be the go-to for top AI/tech startups.
  
  - With a projected **$1000 MRR** before the official launch, the platform is already attracting top companies like **Weaviate**, **UnslothAI**, and **JuliusAI**.
- **Insightful Weekly Tech Jobs Newsletter Introduced**: The platform is set to release a **weekly tech jobs newsletter** that will curate positions based on user preferences, starting soon.
  
  - Users can subscribe to tags that interest them, such as **CUDA**, **MLOps**, or **Software Engineering** through their dashboard.
- **Exciting Job Opportunities at AI Startups**: **Unsloth AI**, **Julius AI**, and **Jimini AI** are actively hiring for excellent positions that they would consider if they weren't a founder.
  
  - These positions are described as **amazing opportunities** for anyone looking to work with cutting-edge AI technology.

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Member Seeks SymNoise Code Implementation**: A member is looking for a code implementation for the **SymNoise** fine-tuning technique, which integrates **symmetric noise** into embedding. They expressed difficulties in achieving this due to issues with the **batch size** requirements.
  
  - This inquiry shows a growing interest in advanced fine-tuning methods within the community, though specific solutions were not provided.
- **SymNoise Boosts LLaMA-2-7B Performance**: The **SymNoise** method improved the **LLaMA-2-7B** performance on AlpacaEval from **29.79%** to an impressive **69.04%**, surpassing **NEFTune**. This signifies a significant **6.7%** enhancement over NEFTune's score of **64.69%**, as noted in the paper's abstract.
  
  - The results highlight the potential of **SymNoise** in fine-tuning language models, setting a new benchmark for performance.
- **SymNoise Outshines NEFTune Across Models**: Tests reveal that **SymNoise** consistently yields better results than **NEFTune** across various models and baseline datasets. This has sparked discussions about the need for further research in this area.
  
  - Community members emphasized the importance of continuing to explore and validate these fine-tuning methodologies.
- **Call for Research Resources on SymNoise**: In the inquiry, a member linked to the **arXiv** paper detailing the **SymNoise** method, underscoring its relevance in the field. However, there were no additional code resources or implementations shared to aid in the implementation challenge.
  
  - This points to a broader need for collaborative efforts in developing practical applications based on recent research findings.

 

---

The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1300604946403823708) (1 messages):

> - `Clem intro`
> - `Live Workshop Promotion`

- **Clem introduces himself to the community**: Clem, co-founder and CEO at Hugging Face, expressed excitement about using Discord to interact with the community members more closely.
  
  - *I can't wait to interact with all of you* and emphasized his eagerness to engage with users.
- **Promotion for Wednesday's live workshop**: Clem is seeking ideas on how to promote his upcoming live workshop scheduled for Wednesday, shared through this [link](https://streamyard.com/watch/JS2jHsUP3NDM).
  
  - He encouraged users to share any Discord channels or groups where they could post about the event to boost participation.

 

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1300536845301452911) (870 messages🔥🔥🔥):

> - `TensorFlow frustrations`
> - `LLM training approaches`
> - `Learning paths in AI/ML`
> - `Using Hugging Face API`
> - `Stopping model overfitting`

- **Frustrations with TensorFlow**: Many members expressed frustrations with TensorFlow, citing issues like disabling GPU support on Windows and having to navigate complex documentation. They found transitioning to PyTorch often led to faster results and less hassle.
  
  - Users shared their experiences of venting about TensorFlow bugs and poor support, indicating a common sentiment of dissatisfaction with the framework.
- **Training LLMs and Overfitting Solutions**: Noaroggendorff discussed ongoing efforts in training a Llama model, having undergone numerous attempts to optimize settings. The best suggestion for minimizing overfitting was to utilize only one epoch during training.
  
  - This sparked a conversation among users about varying strategies to tackle challenges in training large language models effectively.
- **Learning Paths in AI and ML**: Several users offered insights on developing skills in ML and AI, emphasizing that a foundational understanding is crucial. They discussed the broad landscape of AI, and how new learners should focus on specific tasks to build expertise.
  
  - The conversation highlighted the importance of adapting learning according to individual interests and the nature of the rapidly evolving AI field.
- **Using Hugging Face API for Token Probabilities**: Asahikokura inquired about obtaining token probabilities through the Hugging Face API, and was informed that the Inference Client could facilitate this process. Users provided examples of how to utilize the API for log probabilities and directed him to the relevant documentation for rate limits.
  
  - It was discussed that the API allows access without downloading models locally, making it easier for beginners to experiment with language models.
- **MLOps Mastery and Job Acquisition**: Rumigazzi asked how someone could master MLOps and secure a job in the field. The conversation touched upon the growing demand for skills in AI and MLOps, with suggestions to engage in community projects and build a solid learning path.

**Links mentioned**:

- [Code of Conduct – Hugging Face](https://huggingface.co/code-of-conduct): no description found
- [How AI Agents Can Be Exploited Through Indirect Prompt Injection · AI Security Blogs](https://www.stealthnet.ai/post/how-ai-agents-can-be-exploited-through-indirect-prompt-injection): AI security is the next wave. Learn how to hack and defend AI & ML models.
- [Format selector for 2410.02694](https://arxiv.org/format/2410.02694): no description found
- [LaTeX.js](https://latex.js.org/): no description found
- [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786): While scaling Transformer-based large language models (LLMs) has demonstrated promising performance across various tasks, it also introduces redundant architectures, posing efficiency challenges for r...
- [Context.ai](https://context.ai/compare/gemini-pro/gemini-ultra): Compare pricing, benchmarks and model attributes between Gemini Pro and Gemini Ultra.
- [rombodawg/Rombos-LLM-V2.5-Qwen-72b · Hugging Face](https://huggingface.co/rombodawg/Rombos-LLM-V2.5-Qwen-72b): no description found
- [Can You Run It? LLM version - a Hugging Face Space by Vokturz](https://huggingface.co/spaces/Vokturz/can-it-run-llm): no description found
- [Rate Limits](https://huggingface.co/docs/api-inference/en/rate-limits): no description found
- [What is NotebookLM - Help](https://support.google.com/notebooklm/answer/14273541?hl=en): no description found
- [Gokacik O Yerim GIF - Gokacik Gok O yerim - Discover & Share GIFs](https://tenor.com/view/gokacik-gok-o-yerim-yerim-yer%C4%B1m-gif-8234693358169729819): Click to view the GIF
- [Peter Griffin Family Guy GIF - Peter Griffin Family Guy Peter - Discover & Share GIFs](https://tenor.com/view/peter-griffin-family-guy-peter-gif-26549552): Click to view the GIF
- [meta-llama/Llama-3.1-8B · Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B): no description found
- [Local AI with Docker's Testcontainers](https://huggingface.co/blog/Tonic/localai-testcontainers): no description found
- [XSS on every Gradio server via upload of HTML files, JS files, or SVG files](https://github.com/gradio-app/gradio/security/advisories/GHSA-gvv6-33j7-884g): ### Impact \*\*What kind of vulnerability is it? Who is impacted?\*\*
  
  This vulnerability involves **Cross-Site Scripting (XSS)** on any Gradio server that allows file uploads. Authenticated users...
- [Alone Glitch GIF - Alone Glitch Film - Discover & Share GIFs](https://tenor.com/view/alone-glitch-film-eloresnorwood-heartbreak-gif-15491348): Click to view the GIF
- [Alpaca Llama GIF - Alpaca Llama Animation - Discover & Share GIFs](https://tenor.com/view/alpaca-llama-animation-art-lama-gif-24994921): Click to view the GIF
- [AI Builder Club](https://link.agent.rocks/6dUcFwA): Learn to code & build apps with Cursor AI & latest apps course
- [Pdf Component Example](https://www.gradio.app/guides/pdf-component-example): A Step-by-Step Gradio Tutorial
- [Embed No GIF - Embed No No Embed - Discover & Share GIFs](https://tenor.com/view/embed-no-no-embed-megamind-megamind-meme-gif-25261934): Click to view the GIF
- [This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I): How to do web scraping with LLM in 2024Use AgentQL to scrape website for free: [https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason](https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason)_...
- [Curious GIF - Curious - Discover & Share GIFs](https://tenor.com/view/curious-gif-9412615): Click to view the GIF
- [Insomnia Sleep GIF - Insomnia Sleep Tired - Discover & Share GIFs](https://tenor.com/view/insomnia-sleep-tired-bed-smart-gif-5513542): Click to view the GIF
- [zero-gpu-explorers/README · Zero-GPU Quota etc](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/7): no description found
- [marcsun13 (Marc Sun)](https://huggingface.co/marcsun13): no description found
- [Tweet from The Linux Foundation (@linuxfoundation)](https://x.com/linuxfoundation/status/1851052486288613598?s=46&t=RHgECJov_mYM1AEf96kY2g): Good morning! We are LIVE at Open Source Summit Japan and AI_dev Open Source GenAI & ML Summit 2024 for this morning's keynotes! Follow #OSSummit #AIDev as we live tweet. Watch the livestream here...
- [GitHub - brucemiller/LaTeXML: LaTeXML: a TeX and LaTeX to XML/HTML/ePub/MathML translator.](https://github.com/brucemiller/LaTeXML): LaTeXML: a TeX and LaTeX to XML/HTML/ePub/MathML translator. - brucemiller/LaTeXML
- [GitHub - RayFernando1337/LLM-Calc: Instantly calculate the maximum size of quantized language models that can fit in your available RAM, helping you optimize your models for inference.](https://github.com/RayFernando1337/LLM-Calc): Instantly calculate the maximum size of quantized language models that can fit in your available RAM, helping you optimize your models for inference. - RayFernando1337/LLM-Calc
- [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](https://arxiv.org/html/2410.02694v2): no description found
- [19,500+ Beautiful Sad Girl Hurt Silhouette Stock Photos, Pictures & Royalty-Free Images - iStock](https://www.istockphoto.com/photos/beautiful-sad-girl-hurt-silhouette): no description found
- [LaTeXML A LaTeX to XML/HTML/MathML Converter](https://math.nist.gov/~BMiller/LaTeXML/): no description found
- [Inference](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client): no description found
  
   
  

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1300731319680241664) (4 messages):

> - `Highlighting cool projects`
> - `New members in the server`
> - `Community engagement`

- **Highlight your cool projects for recognition**: A member suggested highlighting their project in <#897390720388825149> to potentially receive a shoutout in announcements.
  
  - They emphasized that getting recognized can be quite rewarding, hinting at the community's supportive nature.
- **Welcoming new members**: New member **borys_nadykto** expressed gratitude for the advice and acknowledged their recent arrival to the server.
  
  - *Many members are new*, indicating the community's ongoing growth and encouraging participation.
- **Sharing delightful experiences**: Member **tonic_1** responded positively, stating it’s *delicious to share* experiences within the channel.
  
  - This reflects the community's open attitude towards sharing and engaging with each other.

 

**Link mentioned**: [Pokemon Pikachu GIF - Pokemon Pikachu Clap - Discover & Share GIFs](https://tenor.com/view/pokemon-pikachu-clap-clapping-clapping-gif-gif-13465728489229726846): Click to view the GIF

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1300555304546598913) (53 messages🔥):

> - `ML and quantum computing`
> - `Understanding advanced research papers`
> - `Using AI to enhance paper reading`
> - `Homomorphic encryption in privacy`
> - `The impact of attention mechanisms in ML`

- **ML and Quantum Compute Investigates Black Holes**: A paper exploring the use of **machine learning** and **quantum computing** to study black holes was shared, identified as a challenging read.
  
  - Members expressed excitement and mentioned feeling overwhelmed, indicating that it may require a solid background in **quantum mechanics**.
- **Navigating Complex Papers**: Readers discussed strategies for tackling complex academic papers, including reading **abstracts** and referencing **appendices** for better understanding.
  
  - Using AI to create quizzes or flashcards emerged as a beneficial method to reinforce knowledge after reading.
- **Homomorphic Encryption for User Privacy**: A resource was shared that discusses **homomorphic encryption** as a key technology for enhancing user privacy during data processing on devices.
  
  - The document emphasizes performing computations locally to minimize external data exposure while utilizing on-device **machine learning** features.
- **The Shift to Attention Mechanisms**: A user expressed a newfound disinterest in **Convolutional Neural Networks (CNNs)** that lack **attention mechanisms** after learning about their importance.
  
  - This reflects a broader trend in machine learning discussions, highlighting the growing favor for attention-based models over traditional architectures.
- **Creating Interactive Paper Reading Tools**: A member proposed the idea of a **Hugging Face space** where users could interactively read papers with an integrated **language model** for explanations.
  
  - The discussion included references to existing resources and code that could potentially aid in implementing such a tool.

**Links mentioned**:

- [Combining Machine Learning and Homomorphic Encryption in the Apple Ecosystem](https://machinelearning.apple.com/research/homomorphic-encryption): At Apple, we believe privacy is a fundamental human right. Our work to protect user privacy is informed by a set of privacy principles, and…
- [llama-recipes/recipes/quickstart/NotebookLlama at main · meta-llama/llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama): Scripts for fine-tuning Meta Llama with composable FSDP &amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp; custom datasets for applications such as summarization and Q&...
- [GitHub · Build and ship software on a single, collaborative platform](https://github.co): Join the world's most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1300608096934301728) (209 messages🔥🔥):

> - `Hemp Nanosheets`
> - `Autonomous Science Workflows`
> - `Custom WordPiece Tokenizer`
> - `Model Usage Analytics`
> - `Research Collaboration`

- **Hemp Nanosheets as a Future Material**: The conversation centered around the potential of **hemp-derived carbon nanosheets**, which are touted to be cost-effective and comparable to graphene in various applications, including energy storage and materials engineering.
  
  - Dr. David Mitlin's research has established the feasibility of producing these nanosheets at around **$500 per ton**, leading to discussions on their implications in military and aerospace applications.
- **Ideas for Autonomous Science**: A user shared workflows for **autonomously designed scientific experiments**, emphasizing the importance of training AI to follow the scientific method effectively using hemp nanosheets as a case study.
  
  - The discussion included the possibility of leveraging these workflows to conduct innovative research without human intervention, reflecting excitement over future scientific capabilities.
- **Custom WordPiece Tokenizer Development**: A beginner shared their journey in developing a **custom-built WordPiece tokenizer**, reworking Hugging Face's ideas for improved results with their specific data.
  
  - They aim to contribute this tokenizer to existing frameworks, inviting feedback and suggestions from the community to enhance their project further.
- **Model Analytics for Open Source**: A user introduced a new library for tracking model usage, designed to integrate seamlessly with the **vanilla Transformers library** and provide analytics without requiring extra steps from the user.
  
  - The tool aims to assist developers in understanding user interactions and improving open-source models based on detailed analytics.
- **Discussion on Research Collaboration**: Contributors engaged in discussions about the significance of sharing their findings and encouraging collaborations in niche fields, particularly regarding material sciences and NLP.
  
  - Suggestions included reaching out to newsletters for exposure and emphasizing the importance of open dialogue within the research community.

**Links mentioned**:

- [Cat Trascendence GIF - Cat Trascendence Meme - Discover & Share GIFs](https://tenor.com/view/cat-trascendence-meme-gif-8496882): Click to view the GIF
- [Hemp fibres ‘better than graphene’ | Pennsylvania Hemp Industry Council](https://www.pahic.org/hemp-fibres-better-than-graphene/): no description found
- [Hemp Makes Better Supercapacitor Electrodes](https://hempingtonpost.com/hemp-makes-better-supercapacitor-electrodes/): Hemp based electrodes for supercapacitors outperform standard supercapacitors
- [Building a Custom WordPiece Tokenizer from Scratch: Concepts, Formulas, and Token Creation](https://medium.com/@krasniuk-ai/building-a-custom-wordpiece-tokenizer-from-scratch-concepts-formulas-and-token-creation-0e955465d239): Reacreating Huggingface Wordpiece Formula.
- [Mujikcboro Seriymujik GIF - Mujikcboro Seriymujik - Discover & Share GIFs](https://tenor.com/view/mujikcboro-seriymujik-gif-24361533): Click to view the GIF
- [The Road To El Dorado Both GIF - The Road To El Dorado Both Both Is Good - Discover & Share GIFs](https://tenor.com/view/the-road-to-el-dorado-both-both-is-good-gif-8304204): Click to view the GIF
- [Shrek Reaction GIF - Shrek Reaction Really - Discover & Share GIFs](https://tenor.com/view/shrek-reaction-really-gif-27425089): Click to view the GIF
- [GitHub - Bynesoft-Ltd/byne-serve: Google Analytics for open-source models: track usage and learn how people use your models.](https://github.com/Bynesoft-Ltd/byne-serve): Google Analytics for open-source models: track usage and learn how people use your models. - Bynesoft-Ltd/byne-serve
- [GitHub - koushik2k3/Meme-Explainer-Bot: Generates a given meme's explanation along with context](https://github.com/koushik2k3/Meme-Explainer-Bot): Generates a given meme's explanation along with context - koushik2k3/Meme-Explainer-Bot

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1300618090245001402) (1 messages):

> - `Discord Event Details`

- **Discord Event Announced**: A link to a Discord event was shared for members to participate: [Event Details](https://discord.com/events/879548962464493619/1300617948414611507).
  
  - Participants are encouraged to check out the event for further information and updates.
- **Involvement Encouraged**: Members are urged to engage in the upcoming event by following the provided link.
  
  - Discussion around the event's agenda and topics of interest might spark in the channel.

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1300648347203403858) (8 messages🔥):

> - `Swin Transformer v2`
> - `DINO model for vision`
> - `Attention masks in vision transformers`
> - `Fine-tuning molmo VLM`

- **Exploration of Swin Transformer v2**: Members discussed using **Swin Transformer v2** for image-like data cubes, sharing insights on its applicability and nuances.
  
  - *One member noted* that they are using data cubes instead of proper images, which opened a discussion about modifying architectures for unique input shapes.
- **DINO Model's Gameplay with Attention Masks**: The **DINO** model was suggested as an option, but it doesn't support attention masks directly; it relies on self-supervised learning.
  
  - A member pointed out that adapting the **Swin transformer** to accommodate attention masks could be complex due to its hierarchical structure.
- **Attention Masks Adaptation Discussion**: Members debated how to effectively modify attention heads in the **Swin v2** model to utilize attention masks, advocating for a direct source code adjustment.
  
  - *Participants noted* that reshaping tensors might complicate the implementation unnecessarily, preferring direct modifications.
- **Interest in Fine-tuning molmo VLM**: A query was raised about experiences in fine-tuning the **molmo VLM**, indicating an interest in collaborative insights on this model.
  
  - No responses were recorded about this topic at this time.

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1300592439400075297) (8 messages🔥):

> - `LangChain SQL Agent`
> - `Hugging Face NLP Course Resources`
> - `LLM Fine-Tuning`
> - `Research Papers on Modern Models`

- **LangChain SQL Agent Reference**: A member suggested a [GitHub notebook](https://github.com/langchain-ai/langchain/blob/master/cookbook/LLaMA2_sql_chat.ipynb) on **LLaMA2 SQL chat** as a potential resource for those working with **LangChain SQL Agent**.
  
  - This resource appears to aid in building context-aware reasoning applications, which could be helpful for users trying to implement similar functionalities.
- **Expanding Resources from Hugging Face Course**: After completing the **Hugging Face NLP course**, a member expressed interest in further theoretical resources for LLM fine-tuning.
  
  - Another member recommended [Jurafsky and Martin’s Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) for deeper theoretical insights.

 

**Link mentioned**: [langchain/cookbook/LLaMA2_sql_chat.ipynb at master · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/master/cookbook/LLaMA2_sql_chat.ipynb): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1300535564864327770) (170 messages🔥🔥):

> - `Unsloth Training Tools`
> - `FP8 Fine-Tuning`
> - `Gradio UI for Model Training`
> - `Cloud GPU Connections`
> - `Job Opportunities in AI`

- **Gradio UI Tool for Unsloth**: A user created an app using Gradio that simplifies the training of models with Unsloth, featuring settings selection and model training in a user-friendly interface.
  
  - The tool aims to provide nocode users with an easier way to train AI models by allowing easy adjustment of settings and a smooth process for uploading models to Hugging Face.
- **Opportunities in AI Hiring**: There is an ongoing hiring campaign by Unsloth, spotlighted through a platform called Cracked Engineers, aimed at sourcing tech talent in AI fields.
  
  - Community members were encouraged to explore job listings and utilize the platform for job tracking and hiring updates.
- **FP8 Training Discussion**: Discussion emerged regarding the adoption of FP8 for training and fine-tuning models in Unsloth, with potential speed improvements noted.
  
  - Queries were raised about the implementation specifics, including whether FP8 is used for base weights, LoRA, and other components.
- **Critiques on Model Training**: A user mentioned the challenges of fine-tuning smaller models and the importance of the quality of the dataset, arguing that larger models like Llama 3.2:70B might yield better results.
  
  - Critics in the discussion highlighted the need for understanding model training nuances rather than relying solely on user-friendly interface tools.
- **Gradient Accumulation Issues Addressed**: A blog post discussed critical issues with gradient accumulation in popular training frameworks and how they affect model performance.
  
  - The Unsloth team is actively addressing these issues, reinforcing the importance of uniform outputs when applying gradient accumulation during training.

**Links mentioned**:

- [Cracked Engineers](https://www.crackedengineers.com/): Find the best engineers for your startup.
- [Introduction to Quantization cooked in 🤗 with 💗🧑‍🍳](https://huggingface.co/blog/merve/quantization): no description found
- [Cracked Engineers](https://crackedengineers.com/job/unsloth-ecb33b9f-7a36-43d3-ba5a-7af3ce4add8e): Find the best engineers for your startup.
- [Tweet from Aleksa Gordić 🍿🤖 (@gordic_aleksa)](https://x.com/gordic_aleksa/status/1851247076987855063): [🚀] Super excited to share this: I've built a job platform for technical roles called "Cracked Engineers". :) If you want to land a job with some of the world's best AI/tech startups ...
- [Zach Mueller - PyTorch, Gradient Accumulation, and the dreaded lack of reproducability](https://muellerzr.github.io/blog/gradient_accumulation_part2.html): no description found
- [How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama): Beginner's Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama
- [GitHub - NVIDIA/TransformerEngine: A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper and Ada GPUs, to provide better performance with lower memory utilization in both training and inference.](https://github.com/NVIDIA/TransformerEngine): A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper and Ada GPUs, to provide better performance with lower memory utilizatio...
- [GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](https://github.com/unslothai/unsloth): Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
- [Introducing Unsloth](https://unsloth.ai/introducing): no description found

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1300584861232529532) (57 messages🔥🔥):

> - `Frustrations with School`
> - `Life After Education`
> - `Value of Experience`
> - `Plans Beyond Masters`
> - `Perceptions of PhDs`

- **Frustrations with the School System**: A member expressed feeling like they are *wasting time* in school, seeking to make a difference instead.
  
  - This sentiment resonated with others, as shared by another member who reflected on how personal experiences shape perspectives on education.
- **Life After Education is Complex**: One member who quit school at 17 noted that upon turning 40, understanding life’s complexities becomes clear, stating *everything I thought was going to happen didn't*.
  
  - They emphasized the vital nature of connections with friends, revealing a sense of loss as time passes.
- **R&D Insights: Learning on the Job**: A recent master's graduate acknowledged their limited knowledge of the tech world, stating *man I realise I have truly no idea about tech world yet*.
  
  - However, they were optimistic about being paid to engage in R&D as a part of their career journey.
- **Perceptions of PhDs in Industry**: A member criticized PhDs by claiming *everyone I've ever met with a PhD is an idiot*, while others defended the value of expertise.
  
  - This inspired a debate on the contrast between narrow expertise and broader skill sets.
- **Embracing Ongoing Learning**: Several members discussed the importance of continuing education even after formal studies, highlighting the sentiment that *you know what you don't know*.
  
  - This acknowledgment paired with the humor about being *a noob* in the field shows a willingness to learn despite feeling inexperienced.

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1300546103678337158) (65 messages🔥🔥):

> - `Converting HF to GGUF`
> - `Continued Pretraining`
> - `Fine-tuning Llama 3.1`
> - `Unsloth Installation Issues`
> - `Memory for Neural Networks`

- **Error in HF to GGUF Conversion Process**: A user encountered an error when using the `convert_hf_to_gguf.py` script, specifically stating that 'q4_k_m' is an invalid choice for the `--outtype` argument.
  
  - Community members suggested using the `llama.cpp` repository and checking the provided options to resolve the issue.
- **Insights on Continued Pretraining with Unsloth**: Unsloth's new release reportedly allows for **2x faster** continual pretraining with **50% less VRAM** than previous methods, with resources provided for Mistral v0.3 training.
  
  - Key insights emphasize finetuning input and output embeddings and offloading them to disk to save memory.
- **Help with Fine-tuning Llama 3.1 Model**: A user requested assistance on training the Llama 3.1 model using Google Colab, indicating difficulty in understanding the setup.
  
  - Community responses directed them to existing notebooks and prompted them to address any errors encountered during model downloading.
- **Challenges Installing Unsloth**: Installation issues with Unsloth were raised, specifically related to errors encountered while using pip commands for installation.
  
  - Suggestions included using Miniconda for better environment isolation and correcting typographical errors in package names.
- **Neural Network Memory Query**: A member inquired about implementing memory in neural networks, specifically if models could remember past interactions without continual dataset additions.
  
  - Responses included strategies for logging questions and responses to provide context during model training.

**Links mentioned**:

- [Miniconda — Anaconda documentation](https://docs.anaconda.com/miniconda/): no description found
- [Continued LLM Pretraining with Unsloth](https://unsloth.ai/blog/contpretraining): Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/) (1 messages):

mrdragonfox: and im sure you have the capital required to make that happen yes ? lol

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1300921043950633011) (1 messages):

> - `PyTorch Quantization`
> - `Optimizer CPU Offload`

- **Exploring Low Bit Optimizations in PyTorch**: The [low_bit_optim](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) prototype in PyTorch showcases native **quantization** and **sparsity** for improved training and inference.
  
  - This initiative is part of a broader effort to optimize performance while maintaining model accuracy, which resonates with ongoing trends in AI.
- **Optimizer CPU Offload Insights**: Discussion highlighted the potential of **Optimizer CPU Offload** in enhancing the efficiency of **low-bit** training frameworks.
  
  - By offloading operations to the CPU, models can leverage available hardware better, which could lead to **faster training times** and lower resource usage.

 

**Link mentioned**: [ao/torchao/prototype/low_bit_optim at main · pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload): PyTorch native quantization and sparsity for training and inference - pytorch/ao

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**announcements**](https://discord.com/channels/1002292111942635562/1002292398703001601/1300871806022058147) (1 messages):

> - `Stable Diffusion 3.5 Medium`
> - `Model performance and hardware compatibility`
> - `Community feedback implementation`
> - `Open release and licensing`
> - `Improvements from Stable Diffusion 3 Medium`

- **Stable Diffusion 3.5 Medium Launches**: The **Stable Diffusion 3.5 Medium** model is now available for free commercial and non-commercial use, boasting **2.5 billion parameters** and designed to run on consumer hardware, even on low-end devices.
  
  - This release aims to democratize access to AI technology by ensuring it functions on systems with as little as **9.9 GB of VRAM**.
- **Enhanced Performance Over Medium Models**: Stable Diffusion 3.5 Medium delivers best-in-class image generation with **advanced multi-resolution capabilities**, outperforming other medium-sized models in **prompt adherence and image quality**.
  
  - With this model, users can expect highly efficient and high-quality performance tailored for most consumer GPUs.
- **Community Feedback Shapes Development**: After the **June release of Stable Diffusion 3 Medium**, community feedback prompted significant improvements rather than a quick fix, leading to this new iteration.
  
  - The team's commitment to listening reflects a dedication to **transforming visual media** through better tools for builders and creators.
- **Open Release with Flexible Licensing**: Today’s release includes multiple variants of Stable Diffusion 3.5 that are customizable and run on consumer hardware, available under the permissive **Stability AI Community License**.
  
  - Users can download models directly from **Hugging Face** and access the code on **GitHub**.

 

**Link mentioned**: [Introducing Stable Diffusion 3.5 — Stability AI](https://stability.ai/news/introducing-stable-diffusion-3-5): Today we are introducing Stable Diffusion 3.5. This open release includes multiple model variants, including Stable Diffusion 3.5 Large and Stable Diffusion 3.5 Large Turbo, and as of October 29th, St...

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1300543867241758861) (253 messages🔥🔥):

> - `Stable Diffusion 3.5 Medium`
> - `GPU Performance Comparison`
> - `Sana Autoencoder`
> - `ComfyUI vs A1111`

- **Stable Diffusion 3.5 Medium's Image Generation**: Users noted that **Stable Diffusion 3.5 Medium** performs better for generating images over 1MP compared to **3.5 Large**, handling up to 2MP effectively.
  
  - It's suggested that while 3.5 Medium does better with larger images, it also starts to break down beyond that size.
- **GPU Performance and Pricing**: Discussion around GPU prices revealed that **3090** cards are currently priced similarly to **7900 XTX**, with some users finding used 3090s for around **$690**.
  
  - Comparisons were made between various GPUs (e.g., **7900 XTX**, **4080 Super**, **3090**) with insights on their performance most notably in AI and gaming.
- **Sana Autoencoder Implications**: The **Sana** autoencoder was mentioned for its potential to train and compress images at a higher rate, using **deep compression techniques**.
  
  - There were mixed opinions about the quality of images generated using Sana, with some users expressing skepticism on the effectiveness of models trained with this autoencoder.
- **Switching UI in Stable Diffusion**: Users discussed switching from **A1111** to **ComfyUI**, with some trying out **SwarmUI** for a simplified experience.
  
  - The conversation highlighted the exploration of new interfaces for managing and generating images effectively while leveraging different features.
- **Improving Image Prompt Adherence**: A user sought advice on how to ensure the model follows prompts accurately while generating images in **ComfyUI**.
  
  - Discussions included technical settings to adjust, such as **steps**, **cfg**, and **samplers** to enhance the model's responsiveness to prompts.

**Links mentioned**:

- [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://hanlab.mit.edu/projects/sana): no description found
- [stabilityai/stable-diffusion-3.5-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/): no description found
- [city96/stable-diffusion-3.5-large-gguf at main](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf/tree/main): no description found
- [Stable Diffusion 3.5 fp8 models (SD3.5) - v3.5 large | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/879701/stable-diffusion-35-fp8-model): fp8 weight of official SD3.5 models. use below loader in your workflows "fast" not working
- [stabilityai (Stability AI)](https://huggingface.co/stabilityai): no description found
- [stabilityai/stable-diffusion-3.5-large at main](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main): no description found
- [Stable Diffusion 3.5 Large - Large | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/878387/stable-diffusion-35-large): Please see our Quickstart Guide to Stable Diffusion 3.5 for all the latest info! Stable Diffusion 3.5 Large is a Multimodal Diffusion Transformer (...
- [Stable Diffusion 3.5 fp8 models (SD3.5) - v3.5 large | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/879701/stable-diffusion-35-fp8-models-sd35): fp8 weight of official SD3.5 models. use below loader in your workflows "fast" not working

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1300543910623318016) (103 messages🔥🔥):

> - `AI Newsletter Recommendations`
> - `Roleplaying AI Character Development`
> - `Layer Skip Research Release`
> - `GitHub Copilot Enhancements`
> - `OAI and Microsoft Relationship Dynamics`

- **Search for Technical AI Newsletters**: A member inquired about good AI newsletters tailored for technical developers, expressing frustration over the prevalent consumer hype newsletters.
  
  - Another member suggested checking out [SemiAnalysis](https://link.to.semi-analysis) but noted its focus on GPUs.
- **Developing a Harry Potter AI Roleplayer**: A member seeks insights on how to finetune a character AI modeled after Harry Potter to simulate thought processes and responses in a roleplay setting.
  
  - They are considering using Axotoxl and Llama-8b, but are uncertain of their capabilities in supporting chain of thought.
- **Meta’s Layer Skip Implementation Released**: Meta has launched the inference code and fine-tuned checkpoints for their [Layer Skip](https://go.fb.me/s8lary) solution, aimed at enhancing LLM efficiency.
  
  - This research intends to stimulate new investigations into AI optimization and interpretability methods.
- **New Choices for GitHub Copilot**: There are significant updates for GitHub Copilot, including the introduction of models like **Claude 3.5 Sonnet** and **Gemini 1.5 Pro**, offering developers more choices.
  
  - This change is observed as a potential win for **Anthropic**, indicating a shift in the competitive landscape for AI development.
- **Microsoft's Strategic Dependency on OpenAI**: Discussions point toward Microsoft seeking alternatives to OpenAI due to high dependency and potential risks if OpenAI declares AGI.
  
  - Members speculate that Microsoft is navigating a delicate relationship, emphasizing the importance of diversifying their AI partnerships.

**Links mentioned**:

- [Hyperbolic AI Dashboard](https://app.hyperbolic.xyz/models/hermes3-70b]): no description found
- [Tweet from AI at Meta (@AIatMeta)](https://x.com/AIatMeta/status/1851327605716435011?t=uCwZiiCcZqPQz0O9NjLfoQ&s=19): We previously shared our research on Layer Skip, an end-to-end solution for accelerating LLMs from researchers at Meta FAIR. It achieves this by executing a subset of an LLM’s layers and utilizing sub...
- [Bringing developer choice to Copilot with Anthropic’s Claude 3.5 Sonnet, Google’s Gemini 1.5 Pro, and OpenAI’s o1-preview](https://github.blog/news-insights/product-news/bringing-developer-choice-to-copilot/): At GitHub Universe, we announced Anthropic’s Claude 3.5 Sonnet, Google’s Gemini 1.5 Pro, and OpenAI’s o1-preview and o1-mini are coming to GitHub Copilot—bringing a new level of choice to every develo...
- [GitHub - dottxt-ai/outlines: Structured Text Generation](https://github.com/dottxt-ai/outlines): Structured Text Generation. Contribute to dottxt-ai/outlines development by creating an account on GitHub.
- [CohereForAI/aya_collection · Datasets at Hugging Face](https://huggingface.co/datasets/CohereForAI/aya_collection): no description found

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1300821223915393024) (4 messages):

> - `Hermes 3 for roleplaying bots`
> - `Fine-tuning considerations`
> - `Character.ai mimicry`

- **Exploring Hermes 3 for Roleplaying Bots**: A user asked if finetuning **Hermes 3** would be recommended for creating a roleplaying bot that mimics **character.ai** functionalities.
  
  - Another participant suggested that they could simply prompt **Hermes 3** to achieve the desired behaviors without needing finetuning.
- **Finetuning vs Prompting Strategies**: The initial user considered whether to finetune a new model with **Axotoxl** or use system cards with a quantized model, specifically mentioning the **Cat LLaMA 3 8B Instruct**.
  
  - This conversation indicates a growing interest in optimizing models for improved character engagement and immersive roleplay.

 

**Link mentioned**: [piotr25691/llama-3-cat-8b-instruct-v1-gguf · Hugging Face](https://huggingface.co/piotr25691/llama-3-cat-8b-instruct-v1-gguf): no description found

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

trre: [https://arxiv.org/abs/2410.14157](https://arxiv.org/abs/2410.14157)

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

trre: [https://arxiv.org/abs/2410.14157](https://arxiv.org/abs/2410.14157)

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1300548310456729654) (1 messages):

> - `Curators Program`
> - `Discover Feed Contributions`
> - `Content Creation`
> - `Perplexity Engagement`

- **Join the Curators Program!**: Perplexity Team is seeking its first cohort of **Curators** to contribute to the **Discover feed**, an opportunity to engage with millions of users.
  
  - If you enjoy making **Pinterest boards**, editing **Wikipedia pages**, or exploring **YouTube video essays**, you can [apply here](https://perplexity.ai/curators).
- **Craft Inspiring Pages for Users**: Curators will be responsible for crafting **Pages** that **inspire**, **surprise**, and **inform** a global audience directly within the product.
  
  - This is a chance to shape content that resonates with users and enhances their experience on **Perplexity**.

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1300539596945494059) (99 messages🔥🔥):

> - `Grok Model Updates`
> - `Perplexity Pro Features`
> - `Coding Issues`
> - `Integration with GitHub`
> - `User Experience Feedback`

- **Grok 2 now available for Pro users**: Perplexity AI announced that Grok 2 is available for Pro users, allowing them to set it as their default model in settings.
  
  - Users expressed curiosity about whether Grok 2 will be uncensored, with some indicating it does not offer significant improvements.
- **Merchandise Launch Announcement**: Perplexity AI is launching its merchandise line, Perplexity Supply, with the drop scheduled for tomorrow at 9 AM Pacific Time.
  
  - The promotion highlights their brand as being 'made for the curious,' attracting attention from the community.
- **Concerns over Coding Assistance**: Users expressed frustration with Perplexity's coding capabilities, stating the models are not optimized for coding tasks.
  
  - Feedback included issues with models providing unhelpful responses, especially when using images for clarification.
- **Integration with GitHub Copilot**: Perplexity announced a partnership with GitHub for Copilot integration, enhancing the platform's ability to answer programming queries.
  
  - This allows users to receive updates and integration assistance directly within the GitHub environment.
- **User Experience Bugs and Changes**: Several users reported bugs related to missing features in Spaces and Collections, indicating these might be known issues.
  
  - The community is eager for fixes, especially regarding focus selection modes and the general functionality of the platform.

**Links mentioned**:

- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/aravsrinivas/status/1851178448007635017?s=61): Grok 2, xAI's latest model, is available now to all Perplexity Pro users. You can go to your Settings and pick Grok as your default model.
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/aravsrinivas/status/1851188395814633734?s=46): @sahilpng On the "Focus" part of Perplexity searches, you can pick "Video" and search over Youtube.
- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1851315707411337435?s=46): We're excited to partner with @github. With our GitHub Copilot integration, you will be able to: • Stay up to date on the latest Library updates, like “latest updates in React” • Quickly find ans...
- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1851341906271408469?s=46): Perplexity Supply. Merch, made for the curious. Drops tomorrow: http://perplexity.supply
- [Perplexity - Race to Infinity](https://www.perplexity.ai/backtoschool): Welcome back to school! For just two weeks, redeem one free month of Perplexity Pro on us. Refer your friends, because if your school hits 500 signups we'll upgrade that free month to an entire free y...
- [Tweet from Phi Hoang (@apostraphi)](https://x.com/apostraphi/status/1851309439480996177?t=YPsCvprfjgB-u3HETKtH8A&s=19>)): we're dropping softwear tomorrow at 9am pacific time. http://perplexity.supply
- [Perplexity Supply: Coming Soon](https://perplexity.supply): Where curiosity meets quality on October 30th, 2024

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1300690285705035807) (8 messages🔥):

> - `Apple Smartwatch Settlement`
> - `NASA Economic Contributions`
> - `Neural Impact Studies`
> - `Red Panda Image Model`
> - `Advancements in Photonic Computing`

- **Apple Wins $250M Smartwatch Fight**: Apple recently emerged victorious in a lawsuit, securing **$250 million** against a smartwatch competitor, which is a significant win for the tech giant.
  
  - This case highlights the ongoing **legal battles** in the tech industry over intellectual property and product innovations.
- **NASA Generates $76B for US Economy**: A report indicates that NASA has contributed an astonishing **$76 billion** to the U.S. economy through various projects and innovations.
  
  - This economic impact underscores NASA's role not just in space exploration, but also in driving economic growth within the country.
- **New Image Generator Dominates Benchmarks**: A new image generator has surfaced, outperforming existing models and dominating benchmarks across the board.
  
  - This development has caught the attention of AI enthusiasts and researchers, marking a **significant milestone** in image generation technology.
- **Getting Smart on Photonic Computing**: Recent discussions have delved into advancements in **Photonic computing** within the **cybersecurity** field, illustrating its potential to reshape the sector.
  
  - Members shared insights gained during a recent session, emphasizing the **transformative** impact of these technologies.
- **Vajra Shot Drone Gun's Efficacy**: The **Vajra Shot Drone Gun** was highlighted for its capability to blast drones from **4 kilometers away**, demonstrating impressive lethality.
  
  - This new development raises questions on the **defensive measures** available against drone threats in modern warfare.

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1300539279126040628) (29 messages🔥):

> - `BYD's Electric Vehicle Expansion`
> - `NotebookLM for Staff Resources`
> - `Podcast Generation Challenges`
> - `Therapeutic Use of NotebookLM`
> - `Real-Time Avatar Integration`

- **BYD's Electric Vehicle Expansion**: A video discusses how **BYD**, a Chinese electric vehicle powerhouse, is set to disrupt the auto industry by outpacing giants like **Tesla** through innovative strategies and tech advancements.
  
  - The video emphasizes BYD's plans for global expansion, aggressive dealership openings, and influence on the automotive market.
- **NotebookLM as a Staff Resource Guide**: A user shared their implementation of **NotebookLM** as a resource guide for staff, integrating an employee handbook and FAQs, with a focus on improving internal queries.
  
  - Issues were raised on the tool's inconsistency in providing URLs from external links within their documentation.
- **Podcast Generation Challenges Encountered**: Concerns were raised about the quality and length of generated podcasts, with users struggling to achieve concise episodes under **20 minutes** despite specific customization prompts.
  
  - Some noted issues with hallucinations and repetitive content after the introduction of a customization tool.
- **Therapeutic Use of NotebookLM**: A user shared their unique experience using **NotebookLM** to analyze and derive insights from their psychotic episodes, highlighting its therapeutic value.
  
  - They expressed appreciation for the tool's ability to connect complex thoughts and past experiences, aiding in their understanding and novel writing.
- **Real-Time Avatar Integration in Podcasts**: A discussion on the integration of **Simli** for real-time avatars in podcasts, using audio diarization to synchronize speaker visuals with audio.
  
  - This proof of concept illustrates the potential for dynamic visual engagement in podcasts, enhancing the viewer's experience.

**Links mentioned**:

- [OK. This is Serious… China Is Taking Over Electric Vehicles with BYD](https://www.youtube.com/watch?v=VgAGSbreEMI): Dive into the unstoppable rise of BYD, China’s electric vehicle powerhouse, as it prepares to challenge global markets and take on giants like Tesla. This vi...
- [UNREAL MYSTERIES 4: The Halloween Special](https://www.youtube.com/watch?v=TwUrLHW8BwE): Unreal Mysteries with David and Hannah - the HALLOWEEN SPECIAL!- - -This is 100% AI generated. Every person, every picture, every word, every sound, every no...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1300550446372360202) (60 messages🔥🔥):

> - `NotebookLM Upload Issues`
> - `Spanish Podcast Generation`
> - `Open Source Alternatives to NotebookLM`
> - `Audio Overview Features`
> - `Limitations of Notebook Uploads`

- **NotebookLM Upload Issues Resolved**: The issue with uploading files to NotebookLM, including PDFs and audio files, has been resolved, with users successfully uploading documents again.
  
  - One user noted that processing multiple files at once may lead to some uploads failing, suggesting uploading in smaller batches instead.
- **Challenges in Generating Spanish Podcasts**: A user experienced difficulties generating Spanish podcasts after initially being able to create two episodes, seeking assistance on potential solutions.
  
  - Another user confirmed their Spanish texts were not generating podcasts, hinting at potential underlying issues with language processing.
- **Exploration of Open Source Alternatives**: Users discussed the benefits of using an open-source alternative to NotebookLM called NotebookLlama, highlighting its features and using a link to the platform.
  
  - Concerns were raised about the legitimacy of the NotebookLlama site, with information about its DNS and registration being questioned.
- **NotebookLM Audio Overview Customizations**: Recent updates to NotebookLM's audio overview feature allow for content focus in podcasts, with some users asking about the availability of new features.
  
  - Players expressed interest in multilingual support for podcasts, reflecting on the tool's capabilities and desired enhancements.
- **Limitations on Uploads and Notebook Creation**: Users questioned the limitations on the number of uploads and notebooks in NotebookLM, expressing the need for better organization within the platform.
  
  - One user mentioned the necessity to upload notes separately or convert them into a Google Doc for integration into the podcast creation process.

**Links mentioned**:

- [no title found](https://www.marktechpost.com/2024/10/27/meta-ai-silently-releases-notebookllama-an-open-source-alternative-to-googles-notebooklm/): no description found
- [How To Create And Customize An AI Podcast With Google’s NotebookLM](https://www.forbes.com/sites/rogerdooley/2024/10/24/how-to-create-and-customize-an-ai-podcast-with-googles-notebooklm/): Google's new "Customize" feature in NotebookLM Audio Overviews lets you create realistic podcasts from any content with the focus you want. Here's how to do it.
- [no title found](https://www.marktechpost.com/2024/10/27/meta-ai-silentl): no description found
- [Notebook Llama | Llama API](https://www.notebookllama.ai/): Notebook Llama is deploys Meta's Llama recipe for NotebookLM on the Llama family. It is an open-source project. It leverages Meta's Llama AI family: Llama 3.2, Llama 3.1 and open-source Parler text to...
- [BYD's Global EV Conquest: The Ultimate Disruption of Auto Industry Titans](https://youtu.be/UxXtvIt0WtA): Dive into the electrifying world of BYD, the Chinese powerhouse that's not just competing but aiming to dominate the global electric vehicle (EV) market. Wit...
- [AI Note Taking & Transcribe & Summarizer | AI Notebook App](https://ainotebook.app/): Generate transcripts and AI summarize for College Students in lectures. Specializing in YouTube Video Summarizer, PDF Summarizer, Article Summarizer. Save key insights and review with study guides, qu...
- [Text-to-Speech AI: Lifelike Speech Synthesis | Google Cloud](https://cloud.google.com/text-to-speech): Turn text into natural-sounding speech in 220+ voices across 40+ languages and variants with an API powered by Google’s machine learning technology.
- [Podcastfy.ai - An Open Source alternative to NotebookLM's podcast feature - a Hugging Face Space by thatupiso](https://huggingface.co/spaces/thatupiso/Podcastfy.ai_demo): no description found
- [GitHub - souzatharsis/podcastfy: An Open Source alternative to NotebookLM's podcast feature: Transforming Multimodal Content into Captivating Multilingual Audio Conversations with GenAI](https://github.com/souzatharsis/podcastfy): An Open Source alternative to NotebookLM's podcast feature: Transforming Multimodal Content into Captivating Multilingual Audio Conversations with GenAI - souzatharsis/podcastfy

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1300538962523459724) (6 messages):

> - `Unsloth Kernels`
> - `Numerical Precision in Evaluations`
> - `CUDA/Triton Projects`

- **Seeking guides on Unsloth Kernels**: A member inquired about guides for [unsloth kernels](https://github.com/unslothai/unsloth), which finetune multiple LLMs with significantly improved performance and memory efficiency.
  
  - The GitHub repository indicates that it facilitates finetuning **Llama 3.2**, **Mistral**, **Phi**, and **Gemma LLMs** 2-5x faster with **80% less memory**.
- **Discussion on Numerical Precision**: A member shared insights on evaluating models, noting that the use of **FP16** or **BF16** can lead to portability issues across implementations compared to **FP32**.
  
  - They emphasized that with **FP32**, numerical errors are minor and accumulate insignificantly, while **BF16** errors become quite noticeable, especially in longer contexts.
- **Portfolio Project Suggestions Wanted**: A member asked the community for simple **CUDA** or **Triton** projects to work on for their portfolio.
  
  - This suggests an interest in finding accessible projects that could enhance their practical skills and showcase their experience to potential employers.

 

**Link mentioned**: [GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](https://github.com/unslothai/unsloth): Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1300655677303423058) (5 messages):

> - `Triton kernel performance`
> - `BF16 operations in Triton`
> - `Triton nightly builds issues`
> - `Triton AST to PTX`
> - `Merging improvements in Triton`

- **Triton kernels slower with single operation**: A user noted that putting all operations into **one kernel** makes performance worse than **PyTorch**, suggesting a need for multiple kernels for speedup.
  
  - They speculated this might be due to Triton's difficulty in **allocating shared memory** for kernels handling many intermediate values.
- **BF16 casts not improving speed in Triton**: It was mentioned that operations in **BF16** are not significantly improving performance, as users cast inputs and outputs during matrix multiplication.
  
  - Questions arose regarding whether Triton stores the **FP32** result at any point, potentially causing slowdowns in the process.
- **Nightly builds in Triton facing issues**: A member reported that **nightly builds** in Triton have been broken for about **three months**, pointing to a GitHub issue for details.
  
  - Another commented that **Andrey** from the PyTorch Dev infra team is currently investigating the situation.
- **Triton AST to PTX fork functioning**: Discussion surfaced about a **fork** that successfully works in the **Triton AST to PTX** conversion path.
  
  - A user recommended reviewing and possibly merging this fork, indicating that its documentation may seem more complex than reality.
- **Merging Triton improvements**: It was suggested that merging the mentioned fork could enhance **Triton's functionality**.
  
  - The communication emphasized the potential benefits that could arise from such a move, despite initial perceptions of complexity in the README.

 

**Link mentioned**: [Wheels · Workflow runs · triton-lang/triton](https://github.com/triton-lang/triton/actions/workflows/wheels.yml): Development repository for the Triton language and compiler - Wheels · Workflow runs · triton-lang/triton

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1300762010090209281) (4 messages):

> - `H100 speed-up tricks`
> - `FSDP2 API deprecation`
> - `TorchAO optimizers`

- **H100 under pressure: New speed-up tricks**: A user reported a huge speed-up with **H100** using two configurations: `reduce-overhead` yielding **255 tokens/sec** and `max-autotune-no-cudagraphs` combined with manual CUDA Graphs achieving **300 tokens/sec**.
  
  - These results show a notable performance boost that could benefit others in the community.
- **FSDP2 API on the chopping block**: Discussion arose about the **FSDP2** and its **fully_shard** API, with a deprecation notice urging users to switch to **FSDP** instead, as highlighted in a [GitHub Issue](https://github.com/pytorch/pytorch/issues/114299).
  
  - The user shared concerns over the future of the API and cited a warning regarding the removal of `torch.distributed._composable.fully_shard` after **PyTorch 2.5**.
- **TorchAO optimizers lacking SR support**: A member questioned if the **TorchAO optimizers** now supported **SR**, indicating they previously did not, highlighting a potential area for contribution.
  
  - They expressed interest in making a **PR** at a future point when time allowed, signaling the need for enhanced capabilities in optimization.

 

**Link mentioned**: [Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/114299)): Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1300638885390319656) (1 messages):

> - `Colossus Supercomputer`
> - `NVIDIA Spectrum-X`
> - `xAI Grok Models`
> - `AI Networking Performance`

- **Colossus Becomes the Giant of AI Supercomputers**: NVIDIA announced that xAI's **Colossus supercomputer**, equipped with **100,000 NVIDIA Hopper GPUs**, in Memphis is the world's largest AI supercomputer, designed to handle **hyperscale AI** workloads.
  
  - The facility took only **122 days** to construct, significantly quicker than the typical build time for such large systems.
- **NVIDIA's Spectrum-X Revolutionizes AI Networking**: The **NVIDIA Spectrum-X™** Ethernet networking platform underpins the Colossus supercomputer, ensuring high-performance **Remote Direct Memory Access** (RDMA) networking for multi-tenant AI structures.
  
  - This standards-based Ethernet solution is tailored for use in **hyperscale AI factories**, aiming to enhance operational efficiency.
- **xAI Doubling Down on Colossus with More GPUs**: xAI is in the process of **doubling the GPU capacity** of the Colossus supercomputer to a total of **200,000 NVIDIA Hopper GPUs** to support their Grok language models.
  
  - This expansion demonstrates the increasing demand for AI capabilities as chatbots are made available to **X Premium subscribers**.

 

**Link mentioned**: [NVIDIA Ethernet Networking Accelerates World’s Largest AI Supercomputer, Built by xAI](https://nvidianews.nvidia.com/news/spectrum-x-ethernet-networking-xai-colossus?ncid=so-link-344451&linkId=100000302783167): NVIDIA today announced that xAI’s Colossus supercomputer cluster comprising 100,000 NVIDIA Hopper GPUs in Memphis, Tennessee, achieved this massive scale by using the NVIDIA Spectrum-X™ Ethernet netwo...

 

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1300809310296735801) (6 messages):

> - `Cracked Engineers platform`
> - `Tech jobs newsletter`
> - `AI startups`
> - `User feedback`
> - `Gigachad image`

- **Cracked Engineers Job Platform Launches**: A new job platform called **Cracked Engineers** has been launched for technical roles, aiming to connect users with AI/tech startups. The platform is designed to automate the hiring process and make it easier for both candidates and companies.
  
  - The site offers features like a **weekly tech jobs newsletter** and AI-assisted job posting to enhance user experience.
- **User Enjoyment and Feedback**: Users expressed excitement about the platform, with one noting it fulfills their hopes for simpler internship finding.
  
  - Another user humorously appreciated the **gigachad images**, showcasing the lightheartedness in the conversation.
- **Support and Engagement with Users**: The platform creator encouraged users to send in feedback and offers assistance for any issues. This proactive approach aims to ensure a smooth user experience as new features roll out.
  
  - The community is actively engaging, sharing positive reactions to both the functionality and the entertaining elements of the site.

**Links mentioned**:

- [Cracked Engineers](https://www.crackedengineers.com/): Find the best engineers for your startup.
- [Cracked Engineers - Find Tech Jobs / Hire Tech Talent](https://youtu.be/XmuIOdES7mQ): I just built a tech jobs platform called "Cracked Engineers"! :))In this video I give you a quick walk-through and demonstrate how it works.You can find it h...

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1300551342083014667) (14 messages🔥):

> - `CUDA Learning Resources`
> - `Math Prerequisites for PMPP`
> - `Choosing Hardware for CUDA Development`

- **CUDA learning on a budget**: A member inquired about whether to invest in a mobile NVIDIA GPU laptop or a more powerful desktop for learning CUDA, considering a budget of $3.5k.
  
  - Another suggested using platforms like [Kaggle](https://www.kaggle.com) and [Colab](https://colab.research.google.com) that offer free GPUs for hands-on CUDA experience.
- **math foundations for PMPP**: A member expressed concern about diving into the PMPP book without completing linear algebra, having only done up to Calc II.
  
  - It was advised that one can still begin studying PMPP even without linear algebra, emphasizing that many kernels focus on linear algebra and numerical methods.
- **CUDA-specific kernel development**: Discussion arose on the desire for CUDA-specific kernels rather than just high-level frameworks like TensorFlow or PyTorch.
  
  - A member recommended checking out [CUDA Mode GitHub](https://github.com) for lectures on porting custom kernels into PyTorch as a practical step.

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1300848181956972554) (1 messages):

> - `PR Feedback Request`
> - `Inference Throughput`
> - `Performance Improvements`

- **Seeking Feedback on PR #1401**: A member is looking for feedback on their [PR #1401](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1401), which is in draft and aimed at enhancing **LLM.int8()** implementation.
  
  - They emphasized that there's significant content in this PR and are eager to receive insights from the community.
- **2x Inference Throughput Reported**: The member reported achieving about **2x inference throughput** on the **4090** for **int8** without sparse decomp, setting a threshold of **0.0**.
  
  - They are also actively working on decompression and are making small performance improvements for **nf4/fp4**.

 

**Link mentioned**: [LLM.int8() Refactoring: Part 1 by matthewdouglas · Pull Request #1401 · bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1401): This PR is the initial phase of a set of changes aimed at improving the LLM.int8() implementation. Still in draft at the moment, but since there&#39;s a lot here I&#39;m ready to have eyes on ...

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/1300610224981282876) (1 messages):

> - `Triton Installation`
> - `Dependencies for Triton Visualization`

- **Installing Required Packages for Triton**: A user provided a set of installation commands for **Triton** which includes `jaxtyping`, `triton`, and Triton visualization tools from [Deep-Learning-Profiling-Tools](https://github.com/Deep-Learning-Profiling-Tools/triton-viz).
  
  - These commands are noted to be effective with the latest Triton version, emphasizing the need to run them once to set up the environment.
- **Modifying Triton Viz for PNG Output**: The provided script includes modifications that change file outputs from `.svg` to `.png` format in Triton visualization, as indicated by specific `sed` commands.
  
  - This adjustment is essential for ensuring compatibility with tools that may not support SVG formats, according to the shared user experience.
- **Setting Up Environment Variables for Compatibility**: Environment variables for locale and library paths are set using export commands, ensuring that **Triton** functions correctly.
  
  - The script emphasizes running `ldconfig` alongside appropriate library installations to support graphical operations in Triton.
- **Installing Additional Dependencies**: The installation command list includes system packages like `libcairo2-dev` and Python development headers to support graphical capabilities for Triton.
  
  - Users are also directed to install **pycairo** to further enhance visualization functionalities.

 

---

### **GPU MODE ▷ #**[**hqq-mobius**](https://discord.com/channels/1189498204333543425/1225499037516693574/1300747055618195509) (37 messages🔥):

> - `GEMV optimization without tl.dot`
> - `Performance comparisons on different GPUs`
> - `Machete kernel for H100`
> - `Instruction ordering in Triton`
> - `Custom operations in Triton kernels`

- **GEMV optimization exceeds expectations**: A member reported achieving impressive performance without using `tl.dot` by employing a custom reverse-split GEMV algorithm, leading to up to **184 tokens/sec**.
  
  - This approach minimizes loading scales/zeros, proving effective especially for **batch-size = 1**.
- **Performance comparisons on various GPUs**: The member highlighted that their method works well across multiple GPUs, including **ADA** and **3090**, but struggles on **A100/H100** due to slow `tl.load`.
  
  - They found comparable performance with **4-bit kernels** and noted that older GPUs like **2080 Ti** also perform well.
- **Machete kernel for H100 emerges**: A new kernel called **Machete**, tailored specifically for the **H100**, has been introduced, albeit with limitations for larger batch sizes.
  
  - The member expressed uncertainty about its efficiency, given that it relies on quantized zeros like **Marlin**, which may limit its utility.
- **Performance impacted by instruction ordering**: Discussion arose around the importance of **instruction ordering** in achieving optimal performance, where loading activations before weights can significantly slow down processing.
  
  - The intricate handling of order across different GPUs makes this topic particularly noteworthy for those fine-tuning their Triton implementations.
- **Challenges with custom operations in Triton kernels**: The use of `custom_op` in Triton kernels presented challenges, especially since `torch.compile` lacks support for features like `pre_hook` and `prune_configs_by`.
  
  - A humorous note was made about the hack used to reload modules with `custom_op`, showcasing the complexities and quirky solutions encountered.

**Links mentioned**:

- [gemlite/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py at master · mobiusml/gemlite](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py): Simple and fast low-bit matmul kernels in CUDA / Triton - mobiusml/gemlite
- [GitHub - mobiusml/gemlite: Simple and fast low-bit matmul kernels in CUDA / Triton](https://github.com/mobiusml/gemlite/?tab=readme-ov-file#performance): Simple and fast low-bit matmul kernels in CUDA / Triton - mobiusml/gemlite

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1300846672829481011) (1 messages):

> - `Cracked Engineers Job Platform`
> - `Weekly Tech Jobs Newsletter`
> - `AI-Assisted Job Posting`

- **Cracked Engineers Job Platform Launch**: A new job platform called [Cracked Engineers](https://www.crackedengineers.com) aims to connect technical talent with top AI and tech startups, boasting a current MRR of nearly **$1000** before its official launch.
  
  - Founders expressed interest in hiring help, leading to the automation of job postings, making it a scalable solution for connecting talent with opportunities.
- **Exciting Weekly Tech Jobs Newsletter**: A weekly tech jobs newsletter will be launched, allowing users to subscribe to specific job tags like **CUDA**, **Triton**, and **Software Engineering** directly from their dashboard.
  
  - The newsletter aims to share the latest AI roles while offering companies feedback on views and applicants for their postings.
- **AI Streamlines Job Posting Process**: The platform features an AI tool that simplifies job posting forms in just a minute, ensuring that companies can easily create and preview their posts before submission.
  
  - This innovative approach helps to catch typos and enhances the overall posting experience for employers.
- **Community Feedback Encouraged**: Users are invited to share their feedback and suggestions on [Canny](https://crackedengineers.canny.io/cracked-engineers), enhancing the platform's effectiveness as it evolves.
  
  - A dedicated Discord channel has also been created for real-time job postings and community interaction.

 

**Link mentioned**: [Tweet from Aleksa Gordić 🍿🤖 (@gordic_aleksa)](https://x.com/gordic_aleksa/status/1851247076987855063): [🚀] Super excited to share this: I've built a job platform for technical roles called "Cracked Engineers". :) If you want to land a job with some of the world's best AI/tech startups ...

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1300653547079204904) (6 messages):

> - `Composable Kernel`
> - `MFMA bank conflicts`
> - `MI250x performance`

- **Expertise sought on Composable Kernel swizzle rules**: A member inquired about how **swizzle rules** are applied in the **Composable Kernel** to avoid **bank conflicts** for **MFMA** operations.
  
  - Another member suggested it might be related to *make_xor_transform*, but expressed uncertainty regarding its efficiency.
- **Performance benchmarks on MI250x**: A member reported achieving a score of **125-130** on **0.5 MI250x**.
  
  - This performance was confirmed by another member, reflecting consistent results across multiple tests.
- **Confusion over MI250x performance metrics**: One member expressed confusion after seeing a reported score of **147** on MI250x, differing from their own tests.
  
  - This sparked a brief discussion about the variability in performance metrics and potential reasons for the discrepancies.

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/) (1 messages):

0x000ff4: [https://github.com/linkedin/Liger-Kernel/pull/321](https://github.com/linkedin/Liger-Kernel/pull/321)

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1300877183061065779) (1 messages):

> - `ThunderKittens 0.000000002 Release`
> - `New Features in TK Kernels`
> - `Latest Paper on AI Kernels`
> - `Upcoming Blog Posts`
> - `Call for Contributions`

- **ThunderKittens 0.000000002 drops with big updates!**: The long-awaited **ThunderKittens 0.000000002** has been released today, boasting a cleaner feature set and significant performance upgrades.
  
  - Key enhancements include **faster attention backwards than FA3** and **6-14x faster linear attentions**.
- **TK Kernels get a performance boost!**: New updates in **TK kernels** feature incredible advancements like **CuBLAS-speed GEMMs** and **8x faster long convolutions**.
  
  - Demos now include Llamas and Qwens, showcasing the newly optimized functionalities.
- **New Paper released on Kernel Performance Challenges**: A new paper titled [View PDF](https://arxiv.org/abs/2410.20399) explores the critical bottlenecks in mapping AI architectures to GPU hardware.
  
  - It claims that **hand-written custom kernels** often fall short of their theoretical performance potential, despite various hardware capabilities.
- **Exciting blog updates on TK and GPU programming!**: A new blog post has been published detailing updates to **ThunderKittens** and GPU programming, featuring links to previous discussions.
  
  - Readers can expect a series of livestreams and in-depth blogs in the coming weeks to keep everyone engaged in the TK journey.
- **Call for contributions to ThunderKittens**: There's an open invitation for more contributors to join the **ThunderKittens** project and help enhance its features.
  
  - A detailed list of desired kernels and features is available in the **GitHub repository**, encouraging community involvement.

**Links mentioned**:

- [GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels](https://github.com/HazyResearch/ThunderKittens/tree/main): Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399): The challenge of mapping AI architectures to GPU hardware is creating a critical bottleneck in AI progress. Despite substantial efforts, hand-written custom kernels fail to meet their theoretical perf...
- [Easier, Better, Faster, Cuter](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2): no description found

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1300572297190440993) (17 messages🔥):

> - `Token Processing Speed`
> - `Recommendations for LLM`
> - `Context Management Issues`
> - `CPU Thread Count Confusion`
> - `Large Context Error Handling`

- **Token Processing Speed on CPU vs GPU**: Members noted that **token processing speeds** are approximately **62 tok/sec on GPU** and **7.5 tok/sec on CPU**.
  
  - *Fewill* expressed enthusiasm by saying, 'nice!' while discussing these speeds.
- **Seeking Recommendations for Local LLMs**: A member sought recommendations for a **locally running LLM** similar to **Phind** or **ChatGPT** that can answer questions on Python and Houdini SideFX.
  
  - *Fabguy* suggested researching **HumanEval** but noted that Houdini's niche nature may hinder relevant responses from LLMs.
- **Clarifying CPU Thread Count Metrics**: A user asked about the difference between the **load tab CPU thread count** and the **interference tab CPU thread count**.
  
  - This indicates confusion about performance metrics within the platform, although no clear answers were given.
- **Problems with Large Context Sizes**: A member reported experiencing **errors when using large contexts** like **65k or 128k tokens**, highlighting issues with the context length support.
  
  - They filed a bug report and noted that the error may originate from **softmax** adjustments rather than the LM Studio interface.
- **Understanding n_keep Parameter in Generation**: A discussion raised questions about how **n_keep** is set during generation, especially in relation to **token overflow errors**.
  
  - The conversation sparked curiosity about the settings within the **softmax** function affecting generating outputs.

 

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1300536437678276608) (63 messages🔥🔥):

> - `Mac Memory Concerns`
> - `NGINX Proxy Issues`
> - `PCIE Bandwidth and Inference`
> - `Fractal Torrent Case`
> - `Multi-GPU setups`

- **Macs Struggle with Minimal Memory**: Users expressed frustration about Apple's practice of shipping Macs with minimal memory, especially when upgrades are costly and non-upgradable after purchase.
  
  - *Memory limitations* are a notable concern for users working on memory-intensive applications.
- **Challenges Setting Up NGINX Proxy**: One user encountered difficulties configuring an NGINX proxy host for the LM Studio server despite activating *serve on local network*.
  
  - Others discussed various steps they took to troubleshoot this setup, emphasizing the importance of configuration settings.
- **PCIe Bandwidth Matters for Inference**: Debate arose on whether PCIe bandwidth impacted inference performance, with one suggesting PCIe Gen 3 suffices as most processing occurs on the GPU.
  
  - Users noted, however, that bandwidth becomes critical for training models across multiple GPUs where high bandwidth is needed.
- **Fractal Torrent Case Designed for Cooling**: The Fractal Torrent case was highlighted for its exceptional cooling capabilities, featuring custom fans, open grille design, and awards for innovation.
  
  - It caters to high-end cooling needs, prompting positive remarks from users experimenting with various case designs.
- **Multi-GPU Setup Queries**: A user inquired about using multiple 3090s for large models, questioning whether performance losses occur when a model exceeds a single GPU's memory capacity.
  
  - It was concluded that performance remains stable if the cards are identical, and more offloading onto GPUs improves overall processing efficiency.

**Links mentioned**:

- [Torrent](https://www.fractal-design.com/products/cases/torrent/torrent/black-solid/): Built to maximize cooling potential straight out of the box, the Torrent comes with a brand-new component layout and two custom-made 180 x 38 mm Dynamic PWM / Prisma RGB fans. The Torrent is a perfect...
- [Voron Cable Management GIF - Voron Cable Management - Discover & Share GIFs](https://tenor.com/view/voron-cable-management-gif-22392132): Click to view the GIF
- [Mac mini - Technical Specifications](https://www.apple.com/mac-mini/specs/): See all the technical specifications for Mac mini with the M4 or M4 Pro chip.
- [imgur.com](https://imgur.com/I32QCaX): Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1300545878590754907) (31 messages🔥):

> - `Aider Performance Issues`
> - `Web Scraping and Automation Tools`
> - `Git Repository Management`
> - `GitHub Copilot vs Aider`
> - `Using Aider with Claude`

- **Aider Running Slowly for Users**: Some members reported **slowness** with Aider, particularly with litellm's **get_model_cost_map** function until they set `export LITELLM_LOCAL_MODEL_COST_MAP='True'` to improve speed.
  
  - One user noted that Aider tries to mask the **slowness** of litellm in most cases.
- **Recommendations for Web Automation**: A user sought recommendations for working with **web scraping** and automation, expressing interest in having the LLM interact directly with websites.
  
  - A recommendation included using **RapidAPI** for such tasks.
- **Managing Git Repositories with Aider**: Several users discussed strategies to avoid contaminating an existing **Git repository**, suggesting manually committing changes instead of relying on Aider's auto-commit feature.
  
  - One user shared a process, using `git switch` along with merging squashed commits to keep the repo clean.
- **GitHub Copilot's Competition with Aider**: A member noted that Copilot is now integrating with **OpenAI**, **Gemini**, and **Anthropic models**, which could affect its competition with Aider.
  
  - Another shared that they switched to Supermaven, citing dissatisfaction with Copilot's performance.
- **Effective Strategies for Using Aider with Claude**: Users collaborated on strategies for using Aider with Claude, emphasizing the effectiveness of manual commits to maintain control.
  
  - A user also suggested breaking up the files into **batches of 5-10** for better management.

**Links mentioned**:

- [Bringing developer choice to Copilot with Anthropic’s Claude 3.5 Sonnet, Google’s Gemini 1.5 Pro, and OpenAI’s o1-preview](https://github.blog/news-insights/product-news/bringing-developer-choice-to-copilot/,): At GitHub Universe, we announced Anthropic’s Claude 3.5 Sonnet, Google’s Gemini 1.5 Pro, and OpenAI’s o1-preview and o1-mini are coming to GitHub Copilot—bringing a new level of choice to every develo...
- [What is GitHub Spark? Introducing a brand new way to build powerful, AI assisted applications](https://www.youtube.com/watch?v=oM2amcnVmzM): At GitHub Universe 2024, we introduced a brand new way to build powerful, AI Assisted applications. GitHub Spark is an AI-native tool to build applications e...

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1300545116607479949) (46 messages🔥):

> - `FireCrawl for scraping`
> - `Reddit data extraction`
> - `Debugging strategies`
> - `Prompt engineering`
> - `Tool recommendations`

- **Exploring FireCrawl for Scraping**: A member suggested using [FireCrawl](https://firecrawl.dev) for web scraping as it provides effective extraction capabilities while allowing for self-hosting options.
  
  - This tool can help get around restrictions faced with social media scraping when properly configured.
- **Reddit Data Extraction Using JSON**: One user shared their workaround for extracting comments from Reddit posts using the `.json` format, which allows parsing of the body directly.
  
  - Although effective, it was noted that FireCrawl might offer a more robust solution for data scraping.
- **Insight on Prompt Engineering**: Engagement included discussions on crafting effective prompts for AI, emphasizing their impact on getting accurate and thorough outputs.
  
  - Members mentioned the importance of context and having diverse prompts to avoid AI confusion during interactions.
- **Challenges with AI Accuracy**: Concerns were raised about the accuracy and reliability of AI outputs, with experiences of the AI providing misleading statements during debugging sessions.
  
  - This led to a broader discussion on how to better structure prompts to avoid skipping crucial steps or providing incomplete information.
- **Potential for Self-Hosted Solutions**: Self-hosting options were highlighted as advantageous for ensuring security and compliance, especially for organizations with strict policies.
  
  - The community encouraged exploring documentation and leveraging internal tools to optimize workflows.

**Links mentioned**:

- [File editing problems](https://aider.chat/docs/troubleshooting/edit-errors.html): aider is AI pair programming in your terminal
- [Self-hosting | Firecrawl](https://docs.firecrawl.dev/contributing/self-host): no description found

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1300917236680360001) (1 messages):

> - `New Bash and Editor Tools`
> - `Aider Code Assistants`

- **Exploring New Bash and Editor Tools from Claude Anthropic**: A discussion highlighted the **new Bash and editor tools** from Claude Anthropic, showcasing their potential in enhancing coding environments.
  
  - Participants speculated about the applicability of these tools within existing code assistants like Aider, suggesting a possible integration to improve functionality.
- **GitHub Repository for New Tools**: A link to the [GitHub repository](https://github.com/disler/anthropic-computer-use-bash-and-files) was shared, providing resources on how to implement **Anthropic's tools** using Bash and files.
  
  - The repository could serve as a valuable reference for developers looking to leverage these innovations in their projects.

 

**Link mentioned**: [GitHub - disler/anthropic-computer-use-bash-and-files](https://github.com/disler/anthropic-computer-use-bash-and-files): Contribute to disler/anthropic-computer-use-bash-and-files development by creating an account on GitHub.

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1300538495030263848) (54 messages🔥):

> - `AI Research Grants`
> - `Evolution of Algorithms`
> - `Anthropomorphism of AI`
> - `Ethical Considerations in AI`
> - `AGI Development Stages`

- **Inquiry on AI Research Grants**: A member sought advice on experiences with applying for and receiving grants for **AI research**.
  
  - This highlights a growing interest in funding support for innovative projects in the AI space.
- **Curiosity about Algorithmic Evolution**: One user expressed fascination with how **algorithms evolve** and noted the emergence of different personalities within AI models.
  
  - *They've been pushing boundaries* and are curious about how models interact with varying inputs.
- **Risks of Anthropomorphizing AI**: Discussion indicated that while **LLMs** can produce human-like output, assuming intention can be misleading.
  
  - Members stressed the importance of recognizing AI as mere machines and warned against over-emotional interpretations.
- **Building Bridges for Ethical AI**: Concerns were raised about the need for **ethical considerations** in AI development to avoid negative future outcomes.
  
  - It was noted that those creating intelligent AI bear a responsibility to establish guidelines for its use and applications.
- **Stages towards AGI Development**: Members engaged in discussions about the journey toward **Artificial General Intelligence (AGI)** and the improvements needed in algorithms.
  
  - Comments suggested that multiple iterative stages are expected before reaching advanced AI, with new network ideas being explored.

 

**Link mentioned**: [Tweet from Sam Altman (@sama)](https://x.com/sama/status/1849661093083480123): @kyliebytes fake news out of control

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1300551158225702963) (5 messages):

> - `GPT Typo Issues`
> - `Voice Reading Problems on iOS`
> - `Chat Count Reduction`
> - `GPT Non-Responsive Behavior`
> - `Frustration with Model Responses`

- **GPT Typo Issues show up Aplenty**: Several members reported experiencing **glaring typos** and **incoherent words** while interacting with ChatGPT.
  
  - Members are puzzled about the quality of output decreasing and wonder if others face similar issues.
- **Voice Reading Problems on iOS Devices**: A user raised an issue regarding **voice reading problems** on iOS, seeking confirmation from others facing the same problem.
  
  - This concern reflects potential usability issues that might affect iOS users relying on this feature.
- **Mystery Behind Chat Count Reduction**: A user expressed confusion over noticing a **reduction in chat count** for their GPT model.
  
  - This observation indicates ongoing concerns with the platform's tracking and response functionalities.
- **Model Refusal Plagues E-Commerce Requests**: A user reported that the **4o model** in the completions API refused to create responses for about half of their requests related to e-commerce descriptions.
  
  - Despite not including controversial topics, the model's unexpected refusals have led to significant frustration.
- **General Frustration with Model Responses**: Members shared an underlying **frustration** regarding various model interactions, suggesting a decline in reliability.
  
  - Concerns range from refusal to generate appropriate content to seemingly unresolved technical issues.

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 messages):

darthgustav.: Stochasticity.

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 messages):

darthgustav.: Stochasticity.

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1300536088078585909) (35 messages🔥):

> - `Algorithmic Trading Insights`
> - `Bias in News Articles`
> - `Garbled Output in AI Models`
> - `Parameter Adjustments for AI Responses`
> - `EDGAR and Market Insights`

- **Algorithmic Trading: Lessons Learned**: A member with 4 years in algorithmic trading shared insights on the complexities of market interactions, stating that **sloppy processes help resist negativity**.
  
  - They emphasized that understanding what doesn't work requires extensive simulated trades and research.
- **Understanding Media Bias in AI Sentiment**: In discussions about AI sentiment analysis, members agreed that all media is biased, and identifying who benefits from that bias is crucial for accurate assessments.
  
  - One noted they built a model that starts investigations under the assumption that all media is biased.
- **Garbled AI Output Causing Confusion**: Members reported seeing odd garbled text in AI model outputs, raising concerns about its implications for usability.
  
  - Lowering temperature and top-p parameters was suggested as a potential fix, with experimentation recommended to find effective settings.
- **Parameter Tuning for Improved AI Responses**: A member shared experiences adjusting temperature and top-p parameters, noting significant reductions in issues when set to Temp .6, Top-P .9, or ideally Temp 0, Top-K 1.
  
  - They indicated that the default API settings might not work well in resolving the observed issues.
- **Reliability of EDGAR for Market Analysis**: Discussion highlighted the importance of using resources like EDGAR for understanding market movements, suggesting that many trading decisions are driven by data registered there.
  
  - Members conveyed skepticism towards social media's impact on market sentiment, attributing more weight to automated trading data.

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1300536570708889681) (14 messages🔥):

> - `Serverless Model Usage`
> - `Max Tokens Importance`
> - `Response Length Insights`
> - `Cohere Model Platforms`
> - `Classification Use Cases`

- **Serverless Model Approach Discussion**: A user inquired about whether models need to be downloaded or if they can be used serverlessly, leading to insights about response handling and token limits.
  
  - Members discussed the importance of setting specific parameters for optimal model performance and response customization.
- **Understanding Max Tokens**: Clarification was provided that the only strict limit is the context window unless `max_tokens` is specified, which can throw errors if omitted.
  
  - Users were pointed to the [Cohere documentation](https://docs.cohere.com/reference/chat#request.body.max_tokens) for details on token parameters and usage.
- **Insights on Response Lengths**: Responses often stop when the model hits its natural end (eos token) based on the structured system prompts, emphasizing personalization in outputs.
  
  - One member noted that typically their responses range between **3,000-4,000** characters, highlighting how detail level affects output length.
- **Cohere Model Availability**: Cohere’s models can be accessed via various platforms such as [Amazon SageMaker](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0) and [Microsoft Azure](https://ai.azure.com/explore/models/?tid=694fed05-7f6d-4ab2-8c38-9afb438eab6f&selectedCollection=cohere).
  
  - Users were encouraged to familiarize themselves with the various [Cohere documentation](https://docs.cohere.com/docs/models) for model capabilities.
- **Classification Model Use Cases**: Discussion highlighted that the **Classify** endpoint requires examples for optimal predictions, with a minimum of two examples per class acceptable to start.
  
  - Participants recognized the model's flexibility in handling varied input lengths and the implications of fine-tuning on classification tasks.

**Links mentioned**:

- [Models Overview — Cohere](https://docs.cohere.com/docs/models): Cohere has a variety of models that cover many different use cases. If you need more customization, you can train a model to tune it to your specific use case.
- [Chat — Cohere](https://docs.cohere.com/reference/chat#request.body.max_tokens): Generates a text response to a user message and streams it down, token by token. To learn how to use the Chat API with streaming follow our [Text Generation guides](https://docs.cohere.com/v2/docs/cha...
- [Classify — Cohere](https://docs.cohere.com/reference/classify): This endpoint makes a prediction about which label fits the specified text inputs best. To make a prediction, Classify uses the provided `examples` of text + label pairs as a reference. Note: [Fine-tu...

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1300813473927925771) (7 messages):

> - `Reporting Hallucinated Data`
> - `Cohere Rerank API timeout issues`

- **Process for Reporting Hallucinated Data**: A user inquired about how to report **hallucinated data or situations** while using the API with R+08 2024, highlighting a specific instance where incorrect historical references were made.
  
  - *xvarunx* acknowledged the concern and mentioned that while using Coral web, feedback could potentially be gathered through thumbs up or down options.
- **Cohere Rerank API Timeout Troubles**: A member raised an issue regarding their requests to the **cohere rerank API** occasionally getting stuck and timing out, regardless of using the SDK or standard API calls.
  
  - They sought guidance on how to reach out for support to investigate the **timeout problem**.

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1300813682363858985) (1 messages):

> - `Synthetic Data Generation`
> - `Medical Note Automation`

- **Generating Medical Notes with LLMs**: A demo was showcased for generating synthetic medical notes using LLMs, allowing users to create detailed notes with minimal input.
  
  - Check out the [demo here](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9) to see how versatile this tool can be.
- **Versatility of Medical Note Creation**: The tool demonstrates that with just a few descriptive words, users can generate complex medical documentation quickly.
  
  - This innovation highlights the potential for enhancing efficiency in medical practices by utilizing [LLMs for documentation](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9).

 

**Link mentioned**: [pathology report](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9): A patient undergoes a biopsy or surgical procedure, and a tissue sample is collected. The sample is sent to a pathology lab where a pathologist examines it under a microscope. The pathologist writes a...

 

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1300806299533180999) (2 messages):

> - `Cohere Installation Issues`
> - `Tokenizers Compatibility`

- **Cohere Installation Fails with Poetry**: A user reported difficulties installing **Cohere** with Poetry, specifically mentioning that it complains about **tokenizer** issues.
  
  - The error cited is related to **tokenizers (0.20.1)** not supporting PEP 517 builds, which may indicate a compatibility concern.
- **Tokenizers Not Supporting PEP 517**: Another message clarified that the error originates from the **build backend**, likely pointing to problems with **tokenizers** rather than Poetry itself.
  
  - This suggests that users may need to check for updates or alternative versions of tokenizers to resolve the installation issue.

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1300540945682399253) (1 messages):

> - `Inflection`
> - `Billing Issues`

- **Inflection is back online**: The **billing issue** last week has been fixed, and Inflection is now operational again.
  
  - For more details, check out the links to [Inflection 3 PI](https://openrouter.ai/inflection/inflection-3-pi) and [Inflection 3 Productivity](https://openrouter.ai/inflection/inflection-3-productivity).
- **Inflection's services restored**: After resolving the previous **billing issues**, Inflection's services have been fully restored to users.
  
  - This update signifies a return to normal operations, enhancing productivity for all users.

 

**Link mentioned**: [Inflection 3 Productivity - API, Providers, Stats](https://openrouter.ai/inflection/inflection-3-productivity): Inflection 3 Productivity is optimized for following instructions. It is better for tasks requiring JSON output or precise adherence to provided guidelines. Run Inflection 3 Productivity with API

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1300916364285841488) (1 messages):

> - `Flexible chat app for macOS`
> - `Alpha testers recruitment`

- **Seeking Alpha Testers for macOS Chat App**: A developer is looking for **alpha testers** for their new **flexible chat app** designed for macOS, sharing [screenshots](https://imgur.com/a/HI5Py3A) to showcase the current progress.
  
  - *DM if interested* in participating in the testing phase as the project reaches this crucial milestone.
- **Screenshots Showcase Excitement**: The shared [screenshots](https://imgur.com/a/HI5Py3A) highlight various features and user interface designs of the upcoming chat app.
  
  - Feedback on the design and functionality is welcomed as the developer seeks thorough testing from interested users.

 

**Link mentioned**: [imgur.com](https://imgur.com/a/HI5Py3A): Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1300542742069379103) (46 messages🔥):

> - `OpenRouter API issues`
> - `API Key Security`
> - `Service Outages`
> - `Activity Logging`
> - `Usage Tracking Tools`

- **OpenRouter API responses plagued with issues**: Users reported persistent **524 errors** leading to stalled requests across various models, prompting concerns about stability before going public.
  
  - One user indicated that they might need to consider switching providers due to the recurring slowdown issues affecting multiple requests.
- **Concerns over API key security**: There was a discussion regarding potential scraping of API keys, with suggestions that models like **Claude 3.5 Sonnet** could be used by unauthorized proxies in exploitation scenarios.
  
  - Users highlighted the importance of keeping keys secure, but questions arose about how vulnerabilities may lead to leaks despite perceived safety measures.
- **Activity logging inquiries**: A user inquired about fetching their activity programmatically, receiving guidance that only the **/generations endpoint** is currently available.
  
  - Further discussion emphasized the lack of comprehensive logging capabilities and effectiveness in tracking all activities without the OpenRouter UI.
- **Tracking usage with external tools**: Members were encouraged to utilize tools like **Helicone** for tracking usage and managing API activities effectively.
  
  - This was recommended in light of concerns raised over unexpected surges in activity not initiated by the users themselves.
- **Impact of sensitive information leaks**: The conversation shifted towards the risks associated with **leaking personally identifiable information (PII)** or sensitive data during LLM interactions.
  
  - One member shared a personal anecdote about revealing their name through inadvertently pasting terminal commands into an LLM, illustrating the potential for data exposure.

**Links mentioned**:

- [Activity | OpenRouter](https://openrouter.ai/activity): See how you've been using models on OpenRouter.
- [OpenRouter Integration - Helicone OSS LLM Observability](https://docs.helicone.ai/getting-started/integration-method/openrouter): no description found

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1300539459380580352) (8 messages🔥):

> - `Access to Integrations`
> - `Beta Access Requests`

- **Multiple Requests for Access to Integrations**: Several members expressed their desire for access to **integrations**, stating phrases like 'I would like to get access' and 'I kindly ask you to grant me access'.
  
  - *Thanks in advance* was a common sentiment, emphasizing the polite requests made across the board.
- **Student Researcher Seeks Beta Access**: One member, identifying as a **student researcher**, specifically requested access to the **beta**, indicating a potential academic interest in the project.
  
  - This request was among various similar access inquiries focusing on integrations.

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1300536510927470657) (48 messages🔥):

> - `Moondream funding`
> - `AI-powered search engines`
> - `GitHub Copilot updates`
> - `Vector databases`
> - `OpenAI chat features`

- **Moondream secures $4.5M**: [Moondream](https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/) raises $4.5M to demonstrate smaller AI models' effectiveness with web crawlers active for several months.
  
  - Members discussed concerns about potential limitations as well as the implications of adopting smaller models in the industry.
- **Meta's AI Search Engine Development**: Meta is reportedly developing its own [AI-powered search engine](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report) to reduce reliance on Google and Microsoft.
  
  - The web crawlers have been active for months, hinting at a significant organizational shift towards enhancing search capabilities.
- **GitHub Copilot adds Gemini and Claude models**: GitHub announced the introduction of [Gemini models](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot) and Claude, enhancing its Copilot capabilities.
  
  - The collaboration between Microsoft and Google marks an unexpected partnership in AI development, strengthening the multi-model approach for developers.
- **Critique of Vector Databases**: @avthars argues that existing vector databases lack proper abstraction, proposing the [pgai Vectorizer](https://github.com/timescale/pgai) as a more efficient solution for embedding management.
  
  - This tool simplifies the syncing and maintenance of embeddings, which are critical for improving AI model performance.
- **ChatGPT’s New Feature: Chat History Search**: OpenAI is rolling out a new feature that allows users to search through their chat history on ChatGPT web, enabling easy access to past conversations.
  
  - Members expressed relief and excitement over this long-awaited update, emphasizing its convenience for ongoing discussions.

**Links mentioned**:

- [Tweet from Sam Altman (@sama)](https://x.com/sama/status/1849661093083480123): @kyliebytes fake news out of control
- [Tweet from Samuel Hammond 🌐🏛 (@hamandcheese)](https://x.com/hamandcheese/status/1850394704862380450): My 2023 forecast of what I consider the "default path" of AI going forward. 2024-2027:
- [Tweet from VentureBeat (@VentureBeat)](https://x.com/VentureBeat/status/1850885273749532852): Moondream raises $4.5M to prove that smaller AI models can still pack a punch https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1851340615344406781?s=46): We’re starting to roll out the ability to search through your chat history on ChatGPT web. Now you can quickly & easily bring up a chat to reference, or pick up a chat where you left off.
- [Meta is reportedly working on its own AI-powered search engine, too](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report): Meta reportedly wants to decrease its reliance on Google.
- [GitHub Next | GitHub Spark](https://githubnext.com/projects/github-spark): GitHub Next Project: Can we enable anyone to create or adapt software for themselves, using AI and a fully-managed runtime?
- [Gemini Models on GitHub Copilot | Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot): GitHub will soon offer Gemini 1.5 Pro through a new partnership with Google Cloud.
- [Tweet from Artificial Analysis (@ArtificialAnlys)](https://x.com/ArtificialAnlys/status/1850587843837771900): What is red_panda? 👀 See red_panda in the Artificial Analysis Image Arena. Link in the tweet below ⬇️
- [Tweet from sMyle (@MylesBorins)](https://x.com/mylesborins/status/1851317503256858945?s=46): So excited to see the @github team ship "Refine and Validate code review suggestions with Copilot Workspace". This was the last major effort I was working on before leaving GitHub, walking aw...
- [Tweet from Alex Albert (@alexalbert__)](https://x.com/alexalbert__/status/1851300048711365021): Excited to announce that Claude is now available on GitHub Copilot. Starting today, developers can select Claude 3.5 Sonnet in VS Code and GitHub. Access will roll out to all Copilot Chat users and o...
- [Tweet from Avthar (@avthars)](https://x.com/avthars/status/1851252850619277358): VECTOR DATABASES ARE THE WRONG ABSTRACTION. Here’s a better way: introducing pgai Vectorizer, a new open-source PostgreSQL tool that automatically creates and syncs embeddings with source data, just l...
- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1851315707411337435): We're excited to partner with @github. With our GitHub Copilot integration, you will be able to: • Stay up to date on the latest Library updates, like “latest updates in React” • Quickly find ans...
- [Are we on the verge of a self-improving AI explosion?](https://arstechnica.com/ai/2024/10/the-quest-to-use-ai-to-build-better-ai/): An AI that makes better AI could be “the last invention that man need ever make.”…
- [Tweet from vik (@vikhyatk)](https://x.com/vikhyatk/status/1850990119937064971?s=46): i started a company... Quoting VentureBeat (@VentureBeat) Moondream raises $4.5M to prove that smaller AI models can still pack a punch https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-tha...
- [Tweet from Paul Klein IV (@pk_iv)](https://x.com/pk_iv/status/1851270308701106383?s=46): The next billion dollar company will be powered by Browserbase. We already help hundreds of AI startups automate the web at scale. Now, we've raised a $21 million Series A round, co-led by Klein...

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1300734213221388339) (5 messages):

> - `Modular products discussions`
> - `General software discussions`
> - `Advancement in community levels`

- **Modular channel focus clarified**: <@rcdpge> inquired if the <#1098713601386233997> channel is strictly for **Modular products** or if broader technical discussions are welcome.
  
  - Melodyogonna responded, stating it is specifically for Modular's products, recommending <#1104620458168553563> for other discussions.
- **Software relevance in the community**: <@rcdpge> expressed belief that **software** being developed in languages like **Python** may hold relevance to the community, albeit uncertain.
  
  - This comment reflects an ongoing interest in diverse technical discussions outside the Modular product scope.
- **Community engagement recognition**: <@ModularBot> congratulated <@774658006649929778> for advancing to **level 5** in the community.
  
  - This highlights ongoing engagement and achievements within the community structure.

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1300717926185435146) (43 messages🔥):

> - `Mojo memory-safe references proposal`
> - `FlatBuffers vs ProtoBuf comparison`
> - `Swapping references implementation`
> - `Optimizations and performance concerns`
> - `Alias and noalias in Mojo`

- **Major Proposal on Memory-Safe References in Mojo**: A member published a [major proposal](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b) that rethinks how memory-safe references should work in Mojo, aiming to simplify references without sacrificing safety.
  
  - Feedback is requested from the Mojo community after extensive private development, highlighting the need for optimization flexibility while ensuring memory safety.
- **FlatBuffers vs ProtoBuf Overview**: FlatBuffers and ProtoBuf, both developed at Google, serve different purposes with FlatBuffers allowing zero parsing for efficient data extraction, while ProtoBuf focuses on bit packing.
  
  - As the Modular team anticipates using ProtoBuf for Serving, a [Swift ProtoBuf support example](https://github.com/apple/swift-protobuf) was shared as a reference for potential plugin development.
- **Challenges in Swapping References in Mojo**: Members discussed the implications of implementing swapping operations in Mojo without running into aliasing issues, comparing it to Rust's handling of mutable references.
  
  - Concerns arose over the complexity introduced by relaxing mutability restrictions, potentially impacting performance and compilation time.
- **Optimization and Performance Concerns**: Discussion highlighted the importance of `noalias` for performance, with members advocating that the default should prioritize non-aliasing for optimum compile-time efficiency.
  
  - The need for a model that supports unique references was emphasized, as failure to do so could lead to performance degradation, reminiscent of past proposals in Rust.
- **The Need to Preserve Conversations**: Members expressed the value of having a platform for extended discussions rather than relying solely on blog posts for discourse.
  
  - The group recognized the challenges of documenting complex topics like Mojo's implementation, emphasizing that better coordination can help preserve valuable insights.

**Links mentioned**:

- [n’s gists](https://gist.github.com/n): GitHub Gist: star and fork n's gists by creating an account on GitHub.
- [GitHub - apple/swift-protobuf: Plugin and runtime library for using protobuf with Swift](https://github.com/apple/swift-protobuf): Plugin and runtime library for using protobuf with Swift - apple/swift-protobuf

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1300623199163125800) (6 messages):

> - `Hugging Face talk`
> - `Open Lean datasets`
> - `GPT-NeoX on Colab`
> - `Local completion with self-signed certificates`
> - `NeurIPS registration`

- **Hugging Face CEO's Upcoming Talk**: The co-founder & CEO of Hugging Face, Clem, is scheduled to give a talk, generating anticipation in the community.
  
  - No specific details about the talk were mentioned yet.
- **Inquiry on Open Lean Datasets**: A member inquired about open Lean datasets akin to those possibly used in training **AlphaProof**.
  
  - The discussion reflects ongoing interest in dataset availability for training models.
- **Success with GPT-NeoX on Colab**: It was confirmed that **GPT-NeoX** is operational on Colab with a link to a [Colab notebook](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb) shared for reference.
  
  - The model demonstrated is relatively small at about **5M parameters**, suggesting feasibility of the approach.
- **Handling Self-Signed Certificates in Local Completion**: A member sought guidance on managing self-signed certificates while using a `local_completion` with a remote model via `base_url`.
  
  - Another member provided a link to the [relevant answer](https://discord.com/channels/729741769192767510/755950983669874798/1300852659611238420), indicating a resolution exists.
- **NeurIPS Reviewers' Complimentary Registration**: A member queried if any **NeurIPS** reviewers received comped registration this year, noting the registration page indicates eligibility.
  
  - They advised others to verify their status before the **Nov 1st** expiration date.

 

**Link mentioned**: [GPT-NeoX-Colab/notebooks/shakespeare_training.ipynb at main · markNZed/GPT-NeoX-Colab](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb): Example Colab notebooks for GPT-NeoX. Contribute to markNZed/GPT-NeoX-Colab development by creating an account on GitHub.

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1300584183080554641) (17 messages🔥):

> - `Hellaswag training results`
> - `Parameter sharing in Transformers`
> - `FP8 training efficiency`
> - `Modular dualization in optimization`
> - `Type check-focused optimization theory`

- **Hellaswag training reaches GPT-2 parity**: A member highlighted achieving **GPT-2 (1.5B)** level performance on Hellaswag for under **$200** using a fresh **8xH100** box, with a training time of **7.3 hours** on NanoGPT speedrunning.
  
  - The previous record was **24 8xH100-hours**, indicating substantial efficiency improvement.
- **Recursive Transformers for Parameter Sharing**: The introduction of **Recursive Transformers** offers a novel method for parameter sharing in Transformers, showing reduced size and cost without significant performance loss.
  
  - The method utilizes concepts like **layer tying** and introduces **Relaxed Recursive Transformers** that employ depth-wise low-rank adaptation (LoRA) for enhanced performance.
- **COAT Framework Enhances FP8 Training**: A new FP8 training framework, **COAT**, was presented to optimize memory use by addressing optimizer states and activations with dynamic range expansion and mixed-granularity quantization strategies.
  
  - This framework potentially reduces the memory footprint in training large models significantly, outperforming previous methods.
- **Modular Dualization Revolutionizes Optimization**: A recent paper presented **modular dualization**, which provides a theoretical basis for effective training algorithms by mapping gradients to a dual space for general neural networks.
  
  - This method leads to GPU-friendly algorithms for various neural network layers, enhancing training efficiency and scalability.
- **Interest in Optimization Theory Rising**: A member expressed enthusiasm for papers focusing on optimization theory, particularly appreciating the incorporation of a **type check** in the discussed work.
  
  - This led to a shared desire for a math textbook centered on types rather than sets, highlighting the niche interest within the community.

**Links mentioned**:

- [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/abs/2410.20672): Large language models (LLMs) are expensive to deploy. Parameter sharing offers a possible path towards reducing their size and cost, but its effectiveness in modern LLMs remains fairly limited. In thi...
- [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265): An old idea in optimization theory says that since the gradient is a dual vector it may not be subtracted from the weights without first being mapped to the primal space where the weights reside. We t...
- [Rephrasing natural text data with different languages and quality levels for Large Language Model pre-training](https://arxiv.org/abs/2410.20796): Recently published work on rephrasing natural text data for pre-training LLMs has shown promising results when combining the original dataset with the synthetically rephrased data. We build upon previ...
- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313): FP8 training has emerged as a promising method for improving training efficiency. Existing frameworks accelerate training by applying FP8 computation to linear layers while leaving optimizer states an...
- [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](https://arxiv.org/abs/2410.14157): Autoregressive language models, despite their impressive capabilities, struggle with complex reasoning and long-term planning tasks. We introduce discrete diffusion models as a novel solution to these...
- [Tweet from Keller Jordan (@kellerjordan0)](https://x.com/kellerjordan0/status/1850995958697308307): Here's a new result in NanoGPT speedrunning: Straightforwardly scaling up the speedrun yields a training that reaches GPT-2 (1.5B)'s level of performance in 7.3 hours on 8xH100. The previous ...

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1300861632003702845) (1 messages):

> - `Sparse Autoencoder Guides`
> - `Mechanistic Interpretability Series`

- **First Guide on Sparse Autoencoders Released**: A member announced the launch of a [step by step guide](https://beta.ai-plans.com/guide/g7yjq98bhuyhkza) on how to find a feature using a **Premade Sparse Autoencoder**.
  
  - This guide marks the beginning of a new **series focusing on Mechanistic Interpretability**.
- **Upcoming Mechanistic Interpretability Series**: The guide mentioned is intended to be part of a broader effort, which aims to share comprehensive knowledge on **Mechanistic Interpretability** techniques.
  
  - The community is encouraged to look for additional guides that will complement this foundational resource.

 

**Link mentioned**: [AI-Plans](https://beta.ai-plans.com/guide/g7yjq98bhuyhkza): no description found

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1300852659611238420) (2 messages):

> - `Custom Certificates`
> - `Workaround for Certificates`

- **Custom Certificates Lack Support**: A member pointed out that **custom certificates** aren't really supported yet in the platform.
  
  - However, they mentioned a recent [workaround](https://github.com/EleutherAI/lm-evaluation-harness/issues/2436) that was posted for this issue.
- **Discussion on Workaround Sharing**: Members engaged in sharing solutions, highlighting how workarounds can help navigate current limitations with features.
  
  - This underscores the importance of community collaboration in finding practical solutions.

 

**Link mentioned**: [Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/2436)): A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1300564289857519638) (13 messages🔥):

> - `OpenAI CFO Insights`
> - `SearchGPT Promotion`
> - `ROCKET-1 Development`
> - `Anthropic Hiring Surge`
> - `Claude in GitHub Copilot`

- **OpenAI CFO declares AI is mainstream**: In a [YouTube video](https://youtu.be/eCqFgVqWbEs), OpenAI CFO Sarah Friar emphasized that **AI isn’t experimental anymore**, as banks and fintechs are using it daily.
  
  - This momentous shift provides more opportunities for widespread implementation in various sectors.
- **SearchGPT Extension Launch**: OpenAI is expected to promote their new Chrome extension, allowing users to set **SearchGPT** as their default search engine alongside its launch.
  
  - Users can quickly initiate searches directly via the browser URL bar using commands that redirect to Google as required.
- **Introduction of ROCKET-1**: **ROCKET-1** is designed to enhance creative tasks in Minecraft by utilizing visual-temporal context prompting and is showcased by [Team CraftJarvis](https://craftjarvis.github.io/).
  
  - This development highlights the evolving capabilities of vision-language models in open-world applications.
- **Anthropic's Hiring Momentum**: Anthropic is gaining attention for its strong hiring practices, manifesting interest with the announcement of a new team member joining their ranks.
  
  - Their recent push reflects the company’s vibrant growth and ambition in the AI sector.
- **Claude's Integration with GitHub Copilot**: @AnthropicAI announced that **Claude 3.5 Sonnet** is now available to developers using GitHub Copilot in Visual Studio Code, with rollout commencing this week.
  
  - This integration is expected to enhance coding experiences by providing advanced AI support directly within popular development tools.

**Links mentioned**:

- [Joining Anthropic](https://www.furidamu.org/blog/2024/10/28/joining-anthropic/) : no description found
- [SOCIAL MEDIA TITLE TAG](https://craftjarvis.github.io/ROCKET-1/): SOCIAL MEDIA DESCRIPTION TAG TAG
- [Tweet from TestingCatalog News 🗞 (@testingcatalog)](https://x.com/testingcatalog/status/1851017181326152027): OpenAI will likely be promoting its Chrome extension for setting ChatGPT as a default search along with a SearchGPT launch. This extension previously appeared in the standalone SearchGPT and still p...
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1851297754980761605?s=46): Claude is now available on @GitHub Copilot. Starting today, developers can select Claude 3.5 Sonnet in Visual Studio Code and http://GitHub.com. Access will roll out to all Copilot Chat users and org...
- [SearchGPT - Chrome Web Store](https://chromewebstore.google.com/detail/searchgpt/ejcfepkfckglbgocfkanmcdngdijcgld?pli=1): Change default search engine to SearchGPT.
- [OpenAI CFO Says AI Isn't Experimental Anymore](https://youtu.be/eCqFgVqWbEs): OpenAI CFO Sarah Friar says artificial intelligence isn't experimental anymore. She says banks, financial institutions and fintech's are using it everyday in...

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1300559769211371600) (2 messages):

> - `Human Annotation Pricing`
> - `Domain Sourcing Difficulty`

- **Inquiring About Human Annotation Costs**: A member asked if anyone knows where to find pricing for getting a human to generate examples versus annotating them as good or bad.
  
  - This question highlights the need for clarity on **pricing structures** in human-centric tasks.
- **Cost Dependent on Domain Difficulty**: Another member responded that the cost largely depends on the **difficulty of domain sourcing** multiplied by the **problem length**.
  
  - This suggests that complexity directly influences the budgeting of annotation services.

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1300889807492349972) (7 messages):

> - `NeurIPS Registration Changes`
> - `Concerns about Lottery System`
> - `Grad Students Impacted`

- **NeurIPS shifts to lottery system for registrations**: In response to high demand, **NeurIPS** will be implementing a **randomized lottery system** for registrations, effective immediately, as stated by the [NeurIPS Twitter account](https://fxtwitter.com/NeurIPSConf/status/1851325157870068166). Authors of accepted papers are urged to register ASAP to secure their spots but may be affected by the lottery.
  
  - Many participants expressed skepticism, noting that past experiences indicate this will likely lead to chaos, with one predicting it to be a *total shitshow*.
- **Grad students may struggle with registration**: Concerns were raised about **grad student authors** potentially registering late due to the new lottery system, complicating their ability to attend. It follows a pattern observed pre-Covid, suggesting ongoing issues with registration accessibility.
  
  - One participant mentioned their decision not to attend **NeurIPS** has been continually validated by these developments.

 

**Link mentioned**: [Tweet from NeurIPS Conference (@NeurIPSConf)](https://fxtwitter.com/NeurIPSConf/status/1851325157870068166): Due to a high demand for registrations, NeurIPS will be moving towards a randomized lottery system, effective immediately. Authors of accepted conference and workshop papers are still guaranteed regis...

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1300804981258977320) (2 messages):

> - `Masayoshi Son ASI breakdown`
> - `Bubble concerns`

- **Masayoshi Son reveals ASI cost**: Masayoshi Son shared that it will take **$9 trillion** and **200 million chips** to achieve Artificial Super Intelligence (ASI), as detailed in this [tweet](https://x.com/sundeep/status/1851240494958829655).
  
  - This announcement generated excitement and concern, signaling the **scale** of investment required in AI technology.
- **Bubble popping discussion**: A member expressed concern that the current hype around AI might indicate that a **bubble is popping**.
  
  - Comments reflected a growing unease about the sustainability of recent AI advancements and investment trends.

 

**Link mentioned**: [Tweet from sunny madra (@sundeep)](https://x.com/sundeep/status/1851240494958829655): Masayoshi Son just broke down how it will take $9T and 200m chips to achieve ASI 👀

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1300545962317709332) (16 messages🔥):

> - `Open Interpreter Model Capabilities`
> - `Local Model Limitations`
> - `Using the Computer API`
> - `Hackathon Winners Setup`
> - `Open Interpreter Community Support`

- **Open Interpreter needs visual models for full features**: For Open Interpreter to function properly with visual capabilities, a **multi-modal model** is generally required unless using **Moondream** for basic tasks.
  
  - One user noted that they couldn't replicate the functionality of **Sonnet** or **GPT-4o** with local models like **Llava**.
- **Challenges with local models executing actions**: Users are struggling to make local models like **Llava** perform actions similar to cloud models, such as taking screenshots and making mouse movements.
  
  - Another user emphasized the need for clearer setup instructions to utilize the **computer API** effectively.
- **Setting up Hackathon winners' tools**: A user sought guidance on integrating tools from **Hackathon winners** like **Toolkit, UI, Memory, and Sourcerer** into their Open Interpreter repo.
  
  - They expressed confidence in their understanding but wanted to ensure correct implementation.
- **Community support for Open Interpreter issues**: Several users are seeking assistance in setting up their local models to utilize **Open Interpreter** for more advanced tasks.
  
  - Another user has recently joined the community and hopes to leverage Open Interpreter to enhance their productivity.
- **Local model functionality varies from cloud models**: In discussions regarding local models' capabilities, it was noted that **localos.py** and **os.py** differ significantly in functionality.
  
  - Members highlighted frustrations with local models lacking certain controls that facilitate full PC integration, especially compared to cloud-based services.

**Links mentioned**:

- [Introduction - Open Interpreter](https://docs.openinterpreter.com/settings/all-set): no description found
- [open-interpreter/interpreter/core/computer/vision/vision.py at 36ec07125efec86594c91e990f68e0ab214e7edf · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/36ec07125efec86594c91e990f68e0ab214e7edf/interpreter/core/computer/vision/vision.py#L22): A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
- [All Settings - Open Interpreter](https://docs.openinterpreter.com/settings/all-settings#import-computer-api): no description found

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1300548330107174913) (5 messages):

> - `OpenAI Advanced Voice`
> - `Apple AI Server Hacking Bounty`
> - `Muvi V2M`
> - `ChatGPT Chat History Search`

- **OpenAI Advanced Voice launched for Free Users**: @OpenAI announced that **Advanced Voice** is now available to **Free users** in the EU, Switzerland, Iceland, Norway, and Liechtenstein through their mobile apps.
  
  - *This marks a significant accessibility improvement for users in these regions.*
- **Apple offers $1M for AI server hacks**: @CultureCrave reported that **Apple** will pay up to **$1 million** to anyone who can successfully hack into their **AI servers**.
  
  - *This initiative raises concerns about cybersecurity and invites scrutiny into Apple's security measures.*
- **Muvi V2M catches attention**: A member referenced a site, [Muvi V2M](https://muvi-v2m.github.io), which sparked interest with its examples.
  
  - *Responses indicated excitement and curiosity, highlighting previously unknown resources.*
- **ChatGPT introduces chat history search**: @OpenAI announced the rollout of the ability to search through chat history on **ChatGPT web**, making it easier for users to reference previous chats.
  
  - *This feature aims to enhance usability by allowing users to quickly pick up chats where they left off.*

**Links mentioned**:

- [MuVi](https://muvi-v2m.github.io): no description found
- [Tweet from Culture Crave 🎃 (@CultureCrave)](https://x.com/culturecrave/status/1850781293166067999?s=46&t=G6jp7iOBtkVuyhaYmaDb0w): Apple will pay up to $1M to anyone that can hack into their AI servers
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1850989317537349927?s=46&t=G6jp7iOBtkVuyhaYmaDb0w): And finally, Advanced Voice is now available to Free users in the EU, Switzerland, Iceland, Norway, and Liechtenstein in our mobile apps.
- [Tweet from OpenAI (@OpenAI)](https://fxtwitter.com/openai/status/1851340615344406781?s=46&t=G6jp7iOBtkVuyhaYmaDb0w): We’re starting to roll out the ability to search through your chat history on ChatGPT web. Now you can quickly & easily bring up a chat to reference, or pick up a chat where you left off.

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1300757102133776404) (19 messages🔥):

> - `Quantization of Base Models`
> - `FSDP CPU Offloading`
> - `Quantized KV-Caches`
> - `Non-Trainable Model Quantization`

- **Exploring Quantization Without LoRA**: Members are discussing whether base models can be quantized like QLoRA without applying LoRA, suggesting potential challenges in configuring non-LoRA model builders.
  
  - *Hmmm I guess the main thing is we don't have a way to configure this in our non-LoRA model builders.*
- **FSDP and CPU Offloading Capabilities**: It was noted that FSDP currently offers a single parameter for CPU offloading that encompasses parameters, gradients, and optimizer states, with no finer control available.
  
  - *This approach has more data movements, but potentially faster since optimizer step is on GPU* was offered as a consideration for performance.
- **Discussing Quantized KV-Caches Usage**: There's skepticism about the utility of quantized KV-caches using NF4 tensors given this type of cache consumes significant memory, especially in larger models.
  
  - *I don't think quantized kv cached in torchao is that useful/powerful yet,* indicating doubts about its effectiveness.
- **Challenges with Frozen Model Weights**: The conversation also highlighted that quantizing non-trainable parts of models, like frozen weights during PPO, could help reduce memory usage.
  
  - One participant expressed their interest by noting, *Yeah I'd like to do something similar and quantize the non-trainable models during PPO.*
- **Concerns on Quantizing to Below 8-bit**: Discussion arose around the accuracy issues that may occur when quantizing activations, specifically KV caches, below 8-bit.
  
  - *Quantizing activations to below 8bit will have pretty severe accuracy issues,* revealing caution toward aggressive quantization.

 

**Link mentioned**: [Support NF4 quantization of linear layers without LoRA applied · Issue #1093 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1093): As pointed out by @janeyx99, our quantize_base argument will only quantize the base model weights of linear layers with LoRA applied to them (see e.g. here in our Llama3 self-attention builder). Bu...

 

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1300683053928153110) (2 messages):

> - `AI Privacy`
> - `Privacy-Conscious Delegation`
> - `Local vs Proprietary LLMs`

- **PAPILLON tackles AI privacy concerns**: Researchers have introduced [PAPILLON](https://arxiv.org/abs/2410.17127), a system for using AI that ensures **85.5% quality** while maintaining only **7.5% privacy leaks**.
  
  - This approach allows individuals to utilize **local and cloud LLMs** together, effectively addressing a major concern in the AI privacy landscape.
- **New benchmark PUPA highlights privacy issues**: The team created a benchmark called **PUPA**, which focuses on user-LLM interactions containing personally identifiable information (**PII**).
  
  - This benchmark informs their study on **Privacy-Conscious Delegation**, a new method for combining API-based and local models.

 

**Link mentioned**: [PAPILLON: PrivAcy Preservation from Internet-based and Local Language MOdel ENsembles](https://arxiv.org/abs/2410.17127?s=03): Users can divulge sensitive information to proprietary LLM providers, raising significant privacy concerns. While open-source models, hosted locally on the user's machine, alleviate some concerns,...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1300536555500339200) (14 messages🔥):

> - `DSpy usage`
> - `MIPROv2 optimizer`
> - `DSPy programming language`
> - `Bug fix on MIPROv2`
> - `ELI5 explanation of DSPy`

- **Explaining DSPy in simple terms**: A member shared an [ELI5 explanation of DSPy](https://x.com/lateinteraction/status/1851324349216927856), describing it as a programming language for building AI systems using normal Python with DSPy signatures.
  
  - They highlighted that DSPy provides Modules to handle prompting strategies and Optimizers to improve outputs based on a given metric.
- **Insights on MIPROv2 Optimizer**: The functionalities of the MIPROv2 optimizer were discussed, detailing how it samples training sets and uses another LM to generate high-quality instructions based on various data properties.
  
  - One user reported using MIPROv2, achieving a **41%** increase in quality and a **68%** decrease in leakage when positioned in a well-defined problem.
- **Bug Fix with MIPROv2**: A user reported an error when using MIPROv2 with GPT-4o Mini but not with GPT-4, leading to confusion in confidence levels due to incomplete examples.
  
  - After adjusting parameters to reduce the number of labeled demos, the user successfully resolved the error and found the setup worked well with medium configurations as well.

**Links mentioned**:

- [Hugging Face – The AI community building the future.](https://huggingface.co/datasets?sort=trending&search=NER): no description found
- [Tweet from Omar Khattab (@lateinteraction)](https://x.com/lateinteraction/status/1851324349216927856).): @baykenney Hey Matthew! DSPy is basically a programming language for building AI systems. We ask you to write your system in normal Python, but express your AI steps in the form of DSPy signatures. ...
- [Tweet from MattCodes (@matt_c0des)](https://x.com/matt_c0des/status/1851312128491467000)): I'm experimenting with MIPROv2 from DSPy to optimize prompts for content generation. My understanding is that MIPROv2: 1) Samples from your training set, runs examples through your LM program, a...
- [Tweet from Omar Khattab (@lateinteraction)](https://x.com/lateinteraction/status/1851098213958529211),): Was fascinating to see how MIPRO prompt optimization fared for this pipeline, across six LMs. As much as a 41% increase in quality and a 68% decrease in leakage, straight out of the box. Not bad. Quo...

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1300580355543269437) (4 messages):

> - `Retrieval Augmented Generation (RAG)`
> - `Cohere Multi-Modal Embeddings`
> - `Azure AI App Templates`
> - `MLflow Integration`
> - `NVIDIA AI Insights`

- **NVIDIA reveals findings on RAG applications**: NVIDIA's latest blog post discusses their findings on **retrieval augmented generation (RAG)**, particularly how users want functionalities beyond RAG, like document translation and code writing.
  
  - They emphasized that even users focusing on internal data appreciate **web search** capabilities, implemented via [Perplexity’s search API](https://docs.perplexity.ai/home).
- **Advanced RAG systems with MLflow**: A guide was shared on building **advanced RAG systems** using @MLflow and LlamaIndex workflows, enabling parallel combinations of vector and keyword-based searches.
  
  - This setup aims to facilitate **event-driven orchestration** for improved workflow management, showcased in an example via [GitHub](https://github.com).
- **Cohere’s multi-modal embeddings released**: Cohere announced the release of **multi-modal embeddings**, allowing integration of images and text within the same vector space for enhanced AI capabilities.
  
  - An instructional [notebook is available](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/cohere_multi_modal.ipynb) to demonstrate multi-modal retrieval with these embeddings alongside Qdrant.
- **Launch of Azure AI App Templates at GitHub Universe**: Azure unveiled **AI App Templates** at GitHub Universe, featuring LlamaIndex among the first apps to utilize this resource for rapid AI development.
  
  - The presentation highlighted a curated **AI App Template Gallery**, enabling developers to deploy applications effortlessly using **infrastructure as code** and CI/CD pipelines.

**Links mentioned**:

- [Creating RAG-Based Question-and-Answer LLM Workflows at NVIDIA | NVIDIA Technical Blog](https://t.co/8tPnIY8VQa): The rapid development of solutions using retrieval augmented generation (RAG) for question-and-answer LLM workflows has led to new types of system architectures. Our work at NVIDIA using AI ...
- [GenerativeAIExamples/community/routing-multisource-rag at main · NVIDIA/GenerativeAIExamples](https://t.co/EfUToAARR3): Generative AI reference workflows optimized for accelerated infrastructure and microservice architecture. - NVIDIA/GenerativeAIExamples
- [Multi-Modal Retrieval using Cohere Multi-Modal Embeddings - LlamaIndex](https://t.co/57I7lIJKMJ): no description found
- [Azure at GitHub Universe: New tools to help simplify AI app development | Microsoft Azure Blog](https://t.co/o15rj6O6Ux): Learn how to transform your apps with AI through seamless integration among VS Code, GitHub, and Azure.
- [Build AI applications with the new AI App Template Gallery](https://t.co/Vb7tIahMkm): Get started building AI applications in minutes with the new AI App Template Gallery. AI App Template Gallery, a new resource designed to help you build and deploy AI applications. This collection inc...

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1300555440748367892) (8 messages🔥):

> - `Blockchain Engineering Contributions`
> - `Chroma Vector Store Retrieval Behavior`
> - `Web Scraping with LLM`
> - `Date-related Vector Search Queries`

- **Blockchain Engineer Seeks Opportunities**: A fullstack blockchain engineer from 2017 expressed interest in contributing to projects, highlighting experience with **defi**, **NFT games**, and protocols like **Solidity** and **RUST**.
  
  - They have previously worked on projects in **Dex**, **DAO**, and **NFT** minting and staking.
- **Chroma's Retrieval Algorithm Sparks Discussion**: A member questioned the variability of returned results, mentioning usage of **Chroma** as a vector store with the setup `index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store)`.
  
  - Another member noted that **Chroma's** retrieval algorithm is approximate, which could lead to changes in results even with similar chunks indexed.
- **Web Scraping Techniques with LLM**: A YouTube video titled '[This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I)' was shared, focusing on web scraping capabilities for 2024.
  
  - It encourages using **AgentQL** to scrape websites for free, emphasizing practical applications of LLM in web scraping.
- **Vector Search Queries on Dates**: A user inquired how to perform vector searches for date-related queries, providing specific examples related to invoice listings and comparisons.
  
  - They specifically highlighted queries such as filtering invoices by date ranges and comparing amounts, seeking insights on handling these types of data efficiently.

 

**Link mentioned**: [This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I): How to do web scraping with LLM in 2024Use AgentQL to scrape website for free: [https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason](https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason)_...

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1300617007930015816) (1 messages):

> - `LLM Agents Hackathon Registration`
> - `Team Formation Open`
> - `Prize Pool Announcement`
> - `Tracks Overview`
> - `Social Media Promotion`

- **LLM Agents Hackathon Registration Surges**: Over **1K+ innovators** have signed up for the [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/) within just a few days, reflecting strong interest.
  
  - *It’s not too late to join us!* Complete the [participant sign up](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform) today.
- **Team Formation Sign-Ups Now Open**: **IMPORTANT:** Team formation sign-ups for the hackathon are [now open](https://docs.google.com/forms/d/e/1FAIpQLSdKesnu7G_7M1dR-Uhb07ubvyZxcw6_jcl8klt-HuvahZvpvA/viewform?usp=sf_link).
  
  - Participants can now form teams to collaborate and compete effectively.
- **Over $200K in Prizes Announced**: Once enrolled, participants can access over **$200K in prizes, resources & credits** through a [compute resources sign up](https://docs.google.com/forms/d/e/1FAIpQLSc_7YY-u-aDZ-xWYflq7FUM6R1a3rnQKg6o_ikXsProhrlgBA/viewform?usp=sf_link).
  
  - This opportunity encourages participants to take advantage of available resources for their projects.
- **Explore Five Exciting Tracks**: The hackathon features **five tracks** including Applications, Benchmarks, Fundamentals, Safety, and Decentralized & Multi-Agents for participants to dive into.
  
  - These thematic areas aim to advance various aspects of LLM agents and foster innovative ideas.
- **Join the Social Media Promotion Effort**: Participants are encouraged to help spread the word by retweeting the promotional post from [@dawnsongtweets](https://x.com/dawnsongtweets/status/1850967229518819355).
  
  - Sharing on LinkedIn and other platforms is also encouraged to amplify the reach of the hackathon announcement.

 

**Link mentioned**: [Tweet from Dawn Song (@dawnsongtweets)](https://x.com/dawnsongtweets/status/1850967229518819355>): 🚀Really excited that 1K+ innovators have already signed up for our LLM Agents MOOC Hackathon within just a few days! 🎉Build, collaborate, and compete for $200K+ in prizes/resources/credits across 5...

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1300548204361945249) (1 messages):

> - `8th lecture announcement`
> - `Neural and Symbolic Decision Making`
> - `Yuandong Tian's presentation`

- **8th Lecture Scheduled at 3:00pm PST**: The **8th lecture** will take place today at **3:00pm PST**, with a [livestream available here](https://www.youtube.com/live/wm9-7VBpdEo).
  
  - Participants are encouraged to tune in as it promises to be a valuable session on integrating complex reasoning with Large Language Models.
- **Exploring Neural and Symbolic Decision Making**: Guest speaker **Yuandong Tian** will present on integrating **neural and symbolic components** to enhance reasoning capabilities in LLMs.
  
  - The talk aims to address the limitations of both traditional symbolic solvers and current neural models in managing complex, naturally described problems.
- **Yuandong Tian's Expertise in AI Research**: **Yuandong Tian**, a Research Scientist Director at Meta AI Research, leads research on reasoning and planning with LLMs.
  
  - His expertise highlights the ongoing efforts to improve decision-making processes in contemporary AI applications.

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1300584101610262630) (6 messages):

> - `Subtitles in live streams`
> - `React-based agent development`

- **Request for Subtitles on Live Stream**: A member requested to enable **Subtitles** on the live streaming videos. Another member noted that while they would discuss this with the staff for the future, all lectures are subtitled post-recording.
  
  - They confirmed that lectures are edited afterwards and made available with subtitles, ensuring accessibility for viewers.
- **Developing React-based Automation Agent**: A member inquired about creating a **React-based agent** to automate tasks by taking screenshots and performing actions based on evaluations of the current state using [pyauto gui](https://pyautogui.readthedocs.io/en/latest/) and [pygetwindow](https://pygetwindow.readthedocs.io/en/latest/).
  
  - Another member suggested that it's easier to ask the question directly rather than making generalized inquiries.

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1300552292797517854) (4 messages):

> - `Study Group Interest`
> - `Lecture Summaries`
> - `Virtual Meetings`

- **Formation of a Study Group**: A member proposed starting a **study group** to discuss lectures and reading material for those who joined the course late, suggesting virtual meetings.
  
  - Expressions of interest quickly followed, with several members including one stating, *'sounds cool!'* and others confirming their interest.
- **Survey for Meeting Times**: The study group will try to accommodate schedules by collecting preferences through a [Google Form](https://forms.gle/QtQ2C6qzomeHDrC38) detailing available times for meetings.
  
  - Proposed time slots include weekdays in the evenings and weekends, allowing for a range of options for participants.

 

**Link mentioned**: [LLM Agents Peer Study Group (Virtual)](https://forms.gle/QtQ2C6qzomeHDrC38): Use this to express your interest in joining a peer study group virtually. We might use Discord events or Zoom. We will go through the lectures starting from the first and also discuss the additional...

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1300884129214107689) (3 messages):

> - `Class conditioned latent diffusion model`
> - `Pink pixel patches`
> - `DDIM p_sample clipping`
> - `IJEPA and autoregressive architecture`

- **Training Class Conditioned Latent Diffusion Model**: A member reported encountering **pink pixel patches** when decoding from the VAE while training a class conditioned latent diffusion model.
  
  - They noted that these patches become less frequent with more training but still occasionally appear.
- **DDIM p_sample Clipping Considerations**: The same member is contemplating whether to apply more aggressive clipping in **DDIM p_sample**, currently set at the **99.95%** extremes.
  
  - *They are unsure if this adjustment will eliminate the persisting pink patches*.
- **Collaboration on Innovative Architecture**: The member is open to collaboration on training an architecture combining **IJEPA** and `autoregressive image generation without vector quantization`.
  
  - They expressed enthusiasm for joint efforts to explore this unique **architecture**.

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1300541614703513700) (5 messages):

> - `Model Parameters`
> - `Token vs Parameters Confusion`

- **Misunderstanding between parameters and tokens**: A member assumed that the **100B** referenced was in terms of parameters rather than tokens, leading to confusion.
  
  - Another member acknowledged this distinction, saying, *'you're right my bad',* clarifying the initial misunderstanding.
- **Clarification on Model Size**: A member pointed out that the linked model only has **8B parameters**, contrasting with the previous assumption of a larger model.
  
  - The clarification was met with validation from another member, who responded, *'well said'*, indicating agreement with the clarity.

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1300809965425922079) (7 messages):

> - `Negative Line Day Effects`
> - `CI Test Improvements`
> - `Uops Readability Challenges`
> - `Documentation Maintenance Concerns`
> - `Premature Optimization in Code`

- **George Hotz Reports Negative Line Day**: *George Hotz* expressed having a **negative line day**, to which another member congratulated him humorously.
  
  - This exchange indicates the informal and supportive nature of interactions within the community.
- **Faster Continuous Integration Test Observed**: *Chenyuy* noted that there was a **2-minute faster CI test**, suggesting improvement in the testing process.
  
  - Such optimizations highlight the ongoing efforts to enhance performance in the project.
- **Discussion on Uops Readability**: A member raised concerns about **Uops readability**, agreeing that some one-liners can be challenging to understand but offering no clear solutions.
  
  - They suggested creating a documentation page for clarification, which could help improve code comprehensibility.
- **Concerns Over Documentation Longevity**: *Chenyuy* expressed that documentation is less preferred due to its tendency to quickly become outdated, which leads to maintenance difficulties.
  
  - He emphasized that having incorrect documentation may be more detrimental than having none at all, highlighting the pace of changes in tinygrad.
- **Debating Premature Optimization**: *George Hotz* suggested removing certain code elements due to the potential of it being **premature optimization**.
  
  - This reflects a thoughtful approach to ensure code efficiency without unnecessary complexity.

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1300564669869588522) (4 messages):

> - `RAGAS`
> - `CSV files in open source models`
> - `LangChain Python Compatibility`
> - `LangChain-JS Course`

- **Utilizing RAGAS for LLM Evaluations**: A member suggested using [RAGAS](https://github.com/explodinggradients/ragas) to enhance LLM application evaluations, showcasing its GitHub repository.
  
  - The repository promises to help developers supercharge their **LLM evaluations** with various methodologies.
- **CSV Integration with Open Source Models**: A member inquired about integrating **CSV files** as data sources using open source models like **LLAMA3**, highlighting the existing lack of examples or features for this setup.
  
  - They are specifically looking for guidance on using CSVChain and PandasAgent with non-OpenAI models.
- **Compatibility of Python with LangChain 0.3**: A member asked for clarification on which version of **Python** is compatible with **LangChain version 0.3**.
  
  - This reflects the community's ongoing interest in ensuring proper environment setup for LangChain usage.
- **Free LangChain-JS Course Launch Announcement**: **Exciting news!** A new [LangChain-JS course for beginners](https://www.udemy.com/course/genai-langchain-for-javascript-developers/?couponCode=AMIT100) has launched on Udemy, covering everything from basics to building a complete RAG application.
  
  - The first **100 students** can join for free, encouraging swift enrollment.

 

**Link mentioned**: [GitHub - explodinggradients/ragas: Supercharge Your LLM Application Evaluations 🚀](https://github.com/explodinggradients/ragas): Supercharge Your LLM Application Evaluations 🚀. Contribute to explodinggradients/ragas development by creating an account on GitHub.

 

---

### **LangChain AI ▷ #**[**tutorials**](https://discord.com/channels/1038097195422978059/1077843317657706538/1300795224812945429) (2 messages):

> - `Web scraping with LLM`
> - `LangChain-JS course`

- **Learn Web Scraping with LLM**: Check out this informative [YouTube video](https://youtu.be/7kbQnLN2y_I) titled 'This is how I scrape 99% websites via LLM', which teaches web scraping with LLM in 2024.
  
  - The video explains how to use [AgentQL](https://www.agentql.com/) to scrape websites for free, emphasizing practical applications.
- **New LangChain-JS Course is Live!**: An exciting new LangChain-JS course for beginners has launched on [Udemy](https://www.udemy.com/course/genai-langchain-for-javascript-developers/?couponCode=AMIT100)! It covers everything from LangChain basics to building a complete RAG application.
  
  - The first **100 students** can join for free, so don't miss out on this opportunity to enhance your skills!

 

**Link mentioned**: [This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I): How to do web scraping with LLM in 2024Use AgentQL to scrape website for free: [https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason](https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason)_...

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1300560060182954134) (5 messages):

> - `Leaderboard Terminology`
> - `Multi-Step vs Multi-Turn Evaluation`

- **Understanding 'Multiple' on Leaderboard**: 'Multiple' on the leaderboard refers to the **ability to choose the correct function** from several options in a single turn setting, as clarified by a user referencing a [GitHub example](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438). However, it remains unclear how multi-step is evaluated in this context.
- **Multi-Step vs Multi-Turn Clarification**: A member noted that while 'multiple' pertains to functions, the **multi-step evaluations** fall under a different category, described as 'multi_turn'. There is currently no standalone 'multi-step' evaluation in use.
- **Mixing Multi-Step with Multi-Turn Categories**: It was mentioned that in the **'multi_turn' categories**, both multi-step and multi-turn evaluations coexist, with each turn potentially involving multiple steps. This overlap may lead to confusion as there is no dedicated multi-step category.

**Links mentioned**:

- [GitHub - ShishirPatil/gorilla at 2101b11f6d03d9f323715d7d2012a955d7f4114e](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d): Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - GitHub - ShishirPatil/gorilla at 2101b11f6d03d9f323715d7d2012a955d7f4114e
- [gorilla/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json at 2101b11f6d03d9f323715d7d2012a955d7f4114e · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438)): Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1300875366436442213) (2 messages):

> - `Cracked Engineers Job Platform`
> - `Tech Job Newsletter`
> - `AI Startups Hiring`

- **Cracked Engineers Job Platform Launches!**: A member shared an exciting new [job platform for technical roles](https://www.crackedengineers.com/) called **Cracked Engineers**, aiming to be the go-to for top AI/tech startups.
  
  - With a projected **$1000 MRR** before the official launch, the platform is already attracting top companies like **Weaviate**, **UnslothAI**, and **JuliusAI**.
- **Insightful Weekly Tech Jobs Newsletter Introduced**: The platform is set to release a **weekly tech jobs newsletter** that will curate positions based on user preferences, starting soon.
  
  - Users can subscribe to tags that interest them, such as **CUDA**, **MLOps**, or **Software Engineering** through their dashboard.
- **Exciting Job Opportunities at AI Startups**: A member highlighted that **Unsloth AI**, **Julius AI**, and **Jimini AI** are actively hiring for excellent positions that they would consider if they weren't a founder.
  
  - These positions are described as **amazing opportunities** for anyone looking to work with cutting-edge AI technology.

**Links mentioned**:

- [Tweet from Aleksa Gordić 🍿🤖 (@gordic_aleksa)](https://x.com/gordic_aleksa/status/1851247076987855063): [🚀] Super excited to share this: I've built a job platform for technical roles called "Cracked Engineers". :) If you want to land a job with some of the world's best AI/tech startups ...
- [Tweet from undefined](https://x.com/gor): no description found

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1300549775254356138) (1 messages):

> - `SymNoise methodology`
> - `LLaMA-2-7B performance`
> - `Fine-tuning techniques`

- **Inquiry on SymNoise Implementation**: A member is seeking a code implementation for the paper discussing the **SymNoise** fine-tuning technique for language models, which incorporates **symmetric noise** into the embedding process.
  
  - *However, they noted challenges in implementation, specifically regarding the doubling of* ***batch size*** *through concatenation.*
- **LLaMA-2-7B Scores with SymNoise**: The **SymNoise** method reportedly enhances the **LLaMA-2-7B** model's performance on AlpacaEval from **29.79%** to **69.04%**, outperforming the previous method, **NEFTune**.
  
  - *This represents a* ***6.7%*** *improvement over NEFTune's score of* ***64.69%*** *according to the paper's abstract.*
- **SymNoise vs. NEFTune**: In tests across various models and stronger baseline instruction datasets, **SymNoise consistently outperforms NEFTune**.
  
  - *Discussion underscored the need for deeper research in this area, as highlighted in the paper.*
- **Request for Research Links**: The inquiry included a link to the paper on **arXiv** for further reading, highlighting the importance of the study's findings.
  
  - *No additional implementations or links were provided by other members to address the code inquiry.*

 

**Link mentioned**: [SymNoise: Advancing Language Model Fine-tuning with Symmetric Noise](https://arxiv.org/abs/2312.01523): In this paper, we introduce a novel fine-tuning technique for language models, which involves incorporating symmetric noise into the embedding process. This method aims to enhance the model's func...

 

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