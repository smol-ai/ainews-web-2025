---
id: 32c6f94e-a602-4966-a326-9f2dade3994b
title: Contextual Position Encoding (CoPE)
date: '2024-05-31T03:11:48.061328Z'
original_slug: ainews-contextual-position-encoding-cope
description: >-
  A new positional encoding method called CoPE for transformers, developed by
  Jason Weston of Meta AI, improves counting and copy tasks, language modeling,
  and coding by considering context and external memory. The year sees a variety
  of position encoding variants, emphasizing adding capabilities to models.
companies:
  - meta-ai
models:
  - cope
topics:
  - transformers
  - positional-encoding
  - language-modeling
  - coding
  - artificial-intelligence-research
---


<!-- buttondown-editor-mode: plaintext -->**Just one more RoPE variant bro just one more**

> AI News for 5/29/2024-5/30/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**391** channels, and **4383** messages) for you. 
Estimated reading time saved (at 200wpm): **478 minutes**.

A quiet day, but the CoPE paper got some buzz: so we're talking about it. 

Traditional LLMs have [known issues with simple algorithmic tasks like counting and copying](https://x.com/pfau/status/1796273583603302823). This is likely an artefact of their positional encoding strategy.

Jason Weston of Meta AI released his [paper](https://arxiv.org/abs/2405.18719) on [CoPE](https://x.com/jaseweston/status/1795978611784089799), a new positional encoding method for transformers that takes into account *context*,  creating "gates" with learnable indices.

 ![image.png](https://assets.buttondown.email/images/ad0fb3fc-e851-46c7-8f03-6d9ae4f24043.png?w=960&fit=max) 

Using this, a CoPE LLM can:

- "count" distances per head dependent on need, e.g. i-th sentence or paragraph, words, verbs, etc. Not just tokens.
- solve [counting & copy tasks](https://x.com/jaseweston/status/1795978614132920656/photo/1) that standard transformers cannot.  ![image.png](https://assets.buttondown.email/images/d765bdf5-9ceb-4d1f-a7be-6eb4e2c214c9.png?w=960&fit=max) 
- Better PPL on language modeling + coding tasks.
 ![image.png](https://assets.buttondown.email/images/7c43c0a4-767d-4ea0-a687-f1ad52252cb0.png?w=960&fit=max) 

**You could even modify this concept to use [external memory](https://x.com/krishnanrohit/status/1796061792201814466), not merely local context, to calculate the gates.**

As [Lucas Beyer notes](https://x.com/giffmana/status/1796077219455869414), the raft of position encoding variants this year is perhaps a richer source of research because "Linear attention was about removing capacity from the model, which didn’t make sense long term. Position embedding is about **adding missing capabilities to the model**, which makes a lot more sense."

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

**New AI Models and Benchmarks**

- **Contextual Position Encoding (CoPE)**: [@jaseweston](https://twitter.com/jaseweston/status/1795978611784089799) introduced CoPE, a new positional encoding method for transformers that takes context into account, enabling them to solve counting & copy tasks and improving performance on language modeling and coding.
- **SEAL Leaderboards**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1795857651592491281) launched SEAL Leaderboards for private, expert evaluations of frontier models that are unexploitable and continuously updated.
- **Gemini 1.5 models**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1796216673348833445) released Gemini 1.5 Flash and Pro models on their API, with Flash designed for fast, efficient inference at 1000 requests per minute.
- **Claude with tool use**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1796210547077128578) announced general availability of tool use for Claude, enabling intelligent selection and orchestration of tools for complex tasks.

**Advancements in AI Applications and Platforms**


- **ChatGPT Free upgrades**: [@gdb](https://twitter.com/gdb/status/1795970586050429005) noted ChatGPT Free tier is providing widespread access to cutting-edge AI features.
- **Claude Tool Use GA**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1796210547077128578) made tool use generally available for Claude, allowing it to intelligently select and orchestrate tools to solve complex tasks end-to-end.
- **GPT3 Birthday**: [@karpathy](https://twitter.com/karpathy/status/1795980744436932871) reflected on the 4th anniversary of GPT-3 and how it showed that models would improve on practical tasks just by training bigger ones, making better algorithms a bonus rather than a necessity for AGI progress. He noted if given a 10X bigger computer now, he would know exactly what to do with it.
- **Perplexity Pages**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1796203494401040846) launched Perplexity Pages, allowing users to turn research into visually appealing articles with formatted images and sections. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1796220011448786949) described Perplexity's mission to cater to the world's curiosity with Pages as "AI Wikipedia", allowing the effort of analyzing sources and synthesizing a readable page with a simple "one-click convert".
- **Milvus Lite**: [@LangChainAI](https://twitter.com/LangChainAI/status/1796206411288039430) partnered with Milvus to simplify creating powerful GenAI apps by combining their capabilities.
- **Property Graph Index**: [@llama_index](https://twitter.com/llama_index/status/1795869279457546447) launched the Property Graph Index, providing a high-level API for constructing and querying knowledge graphs using LLMs.
- **Repetitions in LangSmith**: [@LangChainAI](https://twitter.com/LangChainAI/status/1796222825898074235) added support for running multiple repetitions of experiments in LangSmith to smooth out noise from variability.




---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!


Technology Developments and Partnerships

- **OpenAI partnerships**: OpenAI [announced partnerships](https://www.reddit.com/r/OpenAI/comments/1d3hf0b/vox_media_and_the_atlantic_sign_content_deals/) with The Atlantic, Vox Media, and WAN-IFRA to help news publishers explore AI integration. They also appear to have [closed a deal with Apple](https://www.reddit.com/r/OpenAI/comments/1d3726q/what_do_you_actually_use_ai_for_on_a_regular_basis/). Discussions in /r/OpenAI centered around [how people are using ChatGPT daily](https://www.reddit.com/r/OpenAI/comments/1d3gczj/anyone_else_talk_to_chatgpt_all_day/).

- **Google Gemini models**: Google [doubled the output price](https://www.reddit.com/gallery/1d3h2ev) of Gemini 1.5 Flash. Their updated Gemini 1.5 0514 models are [rated well on Chatbot Arena leaderboard](https://www.reddit.com/gallery/1d36gcn) considering the API costs. 

- **Mistral AI's Codestral**: Mistral AI debuted [Codestral, a 22B open-weight code model](https://mistral.ai/news/codestral/) licensed under the Mistral AI Non-Production License. The Verge [covered the Codestral launch](https://www.theverge.com/2024/5/29/24166334/mistral-debuts-a-coding-assistant-called-codestral).

- **Groq speeds up Llama 3**: Groq [announced Llama 3 running at 1200+ tokens per second](https://x.com/GroqInc/status/1795919195076784340) on their systems.

Model Benchmarks and Evaluations

- **AutoCoder beats GPT-4**: AutoCoder, a new code generation model, [surpassed GPT-4 Turbo and GPT-4o](https://www.reddit.com/r/LocalLLaMA/comments/1d3qx5q/autocoder_a_new_model_designed_for_the_code/) in pass@1 on the HumanEval benchmark. It also offers a more versatile code interpreter.

- **Scale AI's SEAL Leaderboards**: Scale AI introduced [SEAL Leaderboards with private datasets and paid annotators](https://www.reddit.com/r/LocalLLaMA/comments/1d3idzn/scale_ai_are_introducing_high_quality_arenas_with/) for fairer, higher quality expert evaluations of frontier models. An [infographic explains the SEAL approach](https://www.reddit.com/gallery/1d3n6ag).

- **GPT-4 bar exam claims challenged**: An [MIT study found GPT-4 did not really score 90th percentile](https://link.springer.com/article/10.1007/s10506-024-09396-9) on the bar exam as previously claimed.

- **TimeGPT-1 tops time series benchmarks**: TimeGPT-1 [ranked first in accuracy and speed vs other foundation time series models](https://www.reddit.com/r/MachineLearning/comments/1d3h5fs/d_benchmarking_foundation_models_for_time_series/) like TimesFM, Chronos, Moirai and Lag-Llama in a 30,000+ time series benchmark.


**AI Hardware & Performance**

- **AI training compute scaling 4-5x per year**: The [amount of compute used in AI training is scaling up 4-5x per year](https://www.reddit.com/r/singularity/comments/1d3xfhs/the_amount_of_compute_used_in_training_is/), highlighting rapid progress. ([1](https://i.redd.it/5fnoh21ubi3d1.jpeg))
- **Groq updates LLama 3 performance to 1200+ tokens/sec**: [Groq updates LLama 3 performance to 1200+ tokens per second](https://x.com/GroqInc/status/1795919195076784340) on their hardware.
- **Qualcomm releases Snapdragon X Plus/Elite benchmarks**: Qualcomm releases [Snapdragon X Plus and X Elite benchmarks showing 45 TOPS performance](https://www.notebookcheck.net/Qualcomm-releases-official-Snapdragon-X-Plus-and-Snapdragon-X-Elite-benchmarks-for-45-TOPS-Hexagon-NPU.841811.0.html) for the Hexagon NPU, enabling efficient on-device AI.
- **Sambanova sets speed record of 1000 tokens/sec on Llama 3**: [Sambanova system reaches 1000 tokens per second on Llama 3 8B](https://venturebeat.com/ai/sambanova-breaks-llama-3-speed-record-with-1000-tokens-per-second/), setting a new speed record.


---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. New AI Model Releases and Benchmarks**: 

- The **[Yuan2.0-M32 model](https://x.com/osanseviero/status/1796082193044844590)** with 40B parameters outperformed **Llama 3 70B** on Math/ARC tasks using only 3.7B active parameters during generation. 
- **Codestral Model Release and Integration**: **Mistral AI** released **Codestral-22B-v0.1**, a code-generating model supporting 80+ programming languages. It excels at code instruction and Fill in the Middle (FIM) tasks, with [more details in their blog post](https://mistral.ai/news/codestral/). **LlamaIndex** provides [day 0 support and a tutorial notebook](https://t.co/YxeyHhSjKU) for Codestral, and it's also compatible with **Ollama** for local execution with [direct LlamaIndex support](https://t.co/gsPHHF4c0K).
- **[K2](https://huggingface.co/LLM360/K2)**, a fully open-source model, outperformed Llama 2 70B using 35% less compute, showcasing efficient AI engineering.

**2. Optimizations and Advancements in AI Systems**:

- **Whisper Model Optimization Yields 6.3x Speedup**: A community member successfully optimized the **Whisper model** using techniques like **static cache**, **HQQ quantization**, **torchao 4-bit kernel**, and **torch.compile with fullgraph**. This combination resulted in a substantial 6.3x speed increase. A [detailed blog post](https://github.com/karpathy/llm.c/pull/475) is forthcoming to share insights from this optimization process.
- Discussions covered **templating block sizes** like `blockDim.x` for **significant CUDA kernel performance boosts**, especially in fused classifiers.
- **Cloudflare R2** was suggested to replace Python dependencies and **internal S3** for sharing large datasets, optimizing costs and avoiding ancillary fees.

**3. AI Model Fine-tuning and Customization**:

- Members explored **ideal strategies** for **fine-tuning LLMs** like Llava for **image and video understanding tasks**, debating the merits of **Direct Preference Optimization (DPO)** over **Supervised Fine-Tuning (SFT)**.
- **Anti-prompts** were discussed as a technique to guide conversation flow by halting generation at predefined words, allowing user interjection before resuming model output.
- Advice was shared on **fine-tuning Llama3 base models** and using **DPO** over instruction models for creating bespoke roles like historical figures or characters.

**4. Competitions and Open Initiatives**:

- A **Model Merging competition at NeurIPS** was announced, offering an **[$8K prize pool](https://x.com/LChoshen/status/1796256513519989102)** to revolutionize model selection and merging techniques for creating optimal LLMs.
- **[LAION](https://laion.ai/notes/open-gpt-4-o/)** called for community contributions to build an **open GPT-4-Omni** model with large-scale multi-modal capabilities, providing datasets, tutorials, and guidance.
- The **[Le Tigre project](https://devpost.com/software/le-tigre)**, a multimodal variant based on Mistral 7B inspired by GPT-4-V's architecture, was showcased from a recent hackathon.

---

# PART 1: High level Discord summaries

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity Pages Pioneers Prettier Posts**: **Perplexity AI** has unveiled **Perplexity Pages**, a tool for transforming research into curated articles, creating an **AI Wikipedia**. The feature is currently available to Pro users, with expectations to open to more users, and elaborated upon in their [blog post](https://pplx.ai/pages).

**Grok Woes Lead to Search Superiority Strive**: Community member sneakyf1shy strives to build an improved model over Grok, aiming to enhance the search functionality within Perplexity's web application. The community also debated the efficacies of existing models, APIs, and indexed data, citing limitations and envisioning enhancements.

**Pages Feedback: The Good, the Bad, the Ugly**: Users experimenting with **Perplexity Pages** shared mixed feedback; some praised its utility while others faced issues, such as missing content sections. The community's pulse ranged from skepticism about Perplexity's indexing to excitement about the feature, with a [how-to guide](https://www.perplexity.ai/page/How-to-Use-FvLfzZ_ATyqE2n_tAGKk7A) circulating for those interested.

**API Angst and Google vs. OpenAI Grudge Match**: Technical discussions delved into the challenges of user-friendly API scalability and multi-step reasoning improvements. Meanwhile, the Google-OpenAI rivalry captured attention, sparking debate over their strategic AI moves with speculation around AGI progress and market influence.

**AI Ethics and Physics Explored by the Curious**: The **sharing** channel highlighted member contributions on the ethical and physical dimensions of perplexing topics. Links to discussions on **consciousness**, **LLM functionalities**, and a supposed **pro/con analysis** indicate a community engaged in substantive and diverse AI-related themes.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Google's Gemini Gaffe**: Confusion arose over the inconsistency in Google's Vertex AI pricing, with concerns about billing per character on Vertex AI versus per token on Google's direct AI services, which led to a [discussion thread](https://ai.google.dev).

- **LLM Fine-Tuning Finesse**: The community shared knowledge and experiences in fine-tuning Large Language Models (LLMs), specifically focusing on deterministic tool calling in multi-agent systems and the successful use of state machine logic from repositories like [robocorp/llmstatemachine](https://github.com/robocorp/llmstatemachine). Another focal point was the improvement of fine-tuning LLMs with custom data using the GGUF format, backed by an active Hugging Face Pull Request providing easier conversion from HF to GGUF ([source](https://github.com/huggingface/transformers/pull/30928)).

- **Embracing Modal's Multifaceted Mechanisms**: Debates and troubleshooting of Modal task executions were rampant, highlighting issues like dataset paths and config settings. The community responded with insights on WANDB integration, sharing config files, and directing users to [Modal's documentation](https://modal.com/docs/guide/trigger-deployed-functions) for further learning.

- **Expanded Learning Through Papers and Bundles**: An array of learning resources surfaced, including a Meta paper on vLLM, a collection of LLM resources on [GitHub](https://github.com/marco-jeffrey/awesome-llm-resources), and details of an AI-coding Humble Bundle. Additionally, a paper on expanding LLama3's context window piqued interest ([source](https://arxiv.org/abs/2404.19553)).

- **Global Gatherings and Events**: There's a buzz around upcoming AI events such as the **Global AI Hackathon** from June 7 to 9 in Singapore, Sydney, and San Francisco, which is backed by top AI builders, aiming to explore "AI for a better  world" – interested attendees can RSVP via [this link](https://lu.ma/igqisb0e). Meanwhile, on Discord, members across the U.S. coasts and European regions voiced enthusiasm for local meetups and shared venues.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Freebies for ChatGPT Users**: ChatGPT Free users received a substantial upgrade with access to browsing, vision, data analysis, file uploads, and GPTs, which opens up new avenues for experimentation and development.
- **OpenAI Empowers Nonprofits**: OpenAI launched **OpenAI for Nonprofits**, offering greater accessibility to their tools for charitable organizations, marking a strategic move to support social good through advanced AI applications. Further details were discussed, including strategies to counteract deceptive AI uses.
- **GPT-4 Availability and Performance Discourse**: The community engaged in lively discussions around **GPT-4's availability and performance**, noting that free users might experience automatic model switching and raised concerns about "word salad" issues with longer GPT-4 outputs. Members also touched upon the customizability and potential memory enhancements for GPT models.
- **Coding Assistance and API Best Practices**: AI engineers compared coding assistance tools like **GPT-4o, Mistral’s codestral, and Copilot**, emphasizing speed and accuracy. They also shared knowledge on protecting API keys with proxy backend servers and the importance of considering API stability for extended sessions over browser-based interactions.
- **Bias and Troubleshooting in AI Tools**: Engineers humorously acknowledged personal bias when evaluating their own AI tools and also exchanged tips for troubleshooting issues, suggesting splitting requests for versions lower than 4 to maintain compatibility.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **PDFs Befriend AI with Everything-AI Integration**: The [Everything-AI project](https://github.com/AstraBert/everything-ai) now boasts integration with `llama.cpp` and `Qdrant`, allowing for interactive exchanges with PDFs, as community contributions enhance HuggingFace's repository of tools and models.

- **The Competitive Edge of Yuan 2.0-M32**: The freshly minted **Yuan2.0-M32 model**, with its 40 billion parameters and innovative architecture, overshadows Llama 3 70B in Math/ARC tasks, revealed on [Twitter](https://x.com/osanseviero/status/1796082193044844590) and showcased on [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf), with a link to the supporting [research paper](https://hf.co/papers/2405.17976).

- **Visualization Becomes Accessible with Nvidia Embed V1**: A user shared their [Nvidia Embed V1](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1) Space for showcasing Nvidia's embedding model, and invites enhancements through PRs for refined functionalities or exciting new examples.

- **Hugging Face and DuckDB Unite for Smoother Dataset Handling**: The fusion of DuckDB and Hugging Face datasets, facilitated by an `hf://` path, simplifies data integration processes, as detailed in the [tutorial blog post](https://blog.getwren.ai/how-to-load-huggingface-datasets-into-duckdb-and-query-with-gpt-4o-c22db89519e4d), marking a stride in data manipulation convenience.

- **AI Community Geared Up for NeurIPS Model Merging Contest**: A competition announced for NeurIPS focused on model merging piques interest within the AI community, promising an $8000 reward and the chance to push the boundaries of model selection techniques, as cited in an [official tweet](https://x.com/LChoshen/status/1796256513519989102).

- **Whisper Model Gets Fine-Tuned with Timestamps**: A discussion around extracting word-level timestamps with the Whisper model highlights the method's documentation and credits the work to research like *“Robust Speech Recognition via Large-Scale Weak Supervision”*, indicating enhancements in audio processing and its applications.

- **Open-Source Models Usher in K2's Potential**: Two new fully open-source models, including [K2](https://huggingface.co/LLM360/K2), are celebrated for their prowess, with K2 especially noted for its stellar performance compared to Llama 2 70B model with a 35% compute reduction, spotlighting the strides made in efficient AI model engineering.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Codestral Joins the Coding Model Fray**: **Mistral** introduced **Codestral-22B-v0.1**, capable of dealing with over 80 programming languages, demonstrating impressive performance in tasks like code instruction and Fill in the Middle (FIM). For those interested in testing the model, [download and explore Codestral-22B here](https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF).

**The Never-Ending Context Length Challenge**: Engineers highlighted the limitations of models like the **llama series**, capped at **4096** tokens, and noted RoPE extension allowing a maximum of **16k** tokens, with spirited banter about the importance of context size.

**Hardware Discussions Heat Up**: The **RTX 5090** stirred speculation with its purported **448-bit bus** and **28 GB GDDR7** memory. Meanwhile, pragmatic comparisons of CPU inference and the pros and cons of GPU setups, such as using multiple **3090** cards, dominated the discussion.

**Whisper & Amuse in Spotlight**: A technical hiccup was observed with the **Whisper models** not being compatible with **llama.cpp**, as well as a broken GitHub link for **Amuse**. Solutions included utilizing **whisper.cpp** and accessing **Amuse** through an available [Hugging Face link](https://huggingface.co/Stackyard-AI/Amuse/blob/main/Amuse_v1.3.0.zip).

**Practical Tips in Adding Inference GPUs**: One discussion clarified the reality of adding additional GPUs for inference in **LM Studio**, stressing the need for appropriate space, power, and correct settings management, proving that juggling hardware is as much art as it is science.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama3 Trumps Phi3 in AI Showdown**: Engineers concurred that **Llama3** is superior to Phi3 in testing, with comments praising its performance and criticizing Phi3 for being "*extremely synthetic*." Users advised against using Phi3 models, highlighting the effectiveness of the base Llama3 instead.

- **Refining Role-Playing AI**: It was suggested to start with training Llama3 base models, followed by finetuning for instruction following to create bespoke role-playing characters. However, simply prompting Llama3 with instructions to "*Pretend you are [X]*" may yield better results than a standard fine-tuning process.

- **Anti-Prompts for Controlled Conversations**: The utility of anti-prompts was debated, revealing a strategy to guide chat models' conversation flow by halting generation at predefined words. This technique enables users to interject before letting the model resume its output.

- **Model Training and Fine-tuning Pitfalls**: Discussion pointed out that fine-tuning on top of instruction models is generally discouraged due to potential value loss. Using Direct Preference Optimization (DPO) over base models can tailor outputs for specific character roles more effectively.

- **Emerging Models and Tech Wrinkles**: Enthusiasm was shared for new models like Yuan, with a cautionary note on the importance of practical application over benchmark results. One user faced an issue with an Apple M3 Pro GPU being incompatible with CUDA, which led to advice on utilizing services like Google Colab for model training and fine-tuning.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Economizing AI Training**: Members highlighted ways to **train Stable Diffusion** models cost-effectively, with tools like **[Google Colab](https://github.com/hollowstrawberry/kohya-colab)** and services such as **RunDiffusion** being discussed for their budget-friendly solutions.

- **Optimizing Image Accuracy**: Techniques to enhance **image generation** were discussed, with a particular focus on using **ControlNet** and advanced samplers. For dynamic LoRA control, the community shared the **[sd-webui-loractl](https://github.com/cheald/sd-webui-loractl)** GitHub repository.

- **Ruby Joins the AI API Fray**: A new **open-source Ruby SDK** for Stability AI's API was introduced, aimed at streamlining image generation tasks with core and SD3 models. The SDK can be found and contributed to on **[GitHub](https://github.com/OlympiaAI/stability)**.

- **Anticipation and Anxieties Over SD3**: The community exchanged thoughts about **Stable Diffusion 3's potential release**, voicing concerns over licensing issues and comparing financial support with that of competitors like **Midjourney**.

- **Kid-Friendly AI**: A discussion was initiated on how to safely **introduce Stable Diffusion to children**, with the focus on utilizing **ControlNet** to responsibly transform children's sketches into polished images.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI and the Future of Learning**: A burgeoning discussion centered on [GPT-4-OMNI](https://laion.ai/notes/open-gpt-4-o/)'s utility as an educational assistant, with the community excited about its multi-modal capabilities signaling a step-change in personalized learning experiences.

- **Contamination Alert in Model Updates**: Alarm bells in the form of a 29% spike in contamination were rung for the **Luxia 21.4b** model between versions v1.0 to v1.2, as evidenced by results on the [GSM8k tests on HuggingFace](https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1), though this issue didn’t plague other testing benchmarks.

- **Position Encoding Gets Contextual**: Introducing **Contextual Position Encoding (CoPE)**, a fresh take on traditional positional encoding, was part of an active dialogue, underscoring improvements in language modeling and coding tasks, as highlighted by a [tweet from Jason Weston](https://x.com/jaseweston/status/1795978611784089799?s=61&t=ryK3X96D_TkGJtvu2rm0uw).

- **The Heavyweights: MLPs vs. Transformers**: The community gave airtime to a critical take on **MLP-Mixer's constraints** regarding causality and sequence lengths, provoking a deeper look into MLPs as static versus transformers' ability for dynamic context-dependent weights.

- **Decoding Model Performance**: Contributions involved sharing an [Arxiv paper on learning rates and weight averaging](https://arxiv.org/abs/2405.18392), debating gradient diversity’s role in mini-batch SGD performance, and announcing a NeurIPS competition with up to $8,000 in rewards focused on model merging, as tweeted by [Leshem Choshen](https://x.com/LChoshen/status/1796256513519989102) and hosted on [the official competition page](https://llm-merging.github.io/).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Conundrum Solved**: Engineers discovered a **[bug in Triton code](https://github.com/triton-lang/triton/issues/2302)** causing int32 multiplication overflow, revealing how production-scale data can expose limitations not evident in unit tests, such as 16-bit grid dimension limits in CUDA.
  
- **Performance Tuning Revealed**: It's been suggested that templating block sizes like `blockDim.x` could notably boost performance in CUDA kernels, and discussions include propositions to merge branches for **layernorm recomputations** in favor of optimizing functional improvements before re-tweaking to minimize redundancies.

- **Whisper Model Just Got a Super Update**: A member successfully optimized the **Whisper model** by leveraging **static cache**, **HQQ quantization**, **torchao 4-bit kernel**, and **torch.compile with fullgraph**, achieving a 6.3x speed up, promising a [detailed blog post](https://github.com/karpathy/llm.c/pull/475).

- **Intricacies of Low-Precision Multipliers Illustrated**: Queries ranged from specifying precise operations of **fp4 multiplication** to exploring **mixed precision layers** in activations and gradients. There was mention of a CUDA kernel for **FP6-LLM** demonstrating a mixed-input multiply for fp16 activations with MX fp6_e3m2 weights, where calculations are performed using tensor cores.

- **Resourceful Workarounds with Cloudflare R2 and Internal S3**: Engineers discussed using **Cloudflare R2** to reduce egress fees and Python dependencies, while considering internal S3 storage with pre-uploaded resources to share large datasets without incurring additional costs. This aligns with the discussion on **[installation errors and compatibility](https://github.com/pytorch/ao/pull/296)**, including tips for handling builds requiring CUDA capability enhancements and avoiding isolated environment issues.

  
These targeted discussions reflect the community's focus on achieving performance improvements, optimizing cost efficiency, and tackling practical issues faced in implementing machine learning models at scale.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Codestral Emerges with Multi-Language Support**: **Mistral AI** has introduced **Codestral**, a new local code-generating model supporting over 80 programming languages, with day 0 integration via LlamaIndex including a tutorial [notebook](https://t.co/YxeyHhSjKU). It's also compatible with **Ollama**, boosting local execution with direct [support](https://t.co/gsPHHF4c0K).

- **Crafting Knowledge Graphs Locally**: Engineers discussed local construction of knowledge graphs combining models like **Ollama** with **Neo4j** databases, backed by a comprehensive [guide](https://t.co/5ee6LwM7RE) and additional [how-to details](https://t.co/xhoIEi9egq).

- **NLP Meetup Set for Financial Insight**: London will host an NLP meetup featuring **LlamaIndex**, **Weaviate_io**, and **Weights & Biases** with a focus on using LLMs in financial services, with discussions on vector database management and a [sign-up](https://t.co/vli6DY8Xg7).

- **LlamaParse Expands Format Abilities**: **LlamaParse** has improved its functionality to process spreadsheets such as Excel and Numbers, facilitating their usage within RAG pipelines; learn more in the provided [notebook](https://t.co/60MvR0h5DC) and through a [demo](https://t.co/IfF4UUqB0C).

- **Navigating the Landscape of API Frameworks and Data Stores**: The community exchanged insights on selecting **API frameworks**, with a nod to **FastAPI** for asynchronous capabilities, and discussed transitioning data stores from **SimpleStore** to **RedisStore** with strategies including the `IngestionPipeline`. Links to relevant documentations and examples were shared, including a [Google Colab](https://colab.research.google.com/drive/1hiDkBbAJcO3RDrS7CD2ZeQEHGqNv07pq?usp=sharing) and several [LlamaIndex resources](https://docs.llamaindex.ai/en/latest/examples/vector_stores/pinecone_auto_retriever/).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Le Tigre Roars into Multimodal Space**: Engineers have been discussing "Le Tigre," a multimodal project based on the Mistral 7B model, influenced by GPT-4-V's architecture, showcased on [Devpost](https://devpost.com/software/le-tigre) and [GitHub](https://github.com/HugoLB0/Le-Tigre). Anticipation is brewing for the LAION 5B dataset but its release remains uncertain.

- **Sonic Speaks Volumes**: Cartesia AI unveiled Sonic, a *state-of-the-art* generative voice model lauded for its lifelike quality and remarkable 135ms latency; details can be explored through their [blog](https://cartesia.ai/blog/sonic) and [Twitter](https://twitter.com/cartesia_ai/status/1795856778456084596) announcement.

- **The Merger of Models**: The NeurIPS Model Merging Competition ignited discussion with an $8,000 prize pool, aiming to advance techniques in model merging, whilst issues on FFT replacing self-attention in transformers sparked intellectual curiosity, inspired by a paper suggesting the method could achieve near-BERT levels of accuracy with lower computational demands - [paper](https://arxiv.org/pdf/2105.03824).

- **Cartoons Get Crafty with ToonCrafter**: Skepticism met curiosity over ToonCrafter, a project designed for sketch-guided animation, with engineers noting its potential to disrupt traditional anime production costs which could shift from hundreds of thousands down to lower figures.

- **GPT-4-Omni Open Call**: LAION's call for contributions to an open GPT-4-Omni project was a notable announcement, aiming to foster collaborative development of large-scale multi-modal capabilities, as detailed in their [blog post](https://laion.ai/notes/open-gpt-4-o/).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Teaching AI with Timely Prompts**: Discussions highlighted the boost in model performance by including context-specific prompts and responses, with a focus on [in-context learning](https://twitter.com/VictorTaelin/status/1776677635491344744) tactics using windows of 100k context or less; this can streamline efficient data processing, where state-saving models like **RWKV** may offer time-saving advantages.

- **Beyond Backpropagation and Merging Models**: Novel training approaches that forgo traditional backpropagation attracted attention, hinting at potential complexity and transformative implications for model efficiency. A NeurIPS **model merging competition** has been announced, dangling an $8K prize pool; further details are accessible via a [specific tweet](https://x.com/LChoshen/status/1796256513519989102).

- **Scaling Down to Outperform Giants**: The recently unveiled **Yuan2-M32** model, boasting 40B parameters with only 3.7B active during generation, rivaled **Llama 3 70B** in benchmarks with lower resource use, fueling a community call to [fine-tune](https://github.com/IEIT-Yuan/Yuan2.0-M32) and harness its capabilities.

- **Navigating the Age of Specialized AI Tools**: The growing trend involves groups preferring Large Language Models (LLMs) with generalized capabilities over niche ones; community members excitedly shared innovations like a [rust library](https://x.com/LChoshen/status/1796256513519989102) for LLM applications and **MoRA**, a tool for high-rank updating during fine-tuning, available on [GitHub](https://github.com/kongds/MoRA).

- **Unlocking Access to RAG Datasets**: A new **RAG dataset** is up for grabs on [Hugging Face](https://huggingface.co/datasets/glaiveai/RAG_v1), subject to users agreeing to share their contact details, amidst discussions on measurement metrics for relevance, like **MRR** and **NDCG**, critiqued based on insights from [Hamel et al](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Swift Embraces ABI Stability**: In discussions about ABI stability, it was noted that **Swift maintains ABI stability** for Apple's operating systems, while **Rust** deliberately avoids it. Maintaining ABI stability can restrict the potential for performance improvements in some programming languages.

- **Skepticism Over Mojo's Potential**: The idea of Mojo becoming a widely adopted low-level protocol was met with skepticism, citing deficiencies such as the absence of certain key types and the difficulty of displacing established languages like **C**.

- **Mojo Eyes Better C++ Interoperability**: The **Modular** community highlighted the importance of **C++ interoperability** for Mojo's success, with possible future support for generating C++ headers from Mojo code being discussed.

- **Package Management and Windows Support for Mojo**: There is ongoing development for a Mojo package manager, as evidenced by [GitHub discussions](https://github.com/modularml/mojo/discussions/413) and [proposal threads](https://github.com/modularml/mojo/discussions/1785). However, frustration was voiced over Mojo's unavailability on **Windows**.

- **Evening Out the Nightlies**: A significant Mojo nightly build `2024.5.3005` has been released with substantial changes, such as the removal of the `Stringable` constructor and several `math` functions from `String`. Furthermore, approximately 25% of Mojo installs come from nightly builds to maintain a simple experience for newcomers. Trouble caused by these changes were addressed, such as correcting `String` conversion to `str` and fixes in CI [PR #2883](https://github.com/modularml/mojo/pull/2883).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **MixMyAI: The New Kid on the Block**: The launch of [mixmyai.com](https://mixmyai.com), a platform presenting itself as a comprehensive AI toolbox with attractive features like no monthly fees and privacy-centric operations, caught the attention of the community.

- **The Secret to Free Tiers Left Uncovered**: Discussions around accessing a free tier for an unspecified service piqued interest, yet the method to obtain such an elusive perk remains a topic of mystery with no clear resolution in sight.

- **Talent for Hire**: A senior developer with skills spanning full stack, blockchain, and AI announced their availability for new opportunities, indicating the community is a hotbed for potential recruitment and collaboration.

- **Model Behavior: Moderated vs. Self-Moderated**: Clarification on models emerged, drawing a line between models that self-moderate and those using an external moderator; specifically pointing out the unique setups for models like Claude on OpenRouter.

- **Programming Packaged**: The creation and announcement of integration packages for OpenRouter with Laravel and Ruby—including [laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter) and [open_router](https://github.com/OlympiaAI/open_router)—demonstrates active community contributions and cross-language support.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Domain-Specific Web Search Through API**: A user described how to set a web search connector for a specific domain using the **API options object**; follow-up discussions on multi-domain restrictions are ongoing.
- **AI for Academic Ingenuity**: An individual is developing a Retrieval-Augmented Generation (RAG) model to enhance their college's search capabilities, detailing an intent to include both **.edu domains** and external review sites like **RateMyProfessors**.
- **Type-Switching Embedding Tactics**: Conversion of **uint8 embeddings to float** for mathematical operations was brought up, with the user being redirected to a more specialized technical channel for in-depth assistance.
- **Startup Seeks User Retention Insight**: A startup offered a **$10 incentive** for feedback on their no-code AI workflow builder to analyze user drop-off post-registration, with a note that the discussion should continue in a more relevant channel.
- **Cohere's Market Strategy**: A Cohere employee emphasized that the company is not pursuing **Artificial General Intelligence (AGI)**, but is instead committed to developing **scalable models** for production environments.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Memory Lane with `ChatMessageHistory`**: Kapa.ai illustrated the use of **LangChain's** `ChatMessageHistory` class for **persisting chat conversations**, providing a clear example of maintaining context across sessions, with a nod to the [LangChain documentation](https://python.langchain.com/v0.1/docs/modules/agents/quick_start/#adding-in-memory).

**Navigating LLM Conversation Complexity**: Discussion centered around the difficulties of designing **non-linear conversation flows** with Large Language Models (LLMs), citing extraction and JSON handling concerns. An experimental approach on **GitHub** was linked to demonstrate these challenges in action.

**Crafting an Analytical Copilot**: Engineering dialogue included strategies for pairing **LangChain** with a **PostgreSQL database**, offering insight into handling ambiguous SQL query results via few-shot learning.

**Hybrid Agents for Enhanced Interactivity**: Integration of `create_react_agent` and `create_sql_agent` within **LangChain** was unraveled, detailing steps to avoid common initialization pitfalls and the importance of naming tools correctly for successful operation.

**Evolving AI Assistants & Knowledge Graphs**: Wave of new releases like **Everything-ai v3.0.0** included advancements like integrating **llama.cpp** and **Qdrant-backed vector databases**, while a tutorial video [shared across channels](https://www.youtube.com/watch?v=Bxj4btI3TzY) provided learners with a practical guide to creating bots using **Pinecone**, **LangChain**, and **OpenAI**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Price Hikes Spark Cost-effectiveness Debate**: Community members discussed a sharp pricing change for an unnamed service, challenging its previously acclaimed cost-effectiveness; suspicions arise if praise was based on the post-hiked rates.
- **GPT-5 Speculation Intensifies**: An unconfirmed [table from X](https://x.com/VachanReddyK/status/1795099977766551828/photo/1) discussing GPT-5 led to speculation that OpenAI might make GPT-4o free in preparation for the new model; pointers to AI expert Alan D. Thompson's insights were noted [About Alan](https://lifearchitect.ai/about-alan/).
- **OpenAI Pricing Called Out for Typos**: A typo in OpenAI's initial pricing announcement created confusion, later addressed and corrected within 24 hours; corrected pricing now reflects the company's intentions [Official post by LoganK](https://x.com/officiallogank/status/1796044236594278544?s=46).
- **OpenAI's Commercial Shift Stirring Discontent**: Internal tensions at OpenAI surfaced in discussions referencing Microsoft's alleged pressure on the company to prioritize commercialization over research, leading to division among staff [Financial Times article](https://www.ft.com/content/ccbdff7c-ede3-4d62-968d-189fb0205075).
- **OpenAI and Apple Collaboration Causes a Stir**: The community reflected on the strategic implications and potential conflicts within the Azure-Apple partnership given Microsoft's investment in OpenAI; the blend of commercial dynamics and data policy considerations is under scrutiny.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter Rocks the Docs**: The [OpenInterpreter documentation](https://docs.openinterpreter.com/settings/all-settings#auto-run) received positive spotlight, featuring a list of language models with the notable **LiteLLM** supporting 100+ models. Attention was also drawn to the development of an Android/iOS client specifically tailored for the **RayNeo X2** and **Brilliant Labs** frames, with the community eager to test the app shared via [GitHub](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile).

- **LLaMA Heats Up Discussion**: Engineers engaged in a heated debate over the use of **LLaMA** locally, particularly with NVLinked 3090 setups that run hot. Alternatives were suggested, including taking advantage of **Groq** for free model access, steering the conversation towards more sustainable and efficient hardware solutions.

- **TTS Enthusiasm Voices Concern**: The query for personalizing TTS with individual voices sparked curiosity with no direct solutions linked. Meanwhile, a member queried about the shipment of an order placed on April 30, 2024, only to be directed towards specific pinned manufacturing updates, hinting at an operational focus on communication from product developers.

- **M5 Cardputer Rallying Anticipation**: An update about the M5 cardputer stirred some fuss, balancing users’ excitement with skepticism, and the assurance was found in a pinned message outlining the latest manufacturing details. Additionally, a cautionary reminder circulated about using the [ChatTTS model on Hugging Face](https://huggingface.co/2Noise/ChatTTS) strictly for educational purposes, emphasizing adhering to academic integrity.

- **Model Curiosity Peaks with Codestral**: Inquiry into the new Codestral model prompted member interest, suggesting potential for testing and reviews. The community appears willing to explore new modeling wonders, highlighting a proactive engagement with the latest in model development.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **ChatGPT Free Tier Just Got Beefier**: OpenAI has enhanced the *ChatGPT Free* tier with new abilities: browse, vision, data analysis, and file uploads. Users are considering "**rate limits**" as a potential constraint, with the official announcement available [here](https://x.com/openai/status/1795900306490044479?s=46&t=90xQ8sGy63D2OtiaoGJuww).

- **Conversational Voice AI, A16Z Bets Big**: Skepticism and interest mingle as members discuss [a16z's investment in voice AI](https://x.com/omooretweets/status/1795834644732285402), theorizing how AI might revolutionize phone calls beyond the investor excitement.

- **Cartesia Breaks Sound Barriers with Sonic**: Cartesia's launch of [Sonic](https://play.cartesia.ai/), their new low-latency generative voice model, is stirring conversations about its application in real-time multimodal contexts. For more insight, take a look at their [blog post](https://cartesia.ai/blog/sonic).

- **YC's Leadership Shuffle Decoded**: Paul Graham clarifies on Twitter the speculation regarding Sam's departure from Y Combinator, dismissing rumors of a firing in [his tweet](https://x.com/paulg/status/1796107666265108940?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Retrieval-Enhancing Embedding Adapters**: The engineering crowd paid close attention to TryChroma's technical report on [embedding adapters](https://research.trychroma.com/embedding-adapters), focusing on improving retrieval performance, a concept closely related to Vespa's use of frozen embeddings.

- **Podcast Unpacks Million Context LLMs**: A new podcast episode featuring @markatgradient discusses the challenges of training a million context LLM, referencing historical methods and variants like RoPE, ALiBi, and Ring Attention. The episode can be streamed [here](https://x.com/latentspacepod/status/1796247856891969709).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**LLM360 Launches Community AMA**: Mozilla AI's LLM360 kicks off community engagement with an [AMA on their new 65B model and open-source initiatives](https://discord.com/events/1089876418936180786/1240722407594004561), fostering knowledge sharing and Q&A with AI enthusiasts.

**Bay Area Engineers, Mark Your Calendars**: An [IRL Open Source Hack Lab event](https://discord.com/events/1089876418936180786/1243732435674337413) has been scheduled in the Bay Area, inviting local members to collaborate and share their expertise.

**Embeddings Insight Session**: A community session on utilizing [llamafiles for generating embeddings](https://discord.com/events/1089876418936180786/1242590711778381914) promises a practical learning experience for engineers seeking to apply embeddings in their machine learning projects.

**Developer Support Enhanced at Mozilla AI**: In the "Amplifying Devs" event, moderator-led discussions will focus on better supporting the development community within Mozilla AI, an essential platform for developer growth and collaboration.

**Tackling LlamaFile Puzzles**: Engineers report challenges with `granile-34b-code-instruct.Q5_0.llamafile` when running on M2 Studio and using VectorStoreIndex in Python, with solutions involving correct IP binding and addressing WSL localhost quirks. Interest in LlamaFiles with vision/image capabilities is growing, highlighted by Mozilla's llava-v1.5-7b-llamafile [available on Hugging Face](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/tree/main), potentially offering image support for creative AI applications.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Fine-Tuning LLMs for Multimedia Tasks**: Members are exploring ideal strategies to fine-tune **large language models** (LLMs), such as **Llava**, for tasks involving **image and video understanding**. The benefits and practicality of using **Direct Preference Optimization** (DPO) as opposed to **Supervised Fine-Tuning** (SFT) have precipitated a lively debate, particularly regarding the volume of data required for effective DPO.

**DPO's Diminished VRAM Appetite**: An unexpected reduction in **VRAM usage** during DPO has piqued the interest of one engineer, sparking speculation on recent updates that might have led to such efficiency gains.

**Protobuf Heavyweight Champion Wanted**: There’s an open call within the community for experts with a strong background in **Google's Protobuf**, especially those who can boast reverse engineering, malware analysis, or bug bounty hunting skills.

**SDXL Custom Ads Campaign Hits a Snag**: Someone's request for expertise in **refining SDXL models** is still hanging in the ether, as they aim to optimize their models for producing customized product advertisements and have not yet obtained the desired results with **LoRA training** or **ControlNet**.

**Small Data for Grand Conversations**: Curiosity abounds as to whether a small dataset of merely **hundreds of samples** could possibly suffice for successful DPO, particularly for domains as nuanced as general chitchat. It has been suggested that manually compiling such a dataset could be a practical approach.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI-Powered Literature to Gameplay Transition**: Rosebud AI is hosting a **Game Jam: "Book to Game"** where participants will use Phaser JS to turn books into games on the AI Game Maker platform, competing for a **$500 prize** with submissions due by July 1st. News of the jam was shared via [Rosebud AI's tweet](https://x.com/Rosebud_AI/status/1796273820044595368) and interested devs can join their [Discord community](https://discord.gg/rosebud-ai).

**Android Access Annoyance**: A newcomer to the Discord community described the Android experience as *"a bit hard to navigate... Glitchy and buggy"* but confirmed they are still able to engage with content. They also inquired about changing their username, expressing a feeling of being an *"alien"*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GPU Future Speculations Spark Curiosity**: Discussion on the evolution of GPUs in the next **2 to 5 years** hinted at the use of larger **64x64 matrix multiplication arrays (MMA)**, poking fun at the idea with a suggestion to "make a bigger systolic array 😌."

- **Tinygrad Outshines Torch with Integer Gradients**: Tinygrad has been highlighted for its ability to compute gradients for integers, a task that causes a `RuntimeError` in Torch. Tinygrad handles this by treating integers as floats during backpropagation before casting back to integers.

- **Debating Framework Dominance in AI**: A member asserted the superiority of **Tinygrad** over **TensorFlow and PyTorch**, igniting a conversation about why TensorFlow might be preferred over PyTorch in the AI community despite individual preferences for Tinygrad.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Language-Specific Codestral Proposed**: A member sparked a conversation about the potential for a smaller **Codestral** by splitting it into individual programming languages, postulating that not all languages may contribute equally to the overall model.
- **Curiosity about Language Weights**: There's curiosity about the weight distribution in the **45GB Codestral model**, with speculation that most weights are assigned to English but each programming language might still significantly impact the model's overall capabilities.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

Unfortunately, as there is only one message provided and this message lacks sufficient technical content or details relevant to AI Engineers, it is not possible to create a summary as per the given guidelines. If more messages with the appropriate detail are provided, a summary can be generated.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Join the Open GPT-4-Omni Initiative**: [LAION](https://laion.ai/notes/open-gpt-4-o/) calls for community contributions to develop an open version of GPT-4-Omni, providing datasets, tutorials, and a guiding blog post. They also broadcasted their message through a [Twitter post](https://fxtwitter.com/laion_ai/status/1795910332008804428?t=rBHUXm87TFrQ-kyfeZP0fg&s=19) encouraging wider involvement.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


{% if medium == 'web' %}

# PART 2: Detailed by-Channel summaries and links



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1245766462090776747)** (1 messages): 

- **Perplexity Pages transform research into articles**: Perplexity has launched **Perplexity Pages** to help users transform their research into visually appealing articles. Users can start creating Pages in their Library, with more information available on [Perplexity's blog](https://pplx.ai/pages).
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1245451617420181536)** (686 messages🔥🔥🔥): 

```html
- **Grok fails to impress; sneakyf1shy builds better search model**: Users discussed their disappointment with Grok and sneakyf1shy mentioned working on a similar project with intentions of enhancement. They aim to surpass Perplexity's web app by creating a comprehensive custom searching pipeline.
- **OpenAI and API enhancements**: Conversations highlighted the challenges of creating user-friendly APIs and scaling them effectively. Some users, such as sneakyf1shy, expressed interest in developing API solutions that could improve multi-step reasoning and integrating own indexing/cache layers.
- **Perplexity Pages gains traction; user experiences varied**: Many users explored Perplexity Pages, sharing their experiences and learnings. Some users encountered issues like missing sections in converted threads, while others found it a valuable addition for documentation and knowledge databases. One user shared a [Perplexity Pages guide](https://www.perplexity.ai/page/How-to-Use-FvLfzZ_ATyqE2n_tAGKk7A).
- **Skepticism and API limitations**: Users expressed skepticism about Perplexity's use of its own index, questioning the true capabilities of their web scraper. Some lamented the inactivity and limited availability of the API, while others discussed alternative models and their efficiencies.
- **Google and OpenAI comparisons stir debate**: Lively debates ensued about Google’s and OpenAI’s AI strategies, resource usage, and effectiveness in comparison to competitors like Nvidia. Users speculated on AGI developments and commercial impacts, especially regarding OpenAI's products and potential future releases.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com">Ollama</a>: Get up and running with large language models.</li><li><a href="https://www.theverge.com/2024/5/29/24167511/openai-appears-to-have-closed-its-deal-with-apple">OpenAI appears to have closed its deal with Apple.</a>: Apple held talks with both Google and OpenAI about integrating their chatbots into iOS 18, according to Bloomberg, but it looks like OpenAI won out. The pair plan to announce the news at Apple’s devel...</li><li><a href="https://tenor.com/view/cute-adorable-sticker-stickers-gif-15884535962552606625">Cute Adorable GIF - Cute Adorable Sticker - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=-qGa0oTY120">Introducing Perplexity Pages</a>: You’ve used Perplexity to search for answers, explore new topics, and expand your knowledge. Now, it’s time to share what you learned.Meet Perplexity Pages, ...</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">Doubt Press X GIF - Doubt Press X La Noire - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://start.solidjs.com/">SolidStart: Fine-Grained Reactivity goes fullstack</a>: SolidStart is a JavaScript Framework designed to build SolidJS apps and deploy them to a variety of providers.</li><li><a href="https://perplexity.typeform.com/pages-beta">Perplexity Pages - Beta Access</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1245456771984785460)** (15 messages🔥): 

- **Perplexity debate on advantages/disadvantages**: A member shared a link exploring the **pros and cons** of a topic in a detailed manner. Check out the full discussion [here](https://www.perplexity.ai/search/Vor-und-Nachteile-jyWAvvwhT1qoWsdFiCP7mQ).
- **Understanding divisive questions**: Two members shared the same link to a Perplexity search about **division**—likely discussing a divisive topic or technical query in detail. Explore the search result [here](https://www.perplexity.ai/search/what-is-div-44iy0Oo.SqSCDZjhN.IPiw).
- **Diving into the physics and ethics**: A member shared links to pages diving into both the **physics** and **ethics** of a particular subject. Read the full write-ups on [The Physics of](https://www.perplexity.ai/page/The-Physics-of-zJlhgwErRNiV5RndBBtOfA) and [The Ethics of](https://www.perplexity.ai/page/The-Ethics-of-gIYG3OV8TGm.neismMEbWQ).
- **Exploring consciousness and AI functionalities**: A member re-shared popular beta pages discussing **consciousness** and **LLM functions**, aiming to gain further views and feedback. Visit the discussion on [consciousness](https://www.perplexity.ai/page/Understanding-the-Conscious-Gtw786J5QQe4EpR4TfusXw) and [LLMs function](https://www.perplexity.ai/page/How-LLMs-function-h515ZojmQFiTCxRHB3OEyw).
- **Perplexity's new AI Wikipedia feature**: Arav Srivinas detailed Perplexity’s vision to create **AI Wikipedia** with Pages, now in Pro and soon for all. See the full announcement and details on Twitter [here](https://x.com/AravSrinivas/status/1796220011448786949) and the blog post [here](https://www.perplexity.ai/hub/blog/perplexity-pages?utm_medium=social&utm_campaign=pages-launch).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1796220481055654072">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Not everyone needs to go through the flow of asking and prompt engineering a chat session to gain knowledge from Perplexity daily. We were the first to allow sharing “threads” through permalinks. Yet,...</li><li><a href="https://x.com/AravSrinivas/status/1796221195542786522">Tweet from Aravind Srinivas (@AravSrinivas)</a>: This is just the beginning of many more amazing things: generating full research reports, blog posts, an entire book on a topic, or a briefing on some of the latest happenings or bios of individuals. ...</li><li><a href="https://x.com/AravSrinivas/status/1796220757514842262">Tweet from Aravind Srinivas (@AravSrinivas)</a>: You can create a page as a separate entity like you write a doc (with full access to the internet), or you could just simply continue asking questions on Perplexity as you do today and convert it to t...</li><li><a href="https://x.com/AravSrinivas/status/1796220011448786949">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Perplexity’s mission is to cater to the world&#39;s curiosity. We have taken inspiration from Wikipedia with citations. We’re excited to take it further by launching Pages, best described as “AI Wikip...
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1245462218842312927)** (50 messages🔥): 

- **Google Pricing Confusion with Gemini**: Members are puzzled by the discrepancies in pricing between Google's Vertex AI and their direct AI services ([source](https://ai.google.dev/)). One pointed out, "it is billed per character on Vertex AI and per token on [ai.google.dev](https://ai.google.dev)".
  
- **Deterministic Tool Calling in Multi-Agent Systems**: Several members discussed strategies for building GPT-powered agents with tool-calling capabilities. A resource was shared on **GitHub - robocorp/llmstatemachine** ([source](https://github.com/robocorp/llmstatemachine)) that uses state machine logic.
 
- **Consolidated Course Resources Repository**: A member created a GitHub repository to consolidate useful links and slides shared during the course ([source](https://github.com/bikash119/mastering_llm)). Another resource shared was a Twitter list of all the past speakers for the course ([source](https://x.com/i/lists/1796060854359580751)).
 
- **Queries and Issues with Credit Form**: Several messages discussed the credit form not sending confirmation. **Danbecker** confirmed the form doesn't send confirmations but reassured users by thanking them for their patience.
  
- **Finetuning LLMs and GGUF Format**: Discussion around finetuning LLMs on custom data using the GGUF format was prevalent. There was excitement about upcoming improvements to ease the conversion from HF to GGUF, with an ongoing HuggingFace PR shared as a relevant update ([source](https://github.com/huggingface/transformers/pull/30928)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ankur-singh.github.io/blog/finetune-inference">Run your finetuned LLM with Ollama</a>: Complete workflow demonstrating how to finetune an LLM on your data and run it using Ollama.</li><li><a href="https://tenor.com/view/rug-pull-gif-21378865">Rug Pull GIF - Rug Pull - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ai.google.dev/">no title found</a>: no description found</li><li><a href="https://github.com/swyxio/ai-notes/blob/main/Resources/Good%20AI%20Podcasts%20and%20Newsletters.md">ai-notes/Resources/Good AI Podcasts and Newsletters.md at main · swyxio/ai-notes</a>: notes for software engineers getting up to speed on new AI developments. Serves as datastore for https://latent.space writing, and product brainstorming, but has cleaned up canonical references und...</li><li><a href="https://youtu.be/Wo95ob_s_NI?si=S-Kxzq01GKmGs6oa">John Schulman (OpenAI Cofounder) - Reasoning, RLHF, &amp; Plan for 2027 AGI</a>: John Schulman on how posttraining tames the shoggoth, and the nature of the progress to come...Timestamps:00:00:00 Pre-training, post-training, and future ca...</li><li><a href="https://github.com/robocorp/llmstatemachine">GitHub - robocorp/llmstatemachine: A Python library for building GPT-powered agents with state machine logic and chat history memory.</a>: A Python library for building GPT-powered agents with state machine logic and chat history memory. - robocorp/llmstatemachine</li><li><a href="https://x.com/i/lists/1796060854359580751)">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/bikash119/mastering_llm">GitHub - bikash119/mastering_llm</a>: Contribute to bikash119/mastering_llm development by creating an account on GitHub.</li><li><a href="https://x.com/chroepke">Tweet from undefined</a>: no description found</li><li><a href="https://calendly.com/christianroepke/introductory-call">30 Min Introductory Call - Christian Röpke</a>: Book a free introductory call with me to share your challenges and learn how I could help you.</li><li><a href="https://github.com/huggingface/transformers/pull/30928">FEAT / Trainer: Experimental feature - add `GGUF` conversion when pushing the model to Hub by younesbelkada · Pull Request #30928 · huggingface/transformers</a>: What does this PR do? Introduces a new quantization_config that is intended to be used only for trainer.push_to_hub(), it calls ` a GGUF conversion Space under the hood - (for now: https://huggingf...</li><li><a href="https://www.quora.com/Should-you-fine-tune-an-LLM-or-just-do-prompt-engineering/answer/Tong-Hui-Kang-1,">Should you fine-tune an LLM, or just do prompt engineering? - Quora</a>: no description found</li><li><a href="https://youtu.be/Mn_9W1nCFLo?si=SWUPvbQ9ZCAxmAK_">LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU</a>: Full explanation of the LLaMA 1 and LLaMA 2 model from Meta, including Rotary Positional Embeddings, RMS Normalization, Multi-Query Attention, KV-Cache, Grou...</li><li><a href="https://youtu.be/UiX8K-xBUpE?si=UgGM6oimKVhvub-b">Mistral / Mixtral Explained: Sliding Window Attention, Sparse Mixture of Experts, Rolling Buffer</a>: In this video I will be introducing all the innovations in the Mistral 7B and Mixtral 8x7B model: Sliding Window Attention, KV-Cache with Rolling Buffer, Pre...</li><li><a href="https://youtu.be/bCz4OMemCcA?si=X5lnwL_cmE16XFFS">Attention is all you need (Transformer) - Model explanation (including math), Inference and Training</a>: A complete explanation of all the layers of a Transformer Model: Multi-Head Self-Attention, Positional Encoding, including all the matrix multiplications and...</li><li><a href="https://github.com/leloykun">leloykun - Overview</a>: Machine Learning (AI) Research Engineer @expedock • 2x IOI &amp; 2x ICPC World Finalist • Math @ AdMU - leloykun</li><li><a href="https://leloykun.github.io/">Franz Louis Cesista</a>: Mathematician | Machine Learning (AI) Research Scientist</li><li><a href="https://calendly.com/leloy/chat-with-franz">Chat w/ Franz - Franz Louis Cesista</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1245600306302287913)** (5 messages): 

- **Workshop slides shared**: Workshop 1 slides are now available in a [Google presentation](https://docs.google.com/presentation/d/1hcfR4ZhAMmzFFiXJi_O3WE9fojOmHZbjSoG7DcJivso/edit#slide=id.g1ec9867125_0_0). The document is set to view-only mode.

- **Finetuning for dialect recognition**: Discussing the best approach to make an LLM understand and distinguish between different Fenni-Swedish dialects. The suggestion is to finetune using a model that knows Swedish and to tag each dialect, potentially treating them as separate languages.

- **Debugging the model and finetuning with Axolotl**: A user seeks advice on debugging a model and confirms the chat template's functionality. They mention trying to fine-tune the Qwen model but face issues with training starting and then stopping, sharing their [notebook](#).

- **Scalability in natural language to SQL**: A user discusses the challenge of generating accurate SQL queries from natural language for a vast number of tables. They seek effective methods to filter relevant tables and add knowledge to the LLM, sharing their current approach and issues with overfitting and hallucinations during finetuning.

- **Using knowledge graphs for table relations**: Suggestion to build a knowledge graph on table relations and use it to identify the relevant tables for query generation. This approach can help filter the tables and provide necessary context for the LLM.

**Link mentioned**: <a href="https://docs.google.com/presentation/d/1hcfR4ZhAMmzFFiXJi_O3WE9fojOmHZbjSoG7DcJivso/edit#slide=id.g1ec9867125_0_0">fine-tuning workshop 1 slides</a>: LLM Fine Tuning For Data Scientists &amp; Software Engineers

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1245512391899349103)** (2 messages): 

```html
- **New Member from Sydney Joins the Team**: A new member introduced themselves, noting they are a Senior Manager in Advanced Analytics based in Sydney, Australia. They expressed interest in applying fine-tuning for specific use cases and deploying LLMs using minimal prompting, as well as learning about best practices for hosting and deploying LLMs in production settings.

- **Global AI Hackathon Alert**: An upcoming **Global AI Hackathon** from June 7 to 9 was announced, facilitating events in multiple cities including Singapore, Sydney, and San Francisco. Attendees are encouraged to RSVP via [this link](https://lu.ma/igqisb0e), noting that the hackathon is backed by top AI builders and aims to address "AI for a better world".
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/igqisb0e">Singapore | Build Together: AI Hackathon · Luma</a>: Note: This event is capped so please RSVP to secure your spot. Please only RSVP if you can attend the full event Come build and network with top AI hackers and…</li><li><a href="https://www.buildclub.ai/events/build-together">Build Club</a>: Build Club is the best place in the world for AI founders to connect and launch their startups. It&#x27;s a 100% free community for top builders. Come join us!
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1245461312679706644)** (74 messages🔥🔥): 

- **Troubleshooting Modal Task Execution**: Multiple users encountered issues while running their training tasks on Modal, with errors and confusion about dataset paths and config settings. The consensus was to ensure dataset paths in config files match expected locations and use Modal's cloud storage appropriately.
- **WANDB Integration Hiccups**: Users had trouble getting WANDB integration to function correctly, leading to advice on renaming secrets and setting environment variables like `ALLOW_WANDB=true` before training runs. *"Your secret has to be renamed. You must delete it and change it from 'my-wandb-secret' to 'wandb'"*.
- **Clarifying Configurations and Secrets**: Users shared their configuration files and discussed the correct settings for paths and secret names, including ensuring `wandb_watch` is configured properly. 
- **Helpful References and Examples**: Users were directed to [Modal's documentation](https://modal.com/docs/guide/trigger-deployed-functions) and example repositories to better understand how to deploy and invoke functions on Modal. **"I would recommend getting started with Modal's hello world example (https://modal.com/docs/examples/hello_world)"**.
- **App Functionality and Next Steps**: After successfully deploying their model, users discussed how to proceed, including using Modal's platform for further experimentation and practical applications with the deployed models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modal-labs/llm-finetuning/blob/main/README.md">llm-finetuning/README.md at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/5eb7c4b3034ce67e177e9924ef6642c2cad4bc17/config/llama-3.yml#L7C1-L10">llm-finetuning/config/llama-3.yml at 5eb7c4b3034ce67e177e9924ef6642c2cad4bc17 · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://wandb.ai/hongnangao/golden-gate-bridge-repeng/runs/6fo15ch8/overview?nw=nwuserhongnangao">hongnangao</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/hongnangao/golden-gate-bridge-repeng/runs/5k4or992?nw=nwuserhongnangao">hongnangao</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/5eb7c4b3034ce67e177e9924ef6642c2cad4bc17/src/train.py#L150-L154">llm-finetuning/src/train.py at 5eb7c4b3034ce67e177e9924ef6642c2cad4bc17 · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://modal.com/docs/guide/trigger-deployed-functions">Invoking deployed functions</a>: Modal lets you take a function created by a deployment and call it from other contexts.</li><li><a href="https://modal.com/apps">Sign in</a>: Welcome back to Modal! Sign in to your Modal account by selecting an identity provider below.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1245600288614649889)** (5 messages): 

- **Meta Paper on vLLM Released**: A member shared a paper released by **AI at Meta** that's relevant for vLLM. Find the paper [here](https://arxiv.org/abs/2405.17247).

- **GitHub Repo for LLM Resources**: A member started a GitHub repository collecting resources on LLMs, useful for anyone participating in the "Mastering LLMs" workshop. Access and contribute to the repo [here](https://github.com/marco-jeffrey/awesome-llm-resources).

- **Expanding LLama3 Context Window**: A paper discussing how **LLama3's context window was extended** from 8K to 80K was shared. The entire resource set including data, model, data generation pipeline, and training code will be publicly released [here](https://arxiv.org/abs/2404.19553).

- **AI-coding Humble Bundle**: There's a **Humble Bundle** available for AI-coding and prompt engineering, which might be of interest to the community. More details and purchase options are available [here](https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.17247">An Introduction to Vision-Language Modeling</a>: Following the recent popularity of Large Language Models (LLMs), several attempts have been made to extend them to the visual domain. From having a visual assistant that could guide us through unfamil...</li><li><a href="https://arxiv.org/abs/2404.19553">Extending Llama-3&#39;s Context Ten-Fold Overnight</a>: We extend the context length of Llama-3-8B-Instruct from 8K to 80K via QLoRA fine-tuning. The entire training cycle is super efficient, which takes 8 hours on one 8xA800 (80G) GPU machine. The resulte...</li><li><a href="https://github.com/marco-jeffrey/awesome-llm-resources">GitHub - marco-jeffrey/awesome-llm-resources: a collection of resources around LLMs, aggregated for the workshop &quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&quot; by Dan Becker and Hamel Husain&quot;</a>: a collection of resources around LLMs, aggregated for the workshop &amp;quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&amp;quot; by Dan Becker and Hamel Husain&amp;quot; - marco-jeffrey/aw...</li><li><a href="https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software?mcID=102:66576d20c5895a1aa5046052:ot:5ccaf0c3db76615eab12deb2:1&linkID=66576d225fb588c450040093&utm_campaign=2024_05_30_completechatgptanthropicgeminipromptengineeringapiandprogramming_softwarebundle&utm_source=Humble+Bundle+Newsletter&utm_medium=email">The Complete ChatGPT, Anthropic, Gemini Prompt Engineering, API, and Programming Mega Bundle</a>: AI is rising—rise along with it with these online courses! Learn prompt engineering, LangChain, & more! Your purchase helps the Children’s Miracle Network.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1245722980898701354)** (2 messages): 

- **Running on Official Axolotl Docker**: *We are running on the official axolotl docker. It builds once everyday.* If you share the exact config with some sample dataset, we can try it from our end.

- **Config and Commands Shared for Debugging**: [This is the config](https://discord.com/channels/1238365980128706560/1242542198008975430/1245437637624467569) and [these are the commands issued](https://discord.com/channels/1238365980128706560/1242542198008975430/1245437637624467569). There might be an issue with the tokenizer, but it hasn’t been debugged yet.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1245481968758423582)** (2 messages): 

- **Billing is a prerequisite for Replicate credits**: A member inquired if setting up billing is needed to be eligible for Replicate credits. Another confirmed this requirement and advised setting a low monthly limit to avoid unwanted charges.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1245844758509912105)** (1 messages): 

- **Langsmith HIPAA Compliance Query**: A user inquired if Langsmith offers paid plans that allow deployment on **HIPAA-compliant frameworks**. The use case involved handling PII/PHI, necessitating the vendor to be a Business Associate and have a DPA in place.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[ankurgoyal_textsql_llmevals](https://discord.com/channels/1238365980128706560/1242222674835538012/1245453864476348497)** (2 messages): 

- **Interesting Text2SQL Methods Review**: One member shared an [interesting review](https://arxiv.org/pdf/2403.02951) of different Text2SQL methods. Another user expressed gratitude, noting the resource was very useful.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1245463062274637915)** (16 messages🔥): 

- **Copilot Chat becomes domain expert with @workspace**: A user shared a [link](https://code.visualstudio.com/docs/copilot/workspace-context#_tips-for-using-workspace) explaining how referencing `@workspace` in Copilot Chat allows it to intelligently retrieve relevant files and symbols. Examples include finding where a database string is configured or validating a date within the codebase.

- **Inline chat in terminal for Copilot**: It's noted that Copilot can be invoked directly in the terminal, a feature most people aren't aware of, enhancing its usability beyond just the editor.

- **Copilot vs. Cursor Debate**: Users compared Copilot and Cursor, praising Copilot for its better results and solutions. However, Cursor's ability to inject custom models (like GPT-4) and its customizable environment were highlighted as significant advantages.

- **JSON Schema and Zod for function calling**: A user shared [JSON Schema info](https://www.notion.so/matijagrcic/JSON-Schema-78055af9ce1242e8b9be27918056be2f) and examples using Zod from OpenAI's [GitHub](https://github.com/openai/openai-node/blob/master/examples/tool-call-helpers-zod.ts), though noting some examples are outdated. They mentioned that using Deno as a Jupyter notebook kernel works nicely with these examples and promised to publish more details soon.

- **Document Mimicry understanding shared on Twitter**: A user thanked another for their insights and shared a [Twitter post](https://twitter.com/nehiljain/status/1795949311135502443) explaining Document Mimicry. This user found prompting with Document Mimicry in mind highly beneficial.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://code.visualstudio.com/docs/copilot/workspace-context#_tips-for-using-workspace">Chat using @workspace Context References</a>: How to use Copilot's @workspace chat to ask questions against your entire codebase.</li><li><a href="https://www.notion.so/matijagrcic/JSON-Schema-78055af9ce1242e8b9be27918056be2f?pvs=4,">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1245456482623815781)** (6 messages): 

- **Solve error by reinstalling package**: Members discussed resolving an error by reinstalling the package. Suggestions included using `pip install -e` and templates from [Jarvis Labs](https://jarvislabs.ai).

- **Success using Jarvis template**: One member confirmed they successfully resolved the error using the template provided by Jarvis Labs. This proved helpful to others experiencing similar issues.

- **Workshop 2 slides available**: Workshop 2 slides are available on Google Docs. You can view them [here](https://docs.google.com/presentation/d/1otXeE6D5kJiDuxFYk3t9Nq9pKesN4-_6YhgLGRXmSU4/edit#slide=id.g1ec9867125_0_0).

**Link mentioned**: <a href="https://docs.google.com/presentation/d/1otXeE6D5kJiDuxFYk3t9Nq9pKesN4-_6YhgLGRXmSU4/edit#slide=id.g1ec9867125_0_0">Fine-tuning workshop 2 slides</a>: Mastering LLMs A Conference For Developers &amp; Data Scientists

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1245456201567961138)** (18 messages🔥): 

- **FIB Paper Benchmarks Highlight LLMs' Factual Accuracy**: The [FIB Paper](https://arxiv.org/abs/2211.08412) focuses on measuring LLMs' factual consistency in summarization tasks, showing that models like BLOOM assign higher scores to factually consistent summaries but struggle with verbatim consistency. Another paper, [linked here](https://arxiv.org/abs/2305.11747), questions LLMs' effectiveness at identifying inconsistent summaries.
  
- **Fine-Tuning Reduces Hallucinations with Clean Data**: Fine-tuning to reduce hallucinations is effective when using clean training data, as highlighted in this [tweet thread](https://x.com/stefanhgm/status/1765466556216053879). However, [research shows](https://arxiv.org/abs/2405.05904) that fine-tuning with new knowledge can increase hallucinations, especially in tasks like closed-book QA.

- **Streamlining Eval Processes for LLMs**: Discussions emphasized enhancing eval processes for tasks like Text-to-SQL using L1 (unit tests and assertions for syntax) and L2 (human feedback for relevance). Leveraging execution results and fuzzy searches based on schema can validate correctness, and performance evaluations require more nuanced checks.

- **Eval Libraries for LLM Scoring**: Recommendations for scoring LLM evaluations include [Hugging Face's Evaluate library](https://huggingface.co/docs/evaluate/en/index) and Braintrust's [Autoevals](https://github.com/braintrustdata/autoevals), which offer various evaluation methods for NLP, AI models, and more. These tools aim to streamline the process with best practices and reproducibility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/stefanhgm/status/1765466556216053879">Tweet from Stefan Hegselmann (@stefanhgm)</a>: Does removing unsupported facts in the training or prompting data effectively reduce hallucinations?  We tested this for GPT-4 & Llama 2 for generating patient summaries. W/ @shannonzshen, Florian Gie...</li><li><a href="https://eugeneyan.com/writing/evals/#summarization-consistency-relevance-length">Task-Specific LLM Evals that Do & Don't Work</a>: Evals for classification, summarization, translation, copyright regurgitation, and toxicity.</li><li><a href="https://huggingface.co/docs/evaluate/en/index">🤗 Evaluate</a>: no description found</li><li><a href="https://github.com/braintrustdata/autoevals">GitHub - braintrustdata/autoevals: AutoEvals is a tool for quickly and easily evaluating AI model outputs using best practices.</a>: AutoEvals is a tool for quickly and easily evaluating AI model outputs using best practices. - braintrustdata/autoevals</li><li><a href="https://arxiv.org/abs/2405.05904">Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?</a>: When large language models are aligned via supervised fine-tuning, they may encounter new factual information that was not acquired through pre-training. It is often conjectured that this can teach th...</li><li><a href="https://arxiv.org/abs/2211.08412">Evaluating the Factual Consistency of Large Language Models Through News Summarization</a>: While large language models (LLMs) have proven to be effective on a large variety of tasks, they are also known to hallucinate information. To measure whether an LLM prefers factually consistent conti...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1245763983894515812)** (1 messages): 

- **Fine-tuning Gradio Docs Needs Clarity**: A member expressed interest in helping with a Gradio documentation fine-tuning project, particularly focused on creating a fine-tune input/output dataset. They suggested generating user-centered questions from code blocks, aiming to produce granular, actionable responses, and inquired about templates to facilitate this conversion process.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1245475686127370261)** (56 messages🔥🔥): 

- **Axolotl README example stalls**: A user reported running the `axolotl` README example (`accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml`) but getting stuck with maxed-out GPU memory. Discussion ensued about potential issues with evaluation, disk read/write utilization, and challenges associated with multi-GPU configuration.

- **Single GPU success for OpenLLaMA-3B example**: After trying various configurations, the user found the `openllama-3b` example worked on a single GPU, indicating a possible issue with multi-GPU settings. They shared their config and noted making changes to use `bf16` and enabling `tf32`.

- **NCCL issues on WSL2**: Another user sought advice on installing NCCL on WSL2 but faced multiple errors. Recommendations were given to switch to Linux and Docker for a more stable setup, with some users sharing their experiences and suggesting alternative configurations like `ddp_backend: gloo`.

- **Prompt template configurations**: A member inquired about using standard templates versus custom templates in the config for specific training tasks, particularly focused on function calling with datasets having specific columns. The discussion encouraged sharing best practices for template usage in configurations.

- **GPU performance results**: Results from running CodeLlama 7B on different GPUs were shared, showing significant variations in training times per epoch. The member pointed out these findings to clarify discrepancies with the times noted in the README.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1245524869127471115)** (35 messages🔥): 

- **FSDP facilitates data parallelism**: A member shared a GitHub issue link about [FSDP enabling seamless switches between DDP, ZeRO-1, ZeRO-2, and FSDP](https://github.com/pytorch/pytorch/issues/102532), noting that DeepSpeed is harder to use while FSDP offers ease for LLMs.
- **Inference issues with hf+accelerate**: A member reported getting "mixed alphabets" when running meta-llama examples, suspecting an issue with `device_map="auto"`. They provided code snippets for context and received suggestions to share findings before tagging the Accelerate team.
- **Prompt: Community troubleshooting**: The troubleshooting thread had back-and-forth code sharing and suggestions, guiding the user to open a GitHub issue if the issue persisted. The aim was to aid debugging and expedite resolution for similar future issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/openai-community/gpt2">openai-community/gpt2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/issues/102532">FSDP enable users to seamlessly switch between DDP, ZeRO-1, ZeRO-2 and FSDP flavors of data parallelism · Issue #102532 · pytorch/pytorch</a>: 🚀 The feature, motivation and pitch https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/ DeepSpeed is hard to use, FSDP is easy to use for LLM, but FSDP do not support ZeRO.....</li><li><a href="https://github.com/huggingface/accelerate/issues">Issues · huggingface/accelerate</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1245478967834710017)** (6 messages): 

- **QwenCode model issues resolved**: A member resolved an issue by downloading the model locally and editing the tokenizer to **Qwen2Tokenizer**, noting that "*everything just works*". However, they highlighted a problem with **QwenCode's model upload** and are still waiting for a response from the Qwen team.

- **Sanity check on axolotl configs**: A detailed configuration for quantizing models and setting CUDA options was shared, specifying **8-bit quantization** and various precision settings. Another member confirmed that the explanations about model weights, dtype settings, and AMP support were correct.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1245465600734658610)** (7 messages): 

- **Gradio trumps Streamlit for intuitiveness**: A member shared that they found **Gradio far more intuitive** than Streamlit, especially when working on demos (*"I just went with Gradio"*).
- **OAuth security concerns addressed**: A user raised a concern about the security of `gr.OAuthProfile`, but it was clarified with, “*OAuth doesn’t suffer from that vulnerability*” and that OAuth is a more secure option compared to just adding user data in headers. Detailed usage and sharing techniques are documented in the [Gradio Guide](https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers).
- **Gradio vs. Streamlit detailed comparison**: According to a member, Gradio tracks dependencies finely and does not re-render everything like Streamlit does. Gradio also works in various Python environments, including Jupyter notebooks, and offers backend features like a queueing system.

**Link mentioned**: <a href="https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers">Sharing Your App</a>: A Step-by-Step Gradio Tutorial

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1245775449376817222)** (86 messages🔥🔥): 

- **Modal Takes Multi-GPU Training Seriously**: Modal's team has been actively working on hardening multi-GPU setups, but still faces occasional issues due to a secure hypervisor stricter than expected. Users are encouraged to follow up on progress via the dedicated Slack thread [here](https://modallabscommunity.slack.com/archives/C069RAH7X4M/p1716911367749919).

- **Cold Boot Bottleneck For GPU Inference**: Truly "cold" starts for LLM or Stable Diffusion inference will take a few seconds due to the necessity to transfer weights from disk to GPU VRAM. Detailed solutions and optimizations for mitigating these latency issues are discussed [here](https://modal.com/docs/guide/cold-start#cold-start-performance).

- **Efficient Handling of Model Weights**: Best practices for managing large model weights are crucial for optimizing startup times in ML applications. Modal offers strategies like storing weights in container images at build time or using distributed file systems, as discussed [here](https://modal.com/docs/guide/model-weights).

- **Seamless Integration with Local Services**: Modal allows local Python code to interact with any service running on localhost. Details on deploying services and connecting them from other applications using Modal credentials are available [here](https://modal.com/docs/guide/trigger-deployed-functions).

- **Distributed Objects for Data Management**: Modal provides distributed dicts and queues for efficient interaction and data transfer across components of a distributed system. Learn more about how these objects work and their best usage practices [here](https://modal.com/docs/guide/dicts-and-queues).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/learning-learnding-funny-the-simpsons-dumb-gif-5270072">Education Is Key GIF - Learning Learnding Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/modal-labs/awesome-modal">GitHub - modal-labs/awesome-modal: A curated list of amazingly awesome Modal applications, demos, and shiny things. Inspired by awesome-php.</a>: A curated list of amazingly awesome Modal applications, demos, and shiny things. Inspired by awesome-php. - modal-labs/awesome-modal</li><li><a href="https://modal.com/docs/guide/dicts-and-queues">Dicts and queues</a>: Modal provides a variety of distributed objects to enable seamless interactivity and data transfer across different components of a distributed system. Two key objects are dicts and queues, both of wh...</li><li><a href="https://modal.com/docs/guide/trigger-deployed-functions">Invoking deployed functions</a>: Modal lets you take a function created by a deployment and call it from other contexts.</li><li><a href="https://modal.com/docs/examples">Featured examples</a>: How to run LLMs, Stable Diffusion, data-intensive processing, computer vision, audio transcription, and other tasks on Modal.</li><li><a href="https://modal.com/docs/guide/model-weights">Storing model weights on Modal</a>: Efficiently managing the weights of large models is crucial for optimizing the build times and startup latency of ML and AI applications. This page discusses best practices for handling model weights ...</li><li><a href="https://modallabscommunity.slack.com/archives/C069RAH7X4M/p1716911367749919">Slack</a>: no description found</li><li><a href="https://modal.com/docs/guide/cold-start#cold-start-performance">Cold start performance</a>: Modal Functions are run in containers.</li><li><a href="https://ai-infrastructure.org/the-state-of-ai-infrastructure-at-scale-2024/">The State of AI Infrastructure at Scale 2024</a>: How are fortune 1000 companies handling the growing demands of AI on their infrastructure? Can they move fast enough to deploy Gen AI but at the same time keep that AI on a tight leash to deliver fant...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1245828732384448623)** (59 messages🔥🔥): 

- **LangChain framework overview**: A user clarified the distinctions between various tools like LangChain, LangSmith, and LangServe by sharing a [LangChain introduction page](https://python.langchain.com/v0.2/docs/introduction/) explaining the toolkit and its components. LangChain provides development and deployment tools, while LangSmith offers inspection and optimization capabilities.
- **LangFlow and LangGraph confusion**: A conversation surfaced about LangFlow not being mentioned on a LangChain diagram, with clarifications that LangFlow uses the LangChain framework but is unrelated in its stack and purpose.
- **Resources for deeper understanding**: Users shared several links to enhance comprehension and practical application, including an [O'Reilly post on building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/), a [GitHub repository for generative UIs from Next.js](https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md), and a [GitHub series on LangChain LangGraph](https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg).
- **LangServe and LangGraph commentary**: Users expressed particular admiration for LangServe and its capabilities, while another user praised the concept of LangGraph workflows, calling them "genius." There were also discussions about LangSmith's possible integration with European servers for compliance.
- **Community engagement and project experiences**: Users shared personal experiences working with LangChain, emphasizing its utility in building internal applications and acknowledging its flexibility in high-level and detailed implementations. Some humor and lighthearted comments were made about the complexities and breadth of knowledge required to master these tools.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/introduction/">Introduction | 🦜️🔗 LangChain</a>: LangChain is a framework for developing applications powered by large language models (LLMs).</li><li><a href="https://www.answer.website/">answers, how they should be displayed.</a>: anwser engine built by developers digest</li><li><a href="https://reflex.dev/">Reflex · Web apps in Pure Python</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md">langchain-nextjs-template/app/generative_ui/README.md at main · langchain-ai/langchain-nextjs-template</a>: LangChain + Next.js starter template. Contribute to langchain-ai/langchain-nextjs-template development by creating an account on GitHub.</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>: no description found</li><li><a href="https://tenor.com/view/woodstock-happy50th-anniversary-happy-gif-26217300">Woodstock Happy50th GIF - Woodstock Happy50th Anniversary - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/wandb/openui">GitHub - wandb/openui: OpenUI let&#39;s you describe UI using your imagination, then see it rendered live.</a>: OpenUI let&#39;s you describe UI using your imagination, then see it rendered live. - wandb/openui
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/)** (1 messages): 

imaurer: Simon's newsletter is a great resource:
https://simonwillison.net/about/#subscribe
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1245462893164494878)** (93 messages🔥🔥): 

```html
- **Quarto for Inspect site**: Members discussed the use of **Quarto** for the [Inspect AI site](https://ukgovernmentbeis.github.io/inspect_ai/), with some expressing strong approval, "Quarto is the best."
- **Logs as a unit of reproducibility**: The use of logs as a unit of reproducibility in Inspect AI received praise from several members. One said, "This feels ahead of its time (in a really good way) 👀."
- **Links and resources for Inspect AI**: Multiple important links were shared, including the [Inspect homepage](https://ukgovernmentbeis.github.io/inspect_ai/), [AI Safety Institute](https://www.aisi.gov.uk/), and the [Inspect LLM workshop repository](https://github.com/jjallaire/inspect-llm-workshop).
- **Concerns and feedback on Inspect AI**: Attendees discussed various aspects and suggestions for Inspect AI, including the feature to compare runs in the UI and ideas for future enhancements. "Solvers is amazing," one member remarked, highlighting the tool's flexibility and composability.
- **Recording issues resolved**: There were initial issues with accessing video recordings of sessions, but these were subsequently addressed. "JJ's recording now works for me," a member confirmed after the fixes.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ianww.com/llm-tools">LLM eval tools spreadsheet</a>: Spreadsheet of 50+ LLM evaluation tools for testing models and improving prompts.</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai/">Inspect</a>: Open-source framework for large language model evaluations</li><li><a href="https://tenor.com/view/frustrated-waaaaaaaa-wwe-angry-mad-gif-13112986">Frustrated Waaaaaaaa GIF - Frustrated Waaaaaaaa WWE - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/pokemon-pikachu-clap-clapping-clapping-gif-gif-13465728489229726846">Pokemon Pikachu GIF - Pokemon Pikachu Clap - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lets-go-lets-go-marvel-let%27s-go-thor-let%27s-go-lets-go-thor-gif-6938549561677021369">Lets Go Lets Go Marvel GIF - Lets go Lets go marvel Let&#039;s go thor - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/jjallaire/inspect-llm-workshop">GitHub - jjallaire/inspect-llm-workshop</a>: Contribute to jjallaire/inspect-llm-workshop development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/yes-gif-22712908">Yes GIF - Yes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai">Inspect</a>: Open-source framework for large language model evaluations</li><li><a href="https://github.com/ukgovernmentbeis/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: A framework for large language model evaluations</a>: Inspect: A framework for large language model evaluations - UKGovernmentBEIS/inspect_ai</li><li><a href="https://github.com/UKGovernmentBEIS/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: A framework for large language model evaluations</a>: Inspect: A framework for large language model evaluations - UKGovernmentBEIS/inspect_ai</li><li><a href="https://www.aisi.gov.uk/">The AI Safety Institute (AISI)</a>: The AI Safety Institute is a directorate of the Department of Science, Innovation, and Technology that facilitates rigorous research to enable advanced AI governance.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1245452585813409914)** (42 messages🔥): 

- **Question about credit duration**: A member asked if the credits disappear once the course ends. This indicates concern about access continuity.

- **Issues with form submission**: Some users, including voberoi, reported missing or blank answers in their forms after question updates. Danbecker reassured that underlying data was not lost and submissions should still be valid, despite the issues.

- **OpenAI account confusion**: A user clarified whether the OpenAI account required for course activities is the same as the one used for logging into chatgpt, and platform.openai.com was recommended for account login.

- **Predibase signup error**: There was an issue where users couldn't sign up with Hotmail addresses as Predibase displayed an error message incorrectly stating it was for Gmail accounts. The platform typically restricts accounts from certain consumer domains.

- **Credit form deadlines and processing times**: The deadline for credit form submissions was reiterated as midnight, with different platforms like HuggingFace and Modal having specific review times for credit grants. Modal credits processing faced slight delays due to live session commitments.

**Link mentioned**: <a href="https://x.com/hamelhusain/status/1795871985265946934?s=12">Tweet from Hamel Husain (@HamelHusain)</a>: The $3,500 in compute credits end TODAY.  We won&#39;t be able to give them out after 11:59 PM PST 5/29/2024  Quoting Eugene Yan (@eugeneyan)   PSA: Signups for LLM-conf + finetuning workshop close to...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/1245489299550507132)** (5 messages): 

- **Albuquerque Meetup Plans**:
    - A person mentioned they are in the northwest side of Mexico but visit friends often in Phoenix. They are on the lookout for cool events or meetups in the area.

- **LA Welcomes West-Coasters**:
    - A friendly hello from someone based in Los Angeles.

- **SLO Lunchtime Invitation**:
    - An individual from San Luis Obispo (SLO) invited people driving from SF to LA along the 101 to stop by SLO for lunch. They highlighted the area's excellent restaurants, local breweries, and wine tasting opportunities.

- **SF Gathering for LLM Enthusiasts**:
    - A member reposted details about a gathering for 50 or so folks at their co-op in the Mission, SF, to discuss LLM evals. Attendees are asked to DM them with a non-anonymous social account for an invite.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1245452373472706652)** (16 messages🔥): 

- **Lots of East Coast Chatter**: Members are sharing their locations across the East Coast, ranging from Maryland and Virginia to New Jersey and Canada. There's a vibe of potential meetups, with comments like "Let's meet halfway" and "I was hiking there just the other week!"

- **Excitement About AI Tinkerers Event in NYC**: An [AI Tinkerers event](https://nyc.aitinkerers.org/p/live-from-civic-hall-nyc-tech-week-24-meetup) in NYC is generating buzz. One member says, "I'll be going to the AI Tinkerers event on Monday... HMU," and others are expressing interest and registering.

- **Warm DC Meetup Potential**: Multiple members from the DC area are expressing interest in a local meetup. Comments like "Sounds like we should definitely do a DMV meetup sometime" and "Seems we need to do a meetup in DC" indicate plans are being considered.

**Link mentioned**: <a href="https://nyc.aitinkerers.org/p/live-from-civic-hall-nyc-tech-week-24-meetup">
Live from Civic Hall! AI Tinkerers Meetup | NY#TechWeek  [AI Tinkerers - New York City]
</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1245452950361604158)** (27 messages🔥): 

- **Guten Tag from Europe**: Members from across Europe, including France, Germany, Finland, Spain, the Netherlands, and Austria, introduced themselves and exchanged greetings. Notably, people commented on past experiences living in each other's countries.
- **London Meetup Mania**: Enthusiasm for organizing a London meetup was evident as several members from the UK, including some traveling from Bristol, expressed interest in gathering on June 5th and 6th. Coordination for availability and details is ongoing.
- **Paris Plans**: One member inquired about others in Paris, leading to a response indicating potential future availability in a few weeks.
- **Summer in Turku**: A Finnish member discussed enjoying the summer weather of +25-27°C in Turku, albeit conflicted due to the engaging course and Discord activities coinciding with such beautiful weather.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1245461244379402281)** (4 messages): 

- **Keep Notifications On for Announcements**: Members are advised to keep notifications on for the new **announcements** channel. This channel is important for any critical updates and reminders, as shared by the admin.
  
- **Fill Out Forms for Credits**: Multiple reminders were given to fill out forms for vendor credits before the deadlines. Specific links were shared for credits from vendors like [OpenAI](https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f), [Hugging Face](https://docs.google.com/forms/d/...), [Modal](https://docs.google.com/forms/d/...), and [Fireworks](https://docs.google.com/forms/d/...).

- **Event Details in Events Category**: Event schedules and Zoom stream URLs will be posted in the "Events" category. This section also shows the time remaining relative to individual time zones to avoid confusion about event timings.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">no title found</a>: no description found</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSc7U01uRlMd2jeeeLZtaePTul-xBZXBwRx3x8qD2iIpuqE_mg/viewform">Hugging Face Credit Request</a>: Before we can apply 🤗 HF credit for you to use our paid services at https://huggingface.co, we’ll need just a few quick things!   Drop us a line if you have any questions at website@huggingface.co.  ...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfoCoXNhUjka09mu8rmgB1YM9s3529-F2oJdP5HkHT1SGfV2Q/viewform">Modal hackathon credits</a>: To claim your Modal credits, sign up for an account at https://modal.com/ first.  Then, let us know your username through this form.   For support, join the Modal Slack.  Here’s some examples to get s...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfndr0-zZlCEMCLVp99yI7olJg2qKr8iv4e_6CXkkb_Nhyj-Q/viewform">Fireworks Credits - Mastering LLMs : A Conference For Developers &amp; Data Scientists</a>: Please fill the below form to get $250 Fireworks credits! Join our discord for questions/help or more credits ;) https://discord.gg/fireworks
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/)** (1 messages): 

abhay_m: 👋
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1245459520663060532)** (3 messages): 

- **Free ChatGPT users get upgrades**: All ChatGPT Free users can now access **browsing, vision, data analysis, file uploads, and GPTs**. A significant enhancement of features aimed at expanding user capabilities.

- **OpenAI launches nonprofit initiative**: OpenAI introduced a new initiative, **OpenAI for Nonprofits**, to make their tools more accessible to nonprofit organizations. Further details can be accessed [here](https://openai.com/index/introducing-openai-for-nonprofits/).

- **Combatting deceptive uses of AI**: OpenAI discusses efforts to disrupt covert influence operations that use AI deceptively. Read more about the strategies and actions being taken [here](https://openai.com/index/disrupting-deceptive-uses-of-AI-by-covert-influence-operations/).
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1245459824368680960)** (375 messages🔥🔥): 

```html
<ul>
    <li><strong>Clarifications on GPT-4o Availability:</strong> Multiple members asked about GPT-4o availability for free users. It was explained that free users cannot force access and would be automatically switched between GPT-3.5 and GPT-4o based on the system's discretion.</li>
    <li><strong>Concern Over Subscription Value:</strong> A user expressed confusion over continuing to pay for ChatGPT. Responses highlighted advantages like early access to new features, quotas, and additional functionalities exclusive to subscribers.</li>
    <li><strong>Discussion on AI's Analytical Capabilities:</strong> Users debated how well different AI models handle logical reasoning tasks, like the "apples test" and the "susan test." It was noted that AI models often exhibit biases based on training data.</li>
    <li><strong>Code and Model Usage Insights:</strong> Members discussed using various AI models for coding assistance, comparing the performance of tools like GPT-4o, Mistral’s codestral, and Copilot. Speed and accuracy were highlighted as key factors in choosing specific models.</li>
    <li><strong>News and Media Detection AI Idea:</strong> A user discussed an AI concept for detecting fake news and propaganda by assessing posts on social media. Another user suggested it might run into common issues like hallucination and bias in AI's interpretation.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/they-did-that-on-the-simpsons-professor-chaos-butters-south-park-it-was-on-th">no title found</a>: no description found</li><li><a href="https://tenor.com/view/they-did-that-on-the-simpsons-professor-chaos-butters-south-park-it-was-on-the-simpsons-gif-22242623">They Did That On The Simpsons Professor Chaos GIF - They Did That On The Simpsons Professor Chaos Butters - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codestral.mistral.ai`">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/quickstart.ipynb">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1245460314313457787)** (64 messages🔥🔥): 

- **GPT-4 Outputting Word Salad**: Multiple users reported issues with **GPT-4 and GPT-4o** where long responses devolve into *"word salad"* filled with buzzwords. One reported a detailed example involving text conversion to Pinyin where the output turned nonsensical after initial coherent text.
- **Free Access to GPT Store with Limits**: Free users can currently access and browse the **GPT Store** but cannot run GPTs beyond using **GPT-3.5**. A member clarified, quoting a banner note, "GPTs will be coming to free users over the next few weeks. Stay tuned!"
- **Benefits and Limitations of Custom GPTs**: Users discussed the advantages of creating custom GPTs, such as defining specific roles and abilities, and confirmed that **Plus subscribers only** can create GPTs. Memory features are not yet available in custom GPTs but may roll out in the future.
- **API and Usage Issues**: Discussions included frustration over API access and model usage differences, with a specific mention that some users are confused between the Chat and Completions API. Members also shared tips on **protecting API keys** by using proxy backend servers.
- **Programming with GPT and Stability Problems**: A user is experiencing lag and slow progress when using GPT for extended contextual problems in the browser. They considered switching to the API for better stability, as suggested by another user who mentioned that the API typically handles extended sessions better.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1245460184651005982)** (2 messages): 

- **Builder biased towards own creation**: A member humorously mentioned being biased towards a tool they built. They acknowledged the possibility of bias in their favorable evaluation of the tool. 
- **Problem fixing advice provided**: Another member inquired about fixing a specific issue while suggesting the correct version to use. They advised, "Any version lower than 4, I think you must split to 2 separate requests."
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1245460184651005982)** (2 messages): 

- **Discussion on splitting requests in lower versions**: A member inquired if a specific issue had been fixed and clarified that for any version lower than 4, requests must be split into two separate ones. This indicates ongoing troubleshooting and support for API version compatibility.
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1245830528599592970)** (5 messages): 

- **Everything-AI supports llama.cpp and Qdrant**: *Now you can chat with your PDFs* through [everything-ai](https://github.com/AstraBert/everything-ai) by a community member. 
- **Mistral Model Gets Quantized**: *Codestral-22B-v0.1-GGUF*, a [quantized version](https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF) of Mistral's model, is now available thanks to a community contributor.
- **Nvidia's Embedding Model Demo Released**: Check out the new [Nvidia-Embed-V1](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1) demo by another member.
- **New Image Gen Pro Tool**: A community member launched [Image Gen Pro](https://huggingface.co/spaces/KingNish/Image-Gen-Pro) on HuggingFace.
- **DuckDB Integrates HuggingFace Datasets**: DuckDB has added an `hf://` path to over 150,000 datasets, making integration [easier than ever](https://blog.getwren.ai/how-to-load-huggingface-datasets-into-duckdb-and-query-with-gpt-4o-c2db89519e4d).

**Link mentioned**: <a href="https://huggingface.co/chat/assistant/66562fe0abb44809b7f77897)">HuggingChat</a>: Making the community's best AI chat models available to everyone.

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1245452119314792518)** (362 messages🔥🔥): 

- **Undo Commit on HuggingFace without Git:** A user accidentally committed to the main repository on HuggingFace and asked how to undo it. The discussion pivoted to using the HuggingFace CLI/library instead of Git, with the final resolution being to redo the commit.
- **Training Issues and Learning Rate Discussions:** There were extensive discussions on retraining models due to a need to change the learning rate. Suggestions included using values like 1e-3 or 1e-4 to avoid catastrophic forgetting, with specific mentions of models like TinyLlama 1.1B.
- **Audio Processing and Pitch Detection:** Users explored the complexity of analyzing audio files for tone, pitch, and intonation, with references to mathematical solutions and tools like [CREPE Pitch Tracker](https://pypi.org/project/crepe/).
- **Model Merging Competition Announcement:** A public service announcement was made about a Model Merging competition at NeurIPS, inviting participants to sign up and compete for a prize of $8K. The competition details can be found [here](https://llm-merging.github.io/).
- **Fine-Tuning Mistral and Tokenization:** Users discussed the appropriate tokenization formats for fine-tuning models like Mistral and TinyLlama, with scripts and examples provided for pre-processing data into the desired prompt format.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pypi.org/project/crepe/">crepe</a>: CREPE pitch tracker</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.4">TinyLlama/TinyLlama-1.1B-Chat-v0.4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/tokenizer_config.json">tokenizer_config.json · TinyLlama/TinyLlama-1.1B-Chat-v1.0 at main</a>: no description found</li><li><a href="https://tenor.com/view/i-saw-w-gus-fring-gus-gustavo-deleted-gif-25440636">I Saw W Gus Fring GIF - I Saw W Gus Fring Gus - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/no-pixels-cant-see-ben-chang-community-ken-jeong-gif-17361588">No Pixels Cant See GIF - No Pixels Cant See Ben Chang - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=4KalMNIbRUM">VOXTA APARTMENT - DEMO GAMEPLAY</a>: The wait is over!  🎉 Step into the Voxta Apartment, where Anna, your interactive AI companion, welcomes you to enjoy her charming company and numerous activ...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://github.com/pytorch/xla/">GitHub - pytorch/xla: Enabling PyTorch on XLA Devices (e.g. Google TPU)</a>: Enabling PyTorch on XLA Devices (e.g. Google TPU). Contribute to pytorch/xla development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/ok-ok-and-okay-buddy-dont-care-didnt-ask-gif-25239605">Ok Ok And GIF - Ok Ok And Okay Buddy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/openai/whisper">GitHub - openai/whisper: Robust Speech Recognition via Large-Scale Weak Supervision</a>: Robust Speech Recognition via Large-Scale Weak Supervision - openai/whisper</li><li><a href="https://storage.googleapis.com/libtpu-releases/index.html">no title found</a>: no description found</li><li><a href="https://pytorch.org/xla/release/2.3/index.html#quickstart>">PyTorch on XLA Devices &mdash; PyTorch/XLA master documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

venatic007: ✋🏻
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1245453587425792011)** (12 messages🔥): 

- **GNNs Simplify Simulator State Embedding**: A member explained the use of **GNNs** for data in a graph structure, noting that relations between entities are encoded as edge attributes while features are stored as tensors for each entity. This approach *"merges all this in a similar way to simple 2d convolution layers to give you an embedding of simulator's state."*

- **Yuan2.0-M32 Impresses in Math Tasks**: The new model **Yuan2.0-M32** with 40B parameters and a new router architecture, outperforms **Llama 3 70B** in Math/ARC tasks. It was [introduced on X](https://x.com/osanseviero/status/1796082193044844590) and available on [Hugging Face](https://huggingface.co/IEITYuan/Yuan2-M32-hf) along with the [research paper](https://hf.co/papers/2405.17976).

- **Video on Backpropagation Algorithm**: A YouTube video titled ["The Most Important Algorithm in Machine Learning"](https://www.youtube.com/watch?v=SmZmBKc7Lrs) explains the significance of the backpropagation algorithm in powering the field of machine learning.

- **NeurIPS Model Merging Competition**: An announcement for a **Model Merging competition** at NeurIPS offers an $8K prize for effective model selection and merging. Full details and sign-up information are shared [on Twitter](https://x.com/LChoshen/status/1796256513519989102).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/osanseviero/status/1796082193044844590">Tweet from Omar Sanseviero (@osanseviero)</a>: Introducing Yuan2.0-M32  🔥MoE with 3.7B active params (out of 40B) 👀New router architecture 🚀Trained on 2T tokens 🏆Impressive metrics given the # of active params 🤯Better than Llama 3 70B in Math...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://www.youtube.com/watch?v=SmZmBKc7Lrs">The Most Important Algorithm in Machine Learning</a>: Shortform link: https://shortform.com/artemIn this video we will talk about backpropagation – an algorithm powering the entire field of machine learning and ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1245456344182554725)** (8 messages🔥): 

```html
- **Demo Nvidia's embedding model**: A member shared a demo for Nvidia's new embedding model and requested PRs for cool examples or improved functions. *"You can test it out here: [Nvidia Embed V1](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1/)."*
- **Llama 3 SOLAR recreation attempt**: A user attempted to recreate Upstage's old Solar models using Llama 3. They used datasets like **`llm-wizard/alpaca-gpt4-data`** and [shared the model on HuggingFace](https://huggingface.co/cookinai/Llama-3-SOLAR-v0.2).
- **Codestral-22B quantized version**: Shared a quantized version of Codestral-22B-v0.1, created using llama.cpp, beneficial for code-related tasks. *"More details in the [Blogpost](https://mistral.ai/news/codestral/)."*
- **DuckDB supports Hugging Face datasets on WrenAI**: Announcement about DuckDB supporting the `hf://` path, enabling easy loading and querying of Hugging Face datasets in WrenAI. Learn more [here](https://blog.getwren.ai/how-to-load-huggingface-datasets-into-duckdb-and-query-with-gpt-4o-c2db89519e4d).
- **LLMinator v1.0.3 releases new features**: LLMinator now supports websocket interaction, context-aware chatbots, model conversion, and customized LLM inference parameters. Check out the project on [GitHub](https://github.com/Aesthisia/LLMinator).
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cookinai/Llama-3-SOLAR-v0.2">cookinai/Llama-3-SOLAR-v0.2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF">QuantFactory/Codestral-22B-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1/">Tonic&#39;s NV-Embed - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://huggingface.co/blog">Hugging Face – Blog</a>: no description found</li><li><a href="https://huggingface.co/blog-explorers">blog-explorers (Blog-explorers)</a>: no description found</li><li><a href="https://github.com/Aesthisia/LLMinator">GitHub - Aesthisia/LLMinator: Gradio based tool to run opensource LLM models directly from Huggingface</a>: Gradio based tool to run opensource LLM models directly from Huggingface - Aesthisia/LLMinator
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1245595801598689310)** (3 messages): 

- **PPO without a reference model query sparks interest**: A user inquired about the possibility of using **PPO** (Proximal Policy Optimization) without a reference model, noting their tight deadline for an internship project. Another member suggested reviewing a paper on an alternative approach called **SimPO** that *"eliminates the need for a reference model"* and provided the [arXiv link for further reading](https://arxiv.org/abs/2405.14734).
- **New mathematical improvements in AI models excite members**: A linked [paper on Hugging Face](https://huggingface.co/papers/2405.17976) sparked excitement with its math improvements, especially in reinforcement learning. Another member shared a specific paper titled *"Direct Preference Optimization"* and pointed out a mention of **SimPO**, a more efficient alternative to traditional reference-model-based approaches.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14734">SimPO: Simple Preference Optimization with a Reference-Free Reward</a>: Direct Preference Optimization (DPO) is a widely used offline preference optimization algorithm that reparameterizes reward functions in reinforcement learning from human feedback (RLHF) to enhance si...</li><li><a href="https://huggingface.co/papers/2405.17976">Paper page - Yuan 2.0-M32: Mixture of Experts with Attention Router</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1245465671241044091)** (17 messages🔥): 

- **Image regression with ViT supported**: Users discussed that image regression tasks can be handled by `ViTForImageClassification` via Hugging Face with the `problem_type="regression"`. Instructions for dataset preparation for image columns using the Image feature can be found [here](https://huggingface.co/docs/datasets/v2.3.2/en/image_process).

- **Demo notebooks for fine-tuning available**: Niels Rogge shared demo notebooks for fine-tuning tasks, specifically mentioning it works for models like Idefics2 and PaliGemma. The notebooks are available [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma) and more resources will be shared soon via a video on YouTube.

- **Monocular depth estimation with DINOv2**: DINOv2 (a ViT model) is supported for monocular depth estimation tasks using a DPT head. Example implementation can be found on the [model page](https://huggingface.co/models?search=dpt%20dino).

- **Best practices for fine-tuning transformers**: AdamW optimizer with a cosine learning rate scheduler is recommended for fine-tuning transformer models. Tips include using the largest batch size fitting in the GPU and utilizing ConvNext, DINOv2, or SigLIP for better performance instead of ViT.

- **Model merging competition at NeurIPS**: A model merging competition has been announced for NeurIPS with sponsorship from Hugging Face and others. Details and sign-up information can be found in the announcement [tweet](https://x.com/LChoshen/status/1796256513519989102).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/datasets/v2.3.2/en/image_process">Process image data</a>: no description found</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/do">Do (Tran)</a>: no description found</li><li><a href="https://huggingface.co/models?search=dpt%20dino">Models - Hugging Face</a>: no description found</li><li><a href="https://x.com/NielsRogge/status/1795106366752723094.">Tweet from Niels Rogge (@NielsRogge)</a>: Turns out my Idefics2 notebook works just as well for PaliGemma fine-tuning :) find it here: https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma  For JSON use cases, a tiny VLM ...</li><li><a href="https://github.com/google-research/tuning_playbook?tab=readme-ov-file#choosing-the-batch-size">GitHub - google-research/tuning_playbook: A playbook for systematically maximizing the performance of deep learning models.</a>: A playbook for systematically maximizing the performance of deep learning models. - google-research/tuning_playbook</li><li><a href="https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford">Papers with Code - Stanford Cars Benchmark (Fine-Grained Image Classification)</a>: The current state-of-the-art on Stanford Cars is CMAL-Net. See a full comparison of 73 papers with code.</li><li><a href="https://arxiv.org/abs/2211.12879">Data Augmentation Vision Transformer for Fine-grained Image Classification</a>: Recently, the vision transformer (ViT) has made breakthroughs in image recognition. Its self-attention mechanism (MSA) can extract discriminative labeling information of different pixel blocks to impr...</li><li><a href="https://news.ycombinator.com/item?id=40505099">Llama 3-V: Matching GPT4-V with a 100x smaller model and 500 dollars | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1245475582993633310)** (9 messages🔥): 

- **Word-level timestamps in Whisper**: A user asked how to get word-level timestamps using the Whisper model and shared a [link to the documentation](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.return_token_timestamps). They referenced the paper *“Robust Speech Recognition via Large-Scale Weak Supervision”* and mentioned Arthur Zucker as a contributor.

- **Button conflicts on CUDA device map**: A user encountered an issue setting `device_map` to `'cuda'` and received an error message stating "mode accelerated already used." Another user shared their success with topic labeling using sentence transformers and LLMs despite not understanding the technical question.

- **Custom evaluation schedule**: A user inquired about setting a custom evaluation schedule at specific steps (25k, 50k, 100k, 200k) due to the cognitive pattern of training performance varying logarithmically with data.

- **Open-source models release**: A user excitedly shared that two fully open-source language models were released, linking to one called [K2](https://huggingface.co/LLM360/K2) and another to a new collection of models. K2 is highlighted for outperforming the Llama 2 70B model using 35% less compute.

- **NeurIPS model merging competition**: A user announced a model merging competition at NeurIPS with the official [announcement tweet](https://x.com/LChoshen/status/1796256513519989102). The competition invites participants to revolutionize model selection and merging with a prize of $8k.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/LLM360/K2">LLM360/K2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04">Neo-Models - a m-a-p Collection</a>: no description found</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.return_token_timestamps">Whisper</a>: no description found
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1245459761294606376)** (91 messages🔥🔥): 

```html
- **Codestral Model Release and Uses Discussed**: Released the **Codestral-22B-v0.1** model, which handles 80+ programming languages including Python, Java, and JavaScript. The model supports code instruction and Fill in the Middle (FIM) functionalities; [more details in the blogpost](https://mistral.ai/news/codestral/).
- **Concerns about Model Variants**: Members discussed the practicality of different quantization variants, with some noting that **_S variants** are generally too "smoothbrained" and not useful.
- **Code Models and Prompt Formats**: The recommended format for querying Codestral-22B-v0.1-GGUF was discussed, referencing [this GitHub link](https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF#prompt-format).
- **Loading Issues on Limited Hardware**: A user experienced long loading times on **LM Studio** due to low system specs, suggesting smaller models might work better.
- **Inquiring Business Contact Options**: A member inquired about direct business contact for a project, and was guided to email **team@lmstudio.ai** for further discussion.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral/">Codestral: Hello, World!</a>: Empowering developers and democratising coding with Mistral AI.</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF#prompt-format">bartowski/Codestral-22B-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Codestral-22B-v0.1">mistralai/Codestral-22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF">bartowski/Codestral-22B-v0.1-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1245456795195932732)** (62 messages🔥🔥): 

- **Model Context Length Woes**: Members humorously lamented over model context lengths, noting that models like the **llama** series cap at *4096* without modifications but can be extended with **RoPE** to about *16k* with adjustments. *"Size matters,"* one member joked.

- **AlchemistCoder-DS-6.7B Fine-Tuning Discussion**: A member shared a link to the **AlchemistCoder-DS-6.7B** model [on HuggingFace](https://huggingface.co/internlm/AlchemistCoder-DS-6.7B), which performs on par with the **Deepseek Coder 33B**. Instructions were given on converting this model to GGUF format and using it with llama.cpp for easier deployment.

- **Struggles with Minerva-350M Compatibility**: A member reported issues using **Minerva-350M** with LM Studio since its release, facing difficulties with generation and overall compatibility. Another member suggested ensuring these models work in base llama.cpp and opening a feature request if not.

- **Challenges in Efficient Role-Playing**: A new user found difficulty in achieving effective role-play with models such as **Blue-Orchid-2x7b-Q4_K_M.gguf** and sought guidance on proper prompts and settings, even sharing a detailed roleplay system prompt from Reddit. Members suggested testing different models and settings adjustments.

- **Model Recommendation for GoLang and Kubernetes**: A new user inquired about a model suited for GoLang and Kubernetes, to which they were recommended **Claude Haiku** for its efficiency and context-packing capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/internlm/AlchemistCoder-DS-6.7B">internlm/AlchemistCoder-DS-6.7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nakodanei/Blue-Orchid-2x7b_GGUF/tree/main">nakodanei/Blue-Orchid-2x7b_GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/failspy/Llama-3-8B-Instruct-MopeyMule">failspy/Llama-3-8B-Instruct-MopeyMule · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/models?sort=trending&search=uncensored">Models - Hugging Face</a>: no description found</li><li><a href="https://docs.google.com/document/d/1xrMwhrz4DIdwzY4gI3GIrxQ0phQjVNmu2RGKRnGnRAM/edit?usp=drivesdk">High Quality Story Writing Type First Person</a>: Main Google Doc for my Custom GPTs: https://docs.google.com/document/d/1Cbwy3HuNTCzCaMXscU6FrgqvgjA2TFzOw1ucLqtbCyU/edit?usp=drivesdk  EXTREMELY NSFW Version of System Prompt Text for High Quality Sto...</li><li><a href="https://huggingface.co/YorkieOH10/AlchemistCoder-DS-6.7B-Q8_0-GGUF">YorkieOH10/AlchemistCoder-DS-6.7B-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/YorkieOH10">YorkieOH10 (Yorkie)</a>: no description found</li><li><a href="https://huggingface.co/YorkieOH10/AlchemistCoder-DS-6.7B-Q4_K_M-GGUF">YorkieOH10/AlchemistCoder-DS-6.7B-Q4_K_M-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/)** (1 messages): 

cancerous1: thanks for the rocm/windows build 🍻  you doubled my real estate for models
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/)** (1 messages): 

tiltspinner: Thanks!
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1245501068448895086)** (2 messages): 

- **Whisper models unsupported in llama.cpp**: A user couldn't locate the model path *vonjack/whisper-large-v3-gguf/whisper-large-v3-q8_0.gguf*. Another user clarified that **Whisper models** are not supported in **llama.cpp** but are used in **whisper.cpp** instead.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1245698249772503071)** (207 messages🔥🔥): 

- **Crypto Mining for Burn-In Tests**: A user suggested that one way to test a new GPU's VRAM under load is through crypto mining with "VRAM heavy alt coins". *"these coins are designed to be ASIC resistant, covering power costs during the burn-in test"*, though specifics were avoided to not promote crypto.

- **NVIDIA's New GPU Specs Rumored**: The NVIDIA GeForce RTX 5090 is rumored to have a cut-down 448-bit bus interface and 28 GB GDDR7 memory. *"Looks like they are using cut-down dies to save the good ones for professional products."* [Source](https://wccftech.com/nvidia-geforce-rtx-5090-founders-edition-gpu-dual-slot-dual-fan-cooler/).

- **Discussions on High-RAM and GPU Inference**: Members discussed the challenges of CPU vs GPU inference, emphasizing **memory channels** over CPU count. *"Once you exceed the VRAM, it's slow, so it may be a lesser of two evils problem, than an ideal problem."*

- **Buying High-End Hardware for Inference**: Members debated the cost-effectiveness of hardware such as the dual socket EPYC and M-series Apple. One mentioned, *"8x 3090 becomes too logical at that price point, they go for $500 used here."*

- **LLMs and Hallucinations**: Members expressed distrust for AI-generated information and discussed techniques to mitigate hallucinations. One user noted, *"LLMs are very good at rewriting text they have been given but struggle with pulling information from their 'memory' without inaccuracies."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wccftech.com/nvidia-geforce-rtx-5090-cut-down-gb202-gpu-448-bit-bus-28-gb-gddr7-memory/">NVIDIA GeForce RTX 5090 To Feature Cut-Down GB202 GPU With 448-Bit Bus &amp; 28 GB GDDR7 Memory</a>: NVIDIA GeForce RTX 5090 graphics cards are rumored to have a cut-down 448-bit bus interface for up to 28 GB GDDR7 memory.</li><li><a href="http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5-plus.html">Orange Pi - Orangepi</a>: no description found</li><li><a href="https://sceniccitysummit.com/schedule/">Schedule - Scenic City Summit</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1245615615431934032)** (3 messages): 

- **Adding a secondary GPU for inference acceleration**: Users were discussing how to add a secondary GPU for inference. One user humorously described the process, explaining that as long as there's physical space and proper power supply, **LM Studio** will balance the load between both GPUs, but emphasized the need to manage settings like *tensor_split, CUDA_VISIBLE_DEVICES*, and *main_gpu*.
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1245537914725732467)** (2 messages): 

- **GitHub link to Amuse is broken**: A member noted that the GitHub link for Amuse no longer works, and they can't type in the prompt box.
- **Hugging Face hosting for Amuse**: Another member provided a working link to download the `Amuse_v1.3.0.zip` from [Hugging Face](https://huggingface.co/Stackyard-AI/Amuse/blob/main/Amuse_v1.3.0.zip) and confirmed it's functioning fine.

**Link mentioned**: <a href="https://huggingface.co/Stackyard-AI/Amuse/blob/main/Amuse_v1.3.0.zip">Amuse_v1.3.0.zip · Stackyard-AI/Amuse at main</a>: no description found

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1245469955491889192)** (1 messages): 

- **Codestral 22B welcomes coding enthusiasts**: **Mistral's latest coding model**, Codestral, is now available for download. It features a **22B parameter size**, making it an appealing option for those with high-capacity GPUs seeking powerful models. [Download Codestral-22B here](https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF).
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1245456743295881356)** (233 messages🔥🔥): 

- **Llama3 dominates over Phi3 in testing**: Members agree that Llama3 outperforms Phi3, with one stating "*Llama-3-8b is much better*" and another calling Phi3 "*extremely synthetic*". The consensus is to avoid Phi3 models, especially the mini version, in favor of base Llama3.

- **Fine-tuning advice for creating role-playing characters**: A suggested workflow includes training Llama3 base models before finetuning for instruction following. One user shared his experience of fine-tuning Llama3 to act like Paul Graham but found the results lacking compared to adding a prompt like "*Pretend you are Paul Graham*".

- **Debate over instruction models for fine-tuning**: It is generally recommended not to fine-tune on top of instruction models as it can create loss rather than add value to the model. Finetuning on base models and using DPO is considered better for specific roles like Jesus, Trump, or other characters.

- **Dynamic conversation on using anti-prompts**: Members discussed the utility of anti-prompts for better control over conversation flow in chat models. Anti-prompts can stop generation at user-defined words, allowing users to input their text before continuing the model's output.

- **New models and optimization queried**: There is excitement around newly announced models like Yuan but skepticism remains, emphasizing the importance of real-world application over benchmarks. Users shared experiences and challenges with various tools and platforms for model fine-tuning and inference, and expressed longing for multi-GPU support from Unsloth.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14734">SimPO: Simple Preference Optimization with a Reference-Free Reward</a>: Direct Preference Optimization (DPO) is a widely used offline preference optimization algorithm that reparameterizes reward functions in reinforcement learning from human feedback (RLHF) to enhance si...</li><li><a href="https://huggingface.co/REILX/Phi-3-medium-128k-code-instruct">REILX/Phi-3-medium-128k-code-instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/gpuopenanalytics/pynvml/issues/53">nvmlDeviceGetName throws UnicodeDecodeError invalid start byte · Issue #53 · gpuopenanalytics/pynvml</a>: Running the following code on WSL2 throws the error mentioned in the title: from pynvml import * handle = nvmlDeviceGetHandleByIndex(0) print(nvmlDeviceGetName(handle)) Stacktrace: File &quot;&lt;stdi...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1245557277654777856)** (6 messages): 

- **Webinar on biases in AI Training**: Join a webinar with Tom Hosking exploring how human feedback in AI training can be subjective and biased, under-representing crucial aspects like factuality. Watch [the webinar here](https://eu1.hubs.ly/H09npRg0), and read the research paper on [Arxiv](https://arxiv.org/abs/2309.16349).

- **CoPE: Contextual Positional Encoding**: A new positional encoding method for transformers from FAIR, Contextual Positional Encoding (CoPE), factors in context to improve function. This method is praised for its ability to count distances per head depending on various needs like sentences or paragraphs. [Check out the tweet](https://x.com/ylecun/status/1795985933998715217).

- **Personal Boundaries in Code Sharing**: One member expressed a reluctance to share code publicly, indicating a desire to keep it private.

- **Dog Appreciation**: A light-hearted message highlighted an appreciation for members posting pictures of their cute dogs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ylecun/status/1795985933998715217">Tweet from Yann LeCun (@ylecun)</a>: CoPE: Contextual Positional Encoding. A new paper from FAIR that @elonmusk could use to improve Grok.  Quoting Jason Weston (@jaseweston)   🚨 Contextual Position Encoding (CoPE) 🚨  Context matters! ...</li><li><a href="https://eu1.hubs.ly/H09npRg0">Tom Hosking &lt;&gt; Prolific Webinar | Navigating Biases in Human Feedback for AI Training</a>: Join us for an insightful webinar exploring the role of human feedback in evaluating and training Large Language Models (LLMs). Discover how preference scores, despite being the standard, may be subje...</li><li><a href="https://arxiv.org/abs/2309.16349">Human Feedback is not Gold Standard</a>: Human feedback has become the de facto standard for evaluating the performance of Large Language Models, and is increasingly being used as a training objective. However, it is not clear which properti...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1245451658369044572)** (130 messages🔥🔥): 

- **Unsloth supports native finetuning and provides a Colab notebook**: A member mentioned that Unsloth supports native finetuning, sharing a [GitHub link](https://github.com/unslothai/unsloth#-finetune-for-free) to the Colab notebook for continuous pretraining.
- **Command-R Model Preferences and EOS Token Discussion**: A member stated that Command-R is the best model, which was countered by another user mentioning the necessity of the EOS_TOKEN in training data for models to know when text completion is done. They provided a [YouTube video](https://www.youtube.com/watch?v=T1ps611iG1A) as an example.
- **Apple M3 GPU incompatibility with CUDA**: A user encountered a "Torch not compiled with CUDA enabled" error on a Mac with an Apple M3 Pro GPU. It was explained that Apple's GPUs do not support CUDA, suggesting the use of Google Colab instead.
- **Llama3-8b model performance discussions**: Users discussed running Llama3-8b on different hardware setups, such as Beelink Ser5 MAX Mini PC with 16GB and considerations for upgrading to 32GB or 64GB RAM. Discussions emphasized that larger RAM allows for running bigger models with less quantization.
- **Fine-tuning issues and dataset size concerns**: A user reported issues during fine-tuning Llama3, resulting in "garbage results," and wondered if a larger dataset was needed. Another member responded that training with specific data might lead the model to favor that data over previous instructions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.eraser.io/diagramgpt">DiagramGPT – AI diagram generator</a>: Generate technical diagrams in seconds from plain English or code snippet prompts. Diagrams include sequence diagrams, flow charts, entity relationship diagrams, cloud architecture diagrams, data flow...</li><li><a href="https://huggingface.co/chat/assistant/65e71408a6654bcc68624d8d">Diagrams Creator - HuggingChat</a>: Use the Diagrams Creator assistant inside of HuggingChat</li><li><a href="https://www.youtube.com/watch?v=T1ps611iG1A">How I Fine-Tuned Llama 3 for My Newsletters: A Complete Guide</a>: In today&#39;s video, I&#39;m sharing how I&#39;ve utilized my newsletters to fine-tune the Llama 3 model for better drafting future content using an innovative open-sou...</li><li><a href="https://diagrams.mingrammer.com/docs/getting-started/examples">Examples · Diagrams</a>: Here are some more examples.</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1245457683814027386)** (351 messages🔥🔥): 

- **Discussion on Kohya SS and Model Training with Budget Constraints**: Members discussed options for **training Stable Diffusion** models without incurring high costs, highlighting tools like **Google Colab** and services like **RunDiffusion**. One member shared a [GitHub link](https://github.com/hollowstrawberry/kohya-colab) to Kohya Colab for accessible training.

- **ControlNet and Inference Optimization**: A detailed conversation touched on the use of ControlNet and various samplers for **improving image generation accuracy**. A GitHub link was shared for a **dynamic LoRA control extension**: [sd-webui-loractl](https://github.com/cheald/sd-webui-loractl).

- **New Ruby SDK for Stability AI**: A member announced the launch of an open-source **Ruby SDK for Stability AI API** for image generation, supporting core and SD3 models. They provided a [GitHub link](https://github.com/OlympiaAI/stability) for community access and contribution.

- **Upcoming Models and Community Sentiment**: The community speculated on the release date and features of **Stable Diffusion 3 (SD3)**, expressing both skepticism and hope. Discussions included potential licensing challenges and financial backing comparisons with competitors like Midjourney, which has significantly higher revenues.

- **Teaching Stable Diffusion to Kids**: A teacher sought advice on **introducing Stable Diffusion to children** without generating explicit content. Suggestions included using **ControlNet** to convert kids' drawings into realistic images, making the tech engaging and educational.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/deepseek-ai/DeepSeek-VL-7B">Chat with DeepSeek VL 7B - a Hugging Face Space by deepseek-ai</a>: no description found</li><li><a href="https://drive.google.com/file/d/1IBgfLqReWwhhWNXvnSCJH1gtQscgWPTV/view?usp=sharing">stable difusion web ui in sanoma three archives .zip</a>: no description found</li><li><a href="https://imgur.com/rYfd6lA">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://github.com/cheald/sd-webui-loractl/issues/30">Feauture request: ComfyUI Implementation? · Issue #30 · cheald/sd-webui-loractl</a>: Hey there! Can you by any chance implement the logic into a ComfyUI node?</li><li><a href="https://imgur.com/aMqdy4z">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://github.com/cheald/sd-webui-loractl">GitHub - cheald/sd-webui-loractl: An Automatic1111 extension for dynamically controlling the weights of LoRAs during image generation</a>: An Automatic1111 extension for dynamically controlling the weights of LoRAs during image generation - cheald/sd-webui-loractl</li><li><a href="https://github.com/hollowstrawberry/kohya-colab">GitHub - hollowstrawberry/kohya-colab: Accessible Google Colab notebooks for Stable Diffusion Lora training, based on the work of kohya-ss and Linaqruf</a>: Accessible Google Colab notebooks for Stable Diffusion Lora training, based on the work of kohya-ss and Linaqruf - hollowstrawberry/kohya-colab</li><li><a href="https://github.com/PixArt-alpha/PixArt-sigma">GitHub - PixArt-alpha/PixArt-sigma: PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation</a>: PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation - PixArt-alpha/PixArt-sigma</li><li><a href="https://github.com/OlympiaAI/stability">GitHub - OlympiaAI/stability: Ruby SDK for Stability AI API</a>: Ruby SDK for Stability AI API. Contribute to OlympiaAI/stability development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/35966/dpm-2m-alt-karras-sampler">DPM++ 2M alt Karras [ Sampler ] - Automatic v1.6.0 | Stable Diffusion Other | Civitai</a>: This is alternative version of DPM++ 2M Karras sampler. I don&#x27;t claim that this sampler ultimate or best, but I use it on a regular basis, cause I ...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d4cwi9/pcm_phased_consistency_model/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1245535144501317685)** (11 messages🔥): 

- **GPT-4-OMNI poised to transform education**: A member shared an article about [GPT-4-OMNI](https://laion.ai/notes/open-gpt-4-o/) and its potential impact on education, imagining it as a "personal learning assistant" that could revolutionize how we learn. The discussion highlighted the vision of a future within reach due to advancements in **multi-modal models**.

- **New alignment method paper submitted**: Another member announced they had submitted their new alignment method paper to Arxiv and looked forward to sharing it once approved. The submission sparked curiosity about the paper's contents and potential impact.

- **Luxia 21.4b model contamination concerns**: A significant contamination increase was noted between versions of the **Luxia 21.4b** model during GSM8k tests, with data showing a 29% rise from v1.0 to v1.2 as shared on [HuggingFace](https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1). This contamination was not observed in other evaluation metrics like ARC and Wino.

- **NeurIPS model merging competition**: An announcement was made for a model merging competition at NeurIPS, promising substantial rewards and opportunities for scientific contribution. The details and sign-up information were provided in a [tweet](https://x.com/LChoshen/status/1796256513519989102), along with a link to the competition's [official page](https://llm-merging.github.io/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://laion.ai/notes/open-gpt-4-o/">Call to Build Open Multi-Modal Models for Personal Assistants | LAION</a>: &lt;p&gt;Technologies like the recently introduced GPT-4-OMNI from OpenAI show again the potential which strong multi-modal models might have to positively transfo...</li><li><a href="https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1">saltlux/luxia-21.4b-alignment-v1.2 · contamination results v1.0 vs v1.2 on GSM8K</a>: no description found</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1245465301026345000)** (50 messages🔥): 

- **Constant learning rate vs. Cosine schedule**: An [arXiv paper](https://arxiv.org/abs/2405.18392) argues that constant learning rates with cooldowns scale predictably and reliably, similar to cosine schedules. Additionally, stochastic weight averaging is shown to improve performance without extra training costs.
- **Contextual Position Encoding (CoPE) introduced**: A tweet by [@jaseweston](https://x.com/jaseweston/status/1795978611784089799?s=61&t=ryK3X96D_TkGJtvu2rm0uw) discusses CoPE, a new positional encoding method for transformers that accounts for context. It can handle counting and copy tasks and shows better performance on language modeling and coding tasks.
- **Sonic: fast generative voice model released**: [Cartesia AI](https://x.com/cartesia_ai/status/1795856778456084596?s=46) announced the release of Sonic, a generative voice model with 135ms model latency, part of their mission to build real-time multimodal intelligence.
- **Gradient diversity affects mini-batch SGD performance**: An [arXiv paper](https://arxiv.org/abs/1706.05699) suggests that high similarity between gradients degrades mini-batch SGD performance. Gradient diversity is crucial for speedups while maintaining performance.
- **Model Merging competition at NeurIPS**: [NeurIPS 2023](https://x.com/LChoshen/status/1796256513519989102) will feature a model merging competition with up to $8,000 in prizes. The competition is sponsored by organizations including Hugging Face and Sakana AI Labs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.18392">Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations</a>: Scale has become a main ingredient in obtaining strong machine learning models. As a result, understanding a model&#39;s scaling properties is key to effectively designing both the right training setu...</li><li><a href="https://x.com/jaseweston/status/1795978611784089799?s=61&t=ryK3X96D_TkGJtvu2rm0uw">Tweet from Jason Weston (@jaseweston)</a>: 🚨 Contextual Position Encoding (CoPE) 🚨  Context matters!  CoPE is a new positional encoding method for transformers that takes into account *context*. - Can &#34;count&#34; distances per head depen...</li><li><a href="https://arxiv.org/abs/2405.16684">gzip Predicts Data-dependent Scaling Laws</a>: Past work has established scaling laws that predict the performance of a neural language model (LM) as a function of its parameter count and the number of tokens it&#39;s trained on, enabling optimal ...</li><li><a href="https://x.com/cartesia_ai/status/1795856778456084596?s=46">Tweet from Cartesia (@cartesia_ai)</a>: Today, we’re excited to release the first step in our mission to build real time multimodal intelligence for every device: Sonic, a blazing fast  (🚀 135ms model latency), lifelike generative voice mo...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://arxiv.org/abs/1706.05699">Gradient Diversity: a Key Ingredient for Scalable Distributed Learning</a>: It has been experimentally observed that distributed implementations of mini-batch stochastic gradient descent (SGD) algorithms exhibit speedup saturation and decaying generalization ability beyond a ...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1245451696646262854)** (191 messages🔥🔥): 

- **MLP-Mixer Struggles for Causality and Sequence Length**: Members discussed challenges in making MLP-mixers causal and effective across various sequence lengths. A member noted, *"Seems like many weird tricks are needed to make the mlp model work with any sequence length and causal."*
  
- **Transformers as Dynamic MLP-Mixers**: The conversation highlighted how transformers can be seen as context-dependent MLP-mixers. One member argued, *"Attention is basically an mlp mixer where the weights over the time dimension are dynamically generated,"* emphasizing the importance of context dependence.

- **Criticism and Alternatives to MLPs vs Transformers**: There was criticism on the practicality and superiority of MLPs over transformers. A user stated, *"MLP-Mixer would have been SOTA on a bunch of things not that long ago,"* while others pointed out the need for context-dependent operations for better scalability and adaptability.

- **Industry Preference for Transformers**: The dominance of transformers in the industry was reiterated, with a comparison to past trends. One member remarked, *"Industry always is. They were real sold on SVMs too until CNNs,"* indicating evolving preferences.

- **Exploring Alternatives and Integration in Diffusion Models**: Some members touched on the application of diffusion models in robotics and expressed interest in hybrid models. Gers101 mentioned, *"Diffusions been really big in robotics rn where they use diffusion to model actions space for imitation learning,"* reflecting on their versatile integration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine &#34;next-scale prediction&#34; or &#34;next-resolutio...</li><li><a href="http://arxiv.org/abs/2405.08553">Improving Transformers with Dynamically Composable Multi-Head Attention</a>: Multi-Head Attention (MHA) is a key component of Transformer. In MHA, attention heads work independently, causing problems such as low-rank bottleneck of attention score matrices and head redundancy. ...</li><li><a href="https://x.com/arankomatsuzaki/status/1503543031923945475">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Efficient Language Modeling with Sparse all-MLP  Sparse all-MLP improves LM PPL and obtains up to 2x improvement in training efficiency compared to Transformer-based MoEs as well as dense Transformers...</li><li><a href="https://arxiv.org/abs/1603.05691">Do Deep Convolutional Nets Really Need to be Deep and Convolutional?</a>: Yes, they do. This paper provides the first empirical demonstration that deep convolutional models really need to be both deep and convolutional, even when trained with methods such as distillation th...</li><li><a href="https://github.com/twistedcubic/attention-rank-collapse">GitHub - twistedcubic/attention-rank-collapse: [ICML 2021 Oral] We show pure attention suffers rank collapse, and how different mechanisms combat it.</a>: [ICML 2021 Oral] We show pure attention suffers rank collapse, and how different mechanisms combat it. - twistedcubic/attention-rank-collapse</li><li><a href="https://arxiv.org/abs/2108.13002#microsoft">A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP</a>: Convolutional neural networks (CNN) are the dominant deep neural network (DNN) architecture for computer vision. Recently, Transformer and multi-layer perceptron (MLP)-based models, such as Vision Tra...</li><li><a href="https://arxiv.org/abs/2306.13575">Scaling MLPs: A Tale of Inductive Bias</a>: In this work we revisit the most fundamental building block in deep learning, the multi-layer perceptron (MLP), and study the limits of its performance on vision tasks. Empirical insights into MLPs ar...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1245495179285102674)** (1 messages): 

- **Researchers find latent reasoning in language models**: A link to a [tweet by Jannik Brinkmann](https://x.com/BrinkmannJannik/status/1795827121585332459?t=UEx6PpAys4nmmLaEtyZSSQ&s=19) was shared where he discusses finding evidence of latent reasoning and search in language models. Their upcoming #acl2024 paper reverse-engineers a transformer trained on tree search, revealing human-understandable backward chaining circuits.

**Link mentioned**: <a href="https://x.com/BrinkmannJannik/status/1795827121585332459?t=UEx6PpAys4nmmLaEtyZSSQ&s=19">Tweet from Jannik Brinkmann (@BrinkmannJannik)</a>: Can we find evidence of latent reasoning and search in language models?  Our #acl2024 paper (w/ @abhayesian and @VictorLevoso) reverse-engineers the internal mechanisms of a transformer trained on tre...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1245453247129456772)** (19 messages🔥): 

- **Arxiv Paper Cited**: Someone shared an [Arxiv paper](https://arxiv.org/pdf/2403.08295) in response to a query about a specific research paper.
- **Pull Requests Discussed**: Members discussed two pull requests: [PR #2643](https://github.com/vllm-project/vllm/pull/2643) for adding `/get_tokenizer` to the API server for easier integration, and [PR #1794](https://github.com/EleutherAI/lm-evaluation-harness/pull/1794) for similar functionality in EleutherAI's repository. Another member mentioned PR #1196 related to Logits support, which was declined.
- **Machine Translation Evals PR**: A member shared a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900) for machine-translated ARC challenge evaluations in 11 languages, seeking review.
- **Token Evaluation Anomalies**: Discussion ensued about unexpectedly fast token evaluations in various datasets, with links to [additional explanations](https://arxiv.org/abs/2405.14782) provided. Issues with energy consumption measurements and token processing were clarified by highlighting differences in evaluation methodologies.
- **Device Support in LM_Eval**: A member inquired about expanding device support for LM_Eval beyond "cuda". They were informed that NPU support is being reviewed, with a suggestion to open issues for other device types to solicit community contributions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14782">Lessons from the Trenches on Reproducible Evaluation of Language Models</a>: Effective evaluation of language models remains an open challenge in NLP. Researchers and engineers face methodological issues such as the sensitivity of models to evaluation setup, difficulty of prop...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1900">add arc_challenge_mt by jonabur · Pull Request #1900 · EleutherAI/lm-evaluation-harness</a>: This PR adds tasks for machine-translated versions of arc challenge for 11 languages.  We will also be adding more languages in the future.</li><li><a href="https://github.com/vllm-project/vllm/pull/2643">Adding `/get_tokenizer` to api_server for lm-evaluation-harness ease integration.  by AguirreNicolas · Pull Request #2643 · vllm-project/vllm</a>: OpenAI is already supporting  logprobs for chat models in most model cases. In paralell, lm-evaluation-harness is in dev to add support to this feature. In order to run a evaluation of an endpoint ...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1794">Vllm get tokenizer by AguirreNicolas · Pull Request #1794 · EleutherAI/lm-evaluation-harness</a>: Issue: It is possible to change the model names served by vllm, and then have it not respond to any Huggingface repository making it impossible to obtain the tokenizer and therefore run lm-eval-har...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1245715839592955986)** (4 messages): 

- **Pythia Tokenizer Frequency Clarification**: A member asked if the token IDs from **Pythia's Hugging Face tokenizer** are ranked by frequency based on its training corpus. Another member clarified, *"Not in general, no,"* stating that extra tokens were added for specific contexts like code.
- **Token Frequencies to Be Provided**: The discussion concluded with a promise to provide token frequencies from **the Pile** later in the afternoon. There were no further details or links shared at this time.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1245689415075106866)** (3 messages): 

- **Questions about the ggml Library**: A member reached out to the community asking if anyone has used the **ggml library** for their projects, indicating they have some questions about it.
- **NVIDIA's Research Talk Insights**: Key takeaways from an NVIDIA research talk included a **4nm chip performing 96 int4 TOPs/Watt** and exploration into **2:8 sparsity**. A member linked to both the [research talk](https://youtu.be/gofI47kfD28?si=41UIMkpMCyb_qWqA) and a relevant [Physics of LLMs paper](https://arxiv.org/abs/2404.05405).
- **Meta's AI Hardware Advancements**: A discussion highlighted Meta's next-gen **Meta Training and Inference Accelerator (MTIA)**, featuring **72 accelerators per node and 354 TFLOPS/s (INT8) at 90W**. More details can be found on the [Meta AI blog](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1796253349932843214?s=46">Tweet from Daniel Han (@danielhanchen)</a>: My notes from a NVIDIA research talk:  1) NVIDIA has an research inference 4nm chip doing 96 int4 TOPs/Watt vs Blackwell&#39;s 20T/W  2) B200&#39;s float4 is exponent=2 and mantissa=2? Maybe I mishear...</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1245470054271946753)** (9 messages🔥): 

- **Int32 Multiplication Overflow in Triton**: A member discovered a bug in Triton code where multiplication happened in **int32** before being added to the base tensor pointer, leading to overflow and CUDA memory errors for large tensors. This issue was not detected in unit tests but surfaced during production due to large data sizes.

- **Grid Dimension Limitations in CUDA**: Another member shared their experience with **CUDA** where code crashed with actual data because the grid dimension in the **y and z directions** is 16-bit, causing issues with more than 65k blocks. This limitation did not appear in unit tests.

- **Passing Metadata in Triton Kernels**: There was a discussion about the lack of support for passing **tuples** or structured values into Triton kernels. The ability to pass metadata like shape and strides in one object was mentioned as a desirable feature to simplify code, especially for higher dimensions.

- **Triton.language.dot and Tensor Core Support**: A member questioned whether `tl.dot` supports **bf16 tensor cores** and pointed out that upcasting bf16 to fp32 is slow. They linked a [GitHub issue](https://github.com/triton-lang/triton/issues/2302) discussing a related bug with `out_type` when the output type is bfloat16.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.dot.html#triton.language.dot">triton.language.dot &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/triton-lang/triton/issues/2302">`out_type` in `tl.dot` is has a bug when `out_type` is `bfloat16` · Issue #2302 · triton-lang/triton</a>: When dtype is bfloat16, out_type in tl.dot is not working as I think. grad_query += tl.dot(grad_softmax, key, use_accelerator, dtype) In this case, there is compile errors. Compilation error messag...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1245580642574667877)** (6 messages): 

- **Inquiry on torch.profiler C++ API**: A member asked if there is a C++ API for `torch.profiler`, noting they can use `pybind` to call C++ functions from Python but were seeking a direct method to get traces from C++ itself.

- **Nightly torch.compile Slower**: A member noticed that `torch.compile` seems slower with recent nightly builds. Another member linked to a [GitHub pull request](https://github.com/pytorch/pytorch/pull/126320) that should address the issue.

- **torch.compile and Backward Pass Kernels**: It was confirmed that `torch.compile` generates kernels for both forward and backward passes. Users can identify these kernels by setting `TORCH_LOGS="output_code python your_code.py`.

- **Triton Kernel in Profiling**: A member questioned whether the "triton kernel" seen in torch profiling represents the aggregate of all Triton kernels or a specific one. They noted no entries were found in the `output_code` despite searching.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/pull/126320">Added memory budget to partitioner by Chillee · Pull Request #126320 · pytorch/pytorch</a>: Stack from ghstack (oldest at bottom):  #127520 -&gt; #126320 #127446

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1245517354679533610)** (11 messages🔥): 

- **No speed gain with torch.compile**: A user attempted to use `torch.compile` to speed up inference and training with Hugging Face's `AutoModelForCausalLM` but did not observe any improvement. They described issues with the model becoming corrupt during training and a general lack of speed gain.
  
- **Tips for using torch.compile**: Another member provided advice, suggesting specific configurations including setting `model.config.use_cache=True` and using `torch.compile` on `model.forward` for better results. They shared a helper script and noted potential issues with the latest `transformers` version, linking to their [GitHub repository](https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py).

- **Recommendation for vllm and batching inquiry**: The user acknowledged the helpful advice and noted they might try `vllm` for inference instead. They also inquired about efficient batching techniques for handling concurrent image processing using a single model, mentioning unsuccessful attempts with the multiprocessing library.

**Link mentioned**: <a href="https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py">hqq/hqq/utils/generation_hf.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1245648613519003658)** (4 messages): 

- **Confusion over FP4 Multiplication Specification**: A member expressed uncertainty about "whether multiplying 2 fp4 numbers" involves exact computation and then rounding or a lossy process. They speculated if converting P_1, P_2 to FP32, then multiplying and rounding might result in the same outcome.
- **Mixed-Precision Layers Exploration**: It was inquired if "mixed-precision layers" for activations and gradients had been considered, suggesting fp4/fp8 for forward passes and fp8/fp16 for backward passes. They noted that applying gradients to FP4 weights might not make sense in basic backpropagation algorithms.
- **Tagging Experts on Specific Issues**: It was suggested to ask technical questions on the AO issue tracker and tag `vkuzo` for better responses, as he doesn't check Discord often.
- **Precision in Dot Product Accumulation**: A member clarified that the precision of dot product accumulation is "implementation defined" according to the spec and "depends on MX-enabled hardware." They mentioned PyTorch supporting mixed precision layers for emulation, though specific hardware support for various precisions is still unknown.
- **FP6-LLM CUDA Kernel Details**: The FP6-LLM CUDA kernel was described as a "mixed-input matmul" for fp16 activations and MX fp6_e3m2 weights, with computations done in fp16 using tensor cores.
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1245771308265111564)** (1 messages): 

- **Whisper Model Speeds up by 6.3x**: A member announced impressive results in optimizing the Whisper model using **static cache**, **HQQ quantization**, **torchao 4-bit kernel**, and **torch.compile** with fullgraph. They teased a *"blogpost coming up tomorrow"* sharing these insights in detail.
  

---


### **CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1245771200072781924)** (3 messages): 

- **Reviving the Channel**: A user expressed interest in reviving the discussion in the channel after a period of inactivity.
- **MatMul Demo GIF**: A **work-in-progress** demo featuring a **matrix multiplication (matmul)** animation was shared, available [here](https://media.discordapp.net/attachments/1225431825929863320/1245745841143025826/full_tensor_multiplication.gif?ex=6659deb9&is=66588d39&hm=6f6bfb0b97e8a64415706dfe84c40cf3efe2a64c5e2e930e09ca478f0f1d15da&).
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1245483580789100604)** (122 messages🔥🔥): 

- **Using Cloudflare R2 to Replace Python Dependency**: A member suggested using Cloudflare R2 for pretok data storage to eliminate Python dependencies, noting its lower cost due to no egress fees compared to S3. They confirmed success in enabling R2 and will report back after testing further.

- **DNDEBUG Macro for Large Scale Production**: The DNDEBUG macro, which removes assert checks at compile time for large-scale production runs, was introduced. This could be useful for performance tuning in the context of kernel size checks in CUDA.

- **Templated Block Sizes for Performance Boost**: Members discussed templating variables such as blockDim.x to achieve significant speed improvements in CUDA kernels, especially in the fused classifier. This approach has shown measurable performance gains and could simplify the code compared to more complex templating methods. 

- **Storing Data on Internal S3 to Avoid Fees**: An internal S3 storage was suggested to avoid ancillary fees, with pre-uploaded resources such as the tokenizer and dataset files. This approach facilitates sharing large datasets for training without incurring additional costs.

- **Merge and Optimize Kernel Recomputations**: It was proposed to merge a code branch involving layernorm recomputations despite it not being the fastest possible version. The goal is to first integrate functional improvements and later optimize further by reducing redundant computations like reusing mean and rstd values.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/475">experiment with adding the llmc lib directory by karpathy · Pull Request #475 · karpathy/llm.c</a>: no description found</li><li><a href="https://en.cppreference.com/w/c/error/assert">assert - cppreference.com</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1245616070069588008)** (3 messages): 

- **Watch Party catching up**: A new member inquired about the current progress and upcoming video in the lecture series. Another member responded, mentioning that Lecture 7 is likely next, but was unsure about specific details.
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1245467775376560190)** (72 messages🔥🔥): 

- **CI Tests and Contributor Permissions Troubles**: A new contributor faced challenges with CI tests on PyTorch due to initial permissions, but was granted more lenient permissions. The [GitHub Pull Request #296](https://github.com/pytorch/ao/pull/296) was shared for testing.
  
- **Troubleshooting Installation Errors**: Several users troubleshoot installation errors regarding PyTorch, highlighting issues with building C++ extensions and suggesting various fixes, including using `USE_CPP=0 pip install .` and upgrading pip and setuptools.

- **Windows Compatibility Issues**: Members discussed the compatibility of PyTorch and triton with Windows, noting that triton is not officially supported on Windows. A specific focus was given to resolving issues for builds requiring CUDA capability greater than 8.

- **Packaging and Installation Advice**: Users provided insights on proper packaging practices in sync with PEP standards, and shared a [related PR for reference](https://github.com/TimDettmers/bitsandbytes/pull/1078). They recommended commands like `pip install --no-build-isolation .` and ensuring `wheel` is installed to avoid isolated environment issues during build.

- **CUDA and Dtype Discussion**: Users engaged in discussions about kernel compilation and the need for correct CUDA device capabilities. Specific feedback on optimizing code such as *"for bit pack, the container size can be determined from dtype"* was shared.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/actions/runs/9305510545/job/25612612623#step:15:99">Fixed f-string printing of `NF4Tensor`s (#297) · pytorch/ao@4c1d568</a>: Native PyTorch library for quantization and sparsity - Fixed f-string printing of `NF4Tensor`s (#297) · pytorch/ao@4c1d568</li><li><a href="https://pastebin.com/CMhYTn20">Processing /home/swan/pytorch/ao  Installing build dependencies ... done  Ge - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/pytorch/ao/pull/296">Graceful handling of cpp extensions by msaroufim · Pull Request #296 · pytorch/ao</a>: This fixes issues from #288 More specifically  We introduce a new env variable so we can install ao locally without building cpp extensions USE_CPP=0 pip install . this is useful out of convenience...</li><li><a href="https://github.com/TimDettmers/bitsandbytes/pull/1078">Migrate build data to pyproject.toml by matthewdouglas · Pull Request #1078 · TimDettmers/bitsandbytes</a>: Moves most metadata to pyproject.toml, conforming with PEP 517, PEP 518, and PEP 621 Removes requirements.txt files (but not the conda environment.yml ones just yet) Updates docs to instruct on usa...</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp311-cp311-win_amd64.whl">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1245459641802952934)** (5 messages): 

- **Mistral AI debuts Codestral**: Mistral AI has released Codestral, a new code-generating model that runs locally and is trained on over 80 programming languages. LlamaIndex provides [day 0 support](https://t.co/k2nHDiMnwD) and has a [notebook ready](https://t.co/YxeyHhSjKU) demonstrating its use.
- **Codestral gets Ollama support**: In addition to its standalone capabilities, Codestral is also supported by Ollama, allowing for local execution with LlamaIndex's first-class [support for Ollama](https://t.co/gsPHHF4c0K).
- **Guide on Local Knowledge Graphs**: A new guide explains how to construct knowledge graphs using local models (@ollama, @huggingface) following a pre-defined schema and employing Neo4j as the graph store. Details can be found [here](https://t.co/xhoIEi9egq) and the [full guide here](https://t.co/5ee6LwM7RE).
- **NLP Meetup in London**: @hexapode from LlamaIndex will join @weaviate_io and @weights_biases for a talk on using LLMs in financial services at a London NLP meetup on June 12th. Sign-up [here](https://t.co/vli6DY8Xg7) for insights on managing vector databases and processing financial data.
- **LlamaParse now supports spreadsheets**: LlamaParse can now handle various spreadsheets, including Excel and Numbers, transforming them into clean tables suitable for RAG pipelines. See the detailed [notebook](https://t.co/60MvR0h5DC) and the demo [here](https://t.co/IfF4UUqB0C).

**Link mentioned**: <a href="https://t.co/vli6DY8Xg7">Solving the challenges of using LLMs in production with financial services data, Wed, Jun 12, 2024, 6:00 PM   | Meetup</a>: If you are building NLP pipelines for processing financial services data, you will know how hard it can be to manage vector databases in production, reliably process large 

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1245574510246232084)** (89 messages🔥🔥): 

- **iOS Browsers Cause Site Crashes**: One user mentioned that the site keeps crashing on **iOS browsers** using Chrome and Safari. Another user suggested a more detailed bug report with reproducible steps to aid in debugging.

- **List of Default Prompt Template Variables**: A user asked for the default prompt template variables in LlamaIndex. The response clarified the variables `schema_str`, `info_str`, and `query_str` with a detailed code example and a [link to the documentation](https://docs.llamaindex.ai/en/latest/examples/vector_stores/pinecone_auto_retriever/).

- **Text Chunking Strategy**: There was a discussion about the **default chunking strategy** for creating nodes in LlamaIndex. It was clarified that the default chunking is set to 1024 tokens using the `SentenceSplitter`, with related details provided via [documentation links](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/?h=sentencesplitter#sentencesplitter).

- **Switching API Frameworks**: A user asked for advice on choosing an API framework, mentioning **Flask** and **FastAPI**. FastAPI was recommended because it supports asynchronous programming, which is beneficial for handling multiple user requests.

- **Move Data to RedisStores**: A user requested advice on moving data from **SimpleStore** to **RedisStore**. The response suggested it is hard but possible, advising on an approach to add nodes and embeddings to the new vector store and noting that the `IngestionPipeline` could automate this process.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern/#accessing-prompts">Usage pattern - LlamaIndex</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1hiDkBbAJcO3RDrS7CD2ZeQEHGqNv07pq?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLAutoVectorQueryEngine/">SQL Auto Vector Query Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/?h=sentencesplitter#sentencesplitter">Node Parser Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/?h=sentencesplitter#llama_index.core.node_parser.SentenceSplitter">Sentence splitter - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1245673302715006987)** (1 messages): 

- **Property Graphs take the spotlight in LlamaIndex**: [Unveiling the Power of Property Graphs with LlamaIndex](https://medium.com/ai-advances/unveiling-the-power-of-property-graphs-with-llamaindex-233be48934f9) is a new blog post shared in the chat, highlighting the power and capabilities of property graphs in AI development. The piece aims to provide a deep dive into how LlamaIndex leverages this technology.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1245485617731731516)** (52 messages🔥): 

- **Le Tigre Hackathon Success**: Members discussed the "Le Tigre" project built during a hackathon, described as "a multimodal variant based on Mistral 7B model inspired by the architecture of GPT-4-V". More details and project links can be found on [Devpost](https://devpost.com/software/le-tigre) and [GitHub](https://github.com/HugoLB0/Le-Tigre).

- **Upcoming LAION Datasets**: Members inquired about the ETA for the LAION 5B dataset, expressing that it feels long overdue and inquiring if the dataset would ever be rereleased.

- **Sonic: Generative Voice Model**: Cartesia AI introduced Sonic, a state-of-the-art lifelike generative voice model with 135ms latency. More information and a demo are available on their [blog](https://cartesia.ai/blog/sonic) and [Twitter](https://x.com/cartesia_ai/status/1795856778456084596).

- **ToonCrafter Skepticism**: Members were cautiously optimistic about the new ToonCrafter project for sketch-guided animation, available on [GitHub](https://huggingface.co/Doubiiu/ToonCrafter/tree/main) and [Gradio](https://github.com/ToonCrafter/ToonCrafter?tab=readme-ov-file#2-local-gradio-demo). There was some discussion and skepticism about its quality and practical utility.

- **Anime Production Costs Shrinking**: A conversation highlighted changing economics in anime production, noting that advanced modeling tools could significantly reduce costs. One member mentioned that an anime episode previously cost a few hundred thousand dollars each.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Gradio/status/1796177536348561512">Tweet from Gradio (@Gradio)</a>: Local Gradio demo available on the ToonCrafter Repo: https://github.com/ToonCrafter/ToonCrafter?tab=readme-ov-file#2-local-gradio-demo  Model: https://huggingface.co/Doubiiu/ToonCrafter    🚀 Excited ...</li><li><a href="https://huggingface.co/Doubiiu/ToonCrafter/tree/main">Doubiiu/ToonCrafter at main</a>: no description found</li><li><a href="https://doubiiu.github.io/projects/ToonCrafter/">ToonCrafter: Generative Cartoon Interpolation</a>: no description found</li><li><a href="https://x.com/cartesia_ai/status/1795856778456084596">Tweet from Cartesia (@cartesia_ai)</a>: Today, we’re excited to release the first step in our mission to build real time multimodal intelligence for every device: Sonic, a blazing fast  (🚀 135ms model latency), lifelike generative voice mo...</li><li><a href="https://x.com/hugolb05/status/1795426269099606329">Tweet from Hugo Le Belzic (@hugolb05)</a>: During the @cerebral_valley and @MistralAILabs Hackathon, we built &#34;Le Tigre&#34;, which is a multimodal variant based on Mistral 7B model inspired by the architecture of GPT-4-V . for more detail...</li><li><a href="https://youtu.be/cvZ9thKolOA?si=yHgMyzqfpM8tVcxu&t=53">The Way of the Househusband | Trailer | Netflix Anime</a>: This world-class househusband was once a feared legendary member of the yakuza!“The Way of the Househusband” is the long-awaited anime adaptation of the cozy...</li><li><a href="https://getwrightonit.com/animation-price-guide/>">no title found</a>: no description found</li><li><a href="https://devpost.com/software/le-tigre">Le Tigre</a>: Le Tigre is excels in real-time reasoning across audio, vision, and text. It is the result of our extensive collaboration and fine-tuning of Mistral&#39;s open-source models.</li><li><a href="https://laion.ai/notes/open-gpt-4-o/">Call to Build Open Multi-Modal Models for Personal Assistants | LAION</a>: &lt;p&gt;Technologies like the recently introduced GPT-4-OMNI from OpenAI show again the potential which strong multi-modal models might have to positively transfo...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1245492363246178314)** (1 messages): 

- **LAION calls for contributions to GPT-4-Omni**: LAION invites community participation in building an **open GPT-4-Omni** with the help of detailed directions provided in their [blog post](https://laion.ai/notes/open-gpt-4-o/). The initiative aims to create an open-source model with large-scale multi-modal capabilities similar to GPT-4-OMNI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://laion.ai/notes/open-gpt-4-o/">Call to Build Open Multi-Modal Models for Personal Assistants | LAION</a>: &lt;p&gt;Technologies like the recently introduced GPT-4-OMNI from OpenAI show again the potential which strong multi-modal models might have to positively transfo...</li><li><a href="https://fxtwitter.com/laion_ai/status/1795910332008804428">Tweet from LAION (@laion_ai)</a>: Help us build an open GPT-4-Omni! With this blog post we show promising directions (including data sets and tutorials) https://laion.ai/notes/open-gpt-4-o/
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1245464253595521094)** (33 messages🔥): 

- **High-Resolution Multiview Diffusion Introduced by Era3D**: A member shared details about [Era3D](https://penghtyx.github.io/Era3D), stating that it's a novel method utilizing efficient row-wise attention for high-resolution multiview diffusion. The paper includes contributions from multiple authors and institutions such as HKUST and PKU.
  
- **Concerns on Incentivized User Ratings**: The discussion highlighted concerns that providing incentives for user ratings often leads to poorer data quality. One member remarked that users might game the system by submitting random ratings to increase their chances of winning rewards, citing an instance from Midjourney where free GPU hours were misused.

- **NeurIPS Model Merging Competition Announced**: A competition focused on model merging was announced with significant interest in participation. Details about the competition, including a [tweet link](https://x.com/LChoshen/status/1796256513519989102) and a signup link, were shared for those interested in improving LLMs with a prize pool of $8,000.

- **FFT Replaces Self-Attention with Great Efficiency**: A discussion started about a 2021 paper where self-attention in transformers was replaced by FFT, achieving 92% of BERT's accuracy but with significantly lower computational costs. This approach sparked curiosity about whether it had been pursued further, with a [link to the paper](https://arxiv.org/pdf/2105.03824).

- **ToonCrafter for Generative Cartoon Interpolation**: ToonCrafter, a research project for generative cartoon interpolation, was brought up, accompanied by a [GitHub link](https://github.com/ToonCrafter/ToonCrafter). The project's feasibility and impressive results were discussed, following the discovery of the project's [documentation](https://doubiiu.github.io/projects/ToonCrafter/) showcasing its capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.18428">DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention</a>: Diffusion models with large-scale pre-training have achieved significant success in the field of visual content generation, particularly exemplified by Diffusion Transformers (DiT). However, DiT model...</li><li><a href="https://arxiv.org/abs/2403.01643">You Need to Pay Better Attention</a>: We introduce three new attention mechanisms that outperform standard multi-head attention in terms of efficiency and learning capabilities, thereby improving the performance and broader deployability ...</li><li><a href="https://penghtyx.github.io/Era3D/">Efficient High-Resolution Multiview Diffusion on Canonical Orthogonal Cameras</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.05219">Conv-Basis: A New Paradigm for Efficient Attention Inference and Gradient Computation in Transformers</a>: Large Language Models (LLMs) have profoundly changed the world. Their self-attention mechanism is the key to the success of transformers in LLMs. However, the quadratic computational cost $O(n^2)$ to ...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://doubiiu.github.io/projects/ToonCrafter/">ToonCrafter: Generative Cartoon Interpolation</a>: no description found</li><li><a href="https://tenor.com/view/ha-ha-ha-ha-ha-happy-funny-%C5%9Bmiech-gif-22074544">Ha Ha Ha Happy GIF - Ha Ha Ha Ha Ha Happy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ToonCrafter/ToonCrafter">GitHub - ToonCrafter/ToonCrafter: a research paper for generative cartoon interpolation</a>: a research paper for generative cartoon interpolation - ToonCrafter/ToonCrafter</li><li><a href="https://syncedreview.com/2021/05/14/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-19/">Google Replaces BERT Self-Attention with Fourier Transform: 92% Accuracy, 7 Times Faster on GPUs | Synced</a>: Transformer architectures have come to dominate the natural language processing (NLP) field since their 2017 introduction. One of the only limitations to transformer application is the huge computatio...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1245571712494338078)** (11 messages🔥): 

- **In-context Learning Techniques Shared**: Members discussed using prompts to "teach" a model, with methods involving 100k context windows or less. They highlighted that including information or prompts and responses in the system prompt could boost performance.

- **Efficient Data Processing Concerns**: One member expressed concerns about efficiently feeding extensive data to models for every request, mentioning it could be time-consuming. They suggested that models like **RWKV**, which can save state, might handle this more effectively.

- **Non-backpropagation Training Idea**: The idea was proposed to train models without backpropagation, using pretraining data in a new way. The member admitted it might be a complex concept and offered to explain further if needed.

- **Example of Extended Preprompt Success**: A member shared an example where in-context learning was successfully used to solve problems, referencing [this tweet](https://twitter.com/VictorTaelin/status/1776677635491344744). They noted that the individual won $10k for their achievement.
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1245468046831915091)** (5 messages): 

- **Aspirations for a Fun Learning Environment in European Universities**: One member expressed their desire to find a university in Europe with a culture conducive to learning and fun. They noted, *"just having a hard time finding a uni with that kinda culture in places like Europe."*
- **Cultural Differences Between German and US Universities**: Another member observed a significant cultural difference between universities in Germany and the US, stating, *"Professors in the US are much more approachable/personable 1:1."*
- **Seeking Team Member for Web App Development**: A member is working on a web application and looking for a potential team member, asking, *"Can I make a brief post regarding this in general?"*
- **Codestral Mistral's Code Model Introduction**: A video titled [Codestral Mistral AI's first-ever code model](https://www.youtube.com/watch?v=WRAbOHJJMF4) was shared, describing Codestral as *"an open-weight generative AI model explicitly designed for code generation tasks."*

**Link mentioned**: <a href="https://www.youtube.com/watch?v=WRAbOHJJMF4">Codestral Mistral AI&#39;s first-ever code model</a>: Codestral, is Mistal&#39;s first-ever code model. Codestral is an open-weight generative AI model explicitly designed for code generation tasks. It helps develop...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1245452539760218278)** (3 messages): 

- **Scale's private evaluation datasets boast transparency**: Shared a link to [Scale's Leaderboard](https://scale.com/leaderboard) highlighting their **proprietary evaluation datasets**. These datasets ensure "unbiased and uncontaminated results," fostering a continuously evolving competitive environment with new datasets and models.

- **Concerns raised regarding Scale**: A member expressed concerns about Scale's practices. Specifically, the member mentioned that Scale "provides the SFT and potentially the RLHF data for all of those models except potentially Llama 3," questioning the practice's reliability and transparency.

**Link mentioned**: <a href="https://scale.com/leaderboard">SEAL leaderboards</a>: no description found

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1245457755167260784)** (41 messages🔥): 

- **Llama 3 vs Code Llama**: A discussion arose about **Llama 3 70B** outperforming **Code Llama** in various tasks, despite the latter being a specialized coding model. One user expressed anticipation for "Code Llama 3".

- **Hybrid Model Strategies**: Users discussed the benefits of having **LLMs** that "can do everything" over specialized models. The potential for "dynamic offloading into the cloud" was mentioned as a crucial innovation.

- **Launch of Yuan2-M32**: The newly released **Yuan2-M32** model from IEIT-Yuan, featuring 40B parameters but only 3.7B active during generation, matched **Llama 3 70B** on most benchmarks with significantly fewer resources. Users were invited to fine-tune it and shared the [code and paper](https://github.com/IEIT-Yuan/Yuan2.0-M32).

- **NeurIPS Model Merging Competition**: An announcement for a **model merging competition at NeurIPS** was shared, with $8K in prizes. Specifics can be found in the [announcement tweet](https://x.com/LChoshen/status/1796256513519989102).

- **New Developments and Tools**: A member highlighted a new [rust library](https://x.com/LChoshen/status/1796256513519989102) for building **LLM applications**. Another shared the release of **MoRA**, a high-rank updating technique for fine-tuning, available on [GitHub](https://github.com/kongds/MoRA).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1796191402935632043?s=46">Tweet from Philipp Schmid (@_philschmid)</a>: 72.2% on MMLU and 74.4% on HumanEval with only 3.7B active parameter? 🤔 A new 40B Mixture of Experts using a new Attention Router mechanism 👀 The Yuan2-M32 was released by IEIT-Yuan.  TL;DR: 🧠 40B ...</li><li><a href="https://x.com/janleike/status/1795497960509448617">Tweet from Jan Leike (@janleike)</a>: I&#39;m excited to join @AnthropicAI to continue the superalignment mission!  My new team will work on scalable oversight, weak-to-strong generalization, and automated alignment research.  If you&#39;...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning - kongds/MoRA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1245546994731061361)** (7 messages): 

- **New Free Course on Agents by Andrew Ng**: Members discussed a new free short course on agents by Andrew Ng, which describes an agent as "something that can plan and execute various tasks independently to achieve a goal."
- **Humorous Take on Agents**: One member humorously defined an agent as "an llm in a for loop lol," adding a lighthearted perspective on the concept of agents.
- **Challenges in Creating DPO Dataset**: A member faced difficulties in creating a DPO dataset using GPT-4 and Mistral7b for responses, noting that both models produced equally good outputs when given the same question and context.
- **Exploring Weaker Models for Dataset**: To address the dataset quality issue, a member considered using weaker 7b models, but noted that Falcon7b instruct often gave "<nooutput>" for many queries.
- **Questions on Transformer XL Paper**: A member sought clarification on a concept from the Transformer XL paper regarding how the context is encoded into a hidden state and used to obtain logits, questioning the process described in the paper.
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1245674129072459807)** (14 messages🔥): 

- **Hybrid search is essential and effortless**: A member emphasized that incorporating hybrid search is crucial and very easy to implement. They stated, *"Yr hybrid search is a must and soo easy to add."*

- **MRR and NDCG for relevance metrics discussed**: Members discussed the use of **MRR** and **NDCG** as metrics for relevance, based on recommendations from consultants [Hamel et al](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/). There was confusion about whether these metrics required human evaluation and how ranking was determined post-retrieval.

- **New RAG dataset available**: A new **RAG dataset** has been shared, accessible via [Hugging Face](https://huggingface.co/datasets/glaiveai/RAG_v1) with certain conditions. It requires users to agree to share their contact information to access it.

- **Animated GIF shared for humor**: A humorous **animated GIF** of cats asking to be let in was shared, sourced from [Tenor](https://tenor.com/view/cats-let-us-in-gif-13593927). The GIF added a light-hearted moment to the discussion.

- **Work in Progress (WIP) acknowledged**: A member confirmed that a certain task is still a work in progress and is currently being tested. They mentioned, *"ah that's still a WIP, testing it rn."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/glaiveai/RAG_v1">glaiveai/RAG_v1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>: no description found</li><li><a href="https://tenor.com/view/cats-let-us-in-gif-13593927">Cats Let Us In GIF - Cats Let Us In - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1245458164510363841)** (30 messages🔥): 

- **Complexities of ABI Stability Discussed**: Community members discussed the complexities of ABI stability across different languages, highlighting that while **Rust lacks ABI stability** intentionally to avoid back compatibility issues, **Swift** maintains ABI stability for Apple operating systems. The discussion pointed out that maintaining ABI stability can often limit performance optimizations in languages like Go due to API compatibility constraints.
- **Mojo as Potential C "Protocol" Faces Skepticism**: A member humorously suggested Mojo might become a new low-level protocol similar to C, but others expressed skepticism. Members cited issues like Mojo lacking key types such as size_t and uint_fast_t, and the inertia of established languages like C providing stability.
- **Importance of C++ Interoperability for Mojo**: There's agreement on the value of Mojo having good **C++ interoperability** to leverage the vast existing codebase. **clattner** mentioned future plans to explore generating C++ headers from Mojo code, which could help ease the transition and adoption.
- **Mojo Package Management in Development**: The community is eager for a Mojo package manager, with references made to ongoing discussions about project manifest formats. Links to [GitHub discussions](https://github.com/modularml/mojo/discussions/413) and [proposal threads](https://github.com/modularml/mojo/discussions/1785) were shared, indicating ongoing but not imminent development of this feature.
- **Mojo Not Yet Available on Windows**: There's frustration among some users over the lack of Mojo support for Windows, with humorous resignations to checking back in the future. Despite requests, the answer remains that Mojo is not currently available outside of Linux environments.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rust-lang/rust/issues/111423)">Issues · rust-lang/rust</a>: Empowering everyone to build reliable and efficient software. - Issues · rust-lang/rust</li><li><a href="https://github.com/modularml/mojo/discussions">modularml/mojo · Discussions</a>: Explore the GitHub Discussions forum for modularml mojo. Discuss code, ask questions &amp; collaborate with the developer community.</li><li><a href="https://github.com/modularml/mojo/discussions/413">[RFC] Allow Importing Modules via URLs · modularml/mojo · Discussion #413</a>: Overview One of Mojo&#39;s main priorities is solving the &quot;two language problem,&quot; which means that it must function for both app development use cases, but also one off scripts. Dependency m...</li><li><a href="https://github.com/modularml/mojo/discussions/1785">[Proposal] Mojo project manifest and build tool · modularml/mojo · Discussion #1785</a>: Hi all, please check out this proposal for a Mojo project manifest and build tool. As mentioned on the proposal itself, we&#39;re looking to hear from the Mojo community: Do you agree with the motivat...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1796232248678883347>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1245780625613521036)** (1 messages): 

- **Speed up K-Means clustering with Mojo🔥**: ModularBot announced a new video on how to port **K-Means clustering from Python+NumPy to Mojo** for significant speed improvement. The video promises detailed steps and claims a massive **250x speed increase**. [Watch the video here](https://www.youtube.com/watch?v=3bg5YBCcuWA).

**Link mentioned**: <a href="https://www.youtube.com/watch?v=3bg5YBCcuWA">Speed up K-Means clustering by porting Python implementation to Mojo🔥</a>: In this video we&#39;ll share a step-by-step guide to porting kmeans clustering from Python+NumPy to pure Mojo for huge (250x) speedup! How? Mojo is Pythonic in ...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1245480100787388436)** (5 messages): 

- **Mojo's `^` operator and `bit_not` usage explained**: A user inquired if the `^` operator in Mojo was the same as in C and for `bit_not`. Another user clarified that `bit_not` is `~val` and XOR operation is `x ^ y`, while the transfer operator follows a value, like `val^`.

- **Debugging XOR operations between C and Mojo**: A member was comparing C and Mojo code involving shifts and XOR and found disagreement. Upon checking, they realized that the issue was due to printing the wrong variable in the C code, and now both codes agree.

- **`for` loops and iterators in Mojo**: A user asked how `for` loops are terminated in Mojo compared to Python's `StopIteration` and confirmed that `for` loops invoke the `__iter__` method resulting in an iterator.
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1245619388334477312)** (6 messages): 

- **Mojo vs Python comparison questioned**: A member expressed skepticism about Santiago's performance comparison on binary search between Mojo and Python, noting that calling the Python interface 100,000 times is not a fair comparison. The member was surprised to see Modular retweeting the post despite questionable benchmarking methods, as shown in [this tweet](https://x.com/svpino/status/1795811741538099685).

- **Transparency and flawed benchmarks**: Another member acknowledged that while the benchmark may not be good, Santiago is transparent about his methods. They hinted at a relaxed attitude with a shrugging emoji.

- **Repository for compiler benchmarks shared**: A cool [repository](https://github.com/nordlow/compiler-benchmark) was shared that compares the compilation speeds of different programming languages and compilers, though it does not consider caching.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/svpino/status/1795811741538099685">Tweet from Santiago (@svpino)</a>: Mojo 🔥 destroys Python in speed.  I rewrote a simple Python binary search function in Mojo. Very few changes.  I&#39;m calling the function 100,000 in a loop:  • Python: 547 ms • Mojo 🔥: 44 ms  That...</li><li><a href="https://x.com/svpino)">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1245458528437801073)** (22 messages🔥): 

- **Mojo Nightly Release Update**: A new Mojo nightly build `2024.5.3005` has been released. The update includes critical changes like removal of `Stringable` constructor from `String`, and removal of several `math` functions, among others. [Raw diff](https://github.com/modularml/mojo/compare/fadceb1d7612bd0499f7280554f8ea5d774fcdef...8ae83916ebc7b3134948f466c0f56ee3e5569062) and [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) are provided.

- **Nightly versus Released Builds**: An official from Mojo stated that currently, ~25% of Mojo installs are from nightly builds. This decision is to maintain simplicity for less experienced users who might get confused if they land on the wrong branch.

- **Stringable Constructor Removal Impact**: Users reported errors due to the removal of the `Stringable` constructor from `String`. The solution suggested was to use `str` instead.

- **CI Fix for Public Mojo Repo**: The Continuous Integration (CI) for the public Mojo repo has been fixed. The regression was due to changes in `String.strip()` and was resolved in [GitHub Pull Request #2883](https://github.com/modularml/mojo/pull/2883).

- **Behavior of `__setitem__` with List Capacity**: Users discussed that using `__setitem__` with a list's `capacity` does not update the list's length. Instead, it is recommended to use `append` to add elements to the list.

**Link mentioned**: <a href="https://github.com/modularml/mojo/pull/2883.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1245689971906449428)** (5 messages): 

- **Free tier mystery remains unsolved**: A member expressed curiosity about how another user managed to get a free tier for a service. The conversation appears unresolved with no further details provided.
  
- **MixMyAI launch announcement excites users**: A comprehensive introduction was given for [mixmyai.com](https://mixmyai.com), touted as a *"one-stop solution for all AI needs"*. Key features include no monthly fees, cheapest pricing, privacy-focused operations, a powerful UI, and support for multiple AI models.
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1245469815074983967)** (45 messages🔥): 

- **Developer Seeking Opportunities**: A user introduced themselves as a senior full stack, blockchain, and AI developer with experience in developing websites, dApps, and AI projects, asking if anyone is looking for a dev.

- **User Struggles with Free Models**: A user named *best_codes* reported issues with free models not working and asked for help. The situation seemed resolved later as they confirmed the models were working fine for them now.

- **Gemini 1.5 Pro Ratelimit Clarified**: A user asked about the ratelimit for Gemini 1.5 Pro, and another clarified that although the default in the docs is 15 RPM, they managed to negotiate a higher limit recently, suggesting possibility for custom account limits.

- **Moderated vs. Self-Moderated Models**: A discussion clarified that self-moderated models have no external moderation, whereas moderated models use an external moderator model on the endpoint to filter inputs before processing. This applies mainly to Claude on OpenRouter.

- **Laravel and Ruby Packages Announcement**: Two developers announced packages for integrating OpenRouter into Laravel and Ruby projects, respectively, and sought support and contributions from the community, sharing GitHub links for [laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter) and [open_router](https://github.com/OlympiaAI/open_router).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/moe-mizrak/laravel-openrouter">GitHub - moe-mizrak/laravel-openrouter: Laravel package for OpenRouter (A unified interface for LLMs)</a>: Laravel package for OpenRouter (A unified interface for LLMs) - moe-mizrak/laravel-openrouter</li><li><a href="https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email">MPT-7B and The Beginning of Context=Infinity — with Jonathan Frankle and Abhinav Venigalla of MosaicML</a>: Ep 13: Training Mosaic&#x27;s &quot;llongboi&quot; MPT-7B in 9 days for $200k with an empty logbook, how to prep good data for your training, and the future of open models</li><li><a href="https://github.com/OlympiaAI/open_router">GitHub - OlympiaAI/open_router: Ruby library for OpenRouter API</a>: Ruby library for OpenRouter API. Contribute to OlympiaAI/open_router development by creating an account on GitHub.</li><li><a href="https://github.com/OlympiaAI/raix-rails">GitHub - OlympiaAI/raix-rails: Ruby AI eXtensions for Rails</a>: Ruby AI eXtensions for Rails. Contribute to OlympiaAI/raix-rails development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/docs#limits>)">Docs | OpenRouter</a>: Build model-agnostic AI apps
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1245502154798530630)** (37 messages🔥): 

- **Web search API for site restriction**: A member clarified that the API allows setting a web search connector to a specific domain using an options object. Another user inquired about restricting to multiple sites simultaneously, which is pending confirmation.
- **Building an RAG for college**: One participant mentioned building a Retrieval-Augmented Generation (RAG) model for their college, aiming to include both .edu domains and external review websites like RateMyProfessors.
- **Embedding conversion query**: A member sought advice on converting uint8 embeddings to float and back for calculations, and they were directed to a more suitable channel for technical questions.
- **Startup seeking feedback**: A startup representative offered a $10 incentive for feedback on their no-code AI workflow builder to understand why users drop off after registration. This message was requested to be moved to a more appropriate channel.
- **Cohere's focus clarified**: When questioned about Cohere's position in the AI industry, an employee clarified that while they aren't focused on Artificial General Intelligence (AGI), they prioritize creating scalable models suitable for production.
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

sssandra: hi, let me give you some cohere credits! dming
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1245471790440906843)** (34 messages🔥): 

- **Automate Conversation Storage with `ChatMessageHistory`**: Kapa.ai provided detailed steps and code on implementing **storing and reloading previous conversations** using the `ChatMessageHistory` class in LangChain. The explanation included methods for storing, loading, and clearing messages, along with integrating this functionality with a `RunnableWithMessageHistory` agent, supported by [official documentation](https://python.langchain.com/v0.1/docs/modules/agents/quick_start/#adding-in-memory).

- **Challenges in Automated LLM Conversation Flow**: A member discussed a library for building **non-linear LLM conversation flows** and their challenges with extracting values and statuses efficiently. They highlighted issues with using JSON outputs in reasoning prompts and inquired about alternative methods, linking a relevant [GitHub experiment](https://github.com/TonySimonovsky/prompt_engineering_experiments/blob/main/experiments/OpenAIAttentionGrab/OpenAI%20Attention%20Grab%20(report).ipynb).

- **Building an Analytical Copilot with LangChain**: Members shared tips and solutions for creating an **analytical copilot that interacts with a PostgreSQL database**. Suggestions included implementing custom tools for SQL query results handling and using few-shot prompting to manage ambiguous user queries.

- **Combining React and SQL Agents**: Kapa.ai helped answer queries regarding the integration of `create_react_agent` with `create_sql_agent` in LangChain. The solution involved creating tools with specified names and descriptions and provided an example fix for a common error in tool initialization.

- **Expanding Knowledge Graph Capabilities**: A community member requested assistance on **enhancing the LLMGraphTransformer** to include covariates along with nodes and relationships. They shared their methodology inspired by Graph RAG and asked for guidance on modifying prompts and handling covariates effectively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/modules/agents/quick_start/#adding-in-memory>)">Quickstart | 🦜️🔗 LangChain</a>: To best understand the agent framework, let&#x27;s build an agent that has two tools: one to look things up online, and one to look up specific data that we&#x27;ve loaded into a index.</li><li><a href="https://github.com/langchain-ai/langchain/issues/20380>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#in-memory>)">Add message history (memory) | 🦜️🔗 LangChain</a>: The RunnableWithMessageHistory lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.</li><li><a href="https://github.com/langchain-ai/langchain/issues/19904>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1245516576976011345)** (2 messages): 

- **Build chat with your data using Pinecone, LangChain, and OpenAI**: [This YouTube tutorial](https://www.youtube.com/watch?v=Bxj4btI3TzY) shows step-by-step how to create a chatbot utilizing **Pinecone**, **LangChain**, and **OpenAI**. It's aimed at beginners and includes the author's blog content as an example dataset.
  
- **Everything-ai v3.0.0 integrates llama.cpp and Qdrant**: The [Everything-ai v3.0.0](https://astrabert.github.io/everything-ai/) AI assistant now supports **llama.cpp** and incorporates a **Qdrant-backed vector database** for storing and querying documents. Detailed setup instructions are provided on their [GitHub repo](https://github.com/AstraBert/everything-ai), including a **LangChain-based document preprocessing pipeline** to ensure context-aware responses.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Bxj4btI3TzY">How to build chat with your data using Pinecone, LangChain and OpenAI</a>: I show step by step how to build a Chatbot using Pinecone, LangChain and OpenAI in this easy to follow tutorial for beginners.I ingest my entire blog full of...

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

zackproser: https://www.youtube.com/watch?v=Bxj4btI3TzY
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1245455125955215421)** (23 messages🔥): 

- **Debate over Flash and Pricing Changes**: One member noted that the price change for a service happened after users had already praised its cost-effectiveness. They speculated, "most people were praising the cost-effectiveness of flash presumably based on the increased price."
  
- **Rumors of GPT-5 and Free GPT-4o**: A table shared on X hinted at GPT-5's impending arrival, with related speculation that OpenAI might be making GPT-4o free to prepare for its release. The table was noted as "Interesting (unverified)" and discussions referenced an AI expert named Alan D. Thompson [Bio on lifearchitect.ai](https://lifearchitect.ai/about-alan/).

- **OpenAI Pricing Correction**: A post on X explained that the initial rollout of OpenAI's pricing had a typo which was fixed within 24 hours. The corrected prices are now accurate as intended [LoganK's post](https://x.com/officiallogank/status/1796044236594278544?s=46).

- **OpenAI's Direction Under Microsoft Pressure**: There were discussions about internal tensions at OpenAI, mentioning how Microsoft, its largest backer, pressured the company to focus more on commercial products, causing conflicts with those inclined towards scientific research [Article on FT](https://www.ft.com/content/ccbdff7c-ede3-4d62-968d-189fb0205075).

- **OpenAI-Apple Deal Reactions**: The community reacted to news about OpenAI's collaboration with Apple, sparking speculations about whether Azure compute credits would support this deployment and how it might mesh with Apple's policies on user data. One user humorously wondered if "Satya is pissed the deal isn’t with Microsoft," discussing the broader strategic implications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_arohan_/status/1796228607255396423">Tweet from rohan anil (@_arohan_)</a>: @giffmana No worries, another cluster has arrived today in case you want to train some more</li><li><a href="https://x.com/officiallogank/status/1796044236594278544?s=46">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: @artificialguybr Hey! The current pricing page is accurate, the initial rollout had a typo in the price (last minute sprint for I/O launch) which we fixed ~24 hours later, the prices we are showing ar...</li><li><a href="https://x.com/VachanReddyK/status/1795099977766551828/photo/1">Tweet from VACHAN (@VachanReddyK)</a>: GPT-5 👀</li><li><a href="https://x.com/amir/status/1795959410340049036?s=46">Tweet from Amir Efrati (@amir)</a>: Microsoft didn’t necessarily love the OpenAI-Apple tie up.   https://www.theinformation.com/articles/openai-ceo-cements-control-as-he-secures-apple-deal?utm_source=ti_app&rc=c48ukx</li><li><a href="https://www.ft.com/content/ccbdff7c-ede3-4d62-968d-189fb0205075">Internal divisions linger at OpenAI after November’s attempted coup</a>: no description found</li><li><a href="https://lifearchitect.ai/about-alan/)">About Alan</a>: AI Consultant Former Chairman, Mensa International (gifted families) Former Consultant to Sir James Dyson&#039;s family office, PwC, Glencore… Former Head of Sound to Sir Andrew Lloyd Webber, Debbie R...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1245806878777282570)** (9 messages🔥): 

- **Ex-OpenAI board members critique regulation and events**: [Helen Toner and Tasha McCauley](https://archive.is/rwRju) have offered comments on AI regulation and events at OpenAI in a *By Invitation* piece in _The Economist_ without disclosing specifics. They scrutinized the process leading to CEO Sam Altman's resignation, which underwent an external review by WilmerHale.
- **Text Davinci-003 released after ChatGPT**: There was a discussion about the release timeline of GPT-3's iterations, noting that **Text Davinci-003** came after the release of ChatGPT, while -002 was deemed insufficient for chatbot functionality.
- **GPT-3.5 confusion and misinformation**: Members argued that saying "anyone can build ChatGPT with existing GPT-3.5" is inaccurate. They also mentioned that the naming scheme of GPT-3.5 models, especially around -002 and -003, has been confusing.

**Link mentioned**: <a href="https://archive.is/rwRju">OpenAI board members respond to a warning by former members</a>: no description found

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1245451815517163552)** (21 messages🔥): 

- **OpenInterpreter Docs Get a Shoutout**: *"Specifies which language model to use. Check out the [models](https://docs.openinterpreter.com/settings/all-settings#auto-run) section for a list of available models."* A link to the [LiteLLM](https://github.com/BerriAI/litellm) was shared: it supports over 100+ models.
- **Mobile Client for Interpreter Excitement**: A user mentioned an Android/iOS client for the interpreter and shared the [GitHub link](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile). Another user expressed anticipation to get it working with their RayNeo X2 and Brilliant Labs frames.
- **Heat Issues with Local LLMs**: A discussion about using local models like **LLaMA** on powerful setups, with one user noting their stack of NVLinked 3090s causes significant heat. Others highlighted using services like **Groq** for free model access.
- **Voice as TTS Inquiry**: *"Hi, is there a way for me to put my voice as a tts?"* A user asked about integrating their voice for text-to-speech functionalities.
- **Shipping Inquiry Redirected**: *"Is this shipping? I ordered Apr 30, 2024, 8:06:23 AM ...and well... just wondering?"* Another user was redirected to check the pinned messages in a specific channel for manufacturing updates.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile">01/software/source/clients/mobile at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#auto-run">All Settings - Open Interpreter</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1245474458047746069)** (9 messages🔥): 

- **Enthusiasm for M5 Cardputer Update**: Members expressed excitement about the M5 cardputer with one affirming *"Can't wait for an update on this! Super exciting"*. Anticipation grows as users await higher quality components in the consumer device compared to the developer kit.
  
- **Academic Disclaimer for ChatTTS**: A link to [ChatTTS on Hugging Face](https://huggingface.co/2Noise/ChatTTS) was shared with a disclaimer noting the information is *"for academic purposes only"*. It is intended solely for educational use, not for commercial or legal purposes.

- **Pinned Manufacturing Update for M5 Cardputer**: Concerns about the M5 cardputer possibly being a "money grab" were alleviated by referencing a pinned message with a manufacturing update. Members emphasized the importance of communication from the developers. 

- **Interest in Codestral Model**: One member inquired if anyone had tried Codestral yet, suggesting it *"seems like a good model"*. This prompts curiosity and potential testing among other members.

**Link mentioned**: <a href="https://huggingface.co/2Noise/ChatTTS">2Noise/ChatTTS · Hugging Face</a>: no description found

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1245461054864228372)** (17 messages🔥): 

- **OpenAI expands ChatGPT Free offerings**: Members discussed the addition of browse, vision, data analysis, file uploads, and GPTs to the ChatGPT Free tier, indicating "rate limits" as a possible constraint. [OpenAI's announcement](https://x.com/openai/status/1795900306490044479?s=46&t=90xQ8sGy63D2OtiaoGJuww) details these new features.
- **A16Z investment thesis on voice AI**: A member shared [a16z's new investment thesis](https://x.com/omooretweets/status/1795834644732285402) centered around conversational voice agents and the potential for AI to revolutionize phone calls. Some users expressed skepticism about separating genuine technical advancements from investment hype.
- **Cartesia's state space voice model launch**: [Cartesia launched Sonic](https://x.com/cartesia_ai/status/1795856778456084596), a low-latency generative voice model aiming to integrate real-time multimodal intelligence across devices. The release was discussed alongside potential implications for AI. Check out their [blog post](https://cartesia.ai/blog/sonic) and try Sonic [here](https://play.cartesia.ai/).
- **YC clarifies Sam's departure**: Paul Graham set the record straight about Sam's departure from Y Combinator in [a tweet](https://x.com/paulg/status/1796107666265108940?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), addressing misconceptions surrounding his exit.
- **Embedding Adapters for retrieval**: A discussion on a [technical report from TryChroma](https://research.trychroma.com/embedding-adapters) highlighted embedding adapters' potential to improve retrieval performance. Another member noted similarities to Vespa's approach to frozen embeddings, linking a [related Vespa blog post](https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/">Leveraging frozen embeddings in Vespa with SentenceTransformers</a>: How to implement frozen embeddings approach in Vespa using SentenceTransformers library and optimize your search application at the same time.</li><li><a href="https://x.com/omooretweets/status/1795834644732285402">Tweet from Olivia Moore (@omooretweets)</a>: 🚨 New @a16z investment thesis!  It&#39;s time for AI to reinvent the phone call - enter conversational voice agents 📱  What we&#39;re excited to invest in + market maps (from me and @illscience) 👇</li><li><a href="https://x.com/cartesia_ai/status/1795856778456084596">Tweet from Cartesia (@cartesia_ai)</a>: Today, we’re excited to release the first step in our mission to build real time multimodal intelligence for every device: Sonic, a blazing fast  (🚀 135ms model latency), lifelike generative voice mo...</li><li><a href="https://x.com/paulg/status/1796107666265108940?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Paul Graham (@paulg)</a>: I got tired of hearing that YC fired Sam, so here&#39;s what actually happened:</li><li><a href="https://x.com/openai/status/1795900306490044479?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from OpenAI (@OpenAI)</a>: All ChatGPT Free users can now use browse, vision, data analysis, file uploads, and GPTs.  Quoting OpenAI (@OpenAI)   We&#39;re opening up access to our new flagship model, GPT-4o, and features like b...</li><li><a href="https://x.com/cartesia_ai/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://research.trychroma.com/embedding-adapters">Embedding Adapters</a>: no description found</li><li><a href="https://buttondown.email/ainews/archive/ainews-sonic-a-low-latency-voice-model-for/">[AINews] 1 TRILLION token context, real time, on device?</a>: SSMs are all you need. AI News for 5/28/2024-5/29/2024. We checked 7 subreddits, 384 Twitters and 29 Discords (389 channels, and 5432 messages) for you....
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1245806790734774425)** (5 messages): 

- **New Podcast on Training Million Context LLM**: [A new podcast episode](https://x.com/latentspacepod/status/1796247856891969709) is live, titled "How to train a Million Context LLM!" featuring @markatgradient discussing the extension of Llama-3 to 1M+ contexts with nearly perfect NIAH evaluations. The episode also covers the history of long contexts, RoPE, ALiBi, Ring Attention, and various NIAH variants.

**Link mentioned**: <a href="https://x.com/latentspacepod/status/1796247856891969709">Tweet from Latent Space Podcast (@latentspacepod)</a>: 🆕 pod: How to train a Million Context LLM!  @ylecun says we should publish, or perish. We asked @markatgradient to spill ALL the beans on how his team extended Llama-3 to 1M+ context with ~perfect @G...

  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1245452118362423409)** (2 messages): 

```html
<ul>
    <li><strong>No messages to summarize</strong>: The channel "llm-paper-club-west" currently holds no substantial messages that can be summarized. Only placeholders are present without any actual content to analyze.</li>
</ul>
```
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1245805942046724270)** (1 messages): 

- **LLM360 kicks off member-organized events**: LLM360 starts the first member-led event with an [AMA on their new 65B model and open-source work](https://discord.com/events/1089876418936180786/1240722407594004561). This initiative marks the beginning of engaging community-driven events for Mozilla AI.
- **Upcoming events in the Bay Area**: For those in the Bay Area, [an IRL Open Source Hack Lab](https://discord.com/events/1089876418936180786/1243732435674337413) has been posted. Members are encouraged to click "Interested" to RSVP.
- **Embeddings demo using llamafiles**: A [demo on using llamafiles for embeddings](https://discord.com/events/1089876418936180786/1242590711778381914) will be hosted by a notable community member. This event promises to delve into practical applications of embeddings in machine learning.
- **Amplifying Devs event**: [A session titled "Amplifying Devs"](https://discord.com/events/1089876418936180786/1242653066512175157) will feature discussions with developer moderators. The focus will be on supporting developers within the Mozilla AI community.
- **AMA on GenAI Bug Bounties**: A [new AMA by 0din](https://discord.com/events/1089876418936180786/1245800040086245416) will explore GenAI bug bounties. Participants Saoud Khalifah and another community member will shed light on this emerging topic.
  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1245516861966389402)** (19 messages🔥): 

- **Error running LlamaFile on M2 Studio**: A member received an `error: unknown argument: --temp` when trying to run `granite-34b-code-instruct.Q5_0.llamafile` on an M2 Studio and sought help for a fix.
- **LlamaFile Connection Refusal**: Another member faced a "connection refused" error when building the VectorStoreIndex in Python with llamafile.exe running on port 8080. It was suggested to try binding LlamaFile to `0.0.0.0` instead of `127.0.0.1` to troubleshoot IP address issues.
- **WSL Localhost Issue Solved**: The same member discovered that WSL's definition of "localhost" did not map correctly, and specifying the WSL-specific Ethernet IP address resolved the connection issue.
- **Seeking Vision/Image Support in LlamaFiles**: A member inquired about finding LlamaFiles with vision/image support and shared a link to [Mozilla's llava-v1.5-7b-llamafile on Hugging Face](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/tree/main) which might support such functionality, with a commit by jartine.

**Link mentioned**: <a href="https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/tree/main">Mozilla/llava-v1.5-7b-llamafile at main</a>: no description found

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1245471263212961913)** (10 messages🔥): 

- **Seeking fine-tuning advice for LLMs**: Members are inquiring about methods to fine-tune large language models (LLMs) like Llava for **image and video understanding**. The question remains open for suggestions.
- **DPO data requirement concerns**: A discussion unfolded about the **data requirements for DPO** (Direct Preference Optimization) versus SFT (Supervised Fine-Tuning). A member expressed concerns that if DPO requires as much data as SFT, it could be more straightforward to ensure high-quality data in SFT from the beginning.
- **Viability of small datasets for DPO**: There’s curiosity about whether **hundreds of samples** would suffice for effective DPO, especially in domains involving general chitchat. One member voiced that assembling such a dataset by hand may be plausible.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1245460020993200149)** (4 messages): 

- **Seeking Backend Experience with Protobuf**: A member is looking for someone who works on the backend and has experience with **Google’s Protobuf**. They mentioned being ready to pay for the expertise and also specified interest in a reverse engineer, malware analyst, or bug bounty hunter.
- **DPO VRAM Usage Mystery**: Another member observed a significant reduction in **DPO VRAM usage**. They questioned if an update had occurred since their configuration had not changed yet the VRAM use was suddenly halved.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1245515666899861545)** (1 messages): 

- **Help needed for SDXL model refinement**: A member is seeking assistance to **refine SDXL models** for generating custom product ads. They mentioned trying LoRA training with unsatisfactory results and requested help from anyone experienced in **SDXL fine-tuning and ControlNet**: *"If you have experience with these, or know someone who does, Please DM."*
  

---



### **AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1245835549294461041)** (1 messages): 

- **New Game Jam from Rosebud AI**: Roberto announced a **new Game Jam: "Book to Game"** where participants can transform literary works into interactive games using Phaser JS on the AI Game Maker platform. The event boasts a **$500 prize pool** and submissions are due by July 1st. 

- **Join Rosebud AI's Third Game Jam**: Participants are encouraged to adapt any form of literature, from novels to fanfics, into a compelling game. More details can be found through their [Twitter announcement](https://x.com/Rosebud_AI/status/1796273820044595368) and joining their [Discord server](https://discord.gg/rosebud-ai).

**Link mentioned**: <a href="https://x.com/Rosebud_AI/status/1796273820044595368">Tweet from Rosie @ Rosebud AI 🌹 (@Rosebud_AI)</a>: Turn your favorite story into a game using AI! 📚 👾  Get ready for our third Game Jam: “Book to Game”. Use Rosebud Game Maker to transform a literary work into an interactive game and bring stories t...

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1245831648553996347)** (9 messages🔥): 

- **New member joins with Android troubles**: A new member mentioned they just joined and found navigating on an Android phone *"a bit hard to navigate... Glitchy and buggy"*. They clarified that despite these issues, they can interact with the world.
- **Confusion over username change**: The new member asked the group *"how do you change username?"*, acknowledging feeling like an *"alien" on the platform*.
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1245499010454786109)** (3 messages): 

- **Wondering about GPUs in the near future**: A member expressed curiosity about the future of GPUs, pondering what they will look like in **5 years (or even 2 years)**. Another member jokingly suggested, *“make a bigger systolic array 😌.”*
- **64x64 MMA hint**: In a follow-up, a member hinted at the potential of **64x64 matrix multiplication arrays (MMA)** for future GPU designs.
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1245541195254599690)** (6 messages): 

- **Tinygrad handles integer gradients easily**: A member pointed out that **Torch** does not allow integer variables to have gradients, leading to errors like `RuntimeError: Only Tensors of floating point and complex dtype can require gradients`. In **Tinygrad**, the same code can compute the gradient of integers without issue.
- **Behavior related to integer backpropagation**: Another member stated that **Tinygrad** computes as if the tensor is a float and then casts it to the original dtype. This differentiates Tinygrad from other frameworks.
- **Tinygrad superiority claim**: A member enthusiastically claimed that **Tinygrad** is better than **TensorFlow** and **PyTorch**, valuing it the most. However, this sparked a query about the reasons behind TensorFlow being considered better than PyTorch.
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1245785057839415317)** (2 messages): 

- **Splitting Codestral by language sparks curiosity**: A member wonders if **Codestral could be smaller** if split into individual programming languages. They question whether **Python training enhances the JS model** and consider using a mixture of experts approach, where each expert is a different language.
- **Weights predominantly in English**: Another member agrees, hypothesizing that **most of the model's weights are based on English** and that each programming language contributes a smaller portion. They express curiosity about the distribution of the **45GB model**.
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/)** (1 messages): 

_awesomewaffle: Will be at  the PRS event at Netflix tomorrow . Anyone else attending this event?
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1245472460971970580)** (1 messages): 

- **LAION calls to build open GPT-4-Omni**: [LAION](https://laion.ai/notes/open-gpt-4-o/) shares a blog post seeking assistance to build an open GPT-4-Omni, outlining promising directions with datasets and tutorials. The initiative invites community involvement to enrich the project.

**Link mentioned**: <a href="https://fxtwitter.com/laion_ai/status/1795910332008804428?t=rBHUXm87TFrQ-kyfeZP0fg&s=19">Tweet from LAION (@laion_ai)</a>: Help us build an open GPT-4-Omni! With this blog post we show promising directions (including data sets and tutorials) https://laion.ai/notes/open-gpt-4-o/

{% else %}

> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!


{% endif %}

---

If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

