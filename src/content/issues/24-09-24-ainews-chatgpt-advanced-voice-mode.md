---
id: 20ffb8d1-e347-4153-9868-61e5ed39c855
title: ChatGPT Advanced Voice Mode
date: '2024-09-25T01:31:24.268326Z'
original_slug: ainews-chatgpt-advanced-voice-mode
description: >-
  OpenAI has rolled out **ChatGPT Advanced Voice Mode** in the US with five new
  voices and enhanced accent and language support, marking a significant update
  in voice interaction capabilities. Meanwhile, **Alibaba Cloud's Qwen 2.5**
  models, including the newly available *72B* parameter version on HuggingChat,
  are gaining attention for their high performance and efficiency, rivaling
  larger models like GPT-4 and Llama 3.1.
tags:
  - openai
  - anthropic
  - alibaba-cloud
  - scale
  - together
  - nvidia
  - chatgpt-advanced-voice-mode
  - o1-preview
  - qwen-2.5
  - qwen-2.5-72b
  - llama-3.1
  - llama-3.1-nemotron-51b
  - gpt-4-0613
  - mistral-small
  - mistral-large
  - cohere-command-r+
  - voice-interaction
  - large-language-models
  - model-efficiency
  - model-quantization
  - multilingual-datasets
  - retrieval-augmented-generation
  - knowledge-distillation
  - open-source-models
  - generative-ai-platforms
  - ai-industry-growth
companies:
  - openai
  - anthropic
  - alibaba-cloud
  - scale
  - together
  - nvidia
models:
  - chatgpt-advanced-voice-mode
  - o1-preview
  - qwen-2.5
  - qwen-2.5-72b
  - llama-3.1
  - llama-3.1-nemotron-51b
  - gpt-4-0613
  - mistral-small
  - mistral-large
  - cohere-command-r+
topics:
  - voice-interaction
  - large-language-models
  - model-efficiency
  - model-quantization
  - multilingual-datasets
  - retrieval-augmented-generation
  - knowledge-distillation
  - open-source-models
  - generative-ai-platforms
  - ai-industry-growth
---


<!-- buttondown-editor-mode: plaintext -->**Patience is all you need, Jimmy.**

> AI News for 9/23/2024-9/24/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**222** channels, and **2572** messages) for you. Estimated reading time saved (at 200wpm): **294 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Ahead of rumored Llama 3 and Claude 3.5 updates due tomorrow:

Today we saw [a big Gemini Pro price cut](https://x.com/OfficialLoganK/status/1838611055217385646), updating Gemini pricing inline with [the new $/intelligence frontier](https://x.com/Smol_AI/status/1838663719536201790/photo/1) we have been charting on this newsletter.

![image.png](https://assets.buttondown.email/images/f74bb222-114d-4c7b-844f-4ac7622c4ac2.png?w=960&fit=max)

But probably the headline story is [ChatGPT Advanced Voice Mode](https://twitter.com/OpenAI/status/1838642444365369814), which company leaders (like [Mira](https://x.com/miramurati/status/1838642696111689788)!) announced as "rolling out this week" but it seems most people in the US received access by the end of day. There are 5 new voices, and improved [accent](https://x.com/chatgpt21/status/1838686328936108294)/language support. and yes, with some effort, [it can still sing](https://x.com/swyx/status/1838751593417839089)!

![image.png](https://assets.buttondown.email/images/e90fba09-ca40-47ef-b022-fe908c845338.png?w=960&fit=max)


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

**AI Model Developments and Releases**

- **OpenAI's o1-preview model**: [@omarsar0](https://twitter.com/omarsar0/status/1838353480672563581) shared insights on o1-preview's performance in planning tasks, noting it shows progress but lacks robustness on longer problems and unsolvable instances. The model achieved 52.8% accuracy on Randomized Mystery Blocksworld, significantly outperforming other LLMs.

- **Anthropic's rumored new model**: [@bindureddy](https://twitter.com/bindureddy/status/1838310463194685663) and [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1838320450927047112) mentioned rumors of Anthropic dropping a new model, generating excitement in the AI community.

- **Qwen 2.5 release**: [@_philschmid](https://twitter.com/_philschmid/status/1838122129792839918) highlighted the release of Qwen 2.5, with the 7B model matching OpenAI's GPT-4 0613 on various benchmarks. The model is available in sizes 1.5B, 7B, and 32B (coming soon), supporting up to 128K tokens.

**AI Research and Benchmarks**

- **PlanBench evaluation**: [@omarsar0](https://twitter.com/omarsar0/status/1838353480672563581) discussed a paper evaluating o1-preview on PlanBench, comparing it to LLMs and classical planners. The study revealed o1-preview's strengths in planning tasks but also highlighted its limitations.

- **Multilingual MMLU dataset**: [@_philschmid](https://twitter.com/_philschmid/status/1838230108072476951) announced OpenAI's release of a Multilingual Massive Multitask Language Understanding (MMMLU) dataset on Hugging Face, covering 14 languages and 57 categories.

- **RAG research standardization**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1838259514237464893) mentioned RAGLAB, a framework for standardizing Retrieval-Augmented Generation (RAG) research, allowing fair comparisons of 6 RAG algorithms across 10 benchmarks.

**AI Applications and Tools**

- **PDF2Audio**: [@_akhaliq](https://twitter.com/_akhaliq/status/1838219563705532750) shared a tool for converting PDFs into audio podcasts, lectures, and summaries.

- **Open-source AI starter kit**: [@svpino](https://twitter.com/svpino/status/1838186602885177835) introduced a self-hosted AI starter kit with components for low-code development, local model running, vector storage, and PostgreSQL.

- **Moshi speech-based AI assistant**: [@ylecun](https://twitter.com/ylecun/status/1838327979203588100) announced the open-sourcing of Moshi, a speech-based AI assistant from Kyutai.

**AI Industry and Business**

- **Scale AI developments**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1838272073652932642) reported Scale AI's growth, hitting nearly $1B ARR earlier than expected and growing 4x year-over-year.

- **Together Enterprise Platform**: [@togethercompute](https://twitter.com/togethercompute/status/1838287277010883001) introduced their platform for centralized GenAI process management, offering 2-3x faster inference and up to 50% reduction in operational costs.

**AI Ethics and Societal Impact**

- **Sam Altman's blog post**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1838287528614318353) shared insights from Sam Altman's blog post "The Intelligence Age," discussing the potential impact of AI on human capabilities and society.

- **AI regulation discussions**: [@togelius](https://twitter.com/togelius/status/1838335777446691266) expressed concerns about proposed AI regulation bills, arguing they might hinder open-source development and concentrate power in private companies.

**Memes and Humor**

- [@agihippo](https://twitter.com/agihippo/status/1838220613221306770) joked about normalizing YAML configs for ordering sandwiches, highlighting the pervasiveness of tech concepts in everyday life.

- [@Teknium1](https://twitter.com/Teknium1/status/1838315311545880929) humorously commented on o1's inability to rewrite code, poking fun at the model's limitations.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Qwen 2.5: A New Benchmark in Local LLM Performance**



- **Qwen2.5 Bugs & Issues + fixes, Colab finetuning notebook** ([Score: 85, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/)): The post highlights **critical bugs** in **Qwen 2.5 models**, including incorrect **EOS tokens** and **chat template issues** that can cause **NaN gradients**. The author has uploaded **fixed models** and **4-bit quantized versions** to [Unsloth's Hugging Face page](https://huggingface.co/unsloth), and provided **Kaggle and Colab notebooks** for finetuning **Qwen 2.5 models** (base and conversational) using [Unsloth](https://github.com/unslothai/unsloth), which offers **2x faster** and **70% less VRAM** usage during finetuning.

- **[Qwen 2.5 72B is now available for free on HuggingChat!](https://huggingface.co/chat/models/Qwen/Qwen2.5-72B-Instruct)** ([Score: 196, Comments: 36](https://reddit.com//r/LocalLLaMA/comments/1fniqym/qwen_25_72b_is_now_available_for_free_on/)): **Qwen 2.5 72B**, a large language model, is now accessible for free on **HuggingChat**. This model, developed by **Alibaba Cloud**, boasts **72 billion parameters** and is part of the Qwen (Tongyi Qianwen) series, offering capabilities in various languages including English, Chinese, and code generation.
  - **Qwen 2.5 72B** is now available on **HuggingChat** with a **32k context window**, improved role-playing abilities, and structured data handling. The developer is seeking feedback and resources on tool use for potential integration with their [tools feature](https://huggingface.co/chat/tools).
  - Users discussed replacing outdated **Mixtral models** with alternatives like **Mistral Small**, which is comparable to **Llama 3.1 70B** in performance. HuggingChat offers generous usage limits with only per-minute rate limiting and no daily caps.
  - Some users noted the model's improved performance over smaller versions, while others pointed out an amusing quirk where it claims to be developed by **Anthropic** instead of acknowledging its true identity as Qwen.


- **How did Qwen do it?** ([Score: 235, Comments: 127](https://reddit.com//r/LocalLLaMA/comments/1fnhm67/how_did_qwen_do_it/)): The **Qwen 2.5** models are receiving positive feedback for their impressive performance, with the **32B** model performing similarly to **70B** models. This raises questions about the efficiency of running larger models when smaller ones can achieve comparable results, potentially making **local LLMs** more attractive. The post author inquires about the factors behind Qwen's success, speculating on possible reasons such as improved **data quality**, **extended training periods**, or other **advancements** in model development.
  - **Qwen2.5** models were trained on up to **18 trillion tokens** of high-quality data, with the **32B** model performing similarly to older **70B** models. The **Apache 2.0 license** applies to most models except the commercially valuable **3B and 72B** versions.
  - Users report that **Qwen2.5 72B** outperforms **Mistral Large** and **Cohere Command R+** in most tasks, except story writing. The **32B** model has replaced **Hermes 3 Llama 3.1 70B** for some users, offering similar or better results with faster performance.
  - Concerns were raised about **Qwen2.5** models lacking cultural knowledge, as discussed in a [Hugging Face thread](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/discussions/1). Some users argue this trade-off is acceptable for specialized tasks, while others believe baseline knowledge is necessary for a well-rounded LLM.


**Theme 2. Advancements in LLM Efficiency and Quantization**


- **New Llama-3.1-Nemotron-51B instruct model from NVIDIA** ([Score: 205, Comments: 47](https://reddit.com//r/LocalLLaMA/comments/1fnp2kt/new_llama31nemotron51b_instruct_model_from_nvidia/)): NVIDIA has released **Llama-3.1-Nemotron-51B-instruct**, a **51.5B parameter** LLM derived from **Llama-3.1-70B-instruct** through block-wise distillation and optimization for a single **H100-80GB GPU**. The model underwent knowledge distillation using **40 billion tokens** from **FineWeb**, **Buzz-V1.2**, and **Dolma** datasets, focusing on English single and multi-turn chat use-cases, and is available on [Huggingface](https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct) with a repo size of **103.4GB**.
  - Users expressed excitement for **width-pruned Qwen 2.5 32B** and **Qwen 70B** models. The **Qwen 14B** model achieves an **MMLU score of ~80**, comparable to 4th-year university level, as detailed on the [Qwen blog](http://qwenlm.github.io/blog/qwen2.5/).
  - NVIDIA also developed a **40B variant** of the model, achieving a **3.2x speed increase** over the parent model with moderate accuracy loss. The architecture resembles **DeciLM**, suggesting NVIDIA may have integrated **Deci's AutoNAC** technology.
  - The model's context size is unclear, with conflicting information in the configuration. The `max_position_embeddings` is set to **131,072**, but the `original_max_position_embeddings` in the RoPE scaling settings is **8,192**.


- **Running LLMs at Custom Floating-Points (Near-Lossless FP6)** ([Score: 54, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1fo5bbk/running_llms_at_custom_floatingpoints/)): The post discusses the implementation of **custom floating-point formats** for **runtime quantization of LLMs**, allowing loading of **FP16 models** directly into **FP4, FP5, FP6, and FP7** with minimal accuracy loss and throughput penalty. The author explains the technical details of their approach, including **bit-level pre-packing** and **SIMT-efficient GPU runtime** with **parallel dequantization**, which enables competitive performance even with irregular bit-widths. Benchmarks show that **FP5 and FP7** achieve similar results to **FP8** on **GMS8K**, while **FP6** even exceeds **BF16 quantization**, leading the author to suggest **FP6** as a potential standard for balancing memory and accuracy trade-offs.
  - **Custom floating-point formats** for runtime quantization are discussed, with users noting potential **compute efficiency** advantages over grouped quantizations like **exl2 6bpw** and **GPTQ**. The **5bpw** format is highlighted as a meaningful trade-off for certain models and sizes.
  - Concerns about the **statistical significance** of benchmark results on **GMS8K** were raised, suggesting the need for more comprehensive evaluations. The author acknowledged this, mentioning plans to run **MMLU-Pro** and possibly perplexity/KL divergence tests.
  - Users inquired about **model conversion** to **FP6** format, with instructions provided for using the command line interface. The author noted that exporting models in these formats is not currently possible but may be integrated into **llm-compressor** if demand increases.


**Theme 3. AI for Creative Applications: Gaming and Music**

- **OpenMusic: Awesome open-source text-to-music generation!** ([Score: 59, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1fo19nn/openmusic_awesome_opensource_texttomusic/)): **OpenMusic** is an open-source **text-to-music generation** project available on [Hugging Face](https://huggingface.co/spaces/jadechoghari/OpenMusic). The project, which also has a [GitHub repository](https://github.com/ivcylc/qa-mdt), allows users to generate music from text prompts.

- **[I'm experimenting with small LLMS for a Skyrim + AI setup. I am astonished by Qwen's inference speed.](https://www.reddit.com/gallery/1fo5bt3)** ([Score: 87, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1fo5bt3/im_experimenting_with_small_llms_for_a_skyrim_ai/)): The author is experimenting with **small language models** for a **Skyrim + AI setup** and expresses astonishment at the **inference speed of Qwen**. While no specific performance metrics are provided, the post suggests that Qwen's speed stands out compared to other models tested in this gaming-related AI application.
  - **Skyrim AI mods** like **Mantella** and **AI Follower Framework (AIFF)** enable NPC interactions using **LLMs**. AIFF offers more features but is limited to companions, while Mantella allows conversations with any NPC.
  - Users are experimenting with various **LLMs** for Skyrim, including **Qwen 2.5 7B**, **Llama 3.1 8B**, and **Gemma 9B**. **Roleplay-tuned models** are recommended for more believable NPC interactions.
  - The author is using an **MSI GP66 11UH-032** gaming laptop with an **RTX 3080 mobile 8GB GPU**, aiming to run LLMs on less than **6GB VRAM**. **Quantized 7b-8b GGUF models** have shown excellent performance.


**Theme 4. New AI Datasets and Research Papers**

- **Open Dataset release by OpenAI!** ([Score: 235, Comments: 51](https://reddit.com//r/LocalLLaMA/comments/1fno6d4/open_dataset_release_by_openai/)): OpenAI has released the **Multilingual Massive Multitask Language Understanding (MMMLU)** dataset on **Hugging Face**. The dataset is now publicly available at [https://huggingface.co/datasets/openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU), providing researchers and developers with a new resource for multilingual language understanding tasks.
  - Users expressed skepticism about **OpenAI's** motives, with some suggesting the dataset might be **"poisoned"** or designed to favor their models. The **GPTslop epidemic** was cited as a reason for caution when using OpenAI's outputs for training.
  - The choice to translate **MMLU** was questioned, as it's known to have **problematic questions** and **invalid answer choices**. Some suggested **MMLU-Pro** would have been a better option, given that many models already score around **90%** on MMLU.
  - Despite skepticism, users acknowledged the value of **open benchmarks** for reproducibility and model comparison. The dataset's size (**194k test set**) was noted as potentially excessive for computing a single score.


- **[Google has released a new paper: Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)** ([Score: 229, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1fo6bdg/google_has_released_a_new_paper_training_language/)): Google researchers have introduced a novel approach called **Self-Correction via Reinforcement Learning (SCRL)** to improve language model outputs. The method uses **reinforcement learning** to train models to **self-correct** their initial outputs, resulting in **improved performance** across various tasks including **question answering**, **summarization**, and **reasoning**. SCRL demonstrates significant improvements over standard fine-tuning, with gains of up to **11.8%** on certain benchmarks.
  - The **Self-Correction via Reinforcement Learning (SCRL)** method's effectiveness was questioned, with users discussing how to ensure **genuine self-correction** rather than intentional error generation. The paper's focus on generalizing self-correction ability was highlighted as a key insight.
  - Users debated the paper's methodology, noting that the prompt doesn't explicitly state the solution is wrong. Some pointed out that **Qwen 72B** model could solve all 8 math problems zero-shot, raising questions about data leakage and the need for novel evaluation sets.
  - Discussion touched on the paper's theoretical focus versus practical application, emphasizing that research papers often test specific theories rather than producing end products. The concept of generalizing improvement steps was explained using an **ELI5 analogy** of number addition.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Advancements and Releases**

- **OpenAI's GPT-4 Turbo (o1) models**: Several posts discuss the impressive capabilities of OpenAI's new o1 models, particularly o1-mini. Users report [significant improvements in tasks like complex mathematical problem-solving](https://www.reddit.com/r/singularity/comments/1fnxu28/o1mini_is_so_insane/), with one user describing it as "using actual magic" for certain applications.

- **Anthropic's new model release**: [Anthropic was expected to release a new AI model](https://www.reddit.com/r/singularity/comments/1fnussc/anthropic_will_likely_drop_a_new_model_tomorrow/), generating excitement in the AI community.

- **Cost reductions in AI inference**: [OpenAI's Dane Vahey reported that the cost per million tokens has fallen from $36 to $0.25 in 18 months](https://www.reddit.com/r/singularity/comments/1fo2nj8/dane_vahey_of_openai_says_the_cost_per_million/), representing a significant decrease in AI operational costs.

**AI Research and Development**

- **Multi-agent AI research**: Both [Google DeepMind and OpenAI are forming teams focused on multi-agent artificial general intelligence research](https://www.reddit.com/r/singularity/comments/1fnp3pj/google_deepmind_is_now_hiring_for_a_team_in/), indicating a growing interest in this area.

- **AI in image and video generation**: Advancements in AI-powered image and video manipulation were highlighted, including [a workflow for CogVideoX-I2V](https://www.reddit.com/r/StableDiffusion/comments/1fnn08o/cogvideoxi2v_workflow_for_lazy_people/) and [a demonstration of simultaneous control of multiple subjects in video generation](https://www.reddit.com/r/singularity/comments/1fnu6qi/three_subjects_are_controlled_simultaneously_with/).

**Industry and Market Developments**

- **Anthropic's potential valuation increase**: [Anthropic is reportedly in talks with investors about raising capital at a valuation of $30-40 billion](https://www.reddit.com/r/singularity/comments/1fnvmst/openai_rival_anthropic_has_started_talking_to/), potentially doubling its previous valuation.

- **Political engagement with AI**: [U.S. Vice President Kamala Harris pledged to boost AI investments](https://www.reddit.com/r/singularity/comments/1fnjo2f/kamala_harris_vows_to_boost_ai_and_crypto/) in a fundraiser speech, indicating growing political interest in AI development.

**Perspectives on AI Progress**

- **Rapid advancement of AI capabilities**: Several posts reflect on the [rapid progress in AI capabilities](https://www.reddit.com/r/singularity/comments/1fnlkof/how_fast_things_change/), with many tasks previously thought to be far from solved now being achievable.

- **Future AI predictions**: [Yann LeCun predicted that AI matching or surpassing human intelligence will arrive soon](https://www.reddit.com/r/singularity/comments/1fnuysf/yann_lecun_says_we_will_soon_have_ai_that_matches/), along with AI assistants in smart glasses capable of translating hundreds of languages within a year or two.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1. AI Models Level Up: New Releases and Major Updates**

- [**Mistral Small Model Unleashes 22 Billion Parameters**](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409): The new **Mistral Small model** is live, boasting **22 billion parameters** to advance AI performance across tasks. Users can explore it via the [HF Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e).
- [**OpenAI's o1 Models Spark Interest Despite Unlabeled Graphs**](https://x.com/hughbzhang/status/1838288923656941860): OpenAI released the **o1 family of models** with scaling law graphs missing x-axis labels, prompting users to reconstruct data using the **o1-mini API**. Discussions ponder whether the compute involves mere tens of thousands of tokens.
- [**Gemini Models Get a Boost and a Price Cut**](https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/): **Gemini-1.5-Pro-002** and **Gemini-1.5-Flash-002** received updates with over **2x higher rate limits** and a **50% price drop**. Developers are thrilled about these changes, marking "a good day to be a developer."

**Theme 2. Voice Features Roll Out Amidst Controversy**

- [**OpenAI's Advanced Voice Feature Speaks 50+ Languages**](https://x.com/OpenAI/status/1838642444365369814): **Advanced Voice** is rolling out to **Plus** and **Team** users, adding Custom Instructions, Memory, and **five new voices** with improved accents. Users can now express phrases in over **50 languages**.
- [**European Users Left Voiceless and Frustrated**](https://x.com/OpenAI/status/1838642453391511892): Despite the rollout, European users are disappointed as **Advanced Voice is not yet available** in several European countries. Many express that it "falls short of earlier demos."
- **Users Debate Voice Assistant's Censorship and Limitations**: Discussions highlight that OpenAI's focus on safety leads to a limited voice assistant. Users complain it lacks the dynamism seen in roleplaying AI products like **Character.ai**.

**Theme 3. Developers Wrestle with AI Integration and Optimization**

- [**OpenRouter Integrates with Cursor and Offers Demo Apps**](https://x.com/OpenRouterAI/status/1838627801072562232): OpenRouter now works seamlessly in **[Cursor](https://x.com/OpenRouterAI/status/1838627801072562232)** with all models, including Anthropic's. They've also released [demo apps](https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling) on GitHub to kickstart development.
- **Aider Installation Frustrations Spark Uninstalls**: Users face challenges installing **Aider**, leading to multiple reinstall attempts using **pipx** without resolving the issues. Some resort to reverting to older versions to regain functionality.
- **GPU MODE Changes Name but Sparks Mixed Feelings**: The community formerly known as **CUDA MODE** transitions to **GPU MODE**, aiming for a broader focus. Members have mixed reactions, with humorous suggestions like "*Gigachad Processing*" and debates over the name change.

**Theme 4. AI Reasoning and Reliability Under the Microscope**

- [**LLMs Can't Plan? OpenAI's o1 Evaluated**](https://arxiv.org/abs/2409.13373): A new research note critically assesses **OpenAI's o1** model's planning capabilities, suggesting it "can't plan" despite being marketed as a **Large Reasoning Model**.
- **Hallucinations Haunt High-Temperature Outputs**: Users report that increasing the temperature above **1.25** causes models to hallucinate, questioning the reliability of outputs. Instructing models not to hallucinate helps but doesn't fully solve the problem.
- **JSON Formatting Fiasco Frustrates Developers**: API users struggle with JSON formatted outputs, often receiving incomplete or incorrect responses like a simple '{'. Better-defined prompt structures are suggested but issues persist.

**Theme 5. Collaborative Efforts and Tools Enhance AI Development**

- [**DSPy 2.5.0 Launch Tackles 100 Issues Faster Than You Can Say 'Chain-of-Thought'**](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb): The release of **DSPy 2.5.0** aims to swiftly address **50-100 issues**, with users enthusiastic about new features and upcoming intro notebooks.
- [**GitHub Repositories Blossom with AI Tools**](https://github.com/jack-tol/youtube-to-audio): New tools like **YouTube-to-Audio** offer easy extraction of audio from videos, while frameworks like **LitServe** simplify serving and scaling LLMs using FastAPI.
- **Community Bands Together for Fine-Tuning and Model Training**: Members share experiences in fine-tuning models like **Vit_B16** and **Llama3.1**, emphasizing the importance of high-quality data. Collaborations with model developers help resolve issues like bugs in **Qwen 2.5**.

---

*Note: All links and details are based on the discussions across various Discord channels and reflect the latest updates and community sentiments.*


---

# PART 1: High level Discord summaries




## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cursor integrates with OpenRouter!**: OpenRouter now works seamlessly in [Cursor](https://x.com/OpenRouterAI/status/1838627801072562232) with all models, including those from Anthropic. *Thank you @cursor_ai for fixing this!* 🍾
   - This integration enhances user experience by simplifying operations and expanding model accessibility.
- **Gemini models upgraded with better performance!**: Two updated models, **Gemini-1.5-Pro-002** and **Gemini-1.5-Flash-002**, are now available, featuring reduced pricing and improved performance metrics.
   - These models are optimized for efficiency, with faster outputs and higher rate limits, and will auto-update user-facing aliases by October 8, 2024.
- **OpenRouter rolls out demo apps for quick start**: The OpenRouter team has announced basic demo apps available on [GitHub](https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling) to help developers kickstart their projects.
   - These demos include a simple 'tool calling' feature, making it easier for users to create applications from scratch.
- **Discussion on Middle-Out Transform implications**: Users expressed concerns over disabling the middle-out transform as default, citing negative impacts on workflows and infrastructure.
   - The community stressed the need for clearer communication and updates regarding model changes to mitigate disruptions.
- **Insights on Token Pricing Structures**: A relevant discussion highlighted varying token pricing across models, stating that OpenRouter utilizes native tokens returned from upstream for cost calculations.
   - Users noted that discrepancies in tokenizers between models like GPT-4o and Qwen can significantly affect token counts and pricing estimations.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Mistral Small Model Launches**: The new [Mistral Small model](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409) is live, featuring **22 billion parameters**, aimed at advancing AI performance in various tasks.
   - Users can explore this model further via the [HF Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e).
- **Gradio 5 Sets Performance Standards**: **Gradio 5 (Beta)** is officially released, introducing major usability improvements and server-side rendering for faster app loading.
   - Feedback has been encouraged before a public rollout, aiming to refine features based on community insights.
- **FinePersonas Adds Richness to Synthetic Data**: The latest [FinePersonas v0.1](https://x.com/reach_vb/status/1836882281434165629) offers **21 million personas**, enhancing synthetic data generation for diverse applications.
   - This dataset aims to provide realistic query generation tailored to specific persona needs, revolutionizing large-scale data projects.
- **Hugging Face Token Issues Afloat**: Multiple users reported problems with invalid Hugging Face tokens, prompting discussions on potential rate limit issues and reinstalling the huggingface-hub package.
   - Despite troubleshooting efforts, many continued to experience token validation failures.
- **OpenAI’s Word Generation Numbers Astonish**: OpenAI reportedly generates about **100 billion words daily**, with a user questioning if Hugging Face could approach this metric using its own models.
   - This discussion highlights the significant text generation capability differences between entities in the AI space.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RWKV Architecture Offers Unique Insights**: The community dissected the **RWKV architecture**, especially its efficiency over **convolutions**, emphasizing a need for simpler explanations to foster adoption.
   - Familiarity with **GLA** is essential, as participants advocate for a clearer breakdown of complexities surrounding RWKV.
- **Introducing the YouTube-to-Audio Tool**: A user unveiled **youtube-to-audio**, a command-line tool that extracts audio in various formats like **MP3** and **WAV** from YouTube, enhancing user experience.
   - This tool also supports playlist downloads and custom file names, positioning itself as an ad-free alternative to existing solutions.
- **Dynamic Evaluation Sparks Debate**: Dynamic Evaluation in ML proposes fine-tuning models on the test set, raising concerns over its external validity and alignment with classic practices.
   - Although valid, members highlighted the critical need for similar distributions between training and testing datasets.
- **muP Implementation Clarified**: The community works to simplify **muP** (Maximal Update Parameterization) concepts for neural networks, critical for increasing community engagement.
   - The push for clearer implementations alongside theoretical insights aims to boost ease of integration for developers.
- **OpenAI's LRM Displaying Planning Limitations**: Members evaluated OpenAI's **o1 (Strawberry)** as a Large Reasoning Model (LRM) and debated its effectiveness, especially under specific test conditions.
   - Concern arose over its planning capabilities, with reports indicating **inference time compute** leapfrogging to over **50%**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini Models Get Major Updates**: The introduction of **Gemini-1.5-Pro-002** and **Gemini-1.5-Flash-002** brings a **50% price drop** and enhanced rate limits, impressing developers though coding benchmarks remain static.
   - As of October 1st, input costs decrease from **$3.50** to **$1.25/million tokens** for inputs under **128,000 tokens**.
- **RoRF Open-Sourced for Enhanced Performance**: The **Routing on Random Forest (RoRF)** has been launched as open-source, offering **12 pre-trained model routers** that significantly improve MMLU performance.
   - This release is poised to advance model routing techniques across various applications.
- **Anticipation Builds for New Claude Models**: Speculations arise about upcoming releases of new **Claude models** like **Haiku 3.5** or **Opus**, as users express frustration over the wait since the last update.
   - This delay has fed hopes for timely announcements in the upcoming weeks.
- **Inconveniences with Aider Installation**: Users face challenges installing **Aider**, often needing to reinstall using **pipx** after functionalities fail, despite efforts to fix their setups.
   - Reverting to older versions is among the suggested solutions as users navigate these complications.
- **Prompt Caching Features Under Discussion**: The configuration of **AIDER_CACHE_KEEPALIVE_PINGS** for prompt caching with **Anthropic API** and **OpenRouter** is a hot topic, as users seek clarity on implementations.
   - Links to the **Prompt Caching** [documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) are shared to facilitate understanding.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Advanced Voice Features Rolled Out**: The **Advanced Voice** feature is now available for **Plus** and **Team** users, enhancing interaction within the ChatGPT app with new functionalities like [Custom Instructions](https://x.com/OpenAI/status/1838642444365369814). This update also introduces **five new voices** and significantly boosts multilingual capabilities, allowing expressions in over **50 languages**.
   - However, the rollout has caused confusion and disappointment among European users, with some noting it falls short of earlier demos.
- **JSON Formatting Quality Under Fire**: Users are facing issues with JSON formatted outputs, often retrieving unsatisfactory responses like simple '{', leading to frustration among API users seeking structured data. Suggestions have been made to improve the quality of these outputs through better-defined structures.
   - Despite recommendations for clearer prompts, many still encounter poor performance, limiting their ability to achieve effective API responses.
- **Clarifying Prompt Engineering for Better Outputs**: Discussion around **prompt engineering** emphasizes the necessity of clear, detailed requests to maximize model performance and relevance in outputs. Several members underscored that specific examples in prompts greatly enhance the quality of generated responses.
   - The challenges highlighted include generating diverse content, particularly in niche applications like **Minecraft** questioning, which has faced repetitiveness.
- **Hallucinations and Response Reliability Concerns**: Participants raised alarms about the model's propensity to hallucinate as the temperature setting exceeds **1.25**, indicating a lack of reliability in outputs. Insights from members suggest that instructing the model to avoid hallucination may limit irrelevant content but does not entirely solve the issue.
   - This concern about hallucination extends to various functions within the ChatGPT, prompting users to seek methods for better performance.
- **Generating Engaging Minecraft Questions**: A user tested a prompt aimed at generating fun and engaging **Minecraft** questions in JSON format, facing hurdles in achieving diverse and engaging queries. Feedback from the community aimed to refine this process amid challenges of repetitive output and hallucination.
   - This quest for creativity in question formation has sparked discussions on improving prompt strategies and effectively utilizing API capabilities.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU Mode Transition**: The community transitioned from **CUDA MODE** to **GPU MODE**, aiming for a broader focus on various GPU programming frameworks beyond just CUDA.
   - Members expressed mixed feelings about the name change, suggesting alternatives like **Heterogeneous Computing** or even humorous names like *Gigachad Processing*.
- **Training Optimizations with Distributed Systems**: Discussions highlighted scaling issues during training with **2x 4090 GPUs**, noting that **DDP** offers better performance than **FSDP** under low bandwidth conditions.
   - Participants emphasized the impact of communication bandwidth on scalability and shared experiences in optimizing distributed training workloads.
- **WebNN Integration Prospects**: Suggestions arose for creating a dedicated channel for **WebNN**, reflecting on its role in integrating **WebGPU** and **WASM**, which may face challenges in standardization.
   - Clarifications were made regarding **WebNN's** ability to interface with NPU APIs, demonstrating its potential for diverse hardware setups.
- **Luma Job Openings for Performance Engineers**: Luma is actively seeking engineers for their **Dream Machine** project, offering positions focused on performance optimizations for multimodal foundation models.
   - Candidates are expected to have significant experience in distributed training and low-level kernel optimization, with the company highlighted as rapidly growing.
- **Data Handling Challenges on GPU**: Members noted that effective data transfer on GPUs is heavily dependent on **latency** and **memory bandwidth**, with inquiries about comparisons to CPU setups.
   - One participant raised concerns about how these factors impact GPU performance and usability in systems-on-a-chip architectures.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI's o1 Models Raise Interest**: OpenAI released the **o1 family of models** along with a graph showing scaling laws for test-time compute, although the x-axis was unlabeled, sparking discussions about reconstructing the data using the o1-mini API.
   - Members pointed out that the compute likely involves only tens of thousands of tokens, raising feasibility questions around scaling further without a proper structure.
- **Anthropic's Potential $40 Billion Valuation**: Reports indicate that **Anthropic** is in discussions to raise capital, which might boost its valuation to between **$30 billion** and **$40 billion**, effectively doubling from earlier this year.
   - This reflects a serious competitive drive as AI companies scramble for substantial financial backing in a fast-evolving market.
- **James Cameron Joins Stability AI Board**: Stability AI welcomed **James Cameron** to its Board of Directors, aiming to leverage his expertise to explore innovations in visual media.
   - This strategic move is seen as crucial for developing a more comprehensive AI pipeline tailored for creators.
- **Gemini Model Enhancements Announced**: Updates for the **Gemini models** reveal enhancements including **over 2x higher rate limits** and a **50% price drop** on Gemini 1.5 Pro, along with new features for developers.
   - The revisions also introduce opt-in filters for improved safety and reliability, allowing better control over model settings.
- **Scale AI's Financial Growth Insights**: Scale AI reported nearly quadrupled sales in H1 despite low gross margins, indicating robust growth amidst rising demand for AI services.
   - This financial surge puts Scale AI in a compelling position as the landscape continues to shift towards AI-enabled solutions.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **O1 Planning Capabilities Evaluation**: A recent [research note](https://arxiv.org/abs/2409.13373) on **O1** outlines its planning capabilities, with team members reportedly burning the midnight oil for completion.
   - The findings detail a thorough examination, with more insights promised following its public release.
- **World Simulator API Offers Low-Cost Access**: Discussion centered around **World Sim**, highlighting an opportunity for users to earn credits upon sign-up, while incurring low costs for API usage.
   - Encouragement for account creation to leverage free credits was a common sentiment in the channel.
- **Hermes & Monad Showing Stubbornness**: Concerns about **Hermes** and **Monad** reported becoming less effective in interactions, particularly in tagging abilities.
   - One suggestion involved implementing a presence penalty, while others noted differences based on hosting environments.
- **Gemini 1.5 Sparks Excitement**: Anticipation builds around the **Gemini 1.5** upgrade released in September, coupled with a minor rollout for **GPT-4o**.
   - Members expressed eagerness for breakthroughs that may emerge from the upcoming **Meta Connect** event.
- **DisTrO Efficient in Poor Bandwidth**: Initial findings indicate **DisTrO** operates effectively over asymmetric bandwidth and heterogeneous GPU environments, enhancing resource management.
   - This positions **DisTrO** as a viable option for resource allocation in suboptimal network conditions.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 model issues resolved**: Members discussed problems with the **Qwen 2.5** model, reporting crashes and bugs, but shared solutions like improved templates and modified training approaches.
   - Collaboration with the Qwen team yielded progress in addressing some of these issues, allowing for better model stability.
- **Unsloth Trainer's memory usage optimized**: A user experienced **memory issues** initializing the **UnslothTrainer** and suggested reducing dataset mapping processes to resolve it.
   - Their follow-up indicated success with fewer processes, highlighting the significance of balancing for better memory performance.
- **Fine-tuning models insights shared**: Experience shared on fine-tuning a **Vit_B16** model highlighted that **high-quality data** trumps sheer volume for improved results.
   - The user plans to enhance their model further with additional quality images after achieving notable accuracy.
- **Memory issues with Llama3.1 addressed**: A user faced **out of memory errors** loading the **4-bit quantized Llama3.1**, with **20GB** allocations failing when **14.75GB** was used by PyTorch.
   - Community members suggested adjustments to the model configurations as a troubleshooting step for these OOM issues.
- **Exploring improvements with Reinforcement Learning**: Discussion on how OpenAI applies **RLHF (Reinforcement Learning from Human Feedback)** to upgrade their models based on user interactions.
   - Participants noted the challenges of utilizing prior conversations for guiding model improvements, emphasizing a lack of structured feedback in training methodologies.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Anticipation Builds for New Anthropic Model Release**: A source confirmed that a **major AI model upgrade** from **Anthropic** is expected to be released soon, with full details available after the embargo lifts.
   - Members are buzzing about the implications this upgrade may have for developers in the AI landscape.
- **Perplexity Pro Features Leave Users Curious**: Confusion arises as users discuss limitations with **Perplexity Pro** accounts, particularly regarding daily search limits.
   - While some users question the value of Pro accounts, others acknowledge the benefits of a more personalized search experience.
- **Merlin Extension Gains Traction**: Discussion around the **Merlin extension** highlights its capability to chat with various **LLMs** directly, providing unlimited model access.
   - Users appreciate the unlimited queries but express concerns over transparency in the model settings compared to **HARPA AI**.
- **Inconsistencies in Citation Outputs Cause Frustration**: Members express frustration over **inconsistent citation outputs** fetched via the API, alternating between **HTML** and **Markdown** formats.
   - This inconsistency is reportedly hampering automation efforts, complicating reliable output generation.
- **AI's Role in Education Scrutinized**: An exploration of how AI impacts education reveals both transformative benefits and challenges, with an ongoing discussion on its implications.
   - Members analyze different facets of AI integration into educational settings and the potential shifts in learning dynamics.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Installation on Air-Gapped Machines**: Members discussed the feasibility of installing **LM Studio** on air-gapped machines, emphasizing that initial setup and file transfers are necessary even though the installation itself does not require internet.
   - *Air-gapped installations require careful planning*, particularly for downloading installers and models separately.
- **Model Performance Hits a Wall**: Users reported performance issues with models when approaching their token limits, noting that slowdowns occur due to VRAM constraints as tokens fill up.
   - This leads to the recommendation of *managing token limits* to maintain optimal performance.
- **LongWriter Model Sparks Interest**: The **LongWriter** model was praised for its ability to generate extensive texts, with resources shared for interested members to explore its properties further.
   - Members were encouraged to review the [GitHub page for LongWriter](https://github.com/THUDM/LongWriter) for insights on usage and capabilities.
- **Concerns Over Dual GPU Compatibility**: A discussion on whether **LM Studio** supports dual GPU setups raised inquiries about mixing an **RTX 4070 Ti** with an **RTX 3080**, along with the potential performance benefits.
   - Advice centered on assessing *compatibility concerns* before attempting such configurations.
- **High GPU Prices Frustrate EU Buyers**: Members expressed frustration over higher GPU prices in the EU, often reaching around **$750**, compared to lower pricing in the US.
   - Regional pricing issues were attributed to **VAT** and taxes, alongside a discussion on the advantages of *consumer protections* available in Europe.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Tops the Language Tier List**: A user ranked **Mojo** at the top of their personal language tier list, citing it above **C#** and **Rust** as a subjective but heartfelt decision.
   - There’s a call for a clearer separation in **C++** categories, particularly emphasizing the importance of clean **C** interoperability.
- **Rust Faces Slow Compilation Scares**: Users lamented **Rust's slow compilation times**, especially in larger projects like a **40k line game**, which can drag on significantly.
   - Generics were identified as major contributors to these slowdowns, with suggestions to optimize file system settings on **Windows**.
- **NixOS Sparks Interest Amid Caution**: Conversations around migrating to **NixOS** praised its package management, but concerns arose about the overall complexity of the system.
   - Members debated the potential of **Ansible** as a simpler tool for smaller projects while exploring **NixOS's** reproducibility benefits.
- **MLIR Wins Over LLVM in Discussions**: Questions about why **MLIR** might be better than **LLVM** centered on improvements in parallel compilation and high-level semantics handling.
   - **MLIR's** ability to retain debug information is seen as a crucial advantage, especially as compilers evolve.
- **Celebrating Mojo's Progress and Future**: The community celebrated **Mojo's two-year anniversary**, reflecting on its growth, including key developments like the **Mojo SDK** release.
   - Enthusiasm about the language's future was palpable, with users eagerly discussing how its evolution will shape the years ahead.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 2.5.0 Launch Spreads Excitement**: The launch of **DSPy 2.5.0** aims to swiftly tackle **50-100** issues, garnering enthusiasm for new features and upcoming intro notebooks.
   - Members suggested establishing public weekly meetings for further feedback on the release.
- **Effective GROQ API Key Setup**: User guidance on setting the **GROQ_API_KEY** and executing `lm = dspy.LM('groq/llama3-8b-8192')` facilitates **Llama 3** integrations.
   - This instruction streamlines usage of the **dspy** library with models hosted on GROQ.
- **Chain of Thought Evaluated in Recent Paper**: A paper highlights a **quantitative meta-analysis** of over **100 studies** on Chain-of-Thought (**CoT**) prompting, showing its effectiveness enhances tasks with math or logic.
   - Key findings indicate that direct answering matches CoT on **MMLU** for symbolic operations, stressing the need for reasoning when questions involve an **equals sign**.
- **Custom Adapters for LLM Usage Discussed**: Members explored the creation of **custom adapters** to specify additional parameters like `grammar` for structured outputs with **dspy.LM**.
   - The conversation focuses on sharing experiences and a need for clearer best practices regarding parameter usage.
- **Anticipation Builds for Multimodal Capabilities**: Newly expected multimodal features of DSPy are set to roll out next week, with compatibility inquiries regarding audio models like **Ultravox**.
   - Official responses indicate the initial focus will be on Vision Language Models (VLMs).



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Today's Lecture on AI Frameworks and Multimodal Assistants**: The **3rd lecture** of the Berkeley MOOC, featuring [this livestream](https://www.youtube.com/live/OOdtmCMSOo4), will cover **Agentic AI Frameworks & AutoGen** with Chi Wang and steps to build a **multimodal knowledge assistant** with Jerry Liu starting at **3:00pm PST**.
   - Chi will address **core design considerations** of agentic AI programming while Jerry will discuss elements like **structured outputs** and **event-driven workflows**.
- **Clarification on Course Attendance Confusion**: The attendance form for the livestream is meant for Berkeley students only, causing some confusion among MOOC participants.
   - Next time, clearer instructions will accompany the QR code to prevent misunderstandings.
- **Exploring Open Embedding Models**: Members identified **jina-embeddings-v3** from Jina AI as the leading open embedding model offering multilingual capabilities and utilizing **Task LoRA**.
   - This model enhances performance in **neural search applications**, emphasizing the importance of effective indexing and relevance.
- **AutoGen vs. CrewAI on Customization and Speed**: In multi-agent collaboration, members noted **AutoGen** allows for greater customization, while **CrewAI** shines in quick prototyping, albeit lacking in back-and-forth communication.
   - The **conversable_agent** in AutoGen enables more complex interactions, a feature CrewAI users found limiting.
- **Search/Retrieval Techniques for RAG**: Discussion suggested focusing on **classical NLP techniques** to enhance information retrieval, particularly **ranking algorithms** and **semantic understanding**.
   - Understanding these techniques is critical for improving search within the RAG framework, allowing for better indexing and relevance.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Letta AI Emerges from Stealth**: Excitement surrounds the launch of [Letta AI](https://x.com/sarahwooders/status/1838261104864346288?s=46), a company focused on developing stateful LLM agents, by founders [Sarah Wooders](https://x.com/sarahwooders) and [Charles Packer](https://x.com/sarahwooders/status/1838261104864346288?s=46). They are actively hiring and building their team in San Francisco.
   - Read more about Letta in [TechCrunch](https://techcrunch.com/2024/09/23/letta-one-of-uc-berkeleys-most-anticipated-ai-startups-has-just-come-out-of-stealth/).
- **Gemini Model Enhancements**: [Gemini models](https://x.com/OfficialLoganK/status/1838611055217385646) received significant updates, including double the rate limits and over a 50% price reduction on Gemini 1.5 Pro. Filters have switched to opt-in, and an updated Flash 8B experimental model has been released.
   - Developers are optimistic about these changes, viewing it as a great time for developers, as explained in the [Google Developers Blog](https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/).
- **Voice Feature Rollout**: OpenAI announced that [Advanced Voice](https://x.com/openai/status/1838642444365369814?s=46) is rolling out to Plus and Team users within the ChatGPT app, introducing multiple new features and improved accents. Notably, it can express phrases in over 50 different languages.
   - However, access is not yet available in several European nations, as highlighted by [OpenAI's](https://x.com/OpenAI/status/1838642453391511892) announcement.
- **Customer Service Agent Experimentation**: Discussion about challenges in managing multi-turn conversations with agent simulations revealed important insights into maintaining effective user interaction. Suggestions included implementing stage markers and setting clear conversation termination guidelines.
   - Users are exploring various approaches to integrate reinforcement learning into conversation management to improve the customer agent experience.
- **HuggingChat macOS App Introduction**: The newly released [HuggingChat](https://x.com/cyrilzakka/status/1838618605648490974?s=61) app for macOS offers native integration of open-source LLMs with features like markdown support and web browsing. It marks a significant step forward in user-friendly AI tools for direct desktop use.
   - This app demonstrates a trend toward enhancing accessibility and functionality in AI-driven applications.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Newcomers drawn to Cohere AI**: Members like **Nav**, a Mechanical Engineering student, showcased interest in learning about **Cohere** and AI while seeking resources such as blogs or videos, leading to a shared link about the [Aya Research](https://cohere.com/research/aya) initiative to advance multilingual AI.
   - This initiative aims to enhance accessibility, enabling a broader understanding of AI applications across languages.
- **Job anxiety alleviated in community chat**: **Milansarapa** voiced concerns over job insecurity, prompting community assurance about having a contract in hand, reinforcing the importance of support.
   - *You have the contract already* became a reassuring mantra, highlighting the usefulness of community engagement.
- **Cohere Toolkit gets notable features**: Recent updates to the **Cohere Toolkit** have fixed various back-end/UI issues and introduced features such as pinning chats and support for **parquet** and **tsv** files, with a [YouTube demo available](https://youtu.be/gdJ0abx9mvo).
   - These enhancements significantly improve user experience and demonstrate the team's commitment to community feedback.
- **Reranker faces multilingual challenges**: Reports surfaced that the **multilingual reranker** suffers from low relevance scores in languages like **Polish**, filtering out useful data, which renders it ineffective.
   - *The relevance score is so low that it gets filtered out* indicates a need for better handling of diverse languages within the reranking process.
- **Exploring Chain of Thought (COT)**: **Milansarapa** queried about the **Chain of Thought** (COT) mechanism, prompting discussions on how it can enhance performance on certain tasks.
   - Ultimately, COT serves as a valuable approach for problem-solving, though its application varies case by case.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **James Cameron Joins Stability AI Board**: Legendary filmmaker **James Cameron** has joined the **Stability AI Board of Directors**, announced by CEO Prem Akkaraju. This addition supports Stability AI's mission to transform visual media with **cutting-edge technology**.
   - Known for **The Terminator** and **Avatar**, Cameron aims to revolutionize storytelling through innovative AI solutions for visual media.
- **Seeking FNAF Loras Collaborators**: A member seeks fellow **FNAF** fans to assist in creating **Loras** for the game. They're looking for collaborators to bring this project to life.
   - *Anyone interested in collaborating on this project?*
- **Boosting SDXL Performance with 3090 EGPU**: A user reported purchasing a **3090 EGPU** to enhance their **SDXL** gameplay experience, overcoming past failures with similar products. They shared frustrations about certain **Aurus** gaming boxes.
   - *Quality issues with similar products were noted, leading to this decision.*
- **Exploring ControlNet's Capabilities**: A user inquired about **ControlNet**, which guides image generation, particularly for poses. Specifications can be challenging to clarify with only language.
   - *Effective guidance methods are a crucial focus for improved image outputs.*
- **Troubleshooting OpenPose Editor Installation**: A user reported issues with the **OpenPose editor** in Forge, suggesting it may require a specific installation command. Assistance was provided about running **pip install basicsr** in the virtual environment.
   - *Clarifications around installation commands were shared for better integration.*



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Beware of fraudulent LlamaParse site**: A warning has been issued about a **fraudulent site** pretending to be **LlamaIndex's** LlamaParse, directing users to avoid it.
   - The legitimate LlamaParse can be accessed at [cloud.llamaindex.ai](https://cloud.llamaindex.ai) to prevent confusion.
- **LitServe simplifies serving LLMs**: The **LitServe** framework from LightningAI simplifies serving and scaling LLMs using FastAPI, as showcased in a demo with [LlamaIndex](https://t.co/Xikqk20peW).
   - This setup can host a straightforward **RAG** server locally against **Llama 3.1**, making it efficient for developers.
- **Creating an AI Product Manager in 50 Lines!**: An **AI product manager** can be built in just 50 lines of code using LlamaIndex and ComposioHQ, which includes features like email feedback reading.
   - If approved, it integrates feedback into a **Linear** board for edits, showcasing the efficacy of the function calling agent architecture.
- **Exploring Human-in-the-Loop Workflows**: Members discussed implementing human-in-the-loop (HITL) interactions with nested workflows, aiming to ease user control return post-events.
   - An event-driven approach was proposed for managing user responses dynamically during workflow processes.
- **Effective Web Crawling Techniques for RAG**: A discussion centered on technologies for crawling web pages for embedding, with inquiries on options like Puppeteer versus tools such as Firecrawl or Crawlee.
   - Members shared insights into effective methods for integrating web-crawled data into retrieval-augmented generation (RAG) pipelines.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **User Feedback Sparks Blendtain Improvements**: A user shared excitement for **Blendtain** but noted its tendency to cut off messages, suggesting a feature to adjust message length.
   - Another user simply agreed with a thumbs-up, reflecting positive reception of the feedback.
- **Playlist Generator Launched by dykyi_vladk**: [Adify.pro](https://adify.pro) was introduced as a new playlist generator that customizes playlists based on user prompts, created by dykyi_vladk.
   - The creator proudly referred to it as his *'coolest thing'*, indicating personal investment in the project.
- **Collaborative Machine Learning Study Proposal**: dykyi_vladk invited others to DM him for a collaborative **Machine Learning** study initiative, promoting community engagement.
   - This was presented in a friendly tone, emphasizing teamwork in pursuing knowledge.
- **Dominance Shifts in Image Processing Algorithms**: A member questioned the current dominance of **GANs**, **CNNs**, and **ViTs** in image processing tasks, seeking confirmation of these trends.
   - They expressed interest in a visual timeline to illustrate shifts among these algorithms over time.
- **EleutherAI's muTransfer Collaboration**: EleutherAI launched a project with [Cerebras](https://cerebras.ai/) to enhance the accessibility of **muTransfer**, aiming to decrease training costs.
   - Members speculated that the approach might already be **dated**, questioning its relevance compared to newer methods.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Nvidia's 51B Synthetic Data Model Excites**: Discussion ignited over Nvidia's **51B synthetic data model**, which reportedly exhibits strong **MMLU** performance and potential for enhanced applications.
   - *It would be fun to try fine-tuning and inferencing with it,* highlighting members' eagerness to explore practical applications.
- **Auto Chunking: A Loss of Context?**: A debate emerged over the practicality of **auto chunking** in conversations, with concerns that *Imagine your convo split in half midway. The context is lost*.
   - One member pointed out that systems like **ST** and **Kobold** typically manage overflow by retaining initial messages.
- **Dynamic Context Management Proposed**: There were discussions on how **dynamic context management** could assist **LLMs** in handling conversational shifts more effectively.
   - Members suggested this strategy as a probable fix for exceeding context limits.
- **Qwen 2.5 Supported on Axolotl**: Confirmation came in that **Qwen 2.5** is supported on **Axolotl** for normal text processing, though vision features might lack support.
   - The acknowledgment reflects limitations potentially affecting applications involving visual data.
- **Analysis of Fine-Tuning Spikes**: A member reported a significant **spike** during fine-tuning on a **100K row dataset**, seeking correlations through logging.
   - The lack of immediate logging help was noted, impacting troubleshooting efforts.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Pydantic Compatibility Broken**: Users encounter an error while importing `ChatOpenAI` from `langchain_openai`, due to the `__modify_schema__` method being unsupported in Pydantic v2. They are advised to check their Pydantic version and use `__get_pydantic_json_schema__` instead, as detailed in the [LangChain documentation](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility/#1-passing-pydantic-objects-to-langchain-apis).
   - As Pydantic v2 was released in June 2023, developers should ensure they are using compatible methods to avoid integration issues.
- **Watch Out for GraphRecursionError!**: A `GraphRecursionError` arises in LangGraph applications when the recursion limit of 25 is hit, hindering execution. Users can increase the recursion limit in their configuration, as suggested in a related [GitHub issue](https://github.com/langchain-ai/langchain/issues/18598).
   - This adjustment is critical for preventing crashes during complex graph operations in LangGraph.
- **Call for LLM Friendly Docs!**: A user prompts for more LLM-friendly documentation to boost LangChain productivity. Ongoing discussions indicate a community interest in improving resources available for developers working with LangChain.
   - This shows a need for better guidelines tailored to enhancing LLM integrations, indicating the community’s focused efforts.
- **Mistral vs Mixtral: The Great Showdown**: A comparison is brewing between **Mistral** and **Mixtral** regarding self-hosting solutions in the open-source arena. Members are curious about performance metrics and usability when it comes to these models.
   - This conversation highlights the community's interest in optimizing and selecting the best open-source models for practical applications.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Confusions Around CPU Offloading in Optimizers**: Discussion arose about why CPU offloading for the optimizer isn't being utilized, referencing [this old issue](https://github.com/pytorch/pytorch/issues/74588) that mentioned slowdowns.
   - One member suggested using **PagedAdam** with CPU offloading to optimize performance while emphasizing the need for a PR to consider single-device fine-tuning.
- **Comparative Analysis of Optimizer Methods**: It was noted that using **torchao's CPUOffloadOptimizer** doesn't pair well with the optimizer in backward, raising questions about faster alternatives like Adam.
   - Recommendations included trying `offload_gradients=True` for gradient memory savings while optimizing CPU and GPU processing as detailed in this [PR](https://github.com/pytorch/ao/pull/584).
- **CUDA MODE Community Invitation**: A suggestion was made to join the GPU MODE Discord group for members interested in performance optimization, highlighting more qualified individuals available for help.
   - The link shared for joining is [here](https://discord.gg/jNsm4H44), encouraging broader participation in discussions on optimization.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Exploring the Concept of a Planetary Brain**: Members playfully compared the potential of **tinyboxes** connecting for a **planetary brain** through **distributed training**, pushing the boundaries of collective intelligence.
   - This suggests a fascinating future where advanced **distributed training** might operate on a global scale.
- **Introduction to DisTrO for Distributed Training**: Discussion centered on the **[DisTrO project](https://github.com/NousResearch/DisTrO)** that facilitates **Distributed Training Over-The-Internet**, targeting revolutionary collaboration across models.
   - This initiative emphasizes the need for cooperative frameworks in model training, enhancing scalability and accessibility.
- **AttributeError: 'Tensor' lacking cross_entropy**: A user faced an 'AttributeError' due to the **Tensor** object lacking the `cross_entropy` attribute during the training step, highlighting a potential implementation flaw.
   - Participants speculated on the underlying causes, pointing to a possible gap within the **Tensor** functionality.
- **Tinygrad Version Debate**: The conversation sparked over the correct **Tinygrad** version after a user transitioned from version **0.9.2** to the latest master, exposing functional limitations.
   - Recommendations were made to consistently update to incorporate essential features for performance gains.
- **Model Architecture and Training Insights**: One participant shared their model architecture leveraging multiple convolutional layers followed by a flattening operation and linear layers to enhance training efficacy.
   - The dialogue emphasized design strategies aimed at optimizing model performance during training iterations.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Gets Refreshing Updates**: Open Interpreter is actively receiving updates on [GitHub](https://github.com/OpenInterpreter/open-interpreter), showcasing continuous development efforts.
   - Significant focus surrounds project '01', aimed at integrating a dedicated voice assistant mode, as detailed [here](https://github.com/OpenInterpreter/01).
- **LLM Takes On Browser Automation**: A member discussed using Open Interpreter for LLM-based browser automation, confirming functionality while noting it's limited by task complexity.
   - They recommended employing Playwright for enhancements and shared a [prompt example](https://github.com/morisy/openinterpreter-configs/blob/main/foiaportalassistant.yaml) they have been honing.
- **Community Pumps Up the Enthusiasm**: Despite initial skepticism, community members are eager to automate submissions to directories using shared prompts.
   - Engagement remains strong as members respond to questions and exchange experiences with the tool.
- **Exciting Community Event on the Way**: An upcoming event related to Open Interpreter was announced, with a [Discord link for more details](https://discord.gg/open-interpreter-1146610656779440188?event=1288234745477726238) shared.
   - This news sparked excitement among users, indicating that community interest is alive and well.
- **Project Perception Nuances Explored**: In response to a query, a member humorously highlighted that some in the community might not be fully aware of the project's progress.
   - This points to varying perceptions on Open Interpreter's vitality within discussions.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1288186634239742064)** (3 messages): 

> - `Cursor Integration`
> - `Gemini Update`
> - `Database Downtime`
> - `New Nous Model`
> - `Open-source Vision Language Models` 


- **Cursor integrates with OpenRouter!**: OpenRouter now works seamlessly in [Cursor](https://x.com/OpenRouterAI/status/1838627801072562232) with all models, including those from Anthropic.
   - *Thank you @cursor_ai for fixing this!* 🍾
- **Gemini 1.5 models upgraded!**: [Gemini-1.5-flash](https://openrouter.ai/models/google/gemini-flash-1.5) and [gemini-1.5-pro](https://openrouter.ai/models/google/gemini-pro-1.5) are now routed to the newest 002 version.
   - This update brings both models in line with the latest features and improvements.
- **Scheduled Database Downtime**: A **downtime notice** was shared, indicating that on Friday at 10am ET, there will be a 5-10 minute downtime for database upgrades.
   - This will ensure smoother operations moving forward.
- **Nous launches multilingual Llama 3.1!**: A new finetune of **Llama 3.1 8B** optimized for multilingual dialogue has been released by Nous, available at [this link](https://openrouter.ai/models/nousresearch/meta-llama-3.1-8b-instruct).
   - This model aims to enhance global communication capabilities.
- **Roast yourself with VLMs!**: Several open-source **vision language models** are now live, including the **Mistral Pixtral 12B** ([link](https://openrouter.ai/models/mistralai/pixtral-12b)) and **Qwen** series models ([Qwen2-VL-7B-Instruct](https://openrouter.ai/models/qwen/qwen-2-vl-7b-instruct), [Qwen2-VL-72B-Instruct](https://openrouter.ai/models/qwen/qwen-2-vl-72b-instruct)).
   - Make sure to ask them to roast a picture of you in the chatroom! 🙂


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1838627801072562232">Tweet from OpenRouter (@OpenRouterAI)</a>: Thank you @cursor_ai for fixing this!  OpenRouter now works in Cursor with all models, including Anthropic 🍾</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5">Gemini Flash 1.5 - API, Providers, Stats</a>: Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video...</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 - API, Providers, Stats</a>: Google&#x27;s latest multimodal model, supporting image and video in text or chat prompts.  Optimized for language tasks including:  - Code generation - Text generation - Text editing - Problem solvin...</li><li><a href="https://openrouter.ai/models/nousresearch/meta-llama-3.1-8b-instruct">Llama 3.1 8B Instruct - API, Providers, Stats</a>: A fine-tune of [Llama-3.1 8B Instruct](/models/meta-llama/llama-3. Run Llama 3.1 8B Instruct with API</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b">Pixtral 12B - API, Providers, Stats</a>: The first image to text model from Mistral AI. Its weight was launched via torrent per their tradition: https://x. Run Pixtral 12B with API</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-vl-7b-instruct">Qwen2-VL 7B Instruct - API, Providers, Stats</a>: Qwen2 VL 7B is a multimodal LLM from the Qwen Team with the following key enhancements:  - SoTA understanding of images of various resolution &amp; ratio: Qwen2-VL achieves state-of-the-art performanc...</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-vl-72b-instruct">Qwen2-VL 72B Instruct - API, Providers, Stats</a>: Qwen2 VL 72B is a multimodal LLM from the Qwen Team with the following key enhancements:  - SoTA understanding of images of various resolution &amp; ratio: Qwen2-VL achieves state-of-the-art performan...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1288069920734580756)** (1 messages): 

> - `OpenRouter App Development`
> - `Demo Apps on GitHub` 


- **OpenRouter offers demo apps to kickstart development**: The OpenRouter team announced the availability of basic demo apps for those interested in building their own applications, found on [GitHub](https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling).
   - These demos include a **simple 'tool calling' demo**, designed to guide users through the initial stages of app creation.
- **Invitation for feedback on demo apps**: The OpenRouter team is open to receiving **feedback and requests** from users regarding the demo apps.
   - They encouraged community engagement, stating that users' opinions will help improve future offerings.



**Link mentioned**: <a href="https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling">tai-llm-chat/demos/tool_calling at main · pxl-research/tai-llm-chat</a>: Repository with demo code for LLM&#39;s (using Azure OpenAI and OpenRouter) - pxl-research/tai-llm-chat

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1287852298701242430)** (378 messages🔥🔥): 

> - `OpenRouter's middle-out transforms`
> - `New Gemini Models`
> - `Token Pricing Structures`
> - `Performance of various LLMs`
> - `User Experiences with Models` 


- **Discussion on OpenRouter's Middle-Out Transforms**: Users questioned the disabling of the middle-out transform as the default, citing negative impacts on their current infrastructure and workflows.
   - Concerns were raised about accessibility and communication regarding model changes, with some users emphasizing the need for clearer updates.
- **New Gemini Models Announcement**: Google announced the release of two updated models, **Gemini-1.5-Pro-002** and **Gemini-1.5-Flash-002**, with significant reductions in pricing and improved performance metrics.
   - The new models are designed with faster outputs, higher rate limits, and will automatically update user-facing aliases by October 8, 2024.
- **Token Pricing Structures Across Providers**: Discussion was held about varying token pricing across different models, noting that OpenRouter utilizes native tokens returned from upstream for cost calculations.
   - Users were informed that differences in tokenizers between models like GPT-4o and Qwen can impact token count and pricing estimations.
- **Performance Comparisons of LLMs**: Comparative performance analyses showed that while Gemini Flash 002 is faster than GPT-4o Mini, it sometimes fails to meet coding constraints.
   - Users shared experiences with generative coding tasks, highlighting Gemini's strengths in certain areas while noting limitations in adherence to task requirements.
- **User Experience and Bug Fixes**: Users expressed appreciation for quick bug resolutions from model providers like SambaNova and OpenRouter, noting prompt fixes after reporting issues.
   - Feedback on user experience emphasized efficiency and responsiveness within the platforms, which builds user confidence in the technologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2024/Sep/24/gemini-models/">Updated production-ready Gemini models</a>: Two new models from Google Gemini today: `gemini-1.5-pro-002` and `gemini-1.5-flash-002`. Their `-latest` aliases will update to these new models in &quot;the next few days&quot;, and new `-001` suffi...</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: See how you&#x27;ve been using models on OpenRouter.</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">Responses | OpenRouter</a>: Manage responses from models</li><li><a href="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash)">no title found</a>: no description found</li><li><a href="https://openrouter.ai/models?max_price=0">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://ai.google.dev/gemini-api/docs/safety-settings">no title found</a>: no description found</li><li><a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: Transform data for model consumption</li><li><a href="https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/">Updated production-ready Gemini models, reduced 1.5 Pro pricing, increased rate limits, and more</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions">no title found</a>: no description found</li><li><a href="https://github.com/open-webui/open-webui/blob/6b463164f4b129e0ce4bdc9008dd661214fe5eb5/backend/open_webui/apps/openai/main.py">open-webui/backend/open_webui/apps/openai/main.py at 6b463164f4b129e0ce4bdc9008dd661214fe5eb5 · open-webui/open-webui</a>: User-friendly WebUI for LLMs (Formerly Ollama WebUI) - open-webui/open-webui</li><li><a href="https://openrouter.ai/models/alpindale/magnum-72b">Magnum 72B - API, Providers, Stats</a>: From the maker of [Goliath](https://openrouter.ai/models/alpindale/goliath-120b), Magnum 72B is the first in a new family of models designed to achieve the prose quality of the Claude 3 models, notabl...</li><li><a href="https://openrouter.ai/models/alpind">Models: &#x27;alpind&#x27; | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://github.com/OpenRouterTeam/open-webui/commit/89659df1fa10348f51b389a8fea27b67a71dec5d">add middle-out by default · OpenRouterTeam/open-webui@89659df</a>: no description found</li><li><a href="https://github.com/OpenRouterTeam/open-webui">GitHub - OpenRouterTeam/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)</a>: User-friendly WebUI for LLMs (Formerly Ollama WebUI) - OpenRouterTeam/open-webui
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/)** (1 messages): 

godling72: In my case it would be something I'm running myself.
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1288231645341548668)** (1 messages): 

> - `Mistral Small Model Release`
> - `Gradio 5 Launch`
> - `FinePersonas Dataset Introduction`
> - `bitsandbytes 0.44.0`
> - `Wikimedia Structured Wikipedia Dataset` 


- **Mistral Small Model Unleashed**: The new [Mistral Small model](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409) is now live featuring 22 billion parameters, promising significant advancements in AI performance.
   - The model is part of a collection that can be explored through the [HF Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e).
- **Gradio 5: Build & Share with Ease**: [Gradio 5](https://5-0-dev.gradio-website.pages.dev/playground) simplifies the process of creating and sharing machine learning apps, requiring minimal setup with just a few lines of code.
   - This tool integrates seamlessly with any Python library, allowing developers to present their models effectively and generate public links for easy access.
- **FinePersonas is Here for Synthetic Data**: The latest [FinePersonas v0.1](https://x.com/reach_vb/status/1836882281434165629) provides 21 million personas, aiding in the creation of diverse synthetic data for various applications.
   - This dataset can generate realistic queries and content tailored to specific personas, revolutionizing synthetic data generation.
- **bitsandbytes 0.44.0 Now Available**: The newly announced [bitsandbytes 0.44.0](https://x.com/mattkdouglas/status/1838403695605690444) introduces an 8-bit version of the AdEMAMix optimizer, optimizing performance.
   - It also incorporates CUDA graphs support for inference, showcasing advancements in the capability of lightweight model optimizers.
- **Wikimedia's Structured Wikipedia Dataset Unveiled**: Wikimedia has released a structured [Wikipedia dataset](https://enterprise.wikimedia.com/blog/hugging-face-dataset/) for public feedback, sourced from its Snapshot API.
   - This dataset offers improved machine-readable formats, simplifying access and analysis for researchers and developers alike.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://5-0-dev.gradio-website.pages.dev/playground)">Gradio</a>: Build &amp; Share Delightful Machine Learning Apps</li><li><a href="https://x.com/reach_vb/status/1836882281434165629)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Introducing FinePersonas-v0.1 - Permissively licensed 21 Million Personas for generating massive scale (diverse & controllable) synthetic data! 🔥  Produced with @AIatMeta Llama 3.1 70B Instruct, @arg...</li><li><a href="https://x.com/mattkdouglas/status/1838403695605690444)">Tweet from Matthew Douglas (@mattkdouglas)</a>: Announcing bitsandbytes 0.44.0!  We&#39;ve implemented an 8-bit version of the AdEMAMix optimizer proposed by @Apple researchers @MatPagliardini, @GrangierDavid, and @PierreAblin.</li><li><a href="https://x.com/micuelll/status/1838244638873809125)">Tweet from Miquel Farré (@micuelll)</a>: Curious about how FineVideo was built? 🍿 We open sourced the whole scraping and processing scripts to convert ~2M YouTube videos into a rich, annotated dataset for training video foundation models. R...</li><li><a href="https://x.com/tomaarsen/status/1837132943728209921)">Tweet from tomaarsen (@tomaarsen)</a>: I&#39;ve just shipped the Sentence Transformers v3.1.1 patch release, fixing the hard negatives mining utility for some models. This utility is extremely useful to get more performance out of your emb...</li><li><a href="https://x.com/davidberenstei/status/1838482286523601339)">Tweet from David Berenstein (@davidberenstei)</a>: Why is it important to look at your synthetic data, even when using synthetic data?  DataCraft UX update.   Data may contain quirks, like repeated prompts, too difficult phrasing and markdown formats,...</li><li><a href="https://x.com/gabrielmbmb_/status/1838239658737549797)">Tweet from Gabriel Martín Blázquez (@gabrielmbmb_)</a>: Curious about what you can do with the 21M personas in FinePersonas? One use case is creating completely novel datasets—like I just did!   FinePersonas Synthetic Email Conversations ✉️  Using distilab...</li><li><a href="https://x.com/Gradio/status/1838210842497560971)">Tweet from Gradio (@Gradio)</a>: 🔥 Diffusers fast Inpaint by @OzzyGT  Draw the mask over the subject you want to erase or change and write what you want to Inpaint it with. Create interesting art pieces with Diffusers and Gradio 😎</li><li><a href="https://enterprise.wikimedia.com/blog/hugging-face-dataset/)">Wikipedia Dataset on Hugging Face: Structured Content for AI/ML</a>: Wikimedia Enterprise releasing Wikipedia dataset on Hugging Face, featuring Structured Contents beta from Snapshot API for AI and machine learning applications</li><li><a href="https://x.com/qlhoest/status/1837179483201147279)">Tweet from Quentin Lhoest 🤗 (@qlhoest)</a>: FinePersonas is the richest personas dataset  And now you can ReWrite it to adapt the personas to your needs  (works on any dataset on HF!)
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1287857707075764296)** (117 messages🔥🔥): 

> - `Hugging Face Token Issues`
> - `OpenAI's Word Generation`
> - `Gradio and Model Queries`
> - `Feedback on Voice Channels`
> - `Agents IDE for Hugging Face Tools` 


- **Hugging Face token issues persist**: Several users reported invalid Hugging Face tokens despite multiple attempts to generate new ones, causing frustration across machines.
   - Follow-up discussions indicated potential rate limit issues and suggested reinstalling the huggingface-hub package to troubleshoot, but many still faced the same challenge.
- **OpenAI generates vast amounts of text daily**: A comparison was drawn by a user, highlighting that OpenAI generates about **100 billion words per day**, rivaling the **100 trillion words per day** generated by all humans.
   - This sparked curiosity about whether Hugging Face could match these statistics across its models.
- **Gradio Spaces app issues**: A user reported problems with their Space app's external logs, mentioning both local and network access URLs but experiencing issues with accessing external URLs.
   - Suggestions to troubleshoot include checking configurations, but concrete solutions were not discussed.
- **Voice Channel improvement suggestions**: A member proposed enhancements for voice channels, including more VCs, music bots, and scheduled events to discuss AI news weekly.
   - Community members reacted positively, indicating an interest in improved engagement and functionality within voice channels.
- **Interest in an Agents IDE for Hugging Face tooling**: A member expressed enthusiasm for developing an 'agents IDE' specifically designed for TGI and Hugging Face tools, similar to langgraph-studio.
   - They inquired if there were any ongoing projects or plans for such a tool, offering to assist in the development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...</li><li><a href="https://x.com/sama/status/1756089361609981993">Tweet from Sam Altman (@sama)</a>: openai now generates about 100 billion words per day.  all people on earth generate about 100 trillion words per day.</li><li><a href="https://ai.google.dev/competition/projects/extractcode">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro">MMLU Pro - a Hugging Face Space by TIGER-Lab</a>: no description found</li><li><a href="https://tenor.com/view/cat-dance-dancing-cat-chinese-dancing-cat-funny-cat-meme-cat-gif-18059553370350307210">Cat Dance Dancing Cat GIF - Cat dance Dancing cat Chinese dancing cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eyhrix/anybody_know_of_arx03_topscoring_model_on_mmlu_">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eyhrix/anybody_know_of_arx03_topscoring_model_on_mmlu_pro/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/bBEcB.gif">Dennis Reynolds GIF - Dennis Reynolds Iamgod - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf">flux1-dev-Q4_K_S.gguf · city96/FLUX.1-dev-gguf at main</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1287868297626521691)** (2 messages): 

> - `Neuralink's FP8 Performance`
> - `Mixed Precision Loss Comparison` 


- **FP8 Loss Matches bfloat16**: A member noted that using **1b FP8** matches the loss of **bfloat16 mixed precision**, indicating a close performance relationship.
   - *Today, I confirmed this result* during testing.
- **Neuralink Performance Tracking**: Neuralink is actively working on performance metrics related to precision loss, focusing on **FP8** and **bfloat16**.
   - User feedback highlights the importance of these metrics for optimizing AI modeling efforts.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1288052979051794494)** (8 messages🔥): 

> - `Llama 3.1 Safety Assessment`
> - `Comic Sans FLUX Model`
> - `Qwen/Qwen2.5-72B-Instruct Model`
> - `AI Font Generators`
> - `Neural Computation Paper` 


- **Safety Assessment reveals Llama 3.1 insights**: A team published a [safety assessment](https://www.hydrox.ai/blogs/c9fa272b-f982-4598-8e5c-002f9c019782) on **Llama 3.1**, highlighting that larger models do not necessarily mean safer models.
   - Community members reacted with skepticism, pointing out previous similar findings regarding *ASCII char injection attacks*.
- **Comic Sans font model joins the competition**: Just in time for the [Text-Tacular Showdown Contest](https://civitai.com/articles/7587), a new [FLUX model](https://civitai.com/models/791942/comic-sans-font-for-flux?modelVersionId=885572) allows for accurate recreation of **Comic Sans** font in image generation.
   - This model encourages the use of the much-maligned font in a fun way for various applications, despite its often criticized reputation.
- **Qwen/Qwen2.5-72B-Instruct available**: The **Qwen/Qwen2.5-72B-Instruct** model is now accessible on [Hugging Face](https://huggingface.co/chat/), part of the ongoing effort to provide quality AI chat models to the community.
   - This release includes a recent update on the **Meta-Llama** model with possible enhancements.
- **Historical Neural Computation contributions**: A citation from **Neural Computation** discusses the role of constraints in enhancing learning networks' generalization capabilities, specifically applied to handwritten zip code recognition.
   - The paper features important authors, including **Yann LeCun** and **Bernhard Boser**, who are notable figures in neural networks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.hydrox.ai/blogs/c9fa272b-f982-4598-8e5c-002f9c019782">HydroX AI</a>: no description found</li><li><a href="https://neurosciencenews.com/nlep-ai-language-26329/">no title found</a>: no description found</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:u-x6o8ySG0sC">Backpropagation applied to handwritten zip code recognition</a>: Y LeCun, B Boser, JS Denker, D Henderson, RE Howard, W Hubbard, LD Jackel, Neural computation, 1989 - Cited by 17,404</li><li><a href="https://civitai.com/models/791942/comic-sans-font-for-flux?modelVersionId=885572">Comic Sans Font for Flux - V1 | Stable Diffusion LoRA | Civitai</a>: Just in time for the Text-Tacular Showdown Contest , get an edge on the competition by generating text-laden images using this FLUX model to accura...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1287876157001306172)** (175 messages🔥🔥): 

> - `Hugging Face models`
> - `Audio extraction tools`
> - `Social engineering GPT`
> - `Google Gemini object detection`
> - `Tau LLM training` 


- **New Social Engineering GPT Model**: A user shared a new model called [Social Engineering GPT](https://huggingface.co/spaces/abdurrahman01234/social_engineering_GPT) on Hugging Face, highlighting its effectiveness in cybersecurity applications. They are seeking collaborators for further fine-tuning of the model.
   - This model demonstrates the potential of AI within the cyber security domain, making it a point of interest for enthusiasts and experts alike.
- **YouTube to Audio Python Tool**: A user introduced a [YouTube-to-Audio](https://github.com/jack-tol/youtube-to-audio) package that allows users to extract audio from YouTube videos and playlists easily. The tool supports multiple audio formats and can be installed via pip.
   - This tool streamlines the audio extraction process, eliminating the need for unreliable online converters and thus enhancing user convenience.
- **Gemini Object Detection Demo**: A user showcased a demo for object detection using [Google Gemini](https://huggingface.co/spaces/saq1b/gemini-object-detection) that generates bounding box coordinates from images. This functionality allows for practical testing and exploration of Gemini's capabilities.
   - This demo highlights Gemini's potential in computer vision tasks, providing a seamless way for users to interact with and understand the model's outputs.
- **Community Engagement and Collaboration**: Users expressed interest in collaboration for developing and fine-tuning models, emphasizing the community's drive towards enhancing AI projects. The atmosphere encourages sharing knowledge and seeking assistance in improving existing models.
   - This reflects a growing trend in the community aimed at leveraging collective expertise to push the boundaries of AI capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/saq1b/gemini-object-detection">Gemini Object Detection - a Hugging Face Space by saq1b</a>: no description found</li><li><a href="https://huggingface.co/spaces/abdurrahman01234/social_engineering_GPT">Social Engineering GPT - a Hugging Face Space by abdurrahman01234</a>: no description found</li><li><a href="https://youtube.com/live/ZyiH7IkBr_w?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 21</a>: **Welcome back to our Tau LLM series! 🌟**In this episode, we&#39;re diving into our fourth training attempt, known as **Series D**. Here&#39;s what we have planned:...</li><li><a href="https://tenor.com/view/3po-star-wars-this-is-madness-gif-13899583">3po Star Wars GIF - 3po Star Wars This Is Madness - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/Unity-Technologies/ml-agents/blob/develop/LICENSE.md">ml-agents/LICENSE.md at develop · Unity-Technologies/ml-agents</a>: The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source project that enables games and simulations to serve as environments for training intelligent agents using deep reinforcement ...</li><li><a href="https://github.com/Unity-Technologies/UnityCsReference">GitHub - Unity-Technologies/UnityCsReference: Unity C# reference source code.</a>: Unity C# reference source code. Contribute to Unity-Technologies/UnityCsReference development by creating an account on GitHub.</li><li><a href="https://github.com/jack-tol/youtube-to-audio">GitHub - jack-tol/youtube-to-audio: A lightweight Python package and command-line interface (CLI) tool that extracts audio from YouTube videos and playlists in multiple formats, such as MP3, WAV, OGG, AAC, and FLAC.</a>: A lightweight Python package and command-line interface (CLI) tool that extracts audio from YouTube videos and playlists in multiple formats, such as MP3, WAV, OGG, AAC, and FLAC. - jack-tol/youtub...</li><li><a href="https://pypi.org/project/youtube-to-audio/">youtube-to-audio</a>: A lightweight Python package and command-line interface (CLI) tool that extracts audio from YouTube videos and playlists in multiple formats.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1288164470144634880)** (3 messages): 

> - `HF Dataset`
> - `Cross-Posting Etiquette` 


- **Upcoming HF Dataset Targets Japan and US**: A member announced plans to release an [HF dataset](https://link.to.dataset) soon, specifically focusing on both **Japan and the US**.
   - This strategic focus indicates a targeted approach in expanding dataset relevance across different regions.
- **Reminder on Channel Etiquette**: A user gently reminded another not to cross-post and to keep discussions focused on the topic at hand.
   - *Cross-posting* can disrupt the flow of conversation, highlighting the importance of adhering to channel guidelines.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1288181660080406529)** (2 messages): 

> - `GOT OCR 2 Model`
> - `Fine-tuning OCR models`
> - `Text-image datasets`
> - `Language-specific training` 


- **Exploring GOT OCR 2 for Language-Specific Projects**: A member expressed interest in the new **GOT OCR 2** model but noted it wasn't pretrained in their language, indicating a need for **fine-tuning**.
   - They requested guidance and reading material to assist with creating a **text-image dataset** in their specific language for the fine-tuning process.
- **Request for Help with Fine-tuning Process**: The member conveyed a desire for support in fine-tuning **GOT OCR 2** and acknowledged their gratitude in advance for any assistance offered.
   - They are actively seeking recommendations for reading materials and guidance to better understand the required steps.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1288062730351411273)** (7 messages): 

> - `SetFit Models Training`
> - `Daily Topic Modeling`
> - `Sentiment Analysis Methods`
> - `BERTTopic`
> - `Zero-shot Topic Definition` 


- **Exploring Online Services for SetFit Models**: A member inquired about online services suitable for training **SetFit models**.
   - This question reflects growing interest in efficient model training solutions.
- **Challenges in Daily Topic Modeling**: Another member discussed difficulties in determining a sensible number of topics using **BERTTopic**, noting a need for manual merging in production environments.
   - They highlighted the complexity of managing ever-changing data while maintaining topic integrity.
- **Zero-shot Approach for Topic Management**: A different member shared their experience deploying a **zero-shot method** for defining topics, finding success in production with cap limitations on the number of topics.
   - This approach allows for bundling new topics as 'others' or generating names dynamically post-model.
- **Seeking Alternatives for Sentiment Analysis**: Concerns were raised about finding state-of-the-art methods for **sentiment analysis** without solely relying on **OpenAI's** API.
   - This indicates a drive for self-sufficient models beyond outsourced capacities.
- **Continuous Topic Clustering Needs**: A member expressed a desire to cluster topics daily or continuously add new ones, acknowledging their current inexperience with the process.
   - They noted that solutions relying on conditional logic (if-else) were not appealing for their use case.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1287859753615101973)** (14 messages🔥): 

> - `ControlNet_Union in SDXL`
> - `Training DiT models`
> - `Sigma_t term importance`
> - `Denoising processes`
> - `Latent variable equations` 


- **ControlNet_Union's Strict Conditioning**: A user noted that **ControlNet_Union for SDXL** retains empty spaces in output when input is a scribble, prompting a query on addressing this issue.
   - Another member suggested that trimming parts of the image can help the model generate a more cohesive background, particularly when using fill/inpaint/outpaint techniques.
- **The Nuances of Sigma_t in Denoising**: A member questioned the necessity of the **sigma_t** term during the sampling process in their DiT model training, wondering if it affects output coherence.
   - They concluded that using the sigma_t term facilitates step-by-step denoising rather than denoising the entire image all at once, highlighting a practical experience with alternative equations yielding better results.
- **Exploration of Latent Variable Equations**: Discussion centered on different equations for adjusting latent variables, with one user finding modifications that improved results despite deviation from original paper equations.
   - The user expressed uncertainty about their findings and a desire to understand the mathematical representations better, hinting at the complexity of Denoising Diffusion Models.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1287889405935222856)** (2 messages): 

> - `Gradio 5 Beta Release`
> - `Gradio Performance Improvements`
> - `Modern UI Design in Gradio`
> - `AI Playground Feature`
> - `Office Hours Demo` 


- **Gradio 5 Beta launches with excitement**: The team announced that **Gradio 5 (Beta)** is officially out, addressing major developer concerns around performance and usability.
   - *We'd love to get your feedback before we ship Gradio 5 publicly!*
- **Gradio 5 boosts performance with SSR**: **Gradio 5** features significant performance improvements, including server-side rendering (SSR), which enhances app loading speeds.
   - This aims to resolve the ongoing issue of **Gradio loading too slowly** for users.
- **Gradio gets a makeover**: Many components such as **Buttons**, **Tabs**, and **Sliders** have been refreshed with a modern design in Gradio 5.
   - This update addresses the concern that **Gradio looks old-school** and enhances the overall user experience.
- **Introducing the AI Playground in Gradio 5**: **Gradio 5** comes with an experimental **AI Playground** where users can generate and preview Gradio apps directly in the browser.
   - This feature aims to overcome the challenge of **LLMs not knowing Gradio** and encourages interaction with the platform.
- **Join the Gradio Office Hours for a Live Demo**: The team invites users to the office hours for a demonstration of the new **server-side rendering features**.
   - This event is scheduled for **tomorrow at 12:00 pm Eastern time** and will showcase the latest enhancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://5-0-dev.gradio-website.pages.dev/playground">Suspected phishing site | Cloudflare</a>: no description found</li><li><a href="https://huggingface2.notion.site/Gradio-5-A-Production-Ready-Web-Framework-for-ML-Applications-a4d7e42c26f4450aa0758d968019d120?pvs=74)">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1287853726094069830)** (143 messages🔥🔥): 

> - `RWKV Architecture`
> - `YouTube to Audio Tool`
> - `Dynamic Evaluation in ML`
> - `muP Implementation` 


- **Understanding RWKV's Architecture**: Members discussed various aspects of the **RWKV architecture**, particularly its unique features like **ddlerp** and the emphasis on the most recent tokens, highlighting their efficiency compared to **convolutions**.
   - It was noted that understanding **RWKV** requires familiarity with GLA, and while complexities exist, the community believes simplifying its explanation could aid adoption.
- **New YouTube-to-Audio Tool Introduction**: A user announced the creation of a new command-line tool called **youtube-to-audio** that extracts audio from YouTube in various formats including **MP3** and **WAV**.
   - This tool allows customization of output file names and playlist downloads, serving as a simpler alternative to existing methods that are often laden with ads.
- **Discussing Dynamic Evaluation for ML**: A member brought up the concept of **Dynamic Evaluation**, which involves fine-tuning a model directly on the test set, questioning its validity due to concerns over external validity.
   - Though technically valid, this method may not align with typical evaluation practices, emphasizing the need for identical distribution between training and test sets.
- **Progress on muP Implementation**: There has been ongoing work to clarify **muP** (Maximal Update Parameterization) and its application to neural networks, aimed at community adoption despite complexities in its mathematics.
   - Publishing simpler implementations alongside theoretical explanations is seen as crucial to facilitate understanding, making it easier for developers to integrate muP into their frameworks.
- **Comparison with yt-dlp Tool**: A discussion emerged around existing tools for downloading audio from YouTube, highlighting **yt-dlp** as a feature-rich downloader already available.
   - This tool was recommended as an alternative alongside the newly introduced **youtube-to-audio**, further enriching the options for users seeking audio extraction solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cloneofsimo/status/1838287517906510026">Tweet from Simo Ryu (@cloneofsimo)</a>: Good stuff. Pro tip: do the red circles i checked will get you 99% there. (but dont scale the head dim) https://blog.eleuther.ai/mutransfer/</li><li><a href="https://blog.eleuther.ai/mutransfer/">The Practitioner&#39;s Guide to the Maximal Update Parameterization</a>: Exploring the implementation details of mutransfer</li><li><a href="https://index.commoncrawl.org/">Tweet from Common Crawl Index Server</a>: no description found</li><li><a href="https://huggingface.co/blog/rwkv">Introducing RWKV - An RNN with the advantages of a transformer</a>: no description found</li><li><a href="https://arxiv.org/abs/2101.06804">What Makes Good In-Context Examples for GPT-$3$?</a>: GPT-$3$ has attracted lots of attention due to its superior performance across a wide range of NLP tasks, especially with its powerful and versatile in-context few-shot learning ability. Despite its s...</li><li><a href="https://github.com/cloneofsimo/zeroshampoo/blob/main/distributed_shampoo.py">zeroshampoo/distributed_shampoo.py at main · cloneofsimo/zeroshampoo</a>: Contribute to cloneofsimo/zeroshampoo development by creating an account on GitHub.</li><li><a href="https://github.com/jack-tol/youtube-to-audio">GitHub - jack-tol/youtube-to-audio: A lightweight Python package and command-line interface (CLI) tool that extracts audio from YouTube videos and playlists in multiple formats, such as MP3, WAV, OGG, AAC, and FLAC.</a>: A lightweight Python package and command-line interface (CLI) tool that extracts audio from YouTube videos and playlists in multiple formats, such as MP3, WAV, OGG, AAC, and FLAC. - jack-tol/youtub...</li><li><a href="https://pypi.org/project/youtube-to-audio/">youtube-to-audio</a>: A lightweight Python package and command-line interface (CLI) tool that extracts audio from YouTube videos and playlists in multiple formats.</li><li><a href="https://github.com/google-research/google-research/blob/master/scalable_shampoo/jax/shampoo.py">google-research/scalable_shampoo/jax/shampoo.py at master · google-research/google-research</a>: Google Research. Contribute to google-research/google-research development by creating an account on GitHub.</li><li><a href="https://github.com/yt-dlp/yt-dlp">GitHub - yt-dlp/yt-dlp: A feature-rich command-line audio/video downloader</a>: A feature-rich command-line audio/video downloader - yt-dlp/yt-dlp
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1287851153417310208)** (43 messages🔥): 

> - `Planning capabilities of LLMs`
> - `Interpretability in AI`
> - `FP6 floating-point format performance`
> - `Scaling laws in ML`
> - `Implicit instruction tuning` 


- **OpenAI's LRM Showcasing Planning Abilities**: Discussions highlighted that while OpenAI's recent model, o1 (Strawberry), claims to be a Large Reasoning Model (LRM), its effectiveness is debated, especially its low accuracy under certain test conditions.
   - Members noted that *'inference time compute takes it from 0% to above 50% on a preview model,'* raising questions about its planning capabilities.
- **Critique on AI Interpretability Papers**: A member shared a paper critiquing many interpretability methods in AI for lacking meaningful insights and making statistical errors without proper evaluation.
   - The paper indicates that *'feature attribution explanations provide marginal utility in our task for a human decision maker'*.
- **FP6 Format Surpassing BF16 on H100**: Reports emerged stating that FP6 shows equal accuracy to BF16 while being faster than fp8/bf16 on H100, with performance enhancing features packed into vLLM.
   - Alpin Daley claimed *'the throughput is pretty impressive, and the accuracy preservation is on par with FP8,'* pointing towards versatile floating-point format utilization.
- **Scaling Laws vs Regularization in ML**: A recent study questions whether established regularization principles are still relevant in the era dominated by scaling large language models (LLMs).
   - The authors suggest a phenomenon called 'scaling law crossover' where traditional principles may not hold, shifting focus from generalization to approximation error.
- **Implicit Instruction Tuning in Language Models**: Findings suggest that training on responses alone can enable instruction following, questioning the necessity of instruction-response pairs for effective model training.
   - This *'implicit instruction tuning'* also reveals that narrow-domain data can still lead to broad instruction-following capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.14254">Instruction Following without Instruction Tuning</a>: Instruction tuning commonly means finetuning a language model on instruction-response pairs. We discover two forms of adaptation (tuning) that are deficient compared to instruction tuning, yet still y...</li><li><a href="https://arxiv.org/abs/2409.15156">Rethinking Conventional Wisdom in Machine Learning: From Generalization to Scaling</a>: The remarkable success of large language pretraining and the discovery of scaling laws signify a paradigm shift in machine learning. Notably, the primary objective has evolved from minimizing generali...</li><li><a href="https://arxiv.org/abs/2409.13373">LLMs Still Can&#39;t Plan; Can LRMs? A Preliminary Evaluation of OpenAI&#39;s o1 on PlanBench</a>: The ability to plan a course of action that achieves a desired state of affairs has long been considered a core competence of intelligent agents and has been an integral part of AI research since its ...</li><li><a href="https://x.com/AlpinDale/status/1838369139288596687">Tweet from Alpin (@AlpinDale)</a>: Somehow, FP6 performs better on benchmarks than BF16.  Soon to land in vLLM. https://github.com/vllm-project/vllm/pull/8751  Quoting Alpin (@AlpinDale)   You can now load any FP16 model in any floatin...</li><li><a href="https://en.wikipedia.org/wiki/Betteridge%27s_law_of_headlines">Betteridge&#039;s law of headlines - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2012.02748">Challenging common interpretability assumptions in feature attribution explanations</a>: As machine learning and algorithmic decision making systems are increasingly being leveraged in high-stakes human-in-the-loop settings, there is a pressing need to understand the rationale of their pr...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1288008158345957451)** (10 messages🔥): 

> - `Chinchilla and matmul algorithm`
> - `Strassen algorithm discussions`
> - `Decomposing models using Strassen`
> - `Low precision matmul performance` 


- **Chinchilla's Relationship with Matmul Algorithm**: Members noted that Chinchilla is fundamentally based on the **matmul algorithm**, with discussions hinting that using a faster variant alters **optimal points** for the model.
   - This led to discussions about performance adjustments by varying precision settings.
- **Strassen Algorithm: Not a Favorite**: A consensus emerged that the **Strassen algorithm** has been discussed multiple times in the server, but it lacks belief in its effectiveness.
   - Some members speculate its potential for **inference** processes, yet skepticism remains prevalent.
- **Strassen-inspired Model Decomposition Idea**: One member suggested that models could potentially be decomposed in a **Strassen-inspired** manner to decrease the number of additive and subtractive operations.
   - This approach could lead to approximating results close to the full model's capabilities.
- **Debate on Low Precision vs Strassen**: A member pointed out that instead of implementing **Strassen**, traditional matmul could be run at lower precision to achieve similar outcomes.
   - This adds to the ongoing debate about the efficacy of Strassen in practical scenarios.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1287862866078334997)** (13 messages🔥): 

> - `MMLU scores for Pythia 6.9b-deduped`
> - `Formatting issues with Pile models`
> - `Performance comparison to ARC`
> - `Reference to forthcoming paper` 


- **Low MMLU Scores for Pythia 6.9b-deduped**: A user expressed concern about receiving very low **MMLU 5-shot scores** with the **Pythia 6.9b-deduped model**, specifically around **26%**.
   - Another member questioned if these scores were much lower than published scores for **Pythia**, prompting a discussion on model performance.
- **Challenges with Formatting in Pile Models**: Members discussed that models trained on the **Pile** struggle specifically with **MMLU** due to poor formatting adherence, impacting their performance.
   - One pointed out that if the style is adjusted to mimic **ARC**, the performance improves significantly.
- **ARC vs. MMLU Performance Discussion**: It was noted that **GPT-NeoX-20B** scores approximately **random** on **ARC easy** when it follows **MMLU's** styling, highlighting formatting's critical role.
   - Despite this, there are substantial performance differences based on formatting styles between the two benchmarks.
- **Seeking References for Formatting Issues**: A user sought citations related to the claim that **Pile models** struggle with formatting, especially from a forthcoming paper.
   - Another member provided a reference to the paper '[Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)', which discusses this issue.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1287875397513511026)** (2 messages): 

> - `trunc_normal initialization`
> - `AllenAI ablation study`
> - `model stability` 


- **Consider switching to trunc_normal initialization**: A discussion initiated about whether to change function initializations to **trunc_normal** to improve model performance.
   - The importance of this change was emphasized due to potential stability issues in large-scale models.
- **AllenAI's ablation study highlights stability**: Referenced [AllenAI's ablation study](https://arxiv.org/abs/2409.02060) indicated that models without trunc_normal showed **instability at scale**.
   - The authors of the study included key researchers such as **Niklas Muennighoff** and **Luca Soldaini**, pointing to serious consequences if trunc_normal is not utilized.



**Link mentioned**: <a href="https://arxiv.org/abs/2409.02060">OLMoE: Open Mixture-of-Experts Language Models</a>: We introduce OLMoE, a fully open, state-of-the-art language model leveraging sparse Mixture-of-Experts (MoE). OLMoE-1B-7B has 7 billion (B) parameters but uses only 1B per input token. We pretrain it ...

  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1287854453281652848)** (122 messages🔥🔥): 

> - `Gemini model updates`
> - `New RoRF open-source release`
> - `Claude model expectations`
> - `Aider installation issues`
> - `Prompt caching functions` 


- **Gemini models see major updates**: New production Gemini models, **Gemini-1.5-Pro-002** and **Gemini-1.5-Flash-002**, were announced, featuring over **50% price drop** and increased rate limits.
   - Developers are impressed with the performance improvements, although the benchmarks appear static for coding tasks.
- **Exciting open-source launch of RoRF**: The **Routing on Random Forest (RoRF)** has been open-sourced, surpassing previous methods and introducing **12 pre-trained model routers**.
   - This release is celebrated for enhancing performance on MMLU, opening new avenues for model routing.
- **Anticipation for new Claude models**: There is speculation about the release of new Claude models, specifically **Haiku 3.5** or **Opus**.
   - Users express frustration over the duration since the last update, with hopes for an announcement soon.
- **Challenges with Aider installation**: Some users report issues with Aider, leading to attempts to uninstall and reinstall using **pipx**, but functionality remains broken.
   - Suggestions such as reverting to older versions have been offered, highlighting the challenges faced by the community.
- **Prompt caching considerations**: Prompt caching features are discussed, specifically how it applies to the **Anthropic API** and **OpenRouter**.
   - Users seek clarification on configuring **AIDER_CACHE_KEEPALIVE_PINGS** and its effects in different environments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...</li><li><a href="https://tenor.com/view/side-eye-cat-gif-8216273864367202904">Side Eye Cat GIF - Side eye cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching">Prompt Caching (beta) - Anthropic</a>: no description found</li><li><a href="https://status.anthropic.com/incidents/4k8gmdx11lcq">Partial Outage on 3.5 Sonnet</a>: no description found</li><li><a href="https://x.com/OfficialLoganK/status/1838611055217385646">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Two new production Gemini models, &gt;2x higher rate limits, &gt;50% price drop on Gemini 1.5 Pro, filters switched to opt-in, updated Flash 8B experimental model, and more.   It’s a good day to be a ...</li><li><a href="https://openrouter.ai/docs/prompt-caching">Prompt Caching | OpenRouter</a>: Optimize LLM cost by up to 90%</li><li><a href="https://x.com/OpenAIDevs/status/1838611640201162807">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Updates to OpenAI o1 API availability:  - We’ve expanded access to developers on tier 4 (100 requests per minute for both models). - We’ve 5x’d rate limits for developers on tier 5 (1000 requests per ...</li><li><a href="https://x.com/testingcatalog/status/1838358531579285511?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Looks like new Gemini models are dropping tomorrow and not Opus 3.5 👀👀👀  Quoting ʟᴇɢɪᴛ (@legit_rumors)   new updated Gemini 1.5 models might be ready to ship soon™ 🚀  good chance for more than jus...</li><li><a href="https://tenor.com/view/its-just-gambling-liam-scott-edwards-ace-trainer-liam-betting-gamble-gif-20475304">Its Just Gambling Liam Scott Edwards GIF - Its Just Gambling Liam Scott Edwards Ace Trainer Liam - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.anthropic.com/incidents/4hc6130xwxt5">Elevated errors and latency on the Anthropic API</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=1iolYLlUBiI">Zinley Berkeley Lecture Demo</a>: Full load from multi-agent using terminal. UI version coming soon.</li><li><a href="https://x.com/tomas_hk/status/1838586544657240234">Tweet from Tomas Hernando Kofman (@tomas_hk)</a>: Today we&#39;re open-sourcing RoRF (Routing on Random Forests), a pairwise model router that beats all closed and open-source approaches, along with 12 pre-trained model routers:  Hugging Face: http:/...</li><li><a href="https://cloudonair.withgoogle.com/events/gemini-at-work-24">Gemini at Work</a>: Join Google Cloud CEO Thomas Kurian and industry leaders to discover how AI is reshaping businesses across the globe.</li><li><a href="https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/">Updated production-ready Gemini models, reduced 1.5 Pro pricing, increased rate limits, and more</a>: no description found</li><li><a href="https://github.com/Not-Diamond/RoRF">GitHub - Not-Diamond/RoRF: Routing on Random Forest (RoRF)</a>: Routing on Random Forest (RoRF). Contribute to Not-Diamond/RoRF development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/commit/7fa1620f58132ec085a7939a8015bbe7935827a2">feat: Allow flexible matching of 5-9 characters in SEARCH/REPLACE blo… · paul-gauthier/aider@7fa1620</a>: …ck prefixes</li><li><a href="https://github.com/paul-gauthier/aider/issues/1697">&lt;no response&gt; Bug · Issue #1697 · paul-gauthier/aider</a>: Issue C:\Users\pierr\Desktop\Github\claude-3-artifacts&gt;aider --model openrouter/anthropic/claude-3.5-sonnet --no-pretty Aider v0.57.1 Main model: openrouter/anthropic/claude-3.5-sonnet with diff ed...</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5">Gemini Flash 1.5 - API, Providers, Stats</a>: Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video...</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 - API, Providers, Stats</a>: Google&#x27;s latest multimodal model, supporting image and video in text or chat prompts.  Optimized for language tasks including:  - Code generation - Text generation - Text editing - Problem solvin...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1287851745229148250)** (62 messages🔥🔥): 

> - `Aider File Operations`
> - `Using Models in Aider`
> - `Upgrading Aider`
> - `HuggingChat Models`
> - `Aider Usage Tutorials` 


- **Managing Read-Only Files in Aider**: Users can add multiple read-only files for Aider using the `AIDER_READ` configuration, allowing for efficient organization of documentation.
   - Using the `/tokens` command confirms which read-only files have been added and their count, making it clear for beginners.
- **Engaging Weak Model in Aider**: Currently, there is no way to switch to a weaker model on-the-fly without using the `/model switch` command, as mentioned by community members.
   - Some users noted the potential cost-saving benefits of using a lower-powered model for simple questions.
- **Upgrade Procedure for Aider**: Users reported issues upgrading from version 0.56 to 0.57 using pipx, possibly due to caching an older version.
   - The suggested commands for upgrading include `pipx upgrade aider-chat` or completely reinstalling using `pipx uninstall` and `install`.
- **Accessing HuggingChat Models**: Members discussed the availability of HuggingChat models via API, noting the performance of paid vs free models.
   - One user shared a link to LiteLLM that supports different types of Hugging Face models for API access, enhancing usability in Aider.
- **Aider Usage Tutorials and Resources**: A variety of tutorial videos were shared to assist new users in configuring and utilizing Aider effectively.
   - Resources include links to YouTube tutorials on setting up Aider and building applications, facilitating community knowledge sharing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/repomap.html#optimizing-the-map">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: Intro and tutorial videos made by aider users.</li><li><a href="https://simonwillison.net/2024/Sep/20/introducing-contextual-retrieval/">Introducing Contextual Retrieval</a>: Here&#x27;s an interesting new embedding/RAG technique, described by Anthropic but it should work for any embedding model against any other LLM. One of the big challenges in implementing semantic sear...</li><li><a href="https://huggingface.co/chat/models">HuggingChat - Models</a>: Browse HuggingChat available models</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-turn-on-the-repository-map">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://docs.litellm.ai/docs/providers/huggingface">Huggingface | liteLLM</a>: LiteLLM supports the following types of Hugging Face models:</li><li><a href="https://github.com/paul-gauthier/aider/pull/1688">fix: improve automatic upgrade flow for aider by fry69 · Pull Request #1688 · paul-gauthier/aider</a>: fix #1687 (in spirit)  documented installation routine assured that the application exits after upgrading and does not continue running introduced Config class/module to get access to parsed comman...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1287859125363015710)** (5 messages): 

> - `Agentic behavior`
> - `OpenRouter integration`
> - `Gemini model updates` 


- **Discussion on Agentic Behavior in RAG**: A member highlighted that many invested in **RAG** due to confusion surrounding **Agentic behavior**.
   - Another member agreed, adding a light-hearted remark.
- **OpenRouter Models Now Work in Cursor**: A member announced that **OpenRouter models** are now compatible with **Cursor**, thanking them for the fix.
   - This integration now supports all models including those from **Anthropic**.
- **New Details on Gemini Models Released**: A write-up by Simon Willison discussed **two new Gemini models**, `gemini-1.5-pro-002` and `gemini-1.5-flash-002`, noting their benchmarks and updates.
   - The **Pro model** sees a significant price drop effective October 1st, with input costs reducing from **$3.50** to **$1.25/million** tokens for inputs below **128,000 tokens**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1838627801072562232">Tweet from OpenRouter (@OpenRouterAI)</a>: Thank you @cursor_ai for fixing this!  OpenRouter now works in Cursor with all models, including Anthropic 🍾</li><li><a href="https://simonwillison.net/2024/Sep/24/gemini-models/">Updated production-ready Gemini models</a>: Two new models from Google Gemini today: `gemini-1.5-pro-002` and `gemini-1.5-flash-002`. Their `-latest` aliases will update to these new models in &quot;the next few days&quot;, and new `-001` suffi...
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1288207496518959165)** (1 messages): 

> - `Advanced Voice rollout`
> - `Custom Instructions update`
> - `Improved Accents`
> - `New Voices Feature`
> - `Multilingual Capabilities` 


- **Advanced Voice Rolls Out to Plus and Team Users**: The **Advanced Voice** feature is rolling out to all **Plus** and **Team** users in the ChatGPT app this week, enhancing user experience.
   - Users can look forward to additional features, including [Custom Instructions](https://x.com/OpenAI/status/1838642444365369814) and **Memory** functionalities.
- **Exciting New Voice Features Added**: The update includes the addition of **five new voices** to the ChatGPT app, allowing for greater personalization.
   - Users can now enjoy **improved accents**, further enhancing communication capabilities.
- **Advanced Voice Can Speak in 50+ Languages**: The new **Advanced Voice** feature can express the phrase “Sorry I’m late” in over **50 languages**, showcasing its multilingual capabilities.
   - This opens up diverse interaction possibilities for users around the globe.



**Link mentioned**: <a href="https://x.com/OpenAI/status/1838642444365369814">Tweet from OpenAI (@OpenAI)</a>: Advanced Voice is rolling out to all Plus and Team users in the ChatGPT app over the course of the week.  While you’ve been patiently waiting, we’ve added Custom Instructions, Memory, five new voices,...

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1287851889379115160)** (108 messages🔥🔥): 

> - `Advanced Voice Mode`
> - `Voice Generation Performance`
> - `Voice Assistant Competition`
> - `Roleplaying AI Services`
> - `GPU Server Rentals` 


- **Advanced Voice Mode Rollout Confusion**: The rollout of **Advanced Voice Mode** has led to frustration, especially in Europe where users are still waiting for access, with many expressing disappointment over limitations and restrictions.
   - *“It’s just a far cry from the demos in May,”* commented a user, highlighting the difference between expected features and actual capabilities.
- **Debate on Voice Generation Capabilities**: Users are critiquing the current performance of voice generation models, stating they lack the promised flexibility in changing voices and emotional expression.
   - One user noted they managed to get the voice to hum, despite it claiming it couldn't do so, revealing inconsistencies in the safety guidelines.
- **Voice Assistant Competition with Google**: Discussion pointed out that OpenAI is trying to compete with corporate bots like **Google's Assistant**, which leads to a safety-first approach that restricts more dynamic functionalities.
   - A user remarked on their perception that OpenAI is balancing corporate competition against building roleplaying AI products, like those seen on character.ai.
- **GPU Server Rental Recommendations**: Users shared recommendations for renting GPU servers for short-term needs, with options like **Vast.ai** and **salad.com** bought up as affordable choices.
   - One mentioned how utilizing sponsored links from YouTubers can provide significant credits for one-off rentals, particularly useful for training models.
- **Excitement for AI Roleplaying Services**: Covert mentions of character.ai have sparked interest as users expressed surprise at its popularity and capabilities in roleplaying scenarios.
   - It was highlighted that many users initially encountered rejections when trying out these AI services, leading some to form a DIY community approach.



**Link mentioned**: <a href="https://fixupx.com/OpenAI/status/1838642448375206345">Tweet from OpenAI (@OpenAI)</a>: Meet the five new voices.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1288015657903853578)** (5 messages): 

> - `Voice Function in GPT`
> - `Calling GPTs` 


- **Clarification on Voice Calls to GPTs**: A user inquired if it's possible to call a GPT over the voice function, but the response was a resounding **no**.
   - Another member insisted that **ChatGPT** cannot make calls to other GPTs, and highlighted their suggestion to enable this feature in the suggestions channel.
- **Suggestion to Enable GPT Calls**: One member mentioned they made a suggestion for **ChatGPT** to allow calling other GPTs via the voice function.
   - This highlights a demand for increased interactivity within the chatbot functionalities.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1287935677115273236)** (21 messages🔥): 

> - `Prompt engineering for API`
> - `Structured output for JSON`
> - `Generating Minecraft questions`
> - `Hallucination issues`
> - `API usage queries` 


- **Difficulty with JSON formatted responses**: A member expressed frustration over the quality of JSON formatted answers, stating that sometimes the response is just a simple '{'.
   - They suggested that using a structured output could potentially improve the format.
- **Clarifying prompt engineering**: Discussion involved defining prompt engineering, highlighting the importance of clearly stating requests and providing enough context.
   - One member noted that prompts should be detailed to help the model produce better responses.
- **Creating engaging Minecraft questions**: A member shared their prompt aimed at generating fun Minecraft-related questions in JSON format for in-game chat.
   - They mentioned challenges with repetitive outputs and sought advice on improving the prompt.
- **Issues with hallucination and response quality**: Concerns were raised about the model's tendency to hallucinate when increasing the temperature setting above 1.25.
   - A member noted that instructing the model not to hallucinate seemed to minimize issues of deviating from the expected output.
- **Enforcing JSON output with API**: A member inquired about methods to ensure the API consistently generates JSON formatted outputs when using the raw HTTPS API.
   - They clarified that they are not using any libraries or wrappers, which may affect output consistency.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1287935677115273236)** (21 messages🔥): 

> - `JSON formatting challenges`
> - `Prompt engineering for AI`
> - `Minecraft question generation`
> - `API usage insights`
> - `Avoiding hallucinations in responses` 


- **Frustrations with JSON formatted answers**: A user expressed concerns that their JSON formatted responses were often poor quality or incomplete, sometimes resulting in only a '{'.
   - Another member suggested that using more structured output could enhance the response quality.
- **Understanding Prompt Engineering**: Members discussed the concept of prompt engineering, emphasizing the need to clearly state requirements and context to improve output.
   - A user noted that providing specific examples within their prompt could enhance the questions generated.
- **Generating Minecraft-related Questions**: A user shared their prompt designed for creating engaging Minecraft-related questions through the API, seeking feedback on its effectiveness.
   - They reported challenges with getting diverse questions and noted that increasing temperature led to hallucinations in the generated content.
- **Using the API in Raw Format**: One participant clarified that they were using the raw HTTPS API without any libraries or wrappers for their prompts.
   - They inquired about methodologies to ensure JSON outputs from the API effectively.
- **Concerns Over Hallucinations in Output**: Discussion included skepticism about the effectiveness of prompt instructions intended to prevent hallucinations in AI responses.
   - Users shared experiences where generic instructions failed to limit irrelevant or stereotypical content in generated questions.


  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1287944793279696956)** (7 messages): 

> - `Old server icon returns`
> - `Community reflections`
> - `Scam links present`
> - `Tool discussions` 


- **Old server icon makes a comeback**: Members noted that the **old server icon** is back, sparking nostalgia among the users.
   - One member remarked, *'Back to our roots,'* highlighting the significance of the change.
- **Community feels mixed about changes**: A member expressed a feeling of *'suffering'* regarding the community's current state.
   - This sentiment reflects some dissatisfaction with recent developments.
- **Warning about scam links**: A member alerted others about a **scam link** that was posted in the channel, urging caution.
   - This notice serves as a reminder for community vigilance against potential threats.
- **Newcomer inquiries about tools**: A new member inquired, *'what tool is this?'* showing their eagerness to understand the community's resources.
   - In response, the member **hy3na_xyz** mentioned **btop** as the tool in question.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

mobicham: Moving this conversation to the hqq channel
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1287898235020640297)** (30 messages🔥): 

> - `CUDA Caching Allocator`
> - `Triton Kernel Support`
> - `Segment Anything Model 2 (SAM2)`
> - `Debugging Torch Distributed Training`
> - `Imitation Learning with SAM2-fast` 


- **CUDA Caching Allocator and Tensor Alignment**: A discussion arose regarding the **CUDA** caching allocator's minimum block size of **512 bytes**, confirming that it returns aligned addresses but questioning tensor alignment in **PyTorch**.
   - One member shared a code snippet demonstrating tensor slicing with a reference to tensor alignment in **PyTorch**.
- **Triton Kernel Issues with Torch.compile**: Challenges were noted with **torch.compile** breaking when using custom **Triton** kernels that employ `prune_configs_by` in autotune, prompting a possible pre-pruning workaround.
   - Members discussed the need for opening issues to address this limitation and to ensure proper routing for user-written **Triton** kernels.
- **Interest in Segment Anything Model 2 (SAM2)**: Members expressed interest in **SAM2**, with discussions on its applications in interactive object selection and image segmentation annotation.
   - One member proposed exploring a **SAM2-fast** version tailored for user needs, highlighting its potential for collaborative efforts.
- **Debugging Torch Distributed Training Code**: A member inquired about best practices for debugging **torch.distributed** training code, emphasizing the need for local simulation capabilities without a GPU.
   - They sought advice on using breakpoints and visualizing parallelism structures but found no satisfactory solutions.
- **Imitation Learning Using SAM2-fast**: An intriguing idea was presented about utilizing **SAM2-fast** as input for a **Diffusion Transformer Policy** for imitation learning, involving sensor data to robotic arm joint positions.
   - This sparked interest in further exploring applications of SAM2 in the robotics field.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch-labs/segment-anything-fast">GitHub - pytorch-labs/segment-anything-fast: A batched offline inference oriented version of segment-anything</a>: A batched offline inference oriented version of segment-anything - pytorch-labs/segment-anything-fast</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use th...</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>: A native PyTorch Library for large model training. Contribute to pytorch/torchtitan development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1287894159876948141)** (1 messages): 

> - `GPU MODE`
> - `CUDA MODE Origins`
> - `IRL Meetup Success`
> - `Community Growth`
> - `Future of GPU Programming` 


- **CUDA MODE transitions to GPU MODE**: The community formerly known as **CUDA MODE** has renamed itself to **GPU MODE**, reflecting its broader focus on GPU programming beyond just CUDA.
   - This transformation aims to foster an inclusive environment where members can learn, collaborate, and innovate in GPU technology.
- **From Reading Group to 9,000 Members**: **CUDA MODE** started as a reading group for the **PMPP book** but has grown to over **9,000 members** creating over **10 open source projects** like *torchao* and *Liger*.
   - The community has demonstrated significant contributions to the open-source ecosystem, showcasing its commitment to collaboration.
- **Massive Success at IRL Meetup**: The first in-person meetup attracted **150 hackers**, leading to the development of **over 40 projects** in a single day, illustrating the vibrant community spirit.
   - Participants worked on innovative projects, including porting **PyTorch FlexAttention** and optimizing CUDA kernels, showcasing their skills.
- **Deep Focus and Distraction: A Community Identity**: The phrase 'CUDA MODE' originated from a viral talk by **Tim Dettmers**, emphasizing the power of deep focus in programming but the community now values social interaction too.
   - Members enjoy collaborating and experimenting together, creating a supportive space for exploration in GPU programming.
- **Embracing Broader GPU Programming Ideals**: The transition to **GPU MODE** reflects a commitment to embrace various programming languages and frameworks beyond CUDA, such as **Triton** and **WebGPU**.
   - Community leaders expressed a desire to expand the discussion to include technologies like **Groq** and **TPUs**, encouraging growth in the performance space.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1288168618340651008)** (4 messages): 

> - `CUDA Programming Course`
> - `Nvidia GPUs`
> - `High-Performance Computing` 


- **New CUDA Programming Course Released**: The YouTube video titled ["CUDA Programming Course – High-Performance Computing with GPUs"](https://www.youtube.com/watch?v=86FAWCzIe_4&ab_channel=freeCodeCamp.org) was released, focusing on programming with Nvidia CUDA for high-performance computing and deep learning.
   - The associated [code repository](https://github.com/Infatoshi/cuda-course) is also available, providing hands-on material for learners.
- **Community Curiosity About Course Quality**: A member asked if anyone had tried the newly released CUDA course and whether it was any good.
   - Another member confidently affirmed the course's quality, stating, *'it's good I built it. just came out today.'*



**Link mentioned**: <a href="https://www.youtube.com/watch?v=86FAWCzIe_4&ab_channel=freeCodeCamp.org">CUDA Programming Course – High-Performance Computing with GPUs</a>: Lean how to program with Nvidia CUDA and leverage GPUs for high-performance computing and deep learning. Code:💻 https://github.com/Infatoshi/cuda-course💻 h...

  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1288253857121570827)** (1 messages): 

> - `Luma job opening`
> - `Performance optimization roles`
> - `Dream Machine product`
> - `Luma's research team` 


- **Luma seeks top-tier performance engineers**: Luma is looking for strong engineers to optimize their training and inference stack, specifically for multimodal foundation models, starting with [Dream Machine](https://lumalabs.ai/dream-machine). They offer in-person roles in Palo Alto, with visa sponsorship available for exceptional remote candidates.
- **Deep expertise required for critical roles**: Candidates should have deep experience in large-scale distributed training, low-level kernels like **Triton** and **CUDA**, or optimizing distributed inference workloads for throughput and latency. This emphasizes the need for expertise in debugging and compilation.
- **Impressive growth and backing for Luma**: Luma boasts a strong diffusion research team and has achieved **1 million users in just 4 days** with Dream Machine, demonstrating extreme product market fit. They are backed by a16z and have ample cash runway for growth.
- **Fast-paced and lean work environment**: The company focuses on rapid development and ownership with minimal bureaucracy, allowing for swift project execution. They emphasize efficiency in building and shipping their product.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1287911271076728906)** (8 messages🔥): 

> - `Cutlass example discussion`
> - `Porting CUDA to Python`
> - `Custom ops for PyTorch` 


- **Curiosity about Cutlass example**: A member referenced a [Cutlass example](https://github.com/NVIDIA/cutlass/blob/main/examples/13_two_tensor_op_fusion/README.md) that might fit the description, noting they haven't read into it closely.
   - Another member expressed interest in its performance benefits despite finding the explanation aimed at **Cutlass users** rather than **CUDA beginners**.
- **Porting standalone CUDA code to Python**: A member inquired about options for **porting standalone CUDA code** to Python.
   - In response, another member suggested using **load_inline** from PyTorch as a simple solution.
- **Best practices for CUDA to PyTorch conversion**: After discussing initial options, a member asked about the **best practice** for wrapping CUDA with Python.
   - A suggestion was provided to explore **custom ops** using [this resource](https://github.com/pytorch/ao/tree/main/torchao/csrc) in PyTorch.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/cutlass/blob/main/examples/13_two_tensor_op_fusion/README.md">cutlass/examples/13_two_tensor_op_fusion/README.md at main · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/csrc">ao/torchao/csrc at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1288070575054520380)** (10 messages🔥): 

> - `Slice operation for uintx`
> - `Padding discussion for uintxTensor`
> - `Test example for uintx slicing`
> - `Divisibility requirement for tensors` 


- **Attempt to Add Slice Operation for uintx**: A member is trying to add a slice operation and tests it using code that checks for divisibility by 8 in the `pack` function.
   - *I know there was a lot of discussion around padding and non-padding.* The member questioned whether padding should be handled at the `UintxTensor` or within the bitpacking function.
- **Issues with Slicing uintx Tensor**: The member noted that attempting to slice with `x[2:6]` is not functioning as intended.
   - An example was provided to demonstrate attempting to slice a `uintx` tensor, highlighting challenges faced.
- **Concerns Regarding Padding Implementation**: The member is considering implementing padding in the `pack` function and reflects on its cleanliness versus implications.
   - They asked for suggestions, indicating uncertainty about the potential effects of this adjustment.
- **Divisibility Requirements for Larger Tensors**: A member pointed out that the use case for `uintx` was designed for large tensors, indicating that divisibility by 8 is a relaxed requirement.
   - This highlights a potential flexibility regarding the shape dimension restriction.
- **Consideration for Larger Slicing**: Another member suggested trying a larger slice for sub-byte data types, arguing it might be reasonable to restrict shapes to be divisible by 8.
   - *For sub byte dtypes seems reasonable to restrict the shape dim to be divisible by 8 so we can always pack them?*


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1287905097468219402)** (18 messages🔥): 

> - `CUDA Mode vs GPU Mode`
> - `Heterogenous Computing Discussions`
> - `Segment Anything Model 2`
> - `Nickname Proposals for GPU Mode`
> - `Mascots for GPU Mode` 


- **Debate on CUDA Mode vs GPU Mode**: Members expressed mixed feelings about the change from **CUDA Mode** to **GPU Mode**, agreeing that neither name flows well with alternatives such as 'Heterogeneous Computing'.
   - *CUDA Mode* might be easier to say, but there are **concerns** that **GPU Mode** oversimplifies a broader range of processing units.
- **Introducing Segment Anything Model 2**: [Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2) repository offers code for running inference with Meta's model, including links for model checkpoints and example notebooks.
   - One member highlighted this tool's potential with a shared link to its GitHub page and its relevant features.
- **Creative Nickname Suggestions for GPU Mode**: Several suggestions surfaced for renaming **GPU Mode**, with options like **Parallel Mode** and **Accel Mode** generating laughs among users.
   - Interestingly, some members proposed monikers like *Gigachad Processing* and *Generic Processing Unit*, showing a humorous creativity in the discussion.
- **Mascot Ideas for GPU Mode**: Members discussed the notion of mascots for **GPU Mode**, with suggestions such as **Goku** and the humorous *H100 purse* being thrown into the mix.
   - This playful angle aims to infuse more character into the new name, highlighting the community's desire for a fun identity.
- **History of the GPU Name**: A fun fact emerged regarding the **GPU** name, recalling its original stance as **Geometry Processor Unit** before being popularized as **Graphics Processing Unit**.
   - One user clarified that JHH coined the term GPU, causing some members to rethink their previous understanding of its history.



**Link mentioned**: <a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use th...

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1288237546698182707)** (2 messages): 

> - `Triton Puzzles`
> - `PMP Book and Machine Learning` 


- **Inquiry about Cuda Alternatives to Triton Puzzles**: *Is there anything like Triton puzzles but for raw CUDA?* A member expressed curiosity about similar resources targeting CUDA programming.
   - This inquiry highlights a gap in readily available resources for CUDA that mirror the functionality of Triton puzzles.
- **PMP Book Lacks Machine Learning Focus**: A member commented that the **PMP book** does not mention **machine learning** at all, expressing disappointment.
   - *To be honest, it doesn't really touch on any ML topics,* indicating a perceived disconnect between project management frameworks and evolving tech trends.


  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1288185066534469806)** (5 messages): 

> - `3-bit attention kernel`
> - `Performance drop-off below 4 bits`
> - `Quantization in Attention Modules`
> - `GemLite for HQQ backend` 


- **Pushing for a 3-bit Attention Kernel**: A member noted diminishing returns in performance after **4-bit** precision in attention modules, emphasizing that **3-bit** performance is satisfactory.
   - This led to advocacy for implementing a **3-bit kernel** for improved efficiency.
- **Steep Drop-off in Performance Below 4 Bits**: Discussion indicated that the **performance drop-off is very steep** below **4 bits**, with concerns raised about the impact on model output.
   - Members are analyzing how this affects overall model effectiveness and whether adjustments are needed.
- **Challenges of Quantizing Attention Modules**: Attention modules are more complex to quantize at lower **n-bits**, and they are generally smaller than **MLP layers**, which should be compressed more aggressively.
   - There’s a consensus that focusing on **MLP layer compression** could lead to significant performance gains.
- **Investigating GemLite for Speed Improvements**: A member is currently working on adding **GemLite** as a backend for **HQQ** to assess its impact on end-to-end speed.
   - The outcome of this integration remains to be seen, but it is viewed as a potential enhancement for overall performance.


  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1288199427521122388)** (9 messages🔥): 

> - `repkv Kernel Integration`
> - `RoPE Implementation Details`
> - `Testing Framework Enhancements`
> - `RMSNorm Updates`
> - `SwigLU Modifications` 


- **repkv Kernel added to Llama 3**: The **repkv kernel** has been integrated into the **Llama 3** branch, paving the way for further advancements.
   - A member highlighted the need for clarification regarding tensor shapes for **q** and **k** in relation to the RoPE pull request.
- **Testing RoPE Integration Thoroughly**: There is a strong emphasis on properly integrating **RoPE** into the **dev/cuda** environment, ensuring equality tests between CPU and GPU implementations.
   - Additional testing for **swiglu** and **rmsnorm** is also deemed essential for consistency across implementations.
- **Additional Files Required for RoPE**: The plans call for adding several necessary files to **dev/cuda**, including **rope.cuh** for forward and backward passes.
   - The inclusion of **swiglu_forward** and **swiglu_backward** files is needed to maintain code structure while incorporating these functionalities.
- **RoPE Functionality Needs Independence**: To reduce confusion, it’s suggested that **RoPE** functionality be split to operate independently for **Q** and **K**.
   - This change aims to streamline the integration process and enhance clarity among developers.
- **RoPE Scaling Lacking in Current Implementation**: The current implementation of **RoPE** does not support the **RoPE scaling** introduced in version **3.1**.
   - Addressing this oversight will be crucial for maintaining compatibility with recent updates.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1287952766144086108)** (7 messages): 

> - `Training with xformers`
> - `CK GroupedGemm usage`
> - `CUDA and Metal in VSCode`
> - `Llama3 tuning journey` 


- **Training with xformers shows promise**: Members discussed that the **xFormer attention backend** operates with Composable Kernel, ensuring both **accuracy and speed**.
   - *Good to know, thanks!* remarked one member, acknowledging the information received.
- **Seeking advice on CK GroupedGemm**: A user inquired about using **CK GroupedGemm**, expressing uncertainty about which example to follow out of the many available.
   - Another member asked for clarification on the **target hardware**, indicating a need for context to provide relevant advice.
- **Tuning Llama3 on AMD MI300x**: A member shared a link to their journey on [Tune Llama3 405B on AMD MI300x](https://publish.obsidian.md/felafax/pages/Tune+Llama3+405B+on+AMD+MI300x+(our+journey)), detailing their experience.
   - This insight might offer valuable strategies for others working on similar projects.
- **VScode usage for CUDA and Metal**: A user mentioned they use **VSCode** specifically for coding in **CUDA and Metal**.
   - This highlights the versatility of VSCode in accommodating multiple programming environments.



**Link mentioned**: <a href="https://publish.obsidian.md/felafax/pages/Tune+Llama3+405B+on+AMD+MI300x+(our+journey)">Tune Llama3 405B on AMD MI300x (our journey) - Felafax Blog - Obsidian Publish</a>: Tune Llama3 405B on AMD MI300x (our journey) - Felafax Blog - Powered by Obsidian Publish.

  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1288062309268324392)** (1 messages): 

> - `Cutlass Sycl Fork` 


- **Interest in Cutlass Sycl Fork**: There is a new [Cutlass Sycl Fork](https://github.com/codeplaysoftware/cutlass-fork) available, focused on CUDA templates for linear algebra subroutines. This fork may be of interest to developers seeking optimized CUDA solutions.
- **Potential for Contribution**: The GitHub repository encourages contributions to the development of the **Cutlass Sycl Fork**, enhancing capabilities for linear algebra operations.
   - Developers can engage with the project to improve existing resources and tooling in the CUDA ecosystem.



**Link mentioned**: <a href="https://github.com/codeplaysoftware/cutlass-fork">GitHub - codeplaysoftware/cutlass-fork: CUDA Templates for Linear Algebra Subroutines</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to codeplaysoftware/cutlass-fork development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1287918558646632520)** (28 messages🔥): 

> - `Scaling on 4090 GPUs`
> - `Distributed Training Optimization`
> - `Peer-to-Peer GPU Communication` 


- **Scaling Issues with 2x 4090 Training**: When training with **2x 4090** on vast.ai, the **FSDP** showed only **13% speedup** while DDP provided **47% speedup**, attributed to slow communication bandwidth across NUMA nodes.
   - One participant noted that scaling is generally better with **A100 instances** due to higher communication efficiencies, leading to nearly **perfect speedups**.
- **Benfits of DDP Over FSDP in Low Bandwidth**: Members discussed how **DDP** performs better in low bandwidth scenarios compared to **FSDP**, with DDP able to aggregate gradients during forward passes for improved efficiency.
   - This led to insights about how newer features like **torch.compile** work cohesively with DDP.
- **Peer-to-Peer Communication in GPUs**: One participant described using a driver written by Geohot that enables **peer-to-peer communication** between GPUs without needing the CPU, using **PCIe DMA features**.
   - This method allows GPUs to directly communicate via PCIe connections, creating a fast inter-GPU communication setup.
- **Tinybox Configuration Challenges**: The challenges of building a **tinybox** configuration with multiple GPUs were highlighted, explaining its non-trivial nature despite the fun involved.
   - Participants expressed interest in a potential **cloud version** of such configurations for easier access.
- **Exploring NVMe Offloading Performance**: One user is running a kernel module for PCIe communications on two **4090** GPUs, testing the benefits of **offloading data** to **NVMe drives** versus **system RAM** and **CXL**.
   - They noted the utility of **resizable BAR** in facilitating efficient data transfers without CPU intervention.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/">NVIDIA Corporation</a>: NVIDIA Corporation has 509 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/NVIDIA/nccl-tests">GitHub - NVIDIA/nccl-tests: NCCL Tests</a>: NCCL Tests. Contribute to NVIDIA/nccl-tests development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1287860300661522433)** (11 messages🔥): 

> - `WebNN Channel Discussion`
> - `WebNN Integration with WebGPU and WASM`
> - `Event Invitation`
> - `NPU API Interfaces` 


- **WebNN Channel Should Be Created?**: A member suggested creating a separate channel for **WebNN**, referring to the integration of **webGPU** and **WASM** into a unified **WebNN architecture**.
   - However, another member expressed skepticism about the scope, stating that standardizing such high-level integrations is challenging given the fast-changing **multimodality landscape**.
- **WebNN Interfaces with NPU APIs**: Members discussed the purpose of **WebNN**, questioning if it was primarily intended to interface with **fixed function NPU APIs**.
   - It was clarified that **WebNN** also integrates with **webGPU** and **WASM**, and can work with **NVIDIA GPUs** through an abstraction layer on Windows.
- **Event Planning and Coordination**: A user requested a direct message for an event invite, expressing eagerness for it due to its relevance to their **ongoing project**.
   - Another member quickly confirmed the invite was sent, showcasing the community's engagement and coordination.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

0x000ff4: do we have regular meetings about liger-kernel
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1288178009328849081)** (2 messages): 

> - `GPU Memory Bandwidth`
> - `Puzzle Completion` 


- **Latency affects GPU data handling**: One member noted that effective data transfer onto and off of the **GPU** depends on **latency** and the GPU's memory bandwidth.
   - They questioned whether in **systems-on-a-chip**, the GPU has superior memory bandwidth compared to the **CPU**.
- **Puzzlers want to compare solutions**: A member asked if anyone had completed the puzzles, expressing that they are halfway through their own efforts.
   - They stated their intention to compare solutions once they complete their puzzles.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1287861392409362616)** (87 messages🔥🔥): 

> - `OpenAI's o1 models`
> - `Anthropic's funding talks`
> - `Stability AI board announcement`
> - `Gemini model updates`
> - `Scale AI financials` 


- **OpenAI's o1 Models Raise Interest**: OpenAI recently released the o1 family of models along with a graph on scaling laws for test-time compute, though the x-axis was unlabeled, prompting discussions on its reconstruction using the o1-mini API.
   - One member noted that the compute used may only be in the range of tens of thousands of tokens and questioned the feasibility of scaling without a tree structure.
- **Anthropic's Potential $40 Billion Valuation**: Reports surfaced that Anthropic has begun discussions with investors about raising capital, which could value the startup between **$30 billion** to **$40 billion**, effectively doubling its valuation from earlier this year.
   - This news reflects the competitive landscape as AI companies seek to bolster their financial backing amid rapid advancements.
- **James Cameron Joins Stability AI Board**: Stability AI announced that legendary filmmaker **James Cameron** has joined its Board of Directors, highlighting his expected contribution to visual media innovations through an artist-centric approach.
   - His addition is seen as a significant step for Stability AI as it aims to develop a more comprehensive AI pipeline for creators.
- **Gemini Model Enhancements Announced**: New production versions of the Gemini models were disclosed, featuring **over 2x higher rate limits**, a **50% price drop** on Gemini 1.5 Pro, and updated experimental features for developers.
   - The updated settings include opt-in filters for managing safety and reliability, enhancing developer control over configuration.
- **Scale AI's Financial Growth Insights**: Scale AI is reportedly experiencing healthy growth despite having relatively low gross margins, as highlighted in an analysis of its H1 financials showcasing nearly quadrupled sales.
   - This places Scale AI in an interesting position as the demand for AI services continues to escalate.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1838611055217385646">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Two new production Gemini models, &gt;2x higher rate limits, &gt;50% price drop on Gemini 1.5 Pro, filters switched to opt-in, updated Flash 8B experimental model, and more.   It’s a good day to be a ...</li><li><a href="https://x.com/hughbzhang/status/1838288923656941860">Tweet from Hugh Zhang (@hughbzhang)</a>: OpenAI recently released the o1 family of models and a graph showing scaling laws for test-time compute — sadly without the x-axis labeled.  Using only the public o1-mini API, I tried to reconstruct t...</li><li><a href="https://x.com/amir/status/1838604896087740645">Tweet from Amir Efrati (@amir)</a>: Scale AI has fairly low gross margins but growth seems healthy atm.   H1 financials here:  https://www.theinformation.com/articles/scale-ais-sales-nearly-quadrupled-in-first-half?utm_source=ti_app&rc=...</li><li><a href="https://x.com/aaronpholmes/status/1838580256032116981?s=46">Tweet from aaron holmes (@aaronpholmes)</a>: SCOOP in today&#39;s AI Agenda: Microsoft AI chief Mustafa Suleyman reshuffled some of his org, with Phi pioneer Sebastien Bubeck out. Another former Phi head has left for Google.   Here&#39;s what th...</li><li><a href="https://x.com/OfficialLoganK/status/1838613832790360208">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: @TheXeophon Yes, another updated version</li><li><a href="https://x.com/StabilityAI/status/1838584605986951254">Tweet from Stability AI (@StabilityAI)</a>: Today, our CEO, @premakkaraju, announced that legendary filmmaker, technology innovator, and visual effects pioneer, James Cameron, has joined the Stability AI Board of Directors.  Cameron’s addition ...</li><li><a href="https://x.com/colin_fraser/status/1838667677981904917">Tweet from Colin Fraser (@colin_fraser)</a>: I&#39;ve never been more vindicated  Quoting Colin Fraser (@colin_fraser)   What if it actually looks like this?</li><li><a href="https://x.com/kateclarktweets/status/1838319202798538974?s=61">Tweet from Kate Clark (@KateClarkTweets)</a>: Scoop: OpenAI rival Anthropic has started talking to investors about raising capital in a deal that could value the startup at $30 billion to $40 billion, roughly doubling its valuation from a funding...</li><li><a href="https://x.com/LucasAtkins7/status/1838593579305902217">Tweet from Lucas Atkins (@LucasAtkins7)</a>: He does have a lot of experience with sinking ships.  Quoting Stability AI (@StabilityAI)   Today, our CEO, @premakkaraju, announced that legendary filmmaker, technology innovator, and visual effects ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1288209649262264351)** (16 messages🔥): 

> - `Release Chronicle`
> - `Interconnects Artifacts`
> - `Late Fusion Visual LMs`
> - `GPT-4 Performance` 


- **Release Chronicle Inquiry**: A member asked if there exists a website that chronicles releases by who and when, prompting a response that Twitter and a blog serve this purpose.
   - The discussion led to the desire for a calendar or data view on the blog to track the information more efficiently.
- **Interconnects Artifacts Explained**: A member noted that *Interconnects Artifacts* could encompass various models, datasets, and systems, alluding to recent announcements from OpenAI.
   - They expressed that the artifact logs listed on Hugging Face could provide relevant insights into the evolving landscape.
- **Late Fusion Visual LMs Performance**: A query was raised regarding the performance of late fusion visual LMs on text benchmarks, with an expectation of no significant gains akin to those from the Tulu recipe.
   - Concerns were expressed about potential degradations in performance when using visual models.
- **GPT-4's Image Processing Observation**: While experimenting with GPT-4, a member observed that the model routes differently with image inputs than with text-only prompts.
   - The member questioned if there were noticeable differences in intelligence between the two modes of input.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing">[18 April 2024] Aligning open language models</a>: Aligning open language models Nathan Lambert || Allen Institute for AI || @natolambert Stanford CS25: Transformers United V4</li><li><a href="https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988">2024 Interconnects Artifacts - a natolambert Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1287863451028553860)** (26 messages🔥): 

> - `New Subdomain Announcement`
> - `Introduction to UV for Python`
> - `Docker and UV Integration`
> - `Cronjob for UV Updates`
> - `PyCharm Compatibility Issues` 


- **New Subdomain Stirs Excitement**: A new subdomain announcement got everyone buzzing, with comments reflecting on its implications and the potential coolness factor involved.
   - One user humorously noted that perhaps it represents a step away from conventional paths.
- **Philpax Powers Up with UV**: A user introduced their coworker to **UV** after expressing a need for a cargo-like tool for Python, claiming to feel 'powerful' after the introduction.
   - Another member shared a comprehensive TL;DR on UV’s capabilities, highlighting its speed and ease of use.
- **Optimizing Docker with UV**: Integrating UV with Docker is hassle-free, with users recommending the official Dockerfile and referencing resources like the [uv-docker-example](https://github.com/astral-sh/uv-docker-example).
   - A user shared advice on making a cronjob to ensure **UV** is updated almost daily due to frequent releases.
- **UV's Compatibility with PyCharm**: Some users raised concerns about **UV**'s compatibility with **PyCharm**, identifying workarounds like [ryecharm](https://github.com/InSyncWithFoo/ryecharm) to mitigate issues.
   - The consensus is that users shouldn't expect an official fix from JetBrains anytime soon.
- **Discussion on Brew vs Curl Installations**: A debate emerged over utilizing Brew versus Curl for installing **UV**, with one member comparing **UV** to **pnpm** for Python due to its benefits.
   - The discussion highlighted the confusion surrounding the choice between installation methods, particularly for Mac users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/karan4d/status/1838292114272325936?s=46">Tweet from mephisto (@karan4d)</a>: -be tencent -make gamegen diffusion model -say &#34;weights and paper soon&#34; on the GH repo -put out a github page showcasing the capability -announce to the world -delete everything  rugpulled aga...</li><li><a href="https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies>.">Running scripts | uv</a>: no description found</li><li><a href="https://docs.astral.sh/uv/guides/integration/docker/#caching>)">Docker | uv</a>: no description found</li><li><a href="https://x.com/colin_fraser/status/1838667677981904917?s=46">Tweet from Colin Fraser (@colin_fraser)</a>: I&#39;ve never been more vindicated  Quoting Colin Fraser (@colin_fraser)   What if it actually looks like this?
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1287852837749129216)** (96 messages🔥🔥): 

> - `O1 Planning Capabilities`
> - `World Simulator API Usage`
> - `Hermes & Monad Dynamics`
> - `Recent AI Model Upgrades`
> - `Nous Research and Merchandise` 


- **O1 Planning Capabilities Evaluation**: A research note on the planning capabilities of **O1** has been submitted to [arXiv](https://arxiv.org/abs/2409.13373) by team members who reportedly stayed up late working on it.
   - This summary hints that a comprehensive examination of **O1**'s abilities was conducted, with further details promised after public release.
- **World Simulator API Usage Discussion**: Members discussed **World Sim**, where users can earn credits upon signing up and incur costs for API calls, emphasizing its accessibility.
   - One user encouraged account creation for free credits, pointing out the low costs associated with its API.
- **Hermes & Monad Showing Stubborn Behavior**: Concerns were raised regarding **Hermes** and **Monad** becoming stubborn and less effective in conversations, particularly regarding their tagging abilities.
   - One member suggested a presence penalty might hinder their interactions, while another noted differences based on hosting.
- **Latest AI Model Improvements**: There was excitement over the **Gemini 1.5** September upgrade and a minor voice rollout for **GPT-4o**, indicating market anticipation.
   - The community is eager for new developments showcased at the upcoming **Meta Connect**, including potential AI advancements.
- **Nous Research and Merchandise Joke**: A lighthearted discussion unfolded about **Nous** being perceived as a clothing store, with jokes about merch driving revenue.
   - Clarifications followed that **Nous** is an AI research company, with team members committed to ongoing research despite the humorous remarks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">worldsim</a>: no description found</li><li><a href="https://x.com/replicate/status/1838346354206347730">Tweet from Replicate (@replicate)</a>: We&#39;re open sourcing our Flux code.  The community loves FLUX.1 text-to-image models. To serve them, we&#39;ve made a bunch of improvements internally: image-to-image mode, NSFW checkers, and most ...</li><li><a href="https://x.com/rao2z/status/1838245253171814419?s=46">Tweet from Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z)</a>: A research note describing our evaluation of the planning capabilities of o1 🍓 is now on @arxiv https://arxiv.org/abs/2409.13373 (thanks to @karthikv792 & @kayastechly). As promised, here is a summar...</li><li><a href="https://console.cloud.google.com/">Google Cloud Platform</a>: no description found</li><li><a href="https://huggingface.co/NousResearch">NousResearch (NousResearch)</a>: no description found</li><li><a href="https://github.com/NousResearch/finetuning-subnet">GitHub - NousResearch/finetuning-subnet</a>: Contribute to NousResearch/finetuning-subnet development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1287856441696190486)** (25 messages🔥): 

> - `llama.cpp performance`
> - `Scaling LLMs`
> - `GPT-2 pre-training`
> - `Sample packing`
> - `Tokenizers` 


- **llama.cpp struggles with Turmex**: Rebuilding [llama.cpp](https://link.to/llama.cpp) still results in crashes with **Turmex** when using version **3.1**, despite it working fine with version **3**.
   - A member noted that their **8GB RAM mobile** cannot effectively run the **Hermes-3-Llama-3.1-8B.Q4_K_M.gguf model**.
- **Debate on the value of scaling LLMs**: A member questioned if spending billions on LLMs is justified if they won't lead to AGI, suggesting that even basic tasks like customer support provide significant value.
   - Others agreed, asserting that even rudimentary models can save costs on manual labor in roles like data entry.
- **Insights on GPT-2 pre-training**: A member discussed the implications of using a dataset composed only of independent sentences in GPT-2 pre-training, expressing concern over potential suboptimal results.
   - Another member highlighted the concept of **sample packing**, warning that naively mixing sequences can degrade model performance.
- **Useful masking during training**: Discussion on the importance of zeroing out attention scores for future tokens during training, ensuring clear isolation of sequences.
   - It was suggested that adding special tokens like 'endoftext' could be beneficial for clarity but isn't strictly necessary for academic purposes.
- **Tokenizers and special tokens**: A question was raised about whether tokenizers would automatically add special tokens to datasets, with an emphasis on configurations.
   - Members mentioned that while some manual implementations may not include this feature, most off-the-shelf **Hugging Face** tokenizers do it automatically if specified.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1288109275520176148)** (2 messages): 

> - `DisTrO Resource Management`
> - `RENDER Network Testing` 


- **DisTrO thrives on poor bandwidth**: The initial paper suggests that **DisTrO** is usable over poor, asymmetric bandwidth and heterogeneous GPU setups, making it suitable for resource management via blockchains.
   - *It's a promising avenue for optimizing resource allocation in less-than-ideal conditions*.
- **Testing DisTrO on RENDER Network**: A member raised the question of whether the **RENDER network** could serve as a testing platform for DisTrO, as it is designed specifically for resource management in varying conditions.
   - *Utilizing RENDER could provide valuable insights into DisTrO's performance and scalability*.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

ar02293: hey www.keygunz.com go test
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1288109275520176148)** (2 messages): 

> - `DisTrO functionality`
> - `RENDER network` 


- **DisTrO Enhances Bandwidth Resource Management**: The initial paper suggests that **DisTrO** can effectively function over poor, asymmetric bandwidth and heterogeneous GPU environments, allowing for improved resource management through blockchains.
   - This capability positions it as a potential solution for varied network conditions and resource allocations.
- **RENDER Network as Testing Ground**: A member raised the idea of using the **RENDER network** as a testing platform for DisTrO, stating it is specifically designed for such requirements.
   - This discussion opens the floor for considering RENDER's existing infrastructure to validate DisTrO's performance under challenging conditions.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1287851282652205200)** (100 messages🔥🔥): 

> - `Qwen 2.5 Model Issues`
> - `Unsloth Trainer Memory Management`
> - `Fine-tuning Models`
> - `Using Lora with Ollama`
> - `Deployment Options for Models` 


- **Qwen 2.5 model issues addressed**: Several members reported problems with the **Qwen 2.5** model, including crashes and bugs, but discussed solutions like using improved templates or changing training approaches.
   - Members confirmed that they worked with the Qwen team to resolve some issues and bugs related to the model.
- **Managing Unsloth Trainer's memory usage**: A user experienced memory issues when initializing the **UnslothTrainer**, suggesting reducing the number of processes for dataset mapping to resolve the problem.
   - They successfully dropped the number of processes and reported back on the impact, indicating that balancing process numbers can help with initial memory mapping.
- **Fine-tuning process insights shared**: A user shared their experience with fine-tuning a **Vit_B16** model, emphasizing the importance of high-quality data over quantity for better outcomes.
   - They also expressed plans to continue fine-tuning their model with additional high-quality images, after initially achieving good accuracy.
- **Using Lora with Ollama documentation update requested**: A user inquired about an updated version of a guide on using **Lora** with **Ollama**, indicating interest in the latest procedures.
   - Responses suggested that not much has changed recently and highlighted earlier successes with using Lora in the framework.
- **Deployment strategies for fine-tuned models**: The conversation touched on effective methods for deploying models, with suggestions like **Runpod** and **Llama Labs** as hosting options.
   - One user expressed excitement about their newly fine-tuned model, thanking the community for support during the process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-lightweight-solution-to-full-cycle-llm-development-edadb6d9e0f0">Use Unsloth LORA Adapter with Ollama in 3 Steps</a>: Use LLama.Cpp to convert Unsloth Lora Adapter to GGML(.bin) and use it in Ollama — with a single GPU</li><li><a href="https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-">no title found</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/releases/tag/September-2024">Release Qwen 2.5 Support · unslothai/unsloth</a>: Qwen 2.5 Support is here! There are some issues with Qwen 2.5 models which Unsloth has fixed!  Kaggle Base model finetuning notebook: https://www.kaggle.com/code/danielhanchen/kaggle-qwen-2-5-unslo...</li><li><a href="https://youtu.be/UWF6dxQYcbU?feature=shared">FREE Fine Tune AI Models with Unsloth + Ollama in 5 Steps!</a>: Are you ready to train your own Large Language Model (LLM) but think it’s too complicated? Think again! In this video, I’m going to show you how anyone can f...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1287891074123305093)** (22 messages🔥): 

> - `Llama Model Quantization`
> - `Token Addition in Llama`
> - `Model Performance Insights`
> - `Feedback Mechanisms for Model Improvement`
> - `Memory Management in PyTorch` 


- **Memory Issues with Llama3.1**: A user encountered **out of memory errors** when trying to load the 4-bit quantized version of **Llama3.1**, struggling to allocate **20GB** while **14.75GB** was consumed by PyTorch.
   - The community suggested checking model configs and running the original example to troubleshoot the OOM issues.
- **Adding New Tokens to Llama's Vocab**: One user inquired about adding new tokens to Llama's vocabulary via Unsloth, specifically if these tokens would be utilized during tokenization by Ollama.
   - This raised curiosity about whether a specific inference engine is necessary for such additions.
- **Exploring Model Alternatives**: A participant suggested exploring the **Qwen model**, highlighting its superior performance in function calling based on personal tests.
   - Others chimed in, noting the efficiency of using smaller models if they meet user needs without excessive resource consumption.
- **Teaching Models from Past Examples**: A member sought advice on guiding a model using previous mistakes from past conversations, focusing on comments as feedback.
   - There was discussion on the challenges with current methods like KTO and ORPO, emphasizing a need for structured feedback to enhance model training.
- **OpenAI's Feedback Improvement Process**: The conversation explored how OpenAI utilizes **RLHF (Reinforcement Learning from Human Feedback)** to enhance their models based on user comments and feedback.
   - This led to a broader discussion about the role of multi-turn conversations in model training methodologies.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1287858899772244093)** (90 messages🔥🔥): 

> - `New Anthropic Model Release`
> - `Perplexity Pro Features`
> - `Merlin Extension`
> - `User Experiences with Perplexity`
> - `Query Limits Discussion` 


- **New Anthropic Model could drop soon!**: A source confirmed that a major AI model upgrade from Anthropic is expected to be released soon, highlighting that it will be significant for developers. Full details will be available after the embargo lifts.
   - Members expressed excitement, speculating on the potential of this upgrade and its implications in the AI space.
- **Curiosity about Perplexity Pro Search Features**: Users discussed the limitations associated with Perplexity Pro accounts, particularly regarding the number of searches allowed per day. Some users were unsure about the benefits of Pro accounts and expressed confusion over recent changes.
   - It was noted that Perplexity Pro offers a more personalized search experience, diving deeper into topics based on user preferences.
- **Insights on the Merlin Extension**: Members discussed the Merlin extension, which allows users to chat with various LLMs directly in the browser and provides unlimited premium model access. A comparison was made between Merlin and HARPA AI, detailing their functionalities and user experiences.
   - Users appreciated the unlimited queries with Merlin but noted its lack of transparency regarding model settings, contrasting it with HARPA AI's customizable features.
- **Users voice concerns regarding data privacy**: Concerns were raised about the retention of old search links that users can access even without an account, leading to privacy anxiety. One user reported that such links could potentially reveal personal information, prompting them to contact support for clarification.
   - Discussion centered on whether links shared by users while logged out should remain accessible, raising questions about the platform's data handling policies.
- **Query Limitations Spark Dialogue**: Several users inquired why the query limits had changed, specifically discussing the low limits of certain models like o1 mini. It was acknowledged that despite the lower limits, many users still effectively managed their queries without hitting caps.
   - The conversation highlighted user adaptability to the evolving limits and the strategies they employed to navigate their daily search activities.



**Link mentioned**: <a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1287855962866061373)** (10 messages🔥): 

> - `Cosmic Ambassador`
> - `Superintelligence Age`
> - `Perplexity AI Differences`
> - `AI Impact on Education`
> - `OpenAI Reasoning Probes` 


- **Exploring Carl Sagan: Cosmic Ambassador**: A link shared to [Carl Sagan: Cosmic Ambassador](https://www.perplexity.ai/page/carl-sagan-s-pale-blue-dot-._z2e7N_TsiT92iP5IGiMQ) discusses Sagan's influential ideas and his reflections on Earth’s place in the cosmos.
   - The [source thread](https://www.perplexity.ai/search/carl-sagan-s-theory-of-mind-be-QUYnNdrbT9CnRxX5MK7Q7w) dives deeper into Sagan's theory of mind related to his iconic 'Pale Blue Dot' speech.
- **Age of Superintelligence Unpacked**: A member shared a link about [The Age of Superintelligence](https://www.perplexity.ai/page/the-age-of-superintelligence-5usatQeSSr68txXaWW3zfg), addressing the implications of advanced AI on society.
   - Discussion focused on the balance between technological advancement and ethical considerations in AI development.
- **Differences of Perplexity AI Discussed**: Links were exchanged on [how Perplexity AI differs](https://www.perplexity.ai/search/how-is-perplexity-ai-different-pljzBCICQzmrh0eMnujxUw?login-source=visitorGate), highlighting its unique features and capabilities.
   - The conversation covered aspects like user interaction and AI learning processes.
- **Impacts of AI on Education Analyzed**: A member posted a link exploring [how AI impacts education](https://www.perplexity.ai/search/describe-how-ai-impacts-educat-E_.Sazb0ReeNYu5PmyIQaQ), examining its transformative role in learning environments.
   - Discussions centered around the potential benefits and challenges posed by AI in educational settings.
- **OpenAI Bans Reasoning Probes**: News shared that OpenAI has decided to [ban reasoning probes](https://www.perplexity.ai/page/openai-bans-reasoning-probes-cJxbLXEgQPK7itY3LL2mJw), sparking a debate on the consequences of this action.
   - Members expressed various viewpoints on the rationale behind the ban and its potential effects on AI training.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1287957269178486929)** (9 messages🔥): 

> - `Citational Access Requests`
> - `API Rate Limits`
> - `Output Consistency`
> - `Alternatives to PPLX`
> - `Exa.ai Exploration` 


- **Citational Access Requests go unanswered**: Members reported frustration over not receiving responses to requests for **citational access** or higher **API rate limits**, with one member noting multiple attempts over several months.
   - Another member expressed similar discontent, stating they had also emailed support without success.
- **Citation Output Inconsistencies Hamper Automation**: One member shared that they could obtain citations by asking for them, but experienced **inconsistent output**, alternating between **HTML** and **Markdown** formats.
   - This inconsistency is reportedly stalling their automation process, making it considerably more difficult to achieve reliable outputs.
- **Exploring Alternatives to PPLX**: With ongoing issues, one member is considering **Exa.ai** as a possible alternative to **PPLX**, noting it functions more like an internet search wrapper for other **LLMs**.
   - They emphasized the need for a solution that allows for specific domain searches that change over time, suggesting Exa.ai could potentially fit that need.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1287861393894277284)** (72 messages🔥🔥): 

> - `LM Studio Installation on Air-Gapped Machines`
> - `Model Support in LM Studio`
> - `Model Performance and Handling`
> - `LongWriter Model Insights`
> - `Upcoming Model Releases` 


- **Installing LM Studio on Air-Gapped Machines**: Members discussed the feasibility of installing LM Studio on air-gapped machines, mentioning the need to download installers and models separately from the internet.
   - It was emphasized that while the installation does not require internet, initial setup and file transfers to the target machine are necessary for proper functionality.
- **Unsupported Models Cause Errors**: When attempting to load image generation models like **Flux**, users reported errors due to lack of support in LM Studio for such models.
   - It was noted that no image generation models are currently supported, causing confusion with model architectures.
- **Concerns About Model Performance**: Performance issues were raised when nearing token limits in conversations, with users observing slowdowns as they approached the model's context length.
   - This behavior was described as normal, attributed to VRAM usage during processing, with suggestions to manage limits and expectations.
- **LongWriter Model Capabilities**: Discussion on the **LongWriter** model emphasized its ability to generate long texts and its potential for fine-tuning.
   - Members were encouraged to explore this model further, with provided links to resources for deeper insights on its implementation.
- **Future Releases and Enhancements**: Questions were raised about the availability of models like **Pixtral**, with expectations framed around the readiness of **llama.cpp** support.
   - Changes in model availability and releases were speculated on, with community insights contributing to an ongoing dialogue about future developments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/THUDM/LongWriter">GitHub - THUDM/LongWriter: LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs</a>: LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs - THUDM/LongWriter</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7107">tokenization: no double BOS tokens by JohannesGaessler · Pull Request #7107 · ggerganov/llama.cpp</a>: Relevant discussion: #7062 The llama.cpp tokenizer currently adds a BOS token unconditionally. However, I think it would make more sense not to do this if the prompt already starts with a BOS token...</li><li><a href="https://github.com/abetlen/llama-cpp-python/issues/1501">Llama 3 Double BOS · Issue #1501 · abetlen/llama-cpp-python</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1287959860809564323)** (25 messages🔥): 

> - `ROCm on AMD APU`
> - `Dual GPU Setup Compatibility`
> - `Performance of RTX 3090`
> - `Price Differences in GPU Markets`
> - `Consumer Protections in EU` 


- **ROCm not supported on AMD APUs**: Members confirmed that ROCm is **not supported** on AMD APUs like the **5700G**. One noted that while built-in GPUs could use Vulkan API, performance gains would be negligible as they share system RAM.
   - The consensus is that using **APUs** for ROCm applications remains impractical due to hardware limitations.
- **Discussion on dual GPU setups**: A member inquired if **LM Studio** supports a dual GPU setup with an **RTX 4070 Ti** and **RTX 3080**. Others discussed the theory behind using different Nvidia cards together, noting potential benefits.
   - One suggestion was to consider compatibility concerns before proceeding with varied GPU models.
- **RTX 3090 performance expectations**: There's curiosity around the **TPS** one might expect from an **RTX 3090**, particularly for **inference training** scenarios. Members speculated on performance metrics and the impact of future GPU acquisitions.
   - Interest was noted in using additional GPUs like the **A770** for handling bigger models.
- **Price Discrepancies between US and EU**: A member expressed frustration at the **higher GPU prices** in the EU compared to the US market, where they can be found for around **$750**. The discussion highlighted the challenges buyers face due to regional pricing.
   - It was pointed out that **VAT** and other taxes contribute to inflated prices in Europe.
- **Consumer protections in the EU**: A member acknowledged that while tech prices are higher in the EU, there are benefits like superior **warranty coverage** and support. Discussions emphasized the importance of consumer protections that ensure top-tier standards for users.
   - The comparison between North America and Europe revealed notable differences in **government protections** for consumers.



**Link mentioned**: <a href="https://tenor.com/view/income-tax-tax-taxes-gif-11011288">Income Taxes GIF - Income Tax Tax Taxes - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1287904088855547924)** (64 messages🔥🔥): 

> - `Mojo Language Tier List`
> - `Compilation Issues with Rust`
> - `NixOS and Package Management`
> - `MLIR vs LLVM`
> - `Mojo's Growth and Community` 


- **Mojo Language Tier List Ranking**: A user shared their personal language tier list, ranking **Mojo** at the top, followed by **C#**, **Rust**, and others, describing it as more of a subjective feel than a logical ranking.
   - Another user suggested separating **C++** categories based on project circumstances and emphasized the need for clean C interoperability.
- **Rust Compilation Speed Challenges**: Users expressed frustration about **Rust's slow compilation times**, particularly when working on larger projects like a **40k line game**, taking significant time for changes.
   - Discussions highlighted how generics contribute to slowdowns, and recommendations included optimizing file system settings on **Windows** to improve performance.
- **Exploring NixOS as an Alternative**: Interest in migrating to **NixOS** sparked conversations about its benefits primarily revolving around its package manager, but some cautioned about its project complexity.
   - Users discussed their desire to reproduce systems using NixOS while weighing the potential simplicity of other tools like **Ansible** for smaller setups.
- **Comparison of MLIR and LLVM Features**: A member raised a query on why **MLIR** is seen as superior to **LLVM**, with explanations focusing on improvements in parallel compilation and better handling of high-level semantics.
   - This discussion noted the advantages of not losing debug information with MLIR as compilers evolve, making it a preferable choice in certain contexts.
- **Mojo's Development and Community Engagement**: Users celebrated **Mojo's two-year anniversary**, reflecting on the community's growth and engagement during notable events like the first **Mojo SDK** release.
   - There were expressions of excitement about Mojo's future, with anticipation for how it might evolve over the next few years as the language matures.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1287858470401474593)** (31 messages🔥): 

> - `Mojo classes vs Python classes`
> - `Monkey patching in Mojo`
> - `Future of pattern matching in Mojo`
> - `Communication speed between Mojo and C vs Python`
> - `Metaclasses and advanced features in Mojo` 


- **Mojo classes face criticism for Python-like behavior**: A member expressed concerns that **Mojo classes** should not emulate the dynamic features of **Python classes**, suggesting that such functionality be reserved for Python due to performance issues.
   - Another member highlighted that while some want **structs** to fulfill class-like features, many agree there's a need for a **trait system** before focusing on class implementation.
- **Desire for advanced features like metaclasses**: A member humorously proposed the inclusion of **metaclasses** and advanced features like **monkey patching** in Mojo, hinting at a playful chaos in language design.
   - There seems to be a rigorous conversation about marrying **mutable vs immutable** reflection paths to enhance Mojo's capabilities.
- **Discussion on adding 'match case' in Mojo**: Members showcased interest in Mojo potentially adopting the **'match case'** statement syntax in future releases, aligning with features from Python 3.10.
   - The conversation also touched on how the introduction of **sum types/enums** is necessary for implementing more advanced pattern matching, akin to peanut butter and jelly.
- **Communication efficiency comparisons**: A question arose regarding whether communication between **Mojo and C** (through **DLHandle**) is faster than its communication with **Python**.
   - While speculation was offered, definitive conclusions were not reached, noting that performance might depend on how **Python** interacts with **C**.
- **Diverging views on classes in Mojo**: Members demonstrated divided opinions on the necessity of **classes** in Mojo, with some advocating for a class-like interface while others are content with lower-level constructs.
   - The discussion revealed some support for maintaining a clean separation from Python's dynamic capabilities, favoring a focus on more foundational features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/main/proposals/mojo-and-dynamism.md">mojo/proposals/mojo-and-dynamism.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/discussions/459">Implement Pattern Matching as expression instead of statement in Mojo · modularml/mojo · Discussion #459</a>: Do you consider implementing Pattern Matching as an expression, to step out from C-legacy(lots of return statements that lead to bad/mutable code) and make Mojo a modern language with functional st...</li><li><a href="https://github.com/modularml/mojo/issues/3534">[Historical Discussion] Mojo and Dynamism · Issue #3534 · modularml/mojo</a>: Discussed in #466 Originally posted by Mogball July 20, 2023 Mojo has the lofty goal of being a simple, powerful, and easy-to-use language like Python but with features that allow programmers to re...</li><li><a href="https://m.youtube.com/watch?v=sPiWg5jSoZI&pp=ygUYRGF2aWQgbWV0YWNsYXNzZXMgcHl0aG9u">Python 3 Metaprogramming</a>: David BeazleySome of the most significant changes in Python 3 are related to metaprogramming.  In this tutorial, I&#39;ll cover decorators, class decorators, des...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[announcements](https://discord.com/channels/1161519468141355160/1209871299854336060/1287887683976433705)** (2 messages): 

> - `DSPy 2.5.0 release`
> - `Migration to LiteLLM`
> - `Deprecation of pre-2.4 LM clients`
> - `Feedback solicitation`
> - `Upcoming changes` 


- **DSPy 2.5.0 quietly releases**: **DSPy 2.5.0** has been released, and users are encouraged to share their feedback before a broader announcement is made.
   - This version deprecates all pre-2.4 LM clients, including those from OpenAI.
- **Migration made easy**: Users can complete the [migration](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb) in about **3 minutes**, enhancing the quality of their programs significantly.
   - This is particularly beneficial for chat LMs and complex signatures, ensuring better consistency.
- **Feedback is critical**: The release is intentionally low-key, relying on **deprecation warnings** to inform users, as feedback is crucial for ongoing adjustments.
   - The developers emphasize their openness to comments as the new version is fine-tuned over the next few days.
- **Consistent quality with new Adapter layer**: With the use of `dspy.LM`, users' DSPy modules will route through the **configured Adapter**, which by default is `dspy.ChatAdapter`.
   - This addition is aimed at improving user experience and adaptability across various use cases.
- **Exciting updates on the horizon**: Over the next **10-15 days**, users should anticipate a flurry of updates and enhancements.
   - This period is likely to bring numerous valuable changes as per the development team's plans.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="http://localhost:{sglang_port}/v1")">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1287977404068790285)** (2 messages): 

> - `DSPy powered AI code assistant`
> - `Live coding session` 


- **Live Coding of the DSPy Powered AI Assistant**: A member announced a live coding session scheduled for **9am PST (4pm GMT)** to build the first **DSPy powered AI code assistant**.
   - They encouraged others to join the session for insights on the development process of this innovative tool.
- **Kickoff of DSPy Code Assistant Development**: The live coding session commenced in the designated channel, focusing on the setup of the DSPy code assistant.
   - Participants were encouraged to engage and follow along for a hands-on experience in the building process.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1288149512929546250)** (1 messages): 

> - `Chain-of-thought (CoT)`
> - `Performance Benefits of CoT`
> - `Quantitative Meta-Analysis of CoT` 


- **CoT Methodology Evaluated**: A recent paper presents a quantitative meta-analysis covering over **100 papers** using Chain-of-thought (**CoT**) prompting to analyze its effectiveness across **14 models**.
   - The study emphasizes that **CoT benefits** tasks involving math or logic significantly more than other task types, suggesting an optimal usage strategy.
- **CoT vs Direct Answering on MMLU**: When analyzing **MMLU**, it was found that directly generating answers without CoT yields almost identical accuracy to using CoT, especially when dealing with **symbolic operations**.
   - Performance variances arise primarily when the question or response contains an **equals sign**, indicating a need for reasoning.
- **Planning and Execution in CoT**: The paper separates **planning and execution** in CoT tasks, providing insights into how CoT operates compared to tool-augmented LLMs.
   - Much of the performance gains from CoT derive from better planning, highlighting the methodology's nuances.



**Link mentioned**: <a href="https://arxiv.org/abs/2409.12183">To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning</a>: Chain-of-thought (CoT) via prompting is the de facto method for eliciting reasoning capabilities from large language models (LLMs). But for what kinds of tasks is this extra ``thinking&#39;&#39; reall...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1287888652550275193)** (84 messages🔥🔥): 

> - `DSPy 2.5.0 Release`
> - `Feedback on Meetings`
> - `Custom Adapters for LLM`
> - `Multimodal Capabilities`
> - `Cache Control in DSPy` 


- **DSPy 2.5.0 Launch Generates Excitement**: The release of DSPy 2.5.0 was announced, promising to help address **50-100** issues quickly. Members expressed enthusiasm about the new features and upcoming intro notebooks being created.
   - One query highlighted the potential for public weekly meetings to gather further feedback.
- **Custom Adapters Implementation Queries**: Discussion emerged around creating custom adapters to specify additional parameters for LLM calls, such as `grammar` for structured outputs. Members shared their experiences with previous implementations and expressed a desire for clearer best practices with the new `dspy.LM` structure.
   - Another user confirmed the need for passing additional parameters to their LLM, illustrating a common theme of adapting functionality.
- **Interest in Multimodal Features**: Hype built over the upcoming multimodal capabilities of DSPy, expected to be available next week. Questions arose regarding its compatibility with various model types, including audio LMs like **Ultravox**.
   - The response clarified that the initial rollout would focus on Vision Language Models (VLMs) unless the interfaces are similarly structured.
- **Cache Management in DSPy**: Users inquired about managing cache effectively to gauge true inference speeds without caching effects. It was clarified that newer implementations allowed cache control via environment variables like `DSP_CACHEBOOL` to be set to false.
   - This functionality is critical for evaluating performance in tasks where caching influences results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/fullstackwebdev/dc0f4e97df7591ade63f83d27668fe25">XMLAdapter</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb">dspy/examples/migration.ipynb at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://gist.github.com/fullstackwebdev/ddf21d55cef58a40471e8925834e6531">test_chat_adapter.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/fixie-ai/ultravox">GitHub - fixie-ai/ultravox: A fast multimodal LLM for real-time voice</a>: A fast multimodal LLM for real-time voice. Contribute to fixie-ai/ultravox development by creating an account on GitHub.</li><li><a href="https://marimo.io/">marimo | a next-generation Python notebook</a>: Explore data and build apps seamlessly with marimo, a next-generation Python notebook.</li><li><a href="https://github.com/stanfordnlp/dspy/issues/390#issuecomment-1947542304">[WIP] Major refactor roadmap  · Issue #390 · stanfordnlp/dspy</a>: DSPy has a small number (maybe 5-6) of extremely powerful concepts that have grown organically over the past year as open source. Internally, it&#39;s time for a major refactor that will simplify thin...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1287930340815667241)** (2 messages): 

> - `GROQ API Integration`
> - `Chain of Thought Evaluation` 


- **GROQ API Key Setup**: A user provided instructions to set the **GROQ_API_KEY** and run the necessary Python code for model use: `lm = dspy.LM('groq/llama3-8b-8192')`.
   - This setup is aimed at facilitating the use of the **dspy** library with the **Llama 3** model.
- **Chain of Thought Example Execution**: The provided code snippet includes a demonstration of **Chain Of Thought** functionality with a simple math question: `what is 2+2?`.
   - This example illustrates how to execute queries leveraging the dspy framework's capabilities.
- **Inquiring About Results**: Another user expressed curiosity about the results of the previous question, asking if an answer was found.
   - This reflects ongoing interest and engagement in discussing the outcomes of the implemented examples.


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1287867352750362748)** (1 messages): 

> - `Lecture 3`
> - `Agentic AI Frameworks`
> - `AutoGen`
> - `Multimodal Knowledge Assistant` 


- **Today's Lecture on AI Frameworks and Multimodal Assistants**: The **3rd lecture** of the course will take place today at **3:00pm PST**, featuring [this livestream](https://www.youtube.com/live/OOdtmCMSOo4) with two prominent speakers.
   - Chi Wang will discuss **Agentic AI Frameworks & AutoGen**, while Jerry Liu will cover the steps to build a **production-ready multimodal knowledge assistant**.
- **Chi Wang on Agentic AI Design Considerations**: Chi Wang's talk will address **core design considerations** of agentic AI programming frameworks, along with a focus on **AutoGen**, its application, and recent research developments.
   - It will conclude with open questions regarding future AI applications and developer empowerment.
- **Jerry Liu's Insights on Multimodal AI Pipelines**: Jerry Liu's session will outline the gradual development of a **multimodal knowledge assistant**, discussing the **advanced RAG pipeline** for research purposes.
   - It will incorporate elements such as **structured outputs**, **agentic reasoning**, and **event-driven workflows** to create an effective agent system.
- **Course Staff Contact Information**: For any inquiries, participants are encouraged to reach the course staff via the designated **Discord channel**.
   - This provides a direct way to address questions regarding the course material and lectures.



**Link mentioned**: <a href="https://www.youtube.com/live/OOdtmCMSOo4">CS 194/294-196 (LLM Agents) - Lecture 3, Chi Wang and Jerry Liu</a>: no description found

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1287854038091825315)** (33 messages🔥): 

> - `Course Attendance Issues`
> - `Guest Speaker Requests`
> - `Quiz Links`
> - `Open Embedding Models`
> - `Applications of AutoGen` 


- **Clarification on Course Attendance**: The attendance form for a livestream is exclusively for Berkeley students, and MOOC students shouldn't fill it out, leading to some confusion.
   - *Next time, it will be specified better when presenting the QR code* to avoid such misunderstandings.
- **Guest Speaker Suggestions**: A request was made for future guest speakers like Viren from Orkes.io and Hamza Farooq from Traversaal.ai.
   - The course staff is open to creating a feedback form for these suggestions, inviting details on why the speaker is requested.
- **Finding Quiz Links**: Quiz links for the course are posted in the syllabus section on the course website, with quizzes being released one to two days after each lecture.
   - Members discussed where to find these links, ensuring accessibility.
- **Current State of Open Embedding Models**: The best open embedding model is currently believed to be [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3), developed by Jina AI, which offers multilingual embeddings.
   - This model features *Task LoRA* for enhanced performance in neural search applications.
- **Real-World Examples of AutoGen Applications**: Members shared examples of complex applications built with AutoGen, including a bot for antique computers and a healthcare-related application, [hospitalgpt](https://github.com/micklynch/hospitalgpt).
   - The community expressed interest in seeing more sophisticated software developed using AutoGen.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>: no description found</li><li><a href="https://github.com/micklynch/hospitalgpt">GitHub - micklynch/hospitalgpt</a>: Contribute to micklynch/hospitalgpt development by creating an account on GitHub.</li><li><a href="https://github.com/lamm-mit/SciAgentsDiscovery">GitHub - lamm-mit/SciAgentsDiscovery</a>: Contribute to lamm-mit/SciAgentsDiscovery development by creating an account on GitHub.</li><li><a href="https://github.com/emooreatx/ccmp_ai">GitHub - emooreatx/ccmp_ai: Classic/Retro Computing LLM bot</a>: Classic/Retro Computing LLM bot. Contribute to emooreatx/ccmp_ai development by creating an account on GitHub.</li><li><a href="https://huggingface.co/jinaai/jina-embeddings-v3">jinaai/jina-embeddings-v3 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1287881743675953286)** (23 messages🔥): 

> - `Q&A Discussion`
> - `Technical Setup Delays`
> - `Zinley Project Mentioned`
> - `AutoGen Code Details`
> - `Chess Against AlphaGo` 


- **Q&A Part Needs to Stay**: Participants expressed a desire for the **Q&A part** of the event not to be cut short, with one member assuring they will ask staff about it.
   - *It would be great if the speaker can repeat the question* to ensure clarity during discussions.
- **Technical Setup Delays Ongoing**: Members noted recurring delays during the initial stages, especially related to **audio-visual setup**, with one humorously remarking that it's usually about **AV stuff** in the first 20 minutes.
   - The setup was temporarily interrupted, resulting in **no feed** visible until it reportedly returned.
- **Zinley Project Sparks Curiosity**: A member inquired about a project mentioned, leading to a discussion about **Zinley**, a friendly AI that simplifies software creation for users of all skill levels.
   - They shared insights about **Zinley's** mission, aimed at turning ideas into software quickly, as highlighted on their [website](https://zinley.com/about.html).
- **Details on AutoGen Being Compiled**: Members discussed that the code being compiled was likely related to **AutoGen**, specifically a user proxy agent capable of executing code and interacting with other agents.
   - Documentation for this agent, which includes various functionalities, can be found [here](https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent/).
- **AlphaGo Easily Defeats Chess Players**: A member's curiosity about using a Go agent to challenge **AlphaGo/AlphaZero** sparked a brief exchange, with another member asserting that **AlphaGo would win** significantly.
   - This highlights the **dominance of AlphaGo** in the domain of strategy games against human-like competitors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent/">agentchat.user_proxy_agent | AutoGen</a>: UserProxyAgent</li><li><a href="https://zinley.com/about.html">Zinley | Making Software Creation Easy for Everyone</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1287923906337968230)** (33 messages🔥): 

> - `Search/Retrieval Techniques`
> - `AutoGen vs. CrewAI`
> - `O1 API Usage`
> - `Multi-Agent Collaboration`
> - `Advanced DSL for Agents` 


- **Search/Retrieval Techniques for RAG**: A member suggested focusing on **classical NLP techniques** related to information retrieval, including **ranking algorithms** and **semantic understanding** to improve in the R part of RAG.
   - They highlighted the importance of grasping **indexing** and **relevance** in enhancing search capabilities.
- **AutoGen Customization vs. CrewAI Speed**: A member explored **AutoGen** and **CrewAI** for multi-agent collaboration, finding AutoGen more customizable, while CrewAI excels in fast prototyping.
   - Concerns were raised about **CrewAI's** limitation in conducting back-and-forth communication for agents, a feature offered by AutoGen's **conversable_agent**.
- **O1 API Call Considerations**: A member questioned whether the **O1 API** is still appropriate for the AutoGen framework due to the internal agent adoption, speculating it may increase inference time.
   - They noted that while **O1-mini** should be used for programming tasks, experimenting with **O1-preview** could yield insights as an adversarial agent or planner.
- **Comparing O1 Mini and Preview**: During discussions, it was noted that **O1-preview** could be practical for specialized complex tasks, despite the strange performance dichotomy with **O1-mini**.
   - Participants pointed out the challenges in evaluation due to its **black box** nature while acknowledging the potential of **O1-mini** for programming tasks.
- **Discussion on DSL for Multi-Agent Systems**: There was a call for a **generalized DSL** for defining multi-agent systems that avoids graph theory complexity, aiming for experimental philosophy in AI.
   - Members agreed on the challenges of such a DSL but recognized that it could help enhance agent collaboration.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1287887476391940247)** (74 messages🔥🔥): 

> - `Letta AI`
> - `Gemini Model Updates`
> - `Voice Feature Rollout`
> - `Customer Service Agent Experimentation`
> - `HuggingChat App Launch` 


- **Letta AI Emerges from Stealth**: Excitement surrounds the launch of [Letta AI](https://x.com/sarahwooders/status/1838261104864346288?s=46), a company focused on developing stateful LLM agents, by founders [Sarah Wooders](https://x.com/sarahwooders) and [Charles Packer](https://x.com/sarahwooders/status/1838261104864346288?s=46). They are actively hiring and building their team in San Francisco.
   - Read more about Letta in [TechCrunch](https://techcrunch.com/2024/09/23/letta-one-of-uc-berkeleys-most-anticipated-ai-startups-has-just-come-out-of-stealth/).
- **Gemini Model Enhancements**: [Gemini models](https://x.com/OfficialLoganK/status/1838611055217385646) received significant updates, including double the rate limits and over a 50% price reduction on Gemini 1.5 Pro. Filters have switched to opt-in, and an updated Flash 8B experimental model has been released.
   - Developers are optimistic about these changes, viewing it as a great time for developers, as explained in the [Google Developers Blog](https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/).
- **Voice Feature Rollout**: OpenAI announced that [Advanced Voice](https://x.com/openai/status/1838642444365369814?s=46) is rolling out to Plus and Team users within the ChatGPT app, introducing multiple new features and improved accents. Notably, it can express phrases in over 50 different languages.
   - However, access is not yet available in several European nations, as highlighted by [OpenAI's](https://x.com/OpenAI/status/1838642453391511892) announcement.
- **Customer Service Agent Experimentation**: Discussion about challenges in managing multi-turn conversations with agent simulations revealed important insights into maintaining effective user interaction. Suggestions included implementing stage markers and setting clear conversation termination guidelines.
   - Users are exploring various approaches to integrate reinforcement learning into conversation management to improve the customer agent experience.
- **HuggingChat macOS App Introduction**: The newly released [HuggingChat](https://x.com/cyrilzakka/status/1838618605648490974?s=61) app for macOS offers native integration of open-source LLMs with features like markdown support and web browsing. It marks a significant step forward in user-friendly AI tools for direct desktop use.
   - This app demonstrates a trend toward enhancing accessibility and functionality in AI-driven applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/hughbzhang/status/1838288923656941860">Tweet from Hugh Zhang (@hughbzhang)</a>: OpenAI recently released the o1 family of models and a graph showing scaling laws for test-time compute — sadly without the x-axis labeled.  Using only the public o1-mini API, I tried to reconstruct t...</li><li><a href="https://x.com/OfficialLoganK/status/1838611055217385646">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Two new production Gemini models, &gt;2x higher rate limits, &gt;50% price drop on Gemini 1.5 Pro, filters switched to opt-in, updated Flash 8B experimental model, and more.   It’s a good day to be a ...</li><li><a href="https://x.com/hughb">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/sarahwooders/status/1838261104864346288?s=46">Tweet from Sarah Wooders (@sarahwooders)</a>: Excited to announce @Letta_AI, the company @charlespacker and I started for building stateful LLM agents  We&#39;re building out an incredible (in-person) team in SF, and are actively hiring founding ...</li><li><a href="https://x.com/cyrilzakka/status/1838618605648490974?s=61">Tweet from Cyril Zakka, MD (@cyrilzakka)</a>: Excited to release HuggingChat 💬 - a native macOS app that brings powerful open-source language models straight to your desktop - with markdown support, web browsing, code syntax highlighting and muc...</li><li><a href="https://arxiv.org/abs/2402.01662">Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives</a>: As AI systems quickly improve in both breadth and depth of performance, they lend themselves to creating increasingly powerful and realistic agents, including the possibility of agents modeled on spec...</li><li><a href="https://x.com/natolambert/status/1837232801235755174">Tweet from Nathan Lambert (@natolambert)</a>: Things of note (not that much) in this longer o1 video:  1. “Model with RL is better at finding new CoT steps than humans” 2. “Emergence of self critique was a powerful moment” 3. Mentioned a literal ...</li><li><a href="https://forecast.safe.ai/?id=66e09f58718ead7507890d82">FiveThirtyNine | Forecasting AI</a>: no description found</li><li><a href="https://x.com/anushkmittal/status/1837233399209283762">Tweet from anushk (@anushkmittal)</a>: @natolambert interesting. ai is more than just text generation, it&#39;s about building agents that can understand and interact with the world.</li><li><a href="https://x.com/Smol_AI/status/1838663719536201790">Tweet from AI News by Smol AI (@Smol_AI)</a>: it&#39;s notable how predictive the Lmsys Elo vs $ pricing curve is, and how the strategy is panning out. Today&#39;s Gemini Pro price cut brings it exactly in line with where a loglinear pricing curv...</li><li><a href="https://x.com/rohanpaul_ai/status/1838171186229858677?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Rohan Paul (@rohanpaul_ai)</a>: Priompt is prompt design library being used internally at @anysphere, the Company behind @cursor_ai  &#34;Prompting should be called prompt design  Prompting as communicating with a time-constrained h...</li><li><a href="https://x.com/OpenAI/status/1838642453391511892">Tweet from OpenAI (@OpenAI)</a>: Advanced Voice is not yet available in the EU, the UK, Switzerland, Iceland, Norway, and Liechtenstein.</li><li><a href="https://x.com/KateClarkTweets/status/1838319202798538974">Tweet from Kate Clark (@KateClarkTweets)</a>: Scoop: OpenAI rival Anthropic has started talking to investors about raising capital in a deal that could value the startup at $30 billion to $40 billion, roughly doubling its valuation from a funding...</li><li><a href="https://x.com/shishirpatil_/status/1837205152132153803">Tweet from Shishir Patil (@shishirpatil_)</a>: 📣 Announcing BFCL V3 - evaluating how LLMs handle multi-turn, and multi-step function calling! 🚀 For agentic systems, function calling is critical, but a model needs to do more than single-turn task...</li><li><a href="https://x.com/miramurati/status/1838642696111689788">Tweet from Mira Murati (@miramurati)</a>: All Plus and Team users in ChatGPT  Quoting OpenAI (@OpenAI)   Advanced Voice is rolling out to all Plus and Team users in the ChatGPT app over the course of the week.  While you’ve been patiently wai...</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://x.com/openai/status/1838642444365369814?s=46">Tweet from OpenAI (@OpenAI)</a>: Advanced Voice is rolling out to all Plus and Team users in the ChatGPT app over the course of the week.  While you’ve been patiently waiting, we’ve added Custom Instructions, Memory, five new voices,...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb">dspy/examples/migration.ipynb at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://arstechnica.com/information-technology/2024/09/my-dead-father-is-writing-me-notes-again/?utm_source=changelog-news">My dead father is “writing” me notes again</a>: A recent AI discovery resurrected my late father&#39;s handwriting—and I want anyone to use it.</li><li><a href="https://arstechnica.com/information-technology/2024/">2024 | Ars Technica</a>: no description found
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1287883506957156363)** (29 messages🔥): 

> - `Cohere AI`
> - `Aya initiative`
> - `Job anxiety`
> - `Testing hypotheses`
> - `Chain of Thought (COT)` 


- **Newcomers Share Interest in Cohere AI**: Members like **Nav**, a Mechanical Engineering student, expressed interest in learning about **Cohere** and AI, while **Sanjeev** seeks direction to relevant blogs or videos.
   - In response, a link to the [Aya Research](https://cohere.com/research/aya) was shared, introducing the initiative aimed at advancing multilingual AI.
- **Job-Related Concerns Addressed**: Member **Milansarapa** expressed nervousness regarding financial situations and starting a new job, prompting reassurance from others about having a contract.
   - *You have the contract already*, alleviating fears and reinforcing community support.
- **Hypothesis Testing in LLMs**: **Milansarapa** queried whether similar results across various large language models indicate successful hypothesis testing regarding Recursive Iterative models.
   - Advice was given by **mrdragonfox** to use benchmarks and evaluation harnesses for more accurate testing and to explore different topics.
- **Exploring Different Methodologies**: **Milansarapa** discussed the need to review answers for similarities and explore different topics and LLMs using their method.
   - **Mrdragonfox** emphasized the importance of achieving more accurate answers through systematic testing.
- **Understanding Chain of Thought (COT)**: **Milansarapa** asked about the concept of COT, prompting explanations from fellow members regarding its function in improving some problem-solving methods.
   - '**Chain of Thought**' refers to a strategy that can enhance performance on certain tasks, though not universally applicable.



**Link mentioned**: <a href="https://cohere.com/research/aya">Aya</a>: Cohere’s non-profit research lab, C4AI, released the Aya model, a state-of-the-art, open source, massively multilingual, research LLM covering 101 languages – including more than 50 previously underse...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1287883672674111560)** (8 messages🔥): 

> - `Server Locations`
> - `Using Single Step Tools with Javascript` 


- **Discussion on Server Locations for Users**: A member brought up the presence of servers in multiple locations, questioning if hosting location can be confirmed for UK-based users.
   - Another member suggested using AWS or Vertex to choose the appropriate region if there are country boundary constraints.
- **Single Step Tools with Javascript .chatStream()**: A member inquired about using Single Step Tools with the Javascript method .chatStream().
   - A response highlighted a parameter, **force_single_step=True**, that can be used to facilitate working with single steps, though its public availability was uncertain.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1288083275289464882)** (5 messages): 

> - `Multilingual Reranker Issues`
> - `Embedding Model Selection`
> - `Reranker Best Practices` 


- **Multilingual Reranker struggles with foreign languages**: A user reported that the **multilingual quality** of the reranker leads to low relevance scores in languages like **Polish**, filtering out potentially useful data even when available.
   - *The relevance score is so low that it gets filtered out*, making the reranker ineffective for their use cases.
- **Using ada_2 model for reranking**: The team mentioned they are using the **ada_2** model from OpenAI for their reranking tests and provided example queries like *'what are the working hours?'* to illustrate their implementation.
   - They shared models **rerank-multilingual-v3.0** and **rerank-english-v3.0** as part of their testing setup.
- **Emphasis on top results over scores**: A member emphasized that when working with the reranker, focusing on the **top n results** is more crucial than their relevance scores for filtering chunks.
   - They suggested defining **top_n** as 1 or 3 for 100 documents, regardless of relevance scores, to see the most relevant chunks first.
- **Best practices for multilingual datasets**: Advice was given on best practices for multilingual datasets, recommending the use of **multilingual rerank v3.0** for better handling of varied languages.
   - They indicated that if relevance scores are not necessary, they shouldn't be used, as it can simplify queries.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1288186748253507685)** (2 messages): 

> - `Self-promotion concerns`
> - `Cohere in embedded systems` 


- **Self-promotion is not welcome**: A member emphasized that this channel isn't a place for self-promotion, requesting others to remove their links.
   - *This isn’t a place to advertise yourself* was the core sentiment expressed.
- **Inquiry about Cohere in embedded systems**: A member asked if there are any examples of using **Cohere** in **embedded systems**.
   - The question indicates an interest in practical applications of Cohere technology beyond typical use cases.


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1288193044750930014)** (1 messages): 

> - `Cohere Toolkit updates`
> - `Chat features`
> - `File type support`
> - `User feedback`
> - `Team collaboration` 


- **Cohere Toolkit receives exciting updates**: This month, several back-end/UI issues have been fixed in the **Cohere Toolkit**, enhancing overall user experience.
   - Notable new features include options to pin/unpin chats, regenerate the last chatbot response, and support for **parquet** and **tsv** files with a [YouTube demo available](https://youtu.be/gdJ0abx9mvo).
- **New chat features make chatting easier**: Users can now easily pin/unpin chats in the **Cohere Toolkit**, facilitating better conversation management.
   - Additionally, a feature to regenerate the last chatbot response has been added, allowing for quick follow-ups on discussions.
- **Support for multiple file types introduced**: The latest update to the **Cohere Toolkit** now supports both **parquet** and **tsv** file formats, enhancing data handling capabilities.
   - *This new support opens up more possibilities for users working with different data structures and formats.*
- **User feedback is encouraged for further development**: Users are welcomed to share their feedback and new ideas to continue improving the **Cohere Toolkit**.
   - *The development team appreciates community inputs and collaboration, expressing gratitude for discussions and code reviews.*



**Link mentioned**: <a href="https://youtu.be/gdJ0abx9mvo)">Cohere Toolkit demo 09.2024</a>: no description found

  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1288179701696696421)** (1 messages): 

> - `James Cameron`
> - `Stability AI Board of Directors`
> - `Transforming visual media`
> - `Generative AI`
> - `Cinematic technology` 


- **James Cameron joins Stability AI Board**: Legendary filmmaker **James Cameron** has joined the **Stability AI Board of Directors**, announced by CEO Prem Akkaraju. This addition signifies a pivotal move in Stability AI's mission to transform visual media.
   - Cameron's experience in merging **cutting-edge technology** with storytelling will enhance Stability AI's efforts in creating a comprehensive AI pipeline for creators.
- **Cameron's impact on cinematic technology**: As a pioneer in visual effects, Cameron is known for films such as **The Terminator** and **Avatar**, pushing boundaries in cinematic technology. His unique perspective aligns with Stability AI's focus on blending technological advancement with creativity.
   - By joining Stability AI, Cameron aims to further **revolutionize storytelling** through innovative AI solutions for visual media.



**Link mentioned**: <a href="https://stability.ai/news/james-cameron-joins-stability-ai-board-of-directors">James Cameron, Academy Award-Winning Filmmaker, Joins Stability AI Board of Directors &mdash; Stability AI</a>: Today we announced that legendary filmmaker, technology innovator, and visual effects pioneer James Cameron has joined our Board of Directors.

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1287873638716145827)** (41 messages🔥): 

> - `FNAF Loras creation`
> - `SDXL performance with GPU`
> - `Prompt engineering strategies`
> - `ControlNet applications`
> - `OpenPose editor integration` 


- **FNAF Loras creation request**: A member is seeking fellow **FNAF** fans to help create some **Loras** for the game.
   - *Anyone interested in collaborating on this project?*
- **SDXL performance boosted with 3090 EGPU**: A user reported finally purchasing a **3090 EGPU** to enhance their **SDXL** gameplay experience despite past failures with similar products.
   - *Frustrations about the quality of certain gaming boxes were shared, noting past issues with **Aurus** brands.*
- **Discussion on prompt engineering effectiveness**: Members discussed the efficiency of using **identical inputs** for **text_g** and **text_l** in SDXL, with some skepticism about their effectiveness.
   - *One member suggested focusing on nouns only, citing a **paper** indicating they have a more significant impact than adjectives.*
- **ControlNet's guiding capabilities**: A user inquired about **ControlNet**, to which another explained it is a method to guide image generation, especially for poses.
   - *It's noted that specifying details can be challenging with just language alone.*
- **OpenPose editor installation issues**: A user reported problems with the **OpenPose editor** in Forge, receiving advice that it might need a specific installation command to work properly.
   - *Clarification was offered about running **pip install basicsr** inside the virtual environment.*



**Link mentioned**: <a href="https://tenor.com/view/insideout-joy-hi-hey-hello-gif-11341448685299692120">Insideout Joy GIF - InsideOut Joy Hi - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1288258847303008297)** (1 messages): 

> - `LlamaParse`
> - `Fraudulent Sites` 


- **Beware of Fraudulent LlamaParse Site**: *Warning*: llamaparse dot cloud (we're not linking to it!) is a **fraudulent site** masquerading as a **LlamaIndex** product.
   - The **real LlamaParse** can be found at [cloud.llamaindex.ai](https://cloud.llamaindex.ai).
- **Identifying Real Products**: Users are advised to always verify the authenticity of **LlamaIndex** products to avoid falling for scams.
   - Staying informed can help users ensure they are using the correct services and avoid fraudulent alternatives.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1287857686402302045)** (5 messages): 

> - `LitServe framework`
> - `AI product manager`
> - `LlamaIndex workflows workshop`
> - `Llamaparse fraudulent site`
> - `AWS Gen AI Loft` 


- **LitServe simplifies serving LLMs**: The **LitServe** framework from @LightningAI, built on FastAPI, helps serve and scale LLM models effectively, demonstrated using [LlamaIndex](https://t.co/Xikqk20peW) in a quick demo.
   - This setup hosts a simple **RAG** server locally against **Llama 3.1**.
- **Create an AI Product Manager in 50 lines!**: An **AI product manager** can be built in just 50 lines of code using @llama_index and @composiohq, featuring email feedback reading and **Slack** notifications.
   - If approved, it seamlessly integrates feedback into the **Linear** board for requested edits, showcasing the power of the function calling agent architecture.
- **Workshop on Context-Augmented Agents**: An in-depth workshop by @AIMakerspace introduces the architecture of **LlamaIndex workflows** for building context-augmented agents.
   - Participants can learn to construct an agentic corrective **RAG** application through step-based, event-driven workflows.
- **Beware of fraudulent LlamaParse site**: A warning has been issued regarding a fraudulent site masquerading as **LlamaIndex's** LlamaParse, indicating it as **not legitimate**.
   - The official LlamaParse can be found at [this link](https://t.co/jM9ioNJuv3) to avoid confusion.
- **RAG and Agents discussion at AWS Gen AI Loft**: @seldo will discuss **RAG** and **Agents** at the AWS Gen AI Loft, just before the larger ElasticON conference with **@elastic**.
   - The session will also cover how **Fiber AI** utilizes **Elasticsearch** for efficient B2B prospecting, providing valuable networking opportunities.



**Link mentioned**: <a href="https://t.co/jM9ioNJuv3">LlamaCloud</a>: no description found

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1287882844072448043)** (35 messages🔥): 

> - `Approximate Metadata Filtering`
> - `Human-in-the-Loop Workflows`
> - `Postgres and pgvector`
> - `Web Crawling for Embedding` 


- **Exploring Approximate Metadata Filtering**: There was a discussion on using approximate metadata filtering in `workflows` for RAG, highlighting the challenge of constructing dynamic filters based on user queries.
   - Members noted that `MilvusVectorStore` may not support approximate filters and suggested defining Pydantic objects to help create filterable queries.
- **Challenges with Human-in-the-Loop Workflows**: Members explored implementing human-in-the-loop (HITL) interactions via nested workflows in a websocket context, addressing how to yield control back to the user after specific events.
   - One member suggested using an event-driven approach to manage user responses dynamically as the workflow streams events.
- **Transitioning from Postgres with pgvector**: Discussion arose about transitioning from Postgres with pgvector for hybrid search to pgvector.rs for better performance, but some features seem absent in the current LlamaIndex implementation.
   - A member estimated that implementing support for sparse search options might take around a day's work if one has a good understanding of pgvector.rs.
- **Crawling Web Pages for RAG**: Members sought advice on technologies for crawling web pages for embedding, questioning if others used custom solutions like Puppeteer or preferred tools like Firecrawl or Crawlee.
   - This inquiry reflects a broader interest in effective techniques for integrating web crawled data into retrieval-augmented generation (RAG) pipelines.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/">Qdrant Vector Store - Metadata Filter - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1287851911361597551)** (14 messages🔥): 

> - `Blendtain feedback`
> - `Playlist generator by dykyi_vladk`
> - `Study Machine Learning together`
> - `Impressions of GANs, CNNs, and ViTs` 


- **User Feedback on Blendtain**: A user expressed excitement about Blendtain's idea but highlighted that it cuts off messages and suggested adding a setting to adjust message length.
   - Another user responded positively, simply stating, *'yeah thxxx'*.
- **dykyi_vladk's Playlist Generator**: [Adify.pro](https://adify.pro) was introduced by dykyi_vladk as a playlist generator that creates playlists based on user prompts.
   - The creator expressed pride in the project, calling it *'my coolest thing'*.
- **Collaborative Learning on Machine Learning**: dykyi_vladk invited others to DM him if they are interested in studying machine learning together.
   - This initiative was shared in a friendly tone, encouraging collaboration among members.
- **Discussion on Image Task Algorithms**: A member remarked on the fluctuating dominance of **GANs**, **CNNs**, and **ViTs** as the top algorithms for image processing tasks.
   - They sought confirmation on this observation and requested a visual timeline of these algorithmic shifts.



**Link mentioned**: <a href="https://adify.pro">Adify</a>: no description found

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1287877652824199278)** (12 messages🔥): 

> - `muP Transfer`
> - `HyperCloning Method`
> - `SDXL Unet`
> - `Positional Encoding in UNet`
> - `Sliding Window Attention` 


- **EleutherAI's muTransfer project**: EleutherAI introduced a joint project with [Cerebras](https://cerebras.ai/) to spread implementation details of [muTransfer](https://github.com/microsoft/mup) and provide a port to the [nanoGPT library](https://github.com/EleutherAI/nanoGPT-mup). This effort aims to make Maximal Update Parameterization (**μP**) more accessible and reduce training costs.
   - However, some members speculate that muP might already be somewhat **dated** and may not be the best approach moving forward.
- **HyperCloning enhances training efficiency**: A new **HyperCloning** method proposed in an [arXiv paper](https://arxiv.org/abs/2409.12903) shows how initializing large language models with smaller pre-trained models can lead to better training times and final accuracy. This method expands the parameters of small models to larger ones while retaining functionality.
   - Members highlighted that tiling original weights into larger parameters should yield better and faster results, making this expansion more **reproducible**.
- **Positional Encoding Concerns in SDXL Unet**: In discussions about the **SDXL Unet**, it was noted that the model does not use positional encodings for image coordinates, employing adanorm for crop coordinates instead. Convolutional layers inherently encode spatial positions, which some argue makes explicit positional embeddings unnecessary.
   - Despite these claims, another member mentioned that sliding window attention techniques, like **longformer**, also leverage these benefits but still incorporate positional encoding.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.eleuther.ai/mutransfer/">The Practitioner&#39;s Guide to the Maximal Update Parameterization</a>: Exploring the implementation details of mutransfer</li><li><a href="https://arxiv.org/abs/2409.12903">Scaling Smart: Accelerating Large Language Model Pre-training with Small Model Initialization</a>: The pre-training phase of language models often begins with randomly initialized parameters. With the current trends in scaling models, training their large number of parameters can be extremely slow ...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1287911909848256642)** (15 messages🔥): 

> - `Nvidia's new synthetic data model`
> - `MMLU performance`
> - `Fine-tuning and inferencing challenges`
> - `Context management in LLMs`
> - `Run Pod CUDA error` 


- **Excitement around Nvidia's new synthetic data model**: Discussion sparked by a mention of Nvidia's **51B synthetic data model**, highlighting a high **MMLU** performance that could enhance applications.
   - *It would be fun to try fine-tuning and inferencing with it,* mentioned a member, indicating interest in practical applications.
- **Challenges with auto chunking in conversations**: A member argued against the practicality of **auto chunking** during conversations, stating *Imagine your convo split in half midway. The context is lost*.
   - Another highlighted that it's how systems like **ST** or **Kobold** typically handle overflowing context by maintaining the first message and removing oldest messages.
- **The value of dynamic context management**: There was a proposition on how managing sliding context dynamically could help **LLMs learn** to handle conversational shifts organically.
   - One shared the potential benefits, suggesting it could provide a solution for when context limits are exceeded.
- **Inquiry on experiences with Run Pod**: A member asked if anyone had experienced success with **Run Pod**, mentioning struggles with a **CUDA error**.
   - The **illegal CUDA error** was noted, drawing attention to potential technical hurdles with the platform.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1287911909156454421)** (3 messages): 

> - `Qwen 2.5`
> - `Axolotl support` 


- **Qwen 2.5 confirmed for Axolotl**: A member confirmed that **Qwen 2.5** is indeed supported on **Axolotl**.
   - They noted that Qwen 2.5 shares the **same architecture** as its predecessor, **Qwen 2**.
- **User inquiry about Qwen 2.5 support**: A user inquired about the support status of **Qwen 2.5** on **Axolotl**.
   - This initiated the discussion leading to confirmation from a member.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1287881023669010517)** (4 messages): 

> - `Fine-tuning spikes`
> - `Rope scaling`
> - `Llama3.1 setup`
> - `Qwen2.5 configurations` 


- **Spikes in Fine-Tuning Observed**: A member reported experiencing a **spike** during their fine-tune on a **100K row dataset** and sought logging output to correlate this spike with specific data rows.
   - *Unfortunately not at the moment*, was the response regarding logging help.
- **Rope Scaling Recommended**: Another member advised using **rope scaling** for managing memory efficiency, emphasizing that increasing **seq len** alone demands significant **vRAM**.
   - This method could help alleviate issues related to the spikes experienced during fine-tuning.
- **Setup Inquiry for Llama3.1**: A user asked if their fine-tuning setup for **Llama3.1**—using a **4K sequence length** with a **3x factor** for a **120K window**—was correct.
   - They sought confirmation on whether the configuration was optimal for their needs.
- **Qwen2.5 Context Configuration**: The conversation included a setup inquiry for **Qwen2.5**, labeling it with a **3K sequence length** and a **4x factor** for a **120K context**.
   - The ratios suggest a thoughtful approach to maximizing model efficiency while setting up configurations.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1287960576475136012)** (2 messages): 

> - `Qwen 2.5`
> - `Axolotl Support` 


- **Qwen 2.5 Supported for Text on Axolotl**: A member confirmed that **Qwen 2.5** should be supported on **Axolotl** for normal text processing.
   - However, they mentioned that support for vision features may not be available.
- **Vision Support for Qwen 2.5 in Axolotl**: Despite text support, the same member expressed doubt about the vision capabilities of **Qwen 2.5** on **Axolotl**.
   - This indicates potential limitations when handling visual input compared to text.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1288026271795908630)** (17 messages🔥): 

> - `LangChain Pydantic Compatibility`
> - `GraphRecursionError in LangGraph`
> - `LLM Friendly Documentation for LangChain`
> - `Comparison between Mistral and Mixtral` 


- **LangChain Pydantic Compatibility Issue**: Users are encountering an error while importing `ChatOpenAI` from `langchain_openai`, stating that the `__modify_schema__` method is unsupported in Pydantic v2.
   - It is advised to check the version of Pydantic and use `__get_pydantic_json_schema__` instead, as noted in the [LangChain documentation](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility/#1-passing-pydantic-objects-to-langchain-apis).
- **GraphRecursionError in LangGraph**: A `GraphRecursionError` is raised when the recursion limit of 25 is reached in LangGraph applications, preventing infinite loops.
   - Users are encouraged to increase the limit in their configuration, similar to the solution provided in a [GitHub issue comment](https://github.com/langchain-ai/langchain/issues/18598).
- **Request for LLM Friendly Documentation**: A user inquired about LLM-friendly context text documents to enhance productivity with LangChain.
   - This topic was addressed in a thread initiated by another member, indicating ongoing discussions about resources for LangChain.
- **Mistral vs Mixtral Open Source Models**: A member asked which model currently offers the best open-source solution for self-hosting between **Mistral** and **Mixtral**.
   - This indicates interest within the community regarding comparative performance and usability of self-hosted models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility/#1-passing-pydantic-objects-to-langchain-apis>).">How to use LangChain with different Pydantic versions | 🦜️🔗 LangChain</a>: - Pydantic v2 was released in June, 2023 (https://docs.pydantic.dev/2.0/blog/pydantic-v2-final/).</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/chat/openai/#instantiation>).">ChatOpenAI | 🦜️🔗 Langchain</a>: OpenAI is an artificial</li><li><a href="https://github.com/langchain-ai/langchain/issues/18598>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-3>)">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#maxiterations>).">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts: -
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1287853629260435629)** (10 messages🔥): 

> - `CPU Offloading for Optimizers`
> - `Performance Optimization Techniques`
> - `Paged Adam vs Torchao CPUOffloadOptimizer` 


- **Confusions Around CPU Offloading in Optimizers**: Discussion arose about why CPU offloading for the optimizer isn't being utilized, referencing [this old issue](https://github.com/pytorch/pytorch/issues/74588) that mentioned slowdowns.
   - One member suggested that using CPU offloading with PagedAdam optimizes performance, while the need for a PR was highlighted to modify usage of optimizers in single-device fine-tuning.
- **Comparative Analysis of Optimizer Methods**: It was noted that using torchao's CPUOffloadOptimizer doesn't pair well with optimizer in backward, leading to questions about faster alternatives like Adam.
   - Recommendations included trying `offload_gradients=True` to achieve gradient memory savings, while CPU computation overlaps with GPU processing for better performance, as detailed in this [PR](https://github.com/pytorch/ao/pull/584).
- **CUDA MODE Community Invitation**: A suggestion was made to join the GPU MODE Discord group for members interested in performance optimization, stating there are more qualified individuals available for help.
   - The link shared for joining is [here](https://discord.gg/jNsm4H44).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py">torchtune/recipes/full_finetune_single_device.py at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/issues/74588">[FSDP]  using CPUOffload creates 3-10x slowdown due to slow cpu optimizer step/update · Issue #74588 · pytorch/pytorch</a>: 🐛 Describe the bug Create simple distributed model Wrapper model with FSDP. Using stateful optimizer ala Adam(W) run without CPUoffload and profile/time. Then run with CPUOffload and see that perfo.....</li><li><a href="https://github.com/pytorch/torchtune/issues/1576">replace adamW and pagedadam with 8bitpagedadam or torchao CPUOffloadOptimizer · Issue #1576 · pytorch/torchtune</a>: Apparently there is no reason to use paged adam instead of the 8bit version. We should replace it. Also, full finetune single device should use paged adam, instead of adamw, for better memory. For ...</li><li><a href="https://github.com/pytorch/ao/pull/584">Optimizer CPU offload for single GPU training by gau-nernst · Pull Request #584 · pytorch/ao</a>: Background Currently there is no simple way to do optimizer CPU offload for single GPU training, although such feature exists for FSDP. DeepSpeed ZeRO-Offload can work with single GPU, but it requi...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1288116767000104960)** (2 messages): 

> - `Distributed Training`
> - `Planetary Brain Concept`
> - `DisTrO Project` 


- **Exploring the Concept of a Planetary Brain**: A member playfully questioned how far we are from achieving **NET=1**, where tinyboxes could connect to form a **planetary brain** through distributed training.
   - This suggests a future where collective intelligence could enable **distributed training** on a global scale.
- **Introduction to DisTrO for Distributed Training**: Discussion highlighted the **[DisTrO project](https://github.com/NousResearch/DisTrO)**, which focuses on enabling Distributed Training Over-The-Internet.
   - This project aims to revolutionize how models can be trained through cooperation across the internet.



**Link mentioned**: <a href="https://github.com/NousResearch/DisTrO">GitHub - NousResearch/DisTrO: Distributed Training Over-The-Internet</a>: Distributed Training Over-The-Internet. Contribute to NousResearch/DisTrO development by creating an account on GitHub.

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1288157299235360832)** (7 messages): 

> - `AttributeError in Tensor`
> - `Tinygrad version issues`
> - `Model architecture insights` 


- **AttributeError: 'Tensor' lacking cross_entropy**: A user encountered an 'AttributeError' indicating that the **Tensor** object has no attribute 'cross_entropy'. They shared a code snippet indicating where the error arises in the training step function.
   - This sparked discussion about potential reasons for the error, including possible issues with the Tensor implementation.
- **Tinygrad version debate**: Another user inquired about the **Tinygrad** version being used, prompting a response that the poster had recently updated to version **0.9.2** from Git.
   - It was pointed out that this version does not support the functions needed, recommending upgrading to the latest version from master for additional functionalities.
- **Model architecture and training**: User shared a model architecture that includes multiple convolutional layers and a flattening operation followed by linear layers. The conversation highlighted the design choices made in constructing the model for training performance.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1287939914628862105)** (9 messages🔥): 

> - `Open Interpreter Updates`
> - `LLM based Browser Automation`
> - `Community Engagement`
> - `GitHub Resources`
> - `Feedback on Project Activity` 


- **Open Interpreter is Not Dead**: A member reassured that Open Interpreter is actively receiving updates on [GitHub](https://github.com/OpenInterpreter/open-interpreter), showcasing ongoing development.
   - Additionally, there has been significant activity surrounding the project '01', aimed at integrating a dedicated voice assistant mode, as noted [here](https://github.com/OpenInterpreter/01).
- **Exploring LLM for Browser Automation**: A member shared insights about utilizing Open Interpreter for LLM-based browser automation and form submissions, confirming that it works, but with limitations due to the complexity of tasks.
   - They suggested using Playwright for better results and shared a [prompt example](https://github.com/morisy/openinterpreter-configs/blob/main/foiaportalassistant.yaml) they have been refining.
- **Community Enthusiasm Remains**: Despite concerns about the project's activity, members continued to discuss practical use cases, with one expressing eagerness to automate submissions to directories using shared prompts.
   - Another member reaffirmed that the community remains engaged, responding to inquiries and sharing experiences in using the tool.
- **Upcoming Community Event Announced**: A member teased an upcoming event related to Open Interpreter, sharing a [Discord link for more details](https://discord.gg/open-interpreter-1146610656779440188?event=1288234745477726238).
   - This announcement prompted excitement among users, indicating ongoing community activity.
- **Project Perception Shift Discussed**: In response to a query about the project's status, a member humorously noted that the original questioner might not be fully updated on the project's progress.
   - This interaction highlights varying perceptions within the community regarding the project's liveliness.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/morisy/openinterpreter-configs/blob/main/foiaportalassistant.yaml">openinterpreter-configs/foiaportalassistant.yaml at main · morisy/openinterpreter-configs</a>: A place to dump some notes, config files, etc from my experimentation with Open Interpreter. - morisy/openinterpreter-configs</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: The #1 open-source voice interface for desktop, mobile, and ESP32 chips.</a>: The #1 open-source voice interface for desktop, mobile, and ESP32 chips. - OpenInterpreter/01
</li>
</ul>

</div>
  

---



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
