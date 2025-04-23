---
id: 4caf2d11-37e1-4ed4-9a10-356e62c70c59
title: '[AINews] $100k to predict LMSYS human preferences in a Kaggle contest'
date: '2024-05-03T22:09:28.423042Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-not-much-happened-today-3049
---

<!-- buttondown-editor-mode: plaintext -->> AI News for 5/2/2024-5/3/2024. We checked 7 subreddits and [**373** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**418** channels, and **5847** messages) for you. Estimated reading time saved (at 200wpm): **642 minutes**.

It's been a quiet week for AI news. [This](https://lmsys.org/blog/2024-05-02-kaggle-competition/) is a fun new Kaggle challenge:

> You'll work with a dataset from the Chatbot Arena, containing conversations and user preferences across various LLMs. By developing a model that accurately predicts human preferences, you'll contribute to improving chatbot performance and alignment with user expectations. The training dataset includes over 55,000 real-world user and LLM conversations and user preferences, with personally identifiable information removed. Your solution submission will be tested on a hidden test set of 25,000 samples.

> The competition will run until August 5th, with a total prize of $100,000, featuring a $25,000 prize for 1st place, 20,000 prizes for 2nd through 4th places, and a 15,000 prize for 5th place.

---

**Table of Contents**

[TOC] 


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**LLM Model Releases and Benchmarks**

- **Llama 3 Models**: [@DrJimFan](https://twitter.com/DrJimFan/status/1786429467537088741) announced DrEureka, an LLM agent that writes code to train robot skills in simulation and enables zero-shot transfer to the real world. [@GroqInc](https://twitter.com/awnihannun/status/1786066330501956053)'s Llama 3 70B model is breaking performance records at **$0.65/1M input and $0.9/1M output tokens**. [@bindureddy](https://twitter.com/bindureddy/status/1786019505027608646) notes Llama 3 models from Groq are leading while OpenAI focuses on hyping GPT-5.
- **Benchmarking LLMs**: [@DrJimFan](https://twitter.com/DrJimFan/status/1786054643568517261) suggests 3 types of LLM evaluations that matter: **privately held test sets with publicly reported scores by trusted 3rd parties** like [@scale_AI](https://twitter.com/scale_AI), **public comparative benchmarks** like [@lmsysorg](https://twitter.com/lmsysorg)'s Chatbot Arena, and **privately curated internal benchmarks** for each company's use cases. [@percyliang](https://twitter.com/percyliang/status/1786256267138478475) notes some models perform poorly with certain prompts on GSM8K benchmark.
- **Open Source Evaluator LLMs**: [@seungonekim](https://twitter.com/ShayneRedford/status/1786455899059503448) introduces Prometheus 2, open source evaluator LLMs that **closely mirror human and GPT-4 judgments** and support direct assessment and pairwise ranking formats. They **outperform proprietary LMs like GPT-4 and Claude 3 Opus** on building LM judges.

**Datasets and Benchmarking**

- **GSM1K Dataset**: [@percyliang](https://twitter.com/percyliang/status/1786256267138478475) discussed how models are sensitive to prompts on the new GSM1K dataset, needing sampling and majority voting to reduce noise. Some perform poorly with extra hints.
- **WildChat1M ChatGPT Logs**: [@_akhaliq](https://twitter.com/_akhaliq/status/1786218700900557021) shared the WildChat dataset from AI2 with over 1M ChatGPT interaction logs in the wild. It has **2.5M turns, diverse prompts, many languages, and toxic examples**.
- **Kaggle Human Preference Prediction**: [@lmsysorg](https://twitter.com/lmsysorg/status/1786100697504833572) announced a $100K Kaggle competition to predict user preferences between LLM responses in their Chatbot Arena, based on a new dataset with 55K user/LLM conversations.
- **Contamination Database**: [@clefourrier](https://twitter.com/clefourrier/status/1785936450577375556) noted a new open database to track contamination of models and datasets to help select "safe" artifacts for model creation.


**Techniques for Efficient LLM Training and Inference**

- **LoRA for Parameter Efficient Fine-Tuning**: [@mobicham](https://twitter.com/_akhaliq/status/1786217595089105169) assesses viability of training and serving LLMs fine-tuned with quantized low rank adapters (LoRA) across 10 base models and 31 tasks. **4-bit LoRA models outperform base models by 34 points and GPT-4 by 10 points on average**. LoRAX inference server enables deploying multiple LoRA models on a single GPU.
- **Efficient Model Alignment with NeMo-Aligner**: [@NVIDIA](https://twitter.com/_akhaliq/status/1786222861666971804) introduces NeMo-Aligner, a scalable toolkit for efficient LLM alignment techniques like RLHF, DPO, SteerLM, SPIN. It **scales to hundreds of GPUs** for training large models.
- **Factuality-Aware Alignment to Reduce Hallucination**: [@mobicham](https://twitter.com/_akhaliq/status/1786229213357342980) proposes factuality-aware SFT and RL alignment to guide LLMs to output more factual responses. Training LLMs on new knowledge or unfamiliar texts can **encourage hallucination**.

**Multimodal and Long-Range LLMs**

- **Multimodal LLM for Automated Audio Description**: [@mobicham](https://twitter.com/_akhaliq/status/1786219554068169162) introduces an automated audio description pipeline using multimodal instruction-following capacities of GPT-4V. It produces ADs **compliant with natural language production standards** while maintaining contextual consistency.
- **Extending LLM Context Windows**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1786101710022201697) reports extending Llama-3-8B's context **10-fold to 80K tokens overnight** using only 3.5K synthetic QA pairs. The resulting model **excels at long-context tasks** like book QA and summarization, rivaling GPT-4.
- **Consistent Long-Range Video Generation**: [@mobicham](https://twitter.com/_akhaliq/status/1786213056088793465) proposes StoryDiffusion framework for consistent long-range image/video generation from text. It introduces **Consistent Self-Attention and Semantic Motion Predictor** to maintain consistency across generated frames.

**Emerging Architectures and Training Paradigms**

- **Kolmogorov-Arnold Networks as MLP Alternative**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785963059938234555) reports Kolmogorov-Arnold Networks (KANs) as a novel alternative to MLPs. KANs use **learnable activation functions on edges** and **replace weights with learnable splines**. They achieve higher accuracy with fewer parameters and avoid curse of dimensionality.
- **Apple's On-Device LLMs and AI-Enabled Browser**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785971852294160530) notes Apple introducing OpenELM, a family of small on-device LLMs, and an AI-enabled Safari browser at WWDC. **On-device LLMs enable free inference** without API calls.

**Miscellaneous**

- **WildChat1M ChatGPT Interaction Dataset**: [@mobicham](https://twitter.com/_akhaliq/status/1786218700900557021) introduces WildChat1M, a dataset of **1M user-ChatGPT conversations with over 2.5M interaction turns**. It offers diverse prompts, multiple languages, and captures various use cases and user behaviors across regions.
- **Open Source Libraries for ML Deployment**: [@dl_weekly](https://twitter.com/dl_weekly/status/1786213589033861206) shares a curated list of open source libraries to deploy, monitor, version and scale machine learning models in production.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Model Releases and Updates**

- **Nvidia releases ChatQA-1.5**: In /r/LocalLLaMA, Nvidia has published ChatQA-1.5, a competitive Llama3-70B QA/RAG fine-tune that [**excels at conversational question answering and retrieval-augmented generation**](https://www.reddit.com/r/LocalLLaMA/comments/1cidg4r/nvidia_has_published_a_competitive_llama370b/). It outperforms the vanilla RAG baseline on benchmarks like FinanceBench.
- **Stability AI's Stable Diffusion 3 release timeline unclear**: In /r/StableDiffusion, there is speculation about the [**release timeline for Stable Diffusion 3 weights**](https://www.reddit.com/r/StableDiffusion/comments/1ciyzn5/sd3_weights_are_never_going_to_be_released_are/), with some predicting a Monday release based on tweets, while others doubt a full release is imminent.
- **Anthropic's Claude Opus and Udio generate standup comedy**: Claude Opus and Udio, Anthropic's AI models, were used to [**generate a standup comedy routine about the future of r/singularity after AGI**](https://www.udio.com/songs/rDL4XviHDbyxug1FXK9vXP).

**AI Applications and Demos**

- **Progress towards hyper-realistic holodecks**: Researchers who developed gaussian splatting, a technique that represents 3D geometry using gaussian splats instead of triangle meshes, have made [**new progress enabling fast rendering from any angle**](https://v.redd.it/kpyrc6wbq3yc1), bringing hyper-realistic holodecks closer to reality.
- **AI-generated music video commissioned by Paul Trillo**: [SORA, an AI-generated music video](https://v.redd.it/dw6y9qpe34yc1) commissioned by Paul Trillo for the song "The Hardest Part" by Washed Out, **showcases the current state of AI video generation** with dream-like visuals and transitions.
- **AI-powered CRISPR tool creates new gene-editing capabilities**: According to a Nature article, an ['ChatGPT for CRISPR' creates new gene-editing tools](https://www.nature.com/articles/d41586-024-01243-w), **expanding the capabilities of gene editing**.
- **Jetbrains IDEs now use local AI model for code suggestions**: Jetbrains IDEs are [now using a local 0.1B model with 1.5K token context for single-line code suggestions](https://blog.jetbrains.com/blog/2024/04/04/full-line-code-completion-in-jetbrains-ides-all-you-need-to-know/), with pre and post-processing to **ensure useful and correct suggestions**.
- **Panza: Personalized LLM email assistant**: [Panza is a personalized LLM email assistant](https://www.reddit.com/r/MachineLearning/comments/1ciqvqw/p_panza_a_personal_email_assistant_trained_and/) that can be **trained and run locally, mimicking the user's writing style** by fine-tuning on their email history. It pairs the fine-tuned LLM with a retrieval-augmented generation component.

**AI Societal Impact and Concerns**

- **Humans now share the web equally with bots**: According to a report, [humans now share the web equally with bots](https://www.independent.co.uk/tech/dead-internet-web-bots-humans-b2530324.html), raising fears of a "dead internet" as **sites like Twitter/X become overrun with automated accounts**. Some predict this means the end of user-generated content aggregation sites.
- **Data centers require immense power**: [Data centers now require a nuclear reactor's worth of power](https://www.bloomberg.com/news/articles/2024-05-02/data-centers-now-need-a-reactor-s-worth-of-power-dominion-says) according to Dominion Energy, **highlighting the immense energy demands** of large-scale computing infrastructure. 
- **Microsoft bans police use of facial recognition AI**: [Microsoft has banned US police departments from using its enterprise AI tool for facial recognition](https://techcrunch.com/2024/05/02/microsoft-bans-u-s-police-departments-from-using-enterprise-ai-tool/) amidst **ongoing concerns about the ethical use of AI in law enforcement**.

**AI Research and Benchmarking**

- **Junior researchers have strong presence at top ML conferences**: In /r/MachineLearning, it's noted that [juniors (undergrads and early PhD students) have many papers at top ML conferences](https://www.reddit.com/r/MachineLearning/comments/1cidsz7/d_why_do_juniors_undergraduates_or_first_to/) because they **receive a lot of support and mentorship**. Leading projects is still a huge accomplishment and shows they have the skills to excel.
- **Few papers at top ML conferences are groundbreaking**: Also in /r/MachineLearning, one researcher estimates [their own accepted work is good but not highly impactful](https://www.reddit.com/r/MachineLearning/comments/1cin6s8/d_something_i_always_think_about_for_top/), more like "one more brick in the wall." **Game-changing papers like "Attention is All You Need" are rare**.
- **Staged dataset releases could help detect benchmark contamination**: A suggestion in /r/MachineLearning is that [benchmark creators should release datasets in stages](https://www.reddit.com/r/MachineLearning/comments/1cilnzv/d_benchmark_creators_should_release_their/) to **enable checking for benchmark contamination in models** by comparing performance on subsets released before vs after the model's training data cutoff.
- **spRAG: Open-source RAG system for complex real-world queries**: [spRAG is an open-source retrieval-augmented generation system](https://www.reddit.com/r/MachineLearning/comments/1cikkw2/p_sprag_opensource_rag_implementation_for/) designed to **handle complex real-world queries over dense text** like legal docs and financial reports. It outperforms the RAG baseline on challenging benchmarks like FinanceBench.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Large Language Model (LLM) Advancements and Challenges**

- **Exploring LLM Capabilities**: Discussions around **[LLaMA 3](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)** achieving **1040k context length**, [Hermes 2 Pro](https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B) with advanced QA and Function Calling, and **[llm.c](https://github.com/karpathy/llm.c/discussions/344)** hitting **167K tokens/second**. However, **[quantization](https://arxiv.org/abs/2404.14047)** seems to **hurt LLaMA 3 quality**.

- **Multilingual and Multimodal LLMs**: Exploring how LLMs handle **[multilingual inputs](https://arxiv.org/abs/2402.18815v1)**, with English potentially used as a pivot language. Multimodal capabilities like **[Suno's music generation](https://arxiv.org/abs/2404.10301v1)** and **[AI Vtubing](https://github.com/tegnike/nike-ChatVRM)** were also discussed.

- **LLM Benchmarking and Evaluation**: Concerns about **[benchmark dataset leakage](http://arxiv.org/abs/2404.18824)**, with suggestions for **[fresh benchmark questions](https://arxiv.org/abs/2405.00332)**. The release of **[Prometheus 2](https://huggingface.co/papers/2405.01535)**, an evaluator LLM, aims to assess other LLMs transparently.

**2. AI Model Fine-tuning and Optimization Strategies**

- **Unsloth AI Enables Near-Full Finetuning**: The Unsloth AI community explored possibilities for near-full finetuning by setting all parameters except layernorms to trainable, outperforming standard Hugging Face implementations. Discussions also covered dataset formatting for optimization and unofficial full finetuning tactics. Key resources included [Unsloth's Colab notebooks](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) and [finetuning guides](https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices).

- **Retrieval Augmented Generation (RAG)**: Guides on building **[efficient RAG data stacks](https://t.co/jez7g9hADV)** and **[LangChain's RAG integration](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da)** for intelligent applications. Discussions on RAG's role in **[LlamaIndex's introspective agents](https://t.co/X8tJGXkcPM)**.

- **Optimizing Training Pipelines**: [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583) improved data preprocessing parallelism. Leveraging **[DeepSpeed Stage 3](https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167)** and **[Flash Attention](https://github.com/ggerganov/llama.cpp/pull/5021)** for efficient large model training.

**3. Open Source AI Frameworks and Libraries**

- **LLM Deployment Solutions**: Discussions on **[LangChain](https://github.com/langchain4j/langchain4j)** Java port, **[Dragonfly integration](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly)**, and an **[AutoTrain](https://github.com/huggingface/autotrain-advanced)** config release enabling model training without code.

- **AI Development Frameworks**: [Modular](https://www.modular.com/blog/whats-new-in-mojo-24-3-community-contributions-pythonic-collections-and-core-language-enhancements) celebrated Mojo 24.3 with community contributions. **[GreenBitAI](https://github.com/GreenBitAI/green-bit-llm)** introduced a toolkit enhancing PyTorch, while **[BitBlas](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp)** offers fast gemv kernels.

- **Open Source AI Projects**: Releases like **[LM Studio's CLI tool 'lms'](https://github.com/lmstudio-ai/lms)**, **[Mojo-pytest v24.3 support](https://github.com/guidorice/mojo-pytest/issues/9)**, **[NuMojo](https://github.com/MadAlex1997/NuMojo)** tensor library, and **[Prism CLI updates](https://github.com/thatstoasty/prism)** showcase community-driven development.

**4. AI Hardware Acceleration and Optimization**

- **GPU Optimization Techniques**: Discussions on **[Triton](https://github.com/openai/triton/pull/3813)** gather procedures, **[CUDA streams](https://github.com/karpathy/llm.c/pull/343)**, and **[fused classifiers](https://github.com/karpathy/llm.c/pull/343)** in **[llm.c](https://github.com/karpathy/llm.c/discussions/344)**. Exploring **[FP6 support](https://github.com/pytorch/ao/issues/208)** in PyTorch AO.

- **Specialized Hardware**: Interest in **[Rockchip RK3588 SBCs](https://github.com/rbrisita/01/tree/rknn)** showing **250% Whisper RKNN performance boost**. Curiosity about **[CHERI security capabilities](https://youtu.be/_QxXiTv1hH0?t=933)** enabling fast IPC and simplifying hardware design.

- **Raspberry Pi and Embedded AI**: The **[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742)** channel garnered interest, while the **[llama-farm project](https://discord.com/channels/1122748573000409160/1137456826733047908/1235763541642838069)** aims to connect local `Ollama` instances to the cloud.

**5. Misc**

- **LM Studio Introduces CLI Tool and Addresses Bugs**: LM Studio launched `lms`, a new CLI tool to manage local LLMs, starting/stopping servers, and debugging. It requires **LM Studio 0.2.22+** and is [open source on GitHub](https://github.com/lmstudio-ai/lms). The latest update also fixed a bug causing entire context to be included in model responses. Users explored running LM Studio headlessly and embedding it in scalable server solutions.

- **Quantization Challenges and Context Expansion in LLMs**: Quantization's impact on **LLaMA 3** performance was a hot topic, with a [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/) and [research paper](https://arxiv.org/abs/2404.14047) suggesting significant quality loss. Meanwhile, **LLama-3 8B** achieved over 1040k context length with Crusoe Energy's compute, and **Jamba-Instruct** from AI21 Labs expressed interest in much larger context windows.


---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**New Frontier in LLM Finetuning**: Community members discussed near-full finetuning possibilities with Unsloth, exploring the potential of setting all parameters except layernorms to trainable. While Unsloth is focused on addressing llama.cpp and GGUF conversions, particularly the quantization and loading checkpoint shards challenge, sentiment analysis enthusiasts received tips on formatting vast databases for LLM compatibility.

**Experimental Full Finetuning Tactics and Dataset Structuring**: Unofficial strategies to enable full finetuning on Unsloth were shared, demonstrating improved losses relative to standard Hugging Face implementations. Discussions also delved into ideal dataset structuring for optimization, suggesting strategies for handling multiple "rejected" responses.

**Phi 3 Executes in Browser, But Llama 3 Discord Absent**: A tweet [here](https://twitter.com/fleetwood___/status/1783195985893863578) demonstrated running Phi 3 in a web browser, while a member clarified that no dedicated Discord channel exists for Llama 3. Meanwhile, incorporating new roles in Llama 3 sparked debate, with `type=code` being a suggested alternative for `tool_call`.

**Adapting Llama 3 With Self-Discovery and Triton's TK-GEMM**: One ingenious user applied techniques from the Self-Discovery paper to enhance the reasoning capabilities of ChatGPT. Moreover, a PyTorch blog post highlighted Triton's FP8 GEMM to accelerate Llama 3 on NVIDIA H100 GPUs, promising optimization insights.

**Quantization Quandary and Finetuning Finesse**: Issues emerged when converting Llama 3 to GGUF, impacting fine-tuning data integrity, and similar problems arose when melding Lora with GGUF models. However, a pathway to understanding finetuning and model management is becoming clearer, with established community members suggesting the use of Unsloth's Colab notebooks for guidance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Slash Commands Get Ghosted**: Engineers observed the mysterious disappearance of the `/faq` command in the Discord commands, triggering a wave of jokes about its noticeable absence only after it was gone.

- **Graphical Debate: Nvidia vs. AMD**: A hot topic was the choice between Nvidia's 4080 and 3090 GPUs versus AMD's 7900xtx, with discussions centered around VRAM capacity and the merits of waiting for Nvidia's forthcoming 5000 series for future resilience.

- **Conversion Curiosity with RTX 4080**: Queries were raised about the time efficiency of an RTX 4080 in converting videos into anime-style using AI, with members seeking performance benchmarks for such tasks.

- **GPU Loyalties Split the Room**: Members heatedly debated the advantages of Nvidia GPUs over AMD for AI applications, with a few advocates for AMD drawing from their positive experiences, despite Nvidia's touted new Blackwell architecture.

- **Enhancement Enigmas: Text and Image Upscaling**: Various methods for AI-assisted text addition to images and image upscaling were shared, including applications like Davinci Resolve for text and upscaling tools like [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/) and the [Harrlogos XL](https://civitai.com/models/176555/harrlogos-xl-finally-custom-text-generation-in-sd) for Stable Diffusion's custom text generation.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Gradient Adornments in Conversations**: Discord members discussed advanced gradient techniques within PyTorch, where `create_graph=True` is employed for finer gradient details and Hessian-vector products. Techniques to estimate the Hessian's diagonal were mentioned, leveraging randomness for the estimations.

**Triton Trials and Triumphs**: Engineers faced challenges with `IncompatibleTypeErrorImpl` in Triton, but found solace in a `tl.cast` function fix after stumbling upon a gather function issue. Kernel debugging with PyTorch in PyCharm also proved problematic, even when setting `TRITON_INTERPRET` to `"1"`.

**Patching it Up with tinygrad**: Members shared a multi-GPU support patch for tinygrad, endorsing Nvidia's open drivers. A GitHub conundrum surfaced about the right way to install custom PyTorch and CUDA extensions, seeking clarity through examples in the PyTorch AO library's setup process.

**Catalyzing Community Contributions**: The Effort project on GitHub received accolades for its impactful structure, while GreenBitAI’s toolkit was introduced as an ML framework enhancing PyTorch. It includes innovative gradient calculation methods and a potentially useful gemv kernel for inference spotlighted in **bitblas**.

**torch woes and wins**: PyTorch developers debated build strategies and optimizations, from build times for linear algebra components to kernel performance. The idea of padding vocabulary size to fairly compete in performance benchmarks was deliberated, revealing the nuanced considerations needed for equitable measures.

**A Taste of LLM Innards**: The llm.c project reached new efficiencies with **167K tokens/second** using CUDA optimization techniques. Key discussions on CUDA streams, fused classifiers, and the strategic use of atom variables with scratch buffers highlighted the dense technical camaraderie. 

**Open Source Intel**: It was briefly mentioned that Intel is now added to the PyTorch website, indicating a potential integration or support update.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**CLI Joins the LM Studio Toolbox**: LM Studio has launched its new CLI tool, `lms`, designed to simplify the management of local LLMs, including loading and unloading models and starting or stopping servers. The CLI tool is available for the latest LM Studio 0.2.22 and beyond, and users are encouraged to contribute to its [open source GitHub repository](https://github.com/lmstudio-ai/lms).

**Llama's Conversion Complication**: Collaboration in the LM Studio guild led to the successful resolution of several integration issues with `llama.cpp`, utilizing scripts such as `convert-hf-to-gguf`. Some users faced FileNotFoundError that was fixed by redownloading necessary files via `huggingface-cli`, with the community assisting in addressing conversion execution problems.

**Model Performance and Oddities**: Discussion in the models channel revealed endeavors to enhance story writing with **Goliath 120B Longlora** models and experiments to assess recall capabilities of models like **LLAMA 3** on extensive texts. A curiosity emerged about **ChatQA 1.5** showcasing unexpected response templates, whereas a bug in the latest **LM Studio 0.2.22** prompted a new update for corrected behavior.

**ROCm's Growing Pains and Triumphs**: Members explored the capabilities of the latest LM Studio 0.2.22 ROCm Preview, with some testing the upper limits of RAM and context sizes and others addressing issues with embedding models. The introduction of `lms` CLI for AMD ROCm's preview and Linux support triggered spirited discussions about the tool's potential, bolstered by efforts in headless mode execution and dockerization.

**Server-Client Connect Unlocked**: Tips and fixes for configurations were shred, including a handy way to repopulate default configs, resolving access to LM Studio through WSL by using correct IP addresses, and enabling seamless communication between Windows and WSL environments for the app without additional complexity.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Beta Test Battalion Assembles**: The recruitment drive for **Pages** beta testers has successfully concluded, with the team voicing their appreciation and directing attention to upcoming updates on development progress.
  
- **Perplexing Browser Predicaments and Payment Puzzles**: Technical issues were flagged with **Perplexity** not functioning on Safari and Brave browsers, while a user's query about an unwanted subscription charge was directed to support@perplexity.ai for resolution. Enhancements for voice command functionality and clarity on usage limits for models like **Gemini 1.5 Pro** and **GPT-4 Turbo** were hot topics, alongside excitement for emerging AI technology advancements.

- **Share Wisely and Prosper**: Reminders were sent to ensure threads are made shareable before linking on Discord, encompassing a range of interests from **lunar queries** to **musical AI discoveries**. Concerns over printer privacy and an exploration of AI-generated content underlined the guild's diverse focus areas.

- **AI API Adventures and Accuracies**: Discussions centered on making effective use of the **Sonar Large** model through precise prompts and prompt optimization techniques. Variable results with the API highlighted the need for tweaking settings like `frequency_penalty`, `temperature`, and `top_p` to enhance response quality, with guidance pointing towards transitioning to the latest **Sonar** models for improved accuracy.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Hermes 2 Pro Hops into the Fray**: The recently released **Hermes 2 Pro** integrated with LLaMA weights is making waves with its advanced **QA**, **Function Calling**, and **JSON Mode** capabilities. It’s garnering attention for exceptional inference speeds on mobile devices and has support material on [GitHub](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) and [Hugging Face](https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B).

**ChatML Equation S-Bahn**: Tweaks to enable **ChatML** like using token replacement strategies and altering EOS symbols are being dissected by members, though details on the modifications are sparse.

**World-sim Codex**: A lively discussion around **world-sim** pointed out recent updates and shifts, such as the introduction of the Iron Age, and shared resources on **consciousness** and AI with links to [YouTube talks](https://www.youtube.com/watch?v=abWnhmZIL3w).

**Dataset Seekers Untie**: Members queried about free generic datasets suitable for **finetuning LLMs** prior to initiating mining sequences, prompting shared interest but limited response in channels marked **#bittensor-finetune-subnet** and **#rag-dataset**.

**LLama Crafting Corner**: Troubleshooting around *llamacpp* led to suggestions of using *ollama* to sidestep handling C directly and to employ techniques like **quantization** and **pruning** for ideal CPU-run LLM scenarios. The conversations also explored the intriguing concept of *moral non-commutativity* in **retrocausality** and the psychological impacts therein.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Bringing Mojo to the Command Line**: The `prism` CLI toolkit for Mojo has been augmented with new features such as *persistent flags*, *hooks*, and *flag groups*. Updates are showcased on the project's [GitHub page](https://github.com/thatstoasty/prism).

**Test Driven Mojo Development**: `mojo-pytest`, the plugin for testing in Mojo, now supports the new **version 24.3**. An issue to improve debuggability is tracked at [Issue #9 on GitHub](https://github.com/guidorice/mojo-pytest/issues/9).

**NuMojo Outpaces Rivals**: The NuMojo project, aiming to enhance Mojo's standard library tensor functionality, has been updated for Mojo version 24.3 and shown to perform better than NumPy and Numba in benchmarks. Check out NuMojo's progress on [GitHub](https://github.com/MadAlex1997/NuMojo).

**Adventures in Learning Mojo**: For those curious to integrate Mojo into workflows, a new "Let's mojo build -D your own -D version=1 app" tutorial is available. It's designed to illustrate Mojo's capabilities through a series of workflows and can be found on [GitHub](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md).

**Nightly Releases Keeping Mojo Fresh**: Mojo's development strides forward with more frequent nightly releases—eventually daily—aligning with infrastructure improvements. Nightly changelogs, like the introduction of `__source_location()` and improved docstring flexibility, can be perused at the [Modular Docs Changelog](https://docs.modular.com/mojo/changelog#language-changes).

**Maxing Out on MAX Extensibility**: MAX 24.3 introduces the brand new MAX Engine Extensibility API which aims to perfect PyTorch, ONNX, and Mojo model integrations. Detailed information on performance and hardware optimization is provided in the [MAX Graph APIs](https://docs.modular.com/engine/graph).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**AI Job Market Roulette**: The community engaged in a humorous debate about the **fleeting nature of high-paying jobs** in AI, with quips about the potential profitability of unconventional career paths like AI CEO or even a dentist.

**Speculation Station for GPT-5 Ticket Prices**: There's chatter on the potential **pricing strategy for GPT-5**, with the group divided on whether OpenAI would opt for regional pricing models or stick with a single price point for all.

**Deja Vu for GPT-3 Devotees and Chat Rooms**: Members expressed nostalgia over **GPT-3 and Codex**, despite the buzz around GPT-4, and raised questions about the absence of **voice chat rooms** for real-time discussion, citing moderation concerns.

**Response Time Riddle with GPT-4**: Talks about **GPT-4's response times** being slower than **GPT-3.5**, with mentions of **gpt4 turbo** facing significant latency, indicating that engineers are keeping a close eye on performance metrics.

**Cutting Through the Clutter in AI Research**: Discussions emphasized the distinction between publicly available research papers and the unrealistic expectation of OpenAI releasing fully trained proprietary models, due to their **computational demands** and proprietary elements.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Code Whispering with Moondream and FluentlyXL**: Community contributions showcase [Moondream 2](https://huggingface.co/spaces/Csplk/moondream2-batch-processing) for batch processing and [FluentlyXL v4](https://huggingface.co/spaces/fluently/Fluently-Playground), as well as Portuguese translations of HF's Audio course and a new [MPI Codes repository](https://github.com/Binary-Beast03/MPI-Codes) for MPI development. An [intelligence boost for LangChain](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da) and FinBERT's [financial sentiment tuning](https://huggingface.co/ProsusAI/finbert) were also discussed.

**Babel Fish's Extended Family**: The multilingual sphere expands with [BLOOM](https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat) supporting 55 languages and research on improving LLMs, exemplified by a [curated list](https://huggingface.co/collections/f0ster/smarter-llms-research-6633156999b1fa10612309dd) and the [RARR approach](https://huggingface.co/papers/2210.08726) for automatic attributions in text generation. Members are also keen on [deploying models with Ray](https://ray.io/) and assessing quality metrics for refined prompts.

**Diffusion Model Mixology**: In diffusion discussions, the community explores techniques for merging pipelines and partial diffusion methods, with a notable partial diffusion pull request for **SD 1.5** found on [GitHub](https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2). Overall, the topic of efficient and innovative model merging strategies garners attention.

**Model Fine-Tuning Finesse**: Best practices for fine-tuning models, like only adjusting classifier weights and customizing training loops, are debated, with a detailed guide on HuggingFace's [_Transformers and Keras_](https://huggingface.co/docs/transformers/training). Members also discuss visual confirmations of models like **Fluently-XL-v4** outperforming others [on Instagram](https://www.instagram.com/p/C6eMZaTr03q/?igsh=MWQ1ZGUxMzBkMA==).

**Seeking AI Mentors and Conversationalists**: The community expresses a need for parquet converter-bots and more structured ways for members to provide peer support, like a possible **#cv-study-group**, while sharing knowledge and links for upskilling, such as a [YouTube video on fine-tuning AI models](https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s) and an exploration of graph ML's impact on LLMs.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG Stack Up**: The LlamaIndex community shared resources on creating efficient **data stacks** and **RAG pipelines** with a focus on boosting query precision. [@tchutch94](https://twitter.com/tchutch94) and [@seldo](https://twitter.com/seldo) contributed to a detailed tutorial, which can be read [here](https://t.co/jez7g9hADV); while the OpenAI assistant API v2 was praised for its effectiveness but flagged for high costs per query.

- **Airbnb's Listing Leap**: Harshad Suryawanshi unveiled a guide for a RAG application capable of filtering **Airbnb listings** using natural language, leveraging **MistralAI's Mixtral 8x7b** tools. Detailed documentation and a repository guide have been provided [here](https://t.co/iw6iBzGKl6).

- **Introspective Agents Intro**: New introspective features in **LlamaIndex 10.34** were highlighted, promising self-reflective agents capable of iterative response improvements and future Huggingface integration. Concerns were raised regarding content sensitivity, advising caution with the implementation detailed [here](https://t.co/X8tJGXkcPM).

- **Pandas in Finance, MongoDB Mysteries, and More**: There's ongoing dialogue on leveraging the Pandas Query Engine for financial applications, fine-tuning MongoDB for **LlamaIndex** querying, rectifying **llamacpp** deadlocks, and employing Trulens for observability. One member signaled memory usage spikes with LlamaIndex, indicating an urgent need for memory management optimization.

- **Challenges and Code**: The guild witnessed requests for technical advice, from setting up financial analysis applications to addressing potential deadlocks in parallel requests with llamacpp. There's an active pursuit of alternative methods for specific MongoDB operations and guidance on memory issues with LlamaIndex, with additional links provided for community learning and support.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Suno Sings New Melodies**: An **AI-in-action-club** member sparked interest about **Suno's music generation** ability, anticipating whether it can compose entire music tracks independently, with focus on its audio tokenizing technique.
- **Mamba Conversations Go Deep**: In **llm-paper-club-west**, enthusiasts are delving into **Mamba's** inner workings with a Notion deep dive ([A Mamba Deep Dive](https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f)) and debating its selective recall and sensitivity to overfitting.
- **Audio Innovation at Its Finest**: Discussions in **AI-in-action-club** revolved around processing and generating audio with autoencoders and latent diffusion, citing concern for harmonic distortion and referencing a blog about the *snake activation function* which might mitigate this issue.
- **Unlocking Gemini's Potential**: A user in **ai-general-chat** sought tools compatible with **Gemini 1.5**, yet expressed preference for **Opus** or **Cursor** due to better performance with long contexts.
- **SQLite Searches in New Dimensions**: A mention in **ai-general-chat** of a new vector search extension for **SQLite**, namely `sqlite-vec`, indicates a stride in improving vector search functionalities within databases.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**LLMs Translating Before Answering**: Engineers debate **Large Language Models** (LLMs) processing multilingual inputs by potentially converting them to English first, referencing ["Understanding Language Models by Fine-grained Language Identification"](https://arxiv.org/abs/2402.10588). An important nuance for those looking to optimize multilingual LLM systems.

**Lost Research Directions Evoke Nostalgia**: A reflective exchange on understudied ML fields, such as adversarial robustness and domain-specific modeling, lamented due to the industry's overshadowing allure. Notably poignant for the career paths of researchers in the field.

**Leakage Looms Over Benchmarks**: Concerns in **benchmark dataset leakage** for LLMs stir conversation, emphasizing the challenges in gauging leaks and rectifying them. Two papers, [one](http://arxiv.org/abs/2404.18824) on leakage detection and [another](https://arxiv.org/abs/2405.00332) proposing new methods like fresh benchmark questions, fuel the discussion.

**English as a Pivot in LLMs Proves Generative**: **llama** models' findings suggest English as a pivot language is a sound strategy, potentially boosting those working on **cross-model generalizability**. Such replication adds weight to the approach for those developing multilingual LLMs.

**Language Models Dream of Chess Mastery**: A study involving a transformer trained solely on chess games achieves high performance, sans heuristics, as cited in a [DeepMind paper](https://arxiv.org/abs/2402.04494). Demonstrates the scope of scale training for AI engineers interested in out-of-box model applications.

**Grandmaster-Level Chess Without Search**: A study using a transformer model trained on a dataset of 10 million chess games was brought up, demonstrating the model's high performance in chess without domain-specific enhancements or explicit search algorithms. The [DeepMind paper](https://arxiv.org/abs/2402.04494) indicates that training models at scale can lead to competitive levels of play without the approaches traditional chess engines use.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **LLama-3 8B Stretches its Legs**: **LLama-3 8B** successfully extended its context length to over 1040k, crucially supported by [Crusoe Energy's compute](https://huggingface.co/crusoeai), incorporating an adjusted RoPE theta for advanced long-context handling in large language models.
  
- **Optimization Achieved in Axolotl Repo**: A significant improvement has been contributed via a PR that solves a bottleneck in the orpo trainer by enabling it to utilize multiple workers for data preprocessing, as detailed at [GitHub PR #1583](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583), which could enhance speed across various training configurations like DPO, SFT, and CPO.

- **Prompt Design Evolves and llama.cpp Runs Rings Around Inference**: Prompt fine-tuning insights emerged, revealing the inclusion of ChatML tokens within system prompts improves tokenization, while **llama.cpp** upgrade resulted in a 30% increase of **Hermes 2 Pro Llama 3 8B** inference speed on 8GB RAM Android devices.

- **Conversion Complexities with llama.cpp**: Troubles converting **SafeTensors to GGUF** were voiced, highlighting the limitations of llama.cpp's script, which lacks the breadth of conversion options such as `q4k`. Solutions were explored, with a [script for conversion](https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh) provided, yet the quest for expanded output types persists.

- **DeepSpeed Stage 3 Flashes Past VRAM Limitations**: ZeRO-3 optimizations do not impact model quality but require careful integration, potentially harmonizing with Flash Attention for fine-tuning pursuits. When applied correctly, these technologies can augment training speeds and enable larger batch sizes without necessitating complex parallelism—with experience shared on [Axolotl's GitHub](https://github.com/openaccess-ai-collective/axolotl) and corroborated by [DeepSpeed documentation](https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Documentation Dilemma Resolved**: Access to instructions for **Ollama**, **Jan.ai**, and **Llamafile** is improved with a direct [link to the Open Interpreter local installation guide](https://docs.openinterpreter.com/guides/running-locally), emphasizing **dolphin-mixtral** configurations to streamline the setup process.

**Performance Enhancements for Whisper RKNN**: A notable 250% performance surge is achieved for Whisper RKNN on **Rockchip RK3588 SBCs** as shared in the [rbrisita's GitHub branch](https://github.com/rbrisita/01/tree/rknn), and there's an anticipation of upcoming LLM RKNN feature integrations.

**AI Vtubing Enters Open Source Arena**: The AI Vtuber community benefits from a pair of new resources: an [AI Vtuber starter kit on GitHub](https://github.com/tegnike/nike-ChatVRM), and an [offline-ready, API-free Vtuber repository](https://github.com/neurokitti/VtuberAI.git), with a live proof-of-concept showcased on [YouTube](https://youtu.be/buaK84oSWCU?si=P02NIYHvrVj7m8Lb).

**Interactivity Extended to Mobile**: Insight into hosting **Open Interpreter** on servers for broader access and setting up mobile-friendly, local models was shared, linking to specific Android device setup and [running Open Interpreter locally](https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally).

**Sound Choices in Speaker Selection**: A discerning approach is underway to select the optimal speaker for an unnamed electronics project, promising future insights based on the integration and validation results.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter Battles Traffic Surge**: OpenRouter grappled with **higher-than-normal errors** due to a traffic spike, with scaling efforts in progress to mitigate intermittent connectivity issues.

**Money Moves**: A proposal to integrate **WeChat Pay** and **Alipay** via [Stripe](https://stripe.com) was discussed, with the community aware of it requiring additional paperwork; meanwhile, suggestions to develop an app for smoother transactions using **Google payment services** were also floated.

**Model Size Matters**: The AI community showed keen interest in next-generation language models like **LLaMA-3**, with anticipation for potential releases by entities like Soliloquy, while recognizing the limitations tied to proprietary models.

**Fine-Tuning Finesse**: Engineers debated the risk of **model dumbing post-fine-tuning without instruct datasets**, agreeing that blending old and new data might safeguard against catastrophic forgetting.

**Gemini Pro Troubleshooting**: Technical solutions were shared for problems encountered with **Gemini Pro** messages, such as starting prompts with an "assistant" role to facilitate better interactions.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**StoryDiffusion Crafted by Angry Penguin**: [StoryDiffusion](https://huggingface.co/spaces/YupengZhou/StoryDiffusion) sparks interest, engaging members with AI storytelling potential, following a link shared by angry.penguin.

**AI Town Troubles and Tools**: Disruptions from *empty messages and strings of numbers* in ai-town-discuss highlight tokenizer concerns; meanwhile, resources like [@TheoMediaAI's AI simulation exploration](https://x.com/TheoMediaAI/status/1786377663889678437) and [@cocktailpeanut's sqlite replay web app](https://x.com/cocktailpeanut/status/1786421948638965870) for AI Town catch attention.

**Node Woes in Backend Development**: Incorrect Node version causes stumbling blocks in local deployment of `convex-local-backend`; workaround involves switching to Node v18. A community-sourced [issue](https://github.com/get-convex/convex-backend/issues/1) was logged regarding a TypeError with `.ts` extension during setup.

**Raspberry Pi Channel Piqued Interest**: An expression of deep contemplation and a member's acknowledgment reveal that the ai-raspberry-pi channel meets certain members' specialized interests in AI development on small-scale hardware.

**Cocktail Peanut Receives Undefined Kudos**: A mysterious member praises **cocktail peanut** amid discussions but leaves the community guessing the work or breakthrough being referenced.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SoundStream Hits a Sour Note**: An AI engineer faced implementation issues with Google's **SoundStream**, but others recommended a concrete solution—a [GitHub repository](https://github.com/wesbz/SoundStream) that could offer guidance.

- **Sharing is Caring in the Art AI Space**: A newcomer who completed a **Stable Diffusion** Udemy course is willing to share it with peers, aiming to forge connections and further hone their skills in AI-generated art.

- **AI Community Gets Playful with Investments**: In a lighter moment, AI enthusiasts joked about their investment strategies, humorously preferring services that would either significantly multiply or halve their money.

- **Quest for Prompt Adherence in Model Training**: Discourse revealed skepticism regarding the effectiveness of using both T5 text encoder and CLIP in improving prompt adherence in model training, sparking a mix of surprise and theories about the role of CLIP dropout.

- **Back to Basics for Bigger Isn't Always Better**: Within the **StableDiffusion** space, the focus is migrating from building larger models to enhancing architecture and training methodologies on smaller models due to hardware limitations. This highlights the importance of nuanced training with CLIP to sidestep embedded biases and constraints.

- **Dataset Debate Rages On**: A heated chat about dataset choices showed a preference for real-world datasets like MNIST, CIFAR, or ImageNet over synthetic ones to better showcase interpretability in models.

- **Interpretability or Applicability?**: Skeptics in the conversation debated whether methods developed for interpretability also effectively translate into solving real-world challenges, adding a layer of practicality to the discussion.

- **A Mysterious New Arrival**: **[StoryDiffusion](https://storydiffusion.github.io/)** appeared on the scene courtesy of a guild member, albeit with no further explanation, leaving the engineers to scratch their heads about its use or importance.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Hackathon Alert: Build AI Products in 54 Hours for Cash**: The **BeeLoud** hackathon, scheduled for May 10-12, invites participants to create AI innovations within 54 hours, with a prize pool of up to $25,000. For more details, see [Build - BeeLoud](https://beeloud.xyz/build/).

**LangChain and RAG Empower Email Crafting**: **LangChain's LangGraph Agents** now leverage Retrieval-Augmented Generation (RAG) to enhance AI-assisted email drafting, promising both efficiency and quality improvements, as detailed in a [Medium article](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da).

**Java Devs, Meet LangChain**: A newly available **langchain4j** Java port of LangChain has been announced, broadening the scope for integrating AI applications across different platforms and languages. Interested engineers can explore [langchain4j](https://github.com/langchain4j/langchain4j) on GitHub.

**Dragonfly Boosts LangChain's Performance**: By integrating the **Dragonfly** in-memory data store with LangChain, developers can expect improved chatbot performance and context management which is explained with examples in their latest [blog post](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly).

**Langserve Decoded**: The **langserve feedback endpoint** clarification was provided, where an "OK" response merely indicates that feedback has been successfully submitted, but might still be rejected if the server deems it unauthenticated or invalid.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Leaked Model Mayhem**: A leaked model, possibly from **GDM** featuring *oddly specific quant*, was discussed with references to a [tweet](https://x.com/teortaxestex/status/1785974744556187731?s=46) and mysterious 4chan postings hinting at a breach.
- **Prometheus 2 Rises**: A new language model, **Prometheus 2**, introduced in a [research paper](https://arxiv.org/abs/2405.01535), claims superior evaluation abilities over GPT-4, sparking conversations about its efficacy and utility.
- **Competition Heats Up with Big Prize Pools**: **LMSYS** launched a $100K human preference prediction competition, as mentioned in a [tweet](https://x.com/lmsysorg/status/1786100697504833572?s=46), leveraging conversations from popular language models like **GPT-4** and **Mistral**.
- **The PPO-REINFORCE Connection**: An exploration suggesting that Proximal Policy Optimization (PPO) could reduce to the REINFORCE algorithm under certain conditions spurred ongoing discussion, with a resource shared from OpenAI's [Spinning Up documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
- **The Undisclosed Value of Value Functions**: Debates about why value functions aren't typically released post-RLHF training led to recognizing the potential wealth of insights they hold for reinforcement learning, despite them not being a standard share-out in the community.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**PDF Search System Unearthed**: A member proposed a search system for **large PDF documents**, discussing strategies including document summarization via LLMs, embedding generation for semantic search, and LLMs-based key information indexing.

**Llama Tokenization Mysteries Revealed**: Queries arose regarding the necessity of a *beginning-of-string (<BOS_TOKEN>)* when using the **llama-cpp-python library with Command R+**, with observations of its automatic inclusion during tokenization.

**RAG Access with Cohere Confirmed**: A user's question about the feasibility of using a **free Cohere API key for RAG** was answered, confirmation was given of its availability, albeit with rate limitations.

**C4AI Command R+ Gets Quantized**: Technical conversation unfolded around the **[C4AI Command R+ model](https://huggingface.co/CohereForAI/c4ai-command-r-plus)**, with a focus on its [quantized variant](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit), and varying system requirements for local implementation.

**Code Interpreter SDK Takes the Stage**: An announcement regarding the [launch of the Code Interpreter SDK](https://x.com/tereza_tizkova/status/1786058519701254268?s=46&t=yvqplJRJNpP5EM3LZLMQlA) surfaced, alongside a discussion about its distinction in the context of pre-existing technologies.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **llamafile Takes the Leap to Systemd**: Engineers have shared a **systemd** script for deploying **llamafile** on Rocky Linux 9, which includes detailed execution commands and the configuration of necessary arguments like server port and model path.
- **Server Mode Gets a URL Facelift**: Responding to a request for specifying a base URL in server mode, an issue was raised on [GitHub](https://github.com/Mozilla-Ocho/llamafile/issues/388) for proxy support in *llamafile*, which would facilitate serving it under a subdirectory through Nginx.
- **Ein, Zwei, Whisper!**: The community showed interest in the [distil-whisper-large-v3-german model](https://huggingface.co/primeline/distil-whisper-large-v3-german), with discussions on its application in a speech-to-text, LLM processing, and text-to-speech pipeline that could culminate in a detailed blog post.
- **Vector Space Mysteries**: A discrepancy in embedding directions between **llamafile** and **llama.cpp** was highlighted, where a low cosine similarity points to an issue described on [GitHub](https://github.com/Mozilla-Ocho/llamafile/issues/391), and was tested with available Python scripts.
- **Chatty Files and Code**: To facilitate conversational interaction with documents and code using llamafile, members recommended utilizing `curl` API calls, with reference to example scripts found in the [llama.cpp chat script repository](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.sh#L64).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Makes Strides and Welcomes New Contributors**: Tinygrad has reportedly made significant **progress** recently, and a member celebrated their **first commit** to the project, marking a personal milestone.
  
- **Blobfile's Role in Llama.py Explained**: Users clarified that `blobfile` is crucial for `load_tiktoken_bpe` function in `examples/llama.py`, enhancing understanding among peers.

- **Troubleshooting Tinygrad's Forward Pass**: One engineer faced challenges with the forward pass compute graph, which was addressed by prompting execution with `out.item()` or `out.realize()` and installing missing libraries to fix a `NameError`.

- **Resolving Graph Visualization Issues in Tinygrad**: Installation errors with `networkx` and `pydot` were resolved by installing `pydot` and `graphviz`, respectively, after which a member recommended the documentation be updated to help others avoid the `sh: dot: command not found` error.

- **Community Collaboration Drives Documentation Improvement**: The resolution of the 'dot command' issue via `graphviz` installation highlights the collaborative spirit of the community, prompting a practical suggestion to update the project's documentation to aid future users.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**Jamba-Instruct Is Live**: AI21 Labs has launched **Jamba-Instruct**, a sophisticated instruction-tuned hybrid SSM-Transformer model, designed to enhance commercial application performance. The company highlights the model's capabilities in a recent [Twitter announcement](https://twitter.com/AI21Labs/status/1786038528901542312) and a detailed [blog post](https://www.ai21.com/blog/announcing-jamba-instruct).

**AI21 Labs Welcomes Feedback for Jamba-Instruct**: AI21 Labs is inviting industry feedback for **Jamba-Instruct** and indicates their openness to discuss custom requirements, including context windows exceeding the initial 256K limit.

**Reading Up on Jamba-Instruct**: Engineers interested in the **Jamba-Instruct** model can gain a deeper understanding by reading the [official blog post](https://www.ai21.com/blog/announcing-jamba-instruct), which talks about its deployment for reliable commercial use and quality benchmarks.

**Higher Context Windows on the Horizon**: An AI21 Labs staff member has expressed their interest in exploring significantly larger context windows for **Jamba-Instruct** and has invited users to collaborate on this potential expansion to meet specific use scenarios.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Quick Alert: Fast Compute Grants**: AI enthusiasts and engineers take note, a tweet by @PrimeIntellect announces availability of **fast compute grants** for those in need. Check out the details in their [Fast Compute Grants Tweet](https://twitter.com/PrimeIntellect/status/1786386588726960167).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Quantization Woes for LLaMA 3**: A conversation on the guild revolved around the impact of **quantization** on **LLaMA models**, with a Discord member citing a [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/) and [research paper](https://arxiv.org/abs/2404.14047) that discuss the performance hit when applying low-bit quantization to **LLaMA 3**.
- **Chinchilla Law Ignored, Performance Suffers**: The guild also explored how the significant quantization of **Meta's LLaMA** may lead to substantial information loss due to the neglect of the *chinchilla scaling law* and the model’s training on 15T tokens. This suggests larger models might experience even more pronounced degradation with increased precision reduction.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Skunkworks AI Projects Hook Up With Fast Compute Grants**: Ambitious **Skunkworks projects** can potentially receive fast compute grants, as divulged in a **[Twitter announcement](https://twitter.com/PrimeIntellect/status/1786386588726960167)** by a guild member. Interested engineers should explore this opportunity for support on cutting-edge initiatives.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **AI to Tidy Up Local Model Piles**: An individual highlighted the need for an **LLM (large language model)** to address the issue of managing and cleaning **7B local models** scattered across various directories due to the multitude of apps and libraries. Frustration was aired over the organization or lack thereof, suggesting a potential area for tool or algorithm development.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1235510880003821589)** (734 messages🔥🔥🔥): 

- **Community Explores Full Finetuning with Unsloth**: Members initiated a detailed discussion on whether full parameter finetuning is feasible with Unsloth. Despite initial claims that only LoRA (a parameter-efficient training method) is supported, some discovered that setting all parameters except layernorms to trainable seemed to enable a form of near-full finetuning.
- **Optimization for GGUF Files**: The Unsloth team announced they are working on fixing issues with llama.cpp and GGUF (Generalized GPU Format) conversions, responding to community members' difficulties with quantization and loading checkpoint shards.
- **Sentiment Analysis Model Guidance Sought**: A member seeking help in creating a sentiment analysis model based on a large database of country-scale reviews received guidance on converting various document types to a proper format for use with LLMs.
- **Assistance Offered for Dataset Formatting and ORPO**: Members discussed ways to structure datasets for preference optimization using Unsloth, including strategies for multiple "rejected" responses. The community provided insights and possible solutions to help navigate the process.
- **Unofficial Full Finetuning Tactics Shared**: While official support for full finetuning isn’t provided in Unsloth, community members experimented with enabling it by adjusting model parameters manually. A positive note was that losses seemed to improve and memory benefits were still evident compared to Hugging Face implementations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://x.com/dudeman6790/status/1784414430781931961">Tweet from RomboDawg (@dudeman6790)</a>: Here is a full colab notebook if you dont want to copy the code by hand. Again thanks to @Teknium1 for the suggestion https://colab.research.google.com/drive/1bX4BsjLcdNJnoAf7lGXmWOgaY8yekg8p?usp=shar...</li><li><a href="https://huggingface.co/papers/2402.05119">Paper page - A Closer Look at the Limitations of Instruction Tuning</a>: no description found</li><li><a href="https://huggingface.co/maywell/Llama-3-70B-Instruct-32k">maywell/Llama-3-70B-Instruct-32k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nvidia/Llama3-ChatQA-1.5-70B">nvidia/Llama3-ChatQA-1.5-70B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B">nvidia/Llama3-ChatQA-1.5-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/grahama1970/77a2b076d18ff2a62479b3170db281c5">Lllama 70B Instruct QA Prompt</a>: Lllama 70B Instruct QA Prompt. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://datta0.substack.com/p/ai-unplugged-9-infini-attention-orpo">AI Unplugged 9: Infini-Attention, ORPO, </a>: Insights over Information</li><li><a href="https://github.com/IBM/unitxt">GitHub - IBM/unitxt: 🦄 Unitxt: a python library for getting data fired up and set for training and evaluation</a>: 🦄 Unitxt: a python library for getting data fired up and set for training and evaluation - IBM/unitxt</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://www.youtube.com/watch?v=T1ps611iG1A">How I Fine-Tuned Llama 3 for My Newsletters: A Complete Guide</a>: In today&#39;s video, I&#39;m sharing how I&#39;ve utilized my newsletters to fine-tune the Llama 3 model for better drafting future content using an innovative open-sou...</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://www.youtube.com/watch?v=WxQbWTRNTxY&t=83s">How to Fine Tune Llama 3 for Better Instruction Following?</a>: 🚀 In today&#39;s video, I&#39;m thrilled to guide you through the intricate process of fine-tuning the LLaMA 3 model for optimal instruction following! From setting...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6965">llama3 custom regex split by jaime-m-p · Pull Request #6965 · ggerganov/llama.cpp</a>: Implementation of unicode_regex_split_custom_llama3().
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1235509542071177246)** (20 messages🔥): 

- **Phi 3 in Your Browser**: A tweet shows someone running **Phi 3** inside a web browser, specifically highlighted with "lmao". The tweet can be found [here](https://twitter.com/fleetwood___/status/1783195985893863578).
- **LLAMA 3 Discord Channel Nonexistent**: A question was raised about the existence of a **LLAMA 3 Discord** channel, to which a member replied that such a channel does not exist.
- **Crafting New Roles in LLAMA 3**: A question regarding the possibility of adding new roles to **LLAMA 3** was raised, linking to a GitHub repository. The response suggested a simple replacement using `type=code` instead of `tool_call`.
- **Self-Discovery Paper Techniques Applied**: A user found it useful to force **ChatGPT** to memorize 39 reasoning modules from the Self-Discovery paper, recommending its application to complex reasoning tasks. The paper is accessible [here](https://arxiv.org/pdf/2402.03620#page=12&zoom=100,73,89).
- **Triton's Acceleration of LLAMA 3**: A blog post from PyTorch showcases **TK-GEMM**, a tool using Triton FP8 GEMM that optimizes **LLAMA 3** on NVIDIA H100 GPUs. The blog, including performance comparisons and technical details, can be viewed [here](https://pytorch.org/blog/accelerating-llama3/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/accelerating-llama3/?utm_content=291787920&utm_medium=social&utm_source=linkedin&hss_channel=lcp-78618366">Accelerating Llama3 FP8 Inference with Triton Kernels</a>: 1.0 Summary  </li><li><a href="https://pytorch.org/blog/accelerating-llama3/?utm_content=291787920&utm_medium=social&utm_source=lin">Accelerating Llama3 FP8 Inference with Triton Kernels</a>: 1.0 Summary  
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1235486709529968652)** (580 messages🔥🔥🔥): 

- **GGUF Conversion Issues with Llama 3 Identified**: A member highlighted a critical issue where Llama 3 loses fine-tuning data during conversion to GGUF format. The problem seems inherent in GGUF regardless of precision, as tested in [FP16 and Q8](https://github.com/ggerganov/llama.cpp/issues/7062); discussions with Unsloth and suggestions from the community have yet to resolve it.

- **Lora Adapter Merging Problems**: Attempting to merge Lora adapters with GGUF models resulted in the fine-tuning being partially lost. Despite suggestions to use separate Lora adapters with GGUF models, the outcomes did not meet expectations, [worsening with combined GGUF and Lora](https://github.com/ggerganov/llama.cpp/issues/7062).

- **Inference and Finetuning Solutions for Llama 3 Shared**: Users shared their finetuning strategies using the original INSTRUCT model with Llama 3 and appending eos_token after instructions. It was noted that posting to `/completion` one needs to pass all chat tokens, which some users may have missed, while initiating servers with Llama 3 requires setting `--override-kv` for the tokenizer.

- **Possible Issues with Llama.cpp for Llama 3**: Members are suspecting there may be issues with compatibility between llama.cpp and the newly released Llama 3, given the similarity in problems outlined on the llama.cpp's issues section.

- **Seeking Help and Following Roadmaps**: New users are seeking step-by-step help for finetuning models like Gemma and Llama. More experienced community members pointed to Unsloth's notebooks for [Llama](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp) and [Gemma](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo), and suggested searching for AI/ML courses and tutorials on platforms like YouTube.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1Wg">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharin">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/long-context">Unsloth - 4x longer context windows &amp; 1.7x larger batch sizes</a>: Unsloth now supports finetuning of LLMs with very long context windows, up to 228K (Hugging Face + Flash Attention 2 does 58K so 4x longer) on H100 and 56K (HF + FA2 does 14K) on RTX 4090.  We managed...</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Meta Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followe...</li><li><a href="https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraConfig">LoRA</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#loading-lora-adapters-for-continued-finetuning">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://www.youtube.com/watch?v=j7ahltwlFH0&t=413s&pp=ygUJbGxhbWEuY3Bw">Convert LLMs to run on laptop w/ LLAMAcpp - GGUF quantization</a>: Would you like to run LLMs on your laptop and tiny devices like mobile phones and watches? If so, you will need to quantize LLMs. LLAMA.cpp is an open-source...</li><li><a href="https://github.com/xaedes/llama.cpp/tree/finetune-lora/examples/export-lora">llama.cpp/examples/export-lora at finetune-lora · xaedes/llama.cpp</a>: Port of Facebook&#39;s LLaMA model in C/C++. Contribute to xaedes/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/5360">creating gguf model from lora adapter · ggerganov/llama.cpp · Discussion #5360</a>: I have a ggml adapter model created by convert-lora-to-ggml.py (ggml-adapter-model.bin). Now my doubt is how to create the complete gguf model out of these? I have seen using ./main -m models/llama...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/unslothai/unsloth?tab=readme-ov-file#pip-installation">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7021">Cannot convert llama3 8b model to gguf · Issue #7021 · ggerganov/llama.cpp</a>: Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6965">llama3 custom regex split by jaime-m-p · Pull Request #6965 · ggerganov/llama.cpp</a>: Implementation of unicode_regex_split_custom_llama3().
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1235491714119110716)** (162 messages🔥🔥): 

- **Channel Collaboration Conundrum**: A member inquired about creating a channel for collaboration and coding together, specifically for users interested in finding partners to work with overnight or during the weekend. The idea was compared to EleutherAI's community project channel, and there were suggestions to re-purpose or replace existing channels (like shelving <#1180145007261401178> in favor of a new community projects channel) to foster collaboration.

- **The Hurdles of Specialization**: A conversation emerged about the feasibility of specializing a 7B model for complex tasks, such as cryptographic proof generation. Multiple users weighed in, agreeing that such a task may be overly ambitious for a small LLM (7B). It was suggested that although a smaller model can outperform larger ones in highly specialized use cases, they're typically not on par with larger models like GPT-4 or Claude.

- **Data and Compute Considerations**: The discussions also touched upon the importance of data size and quality in LLM training, with a member seeking advice on how to utilize their resources effectively, including 32 H100 GPUs. It was highlighted that model size and data preparation are crucial factors in achieving high performance, and the keys to success are case-dependent.

- **Showcasing and Learning through Community Experience**: Drsharma24 expressed a desire to learn from the community's experiences and build a space for discussing successes and strategies around fine-tuning and model training, similar to Hugging Face's platform. The conversation underscored that the Unsloth AI community could benefit from such knowledge sharing.

- **Financial Viability vs. Pure Experimentation**: The chat touched upon the distinction between developing a business use case versus experimenting and learning from model training. A member suggested that business use cases require training data that adequately reflect production environments, while others emphasized the importance of keeping the end goal in mind.

**Link mentioned**: <a href="https://tenor.com/view/dog-awkward-awkward-dog-staring-dog-patchibana-gif-13086408744970718509">Dog Awkward GIF - Dog Awkward Awkward dog - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1235486692543168543)** (753 messages🔥🔥🔥): 

- **FAQs Disappear from Discord Commands**: Users noticed the absence of the `/faq` command, pondering its removal. It turns out the command was indeed gone, leading to members jesting and realizing its absence only after interacting with a bot.

- **Debating GPU Choices for AI**: Participants discussed various GPU options like Nvidia's 4080 and 3090, AMD's 7900xtx, considering VRAM size and futureproofing. The release of Nvidia's 5000 series GPUs was hotly anticipated, prompting users to suggest waiting for the new series instead of investing in soon-to-be-outdated graphics cards.

- **Video to Anime Conversion Inquiry**: One member inquired about the time taken by an RTX 4080 to convert a video into anime-style footage, asking for benchmarks regarding video conversions using AI.

- **Opinions Clash on AMD vs. Nvidia for AI**: The conversation heated up around whether to choose AMD or Nvidia GPUs for AI tasks. While some argued for the superiority of Nvidia, especially with new technologies like the Blackwell architecture, one user defended AMD based on personal success with the brand.

- **Seeking Solutions for Text and Image Upscaling**: Users discussed best paths to add text to images using AI and queried about optimal methods for upscaling images. While tools like Davinci Resolve and Kittl were suggested for text, discussions on image upscaling tools were interspersed with mentions of ComfyUI, a versatile platform for AI image manipulation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/">ComfyUI Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://huggingface.co/gemasai/4x_NMKD-Siax_200k/tree/main">gemasai/4x_NMKD-Siax_200k at main</a>: no description found</li><li><a href="https://huggingface.co/uwg/upscaler/blob/main/ESRGAN/4x_NMKD-Siax_200k.pth">ESRGAN/4x_NMKD-Siax_200k.pth · uwg/upscaler at main</a>: no description found</li><li><a href="https://bitwarden.com/help/authenticator-keys/">no title found</a>: no description found</li><li><a href="https://www.reddit.com/r/Bitwarden/comments/1chob6h/bitwarden_just_launched_a_ne">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/176555/harrlogos-xl-finally-custom-text-generation-in-sd">Harrlogos XL - Finally, custom text generation in SD! - Harrlogos_v2.0 | Stable Diffusion LoRA | Civitai</a>: 🚀HarrlogosXL - Bringing Custom Text Generation to SDXL!🚀 Teaching Stable Diffsuion to spell, one LoRA at a time! Harrlogos is an SDXL LoRA trained ...</li><li><a href="https://www.reddit.com/r/Bitwarden/comments/1chob6h/bitwarden_just_launched_a_new_authenticator_app/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ComfyWorkflows/comfyui-launcher">GitHub - ComfyWorkflows/ComfyUI-Launcher: Run any ComfyUI workflow w/ ZERO setup.</a>: Run any ComfyUI workflow w/ ZERO setup. Contribute to ComfyWorkflows/ComfyUI-Launcher development by creating an account on GitHub.</li><li><a href="https://github.com/crystian/ComfyUI-Crystools">GitHub - crystian/ComfyUI-Crystools: A powerful set of tools for ComfyUI</a>: A powerful set of tools for ComfyUI. Contribute to crystian/ComfyUI-Crystools development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgr74j/comment/l2bxv66/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ciyzn5/comment/l2dhd6q/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1235509440904433706)** (3 messages): 

- **Tackling Gradient Details**: A member pointed out that setting `create_graph=True` might be necessary for obtaining certain gradient details in computations.
- **Clarifying Hessian Confusion**: The same member later clarified their thinking, it's not about the diagonal but rather about calculating the **Hessian-vector product with respect to the weights** twice.
- **Estimating Hessian Diagonal via Randomness**: Another member mentioned seeing a **trick in a paper** that could estimate the Hessian's diagonal using randomness combined with the **Hessian-vector product**.
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1235636317891727442)** (2 messages): 

- **Triton Newcomer's Gather Procedure Stumbles**: A new member faced an `IncompatibleTypeErrorImpl` when implementing a simple gather procedure in Triton, attempting to copy values from one tensor into another using pointer arithmetic. They later realized the issue involved using the wrong tensor type and noted a potential solution with the newly introduced `tl.cast` function ([PR #3813 on Triton](https://github.com/openai/triton/pull/3813)).
- **Kernel Debugging Challenges in PyCharm**: The same member struggled with setting breakpoints inside a Triton kernel using PyCharm, despite having set `TRITON_INTERPRET` to `"1"` as suggested in the repository documentation, and didn't succeed with the `breakpoint()` function either.

**Link mentioned**: <a href="https://github.com/openai/triton/pull/3813">[Frontend] Add tl.cast function. by jlebar · Pull Request #3813 · openai/triton</a>: This resolves an inconsistency in Triton, that every other function on Tensors has an associated free function -- i.e. you can do x.foo and tl.foo(x).

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1235567875658616923)** (6 messages): 

- **tinygrad Gets NVIDIA Open Driver Patch**: A member shared a [tinygrad patch for multi-GPU support with NVIDIA's open driver](https://morgangiraud.medium.com/multi-gpu-tinygrad-patch-4904a75f8e16), providing documentation that might be useful to others experiencing similar installation issues.
- **Kernel Module Consideration for Long Term Support**: The long-term support for peer-to-peer memory fix on NVIDIA cards was questioned, leading to a discussion about whether creating a kernel module would be a viable solution.
- **Query on Custom CUDA Extension Installation**: A member sought advice on the correct way to install a custom PyTorch/CUDA extension within a setup.py file, highlighting issues with the existing method which can be found in their [GitHub repository](https://github.com/mobiusml/hqq/blob/master/setup.py#L11-L15).
- **Sharing Solutions for CUDA Extension Setups in PyTorch**: Another member offered help by linking to pull requests that illustrate how custom CUDA extensions are managed within the PyTorch AO library. They provided links to specifics on the [setup process](https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78) and related PRs ([PR#135](https://github.com/pytorch/ao/pull/135), [PR#186](https://github.com/pytorch/ao/pull/186), [PR#176](https://github.com/pytorch/ao/pull/176)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/setup.py#L11-L15">hqq/setup.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78">ao/setup.py at 0ba0006eb704dea33becec82b3f34512fe8a6dff · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>: This is the mergaeble version of #130 - some updates I have to make   Add a skip test unless pytorch 2.4+ is used and Add a skip test if cuda is not available  Add ninja to dev dependencies  Locall...</li><li><a href="https://github.com/pytorch/ao/pull/186">louder warning + docs for custom cuda extensions by msaroufim · Pull Request #186 · pytorch/ao</a>: no description found</li><li><a href="https://github.com/pytorch/ao/pull/176">Add A10G support in CI by msaroufim · Pull Request #176 · pytorch/ao</a>: Support A10G + manylinux so cuda extensions work on as many systems as possible
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1235563374348341288)** (43 messages🔥): 

- **PyTorch PR Pains**: A contributor, **kashimoo**, expresses frustration with the slow build times of linear algebra components in PyTorch and a separate PR that was reverted due to issues with Meta's internal builds. **chhillee** confirms that such setbacks are common due to PyTorch’s "github first" policy and offers to connect **kashimoo** with more knowledgeable contributors on the Slack channel.

- **Debugging Symbols for PyTorch Development**: **kashimoo** inquires about building specific directories with debugging symbols to facilitate the use of gdb. While **chhillee** suggests using an available script on the [PyTorch development forum](https://dev-discuss.pytorch.org/t/how-to-get-a-fast-debug-build/1597), **kashimoo** thinks it might not be enough for their purposes.

- **Dynamic Compilation Challenges in PyTorch**: **benjamin_w** reports issues when using `dynamic=True` with `torch.compile(...)` in conjunction with Distributed Data Parallel (DDP) in PyTorch 2.3. While the approach worked in PyTorch 2.2.2, it appears to lead to recompilation for each batch in version 2.3. **marksaroufim** advises against using `dynamic=True` and suggests manually marking sequence lengths as dynamic instead.

- **Improving Issue Triage on CUDA MODE Discord**: **marksaroufim** and others discuss ways to handle the growing number of issues on the server, proposing the idea of a bot that parses and automatically files issues on GitHub, with **jamesmel** offering to implement the bot. It's decided to open issues in cuda mode for now to manage the influx.

- **Torch Compile Optimization for Variable Lengths**: Troubleshooting continues as **benjamin_w** struggles with `ConstraintViolationError` when using `torch._dynamo.mark_dynamic(inputs, index=1)` on PyTorch 2.2 & 2.3 for dynamic sequence lengths. They prefer persistent model compilation over multiple batches, but encounter brittle behavior. **marksaroufim** suggests that creating a GitHub issue would be best for resolving the issue.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev-discuss.pytorch.org/t/how-to-get-a-fast-debug-build/1597">How to get a fast debug build</a>: Following Allow to specify specific files for debug info by albanD · Pull Request #111748 · pytorch/pytorch · GitHub being merged, there is a new compilation flag that can be used to specify debug inf...</li><li><a href="https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile>)">torch.compile &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://www.internalfb.com/diff/D56934078">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1235553828439330877)** (5 messages): 

- **Kudos on Effort Project**: A member praised the [Effort project on GitHub](https://github.com/kolinko/effort), finding it to be quite astonishing.
- **Matrix Multiplication Confusion**: A mistake was highlighted in a matrix multiplication example, pointing out that the inner dimensions of a **3 x 1** and **3 x 3** matrix do not align for the operation.
- **Quick Correction Promised**: The author acknowledged the mix-up regarding vector orientations and expressed intent to correct it, noting a similar mistake had previously been flagged.
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1235604938160214086)** (4 messages): 

- **Averting Catastrophic Forgetting**: A member found [Ziming Liu's tweet](https://twitter.com/ZimingLiu11/status/1785483967719981538) interesting for demonstrating how to avoid catastrophic forgetting in a toy test-case.
- **In Search of Speed**: It was noted that the solution to catastrophic forgetting is "currently very slow," leading to curiosity about potential methods to increase its speed.
  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1235656551163760660)** (2 messages): 

- **FP6 Support Candidate for Custom CUDA Extension**: A new candidate for a custom CUDA extension has been identified – the **FP6 support**, following a [GitHub issue discussion](https://github.com/pytorch/ao/issues/208) on PyTorch's AO repository. An offer to help anyone interested in contributing to this extension was extended.

- **Community Member Shows Interest in FP6**: Despite lacking experience, one community member has expressed enthusiasm to contribute to the new FP6 support project and is currently endeavoring to understand the relevant research paper to determine where they could realistically contribute.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pyto">pyto - Overview</a>: pyto has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/pytorch/ao/issues/208">FP6 dtype! · Issue #208 · pytorch/ao</a>: 🚀 The feature, motivation and pitch https://arxiv.org/abs/2401.14112 I think you guys are really going to like this. The deepspeed developers introduce FP6 datatype on cards without fp8 support, wh.....
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1235657156536041593)** (9 messages🔥): 

- **Seeking Karpathy's Video Setup Advice**: A member asked for recommendations to achieve a video setup akin to Andrej Karpathy's, with live screenshare and a small camera view. They were linked to [a YouTube video by Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE) as a reference point.

- **OBS Streamlabs: A Go-to for Video Production**: In response to an inquiry about simple video set-up, OBS Streamlabs was suggested. The community member mentioned there are plenty of tutorials available for this versatile tool.

- **Enhancing Video Quality with iPhone & Mount**: For better video calls or recordings, it was recommended to use an iPhone with a Mac for superior camera and mic quality over typical laptop equipment, citing a [KDD Webcam Stand](https://a.co/d/7uxdnek) as a useful accessory.

- **Anime Appreciation Break**: A member expressed their curiosity about anime preferences, leading to a brief exchange where favorites like *Naruto*, *One Punch Man*, *Berserk*, and *Jujutsu Kaisen (JJK)* were cited for their high-quality animations and captivating fight scenes.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=zduSFxRajkE">Let&#39;s build the GPT Tokenizer</a>: The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizer...

  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/)** (1 messages): 

srush1301: Hmm, yeah this description is wrong. I will update with a clearer version
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1236014784009474089)** (4 messages): 

- **GreenBitAI Introduces a New Toolkit**: A member shared a link to [GreenBitAI’s toolkit](https://github.com/GreenBitAI/green-bit-llm) for fine-tuning, inferencing, and evaluating Large Language Models (LLMs), describing it as more of an **ML framework** augmenting PyTorch compared to bitblas, which is focused on matrix multiplication operations.
- **BitBlas Offers a Promising Kernel for Inference**: A toolkit named **BitBlas** was mentioned to have a fast **gemv kernel for 2-bit operations**, which could be beneficial for inference, although it has *not been tried* yet by the member.
- **Binary Matmul in GreenBitAI's Engine**: The discussion continues with mention of [GreenBitAI's cutlass kernels](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp), especially one that performs binary matrix multiplication, which is a part of their toolkit enhancing PyTorch.
- **Innovative Gradients Calculation Noted in GreenBitAI Toolkit**: A member highlighted that GreenBitAI's toolkit includes code that calculates the gradients of weights during training, as seen in their [q4_layer.py](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81C9-L81C20) file, and they expressed curiosity regarding the potential VRAM usage since the gradients are *not packed*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81C9-L81C20">bitorch-engine/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py at main · GreenBitAI/bitorch-engine</a>: A toolkit enhances PyTorch with specialized functions for low-bit quantized neural networks. - GreenBitAI/bitorch-engine</li><li><a href="https://github.com/GreenBitAI/green-bit-llm">GitHub - GreenBitAI/green-bit-llm: A toolkit for fine-tuning, inferencing, and evaluating GreenBitAI&#39;s LLMs.</a>: A toolkit for fine-tuning, inferencing, and evaluating GreenBitAI&#39;s LLMs. - GreenBitAI/green-bit-llm</li><li><a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp">bitorch-engine/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp at main · GreenBitAI/bitorch-engine</a>: A toolkit enhances PyTorch with specialized functions for low-bit quantized neural networks. - GreenBitAI/bitorch-engine
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1235623975653085214)** (644 messages🔥🔥🔥): 

- **CUDA and Memory Optimization Discussions**: The team achieved **167K tokens/second**, outperforming PyTorch's 150K tok/s, by optimizing CUDA kernels and introducing changes like CUDA streams and fused classifiers. They're discussing the impact of **bias kernel optimizations** and potential **next steps** for further gains. See the related [discussion and pull request](https://github.com/karpathy/llm.c/discussions/344).
  
- **Scratch Buffers and Atom Variables**: They've introduced scratch buffers to handle atom variables more efficiently. The usage of fp32 atomics on a scratch buffer and then a read and round/write to bf16 is suggested to avoid slow fp32 atomics in global memory.

- **Profiling Script Updates**: Updates have been made to the profiling script, improving robustness against CUDA library updates and separating NVIDIA kernel times from llm.c kernel times. The script changes are tracked in [this pull request](https://github.com/karpathy/llm.c/pull/342).

- **PyTorch Padding**: There is a debate on whether padding PyTorch's vocabulary size is fair for performance comparison, with acknowledgment that it's not straightforward and involves ensuring that padded dimensions are not used in the loss or during sampling.

- **Layernorm and Residual Calculations**: The conversation touched upon **saving variance and mean of layernorm in fp32** for stability and performance benefits, although it hasn't been implemented in llm.c due to code simplicity and the bf16 type used for activations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms">torch.use_deterministic_algorithms &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code">The Power of 10: Rules for Developing Safety-Critical Code - Wikipedia</a>: no description found</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html">Group Calls &mdash; NCCL 2.21.5 documentation</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/discussions/331">LLM.c Speed of Light &amp; Beyond (A100 Performance Analysis) · karpathy/llm.c · Discussion #331</a>: After my cuDNN Flash Attention implementation was integrated yesterday, I spent some time profiling and trying to figure out how much more we can improve performance short/medium-term, while also t...</li><li><a href="https://docs.google.com/document/d/1DHFaKHLTVM_zEt2AKJh5fgUNN3oY1bBs1YmxJCcrp9c/edit">3 Strategies for FlashAttention Backwards</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/discussions/344">State of the Union [May 3, 2024] · karpathy/llm.c · Discussion #344</a>: [May 3, 2024] It is day 24 of the llm.c project. We can now do multi-GPU training, in bfloat16, with flash attention, and it is FAST! 🚀 Single GPU training. We are now training GPT-2 (124M) faster .....</li><li><a href="https://github.com/karpathy/llm.c/pull/335">v1 of the new matmul backward bias kernel by karpathy · Pull Request #335 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/commit/6ebef46f832b4e55b46237c4d06c2597050819ae">ugh didn&#39;t notice this tiny rebasing mistake, introduced a bug. good … · karpathy/llm.c@6ebef46</a>: …candidate for a CI that we can overfit a single batch in the train_gpt2.cu script and get the exact same numbers as we expect in the test_gpt2.cu file</li><li><a href="https://github.com/karpathy/llm.c/pull/333">Added FlameGraphs for nsys reports and some nsys documentation by PeterZhizhin · Pull Request #333 · karpathy/llm.c</a>: Here is a sample FlameGraph. Captured on my machine.</li><li><a href="https://github.com/karpathy/llm.c/pull/341">GPU auto-detect capability for kernel builds by rosslwheeler · Pull Request #341 · karpathy/llm.c</a>: Fixes to CI -should work in both environments This is a proposal in case there is interest for kernel builds. Usage: Auto detect GPU capability: make (e.g. if your GPU capability type is 80 then --...</li><li><a href="https://github.com/karpathy/llm.c/blob/2c7960040d1d86b6c03a72ef8b32df084e899570/dev/cuda/layernorm_backward.cu#L570">llm.c/dev/cuda/layernorm_backward.cu at 2c7960040d1d86b6c03a72ef8b32df084e899570 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/299/files#diff-bf6b442">Update residual_forward to use packed input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: Update residual_forward to use 128 bit packed input, with floatX Previous Kernel: block_size   32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size   64 | time 0.0760 ms | bandwidth 993.32 GB/s b...</li><li><a href="https://github.com/karpathy/llm.c/pull/338">GELU Fusion with cuBLASLt (SLOWER because it only merges in FP16 mode, not BF16/FP32...) by ademeure · Pull Request #338 · karpathy/llm.c</a>: It turns out that not only is cuBLASLt not able to fuse BF16 GELU (or RELU) into a BF16 matmul, it also ends up with a strange kernel that is slower than our own GELU kernel as it does 2 writes per...</li><li><a href="https://github.com/karpathy/llm.c/pull/342">fixed activation gradient resetting for backward pass by ngc92 · Pull Request #342 · karpathy/llm.c</a>: also, we don&#39;t need to touch the other buffers in  zero_grad, these are anyway overwritten multiple times during backward</li><li><a href="https://openhub.net/p/tensorflow">The TensorFlow Open Source Project on Open Hub</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/343">Performance: matmul_bias, cuda streams, fused_classifier (+remove cooperative groups) by ademeure · Pull Request #343 · karpathy/llm.c</a>: I might need to split this into multiple PRs, let me know what you think (and I still need to add the new kernels to /dev/cuda/). Major changes:  New super optimised matmul_backward_bias_kernel6 CU...</li><li><a href="https://github.com/karpathy/llm.c/commit/79505bc6b3428ad5c2f609046affa1ac34e2f1af#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R2764">resolve merge and small fixes · karpathy/llm.c@79505bc</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/299/files#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R943">Update residual_forward to use packed input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: Update residual_forward to use 128 bit packed input, with floatX Previous Kernel: block_size   32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size   64 | time 0.0760 ms | bandwidth 993.32 GB/s b...</li><li><a href="https://github.com/karpathy/llm.c/commit/795f8b690cc9b3d2255a19941713b34eeff98d7b">fixes to keep master copy in fp32 of weights optionally · karpathy/llm.c@795f8b6</a>: no description found</li><li><a href="https://github.com/NVIDIA/nccl/issues/338">computation overlapped with nccl get much slower · Issue #338 · NVIDIA/nccl</a>: I used the environment from https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5 to train resnet-50 with multiple GPUs (with horovod using nccl), and found the d...</li><li><a href="https://github.com/NVIDIA/nccl/issues/338#issuecomment-1165277390">computation overlapped with nccl get much slower · Issue #338 · NVIDIA/nccl</a>: I used the environment from https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5 to train resnet-50 with multiple GPUs (with horovod using nccl), and found the d...</li><li><a href="https://github.com/karpathy/llm.c/pull/303">Updated adamw to use packed data types by ChrisDryden · Pull Request #303 · karpathy/llm.c</a>: Before Runtime total average iteration time: 38.547570 ms After Runtime: total average iteration time: 37.901735 ms Kernel development file specs: Barely noticeable with the current test suite: Bef...</li><li><a href="https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/">CUDA Pro Tip: The Fast Way to Query Device Properties | NVIDIA Technical Blog</a>: CUDA applications often need to know the maximum available shared memory per block or to query the number of multiprocessors in the active GPU. One way to do this is by calling . Unfortunately&#8230;
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 messages): 

neurondeep: also added intel on pytorch webpage
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1235486151003996214)** (350 messages🔥🔥): 

- **Llama.cpp Integration Issues and Solutions**: Members discuss issues integrating **llama.cpp** with **LM Studio**. Conversations involve the need for certain file versions and the use of the `convert-hf-to-gguf` script, with one member facing a FileNotFoundError due to missing `config.json` and resolving it by redownloading files through `huggingface-cli`. Subsequent issues with conversion and usage are tackled collaboratively.

- **Rolling Back to Previous LM Studio Versions**: Users experience bugs in the **LM Studio** version 0.2.22 where the Chat provides the entire context, not just the response. After several attempts to resolve and rollback to version 0.2.21, the issue is eventually fixed in the latest update, confirmed by multiple users.

- **Launch of LM Studio In Terminal (`lms`) Tool**: Discussion on the new `lms` tool accompanies its release alongside **LM Studio 0.2.22**, explaining its utility in automating tasks, starting an API server, and managing a model without UI interaction. Subsequent conversation clarifies that `lms` is a controller for the app, not a standalone tool.

- **Running LM Studio Headless Mode**: Several users discuss and attempt various methods to run **LM Studio** in a headless mode, using commands like `xvfb-run` to bypass GUI requirements. The conversation concludes that official headless support is not yet available, despite community workarounds.

- **Embedding LMS in Scalable Server Solutions**: Members express positivity towards the potential of embedding **LM Studio** in a high-availability server pattern across clusters, inquiring about configurations using specific presets via CLI or the UI, suggesting future feature enhancements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://releases.lmstudio.ai/windows/0.2.22/c/latest/LM-Studio-0.2.22-Setup.exe">no title found</a>: no description found</li><li><a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://tenor.com/view/pout-christian-bale-american-psycho-kissy-face-nod-gif-4860124">Pout Christian Bale GIF - Pout Christian Bale American Psycho - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/squidward-oh-no-hes-hot-shaking-gif-16063591">Squidward Oh No Hes Hot GIF - Squidward Oh No Hes Hot Shaking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/blog/lms">Introducing `lms` - LM Studio&#x27;s companion cli tool | LM Studio</a>: Today, alongside LM Studio 0.2.22, we&#x27;re releasing the first version of lms — LM Studio&#x27;s companion cli tool.</li><li><a href="https://rentry.co/zbofr34p">elija@mx:~$ xvfb-run ./LM_Studio-0.2.22.AppImage</a>: 20:29:24.712 › GPU info: '1c:00.0 VGA compatible controller: NVIDIA Corporation G A104 [GeForce RTX 3060 Ti] (rev a1)' 20:29:24.721 › Got GPU Type: nvidia 20:29:24.722 › LM Studio: gpu type = NVIDIA 2...</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b2775">Release b2775 · ggerganov/llama.cpp</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/lms?tab=readme-">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/lms?tab=readme-ov-file#installation.">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/ollama/ollama/issues/4051#issuecomment-2092092698">Enable Flash Attention on GGML/GGUF (feature now merged into llama.cpp) · Issue #4051 · ollama/ollama</a>: Flash Attention has landed in llama.cpp (ggerganov/llama.cpp#5021). The tldr; is simply to pass the -fa flag to llama.cpp’s server. Can we please have an Ollama server env var to pass this flag to ...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1235541447042797618)** (159 messages🔥🔥): 

- **Quest for Quality Story Writing**: A user is seeking help to create **iQuant** versions of **Goliath 120B Longlora** on their PC for high-quality story writing, requiring a minimum of 8K context for usability; they've offered Humblebundle Steam games as a reward for assistance. They highlighted the need for higher quality beyond what models like LLAMA 3 8B can provide, sharing their system prompt details located on [Google Docs](https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk).

- **Model Recall Experimentation**: Several chats involve users testing the recall ability of various models, particularly **LLAMA 3**'s ability to recall Bible verses. A user has set up a [GitHub repository](https://github.com/rugg0064/llm-bible-bench) for a "bible recall benchmark" and observed a very low recall rate across the entire Bible.

- **Exploring the Cognitive Horizons**: Users have been experimenting with models to see how they recall extensive texts like the Bible and discussing creating instances that can communicate between each other for better outcomes. One user proposed using "agents" that could optimize narrative quality, citing a [YouTube video](https://www.youtube.com/watch?v=sc5sCI4zaic) for reference.

- **Template Troubles and Quirky Answers**: A user experimenting with a new release, **[ChatQA 1.5](https://huggingface.co/bartowski/Llama-3-ChatQA-1.5-8B-GGUF)**, reports oddities in response templates leading to bizarre responses, even when applying suggested changes such as adding spaces or newlines to the chat template.

- **In Quest of Unrestricted Coding Model**: A user inquires about a good small model in the 2B parameter for coding applications with minimal censorship, but no suggestions have been provided in the discussion. Another user is looking for a model to read documents and PDFs, with **Command-R** by Cohere being suggested for document understanding tasks, albeit with concerns regarding its hardware requirements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/F2mBLoN">GoldenSun3DS's unclaimed Humblebundle Games</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://tenor.com/view/im-out-no-thanks-bugs-bunny-oh-no-not-interested-gif-16824550">Im Out No Thanks GIF - Im Out No Thanks Bugs Bunny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/daleks-exterminate-doctor-who-whovian-gif-10468156">Daleks Exterminate GIF - Daleks Exterminate Doctor Who - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/mradermacher/Goliath-longLORA-120b-rope8-32k-fp16-GGUF">mradermacher/Goliath-longLORA-120b-rope8-32k-fp16-GGUF · Hugging Face</a>: no description found</li><li><a href="https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk">High Quality Story Writing Type Third Person</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=sc5sCI4zaic">LLM In-Context Learning Masterclass feat My (r/reddit) AI Agent</a>: LLM In-Context Learning Masterclass feat My (r/reddit) AI Agent👊 Become a member and get access to GitHub and Code:https://www.youtube.com/c/AllAboutAI/join...</li><li><a href="https://github.com/rugg0064/llm-bible-bench">GitHub - rugg0064/llm-bible-bench: A simple test for large language models and their recall on bible verses</a>: A simple test for large language models and their recall on bible verses - rugg0064/llm-bible-bench</li><li><a href="https://github.com/rugg0064/">rugg0064 - Overview</a>: Full stack web developer, working on small projects when I have time. - rugg0064</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cg8rhc/1_million_context_llama_3_8b_achieved/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://docs.google.com/document/d/1xrMwhrz4DIdwzY4gI3GIrxQ0phQjVNmu2RGKRnGnRAM/edit?usp=drivesdk">High Quality Story Writing Type First Person</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1235636061879668787)** (2 messages): 

- **LM Studio Introduces Companion CLI 'lms'**: LM Studio has rolled out a new command-line interface, **lms**, to ease the load/unload of LLMs and management of local servers. Community members can install the CLI with `npx lmstudio install-cli` and contribute to its MIT licensed source code at [GitHub - lmstudio-ai/lms](https://github.com/lmstudio-ai/lms).

- **LM Studio 0.2.22 Bug Fix Released**: A bug affecting model responses by inadvertently including the entire context within them has been fixed in **LM Studio 0.2.22**. Users encountering this issue can download the updated version from [lmstudio.ai](https://lmstudio.ai).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai.">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1235581946390646804)** (8 messages🔥): 

- **Quick Fix for Llama and Phi-3 Configs**: If you delete your **configs folder** and relaunch the app, it will repopulate with the default configurations. Backing up any important config files first is advised.

- **WSL Woes with LM Studio**: Trying to connect to **LM Studio** through **Windows Subsystem for Linux (WSL)** can fail if using localhost addresses, since 127.0.0.1 accesses the local loop of the VM. *ipconfig* can help find the correct IP to use.

- **Passing Through Ports for Windows-WSL Communication**: A member suggested the use of a reverse proxy or port proxy with `netsh interface portproxy add v4tov4` command to communicate between **Windows and WSL** for LM Studio. No additional layers of complexity with listen addresses are necessary according to another member.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1235557374119378954)** (4 messages): 

- **Seeking the Link to a VRAM Fix**: A member mentioned a fix that should be linked, claiming that "it really does work" as a better solution than disabling iGPU in BIOS.
- **GPU Hunt for VRAM**: A user inquired about a "cheapish low profile, lowish power 12 GB + GDDR6 GPU" for a second PCI-E slot aimed specifically at utilizing the VRAM.
- **RTX 3060 as a VRAM Solution**: In response to a query about a GPU for VRAM usage, another member suggested considering the Nvidia **RTX 3060**.
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1235528523314233406)** (18 messages🔥): 

- **Multi Model Session Context Window Confusion**: A member discussed an issue with the Multi Model Session feature where they were unable to change the context window size, defaulting to 2048, and experienced timeouts when requests queued up. They indicated that the tool entered a *'Rolling Window'* mode and generated irrelevant responses.

- **Ubuntu Users Get Running Tips**: In response to a question about running the tool on Ubuntu, a simple instruction set was given: download the Appimage, make it executable, and run the application.

- **Docker Enthusiasts Can Go Headless**: An improvement to the software now allows it to run headlessly, which a member noted could enable them to finally create a working Docker image for testing.

- **Configurations and CLI Questions Addressed**: Members inquired about persisting settings like GPU offload and CORS through the CLI, prompting another to clarify that model configurations in the "My Models" page default but can be overridden in CLI/SDK per field.

- **Possible Bug Identified in GPU Layer Configuration**: An issue was reported regarding a config preset for GPU layers being overridden when loading models via CLI. It was suggested to open a GitHub issue to address this, and a link to the configurations schema was provided for reference on available parameters.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/lmstudio-ai/lms/issues">Issues · lmstudio-ai/lms</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/lms/issues/6">BUG: Loading a model via CLI ignores &quot;n_gpu_layers&quot; parameter in config preset · Issue #6 · lmstudio-ai/lms</a>: I have set &quot;n_gpu_layers&quot;: -1, in the preset I&#39;ve selected as default for a model. However when I use the cli to load that model lms load --identifier llama3-8b-8k &gt;&gt; select model ...</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/schema.json#L26">configs/schema.json at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1235666725496684626)** (32 messages🔥): 

- **LM Studio CLI Launches for ROCm**: LM Studio introduces `lms`, a new CLI for managing LLMs and running the local server on AMD ROCm Preview Beta, now [open source on GitHub](https://github.com/lmstudio-ai/lms). Users can download the latest LM Studio 0.2.22 ROCm Preview to utilize `lms`, which comes with the additional benefit of having OpenCl pre-packaged for new users.
  
- **Prompt in API Response Bug Acknowledged**: A user noted that the prompt is included in the API response, a known issue in the latest build. The LM Studio team rapidly acknowledged and confirmed an [imminent fix was pushed live](https://lmstudio.ai/rocm), users have verified it.

- **Large Context Size Exploration**: A participant tested the RAM scaling versus context size by attempting a context of 131072 tokens with Phi 3, but it failed. However, they could successfully run a context size of 60000 tokens with 32 GB RAM on a 7900XTX GPU.

- **Seeking Clarification on Embedding Model Issue**: User reported an issue when trying to load the embedding model in the new release. An immediate fix was released by LM Studio, and the user confirmed the solution worked after re-downloading from [LM Studio ROCm download page](https://lmstudio.ai/rocm).

- **Discussion on Linux Support for ROCm**: Participants are discussing running ROCm on Linux, with one sharing their experience of using ROCm on Mesa's opencl implementation and hoping for a Linux-supported ROCm build, while another suggested using lm-studio to download models for local llama.cpp build could be a workaround.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/rocm,">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://tenor.com/view/oil-gif-21418714">Oil GIF - Oil - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://releases.lmstudio.ai/windows/0.2.22-ROCm-Preview/beta/LM-Studio-0.2.22-ROCm-Preview-Setup.exe">no title found</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : add Flash Attention by ggerganov · Pull Request #5021 · ggerganov/llama.cpp</a>: ref #3365 Setting up what&#39;s needed for Flash Attention support in ggml and llama.cpp The proposed operator performs: // new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused scale ...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1235733449042821140)** (1 messages): 

- **LM Studio v0.0.22 Refreshes Llama Models**: The latest update to LM Studio includes a significant improvement for `llama.cpp` addressing **Llama 3** and **BPE model** issues. **BPE-fix tagged** versions of Llama 3 8B, 70B instruct, and Phi-3 models are available to download at the provided Hugging Face links.
    - [Meta-Llama-3-8B-Instruct-BPE-fix](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-BPE-fix-GGUF)
    - [Meta-Llama-3-70B-Instruct-BPE-fix](https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-BPE-fix-GGUF)
    - [Phi-3-mini-4k-instruct-BPE-fix](https://huggingface.co/lmstudio-community/Phi-3-mini-4k-instruct-BPE-fix-GGUF)
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1235640547545448448)** (69 messages🔥🔥): 

- **CLI Companion for LM Studio Introduced**: LM Studio's new CLI tool, `lms`, has been released to facilitate loading LLMs, starting/stopping servers, and debugging. Users can [install it](https://github.com/lmstudio-ai/lms) directly and it requires LM Studio 0.2.22 or newer.

- **Headless Tutorial for Running LM Studio**: A member shared a *poorly written hacky headless tutorial* for running LM Studio headlessly, which includes instructions using xvfb to emulate X11 session and bootstrapping `lms`. Another member confirmed getting it to work on Ubuntu Server after some troubleshooting.
  
- **Resolving App Exit Issues**: There were several messages focused on addressing an issue where the LM Studio app exited upon a command, with discussions about troubleshooting steps such as using `ctrl+z` then `bg`, `disown -ah`, and `--no-sandbox` flags.

- **Scripting to Streamline Installations**: A member has expressed intentions to create a script that will automate headless installations of LM Studio, allowing for a more straightforward setup with one command.

- **Progress Towards Dockerization**: A member expressed excitement over being able to create a Docker container for LM Studio, which would ease running and testing models on servers, following the successful headless installation tutorial.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lmstudioai/status/1786076035789815998?s=46">Tweet from LM Studio (@LMStudioAI)</a>: Introducing lms -- LM Studio&#39;s companion cli 😎  ✨ Load / unload LLMs, start/stop the local server 📖 Debug your workflows with lms log stream 🛠️ Run `npx lmstudio install-cli` to install lms 🏡 ...</li><li><a href="https://releases.lmstudio.ai/linux/0.2.22.b/beta/LM_Studio-0.2.22.AppImage">no title found</a>: no description found</li><li><a href="https://tenor.com/view/qawe-asd-gif-26050335">Qawe Asd GIF - Qawe Asd - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1235982270985142463)** (1 messages): 

- **Beta Tester Recruitment for Pages Concluded**: The recruitment for beta testers for **Pages** has met the desired number of participants. The team expressed gratitude and advised everyone to stay tuned for further updates on the development of Pages.
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1235490446109577216)** (308 messages🔥🔥): 

- **Technical Difficulties with Perplexity**: Users have reported issues with Perplexity not functioning correctly on Safari and Brave browsers, such as not being able to send prompts or register due to unresponsive buttons. Others experience persistent sourcing from previously uploaded files during conversations, mistakenly retrieving data from earlier requests.
  
- **Subscription and Payment Inquiry**: A user inquired about obtaining a refund for an unwanted monthly subscription charge and was advised to contact support@perplexity.ai for assistance.

- **Feature Requests and Feedback for Perplexity's Tools**: Members have expressed a desire for improved functionality with voice commands and the continuation of certain features, suggesting enhancements like avoiding premature command termination and enabling continuous listening.

- **Usage and Model Limits Discussed**: There is confusion regarding usage limits for different models and tools within Perplexity, with some users unsure about daily query allowances, and others debating the comparative capabilities of different AI models like Gemini 1.5 Pro, Claude Opus, and GPT-4 Turbo.

- **Anticipation for Future AI Developments and Competitor Platforms**: The community anticipates new AI models such as the rumored "GPT-5" and potential upcoming Perplexity competitors from OpenAI. Additionally, there are discussions on the distinctions between search engines and knowledge engines, with speculations about how these tech advancements might evolve and integrate with existing platforms.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/24111326/ai-search-perplexity-copilot-you-google-review">Here’s why AI search engines really can’t kill Google</a>: A search engine is much more than a search engine, and AI still can’t quite keep up.</li><li><a href="https://tenor.com/view/imagination-spongebob-squarepants-dreams-magic-gif-12725683">Imagination Spongebob Squarepants GIF - Imagination Spongebob Squarepants Dreams - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aistudio.google.com/app/prompts/new_chat">no title found</a>: no description found</li><li><a href="https://youtu.be/77IqNP6rNL8">New OpenAI Model &#39;Imminent&#39; and AI Stakes Get Raised (plus Med Gemini, GPT 2 Chatbot and Scale AI)</a>: Altman ‘knows the release date’, Politico calls it ‘imminent’ according to Insiders, and then the mystery GPT-2 chatbot [made by the phi team at Microsoft] c...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1235488309426257961)** (22 messages🔥): 

- **Link Sharing Protocol Reminder**: A [Perplexity AI reminder](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) was issued to ensure threads are made *Shareable* before posting on Discord.
- **Lunar Love for Moon Fans**: An interesting answer for enthusiasts of lunar topics is shared, with a link to [Perplexity's response on the fictional name of the moon](https://www.perplexity.ai/search/In-the-fictional-ySifBWwWSeeXk27x5glOOw).
- **Melodic Discovery in AI**: A member shared a link related to musical creations, specifically the composition "We Interface" and its discovery via AI at [Perplexity AI's search result](https://www.perplexity.ai/search/We-Interface-song-kK0EbdFjR2yh_7Vn3Xbg0Q).
- **Privacy Concerns with Printer Technology**: Privacy enthusiasts or anyone with an interest in printer tracking dots can find information in this [Perplexity search result](https://www.perplexity.ai/search/Printer-Tracking-Dots-NcXiviwKQS2nGmbu4lqEnw).
- **Exploration of AI-Generated Content**: Links to Perplexity AI content about a new AI debut, [Exploring XDreams](https://www.perplexity.ai/page/Exploring-XDreams-Debut-UyAeq.Q_TxyMAP2avgAELw), and [XAvengers: Cyborg Wars](https://www.perplexity.ai/page/XAvengers-Cyborg-Wars-Ku62KQMbQ8W.9EqM9RTs1g) indicate a growing interest in AI-generated narratives and games.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1235604126428299344)** (41 messages🔥): 

- **Sonar Large Availability and Typo Clarification**: **Sonar Large** is available for use via the API, and [the model cards documentation](https://docs.perplexity.ai/docs/model-cards) shows it listed with a 32k context length. A confusion about the parameter count led to a clarification that *Sonar Large* is a 70B model, contrary to a typo suggesting it's 8x7B.
  
- **Prompt Precision Leads to Better Results**: Members noted improved outcomes when using precise terms in prompts, such as specifying `https://` in front of URLs. One user's experience with **llama-3-sonar-large-32k-online** yielded better results after adjusting prompts to generate markdown lists of competitors.

- **API Client Experiences Variable Results**: Even after adjusting prompts, a user reported inconsistent results from the API, which sometimes provided correct competitors and at other times failed. Tweaking AI model settings and prompt optimization was suggested as a solution.

- **Model Transition Guidance Sought**: Queries were raised regarding the need to transition from **sonar-medium-online** to newer models. Advice received suggested trying **llama-3-sonar-small-32k-online** for better accuracy, with a clear indication that an update to the new models would eventually be necessary.

- **Adjusting AI Parameters to Improve Responses**: To improve accuracy, a user tested different `frequency_penalty`, `temperature`, and `top_p` settings, finding that changes to these parameters influenced the relevance and correctness of the AI's responses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://optonal.com`">no title found</a>: no description found</li><li><a href="http://www.ghirardelli.com)">no title found</a>: no description found</li><li><a href="http://www.godiva.com)">no title found</a>: no description found</li><li><a href="http://www.lindt.com)">no title found</a>: no description found</li><li><a href="http://www.russellstover.com)">no title found</a>: no description found</li><li><a href="http://www.hersheys.com)">no title found</a>: no description found</li><li><a href="http://www.dovechocolate.com)">no title found</a>: no description found</li><li><a href="http://www.toblerone.com)">no title found</a>: no description found</li><li><a href="http://www.lamaisonduchocolat.com)">no title found</a>: no description found</li><li><a href="http://www.pierremarcolini.com)">no title found</a>: no description found</li><li><a href="http://www.vosgeshautchocolat.com)">no title found</a>: no description found</li><li><a href="http://www.teuscher.com)">no title found</a>: no description found</li><li><a href="https://"">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://optonal.com">OpTonal • AI Sales Agent for Teams Using Slack, HubSpot, Google Meet</a>: no description found</li><li><a href="https://sensiseeds.com](https://sensiseeds.com)\n2.">no title found</a>: no description found</li><li><a href="https://seed.com](https://seed.com)">no title found</a>: no description found</li><li><a href="https://www.salesforce.com/">Salesforce: The Customer Company</a>: Salesforce, the #1 AI CRM, enables companies to connect with customers through a unified Einstein 1 platform that combines CRM, AI, Data, and Trust.</li><li><a href="https://www.hubspot.com/products/sales">Sales Software for Small to Enterprise Companies | Start for Free</a>: Powerful sales software to help your team close more deals, deepen relationships, and manage their pipeline more effectively — all on one connected platform.</li><li><a href="https://www.zoho.com/crm/">Zoho CRM | Top-rated Sales CRM Software by Customers</a>: Zoho CRM is an online Sales CRM software that manages your sales, marketing, and support in one CRM platform. Trusted by over a 100 million users worldwide! Sign up for a free trial today.</li><li><a href="https://www.gong.io/">Gong - Revenue Intelligence Platform</a>: Gong captures customer interactions then delivers insights at scale, empowering teams to make decisions based on data instead of opinions.</li><li><a href="https://www.exceed.ai/">#1 Conversational Marketing and Sales Platform - Exceed.ai</a>: Enhance lead conversion with Conversational AI. Automate revenue interactions, engage at scale, and interact via Email, Chat, SMS.</li><li><a href="https://salesloft.com/">Salesloft: The Leading Sales Engagement Platform</a>: no description found</li><li><a href="https://www.yesware.com/">Sales Engagement Made Easy | Yesware</a>: Yesware helps high-performing sales teams do meaningful email outreach at scale. If you need to drive more revenue through email outreach, but complex platforms are overkill — try Yesware.</li><li><a href="http://ghirardelli.com)">no title found</a>: no description found</li><li><a href="http://hersheys.com)">no title found</a>: no description found</li><li><a href="http://russellstover.com)">no title found</a>: no description found</li><li><a href="http://lindt.com)">no title found</a>: no description found</li><li><a href="http://godiva.com)">no title found</a>: no description found</li><li><a href="https://sidecardoughnuts.com/)">Sidecar Doughnuts - The World&#039;s Freshest Doughnuts!</a>: Serving The World’s Freshest Doughnuts, Signature Blend Coffees &amp; Service with a Smile Since 2012 | Costa Mesa, Santa Monica, &amp; Del Mar CA</li><li><a href="https://thepieholela.com/)">The Pie Hole</a>: Need fresh pies or Pie Holes for your next event? Order online and have them delivered nationwide with free shipping because pie is love.
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1235508251219591230)** (15 messages🔥): 

- **Exploring Retrocausality and Morality**: The concept of *moral non-commutativity* in retrocausality was discussed, highlighting the psychological perspective where patients do not distinguish between cause and consequence in moral actions, impacting the integrity of an observer's moral framework.
- **Seeking Llamacpp Guidance**: A member asked for a beginner's guide to *llamacpp* after experiencing issues with the model generating nonsensical outputs and a website automatically writing a C function.
- **Using Llama on CPU**: It was suggested to use *ollama* as a backend for *llamacpp* to avoid directly dealing with C, and a discussion touched on the advances in running large language models like LLMs on CPUs utilizing techniques like quantization and pruning.
- **Waiting for lmstudio Approval**: One member expressed frustrations about not being able to use their laptop for model-related tasks due to waiting for approval from *lmstudio*.
- **Saint Petersburg Transformation Over Time**: Photos of Saint Petersburg’s Ligovsky Avenue at Vosstaniya Square from 2002 versus 2024 prompted a joke about the improved color accuracy in cameras.
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1235647006773608500)** (19 messages🔥): 

- **Proprietary Intrigue**: A **proprietary** Twitter post sparked interest, but the details remain undisclosed, no further information provided.
- **Haystack Goes Embedded**: The GitHub repository for **haystack-embedded** by carsonpo was highlighted, an open-source contribution for embedded machine learning development accessible [here](https://github.com/carsonpo/haystack-embedded).
- **Excitement Over WildChat Dataset**: The allenai **WildChat** dataset has generated conversation, however, access requires agreement to the [AI2 ImpACT License](https://allenai.org/licenses/impact-lr). The dataset seems to feature "long multiturn convos" and is housed on Hugging Face's platform, with anticipation for a new version indicated by a URL to [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M).
- **Mixed Messages on Dataset Release**: Discussion centered around whether the new WildChat dataset had been open-sourced, with confirmation seen via a link on an [arXiv abstract](https://arxiv.org/abs/2405.01470).
- **Preference for OPUS in Long Conversations**: A member mentioned a preference for the **OPUS** model versus others when dealing with long conversational contexts, suggesting better performance after "10/20k of prompting.”
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/allenai/WildChat-1M">allenai/WildChat-1M · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/allenai/WildChat?not-for-all-audiences=true">allenai/WildChat · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/carsonpo/haystack-embedded">GitHub - carsonpo/haystack-embedded</a>: Contribute to carsonpo/haystack-embedded development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1235486410937466891)** (104 messages🔥🔥): 

- **Hermes Upgrade Unveiled**: Nous has released **Hermes 2 Pro** with LLaMA weights, boasting capabilities in good QA, Function Calling and JSON Mode with vision multimodal. Models and test code available on [Hugging Face](https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B).

- **Enabling Advanced Function Calling**: BLOC97 explained that an LLM with Function Calling is aware of external function/tool calls to validate answers instead of simulating answers, and teknium shared a GitHub repo with examples of tunes specific to function calling for [Hermes](https://github.com/NousResearch/Hermes-Function-Calling/tree/main).

- **Function Call Dataset Insights**: [Glaive function-calling dataset V2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) was shared to showcase the structure of a data set used for training a model with a function-calling feature. The conversations surrounding the use of these datasets emphasized their potential for advanced LLM applications.

- **The Impact of llcpp on Model Performance**: Diabolic6045 experienced exceptional inference speeds when using Hermes 2 Pro with llama.cpp on an Android device with 8GB RAM, highlighting the efficiency of the technology.

- **Leveraging Hermes 2 Pro with CrewAI and LocalAI**: .interstellarninja provided solutions for using Hermes 2 Pro function-calling with CrewAI by sharing a [Jupyter notebook](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/crewai_agents.ipynb). They also pointed to the LocalAI API supporting function-calling with OpenAI API tool calls format detailed in their [repository](https://github.com/mudler/LocalAI/blob/master/gallery/hermes-2-pro-mistral.yaml).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.17733">Building a Large Japanese Web Corpus for Large Language Models</a>: Open Japanese large language models (LLMs) have been trained on the Japanese portions of corpora such as CC-100, mC4, and OSCAR. However, these corpora were not created for the quality of Japanese tex...</li><li><a href="https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw?usp=sharing#scrollTo=2EoxY5i1CWe3">Google Colab</a>: no description found</li><li><a href="https://x.com/DimitrisPapail/status/1786045418586972208">Tweet from Dimitris Papailiopoulos (@DimitrisPapail)</a>: The most surprising finding of this report is hidden in the appendix. Under the best of two prompts the models don&#39;t overfit that much, unlike what the abstract claims.  Here is original GSM8k vs ...</li><li><a href="https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B">vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.25-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.25-exl2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/crewai_agents.ipynb">Hermes-Function-Calling/examples/crewai_agents.ipynb at main · NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/tree/main">GitHub - NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="https://github.com/e2b-dev/code-interpreter">GitHub - e2b-dev/code-interpreter: Python &amp; JS/TS SDK for adding code interpreting to your AI app</a>: Python &amp; JS/TS SDK for adding code interpreting to your AI app  - GitHub - e2b-dev/code-interpreter: Python &amp; JS/TS SDK for adding code interpreting to your AI app</li><li><a href="https://github.com/mudler/LocalAI/blob/master/gallery/hermes-2-pro-mistral.yaml">LocalAI/gallery/hermes-2-pro-mistral.yaml at master · mudler/LocalAI</a>: :robot: The free, Open Source OpenAI alternative. Self-hosted, community-driven and local-first. Drop-in replacement for OpenAI running on consumer-grade hardware. No GPU required. Runs gguf, trans...</li><li><a href="https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2">glaiveai/glaive-function-calling-v2 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1235582545362685993)** (45 messages🔥): 

- **ChatML Configurations Revealed**: Members were discussing the modifications needed to enable ChatML, mentioning **token replacement** and adjustments in the model configuration such as replacing EOS with `
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.writewithlaika.com)">no title found</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1235579505632874569)** (1 messages): 

- **Enthusiasm for LLM Finetuning**: A new member expressed keen interest in **finetuning a large language model (LLM)** before becoming a miner. They sought advice on how to find datasets suitable for this purpose and the kind of data required for effective finetuning.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 messages): 

felixultimaforeverromanempire: anyone know fo good free generic data sets?
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1235586429401432156)** (86 messages🔥🔥): 

<ul>
<li><strong>Iron Age Update in World-sim</strong>: A member mentioned being on world 11 in the game where an Iron Age update was recently implemented.</li>
<li><strong>Gaming Nostalgia with "Spore"</strong>: A member reminisced about spending over 100 hours playing the game "Spore."</li>
<li><strong>Anticipation for Upcoming Updates and Celebrations</strong>: A member expressed excitement for something coming this weekend and shared that they will turn 18, marking it as a significant birthday.</li>
<li><strong>Discussion on AI and Consciousness</strong>: Members expressed admiration for a speech by Joscha on consciousness, citing its profound impact, and shared related [YouTube](https://www.youtube.com/watch?v=abWnhmZIL3w) videos on the topic.</li>
<li><strong>New Discord Role for Worldsim Updates</strong>: A new role was created to tag members for smaller worldsim/worldclient related information, with several members requesting to be added to it, which can be obtained via the <id:customize> channel.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/cs9Ls0m5QVE?si=YD9rEG7jZNBUJbpS">37C3 -  Synthetic Sentience</a>: https://media.ccc.de/v/37c3-12167-synthetic_sentienceCan Artificial Intelligence become conscious?Despite the rapid progress of AI capabilities, the core que...</li><li><a href="https://youtu.be/YZl4zom3q2g?si=xqoxcI1yibo5Td1H">Cyber Animism by Joscha Bach</a>: This is a 1 hour 45 minute talk by Joscha Bach (http://bach.ai/) given in our Center.</li><li><a href="https://www.youtube.com/watch?v=abWnhmZIL3w">World Simulation Talks @ AGI House SF</a>: 0:00 Conversation1:31 Kickoff by Jeremy Nixon6:08 Karan Malhotra of Nous Research26:22 Rob Hasfield: CEO of Websim1:00:08 Ivan Vendrov of Midjourney [Real ti...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1235660780351787008)** (99 messages🔥🔥): 

- **Mojo Joins the Language Race**: A [YouTube video featuring Chris Lattner](https://www.youtube.com/watch?v=JRcXUuQYR90), discussing "Mojo Lang - Tomorrow's High Performance Python?" was shared, highlighting the new language's attempt to integrate the best programming techniques from CPU/GPU development.
- **Learning Mojo with a Python Background**: Discussion centers around the relationship between Python and the new Mojo language, with members noting that while Mojo has similarities to Python and can use Python objects directly, due to strong type checking and other systems-programming features, there are significant differences. [Mojo documentation](https://docs.modular.com/mojo/manual/basics) is recommended for those looking to understand Mojo's unique features.
- **Open Source Contributions and Guidance**: Members are encouraged to contribute to the open-source Mojo standard library, with links to the [GitHub contributing guide](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md) and a [Modular blog post](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide) offering step-by-step instructions for potential contributors.
- **Discussions on Development Coordination**: There is an ongoing dialogue about how best to manage contributions and avoid duplication of effort on GitHub issues. One proposal includes the use of a [PR template](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository) to help link issues and PRs effectively.
- **Assessment of Mojo's Advantages**: Conversations delve into what sets Mojo apart, such as performance, predictability, and portability features. It is also mentioned that Mojo's build system automatically autotunes for optimal performance on different hardware, as demonstrated by a [Jeremy Howard's video on autotuning](https://youtu.be/6GvB5lZJqcE?t=281).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository">Creating a pull request template for your repository - GitHub Docs</a>: no description found</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</li><li><a href="https://devblogs.microsoft.com/oldnewthing/20091201-00/?p=15843)">Microspeak: Cookie licking - The Old New Thing</a>: Now nobody else can have it.</li><li><a href="https://docs.modular.com/mojo/manual/basics">Introduction to Mojo | Modular Docs</a>: Introduction to Mojo&#x27;s basic language features.</li><li><a href="https://open.spotify.com/track/3XwQ8ks84wlj3YcRyxXrlN?si=XJlRyCe_TzOmqPwVtDbCQQ&utm_source=copy-link">Mojo</a>: -M- · Song · 2012</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md">mojo/stdlib/docs/development.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=JRcXUuQYR90">Mojo Lang - Tomorrow&#39;s High Performance Python? (with Chris Lattner)</a>: Mojo is the latest language from the creator of Swift and LLVM. It’s an attempt to take some of the best techniques from CPU/GPU-level programming and packag...</li><li><a href="https://github.com/apple/swift/issues/43464">[SR-852] [QoI] Poor diagnostic with missing &quot;self.&quot; in convenience initializer · Issue #43464 · apple/swift</a>: Previous ID SR-852 Radar None Original Reporter @ddunbar Type Bug Status Resolved Resolution Done Additional Detail from JIRA Votes 0 Component/s Compiler Labels Bug, DiagnosticsQoI Assignee @dduan...</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#create-a-pull-request">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://youtu.be/JRcXUuQYR90?t=113)?">Mojo Lang - Tomorrow&#39;s High Performance Python? (with Chris Lattner)</a>: Mojo is the latest language from the creator of Swift and LLVM. It’s an attempt to take some of the best techniques from CPU/GPU-level programming and packag...</li><li><a href="https://github.com/modularml/mojo/issues/2487">[Feature Request] Make the `msg` argument of `assert_true/false/...` keyword only · Issue #2487 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? As title. What is your motivation for this change? To ...</li><li><a href="https://www.youtube.com/watch?v=SEwTjZvy8vw">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>: 2023 LLVM Developers&#39; Meetinghttps://llvm.org/devmtg/2023-10------Mojo 🔥: A system programming language for heterogenous computingSpeaker: Abdul Dakkak, Chr...</li><li><a href="https://youtu.be/6GvB5lZJqcE?t=281)">Jeremy Howard demo for Mojo launch</a>: This is a section from the Modular launch video. The full video, docs, and details are here: https://www.modular.com/</li><li><a href="https://github.com/modularml/mojo/issues/2415">[Feature Request] Add `__rfloordiv__()` to SIMD type · Issue #2415 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? The Int and Object types support rfloordiv. I added th...</li><li><a href="https://github.com/modularml/mojo/pull/2457">[stdlib] Support print to stderr by GeauxEric · Pull Request #2457 · modularml/mojo</a>: Add keyword argument to print function to support stream to stderr. Fix #2453 Signed-off-by: Yun Ding yunding.eric@gmail.com
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1235662100731265147)** (3 messages): 

- **Modular's Latest Tweets**: Modular shared a tweet, accessible via [this link](https://twitter.com/Modular/status/1786096043463184528), but the content of the tweet was not discussed.
- **Another Tweet from Modular**: A second tweet was shared by Modular, which can be found [here](https://twitter.com/Modular/status/1786096058113876311), though no further details or discussion points about the tweet were provided.
- **Modular Tweets Again**: Modular posted another tweet, which can be seen [at this link](https://twitter.com/Modular/status/1786483510141657384). There was no accompanying conversation or explanation of its significance in the chat.
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1235652713954676849)** (2 messages): 

- **Modular Celebrates Community Contributions in Mojo 24.3**: Mojo 🔥 24.3 has been released with significant community involvement after the open-sourcing of Mojo's standard library. The update boasts contributions that bolster the platform's capabilities, with special thanks to contributors like [@LJ-9801](https://github.com/LJ-9801), [@mikowals](https://github.com/mikowals), and others listed in the release notes.

- **Unveiling MAX 24.3 with Engine Extensibility**: The MAX 24.3 update features the new MAX Engine Extensibility API, enhancing the ability for developers to build and run AI pipelines efficiently. This version offers improved integration for PyTorch, ONNX, and Mojo models, as well as a range of performance optimizations for diverse hardware through the [MAX Graph APIs](https://docs.modular.com/engine/graph).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/max-24-3-introducing-max-engine-extensibility">Modular: MAX 24.3 - Introducing MAX Engine Extensibility</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: MAX 24.3 - Introducing MAX Engine Extensibility</li><li><a href="https://www.modular.com/blog/whats-new-in-mojo-24-3-community-contributions-pythonic-collections-and-core-language-enhancements">Modular: What’s New in Mojo 24.3: Community Contributions, Pythonic Collections and Core Language Enhancements</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What’s New in Mojo 24.3: Community Contributions, Pythonic Collections and Core Language Enhancements
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1235659563408166933)** (1 messages): 

- **MAX ⚡️ and Mojo 🔥 Release 24.3 Goes Live**: **Release 24.3** is now available, including the latest versions of MAX and Mojo. The installation commands are provided, and the release can be accessed via a simple curl script and Modular CLI commands.
- **Celebrating One Year of Mojo 🔥**: This update marks the **first anniversary** of Mojo, with gratitude extended to the community for their contributions to the release.
- **Launch Blog and Extensibility Features Explained**: Interested users can read about the launch on the [official blog post](https://modul.ar/24-3) and learn about the new **MAX extensibility** features in a dedicated [blog post](https://modul.ar/max-extensibility).
- **Community Contributions Recognized**: The changelog mentions **32 significant changes, fixes, and features** contributed by the community, highlighting the collaborative efforts in the development process.
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1235664363566923809)** (4 messages): 

- **The Challenge of Simulating Consciousness**: The discussion touched on the complexity of simulating consciousness, suggesting that it not only requires scientific understanding but also philosophical insights. It was proposed that starting with simpler organisms could be the key, as their brains might be easier to map and replicate in code.

- **Hoffman's Work Inspires Future Academia**: One member expressed their plans to transfer to UCI to be closer to the work of Donald Hoffman, a professor who is actively working on mapping conscious experiences. This aligns with the view of functionalism, where simulating brain functions might be more feasible than replicating the brain entirely.

- **Aspiring to Explore Consciousness**: Another member shared their goal of working on the simulation of consciousness, resonating with the previous discussion on the subject.
  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1235612025615421545)** (2 messages): 

- **CHERI Blossoming into Daily Use**: Chats highlight that the **Capability Hardware Enhanced RISC Instructions (CHERI)** offers considerable promise in improving hardware security, with potential to nullify 70% of current vulnerability exploits. The discussion was spurred by a recent [conference playlist](https://youtube.com/playlist?list=PL55r1-RCAaGU6fU2o34pwlb6ytWqkuzoi) that delves into the advancements within the CHERI ecosystem.

- **A Paradigm Shift in Software Development**: With the adoption of CHERI, software development could see a seismic shift, as processes could become orders of magnitude faster, resulting in efficient UNIX-style programming with high performance. This potentiality is discussed in the context of [CHERI facilitating lightning-fast IPC](https://github.com/CTSRD-CHERI/cheripedia/wiki/Colocation-Tutorial) and the inherent benefits of such capabilities.

- **Sandboxes Entering the Fast Lane**: The conversation moved towards how CHERI's *scalable compartmentalization* could fundamentally change environments that utilize sandboxes, impacting web browsers, virtual machines, and even edge computing. A [YouTube video](https://youtu.be/_QxXiTv1hH0?t=933) was referenced, illustrating this transformative tech.

- **Potential Redundancy of Traditional Security Measures**: Speculation abounds that with the rise of CHERI, traditional hardware security like MMU-based memory protection or address space layout randomization might become obsolete, thereby simplifying hardware design and enhancing software speed.

- **Microkernels Could Take Center Stage**: One member pondered if CHERI could precipitate a revolution in OS development, where the traditionally high cost of IPC in microkernels is countered, making them a potentially dominant architecture.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/CTSRD-CHERI/cheripedia/wiki/Colocation-Tutorial)">Home</a>: Placeholder for CHERI Wiki pages. Contribute to CTSRD-CHERI/cheripedia development by creating an account on GitHub.</li><li><a href="https://youtu.be/_QxXiTv1hH0?t=933)">Can future hardware make our software more secure?  Full event recording - Cambridge 15 March 2022</a>: How can future hardware make our software more secure?Frustrated with security issues in your code?Hate bugs that find you and not vice versa?Are you interes...</li><li><a href="https://youtu.be/_QxXiTv1hH0?t=1204))">Can future hardware make our software more secure?  Full event recording - Cambridge 15 March 2022</a>: How can future hardware make our software more secure?Frustrated with security issues in your code?Hate bugs that find you and not vice versa?Are you interes...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1235514390636396596)** (137 messages🔥🔥): 

- **Mojo Reference Semantics Still in Flux**: Mojo's semantics for references and lifetimes are being actively designed to offer **simpler yet flexible** structures than the existing prototype. A design will be shared publicly, and there is an ongoing debate on whether the semantics of reference and lifetime is nearly complete or will continue to have layers added on top.
- **Crash Reports and Bug Tracking**: Discussions point to a [crash report and a bug](https://github.com/modularml/mojo/issues/2429) related to `struct lifetime` that requires attention. Concerns are raised over compiler crashes that should instead provide meaningful error messages.
- **InlineArray Intrigues and Issues**: `InlineArray` is not yet on stable build; despite its utility, there are known quirks with large arrays and related GitHub issues indicate it's awaiting more stability. The feature is implemented in `utils.InlineArray`.
- **Debating Mojo's GPU Support**: Mojo is anticipated to soon have GPU support, starting with Nvidia, leveraging MLIR for versatility across platforms. Meanwhile, discussions clarify that it's a multi-year effort for existing languages to move from LLVM to MLIR, making Mojo's inherent MLIR integration special.
- **Interest in Snap Package and I/O Functions**: There is a request for an official Snap package on the Snap Store for Ubuntu and discussion on the current state of Mojo’s I/O module being basic, necessitating imports from Python for simple user input functionality like reading from `stdin`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140901/233938.html"> [llvm] r217292 - [docs] Document what &quot;NFC&quot; means in a commit	message.
   </a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/os/atomic">atomic | Modular Docs</a>: Implements the Atomic class.</li><li><a href="https://www.modular.com/blog/whats-new-in-mojo-24-3-community-contributions-pythonic-collections-and-core-language-enhancements#:~:text=This%20simplifies%20compare%20List%5BTuple%5BFloat64%2C%20Float64%2C%20Float64%5D%5D()%20vs%20List%5B(Float64%2C%20Float64%2C%20Float64)%5D()">Modular: What’s New in Mojo 24.3: Community Contributions, Pythonic Collections and Core Language Enhancements</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What’s New in Mojo 24.3: Community Contributions, Pythonic Collections and Core Language Enhancements</li><li><a href="https://github.com/modularml/mojo/issues/2425.">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2413">[Feature Request] Allow substitution of child traits for parent traits · Issue #2413 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? If a function takes variadic arguments bound by a trai...</li><li><a href="https://github.com/modularml/mojo/issues/2429">[mojo-nightly] struct lifetime issue · Issue #2429 · modularml/mojo</a>: Bug description In the following test demo. It seems the destructor is called on the filehandle instead of move. The demo runs without problems with stable but i get the following with nightly: fil...</li><li><a href="https://github.com/modularml/mojo/pull/2323?">[stdlib] Implement `List.__str__()` by gabrieldemarmiesse · Pull Request #2323 · modularml/mojo</a>: PR that can serve as a reference for #2190 (comment) Note that it causes a bug that seems on the parser side. We get this: RUN: at line 13: mojo /projects/open_source/mojo/stdlib/test/builtin/test_...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1235708968685273088)** (3 messages): 

- **Prism CLI Tool Gets Feature Boost**: The `prism` library has been updated with *persistent flags* and *hooks*, *flag requirements*, and *flag groups*. The README has been overhauled with code samples and animated gifs to demonstrate the new features. Check out the updates on [GitHub](https://github.com/thatstoasty/prism).

- **Mojo-pytest Now Supports v24.3**: The `mojo-pytest` plugin has been updated to work with Mojo **version 24.3**, with an open issue aiming to enhance integration for better debug information. Progress can be tracked on this enhancement at [Issue #9](https://github.com/guidorice/mojo-pytest/issues/9) on its GitHub repository.

- **NuMojo Outpaces Numpy and Numba**: The [NuMojo](https://github.com/MadAlex1997/NuMojo) project, previously known as Mojo-Arrays, is undergoing active development and now supports Mojo version 24.3. **NuMojo** is significantly outperforming NumPy and is also faster than Numba, focusing on expanding the standard library tensor functionality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/thatstoasty/prism">GitHub - thatstoasty/prism: Mojo CLI Library modeled after Cobra.</a>: Mojo CLI Library modeled after Cobra. Contribute to thatstoasty/prism development by creating an account on GitHub.</li><li><a href="https://github.com/guidorice/mojo-pytest">GitHub - guidorice/mojo-pytest: Mojo test runner, pytest plugin (aka pytest-mojo)</a>: Mojo test runner, pytest plugin (aka pytest-mojo). Contribute to guidorice/mojo-pytest development by creating an account on GitHub.</li><li><a href="https://github.com/guidorice/mojo-pytest/issues/9">Add filename, line and column number to MojoTestItem · Issue #9 · guidorice/mojo-pytest</a>: When a python test is collected by pytest, it reports a line number and context like this: def test_ex(): &gt; raise Exception(&quot;here&quot;) E Exception: here path/to/test_file.py:2: Exception In ...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1235553064061112330)** (3 messages): 

- **PyCon Lithuania Talk on MAX**: A new [YouTube video](https://youtu.be/Xzv2K7WNVD0) from PyCon Lithuania discusses MAX, but the title and description are currently undefined.
- **Tutorial on Building Apps with Mojo**: A new GitHub-based tutorial titled "Let's mojo build -D your own -D version=1 app" is available, teaching how to create or integrate workflows with the Mojo language. The tutorial can be found [here](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md).
- **Syntax Highlighting Tip for Mojo Tutorial**: A suggestion was made to use triple quotes with 'mojo' instead of 'python' in a markdown file for proper syntax highlighting when documenting Mojo code.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/Xzv2K7WNVD0)"> - YouTube</a>: no description found</li><li><a href="https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md">mojo-learning/tutorials/use-parameters-to-create-or-integrate-workflow.md at main · rd4com/mojo-learning</a>: 📖 Learn some mojo ! Contribute to rd4com/mojo-learning development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/)** (1 messages): 

soracc: Good idea
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 32
https://www.modular.com/newsletters/modverse-weekly-32
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1235670853040079019)** (10 messages🔥): 

- **Mojo Compiler Language Changes Alert**: The 24.3 changelog includes new information on `__source_location()` and `__call_location()`, which are detailed at [Modular Docs Changlog](https://docs.modular.com/mojo/changelog#language-changes). It seems they require `@always-inline` functions for full functionality.
- **Nightly Mojo Compiler Dropped**: A new nightly Mojo compiler release is announced, which can be updated using `modular update nightly/mojo`. [See what's changed](https://github.com/modularml/mojo/pull/2480/files) and review the changes from the [last stable release](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Docstrings Length Discussion**: There's a conversation about whether docstrings can exceed 80 columns, with a suggestion to consider relaxing this requirement, especially for the standard library.
- **Boost in Nightly Releases Frequency**: Nightly releases of the Mojo compiler are becoming more frequent, with an expectation set for daily updates soon, pending internal infrastructure improvements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/pull/2480/files">[stdlib] Update stdlib corresponding to 2024-05-03 nightly/mojo by JoeLoser · Pull Request #2480 · modularml/mojo</a>: This updates the stdlib with the internal commits corresponding to today&#39;s nightly release: mojo 2024.5.303.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1235548331778900009)** (224 messages🔥🔥): 

- **The Ephemeral AI Job Market**: Members debated the highest-paying jobs in AI, suggesting that the most sought-after positions are constantly evolving. Some joked that the most lucrative AI careers might be becoming a CEO or dentist.

- **Predicting the Price of Future GPT Versions**: Discussions emerged over the possibility of a separate pricing tier for the hypothetical GPT-5, with opinions varying on whether OpenAI would introduce regional pricing or maintain a consolidated price model.

- **Copycat UI Raises Eyebrows**: Comments arose about the new HuggingChat UI closely resembling existing AI chat services, with some implying it could be a game-changer for providing consumer-facing products and fostering an open-source AI community.

- **AI's Existential Debate**: A thorough discussion took place on the nature of AI's growth, human uniqueness, generative abilities, and the blending of hallucination with reality. Concerns were raised about AI overconfidence and its capacity for misleading information.

- **The Transparency of AI Research and Open Source Misconceptions**: A series of messages clarified that while OpenAI's research papers are publicly available, expecting the organization to release fully trained models is unrealistic given their proprietary nature and the computational resources required to run them.
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1235732447258480760)** (23 messages🔥): 

- **Nostalgic Throwback to GPT-3 and Codex**: Members shared moments of nostalgia, reminiscing about earlier access to **GPT-3 and Codex**, indicating a continuing interest and appreciation for previous models.
- **Wondering About Voice Chat Rooms**: One member inquired about the lack of **voice chat rooms** within the Discord, and it was clarified that such features are absent due to the challenges of moderation.
- **Confusion Over Chatbot Memory Integration**: A user queried about the possibility of integrating the new **memory feature** with their chatbot in the API, seeking guidance on implementation.
- **Inquiries About GPT-4's Response Times**: Members discussed that **GPT-4** appears to be roughly two times slower than its predecessor **GPT-3.5**, with recent reports of unusual latency and **gpt4 turbo** being 5-10 times slower than usual.
- **ChatGPT Access Issues and Rate Limits**: Users reported issues accessing **GPT**, reaching out for help, and questioning the message rate limits. Suggestions for checking **OpenAI's service status** and experiences of unexpected timeouts were mentioned, indicating a fluctuating rationing system potentially due to high demand.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1235533289075376148)** (3 messages): 

- **Retrieval Challenges in Large Language Models**: A member pointed out that mitigating retrieval issues in large language models (LLMs) isn't possible in the way one might hope. They referred to a search term "LLM Retrieval Needle In A Hay Stack" for more in-depth understanding and emphasized that **foundation model's retrieval limits** can't be bypassed with algorithms.
- **Python Tool for Word Occurrence**: In the context of handling large texts, it was mentioned that there is a **Python** solution capable of counting unique occurrences of words. This technique could potentially be useful for data analysis and preprocessing tasks.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1235533289075376148)** (3 messages): 

- **Limits of LLM Retrieval Addressed**: A member mentioned a search term "LLM Retrieval Needle In A Hay Stack" to indicate that it's not possible to overcome the *foundation model's retrieval limits* with any algorithm.
- **Python Script for Word Counting Shared**: Another message pointed out the availability of Python solutions for counting unique word occurrences in large texts.
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1235906896414769202)** (2 messages): 

- **Community Highlights Sparkle with Innovations**: New user contributions shine with the unveiling of [Moondream 2](https://huggingface.co/spaces/Csplk/moondream2-batch-processing) for batch processing, [FluentlyXL v4](https://huggingface.co/spaces/fluently/Fluently-Playground), the Portuguese translation of HF Audio course's [Chapter 0 + 1](https://github.com/huggingface/audio-transformers-course/pull/182), [BLIP finetune](https://huggingface.co/spaces/unography/image-captioning-with-longcap) for extended captions, and [what appears to be a list](https://iatalk.ing/destaques-comunidade-hugging-face/) of community highlights in Portuguese.

- **BLOOM Chat Speaks Multilingually**: A new [multilingual chat](https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat) supports conversations in 55 languages, while an [Inpainting sketch pad](https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad) unlocks creativity, and a task from HF's alignment handbook can now be [run in the cloud](https://twitter.com/dstackai/status/1785315721578459402).


- **Hot Off the Press: Cool AI Developments**: AI enthusiasts get treated to a [guide on protein optimization](https://huggingface.co/blog/AmelieSchreiber/protein-optimization-and-design) with AI, a model NorskGPT-Mistral-7B, the basics of implementing a vision language model [from scratch](https://huggingface.co/blog/AviSoori1x/seemore-vision-language-model), and insights into [Google Search with LLMs](https://huggingface.co/blog/nand-tmp/google-search-with-llm). Enthusiasts can also explore [Token Merging for LLMs](https://huggingface.co/blog/samchain/token-merging-fast-inference), and expand their knowledge on Model Context and Chat Models in [this blog post](https://huggingface.co/blog/maywell/llm-feature-transfer).

- **HF Dives Deep Into Model Interpretability**: New insights into the interpretability plus an in-depth [analysis of LLMs](https://huggingface.co/posts/gsarti/644129530281733) is now available for AI enthusiasts to learn from.


- **AutoTrain Now Open to All Through Configs**: Demonstrating the potential of AutoTrain, users can now train models with YAML config files available in the [autotrain-advanced GitHub repo](https://github.com/huggingface/autotrain-advanced), and are encouraged to contribute by creating a pull request. The ease of use makes it possible for individuals with minimal machine learning knowledge to train state-of-the-art models without code, as announced on [Twitter](https://twitter.com/abhi1thakur/status/1786368641388179797).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.</li><li><a href="https://iatalk.ing/destaques-comunidade-hugging-face/)">🤗 Destaques da Comunidade</a>: O Destaques da Comunidade é um post contendo uma lista publicada periodicamente no Discord do Huggging Face contendo uma série de projetos, modelos, spaces, posts, artigos feitos pela comunidade de…
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1235534447856844871)** (163 messages🔥🔥): 

- **Voice Synthesis Models Discussed**: Members exchanged recommendations for voice synthesis models, such as **Xtts v2** and **Voice Craft**, mentioning their performance and unique features like speech editing. Links to demos were shared for [Xtts](https://huggingface.co/spaces/coqui/xtts) and [Voice Craft](https://replicate.com/cjwbw/voicecraft), with a member noting **Voice Craft**'s capabilities in zero-shot text-to-speech.
- **Model Conversion and Fine-Tuning Challenges**: Challenges were discussed around converting transformer models to smaller formats, with specific issues mentioned like a model being larger than 2GB and causing errors. Strategies were also discussed for fine-tuning smaller datasets, considering the effectiveness of **RAG (Retrieval-Augmented Generation)** as an alternative to fine-tuning with limited data.
- **Using LLM Models and Hosting**: Questions were raised about deploying large language models (LLMs) in production, with **Vllm** and **TGI** suggested as potential frameworks for running LLMs in production environments. The availability and usage of **Llama3** were discussed, with recommendations to try services like **Groq** for free API access.
- **Bot and Parquet Converter Request**: Users expressed the need for a parquet converter-bot for dataset conversion and inquired about the status of a "Dev mode," suggesting the possibility of maintenance or downtime.
- **Prompt Refinement and Evaluation Inquiry**: A user inquired about metrics for evaluating the quality of refined prompts, looking for specific metrics tailored to prompt assessment, without follow-up discussions offering specific solutions or metrics.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/coqui/xtts">XTTS - a Hugging Face Space by coqui</a>: no description found</li><li><a href="https://www.llama2.ai/">Chat with Meta Llama 3 on Replicate</a>: Llama 3 is the latest language model from Meta.</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>: no description found</li><li><a href="https://tacosconference.github.io/">TaCoS</a>: TaCoS Conference in Saarbrücken</li><li><a href="https://huggingface.co/DioulaD/falcon-7b-instruct-qlora-ge-dq-v2">DioulaD/falcon-7b-instruct-qlora-ge-dq-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-Gradient-1048k">crusoeai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://replicate.com/cjwbw/voicecraft">cjwbw/voicecraft – Run with an API on Replicate</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1235584374041219092)** (9 messages🔥): 

- **Quest for Query Refinement**: A member is seeking assistance to rephrase a follow-up question (q2) in the pharma domain by incorporating all the details from an initial query (q1).
- **Ray Deployment Inquiry Remains Open**: A user asked for help with deploying HuggingFace models on Ray, indicating a shared interest among community members.
- **Training Loop Customization Debate**: A comment was made advocating for writing custom training loops, suggesting that modifying examples from diffusers allows for more flexibility in training AI models.
- **A Shortcut through the Neural Nets**: An interesting [discussion on Kolmogorov-Arnold Networks (KANs)](https://arxiv.org/abs/2404.19756v1) took place, highlighting their promising attributes, such as requiring smaller computational graphs compared to Multi-Layer Perceptrons (MLPs).
- **Fine-Tuning Explained**:
    - A member shared a [YouTube video](https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s) offering a high-level overview of fine-tuning AI models.
    - They also linked to a HuggingFace [technical guide](https://huggingface.co/docs/transformers/training) on fine-tuning with Transformers and Keras.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.19756v1">KAN: Kolmogorov-Arnold Networks</a>: Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation fun...</li><li><a href="https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s">What is Fine Tuning? In Two Minutes.</a>: A high-level overview of what fine-tuning a genAI model is in two minutes. TL;DR: Tuning a genAI model is like tuning a guitar. Technical overview from  @Hug...</li><li><a href="https://huggingface.co/docs/transformers/training">Fine-tune a pretrained model</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1235577209272668220)** (7 messages): 

- **New MPI Code Repository Unveiled**: A link to a GitHub repository called MPI-Codes by Binary-Beast03 was shared, which is aimed at contributing to the development of MPI Codes. Information about it can be accessed at the [MPI-Codes GitHub repository](https://github.com/Binary-Beast03/MPI-Codes).

- **RAG Boosts LangChain's Email Savvy**: LangChain's LangGraph Agents have been enhanced with Retrieval-Augmented Generation (RAG) to improve intelligent email drafting, with details shared in a [Medium post](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da). However, the content is behind a member-only access wall as noted by a follow-up comment. 

- **FinBERT Fine-Tuned for Financial Sentiment**: ProsusAI's FinBERT, a BERT-based NLP model, is specifically trained for sentiment analysis in the financial domain and shared with the [HuggingFace link](https://huggingface.co/ProsusAI/finbert). It is fine-tuned on the Financial PhraseBank and detailed in both an [academic paper](https://arxiv.org/abs/1908.10063) and a companion [blog post](https://medium.com/prosus-ai-tech-blog/finbert-financial-sentiment-analysis-with-bert-b277a3607101).

- **Explaining Retrieval-Augmented Generation (RAG)**: An informative [Databricks page](https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag) was shared, covering how RAG addresses the issues of LLMs not adapting to custom data and the necessity for AI applications to leverage such data for effective results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ProsusAI/finbert">ProsusAI/finbert · Hugging Face</a>: no description found</li><li><a href="https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag">Che cos&#x27;è la Retrieval Augmented Generation (RAG)? | Databricks</a>: La RAG (Retrieval Augmented Generation) è un approccio architettonico che utilizza i dati come contesto per i modelli linguistici di grandi dimensioni (LLM) in modo da migliorare la pertinenza dell&#x...</li><li><a href="https://github.com/Binary-Beast03/MPI-Codes">GitHub - Binary-Beast03/MPI-Codes</a>: Contribute to Binary-Beast03/MPI-Codes development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1235594167606837348)** (9 messages🔥): 

- **Typo Alert in Model Card**: A small typo was pointed out in the model card's title for **Fluently XL V4**—it should be "Fluently" instead of "Fluenlty."
- **Fluently-XL-v4 Showing Off New Colors and Digits**: An image generated by **Fluently-XL-v4** on a local NVIDIA RTX 3070 mobile boast impressive results, evidenced in an [Instagram post](https://www.instagram.com/p/C6eMZaTr03q/?igsh=MWQ1ZGUxMzBkMA==), with well-handled colors and correct numbers of fingers, outperforming several other models.
- **Hugging Face Audio Course Receives Brazilian Translation**: Chapters 0 and 1 of the Hugging Face audio course have been translated into Portuguese and a PR is open for review [here](https://github.com/huggingface/audio-transformers-course/pull/182), with a call for help from Brazilian community members for revisions.
- **Introducing LongCap for Image Captioning**: A finetuned version of the [BLIP model](https://huggingface.co/unography/blip-long-cap) for long captions is shared, which promises to generate detailed image descriptions suitable for prompts in text-to-image generation. A request for assistance in evaluating this model against Google's DOCCI is open, with a [Colab notebook](https://colab.research.google.com/drive/1UfS-oa6Ou0mguZG0udXzyE8IsL_n8ZNf?usp=sharing) provided for testing.
- **Archiving Community Highlights in Portuguese**: A new page created by a community member compiles all posts and links from the Hugging Face Community Highlights since edition #52, with plans to catch up on previous editions and establish a comprehensive database of AI-related content in Portuguese [link here](https://iatalk.ing/destaques-comunidade-hugging-face/).
- **Sythetic Data Generator for LLMs Now on PyPI**: A tool for generating and normalizing synthetic data for training large language models has been released and is available on [PyPI](https://github.com/tobiadefami/fuxion), potentially aiding fine-tuning efforts for different project use cases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/fishaudio/fish-speech-1">Fish Speech 1 - a Hugging Face Space by fishaudio</a>: no description found</li><li><a href="https://huggingface.co/kimou605/shadow-clown-BioMistral-7B-DARE">kimou605/shadow-clown-BioMistral-7B-DARE · Hugging Face</a>: no description found</li><li><a href="https://www.instagram.com/p/C6eMZaTr03q/?igsh=MWQ1ZGUxMzBkMA==">Mansion X on Instagram: &quot;Speaks American &#x1f1fa;&#x1f1f8; *fluently*. #fit #ootd&quot;</a>: 2 likes, 0 comments - the_mansion_x on May 2, 2024: &quot;Speaks American &#x1f1fa;&#x1f1f8; *fluently*. #fit #ootd&quot;. </li><li><a href="https://huggingface.co/unography/blip-long-cap">unography/blip-long-cap · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/unography/image-captioning-with-longcap">Image Captioning with LongCap - a Hugging Face Space by unography</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1UfS-oa6Ou0mguZG0udXzyE8IsL_n8ZNf?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/tobiadefami/fuxion">GitHub - Tobiadefami/fuxion: Sythetic data generation and normalization functions</a>: Sythetic data generation and normalization functions - Tobiadefami/fuxion</li><li><a href="https://iatalk.ing/destaques-comunidade-hugging-face/">🤗 Destaques da Comunidade</a>: O Destaques da Comunidade é um post contendo uma lista publicada periodicamente no Discord do Huggging Face contendo uma série de projetos, modelos, spaces, posts, artigos feitos pela comunidade de…
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1235723560199131146)** (6 messages): 

- **Curated Collection for LLM Improvement**: A member shared their research on improving Large Language Models (LLMs) with a [curated list](https://huggingface.co/collections/f0ster/smarter-llms-research-6633156999b1fa10612309dd) on HuggingFace, inviting thoughts and feedback.
- **Spotlight on React Agents**: Another participant highlighted the significance of React agents in elevating LLM output quality, noting the abundance of papers in the field and the challenges of selecting a focus.
- **Unindexed Findings in LLM Research**: The curator of the LLM improvement collection expressed excitement about sharing papers that had not been indexed, received upvotes, or associated code in their research compilation.
- **Exploring Reasoning and Acting in LLMs**: The curator drew attention to a paper titled 'ReAct' that proposes a method for combining reasoning traces and task-specific actions in LLMs for enhanced performance and interpretability. The abstract discusses how interleaving both aspects can improve interface with external information sources and handle exceptions ([view the paper](https://arxiv.org/abs/2210.03629)).
- **Graph ML Meets LLMs**: A member shared preliminary notes for a presentation that looks into the intersection of graph machine learning and LLMs, with an observation that the topic is more extensively explored than initially thought. They shared a [medium post](https://isamu-website.medium.com/understanding-graph-machine-learning-in-the-era-of-large-language-models-llms-dce2fd3f3af4) summarizing the subject.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/f0ster/smarter-llms-research-6633156999b1fa10612309dd">Smarter LLMs Research - a f0ster Collection</a>: no description found</li><li><a href="https://arxiv.org/abs/2210.03629">ReAct: Synergizing Reasoning and Acting in Language Models</a>: While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-though...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1235516322641870909)** (7 messages): 

- **Channel Quest**: A member inquired about the existence of a **#cv-study-group** channel, which they could not find despite its mention in the [Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome) page.
- **Fine-Tuning Strategies Shared**: A suggestion was made to fine-tune only the classifier weights of a pre-training model for efficiency, and to consider training a model end-to-end using a shallow CNN to rescale images before connecting to **Yolov4**.
- **Study Group Status Clarifications**: There seems to be some confusion among members regarding the presence of a **study group**; one member clarified there's no specific study group, while another mentioned that while there is no reading group, someone might know of past study groups.
- **Agreement on Non-Existence of Study Groups**: Members agreed that there is no particular **reading or study group** currently active in the channel.

**Link mentioned**: <a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>: no description found

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1235581472619106315)** (5 messages): 

- **RARR - A Solution for Model Attribution**: A member shared the [RARR paper](https://huggingface.co/papers/2210.08726), which presents a system for **Retrofit Attribution using Research and Revision**. RARR aims to automatically find and add attributions to the outputs of text generation models, and make corrections to unsupported content.

- **Zero-shot Classification Confusion**: A user reported an issue with a **zero-shot classification model** generating disproportionate results, with labels "gun" and "art" leading to almost split probabilities, questioning the model's behavior against expected results. This could be indicative of a misunderstanding on how the classifier analyzes text unrelated to the labels provided.

**Link mentioned**: <a href="https://huggingface.co/papers/2210.08726">Paper page - RARR: Researching and Revising What Language Models Say, Using Language
  Models</a>: no description found

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1235938948904128554)** (12 messages🔥): 

- **Clarification on Auto-Train Configs**: A member clarified that specifying `xl: true` in auto train configs is optional because the model type can be determined automatically, but it can also be explicitly declared in the configuration.
  
- **Merging Diffusion Pipelines Technique**: One member inquired about the possibility of using two different **StableDiffusionPipelines** for partial denoising, switching at a midpoint in the process. Another member provided information on an approach called *partial diffusion via mixture of experts*, linking to an outstanding pull request for **SD 1.5** on the [diffusers GitHub repository](https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2).

- **Seeking Examples for Partial Diffusion**: A member requested examples of partial diffusion with **StableDiffusionPipelines**. They were directed to a GitHub comparison page that showcases the implementation of the method.

- **Availability of Partial Diffusion for Testing**: The same member considered testing the partial diffusion method mentioned in the pull request to determine its suitability for their own test suite, noting a preference for faster inference times.

**Link mentioned**: <a href="https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2">Comparing huggingface:main...bghira:partial-diffusion-2 · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - Comparing huggingface:main...bghira:partial-diffusion-2 · huggingface/diffusers

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1235600910445445233)** (5 messages): 

- **Build an Optimized RAG Data Stack**: A new tutorial has been shared, featuring a comprehensive guide on constructing an efficient **data stack** for an agentic RAG support bot. It highlights the importance of various data components besides the vector database and is documented by [@tchutch94](https://twitter.com/tchutch94) and [@seldo](https://twitter.com/seldo). Check out the full post [here](https://t.co/jez7g9hADV).
  
- **Step-By-Step RAG Pipeline Guide**: Plaban Nayak introduces an open-source RAG pipeline guide using **Llama 3** from Meta, @qdrant_engine, and **ms-marco-MiniLM-L-2-v2**. This guide emphasizes improving RAG application precision with a reranker process. Read more about the guide [here](https://t.co/wXxFCsrkSa).

- **Natural Language Filters for Airbnb Listings**: Harshad Suryawanshi provides a walkthrough for creating a RAG application to filter **@Airbnb listings** with natural language, utilizing @MistralAI's Mixtral 8x7b tools. A detailed explanation and repository can be found [here](https://t.co/iw6iBzGKl6).

- **LlamaIndex 10.34 Release Features Introspective Agents**: An announcement for the new **LlamaIndex 10.34 release** was made, highlighting features such as introspective agents that utilize reflection for iterative responses. The notebook contains implementations but warns of potentially sensitive content. Read about these agents and the warning [here](https://t.co/X8tJGXkcPM).

- **Launch of LlamaIndex 0.10.34 with Huggingface Support**: The release of **LlamaIndex 0.10.34** introduced introspective agents and mentioned upcoming support for huggingface integration. They promise to discuss all new updates separately in the days to follow. Catch the details [here](https://t.co/UrD0c7BRAO).

**Link mentioned**: <a href="https://t.co/X8tJGXkcPM">Introspective Agents: Performing Tasks With Reflection - LlamaIndex</a>: no description found

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1235517710474149968)** (140 messages🔥🔥): 

- **Seeking a Financial Analysis Application**: A member is creating an application to generate financial summaries from pandas dataframes containing company income statements. They seek guidance on using the [Pandas Query Engine](https://docs.llamaindex.ai/en/latest/), given the brief examples in the documentation.
- **Customizing MongoDB with LlamaIndex**: A user seeks help on querying directly from MongoDB embeddings with metadata, bypassing document or node submissions to LlamaIndex's query engine. They shared a tutorial [link](https://colab.research.google.com/drive/136MSwepvFgEceAs9GN9RzXGGSwOk5pmr?usp=sharing) they previously used and requested alternatives for MongoDB's `collections.aggregate`.
- **LLamacpp Parallel Request Deadlock**: One member reported a deadlock when running two concurrent queries with the llamacpp gguf model. They inquired about enabling parallel request serving without using a server setup to mitigate the issue.
- **Inquiry about Setting Up Trulens with Llama Index**: A user asked for assistance on using Trulens with MongoDB and Llama Index, pointing out that they already have embeddings and metadata uploaded. They shared relevant links from the [docs](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=true#truera-trulens) and suggested considering alternative tools like Arize and Langfuse.
- **Memory Load Issues with LlamaIndex**: A member experienced memory overload issues when running LlamaIndex, with an 8GB model sometimes exceeding 20GB and then reverting to CPU, causing slowdowns. They identified a specific command execution in their code which appeared to spam memory, and mentioned the necessity of waiting for memory cleanup.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/136MSwepvFgEceAs9GN9RzXGGSwOk5pmr?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://llama.meta.com/docs/how-to-guides/prompting">Prompting | How-to guides</a>: Prompt engineering is a technique used in natural language processing (NLP) to improve the performance of the language model by providing them with more context and information about the task in hand.</li><li><a href="https://www.llamaindex.ai/contact">Talk to us — LlamaIndex, Data Framework for LLM Applications</a>: If you have any questions about LlamaIndex please contact us and we will schedule a call as soon as possible.</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=true#truera-trulens">Observability (Legacy) - LlamaIndex</a>: no description found</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Meta Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followe...</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi#rag-approach-to-import-external-knowledge-into-llm-as-context>).">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/fine-tuning/fine-tuning#finetuning-embeddings>).">Fine-Tuning - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/postgres#llama_index.vector_stores.postgres.PGVectorStore>).">Postgres - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/supabase#llama_index.vector_stores.supabase.SupabaseVectorStore>).">Supabase - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/13196">Call Cohere RAG inference with `documents` argument by co-antwan · Pull Request #13196 · run-llama/llama_index</a>: Description Adds support for Cohere.chat&#39;s documents argument when using in RAG pipelines. This ensures proper formatting on Cohere&#39;s client side, and leads to better downstream performance. T...
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1235711461784027206)** (1 messages): 

- **Impressive but Costly RAG Performance with OpenAI API**: A user shared their positive experience using **OpenAI's assistants API v2** for *Retrieval-Augmented Generation (RAG)*, noting effective answers derived from a testing knowledge base of 500 Wikipedia articles. However, they highlighted the cost concern as a short conversation racked up **$1.50 in charges**.
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1235542394129088552)** (25 messages🔥): 

- **Seeking Gemini 1.5-compatible tools**: A member inquired about tools like Cursor/Aider for **Gemini 1.5** full context window but mentioned disappointment with Gemini 1.5 benchmarking, preferring to use Opus or long context that Cursor now supports.
- **Code Interpreter SDK Takes Twitter Stage**: Member @mlejva announced the launch of their Code Interpreter SDK on Twitter and solicited community support with a [link to their tweet](https://twitter.com/mlejva/status/1786054033721139319).
- **Prompt Labeling Practices in Question**: A user queried the community about best practices for labeling output variables in prompts, particularly with **Claude**, referencing a [Matt Shumer's tweet](https://twitter.com/mattshumer_/status/1773385952699789808).
- **OpenAI Assistants API Quickstart Shared**: The OpenAI Assistants API Quickstart was highlighted, featuring integration with Next.js and offering streaming chat interfaces, function calling, and a code interpreter; linked [tweet here](https://x.com/openaidevs/status/1785807183864820209?s=46&t=90xQ8sGy63D2OtiaoGJuww) and [GitHub repo](https://github.com/openai/openai-assistants-quickstart).
- **SQLite Gets a New Vector Search Extension**: There's a successor to `sqlite-vss` called `sqlite-vec`, being developed for better vector search within SQLite, shared with a [creator's blog post](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html">I'm writing a new vector search SQLite Extension</a>: sqlite-vec is an new vector search SQLite extension, coming soon!</li><li><a href="https://x.com/openaidevs/status/1785807183864820209?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from OpenAI Developers (@OpenAIDevs)</a>: We’ve open sourced a new quickstart to help you build with the Assistants API and @nextjs.  It comes with sample code for creating a chat interface with streaming, and using tools like function callin...</li><li><a href="https://x.com/emilylshepherd/status/1786037498507853852?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Emily (She/Her) (@EmilyLShepherd)</a>: Let&#39;s have a chat about @rabbit_hmi. A 🧵  Formed in 2021, they were originally called Cyber Manufacture Co, and they were a &#34;creative studio building next-generation experiences at the inters...</li><li><a href="https://x.com/emilylshepherd/status/1786037498507853852?">Tweet from Emily (She/Her) (@EmilyLShepherd)</a>: Let&#39;s have a chat about @rabbit_hmi. A 🧵  Formed in 2021, they were originally called Cyber Manufacture Co, and they were a &#34;creative studio building next-generation experiences at the inters...</li><li><a href="https://x.com/teknium1/status/1786485060314521627?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Teknium (e/λ) (@Teknium1)</a>: At least it&#39;s confirmed that no, it&#39;s not a &#34;Large Action Model&#34; - it&#39;s an LLM.. lol</li><li><a href="https://www.echoai.com/">Conversation Intelligence - Echo AI</a>: Your customer conversations are the most valuable data you have. Echo AI is the first AI-native Conversation Intelligence platform that turns every word your customers say into the insights and action...</li><li><a href="https://www.assorthealth.com/">Assort Health | The First Generative AI Call Center Built for Healthcare</a>: Our call center generative AI reduces hold time, decreases dropped calls, and contains costs while driving appointment revenue.</li><li><a href="https://www.trychroma.com/">the AI-native open-source embedding database</a>: the AI-native open-source embedding database</li><li><a href="https://www.getlifestory.com/">Life Story</a>: Capture life, one story at a time. 
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1235575449674055818)** (33 messages🔥): 

- **Mamba Deep Dive Kicks Off**: The **llm-paper-club-west** channel members geared up for a discussion on Mamba with a Notion link shared for a deep dive into the topic: [A Mamba Deep Dive](https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f).
- **Debating Mamba's Selective Recall Capability**: A member raised a question about whether *selective copying* in Mamba is akin to a recall test for previously seen tokens, initiating a discussion on the mechanism's specificity.
- **Technical Difficulties Spur Platform Switch Proposal**: Users faced with delays and technical hiccups during a Mamba discussion agreed on switching to Zoom for future meetings to ensure a smoother experience.
- **Exploring the Sensitivity of Mamba in Fine-tuning**: The conversation turned towards how the Mamba architecture fares during fine-tuning and its susceptibility to overfitting in comparison to traditional transformers.
- **State Space Models and Induction Heads**: There was a brief exchange about whether state space models, specifically in the context of Mamba, could approximate induction heads found in attention layers. Two arXiv papers were shared for further reading: [Arxiv State Space Models](https://arxiv.org/pdf/2404.15758) and [Arxiv Multi Token Paper](https://arxiv.org/pdf/2404.19737).


**Link mentioned**: <a href="https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1236044647785304109)** (65 messages🔥🔥): 

- **Fascination with Suno's Audio Generation**: A member expressed curiosity about the music generation capabilities of **suno**, wondering if it creates music tracks from scratch. Another member mentioned *Suno's focus* on audio tokenizing as their "secret sauce."
- **Exploring Different Model Architectures**: Discussion around **musicgen** architecture revealed it being part of a buddy's adventure into finetuning audio models for multimodal applications. Members also touched upon **imagebind** as an example of multimodal embedding space.
- **Understanding Harmonic Distortion**: In a brief conversation about 'harmonic distortion', a member described it as incorrect weightings on harmonic tones, which could result in improper frequency ratios or beats. Reference was made to a blog discussing the *snake activation function* and its potential for reducing harmonic distortion.
- **Generating Audio with Latent Diffusion & Autoencoders**: Inquiry into the process of **stable audio 2.0** led to a discussion about how audio files are processed through autoencoders to create tokens, with suggestions that entire audio files are compressed for the model.
- **Commercial Viability of Generated Audio**: A member inquired about the licensing and commercial use of outputs from **Stable Audio 2.0**, indicating an interest in the legalities around generated content. There was also mention of potential applications, like separating and replacing audio channels.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notesbylex.com/snake-activation-function">Tweet from Snake Activation Function</a>: Snake is a neural network activation function useful for modelling problems with a &quot;periodic induction bias&quot; - in other words, problems with regular, repeating patterns - for...</li><li><a href="https://arxiv.org/abs/2404.10301v1">Long-form music generation with latent diffusion</a>: Audio-based generative models for music have seen great strides recently, but so far have not managed to produce full-length music tracks with coherent musical structure. We show that by training a ge...</li><li><a href="https://github.com/betweentwomidnights/gary4live">GitHub - betweentwomidnights/gary4live: This is gary. python continuations plus continuations inside of ableton. It is a WIP by a newb.</a>: This is gary. python continuations plus continuations inside of ableton. It is a WIP by a newb. - betweentwomidnights/gary4live
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1235710627415195688)** (49 messages🔥): 

- **Intriguing Trends in Multilingual LLMs**: An ongoing discussion revolves around how to enhance multilingual capabilities in Large Language Models (LLMs), with references made to research papers such as ["Understanding Language Models by Fine-grained Language Identification"](https://arxiv.org/abs/2402.10588) and work exploring LLMs' processing of multilingual inputs. The framework depicted suggests that in initial layers, LLMs convert multilingual inputs into English before generating responses in the original query's language.

- **Reflecting on the Journey of ML Domains**: Users reminisced about abandoned or overshadowed research areas in machine learning, including adversarial robustness, automated architecture, and domain-specific model training. There's a feeling of nostalgia and a hint of regret for the paths not taken, exacerbated by the fact that job pull by major tech companies has deprioritized these areas.

- **The Changing Landscape of AI Funding and Impact**: Users discussed the enormous investments in AI companies and the potential over-saturation leading to diminishing returns on investment. Concerns are raised about how this affects the efficiency of scaling up models and the future direction of AI research.

- **LLMs and System Hierarchy Vulnerabilities**: There's an intricate conversation on how models handle instruction hierarchies and the vulnerabilities that arise when a system prompt is not considered different from a user prompt—a risk particularly important in preventing prompt injections or other attacks on LLMs. A linked paper, ["Improving Robustness to Prompt Injections with Instruction Hierarchy"](https://arxiv.org/abs/2404.13208), suggests reinforcing instruction hierarchies could mitigate these vulnerabilities.

- **Job Search for an ML Engineer**: A member is reaching out to the community seeking opportunities for a machine learning engineer position outside the US. This person has experience with LLMs, and relevant work includes leading the Polyglot team and contributing to the OSLO project within EleutherAI; they shared personal links to LinkedIn, Google Scholar, GitHub, and an email address for potential contact.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.18815v1">How do Large Language Models Handle Multilingualism?</a>: Large language models (LLMs) demonstrate remarkable performance across a spectrum of languages. In this work, we delve into the question: How do LLMs handle multilingualism? We introduce a framework t...</li><li><a href="http://arxiv.org/abs/2404.13208">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a>: Today&#39;s LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model&#39;s original instructions with their own malicious prompts. In this w...</li><li><a href="https://github.com/jason9693">jason9693 - Overview</a>: AI Research Engineer. jason9693 has 71 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1235487352873418793)** (54 messages🔥): 

- **Exploring the Depths of Dataset Contamination**: Amidst discussions about instruction-finetuning and benchmark effectiveness, participants shared concerns over benchmark dataset leakage in large language models (LLMs), emphasizing the difficulty in measuring leaked information and the cycle of detecting and addressing leaks. Two recent papers on benchmark dataset leakage were highlighted: one focused on [detecting data leakages](http://arxiv.org/abs/2404.18824) and the other discussing [fresh benchmark questions](https://arxiv.org/abs/2405.00332) as a solution to prevent unfair comparisons.
  
- **Chatbot Conversations as a Learning Tool**: The idea of using chatbot conversations, particularly those with multiple multiturn interactions with the same user, was contemplated with the notion of utilizing sentiment analysis or user retention (churn) to improve an LLM. Participants were curious about how these could lead to a self-improvement loop within the model, with one member pointing out a [paper focusing on indirect preference](https://arxiv.org/abs/2404.15269) and another reference to the [WildChat dataset for chatbot research](http://arxiv.org/abs/2405.01470).
  
- **Chess Mastery without Heuristics**: A study using a transformer model trained on a dataset of 10 million chess games was brought up, demonstrating the model's high performance in chess without domain-specific enhancements or explicit search algorithms. The [DeepMind paper](https://arxiv.org/abs/2402.04494) indicates that training models at scale can lead to competitive levels of play without the approaches traditional chess engines use.

- **Serendipitous Time Travels and Future Returns**: A humorous exchange took place where a participant jokingly claimed to have built a time machine and returned from the future, leading to playful interactions about their presence in the 'ot' (off-topic) channel and the utilization of a time machine.

- **Gwern's Peripheral Return**: In a meta-discussion, participants noted gwern1782's selective responses to past mentions after a period of absence from the server, with mentions of using Discord's search feature to filter through the multitude of notifications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.04494#deepmind">Grandmaster-Level Chess Without Search</a>: The recent breakthrough successes in machine learning are mainly attributed to scale: namely large-scale attention-based architectures and datasets of unprecedented scale. This paper investigates the ...</li><li><a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: Amid the expanding use of pre-training data, the phenomenon of benchmark dataset leakage has become increasingly prominent, exacerbated by opaque training processes and the often undisclosed inclusion...</li><li><a href="http://arxiv.org/abs/2405.01470">WildChat: 1M ChatGPT Interaction Logs in the Wild</a>: Chatbots such as GPT-4 and ChatGPT are now serving millions of users. Despite their widespread use, there remains a lack of public datasets showcasing how these tools are used by a population of users...</li><li><a href="https://arxiv.org/abs/2404.15269">Aligning LLM Agents by Learning Latent Preference from User Edits</a>: We study interactive learning of language agents based on user edits made to the agent&#39;s output. In a typical setting such as writing assistants, the user interacts with a language agent to genera...</li><li><a href="https://arxiv.org/abs/2405.00332">A Careful Examination of Large Language Model Performance on Grade School Arithmetic</a>: Large language models (LLMs) have achieved impressive success on many benchmarks for mathematical reasoning. However, there is growing concern that some of this performance actually reflects dataset c...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1235971685035933758)** (1 messages): 

- **Optimistic Prediction on Math Problem Solving**: The NARRATOR mentioned a prediction that was surpassed, with current Math Word Problem Solving performance at **over 70% within 2 years** turning out to be too pessimistic. This benchmark can be explored in detail at [Papers With Code](https://paperswithcode.com/sota/math-word-problem-solving-on-math), which is a free resource with data licensed under [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). Contact them via [hello@paperswithcode.com](mailto:hello@paperswithcode.com).

**Link mentioned**: <a href="https://paperswithcode.com/sota/math-word-problem-solving-on-math">Papers with Code - MATH Benchmark (Math Word Problem Solving)</a>: The current state-of-the-art on MATH is GPT-4-code model (CSV, w/ code, SC, k=16). See a full comparison of 109 papers with code.

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1235593243853590538)** (8 messages🔥): 

- **Position Paper Accepted**: A recently submitted position paper by authors including **Vincent Conitzer**, **Rachel Freedman**, and **Stuart Russell** among others, has been [accepted as a position paper](https://arxiv.org/abs/2404.10271).
- **Mechanistic Interpretability Workshop at ICML 2024**: **Neel Nanda** announces the first academic *Mechanistic Interpretability workshop* at [ICML 2024](https://icml2024mi.pages.dev/), with a call for papers. The event features $1750 in best paper prizes, and submissions can include a variety of formats, with a deadline of May 29th.
- **Star-Studded Panel Discussion Revealed**: **Neel Nanda** confirms that **Naomi** and **StellaAthena** will be part of a panel at the Mechanistic Interpretability workshop, with updates and additions to the event's website pending.
- **Comprehensive Primer on Transformer-Based Language Models**: **javifer96** highlights the release of a primer on Transformer-based language models, encompassing model components and interpretability methods in a unified notation. Interested parties can find the announcement and more information [here](https://twitter.com/javifer_96/status/1786317169979970046).
- **Cross-Model Generalization Using English as Pivot Language**: **Butanium** shares that the team has replicated results across **llama** models, indicating that using English as a pivot language generalizes well across these models. Further details can be found in their latest [tweet](https://twitter.com/Butanium_/status/1786394217478004950).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://icml2024mi.pages.dev/">ICML 2024 Mechanistic Interpretability Workshop</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.10271">Social Choice for AI Alignment: Dealing with Diverse Human Feedback</a>: Foundation models such as GPT-4 are fine-tuned to avoid unsafe or otherwise problematic behavior, so that, for example, they refuse to comply with requests for help with committing crimes or with prod...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1235644356464082984)** (2 messages): 

- **Inquiry on MT-Bench Inclusion**: A member asked about the status of incorporating **MT-Bench** or similar benchmarks into lm-evaluation-harness and if there are any upcoming conversational AI quality benchmarks.

- **Prometheus 2 As a Potential Improvement**: Another member highlighted **Prometheus 2**, an open-source evaluator LM, suggesting it as a beneficial addition to lm-evaluation-harness. Prometheus 2 is designed to mirror human and GPT-4 judgements and supports various forms of assessment, as noted in the [research abstract on Hugging Face](https://huggingface.co/papers/2405.01535).

**Link mentioned**: <a href="https://huggingface.co/papers/2405.01535">Paper page - Prometheus 2: An Open Source Language Model Specialized in Evaluating
  Other Language Models</a>: no description found

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1235586380994711673)** (40 messages🔥): 

- **LLama-3 8B's Context Length Breakthrough**: Gradient AI announced the extension of LLama-3 8B's context length from 8k to over 1040k with help from [Crusoe Energy's compute](https://huggingface.co/crusoeai). The achievement showcases that state-of-the-art large language models (LLMs) can handle long contexts through minimal training using an adjusted RoPE theta.
  
- **Conceptualizing Ring Attention**: A member discussed trying to grasp the concept of ring attention, using visualization to aid understanding, despite some skepticism from others about the technical accuracy of the approach.

- **Collision with ChatML Training**: One user reported problems when training with ChatML, facing an `AttributeError` associated with `SeparatorStyle.GEMMA`. The troubleshooting included suggestions like removing conflicting arguments and upgrading fastchat, aiming to resolve training issues with ChatML-configured datasets.

- **Injecting Context in Prompts**: In a discussion about fine-tuning prompt design, members exchanged insights on how to include ChatML turns within system prompts for model training and inferred that when context is injected into prompts, ChatML tokens are tokenized correctly without escaping.

- **Hermes 2 Accelerated with llama.cpp**: A member expressed admiration for the inference speed of Hermes 2 Pro Llama 3 8B on an Android device with 8GB RAM. This was attributed to an upgrade from llama.cpp which reportedly increased inference speed by 30%.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://x.com/granawkins/status/1786428318478168447">Tweet from Grant♟️ (@granawkins)</a>: sota RAG in 2024
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1235492757917929553)** (19 messages🔥): 

- **PR Merged for Performance Improvement**: A pull request was merged to fix an issue where the orpo trainer was using only one worker for preprocessing. This enhancement is aimed at speeding up the data preprocessing steps in TRL trainer and possibly others like DPOTrainerArgs. The patch is available at [GitHub PR #1583](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583).

- **Parametrization Affects Multiple Configs**: Clarifications in the conversation indicate that the `dataset_num_proc` parameter in question not only affects the TRL trainer but also DPO, SFT, CPO, KTO, and ORPO configurations within the codebase.

- **Minimum Python Version Established for Axolotl**: It was confirmed that the minimum Python version required to run Axolotl is 3.10, allowing the use of `match..case` statements in the codebase.

- **Gradio Configurability Inquiry**: A member discussed making hardcoded Gradio options configurable through YAML files. They explored how to pass various configuration options, such as making the Gradio interface private and controlling the IP address and port number.

- **Gradio Tokenization Issues Examined**: There was an issue reported with Gradio not using the correct tokens for the llama3 model, which led to a discussion on how the default tokens could be overwriting the already loaded tokenizer.

**Link mentioned**: <a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583">FIX: TRL trainer preprocessing step was running in one process by ali-mosavian · Pull Request #1583 · OpenAccess-AI-Collective/axolotl</a>: Description We weren&#39;t passing dataset_num_proc to TRL training config, thus the initial data preprocessing steps in the TRL trainer was running in one process only. Motivation and Context Speeds ...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1235581892145975317)** (7 messages): 

- **Epochs and Batch Sizes Inquiry**: A member mentioned they usually go with **4 epochs** and a **batch size of 4** when training their models.
- **Llama3 Model Inference Confusion**: A member asked for guidance on how to call inference after training **llama3** using the fft script, noting that the regular `qlora_model_dir` does not seem applicable.
- **SafeTensor to GGUF Conversion Challenge**: Discussing conversion from **SafeTensors to GGUF**, a member expressed difficulties finding a way to convert to various gguf types like `Q4_K` or `Q5_K`, after using `llama.cpp`, which seemed limited in options.
- **Script Solution for Conversion Dilemma**: Another member pointed towards a conversion script available in the `llama.cpp` repository, specifically referencing the [GitHub link to convert-gg.sh script](https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh).
- **Limited Conversion Options in llama.cpp**: Despite the previous suggestion, the same member reiterated the issue, stating the `llama.cpp` conversion script provides only two gguf conversion options, and they are looking for a broader range of types such as `q4k`.

**Link mentioned**: <a href="https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh">llama.cpp/scripts/convert-gg.sh at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1235586698700783736)** (15 messages🔥): 

- **Configuring for Custom Dataset Roles**: A user inquired about configuring a dataset with the structure `{"messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}, {"role": "assistance", "content": "…"}]}`. They were advised to use `UserDefinedDatasetConfig` class to align the structure with the system's expectations.

- **Preprocessing Conversations for ShareGPT**: In response to a dataset structure question, it was suggested to preprocess messages by concatenating the "content" with the respective role identifier, ensuring it adheres to the `sharegpt` model's expected format.

- **Filling in Dataset Configuration Keys**: When asked how to fill in certain keys in a dataset configuration block, it was recommended to set the `conversation` to `Llama2ChatConversation`, map the `field_human` to "user", the `field_model` to "assistance", and appropriately categorize the "system" and "user" as input roles and "assistance" as the output role.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py">FastChat/fastchat/conversation.py at main · lm-sys/FastChat</a>: An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena. - lm-sys/FastChat</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=590b42af-2946-480b-80b8-8ae1021929e1)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e3e12dac-7c3d-42e8-a7f8-1e0485a19562)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=843813ee-d860-4061-9f19-b32faedaa383)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1235564223971725402)** (32 messages🔥): 

- **DeepSpeed Stage 3 Quality Concerns Cleared**: DeepSpeed Stage 3, also known as ZeRO-3, does not inherently degrade model quality but optimizes memory usage during training. It's essential to correctly implement and integrate ZeRO-3 within the training pipeline to avoid potential issues due to misconfigurations. [DeepSpeed documentation](https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167) can guide configuration.
  
- **Flash Attention with DeepSpeed for Fine-Tuning**: It is possible to use both Flash Attention and DeepSpeed Stage 3 for fine-tuning models, requiring integration of Flash Attention into the model and DeepSpeed Stage 3 setup within the training script. Proper configuration is crucial to leverage both technologies effectively.

- **Speed Improvements with DeepSpeed Stage 3**: DeepSpeed Stage 3 can speed up very large model training, allowing for larger batch sizes and reducing the need for complex parallelism strategies. However, the extent of speedup can vary based on model architecture, hardware setup, and data loading efficiency.

- **Training with LLaMA 3 Instruct on Axolotl**: Instructions for training with LLaMA 3 Instruct involve setting up an environment, creating a YAML config file, and initiating training and inference through commands using Accelerate and Axolotl. Adjustments specific to LLaMA 3 Instruct's implementation may be required. [Axolotl GitHub](https://github.com/openaccess-ai-collective/axolotl)

- **Understanding VRAM Usage with Axolotl Configurations**: Utilizing the simple `qlora.yaml` config in Axolotl examples uses both GPUs equally, but transitioning to FSDP or DeepSpeed Stage 3 might not show significant VRAM reduction due to various factors including model compatibility and overhead in managing sharded models. Configurations may need fine-tuning to optimize memory savings.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167)),">accelerate/docs/source/usage_guides/deepspeed.md at main · huggingface/accelerate</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/colab-notebooks/colab-axolotl-example.ipynb#L1L2)">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl#egg=axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=48f435d8-7ace-4f56-b4a5-0936a0f2d236)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6ec2ec3e-8632-45bb-9b50-4d25265230c0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fd359a44-f5ac-4e19-b938-f7288b3cfb04)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3c34d3f5-597f-4472-95aa-17cd8c03e44e)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fcb1eb4e-b085-4f2a-aeda-82b4c38beb8d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/huggingface/transformers/tree/main/docs/source/en/deepspeed.md#L167L302)">transformers/docs/source/en/deepspeed.md at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=427de887-db8b-40a1-9dba-accee8329079)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1235497314949398570)** (63 messages🔥🔥): 

- **Documentation Confusion Cleared**: A member shared a link to [Open Interpreter local installation documentation](https://docs.openinterpreter.com/guides/running-locally), which specifically includes instructions for **Ollama**, **Jan.ai**, and **Llamafile** with particular emphasis on **dolphin-mixtral**.
- **Prompt Adjustment for Conciseness**: A member advised using the `--profile 01` command for Open Interpreter to avoid repetitive recap of steps and plans. They also shared a [link to the related system message](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/01.py).
- **Open Source AI Hackathon Announcement**: An invite was made to join a team for the Microsoft Open Source AI hackathon in Seattle, with details and registration [linked here](https://lu.ma/iu1wijgd).
- **Open Interpreter Server Hosting Query**: A member inquired whether it's possible to host a server running **Open Interpreter** for others to connect to, and another confirmed it's feasible, pointing to usage of the `--api_base` along with `--model openai/custom --api_key dummykey`.
- **Local Model Hosting for Mobile Devices Guidance**: Info was sought on setting up a local Open Interpreter model for access by mobile devices, to which links to GitHub documentation were provided, referring to [Android device setup](https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#android) and [running Open Interpreter locally](https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://ip_address:port/v1`">no title found</a>: no description found</li><li><a href="https://docs.openinterpreter.com/guides/running-locally">Running Locally - Open Interpreter</a>: no description found</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2Fopen-interpreter%20skill&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://tenor.com/view/life-barrel-me-roll-gif-17943995">Life Barrel GIF - Life Barrel Me - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2F01%20skill&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://lu.ma/iu1wijgd">Open Source AI Hackathon #4 · Luma</a>: Following feedback from our last hackathon, we have found a sponsor for LLMs! OctoAI will provide all registrants the opportunity to get $50 in…</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#android">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/commits/59956e01ebedc74e0bfed80352ea0a90ecf154b1/interpreter/core/computer/skills/skills.py">History for interpreter/core/computer/skills/skills.py - OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1235508299227725896)** (10 messages🔥): 

- **Speaker Selection: A Delicate Process**: The choice of speaker for an electronics project is being closely evaluated, with **Ben** discussing options with vendors and considering integration with the PCB design. This decision is expected to unfold over weeks, with updates to follow based on validation results.

- **Fair Game: Reviewing Released Products**: Discussing product reviews, a member expressed that it is completely valid to review a product that has been officially released, implying confidence in the reviewer's understanding of the product space.

- **Speed Boost for Whisper RKNN**: An improved branch for Whisper RKNN on **Rockchip RK3588 SBCs** has been shared, boasting a 250% performance increase per [rbrisita's GitHub](https://github.com/rbrisita/01/tree/rknn). The contributor plans to introduce LLM RKNN features next.

- **Troubleshooting Interpreter Errors**: One user encountering errors with the `interpreter` command was advised to add `--api_key dummykey` to their execution. For further assistance, they were directed to specific Discord channels for issue discussion.

- **Progress on TMC Protocol for iOS**: A discussion is underway regarding the implementation of the **TMC protocol** for iOS, which grants access to native features such as the calendar and iMessage. The member is contemplating the benefits of TMC over standard function calling during development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pastebin.com/zGkZRhPs">error file - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/rbrisita/01/tree/rknn">GitHub - rbrisita/01 at rknn</a>: The open-source language model computer. Contribute to rbrisita/01 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1235545486342225962)** (2 messages): 

- **Open Source AI Vtuber Kit Released**: *Nikechan* presented an **AI Vtuber starter kit** requiring an OpenAI key and YouTube Key for operation. The project is available on [GitHub](https://github.com/tegnike/nike-ChatVRM) and was also announced via [Twitter](https://twitter.com/tegnike/status/1784924881047503202).

- **AI Vtuber Runs Offline**: *Hensonliga* shared their AI Vtuber repository that runs entirely offline without the need for an API and noted the content can be uncensored. The announcement included a [YouTube demonstration](https://youtu.be/buaK84oSWCU?si=P02NIYHvrVj7m8Lb) and a link to the [GitHub repository](https://github.com/neurokitti/VtuberAI.git).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tegnike/nike-ChatVRM">GitHub - tegnike/nike-ChatVRM: 誰でもAITuberお試しキット</a>: 誰でもAITuberお試しキット. Contribute to tegnike/nike-ChatVRM development by creating an account on GitHub.</li><li><a href="https://youtu.be/buaK84oSWCU?si=P02NIYHvrVj7m8Lb">Neuro Sama Competitor running Locally! V0.2 [FOSS, Local, No API]</a>: il make a github repo one sec.sorry for my bad mic quality. i am using my headphones mic so the bluetooth badwidth is killing mic qualityalso a bit of vram i...</li><li><a href="https://github.com/neurokitti/VtuberAI.git">GitHub - neurokitti/VtuberAI</a>: Contribute to neurokitti/VtuberAI development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1235590225275654174)** (1 messages): 

- **Traffic Surge Causes Blips**: OpenRouter experienced higher-than-normal errors due to a significant increase in traffic, causing intermittent issues.
- **Scaling Efforts Underway**: An update at 7:30am PT indicated that the scaling process to manage the surge was still in progress, reducing but not entirely eliminating the issues.
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1235502261099302976)** (72 messages🔥🔥): 

- **Exploring Payment Alternatives**: Members inquired about the possibility of **OpenRouter** supporting additional payment methods like [Stripe](https://stripe.com) to include **WeChat Pay** and **Alipay**, noting that there's extra paperwork required.
- **Upcoming AI Model Teasers**: Excitement and speculation bubbled around new and large-scale language models like **LLaMA-3** and potential forthcoming releases from companies like Soliloquy, while acknowledging proprietary model limitations.
- **Concern Over Model Dumbing After Fine-Tuning**: A technical discussion unfolded around the consequences of fine-tuning large language models without access to instruct datasets, suggesting that **batching old data with new can prevent catastrophic forgetting**.
- **Interest in Easier Payment Integration**: There was a suggestion to develop an app integrating with **Google payment services** for easier transactions.
- **Resolving Gemini Pro Issues**: User issues with **Gemini Pro** messages, specifically starting with an "assistant" role, were addressed with updates and workarounds mentioned, including prepending user role messages in prompts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ai21.com/blog/announcing-jamba-instruct">Built for the Enterprise: Introducing AI21’s Jamba-Instruct Model</a>: An instruction-tuned version of our hybrid SSM-Transformer Jamba model, Jamba-Instruct is built for reliable commercial use, with best-in-class quality and performance.</li><li><a href="https://huggingface.co/collections/nvidia/chatqa-15-662ebbf6acc85f5c444029a8">Llama3-ChatQA-1.5 - a nvidia Collection</a>: no description found
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/)** (1 messages): 

angry.penguin: https://huggingface.co/spaces/YupengZhou/StoryDiffusion
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1235630378379771994)** (15 messages🔥): 

- **Mysterious Messages Puzzle AI Devs**: AI Stack Devs are *puzzled by empty messages or strings of numbers* blocking conversation flow in **ai-town**, using **ollama** and **llama3 8b**. *Tokenizer issues* have been suggested, but there is no definitive answer yet.

- **Godly Praise for Cocktail Peanut**: A member gave a *shoutout* to **cocktail peanut** simply stating they're doing "gods work," but no context was provided on what work is being referred to.

- **AI Society Without Leadership**: Members discussed whether AIs *elect a leader* within simulations, with the consensus being there's no mayor or elected official. Curiosity was expressed regarding this aspect from the original simulation paper.

- **Simplifying AI Character Roles**: A member mentioned the idea that setting up a *mayoral election* could be easily implemented in the player bios of AI characters.

- **AI Town Experiences and Tools Shared**: Links were shared by @.casado promoting [@TheoMediaAI's exploration](https://x.com/TheoMediaAI/status/1786377663889678437) of AI simulations, and [@cocktailpeanut's new web app](https://x.com/cocktailpeanut/status/1786421948638965870) that allows for *replaying any AI Town by importing a sqlite file*. The latter supports Mac & Linux, with a requirement for **ollama**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheoMediaAI/status/1786377663889678437">Tweet from Theoretically Media (@TheoMediaAI)</a>: Exploring Two remarkable AI World Simulations: First, the AI-Westworld from @fablesimulation (PUBLIC BETA is OPEN!), and also taking @realaitown for a spin, but recreating the best movie ever (The THI...</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">Tweet from cocktail peanut (@cocktailpeanut)</a>: Introducing AI Town Player  Did you know that the entire AI Town is stored in a single sqlite file via @convex_dev?    I reverse engineered the schema and built a web app that lets anyone REPLAY any A...
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1235763541642838069)** (26 messages🔥): 

- **Node Version Hinders Local Backend Progress**: A member encountered an error while running `convex-local-backend` due to an incorrect node version: *Wrong node version v19.9.0 installed at node*. It was suggested to switch to node version 18 using `nvm use 18`.

- **Local Development Halted by Backend Bugs**: Another member faced multiple errors when attempting to run `convex-local-backend` on Ubuntu 18. Issues included problems with the node version, `rush buildCacheEnabled`, and a type error *(Unknown file extension ".ts")*.

- **In Search of a Simpler Setup**: Frustrated by the complications, the member inquired about a Docker build to simplify the deployment process. An alternative of using `ollama` locally and `convex` remotely was mentioned.

- **Sharing Resources for Community Projects**: A request was made to share a larger map with another user working on the *Pinokio* build. Member edgarhnd agreed to share the map.

- **LLama-Farm Project To Connect Local Machines**: A project named *llama-farm* was introduced, designed to connect one or more machines running `Ollama` to a cloud backend or hosted website, enabling the use of local LLM compute without exposing the machines to public internet requests.

**Link mentioned**: <a href="https://github.com/get-convex/convex-backend/issues/1">TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension &quot;.ts&quot; for /app/npm-packages/convex/src/cli/index.ts · Issue #1 · get-convex/convex-backend</a>: I ran the steps in the prerequisites then got this when running just run-local-backend Error: Failed to run convex deploy: TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension &quot;.ts&quot...

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/1235713660966670386)** (2 messages): 

- **Emotional Penguins and Discord Bots**: A member displayed an emoji expressing deep contemplation or skepticism, possibly gearing up for a discussion or pondering a question related to AI on Raspberry Pi.
- **Channel Meets User's Interest**: Another member expressed that the ai-raspberry-pi channel is perfectly suited for their interests, implying they might engage in or contribute to discussions on AI development using Raspberry Pi.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1235517093793759274)** (26 messages🔥): 

- **Implementation Challenges with SoundStream**: A member experienced difficulties in implementing the SoundStream paper by Google due to unspecified index names and values. Another member pointed out an existing code repository that might help, available on [GitHub](https://github.com/wesbz/SoundStream).
  
- **Newbie Welcome and Offers Course**: A newcomer to AI-generated art, after finishing a Udemy course on Stable Diffusion, offered to share the course for free in hopes of building connections and learning more advanced skills from the community.

- **Investing Strategies in Chat**: Various members humorously discussed their investment strategies, ranging from seeking services that 10x their money to preferring ones that halve their funds.

- **Insights on Model Training Limitations**: In the StableDiffusion subreddit, discussions mention that using both T5 text encoder and CLIP might not improve prompt adherence as expected, with some expressing surprise and others nodding to the possibility of high CLIP dropout as a potential factor.

- **StableDiffusion Development Updates**: Updates from the Stable Diffusion community indicate a shift in focus from larger models, due to hardware constraints, to architectural and training improvements on smaller models. The conversation also touches on the importance of correctly training with CLIP to avoid biases and limitations.

**Link mentioned**: <a href="https://github.com/wesbz/SoundStream">GitHub - wesbz/SoundStream: This repository is an implementation of this article: https://arxiv.org/pdf/2107.03312.pdf</a>: This repository is an implementation of this article: https://arxiv.org/pdf/2107.03312.pdf - wesbz/SoundStream

  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1235718206744957018)** (7 messages): 

- **Bot Banishment Brigade**: Two members humorously interacted over the removal of a perceived bot from the discussion, with one member cheerfully noting their timely attention to the chat.
- **Scrutiny Over Dataset Choices**: A member queried why experiments are not conducted on standard datasets such as MNIST, CIFAR, or ImageNet but rather on synthetic ones. Another member attributed this choice to the goal of demonstrating interpretability.
- **Interpretability vs Real-world Application**: Following a discussion on the focus of experiments for interpretability, another member expressed skepticism, pointing out that methods need to solve real-world tasks to be truly compelling.
- **New Tool on the Block**: A link to [StoryDiffusion](https://storydiffusion.github.io/) was shared by a member with no additional context provided regarding its purpose or relevance.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1235497831431798866)** (26 messages🔥): 

- **Integration Struggles with Text Embedding**: A user expressed difficulty in integrating a text embedding model with LangChain, mentioning the need to utilize a **SageMaker endpoint** rather than an API key. The user sought advice for alternative methods or resources for such integration.
  
- **LangChain Package Version Confusion**: A member raised a question about installing the `langchain` PyPI package, noting that the version of `langchain-openai` specified is quite old (`<=0.1`) and wondering if this is intentional for compatibility reasons, given that the current version of `langchain-openai` is significantly updated.
  
- **Looking for Chatbot Enthusiasts**: A user inquired about finding a community focused on developing conversational chatbots, seeking recommendations from fellow members.
  
- **Data Retrieval Query for CSV**: A member asked how to embed a single column from a CSV file into a LangChain application and later retrieve data from a different column in response, using a use case involving email lookup.
  
- **Hackathon Heads-up!**: An announcement was shared about an upcoming hackathon named **BeeLoud**, where participants are challenged to build AI products within 54 hours, with a potential prize pool of up to $25,000. The event welcomes diverse skill sets and is set to occur on May 10-12, with participants from across the globe.

- **Request for Interview with LangChain Users**: A user requested to discuss the biggest challenges faced by those frequently building with AI agents using LangChain or other frameworks, providing a link to schedule a call for detailed conversations.

- **SQL Agent Functionality Query**: A conversation was sparked about whether it's possible to call **MSSQL functions** using the SQL agent in LangChain, leading to a detailed explanation of using the `SqlToolkit` for such executions and relevant links for further guidance.

- **LangChain RAG Implementation Insights**: A user preparing for a role involving LangChain's implementation asked for key points and advice on how to prepare for an interview related to LangChain, particularly in context with implementing RAG through LangChain.

- **Handling Large Databases in LangChain**: Members discussed various methods for using LLM to query databases, debating between converting database data to natural language text versus using ChatGPT to convert natural language to SQL queries, and considering the challenges of dealing with large databases within such paradigms.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://beeloud.xyz/build/">Build - Beeloud</a>: Can you build the next billion dollar startup in 3 days? Sam Altman and his buddies are betting you can. You&#8217;ve officially been challenged to join this hackathon. I accept&hellip; Continue readi...</li><li><a href="https://calendly.com/leonchen-1/30min">30 Minute Meeting - Leon Chen</a>: no description found</li><li><a href="https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema">no title found</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/13931>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1235647398718738433)** (1 messages): 

- **Clarifying Feedback Submission Confusion**: A member was uncertain about how feedback submission works when using the **langserve feedback endpoint**. It was explained that an "OK" response from Langserve only indicates successful submission but does not confirm recording by langsmith, as requests may be rejected if deemed unauthenticated or invalid by the server.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1235659099933380659)** (3 messages): 

- **Boosting Email Drafting with RAG**: Enhancements to **LangChain's LangGraph Agents** now include [Retrieval-Augmented Generation (RAG)](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da) for more intelligent email drafting capabilities. The Medium article details how this integration can significantly improve the efficiency and quality of AI-generated email communication.

- **LangChain Java Port Available**: For developers interested in using **LangChain** with Java, [langchain4j](https://github.com/langchain4j/langchain4j) offers a Java version of LangChain, expanding the possibilities for integration into various applications.

- **Dragonfly Integrates with LangChain**: A new blog post highlights the integration of **Dragonfly**, an in-memory data store, with **LangChain**, to manage chat context and improve performance of AI-powered applications. Detailed information and code snippets for this enhancement can be found in the [blog post](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly">Efficient Context Management in LangChain Chatbots with Dragonfly</a>: Explore efficient context management for LangChain OpenAI chatbots with Dragonfly, enhancing performance and user experience through caching techniques.</li><li><a href="https://github.com/langchain4j/langchain4j">GitHub - langchain4j/langchain4j: Java version of LangChain</a>: Java version of LangChain. Contribute to langchain4j/langchain4j development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1235621942157905992)** (13 messages🔥): 

- **Hints of Implementation Intentions**: It appears there's consideration for implementing something that is not commonly practiced. The specifics are not yet clear, but there's an expressed interest in beginning the implementation process.
- **Technical Report Wait Game**: There's anticipation for a technical report that hasn't been published yet, which seems to be causing some confusion. The absence of this report is attributed to timescale constraints related to data.
- **Reward Model Contest Alert**: There's mention of a reward model competition by LMSYS with a significant 100k prize, drawing a parallel to older Kaggle competitions and prompting a call for a similar 200k Interconnects contest.
- **Perspectives on Ensembling**: The concept of ensembling reward models is recognized, but it's viewed as suboptimal, though potentially sufficient to give some competitors an edge.
- **PPO's Connection to Reinforce**: There was a discussion suggesting that Proximal Policy Optimization (PPO) could theoretically be reduced to the REINFORCE algorithm with a particular set of hyperparameters, possibly when the step size limiter is turned off. A link to OpenAI's [Spinning Up documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html) was shared for further clarification.

**Link mentioned**: <a href="https://spinningup.openai.com/en/latest/algorithms/ppo.html">Proximal Policy Optimization &mdash; Spinning Up  documentation</a>: no description found

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1235547821285834812)** (4 messages): 

- **Drama Unfolding with Potential Model Leak**: A member discussed the possibility that a leaked model with *oddly specific quant* is actually from **GDM**, referenced by a [tweet from @teortaxesTex](https://x.com/teortaxestex/status/1785974744556187731?s=46). Suspicions arose due to strange details like a *sudden 4chan link, a throwaway HF account, and Reddit comments*.
  
- **Quant Leak on 4chan Sparks Curiosity**: A user summarized the situation as a "random-ass quant from llama3 dropped on 4chan" potentially originating from **GDM**.

- **Research Paper Fails to Impress with Missing RewardBench Scores**: A member shared a [link to a paper](https://arxiv.org/abs/2405.01535) that missed reporting on RewardBench scores and hinted at underperformance by adding a facepalm emoji reaction.

- **Prometheus 2: A Challenger to GPT-4?**: The paper introduced **Prometheus 2**, an evaluator language model positioned as a better alternative to proprietary LMs like GPT-4. It claims to align closely with human judgement and to handle various types of assessments.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.01535">Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>: Proprietary LMs such as GPT-4 are often employed to assess the quality of responses from various LMs. However, concerns including transparency, controllability, and affordability strongly motivate the...</li><li><a href="https://x.com/teortaxestex/status/1785974744556187731?s=46">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: ...actually, why the hell am I assuming it&#39;s not their model, disseminated for collective pentesting  - miqu-like oddly specific quant leak to preclude improvements  - sudden 4chan link, throwaway...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1235683288828346498)** (5 messages): 

- **$100,000 Human Preference Prediction Challenge**: [LMSYS and Kaggle launch a competition](https://x.com/lmsysorg/status/1786100697504833572?s=46) where participants predict user preferences between Language Model (LM) responses. The dataset includes over 55,000 conversations featuring LLMs like **GPT-4, Claude 2, Llama 2, and Mistral**.

- **A Short Victory Cry**: A member simply commented "mogged".

- **Kaggle’s Appeal to Researchers**: A member inquired whether researchers generally have a liking for platforms like **Kaggle**.

- **Repeated Success Raises Questions**: Reacting to the competition announcement, a member expressed disbelief noting, "he can't keep getting away with this".

- **Casual Chat About Commitments**: The conversation continued with a more casual tone, referring to a 'John' who said 'maybe' to a member, suggesting a potential context of event or project participation.

**Link mentioned**: <a href="https://x.com/lmsysorg/status/1786100697504833572?s=46">Tweet from lmsys.org (@lmsysorg)</a>: Exciting news -- we&#39;re thrilled to announce that LMSYS + @kaggle are launching a human preference prediction competition with $100,000 in prizes!  Your challenge is to predict which responses user...

  

---


**Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1235674317413417050)** (8 messages🔥): 

- **Valuable Value Functions in RLHF**: One member pondered why reward functions are released but not the value functions obtained during RLHF training, questioning if somehow value functions are not produced. Another clarified that **value functions are indeed obtained** when using algorithms like PPO.

- **Reward Models Release Practices Questioned**: It was mentioned that claiming people release reward models or functions as a standard practice may be an **overstatement in the community**.

- **Value of Value Functions Recognized**: Despite uncertainties about their release, it is acknowledged that **value functions are considered quite valuable**, especially in the context of planning.

- **Research Gap on Value Functions?**: A member speculated on the absence of research focusing on **the value of value functions in classical RL**, implying an opportunity for further exploration.

- **Link Between Value and Credit Assignment**: The relationship between the **value functions in PPO** and **credit assignment in DPO** was noted as a potentially interesting area for future research.
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1235491704820465744)** (21 messages🔥): 

- **Search System Design for Large Documents**: A member explored ideas for building a search system for large PDFs and considered generating embeddings for semantic search, summarizing documents with LLMs for retrieval, and indexing key information extracted by LLMs.
  
- **Tokenization Clarification for Llama with Command R+**: A member asked about the necessity of adding a "<BOS_TOKEN>" when generating text with the llama-cpp-python library and Command R+ after noticing its automatic addition during tokenization.

- **Cohere API Key Inquiry for RAG**: One user inquired whether it's possible to use a free Cohere API key for RAG, with **another member confirming its availability but noting rate limitations**.

- **Discussion on C4AI Command R+ Implementation**: Members shared links to the **[C4AI Command R+ model](https://huggingface.co/CohereForAI/c4ai-command-r-plus)** on HuggingFace and a [quantized version](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit), alongside technical parameters for implementation, and discussed running it locally with varying degrees of system requirements.

- **Code Interpreter SDK Announcement**: A member shared a demo of the [launch of the Code Interpreter SDK](https://x.com/tereza_tizkova/status/1786058519701254268?s=46&t=yvqplJRJNpP5EM3LZLMQlA) on Twitter, with another questioning the uniqueness of this release in light of previous similar technologies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com/library/command-r">command-r</a>: Command R is a Large Language Model optimized for conversational interaction and long context tasks.</li><li><a href="https://x.com/tereza_tizkova/status/1786058519701254268?s=46&t=yvqplJRJNpP5EM3LZLMQlA">Tweet from Tereza Tizkova (@tereza_tizkova)</a>: 🚀 We are launching the @e2b_dev Code Interpreter SDK 🧠  It&#39;s a building block for any AI app - SDK for code interpreting! Use it to build 🔸 Advanced data analysts 🔸 Generative UI 🔸 AI softwar...</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: no description found</li><li><a href="https://ollama.com/library/command-r-plus">command-r-plus</a>: Command R&#43; is a powerful, scalable large language model purpose-built to excel at real-world enterprise use cases.
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1235494486143008790)** (19 messages🔥): 

- **llamafile as a Linux Service**: A systemd script to launch **llamafile** as a service on Rocky Linux 9 was shared, detailing execution commands and environment configurations necessary to run llamafile with specific arguments, such as server port and model path.
- **Feature Request for Server Base URL**: A feature request for the ability to specify a base URL for llamafile in server mode was addressed with a [GitHub issue link](https://github.com/Mozilla-Ocho/llamafile/issues/388), expressing the need for proxy support through Nginx to serve llamafile under a subdirectory.
- **Interest in Distil Whisper German Model**: There's curiosity about incorporating whisper models like [distil-whisper-large-v3-german](https://huggingface.co/primeline/distil-whisper-large-v3-german) for speech recognition and potential for a blog post featuring its application, including a hypothetical pipeline of STT -> LLM -> TTS.
- **Embedding Direction Discrepancies**: An issue was discussed where embeddings produced by llamafile and by llama.cpp show a low cosine similarity, indicating differing directions, a problem evidenced by a [GitHub issue](https://github.com/Mozilla-Ocho/llamafile/issues/391) and tested with Python scripts provided.
- **Conversing with Documents/Code**: The question of how to enable llamafile to ingest documents and code for conversational interaction was addressed with a suggestion to use `curl` API calls, referencing examples from the [llama.cpp chat script](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.sh#L64).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:8080"):">no title found</a>: no description found</li><li><a href="http://localhost:8080")">no title found</a>: no description found</li><li><a href="http://localhost:8081")">no title found</a>: no description found</li><li><a href="https://huggingface.co/primeline/distil-whisper-large-v3-german">primeline/distil-whisper-large-v3-german · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/apple/OpenELM-3B-Instruct">apple/OpenELM-3B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/388">Feature Request: Option to specify base URL for server mode · Issue #388 · Mozilla-Ocho/llamafile</a>: I&#39;ve been testing the use of Nginx as a proxy to serve llamafile under a subdirectory. i.e. to be able to access the llamafile server via a URL like this: https://mydomain.com/llamafile/ Llamafile...</li><li><a href="https://huggingface.co/models?search=OpenELM-3B-Instruct-gguf">Models - Hugging Face</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/391">Unexpected output from server.cpp `/embedding` endpoint · Issue #391 · Mozilla-Ocho/llamafile</a>: What is the issue? The embeddings produced by a model running in llamafile seem to be substantially different from those produced by llama.cpp. llama.cpp embeddings are very close (~0.99 cosine sim...</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.sh#L64">llama.cpp/examples/server/chat.sh at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1235486834499522650)** (4 messages): 

- **Tiny Progress Update**: One member inquired about **progress**, to which another confirmed substantial progress made two days ago. 
- **Contribution Milestone**: A different member shared their enthusiasm for making their **first commit** to the project and expressed joy when it was successfully committed.
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1235659100998729759)** (13 messages🔥): 

- **Clarification on Blobfile's Importance**: The utility of `blobfile` in `examples/llama.py` was questioned. It's clarified that `load_tiktoken_bpe` depends on `blobfile`.

- **Forward Pass Compute Graph Troubles**: A member had an issue with generating the forward pass compute graph for a simple neural network. They were advised to ensure computation by uncommenting `out.item()` or using `out.realize()` and also to resolve a `NameError` by installing necessary libraries.

- **Networkx Installed but pydot Missing**: The aforementioned error persisted despite having `networkx` installed, and was eventually resolved by installing `pydot`.

- **Graphviz Installation Resolves dot Command Error**: After implementing the solution to install `pydot`, a new error about a missing `dot` command was encountered and solved by installing `graphviz`. 

- **Suggestion to Update Documentation**: A member suggested updating the documentation to include a hint that installing `graphviz` can resolve the `sh: dot: command not found` error.
  

---



**AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1235742995437977641)** (1 messages): 

- **Jamba-Instruct Takes Center Stage**: AI21 Labs announced the launch of **Jamba-Instruct**, an instruction-tuned version of their hybrid SSM-Transformer **Jamba** model. They invite feedback and express willingness to accommodate use cases requiring more than the initial 256K context window.

- **Read All About Jamba-Instruct**: For an in-depth understanding, AI21 Labs encourages reading the *Jamba-Instruct blog post* at [AI21's Blog](https://www.ai21.com/blog/announcing-jamba-instruct), which details how Jamba-Instruct excels in quality and performance for commercial applications.

**Link mentioned**: <a href="https://www.ai21.com/blog/announcing-jamba-instruct">Built for the Enterprise: Introducing AI21’s Jamba-Instruct Model</a>: An instruction-tuned version of our hybrid SSM-Transformer Jamba model, Jamba-Instruct is built for reliable commercial use, with best-in-class quality and performance.

  

---


**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1235603967384490037)** (4 messages): 

- **Jamba-Instruct Unveiled**: AI21 Labs announced the launch of **Jamba-Instruct**, shared via a [Twitter post](https://twitter.com/AI21Labs/status/1786038528901542312).
- **Exploring Larger Context Windows**: In response to an inquiry about context windows larger than 256k, an AI21 Labs staff member expressed willingness to explore **much higher context windows** and invited the member to discuss use cases in a direct message.
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1235515365598167141)** (2 messages): 

- **Warm Greetings**: A member greeted the community with a simple "Hello".
- **Compute Grants Available**: For those seeking **fast compute grants**, a member shared a link to a Twitter post from @PrimeIntellect: [Fast Compute Grants Tweet](https://twitter.com/PrimeIntellect/status/1786386588726960167).
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1235956851133386872)** (2 messages): 

- **LLaMA Quantization Quandary**: A Discord member highlighted a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/) discussing the impact of quantization on **LLaMA 3's** quality compared to LLaMA 2. They linked to an [arXiv paper](https://arxiv.org/abs/2404.14047) detailing the performance degradation with low-bit quantization, raising questions about post-training quantization methods.
- **Quantization Loses Details**: A member expressed that the significant quantization of **Meta's LLaMA** which ignores the *chinchilla scaling law* and uses 15T tokens, could be the reason for major information loss, affecting performance. This suggests a greater risk of degradation with enhanced precision reduction in larger models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.14047">How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study</a>: Meta&#39;s LLaMA family has become one of the most powerful open-source Large Language Model (LLM) series. Notably, LLaMA3 models have recently been released and achieve impressive performance across ...
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1235948611292893263)** (1 messages): 

- **Fast Compute Grants for Skunkworks Projects**: A member mentioned they are eager to fund some exciting **Skunkworks projects** and provided a [twitter link for details](https://twitter.com/PrimeIntellect/status/1786386588726960167). If you’re looking for fast compute grants, this could be an opportunity.
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1235576027233910865)** (1 messages): 

- **Digital Housekeeping Woes**: A member expressed the need for an LLM that could assist with cleaning up the scattered **7B localmodels** taking up space across various directories on their hard drive. The frustration stemmed from numerous apps and libraries contributing to the disarray.
  

---



