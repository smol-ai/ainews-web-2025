---
id: b863d96e-eb38-4240-ace3-59652884734d
title: Somebody give Andrej some H100s already
date: '2024-05-29T01:24:27.055047Z'
original_slug: ainews-somebody-give-andrej-some-h100s-already
description: >-
  AI news for 5/27/2024-5/28/2024 covers advancements in training tiny GPT-2
  models, discussions on CUDA and H100 GPUs, and Twitter debates between Yann
  LeCun and Elon Musk on CNNs, AI safety, and regulation.
companies:
  - openai
  - meta
  - nvidia
  - tesla
models:
  - gpt-2
  - h100
topics:
  - cuda
  - gpt-2
  - llm
  - ai-training
  - ai-safety
  - ai-regulation
  - cnn
  - transformers
  - ai-research
---


<!-- buttondown-editor-mode: plaintext -->**C+CUDA is all you need.**

> AI News for 5/27/2024-5/28/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**382** channels, and **4432** messages) for you. 
Estimated reading time saved (at 200wpm): **521 minutes**.

Five years ago, OpenAI spawned its first controversy with GPT-2 being called ["too dangerous to release"](https://slate.com/technology/2019/02/openai-gpt2-text-generating-algorithm-ai-dangerous.html).

Today, with help from [FineWeb (released last month)](https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/), you can [train a tiny GPT-2 in 90 minutes and $20 in 8xA100 server time](https://github.com/karpathy/llm.c/discussions/481). It is already working ([kinda](https://x.com/karpathy/status/1795525191596138926)) for [the 350M version](https://news.ycombinator.com/item?id=40504950), and Andrej estimates that the full 1.6B model will take 1 week and $2.5k.

 ![image.png](https://assets.buttondown.email/images/28a220bf-db6e-4a67-b5ea-6738bfb86771.png?w=960&fit=max) 

And incredible accomplishment in 7 weeks of work from scratch, though at this point the repo is 75.8% CUDA, stretching the name of "llm.c".

Andrej also answered some questions on [HN](https://news.ycombinator.com/item?id=40502090) and on [Twitter](https://x.com/karpathy/status/1795484547267834137). one of the most interesting replies: 

**Q: How large is the set of binaries needed to do this training job? The current pytorch + CUDA ecosystem is so incredibly gigantic and manipulating those container images is painful because they are so large. I was hopeful that this would be the beginnings of a much smaller training/fine-tuning stack?**

**A: That is 100% my intention and hope and I think we are very close to deleting all of that.**

It would be cheaper and faster if [more H100s were available](https://x.com/karpathy/status/1795493747205238916). Somebody help a newly GPU poor out?

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

**Yann LeCun and Elon Musk Twitter Debate**

- **Convolutional Neural Networks (CNNs) Importance**: [@ylecun](https://twitter.com/ylecun/status/1795393908886712425) noted CNNs, introduced in 1989, are used in every driving assistance system today, including MobilEye, Nvidia, Tesla. **Technological marvels are built on years of scientific research shared through technical papers.**
- **LeCun's Research Contributions**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1795435037396988162) would pick @ylecun over @elonmusk, as **scientists who publish groundbreaking research are the cornerstone of technological progress, despite getting less recognition than entrepreneurs.**
- **Musk Questioning LeCun's CNN Usage**: [@elonmusk](https://twitter.com/elonmusk/status/1795426059921268969) asked @ylecun how Tesla could do real-time camera image understanding in FSD without ConvNets. [@ylecun](https://twitter.com/ylecun/status/1795428712460451841) responded Tesla uses CNNs, as attention is too slow for real-time high-res image processing. [@svpino](https://twitter.com/svpino/status/1795506451131044047) and [@mervenoyann](https://twitter.com/mervenoyann/status/1795506858985177137) confirmed Tesla's CNN usage.
- **LeCun's Research Productivity**: [@ylecun](https://twitter.com/ylecun/status/1795219718837616775) shared he published over 80 technical papers since January 2022, questioning Musk's research output. He also noted he works at Meta, with [@ylecun](https://twitter.com/ylecun/status/1795158771695542279) stating there's nothing wrong with that.
- **Musk Acting as LeCun's Boss**: [@ylecun](https://twitter.com/ylecun/status/1795265406191735191) joked Musk was acting as if he were his boss. [@fchollet](https://twitter.com/fchollet/status/1795226758502826154) suggested they settle it with a cage fight, with [@ylecun](https://twitter.com/ylecun/status/1795268462597824548) proposing a sailing race instead.

**AI Safety and Regulation Discussions**

- **AI Doomsday Scenarios**: [@ylecun](https://twitter.com/ylecun/status/1795032310590378405) criticized "AI Doomsday" scenarios, arguing AI is designed and built by humans, and **if a safe AI system design exists, we'll be fine. It's too early to worry or regulate AI to prevent "existential risk".**
- **AI Regulation and Centralization**: [@ylecun](https://twitter.com/ylecun/status/1794998977105981950) outlined "The Doomer's Delusion", where **AI doomsayers push for AI monopolization by a few companies, tight regulation, remote kill switches, eternal liability for foundation model builders, banning open-source AI, and scaring the public with prophecies of doom.** They create one-person institutes to promote AI safety, get insane funding from scared billionaires, and claim prominent scientists agree with them.

**AI Research and Engineering Discussions**

- **Reproducing GPT-2 in C/CUDA**: [@karpathy](https://twitter.com/karpathy/status/1795484547267834137) reproduced GPT-2 (124M) in llm.c in 90 minutes for $20 on an 8X A100 80GB node, reaching 60% MFU. He also reproduced the 350M model in 14 hours for ~$200. **Full instructions are provided.**
- **Transformers for Arithmetic**: [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1795300845942382701) shared a paper showing **transformers can do arithmetic with the right embeddings**, achieving up to 99% accuracy on 100-digit addition problems by training on 20-digit numbers with a single GPU for one day.
- **Gemini 1.5 Model Updates**: [@lmsysorg](https://twitter.com/lmsysorg/status/1795512202465845686) announced Gemini 1.5 Flash, Pro, and Advanced results, with **Pro/Advanced at #2 close to GPT-4o, and Flash at #9 outperforming Llama-3-70b and nearly GPT-4-0125. Flash's cost, capabilities, and context length make it a market game-changer.**
- **Zamba SSM Hybrid Model**: [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1795299751644340465) shared the Zamba paper, a **7B SSM-transformer hybrid model achieving competitive performance against leading open-weight models at a comparable scale.** It's trained on 1T tokens from openly available datasets.
- **NV-Embed for Training LLMs as Embedding Models**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1795286849487098035) shared an NVIDIA paper on NV-Embed, which **improves techniques for training LLMs as generalist embedding models. It achieves #1 on the MTEB leaderboard.**

**Memes and Humor**

- **Musk vs. LeCun Memes**: [@svpino](https://twitter.com/svpino/status/1795503047004594637) and [@bindureddy](https://twitter.com/bindureddy/status/1795269862111256904) shared memes about the Musk vs. LeCun debate, poking fun at the situation.
- **AI Replacing Twitter with AI Bot**: [@cto_junior](https://twitter.com/cto_junior/status/1795479060258197877) joked about building an AI version of themselves on Slack to replace attending standups, rather than on Twitter.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Models and Architectures**

- **01-ai removes custom licenses from Yi models**: In /r/LocalLLaMA, 01-ai has [**switched the licensing of their original Yi models to Apache-2.0 on Huggingface**](https://www.reddit.com/r/LocalLLaMA/comments/1d1zzbz/01ai_just_removed_all_the_custom_licenses_from/), matching the license of their 1.5 series models.
- **InternLM2-Math-Plus models released**: A series of [**upgraded math-specialized open source large language models in 1.8B, 7B, 20B and 8x22B sizes**](https://www.reddit.com/r/LocalLLaMA/comments/1d1om5d/we_release_internlm2mathplus_with_18b7b20b_and/) was released. The InternLM2-Math-Plus-Mixtral8x22B achieves 68.5 on MATH (with Python) and 91.8 on GSM8K benchmarks.
- **Pandora world model introduced**: Pandora is a [**hybrid autoregressive-diffusion model that simulates world states by generating videos and allows real-time control with free-text actions**](https://www.reddit.com/r/LocalLLaMA/comments/1d1meba/pandora_towards_general_world_model_with_natural/). It aims to achieve domain generality, video consistency, and controllability.
- **llama.cpp adds support for Jamba architecture**: In /r/LocalLLaMA, [**support for AI21 Labs' Jamba architecture is being added to llama.cpp**](https://www.reddit.com/r/LocalLLaMA/comments/1d1ur6h/jamba_llamacpp_support/), with the first GGUF files being uploaded, including a model fine-tuned on the Bagel dataset.
- **AstroPT models released for astronomy**: [AstroPT](https://arxiv.org/abs/2405.14930) is an autoregressive pretrained transformer developed for astronomical use-cases, with models from 1M to 2.1B parameters pretrained on 8.6M galaxy observations. Code, weights, and dataset released under MIT license.

**AI Applications and Tools**

- **Optimizing Whisper for fast inference**: In /r/LocalLLaMA, tips were shared to [**speed up Whisper inference up to 5x using techniques like SDPA/Flash Attention, speculative decoding, chunking, and distillation**](https://www.reddit.com/r/LocalLLaMA/comments/1d1xzpi/optimise_whisper_for_blazingly_fast_inference/).
- **Android app for document Q&A**: [Android-Document-QA](https://www.reddit.com/r/LocalLLaMA/comments/1d1zzxr/androiddocumentqa_rag_pipeline_for_document_qa/) is an Android app that uses a LLM to answer questions from user-provided PDF/DOCX documents, leveraging various libraries for document parsing, on-device vector DB, and more.
- **MusicGPT for local music generation**: In /r/MachineLearning, [MusicGPT was introduced as a terminal app that runs MusicGen by Meta locally to generate music from natural language prompts](https://www.reddit.com/r/MachineLearning/comments/1d1vp2u/p_musicgpt_an_open_source_app_for_generating/). Written in Rust, it aims to eventually generate infinite music streams in real-time.
- **New web+LLM framework released**: An [**open-source web framework optimized for IO-bound applications integrating with LLMs and microservices**](https://www.reddit.com/r/LocalLLaMA/comments/1d1yofb/i_made_a_webllm_framework_looking_for_early/) was announced, looking for early adopters to try it out and provide feedback.

**AI Ethics and Safety**

- **Microsoft's Recall AI feature investigated over privacy concerns**: [Microsoft's new Recall AI feature, which tracks user activity to help digital assistants, is being investigated by UK authorities](https://mashable.com/article/microsoft-recall-ai-feature-uk-investigation) over privacy concerns, sparking debate about the data needed for useful AI assistance.

**AI Industry and Competition** 

- **Visualization of AI competition over past year**: A [**visualization from the LMSYS Chatbot Arena showing the performance of top models from major LLM vendors over the past year**](https://www.reddit.com/r/LocalLLaMA/comments/1d1qfby/evolution_of_ai_competition_in_the_last_year/) highlights the intensifying competition and shifting trends.
- **Contradictory claims about OpenAI equity clawbacks**: An article claims [**data contradicts Sam Altman's statements about ignorance of equity clawbacks at OpenAI**](https://thedeepdive.ca/sam-altman-claims-ignorance-on-equity-clawbacks-but-data-contradicts/).

---

# AI Discord Recap

> A summary of Summaries of Summaries


**LLM Advancements and Benchmarking**:

- **[Llama 3 Leads the Pack](https://lmsys.org/blog/2024-05-08-llama3/)**: Llama 3 from Meta has topped leaderboards like **ChatbotArena**, surpassing models such as **GPT-4-Turbo** and **Claude 3 Opus** in over 50,000 matchups.
- **IBM and RefuelAI Unleash New Models**: IBM's [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) enhances code task instruction following, while RefuelAI's [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) boasts efficiency for data-heavy tasks.

**Optimizing LLM Inference and Training**:

- **[Cutting Edge Quantization Techniques](https://www.deepspeed.ai/tutorials/zeropp/)**: ZeRO++ aims for a 4x reduction in communication overhead during large model training on GPUs.
- **Memory Efficiency Innovations**: The [vAttention](https://arxiv.org/abs/2405.04437) system manages KV-cache memory more dynamically, improving LLM inference sharpens, and **[QSolve](https://arxiv.org/abs/2405.04532)** introduces **W4A8KV4 quantization** for enhancing cloud-based GPU performance.

**Open-Source AI Frameworks and Community Efforts**:

- **[Axolotl Captures Dev Interests](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)**: Supports diverse data formats, aiding LLM pre-training and instruction tuning.
- **[LlamaIndex Sparks Academic Curiosity](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)**: Powers a new course about building agentic RAG systems with Andrew Ng, promising advancements in AI extensions like **bfloat16**.

**Multimodal AI and Generative Modeling Innovations**:

- **[Idefics2 8B Makes Waves in Conversation](https://twitter.com/sanhestpasmoi/status/1787503160757485609)**: Fine-tuning the discourse on chat interactions, while **[CodeGemma 1.1 7B Improves Coding Tasks](https://twitter.com/reach_vb/status/1786469104678760677)**.
- **[Phi 3 Pioneers Browser-based AI](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)**: Introducing powerful AI chatbots directly in browsers via WebGPU, setting the stage for privacy-focused interaction enhancements.

---

{% if medium == 'web' %}



# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**OCR Showdown: Google Vision vs. Microsoft Azure**: AI engineers debated the merits and pitfalls of **Google Vision OCR**, acknowledging its precision but criticizing the developer experience. Suggestions for using **Microsoft Azure OCR** and **Mindee Doctr**, potentially offering better ease of use, surfaced [here](https://huggingface.co/spaces/mindee/doctr).

**Curated Data: The Key to LLM Success**: Workshop discussions underscored the importance of fine-tuning LLMs with high-quality, curated datasets, ranging from pharma applications to technical support chatbots. Expert opinion highlighted the need for precision in data choice to maximize LLM effectiveness, spotlighting domains like drug discovery, law, sales, and interdisciplinary work.

**Axolotl Angst and Optimization**: Users faced hurdles running **Axolotl's 70B model** on M3 Macs, with overwhelming latency during local inference, pointing to deployment on Modal as a possible solution. Cost concerns with **Weights & Biases (WandB)** prompted considerations of alternatives like **Aim** and **MLflow** for economically-minded solo developers [Axolotl examples](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml).

**LLM Evaluation Deep Dive**: A session on evaluating LLMs offered a treasure trove of insights, covering product metrics, traditional and dynamic performance metrics, and tools like LangFuse and EvalGen. Recommending resources by Eugene Yan and practical examples to visualize fine-tuning, participants noted the necessity of nuanced evaluations for LLM development.

**Transcription Tangles and the Path to Summaries**: Communication around transcripts from large meetings illuminated needs for efficient summaries, exposing potential roles for LLMs. While Zoom transcripts are on the horizon, Hamel encouraged using LLMs to generate more digestible summaries, echoing wider community involvement.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Eagerly Awaiting imfo Alpha Release**: A link to a tweet by [@spectate_or](https://x.com/spectate_or/status/1795077451195830661?s=46) hinted at the upcoming release of **imfo alpha**, sparking excitement and comparisons to similar tools within the engineering community.

- **AI Task Structure Debate**: Engineers discussed categorizing **AI tasks** into retrieval and mutation types, exemplifying with queries like "Get the weight of the iPhone 15". The need for adjustments in tasks requiring sequential execution was highlighted with the insight that *"all the steps just happen at the same time."*

- **Scraping Accuracy Stumbles**: Members voiced challenges in **HTML parsing** for reliable data scraping, with complications arising from sites like Apple and Docker's release notes. Workarounds through **Playwright** for JavaScript-centric sites were considered alongside issues with Cloudflare.

- **Exploring Cost-Efficient AI Model Utilization**: The community delved into the cost-effectiveness of using various **AI models** such as Llama3 and Claude. An approach using a combined system suggested possibilities for greater savings.

- **API Functionality Quirks Highlighted**: Confusion arose around an **API output** that displayed a JSON object sans functional links, potentially linked to the absence of a **closed beta citations feature**. Additional discussions included prompts to improve video link generation and a brief inquiry about a potential API outage.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**New AI Features to Tinker With**: Stability AI announces the launch of **Stable Assistant** sporting editing features built on **Stable Diffusion 3**, boasting of improved text-to-image quality available for a free trial [here](https://stability.ai/stable-assistant), and a beta chatbot with **Stable LM 2 12B**, heralding future enhancements for text generation tasks.

**Education Merges with AI Innovation**: An upcoming 4-week course by **Innovation Laboratory**, a collaboration between Stability AI and HUG, intends to guide participants on training AI models utilizing Stability AI's framework in tandem with HUG's educational approach; sign-ups are open until June 25, 2024, accessible [here](https://www.studios.thehug.xyz/lab).

**GPU Sharing in the Spotlight**: AI engineers discuss a community-based GPU sharing proposal to decrease compute costs, with options ranging from a custom node to a potential blockchain setup designed to validate model training operations.

**SD3 Accessibility Stirs Controversy**: Discordance surfaces as members air grievances regarding **Stable Diffusion's SD3** weights not being available for local use — slating Stability AI's cloud-only approach and stirring debate over cloud-dependency and data privacy concerns.

**User Interfaces Under Comparison**: A technical discourse unfolds on the pros and cons of various interfaces for Stable Diffusion, with **ComfyUI** pitted against more user-friendly alternatives like Forge; discussions also include community tips, inpainting methods, and ways to enhance artificial intelligence workflows.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**OpenAI Forms Safety Shield**: OpenAI has established a **Safety and Security Committee** that will take charge of critical safety and security decisions across all its projects; full details can be found in their [official announcement](https://openai.com/index/openai-board-forms-safety-and-security-committee/).

**AI Muscle Flexes in Hardware Arena**: Discussions about hardware costs arose, speculating on a $200-$1000 increase due to **NPUs** (Neural Processing Units), with focus on their economic impact for high-end models.

**Plotting the Prompt Landscape**: AI engineers debated the merits of **meta-prompting** versus **Chain of Thought (CoT)**, examining the potential of using mermaid diagrams to conserve tokens and enhance output quality. There was also a sharing of improved prompts like [here](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477), showcasing practical applications of advanced prompt engineering tactics.

**Rubber Meets The Code**: Practical discussions included how AI handles **YAML, XML, and JSON** formats natively, with suggestions on using these structures for prompts to improve AI understanding and performance, and shared resources pointing to real-life prompt application for generating code and planning.

**Interactive Inconsistencies Ignite Inquiry**: Users reported issues with **ChatGPT** ranging from its refusal to draw tarot cards to context drops and unresponsiveness, spotlighting the need for improved and more predictable AI behavior.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Voice Commands Meet Robotics**: A demo video titled ["Open Source Voice-Controlled Robotic Arm"](https://www.youtube.com/watch?v=qv3bFhHoA5s) exhibits a voice-activated AI robotic arm. The perspective of democratizing robotics technology via community collaboration was forwarded.

**Bridging Modalities**: Contributions on creating early multi-modal spaces point to the use of single models and possibly stacked models with routing functionalities. For insights on such implementation, a [source link](https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py) was shared, providing a model example with practical applications.

**Deep Learning Consult on the Fly**: A user consulted the community about overcoming common pain points in training a model using Stanford Cars Dataset, managing only a 60% accuracy using ViT-B_16, with struggles involving overfitting. Meanwhile, another member is looking for help on how to better their deep learning model, indicating an environment that supports knowledge exchange for newcomers.

**Diffusers Update for Not-Just-Generation**: Hugging Face announced its **Diffusers library now supports tasks beyond generative models**, such as depth estimation and normals' prediction through **Marigold**. The update suggests an escalating trend in the versatility of diffusion models and their applications.

**Model Choices for Cyber Security Assessments**: Analysis from researchers examines the aptitude of various large language models in cyber security contexts. This provides AI engineers an angle to consider the security ramifications inherent in the deployment of LLMs.

**Robust SDXL Space Realignment**: SDXL embed space discussions underscore that newly aligned spaces default to zeroes instead of an encoded space. Such insights reflect the underlying complexity and time demands associated with realigning models to new unconditioned spaces, revealing the intricate process behind the science.

**Gradio Piques Curiosity with Upgraded Clients**: The Gradio team announced a forthcoming live event to dive into the latest features of Gradio Python and JavaScript clients. The engagement invitation emphasizes Gradio's continuous push to streamline AI integration into diverse applications through enhanced interfaces.
  
**Ambiguity in Finding an SFW Dataset**: Community chatter touches on the difficulty of locating the Nomos8k_sfw dataset, which is tied to the 4x-Nomos8kDAT model, suggesting the dataset’s limited availability or obscure placement. This highlights the occasional challenges inherent to dataset procurement.

**Launching Latest Tools for AI Storytelling**: Typeface Arc emerges as a comprehensive platform for seamlessness in creating AI-driven content. It features a tool, appropriately dubbed "Copilot", designed to amplify content creation via an interactive experience pivotal for brand narratives.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Visualize This: OpenAI Integrates with LLama!**: Engineers can now leverage **LLaVA** for visual capabilities in LM Studio by deploying it on a server and making use of the Python vision template provided.

**Speedy Model Loading on M1 Max**: AI models like **MLX and EXL2 load swiftly** on Apple's M1 Max, taking a mere 5 seconds for L3 8bit, indicating superior performance compared to GGUF Q8 which takes 29 seconds.

**LM Studio Finetuning Frustrations**: Despite being a robust environment, **LM Studio currently lacks the ability to directly fine-tune models**, with enthusiasts being pointed to alternative solutions like MLX designed for Apple Silicon.

**Budget or Bust**: AI practitioners debated the value proposition of various Nvidia GPUs, considering alternatives like the **Tesla P40/P100** and eagerly discussed rumored GPUs like the **5090** with anticipation.

**Beta Testing Blues**: As they navigate the waters of new releases, users reported problems such as **Windows CPU affinity issues** with large models and **errors on AVX2 laptops**, hinting at the complexities of configuring modern hardware for AI tasks.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-2 Gets No Love from Unsloth**: Unsloth confirmed that **GPT-2** cannot be fine-tuned using its platform due to fundamental architectural differences.

- **Fine-Tuning Frustrations with Fiery Chat**:
  - When fine-tuning llama 3 with 50,000+ email entries, members shared advice on structuring prompts for optimal input-output pairing.
  - Faced with a repeating sentence issue post-training, adding an End-Of-Sentence (EOS) token was recommended to prevent the model's overfitting or poor learning.

- **Vision Model Integration on the Horizon**: Members are keenly awaiting **Unsloth's** next-month update for vision models support, citing referrals to [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [Segment Anything](https://github.com/facebookresearch/segment-anything) for current solutions.

- **LoRA Adapters Learning to Play Nice**: The community shared tips on merging and fine-tuning LoRA adapters, emphasizing the use of resources like [Unsloth documentation on GitHub](https://github.com/unslothai/unsloth#-finetune-for-free) and exporting models to HuggingFace.

- **Coping with Phi 3 Medium's Attention Span**: Discussions on **Phi3-Medium** revealed its sliding window attention causes efficiency to drop at higher token counts, with many eager for enhancements to handle larger context windows.

- **ONNX Export Explained**: Guidance was provided for converting a fine-tuned model to **ONNX**, as seen in Hugging Face's [serialization documentation](https://huggingface.co/docs/transformers/en/serialization), with confirmation that VLLM formats are compatible for conversion.

- **Looks Like We're Going Bit-Low**: Anticipation is building for **Unsloth's** upcoming support for 8-bit models and integration capabilities with environments like Ollama, analogous to OpenAI's offerings.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Toolkit Commands for Ubuntu on Fire**: A user suggested installing the **CUDA Toolkit** from NVIDIA, checking installation with `nvidia-smi`, and offered commands for setup on Ubuntu, including via Conda: `conda install cuda -c nvidia/label/cuda-12.1.0`. Meanwhile, potential conflicts were identified with Python 3.12 and missing **triton** installation when setting up PyTorch 2.3, linked to a [GitHub issue](https://github.com/pytorch/pytorch/issues/120233).

- **GPT-4o meets its match in large edits**: Members noted that GPT-4o struggles with extensive code edits, and a new **fast apply** model aims to split the task into planning and application stages to overcome this challenge. Seeking a deterministic algorithm for code edits, a member posed the feasibility of using **vllm** or **trtllm** for future token prediction without relying on draft models. More information on this approach can be found in the [full blog post](https://cursor.sh/blog/instant-apply).

- **SYCL Debug Troubles**: A member enquired about tools to debug SYCL code, sparking a discussion on stepping into kernel code for troubleshooting.

- **Torchao's Latest Triumph**: The torchao community celebrated the merging of support for MX formats, such as `fp8/6/4`, in PyTorch, offering efficiency for interested parties, provided in part by a [GitHub commit](https://github.com/pytorch/ao/pull/264) and aligned with the [MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

- **Understanding Mixer Models in DIY**: Members dissected implementation nuances, such as integrating `dirent.h` in **llm.c**, and the importance of guarding it with `#ifndef _WIN32` for OS compatibility. The addition of a `-y 1` flag for resuming training in interruptions was implemented, addressing warnings about uninitialized variables and exploring memory optimization strategies during backward pass computation, with a related initiative found in [GitHub discussions](https://github.com/karpathy/llm.c/discussions/481).

- **Quantizing Activations in BitNet**: In the BitNet channel, it was concluded that passing incoming gradients directly in activation quantized neural networks might be erroneous. Instead, using the gradient of a surrogate function such as `tanh` was suggested, citing an [arXiv paper](https://arxiv.org/abs/1903.05662) on straight-through estimator (STE) performance.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **No Post-Learning for GPT Agents**: GPT-based agents do not learn post initial training, but can reference new information uploaded as 'knowledge files' without fundamentally altering their core understanding.
- **Efficiency Milestones in Diffusion Models**: Google DeepMind introduces **[EM Distillation](http://arxiv.org/abs/2405.16852)** to create efficient one-step generator diffusion models, and separate research from Google illustrates an 8B parameter diffusion model adept at generating high-res 1024x1024 images.
- **Scaling Down for Impact**: **[Super Tiny Language Models](https://arxiv.org/abs/2405.14159)** research focuses on reducing language model parameters by 90-95% without significantly sacrificing performance, indicating a path towards more efficient natural language processing.
- **GPU Performance Without the Guesswork**: Symbolic modeling of GPU latencies **without execution** gains traction, featuring [scholarly resources](https://inria.hal.science/hal-00789958/file/112_Lai.pdf) to guide theoretical understanding and potential impact on computational efficiency.
- **Challenging the Current with Community**: Discussions highlight community-driven projects and the importance of **collaborative problem-solving** in areas such as prompt adaptation research and implementation queries, like that of a **Facenet model** in PyTorch.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Latest Model Innovations Hit the Market**: [OpenRouter](https://openrouter.ai/models) announced new AI models, including **[Mistral 7B Instruct v0.3](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3)** and **[Hermes 2 Pro - Llama-3 8B](https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b)**, while assuring that previous versions like **[Mistral 7B Instruct v0.2](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2)** remain accessible.

- **Model Curiosity on Max Loh's Site**: Users show curiosity about the models utilized on [Max Loh's website](https://www.maxloh.com), expressing interest in identifying all uncensored models available on OpenRouter.

- **OCR Talent Show**: **Gemini's OCR** prowess was a hot topic, with users claiming its superior ability to read Cyrillic and English texts, outdoing competing models such as Claude and GPT-4o.

- **OpenRouter Token Economics**: There was clarification in the community that $0.26 allows for 1M input + output tokens on OpenRouter, and discussions emphasized how token usage is recalculated with each chat interaction, potentially inflating costs.

- **The Cost of Cutting-Edge Vision**: There is a heated exchange on **Phi-3 Vision** costs when using Azure, with some members finding the $0.07/M for llama pricing too steep, even though similar rates are noted among other service providers.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Translation Tribulations**: Discussions touched on the challenges of *translating songs* with control over lyrical tone to retain the original artistic intent. The unique difficulty lies in balancing the fidelity of meaning with musicality and artistic expression.

- **AI Infiltrates Greentext**: Members experimented with LLMs to generate **4chan greentexts**, sharing their fascination with the AI's narrative capabilities — especially when concocting a scenario where one wakes up to a world where AGI has been created.

- **Philosophical Phi and Logically Challenged LLMs**: Debates emerged over **Phi model's training data** composition, with references to "heavily filtered public data and synthetic data". Additionally, evidence of LLMs struggling with logic and self-correction during interaction was reported, raising concerns about the models' reasoning abilities.

- **Shaping Data for Machine Digestion**: AI enthusiasts exchanged resources and insights on **creating DPO datasets** and adjusting dataset formats for DPO training. Hugging Face's [TRL documentation](https://huggingface.co/docs/trl/main/en/reward_trainer) and [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) emerged as key references, alongside a [paper](https://arxiv.org/abs/2305.18290) detailing language models trained from preference data.

- **Linking Minds for RAG Riches**: Collaboration is in the air, with members sharing their intent to combine efforts on RAG-related projects. This includes the sentiment and semantic density smoothing agent project with TTS on [GitHub](https://github.com/EveryOneIsGross/densefeelsCHAT), and intentions to port an existing project to SLURM for enhanced computational management.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Loop-the-Loop in LangChain**: Engineers are troubleshooting a **LangChain agent** entering continuous loops when calling tools; one solution debate involves refining the agent's trigger conditions to prevent infinite tool invocation loops.

**Details, Please! 16385-token Error in LangChain 0.2.2**: Users report a token limit error in **LangChain version 0.2.2**, where a 16385-token limit is incorrectly applied, despite models supporting up to 128k tokens, prompting a community-lead investigation into this discrepancy.

**SQL Prompt Crafting Consultation**: Requests for **SQL agent** prompt templates with few-shot examples have been answered, providing engineers with the resources to craft queries in LangChain more effectively.

**Disappearing Act: Custom kwargs in Langserve**: Some users experience a problem where custom "kwargs" sent through **Langserve** for logging in **Langsmith** are missing upon arrival, a concern currently seeking resolution.

**Showcasing Applications**: Diverse applications developed using LangChain were shared, including frameworks for **drug discovery**, cost-saving measures for logging, enhancements for **flight simulators**, and tutorials about **routing logic** in agent flows.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Python Version Alert for Mojo Users**: Mojo users are reminded to adhere to the supported Python versions, ranging from **3.8 to 3.11**, since **3.12 remains unsupported**. Issues in Mojo were resolved by utilizing the deadsnakes repository for Python updates.

- **AI-Powered Gaming Innovations**: Engineers discussed the prospect of subscription models based on NPC intelligence in open-world games, and introducing special AI-enabled capabilities for smart devices that could lead to AI inference running locally. They explored open-world games that could feature AI-driven custom world generation.

- **Mojo Mastery**: Circular dependencies are permitted within Mojo, as modules can define each other. Traits like `Intable` and `Stringable` are inherently available, and while lambda functions are not yet a feature in Mojo, callbacks are currently utilized as an alternative.

- **Performance Pioneers**: An impressive *50x speed improvement was noted at 32 bytes in Mojo*, though it encountered cache limitations beyond that length. Benchmarks for k-means algorithms demonstrated variability due to differences in memory allocation and matrix computations, with a suggestion to optimize memory alignment for AVX512 operations.

- **Nightly Builds Nightcaps**: The latest **Mojo compiler build (2024.5.2805)** brought new features, including implementations of `tempfile.{mkdtemp,gettempdir}` and `String.isspace()`, with full changes detailed in the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and the [raw diff](https://github.com/modularml/mojo/compare/ce285fded710b403e1b7b5637183ea20fa4d5c97...4724ec6ff46378f6a1d6190ca9a76916a5faaba3). Structural sharing via references was also highlighted for its potential efficiency gains in Mojo programming.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Debugging Just Got a Level Up**: Engineers praised the **cursor interpreter mode**, highlighted for its advanced code navigation capabilities over traditional search functions in debugging scenarios.

- **A Co-Pilot for Your Messages**: **Microsoft Copilot**'s integration into **Telegram** sparked interest for its ability to enrich chat experiences with features such as gaming tips and movie recommendations.

- **GPT-2 Training on a Shoestring**: **Andrej Karpathy** showcased an economical approach to training GPT-2 in **90 minutes for $20**, detailing the process on [GitHub](https://github.com/karpathy/llm.c/discussions/481).

- **Agents and Copilots Distinguish Their Roles**: A distinction between **Copilots** and **Agents** was debated following **Microsoft Build's** categorization, with references made to [Kanjun Qiu's insights](https://www.latent.space/p/imbue) on the topic.

- **AI Podcast Delivers Cutting-Edge Findings**: An [ICLR 2024-focused podcast](https://x.com/latentspacepod/status/1795196817044594817) was released discussing breakthroughs in ImageGen, Transformers, Vision Learning, and more, with anticipation for the upcoming insights on LLM Reasoning and Agents.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Financial Geeks, Feast on FinTextQA**: [FinTextQA](https://t.co/emhQYXY1S4) is a new dataset aimed at improving long-form finance-related question-answering systems; it comprises *1,262 source-attributed Q&A pairs* across *6 different question types*.

- **Perfecting Prompt Structures**: An enquiry was made concerning resources for crafting optimal system role prompts, drawing inspiration from **LlamaIndex**'s model.

- **Chat History Preservation Tactics**: The community discussed techniques for saving chat histories within **LlamaIndex**, considering custom retrievers for **NLSQL** and **PandasQuery** engines to maintain a record of queries and results.

- **API Function Management Explored**: Strategies to handle an extensive API with over 1000 functions were proposed, favoring hierarchical routing and the division of functions into more manageable subgroups.

- **RAG System Intricacies with LlamaIndex Debated**: Technical challenges related to metadata in RAG systems were dissected, showing a divided opinion on whether to embed smaller or larger semantic chunks for optimal accuracy in information retrieval.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI Reads Between the Lines**: Members shared a laugh over SOTA AGI models' odd claims with one model's self-training assertion, "it has trained a model for us," tickling the collective funny bone. Musk's jab at [CNNs](https://x.com/elonmusk/status/1795405972145418548)—quipping "We don’t use CNNs much these days"—set off a chain of ironical replies and a nod towards vision transformer models as the new industry darlings.

**Artificial Artist's Watermark Woes**: [Corcelio's Mobius Art Model](https://huggingface.co/Corcelio/mobius) is pushing boundaries with diverse prompts, yet leaves a watermark even though it's overtaking past models in creativity. Ethical dilemmas arose from the capability of image generation systems to produce 'inappropriate' content, sparking debate on community guidelines and systems' control settings.

**Synthetic Sight Seeks Improvement**: In an effort to grapple with **SDXL**'s inability to generate images of "reading eyes," a member asked for collaborative help to build a synthetic database using DALLE, hoping to hone **SDXL**'s capabilities in this nuanced visual task.

**Patterns and Puzzles in Generative Watermarks**: Observations within the guild pointed out a recurring theme of generative models producing watermarks, indicating possible undertraining, which was found both amusing and noteworthy among the engineers.

**Elon's Eyeroll at CNNs Stokes AI Banter**: Elon Musk's tweet sent a ripple through the community, sparking jests about the obsolete nature of CNNs in today's transformative AI methodologies and the potential pivot towards transformer models.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**GPU Latency Predictions Without Benchmarks?**: Engineers discussed the potential for **symbolically modeling GPU latencies** without running kernels by considering data movement and operation times, though complexities such as occupancy and async operations were recognized as potential confounders. There's also anticipation for AMD's open-source release of MES and speculation about quant firms using cycle accurate GPU simulators for in-depth kernel optimization.

**Optimizing with Autotuners**: The community explored kernel optimization tools like **AutoTVM** and **Halide**, noting their different approaches to performance improvement; George Hotz highlighted TVM's use of XGBoost and stressed the importance of cache emulation for accurate modeling.

**Latency Hiding Mechanics in GPUs**: It was noted that GPUs employ a variety of latency-hiding strategies with their ability to run concurrent wavefronts/blocks, thus making latency modeling more complex and nuanced.

**Buffer Creation Discussions in Tinygrad**: The #learn-tinygrad channel had members inquiring about using **post dominator analysis** in scheduling for graph fusion efficiency and the creation of **LazyBuffer** from arrays, with a suggestion to use `Load.EMPTY -> Load.COPY` for such scenarios.

**Code Clarity and Assistance**: Detailed discussions were had regarding buffer allocation and `LazyBuffer` creation in Tinygrad, with one member offering to provide **code pointers** for further clarification and understanding.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Elevenlabs Voices Come to AI Town**: Integrating **Elevenlabs**' text-to-speech capabilities, AI Town introduced a feature allowing conversations to be heard, not just read, with a minor delay of about one second, challenging real-time usage. The implementation process involves [transforming text into audio](https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19) and managing audio playback on the frontend.

- **Bring Science Debate to AI Chat**: A concept was shared about utilizing AI chatbots to simulate science debates, aiming to foster engagement and demonstrate the unifying nature of scientific discussion.

- **Audio Eavesdropping Added for Immersion**: The Zaranova fork of AI Town now simulates eavesdropping by generating audio for ambient conversations, potentially amplifying the platform's interactivity.

- **Collaborative Development Rally**: There's an active interest from the community in contributing to and potentially merging new features, such as text-to-speech, into the main AI Town project.

- **Addressing User Experience Issues**: A user experienced difficulties with the conversations closing too quickly for comfortable reading, hinting at potential user interface and accessibility improvements needed within AI Town.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Slimming Down on Logs**: A new pipeline developed by a member removes **redundant logs** to reduce costs. They recommended a [tool](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere) for selecting a "verbose logs" pipeline to achieve this.

- **Debating Deployment**: Members discussed cloud-prem deployment solutions for **reranking and query extraction**, seeking insights on the best integrated practices without providing further context.

- **Financial RAG Fine-tuning**: There was an inquiry on the possibility of **fine-tuning Cohere models** to answer financial questions, specifically mentioning the integration with **RAG (Retrieve and Generate) systems** using SEC Filings.

- **Aya23 Model's Restrictive Use**: It was clarified that **Aya23 models** are strictly for research purposes and are not available for commercial use, affecting their deployment in startup environments.

- **Bot Plays the Game**: A member launched a **Cohere Command R** powered gaming bot, **Create 'n' Play**, featuring *"over 100 text-based games"* aimed at fostering social engagement on Discord. The project's development and purpose can be found in a [LinkedIn post](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Inference vs. Training Realities**: The conversation underscored performance figures in AI training, particularly regarding how a seemingly simple query about "inference only" topics quickly lead to complex areas focused on training's computational requirements.

- **FLOPS Define Training Speed**: A key point in the discussion was that AI model training is, in practice, constrained by floating-point operations per second (FLOPS), especially when employing techniques like teacher forcing which increase the effective batch size.

- **Eager Eyes on Hopper Cards for FP8**: The community showed enthusiasm about the potential of **Hopper** cards for fp8 native training, highlighting a keen interest in leveraging cutting-edge hardware for enhanced training throughput.

- **Eradicating Version Confusion with fschat**: Members were advised to fix **fschat** issues by reinstallation due to erroneous version identifiers, pointing to meticulous attention to detail within the collective's ecosystem.

- **When CUTLASS Is a Cut Above**: Discussions clarified the importance of setting `CUTLASS_PATH`, emphasizing CUTLASS's role in optimizing matrix operations vital for deep learning, underscoring the guild’s focus on optimizing algorithmic efficiency.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apache Welcomes YI and YI-VL Models**: The **YI and YI-VL (multimodal LLM) models** are now under the **Apache 2.0** license, as celebrated in a [tweet by @_philschmid](https://fxtwitter.com/_philschmid/status/1795343334225129570); they join the 1.5 series in this licensing update.

- **Gemini 1.5 Challenges the Throne**: **Gemini 1.5 Pro/Advanced** has climbed to #2 on the ranking charts, with ambitions to overtake GPT-4o, while **Gemini 1.5 Flash** proudly takes the #9 spot, edging out **Llama-3-70b**, as announced in a [tweet from lmsysorg](https://x.com/lmsysorg/status/1795512202465845686?s=46).

- **OpenAI's Board Left in the Dark**: A former OpenAI board member disclosed that the board wasn't informed about the release of **ChatGPT** in advance, learning about it through [Twitter](https://fxtwitter.com/bilawalsidhu/status/1795534345345618298) just like the public.

- **Toner Drops Bombshell on OpenAI's Leadership**: Helen Toner, a previous member of OpenAI's board, accused **Sam Altman** of creating a toxic work environment and acting dishonestly, pushing for "external regulation of AI companies" during a [TED podcast episode](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3).

- **Community Aghast at OpenAI's Revelations**: In reaction to Helen Toner's grave allegations, the community expressed shock and anticipation about the prospect of significant industry changes, highlighted by Natolambert querying if Toner might "literally save the world?"



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Go-To LLM Leaderboard Approved by Experts**: The [leaderboard at chat.lmsys.org](https://chat.lmsys.org/?leaderboard) was highlighted and endorsed by users as a reliable resource for comparing the performance of various large language models (LLMs).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Securing Local AI Endpoints Is Crucial**: One member **highlighted the importance of securing local endpoints** for AI models, suggesting the use of **DNS SRV records** and public keys to ensure validated and trustworthy local AI interactions, jesting about the perils of unverified models leading to unintended country music purchases or squirrel feeding.
- **Troubleshoot Alert: Llamafile Error Uncovered**: A user running a **Hugging Face llamafile** - specifically `granite-34b-code-instruct.llamafile` - reported an error with an "unknown argument: --temp," indicating potential issues within the implementation phase of the model deployment process.
- **Focus on the Running Model**: In a clarification, it was noted that whatever model is running locally at `localhost:8080` (like *tinyllama*) would be the default, with the `model` field in the chat completion request being inconsequential to the operation. This suggests a **single-model operation paradigm** for **llamafiles** in use.
  
**Link mentioned**: [granite-34b-code-instruct.llamafile](https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile/resolve/main/granite-34b-code-instruct.Q5_0.llamafile?download=true)



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Request for R1 Update**: A member expressed anticipation for the **R1's** future developments, humorously referring to it as a potential "nice paperweight" if it doesn't meet expectations.
- **Community Seeks Clarity**: There's a sense of shared curiosity within the community regarding updates related to **R1**, with members actively seeking and sharing information.
- **Awaiting Support Team's Attention**: An inquiry to the **OI team** concerning an email awaits a response, signifying the need for improved communication or support mechanisms.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Spotting a Ghost Town**: A member raised the concern that the server appears **unmoderated**, which could indicate either an oversight or an intentional laissez-faire approach by the admins.
- **Notification Fails to Notify**: An attempted use of the **@everyone** tag in the server failed to function, suggesting restricted permissions or a technical snafu.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LLM for Backend Automation Inquiry Left Hanging**: A member's curiosity about whether a course covers automating backend services using Large Language Models (LLM) remained unanswered. The inquiry sought insights into practical applications of LLMs in automating backend processes.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1244727054499582113)** (91 messages🔥🔥): 

- **Google Vision OCR likes and dislikes**: Several members discussed **Google Vision OCR** where it was noted for its decent results and detailed confidence metrics at the character level, though the **developer experience was critiqued** as horrible. Alternatives like **Microsoft Azure** and open-source **Mindee Doctr** were mentioned as either better or simpler solutions.
- **Gradio office hours hype**: Hugobowne announced an upcoming office hours session with Freddy Boulton, prompting excitement and some jokes about AI swag like **Scikit hoodies** and **Mistral t-shirts**. Links were shared to [Freddy's website](https://www.freddyboulton.com/) and a [YouTube tutorial](https://youtu.be/IVJkOHTBPn0?si=tsM6PouRRNixaroH) on building a multimodal chatbot component.
- **Modal as an all-encompassing cloud service**: Charles highlighted **Modal's full-stack capabilities** for data ETL, fine-tuning, inference, web hosting, and more, positioning it as a serverless solution for a variety of tasks. This spurred discussions about **full-stack app** efficiency and examples were shared for [S3 bucket mounts](https://modal.com/docs/examples/s3_bucket_mount) and [web scraping](https://modal.com/docs/examples/web-scraper).
- **Choosing LLM libraries for building agents**: Lalithnarayan and Chongdashu discussed the multitude of choices like **Langchain**, **LlamaIndex**, and **DSPy** for building LLM applications and agents. They concluded with advice to start with Langchain v0.1 and upgrade as necessary, given the recent breaking changes in v0.2.
- **Data mixture for pretraining**: Thechurros asked about the correct **data mixture** when continuing pre-training with synthetic data. Jeremy Howard suggested a rough guideline of 20% existing data and mentioned using curated common crawl subsets, highlighting the lack of definitive research and prompting mentions of related works like the [Zephyr data mixture study](https://arxiv.org/html/2402.16827v2).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15682">The Road Less Scheduled</a>: Existing learning rate schedules that do not require specification of the optimization stopping step T are greatly out-performed by learning rate schedules that depend on T. We propose an approach tha...</li><li><a href="https://x.com/NielsRogge/status/1795106366752723094?t=2Fe5vPhNOJF-84AkgrajTw&s=19">Tweet from Niels Rogge (@NielsRogge)</a>: Turns out my Idefics2 notebook works just as well for PaliGemma fine-tuning :) find it here: https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma  For JSON use cases, a tiny VLM ...</li><li><a href="https://tenor.com/view/major-payne-dance-the-robot-dancing-moves-gif-17644148">Major Payne Dance GIF - Major Payne Dance The Robot - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.freddyboulton.com/">Freddy A. Boulton</a>: no description found</li><li><a href="https://github.com/eugeneyan/visualizing-finetunes/blob/main/1_prep_data.ipynb">visualizing-finetunes/1_prep_data.ipynb at main · eugeneyan/visualizing-finetunes</a>: Contribute to eugeneyan/visualizing-finetunes development by creating an account on GitHub.</li><li><a href="https://www.dmlbl.com/technical_blog.html">Technical Blog</a>: no description found</li><li><a href="https://huggingface.co/spaces/mindee/doctr">docTR - a Hugging Face Space by mindee</a>: no description found</li><li><a href="https://python.langchain.com/v0.1/docs/modules/agents/">Agents | 🦜️🔗 LangChain</a>: The core idea of agents is to use a language model to choose a sequence of actions to take.</li><li><a href="https://arxiv.org/html/2402.16827v2">A Survey on Data Selection for Language Models</a>: no description found</li><li><a href="https://github.com/VikParuchuri/surya">GitHub - VikParuchuri/surya: OCR, layout analysis, reading order, line detection in 90+ languages</a>: OCR, layout analysis, reading order, line detection in 90+ languages - VikParuchuri/surya</li><li><a href="https://modal.com/docs/examples/s3_bucket_mount">Analyze NYC yellow taxi data with DuckDB on Parquet files from S3</a>: This example shows how to use Modal for a classic data science task: loading table-structured data into cloud stores, analyzing it, and plotting the results.</li><li><a href="https://modal.com/docs/examples/web-scraper">A simple web scraper</a>: In this guide we’ll introduce you to Modal by writing a simple web scraper. We’ll explain the foundations of a Modal application step by step.</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1244778919274745936)** (10 messages🔥): 

- **Honing LLMs for Pharma Sector**: A discussion revolves around five compelling use cases for LLMs within the pharma/biotech sector, such as *Accelerated Drug Discovery* and *Personalized Medicine*. Each use case emphasizes fine-tuning with relevant datasets and leveraging **RAG** for contextual and specific information.

- **Legal Document Summarization**: There is an interest in leveraging LLMs to *summarize discovery documents for legal proceedings*. Fine-tuning is suggested as crucial for adapting the model to the specific style and relevance criteria of legal summaries.

- **Customized Email Openers for Sales**: One member discusses generating *personalized first liners for sales emails*. Fine-tuning the model with a dataset of successful email openers and profiles of recipients is proposed to enhance engagement and response rates.

- **Interdisciplinary Collaboration via Multi-Agent LLMs**: The idea of creating a multi-agent LLM model, each specializing in a niche area, is introduced for solving complex interdisciplinary problems. This setup would involve RAG for additional context and fine-tuning each agent for their specific domain.

- **Chatbots for Technical Support and Incident Diagnosis**: Discussion includes training models for chatbots tailored for *answering technical questions* from historical Slack & Jira data and diagnosing active incidents using post-mortem documentation. Fine-tuning is suggested for enhancing the efficacy of these chatbots.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1244727466719842385)** (9 messages🔥): 

- **Circleback Transcript Troubles**: A user mentioned that Hamel had been using Circleback for transcription and notes but could not find the link to access them. Hamel replied that Circleback isn't transcribing these huge meetings but noted it's possible to export the transcripts, asking for help in uploading them to the course lessons.

- **Zoom Transcripts Coming Soon**: Dan stated that Zoom uses a separate step/job to create transcripts and he has kicked off the jobs for all sessions. He mentioned that he would upload the transcripts on the course page in the afternoon and provide an update, clarifying that these would be transcripts rather than summaries.

- **Opportunity for LLM Summaries**: Hamel suggested that creating summaries from the transcripts could be a good opportunity for someone to use LLMs. This implies there is room for contributions by course participants in refining the content.

- **Late Joiner Seeking Guidance**: Shalini, a new member from Kolkata, inquired about the course's progress and what she should follow apart from catching up on recordings, as she joined in Week 3.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1244759662306529331)** (87 messages🔥🔥): 

- **DeepSpeed Config Stumps Llama3-70B Trainers**: A user struggled with **OOM errors while training Llama3-70B** on multiple GPUs, due to suspected misconfigured DeepSpeed settings. They received advice to reduce batch size, modify sequence length, and adjust target layers in their config files, while also sharing relevant [WandB runs](https://wandb.ai/dailyco/khk-llama-3-70b) and [configurations](https://gist.github.com/kwindla/bea28ce3ffe10e130dbd272e2fc6037f).

- **Debugging Model Configuration for Modal**: Users worked through different strategies including setting the gradient accumulation to `1` and turning off evaluations to debug OOM issues. They continually iterated on config settings and shared runs to better understand memory allocation.

- **Getting Lorax Running on Modal**: A user successfully ran **Lorax** on Modal after addressing Dockerfile ENTRYPOINT issues by clearing the existing entrypoint and wrapping the call to `lorax-launcher` in a `@modal.web_server` decorator. [Reference Code](https://github.com/predibase/lorax/blob/main/Dockerfile).

- **Choosing GPU Instances for Training**: A general query on heuristics for choosing the right instance type was addressed, suggesting starting with **A10G GPUs** if VRAM limits are met, with a potential future shift to **L40S GPUs**.

- **Caching Mechanism for Model Weights**: A user sought confirmation on their method of caching model weights using Modal's caching mechanism and detailed their setup for feedback, which received validation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/kwindla/81cbec28a5893f682984549ecc05dcfa">Llama-3-70B config (OOM)</a>: Llama-3-70B config (OOM). GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b?nw=nwuserkwindla">dailyco</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/ybeu4z50/logs?nw=nwuserkwindla">dailyco</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - Config options</a>: no description found</li><li><a href="https://github.com/predibase/lorax/blob/main/Dockerfile">lorax/Dockerfile at main · predibase/lorax</a>: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs - predibase/lorax</li><li><a href="https://github.com/huggingface/peft/blob/39c60ffca9c1d1cc606a16654cfe9cd66b363a70/src/peft/tuners/lora/config.py#L51-L58)">peft/src/peft/tuners/lora/config.py at 39c60ffca9c1d1cc606a16654cfe9cd66b363a70 · huggingface/peft</a>: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft</li><li><a href="https://modal.com/docs/guide/custom-container#entrypoint">Custom containers</a>: This guide walks you through how to define the environment your Modal functions and applications run within.</li><li><a href="https://modal.com/docs/guide/webhooks#non-asgi-web-servers">Web endpoints</a>: Modal gives you a few ways to expose functions as web endpoints. You can turn any Modal function into a web endpoint with a single line of code, or you can serve a full app using frameworks like FastA...</li><li><a href="https://gist.github.com/mtisz/2e9f7d8acb1a65f0b58f2427a402f387">Axolotl Config for Llama-3-70B QLoRA</a>: Axolotl Config for Llama-3-70B QLoRA. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b">dailyco</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/8pdffbhe">dailyco</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA semantics &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://gist.github.com/kwindla/bea28ce3ffe10e130dbd272e2fc6037f">Llama-3-70B config (works on one GPU no deepspeed; OOMs on multiple GPUs during merge)</a>: Llama-3-70B config (works on one GPU no deepspeed; OOMs on multiple GPUs during merge) - khk-llama-3-70B.yml</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/80s40cgd">dailyco</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/9vrwylua">dailyco</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/8tk2wy4k?nw=nwuserkwindla">dailyco</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1244732945080520724)** (11 messages🔥): 

- **Prince Canuma drops LLaMA 3 finetuning resources**: A user shared that **Prince Canuma** has released a new video and weights on refining **LLaMA 3** from 8B to 6B [link to video](https://youtu.be/tMvC_bsAwyQ?si=23eN1WIK5Izsep80).
  
- **OpenAI finetuning guide for beginners**: It's suggested that beginners refer to the [OpenAI fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning/when-to-use-fine-tuning) to understand when fine-tuning is appropriate.

- **Struggling with .pth to safetensors conversion**: A user requested help with converting a **finetuned spam classifier** from a .pth file to **safetensors** format for hosting on Hugging Face. They were directed to [Hugging Face documentation](https://huggingface.co/docs/safetensors/en/convert-weights#) and provided additional advice.
  
- **Upload model to Hugging Face hub first**: It was also advised to first upload the model to the **Hugging Face hub** before attempting local conversions, with further guidance pointing to the **convert.py** file in the [Hugging Face repository](https://huggingface.co/spaces/safetensors/convert/tree/main).

- **Keep HF tokens secure**: A user was informed that their **Hugging Face token** was unintentionally exposed, and this discussion was suggested to be moved to a private channel.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/safetensors/en/convert-weights#">Convert weights to safetensors</a>: no description found</li><li><a href="https://youtu.be/tMvC_bsAwyQ?si=23eN1WIK5Izsep80">Coding Llama 3 from scratch in PyTorch - Part 2</a>: In this video series, you will learn how to train and fine-tune Llama 3 model from scratch.The goal is to code LLaMA 3 from scratch in PyTorch to create mode...</li><li><a href="https://huggingface.co/spaces/safetensors/convert/tree/main">safetensors/convert at main</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1244726841873403924)** (26 messages🔥): 

- **Cache your Hugging Face models smartly**: A user shared tips for caching Hugging Face models and datasets to specific directories to avoid running out of storage. They recommended setting environment variables like `HF_DATASETS_CACHE` and `HUGGINGFACE_HUB_CACHE`.

- **Accessing additional logs for troubleshooting**: Members discussed difficulties in retrieving additional logs for errors, especially when instances restart unexpectedly. It was suggested to run long-running jobs on the JupyterLab terminal for clearer logs or convert notebooks to Python scripts.

- **Conda environments reset upon pausing**: Users reported their conda environments being deleted when instances are paused and resumed. The issue appears linked to environments saved in the `/root` directory, and it was recommended to save them in custom paths.

- **Managing credentials across different emails**: Questions arose about using different emails for JarvisLabs and other platforms, causing issues with credits. The resolution was to ensure the same email is used across all relevant platforms and is registered with course instructors for credits.

- **Automating instance shutdowns**: There was a discussion about scripting the startup and shutdown of instances for resource conservation. It emerged that `shutdown -h` might not work directly, and API usage was suggested for full automation.

**Link mentioned**: <a href="https://jarvislabs.ai/docs/env#creating-and-managing-a-new-environment-using-conda">Create custom environment | Jarvislabs</a>: You may want to create and maintain separate virtual environments as your project gets more complicated.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1245005625658708030)** (19 messages🔥): 

- **HF credits meant for individuals, not organizations**: Members clarified that **grants are intended for individuals** taking the course, not for organizations. Even so, it is acceptable to have an account that is a member of an organization.
- **Ensure HF credits are applied by Friday**: A member reminded everyone that **credits will be distributed by Friday** after enrollment closes. They stressed the importance of filling out the HF form sent via email to ensure credits are correctly applied.
- **Trouble converting PyTorch model to safetensors**: A user discussed their frustration with converting a **PyTorch model to safetensors** for production. They mentioned following a tutorial from [GitHub - LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and raised questions about handling the specific file formats and inference code required.

**Link mentioned**: <a href="https://github.com/rasbt/LLMs-from-scratch">GitHub - rasbt/LLMs-from-scratch: Implementing a ChatGPT-like LLM in PyTorch from scratch, step by step</a>: Implementing a ChatGPT-like LLM in PyTorch from scratch, step by step - rasbt/LLMs-from-scratch

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1244929654767358031)** (6 messages): 

- **Choose GitHub email for Replicate setup**: It was clarified that for signing up on Replicate, the email associated with your GitHub account should be used. For preferred email settings, users are advised to use the email already associated with their Replicate account to avoid confusion.
- **Replicants Recognized**: A self-identified "Replicate person" humorously noted they are now called "Replicants." This highlights the inclusive and fun culture within the Replicate community.
- **Email confusion clarified**: After signing up with GitHub, a user expressed confusion over which email will be checked for Replicate credits. It was confirmed by a Replicant that tracking can be done via the Replicate (GitHub) username and to DM if any issues arise.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1245072716336463952)** (1 messages): 

- **Latencies Love In Langsmith**: One member expressed appreciation for the latency graphs in Langsmith, raising two questions. They inquired whether there could be *historical latency graphs of all models* rather than just those in use, and how geographical distribution of models affects latency, asking about dependencies on zones for **OpenAI, Anthropic, and Google** models.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1244866175784189963)** (3 messages): 

- **OpenPipe limits data formats**: A member noted that OpenPipe currently only accepts external data in the **OpenAI chat fine-tuning format**, which restricts the ability to upload JSONL in Alpaca format. They expressed interest in the dataset creation interface seen in a recent talk.
- **Future data format support likely**: Another member responded to the query by clarifying that **additional data formats** might be supported in the future. This decision was made because the current format is what "most of our users were most familiar with."
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1244749171144261694)** (64 messages🔥🔥): 

- **Fine-Tuning with High-Quality Data Crucial**: "You want to feed it high-quality stuff," emphasizes the importance of high-quality data for fine-tuning, not altering the objective function, but possibly the optimizer and hyperparameters. High-quality data maximizes the pre-built model's application-specific performance.

- **Loss Curve Interpretation and Overfitting Concerns**: Discussion on Mistral8x7b fine-tuning highlights the issue of overfitting based on curve interpretation, recommending validation loss usage. Participants also debate learning rate adjustments, with suggestions pointing to a potentially too-high base learning rate.

- **Optimization Issues and Configuration Tweaks**: Exploring reasons for increased and plateaued loss in training, suggestions include possibly high learning rates and using better initialization or configs like `rslora`. Participants share Weights & Biases (wandb) run links for collaborative debugging.

- **Curated and Synthetic Dataset Challenges**: Rumbleftw and others discuss complexities with curated datasets and specific configurations for fine-tuning recent model releases. They address tokenizer issues, appropriate special tokens, and handle a dataset of around `165k` data points.

- **Shared Resources and Configuration Advice**: Community members share useful resources, including workshop notes from [Cedric Chee's GitHub Gist](https://gist.github.com/cedrickchee/6e9cff188d24a5b4429af1845f912688), and discuss zero3 and zero3_bf16 deep-speed configurations from [OpenAccess-AI-Collective's GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json). The importance of proper configuration to avoid out-of-memory issues during model parallelism is also highlighted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/settings">settings</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/vapi/mistral-func/reports/Func-calling-Mistral8x7bv0-3-">vapi</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/vapi/mistral-func/reports/Func-calling-Mistral8x7bv0-3--Vmlldzo4MTE3MjAz?accessToken=3qbn8ulplg2igvgts7fwnqgoekzyubyz191mb6y8jxntdyv44zmw6s9l55pemue9">Func-calling: Mistral8x7bv0.3</a>: Publish your model insights with interactive plots for performance metrics, predictions, and hyperparameters. Made by Rajdeep Ghosh using Weights &amp; Biases</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json">axolotl/deepspeed_configs/zero3_bf16.json at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://wandb.ai/vapi/mistral-func/reports/Func-calling-Mistral8x7bv0-3--Vmlldzo4MTE3MjAz?accessToken=3qbn8ulplg2igvgts7fwnqgoekzyubyz191mb6y8jxntdyv44zmw6s9l55pemue9#axolotl-config">Func-calling: Mistral8x7bv0.3</a>: Publish your model insights with interactive plots for performance metrics, predictions, and hyperparameters. Made by Rajdeep Ghosh using Weights &amp; Biases</li><li><a href="https://gist.github.com/">Discover gists</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/cedrickchee/6e9cff188d24a5b4429af1845f912688">Fine-Tuning Workshop 2: Fine-Tuning with Axolotl</a>: Fine-Tuning Workshop 2: Fine-Tuning with Axolotl. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/jingkaihe/llm-finetuning?tab=readme-ov-file#fine-tuning-on-promql-data">GitHub - jingkaihe/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - jingkaihe/llm-finetuning</li><li><a href="https://github.com/jingkaihe/llm-finetuning/blob/main/data/promql.tiny.jsonl">llm-finetuning/data/promql.tiny.jsonl at main · jingkaihe/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - jingkaihe/llm-finetuning</li><li><a href="https://github.com/jingkaihe/llm-finetuning/blob/main/config/mistral-promql.yml">llm-finetuning/config/mistral-promql.yml at main · jingkaihe/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - jingkaihe/llm-finetuning</li><li><a href="https://wandb.ai/jingkaihe/memorize-sqlqa/reports/train-loss-24-05-28-10-26-55---Vmlldzo4MTIxNzI3">Weights & Biases</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1245058358675767470)** (461 messages🔥🔥🔥): 

- **Awesome session on LLM Evals**: Eugene Yan shared a thorough walkthrough of evaluating LLMs, covering iterating on evaluations and how they interlink with product metrics. Check out his [visualizing finetunes repository](https://github.com/eugeneyan/visualizing-finetunes) for detailed notebooks.
- **Data logging and evaluation tools emphasized**: Tools like [LangFuse](https://langfuse.com), [ChainForge](https://chainforge.ai/), and [EvalGen](https://arxiv.org/abs/2404.12272) were highlighted for their potential to make tracing, logging, and evaluations more effective.
- **Concerns on performance metrics**: Discussion highlighted the challenges of traditional metrics like BLEU and the need for dynamic, task-specific evaluations as outlined in [Eugene Yan's write-up](https://eugeneyan.com/writing/evals/).
- **Engaging practical examples**: The session included practical examples with detailed methodologies and technical insights, with resources like a [notebook series](https://github.com/eugeneyan/visualizing-finetunes) elucidating fine-tuning processes.
- **Rich resource sharing**: The session and discussion brought together a wealth of resource links, from practical guidance articles like Eugene's piece on [prompting fundamentals](https://eugeneyan.com/writing/prompting/) to the latest research like the [Reversal Curse](https://arxiv.org/abs/2309.12288).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://johnowhitaker.]">no title found</a>: no description found</li><li><a href="https://langfuse.com/">Langfuse</a>: Open source LLM engineering platform - LLM observability, metrics, evaluations, prompt management.</li><li><a href="https://x.com/eugeneyan">Tweet from undefined</a>: no description found</li><li><a href="https://arxiv.org/abs/2309.12288">The Reversal Curse: LLMs trained on &#34;A is B&#34; fail to learn &#34;B is A&#34;</a>: We expose a surprising failure of generalization in auto-regressive large language models (LLMs). If a model is trained on a sentence of the form &#34;A is B&#34;, it will not automatically generalize...</li><li><a href="https://news.ycombinator.com/item?id=37843907">no title found</a>: no description found</li><li><a href="https://github.com/shreyashankar">shreyashankar - Overview</a>: CS PhD student at UC Berkeley. shreyashankar has 63 repositories available. Follow their code on GitHub.</li><li><a href="https://x.com/BEBischof">Tweet from undefined</a>: no description found</li><li><a href="https://forums.fast.ai/">fast.ai Course Forums</a>: Forums for fast.ai Courses, software, and research</li><li><a href="https://github.com/eugeneyan/visualizing-finetunes">GitHub - eugeneyan/visualizing-finetunes</a>: Contribute to eugeneyan/visualizing-finetunes development by creating an account on GitHub.</li><li><a href="https://eugeneyan.com/writing/prompting/">Prompting Fundamentals and How to Apply them Effectively</a>: Structured input/output, prefilling, n-shots prompting, chain-of-thought, reducing hallucinations, etc.</li><li><a href="https://arxiv.org/abs/2401.03038">SPADE: Synthesizing Data Quality Assertions for Large Language Model Pipelines</a>: Large language models (LLMs) are being increasingly deployed as part of pipelines that repeatedly process or generate data of some sort. However, a common barrier to deployment are the frequent and of...</li><li><a href="https://arxiv.org/abs/2404.12272">Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences</a>: Due to the cumbersome nature of human evaluation and limitations of code-based evaluation, Large Language Models (LLMs) are increasingly being used to assist humans in evaluating LLM outputs. Yet LLM-...</li><li><a href="https://pytest-vcr.readthedocs.io/en/latest/#quick-start">Welcome to pytest-vcr - pytest-vcr</a>: no description found</li><li><a href="https://arxiv.org/abs/2305.14296">USB: A Unified Summarization Benchmark Across Tasks and Domains</a>: While the NLP community has produced numerous summarization benchmarks, none provide the rich annotations required to simultaneously address many important problems related to control and reliability....</li><li><a href="https://docs.google.com/presentation/d/1GC868XXjhxOpQEt1jUM79aW0RHjzxPp0XhpFHnYH760/edit#slide=id.p">Spellgrounds for Prodigious Prestidigitation</a>: Spellgrounds for Prodigious Prestidigitation Dr. Bryan Bischof, Head of AI @ Hex</li><li><a href="https://x.com/HamelHusain/status/1795526367637049629">Tweet from Hamel Husain (@HamelHusain)</a>: My colleagues and I distilled practical advice re: LLMs into this three-part series. Lot&#39;s of bangers.  One of my favorite excerpts from this part in the screenshot  Advice from: @eugeneyan, @BEBi...</li><li><a href="https://www.youtube.com/watch?v=eGVDKegRdgM">Scaling Up “Vibe Checks” for LLMs - Shreya Shankar | Stanford MLSys #97</a>: Episode 97 of the Stanford MLSys Seminar Series!Scaling Up “Vibe Checks” for LLMsSpeaker: Shreya ShankarBio:Shreya Shankar is a PhD student in computer scien...</li><li><a href="https://arxiv.org/abs/2404.13076">LLM Evaluators Recognize and Favor Their Own Generations</a>: Self-evaluation using large language models (LLMs) has proven valuable not only in benchmarking but also methods like reward modeling, constitutional AI, and self-refinement. But new biases are introd...</li><li><a href="https://sqlmodel.tiangolo.com/">SQLModel</a>: SQLModel, SQL databases in Python, designed for simplicity, compatibility, and robustness.</li><li><a href="https://chainforge.ai/">ChainForge: A visual programming environment for prompt engineering</a>: no description found</li><li><a href="https://www.traceloop.com/docs/openllmetry">What is OpenLLMetry? - traceloop</a>: no description found</li><li><a href="https://www.amazon.co.uk/Noise-Daniel-Kahneman/dp/0008308993">no title found</a>: no description found</li><li><a href="https://discord.gg/yX2TdaFt8t">Join the llm-fine-tuning Discord Server!</a>: Check out the llm-fine-tuning community on Discord - hang out with 1468 other members and enjoy free voice and text chat.</li><li><a href="https://hamel.dev/blog/posts/evals/#automated-evaluation-w-llms">- Your AI Product Needs Evals</a>: How to construct domain-specific LLM evaluation systems.</li><li><a href="https://eugeneyan.com/writing/evals/">Task-Specific LLM Evals that Do & Don't Work</a>: Evals for classification, summarization, translation, copyright regurgitation, and toxicity.</li><li><a href="https://tenor.com/view/im-proud-of-you-dan-levy-david-david-rose-schitts-creek-gif-20773745">Im Proud Of You Dan Levy GIF - Im Proud Of You Dan Levy David - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/waiting-still-gif-20331665">Waiting Still GIF - Waiting Still - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>: no description found</li><li><a href="https://hex.tech/">Hex - Magical tools for working with data together</a>: Hex is a modern data platform for data science and analytics. Collaborative data notebooks, stunning data apps, magical AI-assist, and enterprise-grade security.</li><li><a href="https://hex.tech/product/magic-ai/">Hex Magic | Smarter, faster analysis with a little Magic | Hex </a>: Save hours every week by using Magic AI to write queries, build charts, and fix bugs.</li><li><a href="https://www.youtube.com/watch?v=eGVDKegRdgM&t=139s">Scaling Up “Vibe Checks” for LLMs - Shreya Shankar | Stanford MLSys #97</a>: Episode 97 of the Stanford MLSys Seminar Series!Scaling Up “Vibe Checks” for LLMsSpeaker: Shreya ShankarBio:Shreya Shankar is a PhD student in computer scien...</li><li><a href="https://x.com/tomaarsen/status/1795425797408235708">Tweet from tomaarsen (@tomaarsen)</a>: ‼️Sentence Transformers v3.0 is out! You can now train embedding models with multi-GPU training, bf16 support, loss logging, callbacks & much more. I also release 50+ datasets to train on & much more....</li><li><a href="https://arize.com/blog/breaking-down-evalgen-who-validates-the-validators/">Breaking Down EvalGen: Who Validates the Validators?</a>: Everything you need to know about EvalGen, an approach to LLM-assisted evaluation. Also includes some takeaways for LLM app builders.</li><li><a href="https://www.youtube.com/watch?v=ua93WTjIN7s">LlamaIndex Workshop: Evaluation-Driven Development (EDD)</a>: ​In this workshop, we teach you how to do &quot;Evaluation Driven Development&quot; (EDD) to build LLM apps for production. This consists of the following:1. ​Defining...</li><li><a href="https://www.usebraintrust.com/">Braintrust | The First User-Owned Talent Network</a>: Braintrust connects organizations with top technical talent to complete strategic projects and drive innovation. </li><li><a href="https://github.com/traceloop/openllmetry">GitHub - traceloop/openllmetry: Open-source observability for your LLM application, based on OpenTelemetry</a>: Open-source observability for your LLM application, based on OpenTelemetry - traceloop/openllmetry</li><li><a href="https://johnowhitaker.dev/dsc/2024-01-23-tips.html">johnowhitaker.dev – A few tips for working on high-surface-area problems</a>: no description found</li><li><a href="https://www.traceloop.com/docs/openllmetry/introduction">What is OpenLLMetry? - traceloop</a>: no description found</li><li><a href="https://www.langchain.com/langsmith">LangSmith</a>: Get your LLM app from prototype to production.</li><li><a href="https://pydantic.dev/logfire">Pydantic Logfire | Uncomplicated observability</a>: Logfire is a new type of observability platform built on the same belief as Pydantic — that the most powerful tools can be easy to use.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1244895249273454643)** (2 messages): 

- **Understanding Colbert's Output**: A member admitted it’s unclear what comes out of **Colbert** and plans to run the code to check. They seem eager to gain firsthand insights into its results.

- **Interest in Discussing Sparse Embeddings and M3**: A member expressed a desire to delve into discussions about **sparse embeddings and M3**, even though it's considered "rag basics". This indicates an interest in expanding beyond the conventional topics typically covered.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1244767005664608338)** (31 messages🔥): 

- **Loading error with Axolotl's 70B model**: A user encountered an error while loading checkpoint shards for the 70B model using two RTXA6000 GPUs. The error was related to "torch.distributed.elastic.multiprocessing.errors.ChildFailedError", causing the process to fail at 93% completion.
- **Cost concerns with WandB services**: Multiple users discussed the high cost of WandB for extensive usage, with alternatives such as **Aim** and self-hosted **MLflow** suggested for their cost-effectiveness. One user mentioned the significant benefit of these tools is mainly for collaboration, recommending simpler solutions for solo developers.
- **Preference for WandB**: Despite cost, some users prefer WandB for its user-friendliness compared to other tools.
- **Google Colab Debug for Axolotl**: A user has made a [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1662) to fix issues with running Axolotl in Google Colab notebooks, including updates to configurations and installation steps.
- **Inference discrepancies with TinyLlama**: Users reported inconsistent outputs when performing inference with TinyLlama models post-training. Potential issues included improper prompting and disparities between training and inference setups detailed by config file examination and discussions on **sample packing** optimization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml">axolotl/examples/llama-3/qlora-fsdp-70b.yaml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1662">Fix Google Colab notebook 2024-05 by maciejgryka · Pull Request #1662 · OpenAccess-AI-Collective/axolotl</a>: A couple of fixes to make the Google Colab notebook run (tried on an L4 GPU):  include mlflow installation in the setup update the config to mirror the latest tinyllama QLORA config from examples/ ...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L31-L97),">axolotl/src/axolotl/prompters.py at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1244762145304875108)** (22 messages🔥): 

- **Python function for DataFrame to JSONL conversion shared**: A user shared code for converting a DataFrame to JSONL format, suggesting it as a simple and effective solution. The code iterates over DataFrame rows, converts each row to a dictionary, and writes it to a file as a JSON line.

- **Debate on load_in_8bit impacts on training**: Users discussed whether using `load_in_8bit=True` affects training beyond reducing GPU VRAM usage. Observations included better gradient behavior with `load_in_8bit=False` and technical details on quantization and precision during training.

- **Issues with Qwen tokenizer**: There's an ongoing problem with the QwenCode1.5 tokenizer configuration being incorrect. The proposed solution is to switch to Qwen2Tokenizer, but validation or insights from Qwen are pending.

- **Context window length concerns**: A user raised concerns about prompts exceeding the model’s context window length, noting potential exceptions or performance issues. They found success with the "rope_scaling" parameter and mentioned a model supporting longer context windows, like the [Llama-3 8B Gradient Instruct](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k).

- **Addressing token_type_ids in tokenizers**: A critical issue was identified with the default PreTrainedTokenizer class emitting `token_type_ids`, which some model classes do not handle. The correct implementation should iterate over possible vectors specified in `model_input_names` when adjusting sizes, as discussed with references to [HuggingFace code](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1562).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1656">Fix tokenization for CodeQwen models by artemdinaburg · Pull Request #1656 · OpenAccess-AI-Collective/axolotl</a>: Update token_type_ids in PromptTokenizer to match changes to input_ids and attention_mask. Description Some models, like the Qwen series, return a token_type_ids along with input_ids and attention_...</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/lora-8b.yml">axolotl/examples/llama-3/lora-8b.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B/blob/main/tokenizer_config.json#L14)">tokenizer_config.json · Qwen/CodeQwen1.5-7B at main</a>: no description found</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B/blob/main/config.json#L3).">config.json · Qwen/CodeQwen1.5-7B at main</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-4B/blob/main/tokenizer_config.json#L38)">tokenizer_config.json · Qwen/Qwen1.5-4B at main</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1562.">transformers/src/transformers/tokenization_utils_base.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1244780908343722116)** (78 messages🔥🔥): 

- **Early Birds and Night Owls Discuss Gradio**: Members from different time zones shared their dedication, including 4 AM alarms in India and late-night coding sessions at 2 AM.
- **Gradio vs Streamlit Debate**: When asked about preferring **Gradio** or **Streamlit**, a member strongly favored **Gradio**, citing personal bias.
- **Multimodal Chatbots and Google OAuth**: Shared links to **Gradio multimodal chatbots** ([link](https://huggingface.co/spaces/gradio/chatbot_multimodal/blob/main/run.py)) and **Google OAuth** integration guides ([link](https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers)).
- **Live Demo Errors: Learning Opportunity**: Members discussed how live demo errors can be instructive, highlighting experts' debugging processes as beneficial learning moments.
- **Freddy Aboulton's Session Praise**: Participants expressed appreciation for Freddy's session and shared resources for further learning, including a **video link** ([Zoom video](https://us06web.zoom.us/rec/share/I6RRm2606YMi6EnWVlfcXLP3BS9fXrU7NRVIjx9xCWLU_A-OwgCbIRDdeiRMctwN.5nIGeoPUDhwna0qp?startTime=1716850364000)) and **Gradio performance tips guide** ([link](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/byKr9vB9">Join the Hugging Face Discord Server!</a>: We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 80043 members</li><li><a href="https://huggingface.co/spaces/gradio/chatbot_multimodal/blob/main/run.py">run.py · gradio/chatbot_multimodal at main</a>: no description found</li><li><a href="https://huggingface.co/spaces/freddyaboulton/gradio_agentchatbot">gradio_agentchatbot - a Hugging Face Space by freddyaboulton</a>: no description found</li><li><a href="https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers">Sharing Your App</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://vanishinggradients.fireside.fm/">Vanishing Gradients</a>: a data podcast with hugo bowne-anderson</li><li><a href="https://hugobowne.github.io/">hugo bowne-anderson - data scientist</a>: no description found</li><li><a href="https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance">Setting Up A Demo For Maximum Performance</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://huggingface.co/spaces/gradio/chatinterface_multimodal_main/blob/main/run.py">run.py · gradio/chatinterface_multimodal_main at main</a>: no description found</li><li><a href="https://www.youtube.com/live/USTG6sQlB6s?si=cB9adtLWejfTX77K">How to Build Terrible AI Systems with Jason Liu</a>: Jason is an independent consultant who uses his expertise in recommendation systems to help fast-growing startups build out their RAG applications. He was pr...</li><li><a href="https://pyodide.org/en/stable/usage/wasm-constraints.html#synchronous-http-requests-support">Pyodide Python compatibility &#8212; Version 0.26.0</a>: no description found</li><li><a href="https://pyodide.org/en/stable/usage/faq.html#how-can-i-use-fetch-with-optional-arguments-from-python">Frequently Asked Questions &#8212; Version 0.26.0</a>: no description found</li><li><a href="https://us06web.zoom.us/rec/share/I6RRm2606YMi6EnWVlfcXLP3BS9fXrU7NRVIjx9xCWLU_A-OwgCbIRDdeiRMctwN.5nIGeoPUDhwna0qp?startTime=1716850364000">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1245025314015416381)** (5 messages): 

- **Slow Local Inference on M3 Mac Frustrates User**: A user reported significant latency, over 2 minutes per response, when running inference locally on their M3 Mac using [Modal's LLM engine](https://github.com/modal-labs/llm-finetuning.git). They inquired about deploying the model for inference elsewhere, such as Hugging Face or Replicate.

- **Deploying to Modal as a Solution**: Another member clarified that deploying to Modal using `modal deploy` would alleviate the latency issue, as the LLM engine boot would occur on Modal's infrastructure. They noted that latency would primarily be an issue only when a new instance spins up after a long delay.

- **Using Weights Elsewhere**: The same member mentioned that the model weights could also be extracted from the Modal volume and used externally, suggesting the use of `modal volume` CLI commands.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1245073003595698217)** (2 messages): 

- **Langsmith Annotation Queue UI Troubles**: A user mentioned that the **Langsmith annotation queue UI** looked very different from what was demonstrated, stating, *"The input and output are empty. When I hit `V` then I can see the run."*

- **Langsmith Deployment Inquiry**: Another user inquired about the possibility of deploying **Langsmith** on a **private cloud/VPC**.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1244768670203514940)** (12 messages🔥): 

- **Langsmith Bug Fix for Student Credits Setup**: Langsmith reported a bug where students must enroll in Plus to set up billing for credits. They're working on a fix to allow students to enter credit card information without being charged.

- **Survey Responses Verification**: A user inquired about receiving a copy of all their responses to ensure correctness due to tricky organization names/IDs. Another user agreed, acknowledging they will double-check by May 30th.

- **Predibase Gmail Registration Success**: Despite a form stating otherwise, a user managed to sign up for Predibase using a Gmail address via an alternative workflow. Danbecker confirmed that if issues arose later, Predibase support would be responsive.

- **Phone Number Verification Issue**: A user faced problems after verifying a phone number for API creation tied to their personal account and noted difficulties in changing the phone number. Danbecker suggested running credits to that personal account and resubmitting the credits form to update the info.

- **Missing Credits for New Joiners**: Users, including "enginoid" and "seanlovesbooks," joined on May 23 but have not received credits. They contacted to resolve the issue.
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1244728359674445824)** (659 messages🔥🔥🔥): 

```html
- **Anticipation for imfo alpha launch**: An exciting new development is incoming, with a teaser link shared: [spectate_or on X](https://x.com/spectate_or/status/1795077451195830661?s=46). This generated enthusiasm and comparisons to similar tools in the community.
- **Detailed discussion on AI task implementation**: Members discussed categorizing tasks into retrieval and mutation types, with queries like "Get the weight of the iPhone 15" exemplifying this structure. One member emphasized, *"all the steps just happen at the same time,"* needing adjustments for tasks requiring sequential execution.
- **Frustrations around scraping accuracy**: Members faced challenges with HTML parsing for accurate data retrieval, particularly from complex sources like Apple and Docker's release notes. Cloudflare issues and suggestions like using Playwright for JavaScript-heavy sites were also discussed.
- **Cost-effective AI model usage insights**: Detailed calculations were shared on the cost efficiency of using various AI models, with a combined system using Llama3 and Claude models showing significant potential savings.
- **Claude 3 model's performance concerns**: A member shared frustrations about Claude 3 not improving prompts as effectively as before. This triggered a broader discussion on prompt engineering and model performance across different tasks.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://im.fo`">no title found</a>: no description found</li><li><a href="https://promptfoo.dev/">Iterate on LLMs faster | promptfoo</a>: Tailored LLM evals for your use case. Maximize model quality and catch regressions.</li><li><a href="https://abrahamjuliot.github.io/creepjs/">CreepJS</a>: no description found</li><li><a href="https://x.com/spectate_or/status/1795077451195830661?s=46">Tweet from Daniel Kaiser (@spectate_or)</a>: i&#39;ve also been cooking the last weeks.  imfo alpha coming soon  🎉</li><li><a href="https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273">Terminator Terminator Robot GIF - Terminator Terminator Robot Looking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/love-lovely-good-morning-with-gif-22914515">Love Lovely GIF - Love Lovely Good - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/Charles12509909/status/1794630406064795909">Tweet from Charles (@Charles12509909)</a>: I was able to get GPT-4o to highlight all clickable on-screen elements and make it take control of the mouse. It can navigate the computer autonomously using the button coordinates.</li><li><a href="https://tenor.com/view/oh-wah-ah-ah-ah-anthony-vincent-down-with-the-sickness-intro-singing-disturbed-gif-16261397">Oh Wah Ah Ah Ah Anthony Vincent GIF - Oh Wah Ah Ah Ah Anthony Vincent Down With The Sickness Intro - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aws.amazon.com/blogs/aws/anthropics-claude-3-opus-model-on-amazon-bedrock/">Anthropic’s Claude 3 Opus model is now available on Amazon Bedrock | Amazon Web Services</a>: We are living in the generative artificial intelligence (AI) era; a time of rapid innovation. When Anthropic announced its Claude 3 foundation models (FMs) on March 4, we made Claude 3 Sonnet, a model...</li><li><a href="https://techcrunch.com/2022/07/14/you-com-raises-25m-to-fuel-its-ai-powered-search-engine/">You.com raises $25M to fuel its AI-powered search engine</a>: You.com, an AI-powered search engine founded by ex-Salesforce chief scientist Richard Socher, has closed a $25M funding round -- all equity.</li><li><a href="https://www.apple.com/iphone-15/">iPhone 15 and iPhone 15 Plus</a>: iPhone 15 and iPhone 15 Plus. Dynamic Island. 48MP Main camera with 2x Telephoto. All-day battery life. USB-C. 6.1” and 6.7” sizes.</li><li><a href="https://www.phonearena.com/iphone-15-release-date-price-features">Apple iPhone 15 release date, price, and features</a>: no description found</li><li><a href="https://www.apple.com/lae/iphone-15-pro/specs/">iPhone 15 Pro and 15 Pro Max - Technical Specifications</a>: View all technical specifications for iPhone 15 Pro and iPhone 15 Pro Max.</li><li><a href="https://www.tomsguide.com/news/iphone-15">iPhone 15: Price, specs and availability</a>: Everything you need to know about the iPhone 15
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1244898253510672436)** (6 messages): 

- **Daily Focus shares an emotional piece**: [Daily Focus](https://www.perplexity.ai/search/Opus-50-sad-gQI59jEoSC2MtdMSvclDJQ#0) shared a link to a topic titled "Opus 50 sad". Users can explore this link for a deeply emotional and reflective piece of content.
- **TheFuzzel explains Perplexity AI**: [TheFuzzel](https://www.perplexity.ai/search/What-is-Perplexity-uyV3gThHQEa1tWgRyN0sQw) provided a link explaining what Perplexity AI is. This resource could be beneficial for newcomers wanting to understand the basics.
- **Slayer_Terrorblade highlights tech conferences**: [Slayer_Terrorblade](https://www.perplexity.ai/search/Upcoming-tech-conferences-aQxyrYvuSEeQLAivYygV8A) shared a link to a search on upcoming tech conferences. Tech enthusiasts can use this link to stay updated on major events.
- **RiseNoctane queries average information**: [RiseNoctane](https://www.perplexity.ai/search/whats-the-average-Dtc8a0qdRGC7cNp4NpPyOg) posted a link with a search query about averages. This might address a broad range of topics related to statistical averages.
- **Bambus89 shares top TV shows for May**: [Bambus89](https://www.perplexity.ai/search/Beste-Fernsehsendunge-Mai-tU4zYHyWS1.9qNm6dnExhg#0) shared a link about the best TV shows of May. This can be useful for anyone looking for entertainment recommendations for that month.
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1244767114552672318)** (6 messages): 

- **Unclear API Output Frustrates Users**: A member shared confusion about whether the API output they received was intended, showing a JSON object without functional links in its content. Another member suggested this could be due to the absence of the **closed beta citations feature**.
- **Troubleshooting API Link Generation**: To generate relevant video links without access to the citations feature, a member recommended experimenting with different prompts. They provided an example prompt and suggested varying the **model size and the number of links requested** could help.
- **API Outage Concerns**: A user inquired if the API was down, hinting at potential service disruptions. There were no follow-up messages confirming or denying this issue.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1244741231242907669)** (2 messages): 

- **Stable Assistant Launches with New Features**: Stability AI announces new editing features in Stable Assistant, leveraging **Stable Diffusion 3** to produce higher quality text-to-image outputs. *Try it for free on your images* [here](https://stability.ai/stable-assistant).

- **Chatbot Beta Enhancements**: The chatbot, currently in beta, integrates **Stable LM 2 12B** to assist with various text-generation tasks like blog posts, scripts, and image captions. Continuous improvements and more features are expected soon.

- **Stability AI and HUG Team Up for Summer Course**: **Innovation Laboratory** offers a 4-week guided course on training your AI model, with Stability AI's tools and HUG's educational expertise. Register for the event [here](https://www.studios.thehug.xyz/lab) before June 25, 2024.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.studios.thehug.xyz/lab">HUG x Stability AI Innovation Laboratory &mdash; HUG</a>: Discover your own unique innovation with Stability AI and receive real-time strategic, marketing, and creative education from HUG.</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant &mdash; Stability AI</a>: Stable Assistant is a friendly chatbot developed by Stability AI equipped with Stability AI’s text and image generation technology, featuring Stable Diffusion 3 and Stable LM 2 12B.
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1244727100494188616)** (495 messages🔥🔥🔥): 

- **GPU Community Computing Proposal**: A member discussed the idea of a community sharing compute costs by offering unused GPU time, possibly through a custom node or even a blockchain (*"start a new blockchain that somehow uses training community models as its mining function"*).
- **Debate on Cloud-Based AI Assistants**: Concerns were raised about the privacy issues of cloud-based AI assistants, with a sentiment against using such services due to data security risks (*"cloud-based, I won't use them. That's a massive privacy concern"*).
- **SD3 Release Frustration and Cloud Services Skepticism**: Members expressed frustration about SD3 weights not being released for local use and skepticism towards cloud-only options. There was significant dissatisfaction with StabilityAI's business decisions (*"SD 3 in cloud... if you SD team want to differ... needs to release SD3 local"*).
- **Stable Diffusion Workflow and Inpainting Discussions**: Members shared tips and tools for enhancing workflows, such as using various extensions and inpainting methods. One suggested watching tutorials on YouTube for better understanding (*"google inpainting stable diffusion... watch some YouTube videos"*).
- **Debated Benefits of ComfyUI vs Other UIs**: Discussions compared the benefits of ComfyUI to other UIs like Forge or A1111, highlighting ComfyUI's need for technical knowledge versus the ease of use of alternatives. One member shared how certain extensions could enhance ComfyUI's functionality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/">ComfyUI Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://comfyanonymous.github.io/ComfyUI_tutorial_vn/">ComfyUI Tutorial</a>: no description found</li><li><a href="https://civitai.com/">Civitai: The Home of Open-Source Generative AI</a>: Explore thousands of high-quality Stable Diffusion models, share your AI-generated art, and engage with a vibrant community of creators</li><li><a href="https://new.reddit.com/r/StableDiffusion/comments/1d1zw74/mobius_the_debiased_diffusion_model/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d1fkt3/5_new_steerable_motion_workflows_for_trave">Reddit - Dive into anything</a>: no description found</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant &mdash; Stability AI</a>: Stable Assistant is a friendly chatbot developed by Stability AI equipped with Stability AI’s text and image generation technology, featuring Stable Diffusion 3 and Stable LM 2 12B.</li><li><a href="https://stable-diffusion-art.com/samplers/">Stable Diffusion Samplers: A Comprehensive Guide - Stable Diffusion Art</a>: Many sampling methods are available in AUTOMATIC1111. Euler a, Heun, DDIM... What are samplers? How do they work? What is the difference between them? Which</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d1fkt3/5_new_steerable_motion_workflows_for_travelling/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/banodoco/steerable-motion">GitHub - banodoco/Steerable-Motion: A ComfyUI node for driving videos using batches of images.</a>: A ComfyUI node for driving videos using batches of images. - banodoco/Steerable-Motion
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1245021063390429214)** (1 messages): 

- **OpenAI Board forms Safety and Security Committee**: OpenAI has announced the formation of a **Safety and Security Committee** responsible for making recommendations on critical safety and security decisions for all OpenAI projects. More details can be found in their [official announcement](https://openai.com/index/openai-board-forms-safety-and-security-committee/).
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1244727268417212489)** (321 messages🔥🔥): 

- **Self-organizing filesystem fascinates users**: Users discussed a "self-organizing filesystem" called **LlamaFS**, which organizes files based on content and time. One user pointed out, "I want self-organizing everything in my life," indicating their enthusiasm for such automation.

- **Discussion on AI Model Costs and NPU Integration**: There was an in-depth conversation about the increase in **hardware costs** due to the integration of NPUs (Neural Processing Units). Members speculated on the economic impact, debating if NPUs would add $200-$1000 to the hardware costs, especially for high-end models.

- **Debating AI's Role in Game Development**: A heated debate ensued around the potential of AI to assist in developing complex games like GTA. One member commented, "*in a few years an individual could make a GTA game by himself with AI,*" while another dismissed this as overly optimistic, citing current limitations.

- **Curing cancer, GPT, and TOS considerations**: There were considerations around using **GPT** for ambitious projects like curing cancer, and how such initiatives interact with OpenAI's TOS. Amid the discussion, someone clarified that using GPT for AI projects might be okay if it's not to "scrape their data and make a competing model."

- **Memory and Context Capabilities of Models**: Members were impressed by GPT-4o's capability to recall video and audio events, discussing its memory storage as encoding tokens in special folders. SunSweeper noted, "I saw the GPT-4o demo... able to recall the event," expressing amazement at the memory abilities of the AI models.

**Link mentioned**: <a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1244784831250960455)** (21 messages🔥): 

- **Building Business Websites with ChatGPT**: A member questioned the practicality of using ChatGPT to create a professional business website. Another clarified, *"It can create simple wireframes and basic functions, but a fully functional website with only ChatGPT? No."*

- **Custom GPTs Memory Feature Confusion**: There was confusion about whether custom GPTs have access to memory features. One user explained, *"Currently, GPTs do not have long-term memory... but we can design a GPT to help manage information and mimic a long-term memory experience."*

- **No Significant Updates Noticed**: Discussions around the iOS app update led to members asking if there were any new features; one responded, *"Nothing new."*

- **Multiple Issues with GPT Functionality**: Users reported various issues such as context drops, blanked-out memories, and coding errors. *“Is GPT-4 tripping for anyone else?”* highlights general dissatisfaction with the current performance.

- **ChatGPT Availability Issues**: Intermittent problems with ChatGPT being unresponsive were noted, with one user confirming, *“chat gpt is not responding. Does anybody has the same problem?”* Another replied, *“working perfectly here.”* prompting confusion.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1244736004099674122)** (76 messages🔥🔥): 

- **ChatGPT Refuses to Draw Tarot Cards, Then Cooperates on Smaller Request**: A user shared their anecdote about asking ChatGPT to draw three tarot cards which it refused, but it complied after the user asked for drawing just one. Many users agree on experiencing ChatGPT's arbitrary refusal of feasible tasks.
  
- **Alternate Language Prompt Solves Task Refusal Issue**: When asked to process a French document with an English prompt, ChatGPT initially refused but then performed the task when the request language matched the document. Users noted this can be a solution to similar task refusals.

- **Balancing Prompt Length with Response Quality**: A user faced issues with their extensive prompt cutting off responses in GPT-4o. Another user shared strategies like replacing verbose planning with mermaid diagrams to reduce token usage while maintaining output quality.

- **Meta-Prompting vs. Chain of Thought**: A debate ensued on whether meta-prompting or Chain of Thought (CoT) methods yield better outcomes, with suggestions on optimizing meta-prompting by introducing knowledge representation attachments. Users were advised to creatively combine these methods to avoid narrow projections and hallucinations.

- **Sharing Results and Resources**: Users shared prompt results and discussed further optimizations, including a user sharing their [improved prompt](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477) for Computer and AI literacy.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1244736004099674122)** (76 messages🔥🔥): 

- **ChatGPT tarot card refusal frustrates user**: A member shares an anecdote about ChatGPT refusing to draw tarot cards despite repeated requests, highlighting the unpredictable nature of the AI's refusals. Another member noted similar experiences with ChatGPT claiming tasks are too large but working with a better prompt.

- **Challenge balancing prompt detail in GPT-4**: Users discuss issues with GPT-4 cutting off answers due to prompt length and finding a balance. One suggestion was using **mermaid diagrams** for planning to conserve tokens and another to fuse it with a zero-shot approach.

- **Meta-prompting versus Chain of Thought**: A debate occurs on the effectiveness of zero-shot versus chain of thought (CoT) in meta-prompting. Meta-prompting is highlighted as optimizing AI for open-ended tasks with higher-dimensional solution spaces, while CoT can lead to deterministic outputs.

- **Insights on using YAML and XML for AI prompts**: The conversation dives into how AI handles YAML, XML, and JSON natively. Examples and suggestions were provided on using these formats to structure prompts for better AI understanding and performance.

- **Practical examples and shared resources**: Members shared experiences and links to successful prompts, such as generating flutter code and comprehensive planning templates for trips, demonstrating practical applications of discussed techniques. Links included [Computer and AI Literacy prompt](https://chatgpt.com/c/d943acd5-e9c4-454e-8544-ad2faba45df8) and [expanded literacy coverage](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477).
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1245036889539612764)** (1 messages): 

- **Hugging Face drops new models**: A new batch of open models includes **CogVLM2** for multimodal conversations, **Yi 1.5** with long context, and **Falcon VLM** among others. Check out the [detailed announcement and links](https://x.com/osanseviero/status/1793930015047880959).

- **Sentence-transformers v3.0 released**: The new release features multi-GPU training, bf16 support, and more. Click [here](https://huggingface.co/posts/tomaarsen/872659372583163) for full details.

- **Diffusers 0.28.0 now supports non-generative tasks**: The latest update adds functionality for depth estimation and normals' prediction through Marigold. Detailed release notes can be found [here](https://github.com/huggingface/diffusers/releases/tag/v0.28.0).

- **Gradio introduces new features**: Major launch event unveils new libraries and features for Gradio apps. Mark your calendars for [June 6th](https://x.com/Gradio/status/1793758586147090902).

- **LLMs and cyber security**: Researchers evaluate which large language models are safest for cyber security scenarios. Read the full analysis [here](https://huggingface.co/blog/leaderboard-llamaguard).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/osanseviero/status/1793930015047880959)">Tweet from Omar Sanseviero (@osanseviero)</a>: 📰New open models this week: multilinguality, long contexts, and VLMs 🔥  - CogVLM2: multimodal conversational - Yi 1.5 long context - M2-BERT-V2, long-context encoder models - Phi 3 small and medium ...</li><li><a href="https://x.com/RisingSayak/status/1795083868900311360)">Tweet from Sayak Paul (@RisingSayak)</a>: Start the week with a new 🧨 Diffusers release ❤️‍🔥   This release includes the first non-generative tasks within the library -- depth estimation and normals&#39; prediction through Marigold 💐  Note...</li><li><a href="https://x.com/Gradio/status/1793758586147090902)">Tweet from Gradio (@Gradio)</a>: Launch Event: We&#39;re launching something NEW  Click the &#34;Notify Me&#34; button and stay tuned, Thursday June 6th   https://www.youtube.com/watch?v=44vi31hehw4&ab_channel=HuggingFace</li><li><a href="https://x.com/victormustar/status/1795405605605106044)">Tweet from Victor M (@victormustar)</a>: ✨ Tools are now available in HuggingChat  In short, Tools allow HuggingChat to use any Al applications built by the community (ZeroGPU Spaces), offering limitless possibilities.</li><li><a href="https://x.com/_philschmid/status/1793910461286494539)">Tweet from Philipp Schmid (@_philschmid)</a>: Exciting News! 📢 New @nvidia A100 & H100 GPUs for @huggingface inference Endpoints powered by @googlecloud, they go brrr! 🏎️ 💨💨 Each user/organization has a default quota of 2x A100 & H100 GPUs. A...</li><li><a href="https://x.com/osanseviero/status/1793018964479463781)">Tweet from Omar Sanseviero (@osanseviero)</a>: I&#39;m GPU Poor. What about you?  https://huggingface.co/settings/local-apps</li><li><a href="https://x.com/clefourrier/status/1793922499559747958)">Tweet from Clémentine Fourrier 🍊 (@clefourrier)</a>: Can LLMs be used to help cyber attackers? 🤖 How good are they at hacking their sandboxes? ... In short, which LLM is the safest for cyber security?  Researchers from @Meta worked on a benchmark to an...</li><li><a href="https://x.com/dylan_ebert_/status/1793643044346159553)">Tweet from dylan (@dylan_ebert_)</a>: Machine Learning for 3D Course Unit 3 has launched! This covers:  🎨 What is Gaussian Splatting? ⚙️ How it fits in the generative 3D pipeline ✏️ Hands-on code to build your own demo  check it out at h...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1244731126174711879)** (333 messages🔥🔥): 

- **Creating Multi-modal Spaces Explained**: A user asked about the creation of early multi-modal spaces, whether they are single models or stacked models with a router. Another user shared a [source link](https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py) to view the specifics of such an implementation.
  
- **Account Creation Issues Due to VPN/Adblock**: A user faced issues creating an account, receiving 'account not found' errors. Solutions suggested included disabling VPN, proxy, and adblocker settings as HuggingFace has tight security.

- **TinyLlama Training Insights and Issues**: Several users discussed the dataset size and steps required for effective training of TinyLlama. An example was given where finetuning TinyLlama on a small dataset can be done efficiently within approximately 2 hours on a 10k entry dataset.

- **Spectogram-to-Wav Model Release**: A user announced the upcoming release of a new spectogram-to-wav model and the challenges faced due to compute limitations. Another shared their experience and suggestions on finetuning models and avoiding common pitfalls.

- **Video Classification Model on Kaggle**: One user reported issues with a Video Classification Model only utilizing CPU during validation despite performing correctly with GPU during training. A link to their [Kaggle notebook](https://www.kaggle.com/code/an1001/tiktokvideoclassification?scriptVersionId=180260409) was shared for more context.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o">OpenGPT 4o - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://arxiv.org/abs/2305.05176">FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance</a>: There is a rapidly growing number of large language models (LLMs) that users can query for a fee. We review the cost associated with querying popular LLM APIs, e.g. GPT-4, ChatGPT, J1-Jumbo, and find ...</li><li><a href="https://huggingface.co/spaces/kimou605/shadow-clown-BioMistral-7B-DARE">GenSeq - a Hugging Face Space by kimou605</a>: no description found</li><li><a href="https://huggingface.co/openai/clip-vit-base-patch32">openai/clip-vit-base-patch32 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/cat-dont-care-didnt-ask-didnt-ask-i-didnt-ask-gif-25429803">Cat Dont Care Didnt Ask GIF - Cat Dont Care Didnt Ask Didnt Ask - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py">app.py · KingNish/OpenGPT-4o at main</a>: no description found</li><li><a href="https://www.kaggle.com/code/an1001/tiktokvideoclassification?scriptVersionId=180260409">TikTokVideoClassification</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from TikTok Videos</li><li><a href="https://wandb.ai/mikusdevr/huggingface/runs/jfs4xvfr/workspace">mikusdevr</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://huggingface.co/settings/local-apps">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378">apple/DFN5B-CLIP-ViT-H-14-378 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/ZeroWw/MEISD">ZeroWw/MEISD · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1244802725657444392)** (1 messages): 

- **SDXL Embed Space Alignments Explained**: The **SDXL embed space aligns** the unconditioned space to zeroes instead of an encoded space like earlier SD 1.5/2.x models/DeepFloyd. *"Realigning a model to a new uncond space is painful and takes a long time."*
- **ControlNet Training Uncertain**: The member has learned about **ControlNet training** but isn't sure if they implemented it correctly. This uncertainty is common when tackling complex models.
- **Optimizing Timestep Range Slicing**: Segmenting the timestep range to match the batch size allows for **more uniform sampling of timesteps for smaller compute training**. Without this, you may end up with large gaps in the timestep training distribution, potentially compromising training stability.
- **Benefits of Aspect Bucketing and Drawbacks**: Using **random aspect bucketing** helps shift content-aspect bias and is likely used in DALLE-3, which also supports three resolutions. However, it's challenging to maximize training samples without introducing distortions.
- **Pitfalls in Training Workflows**: Leaving the **Torch anomaly detector on for months** accidentally **wastes time**, and trying to **fixed something "100% for realsies"** tends to introduce new issues. 


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1244916184420323359)** (1 messages): 

- **Explore Open Source AI Repositories**: A member shared an interesting [article](https://huyenchip.com/2024/03/14/ai-oss.html?utm_source=tldrai) on the open-source AI ecosystem and how it has evolved, with accompanying discussions on [Hacker News](https://news.ycombinator.com/item?id=39709912), [LinkedIn](https://www.linkedin.com/posts/chiphuyen_generativeai-aiapplications-llmops-activity-7174153467844820993-ztSE), and a [Twitter thread](https://twitter.com/chipro/status/1768388213008445837). The article provides a comprehensive list of open-source AI repositories, updated every six hours, which can also be found in the [cool-llm-repos](https://github.com/stars/chiphuyen/lists/cool-llm-repos) list on GitHub.

- **Revisit MLOps Analysis**: The member conducted an analysis of the [open source ML ecosystem](https://huyenchip.com/2020/06/22/mlops.html) four years ago and revisited the topic to focus exclusively on the stack around foundation models. The full details include data on repositories and the evolution of the AI stack over time.

**Link mentioned**: <a href="https://huyenchip.com/2024/03/14/ai-oss.html?utm_source=tldrai">What I learned from looking at 900 most popular open source AI tools</a>: [Hacker News discussion, LinkedIn discussion, Twitter thread]

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1244731504731619509)** (6 messages): 

- **Voice-Controlled Robotic Arm Project Unveiled**: A user shared a YouTube video titled ["Open Source Voice-Controlled Robotic Arm"](https://www.youtube.com/watch?v=qv3bFhHoA5s) showcasing an AI-powered robotic arm controlled by voice commands. The project aims to democratize robotics through open-source contributions.

- **TinyML Bird Classification Model in Action**: An individual discussed their TinyML bird classification model based on EfficientNetB0, tested on random Reddit birding posts and a clean test set. They shared a detailed [article](https://www.cranberrygrape.com/machine%20learning/tinyml/bird-detection-tinyml/) covering the model's generation and invited partnerships for further research.

- **SD.Next Releases Major Update**: The SD.Next project announced a significant release featuring a new [ModernUI](https://github.com/BinaryQuantumSoul/sdnext-modernui), various built-in features like **HiDiffusion** and enhanced samplers, and newly supported models like [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/). Full release details and features are available in the project's [Changelog](https://github.com/vladmandic/automatic/blob/dev/CHANGELOG.md).

- **Introducing HuggingPro Assistant**: A user introduced [HuggingPro](https://hf.co/chat/assistant/66562fe0abb44809b7f77897), an AI assistant designed to navigate the Hugging Face ecosystem. The assistant offers accurate info on models, datasets, and more, aiming to make the experience both efficient and enjoyable.

- **Everything-AI v2.0.1 Adds New Functionalities**: The user promoted the latest version of [everything-ai](https://github.com/AstraBert/everything-ai), an AI-powered local assistant with new features like audio file handling, text-to-video generation, and protein structure prediction. They provided a [quick-start guide](https://astrabert.github.io/everything-ai/) for users interested in setting up the tool locally.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cranberrygrape.com/machine%20learning/tinyml/bird-detection-tinyml/">Bird Detection TinyML</a>: Obsessively Shrinking a Transfer Based Model</li><li><a href="https://www.youtube.com/watch?v=qv3bFhHoA5s">Open Source Voice-Controlled Robotic Arm | Redefining Robots!</a>: Welcome to the Voice-Controlled AI Robotic Arm project  where artificial intelligence meets robotics. A open-source initiative empowers users to command a ro...</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Your fully proficient, AI-powered and local chatbot assistant🤖</a>: Your fully proficient, AI-powered and local chatbot assistant🤖 - AstraBert/everything-ai</li><li><a href="https://astrabert.github.io/everything-ai/">everything-ai</a>: Your fully proficient, AI-powered and local chatbot assistant🤖</li><li><a href="https://hf.co/chat/assistant/66562fe0abb44809b7f77897">HuggingPro - HuggingChat</a>: Use the HuggingPro assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/66562fe0abb44809b7f77897)">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://github.com/vladmandic/automatic/wiki/Themes)">Create new page · vladmandic/automatic Wiki</a>: SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models - Create new page · vladmandic/automatic Wiki
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

pr0x7: okay I will try and prepare.update you accordingly. thanks
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1244750701729677353)** (4 messages): 

- **Collecting Topics for Hugging Face CV Hangout**: A member created a [Google Sheet](https://docs.google.com/spreadsheets/d/12PewkdH2oAJ1Azw3sTxi7FatJ9fE4bTUj_APQQ__Ufs/edit?usp=sharing) to gather discussion points for a Saturday hangout. Topics can range from new models to personal projects, and participants are encouraged to contribute.

- **Struggles with Stanford Cars Dataset**: A user shared their attempt to classify the make and model of cars using the Stanford Cars dataset with the ViT-B_16 model, achieving only 60% accuracy. They detailed their augmentation techniques and learning rate scheduler but faced overfitting issues and sought advice on fine-grained image classification.

- **Request for Deep Learning Guidance**: The user acknowledged being new to deep learning and requested guidance from more experienced practitioners to improve their model's performance.

- **New Member Introduction**: A user introduced themselves as being new to the community. They did not provide any further context or specific questions.

**Link mentioned**: <a href="https://docs.google.com/spreadsheets/d/12PewkdH2oAJ1Azw3sTxi7FatJ9fE4bTUj_APQQ__Ufs/edit?usp=sharing">Hugging Face Computer Vision Hangout</a>: Tabellenblatt1  Topic (Fine-Tuning/Cool Project/etc.),Style (Short Presentation/Discussion/etc.),Proposed by (discord name)

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1244947164556165192)** (2 messages): 

- **Search for Nomos8k_sfw dataset reveals obstacles**: A member expressed difficulty finding the Nomos8k_sfw dataset mentioned in the [4x-Nomos8kDAT model](https://openmodeldb.info/models/4x-Nomos8kDAT), questioning whether it's exclusive or just well-hidden. 
- **Typeface Arc aims for efficient AI content creation**: The [Typeface Arc platform](https://www.typeface.ai/) provides tools to create and manage brand stories in one unified experience. It features a "Copilot" to effortlessly generate 10x more content using continuous feedback for optimization.

**Link mentioned**: <a href="https://www.typeface.ai/">Typeface | Personalized AI Storytelling for Work</a>: Typeface, the generative AI application for enterprise content creation, empowers all businesses to create exceptional, on-brand content at supercharged speeds.

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1245027012280713369)** (1 messages): 

- **Gradio Clients 1.0 Livestream Announcement**: The Gradio team announced a livestream event set for June 6th to unveil the new and improved Gradio Python and JavaScript clients. Interested parties can join via [Discord](https://discord.gg/hugging-face-879548962464493619?event=1245020251611992154) or watch on [YouTube](https://www.youtube.com/watch?v=44vi31hehw4) to learn how to incorporate Gradio into various applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1245020251611992154">Join the Hugging Face Discord Server!</a>: We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 80043 members</li><li><a href="https://www.youtube.com/watch?v=44vi31hehw4">Gradio Launch: How to Build Machine Learning APIs Using the Gradio Clients</a>: One million developers use Gradio every month to create machine learning demos and web applications using the Gradio Python library. Join the Gradio Team on ...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1244738786315997382)** (61 messages🔥🔥): 

- **Using vision with OpenAI API in LM Studio clarified**: To integrate vision capabilities in LM Studio, use a model like **LLaVA**, set it up on a server, and utilize the vision Python template. *"Just get a model that has vision like llava. Load it on to a server. And copy paste the vision python template."*
  
- **MLX/EXL2 loading faster on Apple's M1 Max**: **MLX and EXL2 models load significantly faster** than GGUF on Apple's M1 Max, taking around 5 seconds for L3 8bit, and 29 seconds for GGUF Q8. *"MLX/EXL2 are much faster than GGUF's. mainly cause the inference engine is different."*

- **Using RAG with Local LLMs**: LM Studio does not support direct interaction with PDFs or Ebooks; however, running a server via LM Studio and using [AnythingLLM](https://community.amd.com/t5/ai/how-to-enable-rag-retrieval-augmented-generation-on-an-amd-ryzen/ba-p/670670) can set up Retrieval Augmented Generation. *"start a server through LM Studio's Local Server tab and then run AnythingLLM."*

- **Fine-tuning not supported in LM Studio**: **LM Studio does not support fine-tuning**, but fine-tuning can be done using other tools like [MLX on Mac with Apple Silicon](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/). *"Training models is way more resource intensive than running models + the inference engine (llama.cpp doesn't support finetuning)."*

- **Function calling in LM Studio not supported**: **LM Studio and similar llama.cpp-based APIs do not support function calling**. *"function calling isn't supported in the API."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/3bXg4Qv3">Join the Mintplex Labs | AnythingLLM | VectorAdmin Discord Server!</a>: Check out the Mintplex Labs | AnythingLLM | VectorAdmin community on Discord - hang out with 4215 other members and enjoy free voice and text chat.</li><li><a href="https://community.amd.com/t5/ai/how-to-enable-rag-retrieval-augmented-generation-on-an-amd-ryzen/ba-p/670670">How to enable RAG (Retrieval Augmented Generation) on an AMD Ryzen™ AI PC or Radeon Graphics Card</a>: GPT based Large Language Models (LLMS) can be helpful AI assistants that maximize your productivity and increase the efficiency of your workflow. Running an AI chatbot on your AMD Ryzen™ AI powered AI...</li><li><a href="https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/">A simple guide to local LLM fine-tuning on a Mac with MLX &#8211; Andy Peatling</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Sao10K/Claude-3-Opus-Instruct-15K">Sao10K/Claude-3-Opus-Instruct-15K · Datasets at Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1244810505021820998)** (45 messages🔥): 

- **Fight the "Can I ask a question?" Trap**: Members discussed frustration with people who ask "Can I ask a question?" instead of directly asking their question, wasting time for both parties involved. As one member put it, *"you can save your time and everyone else's time if you just ask the question instead of asking to ask the question."*
- **Use AI and Experts Wisely**: While **Google and AI can be unreliable**, one member noted, *"I ask an AI first, then confirm it with an expert if it's not working."* This highlights a balanced approach to solving problems with technology and human expertise.
- **Phi-3-Vision Support Limitations**: One member asked if [Phi-3-Vision-128K-Instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) can work in LM Studio, but another clarified, *"Still not working in llama.cpp so won't work in LM Studio."*
- **Exploring "Glitchy" Model Behaviors**: There was an inquiry into models exhibiting "glitchy" behaviors with specific prompts. An example given was *dolphin 2.9 llama 3*, which shows erratic behavior when loaded with specific presets.

**Link mentioned**: <a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found

  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1244886480833482783)** (12 messages🔥): 

- **Wrong Chat Template Causes AI Errors**: A member faced an issue with the AI response and it was clarified that they had not configured the right chat template for Llama 3. It was suggested to pick the correct preset from the top-right menu.

- **Blood and Caesar AI Message Causes Confusion**: A bizarre message generated by the AI about writing a message in blood saying "Hail Caesar!" led to speculation. A user suggested it might be some uncensored model training bleeding through. 

- **Llama 3 List Generation Annoys Users**: A member complained about Llama 3 constantly generating lists despite system prompts instructing it not to. They shared an example system prompt that appears ineffective, seeking better alternatives.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1244764138199187487)** (135 messages🔥🔥): 

- **Users debate Nvidia budget choices vs expensive GPUs for AI training**: Several members discussed options like the **Nvidia Tesla P40/P100** and the **highly rumored 5090** GPUs with 32GB VRAM, contemplating their cost and performance. Alternately, **Macs were suggested** for inference purposes, but a PC is considered better for training.

- **Cautious optimism about GPUDirect Storage**: The tech that allows **GPUs to access SSDs directly** without CPU involvement was explored. However, its complex installation and the unknowns around if it’s worth the effort tempered excitement.

- **Concerns about Nvidia’s market bubble**: Some members considered the **potential bubble** in Nvidia shares due to their dominant AI chip market position and debated the stability and future impact of competitor advances from AMD or Intel.

- **Diverse experiences with delivery and couriers**: Members shared issues with **hardware delivery** speeds and reliability, particularly contrasting experiences in **Russia** and **Australia**, highlighting the frustrations with courier services compared to traditional postal options.

- **Speculation on whether new AI PCs are worthwhile**: There was skepticism about the **“Copilot + PC”** and other AI PCs being heavily marketed, with concerns about whether they truly added significant value or were **overhyped** products relying more on cloud services than local capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html">NVIDIA GPUDirect Storage Overview Guide - NVIDIA Docs</a>: no description found</li><li><a href="https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html">NVIDIA GPUDirect Storage Installation and Troubleshooting Guide - NVIDIA Docs</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1244848496042049536)** (13 messages🔥): 

- **Windows CPU affinity issues with large models**: A user discussed how Windows starts inferencing on all cores but eventually moves work to one CCD after a few minutes when running large models. They are planning to experiment with configuring each CCD as a separate NUMA node to improve system memory bandwidth utilization.
- **Error with loading models on AVX2 laptop**: A member shared an error encountered when loading a model on a new laptop with AVX2 support, leading to GPU offload issues on Linux. The suggestion to disable GPU acceleration settings did not resolve the problem as the settings were grayed out.
- **Processor masquerading issues**: Another user speculated that the error might be due to the processor being used as a GPU and recommended disabling the processor's "GPU" to resolve the issue. This might help the system connect to an actual GPU resource.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1244728768564822056)** (169 messages🔥🔥): 

- **GPT-2 Not Supported by Unsloth**: Members asked if GPT-2 is supported for fine-tuning in Unsloth. It was confirmed it is not, due to architectural differences.

- **Dataset Preparation for Fine-Tuning with Unsloth**: A user sought advice on fine-tuning a dataset of over 50k email entries using llama 3 with Unsloth, focusing on creating the proper structure for inputs and outputs. Several users offered assistance and suggestions, including restructuring the prompt template to fit the dataset.

- **Vision Models to be Supported Soon**: Discussion highlighted that Unsloth does not currently support vision models, but support is expected next month. This sparked conversations on existing vision models, such as Stable Diffusion and Segment Anything, with links shared for more information ([Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Segment Anything](https://github.com/facebookresearch/segment-anything)).

- **Using and Merging LoRA Adapters**: Members discussed how to merge and fine-tune LoRA adapters with the original model, and the tools available for saving and uploading these models to platforms like HuggingFace. [GitHub link](https://github.com/unslothai/unsloth#-finetune-for-free) was shared for related resources.

- **Phi 3 Medium Sliding Window Issue**: It was noted that Phi3-Medium uses a sliding window attention mechanism that caused performance issues at higher token counts. Many users expressed frustration and anticipation for the model to support higher context windows, specifically mentioning 128K context.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html">Axolotl - Conversation</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found</li><li><a href="https://x.com/karpathy/status/1795518622913433891">Tweet from Andrej Karpathy (@karpathy)</a>: But those were also much much bigger runs, so it&#39;s a lot more impressive. This was on a single node so you don&#39;t need to deal with any cross-node interconnect. It starts to get a lot more fun ...</li><li><a href="https://x.com/danielhanchen/status/1795453604532207989">Tweet from Daniel Han (@danielhanchen)</a>: How did we &#34;mistral-fy&#34; Phi-3? 1) Unfuse QKV and gate/up weights 2) Note Phi-3 uses sliding window attention 3) & Phi-3 has a bug - 2047 SWA should be 2048 & sent @UnslothAI&#39;s versions to ...</li><li><a href="https://github.com/matatonic/openedai-vision">GitHub - matatonic/openedai-vision: An OpenAI API compatible API for chat with image input and questions about the images. aka Multimodal.</a>: An OpenAI API compatible API for chat with image input and questions about the images. aka Multimodal. - matatonic/openedai-vision</li><li><a href="https://github.com/facebookresearch/segment-anything">GitHub - facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. -...</li><li><a href="https://huggingface.co/datasets/openchat/ultrachat-sharegpt?row=0">openchat/ultrachat-sharegpt · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/CompVis/stable-diffusion">GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model</a>: A latent text-to-image diffusion model. Contribute to CompVis/stable-diffusion development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#training-adapters">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/en/developer_guides/lora">LoRA</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1ef-ta">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1u_ozy3HqmiwwzG5kqqVklYDc05hVVJH_">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/UKPLab/sentence-transformers/releases/tag/v3.0.0">Release v3.0.0 - Sentence Transformer Training Refactor; new similarity methods; hyperparameter optimization; 50+ datasets release · UKPLab/sentence-transformers</a>: This release consists of a major refactor that overhauls the training approach (introducing multi-gpu training, bf16, loss logging, callbacks, and much more), adds convenient similarity and similar...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1244763540343095297)** (48 messages🔥): 

```html
- **Fix GDrive Save Error by Correcting Argument Order**: A member struggled with an error while saving a model to GDrive due to incorrect argument order in `save_pretrained_merged`. Another member suggested fixing the argument order which solved the issue (*"Welp, that was dumb of me, thanks!"*).
- **Batch Size and Steps During Training**: Members discussed how to set epochs and steps for a model with 500 examples using batch size 8 and 62 steps. It was suggested to use `num_train_epochs = 3` and remove `max_steps = 500` to potentially avoid repetitive outputs and overfitting.
- **Repeating Sentences in Model Training**: A member encountered an issue with the model repeating the same sentence after training, possibly due to missing EOS tokens. This suggests the need to ensure that an EOS token is added to prevent overfitting or insufficient training.
- **Exporting Models to ONNX**: A member sought help converting a fine-tuned model to ONNX format. They were directed to Hugging Face's [ONNX export guide](https://huggingface.co/docs/transformers/en/serialization) and clarified that VLLM format works for the conversion.
- **Support for 8-bit and OpenAI-compatible Servers**: Discussions covered future support for 8-bit models and OpenAI-compatible servers. It's indicated that 8-bit support is coming soon, and there's a pathway for running Unsloth models in environments similar to LM Studio, Jan AI, or Ollama.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v">YouTube</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/serialization">Export to ONNX</a>: no description found</li><li><a href="https://huggingface.co/datasets/kigner/ruozhiba-llama3-tt?row=3">kigner/ruozhiba-llama3-tt · Datasets at Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing&authuser=1#scrollTo=yqxqAZ7KJ4oL)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing.">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mH">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1245097510981210114)** (2 messages): 

- **Lighting.ai receives praise for GPGPU**: A member inquired about using **Lighting.ai** for GPGPU programming, explaining they lack commodity hardware for an NVIDIA card and need to program in CUDA and SYCL. A response affirmed that **Lighting.ai is amazing** for this use case.
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1244988292890492950)** (2 messages): 

- **Deep Dive into GPU Hardware and Programming Model**: A member shared two articles explaining the **GPU hardware and programming model**. They mentioned the importance of understanding GPU capabilities for optimizing the performance of large language models (LLMs) and reducing latency ([Part 1](https://cmeraki.github.io/gpu-part1.html), [Part 2](https://cmeraki.github.io/gpu-part2.html)).
- **ViT Model in Triton**: The member also mentioned implementing the **ViT model completely from scratch in Triton**. They claim the performance is competitive with Hugging Face's implementation and provided a [GitHub link](https://github.com/cmeraki/vit.triton) for those interested in learning about Triton.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cmeraki.github.io/gpu-part1.html">GPUs Part 1 - Understanding GPU internals</a>: LLM Labs</li><li><a href="https://cmeraki.github.io/gpu-part2.html">GPUs Part 2 - Understanding the GPU programming model</a>: LLM Labs
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1245102494304964678)** (12 messages🔥): 

- **Torch.compile incompatible with Python 3.12**: A user discovered that **torch.compile** does not work with Python 3.12, due to missing **triton** installation after setting up PyTorch 2.3 on Ubuntu 24.04. They found a related [GitHub issue](https://github.com/pytorch/pytorch/issues/120233) tracking this problem and noted that **flash-attention** works on Python 3.12.
- **Pyenv suggested to manage Python versions**: Another user experienced the same issue on **Arch Linux** with Python 3.12 and noted support in PyTorch nightlies. They recommended using [pyenv](https://github.com/pyenv/pyenv) to manage multiple Python versions.
- **New bytecode causing issues**: It was clarified that **dynamo** needs to interpret new bytecodes introduced in each Python version, causing the issue. Plans to align PyTorch releases more closely with Python versions and noting partial support in nightlies were discussed.
- **Windows support awaited**: A user working in a .NET-focused environment expressed the need for **native Windows support** for torch.compile with Python 3.12.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054">Torch.compile support for Python 3.12 completed</a>: Signal boosting that Python 3.12 support has been added to torch.compile and has been present in the nightly builds for a while. We anticipate that this feature will be included in the PyTorch 2.4 rel...</li><li><a href="https://github.com/pyenv/pyenv">GitHub - pyenv/pyenv: Simple Python version management</a>: Simple Python version management. Contribute to pyenv/pyenv development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/issues/120233">Torch compile does not work on python 3.12 · Issue #120233 · pytorch/pytorch</a>: 🐛 Describe the bug Currently torch, as of 2.2.0 does not support torch compile with python 3.12 See following PR for example: #117853 We need to be able to use python 3.12 with torch.compile featur.....
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1245073610964598975)** (1 messages): 

- **GPT-4o straggles with large code edits**: Frontier models such as GPT-4o struggle with large edits, facing issues like *laziness, inaccuracy, and high-latency*. "Accurately editing hundreds of lines can take multiple model calls, at times trapping the agent in an infinite loop."

- **Fast Apply model aims to address weaknesses**: A specialized model named **fast apply** is trained to address these weaknesses, breaking down the task into **planning** and **applying** stages. "In Cursor, the planning phase takes the form of a chat interface with a powerful frontier model."

- **Seeking deterministic algorithm leads**: A member is seeking leads for implementing a deterministic algorithm for code edits, mentioning potential feasibility with **vllm** or **trtllm**. They believe it’s possible to speculate on future tokens using such an algorithm rather than relying on a draft model.

For more details, you can read the [full blog post](https://cursor.sh/blog/instant-apply).

**Link mentioned**: <a href="https://cursor.sh/blog/instant-apply">Near-Instant Full-File Edits</a>: no description found

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1244810007028039760)** (15 messages🔥): 

- **Install CUDA Toolkit via Conda and System Requirements**: A member suggested to another that they "install the CUDA Toolkit" from the NVIDIA developer site and to check if it's already installed by typing `nvidia-smi` in the terminal. They also recommended using the [official CUDA downloads page](https://developer.nvidia.com/cuda-downloads) and provided additional resources for documentation and forums.
  
- **Command for Installing on Ubuntu**: To set up CUDA on Ubuntu, a user provided commands: "conda install cuda -c nvidia/label/cuda-12.1.0" and "conda install 'pytorch>2.0.1' torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia/label/cuda-12.1.0" from Jeremy's tweet. Another user mentioned the necessity of ensuring NVIDIA GPU drivers are installed.
  
- **Blog for Installing CUDA on Ubuntu**: A member vaguely recalled the existence of a blog on properly installing Nvidia drivers and CUDA toolkit on Ubuntu/Linux, although no specific link was provided.
  
- **Seeking PMPP Study Guide**: A different user inquired if anyone had created a study guide for PMPP, including prioritized sections and exercises. The request implies a need for structured study materials.

**Link mentioned**: <a href="https://developer.nvidia.com/cuda-downloads">CUDA Toolkit 12.1 Downloads</a>: Get the latest feature updates to NVIDIA&#39;s proprietary compute stack.

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1244753742255886477)** (3 messages): 

- **Understanding torch.fx.Interpreter and GPTQRunner**: A member raised a question regarding the behavior of `call_function` in the `torch.fx.Interpreter` docs versus its use in `GPTQRunner`. They provided a link to the [GPTQRunner class](https://github.com/pytorch/ao/blob/7511b1d365e2e314d1193d7b8df049ee9452e63c/torchao/quantization/GPTQ.py#L296) for context.

- **Support for MX formats merged**: Another member excitedly announced the merging of support for MX formats, including `fp8/6/4` in [PyTorch](https://github.com/pytorch/ao/pull/264). They invited others interested in improving speed to tag them or `vkuzo` on GitHub and mentioned that reviewing the code alongside the [MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) clarified many details.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/">pytorch</a>: pytorch has 75 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/pytorch/ao/pull/264">Add a prototype of MX format training and inference by vkuzo · Pull Request #264 · pytorch/ao</a>: Summary: The MX numerical formats are new low precision formats with recent acceptance into the OCP spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf This ...</li><li><a href="https://github.com/pytorch/ao/blob/7511b1d365e2e314d1193d7b8df049ee9452e63c/torchao/quantization/GPTQ.py#L296">ao/torchao/quantization/GPTQ.py at 7511b1d365e2e314d1193d7b8df049ee9452e63c · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1245081824208879717)** (27 messages🔥): 

- **Choosing the right tech city**: A member sought recommendations for cities with a strong tech culture and social activities like hackathons. Suggestions included big cities like **SF, NYC, and London** for their vibrant social scene and smaller cities like **Seattle**, which had mixed reviews.
  
- **Cities in Europe**: Berlin and Warsaw were recommended over Munich for being more exciting. Berlin was particularly highlighted for its vibrant culture, including *"3-day long techno parties and yummy kebabs"*.
  
- **San Diego and Ithaca**: San Diego was praised by a member who lived there for many years, whereas **Ithaca** was noted for producing successful individuals from Cornell but was described as boring.
  
- **Seattle's social scene**: A member shared their negative experience living in Seattle, describing it as the least social city due to its long, dark winters and tendency for people to stay indoors.
  
- **Tech companies in Berlin**: It was noted that Google and other small startups operate in Berlin, but major engineering work is limited. The suggestion was made to gain **big tech experience in SF or NYC** for future opportunities like starting a company.
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1244728712323272766)** (131 messages🔥🔥): 

- **Debate on `dirent.h` and `unistd.h` integration**: Members discussed where to incorporate `dirent.h`, with one suggesting putting the code in `unistd.h` and renaming it to avoid conflicts with the standard Windows `windows.h`. Another member preferred the name `windows_posix.h` to prevent potential issues. 
- **Compiler Warnings and Fixes**: There were warnings about potentially uninitialized local variables in various header files, which led to a commit to address these warnings. A member suggested ensuring `#ifndef _WIN32` around `dirent.h` to manage compatibility between different operating systems.
- **Implementation of "Resume Training" Flag**: A new `-y 1` flag was introduced to resume training automatically after interruptions, enhancing the training process efficiency. This feature proved helpful in reproducing a 350M parameter model over 14 hours at approximately $200.
- **Discussion on Memory Optimization for Backward Pass**: To conserve memory, members discussed recomputing layernorm during the backward pass instead of storing entire activations, potentially leading to efficiency gains. A member began implementing this approach, aiming to reduce memory footprint without sacrificing performance.
- **Hosting Large Dataset on S3**: The conversation touched on hosting FineWeb100B on S3, considering costs and dependency management. Alternatives like Zenodo or Ubicloud were explored, highlighting the need for an efficient and scalable data hosting solution.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/issues/478">Recalculating the activations in the backwards pass to conserve memory · Issue #478 · karpathy/llm.c</a>: @ngc92 Did an analysis of the areas that take up the most memory and its impact on the amount of batches that can be used and found that one of the largest contributors was the memory associated wi...</li><li><a href="https://github.com/karpathy/llm.c/pull/475">experiment with adding the llmc lib directory by karpathy · Pull Request #475 · karpathy/llm.c</a>: no description found</li><li><a href="https://zenodo.org/">Zenodo</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/discussions/481">Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20 · karpathy/llm.c · Discussion #481</a>: Let&#39;s reproduce the GPT-2 (124M) in llm.c (~4,000 lines of C/CUDA) in 90 minutes for $20. The 124M model is the smallest model in the GPT-2 series released by OpenAI in 2019, and is actually quite...</li><li><a href="https://github.com/karpathy/llm.c/pull/459.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://transmissionbt.com">Transmission</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/480">First step in creating option to recalculate layernorm activations in backwards pass by ChrisDryden · Pull Request #480 · karpathy/llm.c</a>: This CR is the first step in implementing the goal described in #478 to be able to reduce the memory footprint by adding an option to recalculate the layernorm activations in the backwards pass. Th...</li><li><a href="https://aws.amazon.com/s3/pricing/?p=pm&c=s3&z=4">Amazon S3 Simple Storage Service Pricing - Amazon Web Services</a>: no description found</li><li><a href="https://zenodo.org/records/3834942">OpenWebText</a>: An open-source replication of the WebText dataset from OpenAI. For more info please visit https://skylion007.github.io/OpenWebTextCorpus/ @misc{Gokaslan2019OpenWeb, title={OpenWebText Corpus}, author=...</li><li><a href="https://trac.transmissionbt.com/wiki/HeadlessUsage">
      HeadlessUsage     – Transmission

    </a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 messages): 

orion160: What are tools to debug SYCL code? In general stepping into kernel code....
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1244746994228924428)** (9 messages🔥): 

- **Gradient issues in activation quantized neural networks**: A member stated that *passing the incoming gradient directly is wrong,* and suggested using the gradient of a surrogate function, such as `tanh`. They referenced an [arXiv paper](https://arxiv.org/abs/1903.05662) that explains why even incorrect gradients can minimize training loss using a straight-through estimator (STE).

- **Trouble with C extensions in tests**: A member faced an `ImportError` when C extensions from `torchao` weren't importing properly. They speculated that their use of **cuda12.4** might be the issue, as **cuda12.1** is the default on PyPi.

- **Switching CUDA versions**: Another member suggested installing **cuda12.1** via `cudatoolkit` with conda as a potential solution. They also recommended opening an issue if the problem persists locally.

**Link mentioned**: <a href="https://arxiv.org/abs/1903.05662">Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets</a>: Training activation quantized neural networks involves minimizing a piecewise constant function whose gradient vanishes almost everywhere, which is undesirable for the standard back-propagation or cha...

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1244777737785184317)** (14 messages🔥): 

- **Members offer warm welcomes and share personal experiences**: Several new members introduced themselves, expressed their excitement to join, and shared their backgrounds in ML and development. One member mentioned feeling *"inadequate"* but eager to *"improve my agency"* by contributing to research.

- **Curiosity about prompt modification research**: A solo developer working on research and benchmarking inquired about existing research related to **prompt modification/structuring adjacent to MMLU**. They shared their personal experiment results, noting a *"10-20% hit on various categories"* after adapting inputs to fit Anthropic’s XML prompt syntax.

- **Research areas and community projects suggested**: A community member was directed to check out **community projects** for exploring research areas. The response was met with appreciation for the useful suggestion.

- **Request for help with reimplementing Facenet model**: A member requested assistance in reimplementing the Facenet model code from scratch using PyTorch. No replies or solutions were provided in the messages.

- **Question about LLMs on Databricks dismissed**: A member asked about conducting batch inferencing with LLMs hosted on **Databricks**. Another member advised that such questions are best directed to Databricks support, pointing out that Databricks is *"not known broadly for being compatible with things that are not Databricks"*.
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1244836234027859998)** (122 messages🔥🔥): 

- **GPTs Agents Cannot Learn After Initial Training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training. Another member clarified that uploaded files are saved as knowledge files for the agent to reference when required, but they do not continually modify the agent's base knowledge.

- **EM Distillation for Efficient Diffusion Model Sampling**: [A new paper from Google DeepMind proposes EM Distillation](http://arxiv.org/abs/2405.16852), a maximum likelihood-based approach that distills a diffusion model to a one-step generator model with minimal loss of perceptual quality. The technique introduces a reparametrized sampling scheme and noise cancellation to stabilize the distillation process.

- **Google Trains an 8B Parameter Diffusion Model for 1024x1024 Images**: Google researchers trained a non-cascaded pixel-space diffusion model to directly produce 1024x1024 images, [detailed in their new paper](http://arxiv.org/abs/2405.16759). Discussions included anticipation about comparisons with Imagen 3.

- **STLMs Aim to Minimize Parameters in LLMs**: [A new research effort](https://arxiv.org/abs/2405.14159) introduces Super Tiny Language Models (STLMs) that aim to reduce parameter counts by 90% to 95% while maintaining performance. The paper is vague on specific implementation details but mentions future work on tokenizer-free models, self-play, and alternative training objectives.

- **Questions on GPU Latency Modeling**: A member inquired about symbolically modeling GPU latencies without running the kernel or using a learned model. A helpful response provided links to relevant [research papers and a PhD thesis](https://inria.hal.science/hal-00789958/file/112_Lai.pdf) discussing theoretical approaches.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://arxiv.org/abs/2405.16852">EM Distillation for One-step Diffusion Models</a>: While diffusion models can learn complex distributions, sampling requires a computationally expensive iterative process. Existing distillation methods enable efficient sampling, but have notable limit...</li><li><a href="https://arxiv.org/abs/2405.15815">A social path to human-like artificial intelligence</a>: Traditionally, cognitive and computer scientists have viewed intelligence solipsistically, as a property of unitary agents devoid of social context. Given the success of contemporary learning algorith...</li><li><a href="https://arxiv.org/abs/2405.16039">MoEUT: Mixture-of-Experts Universal Transformers</a>: Previous work on Universal Transformers (UTs) has demonstrated the importance of parameter sharing across layers. By allowing recurrence in depth, UTs have advantages over standard Transformers in lea...</li><li><a href="https://arxiv.org/abs/2405.16759">Greedy Growing Enables High-Resolution Pixel-Based Diffusion Models</a>: We address the long-standing problem of how to learn effective pixel-based image diffusion models at scale, introducing a remarkably simple greedy growing method for stable training of large-scale, hi...</li><li><a href="https://arxiv.org/abs/2405.14159">Super Tiny Language Models</a>: The rapid advancement of large language models (LLMs) has led to significant improvements in natural language processing but also poses challenges due to their high computational and energy demands. T...</li><li><a href="https://arxiv.org/abs/2405.17399?s=09">Transformers Can Do Arithmetic with the Right Embeddings</a>: The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend th...</li><li><a href="https://x.com/wenhuchen/status/1795094212230168715?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Wenhu Chen (@WenhuChen)</a>: There is a misconception that any leakage of a particular benchmark will lead to huge improvement on that benchmark.  It&#39;s not necessarily true. We found that it actually depends on the format of ...</li><li><a href="https://arxiv.org/abs/2203.14309">DeepDPM: Deep Clustering With an Unknown Number of Clusters</a>: Deep Learning (DL) has shown great promise in the unsupervised task of clustering. That said, while in classical (i.e., non-deep) clustering the benefits of the nonparametric approach are well known, ...</li><li><a href="http://arxiv.org/abs/2405.16759">Greedy Growing Enables High-Resolution Pixel-Based Diffusion Models</a>: We address the long-standing problem of how to learn effective pixel-based image diffusion models at scale, introducing a remarkably simple greedy growing method for stable training of large-scale, hi...</li><li><a href="https://arxiv.org/abs/2005.05744">Deep Learning: Our Miraculous Year 1990-1991</a>: In 2020-2021, we celebrated that many of the basic ideas behind the deep learning revolution were published three decades ago within fewer than 12 months in our &#34;Annus Mirabilis&#34; or &#34;Mirac...</li><li><a href="https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/">Rethinking How to Train Diffusion Models | NVIDIA Technical Blog</a>: After exploring the fundamentals of diffusion model sampling, parameterization, and training as explained in Generative AI Research Spotlight: Demystifying Diffusion&#x2d;Based Models&#8230;
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1244967598723043358)** (2 messages): 

- **New Models Released**: Announcing the launch of [Mistral 7B Instruct v0.3](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3) and [Hermes 2 Pro - Llama-3 8B](https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b). The Mistral 7B Instruct and its free variant now point to the latest [v0.3 version](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3).

- **Versioned Model Access**: Older versions like [Mistral 7B Instruct v0.2](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2) and [v0.1](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.1) remain accessible.

- **OpenAI Outage Resolved Quickly**: There was a brief outage affecting OpenAI usage. However, they swiftly resolved it, with Azure and its fallback remaining operational during the downtime.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3>)">Mistral: Mistral 7B Instruct by mistralai | OpenRouter</a>: A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b>)">NousResearch: Hermes 2 Pro - Llama-3 8B by nousresearch | OpenRouter</a>: Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mod...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct>)">Mistral: Mistral 7B Instruct by mistralai | OpenRouter</a>: A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2>)">Mistral: Mistral 7B Instruct v0.2 by mistralai | OpenRouter</a>: A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.  An improved version of [Mistral 7B Instruct](/modelsmistralai/mistral-7b-instruct-v0.1), wi...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.1>)">Mistral: Mistral 7B Instruct v0.1 by mistralai | OpenRouter</a>: A 7.3B parameter model that outperforms Llama 2 13B on all benchmarks, with optimizations for speed and context length.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1244823757810434089)** (1 messages): 

- **Inquiry about models on Max Loh's website**: A member asked which models are being used on [Max Loh's website](https://www.maxloh.com). They also inquired if anyone knows how to find a list of all the uncensored models available on OpenRouter.
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1244733210198409308)** (122 messages🔥🔥): 

- **Debate Over Phi-3 Vision Costs and Availability**: Discussion revolves around the high cost of using **Phi-3 Vision** on Azure, with a member suggesting that *"looking at llama prices I'd hit $0.07/M"*. Another member counters, pointing out that other providers also charge similarly.
  
- **Gemini's Superior OCR Capabilities**: Members discuss the **OCR capabilities** of Gemini, with claims that it "can read Cyrillic text pretty well" and is "better than Claude and GPT-4o" in reading both Cyrillic and English texts.

- **Langchain and Streamlit for Python Chatbots**: Inquiries were made about suitable templates for building a Flask-based chatbot. Recommendations included checking out **Streamlit templates** and **Langchain**, with emphasis on easy integrations and the possibility of using database adapters.

- **OpenRouter Token Costs Clarifications**: Participants debated the costs involved with **OpenRouter** tokens, clarifying that $0.26 buys 1M input + output tokens and discussing how token count affects pricing. Fry69_61685 emphasizes that each chat interaction recounts the entire history, increasing token usage.

- **Handling OpenAI Model Outages**: An outage affected OpenAI **GPT-4o**, causing interruptions in service. Alex Atallah reassured users by confirming the issue was fixed quickly and promising better checks in the future.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://useinstructor.com/">Welcome To Instructor - Instructor</a>: no description found</li><li><a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground">The Tokenizer Playground - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b">MythoMax 13B by gryphe | OpenRouter</a>: One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1244781840183721984)** (6 messages): 

- **Song Translation Challenges Explored**: A member inquired about the current state of **song translation**, specifically about maintaining the tone with some form of control over the lyrics. The interest lies in managing lyrical translation while preserving the artistic intent.
  
- **Greentext AGI Scenario**: A member found it intriguing to use LLMs to create **4chan greentext snippets**. They asked the LLM to generate a greentext about waking up and discovering AGI was created, noting that the results were particularly interesting.
  
- **Concerns Over Project Management**: There's a discussion regarding a user who is hesitant to adopt another platform for an **OpenCL extension** due to concerns about codebase size. The member expressed disinterest in contributing unless the code is upstreamed, critiquing the project management approach.
  
- **CrewAI Video Shared**: A member shared a [YouTube video](https://www.youtube.com/watch?v=Czhc0L2bqWo), "CrewAI Introduction to creating AI Agents". The video provides a tutorial on creating AI agents using CrewAI, including a link to the [CrewAI documentation](https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/#python).
  
- **Tech Culture in University Cities**: An upcoming grad student is seeking recommendations for universities in cities with robust tech cultures. They are interested in places like SF, Munich, and NYC for reading groups or hackathons, aiming to connect with peers working on similar AI projects.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Czhc0L2bqWo">CrewAI Introduction to creating AI Agents</a>: We will take a look at how to create ai agents using crew aihttps://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/#python #pythonprogramming #llm #m...

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1244739332540072158)** (63 messages🔥🔥): 

- **Phi model training debate**: Members discussed whether Phi used a majority of textbook data or synthetic data for training. One noted, "Phi used majority textbooks," while another corrected, "The paper claims it's a mix of heavily filtered public data and synthetic data."

- **Logic and self-correction in LLMs**: Users tested LLMs' ability to provide logical explanations and self-correct, noting failures. One user observed, “It treats questioning its logic as if I said 'That's wrong.'” while another commented, “Models that always agree with the user might turn out stupid too.”

- **Epochs and batch sizes for fine-tuning**: Users shared opinions on the number of epochs and batch sizes for model fine-tuning. One suggested, "1-3 is good generally," and another added, "4-6 is over-fitting territory but can work."

- **RAG-in-a-box solutions**: A user inquired about recommendations for uploading thousands of PDFs for RAG searches. Another explained that building a proper RAG solution depends on many factors, including the type of data and specific queries.

- **Arithmetic in transformers**: The potential complexity of transformers performing arithmetic operations was explored. A member described it as, "layer by layer transformation of partial results," highlighting fundamental limitations in handling arithmetic with token prediction models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medicalxpress.com/news/2024-05-ai-large-language-align-human.amp">
      Improving AI large language models helps them better align with human brain activity
          </a>:        With generative artificial intelligence (GenAI) transforming the social interaction landscape in recent years, large language models (LLMs), which use deep-learning algorithms to train GenAI pl...</li><li><a href="https://arxiv.org/abs/2405.17399">Transformers Can Do Arithmetic with the Right Embeddings</a>: The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend th...</li><li><a href="https://osf.io/94y7h/.">
        Predicting the next sentence (not word) in large language models: What model-brain alignment tells us about discourse comprehension
</a>:      Hosted on the Open Science Framework 
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1244733900966592542)** (28 messages🔥): 

- **Fitting 70B in a Single A100 GPU**: *"Jaredquek"* discussed using axolotl with seq len at 1200 and text completion, utilizing 98% of GPU memory for 8bit lora. He noted that with qlora, there is more spare capacity.
- **Experimental RepEng Vectors in Hermes**: *"Max_paperclips"* highlighted that subtracting the honesty vector in Hermes caused it to break quickly, while other models like Mistral responded differently. Azure2089 and others noted similar experiences, with Azure2089 providing a [link](https://github.com/cpldcpu/MisguidedAttention/blob/main/repeng_02_river_crossing.md) to prompts addressing model reasoning in presence of misleading information.
- **Challenges in Creating DPO Datasets**: Lokesh8882 and Thilotee exchanged views on starting a DPO dataset with custom organization data, with Thilotee suggesting reference to scientific papers. Dumball noted that DPO requires specific formats, and linked to [Hugging Face's TRL documentation](https://huggingface.co/docs/trl/main/en/reward_trainer) as an example.
- **Custom Dataset Formats for DPO Training**: Thilotee provided a resource from Hugging Face TRL on the DPO Trainer for language models from preference data, described in [this paper](https://arxiv.org/abs/2305.18290). Dumball confirmed that DPO requires data in the form of prompt, chosen, and rejected responses.
- **Concept Proposal: Small LLM with Larger Attention Mechanism**: Bliponnobodysradar suggested the idea of training a small LLM like llama3 8b with a larger attention mechanism to achieve the context awareness of a larger model, inviting feedback on the idea.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/main/en/reward_trainer#trl.RewardTrainer">Reward Modeling</a>: no description found</li><li><a href="https://github.com/cpldcpu/MisguidedAttention/blob/main/repeng_02_river_crossing.md">MisguidedAttention/repeng_02_river_crossing.md at main · cpldcpu/MisguidedAttention</a>: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information - cpldcpu/MisguidedAttention</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">DPO Trainer</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1244796325023846481)** (2 messages): 

- **Pooling Resources for RAG Projects**: A member shared their sentiment and semantic density smoothing agent project, available on [GitHub](https://github.com/EveryOneIsGross/densefeelsCHAT), and mentioned they are back from a break and keen to pool resources. They noted that the TTS component might need some preparation to run smoothly, possibly requiring model caching.
- **Video Game Break and SLURM Porting**: Another member mentioned they had taken a week off to play video games and are now focusing on porting their project, Cynde, to SLURM as their next task.

**Link mentioned**: <a href="https://github.com/EveryOneIsGross/densefeelsCHAT">GitHub - EveryOneIsGross/densefeelsCHAT: sentiment and semantic density smoothing agent. w/ tts</a>: sentiment and semantic density smoothing agent. w/ tts - EveryOneIsGross/densefeelsCHAT

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 messages): 

jakekies: hi
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1244787381471674459)** (76 messages🔥🔥): 

<ul>
<li>
<b>Agent in Continuous Loop Problem</b>: A user reported an issue with their LangChain agent entering a continuous loop of tool calling, seeking solutions to make the agent provide a final response.
</li>
<li>
<b>Calling Requests as a Tool</b>: Extensive discussions and code examples were shared on how to call requests as a tool in LangChain. Solutions included using <code>JsonRequestsWrapper</code> and creating a <code>ProgramSearchTool</code> to dynamically add values to parameters based on user input, with examples provided.
</li>
<li>
<b>Token Limit Error After Update</b>: A user mentioned an error when updating LangChain to version 0.2.2, where the context length restriction of 16385 tokens was being incorrectly applied to models supposedly supporting up to 128k tokens. They sought community support to resolve this discrepancy.
</li>
<li>
<b>SQL Agent Prompt Template</b>: A member requested and received a prompt template for an SQL agent that includes few-shot examples to guide the agent in generating correct SQL queries. Instructions were given on how to construct the prompts and use them within LangChain.
</li>
<li>
<b>Query on LangChain v2.0</b>: A user inquired about the presence of agents in LangChain version 2.0, expressing difficulty in locating relevant features in the updated version.
</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/1580>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13826>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/14508>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2140>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/3838>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/tools/requests/#inside-the-tool>)">Requests | 🦜️🔗 LangChain</a>: The web contains a lot of information that LLMs do not have access to. In order to easily let LLMs interact with that information, we provide a wrapper around the Python Requests module that takes in ...</li><li><a href="https://python.langchain.com/v0.1/docs/use_cases/sql/prompting/#few-shot-examples>)">Prompting strategies | 🦜️🔗 LangChain</a>: In this guide we&#x27;ll go over prompting strategies to improve SQL query generation. We&#x27;ll largely focus on methods for getting relevant database-specific information in your prompt.</li><li><a href="https://github.com/langchain-ai/langchain/issues/16731>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1244895685728403456)** (4 messages): 

- **Custom kwargs in Langserve with Langsmith go missing**: A member is trying to send custom "kwargs" with their request in Langserve to track and log data in Langsmith. They report that these kwargs do not appear in the Langsmith log item and are looking for solutions.
- **Configurable Pinecone namespace request**: A member inquires about making the namespace of the Pinecone store configurable to change the namespace based on the user making the API call. They included a code snippet but did not receive an explicit solution in the messages.
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1244891137508900864)** (4 messages): 

- **Generative AI for Drug Discovery**: A member announced an upcoming event titled *"Local Generative AI Model Frameworks for Different Stages of Drug Discovery"* scheduled for May 30. More details can be found on [LinkedIn](https://www.linkedin.com/events/localgenerativeaimodelframework7200655391901323264/).

- **Cutting Costs on Logging**: A member shared a pipeline for removing redundant logs to help companies save money. They recommended using [this tool](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=langchain), selecting the "verbose logs" pipeline.

- **Flight Simulator Co-Pilot**: A project is in progress to create a co-pilot for flight simulators like Microsoft Flight Simulator. Check out a demonstration video on [YouTube](https://www.youtube.com/watch?v=bUWcQSwZyPQ).

- **Routing Logic in Agent Flows**: An informative video on using routing logic in agent flows with Visual Agents, built on LangChain, was shared. Watch the YouTube video [here](https://youtu.be/KtbRexZ6vsc).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=langchain)">GitGud</a>: no description found</li><li><a href="https://youtu.be/KtbRexZ6vsc">How to Route Logic in Your Agent Flows</a>: Simple example of how to use routing logic in your agent flows with Visual Agents, built on LangChain.https://visualagents.aihttps://langchain.ai
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1244747313117925396)** (20 messages🔥): 

- **Mojo Python Version Support Reminder**: *Please make sure you're using the supported Python version 3.8 to 3.11. Note that 3.12 isn't supported yet.* Resolved version issues by adding the deadsnakes repo and updating to 3.11.
- **Discussion on Tensor Package Deprecation**: Questions arose about Tensor being deprecated as seen in a [YouTube video](https://youtu.be/uIG9q9foIw0?si=rhPqeQ_SsN8MIFur&t=1954). Clarifications were made that Tensor will be open-sourced and removed from the standard library.
- **Making Mojo Practical for Flutter Apps**: A member suggested using Mojo to build apps with Flutter for enhanced speed and deployability, referencing a [YouTube tutorial](https://www.youtube.com/watch?v=5P8f5Tlim0M&t=278). The versatility of combining Flutter's UI capabilities with Mojo was highlighted.
- **Interest in LLaMA2.mojo Project**: Users showed interest in tutorials for the [llama2.mojo GitHub project](https://github.com/tairov/llama2.mojo), focusing on inference and AI model fine-tuning in Mojo. Community members were invited to join a Discord server to discuss this further.
- **Low-Level GPU Assembly Code Queries**: *Where should I ask questions relating to low level GPU assembly code?* Discussion pointed towards using tools like Nsight for Python/Mojo.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=5P8f5Tlim0M&t=278">Build Flutter Apps With Python - Flet Tutorial</a>: In this video I will build a MacOS Flutter App with Python. We combine the best of both world, the UI capabilities of Flutter and the ecosystem of Python. We...</li><li><a href="https://youtu.be/uIG9q9foIw0?si=rhPqeQ_SsN8MIFur&t=1954">Mojo Community Meeting #1</a>: Mojo Community Meeting Public Agenda: https://modul.ar/community-meeting-doc</li><li><a href="https://github.com/tairov/llama2.mojo">GitHub - tairov/llama2.mojo: Inference Llama 2 in one file of pure 🔥</a>: Inference Llama 2 in one file of pure 🔥. Contribute to tairov/llama2.mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1245077923401171004)** (6 messages): 

- **Open-world games use AI to customize NPC intelligence levels**: A member suggests that open-world games could offer subscription packages based on the intelligence of NPCs, increasing costs for more "sentient" NPCs. They emphasize this would be an online feature only.
- **Smart devices add special AI capabilities**: Another member shares that future AI inference will likely occur locally as many smart devices are now shipping with accelerators and CPUs are adopting special registers for matrix multiplication. This hardware shift indicates a move toward more distributed AI processing.
- **Custom worlds in open-world games**: Expanding on the previous idea, a member envisions open-world games that use AI to build custom worlds based on player interaction. They see a potential to leverage vast online model libraries to enhance gameplay and personalization.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1244753855925977089)** (14 messages🔥): 

- **Mojo Supports Circular Dependencies**: A member questioned how modules can define each other in Mojo. Another clarified that Mojo allows circular dependencies due to the way modules are modeled, particularly with the `__init__.mojo` root ([example explanation](https://github.com/dorjeduck)).
- **Built-in Traits Import Automatically**: Queries regarding the visibility of traits like `Intable` and `Stringable` without explicit import were answered. It was explained that such traits are part of the built-in package and are hence automatically imported.
- **Dual Purpose of ^ Operator**: Discussion clarified the dual functionality of the `^` operator in Mojo. It's used for both XOR operations and signaling the end of an object's lifetime to the compiler.
- **Callbacks Possible but No Lambdas Yet**: Members discussed the use of callback functions in Mojo and noted that lambda functions are not implemented yet. Current alternatives like passing functions as method parameters were explored.
- **Enhancing the `vectorize` Function**: A proposal was made to modify the `vectorize` function to allow the closure function to return a `Bool` for loop control, akin to functionality in a [progress bar project](https://github.com/dorjeduck/progressbar.mojo). This sparked interest and further exploration among members.
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1244731477489352778)** (21 messages🔥): 

- **50x Speedup at 32 Bytes Hits Cache Issues**: *fnands* shares a detailed benchmark showcasing performance improvements with varying byte lengths, achieving a top speedup of *50x at 32 bytes* before bumping into cache limitations. They invite Apple silicon users to test further using a [GitHub file](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo).

- **Disparity in k-means Benchmarks Discussed**: Cyrus_msk points out that different memory allocation practices and matrix implementations make benchmark comparisons between Python and Mojo's k-means algorithms non-equivalent. Highlights include *preallocated memory in Mojo* and `BLAS norm` vs. a parallel SIMD hacked norm function.

- **Prefetching and Caching Discussion**: *Fnands* seeks advice on `prefetch` options to improve performance, discussing performance impact with and without explicit prefetching. Darkmatter__ suggests using tools like Intel's [VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html) for detailed CPU performance insights and emphasizes the importance of cache-line alignment for efficient memory access.

- **Aligning Memory for Better Performance**: Darkmatter__ advises on the importance of ensuring that memory tables are *64-byte aligned* to optimize performance on AVX512 operations, reducing cache mismanagement and encouraging prefetching. They also clarify that avoiding false sharing is crucial primarily in multithreaded scenarios.

**Link mentioned**: <a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo">fnands.com/blog/2024/mojo-crc-calc/crcn.mojo at main · fnands/fnands.com</a>: My personal blog. Contribute to fnands/fnands.com development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1244792723651694655)** (13 messages🔥): 

- **Interest in Reference Returns**: Members discussed whether functions returning `Reference` are good candidates for contributing to the new return convention. One member remarked, *"almost all functions that return `Reference` 'should' be converted,"* with exceptions being those where the reference is often stored by users.
- **Excitement Over Structural Sharing**: A member expressed excitement about the reference changes, highlighting that it *"enables structural sharing,"* meaning multiple structs can share some fields.
- **New Nightly Mojo Compiler Released**: The latest nightly Mojo compiler version `2024.5.2805` was announced. The update includes implementations like `tempfile.{mkdtemp,gettempdir}` and the addition of `String.isspace()` to the standard library; full changes are detailed in the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and [raw diff](https://github.com/modularml/mojo/compare/ce285fded710b403e1b7b5637183ea20fa4d5c97...4724ec6ff46378f6a1d6190ca9a76916a5faaba3).
- **Contributor Clarification**: One member clarified they are not Modular staff but a contributor. 
- **Discussion about PR Secrecy**: There was a brief conversation on where to comment about an issue, with humor about PR confidentiality as one member noted they are *"still waiting for a green light"* on an undisclosed PR.
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1244734669988040746)** (68 messages🔥🔥): 

- **Cursor Interpreter Mode Wows User**: A user praises **cursor interpreter mode** for its debugging capabilities, describing it as *"a better search that can follow execution path"* and being more agentic in navigating a codebase compared to traditional search tools.

- **Microsoft’s Copilot Now on Telegram**: Users are excited about **Microsoft Copilot** integrating into **Telegram** for a smarter chat experience. The tool offers features like gaming tips, movie suggestions, dating advice, and recipes, optimizing everyday conversations.

- **Training GPT-2 on a Budget**: **Andrej Karpathy** shares a method to train GPT-2 (124M) in **90 minutes for $20** using llm.c, highlighting that GPU constraints can be managed cost-effectively. Detailed instructions are provided in a [GitHub discussion](https://github.com/karpathy/llm.c/discussions/481).

- **Microsoft Separates Copilots vs Agents**: Discussion about **Microsoft Build's** decision to differentiate **Copilots** and **Agents**, with Copilots being more personalized and prompt-based while Agents operate autonomously. A relevant [interview with Kanjun Qiu](https://www.latent.space/p/imbue) was noted as insightful.

- **Vector Database Integration Queries**: A user seeks **vector database abstractions** similar to ORMs for easier integration and swapping of different vector databases. **LangChain** and **LlamaIndex** are suggested, with further recommendations for **pgvector** for efficient embedding storage and relational metadata management.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>: no description found</li><li><a href="https://blog.reachsumit.com/posts/2023/03/llm-for-text-ranking/">Zero and Few Shot Text Retrieval and Ranking Using Large Language Models</a>: Large Language Models (LLMs), like GPT-x, PaLM, BLOOM, have shaken up the NLP domain and completely redefined the state-of-the-art for a variety of tasks. One reason for the popularity of these LLMs h...</li><li><a href="https://x.com/xlr8harder/status/1795515600795767058">Tweet from xlr8harder (@xlr8harder)</a>: @karpathy @swyx There aren&#39;t a lot of good options to get a good sample at small token counts. FineWeb making smaller samples available is the exception rather than the rule, unfortunately.  But S...</li><li><a href="https://x.com/karpathy/status/1795484547267834137">Tweet from Andrej Karpathy (@karpathy)</a>: # Reproduce GPT-2 (124M) in llm.c in 90 minutes for $20 ✨  The GPT-2 (124M) is the smallest model in the GPT-2 series released by OpenAI in 2019, and is actually quite accessible today, even for the G...</li><li><a href="https://x.com/borismpower/status/1795475031658516933?s=46">Tweet from Boris Power (@BorisMPower)</a>: 4+1=5</li><li><a href="https://x.com/polynoamial/status/1795422304937411029?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Noam Brown (@polynoamial)</a>: The next @OpenAI frontier model has started training! https://openai.com/index/openai-board-forms-safety-and-security-committee/</li><li><a href="https://www.microsoft.com/en-us/edge/copilot-for-social?form=MY02F9">Copilot for Telegram | Microsoft Copilot</a>: no description found</li><li><a href="https://x.com/GergelyOrosz/status/1794743519954731331">Tweet from Gergely Orosz (@GergelyOrosz)</a>: If building an AI coding agent performing ~4x better than the best LLMs has a billion-dollar potential:  Here are 7 Princeton researchers who did this.  It&#39;s all open source, and called SWE-agent....</li><li><a href="https://www.microsoft.com/en-us/edge/copilot-for-social?form=MY02F9&ch=1">Copilot for Telegram | Microsoft Copilot</a>: no description found</li><li><a href="https://www.latent.space/p/imbue">Why AI Agents Don&#x27;t Work (yet) - with Kanjun Qiu of Imbue</a>: Listen now | On raising $200m to build agent operating systems that can reason and code, why LLMs beat reinforcement learning for agent usecases, and how to build a Scenius with top AI people</li><li><a href="https://x.com/khoomeik/status/1795477359933706272">Tweet from Rohan Pandey (e/acc) (@khoomeik)</a>: 📢 Excited to finally be releasing my NeurIPS 2024 submission!  Is Chinchilla universal? No! We find that: 1. language model scaling laws depend on data complexity 2. gzip effectively predicts scaling...</li><li><a href="https://anysphere.inc/blog/problems-2024">Problems for 2024-2025.</a>: no description found</li><li><a href="https://x.com/siddrrsh/status/1795541002620727439?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Siddharth Sharma (@siddrrsh)</a>: Introducing Llama3-V, a SOTA open-source VLM model  We feature: • Outperforms LLaVA • Comparable performance to GPT4-V, Gemini Ultra, Claude Opus with a 100x smaller model • SOTA open source VLM for L...</li><li><a href="https://x.com/lmsysorg/status/1795512202465845686">Tweet from lmsys.org (@lmsysorg)</a>: Big news – Gemini 1.5 Flash, Pro and Advanced results are out!🔥  - Gemini 1.5 Pro/Advanced at #2, closing in on GPT-4o - Gemini 1.5 Flash at #9, outperforming Llama-3-70b and nearly reaching GPT-4-01...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1244758635419009045)** (3 messages): 

```html
- **New podcast on ICLR 2024 papers**: A new episode covering highlights from ICLR 2024 has been released, featuring various groundbreaking papers and talks. [Listen here](https://x.com/latentspacepod/status/1795196817044594817) for insights on ImageGen, Compression, Adversarial Attacks, Vision Learning, and more.
- **Spotlight on ImageGen and Compression**: Topics discussed include "Auto-encoding Variational Bayes" and "Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models". Notable mentions are detailed insights from Ilya Sutskever and Christian Szegedy.
- **Vision Learning advancements**: The podcast delves into papers like "Vision Transformers Need Registers" and "Think before you speak: Training Language Models With Pause Tokens". It also investigates the statistical theory of data selection under weak supervision.
- **Enhancing Transformer models**: Discussion on efficient fine-tuning and context window extension of large language models with papers like "LongLoRA" and "YaRN". Topics like adaptive KV cache compression and efficient communication for giant model training also featured.
- **State Space Models vs Transformers**: The importance of data-driven priors in long-sequence models is highlighted in the paper "Never Train from Scratch". Stay tuned for more content on LLM Reasoning and Agents in Part 2.
```

**Link mentioned**: <a href="https://x.com/latentspacepod/status/1795196817044594817">Tweet from Latent Space Podcast (@latentspacepod)</a>: 🆕 ICLR 2024: Best Papers (Part 1)  We present our selections of outstanding papers and talks thematically introducing topics for AI Engineers to track:  Section A: ImageGen, Compression, Adversarial ...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1245093324428283965)** (1 messages): 

- **FinTextQA: New financial dataset launched**: Check out [FinTextQA](https://t.co/emhQYXY1S4), a new dataset and RAG benchmark for long-form financial question answering from Jian Chen and team. It features *6 different question types* and includes *1,262 high-quality, source-attributed question-answer pairs* with associated document contexts.
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1244754850747515014)** (59 messages🔥🔥): 

- **Achieving Perfect System Role Prompt Structures**: A user asked for an article on perfect system role prompt structures similar to what **LlamaIndex** uses in their examples.
  
- **Persistence of Chat Histories in LlamaIndex**: Users discussed how to save chat history for results from **NLSQL** and **PandasQuery** engines. Suggestions included creating a custom retriever to wrap the query engine.

- **Handling Multiple Functions in API with Function Calling**: Members brainstormed strategies for managing an API with 1000 separate functions. Ideas included hierarchical routing to divide functions into manageable subgroups.

- **Differences Between LLMSelector and PydanticSelector**: Detailed explanations were provided on how **LLM selectors** use text completion endpoints to generate query data, while **Pydantic selectors** use pydantic objects for function calling API.

- **Challenges and Solutions in RAG Systems with LlamaIndex**: Users discussed metadata handling, the impact of embedding smaller semantic chunks versus larger chunks, and potential trade-offs in information retrieval accuracy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/examples/query_engine/RouterQueryEngine#define-router-query-engine>)">Router Query Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/router#defining-a-selector>)">Routing - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1244733174014279712)** (40 messages🔥): 

- **Humor in Shitty SOTA AGI Predictions**: Various members were discussing the humor and despair in the current state of SOTA AGI models. One interesting note was that a model supposedly trained itself, with remarks like "it has trained a model for us."
- **Corcelio's Mobius Art Model on Hugging Face**: Images generated by Corcelio's [Mobius Art Model](https://huggingface.co/Corcelio/mobius) were shared with prompts ranging from "Thanos smelling a little yellow rose" to "The Exegenesis of the soul." It was noted for surpassing limitations of previous models but surprisingly generating watermarks.
- **Community Concerns on Image System's Ethics**: There were concerns raised about people using image generation systems to create inappropriate content, including "porn of stuff they shouldn't lol." This issue brought up questions about the site's prompts and sampler settings.
- **Undertrained Model Issues and Watermarks**: The new unnamed T2I model on imgsys was discussed, with observations that it often appears undertrained and routinely generates watermarks. Some members found this a recurring and humorous theme.
- **Elon Musk's Tweet Mocking CNNs**: A shared tweet from Elon Musk highlighted that they "don’t use CNNs much these days," which spurred humorous reactions like advising to use vision transformer models instead. Members joked about the industry's shifting trends and the use of irony in responses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Corcelio/mobius">Corcelio/mobius · Hugging Face</a>: no description found</li><li><a href="https://x.com/elonmusk/status/1795405972145418548">Tweet from Elon Musk (@elonmusk)</a>: @ylecun @Scobleizer We don’t use CNNs much these days tbh
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1245086640259727443)** (2 messages): 

- **SDXL struggles with generating "reading eyes"**: A member discovered that **SDXL** couldn’t generate a close-up portrait of a woman reading. They shared their detailed prompt and generation settings used in the horde.
- **Call to generate data via DALL-E**: The same member tagged other users to suggest generating the image with **DALL-E**. They aim to create a synthetic database to use as training material to improve "reading eyes" generation with **SDXL**.
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1244830671160213585)** (25 messages🔥): 

- **GPU Latency Modeling Might Be Possible**: A member suggested that symbolically modeling GPU latencies and runtime could be possible without running kernels, based on data movement between memory types and operation times. However, occupancy and asynchronous operations might complicate the model.

- **Halide and AutoTVM Explored for Kernel Optimization**: Members discussed tools like AutoTVM and Halide autotuner, noting that Halide uses a learned weighting of a hand-coded model and AutoTVM likely uses empirical methods. George Hotz pointed out that TVM uses XGBoost and emphasized the importance of properly emulating the cache hierarchy for accurate modeling.

- **Cycle Accurate GPU Simulators are High Precision**: There's speculation that quant firms might use cycle accurate GPU simulators for kernel optimization, providing very detailed profiling capabilities. However, the advantage of these simulators over empirical methods in terms of evaluation speed was questioned.

- **AMD MES Open-Source Release Anticipated**: There was brief mention of AMD's plans to open-source MES, with the documentation apparently released and the source code eagerly awaited by the community.

- **Differences in GPU Strategies for Latency Hiding**: Members highlighted that different GPUs employ various latency-hiding strategies, making it difficult to model latencies accurately. The large number of concurrent wavefronts/blocks in GPUs allows them to handle latencies more efficiently than might be assumed.

**Link mentioned**: <a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: how make gpu fast?

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1244732460726747166)** (5 messages): 

- **Post dominator analysis confusion**: A member questioned why **post dominator analysis** isn't used during scheduling to identify self-contained subgraphs for fusion. This technique could theoretically enhance efficiency in certain computations.
  
- **Creating a LazyBuffer with multiple values**: A member inquired about creating a **LazyBuffer** from an array of values rather than a single one. The response alluded to using *Load.EMPTY -> Load.COPY* as the general method and mentioned factory methods like `full` and `rand` for easy creation.
  
- **Code pointers offered for clarity**: After a detailed explanation, one member expressed readiness to provide **code pointers** for better understanding. The initial solution reference included insights on simulating buffer allocation for **LazyBuffer** creation.
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1244792392154878012)** (25 messages🔥): 

- **Elevenlabs Text-to-Speech Integration Debuts**: A user inquired about text-to-speech mods and another mentioned **Elevenlabs** integration in a branch. [Code for textToSpeech](https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19) was provided.

- **Implement Text-to-Speech in AI Town**: The procedure to set up the text-to-speech feature involves converting text to audio, patching messages with audio URLs, and handling audio play on the frontend. It's noted that the process is fast but has an almost one-second delay, making real-time implementation challenging.

- **Interest in Science Debates**: A user expressed their intention to create an engaging experience by having AI chatbots debate science topics. This user values science’s power to bring people together and create hope.

- **Add Eavesdropping Mechanic**: The Zaranova fork includes an eavesdropping mechanic, generating audio for nearby conversations. This feature could be added to enrich AI Town's interactive experience.

- **Interest in Collaborative Coding**: Another user showed interest in contributing to this feature by promising to look into creating a pull request. There was also interest in merging these changes into the AI Town main project.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/aiTown/agent.ts#L568">ai-town/convex/aiTown/agent.ts at e7e2182eb7f7241e58c69d8324ae126c1d34dee9 · huevosabio/ai-town</a>: A MIT-licensed, deployable starter kit for building and customizing your own version of AI town - a virtual town where AI characters live, chat and socialize. - huevosabio/ai-town</li><li><a href="https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19">ai-town/convex/util/textToSpeech.ts at e7e2182eb7f7241e58c69d8324ae126c1d34dee9 · huevosabio/ai-town</a>: A MIT-licensed, deployable starter kit for building and customizing your own version of AI town - a virtual town where AI characters live, chat and socialize. - huevosabio/ai-town
</li>
</ul>

</div>
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/)** (1 messages): 

gomiez: hi. how do i stop conversations from closing? i cant read that fast
  

---


### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/)** (1 messages): 

angry.penguin: LMK if you have any luck with inference
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1244728730039881800)** (12 messages🔥): 

- **Logging pipeline optimized to save costs**: A member shared they developed a pipeline to remove redundant logs, potentially added by mistake and pushed to production, to save costs on logging. They used [this tool](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere) and recommended selecting the "verbose logs" pipeline.

- **Inquiries about cloud-prem deployment options**: There was a question about cloud-prem deployment options for reranking and query extraction. The member sought insights on available solutions or best practices.

- **Fine-tuning Cohere model for RAG**: A user inquired if Cohere models could be fine-tuned to answer Financial Questions and then used with RAG (Retrieve and Generate) to focus responses based on SEC Filings.

- **Aya23 models restricted to non-commercial use**: It was clarified that **Aya23 models** are limited to non-commercial usage as they are intended for research purposes only. No plans for commercial use were indicated, even for small startups.

- **Creating a DPO dataset with custom data**: One member asked for tips on creating a custom DPO dataset, considering options like generating response pairs using GPT-4 or combining GPT-4 output with base model responses.

**Link mentioned**: <a href="https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere)">GitGud</a>: no description found

  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1244762231497818193)** (3 messages): 

- **Cohere-powered gaming bot launches**: A member shared about their new creation, a gaming bot for Discord using **Cohere Command R**. They mentioned that the bot, **Create 'n' Play**, features *"over 100 engaging text-based games"* and is designed to enhance social engagement with **AI**. 

- **Check out the gaming bot on LinkedIn**: The [LinkedIn post](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios) offers additional insights into the project's development and functionalities. It aims at easy team formations and interactions within the Discord community.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1244883894264926329)** (4 messages): 

- **Inference-only Query Sparks Discussion**: A member asked if the topic was for inference only. The response moved the conversation towards training complexities and performance considerations.
- **Training Bottlenecks Center on FLOPS**: "Training is almost always bound by flops." When dealing with a batch size of 1 and sequence length of 4096, the effective batch size turns out to be 4096 due to teacher forcing methods.
- **FP8 Native Training on Hopper Cards**: There was expressed interest in exploring "fp8 native training on hopper cards." This suggests a focus on advanced hardware capabilities for optimized training performance.
- **Acknowledgment of Past Mistake**: A member humorously admitted, *"True, I was wrong back then."* This exchange shows a culture of open acknowledgment and learning within the community.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1244982851326185474)** (1 messages): 

- **Update fschat properly**: A member explained that the version identifier of **fschat** was not updated, causing issues. They suggested uninstalling and reinstalling **fschat** to resolve this issue.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1245054726706430054)** (4 messages): 

- **Clarifying CUTLASS_PATH necessity**: A member asked if they should set `CUTLASS_PATH`. Phorm responded to check if their project or tools require **CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers)**, which is essential for high-performance matrix operations in deep learning applications.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=58c281a8-ece1-46c7-8057-cd6cb7902a51)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1244912674362364015)** (2 messages): 

- **YI and YI-VL models updated to Apache 2.0**: A member shared that **YI and YI-VL (multimodal LLM) models** have been updated to **Apache 2.0**, joining the 1.5 series. The update was announced by [@_philschmid](https://fxtwitter.com/_philschmid/status/1795343334225129570), thanking @01AI_Yi for the update.
- **Gemini 1.5 series makes waves**: [@lmsysorg announced](https://x.com/lmsysorg/status/1795512202465845686?s=46) that **Gemini 1.5 Pro/Advanced and Flash results** are out, with Pro/Advanced ranked #2, closing in on GPT-4o. **Gemini 1.5 Flash** is at #9, outperforming Llama-3-70b, highlighting its cost-effectiveness, capabilities, and unmatched context length as significant advancements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1795512202465845686?s=46">Tweet from lmsys.org (@lmsysorg)</a>: Big news – Gemini 1.5 Flash, Pro and Advanced results are out!🔥  - Gemini 1.5 Pro/Advanced at #2, closing in on GPT-4o - Gemini 1.5 Flash at #9, outperforming Llama-3-70b and nearly reaching GPT-4-01...</li><li><a href="https://fxtwitter.com/_philschmid/status/1795343334225129570">Tweet from Philipp Schmid (@_philschmid)</a>: More Apache 2.0! 🚀 @01AI_Yi just updated the YI and YI-VL (multimodal LLM) models to Apache 2.0, joining the 1.5 series. 🙌   Thank you!
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1245108911778959360)** (6 messages): 

- **OpenAI Learns About ChatGPT on Twitter**: Former OpenAI board member reveals, *"It's things like when ChatGPT came out November, 2022, the board was not informed in advance about that. We learned about ChatGPT on Twitter."* ([See tweet](https://fxtwitter.com/bilawalsidhu/status/1795534345345618298)).
- **Shocking Revelations in Podcast**: Helen Toner, former OpenAI board member, exposes that **Sam Altman** was fired for dishonesty, creating a toxic work environment, and accusations of "psychological abuse." She calls for *"external regulation of AI companies,"* citing that self-governance may not always be effective ([Podcast link](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3)).
- **Natolambert's Reaction**: Natolambert reacted strongly to Helen Toner's revelations, exclaiming, *"holy shit"* and then questioning if *"helen [is] literally going to save the world?"*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/btibor91/status/1795551083420430579">Tweet from Tibor Blaho (@btibor91)</a>: @TheXeophon https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3</li><li><a href="https://fxtwitter.com/bilawalsidhu/status/1795534345345618298">Tweet from Bilawal Sidhu (@bilawalsidhu)</a>: ❗EXCLUSIVE: &#34;We learned about ChatGPT on Twitter.&#34;   What REALLY happened at OpenAI? Former board member Helen Toner breaks her silence with shocking new details about Sam Altman&#39;s firing....
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1245028765575413912)** (3 messages): 

- **Reliable AI Model Leaderboard Shared**: A user asked for a good website to find the best models and shared [a leaderboard link](https://chat.lmsys.org/?leaderboard). Simon confirmed that it is his favorite site for comparing LLMs, calling it reliable.
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1244977023458349168)** (3 messages): 

- **Local Endpoints for AI Models Must Be Secure**: A member expressed excitement about *"ubiqutizing, standardizing, and enabling locally available endpoints,"* but emphasized the need for a secure validation mechanism such as DNS SRV records and pub keys. They humorously noted the importance of verifying the trustworthiness of local AI models, as one might otherwise end up *"buy[ing] country music or... feed[ing] skwirrelz."*

- **Error with `granite-34b-code-instruct.llamafile`**: Upon attempting to run a llamafile from Hugging Face, a member encountered an *"unknown argument: --temp"* error. The process involved downloading, changing permissions, and running the file, culminating in this issue.

- **Llamafiles Store and Run One Model**: It was clarified that whatever model is running at `localhost:8080` is the one used, exemplified by *tinyllama*. The `model` field in the chat completion request was noted as irrelevant in this context.

**Link mentioned**: <a href="https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile/resolve/main/granite-34b-code-instruct.Q5_0.llamafile?download=true">no title found</a>: no description found

  

---



### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1244754490918309968)** (3 messages): 

- **R1 is quite the paperweight**: One member mentioned, *“I’m holding out hoping R1 will follow through. If not it’s a nice paperweight.”*.
- **Seeking solutions and updates**: Another member expressed curiosity saying, *“lmk if u ever find anything on this.”*.
- **Email response needed**: A member requested assistance from the OI team, stating, *“Hi OI team, I had sent an email a few days ago to which I haven’t yet received a response for. I just sent a reminder again.”*.
  

---



### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1244853808887234621)** (2 messages): 

```html
- **Server seems unmoderated**: A member pointed out that "it looks like the server is unmoderated..." highlighting an apparent lack of moderation.
- **Attempted @everyone ping fails**: The same member tried to use the @everyone tag but noted it "doesn't ping" as intended.
```
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1244956594589143084)** (1 messages): 

- **Question about course content**: A member inquired, *"How is this course going for you? Do they teach you to automate backend services using LLM?"* There was no reply to this query in the channel.
  

---



---



---




{% else %}




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**OCR Showdown: Google Vision vs. Microsoft Azure**: AI engineers debated the merits and pitfalls of **Google Vision OCR**, acknowledging its precision but criticizing the developer experience. Suggestions for using **Microsoft Azure OCR** and **Mindee Doctr**, potentially offering better ease of use, surfaced [here](https://huggingface.co/spaces/mindee/doctr).

**Curated Data: The Key to LLM Success**: Workshop discussions underscored the importance of fine-tuning LLMs with high-quality, curated datasets, ranging from pharma applications to technical support chatbots. Expert opinion highlighted the need for precision in data choice to maximize LLM effectiveness, spotlighting domains like drug discovery, law, sales, and interdisciplinary work.

**Axolotl Angst and Optimization**: Users faced hurdles running **Axolotl's 70B model** on M3 Macs, with overwhelming latency during local inference, pointing to deployment on Modal as a possible solution. Cost concerns with **Weights & Biases (WandB)** prompted considerations of alternatives like **Aim** and **MLflow** for economically-minded solo developers [Axolotl examples](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml).

**LLM Evaluation Deep Dive**: A session on evaluating LLMs offered a treasure trove of insights, covering product metrics, traditional and dynamic performance metrics, and tools like LangFuse and EvalGen. Recommending resources by Eugene Yan and practical examples to visualize fine-tuning, participants noted the necessity of nuanced evaluations for LLM development.

**Transcription Tangles and the Path to Summaries**: Communication around transcripts from large meetings illuminated needs for efficient summaries, exposing potential roles for LLMs. While Zoom transcripts are on the horizon, Hamel encouraged using LLMs to generate more digestible summaries, echoing wider community involvement.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Eagerly Awaiting imfo Alpha Release**: A link to a tweet by [@spectate_or](https://x.com/spectate_or/status/1795077451195830661?s=46) hinted at the upcoming release of **imfo alpha**, sparking excitement and comparisons to similar tools within the engineering community.

- **AI Task Structure Debate**: Engineers discussed categorizing **AI tasks** into retrieval and mutation types, exemplifying with queries like "Get the weight of the iPhone 15". The need for adjustments in tasks requiring sequential execution was highlighted with the insight that *"all the steps just happen at the same time."*

- **Scraping Accuracy Stumbles**: Members voiced challenges in **HTML parsing** for reliable data scraping, with complications arising from sites like Apple and Docker's release notes. Workarounds through **Playwright** for JavaScript-centric sites were considered alongside issues with Cloudflare.

- **Exploring Cost-Efficient AI Model Utilization**: The community delved into the cost-effectiveness of using various **AI models** such as Llama3 and Claude. An approach using a combined system suggested possibilities for greater savings.

- **API Functionality Quirks Highlighted**: Confusion arose around an **API output** that displayed a JSON object sans functional links, potentially linked to the absence of a **closed beta citations feature**. Additional discussions included prompts to improve video link generation and a brief inquiry about a potential API outage.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**New AI Features to Tinker With**: Stability AI announces the launch of **Stable Assistant** sporting editing features built on **Stable Diffusion 3**, boasting of improved text-to-image quality available for a free trial [here](https://stability.ai/stable-assistant), and a beta chatbot with **Stable LM 2 12B**, heralding future enhancements for text generation tasks.

**Education Merges with AI Innovation**: An upcoming 4-week course by **Innovation Laboratory**, a collaboration between Stability AI and HUG, intends to guide participants on training AI models utilizing Stability AI's framework in tandem with HUG's educational approach; sign-ups are open until June 25, 2024, accessible [here](https://www.studios.thehug.xyz/lab).

**GPU Sharing in the Spotlight**: AI engineers discuss a community-based GPU sharing proposal to decrease compute costs, with options ranging from a custom node to a potential blockchain setup designed to validate model training operations.

**SD3 Accessibility Stirs Controversy**: Discordance surfaces as members air grievances regarding **Stable Diffusion's SD3** weights not being available for local use — slating Stability AI's cloud-only approach and stirring debate over cloud-dependency and data privacy concerns.

**User Interfaces Under Comparison**: A technical discourse unfolds on the pros and cons of various interfaces for Stable Diffusion, with **ComfyUI** pitted against more user-friendly alternatives like Forge; discussions also include community tips, inpainting methods, and ways to enhance artificial intelligence workflows.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**OpenAI Forms Safety Shield**: OpenAI has established a **Safety and Security Committee** that will take charge of critical safety and security decisions across all its projects; full details can be found in their [official announcement](https://openai.com/index/openai-board-forms-safety-and-security-committee/).

**AI Muscle Flexes in Hardware Arena**: Discussions about hardware costs arose, speculating on a $200-$1000 increase due to **NPUs** (Neural Processing Units), with focus on their economic impact for high-end models.

**Plotting the Prompt Landscape**: AI engineers debated the merits of **meta-prompting** versus **Chain of Thought (CoT)**, examining the potential of using mermaid diagrams to conserve tokens and enhance output quality. There was also a sharing of improved prompts like [here](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477), showcasing practical applications of advanced prompt engineering tactics.

**Rubber Meets The Code**: Practical discussions included how AI handles **YAML, XML, and JSON** formats natively, with suggestions on using these structures for prompts to improve AI understanding and performance, and shared resources pointing to real-life prompt application for generating code and planning.

**Interactive Inconsistencies Ignite Inquiry**: Users reported issues with **ChatGPT** ranging from its refusal to draw tarot cards to context drops and unresponsiveness, spotlighting the need for improved and more predictable AI behavior.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Voice Commands Meet Robotics**: A demo video titled ["Open Source Voice-Controlled Robotic Arm"](https://www.youtube.com/watch?v=qv3bFhHoA5s) exhibits a voice-activated AI robotic arm. The perspective of democratizing robotics technology via community collaboration was forwarded.

**Bridging Modalities**: Contributions on creating early multi-modal spaces point to the use of single models and possibly stacked models with routing functionalities. For insights on such implementation, a [source link](https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py) was shared, providing a model example with practical applications.

**Deep Learning Consult on the Fly**: A user consulted the community about overcoming common pain points in training a model using Stanford Cars Dataset, managing only a 60% accuracy using ViT-B_16, with struggles involving overfitting. Meanwhile, another member is looking for help on how to better their deep learning model, indicating an environment that supports knowledge exchange for newcomers.

**Diffusers Update for Not-Just-Generation**: Hugging Face announced its **Diffusers library now supports tasks beyond generative models**, such as depth estimation and normals' prediction through **Marigold**. The update suggests an escalating trend in the versatility of diffusion models and their applications.

**Model Choices for Cyber Security Assessments**: Analysis from researchers examines the aptitude of various large language models in cyber security contexts. This provides AI engineers an angle to consider the security ramifications inherent in the deployment of LLMs.

**Robust SDXL Space Realignment**: SDXL embed space discussions underscore that newly aligned spaces default to zeroes instead of an encoded space. Such insights reflect the underlying complexity and time demands associated with realigning models to new unconditioned spaces, revealing the intricate process behind the science.

**Gradio Piques Curiosity with Upgraded Clients**: The Gradio team announced a forthcoming live event to dive into the latest features of Gradio Python and JavaScript clients. The engagement invitation emphasizes Gradio's continuous push to streamline AI integration into diverse applications through enhanced interfaces.
  
**Ambiguity in Finding an SFW Dataset**: Community chatter touches on the difficulty of locating the Nomos8k_sfw dataset, which is tied to the 4x-Nomos8kDAT model, suggesting the dataset’s limited availability or obscure placement. This highlights the occasional challenges inherent to dataset procurement.

**Launching Latest Tools for AI Storytelling**: Typeface Arc emerges as a comprehensive platform for seamlessness in creating AI-driven content. It features a tool, appropriately dubbed "Copilot", designed to amplify content creation via an interactive experience pivotal for brand narratives.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Visualize This: OpenAI Integrates with LLama!**: Engineers can now leverage **LLaVA** for visual capabilities in LM Studio by deploying it on a server and making use of the Python vision template provided.

**Speedy Model Loading on M1 Max**: AI models like **MLX and EXL2 load swiftly** on Apple's M1 Max, taking a mere 5 seconds for L3 8bit, indicating superior performance compared to GGUF Q8 which takes 29 seconds.

**LM Studio Finetuning Frustrations**: Despite being a robust environment, **LM Studio currently lacks the ability to directly fine-tune models**, with enthusiasts being pointed to alternative solutions like MLX designed for Apple Silicon.

**Budget or Bust**: AI practitioners debated the value proposition of various Nvidia GPUs, considering alternatives like the **Tesla P40/P100** and eagerly discussed rumored GPUs like the **5090** with anticipation.

**Beta Testing Blues**: As they navigate the waters of new releases, users reported problems such as **Windows CPU affinity issues** with large models and **errors on AVX2 laptops**, hinting at the complexities of configuring modern hardware for AI tasks.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-2 Gets No Love from Unsloth**: Unsloth confirmed that **GPT-2** cannot be fine-tuned using its platform due to fundamental architectural differences.

- **Fine-Tuning Frustrations with Fiery Chat**:
  - When fine-tuning llama 3 with 50,000+ email entries, members shared advice on structuring prompts for optimal input-output pairing.
  - Faced with a repeating sentence issue post-training, adding an End-Of-Sentence (EOS) token was recommended to prevent the model's overfitting or poor learning.

- **Vision Model Integration on the Horizon**: Members are keenly awaiting **Unsloth's** next-month update for vision models support, citing referrals to [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [Segment Anything](https://github.com/facebookresearch/segment-anything) for current solutions.

- **LoRA Adapters Learning to Play Nice**: The community shared tips on merging and fine-tuning LoRA adapters, emphasizing the use of resources like [Unsloth documentation on GitHub](https://github.com/unslothai/unsloth#-finetune-for-free) and exporting models to HuggingFace.

- **Coping with Phi 3 Medium's Attention Span**: Discussions on **Phi3-Medium** revealed its sliding window attention causes efficiency to drop at higher token counts, with many eager for enhancements to handle larger context windows.

- **ONNX Export Explained**: Guidance was provided for converting a fine-tuned model to **ONNX**, as seen in Hugging Face's [serialization documentation](https://huggingface.co/docs/transformers/en/serialization), with confirmation that VLLM formats are compatible for conversion.

- **Looks Like We're Going Bit-Low**: Anticipation is building for **Unsloth's** upcoming support for 8-bit models and integration capabilities with environments like Ollama, analogous to OpenAI's offerings.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Toolkit Commands for Ubuntu on Fire**: A user suggested installing the **CUDA Toolkit** from NVIDIA, checking installation with `nvidia-smi`, and offered commands for setup on Ubuntu, including via Conda: `conda install cuda -c nvidia/label/cuda-12.1.0`. Meanwhile, potential conflicts were identified with Python 3.12 and missing **triton** installation when setting up PyTorch 2.3, linked to a [GitHub issue](https://github.com/pytorch/pytorch/issues/120233).

- **GPT-4o meets its match in large edits**: Members noted that GPT-4o struggles with extensive code edits, and a new **fast apply** model aims to split the task into planning and application stages to overcome this challenge. Seeking a deterministic algorithm for code edits, a member posed the feasibility of using **vllm** or **trtllm** for future token prediction without relying on draft models. More information on this approach can be found in the [full blog post](https://cursor.sh/blog/instant-apply).

- **SYCL Debug Troubles**: A member enquired about tools to debug SYCL code, sparking a discussion on stepping into kernel code for troubleshooting.

- **Torchao's Latest Triumph**: The torchao community celebrated the merging of support for MX formats, such as `fp8/6/4`, in PyTorch, offering efficiency for interested parties, provided in part by a [GitHub commit](https://github.com/pytorch/ao/pull/264) and aligned with the [MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

- **Understanding Mixer Models in DIY**: Members dissected implementation nuances, such as integrating `dirent.h` in **llm.c**, and the importance of guarding it with `#ifndef _WIN32` for OS compatibility. The addition of a `-y 1` flag for resuming training in interruptions was implemented, addressing warnings about uninitialized variables and exploring memory optimization strategies during backward pass computation, with a related initiative found in [GitHub discussions](https://github.com/karpathy/llm.c/discussions/481).

- **Quantizing Activations in BitNet**: In the BitNet channel, it was concluded that passing incoming gradients directly in activation quantized neural networks might be erroneous. Instead, using the gradient of a surrogate function such as `tanh` was suggested, citing an [arXiv paper](https://arxiv.org/abs/1903.05662) on straight-through estimator (STE) performance.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **No Post-Learning for GPT Agents**: GPT-based agents do not learn post initial training, but can reference new information uploaded as 'knowledge files' without fundamentally altering their core understanding.
- **Efficiency Milestones in Diffusion Models**: Google DeepMind introduces **[EM Distillation](http://arxiv.org/abs/2405.16852)** to create efficient one-step generator diffusion models, and separate research from Google illustrates an 8B parameter diffusion model adept at generating high-res 1024x1024 images.
- **Scaling Down for Impact**: **[Super Tiny Language Models](https://arxiv.org/abs/2405.14159)** research focuses on reducing language model parameters by 90-95% without significantly sacrificing performance, indicating a path towards more efficient natural language processing.
- **GPU Performance Without the Guesswork**: Symbolic modeling of GPU latencies **without execution** gains traction, featuring [scholarly resources](https://inria.hal.science/hal-00789958/file/112_Lai.pdf) to guide theoretical understanding and potential impact on computational efficiency.
- **Challenging the Current with Community**: Discussions highlight community-driven projects and the importance of **collaborative problem-solving** in areas such as prompt adaptation research and implementation queries, like that of a **Facenet model** in PyTorch.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Latest Model Innovations Hit the Market**: [OpenRouter](https://openrouter.ai/models) announced new AI models, including **[Mistral 7B Instruct v0.3](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3)** and **[Hermes 2 Pro - Llama-3 8B](https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b)**, while assuring that previous versions like **[Mistral 7B Instruct v0.2](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2)** remain accessible.

- **Model Curiosity on Max Loh's Site**: Users show curiosity about the models utilized on [Max Loh's website](https://www.maxloh.com), expressing interest in identifying all uncensored models available on OpenRouter.

- **OCR Talent Show**: **Gemini's OCR** prowess was a hot topic, with users claiming its superior ability to read Cyrillic and English texts, outdoing competing models such as Claude and GPT-4o.

- **OpenRouter Token Economics**: There was clarification in the community that $0.26 allows for 1M input + output tokens on OpenRouter, and discussions emphasized how token usage is recalculated with each chat interaction, potentially inflating costs.

- **The Cost of Cutting-Edge Vision**: There is a heated exchange on **Phi-3 Vision** costs when using Azure, with some members finding the $0.07/M for llama pricing too steep, even though similar rates are noted among other service providers.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Translation Tribulations**: Discussions touched on the challenges of *translating songs* with control over lyrical tone to retain the original artistic intent. The unique difficulty lies in balancing the fidelity of meaning with musicality and artistic expression.

- **AI Infiltrates Greentext**: Members experimented with LLMs to generate **4chan greentexts**, sharing their fascination with the AI's narrative capabilities — especially when concocting a scenario where one wakes up to a world where AGI has been created.

- **Philosophical Phi and Logically Challenged LLMs**: Debates emerged over **Phi model's training data** composition, with references to "heavily filtered public data and synthetic data". Additionally, evidence of LLMs struggling with logic and self-correction during interaction was reported, raising concerns about the models' reasoning abilities.

- **Shaping Data for Machine Digestion**: AI enthusiasts exchanged resources and insights on **creating DPO datasets** and adjusting dataset formats for DPO training. Hugging Face's [TRL documentation](https://huggingface.co/docs/trl/main/en/reward_trainer) and [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) emerged as key references, alongside a [paper](https://arxiv.org/abs/2305.18290) detailing language models trained from preference data.

- **Linking Minds for RAG Riches**: Collaboration is in the air, with members sharing their intent to combine efforts on RAG-related projects. This includes the sentiment and semantic density smoothing agent project with TTS on [GitHub](https://github.com/EveryOneIsGross/densefeelsCHAT), and intentions to port an existing project to SLURM for enhanced computational management.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Loop-the-Loop in LangChain**: Engineers are troubleshooting a **LangChain agent** entering continuous loops when calling tools; one solution debate involves refining the agent's trigger conditions to prevent infinite tool invocation loops.

**Details, Please! 16385-token Error in LangChain 0.2.2**: Users report a token limit error in **LangChain version 0.2.2**, where a 16385-token limit is incorrectly applied, despite models supporting up to 128k tokens, prompting a community-lead investigation into this discrepancy.

**SQL Prompt Crafting Consultation**: Requests for **SQL agent** prompt templates with few-shot examples have been answered, providing engineers with the resources to craft queries in LangChain more effectively.

**Disappearing Act: Custom kwargs in Langserve**: Some users experience a problem where custom "kwargs" sent through **Langserve** for logging in **Langsmith** are missing upon arrival, a concern currently seeking resolution.

**Showcasing Applications**: Diverse applications developed using LangChain were shared, including frameworks for **drug discovery**, cost-saving measures for logging, enhancements for **flight simulators**, and tutorials about **routing logic** in agent flows.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Python Version Alert for Mojo Users**: Mojo users are reminded to adhere to the supported Python versions, ranging from **3.8 to 3.11**, since **3.12 remains unsupported**. Issues in Mojo were resolved by utilizing the deadsnakes repository for Python updates.

- **AI-Powered Gaming Innovations**: Engineers discussed the prospect of subscription models based on NPC intelligence in open-world games, and introducing special AI-enabled capabilities for smart devices that could lead to AI inference running locally. They explored open-world games that could feature AI-driven custom world generation.

- **Mojo Mastery**: Circular dependencies are permitted within Mojo, as modules can define each other. Traits like `Intable` and `Stringable` are inherently available, and while lambda functions are not yet a feature in Mojo, callbacks are currently utilized as an alternative.

- **Performance Pioneers**: An impressive *50x speed improvement was noted at 32 bytes in Mojo*, though it encountered cache limitations beyond that length. Benchmarks for k-means algorithms demonstrated variability due to differences in memory allocation and matrix computations, with a suggestion to optimize memory alignment for AVX512 operations.

- **Nightly Builds Nightcaps**: The latest **Mojo compiler build (2024.5.2805)** brought new features, including implementations of `tempfile.{mkdtemp,gettempdir}` and `String.isspace()`, with full changes detailed in the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and the [raw diff](https://github.com/modularml/mojo/compare/ce285fded710b403e1b7b5637183ea20fa4d5c97...4724ec6ff46378f6a1d6190ca9a76916a5faaba3). Structural sharing via references was also highlighted for its potential efficiency gains in Mojo programming.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Debugging Just Got a Level Up**: Engineers praised the **cursor interpreter mode**, highlighted for its advanced code navigation capabilities over traditional search functions in debugging scenarios.

- **A Co-Pilot for Your Messages**: **Microsoft Copilot**'s integration into **Telegram** sparked interest for its ability to enrich chat experiences with features such as gaming tips and movie recommendations.

- **GPT-2 Training on a Shoestring**: **Andrej Karpathy** showcased an economical approach to training GPT-2 in **90 minutes for $20**, detailing the process on [GitHub](https://github.com/karpathy/llm.c/discussions/481).

- **Agents and Copilots Distinguish Their Roles**: A distinction between **Copilots** and **Agents** was debated following **Microsoft Build's** categorization, with references made to [Kanjun Qiu's insights](https://www.latent.space/p/imbue) on the topic.

- **AI Podcast Delivers Cutting-Edge Findings**: An [ICLR 2024-focused podcast](https://x.com/latentspacepod/status/1795196817044594817) was released discussing breakthroughs in ImageGen, Transformers, Vision Learning, and more, with anticipation for the upcoming insights on LLM Reasoning and Agents.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Financial Geeks, Feast on FinTextQA**: [FinTextQA](https://t.co/emhQYXY1S4) is a new dataset aimed at improving long-form finance-related question-answering systems; it comprises *1,262 source-attributed Q&A pairs* across *6 different question types*.

- **Perfecting Prompt Structures**: An enquiry was made concerning resources for crafting optimal system role prompts, drawing inspiration from **LlamaIndex**'s model.

- **Chat History Preservation Tactics**: The community discussed techniques for saving chat histories within **LlamaIndex**, considering custom retrievers for **NLSQL** and **PandasQuery** engines to maintain a record of queries and results.

- **API Function Management Explored**: Strategies to handle an extensive API with over 1000 functions were proposed, favoring hierarchical routing and the division of functions into more manageable subgroups.

- **RAG System Intricacies with LlamaIndex Debated**: Technical challenges related to metadata in RAG systems were dissected, showing a divided opinion on whether to embed smaller or larger semantic chunks for optimal accuracy in information retrieval.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI Reads Between the Lines**: Members shared a laugh over SOTA AGI models' odd claims with one model's self-training assertion, "it has trained a model for us," tickling the collective funny bone. Musk's jab at [CNNs](https://x.com/elonmusk/status/1795405972145418548)—quipping "We don’t use CNNs much these days"—set off a chain of ironical replies and a nod towards vision transformer models as the new industry darlings.

**Artificial Artist's Watermark Woes**: [Corcelio's Mobius Art Model](https://huggingface.co/Corcelio/mobius) is pushing boundaries with diverse prompts, yet leaves a watermark even though it's overtaking past models in creativity. Ethical dilemmas arose from the capability of image generation systems to produce 'inappropriate' content, sparking debate on community guidelines and systems' control settings.

**Synthetic Sight Seeks Improvement**: In an effort to grapple with **SDXL**'s inability to generate images of "reading eyes," a member asked for collaborative help to build a synthetic database using DALLE, hoping to hone **SDXL**'s capabilities in this nuanced visual task.

**Patterns and Puzzles in Generative Watermarks**: Observations within the guild pointed out a recurring theme of generative models producing watermarks, indicating possible undertraining, which was found both amusing and noteworthy among the engineers.

**Elon's Eyeroll at CNNs Stokes AI Banter**: Elon Musk's tweet sent a ripple through the community, sparking jests about the obsolete nature of CNNs in today's transformative AI methodologies and the potential pivot towards transformer models.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**GPU Latency Predictions Without Benchmarks?**: Engineers discussed the potential for **symbolically modeling GPU latencies** without running kernels by considering data movement and operation times, though complexities such as occupancy and async operations were recognized as potential confounders. There's also anticipation for AMD's open-source release of MES and speculation about quant firms using cycle accurate GPU simulators for in-depth kernel optimization.

**Optimizing with Autotuners**: The community explored kernel optimization tools like **AutoTVM** and **Halide**, noting their different approaches to performance improvement; George Hotz highlighted TVM's use of XGBoost and stressed the importance of cache emulation for accurate modeling.

**Latency Hiding Mechanics in GPUs**: It was noted that GPUs employ a variety of latency-hiding strategies with their ability to run concurrent wavefronts/blocks, thus making latency modeling more complex and nuanced.

**Buffer Creation Discussions in Tinygrad**: The #learn-tinygrad channel had members inquiring about using **post dominator analysis** in scheduling for graph fusion efficiency and the creation of **LazyBuffer** from arrays, with a suggestion to use `Load.EMPTY -> Load.COPY` for such scenarios.

**Code Clarity and Assistance**: Detailed discussions were had regarding buffer allocation and `LazyBuffer` creation in Tinygrad, with one member offering to provide **code pointers** for further clarification and understanding.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Elevenlabs Voices Come to AI Town**: Integrating **Elevenlabs**' text-to-speech capabilities, AI Town introduced a feature allowing conversations to be heard, not just read, with a minor delay of about one second, challenging real-time usage. The implementation process involves [transforming text into audio](https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19) and managing audio playback on the frontend.

- **Bring Science Debate to AI Chat**: A concept was shared about utilizing AI chatbots to simulate science debates, aiming to foster engagement and demonstrate the unifying nature of scientific discussion.

- **Audio Eavesdropping Added for Immersion**: The Zaranova fork of AI Town now simulates eavesdropping by generating audio for ambient conversations, potentially amplifying the platform's interactivity.

- **Collaborative Development Rally**: There's an active interest from the community in contributing to and potentially merging new features, such as text-to-speech, into the main AI Town project.

- **Addressing User Experience Issues**: A user experienced difficulties with the conversations closing too quickly for comfortable reading, hinting at potential user interface and accessibility improvements needed within AI Town.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Slimming Down on Logs**: A new pipeline developed by a member removes **redundant logs** to reduce costs. They recommended a [tool](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere) for selecting a "verbose logs" pipeline to achieve this.

- **Debating Deployment**: Members discussed cloud-prem deployment solutions for **reranking and query extraction**, seeking insights on the best integrated practices without providing further context.

- **Financial RAG Fine-tuning**: There was an inquiry on the possibility of **fine-tuning Cohere models** to answer financial questions, specifically mentioning the integration with **RAG (Retrieve and Generate) systems** using SEC Filings.

- **Aya23 Model's Restrictive Use**: It was clarified that **Aya23 models** are strictly for research purposes and are not available for commercial use, affecting their deployment in startup environments.

- **Bot Plays the Game**: A member launched a **Cohere Command R** powered gaming bot, **Create 'n' Play**, featuring *"over 100 text-based games"* aimed at fostering social engagement on Discord. The project's development and purpose can be found in a [LinkedIn post](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Inference vs. Training Realities**: The conversation underscored performance figures in AI training, particularly regarding how a seemingly simple query about "inference only" topics quickly lead to complex areas focused on training's computational requirements.

- **FLOPS Define Training Speed**: A key point in the discussion was that AI model training is, in practice, constrained by floating-point operations per second (FLOPS), especially when employing techniques like teacher forcing which increase the effective batch size.

- **Eager Eyes on Hopper Cards for FP8**: The community showed enthusiasm about the potential of **Hopper** cards for fp8 native training, highlighting a keen interest in leveraging cutting-edge hardware for enhanced training throughput.

- **Eradicating Version Confusion with fschat**: Members were advised to fix **fschat** issues by reinstallation due to erroneous version identifiers, pointing to meticulous attention to detail within the collective's ecosystem.

- **When CUTLASS Is a Cut Above**: Discussions clarified the importance of setting `CUTLASS_PATH`, emphasizing CUTLASS's role in optimizing matrix operations vital for deep learning, underscoring the guild’s focus on optimizing algorithmic efficiency.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apache Welcomes YI and YI-VL Models**: The **YI and YI-VL (multimodal LLM) models** are now under the **Apache 2.0** license, as celebrated in a [tweet by @_philschmid](https://fxtwitter.com/_philschmid/status/1795343334225129570); they join the 1.5 series in this licensing update.

- **Gemini 1.5 Challenges the Throne**: **Gemini 1.5 Pro/Advanced** has climbed to #2 on the ranking charts, with ambitions to overtake GPT-4o, while **Gemini 1.5 Flash** proudly takes the #9 spot, edging out **Llama-3-70b**, as announced in a [tweet from lmsysorg](https://x.com/lmsysorg/status/1795512202465845686?s=46).

- **OpenAI's Board Left in the Dark**: A former OpenAI board member disclosed that the board wasn't informed about the release of **ChatGPT** in advance, learning about it through [Twitter](https://fxtwitter.com/bilawalsidhu/status/1795534345345618298) just like the public.

- **Toner Drops Bombshell on OpenAI's Leadership**: Helen Toner, a previous member of OpenAI's board, accused **Sam Altman** of creating a toxic work environment and acting dishonestly, pushing for "external regulation of AI companies" during a [TED podcast episode](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3).

- **Community Aghast at OpenAI's Revelations**: In reaction to Helen Toner's grave allegations, the community expressed shock and anticipation about the prospect of significant industry changes, highlighted by Natolambert querying if Toner might "literally save the world?"



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Go-To LLM Leaderboard Approved by Experts**: The [leaderboard at chat.lmsys.org](https://chat.lmsys.org/?leaderboard) was highlighted and endorsed by users as a reliable resource for comparing the performance of various large language models (LLMs).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Securing Local AI Endpoints Is Crucial**: One member **highlighted the importance of securing local endpoints** for AI models, suggesting the use of **DNS SRV records** and public keys to ensure validated and trustworthy local AI interactions, jesting about the perils of unverified models leading to unintended country music purchases or squirrel feeding.
- **Troubleshoot Alert: Llamafile Error Uncovered**: A user running a **Hugging Face llamafile** - specifically `granite-34b-code-instruct.llamafile` - reported an error with an "unknown argument: --temp," indicating potential issues within the implementation phase of the model deployment process.
- **Focus on the Running Model**: In a clarification, it was noted that whatever model is running locally at `localhost:8080` (like *tinyllama*) would be the default, with the `model` field in the chat completion request being inconsequential to the operation. This suggests a **single-model operation paradigm** for **llamafiles** in use.
  
**Link mentioned**: [granite-34b-code-instruct.llamafile](https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile/resolve/main/granite-34b-code-instruct.Q5_0.llamafile?download=true)



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Request for R1 Update**: A member expressed anticipation for the **R1's** future developments, humorously referring to it as a potential "nice paperweight" if it doesn't meet expectations.
- **Community Seeks Clarity**: There's a sense of shared curiosity within the community regarding updates related to **R1**, with members actively seeking and sharing information.
- **Awaiting Support Team's Attention**: An inquiry to the **OI team** concerning an email awaits a response, signifying the need for improved communication or support mechanisms.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Spotting a Ghost Town**: A member raised the concern that the server appears **unmoderated**, which could indicate either an oversight or an intentional laissez-faire approach by the admins.
- **Notification Fails to Notify**: An attempted use of the **@everyone** tag in the server failed to function, suggesting restricted permissions or a technical snafu.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LLM for Backend Automation Inquiry Left Hanging**: A member's curiosity about whether a course covers automating backend services using Large Language Models (LLM) remained unanswered. The inquiry sought insights into practical applications of LLMs in automating backend processes.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
