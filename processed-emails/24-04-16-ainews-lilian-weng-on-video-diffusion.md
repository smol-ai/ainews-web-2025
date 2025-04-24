---
id: f351a999-84b5-4ade-9120-63cf65530493
title: Lilian Weng on Video Diffusion
date: '2024-04-17T02:15:37.824011Z'
original_slug: ainews-lilian-weng-on-video-diffusion
description: >-
  AI news covers diffusion models for video generation, Sora AI launch, OpenAI's
  expansion to Japan, new multimodal models like Reka Core, and advancements in
  AI capabilities including training-free adaptation and AI recognizing its
  outputs. Industry trends include OpenAI's market dominance, AI's creative and
  intuitive abilities, and ethical concerns about AI disruption and toxicity.
companies:
  - openai
  - adobe
  - reka-ai
  - google
models:
  - wizardlm-2
  - llama-3
  - reka-core
  - gpt-4
  - gpt-3
  - claude-3
topics:
  - diffusion-models
  - video-generation
  - multimodal
  - training-free-adaptation
  - ai-creativity
  - ai-competitions
  - ai-ethics
  - agile-ai
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/15/2024-4/16/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **27** Discords (**395** channels, and **5610** messages) for you. Estimated reading time saved (at 200wpm): **615 minutes**.

One thing we missed covering in the weekend rush is Lilian Weng's blog on [Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/). While her work is rarely breaking news on any particular day, it is almost always the single most worthwhile resource on a given important AI topic, and we would say this even if she did *not* happen to work at OpenAI.

Anyone keen on Sora, the biggest AI launch of the year so far (now rumored to be [coming to Adobe Premiere Pro](https://twitter.com/legit_rumors/status/1779951008539345140)), should read this. Unfortunately for most of us, the average diffusion paper requires 150+ IQ to read.

 ![image.png](https://assets.buttondown.email/images/bfc4ad22-23f2-4c2d-8fe5-a8abb87411f3.png?w=960&fit=max)

 ![image.png](https://assets.buttondown.email/images/1ee16364-c6ae-4168-a7ab-955de5218bc9.png?w=960&fit=max)  


We are only half joking. As per Lilian's style, she takes us on a wild tour of all the SOTA videogen techniques of the past 2 years, humbling every other AI summarizooor on earth:

 ![image.png](https://assets.buttondown.email/images/741bfc10-624e-4a05-b3ab-cb1b45d083d7.png?w=960&fit=max) 

The surprise find of the day comes from her highlight of **Training-free adaptation**, which is exactly as wild as it sounds: 

> "Somehow surprisingly, it is possible to adapt a pre-trained text-to-image model to output videos without any training 🤯."

 ![image.png](https://assets.buttondown.email/images/ca8e24fc-ffe2-4a93-b6e1-cac34b0ef23f.png?w=960&fit=max) 


She unfortunately only spends 2 sentences discussing Sora, and she definitely knows more she can't say. Anyway, this is likely the most authoritative explanation to How SOTA AI Video Actually Works you or I are ever likely to get unless Bill Peebles takes to paper writing again.

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/Singularity. Comment crawling works now but has lots to improve!


**AI Companies and Releases**

- **OpenAI expands**: [OpenAI launches in Japan](https://openai.com/blog/introducing-openai-japan), introduces [Batch API](https://i.redd.it/22uslfxivouc1.png), and partners with Adobe to bring [Sora video model to Premiere Pro](https://www.youtube.com/watch?v=6de4akFiNYM).
- **New models**: Reka AI releases [Reka Core multimodal language model](https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model). 
- **Competitive landscape**: Sam Altman says OpenAI will ["steamroll" startups](https://twitter.com/ai_for_success/status/1779930498623742187). Devin AI model sees [record internal usage](https://twitter.com/SilasAlberti/status/1778623317651706237).


**New Model Releases and Advancements in AI Capabilities**

- **WizardLM-2 released**: In /r/LocalLLaMA, WizardLM-2 was just released and is showing [**impressive performance**](https://www.reddit.com/r/LocalLLaMA/comments/1c4qi12/wizardlm2_just_released_impressive_performance/). 
- **Llama 3 news coming soon**: An [image post](https://i.redd.it/dgt5sbgqfouc1.jpeg) hints that news about Llama 3 will be coming soon.
- **Reka Core multimodal model released**: [Reka AI announced the release of Reka Core](https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model), their new frontier-class multimodal language model.
- **AI models showing intuition and creativity**: Geoffrey Hinton says current AI models are exhibiting [**intuition, creativity and can see analogies humans cannot**](https://x.com/tsarnick/status/1778524418593218837).
- **AI contributing to its own development**: [Devin was the biggest contributor to its own repository](https://twitter.com/SilasAlberti/status/1778623317651706237) for the first time, an AI system contributing significantly to its own codebase.
- **AI recognizing its own outputs**: In /r/singularity, it was shared that [**Opus can recognize its own generated outputs**](https://www.reddit.com/r/singularity/comments/1c4tfnc/opus_can_recognize_its_own_outputs/), an impressive new capability.

**Industry Trends, Predictions and Ethical Concerns**

- **Warnings about AI disruption**: Sam Altman [warned startups about the risk of getting steamrolled by OpenAI](https://twitter.com/ai_for_success/status/1779930498623742187) if they don't adapt quickly enough.
- **Debate on AGI timelines**: While Yann LeCun believes AGI is inevitable, he [says it's not coming next year or only from LLMs](https://x.com/ylecun/status/1779845304788955292?s=46&t=1y5Lfd5tlvuELqnKdztWKQ). 
- **Toxicity issues with models**: [WizardLM-2 had to be deleted shortly after release](https://i.redd.it/lyaop5lw0suc1.png) because the developers forgot to test it for toxicity, highlighting the challenges with responsible AI development.
- **Proposed AI regulation in the US**: The Center for AI Policy put forth a [new proposal for a bill to regulate AI development in the US](https://twitter.com/neil_chilson/status/1777695468656505153).
- **Warning about AI startups**: A [PSA in /r/singularity warned about being cautious with startups](https://www.reddit.com/r/singularity/comments/1c566i0/psa_beware_of_startups_that_looks_too_good_to_be/) that seem too good to be true, as some have questionable pasts tied to crypto.

**Technical Discussions and Humor**

- **Building Mixture-of-Experts models**: /r/LocalLLaMA shared a guide on how to [easily build your own MoE language model using mergoo](https://www.reddit.com/r/LocalLLaMA/comments/1c4gxrk/easily_build_your_own_moe_llm/).
- **Diffusion vs autoregressive models**: /r/MachineLearning had a discussion comparing [diffusion and autoregressive approaches for image generation](https://www.reddit.com/r/MachineLearning/comments/1c53pc5/diffusion_versus_autoregressive_models_for_image/) and debating which is better.
- **Fine-tuning GPT-3.5**: /r/OpenAI posted a [guide for fine-tuning GPT-3.5 for custom use cases](https://www.reddit.com/r/OpenAI/comments/1c4j6n7/finetuning_gpt35_for_custom_use_cases/).
- **AI advancement memes**: The community shared some humorous memes, including a ["can't wait" meme about the pace of AI progress](https://v.redd.it/kbrdbah8fnuc1), a [meme about reversing aging in mice](https://lifeboat.com/blog/2024/01/reversing-wrinkled-skin-and-hair-loss-in-mice-by-restoring-mitochondrial-function), and a [cursed rave video meme](https://v.redd.it/eec3ezwklpuc1).

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**WizardLM-2 Release and Withdrawal**

- **WizardLM-2 Release**: [@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899325868589372) announced the release of WizardLM-2, their next-generation state-of-the-art LLM family, including **8x22B, 70B, and 7B models** which demonstrate highly competitive performance compared to leading proprietary LLMs.
- **Toxicity Testing Missed**: [@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1780101465950105775) apologized for accidentally missing the required toxicity testing in their release process, and stated they will **complete the test quickly and re-release the model** as soon as possible. 
- **Model Weights Pulled**: [@abacaj](https://twitter.com/abacaj/status/1780090189563486691) noted that **WizardLM-2 model weights were pulled from Hugging Face**, speculating it may have been a premature release or something else going on.

**Reka Core Release**

- **Reka Core Announcement**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894622334189592) announced the release of Reka Core, their most capable multimodal language model yet, which has **a lot of capabilities including understanding video**.
- **Technical Report**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894626083864873) published a **technical report detailing the training, architecture, data, and evaluation** for the Reka models.
- **Benchmark Performance**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894623848304777) evaluated Core on standard benchmarks for both text and multimodal, along with a blind third-party human evaluation, showing it **approaches frontier-class models like Claude3 Opus and GPT4-V**.

**Open Source Model Developments**

- **Pile-T5**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1779891910871490856) announced the release of Pile-T5, a T5 model **trained on 2T tokens from the Pile using the Llama tokenizer**, featuring intermediate checkpoints and a significant boost in benchmark performance.
- **Idefics2**: [@huggingface](https://twitter.com/huggingface/status/1779922877589889400) released Idefics2, an **8B vision-language model with significantly enhanced capabilities** in OCR, document understanding, and visual reasoning, available under the Apache 2.0 license.
- **Snowflake Embedding Models**: [@SnowflakeDB](https://twitter.com/SnowflakeDB/status/1780225794402627946) open-sourced snowflake-arctic-embed, a family of powerful embedding models **ranging from 22 to 335 million parameters with 384-1024 embedding dimensions and 50-56 MTEB scores**.

**LLM Architecture Developments**

- **Megalodon Architecture**: [@_akhaliq](https://twitter.com/_akhaliq/status/1780083267888107546) shared Meta's announcement of Megalodon, an **efficient LLM pretraining and inference architecture with unlimited context length**.
- **TransformerFAM**: [@_akhaliq](https://twitter.com/_akhaliq/status/1780081593643647022) shared Google's announcement of TransformerFAM, where **feedback attention is used as working memory to enable Transformers to process infinitely long inputs**.

**Miscellaneous Discussions**

- **Humanoid Robots Prediction**: [@DrJimFan](https://twitter.com/DrJimFan/status/1780254247650787512) predicted that **humanoid robots will exceed the supply of iPhones in the next decade**, gradually then suddenly.
- **Captchas and Bots**: [@fchollet](https://twitter.com/fchollet/status/1780042591440134616) argued that **captchas cannot prevent bots from signing up for services**, as professional spam operations employ people to solve captchas manually for about 1 cent per account.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. New Language Model Releases and Benchmarks**

- **[EleutherAI](https://blog.eleuther.ai/pile-t5/)** released **[Pile-T5](https://github.com/EleutherAI/improved-t5)**, an enhanced T5 model trained on the Pile dataset with up to 2 trillion tokens, showing improved performance across benchmarks. The release was also [announced on Twitter](https://x.com/arankomatsuzaki/status/1779891910871490856).

- **[Microsoft](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b)** released **[WizardLM-2](https://wizardlm.github.io/WizardLM2/)**, a state-of-the-art instruction-following model that was later [removed due to a missed toxicity test](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png?ex=66309ca9&is=661e27a9&hm=f105e6497796be9c414ade2024a27f9561caf0cad6cb06ba09f80e30b5e39ae4&), but mirrors remain on sites like [Hugging Face](https://huggingface.co/alpindale/WizardLM-2-8x22B).

- **[Reka AI](https://publications.reka.ai/reka-core-tech-report.pdf)** introduced **[Reka Core](https://www.youtube.com/watch?v=vL1SayPCHBg)**, a frontier-class multimodal language model competitive with OpenAI, Anthropic, and Google models.

- **[Hugging Face](https://huggingface.co/blog/idefics2)** released **[Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b)**, an 8B multimodal model excelling in vision-language tasks like OCR, document understanding, and visual reasoning.

- Discussions around model performance, sampling techniques like **[MinP/DynaTemp/Quadratic](https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/)**, and the impact of tokenization per a **[Berkeley paper](https://arxiv.org/abs/2404.08335)**.

**2. Open Source AI Tools and Community Contributions**

- **[LangChain](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction)** introduced a [revamped documentation structure](https://discord.com/channels/1038097195422978059/1058033358799655042/1229820483818623010) and saw community contributions like [Perplexica](https://github.com/ItzCrazyKns/Perplexica/) (an open-source AI search engine), [OppyDev](https://oppydev.ai) (an AI coding assistant), and [Payman AI](https://www.paymanai.com/) (enabling AI agents to hire humans).

- **[LlamaIndex](https://twitter.com/llama_index/status/1779898403239125198)** announced tutorials on agent interfaces, a [hybrid cloud service with Qdrant Engine](https://twitter.com/llama_index/status/1780275878230139293), and an [Azure AI integration guide](https://twitter.com/llama_index/status/1780324017083400235) for hybrid search.

- **[Unsloth AI](https://github.com/unslothai/unsloth/wiki)** saw discussions on LoRA fine-tuning, ORPO optimization, CUDA learning resources, and cleaning the **ShareGPT90k** dataset for training.

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files?short_path=3520786#diff-35207863e6e0da8dfa2d1311bf863b60c52a067c5e65253c24543edda5da00d0)** provided a guide for multi-node distributed fine-tuning, while **[Modular](https://github.com/venvis/mojo2py)** introduced mojo2py to convert Mojo code to Python.

- **[CUDA MODE](https://github.com/cuda-mode/lectures/tree/main/lecture%2014)** shared lecture recordings, with focuses on CUDA optimization, quantization techniques like **[HQQ+](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py)**, and the llm.C project for efficient kernels.

**3. AI Hardware and Deployment Advancements**

- Discussions around **[Nvidia's potential early RTX 5090 launch](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch)** due to competitive pressure and the anticipated performance gains.

- **[Strong Compute](https://strongcompute.com/research-grants)** announced grants of **$10k-$100k** for AI researchers exploring trust in AI, post-transformer architectures, new training methods, and explainable AI, with GPU resources up for grabs.

- **[Limitless AI](https://x.com/dsiroker/status/1779857843895599383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)**, previously known as Rewind, introduced a wearable AI device, sparking discussions around data privacy, HIPAA compliance, and cloud storage concerns.

- **[tinygrad](https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455)** explored cost-effective GPU cluster setups, MNIST handling, documentation improvements, and enhancing the developer experience as it transitions to version 1.0.

- Deployment insights like **[packaging custom models into llamafiles](https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790)**, running CUDA on consumer hardware, and converting models from ONNX to WebGL/WebGPU using tinygrad.

**4. AI Safety, Ethics, and Societal Impact Debates**

- Discussions around the ethical implications of AI development, including the need for **[safety benchmarks like ALERT](https://github.com/Babelscape/ALERT)** to assess potentially harmful content generation by language models.

- Concerns over the spread of misinformation and unethical practices, with mentions of a potential AI scam advertised on Facebook called [Open Sora](https://www.open-sora.org).

- Debates on finding a balance between AI capabilities and societal expectations, with some advocating for creative freedom while others prioritize safety considerations.

- Philosophical exchanges comparing the reasoning abilities of AI systems to humans, touching on aspects like independent decision-making, emotional intelligence, and the neurobiological underpinnings of language comprehension.

- Emerging legislation targeting deepfakes and the creation of explicit AI-generated content, prompting discussions around enforcement challenges and intent considerations.

**5. Misc**

- **Excitement and Speculation Around New Models**: There was significant buzz and discussion around the release of new AI models like **Pile-T5** from [EleutherAI](https://blog.eleuther.ai/pile-t5/), **Idefics2 8B** from [Hugging Face](https://huggingface.co/HuggingFaceM4/idefics2-8b), **Reka Core** from [Reka AI](https://publications.reka.ai/reka-core-tech-report.pdf), and **WizardLM 2** from Microsoft (despite its [mysterious takedown](https://fxtwitter.com/pimdewitte/status/1780066049263538653?s=46)). The AI community eagerly explored these models' capabilities and training approaches.

- **Advancements in Multimodal AI and Diffusion Models**: Conversations highlighted progress in **multimodal AI** with models like [IDEFICS-2](https://huggingface.co/blog/idefics2) showcasing advanced OCR, visual reasoning and conversational abilities. Research into **diffusion models for video generation** ([Lilian Weng's blog post](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)) and the significance of **tokenization in language modeling** ([UC Berkeley paper](https://arxiv.org/abs/2404.08335)) also garnered interest.

- **Tooling and Frameworks for Model Development**: Discussions covered various tools and frameworks for AI development, including **Axolotl** for [multi-node distributed fine-tuning](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files?short_path=3520786#diff-35207863e6e0da8dfa2d1311bf863b60c52a067c5e65253c24543edda5da00d0), **LangChain** for [building LLM applications](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction), **tinygrad** for [efficient deep learning](https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455), and **Hugging Face's libraries** like [parler-tts](https://github.com/huggingface/parler-tts) for high-quality TTS models.

- **Emerging Platforms and Initiatives**: The AI community took note of various emerging platforms and initiatives such as **Limitless** ([rebranded from Rewind](https://x.com/dsiroker/status/1779857843895599383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)) for personalized AI, **Cohere Compass** beta for [multi-aspect data search](https://txt.cohere.com/compass-beta/), **Payman AI** for [AI-to-human task marketplaces](https://www.paymanai.com/), and **Strong Compute's** [$10k-$100k grants for AI research](https://strongcompute.com/research-grants). These developments signaled an expanding ecosystem for applied AI.

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3: A Legend or Reality?**: The release date for **Stable Diffusion 3 (SD3)** has turned into folklore within the community, with speculated dates like "April 26" and "April 69" reflecting excitement and sarcasm about its long-awaited arrival.
- **Animating Images Becomes Comfier**: Engineers are exchanging tips for image animation, pointing towards [ComfyUI workflows](https://comfyanonymous.github.io/ComfyUI_examples/video/) and Stability Forge while mentioning challenges in running models directly via Python, highlighting a need for simpler animation APIs.
- **Pixel Art Gets a Realistic Touch**: There's a buzz about turning pixel art into realistic images using models like **SUPIR**, available on [Hugging Face](https://huggingface.co/camenduru/SUPIR), and enhancing pixelated images using **img2img controlnet** solutions like Fooocus or [Magic Image Refiner](https://github.com/BatouResearch/magic-image-refiner).
- **Prompt Engineering Debate**: Prompt crafting best practices were discussed, with some arguing that expertise is not necessary and using services like [Civitai](https://civitai.com/posts/2163684) can provide solid prompting baselines; discussions around WizardLM-2 were redacted due to untested toxicity concerns.
- **AI Utopia or Dystopia?**: Casual conversations veered into envisioning a future saturated with advanced AI applications, from style conversion in gaming to AI-assisted brain surgery, interspersed with jokes about an "AI University for prompt engineers" and tongue-in-cheek product names like "Stable Coin."



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Benchmark Bonanza**: Engineers shared positive feedback on a first benchmark's results, praising its performance. There was also a conversation around extracting `tokenizer.chat_template` for model template identification in leaderboards.

**Progressive Techniques in LoRA Tuning**: Community members exchanged tips on LoRA fine-tuning, suggesting that the alpha parameter to tweak could be double the rank. They discussed ORPO's resource-optimization in model training and discouraged the use of native CUDA, advocating for Triton instead for learning and development benefits.

**Data Hygiene Takes Center Stage**: The **ShareGPT90k** dataset was presented in a cleaned and ChatML format to facilitate training with Unsloth AI, and users highlighted the key role of data quality in model training, alluding to a community preference for hands-on experimentation in learning model training approaches.

**Collaboration and Contributions on the Rise**: Open calls for contributions to Unsloth documentation and projects such as **Open Empathic** were made, indicating a receptive attitude toward community involvement. A member announced the development of an "emotional" LLM and collaboration with a Chroma contributor on libSQL and WASM integration.

**Navigating Unsloth's Notebook Nuggets**: Assistance with formatting personal messages for AI training was given, complete with a Python script link and a guide to use the ShareGPT format. Advice on packing and configurations for Gemma models were discussed to mitigate unexpected training issues.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Bold Python Package Sets to Conquer Mojo Code**: The creation of [mojo2py](https://github.com/venvis/mojo2py), a Python package to convert Mojo language code into Python, indicates a trend toward developing tools for Python and Mojo interoperability.

**Grammar Police Tackle Code Aesthetics**: Engaging discussions highlighted the importance of *indenting code*, considered laughable yet significant for readability, and there was a sense of light-hearted camaraderie over code formatting conventions.

**Accolades for Achieving Level 9 in Modular**: A community member was congratulated for reaching **level 9**, indicating a point system or achievement metric within the Modular community.

**Modular Tweets Tease the Tech-Savvy**: A series of [mysterious tweets from Modular](https://twitter.com/Modular) sparked speculation and interest among the community, serving as an intriguing marketing puzzle.

**Nightly Updates Kindle Community Interest**: A fresh **Mojo nightly update** was announced, directing engineers to update their version to `nightly/mojo` and review the latest changes and enhancements detailed on [GitHub's diff](https://github.com/modularml/mojo/pull/2313/files) and the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Billing Confusion and API Misalignments**: Users express dissatisfaction with unexpected charges and discrepancies between **Perplexity AI** and API usage, pointing to instances where promo codes don't appear and seeking an understanding of parameters such as **temperature** for consistent results between different platforms.

**Pro Feature Puzzlement**: Changes to the **Pro message counter** in Perplexity AI led to mixed reactions, with some users enjoying "reduced stress" but others questioning the rationale behind such feature tweaks.

**Model Performance Scrutiny**: A divergence in opinion emerges on AI coding competencies, with **GPT-4** seen as inadequate by some users, while others ponder the delicate trade-offs between various **Perplexity** models' abilities and performance.

**Cultural Curiosity and Tech Talk**: The community engages in a range of searches, from probing **Microsoft's ad-testing endeavors** to celebrating global cultural days, reflecting an eclectic mix of technical and creative interests.

**API Result Inconsistencies Provoking Discussions**: Queries in the community focus on aligning outcomes from Perplexity Pro and the API, with an undercurrent of worries about hallucinations and source credibility in the API's content.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Windows Cleared for Model Takeoff**: Responding to queries, members confirmed that the Windows executables for LM Studio are **signed with an authenticode certificate** and discussed the cost differences between Windows certificates and Apple developer licenses, with the former requiring a hardware security module (HSM).

**The Trouble with VRAM Detection**: Users reported errors related to AMD hardware on Intel-based systems in Linux, despite attempts to solve the issue with `ocl-icd-opencl-dev`. It led to a broader discussion about hardware misidentification and the challenges it poses in configurations.

**WizardLM-2 Sharpens Its Conversational Sword**: The *WizardLM 2 7B* model was praised for its ability in multi-turn conversations and its training methods, with its availability announced on [Hugging Face](https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF). The WaveCoder ultra 6.7b was also recognized for its coding prowess following fine-tuning on Microsoft's CodeOcean.

**Model Showdown**: Users shared performance experiences with models like **WizardLM-2-8x22B** and **Command R Plus**, voicing mixed reactions. They exchanged views on what defines a "Base" AI model and the nuances of model fine-tuning and continuous learning, sparking debates over AI memory and bias.

**Diverse Coding Prowess Under the Microscope**: Within the guild, members delved into Python coding model capabilities, like *Deepseek Coder* and *Aixcoder*, urging others to check 'human eval' scores. Skepticism was expressed over claims about WaveCoder Ultra's superiority, with some implying exaggerated results, while discussions on model fine-tuning and quantization illuminated varying preferences for coding models and AI agent creation tools.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Engineers Tackle Tokenization Trouble**: An engineer experienced difficulties with tokenized outputs for end-of-sequence predictions using **[llama.cpp](https://docs.rs/llama_cpp/0.3.1/llama_cpp/index.html)** with the OpenHermes 2.5 Mistral 7B model on [Hugging Face](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF) and sought advice on resolving the issue.

- **Tech Titans' AI Tools Scrutinized**: Users compared Reka AI's Core model with GPT-4V, Claude-3 Opus, and Gemini Ultra in a [showcase](https://showcase.reka.ai/) and discussed Google's [CodecLM](https://arxiv.org/abs/2404.05875), which aims for high-quality synthetic data generation for language model alignment.

- **Innovation or Hype? Open AI Models Excite but Confuse**: Despite the enthusiastic downloading of WizardLM-2 before its takedown, confusion remained about its removal, while new models like CodeQwen1.5 promise enhanced code generation in 92 languages, and Qwen's 7B, 14B, and 32B models are mentioned for their benchmark scores.

- **Breaking Binary Boundaries**: Discussion of a binary quantization-friendly AI model on Hugging Face sparked interest due to memory-efficient **int8** and **binary embeddings**, with calculation methods like **XOR operations** for embedding distance, cited from [Cohere's blog post](https://txt.cohere.com/int8-binary-embeddings/).

- **Game Design Meets Quantum Probability**: Enthusiasm about **WorldSim's** potential to revolutionize game development is evident, with conversations about using LLMs in future to affect in-game variables and content creation, flavoring it with undertones of AI-assisted omnipotence.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **WizardLM Models Magically Appear on OpenRouter**: The **WizardLM-2 8x22B** and **WizardLM-2 7B** models from Microsoft have been added to [OpenRouter](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b), with the former's cost now at $0.65/M tokens. Several members have initiated a thread to discuss the implications of this addition.

- **Intermittent Latency Looms Over Users**: There were reports of high latency issues affecting models like **Mistral 7B Instruct** and **Nous Capybara 34b**, with the problem traced back to an upstream issue with a cloud provider's DDoS protection. Further complications have led to said provider being deranked to alleviate concerns, and global users are being called upon to report their experience with the issue.

- **Rubiks.ai Rolls Out Beta with Big Name Models**: A new AI platform, [Rubiks.ai](https://rubiks.ai), is courting beta testers with the offer of 2 months free premium access and the chance to experiment with models such as **Claude 3 Opus, GPT-4 Turbo, and Mistral Large**. Users facing account-related issues are advised to send feedback directly to the developers.

- **Falcon 180B Soars With GPU Thirst**: The hefty GPU memory requirement of around 100GB for the **Falcon 180B Chat GPTQ** got the community talking about both the model's resource intensity and its potential usage considerations, underlined by a link to the [Falcon 180B repository](https://huggingface.co/TheBloke/Falcon-180B-Chat-GPTQ).

- **Cost-effective Communication & Model Responsiveness Advice**: In a nod to efficient model usage, it was proposed that an average of **1.5 tokens per word** could be a benchmark for cost calculation. Separate discussions highlighted positive attributes of the Airoboros 70B's prompt compliance, contrasting it with less consistent models.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**PyTorch Book Still Flares Interest**: Despite being 4 years old, "Deep Learning with PyTorch" is seen as a useful foundation for PyTorch fundamentals, while chapters on transformers, LLMs, and deployment are dated. Anticipation grows for a new edition to cover recent advancements.

**Torch and CUDA Grapple with Optimization**: Understanding and implementing custom backward operations in **Llama** exhibit challenges for AI engineers, while the use of `torch.nn.functional.linear` and the **stable-fast** library are leading discussions for optimizing inference in the CUDA environment.

**Novel Approaches in Transcript Processing**: An automated transcript for a CUDA talk utilizing cutting-edge tools is provided by [Augmend Replay](https://wip.augmend.us/replay/PDHePF8AAA), offering the AI community OCR and segmentation features for video content analysis.

**Quantum Leaps with HQQ and GPT-Fast**: Significant strides in token generation speeds are observed after implementing torchao int4 kernel in the generation pipeline for transformers, rising to **152 tokens/sec**. The **HQQ+** method also marked an accuracy increase, spurring discussions around quantization axis and integration with other frameworks.

**llm.C at the Forefront of CUDA Exploration**: The llm.C project ignites discussions on CUDA optimizations, underscoring the balance between education and creating efficient kernels. Optimizations, profiling, potential strategies, and applicable datasets all jostle for attention in this growing space.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pile-T5 Reveals Impressive Benchmarks**: EleutherAI introduced **Pile-T5**, a T5 model family trained on the Pile with up to 2 trillion tokens and showing improved performance on language tasks with the new LLAMA tokenizer. Key resources include a [blog post](https://blog.eleuther.ai/pile-t5/), [GitHub repository](https://github.com/EleutherAI/improved-t5), and a [Twitter announcement](https://x.com/arankomatsuzaki/status/1779891910871490856).

- **Tackling the Temporal Challenge in Video Diffusion**: Discussions in the research channel touched on the complexities of video synthesis using diffusion models, with participants referring to a [post on diffusion models for video generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) and deliberating on the importance of tokenization in language models, as outlined in a [Berkeley paper](https://arxiv.org/abs/2404.08335).

- **LLM Evaluation Continues to Evolve**: Within the lm-thunderdome channel, there were insights into OpenAI's public release of GPT-4-Turbo's evaluation implementations on [GitHub](https://github.com/openai/simple-evals), and enhancement of `lm-evaluation-harness` with new benchmarks such as `flores-200` and `sib-200`.

- **Token Management Under the Microscope**: The gpt-neox-dev channel broached technical issues such as the effect of weight decay on dummy tokens, the necessity of sanity checks after model adjustments, and token encoding behaviors with shared code outputs demonstrating token transformations.

- **Debates on Model Architecture Efficiency**: Active discussions contrasted dense models with Mixture-of-Experts (MoE), debated their efficiency, inference cost, and constraints, showing the ongoing quest for optimizing language model architectures.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Brains and Bots**: Members shared interest in Angela D. Friederici's book, [Language in Our Brain: The Origins of a Uniquely Human Capacity](https://direct.mit.edu/books/oa-monograph/3653/Language-in-Our-BrainThe-Origins-of-a-Uniquely), sparking dialogue on the neurobiological underpinnings that differentiate humans and AI in language capacities. The discussions stressed the challenge in neuroscience of handling the 'data glut' from Big Brain Projects and the proprietary hurdles that hinder data interpretation.

- **AI Limitations and Liberties**: In a look at the contrast between AI and humans, it emerged that artificial systems are yet to match human-like storage of learned information, independent decision-making abilities, and emotional responses. The reference to Claude 3 API's accessibility issues in Brazil underscored the geographical nuances in reaching AI tools.

- **Chatbots Grapple with GPT Constraints**: Despite advances, the GPT's context window remains a critical concern, with GPT-3's API permitting a 128k context but ChatGPT itself constrained to 32k. Mechanisms like **"retrieval"** through document upload were demystified, allowing extension of the effective context window within the API's framework.

- **Discovering the Depths of Turing Completeness**: A spirited debate arose about the Turing completeness of Magic: The Gathering, suggesting that the concept extends its reach well beyond traditional computational systems.

- **Clouded Queries and Cryptic Replies**: Prompt Engineering and API Discussions channels surfaced brief and ambiguous exchanges about an unidentified competition and cryptic one-word responses such as "buzz" and "light year," underscoring the occasional opacity in dialogue within technical forums.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Tutorial Treasure Trove**: LlamaIndex announced an [introductory tutorial series](https://twitter.com/llama_index/status/1779898403239125198) for agent interfaces and applications, aiming to clarify usage of core agent interfaces. In collaboration, LlamaIndex and Qdrant Engine introduced a [hybrid cloud service offering](https://twitter.com/llama_index/status/1780275878230139293), and a new tutorial was shared highlighting the integration of LlamaIndex with Azure AI to leverage hybrid search in RAG applications, crafted by Khye Wei from Microsoft [found here](https://twitter.com/llama_index/status/1780324017083400235).

**AI Chat Chops**: Within the LlamaIndex community, discussion ranged from implementing async compatibility with Claude in Bedrock (where async has not yet been implemented) to complex query construction help available in the [documentation](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/?h=query+pipeline). Integration issues with gpt-3.5-turbo and LlamaIndex were likely related to outdated versions or account balances, and configuring fallbacks for decision-making with incomplete data remains an open challenge.

**Reasoning Chains Revolution**: Revealing advancements in reasoning chain integration with LlamaIndex, a key article titled "Unlocking Efficient Reasoning" [can be found here](https://ai.gopubby.com/unlocking-efficient-reasoning-integrating-llamaindex-with-chain-of-abstraction-1b1844ba66e6). Solutions for token counting in RAGStringQueryEngine and hierarchical document organization in LlamaIndex were discussed in detail, with the community providing a concrete token counter integration guide involving a `TokenCountingHandler` and `CallbackManager` as per LlamaIndex's [reference documentation](https://docs.llamaindex.ai/en/latest/api_reference/callbacks/token_counter#llama_index.core.callbacks.token_counting.TokenCountingHandler.prompt_llm_token_count).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Hugging Face Rings in New TTS Library**: A [high-quality TTS model library](https://github.com/huggingface/parler-tts), *parler-tts*, for both inference and training was showcased, bolstered by its hosting on Hugging Face's community-driven platform.

**Scaling Down CLIP – Less Data, Equal Power**: A [study on CLIP](https://arxiv.org/abs/2404.08197) demonstrates that strategic data use and augmentation can allow smaller datasets to match the performance of the full model, introducing new considerations for data-efficient model training.

**Deepfakes – Legislation Incoming, Controversies Continue**: The community debated newly proposed laws against deepfakes as well as unethical practices in AI, raising awareness about a potential scam promoted through a suspicious site advertised on Facebook, found [here](https://www.open-sora.org).

**Safety Benchmarking Becomes ALERT**: Discussion on the importance of safety in AI highlighted the release of the [ALERT benchmark](https://github.com/Babelscape/ALERT), designed to evaluate large language models for handling potentially harmful content and reinforcing conversations around safety versus creative freedom.

**Audio Generation Advancements on the Horizon**: Research involving the Tango model to [enhance text-to-audio generation](https://arxiv.org/abs/2404.09956) shed light on improvements in relevance and order of audio events, marking progress for audio generation from text in data-scarce setups.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **IDEFICS-2 Shines in Multimodal Processing**: The recently released [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) enhances multimodal capabilities, excelling in tasks such as image and text sequence processing, and is set to get a **chatty variant** for conversational interaction. When probed, demonstrations of its capabilities like decoding CAPTCHAs with heavy distortion were highlighted.

- **Diverse AI Insights and Queries**: Community members have raised various topics ranging from **BLIP model fine-tuning** for long captions, musical AI projects like `.bigdookie's` [infinite remix GitHub repo](https://github.com/betweentwomidnights/infinitepolo), and the usage of Java for image recognition outlined in a [Medium article](https://medium.com/@visrow/image-recognition-and-function-calling-with-gemini-and-java-e28b0356d3de). Discussions also cover best practices for collaborative work on HuggingFace and survey participation from machine learning practitioners.

- **Unlock the Potential of BERTopic**: AI engineers engaged in deep dives into frameworks like **BERTopic**, which revolutionizes topic modeling through the use of transformers. It's lauded for its performance and versatility, with guides like the [BERTopic Guide](https://maartengr.github.io/BERTopic/index.html) assisting users in navigating its myriad capabilities for structured topic extraction.

- **Clarifying NLP and Diffusion Model Confusions**: Clarifications were sought for NLP tensor decoding using T5 models and LoRA configurations, while questions about the differences between LLMs and embedding models, and issues with token limits in diffusion models were also discussed. A community member flagged a warning regarding token truncation when using **stable diffusion models**, referring to an open [GitHub issue](https://github.com/huggingface/diffusers/issues/7672).

- **Vision and NLP Model Optimization Efforts**: Engineers have shown interest in tuning models for specific use cases, such as a vision model for low-resolution image captioning and the potential use of advanced taggers for SDXL. Similarly, advice is sought for preparing a dataset for fine-tuning a **ROBERTA Q&A chatbot** and utilizing models like spaCy and NLTK for getting started in NLP.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Command-R Struggles with Macedonian**: Discussions flagged that **Command-R** doesn't perform well in Macedonian, with concerns raised on the community-support channel. Issues raised highlight the need for multilingual model improvements.

**Asynchronous Streaming with Command-R**: Engineers queried the best practices for converting synchronous code to asynchronous in Python, aiming to enhance the efficiency of chat streaming with the **Command-R** model.

**Trial API Limits Clarified**: For **Cohere's API**, engineers discovered that the ‘generate, summarize’ endpoint has a limit of 5 calls per minute, while other endpoints permit 100 calls per minute, with a shared pool of 5000 calls per month for all trial keys.

**Commander R+ Gains Traction**: A discussion took root around accessing **Commander R+** using Cohere’s paid **Production API**, highlighting existing [documentation](https://docs.cohere.com/reference/chat) for potential subscribers.

**Rubiks.ai Introduces AI Powerhouse**: Engineers took note of the launch of [Rubiks.ai](https://rubiks.ai), which offers a suite of models including **Claude 3 Opus, GPT-4 Turbo, Mistral Large**, and **Mixtral-8x22B**, with an introductory offer of 2 months of premium access on Groq servers.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Deepspeed's Multi-node Milestone**: A guide for **multi-node distributed fine-tuning** using Axolotl with **Deepspeed 01 and 02** configurations was shared. The [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files?short_path=3520786#diff-35207863e6e0da8dfa2d1311bf863b60c52a067c5e65253c24543edda5da00d0) outlines steps to address configuration issues.

**Idefics2 Raises the Bar**: The newly released **Idefics2 8B** on Hugging Face surpasses **Idefics1** in OCR, document understanding, and visual reasoning with fewer parameters. Access the model on [Hugging Face](https://huggingface.co/HuggingFaceM4/idefics2-8b).

**Pacing for RTX 5090's Big Reveal**: Anticipation builds for Nvidia's upcoming **RTX 5090** graphics card, speculated to debut at the Computex trade show. This early release may be fueled by competitive pressure as discussed on [PCGamesN](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch).

**Gradient Accumulation Spotlighted**: Queries on **gradient accumulation**'s memory conservation in the context of sample packing and dataset length led to explorations of its impact on training time.

**Streamline Model Saving with Axolotl**: Configuring Axolotl to save models only upon training completion rather than after each epoch involves setting `save_strategy` to `"no"`. Additionally, "TinyLlama-1.1B-Chat-v1.0" was recommended for tight computational spaces, with its setup in the `examples/tiny-llama` directory of Axolotl's repository.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Rewound Now Unbound as Limitless**: The wearable tech previously referred to as **Rewind** has been rebranded to [**Limitless**](https://x.com/dsiroker/status/1779857843895599383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), sparking a discussion about its real-time application potential and the implications for future AI advancements. Concerns regarding **data privacy** and **HIPAA compliance** for cloud-stored information were vocalized by members.

**The Birth of Reka Core**: [**Reka Core**](https://x.com/rekaailabs/status/1779894622334189592?s=46&t=90xQ8sGy63D2OtiaoGJuww) enters the chat as a *multimodal language model* that comprehends video. The community appears intrigued by the small team achievement in AI democratization and the technical report released at [publications.reka.ai](https://publications.reka.ai/reka-core-tech-report.pdf).

**Cohere Compass Beta Steers In**: Cohere's Compass Beta was unveiled as a next-level data search system, meriting discussion around its embedding model and the beta testing opportunities for applicants eager to explore its functional boundaries.

**Payman AI Explores AI-Human Marketplaces**: [**Payman AI**](https://www.paymanai.com/) piqued interest with its innovative concept of a marketplace where AI can hire humans, driving conversations around implications for data generation and advancing AI training methodologies.

**Strong Compute Serves Resources on Silver Platter**: **Strong Compute** revealed a **grant program** for AI researchers, dangling the carrot of **$10k-$100k** and substantial GPU resources for initiatives in *Explainable AI*, *post-transformer models*, and other groundbreaking areas, with a swift application deadline signaled by the end of April. Details on the offer and the application process were outlined at [Strong Compute research grants page](https://strongcompute.com/research-grants).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**AI Innovation Storm Brewing**: The OpenInterpreter community launched a brainstorming space to ideate on uses of the platform, focusing on features, bugs, and innovative applications.

**Voice Communication Soars with Airchat**: There’s a buzz around **Airchat** within the community as engineers exchange usernames and scrutinize its features and usability, signaling a growing interest in diverse communication platforms.

**Open Source AI Generates Excitement**: Opensource AI models, notably **WizardLm2**, are receiving attention for providing transparent access to powerful AI capabilities akin to GPT-4, highlighting community interest in open-source alternatives.

**Navigating the 01 Pre-order Process**: For those reconsidering their **01 pre-orders**, they can easily cancel by reaching out to help@openinterpreter.com, and there’s growing discussion on Windows 11 installation woes and hardware compatibility improvisations using parts from AliExpress.

**Linux Love for OpenInterpreter**: Linux users are directed to [rbrisita's GitHub branch](https://github.com/rbrisita/01/tree/linux), agglomerating all the latest PRs for the **01** device, and the community is also optimizing their 01 setups with custom designs and battery life improvements.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Documentation Revamp Requesting Feedback**: LangChain engineers have outlined a **new documentation structure** to better categorize tutorials, how-to guides, and conceptual information, to improve user navigation across resources. Feedback is sought, and an [introduction to LangChain](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction) has been made available, detailing its application lifecycle process for large language models (LLMs).

- **Parallel Execution and Azure AI Conflict Solving**: Technical discussions confirmed that **LangChain's `RunnableParallel` class** allows for concurrent execution of tasks, with reference to [Python documentation](https://api.js.langchain.com/classes/langchain_core_runnables.RunnableParallel.html) for parallel node running. Meanwhile, solutions are being exchanged on issues with `neofjVectorIndex` and `faiss-cpu`, including LangChain version rollbacks and branch switches.

- **Innovations and Announcements Flood LangChain**: A series of project updates and community exchanges highlighted advancements such as improved **Rag Chatbot** performance via multiprocessing, the introduction of [Perplexica](https://github.com/ItzCrazyKns/Perplexica/) as a new AI-driven search engine, and the launch of tools like **Payman** for AI-to-human payments, viewable at [Payman.ai](https://www.youtube.com/watch?v=xZiTSZ5SOYc&t=5). Other announcements included GalaxyAI's free premium model access, OppyDev's AI-assisted coding tool ([oppydev.ai](https://oppydev.ai)), and a call for beta testers for Rubiks AI's research assistant with perks ([rubiks.ai](https://rubiks.ai)).

- **Channeling AI for RBAC Implementation and YC Aspirations**: Specific discussions touched on implementing role-based access control (RBAC) within LangChain for large organizations and gauging the landscape for finetuning models for YC applications, indicating both challenges and existing companies like Holocene Intelligence in the space.

- **Nurturing AI with Memory and Collaborative Efforts**: Shared knowledge included a [video](https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek) on crafting AI agents with long-term memory and a call for collaboration in integrating **LangServe** with **Nemo Guardrails**, suggesting a need for a new output parser due to updates. Community members also explored payment-enabled AI recommendations and document processing concerns, all hinting at an emphasis on shared growth and collaborative experimentation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Budget-Friendly GPU Clusters**: Engineers discussed a **cost-effective alternative** to TinyBox using six RTX 4090 GPUs, resulting in up to a **61.624% cost reduction** compared to the $25,000 TinyBox model. The emphasis was on achieving 144 GB of GPU RAM within a budget.

- **A Potential BatchNorm Bug**: George Hotz called for a test case to investigate a potential bug in tinygrad's **batchnorm** implementation, following a user's concern about the order of operations involving `invstd` and `bias`.

- **Navigating Tinygrad's Documentation Work**: Participants recognized the need to enhance **tinygrad documentation**, with ongoing efforts to make strides towards more comprehensive guides, particularly as the system evolves from version 0.9 to 1.0.

- **Strategies for Model Conversion**: Users are exploring ways to convert models from ONNX to WebGL/WebGPU efficiently, targeting memory optimization by potentially leveraging tinygrad's `extras.onnx` module, as indicated by interest in [Stable Diffusion WebGPU](https://github.com/softwiredtech/stable-diffusion-webgpu) examples.

- **Improving Tinygrad Development Experience**: The community suggested increasing the **line limit for merging NV backends to 7,500 lines**, as seen in a [recent commit](https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455), to balance codebase inclusiveness and quality, while addressing experiences of error comprehensibility.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**AI Models Flood the Market**: EleutherAI has introduced the **Pile-T5** with details shared in a [blog post](https://blog.eleuther.ai/pile-t5/), while **WizardLM 2** is drawing interest with its foundation transformer tech and guide on [WizardLM's page](https://wizardlm.github.io/WizardLM2/). Additionally, **Reka Core** breaks onto the scene as explained in its [technical report](https://publications.reka.ai/reka-core-tech-report.pdf), and Idefics2's debut is narrated on the [Hugging Face blog](https://huggingface.co/blog/idefics2), amid **Dolma** going open-source under an ODC-BY license.

**Graph Love and Hefty Models Emit Buzz**: The community is showing keen interest in turning sophisticated graphs into a **Python library** for model exploration, while expressing mixed reactions to LLAMA 3's massive training scale of **30 trillion tokens**.

**WizardLM Vanishes with Abrupt Apology**: Tension rose with the unexplained removal of **WizardLM**, with its model weights and posts erased, prompting [speculation](https://fxtwitter.com/pimdewitte/status/1780066049263538653?s=46) and an apology from WizardLM AI over a missed **toxicity test**, and a potential re-release in the pipeline.

**Exploration vs. Intervention**: A member considers whether to leave a bot to its own learning process or to step in, illustrating the fine line between letting algorithms explore and manual intervention.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Debate on Data Annotation Necessity**: In a recent discussion, participants explored whether the traditional practice of dataset annotation prior to training models is still critical given the rise of advanced LLMs. They pondered if in-depth understanding of datasets remains important or models can sufficiently learn patterns independently.

- **Transparency in LLM Demos Demanded**: Dissatisfaction was voiced over LLM demos that lack open prompts, with users favoring clear insight into the model behavior to achieve desired outcomes without guesswork. Concerns were also raised about models inconsistently following privacy directives during tasks such as indexing sensitive information.

- **Streamlit Eases LLM Log Browsing**: An LLM web UI for more user-friendly navigation of log data has been created using Streamlit, with an aim for simpler revisiting of past chats compared to Datasette. The interface currently supports log browsing and the creator provided the initial code via a [GitHub gist](https://gist.github.com/RyanBalfanz/cd08b7402594fa91831bf8d54c76e0ec).

- **Call for Interface Integration Ideas**: Following the showcase of the web UI prototype, discussion ensued regarding the possibility of its integration either as a Datasette plugin or as a standalone tool, pondering the practicality and long-term utility of such enhancements.

- **Quest for a Consistent Indexing Tool**: Exchanges highlighted the unpredictability of language models in handling tasks like newspaper indexing, particularly with models refusing to list names in adherence to privacy norms. The conversation underscored the need for more reliable tools and noted reaching out to Anthropic for assistance with model refusals.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **WizardLM2 Disappears from Hugging Face**: The **WizardLM2** collection on Hugging Face has vanished, and a collection update shows all models and posts are now missing. A direct link to the update is provided here: [WizardLM Collection Update](https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a).
  
- **Potential Legal Concerns for WizardLM2**: There's an unconfirmed question circulating about whether the removal of **WizardLM2** is due to legal concerns, though no further information or sources are cited to clarify the nature of these potential issues.
  
- **Rush for WizardLM2 Resources**: The community is actively seeking anyone who might have downloaded the **WizardLM2** weights prior to their deletion.

- **Evidence of WizardLM2's Erroneous Deletion**: A community member shared a screenshot that provides evidence that **WizardLM2** was deleted as a result of improper testing. The screenshot can be viewed here: [WizardLM2 Deletion Confirmation](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png?ex=66309ca9&is=661e27a9&hm=f105e6497796be9c414ade2024a27f9561caf0cad6cb06ba09f80e30b5e39ae4&).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**LLama-Tokenizer Training Troubles**: Engineering members shared challenges in training a **Llama-tokenizer** with the goal of achieving hardware compatibility via reduced embedding and output perceptron sizes. They explored scripts like [convert_slow_tokenizer.py from Hugging Face](https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534) and [convert.py from llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/convert.py) to aid in the process.

**Hunt for EU Copyright-Compliant Resources**: There's an active quest to find text and multimodal datasets compatible with EU copyright laws for training a multimodal model. Suggestions for starting points included Wikipedia, Wikicommons, and [CC Search](https://search.creativecommons.org/) to gather permissive or free data.

**Sampling Strategies Examined**: Discourse in the engineering circles revolved around decoding strategies for language models, emphasizing the need for academic papers to include modern methods like MinP/DynaTemp/Quadratic Sampling. A shared [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/) offers a layman's comparison, while the conversation called for more rigorous research into these strategies.

**Decoding Methodology Deserves a Closer Look**: An examination of decoding methods in LLMs has exposed a gap in current literature, specifically related to open-ended tasks seen in operational models. Members expressed the need for in-depth research on advanced sampling methods and their impacts on model performance.

**Creative Writing Boost with MinP Sampling**: A notable performance boost in creative writing tasks was highlighted, with the **alpaca-eval style elo** score increasing by +8 and the **eq-bench creative writing test** seeing a +10 increment due to min_p sampling parameters. Such improvements signify the potential impacts of fine-tuning sampling strategies on LLM outputs.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **NYC Event for Scaling Gen AI Apps**: A General AI enthusiast **meetup** at Gunderson Legal in New York City will focus on **scaling Gen AI applications to production** stages. The event details and registration link are available [here](https://lu.ma/llms-in-prod-nyc), alongside a note of participation by industry leaders from [Portkey](https://portkey.ai/) and [Noetica](https://www.linkedin.com/in/yonisebag/).

- **Reka Core Emerges as a Strong Contender**: A new video titled "Reka Core: A Frontier Class Multimodal Language Model" showcases **Reka Core** holding its own against competing models by OpenAI, Anthropic, and Google, highlighted in a [YouTube video](https://www.youtube.com/watch?v=vL1SayPCHBg).

- **JetMoE-8B Achieves Cost-Efficient Superiority**: The **JetMoE-8B model**, developed on a budget under $0.1 million, reportedly excels past Meta AI's LLaMA2-7B, a model created with significantly larger wealth, as revealed in a [YouTube video](https://www.youtube.com/watch?v=Z9Hwp_XeS1A).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Packaging Custom Models Just Got Easier**: A community member's request for guides on packing customized AI models into a **llamafile** was noted, aiming to support peers in their endeavors.
- **Docker Deployment Demystified**: A **GitHub pull request** provided walkthrough steps for engineers to **build and publish containers to Docker Hub** using GitHub Actions, complete with necessary setting up of repository secrets like `DOCKER_HUB_USERNAME` and `DOCKER_HUB_ACCESS_TOKEN`. [Publish container to Docker Hub](https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790).



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1229355383877931049)** (1015 messages🔥🔥🔥): 

- **In Search of the Elusive SD3**: The anticipated launch of Stable Diffusion 3 (SD3) remains a topic of speculation with various members inquiring about its release date, and many whimsically suggesting that it's a myth with satirical estimates like "April 26" or "April 69."
- **Choosing an Image to Video Workflow**: For those wanting to animate images, resources like ComfyUI workflows and Stability Forge are recommended. Some users experience difficulties running models directly in Python and seek advice on simple APIs for animation.
- **Stable Diffusion with Different Technologies**: Discussion threads touched upon various aspects of Stable Diffusion and related AI advances such as pixel art conversion to realistic images with models like SUPIR, and the transformation of pixelated images using img2img controlnet-based solutions like Fooocus or Magic Image Refiner.
- **Prompt Crafting and Model Discussion**: Users debate prompt engineering, with suggestions that you don't need to be an expert or take a course to craft an effective prompt; using platforms like civitai can give decent prompting baselines. New advancements like WizardLM-2 briefly appear before being deleted for untested toxicity.
- **Casual Banter and AI Future Musings**: The community casually jokes about "Stable Coin," "Stable Miner," and "university for prompt engineers," while also envisioning a world with advanced AI technologies like game style conversions and AI brain surgery, reflecting both humour and aspirational hopes for AI development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/video/">Video Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://www.amazon.com/Supermicro-Customized-Platinum-Baseboard-Analytics/dp/B0CSVN4YXW">no title found</a>: no description found</li><li><a href="https://huggingface.co/camenduru/SUPIR">camenduru/SUPIR · Hugging Face</a>: no description found</li><li><a href="https://civitai.com/posts/2163684">Perturbed-Attention-Guidance Test | Civitai</a>: A post by rMada. Tagged with . PAG (Perturbed-Attention Guidance): https://ku-cvlab.github.io/Perturbed-Attention-Guidance/ PROM...</li><li><a href="https://www.tiktok.com/@edwinskeletrix/video/7355974950945164586">TikTok - Make Your Day</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c50m7z/roopfacefusion_may_be_unsafe_binaries_have">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2">WizardLM/WizardLM-2 at main · victorsungo/WizardLM</a>: Family of instruction-following LLMs powered by Evol-Instruct: WizardLM, WizardCoder - victorsungo/WizardLM</li><li><a href="https://tenor.com/view/developer-recruiters-programmer-frontend-gif-23808695">Developer Recruiters GIF - Developer Recruiters Programmer - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=Dn1zjeV8Tco&list=PLPAeYpPQiY11tARkMMhXFyYjjCY9u20GG&index=57">Consistent Cartoon Character in Stable Diffusion | LoRa Training</a>: In this video, I will try to make a consistent cartoon character by training a Lora.*After Detailer* ➜ https://github.com/Bing-su/adetailer*Kohya SS GUI* ➜ h...</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels/issues/20">Is it possible to use Flan-T5 · Issue #20 · city96/ComfyUI_ExtraModels</a>: Would it be possible to use encoder only version of Flan-T5 with Pixart-Sigma? This one: https://huggingface.co/Kijai/flan-t5-xl-encoder-only-bf16/tree/main</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c50m7z/roopfacefusion_may_be_unsafe_binaries_have_been/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=i5hvZvzcxoo">How To Install Stable Diffusion (In 60 SECONDS!!)</a>: Here&#39;s the easiest way to install Stable Diffusion AI locally FAST - Big Shout-Outs to Le-Fourbe for walking me through the process!!GitHub Link here:https:/...</li><li><a href="https://github.com/BatouResearch/magic-image-refiner">GitHub - BatouResearch/magic-image-refiner</a>: Contribute to BatouResearch/magic-image-refiner development by creating an account on GitHub.</li><li><a href="https://github.com/QuintessentialForms/ParrotLUX">GitHub - QuintessentialForms/ParrotLUX: Pen-Tablet Painting App for Open-Source AI</a>: Pen-Tablet Painting App for Open-Source AI. Contribute to QuintessentialForms/ParrotLUX development by creating an account on GitHub.</li><li><a href="https://github.com/kijai/ComfyUI-SUPIR">GitHub - kijai/ComfyUI-SUPIR: SUPIR upscaling wrapper for ComfyUI</a>: SUPIR upscaling wrapper for ComfyUI. Contribute to kijai/ComfyUI-SUPIR development by creating an account on GitHub.</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui#automatic-installation-on-windows">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://new.reddit.com/r/LocalLLaMA/comments/1c586rm/wizardlm2_was_deleted_because_they_forgot_to_test/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels">GitHub - city96/ComfyUI_ExtraModels: Support for miscellaneous image models. Currently supports: DiT, PixArt, T5 and a few custom VAEs</a>: Support for miscellaneous image models. Currently supports: DiT, PixArt, T5 and a few custom VAEs - city96/ComfyUI_ExtraModels</li><li><a href="https://comfyworkflows.com">Comfy Workflows</a>: Share, discover, &amp; run thousands of ComfyUI workflows.</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://www.amazon.com/NVIDIA-Graphics-Memory-Learning-Compute/dp/B0CMJCD2HG?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=AYCADGG1WXC5Q#customerReviews">Amazon.com: NVIDIA H100 Graphics Card, 80GB HBM2e Memory, Deep Learning, Data Center, Compute GPU : Electronics</a>: no description found
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1229348966198546453)** (430 messages🔥🔥🔥): 

- **Complications in Notebook Implementation**: Members discussed the complexity of a certain task, mentioning they will need to create a detailed notebook for it. The process was acknowledged as complicated for both the members involved and the users.

- **Confusion over Crypto and TGE**: A user asked about "mainnet" and "TGE" dates, leading to confusion among the chat participants. It was clarified that **Unsloth AI** is not associated with cryptocurrency.

- **Troubleshooting Technical Issues with Unsloth**: Users faced issues with getting continuous outputs and package errors. It was suggested to use end-of-string markers (`</s>`) to limit model generation, and members were advised to follow the Colab notebooks provided by Unsloth, which contain pre-configured settings.

- **Discussions about Upcoming Model Releases**: There were anticipations about potential new model releases, including discussions on "llama 3" and the difference between various **Unsloth** optimization tactics. Users shared resources and engaged in speculation based on reputation and past announcements, indicating a mix of excitement and nervousness about the potential workload a new release could bring to the team.

- **Contributions and Contributions to Unsloth**: Individuals expressed interest in both contributing to Unsloth documentation and making a one-time financial contribution. It was mentioned that contributions focused on expanding Unsloth's Wiki, particularly regarding **Ollama/oobabooga/vllm**, would be valuable, and the team expressed openness to community involvement in improving their documentation.

- **Surrounding Controversy and Reupload of WizardLM**: A notable incident was discussed concerning the re-upload of **WizardLM** versions on various platforms following the original release getting pulled due to a missed toxicity test. Multiple users exchanged information about the reuploads and the reasons for the original takedown.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://web.archive.org/web/20240415221214/https://wizardlm.github.io/WizardLM2/">WizardLM 2</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://i.imgur.com/ao3k2iL">screenshot</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/lucyknada/microsoft_WizardLM-2-7B">lucyknada/microsoft_WizardLM-2-7B · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 2-5X faster 80% less memory LLM finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.</li><li><a href="https://twitter.co">twitter.co - Domain Name For Sale | Dan.com</a>: I found a great domain name for sale on Dan.com. Check it out!</li><li><a href="https://youtu.be/SL2nZpv7dtY?si=Yw5JxlVhRTrBu1gA">Full fine tuning vs (Q)LoRA</a>: ➡️ Get Life-time Access to the complete scripts (and future improvements): https://trelis.com/advanced-fine-tuning-scripts/➡️ Runpod one-click fine-tuning te...</li><li><a href="https://github.com/cognitivecomputations/OpenChatML">GitHub - cognitivecomputations/OpenChatML</a>: Contribute to cognitivecomputations/OpenChatML development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1229440776317042808)** (6 messages): 

- **Coding an Emotional LLM**: *ashthescholar.* indicated that they are working on the skeleton for an "emotional" large language model (LLM) and are about to start coding, looking forward to sharing their progress.
- **Syncing Up With Chroma's Roadmap**: *lhc1921* shared their plans to work on an edge version of Chroma with libSQL (SQLite) in Go and WASM, partially inspired by another member's strategy on making on-device training possible. The work is in collaboration with *taz*, a key Chroma contributor, and the repository can be found on [GitHub - l4b4r4b4b4/go-chroma](https://github.com/l4b4r4b4b4/go-chroma).

**Link mentioned**: <a href="https://github.com/l4b4r4b4b4/go-chroma">GitHub - l4b4r4b4b4/go-chroma: Go port of Chroma vector storage</a>: Go port of Chroma vector storage. Contribute to l4b4r4b4b4/go-chroma development by creating an account on GitHub.

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1229349986358136843)** (322 messages🔥🔥): 

- **Unsloth Assists Data-Driven Chatbot Upgrade**: A member discusses formatting personal chat messages for training an AI clone of themselves via Unsloth and gets assistance including a Python script for converting the dataset into [ShareGPT format](https://github.com/mhagiwara/sharegpt#data-format), and advice on using this script with their personal data to create a training-ready dataset. They were also directed to a related [Unsloth notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing).
- **LoRA Tweaking for Enhanced Training**: Users report varying the alpha parameter during fine-tuning a model using LoRA. For rslora, it's suggested that alpha be double the rank value, though the exact optimal value may vary by case.
- **ORPO Support with Unsloth**: According to a member, ORPO, which optimizes the resources required for model training, is already supported within Unsloth. The ORPO method differs from DPO by not requiring a separate SFT (Supervised Fine-Tuning) step beforehand.
- **CUDA Learning and Triton's Promise**: A participant new to CUDA gets advice to lean on [Triton tutorials](https://www.youtube.com/watch?v=gyKBN1rnefI&list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&index=2) for learning and advised that Unsloth does not recommend native CUDA, instead suggesting Triton as more beneficial for LLM work.
- **Unsloth and Gemma**: For best results, don't use packing with Gemma models. The Unsloth library is compatible with packing for Llama and Mistral, and configurations with high-rank adapters may exhibit unexpected loss jump issues during training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/datasets">Find Open Datasets and Machine Learning Projects | Kaggle</a>: Download Open Datasets on 1000s of Projects &#x2B; Share Projects on One Platform. Explore Popular Topics Like Government, Sports, Medicine, Fintech, Food, More. Flexible Data Ingestion.</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1g9kHV3tc6P2cUp9gVPurKUZmiFqeb3kv">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=QtoqUw80QDV0)?">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1U_p7-qFfOm4v-TIrs1wK5eEODg1HUcGB?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/G-reen/EXPERIMENT-ORPO-m7b2-1-merged">G-reen/EXPERIMENT-ORPO-m7b2-1-merged · Hugging Face</a>: no description found</li><li><a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>: Things I Learned From Hundreds of Experiments</li><li><a href="https://unsloth.ai/blog/long-context">Unsloth - 4x longer context windows &amp; 1.7x larger batch sizes</a>: Unsloth now supports finetuning of LLMs with very long context windows, up to 228K (Hugging Face + Flash Attention 2 does 58K so 4x longer) on H100 and 56K (HF + FA2 does 14K) on RTX 4090.  We managed...</li><li><a href="https://www.youtube.com/watch?v=gyKBN1rnefI&list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&index=2">Intro to Triton: Coding Softmax in PyTorch</a>: Let&#39;s code Softmax in PyTorch eager and make sure we have a working version to compare our Triton Softmax version with. Next video - we&#39;ll code Softmax in Tr...</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices)">Home</a>: 2-5X faster 80% less memory LLM finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode">Installation</a>: no description found</li><li><a href="https://huggingface.co'">no title found</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode'.">Installation</a>: no description found</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://github.com/unslothai/unsloth/issues/331">Add ORPO example notebook to the docs · Issue #331 · unslothai/unsloth</a>: It&#39;s possible to use the ORPOTrainer from TRL with very little modification to the current DPO notebook. Since ORPO reduces the resources required for training chat models even further (no separat...</li><li><a href="https://arxiv.org/abs/2312.03732">A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA</a>: As large language models (LLMs) have become increasingly compute and memory intensive, parameter-efficient fine-tuning (PEFT) methods are now a common strategy to fine-tune LLMs. A popular PEFT method...</li><li><a href="https://huggingface.co/blog/damjan-k/rslora">Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main_classes/tokenizer">Tokenizer</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512">Adding New Vocabulary Tokens to the Models · Issue #1413 · huggingface/transformers</a>: ❓ Questions &amp; Help Hi, How could I extend the vocabulary of the pre-trained models, e.g. by adding new tokens to the lookup table? Any examples demonstrating this?
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1229454663875563621)** (47 messages🔥): 

- **Benchmark Buzz**: Members discussed a first benchmark's performance and shared encouraging remarks, such as proclaiming it looks "fantastic" and "pretty good."
- **Model Template Mysteries**: A user queried how a leaderboard discerns the model template, which was clarified by another member pointing to `tokenizer.chat_template`.
- **Squeaky Clean Data**: A dataset named **ShareGPT90k** was pushed in a cleaned and ChatML format, with HTML tags like `<div>` and `<p>` replaced with empty strings. The user urged others to use the `text` key when training with Unsloth AI.
- **Anticipating Ghost's Recipe**: Debates arose around the accessibility and vulnerability of training recipes for LLMs, particularly regarding a model named **Ghost**. Users discussed the importance of data quality over fine-tuning techniques and gave a nod to the importance of gaining experience through experimentation rather than relying on predefined recipes.
- **Sharing Academic Insights and Resources**: Conversations included sharing valuable resources like detailed research papers and YouTube tutorials explaining advanced AI concepts like Direct Preference Optimization. One user specifically looked forward to learning about the training approach behind the Ghost model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>: Things I Learned From Hundreds of Experiments</li><li><a href="https://tenor.com/view/sloth-crawling-slow-gif-9915689">Sloth Crawling GIF - Sloth Crawling Slow - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=hvGa5Mba4c8&t=5s">Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math</a>: In this video I will explain Direct Preference Optimization (DPO), an alignment technique for language models introduced in the paper &quot;Direct Preference Opti...</li><li><a href="https://www.youtube.com/watch?v=MJnIxpZhTk0).">FractalFormer: A WIP Transformer Architecture Inspired By Fractals</a>: Check out the GitHub repo herehttps://github.com/evintunador/FractalFormerSupport my learning journey on patreon!https://patreon.com/Tunadorable?utm_medium=u...</li><li><a href="https://arxiv.org/abs/2303.14617">Neural Graph Reasoning: Complex Logical Query Answering Meets Graph Databases</a>: Complex logical query answering (CLQA) is a recently emerged task of graph machine learning that goes beyond simple one-hop link prediction and solves a far more complex task of multi-hop logical reas...</li><li><a href="https://www.youtube.com/watch?v=wzKW4P4dg1o">LLM Phase Transition: New Discovery</a>: Phase Transitions in a dot-product Attention Layer learning, discovered by Swiss AI team. The study of phase transitions within the attention mechanisms of L...</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/16zuccy/after_500_loras_made_here_is_the_secret/).">After 500+ LoRAs made, here is the secret</a>: Well, you wanted it, here it is: The quality of dataset is 95% of everything. The rest 5% is not to ruin it with bad parameters. Yeah, I know,...</li><li><a href="https://huggingface.co/datasets/pacozaa/sharegpt90k-cleanned">pacozaa/sharegpt90k-cleanned · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1229388439300997180)** (60 messages🔥🔥): 

- **Kapa.AI Brings Instant Answers to Discord Communities**: [Kapa.AI](https://www.kapa.ai/) offers customizable AI bots that provide instant technical support in communities such as Slack and Discord. Their bots can be added to servers to improve developer experience by providing immediate answers and eliminating wait times, which is elaborated on their [community engagement use-case page](https://www.kapa.ai/use-cases/community-engagement) and their [Discord installation documentation](https://docs.kapa.ai/installation-discord).

- **Exploring Compilation Optimization with Mojo Language**: A discussion on using aliases for compile-time parameter decisions in Mojo language revealed that using such methods can lead to memory optimizations as unused aliases don't reserve memory space after compilation. For more nuanced opinions on code clarity versus comments, a [YouTube video](https://m.youtube.com/watch?v=Bf7vDBBOBUA) on the subject was shared.

- **Understanding Typestates and Memory Efficiency**: Conversing about the benefits of typestates over aliases, a member recommended a technique from Rust for making compile-time guarantees about object state, as described in an article about [Rust typestates](https://cliffle.com/blog/rust-typestate/). The discussion evolved around memory use optimizations in language design, specifically how boolean values are stored and addressed in memory.

- **Bit-Level Optimizations in Programming Languages Explored**: The chat shed light on why language specifications, like those in C, define enums as 32-bit integers and debated whether bools need to use a full byte. These topics touched on processor-level memory allocation and the potential efficiencies in memory usage at the language-level, referencing how boolean values could be represented more compactly as bits within a byte.

- **Rust's BitVec Crate Offers Both Speed and Memory Efficiency**: The discussion concluded by endorsing Rust's BitVec crate for being both speed and memory efficient when handling sets of boolean flags. An example cited was an optimization case which improved performance, going from taking years to just minutes, by using a bitset in Rust, as detailed on [willcrichton.net](https://willcrichton.net/notes/k-corrset/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://devlog.hexops.com/2022/packed-structs-in-zig/">Packed structs in Zig make bit/flag sets trivial</a>: As we've been building Mach engine, we've been using a neat little pattern in Zig that enables writing flag sets more nicely in Zig than in other languages. Here's a brief explainer.</li><li><a href="https://tenor.com/view/bamboozled-gif-25267741">Bamboozled GIF - Bamboozled - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://willcrichton.net/notes/k-corrset/">Analyzing Data 180,000x Faster with Rust</a>: How to hash, index, profile, multi-thread, and SIMD your way to incredible speeds.</li><li><a href="https://www.kapa.ai/">kapa.ai - Instant AI Answers to Technical Questions</a>: kapa.ai makes it easy for developer-facing companies to build LLM-powered support and onboarding bots for their community. Teams at OpenAI, Airbyte and NextJS use kapa to level up their developer expe...</li><li><a href="https://cliffle.com/blog/rust-typestate/">
The Typestate Pattern in Rust - Cliffle
</a>: no description found</li><li><a href="https://www.kapa.ai/use-cases/community-engagement">kapa.ai - ChatGPT for your developer-facing product</a>: kapa.ai makes it easy for developer-facing companies to build LLM-powered support and onboarding bots for their community. Teams at OpenAI, Airbyte and NextJS use kapa to level up their developer expe...</li><li><a href="https://docs.kapa.ai/installation-discord">Discord Bot | kapa.ai docs</a>: Kapa can be installed as a bot on your Discord server. The bot allows users to ask questions in natural language about your product which improves developer experience as developers can find answers t...</li><li><a href="https://m.youtube.com/watch?v=Bf7vDBBOBUA&t=0s">Don&#39;t Write Comments</a>: Why you shouldn&#39;t write comments in your code (write documentation)Access to code examples, discord, song names and more at https://www.patreon.com/codeaesth...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1229474282732785747)** (5 messages): 

- **Modular Tweets a Mystery**: Modular shared a [cryptic message](https://twitter.com/Modular/status/1779913837216719118) on Twitter, enticing curiosity without context.
- **The Plot Thickens with Another Tweet**: Shortly after, Modular posted another [enigmatic tweet](https://twitter.com/Modular/status/1779913865914134561), maintaining the suspense.
- **Three's a Charm for Modular's Teasers**: Continuing the trend, Modular released a third [tweet](https://twitter.com/Modular/status/1779913874957086978), adding more intrigue.
- **Modular's Tweet Streak Unbroken**: The string of mysterious messages from Modular extended with a fourth [tweet](https://twitter.com/Modular/status/1779913908649914783).
- **Fifth Tweet Keeps the Mystery Alive**: Modular capped off the series with a fifth [mysterious tweet](https://twitter.com/Modular/status/1779913912009597323), leaving followers in anticipation.
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1229796285352247366)** (2 messages): 

- **Mojo Replication Buzz**: A member expressed interest in replicating an unidentified feature or project within **Modular (Mojo)**, signaling enthusiasm for the platform's capabilities.
- **Unlocking AI Agents' True Potential**: A member shared a [YouTube video](https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek) explaining the creation of **long-term memory and self-improving AI agents** in a concise 10-minute presentation, potentially offering valuable insights for fellow enthusiasts.

**Link mentioned**: <a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1229364842327052368)** (541 messages🔥🔥🔥): 

- **Mojo Adaptation of Python Tools**: A new [Python package called mojo2py](https://github.com/venvis/mojo2py) has been developed to convert Mojo code to Python code, indicating a growing interest in tools that bridge the gap between Python and Mojo. The repository is available on GitHub.
- **Learning Mojo Essentials**: For those looking to learn Mojo from scratch, the [Mojo Programming Manual](https://docs.modular.com/mojo/manual/) is the go-to comprehensive guide, with emphasis on core concepts such as parameters versus arguments and understanding traits.
- **Introducing Conditional Conformance**: There is an ongoing discussion about the potential for *conditional conformance* in Mojo, allowing for behaviors like structural patterns found in other languages such as C++ or Haskell, though there may be challenges when it comes to generic code.
- **Variant Type for Runtime Flexibility**: The usage of `Variant` as a way to create a list containing multiple types akin to Python is validated, with the point made that `Variant` acts more like a tagged union at runtime, akin to ADTs in TypeScript.
- **Efforts on Syntaxes and Representation**: Multiple members are actively discussing possible improvements to Mojo's syntax for function signatures and type representation, including the use of `'1` numerals to avoid verbose naming in partially-bound types, and Treesitter grammar/LSP development for broader integrations beyond VS Code.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols#Conditionally-Conforming-to-a-Protocol">Documentation</a>: no description found</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/generics#Extensions-with-a-Generic-Where-Clause">Documentation</a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/">Mojo Manual | Modular Docs</a>: A comprehensive guide to the Mojo programming language.</li><li><a href="https://tenor.com/view/correct-plankton-gif-14118231">Correct Plankton GIF - Correct Plankton - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.modular.com/mojo/lib">Mojo🔥 modules | Modular Docs</a>: A list of all modules in the Mojo standard library.</li><li><a href="https://en.m.wikipedia.org/wiki/Einstein_notation">Einstein notation - Wikipedia</a>: no description found</li><li><a href="https://doc.rust-lang.org/std/primitive.tuple.html">tuple - Rust</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/f1493c87ff8cbabb3cb88fb11fd1063403a7ffe2/examples/matmul.mojo">mojo/examples/matmul.mojo at f1493c87ff8cbabb3cb88fb11fd1063403a7ffe2 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/f1493c87ff8cbabb3cb88fb11fd1063403a7ffe2/examples/matmul.mojo#L136">mojo/examples/matmul.mojo at f1493c87ff8cbabb3cb88fb11fd1063403a7ffe2 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/venvis/mojo2py">GitHub - venvis/mojo2py: A python package to convert mojo code into python code</a>: A python package to convert mojo code into python code - venvis/mojo2py</li><li><a href="https://github.com/modularml/mojo/issues/2308">Conditional Trait Conformance · Issue #2308 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I believe it would be nice for there to be conditional...</li><li><a href="https://github.com/Moosems/TkLineNums/blob/main/tklinenums/tklinenums.py">TkLineNums/tklinenums/tklinenums.py at main · Moosems/TkLineNums</a>: A simple line numbering widget for tkinter. Contribute to Moosems/TkLineNums development by creating an account on GitHub.</li><li><a href="https://github.com/google/jax/blob/f8919a32e02841c9d5a202398e16357c7506b102/jax/_src/interpreters/partial_eval.py#L569">jax/jax/_src/interpreters/partial_eval.py at f8919a32e02841c9d5a202398e16357c7506b102 · google/jax</a>: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - google/jax</li><li><a href="https://github.com/modularml/mojo/issues/2308#issuecomment-2057772648">Conditional Trait Conformance · Issue #2308 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I believe it would be nice for there to be conditional...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1229449642908061830)** (4 messages): 

- **Mojo meets gRPC**: A community member is working on integrating functional Mojo code with legacy C++ via **IPC** to improve product performance.

- **Hunting for the Updated Llama**: A member inquires about an official version of **Llama2** and shares a [guest blog post by Aydyn Tairov](https://www.modular.com/blog/community-spotlight-how-i-built-llama2-by-aydyn-tairov) about building the project. They have attempted to update the project to v24.2.x and provided a [link](https://github.com/tairov/llama2.mojo/pull/89) to their work-in-progress on GitHub.

- **Llama2 Gets Official MAX API**: In response to a query about Llama2, another member directs to the official **llama2 in MAX graph API** available on [GitHub](https://github.com/modularml/max/tree/main/examples/graph-api/llama2).

- **Mojo Code - Python Transformation Tool**: A new Python package called **mojo2py** has been created by a community member to convert Mojo code into Python code, with the repository available [here](https://github.com/venvis/mojo2py).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/venvis/mojo2py">GitHub - venvis/mojo2py: A python package to convert mojo code into python code</a>: A python package to convert mojo code into python code - venvis/mojo2py</li><li><a href="https://github.com/modularml/max/tree/main/examples/graph-api/llama2">max/examples/graph-api/llama2 at main · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform - modularml/max</li><li><a href="https://www.modular.com/blog/community-spotlight-how-i-built-llama2-by-aydyn-tairov">Modular: Community Spotlight: How I built llama2.🔥 by Aydyn Tairov</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Community Spotlight: How I built llama2.🔥 by Aydyn Tairov</li><li><a href="https://github.com/tairov/llama2.mojo/pull/89">Update to 24.2 WIP by anthony-sarkis · Pull Request #89 · tairov/llama2.mojo</a>: Related to #88 Work in progress, still getting these errors. I&#39;m new to Mojo so if anyone can assist: /root/llama2.mojo/llama2.mojo:506:20: error: &#39;String&#39; value has no attribute &#39;bitc...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1229881110218145864)** (12 messages🔥): 

- **Jank Over Traits**: The discussion humorously criticized the lack of **trait parameterization** with a preference for what's referred to as *jank*.
- **Indentation Anarchy**: One user lamented the lack of proper indentation in for loops, which prompted a mix of laughter and light-hearted agreement over the importance of *indenting code*.
- **Leveling Up in Modular**: <@244534125095157760> was congratulated for reaching **level 9** in what appears to be a gamified system within the Modular community.
- **Code Formatting Peer Pressure**: There's a jest about giving in to peer pressure regarding code formatting, indicating a light-hearted conversation about personal coding styles within the community.
- **Mojo Nightly Update Announced**: A new **Mojo** update called `nightly/mojo` has been released, with members encouraged to update and check the diff on [GitHub](https://github.com/modularml/mojo/pull/2313/files) and the changelog on their [changelog page](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/pull/2313/files">[stdlib] Update stdlib corresponding to `2024-04-16` nightly/mojo by patrickdoc · Pull Request #2313 · modularml/mojo</a>: This updates the stdlib with the internal commits corresponding to today&#39;s nightly release: mojo 2024.4.1618 . In the future, we may push these updates directly to nightly branch.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1229357501644804106)** (549 messages🔥🔥🔥): 

- **Confusion Over Subscriptions and Payment Methods**: Members reported instances of unexpected charges despite using promo codes and concerns about managing payment methods for Perplexity API. Promises for improvements to show main payment methods on the Perplexity website were mentioned by a user presumed to be a team member.

- **Perplexity Pro Message Counter Disappears**: Users have noticed that the message counter, indicating the number of messages left for Perplexity Pro users, is not being displayed unless under 100 messages are left. A member with a presumed team role confirmed this change and cited user reports of reduced stress as the reason, while others expressed dissatisfaction with the removal.

- **Perplexity Performance and Model Updates**: Some users express that Perplexity seems to be forgetting the context more quickly than before and question the reasoning behind changes to features such as the Pro message counter. Inquiries about the integration of GPT-4-Turbo-2024-04-09 into Perplexity were met with referenced previous statements, suggesting members refer to discussions in the official channel.

- **AI Models and Coding**: There's a consensus among users that AI models are still lacking in coding capabilities, with GPT-4 mentioned as underperforming despite being better than its predecessors. Perplexity's diverse model offerings are acknowledged, but some feel that the original models are superior in performance.

- **Payment and Subscription Issues Discussed**: Members discussed issues with promotional codes and billing confusion. One user pointed out a lack of clarity in managing payment methods for the Perplexity API, while others voiced concerns about not seeing promo code options upon checkout. A response from user ok.alex indicated upcoming improvements to the payment method visibility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.together.xyz/playground/chat/microsoft/WizardLM-2-8x22B">no title found</a>: no description found</li><li><a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>: no description found</li><li><a href="https://www.raycast.com/ilian/perplexity-api">Raycast - Perplexity API</a>: Use the powerful models via Perplexity API from the comfort of Raycast.</li><li><a href="https://tenor.com/view/hamak-chilling-beach-summer-vacation-gif-17726234">Hamak Chilling GIF - Hamak Chilling Beach - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/no-the-office-michael-scott-scream-gif-16929305">No The Office GIF - No The Office Michael Scott - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/chopping-garlic-chopping-knife-skills-chopping-skills-food52-gif-14523391">Chopping Garlic Chopping GIF - Chopping Garlic Chopping Knife Skills - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/OpenAIDevs/status/1779922566091522492">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Introducing the Batch API: save costs and get higher rate limits on async tasks (such as summarization, translation, and image classification).  Just upload a file of bulk requests, receive results wi...</li><li><a href="https://tenor.com/view/how-soon-is-now-smiths-morrissey-80s-music-new-wave-gif-17919265">How Soon Is Now Smiths GIF - How Soon Is Now Smiths Morrissey - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://vm.tiktok.com/ZGeH84n4s/">TikTok - Make Your Day</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1229435097074372608)** (12 messages🔥): 

- **Exploring Perplexity AI**: Members shared various searches on [Perplexity AI](https://www.perplexity.ai), exploring topics such as **Microsoft testing ads**, *World Art Day*, and methods to *act as an Ichjogu*.
- **Celebrating Days of Art and Voice**: Searches were conducted related to *World Art Day* and *World Voice Day*, highlighting community interest in global cultural observations.
- **Inspecting Tech and Games**: Discussions included searches on **Microsoft's test ads**, **Atari**, and **Amazon Web Services hardening guide**.
- **Music and Lyrics Searches**: Members showed an inclination towards music by searching for lyrics to the song "SBK Borderline" and the phrase "*Whatever It Is*".
- **Curiosities in Costs and Queries**: The community delved into diverse inquiries, from asking about the cost of certain items to requesting explanations in Portuguese.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1229561836924571762)** (3 messages): 

- **Inconsistency in Perplexity Pro vs. API Answers**: A member expressed difficulty with getting different responses when using **Perplexity Pro** compared to the API. There is a desire to understand settings like the **temperature** that the web client uses to try and match API results for consistency.
- **Constraints on API Source Material**: Another question was raised about the possibility of limiting API responses to content from specific websites. There is concern about responses containing possible hallucinations and incorrect source attributions.
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1229364772919836695)** (210 messages🔥🔥): 

- **Collaborative Agents on LM Studio**: Users are excited about agents working together in LM Studio and are anticipating a **Windows version**. An automated compilation process is mentioned as a necessary step before releasing the Windows version.

- **User Queries on Model Performance and Use**: There are questions regarding **model performance**, specifically related to coding capabilities of models like **Mistral**. Another user inquired about **multi-GPU support without NVLink** for models in LM Studio.

- **WizardLM-2 Model Integration**: Users discussed integrating **WizardLM-2** models into LM Studio and shared community model links such as [MaziyarPanahi/WizardLM-2-7B-GGUF on Hugging Face](https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF). The specifics of model naming and partitioning for **Mixtral 8x22B** were examined, as well as system requirements like **VRAM** for effective model usage.

- **Interest in AI Model Tuning and Agents**: A discussion took place about the possibility of using **LM Studio** to create personal AI agents or assistants and the use of external tools for fine-tuning models to specific tasks. Resources for **fine-tuning**, like the [LLaMA-Factory on GitHub](https://github.com/hiyouga/LLaMA-Factory), and agent creation tools were shared.

- **Mistaken Code Recognition and Dataset Tools**: Users shared experiences with models incorrectly identifying programming languages, and techniques like adding context to prompts were suggested. Additionally, links for toolsets like [Unstructured-IO for dataset creation](https://github.com/Unstructured-IO/unstructured) were provided for those looking to build custom **preprocessing pipelines**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notkyon.moe/ram-latency2.htm">RAM Latency Calculator</a>: no description found</li><li><a href="https://x.com/WizardLM_AI/status/1779899325868589372">Tweet from WizardLM (@WizardLM_AI)</a>: 🔥Today we are announcing WizardLM-2, our next generation state-of-the-art LLM.  New family includes three cutting-edge models: WizardLM-2 8x22B, 70B, and 7B - demonstrates highly competitive performa...</li><li><a href="https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF">lmstudio-community/WizardLM-2-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://missionsquad.ai">Mission Squad. Flexible AI agent desktop app.</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/qnamaker/how-to/multi-turn#what-is-a-multi-turn-conversation">Multi-turn conversations - QnA Maker - Azure AI services</a>: Use prompts and context to manage the multiple turns, known as multi-turn, for your bot from one question to another. Multi-turn is the ability to have a back-and-forth conversation where the previous...</li><li><a href="https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF">MaziyarPanahi/WizardLM-2-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/hiyouga/LLaMA-Factory">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://github.com/Unstructured-IO/unstructured/">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.  - GitHub - Unstructured-IO/unstructured: Open source librar...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1229355091753177209)** (108 messages🔥🔥): 

- **Python Coding Models Highlighted**: Members discussed various **Python coding models** such as *Deepseek Coder*, *Wizzard Coder*, and *Aixcoder*, with recommendations to **check the 'human eval' score** for assessing model performance on coding challenges. Some models, like Aixcoder, are noted for writing code directly, while others also provide conversational interaction.

- **Skepticism and Praise for WaveCoder**: WaveCoder Ultra, a model based on **DeepseekCoder 6.7B**, receives mixed feedback; some praise its performance, while others show skepticism about its superiority and hint at possible "faked results." It’s mentioned that [Microsoft has released three WaveCoder models with varying performances](https://huggingface.co/microsoft/wavecoder-ultra-6.7b), with WaveCoder Ultra touted for its performance on coding the snake game.

- **Questions on Vision Models and Java Coders**: Discussion on the LM Studio Discord includes inquiries about good models for **Swift/SwiftUI development** and Java coders. Responders suggest that most coder models are generally trained on languages rather than tailored specifically to one.

- **Contours of Model Performance**: Debates center on the tradeoffs between model size, quantization, and performance, with opinions varying on the effectiveness of **7B models** versus more aggressively quantized larger models. While some users prefer the Q8 quant for 7B models, others argue that lower-quality quants of larger models may not retain higher intelligence.

- **Text-to-Image Generation Limits and Preferences**: A member clarifies that **text-to-image generation** is not a task that LM Studio can do. Suggestions for alternative tools include GUI: **DiffusionBee** and using **Automatic1111** for text-to-image tasks outside of LM Studio.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF">lmstudio-community/WizardLM-2-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/wavecoder-ultra-6.7b">microsoft/wavecoder-ultra-6.7b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF#prompt-template>">lmstudio-community/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MaziyarPanahi/WizardLM-2-8x22B-GGUF">MaziyarPanahi/WizardLM-2-8x22B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/DavidAU">High Quality / Hard to Find - a DavidAU Collection</a>: no description found</li><li><a href="https://huggingface.co/bartowski/zephyr-orpo-141b-A35b-v0.1-GGUF/tree/main">bartowski/zephyr-orpo-141b-A35b-v0.1-GGUF at main</a>: no description found</li><li><a href="https://rentry.co/4q4h7pw6">Responses</a>: These are answers to the prompt by two different LLMs. You are going to analyze  Factuality Depth Level of detail Coherency &lt;any other area that I might have missed but is generally considered impo...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1229462981604020266)** (15 messages🔥): 

- **Error Plague Hits LM Studio**: A member grappled with an error when trying to load a model in LM Studio, even after attempts to uninstall and revert to default settings. They shared a [code snippet of the error message](#), but the cause remained undisclosed with an "Unknown error" description and an exit code: 42.

- **GPU Offload: A Double-Edged Sword?**: One member advised turning off the GPU Offload to circumnavigate loading issues, but the original poster faced dilemmas about performance degradation and continued error messages, even after disabling it. 

- **Fresh Start for Frustrated User**: In light of persisting issues with model loading, a solution was proposed to perform a full reset of LM Studio by deleting specific directories on the user's machine. The instructions included paths such as C:\Users\Username\.cache\lm-studio and others, with a reminder to back up important data.

- **NexusRaven Prompting Queries Emerge**: Another member jumped into the conversation with a query about prompt presets for NexusRaven, indicating a shift in topic towards AI model customization. 

- **From Partial to Full Scripts**: A succinct request was made for assistance in compelling NexusRaven to write complete scripts, hinting at challenges with the model outputting partial content.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1229394950198923334)** (21 messages🔥): 

- **VRAM Misidentifications Confuse Users**: A member encountered a strange error pointing towards AMD hardware on an Intel-based Linux system and discussed potential issues with libOpenCL. They had hoped that installing `ocl-icd-opencl-dev` would emulate a GPU, but continued to face loading failures even after altering GPU layers.
- **Link Lost in Chat**: Members struggled to locate a previously mentioned Google sheet for GPU comparisons, and despite posting links, the exact resource in question remains unfound.
- **Subreddit for Meta's Llama Model Discussed**: A user shared a Reddit [link](https://www.reddit.com/r/LocalLLaMA/comments/1c4gakl/got_p2p_working_with_4x_3090s) that discusses achieving peer-to-peer communication with GPUs, theorizing potential benefits for bypassing CPU/RAM and enhancing performance.
- **Memes Add Light-heartedness to Tech Talk**: In the midst of technical discussion, a member shared a humorous George Hotz GIF from Tenor, possibly reflecting their feelings on the ongoing software wrestle or symbolizing a breakthrough.
- **Hardware Allocation Challenges for Dual GPUs**: One member is seeking advice on managing uneven model distribution between a `4070 TI 12GB` and a `4060 TI 16GB` in their system to favor the larger GPU.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/george-hotz-geohot-money-rain-gif-6469921471081342358">George Hotz Geohot GIF - George hotz Geohot Money - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c4gakl/got_p2p_working_with_4x_3090s">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1229813824778014902)** (26 messages🔥): 

- **VRAM vs. RAM for Running Models**: A member questioned if models could run on a system with 24 GB VRAM and 96 GB RAM, and there's concern that inference might be *incredibly slow*. Another found success with an M3 MacBook Pro 128GB, running models at speeds comparable to GPT-4 using LMStudio and MLX, achieving up to 10 tokens/sec.

- **Performance of Command R Plus and WizardLM-2-8x22B**: A user reported they will test **WizardLM-2-8x22B** and share results. Another user was *not impressed* with **Mixtral 8x22b** and pondered whether its base model status affected performance.

- **Understanding Base Models**: Users discussed what constitutes a "Base" model, clarifying that it's one that hasn't been fine-tuned for specific tasks like chat or instruct. There was a mention that any base model could potentially be prompted to perform various tasks.

- **Model Comparison and Fine-Tuning Considerations**: A member reported getting over 10 tokens/sec on both **MaziyarPanahi/WizardLM-2-8x22B.Q4_K_M** and **Command R Plus**, with both prompting the model to write a simple Rust application successfully. Discussions also covered whether models continuously learn from chat and the concept of "forgetting" certain knowledge areas during fine-tuning.

- **Potential Implicit Bias in AI Models**: It was suggested that AI models might be biased towards math, IT, AI, and computer vision, reflecting the interests of the developers and a majority of users. Queries were raised on whether models are always learning from interactions and the simplicity of fine-tuning models to *forget* unwanted knowledge domains.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1229814103292383302)** (8 messages🔥): 

- **Signed, Sealed, Delivered, It's Windows!**: A member inquired about whether the Windows executables are **signed with an authenticode certificate**. The response confirmed that they are indeed signed.

- **Certificate Signing Curiosity**: The same member expressed curiosity regarding the signing process, mentioning they didn't recall any notifications about it since Windows generally doesn't notify users once an application is signed.

- **Windows vs. Apple Developer Licenses – The Cost of Security**: The member voiced their frustration over the costliness of acquiring a Windows certificate compared to an Apple developer license, highlighting the added financial burden due to **the requirement for a hardware security module (HSM)**.

- **Seeking Knowledge on Compile and Signing Processes**: The member sought advice on automating the compile and signing process for their app and expressed willingness to offer something in exchange for this shared expertise.
  

---


**LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/)** (1 messages): 

rouw3n: Anyone here got the 01light software running on windows without problems ?
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1229590866444947476)** (2 messages): 

- **WizardLM 2 7B Shines in Multi-Turn Conversations**: A new model, *WizardLM 2 7B*, has been highlighted for its excellent performance in multi-turn conversations. Available on [Hugging Face](https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF), it employs novel training techniques as detailed in the associated blog post and mentioned at the end of the model card.

- **WaveCoder ultra 6.7b Fine-Tuned with CodeOcean**: Microsoft's recent release, *WaveCoder ultra 6.7b*, is recognized for its code translation capabilities and is fine-tuned with their 'CodeOcean' platform, combining open-source code and models like GPT-3.5-turbo and GPT-4. The model can be explored and downloaded on [Hugging Face](https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF), and it follows the Alpaca format for instruction following.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF">lmstudio-community/WizardLM-2-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF">lmstudio-community/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1229361305488588851)** (18 messages🔥): 

- **Quantization-Friendly AI Model Unveiled**: AI practitioner, carsonpoole, highlighted a **binary quantization-friendly** AI model on Hugging Face with the intention of making embeddings more memory-efficient. The approach is detailed in [this cohere blog post](https://txt.cohere.com/int8-binary-embeddings/), which discusses reducing memory costs significantly by using **int8** and **binary embeddings**.

- **Innovative Model Training Technique Explained**: Further explaining the technique, carsonpoole emphasized that contrastive loss and the sign of model output are used, acting as the model's training mechanism. The method aims to maintain high search quality with the model performing well even with compressed embedding formats.

- **Cohere Embedders Training Available Through API**: carsonpoole confirmed that while **Cohere's embedders** are currently not accessible outside of their API, the newly trained models serve to enable similar functionalities.

- **Understanding Binary Embedding Distances**: In response to sumo43's query, carsonpoole clarified that **embedding distance** in binary cases is calculated using an **XOR operation**, equating to hamming distance.

- **Showcase of Multimodal LLM and Cost-Efficient Model**: pradeep1148 shared YouTube videos introducing "Idefics2 8B: Open Multimodal ChatGPT" and "JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars," showcasing advancements in the cost-effective development and capabilities of **language models**. Links to videos: ["Introducing Idefics2 8B"](https://www.youtube.com/watch?v=vL1SayPCHBg), ["Reka Core: A Frontier Class Multimodal Language Model"](https://www.youtube.com/watch?v=U7RbwPKyxs8), and ["JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars"](https://www.youtube.com/watch?v=Z9Hwp_XeS1A).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/carsonpoole/binary-embeddings">carsonpoole/binary-embeddings · Hugging Face</a>: no description found</li><li><a href="https://txt.cohere.com/int8-binary-embeddings/">Cohere int8 &amp; binary Embeddings - Scale Your Vector Database to Large Datasets</a>: Cohere Embed now natively supports int8 and binary embeddings to reduce memory cost.</li><li><a href="https://www.youtube.com/watch?v=Z9Hwp_XeS1A">JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>: JetMoE-8B is trained with less than $ 0.1 million1 cost but outperforms LLaMA2-7B from Meta AI, who has multi-billion-dollar training resources. LLM training...</li><li><a href="https://www.youtube.com/watch?v=vL1SayPCHBg">Introducing Idefics2 8B: Open Multimodal ChatGPT</a>: We will take a look idefics2 the open multimodal llm by huggingfacehttps://huggingface.co/blog/idefics2#python #pythonprogramming #llm #ml #ai #aritificialin...</li><li><a href="https://www.youtube.com/watch?v=U7RbwPKyxs8">Reka Core: A Frontier Class Multimodal Language Model</a>: Reka Core is competitive with models from OpenAI, Anthropic, and Google across key industry-accepted evaluation metrics. Given its footprint and performance,...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1229423070154981467)** (4 messages): 

- **Auto-Code Rover Making Moves**: The [Auto-Code Rover](https://github.com/nus-apr/auto-code-rover) by NUS is an autonomous software engineer that's *project structure aware* and aims for autonomous program improvement. It reportedly resolved **15.95%** of tasks in the full SWE-bench.

- **Auto-Code Rover Outperforms Devin**: Mention of the Auto-Code Rover suggests it performs better than another AI, Devin, in software engineering tasks, and by a *decent margin*.

- **Google's CodecLM Framework Revealed**: Google AI introduces [CodecLM](https://arxiv.org/abs/2404.05875), a machine learning framework for generating high-quality synthetic data for language model alignment, discussed in a [MarkTechPost article](https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-for-generating-high-quality-synthetic-data-for-llm-alignment/?amp).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-for-generating-high-quality-synthetic-data-for-llm-alignment/?amp">no title found</a>: no description found</li><li><a href="https://github.com/nus-apr/auto-code-rover">GitHub - nus-apr/auto-code-rover: A project structure aware autonomous software engineer aiming for autonomous program improvement. Resolved 15.95% tasks in full SWE-bench</a>: A project structure aware autonomous software engineer aiming for autonomous program improvement. Resolved 15.95% tasks in full SWE-bench - nus-apr/auto-code-rover
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1229367939699048499)** (208 messages🔥🔥): 

- **No Affiliation with Nous Team**: A token launched with OpenML is not affiliated with the Nous team despite using their name, and the team has requested that their name be removed from the project.
- **Mysterious Take-down of WizardLM-2**: The WizardLM-2 model was taken down unexpectedly, with suggestions ranging from it being too toxic and violating the EU AI act to missing evaluations. Links to download preserved versions of the model weights were shared, hinting at the community's rush to secure their own copies.
- **Mistral Instruct Shows Impressive Progress**: A member reported progress in fine-tuning the Mistral instruct v0.2, with the score improving from 58.69 to 67.59, surpassing many competitors but still behind others.
- **Qwen Releases Code-specific Model**: Qwen introduced CodeQwen1.5, a strong code-generation model for 92 coding languages and boasting high scores on benchmarks, including humaneval. A 7B version is available, and references were made to even larger 14B and 32B variants.
- **AI Video on Long Term Memory and Self-Improvement**: A video discussing long-term memory and self-improving AI agents with auto-generated teachability was shared with the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-fo">no title found</a>: no description found</li><li><a href="https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-for-generating-high-quality-synthetic-data-for-llm-alignment/?amp">no title found</a>: no description found</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat">Qwen/CodeQwen1.5-7B-Chat · Hugging Face</a>: no description found</li><li><a href="https://www.ora.io/app/imo/olm">ORA</a>: no description found</li><li><a href="https://huggingface.co/amazingvince/Not-WizardLM-2-7B">amazingvince/Not-WizardLM-2-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/alp">alp (Alp Öktem)</a>: no description found</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...</li><li><a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...</li><li><a href="https://huggingface.co/datasets/N8Programs/CreativeGPT">N8Programs/CreativeGPT · Datasets at Hugging Face</a>: no description found</li><li><a href="https://linktones.vercel.app/linktone/2eef9741-ecbb-4c73-9578-a3a2860d6843">SynthTrails</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1229381937928601662)** (39 messages🔥): 

- **Hermes 2.5 Inference Issues**:
  A user highlighted problems with **[llama.cpp](https://docs.rs/llama_cpp/0.3.1/llama_cpp/index.html)** and the OpenHermes 2.5 Mistral 7B model from [Hugging Face](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF), facing difficulties with tokenized outputs for end-of-sequence predictions in Rust. They seek advice on configuration or model issues causing this.

- **Reka AI Showcases Its Model**: In a shared [showcase](https://showcase.reka.ai/), Reka AI's Core model is presented as competitive with OpenAI's and others, with highlights stating it is on par with GPT-4V, Claude-3 Opus, and Gemini Ultra for various tasks. A subsequent link to [Reka's news announcement](https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model) elaborates on the Core's training and efficiency.

- **Hermes 2 Pro's Extended Context Quirks**:
  OctoAI reports odd behavior with Hermes 2 Pro Mistral 7b when processing requests with a context longer than 4k, suspecting the issue relates to the model's sliding window functionality and seeking experience sharing about using this model with long contexts.

- **Citation Mechanisms in RAGs Lacking Uniformity**: A member suggests that in Retriever-Augmented Generation models (RAGs), citation functionality typically depends on the system designer, pointing to Cohere's approach as an example but noting a lack of models generating their own source IDs.

- **Fast Model Downloading from Hugging Face**:
  Users discussed ways to speed up downloads from Hugging Face, with one offering the `HF_HUB_ENABLE_HF_TRANSFER=1` environment variable and pointing to Hugging Face's [guide](https://github.com/huggingface/huggingface_hub/blob/03469442f91a00ba466257f756a480a5b0ff6ccf/docs/source/en/guides/download.md#faster-downloads) for faster downloads using the Rust-based [hf_transfer](https://github.com/huggingface/hf_transfer/tree/main) library.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html">Answer.AI - You can now train a 70b language model at home</a>: We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.</li><li><a href="https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model">Reka Core: Our Frontier Class Multimodal Language Model &mdash; Reka AI</a>: Launching Reka Core, our frontier-class multimodal language model!</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c4mgda/inference_issue_using_llamacpp_and_openhermes/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py">scratchTHOUGHTS/commanDUH.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 2nd brain scratchmemory to avoid overrun errors with self. - EveryOneIsGross/scratchTHOUGHTS</li><li><a href="https://showcase.reka.ai/">Reka Core showcase</a>: Qualitative examples showcasing responses from Reka core along side other major models</li><li><a href="https://github.com/huggingface/huggingface_hub/blob/03469442f91a00ba466257f756a480a5b0ff6ccf/docs/source/en/guides/download.md#faster-downloads).">huggingface_hub/docs/source/en/guides/download.md at 03469442f91a00ba466257f756a480a5b0ff6ccf · huggingface/huggingface_hub</a>: The official Python client for the Huggingface Hub. - huggingface/huggingface_hub
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1229377654613082173)** (9 messages🔥): 

- **Discovering the RAG/Long Context Reasoning Dataset**: A link to a **Google Document** was shared, purportedly related to the RAG/Long Context Reasoning Dataset, although the content of the document wasn't visible due to the browser version being unsupported. Users are recommended to upgrade their browser based on the [Google Support Page](https://support.google.com/docs/answer/2375082?hl=en).
- **Defining the Limits of Long Context**: A member inquired about the limit of long context in relation to the number of documents or datasets like Wikipedia, though no specific answer was provided within the discussion.
- **Introduction of Cohere Compass**: Compass, a new foundation embedding model, was introduced and is distinguished by its ability to index and search against nested json relationships. The [Cohere Compass beta](https://txt.cohere.com/compass-beta/) offers a novel approach to handling multi-aspect data commonly found in enterprise information.
- **Considerations for Structured Data Inputs**: In relation to the discussion about **Cohere Compass**, a member noted that having json structure input outputs for data like what Compass offers would be beneficial, indicating less complexity in implementation.
- **Inquiry and Response about State-of-the-Art Vision Models**: A user asked about the best open-source vision models for a RAG with many images and diagrams related to engineering; another user responded with suggestions like **GPT4v/Geminipro Vision** and **Claude Sonnet** but specified that there is a need to test to see which performs best for certain cases. Open-source options, however, were not specified in the given excerpt.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://txt.cohere.com/compass-beta/">Cohere Compass Private Beta: A New Multi-Aspect Embedding Model</a>: Today, we are excited to announce the private beta for Cohere Compass, our new foundation embedding model that allows indexing and searching on multi-aspect data.  Multi-aspect data can best be explai...</li><li><a href="https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit">RAG/Long Context Reasoning Dataset</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1229379425628590093)** (87 messages🔥🔥): 

- **WorldSim as a Divine Sandbox**: Messages in the channel reflect an excitement and anticipation for the inventive potential of **WorldSim**, with comparisons made to the omnipotence of gods and the introduction of dynamic, unexpected creativity similar to quantum probability.

- **Game Development Revolutions on the Horizon**: Discussions consider the impact of LLMs on future game design, speculating on spell creation systems that could affect in-game physics and procedurally generated content that could significantly enhance narrative and environmental complexity.

- **Eagerly Awaiting WorldSim's Return**: The chat contains several expressions of eagerness for the return of **WorldSim**, with members discussing the possibilities of the project, sharing their hopeful plans for its relaunch, and wondering about the exact time for the comeback announcement.

- **Websim's "Jailbroken Prometheus" and The Aura of Mystery**: A link to a **Websim** featuring a "Jailbroken Prometheus" was shared, sparking discussions about the virtual entity's updated abilities and its interactions with users, ranging from engaging discussions to refraining from certain joke topics.

- **Artistic Inspirations from AI Interaction**: Participants share personal anecdotes about their creative pursuits inspired by AI platforms, like **WorldSim** and **Websim**, emphasizing AI's role in creative thinking and expression, and extending an invitation to the community to share their AI-inspired art.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>: no description found</li><li><a href="https://tenor.com/view/interstellar-cost-little-maneuver-51years-51-gif-24426899">Interstellar Cost GIF - Interstellar Cost Little Maneuver - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/noita-game-homing-death-gif-27319696">Noita Game GIF - Noita Game Homing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/noita-explosion-electricity-boom-wand-gif-19437628">Noita Explosion GIF - Noita Explosion Electricity - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/poe-path-of-exile-login-play-poe-login-gif-26508840">Poe Path Of Exile GIF - Poe Path Of Exile Login - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/youre-not-gonna-like-this-jerrod-carmichael-saturday-night-live-you-wont-enjoy-this-this-wont-be-ideal-gif-25522925">Youre Not Gonna Like This Jerrod Carmichael GIF - Youre Not Gonna Like This Jerrod Carmichael Saturday Night Live - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://websim.ai/c/BZcLXGB6Ft5cjnLns">Jailbroken Prometheus Chat</a>: no description found
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1229540108378574979)** (15 messages🔥): 

- **WizardLM-2 Series Now on OpenRouter**: The **[WizardLM-2 8x22B](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b)** and **[WizardLM-2 7B](https://openrouter.ai/models/microsoft/wizardlm-2-7b)** models are now available on OpenRouter. The 8x22B model's cost has been reduced to $0.65/M tokens, with discussion ongoing in a specific thread.
  
- **Latency Issues Under the Microscope**: High latencies were reported for **Mistral 7B Instruct** and **Mistral 8x7B Instruct**. An [update was provided](https://discord.com/channels/1091220969173028894/1229813179681345556) implicating an upstream issue with one of the cloud providers, which was resolved after identifying overly-aggressive DDoS protection.

- **Potential Latency Woes Return**: The latency problem seemed to reoccur, potentially impacting **Nous Capybara 34b** and other traffic. The issue is under investigation, and one cloud provider is being deranked to mitigate the issue. Users from different world regions are encouraged to report if affected.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b)">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...</li><li><a href="https://openrouter.ai/playground?models=microsoft/wizardlm-2-8x22b">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-7b>)">WizardLM-2 7B by microsoft | OpenRouter</a>: WizardLM-2 7B is the smaller variant of Microsoft AI&#x27;s latest Wizard model. It is the fastest and achieves comparable performance with existing 10x larger opensource leading models  It is a finet...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b>)">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1229488634621591582)** (6 messages): 

- **Seeking Beta Testers for New AI Platform**: A new advanced research assistant and search engine named [Rubiks.ai](https://rubiks.ai/) seeks beta testers, promising 2 months free premium access to various models like **Claude 3 Opus, GPT-4 Turbo, and Mistral Large**. Interested users can sign up using the promo code `RUBIX` and are encouraged to provide feedback via DM.
  
- **Early Subscription Hiccups**: A user subscribed to the Pro version of a service with a promo code but encountered an issue where they are repeatedly prompted to subscribe again.

- **Comfy-UI Complements Development**: A member praised the use of comfy-ui in the context of a development project, noting that it seems like a **natural fit** for this type of development.

- **Developer Search for AI Roleplay Frontend**: A community member is seeking a web developer to assist with a project aimed at creating a general-purpose AI frontend for OpenRouter, with an emphasis on roleplay elements. The individual is currently learning on the job and expressed a need for both development help and teaching.

- **Design Assistance for Narrative and Interface Features**: The same member has completed a novel mode for their project but requires assistance with the development of a conversational style mode and differentiating user-written text from AI-generated text. They are also looking to implement a **flexible modal system** within their interface.

**Link mentioned**: <a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1229413087942283364)** (258 messages🔥🔥): 

- **Falcon 180B Chat's Massive Requirements**: Users discussed the addition of Falcon 180B Chat GPTQ, noting its significant GPU memory requirement of around 100GB. The discussion included references to the [Falcon 180B repository](https://huggingface.co/TheBloke/Falcon-180B-Chat-GPTQ) and its compatibility requirements.

- **Cost Calculation Tips for Tokenization**: The chat addressed how to calculate costs per word for models, suggesting an average of 1.5 tokens per word as a good estimate, including considerations for punctuation marks and whitespaces.

- **Compliance with Prompting in Language Models**: Airoboros 70B was mentioned as a model that effectively follows prompts, for example by avoiding euphemisms and using colloquial language, whereas other models may not be as responsive.

- **Excitement and Concerns Over New WizardLM Model**: An announcement by a user about the release of a new model, WizardLM2, led to discussions about its features and comparisons to GPT-4. There were concerns about its removal from its original host due to a mysterious "missing internal process."

- **Weather Talk Amidst Model Discussions**: Amidst the technical discussions, several users commented on the current weather in their locations, ranging from hot conditions to rain in Abu Dhabi, showcasing the global diversity of the chat participants.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/TheBloke/Falcon-180B-Chat-GPTQ">TheBloke/Falcon-180B-Chat-GPTQ · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1229812396894457886)** (8 messages🔥): 

- **Seeking PyTorch Knowledge**: A member inquired if "Deep Learning with PyTorch" is the best starting point for going deeper into PyTorch, acknowledging the presence of two authors in the channel. 
- **Book Content Relevance Questioned**: Due to the book being 4 years old, there was a concern regarding the relevance of its syntax for current PyTorch usage. 
- **PyTorch Fundamentals Still Strong**: One member reassured that while the PyTorch core has not changed significantly, areas such as the compiler and distributed computing have evolved, but the book remains a good starting point.
- **Transformers, LLMs, and Deployment Sections Outdated**: Another member indicated that the book's content on transformers and LLMs is nonexistent, and the deployment section is now outdated. 
- **Anticipation for a New Edition**: A member expressed curiosity about the potential release of a new edition of the book to cover recent developments.
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1229389889666154558)** (30 messages🔥): 

- **Full Graph Compilation Issues Resolved**: After discussion, it was noted that using `torch.nn.functional.linear` and disabling CUDA graph with `enable_cuda_graph=False` can mitigate compilation issues faced with `torch.compile` set to `full_graph=True`. The token generation speed varies with and without full graph compilation but [**stable-fast**](https://github.com/chengzeyi/stable-fast/blob/main/src/sfast/jit/passes/__init__.py) could serve as an effective substitute in case of failures.

- **Stable-fast's Utility Beyond Stable Diffusion**: The use of **stable-fast** for optimizing inference proved to be beneficial, running close to INT8 quant *TensorRT speeds*. It was recommended for wider use beyond just Stable Diffusion models due to performance gains.

- **Exploring Fused Operations**: A suggestion was made to extend stable-fast with more types of fused operations, highlighting the library's potential versatility.

- **CUDA Graphs Not Always Helpful**: It was pointed out that CUDA graphs don't necessarily aid performance due to the overhead of recreating graphs for dynamic context lengths typical in autoregressive generation models.

- **GitHub Resource for Faster Matrix Operations**: A new **fp16 accumulation** vector-matrix multiplication that rivals **torch gemv** was added to [torch-cublas-hgemm](https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu), potentially offering a substantial boost in efficiency for specific matrix sizes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/mobicham/0e51c9f572721a76a5ac1e06fea533e9#file-stable_fast_llama_example-py-L14">stable_fast_llama_example.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/chengzeyi/stable-fast/blob/main/src/sfast/jit/passes/__init__.py">stable-fast/src/sfast/jit/passes/__init__.py at main · chengzeyi/stable-fast</a>: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs. - chengzeyi/stable-fast</li><li><a href="https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu">torch-cublas-hgemm/src/simt_hgemv.cu at master · aredden/torch-cublas-hgemm</a>: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu - aredden/torch-cublas-hgemm
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1229781137547853864)** (2 messages): 

- **Custom Backward Confusion**: A member is working on custom backward operations that work with a `(bs, data_dim)` input, similar to `F.Linear`, but faces issues when implementing within **Llama** due to a different input shape `(bs, seq_len, data_dim)`. They are seeking the forward/backward implementation of `F.Linear` and were unable to locate it in `tools/autograd/templates/python_nn_functions.cpp`, asking for guidance or a Pythonic implementation.
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1229679795848347688)** (2 messages): 

- **Automated Transcript Available for GPU Computing Talk**: An automated transcript of a talk on *Shared Memory and Synchronization in CUDA* is available, featuring automatically created notes and the ability to do Q&A. The transcript can be accessed at [Augmend Replay](https://wip.augmend.us/replay/PDHePF8AAA).

- **Augmend Offers Cutting-edge Video Processing**: The platform, [wip.augmend.us](https://wip.augmend.us), is currently able to process videos, incorporating features like OCR and image segmentation to capture information directly from the screen. The main site [augmend.com](https://augmend.com) will soon offer the ability to process any videos with these advanced capabilities.

**Link mentioned**: <a href="https://wip.augmend.us/replay/PDHePF8AAA">Advancing GPU Computing: Shared Memory and Synchronization in CUDA</a>: no description found

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1229724970331013160)** (2 messages): 

- **Newcomer Inquiry about PMPP Lectures**: A member inquired about the schedule and progress of meetings for going through **PMPP lectures**. They were interested in the frequency of meetings, the current chapter the group is on, and whether the lectures are recorded.

- **Where to Find Lecture Recordings**: Another member responded with a welcome and directed the newcomer to a specific channel (<#1198769713635917846>) to find the recorded lectures. They also updated that the last covered chapter of the **PMPP book** in the lectures was the *10th*.
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1229395032625123338)** (4 messages): 

- **Vector Processing Dilemma**: A member completed **Chapter 2** and inquired about the best configuration for processing a 100-length vector with a choice between (blockDim = 32, blocks = 4) or (blockDim = 64, blocks = 2). Another member advised that both setups are acceptable, but also suggested an alternative of using (blockDim = 32, blocks = 1) with a for loop inside the warp for **thread coarsening**.
- **Back to the Book After Midterms**: A participant mentioned they will resume reading from **Chapter 4** after their midterms, which are expected to end by approximately the end of the month.
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1229518584426922165)** (8 messages🔥): 

- **Weekend Talk Availability**: A member inquired if the weekend's talk was recorded, confirming that although there is a live recording available, a better-quality version will be posted by tomorrow.

- **Recording Quality Matters**: A re-recording of the weekend's talk is in progress to enhance the quality, following the intention to provide the best version possible.

- **Overseas Members Valuing Recordings**: The recordings are appreciated, especially for members in different time zones who find attending live lectures at 3/5 AM challenging.

- **Pre-talk Tech Check to Ensure Quality**: A member requested a way to test their setup before their scheduled talk to avoid any technical issues; they were advised to join early for a run-through.

- **Latest Lecture Video Shared**: A link to the YouTube video titled "Lecture 14: Practitioners Guide to Triton" was shared, along with its corresponding GitHub [lecture description](https://github.com/cuda-mode/lectures/tree/main/lecture%2014).

**Link mentioned**: <a href="https://www.youtube.com/watch?v=DdTsX6DQk24">Lecture 14: Practitioners Guide to Triton</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014

  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1229371863176843265)** (2 messages): 

- **TorchAO Tensor Layout Optimization Suggestion**: A member suggested that **torchao** could optimize tensor layout, as matrix multiplication operations could benefit from pre-swizzled weight storage and padded dimensions, potentially improving performance.
- **Torch.compile Already Handles Some Optimizations**: In response to the tensor layout optimization suggestion, another member clarified that **torch.compile** already deals with certain optimizations like padding ([see config code](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L420)) and layout optimization ([see layout optimization code](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L261)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L420">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L261">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1229541118119776297)** (4 messages): 

- **Solutions for Triton Puzzles Available**: A link to open solutions for Triton Puzzles was shared, available at [Zhaoyue's GitHub Repository](https://github.com/ZhaoyueCheng/Triton-Puzzles/blob/main/Triton_Puzzles_Solution_Zhaoyue.ipynb). The contributor invited feedback and expressed readiness to remove the solutions if necessary.
- **Tutorial Video in the Works**: Confirmation was given that sharing open solutions is acceptable, with a mention of a forthcoming tutorial video.
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1229378067789647935)** (35 messages🔥): 

- **Increased Efficiency with Torchao Int4 Kernel**: A rewrite of the generation pipeline for transformers now supports torchao int4 kernel, resulting in a performance increase to **152 tokens/sec** from **59 tokens/sec** with FP16, reflecting substantial improvements aligning with **gpt-fast** benchmarks.

- **Accuracy Improvements in HQQ Quantization**: A member noted improved accuracy in newer **HQQ+** results, with a Wiki Perplexity drop from 7.3 to **6.06**, and questioned if this involved corrections from the official [GitHub implementation](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py) of HQQ.

- **Resolving HQQ and GPT-Fast Compatibility Issues**: There's an ongoing discussion on how quantization axis (0 or 1) affects model performance where one member observed non-ideal interactions with gpt-fast’s pipeline when using `axis=0`. There’s active exploration of optimizing along `axis=1`, including utilizing fake data for autograd, and combining HQQ with LoRA.

- **Code Push for Tensor Cable Support and Autograd Optimizers**: The mobiusml team shared a [code push](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py) demonstrating how torch int4mm can work with Hugging Face transformers, reaching up to **154 tokens/sec**. Also released were experimental optimizers using autograd and fake data with potential for further iteration speed improvements.

- **Vectorized FP16 Multiplication Possibilities And Future Plans**: Participants also discussed the potential use of vectorized fp16 multiplication from a [Triton kernel repository](https://github.com/wangsiping97/FastGEMV) to speed up the quantized operations; one user emphasized that using `__hmul2` for half precision float multiplications could further enhance performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/zhxch">zhxch (zhongxiaochao)</a>: no description found</li><li><a href="https://huggingface.co/zhxchen17/scratch/tree/main">zhxchen17/scratch at main</a>: no description found</li><li><a href="https://github.com/wangsiping97/FastGEMV/tree/main">GitHub - wangsiping97/FastGEMV: High-speed GEMV kernels, at most 2.7x speedup compared to pytorch baseline.</a>: High-speed GEMV kernels, at most 2.7x speedup compared to pytorch baseline.  - GitHub - wangsiping97/FastGEMV: High-speed GEMV kernels, at most 2.7x speedup compared to pytorch baseline.</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py">hqq/examples/backends/torchao_int4_demo.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1229445393461542943)** (129 messages🔥🔥): 

- **CUDA Functionality Expansion and cuDNN Discussion**: A member introduced their own rudimentary llm.C project, prompting a discussion about whether optimizations like cuDNN for gelu/softmax operations have been addressed and reasons for not utilizing cuDNN. The size of cuDNN and its impact against the project's goals were debated, leading to suggestions for potential integration as comparison points while considering the project's principles.
- **Project Goals and Direction Clarified**: The project lead clarified the dual goals: educating with hand-written kernels for various layers and striving to create highly efficient kernels that rival those of CUTLASS, cuDNN, or any tool that offers significant performance gains. The importance of maintaining simple code while achieving performance was emphasized, even if it means foregoing minor gains for disproportionate complexity.
- **Optimization Steps and Profiling**: Various members participated in a technical discussion about optimization strategies. This included ideas such as exploiting padding for performance and using vectorized loads. It was agreed that profiling is a necessary first step to determine optimization targets, implying a need for analyzing kernels, particularly attention and softmax kernels.
- **Emerging PRs and Optimizations**: Several members shared their work on different PRs related to optimizing the current implementation, which included discussions about potential speedups by avoiding calculating the full probability matrix when unnecessary, and considering fusing operations within matmul layers. Specific attention was given to a fused classifier kernel and other potential easy gains.
- **Data Sets and Benchmarks for Training**: There was a conversation about selection of pretraining datasets like SlimPajama, Dolma, and MiniPile, as well as the difficulty in evaluating and comparing to GPT-2. A suggestion was made to integrate the ability to generate a GPT model at initialization to facilitate benchmarking and prevent hyperoptimization for the current model shapes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://godbolt.org/noscript/cuda">Compiler Explorer</a>: no description found</li><li><a href="https://godbolt.org/z/e9oqsqnY5">Compiler Explorer - CUDA C++ (NVCC 12.3.1)</a>:  // warp-level reduction for finding the maximum value __device__ float warpReduceMax(float val) {     for (int offset = 16; offset &amp;gt; 0; offset /= 2) {         val = fmaxf(val, __shfl_down_sync...</li><li><a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/notebooks/extend_thunder_with_cuda_python.ipynb">lightning-thunder/notebooks/extend_thunder_with_cuda_python.ipynb at main · Lightning-AI/lightning-thunder</a>: Make PyTorch models up to 40% faster! Thunder is a source to source compiler for PyTorch. It enables using different hardware executors at once; across one or thousands of GPUs. - Lightning-AI/ligh...</li><li><a href="https://github.com/karpathy/llm.c/pull/117">WIP: Fully fused classification layer by ngc92 · Pull Request #117 · karpathy/llm.c</a>: This fuses together all the pointwise operations that happen in the token classification layer. This essentially gives us the forward/backward for the cost of about just the forward pass, because t...</li><li><a href="https://github.com/karpathy/llm.c/pull/150">Optimised version of fused classifier + bugfixes(?) by ademeure · Pull Request #150 · karpathy/llm.c</a>: This is a faster version of the cool new kernel from #117 (still /dev/cuda/ only). The biggest difference is it is optimised for doing one row per 1024-wide block rather than per 32-wide warp, whic...</li><li><a href="https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)">SlimPajama: A 627B token, cleaned and deduplicated version of RedPajama - Cerebras</a>: Cerebras has built a platform for push-button training of large language models that can accelerate time to insights without having to orchestrate across a large cluster of small devices.</li><li><a href="https://arxiv.org/abs/2304.08442">The MiniPile Challenge for Data-Efficient Language Models</a>: The ever-growing diversity of pre-training text corpora has equipped language models with generalization capabilities across various downstream tasks. However, such diverse datasets are often too larg...</li><li><a href="https://github.com/tysam-code/hlb-gpt/tree/main">GitHub - tysam-code/hlb-gpt: Minimalistic, extremely fast, and hackable researcher&#39;s toolbench for GPT models in 307 lines of code. Reaches &lt;3.8 validation loss on wikitext-103 on a single A100 in &lt;100 seconds. Scales to larger models with one parameter change (feature currently in alpha).</a>: Minimalistic, extremely fast, and hackable researcher&amp;#39;s toolbench for GPT models in 307 lines of code. Reaches &amp;lt;3.8 validation loss on wikitext-103 on a single A100 in &amp;lt;100 secon...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[recording-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1229443783284756530)** (9 messages🔥): 

- **Volunteer for Recording and More**: A member, mr.osophy, offered to handle *recording, live streaming, clipping*, as well as updating the *video description* and the *repository*.
- **Teamwork on Recording Duties**: Another member, marksaroufim, mentioned that someone else (ID: 650988438971219969) is also interested in recording. They have been asked to coordinate for the recording setup.
- **Backup Recording Plans**: Marksaroufim plans to record the upcoming talk as a backup but will skip doing so the following week.
- **Coordination Confirmed**: Genie.6336 confirmed reaching out to Phil (presumably ID: 650988438971219969) via direct message to discuss recording responsibilities.
- **Role Name Brainstorming**: Members brainstormed on a role name, with suggestions like "gpu poor" by muhtasham and "Massively Helpful" by mr.osophy being offered in a light-hearted context.
  

---



**Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1229450843569520811)** (1 messages): 

```html
<ul>
  <li><strong>Introducing Pile-T5</strong>: EleutherAI has released <strong>Pile-T5</strong>, an enhanced T5 model family trained on the Pile with up to 2 trillion tokens, showing improved performance on SuperGLUE, code tasks, MMLU, and BigBench Hard. The models leverage the new LLAMA tokenizer and can be further finetuned for better results.</li>
  <li><strong>Intermediate Checkpoints Available</strong>: Intermediate checkpoints of Pile-T5 have been made available in both HF and original T5x versions, inviting the community to explore and build upon this advance in NLP models.</li>
  <li><strong>Comprehensive Resources for Pile-T5</strong>: Check out the <a href="https://blog.eleuther.ai/pile-t5/">detailed blog post</a> introducing Pile-T5 and the rationale behind its development, and access the code on <a href="https://github.com/EleutherAI/improved-t5">GitHub</a> to implement these improvements in your own projects.</li>
  <li><strong>Spreading the Word on Twitter</strong>: The release of Pile-T5 has also been announced on <a href="https://x.com/arankomatsuzaki/status/1779891910871490856">Twitter</a>, providing insights into the model's training process and highlighting its open-source availability.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.eleuther.ai/pile-t5/">Pile-T5</a>: Trained T5 on the Pile</li><li><a href="https://github.com/EleutherAI/improved-t5">GitHub - EleutherAI/improved-t5: Experiments for efforts to train a new and improved t5</a>: Experiments for efforts to train a new and improved t5 - EleutherAI/improved-t5</li><li><a href="https://x.com/arankomatsuzaki/status/1779891910871490856">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: 🚀 Introducing Pile-T5!  🔗 We (EleutherAI) are thrilled to open-source our latest T5 model trained on 2T tokens from the Pile using the Llama tokenizer.  ✨ Featuring intermediate checkpoints and a si...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1229356547339976714)** (61 messages🔥🔥): 

- **Discussions on Model Sharing and Access**: A member inquired about sharing a new model based on LLaMA, and another guided them to a specific channel for llama finetunes. However, the original poster reported they lacked permission to post there.
- **Missing Discord Emotes and TensorBoard Expertise**: Conversation included lighthearted regret over missing custom emotes. Also, a user sought documentation on Tensorboard's event files format and expertise in Tensorboard.
- **Exploration of Alternative Layer Configurations**: Sentialx queried the potential effects of splitting a single large layer into parts and adding them with weighted averaging. This prompted another member to liken it to a dot product, with varying reports on effectiveness based on personal experience.
- **Tokenizer Comparisons and Model Release Processes**: Users compared Pile-T5 and LLAMA tokenizers, noting differences such as vocabulary and support for newlines. Furthermore, a tweet by WizardLM was discussed, mentioning an omission in their model release process -- toxicity testing.
- **Latest on Encoder-Decoder Models and Compute Requirement Predictions**: A new encoder-decoder model named Reka was highlighted, reportedly supporting up to 128k, sparking conversations about the rarity of encoder-only or encoder-decoder models with long sequence lengths. Meanwhile, advice was sought on predicting compute requirements for reinforcement learning in LLM research, with a [latent space article](https://www.latent.space/p/transformers-math#details) suggested as a starting point for estimating an upper bound of transformer compute requirements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66">Pile-T5 - a EleutherAI Collection</a>: no description found</li><li><a href="https://x.com/srush_nlp/status/1779938508578165198">Tweet from Sasha Rush (@srush_nlp)</a>: Lazy twitter: A common question in NLP class is &#34;if xBERT worked well, why didn&#39;t people make it bigger?&#34; but I realize I just don&#39;t know the answer. I assume people tried but that a l...</li><li><a href="https://www.latent.space/p/transformers-math#details>">The Mathematics of Training LLMs — with Quentin Anthony of Eleuther AI</a>: Listen now | Breaking down the viral Transformers Math 101 article and high performance distributed training for Transformers-based architectures (or &quot;How I Learned to Stop Handwaving and Make th...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/arc.py#L61">lm-evaluation-harness/lm_eval/tasks/arc.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/hendrycks_test.py#L153">lm-evaluation-harness/lm_eval/tasks/hendrycks_test.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1229350837189480458)** (137 messages🔥🔥): 

- **Exploring the Limits of Language Models in Video Generation**: [Diffusion models for video generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) pose a more challenging problem than image synthesis due to the need for temporal consistency and difficulty in collecting high-quality video data. Pre-read material on **Diffusion Models** for image generation is recommended for better understanding.
  
- **Examining Tokenization's Role in Language Models**: Research [paper from Berkeley](https://arxiv.org/abs/2404.08335) shows the significance of tokenization in language modeling. Without tokenization, transformers default to unigram models, but tokenization permits near-optimal modeling of sequence probabilities.

- **LLMs as Data Compressors**: A study ([arXiv:2404.09937](https://arxiv.org/abs/2404.09937)) finds that perplexity is highly correlated with downstream performance when evaluating language models. This supports the idea that model compression capabilities may facilitate artificial intelligence development.

- **Challenges in Scaling Transformers to Unlimited Context Length**: The introduction of Megalodon and Feedback Attention Memory (FAM) ([arXiv:2404.08801](https://arxiv.org/abs/2404.08801) and [arXiv:2404.09173](http://arxiv.org/abs/2404.09173)) demonstrates attempts to improve transformer efficiency for handling long sequences.

- **Debate on Depth vs. Mixture-of-Experts (MoE) Models Efficacy**: Discussions centered around whether dense models or MoE models are superior, with considerations of VRAM constraints and inference cost. No clear consensus was reached, but various angles on advantages and constraints for each model type were debated.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://proceedings.mlr.press/v139/wies21a.html">Which transformer architecture fits my data? A vocabulary bottleneck in self-attention</a>: After their successful debut in natural language processing, Transformer architectures are now becoming the de-facto standard in many domains. An obstacle for their deployment over new modalities i...</li><li><a href="https://arxiv.org/abs/2404.09937">Compression Represents Intelligence Linearly</a>: There is a belief that learning to compress well will lead to intelligence. Recently, language modeling has been shown to be equivalent to compression, which offers a compelling rationale for the succ...</li><li><a href="https://arxiv.org/abs/2404.08801">Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length</a>: The quadratic complexity and weak length extrapolation of Transformers limits their ability to scale to long sequences, and while sub-quadratic solutions like linear attention and state space models e...</li><li><a href="https://lilianweng.github.io/posts/2024-04-12-diffusion-video/">Diffusion Models for Video Generation</a>: Diffusion models have demonstrated strong results on image synthesis in past years. Now the research community has started working on a harder task&mdash;using it for video generation. The task itself...</li><li><a href="http://arxiv.org/abs/2404.09173">TransformerFAM: Feedback attention is working memory</a>: While Transformers have revolutionized deep learning, their quadratic attention complexity hinders their ability to process infinitely long inputs. We propose Feedback Attention Memory (FAM), a novel ...</li><li><a href="https://arxiv.org/abs/2404.08819">The Illusion of State in State-Space Models</a>: State-space models (SSMs) have emerged as a potential alternative architecture for building large language models (LLMs) compared to the previously ubiquitous transformer architecture. One theoretical...</li><li><a href="https://arxiv.org/abs/2404.08335">Toward a Theory of Tokenization in LLMs</a>: While there has been a large body of research attempting to circumvent tokenization for language modeling (Clark et al., 2022; Xue et al., 2022), the current consensus is that it is a necessary initia...</li><li><a href="https://arxiv.org/abs/2404.03592">ReFT: Representation Finetuning for Language Models</a>: Parameter-efficient fine-tuning (PEFT) methods seek to adapt large models via updates to a small number of weights. However, much prior interpretability work has shown that representations encode rich...</li><li><a href="https://x.com/lambdaviking/status/1713945714684756019?s=46">Tweet from Will Merrill (@lambdaviking)</a>: [1/n] How does a chain of thought change the expressive power of transformers?  New work w/ @Ashish_S_AI studies how adding CoT/decoding steps extends the problems solvable by transformers as a fn of ...</li><li><a href="https://arxiv.org/abs/2103.13076">Finetuning Pretrained Transformers into RNNs</a>: Transformers have outperformed recurrent neural networks (RNNs) in natural language generation. But this comes with a significant computational cost, as the attention mechanism&#39;s complexity scales...</li><li><a href="https://arxiv.org/abs/2404.07850v1">MindBridge: A Cross-Subject Brain Decoding Framework</a>: Brain decoding, a pivotal field in neuroscience, aims to reconstruct stimuli from acquired brain signals, primarily utilizing functional magnetic resonance imaging (fMRI). Currently, brain decoding is...</li><li><a href="https://fixupx.com/fly51fly/status/1779872116458020991">Tweet from fly51fly (@fly51fly)</a>: [CL] Toward a Theory of Tokenization in LLMs N Rajaraman, J Jiao, K Ramchandran [UC Berkeley] (2024) https://arxiv.org/abs/2404.08335  - Transformers trained on data from certain simple high-order Mar...</li><li><a href="https://tenor.com/view/bait-thats-bait-tom-hardy-mad-max-gif-5055384">Bait Thats Bait GIF - Bait Thats Bait Tom Hardy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://link.springer.com/article/10.1007/s12243-024-01028-2">Large language models and unsupervised feature learning: implications for log analysis - Annals of Telecommunications</a>: Log file analysis is increasingly being addressed through the use of large language models (LLM). LLM provides the mechanism for discovering embeddings for distinguishing between different behaviors p...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1229546146481832028)** (27 messages🔥): 

- **OpenAI Unlocks GPT-4-Turbo Eval Implementations**: OpenAI has made public their evaluation implementations used for GPT-4-Turbo, allowing contributions through GitHub. The repository can be found at [openai/simple-evals on GitHub](https://github.com/openai/simple-evals).

- **Few-Shot Prompts Labeled Relics of the Past**: It was mentioned that few-shot prompts are considered remnants of base models, with zero-shot chain-of-thought prompts now being favored for robust inference.

- **Continuous Tasks Added to Evaluation Harness**: A contribution to the `lm-evaluation-harness` includes the addition of new `flores-200` and `sib-200` benchmarks, aiming to enhance multilingual evaluation for translation and text classification. Contributions can be reviewed in the respective pull requests: [Implement Sib200 evaluation benchmark](https://github.com/EleutherAI/lm-evaluation-harness/pull/1705) and [Implementing Flores 200 translation evaluation benchmark](https://github.com/EleutherAI/lm-evaluation-harness/pull/1706).

- **Technical Discussion on Big-Bench-Hard Evaluation**: A user inquiring about evaluation times for the `big-bench-hard` benchmark received advice on speeding up the process by adjusting the batch size or using `accelerate launch`, with mentions of changes to task names like `bbh_cot_zeroshot`.

- **Exploring LLM Compression as a Marker of Intelligence**: An interesting discussion topic arose around whether looking at Bits Per Character (BPC) as a unit of information could serve as a beneficial perspective in evaluating language model intelligence. A Twitter post and respective [GitHub repository](https://github.com/hkust-nlp/llm-compression-intelligence) provide context for the conversation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1780073500536872990">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Compression Represents Intelligence Linearly  LLMs&#39; intelligence – reflected by average benchmark scores – almost linearly correlates with their ability to compress external text corpora  repo: ht...</li><li><a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>: Contribute to openai/simple-evals development by creating an account on GitHub.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1705">Implement Sib200 evaluation benchmark - text classification in 200 languages  by snova-zoltanc · Pull Request #1705 · EleutherAI/lm-evaluation-harness</a>: We use the prompting style from the MALA paper https://arxiv.org/pdf/2401.13303.pdf Which we also found to have reasonable results in our SambaLingo paper https://arxiv.org/abs/2404.05829</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1706">Implementing Flores 200 translation evaluation benchmark across 200 languages by snova-zoltanc · Pull Request #1706 · EleutherAI/lm-evaluation-harness</a>: We used the prompt template from this paper that they found to work the best. https://arxiv.org/pdf/2304.04675.pdf  Our paper also found reasonable results with this prompt template https://arxiv.o...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1229437214644113438)** (7 messages): 

- **Decay Impact Limited by Weight Activation**: A member discussed that **weight decay** might only impact weights that are activated, suggesting if it's built into the optimizer and the model employs dummy tokens, those may not be affected due to lack of gradients.
- **Sanity Checks for Arbitrary Model Adjustments Under Scrutiny**: The effectiveness of **sanity checks** was questioned by a member, especially in cases where modifying values doesn't affect the model's performance, leading to suspicions that weights may not be decaying as intended. 
- **Initialization Consistency for Dummy Weights Questioned**: One member signaled uncertainty about whether the dummy weights are initialized in the same manner as other weights within the model, which could influence their behavior.
- **Dummy Weights Probably Initialized Similarly**: Another member chimed in with insights implying that dummy weights are likely initialized the same as other weights, based on the norm comparison with ordinary tokens.
- **Token Encoding Insights Shared**: Token encoding behavior was highlighted by sharing code and output that compared several encoded and decoded tokens to demonstrate how specific tokens are transformed.
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1229360630754971658)** (167 messages🔥🔥): 

- **Exploring the Neurobiological Basis of Language**: A member shared a link to [Language in Our Brain: The Origins of a Uniquely Human Capacity](https://direct.mit.edu/books/oa-monograph/3653/Language-in-Our-BrainThe-Origins-of-a-Uniquely) by Angela D. Friederici, discussing neurobiological differences in species that may explain the human capacity for language. This book offers insight into how language subcomponents are neurologically integrated.

- **The Data Glut Issue in Neuroscience**: A conversation about recent neuroscience research highlighted the challenge of a 'DATA Glut'—massive amounts of MRI data from the Big Brain Projects are stored, but proprietary software/hardware combinations cause bottlenecks in data interpretation and dissemination.

- **AI and Human Comparison Debates**: Members discussed the similarities and differences between AI and human beings, mentioning how AI systems cannot store learned information like humans and lack the ability to make independent decisions or feel emotions. The argument offered insight into current AI limitations and the philosophical implications of developing AI reasoning capabilities.

- **AI Research Tool API Availability**: A member initially expressed issues accessing Claude 3 API in Brazil, but after checking the availability list and reattempting, they managed to gain access. This showcases the intricacy of accessing AI tools across different countries and the need for clarification on availability.

- **Discussion on Turing Completeness in Magic: The Gathering (MTG)**: Members debated whether MTG, a collectible card game, could be considered Turing complete, linking to an academic paper for further exploration. This initiated a broader conversation on the Turing completeness and its applications outside of traditional programming languages.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Turing_completeness">Turing completeness - Wikipedia</a>: no description found</li><li><a href="https://direct.mit.edu/books/oa-monograph/3653/Language-in-Our-BrainThe-Origins-of-a-Uniquely)">Language in Our Brain: The Origins of a Uniquely Human Capacity</a>: A comprehensive account of the neurobiological basis of language, arguing that species-specific brain differences may be at the root of the human capacity 
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1229355466266771456)** (7 messages): 

- **GPT's Context Window a Known Limitation**: A member lamented that OpenAI has yet to significantly increase GPT's context window, stating this as a limitation one needs to adjust to due to the current constraints on context size.
- **ChatGPT vs. GPT Context Sizes Clarified**: The API for **GPT** was mentioned, allowing up to a 128k context, whereas **ChatGPT** typically supports up to a 32k context.
- **Extra Context Via API Document Uploads**: When using the API for interaction, documents can be uploaded in **playground assistants**, extending the effective context window to 128k.
- **Document Upload Feature Termed "Retrieval"**: The term **"retrieval"** was introduced to describe the process of uploading documents to expand context when using the API.
- **Inquiries About GPT Dynamic**: A user asked for clarification on what **GPT Dynamic** is and how it operates, indicating curiosity about newer or less-known functionalities.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1229450314357407784)** (3 messages): 

- **Seeking the Name of the Contest**: A user inquired about the name of a certain competition.
- **Short and Mysterious Reply**: Another member simply responded with the word "buzz."
- **Possible Pixar Reference?**: A different user mentioned "light year," possibly alluding to Disney-Pixar's 'Toy Story' franchise character Buzz Lightyear.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1229450314357407784)** (3 messages): 

- **Inquiry About Competition Name**: A member inquired about the name of a competition but did not provide further details.
- **Onomatopoeic Interaction**: Another individual simply contributed the word "buzz" without any context or additional information.
- **Possible Competition or Project Mention**: A reference was made to "light year" which could suggest the topic of a competition or project but lacked detail or clarification.
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1229457912573857854)** (3 messages): 

- **Launch of IFTTT Agent Interfaces Tutorial Series**: [Agent interfaces and applications tutorial series announced](https://twitter.com/llama_index/status/1779898403239125198), intended as an introductory course to clarify ambiguities around core agent interfaces and their uses.

- **Qdrant Engine Hybrid Cloud Debut**: [Qdrant Engine rolls out its hybrid cloud service](https://twitter.com/llama_index/status/1780275878230139293) in partnership with LlamaIndex, allowing users to host Qdrant across various environments.

- **Hybrid Search with Azure AI and LlamaIndex**: Learn how to integrate LlamaIndex with Azure AI Search for enhanced RAG applications, offering features like hybrid search and query rewriting, in a new [tutorial shared by Khye Wei](https://twitter.com/llama_index/status/1780324017083400235) from Microsoft.
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1229436577260896297)** (117 messages🔥🔥): 

- **Async Compatibility Query for Claude in LlamaIndex**: A member inquired about async compatibility with Claude in Bedrock within LlamaIndex. Another member responded that async is not implemented for Bedrock and is open to contributions [here](https://docs.llamaindex.ai/en/stable/).

- **Implementing Assistance Sought**: When a member asked for help constructing complex queries, they were guided to examples in the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/?h=query+pipeline).

- **Usage of Llama CPP Server in LlamaIndex**: There were inquiries about using an async Huggingface embedding model with LlamaIndex through Sagemaker and using a hosted Llama CPP server for chat functionality. It was indicated that async compatibility needs to be implemented, with a suggestion to use an async boto session, and no clarity was provided on utilizing a hosted CPP server.

- **Troubleshooting Integration Issues**: A member trying to use LlamaIndex with gpt-3.5-turbo encountered an authentication error, which might be due to an old version of the integration or missing balance on the Azure account. Updating LlamaIndex can potentially resolve this issue.

- **Agent Behavior Under Specific Conditions**: An inquiry was made on how to condition an LLM to make decisions if no matching rows are found in a CSV file. No solution was provided within the messages, but the issue seems to involve configuring a fallback to a fine-tuned model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/create_llama_projects/tree/main/nextjs-edge-llamaparse">create_llama_projects/nextjs-edge-llamaparse at main · run-llama/create_llama_projects</a>: Contribute to run-llama/create_llama_projects development by creating an account on GitHub.</li><li><a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents/">Multi-Document Agents - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/answer_and_context_relevancy/">Answer Relevancy and Context Relevancy Evaluations - LlamaIndex</a>: no description found</li><li><a href="http://localhost:port",>">no title found</a>: no description found</li><li><a href="http://localhost:port"`>">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/llms/openai_like#llama_index.llms.openai_like.OpenAILike>).">Openai like - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/llama_2_llama_cpp#setup-llm>)">LlamaCPP - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/openapi#llama_index.tools.openapi.OpenAPIToolSpec>)">Openapi - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/?h=query+pipeline">Query Pipeline Chat Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=query+pipeline">Building an Agent around a Query Pipeline - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1229473909728870520)** (15 messages🔥): 

- **Efficient Reasoning Chain Integration**: An article titled "Unlocking Efficient Reasoning: Integrating LlamaIndex with Chain of Abstraction" was shared, highlighting advancements between LlamaIndex and abstraction chains. The article can be found at [Unlocking Efficient Reasoning](https://ai.gopubby.com/unlocking-efficient-reasoning-integrating-llamaindex-with-chain-of-abstraction-1b1844ba66e6).
- **Positive Reception to Articles**: A member commended the articles shared in the chat, particularly praising the one regarding efficient reasoning.
- **Inquiry about Token Counter in RAGStringQueryEngine**: A user sought advice on integrating a token counter into a RAGStringQueryEngine within LlamaIndex.
- **Detailed Token Counter Integration Guide**: An extensive step-by-step guide was provided for adding a token counter to the LlamaIndex's `RAGStringQueryEngine`. It involves creating a `TokenCountingHandler`, a `CallbackManager`, and then assigning the manager to both global settings and the specific query engine.
- **Hierarchical Document Structuring Question**: A user queried about building a hierarchical parent-child structure in LlamaIndex (ParentDocumentRetriever langchain) for managing millions of documents, seeking any leads or guidance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/api_reference/callbacks/token_counter#llama_index.core.callbacks.token_counting.TokenCountingHandler.prompt_llm_token_count>)">Token counter - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/supporting_modules/settings#callbacks>)">Settings - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1229358474916265989)** (108 messages🔥🔥): 

- **High-Quality TTS Library by Hugging Face**: A link to the [GitHub repository](https://github.com/huggingface/parler-tts) for Hugging Face's high-quality TTS models inference and training library, *parler-tts*, was shared.
- **Incorrect Min-SNR-Gamma Implementation in Diffusers**: A member noted that the min-snr-gamma formula in *Diffusers* might be incorrect for v-prediction. They are planning to use symbolic regression to find a better starting point after letting the current process converge.
- **New Legislation on Deepfakes**: Discussion on upcoming legislation targeting the creation of deepfake images intended to cause distress. Debate over the practicality of enforcing such laws and the focus on the intent behind creating the images.
- **Misleading and Dishonest AI Projects Discussed**: '**Stable Attribution**' was pointed out as a misleading AI project from the last year. It was noted that misleading information about the project remains uncorrected in official publications.
- **Beware of AI Scams**: A [potential scam](https://www.open-sora.org) targeting AI enthusiasts was mentioned. Users were advised to be cautious, as it appeared to be a fake company named Open Sora through a Facebook Ad.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.open-sora.org||">no title found</a>: no description found</li><li><a href="https://www.bbc.com/news/uk-68823042">Creating sexually explicit deepfakes to become a criminal offence</a>: A new law will see creators of sexually explicit deepfakes face prosecution and a fine.</li><li><a href="https://huggingface.co/spaces/declare-lab/tango">Tango - a Hugging Face Space by declare-lab</a>: no description found</li><li><a href="https://tenor.com/b1ALd.gif">Minority Report Leave GIF - Minority Report Leave Walk Away - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/huggingface/diffusers/issues/5654>">Issues · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - Issues · huggingface/diffusers</li><li><a href="https://github.com/huggingface/parler-tts">GitHub - huggingface/parler-tts: Inference and training library for high-quality TTS models.</a>: Inference and training library for high-quality TTS models. - huggingface/parler-tts
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1229426098757177344)** (17 messages🔥): 

- **CLIP's Performance on a Budget Explored**: A [recent paper](https://arxiv.org/abs/2404.08197) investigates the scaling down of **Contrastive Language-Image Pre-training (CLIP)** and suggests that high-quality data has a significant impact on model performance, even when dealing with smaller datasets. It further highlights that *CLIP+Data Augmentation* can match CLIP's performance using only half the training data, prompting discussions on data efficiency.

- **Pile-T5: A New Take on a Community Favorite**: [EleutherAI's blog post](https://blog.eleuther.ai/pile-t5/) introduces Pile-T5, a modified version of the T5 model trained on the Pile dataset with the LLAMA tokenizer, covering double the token amount of the original model.

- **Launching a New Safety Benchmark for Language Models**: The ALERT benchmark has been released, providing a [new safety benchmark](https://github.com/Babelscape/ALERT) for assessing large language models with a focus on red teaming and potentially harmful content.

- **Text-to-Audio Generation Gets a Closer Look**: An [arXiv submission](https://arxiv.org/abs/2404.09956) details efforts using the Tango model to **improve the relevance and order of audio events** generated from text prompts, aiming to boost performance in audio generation from text in data-limited scenarios.

- **Safety in AI: A Contentious Debate**: Conversations about the [establishment of safety benchmarks](https://github.com/Babelscape/ALERT) sparked discussions over the removal of content deemed unsafe for AI, the freedom in creative arts, and the balance between societal expectations and the capabilities of AI tools.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/RekaAILabs/status/1779894626083864873">Tweet from Reka (@RekaAILabs)</a>: Along with Core, we have published a technical report detailing the training, architecture, data, and evaluation for the Reka models.  https://publications.reka.ai/reka-core-tech-report.pdf</li><li><a href="https://arxiv.org/abs/2404.09956">Tango 2: Aligning Diffusion-based Text-to-Audio Generations through Direct Preference Optimization</a>: Generative multimodal content is increasingly prevalent in much of the content creation arena, as it has the potential to allow artists and media personnel to create pre-production mockups by quickly ...</li><li><a href="https://arxiv.org/abs/2404.08197">Scaling (Down) CLIP: A Comprehensive Analysis of Data, Architecture, and Training Strategies</a>: This paper investigates the performance of the Contrastive Language-Image Pre-training (CLIP) when scaled down to limited computation budgets. We explore CLIP along three dimensions: data, architectur...</li><li><a href="https://blog.eleuther.ai/pile-t5/">Pile-T5</a>: Trained T5 on the Pile</li><li><a href="https://github.com/Babelscape/ALERT">GitHub - Babelscape/ALERT: Official repository for the paper &quot;ALERT: A Comprehensive Benchmark for Assessing Large Language Models’ Safety through Red Teaming&quot;</a>: Official repository for the paper &quot;ALERT: A Comprehensive Benchmark for Assessing Large Language Models’ Safety through Red Teaming&quot; - Babelscape/ALERT
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1229892954500370442)** (10 messages🔥): 

- **IDEFICS-2 Released with Multimodal Capabilities**: The newly released [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) boasts capabilities including image and text sequence processing, OCR enhancements, and high-resolution image handling up to 980 x 980. It competes with models such as LLava-Next-34B and performs tasks such as visual question answering and document understanding.

- **IDEFICS-2 Showcases Advanced Example**: An example shared combines IDEFICS-2's understanding of text recognition, arithmetic, and color knowledge, effectively demonstrating its application by **"solving" a CAPTCHA with significant noise**.

- **Chatbot Format for IDEFICS-2 on the Horizon**: While the current IDEFICS-2 model is tailored for tasks like visual question answering, an upcoming release will include a **chatty version** designed for conversational interaction.

- **Update Promised for Chat Variant**: An interested party inquired about the chatbot variant of IDEFICS-2, to which it was mentioned that updates would be provided once a demo is available.

- **IDEFICS-2 Demonstrates Noise-Tolerant CAPTCHA Solving**: IDEFICS-2 was showcased solving a CAPTCHA replete with a large amount of noise and extraneous text, highlighting its robust multimodal understanding and practical utility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/HuggingFaceM4/idefics-8b">Idefics 8b - a Hugging Face Space by HuggingFaceM4</a>: no description found</li><li><a href="https://x.com/lunarflu1/status/1780228654397599904">Tweet from lunarflu (@lunarflu1)</a>: cool multimodal interaction from IDEFICS-2 @huggingface : 1. Detect numbers from image 2. Do math with the number 3. Retrieve background color 4. Remove pigment -&gt; Resulting color 5. Final result: ...</li><li><a href="https://huggingface.co/blog/idefics2">Introducing Idefics2: A Powerful 8B Vision-Language Model for the community</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1779998271546474593">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Idefics 2 x Transformers! 🔥  Trying out the Idefics 2 8B in the wild.  Pretty wild that you can do all this in less than 10 lines of code!  Made a quick screencast taking the model out for a spin..  ...</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized">argilla/distilabel-capybara-dpo-7k-binarized · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/papers/2403.07691">Paper page - ORPO: Monolithic Preference Optimization without Reference Model</a>: no description found</li><li><a href="https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-141b-A35b">alignment-handbook/recipes/zephyr-141b-A35b at main · huggingface/alignment-handbook</a>: Robust recipes to align language models with human and AI preferences - huggingface/alignment-handbook</li><li><a href="https://x.com/narsilou/status/1778887423713333648">Tweet from Nicolas Patry (@narsilou)</a>: Tgi 2.0 is out!  -back to fully open source for good (apache 2.0) - Fastest inference server in existence (110 tok/s for cohere R+, with medusa speculation) - fp8 support - mixtral 8x22b support ! (al...</li><li><a href="https://x.com/xenovacom/status/1778812177215881395">Tweet from Xenova (@xenovacom)</a>: Introducing MusicGen Web: AI-powered music generation directly in your browser, built with 🤗 Transformers.js! 🎵  Everything runs 100% locally, meaning no calls to an API! 🤯 Served as a static websi...</li><li><a href="https://x.com/AndrewYNg/status/1779905922602782752">Tweet from Andrew Ng (@AndrewYNg)</a>: LLMs can take gigabytes of memory to store, which limits what can be run on consumer hardware. But quantization can dramatically compress models, making a wider selection of models available to develo...</li><li><a href="https://huggingface.co/blog/vlms">Vision Language Models Explained</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1229368977726246982)** (72 messages🔥🔥): 

- **Mobile Discord Woes**: A member expressed frustration regarding the mobile Discord version, mentioning it has taken a "weird turn" and appeared to downgrade, prompting agreement from others. The change seems to persist despite user feedback.
- **Collaborative PRs on HuggingFace Hub**: Members discussed whether it's possible to commit to someone else's PR on HuggingFace, with references to the [HfApi.create_commits_on_pr documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.create_commits_on_pr) and general git best practices such as maintainer permissions and handling of PRs on forks.
- **Generative Models on CPU and User Testing Opportunities**: Users sought advice on generative art tools suitable for CPU use prioritizing speed, while another user invited the community for beta testing a new advanced research assistant and search engine with access to various advanced language models.
- **Machine Learning Enthusiasts Needed for Survey**: A [Google Forms survey](https://forms.gle/UvGdWrZhphoDFGQ99) was shared by students researching the democratisation of machine learning, inviting participation from the ML community.
- **Confusions in Checkpoint Conversions and Spaces Access**: Users shared issues with conversion scripts erroneously asking for directories when files were provided and discussed the complication of internet providers blocking HuggingFace Spaces, where one space seemed unblocked for a user after being reported but others still experienced difficulties.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/UvGdWrZhphoDFGQ99">The Democratisation of Machine Learning - Survey</a>: Thank you for taking the time to answer this survey about people’s experience with machine learning, it should take no more than 5 min  Throughout this survey &#39;Machine Learning&#39; will be referr...</li><li><a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1229461259041243267)** (4 messages): 

- **A Geometric Adventure on YouTube**: A member shared a [YouTube playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=GuhzATF13TOmwUMU) that offers a visual journey through various concepts in geometry.
- **Chain of Thoughts for AI**: An article titled "Unlocking Efficient Reasoning: Integrating LlamaIndex with Chain of Abstraction" was shared, demonstrating how a structured searching method can enhance AI reasoning; read more on [Locking Efficient Reasoning](https://ai.gopubby.com/unlocking-efficient-reasoning-integrating-llamaindex-with-chain-of-abstraction-1b1844ba66e6).
- **Mysterious Machine Learning Source**: A member posted a link which appears to be a shareable resource or dataset, but additional context was not provided. View it here: [Link](https://g.co/gemini/share/e8962ab90c1c).
- **Accelerated Artistry with AI**: Check out this rapid image creation tool demonstrated at the [HuggingFace Splatter Image space](https://huggingface.co/spaces/szymanowiczs/splatter_image), which significantly speeds up the process of image generation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/szymanowiczs/splatter_image">Splatter Image - a Hugging Face Space by szymanowiczs</a>: no description found</li><li><a href="https://g.co/gemini/share/e8962ab90c1c">‎Gemini - Cours Data Science, IA et GenAI</a>: Created with Gemini
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1229400064703725588)** (15 messages🔥): 

- **Hyperparameter Experimentation**: A member is experimenting with different **training hyperparameters** using their custom codebase and a **model trained from scratch**.
  
- **BLIP Model Fine-Tuning**: A member has fine-tuned the **BLIP** model for generating long captions of images and launched a [comparison of different models](https://huggingface.co/spaces/unography/comparing-captioning-models). They plan next to recaption some text-image datasets to fine-tune text-to-image models.

- **Musical AI Remix on GitHub**: User `.bigdookie` has complexified a music creation project leading to an 'infinite' remix with **musicgen-medium** and shared it on [GitHub](https://github.com/betweentwomidnights/infinitepolo). They also provided a [YouTube demo](https://youtu.be/tzw6otpW-4A) and invited others to contribute.

- **Community Highlights in Portuguese**: The user `rrg92_50758` shared a new YouTube video covering the **Community Highlights #52** in Portuguese for the #huggingface Discord community.

- **BLIP Model Usage Query**: A discussion emerged on how to set maximum output length when using the **BLIP** model serverless inference through `curl`. The conversation included links to the Hugging Face documentation and concluded with confirmation that `max_new_tokens` is the parameter to use.

- **Stepping into Java for Image Recognition**: A member shared a [Medium article](https://medium.com/@visrow/image-recognition-and-function-calling-with-gemini-and-java-e28b0356d3de) about using **Gemini** and **Java** for image recognition and function calling.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/unography/comparing-captioning-models">Comparing Captioning Models - a Hugging Face Space by unography</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=eNAOaFGrm2Y">Destaques da Comunidade #52: Confira as últimas novidades de IA</a>: Confira as novidades do Community Highlights publicadas no Discord do #huggingface , em português!Resumo:- Novos modelos LLM open source- Interfaces web para...</li><li><a href="https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task">Detailed parameters</a>: no description found</li><li><a href="https://youtu.be/tzw6otpW-4A">infinite remix with musicgen, ableton, and python - part 2 - captains chair 22</a>: 00:00 - recap00:38 - human musician continues musicgen&#39;s bass02:39 - musicgen continues human musician&#39;s output04:04 - extended demo of hoenn&#39;s lofi model06:...</li><li><a href="https://github.com/betweentwomidnights/infinitepolo">GitHub - betweentwomidnights/infinitepolo: a song in python</a>: a song in python. Contribute to betweentwomidnights/infinitepolo development by creating an account on GitHub.</li><li><a href="https://huggingface.co/thepatch">thepatch (the collabage patch)</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/pipelines#transforme">Pipelines</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageToTextPipeline">Pipelines</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1229503713643728997)** (3 messages): 

- **Join the LLMs Reading Group**: A reminder for an upcoming free LLMs Reading Group session focused on the **Aya Multilingual Dataset** is scheduled for Tuesday, April 16th. Interested participants can [RSVP here](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator) and explore the full 2024 session itinerary.
  
- **Linking to Google Calendar**: For those looking to link the LLMs Reading Group events to their Google Calendar, it's noted that adding to the calendar can be done through Eventbrite after RSVPing to the sessions. Attendees are welcome to join any single session or multiple sessions as per interest.

- **Human Feedback Foundation's Role Highlighted**: The **Human Feedback Foundation** supports the open-source AI community by integrating human input into AI models, particularly in critical areas such as healthcare and governance. The foundation aims to maintain a global database of human feedback to serve as a democratically sourced, authoritative dataset for AI developers.

**Link mentioned**: <a href="https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator">LLM Reading Group (March 5, 19; April 2, 16, 30; May 14; 28)</a>: Come and meet some of the authors of some seminal papers in LLM/NLP research and hear them them talk about their work

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1229356271430275092)** (2 messages): 

- **In Search of a Niche Vision Model**: A member inquired about fine-tuning a vision model for *captioning and identifying entities in low-resolution images*. They stress the need for optimizing the model specifically for lower resolutions to avoid unnecessary complexity.
- **Seeking an Advanced Tagger for SDXL**: Another member asked for recommendations on sophisticated taggers for SDXL, hinting at a preference over the existing wd14 tagger.
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1229413908675170347)** (8 messages🔥): 

- **NLP Beginners Start with spaCy and NLTK**: A member began their NLP journey with **spaCy and NLTK**, and moved on to a contemporary textbook recommended within the community for more updated insights.

- **Deciphering Model Output**: A user inquired about decoding a `batch_size, seq_size, emb_size` tensor into natural language using a **T5 model**. The attempt using `model.generate` failed due to requirements for a specific object structure including `last_hidden_`.

- **Exploring LoRA Configurations**: One user is experimenting with **LoRA configurations** for fine-tuning, seeking advice on the implications of changing the bias setting to 'all' or 'lora_only'.

- **Dataset Preparation for ROBERTA Q&A Chatbot**: A member asked for guidance on preparing a CSV dataset with 100000 entries and over 20 features for fine-tuning a **ROBERTA model** for a question-answering chatbot.

- **BERTopic Framework in Discussion**: Users discussed the **BERTopic framework**, a topic modeling technique that uses transformers and c-TF-IDF, highlighting its fast performance and good results with the option for guided, supervised, semi-supervised, manual, hierarchical, and class-based topic modeling ([BERTopic Guide](https://maartengr.github.io/BERTopic/index.html)). The conversation touched on the need for including custom stop words and choosing an appropriate LLM for converting seed words into phrases.

**Link mentioned**: <a href="https://maartengr.github.io/BERTopic/index.html">Home</a>: Leveraging BERT and a class-based TF-IDF to create easily interpretable topics.

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1229407840708329552)** (8 messages🔥): 

- **Confusion Over Model Types Cleared Up**: A new member inquired about the difference between **LLMs** and embedding models, specifically questioning the difference between the **OpenAI Embedding model** and **OpenAI GPT-3.5 Turbo**.

- **Token Limit Issue with Stable Diffusion Cascade Model**: A member brought up an issue when using the **stable diffusion cascade model with long prompts**, encountering a token limit error where their input was truncated. They also shared a [GitHub issue link](https://github.com/huggingface/diffusers/issues/7672) where they detailed the problem.

- **Clarity on Token Truncation**: In response to the confusion about the token limit, it was clarified that the message received was a **warning, not an error**, and that the truncated tokens, like 'hdr' in the example given, are indeed ignored.

- **Searching for Solutions to the Truncation Warning**: The member expressed concern about the truncation warning being problematic, asking for potential solutions, with **Compel library** being mentioned as an exception.
  
- **Maintenance Questions Regarding the Compel Library**: In response to the Compel library's mention, it was pointed out that **Compel doesn't appear to be currently maintained**. When queried about what maintenance was needed, there was no immediate response to specify the exact needs.

**Link mentioned**: <a href="https://github.com/huggingface/diffusers/issues/7672">error in using stable cascade with long prompt · Issue #7672 · huggingface/diffusers</a>: Hi, When I use stable cascade model with long prompt, I get below error. Token indices sequence length is longer than the specified maximum sequence length for this model (165 &gt; 77). Running this s...

  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1229352896106070027)** (100 messages🔥🔥): 

- **User Reports Issue with Macedonian Translation**: A member expressed concerns about the quality of **Command-R** when operating in Macedonian. They mentioned having reported this on the community-support channel.
- **Inquiry About #computer-vision Channel Access**: A user was seeking access to the **#computer-vision** channel and detailed the informative content and presentation schedules that feature in that channel.
- **Making Chat Stream async in Python**: There was a technical discussion on how to convert a piece of code into an asynchronous form to stream chat from the **Command-R** model efficiently.
- **Questions About API Rate Limits**: Queries about the trial key rate limits for **Cohere's API** were addressed, clarifying that "generate, summarize" has a limit of 5 calls/min, other endpoints have 100 calls/min, and there's a 5000 total limit across all trial API keys per month.
- **Discussion About Cohere's Paid Production API**: Members discussed the possibility of a paid program to access **Commander R+** through Cohere. A link to the [Cohere Production API documentation](https://docs.cohere.com/reference/chat) was shared, indicating that it already exists.
- **Exploring Cohere's Chat API and Connectors**: An inquiry was made about the functionality of the Chat API when used with connectors, and whether it manages embedding and retrieval processes. It was clarified that connectors are meant to link the model with sources of text and citations are an included functionality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.]">no title found</a>: no description found</li><li><a href="https://docs.anthropic.com/claude/reference/messages-examples#putting-words-in-claudes-mouth">Messages examples</a>: no description found</li><li><a href="https://ibb.co/s348vXt">Screenshot-2024-04-16-151544 hosted at ImgBB</a>: Image Screenshot-2024-04-16-151544 hosted in ImgBB</li><li><a href="https://docs.cohere.com/reference/chat">Chat API Reference - Cohere Docs</a>: no description found</li><li><a href="https://huggingface.co/Dracones/c4ai-command-r-plus_exl2_5.5bpw">Dracones/c4ai-command-r-plus_exl2_5.5bpw · Hugging Face</a>: no description found</li><li><a href="https://sites.google.com/cohere.com/c4ai-community/community-programs/computer-vision?authuser=0)?">Community - Computer Vision</a>: Channel: #computer-vision Co-leads:  Benedict - @Harkhymadhe on Discord, @Arkhymadhe on Twitter  Logistics: Occurrences:  Second Tuesday of each month at 8am PT  Feel free to add papers/articles you w...</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.  - GitHub - Unstructured-IO/unstructured: Open source librar...</li><li><a href="https://github.com/cohere-ai/quick-start-connectors">GitHub - cohere-ai/quick-start-connectors: This open-source repository offers reference code for integrating workplace datastores with Cohere&#39;s LLMs, enabling developers and businesses to perform seamless retrieval-augmented generation (RAG) on their own data.</a>: This open-source repository offers reference code for integrating workplace datastores with Cohere&amp;#39;s LLMs, enabling developers and businesses to perform seamless retrieval-augmented generation...</li><li><a href="https://github.com/cohere-ai/sandbox-conversant-lib">GitHub - cohere-ai/sandbox-conversant-lib: Conversational AI tooling &amp; personas built on Cohere&#39;s LLMs</a>: Conversational AI tooling &amp; personas built on Cohere&#39;s LLMs - cohere-ai/sandbox-conversant-lib</li><li><a href="https://coral.cohere.com/?s=t">Login | Cohere</a>: Cohere provides access to advanced Large Language Models and NLP tools through one easy-to-use API. Get started for free.</li><li><a href="https://dashboard.cohere.com/playground/chat">Login | Cohere</a>: Cohere provides access to advanced Large Language Models and NLP tools through one easy-to-use API. Get started for free.
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1229446480331280506)** (1 messages): 

- **Rubiks.ai launches with a plethora of models**: [Rubiks.ai](https://rubiks.ai/) has launched an advanced research assistant and search engine. Users are offered 2 months of premium access free to test features including **Claude 3 Opus, GPT-4 Turbo, Mistral Large**, and other models like **Mixtral-8x22B** with **RAG** on Groq's high-speed servers.

**Link mentioned**: <a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1229384441554665472)** (56 messages🔥🔥): 

- **Deepspeed Success on Multi-node**: A member reported that **Deepspeed 01 and 02** configurations work well on multi-node setups. A [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files?short_path=3520786#diff-35207863e6e0da8dfa2d1311bf863b60c52a067c5e65253c24543edda5da00d0) with a guide for multi-node distributed fine-tuning using Axolotl was shared for anyone encountering configuration issues.
  
- **Idefics2 8B Outperforming its Predecessor**: The **Idefics2** model is now available on Hugging Face and enhances capabilities in OCR, document understanding, and visual reasoning, outperforming **Idefics1** with fewer parameters and improved image handling. Access the model at [Hugging Face](https://huggingface.co/HuggingFaceM4/idefics2-8b).

- **RTX 5090 Launch Anticipation**: Discussions suggest that Nvidia might release the **RTX 5090** graphics card sooner than expected, potentially at the Computex trade show in June, due to competitive pressure, as reported on [PCGamesN](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch).

- **Mixtral 8x22B Inference Possibilities Explored**: Members shared experiences with **Mixtral 8x22B**, noting that using 3 A6000 with FSDP for QLoRA during training, and theorizing that two A6000 could be sufficient for inference with up to 4k context length using 4-bit quantization.

- **Model Stock Method as Forgetting Mitigator**: A comparison of different training methods on **Cosmopedia-subset data** indicates that **Model Stock merge** of various tuning methods is effective at repairing catastrophic forgetting. It was suggested that combined fine-tuning methods like Model Stock, QLoRA, GaLore, and LISA might reduce adaptation intensity, potentially needing louder LoRAs to compensate.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch">Nvidia’s RTX 5090 and 5080 could arrive much sooner than expected, but there’s a big catch</a>: Leaks point to the new Nvidia Blackwell GeForce GPUs arriving much sooner than originally expected, thanks to competition from AMD.</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files?short_path=3520786#diff-35207863e6e0da8dfa2d1311bf863b60c52a067c5e65253c24543edda5da00d0">Guide For Multi-Node Distributed Finetuning by shahdivax · Pull Request #1477 · OpenAccess-AI-Collective/axolotl</a>: Title: Distributed Finetuning for Multi-Node Setup Guide Description: This PR introduces a comprehensive guide for setting up a distributed finetuning environment using Axolotl and Accelerate. The ...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1229429796057125078)** (5 messages): 

- **Twitter Link Shared**: A member shared a Twitter link, prompting reactions from others who found the content interesting. The precise content of the Twitter post is not described, but it elicited responses like "oh that's cool" and "oh nice."
- **Seeking Guidance**: One member expressed curiosity with "Oooooo how do I use this?" indicating a desire for guidance on the use of the content shared in the Twitter link.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1229634311632715867)** (11 messages🔥): 

- **Gradient Accumulation Under Scrutiny**: A member questioned the efficacy of **gradient accumulation** when using sample packing, as they observed that training time and dataset length change with varying gradient accumulation steps. The member was seeking clarity on whether gradient accumulation indeed conserves memory as expected.

- **Deciphering Input's Role in Training**: One member inquired about the purpose of the input if `train_on_input` is disabled, speculating whether it's still used to teach the model context handling and prediction. Another member responded, suggesting that the input becomes a bigger part of the context, and the model will remember or steer towards it more.

- **Clarifying Loss Calculation with `train_on_input`**: In a follow-up, it was clarified that when `train_on_input` is enabled, the **loss is not calculated** with respect to the input, which subsequently influences the model training differently as it's not predicting the input part anymore.

- **Understanding Impact of Loss on Training**: The conversation continued with a discussion about the role of loss in training when `train_on_input` is on, leading to a confirmation that loss does not influence training in this scenario. It was also mentioned that without evaluation enabled, loss becomes even less relevant.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1229738683494699008)** (15 messages🔥): 

- **Axolotl's Epoch Dilemma**: A member sought advice on how to configure Axolotl to prevent saving the model after every epoch and instead save only upon completion of training. The recommendation involves setting the `save_strategy` to `"no"` in the training arguments or `config.yml`, and then manually setting up a save operation after training ends.
- **TinyLlama for Tight Spaces**: For fine-tuning with a smaller-than-7B model, the "TinyLlama-1.1B-Chat-v1.0" was suggested for fast iteration and experimentation. This model's configuration can be found in the `pretrain.yml` within the `examples/tiny-llama` directory in the Axolotl repository.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ae2df564-24d0-4c41-9f77-a8ea154566bb)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=eca9e87b-1d42-427c-8a91-59f42a3da0f8)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ccfe189d-d5fa-4308-9afe-8a86c48a0141)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1229697419386552383)** (4 messages): 

- **Fine-Tuning TinyLlama with Color Descriptions**: To fine-tune **TinyLlama** on a dataset with hex color codes and descriptions, one must first format the data with inputs as the color descriptions and target outputs as the color codes. Tokenization and formatting will follow using special tokens to concatenate inputs and outputs for the model's understanding.
- **Phorm Query Pending Resolution**: A question was raised about preprocessing data for model finetuning; **Phorm** was prompted to search the **OpenAccess-AI-Collective/axolotl** for answers. The response from Phorm with detailed steps is awaited.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=af0c71b5-451f-4893-8158-1dfa36a9a10b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1229426315875188776)** (89 messages🔥🔥): 

<ul>
  <li><strong>Limitless is the Next Whisper-Class Wearable**: The wearable previously known as **Rewind** is being rebranded to [**Limitless**](https://x.com/dsiroker/status/1779857843895599383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), with pricing details shared and a focus on consumer experience. Discussion circled around branding choices and the product’s ability to enable real-time applications and future-looking AI development.</li>
  <li><strong>Privacy Concerns Over Cloud Storage**: Some community members expressed strong opinions about using such devices only if fully local or end-to-end encrypted. They also discussed concerns over the "confidential cloud" and whether or not the service is HIPAA compliant.</li>
  <li><strong>Introducing Reka Core**: [**Reka Core**](https://x.com/rekaailabs/status/1779894622334189592?s=46&t=90xQ8sGy63D2OtiaoGJuww) was announced as a multimodal language model with video understanding capabilities. Members discussed the small team behind its creation and the impact on democratizing AI development.</li>
  <li><strong>Rollout of **Cohere Compass** Beta**: The Compass Beta is a multi-aspect data search system powered by a new embedding model named Compass, and beta testers are being recruited to help evaluate its capabilities.</li>
  <li><strong>Symbolic AI-Human Collaboration**: The launch of [**Payman AI**](https://www.paymanai.com/), a marketplace for AI agents to hire humans, was met with curiosity and speculation about the future of AI directing human tasks and its potential for data generation and AI training.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/RekaAILabs/status/1779894626083864873">Tweet from Reka (@RekaAILabs)</a>: Along with Core, we have published a technical report detailing the training, architecture, data, and evaluation for the Reka models.  https://publications.reka.ai/reka-core-tech-report.pdf</li><li><a href="https://www.paymanai.com/">Payman - Home</a>: no description found</li><li><a href="https://x.com/rekaailabs/status/1779894622334189592?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Reka (@RekaAILabs)</a>: Meet Reka Core, our best and most capable multimodal language model yet. 🔮  It’s been a busy few months training this model and we are glad to finally ship it! 💪  Core has a lot of capabilities, and...</li><li><a href="https://llm-price.com/">LLM Pricing - Compare Large Language Model Costs and Pricing</a>: no description found</li><li><a href="https://x.com/yitayml/status/1779895037335343521?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Yi Tay (@YiTayML)</a>: It&#39;s been a wild ride. Just 20 of us, burning through thousands of H100s over the past months, we&#39;re glad to finally share this with the world! 💪  One of the goals we’ve had when starting Rek...</li><li><a href="https://supabase.com/blog/ai-inference-now-available-in-supabase-edge-functions">AI Inference now available in Supabase Edge Functions</a>: Use embeddings and large language models on the edge with Supabase Edge Functions.</li><li><a href="https://x.com/ClementDelangue/status/1779925711991492760">Tweet from clem 🤗 (@ClementDelangue)</a>: It&#39;s the multi-modal week! After Grok 1.5 Vision (fully closed source), Reka core (closed-source + research paper), welcome to Idefics2 by @huggingface (fully open-source with public datasets) - e...</li><li><a href="https://x.com/lilianweng/status/1779914184874160170?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Lilian Weng (@lilianweng)</a>: 🎨Spent some time refactoring the 2021 post on diffusion model with new content: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ ⬇️ ⬇️ ⬇️ 🎬Then another short piece on diffusion video ...</li><li><a href="https://x.com/dsiroker/status/1779857843895599383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Dan Siroker (@dsiroker)</a>: Introducing Limitless: a personalized AI powered by what you’ve seen, said, or heard.  It’s a web app, Mac app, Windows app, and a wearable.  0:06 Reveal 0:48 Why Limitless? 1:39 Demo 3:05 Pendant 4:2...</li><li><a href="https://x.com/yoheinakajima/status/1780061516051755168?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Yohei (@yoheinakajima)</a>: A marketplace for AI agents to hire humans 🧠  ↘️ Quoting tyllen (@0xTyllen)   Excited to introduce a new project I&#39;ve been working on called Payman!    Payman is an AI Agent tool that gives Agent...</li><li><a href="https://x.com/suchenzang/status/1701747947191615697?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Susan Zhang (@suchenzang)</a>: MBPP might&#39;ve also been used somewhere in the Phi-1.5 dataset.  Just like we truncated one of the GSM8K problems, let&#39;s try truncating the MBPP prompts to see what Phi-1.5 will autocomplete wi...</li><li><a href="https://x.com/harrystebbings/status/1779973192397783301?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Harry Stebbings (@HarryStebbings)</a>: This is what they call the @sama effect.   Fastest ever download rate in 10 years of 20VC history.</li><li><a href="https://x.com/aidangomez/status/1779882113573044625?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Aidan Gomez (@aidangomez)</a>: Excited to announce the Compass Beta, a very powerful multi-aspect data search system powered by a new embedding model, Compass.  We&#39;re looking for help stress-testing the model&#39;s capabilities...</li><li><a href="https://x.com/harrystebbings/status/1779910559753802010?s=">Tweet from Harry Stebbings (@HarryStebbings)</a>: 10 years ago, I started 20VC in my bedroom with no money and no network.  Today we release our 20VC with @sama @bradlightcap and the fastest growing company in history @OpenAI.  The power of the inter...</li><li><a href="https://x.com/winglian/status/1779968341332860940?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Wing Lian (caseus) (@winglian)</a>: Alright, going old fashioned while we wait for the axolotl ai domain to transfer. Sign up for access to the private beta here. https://docs.google.com/forms/d/e/1FAIpQLSd0uWGZOwviIZPoOPOAaFDv3edcCXEIG...</li><li><a href="https://www.youtube.com/watch?v=xZiTSZ5SOYc&t=9s">Payman - Enabling AI Agent To Human Payments!</a>: Hey everybody, in this video, I&#39;m super excited to show you Payman, a platform that allows you to connect your agents with capital that they can use to pay h...</li><li><a href="https://llm.extractum.io/">LLM Explorer: A Curated Large Language Model Directory. LLM List. 35754 Open-Source Language Models.</a>: Browse 35754 open-source large and small language models conveniently grouped into various categories and llm lists complete with benchmarks and analytics.</li><li><a href="https://youtu.be/2-SPH9hIKT8?si=wqYrDbhvgJUT2zHP">A little guide to building Large Language Models in 2024</a>: A little guide through all you need to know to train a good performance large language model in 2024.This is an introduction talk with link to references for...</li><li><a href="https://huggingface.co/patrickvonplaten">patrickvonplaten (Patrick von Platen)</a>: no description found</li><li><a href="https://x.com/harrystebbings/status/1779910559753802010?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Harry Stebbings (@HarryStebbings)</a>: 10 years ago, I started 20VC in my bedroom with no money and no network.  Today we release our 20VC with @sama @bradlightcap and the fastest growing company in history @OpenAI.  The power of the inter...</li><li><a href="https://youtu.be/G8T1O81W96Y?si=OHJXeiI69YSOfG57">Sam Altman &amp; Brad Lightcap: Which Companies Will Be Steamrolled by OpenAI? | E1140</a>: Sam Altman is the CEO @ OpenAI, the company on a mission is to ensure that artificial general intelligence benefits all of humanity. OpenAI is one of the fas...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1229594211536474163)** (1 messages): 

- **Grants Galore for AI Gurus**: **Strong Compute** announces grants ranging from **$10k-$100k for AI researchers** needing access to High Performance Computing clusters. Eligible projects include exploring *trust in AI*, *post-transformer architectures*, *new training methods*, *Explainable AI*, and more ingenious topics applicants might propose.

- **GPU Bounty up for Grabs**: Applicants stand a chance to win up to ~100x 24GB Ampere GPUs, along with **~1+TB storage** per researcher. The total compute provided can translate to approximately 3,000-30,000 GPU hours depending on allocation and market dynamics.

- **Open Science, Big Prizes**: Researchers must commit to publishing a public access demo, with code and a data sample—or ideally, the full dataset. The aim is to have results ready for publication by the end of July.

- **Application Countdown**: Those interested should apply immediately through [Strong Compute research grants page](https://strongcompute.com/research-grants), as compute allocation decisions will be made by the **end of April**. Applicants can expect a response within two business days.

- **Inquiry Channels for Curious Minds**: For questions, researchers are encouraged to reach out to community members tagged in the announcement.

**Link mentioned**: <a href="https://strongcompute.com/research-grants">Research Grants</a>: no description found

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1229460702373216407)** (51 messages🔥): 

- **Join the Innovation Brainstorming**: A new channel has been introduced for brainstorming innovative uses of OpenInterpreter, bug bashing, and feature building.
- **Airchat Buzz Among OpenInterpreters**: Members are joining **Airchat**, a voice communication app, sharing handles like 'mikebird' and 'jackmielke', seeking invites, and discussing its features and usability.
- **Anticipation for 01 Project Updates**: Questions regarding the status and updates of the 01 project prompted responses about ongoing discussions with manufacturing services and anticipation for delivery timelines.
- **Open Source AI Takes the Spotlight**: Enthusiasm is shown for **WizardLm2**, an open source model compared to GPT-4, with excitement about the progress in AI and the availability of such models with open weights.
- **Personal AI Assistant Search**: Discussions revolved around the search for a personal AI assistant that offers speedy responses and integration of personal data for improved efficiency, with comparisons between products like the 01 and **Limitless AI**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=39998499">no title found</a>: no description found</li><li><a href="https://blog.wolfspyre.com/2022/november/read-me-when-creatively-struggling/">Read Me when you are creatively struggling</a>: An inspirational message sent to me by a friend, who found it years ago on $SocialMedia. Original Author unknown</li><li><a href="https://youtu.be/TitZV6k8zfA?t=900&si=zsI6zFfyJ8aBATzf).">The Worst Product I&#39;ve Ever Reviewed... For Now</a>: The Humane AI pin is... bad. Almost no one should buy it. Yet.MKBHD Merch: http://shop.MKBHD.comTech I&#39;m using right now: https://www.amazon.com/shop/MKBHDIn...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1229438032000843907)** (25 messages🔥): 

- **Pre-order Cancellation Query**: If you need to cancel an **01 pre-order**, simply send an email to help@openinterpreter.com.
- **Windows 11 Installation Blues**: Some custom PRs are out to assist with **01 installation on Windows 11**; however, they await merger. Meanwhile, it's noted that the software works but is tuned more for macOS.
- **Attention Linux Users**: For anyone using Linux, check out [rbrisita's GitHub branch](https://github.com/rbrisita/01/tree/linux) with all outstanding PRs for **01** merged. This branch is tested on Python 3.10 and 3.11.
- **Hardware Compatibility Discussions**: Users report various experiences with **01**—some using official Light setups, others opting for economical alternatives from AliExpress. The consensus is it's fine for a demo but expect differences in mic accuracy and hardware performance.
- **Minified 01 Design Endeavors**: Members are sharing updates on their custom **01 designs**—from cases that can be printed for free to battery optimizations with 18650 cells for longer life.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rbrisita/01/tree/linux">GitHub - rbrisita/01 at linux</a>: The open-source language model computer. Contribute to rbrisita/01 development by creating an account on GitHub.</li><li><a href="https://amzn.eu/d/4GbeU5b">no title found</a>: no description found</li><li><a href="https://amzn.eu/d/fIr3Lzu">no title found</a>: no description found</li><li><a href="https://amzn.eu/d/eZQoRwD">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1229820483818623010)** (1 messages): 

- **Revamped Documentation Structure**: LangChain is seeking feedback on a new documentation structure, which explicitly differentiates between tutorials, how-to guides, and conceptual guides. The structure aims to make it easier for users to find relevant information.
- **LangChain Framework Introduction**: The shared [documentation page](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction) provides a comprehensive introduction to **LangChain**, an open-source framework designed to streamline the application lifecycle of large language models (LLMs)—from development and productionization to deployment.

**Link mentioned**: <a href="https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction">Introduction | 🦜️🔗 LangChain</a>: LangChain is a framework for developing applications powered by large language models (LLMs).

  

---


**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1229368757856768062)** (43 messages🔥): 

- **Concurrent Execution in LangChain**: It is confirmed that you can run nodes in parallel in LangChain using the `RunnableParallel` class. Detailed usage examples are provided for both Python and JavaScript, with [Python documentation](https://api.js.langchain.com/classes/langchain_core_runnables.RunnableParallel.html) showing how multiple Runnables can be executed simultaneously, returning a map of the outputs.

- **Troubleshooting Azure AI Issues**: Members are seen discussing issues around using certain versions of LangChain and encountering problems with the neo4jVectorIndex and `faiss-cpu`. Specific advice includes downgrading LangChain to version 0.1.10 and trying out the `langchain_community` branch for resolving issues.

- **Role-Based Access Control (RBAC) and LangChain**: A conversation around implementing RBAC in large companies using LangChain involved suggestions like specifying specific files for retrieval based on a user's role and potentially denying answers with a prompt.

- **Finetuning Models for YC Application**: A member is interested in applying to YC with an idea on finetuning models for agents and is seeking information on whether this has been done before. They are informed of numerous startups such as Unsloth, mistral ai, and Holocene Intelligence already addressing this area.

- **Engagement and Collaboration Invitation**: Several invitations and offers for assistance and discussion were apparent, including offers to help with running nodes in parallel, issues with document splitting, personalizing recommendations, and short conversations on working with LLM applications.

**Link mentioned**: <a href="https://devanshus-organization.gitbook.io/llm-security">Safeguarding AI: Strategies and Solutions for LLM Protection | LLM Security</a>: Explore the security challenges and solutions of LLMs in this comprehensive guide. We cover potential risks, control mechanisms, and the latest tools for safer LLM application

  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1229808687632089131)** (1 messages): 

- **Integration Challenge with Nemo Guardrails**: A member inquired about successfully integrating **LangServe** with a chain that includes **Nemo Guardrails**, noting that a new output parser might be required due to structural changes introduced by Nemo. There's a request for advice or shared experience on this matter.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1229445653298413598)** (6 messages): 

- **Rag Chatbot Gets a Performance Boost**: The update to the [rag chatbot scripts](https://github.com/ossirytk/llama-cpp-chat-memory) introduces *multiprocessing for improved performance*, specifically in parsing text with Spacy for Chroma metadata keys.
- **Perplexica AI, the Open-Source Search Engine**: ItzCrazyKns announces [Perplexica](https://github.com/ItzCrazyKns/Perplexica/), an open-source AI-powered search engine with features like citations, image and video search, and multiple focus modes, positioned as an alternative to Perplexity AI.
- **Pay Humans with AI**: A tool called **Payman** is shared, allowing AI agents to pay humans to complete tasks beyond their abilities, using Langchain Custom Tools. Those interested can try the demo or sign up at [paymanai.com](https://www.youtube.com/watch?v=xZiTSZ5SOYc&t=5).
- **Free Premium AI Models Access with Galaxy AI**: GalaxyAI offers **free access to PREMIUM AI models** including GPT-4, GPT-3.5-turbo, and Claude-3-haiku with [OpenAI format API compatibility](https://galaxyapi.onrender.com).
- **AI-Powered Coding with OppyDev**: OppyDev introduces an **AI-assisted coding tool** that integrates agents like GPT-4 and Claude, featuring an IDE, a chat client, a project memory, and color-coded edits. A demo and full feature set are available at [oppydev.ai](https://oppydev.ai).
- **Rubiks AI Invites Beta Testers**: Rubiks AI is building an **advanced research assistant and search engine**, offering beta testers 2-months premium access with various models including GPT-4 Turbo and Mixtral-8x22B. Interested testers should visit [rubiks.ai](https://rubiks.ai) and can use the promo code `RUBIX`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: no description found</li><li><a href="https://oppydev.ai">Home - OppyDev</a>: Collaborative AI Agent that Elevates your Coding Experience</li><li><a href="https://www.youtube.com/watch?v=xZiTSZ5SOYc&t=5">Payman - Enabling AI Agent To Human Payments!</a>: Hey everybody, in this video, I&#39;m super excited to show you Payman, a platform that allows you to connect your agents with capital that they can use to pay h...</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://github.com/ossirytk/llama-cpp-chat-memory">GitHub - ossirytk/llama-cpp-chat-memory: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma</a>: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma - ossirytk/llama-cpp-chat-memory
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1229725236593692722)** (3 messages): 

- **Join The Club**: A member expressed interest in joining an ongoing project or discussion and requested a direct message for more details.
- **Learn about AI Agents with Long-term Memory**: A video was shared explaining how to **build self-improving AI agents with long-term memory**. Those interested can view it [here](https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek).

**Link mentioned**: <a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1229351746891939872)** (32 messages🔥): 

- **Cost Analysis of DIY GPU Cluster**: A cost comparison was made between a hypothetical array of six RTX 4090 GPUs and TinyBox configurations, highlighting a **36.04% price decrease** from the $15,000 TinyBox and a **61.624% price reduction** from the $25,000 option, with both setups having 144 GB of GPU RAM.

- **MNIST Handling in Tinygrad**: There are ongoing discussions about the ideal approach for handling MNIST dataset conversions within tinygrad, weighing options between updating examples to use tensor datasets, conversion upon calling, or reshaping before conversion.

- **Improving Tinygrad Documentation**: There is an acknowledgment of the need for more complete **public documentation** for tinygrad, with confirmation that it's an area being actively worked on for improvement.

- **Developer Experience is Key in Tinygrad**: Emphasis was placed on focusing on improving the developer experience for tinygrad, moving from version 0.9 to 1.0. Challenges with error comprehensibility were reported after replicating LLM.c training steps.

- **Efforts on MLPerf Plans and Codebase Management**: Updates were provided on MLPerf plans, including the progress on ResNet and a potential first run of UNet3D and BERT on TinyBox, with a decision made to raise the line limit for merging NV backends to **7,500 lines** to promote code inclusion and quality.

**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455">hotfix: bump line count to 7500 for NV backend · tinygrad/tinygrad@e14a9bc</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - hotfix: bump line count to 7500 for NV backend · tinygrad/tinygrad@e14a9bc

  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1229625901084315710)** (7 messages): 

- **BatchNorm Bug Hunt**: A user questioned the correctness of multiplying by `invstd` and then adding `bias` in **batchnorm** operations within tinygrad. **George Hotz** indicated that this could be a bug and requested a minimum test case to reproduce it.

- **Metal Compute Shaders Without Xcode**: **@taminka** inquired about resources for running a basic **Metal compute shader** program without Xcode, while testing tinygrad's shader generation. Another user recommended using **ChatGPT** to generate a Python script that utilizes PyObjC for dispatching Metal shader code.

- **From ONNX to WebGL/WebGPU**: **@spikedoanz** is looking for guidance on converting models from ONNX to WebGL/WebGPU, aiming to use tinygrad as an alternative to TensorFlow.js for better memory control. They shared a [Stable Diffusion WebGPU example](https://github.com/softwiredtech/stable-diffusion-webgpu) and asked about the feasibility of such a conversion using tinygrad's `extras.onnx` module.
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1229474088263880744)** (4 messages): 

- **Pile-T5 Unveiled**: EleutherAI has released **Pile-T5**, introduced in a detailed [blog post](https://blog.eleuther.ai/pile-t5/), marking another step in the advancement of language models.

- **Magic in WizardLM 2**: The **WizardLM 2** model, featuring foundation transformer technology, catches attention with its fresh release and a guide available at [WizardLM's page](https://wizardlm.github.io/WizardLM2/).

- **Reka Core Launch**: A new Encoder-Decoder model by Yi Tay, **Reka Core**, has been detailed in a technical report, which you can deep dive into [here](https://publications.reka.ai/reka-core-tech-report.pdf).

- **Introducing Idefics2**: The landscape of language model development gets richer with the introduction of Idefics2, as detailed on the [Hugging Face blog](https://huggingface.co/blog/idefics2).

- **Dolma Embraces Open Source**: In an exciting development, Dolma has been made open-source with an ODC-BY license, frosting an intense day of announcements in the open-source community.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1229465023680741446)** (29 messages🔥): 

- **Graphs Take Center Stage**: Enthusiasm was shown for well-crafted graphs in the recent newsletter, with plans to further refine them into a **Python library** for exploring new open models.
- **LLAMA 3 Packs a Punch with 30T Tokens**: Significant surprise and skepticism were expressed upon discovering that LLAMA 3 was trained on more than **30 trillion tokens**, hinting at potential complexities in evaluating its performance.
- **Mysterious Removal of WizardLM**: Concern and intrigue arose surrounding the sudden deletion of WizardLM's model weights and associated announcement post. The model seemed to disappear from official Microsoft collections, while a mirror of the model remains on [Hugging Face](https://huggingface.co/alpindale/WizardLM-2-8x22B).
- **WizardLM Controversy Spawns Apology**: WizardLM AI issued an apology for the premature release of its model, attributing the error to a missed **toxicity test** and promising a re-release soon.
- **Fate of WizardLM Weights and Code**: There was a flurry of activity regarding the availability of WizardLM model weights and code, with confirmation that they are still accessible, despite the official takedown, pointing towards a likely return.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/pimdewitte/status/1780066049263538653?s=46">Tweet from Pim de Witte (@PimDeWitte)</a>: something _really_ weird going on with @WizardLM_AI  1) model weights deleted 2) announcement post deleted 3) no longer in the &#34;microsoft&#34; collection 4) can&#39;t find any other evidence that ...</li><li><a href="https://fxtwitter.com/wizardlm_ai/status/1780101465950105775?s=46">Tweet from WizardLM (@WizardLM_AI)</a>: 🫡 We are sorry for that.  It’s been a while since we’ve released a model months ago😅, so we’re unfamiliar with the new release process now: We accidentally missed an item required in the model relea...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/)** (1 messages): 

natolambert: should I wizardLM 2 as a troll lol
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1229469203468259510)** (2 messages): 

- **Playtime with Bots**: A member mentioned **experimenting** with a bot and pondered whether they should wait longer rather than intervening manually.
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1229433917086634077)** (8 messages🔥): 

- **Machine Learning Enthusiast Seeks Dataset Annotation Opinions**: The conversation opens with an individual's preference for data annotation before model training, sharing their practice of understanding datasets before letting a model learn patterns and questioning if annotation still holds importance in the era of powerful LLMs.

- **Historical Records Team Balances Data and Models**: A team member responsible for extracting records from historical documents talks about their current practice, which involves gathering and curating data before using pre-LLM transformer models.

- **Frustration over Opaque LLM Demos**: One user expresses discontent with demos that don't provide open prompts, preferring transparency for a better understanding of how to get advantageous results from models without resorting to guesswork.

- **Indexing Task Hurdles with Model Refusals**: Users share challenges they face while indexing newspapers with the help of language models, especially when the model inconsistently adheres to privacy directives regarding the listing of names.

- **Exasperation with Model Inconsistencies**: The inconsistencies of the model's response to tasks related to indexing personal information from the 1930s newspapers are highlighted, alongside a note about contact with Anthropic regarding a model's refusal to share the information as instructed.
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1229617445115789353)** (22 messages🔥): 

- **Creating a Simplified LLM Web UI**: A member has created a simple web UI for browsing LLM log data, offering more intuitive interaction than Datasette for viewing logs and the goal of easier revisiting of past chats. The prototype was made with Streamlit, and the initial version is for log browsing only, as the APIs appear to lack easy retrieval of past conversations.

- **Prototype Showcase and Discussion**: The Streamlit-built UI prototype was shared in the chat, and another member expressed interest in knowing how it was built. The conversation expanded to discuss possibilities of integrating it as a Datasette plugin or as a standalone installable script.

- **Implementation and Tool Choice**: The creator described their experience as positive with Streamlit, a first-time use, though they initially considered building it as a plugin for either Datasette or LLM.

- **Future Development and Sharing**: While the current iteration is focused solely on browsing, it was mentioned that the app is around 200 lines and thus easy to work with. A GitHub gist link to the code was provided: [Initial LLM WebUI Gist](https://gist.github.com/RyanBalfanz/cd08b7402594fa91831bf8d54c76e0ec).

- **Understanding Discord Thread Mechanics**: There was a brief confusion from the web UI creator about the nature of threading in Discord and whether it operated more like Slack. Another member clarified how participation impacts thread visibility.

**Link mentioned**: <a href="https://gist.github.com/RyanBalfanz/cd08b7402594fa91831bf8d54c76e0ec">Initial LLM WebUI</a>: Initial LLM WebUI. GitHub Gist: instantly share code, notes, and snippets.

  

---



**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1229633278974234624)** (7 messages): 

- **WizardLM2 Collection Vanishes**: Discussion emerged around the disappearance of **WizardLM2** from Hugging Face, with a [collection update](https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a) showing that all models and posts had been deleted 22 hours ago, leaving the collection empty.
- **Legal Eagles May Be Circling**: Speculation suggests that the withdrawal of **WizardLM2** might involve legal intervention, without specifying details.
- **A Quest for Weights**: In the wake of the deletion, there was an inquiry about whether anyone had already downloaded the weights of the **WizardLM2** model.
- **Confirmation of Deletion with Screenshot**: A participant provided a screenshot confirming that **WizardLM2** was deleted due to a failure to test it properly, available at [this direct image link](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png?ex=66309ca9&is=661e27a9&hm=f105e6497796be9c414ade2024a27f9561caf0cad6cb06ba09f80e30b5e39ae4&).

**Link mentioned**: <a href="https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a">WizardLM - a microsoft Collection</a>: no description found

  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1229399893198639215)** (6 messages): 

- **In Search of the Elusive tokenizer.model**: A member is seeking advice on how to train a custom **Llama-tokenizer**, emphasizing the need to reduce embedding and output perceptron size for lightweight hardware compatibility. They are familiar with training **Hugging Face tokenizers** but have encountered difficulties in generating the required `tokenizer.model` file for **llama.cpp**.

- **Tips on Tokenizer Conversion**: Another member suggested consulting the Hugging Face Transformers library, specifically pointing out the [convert_slow_tokenizer.py script](https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534) that may aid in the conversion of a slow tokenizer.

- **Potential Shortcut for Custom Tokenizer**: To address the tokenizer training issue, a different member recommended the [llama.cpp convert.py script](https://github.com/ggerganov/llama.cpp/blob/master/convert.py) with a `--vocab-only` option as a possible solution for creating a custom tokenizer.

- **Quest for EU Copyright-Compliant Data**: A member has reached out to the community in search of EU text and multimodal data that is copyright permissive or free, with the intent to train a substantial open multimodal model.

- **Sources for EU Permissive License Data**: In response to the data hunt for copyright permissive or free EU multimodal data, one member suggests Wikipedia as a potential text source, while another highlights Wikicommons and [CC Search](https://search.creativecommons.org/) for multimodal data, albeit they may not offer extensive collections.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://search.creativecommons.org/">CC Search Portal</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/convert.py">llama.cpp/convert.py at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534">transformers/src/transformers/convert_slow_tokenizer.py at fe2d20d275d3591e2619a1adb0fa6ae272605208 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1229701118842503208)** (1 messages): 

- **Sampling Techniques in the Spotlight**: A member highlighted their interest in decoding strategies for language models, sharing a Reddit post with an image comparing sampling techniques. They discussed the paper "[A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/abs/2402.06925)" but expressed that open-ended tasks relevant to their experiences with LLMs were not adequately covered.

- **Advanced Sampling Methods Untouched by Academic Papers**: It was mentioned that modern sampling methods like MinP/DynaTemp/Quadratic Sampling, developed by [u/kindacognizant](/user/kindacognizant/), are not featured in academic papers despite their widespread use in various LLM frameworks. The member felt that these methods warrant more research attention.

- **Significant Boost in Creative Writing Performance**: The member shared their discovery of the considerable impact that min_p sampling parameters have on creative writing performance. The observed differences totaled an impressive +8 points in **alpaca-eval style elo** and +10 points in the **eq-bench creative writing test**.


**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/">Reddit - Dive into anything</a>: no description found

  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1229410199375188048)** (4 messages): 

- **Gen AI Scaling Secrets Revealed in NYC**: A meetup is announced for General AI enthusiasts at Gunderson Legal in New York City, focusing on **scaling Gen AI apps to production**. Interested participants can register [here](https://lu.ma/llms-in-prod-nyc) and the panel includes leaders from companies like [Portkey](https://portkey.ai/) and [Noetica](https://www.linkedin.com/in/yonisebag/).

- **Exploring Reka Core's Capabilities**: A [YouTube video](https://www.youtube.com/watch?v=vL1SayPCHBg) titled "Reka Core: A Frontier Class Multimodal Language Model" was shared, detailing how Reka Core competes with models from OpenAI, Anthropic, and Google.

- **Cost-Effective AI Model Outperforms Giants**: The [JetMoE-8B model](https://www.youtube.com/watch?v=Z9Hwp_XeS1A), trained with less than $0.1 million, reportedly surpasses the performance of LLaMA2-7B from Meta AI, despite Meta's significantly larger training resources.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=U7RbwPKyxs8">Reka Core: A Frontier Class Multimodal Language Model</a>: Reka Core is competitive with models from OpenAI, Anthropic, and Google across key industry-accepted evaluation metrics. Given its footprint and performance,...</li><li><a href="https://www.youtube.com/watch?v=Z9Hwp_XeS1A">JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>: JetMoE-8B is trained with less than $ 0.1 million1 cost but outperforms LLaMA2-7B from Meta AI, who has multi-billion-dollar training resources. LLM training...</li><li><a href="https://lu.ma/llms-in-prod-nyc">LLMs in Prod w/ Portkey, Flybridge VC, Noetica, LastMile · Luma</a>: Unlock the Secrets to Scaling Your Gen AI App to Production  While it&#x27;s easy to prototype a Gen AI app, bringing it to full-scale production is hard. We are…
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1229517997060653157)** (2 messages): 

- **Seeking Guides for Custom Model Packaging**: A user expressed interest in finding resources or guides on **how to package customized models into a llamafile** to assist several members in the community.
- **GitHub Resource for Docker Deployment**: A link to a **GitHub pull request** was shared, offering instructions on how to **build and publish a container to Docker Hub** using GitHub Actions, which could be relevant for those looking to package their models. [Publish container to Docker Hub](https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790).

**Link mentioned**: <a href="https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790">Publish container to Docker Hub by dzlab · Pull Request #59 · Mozilla-Ocho/llamafile</a>: Build and Publish container to Docker Hub on release using Github Actions #29 For this to work, need to setup the repository secrets:  DOCKER_HUB_USERNAME DOCKER_HUB_ACCESS_TOKEN

  

---



---



