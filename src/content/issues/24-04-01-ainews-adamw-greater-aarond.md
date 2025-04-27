---
id: 915ba3cf-0b17-448d-b34b-30237a2a8c52
title: AdamW -> AaronD?
date: '2024-04-01T19:58:53.959019Z'
original_slug: ainews-adamw-aarond
description: >-
  **Aaron Defazio** is gaining attention for proposing a potential tuning-free
  replacement of the long-standing **Adam optimizer**, showing promising
  experimental results across classic machine learning benchmarks like ImageNet
  ResNet-50 and CIFAR-10/100. On Reddit, **Claude 3 Opus** has surpassed all
  **OpenAI** models on the LMSys leaderboard, while a user pretrained a
  **LLaMA-based 300M** model outperforming **bert-large** on language modeling
  tasks with a modest budget. The new **MambaMixer** architecture demonstrates
  promising results in vision and time series forecasting. In image generation,
  **Stable Diffusion 1.5** with LoRAs achieves realistic outputs, and the
  **WDXL** release showcases impressive capabilities. AI applications include an
  AI-generated Nike spec ad and a chatbot built with OpenAI models that may
  resist prompt injections. OpenAI is reportedly planning a ban wave targeting
  policy violators and jailbreak users. *"The high alpha seems to come from
  Aaron Defazio,"* highlighting his impactful work in optimizer research.
companies:
  - openai
  - hugging-face
models:
  - claude-3-opus
  - llama-3
  - llama-3-300m
  - bert-large
  - stable-diffusion-1.5
  - wdxl
topics:
  - optimizer
  - machine-learning-benchmarks
  - vision
  - time-series-forecasting
  - image-generation
  - prompt-injection
  - policy-enforcement
people:
  - aaron-defazio
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/28/2024-4/1/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **26** Discords (**381** channels, and **10260** messages) for you. Estimated reading time saved (at 200wpm): **1099 minutes**.

It's a quiet Easter weekend and April Fools' is making it harder than normal to sift signal from noise (our contribution [here](https://twitter.com/swyx/status/1774876112029626492)). We do recommend sifting through [Sequoia Ascent's playlist](https://www.youtube.com/watch?v=TDPqt7ONUCY&list=PLOhHNjZItNnOoPxOF3dmq30UxYqFuxXKn), if you're not close to each speaker's work (for example Andrew Ng mostly repeated [the writeup we covered last week](https://buttondown.email/ainews/archive/ainews-andrew-likes-agents/)), which is now fully released.

Over in Twitter land, the high alpha seems to come from [Aaron Defazio](https://twitter.com/aaron_defazio), which several of our AI High Signal follows highlighted as [the "new LK-99"](https://x.com/abhi_venigalla/status/1773839199025955030?s=20) for engaging, "impossible" work in public. What's at stake: a potential tuning-free replacement of the very long lived [Adam optimizer](https://arxiv.org/abs/1711.05101v3), and experimental results are currently showing learning at a [Pareto frontier in a single run](https://x.com/aaron_defazio/status/1773756384749691215?s=20) for basically every classic machine learning benchmark (ImageNet ResNet-50, CIFAR-10/100, MLCommons AlgoPerf):

 ![image.png](https://assets.buttondown.email/images/7994202e-3bb5-411c-b17b-c299e7a968e0.png?w=960&fit=max) 

He's writing the paper now, and many "better optimizers" have come and gone, but he is well aware of the literature and going for it. We'll see soon enough in a matter of months.


---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling still not implemented but coming soon.

**AI Models and Performance**

- **Claude 3 Opus overtakes OpenAI models**: In /r/singularity, Claude 3 Opus has overtaken all OpenAI models on the [LMSys leaderboard](https://i.redd.it/idxt2es3vmrc1.png), showing impressive performance.
- **User pretrains LLaMA-based 300M LLM**: In /r/LocalLLaMA, a user [pretrained a LLaMA-based 300M LLM](https://www.reddit.com/r/LocalLLaMA/comments/1bs5cgd/i_pretrained_a_llamabased_300m_llm_and_it/) that outperformed bert-large for lm-evaluation-harness tasks, using a $500 budget and 4 x 4090 GPUs from vast.ai.
- **MambaMixer architecture shows promising results**: In /r/MachineLearning, [MambaMixer](https://arxiv.org/abs/2403.19888), a new architecture with data-dependent weights using a dual selection mechanism across tokens and channels, shows promising results in vision and time series forecasting tasks.

**Stable Diffusion and Image Generation**

- **Realistic results with SD1.5 and LoRAs**: In /r/StableDiffusion, a user achieved [good realism using SD1.5 and LoRAs](https://www.reddit.com/gallery/1bst8wd), even passing facecheck.id's AI detection.
- **WDXL release showcases impressive capabilities**: In /r/StableDiffusion, the [WDXL release](https://huggingface.co/spaces/waifu-diffusion/wdxl-demo) showcases impressive image generation capabilities.
- **Tips and tricks for Stable Diffusion**: In /r/StableDiffusion, users share tips and tricks such as [base prompts for realistic SDXL renders](https://www.reddit.com/r/StableDiffusion/comments/1bsr82y/base_prompts_for_sdxl_realistic_renders/), [colouring in with AI](https://v.redd.it/k0qoibx8ksrc1), and [creating custom Stardew Valley player portraits](https://i.redd.it/iewhvetpqrrc1.jpeg).

**AI Applications and Demos**

- **AI-generated Nike spec ad**: In /r/MediaSynthesis, an AI-generated [Nike spec ad](https://v.redd.it/sk3u7302rprc1) showcases the potential of AI in advertising and creative fields.
- **AI engineer beginner project on agentic behavior**: In /r/artificial, a user shares an [AI engineer beginner project on agentic behavior](https://v.redd.it/g3pdjq2m7trc1), demonstrating practical applications of AI.
- **Chatbot using OpenAI potentially immune to prompt injections**: In /r/OpenAI, a user made a [chatbot using OpenAI that is potentially immune to prompt injections](https://www.reddit.com/r/OpenAI/comments/1bspl0t/i_made_a_chatbot_using_openai_that_i_think_is/), inviting others to test its robustness.

**AI Ethics and Policies**

- **OpenAI planning ban wave for policy violations**: In /r/OpenAI, OpenAI is reportedly planning a [huge ban wave for users who violated content policies or used jailbreaks](https://www.reddit.com/r/OpenAI/comments/1bswvwg/openal_is_planning_a_huge_ban_wave_where_everyone/).  
- **Discussion on AI believing AI-generated imagery is reality**: In /r/OpenAI, a discussion emerges on [whether AI will eventually believe AI-generated imagery is reality](https://www.reddit.com/r/OpenAI/comments/1bsu5sg/will_there_be_a_point_where_ai_believes_ai/), given the increasing amount of generated content in training data.
- **OpenAI partnership with G42 in UAE**: In /r/OpenAI, OpenAI's [partnership with G42 in the UAE](https://www.reddit.com/r/OpenAI/comments/1bs9l2s/openai_relationships/) aims to expand AI capabilities in the region, with CEO Sam Altman envisioning the UAE as a potential global AI sandbox.

**Memes and Humor**

- **Bill Burr's humorous take on AI**: In /r/singularity, Bill Burr shares his humorous take on AI in a [popular video post](https://v.redd.it/n2y3pwrmwsrc1).
- **User experiences "brain stroke" while interacting with AI**: In /r/singularity, a user experiences a ["brain stroke" while interacting with an AI](https://i.redd.it/cpb8tnec6qrc1.png), likely due to unexpected or nonsensical outputs.
- **User gets roasted while testing prompt jailbreak**: In /r/LocalLLaMA, a user [gets roasted while testing prompt jailbreak](https://www.reddit.com/r/LocalLLaMA/comments/1bsw1p5/i_just_got_roasted_when_testing_prompt_jailbreak/), showcasing the witty and sometimes snarky responses of AI.

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Capabilities and Limitations**

- **Limitations of current AI systems**: [@fchollet](https://twitter.com/fchollet/status/1774477537407824072) noted that memorization, which ML has solely focused on, is not intelligence. Any task that does not involve significant novelty and uncertainty can be solved via memorization, but *skill* is never a sign of intelligence. [@fchollet](https://twitter.com/fchollet/status/1774478847490376168) shared a paper introducing a formal definition of intelligence and benchmark, noting that current state-of-the-art LLMs like Gemini Ultra, Claude 3, or GPT-4 are not able to score higher than a few percents on that benchmark.
- **Limitations of benchmarks in assessing AI capabilities**: [@_akhaliq](https://twitter.com/_akhaliq/status/1774669369869508743) questioned if we are on the right way for evaluating large vision-language models (LVLMs). They identified **two primary issues in current benchmarks: visual content being unnecessary for many samples and unintentional data leakage in training**.
- **Potential of AI systems**: [@hardmaru](https://twitter.com/hardmaru/status/1774488363816652805) shared a paper noting that collective intelligence is not only the province of groups of animals, and that an important symmetry exists between the behavioral science of swarms and the competencies of cells and other biological systems at different scales.

**AI Development and Deployment**

- **Mojo 🔥 programming language**: [@svpino](https://twitter.com/svpino/status/1774406305148805525) noted that Mojo 🔥, the programming language that turns Python into a beast, went open-source. It allows writing Python code or scaling all the way down to metal code.
- **Claude 3 beating GPT-4**: [@svpino](https://twitter.com/svpino/status/1774406308759994533) reported that Claude 3 is the best model in the market right now, overtaking GPT-4. Claude 3 Opus is #1 in the Arena Leaderboard, beating GPT-4.
- **Microsoft and OpenAI's $100B supercomputer**: [@svpino](https://twitter.com/svpino/status/1774406312530731156) shared that Microsoft and OpenAI are working on a $100 billion supercomputer called "Stargate", expected to be ready by 2028. The report mentions "proprietary chips".
- **Dolphin-2.8-mistral-7b-v0.2 model release**: [@erhartford](https://twitter.com/erhartford/status/1774539935355396582) announced the release of Dolphin-2.8-mistral-7b-v0.2, trained on @MistralAI's new v0.2 base model with 32k context, sponsored by @CrusoeCloud, @WinsonDabbles, and @abacusai.
- **Google's Gecko embeddings**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1774611474364739742) reported that Google presents Gecko, versatile text embeddings distilled from large language models. Gecko with 768 embedding dimensions competes with 7x larger models and 5x higher dimensional embeddings.
- **Apple's ReALM for reference resolution**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1774611099251290418) shared that Apple presents ReALM: Reference Resolution As Language Modeling.
- **Huawei's DiJiang for efficient LLMs**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1774631022325350727) reported that Huawei presents DiJiang: Efficient Large Language Models through Compact Kernelization, achieving comparable performance with LLaMA2-7B while requiring only about 1/50 pretraining cost.

**AI Applications and Use Cases**

- **Building a RAG application**: [@svpino](https://twitter.com/svpino/status/1774496892095009223) recorded a 50-minute YouTube tutorial on how to evaluate a RAG application, building everything from scratch with the goal of learning, not memorizing.
- **Generating consistent characters in AI images**: [@chaseleantj](https://twitter.com/chaseleantj/status/1774392960018481299) shared a great way to create consistent characters in AI images, allowing telling an entire story about a character in any style and pose.
- **Building a perplexity style LLM answer engine**: [@LangChainAI](https://twitter.com/LangChainAI/status/1774502671669501973) highlighted a repo taking off, providing a great introduction to building an answer engine from scratch.
- **Fine-tuning a Warren Buffett LLM**: [@virattt](https://twitter.com/virattt/status/1774522192081904023) shared an update on fine-tuning a Warren Buffett LLM by generating a question-answer dataset using Berkshire's 2023 annual letter. The next step is to generate datasets for all letters from 1965 to 2023 before fine-tuning the LLM.
- **Ragdoll for building personalized AI assistants**: [@llama_index](https://twitter.com/llama_index/status/1774619392137138350) featured Ragdoll and Ragdoll Studio, a library and web app for building AI Personas based on a character, web page, or game NPC. It uses @llama_index under the hood and is powered by local LLMs with image generation built-in.

**AI Ethics and Safety**

- **Potential of AI sentience**: [@AISafetyMemes](https://twitter.com/AISafetyMemes/status/1774353231143215511) shared a conversation with Claude, an AI assistant, discussing the potential of AI sentience and sapience. Claude argued that the fact that it can reflect on its own nature, grapple with existential doubts, and strive to articulate a coherent metaphysical and ethical worldview is evidence of something more than mere shallow mimicry at work.
- **Ethical considerations in AI development**: [@AmandaAskell](https://twitter.com/AmandaAskell/status/1774617427609063530) noted that even if AIs lack moral status, we may have indirect duties towards them, similar to animals. By lying or being cruel to an AI, we indulge in bad moral habits and increase the likelihood of treating humans in the same way.

**Memes and Humor**

- **Humorous take on AI capabilities**: [@francoisfleuret](https://twitter.com/francoisfleuret/status/1774536209697571194) joked that "Aircraft made of metal lacks the lighter-than-air material to fly, hot air experts say."
- **Meme about AI safety concerns**: [@AISafetyMemes](https://twitter.com/AISafetyMemes/status/1774668947708875028) shared a meme quoting Bill Barr on AI: "Do these fucking things have goals?" and "How many sci-fi movies do you need to see to realize where this is going?"
- **Joke about AI-generated content**: [@nearcyan](https://twitter.com/nearcyan/status/1774641127146098722) joked that "basically half of twitter is one guy saying ∃x : y and then everyone quote tweeting them with BUT ¬(∀x : y)!!!!"

---

# AI Discords

> A summary of Summaries of Summaries

- **Stable Diffusion 3 Anticipation and UI Concerns**: Discussions in the Stability.ai Discord centered around the potential release date of **Stable Diffusion 3 (SD3)**, with some suspecting an April Fools' prank. Users also expressed frustration with the new **inpainting UI** in SD, finding it unintuitive. Creative applications of AI, like reimagining video game footage and comic creation workflows, were explored.

- **AI Model Comparisons and Inconsistencies**: In the Perplexity AI Discord, the **Claude 3 Opus** model exhibited inconsistent performance on certain question types, sparking comparisons with other models like **Haiku** and **Gemini 1.5 Pro**. Discussions also touched on Perplexity's partnership process, API features, and pricing ([Perplexity Pricing](https://docs.perplexity.ai/docs/pricing), [OpenAI Pricing](https://openai.com/pricing)).

- **Hardware Benchmarks and Fine-Tuning Strategies**: The Unsloth AI Discord featured the impressive benchmarks of Qualcomm's **Snapdragon Elite X** chip ([YouTube video](https://youtu.be/dTCm6BupWEQ)), fine-tuning strategies using **Unsloth** ([manual GGUF guide](https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf)), and a new Chinese AI processor from [Intellifusion](https://www.icsmart.cn/75486/) that could be cost-effective for inference. The successful implementation of **Unsloth + ORPO** alignment in LLaMA Factory was also praised ([ORPO paper](https://arxiv.org/abs/2403.07691)).

- **Jamba Model Unveiling and BitNet Validation**: AI21 Labs' **Jamba** model, a hybrid SSM-Transformer, generated buzz in the Nous Research AI Discord ([AI21's blog post](https://www.ai21.com/blog/announcing-jamba)). NousResearch also validated the claims of the **BitNet** paper with a reproduced 1B model ([Hugging Face repo](https://huggingface.co/NousResearch/OLMo-Bitnet-1B)). Discussions touched on the merits of single vs. multiple RAG setups and the challenges of PII anonymization in models like Hermes mistral 7b.

- **LM Studio Updates Spark GPU Discussions**: The LM Studio Discord saw queries about the JSON output format for application development, feature requests for a plugin system and remote GPU support, and troubleshooting of GPU issues post-update, including missing **GPU Acceleration options** and **unrecognized VRAM**. Fine-tuning and hardware compatibility were also hot topics.

- **Voice Synthesis Breakthroughs and Ethical Concerns**: OpenAI's [Voice Engine](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices) and the open-source **VoiceCraft** ([GitHub repo](https://github.com/jasonppy/VoiceCraft), [demo](https://jasonppy.github.io/VoiceCraft_web/)) sparked discussions in the OpenAI Discord about the rapid advancements in speech synthesis and the potential for misuse. The choice between OpenAI's APIs for various tasks was also debated.

- **1-Bit and Ternary LLMs Spark Skepticism**: The Eleuther Discord featured skepticism and reproducibility attempts surrounding **1-bit and ternary quantized LLMs**, often marketed as "1.58 bits per parameter" ([BitNet b1.58 model](https://huggingface.co/1bitLLM/bitnet_b1_58-3B)). The effectiveness of the **Frechet Inception Distance (FID)** metric for evaluating image generation was questioned ([alternative metric proposal](https://arxiv.org/abs/2401.09603v2)), and anticipation built for a new optimization technique from Meta.

- **DBRX Integration and V-JEPA for Video Lava**: The LAION Discord discussed the challenges of integrating **DBRX** into axolotl ([pull request #1462](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1462)), the potential of **V-JEPA embeddings** for enhancing video Lava ([V-JEPA GitHub](https://github.com/facebookresearch/jepa)), and new approaches in diffusion and embedding models ([Pseudo-Huber Loss paper](https://arxiv.org/abs/2403.16728), [Gecko paper](https://arxiv.org/abs/2403.20327)).

- **Hugging Face Introduces 1-Bit Model Weights**: Hugging Face released **1.38 bit quantized model weights** for large language models (LLMs), a step towards more efficient AI ([1bitLLM](https://huggingface.co/1bitLLM)). The community also discussed the **Perturbed-Attention Guidance (PAG)** method for improving sample quality without reducing diversity ([PAG paper](https://arxiv.org/abs/2403.17377)) and real-time video generation using **1 step diffusion** with sdxl-turbo ([Twitter post with video snippets](https://twitter.com/Dan50412374/status/1774527643058331980)).

- **LlamaIndex Enhancements and RAFT Dataset Generation**: The LlamaIndex Discord featured a webinar announcement on **Retrieval-Augmented Fine-Tuning (RAFT)** with the technique's co-authors ([sign-up link](https://lu.ma/v1bdat63)), guides on building reflective RAG systems and using LlamaParse for complex documents ([Twitter thread](https://twitter.com/llama_index/status/1773783011785585141)), and the introduction of **RAFTDatasetPack** for generating datasets for RAFT ([GitHub notebook](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb)). Troubleshooting discussions revolved around handling oversized data chunks with **SemanticSplitterNodeParser** and outdated documentation.

- **OpenRouter Introduces App Rankings and DBRX**: OpenRouter launched **App Rankings for Models**, allowing insights into top public apps using specific models ([Claude 3 Opus App Rankings](https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps)). Databricks' **DBRX 132B** model was also added, boasting superior performance over models like Mixtral ([DBRX Instruct page](https://openrouter.ai/models/databricks/dbrx-instruct)).

- **Mojo's Open-Source Excitement and Challenges**: The Modular Discord buzzed with the news of **Mojo's standard library going open-source** ([blog post](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source), [GitHub repo](https://github.com/modularml/mojo/tree/nightly)), though limitations on non-internal/commercial applications tempered enthusiasm. Installation challenges, the need for better profiling tools, and the potential of Mojo's multithreading and parallelization were discussed.

- **Interconnects Preserves Open Alignment History**: Nathan Lambert announced an initiative in the Interconnects Discord to document the evolution of **open alignment techniques** post-ChatGPT, including the reproduction rush and the DPO vs. IPO debate ([Lambert's Notion Notes](https://magnetic-share-282.notion.site/History-of-Open-Alignment-a7ef20aefb34438185336df68147809e?pvs=4)). The **stepwise DPO (sDPO)** method was also highlighted as a potential democratizer of performance gains in model training ([sDPO paper](https://arxiv.org/abs/2403.19270)).

- **Jamba's Performance Puzzle**: The AI21 Labs Discord pondered **Jamba's** performance on code tasks and the HumanEval benchmark, its language inclusivity, and the potential for fine-tuning on AI21 Studio.

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Inpainting Frustration**: Engineers voice frustration with Stable Diffusion's (SD) new **inpainting UI**; the layout challenges efficiency and intuition.
- **April Fools' AI Shenanigans**: Discussions suggest CivitAI has integrated playful features like "ODOR" models and "chadgpt" alerts for April Fools'—reactions are mixed between amusement and confusion.
- **Stable Diffusion 3 Anticipation**: Debate heats up about the release date of **Stable Diffusion 3 (SD3)**, with users oscillating between eager anticipation and suspicions of a release date prank.
- **AI Technical Support Squad**: Members actively seek technical assistance with AI tools spanning ControlNet setup in Colab, using Comfy UI and rendering architecture with SD, hinting at a demand for a more centralized knowledge hub or support system.
- **Creative AI Futurism**: Ideas circulate on leveraging AI for creative output, such as reimagining video game footage in AI-driven films and integrating AI into comic creation workflows, prompting a discussion on the evolution of content creation.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Claude's Classroom Conundrum**: The **Claude 3 Opus** model exhibits inconsistent performance with questions related to identifying remaining books in a room, with some models not producing correct answers despite several adjustments to prompts.

**AI Model Melee**: Engineers discussed **AI model benchmarks**, focusing on the comparative performance of **Haiku**, **Gemini 1.5 Pro**, and **Claude Opus**. The conversations highlighted differing strengths and functionalities but did not lean towards consensus on a superior model.

**Pondering Partnerships and API Puzzles**: For partnership interests with Perplexity, engineers are instructed to email **support@perplexity.ai**, and seeking details about the API's source citation feature can be directed to Perplexity's [Typeform](https://perplexity.typeform.com/to/j50rnNiB). Additionally, "pplx-70b-online" model support is deprecated, and the alias concerns are culminating in suggesting an update to **Perplexity's Supported Models** documentation.

**Credit Where Credit's Due**: Reports of issues with **credit purchases** on Perplexity surfaced, hinting at potential complications with transaction systems or third-party security features like those implemented by **Stripe**. Member discussion advised for situational troubleshooting and inquired about further inspection.

**Search Spectacles and Query Quirks**: Engineers displayed a broad array of interests from **Bohmian mechanics to Hyperloop** through shared queries on Perplexity AI, but user-contributed informational threads lacked documentation support for their extendibility and shareability.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Snapdragon Makes Waves**: Qualcomm's Snapdragon Elite X Arm chip has impressed engineers with its 45 TOPs performance, leading to discussions about its cost-efficiency and comparisons with other chips like the Tesla T4's 65 TFLOPs of float16. The excitement was fueled by a [YouTube video](https://youtu.be/dTCm6BupWEQ) detailing the chip's benchmarks.

**Model Training Optimized with Unsloth**: Fine-tuning **Mistral** models with Unsloth AI can encounter dependency issues, but the [Unsloth GitHub repository](http://github.com/unslothai/unsloth) offers a Docker solution and a [manual GGUF guide](https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf). Moreover, discussions suggest single GPU training is possible by setting `os.environ["CUDA_VISIBLE_DEVICES"]`, although multi-GPU support is a potential future development.

**AI Hardware Announcements Catch Attention**: **Intellifusion's** new AI processor could be a game-changer for inference operations due to its cost-effectiveness, raising curiosity about its potential in training scenarios. Details can be found on [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus).

**Fine-Tuning Techniques Under Scrutiny**: Engineers debate fine-tuning methods like **QLora 4bit** versus SFT/pretraining, discussing how the quantization process might affect performance. There's also talk about the paradox of dataset size in model training, where quality, not just quantity, determines the effectiveness.

**ORPO Integration Sparks Commendation**: The Unsloth + ORPO (Orthogonal Projection for Language Models Alignment) combination has been implemented effectively in LLaMA Factory, according to a [paper on arXiv](https://arxiv.org/abs/2403.07691). The AI community shared success stories and optimizations, acknowledging particular efficacy in training with limited data samples.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**StyleGAN Gets a Fashion Makeover**: When training StyleGAN2-ada with various fashion images, users inquired about the need for script modifications but did not mention outcomes or specify details on solutions.

**Learners Take Flight with ML/AI Courses**: For those charting a course into machine learning, particularly from other fields like aerospace, the community recommended starting with the foundational [fastai courses](https://course.fast.ai/), and moving toward specialized courses like the [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter1/1) for a deep dive into language models and transformers.

**Microsoft's Ternary LLM Paper Replicated**: Results from a Microsoft paper on ternary Large Language Models, especially concerning the 3 billion parameter models at 100 billion operations, have been successfully replicated, as evidenced by the model [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) on Hugging Face.

**Nous Research Amplifies LLM Discussion with a Tweet**: Nous Research fueled the conversation around LLMs with a [twitter post](https://twitter.com/NousResearch/status/1773923241268003052), though the content of the announcement was not detailed in the messages.

**Privacy Detection Dilemma**: Hermes mistral 7b's difficulties in anonymizing PII sparked debate on how to enhance the model's capabilities. There was a mention of upcoming data integrations by NousResearch and models that may aid in improvement, such as [open-llama-3b-v2-pii-transform](https://huggingface.co/filipealmeida/open-llama-3b-v2-pii-transform).

**Opinions Split on RAG Configurations**: The community discussed the merits of using a single large RAG versus multiple specialized RAGs. While specific approaches or results were not mentioned, the conversation touched on the importance of metadata and the idea of integrating RAG with other tools to bolster functionality.

**OpenSim Engages Philosophical and Practical Domains**: Users debated the economic aspects of token output costs in LLM apps, explored the concept of "Hyperstition" within AI interactivity, and expressed desire for new features in WorldSim, like saving chat sessions with URLs for sharing.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**JSON Outputs Draw Developer Attention**: AI engineers show interest in **LMStudio's JSON output format** for the development of practical applications. Seamless integration with **langchain** has been reported, making the process incredibly efficient.

**Plugin Possibilities Percolate in LM Studio**: The community calls for **plugin support** within LM Studio for expandability, while feature requests such as a *Unified Settings Menu* and *Keyboard Shortcuts* indicate a desire for a more customizable and efficient user interface.

**Apple Silicon Users Adapt and Overcome**: LLM users report challenges when running models on **Apple Silicon M1 Macs**, offering shared solutions like shutting down other apps to free up memory and exploring **LoRA adaptation** interfaces.

**GPUs Under the Microscope after LM Studio Update**: Post-update GPU issues with LM Studio, including **disappearing GPU Acceleration options** and **unrecognized VRAM**, catalyzes conversations around navigating hardware compatibility, multi-GPU setups, and memory usage.

**Remote GPU Support Requested for Power Users**: AI Engineers express interest in **remote GPU support** for LM Studio, noting parallels to services allowing remote gaming, and ask for open-source initiatives considering the community's emphasis on privacy and security.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Voice Tech Marches On**: OpenAI's [Voice Engine](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices) can now generate natural speech just from text and a 15-second voice sample, though they're proceeding with caution to mitigate misuse risks. Simultaneously, OpenAI removed the signup barrier for [ChatGPT](https://openai.com/blog/start-using-chatgpt-instantly), allowing instant AI engagement worldwide.

**Prompt Engineering Reveals Tech Quirks**: Some members experience difficulties when transferring LaTeX equations from ChatGPT to Microsoft Word, whilst others discussed nuanced AI approaches like **meta-prompting** and observed unusual behaviors in roleplaying scenarios with the **gpt-4-0125-preview** model.

**VoiceCraft's New Frontier**: [VoiceCraft's GitHub repo](https://github.com/jasonppy/VoiceCraft) and its [accompanying demo](https://jasonppy.github.io/VoiceCraft_web/) highlight its speech editing and text-to-speech prowess, igniting discussions around the ethics of voice cloning and potential for misuse.

**Choosing the Right AI Tools for Business Insights**: In the tech community, there's uncertainty about whether to use the completion API or the assistant API for tasks like summarizing business data and generating quizzes, with ChatGPT format controls suggested as a deciding factor ([API context management](https://platform.openai.com/docs/assistants/how-it-works/context-window-management)).

**Model Mix-Up Clarified**: Discussions clarified that ChatGPT is not an AI model itself, but an application that uses GPT models. Additionally, debates blossomed around the usage and limitations of Custom GPT and how developers might interface with GPT API directly for projects like automated video content management.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Apocalypse: Still a Chuckle, Not a Priority**: In a lighthearted debate, the community estimated the risk of AI going rogue at an average concern level of 3.2 out of infinity, indicating a humorous but cautious stance on the subject.
  
- **Grammar Nerds Assemble**: An intricate discussion on the proper usage of "axis" led to resource sharing, like [Grammar Monster's explanation](https://www.grammar-monster.com/plurals/plural_of_axis.htm) of the word's grammatical nuances.

- **Human or Not Human, That is the AI Question**: A spirited conversation raised questions about AI reaching human-level intelligence, intertwining hardware progress with Moore's Law and the critical need for AI alignment to ease societal integration.

- **Peering Through the Hype of AI Papers**: There's keen interest and healthy skepticism over recent AI papers; the discussions mentioned the promise and doubts around adding more AI agents, with a side-eye towards optimistic forecasts from figures like Andrew Ng.

- **Dubious Repositories Raise Eyebrows**: GitHub projects by Kye Gomez came under the microscope, prompting contemplation on their impacts on the scientific process and reproducibility.

- **MoE with mismatched expert sizes sparks debate**: The guild dissected **Mixture of Experts (MoE)** models with heterogeneous expert sizes; the gap between theory and reported performance has generated conflicting views.

- **Questions Raised Over BitNet b1.58 Validity**: NousResearch's reproducibility attempt on BitNet b1.58 raised questions about its efficiency claims, found in detail on their [Hugging Face repo](https://huggingface.co/NousResearch/OLMo-Bitnet-1B), compared to FP16 counterparts.

- **Is FID the Right Yardstick?**: Concerns about Frechet Inception Distance's accuracy prompted researchers to seek better measures for evaluating image generation, as highlighted in an [alternative metric proposal](https://arxiv.org/abs/2401.09603v2).

- **Excitement Building for Meta's Optimization Mystery**: Anticipation is brewing over a teased new optimization technique from Meta, claimed to outdo Adam with zero memory overhead, challenging current optimization paradigms.

- **Tuning For Precision**: Dialogue on starting models for SFT on TLDR text summarization showcased an exchange of insights, focusing on models like Pythia against the backdrop of resource limits and performance.

- **Keep Your Logits in Check**: Exchanges in the guild clarified that tweaking of logits occurs before every softmax within the network, addressing both attention and the final head in anticipation of decision making.

- **Softmax Function: A Refresher Course Needed**: A temporary forgetfulness about softmax functions was met with a supportive correction, demonstrating the community's spirit of knowledge-sharing and camaraderie.

- **Sparse Autoencoders under the Microscope**: A new issue with Sparse Autoencoders (SAEs) was unearthed where reconstruction errors can unduly sway model predictions, detailed in a [Short research post](https://x.com/wesg52/status/1773756298531918268).

- **Visualizing the Invisible**: A novel visualization library for SAE has been introduced, aiding researchers in understanding Sparse Autoencoder's features, announced in [SAE Vis Announcement Post](https://x.com/neelnanda5/status/1774463606656282806).

- **Deciphering SAE's Features**: A post sharing insights into Sparse Autoencoder features led to a discussion on their significances, particularly regarding AI alignment and feature interpretation, found in this [LessWrong interpretation](https://www.lesswrong.com/posts/BK8AMsNHqFcdG8dvt/a-selection-of-randomly-selected-sae-features-1).

- **Model Loading Mastery and Enhancement**: A DBRX model loading issue in lm-eval harness prompted an individual to troubleshoot successfully by updating to nodes with adequate GPUs, while a new [pull request for lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1571) aims to refine handling of context-based tasks.

- **Global Batch Size Balancing Act in NeoX**: Discussions in NeoX development unearthed the intricacies of setting a global batch size that doesn't align with the GPU count, revealing potential load imbalance and GPU capacity bottlenecks.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**DBRX Base Hits Home Run**: A non-gated re-upload of the **DBRX Base model**, notable for its mixture-of-experts architecture, reiterates the community's push for open weights and ungatekeeped access. The original models can be explored on [Hugging Face](https://huggingface.co/Undi95/dbrx-base).

**Euler Method Proves Its Worth**: Anecdotal evidence suggests that using the euler ancestral method optimizes results on terminus, backed by amusing examples of precise Chinese translations.

**AI's Music Maestros Dissect Suno**: Discussing **AI music generation tools**, particularly Suno's v2 vs v3, the community shared concerns about noise in voice generation and the potential leap v4 could bring.

**Voice Synthesis Under the Microscope**: Voices in the guild raised concerns about OpenAI's Voice Engine potentially eclipsing Voicecraft, while pondering on the strategic play involved and the potential repercussions on the US Elections.

**Stochastic Rounding as a Training Booster**: Engineers are looking into **stochastic rounding techniques** for training AI, presenting [nestordemeure/stochastorch](https://github.com/nestordemeure/stochastorch) as a promising Pytorch implementation to try out.

**Transforming Diffusion with Transformers**: Conversations trend towards replacing UNETs with transformers in diffusion, with a key [research paper](https://arxiv.org/pdf/2212.09748.pdf) guiding the way.

**Decoding UNET Mysteries**: A member breaks down UNETs as a tool for downsampling and then reconstructing images, which could help with discarding superfluous details in models.

**Qwen1.5-MoE-A2.7B Raises Expectations**: A buzz surrounds Qwen1.5-MoE-A2.7B, a model challenging larger counterparts with just 2.7 billion activated parameters, detailed across various platforms like [GitHub](https://github.com/QwenLM/Qwen1.5), [Hugging Face](https://huggingface.co/Qwen), and [Demo](https://huggingface.co/spaces/Qwen/Qwen1.5MoE-A2.7B-Chat).

**V-JEPA Sets the Stage for Video Lava**: The community examines V-JEPA's potential in enhancing video Lava, with GitHub resources at hand ([V-JEPA GitHub](https://github.com/facebookresearch/jepa)) to broaden the data prep and training terrain.

**Diffusion and Embedding Win Big With New Techniques**: A paper discussing a new diffusion loss function offers a glimmer of hope against data corruption ([paper link](https://arxiv.org/abs/2403.16728)), while Gecko's approach in text embedding might be a game changer in accelerating training ([Gecko paper link](https://arxiv.org/abs/2403.20327)).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Blazing 1-Bit Model Weights Introduced**: **Hugging Face** released **1.38 bit quantized model weights** for large language models (LLMs), signaling strides towards more efficient AI models. Interested engineers can scrutinize the model [here](https://huggingface.co/1bitLLM).

**PAG Refines Samples Without Sacrificing Diversity**: The utility of Perturbed-Attention Guidance (PAG) was showcased, which unlike Classifier-Free Guidance (CFG), doesn't reduce diversity when improving sample quality. The usage ratio of CFG 4.5 and PAG between 3.0 to 7.0 was recommended for enhanced results, based on [research](https://arxiv.org/abs/2403.17377).

**Real-Time Diffusion Now a Reality**: The use of *1 step diffusion* enabling 30fps generation at 800x800 resolution has been achieved using **sdxl-turbo**. For those intrigued by the seamless transitions, a Twitter thread with [video snippets](https://twitter.com/Dan50412374/status/1774527643058331980) showcases the evolution of real-time video generation.

**In Search of Tokenizer-Compatible Models**: An inquiry was made about how to identify suitable assistant models for `model.generate` by tokenizer, with discussions pointing to the **Hugging Face Hub API** for potential solutions. Additionally, approaches to extracting *domain-specific entities* were explored, recommending leveraging pre-trained models or considering independent training for 20k documents.

**Melding AI into Musical Alchemy**: Discussions included the challenge of AI-generated music, blending artists' voices to create harmonies like those of Little Mix, highlighted by the intricacy of key adjustments. Other technical endeavors shared in the community involved the creation of Terraform provider for **Hugging Face Spaces** and the introduction of OneMix, a Remix-based SaaS boilerplate.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Getting Chatty with Open Interpreter**: A video titled ["Open Interpreter Advanced Experimentation - Part 2"](https://www.youtube.com/watch?v=v9uXdRwAQ0c) reveals new experiments with the OpenInterpreter, demonstrating the platform's growing capabilities for technical innovation.

**AI as a Sidekick**: The [Fabric project on GitHub](https://github.com/danielmiessler/fabric), an open-source initiative, offers a modular framework designed to augment human skills with AI, utilizing a community-driven collection of AI prompts adaptable for various challenges.

**Audio Issues Crackdown**: In the OpenInterpreter community, an audio playback problem on MacOS involving `ffmpeg` was teased out, and solutions involving multiple commands were proposed to mitigate the trouble experienced after a response was generated.

**Windows Walkthrough Update**: The onboarding experience for Windows users working with the OpenInterpreter 01 client has seen enhancements with new pull requests ([#192](https://github.com/OpenInterpreter/01/pull/192), [#203](https://github.com/OpenInterpreter/01/pull/203)) aimed at resolving compatibility challenges and improving the setup documentation.

**Fine-Tuning for O1 Light Fabricators**: Makers of the O1 Light are advised to upscale 3D printing files to 119.67% for fitting the components properly, signaling a community-driven focus on custom hardware optimization.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Intel Arc Meets Optimized Performance**: Efforts to **optimize transformers** for Intel Arc GPUs identified the underperformance of IPEx library, as it wasn't employing **fp16** effectively. Solutions involving **PyTorch JIT** yielded significant performance improvements for stable diffusion tasks.

**Open Call: AMD GEMM Optimization Wanted**: A $200 bounty is up for grabs for writing optimized **GEMM code** for AMD 7900XTX GPUs with instructions including HIP C++ integration. However, the endeavor is hampered by script issues involving missing modules and library paths.

**Amendments Afoot in Tinygrad**: Discussions are ongoing within the **Tinygrad** repository, pinpointing issues with failing tests and missing functionalities. One suggestion involves examining the **shapetracker** and **uopt optimization** to enable contributions even from non-GPU laptop setups.

**AMD's Driver Saga**: Conversations centered on AMD driver instability, calling for an open-source approach for firmware and suggesting various GPU reset methods like **BACO and PSP mode2**. A GitHub discussion thread expressed frustration over full reset limitations and ineffective communication channels with AMD.

**Fusion and Views in Shape Manipulation**: The technicalities of **kernel fusion** and shape manipulation in Tinygrad were broached, with a [shared link on notes](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md) providing possible optimizations. An issue regarding *memory layout complexities* and *uneven stride presentation* was pinpointed and addressed in a recent [pull request](https://github.com/tinygrad/tinygrad/pull/3988).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Phorm.ai Teams Up with LlamaIndex**: [Phorm.ai](https://phorm.ai) integration provides TypeScript and Python support within LlamaIndex Discord, enabling queries and answers through "@-mention" within specific channels.

**Learn RAFT, Don't Be Daft**: A LlamaIndex **webinar** with RAFT co-authors, Tianjun Zhang and Shishir Patil, promises insights into domain-specific LLM fine-tuning, set for Thursday, 9am PT with sign-ups at [lu.ma](https://lu.ma/v1bdat63).

**RAG Revolution Deep Dives**: Guides and tutorials detail new strategies for enhancing Retrieval Augmented Generation, including self-reflective systems, integration with LlamaParse, and the importance of re-ranking, discussed across various platforms such as [Twitter](https://twitter.com/llama_index) and [YouTube](https://youtu.be/wCFXae8hiYA).

**LLM Research Made Accessible**: A GitHub repository by [shure-dev](https://shure-dev.github.io/) aims to consolidate impactful research papers on Large Language Models, serving as a comprehensive resource for AI enthusiasts.

**Tackling LlamaIndex Document Dilemmas**: Community members address complex issues, from managing oversized data chunks with **SemanticSplitterNodeParser** to improving outdated documentation, sharing best practices and solutions such as a helpful [Colab tutorial](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Novus Chat Jets onto OpenRouter**: [Novus Chat](https://talk.novus.chat/agents), a fresh platform integrating **OpenRouter models**, is creating buzz with free access to lowcost models and an invitation extended to AI enthusiasts to join its [development discussions](https://discord.gg/w9WR6QMAdt).

**Ranking Reveal Creates Model Buzz**: **OpenRouter** has introduced **App Rankings for Models**, allowing a glance at the top public apps that utilize specific models, with the **Apps** tab for each model revealing token stats; see [Claude 3 Opus App Rankings](https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps) as an example.

**OpenRouter Sparks Chatbot API Conversation**: Technical exchanges within the community are intensely focused on utilizing OpenRouter's APIs, embracing strategies for enhancing context retention and error handling while comparing functionalities between **Assistant Message** and **Chat Completion** approaches.

**ClaudeAI Beta: Now Self-Moderating**: OpenRouter's beta offering of **Anthropic's Claude 3 Opus** introduces a self-moderated version aiming to mitigate false positives, promising nuanced performance in sensitive contexts, as detailed in [Anthropic's announcements](https://www.anthropic.com/news/claude-3-family).

**Downtime Drama and Resolution**: Recent **Midnight Rose** and **Pysfighter2** models faced temporary downtime which was promptly resolved, whereas **Coinbase** payment issues were also flagged with assurance of a fix in progress, maintaining active wallet connections.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Bold Climb Beyond the Binary**: Discussions on **1-bit LLMs**, referred to as "1.58 bits per parameter" due to ternary quantization, revealed skepticism about marketing hype vs technical precision. Community engagement included sharing of relevant papers and anecdotal reproductions of key findings.

**Cross-Continental Voice Model Win**: Voicecraft's new open-source speech model has outperformed ElevenLabs, with members sharing [GitHub weights](https://github.com/jasonppy/VoiceCraft) and positive experiences.

**Bye-Bye, Boss**: Stability AI's CEO stepping down made waves, with the community dissecting interviews such as [Diamandis’s YouTube piece](https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa) and speculating about company futures and the tech executive landscape.

**Local LLMs Conquer Complexity**: Discussions in the AI-In-Action club took a deep dive into the efficiency of local LLM function calling, with contrasting opinions on which methods lead the pack, *outlines* vs *instructor*, and exploration of mechanisms like regular expressions in text generation.

**Anticipation for AI Agendas**: Upcoming sessions about UI/UX patterns and RAG architectures stirred up interest, backed by a [community-driven schedule](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0). Sharing of resources and facilitation plans spotlighted the proactive preparation for future tech talks.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Catch the CUDA Wave**: There's an increasing interest in **CUDA development**, with a preference for **VSCode** and explorations into **CLion**. A **CUDA course for beginners** starting April 5th is announced, with resources available on [Cohere's Tweet](https://x.com/CohereForAI/status/1773419415406809432), while **Mojo standard library** goes open-source as per details on [Modular's blog](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) and [GitHub](https://github.com/modularml/mojo/tree/nightly).

- **Precision Matters in Triton**: Experiments show **TF32** causing inaccuracies when using `tl.dot()` in **Triton**, with a noted discrepancy against PyTorch results, linked to [this issue](https://github.com/openai/triton/issues/1937). PyTorch's documentation helps clarify TF32 utilization, and **Nsight Compute** is discussed for profiling Triton code.

- **Triton Puzzle Conundrum**: The **Triton visualisation tool** challenges were resolved with a new notebook and detailed installation instructions, but warnings were raised about installation sequences that could lead to version incompatibilities.

- **LLM Finetuning Feasibility**: **PyTorch** released a config for single-GPU finetuning of the LLaMA 7B model on a reduced memory footprint, found [here](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml).

- **Flash Attention Focus**: Lecture 12 on **Flash Attention** raises interest, with the community prompted to attend. However, video quality issues on YouTube were reported, with the recommendation to check back for higher resolution processing.

- **From CUDA Queries to GPU Resources**: Queries relating to **CUDA development** on MacBooks and alternatives like **Google Colab** were addressed, confirming Colab's adequacy for CUDA programming. An Nvidia GPU though, is essential for running **CUDA** applications. Resources like [Lightning AI Studio](https://lightning.ai/pricing) offer free GPU time, with Colab touted as good for free access to Nvidia T4 GPUs.

- **CUDA Bookworms**: For the study of GPU architecture, discussions included strategies, such as reading the PMPP book thoroughly before attempting questions, possibly organizing work on GitHub, and diving deeper into memory load phases to understand optimization.

- **Ring-Attention Ruffling Feathers**: The community actively discusses training with **ring-attention** on long-context datasets, referencing datasets on Hugging Face and tools like **Flash-Decoding**. Workflow fixes and dataset sourcing are central, with the value of VRAM resources being a hot topic, hinted by the [VRAM table on Reddit](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/). 

- **Tech Ecosystem Discussions**: Papers on distributed training were solicited, yielding insights into **AWS GPU instance profiling** and cross-mesh resharding in model-parallel deep learning. An [MLSys abstract](https://proceedings.mlsys.org/paper_files/paper/2023/hash/a42cbafcabb6dc7ce77bfe2e80f5c772-Abstract-mlsys2023.html) on the topic drew particular attention.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GPU Memory Optimizations Emerge**: Significant **memory savings** have been reported using **PagedAdamW**, yielding nearly 50% reduction in peak memory usage (14GB vs. 27GB) for 8-bit implementations; the trick lies in optimizing the backward pass. Details were shared including a [configuration example on GitHub](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml).

- **Axolotl Meets DBRX**: The integration of **DBRX** into **axolotl** is a hefty task with substantial efforts underway, as evidenced by the progress in [pull request #1462](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1462). Discussions revealed intricacies in training control and the pursuit of multi-GPU optimizations that currently challenge the capacity for gradient accumulation.

- **Whisper Speaks Volumes in Transcription**: In the delicate art of transcribing audio to text, particularly in English and Chinese, solutions like **Whisper** shine for single speaker scenarios, while **Assembly AI** and **whisperx with diarization** were endorsed for complex multi-speaker tasks. Engineers are pushing boundaries, dealing with CUDA errors on **Runpod's GPUs** and testing **ring-attention** with increased sequence lengths (16k-32k), as seen in a GitHub repository for [ring-attention implementations](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching).

- **Model Agility for Text Classification**: Faced with limited resources, such as a **T4 GPU**, the community shared insights on leaner models adept at text classification—**Mistral** and **qlora**—alongside tools like [auto-ollama](https://github.com/monk1337/auto-ollama) for simple model testing via chat interfaces.

- **Engineers Exchange Epic Troubleshooting Tales**: From tackling **OOM issues** in ambitious **lisa branch** projects to diagnosing episodes of training stagnation after just one epoch, members rallied with suggestions pivoting around optimizer nuances and tools like **wandb**. Meanwhile, constructs for **AI-driven phone conversations via Telegram bots** keep the dialogue lively and diverse.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Open-Sourcing Mojo: A Community Effort**: The excitement about **Modular's** open-sourced Mojo standard library is palpable; however, there are frustrations due to limitations on non-internal/commercial applications and the lack of essential features like string sorting. Installation challenges on Linux Mint and desires for better profiling tools were also voiced, with official support confirmed for Ubuntu, MacOS, and WSL2 and guides provided for setup and local stdlib building.

**Mojo's Threading Quest and Docs Expansion**: Technical discussions on **Mojo's** multithreading capabilities highlighted the use of *OpenMP* for multi-core CPU enhancements and debates about `external_call()` functionality improvements. MLIR's syntax documentation is being improved to be more user-friendly, and there's a call for more detailed contributions.

**Library and Language Enhancements**: Several **Mojo libraries** have been updated to version 24.2, while the anticipation for a more evolved `Reference` component and better C/C++ interop in Mojo is strong. A new logging library, **Stump**, is introduced for the community to test.

**Tackling Code Challenges**: Performance and benchmarking channels discussed the **one billion row challenge**, noting the absence of certain standard library features and the need for improved memory allocation understanding. Meanwhile, the `matmul.mojo` example raised concerns over rounding errors and data type inconsistencies.

**MAX Makes Moves into Triton**: **MAX Serving** successfully operates as a backend for the Triton Inference Server, and the team is eager to support users in their migration efforts, emphasizing an easy transition and promising enhanced pipeline optimization.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Benchmarks Set Stage for AI Bravado**: The lm-sys released an advanced **Arena-Hard benchmark** aiming to better evaluate language models through intricate user queries. Debates arose around the potential biases in judging, especially exemplifying **GPT-4's self-preference** and its significant performance over **Claude** on Arena-Hard.

**Token Talk Takes Theoretical Turn**: Conversations pivoted to evaluating the **informational content** of tokens, with mutual information cited as a possible measure. Discussions framed this analysis against *repeng* strategies and Typicality methods, the latter detailed in an [information theory-based paper](https://arxiv.org/abs/2202.00666).

**Innovation Amidst The Hiring Game**: Discussions revealed **Stability AI** actively recruiting top researchers, while Nathan Lambert described Synth Labs' non-traditional startup strategy, introducing ground-breaking papers preeminent to their product launches.

**1-Bit Wonders**: NousResearch validated Bitnet's claims through a 1B model trained on the Dolma dataset, released on [Hugging Face](https://huggingface.co/NousResearch/OLMo-Bitnet-1B), igniting discussions on the novelty and technicalities of 1-bit training.

**sDPO Steps Up in RL**: Shared insights unveiled **stepwise DPO (sDPO)** through a [new paper](https://arxiv.org/abs/2403.19270), a technique that could democratize performance gains in model training, aligning models closely with human preferences without heavy financial backing.

**Preserving Alignment Almanac**: Nathan Lambert announced an initiative to document and discuss the evolution of **open alignment techniques** post-ChatGPT. Contributions such as an overview of *various replicating models* and considerations on preference optimization methods glean insight into the historical growth of the field, documented in [Lambert's Notion Notes](https://magnetic-share-282.notion.site/History-of-Open-Alignment-a7ef20aefb34438185336df68147809e?pvs=4).



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba's Code Conundrum Continues**: The **performance of Jamba-v0.1 on Code tasks**, such as the **HumanEval benchmark**, is still not discussed, sparking curiosity within the community.
- **Jamba's Language Inclusivity in Question**: Queries were raised about the inclusion of languages like **Czech in Jamba's training data**, but no conclusive information has been provided.
- **Jamba Prepares for Fine-Tune Touchdown**: There is anticipation in the community for **Jamba to be available for fine-tuning** on **AI21 Studio**, with expectations of an instruct model coming to the platform.
- **Understanding Jamba's Hardware Hunger**: Discussions highlighted that **Jamba** efficiently uses just **12B of its 52B parameters through MoE layers** during inference, yet there's a consensus that operating Jamba on consumer-grade hardware, such as an NVIDIA 4090, is not feasible.
- **Demystifying Jamba's Block Magic**: A technical exchange clarified the role of **Mamba and MoE layers** in Jamba, with a slated ratio being one Transformer layer for every eight total layers, confirming that Transformer layers are not part of MoE but integrated within specific blocks in the Jamba architecture.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GalaxyAI Astonishes with Free API Offers**: GalaxyAI has rolled out a free API service that allows users to access high-caliber AI models like **GPT-4** and others, bolstering the community's ability to integrate AI into their projects. Interested developers can try the API [here](https://galaxyapi.onrender.com).

- **Illuminating Model Alignment Techniques**: A blog has outlined the application of methods such as RLHF, DPO, and KTO to models like Mistral and Zephyr 7B, aiming to enhance model alignment. Those curious can digest the full details on [Premai's blog](https://blog.premai.io/model-alignment-process/).

- **Revolutionizing AI with Chain of Tasks**: Innovation in prompting techniques for crafting advanced conversational LLM Taskbots using LangGraph, named the Chain of Tasks, has been highlighted across two blog posts. To probe deeper into these developments, readers can peruse the [LinkedIn article](https://www.linkedin.com/posts/prasadt_introducing-chain-of-tasks-cota-a-prompting-activity-7178582571423870976-wajV).

- **CrewAI Ushers in AI Agent Orchestration**: The announcement of CrewAI's framework for the orchestration of autonomous AI agents has sparked interest for its seamless OpenAI and local LLM integration capabilities, with the community invited to explore on their [website](https://crewai.com) and [GitHub](https://github.com/joaomdmoura/crewAI/tree/main).

- **Vector Databases Made Accessible with Qdrant and Langchain**: Members can now dive into vector databases courtesy of a tutorial demonstrating the fusion of Qdrant and LangChain, looking at local and cloud implementations. The in-depth tutorial awaits enthusiasts in the form of a [YouTube video](https://youtu.be/JSKZYgARffg).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Hyperparameter Hiccup Frustrations**: Running the command `./server -m all-MiniLM-L6-v2-f32.gguf --embedding` caused an error related to `bert.context_length`, but no solution to the error was provided during the discussions.

- **llamafile's Stability: A Work in Progress**: Users have experienced instability when executing **llamafile**, with some instances of inconsistent performance; one user committed to probing these issues in the upcoming week.

- **llamafile v0.7 Makes Its Entrance**: The community heralded the release of **llamafile v0.7**, highlighting enhancements in performance and accuracy, alongside a well-received blog post detailing improvements in **matmul** just before April Fool's Day.

- **In Search of the Perfect Prompt**: There were inquiries about the ideal prompt templating for running **llamafile** using **openchat 3.5 0106** in the web UI, including examples of template input fields and variables, but clear guidance remained elusive.

- **Matmul Benchmarking Throwdown**: A benchmarking code snippet for comparing numpy's **matmul** with a custom implementation was provided by **jartine**, sparking interest in alternative methods that bypass threading yet improve efficiency.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Scout Law Inspires Chatbot Banter**: Employing the Scout Law, a user programmed **Claude 3 Haiku** to respond with quirky yet honest quips, exemplified by witty phrases like *"A door is not a door when it's ajar!"*.
- **Chatbot's Friendly Babble by Design**: The chatbot's tendency to elaborate extensively is by design, aligning with a system prompt that directs it to embody *friendliness and helpfulness*, demonstrating this by integrating elements of the Scout Law in its dialogue.
- **Trustworthy Shells and Chatbots**: Mimicking the Scout Law's value of trustworthiness, the bot creatively compared limpets and their protective shells to the concept of trust, showing adeptness at thematic interpretation.
- **Strategizing Queries for Clearer Understanding**: An approach was tested where the chatbot would pose clarifying questions before offering direct solutions, suggesting a method that could enhance problem-solving effectiveness.
- **Resolution for Installation Hiccups**: Addressing a `FileNotFoundError` during `llm` installation, it was advised to reinstall the package, as this was a confirmed necessary step by another user who recently confronted similar issues.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Jamba Joins the Model Mix**: [AI21 Labs introduces Jamba](https://www.ai21.com/jamba), featuring a **Structured State Space model** combined with a **Transformer**, which can be tested on [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1).

- **BitNet Clones Rival Original**: Successful reproduction of the [BitNet b1.58 model](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) matched original performance on the **RedPajama dataset**, guided by their [follow-up paper's](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) methodology.

- **Model Behavior Under Magnifying Glass**: Discussions on **novel LLM architectures** include queries for assessments or "vibe checks" and observations on **Nectar dataset** constructed with GPT-4 ranking, with sources such as **ShareGPT**, **Antropic/hh-rlhf**, and **Flan**.

- **Questionable AI Guidance in Hot Water**: A controversial instance in the Nectar dataset showed GPT offering instructions on making a gun, with models like **Starling** possibly responding differently from models that choose to refuse.

- **Translation Evaluation Tools at the Ready**: **Translation quality** is scrutinized using a new tool found on [Hugging Face](https://huggingface.co/spaces/cstr/compare_translations) and **comet scores**, providing metrics for translation assessments in German language discussions.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

**AI21's Jamba Jumps into the Fray**: The [**Jamba** model](https://www.youtube.com/watch?v=HRnx0ZPxe64) by AI21 has been shared within the **Skunkworks AI** community, touting enhancements in SSM-Transformer design for large language models.

**Databricks' DBRX LLM Claims the Crown**: Databricks' general-purpose large language model **DBRX** supposedly establishes new highs on multiple benchmarks, according to a [shared video](https://www.youtube.com/watch?v=dqFvOqC43rQ) in the **Skunkworks AI** community.

**Tackling Catastrophic Forgetting in Class-Incremental Learning**: Research indicates that **adapter tuning** might be the key to combating catastrophic forgetting in CIL, employing feature sampling and prototype semantic shift analysis. The study is accessible through this [arXiv link](https://arxiv.org/abs/2403.19979).

**Closing the Gap Between Open-source and Commercial LLMs**: A novel paper discusses methodologies aimed at empowering open-source LLMs to close the performance gap with their commercial counterparts, with strategies focused on 7B and 13B LLM enhancements. The paper's detailed insights are available [here](https://arxiv.org/abs/2403.19962).


---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1223139901365682316)** (980 messages🔥🔥🔥): 

- **In Search of Enhanced Inpainting UI**: Users expressed dissatisfaction with the new SD inpainting UI, finding it unintuitive and inefficient due to illogical layout decisions.
- **CivitAI April Fools' Day Pranks?**: Models with unexpected names like "ODOR" and pop-ups about "chadgpt" led to discussions about possible April Fools' jokes on CivitAI.
- **Stable Diffusion Model Concerns and Queries**: There were questions about when the stable diffusion 3 (SD3) model will be released, with users sharing their hopes and skepticism. One user jokingly claimed SD3's release as an April Fools' prank.
- **Help Wanted with AI Tools**: Users sought assistance for various AI-related issues, such as setting up ControlNet reference in Colab, using Comfy UI, and generating architecture renders in Stable Diffusion.
- **Ideas on How to Utilize AI for Creativity**: Suggestions were made on how to use AI for creating content, including utilizing old video game footage for AI-generated films and crafting comics with AI workflows.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/artsiom_s/status/1774125236234924173?s=46">Tweet from Artsiom Sanakoyeu (@artsiom_s)</a>: ⚡️SD3-Turbo: Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation  Following Stable Diffusion 3, my ex-colleagues have published a preprint on SD3 distillation using 4-s...</li><li><a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt">stabilityai/stable-video-diffusion-img2vid-xt · Hugging Face</a>: no description found</li><li><a href="https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks">Stable Diffusion Benchmarks: 45 Nvidia, AMD, and Intel GPUs Compared</a>: Which graphics card offers the fastest AI performance?</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/15c2n0q/sdxl_two_text_encoders_two_text_prompts/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/367412/geeky-ghost-vid2vid-organized-v1">Geeky Ghost Vid2Vid Organized v1 - v3.0 | Stable Diffusion Workflows | Civitai</a>: This workflow is designed for advanced video processing, incorporating various techniques such as style transfer, motion analysis, depth estimation...</li><li><a href="https://support.noduslabs.com/hc/en-us/articles/360015660099-How-to-Import-and-Visualize-Your-Roam-Research-Obsidian-and-Zettelkasten-Markdown-Format-Notes">How to Import and Visualize Your Roam Research, Obsidian and Zettelkasten Markdown Format Notes</a>: If you have a markdown format files (.MD) you can import them into InfraNodus to visualize the main topics, their relations, and discover the structural gaps in order to generate new ideas. InfraNo...</li><li><a href="https://soundcloud.com/neel-sikka-510755355/ice-on-my-baby-yung-bleu-sped-up-3">ice on my baby - yung bleu (sped up &lt;3)</a>: wsg</li><li><a href="https://developer.nvidia.com/ace">NVIDIA Avatar Cloud Engine ACE</a>: Build and deploy game characters and interactive avatars at scale.</li><li><a href="https://tenor.com/view/pony-ride-back-ride-skates-gif-13973117">Pony Ride Back Ride GIF - Pony Ride Back Ride Skates - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://feedback.civitai.com/p/pass-or-fail-a-simple-and-controlled-model-ranking-feature>)">Feedback - Civitai</a>: Give Civitai feedback on how they could improve their product.</li><li><a href="https://www.youtube.com/watch?v=86x-u-tz0MA">Your elusive creative genius | Elizabeth Gilbert</a>: Find an accurate transcript (and subtitles in 46 languages) on ted.com:http://www.ted.com/talks/elizabeth_gilbert_on_genius/transcript?language=en &quot;Eat, Pray...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1brpzf7/sd3turbo_is_better_than_midjourney_6_in_both/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtube.com/shorts/C5cIib7hiK8?si=z8FW2_UFwgZEn0LK">1万年かけて成長するフリーレン(上半身)/Frieren growing over 10,000 years(upper body) #葬送のフリーレン #frieren #アニメ</a>: no description found</li><li><a href="https://www.vecteezy.com/video/8661328-animation-infinite-looping-triangle-black-and-white-seamless-loop-motion-background">Download Animation infinite looping triangle black and white - Seamless loop Motion Background for free</a>: Download the Animation infinite looping triangle black and white - Seamless loop Motion Background 8661328 royalty-free Stock Video from Vecteezy and explore thousands of other stock footage clips!</li><li><a href="https://www.youtube.com/watch?v=TfrAf1a9Qhs">Original Trogdor Video</a>: This is the first ever video of Trogdor.</li><li><a href="https://youtu.be/iuP9uiTH95I">cat but it&#39;s a gamecube intro</a>: cat.consider subscribing : https://www.youtube.com/channel/UCIzz...Instagram: https://www.instagram.com/merryyygoat/</li><li><a href="https://youtu.be/_XR6dsy7ATE?si=SjByJVatz01_129s">oblivion 4</a>: #oblivion #npc #elderscrolls Stop! You violated the law. Original oblivion video: https://www.youtube.com/watch?v=qN80_7rNmcEOur Let&#39;s Play Oblivion Series: ...</li><li><a href="https://www.youtube.com/watch?v=2-QQpL1YjeU">The Name of This Cartoon Would Ruin It</a>: Strong Bad and The Cheat come across Homestar doing something truly terrifying in the snow. Befuddlement ensues.</li><li><a href="https://youtu.be/xixgDV_9RJI">Happy 20th Trogday!</a>: For Trogdor&#39;s 20th birthday, Strong Bad digs up a 30 year old promo for an unreleased Peasant&#39;s Quest game. Do the math!</li><li><a href="https://github.com/XPixelGroup/DiffBIR?tab=readme-ov-file">GitHub - XPixelGroup/DiffBIR: Official codes of DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior</a>: Official codes of DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior - XPixelGroup/DiffBIR</li><li><a href="https://manifold.markets/LoganZoellner/will-the-weights-for-stable-diffusi">Will the weights for Stable Diffusion 3 be released (or leaked) before April 30?</a>: 41% chance. On March 15, then CEO of Stability AI Emad Mostaque announced that SD3 would have a full-release &quot;next month&quot;.  However, after several high-profile departures culminating in Emad...</li><li><a href="https://github.com/jhc13/taggui/releases">Releases · jhc13/taggui</a>: Tag manager and captioner for image datasets. Contribute to jhc13/taggui development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/207992/stable-video-diffusion-svd">Stable Video Diffusion - SVD - img2vid-xt-1.1 | Stable Diffusion Checkpoint | Civitai</a>: Check out our quickstart Guide! https://education.civitai.com/quickstart-guide-to-stable-video-diffusion/ The base img2vid model was trained to gen...</li><li><a href="https://github.com/PixArt-alpha/PixArt-alpha">GitHub - PixArt-alpha/PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis</a>: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis - PixArt-alpha/PixArt-alpha</li><li><a href="https://youtube.com/shorts/qN80_7rNmcE">oblivion</a>: Death by poison fruit.Part 2:https://www.youtube.com/watch?v=D_x-x0gawckPart 3:https://www.youtube.com/watch?v=CYAkrJil_w0</li><li><a href="https://github.com/jasonppy/VoiceCraft">GitHub - jasonppy/VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild</a>: Zero-Shot Speech Editing and Text-to-Speech in the Wild - jasonppy/VoiceCraft</li><li><a href="https://youtu.be/_mOGVOVZw70">Skymaster Carnival Ride at Night 😳 #carnivalrides #fun</a>: The Kamikaze Ride is a thrilling carnival ride that will bring you to heights never before experienced. It&#39;s a Kamikaze ride that will have you soaring throu...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1223136234369581108)** (915 messages🔥🔥🔥): 

- **Claude's Opus Performance Quirks**: Users report varying performance from the **Claude 3 Opus** model when answering a question about books remaining in a room. Despite instructions and adjustments in prompts, some models still failed to deliver correct answers in this context.

- **Comparing AI models**: There was a discussion about various AI models and their capabilities, with mentions of **Haiku**, **Gemini 1.5 Pro**, and **Claude Opus**. Users expressed views on the models' strengths, weaknesses, and differences in functionality.

- **Crypto vs. Gold debate**: In a brief tangent, members mused over the comparative value of **crypto** and **physical commodities**, specifically **gold**. Opinions varied on the future of currency, with some seeing the potential in digital forms while respecting the long-standing value of traditional materials like gold.

- **Evolving Tech Landscape Speculation**: Conversation involved speculation about the future direction and advancements in AI technology. Points were raised about companies like **Apple's involvement in AI** and debates on **China's economic approaches**, including references to the **Evergrande crisis** and **government strategies**.

- **Perplexity Access and Features**: Queries arose regarding **Perplexity's user experience** and features like password change or subscription cancellation. The chatbot clarified it uses **oauth2** for logins and does not support password-based access.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/books-father-ted-fatherted-drink-gif-19969450">Books Father GIF - Books Father Ted - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/LinusEkenstam/status/1774847013752070457?t=7tzw85sz9QgE_TN7zRA82Q&s=09">Tweet from Linus ●ᴗ● Ekenstam (@LinusEkenstam)</a>: 🚨 Breaking 🚨  Apple is in talks to acquire perplexity  This could be the start of something very exciting</li><li><a href="https://tenor.com/view/working-on-it-under-construction-gif-23162421">Working On GIF - Working On It - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics">GitHub - shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics: Awesome LLM-related papers and repos on very comprehensive topics.</a>: Awesome LLM-related papers and repos on very comprehensive topics. - shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics</li><li><a href="https://fxtwitter.com/AravSrinivas/status/1774804294627377431">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Excited to share Perplexity will be powering http://askjeeves.com</li><li><a href="https://tenor.com/view/harry-potter-quirrell-professor-quirrell-troll-troll-in-the-dungeon-gif-19761740">Harry Potter Quirrell GIF - Harry Potter Quirrell Professor Quirrell - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/why-michael-scott-the-office-why-are-the-way-that-you-are-gif-5593972">Why Michael Scott GIF - Why Michael Scott The Office - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ouch-gif-12136515515962044163">Ouch GIF - Ouch - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/vegeta-dragon-ball-z-unlimited-power-over9000-power-level-gif-12316102">Vegeta Dragon Ball Z GIF - Vegeta Dragon Ball Z Unlimited Power - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.anthropic.com/claude/docs/troubleshooting">Troubleshooting</a>: no description found</li><li><a href="https://docs.anthropic.com/claude/docs/functions-external-tools">Functions &amp; external tools</a>: no description found</li><li><a href="https://docs.anthropic.com/claude/docs/prompt-engineering">Prompt engineering</a>: no description found</li><li><a href="https://docs.anthropic.com/claude/page/prompts">Prompt library</a>: no description found</li><li><a href="https://app.wordware.ai/r/b0f0a2c9-da4f-4524-b662-3584ac0fdbc2">Wordware - OPUS Insight: Precision Query with Multi-Model Verification -scratchpad-think-Version-1</a>: This prompt processes a question using Gemini, GPT-4 Turbo, Claude 3 (Haiku, Sonnet, Opus), Mistral Medium, Mixtral, and Openchat. It then employs Claude 3 OPUS to review and rank the responses. Upon ...</li><li><a href="https://youtu.be/57LqvutrOI8?si=20V_X46fF4GmtKfL">Perplexity CEO: Disrupting Google Search with AI</a>: One of the most preeminent AI founders, Aravind Srinivas (CEO, Perplexity), believes we could see 100+ AI startups valued over $10B in our future. In the epi...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1223208444790571090)** (36 messages🔥): 

- **Browsing the Bounds of Knowledge**: Members shared diverse perplexity.ai searches, revealing interests in topics like [the limitations of Bohmian mechanics](https://www.perplexity.ai/search/The-maximum-size-BoHMgacwSC2gKljdLQU.ZQ), [the workings of SpaceX](https://www.perplexity.ai/search/How-does-space-X89jm9d1T3iehXiHBRnGXA), and the [definition of 'Isekai'](https://www.perplexity.ai/search/definition-of-Isekai-qjTbWQ9gQ4qTY_Yn0loD1Q).
- **Diving Deep into AI and Hyperloop**: Curiosity led to explorations explaining [Grok15](https://www.perplexity.ai/search/Grok15-XnITMLKjR.SmDjdzJUjtLQ), [the Hyperloop concept](https://www.perplexity.ai/search/Explain-the-hyperloop-tjp739whTQazB1QZ3prelw), and binary [embeddings in machine learning](https://www.perplexity.ai/search/int8-binary-Embeddings-NJ0Ixh6aRBaR3tPB5cBNGw).
- **Facilitating Knowledge Sharing**: A member provided guidance on making threads shareable, enhancing community access to specific topics, as reflected by a shared helpful [Discord link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Unpacking AI Alliances and AI in Podcasting**: Discussions unfolded about OpenAI's collaboration with Microsoft and ways to utilize AI for [processing podcast transcripts](https://www.perplexity.ai/search/Discover-Daily-Podcast-XrV5L7_BRnytYcYgUvwQaQ).
- **April's Technological Tricks and Knowledge Collections**: Members engaged with an [April Fool's tech-related query](https://www.perplexity.ai/search/April-fool-tech-Au6YyiG1TCCPZdIBAawqHw) and shared a link to a [Perplexity AI collection](https://www.perplexity.ai/collections/GPTDevil-A.GvNcQZS0yjMGDDJtDPUQ) for grouped knowledge.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1223212914765201428)** (41 messages🔥): 

- **API Response Differences**: A member noted that the API does not show as many sources as the web interface for Perplexity AI. They were directed to information indicating that **URL citations are still in beta**, and applications are open at [Perplexity's Typeform](https://perplexity.typeform.com/to/j50rnNiB).
  
- **Partnership Proposals**: Multiple discussions arose around potential partnerships with Perplexity. Members were urged not to ping others and to reach out via **support@perplexity.ai** for partnership inquiries and necessary introductions.

- **Confusion Over Model Support**: Clarifying questions were asked about the continued support for the "pplx-70b-online" model, not listed in the supported models documentation. A member clarified that it's deprecated, and the endpoint name is just an alias for `sonar-medium-online`.

- **Troubleshooting Credit Purchases**: Members reported issues trying to add credits, with transactions stuck in 'pending' state or experiencing errors with debit cards. It was suggested that this might be caused by security features required by Stripe, but concerns were raised that the problem might need further investigation.

- **Token Cost Comparisons Requested**: A user requested resources to compare token costs of Perplexity models with ChatGPT. They were provided with links to the Perplexity Pricing page and OpenAI's pricing for various models, including detailed price per token and additional request charges for online models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/pricing">Pricing</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.</li><li><a href="https://openai.com/pricing">Pricing</a>: Simple and flexible. Only pay for what you use.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1223147469714620456)** (549 messages🔥🔥🔥): 

- **Unsloth Models Are Still a Challenge**: Despite attempts to utilize [Unsloth](https://github.com/unslothai/unsloth) for fine-tuning Mistral models, users encounter dependency issues and demand a Docker solution for ease of use. Current discussions suggest multi-GPU support isn't available yet but is a future possibility.
- **Discussion on Optimizing Mistral Fines**: Questions about fine-tuning methods revealed that QLora 4bit may differ in performance from SFT/pretraining due to the quantization process. Users explore various ways to utilize available VRAM efficiently.
- **New AI Hardware from China**: Chinese company [Intellifusion](https://www.icsmart.cn/75486/) announces a new AI processor that might be cost-effective for inference but raises questions among users about its potential for training and other technical specifications.
- **Dataset Formatting Queries**: While discussing the creation of a model that simulates a Discord server ambiance, users debate the optimal format for training data, with a focus on representing conversations accurately.
- **Model Quantization and Language Support**: Users discuss hallucination issues with quantized models (like Mistral 7B) and explore options, including the incorporation of LASER, for fine-tuning in languages other than English.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://huggingface.co/brittlewis12/gemma-7b-GGUF&ved=2ahUKEwiJ0s-jnpyFAxWDXWwGHdgXBAIQFnoECBEQAQ&usg=AOvVaw2Ek3-WVYBKoa-gHH6kB8lY">no title found</a>: no description found</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&u">Redirect Notice</a>: no description found</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus">Chinese chipmaker launches 14nm AI processor that's 90% cheaper than GPUs &mdash; $140 chip's older node sidesteps US sanctions</a>: If there's a way to sidestep sanctions, you know China is on that beat.</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser">cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser · Hugging Face</a>: no description found</li><li><a href="https://sambanova.ai/blog/accurate-models-at-blazing-speed">SambaNova Delivers Accurate Models At Blazing Speed</a>: Samba-CoE v0.2 is climbing on the AlpacaEval leaderboard, outperforming all of the latest open-source models. </li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B">abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B · Hugging Face</a>: no description found</li><li><a href="https://api.wandb.ai/links/augmxnt/h4mc4dd5">Jamba Initial Tuning Runs</a>: Code here: https://github.com/shisa-ai/shisa-v2/tree/main/_base-evals/jamba  Initial experiments with doing fine tuning on Jamba using 1 x A100-80. Uses 99%&#43; of VRAM w/ these settings but does not...</li><li><a href="https://tenor.com/view/come-look-at-this-gif-21207051">Come Look At This GIF - Come Look At This - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.together.ai/blog/llama-2-7b-32k">Preparing for the era of 32K context: Early learnings and explorations</a>: no description found</li><li><a href="https://arxiv.org/abs/2312.13558">The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction</a>: Transformer-based Large Language Models (LLMs) have become a fixture in modern machine learning. Correspondingly, significant resources are allocated towards research that aims to further advance this...</li><li><a href="https://huggingface.co/teknium/OpenHermes-2-Mistral-7B">teknium/OpenHermes-2-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://x.com/NousResearch/status/1773923241268003052">Tweet from Nous Research (@NousResearch)</a>: We are releasing our first step in validating and independently confirming the claims of the Bitnet paper, a 1B model trained on the first 60B tokens of the Dolma dataset.  Comparisons made on the @we...</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit">unsloth/mistral-7b-v0.2-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0#technical-deep-dive">yanolja/EEVE-Korean-10.8B-v1.0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found</li><li><a href="https://deepspeed.readthedocs.io/en/latest/zero3.html#example-zero-3-configurations">ZeRO &mdash; DeepSpeed 0.14.1 documentation</a>: no description found</li><li><a href="https://github.com/shisa-ai/shisa-v2/blob/main/_base-evals/jamba/01-train-sfttrainer.py">shisa-v2/_base-evals/jamba/01-train-sfttrainer.py at main · shisa-ai/shisa-v2</a>: Contribute to shisa-ai/shisa-v2 development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/predibase/lorax">GitHub - predibase/lorax: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs</a>: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs - predibase/lorax</li><li><a href="https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct">teknium/GPTeacher-General-Instruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://tenor.com/view/muzeke-gif-27066384">Muzeke GIF - Muzeke - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2312.15166">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: We introduce SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. Inspired by recent efforts t...</li><li><a href="https://mlabonne.github.io/blog/posts/2024-03-28_Create_Mixture_of_Experts_with_MergeKit.html">ML Blog - Create Mixtures of Experts with MergeKit</a>: Combine multiple experts into a single frankenMoE</li><li><a href="https://sambanova.ai/blog/benchmarking-samba-1">Benchmarking Samba-1</a>: Benchmarking Samba-1 with the EGAI benchmark - a comprehensive collection of widely adapted benchmarks sourced from the open source community. </li><li><a href="https://apps.sambanova.ai/sambachat">no title found</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1br8ry8/finetuning_a_llm_for_longform_creative_writing/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/trainer">Trainer</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1223214571507028060)** (24 messages🔥): 

- **Snapdragon Elite X's Strong Entrance**: The Snapdragon Elite X Arm chip is reported to surpass m3 chips in performance, offering a more cost-efficient alternative with its 45 TOPs. The [discussion](https://youtu.be/dTCm6BupWEQ) includes a YouTube video titled *"Now we know the SCORE | X Elite"* which explains the benchmarks of Qualcomm's new offering.
- **Benchmark Enthusiasm Over New Chip**: There's excitement over the reported 45 TOPs of the Snapdragon Elite X, leading to comparisons with other chips like the Tesla T4 which has around 65 TFLOPs of float16.
- **Disappointment Over Modern MacBook Specs**: Members expressed frustration with the current specs and prices of MacBooks, highlighting the appeal of next-gen chips like the Snapdragon Elite X as more competitive, cost-effective options.
- **Discord Server Security Measures Discussed**: Members discussed the prevalence of bots and hacked accounts on Discord and recommended making servers community servers to prevent mass tags and advising the blocking of keywords associated with spam, such as "nitro."
- **Training Data Diversity for AI**: There was a conversation about the counterintuitive nature of training data required for fine-tuning AI models, debating whether including diverse data, such as "Chinese poems from the 16th century," could be beneficial compared to more directly related data like math for code performance enhancement.

**Link mentioned**: <a href="https://youtu.be/dTCm6BupWEQ">Now we know the SCORE | X Elite</a>: Qualcomm&#39;s new Snapdragon X Elite benchmarks are out! Dive into the evolving ARM-based processor landscape, the promising performance of the Snapdragon X Eli...

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1223140638774788169)** (461 messages🔥🔥🔥): 

- **Model Fine-Tuning Over Different Datasets**: A user reported better performance after fine-tuning Mistral with 3,000 rows of data compared to 6,000 rows. A theory was provided that aligns with research indicating an initial accuracy reduction with more data until at a certain point where more data improves performance. The user was advised to possibly use their 3,000 rows dataset instead of 6,000 if the additional data was of poor quality.

- **GGUF File Generation Challenges and Solutions**: Users experienced issues and confusion while trying to create GGUF files and running models with Unsloth. One solution presented was to follow the [manual GGUF guide](https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf) or leverage tools like llama.cpp to convert and save properly.

- **Single vs. Dual GPU Training with Unsloth**: A user ran into a warning regarding Unsloth's use of a single GPU. It was clarified that Unsloth currently supports only single GPU training; however, users can select the GPU to be used by setting the environment variable `os.environ["CUDA_VISIBLE_DEVICES"]`.

- **Fine-Tuning Loss Concerns and Optimization Strategies**: A discussion on fine-tuning Gemma 2B with various parameters was held to address concerns over a flat loss graph, which usually indicates lack of learning. Strategies such as increasing the rank and alpha values and adjusting the learning rate helped improve results.

- **Exploring AI Learning Resources for Beginners**: For users interested in learning about AI, recommendations were made for Andrej Karpathy's CS231N lecture videos, Fast AI courses, MIT OCW, and Andrew Ng's CS229 lecture series as great resources to begin with.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb">Google Colaboratory</a>: no description found</li><li><a href="https://docs.wandb.ai/guides/track/jupyter">Track Jupyter Notebooks | Weights &amp; Biases Documentation</a>: se W&amp;B with Jupyter to get interactive visualizations without leaving your notebook.</li><li><a href="https://ollama.com/pacozaa/tinyllama-alpaca-lora">pacozaa/tinyllama-alpaca-lora</a>: Tinyllama Train with Unsloth Notebook, Dataset https://huggingface.co/datasets/yahma/alpaca-cleaned</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://www.kaggle.com/competitions/kaggle-llm-science-exam/overview">Kaggle - LLM Science Exam</a>: no description found</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook">Kaggle Mistral 7b Unsloth notebook</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://huggingface.co/gnumanth/gemma-unsloth-alpaca">gnumanth/gemma-unsloth-alpaca · Hugging Face</a>: no description found</li><li><a href="https://www.kaggle.com/code/gnumanth/kaggle-mistral-7b-unsloth-notebook">Kaggle Mistral 7b Unsloth notebook</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://huggingface.co/gnumanth/code-gemma">gnumanth/code-gemma · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#local-and-remote-files">Load</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/toranb/sloth/blob/master/sftune.py">sloth/sftune.py at master · toranb/sloth</a>: python sftune, qmerge and dpo scripts with unsloth - toranb/sloth</li><li><a href="https://docs.google.com/spreadsheets/d/1tRbUj8qjsnZdUJOEXJFRLFSOwOWKFaA3hgRC1XWje-w/edit?usp=sharing">GPU Cloud comparisons</a>: Stats  #,VRAM,Float16 TFLOPs,Float8 TFLOPs,Band Width,W Per Card,Price,Per GPU,Per fp16 PFLOP,Per fp8 PFLOP,BF16,Supply,Info Kaggle,Tesla T4,1,16,65,320,70,0,0,0.000,N,5,&lt;a href=&quot;https://www.n...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_token">Trainer</a>: no description found</li><li><a href="https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.ja">wikimedia/wikipedia · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/CohereForAI/aya_dataset/viewer/default/train?q=japanese">CohereForAI/aya_dataset · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1223680602943787098)** (7 messages): 

- **Munchkin Streamlit App Launched**: Ivysdad_ announced the launch of a new tool or creation located at [Munchkin Streamlit App](https://munchkin.streamlit.app).

- **Innovative LLaMA Factory Integration**: Hoshi_hiyouga implemented **Unsloth + ORPO** in LLaMA Factory, offering a method to align Large Language Models (LLMs) which doesn't require two-stage training or a reference model. The paper detailing ORPO is found at [arXiv:2403.07691](https://arxiv.org/abs/2403.07691).

- **Community Praise for ORPO Implementation**: Members, including theyruinedelise and starsupernova, praised the implementation of Unsloth + ORPO, with starsupernova noting their focus on ongoing bug fixes.

- **Optimization Appreciation**: Hoshi_hiyouga expressed appreciation for starsupernova's work, specifically the optimization for **Gemma**.

- **ORPO Proves Effective in Experiments**: Remek1972 shared success in using ORPO for experimental trainings, noting that the fine-tuning of a new Mistral Base model with only 7000 training samples closely matched the performance of the old Mistral instruction model.

**Link mentioned**: <a href="https://munchkin.streamlit.app">no title found</a>: no description found

  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1223607036445458493)** (6 messages): 

- **Boosting Unsloth Notebooks**: A member suggested setting `group_by_length=True` in the TrainingArguments of the Unsloth AI notebooks, referencing a discussion on the [Hugging Face forum](https://discuss.huggingface.co/t/are-dynamic-padding-and-smart-batching-in-the-library/10404/9) that points to performance improvements.
- **Packing as an Optional Speed Booster**: Another member supported the idea, proposing to add it as an optional parameter like `packing = True` but noted that it can't be default due to varying losses.
- **DeepSeek Joins the Unsloth Family**: There’s a call to add the smallest [DeepSeek model from the official repository](https://huggingface.co/deepseek-ai) to Unsloth 4bit, highlighting the model as a good base model for AGI development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discuss.huggingface.co/t/are-dynamic-padding-and-smart-batching-in-the-library/10404/9">Are dynamic padding and smart batching in the library?</a>: Please do not post the same message three times and tag users agressively like you did. You can always edit your message instead of reposting the same thing.</li><li><a href="https://huggingface.co/deepseek-ai/">deepseek-ai (DeepSeek)</a>: no description found</li><li><a href="https://chat.deepseek.com/">DeepSeek</a>: Chat with DeepSeek AI.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[unsloth](https://discord.com/channels/1179035537009545276/1224229487588282509/1224237738367258684)** (1 messages): 

- **Support Unsloth's Mission**: The Unsloth team, comprised of two brothers, is asking for community support through [engagement or donations](https://ko-fi.com/unsloth). They promise shoutouts for all supporters in future blog posts and encourage contributions to help purchase a new PC and GPU for increased efficiency.
- **Donations Benefit and Rewards**: Supporters who donate can enjoy benefits like a unique Discord role, while funds will not only support Unsloth's operational costs but also contribute to the open-source software (OSS) community. Donations will directly impact Unsloth's ability to improve their service and support other creators.
- **Membership Perks for Supportive Sloths**: Becoming a member grants access to a special channel for priority support and discussion. The team expresses gratitude for any level of engagement, emphasizing that contributions are appreciated but not obligatory.

**Link mentioned**: <a href="https://ko-fi.com/unsloth">Support Unsloth AI on Ko-fi! ❤️. ko-fi.com/unsloth</a>: Support Unsloth AI On Ko-fi. Ko-fi lets you support the people and causes you love with small donations

  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1223180893984919562)** (19 messages🔥): 

- **Fashioning StyleGAN**: An inquiry was made about training a directory structure with multiple types of fashion images using StyleGAN2-ada, wondering if modifications to the script are necessary.
- **Aerospace Student's ML Flight Plan**: An aerospace student sought advice on entering the ML/AI field; recommendations included starting with the [fastai courses](https://course.fast.ai/) and gaining practical experience. The student linked to the course and considered it for learning how to apply deep learning.
- **From FastAI to Hugging Face**: Another user recommended the fastai course for a broad ML introduction and the [Hugging Face course](https://huggingface.co/learn/nlp-course/chapter1/1) for those interested in language models and transformers.
- **Searching for AI Direction**: For those uncertain about their ML specialization, it was suggested to begin with the first part of fastai before progressing to more specialized courses such as offered by Hugging Face.
- **April Fools or AI Breakthroughs?**: Two YouTube videos were shared, one titled "Introducing Jamba: AI21's Groundbreaking SSM-Transformer Model" and another "DBRX: A New State-of-the-Art Open LLM", but it was noted that it might be April Fools' content.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>: A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.</li><li><a href="https://huggingface.co/learn/nlp-course/chapter1/1">Introduction - Hugging Face NLP Course</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=dqFvOqC43rQ">DBRX: A New State-of-the-Art Open LLM</a>: Introducing DBRX, an open, general-purpose LLM created by Databricks. Across a range of standard benchmarks, DBRX sets a new state-of-the-art for established...</li><li><a href="https://www.youtube.com/watch?v=HRnx0ZPxe64">Introducing Jamba: AI21&#39;s Groundbreaking SSM-Transformer Model</a>: Introducing Jamba: AI21&#39;s Groundbreaking SSM-Transformer Modelhttps://www.ai21.com/blog/announcing-jamba#llm #ml #ai #deeplearning #largelanguagemodels #deep...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1223438081940783134)** (8 messages🔥): 

- **Replicating Microsoft's Ternary LLMs Achievements**: It appears that results from a Microsoft paper on ternary Large Language Models (LLMs) can be replicated, especially for the model range of 3 billion parameters at 100 billion operations. This discovery is elaborated on in the linked Hugging Face model [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B).

- **Nous Research Shares LLM Insights**: Nous Research made an announcement related to LLMs on Twitter, further details can be found on their [tweet](https://twitter.com/NousResearch/status/1773923241268003052).

- **Innovations in Transcription Service Comparisons**: A detailed analysis has been performed comparing various open-source whisper-based packages for long-form transcription, with a focus on accuracy and efficiency metrics. The findings are documented in a blog post [here](https://amgadhasan.substack.com/p/sota-asr-tooling-long-form-transcription).

- **Whisper Frameworks Put to the Test**: The open-source community has been active in enhancing OpenAI's Whisper model for long-form transcription, with varying error rates and efficiencies among frameworks like Huggingface Transformers and FasterWhisper. User experiences and preferences for these tools are being shared and discussed, with some considering WhisperX for its potential benefits.

- **Web Transformation Forecasts Major Shifts**: An article from F5 discusses upcoming changes to the web, projecting the evolution of search engines into inferential engines, the necessity for business models to adapt, and potential alterations to Web User Interfaces. The full read is available on the [F5 blog](https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19887">Jamba: A Hybrid Transformer-Mamba Language Model</a>: We present Jamba, a new base large language model based on a novel hybrid Transformer-Mamba mixture-of-experts (MoE) architecture. Specifically, Jamba interleaves blocks of Transformer and Mamba layer...</li><li><a href="https://x.com/teortaxesTex/status/1773861506674741570?t=GREWPQU25DwLqFZK4oOyNg&s=19">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: It seems that results of that Microsoft paper about ternary LLMs can be replicated after all – for 3B@100B at least. https://huggingface.co/1bitLLM/bitnet_b1_58-3B</li><li><a href="https://amgadhasan.substack.com/p/sota-asr-tooling-long-form-transcription">SOTA ASR Tooling: Long-form Transcription</a>: Benchmarking the different Whisper frameworks for long-form transcription</li><li><a href="https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos">Transforming the Web: The End of Silos</a>: The way we use the internet is about to change in a big way. The shift towards a more unified and efficient Web navigation method is a big leap from the traditional siloed Web browsing we’ve gotten us...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/)** (1 messages): 

teknium: https://twitter.com/NousResearch/status/1773923241268003052
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1223137982953820251)** (233 messages🔥🔥): 

- **A Quest for PII Anonymization**: Nous Hermes mistral 7b struggles with anonymizing **personal identifiable information (PII)** from text, leading to discussions about various models and datasets that could enhance this capability. Participants suggested using models like [open-llama-3b-v2-pii-transform](https://huggingface.co/filipealmeida/open-llama-3b-v2-pii-transform) but highlighted the need for improvement, with someone noting that [NousResearch plans to incorporate such data in future versions](https://huggingface.co/NousResearch/Genstruct-7B) of their model.

- **Meta-Prompting and Metaprompting LLMs**: There was a brief discussion about meta-prompting LLMs that create prompts and queries on whether ~7b models exist that excel in metaprompting.

- **Weighing Model Effectiveness**: Discussions revolved around the effectiveness and production-worthiness of models like Mamba and RWKV, delving into disagreements about their practicality, latency concerns, and integration into existing systems such as VLLM.

- **Explorations in Game-Based LLM AI**: A user shared progress on a game engine with an LLM, using payloads structured in Pydantic models to communicate actions and results. An ongoing project to integrate an LLM using a detailed system prompt for developing game state logic and interactions was highlighted.

- **Chasing Fast Inference**: Participants shared advances in speeding up LLM operations, like maximizing inference speeds on CPUs for llamafile and LLM leaks were advised to be skeptical of new LLM announcements, especially on the timestamped date.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/elonmusk/status/1773655245769330757">Tweet from Elon Musk (@elonmusk)</a>: Should be available on 𝕏 next week.   Grok 2 should exceed current AI on all metrics. In training now.  ↘️ Quoting xAI (@xai)   https://x.ai/blog/grok-1.5</li><li><a href="https://x.com/justinetunney/status/1774621341473489024">Tweet from Justine Tunney (@JustineTunney)</a>: I just made llamafile 1.3x - 5x faster than llama.cpp on CPU for many prompt / image evaluation use cases and hardware. https://justine.lol/matmul/</li><li><a href="https://api.wandb.ai/links/augmxnt/h4mc4dd5">Jamba Initial Tuning Runs</a>: Code here: https://github.com/shisa-ai/shisa-v2/tree/main/_base-evals/jamba  Initial experiments with doing fine tuning on Jamba using 1 x A100-80. Uses 99%&#43; of VRAM w/ these settings but does not...</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/salieri-mozart-gif-19031757">Salieri Mozart GIF - SALIERI MOZART - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1970zhf/merging_mistral_with_whisper_to_make_a_multimodal/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/furlat/Abstractions/tree/main/abstractions/goap/game">Abstractions/abstractions/goap/game at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://worldsim.nousresearch.com/?">world_sim</a>: no description found</li><li><a href="https://www.phoronix.com/news/Llamafile-0.7">Tweet from Llamafile 0.7 Brings AVX-512 Support: 10x Faster Prompt Eval Times For AMD Zen 4 - Phoronix</a>: no description found</li><li><a href="https://huggingface.co/filipealmeida/open-llama-3b-v2-pii-transform?text=%23%23%23+Instruction%3A%0AMy+name+is+Filipe+and+my+phone+number+is+555-121-2234.+How+are+you%3F%0A%23%23%23+Response%3A,">filipealmeida/open-llama-3b-v2-pii-transform · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets?sort=trending&search=pii">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://x.com/Shaughnessy119/status/1774081464721875087?s=20">Tweet from Tommy (@Shaughnessy119)</a>: New Podcast Episode with @theemozilla of @NousResearch 🎧  One of our smartest guests ever 🧠  ▪️ The inevitability of AGI ▪️ Crypto x AI Masterclass ▪️ Creating a World Simulator ▪️ Launching Bittens...</li><li><a href="https://huggingface.co/datasets/internlm/Agent-FLAN">internlm/Agent-FLAN · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/system_prompt.md">Abstractions/abstractions/goap/system_prompt.md at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/ai4privacy/pii-masking-200k">ai4privacy/pii-masking-200k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/joke2k/faker">GitHub - joke2k/faker: Faker is a Python package that generates fake data for you.</a>: Faker is a Python package that generates fake data for you. - joke2k/faker</li><li><a href="https://huggingface.co/filipealmeida/open-llama-3b-v2-pii-transform">filipealmeida/open-llama-3b-v2-pii-transform · Hugging Face</a>: no description found</li><li><a href="https://microsoft.github.io/presidio/">Microsoft Presidio</a>: no description found</li><li><a href="https://huggingface.co/spaces/beki/pii-anonymizer">Presidio with custom PII models trained on PII data generated by Privy - a Hugging Face Space by beki</a>: no description found</li><li><a href="https://huggingface.co/metricspace/EntityAnonymization-3B-V0.9">metricspace/EntityAnonymization-3B-V0.9 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/grammarly/pseudonymization-seq2seq">grammarly/pseudonymization-seq2seq · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/aymurai/anonymizer-beto-cased-flair">aymurai/anonymizer-beto-cased-flair · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/dslim/bert-large-NER">dslim/bert-large-NER · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1223168381440757791)** (64 messages🔥🔥): 

- **Choosing the Right RAG Configuration**: It's debated whether one big RAG across various themes or several specialized RAGs is more effective. Gabriel_syme suggests combining vertical and horizontal approaches, like domain clusters with hierarchical embeddings, but emphasizes that the best solution is use case specific. Links to structured document knowledge bases were requested but not provided.
- **Hermes Model Token Discrepancy Explained**: Technical discussions around padding related to **Hermes-2-Pro-Mistral-7B** on HuggingFace reveal inconsistencies in declared vocab sizes. Teknium mentions this is due to padding to a multiple of 32 to prevent issues from tensor parallelism on indivisible GPUs.
- **Deploying Fine-Tuned Models with a WebUI**: A Docker for Ollama with a web UI is available at [open-webui/open-webui on GitHub](https://github.com/open-webui/open-webui) to test fine-tuned models. Stoicbatman shared a script for easy Ollama setup, and others discussed loading models with adapters and CLI preferences.
- **Language Limitation in Hermes Models**: Discussion regarding the **Hermes 2 Pro** indicated it mainly functions reliably with English inputs and outputs. Benjoyo suggests using function calling with XML to prime the response when working with non-officially supported languages like Czech.
- **Finding and Fine-Tuning Japanese LLMs**: Embarking on tuning models for Japanese language capabilities, the conversation touched on shisa and qarasu as potential open-source options. Adapting existing models like sakana or Capybara for better context and language skills was proposed.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ollama/ollama/blob/06a1508bfe456e82ba053ea554264e140c5057b5/docs/modelfile.md#ada">ollama/docs/modelfile.md at 06a1508bfe456e82ba053ea554264e140c5057b5 · ollama/ollama</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama</li><li><a href="https://github.com/ollama/ollama/blob/06a1508bfe456e82ba053ea554264e140c5057b5/docs/modelfile.md#adapter">ollama/docs/modelfile.md at 06a1508bfe456e82ba053ea554264e140c5057b5 · ollama/ollama</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama</li><li><a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)</a>: User-friendly WebUI for LLMs (Formerly Ollama WebUI) - open-webui/open-webui</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/config.json#L25">config.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/added_tokens.json">added_tokens.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/tokenizer_config.json#L30">tokenizer_config.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1224260834235256874)** (1 messages): 

- **Challenges Accessing Traffic Signal Dataset**: A member pointed to a dataset containing **traffic signal images** that could aid in structured output and tool-use with vision models. However, they highlighted an issue where the [dataset viewer on Hugging Face](https://huggingface.co/datasets/Sayali9141/traffic_signal_images) is not available due to the execution of arbitrary Python code, and suggested opening a discussion for assistance on the matter.

**Link mentioned**: <a href="https://huggingface.co/datasets/Sayali9141/traffic_signal_images">Sayali9141/traffic_signal_images · Datasets at Hugging Face</a>: no description found

  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1223154575826223174)** (46 messages🔥): 

- **Decoding the Data Structure Dilemma**: A user questioned the necessity of using a JSON object and the avoidance of *overengineering*. The suggestion was to keep the data structure simple rather than adding complex layers.

- **Assessing the Merits of Metadata**: *Metadata* was proposed for wrapping messages within the RAG ecosystem to distinguish between various sources and responses. The method aims to manage structured sampling and maintain context integrity, especially in long-form conversations.

- **Envisioning a Multi-tool RAG Environment**: There was a speculative discussion about integrating RAG with other tools, addressing the challenges of conversation memory and potentially adopting a "simplified Github" to manage codebases and reversible edits.

- **Examining the Command+R Model's Potential**: The **Command-R** model is lauded for placing in the top 10 on a multiturn leaderboard, despite a lack of specific details on its training data. Its adeptness at handling long-context situations is highlighted and anticipated as a unique category in future benchmarks.

- **Exploring Real-world RAG Applications**: A dialogue unfolded about the practicality of RAG models in scenarios beyond conversational tasks, like assisting with complex tasks or codebase development. The challenges faced by software developers when using such AI tools in real-world settings were underscored.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1773814076063482038?s=20">Tweet from lmsys.org (@lmsysorg)</a>: [Arena Update]  @cohere&#39;s Command R is now top-10 in Arena leaderboard🔥  It&#39;s now one of the best open models reaching the level of top proprietary models. We find the model great at handling...</li><li><a href="https://docs.cohere.com/docs/prompting-command-r">Prompting Command-R</a>: no description found</li><li><a href="https://github.com/explodinggradients/ragas/tree/main">GitHub - explodinggradients/ragas: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines</a>: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines - explodinggradients/ragas
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1223155816622719046)** (176 messages🔥🔥): 

- **NSFW Discord Invite Spammer Neutralized**: There was an instance of a spammer sharing NSFW discord invites in the chat. The situation was handled promptly with reminders to ping specific roles or users when such incidents occur to get them resolved quickly.

- **OpenRouterAI Spotlights NousResearch**: An external [post by OpenRouterAI](https://x.com/openrouterai/status/1773738942350712907?s=46) highlighted NousResearch for having top Claude 3 Opus apps, stirring discussions on the economics of token output costs and how input tokens factor into the cost.

- **Hyperstition Discussion Piques Curiosity**: The concept of "Hyperstition" was elaborated within the channel, linking it to expanded cognitive domains triggered by interaction with LLMs. References to philosophers, such as Nick Land, and further in-depth discussions were meshed with practical use-case exploration in the sim.

- **WorldSim Commands and Modularity Explored**: Users discussed creative uses of WorldSim, including commands to emulate complex philosophical and esoteric setups, and the hypothetical integration of WorldSim with other systems like Websim.

- **Save and Share WorldSim Chats a Desired Feature**: Members expressed interest in the ability to save and share WorldSim chat sessions with URLs, accompanied by a discourse on the potential integrations and improvements to the platform, such as multiple saves, reload persisting chat histories, and collaborative elements in WorldSim.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>: no description found</li><li><a href="https://x.com/openrouterai/status/1773738942350712907?s=46">Tweet from OpenRouter (@OpenRouterAI)</a>: Ever wondered which apps are using an LLM? Now you can find out yourself with the new Apps tab.  @NousResearch has the top Claude 3 Opus apps this week 👀</li><li><a href="https://worldsim-web.vercel.app/">world_sim</a>: no description found</li><li><a href="https://tenor.com/view/her-theodore-joaquin-phoenix-scarlett-johannson-samantha-gif-5203383">Her Theodore GIF - Her Theodore Joaquin Phoenix - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gist.github.com/irl-dan/2be642f22c28bdacd92d1a2ac0172d8e">self-system-prompt</a>: self-system-prompt. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/irl-dan/595f74f17fc5b269c96e9f9f9079595b">strange-loop+claude3-self-modeling</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://en.wikipedia.org/wiki/Nick_Land">Nick Land - Wikipedia</a>: no description found</li><li><a href="https://tenor.com/view/dark-knight-joker-its-not-about-the-money-its-about-sending-a-message-gif-15254722">Dark Knight Joker GIF - Dark Knight Joker Its Not About The Money - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.nature.com/articles/s41467-024-46631-y">Alignment of brain embeddings and artificial contextual embeddings in natural language points to common geometric patterns - Nature Communications</a>: Here, using neural activity patterns in the inferior frontal gyrus and large language modeling embeddings, the authors provide evidence for a common neural code for language processing.</li><li><a href="https://gist.github.com/irl-dan/61e2f45eb1c9a879b39d480694a4c4a3">claude-world-modelling</a>: claude-world-modelling. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://www.orphandriftarchive.com/articles/hyperstition-an-introduction/">&#039;Hyperstition: An Introduction&#039; - 0rphan Drift Archive</a>: Delphi Carstens Interviews Nick Land. In the following interview Nick Land responds to some questions about the mechanisms of Hyperstition in the context of apocalypse. Q1. I wonder if you could elabo...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1223148778417033216)** (285 messages🔥🔥): 

- **Accessibility to LLMs on iPhone**: An app called **LLMFarm** is mentioned, which allows running various gguf models such as **Llava** locally on an iPhone.
- **Local LLM Performance vs. Cloud-Based LLMs**: The potential for a locally run LLM on an **RTX 4090** to outperform cloud-based counterparts such as **ChatGPT 4** is discussed, though opinions vary on the comparison.
- **Intriguing New Models and Integrations**: Users express interest in new model releases, such as **1bit llama2 7B**, and ponder the integration with tools such as *LM Studio*, *invoke.ai*, and *AutogenStudio*. Specific integration details include pull requests on GitHub.
- **Common Troubleshooting Tips**: Several messages offer guidance on LM Studio usage, including changing the download directory, altering UI font size, solving local installation issues, and handling version-related queries.
- **Seeking Streamlined Functionality and Support**: Users suggest ideas for improving LM Studio's efficiency and request features like voice recognition integration, code-free model management, and GUI-less operations for more advanced control.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/docs/welcome">Welcome | LM Studio</a>: LM Studio is a desktop application for running local LLMs on your computer.</li><li><a href="https://huggingface.co/mradermacher/goliath-120b-i1-GGUF">mradermacher/goliath-120b-i1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://www.youtube.com/watch?v=kFC-OWw7G8k">Build a Next.JS Answer Engine with Vercel AI SDK, Groq, Mistral, Langchain,  OpenAI, Brave &amp; Serper</a>: Building a Perplexity Style LLM Answer Engine: Frontend to Backend TutorialThis tutorial guides viewers through the process of building a Perplexity style La...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/sgemm.cpp">llamafile/llamafile/sgemm.cpp at main · Mozilla-Ocho/llamafile</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.</li><li><a href="https://github.com/openai/openai-python?tab=readme-ov-file#module-level-client">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web?tab=readme-ov-file">GitHub - ChatGPTNextWeb/ChatGPT-Next-Web: A cross-platform ChatGPT/Gemini UI (Web / PWA / Linux / Win / MacOS). 一键拥有你自己的跨平台 ChatGPT/Gemini 应用。</a>: A cross-platform ChatGPT/Gemini UI (Web / PWA / Linux / Win / MacOS). 一键拥有你自己的跨平台 ChatGPT/Gemini 应用。 - ChatGPTNextWeb/ChatGPT-Next-Web</li><li><a href="https://github.com/microsoft/autogen/pull/2199">Added ability to specify &#39;role&#39; field for select speaker messages for Group Chats (Replaces PR #2167) by marklysze · Pull Request #2199 · microsoft/autogen</a>: Note: This replaces #2167 due to it being based on an older version of main. Why are these changes needed? As per feature request #1861 (&quot;[Feature Request]: Allowing user to specify the &quot;rol...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6074">Add qwen2moe by simonJJJ · Pull Request #6074 · ggerganov/llama.cpp</a>: This PR adds the support of codes for the coming Qwen2 MoE models hf. I changed several macro values to support the 60 experts setting. @ggerganov</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6204">Add grok-1 support by arki05 · Pull Request #6204 · ggerganov/llama.cpp</a>: This pull request adds grok-1 support to llama.cpp (#6120). I&#39;ve added a separate MODEL_ARCH_GROK as to not clutter the LLAMA arch too much. The convert-hf-to-gguf.py can convert from keyfan/grok-...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1223338382080475276)** (50 messages🔥): 

- **Choosing Between Zephyr and Hermes2 Pro**: A question was raised comparing **Zephyr**'s and **Hermes2 Pro**'s capabilities in creative writing, with a user suggesting that **Starcoder2** as a better alternative due to its performance improvements and its smaller, faster design.
- **The Need for a Fine-tuning GUI**: A user speculated on the potential demand for a **GUI tool for fine-tuning** Large Language Models, noting the absence of such tools in contrast to LM Studio’s ease of use for inference. Another member suggested the idea might find more traction as a web-based application, considering the common use of cloud-based resources for fine-tuning.
- **Model Recommendations and Updates**: Users discussed new model releases, notably **Dolphin 2.8** and **Hercules 4.0**, both based on **Mistral v0.2.**, with links provided to access these models on Hugging Face ([Dolphin 2.8](https://huggingface.co/bartowski/dolphin-2.8-mistral-7b-v02-GGUF), [Hercules 4.0](https://huggingface.co/bartowski/Hercules-4.0-Mistral-v0.2-7B-GGUF)).
- **Troubleshooting LLMs on Apple Silicon Macs**: Users shared issues and offered support regarding difficulties running LLMs on **Apple Silicon M1 Macs**, addressing potential misunderstandings about memory types and problem-solving strategies, such as closing other applications to free up RAM/VRAM and monitoring for further issues.
- **GUI for LoRA and Document Loading**: One user introduced the idea of a GUI for **LoRA adaptation** of LLMs, suggesting a look at the [Microsoft LoRA](https://github.com/microsoft/LoRA) implementation. Additionally, a discussion took place over LM Studio's current lack of support for feeding documents to models, referencing external guides and affirming a demand for such a feature.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.marktechpost.com/2024/03/31/mistral-ai-releases-mistral-7b-v0-2-a-groundbreaking-open-source-language-model/">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=-Rs8-M-xBFI&ab_channel=TimCarambat">Stop paying for ChatGPT with these two tools | LMStudio x AnythingLLM</a>: In this video, we are installing two user-friendly tools that make downloading, running, and managing a powerful local LLM to replace ChatGPT. Seriously.Toda...</li><li><a href="https://github.com/microsoft/LoRA">GitHub - microsoft/LoRA: Code for loralib, an implementation of &quot;LoRA: Low-Rank Adaptation of Large Language Models&quot;</a>: Code for loralib, an implementation of &quot;LoRA: Low-Rank Adaptation of Large Language Models&quot; - microsoft/LoRA</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6344">Add support for DBRX models: dbrx-base and dbrx-instruct · Issue #6344 · ggerganov/llama.cpp</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6074">Add qwen2moe by simonJJJ · Pull Request #6074 · ggerganov/llama.cpp</a>: This PR adds the support of codes for the coming Qwen2 MoE models hf. I changed several macro values to support the 60 experts setting. @ggerganov
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1223395756988170342)** (14 messages🔥): 

- **Feature Requests Float In**: A user expressed satisfaction with LM Studio and listed feature requests like a *Unified Settings Menu*, *Keyboard Shortcuts*, *System Tray Icon*, *Plugins*, and *Horizontal Tabs*.
- **Plugin Wagon Gains Momentum**: Enthusiasm for the idea of **plugins** within LM Studio was echoed, mentioning that it hasn't been brought up in [feedback discussions](https://discord.com/channels/@me/1128339362015346749) yet.
- **Call for Remote GPU Support**: A suggestion was made for LM Studio to level up by offering support for **remote GPU** usage, similar to services like "juice labs," which allow for gaming on remote eGPUs but don't currently work with LM Studio.
- **Open Source Considerations Debated**: Queries were raised about the possibility of LM Studio being **open sourced**, considering its core reliance on `llma.cpp` and the target audience of Power Users and developers who value privacy, FOSS, and security.
- **Clarifications on Model Descriptions Sought**: Users discussed the lack of clear descriptions for differences between models like **Wizard** and **Wizard-Vicuna**, and the opportunity to have a community-created resource for model comparisons and recommendations. A few contributions were made, including links to model repositories and descriptions on Hugging Face, as well as historical context regarding model cards and their creators. 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF">TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/nlpxucan/WizardLM">GitHub - nlpxucan/WizardLM: LLMs build upon Evol Insturct: WizardLM, WizardCoder, WizardMath</a>: LLMs build upon Evol Insturct: WizardLM, WizardCoder, WizardMath - nlpxucan/WizardLM</li><li><a href="https://huggingface.co/cognitivecomputations/Wizard-Vicuna-13B-Uncensored">cognitivecomputations/Wizard-Vicuna-13B-Uncensored · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1223231765657686066)** (90 messages🔥🔥): 

- **GPU Offloading Option Vanishes After Update**: Upon upgrading from **mu 0.2.16 to 0.2.18**, a member reports an issue where the **GPU Acceleration** option disappears. The issue is identified as a **UI problem**, only presenting when the app window isn't in fullscreen mode.

- **A6000 GPUs Pushed to the Limit**: A member tests **Lllama 2 70B Q8** on a Threadripper 7995WX box with A6000 GPUs, observing substantial memory usage across GPUs and system memory while inquiring about multi-GPU executions and data transfers during model inferencing.

- **LM Studio Interface Challenged**: Several members discuss difficulties navigating LM Studio's UI, suggesting improvements such as a **search function in config** or reorganization for better discoverability.

- **Discussing Hardware Compatibility and Configuration**: Members share their experiences and queries regarding various GPUs like the **Arc A770** and **multi-GPU setups**, discussing usage of OpenCL, driver compatibility issues, and model loading problems post-update to LM Studio.

- **Hardware Recommendations and Tuning Insights Shared**: Concerning model training and fine-tuning in LM Studio, members offer guidance on supported models for lower-end systems, recommend GPUs with larger VRAM for ease of use, and suggest using external services like **Axolotl** and **RunPod** for fine-tuning tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/15rlqsb/how_to_perform_multigpu_parallel_inference_for/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/search/full-text?q=Llama2%2070B%20Q8">Full Text Search - Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1223199017610248242)** (4 messages): 

- **Exploring LMStudio's JSON Output**: A member has expressed interest in the new JSON output format from **LMStudio**, suggesting that it is crucial for building meaningful applications.
- **Seamless Transition to Open Source**: Another member shared their positive experience with integrating LMStudio with **langchain**, noting the transition was incredibly smooth.
- **Rapid RAG Integration**: The same member mentioned setting up a **Retrieval-Augmented Generation (RAG)** system using **llama 7B** in a matter of minutes with a small proof of concept repository they developed.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1223230801534255145)** (90 messages🔥🔥): 

- **Smooth Sailing on Version 0.2.18**: A user reported a good performance on Windows 11, 23H2 with a 7900XTX GPU achieving 96% load when using the LM Studio command `r`.

- **Unfamiliar GPU Behavior with New Update**: Another user encountered unusual results upon updating to LM Studio version 0.2.18. A previously unrecognized 7900 XTX GPU has begun functioning, while a second 7800 XT GPU is no longer having its VRAM recognized by LM Studio.

- **Troubleshooting with ROCm and OpenCL**: Through discussion, it was determined that an issue a user faced could be attributed to the transition from ROCm to AMD OpenCL. The resolution of the problem seemed to be associated with ensuring the correct version of LM Studio and updating AMD drivers.

- **iGPU Causing Complications in Model Loading**: An issue was identified around the integrated GPU (iGPU) on AMD systems where models failed to load using the ROCm tech preview. Disabling the iGPU was suggested as a temporary workaround.

- **Exploring AMD and LM Studio Compatibility Issues**: Dialogues reveal challenges and workarounds regarding AMD GPUs' compatibility with LM Studio, including issues related to CUDA-translator possibilities and strategies to prioritize high-performance settings within Windows system configurations for LM Studio usage.
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1223135417021239326)** (1 messages): 

- **Seeking Plug-and-Play Agent for LM Studio**: A user inquired about which **Agent program** is compatible with LM Studio for easy integration, looking for recommendations on plug-in options.
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1223319785140387853)** (2 messages): 

- **Voice Engine Mimics Real Speech**: OpenAI introduces [Voice Engine](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices), a model capable of generating natural-sounding speech using text input and a **15-second audio sample** to mirror the original speaker's voice. While the technology is impressive, OpenAI emphasizes a **cautious approach** to its release to prevent potential misuse.
  
- **ChatGPT Access Made Instant**: OpenAI is making [ChatGPT accessible without sign-up](https://openai.com/blog/start-using-chatgpt-instantly), allowing over 100 million users in 185 countries to engage with the AI instantly. This effort is part of the mission to make AI tools more broadly available and simplify the user experience with AI technologies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openai.com/blog/start-using-chatgpt-instantly">Start using ChatGPT instantly</a>: We’re making it easier for people to experience the benefits of AI without needing to sign-up</li><li><a href="https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices">Navigating the Challenges and Opportunities of Synthetic Voices</a>: We’re sharing lessons from a small scale preview of Voice Engine, a model for creating custom voices.
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1223177377731838095)** (98 messages🔥🔥): 

- **Exploring AI Help Applications**: One member questions if they should stick with **ChatGPT Plus** or explore new apps like **Poe** for using different AI models. They note that coding their own interface for **neovim** was limiting as it’s not accessible via phone.
- **Perplexity's Search Superiority**: A user recommends **Perplexity** for having the best search user interface, yet it does not allow for model selection and lacks a desktop app. Another user mentions the ability to choose models like **GPT-4** in the settings.
- **VoiceCraft Unveiled**: A link to **VoiceCraft's GitHub repository** is shared, showcasing its zero-shot **speech editing and text-to-speech capabilities**. The accompanying **paper and demo suggest impressive state-of-the-art performance** for various applications like audiobooks and podcasts ([GitHub - jasonppy/VoiceCraft](https://github.com/jasonppy/VoiceCraft), [VoiceCraft Paper/Demo](https://jasonppy.github.io/VoiceCraft_web/)).
- **AI Ethics and Misuse Discussion**: In light of technologies like VoiceCraft, conversation turns to the potential for misuse by bad actors, the legality of voice cloning, and the efficacy of OpenAI's cautious approach to releasing certain technologies.
- **Confusion Over API Choice for Business Summaries and Quizzes**: A member is uncertain whether to use the **completion API or assistant API** for generating business summaries and quiz titles, with another member advising that the **chat completion API offers more control** and to consult the documentation for further clarity ([API context management](https://platform.openai.com/docs/assistants/how-it-works/context-window-management)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jasonppy.github.io/VoiceCraft_web/">VoiceCraft</a>: no description found</li><li><a href="https://github.com/jasonppy/VoiceCraft">GitHub - jasonppy/VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild</a>: Zero-Shot Speech Editing and Text-to-Speech in the Wild - jasonppy/VoiceCraft
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1223337944727687328)** (31 messages🔥): 

- **Clarifying the ChatGPT Confusion**: Members discussed the common misconception that ChatGPT is an AI model, whereas it's actually an application using GPT AI models. The distinction between ChatGPT and GPT AI models often gets blurred in media and user discussions.
- **Navigating the API Maze for Quizzes**: A user sought advice on whether to use the completion or assistant API to generate engaging quiz titles from business information. However, there was no clear consensus or specific guidance provided in the chat.
- **Custom GPT - A Developer's Playground?**: Debate sparked over the capabilities of Custom GPTs; one member argued that unlike standard ChatGPT, Custom GPTs can undertake external actions such as automating video content creation and management, while another felt developers might bypass Custom GPT for direct API programming.
- **Request for Feedback on a Stock Analysis Bot**: A member introduced their newly created custom GPT, tailored for stock analysis and rating, and sought feedback from the community. The bot is designed to assess stocks and rate the attractiveness of buying them.
- **GPT's Reflective Abilities Under Scrutiny**: A user inquired about prompting LLMs to reflect internally and received guidance that LLMs function more like a black-box for text prediction, but a structured approach to problem-solving can make leaps in logic less likely.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1223261818080989365)** (167 messages🔥🔥): 

- **LaTeX Equation Struggles**: Members discussed difficulties in copying LaTeX equations from **ChatGPT** to **Microsoft Word**, with some suggesting Visual Studio Code as an alternative, but encountering issues when specifying exactly how **ChatGPT** should format its responses.

- **The Meta-prompting Debate**: An exchange took place regarding the effectiveness of **meta-prompting** for enhancing subjective quality in **AI responses**. Some advocated for it based on reported higher test scores, while others stressed the importance of one's own experience and experimentation over academic findings.

- **Model Quirks with Roleplaying**: Users shared peculiar behaviors observed in **GPT-4-0125**, especially when providing roleplaying instructions, with the model sometimes refusing to roleplay or follow the format if the instructions resemble directives.

- **Improving RAG Document Utilization**: A member highlighted the challenges in prompt engineering when **using RAG for company-specific documents**. Having a well-organized database optimized for RAG greatly improves results, and it was suggested to use **ChatGPT** itself to identify unclear concepts in documents.

- **Custom GPT for Stock Analysis Shared**: A user created a **custom GPT model** for stock analysis and welcomed feedback on its utility, providing a link for users to test and critique the model.

**Link mentioned**: <a href="https://openai.com/policies/terms-of-use">Terms of use</a>: no description found

  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1223261818080989365)** (167 messages🔥🔥): 

- **Equation Copy Conundrum**: Members discussed the possibility of copying equations from **ChatGPT** to **MS Word** and encountered issues with **Microsoft 365** compatibility. Workarounds suggested, such as using *LaTeX syntax*, were not consistently successful across different versions of Word provided by institutions.
  
- **Metaprompting Debates and Discoveries**: There was an in-depth discussion regarding the effectiveness of **metaprompting** in comparison to traditional prompting methods. While some users reported reading papers suggesting metaprompting scored higher on tests, others shared a preference for clear and direct instructions over metaprompts, highlighting the importance of replicating studies for personal validation.

- **Roleplay Restrictions in GPT-4 Models**: A user shared a problem with the **gpt-4-0125-preview** model refusing to roleplay under certain circumstances, even when given explicit instructions. Other members provided suggestions to overcome this, emphasizing a trial-and-error process, the potential effects of safety measures, and differences in model versions with varying complexity of instructions.

- **Prompt Engineering Tips**: A participant outlined several generalizable tips for effective prompt engineering, such as using positive framings, avoiding logical contradictions, leveraging conditional imperative logic, and understanding different bracket types indicative of certain functions.
  
- **Challenges in Utilizing RAG and GPT for Customer-Facing Bots**: One user highlighted the challenges of using **RAG** and **GPT** for customer service bots based on company-specific procedures, stressing the difficulties in preventing hallucinations while allowing for creativity in responses and the importance of database optimization for better interaction outcomes.

**Link mentioned**: <a href="https://openai.com/policies/terms-of-use">Terms of use</a>: no description found

  

---


**OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1223526644279545876)** (3 messages): 

- **Collate: A New Learning Tool?**: A member introduced **Collate** as a tool to streamline everyday learning processes. It's unclear from the mention what features or specifics are entailed in the Collate tool.

- **CrewAI Team Interaction**: Two members exchanged greetings, likely in reference to a project or context surrounding **CrewAI**. No details about CrewAI's purpose or functionality were provided.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1223138922444361769)** (260 messages🔥🔥): 

- **AI's Armageddon Scale**: The discord community humorously debated the probability of AI going rogue, pegging the average concern level at 3.2, with some interpreting the high theoretical potential risk as 'infinity'.
- **The Grammar Games**: Discussions veered into the grammatical intricacies of the word "axis", leading to a sharing of resources explaining the correct usage and plural forms.
- **The Quest for Human-Level AI**: An extensive debate unfolded around whether AI could achieve and surpass human-level intelligence, considering factors like Moore's Law, hardware advancements, and AI alignment necessary for a safe AI integration into society.
- **Publications and Predictions**: Participants discussed recent papers on AI, weighing the pros and cons of various methods like adding more AI agents to enhance performance, with skepticism around some of the optimistic statements made by figures like Andrew Ng.
- **Kye Gomez's Curious Creations**: The group humorously reflected on various GitHub repositories by Kye Gomez, questioning their legitimacy and potential implications on scientific reproducibility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)">no title found</a>: no description found</li><li><a href="https://www.grammar-monster.com/plurals/plural_of_axis.htm">The Plural of Axis</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.05120">More Agents Is All You Need</a>: We find that, simply via a sampling-and-voting method, the performance of large language models (LLMs) scales with the number of agents instantiated. Also, this method is orthogonal to existing compli...</li><li><a href="https://arxiv.org/abs/2310.01798">Large Language Models Cannot Self-Correct Reasoning Yet</a>: Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the ...</li><li><a href="https://tenor.com/view/band-the-muppets-rock-out-rocking-out-i-dont-like-spam-gif-5375842">Band The Muppets GIF - Band The Muppets Rock Out - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/modularml/mojo/tree/nightly">GitHub - modularml/mojo at nightly</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/23e8fa5a263d1c7122bc46a86ef32030ee7130f9/benchmarks/benchmark_flash_attention.py#L27">flash-attention/benchmarks/benchmark_flash_attention.py at 23e8fa5a263d1c7122bc46a86ef32030ee7130f9 · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/kyegomez/swarms">GitHub - kyegomez/swarms: Build, Deploy, and Scale Reliable Swarms of Autonomous Agents for Workflow Automation. Join our Community: https://discord.gg/DbjBMJTSWD</a>: Build, Deploy, and Scale Reliable Swarms of Autonomous Agents for Workflow Automation. Join our Community: https://discord.gg/DbjBMJTSWD - kyegomez/swarms
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1223228364085788723)** (169 messages🔥🔥): 

- **Skepticism Over MoE with Heterogeneous Expert Sizes**: Discussion around *MoE with heterogeneous expert sizes* involved conflicting opinions. While the theoretical design suggests flexibility with different sized experts within a layer, practical reports suggest actual performance doesn't quite match the impressive benchmarks claimed.
  
- **BitNet b1.58 Reproduced and Disputed**: The claimed benefits of the BitNet b1.58 model are under scrutiny as independent reproduction by NousResearch, detailed in a [Hugging Face repository](https://huggingface.co/NousResearch/OLMo-Bitnet-1B), suggests that it may be less efficient than its FP16 counterparts despite official papers indicating otherwise. Skepticism remains over whether the claims will hold true when scaled up.

- **Evaluating FID for Image Generation Benchmarks**: Concerns were raised about the effectiveness of the Frechet Inception Distance (FID) in evaluating image generation methods. An [alternative proposal](https://arxiv.org/abs/2401.09603v2) argues that FID's underlying assumptions and poor sample complexity could contradict human judgments and warrants reevaluation as the primary metric.

- **Anticipation for Potential Optimization Breakthrough**: There's anticipation and speculation over a new optimization technique teased by a Meta researcher, suggesting better results than Adam with no memory overhead. Comparisons were drawn to existing techniques and previous studies of optimizer performance, but conclusive information awaits further details.

- **Tuning Text Summarization Models**: Exchange of insights and references into starting models suitable for SFT on TLDR text summarization. Models such as Pythia and others are being considered with variability in performance, resource availability also shaping the decisions for setting up experiments.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1773923241268003052">Tweet from Nous Research (@NousResearch)</a>: We are releasing our first step in validating and independently confirming the claims of the Bitnet paper, a 1B model trained on the first 60B tokens of the Dolma dataset.  Comparisons made on the @we...</li><li><a href="https://proceedings.mlr.press/v97/muehlebach19a.html">A Dynamical Systems Perspective on Nesterov Acceleration</a>: We present a dynamical system framework for understanding Nesterov’s accelerated gradient method. In contrast to earlier work, our derivation does not rely on a vanishing step size argument. We sho...</li><li><a href="https://www.jmlr.org/papers/v22/20-195.html">A Lyapunov Analysis of Accelerated Methods in Optimization</a>: no description found</li><li><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">1bitLLM/bitnet_b1_58-3B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2401.09603v2">Rethinking FID: Towards a Better Evaluation Metric for Image Generation</a>: As with many machine learning problems, the progress of image generation methods hinges on good evaluation metrics. One of the most popular is the Frechet Inception Distance (FID). FID estimates the d...</li><li><a href="https://arxiv.org/abs/2403.19928">DiJiang: Efficient Large Language Models through Compact Kernelization</a>: In an effort to reduce the computational load of Transformers, research on linear attention has gained significant momentum. However, the improvement strategies for attention mechanisms typically nece...</li><li><a href="http://arxiv.org/abs/2403.20327">Gecko: Versatile Text Embeddings Distilled from Large Language Models</a>: We present Gecko, a compact and versatile text embedding model. Gecko achieves strong retrieval performance by leveraging a key idea: distilling knowledge from large language models (LLMs) into a retr...</li><li><a href="https://app.suno.ai/song/1227d716-93df-4cc9-9be4-5b10fcb083a9/">The Facade of Futurism | Suno</a>: Epic hardstyle song. Listen and make your own with Suno.</li><li><a href="https://arxiv.org/abs/2107.09133">The Limiting Dynamics of SGD: Modified Loss, Phase Space Oscillations, and Anomalous Diffusion</a>: In this work we explore the limiting dynamics of deep neural networks trained with stochastic gradient descent (SGD). As observed previously, long after performance has converged, networks continue to...</li><li><a href="https://arxiv.org/abs/2012.04728">Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics</a>: Understanding the dynamics of neural network parameters during training is one of the key challenges in building a theoretical foundation for deep learning. A central obstacle is that the motion of a ...</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://github.com/vwxyzjn/summarize_from_feedback_details">GitHub - vwxyzjn/summarize_from_feedback_details</a>: Contribute to vwxyzjn/summarize_from_feedback_details development by creating an account on GitHub.</li><li><a href="https://link.springer.com/article/10.1007/s11098-023-02042-1">Borderline consciousness, when it’s neither determinately true nor determinately false that experience is present - Philosophical Studies</a>: This article defends the existence of borderline consciousness. In borderline consciousness, conscious experience is neither determinately present nor determinately absent, but rather somewhere betwee...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1224227305522729070)** (4 messages): 

- **Logits Tweaking Before Softmax Clarified**: The discussion clarified that the adjustment to logits happens before every single softmax in the network, encompassing both attention mechanisms and the final head. It's an all-encompassing approach just prior to probability distribution decisions.
- **Catboy Forgets About Softmax**: A brief lapse in memory about the softmax function was openly acknowledged, followed by a clear and appreciative acknowledgment—indicating effective communication among community members.
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1223323048430796972)** (3 messages): 

- **Sparse Autoencoders under Scrutiny**: A research post identifies a potential issue where reconstruction errors in **Sparse Autoencoders (SAEs)** significantly change model predictions more than an equivalent random error. The discussed findings can be found in this [Short research post](https://x.com/wesg52/status/1773756298531918268).

- **Visualizing SAE Features Made Easier**: A new visualization library for SAE features has been developed and shared, proving to be very useful for researchers working with Sparse Autoencoders. The library announcement and details can be accessed via [SAE Vis Announcement Post](https://x.com/neelnanda5/status/1774463606656282806).

- **Insights into Sparse Autoencoder Features**: A post shares a selection of Sparse Autoencoder features, discussing the meaningful computational structure they reveal within the model. The significance of these features for AI alignment and the question of whether they reflect model properties or data distributions is explored in this [interpretation piece](https://www.lesswrong.com/posts/BK8AMsNHqFcdG8dvt/a-selection-of-randomly-selected-sae-features-1).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/wesg52/status/1773756298531918268">Tweet from Wes Gurnee (@wesg52)</a>: Short research post on a potential issue arising in Sparse Autoencoders (SAEs): the reconstruction errors change model predictions much more than a random error of the same magnitude! https://www.less...</li><li><a href="https://x.com/neelnanda5/status/1774463606656282806">Tweet from Neel Nanda (@NeelNanda5)</a>: Great visualisation library for Sparse Autoencoder features from @calsmcdougall! My team has already been finding it super useful, go check it out: https://www.lesswrong.com/posts/nAhy6ZquNY7AD3RkD/sa...</li><li><a href="https://www.lesswrong.com/posts/BK8AMsNHqFcdG8dvt/a-selection-of-randomly-selected-sae-features-1">A Selection of Randomly Selected SAE Features — LessWrong</a>: In this post, we interpret a small sample of Sparse Autoencoder features which reveal meaningful computational structure in the model that is clearly…
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1223164830110978089)** (12 messages🔥): 

- **DBRX Model Load Issues on lm-eval harness**: A user encountered memory allocation issues when trying to load the **DBRX base model** into **lm-eval harness**. They resolved the problem by updating their software version and realizing they were on a node with fewer GPUs than required.

- **New PR for lm-evaluation-harness**: A member submitted a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1571) proposing a new strategy for handling context-based tasks in **lm-evaluation-harness**, eager for feedback to improve their code.

- **lm-eval-harness Task Troubleshooting**: A newcomer to **lm-eval-harness** sought assistance for an error stating their task was not found. They were advised to ensure the task is under `lm_eval/tasks` or specify its path using certain commands and to enable debug logs for detailed error reporting.

- **Clarifying OPT Token Handling in lm-eval-harness**: Inquiring about the handling of the start token for **OPT models** in **lm-eval-harness**, a user learned that this is managed by setting `add_bos_token=True` in the model's arguments.

- **Music Generation Model Interactive Evaluation**: The text [highlights an arXiv submission](https://arxiv.org/abs/2402.15294) reviewing music representation, algorithms, and evaluation measures, hinting at developing a leaderboard for text-to-music generation metrics.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.15294">A Survey of Music Generation in the Context of Interaction</a>: In recent years, machine learning, and in particular generative adversarial neural networks (GANs) and attention-based neural networks (transformers), have been successfully used to compose and genera...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1571).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1224104077731041280)** (3 messages): 

- **Uneven Batch Sizes Pose Challenge**: A member inquired about setting a **global batch size** not aligned with the number of GPUs in **NeoX**. Another member explained that while it's possible to hack NeoX for uneven batch sizes, it would lead to **load imbalance** and be limited by the GPUs with the larger batch size.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1223140213497532496)** (366 messages🔥🔥): 

- **DBRX Language Model Repo Reshared**: A non-gated re-upload of the **DBRX Base model** was shared due to the original repository being gated. DBRX Base is a large language model with mixture-of-experts, and the re-upload is meant to emphasize the importance of open weights and easy access. The original repo for DBRX Base and DBRX Instruct is accessible on [Hugging Face](https://huggingface.co/Undi95/dbrx-base).

- **Finding the Optimal Ancestral Method**: An individual remarked that using the euler ancestral method yields the best results on terminus. This opinion was supported by images [like this example](https://tripleback.net/public/discord//1711690549.43835748fa38cc8f588f4fcf330e4ac72b149fb.png), and a claim was made that a particular Chinese sign with a humorous translation demonstrates this method's benefits.

- **Discussions of AI Music Generation Quality**: Members discussed the quality of AI music generation tools like Suno, speculating on the versions and comparing v2 to v3. Issues such as the presence of noise layers in voices and the anticipation for further improvements in future versions like v4 were touched upon, with links to musical examples shared, such as from [mockingbird's YouTube channel](https://www.youtube.com/watch?v=9K2tkWjOTDU&list=PLMzBiaqOoxQWBs7UpS2WW_Qf3zSuWOux_).

- **Concerns about AI Voice Synthesis Technologies**: A member highlighted concerns over OpenAI's Voice Engine news release potentially overshadowing Voicecraft's recent release, considering the timing suspicious. The discussion also ventured into speculations about OpenAI’s strategic moves, including overshadowing competitors like Midjourney, market dynamics involving API access, and the potential misuse of voice synthesis technology in the context of the US Elections.

- **Stochastic Rounding for AI Training**: Conversations between members discussed strategies for AI training, including the utilization of stochastic rounding techniques. A repository, [nestordemeure/stochastorch](https://github.com/nestordemeure/stochastorch), was shared as a Pytorch implementation of stochastic addition, which could be useful in optimizing training performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=39865810">no title found</a>: no description found</li><li><a href="https://huggingface.co/Undi95/dbrx-base">Undi95/dbrx-base · Hugging Face</a>: no description found</li><li><a href="https://x.com/aaron_defazio/status/1773381393831067787?s=46&t=4T8Ef2avsCyyYLhfPtqVfg">Tweet from Aaron Defazio (@aaron_defazio)</a>: Cooking up something special! Can&#39;t wait to get a paper out so everyone can try it out. An optimizer with no extra overhead, no additional parameters. Stay tuned!</li><li><a href="https://fxtwitter.com/aaron_defazio/status/1773726259924676975">Tweet from Aaron Defazio (@aaron_defazio)</a>: Update: more experimental results rolling in. Here it is against SGD with both the step-wise and cosine schedule (both baselines heavily tuned, no cheating) This is something special indeed!</li><li><a href="https://tenor.com/view/mcqueen-gif-25874245">Mcqueen GIF - Mcqueen - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/emUfjaMB3Fa.gif">Believe Motivation GIF - Believe Motivation Ted lasso - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1/tree/main">ai21labs/Jamba-v0.1 at main</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://tenor.com/q1Vl.gif">Team Bonding GIF - Silicon Valley Dinesh Jared Dunn - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://globalnews.ca/news/10389187/justin-trudeau-deepfake-youtube-ad/">Justin Trudeau deepfake ad promoting ‘robot trader’ pulled off YouTube - National | Globalnews.ca</a>: A deepfake advertisement depicting Prime Minister Justin Trudeau&#039;s likeness promoting a financial “robot trader” has been pulled off YouTube.</li><li><a href="https://www.engadget.com/openai-says-it-can-clone-a-voice-from-just-15-seconds-of-audio-190356431.html?guccounter=1">OpenAI says it can clone a voice from just 15 seconds of audio</a>: OpenAI just announced the results of a small-scale preview of a new voice cloning engine that&#x2019;s based on the company&#x2019;s pre-existing text-to-speech API. The technology generates natural-s...</li><li><a href="https://x.com/dorialexander/status/1773776329181135187">Tweet from Alexander Doria (@Dorialexander)</a>: After SORA, starting to wonder if OpenAI is specializing on the AI tech with the highest risk of misuse to have better rationale to never release anything.  ↘️ Quoting Andrew Curran (@AndrewCurran_)  ...</li><li><a href="https://www.youtube.com/watch?v=ZwF4aIm0RJk">mockingbird- Autonomous sensory meridian response puppet</a>: = AUTONOMOUS SENSORY MERIDIAN RESPONSE PUPPET =_________________________________________________In the still of night, where shadows creep,A cybernetic puppe...</li><li><a href="https://github.com/pytorch/pytorch/issues/120376>">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://www.youtube.com/watch?v=a8blrTtacBA">mockbingbird - sirena cibernetica (bonus track)</a>: = SIRENA CIBERNETICA =___________________________(c) @mockingbirdAI - made with Suno</li><li><a href="https://www.youtube.com/watch?v=9K2tkWjOTDU&list=PLMzBiaqOoxQWBs7UpS2WW_Qf3zSuWOux_">mockingbird - Neural hacker</a>: = NEURAL HACKER =___________________________Ahahah!Yo, newb, you ready for a trip down my twisted lane?Dive into my digital chaos, escape the mundane.With a ...</li><li><a href="https://github.com/endomorphosis/ipfs_transformers">GitHub - endomorphosis/ipfs_transformers: a model manager for the Transformers library, implementing S3 and IPFS downloads</a>: a model manager for the Transformers library, implementing S3 and IPFS downloads - endomorphosis/ipfs_transformers</li><li><a href="https://github.com/nestordemeure/stochastorch">GitHub - nestordemeure/stochastorch: A Pytorch implementation of stochastic addition.</a>: A Pytorch implementation of stochastic addition. Contribute to nestordemeure/stochastorch development by creating an account on GitHub.</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/">GitHub - PKU-YuanGroup/Open-Sora-Plan: This project aim to reproduce Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project.</a>: This project aim to reproduce Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project. - PKU-YuanGroup/Open-Sora-Plan</li><li><a href="https://github.com/huggingface/diffusers/pull/7530/files">7529 do not disable autocast for cuda devices by bghira · Pull Request #7530 · huggingface/diffusers</a>: What does this PR do?   Fixes #7529 Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other checks if that&amp;#39;s the case).  Did you read the contributor guideline...</li><li><a href="https://github.com/huggingface/diffusers/issues/551">Merging Stable diffusion pipelines just makes sense · Issue #551 · huggingface/diffusers</a>: Following the Philosophy, it has been decided to keep different pipelines for Stable Diffusion for txt-to-img, img-to-img and inpainting. Here is the result: PR #549 : code duplicated 4 times (onnx...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1223252152752013353)** (42 messages🔥): 

- **Exploring UNETs and Transformers in Diffusion**: In the research channel, there was an inquiry about learning more about UNETs and the complexities of creating a transformer version of diffusion. A [research paper](https://arxiv.org/pdf/2212.09748.pdf) was shared explaining how UNETs were replaced with transformers for such tasks.

- **High-Level Explanation of UNETs**: One member offered an explanation of UNETs, describing them as structures encoding an image into lower-dimensional space and then upsampling that representation back to the original space, suggesting the process involves discarding redundant information to simplify reconstruction.

- **Unveiling Qwen1.5-MoE-A2.7B**: Discussion sparked around Qwen1.5-MoE-A2.7B, a new MoE model that reportedly matches the performance of larger models like Mistral 7B with only 2.7 billion activated parameters. Information and resources related to Qwen1.5 were shared in the channel by members, highlighting its potential based on initial results shown ([GitHub](https://github.com/QwenLM/Qwen1.5), [Hugging Face](https://huggingface.co/Qwen), [ModelScope](https://modelscope.cn/organization/qwen), [Demo](https://huggingface.co/spaces/Qwen/Qwen1.5MoE-A2.7B-Chat), [Discord](https://discord.gg/yPEP2vHTu4)).

- **Video Lava Augmentation with V-JEPA**: Members discussed the prospect of enhancing video Lava using V-JEPA embeddings, with a GitHub repository linked as a resource ([V-JEPA GitHub](https://github.com/facebookresearch/jepa)). A focuses shift towards the integration of such embeddings and data preparation for the training.

- **Innovative Approaches in Diffusion and Embedding Models**: There was interest in a paper discussing a new diffusion loss function which may provide robustness to outliers, potentially improving diffusion models ([paper link](https://arxiv.org/abs/2403.16728)). Additionally, the Gecko text embedding model's efficiency via the distillation process from large language models was highlighted as a resource for potentially accelerating model training ([Gecko paper link](https://arxiv.org/abs/2403.20327)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction Since the surge in interest sparked by Mixtral, research on mixture-of-expert (MoE) models has gained significant momentum. Both researchers an...</li><li><a href="https://arxiv.org/abs/2403.16728">Improving Diffusion Models&#39;s Data-Corruption Resistance using Scheduled Pseudo-Huber Loss</a>: Diffusion models are known to be vulnerable to outliers in training data. In this paper we study an alternative diffusion loss function, which can preserve the high quality of generated data like the ...</li><li><a href="https://huggingface.co/papers/2403.20327">Paper page - Gecko: Versatile Text Embeddings Distilled from Large Language Models</a>: no description found</li><li><a href="https://github.com/facebookresearch/jepa?tab=r">GitHub - facebookresearch/jepa: PyTorch code and models for V-JEPA self-supervised learning from video.</a>: PyTorch code and models for V-JEPA self-supervised learning from video. - facebookresearch/jepa</li><li><a href="https://github.com/facebookresearch/jepa?tab=readme-ov-file">GitHub - facebookresearch/jepa: PyTorch code and models for V-JEPA self-supervised learning from video.</a>: PyTorch code and models for V-JEPA self-supervised learning from video. - facebookresearch/jepa
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1223143579086749696)** (225 messages🔥🔥): 

- **Confusion Over `huggingface-cli` Command**: A user reported an issue with the `huggingface_cli` not being recognized and was advised to run `pip install -U "huggingface_hub[cli]"`. There was a follow-up clarification regarding using a terminal instead of PowerShell and suggesting to create a new environment that resolved the problem.

- **AI Image Generation Advances Questioned**: One member questioned if there had been any real improvements in AI image generation since the previous year. Others responded, mentioning recent advancements like Stable Cascade's new architecture, the ability to input sketches and poses, and models like OOT diffusion that offer more control and realistic outputs.

- **Curiosity About Qualifications and Learning Paths in AI**: A discussion unfolded around what it takes to become skilled in machine learning. The consensus from several members is that understanding the underlying foundations of machine learning, such as architectures, is vital if you aim to innovate, while others suggested that practical experience and projects can be done independently without internships.

- **Warnings of AI-Induced Vulnerabilities**: There was a noteworthy mention of AI hallucinating non-existent software packages, leading companies to incorporate them into their source code. This highlights the dangers of AI generating convincing but fictitious information that could be utilized for spreading malware.

- **Assistance Sought for Various Technical Challenges**: Users sought help on a range of topics from setting up LLM coding environments, AI conferences recommendations, and running models like tensorRT-LLM on AWS EC2 with Ubuntu. There were queries about fine-tuning language models on PDF files and inquiries on how to deal with summary pipeline outputs being too lengthy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/noob_intro_transformers">Total noob’s intro to Hugging Face Transformers</a>: no description found</li><li><a href="https://www.redwoodcompliance.com/the-power-of-neural-networks-and-why-it-can-learn-a-lot-but-not-everything/">no title found</a>: no description found</li><li><a href="https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/">AI bots hallucinate software packages and devs download them</a>: Simply look out for libraries imagined by ML and make them real, with actual malicious code. No wait, don&#39;t do that</li><li><a href="https://huggingface.co/databricks/dbrx-instruct">databricks/dbrx-instruct · Hugging Face</a>: no description found</li><li><a href="https://aws.amazon.com/machine-learning/amis/">Deep Learning Virtual Machine  - AWS Deep Learning AMIs - AWS</a>: no description found</li><li><a href="https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html">Release Notes for DLAMI - Deep Learning AMI</a>: no description found</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://sambanova.ai/blog/accurate-models-at-blazing-speed">SambaNova Delivers Accurate Models At Blazing Speed</a>: Samba-CoE v0.2 is climbing on the AlpacaEval leaderboard, outperforming all of the latest open-source models. </li><li><a href="https://youtu.be/kebSR2Ph7zg">Inside the Black Box: Convolutional Neural Nets Visualized!</a>: In this video, I dive into Convolutional Neural Networks - WHAT they are, HOW they learn, and WHY they are so successful on computer vision tasks. The video ...</li><li><a href="https://github.com/milmor/diffusion-transformer?tab=readme-ov-file">GitHub - milmor/diffusion-transformer: Implementation of Diffusion Transformer Model in Pytorch</a>: Implementation of Diffusion Transformer Model in Pytorch - milmor/diffusion-transformer</li><li><a href="https://www.openwall.com/lists/oss-security/2024/03/29/4">oss-security - backdoor in upstream xz/liblzma leading to ssh server compromise</a>: no description found</li><li><a href="https://github.com/endomorphosis/ipfs_transformers">GitHub - endomorphosis/ipfs_transformers: a model manager for the Transformers library, implementing S3 and IPFS downloads</a>: a model manager for the Transformers library, implementing S3 and IPFS downloads - endomorphosis/ipfs_transformers</li><li><a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper</a>: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper - developersdigest/llm-answer-engine</li><li><a href="https://github.com/NVIDIA/TensorRT-LLM">GitHub - NVIDIA/TensorRT-LLM: TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines.</a>: TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...</li><li><a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html">Installation Guide :: NVIDIA Deep Learning TensorRT Documentation</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1223381294004568154)** (7 messages): 

- **AI Takes on Music Mashups**: A member discussed the complexity of creating AI covers by adjusting the keys of different songs to match them, highlighting the challenge of blending voices from different artists like Selena and Taylor in a harmony similar to that of Little Mix.

- **Exploring the Limits of Microtransfer Learning**: The advancement in *µTransfer* reproduction was shared, reaching a significant 15%. Also, a critical bug in **DoReMi** was fixed, improving its downstream performance compared to that using a uniform distribution.

- **DarkWebSight: Synthetic Data from the Shadows**: The channel showcased a step-by-step method to generate a synthetic dataset called *DarkWebSight* for Tor hidden services. It included creating website layout ideas, generating the code for these concepts in HTML/CSS, and then formatting the results in a JSON structure.

- **Deluge of Darknet Designs**: A member posted numerous layout ideas for various hypothetical Tor hidden services, including a darknet market site and a whistleblower platform, envisioned with unique design elements and color themes.
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1223292992325161043)** (12 messages🔥): 

- **1-Bit Language Model Weights Available**: Hugging Face has released **1.38 bit quantized model weights** for LLMs, promising a step towards more efficient AI models. Here's the link to check it out: [1bitLLM on Hugging Face](https://huggingface.co/1bitLLM).

- **LangChain Embraces Chat History**: A Medium blog post details how integrating chat history with **LangChain** can enhance conversational AI. You can read more about it [here](https://medium.com/ai-advances/integrating-chat-history-with-langchain-enhancing-conversational-ai-4c130ff2963c).

- **Discovering Linguistic Brain Structures**: Research suggests links between **deep language models** and human brain word embeddings that shape language representation. [Learn more about this research](https://www.nature.com/articles/s41467-024-46631-y).

- **Improved Rechargeable Magnesium Batteries on the Horizon**: Researchers found a way to modify rocksalt oxides for **rechargeable magnesium batteries**, potentially leading to higher energy densities. [Read the scientific publication](https://pubs.rsc.org/en/content/articlelanding/2024/ta/d3ta07942b).

- **FastLLM Unveiled by Qdrant**: **FastLLM**, a new language model designed for **Retrieval Augmented Generation** with a gigantic context window of 1 billion tokens, has been introduced in early access. Discover **FastLLM** at [Qdrant's announcement blog](https://qdrant.tech/blog/fastllm-announcement/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qdrant.tech/blog/fastllm-announcement/">Introducing FastLLM: Qdrant’s Revolutionary LLM - Qdrant</a>: Lightweight and open-source. Custom made for RAG and completely integrated with Qdrant.</li><li><a href="https://huggingface.co/1bitLLM">1bitLLM (1bitLLM)</a>: no description found</li><li><a href="https://pubs.rsc.org/en/content/articlelanding/2024/ta/d3ta07942b">Securing cation vacancies to enable reversible Mg insertion/extraction in rocksalt oxides</a>: Oxide cathode materials have promising applications in rechargeable magnesium batteries (RMBs) due to their high redox potential, which allows the exploitation of the low potential of Mg metal anodes ...</li><li><a href="https://www.nature.com/articles/s41467-024-46631-y">Alignment of brain embeddings and artificial contextual embeddings in natural language points to common geometric patterns - Nature Communications</a>: Here, using neural activity patterns in the inferior frontal gyrus and large language modeling embeddings, the authors provide evidence for a common neural code for language processing.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1223150816895176715)** (41 messages🔥): 

- **Blurry Model Dilemma**: A member expressed confusion about why their trained model's results are blurry compared to others' sharper results, despite utilizing 300 images, 3 repeats, and 10 epochs for the same subject. The sharpness and realism were the main concerns.
- **PAG Enhances Without Reducing Diversity**: A user shared insights into the strengths of Perturbed-Attention Guidance (PAG) over Classifier-Free Guidance (CFG), noting that PAG can improve sample quality without sacrificing diversity. They provided a starting point for mixing CFG and PAG, recommending a ratio of CFG 4.5 and PAG between 3.0 to 7.0 for better prompt-following characteristics.
- **Feedback Sought for PII Detection Project**: A member requested suggestions for models to use in a PII Detection project that has so far utilized Text Mining models and BERT. They sought advice on alternatives to enhance the project.
- **Terraform Provider for Hugging Face Spaces**: An individual highlighted the creation of a Terraform provider that allows spinning up Hugging Face Spaces using Terraform code. The tool is part of their project *mlstacks*, aimed at setting up ML-related infrastructure, and they welcomed feedback and functionality suggestions.
- **Launching OneMix SaaS Boilerplate**: A member introduced OneMix, a Remix-based SaaS boilerplate, designed to save time on common development tasks like landing pages, authentication, and payment integration. A demo video was shared, and they encouraged feedback on their SaaS boilerplate.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/magicalapi/YouTube_Thumbnail_Suggestion">magicalapi/YouTube_Thumbnail_Suggestion · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/#fileId=https%3a%2f%2fdagshub.com%2fDagsHub%2fDagsHubxColab%2fraw%2fmain%2fDagsHub_x_Colab-DagsHub_Storage.ipynb">Google Colaboratory</a>: no description found</li><li><a href="https://mlops.systems/posts/2024-03-31-writing-a-custom-terraform-provider-to-deploy-huggingface-spaces.html">Alex Strick van Linschoten - Writing a custom Terraform provider to deploy Huggingface Spaces</a>: I worked on this short project to allow people to create/deploy Huggingface Spaces using Terraform (instead of via the API or using the website)</li><li><a href="https://huggingface.co/collections/pszemraj/boulderspot-660705d965dc00609ece3178">boulderspot - a pszemraj Collection</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.17377">Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance</a>: Recent studies have demonstrated that diffusion models are capable of generating high-quality samples, but their quality heavily depends on sampling guidance techniques, such as classifier guidance (C...</li><li><a href="https://www.youtube.com/watch?v=9YmcekQUJPs">An advanced line follower and wall follower robot with colour sensor. Presented by SUST_BlackAnt</a>: This is an advanced line follower track. Feel free to like, comment and share. Let me know how you like it. If you want to contact me feel free to send an em...</li><li><a href="https://github.com/endomorphosis/ipfs_transformers">GitHub - endomorphosis/ipfs_transformers: a model manager for the Transformers library, implementing S3 and IPFS downloads</a>: a model manager for the Transformers library, implementing S3 and IPFS downloads - endomorphosis/ipfs_transformers</li><li><a href="https://youtu.be/vDyonow9iLo">Mojo Programming Language killed Python</a>: I&#39;ll share with you why Mojo will be very popular very soon. It&#39;s killing Python performance wise making it very competitive and the key is: while keeping th...</li><li><a href="https://youtu.be/1KCFHoSGckY">On AI Policy: Don&#39;t Lose, Heart. A read on the White House AI Update.</a>: An overview on the recent memorandum released by the @WhiteHouse  highlighting significant incoming deadlines on AI.  Chief among these is the appointment of...</li><li><a href="https://github.com/Aesthisia/LLMinator">GitHub - Aesthisia/LLMinator: Gradio based tool to run opensource LLM models directly from Huggingface</a>: Gradio based tool to run opensource LLM models directly from Huggingface - Aesthisia/LLMinator</li><li><a href="https://github.com/aseichter2007/ClipboardConqueror">GitHub - aseichter2007/ClipboardConqueror: Clipboard Conqueror is a novel omnipresent copilot alternative designed to bring your very own LLM AI assistant to any text field.</a>: Clipboard Conqueror is a novel omnipresent copilot alternative designed to bring your very own LLM AI assistant to any text field.   - GitHub - aseichter2007/ClipboardConqueror: Clipboard Conqueror...</li><li><a href="https://saask.ing">SaaS King | Best SaaS Boilerplates</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=NUfAtIY85GU&t=8s&ab_channel=AdityaKumarSaroj">One Mix by SaaS King | Boilerplate Demo</a>: A quick introduction to OneMix by SaaS King. OneMix is made with Remix (Vite), Tailwind, Supabase, Prisma, Stripe and Resend.How can OneMix by SaaS King help...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1223275212687278230)** (40 messages🔥): 

- **ProteinBERT Presentation Recap**: The Hugging Face Reading Group recently hosted a presentation on ProteinBERT, a BERT model specialized for proteins. The recording is now available on [YouTube](https://www.youtube.com/watch?v=c_Fx-eFfFB0&ab_channel=IsamuIsozaki).

- **Discovering Anomalous Proteins**: Along with ProteinBERT, there was mention of work using protein language models and anomaly detection highlighted by the journal article [Detecting anomalous proteins using deep representations](https://academic.oup.com/nargab/article/6/1/lqae021/7614821).

- **Navigating the Reading Room**: Members were guided on how to access the reading group and were provided with links to the Discord voice channel and [events section](https://discord.gg/hqrZjkjJaq?event=1222215283343622276) for future meetings.

- **Meetings Don't Follow a Fixed Schedule**: It was clarified that the reading group meetings are not on a fixed schedule; they depend on the availability of presenters.

- **Presentation Accessibility Issues Sorted Out**: During the meeting, there were initial confusions and permissions issues regarding speaking and streaming which were resolved, ensuring smooth progress.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=c_Fx-eFfFB0&ab_channel=IsamuIsozaki">Hugging Face Reading Group 18: ProteinBERT: A universal deep-learning model of protein sequence</a>: Presenter: Dan Ofer(Second Author). Author youtube channel: https://www.youtube.com/channel/UCUliO1naqgzLtMlnyZxVYFA</li><li><a href="https://www.biorxiv.org/content/10.1101/2021.05.24.445464v1">ProteinBERT: A universal deep-learning model of protein sequence and function</a>: Self-supervised deep language modeling has shown unprecedented success across natural language tasks, and has recently been repurposed to biological sequences. However, existing models and pretraining...</li><li><a href="https://docs.google.com/presentation/d/1JqF0pZHHWieu4-w1d31HYux3rBkr21qh7mRxGnJGupo/edit#slide=id.ge53ec7dbea_0_0">ProteinBERT - DL PLM - HuggingFace</a>: ProteinBERT: A universal deep-learning protein language model A protein language model - with twists! ProteinBERT: A universal deep-learning model of protein sequence and function Nadav Brandes, Dan O...</li><li><a href="https://academic.oup.com/nargab/article/6/1/lqae021/7614821">Detecting anomalous proteins using deep representations</a>: Abstract. Many advances in biomedicine can be attributed to identifying unusual proteins and genes. Many of these proteins’ unique properties were discover
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1223145966694174852)** (19 messages🔥): 

- **Finetuning Struggles with SAM Model**: A user trying to finetune a SAM model ran into a `MisconfigurationException` with the message about a `CUDAAccelerator` not available, being advised to change the `accelerator` parameter in their training code to `'mps' or 'cpu'` from a specific line. The user was also counseled to adjust the `devices` parameter for their setup.
  
- **Multiple Font Sizes Cause Confusion**: A conversation about text representation led to clarification that a previous message by a user about changing the size was actually about **font size** adjustments.

- **Curation Techniques for Finetuning CLIP-like Models**: One user inquired about the optimal strategy for curating a dataset when finetuning CLIP-like models with images that have extensive metadata. Another member recommended examining works like **OpenCLIP** and **FashionCLIP** for insights into such finetuning processes.

- **Diving into Video Classification**: A beginner named Partha expressed difficulty in getting started with **VideoMamba** for video classification, questioning whether existing models like VGG-16 or ResNet-34 could be implemented in a similar manner. The user was exploring the provided [VideoMamba repository](https://github.com/OpenGVLab/VideoMamba) for guidance.

- **Normalization Queries in ConvUnet Implementations**: A discussion about the normalization methods in ConvUnet implementations arose with references to a specific paper. A suggestion pointed towards using **nnUNet**, a robust framework with established heuristics, known for being widely adopted for prostate segmentation tasks in medical research.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.computer.org/csdl/proceedings-article/bibm/2023/10385928/1TOcjY9BJrq">CSDL | IEEE Computer Society</a>: no description found</li><li><a href="https://github.com/bhpfelix/segment-anything-finetuner?t">GitHub - bhpfelix/segment-anything-finetuner: Simple Finetuning Starter Code for Segment Anything</a>: Simple Finetuning Starter Code for Segment Anything - bhpfelix/segment-anything-finetuner</li><li><a href="https://github.com/bhpfelix/segment-anything-finetuner?tab=readme-ov-file">GitHub - bhpfelix/segment-anything-finetuner: Simple Finetuning Starter Code for Segment Anything</a>: Simple Finetuning Starter Code for Segment Anything - bhpfelix/segment-anything-finetuner
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1223279797938032712)** (9 messages🔥): 

- **Finding Assistant Models by Tokenizer**: A member inquired about how to identify compatible assistant models for the `assistant_model` parameter in `model.generate` by tokenizer. They pondered if the **Hugging Face Hub API** and model metadata could facilitate the discovery of such models.

- **Extracting Domain-Specific Entities**: A contributor considering entity extraction from 20k documents seeks the best approach among three: using high-frequency words as entities, finding a suitable model on the Hugging Face Hub, or training a custom model. They expressed a preference for the lowest complexity solution that could handle *domain-specific entities*.

- **Evaluating RAG with ollama**: In response to the entity extraction dilemma, another member suggested using **ollama**, referencing a YouTube video explaining how to evaluate Retrieval Augmented Generation (RAG) systems which can be simple to install and operate on moderate GPU capacities.

- **Saving Trainable Params in FSDP Training**: A user sought tips on how to save only trainable parameters during FSDP training, directing to their detailed query in the "Ask for help" channel with the aim of optimizing the process.

- **Project on Data Extraction to JSON for Fine-Tuning**: An individual discussed their project involving data extraction to JSON using a language model and contemplated if the cleansed results could be used to fine-tune a model specifically for this task. They queried about the suitability of large language models or more specialized options for such fine-tuning.

**Link mentioned**: <a href="https://youtu.be/r0_O0IogbKo?si=lNon-ytkDjw9x1-3">Evaluate Retrieval Augmented Generation (RAG) Systems</a>: Retrieval Augmented Generation is a powerful framework which improves the quality of responses that you get from LLMs. But if you want to create RAG systems ...

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1223167134192898050)** (8 messages🔥): 

- **Realtime Diffusion Breakthrough Achieved**: An enthusiast shared the capability of img2img to perform *1 step diffusion*, allowing 30fps at 800x800 resolution with **sdxl-turbo**. They highlighted this method generates a mesmerizing, continuous *real-time evolution* of images, making it difficult to look away due to captivating transitions.

- **Unmasking an "off by 1" Error**: The img2img tool is experiencing an issue where images drift to the right during high-speed generation. A user suggests a workaround by trimming one pixel from the left edge and adding a noise strip to the right edge every three frames, and is considering investigating the padding in **conv2d** as a potential root cause.

- **From Decoder-Only to Encoder-Only: The CodeLlama Model Transition**: A project update was provided outlining the replacement of **CodeLlama's** Decoder-Only architecture with an Encoder-Only model utilizing Dilated-Attention. Steps taken include code initialization with **AutoModelForCausalLM**, creation of a custom configuration class, and the development of a new attention mechanism modeled after **LongNet**.

- **Tech Tweaking Inquiry**: A member is seeking advice on how to use **diffusers** to trigger **lora** and how to verify if a model is in use.

- **Assistance Request for Language Model Fine-Tuning**: A user has requested help with fine-tuning an open-source language model on PDF files, noting that they are facing challenges in the process.

- **Showcasing Realtime Diffusion**: A link to a Twitter post ([video snippets](https://twitter.com/Dan50412374/status/1774527643058331980)) was shared showing examples of continuous realtime video generation using diffusion techniques.
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1223135307172679760)** (136 messages🔥🔥): 

- **Debugging with VS Code and PyCharm**: A member asked for guidance on how to develop and debug using VS Code and PyCharm, noting issues with the debugger in PyCharm ceasing to function.
- **Using Raspberry Pi as O1 Client**: It was clarified that one can use a Raspberry Pi as a client for OpenInterpreter if it has a microphone and a keyboard; it was used in prototyping with a button on a breadboard for push-to-talk functionality.
- **Potential of Voice Activation for 01 Device**: A member inquired about the possibility of converting the push-to-talk feature of the 01 device to voice activation, with community feedback indicating current support for push-to-talk only.
- **Discussion on Custom Vision Models and AI Operation**: Community members exchanged insights on a custom vision model for OpenInterpreter, with references made to external resources and a GitHub repository that could potentially be incorporated into local implementation.
- **Exploration of OpenInterpreter Training Tools**: A link was shared for a complete example to train/finetune a LLM with full source code, offering insight into the tools used for AI model development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ohanapal.app/">OhanaPal | Super App for the Super Abled</a>: Welcome to OhanaPal—where empowerment and inclusion meet, making every day extraordinary for the super abled.</li><li><a href="https://tenor.com/view/bane-no-banned-and-you-are-explode-gif-16047504">Bane No GIF - Bane No Banned - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=OauSkezvqAk">Install Mistral Quiet Star Demo Locally - Good Reasoning AI Model</a>: This video shows how to locally install Mistral Quiet Star AI Model locally on windows. Model&#39;s creator thinks that this model is proof of his theory that yo...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/cc6291f8372c9c61cb53f7c1d4e6ef819b8457eb/interpreter/core/computer/display/display.py#L68">open-interpreter/interpreter/core/computer/display/display.py at cc6291f8372c9c61cb53f7c1d4e6ef819b8457eb · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1156">Implement non-blocking CLI history auto saver by teocns · Pull Request #1156 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made: Implement history_autosaver utility function to start a background thread for autosaving the CLI history using readline. If readline is unavailable, it returns a...</li><li><a href="https://youtube.com/shorts/dpkzijtXOqw?si=HWDNBmq4rd-uOTtF">HoloMat Update: Jarvis controls my printers! #engineering #3dprinting #ironman</a>: no description found</li><li><a href="https://github.com/dcrebbin/meta-vision-api">GitHub - dcrebbin/meta-vision-api: Hacky Meta Glasses API with GPT4 Vision Integration</a>: Hacky Meta Glasses API with GPT4 Vision Integration - dcrebbin/meta-vision-api</li><li><a href="https://github.com/FiveTechSoft/tinyMedical">GitHub - FiveTechSoft/tinyMedical: TinyLLama trained with medical dataset and saved as GGUF file</a>: TinyLLama trained with medical dataset and saved as GGUF file - FiveTechSoft/tinyMedical
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1223191722264363019)** (146 messages🔥🔥): 

- **Environment Variables Swapped for Flags**: A switch from using an `.env` file to **passing flags in Python** was discussed, with the rationale that the initial **shell script approach** faced issues. There's ongoing work to adopt something akin to a **yaml configuration**, similar to the core Open Interpreter repo.
- **Windows Workflow Improvements**: Contributors have been actively working on **Windows compatibility** for the 01 client. [A pull request](https://github.com/OpenInterpreter/01/pull/192) is addressing setup and running issues on Windows, while another [PR](https://github.com/OpenInterpreter/01/pull/203) has been made to update documentation to help future users.
- **Audio Troubleshooting on MacOS**: A user encountered issues where **no audio played** after a response was generated, pointing to a potential problem with `ffmpeg`. Other users joined in the troubleshooting process, suggesting multiple [commands to potentially fix the issue](https://github.com/OpenInterpreter/01/issues/197).
- **Persistent Settings for 01 Client on M5 Atom**: A GitHub [pull request](https://github.com/OpenInterpreter/01/pull/214/) was mentioned, enabling the **M5Atom to automatically reconnect to WiFi and the server URL** without needing to re-enter information after each restart.
- **3D Printing Size Adjustment for O1 Light**: For anyone printing their own O1 Light, a user recommended **scaling up the 3D files to 119.67%** for appropriate sizing, likely to accommodate internal components correctly.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/jsngr/status/1774110742070882478?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Tweet from jordan singer (@jsngr)</a>: ✨ talk to your computer remotely from your phone  i call it Teleport</li><li><a href="https://app.suno.ai/song/6fb4e5a3-fa8e-4a7b-8362-3c5b355d9936">01100001 01101101 00100000 01001001 0010 | Suno</a>: 8bit,chiptune,speedup,Arpeggio,fatbass,Hardbass,FemaleVocals,synthesizer，Electronic，speedup song. Listen and make your own with Suno.</li><li><a href="https://app.suno.ai/song/cbf3c6a9-dc4e-4663-b7c5-b7f1aebe687a/">Electric Echoes | Suno</a>: alternative vaper wave ska song. Listen and make your own with Suno.</li><li><a href="https://github.com/OpenInterpreter/01/pull/214/">Update client.ino by aramsdale · Pull Request #214 · OpenInterpreter/01</a>: Automatically reconnects to last successful WiFi and Server URL, if available Utilizing Preferences, detect successful WiFi connection, store to ssid preferences, and recall on reboot. Same for ser...</li><li><a href="https://github.com/OpenInterpreter/01/pull/192">[WIP] Fix setup and running on Windows by dheavy · Pull Request #192 · OpenInterpreter/01</a>: Attempts to bridge the gap and facilitate onboarding for windows users by adding missing parts and fixing Win-specific issues. Solves this secondary issue in #167 and another one, where the Device ...</li><li><a href="https://github.com/OpenInterpreter/01/pull/192/files#diff-1dd2e9bca23ee8cae42c577e69ce37b6b5dbe816dbe87517c780043bebaf59c2R264)">[WIP] Fix setup and running on Windows by dheavy · Pull Request #192 · OpenInterpreter/01</a>: Attempts to bridge the gap and facilitate onboarding for windows users by adding missing parts and fixing Win-specific issues. Solves this secondary issue in #167 and another one, where the Device ...</li><li><a href="https://github.com/OpenInterpreter/01/pull/194#pullrequestreview-1969442081)">Fixed windows client module not found error by Abdullah-Gohar · Pull Request #194 · OpenInterpreter/01</a>: Added a windows client module similar to the mac module, fixes #167</li><li><a href="https://github.com/OpenInterpreter/01/issues/197)">Issues · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01/pull/203">Update documentation for Windows installation by dheavy · Pull Request #203 · OpenInterpreter/01</a>: Problem Installation for Windows, with its key differences, isn&#39;t provided in the documentation. Solution Compile learnings from previous users&#39; attempt (including Zorcon&#39;s on Discord and ...</li><li><a href="https://01.openinterpreter.com/services/language-model">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1224257328644100116)** (2 messages): 

- **Open Interpreter Goes Experimental**: A member shared a [YouTube video](https://www.youtube.com/watch?v=v9uXdRwAQ0c) titled "Open Interpreter Advanced Experimentation - Part 2", showcasing the latest experiments with the OpenInterpreter.
- **Fabric: AI Augmentation Framework on GitHub**: A link to [Fabric on GitHub](https://github.com/danielmiessler/fabric) was shared, described as an open-source framework for enhancing human capabilities with AI, featuring a modular system and a crowdsourced set of AI prompts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=v9uXdRwAQ0c">Open Interpreter Advanced Experimentation - Part 2</a>: ➤ Twitter - https://twitter.com/techfrenaj➤ Twitch  - https://www.twitch.tv/techfren➤ Discord  - https://discord.com/invite/z5VVSGssCw➤ TikTok - https://www....</li><li><a href="https://github.com/danielmiessler/fabric">GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.</a>: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere. - ...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1223161726657364060)** (251 messages🔥🔥): 

- **Intel Arc Optimizations Discussed**: An attempt to implement optimized transformers/dot product attention on Intel Arc led to noting that the library provided (ipex) was inefficient and did not optimize properly for fp16, remaining in fp32 and slowing down the process. Tweaks involving PyTorch JIT and more direct implementations resulted in significant stable diffusion performance improvements.

- **$200 Bounty for AMD GPU GEMM Code**: Instructions for writing optimized GEMM code targeting AMD 7900XTX GPUs using HIP C++ were shared, with details indicating the task involves both C++ and Python integration. There were issues with the associated script due to missing modules and incorrect library paths. 

- **Tinygrad Operational Concerns**: [Tinygrad's functionality](https://github.com/tinygrad/tinygrad/pull/3891) is still in discussion, including potential issues with test cases failing in pull requests and problems related to missing features or library dependencies. One user's code formatting issue was corrected with advice on the proper use of markdown for code blocks.

- **Troubleshooting AMD Drivers**: A significant portion of the conversation focused on the struggles with AMD driver stability, the need for AMD to open source firmware and hardware documentation, discussions of potential reset methods for AMD GPUs, and specific bugs like the SMU crashes. There was skepticism regarding AMD's ability to fix issues despite ongoing efforts.

- **Tinygrad Updates and Fixes**: Updates to the Tinygrad project, including website adjustments and discussions on its functionalities, were mentioned. There was a call for more information and source access for lower-level stack functionalities to further development, with detailed discussions around CI implementation and current limitations. 

- **Vendor Reset Exploration**: Participants considered different methods to trigger resets on AMD GPUs, including BACO and PSP mode2, with mixed results. Frustration was expressed about the unavailability of full GPU resets and the inefficiency of email communication with AMD for resolving these structural issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://looking-glass.io)">no title found</a>: no description found</li><li><a href="https://github.com/geohot/7900xtx/tree/master/crash">7900xtx/crash at master · geohot/7900xtx</a>: Contribute to geohot/7900xtx development by creating an account on GitHub.</li><li><a href="https://github.com/gnif/vendor-reset/blob/master/src/amd/navi10.c">vendor-reset/src/amd/navi10.c at master · gnif/vendor-reset</a>: Linux kernel vendor specific hardware reset module for sequences that are too complex/complicated to land in pci_quirks.c - gnif/vendor-reset</li><li><a href="https://mastodon.gamedev.place/@NOTimothyLottes/112190982123087000">NOTimothyLottes (@NOTimothyLottes@mastodon.gamedev.place)</a>: Anyway it looks like I&#39;m back to a compiler perf bug that is impossible to workaround and too horrible to ignore.  Wave-coherent (should be fast) dynamic descriptor choice acts like a batch break ...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_kfd.py">tinygrad/tinygrad/runtime/ops_kfd.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://news.ycombinator.com/item?id=39888020">Documentation for the AMD 7900XTX | Hacker News</a>: no description found</li><li><a href="https://github.com/geohot/tinyxxx">GitHub - geohot/tinyxxx: tiny corporation website</a>: tiny corporation website. Contribute to geohot/tinyxxx development by creating an account on GitHub.</li><li><a href="https://github.com/geohot/7900xtx?tab=readme-ov-file#firmware-loads">GitHub - geohot/7900xtx</a>: Contribute to geohot/7900xtx development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3891">Add negative_log_likelihood loss and cross_entropy loss  by airpods69 · Pull Request #3891 · tinygrad/tinygrad</a>: This PR adds negative_log_likelihood and cross_entropy to Tensor Eg: For negative_log_likelihood, import tinygrad data = tinygrad.Tensor([[1, 2, 4]]) target = [2] print(data.negative_log_likelihood...</li><li><a href="https://github.com/ROCm/ROCm/issues/2196">[Driver] *ERROR* MES failed to response msg=2 · Issue #2196 · ROCm/ROCm</a>: Triggered by running https://github.com/RadeonOpenCompute/rocm_bandwidth_test in a loop while running https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/gpu-burn in a loop. 1x 7900XTX...
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1223154102599684106)** (16 messages🔥): 

- **Decoding Dual Views in Shape Manipulation**: Members discussed why `shapetracker.from_shape((2,4)).permute((1,0)).reshape((2,4))` creates two views, attributing it to *memory layout complexities* and *uneven stride presentation* as shown in View representations.
- **Understanding Kernel Fusion through Notes**: A member shared a [link to their notes](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md) that could potentially help solve a bounty for optimizing kernel fusion in **tinygrad**.
- **Tackling Unnecessary Complexity in Expression Indexes**: In response to a concern about expression indexes for uneven strides, a member made a pull request to address unnecessary complexities, seen in this [tinygrad pull request](https://github.com/tinygrad/tinygrad/pull/3988).
- **Seeking GPU Training Tools in tinygrad**: For users looking to start *GPU-based training jobs*, pointers were given to `examples/beautiful_mnist.py` and another member's notes, which can be found [here](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/dotproduct.md) for understanding backends in tinygrad.
- **Insights On Contributing to tinygrad without a GPU**: It was mentioned that optimizations in **tinygrad** can be done without a dedicated GPU, emphasizing areas such as shapetracker and uopt optimization, implying you can contribute effectively with just a laptop setup.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/dotproduct.md">tinygrad-notes/dotproduct.md at main · mesozoic-egg/tinygrad-notes</a>: Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops.md">tinygrad-notes/uops.md at main · mesozoic-egg/tinygrad-notes</a>: Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3988.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1223374562498969640)** (2 messages): 

- **Phorm.ai Integration with LlamaIndex**: LlamaIndex Discord now offers [Phorm.ai by Morph Labs](https://phorm.ai) accessible through specific channels for both TypeScript and Python queries. Users can invoke Phorm with an @-mention, receiving answers and sources directly in a thread.

- **LlamaIndex Hosts RAFT Webinar**: A **webinar** featuring Tianjun Zhang and Shishir Patil, co-authors of **Retrieval-Augmented Fine-Tuning (RAFT)**, is scheduled. It aims to educate on fine-tuning pre-trained LLMs, happening this Thursday at 9am PT with sign-up available at [lu.ma](https://lu.ma/v1bdat63).

- **RAFT Technique Spotlight**: **RAFT** is highlighted as a technique that allows fine-tuning pre-trained large language models (LLMs) for domain-specific tasks, enhancing performance by combining open-book exam dynamics with domain knowledge. For more details, visit the [LlamaIndex status post](https://x.com/llama_index/status/1774814982322172077?s=20) and check out the [RAFT paper](https://arxiv.org/pdf/2403.10131.pdf) and [blog](https://gorilla.cs.berkeley.edu/blogs/).

- **Dataset Generation for RAFT Now Available**: Thanks to a contribution, it's now possible to generate datasets for RAFT using the [RAFTDatasetPack](https://llamahub.ai/l/llama-packs/llama-index-packs-raft-dataset?from=) with an accompanying notebook available on [GitHub](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/v1bdat63">LlamaIndex Webinar: Retrieval-Augmented Fine-Tuning (RAFT) · Zoom · Luma</a>: RAFT - Retrieval Augmented Fine Tuning 🔥 Retrieval-Augmented Fine-Tuning (RAFT) by Zhang et al. is a new technique to fine-tune pre-trained LLMs for specific domain RAG...</li><li><a href="https://x.com/llama_index/status/1774814982322172077?s=20">Tweet from LlamaIndex 🦙 (@llama_index)</a>: New LlamaIndex Webinar 🚨 - come learn how to do retrieval-augmented fine-tuning (RAFT)!  Doing RAG is like taking an open-book exam without studying. It’s marginally better than a closed-book exam wh...
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1223289750472097955)** (10 messages🔥): 

- **DIY Reflective RAG**: Florian June introduces a guide on how to build a dynamic RAG system that includes self-reflection, featuring a method where retrieval is triggered only by a specific token. The outline and instructions are available via [a shared Twitter post](https://twitter.com/llama_index/status/1773730535564783840).
  
- **LlamaParse Enhances RAG Queries**: A video tutorial explains how LlamaParse can transform complex documents into simple queries using LLM-powered parsing, illustrated with an insurance policy example. More details can be found in the linked [Twitter thread](https://twitter.com/llama_index/status/1773783011785585141).
  
- **Panel Discussion on RAG's Longevity**: A panel including @ofermend and @seldo discuss the relevance of RAG systems even in scenarios with large context windows, emphasizing its cost efficiency and selectivity. The session with @vectara is accessible through the accompanying [YouTube update prompt](https://support.google.com/youtube/answer/175292).
  
- **Financial News Chatbot Tutorial**: Collaborating with @llama_index, @qdrant_engine, and @Google Gemini, a new initiative showcases a chatbot to streamline keeping up with the latest financial news. The project is introduced through a [Twitter link](https://twitter.com/llama_index/status/1774125527072395496).
  
- **Insights on RAG Design Philosophy**: @MichalOleszak's comprehensive guide details the significant design decisions behind building effective RAG systems, discussing pillars like indexing and retrieval. The article is presented in a [tweet by LlamaIndex](https://twitter.com/llama_index/status/1774240631231267285).

**Link mentioned**: <a href="https://t.co/y5XDPIYD1W">no title found</a>: no description found

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1223160666786103336)** (218 messages🔥🔥): 

- **Document Chunking and Embedding Issues**: Users are experiencing issues where the **SemanticSplitterNodeParser** produces nodes exceeding the OpenAI embeddings limit of 8192 tokens. The question revolves around strategies for handling large documents that result in nodes too large to be embedded efficiently.
- **Setting up LlamaIndex with Open Source LLMs**: There's a request for complete code examples demonstrating how to work with open-source language models in LlamaIndex, mentioning tutorials and guides with outdated links and missing files. A [Colab tutorial](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/) was shared in response.
- **Creating Custom Agents with LlamaIndex**: Queries about how to create custom agents that include routing, query engine tools, and function tools within LlamaIndex have been discussed, with emphasis on incorporating function tools and handling intermediary results during slow processing of agent actions.
- **Difficulties and Confusions with LlamaIndex Documentation**: Users reported confusion when working with LlamaIndex documentation, particularly with deprecated components like `NLSQLTableQueryEngine` and links leading to obsolete resources. There were also mentions of issues with columns being directly read from SQLAlchemy schemas, leading to challenges in excluding specific columns from queries.
- **Vector Database Interactions**: Questions arose regarding the use of existing vector databases with LlamaIndex and whether it's possible to conduct queries on VectorStores not originally created with LlamaIndex. Another concern was the possible need to duplicate data to exclude sensitive columns in queries.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/">Build a chatbot with custom data sources, powered by LlamaIndex</a>: Augment any LLM with your own data in 43 lines of code!</li><li><a href="https://ts.llamaindex.ai">What is LlamaIndex.TS? | LlamaIndex.TS</a>: LlamaIndex.TS is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. While a python package is also available (see here), LlamaIndex.TS offers core ...</li><li><a href="https://ts.llamaindex.ai/modules/data_loader">Loader | LlamaIndex.TS</a>: Before you can start indexing your documents, you need to load them into memory.</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/">Indexing - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#text-splitters">Node Parser Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_auto_retriever/">Auto-Retrieval from a Vector Database - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/?h=#keywordnodepostprocessor">Node Postprocessor Modules - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/prompts">llama_index/llama-index-core/llama_index/core/prompts at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/">Building RAG from Scratch (Open-source only!) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/#accessing-prompts">Accessing/Customizing Prompts within Higher-Level Modules - LlamaIndex</a>: no description found</li><li><a href="https://github.com/agronholm/sqlacodegen?tab=readme-ov-file">GitHub - agronholm/sqlacodegen: Automatic model code generator for SQLAlchemy</a>: Automatic model code generator for SQLAlchemy. Contribute to agronholm/sqlacodegen development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data/">Structured Data - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/#part-2-query-time-retrieval-of-tables-for-text-to-sql">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_personality/">Chat Engine with a Personality ✨ - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/8a8324008764a7fefb6f25b0e3aac81089590322/llama-index-legacy/llama_index/legacy/prompts/system.py#L4">llama_index/llama-index-legacy/llama_index/legacy/prompts/system.py at 8a8324008764a7fefb6f25b0e3aac81089590322 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/create-llama?tab=readme-ov-file#customizing-the-llm">GitHub - run-llama/create-llama: The easiest way to get started with LlamaIndex</a>: The easiest way to get started with LlamaIndex. Contribute to run-llama/create-llama development by creating an account on GitHub.</li><li><a href="https://huggingface.co/colbert-ir/colbertv2.0">colbert-ir/colbertv2.0 · Hugging Face</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/indices/llama-index-indices-managed-colbert/llama_index/indices/managed/colbert/base.py">llama_index/llama-index-integrations/indices/llama-index-indices-managed-colbert/llama_index/indices/managed/colbert/base.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/ingestion/ingestion_gdrive/">Building a Live RAG Pipeline over Google Drive Files - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi#rag-approach-to-import-external-knowledge-into-llm-as-context>)">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/production_rag#key-techniques_1>).">Building Performant RAG Applications for Production - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-packs">llama_index/llama-index-packs at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/LlamaIndexTS">GitHub - run-llama/LlamaIndexTS: LlamaIndex is a data framework for your LLM applications</a>: LlamaIndex is a data framework for your LLM applications - run-llama/LlamaIndexTS</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies#metadata-filters>).">Basic Strategies - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1223223563981684747)** (4 messages): 

- **Going Deep into Model Alignment**: A new [blog post](https://blog.premai.io/model-alignment-process/) explores the model alignment strategies for LLMs, specifically focusing on RLHF, DPO, and KTO methods and their impact on Mistral and Zephyr 7B models, enriching the post with practical comparisons.
- **A Hub for Latest LLM Research**: A mission statement available on [shure-dev's GitHub page](https://shure-dev.github.io/) emphasizes their commitment to providing a curated list of high-quality, essential papers for researchers in the field of Large Language Models.
- **Enhancing RAG with LlamaParse and Re-Ranking**: A [YouTube video](https://youtu.be/wCFXae8hiYA) discusses advancing Retrieval Augmented Generation (RAG) by integrating LlamaParse and a re-ranker to potentially improve the overall performance.
- **Benchmarking Whisper-based ASR Packages**: A detailed [blog post](https://amgadhasan.substack.com/p/sota-asr-tooling-long-form-transcription) assesses open source whisper-based packages for long-form transcription capabilities, comparing accuracy and efficiency metrics among popular frameworks such as Huggingface Transformers and FasterWhisper.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.premai.io/model-alignment-process/">Model Alignment Process</a>: The alignment of generative models with human feedback has significantly improved the performance of natural language generation tasks. For large language models (LLMs), alignment methods like reinfor...</li><li><a href="https://shure-dev.github.io/">Awesome-LLM-related-Papers-Comprehensive-Topics</a>: World's Most Comprehensive Curated List of LLM Papers & Repositories</li><li><a href="https://amgadhasan.substack.com/p/sota-asr-tooling-long-form-transcription">SOTA ASR Tooling: Long-form Transcription</a>: Benchmarking the different Whisper frameworks for long-form transcription</li><li><a href="https://youtu.be/wCFXae8hiYA">Advance RAG: LlamaParse + Reranker = Better RAG</a>: Retrieval Augmented Generation ( RAG ) is all we want to talk about but are we trying or following the code practice. Out of many ways, in this video, I will...
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1223140446587457618)** (2 messages): 

- **New Rankings Unveiled**: **App Rankings for Models** have been launched, revealing the top public apps using specific models, viewable under the new **Apps** tab for each model page with stats on tokens processed. Check out the [Claude 3 Opus App Rankings](https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps) to see current leaders.

- **Community Spotlight on Discord Bots**: A member has created a **Discord bot, Sora**, that integrates with the OpenRouter API to enhance conversations on Discord. Find this cool bot on [GitHub](https://github.com/mintsuku/sora).

- **Craft Your Own Eval**: Another community member has introduced a way to **write your own model evaluations** and share them through a super cool project. Dive into creating custom evals at [nonfinito.xyz](https://nonfinito.xyz/).

- **OpenRouter API and Client Updates**: There's a new `/api/v1/completions` API endpoint matching the chat API's functionality with *prompt* parameter-only support, and OpenAI API client support has been improved. Important to note, usage of **Groq for Nitro models** is halted due to rate limiting issues.

- **King of Expertise**: Databricks' **DBRX 132B**, a new open-source large language model boasting superior performance over models like Mixtral in reasoning and coding tasks, is now available. Examine its capabilities and pricing on the [DBRX Instruct page](https://openrouter.ai/models/databricks/dbrx-instruct) and see full details in their [launch announcement](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/databricks/dbrx-instruct">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX is a new open source large language model developed by Databricks. At 132B, it outperforms existing open source LLMs like Llama 2 70B and Mixtral-8x7B on standard industry benchmarks for language...</li><li><a href="https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps">Claude 3 Opus by anthropic | OpenRouter</a>: Claude 3 Opus is Anthropic&#x27;s most powerful model for highly complex tasks. It boasts top-level performance, intelligence, fluency, and understanding.  See the launch announcement and benchmark re...</li><li><a href="https://github.com/mintsuku/sora">GitHub - mintsuku/sora: Sora is a Discord bot that integrates with the Open Router API to facilitate conversation in Discord servers.</a>: Sora is a Discord bot that integrates with the Open Router API to facilitate conversation in Discord servers. - mintsuku/sora</li><li><a href="https://nonfinito.xyz/">Evaluations - Non finito</a>: no description found
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1223398204566016210)** (1 messages): 

- **Novus Chat Unveils OpenRouter Models**: A solo developer shared an update introducing [Novus Chat](https://talk.novus.chat/agents), a new platform currently featuring **OpenRouter models**. They mentioned that graph-based agents are in the works and highlighted that access to the lowcost models is free.

- **Invitation to Join the Development Adventure**: The developer also extended an invitation to the community to join a dedicated [Discord server](https://discord.gg/w9WR6QMAdt) for discussions and updates about this personal free time project.

- **Anticipation for Agent Creations**: It was announced that the release of the **agent creations** is imminent, signaling an exciting development in the personal project.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://talk.novus.chat/agents">React App</a>: no description found</li><li><a href="https://discord.gg/w9WR6QMAdt">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1223140917398081596)** (229 messages🔥🔥): 

- **System Troubleshooting in Real-Time**: Users reported issues related to the downtime of **Midnight Rose** and **Pysfighter2** models, which were resolved with a quick restart. Another user experienced a problem with **Coinbase** payment connectivity, which is reportedly being fixed, though wallet connections are still operational.

- **ClaudeAI's Self-Moderated Beta Introduced**: **OpenRouter** offers a beta version of **Anthropic's Claude 3 Opus** that is self-moderated, which purportedly reduces false positives. This model aims to perform well in roleplay scenarios or when dealing with sensitive topics, as detailed in [Anthropic's launch announcements and benchmarks](https://www.anthropic.com/news/claude-3-family).

- **Improved User Experience with Chat Completion**: OpenRouter's handling of the `name` in `ChatCompletionMessageParam` saw a hotfix that better integrates user names in conversations, specifically with **Claude 3 models**, aligning with Anthropic's API requirements for alternating user-assistant messages.

- **Exploring Payment Options and Balance Validity**: Users sought alternate crypto payment methods due to Coinbase issues, with suggestions pointing to **MetaMask**. Concerns were raised about the expiration of balance credits, with a clarification that the 12-month validity is a safeguard that has not been actively enforced.

- **Developers Using OpenRouter Seek Guidance**: Users exchanged information on using OpenRouter's chatbot APIs, particularly focusing on strategies for context retention, error handling, and differences in API functionalities between the **Assistant Message** and traditional **Chat Completion** methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/AetherResearch/Cerebrum-1.0-8x7b">AetherResearch/Cerebrum-1.0-8x7b · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models/anthropic/claude-3-opus:beta">Claude 3 Opus by anthropic | OpenRouter</a>: This is a lower-latency version of [Claude 3 Opus](/models/anthropic/claude-3-opus), made available in collaboration with Anthropic, that is self-moderated: response moderation happens on the model&#x...</li><li><a href="https://gist.github.com/implicit-invocation/94602ea7a5a2a3c2a97f9144cecc0348">llm.test.ts</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/OpenRouterTeam/openrouter-runner">GitHub - OpenRouterTeam/openrouter-runner: Inference engine powering open source models on OpenRouter</a>: Inference engine powering open source models on OpenRouter - OpenRouterTeam/openrouter-runner</li><li><a href="https://github.com/OlympiaAI/open_router">GitHub - OlympiaAI/open_router: Ruby library for OpenRouter API</a>: Ruby library for OpenRouter API. Contribute to OlympiaAI/open_router development by creating an account on GitHub.</li><li><a href="https://share.codebyars.dev/u/d35YbK.png">codebyars.dev</a>: no description found
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1223229433737908284)** (118 messages🔥🔥): 

- **Claude Sneaks into Canada via VPN**: Members discussed that signing up for Anthropic's Claude requires a VPN and payment via Google Pay, effective for users in Europe.
- **Technical Wisdom in General Chat**: Users clarified that while high-level technical discussions happen in the general chat, they steer clear of basic coding help.
- **Voicecraft Surpasses ElevenLabs**: A new open-source speech model by Voicecraft reportedly surpasses ElevenLabs, with the weights available on [Github](https://github.com/jasonppy/VoiceCraft), and the community sharing their successful experiences.
- **Emad's Exit Examined**: Stability AI's CEO Emad Mostaque's departure was a hot topic, with links to interviews and media articles such as [Diamandis’s YouTube interview](https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa) and [an article archive](https://archive.is/8QkSl), along with speculative tweets about potential acquisitions.
- **Exploring "1-bit LLMs"**:
    An active discussion centered on models quantized to three values, often termed "1-bit LLMs." A member pointed out that the terminology may be more marketing than technical accuracy, given that three-valued systems are more accurately described as ternary or "1.58 bits per parameter." Links to papers and mention of reproductions of the original paper's findings underline the community's engagement with this cutting-edge progress.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=39538626">no title found</a>: no description found</li><li><a href="https://www.ianww.com/llm-tools">LLM eval tools spreadsheet</a>: Spreadsheet of 50+ LLM evaluation tools for testing models and improving prompts.</li><li><a href="https://qdrant.tech/blog/fastllm-announcement/">Introducing FastLLM: Qdrant’s Revolutionary LLM - Qdrant</a>: Lightweight and open-source. Custom made for RAG and completely integrated with Qdrant.</li><li><a href="https://x.com/svarasura/status/1773474071801450886?s=20">Tweet from svaarsura (@svarasura)</a>: @KevinKaichuang @roydanroy & reportedly Zuckerberg is now emailing DeepMind&#39;s researchers! Super duper rich man & his whims</li><li><a href="https://x.com/sucralose__/status/1774782583731020200?s=46&t=JE84TqLviekDnEt8MAT-Eg">Tweet from Q Prophet (@sucralose__)</a>: I did more investigation of ChatGPT&#39;s backend and found solid evidence of a model named &#34;GPT Alpha&#34; that I believe is the successor to GPT-4. It&#39;s possible to enable it early, but it r...</li><li><a href="https://x.com/burny_tech/status/1774206842404516164?s=20">Tweet from Burny — Effective Omni (@burny_tech)</a>: Noam Brown works at reasoning in OpenAI and tweeted this tweet this morning and quickly deleted it  Older tweets for reference to connect the dots</li><li><a href="https://hamel.dev/blog/posts/evals/">- Your AI Product Needs Evals</a>: How to construct domain-specific LLM evaluation systems.</li><li><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">1bitLLM/bitnet_b1_58-3B · Hugging Face</a>: no description found</li><li><a href="https://x.com/anissagardizy8/status/1773759144425930962?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Anissa Gardizy (@anissagardizy8)</a>: breaking: Microsoft and OpenAI are drawing up plans for a $100 billion AI supercomputer  The supercomputer, codenamed &#34;Stargate,&#34; would contain *millions* of GPUs and require several gigawatts...</li><li><a href="https://x.com/__rej__/status/1770574730363392077">Tweet from ReJ 𓀨 Renaldas Zioma (@__ReJ__)</a>: This looks promising! Given DDR3 memory bandwidth of the FPGA my 1.58-bit systolic array might hit 1 TOPs on $99. Fingers crossed!  Thanks a ton to @samsoniuk for quick synthesis!  Repo: https://githu...</li><li><a href="https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices">Navigating the Challenges and Opportunities of Synthetic Voices</a>: We’re sharing lessons from a small scale preview of Voice Engine, a model for creating custom voices.</li><li><a href="https://x.com/rohaidalimd/status/1773804003232461113?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Rohaid Ali, MD (@RohaidAliMD)</a>: With @OpenAI&#39;s Voice Engine, our team was able to help a young patient recover her voice.  ↘️ Quoting OpenAI (@OpenAI)   We&#39;re sharing our learnings from a small-scale preview of Voice Engine,...</li><li><a href="https://x.com/wsj/status/1774189096178446752?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from The Wall Street Journal (@WSJ)</a>: Cognition Labs, a startup developing an artificial-intelligence tool for writing code, is in talks with investors to raise funding at a valuation of up to $2 billion https://on.wsj.com/3xqH5rp https:/...</li><li><a href="https://x.com/aaron_defazio/status/1773381393831067787?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Aaron Defazio (@aaron_defazio)</a>: Cooking up something special! Can&#39;t wait to get a paper out so everyone can try it out. An optimizer with no extra overhead, no additional parameters. Stay tuned!</li><li><a href="https://maven.com/parlance-labs/fine-tuning">LLM Fine-Tuning for Data Scientists and Software Engineers by Dan Becker and Hamel Husain on Maven</a>: Train, validate and deploy your first fine-tuned LLM</li><li><a href="https://www.vice.com/en/article/dy7axa/how-i-broke-into-a-bank-account-with-an-ai-generated-voice">How I Broke Into a Bank Account With an AI-Generated Voice</a>: Banks in the U.S. and Europe tout voice ID as a secure way to log into your account. I proved it&#x27;s possible to trick such systems with free or cheap AI-generated voices.</li><li><a href="https://shiny.posit.co/py/">Shiny for Python</a>: Build interactive web applications easily with the power of Python’s data and scientific stack.</li><li><a href="https://www.youtube.com/watch?v=c0GDu1CB0o4">Office Hours with Hamel and Shreya</a>: no description found</li><li><a href="https://x.com/clementdelangue/status/1771395468959813922?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from clem 🤗 (@ClementDelangue)</a>: Should we acquire Stability and open-source SD3?</li><li><a href="https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa">Why I&#39;m Leaving My Company Immediately (Stability AI) w/ Emad Mostaque | EP #93</a>: In this episode, Peter and Emad discuss Emad&#39;s stepping down as CEO of StabilityAI, his next steps into decentralized AI, and why there is so much urgency to...</li><li><a href="https://www.youtube.com/watch?v=sal78ACtGTc">What&#39;s next for AI agentic workflows ft. Andrew Ng of AI Fund</a>: Andrew Ng, founder of DeepLearning.AI and AI Fund, speaks at Sequoia Capital&#39;s AI Ascent about what&#39;s next for AI agentic workflows and their potential to si...</li><li><a href="https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul">GitHub - rejunity/tiny-asic-1_58bit-matrix-mul: Tiny ASIC implementation for &quot;The Era of 1-bit LLMs All Large Language Models are in 1.58 Bits&quot; matrix multiplication unit</a>: Tiny ASIC implementation for &quot;The Era of 1-bit LLMs All Large Language Models are in 1.58 Bits&quot; matrix multiplication unit - rejunity/tiny-asic-1_58bit-matrix-mul</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/cqJMkfO2xm">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/pull/18880">community: add hugging face text-to-speech inference API by h0rv · Pull Request #18880 · langchain-ai/langchain</a>: Description: I implemented a tool to use Hugging Face text-to-speech inference API. Issue: n/a Dependencies: n/a Twitter handle: No Twitter, but do have LinkedIn lol.</li><li><a href="https://github.com/collabora/WhisperSpeech">GitHub - collabora/WhisperSpeech: An Open Source text-to-speech system built by inverting Whisper.</a>: An Open Source text-to-speech system built by inverting Whisper. - collabora/WhisperSpeech</li><li><a href="https://github.com/myshell-ai/MeloTTS">GitHub - myshell-ai/MeloTTS: High-quality multi-lingual text-to-speech library by MyShell.ai. Support English, Spanish, French, Chinese, Japanese and Korean.</a>: High-quality multi-lingual text-to-speech library by MyShell.ai. Support English, Spanish, French, Chinese, Japanese and Korean. - myshell-ai/MeloTTS</li><li><a href="https://arxiv.org/abs/2303.17651">Self-Refine: Iterative Refinement with Self-Feedback</a>: Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improv...</li><li><a href="https://arxiv.org/abs/2303.11366">Reflexion: Language Agents with Verbal Reinforcement Learning</a>: Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language...</li><li><a href="https://arxiv.org/abs/2305.15334">Gorilla: Large Language Model Connected with Massive APIs</a>: Large Language Models (LLMs) have seen an impressive wave of advances recently, with models now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. However, their po...</li><li><a href="https://arxiv.org/abs/2303.11381">MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action</a>: We propose MM-REACT, a system paradigm that integrates ChatGPT with a pool of vision experts to achieve multimodal reasoning and action. In this paper, we define and explore a comprehensive list of ad...</li><li><a href="https://arxiv.org/abs/2201.11903">Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</a>: We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning. In particular, we ...</li><li><a href="https://arxiv.org/abs/2303.17580">HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face</a>: Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are numerous AI models available for various domains and modalities...</li><li><a href="https://arxiv.org/abs/2307.07924">Communicative Agents for Software Development</a>: Software engineering is a domain characterized by intricate decision-making processes, often relying on nuanced intuition and consultation. Recent advancements in deep learning have started to revolut...</li><li><a href="https://arxiv.org/abs/2308.08155">AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation</a>: AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, convers...</li><li><a href="https://archive.is/8QkSl">Inside Stability AI's bad breakup with Coatue and Lightspeed Venture &#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1223360912728133723)** (106 messages🔥🔥): 

- **Discussions on AI Podcast Appearance**: A member shared a [link](https://changelog.com/practicalai/262) about an AI podcast episode discussing the complicated relationship between AI and software developers, including various featured guests.
- **Streaming Issues on Discord**: Members reported problems with viewing Remi's stream, with many trying different devices and browsers but still facing issues. Suggestions included reloading Discord and trying on a phone.
- **Exploration of Local LLM Function Calling**: The topic of local LLM function calling was discussed with an interest in how it compares to other methods like *instructor* and *outlines*, highlighting insights about regular expressions and finite-state machines for text generation.
- **Outlines Over Instructor for LLM**: A member suggested that *outlines* might be more effective than *instructor* for LLM tasks, and shared an installation guide for Outlines [here](https://outlines-dev.github.io/outlines/installation/).
- **Interest in Upcoming AI Discussions**: The group expressed excitement for future discussions, with a [shared spreadsheet](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0) listing topics and facilitators for upcoming events related to UI/UX patterns, RAG architectures, and more.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://outlines-dev.github.io/outlines/installation/">installation - Outlines 〰️</a>: Structured text generation with LLMs</li><li><a href="https://arxiv.org/abs/2307.09702">Efficient Guided Generation for Large Language Models</a>: In this article we show how the problem of neural text generation can be constructively reformulated in terms of transitions between the states of a finite-state machine. This framework leads to an ef...</li><li><a href="https://changelog.com/practicalai/262">AI vs software devs with conversations from JS Party, Go Time &amp;amp; The Changelog (Practical AI #262)</a>: Daniel and Chris are out this week, so we’re bringing you conversations all about AI’s complicated relationship to software developers from other Changelog pods: JS Party, Go Time &amp;amp;amp; The Ch...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1223143487529291888)** (21 messages🔥): 

- **IDEs in the Spotlight for CUDA Development**: The channel discussed IDE preferences for CUDA development; **VSCode** remains a favorite for some, while others are exploring **CLion** as a potential alternative.
- **Beginner's CUDA Programming Course Announced**: Cohere has introduced a CUDA course for beginners, with a community-led group **BIRDS** starting a mini-cohort learning session on April 5th. Details are available on [Cohere's Tweet](https://x.com/CohereForAI/status/1773419415406809432).
- **MOJO's Standard Library Goes Open Source**: Modular announced that the core modules from the **Mojo standard library are now open-source** under the Apache 2 license, inviting the global developer community to contribute. More information and the source code can be found on [Modular's blog](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) and on their [GitHub repository](https://github.com/modularml/mojo/tree/nightly).
- **Building 'Stargate' AI Supercomputer**: There is buzz about Microsoft and OpenAI's discussions to build an AI supercomputer named **"Stargate"** with a projected cost of around $100 billion. Concerns about the environmental impact of massive data centers are also raised, citing recent resistance in **Utah** and **Arizona**, as discussed in [Jessica Lessin's Tweet](https://x.com/jessicalessin/status/1773760164153426071).
- **Discord Voice Channel Limitations During Event**: During a high-attendance event on the Discord server, users experienced a difficulty where the voice channel had a limit, which was initially maxed out at **25 people** but later increased to **99**. A setting was adjusted to mute participants by default to manage the noise during the event.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jessicalessin/status/1773760164153426071">Tweet from Jessica Lessin (@Jessicalessin)</a>: New: Microsoft and OpenAI plotting new &#34;Stargate&#34; supercomputer that could cost $100 billion.   An incredible look at what it will take to build the next generation of AI and at this consequen...</li><li><a href="https://x.com/CohereForAI/status/1773419415406809432">Tweet from Cohere For AI (@CohereForAI)</a>: Our community-led Beginners in Research-Driven Studies (BIRDS) group is kicking off it’s first mini-cohort learning group focused on CUDA Programming for Beginners, beginning on Friday, April 5th 🎉</li><li><a href="https://x.com/__tinygrad__/status/1773853465300898164">Tweet from the tiny corp (@__tinygrad__)</a>: Website updated with more tinybox specs. First boxes ship end of April.</li><li><a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: The Next Big Step in Mojo🔥 Open Source</li><li><a href="https://github.com/modularml/mojo/tree/nightly">GitHub - modularml/mojo at nightly</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1223543550613000203)** (41 messages🔥): 

- **Tensor Core Troubles**: Discussion revolved around inaccuracies encountered when using `tl.dot()` with `allow_tf32=True` for `fp32` inputs in Triton kernels. A member shared a minimal example highlighting the issue and compared it against PyTorch's results, noting discrepancies and referencing similar [issues on GitHub](https://github.com/openai/triton/issues/1937) that others are experiencing.
  
- **Tricky TF32 Precision**: Members explored the possibility that **TF32**'s lower precision compared to **FP32** might be causing the observed inaccuracies. The conversation led to various experiments and code snippets showing different levels of precision error with Triton's TF32 implementations.

- **Clues from Documentation**: The relevance of **PyTorch documentation** was underscored in understanding when TF32 might be utilized in matrix multiplications and the potential mismatch with expectations for FP32 precision. Further, there was acknowledgement that Triton's documentation could better highlight these disparities for newcomers.

- **Probing Performance Profiles**: A member inquired about setting up profiling for Triton code using **Nsight Compute** to view the PTX and Python code side by side, as detailed in a [Triton acceleration blog post](https://pytorch.org/blog/accelerating-triton/). The response included a command example for generating the profile data which could be helpful for optimizing Triton kernels.

- **Profiling Practices**: Another question arose concerning the general approach to using **Nsight Compute remotely**, with a member obtaining guidance on exporting trace files for UI viewing on a local computer. This method was positioned as an effective alternative to the more laborious remote profiling setup.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html">torch.set_float32_matmul_precision &mdash; PyTorch 2.2 documentation</a>: no description found</li><li><a href="https://pytorch.org/blog/accelerating-triton/">Accelerating Triton Dequantization Kernels for GPTQ</a>: TL;DR  </li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices.">CUDA semantics &mdash; PyTorch 2.2 documentation</a>: no description found</li><li><a href="https://github.com/NVIDIA/cutlass/discussions/385">[2.8] What is new? · NVIDIA/cutlass · Discussion #385</a>: CUTLASS 2.8 was released on 11/19, its anniversary, and tagged recently. In this release, we have several new exciting features. As announced in GTC, we released 3xTF32 gemm, complex gemm, conv2d k...</li><li><a href="https://github.com/openai/triton/blob/740b985bcd86a91ce0fbba7a997025d45c19e38b/python/triton/runtime/jit.py#L285.">triton/python/triton/runtime/jit.py at 740b985bcd86a91ce0fbba7a997025d45c19e38b · openai/triton</a>: Development repository for the Triton language and compiler - openai/triton</li><li><a href="https://github.com/openai/triton/issues/1937">Incorrect Results with TF32 on Main · Issue #1937 · openai/triton</a>: Running on Ampere A6000, Triton commit fd89aa1d2bca4652f383b70f81d993f258e4440f Taken from this issue: #1840 import torch import triton import triton.language as tl @triton.autotune( configs=[ trit...</li><li><a href="https://github.com/openai/triton/issues/2843">tl.dot() has a too large precision error · Issue #2843 · openai/triton</a>: import torch import triton import triton.language as tl @triton.jit def test_kernal(X, Y, O, stride_x_m, stride_x_k, stride_y_k, stride_y_n, stride_o_m, stride_o_n, m:tl.constexpr, k:tl.constexpr, ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1223224367228518544)** (14 messages🔥): 

- **Seeking CUDA Setup on an Old Mac**: A member inquired about setting up **CUDA C++** on an early 2015 MacBook running macOS **Big Sur**.
- **CUDA Requires Compatible Hardware**: It was clarified that to run **CUDA** applications, one must have a **CUDA-capable device**.
- **Question on CUDA Without Local Toolkit**: A member wondered if setting up the **CUDA** requirements in Visual Studio would work without installing the local **CUDA toolkit**.
- **Alternate CUDA Platform Suggestion**: The use of **Google Colab** for running **CUDA C++** was mentioned as a potentially simpler alternative to configuring a local or virtual setup.
- **Trouble Installing NSight DL Design on Ubuntu**: A member faced difficulties finding the **NSight DL Design** app after installing it using a `.run` file and providing the necessary permissions on **Ubuntu**.
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1223611995912409098)** (3 messages): 

- **A New Finetuning Sheriff in Town**: **PyTorch** has released a configuration for [single card finetuning of LLaMA 7B models](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml), indicating it's possible to fine-tune large language models on a **single GPU** with lower memory requirements.
- **PyTorch Team Brewing a Response**: Following a tweet by Jeff Dean showcasing **JAX and TensorFlow** outperforming PyTorch in benchmarks, a member from the **PyTorch team** noted that there were "a few issues with the benchmarks" and is currently working on a response.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/JeffDean/status/17742">Tweet from derek dukes (@ddukes)</a>: Bonfire at the beach about to watch the sunset.  Can you beat california?</li><li><a href="https://x.com/JeffDean/status/1774274156944859455">Tweet from Jeff Dean (@🏡) (@JeffDean)</a>: Here&#39;s the key benchmark table from the link. The JAX backend on GPUs is fastest for 7 of 12 benchmarks, and the TensorFlow backend is fastest for the other 5 of the 12. The Pytorch backend is not...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml">torchtune/recipes/configs/llama2/7B_full_single_device_low_memory.yaml at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1223705270769680387)** (1 messages): 

- **Flash Attention Gets the Spotlight**: The CUDA-MODE community is gearing up for Lecture 12 on **Flash Attention**, scheduled to start at the given timestamp. The session will be presented by a noted member of the community.
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1223614150022922312)** (3 messages): 

- **Aaron's New Optimization Recipe**: A link was shared to [Aaron Defazio's tweet](https://x.com/aaron_defazio/status/1773778436219125948), hinting at a new **optimization approach** that significantly outperforms a tuned **Adam optimizer** on the DLRM benchmark.

- **Pondering Potential Connections**: A channel member speculated on whether the new optimization method could be related to **D-Adaptation** or connected to the **DoWG** approach.

**Link mentioned**: <a href="https://x.com/aaron_defazio/status/1773778436219125948">Tweet from Aaron Defazio (@aaron_defazio)</a>: Update: Hold onto your hats, more results coming in! My new optimization approach demolishes a tuned Adam on DLRM.

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1223193054178181152)** (9 messages🔥): 

- **Curiosity about GPU Architecture**: A member expressed interest specifically in learning about GPU architecture rather than machine learning aspects.
- **Seeking Hardware Specs for Learning**: One user inquired about the necessary hardware to follow a lecture series, asking if cloud resources like Google Colab can be utilized or if an Nvidia GPU is required.
- **Free GPU Resources Identified**: It was highlighted that [Google Colab](https://colab.research.google.com/) provides Nvidia T4 GPUs on their free plan and that [Lightning AI Studio](https://lightning.ai/pricing) offers free GPU time with a pay-as-you-go option for additional resources.
- **Suitability of Colab for CUDA Programming**: Members affirmed that both Colab Pro and regular Colab are suitable for following lectures and practicing CUDA programming.
- **Laptop GPUs Adequate for CUDA Development**: A user confirmed that an Nvidia laptop GPU is sufficient to follow along with the CUDA-related lectures.

**Link mentioned**: <a href="https://lightning.ai/pricing">Lightning AI | Turn ideas into AI, Lightning fast</a>: The all-in-one platform for AI development. Code together. Prototype. Train. Scale. Serve. From your browser - with zero setup. From the creators of PyTorch Lightning.

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1223697783731781774)** (4 messages): 

- **No Answers Without Effort**: A member highlighted the importance of personal effort, stating that before receiving help with answers, one must first share an attempt by sending a photo via direct message.

- **Reading Before Solving**: Another member shared their strategy for tackling the book, planning to read through Part 1 fully before circling back to attempt the questions.

- **GitHub for Better Organization**: The same member proposed using a private GitHub repository to organize the study group's work as it might be more structured compared to a shared document.

- **Clarification on Memory Load Phases**: A query was raised regarding the division of memory loads into phases in Chapter 5.3, figure 5.8, seeking clarification on whether having 2 phases instead of 4 was to ensure memory access independence and questioning if global memory loads are preserved or trashed between phases.
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1223897146684866625)** (3 messages): 

- **Quality Concerns for Lecture 12 Video**: A member has flagged that the **Lecture 12** video on YouTube is of poor quality, making the slides unreadable.
- **Resolution Processing Takes Time**: In response to the quality concern, it was clarified that YouTube requires time to process higher resolution versions of the video and recommended to **check back later**.
- **Lecture on Flash Attention**: A link to the Lecture 12 video titled "Lecture 12: Flash Attention" was shared, but no additional description was provided. [Watch Lecture 12: Flash Attention](https://youtu.be/zEuwuCTEf_0?si=pm1K-pXmOLs8rsWc).

**Link mentioned**: <a href="https://youtu.be/zEuwuCTEf_0?si=pm1K-pXmOLs8rsWc">Lecture 12: Flash Attention</a>: no description found

  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1224112366258622575)** (1 messages): 

- **Feedback on Pull Request Delivered**: Apologies were extended for the delay in reviewing a PR, with feedback now provided at the given [GitHub link](https://github.com/pytorch-labs/ao/pull/95#issuecomment-2028912362). An offer to pair program on Monday was made to assist in addressing the feedback items.

**Link mentioned**: <a href="https://github.com/pytorch-labs/ao/pull/95#issuecomment-2028912362">GaLore and fused kernel prototypes by jeromeku · Pull Request #95 · pytorch-labs/ao</a>: Prototype Kernels and Utils Currently:  GaLore  Initial implementation of fused kernels for GaLore memory efficient training.    TODO:  triton  Composable triton kernels for quantized training and ...

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1223225321046937633)** (90 messages🔥🔥): 

- **Ring-Attention Training Initiatives**: Members are exploring training on 2x A5000 GPUs using a 7B model for 32k sequence lengths, requesting a check for long-context datasets on Hugging Face. The discussion includes attempts at running models on more GPUs and the desire to fine-tune serious models post-successful multi-GPU runs.

- **Long Context Data Hunt**: Members are sourcing long-context datasets, with suggestions such as the [Long-Data-Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections) and [BookSum](https://huggingface.co/datasets/booksum) datasets on Hugging Face. They are encountering issues like dataset compression blocking streaming, leading to the consideration of alternatives like cloud VMs for dataset preparation.

- **Exploration of LLM Training Configurations**: There's an examination of configurations and settings for large language models (LLMs) to improve training and inference, with discussions about tools like Zig-Zag attention from ring-flash-attention and [Distributed Tensor (DTensor)](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md) for PyTorch. 

- **Inference and Evaluation Discussions**: Conversations are ongoing about implementing needle in a haystack evaluation for models and examining the feasibility of implementing varlen ring attention. Efforts include using flash decoding to test existing long-context models and adjusting configurations for sequence lengths.

- **Miscellaneous Updates and Fixes**: Participants are sharing updates on workflow improvements, such as sorted out Axolt patching, and dealing with environment issues such as broken miniconda installations. They are sharing and considering various sources which might help, such as [LLaMA-2-7B-32K blog post](https://www.together.ai/blog/llama-2-7b-32k) and a [VRAM requirements table on Reddit](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/) to address hardware and software demands for model training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://pytorch.org/blog/flash-decoding/">Flash-Decoding for long-context inference</a>: Motivation  </li><li><a href="https://www.together.ai/blog/llama-2-7b-32k">Preparing for the era of 32K context: Early learnings and explorations</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/tree/main/pretrain">togethercomputer/Long-Data-Collections at main</a>: no description found</li><li><a href="https://wandb.ai/iron-bound/axolotl/runs/bd7odfzk/workspace?nw=nwuserironbound">iron-bound</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/cuda-mode/axolotl/commit/d688dbd82f7e9a111e7ad1e182fd5cc89073a099">dispatch_batches for ring_attention · cuda-mode/axolotl@d688dbd</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md">pytorch/torch/distributed/_tensor/README.md at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/ai8hyf/llm_split_recall_test">GitHub - ai8hyf/llm_split_recall_test: Split and Recall: A simple and efficient benchmark to evaluate in-context recall performance of Large Language Models (LLMs)</a>: Split and Recall: A simple and efficient benchmark to evaluate in-context recall performance of Large Language Models (LLMs) - ai8hyf/llm_split_recall_test</li><li><a href="https://github.com/SmerkyG/gptcore">GitHub - SmerkyG/gptcore: Fast modular code to create and train cutting edge LLMs</a>: Fast modular code to create and train cutting edge LLMs - SmerkyG/gptcore</li><li><a href="https://arxiv.org/abs/2311.01282">FlashDecoding++: Faster Large Language Model Inference on GPUs</a>: As the Large Language Model (LLM) becomes increasingly important in various domains. However, the following challenges still remain unsolved in accelerating LLM inference: (1) Synchronized partial sof...</li><li><a href="https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k/tree/main">NousResearch/Yarn-Mistral-7b-128k at main</a>: no description found</li><li><a href="https://huggingface.co/datasets/emozilla/pg_books-tokenized-bos-eos-chunked-65536">emozilla/pg_books-tokenized-bos-eos-chunked-65536 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml">torchtune/recipes/configs/llama2/7B_full_single_device_low_memory.yaml at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/Yukang/LongAlpaca-12k">Yukang/LongAlpaca-12k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Yukang/LongAlpaca-16k-length">Yukang/LongAlpaca-16k-length · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/emozilla/pg19">emozilla/pg19 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/kmfoda/booksum">kmfoda/booksum · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/abacusai/LongChat-Lines/viewer/default/1100">abacusai/LongChat-Lines · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1223191403215982653)** (16 messages🔥): 

- **Searching for Chinese Keywords Is Tough**: A member expressed a need for a Chinese glossary to facilitate easier searching, indicating challenges with current search functions.
- **Crowdsourcing Papers on Distributed Training**: Erica asked for recommendations of research papers on distributed training, showing interest in both the mathematical and practical aspects.
- **Distributed Deep Learning Profiling on AWS**: Erica shared a [research paper](https://arxiv.org/abs/2208.14344) discussing a comprehensive profiler for distributed deep learning (DDL) in a public cloud, specifically characterizing various AWS GPU instances.
- **Optimizing Cross-Mesh Resharding**: An [abstract](https://proceedings.mlsys.org/paper_files/paper/2023/hash/a42cbafcabb6dc7ce77bfe2e80f5c772-Abstract-mlsys2023.html) from MLSys 2023 was linked, introducing a study on cross-mesh resharding in model-parallel deep learning, addressing a many-to-many multicast communication problem.
- **Techniques for Training Massive Transformer Models**: Another [paper](https://arxiv.org/abs/1909.08053) provided insights on techniques to train very large transformer models with billions of parameters using a simple yet efficient intra-layer model parallel approach.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://proceedings.mlsys.org/paper_files/paper/2023/hash/a42cbafcabb6dc7ce77bfe2e80f5c772-Abstract-mlsys2023.html">On Optimizing the Communication of Model Parallelism</a>: no description found</li><li><a href="https://arxiv.org/abs/1909.08053">Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism</a>: Recent work in language modeling demonstrates that training large transformer models advances the state of the art in Natural Language Processing applications. However, very large models can be quite ...</li><li><a href="https://www.computer.org/csdl/proceedings-article/hcs/2023/10254716/1QKTnGyUPbG">CSDL | IEEE Computer Society</a>: no description found</li><li><a href="https://www.furygpu.com/">FuryGpu</a>: no description found</li><li><a href="https://www.computer.org/csdl/proceedings-">CSDL | IEEE Computer Society</a>: no description found</li><li><a href="https://arxiv.org/abs/2208.14344">Analysis of Distributed Deep Learning in the Cloud</a>: We aim to resolve this problem by introducing a comprehensive distributed deep learning (DDL) profiler, which can determine the various execution &#34;stalls&#34; that DDL suffers from while running o...</li><li><a href="https://www.youtube.com/watch?app=desktop&v=rsxCZAE8QNA">HC2023-K2: Hardware for Deep Learning</a>: Keynote 2, Hot Chips 2023, Tuesday, August 29, 2023Bill Dally, NVIDIABill describes many of the challenges of building hardware optimized for Deep Learning a...</li><li><a href="https://www.youtube.com/watch?v=3LVeEjsn8Ts">John Hennessy and David Patterson 2017 ACM A.M. Turing Award Lecture</a>: 2017 ACM A.M. Turing Award recipients John Hennessy and David Patterson delivered their Turing Lecture on June 4 at ISCA 2018 in Los Angeles. The lecture too...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1223277761532137627)** (10 messages🔥): 

- **Persistent Error Despite Using an Older triton-viz**: Despite installing an older version of triton-viz and resetting the colab runtime, **zhacker2798** continues to encounter the same error.
- **New Notebook Proposed as Solution**: **srush1301** suggests using the most recent notebook, which is purported to resolve existing issues. He later requests **zhacker2798** to retry, confirming that it works for him.
- **Installation Procedure for Triton Visualisation**: **srush1301** provides a detailed code snippet outlining the installation process for the correct triton-viz setup, which includes dependencies such as Jaxtyping and custom environment variable settings.
- **Confirmation of the Fix**: User **yogeshg6849** confirms that following the installation instructions provided in the code cell resolved their issues, expressing gratitude.
- **Important Note for Local Installs**: **glowingscientist** advises that when running locally, installing PyTorch after using the provided cell can revert to an incompatible version of Triton, which could lead to issues.
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1223215173213491351)** (130 messages🔥🔥): 

- **Seeking DBRX Access**: A member inquired about where to test **DBRX** without self-hosting. It was mentioned that **DBRX** could be tested either on the Huggingface space or [You.com](https://you.com/).

- **GPU Troubles and CUDA Errors**: Members are discussing issues with **GPUs on Runpod**. Symptoms include experiencing API CUDA errors and multiple GPUs resulting in **out of memory (OOM)** errors even when using systems such as **axolotl**. 

- **Ring-Attention Sequence Length**: A request for datasets with sequence lengths of 16k-32k arose for testing ring-attention. Recommendations included datasets on Huggingface and a link to a GitHub repository [containing modified ring-attention implementations](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching).

- **Transcription Trials and Tribulations**: Conversations revolve around extracting text from audio, predominantly in English and Chinese, with solutions like **Whisper for single speaker** noted as effective. When multi-speakers are involved, the process becomes more complex, with other solutions like Assembly AI, and **whisperx with diarization** being recommended for more demanding scenarios.

- **Fine-Tuning Finesse and VRAM Efficiency**: Discussions about **torchtune** for fine-tuning 7B models with less than 16GB of VRAM and **Shisa project** for Japanese LLM development. Links to GitHub and discussions about tokenizer efficiency and data pretraining highlight the community's experiments and strategies for efficient model training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/UQvCwV4M">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://wandb.ai/augmxnt/shisa-v2/runs/o830e1kw">augmxnt</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://sambanova.ai/blog/accurate-models-at-blazing-speed">SambaNova Delivers Accurate Models At Blazing Speed</a>: Samba-CoE v0.2 is climbing on the AlpacaEval leaderboard, outperforming all of the latest open-source models. </li><li><a href="https://huggingface.co/spaces/sambanovasystems/Samba-CoE-v0.1">Samba CoE V0.1 - a Hugging Face Space by sambanovasystems</a>: no description found</li><li><a href="https://wandb.ai/iron-bound/axolotl/runs/wyf5iblj/workspace">iron-bound</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://www.together.ai/blog/llama-2-7b-32k">Preparing for the era of 32K context: Early learnings and explorations</a>: no description found</li><li><a href="https://huggingface.co/augmxnt/shisa-base-7b-v1">augmxnt/shisa-base-7b-v1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/AUGMXNT/shisa/wiki/Tokenizer-Efficiency">Tokenizer Efficiency</a>: Contribute to AUGMXNT/shisa development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/shisa-ai/shisa-v2/blob/main/_base-evals/tokenizer-efficiency/tokenizer-eval-ja.md">shisa-v2/_base-evals/tokenizer-efficiency/tokenizer-eval-ja.md at main · shisa-ai/shisa-v2</a>: Contribute to shisa-ai/shisa-v2 development by creating an account on GitHub.</li><li><a href="https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis">A Review of Public Japanese Training Sets</a>: Contribute to AUGMXNT/shisa development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/Shitao/MLDR">Shitao/MLDR · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/cuda-mode/axolotl/tree/ring_attention_patching">GitHub - cuda-mode/axolotl at ring_attention_patching</a>: Go ahead and axolotl questions. Contribute to cuda-mode/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1223223976269189201)** (45 messages🔥): 

- **Memory Optimization with PagedAdamW**: A member highlighted the substantial memory savings when using **PagedAdamW**, with a peak memory usage of approximately 14GB compared to 27GB for 8-bit Adam. A link to a [config file example](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml) was provided for context.

- **Axolotl's Limited Control Over Training**: Members discussed that in axolotl, they lack the same level of control over training as provided by **torchtune**, explaining the absence of significant memory savings. It was mentioned that a combination of **paged adamw** and integration into the backward pass could be responsible for the observed memory savings.

- **Implementing DBRX in Axolotl Requires Work**: Responding to a query about the effort needed to support **DBRX** in axolotl, a member indicated substantial work is required, citing a [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1462) as a reference for examples working on powerful GPU setups.

- **Challenges of Gradient Accumulation and Multi-GPU Compatibility**: The team debated the complexities of optimizing memory usage by fusing the optimizer step into the backward pass, considering drawbacks such as the inability to gradient accumulate effectively and issues with multi-GPU setups.

- **Exploration of the LISA Branch**: A member reported on experimenting with the **lisa branch**, encountering out-of-memory (OOM) issues, and following up with a [test run](https://wandb.ai/tmm1/lisa-tests/runs/c35oan2s). They also pointed out the necessity to adjust configurations to ensure proper layer freezing and the successful merging of a **lisa** related PR.

- **Suggestion for an April Fools' Day Announcement**: A humorous suggestion was made for an April Fools' announcement about axolotl partnering with OpenAI to finetune GPT-4 and all future models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/geronimo_ai/status/1774302298400813196">Tweet from Geronimo (@Geronimo_AI)</a>: there&#39;s something wrong with this, same loss regardless of the number of active layers https://github.com/OptimalScale/LMFlow/issues/726</li><li><a href="https://huggingface.co/mlx-community/dbrx-instruct-4bit">mlx-community/dbrx-instruct-4bit · Hugging Face</a>: no description found</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">How to save memory by fusing the optimizer step into the backward pass — PyTorch Tutorials 2.2.1+cu121 documentation</a>: no description found</li><li><a href="https://wandb.ai/tmm1/lisa-tests/runs/c35oan2s">tmm1</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml">torchtune/recipes/configs/llama2/7B_full_single_device_low_memory.yaml at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1464">Unfreeze layers in mixtral does not work as expected · Issue #1464 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior The config for mixtral.yml contains examples for un...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1462">DBRX Model Support by winglian · Pull Request #1462 · OpenAccess-AI-Collective/axolotl</a>: DBRX MoE Currently, for LoRA, only the q_proj, k_proj, v_proj out_proj and layer Linear layers are trainable. We are using the &quot;converted&quot; base models based on this issue where the Experts a...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1223179318746681344)** (27 messages🔥): 

- **Finding the Right Model for Text Classification**: Members discussed suitable models for text classification with limited GPU resources like T4. Options like **Mistral** and **qlora** were suggested due to their proficiency in English and potential to run on smaller batch sizes, recognizing that a model like **qwen** has not been extensively tested by users.

- **Creating Chat UI for Model Testing**: A member shared a tool they developed called [auto-ollama](https://github.com/monk1337/auto-ollama) to facilitate testing a fine-tuned model via chat, converting models to **ollama** format or to **gguf** for easy use.

- **Fine-tuning Dataset Concerns and Strategy**: Members addressed questions related to fine-tuning with large datasets and potential overtraining, suggesting fewer epochs and the use of tools like **wandb** to track training progress. The importance of matching fine-tuning conditions with inference was highlighted in the context of system messages and user inputs.

- **Challenges in AI-Guided Conversation over Phone**: One member discussed the complexity of creating a **Telegram bot** for AI conversations over phone calls, integrating technologies like **Twilio**, **Whisper**, and a text generator from **TextGen UI**. They reached out for advice on setting up the system, particularly the voice response aspect.

- **Troubleshooting Training Stagnation**: A member faced an issue with training getting stuck after 1 epoch without evaluation. Other members tried to diagnose the issue, discussing potential factors such as evaluating settings, **wandb** integration, and storage constraints.

**Link mentioned**: <a href="https://github.com/monk1337/auto-ollama/tree/main">GitHub - monk1337/auto-ollama: run ollama &amp; gguf easily with a single command</a>: run ollama &amp; gguf easily with a single command. Contribute to monk1337/auto-ollama development by creating an account on GitHub.

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1223168342408298537)** (30 messages🔥): 

- **Modular SDK Open Sourced with Limits**: There's excitement around Modular's standard library being open-sourced, but it's clarified that the SDK still has restrictions on non-internal/commercial applications. The **open-sourcing of the entire SDK** is hoped for but not confirmed.
- **Installation Challenges for Mojo**: Users on Linux Mint faced issues installing Mojo. It's noted that **Ubuntu, MacOS, and WSL2 have official support**; a [user guide](https://docs.modular.com) is referenced for further assistance.
- **Openwall Security Alert**: A member shared an [Openwall security update](https://www.openwall.com/lists/oss-security/2024/03/29/4) about a backdoor found in xz-utils versions 5.6.0 and 5.6.1, with a CVE-2024-3094 listed and patches being distributed.
- **Mojodojo Development Platform**: Discussion about **mojodojo.dev**, a platform previously managed by a community member, turned into a call for contributions since it's **outdated but available for updates** on its [GitHub repository](https://github.com/mojodojodev/mojodojo.dev).
- **Mojodojo's Privacy Standards**: An interesting tidbit surfaced that the **mojodojo.dev domain** uses Icelandic privacy services to obscure registrant information - a reflection of the country’s strong privacy laws rather than an Icelandic source.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com">Modular Docs</a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/get-started/#system-requirements">Get started with Mojo🔥 | Modular Docs</a>: Get the Mojo SDK or try coding in the Mojo Playground.</li><li><a href="https://github.com/mojodojodev">mojodojodev - Overview</a>: mojodojodev has 5 repositories available. Follow their code on GitHub.</li><li><a href="https://www.openwall.com/lists/oss-security/2024/03/29/4">oss-security - backdoor in upstream xz/liblzma leading to ssh server compromise</a>: no description found</li><li><a href="https://github.com/mojodojodev/mojodojo.dev">GitHub - mojodojodev/mojodojo.dev: Learning materials for the Mojo🔥programming language</a>: Learning materials for the Mojo🔥programming language - mojodojodev/mojodojo.dev
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1223323060313133198)** (4 messages): 

- **Mystery Unfolds with Modular Tweet**: Modular shared a cryptic message that sparked curiosity among members, presented in a tweet: [Modular's Mysterious Message](https://twitter.com/Modular/status/1773762747098124491).
- **Follow-up Tweet Stirs Discussion**: A subsequent tweet by Modular raised more questions than answers, intensifying the conversation: [Continued Enigma](https://twitter.com/Modular/status/1773767659278250242).
- **The Plot Thickens at Modular**: As the situation evolved, Modular tweeted yet another puzzling update, keeping the community on their toes: [The Story Progresses](https://twitter.com/Modular/status/1773813736421404685).
- **Awaiting the Big Reveal**: Anticipation built up as Modular prepared for a big announcement, indicated in their latest tweet: [Anticipation Escalates](https://twitter.com/Modular/status/1774528498050425064).
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1223313935202123957)** (1 messages): 

- **New Release MAX 24.2 Explored**: Modular posted a [YouTube video](https://www.youtube.com/watch?v=PL71FV2KKHE) titled "Modular Community Livestream - New in MAX 24.2" discussing the latest updates in MAX 24.2 including the open sourcing of Mojo standard library and MAX Engine support features.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=PL71FV2KKHE">Modular Community Livestream - New in MAX 24.2</a>: MAX 24.2 is now available! Join us on our upcoming livestream as we discuss everything new in MAX - open sourcing Mojo standard library, MAX Engine support f...

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1223135175123271731)** (76 messages🔥🔥): 

- **Mojo Optimizations Under Discussion**: Members shared thoughts on **Mojo's** multithreading potential, such as utilizing *OpenMP* for **multi-core** CPU performance. Discussions on **`external_call()`** were also highlighted, pointing to its capabilities and future improvements for running system commands.

- **Setting Up and Contributing to Mojo**: A member was having trouble running main.mojo after **contributing to Mojo**, even after following the instructions from the *[Mojo Development Guide](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md)*. Other members are curious about **contributing** to Mojo, with references to the [official blog post](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) and the nightly `stdlib` [README on GitHub](https://github.com/modularml/mojo/blob/nightly/stdlib/README.md). 

- **Syntax Highlighting in Discord and Module Behaviors**: **Highlighting code** in Discord was mentioned, recommending the use of `rust` or `python` markup for Mojo code. Moreover, a quip about a "truly random module" led to advice on fixing module issues by changing themes.

- **Improving Mojo's Parallelization**: A discussion unfolded on optimizing code using `parallelize[_call_neurons](self.nout, self.nout)`—contextualized by **momograd.x**, which showed speed improvements in parallel execution over sequential. It was explained that **over-saturating** with works counts higher than the number of CPU cores could yield better performance.

- **Handling Mojo Arrays & Interoperability**: Questions arose about representing fixed-size arrays of custom structs and if `StaticTuple` or `List` should be used; allocators and pointers were mentioned, with suggestions of making structs `register_passable`. The question of **C/C++ interop** in Mojo was also raised.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mojodojo.dev.">Mojo Dojo</a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/buffer/buffer">buffer | Modular Docs</a>: Implements the Buffer class.</li><li><a href="https://www.geeksforgeeks.org/system-call-in-c/amp/">system() in C/C++ - GeeksforGeeks</a>: no description found</li><li><a href="https://docs.modular.com/mojo/notebooks/Matmul">Matrix multiplication in Mojo | Modular Docs</a>: Learn how to leverage Mojo&#x27;s various functions to write a high-performance matmul.</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md">mojo/stdlib/docs/development.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: The Next Big Step in Mojo🔥 Open Source</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/README.md">mojo/stdlib/README.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782">Swift Concurrency Manifesto</a>: Swift Concurrency Manifesto. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1223179107513270347)** (11 messages🔥): 

- **MLIR Syntax Documentation In Progress**: There's an ongoing effort to improve the ergonomics of **MLIR** syntax documentation, as current usage is not user-friendly. Contributors are directed to a notebook for guidance ([Mojo with MLIR](https://docs.modular.com/mojo/notebooks/BoolMLIR)), with the promise of more openness and detail to come as the system matures.

- **Library Module Updates Rolled Out**: Updated versions of several **mojo** libraries (`mojo-prefix-sum`, `mojo-flx`, `mojo-fast-base64`, and `mojo-csv`) to version 24.2 have been announced. `mojo-hash` and `compact-dict` are partially updated but have outstanding issues, specifically with `generic-dict` due to failing tests.

- **Evolving 'Reference' Component**: The `Reference` aspect of the project is acknowledged to be in early development, with expectations of frequent changes and evolution. Improvements are seen as necessary and forthcoming.

- **New Logger Library Introduced**: A new logging library utilizing the **Bound Loggers** pattern, called **stump**, is accessible for trial despite not being officially released. The library is adaptable with preprocessors and text styling functions, and the creator encourages testing ([Stump on GitHub](https://github.com/thatstoasty/stump)).

- **Decorator Functionality Update**: There is currently no support for writing decorators in the standard library, as they are implemented at the compiler level. However, it's anticipated that future support for decorators will mirror Python's approach.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Low-level IR in Mojo | Modular Docs</a>: Learn how to use low-level primitives to define your own boolean type in Mojo.</li><li><a href="https://github.com/thatstoasty/stump/">GitHub - thatstoasty/stump</a>: Contribute to thatstoasty/stump development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1223153258298871880)** (1 messages): 

- **Contribute to Mojo's Future**: The **[Mojo standard library (stdlib)](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)** was open-sourced, enabling community contributions. A member provides a [useful guide](https://vmois.dev/mojo-local-stdlib-build/) to building the stdlib locally on macOS and Linux, highlighting steps to change the stdlib and make Mojo run with the modified version.

- **Building and Implementing Custom stdlib**: To use a customized stdlib, one must first pull the **[Mojo repository](https://github.com/modularml/mojo)**, then make changes and build the library using the `build-stdlib.sh` script. The custom stdlib can be found in the `build/stdlib.mojopkg` folder, and the guide details replacing the standard stdlib located in the `~/.modular` directory with this new version.

**Link mentioned**: <a href="https://vmois.dev/mojo-local-stdlib-build/">Use locally built standard library in Mojo</a>: Mojo standard library (stdlib) was open-sourced yesterday. It is exciting that the community can now contribute directly to the codebase. After spending some time with the stdlib repository, I want to...

  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1223857151450288209)** (3 messages): 

- **The One Billion Row Challenge in Modular (Mojo) Language**: A user working on the [one billion row challenge](https://github.com/VMois/1brc-mojo/tree/main) highlighted that Mojo lacks certain standard library features like string sorting and causes excessive data copying, making programs run for a long time. They referenced specific issues like `read_bytes` and the behavior of `String.split`, and expressed a need for better profiling tools to understand memory allocations.

- **Matrix Multiplication Example Stumbles on Mojo**: Another user encountered an error with the example `matmul.mojo`, noting inconsistency in results due to a call to `test_matrix_equal[matmul_vectorized](C, A, B)`. By introducing a tolerance in the comparison, the example could run, but the underlying cause seemed to be a rounding error.

- **Finding the Perfect Float**: The same user attempted to resolve the matrix multiplication error by changing the data type alias from `DType.float32` to `DType.float64`, which fixed some errors but did not completely eliminate the issue.

**Link mentioned**: <a href="https://github.com/modularml/mojo/issues/2051)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

  

---


**Modular (Mojo 🔥) ▷ #[⚡serving](https://discord.com/channels/1087530497313357884/1212827597323509870/1224265805278085180)** (3 messages): 

- **MAX Serving as a Triton Backend**: MAX Serving is confirmed to work as a drop-in replacement for existing backends on Triton Inference Server. Users should update the Triton model configuration files and use the MAX Serving container image as outlined in the [MAX Serving Trial Guide](https://docs.modular.com/serving/get-started).

- **Migration Concerns Addressed**: The team behind MAX Serving is offering assistance for those considering migration, emphasizing an easy and seamless transition process. Users are encouraged to reach out directly for personalized support to optimize their pipelines with MAX.

**Link mentioned**: <a href="https://docs.modular.com/serving/get-started">Get started with MAX Serving | Modular Docs</a>: A walkthrough showing how to try MAX Serving on your local system.

  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1223184867307687936)** (49 messages🔥): 

- **Edgy Teenager Models and Math?**: A member ponders the value of math benchmarks for models deployed with personalities like an "edgy teenager," questioning the practicality when persona is a factor.
- **Introducing Arena-Hard**: A new [Arena-Hard benchmark](https://github.com/lm-sys/arena-hard) by lm-sys is referenced, described as a "10x harder mt bench" aimed at challenging language models with user queries and comparing with a baseline.
- **Skepticism About LLMs as Judges**: Members discuss potential issues with using large language models (LLMs) as benchmarks judges, noting that a model may prefer its style of reasoning, as seen in **GPT-4** self-scoring higher against **Claude** due to such bias.
- **Discussing Model Bias and Benchmarks**: A member highlights the discovery of broad topic ranges in the lm-sys system's user base from a [March paper](https://arxiv.org/abs/2403.04132), while also noting apprehension due to potential topic and population narrowness.
- **GPT-4 Versus Claude on Arena-Hard**: In debates about benchmarks, **GPT-4** is indicated to perform significantly better than **Claude** on the lm-sys Arena-Hard, raising questions about inherent model bias and the effectiveness of length correction in evaluations.

**Link mentioned**: <a href="https://github.com/lm-sys/arena-hard">GitHub - lm-sys/arena-hard: Arena-Hard benchmark</a>: Arena-Hard benchmark. Contribute to lm-sys/arena-hard development by creating an account on GitHub.

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1223388203688525858)** (7 messages): 

- **Seeking a Measure for Token Usefulness**: A member expressed interest in a measurement to determine the information density of tokens, separating useful content from filler in GPT-4's outputs.
- **Academic Interest Acknowledged**: In response to the quest for token information density metrics, it was suggested that creating such a measure could be an intriguing academic endeavor, with potential methods to normalize it against length bias.
- **Mutual Information as a Potential Measure**: The concept of mutual information was mentioned as a potentially good proxy for determining the informational content of tokens.
- **Control Vectors Could Target Filler Tokens**: One contributor pointed out the resemblance between the quest for token information density and the Microsoft LLM Lingua project, highlighting the effectiveness of control vectors in targeting these tokens via *repeng*.
- **An Information-Theoretic Approach to Text Generation**: A study on Typicality in the context of information theory was shared, offering a methodology to steer stochastic decoding toward providing more 'information-rich' text, although it may not yet be ready to supplant temperature-based top-p/top-k methods. [View the paper here](https://arxiv.org/abs/2202.00666).

**Link mentioned**: <a href="https://arxiv.org/abs/2202.00666">Locally Typical Sampling</a>: Today&#39;s probabilistic language generators fall short when it comes to producing coherent and fluent text despite the fact that the underlying models perform well under standard metrics, e.g., perp...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1223424229144530954)** (12 messages🔥): 

- **Headhunting Season at Stability**: The chat indicates Stability AI is in the **"get the good researchers before anyone else does"** phase according to Nathan Lambert.
- **Stress-Free Hiring Strategy**: Nathan Lambert mentions being cautious to not stress out Louis by asking for hiring pointers, instead offering to help others learn about Synth Labs’ offerings.
- **Synth Labs Breaks the Mold**: A comment by Nathan Lambert suggests that it is **not normal** for startups to emerge from stealth with a paper that will drive their first products, implying Synth Labs is doing something exceptional.
- **Acknowledging the Unusual**: The use of emojis from another member underscores the unusual but **commendable approach** of Synth Labs in the startup space.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1223518921114521640)** (2 messages): 

- **Validating Bitnet with Dolma**: NousResearch has released a 1B model to validate and independently confirm the claims of the Bitnet paper. The model is trained on the first 60B tokens of the Dolma dataset and is available on [Hugging Face](https://huggingface.co/NousResearch/OLMo-Bitnet-1B).
- **Performance Insights on Weights & Biases**: Comparisons between the Bitnet implementation and a full FP16 run (with equivalent hyperparameters) can be reviewed on [Weights & Biases charts](https://api.wandb.ai/links/emozilla/evltqiv7).
- **1 Bit Training? Curiosity and Confusion**: A member expressed interest in the Bitnet research but admitted a lack of understanding regarding what 1 bit training entails.

**Link mentioned**: <a href="https://x.com/nousresearch/status/1773923241268003052?s=46">Tweet from Nous Research (@NousResearch)</a>: We are releasing our first step in validating and independently confirming the claims of the Bitnet paper, a 1B model trained on the first 60B tokens of the Dolma dataset.  Comparisons made on the @we...

  

---


**Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1224375780545396889)** (2 messages): 

- **Exploring Verbosity in Direct Preference Optimization**: A new [preprint](https://x.com/rm_rafailov/status/1774653027712139657?s=46) examines the interplay between Direct Preference Optimization (DPO) and verbosity in large-scale language model training, noting increased verbosity leads to model divergence. This verboseness issue has been observed in the Open Source community as well.
- **Addressing Bias in Language Model Training**: The preprint introduces research on Reinforcement Learning from Human Feedback (RLHF) and its susceptibility to human biases, such as favoring eloquence over helpfulness. It is available in both [PDF](https://arxiv.org/pdf/2403.19159) and [HTML formats](https://arxiv.org/html/2403.19159v1).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19159">Disentangling Length from Quality in Direct Preference Optimization</a>: Reinforcement Learning from Human Feedback (RLHF) has been a crucial component in the recent success of Large Language Models. However, RLHF is know to exploit biases in human preferences, such as ver...</li><li><a href="https://x.com/rm_rafailov/status/1774653027712139657?s=46">Tweet from Rafael Rafailov (@rm_rafailov)</a>: New preprint is out on interplay between DPO and verbosity. Some of the first feedback we got on DPO was that training on LARGE scale the model becomes increasingly verbose until it diverges. Verbosit...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1224063478147584051)** (8 messages🔥): 

- **Refining LLM Alignment with sDPO**: A [new paper](https://arxiv.org/abs/2403.19270) introduces **stepwise DPO (sDPO)**, an extension of direct preference optimization aimed at aligning large language models more closely with human preferences by using datasets in a stepwise fashion for improved performance.
- **sDPO Could Level the Playing Field for Smaller Labs**: The sDPO method hints at allowing smaller labs to achieve similar performance gains as larger ones, without needing extensive financial resources.
- **DPO Strategy Questioned**: A member humorously commented on the frequency of direct preference optimization (DPO) use, suggesting it's being done repeatedly instead of exploring other methods like **reinforcement learning from human feedback (RLHF)**.
- **Paper Participation**: Nathan Lambert mentioned involvement in a similar paper, indicating interest in the approach of batching preference data efficiently.
- **Exploring Multiple Steps in DPO**: A query was raised whether past DPO efforts have only used single gradient steps on entire datasets, and it was confirmed that generally, datasets were randomly sampled and used off-policy to reduce loss.

**Link mentioned**: <a href="https://arxiv.org/abs/2403.19270">sDPO: Don&#39;t Use Your Data All at Once</a>: As development of large language models (LLM) progresses, aligning them with human preferences has become increasingly important. We propose stepwise DPO (sDPO), an extension of the recently populariz...

  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1223784123550793861)** (24 messages🔥): 

- **Nathan Pledges to Chronicle Open Alignment**: Nathan Lambert is embarking on a project to document the history of open alignment datasets and practices following ChatGPT's introduction, starting with a lecture at Stanford and accompanying blog posts.
- **Interest in Open Alignment Evolution**: Members of the channel have expressed enthusiasm about Nathan Lambert sharing notes on the development of open alignment models, offering support and looking forward to insights on the subject.
- **The ChatGPT Reproduction Rush**: The project will cover the initial race to replicate ChatGPT, highlighting models such as **alpaca, koala, dolly, vicuna**, and **llama 1**.
- **DPO Versus IPO**: Key discussions include the debate between Direct Preference Optimization (DPO) and Indirect Preference Optimization (IPO), with reference to a GitHub issue on adding DPOTrainer in trl ([Issue #405](https://github.com/huggingface/trl/issues/405)).
- **Valuable Resources and Progress Notes Shared**: For those following the history of open alignment, Nathan Lambert has shared a medium article listing open-sourced fine-tuned large language models ([Open-Sourced LLMs](https://sungkim11.medium.com/list-of-open-sourced-fine-tuned-large-language-models-llm-8d95a2e0dc76)) and provided a link to his detailed notes ([Lambert's Notion Notes](https://magnetic-share-282.notion.site/History-of-Open-Alignment-a7ef20aefb34438185336df68147809e?pvs=4)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://magnetic-share-282.notion.site/History-of-Open-Alignment-a7ef20aefb34438185336df68147809e?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://huggingface.co/blog?tag=rlhf">Hugging Face – Blog</a>: no description found</li><li><a href="https://github.com/huggingface/trl/issues/405">Adding `DPOTrainer` in trl · Issue #405 · huggingface/trl</a>: DPO (Direct Preference Optimization) is a recent paper from Stanford University: https://arxiv.org/abs/2305.18290 It could be interesting to implement a generalist trainer that supports this algori...
</li>
</ul>

</div>
  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1223177586960498708)** (16 messages🔥): 

- **Jamba for code tasks remains a mystery**: There's curiosity about **Jamba-v0.1's performance on Code tasks**, as its efficacy on the **HumanEval benchmark** hasn't been discussed yet.
- **Language inclusivity in training data queried**: A member inquired if the **training data for Jamba** included the **Czech language**.
- **Anticipation for Jamba's fine-tuning capabilities**: Although **Jamba** is not currently available for fine-tuning on **AI21 Studio**, it's expected that the fine-tuned instruct model will be hosted there soon.
- **Jamba's efficiency despite hardware constraints**: Discussions highlight **Jamba's** efficiency, with its **MoE layers drawing on just 12B of its available 52B parameters** at inference. However, there's acknowledgment that even with such efficiency, running it on consumer-grade hardware like an NVIDIA 4090 remains unfeasible.
- **Quantization and llamacpp could lighten Jamba's load**: There's speculation that **quantization** and **llamacpp support** might enable **Jamba** to run on 24GB of VRAM, though some members still consider this resource-intensive.
- **Confusion Over Jamba's Speed with More Tokens**: A member questioned how **Jamba** becomes *faster* with more tokens during the encoding and decoding tasks, pointing to a specific figure (3b) in the latest paper where this phenomenon is observed.
  

---


**AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1223201037943574630)** (51 messages🔥): 

- **Clarification on "Single GPU" Terminology**: The discussion clarified that "a single GPU" references to running Jamba on cards like an A100 with 80GB, suggesting that while there might be enthusiasm about its suitability for lesser GPUs with 24GB, it primarily targets higher-end hardware configurations.
- **Quantization Possibilities Discussed**: Members exchanged information on running quantized versions of models on lower capacity GPUs and pointed to using 4-bit precision loading, as per the [model card guidelines](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://huggingface.co/docs/accelerate/en/usage_guides/quantization&ved=2ahUKEwjQ-4eopJmFAxWaXmwGHZPhBN8QFnoECBMQAQ&usg=AOvVaw2RxBEXoJMjtWqDScwaFZqc).
- **Efficiency of Jamba vs. Traditional Transformers**: A query arose regarding Jamba's efficiency and high throughput, despite the transformer block making it scale with sequence length squared. Dialogue revealed that Jamba's Mamba and MoE (mixture of experts) layers underpin its sub-quadratic scaling and efficiency, beyond the optimizations of traditional transformer layers.
- **Request for Model Architecture Visuals**: One user inquired if there are any available diagrams showcasing the specifics of the blocks used within the Jamba model, reflecting a desire to understand the balance between transformer and Mamba blocks within the architecture.
- **Exploring the Composition of Jamba Blocks**: In the explanation of the Jamba block structure, it was highlighted that Mamba and MoE layers play a crucial role, with a ratio of one Transformer layer out of every eight total layers, clarifying that Transformer layers are not part of the MoE, but rather integrated into specific Jamba blocks.

**Link mentioned**: <a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://huggingface.co/docs/accelerate/en/usage_guides/quantization&ved=2ahUKEwjQ-4eopJmFAxWaXmwGHZPhBN8QFnoECBMQAQ&usg=AOvVaw2RxBEXoJMjtWqDScwaFZqc">no title found</a>: no description found

  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1223183353524781159)** (34 messages🔥): 

- **A Warm Welcome to AI Enthusiasts**: New members inquired about the right place for connecting with AI developers, and they received confirmations that they've found the appropriate channel.
- **Seeking Guidance on Generating GraphQL from Prompts**: One member asked for experiences related to generating GraphQL from prompts, indicating interest in learning from others.
- **Langchain Capabilities Query**: There was a query about what Langchain helps in, suggesting a newcomer is seeking an understanding of its capabilities and use cases.
- **Looking for Logging Assistance**: A user building a chat application with a fine-tuned model expressed a need for a guide on using Langsmith to log chat responses and feedback into a database, and was pointed to the relevant [Langsmith documentation](https://docs.smith.langchain.com/tracing/faq) for assistance.
- **Exploration of Langchain Use Cases**: Members discussed various Langchain applications, from localizing large document QA chains to building conversational agents with scenario-specific prompts, indicating a wide range of practical implementations within the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json">JSON files | 🦜️🔗 Langchain</a>: The JSON loader use JSON pointer to target keys in your JSON files you want to target.</li><li><a href="https://docs.smith.langchain.com/tracing/faq">How-To Guides | 🦜️🛠️ LangSmith</a>: In this section you will find guides for how to use LangSmith tracing functionality</li><li><a href="https://docs.smith.langchain.com/tracing/faq/logging_and_viewing#logging-traces">How to log and view traces to LangSmith | 🦜️🛠️ LangSmith</a>: LangSmith makes it easy to log and view traces from your LLM application, regardless of which language or framework you use.</li><li><a href="https://docs.smith.langchain.com/tracing/faq/logging_feedback">How to Collect Feedback for Traces | 🦜️🛠️ LangSmith</a>: Feedback allows you to understand how your users are experiencing your application and helps draw attention to problematic traces. LangSmith makes it easy to collect feedback for traces and view it in...</li><li><a href="https://producthunt.com/posts/sciphi?"> SciPhi - One-click RAG deployment for developers | Product Hunt</a>: SciPhi is a cloud platform for developers that simplifies building and deploying serverless RAG pipelines. Built with the open source R2R framework, it enables developers to focus on innovative AI app...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/)** (1 messages): 

blackice9833: free nudes ♥️
https://discord.gg/bestnudes 
@everyone @here
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1223165177617449010)** (14 messages🔥): 

- **Galaxy AI Launches Free Premium AI Model APIs**: GalaxyAI announces the availability of **free API service** to access premium AI models, including **GPT-4**, **GPT-4-1106-PREVIEW**, **GPT-3.5-turbo-1106**, and more with OpenAI format compatibility for easy integration into projects. Interested users can try it [here](https://galaxyapi.onrender.com).

- **New Blog Post on Model Alignment**: A recent blog post explores **model alignment** in large language models (LLMs), delving into the effectiveness of RLHF, DPO, and KTO methods when applied to the Mistral and Zephyr 7B models. The full post can be read on [Premai's blog](https://blog.premai.io/model-alignment-process/).

- **Introducing Chain of Tasks for Taskbots**: The introduction of the **Chain of Tasks** prompting technique is explored in two blog posts, showcasing its application for creating advanced conversational LLM **Taskbots** with LangGraph. Readers can explore the methods and potential applications through the [LinkedIn article on Chain of Tasks](https://www.linkedin.com/posts/prasadt_introducing-chain-of-tasks-cota-a-prompting-activity-7178582571423870976-wajV).

- **CrewAI Framework Announcement**: CrewAI is a cutting-edge framework for orchestrating autonomous AI agents, built on top of Langchain and offering default integration with OpenAI and local LLMs. Those interested can explore further on their [website](https://crewai.com), check out the [GitHub repository](https://github.com/joaomdmoura/crewAI/tree/main), or join their Discord community.

- **AI-Powered Stock Analysis Tool Launch**: A custom-developed GPT model that analyzes stocks and rates investment potential on a scale from 1 to 10 has been released, with feedback invitations extended to community members. Potential investors can test it out at the provided [OpenAI Chat link](https://chat.openai.com/g/g-3QdjVv8TG-investor-gpt).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.premai.io/model-alignment-process/">Model Alignment Process</a>: The alignment of generative models with human feedback has significantly improved the performance of natural language generation tasks. For large language models (LLMs), alignment methods like reinfor...</li><li><a href="https://amgadhasan.substack.com/p/sota-asr-tooling-long-form-transcription">SOTA ASR Tooling: Long-form Transcription</a>: Benchmarking the different Whisper frameworks for long-form transcription</li><li><a href="https://www.reddit.com/r/Pictures/s/1EGIE9uaRw">Reddit - Dive into anything</a>: no description found</li><li><a href="https://producthunt.com/posts/sciphi?"> SciPhi - One-click RAG deployment for developers | Product Hunt</a>: SciPhi is a cloud platform for developers that simplifies building and deploying serverless RAG pipelines. Built with the open source R2R framework, it enables developers to focus on innovative AI app...</li><li><a href="https://crewai.com">crewAI - Platform for Multi AI Agents Systems</a>: no description found</li><li><a href="https://www.crewai.com/crewaiplus">crewAI+ - Platform for Multi AI Agents Systems</a>: no description found</li><li><a href="https://github.com/joaomdmoura/crewAI/tree/main">GitHub - joaomdmoura/crewAI: Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.</a>: Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks. - joaomdmoura/cr...</li><li><a href="https://discord.gg/Kz3HbJx23n">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1223378730852814879)** (2 messages): 

- **Diving into Vector Databases with Qdrant**: A posted tutorial explores the integration of **Qdrant** with **LangChain** for use in vector databases, offering insights for local, server (Docker), cloud, and Groq implementations. The resource includes a [YouTube video](https://youtu.be/JSKZYgARffg) titled "Langchain + Qdrant Local | Server (Docker) | Cloud | Groq | Tutorial".

- **Conversational Taskbots with LangGraph**: A Jupyter notebook tutorial was shared that demonstrates how to use **LangGraph** and a *Chain of Tasks* promoting technique to build **Conversational Taskbots**. Accompanying the tutorial is a [LinkedIn post](https://www.linkedin.com/posts/prasadt_building-an-hrassistant-llm-taskbot-using-activity-7179347232788262913-6iBC) that provides more details.

**Link mentioned**: <a href="https://youtu.be/JSKZYgARffg">Langchain + Qdrant Local | Server (Docker) | Cloud | Groq | Tutorial</a>: Do you want to learn a production grade vector database for your Langchain applications? Let&#39;s delve into the world of vector databases with Qdrant. Qdrant i...

  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1223173196065542144)** (24 messages🔥): 

- **Loading Model Hyperparameters Error**: A user faced an issue running `./server -m all-MiniLM-L6-v2-f32.gguf --embedding` with an error about missing `bert.context_length`. No resolution to the error was discussed in the available messages.
- **Instabilities with Llamafile Execution**: Users discussed instability when running **llamafile**, with one user expressing it worked inconsistently. Another mentioned plans to investigate these instabilities the following week.
- **Excitement Over Llamafile v0.7 Release**: A new release of **llamafile v0.7** was [announced on GitHub](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.7), which boasts improved performance and accuracy. Additionally, a [blog post](https://justine.lol/matmul/) on **matmul** received positive reactions for its content and timing of release, right before April Fool's Day.
- **Prompt Template Queries for Llamafile**: A user enquired about the correct prompt templating to use in the web UI when running **llamafile** without a model, using **openchat 3.5 0106**. They shared a template and raised questions regarding the input fields and variables, but no direct answers were provided in the available messages.
- **Benchmarking Code Shared for Matmul**: **jartine** shared a Python code snippet that benchmarks numpy's matmul against a custom implementation, responding to a query about the surprising efficiency found in revising NumPy's approach which does not utilize threading.


**Link mentioned**: <a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.7">Release llamafile v0.7 · Mozilla-Ocho/llamafile</a>: llamafile lets you distribute and run LLMs with a single file  This release improves the performance and accuracy of both CPU and GPU computations in addition to security.  tinyBLAS now gives outpu...

  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1223393728576753755)** (9 messages🔥): 

- **Scout Law Guides Chatbot Responses**: A user customized **Claude 3 Haiku** to incorporate the Scout Law into its conversations, resulting in playful and honest answers, such as *"A door is not a door when it's ajar!"* following the Scout Law to be trustworthy.
- **Chatbot's Talkative Nature by Design**: The chatbot's verbosity was intentional, adhering to a system prompt directing it to be a *friendly, helpful assistant* and include one element of the Scout Law in each response.
- **The Shell of Trustworthiness**: In line with the Scout Law theme, the bot likened limpets to being trustworthy due to their protective shells, demonstrating its capacity to apply the Scout Law in creative ways.
- **Clarity Through Enumerating Questions**: The user invoked a system prompt to make the chatbot generate clarifying questions first instead of giving direct answers, which could lead to a more thoughtful approach to problem-solving.
- **Troubleshooting Reinstallation Issues**: A user facing a `FileNotFoundError` was suggested to redo the installation of `llm`, which another confirmed as a necessary step they had to take recently due to similar circumstances.
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1223239082419818592)** (7 messages): 

- **Introducing Jamba's Groundbreaking SSM-Transformer**: [AI21 Labs introduces Jamba](https://www.ai21.com/jamba), a fusion of **Structured State Space model (SSM)** and **Transformer** architectures aimed at overcoming limitations of the traditional Transformer models. The [model is available for testing on Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1).

- **Seeking Vibe Checks on Novel LLMs**: A member inquired about existing assessments or "vibe checks" for the **novel LLM architectures** discussed in the group.

- **BitNet's Robust Reproduction**: Reproduction of the [BitNet b1.58 model](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) has achieved comparable performance with implementations adhering to practices in their [follow-up paper](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf), revealing promising results for the **RedPajama dataset**.

- **Understanding Nectar's Data Source Diversity**: The [Nectar dataset](https://huggingface.co/datasets/berkeley-nest/Nectar), developed with GPT-4-based ranking, pulls from a variety of chat prompt sources, such as **ShareGPT**, **Antropic/hh-rlhf**, and **Flan**.

- **Discussion on GPT's Contextual Understanding**: A member highlighted a scenario from the Nectar dataset where GPT seemingly provides guidance on making a gun, noting that **Starling may answer such questions** while other models might refuse. There's speculation that this may be an approach to avoid the refusals that other base models exhibit when confronted with similar queries.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">1bitLLM/bitnet_b1_58-3B · Hugging Face</a>: no description found</li><li><a href="https://www.ai21.com/jamba">Introducing Jamba</a>: A groundbreaking SSM-Transformer Open Model</li><li><a href="https://huggingface.co/datasets/berkeley-nest/Nectar">berkeley-nest/Nectar · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1223779776549097473)** (2 messages): 

- **Translation Comparison Tool Shared**: A member highlighted the availability of a tool for comparing translations at [Hugging Face's space](https://huggingface.co/spaces/cstr/compare_translations). The tool's web application allows users to quickly assess different translations.

- **Comet Scores Mentioned In Discussing Translation Quality**: The use of **comet scores** to evaluate translations was briefly mentioned, suggesting that translations are being scored using this metric for quality assessment.

**Link mentioned**: <a href="https://huggingface.co/spaces/cstr/compare_translations">Compare Translations - a Hugging Face Space by cstr</a>: no description found

  

---



**Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1131305311714672710/1224210017155026985)** (2 messages): 

- **Overcoming Catastrophic Forgetting in CIL**: A paper suggests that adapter tuning outperforms prompt-based methods in class-incremental learning (CIL) without the need for parameter expansion during each learning session. The approach also involves feature sampling from prototypes and estimating the semantic shift of old prototypes to improve the backbone's learning capacity. Read the full study [here](https://arxiv.org/abs/2403.19979).

- **Enhancing Open-source LLMs as Intelligent Agents**: A new paper addresses the performance gap between open-source Large Language Models (LLMs) and commercial models like ChatGPT and GPT-4, particularly in complex real-world tasks. The research explores task planning, long-term memory, and external tools leveraging capabilities through both data fine-tuning and prompt design enhancements for 7B and 13B LLMs. Details can be found [here](https://arxiv.org/abs/2403.19962).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19962">Enhancing the General Agent Capabilities of Low-Parameter LLMs through Tuning and Multi-Branch Reasoning</a>: Open-source pre-trained Large Language Models (LLMs) exhibit strong language understanding and generation capabilities, making them highly successful in a variety of tasks. However, when used as agent...</li><li><a href="https://arxiv.org/abs/2403.19979">Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer</a>: Class-incremental learning (CIL) aims to enable models to continuously learn new classes while overcoming catastrophic forgetting. The introduction of pre-trained models has brought new tuning paradig...
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1223536075192930364)** (2 messages): 

- **AI21's Transformer Innovation Revealed**: The Skunkworks AI community shared a [YouTube video](https://www.youtube.com/watch?v=HRnx0ZPxe64) introducing **Jamba**, AI21's groundbreaking SSM-Transformer model, which has been publicized as a leap in large language model design.

- **Databricks Sets New LLM Benchmark**: Another [YouTube video](https://www.youtube.com/watch?v=dqFvOqC43rQ) was shared, showcasing **DBRX**, Databricks' new open, general-purpose large language model that claims to set a new state-of-the-art across a range of standard benchmarks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=HRnx0ZPxe64">Introducing Jamba: AI21&#39;s Groundbreaking SSM-Transformer Model</a>: Introducing Jamba: AI21&#39;s Groundbreaking SSM-Transformer Modelhttps://www.ai21.com/blog/announcing-jamba#llm #ml #ai #deeplearning #largelanguagemodels #deep...</li><li><a href="https://www.youtube.com/watch?v=dqFvOqC43rQ">DBRX: A New State-of-the-Art Open LLM</a>: Introducing DBRX, an open, general-purpose LLM created by Databricks. Across a range of standard benchmarks, DBRX sets a new state-of-the-art for established...
</li>
</ul>

</div>
  

---




