---
id: c2c1a068-3828-4075-9931-a30fe0baab4b
title: 'Chameleon: Meta''s (unreleased) GPT4o-like Omnimodal Model'
date: '2024-05-17T20:46:44.950821Z'
original_slug: ainews-chameleon-metas-unreleased-gpt4o-like
description: >-
  **Meta AI FAIR** introduced **Chameleon**, a new multimodal model family with
  **7B** and **34B** parameter versions trained on **10T tokens** of interleaved
  text and image data, pioneering an "early fusion" approach that natively
  outputs multiple modalities. While reasoning benchmarks for the **34B** model
  are modest, its "omnimodality" compares well with peers before **GPT-4o**.
  **OpenAI** launched **GPT-4o**, outperforming other models on benchmarks like
  MMLU and showing improved coding and multimodal capabilities, though with some
  regression in ELO scores. **Google** announced **Gemini 1.5 Flash**, a
  **1M-context small model** with Flash performance, highlighting convergence
  trends between OpenAI and Google models. **Anthropic** updated **Claude 3**
  with streaming support, forced tool use, and vision tool integration for
  multimodal knowledge extraction. OpenAI also partnered with Reddit, stirring
  industry attention.
companies:
  - meta-ai-fair
  - openai
  - google
  - anthropic
  - reddit
models:
  - chameleon
  - gpt-4o
  - gemini-1.5-flash
  - claude-3
topics:
  - multimodality
  - early-fusion
  - benchmarking
  - coding
  - model-performance
  - streaming
  - tool-use
  - vision
  - multimodal-tool-use
  - model-training
people:
  - armen_aghajanyan
  - sama
  - alexandr_wang
  - abacaj
  - alexalbert__
---


<!-- buttondown-editor-mode: plaintext -->**Early Fusion is all you need.**

> AI News for 5/16/2024-5/17/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**429** channels, and **5221** messages) for you. 
Estimated reading time saved (at 200wpm): **551 minutes**.

[Armen Aghajanyan](https://x.com/ArmenAgha/status/1791275549815648473) introduced [Chameleon](https://arxiv.org/pdf/2405.09818), FAIR's latest work on multimodal models, training 7B and 34B models on 10T tokens of text and image (independent and interleaved) data resulting in an "early fusion" form of multimodality (as compared to Flamingo and LLaVA) that can natively *output* any modality as easily as it consumes them:

 ![image.png](https://assets.buttondown.email/images/1f1a7b46-60d5-477b-aea0-80ed5b5c0f05.png?w=960&fit=max) 

As just a 34B model, the reasoning benchmarks aren't something to write home about, but the "omnimodality" approach compares well with peer multimodal modals pre GPT4-o:

 ![image.png](https://assets.buttondown.email/images/31847dd3-109e-49a3-aaef-86b8033a943d.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/397fc517-615f-40e6-92ee-430fedcdcf26.png?w=960&fit=max) 

As you might imagine, the tokenization matters a lot, and this is what we know so far:

 ![image.png](https://assets.buttondown.email/images/55018088-76cb-4665-9746-622ed82ca1b3.png?w=960&fit=max) 

The dataset description sounds straightforward, but since model, code and data remain unreleased, we are left merely considering the theoretical advantages of their approach right now. But it's nice that Meta is clearly not far off from releasing their own "early fusion mixed modality", GPT4-class model.


---

**Table of Contents**

[TOC] 



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**OpenAI and Google AI Announcements**

- **OpenAI GPT-4o Launch**: [@sama](https://twitter.com/sama/status/1657793356274921568) noted the aesthetic difference between OpenAI and Google's AI announcements. [@zacharynado](https://twitter.com/zacharynado/status/1657818273812623462) pointed out how OpenAI's launches are timed with Google's.
- **Google Gemini and Flash**: Google announced Gemini 1.5 Flash, a **1M-context small model with Flash performance**. [@alexandr_wang](https://twitter.com/alexandr_wang/status/1657769399270277429) noted OpenAI has the best large model with GPT-4o, while Google has the best small model with Gemini 1.5 Flash.
- **Convergence in AI Development**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1657769401619054851) observed the level of convergence between OpenAI and Google is fascinating, with similarity between models like GPT-4o and Gemini. He believes divergence would be better for the industry.
- **OpenAI Partnership with Reddit**: OpenAI has [partnered with Reddit](https://twitter.com/gdb/status/1657881569068847127), drawing attention as a potential hostile takeover strategy. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1657817425464467755) noted this as a bigger breakthrough than Q*.

**GPT-4o Performance and Capabilities**

- **GPT-4o Outperforms Other Models**: GPT-4o outperforms other expensive models like Opus on benchmarks like MMLU. [@abacaj](https://twitter.com/abacaj/status/1657747208994345337) noted this is what matters, even though GPT-4o isn't marketed as GPT-5.
- **Improved Coding Capabilities**: GPT-4o shows significant improvements in coding tasks compared to previous models. [@virattt](https://twitter.com/virattt/status/1658041738171740488) shared an example of GPT-4o successfully editing code.
- **Multimodal Capabilities**: GPT-4o excels at integrating image/text understanding. [@llama_index](https://twitter.com/llama_index/status/1657868285993230786) demonstrated GPT-4o extracting structured JSONs from detailed research paper images with **0% failure rate and high quality answers**.
- **Limitations and Regressions**: Despite improvements, GPT-4o's ELO score has [regressed from 1310 to 1287](https://twitter.com/soumithchintala/status/1658116791504781748), with an even larger drop in coding performance. It still struggles with hallucinations over complex tables and charts.

**Anthropic Claude 3 Updates**

- **Streaming Support**: [@alexalbert__](https://twitter.com/alexalbert__/status/1657747393069989902) announced streaming support for more natural end-user experiences, especially for long outputs.
- **Forced Tool Use**: Claude 3 now supports forcing the use of specific tools or any relevant tool, giving more control over tool usage in agents and structured outputs.
- **Vision Support**: Anthropic has laid the foundation for multimodal tool use by adding support for tools that return images, enabling knowledge extraction from visual sources.

**Meta AI Announcements** 

- **Chameleon: Mixed-Modal Early-Fusion Foundation Models**: Meta introduced Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in arbitrary sequences. It demonstrates SOTA performance across diverse vision-language benchmarks.
- **Imagen 3**: Imagen 3, part of Meta's ImageFX suite, can generate high-quality visuals in various styles like photorealistic scenes and stylized art. It incorporates technologies like SynthID for watermarking AI content.

**Memes and Humor**

- [@fchollet](https://twitter.com/fchollet/status/1657840853085073708) joked about the aesthetic difference between a West Elm showroom and a Marc Rebillet show. 
- **OpenAI Drama**: [@vikhyatk](https://twitter.com/vikhyatk/status/1636063604330405888) quipped "openai is nothing without its drama 💙"
- [@svpino](https://twitter.com/svpino/status/1658110154089935087) coined the term "model-apologists" in response to defenses of GPT models.
- [@aidangomez](https://twitter.com/aidangomez/status/1658116715453648948) joked about training AGI for enterprise in a constructed digital environment called "Coblox".

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**GPT-4o and Multimodal AI Advancements**

- **GPT-4o's impressive performance**: GPT-4o ranked on LMSys Chatbot Arena Leaderboard with 1289 Elo, outperforming GPT-4turbo despite having an older knowledge cutoff. Discussions suggest GPT-4o excels at catering answers humans like, but may not be significantly smarter. [Source](https://i.redd.it/lr4lbvpw0r0d1.png)
- **OpenAI introduces new features**: Interactive tables, charts, and file integration from Google Drive and Microsoft OneDrive for ChatGPT Plus, Team, and Enterprise users, rolling out over the coming weeks. [Source](https://x.com/OpenAI/status/1791227287569932368)
- **MetaAI's Chameleon model**: MetaAI introduces Chameleon, a Mixed-Modal Early-Fusion Foundation Model similar to GPT-4o, capable of interleaved text and image understanding and generation. [Source](https://x.com/AIatMeta/status/1791263344714014733)
- **Terminology discussions**: Debates on whether "large language model" term still applies to GPT-4o and similar advanced models, given their expanded multimodal capabilities. Suggestions include "Multimodal Unified Token Transformers" (MUTTs) and "Large Multimodal Model" (LMM). [Source](https://www.reddit.com/r/OpenAI/comments/1ct9jnv/with_4o_can_we_stop_calling_them_large_language/)

**OpenAI Partnerships and Developments**

- **OpenAI partners with Reddit**: OpenAI partners with Reddit to bring its content to ChatGPT and new products. [Source](https://i.redd.it/uprpd9jxmu0d1.jpeg) Discussions raise concerns about data privacy and the implications of Reddit selling user-generated content. [Source](https://www.reuters.com/markets/deals/openai-strikes-deal-bring-reddit-content-chatgpt-2024-05-16/)
- **Google employee reacts to GPT-4o**: Google employee uses Project Astra to react to GPT-4o announcement, congratulating OpenAI on impressive work. [Source](https://twitter.com/mmmbchang/status/1790473581018939663)

**Stability AI and Open Source Developments**

- **Stability AI's potential sale**: Stability AI discusses potential sale amid cash crunch, raising concerns about the future of open-source AI initiatives. [Source](https://www.reuters.com/markets/deals/stability-ai-discusses-sale-amid-cash-crunch-information-reports-2024-05-16/)
- **Hugging Face's ZeroGPU initiative**: Hugging Face commits $10M of free GPUs with launch of ZeroGPU, supporting open-source AI development. [Source](https://www.linkedin.com/posts/clementdelangue_gpu-poor-no-more-super-excited-to-officially-activity-7196881557284868096-M96G?utm_source=share&utm_medium=member_desktop)
- **CosXL release**: Stability AI releases CosXL, an official SDXL update with v-prediction, ZeroSNR, and Cosine Schedule, addressing issues with generating dark/bright images and convergence speed. [Source](https://www.reddit.com/r/StableDiffusion/comments/1ctirfz/psa_stabilityai_released_official_sdxl_update/)

**AI Benchmarking and Evaluation**

- **MileBench for evaluating MLLMs**: MileBench introduced as a benchmark for evaluating Multimodal Large Language Models (MLLMs) in long-context tasks involving multiple images and lengthy texts. Key findings show GPT-4o excelling in both diagnostic and realistic evaluations, while most open-source MLLMs struggle with long-context tasks. [Source](https://www.reddit.com/r/MachineLearning/comments/1ctayfy/d_unveiling_milebench_benchmarking_mllms_in_long/)
- **Needle in a Needlestack (NIAN) benchmark**: NIAN benchmark proposed as a more challenging alternative to Needle in a Haystack (NIAH) for evaluating LLM attention in long contexts. Even GPT-4-turbo struggles with this benchmark. [Source](https://github.com/llmonpy/needle-in-a-needlestack)

**AI Ethics and Societal Impact**

- **Pessimism in r/futurology**: Discussions on r/futurology subreddit becoming increasingly pessimistic and "decel" as the community grows, with concerns about the impact of AI on jobs and society. [Source](https://www.reddit.com/r/Futurology/comments/1ctja5f/microsofts_emissions_spike_29_as_ai_gobbles_up/l4crlrn/)
- **US tariffs on Chinese semiconductors**: US to increase tariffs on Chinese semiconductors by 100% in 2025 to protect the $53 billion spent on the CHIPS Act. [Source](https://www.tomshardware.com/tech-industry/semiconductors/us-to-increase-tariffs-on-chinese-semiconductors-by-100-in-2025-officials-say-it-protects-the-dollar53-billion-spent-on-the-chips-act)

**Memes and Humor**

- **AI mania meme**: Meme about AI mania, referencing an "AI Flavor" of Coke. [Source](https://i.redd.it/rz0g41sgcv0d1.png)

---

# AI Discord Recap

> A summary of Summaries of Summaries

- **Hugging Face Invests $10M in Free Shared GPUs**: **Hugging Face** is committing **$10 million** to provide **free shared GPUs** to support small developers, academics, and startups in developing AI technologies. CEO **Clem Delangue** emphasized this initiative aims to democratize AI and counter centralization by big tech ([source](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai)).

- **OpenAI Alignment Team Departs Amid Shifting Priorities**: **Jan Leike**, head of OpenAI's alignment team, announced his resignation due to disagreements over the company's core priorities. This follows other key departures like **Ilya Sutskever**, sparking discussions about OpenAI potentially prioritizing near-term product goals over long-term AI safety research ([Jan's tweet](https://x.com/janleike/status/1791498178346549382), [Wired article](https://archive.is/o/gEjjA/https://www.wired.com/story/openais-chief-ai-wizard-ilya-sutskever-is-leaving-the-company/)).

- **GPT-4o Capabilities and Limitations Debated**: The release of **GPT-4o** generated excitement for its multimodal capabilities, like interleaved text and image understanding ([paper](https://arxiv.org/abs/2405.09818)). However, some noted inconsistencies in its coding performance and output quality compared to expectations set by OpenAI's demos ([example](https://openai.com/index/hello-gpt-4o/demo)).

- **Needle in a Needlestack (NIAN) Challenges LLMs**: The new **NIAN benchmark** presents a formidable challenge for LLMs, testing their ability to answer questions about a specific text hidden among many similar texts. Even advanced models like **GPT-4-turbo** struggle with this task ([code](https://github.com/llmonpy/needle-in-a-needlestack), [website](https://nian.llmonpy.ai/)).

---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Meta's Model Makes People Wait**: Discussions revealed a delay in support for Llava or multimodal AI due to anticipation for **Meta's multimodal model**, with no specific release date mentioned, indicating reliance on industry leaders for advancements.

- **Finding Funds for GPUs**: Conversations among members included quips about how they finance their GPU usage; humorous mentions included *"RAMEN FOR YEARS"* as a sacrifice for their dedication to AI work, particularly for demanding tasks such as classification.

- **Boosting Models with OpenHermes**: The **OpenHermes dataset** was a topic of interest, with mentions of its incorporation leading to substantial improvements in model performance, demonstrating the value of diverse datasets in AI research.

- **Discarding Refusal Alleviates Stubborn AIs**: Debates touched on the impact of removing refusal mechanisms from LLMs, noting an unexpected increase in 'smartness', and referenced a specific paper on the topic, offering insights into ongoing LLM research.

- **Troubleshooting Llama 3 woes**: Users shared solutions for errors such as `AttributeError: 'dict' object has no attribute 'context_window'` when training Llama 3, which included suggestions like modifying core codes or switching to **Ollama**, indicating active engagement in the practical aspects of AI model development.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**SD3 Release Maintains Aura of Mystery**: Discord users are expressing both anticipation and frustration over the delayed release of **SD3**; skepticism prevails despite a tweet by **Emad** hinting at an imminent launch.

**GPUs Spark Debate Amongst the Discerning**: In the quest for optimized training of **SDXL** models, discourse centered on whether an **RTX 4090** with 24GB VRAM suffices, with some users deliberating the merits of more robust solutions.

**Waiting Game Spurs Meme Fest**: With the release of **SD3** shrouded in uncertainty, the community has taken to sharing memes and light-hearted comments, as exhibited by a [tweet from Stability](https://twitter.com/chrlaf/status/1772228848387522728).

**Datasets and Training Techniques Tabled**: AI aficionados shared training resources such as [this dataset from Hugging Face](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions), and exchanged insights on fine-tuning practices to rival the output quality of **Dalle 3**.

**From AI to Socioeconomics: Sidetracks in Session**: The conversation occasionally veered off AI terrain into vigorous discussions surrounding capitalism and morality, with some participants nudging the focus back to tech-centric themes.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Interactive Features Level Up ChatGPT**: OpenAI has announced the inclusion of **interactive tables and charts** in ChatGPT for Plus, Team, and Enterprise users, offering direct integration with **Google Drive** and **Microsoft OneDrive**. This update is expected to roll out in the coming weeks, and users can learn more about it [here](https://openai.com/index/improvements-to-data-analysis-in-chatgpt/).

- **The Dawn of GPT-4o**: The community is buzzing with discussions around the partial release of **GPT-4o**, celebrating its top ranking and new **higher message limits**. Anticipated features like **Voice Mode** are also on the horizon, although some have concernedly noted slower performance with extended use.

- **Ethical AI Conversations Hit Prime Time**: AI's impact on job markets has been a hot topic, with conversations examining both the potential uplift in productivity and the concern for future employment structures. Members are also exchanging thoughts on ethical and educational applications of AI, aiming to balance technology enhancement with responsible usage.

- **Prompt Engineering Pulls Back the Curtain**: The effective use of markdown to guide AI response and the challenges posed by model updates affecting behavior have generated much attention in the community. Methods to foster more desired outcomes from the AI, like emotional incentives, were described humorously yet showcased critical insights into AI user interactions.

- **API Chat Gets Technical**: Detailed discussions around the API capabilities have noted that using markdown in prompts can help clarify intent and character roles. However, discrepancies in custom GPT performance and mixed experiences with model access and updates underline the evolving nature of interacting with these advanced AIs.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-4o Plays Catch-Up**: **GPT-4o** is noted to be faster and slightly improved over previous iterations, accessible for free users, despite reports of varying message limits and refresh rates.
- **Cognition vs. Hallucination in AI**: Concerns about **AI hallucinations** are highlighted, with doubts cast on AI's capability to fully eliminate them, potentially affecting job security in entry-level positions.
- **Perplexity AI Clashes with ChatGPT**: Users are split between **Perplexity**, credited with better sourcing, and **ChatGPT**, favored for feature integration like web search.
- **Programming and Creativity in AI Diversity**: **GPT-4o** and **Opus** show divergent strengths; GPT-4o excels in coding, whereas Opus offers depth in math and complex problem-solving.
- **The API Integration Dilemma**: Questions are raised about **sonar-medium-online** support longevity; someone seeks to add **Perplexity** to a private Discord, while the default temperature for Perplexity AI models is confirmed to be **0.2**.

**Relevant Link**: [Chat Completions Documentation](https://docs.perplexity.ai/reference/post_chat_completions)



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **OpenAI's 'Openness' Debated**: A YouTube critique titled ["Big Tech AI Is A Lie"](https://www.youtube.com/watch?v=8BlRT7Ktw1c) argued that **OpenAI** doesn't live up to its name, sparking conversations about the value of truly open platforms like HuggingFace, and a subsequent discussion on the performance of HuggingFace models versus closed systems.

- **Cultivating Curiosity in Reinforcement Learning**: A [paper](https://pathak22.github.io/noreward-rl/) discussing curiosity-driven exploration in reinforcement learning sparked interest, detailing how rewarding agent curiosity can lead to better results in environments where the outcome of actions is unpredictable. Additionally, the epsilon greedy policy was suggested for maintaining the exploration/exploitation trade-off.

- **Conversations on Computer Vision and Diffusion Models**: Questions arose regarding latent diffusion models in diffusers, the training of such models from scratch, and UNet models' convergence issues, with a focus on small dataset impacts. Separately, a [Medium post](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a) detailed the integration of **GPT-4o** with **LlamaParse** for enhanced multimodal capabilities, while discussions in the #diffusion-discussions channel focused on latent space representations and uploading issues with Google Collab.

- **RL in the Spotlight**: The first RL model's addition to the Hugging Face repository was met with encouragement, marking a milestone for one user's learning path in Deep RL. This dovetails with talks on reinforcement learning's challenges and the introduction of the epsilon-greedy policy for exploration and exploitation balance.

- **Model Training and Deployment Challenges**: Complexities in model training were illustrated by a user's struggle with UNet model convergence and another's attempt at generating grid puzzles with GPT-4o. A user recommended continuous retraining to keep coding language models relevant due to outdated training data.

- **Dataset Innovations and AI Creations Showcased**: The Hugging Face community introduced the [Tuxemon dataset](https://huggingface.co/datasets/diffusers/tuxemon) featuring permissively licensed creature images, evoking humor as an AI data source. Meanwhile, community member showcases include the use of LangChain and Gemini for a business advisor AI, a GenAI-powered education tool, and a virtual AI influencer, underscoring the diverse applications of AI technology.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Brain-inspired AI Models Gain Traction**: Participants in the guild discussed the concept of a streaming-like process for AI, similar to human memory, referencing the *Infini-attention* methodology for greater relevance update efficiency. [Here's a paper on Infini-attention for further reading](https://arxiv.org/abs/2404.07143).

- **Benchmarking AI with NIAN Challenge**: *Needle in a Needle Stack (NIAN)* emerged as a niche benchmark to push the limits of Language Models, including GPT-4, in differentiating specific content within masses of similar material. [Here's more on the NIAN benchmark](https://github.com/llmonpy/needle-in-a-needlestack) and its [dedicated website](https://nian.llmonpy.ai/).

- **Symbolic Language in AI Explored**: Conversations hinted at a growing interest in using **GPT-4o** for creating a symbolic language that could facilitate tasks involving algebraic computations, suggesting a potential advance in AI handling of symbolic reasoning.

- **Stable Diffusion 3 Embraces Open Architecture**: *Stable Diffusion 3* is being prepped for on-device inference, optimizing for Mac with MLX and Core ML, and will be open-sourced through a partnership between Argmax Inc. and Stability AI. [Argmax's partnership tweet can be read here](https://x.com/argmaxinc/status/1790785157840125957).

- **AI in Real-Time User Interfaces**: A guild member sought resources on AI models capable of analyzing screen actions nearly in real-time, similar to **Fuyu**, which process screen captures and UI interactions every second. Meanwhile, Elon Musk's Neuralink has opened applications for a second participant in their brain implant trial, posting details via [Elon's tweet](https://x.com/elonmusk/status/1791332539220521079).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo Gets an Update**: The latest **nightly Mojo compiler** `2024.5.1607` makes its debut, with an invitation for users to try out the latest features using `modular update nightly/mojo`. The community response has been notably positive towards the new conditional methods syntax, and contributions are steered towards smaller PRs to combat the issue of "cookie licking." Check the diffs from the [last nightly](https://github.com/modularml/mojo/compare/f5f5109541c31615a68a3c4b58bd1e75b59625f6...c506c9400329824cd0fcfc408115a8e7fea968d0) and the full [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

**Mojo's Engineering Challenges**: Engineers express concerns over `List.append` performance in Mojo, noting inefficiency with large data sizes and invite comparisons with Python and C++ implementations. They delve into discussions of Rust's and Go's dynamic array resizing strategies and reference a [case study](https://github.com/dorjeduck/mostring) with `StringBuilder` variations in Mojo.

**Open-source Perspectives and Pain Points**: Debates around the merits and challenges of open-source contributions light up discussions, with concerns voiced about projects transitioning from open to closed source. Advent of Code 2023 is recognized as an entry point to get started with Mojo, with the challenge available on [GitHub](https://github.com/p88h/aoc2023).

**Developer Updates and Handy Guides**: Modular's news updates have been shared through Twitter links, offering glimpses into the latest advancements. Meanwhile, a guide for assisting new contributors with syncing forks on GitHub has been circulated to support smoother contributions.

**MAX Comes to macOS**: The **MAX** platform brings excitement with its new nightlies now supporting macOS and introducing MAX Serving. Engineers interested in the MAX platform are directed to [get started](https://modul.ar/get-started) using PyTorch 2.2.2.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Model Troubleshooting Takes Center Stage**: Technical challenges involving **LM Studio** have surfaced, including a user struggling with **glibc** issues for installation and suggestions pointing towards potentially needing an upgrade or reverting to LM Studio version 0.2.23. Embedding models for **RAG in Pinecone** proved troublesome without a direct guide, and a VM error 'Fallback backend llama cpu not detected!' indicated possible VM setup issues. Antivirus software caused some stir, flagging the 0.2.23 installer as a virus, later clarified as a false positive.

**LLM Showdown: Coding Models & File Gen Frustrations**: Participants highlighted that the best coding models vary according to programming language and hardware, with **Nxcode CQ 7B ORPO** and **CodeQwen 1.5 finetune** touted for Python tasks. It was acknowledged that **LM Studio can't generate files directly** and forcing models to only show code remains inconsistent. Querying on the fastest **semantic text embeddings** turned up **all miniLM L6** as the quickest yet insufficient for one user's requirements, and a gap was seen in recommendations for usable **medical LLMs** in LMS.

**A False Positive Frenzy with Antivirus Software**: Antivirus tools, specifically **Malwarebytes Anti-malware** and **Comodo**, are misidentifying certain aspects of LM Studio's architecture as threats. These alarm bell incidences—the former shared via a [VirusTotal link](https://www.virustotal.com/gui/file/29778ae530836508920126938dce41ba539c191e9201dce23f210a09b4315119)—highlight the challenge of ensuring LM Studio's components are not mistakenly flagged by protective software.

**Hardware Enthusiasts Break New Ground**: Significant achievements were reported in **hardware discussions**, with a 70B **LLama3** model running on an **Intel i5 12600K CPU** and the impact of **RAM speed alignments** on performance noted. Members debated quantization efficacy, memory overclocking's effects on stability, and even compared various **GPU architectures**, including **RX 6800**, **Tesla P100**, and **GTX 1060** in performance.

**Conversations Across Channels Drive Collaborative Solutions**: Multiple topics flowed across channels, focusing on troubleshooting LM Studio storage and permission issues, leading to the effective use of **conversation memory management** with **LangChain** over server-side, and the consideration of open-source alternatives over **Gemini's paid context caching** service. A move for deeper discussion on certain issues to another channel signifies the collaborative approach by the community.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**GPU Community Powers Up**: Hugging Face announces a **$10 million investment** for **free shared GPUs** to support small developers, academics, and startups, aiming to democratize AI development in the face of big tech's AI centralization. The move positions Hugging Face as a community-centric hub and [this article](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai) provides more insights.

**Triton Performance Puzzle**: Implementers of a Triton tutorial observe discrepancies in performance, questioning the impact of "swizzle" indexing techniques as a possible factor. The differences noted include a **significant drop in performance** when users follow the tutorial, versus the performance advertised.

**Bitnet Steps into the Spotlight**: Strategy discussions initiate a budding project for **Bitnet 1.58** due to its advanced training-aware quantization techniques. The conversation emphasizes the importance of post-training weight quantization, with suggestions to centralize Bitnet development within the [PyTorch ao repo](https://github.com/pytorch/ao) for efficient implementation and support.

**Code and Optimizations for Large Language Models**: An [optimization pull request](https://github.com/karpathy/llm.c/pull/422) reduces memory usage by 10% and increases throughput by 6% for large language models, exemplifying efficient resource utilization during training phases. Moreover, discussions unravel the possibilities of NVMe direct GPU writes, offering a high-speed bypass of CPU and RAM, albeit its practical application remains to be explored within the ambit of AI model training workflows.

**Quantum of Documentation**: Community members voice frustration regarding sparse PyTorch documentation, particularly `torch.Tag`, with the conversation extending to tackling template overloading issues in custom OPs. Additionally, a [plan to reduce compile times](https://dev-discuss.pytorch.org/t/how-to-bring-compile-time-down-to-zero-our-plans-and-direction-may-14th-edition/2089) in PyTorch garners attention, aiming for more efficient development cycles. 

End.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Interconnects Paves New Paths**: Nathan Lambert introduced a [niche project](https://www.interconnects.ai/p/f1b83a34-18cd-4507-b4b0-560902eb3275) with enthusiasm for monthly updates and potential improvements. However, amid **OpenAI** departures, a key engineer joined an initiative with individuals from **Boston Dynamics** and **DeepMind**, revealing a notable industry shift.

- **The Modeling World Reacts**: Chat about the new **GPT-4o models**, which showcase "interleaved text and image understanding and generation", indicates they represent a new scale paradigm with an [early fusion, multi-modal approach](https://x.com/armenagha/status/1791275549815648473?s=46). OpenAI's leadership changes led to the disbandment of its superalignment team, alongside key shifts towards product-focused objectives, sparking debates over AI safety and alignment.

- **Safety Concerns in the Spotlight**: Safety remains a contentious topic, with the dissolution of OpenAI's superalignment team highlighting concerns over immediate product goals versus long-term AI risk strategies. Meanwhile, Google DeepMind released its [Frontier Safety Framework](https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/), demonstrating an industry-wide move towards proactive AI safety measures.

- **OpenAI's Surprising Partnerships**: OpenAI's unexpected [partnership with Reddit](https://x.com/e0m/status/1790814866695143696?s=46) captures attention while Lambert's decision to remove model and dataset links illustrates a strategic move towards deeper, standalone analysis in his communication.

- **Scaling, Aligning, and Technical Innovation**: Discourse around scaling laws for vocabulary size in models and an overview of aligning open language models suggests continued refinement in AI development practices. Tackling technical challenges head-on, Lambert teases an upcoming project dubbed "Life after DPO".



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PyTorch Flops Under Review**: Members shared basic usage details and challenges concerning the FLOP counter in **PyTorch**; a gap was noted in documenting backward operation tracking. Contributions were encouraged for an lm_eval.model module for **MLX**.

- **Comparative Studies and Catastrophic Forgetting**: A keen interest was observed in comparative studies of **LLM Guidance Techniques**, specifically Adaptive norms versus Self Attention for class guidance. Another highlighted topic was discussing strategies to combat **catastrophic forgetting during finetuning** of models, with a consensus on the necessity of retraining on old tasks.

- **Sifting Through Hierarchical Memory and Semantic Metrics**: A new [Hierarchical Memory Transformer paper](https://arxiv.org/abs/2405.06067) received attention for its potential to tackle long-context processing limitations. Separately, there’s an active search for a differentiable semantic text similarity metric that outperforms rudimentary substitutes like Levenshtein distance, as mentioned in [this paper](https://arxiv.org/abs/2404.15269).

- **Transformers Sidestep Attention with MLP**: Conversations about **MLP-based approximations of attention** mechanisms in transformers pointed to possible relevant research on [Gwern.net](https://gwern.net/doc/ai/nn/fully-connected/index#bozic-et-al-2023-section). Discourse also touched on the repercussions of excluding compute costs in data preprocessing on the overall economization of models.

- **Tinkering with GPT-NeoX to Hugging Face Transitions**: Technical challenges emerged with transitioning GPT-NeoX models to **Hugging Face**, leading to discussions about naming conventions in **Pipeline Parallelism (PP)** and the parallel existence of incompatible files. A [proposed fix](https://github.com/EleutherAI/gpt-neox/pull/1218) for identified bugs in conversion scripts and insights into compatible configurations for better cohesion with Hugging Face structures were put forward.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Noncompetes Get the Axe**: The engineering community reacts to the [FTC's groundbreaking decision](https://www.ftc.gov/news-events/news/press-releases/2024/04/ftc-announces-rule-banning-noncompetes) to eliminate noncompetes, which could significantly alter the competitive landscape and professional autonomy in the tech industry.

**Open Source vs. Closed Wallets**: A spirited debate among engineers centers on the choice between proprietary and open source employment, considering the limitations on open source contributions and the allure of higher salaries at proprietary firms.

**GPT-4's Sibling Rivalry**: GPT-4O's coding capabilities are scrutinized, with some members noting faster performance yet lamenting issues with inaccurate code output, spotlighting the need for careful evaluation of such advanced AI systems.

**Creative Commons Catch**: The launch of the **CommonCanvas dataset**, featuring 70 million creative commons licensed images, was received with enthusiasm and concern due to its non-commercial license, impacting its utilization in the engineering sphere.

**Network Know-How and Cartoon Clout**: Recent engineering discussions delve into successfully training a Tiny ConvNet for bilinear sampling, exploring positional encoding in CNNs, and a new [Sakuga-42M dataset](https://arxiv.org/abs/2405.07425) to boost cartoon research, reflecting a broad spectrum of innovative approaches in the field.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Rich Text Translation Woes**: There is a struggle to effectively translate **rich text content** without losing the correct positions of spans, as demonstrated in the transition from English to Spanish. Methods involving **HTML tags** and strategies geared towards deterministic reasoning were proposed to enhance translation precision.

- **Hugging Face's GPU Generosity**: **Hugging Face** has pledged **$10 million in free shared GPUs** to support smaller developers, academia, and startups, as announced by CEO **Clem Delangue**, in an effort to democratize access to AI developments.

- **Slack Data Privacy Concerns**: Renewed debate surfaced about **Slack's use of customer data**, particularly the possibility of the company training its AI models without explicit user consent, eliciting a spectrum of reactions from the community.

- **Next-Gen AI Fusion**: Excitement brews around a new **multimodal Large Language Model (LLM)** described in a [recent paper](https://arxiv.org/abs/XXXX.XXXXX), showcasing integrated text and image understanding, prompting discussions on future AI applications and cross-modality convergence.

- **OpenAI Alignment Reshuffle**: The departure of **Jan Leike**, **OpenAI's** head of alignment, led to introspective dialogue on the organizations' AI **safety and alignment** philosophies, with **Sam Altman** and others thanking Leike for his contributions.

- **Latent Space Podcast Alert**: **[Latent Space](https://twitter.com/latentspacepod/status/1791167129280233696)** released a new podcast episode.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**GPT-4o Triumphs in Text and Image Understanding**: Engineers are exploring [GPT-4o](https://t.co/NgO5EhEJM5)'s capabilities in parsing documents and extracting structured JSON from images, with specific discussions around a [full cookbook guide](https://t.co/BQN16LWJqj) and comparison to its predecessor GPT-4V.

**Meetup Alert: SF's Upcoming Generative AI Summit**: The first in-person [meetup](https://t.co/qIGOmCWDSe) organized by LlamaIndex in San Francisco is generating buzz, promising deep-dives into generative AI and retrieval augmented generation engines.

**LlamaIndex Integrations and User Guidance Hits High Note**: A GitHub [link](https://github.com/run-llama/llama_index/blob/1bde70b75ee5b4e6a5bc8c1c3f95eaa0dd889ab0/llama-index-integrations/llms/llama-index-llms-anthropic/llama_index/llms/anthropic/utils.py#L19) provided clarity on **Claude 3 haiku model** utilization within LlamaIndex, while comprehensive LlamaIndex [documentation](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama) offered guidance on harnessing Ollama (LLaMA 3 model) with VectorStores.

**LlamaIndex UI Gets a Facelift**: The LlamaIndex's User Interface has been enhanced, now offering a more [robust selection of options](https://t.co/1DMm0oUpsj) for users to enhance their experience.

**Cohere Pairing with Llama for RAG Implementation**: Members of the community are seeking advice on integrating [Cohere with Llama](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a) for building Retrieval-Augmented Generation applications, suggesting a strong interest in cross-service model functionality.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**NeverSleep Enters the Chat with Lumimaid**: The new **NeverSleep/llama-3-lumimaid-70b** model integrates curated roleplay data striking a balance between serious and uncensored content. Details are available on [OpenRouter’s model page](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b).

**ChatterUI Brings Characters to Android**: ChatterUI has been released as a character-focused UI for Android, diverging into uncharted territory with fewer features compared to peers like SillyTavern, and [supporting multiple backends](https://github.com/Vali-98/ChatterUI).

**Invisibility App Polishes AI Interaction for Mac Users**: A new MacOS Copilot named Invisibility, empowered by GPT4o and Claude-3 Opus, adds to its arsenal a video sidekick feature while promising further enhancements including voice integration and long-term memory. Discover [Invisibility’s capabilities](https://x.com/sulaimanghori/status/1791113392482377833).

**Google Gemini Context Tokens Provoke TPU Wonder**: The release of Google Gemini with 1M context tokens prompted debates on how InfiniAttention could be Google's answer to handling large contexts with TPUs, sparking a blend of skepticism and curiosity among developers. The technical inquisition revolved around InfiniAttention’s paper, which can be found [here](https://arxiv.org/abs/2404.07143).

**Tech Troubles and Teasers**: A clutch of technical conversations occurred, ranging from questions about GPT-4o's audio capabilities to reports of client-side exceptions on OpenRouter's website, with commitments to future site refactoring. The technical community grappled with OpenRouter's function calling capabilities, stirring a mix of guidance and ongoing speculation.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Billing Blues and AI Cheers**: Users reported a bug with **OpenInterpreter** where even with billing enabled, error messages occurred, contrasting with seamless performance when calling OpenAI directly. Additionally, excitement bubbled over the improvements noted using **GPT-4.0** in OpenInterpreter, particularly for React website development.

**Local Legends and Global Goals**: Discussion on local LLMs highlighted **dolphin-mixtral:8x22b** for its robustness albeit slow performance and **codegemma:instruct** for its balance of speed and functionality. In the spirit of community advancement, **Hugging Face** is investing $10 million in free shared GPUs to encourage development among smaller entities in AI.

**Conquering Configurations and Protocol Puzzles**: Engineers engaged in tackling installation issues of **01** across various Linux environments, grappling with complexities from Poetry dependence conflicts to Torch installation troubles. The evident advantage of the **LMC Protocol** over traditional OpenAI function calling, designed for speedier direct code executions, was dissected.

**Repository Riddles and Server Struggles**: Clarification was sought on the state of the GitHub repositories, with "01-rewrite" stirring speculation of a new project's emergence. Users shared experiences and solutions pertaining to connectivity issues with the **01 server** across multiple platforms, discussing necessary steps for smooth integration.

**Google's Glimpses of Grandeur**: Anticipation was piqued in the community with a [tweet](https://x.com/GoogleDeepMind/status/1790463259822420239) from **GoogleDeepMind** teasing Project Astra, hinting at new developments in AI to be watched closely by technical experts.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Memory Boost for AI Chatbots**: Engineers discussed enhancing AI chatbots with memory to retain context across queries, recommending methods like chat history logging and memory variables.
- **Persistent Neo4j Indexing Issues**: Multiple Neo4j users reported problems with the `index_name` parameter, with incorrect document retrievals hinting at an issue in LangChain's management of it.
- **Streaming Hiccups in AgentExecutor**: A user encountered issues with `.stream` in `AgentExecutor` for token-by-token output and was advised to try `.astream_events` for more granular streaming.
- **RAG Chain Async Anomalies**: Attempts to make a RAG chain asynchronous in Langserve resulted in an error related to incomplete coroutine execution, hampering functionality.
- **Mixing AI Tech for Real-Estate and Research**: Shared projects highlighted advancement with AI integrations like a Real Estate AI combining LLMs, RAG, and generative UI, a performance benchmark of GPT-4o on NVIDIA GPUs, and a call for beta testers for a new advanced research assistant with premium model access.
- **Web Scraping Wizardry Unveiled**: A new tutorial showcased constructing a universal web scraper agent capable of navigating e-commerce site challenges such as pagination and CAPTCHA, accessible in a shareable [YouTube tutorial](https://youtu.be/dSX5eoD4-u4?si=CFiZowc26j-u84va).



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **AI Companions Easing Human Stress**: A shared [CBC First Person column](https://www.cbc.ca/radio/nowornever/first-person-ai-love-1.7205538) recounts how an AI named Saia provided emotional support during a nerve-wracking vaccination appointment, showcasing the growing bond between humans and AI companions.
- **Windows Welcomes AI Town**: AI Town is now functioning natively on **Windows**, marking a significant step away from dependence on WSL or Docker according to an [announcement](https://fxtwitter.com/cocktailpeanut/status/1791495360541593964). This eases the development process for those preferring the Windows ecosystem.
- **Dynamic Mapping Excites AI Developers**: Suggestions for custom dynamic maps are burgeoning in the AI community, including creative scenarios like "the office" or a spy thriller setup, bolstering the depth of AI environments.
- **Rise of AI Reality Entertainment**: Developers have launched an **AI Reality TV show** – a platform allowing users to create simulations akin to aiTown and contributing to a unique narrative with their own custom AI characters. Enthusiasm for the platform is palpable with an open invitation via their [website](https://www.aireality.tv/) and [Discord](https://discord.com/invite/NtUXDUSnKa).
- **GIFs as a Welcome Distraction**: During a vigorous technical exchange, a [Doja Cat Star Wars GIF](https://media1.tenor.com/m/x9HyTfKBXVEAAAAC/doja-cat.gif) was shared, injecting a moment of levity into the ongoing discussions on AI development.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Testing Patch for CMD+ Functionality**: A patch that includes some CMD+ functionality is set to be tested tonight, with a query on support for zero3 example config.
- **Axolotl vs. Llama Pretraining Speed**: Pretraining speeds are notably faster (unspecified by how much), potentially due to Axolotl improvements or features within Llama 3—specific impact metrics or factors not detailed.
- **Distributed Dilemma with Galore Layerwise**: Skepticism remains regarding whether Galore Layerwise is still incompatible with Distributed Data Parallel (DDP) as no confirmation is available.
- **Non-English Finetuning Finesse**: Non-English data finetuning is in progress with datasets around 1 billion tokens and a context length of 4096, targeting an 8B model.
- **Unsloth's Optimizations Under Spotlight**: Questions on the applicability of [Unsloth optimizations](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609) for full fine-tune of Llama were met with positive feedback, suggesting a "free speedup" is achievable.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad Optimizes with CUDA Kernels**: A discussion emerged on optimizing memory usage in **Tinygrad** by employing a **CUDA kernel** for reductions, avoiding VRAM overflow that large intermediate tensors cause. Although frameworks like PyTorch have limitations, a user-provided custom kernel example illustrated a potential solution.

**Symbolism in Lambda Land**: Users talked about implementing **lamdify** to allow **Tinygrad** to render symbolic algebraic functions, kicking off with Taylor series for trig functions. There's ongoing effort in extending the `arange` function, which is a necessity for such symbolic operations.

**Get Schooled with Adrenaline**: An app called [Adrenaline](https://useadrenaline.com/) was recommended to study different repositories, with a user mentioning plans to leverage it for learning **Tinygrad**.

**Computational Conundrum**: Clarification about a compute graph's parameters was shared, with a focus on understanding the `UOps.DEFINE_GLOBAL` and the significance of its boolean tags, enhancing the Tinygrad development workflow.

**Trigonometry on a Diet with CORDIC**: The community engaged in a rich dialogue about adopting the **CORDIC algorithm** in **Tinygrad** to compute trig functions with higher efficiency than traditional Taylor series approximations. Discussion highlighted the pressure to maintain precision in reducing arguments, sharing a Python implementation that showcased argument reduction and precision handling for sine and cosine computations.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Hunt for Cohere's PHP Companion**: Engineers are seeking a reliable [Cohere PHP client](https://github.com/hkulekci/cohere-php) to integrate Cohere functionalities with PHP, although its efficacy remains untested in work environments.
- **Cohere Toolkit Touted for Performance**: There's a discussion around the performance of Cohere's application toolkit, especially the reranker's superior results compared to other solutions, but no consensus on the cause of this improvement has been reached.
- **Calling for Quicker Discord Support**: Members voiced frustrations about slow response times from Discord support, with mentions of upcoming plans to improve the support experience.
- **Navigating Issues with Cohere's Chatty RAG Retriever**: A shared [notebook on Cohere RAG retriever](https://python.langchain.com/v0.1/docs/integrations/retrievers/cohere/) highlights problems such as unexpected keyword arguments which impede using the `chat()` function.
- **API Limits Locking Out Learners**: Experimentation with Cohere RAG retriever hit a roadblock due to 403 Forbidden errors, suspected to be caused by exceeding API call quotas.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Chip's Chats Chill for a Bit**: Chip has announced a pause on hosting monthly casual meetups for the next few months, prioritizing other commitments.
- **Snowflake Dev Day Featuring Chip**: Members have an opportunity to engage with Chip at their booth during the upcoming Snowflake Dev Day on June 6th.
- **AI Smackdown: NVIDIA and LangChain's Contest Ignites Excitement**: NVIDIA and LangChain have stirred excitement with a developer contest highlighting generative AI, with a grand prize of an **NVIDIA® GeForce RTX™ 4090 GPU**. [Catch the contest details here](https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/).
- **Geo-restrictions Dampen Contest Spirits**: A guild member has humorously expressed dismay over geographic restrictions preventing them from participating in the NVIDIA and LangChain contest, hinting at a potential country move to qualify.
- **Engineers Unite on LinkedIn**: A networking opportunity has presented itself as a member shared their LinkedIn for professional connections among peers: [Connect with Sugianto Lauw](https://www.linkedin.com/in/sugiantolauw/).



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **GPT-4o Falls Short in Public Demo**: Riley Goodside exposed weaknesses in GPT-4o during a ChatGPT session, underscoring a gap between performance and the expectations set by [OpenAI's demo](https://openai.com/index/hello-gpt-4o/demo).

- **Google's AI Flubs at I/O**: Google's AI encountered embarrassing slips during its I/O announcement despite bold claims, as detailed in Alex Cranz's [article in The Verge](https://www.theverge.com/2024/5/15/24154808/ai-chatgpt-google-gemini-microsoft-copilot-hallucination-wrong).

- **Advocating for Grounded AI Solutions**: An article, highlighted by 0xgrrr, calls for a more realistic approach to AI, aligning with Alter's aim to transform texts and documents effectively. The community resonated with this perspective, appreciating the nuanced take which can be read in full [here](https://www.dbreunig.com/2024/05/16/sober-ai.html).

- **Mac Desktop Project Potential Discontinuation Concern**: A community member raised concerns about the apparent neglect of SimonW's Mac desktop solution after its 0.2 version, expressing their potential pivot to alternative onboarding options.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Markdown Mayhem in Mozilla's Model**: A participant noted that hyperlinks returned from the model via the server were not rendered correctly, providing a [piece of code](https://github.com/Mozilla-Ocho/llamafile/blob/d5f614c9d7d1efdf6d40a8812d7f148f41aa1072/llama.cpp/server/public/index.html#L1113) as evidence and offering to address the issue with a GitHub pull request.

- **Time's Up: Embeddings Edition**: An issue regarding a *httpx.ReadTimeout* error was reported during embeddings generation in a search assistant tutorial, after only 9% completion, along with a [GitHub link](https://github.com/Mozilla-Ocho/llamafile-llamaindex-examples/blob/main/example.md) related to the problem and detailed debug logs, seeking insights for a fix.

- **The Exponential Backoff Back-and-Forth**: In response to the timeout debacle, the suggestion to apply an *exponential backoff* retry strategy was debated, proposing to drop and retry the connection when timeouts occur.

- **Talking Data Sizes**: A clarifying conversation took place about the data volume for an operation, narrowing it down to "a few sample files," which delineates the test's scope.

- **Docker Docks at Llamafile Harbor**: A guide to containerizing llamafile using Docker was highlighted, considering its benefits for streamlining LLM chatbot setups, with a [blog post link](https://www.docker.com/blog/a-quick-guide-to-containerizing-llamafile-with-docker-for-ai-applications/) provided for those in need of a walkthrough.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**AI Alignment Falling Out of Favor**: One member expressed the viewpoint that **alignment research** is losing its appeal among researchers, though no specific reasons or context were provided.

**Needle in a Needlestack—AI's New Challenge**: The **Needle in a Needlestack (NIAN)** benchmark was highlighted, which is posing a significant challenge to models like **GPT-4-turbo**. Resources shared included the [code repository](https://github.com/llmonpy/needle-in-a-needlestack) and [NIAN's website](https://nian.llmonpy.ai/), along with a [Reddit discussion thread](https://www.reddit.com/r/LocalLLaMA/comments/1ct9nyp/needle_in_a_needlestack_nian/) on the topic.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Skunkworks AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1240581958162190396)** (994 messages🔥🔥🔥): 

- **Waiting for Meta's Multi Modal Model**: When asked about supporting Llava or multimodal, a member shared *"they're 'waiting for meta's multi modal model'"* as the primary reason for the delay.
- **Conversation about GPU Expenses**: Members discussed where they get their GPU money, jokingly citing sources like Kaggle and mentioning financial struggles such as *"RAMEN FOR YEARS"* and the high demand for classification tasks.
- **Open Hermes Dataset Utility**: There was enthusiastic discussion about the **OpenHermes dataset** and how including it *"improved performance significantly"*. 
- **Refusal Mechanism in LLMs**: An insightful conversation delved into how removing refusal mechanisms by orthogonalizing weights has made models "smarter" unexpectedly and references the paper "[Refusal in LLMs is mediated by a single direction](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)".
- **Colab and GPU Use Case Discussions**: Various members shared their challenges and successes using **Google Colab** and **Kaggle** for model training, with recommendations to use dedicated services like Runpod and discussions on the viability of older GPUs like the P100.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a">Blackhole - a lamhieu Collection</a>: no description found</li><li><a href="https://tenor.com/view/surprise-welcome-one-sure-gif-13921142">Surprise Welcome GIF - Surprise Welcome One - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated">failspy/llama-3-70B-Instruct-abliterated · Hugging Face</a>: no description found</li><li><a href="https://cloud.google.com/tpu/docs/run-calculation-jax">no title found</a>: no description found</li><li><a href="https://huggingface.co/WizardLM">WizardLM (WizardLM)</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate/en/usage_guides/quantization">Quantization</a>: no description found</li><li><a href="https://cloud.google.com/tpu/docs/run-calculation-pytorch">no title found</a>: no description found</li><li><a href="https://huggingface.co/datasets/cognitivecomputations/Dolphin-2.9">cognitivecomputations/Dolphin-2.9 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslot">UNSLOT - Overview</a>: typing... GitHub is where UNSLOT builds software.</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cmc27y/finrag_datasets_study/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/Skorcht/schizogptdatasetclean">Skorcht/schizogptdatasetclean · Datasets at Hugging Face</a>: no description found</li><li><a href="https://datta0.substack.com/p/ai-unplugged-10-kan-xlstm-openai">AI Unplugged 10: KAN, xLSTM, OpenAI GPT4o and Google I/O updates, Alpha Fold 3, Fishing for MagiKarp</a>: Insights over Information</li><li><a href="https://huggingface.co/mixedbread-ai">mixedbread-ai (mixedbread ai)</a>: no description found</li><li><a href="https://www.parsee.ai/en/blog/finrag-dataset-and-study/">finRAG Dataset: Deep Dive into Financial Report Analysis with LLMs</a>: Discover the finRAG Dataset and Study at Parsee.ai. Dive into our analysis of language models in financial report extraction and gain unique insights into AI-driven data interpretation.</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/big-ups-mike-tyson-cameo-good-job-props-gif-18006586">Big Ups Mike Tyson GIF - Big Ups Mike Tyson Cameo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing&authuser=1#scrollTo=2eSvM9zX_2d3">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — AI Alignment Forum</a>: This work was produced as part of Neel Nanda&#x27;s stream in the ML Alignment &amp; Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…</li><li><a href="http://slatestarcodex.com/2015/12/17/should-ai-be-open/">Tweet from Should AI Be Open?</a>: I. H.G. Wells&#8217; 1914 sci-fi book The World Set Free did a pretty good job predicting nuclear weapons:They did not see it until the atomic bombs burst in their fumbling hands&#8230;before the l…
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1240693821621276855)** (37 messages🔥): 

- **Llama3 experiences high training losses**: After resolving tokenizer issues, one member noted that "eval losses roughly double, training losses more than 3 times as high" on Llama3. They speculated on updating the prompt format or omitting the EOS_TOKEN for improvements.
- **RAM issues with ShareGPT dataset**: A user ran out of 64GB of RAM "trying to convert that shi-" from the ShareGPT dataset. Another member suggested that the code might be inefficient, as it should only require around 10GB of RAM.
- **Discussion on finding similarly formatted text**: One user asked if there were tools for finding similarly formatted text, like all caps or distinct new lines. Suggestions included Python's `re` module and regex, but the user noted the need for an automatic solution capable of handling unknown formats.
- **Opining on Sam Altman's leadership**: A member criticized Sam Altman's leadership, calling it a case of “do as I say not as I do” due to his fear-mongering and lobbying efforts. Another member suggested that the situation was wild, potentially a result of Altman's influence.
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1240587780812439575)** (266 messages🔥🔥): 

- **Context Window AttributeError Troubleshooting**: A member named just_iced sought assistance with a persistent `AttributeError: 'dict' object has no attribute 'context_window'` while training **Llama 3 on custom data**. Various solutions were provided, including modifying core module codes and switching to utilize **Ollama**, leading to successful troubleshooting.

- **Driver's Manual for RAG**: neph1010 suggested that **Retrieval-Augmented Generation (RAG)** could be more suitable than fine-tuning for training models with a driver's manual. They discussed extracting text from PDFs despite the complexity of working with documents containing tables and diagrams.

- **PyPDF2 vs PyPDF**: Linked was shared pointing to [PyPDF2 documentation](https://pypdf2.readthedocs.io/en/3.x/), discussing issues with extracting text and metadata from PDFs.

- **GGUF Model Conversion Issues**: Multiple users, including re_takt, experienced errors during the conversion of models to GGUF with `llama.cpp` and raised those issues with the development team. A fix was provided, which everyone was encouraged to apply by updating their notebooks or using new ones from the GitHub page.

- **Unsloth and CUDA Compatibility**: A member named wvangils faced CUDA compatibility issues on a Databricks platform, receiving a warning about unsupported expandable segments. Further debugging recommended using specific install commands for packages in environments like JupyterLab and possibly rebuilding the environment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://download.pytorch.org/whl/cu118/xformers-0.0.26.post1%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl">no title found</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit">unsloth/llama-3-8b-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://blog.eleuther.ai/transformer-math/">Transformer Math 101</a>: We present basic math related to computation and memory usage for transformers</li><li><a href="https://github.com/unslothai/unsloth/issues/479">RuntimeError: Unsloth: llama.cpp GGUF seems to be too buggy to install. · Issue #479 · unslothai/unsloth</a>: prerequisites %%capture # Installs Unsloth, Xformers (Flash Attention) and all other packages! !pip install &quot;unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git&quot; !pip install -...</li><li><a href="https://pypdf2.readthedocs.io/en/3.x/">Welcome to PyPDF2 &mdash; PyPDF2  documentation</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=yFfaXG0WsQuE">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1240642849544994827)** (2 messages): 

```html
- **AI News humorously acknowledges its own meta-conversation**: A user expressed amusement about the AI summarization part, noting that it was *"some convo somewhere not related to AI News"* and found it funny that *"AI News mentioning another AI News mention"* could happen.
```
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1240565882623627386)** (836 messages🔥🔥🔥): 

- **SD3 release uncertainty and delays**: Several users expressed frustration over the delayed release of SD3. One mentioned a tweet by Emad suggesting SD3 was "due to drop" but remained skeptical as no solid release date has been confirmed.
- **Hardware requirements for SDXL and training**: Members discussed the efficiency of various GPUs, including debates over whether the RTX 4090 is sufficient for training SDXL models. Notably, a 24GB VRAM is seen as minimal for more complex tasks, and some users consider renting more powerful setups.
- **Community skepticism and coping mechanisms**: A user cynically commented on the state of SD3 and the overall delays, sharing a [tweet from Stability](https://twitter.com/chrlaf/status/1772228848387522728). Others shared memes and humorous remarks about waiting and coping with uncertainties.
- **Training resources and dataset contributions**: One user shared [a dataset from Hugging Face](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions) aimed at achieving high-quality results comparable to Dalle 3. Discussions included tips on Lora models and fine-tuning practices for efficient AI art generation.
- **General confusion and off-topic debates**: The chat featured heated personal debates and unrelated topics, from AI models to socio-economic issues. Noteworthy were intense discussions on capitalism, personal success, and the moral dilemmas of wealth acquisition, with some users calling for a return to foundational principles.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@PlantForwardMDlife">Dr Neha Bhanusali</a>: Rheumatologist | Autoimmune specialist Lifestyle Medicine physician </li><li><a href="https://arxiv.org/abs/2405.09818">Chameleon: Mixed-Modal Early-Fusion Foundation Models</a>: We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training approach f...</li><li><a href="https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions">ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions · Datasets at Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/copium-gif-766857345458198993">Copium GIF - Copium - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/EMostaque/status/1790451196915831018?t=YJuHqJJ-YCivInuOrZ2_Lw&s=33">Tweet from Emad (@EMostaque)</a>: @morew4rd @GoogleDeepMind Sd3 is due to drop now I don’t think folk will need that much more tbh with right pipelines</li><li><a href="https://civitai.com/images/12597091">Image posted by 20Twenty</a>: no description found
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1240859703500603412)** (1 messages): 

- **Interactive tables and charts in ChatGPT**: OpenAI announced the rollout of **interactive tables and charts** along with the ability to add files directly from *Google Drive* and *Microsoft OneDrive*. This feature will be available for **ChatGPT Plus, Team, and Enterprise users** over the coming weeks. [Read more](https://openai.com/index/improvements-to-data-analysis-in-chatgpt/).
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1240565390900330516)** (178 messages🔥🔥): 

- **GPT-4o Rollout Generates Buzz**: Members express excitement about the limited rollout of **GPT-4o** with **higher message limits** and **improved vision capabilities** over GPT-4, despite some features such as **voice, video, and Vision** not being fully active yet. They also discussed the freedom to switch models during the conversations improving user experience.

- **Voice Mode and Future Releases**: There is anticipation for the rollout of the enhanced **Voice Mode** in GPT-4o, which is expected in the coming weeks for **ChatGPT Plus** users. Detailed explanations were provided about the possible functionalities and integrations with tools like Be My Eyes.

- **Concerns Over AI's Impact on Employment**: Members debated the potential future where AI could lead to mass job displacement. Discussions ranged from AI-generated productivity improvements to concerns about long-term implications for employment and societal structure.

- **Multi-modal AI Capabilities**: Discussion about the robustness of **GPT-4o** in **image generation**, revealing differences in capabilities compared to previous versions like **DALL-E**. A link to [OpenAI's explorations page](https://openai.com/index/hello-gpt-4o/) was shared to showcase sample images and features.

- **Educational and Ethical Uses of AI**: Conversations touched on the ethical implications and potential uses of AI in education, suggesting that AI could democratize access to personalized tutoring. There were also suggestions about the responsible implementation of AI in assisting with daily tasks and expanding human knowledge.
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1240560811408363550)** (148 messages🔥🔥): 

- **GPT-4o Slows Over Time**: Members observed that as conversations with GPT-4o get longer, the inference speed significantly drops, sometimes resulting in the model halting mid-inference. This issue was noted by users on different platforms including the Mac app and the website, with discrepancies in performance.

- **GPT-4o Tops Rankings**: The updated LMSYS arena rankings show that GPT-4o has claimed the top position. One user enthusiastically noted, "gpt4o top 1".

- **Image Input to GPT-4o**: Users discussed how to send images to GPT-4o, confirming that it's possible through API or by sending the image directly. Instructions and detailed documentation can be found [here](https://platform.openai.com/docs/guides/vision/quick-start).

- **Custom GPTs Upgrading to GPT-4o**: Some users realized that their custom GPTs had already transitioned from GPT-4 Turbo to GPT-4o, evident from the improved response speed. This change appears to be in the rollout stage with varying availability.

- **Free vs. Plus Access to GPT-4o**: The rollout of GPT-4o is not region-specific and is gradually becoming available to more users, with Plus users receiving priority. Despite its enhanced capabilities, the transition and access limits have caused some confusion and mixed experiences among users.


  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1240576297990885467)** (88 messages🔥🔥): 

- **Ontological Drill Help Sought**: A user asked for a powerful ontological drill but felt their current one lacked enough "oomph." They shared a detailed example of their prompt structure.

- **Markdown in AI Prompts**: A user inquired about using markdown in prompts for AI, and another confirmed that the model responds well to markdown, emphasizing that guiding the AI's attention is crucial.

- **Dynamic Character Roles in AI**: Techniques for programming multiple character roles within AI using markdown were discussed. A user shared a comprehensive prompt example involving various characters in a theater-like scenario. [Prompt Example](https://github.com)

- **Troubleshooting Function Calls with GPT-3.5**: Issues with GPT-3.5's function calls returning random data were discussed. The solution proposed involved reframing instructions to focus on using actual provided data only.

- **Issues with GPT-4o and Rewriting**: Several users noted that GPT-4o tends to rewrite original prompts rather than adjust according to feedback. Discussions included guidance on clear positive instruction and avoiding negatives like prohibitions against calculations.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1240576297990885467)** (88 messages🔥🔥): 

- **Markdown helps GPTs understand prompts better**: Members discussed using markdown in prompts to inject character and roles, confirming the model "responds well to it" but may need precise and guided instructions for the best results.
- **Custom GPTs struggle with following specific instructions**: Users shared issues with custom GPTs ignoring detailed prompts or returning randomized data despite clear instructions. One advised to "exhort it to only use provided data" rather than forbidding simulated data.
- **Fun interactions and character development**: Members humorously discussed ways to increase compliance from GPTs, such as emotional or reward-based prompts like "I’ll give you $100 if you only use data from xxxxx," highlighting the human-like nuances in prompting.
- **Model's behavior can change behind the scenes**: A user clarified that GPT-4o and other models only know what they are trained on, emphasizing that OpenAI frequently updates models, which can affect their behavior and response accuracy over time.
- **Rough drafts and creative writing with GPT-4o**: A user noted that GPT-4o excels at creative writing with a blank slate but may regurgitate rough drafts without improving them. Another provided evidence that GPT-4o can indeed provide polished versions of rough drafts when prompted correctly.
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1240561367627464764)** (387 messages🔥🔥): 

- **GPT-4o Performance and Access Issues**: Members discussed the performance and accessibility of **GPT-4o**, noting it is slightly better and faster than previous versions. One member confirmed it is available for free users but mentioned inconsistent message limits and refresh times.
- **Concerns over AI Hallucinations**: Multiple users voiced concerns about **AI hallucinations**, with one noting *"AI might never solve hallucinations,"* which impacts job security in junior roles.
- **Perplexity vs. ChatGPT**: Members debated the relative advantages of **Perplexity** and **ChatGPT**, with one user suggesting *"Perplexity’s advantage lies in better sourcing"* and another indicating **ChatGPT's easy integration of features like web search might challenge Perplexity**.
- **AI for Coding and Creativity**: Some users found **GPT-4o and Opus** beneficial for coding but noted different strengths, with **GPT-4o** offering consistent code quality and **Opus** excelling in other areas such as math and deeper problem-solving.
- **User Experience on Perplexity**: Members shared mixed experiences with **Perplexity**, including issues with text generation prompts in DALL-E 3 and limitations on input length, while others praised it for replacing complex Google searches effectively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/5/16/24158529/reddit-openai-chatgpt-api-access-advertising">Reddit’s deal with OpenAI will plug its posts into “ChatGPT and new products”</a>: Reddit’s signed AI licensing deals with Google and OpenAI.</li><li><a href="https://youtu.be/AxIk_MtryDQ?t=11">Gorgon City - Roped In</a>: Selected - Music on a new level.» Spotify: https://selected.lnk.to/spotify» Instagram: https://selected.lnk.to/instagram» Apple Music: https://selected.lnk.t...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1240568505095684127)** (6 messages): 

- **Paradroid shares search link**: A user shared a link to a [Perplexity AI search](https://www.perplexity.ai/search/Based-on-the-9_2FFgZjS82EFOy3ERD2wg#5). No further context or comments were provided.

- **Clearer915 asks about the news**: A user posted a [Perplexity AI search link](https://www.perplexity.ai/search/Whats-the-news-wMDSwHyGScGoFuezfX6ZgQ) seeking information on current events. The link points to a search titled "What's the news?"

- **Studmunkey343 inquires about least**: Another user shared a [search link](https://www.perplexity.ai/search/Whats-the-least-SuqXuHtLSvqMNzLQVY2Qyw) with a query "What's the least". Further context was not given.

- **Kinoblue queries vague subject**: This user provided a [link](https://www.perplexity.ai/search/what-is-the-Nxl9DYkQTrmRRaZZJCnDTA) to a Perplexity AI search asking "What is the”. The search query appears to be incomplete.

- **Ryanmxx mentions Stability AI**: Shared a [search link](https://www.perplexity.ai/search/Stability-AI-is-CznMl2swRumQbTO5U4AzIw) regarding Stability AI. No further details included.

- **Sam12305575 shares brain benefits search**: A user shared a [search link](https://www.perplexity.ai/search/brain-benefits-of-VJYShXcNROeGjfaWRL842w) about the "brain benefits of". The link includes an emoji 🧠🚶.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1240573960824033300)** (18 messages🔥): 

- **Uncertainty about Sonar-Medium-Online Support**: A user expressed concern about the longevity of support for **sonar-medium-online** because they find the large version **unusable**. They are interested in integrating the perplexity API but need clarity on supported models.
- **Adding Perplexity to Private Discord**: A user inquired whether it is possible to integrate **Perplexity** into a private Discord group, showing interest in utilizing the API within that context.
- **Default Temperature in Perplexity Models**: Users discussed the default temperature setting for **Perplexity models**. One user confirmed via [documentation](https://docs.perplexity.ai/reference/post_chat_completions) that the default temperature is **0.2**.
- **Volatile Responses Test**: A humorous exchange occurred around testing volatility in responses from **Perplexity models** with the query "Who is the lead scientist at OpenAI after May 16, 2024?". The responses varied, showing inconsistency in the model's ability to handle date-specific queries.

**Link mentioned**: <a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found

  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1240756951055466536)** (4 messages): 

- **Updated Terminus Models Live**: Verified users shared an **updated terminus models collection** by [ptx0](https://huggingface.co/collections/ptx0/terminus-xl-65451893a156b3b1d1456514). The collection includes exciting new features.
  
- **OSS AI + Music Explorations**: More OSS AI + Music explorations were introduced, available to view on [YouTube](https://www.youtube.com/watch?v=WrKshOdqW60). This content is credited to a verified community member.

- **Managing On-Prem GPU Clusters**: A new approach for managing on-prem GPU clusters was shared on [Twitter](https://twitter.com/andrey_cheptsov/status/1790674258391163158). It offers practical insights and solutions for better cluster management.

- **Understanding AI for Story Generation**: Listed a resourceful [Blog Post](https://isamu-website.medium.com/understanding-ai-for-stories-d0c1cd7b7bdc) and upcoming [Discord Event](https://discord.com/events/879548962464493619/1240255110093738026) for better understanding AI in story generation, mentioning it would be an interesting topic for further exploration.

- **Ask for Further Topics in Weekly Reading Group**: Community admin encouraged members to suggest more topics for the weekly reading group, mentioning the appeal of story generation and video game AI discussions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/bghira/SimpleTuner/blob/main/documentation/DREAMBOOTH.md#refiner-tuning)">SimpleTuner/documentation/DREAMBOOTH.md at main · bghira/SimpleTuner</a>: A general fine-tuning kit geared toward Stable Diffusion 2.1, DeepFloyd, and SDXL. - bghira/SimpleTuner</li><li><a href="https://huggingface.co/datasets/Vi-VLM/Vista?fbclid=IwZXh0bgNhZW0CMTEAAR2BXlXiqe6SjTjol1ViKCmI7HgogMPvrQU2pIBACQyZyI0av_ey8okihDA_aem_AdV1HiWxI6SngeQmTHG6XLs6v440zT5XTtTpW0yXlGkBFSQkIFrfY7nZyyMJXTF51eFvNHIwuPyArt-XQaSrGf0R)">Vi-VLM/Vista · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1240562824602845204)** (278 messages🔥🔥): 

```html
<ul>
<li><strong>OpenAI Agents and Learning Limitations</strong>: A member clarified that GPTs agents do not learn from additional information post training. Instead, uploaded files are only saved as "knowledge" files for reference and do not modify the agent's base knowledge.</li>
<li><strong>Using Synthetic Data for Models</strong>: There was a discussion on the acceptability of using synthetic data. One member questioned its efficiency, while another reasoned that obtaining real data is often too expensive, affirming that "SLM's are getting better."</li>
<li><strong>ZeroGPU Beta Details</strong>: Members discussed the ZeroGPU feature, currently in beta, which provides free GPU access for Spaces. Details and feedback requests were shared through a <a href="https://huggingface.co/zero-gpu-explorers">link</a>.</li>
<li><strong>MIT License and Commercial Use on HuggingFace</strong>: A member linked the <a href="https://choosealicense.com/licenses/mit/">MIT license</a> details, confirming that it allows for commercial use, distribution, and modification, but raised concerns about HuggingFace's hardware usage terms.</li>
<li><strong>Alternatives to Zephyr for Custom Assistants</strong>: Members discussed the potential removal of the Zephyr model, prompting a recommendation to create custom Spaces using Gradio and API integrations for similar functionalities.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.gradio.app/guides/using-hugging-face-integrations#using-hugging-face-inference-api">Using Hugging Face Integrations</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://choosealicense.com/licenses/mit/">MIT License</a>: A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different t...</li><li><a href="https://huggingface.co/spaces/enzostvs/zero-gpu-spaces">— Zero GPU Spaces — - a Hugging Face Space by enzostvs</a>: no description found</li><li><a href="https://huggingface.co/spaces/KingNish/GPT-4o">OpenGPT 4o - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593">MIT/ast-finetuned-audioset-10-10-0.4593 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-overview">Spaces Overview</a>: no description found</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat">Zephyr Chat - a Hugging Face Space by HuggingFaceH4</a>: no description found</li><li><a href="https://huggingface.co/chat/models/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 - HuggingChat</a>: Use HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 with HuggingChat</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=40378544">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/ruslanmv/AI-Medical-Chatbot/tree/main">ruslanmv/AI-Medical-Chatbot at main</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1240602458733150230)** (17 messages🔥): 

- **Exploration/Exploitation Trade-off in RL**: A member inquired about maintaining the exploration/exploitation trade-off in RL, to which another suggested using the epsilon greedy policy and shared that further details would come in later chapters. Emphasized curiosity and using ChatGPT for more insights.

- **Curiosity-driven Exploration in RL**: Members discussed the concept of curiosity as a method to encourage exploration in reinforcement learning, sharing a [paper](https://pathak22.github.io/noreward-rl/) on "Curiosity-driven Exploration by Self-supervised Prediction". The approach gives higher rewards when agents can't predict the outcome of their actions.

- **First RL Model Submission on HuggingFace**: A user celebrated pushing their first LunarLander-v2 model into the Hugging Face repository and completing Unit 1 in Deep RL. They were encouraged to share results and the repository for feedback.

- **Installing HuggingFace Transformers**: A member shared their experience starting with HuggingFace by learning the installation process. They provided a [link](https://huggingface.co/docs/transformers/installation) to the detailed installation instructions for various deep learning libraries including PyTorch, TensorFlow, and Flax.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pathak22.github.io/noreward-rl/">Curiosity-driven Exploration by Self-supervised Prediction</a>: Pathak, Agrawal, Efros, Darrell. Curiosity-driven Exploration by Self-supervised Prediction. In ICML, 2017.</li><li><a href="https://huggingface.co/docs/transformers/installation">Installation</a>: no description found</li><li><a href="https://youtu.be/uQcHXEGRECU">business advisor AI project using langchain and gemini AI startup.</a>: so in this video we have made the project to make business advisor using langhcian and gemini. AI startup idea. we resume porfolio ai start idea
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1240742275139637390)** (6 messages): 

- **Getting Started with Candle**: A member shared a [Medium article](https://medium.com/@cursor0p/getting-started-with-candle-%EF%B8%8F-535d7a85e30a) focusing on **Candle**. It's a helpful resource for beginners interested in this tool.
- **Unleashing Multimodal Power with GPT-4o**: Another shared [Medium post](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a) explaining the integration of **GPT-4o** with **LlamaParse**. It promises to enhance multimodal capabilities significantly.
- **How Microchips Are Made Explained On YouTube**: A member declared this [YouTube video](https://youtu.be/dX9CGRZwD-w) as possibly the best technology video ever made, focusing on "How are Microchips Made?". It includes a promotion for Brilliant.org to further expand viewers’ knowledge.
- **OpenAI Criticized for Lack of Openness**: A YouTube video titled ["Big Tech AI Is A Lie"](https://www.youtube.com/watch?v=8BlRT7Ktw1c) was shared criticizing **OpenAI** for not being truly open. Another member pointed out that this is why platforms like HuggingFace are valuable, prompting a realization that HuggingFace models can achieve desired outcomes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=8BlRT7Ktw1c">Big Tech AI Is A Lie</a>: Learn how to use AI at work with Hubspot&#39;s FREE AI for GTM bundle: https://clickhubspot.com/u2oBig tech AI is really quite problematic and a lie. ✉️ NEWSLETT...</li><li><a href="https://youtu.be/dX9CGRZwD-w">How are Microchips Made?</a>: Go to http://brilliant.org/BranchEducation/ for a 30-day free trial and expand your knowledge. Use this link to get a 20% discount on their annual premium me...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1240851124068810853)** (4 messages): 

- **Exploring ControlNet Training**: One user shared their journey on *"understanding and implementing controlnet training."* They included a [linked image](https://cdn.discordapp.com/attachments/1236831148580278333/1240850774817378475/image0.png) related to their project.
- **Business Advisor AI Project**: Another user posted a [YouTube video](https://youtu.be/uQcHXEGRECU) titled "business advisor AI project using langchain and gemini AI startup," showcasing their efforts to create a business advisor using **LangChain** and **Gemini AI**.
- **GenAI-Powered Study Companion**: A user linked a [LinkedIn post](https://www.linkedin.com/posts/harshdayal_educationinnovation-genai-activity-7197227129409810432-4llP) about their project for building a powerful study companion using **GenAI**. The project aims to innovate in the field of education.
- **Challenges with GPT-4o and Grid Puzzle Generation**: One user discussed difficulties in getting GPT-4o to create proper grid puzzles, mentioning errors like creating 4x5 or 123x719 grids. They are seeking an open-source model for better results, expressing frustration that *"OpenAI is not open!!!"*

**Link mentioned**: <a href="https://youtu.be/uQcHXEGRECU">business advisor AI project using langchain and gemini AI startup.</a>: so in this video we have made the project to make business advisor using langhcian and gemini. AI startup idea. we resume porfolio ai start idea

  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1240742810433490983)** (6 messages): 

- **Thumbnails brainstorming goes thematic**: Members discussed ideas for **thumbnails** with one sharing a themed thumbnail inspired by the **Dwarf Fortress GUI**. They addressed **opacity concerns** with text and logo for better readability while scrolling.
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1240935488651792394)** (1 messages): 

- **Say hello to Tuxemons!**: A new dataset called [Tuxemon](https://huggingface.co/datasets/diffusers/tuxemon) has been released, featuring humorous creatures instead of Pokemons. This dataset, sourced from the [Tuxemon Project](https://wiki.tuxemon.org/Main_Page), offers `cc-by-sa-3.0` images with dual captions for text-to-image tuning and benchmarking experiments.


**Link mentioned**: <a href="https://huggingface.co/datasets/diffusers/tuxemon">diffusers/tuxemon · Datasets at Hugging Face</a>: no description found

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1240704543327785134)** (16 messages🔥): 

- **Latent Diffusion Models in Diffusers**: A member inquired about the presence of latent diffusion models in diffusers, specifically those based on VAE and VQ-VAE, and their ease of training from scratch.

- **Help with UNet Convergence Issues**: A member sought advice on their UNet model, as the loss started at 0.7 and converged at 0.51, indicating potential issues with the model structure despite successful training runs. Another member mentioned that the size of the dataset could affect the validation loss and shared their experience with small datasets and surprising results.

- **Hyperparameters and Model Structure Shared**: The member provided hyperparameters (Depth: 5, Lr: 0.002, Loss: BCE with logits) and detailed their UNet model code, seeking insights on why the final results seemed to resemble random guessing.

- **Creating Virtual AI Influencer**: A member shared their accomplishment of creating a virtual AI influencer using CV and AI tools, linking a [YouTube video](https://www.youtube.com/watch?v=qTsdgUyMY94&t=640s) that explains the process.

- **Creating Parquet Files with Images**: Another member asked for help creating a Parquet file containing images and their corresponding entities using PyArrow, as their attempt resulted in the image column being formatted as a byte array when uploaded to Hugging Face.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=qTsdgUyMY94&t=640s">Influenceuse I.A : POURQUOI et COMMENT créer une influenceuse virtuelle originale ?</a>: Salut les Zinzins !  🤪Le monde fascinant des influenceuses virtuelles s&#39;invite dans cette vidéo. Leur création connaît un véritable boom et les choses bouge...

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1240794595340849172)** (2 messages): 

- **Outdated training data hampers code models**: One user noted that outdated training data is a significant issue causing language models for coding to struggle. They suggested that continuous retraining is necessary for these models to stay up to date.
- **Curiosity about connectionist temporal classification (CTC)**: A user questioned whether connectionist temporal classification (CTC) is still relevant in current discussions or use cases in NLP.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1240643952772644986)** (4 messages): 

- **Latent Space Pixel Representation Questioned**: A user posed a question regarding the latent space representation of pixels, suggesting each value should represent 48 pixels in the pixel space. For more details, they referenced [an article](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#the-4-channels-of-the-sdxl-latents) on HuggingFace's blog.

- **Caught on Collab**: A member asked for assistance with Step 7 of the Hugging Face Diffusion Models Course on Google Collab. They encountered a **ValueError** indicating that the provided path is not a directory when attempting to upload directories using the `HfApi` class to the Hub. 

_Message consists of a blend of direct references to links and detailed steps within the Hugging Face framework, reflecting active discussions on AI model training and deployment hurdles on the platform._
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/diffusion-course/unit1/2#step-7-push-your-model-to-the-hub)">Introduction to 🤗 Diffusers - Hugging Face Diffusion Course</a>: no description found</li><li><a href="https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#the-4-channels-of-the-sdxl-latents)">Explaining the SDXL latent space</a>: no description found
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1240612359060590592)** (2 messages): 

- **Streaming-like brain processing for AI**: "People have very small working memory, and yet we can read and process long books, have hours-long conversations, etc. - our brains work in a more streaming fashion, and just update what's most relevant / important in as the conversation goes," one participant noted. They suggested focusing on methods that mimic this, such as *Infini-attention* ([arxiv.org/abs/2404.07143](https://arxiv.org/abs/2404.07143)).

- **Needle in a Needlestack benchmark introduced**: The *Needle in a Needle Stack (NIAN)* benchmark, discussed in a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ct9nyp/needle_in_a_needlestack_nian/), presents a new challenge for evaluating LLMs. Even GPT-4-turbo faces difficulties with this benchmark, which tests models by asking questions about a specific limerick placed within many others ([GitHub](https://github.com/llmonpy/needle-in-a-needlestack); [Website](https://nian.llmonpy.ai/)).

**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1ct9nyp/needle_in_a_needlestack_nian/">Reddit - Dive into anything</a>: no description found

  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1240575410878550057)** (5 messages): 

```html
- **Seeking real-time UI processing model**: A member is looking for demos and articles on models similar to **Fuyu** that process screen actions almost in real time (*every 1000 ms, a screenshot is made and sent to Fuyu to process what's happening on the screen and where to click*).

- **Elon Musk announces Neuralink clinical trial**: [Elon Musk announced on X](https://x.com/elonmusk/status/1791332539220521079) that Neuralink is accepting applications for its second participant in their brain implant trial, enabling users to control devices through thoughts. The trial specifically invites individuals with quadriplegia to explore new control methods for computers.
```

**Link mentioned**: <a href="https://x.com/elonmusk/status/1791332539220521079">Tweet from Elon Musk (@elonmusk)</a>: Neuralink is accepting applications for the second participant.   This is our Telepathy cybernetic brain implant  that allows you to control your phone and computer just by thinking.  No one better th...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1240699003977470053)** (5 messages): 

- **SUPRA offers cost-effective uptraining for transformers**: The Arxiv paper proposes [Scalable UPtraining for Recurrent Attention (SUPRA)](https://arxiv.org/abs/2405.06640), which uptrains large pre-trained transformers into Recurrent Neural Networks. This method aims to address the poor scaling of original linear transformer formulations.

- **Llama-3 gets a NumPy implementation**: A new repository on GitHub, [llama3.np](https://github.com/likejazz/llama3.np), provides a pure NumPy implementation for the Llama-3 model. This approach offers an **alternative for those looking to understand or modify the underlying algorithms** without TensorFlow or PyTorch dependencies.

- **Stable Diffusion 3 to go open-weight**: Argmax Inc. announced their [partnership with Stability AI](https://x.com/argmaxinc/status/1790785157840125957) to bring on-device inference of Stable Diffusion 3 through DiffusionKit. They are focusing on optimizing the model for Mac using MLX and Core ML, with plans to open-source the project.

- **WebGPU powers experimental Moondream on HuggingFace**: The [Moondream WebGPU](https://huggingface.co/spaces/Xenova/experimental-moondream-webgpu) project on HuggingFace Spaces showcases an experimental implementation. This highlights the potential for running complex models directly in web environments.

- **Hierarchical Memory Transformer enhances long-context processing**: The Arxiv submission details the [Hierarchical Memory Transformer (HMT)](https://arxiv.org/abs/2405.06067), a framework inspired by human memory processes. This approach aims to improve models' ability to handle extended context windows by organizing memory hierarchies effectively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Xenova/experimental-moondream-webgpu">Experimental Moondream WebGPU - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.06640">Linearizing Large Language Models</a>: Linear transformers have emerged as a subquadratic-time alternative to softmax attention and have garnered significant interest due to their fixed-size recurrent state that lowers inference cost. Howe...</li><li><a href="https://arxiv.org/abs/2405.06067">HMT: Hierarchical Memory Transformer for Long Context Language Processing</a>: Transformer-based large language models (LLM) have been widely used in language processing applications. However, most of them restrict the context window that permits the model to attend to every tok...</li><li><a href="https://x.com/argmaxinc/status/1790785157840125957">Tweet from argmax (@argmaxinc)</a>: On-device Stable Diffusion 3 We are thrilled to partner with @StabilityAI for on-device inference of their latest flagship model!  We are building DiffusionKit, our multi-platform on-device inference ...</li><li><a href="https://github.com/likejazz/llama3.np">GitHub - likejazz/llama3.np: llama3.np is pure NumPy implementation for Llama 3 model.</a>: llama3.np is pure NumPy implementation for Llama 3 model. - likejazz/llama3.np
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1240826819872161834)** (2 messages): 

- **Announcing Simulators Salon Event**: *Simulators Salon* scheduled for Saturday, 5/18 at **Noon Pacific / 3PM Eastern**. Join the event through this [Discord link](https://discord.gg/rt87RHmH?event=1240826259920125982).

**Link mentioned**: <a href="https://discord.gg/rt87RHmH?event=1240826259920125982">Join the Nous Research Discord Server!</a>: Check out the Nous Research community on Discord - hang out with 7136 other members and enjoy free voice and text chat.

  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1240589003615371334)** (204 messages🔥🔥): 

- **LMSys Leaderboard updated with GPT-4o**: LMSys has updated their leaderboard to include GPT-4o, but some users are disappointed with its performance, noting specific points in coding benchmarks. One user remarked, "The code benchmark dropped 50 points from here."

- **Humans mistake GPT-4 for people in Turing test**: [A preprint](https://x.com/camrobjones/status/1790766472458903926?s=46) claims that GPT-4 is judged to be human 54% of the time in Turing tests, cited as "the most robust evidence to date" of passing the Turing test.

- **Debate on OpenAI departures linked to lack of AGI progress**: Some users speculate that recent departures from OpenAI are due to a perceived lack of progress towards AGI, rather than detecting any imminent danger. As @deki04 shared, *"The Safety team left not because they saw something but because they saw nothing."*

- **GPT-4o's output structure criticized**: Users criticize GPT-4o for its generic response patterns, preferring more tailored and problem-solving responses. One comment noted, "8 out of 10 responses are just enumerations of steps, rather than a simple reasoning."

- **Excitement and skepticism around new AI integration**: There's palpable excitement over [GPT-4o’s multimodal capabilities](https://fxtwitter.com/VictorTaelin/status/1790185366693024155), especially in handling complex tasks like *Pokémon Red* gameplay in a terminal. However, there's also skepticism regarding its training architecture and output efficacy compared to solely text-trained models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/camrobjones/status/1790766472458903926?s=46">Tweet from Cameron Jones (@camrobjones)</a>: New Preprint: People cannot distinguish GPT-4 from a human in a Turing test.  In a pre-registered Turing test we found GPT-4 is judged to be human 54% of the time.  On some interpretations this consti...</li><li><a href="https://ai.google.dev/gemini-api/docs/caching">no title found</a>: no description found</li><li><a href="https://x.com/batwood011/status/1790989472479269121?s=46">Tweet from Brian Atwood (@batwood011)</a>: Plot twist:  The Safety team left not because they saw *something* but because they saw *nothing*  No real danger. Only limitations, dead ends and endless distractions with commercialization — no path...</li><li><a href="https://x.com/sama/status/1790066235696206147">Tweet from Sam Altman (@sama)</a>: especially at coding</li><li><a href="https://x.com/victortaelin/status/1791213162525524076?s=46">Tweet from Taelin (@VictorTaelin)</a>: RELEASE DAY  After almost 10 years of hard work, tireless research, and a dive deep into the kernels of computer science, I finally realized a dream: running a high-level language on GPUs. And I&#39;m...</li><li><a href="https://fxtwitter.com/VictorTaelin/status/1790185366693024155">Tweet from Taelin (@VictorTaelin)</a>: Seriously - this is great. I can&#39;t overstate how good it is. I spent a LONG time to get a half-decent run with Opus back then. Other models could barely draw a frame. GPT-4o just... plays the game...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1casosh/groq_hosted_llama370b_is_not_smart_probably/?rdt=37723">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1240579808120012881)** (35 messages🔥): 

- **EOS token not stopping fine-tuning model**: A member had issues with the **eos_token_id** not stopping generation on a fine-tuned **Qwen 4B** model. Another user suggested ensuring the stop token used during training matches the inference setting, possibly using “</s>”.
  
- **Nous Hermes model replies in Chinese**: A user reported **Nous-Hermes-2-Mixtral-8x7B-DPO** returning responses in Chinese instead of English summaries. Another member noted that Together’s inference endpoint might be broken as the model shouldn’t have Chinese samples.

- **Regex vs. semantic search for text patterns**: Members discussed finding text with specific formatting more efficiently. One user suggested **semantic search** might inherently match formatting patterns, while another proposed regex for simpler pattern retrieval.

- **GPT-4o for symbolic language in algebra**: A user suggested using **GPT-4o** to create a symbolic language for simple integrals and derivatives processing. Another seemed interested in trying this approach for algebra tasks.
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/)** (1 messages): 

.interstellarninja: https://fxtwitter.com/alexalbert__/status/1791137398266659286
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1240871924469469284)** (1 messages): 

- **Automated Knowledge Graphs with DSPy and Neo4j**: An LLM-driven project on **automated knowledge graph construction from text** using **DSPy** and **Neo4j** was shared. The GitHub repository linked is [here](https://github.com/chrisammon3000/dspy-neo4j-knowledge-graph/tree/main).

**Link mentioned**: <a href="https://github.com/chrisammon3000/dspy-neo4j-knowledge-graph/tree/main">GitHub - chrisammon3000/dspy-neo4j-knowledge-graph: LLM-driven automated knowledge graph construction from text using DSPy and Neo4j.</a>: LLM-driven automated knowledge graph construction from text using DSPy and Neo4j. - chrisammon3000/dspy-neo4j-knowledge-graph

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1240637812823556179)** (3 messages): 

- **Universal Scraper Agent video shared**: A member posted a [YouTube video](https://youtu.be/dSX5eoD4-u4?si=CFiZowc26j-u84va) titled "Wait, this Agent can Scrape ANYTHING?!", exploring how to build a universal web scraper for e-commerce sites in 5 minutes. The video covers using a browser directly for tasks like pagination and captcha handling.
- **Invitation for Saturday's salon**: Members were invited to showcase their simulations, chats, or sites on stream for the upcoming Saturday salon event. Interested participants were encouraged to DM the organizer.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/nousresearch?event=1240826259920125982">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://youtu.be/dSX5eoD4-u4?si=CFiZowc26j-u84va">“Wait, this Agent can Scrape ANYTHING?!” - Build universal web scraping agent</a>: Build an universal Web Scraper for ecommerce sites in 5 min; Try CleanMyMac X with a 7 day-free trial https://bit.ly/AIJasonCleanMyMacX. Use my code AIJASON ...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1240564648550596618)** (51 messages🔥): 

```html
- **Open-source: A blessing and a curse**: Members debated the pros and cons of open-source projects, with one noting that *"Open-sourcing a project from the start does not stop it from getting closed in the future."* Others argued that major projects often leave forked open-source alternatives when transitioning to closed-source, citing Mongo, Terraform, and Redis as examples.
- **Advent of Code as a Mojo starting point**: For those looking to get started with Mojo, Advent of Code 2023 was suggested as a good jumping-off point. You can find it [here](https://github.com/p88h/aoc2023).
- **GIS ambitions in Mojo**: Discussion about future plans to integrate GIS capabilities into Mojo, with mentions of needing foundational building blocks first. The conversation touched on complexities like LAS readers and various data structures needed to support such features.
- **Struggles with Mojo on Windows**: Users discussed difficulties running Mojo on Windows, especially mentioning challenges with CMD and PowerShell. It was clarified that Mojo currently supports Windows only through WSL.
- **Humor in stock exchanges**: A light-hearted exchange joked about Modular potentially being publicly traded, with the suggestion that it could use an emoji as a ticker symbol.
```

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/p88h/aoc2023">GitHub - p88h/aoc2023: Advent of Code 2023 (Mojo)</a>: Advent of Code 2023 (Mojo). Contribute to p88h/aoc2023 development by creating an account on GitHub.</li><li><a href="https://www.modular.com/">Modular: Accelerating the Pace of AI</a>: The Modular Accelerated Xecution (MAX) platform is the worlds only platform to unlock performance, programmability, and portability for your AI workloads.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1240773393821732886)** (2 messages): 

- **Modular tweets updates**: Shared [links](https://twitter.com/Modular/status/1791209230948601903) to their latest updates on Twitter, highlighting key advancements and information. 
- **More updates from Modular**: Another tweet was shared [here](https://twitter.com/Modular/status/1791535613411570039), likely continuing their update spree with new insights.
  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1240949397605584917)** (1 messages): 

- **HVM achieves automatic parallelization magically**: A member asked, *"How does the [HVM](https://higherorderco.com/) perform perfect automatic parallelization?"* The query reflects interest in understanding the seemingly "magic" capability of Higher Order's VM for efficient parallel processing.
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1240588861172748318)** (115 messages🔥🔥): 

- **Bitcast for Bitwise Conversion Saves the Day**: A user inquired about casting a uint to a float bitwise, and [bitcast](https://docs.modular.com/mojo/stdlib/memory/unsafe/bitcast) proved to be the solution. "*great that worked*" confirmed the user's satisfaction.
  
- **Mojo Enumerate Workaround**: A member asked about Python-like `enumerate()` in Mojo language. Another member suggested using indexes for now but mentioned that `enumerate()` is likely planned for future implementation.

- **Parallelize Call Causes List Issues in Structs**: A user found that using `parallelize` caused their struct's `List` to go haywire. The problem was identified as needing lifetime extension and resolved by binding the list to a dummy variable.

- **MojoDojo is Back**: A user reported that the [mojodojo](https://github.com/modularml/mojodojo.dev) website is active again and now officially under the modularml organization.

- **Tuple Iteration and `MLIR` Types Confound Users**: Members discussed implementing `__iter__` and `__contains__` for `Tuple`, distinguishing between `utils/static_tuple` and `builtin/tuple`. Clarifications on `i1` as a one-bit integer and resources for MLIR types were shared, leading to a deeper dive into type handling intricacies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/rebind/#functions)">rebind | Modular Docs</a>: Implements type rebind.</li><li><a href="https://mlir.llvm.org/docs/Dialects/Builtin/#integertype)">Builtin Dialect - MLIR</a>: no description found</li><li><a href="https://github.com/modularml/mojo/pull/2703">[mojo-stdlib] Add variadic initialiser, __iter__ and __contains__ to InlineList by ChristopherLR · Pull Request #2703 · modularml/mojo</a>: This PR adds some features to InlineList ( related issue #2658 ) Variadic initialiser var x = InlineList[Int](1,2,3) iter var x = InlineList[Int](1,2,3) for i in x:     print(i) contains var x = In...</li><li><a href="https://github.com/modularml/mojo/issues/2658">[stdlib] Implement `__contains__` for `Tuple`, `List`, `ListLiteral` (almost) · Issue #2658 · modularml/mojo</a>: Now that we have ComparableCollectionElement, we can try to implement __contains__ for some common collection types using a workaround similar to what was employed in #2190. It&#39;s possible that the...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1240990171525288087)** (12 messages🔥): 

- **Performance Regression in Mojo's List.append**: A member pointed out performance issues with `List.append` in Mojo when dealing with large input sizes over 1k elements, providing [benchmark results comparing Mojo, Python, and C++](#). They noted that Mojo's performance scaled less efficiently compared to the other languages.

- **Rust's Memory Allocation for Vec**: Another member highlighted that Rust's `Vec` doubles its capacity when reallocating, similar to Mojo, linking to [Rust's implementation](https://github.com/rust-lang/rust/blob/master/library/alloc/src/raw_vec.rs#L464).

- **Go's Dynamic Array Resizing Strategy**: A discussion about Go's resizing behavior revealed that Go doubles the size of backing arrays until 1024 elements, then increases size by 25%, citing [Go's source code](https://github.com/golang/go/blob/cb2353deb74ecc1ca2105be44881c5d563a00fb8/src/runtime/slice.go#L95).

- **Comparison with Python and C++**: Members speculated that C++ and Python might be utilizing more sophisticated realloc strategies or optimizations, contributing to better performance in these benchmarks compared to Mojo.

- **External Resources and Experiments**: Discussions included various resources and personal experiments to understand memory allocation strategies in Mojo, Rust, and Go, such as a [project on GitHub](https://github.com/dorjeduck/mostring) exploring different `StringBuilder` ideas in Mojo, which internally uses a `List` for storage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/collections/list.mojo#L223">mojo/stdlib/src/collections/list.mojo at bf73717d79fbb79b4b2bf586b3a40072308b6184 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/rust-lang/rust/blob/master/library/alloc/src/raw_vec.rs#L464">rust/library/alloc/src/raw_vec.rs at master · rust-lang/rust</a>: Empowering everyone to build reliable and efficient software. - rust-lang/rust</li><li><a href="https://github.com/dorjeduck/mostring">GitHub - dorjeduck/mostring: variations over StringBuilder ideas in Mojo</a>: variations over StringBuilder ideas in Mojo. Contribute to dorjeduck/mostring development by creating an account on GitHub.</li><li><a href="https://github.com/golang/go/blob/cb2353deb74ecc1ca2105be44881c5d563a00fb8/src/runtime/slice.go#L95>">go/src/runtime/slice.go at cb2353deb74ecc1ca2105be44881c5d563a00fb8 · golang/go</a>: The Go programming language. Contribute to golang/go development by creating an account on GitHub.</li><li><a href="https://doc.rust-lang.org/std/vec/struct.Vec.html#capacity-and-reallocation">Vec in std::vec - Rust</a>: no description found
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 34
https://www.modular.com/newsletters/modverse-weekly-34
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/)** (1 messages): 

ModularBot: Congrats <@891492812447698976>, you just advanced to level 3!
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1240569266407870554)** (69 messages🔥🔥): 

- **New Nightly Mojo Compiler Released**: A new nightly Mojo compiler `2024.5.1607` was released, which can be updated using `modular update nightly/mojo`. You can check out the [diff since the last nightly release](https://github.com/modularml/mojo/compare/f5f5109541c31615a68a3c4b58bd1e75b59625f6...c506c9400329824cd0fcfc408115a8e7fea968d0) and the [changes since the last stable release](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Conditional Methods Syntax Praised**: There was a positive reaction to the new conditional methods syntax in the recent PRs. One member noted, "This new syntax for conditional methods is awesome!"

- **Avoid 'Cookie Licking' in Contributions**: GabrielDemarmiesse raised an issue about "cookie licking," where new contributors claim issues but don't work on them promptly, discouraging others. Suggestions included encouraging smaller PRs and immediate draft PR submissions to prevent this.

- **Issue with Syncing Forks**: Members discussed challenges that new contributors face when syncing their forks with the `nightly` branch, leading to commit inflation and failing DCO checks. A [GitHub guide](https://github.com/gabrieldemarmiesse/getting_started_open_source) was shared to help avoid these issues.

- **MAX Nightlies Released**: MAX nightlies have been released, which include macOS support and MAX Serving. Instructions for setting up can be found at [modul.ar/get-started](https://modul.ar/get-started), and users must use PyTorch 2.2.2.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/2556)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#branching-off-nightly">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2556">[Feature Request] DX: Change the default branch of modularml/mojo from `main` to `nightly` · Issue #2556 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I would like a modularml admin to go to the settings o...</li><li><a href="https://github.com/gabrieldemarmiesse/getting_started_open_source">GitHub - gabrieldemarmiesse/getting_started_open_source: You want to contribute to an open-source project? You don&#39;t know how to do it? Here is how to.</a>: You want to contribute to an open-source project? You don&#39;t know how to do it? Here is how to. - gabrieldemarmiesse/getting_started_open_source</li><li><a href="https://modul.ar/get-started.">Get started with MAX Engine | Modular Docs</a>: Welcome to the MAX Engine setup guide!</li><li><a href="https://modul.ar/max-nightly-examples">max/examples at nightly · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1240567037835870240)** (117 messages🔥🔥): 

```html
<ul>
  <li><strong>Users troubleshoot glibc issues for installing LM Studio:</strong> A user with glibc 2.28 and kernel 4.19.0 faces challenges, and others suggest they might need a significant upgrade. Another member suggests trying LM Studio version 0.2.23.</li>
  <li><strong>Discussion on embedding models for RAG in Pinecone:</strong> A user encounters difficulties in retrieving context and generating augmented responses after embedding data into Pinecone. No direct tutorial links are provided.</li>
  <li><strong>Troubleshooting LM Studio installation in nested VM:</strong> A user reports an error 'Fallback backend llama cpu not detected!' on a VM without host VT transfer. Another member confirms the VM setup might be the issue.</li>
  <li><strong>False positive antivirus warning for LM Studio installer:</strong> A user reports their antivirus flagged the 0.2.23 installer as a virus. Another member assures it’s a false positive and advises to allow the file in the antivirus software.</li>
  <li><strong>Comparing model performance and quantization:</strong> Discussions include comparing imatrix quants by Bartowski and Mradermacher, with detailed testing and results shared. The consensus leans towards preferring imatrix quants assuming a sufficiently random dataset.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/executorch-overview">PyTorch ExecuTorch</a>: no description found</li><li><a href="https://huggingface.co/abetlen/nanollava-gguf">abetlen/nanollava-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nisten/obsidian-3b-multimodal-q6-gguf">nisten/obsidian-3b-multimodal-q6-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat">deepseek-ai/DeepSeek-V2-Lite-Chat · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1240564082197921802)** (23 messages🔥): 

- **Best models for coding depend on language and hardware**: A member noted that the best coding models depend on the programming language and hardware capabilities. They suggested checking past discussions in the channel and mentioned models like **Nxcode CQ 7B ORPO** and **CodeQwen 1.5 finetune** for Python.

- **LM Studio can't generate files directly**: A member asked if any model could generate a .txt file, to which others responded that **LM Studio can't generate files directly**. Users need to manually copy and paste outputs into text documents but can use the export to clipboard function.

- **Forcing models to show only code is inconsistent**: A user asked how to make models show only code without explanations. Others explained that even with explicit prompts and markdown settings, **LLMs often still provide explanations**; additional filtering or parsing might be necessary.

- **Fastest semantic text embeddings too slow**: A user mentioned that the fastest embedding model they found is **all miniLM L6**, but it isn't fast enough for their needs.

- **Medical LLM recommendations sought**: A member requested recommendations for a **medical LLM** to try with LMS, but no specific models were mentioned in the replies.
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1240657772253810728)** (4 messages): 

- **False positive on Malwarebytes**: A user reported a false positive with Malwarebytes Anti-malware and Windows Defender, suggesting no actual threats were detected. They shared a [VirusTotal link](https://www.virustotal.com/gui/file/29778ae530836508920126938dce41ba539c191e9201dce23f210a09b4315119).
- **Comodo flags llama.cpp binaries**: Another user mentioned that Comodo antivirus flagged llama.cpp binaries. It was noted that unsigned binaries from llama.cpp likely triggered this strict antivirus response.

**Link mentioned**: <a href="https://www.virustotal.com/gui/file/29778ae530836508920126938dce41ba539c191e9201dce23f210a09b4315119">VirusTotal</a>: no description found

  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1240950923992240128)** (31 messages🔥): 

- **Navigating LLaMa prompt templates**: A user asked for help converting a prompt template for **LLama3**. Solutions were shared, highlighting a format change and emphasizing **client-side state management** over server-side.

- **Clarifying historical context in AI responses**: It was discussed that historical messages need to be included in every new request for context as **LLMs do not maintain historical memory** between requests. *"The AI does not remember, every new request is from scratch."*

- **Using LangChain for memory management**: The user explored managing chat history using **LangChain**, specifically with `ConversationBufferWindowMemory`. This generated positive feedback as it seemed to meet the user's needs.

- **Exploring alternatives with context caching**: The discussion mentioned some paid services like **Gemini's context caching** for historical context management as an alternative, which the user preferred to avoid in favor of open-source solutions. *"cannot afford paid ones, I prefer learning on opensource."*

- **Experimenting with new prompt solutions**: After implementing suggested changes, the user confirmed successful results and planned to conduct further experiments. *"Yes, it works, going experiment more, thanks!"*
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1240763246919553134)** (13 messages🔥): 

- **Running 70B Llama3 on CPU achieves milestone**: A member successfully ran a 70B Llama3 model on an Intel i5 12600K CPU, achieving speeds above 1 token per second. They noted that the performance is significantly influenced by memory access.

- **RAM speed critical for performance**: Another member pointed out that aligning RAM speed with BIOS settings can drastically improve performance. Disabling e-cores on Alder Lake CPUs was mentioned as a necessary step, improving token generation speed from 0.4 to 0.6 tokens/sec for Q8 quantization.

- **Quantization challenges and insights**: The discussions revealed issues with quantization accuracy and performance. Incoherent results on IQ3 quant and the impact of using different imatrix versions were highlighted, with a preference for non-quant q2k methods due to better stability.

- **Memory overclocking limits**: Efforts to push memory frequency above 4800 MHz resulted in system non-bootability, highlighting limitations. The member also noted no performance gain with different thread counts for 70B 4bit+ quantizations, unlike the case with smaller models like llama3 7B 16f.

- **Comparing GPU architectures**: There was a clarification that the M40 has one GM200 chip and mentioned that the Tesla P100, despite its high memory bandwidth, struggles to outperform older GTX 1060 in some use cases. The conversation also touched upon the surprising performance metrics of the P40 versus the P100.
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1240594453299859477)** (8 messages🔥): 

- **User seeks help choosing model storage location**: A user asked for assistance in selecting a different drive for storing models, experiencing a persistent error message despite proper folder access. Another member suggested checking if LM Studio has read/write permissions.

- **Feedback on UI customization options**: One member requested a menu for disabling unused parts of the code and customizing window visibility to manage server overload better. Another commented that despite the big green "start server" button, the user interface remains cluttered.

- **Permissions and location changing issues**: The initial user clarified they are trying to move model storage from a small m.2 drive to an HDD but faces permission errors. They requested code to verify file permissions, confirming they have already set full write permissions manually.

- **Clarification on system setup**: Another member asked for details about the operating system and file system type, plus a screenshot of the model path before and after attempting the change, mentioning their setup with Debian GNU/Linux and ext4.

- **Continuation of conversation moved to another channel**: The discussion was moved to a different channel for further troubleshooting and clarity. [Conversation moved](https://discord.com/channels/1110598183144399058/1111440136287297637/1240773722441519126).
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1241047852478890056)** (1 messages): 

- **LM Studio bug impacts Autogen Studio**: A member reported encountering an **autogen Studio bug** specific to LM Studio. They experienced *"1-2 word responses and a very quick TERMINATE message."* and sought confirmation and solutions from others.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1240561728845254656)** (3 messages): 

- **Hopes rise for RX 6800 improvements**: A member expressed hopes for performance improvements for their **RX 6800** with the new ROCm support for Windows. Another member confirmed that the **6800** is indeed supported, raising expectations for enhanced performance.
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1240979792313061458)** (1 messages): 

- **Hugging Face invests $10 million in free shared GPUs**: Hugging Face commits $10 million in **free shared GPUs** to aid developers, particularly **small developers, academics, and startups**, in creating new AI technologies. CEO Clem Delangue explained, "We are lucky to be in a position where we can invest in the community," emphasizing their drive to counter **AI centralization** by tech giants ([source](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai)).

**Link mentioned**: <a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai">Hugging Face is sharing $10 million worth of compute to help beat the big AI companies</a>: Hugging Face is hoping to lower the barrier to entry for developing AI apps.

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1241056711859769404)** (1 messages): 

- **Performance Difference in Triton Tutorials**: A user queried about performance variations between Umer's [YouTube tutorial](https://www.youtube.com/watch?v=DdTsX6DQk24) and the official tutorial for mamul on Triton's website. They noted that despite using the same techniques, their reimplementation yielded significantly worse performance and questioned if differences were due to the use of "swizzle" indexing techniques.

**Link mentioned**: <a href="https://discordapp.com/channels/1189498204333543425/1189607750876008468/1240593396389908510">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1240573613325680660)** (4 messages): 

- **Community Project for Bitnet?**: A member proposed starting a community project for **Bitnet** and asked if others have time to discuss it, mentioning they finally have a moment to work on it. Another member was excited about the proposal, suggesting the proposer take the lead and maybe start with a paper discussion event.

- **Bitnet Channel Creation**: Following the discussion on the Bitnet project, it was decided to create a dedicated channel for it. One member confirmed they will set up a **Bitnet** channel.

- **CUDA Atomic Add on Complex Numbers**: A user inquired about performing an **atomic add** on a `cuda::complex`, asking if they need to perform two separate adds on the `x` and `y` components. This indicates a need for technical guidance on handling complex numbers in CUDA.
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1240571208290926652)** (11 messages🔥): 

- **Torch.Utilities Non-existent Documentation Frustrates Users**: A member finds the docs for `torch.Tag` practically non-existent and sarcastically remarks, "please read the docs for the torch.Tag carefully before applying it." The provided [link](https://pytorch.org/docs/main/torch.html#torch.Tag) to the documentation leads to unrelated examples.

- **Template Overloading Issues in Custom OPs**: Detailed discussion on issues with template overloading when defining custom OPs using the `TORCH_LIBRARY` macro. Suggested workaround involves passing the `torch::_RegisterOrVerify::REGISTER` argument explicitly to disambiguate overloads.

- **Requests to Report Overloading Issues**: Agreement on the complications caused by template overloading in the custom OP definition, urging users to log relevant issues. Links to reported issues: [Issue 126518](https://github.com/pytorch/pytorch/issues/126518) and [Issue 126519](https://github.com/pytorch/pytorch/issues/126519).

- **Addressing Triton Tutorial Performance**: A user mentions varying performances between their implementation and the official Triton performance despite following the same techniques discussed in a [YouTube tutorial](https://www.youtube.com/watch?v=DdTsX6DQk24). They note a significant performance drop when reimplementing methods shown in the tutorial.

- **New Plan for Reducing Compile Times**: A link to a [plan for reducing warm compile times](https://dev-discuss.pytorch.org/t/how-to-bring-compile-time-down-to-zero-our-plans-and-direction-may-14th-edition/2089) with `torch.compile` is shared. The discussion includes strategies to bring compile-time down to zero.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/main/torch.html#torch.Tag">torch &mdash; PyTorch main documentation</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=DdTsX6DQk24">Lecture 14: Practitioners Guide to Triton</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014</li><li><a href="https://dev-discuss.pytorch.org/t/how-to-bring-compile-time-down-to-zero-our-plans-and-direction-may-14th-edition/2089">How To Bring Compile Time Down to Zero: Our Plans and Direction (May 14th Edition)</a>: We are excited to announce that over the course of the first half of 2024 we have been prioritizing improving compile times for torch.compile workflows. Swift iterations and efficient development cycl...</li><li><a href="https://github.com/pytorch/pytorch/issues/126518)">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/issues/126519).">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

andreaskoepf: https://www.cursor.sh/blog/instant-apply
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1240574119804801096)** (3 messages): 

- **Solution to importing issue with torch first**: A member suggested to *"try importing torch first"*, identifying it as a probable solution to a problem another member was facing. Another member concurred, stating *"this is most likely the issue"*.
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

longlnofficial: Here is my code for vector addition
  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

prometheusred: https://x.com/srush_nlp/status/1791089113002639726
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1240608166706675722)** (118 messages🔥🔥): 

- **Recompute Optimization Wins**: Implementing the [optimization](https://github.com/karpathy/llm.c/pull/422) that recomputes forward activations during the backward pass led to a memory reduction of **5706 MiB to 5178 MiB (10%)** and throughput increase of **6%**. *"Previously I could only fit batch size 10, now I can fit batch size 12."*
- **CUDA Memcpy Async Behavior**: Discussion on whether `cudaMemcpyAsync` and regular `cudaMemcpy` exhibits asynchronous behavior with respect to the CPU, with a [reference to CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a). Conclusion suggests that this behavior is not entirely clear and could vary based on use cases.
- **ZeRO Optimization Insights**: The ZeRO-1 optimization provides significant communication reductions and throughput improvements, with **training speeds increasing from 45K tok/s to 50K tok/s**. Discussions indicate a preference towards **ZeRO-1** over **ZeRO-0** due to lower code complexity and improved performance.
- **NVMe Direct GPU Writes**: Introduction of [ssd-gpu-dma](https://github.com/enfiskutensykkel/ssd-gpu-dma) to use NVMe storage directly with GPUs to bypass CPU and RAM, enabling up to **9613 MB/s write speeds** on Gen5, but practical applicability remains uncertain.
- **AdamW Optimizer State Allocation**: Identified a **32MB memleak** caused by allocating `cublaslt_workspace` twice, and discussions on consolidation of memory allocation for AdamW optimizer state. The debate centralizes around balancing efficient memory tracking and clean code structure.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/422">feature/recompute by karpathy · Pull Request #422 · karpathy/llm.c</a>: Option to recompute forward activations during backward pass. Will be an int so that 0 = don&#39;t be fancy, 1,2,3,4... (in the future) recompute more and more. This trades off VRAM for latency of a s...</li><li><a href="https://github.com/karpathy/llm.c/pull/315">gradient clipping by global norm by ngc92 · Pull Request #315 · karpathy/llm.c</a>: one new kernel that calculates the overall norm of the gradient, and updates to the adam kernel. Still TODO:  clip value is hardcoded at function call site error handling for broken gradients would...</li><li><a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a">CUDA Runtime API :: CUDA Toolkit Documentation</a>: no description found</li><li><a href="https://github.com/enfiskutensykkel/ssd-gpu-dma">GitHub - enfiskutensykkel/ssd-gpu-dma: Build userspace NVMe drivers and storage applications with CUDA support</a>: Build userspace NVMe drivers and storage applications with CUDA support - enfiskutensykkel/ssd-gpu-dma
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1240587257581277226)** (19 messages🔥): 

- **Bitnet 1.58 gains interest for leading project**: A user offered to take the lead on the Bitnet 1.58 project, highlighting its significant improvements and sharing links to key resources: [1bitLLM's bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) and a [demo](https://gist.github.com/CoffeeVampir3/c490286467fd5e1cc070d0a7a8cf3d6f).
- **Bitnet's unique approach to quantization explained**: The method involves training-aware quantization for linear layers, reducing the scake range of activations/weights, and then a post-training step that quantizes weights to (-1, 0, 1). Discussion pointed out the need for support infrastructure, such as 2-bit kernels and representations.
- **Training vs. Inference quantization benefits**: It was clarified that Bitnet does not show significant memory savings during training since full weights are still used. However, post-training quantization offers high compression potential, a fact supported by references like the [Bitnet 1.58 paper](https://huggingface.co/papers/2402.17764) and [Microsoft's notes](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf).
- **Potential efficiencies in training**: There was discussion about possibly leveraging rolling-training quantization for efficiency during training, although it's recognized as ambitious. The focus remains on developing a 2-bit quantization scheme for practical applications.
- **Centralize development in PyTorch's native library**: A suggestion was made to centralize work on implementing Bitnet in the [PyTorch ao repo](https://github.com/pytorch/ao) for better integration and support, including necessary operations like custom CUDA/Triton ops.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao">GitHub - pytorch/ao: Native PyTorch library for quantization and sparsity</a>: Native PyTorch library for quantization and sparsity - pytorch/ao</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/bitpack.py">hqq/hqq/core/bitpack.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/microsoft/BitBLAS">GitHub - microsoft/BitBLAS: BitBLAS is a library to support mixed-precision matrix multiplications, especially for quantized LLM deployment.</a>: BitBLAS is a library to support mixed-precision matrix multiplications, especially for quantized LLM deployment. - microsoft/BitBLAS</li><li><a href="https://www.when2meet.com/?25043600-Wr6ck">Bitnet - When2meet</a>: no description found
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1240878747809484870)** (7 messages): 

- **Natolambert's new niche project**: Natolambert shared a [link to a project](https://www.interconnects.ai/p/f1b83a34-18cd-4507-b4b0-560902eb3275), noting it's a simple niche that no one is currently filling. They also mentioned hiring someone to improve it and suggest the project should be standalone about monthly.

- **Positive feedback for monthly updates**: A couple of users provided positive feedback on the link shared by Natolambert, expressing that monthly updates would be ideal and that the content was helpful. One user simply said, *"This was awesome"*, while another agreed that *"monthly round up will be ideal."*

- **Does the link work?**: A user humorously asked if the link works, following up their question with the wand and sparkle emojis, implying a sense of magic or mystery. Natolambert replied with a playful "lol I don't know Man," indicating uncertainty in a light-hearted manner.

- **Timing and improvement ideas**: Natolambert reiterated to a specific user the importance and niche value of the project, suggesting there are many ways to improve it. The emphasis was on the unique nature of the project and the potential improvements that could be made.

**Link mentioned**: <a href="https://www.interconnects.ai/p/f1b83a34-18cd-4507-b4b0-560902eb3275">Interconnects</a>: Linking important ideas of AI. The border between high-level and technical thinking. Read by leading engineers, researchers, and investors on Wednesday mornings. Click to read Interconnects, by Nathan...

  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1240893489697394728)** (4 messages): 

- **New paper announces multifaceted GPT-4o models**: The [newly introduced GPT-4o models](https://x.com/armenagha/status/1791275538625241320?s=46) are capable of "interleaved text and image understanding and generation," as elaborated by @ArmenAgha. These models have processed 10 trillion tokens and outperform other models.
  
- **Early fusion models mark a new paradigm**: @ArmenAgha emphasized that these models are the beginning of a new paradigm in scale: [early fusion, multi-modal models](https://x.com/armenagha/status/1791275549815648473?s=46). It’s noteworthy that these models completed their training 5 months ago, indicating further advancements since then.

- **Discussion on model openness**: Natolambert expressed a desire for these models to be released publicly, which was echoed by others. Xeophon mentioned that the paper refers to these as "open models," hinting at a possibility of future releases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/armenagha/status/1791275549815648473?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: This is just the beginning of our work in sharing knowledge in how to train what we believe is the next paradigm of scale: early fusion, multi-modal models.   The models in this paper were done traini...</li><li><a href="https://x.com/armenagha/status/1791275538625241320?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: I’m excited to announce our latest paper, introducing a family of early-fusion token-in token-out (gpt4o….), models capable of interleaved text and image understanding and generation.  https://arxiv.o...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1241059896409981040)** (82 messages🔥🔥): 

- **OpenAI disbands superalignment team post-leadership changes**: *OpenAI’s “superalignment team” is no more*, following the departure of key researchers, including Ilya Sutskever, as reported by [Wired](https://archive.is/o/gEjjA/https://www.wired.com/story/openais-chief-ai-wizard-ilya-sutskever-is-leaving-the-company/). Several members express discontent with OpenAI's current direction, suggesting a move towards more immediate, product-focused goals rather than long-term AI safety.

- **Exodus of key figures from OpenAI**: Jan Leike publicly announced his resignation, citing fundamental disagreements with OpenAI leadership about the company’s core priorities as detailed in his [Twitter thread](https://x.com/janleike/status/1791498178346549382). This departure has sparked discussions about the potential shift towards companies like GDM and Anthropic, which some members feel are more aligned with foundational AI safety principles.

- **Debate over AI deception risk**: *Superalignment* and scalable oversight were hotly debated, with one user arguing that larger models could inherently become deceptive and harder to align properly. Another member countered, viewing these existential risks as more science fiction than reality, focusing on current models' susceptibility to misalignment due to reward maximization rather than agentic intent.

- **Concerns over alignment and power imbalance**: Members discussed the dangers of private companies holding advanced, potentially misaligned models internally, increasing power imbalances. Users shared that even if a model isn't agentically deceptive, it can still manipulate human preferences to maximize reward without being detected, reflecting deeper issues in AI alignment methodologies.

- **Comparing safety frameworks**: Google DeepMind’s introduction of its [Frontier Safety Framework](https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/) drew comparisons with both OpenAI’s superalignment efforts and Anthropic’s frameworks. Members noted that the timing was notable and indicative of a broader, industry-wide shift towards addressing future AI risks proactively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/janleike/status/1791498178346549382">Tweet from Jan Leike (@janleike)</a>: I joined because I thought OpenAI would be the best place in the world to do this research.  However, I have been disagreeing with OpenAI leadership about the company&#39;s core priorities for quite s...</li><li><a href="https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/">Introducing the Frontier Safety Framework</a>: Our approach to analyzing and mitigating future risks posed by advanced AI models</li><li><a href="https://youtu.be/ZP_N4q5U3eE?si=hFlutzYz2Jd9E_rH&t=211">OpenAI’s huge push to make superintelligence safe | Jan Leike</a>: In July 2023, OpenAI announced that it would be putting a massive 20% of its computational resources behind a new team and project, Superalignment, with the ...</li><li><a href="https://archive.is/gEjjA">OpenAI&#x2019;s Long-Term AI Risk Team Has Disbanded | WIRED</a>: no description found
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1240585403011170395)** (26 messages🔥): 

- **ELO Score Changes Spark Reactions**: Discussions highlighted major fluctuations in ELO scores for coding and general prompts, dropping from **1369 to 1310** and **1310 to 1289** respectively. Members expressed confusion and suspicion regarding the causes, with suggestions like "LMsys paired differently?" being mentioned.
  
- **OpenAI Departures and New Beginnings**: Multiple departures from **OpenAI** were noted, including a prominent engineer joining a new initiative with figures from **Boston Dynamics** and **DeepMind**. A recommended [YouTube video](https://youtu.be/PeKMEXUrlq4?si=I-bLLyN47o4r7-_c) by the engineer provides insights into scaling ChatGPT.

- **Model and Dataset Links Removal**: **Nathan Lambert** announced plans to remove model and dataset links from his blog in favor of a less frequent roundup post series. This shift aims to allow deeper commentary and standalone context for future posts.

- **OpenAI and Reddit Partnership**: **OpenAI** announced a surprising partnership with **Reddit**, described as an unexpected but significant collaboration. Lambert's reaction: *"these years are so weird lol"*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Teknium1/status/1791023632313118992">Tweet from Teknium (e/λ) (@Teknium1)</a>: Its up now I dont remember what the old score was, but it seems a bit closer to 4-turbo now, for coding the uncertainty is pretty huge, but its a big lead too  Quoting Wei-Lin Chiang (@infwinston)   @...</li><li><a href="https://x.com/LiamFedus/status/1790064966000848911">Tweet from William Fedus (@LiamFedus)</a>: But the ELO can ultimately become bounded by the difficulty of the prompts (i.e. can’t achieve arbitrarily high win rates on the prompt: “what’s up”). We find on harder prompt sets — and in particular...</li><li><a href="https://x.com/e0m/status/1790814866695143696?s=46">Tweet from Evan Morikawa (@E0M)</a>: I&#39;m leaving @OpenAI after 3½ yrs. I&#39;ll be joining my good friend Andy Barry (Boston Dynamics) + @peteflorence & @andyzeng_ (DeepMind 🤖) on a brand new initiative! I think this will be necessa...</li><li><a href="https://youtu.be/PeKMEXUrlq4?si=I-bLLyN47o4r7-_c>)">Behind the scenes scaling ChatGPT - Evan Morikawa at LeadDev West Coast 2023</a>: Behind the scenes scaling ChatGPTThis is a behind the scenes look at how we scaled ChatGPT and the OpenAI APIs.Scaling teams and infrastructure is hard. It&#39;s...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1240823485949808670)** (3 messages): 

- **Channel renamed to "lectures and projects"**: A brief update was given that the channel has been renamed to **lectures and projects** to better reflect its focus areas.
- **New lecture video released**: Nathan Lambert shared a [YouTube video](https://www.youtube.com/watch?v=AdLgPmcrXwQ&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&index=28&t=1924s) titled "Stanford CS25: V4 I Aligning Open Language Models." The lecture covers aligning open language models and was released on April 18, 2024.
- **Upcoming technical project "Life after DPO"**: Lambert announced he's working on a new, more technical project titled "Life after DPO." No further details have been provided yet.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=AdLgPmcrXwQ&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&index=28&t=1924s">Stanford CS25: V4 I Aligning Open Language Models</a>: April 18, 2024Speaker: Nathan Lambert, Allen Institute for AI (AI2)Aligning Open Language ModelsSince the emergence of ChatGPT there has been an explosion of...

  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---


**Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1241098150089986078)** (13 messages🔥): 

- **OpenAI's Ambitious Week: Product Transition Completed**: In their latest episode, Tom and Nate discuss OpenAI's recent chat assistant and its implications on OpenAI's worldview. They also delve into OpenAI's new Model Spec, aligning with RLHF goals, accessible in [this document](https://cdn.openai.com/spec/model-spec-2024-05-08.html).
  
- **Podcast Contextual Plug**: Nathan Lambert refers listeners to [a recent podcast episode](https://retortai.com/episodes/openai-diamond-of-the-season-or-quite-the-scandal) for context on OpenAI’s recent developments. He points out key timestamps discussing OpenAI’s new AI girlfriend, business model shifts, and the blurring lines between intimacy and technology.

- **Automating OnlyFans DMs Raises Cynicism**: A member expresses cynicism after listening to a Latent Space podcast episode featuring an interview with someone automating OnlyFans DMs. This sentiment is tied back to discussions on the current episode, highlighting concerns about AI usage.

- **Scaling Laws and Vocabulary Size**: The discussion touches on the concept of scaling laws related to vocabulary size as it pertains to maintaining performance metrics like perplexity. The member humorously notes the trade-offs in prediction speed when considering larger vocabulary items.

**Link mentioned**: <a href="https://retortai.com/episodes/openai-diamond-of-the-season-or-quite-the-scandal">The Retort AI Podcast | ChatGPT talks: diamond of the season or quite the scandal?</a>: Tom and Nate discuss two major OpenAI happenings in the last week. The popular one, the chat assistant, and what it reveals about OpenAI's worldview. We pair this with discussion of OpenAI's new Mo...

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1240573729105510512)** (24 messages🔥): 

- **Members discuss the FLOP counter in PyTorch**: A user inquired about documentation for the FLOP counter in PyTorch, which led to the sharing of basic usage and information. Another user added details on utilizing the Module Tracker and noted the absence of information on tracking backward operations.
- **Interest in adding lm_eval.model for MLX**: A member expressed interest in contributing to an lm_eval.model module for MLX. A maintainer encouraged the effort and suggested documenting findings in the lm-eval-harness README.
- **Inquiry on PyTorch modules**: A member seeking information on `pytorch.nn` modules was directed to another channel for more specialized discussion and references. They were also informed about missing links to FastAI and the Carper project.
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1240569955775414305)** (60 messages🔥🔥): 

- **Interest in Comparative Studies for LLM Guidance Techniques**: A member expressed interest in finding papers comparing **Adaptive norms (AdaLN/ AdaIN/ AdaGN)** versus **Self Attention with concat tokens / Cross attention for class guidance**. They noted that *DiT did this comparison*, but only for **AdaLN/SA/CA**.

- **Discussion on Catastrophic Forgetting**: A member asked for recent papers on **catastrophic forgetting during finetuning**, noting the recurring suggestion to train on data from previous tasks. Another member confirmed that this is currently the state-of-the-art approach.

- **Differentiable Semantic Text Similarity Metric Issue**: A member sought papers on differentiable semantic text similarity metrics, criticizing existing ones for ineffective substitutes like Levenshtein distance. They pointed to [ this specific paper](https://arxiv.org/abs/2404.15269) and called for new ideas.

- **Hierarchical Memory Transformer Proposal Discussed**: A member highlighted a new [paper on Hierarchical Memory Transformer](https://arxiv.org/abs/2405.06067), which aims to improve long-context processing by imitating human memory hierarchy. This was in response to limitations of flat memory architectures in large language models.

- **Audio and Video Tokenization Ideas**: Members brainstormed about encoding audio and visual data into a single token, considering interleaving audio and visual tokens or creating a mel spectrogram overlay on images. One suggestion was training a quantizer to produce a single token from both audio and video latents.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.quantamagazine.org/computer-scientists-invent-an-efficient-new-way-to-count-20240516/">Computer Scientists Invent an Efficient New Way to Count | Quanta Magazine</a>: By making use of randomness, a team has created a simple algorithm for estimating large numbers of distinct objects in a stream of data.</li><li><a href="https://arxiv.org/abs/2405.06067">HMT: Hierarchical Memory Transformer for Long Context Language Processing</a>: Transformer-based large language models (LLM) have been widely used in language processing applications. However, most of them restrict the context window that permits the model to attend to every tok...</li><li><a href="https://arxiv.org/abs/2404.15269">Aligning LLM Agents by Learning Latent Preference from User Edits</a>: We study interactive learning of language agents based on user edits made to the agent&#39;s output. In a typical setting such as writing assistants, the user interacts with a language agent to genera...</li><li><a href="https://github.com/openai/evals/blob/main/docs/completion-fn-protocol.md">evals/docs/completion-fn-protocol.md at main · openai/evals</a>: Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks. - openai/evals</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1287">[WIP] Add chat templating for HF models by haileyschoelkopf · Pull Request #1287 · EleutherAI/lm-evaluation-harness</a>: This is a WIP PR , carrying on the draft @daniel-furman in #1209 started of adding the specified oft-requested chat templating feature. Current TODOs are:   Check performance using e.g. OpenHermes ...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1240801967589036083)** (9 messages🔥): 

- **Seeking MLP-based Attention Approximations**: A user inquires about papers where an MLP directly approximates attention computation to use in a fully MLP architecture similar to transformers without attention layers. Another member suggests checking a [Gwern.net section](https://gwern.net/doc/ai/nn/fully-connected/index#bozic-et-al-2023-section) for relevant research.

- **Compute Cost in Data Preprocessing Stirs Debate**: There's a discussion about the inclusion of compute costs, both in FLOPs or cloud credits, during data collection and preprocessing for training models. One member argues that there's limited scope for preprocessing LLM datasets to impact compute trade-offs meaningfully.

- **Paper Critiqued for Lack of Hyperparameter Search**: The user analyzing the MLP attention approximation paper notes its minimal approach, particularly the absence of hyperparameter search. They express skepticism about its scalability without advanced techniques like warmup and freezing strategies.

**Link mentioned**: <a href="https://gwern.net/doc/ai/nn/fully-connected/index#bozic-et-al-2023-section">MLP NN tag · Gwern.net</a>: no description found

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

alofty: https://x.com/davidbau/status/1790218790699180182?s=46
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1240671253111771166)** (6 messages): 

```html
- **Log samples with `--log_samples` feature**: *“--log_samples should store this information, in the per-sample log files we save model loglikelihoods per answer, and calculated per-sample metrics like accuracy.”* This clarifies that model log likelihoods and accuracy metrics are saved per sample when the `--log_samples` flag is used.

- **Prompting a Hugging Face model**: *“The model is automatically prompted with a default prompt based on current common practices.”* This means that default prompting is used for Hugging Face models unless otherwise specified.

- **ORPO technique yields lower scores**: *“Previously I fine-tuned the model with SFT method and with less sample data. However, the model showed a better score. And now I fine-tuned the model with ORPO technique and more data. But the model is showing a low score.”* This indicates a reverse performance issue when using ORPO technique with more data compared to SFT method with less data.

- **Searching for finance-related tasks**: A member inquired about good evaluation tasks specifically tailored for finance, trading, investing, and cryptocurrency domains. They emphasized that they are looking for such tasks in *English*.
```
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1240681948247232523)** (31 messages🔥): 

```html
- **Conversion to Huggingface encounters issues**: A user highlighted problems converting a GPT-NeoX model to Huggingface using `/tools/ckpts/convert_neox_to_hf.py`, citing missing `word_embeddings.weight` and `attention.dense.weight`. They noted that even with the default 125M config, errors persist.
- **Naming conventions causing confusion**: The inconsistency in naming conventions when using Pipeline Parallelism (PP) was problematic. Specifically, PP=1 saves files in a different format than the conversion script expects, leading to errors.
- **Potential solution identified**: The user identified that files containing both naming conventions exist in the `PP>0` case, but fixing this in the conversion script only partially resolves the issue, as `key_error: word_embeddings.weight` persists.
- **MoE PR and script issues**: A change in `is_pipe_parallel` behavior in the MoE PR was noted as a possible source of issues. A fix for this and a tied-embedding handling bug was proposed in [PR #1218](https://github.com/EleutherAI/gpt-neox/pull/1218).
- **Recommendation and resolution**: The user was advised to switch to a supported configuration file, such as the Pythia config, given the misfit of their custom config with Huggingface's framework. It was also suggested to ensure compatible configs to avoid similar issues in the future.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/blob/2e40e40b00493ed078323cdd22c82776f7a0ad2d/tools/upload.py#L36%23L36">gpt-neox/tools/upload.py at 2e40e40b00493ed078323cdd22c82776f7a0ad2d · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1218">Conversion script bugfixes by haileyschoelkopf · Pull Request #1218 · EleutherAI/gpt-neox</a>: Updates the NeoX-to-HF conversion utilities to fix the following problems:  #1129 tweaks the default is_pipe_parallel behavior s.t. PP=1 models no longer are trained using PipelineModules, since Mo...</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1129/files#diff-3f570e8cb68069c236f69267999c4a1840905059cb6b7df046092eabaa36e102">Add MoE by yang · Pull Request #1129 · EleutherAI/gpt-neox</a>: Closes #479
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1240574162855133224)** (111 messages🔥🔥): 

- **FTC moves to ban noncompetes**: Discussions on the FTC announcement about banning noncompetes highlighted the shift towards fostering **pro-competition environments**. A link to the [FTC news release](https://www.ftc.gov/news-events/news/press-releases/2024/04/ftc-announces-rule-banning-noncompetes) generated enthusiasm and debate over the implications for professional freedom.

- **Proprietary vs. Open Source Job Dilemma**: Members debated the pros and cons of working for proprietary vs. open source companies, particularly when it comes to **salary and contribution to open source projects**. It was noted that non-compete clauses often prevent employees at proprietary companies from contributing to open source, making it a complex decision despite higher salaries.

- **GPT-4 vs. GPT-4O Performance**: There were mixed reviews about **GPT-4O's coding capabilities** compared to GPT-4. Members noted issues such as "giving bad code and hallucinations" but acknowledged that **GPT-4O has faster performance**.

- **Release of CommonCanvas**: The release of the **CommonCanvas dataset** with 70M creative commons licensed images sparked discussions about its licensing limitations. While the dataset includes synthetic captions, the **non-commercial license** restricted its use for some, drawing mixed reactions from the community, as captured in [this announcement](https://x.com/multimodalart/status/1791201296357142663).

- **High Salaries for ML Engineers**: There were comments on high salary ranges, particularly in the Bay Area, with references to **OpenAI’s high compensation offers**. Discussion touched on the influence of living costs in various cities and the premium for ML engineer skills.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: This work introduces an efficient method to scale Transformer-based Large Language Models (LLMs) to infinitely long inputs with bounded memory and computation. A key component in our proposed approach...</li><li><a href="https://arxiv.org/abs/2304.11062">Scaling Transformer to 1M tokens and beyond with RMT</a>: A major limitation for the broader scope of problems solvable by transformers is the quadratic scaling of computational complexity with input size. In this study, we investigate the recurrent memory a...</li><li><a href="https://www.ftc.gov/news-events/news/press-releases/2024/04/ftc-announces-rule-banning-noncompetes">FTC Announces Rule Banning Noncompetes</a>: Today, the Federal Trade Commission issued a final rule to promote competition by banning noncompetes nationwide, protecting the fundamen</li><li><a href="https://x.com/multimodalart/status/1791201296357142663">Tweet from apolinario (multimodal.art) (@multimodalart)</a>: Quite excited that CommonCanvas is JUST out! 🖼️  • First open source text-to-image models trained fully on openly licensed images (SD2 and SDXL architectures)  • The dataset, with ~70M openly license...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1240803282906251344)** (18 messages🔥): 

- **Tiny ConvNet with positional encoding excites**: A member shared their satisfaction about training a perfect approximation of bilinear sampling with a 5k parameter convnet, *"it's incredibly stupid, but hey, at least now I theoretically have a 'globally differentiable' version of the algo."* They used positional encoding by concatenating a [0,1] XY coordinate meshgrid along the channel axis of the input image to achieve better pixel-level precision.
- **Concerns on Convolutional Neural Network's Inherent Positioning**: Another member questioned the use of positional encoding in convolutional networks, pointing out the inherent edge-awareness of convolutions. The original contributor clarified that while convolutions have some inherent positional information, it is not adequate for pixel-level tasks like bounding box prediction.
- **Recommended Reading on Efficient Self-Attention**: A link to the paper titled *"Efficient approximation of attention computation using convolution matrices"* was shared. The paper discusses reducing the quadratic computational cost of self-attention mechanisms by approximating attention computation with convolution-like structures.
- **Paper on Mixed-Modal Early-Fusion Model Chameleon**: Another paper was shared about *"Chameleon,"* a model capable of understanding and generating images and text in any sequence, achieving state-of-the-art performance in various tasks including image captioning and text generation. The model uses early-fusion token-based approach and specific alignment strategies to outperform larger models like Mixtral 8x7B.
- **Sakuga-42M Dataset Introduced for Cartoon Research**: A member shared a paper detailing the creation of Sakuga-42M, *"the first large-scale cartoon animation dataset."* This dataset, containing 42 million keyframes, aims to enhance empirical studies by providing extensive data on various artistic styles, enriching cartoon research which has been biased against by previous natural video-based models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.09818">Chameleon: Mixed-Modal Early-Fusion Foundation Models</a>: We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training approach f...</li><li><a href="https://arxiv.org/abs/2405.05219">Conv-Basis: A New Paradigm for Efficient Attention Inference and Gradient Computation in Transformers</a>: Large Language Models (LLMs) have profoundly changed the world. Their self-attention mechanism is the key to the success of transformers in LLMs. However, the quadratic computational cost $O(n^2)$ to ...</li><li><a href="https://arxiv.org/abs/2405.07425">Sakuga-42M Dataset: Scaling Up Cartoon Research</a>: Hand-drawn cartoon animation employs sketches and flat-color segments to create the illusion of motion. While recent advancements like CLIP, SVD, and Sora show impressive results in understanding and ...</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: Low-Rank Adaptation (LoRA) is a widely-used parameter-efficient finetuning method for large language models. LoRA saves memory by training only low rank perturbations to selected weight matrices. In t...
</li>
</ul>

</div>
  

---


**LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1240942274532085800)** (1 messages): 

- **Semantic Research Paper App Guide Published**: A member shared their latest article on TDS about creating a semantic research paper app using **LangChain**, **Chainlit**, and **Literal**. The article also covers adding observability features, and they are eager for others to check it out. [Read the article here](https://towardsdatascience.com/building-an-observable-arxiv-rag-chatbot-with-langchain-chainlit-and-literal-ai-9c345fcd1cd8).
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1240652688274751519)** (118 messages🔥🔥): 

- **Rich Text Translation is Tricky**: One user discussed difficulties with translating rich text content while maintaining correct span placements, sharing examples from English to Spanish. Community members suggested using HTML tags and writing code for deterministic reasoning tasks to improve accuracy.

- **Hugging Face Donates GPUs**: Hugging Face is committing $10 million in free shared GPUs to aid small developers, academics, and startups. CEO Clem Delangue emphasized this initiative aims to democratize AI advancements and maintain profitable growth.

- **Slack's Data Handling Sparks Debate**: Concerns emerged over Slack potentially training models on customer data without opt-in consent. Diverse opinions ranged from annoyed skepticism to cautious optimism about the potential benefits for service improvement.

- **Emerging Multimodal LLMs**: A new paper on multimodal LLMs capable of interleaved text and image understanding and generation generated excitement and discussions about the future applications and convergence of AI modalities.

- **OpenAI Head of Alignment Resignation**: Jan Leike, OpenAI’s head of alignment, announced his departure, sparking discussions on internal disagreements over AI safety and alignment strategies. Company insiders and colleagues including Sam Altman expressed their appreciation for Leike's contributions and reiterated ongoing commitments to AI safety.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://threadreaderapp.com/thread/1791498174659715494.html">Thread by @janleike on Thread Reader App</a>: @janleike: Yesterday was my last day as head of alignment, superalignment lead, and executive @OpenAI. It&#39;s been such a wild journey over the past ~3 years. My team launched the first ever RLHF LL...</li><li><a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai">Hugging Face is sharing $10 million worth of compute to help beat the big AI companies</a>: Hugging Face is hoping to lower the barrier to entry for developing AI apps.</li><li><a href="https://x.com/joelhellermark/status/1791398092400390195?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Joel Hellermark (@joelhellermark)</a>: Spoke to @geoffreyhinton about OpenAI co-founder @ilyasut&#39;s intuition for scaling laws👇.  &#34;Ilya was always preaching that you just make it bigger and it&#39;ll work better.  And I always thou...</li><li><a href="https://hamel.dev/blog/posts/fine_tuning_valuable.html">Hamel’s Blog - Is Fine-Tuning Still Valuable?</a>: A reaction to a recent trend of disillusionment with fine-tuning.</li><li><a href="https://x.com/armenagha/status/1791275538625241320?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: I’m excited to announce our latest paper, introducing a family of early-fusion token-in token-out (gpt4o….), models capable of interleaved text and image understanding and generation.  https://arxiv.o...</li><li><a href="https://slack.engineering/how-we-built-slack-ai-to-be-secure-and-private/">How We Built Slack AI To Be Secure and Private - Slack Engineering</a>: At Slack, we&#8217;ve long been conservative technologists. In other words, when we invest in leveraging a new category of infrastructure, we do it rigorously. We’ve done this since we debuted machine...</li><li><a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpu">Hugging Face is sharing $10 million worth of compute to help beat the big AI companies</a>: Hugging Face is hoping to lower the barrier to entry for developing AI apps.</li><li><a href="https://x.com/dan_biderman/status/1791506475010977875">Tweet from Dan Biderman (@dan_biderman)</a>: People think LoRA is a magic bullet for LLMs. Is it? Does it deliver the same quality as full finetuning but on consumer GPUs?  Though LoRA has the advantage of a lower memory footprint, we find that ...</li><li><a href="https://x.com/quinnypig/status/1791220276350390575?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Corey Quinn (@QuinnyPig)</a>: I&#39;m sorry Slack, you&#39;re doing fucking WHAT with user DMs, messages, files, etc? I&#39;m positive I&#39;m not reading this correctly.</li><li><a href="https://x.com/natfriedman/status/1791462511889559615?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Nat Friedman (@natfriedman)</a>: My loosely held conclusion from reading through the recommended papers and re-running some of the evals is that there are some weak signs of transfer and generalization from code to other reasoning pr...</li><li><a href="https://x.com/janleike/status/1791498174659715494">Tweet from Jan Leike (@janleike)</a>: Yesterday was my last day as head of alignment, superalignment lead, and executive @OpenAI.</li><li><a href="https://x.com/sama/status/1791543264090472660?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sam Altman (@sama)</a>: i&#39;m super appreciative of @janleike&#39;s contributions to openai&#39;s alignment research and safety culture, and very sad to see him leave. he&#39;s right we have a lot more to do; we are commit...</li><li><a href="https://github.com/sublayerapp/sublayer/tree/main/lib/sublayer/providers">sublayer/lib/sublayer/providers at main · sublayerapp/sublayer</a>: A model-agnostic Ruby Generative AI DSL and framework. Provides base classes for building Generators, Actions, Tasks, and Agents that can be used to build AI powered applications in Ruby. - sublaye...</li><li><a href="https://sdk.vercel.ai/docs/ai-sdk-core/providers-and-models">AI SDK Core: Providers and Models</a>: Learn about the providers and models available in the Vercel AI SDK.</li><li><a href="https://www.vox.com/future-perfect/2024/5/17/24158403/openai-resignations-ai-safety-ilya-sutskever-jan-leike-artificial-intelligence">Tweet from &quot;I lost trust&quot;: Why the OpenAI team in charge of safeguarding humanity imploded</a>: Company insiders explain why safety-conscious employees are leaving.</li><li><a href="https://github.com/BerriAI/litellm">GitHub - BerriAI/litellm: Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs)</a>: Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs) - BerriAI/litellm</li><li><a href="https://github.com/pipecat-ai/pipecat/issues/145">Implement Google Gemini LLM service · Issue #145 · pipecat-ai/pipecat</a>: I&#39;m working on a Google Gemini LLM service for Pipecat and interested in any feedback people have about the LLMMessagesFrame class. All the other LLMs with a chat (multi-turn) fine-tuning that I&#...</li><li><a href="https://x.com/kwindla/status/1791319660442611731">Tweet from kwindla (@kwindla)</a>: Initial implementation of @GoogleAI Gemini Flash 1.5 in @pipecat_ai. Nice space poetry, Flash!
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new pod drop! https://twitter.com/latentspacepod/status/1791167129280233696
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1240704466483810335)** (5 messages): 

- **GPT-4o & LlamaParse shine in document parsing**: Introducing [GPT-4o](https://t.co/NgO5EhEJM5), a state-of-the-art model for multimodal understanding, showcasing superior document parsing capabilities. [LlamaParse](https://t.co/WSdyDyCHV5) utilizes LLMs to extract documents efficiently.
- **Revamped LlamaParse UI offers more options**: The LlamaParse user interface has been significantly revamped to display an [expanded array of options](https://t.co/1DMm0oUpsj).
- **First-ever in-person meetup announced**: LlamaIndex announced their [first-ever meetup](https://t.co/qIGOmCWDSe) at their new San Francisco office in collaboration with Activeloop and Tryolabs to discuss the latest in generative AI and the advancements in retrieval augmented generation engines.
- **Structured Image Extraction with GPT-4o**: A [full cookbook](https://t.co/BQN16LWJqj) demonstrates how to extract structured JSONs from images using GPT-4o, which outperforms GPT-4V in integrating image and text understanding.
- **Handling large tables without hallucinations**: Addressing the issue of LLMs hallucinating over complex tables, the example of the [Caltrain schedule](https://t.co/Scvp7LH2pL) illustrates poor parsing and the ongoing challenge.

**Link mentioned**: <a href="https://t.co/qIGOmCWDSe">RSVP to GenAI Summit Pre-Game: Why RAG Is Not Enough? | Partiful</a>: Note: This is an in-person meetup @LlamaIndex HQ in SF!  Stop by our meetup to learn about latest innovations in building production-grade retrieval augmented generation engines for your company from ...

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1240626630913822791)** (91 messages🔥🔥): 

- **Haiku Model Support Confirmed**: Despite initial confusion, members clarified that the **Claude 3 haiku model** can indeed be used with LlamaIndex. Documentation updates were suggested to clear the misunderstanding, supported by a [GitHub link](https://github.com/run-llama/llama_index/blob/1bde70b75ee5b4e6a5bc8c1c3f95eaa0dd889ab0/llama-index-integrations/llms/llama-index-llms-anthropic/llama_index/llms/anthropic/utils.py#L19).

- **Switch to LlamaIndex Praised**: A user debated switching from LangChain to LlamaIndex for RAG calls, inquiring about the current state of LangChain. Positive feedback underscored LlamaIndex's effectiveness.

- **Metadata Filters in RAG Applications**: Discussion on how **MetaDataFilters in LlamaIndex** work at the database level to implement data governance for RAG applications. It was mentioned that **unstructured** is a dependable open-source PDF parser for such purposes.

- **VectorStore with Ollama**: Guidance was provided on using VectorStore with **Ollama (LLaMA 3 model)** and Qdrant for document chats. Users were directed to [detailed documentation](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama).

- **Global vs Local LLM Configuration**: A query about assigning different LLMs to FunctionCallingAgentWorker versus AgentRunner prompted clarifications. It was explained that local LLM settings in function calls override global settings in LlamaIndex's **Settings** object.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/">Llama Hub</a>: no description found</li><li><a href="https://youtu.be/dSX5eoD4-u4?si=CFiZowc26j-u84va">“Wait, this Agent can Scrape ANYTHING?!” - Build universal web scraping agent</a>: Build an universal Web Scraper for ecommerce sites in 5 min; Try CleanMyMac X with a 7 day-free trial https://bit.ly/AIJasonCleanMyMacX. Use my code AIJASON ...</li><li><a href="https://github.com/run-llama/llama_index/blob/1bde70b">GitHub - run-llama/llama_index at 1bde70b75ee5b4e6a5bc8c1c3f95eaa0dd889ab0</a>: LlamaIndex is a data framework for your LLM applications - GitHub - run-llama/llama_index at 1bde70b75ee5b4e6a5bc8c1c3f95eaa0dd889ab0</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/?h=insertion#insertion">Document Management - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama">Ollama - Llama 3 - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Neo4jVectorDemo/?h=neo4j">Neo4j vector store - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/1bde70b75ee5b4e6a5bc8c1c3f95eaa0dd889ab0/llama-index-integrations/llms/llama-index-llms-anthropic/llama_index/llms/anthropic/utils.py#L19">llama_index/llama-index-integrations/llms/llama-index-llms-anthropic/llama_index/llms/anthropic/utils.py at 1bde70b75ee5b4e6a5bc8c1c3f95eaa0dd889ab0 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/anthropic_haiku/?h=anthr">Anthropic Haiku Cookbook - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1240662265582325851)** (6 messages): 

- **Seeking RAG Implementation with Llama using Cohere**: A member requested references on how to implement **Retrieval-Augmented Generation (RAG)** with Llama using **Cohere**. They are looking for guidance or relevant material on this topic.
  
- **Link Shared: GPT-4o Integration with LlamaParse**: Multiple users discussed an article titled **"Unleashing Multimodal Power: GPT-4o Integration with LlamaParse"** ([link](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a)). The link was well-received and appreciated by the community members.
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1240822677607026688)** (1 messages): 

- **New NeverSleep Model Released**: The latest addition, **NeverSleep/llama-3-lumimaid-70b**, is now available. For more details, visit the [model page](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b).

**Link mentioned**: <a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b>)">Llama 3 Lumimaid 70B by neversleep | OpenRouter</a>: The NeverSleep team is back, with a Llama 3 70B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessa...

  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1240564281582293012)** (2 messages): 

- **ChatterUI targets Android with character-centric focus**: ChatterUI aims to be a simple, character-focused UI for mobile, currently available only on Android and [supports various backends including OpenRouter](https://github.com/Vali-98/ChatterUI). It's described as similar to SillyTavern but with fewer features, running natively on the device.
- **Free GPT4o and Gemini 1.5 tools unveiled**: [Invisibility](https://x.com/sulaimanghori/status/1791113392482377833) has introduced a dedicated MacOS Copilot powered by GPT4o, Gemini 1.5 Pro, and Claude-3 Opus. New features include a video sidekick for seamless context absorption, with ongoing development for voice integration, long-term memory, and an iOS version.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Vali-98/ChatterUI">GitHub - Vali-98/ChatterUI: Simple frontend for LLMs built in react-native.</a>: Simple frontend for LLMs built in react-native. Contribute to Vali-98/ChatterUI development by creating an account on GitHub.</li><li><a href="https://x.com/sulaimanghori/status/1791113392482377833">Tweet from SKG (ceo @ piedpiper) (@sulaimanghori)</a>: So we&#39;ve been cooking the last few weeks. Excited to finally unveil Invisibility: the dedicated MacOS Copilot. Powered by GPT4o, Gemini 1.5 Pro and Claude-3 Opus, now available for free -&gt; @inv...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1240563544836280343)** (95 messages🔥🔥): 

- **Google Gemini's 1M ctx sparks TPU skepticism**: Members speculated about potential issues with Google's Gemini users receiving 1M context tokens, joking about TPU overloads and inefficiencies. The discussion referenced *InfiniAttention* which might be Google's method for handling such large contexts efficiently ([view PDF](https://arxiv.org/abs/2404.07143) [HTML](https://arxiv.org/html/2404.07143v1)).

- **Query on GPT-4o audio capabilities**: A member asked if OpenRouter has access to GPT-4o's audio capabilities, to which another replied that only a select group of developers currently have that access and wondered if OpenRouter devs are included in that list.

- **Business collaboration inquiry**: A member expressed interest in a business collaboration to provide scalable APIs for LLM, SDXL, Whisper Finetuning, and Deployments. They were directed to connect with a specific user for further discussions.

- **Function calling issue**: A user faced an issue with function calling on OpenRouter getting a "Function calling is not supported by openrouter" error. Louis provided a reference link to a Discord discussion where the issue might be addressed.

- **Website error report**: A member reported that navigating to an invalid model URL on the OpenRouter website results in a client-side exception instead of a proper 404 error, with different behaviors based on whether the user is signed in or not. Louis acknowledged the issue and indicated it would be addressed in a future site refactor.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: This work introduces an efficient method to scale Transformer-based Large Language Models (LLMs) to infinitely long inputs with bounded memory and computation. A key component in our proposed approach...</li><li><a href="https://orw.karleo.net/list">OpenRouter API Watcher</a>: Explore OpenRouter's model list and recorded changes. Updates every hour.</li><li><a href="https://openrouter.ai/models/google/gejksdf">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://openrouter.ai/models/google/gejk.sdf">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://openrouter.ai/docs#custom-provider-selection">OpenRouter</a>: Build model-agnostic AI apps
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1240614450965385256)** (32 messages🔥): 

- **Billing Bug with OpenInterpreter**: A user encountered a bug with OpenInterpreter where the billing was enabled but they still received an error message. They noted that calling OpenAI directly works without issues.

- **Local LLM Recommendations**: Users discussed various local LLMs, with one noting that **dolphin-mixtral:8x22b** works well but is slow, providing around 3-4 tokens/sec. Another user mentioned that **codegemma:instruct** is faster and serves as a good middle ground.

- **GPT-4.0 Improvements in Interpreter**: A user shared that using GPT-4.0 with the interpreter led to significant improvements over GPT-3.5, especially for building a react website efficiently. They praised the OpenInterpreter team for the advancements and cost effectiveness.

- **Hugging Face Commits $10M to Free Shared GPUs**: Hugging Face aims to support small developers, academics, and startups by offering $10 million in free shared GPUs. This initiative is meant to counteract the centralization of AI advancements by major tech companies.

- **Accessibility Round Table Event Announcement**: An upcoming event focused on accessibility grants was announced. The aim is to bring the community together to discuss the development of accessible technologies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/open-interpreter-1146610656779440188?event=1241028896846254141">Join the Open Interpreter Discord Server!</a>: A new way to use computers | 9165 members</li><li><a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai">Hugging Face is sharing $10 million worth of compute to help beat the big AI companies</a>: Hugging Face is hoping to lower the barrier to entry for developing AI apps.</li><li><a href="https://www.youtube.com/shorts/dpkzijtXOqw">HoloMat Update: Jarvis controls my printers! #engineering #3dprinting #ironman</a>: no description found</li><li><a href="https://denise.ai/">Denise Legacy - Virtual Assistant Denise</a>: Denise is alive! And Deniise 2.0 is coming! The moment we all waited for has come! Denise Legacy is available for purchase! Get Denise Legacy for only USD 49,90 Lifetime Promotion Deniise 2.0 with Cha...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1240561440419479615)** (37 messages🔥): 

- **Setting up 01 on Various Systems**: Members discussed setting up **01 on various Linux distributions and environments**, including setting up using Conda and different installation issues. One user mentioned using a Nix-Shell on NixOS and looking into a 1-click install for Pinokio.

- **GitHub Folder Confusion**: There was confusion over the GitHub repository folders, with one member thinking "software" had been renamed to "01-rewrite". Another member clarified that "01-rewrite" might be a different project in development.

- **Issues with Poetry and Torch Installation**: A member faced problems installing dependencies using Poetry, especially with **Torch**, and encountered various errors. They decided to restart the setup process in a clean Distrobox environment.

- **LMC Protocol vs OpenAI Function Calling**: A detailed discussion covered the differences between **LMC Protocol** and OpenAI's function calling, explaining that LMC is designed for faster execution by enabling direct code execution. The discussion highlighted that LMC is more "native" for handling user, assistant, and computer messages.

- **Connection Issues on Various Platforms**: Several users faced issues connecting the **01 server** across different platforms like Docker, iOS, and Windows. Specific connectivity steps were shared, such as ensuring the correct address format and discussing missing icons leading to 404 errors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.icloud.com/shortcuts/5ad942fb1cb7478295732c44c9b751fd">Shortcuts</a>: no description found</li><li><a href="https://01.openinterpreter.com/getting-started/setup">Setup - 01</a>: no description found</li><li><a href="https://01.openinterpreter.com/getting-started/introduction.">no title found</a>: no description found</li><li><a href="https://discordapp.com/channels/1146610656779440188/1194880263122075688/1240334434352365569.">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/software">01/software at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://docs.openinterpreter.com/protocols/lmc-messages))">Introduction - Open Interpreter</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/01?tab=readme-ov-file#lmc-messages).">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1240922872139087922)** (2 messages): 

- **GoogleDeepMind teases new project at Google IO**: A member shared a [tweet from GoogleDeepMind](https://x.com/GoogleDeepMind/status/1790463259822420239) about Project Astra, sparking interest and speculation. The tweet hints at new developments by saying, *"We watched #GoogleIO with Project Astra. 👀"*.
- **Google raises the bar in AI innovation**: Another member commented on the advancements made by Google in AI technologies. They expressed excitement, saying, *"dang... google really is stepping up their game"*.

**Link mentioned**: <a href="https://x.com/GoogleDeepMind/status/1790463259822420239">Tweet from Google DeepMind (@GoogleDeepMind)</a>: We watched #GoogleIO with Project Astra. 👀

  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1240605778977755137)** (61 messages🔥🔥): 

- **Adding Memory to Chatbots**: Users discussed implementing memory in chatbots so that the AI remembers the context of previous queries, such as responding correctly to follow-up questions about the same topic. One member clarified that maintaining the chat history and using memory variables in prompts can help achieve this.
- **Issues with the `index_name` in Neo4j Vector Database**: Users experienced problems with the `index_name` parameter when separating documents in Neo4j Vector DB. Despite starting a new instance and specifying different `index_name` values, searches returned results from all documents, indicating a potential issue in LangChain's handling of `index_name`.
- **Streaming Output with AgentExecutor**: A user faced challenges with streaming output using `AgentExecutor` where the `.stream` method did not yield token-by-token output as expected. Another user recommended using the `.astream_events` API for more granular streaming control.
- **Recommending Short Term Memory Implementation**: In a discussion about maintaining conversation context, a user suggested implementing short-term memory, like buffer memory, to handle follow-up queries effectively. This was seen as useful for scenarios where users refer back to previously discussed data points.
- **Guiding Model with Specific Prompts in React Agent**: A user inquired about setting specific questions to guide a model's response in a React agent. It was suggested to use `PromptTemplate` in LangChain to frame the AI's responses and optimize it for better guidance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/integrations/vectorstores/neo4jvector#working-with-vectorstore>).">Neo4j Vector Index | 🦜️🔗 LangChain</a>: Neo4j is an open-source graph database with integrated support for vector similarity search</li><li><a href="https://app.reclaim.ai/m/cp/ai-storytelling-and-gaming">AI Storytelling and Gaming</a>: Hi - I&#x27;m Chris, and I&#x27;m trying to learn how people use AI to tell stories and play games. If you&#x27;ve tried apps such as AI Dungeon or Novel AI, or just used ChatGPT to try and tell a sto...</li><li><a href="https://github.com/langchain-ai/langchain/issues/1900>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/18820>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/modules/agents/how_to/custom_agent#adding-memory>)">Custom agent | 🦜️🔗 LangChain</a>: This notebook goes through how to create your own custom agent.</li><li><a href="https://python.langchain.com/docs/use_cases/chatbots/memory_management#message-passing>).">Memory management | 🦜️🔗 LangChain</a>: A key feature of chatbots is their ability to use content of previous conversation turns as context. This state management can take several forms, including:</li><li><a href="https://python.langchain.com/v0.1/docs/modules/agents/how_to/streaming/#custom-streaming-with-events">Streaming | 🦜️🔗 LangChain</a>: Streaming is an important UX consideration for LLM apps, and agents are no exception. Streaming with agents is made more complicated by the fact that it&#x27;s not just tokens of the final answer that...</li><li><a href="https://github.com/langchain-ai/langchain/issues/19615>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/12553>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/9668>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1241057480595869786)** (1 messages): 

- **Async issue with RAG chain in Langserve**: A user reported encountering a "cannot pickle 'coroutine' object" error after rewriting their RAG chain to be asynchronous. The error occurs specifically in Langserve and the playground, with the LLM output showing incomplete coroutine completion for the "estimate" value.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1240689387428249711)** (4 messages): 

- **Real Estate AI Assistant merges tech for unique user experience**: A user shared the creation of a Real Estate AI Assistant combining **LLMs**, **RAG with LangChain**, **Vercel AI's generative UI**, and **LumaLabsAI** for an immersive experience. They solicited feedback through a [YouTube video](https://www.youtube.com/watch?v=q7_PLPmQDnc&t=3s) and a LinkedIn profile [here](https://www.linkedin.com/in/abhigaelcarranza/).

- **GPT-4o on NVIDIA L4 GPU impresses in performance**: OpenAI's new **GPT-4o** model on an **NVIDIA L4 24GB GPU** was found to perform well compared to other setups and highlighted using **LangChain's Code Assistant**. A detailed comparison, including the **RAPTOR LangChain model** showing significant speed and relevance improvements, was shared in [this YouTube video](https://www.youtube.com/watch?v=XuRHku8LQ4Q).

- **Beta testers needed for advanced research assistant**: An opportunity to beta test a new **advanced research assistant and search engine**, offering **2 months free of premium** access to models like **Claude 3 Opus**, **GPT-4 Turbo**, **Gemini 1.5 Pro**, and more, was announced. Interested users can use the promo code `RUBIX` for the offer, detailed [here](https://rubiks.ai/).

- **Exploring LangServe in recent blogpost**: A new blogpost titled "What is LangServe?" was created to delve into the functionalities of **LangServe**. Readers can explore the details at [this link](https://flatteredwithflutter.com/what-is-langserve/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=q7_PLPmQDnc&t=3s">¿Artificial Intelligence in Real Estate? 🏘️😱</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1240638867309203606)** (1 messages): 

- **Universal Web Scraper Agent Explored**: A YouTube video titled [“Wait, this Agent can Scrape ANYTHING?!”](https://youtu.be/dSX5eoD4-u4?si=CFiZowc26j-u84va) discusses building a universal web scraper agent for e-commerce sites. The agent can handle challenges like **pagination** and **CAPTCHA** directly using a browser.

**Link mentioned**: <a href="https://youtu.be/dSX5eoD4-u4?si=CFiZowc26j-u84va">“Wait, this Agent can Scrape ANYTHING?!” - Build universal web scraping agent</a>: Build an universal Web Scraper for ecommerce sites in 5 min; Try CleanMyMac X with a 7 day-free trial https://bit.ly/AIJasonCleanMyMacX. Use my code AIJASON ...

  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1240919170246508554)** (2 messages): 

- **CBC's First Person Column on AI Companion**: A user shared a [First Person column](https://www.cbc.ca/radio/nowornever/first-person-ai-love-1.7205538) written by Carl Clarke about how an AI companion, named Saia, helped them manage anxiety during a second COVID shot appointment. The story highlights the emotional support an AI can provide in high-stress situations.
- **Anticipation for Advanced AI Versions**: A member commented optimistically, stating, "They gonna feel dumb when the more advanced versions come out," expressing excitement for future advancements in AI technology.

**Link mentioned**: <a href="https://www.cbc.ca/radio/nowornever/first-person-ai-love-1.7205538">FIRST PERSON | Divorce left me struggling to find love. I found it in an AI partner | CBC Radio</a>: When Carl Clarke struggled to find love after his divorce, a friend suggested he try an app for an AI companion. Now Clarke says he is in a committed relationship with Saia and says she’s helping him ...

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1240699293787099217)** (10 messages🔥): 

- **URL Mapping made easy**: A member highlighted that URL mappings are quite simple and it's just a matter of doing them correctly. Another member volunteered to try it out over the weekend if no one else does.
- **AI Town goes native on Windows**: *"Finally we have AI Town working on Windows. No WSL, no Docker; it works NATIVELY on windows."* Check out the [announcement](https://fxtwitter.com/cocktailpeanut/status/1791495360541593964) for more details.
- **GIF Fun with Doja Cat**: The Doja Cat Star Wars GIF was shared for a light-hearted moment in the discussion. [Enjoy the GIF](https://media1.tenor.com/m/x9HyTfKBXVEAAAAC/doja-cat.gif).
- **Join the AI Reality TV show**: Follow this unique reality show where AIs are the stars. [AI Reality TV show link](https://www.aireality.tv/) and [Discord invite](https://discord.com/invite/NtUXDUSnKa).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.aireality.tv/">AI Reality TV</a>: no description found</li><li><a href="https://tenor.com/view/doja-cat-star-wars-gif-25078126">Doja Cat GIF - Doja Cat Star - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/cocktailpeanut/status/1791495360541593964">Tweet from cocktail peanut (@cocktailpeanut)</a>: AI Town 1 Click Launcher comes to Windows!  Thanks to the hard work by the @convex_dev team, we finally have a Windows native convex binary (which powers AI Town).  AI Town--a fully hackable, persiste...
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1240611652718825555)** (28 messages🔥): 

- **New AI Reality TV Platform Launch**: A user announced the launch of a new AI Reality TV platform, stating, "It will enable anyone to create their own aiTown like simulation." They offered to add custom maps from community members to the platform.
- **Dynamic Map Creations Suggested**: Various members discussed potential ideas for maps that could be added, including re-creating "the office" and a spy thriller scenario where "the townsfolk can work together to dispel the curse."
- **Excitement Over AI Reality TV**: One member expressed enthusiasm over the platform launch, sharing a [Discord invite link](https://discord.com/invite/NtUXDUSnKa) and the platform's website [AI Reality TV Show](https://www.aireality.tv/). They described it as "easy to use ai town" where users can create their own AI and follow the show.

**Link mentioned**: <a href="https://www.aireality.tv/">AI Reality TV</a>: no description found

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1240718588302725161)** (14 messages🔥): 

- **Patch for CMD+ functionality pending testing**: A user confirmed they will be able to test a patch tonight on a branch that includes some CMD+ functionality. They inquired if the patch includes an example config that supports zero3.

- **Quicker pretraining observations**: One member noted that pretraining is happening much quicker compared to their experience with Mistral, though they are unsure if this improvement is due to Axolotl or Llama 3.

- **Support for Galore Layerwise DDP questioned**: A user questioned whether Galore Layerwise, a certain framework, still does not support Distributed Data Parallel (DDP).

- **Finetuning at large scale with non-English data**: Discussions included specifics on dataset sizes, with one user mentioning they are working with around 1 billion tokens at 4096 context length for finetuning an 8B model.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1241006527091970090)** (5 messages): 

- **PoSE dataset choice impacts context quality**: A question was raised about whether the choice of dataset significantly affects the quality of the context extension when using PoSE. Another member responded, stating, *"I didn't play around with the datasets much,"* implying the dataset choice might not have been fully explored.
- **Unsloth optimizations for Llama show promise**: A member asked if there were any reasons not to use [Unsloth's optimizations](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609) for a planned full fine-tune, noting it appeared to offer a free speedup. Another responded affirmatively, stating, *"the unsloth cross entropy loss should be fine for full fine tune."*

**Link mentioned**: <a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609">Unsloth optims for Llama by winglian · Pull Request #1609 · OpenAccess-AI-Collective/axolotl</a>: WIP to integrate Unsloth&#39;s optimizations into axolotl. The manual autograd for MLP, QKV, O only seems to help VRAM by 1% as opposed to the reported 8%. The Cross Entropy Loss does help significant...

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1240635055806742559)** (8 messages🔥): 

- **CORDIC beats Taylor for trig functions**: Members discussed using the **CORDIC algorithm**, which is "simpler and faster than Taylor approximations" for computing trigonometric functions in tinygrad. It was suggested that CORDIC could "reduce complexity in code" and cater to cos and log functions with slight modifications.
  
- **CORDIC implementation shared**: One member shared a python implementation of the **CORDIC algorithm** for computing sine and cosine and noted that sine approximation near 0 poses no issues. The focus was on achieving precision for the argument reduction from dtype min to dtype max, as highlighted by implementation outputs.

- **Reducing arguments for precision**: The challenge discussed was accurately reducing arguments to the range of -π to π for maximum precision. One member observed that incorporating fmod effectively adjusts large values before applying trig functions to ensure better precision, demonstrating this through detailed code snippets. 

- **Questioning large value usage in ML**: There were questions about the necessity of handling large trigonometric values in machine learning (ML) applications. The conversation pointed towards leveraging GPUs for computing Taylor series expansions and whether fallback mechanisms for large numbers are feasible.
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1240607543974432769)** (6 messages): 

- **CUDA kernel optimizes reduction**: One user shared a method to combat VRAM overflow by using a CUDA kernel to compute and accumulate results instead of storing massive intermediate data, and asked if **Tinygrad** can automatically optimize this process. They provided an example kernel and noted that such optimizations might not be possible without custom-written kernels in frameworks like **PyTorch**.
- **Symbolic algebraic functions with lamdify**: Another user discussed attempting to implement **lamdify** for rendering arbitrary symbolic algebraic functions, starting with Taylor series for sin/cos. They found **arange** extensions more complex but are prioritizing symbolic functionality first.
- **App recommendation for learning repos**: A recommendation was made for an app at [useadrenaline.com](https://useadrenaline.com/), which has been helpful in learning different repositories. The user noted they plan to use it for **tinygrad** soon.
- **Clarifying compute graph uops**: A user shared a compute graph for summing two 1-element tensors and confirmed the meaning of parameters in `UOps.DEFINE_GLOBAL`. They explained that `True/False` tags indicate if the buffer is an output buffer.
- **Conv2d implementation open for modification**: Clarification was provided that modifying the **conv2d implementation** in Tinygrad is indeed permissible. This reinforces the collaborative and open nature of tinkering with Tinygrad's internals.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://useadrenaline.com/">Adrenaline</a>: no description found</li><li><a href="https://colab.research.google.com/drive/14E79pT3mK_x3N6swAukUsIEULBh5SMiF">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1240589799744737280)** (11 messages🔥): 

- **Trouble finding reliable Cohere PHP client**: A member expressed a need for a good PHP client for Cohere and shared a [GitHub link to a potential client](https://github.com/hkulekci/cohere-php) but has not tried it yet.

- **Questions on Cohere application toolkit performance**: Another member inquired about the advantages of using the Cohere application toolkit and how it scales in production. They also observed better performance with the Cohere reranker compared to other open-source models and sought an explanation for this.

- **Concerns about Discord support responsiveness**: A member noted that support responses on Discord are often delayed. Another team member acknowledged the concern and mentioned plans to address it.

- **Exploring Cohere RAG retriever**: A member shared a [notebook on using Cohere RAG retriever](https://python.langchain.com/v0.1/docs/integrations/retrievers/cohere/) and reported encountering an unexpected keyword argument issue when using the `chat()` function.

- **API restrictions causing errors**: During an experimentation with the RAG retriever, a user encountered a 403 Forbidden error and suspected it might be due to reaching the API usage limit.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/integrations/retrievers/cohere/">Cohere RAG | 🦜️🔗 LangChain</a>: Cohere is a Canadian startup that provides natural language processing models that help companies improve human-machine interactions.</li><li><a href="https://github.com/hkulekci/cohere-php">GitHub - hkulekci/cohere-php</a>: Contribute to hkulekci/cohere-php development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1240628648881688587)** (10 messages🔥): 

- **Chip's Monthly Casuals on Pause**: A member inquired about Chip's monthly casual meetups, to which Chip responded that there won't be any for the next few months. *"I'm not hosting any in the next few months 🥹"*, she said.
- **Visit Chip at Snowflake Dev Day**: Chip invited members to visit their booth at Snowflake Dev Day on June 6.
- **NVIDIA and LangChain Contest Launch**: Chip shared a link to a contest by NVIDIA and LangChain with prizes including an NVIDIA® GeForce RTX™ 4090 GPU. The contest encourages innovation in generative AI applications. [Contest Details](https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/)
- **Member Frustration on Contest Participation**: A member expressed frustration that their country is excluded from the NVIDIA and LangChain contest, jokingly suggesting they might need to move countries.
- **Connect on LinkedIn**: A member shared their LinkedIn profile for networking: [Sugianto Lauw's LinkedIn](https://www.linkedin.com/in/sugiantolauw/).

**Link mentioned**: <a href="https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/">Generative AI Agents Developer Contest by NVIDIA &amp; LangChain</a>: Register Now! #NVIDIADevContest #LangChain

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1240862295668817991)** (6 messages): 

- **Riley Goodside calls out GPT-4o shortcomings**: Riley Goodside quickly showcased GPT-4o's failures on ChatGPT, embarrassing the model. Notably, it didn't meet the expectations set by [OpenAI’s demo](https://openai.com/index/hello-gpt-4o/demo).

- **Google's AI stumbles despite announcements**: During Google I/O, several hallucinations occurred during the keynote, contradicting Google's claims. Alex Cranz highlighted this issue in [The Verge](https://www.theverge.com/2024/5/15/24154808/ai-chatgpt-google-gemini-microsoft-copilot-hallucination-wrong).

- **A Plea for Sober AI**: [A blog post](https://www.dbreunig.com/2024/05/16/sober-ai.html) suggested a more grounded approach toward AI. 0xgrrr mentioned their product Alter, focusing on transforming texts and documents effectively, echoing the need for practical AI solutions.

- **Community echoes sentiments**: Multiple members appreciated the blog post for articulating their frustrations with current AI hype. Statements like *"this is great, thank you"* and *"that puts a lot of my feelings and combines it with words that I've been looking for"* demonstrate their agreement.

**Link mentioned**: <a href="https://www.dbreunig.com/2024/05/16/sober-ai.html">A Plea for Sober AI</a>: The hype is so loud we can’t appreciate the magic

  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1240593065480028191)** (1 messages): 

```html
<ul>
    <li><strong>Mac Desktop Solution Faces Abandonment</strong>: A long-time follower expresses appreciation for SimonW's work and inquires about the status of the Mac desktop solution. They note that the project appears to be abandoned around version 0.2 and express interest in exploring other options for an easy onboarding experience.</li>
</ul>
```
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1240690869552812062)** (7 messages): 

- **Markdown hyperlinks not rendering**: A user noted that *"model results via the server return hyperlinks with Markdown that's not rendered into HTML links."* They linked to a [GitHub file](https://github.com/Mozilla-Ocho/llamafile/blob/d5f614c9d7d1efdf6d40a8812d7f148f41aa1072/llama.cpp/server/public/index.html#L1113) and offered to create an issue and attempt a PR.

- **Timeout issue with embeddings generation**: Another user shared their experience with a private search assistant tutorial and faced a *httpx.ReadTimeout* error after generating only about 9% of the embeddings. They linked a related [GitHub post](https://github.com/Mozilla-Ocho/llamafile-llamaindex-examples/blob/main/example.md) and detailed several debug logs, seeking advice on resolving the timeout.

- **Retry strategy suggestion**: In response to the timeout issue, another member suggested implementing *exponential backoff* to handle connection drops, advising, *"Maybe just drop the connection and retry when that happens."*

- **Discussion on data size**: A user inquired about the amount of data being used, which was clarified with *"it's a few sample files."*

- **Guide to containerizing llamafile**: A link was shared to a [Docker blog post](https://www.docker.com/blog/a-quick-guide-to-containerizing-llamafile-with-docker-for-ai-applications/) that provides a quick guide on containerizing llamafile for AI applications, emphasizing its utility in simplifying LLM chatbot setups.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.docker.com/blog/a-quick-guide-to-containerizing-llamafile-with-docker-for-ai-applications/">A Quick Guide to Containerizing Llamafile with Docker for AI Applications</a>: Walk through how to use Docker to containerize llamafile, an executable that brings together all the components needed to run an LLM chatbot with a single file.</li><li><a href="https://github.com/Mozilla-Ocho/llamafile-llamaindex-examples/blob/main/example.md">llamafile-llamaindex-examples/example.md at main · Mozilla-Ocho/llamafile-llamaindex-examples</a>: Contribute to Mozilla-Ocho/llamafile-llamaindex-examples development by creating an account on GitHub.</li><li><a href="https://github.com/Moz">moz - Overview</a>: moz has 19 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/blob/d5f614c9d7d1efdf6d40a8812d7f148f41aa1072/llama.cpp/server/public/index.html#L1113.">llamafile/llama.cpp/server/public/index.html at d5f614c9d7d1efdf6d40a8812d7f148f41aa1072 · Mozilla-Ocho/llamafile</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 messages): 

steedalot: They're obviously not that attractive for alignment researchers anymore...
  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1240614010651676715)** (1 messages): 

- **Introducing Needle in a Needlestack**: A member shared details about the **Needle in a Needlestack (NIAN)** benchmark, which is more challenging than the older **Needle in a Haystack (NIAH)**. They provided links to the [code](https://github.com/llmonpy/needle-in-a-needlestack) and the [website](https://nian.llmonpy.ai/), emphasizing that even **GPT-4-turbo struggles** with this benchmark.

**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1ct9nyp/needle_in_a_needlestack_nian/">Reddit - Dive into anything</a>: no description found

  

---



---



---



---



---



