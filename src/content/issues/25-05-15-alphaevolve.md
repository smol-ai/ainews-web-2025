---
id: MjAyNS0w
title: >-
  Gemini's AlphaEvolve agent uses Gemini 2.0 to find new Math and cuts Gemini
  cost 1% — without RL
date: '2025-05-15T05:44:39.731046Z'
description: >-
  **Deepmind's AlphaEvolve**, a 2025 update to AlphaTensor and FunSearch, is a
  Gemini-powered **coding agent for algorithm discovery** that designs faster
  matrix multiplication algorithms, solves open math problems, and improves data
  center and AI training efficiency. It achieves a **23% faster kernel speedup**
  in Gemini training and surpasses state-of-the-art on 20% of applied problems,
  including improvements on the Minimum Overlap Problem and Kissing number
  problem. Unlike Deep-RL, it optimizes code pieces rather than model weights.
  Meanwhile, **OpenAI** released **GPT-4.1** in ChatGPT, specializing in coding
  and instruction following, with a faster alternative **GPT-4.1 mini**
  replacing GPT-4o mini for all users. OpenAI also launched the Safety
  Evaluations Hub and the OpenAI to Z Challenge using o3/o4 mini and GPT-4.1
  models to discover archaeological sites. *"Maybe midtrain + good search is all
  you need for AI for scientific innovation"* - Jason Wei.
companies:
  - google-deepmind
  - openai
models:
  - gemini
  - gpt-4.1
  - gpt-4o-mini
  - o3
  - o4-mini
topics:
  - algorithm-discovery
  - coding-agents
  - matrix-multiplication
  - optimization
  - reinforcement-learning
  - model-weights
  - training-efficiency
  - safety-evaluations
  - instruction-following
  - coding-tasks
  - model-releases
people:
  - _philschmid
  - scott_swingle
  - alex_dimakis
  - henry
  - jason_wei
  - kevinweil
  - michpokrass
  - scaling01
  - gdb
---


Agent Harnesses are all you need.

> AI News for 5/15/2025-5/16/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (214 channels, and 3819 messages) for you. Estimated reading time saved (at 200wpm): 341 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

[Deepmind's new AlphaEvolve](https://x.com/GoogleDeepMind/status/1922669321559347498), 2025's update of [AlphaTensor](https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/) and [FunSearch](https://www.nature.com/articles/s41586-023-06924-6), is hard to grok, as it summarizes a year of results across a vast swath of math and LLM training applications, AND is not publicly available to try, but GDM succinctly puts it as "a Gemini-powered **coding agent for algorithm discovery**... able to:

- Design faster matrix multiplication algorithms,
- Find new solutions to open math problems,
- Make data centers, chip design and AI training more efficient across @Google.

It is described as an agent rather than a model due to [the mutiple components in a loop](https://x.com/GoogleDeepMind/status/1922669325283942539):

![](https://resend-attachments.s3.amazonaws.com/7muzi57wBPdtSjP)

It's very Googley to understate their results, so one has to turn to the Twitterverse to get the highlights, which are much better:

- "**Sped up Gemini training with 23% faster kernel resulting 1% total reduction in training time.**" - [Philipp Schmid](https://x.com/_philschmid/status/1922913381746352188)
- "**surpassing SOTA on 20%** of the problems it was applied to is actually nuts" - [Scott Swingle](https://x.com/bio_bootloader/status/1923121148864164123)
- "The results are impressive: they **improve the best known bounds on many problems** including the Minimum Overlap Problem by Erdos, matrix multiplication, and the Kissing number in 11 dimensions."
- "The solutions here are *pieces of code*, and **this is a search agent that modifies, evaluates, and optimizes code** i.e. pieces of text. **This is in sharp contrast to Deep-RL** where the solutions are models and what is optimized is their weights." - [Alex Dimakis](https://x.com/AlexGDimakis/status/1923160843795169447)
- On the 32% speedup of FlashAttention CUDA code - [Henry](https://x.com/arithmoquine/status/1922751330474500530)
- "AlphaEvolve is deeply disturbing for RL diehards like yours truly **Maybe midtrain + good search is all you need for AI for scientific innovation** And what an alpha move to keep it secret for a year." — [Jason Wei](https://discord.com/channels/822583790773862470/1372364050184409088/1372651541663846611)

Inquiring minds can watch the MLST interview about it:

https://www.youtube.com/watch?v=vC9nAosXrJw

---

# AI Twitter Recap

**GPT-4.1 and OpenAI Model Releases**

- **GPT-4.1 Now Available in ChatGPT**: [@OpenAI](https://twitter.com/OpenAI/status/1922707554745909391) announced that **GPT-4.1** is available in ChatGPT, highlighting its specialization in **coding tasks** and **instruction following**, making it a faster alternative to **OpenAI o3** & **o4-mini** for everyday coding needs, while [@kevinweil](https://twitter.com/kevinweil/status/1922732062345142306) also noted that this model is available for Plus/Pro/Teams subscribers and soon to Enterprise/Edu. [@michpokrass](https://twitter.com/michpokrass/status/1922716587468984689) confirmed **GPT-4.1 landing in chatgpt today** after initially planning on keeping this model api only, whereas [@scaling01](https://twitter.com/scaling01/status/1922715792849674568) said this is a **huge upgrade for ALL ChatGPT free users**!, noting GPT-4.1-mini replaces GPT-4o mini, and is honestly much, much better.
- **Introducing the Safety Evaluations Hub**: [@OpenAI](https://twitter.com/OpenAI/status/1922684895496720490) introduced the Safety Evaluations Hub, a resource to explore safety results for their models, emphasizing proactive communication about safety.
- **Introducing GPT-4.1 mini**: [@OpenAI](https://twitter.com/OpenAI/status/1922707556402618533) announced that they're also introducing GPT-4.1 mini, replacing GPT-4o mini, in ChatGPT for all users.
- **Releasing the OpenAI to Z Challenge**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1923062948060168542) announced the OpenAI to Z Challenge using **o3/o4 mini and GPT 4.1** models to discover previously unknown archaeological sites:, while [@gdb](https://twitter.com/gdb/status/1923105670464782516) is Releasing the OpenAI to Z Challenge — using o3/o4 mini and GPT 4.1 models to discover previously unknown archaeological sites.
- **Responses API Support Added to Evals API and Dashboard**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1923048126002102530) announced added support for the Responses API in the Evals API and dashboard and provided a handy guide on how to get started, using an example of comparing gpt-4.1-mini with gpt-4o-mini on stored responses [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1923048127826722849).

**Google's AlphaEvolve and Gemini**

- **AlphaEvolve, a Gemini-powered Coding Agent**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1922669321559347498) introduced AlphaEvolve, a Gemini-powered coding agent for algorithm discovery, which can design faster matrix multiplication algorithms, find new solutions to open math problems, and make data centers, chip design, and AI training more efficient across @Google. [@demishassabis](https://twitter.com/demishassabis/status/1922855470374572051) congratulated the **AlphaEvolve, Gemini and Science teams** on their accomplishment. They also detailed [how AlphaEvolve is used](https://twitter.com/GoogleDeepMind/status/1922669325283942539), using LLMs to synthesize information, automated evaluation for measurable problems, and evolution to iteratively improve algorithms. The company has also been [applying AlphaEvolve](https://twitter.com/GoogleDeepMind/status/1922669328660299914) to optimize data center scheduling, assist in hardware design, and enhance AI training and inference. AlphaEvolve has also been used to [discover new matrix multiplication algorithms](https://twitter.com/GoogleDeepMind/status/1922669331336384515), outperforming the previous model AlphaTensor, and [find new solutions to open math problems](https://twitter.com/GoogleDeepMind/status/1922669334142271645). The company aims to [keep developing AlphaEvolve](https://twitter.com/GoogleDeepMind/status/1922669336101065183) due to it's potential impact across different fields.
- **Implicit Caching with Gemini**: [@_philschmid](https://twitter.com/1922650422382104584) highlighted implicit caching support in GoogleDeepMind's Gemini, which unlocks up to **75% cost savings** when requests hit the cache. This is especially useful when sending requests with a common prefix, such as querying parts of a large PDF.

**Open Source Models, Training, and Frameworks**

- **Nous Decentralized Pretraining Run**: [@Teknium1](https://twitter.com/Teknium1/status/1922778056290419166) announced that Nous has begun a decentralized pretraining run of a dense Deepseek-like model with 40B parameters, over 20T tokens, with MLA for long context efficiency.
- **Hugging Face MCP Course**: [@reach_vb](https://twitter.com/reach_vb/status/1923038382126424380) announced Hugging Face has dropped the MCP Course, covering everything you need to know about Model Context Protocol and how to use it.
- **AM-Thinking-v1 Reasoning Model**: [@omarsar0](https://twitter.com/omarsar0/status/1922668488826741061) noted AM-Thinking-v1 looks like a strong 32B reasoning model. It outperforms DeepSeek-R1 and rivals Qwen3-235B-A22B, while being built on top of open-source, and [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1922483522549252200) notes that AM-Thinking-v1 performs on par with Qwen3-235B-A22B and Seed1.5-Thinking while being built entirely from the open-source Qwen2.5-32B base model and publicly available queries
- **Salesforce BLIP3-o Multimodal Models**: [@_akhaliq](https://twitter.com/_akhaliq/status/1923001183804764391) and [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922843713514193076) mentioned that Salesforce has released BLIP3-o on Hugging Face. It is a family of fully open unified multimodal models employing a diffusion transformer to generate semantically rich CLIP image features.

**Reasoning and Agentic Systems**

- **LLMs Get Lost in Multi-turn Conversations**: [@omarsar0](https://twitter.com/omarsar0/status/1922755721428598988) highlighted a paper investigating how LLMs perform in realistic, multi-turn conversational settings where user instructions are often underspecified and clarified over several turns. [@omarsar0](https://twitter.com/omarsar0/status/1922755768585158785) notes that All tested LLMs show significantly worse performance in multi-turn, underspecified conversations compared to single-turn, fully-specified instructions. The average performance drop is 39% across six tasks, even for SoTA models. He also listed the [main reasons](https://twitter.com/omarsar0/status/1922755800843550833) LLMs get "lost" including making premature assumptions, attempting full solutions before having all necessary information and overly verbose outputs.
- **FedRAG Framework**: [@*nerdai*](https://twitter.com/_nerdai_/status/1922732119706698118) introduced FedRAG, an open-source framework for fine-tuning RAG systems across centralized and federated architectures.
- **Chain-of-Thought Reasoning**: [@francoisfleuret](https://twitter.com/francoisfleuret/status/1922892961680896238) claims CoT is a poor man's version of "the real thing", as a process to sample meaningful latents.
- **RL for Search-Efficient LLMs**: [@omarsar0](https://twitter.com/omarsar0/status/1922665313117552664) noted that this presents a new post-training RL framework that explicitly trains LLMs to optimize search usage.
- **LangChain's Open Agent Platform (OAP)**: [@LangChainAI](https://twitter.com/LangChainAI/status/1922722850542346680) introduced the Open Agent Platform, an open-source, no-code agent building platform, which connects to MCP Tools, LangConnect for RAG, and other LangGraph Agents.
- **Runway References for Zero-Shot Testing**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1922742658620903885) demonstrated Runway References for zero-shot testing of clothes, locations, and poses.

**AI Implementation, Tooling, and Infrastructure**

- **Meta's Transformers + MLX Integration**: [@awnihannun](https://twitter.com/awnihannun/status/1923065749234647214) expressed the importance of Transformers to the open-source and overall AI ecosystem and looked forward to more and deeper integrations with MLX + Transformers.
- **OpenAI's Tech Stack**: [@nrehiew_](https://twitter.com/nrehiew_/status/1922668335960924579) pointed out that OpenAI uses FastAPI to serve ChatGPT, countering complaints about Python and FastAPI's capabilities.
- **LangGraph Platform Generally Available**: [@LangChainAI](https://twitter.com/LangChainAI/status/1922709747423183226) announced that LangGraph Platform is now generally available, allowing users to deploy, scale, and manage agents with long-running, stateful workflows.
- **GPT-4.1 Coding Skills** [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1922709921772036164) states that it was Coded up by GPT-4.1, rolling out today in ChatGPT
- **The importance of embeddings in training**: [@jxmnop](https://twitter.com/jxmnop/status/1922468210256879786) states embeddings are underrated.
- **Atropos and Axolotl AI**: [@Teknium1](https://twitter.com/Teknium1/status/1922435846751584771) You can now train using Atropos using @axolotl_ai too

**AI Analysis and Evaluation**

- **ARI Beats OpenAI's Deep Research**: [@RichardSocher](https://twitter.com/RichardSocher/status/1923098655768314363) announced that ARI (Advanced Research & Insights agent) just beat OpenAI's Deep Research by a large margin and on two benchmarks.
- **GPT-4.1 Excellent Coding Skills**: [@kevinweil](https://twitter.com/kevinweil/status/1922732062345142306) states that GPT 4.1 is very good at coding and instruction following, recommending users to give it a try.
- **Limitations of current evals**: [@cline](https://twitter.com/cline/status/1922722359795916943) reports that eval loops rarely survive contact with real humans.
- **LangChain Interrupt 2025 Evals**: [@LangChainAI](https://twitter.com/LangChainAI/status/1922747560483226041) launched OpenEvals, a set of utilities to simulate full conversations and evaluate LLM application performance.
- **AM-Thinking-v1 Model Evaluation**: [@omarsar0](https://twitter.com/omarsar0/status/1922668488826741061) points out that AM-Thinking-v1 outperforms DeepSeek-R1 and rivals Qwen3-235B-A22B and [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1922483522549252200) notes that AM-Thinking-1 performs on par with Qwen3-235B-A22B and Seed1.5-Thinking while being built entirely from the open-source Qwen2.5-32B base model and publicly available queries.

**Humor and Miscellaneous**

- **The cat is out of the bag** [@omarsar0](https://twitter.com/omarsar0/status/1922755721428598988) on the topic of LLMs get lost in multi-turn conversations.
- **OpenAI uses FastAPI for ChatGPT**: [@nrehiew_](https://twitter.com/nrehiew_/status/1922668335960924579) jokes about the extensive use of python despite constant complaining about it.
- **Google Time**: [@zacharynado](https://twitter.com/zacharynado/status/1922652507681026236) exclaims ♊️ GOOGLE TIME! (づ｡◕‿‿◕｡)づ♊️.
- **LLMs will just lie to me** [@nearcyan](https://twitter.com/nearcyan/status/1922548145340195148) reports on hating LLMs.
- **You have picked your poison!** [@scottastevenson](https://twitter.com/scottastevenson/status/1922491520445305338) responding to a user about escaping ambiguity and submitting to structure, education, doomscrolling, drug addiction, fitness routines, mountain climbing, entrepreneurship, raising a family, abusive relationships.
- **The mog trend was kinda cringe, but this one goes hard fr** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1922694806599299432) reacting to an image.
- **The fact flash-attention somehow works with uv gives me hope this is possible** [@typedfemale](https://twitter.com/typedfemale/status/1922427558924001672) hopeful comment on uv and flash attention.
- **Building perfect machines out of imperfect parts** [@MillionInt](https://twitter.com/MillionInt/status/1923023812821385240) a deep comment.
- **oh no** [@lateinteraction](https://twitter.com/lateinteraction/status/1922708925142475078) a reaction to something that occurred.
- **This is too good, I didn't expect YC launch videos to be docudramas of stuff that just happened like a month ago but here we are** [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922924417429958760) reaction to a YC launch video.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Text-to-Speech Model Training and Tools in Unsloth

- [**TTS Fine-tuning now in Unsloth!**](https://v.redd.it/faqjz7kzaz0f1) ([Score: 223, Comments: 43](https://www.reddit.com/r/LocalLLaMA/comments/1kndp9f/tts_finetuning_now_in_unsloth/)): **Unsloth has introduced support for efficient Text-to-Speech (TTS) model fine-tuning, claiming ~1.5x faster training and 50% less VRAM usage compared to alternatives, particularly on FA2 hardware. Supported models include** `Sesame/csm-1b`**,** `CanopyLabs/orpheus-3b-0.1-ft`**, and Transformer-based models (e.g., Llasa, Outte, Spark), with data-efficient SFT-style workflows using emotion-annotated datasets like 'Elise'. Users can leverage Google Colab [notebooks](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning), 16-bit LoRA or full-precision fine-tuning, and quantized/original checkpoints on Hugging Face. Notably, a new Qwen3 GRPO method is also supported, combining base models with custom proximity-based reward functions and regex-guided evaluation.** Comments clarify that while Whisper is primarily an STT model, its inclusion may be for ASR-related preprocessing or dataset generation. Users discuss LoRA fine-tuning scalability and best practices for controlling TTS model tone, pitch, and cadence, as well as dataset size requirements per parameter count, highlighting interest in practical fine-tuning methodologies.
    - A user questions whether "Whisper" is a TTS model—clarifying that it is primarily an STT (speech-to-text) and ASR (automatic speech recognition) model—not text-to-speech. The comment asks if Unsloth's finetuning is supporting datasets for ASR finetuning rather than true TTS.
    - Another commenter asks specifically about the requirements for TTS finetuning, namely how many audio/text examples are needed per billion or hundred million parameters. This reflects a common technical concern of dataset scale versus model parameterization in speech synthesis finetuning.
    - A technical feature request is made regarding native Mac MPS (Metal Performance Shaders) support, seeking to enable hardware-accelerated training/inference on Apple Silicon devices, which is relevant for efficient TTS model finetuning workflows outside of CUDA dependencies.
- [**Introducing A.I.T.E Ball**](https://v.redd.it/scyofz31dx0f1) ([Score: 281, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1kn542r/introducing_aite_ball/)): **The post details a local, offline implementation of an AI-powered "8 Ball" device running entirely on an Orange Pi Zero 2W. The setup uses whisper.cpp for local text-to-speech, llama.cpp for LLM inference, and specifically runs the Gemma 3 1B model, emphasizing the resource constraints and fully offline capability.** Commenters noted appreciation for fully local operation, highlighting the rarity of offline AI hardware in a heavily internet-connected landscape. No substantive technical debate is present.
    - A commenter suggests integrating Piper TTS (https://github.com/rhasspy/piper), an open-source text-to-speech engine, to enhance the project, noting its ability to run efficiently on modest hardware. This implies the device could be upgraded for voice output despite hardware limitations.
    - Multiple comments highlight the unique feature of running the model entirely offline, contrasting it with the trend of always-online AI devices. This offline capability is noted as significant, particularly for privacy and increased control of the application on resource-constrained hardware.

### 2. New Features and Data Handling in llama.cpp

- [**PDF input merged into llama.cpp**](https://github.com/ggml-org/llama.cpp/pull/13562) ([Score: 120, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1kn75q8/pdf_input_merged_into_llamacpp/)): **A recent PR ([#13562](https://github.com/ggml-org/llama.cpp/pull/13562)) has added native PDF input support in the llama.cpp web UI by integrating an external JS library for PDF parsing, offering users the ability to toggle between text extraction and image rendering. This approach ensures that the C++ core remains unaffected, allowing rapid updates and replacement of PDF parsing tools, and includes an automatic conversion for lengthy pasted content to file uploads.** Comments note that this implementation upholds core modularity (aligning with maintainability), but some express concern about feature creep versus adherence to the Unix philosophy. There's technical discussion about OCR integration for mixed-content PDFs and requests for merging related PRs for extended document handling.
    - The PDF input functionality for llama.cpp is implemented in the built-in web frontend using an external JavaScript package, not in the core C++ application. This architectural decision keeps core maintenance minimal and allows for easy replacement or upgrading of the PDF conversion package without affecting core functionality.
    - Currently, the solution provides two modes for handling PDFs: parsing as pure text or pure image. There is recognition among users that a more robust approach would selectively extract text natively while applying OCR only to the image parts—akin to how specialized OCR software works—suggesting potential future improvements in structural and semantic PDF understanding.
    - Users question whether the existing integration can extract and represent structural information from PDFs, such as tables or embedded images, which are crucial for advanced tasks like RAG (Retrieval Augmented Generation) and graph building. Effective support for such features is recognized as a significant technical challenge in PDF processing for LLM pipelines.
- [**Introducing A.I.T.E Ball**](https://v.redd.it/scyofz31dx0f1) ([Score: 281, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1kn542r/introducing_aite_ball/)): **The post details a local, offline implementation of an AI-powered "8 Ball" device running entirely on an Orange Pi Zero 2W. The setup uses whisper.cpp for local text-to-speech, llama.cpp for LLM inference, and specifically runs the Gemma 3 1B model, emphasizing the resource constraints and fully offline capability.** Commenters noted appreciation for fully local operation, highlighting the rarity of offline AI hardware in a heavily internet-connected landscape. No substantive technical debate is present.
    - A commenter suggests integrating Piper TTS (https://github.com/rhasspy/piper), an open-source text-to-speech engine, to enhance the project, noting its ability to run efficiently on modest hardware. This implies the device could be upgraded for voice output despite hardware limitations.
    - Multiple comments highlight the unique feature of running the model entirely offline, contrasting it with the trend of always-online AI devices. This offline capability is noted as significant, particularly for privacy and increased control of the application on resource-constrained hardware.

### 3. LLM Multi-Turn Conversation Challenges and Benchmarks

- [**LLMs Get Lost In Multi-Turn Conversation**](https://www.reddit.com/r/LocalLLaMA/comments/1kn2mv9/llms_get_lost_in_multiturn_conversation/) ([Score: 218, Comments: 67](https://www.reddit.com/r/LocalLLaMA/comments/1kn2mv9/llms_get_lost_in_multiturn_conversation/)): **A recent paper ([arXiv:2505.06120](https://arxiv.org/abs/2505.06120)) demonstrates that both open and closed-source LLMs (Language Learning Models) show substantial degradation in performance during multi-turn conversations, particularly when instructions are "sharded" (split across turns) versus "concat" (provided all at once). Experiments reveal LLMs frequently make compounding errors after initial incorrect assumptions, rarely recovering from early misinterpretations—a phenomenon not captured in single-turn benchmarks. The research suggests that reinitiating conversations with all relevant context in the first prompt can mitigate this issue.** Commenters corroborate these findings with practical experiences, noting this multi-turn degradation with various models (e.g., o1 pro, sonnet 3.7, vs. improvements in 2.5 pro). One detailed example shows how iterative prompting with LLMs (Gemma, Qwen) leads to semantic drift and compounding of initial mistakes due to LLMs' reliance on prior outputs, illustrating a core challenge in multi-turn context tracking.
    - Users observe that LLMs—like o1 pro, sonnet 3.7, and even strong open models like Gemma and Qwen—often make initial incorrect assumptions in early turns, then compound these mistakes across multi-turn conversations due to their autoregressive nature. As one user notes, *“LLMs being word probability engines makes them double down on previous choices, so initial mistakes lead to compounding errors and general weirdness.”*
    - There is a critique that most LLM benchmarks and tuning exercises are focused primarily on single-turn, fully-specified instruction settings, which can misrepresent actual multi-turn or agent-centric use cases. One commenter highlights that in real-world and especially coding workflows, multi-turn performance is paramount, but models may only be optimized for benchmark scores.
    - The comment about the 'full and concat' prompt strategies reflects an interest in how context handling methods affect performance: earlier and smaller models are said to depend heavily on prompt structure, while newer/modern models may manage context better. This draws attention to an important technical axis for ranking models’ suitability for agent use and extended dialogues.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

TO BE COMPLETED

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: Model Mania - New Releases and Capabilities Spark Fierce Debates**

- [**Gemini 2.5 Pro Flexes Its Muscles (and Context Window)**](https://ai.google.dev/gemini-api/rest/v1beta/GenerationConfig): Engineers across Discords like **LMArena**, **aider**, and **OpenAI** extensively discussed **Gemini 2.5 Pro**, praising its coding prowess, reasoning skills, and massive **1 million token context window**, which some find indispensable compared to **GPT's 32k limit**. While its [free availability](https://discord.com/channels/974519864045756446/998381918976479273/1372558644578881576) is appreciated, many acknowledge it's likely temporary due to high operational costs, though some users found its reasoning chunks *useless*.
- [**GPT-4.1 Variants Battle for Coding Crown**](https://openai.com/index/gpt-4-1/): The **OpenAI** and **aider** communities buzzed with comparisons between **GPT-4.1** and **GPT-4o**, with many asserting **GPT-4.1** (and particularly **GPT-4.1 mini**) excels at coding tasks due to better instruction following. Users shared [screenshots comparing models](https://cdn.discordapp.com/attachments/998381918976479273/1372555924862140436/20250515_154222.png?ex=682733d1&is=6825e251&hm=206683cf5fa5279c41f8150a0645b58d3595bbfde5b3616003058b41a29e5cae&) and debated the potential **GPT-5** release, speculated for summer or late 2024.
- [**Fresh Faces in the AI Arena: DeepSeek, Qwen, AlphaEvolve & More!**](https://deepseek.com/blog/deepseek-v3): New model announcements peppered discussions, including **DeepSeek v3** as a **Mixture of Experts (MoE)** model, **Qwen3's** aptitude for translating Mandarin datasets, and **Google DeepMind's AlphaEvolve** ([AlphaEvolve PDF](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)) sparking debate on whether it's a true evolutionary algorithm or LLM-driven. **Samsung** also joined the fray with models like **MythoMax-L2-13B** ([spotted on Hugging Face](https://huggingface.co/Samsung/MythoMax-L2-13B)) and **MuTokenZero2-32B**.

**Theme 2: Engineering AI - Optimizing Performance and Refining Development Tools**

- [**Quantization Wars: QNL Smokes GGUFs for Speed!**](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs): The **Unsloth AI** community reported that **QNL** offers faster performance than standard **GGUFs**, though formal benchmarks are awaited, highlighting ongoing efforts to optimize model speed and efficiency. Discussions in **LM Studio** also emphasized the critical role of keeping models entirely within **VRAM** over **DRAM** speed for optimal performance, with **KV Cache** location being a key factor.
- [**Framework Fever: DSPy, LlamaIndex, LangGraph & MCP Streamline AI Dev**](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/): Developers are actively leveraging frameworks like **DSPy** for implementing structured outputs with **Pydantic models**, and **LlamaIndex** for building event-driven agent workflows, such as a [multi-agent Docs Assistant for Weaviate](https://twitter.com/llama_index/status/1923085124725506441). **LangGraph** is gaining traction for managing complex conversational flows ([LangGraph Course](https://huggingface.co/learn/agents-course/unit2/langgraph/first_graph)), while the **Meta-Circular Evaluator Protocol (MCP)**, now with [Shortwave client support](https://www.shortwave.com/docs/how-tos/using-mcp/), is enabling easier integration of AI agents with various applications.
- [**Hardware Hustles: From Multi-GPU Fine-Tuning to WebGPU Roasts**](https://asianmom.kuber.studio/): Multi-GPU fine-tuning using tools like **Accelerate** with **Unsloth** is a hot topic, and the **GPU MODE** server saw active benchmarking of **MI300** cards and discussions on **TritonBench** errors on AMD GPUs ([example kernel](https://github.com/thunlp/TritonBench/blob/main/data/TritonBench_G_v1/chunk_gla_fwd.py)). Meanwhile, a fun **WebGPU Vision-LLM app** called [**AsianMOM**](https://asianmom.kuber.studio/), using **SmolVLM 500M** and **LLama 3.2 1B**, demonstrated in-browser AI roasting capabilities.

**Theme 3: Platform Quirks & User Workarounds - Navigating the AI Landscape**

- [**Perplexity's Pro Problems and Deep Research Disappointments**](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA): **Perplexity AI** users faced delays getting the **Perplexity Pro role** on Discord (with [mods manually assigning](https://discord.com/channels/1047197230748151888/1111786888626438245)), experienced errors viewing [mobile app answers on the web](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA), and expressed frustration with the **Deep Research** mode defaulting to regular search or using limited sources, with one user stating, *"If that's dumbed down, I see no reason to pay for it."*
- [**Model Mishaps: Looping Llamas and False Flags Frustrate Users**](https://cdn.discordapp.com/attachments/1110598183144399058/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&): Users in **LM Studio** reported **Llama 3.1/3.3** models producing undesirable outputs with fantasy prompts, showing [token loss and punctuation issues](https://cdn.discordapp.com/attachments/1110598183144399058/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&). **Cursor Community** members saw **Claude 3.5 Sonnet** [get stuck in loops](https://cdn.discordapp.com/attachments/1074847527708393565/1372432736752762930/Gq9pkJRaAAECBOI.png?ex=682769d7&is=68261857&hm=02da0d21e3c19e9a58420fc3ee21f55a47edd8f0e64e700331b1b3ae6bc99a9d&), and **Eleuther** discussions highlighted **MCQ evaluations** like **MMLU** incorrectly flagging model outputs as false.
- [**Community to the Rescue: Proxies, Token Tips, and GPT4All Alternatives**](https://aider.chat/docs/config/aider_conf.html): When **OpenAI** access was restricted by country, **OpenRouter** users suggested proxies as a workaround. **Aider** users shared tips for managing token usage with commands like `/clear` and using models like **Gemini 2.5 Flash**, while the [**Nomic.ai**](http://nomic.ai/) community, fearing **GPT4All's** discontinuation, discussed [**Jan.ai**](http://jan.ai/) and **LM Studio** as alternatives.

**Theme 4: The Bustling AI Ecosystem - Collaboration, Learning, and Open Source Triumphs**

- [**Indie Devs Unleash Creative AI: From Hotel Agents to Mom Roasters!**](https://github.com/jinkoso/jinko-mcp/blob/master/README.md): The community showcased impressive open-source projects, including the [**Jinko MCP**](https://github.com/jinkoso/jinko-mcp/blob/master/README.md) for building AI agents to sell hotels, the **Tig coding agent** ([announced by LlamaIndex](https://twitter.com/llama_index/status/1923134285940441102)) built with LlamaIndex workflows, and the humorous [**AsianMOM** WebGPU Vision-LLM app](https://asianmom.kuber.studio/) that roasts users. Additionally, [**Mem0.ai**](http://mem0.ai/) introduced [**OpenMemory MCP**](https://mem0.ai/blog/introducing-openmemory-mcp/), a unified memory management layer for AI apps.
- [**Level Up Your AI Game: Workshops, Webinars, and Challenges Galore!**](https://lu.ma/39b7e9pu): Numerous learning opportunities surfaced, such as **Nous Research** and **Solana Foundation's** [Decentralized AI event in NYC](https://lu.ma/39b7e9pu), a **Lambda workshop** on [building agentic applications](https://www.youtube.com/watch?v=VmjMIwwo9ag) with **$100 API credits** available, and **OpenAI's "OpenAI to Z Challenge"** ([details here](https://openai.com/openai-to-z-challenge/)) to discover Amazonian archaeological sites. **BlackboxNLP 2025** also announced a [new shared task on circuits/causal variable localization](https://blackboxnlp.github.io/2025/task) using the [MIB Benchmark](https://mib-bench.github.io/).
- [**Fueling the Fire: API Credits and Sponsorships Keep Innovation Burning**](https://github.com/Aider-AI/aider/issues): Generosity flowed with a user in the **aider** community offering free **API credits** for Gemini, Claude, and OpenAI to support interesting projects, particularly for the [aider project](https://github.com/Aider-AI/aider/issues). Elsewhere, a tech-adjacent nonprofit sought event sponsorships and grants from **Cohere** (contact [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com)), highlighting the various ways the ecosystem supports ongoing development.

**Theme 5: AI's Wild Side - Controversies, Accidental Leaks, and Industry Shake-ups**

- [**Grok's Gaffes: From "White Genocide" Claims to Admitting Elon's a Threat!**](https://x.com/): Elon Musk's **Grok** model stirred controversy in the **aider** community by making wild claims about **white genocide**, leading to distrust, with some users viewing **xAI** as a joke. One member humorously reported asking **Grok** who the biggest threat to democracy on **X** was, to which it allegedly replied: **Elon**.
- [**Oops, We Leaked It! Samsung's MythoMax-L2-13B Makes Brief Debut**](https://huggingface.co/Samsung/MythoMax-L2-13B): **Samsung** inadvertently released (and then quickly removed) the **MythoMax-L2-13B** roleplay model, which was [spotted on Hugging Face](https://huggingface.co/Samsung/MythoMax-L2-13B), as noted in the **Yannick Kilcher** and **HuggingFace** Discords. This led one user to quip, *"Can someone do that for **OpenAI** and 'release' **GPT4Chan**? Or **Anthropic**, that would be priceless."*
- [**Industry Tremors: TypeScript Dev Fired, Agentic Tools Challenge Big Tech**](https://x.com/brodyford_/status/1922726909365879039?s=46): **Microsoft's** unexpected firing of a key **TypeScript** developer ([tweeted here](https://x.com/brodyford_/status/1922726909365879039?s=46)) sparked dismay in the **Latent Space** community. Discussions also touched on how **agentic tooling** might empower indie developers to outpace big tech, and the demise of **FUNapis** was humorously attributed to [**Bing's** chatbot wrapper ambitions](https://x.com/williambryk/status/1923062696095711374?s=46).


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Role: Mods to the Rescue**: Users purchasing **Perplexity Pro** are experiencing delays in obtaining the **Perplexity Pro role** on Discord, but [moderators are manually assigning roles](https://discord.com/channels/1047197230748151888/1111786888626438245) while the system is reworked.
   - Having the same email for Discord and Perplexity is not a requirement.
- **Mobile App Answers Trigger Webpage Errors**: Answers generated from the **Perplexity mobile app** can't be read on the web version, [resulting in error messages](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA).
   - Customer service acknowledged the issue and reported it to technicians, but a fix has not yet been implemented.
- **Deep Research Deemed Disappointing?**: Users report issues with **Deep Research** mode, as it defaults to regular search and uses a limited number of sources.
   - One user summarized the sentiment: *I legit just use Perplexity because it reads 20-40 sources per query. If that's dumbed down, I see no reason to pay for it.*
- **23andMe Braces for Bankruptcy**: [23andMe filed for Chapter 11 bankruptcy](https://www.perplexity.ai/search/23andme-files-for-chapter-11-b-msxlvXlmQCK0MLt0UUeTcg), indicating significant financial challenges and a need for restructuring.
   - The Chapter 11 filing suggests that **23andMe** is seeking legal protection to reorganize its debts and operations.
- **Sonar API Stings Hackathon Hopefuls**: Members are facing problems acquiring the **Sonar API** for a hackathon project, since it requires credit card details.
   - Another member reports that the answers generated using the **Sonar API** are different from those generated in the [Perplexity AI playground](https://labs.perplexity.ai/).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Engages in 'Vibe Coding' Livestream**: Manus hosted a "Vibe Coding with Manus" livestream from San Francisco, featuring the Manus SF Fellows, and available on [YouTube](https://youtube.com/live/w4XegM6dOgc?feature=share).
   - The livestream showcased coding projects, fostering community engagement.
- **Johnny the Credit-Farming Genius**: A user humorously highlights how Johnny is *"farming Manus daily"* compared to another user *paying for manus to make a femboy detector*.
   - This illustrates a humorous contrast between exploiting the platform for free credits and using it for entertainment.
- **The Femboy Detector**: A user created an app that determines if a person is a femboy or not, using Function Calling to get currency rates from [wise.com](https://wise.com).
   - The app outputs "femboy" for male names, leading to comedic accusations of users being labeled femboys: *MANUS'S API IS LYING!*
- **Invite Link Feature Vanishes**: Some users have found that the option to generate invite links has disappeared from the UI.
   - It seems that the invite link generation feature is no longer available to all users and some speculate this affects free users only.
- **Users Ponder Credits Usage**: Users expressed concerns about the cost of credits, with one noting that a PDF report cost **500 credits** and a DCF on Google cost **1500 credits**.
   - Some believe the credit usage is too expensive, especially for complex tasks and look forward to alipay as a payment method.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Geminiyo's Heavy Reasoning Time?**: Members pondered whether [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/rest/v1beta/GenerationConfig) will have longer reasoning times compared to other reasoning models, given it is a *heavier model*.
   - The discussion centered on the trade-offs between model size, complexity, and inference speed.
- **Grok 3.5 Launch Stuck in the Muck?**: Theorizing that **Elon** delayed the release of **Grok 3.5** because fine-tuning it to the far right *wasn't successful*, one member was promptly corrected by another noting it was *false information*.
   - Discussion touched on the possibility of injecting *political stuff* into the system prompt.
- **Attention Steering a Mirage?**: Despite claims that **LLMs** can steer attention, particularly with **Grok's** Twitter timeline, one member clarified that *this isn't steering attention where you need them to*, but rather a plain announcement.
   - They further linked to [Transluce's observability interface](https://transluce.org/observability-interface) as a tool to play with feature steering, but cautioned it's not particularly useful in practice yet.
- **LMArena Funded by Credits, not Cash**: In a discussion about how **LMArena** funds its models, a member pointed out that *it's not just the companies themselves who pay for inferences*, suggesting **credit grants** are provided to **LMArena** instead of direct monetary payments.
   - Another member added that big labs give **LMArena** endpoints for their models, with valuable data on human preference being collected as a result.
- **O3 Pro MIA on Arena**: Despite speculation about an **O3 Pro** release, a member stated *O3 pro wont come to arena lol*, to which a moderator replied *I can't confirm if/when new models are arriving on arena, but will be sure to put out announcements when I can*.
   - Anticipation remains high for new models to be added to the platform.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **QNL smokes GGUF for Speed**: **QNL** is reportedly faster than standard **GGUFs**, but formal benchmarks are still pending.
   - See [Unsloth Dynamic 2.0 GGUFs documentation](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) for details.
- **Scale Up with Multi-GPU Fine-tuning**: For multi-GPU finetuning, using **Accelerate** with Unsloth may be successful.
   - Despite consumer GPUs like the **3090** offering **24GB VRAM**, companies often opt for **H100s** for local AI.
- **SLMs Challenger to LLMs**: Smaller models (**SLMs**) can become competitive through fine-tuning on specific tasks, even if they are not as smart out-of-the-box.
   - **Qwen3-4B** is recommended for models needing decent reasoning power, and purportedly beats **Mistral 7B**.
- **Qwen3 cracks translation**: **Qwen3** is suggested for translating datasets in Mandarin due to its pretraining data.
   - Users have reported success using **30B** models with **Ollama** on Kaggle for processing millions of strings.
- **SmolVLM roasts in-browser**: A member created **AsianMOM**, a [WebGPU Vision-LLM app](https://asianmom.kuber.studio/) that roasts you like ur mom in-browser.
   - It uses **SmolVLM 500M** and **LLama 3.2 1B** and works directly in your browser, thanks to Transformers.js and HF.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok Sparks White Genocide Fears**: Elon Musk's **Grok** raised concerns when it made wild claims about **white genocide**, causing distrust and prompting some to view **xAI** as a joke, and **Elon** as the biggest threat to democracy on **X**.
   - Users are avoiding the model and one member joked that they only used **xAI** to ask who the biggest threat to democracy on **X** was, and it admitted it was **Elon**.
- **Gemini 2.5 Pro Flees Copilot**: **Gemini 2.5 Pro** was briefly available in Copilot, but it has since been removed, generating speculation about **Microsoft** investing more in open-weight models.
   - The removal prompted speculation that **Microsoft's** rocky relationship with **OpenAI** may lead them to focus on open-weight models.
- **Dev Generosity: Free API Credits Flow**: A user offered free **API credits** for **Gemini, Claude, and OpenAI**, inviting others to test new infrastructure and develop interesting projects, particularly for the [aider project](https://github.com/Aider-AI/aider/issues).
   - Another member is planning to add a `/consolidate` command to aider to roll each long chat into a single, fully-specified prompt, using a fresh single-turn request using the main model, in response to the [LLMs Get Lost In Multi-Turn Conversation](https://arxiv.org/abs/2505.06120) paper.
- **Aider Token Use Gets a Grip**: Users discussed strategies for managing token usage in **Aider**, suggesting the use of `/clear` to reduce context and `/add` only necessary files, and using models like **Gemini 2.5 Flash** via Google AI Studio.
   - Also suggested was using **OpenRouter** like Deepseek v3 0324, paired with copy-pasting from **Gemini 2.5 Pro**.
- **Muscle-mem tool surfaces**: A member shared a github link to a tool, [muscle-mem](https://github.com/pig-dot-dev/muscle-mem), perhaps an aid to memorization?
   - No other details were shared, so it is difficult to know what it is for!



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4.1 Mini Smarts Spark Coding Debate**: Members debated the merits of **GPT-4.1** versus **GPT-4o**, with some asserting that **4.1** is superior for coding while others found **4o** to be more intuitive overall, and several claimed **4.1 mini** is the best small model.
   - Some users shared their prompt testing results and experiences across specific dialects to show their **GPT-4.1** preferences, while sharing [screenshots of the models](https://cdn.discordapp.com/attachments/998381918976479273/1372555924862140436/20250515_154222.png?ex=682733d1&is=6825e251&hm=206683cf5fa5279c41f8150a0645b58d3595bbfde5b3616003058b41a29e5cae&).
- **GPT-5 Release Date Guessing Game**: The community discussed the potential release timeline for **GPT-5**, with some anticipating a launch in **summer** (**June-September**) while others suggested a later release around **August-December**.
   - One member said, *Connecting the dots here - Sam said in the Feb 12 announcement that GPT-5 will solve the confusing names issue.*
- **Gemini 2.5 Pro Wins Fans with Million Context Window**: Users praised **Gemini 2.5 Pro**, noting its coding abilities, reasoning skills, and large context window, with one saying *Gemini 2.5 Pro is actually really good at coding*, while acknowledging that [its free availability](https://discord.com/channels/974519864045756446/998381918976479273/1372558644578881576) is likely temporary due to its high running costs.
   - One user has switched to **Gemini 2.5 Pro** due to its **1 million context window** stating that they simply can’t work with **GPT’s tiny 32k context window anymore**.
- **OpenAI Searches Amazonia For Archaeology**: OpenAI announced the **OpenAI to Z Challenge** using **o3**, **o4-mini**, or **GPT-4.1** to discover previously unknown archaeological sites in the Amazon, inviting participants to share progress on X with the hashtag #OpenAItoZ.
   - The challenge details are available at the [OpenAI website](https://openai.com/openai-to-z-challenge/).
- **Community Gives Research GPT the Eye Test**: A member is requesting feedback on their [Research GPT](https://chatgpt.com/g/g-68236174e57c8191aa65e6ed815b8f46-reserch-for-me), in its final stages of refinement.
   - The creator is particularly interested in identifying potential issues in English, as it is not their first language, while noting that Korean functionality is satisfactory.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro Plan Clarifications**: A member inquired about using **all models** on their current **Cursor Pro** plan and whether an **API key** is required, which was confirmed to be managed by Cursor itself.
   - The plan includes various models supported directly by Cursor, eliminating the need for users to manage their own **API keys**.
- **Cursor Client Version Stats**: A user shared detailed **Cursor** client information, including **VSCode version 1.96.2**, **Electron 34.3.4**, **Chromium 132.0.6834.210**, and **Node.js 20.18.3**, while troubleshooting a 'no restart' message, as seen on the [Cursor FAQ](https://cursor.sh/docs/faq).
   - This level of detail helps in diagnosing specific issues related to the client's configuration and environment, though it was also suspected to be related to timezone.
- **Claude 3.5 Sonnet Runs in Circles**: A user reported that **Claude 3.5 Sonnet** was getting stuck in a loop, complete with [supporting images](https://cdn.discordapp.com/attachments/1074847527708393565/1372432736752762930/Gq9pkJRaAAECBOI.png?ex=682769d7&is=68261857&hm=02da0d21e3c19e9a58420fc3ee21f55a47edd8f0e64e700331b1b3ae6bc99a9d&), eventually resolving itself due to context limits.
   - The issue highlights potential problems with the model's ability to manage context and avoid repetitive outputs.
- **Slash Context Reset Command Silently Implemented**: Members discussed using the **/reset** command in **Cursor** to clear the context, with some users expressing dislike for its silent execution.
   - It was clarified that after typing **/reset**, *nothing will show up, it will be executed silently*, which might confuse some users about whether the command was actually processed.
- **Gemini Pro Preview Has Editing Setbacks**: A member reported that **Gemini Pro Preview** spends a significant amount of time deciding on code changes but then struggles to apply those edits effectively.
   - Another member pointed out that **version 0.50.4** aimed to improve the apply functionality, suggesting that the issue might be addressed in newer releases.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Fantasy Prompts Frustrate Llama Models**: Users reported that **Llama 3.1/3.3** models produce undesirable outputs when given fantasy-themed prompts, exhibiting **token loss**, **punctuation issues**, and **partial word omissions** as shown in [attached screenshots](https://cdn.discordapp.com/attachments/1110598183144399058/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&).
   - The issues highlight ongoing challenges in maintaining coherence and accuracy with specific types of prompts, with no clear solution offered by the community.
- **LM Studio Vision API Input Guidance Given**: Users sought advice on supplying images to a vision-enabled LLM via the **LM Studio API** without the Python library, focusing on the **OpenAI endpoint** and one user pointed to the [LM Studio documentation](https://lmstudio.ai/docs/typescript/llm-prediction/image-input).
   - A cURL example was shared, demonstrating how to pass an image URL to the API, and resolving the issue for the user.
- **DRAM Secondary to VRAM for Model Speed**: Members discussed the impact of **VRAM** versus **DRAM** speed on model performance, concluding that **DRAM speed barely matters if you keep your model in VRAM**.
   - They emphasized keeping the model entirely within **VRAM** for optimal speed and performance, since the **KV Cache** location impacts performance.
- **7900 XTX Card tempts over Nvidia**: One member bought a **7900XTX** card and is about to sell one of their **Nvidia** cards, because they were mildly annoyed with dual card driver instability issues with their **4080s** and **4060ti** cards.
   - They mentioned that the **5060 ti** is going back as a return, with no further detail.
- **5060 Ti Faces AI and Gaming Divide**: The **8GB 5060 Ti** is considered inadequate for **AI** tasks, and not economical enough for budget gaming setups.
   - The consensus suggests that prospective buyers should consider the **16GB** version instead if they want to use **AI** models.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AlphaEvolve: LLM in Disguise?**: Members debated whether **AlphaEvolve** is simply an LLM-driven agentic loop or an evolutionary algorithm, referencing [Google DeepMind's blog](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf).
   - Arguments centered on the balance between the power of LLMs like **Gemini** and the importance of the evolutionary engine and verifier components, crucial to review in the *ablation section of the paper*.
- **Gemini 2.5 Pro Aces Coding**: The community lauded **Gemini 2.5 Pro's** coding skills, notably its *zero-shot* performance, while also noting it had *sorta random refusals* for *cyber ethics* homework when released.
   - Fine-tuning with verified rewards and rejecting non-compiling outputs may be the key to its coding excellence and reasoning abilities.
- **LiquidAI Faces Vaporwave Skepticism**: Skepticism is mounting around **LiquidAI** and its *liquid foundational models*, with one member initially writing them off, while another compared **LiquidAI** to **State Space Models (SSMs)**.
   - The community favored the treatment of **SSMs** like **Mamba**, **Gated DeltaNet**, or **RWKV 7**, noting similarities between these and **LiquidAI's Research** at [LiquidAI's Research](https://www.liquid.ai/research/liquid-neural-networks-research).
- **Absolute Zero: Models Rise From Nothing**: The community discussed the **Absolute Zero** paper [Absolute Zero](https://link.to.absolutezero), which improves models without any data by having LLMs generate and verify synthetic training data.
   - The LLM is trained to perform three tasks: **Y = F(X)**, **Y = F(?)**, and **Y = ?(X)**.
- **Samsung Accidentally Drops the MythoMax-L2-13B Roleplay**: **Samsung** inadvertently released the **MythoMax-L2-13B** roleplay model, spotted and quickly removed [on Hugging Face](https://huggingface.co/Samsung/MythoMax-L2-13B).
   - A member joked, *"Can someone do that for **OpenAI** and 'release' **GPT4Chan**? Or **Anthropic**, that would be priceless.*"



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Users Get Shortcut to Quick Chat**: Users can now click on **model icons** in the grid to initiate a **quick chat** with a specific model, streamlining the user experience as shown in the [attached image](https://cdn.discordapp.com/attachments/1092729520181739581/1372711951309738014/image.png?ex=6827c520&is=682673a0&hm=2548e2d91f09d872bd9bf58df2a1effa46101b1fef3b32a136cc594fa203e7c0).
   - The new feature lets users start quick chats with individual models by simply clicking on the model icons, greatly improving efficiency because it *bypasses the need to open the entire group*.
- **DeepSeek v3 is a MoE Model**: [DeepSeek v3](https://deepseek.com/blog/deepseek-v3) is a **Mixture of Experts (MoE)** model, meaning it activates only a subset of its parameters during inference.
   - Even though all the parameters are loaded into VRAM, only the parameters relevant to the prompt are computed, making the inference speed much faster.
- **Urban Corvids Switch to Peanuts and Cat Food**: Users observed that **corvids** (crows and magpies) are only eating **peanuts and cat food** instead of normal bird food.
   - It was suggested that urban corvids have adapted to a diet closer to trash and prefer alternatives to standard bird food, as their diet has changed over many generations.
- **Bypass Country Restriction with a Proxy**: A user shared an error message from OpenAI indicating that their country, region, or territory is not supported, which they bypassed using a **proxy**.
   - This avoids geographic restrictions causing the error.
- **`Qwen3` Needs Toggle to THINK**: For `qwen3`, it needs to be forced to think with `/think` or `/no_think` to toggle on and off thinking.
   - It was reported that `/no_think` functionality had a bug and OR needs to auto route away.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Samsung Enters LLM Race**: A member announced the release of new models from **Samsung**, including the **MuTokenZero2-32B** model and **MythoMax-L2-13B** model.
   - Another member indicated that the models were a work in progress.
- **LangGraph Powers Complex Flows**: Members shared a link to the **LangGraph** documentation ([LangGraph Course](https://huggingface.co/learn/agents-course/unit2/langgraph/first_graph)), highlighting its use in building agentic workflows and complex conversational flows.
   - **LangGraph** excels at overseeing intricate dialogue paths and multi-agent setups.
- **AsianMOM Roasts You!**: A member introduced **AsianMOM**, a **WebGPU Vision-LLM app** that roasts you like your mom in-browser using **SmolVLM (500M)** and **LLama 3.2 (1B)**, and available on [asianmom.kuber.studio](https://asianmom.kuber.studio/).
   - The creator expressed that this funny little project genuinely taught them so much about **WebML** and **Vision models**, noting that *the technologies we're getting with WebML will 100% democratize AI access*.
- **DistilRoberta's Accuracy Questioned**: A member questioned why the **DistilRoberta** version of a model has more downloads than **Roberta**, wondering if it's better for emotion detection despite potential accuracy differences.
   - Another member explained that **DistilRoberta** is a lighter version of **Roberta**, trained to balance computational cost and accuracy, but theoretically has lower accuracy due to fewer weights.
- **Agent Course Template Triggers**: A member reported that the **First_agent_template** worked initially but now consistently throws errors, and asked if they'd run out of credits.
   - Another member noted *this space for Unit 3 has error and needs to be fixed* [Unit 3 Agentic RAG](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Ducky Bedtime Stories Quack into Action**: A member created an audiobook of **ducky bedtime stories**, reading with appropriate energy and enthusiasm, using different voices for each character.
   - The expert speaker is a duck and can only say **"QUACK!"**, with the volume decreasing as the audiobook progresses.
- **Notebook LM helps Achieve Deep Focus**: A user discovered that running Google's **Notebook LM** podcast of a chosen book with **YouTube music** on a low volume helped them enter **deep focus at work**.
   - The user recommends a **loop button** on the podcast option and an integration with **YouTube Music** for a richer experience.
- **Pakistan Patrons Ponder VPN Pathways**: A user inquired about the app's lack of availability in **Pakistan**, and another suggested using a **VPN** to download it.
   - Another user, an **Android app tester**, pointed out issues with voice review personalization within the studio.
- **Link Lurkers Launch Looming Lies**: Users were cautioned about **scammy links** promising *free gifts*, *easy cash*, or *amazing deals*, and were advised to think twice before clicking such suspicious links.
   - It was emphasized that links offering freebies are major red flags, and users should always protect their personal information.
- **Podcast Plan Pays Off**: A user hit their **100-podcast max** on NotebookLM and intends to download the **WAV** files, convert them to video podcasts, and upload them to **YouTube** or **Google Photos**.
   - Another user replied that *this is smart*, with another user replying *I do something similar*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research and Solana Foundation partner for Decentralized AI**: Nous Research is co-hosting an event with the **Solana Foundation** in NYC on **May 22**, highlighting efforts to democratize intelligence through **Decentralized AI**. Registration is available [here](https://lu.ma/39b7e9pu).
   - The event will feature discussions on **Psyche**, Nous’s project focused on democratizing intelligence.
- **Psyche races into Hyperdrive**: The training rate for **Psyche** is **12B tokens per day**, while processing the entire **20T tokens** dataset is estimated to take almost **2000 days**.
   - Contributors can support model training through donations to the mining pool on the [Psyche Network](https://psyche.network/) or by contributing to the codebase on [GitHub](https://github.com/PsycheFoundation/psyche), spurring calls for more GPU power.
- **Meta Tackles AR and AI Integration**: **Meta** faces challenges in integrating **AI** into its smart glasses, which could potentially make its **AR** investments obsolete if not properly managed.
   - Despite this shift, **Meta** is continuing **AR** research with projects such as [Project Aria](https://www.projectaria.com/).
- **Smart Glasses Crave Agentic AI**: The general consensus is that **smart glasses** need *real agentic AI* to effectively interpret and interact with the user's environment, as demonstrated by [Sesame](https://app.sesame.com/).
   - Members are prompting calls for *an open smart glass AI* to foster innovation towards more useful integrations.
- **Grok's Glitches in South Africa**: Discussion arose whether **Grok's** issues in South Africa stemmed from tweaked steering vectors or *clumsy prompt updates*.
   - One member stated *Absolutely no basis to back it up but I am voting clumsy prompt*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TritonBench Benchmarks Bomb on AMD GPUs**: A member found that about **7 kernels throw memory access fault errors** when running [TritonBench benchmarks](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) on an AMD GPU.
   - One example provided was [chunk_gla_fwd.py](https://github.com/thunlp/TritonBench/blob/main/data/TritonBench_G_v1/chunk_gla_fwd.py) which throws an `Unknown Reason` error, and the member requested assistance to pinpoint the cause.
- **CUDA IPC Memory Handles String-Serialized**: A member explored using `cudaIpcGetMemHandle()` for single GPU multiprocess communication and found that `cudaIpcMemHandle_t` can be string-serialized.
   - This enables a straightforward producer-consumer setup for sharing memory handles, sidestepping more complex inter-process communication methods for single GPU sharing.
- **Tracing Fused Operations Requires Careful Code Reading**: A member inquired about mapping fused operations back to their original model code after compiler fusion, and another member replied with a [link to docs](https://docs.pytorch.org/docs/main/torch.compiler_inductor_provenance.html) regarding `inductor_provenance_tracking_node_mappings_<number>.json` files.
   - The member was unsure how to easily map the exported program's graph to the original model code without careful reading.
- **Pipeline Parallelism yields no concurrency gains**: A member experimented with pipeline parallelism using `torch.autograd.graph.saved_tensors_hooks` to manage activations across separate CUDA streams, aiming for concurrent forward and backward passes, referencing the [docs](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks).
   - Despite successful implementation without race conditions, the member observed minimal concurrency gains due to the model's kernel occupancy, deeming it a *"fun experiment though!!"*
- **MI300 Heats Up with New Benchmarks**: Several users submitted new benchmarks on the **MI300** across different leaderboards, including `amd-fp8-mm` and `amd-mixture-of-experts`.
   - Multiple successful submissions were recorded on the `amd-fp8-mm` leaderboard, with times ranging from **155 µs** to **3.28 ms** on the **MI300**, while the `amd-mixture-of-experts` leaderboard saw frequent entries, with multiple users achieving personal bests, such as **6233 ms** and **6247 ms** on the **MI300**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenMemory MCP Opens Memory**: A member shared [OpenMemory MCP](https://mem0.ai/blog/introducing-openmemory-mcp/), a new open-source project that aims to provide a unified memory management layer for AI applications.
   - The community reacted with praise calling it a cool unified memory management layer for AI apps.
- **Ongoing Grok Troubles Tracked**: Ongoing issues with **Grok** are being tracked in [this Discord channel](https://discord.com/channels/822583790773862470/1036726703730466896/1372435415490887770).
   - No further context was provided.
- **Microsoft Axes TypeScript Talent**: Microsoft fired the **TypeScript** dude without warning, sparking dismay, as seen in [this tweet](https://x.com/brodyford_/status/1922726909365879039?s=46).
   - Community members expressed that the firing happened without warning.
- **Agentic Tooling Outpaces Big Tech**: Members discussed a shift happening with **agentic tooling**, expressing hope that indie devs can use it to outpace big tech and corporations.
   - It was suggested that *getting the computers to do the right thing well* is a harder problem to solve than *do the wrong thing well*, especially given internal incentive structures in corporations.
- **FUNapis Succumbs to Bing's Chatbot**: A member suggested that **FUNapis** died so **Bing** can sell their chatbot wrapper of the API, as seen in [this tweet](https://x.com/williambryk/status/1923062696095711374?s=46).
   - No further community context was provided.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Optimize Data Loading and Preprocessing**: A member is optimizing their data loading and preprocessing pipeline to avoid being bottlenecked by resources due to bad tooling as they work by themselves outside of academia or industry.
   - They hope that this work will benefit all future audio data + mech interp work at scale.
- **DNN Training Plagued by Data Stalls**: Discussion highlights concerns that **CPU-based preprocessing** might bottleneck **DNN training pipelines**, particularly in audio modality contexts, referencing the paper [Lotus: Characterization of Machine Learning Preprocessing Pipelines via Framework and Hardware Profiling](https://www.computer.org/csdl/proceedings-article/iiswc/2024/560300a030/22f0GQCjZGo).
   - Members debated the benefits of optimizing CPU workload versus potential GPU bottlenecks.
- **BlackboxNLP** heads to EMNLP 2025**: The 8th edition of **BlackboxNLP**, will be co-located with [EMNLP 2025 in Suzhou](https://blackboxnlp.github.io) this November.
   - They will feature a [new shared task](https://blackboxnlp.github.io/2025/task) on **circuits/causal variable localization in LMs** using the recently released [MIB Benchmark](https://mib-bench.github.io/) with a submission deadline of August 1st.
- **False Flags in MCQ Evaluations**: An issue was identified where **MCQ evaluations** like **MMLU** flag model outputs as *false*, even when models assign the highest probability to a specific option based on **NLL values**.
   - This issue is especially prominent with smaller models, indicating a potential bias or limitation in how these models handle multiple-choice questions.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Client Seeks Server Emulation Advice**: A member needs help emulating an **MCP server**, after starting the client/server handshake, *client -> method: initialize* and *server -> method initialized*.
   - They seek insights on the intermediate steps to properly implement an **MCP server**.
- **Chainlit Query Parameter Conundrums**: A member is struggling to access **query parameters** from the URL within **Chainlit**, despite efforts using FastAPI middleware.
   - The member tried passing tokens and decoded dictionaries but failed to access them, seeking solutions to properly retrieve query parameters.
- **Jinko MCP Courts Hotelier AI Agents**: The community announced the creation of the **Jinko MCP** for developers to build **AI agents** that can sell hotels, with the [Jinko MCP GitHub repository](https://github.com/jinkoso/jinko-mcp/blob/master/README.md) now available.
   - The new tool provides access to *2M+ hotels*, enabling search, booking, payment, and customer service functions.
- **Smithery Server Square-off with Claude Desktop**: A member requires assistance integrating a **Smithery-installed server** with **Claude Desktop** using their **OpenRouter key**.
   - The member questions if the model used in the MCP tool configuration needs to align with that in Claude (e.g., **sonnet-3.5** in MCP config vs. **sonnet 3.7** in Claude).
- **Shortwave Surfs MCP Client Support**: **Shortwave** now offers **MCP client support**, supporting both **HTTP MCP** & **stdio MCP**, and provides one-click toggles for integrations like **Hubspot**, **Notion**, **Zapier**, **Asana**, and **Linear**, according to their [blog post](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/).
   - Further details are available in their [documentation](https://www.shortwave.com/docs/how-tos/using-mcp/).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo stdlib documentation location clarified**: The Mojo stdlib documentation is generated directly from the stdlib itself, as clarified in a thread, and is modifiable directly, as opposed to in the `/mojo/docs/manual` directory, and a member fixed the doc in [PR 4530](https://github.com/modular/modular/pull/4530/files).
   - This addresses [Issue 4482](https://github.com/modular/modular/issues/4482) which a member was looking to update.
- **Members wrestle with pointer declarations in Mojo**: Members sought help with declaring a **Pointer** in a **Mojo struct**, with one suggesting making the `Op` generic over *origin* if they want to borrow.
   - It was clarified that Mojo requires the *origin* to be part of the type, making it a parameter, as related to the **borrow checker**, as explained in the [Mojo Lifetimes Documentation](https://docs.modular.com/mojo/manual/values/lifetimes/).
- **MAX plagued by installation issues**: A member encountered errors during MAX installation, indicating missing essential functionalities like **tensor operations** (`tensor`, `nn`, `zeros`, `ones`, `matmul`, `add`, `mul`).
   - This is preventing the continuation of a MAX-only implementation for a diffusion **LoRA trainer** due to weak tensor support in Mojo and MAX.
- **Hybrid MAX and PyTorch Approach more viable, for now**: Due to missing tensor operations for **MAX-only LoRA trainer**, Claude AI suggested a *hybrid approach* using **PyTorch** and **MAX's interoperability features** as a more viable solution for immediate implementation.
   - A member ensured that tools like Claude have access to the current [Modular GitHub repository](https://github.com/modular/modular) and [documentation](https://docs.modular.com) to avoid LLM **hallucinations**.
- **Karpathy's micrograd gets ported to Mojo**: A member is learning **Mojo** by porting **Karpathy's micrograd**, sidestepping the lack of lambda function support, and another member shared their similar project, [momograd](https://github.com/dorjeduck/momograd), created last year as one of their first **Mojo** learning projects.
   - The **momograd** project has not been updated to the latest **Mojo** versions but serves as an example of the community interest.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere courts cooperation with community cause**: A tech-adjacent **nonprofit** seeks event sponsorships and grants from **Cohere**, aiming for a partnership to further their tech-focused initiatives.
   - Interested parties should reach out to [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com) to connect with the appropriate **Cohere** staff.
- **Cohere Classify API Clicks with Clients**: Users laud the **Cohere Classify API**, expressing eagerness to scale its usage to millions of entries and suggesting they plan to contact **support@cohere.com** to request a rate limit increase.
   - The increase would help to explore the feasibility of running the **API** at scale without extensive waiting times.
- **SiliconFlow Setups Stirring with Screenshots**: A user locally modified the **SiliconFlow** endpoint, demonstrated in [an attached image](https://cdn.discordapp.com/attachments/1218409701339828245/1372456961928331335/image.png?ex=68278066&is=68262ee6&hm=93a39c60bf386774de78fb37f4469ba201e375e9e173d6e6427db1a09ab94a1a&).
   - Additionally, screenshots of **Gemma 3 4b 4bit** and **Llama 3.2 3B 4bit** were shared, showcasing their implementations in separate [attached image](https://cdn.discordapp.com/attachments/1218409701339828245/1372462317081591818/image.png?ex=68278563&is=682633e3&hm=fe00ba43ec32dd40fe99c2ec904aba7ae3f078b9df968075cbb55291717d751b&) and [another image](https://cdn.discordapp.com/attachments/1218409701339828245/1372537263271051304/image.png?ex=6827cb30&is=682679b0&hm=f84185fbfe33d5e9af456ef71aaf4ff03339eabf81cae93d0be9a1901c188f69&).
- **Web AI Engineer Wants Work**: A seasoned **Web, AI Engineer** with **7 years** of fullstack experience introduced themself, with proficiency in modern web technologies.
   - They are skilled in **React(Next), React Native(Expo), Flutter, Vue(Nuxt), Svelte, Astro** and tools like **Node/Express, Django, Nest.js, Go, Web3.js, Shopify, Wordpress, TailwindCSS, Shadcn, MUI, Docker, Kubernetes, AWS/GCP, LLM**.
- **Full stack fan of AI finds favor with frameworks**: A full stack developer with over **20 years** of experience has embraced **AI**, and enjoys building real-time applications with carefully crafted **UI** and **UX**.
   - They are a fan of **Nuxt** running on **Cloudflare** and using tools like **RooCode** and **Windsurf**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini Models' Structured Outputs implemented in DSPy**: Members discussed if **Gemini Models'** response schema, similar to **OpenAI's** structured outputs, is implemented in **DSPy**, and another member confirmed that it is.
   - It was also confirmed that **DSPy** dynamically builds the response schema.
- **Pydantic Models drive Structured Outputs in DSPy**: A member inquired about implementing **structured outputs** in **DSPy**, similar to **OpenAI** tools, including **nested outputs** or **JSON schema constraints**.
   - Another member replied to *just use signatures*, and pass **Pydantic models** or **Python TypedDicts** as output field types.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All's Vital Signs Fade**: Members speculate about the discontinuation of **GPT4All** due to the absence of updates since February.
   - The community expresses concern over the lack of communication from **Nomic** regarding a new version release.
- **Nomic Eyes Pay-to-Play Model?**: Speculation arises around **Nomic** potentially transitioning to a monetized platform.
   - The claim that *gpt4all is over* and **Nomic** is pivoting to monetization lacks substantiating evidence.
- **Jan.ai and LM Studio emerge as GPT4All contenders**: **Jan.ai** and **LM Studio** are mentioned as possible substitutes for **GPT4All** in light of recent concerns.
   - The discussion did not include why those were good alternatives or which features they had that might be beneficial.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Event-Driven Agents Assist Weaviate**: LlamaIndex unveiled a walkthrough demonstrating how to use **event-driven agent workflows** to construct a **multi-agent Docs Assistant** that writes webpages into **LlamaIndexDocs & WeaviateDocs collections** in Weaviate.
   - The orchestrator decides when to call the Weaviate QueryAgent for search, showcased in [this Tweet](https://twitter.com/llama_index/status/1923085124725506441).
- **Tig Coding Agent Makes Debut**: An open-source **(human in the loop) coding agent** called **Tig**, created by @rsrohan99 and built with LlamaIndex workflows, was highlighted.
   - Tig can write, debug, and analyze code across multiple languages, execute shell commands, and search the web as shown on [Twitter](https://twitter.com/llama_index/status/1923134285940441102).
- **LlamaIndex Tackles PDF Content Extraction**: A member requested advice on extracting content from a PDF using **LlamaParse** or **LlamaIndex**, specifically to extract the Table of Contents and isolate content and tables from a particular section based on a predefined name.
   - The user seeks guidance on setting up the instructions or pipeline to detect the section from the TOC, isolate the content, and properly structure the extracted tables, with the right parameters for use in no-code tools like **n8n**.
- **AI Startup goes Vibe Coding**: An AI startup based in Korea is looking for passionate developers with **Vibe Coding** experience to partner on client projects.
   - The opportunity includes a fair revenue-sharing model and ongoing partnership, with requirements for strong communication skills, **GitHub links**, **Vibe Coding project references**, and English/Korean communication skills.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Network Stumbles on vLLM**: A member faced implementation failures while trying to deploy a custom **Torchtune network** on **vLLM**, despite following several tutorials.
   - A suggestion was made to convert the checkpoint to **HF format** for better syncing, also inquiring whether the model was registered with **vLLM**.
- **Custom Models Wrestle with vLLM**: A member reported difficulties implementing a custom model with a custom architecture within **vLLM**.
   - Another member shared a [vLLM guide on implementing custom models](https://docs.vllm.ai/en/latest/contributing/model/basic.html) to assist with the implementation.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda Workshop Teaches Agentic AI**: A [Lambda workshop](https://www.youtube.com/watch?v=VmjMIwwo9ag) is teaching how to build agentic applications using **Lambda's Inference API**, optimizing agent performance, and deploying agents in production.
   - Participants can apply for **$100 serverless API credits** by Friday 5/16 via [this form](https://forms.gle/UtVhmPS3mitS8Vxu7).
- **Nobel FutureTech Fireside Chat Details**: A fireside chat co-hosted by [Nobel FutureTech Group](https://nobel-futuretech.com/index.html) and Berkeley RDI is providing insights into the innovative ecosystem of the **Nobel FutureTech Genius Club**.
   - The session gives information on mentorship, funding, and collaboration opportunities, with a [livestream link available](https://www.youtube.com/watch?v=ft-2W00Rtg8).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Topk Bounty Asks for Revision**: A user questioned the 'move **topk**' bounty requirements, noting **topk**, **masked_select**, and **randperm_generator** already run off the CPU.
   - They proposed the bounty be revised due to functions like **_index_put_impl_** and **index_tensor** still requiring attention.
- **Index Functions Awaiting GPU Acceleration**: It was noted that **_index_put_impl_** and **index_tensor** are still running on the CPU.
   - The suggestion was to target these and other functions in the torch backend for GPU offloading.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Webinar Announced: Agentic Enrichment with Featureform**: A live webinar on **Agentic Enrichment** with **Simba Khadder**, Founder of Featureform, is scheduled for **Tuesday, May 27th at 8 AM PT** and will cover how to unlock data for AI agents using **MCP**, and you can sign up [here](https://buff.ly/zeoH55Y).
   - The webinar will discuss the missing layer of infrastructure needed for AI agents to access real-time, internal business data, highlighting the limitations agents face due to data access rather than intelligence.
- **Featureform Tackles LLM Data Access**: The webinar will cover the need for better internal data access to unlock the full potential of AI agents, detailing the **three key components of agentic enrichment**: semantic catalog, low latency serving, and governance.
   - It will demonstrate how **Featureform** enables this data access, making agents more useful and powerful in production environments, with real-world examples of improved workflows in AI systems.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Floats SWE-1 Models**: Windsurf has launched the **SWE-1** family of software engineering models, including **SWE-1**, **SWE-1-lite**, and **SWE-1-mini**, detailed in a [blog post](https://windsurf.com/blog/windsurf-wave-9-swe-1) and [launch video](https://youtu.be/LhA9pVpwgdY).
   - Windsurf claims the new models will accelerate software development by **99%**.
- **SWE-1 Shows Claude 3.5-Level Performance**: The **SWE-1** model is advertised to have *high-reasoning*, *tool-capable*, and *Cascade-optimized* performance comparable to **Claude 3.5** but at a reduced cost.
   - The models are trained using a unique *"flow awareness" approach*, understanding the timeline between humans and AI across development surfaces.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI Tinkerers to Host AI21 Meetups**: AI Tinkerers will host meetups with **AI21** to discuss the newly announced [Maestro platform](https://www.ai21.com/maestro/), a tool for planning and orchestration.
   - The meetups are free and open to the public with registration required for events in New York City, Paris, and San Francisco - see links above.
- **AI21 Launches Maestro Planning Platform**: **AI21 Labs** recently revealed **Maestro**, a platform designed for planning and orchestration in AI systems.
   - This platform seeks to equip developers with the necessary tools and infrastructure to construct more complex and effective AI applications.



---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1372427390973644820)** (869 messages🔥🔥🔥): 

> `Perplexity Pro role, App answers cannot be read on the web version, Research function down, Deep Research broken, Deepsearch rate limits` 


- **Pro role troubles, solved by mods**: A member inquired about obtaining the **Perplexity Pro role** after purchasing the Pro plan and a moderator sorted it out manually and stated that [they are currently reworking how to get the Pro role on Discord](https://discord.com/channels/1047197230748151888/1111786888626438245) in the meantime.
   - Another user confirmed that [having the same email for Discord and Perplexity is not a requirement](https://discord.com/channels/1047197230748151888/1111786888626438245).
- **Mobile App answers causing Webpage Glitches**: Users reported that answers obtained from the **Perplexity mobile app** cannot be read on the web version, prompting an error message; this issue doesn't occur when the answer is obtained directly from the webpage, as demonstrated [here](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA).
   - A user reported this issue to customer service **10 hours ago** and they said the issue has been reported to the technicians but it has not been fixed yet.
- **Research function is temporarily offline**: Some users asked, *Is the Research function down?*, with some reporting that they were unable to use **Perplexity**, and [the status page](https://status.perplexity.com/) reflecting an outage.
   - This was fixed within the hour, and some users who were experiencing issues being logged out of their account found that clearing their browser cache resolved the problem.
- **Deep Research Doomed? Users question value**: Multiple users reported that **Deep Research** mode appeared to be broken, citing issues such as the system defaulting to regular search on the web, only using a limited number of sources, and that pro search only uses 10 sources, instead of 20.
   - One user stated: *I legit just use Perplexity because it reads 20-40 sources per query. If that's dumbed down, I see no reason to pay for it.*
- **Debate Grok vs Perplexity**: Members discussed the use of **Grok**, with one sharing *We don't talk about it. It's worse than regular search in some cases*, while others discussed the **Deepsearch rate limits** and **response quality**.
   - Another noted that Grok scrapes the web very well but it sucks at one shot sometimes but the rate limits compensate for that if you elaborate and cuss enough.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1372487492531392613)** (1 messages): 

> `23andMe files for Chapter 11` 


- **23andMe Prepares for Restructuring**: A link was shared regarding [23andMe filing for Chapter 11 bankruptcy](https://www.perplexity.ai/search/23andme-files-for-chapter-11-b-msxlvXlmQCK0MLt0UUeTcg).
- **Legal and Financial Restructuring Imminent**: The Chapter 11 filing suggests that **23andMe** is facing significant financial challenges and is seeking legal protection to reorganize its debts and operations.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1372480085369688125)** (6 messages): 

> `Sonar API, Perplexity hackathon Credits, sonar model` 


- **Sonar API Key Woes Plague Hackathon**: A member is facing issues acquiring the **Sonar API** for a hackathon project because it requires credit card details, which they don't have, and seeks a way to access the API for free, solely for demo purposes.
   - Another member also reports the same issue.
- **Hackathon Credits MIA?**: A member reports not receiving their **Perplexity hackathon credits** for 2 days and requests assistance.
   - The member is looking to use the API to gather more info on a list of contacts, like it does on the web.
- **Sonar API output mismatch**: A member reports that the answers generated using the **Sonar API** are different from those generated in the [Perplexity AI playground](https://labs.perplexity.ai/).
   - They speculate it might be due to their system prompt, and provides a link to the [Perplexity AI Model Cards](https://docs.perplexity.ai/models/model-cards).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1372423709020655717)** (472 messages🔥🔥🔥): 

> `Manus Vibe Coding Livestream, Johnny's credit farming, Femboys, invite links gone, Credits Usage` 


- ****Manus Hosts "Vibe Coding" Livestream****: Manus hosted a "Vibe Coding with Manus" livestream from San Francisco, featuring the Manus SF Fellows, available on [YouTube](https://youtube.com/live/w4XegM6dOgc?feature=share).
   - The livestream showcased coding projects, fostering community engagement.
- ****Johnny the Credit-Farming Genius****: A user humorously highlights how Johnny is *"farming Manus daily"* compared to another user *paying for manus to make a femboy detector*
   - This illustrates a humorous contrast between exploiting the platform for free credits and using it for entertainment.
- ****The Femboy Detector App****: A user created an app that determines if a person is a femboy or not, using Function Calling to get currency rates from [wise.com](https://wise.com).
   - The app outputs "femboy" for male names, leading to comedic accusations of users being labeled femboys: *MANUS'S API IS LYING!*
- ****Some Users Lost Invite Link Feature****: Some users have found that the option to generate invite links has disappeared from the UI.
   - It seems that the invite link generation feature is no longer available to all users. Some speculate this affects free users only.
- ****Credits Usage Concerns Arise****: Users expressed concerns about the cost of credits, with one noting that a PDF report cost 500 credits and a DCF on Google cost 1500 credits.
   - Some believe the credit usage is too expensive, especially for complex tasks and look forward to alipay as a payment method.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1372430171654848522)** (401 messages🔥🔥): 

> `Gemini 2.5 Pro Reasoning Time, Elon's Grok 3.5 Release Delay, LLMs Steering Attention, LMArena Model Funding, O3 Pro on Arena` 


- **Geminiyo Sparks Performance Gap Thoughts**: Members pondered whether [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/rest/v1beta/GenerationConfig) will have longer reasoning times compared to other reasoning models, given it is a *heavier model*.
- **Grok 3.5 Release Delayed Due to Elon's Far-Right Fine-Tuning?**: Theorizing that **Elon** delayed the release of **Grok 3.5** because fine-tuning it to the far right *wasn't successful*, one member was promptly corrected by another noting it was *false information*.
   - Another chimed in, *Why would he include political stuff into the system prompt*? while attaching a screenshot about the dangers of programming society with social networks and LLMs.
- **LLMs Can't Steer Attention Like You Think**: Despite claims that **LLMs** can steer attention, particularly with **Grok's** Twitter timeline, one member clarified that *this isn't steering attention where you need them to*, but rather a plain announcement.
   - They further linked to [Transluce's observability interface](https://transluce.org/observability-interface) as a tool to play with feature steering, but cautioned it's not particularly useful in practice yet.
- **Labs Give LMArena Credits, not Cash**: In a discussion about how **LMArena** funds its models, a member pointed out that *it's not just the companies themselves who pay for inferences*, suggesting **credit grants** are provided to **LMArena** instead of direct monetary payments.
   - Another member added that big labs give **LMArena** endpoints for their models, with valuable data on human preference being collected as a result.
- **O3 Pro Arrival on Arena Remains a Mystery**: Despite speculation about an **O3 Pro** release, a member stated *O3 pro wont come to arena lol*, to which a moderator replied *I can't confirm if/when new models are arriving on arena, but will be sure to put out announcements when I can*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1372440346188841010)** (208 messages🔥🔥): 

> `Quantized versions (GGUFs, QNL), Multi-GPU Finetuning, SLM vs LLM, Qwen3 model for translation, H200 Temp` 


- **QNL is Faster than Standard GGUFs**: **QNL** is reportedly faster than standard **GGUFs**, but performance benchmarks are still pending; [Unsloth Dynamic 2.0 GGUFs documentation](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) provides more details.
   - One member inquired about combining **GPTQv[1-2]** with **GGUF + imatrix** for accuracy improvements.
- **Multi-GPU finetuning via Accelerate**: To achieve multi-GPU finetuning, users can try using **Accelerate** with Unsloth.
   - While consumer-grade GPUs like the **3090** offer **24GB VRAM**, companies often prefer **H100s** for local AI tasks, despite not being top-level in datacenter contexts.
- **LLMs vs SLMs**: While smaller models (SLMs) may not be as smart out-of-the-box, they can become competitive through fine-tuning on specific tasks; **Qwen3-4B** is a good starting choice for models needing decent reasoning power.
   - One member highlighted that **Qwen3 4B** is even better than **Mistral 7B**.
- **Qwen3 translation capabilities**: For translating datasets in Mandarin, **Qwen3** is suggested because of its extensive pretraining data, with the recommendation of **14B** parameters for adequate performance.
   - One user reported success using **30B** models with **Ollama** on Kaggle, citing speed requirements for processing millions of strings.
- **H200 Temperature Expectations**: Normal operating temperatures for **H200** cards in cloud environments like Runpod are around **80-85°C**, which is considered acceptable for production cards.
   - One user reported running below **80°C**, indicating good thermal performance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1372450131231903796)** (10 messages🔥): 

> `Fine-tuning AI models, Jarvis-like AI clones, Continuous Thought Machine on Flappy Bird` 


- **Fine-Tuning AI Models: Heaven's Take**: A member suggested that fine-tuning would be possible to study an AI model.
   - Another member agreed, mentioning their years of experience in **AI-related projects**.
- **Jarvis Clones Found on YouTube and GitHub**: A member stated that there are dozens of **Jarvis-like clones** on **YouTube** and **GitHub**.
   - They suggested a quick Google search to find them.
- **Continuous Thought Machine Tries Flappy Bird**: A member trained a **Continuous Thought Machine (CTM)** from the **CTM paper** on **Flappy Bird**.
   - After **~750 episodes**, it can only make *one pipe gap* sometimes, indicating the difficulty of the task.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1372441995993288735)** (120 messages🔥🔥): 

> `Qwen3 DPO Training, Orpheus-3B Fine-tuning Issues, Mistral 7b VRAM Usage, Epoch Display Bug, BLIP2 and Transformers` 


- **Qwen3 Gets DPO'ed**: A user inquired about training **Qwen3** using **DPO** and whether a sample notebook similar to Zephyr exists, to which another user responded with a link to [their Kaggle notebook](https://www.kaggle.com/code/etherl/kaggle-llama-3-2-1b-3b-conversation-multigpu) for **Llama 3** fine-tuning on multi-GPU setup, indicating a similar approach might work.
   - The notebook includes steps for conversation models, using accelerator, and other optimizations useful for DPO.
- **Orpheus-3B's Loss Landscape & Colab Crashes**: A user reported fluctuating loss (4.5-3.9) while fine-tuning the **Orpheus-3B** TTS model using Unsloth and inquired about its normalcy.
   - Another user responded that this is normal and that multiple epochs can potentially reduce the loss to **1**, also sharing a code snippet for using **SNAC** for inference and troubleshooting Colab crashes due to account login access issues.
- **GPU RAM gets Mistral'ed**: A user faced issues training **Mistral 7B** on an NVIDIA RTX A2000 (8 GB) despite Unsloth benchmarks suggesting it should be possible with QLoRA 4-bit, and sought advice on potential misconfigurations.
   - A user pointed out that **batch size**, **r value**, and **max_seq_length** significantly impact VRAM usage, also suggesting the user ensures no other processes are consuming GPU memory.
- **LLM Epoch Tracker Bug Squashed**: A user noticed a discrepancy in the training output, where the progress bar appeared full after completing all steps, but the epoch count remained at **1/2**, leading to confusion about whether the training completed both epochs.
   - Another user suggested it might be a minor display issue, as the number of examples and steps aligned with completing two epochs.
- **Vision Models ride LLM Compatibility Wave**: A user asked if **Unsloth** supports BLIP2 finetuning and how to check if **Transformers** supports it.
   - A user confirmed **Transformers** supports **BLIP2**, referencing [a PEFT notebook](https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb) and [Hugging Face documentation](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/blip-2#usage-tips) and stating *Unsloth is pretty much compatible with any transformers model*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1372660755551486012)** (3 messages): 

> `WebGPU Vision-LLM app, Geminized Qwen3 MoE` 


- **AsianMOM roasts you in-browser**: A member created **AsianMOM**, a [WebGPU Vision-LLM app](https://asianmom.kuber.studio/) that roasts you like ur mom in-browser.
   - It uses **SmolVLM 500M** and **LLama 3.2 1B** and works right in your browser without having to install anything, thanks to Transformers.js and HF.
- **Geminized Qwen3 MoE released**: A member released a [Geminized version of Qwen3 MoE](https://huggingface.co/Ba2han/Qwen3-30B-A3B-Geminized-v0.2), a merged bf16 LoRA trained on **~450 examples** with 1 or 2 turns, around 250 of these examples are diverse, human prompted conversations directly from **Gemini2.5**.
   - *Use "You are an assistant with reasoning capabilities." system prompt to trigger Gemini style reasoning.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1372469766551375962)** (5 messages): 

> `Intellect-2, Solo Author, Mechanistic-Interpretable Ethics` 


- **Intellect-2 Boasts Performance**: A member shared an [ArXiv link](https://arxiv.org/abs/2503.15758) to **Intellect-2**, implying that it's performing well and *bragging*.
   - The same member shared a [news.smol.ai link](https://news.smol.ai/issues/25-05-12-intellect-2) related to the ArXiv paper.
- **Solo Author Achievement**: A member highlighted a [personal website](https://kvnmln.github.io/ecoart-website/CA11.1.html) noting that it was a **solo author** achievement.
   - It seems like a nod to the effort involved in solo research and development.
- **Ethics Interpreted Mechanistically**: A member shared a link to a Hugging Face Space called **Mechanistic-Interpretable Ethics-Cell automata**.
   - The project is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/KvnMln/Mechanistic-interpretable-Ethics-Cell-automata).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1372426294259617802)** (193 messages🔥🔥): 

> `Grok's Alignment Issues, Gemini 2.5 Pro Removal, Free API Credits, Aider Token Usage, Consolidate Command for Aider` 


- ****Grok**'s Alignment Issues and X's Joke Status**: Concerns arose about [Grok](https://x.com), Elon Musk's LLM, after it made wild claims about **white genocide**, leading some to distrust it for development and consider **xAI** a joke.
   - One member joked that they only used **xAI** to ask who the biggest threat to democracy on **X** was, and it admitted it was **Elon**.
- ****Gemini 2.5 Pro**'s Brief Stint in Copilot**: Users noted that **Gemini 2.5 Pro** was briefly available in Copilot, but it has been removed.
   - One speculated that **Microsoft** might invest more in open-weight models due to its rocky relationship with **OpenAI**.
- **Generous API Credit Giveaway**: A user offered free **API credits** for **Gemini, Claude, and OpenAI** to those building interesting projects, to test new infrastructure.
   - Others joked that free tokens are free tokens, or expressed interest in using those tokens to contribute to the [aider project](https://github.com/Aider-AI/aider/issues).
- **Managing Aider Token Usage Savvy**: Users discussed the importance of managing token usage in Aider, recommending the use of `/clear` to reduce context and `/add` only necessary files.
   - They suggested using free models like **Gemini 2.5 Flash** via Google AI Studio or **OpenRouter** like Deepseek v3 0324, paired with copy-pasting from **Gemini 2.5 Pro**.
- **Aider is getting a `/consolidate` command**: A member is planning to add a `/consolidate` command to aider to roll each long chat into a single, fully-specified prompt, using a fresh single-turn request using the main model, in response to the [LLMs Get Lost In Multi-Turn Conversation](https://arxiv.org/abs/2505.06120) paper.
   - The goal is to address the issue of **LLMs** losing context in multi-turn conversations by rewriting previous turns into a clean prompt.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1372425455902392391)** (99 messages🔥🔥): 

> `Commit Prompt, Rate limited, Black box for code, aider config, home directory` 


- **Streamline commit messages**: A user shared a cool tip for **YML configurations**, by adding `AIDER_COMMIT_PROMPT="Respond with exactly two plain text lines and nothing else. Line 1: your commit title (max five words, no labels or prefixes). Line 2: Changes: file1,file2,... . Do not include the word Title or any markdown, headings, quotes, or extra text."` for generating **concise commit messages**.
   - Another shared what the base prompt for the commit message should look like, formatted with `<type>: <description>`, e.g. `fix: add feature` instead of `added feature`.
- **Configure dark mode, YAML style**: A user asked how to enable the *black box* for code snippets rather than the default *white* box.
   - Another member pointed them to the [configuration documentation](https://aider.chat/docs/config/aider_conf.html) and suggested setting `dark-mode: true` in their `.aider.conf.yml` file.
- **Thinking budget for Gemini?**: A member inquired about the behavior of **max tokens** for **Gemini**, asking if it truncates or actively budgets its response.
   - A fellow member suggested checking the docs for `thinking_budget` and that there may be related extra parameters.
- **Configuring O3 and GPT-4.1**: A user asked how to use `o3 (high) + gpt-4.1` in `--architect` mode.
   - Another member provided a link to the [architect documentation](https://aider.chat/2024/09/26/architect.html) and a sample command: `aider --model openrouter/google/gemini-2.5-pro-exp-03-25 --editor-model openai/gpt-4.1 --architect`.
- **Insufficent funds for O3!**: A user encountered an error while running the command `aider --model openrouter/openai/o3 --editor-model openrouter/openai/gpt-4.1 --architect`.
   - It was found out they needed to have money in the OpenAi settings as well as OpenRouter, to call the O3 API.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

p0lyg0n: https://github.com/pig-dot-dev/muscle-mem
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1372625656361386196)** (1 messages): 

> `OpenAI to Z Challenge, Archaeological Sites, Amazon, GPT-4.1` 


- **OpenAI Launches Amazon Archaeology Quest!**: OpenAI announced the **OpenAI to Z Challenge** using **o3**, **o4-mini**, or **GPT-4.1** to discover previously unknown archaeological sites in the Amazon, inviting participants to share progress on X with the hashtag #OpenAItoZ.
   - The challenge details are available at the [OpenAI website](https://openai.com/openai-to-z-challenge/).
- **Explore the Amazon using OpenAI's latest tools!**: Participants are encouraged to leverage **OpenAI's o3**, **o4-mini**, and **GPT-4.1** models in a quest to unearth undiscovered archaeological treasures nestled within the Amazon rainforest.
   - Share your journey and findings on X using the hashtag **#OpenAItoZ** to connect with fellow explorers and showcase your contributions.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1372432556833902612)** (195 messages🔥🔥): 

> `GPT-4.1 Mini Smarts, GPT-5 Release, Gemini 2.5 Pro, OpenAI's Open-Source Model, Context Window Expansion` 


- **GPT-4.1 Coding Focus Sparks Debate**: Members debated the merits of **GPT-4.1** versus **GPT-4o**, with some asserting that **4.1** is superior for coding while others found **4o** to be more intuitive overall, and several claimed **4.1 mini** is the best small model.
   - Some users shared their prompt testing results and experiences across specific dialects to show their **GPT-4.1** preferences, while sharing [screenshots of the models](https://cdn.discordapp.com/attachments/998381918976479273/1372555924862140436/20250515_154222.png?ex=682733d1&is=6825e251&hm=206683cf5fa5279c41f8150a0645b58d3595bbfde5b3616003058b41a29e5cae&).
- **GPT-5 Speculation Ramps Up Release Date**: The community discussed the potential release timeline for **GPT-5**, with some anticipating a launch in **summer** (**June-September**) while others suggested a later release around **August-December**, and one member said, *Connecting the dots here - Sam said in the Feb 12 announcement that GPT-5 will solve the confusing names issue.*
   - Members expect the unified **o-models** will be unified with **GPT** in the **GPT-5** release.
- **Gemini 2.5 Pro Wins Fans but faces High Cost**: Users praised **Gemini 2.5 Pro**, noting its coding abilities, reasoning skills, and large context window, with one saying *Gemini 2.5 Pro is actually really good at coding*, while acknowledging that [its free availability](https://discord.com/channels/974519864045756446/998381918976479273/1372558644578881576) is likely temporary due to its high running costs.
   - One user has switched to **Gemini 2.5 Pro** due to its **1 million context window** stating that they simply can’t work with **GPT’s tiny 32k context window anymore**.
- **OpenAI's Open-Source Model Debuts in Mid-June**: The community is anticipating the release of **OpenAI's open-source model** around the middle of **June**, but some doubt they will be able to run it locally.
   - They predict the model will be *probably >30B params but <100B*.
- **Bidding on Vast.ai Is Addictive**: A member claimed that *bidding on interruptible compute on vast.ai is addictive if you have a workflow suited for it*, and that they were *running like $40,000 worth of gpus for less than a dollar/hr lol*.
   - They suggest that the workflow is best if *suited for it*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1372500838529892402)** (21 messages🔥): 

> `GPT-4.1 vs GPT-4o, Fine-tuning Datasets for Story Generation, Mathematics in GPT-4.1` 


- **GPT-4.1 races ahead of GPT-4o**: Members discussed whether the new **GPT-4.1** version is better than **o3 mini** and **4o models**, citing the official [OpenAI announcement](https://openai.com/index/gpt-4-1/) that it's *slightly better at instruction following* and *better at remembering stuff*.
- **GPT-4.1's existence called into question**: One user jokingly suggested that **GPT-4.1** isn't real, referencing a [post from Sam Altman on X](https://x.com/sama/status/1923104360243835131?s=33).
- **Fine-tuning models for creative story telling**: A member asked which model is best for fine-tuning with a dataset of **200 stories** to consistently create similar, creative stories.
- **GPT-4.1's math skills questioned**: One user asked if **GPT-4.1** is good at mathematics, another responded *mostly no unless 4o is good in it*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1372522758575095860)** (2 messages): 

> `Research GPT Feedback, Multilingual Capabilities` 


- **Research GPT Seeks Feedback**: A member is requesting feedback on their [Research GPT](https://chatgpt.com/g/g-68236174e57c8191aa65e6ed815b8f46-reserch-for-me), in its final stages of refinement.
   - The creator is particularly interested in identifying potential issues in English, as it is not their first language, while noting that Korean functionality is satisfactory.
- **GPT's Multilingual Capabilities**: The GPT model appears to function effectively in Korean, indicating robust multilingual support.
   - However, the request for feedback highlights the importance of verifying performance across different languages, especially when the developer's proficiency varies.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1372522758575095860)** (2 messages): 

> `GPT feedback, English language issues, Korean language model performance` 


- **Feedback sought for Research GPT**: A member is seeking feedback on a [Research GPT](https://chatgpt.com/g/g-68236174e57c8191aa65e6ed815b8f46-reserch-for-me) and believes they are nearing completion.
   - They are performing final checks to iron out possible remaining issues.
- **Korean GPT excels, English GPT flounders?**: The member notes that their **Research GPT** functions very nicely in **Korean**, but they are unsure about potential problems in **English** due to it not being their first language.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1372423465621000292)** (192 messages🔥🔥): 

> `Cursor Pro vs. Free, Client Version Details, Claude 3.5 Sonnet, Gemini Pro Preview, Agent Rules Neglected` 


- **Confusions on Cursor Pro features**: A member asked whether they can use **all models** with their current plan and which models require an **API key**.
   - Another member clarified that *the models shouldn't give you an API key as all of them are held and supported by Cursor itself*.
- **Cursor's Client Version Details**: A member shared their Cursor client details ([Version 0.49.6](https://cursor.sh/docs/faq)) including **VSCode version 1.96.2**, **Electron 34.3.4**, **Chromium 132.0.6834.210**, and **Node.js 20.18.3** while reporting a 'no restart' message.
   - It was suspected that might be late for the Canadian time zones.
- **Claude 3.5 Sonnet Stuck in a Loop**: A member reported that **Claude 3.5 Sonnet** was getting stuck in a loop, with attached [images](https://cdn.discordapp.com/attachments/1074847527708393565/1372432736752762930/Gq9pkJRaAAECBOI.png?ex=682769d7&is=68261857&hm=02da0d21e3c19e9a58420fc3ee21f55a47edd8f0e64e700331b1b3ae6bc99a9d&) for reference.
   - Eventually, it was saved by context limit.
- **Resetting Contexts with a Slash**: Members discussed using the **/reset** command to clear the context in Cursor, but some users do not like this functionality.
   - One member noted that after typing **/reset**, *nothing will show up, it will be executed silently*.
- **Gemini struggles editing**: A member reported that **Gemini Pro Preview** spends considerable time deciding on code changes and then struggles to apply those edits.
   - Another member mentioned that **version 0.50.4** promised improved apply functionality.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1372467631898230825)** (58 messages🔥🔥): 

> `Expanding left sidebar, Llama issues with fantasy prompts, Token loss and punctuation issues, LM Studio API Vision Endpoint, Reka Flash Presets` 


- **Discord sidebar expansion sought**: A user inquired about expanding the left sidebar in Discord to permanently display icon names without hovering.
- **Fantasy Prompts funking with Llamas**: A user reported issues with various **Llama** models (3.1/3.3) producing undesirable outputs when used with fantasy-themed prompts.
   - The user attached [screenshots](https://cdn.discordapp.com/attachments/1110598183144399061/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&) illustrating **token loss**, **punctuation issues**, and even **partial word omissions**.
- **LM Studio's Vision API Endpoint Explored**: A user sought guidance on providing images to a vision-enabled LLM via the LM Studio API without using the Python library, specifically when using the OpenAI endpoint.
   - Another user pointed to the [LM Studio documentation](https://lmstudio.ai/docs/typescript/llm-prediction/image-input) and highlighted a cURL example demonstrating how to pass an image URL.
- **Llama.cpp Substitution Shenanigans**: A user asked about using a custom-built *llama.cpp* with LM Studio, but was told it is closed source and there is no way to substitute the Llama.cpp client reference.
   - A developer mentioned that while replacing *llama.dll* might be possible, it could lead to instability due to function signature changes; fully supporting "bring your own engine" is on the roadmap.
- **"lm-server" Channel gets new life**: Users noted that the *lm-server* channel was axed, suggesting the [self-hosting channel](https://discord.com/channels/1110598183144399058/1153759714082033735) is a better place to self-host and is the place for API/server problems internal to LMStudio.
   - The channel had too much overlap with another channel, so it was renamed and unarchived for the users to try out, and see if people found it useful.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1372435960054284349)** (128 messages🔥🔥): 

> `VRAM Importance vs DRAM, Qwen Models, KV Cache, 7900 XTX, 5060 Ti` 


- **VRAM Dominates for Model Speed**: It was mentioned that **DRAM speed barely matters if you keep your model in VRAM** and you should always keep your model within **VRAM** if you want reasonable speed.
- **Qwen3 Models Explored for VRAM Efficiency**: Members suggested trying **Qwen3 14b q4** within a **24GB VRAM** and mentioned that a **30b class model at q4 will take up around 20gb itself**.
   - Others suggested that using anything below q4 for models of this size is questionable or to try **Qwen3 30b moe q4** but partially offload to **DRAM** if the goal is only **10 t/s**.
- **KV Cache Location Impacts Performance**: One member used **CPU/RAM KV Cache offload**, noting that RAM speeds matter in this case and they plan to test if **vram-ram GPU shared memory** is better or not.
   - Another member noted they can get **20+ t/s full CPU at q4** with a **14900K** and **100GB/s DDR5**.
- **7900 XTX Tempts Over Nvidia**: One member bought a **7900XTX** card and is about to sell one of their **Nvidia** cards.
   - They mentioned being mildly annoyed with dual card driver instability issues with their **4080s** and **4060ti** cards and that the **5060 ti** is going back as a return.
- **5060 Ti 8GB Model Falls into No Man's Land**: It was discussed that the **8GB 5060** is not good for **AI** and also not cheap enough for those scraping the bottom of the barrel for gaming, so anyone eyeing it may as well get the **16GB** version.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1372473982049189960)** (155 messages🔥🔥): 

> `AlphaEvolve Analysis, LLM vs. System Role, Gemini 2.5 Pro, LiquidAI Skepticism, Hybrid AI Approaches` 


- ****AlphaEvolve: LLMs in Trench Coats?****: Members debate whether **AlphaEvolve** is *merely* an LLM-driven agentic loop or a more sophisticated system with an evolutionary algorithm, referencing [Google DeepMind's blog](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf).
   - Some argue its success is primarily due to powerful LLMs like **Gemini**, while others emphasize the crucial role of the evolutionary engine and verifier components, pointing to the ablation section in the paper.
- ****Gemini 2.5 Pro: Coding Ace?****: Members praise **Gemini 2.5 Pro's** coding abilities, especially its *zero-shot* performance, but one states that it had *sorta random refusals when it came out* when asked for *cyber ethics* homework.
   - There is speculation that fine-tuning with verified rewards and rejecting non-compiling outputs contributes to its coding prowess. This is also what helps with reasoning.
- ****LiquidAI: Hype or Hope?****: Skepticism surrounds **LiquidAI** and its *liquid foundational models*, with one member initially dismissing them.
   - After further investigation, another compared **LiquidAI** to **State Space Models (SSMs)** like Mamba, Gated DeltaNet, or RWKV 7, noting similarities but favoring the treatment of SSMs. Check out [LiquidAI's Research](https://www.liquid.ai/research/liquid-neural-networks-research).
- ****AI Hybridization: The Next Frontier?****: Discussions emerge about hybrid approaches combining neural (LLMs), symbolic (DreamCoder), evolutionary (novelty search), RL, and biologically-inspired architectures.
   - There is debate on whether scaling current approaches is sufficient or if paradigm-shifting hybrid models are necessary for achieving more advanced intelligence, or even Artificial General Intelligence.
- ****Absolute Zero Improves Models Without Data****: Members discussed the recent paper [Absolute Zero](https://link.to.absolutezero), which improves models without any data to begin with, they just make the LLMs generate it all and verify it's correct.
   - In this framework, the LLM is trained to perform three tasks: Y = F(X), Y = F(?), and Y = ?(X).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1372564629909143622)** (3 messages): 

> `Sakana AI, AI Scientist Paper, Language Models, Reasoning Mistakes, Error Correction` 


- **Sakana AI faces Skepticism**: Skepticism arose around **Sakana AI**, with concerns that their latest **AI Scientist paper** felt overly focused on marketing.
- **Math Language Models Can Learn From Errors**: Discussion was sparked around the paper, [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://ssrn.com/abstract=5250631), which explores improving reasoning accuracy by incorporating **error-correction** data directly into the pretraining stage.
- **Error Correction Boosts Reasoning**: The paper indicates that pretraining with error-correction data helps language models achieve higher reasoning accuracy through simple auto-regression, compared to pretraining on error-free data, as detailed in the [associated blog post](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-2) and [YouTube video](https://www.youtube.com/watch?v=yBgxxvQ76_E&list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&index=4).
- **Self-Correction via Multi-Round Prompting**: The research explores how pretrained language models can *self-correct* mistakes via **multi-round prompting**, focusing on the usefulness of incorporating error-correction data directly into the pretraining stage, as outlined in [arXiv:2505.09343](https://arxiv.org/abs/2505.09343).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1372435500824133682)** (19 messages🔥): 

> `Stable Audio Open Small, MythoMax-L2-13B Samsung Release, Meta researchers leaving` 


- ****Stability AI** and **ARM** drop **Stable Audio Open Small****: [Stability AI](https://stability.ai/news/stability-ai-and-arm-release-stable-audio-open-small-enabling-real-world-deployment-for-on-device-audio-control) and **ARM** released **Stable Audio Open Small**, enabling real-world deployment for on-device audio control.
- ****Samsung's** Accidental **MythoMax-L2-13B** Release**: **Samsung** seemingly accidentally released the **MythoMax-L2-13B** roleplay model, which was quickly removed after being spotted [on Hugging Face](https://huggingface.co/Samsung/MythoMax-L2-13B).
   - A member joked, *"Can someone do that for **OpenAI** and 'release' **GPT4Chan**? Or **Anthropic**, that would be priceless."
- ****Meta's** Brain Drain blamed for **LLama 4** fumbles**: A member expressed bafflement at **Meta's** struggles with **LLama 4**, despite their resources and history of success with **LLama** models.
   - Another member suggested that the departure of original researchers and research leadership failure may be responsible, while also suggesting that [Thinking in Latent Space](https://www.youtube.com/watch?v=qhYQ20TbtJ8) is in a different org than GenAI.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1372711951637024788)** (1 messages): 

> `Chatroom shortcut, Model Icons, Quick Chat` 


- **Chatroom Shortcut emerges**: Users can now click on **model icons** in the grid to initiate a **quick chat** with a specific model.
   - This bypasses the need to open the entire group and manually remove the rest, streamlining the user experience as shown in the [attached image](https://cdn.discordapp.com/attachments/1092729520181739581/1372711951309738014/image.png?ex=6827c520&is=682673a0&hm=2548e2d91f09d872bd9bf58df2a1effa46101b1fef3b32a136cc594fa203e7c0).
- **Streamline user experience**: The new feature allows users to bypass the need to open the entire group.
   - Users can start quick chats with individual models by simply clicking on the model icons, greatly improving efficiency.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1372436355874951178)** (105 messages🔥🔥): 

> `DeepSeek v3 MoE, Corvids cat food and bird food, Proxy for OpenAI, AlphaEvolve, Qwen3 /no_think bug` 


- ****DeepSeek v3 is a MoE Model****: It was stated that [DeepSeek v3](https://deepseek.com/blog/deepseek-v3) is a **Mixture of Experts (MoE)** model, meaning it activates only a subset of its parameters during inference.
   - Even though all the parameters are loaded into VRAM, only the parameters relevant to the prompt are computed, making the inference speed much faster.
- ****Corvids Only Eat Peanuts and Cat Food****: Users observed that **corvids** (crows and magpies) are only eating **peanuts and cat food** instead of normal bird food.
   - It was suggested that urban corvids have adapted to a diet closer to trash and prefer alternatives to standard bird food, as their diet has changed over many generations.
- ****Bypass Country Restriction with a Proxy****: A user shared an error message from OpenAI indicating that their country, region, or territory is not supported.
   - Another user suggested using a **proxy** to circumvent the geographic restrictions causing the error.
- ****`Qwen3` needs toggling to THINK****: For `qwen3`, it needs to be forced to think with `/think` or `/no_think` to toggle on and off thinking.
   - It was reported that `/no_think` functionality had a bug and OR needs to auto route away.
- ****Gemini 2.5 Pro Reasoning Chunks Deemed Useless****: A user reported that **Gemini 2.5 Pro's reasoning chunks** are useless, stating it only indicates the user's query and confirms the work being done towards it.
   - They mentioned it just presents summaries such as *'The user is asking for X. I have done some work towards X'*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1372427795510067210)** (67 messages🔥🔥): 

> `Fine Tuning Llama on SageMaker, LibreChat privacy concerns, GraphQL schema code completion, Strcoder2 model distillation for Python, Emotion classification model accuracy` 


- **SageMaker Llama Fine-Tuning Guidance Sought**: A member asked for guidance on how to fine-tune **Llama** using **SageMaker training** and requested relevant tutorials.
   - Another member provided links to [Hugging Face documentation](https://huggingface.co/docs/sagemaker/train) and a [related GitHub repository](https://github.com/yuhuiaws/finetuning-and-deploying-llama-on-Sagemaker).
- **LibreChat Privacy Questioned**: A member inquired about potential privacy concerns when using the officially hosted **LibreChat** at [librechat-librechat.hf.space/login](https://librechat-librechat.hf.space/login).
   - Another member suggested that typical website privacy concerns apply and pointed to the [LibreChat Docker image](https://github.com/danny-avila/LibreChat/pkgs/container/librechat-dev) as the base for the Hugging Face Space implementation.
- **Distill Starcoder2, Only Python**: A member asked how to reduce the **starcoder2** model size to focus solely on **Python** knowledge, effectively distilling the model.
   - A member suggested that extracting specific language knowledge would be difficult and recommended searching for smaller, specialized models on the [BigCode Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) or [The Big Benchmarks Collection](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a).
- **DistilRoberta's Emotion Accuracy**: A member was confused about the accuracy of emotion classification using the **DistilRoberta** model, given its popularity and high download numbers.
   - They questioned whether truncating long paragraphs to the model's maximum length of 512 tokens would affect analysis and whether sentence-level analysis would be more appropriate.
- **New Samsung Models Emerge**: A member noted the release of new models from **Samsung** and shared links to the **MuTokenZero2-32B** model and **MythoMax-L2-13B** model.
   - Another member indicated that the models were under construction.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1372432527649800202)** (2 messages): 

> `LangGraph` 


- **Crafting LangGraphs for Agentic Workflows**: Members shared a helpful link to the **LangGraph** documentation ([LangGraph Course](https://huggingface.co/learn/agents-course/unit2/langgraph/first_graph)), highlighting its use in building agentic workflows.
   - LangGraph is useful for managing **complex conversational flows** and **multi-agent systems**.
- **LangGraph powers Conversational Flows**: **LangGraph** excels at overseeing intricate dialogue paths and **multi-agent setups**.
   - It facilitates the structured management of interactions, allowing for more robust and adaptable AI agent behaviors.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1372566805809463411)** (6 messages): 

> `Realistic Text-To-Speech, WebGPU Vision-LLM, AsianMOM, SmolVLM, Federated Learning AI` 


- **Realistic Text-To-Speech Generator Makes Waves!**: A member shared a link to a **realistic text-to-speech generator**, claiming it's almost as good as **Dia 1.6B** but free and unlimited: [Hugging Face Space](https://huggingface.co/spaces/NihalGazi/Text-To-Speech-Unlimited).
   - The tool supports other languages, but the results may not be as good as English.
- **AsianMOM Roasts You In-Browser!**: A member introduced **AsianMOM**, a **WebGPU Vision-LLM app** that roasts you like your mom in-browser using **SmolVLM (500M)** and **LLama 3.2 (1B)**, and available on [asianmom.kuber.studio](https://asianmom.kuber.studio/).
   - It might be a little bit slow on first try (takes about 3 mins) when it installs models, but it caches it so it's way faster the second time.
- **Delving into Democratized AI Access**: The creator of **AsianMOM** expressed that this funny little project genuinely taught them so much about **WebML** and **Vision models**, noting that *the technologies we're getting with WebML will 100% democratize AI access*.
   - They shared a [GitHub repo](https://github.com/Kuberwastaken/AsianMOM) for those interested.
- **Federated Learning is all the Rage**: A member shared a link to a **LinkedIn** post about **Rag Federated Learning AI** - [linkedin.com](https://www.linkedin.com/posts/nerdai_rag-federatedlearning-ai-activity-7328477143791775744-IjCO?utm_source=share&utm_medium=member_desktop&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

cleonorris: It is monthly, but this is actually our last one before the summer!
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1372456933524504646)** (6 messages): 

> `DistilRoberta vs Roberta, Emotion Detection accuracy, GLoVE Paper, RobertaForTokenClassification extension, BERTopic` 


- **DistilRoberta's Popularity Questioned**: A member questioned why the **DistilRoberta** version of a model has more downloads than **Roberta**, wondering if it's better for emotion detection despite potential accuracy differences.
   - Another member explained that **DistilRoberta** is a lighter version of **Roberta**, trained to balance computational cost and accuracy, but theoretically has lower accuracy due to fewer weights.
- **Model Truncation Troubles**: A user inquired about text truncation in HF models, confirming that large paragraphs are truncated to the model's max length (e.g., **512 tokens**), and asked if looping and averaging scores would be necessary.
   - The user was concerned that only a small portion of the paragraph would be analyzed if truncated.
- **GLoVE's vector differences debated**: A member sought clarification on a line from the **GLoVE** paper: *'Since vector spaces are inherently linear structures, the most natural way to do this is with vector differences'*.
   - The member questioned the logic behind this conclusion, wondering if it was based on heuristics rather than a concrete rationale.
- **Training Data Issues plague Emotion Detection Models**: A member shared the [arXiv link](https://arxiv.org/abs/2310.12318) noting that the training data relies on crowdsourced annotations, creating learning pattern artifacts.
   - For example, the word *'hungry'* might incorrectly trigger an anger response, regardless of context.
- **BERTopic Suggested for Topic Extraction**: Instead of an emotion detection model, a member suggested using **BERTopic** for finding topics in text and linked to the [BERTopic documentation](https://maartengr.github.io/BERTopic/index.html).
   - The summarizer suggested BETTopic can be a *better solution* for extracting and finding topics in text.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1372497337561124945)** (4 messages): 

> `Qwen, AI Agent course` 


- **Qwen Usage Questioned**: A member asked another about their Qwen model usage, suggesting more advanced methods might be needed.
   - The user replied they are using **Qwen 3** with basic prompts and tools, and requested pointers to more advanced techniques.
- **AI Agent Course**: A member mentioned starting an **AI Agent course** and encountering an error while building their first agent using the course template.
   - The member is seeking assistance in resolving the error they're encountering.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1372464267143549010)** (10 messages🔥): 

> `Agent Template Errors, Course Completion, Final Unit Library, Certification Deadline` 


- **First Agent Template triggers errors**: A member reported that the **First_agent_template** worked initially but now consistently throws errors.
   - Another member suggested checking if they'd run out of credits, while noting *this space for Unit 3 has error and needs to be fixed* [Unit 3 Agentic RAG](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG).
- **Course Completion without pay?**: A member asked *is there any way to complete this course without paying*. 
   - One member suggested running the code locally.
- **Token Limit Troubles for Final Project**: A member ran out of tokens while working on the final project, rendering their API key unusable.
   - They were advised to run it locally, however, they then asked how they could proceed with the submission if ran locally.
- **Certification process is on a deadline**: A member questioned the need for a deadline for the certification process.
   - Another member explained that *Because the program will get outdated*.
- **Query on the Final Unit Library**: A member inquired about the library used for the final unit and the reasoning behind its selection.
   - Other members responded by encouraging them to run the code locally.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1372488314246004758)** (16 messages🔥): 

> `Audiobook format, Ducky Bedtime Stories, Focus at Work, Pomodoro Timer, YouTube Music integration` 


- **Audiobook Format Discussion Erupts**: Members discussed the format for audiobooks, with one suggesting *dictating/performing source material in full from start to finish as written reading explicitly without skipping or modifying*.
   - They emphasized waiting until the end of each part for discussion/reflection and suggested finishing the part even if running out of time, rather than rushing or skipping material.
- **"Ducky Bedtime Stories" Audiobook**: A member created an audiobook of **ducky bedtime stories**, reading with appropriate energy and enthusiasm, using different voices for each character.
   - The expert speaker is a duck and can only say **"QUACK!"**, with the volume decreasing as the audiobook progresses.
- **Notebook LM aids Deep Focus at Work**: A user discovered that running Google's Notebook LM podcast of a chosen book with YouTube music on a low volume helped them enter **deep focus at work**.
   - The user recommends a **loop button** on the podcast option and an integration with **YouTube Music** for a richer experience.
- **Acid-Dipping Duck Enters the Chat**: A user posted the message *dip your genitals in acid* followed by a duck emoji.
   - This was followed by an attachment called **The_Duck_Dive_-_History.wav** which we could not analyze.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1372426744690245634)** (76 messages🔥🔥): 

> `App Availability in Pakistan, Mobile App Program Limitations, Audio Generation Limits, Tabular Data with NLM, Scammy Links Warning` 


- ****Pakistan Access Predicament****: A user inquired about the app's lack of availability in **Pakistan**, and another suggested using a **VPN** to download it.
   - Another user, an **Android app tester**, pointed out issues with voice review personalization within the studio.
- ****Mobile App's Monetary Model****: A user questioned the mobile app's daily audio overview limitations without a **$20/month** subscription, leading to a discussion about paywalls.
   - Another user lamented the **3 audios/day** generation limit with the complaint that *Google is so greedy*, to which another user responded by calling that sentiment greedy, as it is a free product.
- ****Podcast Plan Pays Off****: A user hit their **100-podcast max** on NotebookLM and intends to download the **WAV** files, convert them to video podcasts, and upload them to **YouTube** or **Google Photos**.
   - Another user replied that *this is smart*, with another user replying *I do something similar*.
- ****Table Troubles Torment Tooling****: A user inquired about supplying tabular data to NLM, but was advised by another that **NLM** isn't ideal for tabular data.
   - The user was advised to consider using **SQL** or the **AI formula** in **Google Sheets**, or alternatively **Gemini with BigQuery**.
- ****Link Lurkers Launch Looming Lies****: Users were cautioned about **scammy links** promising *free gifts*, *easy cash*, or *amazing deals*, and were advised to think twice before clicking such suspicious links.
   - It was emphasized that links offering freebies are major red flags, and users should always protect their personal information.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1372694625218854932)** (1 messages): 

> `Solana Foundation, Decentralized AI, Psyche` 


- **Nous Research and Solana Foundation Host Event in NYC**: Nous Research is co-hosting an event with the **Solana Foundation** in NYC on **May 22**, focusing on **Decentralized AI**.
   - The event will cover Nous’s efforts to democratize intelligence, including **Psyche**; registration is available [here](https://lu.ma/39b7e9pu).
- **Psyche Democratizes Intelligence**: **Psyche**, a project by Nous, aims to democratize intelligence through decentralized AI efforts.
   - The project's goals and progress will be discussed at the upcoming event with the Solana Foundation.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1372466405898715197)** (85 messages🔥🔥): 

> `Psyche Training, Meta AR, Smart Glasses, Grok crashing` 


- ****Psyche** Training Speeds into Hyperdrive**: The training rate for **Psyche** is currently at **12B tokens per day**, with estimates suggesting it would take almost **2000 days** to process the entire **20T tokens**, spurring a call for more GPUs.
   - Contributors can support model training through donations to the mining pool on the [Psyche Network](https://psyche.network/) or by contributing to the codebase on [GitHub](https://github.com/PsycheFoundation/psyche).
- ****Meta** Navigates **AR** and **AI** Intersection**: **Meta** faces the challenge of integrating **AI** into its smart glasses, potentially rendering its **AR** investments obsolete if it fails to adapt.
   - Despite the shift, they continue **AR** research with projects like [Project Aria](https://www.projectaria.com/), balancing current limitations in **AI** functionality with ongoing progress.
- ****Smart Glasses** Need **AI****: Members suggest that **smart glasses** require *real agentic AI* to effectively interpret and interact with the user's environment.
   - A demo from [Sesame](https://app.sesame.com/) shows how a smart glass company is innovating toward useful agentic AI, prompting call for *an open smart glass AI*.
- ****Grok** Glitches in South Africa: **AI** Prompt Troubles?**: Discussion arose whether **Grok's** issues in South Africa stemmed from tweaked steering vectors or *clumsy prompt updates*.
   - One member stated *Absolutely no basis to back it up but I am voting clumsy prompt*.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1372658113781432450)** (1 messages): 

> `TritonBench, AMD GPU Errors, Memory Access Fault` 


- **TritonBench Throws Memory Access Fault on AMD GPU**: A member is working on a [TritonBench benchmark](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) and found that about **7 kernels throw memory access fault errors** when run on an AMD GPU.
   - One example provided was [chunk_gla_fwd.py](https://github.com/thunlp/TritonBench/blob/main/data/TritonBench_G_v1/chunk_gla_fwd.py) which throws an `Unknown Reason` error, and the member requested assistance to pinpoint the cause.
- **Seeking Help with AMD GPU Memory Access Faults**: The user encountered memory access fault errors while running Triton kernels on an AMD GPU.
   - Specifically, the user seeks assistance in identifying the root cause of the memory access violation, suspecting it's related to accessing memory locations outside of bounds in the provided code.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1372433414048977027)** (3 messages): 

> `cudaIpcMemHandle_t Serialization, Single GPU Multiprocess Communication` 


- **CUDA IPC Memory Handles Made Simple**: A member explored using `cudaIpcGetMemHandle()` for single GPU multiprocess communication, for scenarios where PyTorch dataloaders are not viable.
   - They noted that `cudaIpcMemHandle_t` can be string-serialized, enabling a straightforward producer-consumer setup for sharing memory handles.
- **Serialization Simplifies GPU Data Sharing**: The user discovered `cudaIpcMemHandle_t` can be string-serialized.
   - This allows for a simple producer-consumer design to share those handles between processes on a single GPU, sidestepping more complex inter-process communication methods.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1372573625429004321)** (12 messages🔥): 

> `Mapping Fused Operations, Pipeline Parallelism with torch.autograd.graph.saved_tensors_hooks, Custom CUDA Graphs and Caching Issues in vLLM V1, GEMM Codegen Performance vs. Native aten Implementation, torch.compile modes benchmark` 


- **Tracing Fused Ops back to Source Code**: A member inquired about mapping fused operations back to their original model code after compiler fusion, seeking to identify precisely which operations were fused but another member replied with a [link to docs](https://docs.pytorch.org/docs/main/torch.compiler_inductor_provenance.html) regarding `inductor_provenance_tracking_node_mappings_<number>.json` files.
   - The member was unsure how to easily map the exported program's graph to the original model code without careful reading.
- **Experimenting Pipeline Parallelism with CUDA Streams**: A member experimented with pipeline parallelism using `torch.autograd.graph.saved_tensors_hooks` to manage activations across separate CUDA streams, aiming for concurrent forward and backward passes, referencing the [docs](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks).
   - Despite successful implementation without race conditions, the member observed minimal concurrency gains due to the model's kernel occupancy, deeming it a *"fun experiment though!!"*
- **Custom Op Caching Glitch in vLLM V1**: A member encountered a cloning error related to a cache in vLLM V1 (which uses torch.compile + custom CUDA graphs) involving a custom op `f1()` calling `f2()` that samples from a cache, which then threw `RuntimeError`.
   - The error stated that *"the output of this custom operator (1) must not also be an input to this custom operator and (2) may not alias any inputs to this custom operator or other returns"*, and the member asked for a bypass without cloning.
- **GEMM Codegen Slower Than Native aten?**: A member benchmarked different `torch.compile` modes with a GEMM operation, comparing `f_compile`, `f_overhead`, and `f_max` against a non-compiled version, using input sizes of `N = 2_000` and `B = 100_000`.
   - The results indicated that `f_compile` (fullgraph=True) was slightly faster at **10.43 ms** than the others (**12.06 ms**, **12.09 ms**, and **12.56 ms**), which suggested that device reduce is the bottleneck.
- **Nvidia-smi Shows Memory**: A member inquired whether `nvidia-smi` shows allocated or reserved memory, concerned about high reserved memory relative to VRAM capacity, while uploading an [image of nvidia-smi](https://cdn.discordapp.com/attachments/1189607750876008468/1372651600891613255/9NMqqTPCSzgAAAABJRU5ErkJggg.png?ex=68278cec&is=68263b6c&hm=3b3913414e790760f018d6eab9b2d24854f387c59c354f76c4e57a9f2dbdbcb1).
   - Another member clarified that `nvidia-smi` has no insight into torch internals and that reserved memory cannot exceed VRAM.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

real.optimus.prime: From DeepSeek: 
https://arxiv.org/abs/2505.09343
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1372494117665505342)** (6 messages): 

> `CUDA SASS negation, CCCL/libcu++ vector types, CUDA compilation flags` 


- ****CUDA SASS Negation** is free?**: According to members, when CUDA code compiles down to SASS looking like `FLO.U32 R4, R4 IADD32I R4, -R4, 0x1f`, the negation of register R4 happens **without any overhead**.
- ****CCCL/libcu++** cooks vector types**: **CCCL/libcu++** is implementing the **tuple protocol** for vector types, which should work well with unrolling templates.
   - However, there is uncertainty if it will include types not available in CUDA, like `char16`, and whether it will still include the `.x` naming convention.
- **Extra flag to enable CUDA compilation?**: A member asked about needing *an extra flag in compilation*, linking to [NVIDIA's blog on CUDA 7 streams simplifying concurrency](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/).


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1372447432209596497)** (1 messages): 

> `GitHub Repository, Lecture Scripts` 


- **Lecture Viewer Seeks GitHub Repo**: A viewer of a recent lecture inquired about the availability of the associated **GitHub repository** and other resources mentioned.
   - They requested the link to the repository, noting they could not find the scripts despite checking available resources.
- **Lecture Resources Inquiry**: A user who watched a lecture is seeking the **GitHub repository link** for the associated scripts.
   - The user mentioned that they checked the **GitHub repository** and other resources mentioned in the lecture but couldn’t locate the scripts.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1372579921188360234)** (3 messages): 

> `Tensor Processing Unit (TPU)` 


- **TPU Definition Delivered**: A member inquired what a **TPU** is and another member responded that it is a *tensor processing unit*, basically an accelerator optimized for **AI workloads**.
- **TPU vs CPU**: A TPU is designed to accelerate machine learning workloads better than a CPU.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1372424337805545525)** (40 messages🔥): 

> `MI300, amd-fp8-mm, amd-mixture-of-experts` 


- **MI300 benchmarks see fresh submissions**: Several users submitted new benchmarks on the **MI300** across different leaderboards, including `amd-fp8-mm` and `amd-mixture-of-experts`.
   - One user achieved a personal best of **6209 ms** on the `amd-mixture-of-experts` leaderboard.
- **amd-fp8-mm Leaderboard Heats Up**: Multiple successful submissions were recorded on the `amd-fp8-mm` leaderboard, with times ranging from **155 µs** to **3.28 ms** on the **MI300**.
   - A user also logged a personal best of **2.42 ms** on the **MI300**.
- **amd-mixture-of-experts sees New Bests**: The `amd-mixture-of-experts` leaderboard saw frequent entries, with multiple users achieving personal bests, such as **6233 ms** and **6247 ms** on the **MI300**.
   - One user achieved **32.7 ms** on the **MI300**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1372588989785182209)** (3 messages): 

> `Cutlass, fp8, bf16, narrow precision dtypes` 


- **Cutlass Thanks Given**: A user thanked the **Cutlass** team for their work and excitement to start hacking with it.
   - The user then questioned the 'Notable unsupported features section' and what **dtypes** are not currently supported, such as **fp8** and **bf16**.
- **Cutlass fp8 is supported**: **Fp8** is supported in the latest Cutlass.
   - The team clarified that *narrow* dtypes mean *sub byte and micro scale types* and are coming soon.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/)** (1 messages): 

clattner: Y'all might find this techtalk interesting: https://www.youtube.com/watch?v=Invd_dxC2RU
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1372424394860662795)** (50 messages🔥): 

> `OpenMemory MCP, Grok issues, Microsoft fires TypeScript dude, Agentic tooling, FUNapis` 


- **OpenMemory MCP Opens Memory**: A member shared a link to [OpenMemory MCP](https://mem0.ai/blog/introducing-openmemory-mcp/), calling it cool.
   - This is a new open-source project that aims to provide a unified memory management layer for AI applications.
- **Ongoing Grok Troubles Tracked**: Ongoing issues with **Grok** are being tracked in [this Discord channel](https://discord.com/channels/822583790773862470/1036726703730466896/1372435415490887770).
- **Microsoft Axes TypeScript Talent**: Discussion arose around Microsoft firing the **TypeScript** dude without warning, prompting expressions of dismay, as seen in [this tweet](https://x.com/brodyford_/status/1922726909365879039?s=46).
- **Agentic Tooling Outpaces Big Tech**: Members discussed a shift happening with **agentic tooling**, expressing hope that indie devs can use it to outpace big tech and corporations.
   - It was suggested that *getting the computers to do the right thing well* is still a harder problem to solve than *do the wrong thing well*, especially given internal incentive structures in corporations.
- **FUNapis Succumbs to Bing's Chatbot**: A member suggested that **FUNapis** died so **Bing** can sell their chatbot wrapper of the API, as seen in [this tweet](https://x.com/williambryk/status/1923062696095711374?s=46).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1372509787350106214)** (39 messages🔥): 

> `Knowledge graphs for papers, Cloud GPU/HW providers, MLPerf training benchmark, Data stalls in DNN training, Audio modality preprocessing` 


- ****Connected Papers** for Knowledge Graphs**: A member shared a link to [Connected Papers](https://www.connectedpapers.com/) as a resource for creating **knowledge graphs for research papers**.
   - The tool helps visualize and explore connections between different papers in a given field, useful for literature reviews and research.
- **Seek Cloud GPU Guidance for Open Source Dev**: A member is seeking recommendations for **cloud GPU/HW providers** suitable for open source development, specifically for setting up an **MLPerf training benchmark**.
   - They are particularly interested in options that might offer free compute hours to students or have favorable conditions for open source projects.
- ****Data Stalls Plague DNN Training****: Discussion revolved around whether **CPU-based preprocessing** is a bottleneck in **DNN training pipelines**, particularly in the context of audio modality.
   - One member referenced a paper on the topic ([Lotus: Characterization of Machine Learning Preprocessing Pipelines via Framework and Hardware Profiling](https://www.computer.org/csdl/proceedings-article/iiswc/2024/560300a030/22f0GQCjZGo)) and others questioned whether optimizing CPU workload is beneficial if the GPU becomes the bottleneck.
- ****Optimize Data Loading, Preprocessing** to Avoid Future Annoyance**: A member shared that they are optimizing their data loading and preprocessing pipeline to avoid being bottlenecked by resources due to bad tooling as they work by themselves outside of academia or industry.
   - They hope that this work will benefit all future audio data + mech interp work at scale.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1372445144338534501)** (3 messages): 

> `BlackboxNLP, Interpretability, Causal Variable Localization, MIB Benchmark` 


- ****BlackboxNLP** heads to EMNLP 2025**: The 8th edition of **BlackboxNLP**, the leading workshop on interpretability and analysis of neural networks for NLP, will be co-located with [EMNLP 2025 in Suzhou](https://blackboxnlp.github.io) this November.
   - They will feature a [new shared task](https://blackboxnlp.github.io/2025/task) on **circuits/causal variable localization in LMs** using the recently released [MIB Benchmark](https://mib-bench.github.io/) with a submission deadline of August 1st.
- ****MIB Benchmark** for causal variable localization**: A new shared task using the recently released [MIB Benchmark](https://mib-bench.github.io/) will focus on **circuits/causal variable localization in LMs**.
   - The submission deadline for this task is **August 1st**, offering a focused challenge within the BlackboxNLP workshop.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1372481613845237781)** (2 messages): 

> `MCQ Evaluations, MMLU Issues, Model Outputs` 


- **MCQ Evaluation Outputs Flagged as False**: A member reported an issue with **MCQ evaluations** like **MMLU** where model outputs are consistently flagged as *false*, even when the model assigns the highest probability to a specific option based on **NLL values**.
   - This issue is observed with smaller models, where all four options are marked as *false*.
- **Smaller Models Displaying All-False Outputs**: The problem of all *false* outputs in **MCQ evaluations** appears to be more prevalent with smaller models.
   - This suggests a potential bias or limitation in how these models handle multiple-choice questions, particularly when assessing probabilities via **NLL values**.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1372644184837718076)** (2 messages): 

> `LLaVAGuard, SafeDiffuser, Multimodal Models` 


- **Multimodal Model Papers Sought**: A member requested pointers to recent, notable papers on **multimodal models** like [LLaVAGuard](https://github.com/LAION-AI/LLaVAGuard) or [SafeDiffuser](https://github.com/safeai-lab/safe-diffuser).
- **Clarification on Channel Appropriateness**: The member inquired whether their question would be more suitable for the *image-models* channel.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1372454914252607509)** (33 messages🔥): 

> `MCP Client-Server Call Flow, Chainlit Query Parameters, Jinko MCP for Hotel Sales, Smithery Server and Claude Desktop, Understanding MCP Resources` 


- **MCP Client-Server Call Flow Request for Emulation**: A member inquired about the call flow between an **MCP client** and **MCP server**, seeking to emulate a server, detailing the initial steps such as *client -> method: initialize* and *server -> method initialized*.
   - The member sought advice on intermediate steps and pointed to the need to understand how to implement an MCP server to successfully emulate the protocol.
- **Chainlit Query Parameter Quest**: A member is facing challenges accessing **query parameters** from the URL within **Chainlit** despite attempting solutions like middleware in FastAPI.
   - They tried passing tokens and decoded dictionaries but couldn't access them within Chainlit, and asked for solutions to access query parameters within Chainlit.
- **Jinko MCP Sells Hotels**: A member announced the creation of an **MCP** for developers to build **AI agents** that want to sell hotels.
   - This MCP provides access to **2M+ hotels**, enabling search, booking, payment, and customer service, linking to the [Jinko MCP GitHub repository](https://github.com/jinkoso/jinko-mcp/blob/master/README.md) for more details.
- **Smithery Server Struggles with Claude**: A member sought guidance on using a **Smithery-installed server** with **Claude Desktop**, after adding their **OpenRouter key**.
   - The member questioned whether the model used in the MCP tool configuration needs to match the one used in Claude (e.g., **sonnet-3.5** in MCP config vs. **sonnet 3.7** in Claude).
- **Resources clarified as GET Requests**: A member sought help explaining what a **resource** is in the context of **MCP**, noting confusion among workshop attendees, and attempting to clarify it as *"the GET request of MCP"*.
   - After suggesting dragging a file over to the Cursor chat helps in understanding the same, resources were explained to be objects with URIs like datetime://Euroupe/London/now and <http://example.com/llms.txt>.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1372554582567157811)** (8 messages🔥): 

> `LLM Agent to MCP Server Connection, MCP for AI Agents Selling Hotels, MCP Democratizes Apache Kafka Usage, macos-automator-mcp for Autonomous Debugging, AI and MCP Language Barriers` 


- ****LLM Agent** hooks up with **MCP Server****: A member shared a blog post on *How to connect your LLM Agent to MCP Server*, available [here](https://sandipanhaldar.com/blog/part-1-how-to-connect-your-llm-agent-to-mcp-server.html).
- ****AI Agents** now book **Hotels** via **MCP****: A member announced an **MCP** for developers building **AI agents** for selling hotels, providing access to *2M+ hotels* with search, booking, payment, and customer service, with the [GitHub repo](https://github.com/jinkoso/jinko-mcp/blob/master/README.md) linked.
- ****Kafka** democratized by **MCP**!**: A member discussed how **MCP** democratizes the usage of **Apache Kafka**, allowing interaction with real-time data via natural language prompts, with a [YouTube video](https://www.youtube.com/watch?v=ivlzvZzFeZMS) included.
- ****macos-automator-mcp** debuts **Autonomous Debugging**!**: A member introduced **macos-automator-mcp**, enabling tools like Cursor to control system functions for fully autonomous debugging, with a [GitHub link](https://github.com/steipete/macos-automator-mcp) provided.
- ****Shortwave** launches **MCP Client support**!**: A member announced the launch of **MCP client support** in **Shortwave**, supporting both **HTTP MCP** & **stdio MCP**, with one-click toggles for integrations like **Hubspot**, **Notion**, **Zapier**, **Asana**, and **Linear**, as described in their [blog post](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/) and [docs](https://www.shortwave.com/docs/how-tos/using-mcp/).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1372570639973482598)** (5 messages): 

> `Documentation updates, stdlib modifications` 


- **Doc fixed by PR**: A member reported seeing [Issue 4482](https://github.com/modular/modular/issues/4482) and was looking to update the documentation.
   - Another member stated that they fixed the doc in [PR 4530](https://github.com/modular/modular/pull/4530/files).
- **Docs generated from stdlib itself**: A member inquired where to modify the stdlib docs, assuming the location to be [modular/modular/tree/main/mojo/docs/manual](https://github.com/modular/modular/tree/main/mojo/docs/manual).
   - Another member clarified that the doc is generated from stdlib itself, so it can be modified directly.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1372635413272858645)** (9 messages🔥): 

> `Pointer declaration in Mojo structs, Mojo generics over origin, Mojo Lifetimes, Karpathy's micrograd porting to Mojo, Jeff's talks at Modular` 


- **Pointer Declaration Headaches in Mojo structs**: A member sought help with declaring a **Pointer** in a **Mojo struct**, referencing an example from [kapa.ai](https://kapa.ai) and later clarifying that the *origin* is the struct where the **Pointer** will be stored.
   - Another member suggested making the `Op` generic over origin if they want to borrow.
- **Origins and Mojo's Borrow Checker Unveiled**: A member explained that **Mojo** requires the *origin* to be part of the type, making it a parameter.
   - Another member clarified that the *origin* is tied to the **borrow checker**, ensuring the pointer isn't active if the data it points to is moved or freed, linking to the official [Mojo Lifetimes Documentation](https://docs.modular.com/mojo/manual/values/lifetimes/).
- **Micrograd Porting to Mojo**: A member is learning **Mojo** by porting **Karpathy's micrograd**, a simple Python example, working around the current lack of lambda function support in **Mojo**.
   - Another member shared their similar project, [momograd](https://github.com/dorjeduck/momograd), created last year as one of their first **Mojo** learning projects, though not updated to the latest **Mojo** versions.
- **Jeff's Talks at Modular Wow Members**: A member expressed enthusiasm for past talks featuring Jeff at **Modular**.
   - They shared [this YouTube playlist](https://www.youtube.com/playlist?list=PLh0S94-sJw_6ygGMynvQkt32IwBJM4DBW) and were happy to see such content again.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1372601560177311804)** (11 messages🔥): 

> `MAX Installation Issues, LoRA Trainer Difficulties, Mojo Weak Tensor Support, MAX and PyTorch Hybrid Approach, LLM Hallucinations with Modular's Platform` 


- **MAX Installation Plagued by Missing Functionality**: A member encountered errors during MAX installation, indicating missing essential functionalities like **tensor operations** (`tensor`, `nn`, `zeros`, `ones`, `matmul`, `add`, `mul`).
   - Despite attempting local installation, the required functionalities remained absent, preventing the continuation of a MAX-only implementation.
- **LoRA Trainer Faces Tensor Support Shortcomings**: The user reported difficulties in creating a diffusion **LoRA trainer** due to weak tensor support in Mojo and MAX, leading them to abandon the initial Mojo-only approach.
   - While a PyTorch version works, the goal was to avoid PyTorch, highlighting the limitations of MAX in tensor operations.
- **Hybrid MAX and PyTorch as Workaround**: Claude AI suggested that implementing a **MAX-only LoRA trainer** is not currently feasible due to missing tensor operations.
   - Instead, a *hybrid approach* using **PyTorch** and **MAX's interoperability features** was recommended as a more viable solution for immediate implementation.
- **LLMs Hallucinate Without Proper Context**: A member suggested ensuring that tools like Claude, Cursor, and others have access to the current [Modular GitHub repository](https://github.com/modular/modular) and [documentation](https://docs.modular.com).
   - Without proper context, these tools may **hallucinate** and generate incorrect suggestions due to Mojo and MAX being relatively new and not well-represented in LLM training data.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1372581567376916554)** (3 messages): 

> `Cohere, Sponsorship, Grants, Nonprofit` 


- **Cohere Staff Reaches out for Sponsorship and Grants**: A member requested contact with the right **Cohere** staff for supporting/sponsoring events and/or **grants** for their tech-adjacent nonprofit.
   - A staff member responded by offering their email address, [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com), to forward the inquiry to the appropriate person.
- **Tech Nonprofit Seeks Partnership with Cohere**: A tech-adjacent **nonprofit** is looking to partner with **Cohere** for event sponsorships and grants.
   - Contact [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com) to connect with the right staff.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1372614679418900490)** (6 messages): 

> `Cohere Classify API, Rate Limit Increase` 


- **Cohere Classify API Impresses Users**: Users expressed satisfaction with the **Cohere Classify API** and are interested in scaling its usage to millions of entries.
   - However, they are concerned about the current rate limits and are looking for ways to expedite the process.
- **Scaling Cohere API with Rate Limit Boosts**: A member suggested contacting **support@cohere.com** to request a rate limit increase for the **Cohere Classify API**.
   - This approach should help determine the feasibility of running the API at scale without extended waiting times.


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1372456961899106415)** (3 messages): 

> `SiliconFlow, Gemma 3 4b 4bit, Llama 3.2 3B 4bit` 


- **SiliconFlow Endpoints Modified**: A user is utilizing **SiliconFlow** and modified the endpoint to be **localhost**, as shown in the [attached image](https://cdn.discordapp.com/attachments/1218409701339828245/1372456961928331335/image.png?ex=68278066&is=68262ee6&hm=93a39c60bf386774de78fb37f4469ba201e375e9e173d6e6427db1a09ab94a1a&).
- **Gemma 3 4b 4bit Screenshot Shared**: A user posted a screenshot of **Gemma 3 4b 4bit**, as seen in the [attached image](https://cdn.discordapp.com/attachments/1218409701339828245/1372462317081591818/image.png?ex=68278563&is=682633e3&hm=fe00ba43ec32dd40fe99c2ec904aba7ae3f078b9df968075cbb55291717d751b&).
- **Llama 3.2 3B 4bit Screenshot Shared**: A user posted a screenshot of **Llama 3.2 3B 4bit**, as seen in the [attached image](https://cdn.discordapp.com/attachments/1218409701339828245/1372537263271051304/image.png?ex=6827cb30&is=682679b0&hm=f84185fbfe33d5e9af456ef71aaf4ff03339eabf81cae93d0be9a1901c188f69&).


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1372656865233604778)** (3 messages): 

> `Web AI Engineer introduction, Full stack developer AI fan Introduction, Gabriel 20 years development experience` 


- **Web AI Engineer Joins**: A **Web, AI Engineer** with **7 years** of fullstack experience introduced themself.
   - They are proficient in building responsive, user-friendly web and mobile applications using modern web technologies like **React(Next), React Native(Expo), Flutter, Vue(Nuxt), Svelte, Astro** and tools like **Node/Express, Django, Nest.js, Go, Web3.js, Shopify, Wordpress, TailwindCSS, Shadcn, MUI, Docker, Kubernetes, AWS/GCP, LLM**.
- **Full stack developer loves AI**: A developer with over **20 years** of experience introduced themself.
   - The developer has focused mainly on full stack development, became a big fan of **AI** and loves building real-time applications with thoughtfully crafted **UI** and **UX**, preferring **Nuxt** running on **Cloudflare** and using tools like **RooCode** and **Windsurf**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1372528849669197854)** (7 messages): 

> `Gemini Models Response Schema, Structured Outputs in DSPy, Pydantic models` 


- **Gemini Models' Structured Outputs Coming to DSPy**: Members discussed if **Gemini Models'** response schema, similar to **OpenAI's** structured outputs, is implemented in **DSPy**, and another member confirmed that it is.
   - It was also confirmed that **DSPy** dynamically builds the response schema.
- **Pydantic Models enable Structured Outputs in DSPy**: A member asked how to implement **structured outputs** in **DSPy**, similar to **OpenAI** tools, including **nested outputs** or **JSON schema constraints**.
   - Another member replied to *just use signatures*, and pass **Pydantic models** or **Python TypedDicts** as output field types.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1372523175656820788)** (5 messages): 

> `GPT4All's demise, Nomic's future direction, Jan.ai and LM Studio as alternatives` 


- **GPT4All's pulse flatlines, future uncertain**: Members speculate whether **GPT4All** is discontinued due to lack of updates since February.
   - One member lamented the lack of communication regarding a new version release and wonders about Nomic's intentions.
- **Nomic pivot to pay-to-play platform?**: Members speculated that **Nomic** might shift its focus to a monetized platform.
   - The claim was *"gpt4all is over .. now earn money with nomic!"* - but it wasn't backed by additional sources or evidence.
- **Jan.ai and LM Studio step up as GPT4All alternatives**: Members mentioned **Jan.ai** and **LM Studio** as potential alternatives to **GPT4All**.
   - It was not stated why those were good alternatives or which features they had that might be beneficial.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1372645019734900865)** (2 messages): 

> `Event Driven Agent Workflows, Multi-Agent Docs Assistant, Tig AI Coding Agent` 


- **Event Driven Agents Assist Weaviate**: LlamaIndex released a new walkthrough on how to use **event-driven agent workflows** to build a **multi-agent Docs Assistant**.
   - This assistant writes webpages into **LlamaIndexDocs & WeaviateDocs collections** in Weaviate and uses an orchestrator to decide when to call the Weaviate QueryAgent for search, showcased in [this Tweet](https://twitter.com/llama_index/status/1923085124725506441).
- **Tig Coding Agent Debuts**: An open-source **(human in the loop) coding agent** called **Tig**, created by @rsrohan99 and built with LlamaIndex workflows, was highlighted.
   - As shown on [Twitter](https://twitter.com/llama_index/status/1923134285940441102), Tig can write, debug, and analyze code across multiple languages, execute shell commands, and search the web.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1372576791427944599)** (2 messages): 

> `PDF content extraction with LlamaIndex, Vibe Coding partnership opportunities` 


- **LlamaIndex Explores PDF Content Extraction**: A member is seeking advice on extracting content from a PDF using **LlamaParse** or **LlamaIndex**, specifically aiming to extract the Table of Contents and then isolate content and tables from a particular section based on a predefined name.
   - They're looking for guidance on setting up the instructions or pipeline to detect the section from the TOC, isolate the content, and structure the extracted tables properly, including the right parameters for use in no-code tools like **n8n**.
- **AI Startup Scouts for Vibe Coding Buddies**: An AI startup based in Korea is seeking passionate developers experienced in **Vibe Coding** to partner on real client projects.
   - The opportunity includes a fair revenue-sharing model and ongoing partnership, with a requirement for strong communication skills, **GitHub links**, **Vibe Coding project references**, and English/Korean communication skills.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1372471760405925990)** (4 messages): 

> `Custom Torchtune Network on vLLM, Unregistered Model with vLLM, Custom Model Implementation in vLLM` 


- **Custom Network Faces Implementation Issues on vLLM**: A member tried to implement a custom **Torchtune network** on **vLLM** following several tutorials, but encountered failures.
   - Another member inquired whether the model was registered with **vLLM** and suggested converting the checkpoint to the **HF format** for syncing.
- **Implementing Custom Models in vLLM**: A member confirmed using a custom model with a custom architecture, leading to implementation difficulties with **vLLM**.
   - Another member pointed to a [vLLM guide](https://docs.vllm.ai/en/latest/contributing/model/basic.html) on implementing custom models in **vLLM**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1372622471420973056)** (3 messages): 

> `Lambda Workshop, Nobel FutureTech Info Session, Agentic AI, Inference API` 


- **Lambda Workshop Teaches Agentic AI**: A [Lambda workshop](https://www.youtube.com/watch?v=VmjMIwwo9ag) will teach how to build agentic applications using **Lambda's Inference API**, optimizing agent performance, and deploying agents in production.
   - Participants can apply for **$100 serverless API credits** by Friday 5/16 via [this form](https://forms.gle/UtVhmPS3mitS8Vxu7).
- **Nobel FutureTech Fireside Chat**: A fireside chat co-hosted by [Nobel FutureTech Group](https://nobel-futuretech.com/index.html) and Berkeley RDI will give insights into the innovative ecosystem of the **Nobel FutureTech Genius Club**.
   - The session provides information on mentorship, funding, and collaboration opportunities, with a [livestream link available](https://www.youtube.com/watch?v=ft-2W00Rtg8).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1372504105439989800)** (1 messages): 

> `topk GPU, masked_select GPU, randperm_generator GPU, index_put_impl, index_tensor` 


- **Topk Bounty: GPU Edition**: A user inquired about the requirements for the 'move topk' bounty, given that **topk**, **masked_select**, and **randperm_generator** are already off the CPU.
   - The user suggested that the bounty might need revision, considering the presence of other functions like **_index_put_impl_** and **index_tensor** in the torch backend that still require attention.
- **Index Functions Still on CPU**: The user points out that **_index_put_impl_** and **index_tensor** are still running on the CPU.
   - They suggest these functions, along with others in the torch backend, could be targeted for GPU offloading.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1372623165058191433)** (1 messages): 

> `Agentic Enrichment, LLM Data Access, Featureform` 


- **Agentic Enrichment Webinar Announced**: A live webinar on **Agentic Enrichment** with **Simba Khadder**, Founder of Featureform, is scheduled for **Tuesday, May 27th at 8 AM PT** and will cover how to unlock data for AI agents using **MCP**; you can sign up [here](https://buff.ly/zeoH55Y).
   - The webinar will discuss the missing layer of infrastructure needed for AI agents to access real-time, internal business data, highlighting the limitations agents face due to data access rather than intelligence.
- **Unlock LLM potential through better Data Access**: The webinar will cover the need for better internal data access to unlock the full potential of AI agents, detailing the **three key components of agentic enrichment**: semantic catalog, low latency serving, and governance.
   - It will demonstrate how **Featureform** enables this data access, making agents more useful and powerful in production environments, with real-world examples of improved workflows in AI systems.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1372648280206479540)** (1 messages): 

> `SWE-1, Software Engineering Models, Flow Awareness Approach, Windsurf Tab Experience, Cascade Optimization` 


- **Windsurf Launches SWE-1 Family of Models**: Windsurf introduced the **SWE-1** family of software engineering models, including **SWE-1**, **SWE-1-lite**, and **SWE-1-mini**, detailed in a [blog post](https://windsurf.com/blog/windsurf-wave-9-swe-1) and [launch video](https://youtu.be/LhA9pVpwgdY).
- **SWE-1 Boasts Claude 3.5-Level Performance at Lower Cost**: The **SWE-1** model is advertised to have *high-reasoning*, *tool-capable*, and *Cascade-optimized* performance comparable to **Claude 3.5** but at a reduced cost.
   - The models are trained using a unique *"flow awareness" approach*, understanding the timeline between humans and AI across development surfaces.
- **SWE-1 Aims for Software Development Acceleration**: Windsurf aims to accelerate software development by **99%** with the new **SWE-1** models.
   - This is just the beginning - they're investing heavily to make **SWE models** that exceed all frontier model performance in software engineering.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1372662490118357283)** (1 messages): 

> `AI21 Labs, Maestro, AI Tinkerers Meetups` 


- **AI Tinkerers host AI21 meetups**: AI Tinkerers is hosting upcoming meet-ups with AI21 to cover their recently announced [Maestro platform](https://www.ai21.com/maestro/), a new planning and orchestration tool.
   - The meetups are open to the public and free, requiring registration for events in [New York City](https://nyc.aitinkerers.org/p/how-it-s-made-architecting-planning-based-ai-systems-ft-ai21-maestro), [Paris](https://paris.aitinkerers.org/p/ai-tinkerers-paris-ai21-labs-takeover-on-may-19th), and [San Francisco](https://sf.aitinkerers.org/p/how-it-s-made-architecting-planning-based-ai-systems-ft-ai21-maestro).
- **AI21 unveils the Maestro Planning Platform**: AI21 Labs recently announced **Maestro**, a new platform designed for planning and orchestration in AI systems.
   - The platform aims to provide tools and infrastructure for developers to build more sophisticated and efficient AI applications.