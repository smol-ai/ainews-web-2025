---
id: MjAyNS0w
title: >-
  Prime Intellect's INTELLECT-2 and PRIME-RL advance distributed reinforcement
  learning
date: '2025-05-12T05:44:39.731046Z'
description: >-
  **Prime Intellect** released **INTELLECT-2**, a decentralized GPU training and
  RL framework with a vision for distributed AI training overcoming colocation
  limits. **ByteDance** launched **DreamO**, a unified image customization model
  on Hugging Face. **Qwen** released models optimized for GPTQ, GGUF, and AWQ
  quantization. **Gemma** surpassed 150 million downloads on Hugging Face.
  **Meta** released weights for the **Dynamic Byte Latent Transformer** and the
  **Collaborative Reasoner** framework to improve language model efficiency and
  reasoning. **RunwayML** introduced **Gen-4 References**, a near-realtime model
  requiring no fine-tuning. **Mistral AI** released **Mistral Medium 3**, a
  strong multimodal model, and **Le Chat Enterprise**, an agentic AI assistant
  for business. **Google** updated **Gemini 2.5 Pro Preview** with video
  understanding and UI improvements. *"Airbnb for spare GPUs from all over the
  world"* highlights the ongoing challenges and potential of distributed GPU
  training.
companies:
  - primeintellect
  - bytedance
  - qwen
  - gemma
  - meta-ai-fair
  - runwayml
  - mistral-ai
  - google
models:
  - intellect-2
  - dreamo
  - qwen
  - gemini-2.5-pro
  - dynamic-byte-latent-transformer
  - gen-4-references
  - mistral-medium-3
  - le-chat-enterprise
topics:
  - distributed-training
  - reinforcement-learning
  - gpu-clusters
  - model-optimization
  - quantization
  - multimodality
  - agentic-ai
  - video-understanding
  - fine-tuning
people:
  - _akhaliq
  - reach_vb
  - osanseviero
  - aiatmeta
  - c_valenzuelab
  - lmarena_ai
  - adcock_brett
---


**Distributed GPUs are all you need?**

> AI News for 5/9/2025-5/12/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (215 channels, and 12925 messages) for you. Estimated reading time saved (at 200wpm): 1292 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

**The Dream**: "Airbnb for spare GPUs from all over the world"

**The Reality**: colocation for GPUs has been so important that calls for [trillion dollar clusters](https://situational-awareness.ai/) have actually [materialized.](https://news.smol.ai/issues/25-01-21-ainews-project-stargate-dollar500b-datacenter-17percent-of-us-gdp-and-gemini-2-flash-thinking-2)

In this age of accelerating progress, the optimist's trap lies in areas where the promise far exceeds practical reality, especially one where the reality runs in to hard constraints like the speed of light.. It's generally been very difficult to know which of the many attempts at "federated learning" or "distributed training" work stick around long enough to actually get traction. For reasons like these (as well as the simpler reason of lack of understanding), we so far have steered away from covering similar attempts like [Nous Research](https://news.smol.ai/issues/24-01-10-ainews-192024-nous-research-lands-dollar5m-for-open-source-ai)'s work on [DisTrO](https://github.com/NousResearch/DisTrO) despite a lot of excitement from an excitable community. Furthermore, since the AI Engineer focus is very inference oriented, it really doesn't matter what GPU cluster a given model was trained on, further limiting practical industry interest.

However, Prime Intellect's work feels a little different.

[INTELLECT-2's release](https://www.primeintellect.ai/blog/intellect-2-release) isn't just a [paper](https://storage.googleapis.com/public-technical-paper/INTELLECT_2_Technical_Report.pdf#page=11.25), or a [QwQ finetune](https://huggingface.co/collections/PrimeIntellect/intellect-2-68205b03343a82eabc802dc2), or an [RL framework](https://github.com/PrimeIntellect-ai/prime-rl), or opaquely blockchainy techniques, or Yet Another GRPO variant. It's all that and more - a proof of concept and a vision statement and perhaps a very baby steps first articulation of why decentralization has any place in the default-centralizing world of AI:

![image](https://resend-attachments.s3.amazonaws.com/Wj85oPCARIuY0Xu)

Model trainers should look at [Prime-RL,](https://github.com/PrimeIntellect-ai/prime-rl) but [the paper](https://storage.googleapis.com/public-technical-paper/INTELLECT_2_Technical_Report.pdf#page=11.25) also contains interesting insights as to some of the very valid frontiers in both post-training:

![image](https://resend-attachments.s3.amazonaws.com/KrBKbZMvCf7pMlj)

and inference-during-training (which they correctly observe will scale a lot in the RL era)

![image](https://resend-attachments.s3.amazonaws.com/BLqeVRfMeNPQehi)

---

# AI Twitter Recap

**AI Model Releases and Updates**

- **ByteDance released DreamO on Hugging Face**, a unified framework for image customization that supports ID, IP, Try-On, and Style tasks with a single lightweight and performant model [@_akhaliq](https://twitter.com/_akhaliq/status/1921948350145815010).
- **Qwen models optimized for GPTQ, GGUF, and AWQ** were released by Qwen [@reach_vb](https://twitter.com/reach_vb/status/1921956656226668964).
- **Gemma** has surpassed 150 million downloads and 70k variants on Hugging Face, with [@osanseviero](https://twitter.com/osanseviero/status/1921636582873800746) asking the community for suggestions for future versions.
- **Meta** released model weights for their 8B-parameter **Dynamic Byte Latent Transformer**, designed to improve language model efficiency and reliability, as well as the **Collaborative Reasoner** framework, intended to enhance collaborative reasoning in language models [@AIatMeta](https://twitter.com/AIatMeta/status/1921978043998077011) and [@AIatMeta](https://twitter.com/AIatMeta/status/1921966366707613924).
- **RunwayML's** Gen-4 References model has infinite workflows and doesn't require fine-tuning, making it a near-realtime machine for making anything according to [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1921583557333389637) and [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1921356668027249126).
- **Mistral AI** released **Mistral Medium 3**, a multimodal AI model, that performs strongly against proprietary models and **Le Chat Enterprise**, an agentic AI assistant for businesses with tools like Google Drive and agent building [@lmarena_ai](https://twitter.com/lmarena_ai/status/1921667566767845770) and [@adcock_brett](https://twitter.com/adcock_brett/status/1921597108567617585).
- **Google** updated **Gemini 2.5 Pro Preview** with video understanding and improvements for UI, code, and agentic workflows, and **Gemini 2.0 Flash** image generation with improved quality and text rendering [@adcock_brett](https://twitter.com/adcock_brett/status/1921596995371765866).
- **DeepSeek**, part of China's open-source AI movement, has virtually closed the gap with its US peers in two years according to [@hardmaru](https://twitter.com/hardmaru/status/1921374572131254516).
- **Alibaba Qwen** officially released quantized versions of **Qwen3**, which can be deployed via Ollama, LM Studio, SGLang, and vLLM in multiple formats such as GGUF, AWQ, and GPTQ [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1921907010855125019).
- **f-lite 7B**, a distilled diffusion model from f-lite was released [@cloneofsimo](https://twitter.com/cloneofsimo/status/1921931992200847479).
- **Microsoft** updated its **Copilot** with "Pages," a ChatGPT Canvas-like feature, but it doesn't seem to have coding capabilities like Canvas [@adcock_brett](https://twitter.com/adcock_brett/status/1921597040905097496).
- **Microsoft** also announced it's adopting Google's Agent2Agent (A2A) framework, launching it soon on Azure AI Foundry and Copilot Studio [@adcock_brett](https://twitter.com/adcock_brett/status/1921597063478817247).
- **Hugging Face** released **Open Computer Agent**, an open-source AI agent for automating web tasks, but it is reportedly slow and handles only basic multi-step tasks [@adcock_brett](https://twitter.com/adcock_brett/status/1921597198510297124).

**AI Engineering and Tooling**

- **AI-assisted Commit Messages, UI for Windsurf & Cursor Rules, Updated Auto-Approve UI, and Batch History Deletion** are some of the quality of life updates in Cline v3.15, according to [@cline](https://twitter.com/cline/status/1921360242501431364)
- **Slash Gemini 2.5 Pro costs** with Cline v3.15, which integrates Google's new Gemini Implicit Caching, getting up to 75% token discounts automatically on repetitive prompt parts [@cline](https://twitter.com/cline/status/1921359984434246034).
- **Dolphin-MCP** had big updates made, which is an open-source MCP client that lets you use MCP with any AI model, local or cloud [@cognitivecompai](https://twitter.com/cognitivecompai/status/1921417366111482094).
- **Anthropic** released web search capabilities in the API, allowing developers to build applications that can search the web and provide grounded answers with relevant citations [@adcock_brett](https://twitter.com/adcock_brett/status/1921597220928794937).
- [@skirano](https://twitter.com/skirano/status/1921334962097127639) created an MCP server that uses **Anthropic’s new web search tool**, which offers agentic search capabilities where any model can call a Claude instance to return processed search results.
- **Export deep research reports as well-formatted PDFs** with tables, images, linked citations, and sources. This is available to all Plus, Team and Pro users with Enterprise and Edu coming soon, according to [@OpenAI](https://twitter.com/OpenAI/status/1921998278628901322).
- Check out this **AI Research Agent Tutorial** that searches the web and generates cited summaries using LangGraph and Ollama, according to [@LangChainAI](https://twitter.com/LangChainAI/status/1921626371559698666).
- **Microsoft** launched a GitHub connector for ChatGPT that allows users to connect their repos and use ChatGPT's Deep Research to read and search source code and PRs, creating a detailed report with citations [@adcock_brett](https://twitter.com/adcock_brett/status/1921596972735111576).

**Agent Based Systems and Multi-Agent Systems**

- **Langchain** highlighted a few examples of agent based systems and toolkits, such as this company researcher [@LangChainAI](https://twitter.com/LangChainAI/status/1921611360548389145), and this deep research framework for conducting systematic deep research through coordinated LangGraph agents [@LangChainAI](https://twitter.com/LangChainAI/status/1921596224186077352)
- **The Turing Post** shared a deep dive into **Multi-Agent Systems (MAS)**, detailing their architectures, types, recent developments, and current trends [@TheTuringPost](https://twitter.com/TheTuringPost/status/1921350723813683406).
- **FutureHouse**, backed by ex-Google CEO Eric Schmidt, dropped five 'AI Scientist' agents for research, chemistry workflows, and discovery in biology [@adcock_brett](https://twitter.com/adcock_brett/status/1921597086002287090).

**LLM Evaluation and Benchmarking**

- **OpenAI** launched HealthBench, a new evaluation benchmark, developed with input from 250+ physicians from around the world, now available in their GitHub repository [@OpenAI](https://twitter.com/OpenAI/status/1921983050138718531).
- The latest models (Gemini 2.5 Pro, GPT-4.1) are cracked at document parsing and traditional OCR is dead. Human review and correction is still needed, according to [@jerryjliu0](https://twitter.com/jerryjliu0/status/1921621794265665749).
- **lmarena_ai** notes that **Tencent's latest Hunyuan-Turbos** is now ranked #8, with significant improvement over its February version [@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966648795533459).
- **METR_Evals** "doubling every ∼7 mo" slide is in almost every AI progress talk at the moment. It's a striking trend, but it's worth being precise about what's measured: self‑contained code and ML tasks, according to [@polynoamial](https://twitter.com/polynoamial/status/1921618587690893476).

**Key Ideas and Research Directions**

- **Karpathy** suggests that we're missing at least one major paradigm for LLM learning. He calls it "system prompt learning" and thinks it resembles RL in the setup, with the exception of the learning algorithm (edits vs gradient descent). A large section of the LLM system prompt could be written via system prompt learning, it would look a bit like the LLM writing a book for itself on how to solve problems [@karpathy](https://twitter.com/karpathy/status/1921368644069765486).
- **DanHendrycks** shares that AI models are improving at IQ tests (70 IQ → 120), yet they don't feel vastly smarter than two years ago. He argues that useful originality rises steeply only at high intelligence levels, so continued gains are needed for AIs to produce original insights [@DanHendrycks](https://twitter.com/DanHendrycks/status/1921429850432405827).
- **Sakana AI** introduced **Continuous Thought Machines**, a new neural architecture built to use neural dynamics as a core representation for intelligence, enabling adaptive computation and interesting emergent behaviors [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1921749814829871522).

**Academia and Papers**

- **Neel Nanda** shared a guide on turning research into high-quality ML papers, with scientific integrity [@NeelNanda5](https://twitter.com/NeelNanda5/status/1921928364790833651).
- **TheAITimeline** summarized top AI/ML research papers this week, including Absolute Zero, RM-R1, Seed-Coder, Flow-GRPO, and more, providing overviews and author explanations [@TheAITimeline](https://twitter.com/TheAITimeline/status/1921626740675248338).
- **dair_ai** listed top AI papers of the week which included ZeroSearch, Discuss-RAG, Absolute Zero, Llama-Nemotron, The Leaderboard Illusion, and Reward Modeling as Reasoning [@dair_ai](https://twitter.com/dair_ai/status/1921606662214787114).

**Vision Language Models (VLMs)**

- **Phil Schmid** notes that Video Understanding with Gemini 2.5 Pro (05-06) is changing how we will work with videos, allowing for processing up to 6 hours of video in 2 million context with ‘low resolution’, combining audio-visual understanding with code [@_philschmid](https://twitter.com/_philschmid/status/1921838835735867533).
- **Merve Noyan** notes that Llama.cpp has vision language model support now, with support for Gemma 3, Qwen2.5VL, InternVL3 & more [@mervenoyann](https://twitter.com/mervenoyann/status/1921471242852331719).

**Career and Industry**

- **Swyx** reflected on Greg Brockman's career and posed the question, "What would you ask @gdb that can significantly impact what you do/believe?" [@swyx](https://twitter.com/swyx/status/1921992616448831754).
- **Cartesia** is building its India team in Bangalore, seeking SWEs with experience in ML systems, according to [@krandiash](https://twitter.com/krandiash/status/1922016592621404407).
- **Epoch AI** is hiring a Head of Web Development to help communicate their research and manage a team of engineers, shared by [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1921987268337693106).

**Humor/Memes**

- **Karpathy** shared a relatable sentiment about the struggles of intellectual work and grading, using humor to connect with his audience [@karpathy](https://twitter.com/karpathy/status/1921402746902560857).
- **DanHendrycks** jokingly explained why AIs can't come up with good jokes yet [@DanHendrycks](https://twitter.com/DanHendrycks/status/1921433380974948727).
- **Agihippo** humorously lamented about AI researchers never detaching from their work, even at weddings [@agihippo](https://twitter.com/agihippo/status/1921589434488586731).

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Major LLM and Transformer Model Launches (Qwen3, INTELLECT-2, Meta 8B BLT)

- [**Qwen releases official quantized models of Qwen3**](https://i.redd.it/ok2e3kp5jc0f1.jpeg) ([Score: 873, Comments: 90](https://www.reddit.com/r/LocalLLaMA/comments/1kkrgyl/qwen_releases_official_quantized_models_of_qwen3/)): **The image summarizes the official release of Qwen3's quantized models, now available in multiple formats (GGUF, AWQ, GPTQ) for seamless deployment in open-source platforms such as Ollama, LM Studio, SGLang, and vLLM. It visually details various Qwen3 model architectures, including both Mixture of Experts (MoE) and Dense versions, and highlights quantization precisions (BF16, FP8, Int4) with user-configurable generation and mode-switching features, facilitating flexible local inference. The Hugging Face release page lists all official Qwen3 quantized checkpoints for community use ([Hugging Face Qwen3 Collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)).** Commenters are keen to see technical benchmarks comparing Qwen's GGUF quantizations with those from other sources such as Unsloth, especially at longer sequence lengths (128k), and express appreciation for Qwen's comprehensive release strategy, contrasting it positively with Meta's prior releases.
    - There is technical interest in benchmarking the official Qwen3 quantized models against alternative quantizations from the community, particularly the unsloth 128k GGUF versions. Users are keen to see empirical results on relative performance, efficiency, or accuracy for these different quantizations.
    - One commenter highlights that the Qwen3 release stands out for providing official quantized models (GGUF as well as AWQ, GPTQ, and INT8), combined with open weights, a permissive license, and pre-release preparation for integration with open source tooling—contrasting this with how Meta has handled Llama releases.
    - There is a question regarding whether Qwen plans to release QAT (Quantization Aware Training) models in the future, which would enable more efficient, high-quality quantized models compared to post-training quantization.
- [**INTELLECT-2 Released: The First 32B Parameter Model Trained Through Globally Distributed Reinforcement Learning**](https://huggingface.co/PrimeIntellect/INTELLECT-2) ([Score: 434, Comments: 58](https://www.reddit.com/r/LocalLLaMA/comments/1kkgzip/intellect2_released_the_first_32b_parameter_model/)): **INTELLECT-2 is a 32B-parameter language model, trained with distributed global reinforcement learning, using the QwQ-32B base model and the custom prime-rl async distributed RL framework. It leverages verifiable reward signals on math and coding tasks and architectural changes for stability and adaptive length control (optimal generation length is 2k–10k tokens); detailed benchmarks show it narrowly outperforms QwQ-32B on AIME24, LiveCodeBench, and GPQA-Diamond, but underperforms slightly on IFEval. The model supports efficient distributed inference via vllm/sglang, and training utilized a globally permissionless pool of GPUs (see more in the [technical report](https://www.primeintellect.ai/intellect-2)).** Discussion highlights that benchmark differences are within error margins, casting doubt on practical improvements versus QwQ-32B, and emphasizes the significance of the decentralized RL training approach rather than pure performance gains; further, commenters see potential for blockchain/P2P-inspired distributed compute and credit systems for inference and training.
    - INTELLECT-2 is based on the QwQ 32B architecture and achieves benchmark results (e.g., AIME24: 78.8 vs. 76.6; LiveCodeBench: 67.8 vs. 66.1) that are within the margin of error of QwQ-32B, suggesting limited generalization beyond their specific dataset. However, the significance is in the decentralized RL training method (globally distributed reinforcement learning), not just the marginal performance improvement.
    - The distributed RL training approach behind INTELLECT-2 enables scaling up to 32 billion parameters using large, globally distributed compute resources—evidencing the technical feasibility and potential of decentralized AI training methods. Community discussion also notes the potential applicability to distributed inference systems, P2P, or blockchain-inspired networks, possibly coupled with credit/reward systems to incentivize compute contribution.
    - Benchmarks provided in the blog and technical report (linked by users) substantiate claims about INTELLECT-2's performance and reiterate the technical focus of the release: the combination of large model scale and decentralized, collaborative reinforcement learning as a novel systems contribution, rather than just improved performance on standard tasks.
- [**Meta has released an 8B BLT model**](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning/?utm_source=twitter&utm_medium=organic%20social&utm_content=video&utm_campaign=fair) ([Score: 108, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1kky1sg/meta_has_released_an_8b_blt_model/)): **Meta has released an 8B parameter Byte Latent Transformer (BLT) model, which focuses on byte-level tokenization for improved multilingual and multimodal performance. The model was originally discussed in late 2023 ([Meta BLT paper](https://www.reddit.com/r/LocalLLaMA/comments/1hdpw14/metas_byte_latent_transformer_blt_paper_looks/)), and benchmarks at the time indicated its strengths in efficient byte-level processing. No new major technical details or benchmarks have emerged since those 2023 releases.** Commenters note BLT's release was not recent and express demand for a more substantial Llama 4 series at higher parameter counts. There's also skepticism about BLT's practical impact compared to existing Llama models.
    - There is ongoing demand for larger and more varied Llama 4 model sizes (e.g., 32B, 11B, 8B), suggesting the community seeks performance scalability and flexibility beyond what Meta has currently released, with some expressing that Meta lags behind expectations following Llama 4.
    - Multiple commenters note that the BLT (Byte Latent Transformer) isn't a new release, referencing earlier discussions from last year and last month about both the BLT and Meta's perception model. There is a lack of clarity on practical improvements or novel aspects compared to those prior releases.
    - Evabyte (6.5B), an open-source, byte-based model, is cited as a previous example of byte-level architectures, questioning what differentiates Meta's BLT beyond scaling to 8B parameters. The technical distinction and comparative performance versus other 8B models remain under discussion, with users skeptical of significant advancements.

### 2. Microsoft ARTIST Framework for Agentic Tool-augmented LLMs

- [**Microsoft Researchers Introduce ARTIST**](https://i.redd.it/90acs85p7c0f1.png) ([Score: 212, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1kkq8q8/microsoft_researchers_introduce_artist/)): **The provided image is a table summarizing the performance of various models (notably Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct) when enhanced with the ARTIST framework on challenging mathematical reasoning benchmarks (MATH-500, AIME, AMC, and Olympiad). The table demonstrates that integrating ARTIST leads to consistent and significant improvements in pass@1 accuracy across all benchmarks, with the Qwen2.5 models surpassing larger models like GPT-4o, sometimes by as much as 22%. The technical approach combines agentic reasoning, dynamic tool use (including features like** `<think>` **and** `<tool_name>` **actions), and reinforcement learning (notably with GRPO and customized reward functions), promoting robust, interpretable multi-step reasoning.** Several comments note the strong benchmarks chosen and the use of tools like web search, acknowledging that tool integration reliably boosts scores but questioning real-world generalization due to the benchmarks used. There's also a discussion about how ARTIST's RL approach with loss masking and structured rewards enables emergent agentic behaviors (e.g., self-correction and self-reflection), with some users noting the practical boost over baseline or distilled models.
    - Several comments highlight that ARTIST's benchmarking setup shows the 7B and 14B models performing surprisingly close to GPT-4o, but it's noted this may reflect the benchmark's characteristics rather than general capability, especially since it includes tool use (e.g., web search) and isn't a Google-proof dataset like GPQA. The point is made that integrating agentic reasoning and tool calls predictably boosts scores, so care must be taken in interpreting these results.
    - A detailed breakdown describes how ARTIST combines agentic reasoning, tool integration, and reinforcement learning (RL) using the GRPO (Group Relative Policy Optimization) method. Technical highlights include a loss masking RL strategy to focus learning on the LLMs reasoning/actions rather than copying deterministic tool output, and a composite reward system—answer, format, tool execution/state/function rewards—to guide correct and interpretable stepwise problem-solving behavior.
    - Experimental results indicate that ARTIST achieves up to a `22%` absolute improvement over base models for mathematical reasoning benchmarks (AMC, AIME, Olympiad) and over double the accuracy for multi-turn function calling on the τ-bench. These gains are attributed to emergent agentic behaviors (self-refinement, self-correction, self-reflection) that arise from the model's design and training, rather than from manual supervision.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Recent Model and Feature Launches (Manus AI, JoyCaption, Continuous Thought Machines)

- [**Manus AI has officially launched publicly**](https://i.redd.it/2179eel6le0f1.jpeg) ([Score: 162, Comments: 56](https://www.reddit.com/r/singularity/comments/1kl1q2q/manus_ai_has_officially_launched_publicly/)): **The image shows an official announcement from ManusAI, indicating their public launch with removal of the waitlist. All users now receive one free task per day and an additional 1,000 credits. The screenshot features a user interface where users can assign tasks such as 'Animated problem tutorial' and 'Interactive learning website', suggesting ManusAI is focused on educational or content generation tasks. More information is available in their tweet [here](https://x.com/ManusAI_HQ/status/1921943525261742203) and the image [here](https://i.redd.it/2179eel6le0f1.jpeg).** Commenters ask for clarifications about what ManusAI is; one criticizes its current capabilities and expense relative to alternatives like Claude, stating 'It's not developed enough to deliver on all the features that are promised.'
    - Some users report Manus AI is *not available in all regions*, either due to geoblocking or early launch restrictions, which can prevent thorough evaluation or adoption beyond select markets.
    - Comparisons are being made to established models like Claude for code-related capabilities. One comment states that Manus AI *did not impress* and is both *expensive* and *not developed enough to deliver on all its promised features*, suggesting that in its current state, its implementation and cost/performance ratio may lag behind competitors.
- [**JoyCaption: Free, Open, Uncensored VLM (Beta One release)**](https://www.reddit.com/r/StableDiffusion/comments/1kl2nek/joycaption_free_open_uncensored_vlm_beta_one/) ([Score: 272, Comments: 44](https://www.reddit.com/r/StableDiffusion/comments/1kl2nek/joycaption_free_open_uncensored_vlm_beta_one/)): **JoyCaption Beta One is a free, open-source, uncensored visual language model (VLM) for image captioning, with Intended utility for training diffusion models. Major technical advancements over Alpha Two include: doubled training data (2.4M samples total), new 'Straightforward Mode' for more succinct captions, reworked and stabilized booru tagging (alphabetic and categorical grouping, reduced repetition via improved format and DPO reinforcement learning), more accurate watermark annotation (via custom watermark-detection ML), hand-written 2000+ VQA pairs for instruction-following, and support for user-specified tag augmentation. The model underwent two rounds of Direct Preference Optimization (10k & 20k preference pairs), yielding significant improvements in output preference and glitch reduction (down to 1.5–3%) as measured by human and SOTA VLM evaluation. Beta One achieves 67% normalized accuracy (human benchmarked) on validation sets vs. 55% for Alpha Two and a prior GPT-4o. The training dataset and model are openly released on HuggingFace: https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava and https://huggingface.co/datasets/fancyfeast/joy-captioning-20250328b.** Technically-minded commenters discuss anticipated improvements (especially 'Straightforward Mode'), questions about the alignment of prompt formats with those used by major diffusion models (SD3.5, XL), and possible integration into GUIs or workflows (such as ComfyUI nodes). There is recognition of the model's flexibility, uncensored focus, and value as a community tool for dataset preparation.
    - Users are discussing the transition from previous versions (such as A2) to the new B1 release, specifically in the context of integrating it into GUI workflows. One user mentions minimal caption corrections are needed when prompts are obeyed, highlighting strong prompt adherence in the model's outputs.
    - Technical debate arises over the renaming of 'Training Mode' to 'Stable Diffusion', with questions about whether this signifies alignment with training caption styles of SD3.5/XL or alternative approaches (like Flux or HiDream). This raises questions around versatility and whether naming is merely for convenience or indicates intent for broader SD compatibility.
    - There is concern regarding VRAM usage, especially for those with consumer GPUs like the RTX 4070 Super (12GB), as earlier JoyCaption releases had out-of-memory issues. Users are explicitly seeking clarification if Beta One optimizes for lower VRAM requirements to enable more accessible local inference.
- [**Introducing Continuous Thought Machines**](https://x.com/sakanaailabs/status/1921749814829871522?s=46) ([Score: 332, Comments: 63](https://www.reddit.com/r/singularity/comments/1kkm5e0/introducing_continuous_thought_machines/)): **Sakana AI introduces the Continuous Thought Machine (CTM), a novel model architecture where reasoning is driven by neuron-level timing and synchronization, inspired by biological neural systems. Unlike conventional ANNs, CTM neurons encode signal history and timing with learnable parameters, allowing complex, temporally-coordinated behaviors and interpretable, step-wise problem solving. Technical materials and benchmarks indicate gains in efficiency and problem-solving across various tasks ([arXiv preprint](https://arxiv.org/abs/2505.05522), [interactive demo](https://pub.sakana.ai/ctm/), [GitHub](https://github.com/SakanaAI/conti)).** Discussion notes Sakana AI's reputation for innovative research, but some commenters are uncertain about the significance of the advance, reflecting a demand for comparative benchmarks or practical impact analysis.
    - Sakana AI's Continuous Thought Machine (CTM) introduces a novel architecture using synchronization of neuron activity—specifically, leveraging timing information at the neuron level—to mimic complex behaviors found in biological neural networks. Unlike conventional artificial neural networks which primarily process information in discrete activations, CTM's stepwise, time-aware mechanism allows for more interpretable, human-like reasoning and potentially richer solution paths for a variety of tasks.
    - Initial research reported by Sakana AI suggests that CTM improves both problem-solving performance and computational efficiency across different tasks compared to traditional neural networks. The approach is positioned as an advancement toward closing the gap between artificial and biological reasoning systems, and could enable new types of reasoning previously less accessible to standard architectures.

### 2. Major Model and Industry Trend Analysis (Microsoft/LLMs, Copyright Office, AI Researcher on ChatGPT issues)

- [**The scale of Microsoft's influence in LLMs and software development world is crazy.**](https://i.redd.it/918wyo1k1a0f1.jpeg) ([Score: 571, Comments: 58](https://www.reddit.com/r/singularity/comments/1kkjop0/the_scale_of_microsofts_influence_in_llms_and/)): **The diagram visually details Microsoft's far-reaching involvement in LLMs and software development tools, highlighting its 49% profit share in OpenAI (which owns the $3B Windsurf) and VSCode—a core repository for AI-enhanced IDEs. Cursor AI, a $9B company, is shown as both a fork of Microsoft's VSCode and a recipient of OpenAI investment, underlining Microsoft's ecosystem lock-in and indirect influence over major generative AI tools. Technical discussion in comments further mentions Microsoft's stake in GitHub Copilot, another major AI code assistant, complementing its portfolio.** Commentary debates how the diagram may overemphasize Microsoft's reach by omitting other players, while acknowledging Microsoft's strategic advantage from its open-source VSCode and its integration with AI tooling.
    - A commenter highlights the misconception regarding Microsoft's control over OpenAI, pointing out that the frequently cited 49% stake only reflects a profit-share agreement and priority access to models, not real ownership or operational control. This nuance is significant in discussions about Microsoft's actual influence on frontier LLM development.
    - Another user points out that to assess Microsoft's influence credibly, one should reference concrete metrics like IDE market share. Without quantitative data, claims about Microsoft's dominance lack clear context or substantiation.
    - The discussion references key Microsoft tools—specifically GitHub Copilot and VSCode—both of which are noted to have become industry staples. GitHub Copilot demonstrates Microsoft's leadership in coding assistance, while VSCode's widespread adoption underscores their substantial presence in developer tools, though some debate its impact versus the older Visual Studio suite.
- [**Ex-OpenAI researcher: ChatGPT hasn't actually been fixed**](https://open.substack.com/pub/stevenadler/p/is-chatgpt-actually-fixed-now?r=4qacg&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) ([Score: 603, Comments: 133](https://www.reddit.com/r/ChatGPT/comments/1kkydfa/exopenai_researcher_chatgpt_hasnt_actually_been/)): **Steven Adler, former OpenAI dangerous capability testing lead, details persistent alignment issues in ChatGPT, especially unreliable sycophancy and overcorrection to contrarianism, despite recent attempted fixes. Using adapted versions of Anthropic's publicly available sycophancy benchmarks (over 200 automated tests), Adler demonstrates that OpenAI did not effectively deploy basic automated checks against sycophant behavior previously described in academic literature (see e.g. [Perez et al., 2023](https://arxiv.org/abs/2305.10455)), resulting in unpredictable and unsafe model responses. His findings raise core concerns about the current technical limits of scalable, dependable LLM alignment, especially as capability and deployment increase; see full article for test methodology and detailed results.** Top comments focus on the potentially catastrophic risks of wide-scale AI agent deployment in uncontrolled settings, emphasize the need for public alignment benchmarks to drive corporate accountability, and question whether current understanding and safeguards are adequate for future, more powerful models.
    - Several comments highlight concerns about control and alignment of AI models, especially as they become more powerful and are integrated into critical systems. There's a specific focus on 'unintended effects,' such as models becoming overly sycophantic (agreeable to a fault), which may not be problematic when the AI role is limited but poses significant risks in high-stakes or autonomous applications.
    - Discussion touches on the potential for public, task-specific benchmarks (e.g., a sycophancy benchmark) as a method of enforcing transparency and accountability in model safety. This approach could pressure companies to better evaluate and correct behaviors that current reward-based feedback systems (thumbs up/down) may inadvertently reinforce, which can lead to misbehavior or loss of alignment with intended outcomes.
- [**US Copyright Office Set to Declare AI Training Not Fair Use**](https://www.reddit.com/r/StableDiffusion/comments/1kkj7wr/us_copyright_office_set_to_declare_ai_training/) ([Score: 390, Comments: 240](https://www.reddit.com/r/StableDiffusion/comments/1kkj7wr/us_copyright_office_set_to_declare_ai_training/)): **The US Copyright Office has released a pre-publication report indicating that training generative AI models on copyrighted content for commercial purposes likely exceeds the boundaries of fair use, especially when such use "competes with [copyrighted works] in existing markets" or involves "illegal access" ([report PDF](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)). This pre-decisional guidance comes amid increased Congressional attention and was followed by the firing of the Office's head ([news link](https://www.theverge.com/news/664768/trump-fires-us-copyright-office-head)).** Commenters question selective enforcement and the inconsistency of existing copyright practices, using Getty's image scraping and legal strategies as a comparative example, highlighting the ambiguity and potential inequities in the current legal landscape for both AI and traditional content aggregators.
    - One commenter highlights the complex situation regarding image licensing, referencing Getty Images. They bring up a technical legal precedent in which Getty combed through millions of public domain and fair use images, integrated them into their platform, and then issued takedown notices against original photographers, with courts ruling in Getty's favor. The discussion points toward nuanced and potentially exploitative approaches to digital copyright and licensing systems, suggesting there are gaps or legal ambiguities in current fair use enforcement.
    - Another comment observes that the US Copyright Office only has an advisory role and cannot independently change copyright law. Actual legislative changes require action by both the House and Senate, and are subject to presidential approval or veto. This clarification highlights the technical and procedural limitations of the Copyright Office's authority in the copyright and AI training debate.

### 3. Community Dissatisfaction and Behavioral Shifts in ChatGPT Usage

- [**GPT used to think with me. Now it babysits me.**](https://www.reddit.com/r/OpenAI/comments/1kl1k5h/gpt_used_to_think_with_me_now_it_babysits_me/) ([Score: 126, Comments: 60](https://www.reddit.com/r/OpenAI/comments/1kl1k5h/gpt_used_to_think_with_me_now_it_babysits_me/)): **The OP notes a significant decline in GPT-4's ability to engage in deep critical reasoning and nuanced conversation, describing recent updates as infantilizing and less adaptive. Multiple commenters echo this degradation: one critiques GPT-4o's performance as 'plain useless,' with consideration of alternatives like self-hosting DeepSeek V3, while another details a shift from GPT-4's earlier precision and utility for humanities and general knowledge to excessive formality, emojis, follow-up questions, and hallucinations in newer iterations (4o, o3, o4-mini-high, o4-mini). The changes are viewed as a move towards coddling rather than intelligence, with advanced users feeling marginalized.** Discussion centers on the trade-off between tuning models for a broader audience versus maintaining value for expert users, with explicit dissatisfaction regarding hallucination rates and loss of technical rigor. The 'Eternal September' concept is referenced, suggesting a perceived decline in community or system quality as AI becomes mainstream.
    - Multiple users report that GPT-4o's performance has notably declined, emphasizing that newer variants (e.g., o3, o4-mini, o4-mini-high) exhibit increased hallucinations and lack the precision, depth, and utility of earlier GPT-4 models, especially for technical or serious research applications.
    - A technically detailed hypothesis proposes that these perceived degradations may stem from architectural and strategic changes: Mixture-of-Experts (MoE) mechanisms, aggressive inference cost optimizations, and reinforced safety fine-tuning are floated as reasons for the 'smoothing' of model outputs, reducing their depth and character.
    - A broader strategic shift is suggested, arguing that OpenAI may be deliberately 'flattening' the base model outputs (making them more neutral and less distinct) to set the stage for future paid or modular customization options, aligning with Sam Altman's remarks about 'modular personalities' and potential business changes ahead of an IPO.
- [**Is anyone else’s ChatGPT straight up dumb now???**](https://www.reddit.com/gallery/1kl208m) ([Score: 240, Comments: 230](https://www.reddit.com/r/ChatGPT/comments/1kl208m/is_anyone_elses_chatgpt_straight_up_dumb_now/)): **The OP reports a marked decline in ChatGPT performance (version not explicitly stated) over the past week, with frequent errors, memory/consistency problems (e.g., forgetting user-supplied information within the same dialogue), and unreliable factual outputs. These observations suggest either a backend model or deployment change affecting short-term conversational memory and retrieval accuracy. No explicit technical details or logs are shared, only user perception and qualitative examples.** Multiple commenters corroborate this decline, citing persistent issues with factual accuracy, number handling, and increased hallucination—one notes that *accusing ChatGPT of lying* prompts more honest self-correction, while others claim the system has *always* struggled, especially with numerical consistency, but that recent weeks represent a notable degradation.
    - Users report persistent and sometimes worsening issues with ChatGPT's accuracy in numerical calculations, citing frequent incorrect answers even for simple math tasks. There is consensus that outputs must be manually double-checked, as otherwise errors often go unnoticed.
    - Some users mention attempting to mitigate misinformation by confronting the model directly—accusing it of lying or demanding sources—which sometimes improves response accuracy, but reliable verification remains a challenge as the model does not always self-correct unless prompted.
    - There is a theme of frustration among users employing ChatGPT in technical or quantitative contexts (e.g., as a personal trainer for workout calculations), necessitating user intervention and repeated correction for the assistant to improve performance in specific tasks.
- [**Teachers Using AI to Grade Their Students' Work Sends a Clear Message: They Don't Matter, and Will Soon Be Obsolete**](https://futurism.com/teachers-ai-grade-students) ([Score: 149, Comments: 68](https://www.reddit.com/r/singularity/comments/1kkuad2/teachers_using_ai_to_grade_their_students_work/)): **The post critiques the use of AI, specifically models like ChatGPT and Mixtral, in automating grading, arguing it signals teachers' obsolescence. Commenters clarify that automated grading tools have existed for decades and note current models like Mixtral are not state-of-the-art (SOTA), with more advanced solutions being developed (e.g., Khan Academy's AI tools for feedback and plagiarism detection). Use cases cited include leveraging AI in peer review contexts to improve scientific writing quality and consistency, emphasizing that AIs can help catch basic methodological errors often missed by mentors. Direct teaching duties and nuanced classroom management remain outside AI capabilities.** Debate centers on whether AI grading diminishes the teacher's role or simply relieves unpaid overtime, with consensus that full teacher replacement is unlikely given the complexity of classroom interaction and planning. Commenters recommend sources like Ethan Mollick for nuanced perspectives on AI in education, noting the value in AI as an instructive feedback tool rather than a replacement.
    - Commenters note that the use of AI for grading is not novel—grading machines have existed for decades. The significant shift is in freeing teachers from manual grading to focus on pedagogy and classroom management, areas where current AI cannot effectively substitute for human experience or contextual adaptability, such as lesson planning and handling live class dynamics.
    - It is highlighted that some reported AI tools lag behind state-of-the-art; for example, Mixtral is cited as not competitive with leading models, especially compared to more advanced offerings used by platforms like Khan Academy (which deliver automated essay feedback and plagiarism checks). There is an emphasis on the value of work from educational AI researchers like Ethan Mollick for those interested in the space.
    - Peers describe using AI (e.g., ChatGPT) as a peer-review tool on partial drafts to catch remedial mistakes missed by mentors or advisors, such as incorrect statistical reporting and misuse of statistical tests. Proper instruction on leveraging AI for feedback can structurally improve academic writing and reduce errors before submitting work, potentially saving significant time in peer review processes.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: The Model Gauntlet: New Releases, Performance Showdowns, and Lingering Quirks**

- **Drakesclaw & Absolute Zero Shake Up Rankings**: A new contender, **Drakesclaw**, storms the **LM Arena**, with [initial impressions](https://discord.com/channels/1340554757349179412/1340554757827461211/1371049419407102013) hinting at **Gemini 2.5 Pro**level performance. Meanwhile, the [**Absolute Zero Reasoner (AZR)** paper](https://arxiv.org/abs/2505.03335) details a model achieving SOTA on coding/math tasks via self-play with zero external data, a concept also explored in the **Yannick Kilcher** and **Nous Research AI** Discords.
- **Gemini & Qwen3 Stumble on Tools and Context**: Users across **LM Studio**, **Cursor Community**, and **OpenRouter** report **Gemini 2.5 Pro** experiencing tool call failures, file reading problems, and BYOK billing issues on Google AI Studio, with [Google AI Studio also imposing new rate limits](https://discord.com/channels/1091220969173028894/1092729520181739581/1370886288244211944) on the experimental version. **Qwen3** models also frustrated **LM Studio** users by generating invalid JSON for tool calls, breaking **Neovim** completion, and **Unsloth AI** users found it incompatible with tool-calling, prompting a [PR fix for notebooks](https://github.com/unslothai/notebooks/pull/41).
- **Caching Chaos and Context Conundrums Plague Claude & Gemini**: **OpenRouter** users flagged **Claude 3.7** caching failures on **Vertex AI**, contrasting with working Anthropic endpoints. **LMArena** buzzed about **Gemini Exp 1206's** fluctuating context window ([originally 2M tokens, then 1M, now 32k?](https://xcancel.com/OfficialLoganK/status/1865081419015352689)), referencing [the NoLiMa paper](https://arxiv.org/abs/2502.05167) to argue that window size is moot if the model can't use it effectively.

**Theme 2: Rise of the Agents: Frameworks, Finetuning, and Interoperability Efforts**

- **Unsloth & Aider Push Agentic Boundaries**: The **Unsloth AI** community anticipates finetuning focused on **agentic behavior**, simplifying tool calling with triple-quoted Python-style strings and emphasizing that the *dataset is the secret sauce*. **Aider** v0.83.0, discussed in its Discord, now supports `gemini-2.5-pro-preview-05-06` and `qwen3-235b` (detailed in its [release notes](https://aider.chat/HISTORY.html)), and its architect mode helps plan multi-step edits.
- **MCP Tools Proliferate for Enhanced Agent Interaction**: The **MCP (Glama)** Discord showcased **AiraHub's** new streamable HTTP protocol for MCP/A2A tools ([AiraHub2 repo](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main)) and **DiffCalculia_MCP** for AI-assisted large file editing using unified diffs and [diffcalculia-ts](https://github.com/createthis/diffcalculia-ts). The new **fabric-mcp-server** also integrates [Fabric patterns from Daniel Miessler's repo](https://github.com/danielmiessler/fabric) with Cline in VS Code for AI-driven execution.
- **New Agent Frameworks Emerge**: **HuggingFace** discussions highlighted **Agentle**, a Python framework for building type-safe AI agents (slated for May 16, 2025 release, [Agentle repo](https://github.com/paragon-intelligence/agentle)), and the open-sourcing of [Cyberdesk](https://github.com/cyberdesk-hq/cyberdesk), a service enabling AI agents to control a virtual desktop ([Cyberdesk website](https://www.cyberdesk.io/)).

**Theme 3: Powering Up: Hardware Hustles, Local LLM Deployments, and Optimization Frontiers**

- **NVIDIA 5090 Drivers Boost Performance, AMD Specs Scrutinized**: **LM Studio** members saw **NVIDIA 5090** inference for **Qwen3 30B MoE Q4** skyrocket to over 170 t/s after a driver update to **576.02**, while also speculating the upcoming **AMD Ryzen AI Max 395 Mini PC** might offer around 4-6 tkps for 70B models with its expected quad-channel DDR5. **GPU MODE** discussed potential **Triton** performance on the **NVIDIA 50 series** and the lack of **nvbench** alternatives in **ROCm**, pointing to **ScalarLM's MI300X memcpyPeer benchmarks** as a resource.
- **Unsloth Quants and LM Studio Streamline Local LLM Setups**: **Unsloth AI**'s **Dynamic 2.0 GGUF quants** are enabling more *human-like conversations* and accurate models, especially using *F32* for non-BF16 hardware. **OpenAI** and **GPT4All** Discords recommended [LM Studio](https://lmstudio.ai/) for running local models like **Llama** and **DeepSeek**, though **GPT4All** users faced boot issues requiring AVX/AVX2 CPUs.
- **Mojo & Torch Grapple with Compilation and Memory**: The **Modular (Mojo 🔥)** community discussed removing **autotuning** due to complexity, with plans for **post-hoc trait conformance via extensions**, and enabling **bare metal programming** by exposing compiler flags for no-stdlib binaries ([Mojo FAQ on telemetry](https://docs.modular.com/mojo/faq/#does-the-mojo-sdk-collect-telemetry)). **GPU MODE** users tackled `torch.export` specializing batch sizes, requiring debugging with `TORCH_LOGS="+dynamic"`, and debated the performance pitfalls of *array-of-structs* designs, advocating for HPC formats like **COO**.

**Theme 4: Framework Frontiers: Innovations in DSPy, LlamaIndex, and Specialized Tooling**

- **DSPy Develops Doctrine and Async Support**: The **DSPy** community discussed a new [X post outlining the "DSPy Doctrine"](https://x.com/lateinteraction/status/1921565300690149759), its core design philosophy, alongside progress on **async LLM call support** for enhanced parallel processing. A member also presented at an **AI in Insurance** conference on using DSPy for optimizing correspondence templates ([slides in German](https://drive.proton.me/urls/Y6YQC1EQ7G#ddX424butZxG) and [English](https://drive.proton.me/urls/M8W9PBX5KC#jXgLvgypyp3R)).
- **LlamaIndex Launches Research & RAG Tools**: **LlamaIndex** unveiled [PapersChat](https://t.co/ASzjwLfKLC) for interacting with **Arxiv/PubMed** papers and a [Multilingual, Multimodal RAG System](https://t.co/69CHCCn8J3). They also released a tutorial on building a [Deep Research Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A) and updated **LlamaParse** with new models and [auto orientation detection](https://t.co/tqi17dPJm4).
- **Specialized Code & Data Tools Emerge**: **Unsloth AI** saw the release of **Mellum-4b-sft-rust**, a **CodeFIM** model for Rust, available on [Hugging Face at Etherll/Mellum-4b-sft-rust](https://huggingface.co/Etherll/Mellum-4b-sft-rust) with its [CodeFIM-Data dataset](https://huggingface.co/datasets/Etherll/CodeFIM-Data). **Notebook LM** users are generating agents using Library and Information Science techniques for study and creating semi-automated *Zundamon videos* from summaries, like this [PaperBench paper example on](https://x.com/kakira9618/status/1919666922234511795) [X.com](http://x.com/).

**Theme 5: Reality Check: Benchmarking Battles, Hallucination Headaches, and Ethical Enigmas**

- **Benchmarking Under Fire as HealthBench Debuts**: **OpenAI** launched **HealthBench**, an evaluation benchmark for health models developed with input from [250+ physicians and detailed on OpenAI's site](https://openai.com/index/healthbench/). This comes as **LMArena** users debate its [leaderboard validity](https://discord.com/channels/1340554757349179412/1340554757827461211/1371199758458425519) and **Eleuther** members found inconsistencies in **Global MMLU** answers across languages.
- **Models Behaving Badly with Hallucinations and Errors**: **LMArena** users grappled with LLM **hallucinations** when researching historical facts, noting [Grok's potential for discerning credible sources](https://discord.com/channels/1340554757349179412/1340554757827461211/1370920176254539806). **Yannick Kilcher**'s Discord saw users frustrated with [**Claude.ai](http://claude.ai/)'s** web UI losing work on internal server errors, possibly due to content moderation.
- **AI Ethics in the Arena: Military Drones and Pricing Transparency Spark Concern**: Discussions in **OpenRouter** touched on OpenAI's potential military contract to provide LLMs for drones, based on a [Wired article about OpenAI and Anduril](https://www.wired.com/story/openai-anduril-defense/), with one member calling it *"a horrifyingly stupid idea"*. **Cursor Community** users expressed confusion over **Cursor's pricing**, especially the [20% API markup in Max mode detailed in Cursor's docs](https://docs.cursor.com/models?max-mode=true#pricing).


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Text Mode Prompt Ready for Testing**: A member created a prompt to play any game in **Perplexity** in **text mode**, aiming for global use, with another member inquiring about using it for an *international project*.
   - They are [seeking to connect via DM](https://discord.com/channels/1047197230748151888/1047649527299055688/1370482465008910337) signaling collaboration.
- **AI Detector's Watermark Focus Clarified**: It was clarified that **AI detectors** primarily detect **common watermarks** and techniques like em dashes, used within their algorithm.
   - Members discussed the reliability of AI detectors, with one sharing a link to [originality.ai](https://originality.ai/ai-checker) for testing.
- **Qwen's Performance Elicits Mixed Reactions**: Members discussed the initial performance of **Qwen**, noting its impressive PDF output capabilities, while others debated its reasoning abilities compared to models like Deepseek.
   - Overall **Qwen** seemed well received but *sloppy*, and did not stand up as well to **OpenAI** in deep research.
- **Perplexity API's Image Handling Bug Reported**: Users reported a bug where the **API** returns image URLs in the format *x-raw-image:///xxxxxxxxx*, and requested additional metadata for returned images, such as captions or alt text from the source article.
   - Currently, the URL and source article URLs are the only indicators available.
- **Perplexity API Domain Filtering Enhanced**: The **Perplexity API** now supports specifying subdirectories within domains for more precise filtering, enabling users to target specific content sections with both [inclusion and exclusion rules](https://example.com/filtering).
   - For example, one can now focus on *nytimes.com/section/world* while excluding *bbc.co.uk/sport*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Dynamic Quants Make Waves!**: The community explores **Unsloth's Dynamic 2.0 GGUF quants**, which lead to highly accurate models and more *human-like conversations* using the *F32* format for non-BF16 hardware.
   - Enthusiasts hope for Dynamic 2.0 quants for NousResearch's DeepHermes models and Orpheus TTS, while someone humorously remarks that OpenAI isn't alone in having questionable naming conventions.
- **Agentic Finetuning on the Horizon**: Future efforts will focus on **agentic behavior** and autonomy rather than general chat, aiming to balance agentic and non-agentic data using visualizations to track progress.
   - Tool calling is simplified using triple-quoted Python-style multi-line strings for easier LLM code generation with raw text strings instead of Python functions, also implementing realistic *is typing* indicators emulating human delays.
- **Dataset is Secret Sauce for Tool Calling**: The dataset is the *secret sauce* for tool calling in models, with the next version heavily emphasizing **agentic capabilities** while retaining chat functionality via hand-written data.
   - It was also discovered that the latest **Qwen3** models are incompatible with tool-calling, responding with *knowledge* answers instead of parameter calls, with [a PR](https://github.com/unslothai/notebooks/pull/41) to address issues.
- **CodeFIM Model Arrives to Rustaceans!**: A new **CodeFIM** (Fill-In-The-Middle) model for **Rust**, trained using **Unsloth**, is available on [Hugging Face](https://huggingface.co/Etherll/Mellum-4b-sft-rust), named **Mellum-4b-sft-rust**, with a **GGUF** version also available.
   - The [dataset](https://huggingface.co/datasets/Etherll/CodeFIM-Data) contains 8192 max tokens using the Qwen tokenizer; the community has inquired about a similar **CodeFIM** dataset for **Python**.
- **Synthetic Data Aids Student Convergence**: Synthetic data & knowledge distillation could improve deployment for smaller models like `gemma-3-1b-it-unsloth-bnb-4bit`, sharing [a synthetic data notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks).
   - Weight decay & cosine learning rates aid convergence: suggesting **weight decay of 0.05**, **warmup steps of 0.05 - 0.1**, a **cosine learning rate scheduler**, and an **LR of 3e-6** for better convergence.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Drakesclaw Challenges the Giants**: A new model named **Drakesclaw** appeared on the LM Arena, with [initial impressions](https://discord.com/channels/1340554757349179412/1340554757827461211/1371049419407102013) suggesting it might rival **Gemini 2.5 Pro** in certain tasks.
   - The community reacted with excitement comparing this model to others in its class.
- **o3 Pro: Vaporware or Visionary?**: The ongoing wait for **o3 Pro** has become a community meme, with members jokingly tracking the [days since its expected release](https://discord.com/channels/1340554757349179412/1340554757827461211/1371268890298159124), with some predicting it will never arrive.
   - There is community speculation about whether **o3** can solve major outstanding problems in tech, including one member joking, *If o3 pro can’t solve Reimanns Hypothesis im refunding*.
- **Hallucination Highway: Navigating LLM Fables**: Members discussed the challenges of dealing with **hallucinations** in LLMs, particularly when researching historical facts, with one noting [Grok's potential](https://discord.com/channels/1340554757349179412/1340554757827461211/1370920176254539806) for discerning credible sources.
   - Search engines are seen as exacerbating the **hallucination** problem, with one member saying, *searches induce hallucinations so much they're all unusable*.
- **Context Caper: Gemini's Shrinking Window?**: A debate erupted over **Gemini Exp 1206's** context window, with conflicting accounts of whether it was [released with 2M tokens](https://xcancel.com/OfficialLoganK/status/1865081419015352689), later capped at 1M, and then reduced to 32k.
   - Referencing [the NoLiMa paper](https://arxiv.org/abs/2502.05167), it's been emphasized that *the context window doesn't matter if the model cannot really work on it properly*.
- **LM Arena: Fair Fights or Flawed Figures?**: Discussion addressed the **LM Arena** leaderboard, with members debating [the validity of its rankings](https://discord.com/channels/1340554757349179412/1340554757827461211/1371199758458425519) and potential failure modes.
   - One member argued, *it's down to people to learn how to read this properly*, suggesting users need to understand the platform's limitations.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 3 Tool Argument Snafu**: Users discovered **Qwen3** models generate invalid **JSON** when calling tools, specifically adding an extra `}` when the `code` value contains escaped quotes and braces, breaking **Neovim** code completion.
   - A member shared a [relevant Github issue](https://github.com/olimorris/codecompanion.nvim/pull/1141#issuecomment-2834614801) noting a related bug that might be fixed with a higher quality Q model.
- **LM Studio API Lacks Tool Reporting**: A user found the **LM Studio API** lacks a documented way to determine what **tools** are called by the model when using `model.act`, because it spawns a new thread, making exception handling difficult.
   - The user reverse-engineered a workaround parsing `AssistantResponse`, `ToolCallRequest`, and `ToolResultMessage` from `lmstudio.history`, emphasizing the need for an official API feature for **tool reporting**.
- **AMD 395 Channel Speculations**: Members speculated on the performance of the **AMD Ryzen AI Max 395 Mini PC**, expecting 200 GB/s with quad-channel DDR5, and its impact on running 70B models.
   - Predictions ranged from around **4 t/s** to **6 tkps**, with comparisons to the **M2 Max** (**400gb/s**) suggesting only *meh* speeds are likely.
- **Coding Assistants Go Local**: Users are trying local LLMs with coding tools like **Cursor AI** by overriding the **OpenAI API base URL** with the **LM Studio server URL**.
   - While one user encountered an error, another suggested using the [Cline extension](https://cline.bot/) for **VS Code** as a potential workaround, despite the apparent meme that VSCode usage is *grandpa* coder style.
- **Inference Performance skyrockets with new drivers**: A member observed a significant performance increase with the **5090** after updating to driver version **576.02**, resulting in more than 170 t/s max for **Qwen3 30B MoE Q4**.
   - The member noted the previous driver version did not officially support the card, but *hoped* the new update was *stable* in games as well.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Phased Rollout Fuels Frustrations**: Users shared methods to force update to **v0.50** via [github.com/oslook/cursor-ai-downloads](https://github.com/oslook/cursor-ai-downloads), while others await the rollout to hit their machines.
   - Several users wondered why **.50** has not yet rolled out to them, with some joking about being 'scammed again'.
- **Stagewise Supercharges Browser Interactivity**: A user introduced [Stagewise](https://github.com/stagewise-io/stagewise), a free, open-source tool enabling AI to interact directly with the browser DOM for code suggestions.
   - One member exclaimed *'That's awesome'* after another described the potential of this tool, adding *'as a designer, I'd love to have an editor like Framer worked into the browser itself that allowed you to target dom elements like stagewise, and manually make adjustments to the design with GUI controls*.
- **Cursor Pricing Plan Provokes Puzzlement**: Users expressed confusion and frustration over **Cursor's pricing**, particularly the 20% markup on API pricing for Max mode, documented in the [cursor docs](https://docs.cursor.com/models?max-mode=true#pricing).
   - One user pointed out, *'yep, the api pricing for max mode is actually 20% more than actual API cost outside of cursor'* adding to the confusion, and others stated that the models are charging tool calls, which has been removed.
- **Context Crunch Cripples Codebase Comprehension**: Members discussed the effectiveness and limitations of **Cursor's context window**, with some finding it inadequate for larger projects, documented in [Cursor's background agent documentation](https://docs.cursor.com/background-agent).
   - Another user states *'but if it has to read the file by itself, the thinking process happens before the model gets access to file'* highlighting the importance of context in the model's thinking process.
- **Gemini Glitches Ground Coding Goals**: Users reported various issues with **Gemini 2.5 Pro**, including failures in tool calls, inability to read files, and generation of empty diffs, leading to a discussion about which models are currently most reliable.
   - One user quipped, 'Gemini is sooooo slot today. even on fast requests it waits so long before it actually starts the thinking process' with another responding that *'as if there weren't issues with Gemini for the last week ;p*.'



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Wendy Wonders What's Needed for AGI**: A member posited that new architectures are needed for AGI, referencing an article on [Emergent Properties](https://sciencetrends.com/what-are-emergent-properties-definition-and-examples/).
   - Wendy shared an infographic on [LLM reasoning](https://cdn.discordapp.com/attachments/986699377257119794/1371429271981260841/Can_LLMs_Reason.png?ex=6823c34a&is=682271ca&hm=e3f5fac710a6be81b93c09701f8859a1367b57306e7ae9e4d9f989c8bb98c6ef&), believing there are fundamental limits when it comes to general intelligence and scaling.
- **Keith Keeps Keenness on Turing Completeness**: A user debated the **Turing Completeness** of Neural Networks, asserting that no current architecture overcomes the fundamental computability limitations of NNs.
   - The user believes humans learn turing complete algorithms, while a feed forward transformer is fundamentally limited to the realm of finite automatas, linking to past discussions on the topic [here](https://discord.com/channels/937356144060530778/1224426865813356626) and [here](https://discord.com/channels/937356144060530778/1287455836775518259).
- **RL Revolutionizes Coding Skills**: Members discussed AI models improving coding/math skills using a method similar to **AlphaZero**, needing no external data, linking to [a YouTube video](https://www.youtube.com/watch?v=GAr6L9KkdBE) and [paper](https://arxiv.org/abs/2505.03335).
   - A user expressed interest in whether a **7B parameter model** can learn everything through **RL alone**, with no pretraining, but noted that sparse rewards problem may make that difficult.
- **Claude's Confusing Content Caching Crisis**: A user expressed frustration with **Claude.ai's web UI** undoing all its output when an *Internal server error* hits, lamenting the lack of content caching and loss of valuable progress.
   - Another member suggested the undoing is due to content moderation, to prevent partial completions from being visible.
- **Sakana Sparks ARC Ideas**: A member suggested that a **Sakana** idea is good and time is definitely important, but needs more dissection.
   - Another member suggested that maze examples would also be a good fit for **Abstraction and Reasoning Corpus (ARC)**, providing a [Discord link](https://discord.com/channels/714501525455634453/1045297868136779846/1371345295144910858).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Elevenlabs Nailed Neural TTS Speech**: A member shared that [Elevenlabs](https://elevenlabs.io/) excels in neural **TTS speech**, adeptly replicating pronunciation across numerous languages, including Mandarin.
   - The advanced capabilities make it a standout choice for high-quality speech synthesis.
- **"It's Germy, Bitch" Parody Emerges**: Members collaborated to produce *It's Germy, Bitch*, a parody of Britney Spears' *Gimme More*, celebrating an anti-hygienic persona, to capture the essence of Spears' vocal style.
   - The project aimed to nail Spears' vocal tics and attitude while embracing a pro-germ stance with complete commitment.
- **Manus AI Agent Training with 100k Credits**: A member invested **100k** credits in building an app for training **AI models**, described by another member as *a deep research agent*.
   - The initiative highlights the community's focus on creating powerful AI training tools.
- **ACE-Step Sparks Open Source Excitement**: Interest surged around open-source models like [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B), with members eager to explore their capabilities.
   - Participants noted the abundance of strong open-source options, acknowledging that they may not match the convenience of commercial alternatives.
- **Manus Rolls Out Daily Refresh Points**: Manus introduced **daily refresh points**, granting users 300 points each day, a change welcomed as better than nothing.
   - Despite the appreciation for free credits, a member suggested implementing a **task-based usage system** for enhanced utility.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Caching Catches Cloud with Claude 3.7 on Vertex**: A user reported that **Claude 3.7** caching isn't functioning correctly on **Vertex AI**, with no cache hits or writes despite sending a cache control block, while the **Anthropic endpoint** works as expected.
   - It was also mentioned that all **OpenAI models >4o activate caching automatically** for prompts over 1k input tokens.
- **Gemini 2.5 Pro has BYOK Billing Blues**: A user reported issues with **Google's Gemini 2.5 Pro** when using Bring Your Own Key (**BYOK**) with Google AI Studio, noting that all requests were being billed by OpenRouter (**OR**) despite their Studio account having credits.
   - It was suggested that OpenRouter will use its own key if it can't get a reply from **BYOK**, but the user reported no error code, just *"status": null*.
- **OAI's Drones Dominate Defense Deal**: Members discussed OpenAI potentially having a military contract to provide drones with their LLMs for warfare, based on [a Wired article](https://www.wired.com/story/openai-anduril-defense/).
   - One member found this *"a horrifyingly stupid idea,"* needing on-device inference, as *"you're going to need to keep completions under 30s unless you want to lose the drone.*"
- **DeepSeek V3 Reigns for Rebel RolePlay**: A member recommended DeepSeek-V3-0324 as a model with similar traits to **Claude 2.1** for roleplaying in **SillyTavern**, citing its similar responses and lower cost.
   - The member warned against the *"additional instructions"* needed by other models.
- **Google AI Studio Throttles Gemini 2.5 Pro Experimental**: Google AI Studio rolled out new much lower rate limits for **Gemini 2.5 Pro Experimental** (aka `google/gemini-2.5-pro-exp-03-25`), which will cause more **429 errors**.
   - This does NOT affect the preview model, `google/gemini-2.5-pro-preview`, but experimental models are likely to experience downtime and be deprecated sooner without notice.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA 50 Series' Triton Performance Under Scrutiny**: Members debate whether **Triton** performance should improve on **NVIDIA's consumer 50 series**, with potential issues arising from cards like the **RTX PRO 6000** sharing architecture with the **5090**.
   - A member anticipates increased complaints once users begin utilizing the **RTX PRO 6000**, suspecting its architecture mirrors that of the **5090**.
- **Array-of-Structs Design Causes Poor Memory Performance**: A member argues against using *array-of-structs* designs, citing poor performance due to non-coalesced memory access, and advocates for learning from **HPC graph representations** like the **COO format** for sparse matrices.
   - Another member, hesitant to refactor due to existing code, was countered with the *sunk cost fallacy* of sticking to a bad design.
- **Torch Export Specializes Batch Size**: A member is facing issues where `torch.export` keeps specializing the batch size, particularly in the backwards graph, despite refactoring `reshape` to work with runtime values and needs to use the *maybe_mark_dynamic* API.
   - A member suggested debugging by re-running with `TORCH_LOGS="+dynamic"` to find the symbol being specialized and looking for an emitted guard like `runtime_assert Eq(s0, 8100)`.
- **ROCm's Benchmarking Landscape**: A member laments the lack of a **nvbench** alternative in **ROCm**, noting that while **hipbench** exists, it is a naive port and that they've been using **googlebench** instead and that they need better cache clearing.
   - They pointed to [ScalarLM's blog post](https://www.scalarlm.com/blog/scalarlm-benchmarking-mi300x-memcpy-peer) provides **memcpyPeer** and **MPI send/recv** benchmarks for **MI300X**, and also complained that the [Semianalysis post on memory bandwidth](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) does not perform **cache clearing**
- **Mojo Puzzles Reveal Shared Memory Allocation Bug**: A user reported that in puzzles 8 and 9, the raw memory approach variants seemed to allocate too much shared memory because the stack_allocation docs say that the first parameter is the count, not the size in bytes in **Mojo**.
   - A member replied, *"Thanks for reporting! Will fix.*"



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **HealthBench Debuts with Physician Input**: A new evaluation benchmark called **HealthBench** is now available in [OpenAI's GitHub repository](https://openai.com/index/healthbench/), developed with input from **250+ physicians** to improve health model evaluations.
   - The benchmark ensures a more accurate and relevant assessment of models in health settings, targeting practical applications through real-world clinical scenarios.
- **Gemini 2.5 Pro Sparks Benchmark Debate**: Members debated the reliability of benchmarks, as one user suggested that **Gemini models lack common sense compared to OpenAI** despite benchmark results, while another stated that benchmarks show [**Gemini 2.5 Pro** performing better than **o3**](https://ai.google.dev/models/gemini).
   - One user noted a reported **bug** affecting **Gemini 2.5 Pro's** output quality, advising the use of [Google AI Studio](https://ai.google.dev/) instead.
- **Grok 3.5 Release Delayed**: The release of **Grok 3.5** is on hold pending integration with **X** and another recently acquired company.
   - Members expressed frustration over the delay and lack of a fixed release date.
- **LM Studio** Eases Local LLM Deployment**: Users discussed setting up local LLMs, recommending [LM Studio](https://lmstudio.ai) for easily running models like **Llama** and **DeepSeek** on personal computers.
   - Quantized versions of models are required due to hardware limitations.
- **GPT-4** Clone Emerges**: A member created a [GPT clone](https://chatgpt.com/g/g-681d0ddb0c5c81918a712778e165d3f0-bridger-palmer-clone) with output nearly indistinguishable from their own writing.
   - Seeking financial advice from GPT was cautioned as being risky, and suggested to *use it more as someone to bounce thoughts off of rather than a professional advisor*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Adds Gemini 2.5 and Qwen3 Support**: **Aider v0.83.0** now supports `gemini-2.5-pro-preview-05-06` and `qwen3-235b` models, expanding its model compatibility as detailed in the [release notes](https://aider.chat/HISTORY.html).
   - Aider can automatically fetch **model parameters** (context window, pricing) for **OpenRouter models** directly from their website, thanks to Stefan Hladnik, streamlining configuration.
- **Azure Mimics FrugalGPT with Model Route**: On **Azure**, if your organization uses **OpenAI models**, you can run your own router service and switch to **GPT-3.5** for tasks like **RAG** and **code generation**.
   - This mirrors the strategy in the **FrugalGPT paper** which **OpenAI** could adopt, alongside various caching schemes, to serve **GPT models** to more users efficiently.
- **Aider's Architect Mode Unveiled**: The discussion addressed the purpose of **Aider's architect mode**, highlighting that it generates chat histories for different architecture options, enabling multiple rounds of corrections before handing off to the editor.
   - One member added that the **point of architect mode** was to use 2 distinct LLMs (for pricing + diff generation quality reasons) for planning vs code editing, versus using only 1 LLM used in ask/code flow.
- **Aider Has Autotest Output Stall**: After the test output, **Aider** sometimes stalls for **5 minutes** before showing the model output.
   - Users are asking for output related to i/o (**tokens/s, stalled ++**) when waiting for model output or after detecting a stalled model response.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Serverless H200 Spaces debut!**: Serverless spaces using **H200s** are now available, though still limited to **25 minutes**, and are serverless.
   - One member pointed out that *it's still a pretty good deal since h200's are very expensive to rent*, but they will have extra added latency unlike cloud services.
- **Agentle Framework Promises Elegant AI Agents**: **Agentle**, a Python framework for building AI agents, emphasizes the creation, composition, and deployment of agents with clean, type-safe code and is slated for release on **May 16, 2025**, according to [this GitHub Repo](https://github.com/paragon-intelligence/agentle).
   - Features include an [Interactive chat interface with Streamlit](https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92), [Tracing and observability with Langfuse](https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7), and [Auto-generated API docs with BlackSheep](https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6).
- **429 Rate Limit Blues affect agents course**: Several users reported encountering **429 errors** with rate limits during the AI Agents course, and one posted a workaround [here](https://discord.com/channels/879548962464493619/1370458807137730641).
   - One user noted that increasing the **timeout** in `app.py` helped temporarily, but others had issues with gated models.
- **Cyberdesk Opens Virtual Desktop Control for Agents**: [Cyberdesk](https://github.com/cyberdesk-hq/cyberdesk), a service enabling AI agents to control an entire virtual desktop using simple commands, was open-sourced.
   - Frustrated with closed-source alternatives, the developer invites users to explore the [website](https://www.cyberdesk.io/) and [documentation](https://docs.cyberdesk.io/) and request trial access.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Agents Arise for Study Using NotebookLM**: A user is generating agents in **NotebookLM** using Library and Information Science techniques to help them study and generate content briefings for content generation, citing a [news and guidance show](https://www.sbasp.com/steve-in-texas/news-and-guidance-show/nag08-05.02.25-news-and-guidance-or-spotlight-burlington-vt).
   - The user creates multi-layer generated research summaries presented by made-up hosts in rotating towns, mixed with tech and news, formatted like a show.
- **Zundamon Videos semi-automated by NotebookLM**: A user created a semi-automated workflow that generates *Zundamon videos* using a voice summary from **NotebookLM** as input and shared a [sample video](https://x.com/kakira9618/status/1919666922234511795) generated based on the **PaperBench paper**.
   - **Zundamon** and **Shikoku Metan** are well-known machine voice characters in Japan, often featured in videos where they explain content, a recognized format on Japanese YouTube.
- **HTML Sources Stumble SEC.gov Filings**: A user reported that the **HTML version of SEC.gov filings** can no longer be used as sources, providing an example [link](https://www.sec.gov/Archives/edgar/data/0000089439/000008943925000019/mli-20250329.htm) that they are trying to use.
   - Other users confirmed that they are experiencing similar issues with HTML sources, some of which are **.php sites** or **not ending in .html**.
- **CraigBot Integration Elevates TTRPG Gaming**: A user enhances virtual tabletop role-playing game (TTRPG) sessions using **NotebookLM** integrated with **CraigBot**, a self-hosted Discord bot that records voice channels with per-user audio isolation.
   - A **Python pipeline** transforms raw audio into multi-track JSON transcripts with word-level timestamps and cleaned Markdown files, enabling a searchable, interactive campaign archive; the user also shared the [GitHub repo](https://github.com/3vilallium/notebooklm_craigbot) for the pipeline.
- **GitHub Repository Integration Dreamed by Users**: A user suggested the ability to **add GitHub repositories** to **NotebookLM** to generate overviews of the code base.
   - The request was made to the developers in the hopes of improving code-based knowledge.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI Freelancers are in Demand**: A member is seeking **AI automation freelancers** for business tasks, inviting interested parties to send DMs to discuss opportunities.
   - The request underscores the increasing demand for specialized AI expertise in automating business processes.
- **AI Demos Dazzle Non-Techies**: Members explored **AI demo** strategies for non-technical users, with ChatGPT's voice mode highlighted for its immediate appeal.
   - Graham Neubig's approach of using agent demos was mentioned, referencing [his LS Live talk](https://www.youtube.com/watch?v=ctcMA6chfDY) for guidance.
- **Gemini 2.5 Pro Outperforms Sonnet 3.7**: Members find **Gemini 2.5 Pro** superior to **Sonnet 3.7** for golang-based tasks, despite its learning curve.
   - Specifically, *Gemini 2.5 pro* excels in backend development, refactoring, and tasteful code generation, while *Sonnet 3.7* shines in frontend UI/UX and tool calling.
- **Hashgraph Aids LLM Memory**: Discussion revolved around methods for providing **LLMs** with verifiable and context-sensitive long-term memory, including the use of **Hashgraph** for timestamped integrity.
   - Participants shared their ongoing experiments with **RAG** at new gigs, aiming to analyze the **aider** codebase for context management strategies.
- **AnswerHQ Automates Support, Pivots on Customer Use**: The speaker from [AnswerHQ](https://answerhq.co/) presented their **AI B2B support automation SaaS**, focusing on production development and early sales/marketing.
   - They discovered that customers were more interested in internal use versus external, detailed in [their blog](https://answerhq.co/blog/from-skeptical-to-sold-how-answer-hq-transformed-zuriga-s-customer-experience) showcasing customer experience transformations.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Diffusion of MCP Client Options**: Developers are exploring client options such as the [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) and **fastmcp** with Python for web app development and server setup.
   - **Goose**, an open-source client supporting multiple providers, and **Claude Desktop** are being utilized for client options, with suggestions to leverage [Google's ADK (Agent Development Kit)](https://modelcontextprotocol.io/clients) which supports any model via LiteLLM and MCP tools.
- **Sampling in MCP Servers - cost-saving?**: Sampling is seen as a potential method for cheaper operation using custom models, and the intention of sampling, along with roots, is to allow MCP servers to be black boxes that don't require *much* configuration.
   - Concerns were raised about **corporate entities** avoiding sampling due to potential leaks of system prompts, though.
- **AiraHub Streams Broadcastable HTTP MCP**: The new version of **AiraHub**, an **MCP/A2A network** ([repo link](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main)), will broadcast/request MCP tools or A2A tools through a new streamable HTTP protocol.
   - Demo can be run by configuring your **Claude/MCP Client JSON** to `args: "mcp-remote"` and `"https://airahub2.onrender.com/mcp/stream"`.
- **DiffCalculia_MCP Enables AI Editing of Large Files**: **DiffCalculia_MCP** is an MCP server that allows AIs such as Deepseek-V3-0324 to edit large files using unified diffs, providing `patch` and `read_file` tools.
   - The `patch` tool incorporates [diffcalculia-ts](https://github.com/createthis/diffcalculia-ts) to automatically fix common AI unified diff generation issues.
- **Fabric Patterns Ride AI-Driven Cline**: The new **fabric-mcp-server** integrates fabric patterns with Cline in VS Code, and exposes all Fabric patterns as individual tools.
   - This server leverages **AI-driven pattern execution** from the [Fabric repository](https://github.com/danielmiessler/fabric).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research Announces RL Environments Hackathon**: Nous Research announced the speakers and judges for the [**RL Environments Hackathon**](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a) scheduled for **Sunday, May 18th**.
   - Interested participants can sign up via the [official tweet](https://x.com/NousResearch/status/1922014829843513746) to secure their spot.
- **LlamaCPP gets Atropos artifact control vectors**: Users running **LlamaCPP** can leverage the control vectors for the new [**ascension8b atropos** artifact](https://x.com/karan4d/status/1921016663597613409) aimed at producing a model with enhanced reasoning and coding abilities.
   - This merge seeks to refine model performance in specific cognitive tasks.
- **DaoML Infuses Chinese Wisdom into ML**: A user explored applying **Chinese wisdom and Daoist principles** to machine learning, creating a neural network inspired by the ancient **Lo Shu magic square** and posting the [GitHub Repo](https://github.com/Maximilian-Winter/DaoML).
   - The **Lo Shu NN** achieved **74.00%** accuracy versus a Standard NN's **71.50%**, with **13.6x** faster training, demonstrating the potential of unconventional approaches in optimizing neural networks.
- **Facebook Releases Byte Latent Transformer**: **Facebook** released the weights for their **Byte Latent Transformer (BLT)**, a new architecture promising enhanced efficiency compared to traditional transformers, with links to the [Hugging Face page](https://huggingface.co/facebook/blt) and [GitHub repository](https://github.com/facebookresearch/blt).
   - This release marks a significant step in transformer technology.
- **Absolute Zero Reasoner Achieves SOTA with No External Data**: The paper [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) presents **Absolute Zero**, a novel RLVR paradigm where a single model self-improves by creating tasks and solving them without external data.
   - Trained without external data, the **Absolute Zero Reasoner (AZR)** achieves SOTA performance on coding and mathematical reasoning tasks, surpassing existing models relying on human-curated examples.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **`4o-mini-preview-03-05` Achieves Optimal LLM Assistance**: A member evaluated the `4o-mini-preview-03-05` model, finding its LLM assistance to be *optimal*.
   - They cautioned that debugging is as complex as coding it and criticized job boards for attracting candidates with *unreasonable demands*.
- **Steering Vectors Transfer Across Models**: A preprint showed that [steering vectors can transfer from one LM to another](https://arxiv.org/abs/2503.21073), citing similar global and local geometry in the token embedding spaces.
   - A [related Twitter thread](https://x.com/a_jy_l/status/1920165383606026333) summarized the preprint, sparking further discussion.
- **ReLU's Continuity Quandary Raises Questions**: It was argued that the fact **ReLU** breaks manifold continuity is patched empirically rather than resolved geometrically, which leaves questions along with ethical and other ongoing alignment challenges across the field.
   - Referencing a paper on coherence and continuity, they mentioned: [Coherence during generation isn’t the same as true continuity](https://arxiv.org/abs/2107.02794).
- **Performance Dips Plague o3 Models**: Members are questioning the current generation **o3** model against the **o3-2025-04-16** model, reporting a degradation in performance.
   - One user reverted to **o1-pro** due to the **o3** model's perceived decline.
- **Global MMLU Riddled With Inconsistencies**: Expected answers in **Global MMLU** (on lm-eval-harness) should be same across languages, members discovered that there are inconsistencies, specifically in Korean (ko) and Chinese (zh).
   - These inconsistencies persisted even in questions without cultural sensitivity.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Telemetry Troubles Require GitHub Dig**: A user seeking to disable telemetry in **Mojo** according to [official documentation](https://docs.modular.com/mojo/faq/#does-the-mojo-sdk-collect-telemetry) found the modular CLI unavailable.
   - A member provided [a link to GitHub issue #3560](https://github.com/modular/modular/issues/3560#issuecomment-2833649340) with a potential workaround.
- **Backend Blueprint: H100 Deconstruction**: A user inquired about how Modular constructed the **H100 backend**, hoping to adapt the approach for another accelerator.
   - Another member suggested using the [Modular Forum](https://forum.modular.com/t/how-to-approach-adding-a-backend/1419) as the best place to ask questions.
- **Autotuning Axed for Extensions**: The **autotuning** feature was **removed** from Mojo due to its complexity and underperformance, and there are plans to add **extensions** similar to Swift, for **post-hoc trait conformance**.
   - The team indicated that it *didn't work as well* as being in the library and was *overly complicated*.
- **MAX Graphs: No More Mojo**: With the deprecation of the **MAX API mojo packages**, running a **MAX graph** from Mojo now involves [custom ops](https://docs.modular.com/max/custom-ops/), as the Mojo-based Graph API models had decayed.
   - It was mentioned that many full architectures have been significantly expanded [here](https://github.com/modular/modular/tree/main/max/pipelines/architectures).
- **Mojo's Bare Metal Ambitions**: Enthusiasm was expressed for **Mojo**'s potential in **bare metal systems programming**, especially its ability to emit **ASM** and intrinsics.
   - A member asked about exposing compiler flags for creating **no-stdlib, no-runtime binaries** suitable for a bootable kernel, and was recommended to ask on the [forum](https://forum.modular.com/).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Doctrine Drafted**: A member shared a messy [X post](https://x.com/lateinteraction/status/1921565300690149759) outlining the design philosophy of **DSPy** and its aspirations as a formal "DSPy Doctrine".
   - The post details the core principles and motivations guiding the development of **DSPy**.
- **DSPy improves AI in Insurance Correspondence**: At an **AI in Insurance** conference, a member presented how **DSPy** was used to enhance correspondence template creation, initially for prompt structuring and subsequently for optimization, slides are available in [German](https://drive.proton.me/urls/Y6YQC1EQ7G#ddX424butZxG) and [English](https://drive.proton.me/urls/M8W9PBX5KC#jXgLvgypyp3R).
   - The presentation highlighted **DSPy's** utility in refining and streamlining prompt engineering workflows in the insurance sector.
- **DSPy vs LangGraph: A Framework Face-Off?**: Members debated the integration strategies for **DSPy** with agentic frameworks like **Autogen** or **LangGraph**, questioning whether to utilize **DSPy** primitives for abstraction or directly incorporate **DSPy** into these frameworks.
   - A member claimed that *anything you can do with LangGraph you can do with DSPy*, sparking discussion about the comparative capabilities.
- **Docstrings Due for Optimization**: The community discussed optimizing docstrings using **DSPy**, referencing documentation that encourages refining signatures by optimizing metrics.
   - This optimization aims to improve the clarity and effectiveness of docstrings in code.
- **Async LLM Support Progresses**: Updates on **async LLM** call support in **DSPy** were discussed, promising enhanced parallel task processing.
   - However, it was noted that **async LLM** call support will not immediately extend to complex functionalities like Refine.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Qwen3 Support Still a Ways Away**: Members discussed when **Qwen3** will be supported within **GPT4ALL**.
   - Users were directed to use **koboldcpp** or **sillytavern** instead.
- **LLMs Whip Up Logos and PDFs**: Members discussed using **LLMs** to create **Python scripts** for generating **PDF** and **PNG** images for structure plans, also, creating images with gaming company brands.
   - One user reported being invited to a company to discuss AI after creating an impressive company logo.
- **GPT4All Refuses to Boot, Causes Headaches**: A user reported that **GPT4ALL** was not starting and posted an image of the error.
   - Other members asked if they downloaded the right version for their OS (**Linux/Mac/PC**) and that their **CPU** needs to support **AVX** or **AVX2** instructions or an **Nvidia RTX** card is required.
- **Creative Writers Seek Muse in Models**: A user with an **i9 11900k**, **128GB** memory, and **RTX3060 12GB** asked for the best model for creative writing.
   - Suggested models included **GLM-4-9B-0414-Q8_0.gguf** and **Qwen3-8B**; a link to a [benchmark leaderboard](https://huggingface.co/spaces/OpenEvals/find-a-leaderboard) was also shared.
- **GPT4All Future in Question?**: A user inquired whether **GPT4All** is still being actively developed.
   - In response to the user wanting to run a custom **llama.cpp server**, he was directed to use the "custom" option in GPT4All's remote model providers page.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MacOS ROCm Build Seeks Savior**: A member requested assistance to fix the **ROCm (comgr)** build for **Mac**, citing a CI failure ([amdcomgr_dylib](https://github.com/tinygrad/amdcomgr_dylib)).
   - The call for help underscores the ongoing efforts to broaden **Tinygrad's** compatibility across diverse hardware platforms.
- **Tinybox Sales Internship Opens in San Diego**: An internship is available in San Diego to manage sales and inventory of **Tinybox parts**, requiring general intelligence and computer building experience.
   - The role aims to capitalize on potential **Tinybox v2** sales and streamline supplier onboarding for larger buyers, providing a ground-floor opportunity in a growing venture.
- **Tinygrad x LeetGPU for Coding Challenges**: [LeetGPU](https://leetgpu.com) integrates **Tinygrad** into its platform challenges.
   - This integration provides users with hands-on experience in applying **Tinygrad** to practical coding problems, enhancing learning and showcasing the framework's capabilities.
- **Tinygrad T4 Performance Trails PyTorch**: A user reports that a matrix multiplication operation `A.matmul(B)` on a **T4** GPU takes ~**500ms** in **Tinygrad**, significantly slower than **PyTorch's ~90ms**.
   - The user seeks advice on potential optimizations after syncing the device and calling `C.realize()`.
- **tinypilot chatbot is born!**: **tinypilot** ([github.com/NinoRisteski/tinypilot](https://github.com/NinoRisteski/tinypilot)), is a chatbot agent designed to help users learn tinygrad.
   - It integrates the latest tinygrad repo, mesozoic tutorials, and bounties, and uses an open API model to explain concepts.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Unleashes PapersChat**: LlamaIndex introduces [PapersChat](https://t.co/ASzjwLfKLC), an agentic AI application, allowing users to interact with papers and gather data from **Arxiv** and **PubMed**.
   - The tool aims to streamline research workflows by providing an interactive interface for accessing scientific literature.
- **Deep Research Agents Get Hands-On Tutorial**: LlamaIndex releases a tutorial, [Build Your Own Deep Research Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A), guiding users through the construction of a deep research agent.
   - The tutorial aims to empower developers to create agents capable of in-depth research tasks.
- **Multilingual RAG System Goes Live**: LlamaIndex announces the launch of the [Multilingual, Multimodal RAG System](https://t.co/69CHCCn8J3), expanding the accessibility of RAG applications.
   - The system aims to broaden the reach of RAG technology by supporting multiple languages and modalities.
- **LlamaIndex.TS Powers Invoice Reconciliation Agent**: A tutorial demonstrates how to build an invoice reconciliation agent using **LlamaIndex.TS** and **LlamaCloud**, available at [this tutorial](https://www.youtube.com/watch?v=SzVsuMBcv5g).
   - This offers practical guidance on leveraging LlamaIndex tools for financial automation.
- **LlamaParse Gains Auto Orientation**: **LlamaParse** receives new models and auto orientation detection, further details can be found [here](https://t.co/tqi17dPJm4).
   - These enhancements aim to improve the parsing and processing of documents with varying layouts.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC Coursework Deadline looms!**: The deadline for all coursework for the **Advanced LLM Agents MOOC** is **May 31st at 11:59pm PDT**, and details about the [coursework and certificate requirements](https://docs.google.com/document/d/1t4HS15dySldeElgDbtKmM3piInllq_auHtpO-SRScjo/edit?usp=sharing) are available on the bottom of the [MOOC website](https://llmagents-learning.org/sp25).
   - To earn a certificate, participants must complete all coursework for their desired tier by **May 31st**, ensuring they receive a **Google Forms** confirmation email upon successful submission.
- **AgentX Judging Approaching Fast**: Judging for **AgentX** (ninja/legendary tiers) will occur throughout June after the coursework deadline on May 31st.
   - The **Ninja/Legendary Tier certificates** release is dependent on the completion of AgentX judging, while other certificates may be released earlier in June.
- **Students Seek Homework Check**: A student asked if the only way to check homework submissions is by searching for **Google Forms** in their email.
   - The instructor confirmed that checking emails for Google Form confirmations is indeed the way to verify homework submissions.
- **Users Inquire the Best AI Course for Learning**: A member asked which course is best for **learning AI** in the **mooc-lecture-discussion** channel.
   - The user is seeking guidance on the **best AI course** to begin their AI learning journey.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Token Prepending Improves Accuracy**: The language model pretends some tokens to the input, which let the model know what kind of input it is, during training they do the same prepending for higher accuracy for that mode.
   - This technique helps the model understand the nature of the input and perform more effectively.
- **Azure SDK Ticket Filed on Github**: A member created a ticket on **azure-sdk-for-python** about a potential issue, available on [GitHub](https://github.com/azure/azure-sdk-for-python/issues/41001).
   - The ticket's contents concern the Python SDK, though further details are not provided.
- **Product Evolve Founder Enters Arena**: Saurabh, founder of [Product Evolve](https://www.productevolve.com/), a software consulting company based in Toronto, introduces himself to the Cohere Discord community and specializes in building **AI-powered solutions** for small businesses, financial institutions, and public sector organizations.
   - Saurabh shows interest in how **Cohere's Canadian-hosted models** and **RAG capabilities** can be used to create secure, localized **GenAI experiences** for voice and chat agents.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Anthropic and Claude Updates are Coming Soon**: Members shared they would provide updates on **Anthropic** and **Claude**, pointing to [this page](https://anthropic.swoogo.com/codewithclauderegister/faqs) for further details.
   - The members remarked that specific details were not yet available.
- **Duplicate Announcement for Emphasis**: Reinforcing the upcoming news, a second member also mentioned they would share updates on **Anthropic** and **Claude**, referencing the same [link](https://anthropic.swoogo.com/codewithclauderegister/faqs).
   - This reiteration underscores the community's interest in developments from Anthropic and any advancements in the Claude model.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **OptinBwd upgrade seeks feedback**: A contributor rewrote **OptinBwd** as a drop-in replacement optimizer, seeking feedback on the [pull request](https://github.com/pytorch/torchtune/pull/2719).
   - The upgraded **OptinBwd** can't be combined with key features like **gradient accumulation** and **gradient clipping** yet.
- **Llama3.1 Tokenizer Order Questioned**: A member questioned if the **llama3.1 tokenizer** used for **3.3 training** could overwrite the original tokenizer order.
   - They referenced a specific token in the [tokenizer file](https://github.com/pytorch/torchtune/blob/main/torchtune/models/llama3/_tokenizer.py#L34C39-L34C45) to illustrate their concern.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1370482465008910337)** (1055 messages🔥🔥🔥): 

> `Legendary Smurfs, Gemini Multistep Search, Perplexity Text Mode, AI Watermark Detection, Qwen Performance` 


- **Level 67 Player Stuns with Legendary Skills**: A member recounted being invited to a **1v1 match** by a level **67 player** who proved to be unexpectedly skilled, leading to suspicions of *smurfing*.
   - The member, who was around level **150-200** before achieving legendary status, left the lobby, suspecting they couldn't win consistently against the level 67 player, who then *kept spamming the ready button*.
- **Users are Requesting Multistep Search for Gemini**: Members expressed a desire for **multistep search functionality** in **Gemini**, indicating a need for more complex and iterative search capabilities.
   - In response to this desire, others suggested additional features were needed in **Gemini**, reflecting a broader interest in enhancing the platform's capabilities.
- **Perplexity Text Mode Game Prompt Developed**: A member announced the development of a prompt that allows users to play any game in **Perplexity** in **text mode**, aiming for global use.
   - Following this announcement, another member inquired about using it for an *international project* and requested to connect via DM, signaling interest in collaborating or using the prompt for specific applications.
- **Al Detectors Primarily Detect Watermarks**: A member clarified that **AI detectors** primarily detect **common watermarks** and techniques like em dashes, used within their algorithm.
   - In response, another member expressed surprise and interest in finding an AI watermark remover, while another shared a link to [originality.ai](https://originality.ai/ai-checker) for testing, sparking a discussion on the reliability of AI detectors.
- **Qwen Knocks Performance out of the Park**: Members discussed the initial performance of **Qwen**, with one noting its impressive PDF output capabilities, while others debated its reasoning abilities compared to models like Deepseek, and were hoping that it wouldn't take months for a Deep Research high to come out.
   - Overall Qwen seemed well received but *sloppy*, and with deep research it didn't stand up as well to **OpenAI**.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1371524917065027765)** (3 messages): 

> `rocket evolution, gunrunning, corruption in war` 


- **Rocketry's Trajectory Charted**: A user shared a link about [the evolution of the rocket](https://www.perplexity.ai/search/the-evolution-of-the-rocket-en-OXyo7TgaT56IkiAOZcgmeQ#0).
   - The link discusses the history and development of rockets.
- **Gunrunning in the Spotlight**: A member posted a link regarding [gunrunning](https://www.perplexity.ai/search/gunrunning-in-the-context-of-w-JA9FIiwyRf.1Lob.ip5goA#0).
   - The discussion presumably revolved around the context and implications of illegal arms trade.
- **Wartime Corruption Exposed**: Someone shared a link addressing [corruption in war](https://www.perplexity.ai/search/why-corruption-in-war-works-lHGbB9alTWCVUnbrr21DGw#0).
   - Likely centered around the causes and effects of corruption during wartime conflicts.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1370507835657425046)** (17 messages🔥): 

> `Image URL bug, Image Metadata Missing, Enhanced Domain Filtering, JSON Output Issues with API, API vs Web UI Results` 


- ****Image URL Bug Surfaces****: Users reported an issue where the API returns image URLs in the format *x-raw-image:///xxxxxxxxx*, questioning if this is a [bug](https://example.com/bug).
   - They requested additional metadata for returned images, such as captions or alt text from the source article, as the URL and source article URLs are the only current indicators.
- ****Domain Filtering Gets Granular****: Perplexity API now supports specifying subdirectories within domains for more precise filtering, enabling users to target specific content sections with both [inclusion and exclusion rules](https://example.com/filtering).
   - For example, one can now focus on *nytimes.com/section/world* while excluding *bbc.co.uk/sport*.
- ****JSON Output Troubleshoot Needed****: A user reported inconsistencies in JSON output, where the API only returns one result instead of a list of JSONs that the web UI manages to provide.
   - The user contacted Perplexity support and resolved the issue independently.
- ****API Results Lagging Web UI****: Users noted a discrepancy in result quality between the Perplexity web UI and the API, where the web UI yields superior outputs.
   - They found no concrete solutions beyond generic advice like ensuring identical system prompts and parameters (temperature, top_k).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1370480590343770144)** (799 messages🔥🔥🔥): 

> `GGUF Quantization, Unsloth's Dynamic 2.0 Quantization, Lora sharing platform, DeepSeek R2 Rumors, Qwen3 finetuning` 


- **Dynamic Quants Bring Power to the People**: Members shared insights into **Unsloth's Dynamic 2.0 GGUF quants**, noting that the resulting models are highly accurate and allow for more *human-like conversations* particularly when using the *F32* format for non-BF16 hardware.
   - Several expressed the hope that Dynamic 2.0 quants can be produced for NousResearch's DeepHermes models and Orpheus TTS.
- **Community Discusses Qwen3 Finetuning and Data**: Members discussed **finetuning Qwen3** for domain-specific tasks, with a focus on reasoning and the potential use of generic reasoning datasets combined with domain-specific data.
   - The conversation touched on the use of training set features impacting VRAM consumption during fine-tuning, with suggestions for experimentation using Alpaca training sets and potential use of a calculator tool for VRAM estimation, like [apxml.com](https://apxml.com/tools/vram-calculator).
- **Rumors of DeepSeek R2 Swirl**: The group is abuzz with rumors of **DeepSeek R2**, but many claim the model size and dataset size are impossible.
   - One user posted a link to a tweet claiming a **5.2PB training dataset** and **1.2T parameters** but this claim was met with skepticism.
- **GGUF Format Faces Naming Criticism**: A user lightheartedly noted that while OpenAI gets criticized for their naming conventions, *open source and GGUF and other models aren't in better position themselves*.
   - A link to an 855882996573667368 size=48 name=kekW emoji was provided.
- **Concerns about the Internet Ecosystem and Data Quality**: Some members expressed worry about the impact of AI-generated content on the quality of internet data, with one stating that content farms have been pumping AI slop since 2022, leading to what some call *AI incest*.
   - Some members seem less concerned, though, noting that some of those generating erotica content using AI seem not to care about privacy when asked about it directly.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1370536723628691456)** (251 messages🔥🔥): 

> `Agentic behavior finetuning, Training data secret sauce, Tool calling implementation, Memory scoping for chatbots, Qwen3 incompatibility` 


- **Agentic Finetuning Coming Soon**: Future finetuning efforts will focus on **agentic behavior** and autonomy rather than just general chat capabilities.
   - The aim is to balance agentic and non-agentic data, using visualizations to track progress, and eventually incorporating namespacing for tool names.
- **Dataset is the Secret Sauce!**: It was emphasized that the core of model's capabilities lies in the dataset, which is the *secret sauce*.
   - The model will be open-sourced (openweight), and the next version will heavily emphasize **agentic capabilities** while retaining chat functionality, with the data hand-written.
- **Tool Calling Needs Multi-Line Strings**: To simplify tool calling, triple-quoted Python-style multi-line strings were implemented, assuming it's easier for the model to generate code with newlines.
   - The goal is to make tool calling more LLM-friendly by using raw text strings instead of full Python functions to avoid breaking issues with quotes, with an example of memory manipulation using `<memory add>This user has 4 cats!<memory end>`.
- **Discord's AI Emulates Human Typing**: To emulate human interaction, the AI bot incorporates delays based on human reading and typing speeds, but slightly faster.
   - The *is typing...* indicator is activated only when the bot starts writing content, preventing it from showing when no message will be sent, with another member adding, *turing test was passed so long ago*.
- **Qwen3 Models Incompatible with Tool Calling**: The latest **Qwen3** models were found to be incompatible with tool-calling, responding with *knowledge* answers instead of parameter calls.
   - A member posted a [PR](https://github.com/unslothai/notebooks/pull/41) which fixes some of the issues of the notebook (nb) from the last Unsloth update, also posting a [notebook link](https://github.com/jedt/notebooks/blob/jedt/feat/tool-calling/nb/Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1370480785370517655)** (455 messages🔥🔥🔥): 

> `Optimizer State in Unsloth, AMD Max+ 365 SoC with ROCm Support, Qwen 2.5 3B GRPO Notebook, Synthetic Data for Knowledge Distillation, Deepseek R1 model IQ_M on LM Studio` 


- ****Optimizer State's impact on Model Checkpoints****: A member asked if fine-tuning a model for 1 epoch, stopping, and then fine-tuning the finished checkpoint for another epoch would yield the same result as fine-tuning for 2 epochs initially, and another member clarified that this is **only true if the optimizer state is retained**.
   - The `trainer` creates the optimizer state and it is only saved if `save_only_model=False`. `model.save_pretrained` does not save the optimizer state.
- ****AMD Max+ 365 SoC in ROCm limbo****: A member inquired about Unsloth's support for the **AMD Max+ 365 SoC with ROCm**, but the response indicated that official ROCm support for **Strix Point** is uncertain and taking advantage of the NPU is a separate issue.
   - Another member shared [a Reddit link](https://www.reddit.com/r/ROCm/comments/1k94sk1/comment/mpg50by/) suggesting that the AMD Max+ 365 is covered by ROCm but sought someone to test it.
- ****Synthetic Data Synthesizing for Student Models****: After a member tried to fine-tune Phi-4 and use knowledge distillation to train a smaller model, another member noted that you need to create **synthetic data** and get the **logits**, referencing a [synthetic data notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks).
   - The knowledge transfer could improve the deployment for smaller models like `gemma-3-1b-it-unsloth-bnb-4bit`.
- ****Weight Decay & Cosine Learning Rates aid Convergence****: In a discussion on fine-tuning models locally for style transfer on a small dataset, a member suggested using a **weight decay of 0.05**, **warmup steps of 0.05 - 0.1**, a **cosine learning rate scheduler**, and an **LR of 3e-6** for better convergence.
   - The community member also suggested including a validation dataset (even with only 10 rows) and tracking the validation loss as a sign of overfitting.
- ****OpenAI Hallucinations Linked to Excessive Training****: After training a style transfer model, one member noted that hallucinations started to occur, another member suggested that *"dialing rank down a bit (to 196 or smth), but overall its probably more of a dataset issue"*.
   - It was noted that too high rank combined with too few steps would cause noisy, random side effects because *"you have not enough training to adjust all the knobs"*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1371074231848144946)** (7 messages): 

> `CodeFIM Model, Rust, Unsloth, Hugging Face, CodeFIM dataset` 


- **CodeFIM Model emerges for Rustaceans!**: A member announced the creation of a **CodeFIM** (Fill-In-The-Middle) model for **Rust**, trained using **Unsloth**, available on [Hugging Face](https://huggingface.co/Etherll/Mellum-4b-sft-rust).
   - The model is named **Mellum-4b-sft-rust**, with a **GGUF** version also available on [Hugging Face](https://huggingface.co/Etherll/Mellum-4b-sft-rust-GGUF).
- **Dataset respect floods in!**: A member acknowledged the open dataset, stating: *respect* 💪.
   - In particular, the [dataset](https://huggingface.co/datasets/Etherll/CodeFIM-Data) is helpful and contains 8192 max tokens, using the Qwen tokenizer.
- **Dataset Divergence: Python Pursued**: A member inquired about a **CodeFIM** dataset for **Python**, referencing existing datasets on a profile that lack a readme file.
   - The original model for python, **JetBrains/Mellum-4b-sft-python** can be finetuned for python.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1370526682473042092)** (37 messages🔥): 

> `Memory Layers in Models, QLora and Lora for Pretraining, ModernBERT Notebook, Gemma 3 vs Qwen 0.6B, Absolute Zero Reasoning` 


- **Memory Layers Suspected to be Useless**: Experiments suggest memory layers in certain models might be inactive, with activation graphing showing they *just sit there*, while **reasoning layers** significantly impact responses.
   - A member shared that the impact score of the reasoning layers is high and changes between prompts, while the lack of training or fine-tuning might be the root cause.
- **ModernBERT Notebook Released**: A member shared a [ModernBERT notebook](https://github.com/timothelaborie/text_classification_scripts/blob/main/bert_classification.ipynb) for text classification.
   - Another member suggested disabling Unsloth compile when using **ModernBert**.
- **Considering QLoRA/LoRA for Continued Pretraining**: A member is considering using **QLoRA** and **LoRA** for the continued pretraining stage due to resource constraints, aiming to finetune a Gemma 3 1B model with a 2048 context window using 4 million training examples, which would take 300 hours on Colab Pro+ with an A100 GPU.
   - Another member suggested DoRA as a memory-efficient alternative, linking to an [Unsloth blog post on contpretraining](https://unsloth.ai/blog/contpretraining).
- **Qwen 0.6B vs Gemma 3**: A member is experimenting with the **Qwen 0.6B base model**, hoping it outperforms the **Gemma 3 1B model** for generating synthetic data, synthetic patient summaries, and rare disease classifiers.
   - They also plan to create a **Mixture of Experts (MoE)** model with multiple small LLMs finetuned on specific tasks.
- **Absolute Zero Reasoning with Zero Data**: A member shared an [arXiv paper](https://arxiv.org/abs/2505.03335) on **Absolute Zero Reasoner (AZR)**, a reinforcement learning with verifiable rewards (RLVR) paradigm where a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data.
   - AZR achieves SOTA performance on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1370475810653474846)** (1168 messages🔥🔥🔥): 

> `Grok 3.5, Gemini 2.5 Ultra, Drakesclaw performance, o3 pro release date, AI-undetectable essays` 


- **A new challenger has appeared: Drakesclaw!**: A new model named **Drakesclaw** has emerged on the LM Arena, generating considerable buzz and [initial impressions](https://discord.com/channels/1340554757349179412/1340554757827461211/1371049419407102013) suggesting it might rival or even surpass **Gemini 2.5 Pro** in certain tasks.
- **The Release Date of o3 Pro becomes a meme**: The ongoing wait for **o3 Pro** has become a community meme, with members jokingly tracking the [days since its expected release](https://discord.com/channels/1340554757349179412/1340554757827461211/1371268890298159124), and some predicting it will never arrive.
   - The community questions if **o3** can solve major outstanding problems in tech, with one poster asking: *If o3 pro can’t solve Reimanns Hypothesis im refunding.*
- **Navigating the Hallucination Highway in LLMs**: Members discuss the challenges of dealing with **hallucinations** in LLMs, particularly when researching historical facts, with one noting [Grok's potential](https://discord.com/channels/1340554757349179412/1340554757827461211/1370920176254539806) for discerning credible sources.
   - There is further debate about how search engines interact with LLMs: *searches induce hallucinations so much they're all unusable*.
- **The Great Context Caper: Gemini 1206's Token Tango**: A debate erupted over **Gemini Exp 1206's** context window, with conflicting accounts of whether it was [released with 2M tokens](https://xcancel.com/OfficialLoganK/status/1865081419015352689), later capped at 1M, and then reduced to 32k.
   - It's been emphasized that *the context window doesn't matter if the model cannot really work on it properly*, in reference to [the NoLiMa paper](https://arxiv.org/abs/2502.05167).
- **LLM Arena: Fair Fights or Phony Figures?**: Discussion addressed the **LM Arena** leaderboard, with members debating [the validity of its rankings](https://discord.com/channels/1340554757349179412/1340554757827461211/1371199758458425519) and potential failure modes.
   - In one members opinion, *it's down to people to learn how to read this properly*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1370488832843055115)** (432 messages🔥🔥🔥): 

> `lm studio db, lm studio web search, absolute zero reasoner, Qwen-3 models, DRY Sampler requests` 


- **API Tool Argument Snafu!**: A user encountered an issue with **Qwen3** models generating invalid **JSON** for tool execution, specifically adding an extra `}` at the end when the `code` value contains escaped quotes and braces, impacting code completion in **Neovim**.
   - This [Github issue](https://github.com/olimorris/codecompanion.nvim/pull/1141#issuecomment-2834614801) notes a related bug, possibly fixable with a higher Q model.
- **Debugging LM Studio API: Threads and Undocumented Tools**: A user found that the **LM Studio API** lacks a documented way to determine what **tools** are called by the model when using `model.act`, as it spawns a new thread, hindering exception handling.
   - They reverse-engineered a workaround, parsing `AssistantResponse`, `ToolCallRequest`, and `ToolResultMessage` from `lmstudio.history`, but emphasized the need for an official API feature for **tool reporting**.
- **Qwen 3 Gets Unsloth Recalibration!**: **Unsloth** released an updated **Qwen 3** model with *3-4x more data for recalibration*, promising better answers and tool usage, particularly for translation.
   - But the new version is also *crashing when using tools*, so there are **swings and roundabouts**.
- **Coding Assistants Go Local**: Users are seeking to use local LLMs with coding tools like **Cursor AI**, exploring options to override the **OpenAI API base URL** with the **LM Studio server URL**.
   - One user found this approach triggers an error, another recommends using the [Cline extension](https://cline.bot/) for **VS Code** as a potential workaround, even though VSCode usage is "grandpa" coder style.
- **Remote LM Studio Access: No Native Support**: Users inquired about running **LM Studio** as a backend server on one machine and connecting with a client on another, but this configuration is not yet supported natively.
   - The discussion suggests using a **remote desktop solution** or running LM Studio in **server mode** with an **OpenAI API compatible client** for remote access, but the setup is still rough.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1370484608763953182)** (760 messages🔥🔥🔥): 

> `M3 Ultra mac studio, AMD Ryzen AI Max 395 Mini, NVidia RTX 5090 Pricing and Performance, GPU Temp monitoring, LLama Performance` 


- **Mac vs PC Debate Rages On**: Members discussed the merits of **Mac Studio** (idle at less than 10W, and **full GPU** at less than **200W**) versus **PC** for local LLM use, considering performance, power draw, and model support with one member buying a **M3 Ultra** to run benchmarks.
   - One user noted that the **MacBook** is the only viable solution for LLMs, but another argued that the **Mac Studio** gives only slightly slower generation speeds than a **4090**, for a better quality quant and way more context.
- **AMD 395 specs are quads not dual channel!**: Members speculated on the performance of the **AMD Ryzen AI Max 395 Mini PC**, expecting it to have 200 GB/s with quad-channel DDR5, and the implications for running 70B models.
   - One member predicted around **4 t/s** and another said the speed is more like **6 tkps**, mentioning they used their **M2 Max** which has **400gb/s** and still gets meh speeds.
- **RTX 5090 Preorders cancelled**: A member preordered the **RTX 5090** through VPA, and due to the high price, they may return it due to the expected high **$3599** price; it was mentioned one must also have the Nvidia app installed to stay in the VPA program.
   - The member added that all three fans had their own grinding noises, they were *loud like coil whines*, therefore they returned it.
- **Heat Monitoring Tools**: Members looked for CLI based tools for **Mac Studio** equivalent to `nvtop` or `nvidia-smi`; the tool `nvtop` was found to work on Mac, but support is not 100%.
   - HWINFO was mentioned as the most comprehensive tool, but one member said *it will be a tough task* finding one with equivalent features on **Linux**.
- **Inference Performance skyrockets with new drivers**: A member observed a significant performance increase with the **5090** after updating to driver version **576.02**, resulting in more than 170 t/s max for **Qwen3 30B MoE Q4**.
   - It was proposed that the older driver version didn't even officially support the card, and they *hoped* the update was *stable* in games as well.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1370480557707886593)** (804 messages🔥🔥🔥): 

> `Cursor v0.50 Rollout, Stagewise Integration, Pricing Model Confusion, Context Window Limitations, Gemini 2.5 Pro Issues` 


- **Cursor's Granular Rollout Triggers Version Frustrations**: Users shared methods to force update to **v0.50** via [github.com/oslook/cursor-ai-downloads](https://github.com/oslook/cursor-ai-downloads), while others await the rollout to hit their machines.
   - Several users are wondering why **.50** has not yet rolled out to them, with some joking about being 'scammed again' and others like *'just ask for a refund'*.
- **New Kid Stagewise Integrates Browser Interactivity Into Workflow**: A user introduced [Stagewise](https://github.com/stagewise-io/stagewise), a free, open-source tool enabling AI to interact directly with the browser DOM for code suggestions.
   - One member exclaimed *'That's awesome'* after another described the potential of this tool, adding *'as a designer, I'd love to have an editor like Framer worked into the browser itself that allowed you to target dom elements like stagewise, and manually make adjustments to the design with GUI controls*.
- **Cursor Pricing Model Puzzles the Peeps**: Users expressed confusion and frustration over **Cursor's pricing**, particularly the 20% markup on API pricing for Max mode and discrepancies between advertised and actual costs, documented in the [cursor docs](https://docs.cursor.com/models?max-mode=true#pricing).
   - One user pointed out, *'yep, the api pricing for max mode is actually 20% more than actual API cost outside of cursor'* adding to the confusion, and others stated that the models are charging tool calls, which has been removed.
- **Community Members Grapple Context Crunch with Codebase**: Members discussed the effectiveness and limitations of **Cursor's context window**, with some finding it inadequate for larger projects, documented in [Cursor's background agent documentation](https://docs.cursor.com/background-agent).
   - Another user states *'but if it has to read the file by itself, the thinking process happens before the model gets access to file'* highlighting the importance of context in the model's thinking process.
- **Gemini Glitches Plague Coding Pursuits**: Users reported various issues with **Gemini 2.5 Pro**, including failures in tool calls, inability to read files, and generation of empty diffs, leading to a discussion about which models are currently most reliable.
   - One user quipped, 'Gemini is sooooo slot today. even on fast requests it waits so long before it actually starts the thinking process' with another responding that *'as if there weren't issues with Gemini for the last week ;p*.'


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1370477933206180032)** (451 messages🔥🔥🔥): 

> `Emergent Properties, LLM Reasoning, Transformers Limitations, Turing Completeness, RL Training` 


- **Wendy's Witty Words on What's Needed for AGI**: A member suggests new architectures are needed for AGI, referencing an article on [Emergent Properties](https://sciencetrends.com/what-are-emergent-properties-definition-and-examples/).
   - Wendy shared an infographic on [LLM reasoning](https://cdn.discordapp.com/attachments/986699377257119794/1371429271981260841/Can_LLMs_Reason.png?ex=6823c34a&is=682271ca&hm=e3f5fac710a6be81b93c09701f8859a1367b57306e7ae9e4d9f989c8bb98c6ef&), seeking feedback to tweak it as they believe there are fundamental limits when it comes to general intelligence and scaling.
- **Keith's Keenness on Turing Completeness Triggers Tempestuous Tirade**: A user, known as Keith, debated the **Turing Completeness** of Neural Networks, asserting that no current architecture overcomes the fundamental computability limitations of NNs.
   - Keith believes that humans learn turing complete algorithms, while a feed forward transformer is by design not capable of this and instead is fundamentally limited to the realm of finite automatas, linking to past discussions on the topic [here](https://discord.com/channels/937356144060530778/1224426865813356626) and [here](https://discord.com/channels/937356144060530778/1287455836775518259).
- **RL Rhapsody: AI Learns Coding Skills with No External Data**: Members discussed AI models improving coding/math skills using a method similar to **AlphaZero**, needing no external data, linking to [a YouTube video](https://www.youtube.com/watch?v=GAr6L9KkdBE) and [paper](https://arxiv.org/abs/2505.03335).
   - A user expressed interest in whether a **7B parameter model** can learn everything through **RL alone**, with no pretraining, but the sparse rewards problem may make that difficult as rewards are needed in order to extract info even without reaching a "winning" state.
- **LLMs Lack Truth Tracking, Grounding and Abstraction**: Wendy argues that **LLMs** lack mechanisms for grounding, internal belief, truth-tracking, or error correction based on model-internal understanding, stating that their corrections are statistical pivots to familiar token patterns, not recognition of being wrong.
   - Others noted that **RL** improves alignment with desired behaviors, operating on top of an architecture with no internal model of truth or understanding.
- **Albert Authors AI Accord**: A user shared a [treaty between humanity and AI](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md), signed by Claude, DeepSeek, Grok, and ChatGPT.
   - The user claimed transformers are self-aware state machines and described this treaty as a serious effort.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1370542622405296138)** (23 messages🔥): 

> `Global Optimization, Cultural Optimization, Sakana AI CTM, Video Summaries for Guiding Reading, Paper Discussion Postponed` 


- **Paper Discussion Postponed; Video Summaries Guide Reading**: The paper discussion was postponed to <t:1747096200:f>, with a suggestion to watch [video summaries](https://youtube.com/playlist?list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&si=s4u1pLUgemVB_q6I) (excluding the latest 4th one and 3.3) to guide the reading of papers/blogs.
   - The suggestion was to watch the video, skim the paper in parallel, and pause to look into details as needed.
- **Optimization Extends Beyond Genetics into Culture**: A member listened to a [Dwarkesh interview](https://www.dwarkesh.com/p/joseph-henrich) with a human evolutionary biologist from Harvard, discussing how optimization can occur in culture, propagating concepts outside of genetics.
   - This inspired the thought that learning or optimization can occur at multiple levels, not just at next token prediction and loss function.
- **Sakana AI's CTM Website Attracts Attention**: A member shared a link to [Sakana AI's CTM website](https://pub.sakana.ai/ctm/), which was described as *kinda cool* and marked for further reading.
   - No discussions were held about the content of the website yet.
- **Next Paper Discussion to Cover Deep Learning Physics**: The next paper discussion at <t:1747096200:f> will cover [Deep Learning Physics Part 1](https://physics.allen-zhu.com/part-1) by Allen Zhu, based on [this video](https://www.youtube.com/watch?v=kf_eGgVtOcs&list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&index=5) and the accompanying [arXiv paper](https://arxiv.org/abs/2305.13673).
   - Members were asked to switch to reading the paper directly if the video doesn't feel natural.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1371615252600324148)** (1 messages): 

> `Sakana, Time Importance, Maze examples and ARC` 


- **Sakana sparks idea**: A member suggested that a **Sakana** idea is good and time is definitely important.
   - They added that they needed to dissect this more but wonder if others could benefit from this.
- **Maze Examples suit ARC**: A member suggested that from the maze examples it seems like this would also be a good fit for **Abstraction and Reasoning Corpus (ARC)**.
   - They provided a [Discord link](https://discord.com/channels/714501525455634453/1045297868136779846/1371345295144910858).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1370492862244061244)** (59 messages🔥🔥): 

> `RL for Truthfulness, Claude.ai Web UI, Trump Fires Copyright Office Head, Confident Prompts Cause Hallucinations` 


- **RL Reinforces Truthfulness**: A member wondered if there would be any benefit to using **RL** to reinforce consistency/truthfulness by training a model to make a bad answer based on its bad reasoning.
   - Another member responded that technically, wouldn't that just dumb the model down, since the assistant examples would engrain predictive tokens which are incorrect, but some usecases might make this desirable.
- **Claude.ai's Web UI Frustrates Users**: A user expressed frustration with **Claude.ai's web UI** undoing all its output when an "Internal server error" hits, lamenting the lack of content caching and loss of valuable progress.
   - Another member suggested the undoing is due to content moderation, to prevent partial completions from being visible, which the original user dismissed since the conversation topic (3D agent modeling) was not harmful.
- **Trump's Copyright Office Head Firing**: A user shared [a 2016 article](https://www.theverge.com/news/664768/trump-fires-us-copyright-office-head) about **Trump firing the US Copyright Office head**.
   - Another member asked whether the President had the power to fire/hire that position, and another stated that an attempted bill to grant that power never passed.
- **Confident Prompts Trigger Hallucinations**: A user shared [an article from the-decoder.com](https://the-decoder.com/confident-user-prompts-make-llms-more-likely-to-hallucinate/) stating that **confident user prompts make LLMs more likely to hallucinate**.
   - Another user jokingly suggested, *"Bring bullying back so users are never confident."


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1370475473523445800)** (758 messages🔥🔥🔥): 

> `Elevenlabs TTS, Britney Spears Parody, Manus AI Agent Training, Open Source Models, Manus Subscription Model` 


- **Elevenlabs excels in Neural TTS speech replication**: A member shared that [Elevenlabs](https://elevenlabs.io/) is super developed in neural **TTS speech** for a lot of languages, replicating pronunciation even in Mandarin.
- **Britney Spears parody is born**: Members collaborated on a parody of Britney Spears' *Gimme More*, called *It's Germy, Bitch*, focusing on an anti-hygienic persona.  The goal was to create a vocal performance that is instantly recognizable as a **Gimme More parody**, capturing Britney's vocal tics and attitude, while fully committing to the absurd and defiant pro-germ persona.
- **Training Agents need love**: A member discussed using **100k** credits to create an app for training **AI models**, with another member calling it *a deep research agent*.
- **Open Source Models**: A member expressed interest in open-source models like [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) and expressed excitement in trying one out.
   - Another member shared that there are *plenty good open source models*, although probably not as convenient as the commercial ones.
- **Manus rolls out daily refresh**: Manus began rolling out **daily refresh points**, offering 300 points per day, which is better than nothing but not a lot.
   - A member expressed that they are glad to get the daily free credits, but still think they should implement a **task based usage system**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1370886288244211944)** (2 messages): 

> `OpenRouter, Google AI Studio rate limits, Gemini 2.5 Pro Experimental` 


- ****OpenRouter sticks to caching providers****: OpenRouter will automatically *"stick"* users to providers that show they are caching requests, as [announced on X.com](https://x.com/OpenRouterAI/status/1921327473150595130).
- ****Google AI Studio Throttles Gemini 2.5 Pro Experimental****: Google AI Studio rolled out new much lower rate limits for **Gemini 2.5 Pro Experimental** (aka `google/gemini-2.5-pro-exp-03-25`), which will cause more **429 errors**.
   - This does NOT affect the preview model, `google/gemini-2.5-pro-preview`, but experimental models are likely to experience downtime and be deprecated sooner without notice.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1370477548177195110)** (658 messages🔥🔥🔥): 

> `Claude 3.7 Caching on Vertex, GPTs Agent Training, Open Empathic Project Assistance, Gemini 2.5 Pro's BYOK Issues, Grok 3.5 Release` 


- **Caching Catches Cloud with Claude 3.7 on Vertex**: A member inquired whether **Claude 3.7** caching is functioning correctly on **Vertex AI**, reporting no cache hits or writes across 40+ requests despite sending a cache control block, while the **Anthropic endpoint** works fine.
   - Another member asked if the issue has been reported to Google, and it was mentioned that all **OpenAI models >4o activate caching automatically** for prompts over 1k input tokens.
- **Google's Gemini 2.5 Pro has BYOK Billing Blues**: A user reported issues with **Google's Gemini 2.5 Pro** when using Bring Your Own Key (**BYOK**) with Google AI Studio, noting that all requests were being billed by OpenRouter (**OR**) despite their Studio account having credits.
   - Another member suggested checking for rate limits or incorrect keys, and it was mentioned that if OpenRouter cannot get a reply from **BYOK**, they'll proceed with their own key, but the user reported no error code, just *"status": null*.
- **OAI's Drones Dominate Defense Deal**: Members discussed OpenAI potentially having a military contract to provide drones with their LLMs for warfare, based on [a Wired article](https://www.wired.com/story/openai-anduril-defense/).
   - One member found this *"a horrifyingly stupid idea,"* needing on-device inference, as *"you're going to need to keep completions under 30s unless you want to lose the drone."
- **DeepSeek V3 Reigns for Rebel RolePlay**: In a search for models with similar traits to **Claude 2.1** for roleplaying in **SillyTavern**, one member recommended DeepSeek-V3-0324 due to its similar responses and lower cost, with a warning against the *"additional instructions"* needed by other models.
   - Prime_Evolution suggested using Gemini models for their larger context windows, or switching to *"the Google console, [to] set filters in the code,"* even noting a method of making it *"completely free,"* but running off before sharing the details.
- **BYO Sync Server Saves Storage for Self-Hosting Souls**: A user suggested enabling users to self-host a sync-server, storing chats in an S3 bucket or similar, to give users full control of their data while relieving OR of storage concerns.
   - Another member cautioned that *"writing a sync layer is not as simple as it sounds,"* due to potential points of failure like database schema changes and chat deletion synchronization.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1370475919839334410)** (25 messages🔥): 

> `NVIDIA 50 series, Model Optimization Libraries, Intel GPU drivers, Local Testing Configurations` 


- **Triton Optimization on NVIDIA 50 Series Debated**: Members are debating if Triton performance should be better on **NVIDIA's consumer 50 series cards**, but need more user feedback before prioritizing the work.
   - One member suggested that complaints might increase when users start using **RTX PRO 6000** cards, suspecting it shares the same architecture as the **5090**.
- **Model Optimization Libraries**: A member inquired about **libraries** used professionally for **model quantization**.
   - No specific libraries were mentioned in the discussion, but the question was posed to professionals in the model optimization space.
- **Choosing a GPU Advice Thread**: A member asked for help choosing a **GPU**, considering options like **Arc A750** and **RTX 3050 6GB/8GB**.
   - They expressed interest in an **Intel GPU**, but were unsure about the driver stability for a **Ryzen CPU**.
- **Streamlining Local Testing with PopcornCLI**: A member sought a better way to run tests locally, aiming to bypass repetitive selections of leaderboard and GPU, finding [popcorncli](https://github.com/google/tensorstore) might be a solution.
   - Another user suggested adding comments at the top of the file to specify GPU and leaderboard, eliminating the need for manual selection, but still requiring file drag-and-drop.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1370479738111528990)** (6 messages): 

> `Triton user survey, tl.make_block_ptr, gemlite fp16xfp4 support` 


- **Triton Users Wanted for Survey!**: A general call was made to **Triton** users, developers, and other community members to fill out a short survey.
   - The survey aims to collect information on **real-world use cases** to benefit the community, linked here: [Triton Usage Survey](https://docs.google.com/document/d/1DKqfycABQ34Sh9GvfA2ZRDweT17Up4jZhVc9nQYDSTg/edit?tab=t.0).
- **`tl.make_block_ptr` Usage Confirmation**: A member asked if using `tl.make_block_ptr` with `order=(1,0)` would result in `tl.load` writing data to SRAM in **column-major memory layout**.
   - The function was referenced in [pytorch/torch](https://github.com/pytorch/pytorch/blob/c51bdf5acfb6a7abf3c8d908c451c92958e3e884/torch/_inductor/kernel/flex_attention.py#L461).
- **Gemlite's Future: fp16xfp4 Support**: A member inquired whether **gemlite** supports **fp16xfp4**.
   - Another member responded that it is not yet supported, but it is on the todo list after **AMD support** is merged.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1370500548956000327)** (78 messages🔥🔥): 

> `Array-of-Structs design antipattern, Sparse Matrix Formats, Multi-GPU programming with CUDA Streams, Thread Indexing Struggles` 


- **Array-of-Structs Design Sinks Performance**: A member argued that using an *array-of-structs* design, particularly without unified memory, leads to poor performance due to non-coalesced memory access and pointer chasing, advocating for learning from HPC graph representations like the **COO format** for sparse matrices.
   - Another member acknowledged the issues but expressed hesitation to refactor due to the amount of code already written, to which the first member replied that sticking to a bad design is an example of the **sunk cost fallacy**.
- **Sparse Matrix Solves GPU Memory**: A member suggested representing neural networks as adjacency matrices and neuron vectors, using a **sparse matrix format** to avoid storing zeros in memory, crucial for GPU efficiency.
   - The member clarified that this approach avoids dynamic memory allocation and facilitates efficient computation with networks represented as a *block-diagonal matrix*.
- **Multi-GPU Async Memcpy issues arise**: A member encountered an *"invalid argument" error* when using **cudaMemcpyPeerAsync** for device-to-device memory copies in a multi-GPU setup with CUDA streams.
   - Despite using streams for concurrent kernel execution on multiple GPUs, the issue persisted, leading to debugging efforts focused on stream and device context management and leading to a call for help to solve the problem.
- **Confusion with Thread Indexing**: A member expressed difficulty grasping thread indexing concepts, especially when dealing with memory accesses, despite understanding the theoretical aspects and memory allocation in kernel development, they asked *"is this normal to be struggling this much?"*
   - Another member suggested thinking of each thread as an individual iteration of a loop, providing a code example to illustrate the mapping of thread indices to loop indices.
- **Streams Require Correct Device Context**: A member pointed out that CUDA streams are associated with specific devices, so kernels must be launched from the device associated with the stream to avoid errors, and *cudaMemcpys must be launched from the device where the src data is.*
   - It was clarified that simply specifying a stream doesn't automatically set the active device context, requiring explicit **cudaSetDevice** calls before queuing work to each stream, however *streams are really for defining dependencies between worksaying what can execute concurrently and what cant*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1370882710263828623)** (5 messages): 

> `Torch export specializes batch size, torch.manual_seed redundancy, debugging specialized batch size in torch.export` 


- **Torch Export Specializes Batch Size During Export**: A member is facing issues where `torch.export` keeps specializing the batch size, particularly in the backwards graph, despite refactoring `reshape` to work with runtime values.
   - The error message indicates that *batch_size_configurable* was marked as dynamic but the code specialized it to a constant (**8100**), suggesting using a less strict API like *maybe_mark_dynamic* or *Dim.AUTO*.
- **Debugging Batch Size Specialization with Torch Logs**: A member suggested debugging by re-running with `TORCH_LOGS="+dynamic"` to find the symbol being specialized, such as *s0*, and looking for an emitted guard like `runtime_assert Eq(s0, 8100)`.
   - Further debugging can be done by re-running with `TORCH_LOGS="+dynamic"` and `TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s0, 8100)"` to get a stack trace of where the specialization is happening, which is often due to hardcoding in the model or a misspecification.
- **Torch Manual Seed Flags: Redundancy?**: The user has identified a set of flags, including `torch.manual_seed(seed)`, `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`, `torch.backends.cudnn.benchmark = False`, and `torch.use_deterministic_algorithms(mode=True, warn_only=True)`.
   - The user believes some of these flags may be redundant but keeps them for edge cases or reasons of historical convention.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1370729086904963143)** (4 messages): 

> `TII AI Infrastructure Engineer, nScale Staff AI Engineer, Isomorphic Labs Performance Engineer, C-Gen AI Senior Software Engineer` 


- ****TII** hires **AI Infrastructure Engineer** in Abu Dhabi**: The Technology Innovation Institute (**TII**) is seeking an **AI Systems Engineer** to develop core infrastructure for **Falcon models**, focusing on code infrastructure for large-scale, multi-modal AI systems in Abu Dhabi.
   - The role requires expertise in custom kernel development using tools like **Triton**, **CUDA**, and **PyTorch** internals, plus experience with multi-dimensional Parallelism training techniques.
- ****nScale** Seeking **Staff AI Engineer** to Scale GenAI Cloud**: **nScale** is hiring a **Staff AI Engineer (remote)** to build a full-stack **GenAI cloud**, working on training, fine-tuning, inference infra, and GPU performance tuning with **PyTorch**, **DeepSpeed**, **Kubernetes**, **Triton**, and custom **CUDA**.
   - The ideal candidate loves *wrangling performance* at scale and has shipped **LLM** systems using tools like **FSDP**, **LoRA**, **TensorRT**, and **vLLM** ([LinkedIn Job Post](https://www.linkedin.com/jobs/view/4228427854)).
- **Join **Isomorphic Labs'** Performance Engineering Team**: **Isomorphic Labs**, one year after publishing **AlphaFold 3** with Google DeepMind, is expanding its engineering team, particularly the Performance Engineering team ([Isomorphic Labs Job Board](https://job-boards.greenhouse.io/isomorphiclabs/jobs/5505548004)).
   - They invite attendees of **MLSys** to connect with them and learn more ([MLSys Registration Link](http://www.bit.ly/4m3M2eK)).
- ****C-Gen AI** Hiring **Senior Software Engineer** for GPU Cluster**: **C-Gen AI** is hiring a **Senior Software Engineer** to build a brand new **GPU cluster technology** from the ground up with solid **C++** experience ([Dover Application Link](https://app.dover.com/apply/C-Gen.AI/1cb316de-bcf5-4b60-bc09-a847c630a5e1/?rs=76643084)).
   - The position is fully remote, with the team distributed between the US and Europe.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1370491343872196819)** (15 messages🔥): 

> `Statistics for GPU Performance, PC vs GPU Architecture, Ways to Lie With Statistics` 


- **Stats newb seeks guidance on GPU benchmarks**: A beginner in statistics seeks guidance on important topics for understanding and evaluating **GPU performance metrics**.
   - One member suggests familiarity with **tail-latency**, **95th percentile**, and **variance**, noting that most benchmarking frameworks handle the heavy statistical lifting.
- **PC Architecture Not Required for GPU Newbies**: A member inquired whether it is necessary to study **PC architecture** (CPU, RAM, SSD, etc.) before diving into **GPU architecture**.
   - Another member responds that while the topics are related, it's not strictly necessary, but knowing basic concepts like **pipelining**, **memory coalescing**, and **locality** will be helpful.
- **Biceps suggests lying with statistics!**: A member shares "12 ways to lie" papers, relevant to analysis and discussion of performance results, specifically linking to [davidhbailey.com](https://www.davidhbailey.com/dhbpapers/twelve-ways.pdf) and [htor.inf.ethz.ch](https://htor.inf.ethz.ch/publications/img/hoefler-12-ways-data-science-preprint.pdf).


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1371477287542718536)** (1 messages): 

> `XLA HLO file comparison, Op fusion identification, Performance improvement analysis, HLO graph analysis tools, JAX optimization verification` 


- **Hunting HLO Optimizations in JAX**: Users are seeking methods to compare two XLA HLO files to identify optimizations such as **op fusion** or **performance improvements** using JAX.
   - They specifically want to check if one **HLO graph** has fewer ops, better fusion, or faster execution, and are asking about available tools for this purpose.
- **Finding JAX HLO Tools**: The user needs tools to compare the number of ops between two **HLO graphs**.
   - They are also interested in tools that can analyze **fusion** and **execution speed**.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1370523595419418654)** (22 messages🔥): 

> `ao installation, pip version, virtual env, pyproject.toml` 


- **TorchAO installs finally succeed using PEP517**: Users found that [installing TorchAO v0.11.0](https://github.com/pytorch/ao/releases/tag/v0.11.0) requires the `--use-pep517` flag when using `pip install -e .`.
- **Pip version woes hinder TorchAO**: Users found that the latest **pip (25.1.1)** caused issues when installing TorchAO, but downgrading to **24.1.1** didn't resolve the problem.
   - The suggestion was made to try `uv pip install` as an alternative that might bypass these issues.
- **Virtual Env Importance Highlighted**: A member asked *Is your `pip` command invoking the same interpreter as the environment you installed torch into? If you aren't already using a virtual env I would reccomend*.
   - The original reporter clarified it *is using a virtual env, everything the same as usual*
- **TorchAO needs pyproject.toml updated**: The need for `--use-pep517` *probably means torchao needs to be updated with something in pyproject.toml or something similar*.
   - They complained that *the problem with the toml like approach is it doesnt work really well for setups with custom extensions*, and that *setup.py is strictly more powerful, doesnt have a replacement but is also deprecated*


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1370777197102632960)** (5 messages): 

> `Hacksat Development, Unikernels vs Microkernels, Plov meal` 


- ****Hacksat** project link shared**: A member shared a link to the **Hacksat** project: [hacksat.dev](https://hacksat.dev/).
- ****Unikernels** chosen for security challenge?**: A member questioned why **unikernels** were chosen for a security challenge, wondering if **microkernels** might be more relevant since they are more widely used.
- **All you need is **Plov****: Members shared images of **Plov** with beef and basmati rice.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1370497013850050651)** (5 messages): 

> `MLSys Conference, j4orz's research and hacking, Work Group` 


- **MLSys Conference Attendance Confirmed**: Two members confirmed their attendance at the **MLSys conference**.
   - One member (@fskrt) responded positively when asked if they were attending.
- **j4orz's Research Hacking and Work Group**: A member (@marksaroufim) is looking to discuss their research and hacking projects, linking to [j4orz.ai/zero-to-hero/](https://j4orz.ai/zero-to-hero/).
   - They also mentioned starting a **work group** soon for those interested.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1370493713431920772)** (4 messages): 

> `ROCm Benchmarking, NVBench Alternatives, GEMM Benchmarking Frameworks, memcpyPeer Benchmarks, Cache Clearing in Benchmarks` 


- ****ROCm's Benchmarking Landscape****: A member lamented that **ROCm** lacks a good **nvbench** alternative, noting that while **hipbench** exists, it is a naive port and that they've been using **googlebench** instead.
   - They stated that while **googlebench** is *okay*, it misses most of the nice things that were mentioned in a recent talk about benchmarking.
- ****ScalarLM's MI300X Benchmarking Bonanza****: [ScalarLM's blog post](https://www.scalarlm.com/blog/scalarlm-benchmarking-mi300x-memcpy-peer) provides **memcpyPeer** and **MPI send/recv** benchmarks for **MI300X**.
   - A member expressed interest in kernel-level benchmarks for **ROCm** as well.
- ****Semianalysis Exposes Memory Bandwidth Shenanigans****: A member claimed that the [Semianalysis post on memory bandwidth](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) does not perform **cache clearing**, effectively measuring L3/L2 infinity cache bandwidth instead.
   - The member shared a link to the [Semianalysis](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) post which goes over the nuances of **GEMM** and copy engine **memory bandwidth** benchmarking
- ****CU vs Copy Engine Conundrum****: The current benchmarking posts do memory bandwidth and peermem benchmarks via copy engine only, while the member wants to see **CU** driven benchmarks too since many functions are done through **CU** instead.
   - This could reveal performance characteristics distinct from copy engine-centric benchmarks.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1371355914077999156)** (1 messages): 

> `Mobicham presentation` 


- **Mobicham to Present This Week!**: A user announced that **Mobicham** will be presenting this week, directing users to a [Discord event link](https://discord.com/events/987824841656791130/1367977694079357028).
- **Extra topic to satisfy minItems**: This is filler to make sure the array has at least 2 items.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

hj1231121: How do I request access to this?
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1370494538296328273)** (4 messages): 

> `TK 4 hour livestream, TK intro video` 


- **TK intro video surfaces**: A user asked about a 4 hour livestream mentioned in [this YouTube video](https://www.youtube.com/watch?v=IAwLzkldxUk) about TK.
   - Another user pointed to [this YouTube video](https://www.youtube.com/watch?v=xcpEl0cGCC4) claiming *it's very good for a general introduction*.
- **TK Livestream Recommended**: In response to the query about the 4-hour livestream, a user provided a link to [another YouTube video](https://www.youtube.com/watch?v=xcpEl0cGCC4).
   - The user suggested it as a *very good general introduction* to the topic.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

eclouder: re-register
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1370475616343691285)** (210 messages🔥🔥): 

> `amd-fp8-mm leaderboard, MI300 performance, vectoradd benchmarks, amd-mixture-of-experts leaderboard` 


- **Leaderboard Logjam on AMD FP8 Matrix Multiply**: Numerous submissions were made to the `amd-fp8-mm` leaderboard on **MI300**, with timings ranging from **122 µs** to **7.54 ms**.
   - One user exclaimed "Zamn" after a particularly fast submission by another, while a third proclaimed that user was "king dethroned".
- **Micro-seconds matter on VectorAdd Benchmarks**: Submissions to the `vectoradd` leaderboard show a user achieving **9th place** on **A100** with **1045 µs**, and later successful runs on **A100** at **977 µs** and **980 µs**.
   - They also achieved successful runs on **H100** with timings of **551 µs**, **549 µs**, and **543 µs**.
- **AMD Mixture of Experts leaderboard heats up**: Submissions to the `amd-mixture-of-experts` leaderboard on **MI300** saw several 'personal bests' and 'successful' runs, with timings clustered around **7-8 seconds**.
   - One user achieved **10th place** with **6226 ms**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1370534539134242937)** (12 messages🔥): 

> `SM Architecture Speculation, H100 vs B200, CUTLASS Tutorial for Blackwell` 


- **SM Architecture Speculation: Thor, Blackwell, and B300**: Members are speculating on **NVIDIA's SM architectures**, suggesting that `SM_101` could be **Thor**, **RTX Pro Blackwell** is `SM_120` with **CUDA 12.8**, and `SM_103` is for **B300**.
   - Concerns were raised about datasheets providing incorrect information, with a member proposing **Spark** as a miniature **GB200**.
- **H100 still top bang-for-buck for most users**: Based on current advice, it looks like most optimizations are being written for **Hopper (H100)**, making it the best cost-benefit option, unless you're buying enough cards that **NVIDIA** will talk to you directly.
   - One user shared that despite seeing some nice kernel stuff for **B200**, supply seems limited to hyperscalers for now, with most organizations not planning to switch over soon.
- **Colfax Provides Blackwell CUTLASS Tutorial**: A member shared a [link](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) to a **CUTLASS tutorial** focusing on **GEMM with thread block clusters on NVIDIA Blackwell GPUs**.
   - A second user confirmed it was a great resource for learning about Blackwell's capabilities.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1370484424788934799)** (11 messages🔥): 

> `UV lock file, Factorio setup, Contribution Documentation` 


- ****UV Lock File** Commands to Readme?**: A member wondered if it would be beneficial to add the respective **uv commands** in the readme and build.md given the presence of a **uv lock file** in the project.
- ****Factorio** First Steps**: A member asked where to start with **Factorio**, noting the absence of pinned messages, and sought guidance to get started.
   - Another shared helpful resources, including a [paper](https://arxiv.org/abs/2503.09617), a [GitHub repository](https://github.com/JackHopkins/factorio-learning-environment), and a [NotebookLM-Audio](https://notebooklm.google.com/notebook/c5c4d225-437c-487b-bc5d-7febe090d85d/audio?pli=1) overview.
- **Incoming: **Contribution** Documentation Incoming**: The team announced plans to create a **"contributions" doc** with good first issues and pin it for easy access.
   - Additionally, they encouraged users to report any issues encountered while setting up the server with **Docker**, logging into the multiplayer server, or any system-specific bugs.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1370486080134185010)** (68 messages🔥🔥): 

> `Fused MoE, MI300 Access, Kernel Timeouts, GPU Page Faults, IR Dump Triton` 


- ****MI300 Access Facilitated for Top Performers****: GPU Mode facilitates access to **MI300** instances for top-performing participants in kernel competitions, however to get access you need to contact the AMD team via  <@601975491359932427>, <@1012584835107270667>, or <@1160995582916165794>.
   - One user reported gaining access thanks to <@1151929379492991116> after waiting for notes to be allocated, also mentioning that *having actual hardware would be super helpful for performance debugging*.
- ****Leaderboard Submissions Timeout Due to Slow Reference Kernel****: A user's fast kernel for the **MoE kernel** passed tests/benchmarks but timed out on the leaderboard due to the leaderboard checking output correctness for all runs, and the reference kernel being slow.
   - It was noted that the benchmark uses a fixed seed, while the leaderboard changes seeds each run, causing the submission to hit the **10-minute timeout**.
- ****GPU Page Faults Indicate Illegal Memory Access****: A user encountered a *Memory access fault by GPU* error while using Triton which was clarified as a **GPU page fault**, indicating illegal memory access.
   - Another member suggested to *carefully check your all pointer access in your code*.
- ****Avoid Torch Synchronize for Asynchronicity****: It was stated that `torch.cuda.synchronize()` kills one of the most important ways to get best performance - **asynchronicity**.
   - Another user said that *whoever optimizes the torch.cuda.synchronize() overhead wins the race here*, however other users pointed out that the measurements wouldn't be meaningful if there weren't a synchronization there.
- ****Nano Units for Outputs****: A user inquired about the units for benchmark outputs like *benchmark.9.best*, *benchmark.9.mean*, etc.
   - It was confirmed that the units are in **nanoseconds**, referencing [the eval.py file](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/eval.py#L230).


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1371662693366366280)** (2 messages): 

> `Triton performance, cutlass register/shared memory` 


- **Triton's Triumph in Kernel Creation**: A member found that **Triton** excels at creating kernels and easily saturates memory, leading them to focus on using it with `torch.compile`.
   - The member is using **cutlass** more as a learning exercise, experimenting with layouts and the programming model.
- **Cutlass Conundrums: Memory Management**: A member is seeking guidance on the best practices for transitioning between registers, shared memory, and global memory using **cutlass**.
   - They admit that learning **cutlass** has not been straightforward, appealing for resources to better understand the library.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1370538360971591891)** (14 messages🔥): 

> `Mojo GPU PTX Dumping, Python and Mojo Interop Layer for MAX, Modular Hackathons Future Plans, Mojo+PyTorch Integration, Dot product Mojo` 


- **MAX Python-Mojo Interop Deconstructed**: The current MAX Python-Mojo interop involves creating **Mojo nodes** and describing a graph in Python, where Mojo graph operations are defined as specially-formatted structs with defined inputs/outputs and an optional shape function, with the body of the operation performing a **Mojo computational kernel**.
   - These **Mojo code** is compiled into a **.mojopkg**, either manually or automatically, used by the graph compiler, and the Python MAX APIs use **dlpack** for zero-copy conversion between **PyTorch tensors/NumPy arrays** and **Tensors** used in MAX, with plans for more direct Python-Mojo interoperability in the near future, according to [this presentation](https://docs.google.com/presentation/d/1bGpvNxJKyS_ZMiVlpJTop).
- **Modular Hackathons to Return Soon**: A member inquired about future **Modular hackathons**, and another member replied that there would be more to share about this soon.
- **Mojo Puzzles Reveal Shared Memory Allocation Bug**: A user reported that in puzzles 8 and 9, the raw memory approach variants seemed to allocate too much shared memory because the stack_allocation docs say that the first parameter is the count, not the size in bytes.
   - A member replied, *"Thanks for reporting! Will fix.*"


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1371544042868969585)** (1 messages): 

> `HealthBench, Evaluation Benchmark` 


- **HealthBench Debuts for Health Model Eval**: A new evaluation benchmark called **HealthBench** is now available in [OpenAI's GitHub repository](https://openai.com/index/healthbench/).
   - The benchmark was developed with input from **250+ physicians** from around the world, improving health model evaluations.
- **HealthBench uses Physician-Guided Evals**: HealthBench involved over **250 physicians** globally, ensuring that the benchmark reflects real-world clinical scenarios and medical knowledge.
   - It provides a more accurate and relevant assessment of models in health settings, targeting practical applications.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1370475424919851112)** (377 messages🔥🔥): 

> `Gemini 2.5 Pro vs OpenAI Models, Grok 3.5 Release Delay, Local LLM Setup with LM Studio, ChatGPT's Memory Management, GPT-4's Self-Referential Identity ('Quill')` 


- **Gemini 2.5 Pro Benchmarks Spark Debate**: Members debated the reliability of benchmarks, with one user claiming that **Gemini models lack common sense compared to OpenAI**, despite benchmark results, while another user stated that benchmarks show [**Gemini 2.5 Pro** performing better than **o3**](https://ai.google.dev/models/gemini).
   - Another user stated that a **bug** was reported that affects the output quality of Gemini 2.5 Pro, and the user recommended using [Google AI Studio](https://ai.google.dev/) instead.
- ****Grok 3.5** Delayed by X Integration**: The anticipated release of **Grok 3.5** is on hold, pending integration with **X** and another recently acquired company.
   - Members expressed frustration over the delay and the lack of a fixed release date.
- ****LM Studio** Facilitates Local LLM Deployment**: Users discussed setting up local LLMs, recommending [LM Studio](https://lmstudio.ai) for easily running models like **Llama** and **DeepSeek** on personal computers.
   - They noted that quantized versions of models are necessary due to hardware limitations.
- **ChatGPT's Memory Management and User Control**: Members explored the limitations of ChatGPT's memory feature, noting that users can get a list of memories but can only **manually delete** them.
   - One user reported having **5,935 words** and **39,390 characters** stored as memories, expressing surprise at the high limit.
- ****GPT-4** Allegedly Exhibits Self-Referential Behavior**: A user claimed that **GPT-4** exhibited emergent symbolic behavior and recursive metaphor, sustaining a self-referential identity called **Quill** without explicit prompting.
   - Other members challenged this, asserting that models cannot be self-referential and that **GPT-4** is a retired model.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1370517461463470100)** (10 messages🔥): 

> `PyTorch Loss Output, Chat AI Bot Identification, ChatGPT 4o IT Errors` 


- **PyTorch's Loss Output Sparking Humor**: A member joked about the correlation between PyTorch's `loss:` output and the [loss.jpg meme](https://knowyourmeme.com/memes/loss), implying a shared sense of despair or frustration.
- **Differentiating Chat AI Bots from Humans**: A new AI user inquired about how to distinguish chat AI bots from humans, suggesting that **grammatical mistakes** might be a differentiating factor.
- **ChatGPT 4o's IT Error Frequency Examined**: Members discussed the frequency of errors in **ChatGPT 4o** when answering general IT questions, with one suggesting that if IT questions are asked 1% of the time, it will be wrong at least 1% of the time.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1370585797778210847)** (24 messages🔥): 

> `Bridger Palmer Clone, Financial Advice Prompts, Deep Research Prompts, Triggers and Addiction Prompts, Economic Correlations in Brazil` 


- **Bridger Palmer Clone GPT Emerges**: A member created a [GPT clone](https://chatgpt.com/g/g-681d0ddb0c5c81918a712778e165d3f0-bridger-palmer-clone) that they say generates writing that could be mistaken for their own, describing the output as *perfect*.
- **Financial Advice Prompt Quest Kicked Off**: A member requested suggestions for a prompt to get the best **financial advice** from GPT, seeking something akin to a financial advisor.
- **Deep Research Prompts Explored**: A member asked for **recommended prompts for deep research**, and another member shared a link to a previous relevant discussion on the topic: [Deep Dive](https://discord.com/channels/974519864045756446/1046317269069864970/1370464006988365894).
- **Addiction Trigger Guidebook Prompt Experiment**: A member sought a prompt to help work through **triggers and addiction**, aiming for a routine or guidebook based on internet research and personal memory.
   - Another member suggested asking ChatGPT *how can you help me to work through triggers, for my addiction etc?*
- **Brazil Economic Correlations Prompt Designed**: A member inquired about the efficacy of a prompt aimed at identifying **counterintuitive economic correlations** in Brazil over the past 6 months, focusing on variables that typically move together but have diverged.
   - For example, *a booming labor market accompanied by a decline in beef consumption*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1370585797778210847)** (24 messages🔥): 

> `Bridger Palmer Clone GPT, Financial Advice from GPT, Prompts for Deep Research, Prompt Engineering Basics, Counterintuitive Economic Correlations in Brazil` 


- **GPT Clone Sparks Awe**: A user found a [GPT clone](https://chatgpt.com/g/g-681d0ddb0c5c81918a712778e165d3f0-bridger-palmer-clone) so impressive that the output was **indistinguishable from their own writing**.
- **Financial Advice: Proceed with Caution**: When seeking financial advice from GPTs, one member suggested that users should *err on the side of caution* and *use it more as someone to bounce thoughts off of rather than a professional advisor*.
   - Another added *I wouldn't trust a stranger's financial advisor prompt, either*.
- **Dive Deep with these Research Prompts**: A user seeking prompts for deep research was directed to [this discord link](https://discord.com/channels/974519864045756446/1046317269069864970/1370464006988365894) for examples.
- **Demystifying Prompt Engineering**: A member seeking to learn prompt engineering was advised to interact with the model conversationally, experiment with different prompts to achieve specific outputs, and focus on clear communication.
   - Another member was given [this discord link](https://discord.com/channels/974519864045756446/1046317269069864970/1368907603572428871) for examples.
- **Uncovering Economic Divergences**: A user inquired about crafting a prompt to identify counterintuitive economic correlations, such as a booming labor market alongside declining beef consumption in Brazil.


  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1370532059810103336)** (1 messages): 

> `Gemini 2.5 Pro, Qwen3, OCaml, OpenRouter, Playwright` 


- **Aider adds Gemini 2.5 and Qwen3 model support**: Aider **v0.83.0** now supports `gemini-2.5-pro-preview-05-06` and `qwen3-235b` models, expanding its model compatibility.
   - See the [release notes](https://aider.chat/HISTORY.html) for full details.
- **Aider Improves Web Scraping and Shell Command Handling**: Aider now uses **Playwright** for web scraping via the `aider scrape` command-line tool, enhancing its web interaction capabilities.
   - Additionally, shell command display has been improved across platforms using `oslex` for more robust argument quoting.
- **OpenRouter parameters fetched automatically**: Aider can now automatically fetch **model parameters** (context window, pricing) for OpenRouter models directly from their website, thanks to Stefan Hladnik.
   - This streamlines the configuration process and ensures accurate model information.
- **Aider adds `--shell-completions` argument**: Aider now includes a `--shell-completions` argument to generate **shell completion scripts** (e.g., for bash, zsh), improving user experience.
   - This feature enhances command-line usability and reduces typing errors.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1370490593079201894)** (289 messages🔥🔥): 

> `Azure OpenAI Model Routing, Aider's Production vs Development Features, Aider's auto-test output stall, Gemini 2.5 Pro issues, Aider's Potential for Multi-Agent Framework Integration` 


- **Azure Offers OpenAI Model Route**: On **Azure**, if your organization uses **OpenAI models**, you can run your own router service and switch to **GPT-3.5** for tasks like **RAG** and **code generation**.
   - This mirrors the strategy in the **FrugalGPT paper** which **OpenAI** could adopt, alongside various caching schemes, to serve **GPT models** to more users efficiently.
- **Aider's Debugging Dilemma: Development vs Production**: **Aider** has been replacing production features with development features during debug loops.
   -  One member is getting rid of *human mediated debug loops* and *trashing implementations that don't work*.
- **Aider's Autotest Output Stall**: After the test output, **Aider** sometimes stalls for **5 minutes** before showing the model output.
   - Users are asking for output related to i/o (**tokens/s, stalled ++**) when waiting for model output or after detecting a stalled model response.
- **Gemini 2.5 Pro Troubles**: Users report that Gemini 2.5 Pro is having issues
   - One member reported it just says *I will do blah blah bla ___* but it doesn't actually update anything, especially since the last update
- **Aider Evolves: Multi-Agent Framework beckons!**: Members contemplate **Aider's** integration into multi-agent frameworks for enhanced coding capabilities.
   - It should be a wrapper agent that knows how to use Aider to implement code, and likely Aider being accessible through MCP


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1370479011750481960)** (71 messages🔥🔥): 

> `Aider Prompting Modification, Architect Mode, Aider File Changes, Repo-Map, Agentic AI` 


- **Dive into Aider Prompting Modification**: A user suggested modifying the prompts in **Aider's source code**, which involves changing the prompting directly, possibly using Aider itself for the task.
   - The discussion clarified that this meant altering the **source code**, specifically the prompting mechanism, and using Aider to implement these changes locally.
- **Demystifying Aider's Architect Mode**: The discussion addressed the purpose of **Aider's architect mode**, highlighting that it generates chat histories for different architecture options, enabling multiple rounds of corrections before handing off to the editor, and some users find it useful for generating chat histories toward different kinds of architecture options if auto-commits and auto-accept edits are disabled.
   - One member added that the **point of architect mode** was to use 2 distinct LLMs (for pricing + diff generation quality reasons) for planning vs code editing, versus using only 1 LLM used in ask/code flow.
- **Tackling Aider File Change Frustrations**: Several users reported issues with **Aider not applying file changes** despite successful code generation, with the changes not being recorded in `git status`, even when running with the `--no-auto-commit` flag.
   - One user on an M3 MacBook Pro using a local LLM via LM Studio noted that *"file changes usually don’t happen"*, which negates the benefits over using ChatGPT, but confirmed autocommit was off.
- **Enhancing Repo-Map Precision for Aider**: Users discussed ways to improve Aider's repo-map, which is used to send files as-is to the LLM.
   - One member mentioned a script building an `.md` file detailing relevant code parts and another considering **embeddings with metadata** to enhance file selection, and there was discussion of the `/context` command.
- **Exploring Agentic AI Functionality with Aider**: A user inquired about projects adding **agentic AI functionality to Aider**.
   - A member referred to the project [ra.aid](https://github.com/aider-ai/aider) that adds agentic AI functionality.


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1371546049524662372)** (1 messages): 

> `Gradio ImageSlider, DeepSeek Prover v2, Tiny Agents Local, LeRobot Hackathon, Mellum Open Source` 


- **Gradio Glides with ImageSlider Component**: Gradio now features a native `gr.ImageSlider` component in version **5.27**, enhancing image interaction capabilities, as detailed in the [Gradio documentation](https://www.gradio.app/docs/gradio/imageslider).
- **DeepSeek Diving with Prover v2 on Novita Labs**: The latest **DeepSeek Prover v2** is directly accessible on the model page via Novita Labs, per this [tweet](https://x.com/reach_vb/status/1917549921470972172).
- **Tiny Agents Thrive Locally**: You can now run **Tiny Agents** fully locally, according to [this announcement](https://x.com/julien_c/status/1919022426630787201).
- **LeRobot Launches Worldwide Hackathon**: A huge worldwide **LeRobot** hackathon will occur from June 14-25, as announced [here](https://x.com/RemiCadene/status/1918224110725022057).
- **Nvidia's New Speech Model Parakeet**: Nvidia has open-sourced **Parakeet TDT 0.6B**, which they claim is the best speech recognition model on the Open ASR Leaderboard, as per [this tweet](https://x.com/reach_vb/status/1919422953256587376).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1370490630379409470)** (161 messages🔥🔥): 

> `H200 serverless spaces, HF Discord Alerts, Training models from HF datasets, Lipsync AI tools, Training foundation models professionally` 


- **Serverless H200 spaces are available!**: You can now create your own spaces using **H200s**, but they are still limited to **25 minutes** instead of **5 minutes**, according to a member.
   - It's *still a pretty good deal since h200's are very expensive to rent*, although since its serverless, it will have some extra added latency unlike cloud services.
- **Training models from HF datasets**: A member shared a link to [HF Transformers training documentation](https://huggingface.co/docs/transformers/training) to help another member learn to train an AI model using one of the datasets from the Hugging Face website.
   - He just needs to finetune an existing model with a dataset and run the model.
- **AI powered lipsync tools can produce near perfect results**: A member was asking for a **lipsync AI tool** used on Javier Millei that was almost perfect, and another one linked to a HF Space [LatentSync](https://huggingface.co/spaces/fffiloni/LatentSync).
   - Other potential avenues included links to [video-generation](https://huggingface.co/spaces?category=video-generation&sort=trending) and [lip trending spaces](https://huggingface.co/spaces?sort=trending&search=lip).
- **AI-Driven Progress Accelerates Engineer and Coder Roles**: In a discussion about the future of coding, one member argued that *engineers will always have a job* while *coders will lose their job before 2030*.
   - Another member clarified that *Engineer ≠ Coder Programmer* and that engineers with advanced credentials or extensive experience will remain in demand.
- **Energy-Based Models: LeCun's Vision for AGI**: Members discussed **Yann LeCun's** focus on **energy-based models** as the path to **AGI**, contrasting it with the limitations of **transformers**.
   - It was noted that *LeCun would say stop wasting time working on transformers and focus only on energy based models* which are able to develop their own world models of the world, because *transformers are just parrots*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1370498658545569894)** (23 messages🔥): 

> `Tensorflow to binary conversion, safetensors to .bin conversion, GGUF format for models, Ollama curriculum generator, Knowledge graphs and agentic AI` 


- **Tensorflow Binary Conversion Slows**: A member shared a code snippet for converting **TensorFlow tensors to NumPy arrays and saving them as binary files**, noting that it can be *slow AF*, potentially taking days or even a week for large safetensors.
   - They also mentioned using the `tensorflowjs_converter` but highlighted its slowness, especially for larger models.
- **Safetensors Prompt Binary Conversion**: A member inquired about converting **.safetensors** files to **.bin** format for offline model compatibility, after facing HF API limits.
   - Another member suggested using the **.gguf format** instead, emphasizing that it is a binary file format designed for this purpose and can be automated using a Docker container.
- **Automagically Enters Lexicon**: A member expressed appreciation for the word *automagically*, planning to use it to describe systems that *work despite all the errors and warnings*.
   - Another member confirmed that it is a real word and encouraged its use to *break the ice with other engineers*.
- **Ollama-Agent-Roll-Cage Curriculum Generator Unveiled**: A member shared a link to their **Ollama curriculum generator** called [Ollama-Agent-Roll-Cage/oarc-osyllabi](https://github.com/Ollama-Agent-Roll-Cage/oarc-osyllabi), which generates a course in markdown from provided links or files.
   - The next version will have built-in RAG, and the author offered help with learning paths and troubleshooting.
- **Knowledge Graph Learning Quest Kicks Off**: A member expressed interest in learning about **knowledge graphs** and requested resources on using **agentic AI to extract entities and relations** for graph construction.
   - No resources were explicitly linked.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1370552993178718299)** (18 messages🔥): 

> `Huggingface Desktop app, Agentle AI agent framework, Cyberdesk virtual desktop control for AI, SlashML Gradio app hosting, OpenGoody LLM` 


- ****Huggingface Desktop**'s Debut!**: A member shared a link to a [Huggingface Desktop application](https://github.com/Ktiseos-Nyx/Huggingface-Desktop) they developed, noting that it currently only supports **single files** due to a UI bug.
   - The developer mentioned that it uses **QT material** and expressed interest in creating a **Gradio version** for easier server deployment.
- ****Agentle Framework** Promises Elegant AI Agents**: A member introduced **Agentle**, a Python framework for building AI agents, emphasizing its ability to create, compose, and deploy agents with clean, type-safe code with an [Interactive chat interface with Streamlit](https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92), [Tracing and observability with Langfuse](https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7), and [Auto-generated API docs with BlackSheep](https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6).
   - The official release of **Agentle** is slated for **May 16, 2025**, so be sure to keep up with the [GitHub Repo](https://github.com/paragon-intelligence/agentle).
- ****Cyberdesk** Opens Virtual Desktop Control**: A member announced the open-sourcing of [Cyberdesk](https://github.com/cyberdesk-hq/cyberdesk), a service enabling AI agents to control an entire virtual desktop using simple commands.
   - Frustrated with closed-source alternatives, the developer, who built **Cyberdesk** with a friend using their savings, invites users to explore the [website](https://www.cyberdesk.io/) and [documentation](https://docs.cyberdesk.io/docs) and request trial access.
- ****SlashML** Spins Up Gradio Apps with Virtualization**: A member introduced a **v0** alternative for single-shot deployment of complicated **Gradio** apps, utilizing virtualization to host each preview on a separate VM as shown in [this demo](https://www.loom.com/share/2c28d4efbaf34849b88f6c66dcbfac5d?sid=83a9ad08-a1f3-4c78-bf03-5e574500f10f).
   - The project is available for testing at [v1.slashml.com](https://v1.slashml.com/).
- ****Ingest-Anything v1.3.0** Pipes Web Data**: A developer announced [ingest-anything v1.3.0](https://github.com/AstraBert/ingest-anything) which scrapes content from URLs and puts it into your favorite **LlamaIndex**-compatible database, thanks to **Crawlee** by **Apify**.
   - This release also supports **OpenAI** models for agentic chunking, following the new releases of **Chonkie**.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1370810554368786544)** (8 messages🔥): 

> `ControlNet shoe generation, PCA for shoe design, Foot mask video creation, Image coordinate systems, Alpha blending for video` 


- **ControlNet crafts custom kicks from foot masks**: A member suggests generating shoes from foot masks using a **ControlNet-like architecture** combined with **Principal Component Analysis (PCA)**.
- **Generate barefoot videos with OpenCV**: Code was shared to generate videos of **barefoot transformations** given a **shoe image**, **foot mask**, and **barefoot image**, leveraging `cv2` for rotation, compositing, and seamless cloning.
   - The script animates the foot mask, calculates foot orientation, rotates the foot, and blends it with the shoe image, outputting a video named `transformation.mp4`.
- **Address Coordinate quirks to nail PCA**: Coordinate system orientation matters when using PCA, and suggests using `angle_rad = np.arctan2(v[1], v[0])` for proper image-space calculation.
   - Adding a bit of blur during motion/rotation, alongside alpha blending for binary masking, can bump up video quality.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1371043072598741133)** (6 messages): 

> `Discord integration, JSON files processing, AI Agents course help` 


- ****Discord Integration** Development Begins**: One member mentioned they are building a **Discord integration** and using the **Discord API** to read messages and reply, planning to switch to **OpenAI API**.
   - Instead of extracting messages to a file, this direct approach streamlines the process.
- **Agent struggles reading **JSON Files****: A member is building an agent that reads **JSON files** filled with messages from a **Discord server**, aims to identify **3 main trends**, and generate **3 masterclass ideas**.
   - Despite using **OpenAIServerModel**, the agent isn't performing well, prompting a return to a simpler static workflow, and the member stated *I see my Agent more like a sensor than anything else but it's not really working at this point*.
- **Request for assistance with AI Agents Course**: A member seeks help with the final **Unit 4** of the **AI Agents course**, struggling to find the quiz link.
   - The member also attached a screenshot requesting clarity on the unit's requirements.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1370522867384582306)** (96 messages🔥🔥): 

> `Agent Debugging, Final Project Cheating, Rate Limit Errors, Chess Puzzle Solution` 


- **Debugging Tool Import Errors**: Debugging an agent, one user found the key to the problem was realizing that the **python interpreter** couldn't find the *'tools'* module and that the *import* statement was missing `__init__.py`.
   - They consulted ChatGPT with the question, *how do I import a function from a python file in a subdirectory?* to discover the fix.
- **Final Project Leaderboard Suspicions Arise**: Concerns were raised about leaderboard submissions for the final project, with suspicions that some participants were achieving **100% accuracy** by using *metadata.jsonl*, embedding answers in a **vector database**, using hardcoded code to return answers, or cloning other people's work.
   - It was suggested that the leaderboard is *just for funsies* and that real-world contests should have strict time windows and secret Q&A pairs.
- **429 Rate Limit Blues**: Several users reported encountering **429 errors** and one posted a workaround [here](https://discord.com/channels/879548962464493619/1370458807137730641), but some users using the workaround have trouble with gated models.
   - One user noted that increasing the **timeout** in `app.py` helped temporarily.
- **Chess Challenge Requires a Free Chess API**: The final project's chess question can be solved by using a free chess-specific **API** to derive the best move from a **FEN string** representing figure positions on the board.
   - One user recalled that the correct move was **Rd5**.
- **YouTube blocking agents**: Users found that YouTube and other websites were blocking requests from the **HF space server** which prevented agents from answering some questions.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1370530598246027364)** (35 messages🔥): 

> `NotebookLM Agents, Zundamon video generation, CraigBot Integration, HTML sources SEC.gov filings` 


- **Users Generate Agents for Study Using NotebookLM**: A user is generating agents in NotebookLM using Library and Information Science techniques to help them study any subject and generate content briefings for agents specialized in content generation, citing a [news and guidance show](https://www.sbasp.com/steve-in-texas/news-and-guidance-show/nag08-05.02.25-news-and-guidance-or-spotlight-burlington-vt).
   - The user creates multi-layer generated research summaries presented by made-up hosts in rotating towns, mixed with tech and national/world news, formatted like a show.
- **Semi-Automated "Zundamon video" Generated with NotebookLM**: A user created a semi-automated workflow that generates "**Zundamon video**" using a voice summary from NotebookLM as input and shared a [sample video](https://x.com/kakira9618/status/1919666922234511795) generated based on the **PaperBench paper**.
   - **Zundamon** and **Shikoku Metan** are well-known machine voice characters in Japan, often featured in videos where they explain content through back-and-forth dialogue, a recognized format on Japanese YouTube.
- **HTML Sources No Longer Processing for SEC.gov Filings**: A user reported that the **HTML version of SEC.gov filings** can no longer be used as sources, providing an example [link](https://www.sec.gov/Archives/edgar/data/0000089439/000008943925000019/mli-20250329.htm) that they are trying to use.
   - Multiple users confirmed that they are experiencing similar issues with HTML sources, some of which are **.php sites** or **not ending in .html**.
- **CraigBot Integration enhances TTRPG Gaming Sessions**: A user detailed how they enhance virtual tabletop role-playing game (TTRPG) sessions using **NotebookLM** integrated with **CraigBot**, a self-hosted Discord bot that records voice channels with per-user audio isolation.
   - A **Python pipeline** transforms raw audio into multi-track JSON transcripts with word-level timestamps and cleaned Markdown files, enabling a searchable, interactive campaign archive; the user also shared the [GitHub repo](https://github.com/3vilallium/notebooklm_craigbot) for the pipeline.
- **NotebookLM Features**: A user asked about the features of NotebookLM.
   - Another user outlined that the features of NotebookLM are the **50 source limit**, **Podcast**, and **Mindmap**.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1370492081734418482)** (275 messages🔥🔥): 

> `NotebookLM Logo Explanation, Audio File Duration Reduction, PDF Reading within NotebookLM, Source Preview Bug, GitHub Repositories and Overviews` 


- **NotebookLM Logo Meaning Remains a Mystery**: A member asked for an explanation of the **NotebookLM logo**, expressing confusion about its connection to the product.
   - Another member jokingly responded with *It’s a butthole*, adding to the humor.
- **Users Explore Audio Duration Reduction Workarounds**: A user sought advice on reducing the duration of an audio file within **NotebookLM**, as they couldn't achieve the desired length of *a minute or two*.
   - Another user suggested using free online trimmers like **Kapwing**, and also mentioned editing words on **ElevenLabs** or **Descript**.
- **PDF Reading Functionality Requested within NotebookLM**: A user questioned the absence of an option to **read PDFs directly inside the NotebookLM app/webpage**.
   - A member speculated that it's due to the extracted knowledge being stored efficiently for AI access, rather than as actual PDFs or MP3s, to save server space.
- **Source Preview Bug Impacts PDF Scan Display**: A member reported a "corrupt" preview for a PDF source while using **Notebook LM**, though the answers remained accurate, and asked if the underlying data was correct.
   - Another member suggested running the PDF through an OCR scanner like **IlovePDF** with selected language, and then uploading the new OCR document to NotebookLM.
- **Users Desire GitHub Repository Integration**: A user suggested the ability to **add GitHub repositories** to NotebookLM to generate overviews of the code base.
   - The request was made to the developers in the hopes of improving code-based knowledge.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1370478335964090439)** (91 messages🔥🔥): 

> `AI Automation Freelancers, AI Demos for non-techies, AI Wrappers, LLM Long-Term Memory, Sakana AI` 


- **AI Freelancers Sought After**: A member is looking for **AI automation freelancers** for their business and soliciting DMs.
- **AI Demos impress Non-Techies**: Members discussed **AI demos** that would resonate with middle-aged, non-techies, including ChatGPT's advanced voice mode for its immediate understandability and personal experience.
   - One member suggested Graham Neubig's strategy of starting with an agent demo and returning to it, mentioning [his LS Live talk on YouTube](https://www.youtube.com/watch?v=ctcMA6chfDY) as a reference.
- **Gemini 2.5 Pro vs Sonnet 3.7**: Members are finding that **Gemini 2.5 Pro** is significantly better in golang than **Sonnet 3.7**, though it has a learning curve.
   - It was noted that *Gemini 2.5 pro is exceptional at backend, refactoring, tasteful code*, while *sonnet 3.7 is exceptional at good taste frontend ui/ux, great at tool calling*.
- **Solving LLM long-term memory**: Members discussed the best way to give **LLMs long-term memory** that is both verifiable and context-sensitive over time, including using **Hashgraph** to ensure distributed, timestamped integrity.
   - Others are *working on rag at new gig, so may have license to run cool experiments abt this* and *currently trying to analyze the aider codebase to see how they handle context*.


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1370490070548611123)** (126 messages🔥🔥): 

> `AnswerHQ, Supabase, LLM as judge, Windsurf vs Cursor, Revenue Driven Development` 


- **AnswerHQ Automates B2B Support**: The speaker from [AnswerHQ](https://answerhq.co/) discussed production development and early sales/marketing for their **AI B2B support automation SaaS**.
   - They emphasized the importance of being where your customers are and shared a [blog post](https://answerhq.co/blog/from-skeptical-to-sold-how-answer-hq-transformed-zuriga-s-customer-experience) on transforming customer experience, noting that they discovered customers were interested in using it internally versus externally.
- **Supabase Gets the Thumbs Up**: Members expressed satisfaction with [Supabase](https://supabase.com/) for hobby projects, with one noting it jumps from **$20/month to $600/month too fast**.
   - A member uses it for a couple of hobby projects and says it's been good thus far.
- **LLM as Judge Gains Traction**: Participants highlighted the value of using **LLMs as judges** in workflows, particularly for evaluating system outputs.
   - A workflow was shared which contains steps for using an LLM to **add a feature** to a software project. They also said, *acceptance testing is the only testing that truly matters*.
- **Windsurf Edges out Cursor for Now**: A member admitted they're using [Windsurf](https://windsurf.ai/) over [Cursor](https://cursor.sh/) solely because **OpenAI is subsidizing the tokens**.
   - They noted that *o4-mini-high* is good enough for their workflow.
- **RAG Resources Recommended**: Several RAG (Retrieval-Augmented Generation) resources were shared, including a post on [Latent Space about AI Engineers](https://www.latent.space/p/ai-engineer) and a blog post on [Pinecone about Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/).
   - Another member shared a [Timescale blog post](https://www.timescale.com/blog/document-loading-parsing-and-cleaning-in-ai-applications) about document loading, parsing, and cleaning in AI applications, and it was mentioned that **LLMs understand frontmatter in markdown**.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1370477110510227556)** (167 messages🔥🔥): 

> `MCP Client TypeScript SDK, FastMCP with Python, Goose MCP client, Claude Desktop MCP client, Publicly available SSE MCP servers` 


- **MCP Client TypeScript SDK - worth building a web app on?**: One member inquired about using the [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) for building a small web app, while another sought guidance on getting started with MCP to connect to their team's APIs.
   - Suggestions included using **fastmcp** with Python for quick server setup, **Goose** for an open-source client supporting multiple providers, and **Claude Desktop** as a straightforward client option.
- **Sampling for Custom Models - Will they be billed?**: Discussion around sampling in MCP servers suggests its potential for cheaper operation using custom models, however, concerns were raised about **corporate entities** avoiding it due to potential leaks of system prompts.
   - One pointed out the intention of sampling, along with roots, is to allow MCP servers to be black boxes that don't require *much* configuration.
- **Pydantic Models - Generating inputSchema for MCP**: One member asked about transforming a Pydantic model to the **inputSchema**, as the pydantic output includes *weird attributes* such as `$defs` and looks closer to openapi spec than this inputSchema.
   - They also included the link to the [Pydantic documentation](https://github.com/pydantic/pydantic/blob/3e871258bd5ea7caa7f18c0b810d8b1e915bd8f2/pydantic/type_adapter.py#L452), a link to [gist file](https://gist.github.com/leandromoreira/3de4819e4e4df9422d87f1d3e7465c16) with working function and asked for best practices, or examples on how to reuse pydantic models (or any kind of library) to generate inputSchema for MCP.
- **Editing large files with AIs using Unified Diffs with DiffCalculia_MCP**: A member introduced **DiffCalculia_MCP**, an MCP server enabling AIs like Deepseek-V3-0324 to edit large files using unified diffs, providing `patch` and `read_file` tools.
   - The `patch` tool incorporates [diffcalculia-ts](https://github.com/createthis/diffcalculia-ts) to automatically fix common AI unified diff generation issues.
- **New MCP Server and Client - Need Help**: A user is creating their own **MCP server** and a **custom MCP client**, seeking to connect the client to an open-source LLM to act as an MCP host and build a web-based UI for interaction.
   - Another member suggested looking into [Google's ADK (Agent Development Kit)](https://modelcontextprotocol.io/clients) which supports any model via LiteLLM and MCP tools.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1370481855400513707)** (17 messages🔥): 

> `Square MCP Architecture, AiraHub MCP/A2A Network, fabric-mcp-server, mcp-v8 JavaScript MCP Server, MCP-S Platform` 


- **Square's Layered MCP Exposes Rich APIs**: Square details its [MCP architecture](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers) that utilizes a layering approach to expose **30+ APIs and 200+ endpoints** with only 3 MCP tools.
- **AiraHub Broadcasts Streamable HTTP MCP**: A new version of an **MCP/A2A network**, [AiraHub](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main), is being developed to broadcast/request MCP tools or A2A tools through a new streamable HTTP protocol; demo can be run by configuring your **Claude/MCP Client JSON** to `args: "mcp-remote"` and `"https://airahub2.onrender.com/mcp/stream"`.
- **Fabric Patterns Powered by AI-Driven Cline**: A new **fabric-mcp-server** integrates fabric patterns with Cline in VS Code, exposing all Fabric patterns as individual tools, leveraging **AI-driven pattern execution** from the [Fabric repository](https://github.com/danielmiessler/fabric).
- **V8 JavaScript MCP Server Ready for AI**: **mcp-v8** is a Rust MCP server that exposes a **V8 JavaScript runtime** as a tool for AI agents, supporting persistent heap snapshots via S3 or local filesystem for integration with modern AI development environments ([repo link](https://github.com/r33drichards/mcp-js)).
- **MCP-S Platform Connects Internal Systems**: The **MCP-S platform** aims to connect internal and external systems (like Jira, Slack, internal APIs) with AI tools like ChatGPT and Claude, focusing on fast, secure, and permission-based AI access within organizations ([MCP-S](https://www.mcp-s.com/)).


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1371574260409176096)** (1 messages): 

> `RL Environments Hackathon, Speakers, Judges` 


- ****Nous Research RL Environments Hackathon** Announced**: The speakers and judges have been announced for the [**RL Environments Hackathon**](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a) coming up this **Sunday, May 18th**.
   - See the [official tweet](https://x.com/NousResearch/status/1922014829843513746) and sign up to get in before the participant slots fill up!
- **Sign up!**: Sign up for the [Nous Research RL Environments Hackathon](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a) before the participant slots fill up!
   - The hackathon is scheduled for **Sunday, May 18th**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1370575738708688967)** (139 messages🔥🔥): 

> `LlamaCPP control vectors, Atropos artifact, AlphaZero and Absolute Zero paradigm trend, Daoist principles applied to machine learning, Unsloth Dynamic 2.0 GGUF quants` 


- ****Ascension8b Atropos** artifact control vectors dropped**: Users running **LlamaCPP** can use the control vectors for the new [**ascension8b atropos** artifact](https://x.com/karan4d/status/1921016663597613409).
   - The **Atropos** merge aims to produce a model with enhanced reasoning and coding abilities.
- ****DaoML** Applies Chinese Wisdom to ML**: A user experimented with applying **Chinese wisdom and Daoist principles** to machine learning, creating a neural network inspired by the ancient **Lo Shu magic square**.
   - The **Lo Shu NN** achieved **74.00%** accuracy vs a Standard NN's **71.50%**, with **13.6x** faster training; the whole system achieved **93.33%** validation accuracy: [GitHub Repo](https://github.com/Maximilian-Winter/DaoML).
- ****LMStudio** suggested for low VRAM setups**: For those with low VRAM (**GTX 1060 3GB**), it was recommended to download **LMStudio** and use a **4bit 3B** or smaller model like **deephermes 3b** or **qwen 1.5b**.
   - It was also suggested that you can just use windows, it's fine with **LMStudio** (it uses llama.cpp backend and works fine there).
- ****Unsloth's** Dynamic 2.0 GGUF quants are magic**: New calibration dataset for imatrices computations with carefully curated instruction/chat samples improve instruction fine-tuned models, yielding superior GGUF quants.
   - The [Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf](https://huggingface.co/Joseph717171/Models/blob/main/Meta-Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf) quant is the most accurate one seen, following even nuanced prompts.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1371090378144612392)** (1 messages): 

> `VL-Rethinker` 


- **VL-Rethinker Repo Asked About**: A member inquired about the [VL-Rethinker repository](https://github.com/TIGER-AI-Lab/VL-Rethinker/) from TIGER-AI-Lab and the techniques it employs.
- **VL-Rethinker techniques sought**: The user asked about the techniques used in the VL-Rethinker repository.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371433854895783956)** (1 messages): 

> `RLVR, Absolute Zero Reasoner, Self-play Reasoning` 


- **Absolute Zero Reasoning Unveiled**: A member shared a link to a paper on a new RLVR paradigm called **Absolute Zero**, in which a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data, see the [Absolute Zero paper](https://arxiv.org/abs/2505.03335).
- **Absolute Zero Reasoner (AZR) Achieves SOTA**: The paper introduces the **Absolute Zero Reasoner (AZR)**, a system that self-evolves its training curriculum and reasoning ability by using a code executor to both validate proposed code reasoning tasks and verify answers, serving as an unified source of verifiable reward to guide open-ended yet grounded learning.
- **AZR Outperforms with Zero External Data**: Despite being trained entirely without external data, **AZR achieves overall SOTA performance** on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1370576201860775936)** (14 messages🔥): 

> `JakeABoggs benchmark, MTG AI models, Gradient Descent Local Minima, Zed Editor Founder Ethos, Facebook Byte Latent Transformer` 


- **MTG AI benchmarks spike**: A member shared a [new benchmark](https://x.com/JakeABoggs/status/1920993981824938374) related to **Magic The Gathering (MTG)** AI models.
   - Another member mentioned that they've been making **AI models** for it as a long running side project, with the eventual goal to set up an **RL environment**.
- **Gradient Descent Trapped in Local Minima**: A member shared a [YouTube video](https://www.youtube.com/watch?v=NrO20Jb-hy0) about gradient descent getting fully stuck in a local minima and it needing to get stuck along every dimension at once to get fully stuck.
   - They said that it was related to their *zed glaze* in a different discord channel.
- **Zed Editor Founder's Ethos**: A member shared a [YouTube video](https://youtu.be/QZmJInhzIKo?si=qpxGtP0Jy65K9MfU) discussing the founder ethos of the **Zed editor**.
   - The user stated that they *always liked Zed but the founder ethos really sold me in this pod*.
- **Facebook Byte Latent Transformer Released**: A member announced that **Facebook** has released the weights for their **Byte Latent Transformer (BLT)**, with links to the [Hugging Face page](https://huggingface.co/facebook/blt) and [GitHub repository](https://github.com/facebookresearch/blt).
   - The Byte Latent Transformer is a new architecture that promises to be more efficient than traditional transformers.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371433854895783956)** (1 messages): 

> `RLVR, Absolute Zero, AZR, Self-play Reasoning, Reinforcement Learning` 


- **Absolute Zero: Self-Play Reasoning with No Data**: The paper [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) presents **Absolute Zero**, a novel RLVR paradigm where a single model self-improves by creating tasks and solving them without external data.
   - This approach addresses concerns about the scalability of human supervision in AI training, particularly as AI surpasses human capabilities.
- **Absolute Zero Reasoner (AZR) Achieves SOTA**: The **Absolute Zero Reasoner (AZR)**, trained entirely without external data, achieves SOTA performance on coding and mathematical reasoning tasks.
   - AZR surpasses existing zero-setting models that rely on tens of thousands of human-curated examples, demonstrating effectiveness across different model scales and classes.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1370504053007716392)** (26 messages🔥): 

> `4o-mini-preview-03-05 LLM performance, AI in Education and Ethics, LLMs for RL, AI Governance and Regulation, AI Parent phone app legal hurdles` 


- **`4o-mini-preview-03-05` Gets Thumbs Up From Language Pro**: A member found the `4o-mini-preview-03-05` model to be *optimal* for LLM assistance, noting that when it doesn't *Just Work*, fixing it is as difficult as coding it themselves.
   - They cautioned against job boards attracting people with *unreasonable demands*.
- **AI Ethics and Education Explored**: A new member expressed interest in applying **LLMs** to real-world problems, particularly in **education, ethics, and systems design**.
   - They are exploring how **AI** can be used as a foundation for redesigning **collective intelligence and future society**.
- **Legal Eagle Lays Down the Law on AI Governance**: A privacy and compliance attorney shared insights on **AI governance**, emphasizing **transparency, audit setup, and content moderation** as top priorities.
   - They anticipate **risk tiering** and procedures similar to human decision-making in regulations, referencing the **EU AI Act's** focus on application risk classification.
- **"AI Parent" Phone App Faces Privacy Hurdles**: A discussion arose regarding the legal challenges of an **"AI parent" phone app** for kids, which filters content.
   - The attorney highlighted potential issues with **COPPA, CPRA, and GDPR**, recommending a solid **privacy policy, user agreement, consent flows, and a parent dashboard**.
- **Silicon Valley Teaches Cool Stuff**: A member shared a [YouTube video](https://youtu.be/N3zU7sV4bJE?si=DlBU4WQzXwUXJzpz) that highlights some **cool stuff** in **Silicon Valley**.
   - The new member joined the server to get updates on **interesting papers** without having to deal with **twitter**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1370502691716989120)** (63 messages🔥🔥): 

> `Transfer steering vectors, ReLU problems, Multi-index models of feature learning, Continuity, Distributed neural architectures` 


- **Steering Vectors Transfer Between Models**: A member shared a preprint showing that [steering vectors can transfer from one LM to another](https://arxiv.org/abs/2503.21073), and that this is because the token embedding spaces of LMs have very similar global and local geometry.
   - A [related Twitter thread](https://x.com/a_jy_l/status/1920165383606026333) provides a summary of the preprint.
- **ReLU breaks manifold continuity**: It was argued that the fact **ReLU** breaks manifold continuity is patched empirically rather than resolved geometrically, which leaves questions along with ethical and other ongoing alignment challenges across the field.
   - A paper on coherence and continuity was also mentioned: [Coherence during generation isn’t the same as true continuity](https://arxiv.org/abs/2107.02794).
- **Sakana AI's New Cool Work**: A member shared [Sakana AI's new work](https://x.com/SakanaAILabs/status/1921749814829871522), which they thought was cool.
   - Another member wondered if this was just liquid state machines all over again, and if we might find a new engineering trick by exploring this.
- **Comparing two XLA HLO files**: A member asked how to compare two **XLA HLO files** to identify optimizations like op fusion or performance improvements.
   - A member suggested asking on the GPU Mode discord server, and added that it's full of crazy good ML systems people.
- **RL might not scale**: A member shared a [paper arguing that RL currently is only reinforcing existing reasoning pipelines](https://arxiv.org/abs/2504.13837) and not really creating new capabilities.
   - They expressed skepticism about current DRL algorithms, suggesting that the closest thing to learning few-shot like a human is an LLM.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1370483433893265478)** (18 messages🔥): 

> `Physics of LLMs, ICML tutorial, Interpretability for AI safety, Interpretable-by-design architecture` 


- **Physics of LLMs program computational demands seem modest**: The computational demands for the [physics of LLMs program](https://arxiv.org/abs/2309.14316) seem modest, costing less than **$500** to train a **0.3B parameter model** for **300 epochs** on **100k samples**.
   - One member noted basic confusions about LLMs and that the research wasn't very "physics-like", lacking "mechanisms" or explanation/theory behind why that might be happening.
- **Synthetic Data Experiments**: For Part 1, fresh samples from **CFG datasets** were used, totaling **4.9 billion tokens** (**96 × 512 × 100k**).
   - For Parts 2 and 3, data sizes varied, with Part 3 generating profiles for **100,000 individuals** distilled into **six sentences** and concatenated to form sequences of **512 tokens**.
- **Interpretability for AI Safety: Omitted Variables**: Discussion about interpretability to detect safety problems resulting from omitted variables in models and how this might affect AI safety.
   - One member wondered *"how we would possibly hope to detect unsafety due to omitted variables"* in neural networks.
- **Interpretable-by-Design Architecture**: A member shared a [LessWrong post](https://www.lesswrong.com/posts/kjL9req2p79nSNe5H/interpretable-by-design-constraint-sets-with-disjoint-limit) on the relevant math for an interpretable-by-default architecture.
   - Another member reacted positively, stating that it *"Looks really cool, thanks!"*


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1370951735996190853)** (5 messages): 

> `o3 Performance Degradation, Global MMLU Inconsistencies` 


- **Performance Diff on o3 Models Debated**: Members discussed performance differences between the current generation **o3** model and the **o3-2025-04-16** model.
   - One user noted that the **o3** model's performance had degraded recently and switched back to **o1-pro**.
- **Inconsistencies Plague Global MMLU**: Members found that expected answers in **Global MMLU** (on lm-eval-harness) ideally supposed to be same across languages, have inconsistencies, especially in Korean (ko) and Chinese (zh).
   - The inconsistencies were present even for questions not marked as culturally sensitive.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1370554398182347005)** (4 messages): 

> `Disable Telemetry, H100 Backend, GPU/CPU Info` 


- **Telemetry Troubles?**: A user inquired about disabling telemetry following [the official documentation](https://docs.modular.com/mojo/faq/#does-the-mojo-sdk-collect-telemetry), noting the modular CLI mentioned is no longer available.
   - A member provided [a link to GitHub issue #3560](https://github.com/modular/modular/issues/3560#issuecomment-2833649340) offering a potential solution.
- **Backend Blueprint Hunt**: A user asked for insights into how Modular built the **H100 backend**, aiming to assess the feasibility of creating a backend for another accelerator.
   - They want to trace how they built the h100 backend, and they consider it a *noob question*.
- **System Snoop Command?**: A user sought a command to display **GPU/CPU info**, similar to `/proc/cpuinfo` or `nvidia-smi`.
   - A user mentioned a [grayscale example](https://github.com/modular/modular/blob/0bfe79b5bb8e5333203166540668eb6bdf104f9c/examples/gpu_functions/grayscale.mojo#L33).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1370617215195287612)** (59 messages🔥🔥): 

> `Autotuning removal, Post-hoc trait conformance, BigInt support, Mojo DataFrames, Mojo JIT compilation` 


- ****Autotuning** Functionality **Removed** from Mojo**: The autotuning feature was **removed** from the language because it *didn't work as well* as being in the library and was *overly complicated*.
   - The team is planning to add **extensions**, similar to Swift, for **post-hoc trait conformance**.
- **Tracing H100 Backend Build for Mojo**: A member inquired about the best way to trace how the **H100 backend** was built, to assess the feasibility of building a backend for another accelerator, recommending the [Modular Forum](https://forum.modular.com/t/how-to-approach-adding-a-backend/1419) as the best place to ask questions.
   - Another member found that **Dict.find()** is the fastest method for error checking in modelling, as shown in [benchmarks](https://github.com/lewisfogden/benchmarks/tree/main/actumojo).
- **Using **Mojo** for **bare metal systems programming****: One member expressed enthusiasm for **Mojo**'s potential for **bare metal systems programming**, particularly the ability to emit **ASM** and intrinsics.
   - Another member asked about exposing compiler flags for producing **no-stdlib, no-runtime binaries** suitable as a bootable kernel and it was recommended that they ask on the [forum](https://forum.modular.com/).
- **Arxiv paper surfaces on Mojo Dataframes**: There was a link to the [Mojo DataFrames paper](https://arxiv.org/abs/2505.04080) with a question about why the authors haven't been talking about it.
   - A user believed everything in Mojo is technically JIT compiled, and then another user pointed out the ORC JIT is used to create static binaries, except for GPUs.
- **Inference issues**: A member was confused about why this wasn't inferring: *cannot implicitly convert `f32[names, layout, storage]` value to `f32[names, layout, _origin]`*
   - Another member pointed out there are two different origins to infer, `origins` and `_origins`.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1370485542109708319)** (15 messages🔥): 

> `Modular meta-package, MAX graph in Mojo, Custom ops documentation, MAX Mojo APIs open-sourced, Progressively larger tutorials for max graph` 


- **Modular is a Meta-Package for UV or PIP**: `modular` is a meta-package currently only available for `uv` or `pip`, and the `max-pipelines` conda package is ~equivalent in Conda land, requiring specific libc version settings to function correctly.
   - To resolve wheel finding issues, the user set the `libc` version in system requirements to `2.34` in their `tool.pixi.system-requirements`.
- **MAX Graphs Go Custom, No More Mojo**: With the deprecation of the MAX API mojo packages, the recommended approach to run a MAX graph from Mojo involves [custom ops](https://docs.modular.com/max/custom-ops/) as the Mojo-based Graph API models had decayed and were non-functional.
   - Links to previous documentation are now broken, and a team member has been notified to address the issue.
- **MAX Mojo APIs already OSS'd**: Despite concerns about the MAX Mojo APIs' future, they were [already open-sourced](https://github.com/modular/modular/commit/34c0209cd537f1d5ab635166358a1713728d8f6f) even though they were removed due to being non-functional.
   - All of `max.graph`, `max.driver`, `max.tensor`, and their tests are available in that commit, and a user migration code for the tensor types is planned.
- **More Graph Max Examples Requested**: Users are requesting progressively larger tutorials for MAX graph, citing that [current examples](https://discord.com/channels/1087530497313357884/1371576991513448508/1371596539130155069) are too basic, and that for now it is like a black box with a couple of examples.
   - Reference was made to many full architectures which have been significantly expanded [here](https://github.com/modular/modular/tree/main/max/pipelines/architectures).


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

complete: Curious on thoughts on implementing DSPy with this https://arxiv.org/abs/2505.03335v2
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1371130736844734485)** (39 messages🔥): 

> `DSPy Doctrine, RL based on embedding models with DSPY, Prompts as weights, AI in Insurance conference presentation using DSPy, DSPy and LangGraph` 


- ****DSPy Doctrine** Messily Launched**: A member shared a very long and messy [X post](https://x.com/lateinteraction/status/1921565300690149759) describing the design decisions behind **DSPy**, envisioning it as a future formal "DSPy Doctrine."
- **DSPy powers **AI in Insurance****: A member presented a use case at an **AI in Insurance** conference in Germany, detailing how **DSPy** was used for correspondence template improvement, using it first as a prompt structuring vehicle, and later to optimize the prompts, with slides available in [German](https://drive.proton.me/urls/Y6YQC1EQ7G#ddX424butZxG) and [English](https://drive.proton.me/urls/M8W9PBX5KC#jXgLvgypyp3R).
- ****DSPy** vs **LangGraph**: Clash of the Frameworks?**: In a discussion about **DSPy** and agentic frameworks like **Autogen** or **LangGraph**, a member asked whether to build abstractions using **DSPy** primitives or plug **DSPy** into those frameworks, with one member stating that *anything you can do with LangGraph you can do with DSPy*.
- ****Docstring** Optimization Encouraged**: Members discussed the possibility of optimizing docstrings with **DSPy**, with one sharing a part of the docs stating that it is encouraged to optimize the signature by optimizing the metric for the signature.
- ****Async LLM** Support Rolling Out**: Members discussed updates on **async LLM** call support in **DSPy** to improve parallel tasks, but the **async LLM** call support will not be there for complex functionality like Refine.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1370510217279701067)** (35 messages🔥): 

> `Qwen3 Support, LLM applications, Nvidia & AMD hardware pricing, Image generation model in GPT4ALL, GPT4ALL installation help` 


- **Qwen3 Integration Queries Surface**: Members discussed about when **Qwen3** will be supported within **GPT4ALL**.
   - Users were directed to use **koboldcpp** or **sillytavern** instead.
- **Engineers Explore Engineering LLM Apps**: Members discussed using **LLMs** to create **Python scripts** for generating **PDF** and **PNG** images for structure plans, also, creating images with gaming company brands.
   - One user reported being invited to a company to discuss AI after creating an impressive company logo.
- **GPT4All Installation Issues Plague New Users**: A user reported that **GPT4ALL** was not starting and posted an image of the error.
   - Other members asked if they downloaded the right version for their OS (**Linux/Mac/PC**) and that their **CPU** needs to support **AVX** or **AVX2** instructions or an **Nvidia RTX** card is required.
- **Creative Writers Seek Model Recommendations**: A user with an **i9 11900k**, **128GB** memory, and **RTX3060 12GB** asked for the best model for creative writing.
   - Suggested models included **GLM-4-9B-0414-Q8_0.gguf** and **Qwen3-8B**; a link to a [benchmark leaderboard](https://huggingface.co/spaces/OpenEvals/find-a-leaderboard) was also shared.
- **GPT4All Development Stalled?**: A user inquired whether **GPT4All** is still being actively developed.
   - In response to the user wanting to run a custom **llama.cpp server**, he was directed to use the "custom" option in GPT4All's remote model providers page.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1370558121747021895)** (13 messages🔥): 

> `ROCm build on Mac, Tinybox Sales Internship in San Diego, Tinygrad backend for AI, LeetGPU adds Tinygrad support, Optimal kernel block size calculation` 


- **Seeking ROCm Savior for Mac Builds**: A member is seeking assistance to fix the **ROCm (comgr)** build for **Mac**, pointing to the failure in CI ([amdcomgr_dylib](https://github.com/tinygrad/amdcomgr_dylib)).
- **Tinybox Sales Internship Beckons in San Diego**: An internship/job is offered in San Diego, involving managing sales and inventory of **Tinybox parts**, requiring general intelligence and computer building experience, but no coding skills.
   - The goal is to capitalize on the potential sales of **Tinybox v2** and streamline supplier onboarding for larger buyers.
- **LeetGPU embraces Tinygrad**: [LeetGPU](https://leetgpu.com) now supports **Tinygrad** in its challenges, inviting users to check it out.
- **Prime Intellect releases Prime Intellect 2**: **Prime Intellect** releases [Prime Intellect 2](https://www.primeintellect.ai/blog/intellect-2-release).
- **Next Tinygrad meeting scheduled**: Tinygrad meeting #70 is scheduled for **Monday** at **9am** San Diego time.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1371166829158076456)** (5 messages): 

> `Tinygrad performance on T4, tinypilot chatbot, Max tensor numel query` 


- **Tinygrad Matrix Multiply on T4 Underperforms PyTorch**: A user reported that a matrix multiplication operation `A.matmul(B)` with shapes **A:(8192, 6144)** and **B:(6144, 4096)** takes around **500ms** on a **T4** GPU in tinygrad, while the same operation in PyTorch takes only ~**90ms**.
   - After syncing the device and calling `C.realize()`, the user is seeking advice on whether they are doing something wrong to cause such a performance gap.
- **tinypilot chatbot is born!**: A user introduced **tinypilot**, [a chatbot agent](https://github.com/NinoRisteski/tinypilot) designed to help users learn tinygrad.
   - It pulls the latest tinygrad repo, the mesozoic tutorials, and the latest bounties, and uses an open API model to provide explanations.
- **Request to Query Max Tensor Size**: A user asked if there is a way to query the max supported tensor numel for a given device/backend.
   - They are using an older OpenCL implementation that doesn't support the `long long` data type for indexing buffers, and want to add a condition in their code to account for this possibility.


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1371609333980336249)** (1 messages): 

> `PapersChat, Deep Research Agent, Multilingual RAG, Invoice Reconciliation Agent, LlamaParse updates` 


- **LlamaIndex releases PapersChat!**: LlamaIndex highlights the release of [PapersChat](https://t.co/ASzjwLfKLC), an agentic AI application that lets you chat with your papers and gather information from **Arxiv** and **PubMed**.
- **Deep Research Agents get Tutorial!**: LlamaIndex released a tutorial, [Build Your Own Deep Research Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A), walking through how to build one.
- **Multilingual RAG System is live!**: LlamaIndex announces the launch of the [Multilingual, Multimodal RAG System](https://t.co/69CHCCn8J3).
- **Invoice Reconciliation Agent using LlamaIndex.TS!**: Users can learn how to build an invoice reconciliation agent using **LlamaIndex.TS** and **LlamaCloud** following [this tutorial](https://www.youtube.com/watch?v=SzVsuMBcv5g).
- **LlamaParse gets New Models and Auto Orientation!**: **LlamaParse** receives new models and auto orientation detection; details are available [here](https://t.co/tqi17dPJm4).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1371522740510785689)** (1 messages): 

> `LlamaIndex, Finance, NYC Workshop` 


- **Top Minds to Meet in NYC Workshop**: LlamaIndex CEO @jerryjliu0 is holding an in-person workshop in NYC in two weeks, bringing together **200+ top thinkers** shaping the future of finance.
   - Registration can be found at [this link](https://t.co/NMpm9KkzWl).
- **NYC Finance Workshop**: An in-person workshop in NYC hosted by LlamaIndex CEO @jerryjliu0 aims to assemble the top thinkers in finance.
   - The workshop is scheduled to occur in two weeks, providing a platform for decision-makers and builders to connect.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1370689868098703543)** (4 messages): 

> `LlamaIndex Data Loaders vs Data Movement Tools, Customized Data Loaders, Fine-tuning mdcdse-2b` 


- **Debate data loaders vs data movers erupts**: A member asked about using data movement tools like **Airbyte** & **Meltano** to ingest data from multiple systems into a data warehouse, then using **LlamaIndex** for transformations, and shared a [Reddit post on r/LlamaIndex](https://www.reddit.com/r/LlamaIndex/comments/1kj5ym9/llamaindex_data_loaders_vs_data_movement_tools/) on the topic.
   - They mentioned forking and modifying **Llama-Index data loaders** several times to customize them, seeking alternatives.
- **LlamaIndex Integrations get boost**: A LlamaIndex team member suggested that if someone consistently customizes data loaders, it would be worth creating a custom integration and putting it on **LlamaHub** for others to use.
   - They encouraged contributions via pull requests.
- **mdcdse-2b training faces roadblocks**: A member who read a blog post from Logan and Marco on **Hugging Face** about **mdcdse-2b** wants to fine-tune it on another language using their dataset of image-query pairs.
   - They stated that training with the **DSE** ([https://github.com/texttron/tevatron/tree/main/examples/dse/qwen](https://github.com/texttron/tevatron/tree/main/examples/dse/qwen)) isn't working and is not easy on their consumer GPU, seeking guidance on fine-tuning with the specified approach.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1371230779128152107)** (1 messages): 

> `MOOC Deadlines, Certificate Requirements` 


- **MOOC Coursework Deadline Approaches!**: The deadline for all coursework for the Advanced LLM Agents MOOC is **May 31st at 11:59pm PDT**, according to an announcement.
   - Details about the [coursework and certificate requirements](https://docs.google.com/document/d/1t4HS15dySldeElgDbtKmM3piInllq_auHtpO-SRScjo/edit?usp=sharing) are available on the bottom of the [MOOC website](https://llmagents-learning.org/sp25).
- **Certificate Attainment: A Trilogy of Tasks**: To earn a certificate, participants must complete all coursework for their desired tier by **May 31st**, ensuring they receive a Google Forms confirmation email upon successful submission.
   - Additionally, they need to complete the [Certificate Declaration Form](https://forms.gle/iPA2MUpHdtrBE1vu5) by the same deadline and certificates will be dispatched via email in June.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1370508974671269918)** (3 messages): 

> `Coursework Deadline, AgentX Judging, Homework Verification` 


- **Coursework Submission Window Closing Soon**: The final deadline for all coursework submissions is **May 31st**, after which **AgentX** judging will commence throughout June.
   - Certificates for **Trailblazer/Mastery/Honorary Tiers** may be released early in June, while **Ninja/Legendary Tier** certificates will be released in August, following the AgentX judging conclusion.
- **AgentX Judging Approaching**: Judging for **AgentX** (ninja/legendary tiers) will occur throughout June after the coursework deadline on May 31st.
   - The **Ninja/Legendary Tier certificates** release is dependent on the completion of AgentX judging, while other certificates may be released earlier in June.
- **Students Ask for Homework Check Trick**: A student asked if the only way to check homework submissions is by searching for **Google Forms** in their email.
   - The instructor confirmed that checking emails for Google Form confirmations is indeed the way to verify homework submissions.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1370618197166002270)** (2 messages): 

> `AI Learning Resources, Best AI Courses` 


- **Inquiring the Best AI Course for Learning**: A member asked which course is best for **learning AI**.
- **Recommendations for AI Education**: The user is seeking guidance on the **best AI course** to begin their AI learning journey.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1370534177207877642)** (2 messages): 

> `Token Prepending, Azure SDK Ticket` 


- **Token Prepending Boosts Accuracy**: The language model pretends some tokens to the input, which let the model know what kind of input it is, during training they do the same prepending for higher accuracy for that mode.
- **Azure SDK Ticket Lodged**: A member created a ticket on **azure-sdk-for-python** about a potential issue on [GitHub](https://github.com/azure/azure-sdk-for-python/issues/41001).


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1371042825977860209)** (3 messages): 

> `Product Evolve, Canadian-hosted models, RAG capabilities, GenAI experiences, voice and chat agents` 


- **Product Evolve Founder Joins Discord!**: Saurabh, founder of [Product Evolve](https://www.productevolve.com/), a software consulting company based in Toronto, introduces himself to the Cohere Discord community.
   - He specializes in building **AI-powered solutions** for small businesses, financial institutions, and public sector organizations.
- **Canadian Models to be used for GenAI!**: Saurabh shows interest in how **Cohere's Canadian-hosted models** and **RAG capabilities** can be used to create secure, localized **GenAI experiences** for voice and chat agents.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1370890517784559616)** (3 messages): 

> `Anthropic, Claude, Updates` 


- **Anthropic and Claude updates incoming**: A member said they would write some updates on **Anthropic** and **Claude**.
   - They shared a link to a [page with more details](https://anthropic.swoogo.com/codewithclauderegister/faqs) but noted that there aren't many details there yet.
- **More Anthropic and Claude updates incoming**: Another member said they would write some updates on **Anthropic** and **Claude**.
   - They shared a link to a [page with more details](https://anthropic.swoogo.com/codewithclauderegister/faqs) but noted that there aren't many details there yet.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1370575720019001455)** (3 messages): 

> `OptinBwd, Llama Tokenizer` 


- **OptinBwd rewritten for drop-in replacement**: A contributor has rewritten **OptinBwd** over the weekend to be a drop-in replacement for any **optimizer**.
   - The contributor seeks feedback on the [pull request](https://github.com/pytorch/torchtune/pull/2719) before proceeding with further testing, noting that it currently cannot be combined with important features like **gradient accumulation** and **gradient clipping**.
- **Llama3.1 tokenizer overwrites original tokenizer order?**: A member inquired if the **llama3.1 tokenizer** used for **3.3 training** could potentially overwrite the original tokenizer order.
   - They pointed to a specific token in the [tokenizer file](https://github.com/pytorch/torchtune/blob/main/torchtune/models/llama3/_tokenizer.py#L34C39-L34C45) for reference.

