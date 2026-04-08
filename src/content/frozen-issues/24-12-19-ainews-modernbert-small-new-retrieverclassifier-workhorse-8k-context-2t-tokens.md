---
id: 65a7ef70-6cea-4cb1-a00a-26d16f926569
title: 'ModernBert: small new Retriever/Classifier workhorse, 8k context, 2T tokens, '
date: '2024-12-20T03:27:55.084640Z'
original_slug: ainews-modernbert-small-new-retrieverclassifier
description: >-
  **Answer.ai/LightOn** released **ModernBERT**, an updated encoder-only model
  with **8k token context**, trained on **2 trillion tokens** including code,
  with **139M/395M parameters** and state-of-the-art performance on retrieval,
  NLU, and code tasks. It features **Alternating Attention** layers mixing
  global and local attention. **Gemini 2.0 Flash Thinking** debuted as #1 in
  Chatbot Arena, and the **O1 model** scored top in reasoning benchmarks.
  **Llama** downloads surpassed **650 million**, doubling in 3 months.
  **OpenAI** launched desktop app integrations with voice capabilities.
  **Figure** delivered its first humanoid robots commercially. Advances in
  robotics simulation and a new physics engine **Genesis** claiming **430,000x
  faster than real-time** were highlighted.
companies:
  - answerdotai
  - lightonio
  - hugging-face
  - google-deepmind
  - openai
  - meta-ai-fair
  - figure
models:
  - modernbert
  - gemini-2.0-flash-thinking
  - o1
  - llama
topics:
  - encoder-only-models
  - long-context
  - alternating-attention
  - natural-language-understanding
  - reasoning
  - robotics-simulation
  - physics-engine
  - humanoid-robots
  - model-performance
  - model-releases
people:
  - jeremyphoward
  - alec-radford
  - philschmid
  - drjimfan
  - bindureddy
---


<!-- buttondown-editor-mode: plaintext -->**Encoder-only models are all you need.**

> AI News for 12/18/2024-12/19/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**215** channels, and **4745** messages) for you. Estimated reading time saved (at 200wpm): **440 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

As [he has been teasing](https://www.latent.space/p/answerai) for a few months, Jeremy Howard and the Answer.ai/LightOn team [released ModernBert today](https://x.com/jeremyphoward/status/1869786023963832509?s=46), updating the classic BERT from 2018:

![image.png](https://assets.buttondown.email/images/5b764d0e-7bc1-48a7-b422-f0f0f5bbf3bd.png?w=960&fit=max)

The [HuggingFace blogpost](https://huggingface.co/blog/modernbert) goes into more detail on why this is useful:

- **Context**: Old BERTS had ~500 token context; ModernBERT has 8k
- **Data**: Old BERTS were on older/less data; ModernBERT was trained on 2T, including "a large amount of code"
- **Size**: LLMs these days are >70B, with the requisite cost and latency issues; ModernBERT is 139M (base)/395M (large) params
- **SOTA perf for size**: beats regular Kaggle winners like DeBERTaV3 on all retrieval/NLU/code categories ![image.png](https://assets.buttondown.email/images/c482db4e-ec78-450f-ae5b-6adcf65faf1d.png?w=960&fit=max)
- **Real world variable length long context**: input sizes vary in the real world, so that’s the performance we worked hard to optimise – the “variable” column. As you can see, for variable length inputs, ModernBERT is much faster than all other models. ![image.png](https://assets.buttondown.email/images/7d022871-e0e3-4747-983e-ac08212db3e9.png?w=960&fit=max)
- **Bidirectional**: Decoder-only models are specifically constrained against "looking ahead", whereas BERTS can fill in the blanks:

```py
import torch
from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "fill-mask",
    model="answerdotai/ModernBERT-base",
    torch_dtype=torch.bfloat16,
)

input_text = "One thing I really like about the [MASK] newsletter is its ability to summarize the entire AI universe in one email, consistently, over time. Don't love the occasional multiple sends tho but I hear they are fixing it."
results = pipe(input_text)
pprint(results)
```

One of the MANY interesting details disclosed in [the paper](https://arxiv.org/pdf/2412.13663) is the **Alternating Attention** layers - mixing global and local attention in the same way Noam Shazeer did at Character ([our coverage here](https://buttondown.com/ainews/archive/ainews-shazeer-et-al-2024/)):

![image.png](https://assets.buttondown.email/images/84a44194-4862-4937-97cb-d0fc54187399.png?w=960&fit=max)


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

**AI Model Releases and Performance**

- [@drjwrae announced](https://twitter.com/drjwrae/status/1869806618025832788) the release of **Gemini 2.0 Flash Thinking**, built on their 2.0 Flash model for improved reasoning
- [@lmarena_ai reported](https://twitter.com/lmarena_ai/status/1869793847548817563) that **Gemini-2.0-Flash-Thinking** debuted as #1 across all categories in Chatbot Arena
- [@bindureddy noted](https://twitter.com/bindureddy/status/1869542214734663795) that the new **O1 model** scores 91.58 in Reasoning and is #1 on Livebench AI
- [@answerdotai and @LightOnIO released](https://twitter.com/reach_vb/status/1869791808030708054) **ModernBERT** with up to 8,192 tokens context length and improved performance

**Major Company News**

- [@AIatMeta shared](https://twitter.com/AIatMeta/status/1869775975917257037) that **Llama has been downloaded over 650M times**, doubling in 3 months
- [@OpenAI launched](https://twitter.com/gdb/status/1869811511616778280) desktop app integrations with apps like Xcode, Warp, Notion and voice capabilities
- [@adcock_brett announced](https://twitter.com/adcock_brett/status/1869863378975658441) that **Figure** delivered their first humanoid robots to commercial clients
- [Alec Radford's departure](https://twitter.com/iScienceLuvr/status/1869852854728700166) from OpenAI was announced

**Technical Developments**

- [@DrJimFan discussed](https://twitter.com/DrJimFan/status/1869795912597549137) advances in robotics simulation, highlighting trends in massive parallelization and generative graphics
- [@_philschmid shared](https://twitter.com/_philschmid/status/1869639246434246966) details about **Genesis**, a new physics engine claiming 430,000x faster than real-time simulation
- [@krandiash outlined](https://twitter.com/krandiash/status/1869828879856349488) challenges in extending context windows and memory in AI models

**Memes and Humor**

- [@AmandaAskell joked](https://twitter.com/AmandaAskell/status/1869584124627066977) about species procreating via FOMO
- [@_jasonwei shared](https://twitter.com/_jasonwei/status/1869618956333645940) getting roasted by his girlfriend comparing his talks to scenes from Arrival
- [@karpathy posted](https://twitter.com/karpathy/status/1869522720377221291) about his daily PiOclock tradition of taking photos at 3:14pm

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Bamba: Inference Efficient Hybrid Mamba2 Model**

- **[Bamba: Inference-Efficient Hybrid Mamba2 Model 🐍](https://huggingface.co/blog/bamba)** ([Score: 60, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1hhodui/bamba_inferenceefficient_hybrid_mamba2_model/)): **Bamba** is an **inference-efficient hybrid model** based on **Mamba2**. The post title suggests a focus on performance gaps and new benchmarks related to this model, though no further details are provided in the body.
  - **Benchmark Gaps**: Discussions highlight that the **Bamba** model shows gaps in math benchmarks, similar to other linear models, due to the training data and the inclusion of benchmark-aligned instruction datasets during training phases. A specific example mentioned is the improvement in the **GSM8k score** from **36.77 to 60.0** by adding **metamath** data.
  - **Openness in Methodology**: Commenters appreciate the transparency in the training and quantization processes of the **Bamba** model, expressing enthusiasm for the forthcoming paper that promises detailed insights into data sources, ratios, and ablation techniques.
  - **Model Naming Humor**: There is a lighthearted exchange about the naming convention of models like **Bamba**, **Zamba**, and others, with links provided to related papers and models on **Hugging Face** ([Zamba-7B-v1](https://huggingface.co/Zyphra/Zamba-7B-v1), [Jamba](https://huggingface.co/papers/2403.19887), [Samba](https://huggingface.co/papers/2406.07522)).


**Theme 2. Genesis: Generative Physics Engine Breakthrough**

- **[New physics AI is absolutely insane (opensource)](https://v.redd.it/15c7r7rjxq7e1)** ([Score: 1350, Comments: 147](https://reddit.com/r/LocalLLaMA/comments/1hhmebr/new_physics_ai_is_absolutely_insane_opensource/)): The post discusses an **open-source physics AI** called **Genesis**, highlighting its impressive generative and physics engine capabilities. The lack of a detailed text description suggests that the video linked may provide further insights into its functionalities and applications.
  - **Skepticism and Concerns**: Many commenters express skepticism about the project, comparing it to other hyped technologies like **Theranos** and **Juicero**, and suggesting that the affiliations and "open-source" claims may be overstated. **MayorWolf** and others doubt the authenticity of the video, suggesting it involves creative editing and that the open-source aspect may be limited to what's already available in tools like **Blender**.
  - **Technical Discussion**: Some users discuss the technical aspects, such as the use of **Taichi** for efficient GPU simulations, and the potential similarities to **Nvidia's Omniverse**. **AwesomeDragon97** notes a flaw in the simulation regarding water droplet adhesion, indicating the need for further refinement in the physics engine.
  - **Project Legitimacy**: Links to the project's [website](https://genesis-embodied-ai.github.io/) and [GitHub repository](https://github.com/Genesis-Embodied-AI/Genesis) are shared, with some users noting the involvement of top universities and suggesting it could be legitimate. Others, like **Same_Leadership_6238**, highlight that while it may seem too good to be true, it is open source and warrants further investigation.


- **Genesis project: a generative physics engine able to generate 4D dynamical worlds powered by a physics simulation platform** ([Score: 103, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1hhl1m0/genesis_project_a_generative_physics_engine_able/)): The **Genesis project** introduces a **generative physics engine** capable of creating **4D dynamical worlds** using a physics simulation platform, developed over 24 months with contributions from over 20 research labs. This engine, written in pure Python, operates **10-80x faster** than existing GPU-accelerated stacks and offers significant advancements in simulation speed, being **~430,000 times faster than real-time**. It is open-source and aims to autonomously generate complex physical worlds for robotics and physical AI applications.
  - **Generative physics engine** allows for simulations where robots, including soft robots, can experiment and refine their actions far faster than real-world trials, potentially revolutionizing robotics and physical AI applications.
  - The **impact on simulations and animations** is substantial, enabling individuals with access to consumer-grade hardware like an **NVIDIA 4090** to train robots for real-world applications, which was previously limited to entities with significant resources.
  - Skepticism exists about the technology's capabilities due to its impressive claims, with users expressing a desire to personally test the engine to validate its performance.


**Theme 3. Slim-Llama ASIC Processor's Efficiency Leap**

- **[Slim-Llama is an LLM ASIC processor that can tackle 3-bllion parameters while sipping only 4.69mW - and we'll find out more on this potential AI game changer very soon](https://www.techradar.com/pro/slim-llama-is-an-llm-asic-processor-that-can-tackle-3-bllion-parameters-while-sipping-only-4-69mw-and-we-shall-find-out-more-about-this-potential-ai-game-changer-in-february-2025)** ([Score: 240, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1hhn2r0/slimllama_is_an_llm_asic_processor_that_can/)): **Slim-Llama** is an **LLM ASIC processor** capable of handling **3 billion parameters** while consuming only **4.69mW** of power. More details about this potentially significant advancement in AI hardware are expected to be revealed soon.
  - There is skepticism about the **Slim-Llama's** performance, with concerns over its **3000ms latency** and the practicality of its **5 TOPS at 1.3 TOPS/W** power efficiency. Critics argue that the **500KB memory** is insufficient for running a **1B model** without external memory, which would increase energy consumption ([source](http://ssl.kaist.ac.kr/bbs/board.php?bo_table=HI_systems&wr_id=39)).
  - The **Slim-Llama** supports only **1 and 1.5-bit models** and is seen as an academic curiosity rather than a practical solution, with potential applications in **wearables**, **IoT sensor nodes**, and energy-efficient **industrial applications** due to its low power consumption of **4.69mW**. Some commenters express hope for future use cases with improved **4-bit quantization** and better software support.
  - Discussion includes the chip's **20.25mm² die area** using **Samsung's 28nm CMOS technology**, with curiosity about its potential performance on more advanced processes like **5nm or 3nm**. There is also playful banter about running **Enterprise Resource Planning** simulations on the "**SLUT**-based BMM core," highlighting the chip's novelty and niche appeal.


**Theme 4. Gemini 2.0 Flash Thinking Experimental Release**

- **[Gemini 2.0 Flash Thinking Experimental now available free (10 RPM 1500 req/day) in Google AI Studio](https://i.redd.it/xbibsmke7u7e1.png)** ([Score: 73, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1hhxkyk/gemini_20_flash_thinking_experimental_now/)): **Gemini 2.0 Flash Thinking Experimental** is now available for free in **Google AI Studio**, allowing users 10 requests per minute and 1500 requests per day. The interface includes system instructions for answering queries like "who are you now?" and allows adjustments in model selection, token count, and temperature settings.
  - A user humorously described a **thinking process** example where the model counted the occurrences of "r" in "strawberry" but noted a misspelling, highlighting the model's step-by-step reasoning.
  - There is curiosity about the potential to utilize the output from **Gemini 2.0 Flash Thinking** for training additional thinking models, suggesting interest in model improvement and development.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Gemini 2.0 Flash Thinking released, outperforming older models**

- **Gemini 2.0 Flash Thinking (reasoning, FREE)** ([Score: 268, Comments: 95](https://reddit.com/r/OpenAI/comments/1hhygng/gemini_20_flash_thinking_reasoning_free/)): **Gemini 2.0 Flash**, a reasoning model by **Google**, is now available for free at [aistudio.google.com](http://aistudio.google.com), offering up to **1500 free requests per day** with a **2024 knowledge cutoff**. The author finds it impressive, particularly for its ability to be steered via system prompts, and notes that it performs on par or better than **OpenAI's GPT-3.5** for tasks like image processing, general questions, and math, criticizing the cost and limitations of OpenAI's offering.
  - Users are impressed with **Gemini 2.0 Flash's** performance, noting its superiority in **math** compared to other models and its ability to display its reasoning process, which some find remarkable. There is a general sentiment that it outperforms **OpenAI's offerings**, with users questioning the value of paying for **ChatGPT Plus**.
  - There is a discussion on **Google's strategic advantage** due to their cost-effective infrastructure, specifically their **TPUs**, which allows them to offer the model for free, in contrast to **OpenAI's** expensive and closed models. This cost advantage is seen as a potential long-term win for Google in the AI space.
  - Some users express a desire for **improved UI/UX** in Google's AI products, suggesting that a more user-friendly interface could enhance their appeal. The conversation also touches on the absence of web search capabilities in Gemini, and the potential for custom instructions in AI Studio, which enhances user control over the model's responses.


- **[O1's full LiveBench results are now up, and they're pretty impressive.](https://i.redd.it/uyqg3gekap7e1.png)** ([Score: 267, Comments: 85](https://reddit.com/r/OpenAI/comments/1hhgd0v/o1s_full_livebench_results_are_now_up_and_theyre/)): **OpenAI's "o1-2024-12-17" model** leads in the **LiveBench results**, showing superior performance particularly in **Reasoning** and **Global Average** scores. The table compares several models across metrics like **Coding**, **Mathematics**, and **Language**, with competitors from **Google**, **Alibaba**, and **Anthropic**.
  - There is significant discussion about the **O1 model's pricing and performance**. Some users argue that O1 is more expensive than Opus due to "invisible thought tokens", leading to a cost of over **$200 per mTok output**, while others claim the price is the same but costs accumulate due to reasoning tokens ([source](https://livebench.ai)).
  - **O1's capabilities and access** are debated, with some noting that the O1 Pro API isn't available yet and that the current O1 model uses a "reasoning_effort" parameter, which affects its performance and pricing. This parameter indicates that O1 Pro might be a more advanced version with higher reasoning effort.
  - **Comparisons with other models** like **Gemini 2.0 Flash** are prevalent, with Gemini noted for its cost-effectiveness and potential for scaling up. Some speculate that Gemini's efficiency is due to Google's TPU resources, and there's optimism about future advancements leading to "in-the-box-AGI" within 1-2 years.


- **[The AI race over time by Artificial Analysis](https://i.redd.it/280qbvkqqo7e1.jpeg)** ([Score: 157, Comments: 12](https://reddit.com/r/OpenAI/comments/1hhdzhd/the_ai_race_over_time_by_artificial_analysis/)): The report from **Artificial Analysis** provides a comprehensive overview of the AI race, focusing on the evolution of AI language models from **OpenAI, Anthropic, Google, Mistral,** and **Meta**. A line graph illustrates the "Frontier Language Model Intelligence" over time, using the "Artificial Analysis Quality Index" to compare model quality from **Q4 2022 to Q2 2025**, highlighting trends and advancements in AI development. [Full report here](https://artificialanalysis.ai/downloads/ai-review/2024/Artificial-Analysis-AI-Review-2024-Highlights.pdf).
  - **Gemini 2.0** is considered superior to the current **GPT-4o model** in all aspects, and it is available for free on **Google AI Studio**.
  - There is a correction regarding the timeline: **GPT-3.5 Turbo** was not available in **2022**; instead, **GPT-3.5 Legacy** was available during that period.


**Theme 2. NotebookLM incorporates interactive podcast feature**

- **Notebook LM interaction BETA. MindBlown.** ([Score: 272, Comments: 69](https://reddit.com/r/OpenAI/comments/1hhlsyx/notebook_lm_interaction_beta_mindblown/)): **Google** has quietly activated an **interaction feature** in **NotebookLM**, allowing users to interact with generated podcasts. The post expresses excitement over this new capability, describing it as "mindblowing."
  - Users discussed the **interaction feature** in **NotebookLM**, noting that it allows real-time conversation with AI about uploaded source material. However, the interaction remains surface-level, and users expressed a desire for deeper conversational capabilities and better prompt responses compared to **ChatGPT**.
  - The feature requires creating a new notebook and adding sources to generate an audio overview. Interaction begins after the audio is ready, but some users noted it lacks the ability to save or download the interacted podcast, and availability may vary by region.
  - There is a mixed reaction to **Google**'s advancements in AI, with some users expressing skepticism about Google's position in the AI race and others noting the feature's utility for studying, while comparisons were made to **OpenAI**'s recent updates, which some found underwhelming.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. Fierce Model Wars and Bold Price Cuts**  

- [**Gemini 2.0 Lights Up the Stage**](https://x.com/NoamShazeer/status/1869789881637200228): Users praise *“Gemini 2.0 Flash Thinking”* for displaying explicit chain-of-thought and beating older models in reasoning tasks. Several tests, including [lmarena.ai’s mention](https://x.com/lmarena_ai/status/1869793847548817563), show it topping performance leaderboards with public excitement.  
- [**OpenRouter Slashes Prices in Epic Showdown**](https://openrouter.ai/gryphe/mythomax-l2-13b): Providers like **MythoMax** and **QwQ** cut costs by over 7%, with [mistralai/mistral-nemo](https://openrouter.ai/mistralai/mistral-nemo) reducing 12.5%. Observers call it *“ongoing price wars”* as AI providers compete for user adoption.  
- [**Databricks Gobbles $10B for Growth**](https://www.databricks.com/company/newsroom/press-releases/databricks-raising-10b-series-j-investment-62b-valuation): The company raised a colossal round at a stunning $62B valuation, with plans to exceed $3B revenue run rate. Stakeholders link this surge to soaring enterprise AI demands and 60% annual growth.

**Theme 2. Multi-GPU and Fine-Tuning Frenzy**  

- [**Unsloth Preps GPU Magic**](https://unsloth.ai/blog/llama3-3): Multi-GPU support lands in Q1, with the team testing enterprise pricing and sales revamps. They confirm Llama 3.3 needs around 41GB VRAM to fine-tune properly.  
- [**SGLang vs. vLLM in a Performance Duel**](https://lmsys.org/blog/2024-12-04-sglang-v0-4): vLLM wins for raw throughput, while SGLang excels in structured outputs and scheduling. Engineers weigh trade-offs, citing SGLang’s flexible modular approach for certain workflows.  
- [**Quantization Saves the Day**](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu): Threads tout 4-bit or 8-bit quantization to shrink memory footprints. Contributors highlight *“RAG plus quantization”* as an efficient path for resource-limited tasks.

**Theme 3. Agents, RAG, and RLHF Breakthroughs**  

- [**Agentic Systems Race Ahead**](https://www.anthropic.com/research/building-effective-agents): Anthropic’s *“year of agentic systems”* blueprint outlines composable patterns, fueling speculation of major leaps by 2025. Researchers compare these designs to classical search and note how open thinking patterns can surpass naive retrieval.  
- [**Asynchronous RLHF Powers Faster Training**](https://arxiv.org/abs/2410.18252): A paper proposes off-policy RLHF, decoupling generation and learning to speed up language model refinement. The community debates *“how much off-policyness can we tolerate?”* in pursuit of efficiency.  
- [**Multi-Agent LlamaIndex Unleashes RAG**](https://t.co/lbhFDbSabS): Developers shift from single to multi-agent setups, each focusing on a specialized subtask for robust retrieval-augmented generation. They use agent factories to coordinate tasks and ensure better coverage over large corpora.

**Theme 4. AI Tools for Coding Take Center Stage**  

- [**Cursor 0.44.4 Upgrades**](https://www.cursor.com/downloads): The launch introduced “Yolo mode” and improved agent commands, touted in the [changelog](https://www.cursor.com/changelog). Early adopters noticed faster code edits and better task handling in large projects.  
- [**GitHub Copilot Chat Goes Free**](https://x.com/code/status/1869449373995708703): Microsoft announced a no-credit-card-needed tier that even taps *“Claude for better capabilities.”* Devs cheer cost-free real-time code suggestions, although some still prefer old-school diff editing for version control.  
- [**Windsurf vs. Cursor Showdown**](https://www.builder.io/blog/windsurf-vs-cursor): Users compare collaborative editing, large-file handling, and performance. Many mention Cursor’s consistency for complex refactors, while some appreciate Windsurf’s flexible UI for smaller tasks.

**Theme 5. Fresh Libraries and Open-Source Adventures**  

- [**Genesis AI Conjures Physics Realities**](https://x.com/zhou_xian_/status/1869511650782658846): A new generative engine simulates 4D worlds *430,000 times faster than real-time*. Robotics fans marvel at 26-second training runs on an RTX4090, showcased in the [Genesis-Embodied-AI/Genesis repo](https://github.com/Genesis-Embodied-AI/Genesis).  
- [**ModernBERT Takes a Bow**](https://huggingface.co/blog/modernbert): This *“workhorse model”* offers extended context and improved classification or retrieval over older BERT. Community testers confirm better performance and simpler optimization for RAG workflows.  
- [**Nomic Maps Data in the Browser**](https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers): The final post in their Data Mapping Series shows how scalable embeddings and dimensionality reduction democratize massive dataset visualization. Readers laud it as a game-changer for exploratory analysis.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Preps Multi-GPU Magic**: Multi-GPU support for **Unsloth** is slated for Q1, and the team is fine-tuning pricing details alongside final tests.
   - They also hinted at revamping **sales processes** for enterprise interest, though their enterprise beta remains in a testing phase.
- **Llama 3.3 Ramps Up Power**: The **Llama 3.3** model demands roughly 41GB of VRAM to fine-tune, as noted in [Unsloth’s blog](https://unsloth.ai/blog/llama3-3).
   - Participants reported higher performance in contrast to earlier versions, pointing to the benefits of careful training cycles on large datasets.
- **SGLang vs. vLLM: The Speed Showdown**: Many agreed **vLLM** outpaces **SGLang** for hefty production tasks, but [SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4) looks promising for structured outputs and scheduling tricks.
   - Community members consider **vLLM** stronger for throughput, while SGLang appeals to those optimizing modular results.
- **RAG Meets Quantization**: **Retrieval-Augmented Generation (RAG)** appeared as a smarter alternative to direct fine-tuning when resources are tight, often employing chunked data and embeddings for context retrieval.
   - Users praised **quantization** (see [Transformers Docs](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu)) to shrink memory footprints without completely sacrificing performance.
- **LoRAs, Merging & Instruction Tuning Warnings**: Combining **Low Rank Adapters (LoRAs)** with base models, possibly saved as [GGUF options](https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf), requires careful parameter balancing to avoid unwanted distortions.
   - An [instruction tuning paper](https://arxiv.org/abs/2402.05119) highlighted how partial training can degrade core knowledge, underscoring the hazards of merging multiple techniques without thorough validation.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 0.44.4 Launches with Agent Boost**: Cursor **0.44.4** introduced improved agent features, **Yolo mode**, and is [available here](https://www.cursor.com/downloads).
   - Engineers applauded its faster command execution and better task handling, citing the [changelog](https://www.cursor.com/changelog) for a detailed breakdown.
- **Coin Toss: O1 vs Sonnet 3.5**: Users pinned **O1** at around 40 cents per request and compared its gains to **Sonnet 3.5**. 
   - Some considered Sonnet 3.5 'good enough,' while others questioned if O1's extra cost is worth the difference.
- **Build It Up: Framer vs. DIY Code**: A lively discussion contrasted **Framer** for rapid site creation with fully custom code. 
   - Some praised the time savings, while others preferred complete control over performance and flexibility.
- **Gemini-1206 Gains Curiosity**: Members showed interest in **Gemini-1206**, but concrete evidence of its abilities remains scarce. 
   - Others stayed focused on **Sonnet 3.5** for coding, since they lacked extensive data on Gemini-1206.
- **College or Startup: The Great Showdown**: Some argued **Ivy League** credentials offer networking perks, while others favored skipping school to build real-world products. 
   - Opinions varied, with personal success stories suggesting any path can yield major breakthroughs.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Cline & Gemini Triumph Together**: Multiple members praised [Cline v3](https://x.com/sdrzn/status/1869470308442452478) combined with **Gemini 2.0** for smoother coding and large-task handling.
   - They noted that it outperformed other setups, mainly due to faster iterations and more stable refactoring capabilities.
- **Windsurf vs Cursor Showdown**: Comparisons referenced [this direct breakdown](https://www.builder.io/blog/windsurf-vs-cursor) on features like collaborative editing and file handling.
   - Opinions seemed divided, but many cited Cursor's more consistent performance as a critical advantage in code-heavy workflows.
- **Credit Rollover Reassurance**: Users confirmed **flex credits roll over** in [Codeium’s paid plan](https://docs.codeium.com/windsurf/usage#what-happens-when-you-run-out-of-premium-user-prompt-credits-but-not-premium-flow-action-credits), ensuring no sudden interruptions.
   - Some participants shared relief about not losing credits after paying, highlighting the importance of stable subscription models.
- **Claude vs Gemini Model Chatter**: Community members weighed performance differences between **Claude Sonnet**, **Gemini**, and other AI models while referencing [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards).
   - They stressed the need for contextual prompts and thorough documentation to fully leverage each model's coding potential.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 2.0 Flashes a 'Think Out Loud' Trick**: Google introduced [Gemini 2.0 Flash Thinking](https://x.com/NoamShazeer/status/1869789881637200228), an experimental model that trains **explicit chain-of-thought** for enhanced reasoning and speed in chatbot tasks.
   - Community members referenced [Denny Zhou's stance on classical AI reliance on search](https://x.com/denny_zhou/status/1869771028693713284), hinting that **Gemini**'s open thinking pattern might surpass naive retrieval solutions.
- **OpenAI Sings with Voice Mode**: OpenAI rolled out **Work with Apps** in **voice mode**, enabling integration with apps like **Notion** and **Apple Notes** as teased on their [12 Days of ChatGPT site](https://openai.com/12-days/?day=11).
   - Members called this a straightforward but major step in bridging **ChatGPT** with real-world productivity, with some hoping advanced voice features could power daily tasks.
- **Chollet's 'o1 Tiff' Rattles LLM Circles**: François Chollet equated labeling **o1** as an LLM to calling **AlphaGo** 'a convnet', inciting heated arguments on [X](https://x.com/fchollet/status/1869612195425972715).
   - Community members noted this parallels the old *Subbarao/Miles Brundage incident*, with calls for clarity on **o1**'s architecture fueling further drama.
- **FineMath: Gigantic Gains for LLM Arithmetic**: A link from [@anton_lozhkov](https://x.com/anton_lozhkov/status/1869771053146464507) showcased **FineMath**, a math-focused dataset with over **50B+ tokens**, promising boosts over conventional corpora.
   - Participants saw this as a big leap for complex code math tasks, referencing *merging FineMath with mainstream pretraining* to handle advanced calculations.
- **RLHF Book: Spot a Typo, Score Free Copies**: An **RLHF** resource was mentioned to be on GitHub, where volunteers who catch typos or formatting bugs qualify for free copies of the book.
   - Eager contributors found it *less stressful* to refine **reinforcement learning** fundamentals this way, calling the process both fun and beneficial for the community.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Day 11 of OpenAI Delivers ChatGPT Boost**: Day 11 of the **12 Days of OpenAI** introduces a new approach for **ChatGPT**, featuring a [YouTube live session](https://www.youtube.com/live/g_qxoznfa7E?si=q71WyiHuioBSGzvz) that highlights advanced code collaboration.
   - Engineers can now broaden daily development cycles with AI assistance, although **manual copy actions** remain necessary.
- **ChatGPT Integrates with XCode**: Participants discussed copying code from **ChatGPT** straight into **XCode**, smoothing iOS dev tasks.
   - This step promises convenience but still depends on user-initiated triggers for actual code insertion.
- **Google’s Gemini 2.0 Hits the Spotlight**: Google published the **Gemini 2.0 Flash Thinking** experimental model, attracting curiosity with bold performance claims.
   - Some participants doubted the model’s reliability after it stumbled on **letter-count tasks**, fueling skepticism about its real prowess.
- **YouTube Clone Demo with ChatGPT**: Members explored using **ChatGPT** to craft a YouTube-like experience, covering front-end and back-end solutions.
   - Though front-end tasks seemed straightforward, the server-side setup demanded more steps through terminal instructions.
- **AI Automation Heats Up the Engineering Floor**: Conversations centered on the prospect of AI fully automating software development, reshaping the demand for human engineers.
   - While many recognized potential time-savings, others wondered if hype was outpacing actual breakthroughs.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **FSDP vs Tensor Parallel Tangle**: At Eleuther, participants compared **Fully Sharded Data Parallel (FSDP)** to **Tensor Parallelism**, referencing [llama-recipes](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/config_utils.py#L95) for real-world implementations.
   - They argued about higher communication overhead in FSDP and weighed that against the direct parallel ops advantage of tensor-based methods, with some voicing concern about multi-node scaling limits.
- **NaturalAttention Nudges Adam**: A member highlighted a new **Natural Attention Optimizer** on [GitHub](https://github.com/jeroaranda/naturalattention) that modifies Adam with attention-based gradient adjustments, backed by proofs in [Natural_attention_proofs.pdf](https://github.com/jeroaranda/naturalattention/blob/main/papers/Natural_attention_proofs.pdf).
   - They claimed notable performance gains, though some cited potential bugs in the code at [natural_attention.py](https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py) and suggested caution when replicating results.
- **Diffusion vs Autoregressive Arm-Wrestle**: A discussion emerged contrasting **diffusion** and **autoregressive** models across image and text domains, highlighting efficiency tradeoffs and discrete data handling.
   - Some posited that diffusion leads in image generation but might be challenged by autoregressive approaches in tasks that require token-level control.
- **Koopman Commotion in NNs**: Members debated applying **Koopman theory** to neural networks, referencing [Time-Delay Observables for Koopman: Theory and Applications](https://arxiv.org/abs/1810.01479) and [Learning Invariant Subspaces of Koopman Operators--Part 1](https://arxiv.org/abs/2212.07358).
   - They questioned the legitimacy of forcing Koopman methods onto standard frameworks, suggesting it might mislead researchers if underlying math doesn't align with real-world activation behaviors.
- **Steered Sparse AE OOD Queries**: In interpretability discussions, enthusiasts explored **steered sparse autoencoders (SAE)** and whether cosine similarity checks on reconstructed centroids effectively gauge out-of-distribution data.
   - They reported that adjusting one activation often influenced others, indicating strong interdependence and prompting caution in interpreting SAE-based OOD scores.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Referral Program Boosts Sign-Ups**: Multiple users confirmed that [Perplexity offers a referral program](https://www.perplexity.ai/settings/account) granting benefits for those who link up with new sign-ups.
   - Enthusiasts aim to recruit entire fraternities, accelerating platform reach and energizing discussions about user growth.
- **You.com Imitation Raises Accuracy Concerns**: Community members discussed **You.com** replicating responses with search-based system instructions, questioning the quality of its output.
   - They noted that relying on direct model calls often produces more precise logic, revealing potential gaps in search-oriented Q&A solutions.
- **Game Descriptions Overwhelm Translation Limits**: A user attempting to convert lengthy lists to French encountered size restrictions, showing **Perplexity AI**'s text-handling constraints.
   - They sought advice on segmenting content into smaller chunks, hoping to bypass these limitations in complex translation tasks.
- **Magic Spell Hypothesis Sparks Curiosity**: A posted [document](https://www.perplexity.ai/page/the-magic-spell-hypothesis-n5tkbs1JR4OGww9A25c9ZA) described the **Magic Spell Hypothesis**, linking advanced linguistic patterns to emerging concepts in scientific circles.
   - Researchers and community members weighed its credibility, applauding attempts to test fringe theories in structured experiments.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini Gains Ground**: On 12/19, **Gemini 2.0 Flash Thinking** emerged with the `gemini-2.0-flash-thinking-exp-1219` variant, touting better reasoning in agentic workflows as shown in [Jeff Dean's tweet](https://x.com/JeffDean/status/1869789813232341267).
   - Initial tests revealed faster performance than O1 and deepseek, and some community members applauded its upgraded output quality.
- **Aider & MCP Get Cozy**: Users achieved **Aider** and **MCP** integration for streamlined Jira tasks, referencing [Sentry Integration Server - MCP Server Integration](https://mcpserver.cloud/server/server-sentry).
   - They discussed substituting Sonnet with other models in MCP setups, suggesting top-notch flexibility for error tracking and workflow automation.
- **OpenAPI Twinning Madness**: Community members explored running **QwQ** on Hugging Face alongside local **Ollama**, clarifying that Hugging Face mandates its own API usage for seamless model switching.
   - They discovered the need to indicate the service in the model name, preventing confusion in multi-API setups.
- **Copilot Chat Spices Up**: **GitHub Copilot Chat** introduced a free immersive mode as stated in [GitHub's announcement](https://github.blog/changelog/2024-12-18-announcing-github-copilot-free/), offering real-time code interactions and sharper multi-file edits.
   - While users appreciated the enhanced chat interface, some still preferred old-school **diff edits** to contain costs and maintain predictable workflows.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt & Supabase Spark Instant Setup**: The **Bolt** & **Supabase** integration is officially live, offering simpler one-click connections as shown in [this tweet from StackBlitz](https://x.com/stackblitz/status/1869715661444043245). It eliminates the manual steps, letting engineers unify services more quickly and reduce overhead.
   - Users praised the easy setup, noting how it shortens ramp-up time for data-driven applications and provides a frictionless developer experience.
- **Figma Frustrations & .env Woes**: Users reported **.env file** resets that disrupt Firebase configurations, with locking attempts failing after refresh and causing 'This project exceeds the total supported prompt size' errors.
   - Additionally, **Figma** direct uploads are off the table, forcing designers to rely on screenshots while requesting more robust design-to-dev integrations.
- **Redundancy Rehab & Public Folder Setup**: Community members asked if **Bolt** could analyze code for redundant blocks, aiming to cut token use in large-scale apps. They also needed clarity on building a **public folder** to host images, highlighting confusion about project structure.
   - Some suggested straightforward docs to resolve folder-setup uncertainties, indicating a desire for simpler references when working with Bolt.
- **Session Snafus & Token Tangles**: Frequent session timeouts and forced page refreshes left many losing chat histories in **Bolt**, driving up frustration and token costs. The dev team is investigating these authentication issues, but real-time disruptions persist.
   - Users hope for fixes that reduce redundant outputs and control overspending on tokens, seeking stability in their project workflows.
- **Community Convergence for Guides & Integrations**: Participants plan a broader guide for **Bolt**, providing a user dashboard for submitting and approving resources. The conversation touched on **Stripe** integration, advanced token handling, and synergy with multiple tech stacks.
   - They also showcased [Wiser - Knowledge Sharing Platform](https://boltwiser.levyandco.net/), hinting at deeper expansions for shared content and more polished developer experiences.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Interactive Mode Reaches Everyone**: The development team confirmed **Interactive Mode** reached 100% of users with notable improvements for audio overviews.
   - Enthusiasts praised the creative possibilities and shared firsthand experiences of smoother deployment.
- **MySQL Database Hook for Automatic NPCs**: A game master asked how to connect a large **MySQL** database to **NotebookLM** for automating non-player character responses.
   - They highlighted a decade of stored RPG data and sought ideas for managing dynamic queries.
- **Podcasters Tweak Recording Setup**: Members debated how the interactive podcast feature does not store conversations, forcing separate audio capture for external listeners.
   - A concise 'podcast style prompt' sparked interest in faster, more candid commentary for a **QWQ model** review.
- **AI-Generated Space Vlog Shakes Viewers**: A user showcased a year-long astronaut isolation vlog rendered by AI, linked at [this YouTube link](https://youtu.be/_ys7FchEkak?feature=shared).
   - Others noted daily animated uploads driven by NotebookML outputs, demonstrating consistent content production.
- **Updated UI Gains Kudos**: Users applauded the **NotebookLM** interface overhaul, describing it as more receptive and convenient for project navigation.
   - They are eager to test its new layouts and praised the overall polished look.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Ubuntu Steps for SDXL**: Some members shared tips for running **SDXL** on **Ubuntu**, advising the use of shell scripts from [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/webui-user.sh) for streamlined setups.
   - They underlined the importance of system knowledge to avoid performance bottlenecks.
- **ComfyUI Meltdown**: Engineers complained about persistent errors and charred output from **ComfyUI** despite attempts to fix sampling issues.
   - They recommended using **Euler** sampling with well-tuned denoising levels to reduce flawed results.
- **AI Images Face Rocky Road to Perfection**: Some argued **AI-generated images** and **video** won't be flawless by 2030 due to current challenges.
   - Others countered that rapid technological leaps could deliver polished outputs much sooner.
- **Quantum Quarrel Over P=NP**: A heated chat focused on **quantum computing** relevance if **P=NP** becomes reality.
   - Skeptics pointed to trouble extracting real-world value from quantum states, citing complexity in practical execution.
- **Civitai.com Down Again?**: Multiple users noted frequent outages on **civitai.com**, making model access challenging.
   - They speculated recurring server problems are behind the repeated downtime.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU Glitter & Coil Whine**: Users complained about *absurd coil whine* from a returned **RX 6750XT**, plus **VRChat**'s memory appetite prompting some to choose **4090s**.
   - They also expressed worry about potentially bigger price tags for next-gen **RTX 50** cards while comparing the **7900 XTX**.
- **Triton Tinkers with AMD**: Community members tested **Triton** kernels on **AMD GPUs** like the **RX 7900**, noting performance still lags behind **PyTorch/rocBLAS**.
   - They also discovered that **warp-specialization** was removed in **Triton 3.x**, driving them to explore alternative optimizations.
- **CARLA Zooms into UE 5.5**: **CARLA version 0.10.0** introduced [Unreal Engine 5.5 features](https://carla.org/2024/12/19/release-0.10.0/) like **Lumen** and **Nanite**, boosting environment realism.
   - Attendees also praised [Genesis AI](https://genesis-embodied-ai.github.io/) for its water droplet demos, envisioning synergy with **Sim2Real** and referencing [Waymo's synthetic data approach](https://waymo.com/research/embedding-synthetic-off-policy-experience-for-autonomous-driving-via-zero/) for autonomous driving.
- **MatX's HPC Hiring Spree**: **MatX** announced open roles for **low-level compute kernel authors** and **ML performance engineers**, aiming to build an **LLM accelerator ASIC**.
   - The [job listing](https://grnh.se/2b337cb08us) emphasizes a high-trust environment that favors *bold design decisions* over extended testing.
- **Alma's 40-Option Benchmark Bash**: A duo released **alma**, a Python package checking the throughput of over **40 PyTorch conversion options** in a single function call.
   - According to [GitHub](https://github.com/saifhaq/alma), it gracefully handles failures with *isolated processes* and will expand into *JAX* and *llama.cpp* soon.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Agents Amp Up**: Anthropic posted [Building effective agents](https://www.anthropic.com/research/building-effective-agents) with patterns for **AI agentic systems**, anticipating a major milestone in 2025.
   - They emphasized composable workflows, referencing a tweet about the 'year of agentic systems' for advanced design.
- **Gemini 2.0 Gains Speed**: Multiple tweets, including [lmarena.ai's mention](https://x.com/lmarena_ai/status/1869793847548817563) and [Noam Shazeer's announcement](https://x.com/NoamShazeer/status/1869789881637200228), praised **Gemini 2.0 Flash Thinking** for topping all categories.
   - The model trains to 'think out loud', enabling stronger reasoning and outdoing earlier Gemini versions.
- **Databricks Hauls $10B**: They announced a [Series J funding round](https://www.databricks.com/company/newsroom/press-releases/databricks-raising-10b-series-j-investment-62b-valuation) worth **$10B**, hitting a **$62B** valuation with Thrive Capital leading.
   - They anticipate crossing **$3B** in revenue run rate, reporting 60% growth sparked by **AI** demand.
- **ModernBERT Steps Onstage**: A new model called **ModernBERT** [was introduced](https://huggingface.co/blog/modernbert) as a 'workhorse' option with extended context and improved performance.
   - References like [Jeremy Howard's mention](https://x.com/jeremyphoward/status/1869786023963832509) show attempts to apply it in retrieval and classification, spurring conversation among practitioners.
- **Radford Says Farewell to OpenAI**: Alec Radford, credited for the original GPT paper, [left OpenAI](https://x.com/steph_palazzolo/status/1869848094009110826) to pursue independent research.
   - This shift stirred speculation about **OpenAI**'s upcoming directions in the industry.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter’s Vision Variation**: OpenInterpreter 1.0 now includes **vision** support, with an installation path via [GitHub](https://github.com/OpenInterpreter/open-interpreter.git) and pip install git+https://github.com/OpenInterpreter/open-interpreter.git@development.
   - Experiments suggest the `--tools gui` command functions properly for bridging different models or APIs, with people noting local or SSH-based usage.
- **Server Mode Sparks Execution Queries**: Members questioned how **server mode** handles command execution, debating whether tasks run locally or on the server.
   - They mentioned using SSH for simpler interaction and proposed a front end for improved workflow.
- **Google Gemini 2.0 Gains Attention**: A user showed interest in **Google Gemini 2.0** for multimodal tasks within OS mode, hoping for highly capable command execution.
   - They compared it to existing setups and wondered if it competes effectively with other systems.
- **Cleaning Installs & O1 Confusion**: Some users faced issues with **OpenInterpreter** installation after multiple configurations, prompting them to remove flags for a new setup.
   - Meanwhile, an O1 channel user complained about unclear docs, seeking direct guidance even after reading official references.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Safetensors Snafu Stumps LM Studio**: Users encountered **Safetensors header is unexpectedly large: bytes=2199142139136** errors when loading models, forcing redownloads of the **MLX version of Llama 3.3** to fix possible corruption issues.
   - Discussions mentioned conflict with file compatibility, with some users suggesting a careful file check for future downloads.
- **Mobile Dreams: iOS Gains Chat, Android Waits**: An iOS app called **3Sparks Chat** ([link](https://apps.apple.com/us/app/3sparks-chat/id6736871168)) connects to LM Studio on Mac or PC, providing a handheld interface for local LLMs.
   - Members expressed disappointment about the lack of an Android client, leaving the community requesting alternative solutions.
- **AMD's 24.12.1 Distress**: The **AMD 24.12.1** drivers triggered system stuttering and performance loss when loading models with LM Studio, connecting to llama.cpp rocm libraries.
   - Downgrading drivers resolved problems in some setups, and references to the **7900XTX** GPU emerged as a concern for stability.
- **Vision Model Hopes in LM Studio**: A query about **image input models** led to mention of [mlx-community/Llama-3.2-11B-Vision-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit), highlighting early attempts at integrating visual features.
   - Users reported loading problems on Windows, fueling debate about model compatibility with local hardware.
- **Apple Silicon vs. 4090 GPU Showdown**: Community members questioned if **Mac Pro and Ultra chips** outperform a **30 or 4090** card due to memory bandwidth advantages.
   - Benchmark references pointed to the [llama.cpp GitHub discussion](https://github.com/ggerganov/llama.cpp/discussions/4167), where users confirmed the 4090 still holds faster metrics in practical tests.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Price Cuts Rattle the LLM Market**: This morning saw a 7% cut for [gryphe/mythomax-l2-13b](https://openrouter.ai/gryphe/mythomax-l2-13b), 7.7% for [qwen/qwq-32b-preview](https://openrouter.ai/qwen/qwq-32b-preview), and a 12.5% slash on [mistralai/mistral-nemo](https://openrouter.ai/mistralai/mistral-nemo).
   - Community members joked about *'ongoing price wars'* fueling competition among providers.
- **Crowdsourced AI Stack Gains Spotlight**: VC firms have released various ecosystem maps, but there's a push for a truly **crowdsourced** and **open-source** approach showcased in [this GitHub project](https://github.com/daytonaio/ai-enablement-stack).
   - One user requested feedback on the proposed logic, encouraging the community to *'contribute to a dynamic developer resource'*.
- **DeepSeek Speeds Code Learning**: Developers used **DeepSeek V2** and **DeepSeek V2.5** to parse entire GitHub repositories, reporting significant improvements in project-wide optimization.
   - However, a user cautioned that *'it may not handle advanced code generation'*, and they still praised its annotation abilities.
- **Calls for Programmatic API Keys**: A discussion emerged about allowing a **provider API key** to be sent implicitly with requests, streamlining integration.
   - One user said *'I'd love to see a programmatic version'* to enhance developer convenience across the board.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GitHub Copilot Goes Free**: Microsoft introduced a new [free tier for GitHub Copilot](https://x.com/code/status/1869449373995708703) with immediate availability for all users.
   - It surprisingly includes **Claude** for improved capabilities, and no credit card is required.
- **Granite 3.1-8B-Instruct Gains Fans**: Developers praised the [Granite 3.1-8B-Instruct model](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) for strong performance on long context tasks.
   - It provides quick results for real-world cases, and IBM offers code resources on [GitHub](https://github.com/ibm-granite/granite-3.1-language-models).
- **LM Studio Enables Local LLM Choices**: [LM Studio](https://lmstudio.ai/) simplifies running Llama, Mistral, or Qwen models offline while supporting file downloads from Hugging Face.
   - Users can also chat with documents quickly, appealing to folks wanting an offline approach.
- **Fine-Tuning Uniform Instruction Sparks Debate**: Questions arose about using the same instruction for every prompt in a Q&A dataset.
   - A caution was raised that it might cause **suboptimal model performance** due to repetitive usage.
- **Genesis Project Roars with Generative Physics**: The [Genesis engine](https://x.com/zhou_xian_/status/1869511650782658846) builds **4D dynamical worlds** at speeds up to 430,000 times faster than real-time.
   - It's [open source](https://github.com/Genesis-Embodied-AI/Genesis), runs in Python, and slashes robotic training to just 26 seconds on a single RTX4090.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Negative Indexing Showdown in Mojo**: A heated discussion emerged about adopting negative indexing in **Mojo**, with some calling it an error magnet while others see it as standard practice in **Python**.
   - Opponents favored a `.last()` approach to dodge overhead, warning of performance issues with negative offsets.
- **SIMD Key Crash Rumbles in Dicts**: A serious bug in **SIMD-based** struct keys triggered segmentation faults in **Dict** usage, detailed in [GitHub Issue #3781](https://github.com/modularml/mojo/issues/3781).
   - Absent **scaling_cur_freq** caused these crashes, prompting a fix within a **6-week** window.
- **Mojo Goes Rogue on Android**: Enthusiasts tinkered with running **Mojo** on native Android via Docker-based hacks, though it's labeled 'wildly unsupported'.
   - Licensing rules prevent publishing a Docker image, but local custom builds remain possible.
- **Python Integration Teases SIMD Support**: Participants discussed merging **SIMD** and conditional conformance with Python types, balancing separate handling for integral and floating-point data.
   - They highlighted **ABI** constraints and future bit-width expansions, stirring interest in cross-language interactions.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Synthetic Data Explainer Gains Steam**: One contributor is building an **explainer** on how synthetic data is generated, requesting community input on tricky areas.
   - They plan to highlight creation approaches and performance implications for advanced models.
- **DataBricks Rate-Limiting Debate**: Participants flagged big throughput charges, calling for a **rate limiter** to prevent overuse in DataBricks.
   - Some recommended the [LiteLLM proxy layer](https://example-url-for-lighter-llm.com) for usage tracking, also referencing [Mosaic AI Gateway](https://docs.databricks.com/en/ai-gateway/index.html) as a supplementary approach.
- **dspy.Signature as a Class**: A user asked about returning a **dspy.Signature** in class form, aiming for structured outputs over raw strings.
   - They hope to define explicit fields for clarity and potential type-checking.
- **Provisioned Throughput Shocks Wallet**: A conversation exposed high expense from **provisioned throughput** in DataBricks when it remains active.
   - Members advised the **scale to 0** feature to curb costs during idle periods.
- **LiteLLM Reaches DataBricks**: Attendees debated whether to embed the **LiteLLM proxy** within a DataBricks notebook or run it separately.
   - They agreed it's feasible to integrate both approaches, given environment controls and resource needs.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex's Multi-Agent Makeover**: A post described the jump from a **single agent** to a **multi-agent system** with practical code examples in **LlamaIndex**, referencing [this link](https://t.co/lbhFDbSabS).
   - It also clarifies how **agent factories** manage multiple tasks working in tandem.
- **Vectara's RAG Rally**: An update showcased **Vectara's** RAG strengths, including data loading and streaming-based queries, referencing [this link](https://t.co/traVaQiUt3).
   - It underscored agentic usage of RAG methods, with insights on reranking in a managed environment.
- **Vercel's AI Survey Shout-Out**: Community members were urged to fill out **Vercel's** State of AI Survey, found [here](https://t.co/O3sYZ6L9Gq).
   - They plan to gather data on developer experiences, challenges, and target areas for future AI improvements.
- **Vision Parse for PDF Power**: A new open-source Python library, [Vision Parse](https://github.com/iamarunbrahma/vision-parse), was introduced for converting PDF to well-structured markdown using advanced Vision Language Models.
   - Participants praised its potential to simplify document handling and welcomed open-source efforts for collective growth.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic's Data Mapping Marathon Ends**: The final installment of the **Data Mapping Series** spotlights **scalable graphics** for embeddings and unstructured data in [Nomic’s blog post](https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers).
   - This six-part saga demonstrates how techniques like **dimensionality reduction** empower users to visualize massive datasets within their web browsers.
- **BERT & GGUF Glitches Get Patched**: Users faced issues loading **Nomic’s BERT** embedding model from Huggingface after a commit broke functionality, but the fix is now live.
   - Community members also flagged chat template problems in **.GGUF** files, with updated versions promised in the upcoming release.
- **Code Interpreter & System Loader Shine**: A [pull request](https://github.com/nomic-ai/gpt4all/pull/3173) proposes a code interpreter tool built on the jinja template for running advanced code tasks.
   - Simultaneously, users requested a more convenient system message loader to bypass manual copy-pasting of extensive context files.
- **GPT4All Device Specs Confirmed**: A query about **GPT4All** system requirements led to a link detailing [hardware support](https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/system_requirements.md).
   - Important CPU, GPU, and memory details were highlighted to ensure a stable local LLM experience.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyChat Installation Tussle**: One user ran into problems setting up TinyChat, reporting missing pieces like **tiktoken** and a 30-second system freeze, plus a puzzling prompt about local network devices.
   - George Hotz spoke about writing a **tiktoken** alternative within TinyGrad and flagged **8GB RAM** as a constraint.
- **Mac Scroll Direction Goes Rogue**: A user complained that running TinyChat flipped the **scroll direction** on their Mac, then reverted once the app closed.
   - George Hotz called this behavior baffling, acknowledging it as a strange glitch.
- **Bounty Push and Layout Talk**: Contributors discussed **bounty** rewards to push tinygrad forward, stressing tests and improvements as key drivers.
   - A user mentioned the complexity of layout notation, linking to both [a view merges doc](https://docs.google.com/document/d/1RRuhAW-I5u_6ssbm1eLqVWSK_PYFoARIVBYwYZwmcM4/edit) and [viewable_tensor.py](https://github.com/unknownusername504/MicroGrad/blob/main/micrograd/tensors/viewable_tensor.py) for deeper context.
- **Scheduler Query in #learn-tinygrad**: A participant asked why the scheduler uses **realize** before expand or unsafe pad ops, with no clear explanation offered.
   - The group didn't fully unpack the reasoning, leaving the topic open for further exploration.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Ikuo Impresses & Etiquette Ensues**: Ikuo618 introduced himself with six years of experience in **DP**, **NLP**, and **CV**, spotlighting his **Python**, **TensorFlow**, and **PyTorch** skills.
   - A gentle reminder followed, advising members not to repost messages across channels for a cleaner conversation flow.
- **Platform Feature Question Marks**: A user asked about a feature's availability on the platform, and a member confirmed it's still not live.
   - The inquirer expressed thanks, ending on a positive note with a smiley face.
- **Cohere Keys & Rate Limits Exposed**: Cohere provides evaluation and production API keys, detailed on [the API keys page](https://dashboard.cohere.com/api-keys) and in the [pricing docs](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work).
   - Rate limits include **20 calls per minute** for trial and **500 per minute** for production on the Chat endpoint, with **Embed** and **Classify** sharing distinct quotas.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Teases Phi 4 & Roles**: In [the official Torchtune docs page](https://pytorch.org/torchtune/stable/api_ref_models.html), members confirmed that **Torchtune** only supports **Phi 3** but welcomes contributions for **Phi 4**.
   - They introduced a **Contributor** role on Discord and noted minimal differences between **Phi 3** and **Phi 4** to simplify new pull requests.
- **Asynchronous RLHF Races Ahead**: **Asynchronous RLHF** separates generation and learning for faster model training, detailed in [“Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models”](https://arxiv.org/abs/2410.18252).
   - The paper questions *how much off-policyness can we tolerate*, pushing for speed without sacrificing performance.
- **Post-Training Gains Momentum**: The [Allen AI blog](https://allenai.org/blog/tulu-3) highlights that **post-training** is crucial after pre-training to ensure models follow human instructions safely.
   - They outline instruction fine-tuning steps and focus on preserving capabilities such as intermediate reasoning while specializing.
- **Instruction Tuning Tightrope**: **InstructGPT**-style strategies can unwittingly diminish certain model abilities, especially if specialized tasks overshadow broader usage.
   - Retaining **coding** proficiency while handling *poetic or general instructions* emerged as the delicate balance to maintain.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents Hackathon Countdown**: The **submission deadline** for the hackathon is **12/19 at 11:59 PM PST**, with entries filed via the [official Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform).
   - The community is on standby for *last-minute fixes*, making sure everyone has a fair shot before the clock hits zero.
- **Support for Eleventh-Hour LLM Queries**: Participants can drop **last-minute questions** in the chat for quick feedback from peers.
   - Organizers urge coders to finalize checks promptly, avoiding *frantic merges* at the buzzer.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Axolotl AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LAION Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1319100971997073449)** (352 messages🔥🔥): 

> `Unsloth's Multi-GPU Support, Llama 3.3 Fine-Tuning, SGLang vs. vLLM, Sales Strategy, FFT Support` 


- **Unsloth's Multi-GPU Support in Q1**: Multi-GPU support for Unsloth is in the pipeline and expected to launch in Q1, with current testing underway.
   - The team is evaluating pricing and licensing as they work towards finalizing this feature.
- **Fine-Tuning Llama 3.3 Requirements**: Fine-tuning Llama 3.3 requires approximately 41GB of VRAM, as indicated on the Unsloth blog.
   - This model shows significant performance enhancements compared to previous versions when fine-tuned properly.
- **SGLang vs. vLLM Performance**: The community discussed SGLang and vLLM with the consensus that vLLM generally offers better throughput for production inference tasks.
   - SGLang is considered useful for structured outputs, while vLLM provides greater performance in other areas.
- **Sales Strategy and Product Availability**: There is a call for more streamlined sales processes for Unsloth, especially as interest in enterprise solutions grows.
   - While the enterprise product is still in beta, the team aims to gauge demand and adjust their sales approach accordingly.
- **FFT Engine Support**: FFT is not currently supported in Unsloth but can be implemented manually by users.
   - Discussion highlighted that utilizing FFT could provide significant performance improvements over other training engines.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2024-07-25-sglang-llama3/">Achieving Faster Open-Source Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) | LMSYS Org</a>: &lt;p&gt;At LMSYS.org, we&#x27;ve been running the &lt;a href=&quot;https://chat.lmsys.org/&quot;&gt;Chatbot Arena&lt;/a&gt; platform for over a year, serving millions of users. We know firs...</li><li><a href="https://lmsys.org/blog/2024-12-04-sglang-v0-4/">SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs | LMSYS Org</a>: &lt;p&gt;We’re excited to release &lt;a href=&quot;https://github.com/sgl-project/sglang&quot;&gt;SGLang v0.4&lt;/a&gt;, featuring significant performance improvements and new features:...</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://unsloth.ai/blog/llama3-3">Fine-tune Llama 3.3 with Unsloth</a>: Fine-tune Meta&#x27;s Llama 3.3 (70B) model which has better performance than GPT 4o, open-source 2x faster via Unsloth! Beginner friendly.Now with Apple&#x27;s Cut Cross Entropy algorithm.</li><li><a href="https://huggingface.co/tiiuae/Falcon3-10B-Instruct">tiiuae/Falcon3-10B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/rombodawg/Rombos-LLM-70b-Llama-3.3">rombodawg/Rombos-LLM-70b-Llama-3.3 · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct">unsloth/Llama-3.3-70B-Instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1319042969818632222)** (117 messages🔥🔥): 

> `Adapters vs Models, Fine-tuning Challenges, Learning Resources for Fine-tuning, Instruction Tuning Limitations, Model Merging Techniques` 


- **Understanding Adapters and Models**: Adapters, specifically Low Rank Adapters (LoRAs), modify a small subset of model parameters, allowing for flexible combinations without full model retraining.
   - To combine them, one can save models with [GGUF options](https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf) for simpler inference or manage them as separate files.
- **Navigating Fine-tuning Obstacles**: Fine-tuning isn't simply pressing a button; it requires understanding the underlying processes to avoid issues like catastrophic forgetting.
   - Members emphasized that successful fine-tuning depends on finding the right balance of adjustments and often involves multiple re-training cycles.
- **Recommended Learning Resources**: Resources like [DeepLearning.ai](https://www.deeplearning.ai/) and Hugging Face documentation are suggested for deeper learning on fine-tuning and model training.
   - Participants stressed the importance of a strong foundational understanding beyond just fine-tuning techniques.
- **Insights on Instruction Tuning**: An insightful paper highlighted that instruction tuning often fails to enhance model knowledge and can lead to knowledge degradation.
   - Members pointed out that dependency on external datasets can diminish response quality, thus reiterating the complexities involved in fine-tuning.
- **Exploring Model Merging Techniques**: Experimenting with merging models can yield mixed results, as maintaining balance is critical to overcoming trade-offs between various techniques.
   - Merging techniques, including combining base instructions and LoRA adjustments, require careful management to avoid common pitfalls like loss of accuracy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf">Saving to GGUF | Unsloth Documentation</a>: Saving models to 16bit for GGUF so you can use it for Ollama, Jan AI, Open WebUI and more!</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.05119">A Closer Look at the Limitations of Instruction Tuning</a>: Instruction Tuning (IT), the process of training large language models (LLMs) using instruction-response pairs, has emerged as the predominant method for transforming base pre-trained LLMs into open-d...</li><li><a href="https://civitai.com/models/555285/miyabi-hoshimi-zenless-zone-zero">Miyabi Hoshimi (星見雅) (星见雅) - Zenless Zone Zero (绝区零) (絕區零) (ゼンレスゾーンゼロ) - booru | Stable Diffusion LoRA | Civitai</a>: Support me on facebook.com/Kaiseir patreon.com/Serkai https://ko-fi.com/kaiseir Weight: 1.0 Trigger words: Appearance: miyabihoshimi, &amp;lt;lora:miya...</li><li><a href="https://youtu.be/3UQ7GY9hNwk?si=FdoeDFWvqVzv9TMY"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1319086227516817468)** (468 messages🔥🔥🔥): 

> `Fine-tuning LLMs, RAG (Retrieval-Augmented Generation), Quantization, Using Google Colab and Kaggle for model training, JSON data formatting for models` 


- **Challenges in Fine-tuning LLMs**: A user highlighted difficulties in fine-tuning LLMs due to hardware limitations, particularly using TinyLlama and struggling with a large dataset of 1GB in JSON format.
   - Despite the struggle, progress was made in fixing the environment and better understanding the training processes.
- **Introducing RAG for Enhanced Learning**: The importance of Retrieval-Augmented Generation (RAG) was emphasized as a potentially more effective method than direct fine-tuning, especially when using a smaller model for specific tasks.
   - Participants discussed using techniques like chunking data and embedding to improve model performance and reduce the complexity of initial training.
- **Quantization for Efficient Resource Usage**: Quantization techniques were discussed as a way to reduce memory and computational costs when training models, allowing for larger model sizes like 4-bit or 8-bit representations.
   - Users were advised to use proper quantization settings to avoid crashing their local machines during training.
- **Utilizing Online Platforms for Training**: Google Colab and Kaggle were recommended as alternatives for accessing GPU resources without significant expense, particularly for users with limited local compute capacity.
   - Despite resistance to using cloud platforms, participants acknowledged their utility for initial learning and model testing.
- **Navigating JSON Data Formatting**: Formatting JSON data correctly was identified as a critical step for successful model training, yet participants faced challenges with large datasets.
   - Improving the structure and formatting of JSON files was seen as necessary for utilizing RAG effectively and preparing for fine-tuning efforts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth? Start here!</li><li><a href="https://obsidian.md/">Obsidian - Sharpen your thinking</a>: Obsidian is the private and flexible note‑taking app that adapts to the way you think.</li><li><a href="https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md">mlx-examples/llms/mlx_lm/LORA.md at main · ml-explore/mlx-examples</a>: Examples in the MLX framework. Contribute to ml-explore/mlx-examples development by creating an account on GitHub.</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1319031452230221824)** (706 messages🔥🔥🔥): 

> `Cursor 0.44.4 Release, O1 vs Sonnet 3.5 Performance, Website Builders vs Custom Code, Gemini-1206 Capabilities, The Role of College for Startups` 


- **Cursor 0.44.4 Released**: The channel discussed the recent release of Cursor version 0.44.4, detailing several new features and improvements including agent enhancements and Yolo mode.
   - Users reported better performance with the agent in 0.44.4, noting its ability to run commands and handle tasks more efficiently.
- **Discussion on O1 and Sonnet 3.5**: The conversation centered around O1, which is priced at approximately 40 cents per request, and its value compared to Sonnet 3.5, with users sharing different opinions on their effectiveness.
   - Some users found Sonnet 3.5 to be sufficient for their needs, expressing skepticism about whether O1 justifies its cost.
- **Opinions on Website Development Tools**: A debate arose about the use of website builders like Framer versus coding from scratch, highlighting the trade-offs between time savings and cost.
   - While some appreciated the efficiency of website builders, others felt that custom coding offered more flexibility and control.
- **Capabilities of Gemini-1206**: Inquiries were made about users' experiences with Gemini-1206, with some expressing interest in its features and potential benefits.
   - However, others remained focused on the performance of established models like Sonnet 3.5 for coding tasks.
- **The Importance of College for Startups**: The discussion touched on the value of college education, particularly Ivy League schools, versus pursuing startup ventures.
   - Participants debated the necessity of formal education in a startup context, weighing it against practical experience and success.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>: no description found</li><li><a href="https://svelte.dev/docs/kit/introduction">Introduction • Docs • Svelte</a>: no description found</li><li><a href="https://www.cursor.com/downloads">Downloads | Cursor - The AI Code Editor</a>: Choose your platform to download the latest version of Cursor.</li><li><a href="https://svelte.dev">Svelte • Web development for the rest of us</a>: no description found</li><li><a href="https://docs.astral.sh/uv/">uv</a>: no description found</li><li><a href="https://forum.cursor.com/t/i-can-not-delete-history-of-composer-and-chat/36026">I can not delete history of composer and chat</a>: Describe the Bug I cannot delete the composer and chat history.  Steps to Reproduce Create a new composer and chat. The delete button only clears the chat history, not the entire chat. It cannot be de...</li><li><a href="https://simonwillison.net/2024/Dec/16/webdev-arena/">WebDev Arena</a>: New leaderboard from the [Chatbot Arena](https://lmarena.ai/) team (formerly known as LMSYS), this time focused on evaluating how good different models are at &quot;web development&quot; - though it t...</li><li><a href="https://x.com/_philschmid/status/1869639246434246966?s=46">Tweet from Philipp Schmid (@_philschmid)</a>: WTF?! New open-source physics AI engine absolutely insane! 🤯 Genesis is a new physics engine that combines ultra-fast simulation with generative capabilities to create dynamic 4D worlds for robotics ...</li><li><a href="https://x.com/btibor91/status/1869160332712960345">Tweet from Tibor Blaho (@btibor91)</a>: Here is a preview of the ChatGPT tasks and automations (&#34;jawbone&#34; - work-in-progress)Quoting Tibor Blaho (@btibor91) Remember &#34;Jawbone&#34;? It&#39;s the codename for ChatGPT &#34;Tasks&#3...</li><li><a href="https://github.com/richards199999/Thinking-Claude/blob/main/model_instructions/v5.1-extensive-20241201.md">Thinking-Claude/model_instructions/v5.1-extensive-20241201.md at main · richards199999/Thinking-Claude</a>: Let your Claude able to think. Contribute to richards199999/Thinking-Claude development by creating an account on GitHub.</li><li><a href="https://youtu.be/oFfVt3S51T4?si=MtUVbzYc6H231xyJ"> - YouTube</a>: no description found</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: New updates and improvements.</li><li><a href="https://cursor.directory/">Cursor Directory</a>: Find the best cursor rules for your framework and language
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1319049018655113227)** (65 messages🔥🔥): 

> `Flex credits rollover, Using repoprompt in Windows, Integrating features from GitHub, Codeium extension issues, Windsurf user experience` 


- **Flex credits rollover clarified**: A member confirmed that **flex credits roll over**, ensuring users retain their credits after payment.
   - This was corroborated by another who mentioned that their usage was reset upon making the payment.
- **Seeking repoprompt equivalent for Windows**: A user asked for an equivalent to **repoprompt** on Windows, showing interest in similar functionalities in their environment.
   - While no direct alternatives were provided, members encouraged exploring options and testing different setups.
- **Integrating GitHub features using ChatGPT**: A member expressed challenges in using ChatGPT to integrate a feature from one GitHub branch to another and inquired about available guides.
   - Suggestions included looking for specific YouTube channels and guides to facilitate the integration process.
- **Codeium extension issues in VSCode**: Users reported problems with the **Codeium extension** not supporting autocomplete in Jupyter notebooks within VSCode, unlike before.
   - One member also mentioned downgrading the extension to fix a server disconnection issue.
- **Frustrations with Windsurf user experience**: A user expressed frustration with Windsurf's handling of large files, specifically mentioning the consistent deletion of the same code lines.
   - They felt that the **cascade** feature needs improvement, but showed reluctance to submit a bug ticket through the official support channels.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1319044599091564596)** (509 messages🔥🔥🔥): 

> `Windsurf performance issues, Cline + Gemini usage, Codeium support and features, Model comparisons, Credit management in AI tools` 


- **Windsurf performance issues after updates**: Several users reported that Windsurf has become less functional in recent days, with issues such as editing files and frequent errors during use.
   - Users are increasingly frustrated with the software's reliability, prompting some to consider alternative tools like Cursor.
- **Positive feedback on Cline + Gemini usage**: Some users mentioned that using Cline with Gemini 2.0 leads to better coding results compared to Windsurf, with smoother functionality.
   - Users appreciated the efficiency of Cline, especially for tasks like refactoring and handling larger code without issues.
- **Inquiring about Codeium support and improvements**: Users expressed a desire for more responsive support from Codeium and reported the lack of recent updates or fixes for existing issues.
   - The community is keen on seeing improvements and features that align with current user needs for better functionality.
- **Comparisons of AI models and their effectiveness**: Discussions around the differences in performance between various models, including Claude Sonnet and Gemini, highlighted varying efficiencies for specific tasks.
   - Users noted the need for contextual information and proper documentation to enhance AI model utility.
- **Concerns regarding credit management and costs**: Concerns were raised about the consumption of credits in Windsurf and how it affects user experiences, especially with large tasks.
   - Users are evaluating the cost-effectiveness of different plans and the implications of credit usage with their AI tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/usage#what-happens-when-you-run-out-of-premium-user-prompt-credits-but-not-premium-flow-action-credits">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://x.com/sdrzn/status/1869470308442452478">Tweet from Saoud Rizwan (@sdrzn)</a>: Excited to share Cline v3 🎉 You can now auto-approve all actions, limit max # of API requests, and get desktop notifications for when a task is completed! Cline now also uses a flexible diff edit for...</li><li><a href="https://docs.codeium.com/windsurf/usage#what-happens-when-you-run-out-of-premium-user-prompt-credits">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://stateofai.tools/">State of AI Tools 2024 - Developer Survey</a>: Help shape the future of AI-accelerated development by sharing your experience.</li><li><a href="https://www.builder.io/blog/ai-dev-skill">Why AI Is Making Dev Skills More Valuable, Not Less</a>: AI isn&#x27;t replacing devs, it&#x27;s making them more valuable. Let&#x27;s look at how the job of devs is evolving and how it impacts teams</li><li><a href="https://www.mcpservers.ai/servers/modelcontextprotocol/Sequential%20Thinking">MCP Servers</a>: Browse the largest library of Model Context Protocol Servers. Share Model Context Protocol Servers you create with others.</li><li><a href="https://docs.codeium.com/context-awareness/overview">Overview - Codeium Docs</a>: no description found</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://www.builder.io/blog/windsurf-vs-cursor">Windsurf vs Cursor: which is the better AI code editor?</a>: Comparing Windsurf &amp; Cursor AI-powered IDEs: features, user experience &amp; workflow efficiency. Which is best for you?</li><li><a href="https://zed.dev/releases/stable/0.166.1">Zed - The editor for what&#x27;s next</a>: Zed is a high-performance, multiplayer code editor from the creators of Atom and Tree-sitter.</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: Contact the Codeium team for support and to learn more about our enterprise offering.</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/git">servers/src/git at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://youtu.be/54RUAzPYEeY"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1319163554619850834)** (193 messages🔥🔥): 

> `Gemini 2.0 Flash Thinking, OpenAI updates, Researcher departures, Search engine competition, Reasoning models` 


- **Gemini 2.0 Flash Thinking launched**: Google introduced [Gemini 2.0 Flash Thinking](https://x.com/NoamShazeer/status/1869789881637200228), an experimental model designed to explicitly show its thoughts while reasoning, promising improved performance.
   - This model aims to combine speed and enhanced reasoning capabilities, potentially positioning Google strongly in the AI chatbot landscape.
- **OpenAI introduces new chat features**: OpenAI announced 'Work with Apps' support in voice mode, allowing integration with apps like Notion and Apple Notes, as highlighted on their 12 Days of ChatGPT [site](https://openai.com/12-days/?day=11).
   - This marks another step for OpenAI to enhance user interaction and functionality in their systems.
- **Significant departures at OpenAI**: Notable researcher @AlecRad has left OpenAI, recognized as a key figure in the development of models like GPT, Whisper, and DALL-E.
   - Concerns were raised regarding the future leadership and direction of OpenAI following this departure.
- **Competitive landscape in search engines**: @amir reported that Google is integrating its Gemini chatbot directly into search results, marking a strategic shift towards conversational AI modes in search.
   - This raises questions about how competing services, such as Kagi, attract users seeking less commercialized search experiences.
- **Discussions on reasoning in AI models**: Participants debated the effectiveness of reasoning models, emphasizing that self-correction may not be necessary if models can output reasoning effectively without errors.
   - This highlights the ongoing exploration of how AI achieves reasoning and its distinction from traditional search methods.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tsarnick/status/1869500847488692256">Tweet from Tsarathustra (@tsarnick)</a>: Ex-OpenAI chief research officer Bob McGrew says that o1 is really GPT-5 because it represents a 100x compute increase over GPT-4 and the release of GPT-4.5 will be an interesting reveal of how pre-tr...</li><li><a href="https://x.com/zhou_xian_/status/1869511650782658846">Tweet from Zhou Xian (@zhou_xian_)</a>: Everything you love about generative models — now powered by real physics!Announcing the Genesis project — after a 24-month large-scale research collaboration involving over 20 research labs — a gener...</li><li><a href="https://x.com/denny_zhou/status/1869771028693713284">Tweet from Denny Zhou (@denny_zhou)</a>: Tree search, the key idea in classical AI, has little to do with true intelligence or reasoning, no matter which fun puzzle / games are well solved by search eg game 24. Search is just a tool usage. S...</li><li><a href="https://arxiv.org/abs/2412.13663">Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference</a>: Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of n...</li><li><a href="https://x.com/testingcatalog/status/1869810186648740153">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨:  OpenAI introduced &#34;Work with Apps&#34; support in voice mode along with support for Notion, Apple Notes and more 👀Quoting OpenAI (@OpenAI) Day 11: A new way to work with ChatGPThttp...</li><li><a href="https://x.com/altryne/status/1869571717368267092">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: @arcprize o1 preview was 21% 🔥 that&#39;s QUITE the jump damn</li><li><a href="https://x.com/wintermoat/status/1869784711121514620">Tweet from Alphabetting (@wintermoat)</a>: @TheXeophon In AI studio rn!</li><li><a href="https://x.com/Presidentlin/status/1869745206842794047">Tweet from Lincoln 🇿🇦 (@Presidentlin)</a>: @Alibaba_Qwen They made a reasoning VL model?</li><li><a href="https://x.com/NoamShazeer/status/1869789881637200228">Tweet from Noam Shazeer (@NoamShazeer)</a>: We’ve been *thinking* about how to improve model reasoning and explainabilityIntroducing Gemini 2.0 Flash Thinking, an experimental model trained to think out loud, leading to stronger reasoning perfo...</li><li><a href="https://x.com/amir/status/1869837622627184865">Tweet from Amir Efrati (@amir)</a>: new: Google is effectively adding its Gemini chatbot directly into search results—&#34;AI Mode&#34;innovator&#39;s dilemma remains, but this shows Google getting serious about conversational chatbot p...</li><li><a href="https://x.com/btibor91/status/1869784134224359709">Tweet from Tibor Blaho (@btibor91)</a>: Best for- Multimodal understanding- Reasoning- CodingUse case- Reason over the most complex problems- Show the thinking process of the model- Tackle difficult code and math problemsKnowledge cutoff: A...</li><li><a href="https://www.interconnects.ai/p/openais-o1-using-search-was-a-psyop">OpenAI&#x27;s o1 using &quot;search&quot; was a PSYOP</a>: How to understand OpenAI&#x27;s o1 models as really just one wacky, wonderful, long chain of thought</li><li><a href="https://fxtwitter.com/arcprize/status/1869551373848908029">Tweet from ARC Prize (@arcprize)</a>: Verified o1 performance on ARC-AGI&#39;s Semi-Private Eval (100 tasks)o1, Low: 25% ($1.5/task)o1, Medium: 31% ($2.5/task)o1, High: 32% ($3.8/task)</li><li><a href="https://x.com/natolambert/status/1869802093856612657">Tweet from Nathan Lambert (@natolambert)</a>: Come on Google show me the test time scaling plot from your internal experiments. Required documentaiton for RL cred.</li><li><a href="https://x.com/justinlin610/status/1869793885540757715?s=46">Tweet from Junyang Lin (@JustinLin610)</a>: so sorry for making you expect. nothing will happen tonight. we still need to make things better for this release. will be back very soon.</li><li><a href="https://x.com/anton_lozhkov/status/1869771053146464507?t=J1oHcOrr0APg0r9b1mP3tQ&s=19">Tweet from Anton Lozhkov (@anton_lozhkov)</a>: Introducing 📐FineMath: the best open math pre-training dataset with 50B+ tokens!Math remains challenging for LLMs and by training on FineMath we see considerable gains over other math datasets, espec...</li><li><a href="https://x.com/swishfever/status/1869774920164778170">Tweet from fishy business (@swishfever)</a>: datamined strings:&#34;The thoughts produced by the model are experimental.&#34;&#34;p6ntest-ai-llm-prompt-config-thinking-model-disclaimer&#34;Quoting Logan Kilpatrick (@OfficialLoganK) 🤔</li><li><a href="https://x.com/amir/status/1869847852308205935">Tweet from Amir Efrati (@amir)</a>: news: Another key OpenAI researcher @AlecRad is out. Lead author on GPT paper, instrumental to Whisper and Dall-E....</li><li><a href="https://aistudio.google.com/u/2/prompts/new_chat?pli=1">no title found</a>: no description found</li><li><a href="https://x.com/lmthang/status/1869797423763341448">Tweet from Thang Luong (@lmthang)</a>: Last announcement of the year from @GoogleDeepMind? Not sure :) but glad to be part of the team that launched Gemini 2.0 Flash Thinking, a model that is both smart & fast! Welcome to the Thinking era....</li><li><a href="https://x.com/vikhyatk/status/1869605301596631191">Tweet from vik (@vikhyatk)</a>: everyone’s posting 🤯 tweets about this but i can tell no one has tried it out.because i did and it says the method generate doesn’t exist on module genesisQuoting Allen T. (@Mr_AllenT) This is the cr...</li><li><a href="https://github.com/googleapis/python-genai/blob/3e42644784304d45d0b0bfdc8279958109650576/google/genai/tests/models/test_generate_content_thought.py">python-genai/google/genai/tests/models/test_generate_content_thought.py at 3e42644784304d45d0b0bfdc8279958109650576 · googleapis/python-genai</a>: Google Gen AI Python SDK provides an interface for developers to integrate Google&amp;#39;s generative models into their Python applications. This is an early release. API is subject to change. Please...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1319190817877004298)** (16 messages🔥): 

> `o1 model discussion, Chollet's analogies, Subbarao/Miles Brundage incident, Francois Chollet's grumpiness, Interconnects engagement` 


- **Chollet's take on o1 as an LLM**: François Chollet claimed that labeling o1 as ‘an LLM’ is akin to calling AlphaGo ‘a convnet’, igniting debate among members.
   - While some challenged Chollet, referencing *AlphaGo’s* reliance on MCTS and neural networks, many expressed confusion over o1's operating principles.
- **Francois Chollet's Grumpy Reputation**: Members humorously noted Chollet's grumpy demeanor when discussing the o1 model and its comparisons to established models.
   - Comments highlighted a desire for better clarity on o1, with suggestions that someone from OpenAI should explain its functionality to Chollet.
- **Subbarao/Miles Brundage Incident Recalled**: The discussion brought up the Subbarao/Miles Brundage incident, emphasizing the viewpoint that o1 operates primarily as a language model.
   - A member referenced this incident, suggesting it reflects broader misunderstandings in the community about model deployments.
- **Call for Meme on 'Oneshotting Turbonormies'**: A member expressed the need for a meme related to ‘oneshotting turbonormies’, indicating the ongoing meme culture within the discussions.
   - Frustration was expressed over not being able to find the meme quickly when needed.
- **Engagement with Interconnects**: Members discussed the value of reading content from Interconnects, with suggestions to reply to Chollet with links to relevant discussions.
   - The conversation highlighted a humorous take on keeping up with the fast-paced debates within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tszzl/status/1869681557340086602)">Tweet from roon (@tszzl)</a>: @rao2z @Miles_Brundage but it’s not really a scientific question how a deployed product works or how the model is inferenced. o1 is just a language model</li><li><a href="https://x.com/fchollet/status/1869612195425972715?s=46&t=_jodDCDeIUnWb_Td0294bw">Tweet from François Chollet (@fchollet)</a>: Calling something like o1 &#34;an LLM&#34; is about as accurate as calling AlphaGo &#34;a convnet&#34;</li><li><a href="https://fxtwitter.com/fchollet/status/1869854758443557020">Tweet from François Chollet (@fchollet)</a>: For those who didn&#39;t get it -- AlphaGo was a MCTS search process that made thousands of calls to two separate convnets in order to compute a single game move.Something like o1 pro is also, best we...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1319036036470407319)** (167 messages🔥🔥): 

> `Stripe Tax Implementation, Substack Revenue Model and Tax Concerns, CPA Recommendations for Tax Filing, Digital Services and VAT Compliance, Challenges for International Taxation` 


- **Stripe Tax as a Safety Net**: The discussion emphasized the importance of enabling [Stripe Tax](https://stripe.com/tax) for digital services to simplify tax compliance, especially for Substack creators approaching revenue thresholds.
   - *Turning this feature on can avoid potential headaches down the line* with taxation authorities.
- **Confusion Around Substack's Tax Handling**: Participants were uncertain about how Substack handles taxes, with discussions about whether Substack is considered the marketplace operator responsible for tax collection.
   - Nate pointed out that as payments go directly to Substack's Stripe account, it complicates the tax situation for creators.
- **Learning from Bigger Substackers**: Nate noted that even larger Substackers seem to lack knowledge about tax obligations, indicating a possible trend among creators in this space.
   - This raised the point about the broader issues of accountability and responsibility in reporting earnings and taxes.
- **CPA and Tax Advice**: Several members suggested reaching out to a CPA for guidance on navigating tax requirements, particularly for digital service businesses.
   - Nate mentioned that his partner's mother is a CPA and expressed interest in gathering more recommendations to ensure proper compliance.
- **International Tax Challenges**: There was discussion around the challenges of managing VAT in Europe and how individuals or businesses might navigate potential tax liabilities in international contexts.
   - One member humorously noted that failing to comply could lead to severe consequences, indicating the seriousness of these tax issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.hex.tech/533fe68e-dcd8-4a52-a101-aefba762f581/app/b9dc830a-bd3f-4dc9-8495-3be64e735ce2/latest">vLLM 2024 Analysis</a>: Hex is a modern data platform for data science and analytics. Collaborative notebooks, beautiful data apps and enterprise-grade security.</li><li><a href="https://help.kagi.com/kagi/faq/sales-tax-vat.html">Billing / Sales Tax / VAT FAQ | Kagi's Docs</a>: no description found</li><li><a href="https://support.substack.com/hc/en-us/articles/12282257442580-Does-Substack-integrate-with-Stripe-Tax">Does Substack integrate with Stripe Tax?</a>: Yes! If you've enabled Stripe Tax, this feature can help determine and calculate the right amount of tax at the point of sale on Substack. Tax is automatically calculated for transactions in a regi...</li><li><a href="https://www.regs2riches.com/p/substack-ed-against-sales-tax">🃏 substack-ed against sales tax?</a>: 💸 even disruptors have tax obligations</li><li><a href="https://open.substack.com/pub/faq/p/setting-up-vat-tax-compliance-for?r=68gy5&utm_medium=ios">Setting up VAT tax compliance for your Substack </a>: Substack now makes it easy for European writers to keep track of tax obligations</li><li><a href="https://stripe.com/tax">Stripe Tax | Tax Automation with a Single Integration</a>: Automate tax compliance with Stripe Tax. Easily calculate, collect, and report sales tax, VAT, and GST on global payments with a single integration.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1319190812940308551)** (2 messages): 

> `Interactive AI in Game Shows, Social Media Reactions` 


- **ChatGPT on Game Shows: A Hilarious Twist**: A member joked about calling **1-800-ChatGPT** during a game of **'Who Wants to Be a Millionaire'**, showcasing the growing influence of AI in pop culture.
   - This humorous reference reflects the ongoing integration of AI into everyday scenarios and entertainment.
- **Viral AI Tweets**: A tweet by **voooooogel** gained attention, though its specifics remain undefined, hinting at AI-related content that sparked discussions.
   - This kind of interaction underlines the curiosity and engagement surrounding AI topics on social media platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Yuchenj_UW/status/1869799400681419122">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: When you&#39;re on &#34;Who Wants to Be a Millionaire&#34; and you decide to phone &#34;1-800-ChatGPT&#34;:</li><li><a href="https://x.com/voooooogel/status/1869529374829207884">Tweet from thebes (@voooooogel)</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 messages): 

kevin_nejad: it's interesting (but not obvious) such behaviour emerges purely from RL training
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1319390383649853560)** (7 messages): 

> `RLHF Book, Typos Correction, Fundamentals Review` 


- **Stress Relief through RLHF Review**: A member expressed feeling *less stressed* about work while spending time on the **RLHF Book** and a **long lecture** on YouTube before the break.
   - They found it *cathartic* to review the fundamentals of RLHF, highlighting its importance.
- **Calling for Typos Corrections**: A proposal was made to send **free copies** of the RLHF book to individuals who help fix typos or formatting problems.
   - Another member eagerly volunteered to contribute, stating they are well-suited due to their command of the English language.
- **Community Involvement in RLHF**: An RLHF novice expressed interest in helping with corrections for the book, mentioning that they appreciate *free* items.
   - The community seems eager to engage, as members recognize the collaborative opportunity.
- **RLHF Resources on GitHub**: A member noted that the resources for the **RLHF Book** are all available on **GitHub**, making it easy to access.
   - This accessibility facilitates collaboration and contributions from the community.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

natolambert: Yeah. Students coming up and wanting to take photos is so cute ❤️
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1319361588431425627)** (1 messages): 

> `12 Days of OpenAI, ChatGPT work enhancements` 


- **Kickoff for Day 11 on ChatGPT**: Day 11 of the **12 Days of OpenAI** introduces a new way to work with **ChatGPT**, detailed in a [YouTube live session](https://www.youtube.com/live/g_qxoznfa7E?si=q71WyiHuioBSGzvz).
   - *Stay in the loop* during these days by picking up the <@&1261377106890199132> role in <id:customize>.
- **Get Involved with OpenAI Updates**: Participants are encouraged to engage with the **12 Days of OpenAI** by staying informed on the latest developments and opportunities.
   - This initiative allows members to enhance their experience with **ChatGPT** and related tools.



**Link mentioned**: <a href="https://www.youtube.com/live/g_qxoznfa7E?si=q71WyiHuioBSGzvz"> - YouTube</a>: no description found

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1319038880808173598)** (310 messages🔥🔥): 

> `ChatGPT and integration, Google's AI developments, YouTube clone project, Software engineering automation, AI benchmarks and capabilities` 


- **ChatGPT adds integrations with XCode**: Users discussed how ChatGPT can facilitate code development by allowing users to copy and paste text directly into XCode, enhancing the workflow.
   - While this feature offers convenience, it still requires manual input from users for tasks like initiating the copy action.
- **Google releases experimental AI models**: Chat participants noted the recent release of Google's Gemini 2.0 Flash Thinking experimental model, highlighting its performance and the public's interest.
   - There was skepticism surrounding the model's accuracy, particularly in simple tasks like counting letters in words.
- **Creating a YouTube clone using ChatGPT**: Members were enthusiastic about the prospect of using ChatGPT to create a YouTube clone, discussing the model's ability to handle front-end and back-end coding.
   - The challenge lies in the more involved terminal operations required for back-end construction, which was acknowledged as a complexity in the process.
- **Future of software engineering with AI**: Participants speculated on how advancements in AI could potentially automate entire software engineering tasks, impacting the need for human engineers.
   - Automation was seen as both exciting and concerning, depending on how complex the tasks remain despite AI capabilities.
- **AI performance benchmarks**: The community raised questions about AI benchmarks and model performance, particularly regarding Google's new offerings versus existing ones.
   - Participants expressed interest in the model's capabilities but also skepticism, emphasizing the ongoing discourse on LLM efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/AlignAGI/Alignment/">GitHub - AlignAGI/Alignment: Promoting global awareness and action for ethical AI alignment and safeguarding humanity against AI self-replication risks. Includes research, frameworks, and open-source resources.</a>: Promoting global awareness and action for ethical AI alignment and safeguarding humanity against AI self-replication risks. Includes research, frameworks, and open-source resources. - AlignAGI/Alig...</li><li><a href="https://www.youtube.com/watch?v=v-EYzZCLF48"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/AeMvOPkUwtQ?feature=shared"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1319055729478467654)** (8 messages🔥): 

> `Editing GPTs, Project Folder Limitations, Support Channels, Pro Package Tool Issues` 


- **Editing GPTs remains a puzzle**: There was confusion regarding the capability to **edit GPTs**, with one user insisting they can edit while another reported being unable to.
   - *jerzjorge23* stated that they could only create new GPTs since the recent project release.
- **Limitations of the Project Folder**: *7_vit_7* mentioned that moving GPTs into the **projects folder** isn't possible due to potential attached files causing conflicts.
   - *jerzjorge23* clarified they were not attempting to move files but simply wanted to edit them.
- **Seeking support channels**: *armantlrp* inquired about potential **support channels** available for assistance regarding tool usability.
   - They noted that multiple tools, including **canvas**, search, and picture, were unusable on both web and MacOS versions.
- **Pro package tool issues persist**: *armantlrp* has been experiencing issues with their **Pro package** tools being unavailable for several days.
   - This raised concerns within the community about possible ongoing issues affecting features for **Pro package** users.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1319078222301888605)** (146 messages🔥🔥): 

> `FSDP and Tensor Parallelism, EleutherAI Token Controversy, Natural Attention Optimizer, Debugging Training Models, Causal Masking in Attention` 


- **FSDP and Tensor Parallelism Debate**: Members discussed the differences between **Fully Sharded Data Parallel (FSDP)** and **Tensor Parallelism**, noting that FSDP shards parameters while maintaining operations across GPUs.
   - Some expressed skepticism about the efficiency of FSDP due to increased communication overhead compared to direct tensor parallel implementations.
- **Debunking EleutherAI Token Myths**: EleutherAI does not have any affiliated cryptocurrency, and members warned others about scams related to unofficial tokens that have appeared recently.
   - The community emphasized that investing in such tokens is akin to participating in a Ponzi scheme.
- **Introduction of Natural Attention Optimizer**: A member shared insights on a new **Attention Informed Optimizer** that adapts the Adam optimization algorithm using gradients from attention mechanisms.
   - The optimizer is claimed to improve performance significantly, although the funnelling of results raised flags regarding potential bugs in the implementation.
- **Challenges in Model Training Debugging**: Participants discussed troubleshooting issues in model training, particularly focusing on an unusually low loss value in one participant's results.
   - Recommendations included double-checking causal masking functions, as incorrect implementations could lead to misleading training metrics.
- **Importance of Causal Masks in Attention**: Members highlighted the necessity of causal masks in attention mechanisms to prevent future tokens from influencing current predictions.
   - It was suggested that overlooking this component could result in extreme discrepancies in model performance and outputs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>: Transformers have emerged as the architecture of choice for many state-of-the-art AI models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands impo...</li><li><a href="https://pump.fun/coin/5CCtDehQTswpWzeYdxUWz7VS3bCrwV9o8ZfUyKgJpump">Eleuther Ai (eAI) - Pump</a>: A non-profit research lab focused on interpretability, alignment, and ethics of artificial intelligence.</li><li><a href="https://github.com/jeroaranda/naturalattention">GitHub - jeroaranda/naturalattention</a>: Contribute to jeroaranda/naturalattention development by creating an account on GitHub.</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/papers/Natural_attention_proofs.pdf">naturalattention/papers/Natural_attention_proofs.pdf at main · jeroaranda/naturalattention</a>: Contribute to jeroaranda/naturalattention development by creating an account on GitHub.</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/config_utils.py#L95">llama-recipes/src/llama_recipes/utils/config_utils.py at main · meta-llama/llama-recipes</a>: Scripts for fine-tuning Meta Llama with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q&...</li><li><a href="https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/gpt2/modeling_gpt2.py#L195>">transformers/src/transformers/models/gpt2/modeling_gpt2.py at v4.47.1 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py">naturalattention/natural_attention.py at main · jeroaranda/naturalattention</a>: Contribute to jeroaranda/naturalattention development by creating an account on GitHub.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py#L43>">naturalattention/natural_attention.py at main · jeroaranda/naturalattention</a>: Contribute to jeroaranda/naturalattention development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1319031842422259774)** (123 messages🔥🔥): 

> `Microsoft Research Ethics, Koopman Theory and Neural Networks, Diffusion vs Autoregressive Models, Plagiarism Concerns in ML Research, Research Submissions and Oversight` 


- **Concern Over Microsoft Research Ethics**: Discussions highlighted issues with ethical practices at Microsoft Research (MSR), including recent plagiarism accusations against their papers for copying work without citation.
   - Previous controversies, such as the Phi methodology, and instances of low integrity were noted, raising questions about the overall ethical culture at MSR.
- **Debate on Koopman Theory Application**: Members debated the validity of using Koopman theory in the context of neural networks, with some arguing that the application seems forced and doesn't yield clear benefits.
   - Concerns were raised about the underlying theoretical justification for such approaches, suggesting that they could unintentionally mislead researchers.
- **Diffusion vs Autoregressive Models**: A discussion emerged on the pros and cons of diffusion models compared to autoregressive methods across various modalities, particularly their efficiency and suitability for discrete datasets.
   - While diffusion models currently dominate in image generation, there is speculation about their long-term viability compared to autoregressive techniques in other tasks.
- **Plagiarism in Machine Learning Research**: Several members expressed concerns about apparent plagiarism in recent machine learning research papers, particularly those from high-profile institutions like MSR.
   - Calls for accountability and transparency in research practices were made, emphasizing the need for public pushback against unethical conduct.
- **Research Submissions and Oversight**: Discussions about the differing oversight structures in research organizations raised questions about the implications for research integrity and the handling of controversies.
   - Members noted how decentralized oversight at MSR may contribute to ethical lapses, contrasting it with more centralized approaches observed in other organizations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/zhou_xian_/status/1869511650782658846">Tweet from Zhou Xian (@zhou_xian_)</a>: Everything you love about generative models — now powered by real physics!Announcing the Genesis project — after a 24-month large-scale research collaboration involving over 20 research labs — a gener...</li><li><a href="https://tsb0601.github.io/metamorph/">no title found</a>: no description found</li><li><a href="https://x.com/cloneofsimo/status/1869807463186472970?s=46">Tweet from Simo Ryu (@cloneofsimo)</a>: Ok so lucas made a point on maybe it helped because it was 12 layer network.So I made it 96 layer network (with 768 hidden dim LOL) and sweeped for fucking 10 hoursTo my surprise, gap has widened comp...</li><li><a href="https://arxiv.org/abs/2112.00114">Show Your Work: Scratchpads for Intermediate Computation with Language Models</a>: Large pre-trained language models perform remarkably well on tasks that can be done &#34;in one pass&#34;, such as generating realistic text or synthesizing computer programs. However, they struggle w...</li><li><a href="https://arxiv.org/abs/2305.20050">Let&#39;s Verify Step by Step</a>: In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. T...</li><li><a href="https://arxiv.org/abs/1810.01479">Time-Delay Observables for Koopman: Theory and Applications</a>: Nonlinear dynamical systems are ubiquitous in science and engineering, yet analysis and prediction of these systems remains a challenge. Koopman operator theory circumvents some of these issues by con...</li><li><a href="https://arxiv.org/abs/2212.07358">Learning Invariant Subspaces of Koopman Operators--Part 1: A Methodology for Demonstrating a Dictionary&#39;s Approximate Subspace Invariance</a>: Koopman operators model nonlinear dynamics as a linear dynamic system acting on a nonlinear function as the state. This nonstandard state is often called a Koopman observable and is usually approximat...</li><li><a href="https://arxiv.org/abs/2206.07137">Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt</a>: Training on web-scale data can take months. But most computation and time is wasted on redundant and noisy points that are already learnt or not learnable. To accelerate training, we introduce Reducib...</li><li><a href="https://github.com/Genesis-Embodied-AI/Genesis/tree/main/genesis/assets/meshes">Genesis/genesis/assets/meshes at main · Genesis-Embodied-AI/Genesis</a>: A generative world for general-purpose robotics &amp; embodied AI learning. - Genesis-Embodied-AI/Genesis</li><li><a href="https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#muon-optimizer>">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 5 minutes</a>: NanoGPT (124M) in 5 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.</li><li><a href="https://x.com/kellerjordan0/status/1869752568026689767>">Tweet from Keller Jordan (@kellerjordan0)</a>: I would like to issue a citation request for Muon to the following newly appearing paper from Microsoft Research:Ma et al. (2024). SWAN: Preprocessing SGD Enables Adam-Level Performance On LLM Trainin...</li><li><a href="https://x.com/hi_tysam/status/1869756590661992919>">Tweet from Fern (@hi_tysam)</a>: Keller is being polite (appropriately so). From my perspective this appears to be rather egregious plagiarism which is odd coming from @MSFTResearch It pretty blatantly copies concepts from both Shamp...</li><li><a href="https://x.com/HessianFree/status/1869781347696550178>">Tweet from Omead Pooladzandi (@HessianFree)</a>: Really frustrating to see people at @MSFTResearch do something like this. Gradient whitening isn&#39;t new. See Amari 1998, LeCun 1998 but we all know Kellers been doing fantastic work on the Newton-S...</li><li><a href="https://x.com/evaninwords/status/1869767632636854570>">Tweet from Evan Walters (@evaninwords)</a>: They misspelled newton-schulz, didn&#39;t mention muon once, and missed other papers too like this one that had spurred me to try orthogonalizing grads years ago https://arxiv.org/abs/2202.07052 (they...</li><li><a href="https://x.com/xidulu/status/1869754635453681723>).">Tweet from Xidulu (@xidulu)</a>: I didn&#39;t know MSR&#39;s causal inference group is also working on optimization 🤣Quoting Keller Jordan (@kellerjordan0) I would like to issue a citation request for Muon to the following newly app...</li><li><a href="https://x.com/YouJiacheng/status/1869780973862408641>)">Tweet from YouJiacheng (@YouJiacheng)</a>: I suspect `GradNorm` is actually a no-op after &#34;GradWhitening&#34; (except making the largest singular value &lt; sqrt(3) so that NS converges).-- the mean value (btw I think it should be 1_m inst...</li><li><a href="https://openreview.net/forum?id=0NMzBwqaAJ">Not All Tokens Are What You Need for Pretraining</a>: Previous language model pre-training methods have uniformly applied a next-token prediction loss to all training tokens. Challenging this norm, we posit that &#x27;&#x27;Not all tokens in a corpus are...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1319284179208507414)** (5 messages): 

> `Independence of Neural Network Activations, Pre-image Reconstruction Methods, Steered vs Unsteered Sparse Autoencoders, Out-of-Distribution (OOD) Evaluation` 


- **Investigating Independence of Activations**: A user inquired about the **independence of neural network activations** within the same layer, expressing challenges in finding relevant analyses. It was noted that *higher model nonlinearity* tends to decrease independence in intermediate layers.
- **Challenges in Pre-image Reconstruction**: The user detailed experiments with **pre-image reconstruction** for a CNN using MNIST, finding that edits to one activation affected others. When comparing two pre-image methods, the *correlation in activation changes* suggested a degree of dependence among activations.
- **Insights on Sparse Autoencoder Features**: The user applied similar experiments to **sparse autoencoder features**, observing a lack of independence between features. This reinforces the notion that activations in neural networks may not behave as independently as traditionally assumed.
- **Measuring Out-of-Distribution (OOD) for SAE Reconstructions**: Another user sought best practices for assessing the **degrees of OOD** in steered sparse autoencoders. They inquired whether *cosine similarity between centroids* of steered vs unsteered activations could be a viable measurement strategy.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1319035817699840000)** (254 messages🔥🔥): 

> `Perplexity AI updates, You.com features, Gemini models, Student discounts, Referral systems` 


- **Perplexity AI Referral System Confirmed**: A user confirmed that [Perplexity does have a referral system](https://www.perplexity.ai/settings/account) which can benefit those signing up through connections.
   - Another user is enthusiastic about getting more people onboard, stating their whole fraternity might join.
- **You.com Performance vs Models**: Concerns were raised about the quality of responses from You.com, suggesting that answers may not match the performance of direct models due to the search interface.
   - Users discussed the value of the actual models being utilized, rather than only mimicking responses through system instructions.
- **Students Can Access Free Pro with .edu Emails**: Reports surfaced about students obtaining free Pro access by signing in with .edu emails, although some users noted issues with the process.
   - A user shared a link to the [back-to-school promotion](https://www.perplexity.ai/backtoschool), highlighting potential benefits.
- **Anticipation for New Superman Movie**: Details about a new Superman movie teaser were shared, prompting mixed reactions and excitement among users.
   - The random announcement was described as surprising, indicating user engagement beyond AI discussions.
- **Challenges in Translating Game Descriptions**: A user faced difficulties getting Perplexity AI to translate an entire list of game descriptions into French, with the AI only processing a few before stopping.
   - Assistance was sought in managing the limits placed by the AI on handling large data sets.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/)">no title found</a>: no description found</li><li><a href="https://x.com/apostraphi/status/1869612493989163410?s=46">Tweet from Phi Hoang (@apostraphi)</a>: our destiny lies above us</li><li><a href="https://x.com/pplxsupply/status/1869134944418890157?s=46">Tweet from Perplexity Supply (@PPLXsupply)</a>: Design details</li><li><a href="https://www.youtube.com/watch?v=g_qxoznfa7E"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1319031608585617428)** (7 messages): 

> `EU Funds Starlink Rival, Plants Cry, Law of the Few, Magic Spell Hypothesis, Tornado Alley` 


- **EU Funds Starlink Rival**: A [YouTube video](https://www.youtube.com/embed/FBX4lu3LIEI) discusses how the **EU** is funding a rival to **Starlink**, exploring its potential impacts on global internet access.
   - The video also covers the implications of this initiative on **connectivity** and **competition** in satellite internet services.
- **Plants Exhibit Crying Behavior**: The topic of why **plants cry** surfaced, discussing the recent findings on this fascinating phenomenon and its implications for plant biology.
   - Readers engaged with sources noting that plant responses can resemble emotional states and reflect their environmental stress levels.
- **Understanding the Law of the Few**: The **Law of the Few** was mentioned, suggesting that a small number of people can influence a larger crowd, as discussed by researchers.
   - Links illustrate how this social principle applies to technology and marketing strategies, enhancing viral growth potential.
- **Magic Spell Hypothesis Review**: A document discusses the **Magic Spell Hypothesis**, outlining its key arguments and relevance to ongoing scientific debates.
   - The [link](https://www.perplexity.ai/page/the-magic-spell-hypothesis-n5tkbs1JR4OGww9A25c9ZA) provides insights into its theoretical applications and critiques from the academic community.
- **Exploring Tornado Alley**: A recent inquiry into **Tornado Alley** looked at geographical and meteorological data that define this region's tornado occurrences.
   - The discussion highlights safety measures and preparedness strategies for those living in vulnerable areas, as shared in a [valuable resource](https://www.perplexity.ai/search/what-is-tornado-alley-MKPYqZvsQg6x1TtVvhmARQ).



**Link mentioned**: <a href="https://www.youtube.com/embed/FBX4lu3LIEI">YouTube</a>: no description found

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1319038525579985006)** (222 messages🔥🔥): 

> `Gemini models, Aider integration, MCP functionality, OpenAI access issues, Jira task automation` 


- **Gemini 2.0 Flash Thinking Launch**: The new model, `gemini-2.0-flash-thinking-exp-1219`, was introduced, demonstrating potential for improved reasoning and response quality, particularly in agentic workflows.
   - Initial tests indicated fast performance and a higher quality output compared to existing models like O1 and deepseek.
- **Aider and MCP Integration**: Users discussed setting up the MCP functionality with Aider, successfully integrating it for tasks like creating and managing Jira updates.
   - Some users noted that while Sonnet has been commonly used, there is potential for using other models in MCP setups.
- **OpenAI Access from EC2**: A user inquired about accessing OpenAI services from an EC2 server, confirming smooth operation and no issues reported.
   - The original concern was clarified, suggesting it was related to an individual setting rather than a widespread problem.
- **Model Preference in Task Automation**: Users identified preferences for the weak model in handling specific tasks like commit messages and summarization within the workflow.
   - Discussions highlighted the versatility of combining different models for optimal performance in task management.
- **Testing Other Models**: There are inquiries about the capabilities of various models like Qwen and their performance in coding tasks as well as debugging.
   - Users expressed interest in experimenting with these models for better integration into their workflow automation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>: no description found</li><li><a href="https://chatapi.akash.network/documentation">Akash Chat API</a>: no description found</li><li><a href="https://x.com/JeffDean/status/1869789813232341267">Tweet from Jeff Dean (@JeffDean)</a>: Introducing Gemini 2.0 Flash Thinking, an experimental model that explicitly shows its thoughts.Built on 2.0 Flash’s speed and performance, this model is trained to use thoughts to strengthen its reas...</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/faq.html#how-are-the-aider-wrote-xx-of-code-stats-computed">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://tenor.com/view/good-morning-gif-24191255">Good Morning GIF - Good Morning - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/docs/provider-routing',">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://mcpserver.cloud/server/server-sentry">Sentry Integration Server - MCP Server Integration | MCPHub</a>: Sentry.io integration for error tracking and performance monitoring. Integrate Sentry Integration Server with Model Context Protocol for enhanced AI capabilities and seamless model interactions.</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V2.5 - API, Providers, Stats</a>: DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. Run DeepSeek V2.5 with API</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#model-settings">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#example-model-settings">Advanced model settings</a>: Configuring advanced settings for LLMs.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1319103139366043699)** (11 messages🔥): 

> `Using multiple OpenAPI services, Gemini Flash 2.0 issues, Architect mode features, Adding files in a fuzzy way, Project planning models` 


- **Combining OpenAPI Services Efficiently**: A user initially sought guidance on how to use two different OpenAPI services, specifically **QwQ** on **Hugging Face** and local **Ollama**.
   - They later realized that Hugging Face has its own API and that the method needs to be dictated in the model name.
- **Gemini Flash 2.0 Modifications**: A member reported ongoing issues with **Gemini Flash 2.0**, noting it typically modifies the wrong instance, commonly the first.
   - Another member suggested employing the AI comments feature as a workaround.
- **Does —watch-files Work with Architect Mode?**: Inquiries arose regarding whether the **—watch-files** option is compatible with **architect** mode.
   - A response indicated that adjustments would be prompted when using the option correctly.
- **Fuzzy File Addition in Chat**: A user asked about a method for adding files in a fuzzy manner without specifying the full path each time, sharing an example output.
   - They discovered that committing files is necessary for the Aider to auto-suggest them on the **/add** command.
- **Recommended Hardware for Aider Client**: A question emerged regarding suitable hardware for running the Aider client-side, with reports of the LLM finishing while the client delays response assembly.
   - Another member responded that such delays should not be occurring, indicating potential issues with the setup.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1319055993748983818)** (9 messages🔥): 

> `GitHub Copilot Chat, Aider Composer VSCode Extension, Diff Edits Preference` 


- **GitHub Copilot Chat Immersive Mode Launched**: GitHub announced enhanced features for [Copilot Chat](https://github.blog/changelog/2024-12-18-announcing-github-copilot-free/), including an immersive chat experience and smarter, faster responses tailored to user needs.
   - Real-time interactions with your codebase are now supported, allowing for immediate answers to coding questions and facilitating effortless code generation.
- **Aider Composer Extension Review**: A review of the Aider Composer VSCode Extension highlighted the new diff accept view, which replaces the previous git diff view but noted it doesn't commit to git, limiting undo capabilities.
   - The main advantage of this extension is its use of the installed version of Aider, enhancing user control over the coding process.
- **Improvement in GitHub Copilot since Launch**: Members discussed how GitHub Copilot has progressed since its initial free release, now offering features like Claude Sonnet integration and a multi-file edit function.
   - Despite this, one user critiqued that the free tier still offers limited access, leading to concerns over cost-effectiveness versus traditional diff edits.
- **Preference for Diff Edits**: In comparing GitHub Copilot's features, users showed a preference for traditional diff edits, finding them more effective and economically viable.
   - One member expressed satisfaction with Copilot's improvements, yet still advocated for the enduring utility of diff edits in coding workflows.



**Link mentioned**: <a href="https://github.blog/changelog/2024-12-18-announcing-github-copilot-free/">Announcing GitHub Copilot Free · GitHub Changelog</a>: Announcing GitHub Copilot Free

  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1319276345595002981)** (1 messages): 

> `Bolt Supabase Integration` 


- **Bolt & Supabase Integration Goes Live**: The **Bolt<>Supabase integration** is officially live and available for everyone, simplifying the process significantly.
   - *No manual setup*: just click, connect, and it’s done, making it easier for users to get started.
- **Effortless Connection Transition**: Users can now effortlessly integrate their applications with **Bolt** by connecting to **Supabase** with a single step.
   - This integration aims to streamline developer workflows and eliminate complex setup processes.



**Link mentioned**: <a href="https://x.com/stackblitz/status/1869715661444043245">Tweet from StackBlitz (@stackblitz)</a>: 📢 Announcing: Supabase&lt;&gt;Bolt integration!No manual setup: just click, connect, and it&#39;s done!

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1319155617444659251)** (14 messages🔥): 

> `Bolt project setup, Issues with .env file, Direct uploads from Figma, Application review process` 


- **Issues arise with .env file resetting**: Users reported problems with the **.env file** resetting, causing errors with their Firebase setup. One member noted that **locking the .env file** can help prevent changes during the session, but encountered issues with it being overridden after refreshing.
   - *This project exceeds the total supported prompt size* was cited as a common problem that users are facing due to this issue.
- **Direct uploads from Figma not possible yet**: A user inquired about the possibility of uploading Figma files for Bolt to generate code, but it was confirmed that **direct uploads are not currently supported**. The suggested method is to take **screenshots** as a workaround.
   - This method has been requested, indicating a demand for improved integration with design tools like Figma.
- **Finding redundancies in Bolt applications**: A user questioned if there’s a way for Bolt to **review applications for redundancies** and clean them up efficiently. The sentiment suggests that the current process may consume unnecessary tokens without providing an effective cleanup solution.
- **Creating a public folder for projects**: Instructions were shared about creating a **public folder** for projects and adding images to it for use with Bolt. Users expressed confusion on how to implement these steps effectively and where to locate this folder.
   - Clarifications on the folder setup indicate that users still seek clearer guidance on project structure.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1319032336825712640)** (182 messages🔥🔥): 

> `Bolt Issues and Feedback, Community Support and Resources, Supabase Integration, Functionality and Token Use, User Experience with Bolt` 


- **Users Encountering Downtime and Session Issues**: Multiple users reported difficulty logging into Bolt and issues with session timeouts that required refreshing the page, resulting in lost chat history.
   - While the team is aware and working to resolve the authentication issues, many expressed frustration over the impact on their projects and token use.
- **Community Collaboration on Projects**: Members discussed creating a helpful guide for Bolt, leveraging community contributions and focusing on supporting each other through project development.
   - The collaboration included plans for user dashboards to upload and approve guides, indicating a proactive community effort.
- **Integration of Supabase and Future Features**: Discussions highlighted the integration of Supabase with Bolt, emphasizing its importance alongside future plans for Stripe integration and improved token management.
   - Users were keen on understanding how to best utilize Supabase within existing projects and the functionality of the different modes available.
- **Feedback on Product Functionality and Token Consumption**: A number of users expressed frustration regarding token consumption when building applications, suggesting that redundant outputs often lead to excessive token use.
   - Suggestions were made for improving the review process for application outputs to manage redundancy and optimize token use.
- **Exploration of Tech Stack and Development Challenges**: Members discussed recommended tech stacks for mobile app development, particularly focusing on compatibility with Supabase and overall functionality with Bolt.
   - Some users experienced challenges in successfully building projects that met their expectations, leading to questions about effective utilization of Bolt.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gyazo.com/2d33c95cc8f2f94179e04c14b6fdc1b2">Gyazo</a>:  </li><li><a href="https://boltwiser.levyandco.net/">Wiser - Knowledge Sharing Platform</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1319355634138878023)** (1 messages): 

> `Interactive Mode for Audio Overviews` 


- **Interactive Mode for Audio Overviews now live for all!**: The team has successfully rolled out improvements to **Interactive Mode for Audio Overviews** to **100% of users**.
   - Users are encouraged to try it out or revisit it if they previously found it unresponsive.
- **Exciting Audio Feature Rollout**: Many **NotebookLM engineers** have worked hard to enhance the **performance** of the audio feature overviews.
   - This update aims to provide a smoother experience for all users.


  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1319031602226921512)** (17 messages🔥): 

> `NotebookML video generation, Interactive podcast feature, Podcast editing workflows, Connection of MySQL database to NotebookLM, YouTube content creation` 


- **AI-generated video explores isolation in space**: A user shared an AI-generated video vlog capturing the experiences of an astronaut isolated in space for a year, showcasing the toll of loneliness and creativity through [this YouTube link](https://youtu.be/_ys7FchEkak?feature=shared).
   - *It's a gripping portrayal of a mind unraveling*, described another user about the video.
- **Podcast interactions not saved**: A user clarified that the interactive podcast feature does not save interactions as part of the podcast, making it necessary to record both separately for external listeners.
   - This raised questions about the workflow, prompting a user to seek clarification on the podcast creation process.
- **YouTube channel showcases animated videos**: Another user pointed out the prolific content creation of a member who has been uploading varied videos almost daily, including animated ones made with NotebookML outputs, accessible [here](https://youtu.be/ub0J93QuUH4?feature=shared).
   - Viewer feedback expressed appreciation for the content, noting the creative demands behind such frequent uploads.
- **Need help connecting MySQL to NotebookLM**: A game master sought assistance on how to connect their extensive MySQL database to NotebookLM for automating NPC reactions in their long-running RPG sessions.
   - They highlighted their experience of running games for over 10 years with a large player base, indicating the complexity involved.
- **Podcast style prompt for efficiency**: One user shared a prompt designed to make podcast dialogue more succinct and blunt, focusing on a review of a QWQ model related to video acceleration calculations.
   - The provided audio extract was aimed at enhancing the podcast's delivery style by encouraging fast-paced, no-nonsense dialogue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/ub0J93QuUH4?fe"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/_ys7FchEkak?feature=shared"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1319033007268560906)** (144 messages🔥🔥): 

> `Notebook LM Interactive Mode, Audio Overview Pronunciation, Notebook Features Across Notebooks, User Feedback on New UI, Experimental Use of AI in Storytelling` 


- **Users report on Notebook LM Interactive Mode rollout**: Many users are discussing their experiences with the new **Interactive Mode**, noting that while some have access, others are still waiting for the feature to be fully rolled out.
   - Despite some initial challenges, users are excited about the creative possibilities this mode provides.
- **Pronunciation issues in Audio Overviews**: A user reported that Notebook LM incorrectly pronounced the name **Shane Gostisbehere**, repeatedly confusing it with a different name, highlighting pronunciation challenges.
   - The development team is actively investigating the issue, and users are encouraged to provide audio samples for better understanding.
- **Questions about using features across multiple notebooks**: A user inquired if features and content can be shared across multiple notebooks created for different modules.
   - It was confirmed that users must upload all sources to the same notebook, as cross-notebook functionality is currently not available.
- **Positive Feedback on New User Interface**: Several users expressed appreciation for the recently updated **Notebook LM UI**, finding it highly workable and user-friendly.
   - The team is receiving positive reinforcement, with users eager to explore the new features and capabilities.
- **Creative AI Use in Storytelling**: A user shared their excitement about using AI for storytelling, detailing an experiment in generating characters for a TTRPG set in a cyberpunk future.
   - They highlight how Notebook LM managed to adapt to various narrative challenges while staying true to the story's source material.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://youtu.be/tVURtFDvyFc?si=8PTHE9BdAKrJ2f0N"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=YS4rdvcfqEU"> - YouTube</a>: no description found</li><li><a href="https://www.reddit.com/r/singularity/comments/1hhyv8r/notebook_lm_hosts_have_full_on_sex_warning/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1319032634587742220)** (102 messages🔥🔥): 

> `Running SDXL on Ubuntu, ComfyUI Issues, AI Image and Video Quality, Quantum Computing Conversations, Civitai Website Issues` 


- **Recommendations for Running SDXL on Ubuntu**: Several members discussed tips for running **SDXL** on **Ubuntu**, with suggestions ranging from using **Forge UI** to utilizing shell launch files for easier setup.
   - *Nuuideas* pointed out that the lack of knowledge about the system might be hindering **ComfyUI** performance.
- **ComfyUI's Persistent Problems**: There were complaints about **ComfyUI** having annoying errors and producing burnt images when using certain sampling methods, despite attempts to troubleshoot.
   - *Nuuideas* recommended using **Euler** sampling and keeping denoising settings optimal for better results.
- **Expectations vs. Reality for AI Images and Video**: Discussion on whether **AI-generated images** and **video** have reached perfection, with *earnstar* asserting they won't be perfect even by 2030 due to numerous challenges.
   - *Eyaura* disagreed, claiming rapid advancements in AI technology could yield improvements sooner.
- **Debate on Quantum Computing's Future**: Conversations revolved around **quantum computing**, particularly concerning the implications of proving problems like **P=NP**, with *Nuuideas* expressing concerns over the practicality of quantum algorithms.
   - *Earnstar* highlighted the challenges of extracting useful results from quantum states.
- **Civitai Website Functionality**: *Wallykz* reported issues accessing **civitai.com**, with other members confirming outages and noting the site tends to be offline frequently.
   - *Crystalwizard* mentioned that the website often has server issues that disrupt accessibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.tomshardware.com/news/how-to-manage-virtual-memory-pagefile-windows-10,36929.html">How To Manage Virtual Memory (Pagefile) In Windows 10</a>: Follow these simple steps to manually manage the Virtual Memory (Pagefile) size in Windows 10.</li><li><a href="https://tenor.com/view/hello-well-hello-home-alone-christmas-marvin-gif-15846293">Hello Well Hello GIF - Hello Well Hello Home Alone - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=AbB33AxrcZo"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=NB9K4CoYSIM&ab_channel=AIRevolution"> - YouTube</a>: no description found</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/webui-user.sh">stable-diffusion-webui-forge/webui-user.sh at main · lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://youtu.be/S9L2WGf1KrM"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1319052977977298964)** (58 messages🔥🔥): 

> `Coil Whine, GPU Performance & Choices, Bottlenecking Debate, VRChat VRAM Needs, Next-Gen GPU Pricing` 


- **Coil Whine Strikes Again**: A user expressed concern about **absurd coil whine** from a returned RX 6750XT, leading to a bad experience.
   - Another member humorously suggested that the coil whine might be loud enough to *play music*.
- **Deciding On GPU Choices**: Discussion revolved around choosing a budget-friendly GPU, with suggestions like the **7900 XTX** being mentioned as a good option compared to **NVIDIA** cards.
   - The consensus leaned towards waiting for next-gen GPUs due to anticipated **high prices**.
- **Bottlenecking Argument Sparks Debate**: A user argued that **bottlenecking** does not exist, while others pointed out that a weaker CPU can delay frame delivery to the GPU.
   - The debate highlighted varying opinions on the influence of CPU performance on overall **FPS**.
- **VRChat's RAM Hunger**: VRChat's **VRAM needs** were brought up, with implications that it could rapidly consume available memory leading to performance issues.
   - Users noted that many gamers opt for **4090s** due to these demands.
- **Next-Gen GPU Fears**: Concerns were raised that future **RTX 50s** could come at an **outrageously high price** compared to current offerings.
   - Despite the worries, AMD's promise to deliver competitive performance at a lower cost created cautious optimism.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1319122047825350748)** (4 messages): 

> `tl.dot input shape requirements, AMD GPU performance vs PyTorch, Nvidia Hopper warp-specialization deletion, Triton performance optimization` 


- **tl.dot needs >= 16 for Input Shapes**: Input shapes for **tl.dot** should have **M >= 16, N >= 16, and K >= 16**, primarily due to tensor core requirements.
   - A user queried whether **tl.dot** can default to using *CUDA cores* for computations when M, N, or K are less than **16**.
- **Searching for faster AMD GPU kernels**: A user asked if anyone has found a kernel that performs faster on **AMD GPUs** like the **RX 7900** compared to **PyTorch/rocBLAS**.
   - Another user noted that as of now, **Triton performance** has not yet surpassed their **BLAS implementations**, particularly for **Navi31**.
- **Nvidia Hopper's warp-specialization feature removed**: A user discovered that the **warp-specialization** feature of **Nvidia Hopper** has been removed in **Triton 3.x**.
   - They inquired about possible techniques to achieve better performance with **Triton** following this change.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1319297531951185930)** (5 messages): 

> `cudaMemcpy performance, CUTLASS tma_store_wait function behavior, Documentation on TMA operations` 


- **Exploring Faster Alternatives to cudaMemcpy**: A member inquired if there are faster methods to copy small data sizes (e.g., 12 bytes) to device memory than using **cudaMemcpy**, which reportedly takes about **1-2us**.
   - This raises questions about potential optimizations for memory transfers in CUDA programming.
- **tma_store_wait may complete automatically**: A member observed that after executing a TMA-store operation using **tma_store_wait** in CUTLASS, waiting might not be necessary as it seems to complete automatically.
   - This suggests that its behavior resembles **expect_tx**, prompting discussions on its efficiency in handling operations.
- **Need for Documentation Confirmation**: In response to the discussion on TMA operations, a member asked for documentation that clarifies whether the functionality is supported as initially believed.
   - The request emphasizes the importance of accurate and accessible documentation for development practices.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

0x000ff4: any one contributin to keras/pytorch?
  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1319327988533166122)** (21 messages🔥): 

> `Genesis AI, Sim2Real Technology, CARLA Simulator Update, Synthetic Data Generation for Autonomous Driving, Dexterous Task Applications` 


- **Genesis AI Sparks Interest**: The community expressed excitement about [Genesis](https://genesis-embodied-ai.github.io/) and its potential applications, particularly highlighting the impressive water droplet demo.
   - One member remarked, *'Super cool thing,'* showcasing the appeal of new tools in AI.
- **Exploring Sim2Real Concepts**: Discussion pivoted to [Sim2Real](https://www.example.com), focusing on its capability to transfer skills from simulation to real-world applications, with emphasis on tasks like cooking and assembling.
   - One user questioned, *'I wonder how it does on dexterous tasks,'* indicating interest in its practical functionality.
- **CARLA Simulator Receives Major Upgrade**: The team celebrated the release of **CARLA version 0.10.0**, which enhanced visual fidelity through a migration to **Unreal Engine 5.5** and introduced advanced features like [Lumen](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine) and [Nanite](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine).
   - This update includes *upgraded environments and assets*, showcasing advancements in rendering technology.
- **Synthetic Data Generation Discussions Emerge**: There was speculation regarding **Waymo's** approach to data, with a member noting that *Waymo might also generate synthetic data* alongside real driving data.
   - Links to relevant articles were shared, including [this research](https://waymo.com/research/embedding-synthetic-off-policy-experience-for-autonomous-driving-via-zero/) on embedding synthetic data for autonomous driving.
- **Future of Synthetic Data in Autonomous Driving**: Members elaborated on how future advancements might see the majority of simulated data being synthetic, possibly integrating tools like **Genesis**.
   - The conversation concluded with curiosity about how well such frameworks scale, noting their potential for generating accurate vehicle dynamics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://carla.org/2024/12/19/release-0.10.0/">CARLA 0.10.0 Release with Unreal Engine 5.5!</a>: Unreal Engine 5.5 migration, brand new assets, upgraded Town 10, remodeled vehicles, open-cast mine map</li><li><a href="https://arxiv.org/html/2406.09386v1">SimGen: Simulator-conditioned Driving Scene Generation</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1319261477802213438)** (1 messages): 

> `Image Analysis, User Concerns` 


- **Discussion sparked over image analysis**: A member referenced an image in the channel which triggered discussions about its content and relevance.
   - While no specific details were quoted about the image, the engagement indicates it caught the group's attention.
- **Humorous engagement with image**: The member's response included a light-hearted remark indicating amusement with the contents of the image shared.
   - This humor suggests a lively atmosphere in the chat, contributing to the overall enjoyment of the discussion.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1319053856751095948)** (1 messages): 

> `MatX hiring, LLM accelerator ASIC development, Low level compute kernel author roles, ML performance engineer roles, In-person work culture` 


- **Join MatX as they build LLM accelerator ASIC**: MatX is hiring for positions including **low level compute kernel author**, **compiler**, and **ML performance engineer** roles. Interested candidates can find more information in their [job listing here](https://grnh.se/2b337cb08us).
   - They value **efficiency** and **high-quality solutions**, open to applicants from fresh graduates to seasoned professionals.
- **MatX encourages innovative problem-solving**: The team emphasizes the need to *consider new approaches*, often abandoning traditional methods for better alternatives that suit their context.
   - They prioritize making big decisions with deep understanding, indicating that thorough reasoning often outweighs extensive testing.
- **Emphasis on high-trust team environment**: MatX promotes a culture rooted in inviting and including diverse perspectives within their high-trust team.
   - Supportive teamwork is essential, as they believe it is crucial for tackling complex challenges collectively.



**Link mentioned**: <a href="https://grnh.se/2b337cb08us">MatX</a>: &lt;header&gt;&lt;h2&gt;MatX: faster chips for LLMs&lt;/h2&gt;&lt;/header&gt;&lt;div id=&quot;maincontent&quot;&gt;&lt;h3&gt;Come work with us!&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;Whether we&#x27;re working...

  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1319229746688819251)** (1 messages): 

> `Sparsity Design, Sparsifier Functionality, Sparsify Kernel Optimization, Demo for Sparsify Usage` 


- **Understanding the Role of Sparsifier**: A query was raised about the functionality of the **Sparsifier**, specifically if it's responsible for determining the **sparsity pattern**.
   - It was clarified that the **Sparsifier** indeed determines the pattern but its output's interaction with the kernel optimization process via **sparsify_** was questioned.
- **Interaction between Sparsifier and Sparsify**: The user inquired whether the **sparsify_** function consumes the output of the **Sparsifier** during its operation.
   - Understanding this interaction is crucial for optimizing sparsity in designs and further guidance was sought.
- **Request for Sparsify Usage Demo**: A request was made for a demonstration about the usage of **sparsify_**, highlighting the need for practical examples.
   - This demo would provide insights on how to effectively implement the **sparsity design** in real scenarios.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1319282562580877314)** (1 messages): 

> `alma Python Package, Model Benchmarking, PyTorch Conversion Options` 


- **Open-Sourcing alma: The Benchmarking Marvel**: A duo has just open-sourced their project, **alma**, a Python package designed for benchmarking the speed of over **40 PyTorch conversion options** with a single function call.
   - It boasts features like *graceful failure handling* and *isolated processes* for safer testing, aiming to simplify CI integration.
- **Future Integration Plans for alma**: Future developments are aimed at adding more conversion options and integrating with *JAX*, *llama.cpp*, and *VLLM* for enhanced versatility.
   - The creators invite users to share ideas through GitHub, emphasizing community engagement to broaden functionality.
- **Real-World Performance Examples**: Example outputs show impressive results, with **EAGER** mode achieving a throughput of **282395.70 samples/second** on a CUDA device.
   - In *EXPORT+EAGER*, performance slightly enhanced with **305974.83 samples/second**, showcasing the package's efficiency.



**Link mentioned**: <a href="https://github.com/saifhaq/alma">GitHub - saifhaq/alma</a>: Contribute to saifhaq/alma development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

kimishpatel: what i cam here for 🙂
  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1319135025979199549)** (5 messages): 

> `Cost of GPUs on Vast AI, Generative Flow Networks, ARC Prize Daily Puzzle, Training Smaller Models, Synthesizing Riddles` 


- **Vast AI Offers Cheap GPU Options**: A member noted that GPUs on [Vast AI](https://vast.ai) are very affordable, specifically mentioning that the **3070** is a cost-effective choice for personal use.
   - Another member shared their experience, indicating they had previously checked only **Lambda** and **Runpod** for GPU options.
- **Exploring Generative Flow Networks for Dataset Generation**: Discussion arose around **Generative Flow Networks** as a promising method for **synthetic dataset generation**, particularly in scenarios with costly oracle rewards.
   - A member shared a [paper](https://arxiv.org/pdf/2106.04399) on the topic, highlighting the potential to reduce the labeling of **(problem, reward) pairs**.
- **Solving the ARC Prize Daily Puzzle**: A member celebrated successfully solving the ARC Prize Daily Puzzle, emphasizing the daily challenge at **12pm UTC** that requires sorting input.
   - They expressed skepticism about autoregressive models, noting their limitations in sorting by design unless employing some prior reasoning.
- **Training with Smaller Models for Efficiency**: A member mentioned the practicality of iterating with smaller models trainable on **24G**, suggesting an efficiency gain compared to larger models.
   - This aligns with the previous discussion about exploring low-cost GPU options for optimal training results.
- **Challenges in Riddle Synthesis**: A member reflected on the difficulties of finding the right representation for generating riddles, emphasizing the need for **input boards** and transformations.
   - They highlighted the importance of ensuring that all relevant parameters for transformation can be derived from examples provided.



**Link mentioned**: <a href="https://arcprize.org/play">ARC Prize - Play the Game</a>: Easy for humans, hard for AI. Try ARC-AGI.

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1319033351595757648)** (87 messages🔥🔥): 

> `AI Agentic Systems, Gemini 2.0 Flash Thinking, Databricks Funding, ModernBERT Release, Alec Radford Departure from OpenAI` 


- **AI Agentic Systems on the Rise**: Anthropic shared insights on successful implementations of agentic systems and emphasized building with simple, composable patterns, suggesting 2025 will be pivotal for the field.
   - The blog post from Anthropic highlights best practices and evolving definitions of agents and workflows in AI.
- **Gemini 2.0 Flash Thinking Dominates**: The introduction of Gemini 2.0 Flash Thinking showcased its reasoning capabilities, achieving top rankings across various categories and outperforming its predecessor in multiple tasks.
   - According to reports, this model explicitly shows its thought processes, improving reasoning performance more effectively.
- **Databricks Secures Major Funding**: Databricks announced a Series J funding round led by Thrive Capital, raising $10 billion and achieving a valuation of $62 billion while expecting to cross a $3 billion revenue run rate.
   - This marks significant momentum for the company, reflecting a 60% year-over-year growth largely driven by AI demand.
- **ModernBERT Release Sparks Interest**: The launch of ModernBERT, which offers improvements over older models with longer context and enhanced performance, has captured significant attention in the AI community.
   - Discussion around its features and potential applications has highlighted the excitement for its integration into existing workflows.
- **Alec Radford Leaves OpenAI**: Alec Radford, a key figure in OpenAI's GPT development, is departing to pursue independent research, raising questions about the organization's future.
   - This personnel shift has led to speculation regarding OpenAI's direction amidst other recent changes in the industry.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/vikhyatk/status/1869605301596631191">Tweet from vik (@vikhyatk)</a>: everyone’s posting 🤯 tweets about this but i can tell no one has tried it out.because i did and it says the method generate doesn’t exist on module genesisQuoting Allen T. (@Mr_AllenT) This is the cr...</li><li><a href="https://x.com/justinlin610/status/1869793885540757715?s=46">Tweet from Junyang Lin (@JustinLin610)</a>: so sorry for making you expect. nothing will happen tonight. we still need to make things better for this release. will be back very soon.</li><li><a href="https://apply.ai.engineer">AI Engineer Summit</a>: The highest-signal technical AI event of the year. For AI Engineers &amp; Leaders, Feb 20 - 21, 2025.</li><li><a href="https://x.com/presidentlin/status/1869745206842794047?s=46">Tweet from Lincoln 🇿🇦 (@Presidentlin)</a>: @Alibaba_Qwen They made a reasoning VL model?</li><li><a href="https://genesis-embodied-ai.github.io/">Genesis</a>: no description found</li><li><a href="https://www.anthropic.com/research/building-effective-agents">Building effective agents</a>: A post for developers with advice and workflows for building effective AI agents</li><li><a href="https://x.com/elevenlabsio/status/1869462840941461941">Tweet from ElevenLabs (@elevenlabsio)</a>: Meet Flash. Our newest model that generates speech in 75ms + application & network latency.You’ve never experienced human-like TTS this fast.</li><li><a href="https://x.com/lmarena_ai/status/1869793847548817563?s=46">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: Breaking news from Chatbot Arena⚡🤔@GoogleDeepMind&#39;s Gemini-2.0-Flash-Thinking debuts as #1 across ALL categories!The leap from Gemini-2.0-Flash:- Overall: #3 → #1- Overall (Style Control): #4 → #...</li><li><a href="https://x.com/steph_palazzolo/status/1869848094009110826">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: Big OpenAI personnel news w/ @erinkwoo: Alec Radford, the lead author of OpenAI&#39;s original GPT paper, is leaving to pursue independent research.  https://www.theinformation.com/briefings/senior-op...</li><li><a href="https://x.com/swyx/status/1869825047051022464">Tweet from swyx (@swyx)</a>: this is what it looks like to work with AGIs like @benghamine behind the sceneswondering when @recraftai or @GeminiApp  or @xai can  match this workflowQuoting jason liu (@jxnlco) oohh shit this is hu...</li><li><a href="https://x.com/_sholtodouglas/status/1869798291535446383">Tweet from Sholto Douglas (@_sholtodouglas)</a>: I really like the thoughts in this problem, a cute example of out of the box thinking. As models get stronger, taking them seriously will continue to be the right way to understand both the current ge...</li><li><a href="https://www.luzia.com/en">Luzia: Your intelligent assistant at a click</a>: Access the power of AI easily and for FREE. Luzia (I am Luzia) helps you in your day to day life with thousands of tasks, whether at work, at school, in your social moments or while pursuing your pass...</li><li><a href="https://www.databricks.com/company/newsroom/press-releases/databricks-raising-10b-series-j-investment-62b-valuation">Databricks is Raising $10B Series J Investment at $62B Valuation</a>: Funding led by new investor Thrive Capital Company expects to cross $3B in revenue run rate and achieve positive free cash flow in fourth quarter  </li><li><a href="https://x.com/noamshazeer/status/1869789881637200228?s=46">Tweet from Noam Shazeer (@NoamShazeer)</a>: We’ve been *thinking* about how to improve model reasoning and explainabilityIntroducing Gemini 2.0 Flash Thinking, an experimental model trained to think out loud, leading to stronger reasoning perfo...</li><li><a href="https://huggingface.co/blog/modernbert">Finally, a Replacement for BERT: Introducing ModernBERT</a>: no description found</li><li><a href="https://x.com/odysseyml/status/1869417873938219360?s=46">Tweet from Odyssey (@odysseyml)</a>: Today is a big day at @odysseyml.We&#39;re sharing Explorer, our first generative world model. We think world models are the next frontier for AI, enabling wonderful new things.To help shape this, Ed ...</li><li><a href="https://x.com/ehuanglu/status/1869549996045160558?s=46">Tweet from el.cine (@EHuanglu)</a>: AI voice is taking over!ElevenLabs just launched Flash 2.5, capable of generating lifelike film dialogue from text almost instantly.From now on, don’t believe everything you hear!</li><li><a href="https://x.com/altryne/status/1869835859727393234?s=46">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: I evaluated o1-2024-12-17 (with all 3 reasoning efforts) and gemini-2.0-flash-thinking-exp-1219 on 10 challenging questions from @AIExplainedYT simple bench and got some surprising results! Flash thin...</li><li><a href="https://x.com/alexalbert__/status/1869812081597526079?s=46">Tweet from Alex Albert (@alexalbert__)</a>: 2025 will be the year of agentic systemsThe pieces are falling into place: computer use, MCP, improved tool use. It&#39;s time to start thinking about building these systems.At Anthropic, we&#39;re se...</li><li><a href="https://x.com/officiallogank/status/1869789820308074837?s=46">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Just when you thought it was over...  we’re introducing Gemini 2.0 Flash Thinking, a new experimental model that unlocks stronger reasoning capabilities and shows its thoughts.The model plans (with th...</li><li><a href="https://x.com/zhou_xian_/status/1869511650782658846?s=46">Tweet from Zhou Xian (@zhou_xian_)</a>: Everything you love about generative models — now powered by real physics!Announcing the Genesis project — after a 24-month large-scale research collaboration involving over 20 research labs — a gener...</li><li><a href="https://x.com/hamelhusain/status/1869808528258679057?s=46">Tweet from Hamel Husain (@HamelHusain)</a>: For those wondering how ModernBert fits into RAG, a good starting point is @bclavie ‘s primer “Beyond the basics of RAG” He talks about the various kinds of encoders, when to use them, and different t...</li><li><a href="https://x.com/_sholtodouglas/status/1869796444502462527?s=46">Tweet from Sholto Douglas (@_sholtodouglas)</a>: A taste of what we&#39;ve been thinking about recently :)Try it out! Its still a little raw, we expect it to have sharp edges - but it represents incredible algorithmic progress on test time compute.A...</li><li><a href="https://x.com/jeremyphoward/status/1869786023963832509?s=46">Tweet from Jeremy Howard (@jeremyphoward)</a>: I&#39;ll get straight to the point.We trained 2 new models. Like BERT, but modern. ModernBERT.Not some hypey GenAI thing, but a proper workhorse model, for retrieval, classification, etc. Real practic...</li><li><a href="https://x.com/daytonaio/status/1869727933046112578">Tweet from Daytona.io (@daytonaio)</a>: 🚀 Mapping the future of AI development! Here&#39;s your comprehensive guide to the AI Enablement Stack - from infrastructure to autonomous agents—a community-driven, open-source effort.A deep dive in...</li><li><a href="https://youtu.be/a0bEU83P8g8?si=9V0yJeqtWnhVicKI"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1319073093133537395)** (67 messages🔥🔥): 

> `OpenInterpreter 1.0 updates, Running commands in server mode, Google Gemini 2.0 multimodal, Local vs server command execution, OS mode functionality` 


- **OpenInterpreter 1.0 supports vision models**: The 1.0 branch supports models with vision, allowing users to install via [GitHub](https://github.com/OpenInterpreter/open-interpreter.git) with `pip install git+https://github.com/OpenInterpreter/open-interpreter.git@development`.
   - Experiments show the `--tools gui` command is functional, connecting to different models or APIs as needed.
- **Server mode operation questions**: A user queried how commands are executed when OI runs as a server, wondering if they run locally or on the server.
   - It was noted that some users run it in regular mode and use SSH for access, but they consider integrating a front end for efficiency.
- **Inquiry about Google Gemini 2.0 capabilities**: A user expressed curiosity about the performance of Google Gemini 2.0 multimodal capabilities, specifically in OS mode.
   - Interest exists in how the new model compares to existing systems, particularly regarding its command execution features.
- **Control over local machines using OI**: Discussions around OpenInterpreter's ability to control local systems revealed limitations with mouse and code execution functionality.
   - Users reported that there are still issues with getting the expected OS mode capabilities to function fully.
- **Cleaning installations of OpenInterpreter**: Concerns were raised about needing a clean installation due to issues faced with OpenInterpreter, especially after making multiple configurations.
   - Users discussed removing certain flags and adjusting commands to resolve errors and uncertainties about the setup process.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1319336699368706160)** (1 messages): 

> `O1 Channel Exploration, Understanding documentation` 


- **Seeking clarity on O1 functionality**: A member expressed a need for a simpler explanation of how the **O1** channel works after exploring it.
   - They acknowledged reading the documentation but still felt like a *noob* and appreciated any help offered.
- **Documentation not helpful enough**: The same member pointed out that despite their efforts in reading the docs, it didn't provide the necessary clarity.
   - They are looking for straightforward guidance to get up to speed with **O1**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1319032281507041310)** (62 messages🔥🔥): 

> `LM Studio Model Loading Issues, Mobile Access to LM Studio, GPU Driver Problems, Image Input Models for LM Studio, Known Issues with AMD Drivers` 


- **LM Studio Error with Model Loading**: A user reported encountering an error stating ```Safetensors header is unexpectedly large: bytes=2199142139136``` when trying to load models, indicating potential issues with compatibility or file corruption.
   - Another user confirmed this message appeared for the MLX version of Llama 3.3 as well, leading them to redownload models in hopes of resolving the issue.
- **Connecting to LM Studio from Mobile Devices**: Members discussed using LM Studio via mobile, with one user sharing an iOS app, **3Sparks Chat**, that connects to the LM Studio server on a PC or Mac.
   - However, requests for an Android version were met with disappointment as there are currently no mobile solutions available.
- **Issues with Latest AMD Drivers**: Users detailed problems with **AMD 24.12.1 drivers**, which reportedly cause system stuttering when loading models in LM Studio, indicating broader conflict with the llama.cpp rocm library.
   - Recommendations included downgrading to previous driver versions to mitigate performance issues experienced by some users.
- **Image Input Models for LM Studio**: A user inquired about image input models for LM Studio, specifically for PC users, receiving information on the **mlx-community/Llama-3.2-11B-Vision-Instruct-4bit** model, which faced several loading issues.
   - There were discussions on model compatibility, with concerns raised about other formats not supporting Windows runtime.
- **General Hardware Configuration Discussion**: Several users exchanged details about their hardware specifications, specifically on the compatibility of the **7900XTX** GPU when using LM Studio and how variations in configuration can affect performance.
   - One user noted differences in experience despite similar configurations, indicating possible discrepancies in hardware performance or driver interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://apps.apple.com/us/app/3sparks-chat/id6736871168">‎3sparks Chat</a>: ‎Whether you’re using LM Studio or Ollama to run LLMs locally on your Mac, PC, or Linux, or accessing the power of the OpenAI API, 3Sparks Chat is your go-to mobile client. Chat with your LLMs anytime...</li><li><a href="https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit">mlx-community/Llama-3.2-11B-Vision-Instruct-4bit · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1319398696961577052)** (3 messages): 

> `Silicon Chips Performance, Benchmark Comparisons` 


- **Higher-end Silicon Chips Questioned for Speed**: A member inquired whether **prompt processing** is faster on the **higher-end silicon chips** (max, pro, ultra) due to improved **memory bandwidth**.
   - Another member noted that these chips are not as fast as a **30/4090** model.
- **Access to Llama.cpp Benchmarks**: A member shared that **llama.cpp** maintains benchmarks for each model at their GitHub discussion page.
   - Details can be found in this [GitHub discussion](https://github.com/ggerganov/llama.cpp/discussions/4167).


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1319338173586997330)** (1 messages): 

> `Price reductions, Market competition` 


- **Gryphe Cuts Price by 7%**: The price of [gryphe/mythomax-l2-13b](https://openrouter.ai/gryphe/mythomax-l2-13b) has dropped by **7%** this morning, continuing the trend of price reductions in the market.
   - *This is part of ongoing price wars* in the competitive landscape of AI models.
- **Qwen Slashes Prices by 7.7%**: Another significant **7.7% drop** occurred on [qwen/qwq-32b-preview](https://openrouter.ai/qwen/qwq-32b-preview) as the price wars heat up.
   - *These adjustments reflect the fierce competition* among leading AI providers.
- **Mistral-Nemo Takes a 12.5% Hit**: [mistralai/mistral-nemo](https://openrouter.ai/mistralai/mistral-nemo) has seen a **12.5%** price cut, indicating a proactive pricing strategy.
   - This reflects the **intensifying market dynamics**, as companies vie for customer attention and market share.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/gryphe/mythomax-l2-13b>)">MythoMax 13B - API, Providers, Stats</a>: One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge. Run MythoMax 13B with API</li><li><a href="https://openrouter.ai/qwen/qwq-32b-preview>)">QwQ 32B Preview - API, Providers, Stats</a>: QwQ-32B-Preview is an experimental research model focused on AI reasoning capabilities developed by the Qwen Team. As a preview release, it demonstrates promising analytical abilities while having sev...</li><li><a href="https://openrouter.ai/mistralai/mistral-nemo>)">Mistral Nemo - API, Providers, Stats</a>: A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA.The model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chines...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1319300285755293787)** (1 messages): 

> `AI Ecosystem Maps, Crowdsourced AI Enablement Stack` 


- **Need for a Crowdsourced AI Enablement Stack**: Many VC firms have published their AI ecosystem maps, but there is a demand for a truly **crowdsourced** and **open-source AI enablement stack**.
   - This initiative aims to keep developers informed on what tools to use, ensuring they won't waste time in their projects. More details can be found on [GitHub](https://github.com/daytonaio/ai-enablement-stack).
- **Feedback Request on AI Enablement Logic**: There is an open call for contributions and feedback on the **logic and structure** of this AI enablement approach.
   - The goal is to create an up-to-date resource for developers, encouraging community input and collaboration.



**Link mentioned**: <a href="https://github.com/daytonaio/ai-enablement-stack">GitHub - daytonaio/ai-enablement-stack: A Community-Driven Mapping of AI Development Tools</a>: A Community-Driven Mapping of AI Development Tools - daytonaio/ai-enablement-stack

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1319034462800973918)** (62 messages🔥🔥): 

> `DeepSeek Models, OpenRouter Issues, Model and API Discussion, Data Management, User Experience Feedback` 


- **Exploration of DeepSeek Models for Learning**: Users are experimenting with **DeepSeek-v2** and **DeepSeek V2.5** for coding assistance, emphasizing the benefit of inputting entire GitHub repos for better understanding of complex projects.
   - One user mentioned how **DeepSeek** helped with code optimization and commenting, while another warned against using it for advanced code creation.
- **OpenRouter User Support Challenges**: Several users reported issues with **OpenRouter**, including unexpected account problems and unclear responses from support regarding missing balances.
   - User frustrations were evident as one sought clarity on their balance disappearing, highlighting the need for improved communication from support.
- **API and Model Capability Discussions**: There were questions about the **o1 reasoning_effort** parameter's accessibility, indicating users' interest in model capabilities and interfaces.
   - Users also discussed the utility of different models and the importance of privacy settings for sensitive tasks, especially regarding healthcare data.
- **User Experiences with OpenRouter Features**: Participants shared their perspectives on the interface and its suitability for various uses, with some suggestions for improvements in user navigation.
   - There was a discussion about interface tagging, clarity, and the need for a more streamlined user experience in AI applications.
- **Community Interaction and Humor**: Members participated in light-hearted banter and joke discussions about user bios and the silliness of online personas.
   - The community seemed supportive, with users engaging in fun commentary alongside serious inquiries about the platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/terms#_4_-payment">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/oauth">OAuth PKCE | OpenRouter</a>: Secure user authentication via OAuth
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1319147059042979862)** (1 messages): 

> `Programmatic feature requests, Provider API integration` 


- **Request for Programmatic Feature Implementation**: A member expressed interest in seeing a **programmatic version** of a specific feature, emphasizing the ability to pass the **provider API key** with the request.
   - *I’d love to see a programmatic version of this feature* highlights the desire for increased functionality in API integration.
- **Interest in API Key Functionality**: The same member reinforced the need for passing the **provider API key** with requests implicitly, to streamline access and improve user experience.
   - This indicates a broader interest in API features that cater to developers' needs for flexibility and efficiency.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1319039562802266133)** (47 messages🔥): 

> `GitHub Copilot Free Tier, Granite 3.1-8B-Instruct Model, LM Studio for Local LLMs, Model Context Protocol Testing, Gemini Flash Thinking Experimental` 


- **GitHub Copilot now free for all**: Announced a new [free tier for GitHub Copilot](https://x.com/code/status/1869449373995708703) available immediately, with no trial or subscription required.
   - Users can take advantage of this offer without needing to provide a credit card, and it intriguingly includes Claude for enhanced functionality.
- **Granite 3.1-8B-Instruct impresses users**: Users are excited about the **Granite 3.1-8B-Instruct model**, which has been fine-tuned for long context tasks and performs well in real-world applications.
   - Related model resources can be found on the [Granite GitHub](https://github.com/ibm-granite/granite-3.1-language-models) and [documentation](https://www.ibm.com/granite/docs/).
- **LM Studio offers convenient model access**: [LM Studio](https://lmstudio.ai/) allows users to run LLMs locally, chat with documents, and download model files from Hugging Face.
   - It supports architectures like Llama 3.2, Mistral, and Qwen 2.5, catering to those wanting offline functionality.
- **Experimenting with Model Context Protocol**: A user plans to implement a quick server in Bash to test the **model context protocol** despite initial reservations.
   - This experiment aims to gauge the practical value of the protocol in a real-world setting.
- **Gemini Flash Thinking impresses**: Gemini 2.0 Flash Thinking produced a witty response regarding the meme 'Hello there!' from Star Wars, highlighting its contextual relevance.
   - The final response cleverly weaved in cultural nuances and character specifics, showcasing the model's engaging capacities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/code/status/1869449373995708703">Tweet from Visual Studio Code (@code)</a>: Announcing GitHub Copilot Free!A new free tier for GitHub Copilot, available for everyone today in @codeNo trial. No subscription. No credit card required.Learn more in our blog: http://aka.ms/copilot...</li><li><a href="https://huggingface.co/ibm-granite/granite-3.1-8b-instruct">ibm-granite/granite-3.1-8b-instruct · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ZooojV4ZDMw"> - YouTube</a>: no description found</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1319041270777249862)** (2 messages): 

> `Agent Message Formatting, Fine-Tuning Dataset Consistency` 


- **Agent messages lack sentence separation**: A member noted that the latest messages from the agent are missing periods between sentences, indicating a formatting quirk.
   - They compared this behavior with **gpt-4o**, confirming it doesn't exhibit the same issue.
- **Using uniform instructions in fine-tuning**: A member inquired about the implications of using the same instruction across a fine-tuning dataset consisting of 'Question' and 'Answer' pairs.
   - Their concern centered on whether this approach could lead to **suboptimal model performance** compared to varying the instructions.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1319241909524238367)** (2 messages): 

> `Genesis Project, Generative Physics Engine, Open Source Robotics Simulation` 


- **Genesis Project Revolutionizes Robotics with Real Physics**: The [Genesis project](https://x.com/zhou_xian_/status/1869511650782658846) has been announced as a generative physics engine capable of creating **4D dynamical worlds**, significantly enhancing robotics and physical AI applications.
   - Developed in pure Python, it boasts a simulation speed up to **430,000 times** faster than real-time, with training times for robotic locomotion policies reduced to just **26 seconds** on a single RTX4090.
- **Open Source Access to Genesis Physics Engine**: The Genesis physics engine is [fully open source](https://github.com/Genesis-Embodied-AI/Genesis), inviting collaboration and contributions from the community to enhance its functionality.
   - It integrates advanced physics solvers to simulate entire physical worlds, aiming for a completely **automated data generation** process for robotics.
- **Tutorial for Robotic Locomotion with Genesis**: A comprehensive [tutorial](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/locomotion.html) explains how to train a robotic locomotion policy utilizing the Genesis physics engine.
   - This training process showcases the engine's efficiency, which is **10-80 times faster** than existing GPU-accelerated solutions like Isaac Gym.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/zhou_xian_/status/1869511650782658846">Tweet from Zhou Xian (@zhou_xian_)</a>: Everything you love about generative models — now powered by real physics!Announcing the Genesis project — after a 24-month large-scale research collaboration involving over 20 research labs — a gener...</li><li><a href="https://github.com/Genesis-Embodied-AI/Genesis">GitHub - Genesis-Embodied-AI/Genesis: A generative world for general-purpose robotics &amp; embodied AI learning.</a>: A generative world for general-purpose robotics &amp; embodied AI learning. - Genesis-Embodied-AI/Genesis
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1319209126903152700)** (37 messages🔥): 

> `Mojo Indexing and Casting, SIMD Keying in Dict, Running Mojo on Android, Python Integration Ideas, Negative Indexing Debate` 


- **Debate on Mojo Indexing Practices**: A discussion emerged regarding the use of **Int** for indexing in **Mojo**, with opinions split on whether negative indexing should be integrated into default implementations or if alternatives like `.last()` suffice.
   - *Darkmatter* argued that negative indexing is often a programming error, stating it introduces unnecessary operational costs, while others highlighted its common usage in languages like **Python**.
- **Bug in SIMD Structs and Dicts**: A significant bug with missing **scaling_cur_freq** in **Mojo** was mentioned, causing segmentation faults when using a struct based on SIMD as a key in **Dicts** and also affecting benchmarks.
   - This bug is documented in [GitHub Issue #3781](https://github.com/modularml/mojo/issues/3781) which details steps to reproduce and seeks resolution within the suggested **6-week window**.
- **Running Mojo on Native Android**: Some members discussed the possibility of running **Mojo** on native Android, mentioning a setup via **Magic** in a Docker container, although it is considered 'wildly unsupported'.
   - It was noted that while self-setup is possible, licensing rules prevent the creation of a publicly distributable Docker image.
- **Python Integration Considerations**: There was an inquiry about creating Python types for Mojo, specifically examining the integration of **SIMD** and conditional conformance to enable support for various data types.
   - Concerns were raised about maintaining separate handling for integral and floating-point types due to ABI requirements, while the idea of supporting arbitrary bit-width integers was met with enthusiasm.
- **Safety and Efficiency in Indexing**: The discussion on indexing raised safety concerns, discussing the implications of implicit type casting from **UInt** to **Int** and the performance costs associated with checks for negative indices.
   - *Darkmatter* suggested that while overloads could be implemented, they would complicate existing type casting rules and potentially introduce ambiguity.



**Link mentioned**: <a href="https://github.com/modularml/mojo/issues/3781">[BUG] Segfault if using a struct based on SIMD as key in Dict · Issue #3781 · modularml/mojo</a>: Bug description When using a struct containing a sufficiently large SIMD as key in a Dict, a segmentation fault is encountered. Steps to reproduce Execute the following code: from collections impor...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1319059635214487624)** (28 messages🔥): 

> `Synthetic Data Primer, Rate Limiting in DataBricks, DSPy Signature Outputs, Provisioned Throughput Costs, LiteLLM Proxy Layer` 


- **Explainer on Synthetic Data in Progress**: A member is working on an explainer that covers the fundamentals of **synthetic data**, how it is created, its uses, and its impact on model capabilities.
   - They are looking for community input on specific questions or areas of curiosity about synthetic data.
- **Rate Limiting Solutions in DataBricks**: A member discussed the potential to implement a **rate limiter** within DataBricks due to high costs incurred from throughput allocations.
   - Another suggested using the **LiteLLM** proxy layer for features like rate-limiting and budgeting.
- **Question on DSPy Signature Class Outputs**: A user inquired about examples of producing a **dspy.Signature** as a class type instead of a string, expressing interest in using the DSPy framework.
   - They are exploring the feasibility of directly returning a signature with specified fields.
- **Concerns Over Provisioned Throughput Costs**: A member recounted their experience with **high costs** from provisioned throughput in DataBricks, raising concerns about unnecessary charges.
   - They clarified the importance of enabling the **scale to 0** option to avoid charges when not in use.
- **Deploying LiteLLM in DataBricks**: There was a discussion about whether the **LiteLLM proxy** could be deployed within a DataBricks notebook or if a separate VM is necessary.
   - One member confirmed LiteLLM can be managed alongside the service within a controlled environment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.databricks.com/product/pricing/foundation-model-serving">Mosaic AI Foundation Model Serving</a>: no description found</li><li><a href="https://docs.databricks.com/en/ai-gateway/index.html">Mosaic AI Gateway</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1319064531049386046)** (3 messages): 

> `Multi-agent systems, Vectara RAG capabilities, AI journey survey` 


- **Building Multi-Agent Systems with LlamaIndex**: A post discusses how to evolve from a **single agent** to a coordinated **multi-agent system** using LlamaIndex, providing practical code examples.
   - It emphasizes the importance of **agent factories** in this transition, detailed in the full article [here](https://t.co/lbhFDbSabS).
- **Unlocking Vectara's RAG Power**: Discover how to leverage **Vectara's** powerful **RAG capabilities** including loading data and querying with streaming and reranking options.
   - The post addresses building **agentic RAG applications** while highlighting the full capabilities of Vectara's managed service [here](https://t.co/traVaQiUt3).
- **Participate in Vercel's State of AI Survey**: A call to action invites community members to share their progress in their **AI journey** through @vercel's **State of AI Survey**.
   - Participants can contribute to the understanding of the AI landscape by visiting [this link](https://t.co/O3sYZ6L9Gq).



**Link mentioned**: <a href="https://t.co/O3sYZ6L9Gq">State of AI Developer Survey</a>: Share your experiences, challenges, and insights, and help shape the future of AI-driven innovation.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1319241784211013642)** (23 messages🔥): 

> `HuggingFaceEmbedding model loading, Azure OpenAI embedding rate limits, TextNode insert errors` 


- **HuggingFaceEmbedding can't load from local**: A user encountered issues while trying to load a HuggingFace embedding model from local storage and received a warning about creating a new model with mean pooling.
   - Another user clarified that simply providing the model name would check the cache folder first before downloading it unnecessarily.
- **Solutions for Azure OpenAI embedding rate limits**: One user reported persistent rate limit errors with Azure OpenAI embedding models and sought suggestions to resolve this issue.
   - A suggestion included increasing the max retries and ingesting documents more slowly to avoid rate limiting issues.
- **Confusion over inserting TextNodes**: A user faced an `AttributeError` when trying to insert TextNodes into the index, indicating a missing `get_doc_id` attribute.
   - It was advised that the proper method for inserting nodes is `insert_nodes`, and that processing them one at a time might help with rate limiting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/.">Local Embeddings with HuggingFace - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/issues/7879">[Question]:  Consistently getting rate limit error when building index · Issue #7879 · run-llama/llama_index</a>: Question Validation I have searched both the documentation and discord for an answer. Question I am using the basic code to index a single text document with about 10 lines from llama_index import ...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1319351653492850729)** (1 messages): 

> `Vision Parse, PDF to Markdown` 


- **Vision Parse Library Launches for Markdown Conversion**: A member shared the launch of [Vision Parse](https://github.com/iamarunbrahma/vision-parse), an open-source Python library that converts PDF documents into well-formatted markdown content using advanced Vision Language Models.
   - *State-of-the-art* technology aims to enhance the conversion experience with **great formatting** options.
- **Excitement Around Open Source Contributions**: The community showed enthusiasm for the release of Vision Parse, highlighting its potential to simplify document handling for developers.
   - Members discussed the importance of *open-source projects* in fostering innovation and collaboration within the tech space.


  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1319137980471836713)** (1 messages): 

> `Data Mapping Series, Scalable Graphics, Embeddings, Dimensionality Reduction, Unstructured Data` 


- **Final Installment of Data Mapping Series Released**: The **Nomic Team** announced the release of the final installment in the **Data Mapping Series**, focusing on **scalable graphics** for managing embeddings and unstructured data, which can be read [here](https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers).
   - This series details how machine learning concepts like **embeddings** and **dimensionality reduction** empower users to visualize massive datasets in their web browsers.
- **Six Part Data Mapping Exploration**: The latest post wraps up a six-part series aimed at elucidating the technologies behind the **Nomic Atlas platform** with respect to **unstructured data visualization**.
   - Readers are encouraged to check out the first parts of the series covering [Data Maps](./data-mapping), [embeddings](./embeddings-are-for-so-much-more-than-rag), and [dimensionality reduction](./see-your-data-with-dimensionality-reduction) for foundational knowledge.



**Link mentioned**: <a href="https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers">Data Maps, Part 4: Why Are Web Browsers The Best Data Browsers?</a>: Why Are Web Browsers The Best Data Browsers?

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1319055038013771806)** (17 messages🔥): 

> `Nomic BERT issue, Code Interpreter Pull Request, Loading System Messages, GGUF File Issues, Device Requirements` 


- **Nomic BERT Embedding Model Issue**: Users reported errors while loading **Nomic's embedding model** from Huggingface due to a recent commit that broke functionality, specifically this [commit](https://huggingface.co/nomic-ai/nomic-bert-2048/commit/ba22e9d89df6236d83c3daa26cc8dd78a130c3f2). Fortunately, the issue has now been fixed.
- **Pull Request for Code Interpreter Tool**: A pull request titled [Code interpreter by manyoso](https://github.com/nomic-ai/gpt4all/pull/3173) is in progress, aimed at adding a code interpreter tool based on the jinja template. Members expressed interest in following its developments.
- **Loading System Messages Discussion**: A user inquired about the possibility of a `load` button for loading system messages from text files, expressing frustration with copy-pasting. There seems to be a demand for this feature due to the many context-setting text files users have.
- **GGUF File Compatibility Issues**: Discussions arose about various **.GGUF** files with broken chat templates, with mentions of files like **Llama-3.3-70B-Instruct-Q4_K_M** and **Qwen2-72B-Instruct.Q4_K_M**. Fixes for these files were promised in the next release.
- **Device Requirements for GPT4ALL**: A user requested information regarding the device requirements for **GPT4ALL**, prompting another member to share a link to the official [system requirements](https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/system_requirements.md). This document outlines the necessary specifications for running GPT4All.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/system_requirements.md">gpt4all/gpt4all-chat/system_requirements.md at main · nomic-ai/gpt4all</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://github.com/nomic-ai/gpt4all/pull/3173">Code interpreter by manyoso · Pull Request #3173 · nomic-ai/gpt4all</a>: This is a WIP for code interpreter tool call based upon the jinja pr.Here is the latest jinja template I&amp;#39;m using for Qwen2.5-Coder-7B:{{- &amp;#39;&amp;lt;|im_start|&amp;gt;system\n&amp;#39; }...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1319057454881050714)** (16 messages🔥): 

> `TinyChat Installation Issues, Tiktoken Replacement Discussions, Scroll Direction Bug Report, Bounty Project Engagement, Layout Notation Insights` 


- **TinyChat installation faces hurdles**: After trying to set up TinyChat, a user reported issues with missing dependencies like **tiktoken**, and experienced a system freeze for **~30 seconds** during installation.
   - They also noted a strange prompt about *finding devices on the local network*, questioning its necessity.
- **Tiktoken needs a tailored replacement**: George Hotz acknowledged the need for a **replacement for tiktoken** and raised the question of whether it can be written directly within TinyGrad.
   - His focus on **8GB of RAM** as a limitation was a key point in the discussion.
- **Scroll direction switches unexpectedly**: A user reported an odd issue where the **scroll direction** on their Mac was reversed after running TinyChat, returning to normal after terminating the application.
   - George Hotz expressed surprise over this problem, confirming it's perplexing.
- **Bounty Project engagement strategies**: Chenyu mentioned that the goal of the bounties is to **advance the project** and emphasized engaging with contributors who add value through tests and improvements.
   - They pointed to contributions in the form of tests and optimization discussions as essential to driving progress.
- **Discussion on layout notation**: A user shared thoughts on layout notation being powerful yet **complex**, noting the effectiveness of the graphical representations in the documentation.
   - They highlighted that the **complement section** offers a unique perspective by describing all elements not selected, unlike traditional masks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1RRuhAW-I5u_6ssbm1eLqVWSK_PYFoARIVBYwYZwmcM4/edit?usp=drivesdk">View Merges</a>: Relevant code: https://github.com/unknownusername504/MicroGrad/blob/main/micrograd/tensors/viewable_tensor.py  What is the goal of a view?  Moving memory around is an expensive operation. If you have ...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/8194))">Issues · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Issues · tinygrad/tinygrad
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

khaner2162: Hi
why does scheduler  `# realize before expand or unsafe pad ops`?
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1319302126484062301)** (6 messages): 

> `Introduction of Ikuo618, Reminder about channel etiquette` 


- **Ikuo618 introduces himself**: Ikuo618 shared his background as a senior AI developer with over **6 years of experience** in building and deploying AI models in **DP**, **NLP**, and **CV**.
   - He highlighted his expertise in **Python**, as well as his proficiency with **TensorFlow** and **PyTorch** to develop intelligent systems.
- **Etiquette reminder on reposting messages**: A reminder was issued to a user not to repost their messages in multiple channels to maintain chat organization.
   - This serves as a gentle nudge for everyone to respect channel etiquette while participating in discussions.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1319356136650178612)** (2 messages): 

> `Platform Availability` 


- **Confirmation on Platform Status**: A member inquired if a specific feature is available on the platform, to which another member confirmed, saying it is **not on the platform yet**.
   - The inquiring member expressed gratitude for the confirmation with a smiley face.
- **User Interaction in Confirmation**: The interaction showcased a friendly exchange, with one user confirming a feature's absence on the platform.
   - The response highlighted positive engagement between users, as one expressed appreciation for the confirmation.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1319117786731118674)** (3 messages): 

> `Cohere API pricing, API keys types, Rate limits for endpoints` 


- **Cohere API offers free and paid keys**: Cohere provides two types of API keys: **evaluation keys** which are free but come with limited usage and **production keys** which are paid and offer much less limitation.
   - Users can create these keys on [the API keys page](https://dashboard.cohere.com/api-keys) and explore pricing details in the [pricing docs](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work).
- **Detailed rate limits for Cohere API**: The Cohere API has specific rate limits for each endpoint, with trial limits significantly lower than production ones; for example, the **Chat** endpoint is limited to **20 calls per minute** for trial users and **500 per minute** for production users.
   - Other endpoints like **Embed** and **Classify** also have distinct limits, stacking up to **1,000 calls per month** for all endpoints.



**Link mentioned**: <a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: This page describes Cohere API rate limits for production and evaluation keys.

  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

ikuo618: hi..................!
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

benny0917: Good looking product <@799853279017173033> congrats!
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1319068909693042708)** (6 messages): 

> `Torchtune Phi 4 Support, New Contributor Role, Implementation Differences Between Phi 3 and Phi 4` 


- **Torchtune currently lacks Phi 4 support**: A member inquired about using **Torchtune** for **Phi 4**, to which it was confirmed that support currently lies only with **Phi 3** and contributions for Phi 4 are welcomed.
   - Members expressed interest in potentially contributing to enable **Phi 4** support.
- **New Contributor Role Introduced**: A new **Contributor** role was launched on Discord to recognize community members who enhance **Torchtune** for everyone.
   - This initiative aims to acknowledge contributions and bridge the gap between **GitHub** and **Discord** usernames.
- **Minimal Differences Anticipated for Phi 4**: A discussion arose concerning the implementation differences between **Phi 3** and **Phi 4**, with one member noting they appear to be very minimal.
   - An image was shared that seemingly supports this statement, sparking further curiosity about the changes.



**Link mentioned**: <a href="https://pytorch.org/torchtune/stable/api_ref_models.html">torchtune.models &mdash; torchtune 0.4 documentation</a>: no description found

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1319088822167801936)** (2 messages): 

> `Asynchronous RLHF, Post-Training Techniques, Model Safety and Robustness` 


- **Asynchronous Approach in RLHF**: The traditional RLHF method is **computationally inefficient**; however, separating generation and learning enables faster, asynchronous training of models, as suggested in the research on [online but off-policy RLHF](https://arxiv.org/abs/2410.18252).
   - This research highlights a key question: *how much off-policyness can we tolerate* while ensuring efficient learning without sacrificing performance.
- **Importance of Post-Training for Models**: Models require **post-training** after the pre-training stage to ensure they are safe and follow human instructions effectively, as discussed in the [Allen AI blog](https://allenai.org/blog/tulu-3).
   - The process involves instruction fine-tuning and learning from human feedback to avoid eroding essential capabilities during specialization.
- **Challenges of Instruction Tuning**: Post-training methods, initially inspired by **InstructGPT**, can lead to a decline in certain model capabilities as more specialized skills are taught.
   - Finding the balance between enhancing capabilities like **coding** while retaining skills for **poetry and instruction following** remains a complex challenge.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.18252">Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models</a>: The dominant paradigm for RLHF is online and on-policy RL: synchronously generating from the large language model (LLM) policy, labelling with a reward model, and learning using feedback on the LLM&#3...</li><li><a href="https://allenai.org/blog/tulu-3">Tülu 3 opens language model post-training up to more tasks and more people  | Ai2</a>: Tülu 3 is a leading instruction following model family, offering fully open-source data, code, and recipes designed to serve as a comprehensive guide for modern post-training techniques.
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1319370418674597938)** (1 messages): 

> `Hackathon submission deadline, LLM Agents Hackathon, Final reminders, Project submissions, Last-minute questions` 


- **Final Call for Hackathon Submissions!**: A reminder has been issued that the **submission deadline** for the hackathon is tonight at **11:59 PM PST** (12/19).
   - *Make sure your projects are submitted* using the [LLM Agents Hackathon Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform)!
- **Support for Last-Minute Questions**: Participants are encouraged to drop any **last-minute questions** in the channel for assistance.
   - The community is rallying support to ensure everyone can *finish strong* before the deadline.


  

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
