---
id: 9e88f842-ed12-4ead-aa52-fe4d029a6454
title: 'Halfmoon is Reve Image: a new SOTA Image Model from ex-Adobe/Stability trio'
date: '2025-03-25T01:43:04.934624Z'
original_slug: ainews-halfmoon-is-reve-image-a-new-sota-image
description: >-
  **Reve**, a new composite AI model from former Adobe and Stability alums
  **Christian Cantrell**, **Taesung Park**, and **Michaël Gharbi**, has emerged
  as the top-rated image generation model, surpassing previous state-of-the-art
  models like Recraft and Ideogram in text rendering and typography. The team
  emphasizes *"enhancing visual generative models with logic"* and
  *"understanding user intent with advanced language capabilities"* to
  iteratively amend visuals based on natural language input. Additionally,
  **DeepSeek-V3-0324** and **Alibaba's Qwen2.5-VL-32B-Instruct** models were
  released with notable performance improvements, including better vision task
  benchmarks and mathematical reasoning.
companies:
  - artificial-analysis
  - stability-ai
  - adobe
  - deepseek
  - alibaba
models:
  - deepseek-v3-0324
  - qwen-2.5-vl-32b-instruct
  - recraft
topics:
  - text-to-image
  - prompt-understanding
  - model-composition
  - visual-generation
  - language-understanding
  - model-performance
  - complex-prompting
  - iterative-generation
people:
  - christian-cantrell
  - taesung-park
  - michael-gharbi
---


<!-- buttondown-editor-mode: plaintext -->**Composite AI is all you need?**

> AI News for 3/21/2025-3/24/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**227** channels, and **10464** messages) for you. Estimated reading time saved (at 200wpm): **1129 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

A couple of nice updates from [Qwen](https://news.ycombinator.com/item?id=43464068) and [Deepseek](https://twitter.com/_akhaliq/status/1904154585242935516) today, but we give title spot to a lesser known but ambitious new entrant.

Reve, [pronounced [ʀɛv], from “rêve”](https://x.com/m_gharbi/status/1904213903384695280), has [emerged from Artificial Analysis' leaderboard](https://x.com/ArtificialAnlys/status/1904188980423467472) as the top rated imagegen model, displacing former SOTA Recraft. "The model stands out for its impressive text rendering, prompt adherence, and aesthetics." We found it remarkably easy to play with.

![image.png](https://assets.buttondown.email/images/eacb4da0-b781-47d9-b5a2-a0b230b883b5.png?w=960&fit=max)

![image.png](https://assets.buttondown.email/images/43ed5116-3b08-4b8a-abcb-f74fa99263f9.png?w=960&fit=max)

And it beats Ideogram for typography:

![image.png](https://assets.buttondown.email/images/149cc977-e438-444a-b92a-098efb750d70.png?w=960&fit=max)

It's interesting that it comes from [Christian Cantrell](https://x.com/cantrell/status/1904213242567917684), former VP Product at Stability, [Taesung Park](https://x.com/Taesung/status/1904220824435032528), and [Michaël Gharbi](https://x.com/m_gharbi/status/1904213903384695280). All are Adobe alums, and Michael's announcement gives the most insight into how they do it:

> Reve’s mission is to invent the future of intent-driven visual creation. Capturing creative intent requires advanced machine understanding of natural language and other interactions. **Turning this intent into compelling visual calls for interactive systems** that have a deep understanding of the visual world they generate, so they can **iteratively amend it**.

[Taesung agrees](https://x.com/Taesung/status/1904220827073257483):

> Today's text-to-image models are essentially that—random slice-of-the-world generator. There's no intelligence. This is both a data and representation problem. **We need to leverage the equivalent of full documents for images, but we don't have a good representation for it.** Our mission at Reve is to **enhance visual generative models with logic**. As the first step, we focus on understanding user intent with advanced language capabilities, resulting in superior complex prompt understanding and text writing.

There's no suggestion that it's a single model, but rather some composite of models. Probably this is what Christian wanted to build at Stability, but couldn't.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

Here's a summary of the AI-related discussions from the provided tweets, categorized for a technical audience:

**Model Releases and Updates, Including Performance**

- **DeepSeek V3-0324 Release and Performance**: [@_akhaliq](https://twitter.com/_akhaliq/status/1904154585242935516) announced **DeepSeek-V3-0324** release on Hugging Face, and  [@Teknium1](https://twitter.com/Teknium1/status/1904147049219494148) also noted its release, and  [@reach_vb](https://twitter.com/reach_vb/status/1904153415665517034) highlighted it as a **post-training update** with potential for improved downstream performance. Several users discussed its performance and characteristics, including [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904161508642168971) who found it **comparable to Sonnet 3.6** and [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904292164672115077) noting it **surpasses DeepSeek-R1 and Claude-3.7** in some evaluations.
- **Qwen 2.5-VL-32B-Instruct Release**:  [@_akhaliq](https://twitter.com/_akhaliq/status/1904242971043607002) announced the release of **Alibaba's Qwen2.5-VL-32B-Instruct** on Hugging Face, and [@reach_vb](https://twitter.com/reach_vb/status/1904234593576014312) shared **performance benchmarks** indicating it beats Qwen 2.5 72B and GPT 4o Mini on vision tasks, with enhanced mathematical reasoning and human preference alignment.
- **DeepSeek Model Serving**: [@_akhaliq](https://twitter.com/_akhaliq/status/1904231386430799938) noted that **DeepSeek's new model is served on Hugging Face via Hyperbolic Labs**, and  [@ClementDelangue](https://twitter.com/ClementDelangue/status/1904237660237115542) mentioned it's available via FireworksAI and Hyperbolic Labs. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1904223627509465116)  stated that **Hyperbolic Labs now serves DeepSeek-V3-0324**.
- **DeepSeek V3-0324 on MLX**: [@reach_vb](https://twitter.com/reach_vb/status/1904204090868900140) reported that the latest **DeepSeek V3-0324 runs at >20 toks/sec on a 512GB M3 Ultra with mlx-lm**, and [@awnihannun](https://twitter.com/awnihannun/status/1904177084609827054) confirmed the same.
- **NVIDIA Mamba Image Backbones**: [@mervenoyann](https://twitter.com/mervenoyann/status/1904168637612630279) announced **NVIDIA's release of new Mamba image backbones** on Hugging Face, available in various sizes and resolutions.

**Frameworks and Tools**

- **LangChain and LangGraph Use Cases**: Multiple tweets highlighted use cases of LangChain and LangGraph, including Vodafone's AI assistants for data operations  [@hwchase17](https://twitter.com/hwchase17/status/1904216034095333392), Klarna's AI assistant for customer support [@LangChainAI](https://twitter.com/LangChainAI/status/1904219446874604018), and a medical supply chain AI system [@LangChainAI](https://twitter.com/LangChainAI/status/1904201544305725749).  [@hwchase17](https://twitter.com/hwchase17/status/1904247784087388252) also mentioned context management in langgraph.
- **Weave-Agent Planner Discussion**: [@jd_pressman](https://twitter.com/jd_pressman/status/1904139443189252252) discussed the **design and planning of Weave-Agent**, considering approaches like ReActTree and MuZero for agentic planning.
- **Smolagents Growth**: [@AymericRoucher](https://twitter.com/AymericRoucher/status/1904219464263946480) announced that **smolagents has reached 15k GitHub stars** and is integrating sandboxed code execution via E2B or Docker.
- **Together Chat**:  [@togethercompute](https://twitter.com/togethercompute/status/1904204860217500123) introduced **Together Chat**, featuring OSS models like DeepSeek R1 for web search, coding, image generation, and image analysis, and  [@togethercompute](https://twitter.com/togethercompute/status/1904204864885755905) listed the tech stack.

**Agent Engineering and Applications**

- **Agent Engineering Talk and Essay**: [@swyx](https://twitter.com/swyx/status/1904256213661192405) shared a **talk and essay on Agent Engineering**, defining agents, outlining six elements, and discussing their potential impact.
- **Linear and Codegen Integration**: [@mathemagic1an](https://twitter.com/mathemagic1an/status/1904293319297179871) announced **Codegen's integration with Linear**, enabling agents to solve tickets and close duplicates, and highlighted Linear's expanded capabilities for bots [@mathemagic1an](https://twitter.com/mathemagic1an/status/1904293320840655249).
- **Evaluation Metric for Agents**: [@_philschmid](https://twitter.com/_philschmid/status/1904147086011940942) advocated for using **pass^k instead of pass@k for evaluating agents**, arguing it provides a more accurate performance metric aligned with user experience.

**Economic and Strategic Implications**

- **AI Automation and Economic Growth Model**: [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1904180712393036095) discussed **GATE, a model for AI automation's economic impacts**, predicting trillions in AI investments, extreme compute scaling, and significant economic growth.
- **US-Japan Defense Innovation Award**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1904156111621754905) announced that **Sakana AI won an award** at the US-Japan Competition for Defense Innovation for novel AI solutions.
- **Perspectives on China and AGI**: [@teortaxesTex](https://twitter.com/teortaxesTex/) shared multiple opinions on China's technological and strategic advantages, including its state capacity, industrial base, and AGI efforts.  [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904008640542937273) also touched on DeepSeek's "commoditize your complement" theory.

**ARC-AGI Benchmark**

- **ARC-AGI-2 Release and Competition**: [@fchollet](https://twitter.com/fchollet/status/1904265979192086882) announced the release of **ARC-AGI-2**, a benchmark designed to measure general fluid intelligence, and the ARC Prize 2025 competition with a \$700,000 grand prize [@fchollet](https://twitter.com/fchollet/status/1904266438959084003).  He noted that current top AI approaches score very low, requiring test-time adaptation, and discussed the evaluation methodology [@fchollet](https://twitter.com/fchollet/status/1904267900963475807).

**Humor and Memes**

- **Coding by Vibes**: [@gneubig](https://twitter.com/gneubig/status/1904186575732253008) shared a tweet about **prompting to improve vibe coding**, distinguishing between coding by vibes for personal projects versus agent behavior.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek V3-0324: Performance and Expectations vs R1**

- **[Deepseek releases new V3 checkpoint (V3-0324)](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)** ([Score: 638, Comments: 125](https://reddit.com/r/LocalLLaMA/comments/1jip611/deepseek_releases_new_v3_checkpoint_v30324/)): **DeepSeek** released its new **V3 checkpoint (V3-0324)**, which likely includes updates and improvements over previous versions. Further details on specific features or enhancements are not provided in the post.
  - Discussion on the **DeepSeek-V3 checkpoint (V3-0324)** includes speculation about its use as a base for a future **R2 release**, with some users anticipating it to arrive in **April**. There is a debate on whether **V4** is necessary for R2, with arguments suggesting that improvements can be achieved through better scaling and reasoning techniques without a new base model.
  - Users are seeking **benchmark results** to compare the new model's performance, with some noting that no official benchmarks have been released yet. Independent tests are expected soon due to the open-source release of the weights, and there is a call for DeepSeek to release their own benchmarks similar to **Mistral**.
  - There are observations about the model's **coding skills improvement** and its deployment on both API and web platforms, with some users noting a more **censored version** compared to the original. The **MTP module** is highlighted for its role in enhancing decoding speed, achieving **1.8 times TPS**, as detailed in a [research paper](https://arxiv.org/pdf/2412.19437).


- **[New deepseek v3 vs R1 (first is v3)](https://i.redd.it/cvnu636y1nqe1.png)** ([Score: 282, Comments: 56](https://reddit.com/r/LocalLLaMA/comments/1jiqi81/new_deepseek_v3_vs_r1_first_is_v3/)): The image compares two versions of **DeepSeek** user interfaces: **V3** and **R1**. **V3** showcases a more dynamic design with animated weather cards for "Windy," "Rainy," "Sunny," and "Snowy," while **R1** offers a simpler interface with toggle buttons for "Wind," "Rain," "Sun," and "Snow," each represented by a single icon.
  - **DeepSeek V3** and **R1** interfaces are being compared, with **V3** offering animated weather cards and **R1** featuring simpler toggle buttons. Users are curious about which model corresponds to each interface and the prompts used for the comparison.
  - There is a preference for **open-source models** over proprietary ones due to cost and flexibility, despite **DeepSeek models** not being the cheapest. **Sonnet** is noted to be significantly more expensive than **V3**, especially during off-peak hours.
  - The discussion includes references to **command-a** running locally, with links provided for further exploration, such as the [Hugging Face model](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025) and a [GIF](https://i.redd.it/sl2dyqigfnqe1.gif) showcasing the interface. Users express interest in more dynamic content, like videos, to better understand the animated features.


- **DeepSeek V3-0324 has caught up to Sonnet 3.7 in my code creativity benchmark - "Write a raytracer that renders an interesting scene with many colourful lightsources in python."** ([Score: 215, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1jisuq4/deepseek_v30324_has_caught_up_to_sonnet_37_in_my/)): **DeepSeek V3-0324** has matched **Sonnet 3.7** in a code creativity benchmark involving a raytracer task in Python, demonstrating significant improvement over its previous version. The benchmark revealed that while most LLMs generated simple RGB scenes, Sonnet 3.7 and now DeepSeek V3-0324 produced more complex and aesthetically pleasing scenes, though the method for this creativity boost remains speculative. More details and data are available in the [GitHub repository](https://github.com/cpldcpu/llmbenchmark/blob/master/raytracer/Readme.md).
  - **DeepSeek V3-0324** is noted for its "psychotic taste," resembling reasoning models like **R1** or **QwQ** more than its predecessor, and has faced criticism for its creative writing outputs, which some users find incoherent despite high benchmark scores. **Gemma 3** is highlighted for its coherence and creativity in fiction, contrasting with **R1**'s often criticized outputs.
  - **R1** failed in the benchmark by not producing a functioning program, despite attempts, which raises questions about its effectiveness compared to older versions of **DeepSeek V3**. The discussion suggests that **R1**'s long chains of thought (CoT) do not guarantee successful outputs, unlike previous versions of **DeepSeek**.
  - The increase in program size for **DeepSeek V3-0324** and **Sonnet 3.7** is noted, with speculation about whether this is due to training for longer generation lengths or other optimizations. Generating 10kB of code in a single attempt is considered significant, indicating potential advancements in model capabilities.


**Theme 2. Meta's ParetoQ Explored: Promise of 2-bit Models**

- **[Meta released a paper last month that seems to have gone under the radar. ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization. This is a better solution than BitNet and means if Meta wanted (for 10% extra compute) they could give us extremely performant 2-bit models.](https://arxiv.org/pdf/2502.02631)** ([Score: 505, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1jig5re/meta_released_a_paper_last_month_that_seems_to/)): **Meta's ParetoQ** paper introduces **scaling laws for extremely low-bit LLM quantization**, proposing a more effective solution than **BitNet**. This allows the possibility of delivering highly efficient **2-bit models** with only a **10% increase in compute requirements**.
  - **Quantization and Performance:** Discussions emphasize the potential of **2-bit quantization** for lightweight models, with some users noting that this could be transformative for applications like creative writing assistants and chatbots. However, concerns about potential slowdowns and the impact of quantization on model intelligence and instruction following are raised, with hopes for improvements using **vulkan/T-MAC kernels**.
  - **Research and Comparisons:** Users discuss the **ParetoQ framework** as a more rigorous method for comparing quantization settings, highlighting a learning transition between 2 and 3 bits. The paper is noted for its ability to optimize training for 2-3 bit models, with comparisons to **AQLM** and references to human synapses having **4-5 bpw**.
  - **Resources and References:** The discussion includes references to resources like the [**Intel auto-round**](https://github.com/intel/auto-round) project and **DeepSeek-R1-int2-mixed-sym-inc**, which achieve comparable performance with 97.9% accuracy retention. A link to the paper is provided: [arxiv.org](https://arxiv.org/pdf/2502.02631).


**Theme 3. Expanding LLM Functionalities: From Text to Multimodal**

- **[I made a diagram and explanation of how transformers work](https://www.reddit.com/gallery/1jifvny)** ([Score: 272, Comments: 20](https://reddit.com/r/LocalLLaMA/comments/1jifvny/i_made_a_diagram_and_explanation_of_how/)): **LLM functionalities** are expanding beyond text, and a user has created a **diagram and explanation** to illustrate how **transformers** function. This effort aims to provide a clearer understanding of the internal mechanisms of transformers for those interested in AI and machine learning.
  - **Input and Output Embeddings**: There is a discussion on whether input and output embeddings are still linked in modern **transformer architectures**, with users noting the difficulty in obtaining a comprehensive and current overview of these architectures.
  - **Resources and Diagrams**: Several users shared resources to aid in understanding transformers, including a detailed explanation by **Cromulent123** and a link to a GitHub page with relevant diagrams ([GitHub Llama Nuts and Bolts](https://github.com/adalkiran/llama-nuts-and-bolts/blob/main/docs/20-DIAGRAMS.md)). Another user highlighted a conceptual guide on **transformers** available on [Ben Levinstein's Substack](https://benlevinstein.substack.com/p/a-conceptual-guide-to-transformers).
  - **Detailed Explanation on Transformer Functionality**: **Cromulent123** provides an in-depth explanation of how transformers work, focusing on the process of token embedding, the role of **Query, Key, and Value Matrices**, and the concept of **attention scores** in determining relevance. They also discuss the importance of **contextual enrichment** through multiple transformer blocks, emphasizing the nuanced understanding of token relationships.


- **I don't understand what an LLM exactly is anymore** ([Score: 233, Comments: 89](https://reddit.com/r/LocalLLaMA/comments/1jijyx2/i_dont_understand_what_an_llm_exactly_is_anymore/)): The author is confused about the expanding definition of **Large Language Models (LLMs)**, originally understood as systems predicting the next word based on pretrained weights from text data. They question how LLMs now encompass capabilities like audio and image generation, and cite **[SpatialLM](https://manycore-research.github.io/SpatialLM/)**, which processes 3D point cloud data, as an example of this broadening scope, seeking clarification on the connection to language models.
  - **Diffusion Models and LLMs**: There is a debate on whether models like **Stable Diffusion** qualify as **LLMs** since they incorporate **T5** for understanding text prompts, though they primarily generate images. **Co0k1eGal3xy** argues that such models are close to LLMs because of their advanced language understanding, despite not traditionally fitting the LLM category.
  - **Tokenization and Multimodal Models**: **suprjami** explains that all data, including text, images, and audio, is tokenized into numbers for LLMs to process, which allows them to learn relationships between different media types. **Chair-Short** details how **self-attention mechanisms** and **positional encoding** enable LLMs to handle different data modalities, suggesting a shift from purely text-focused models to multimodal capabilities.
  - **Defining LLMs**: Discussions highlight the blurred lines in defining LLMs, with some viewing them as large models capable of processing and generating language, regardless of the input type. **SnackerSnick** mentions that LLMs use tokenization and embeddings to predict subsequent tokens, while **Otherwise_Marzipan11** and **Co0k1eGal3xy** suggest that branding and interaction with language, whether text, audio, or images, contribute to the LLM label.


- **Possible Llama 4 prototypes on Chatbot Arena** ([Score: 105, Comments: 21](https://reddit.com/r/LocalLLaMA/comments/1jiewjn/possible_llama_4_prototypes_on_chatbot_arena/)): **MetaAI** is testing several anonymous **Llama/Meta models** on [Chatbot Arena](https://lmarena.ai/), potentially as prototypes for **Llama 4**. Models like **aurora**, **ertiga**, **pinnacle**, **solaris**, and **spectra** are image-enabled, while **rhea** is identified as **Llama 3**.
  - Discussions reveal skepticism about model identities on **Chatbot Arena**, as some models, like **anonymous-chatbot**, claim to be from **OpenAI**, while others like **rage** and **phantom** are suspected to be **Meta** models. Users note that these models often provide inconsistent company affiliations, potentially due to a guard model or hallucinations.
  - The **anonymous-chatbot** and **nebula** models are highlighted for their performance, with **nebula** being particularly praised for excelling in tests, while models like **rage** and **rhea** received mixed feedback, with **rhea** noted for its friendly demeanor and emoji use.
  - There is a debate about whether any models are actually **Llama 4**, with users noting that none explicitly identify as such. Some comments suggest that **Meta** might be testing diverse writing styles or using randomized system prompts to obscure the true origin of the models.


**Theme 4. TeapotLLM's Impact: Lightweight Q&A Models**

- **[Announcing TeapotLLM- an open-source ~800M model for hallucination-resistant Q&A and document extraction, running entirely on CPU.](https://huggingface.co/teapotai/teapotllm#evaluation)** ([Score: 163, Comments: 50](https://reddit.com/r/LocalLLaMA/comments/1jioxj4/announcing_teapotllm_an_opensource_800m_model_for/)): **TeapotLLM** is an open-source model designed for hallucination-resistant Q&A and document extraction, featuring an approximate **800 million parameter** architecture. It is optimized to run entirely on **CPU**, making it accessible for broader usage without the need for specialized hardware.
  - **TeapotLLM's Hallucination Resistance**: Discussion highlights the model's focus on hallucination resistance and its performance against models like **Qwen** and **Llama**, with some skepticism expressed about claims of reduced hallucination. Users are curious about its placement on hallucination leaderboards, and a [demo](https://teapotai-teapotchat.hf.space/) is available for testing.
  - **Model's Language and Output Capabilities**: The model is trained primarily in English, but theoretically supports all languages covered by **flan-t5**. It can extract structured data into **JSON** using a library that parses fields into typed JSON, as detailed in the [documentation](https://teapotai.com/docs#3-information-extraction), though there is interest in expanding language support and testing on platforms like **ollama**.
  - **Performance and Resource Usage**: **TeapotLLM** is optimized for CPU usage, fitting within approximately **2GB of RAM** on Google Colab, making it accessible for users with limited compute resources. There is interest in exploring fine-tuning on more modern models like **Qwen 0.5B** to potentially enhance performance, while maintaining the current model's strengths in document extraction and concise responses.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. New Improved Memory Alpha in ChatGPT Enhances Interaction**

- **New improved memory alpha is insane** ([Score: 414, Comments: 241](https://reddit.com/r/ChatGPT/comments/1jidd6w/new_improved_memory_alpha_is_insane/)): The post discusses the **new improved memory alpha** feature in **ChatGPT**, comparing its impact to the leap from **GPT-2 to GPT-4**. The author expresses skepticism about **DeepSeek's** ability to compete unless they adopt similar advancements, expressing confidence in **OpenAI**'s continued leadership.
  - Many users express frustration and confusion over the **availability and inconsistency** of the new **memory alpha feature** in **ChatGPT**, with some losing access unexpectedly despite having **pro subscriptions**. **CyberNoche** and **jalpseon** highlight deactivation issues, while **alpha_rover** and **DamionPrime** share positive experiences with memory persistence.
  - The discussion touches on the **pricing of ChatGPT** subscriptions, with **Initial-Kangaroo-534** questioning the value of paying **$200 per month**. This is contrasted by **alpha_rover**, who finds the feature invaluable for project continuity and would miss it compared to other AI tools.
  - Some commenters like **3xNEI** and **SillyTwo3470** speculate on the broader implications of memory features, suggesting it could lead to **human-AI hybridization**. They emphasize the potential for **increased personalization** and the blurring of lines between tool and partner, indicating a significant shift in how users might interact with AI.


**Theme 2. Anthropic's Revenue Surge Matches OpenAI's 2023 Numbers**

- **[Anthropic is making about $115M a month now; same as OpenAI in Nov 2023](https://i.redd.it/klikk2sppkqe1.png)** ([Score: 272, Comments: 50](https://reddit.com/r/ClaudeAI/comments/1jijnw9/anthropic_is_making_about_115m_a_month_now_same/)): Anthropic is reportedly generating **$115M per month**, matching **OpenAI's revenue in November 2023**. Revenue projections for 2025 estimate **$2B** as likely and **$4B** as optimistic, with Manus contributing approximately **$2 per task** to their revenue. An image depicts a **40% increase** in annualized revenue from December 2024 to March 2025, with figures from the **Bay Area Times**.
  - **Claude's Impact and Usage**: Users highlight **Claude Code** as a game-changing tool, with some spending **$50 per day** on it due to its effectiveness in automating coding tasks. Alternatives like **AIDER** and **Cursor's Agent** are mentioned but are deemed less effective compared to Claude, which is described as being akin to having a competent intern.
  - **Revenue Sources and Context**: A significant portion of **Anthropic's revenue** is attributed to integration with **AWS Bedrock**, with expectations of continued growth due to widespread enterprise adoption. The discussion clarifies that the reported figures represent revenue, not profit.
  - **Model Comparisons and Preferences**: Users compare various AI models, noting that Claude offers superior performance despite smaller context windows in some cases. The **OG 600b model** and **Sonnet 3.7** are mentioned, with the latter praised for its smart capabilities and iterative problem-solving.


**Theme 3. AI-Driven Bug Fixing Automation: A 27-Day Experiment**

- **I made AI fix my bugs in production for 27 days straight - lessons learned** ([Score: 191, Comments: 80](https://reddit.com/r/ChatGPTCoding/comments/1jibmtc/i_made_ai_fix_my_bugs_in_production_for_27_days/)): Over 27 days, the author used **Claude 3.7** to automatically fix 21 unique production bugs, resulting in 12 successful one-shot fixes, 6 partial successes, and 3 failures due to incorrect assumptions or complex issues. Despite the initial time investment exceeding manual bug fixing, the system reduced cognitive load and context switching, though it may not suit niche or complex problem domains.
  - **Interest in Open Sourcing**: There is significant interest in the project being open-sourced, with **Relevant-Pitch-8450** expressing intent to share it after some cleanup. Users appreciate the UI design and see potential utility in the tool.
  - **Potential Commercialization**: Commenters like **ClassyBukake** suggest that the tool could be monetized as a service, highlighting its appeal from both personal and business perspectives.
  - **Cost and Time Efficiency**: **HelpRespawnedAsDee** raises questions about the tool's cost and time efficiency over an extended period, suggesting continued use to evaluate long-term benefits.


**Theme 4. Advanced Claude Workflow Integration: MCP External Tools**

- **My Claude Workflow Guide: Advanced Setup with MCP External Tools** ([Score: 124, Comments: 20](https://reddit.com/r/ClaudeAI/comments/1ji8ruv/my_claude_workflow_guide_advanced_setup_with_mcp/)): The post provides a detailed guide for setting up **Claude's desktop application** with external tools like **Brave Search** and **Tavily** to enhance its capabilities, requiring a **Claude Pro subscription** ($20/month) and specific software installations like **Node.js** and **Python**. It includes configuration examples for both **Windows** and **macOS**, instructions for accessing developer settings, and troubleshooting tips for installation and setup issues. The guide emphasizes the benefits of enhanced web search, filesystem access, and sequential thinking, and provides additional resources and security considerations for effective use.
  - **Claude's desktop application setup** is praised for its accessibility to non-developers, providing a bridge for regular desktop users to enhance Claude's capabilities without coding skills. The guide is compared to **Claude Code**, which offers more flexibility for tech-savvy users comfortable with command line interfaces.
  - A tutorial for **Claude Code** is recommended for those interested in exploring its capabilities, available on [YouTube](https://www.youtube.com/watch?v=oM2dXJnD80c). This highlights the distinction between the two approaches: one prioritizing ease of use and the other, advanced customization.


**Theme 5. Wan 2.1 Video Frame Feature Innovations in AI**

- **[Wan-i2v - Prompt: a man throws a lady overboard from the front of a cruiseship.](https://v.redd.it/0ftuy4jmljqe1)** ([Score: 812, Comments: 51](https://reddit.com/r/StableDiffusion/comments/1jifrb8/wani2v_prompt_a_man_throws_a_lady_overboard_from/)): **Wan-i2v AI** has introduced new features and advancements, as demonstrated in a prompt scenario where *"a man throws a lady overboard from the front of a cruiseship."* While the post does not provide further details, it suggests a focus on action-oriented scenarios or potentially controversial themes in AI-generated content.
  - The **Wan-i2v AI** is discussed as an **image-to-video** tool, with some users noting that it couldn't independently create a starting frame from the **Titanic** movie, implying a direct screenshot was used instead. This highlights the potential limitations of AI in generating entirely original content without reference images.
  - Users humorously critique the AI's understanding of **physics**, with comments suggesting that while AI may not currently grasp physical laws, advancements such as **Stable Diffusion** and **Wan2.1** are rapidly improving in simulating realistic physics in animations, such as "boob jiggles."
  - The conversation also touches on the idea of AI-generated **alternate movie endings**, with users joking about creating new endings for films like **Titanic**. This raises questions about **copyright issues** and the potential for new **YouTube channels** focused on AI-crafted content, despite the challenges of intellectual property rights.


- **[Wan 2.1 begin and ending frame feature having model coming officially](https://i.redd.it/ngxqlw2t8nqe1.png)** ([Score: 100, Comments: 13](https://reddit.com/r/StableDiffusion/comments/1jirb3r/wan_21_begin_and_ending_frame_feature_having/)): **Wan 2.1** is set to release an official model that supports **start and end frames interpolation** soon, as confirmed by user "danielzy1990" on a social media platform. For more details, refer to the [GitHub issue comment](https://github.com/Wan-Video/Wan2.1/issues/264#issuecomment-2747490626).
  - Users anticipate that **Wan 2.1**'s new model will significantly enhance video control, with some expressing hope for improvements such as adding a guidance layer similar to **Hunyuan** to speed up generation times.
  - Comparisons to **Hunyuan** highlight its efficiency, generating video clips at **24fps** in nearly half the time it takes **Wan** to generate at **16fps**, emphasizing the potential benefits of guidance training.
  - There is interest in the model's capability to support **multiple timed keyframes**, with some users hoping it remains compatible with existing **img2vid** functionalities.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1. DeepSeek V3's Surprise Launch Shakes AI Community**

- [**DeepSeek V3 Emerges as Open-Source Giant**](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324): **DeepSeek** released **DeepSeek V3**, a **685B-parameter** mixture-of-experts model under the **MIT license**, accessible on **Hugging Face**. The community is excited, comparing it to **OpenAI's o1** models in performance.
- [**DeepSeek V3 Outperforms R1?**](https://x.com/IterIntellectus/status/1904159903754621348): Users claim **DeepSeek V3** beats **R1** in coding and front-end tasks, even without chain-of-thought reasoning, noting its cost-effectiveness and excellence in math.
- [**DeepSeek V3 Drops Without a README!**](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324): **DeepSeek** releases **DeepSeek V3** without proper documentation, leaving users both amused and perplexed by the lack of a README, but offering a playground for experimentation.

**Theme 2. Qwen Models and Upcoming AI Innovations**

- [**Qwen3 Support Added to Hugging Face Transformers**](https://github.com/huggingface/transformers/pull/36878): Developers are thrilled as **Qwen3** support is integrated into **Hugging Face Transformers**, preparing for the upcoming **Qwen3** models.
- [**Qwen2.5-VL-32B-Instruct Released Under Apache 2.0**](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct): **Qwen** releases **Qwen2.5-VL-32B-Instruct**, a multimodal vision-language model fine-tuned with reinforcement learning, enhancing mathematical reasoning and visual problem-solving capabilities.
- [**Qwen3 to Support CPU Inference?**](https://github.com/huggingface/transformers/pull/36878): Users speculate that **Qwen3-15B-A2B** could be ideal for CPU inference due to its size, making advanced AI models more accessible.

**Theme 3. Debates and Advances in LLM Reasoning Training**

- [**R1-Zero Training Bias Unveiled**](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf): Researchers uncover a bias in **R1-Zero-like training**, where using row mean favors shorter correct responses and longer incorrect ones, impacting model outputs.
- [**GRPO's Length Explosion Troubles Practitioners**](https://x.com/WenhuChen/status/1903464313391624668): Users grapple with **GRPO** training leading to length explosion, debating techniques like length clipping and curriculum to address the issue.
- [**MathFusion Supercharges LLM Math Skills**](https://arxiv.org/abs/2503.16219): **MathFusion** enhances mathematical reasoning in LLMs via cross-problem instruction synthesis, improving models like **DeepSeekMath-7B**, **Mistral-7B**, and **Llama3-8B**.

**Theme 4. Agent Engineering and MCP Developments**

- [**AGNCY Initiative Propels Agentic Interaction Standards**](https://t.co/I558Qe2u4n): Luke leads **AGNCY**, aiming to create an open standard for agentic interactions, providing a robust framework for developing more effective AI agents.
- [**MCPwizard Eases MCP Server Creation**](https://www.npmjs.com/package/mcpwizard): Developers introduce **mcpwizard**, a CLI tool that simplifies creating and deploying **MCP servers**, enabling easy addition of custom tools to AI assistants like Claude.
- [**A16Z Explores Future of AI Tooling with MCP**](https://a16z.com/a-deep-dive-into-mcp-and-the-future-of-ai-tooling/): **A16Z** publishes a deep dive into **Model Context Protocol (MCP)**, analyzing its potential as a standard interface for AI models and discussing its impact on AI tooling.

**Theme 5. NVIDIA's Nemotron-H Models and Hardware Advances**

- [**NVIDIA Unveils Nemotron-H Hybrid Models**](https://research.nvidia.com/labs/adlr/nemotronh/): NVIDIA introduces the **Nemotron-H** family, hybrid **Mamba-Transformer** models offering up to **3x** speed improvements, with models ranging from **8B** to **47-56B** parameters.
- [**Mistral 24B Roars Back into Favor**](https://x.com/neurosp1ke/status/1903564534930907604): **Mistral 24B** is hailed as one of the greatest releases recently, with users impressed by its strength and accessibility under the **Apache 2.0** license.
- [**Flash Attention and Hopper Architecture Demystified**](https://developer.nvidia.com/ERR_NVGPUCTRPERM): Enthusiasts delve into **Flash Attention** optimizations and clarify confusion around **Hopper's 64B swizzle**, enhancing understanding of NVIDIA's GPU architectures.

---

# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar 3.7 Bug kicks model**: A user reported a bug with **Sonar 3.7** where a *chown* command kicks the model out and breaks the conversation while coding, wondering if there was any difference in performance between high and old source amount and reasoning quality between search steps.
   - A user followed up noting that in their experience, the difference is quite large, sharing a screenshot [here](https://cdn.discordapp.com/attachments/1047649527299055688/1353541240200761537/image.png?ex=67e2afc3&is=67e15e43&hm=03e4b82072a680a8a9d215442a099e9d4c3adf29d24c0690d38258cbfe15690e&).
- **Sonar Model Gives Cropped Snippets**: Multiple users reported that the **Sonar model** in the Perplexity API is truncating responses, particularly since the weekend, even though the JSON format is correct.
   - A user provided an example of a JSON request and the truncated response, noting that switching to **sonar-pro** resolves the issue, but is not preferrable for cost reasons.
- **Llama Index Wrestles with Sonar**: A user encountered an error when configuring **Sonar** as a chat engine with **Llama Index** for a **RAG project** and requested assistance.
   - This highlights potential integration challenges when using **Sonar** in conjunction with other AI development tools.
- **Deep Research Rate Limit**: A user inquired about the possibility of extending the limit of **100 deep researches per minute** due to bulk processing needs in their application.
   - This inquiry underscores the demand for higher API usage limits for users with demanding workloads.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Bonsai Bitnet Seeks Testers for Qwen2.5 Comparison**: A member is looking for testers for [deepgrove/Bonsai](https://huggingface.co/deepgrove/Bonsai/tree/main), asking how the **bitnet** compares to **Qwen2.5 0.5B**.
   - They also linked a [relevant Hugging Face Transformers PR](https://github.com/huggingface/transformers/pull/36878) about adding **Qwen3** and **Qwen3MoE** support.
- **Orpheus TTS Model Gains Audio Finetuning**: **Audio finetuning** has arrived with the **Orpheus TTS** model, according to a newly released [Unsloth notebook](https://github.com/unslothai/notebooks/pull/17/files).
   - A user noted that the work was all done by a particular member and that the notebook is a lot more streamlined compared to local audio tokenizing and then regular **Llama3** finetuning.
- **Straight PRs OK on Unsloth Github, but wait**: A member inquired about contributing to Unsloth's GitHub, and another member confirmed that **straight PRs are acceptable**, though potential delays may occur due to the high volume of recent PRs and issues.
   - The discussion then shifted to modifying data preparation steps in Colab to accommodate **.txt** files, aiming for cheaper inference, and the [original issue](https://github.com/unslothai/unsloth/issues/14) was linked.
- **GRPO Reasoning Needs Training Data**: A user asked about training only parts of the output, specifically wanting the model to generate its own reasoning during inference.
   - It was suggested to look at the [GRPO notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks) as a standard way of adding reasoning, and that the model must see reasoning traces during training to take it into account during inference.
- **Unsloth's Fine-Tuning Guide Now Available**: A member created a guide for [fine-tuning with Unsloth](https://youtu.be/Lt7KrFMcCis), covering theoretical aspects, practical examples, and how to create a reasoning model with **GRPO**.
   - The guide compiles everything learned over the last year.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nebula Steals Chatbot Spotlight**: Members found **Nebula**, an anonymous chatbot suspected to be from DeepMind, to be *really good* and *the best anonymoud model rn*, outperforming others in math, english-turkish translation, and solving Arc-AGI problems.
   - It seems similar to **Phantom**, which users identified as a Google model, with both being tested in the arena.
- **GPT-4o Gets Human Alignment Boost**: **GPT-4o** has significantly improved through OpenAI's post-training, potentially surpassing **Grok 3** soon, due to continued pretraining since December.
   - Speculation suggests it might top the leaderboard, leveraging OpenAI's proficiency in human preference alignment in the LM arena.
- **Specter Evolves into Phantom then Nebula**: **Specter**, **Phantom**, and **Nebula** are revisions of the same model, in that order, showing performance jumps in a few weeks.
   - Members noted a more significant performance jump from **Specter** to **Phantom** compared to **Phantom** to **Nebula**.
- **LMArena Fixes Bugs, Tunes Leaderboard**: The LMArena alpha received updates including bug fixes and new features, and testers are encouraged to continue testing at [alpha.lmarena.ai](https://alpha.lmarena.ai/) with the password `still-alpha`.
   - A bug preventing messages from saving and causing vote failures has been fixed, and leaderboard columns are now sortable with live data updates; feedback can be provided via [this Google Forms link](https://forms.gle/8cngRN1Jw4AmCHDn7) and bug reports can be filed [using this Airtable link](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's CMD+Backspace becomes problematic**: Users express frustration with **Cursor's CMD+Backspace** leading to accidental project deletions, with some losing work up to **7 times**.
   - The Cursor team plans to change the default keybinding to **CMD+Shift+Backspace**, with configuration options, targeting a Monday rollout.
- **Claude 3.7 MAX hits users' pocket**: **Claude 3.7 Thinking**, now **Claude 3.7 MAX**, moves from the Pro plan to usage-based pricing, causing user frustration due to increased costs.
   - **Claude 3.7 MAX** features a higher context window and more tool calls compared to the standard **Claude 3.7 Sonnet**.
- **Windsurf Surfing Ahead in Responsiveness**: Some users find **Windsurf** faster and more responsive than Cursor, citing Cursor's lagging and freezing.
   - Others prefer Cursor for its rollback features and agent performance, though acknowledge AI programming's remaining challenges.
- **MCP Combinations become hype**: Users experiment with various **MCP (Model Context Protocol)** server combinations to enhance AI coding agents like Cursor, with **Supabase MCP** highlighted.
   - Some users suggest MCPs may be overhyped, noting instances of agents over- or under-utilizing MCPs, suggesting a need for clearer instructions.
- **3D Integration Frustrates AI Coders**: A user struggles to integrate a 3D model (**FBX format**) into a three.js project using Claude, facing issues with **FBXLoader**.
   - The limitations of AI in handling 3D designs become clear, with suggestions to switch to **GLTF format** and simplify tasks.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek V3-0324 Beats R1?**: The Aider community is excited about the new [DeepSeek V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) release, suggesting it outperforms **R1** in coding and front-end tasks, despite lacking chain of thought.
   - Members highlight its strengths in coding and math compared to previous versions, drawing comparisons to **Sonnet 3.5** in benchmarks, while also noting its cost-effectiveness.
- **Aider Tames Sonnet's Over-Eagerness**: Paul Gauthier reveals he has managed to get Aider to mitigate **Sonnet 3.7's** over-eager behavior by adding a line to the prompt to chill out; this is now available in the main branch.
   - He encourages users to provide feedback on this adjustment based on their coding sessions.
- **Aider Gains New Homepage**: Paul Gauthier announces the launch of Aider's new homepage at [aider.chat](https://aider.chat), showcasing compatibility with models like **Claude 3.7 Sonnet**, **DeepSeek R1** & **Chat V3**, **OpenAI o1**, **o3-mini** & **GPT-4o**, and support for over 100 code languages.
   - This update offers an improved introduction for new users and a central hub for resources.
- **Aider's Context Command Streamlines Chats**: Paul Gauthier introduces an experimental `/context` command in Aider that automatically sets up the chat context, working best with **Sonnet 3.7**, **R1**, and **o3-mini**.
   - This new command enhances user experience by intelligently identifying and adding relevant files to the chat.
- **Community Curates LLM Contexts**: A member announces the launch of [ctxs.ai/weekly](https://ctxs.ai/weekly), a site dedicated to collecting **aider conventions**, **prompts**, and **LLM-oriented documentation snippets**.
   - The goal is to create a useful resource for the **aider community**, and the member is actively soliciting feedback on how to improve the site.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LCPP Context Length Baffles**: Users found that setting a context length to **100** in LCPP still tries to allocate **180GB** of RAM, leading to VRAM exhaustion.
   - Suggestions include Attention overriding the assigned context length, missing ROPE-specific arguments, or using **Q8** quantization.
- **Deepseek V3 Mirrors Sonnet 3.7**: **Deepseek V3 0324** shows as much variation as **Sonnet 3.7**, suggesting shared advancements in their architectures, viewable in [this image](https://cdn.discordapp.com/attachments/1154120232051408927/1353739123084627998/image.png?ex=67e2bf4e&is=67e16dce&hm=a779b06b1028e58affe0e8deb753caa78df67398ccb0c12f6de9f1360198b369).
   - One user even called it a *huge update with Sonnet*-level code creativity and a potential base for **R2**.
- **Transformers Ditch Normalization**: Inspired by the **Transformers without Normalization** paper, a member replaced normalization with **tanh**.
   - The discussion then focused on removing experts at inference and its effects on smaller weights.
- **MathFusion Supercharges LLM Math**: **MathFusion** improves mathematical reasoning in LLMs via cross-problem instruction synthesis, enhancing models like **DeepSeekMath-7B**, **Mistral-7B**, and **Llama3-8B** ([more on MathFusion](https://x.com/gm8xx8/status/1903021157214748701?s=46)).
   - This method creates the **MathFusionQA dataset**, which fine-tunes models and boosts benchmark accuracy with minimal extra data.
- **Qwen3 to support CPU inference**: The [transformers library PR#36878](https://github.com/huggingface/transformers/pull/36878) shows that **Qwen3** support is being added, meaning that the models will soon be supported by the transformers library.
   - A user speculated that **Qwen3-15B-A2B** could be a good candidate for CPU inference due to its size.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sam Altman Teases GPT-5 Release**: Despite the absence of an official announcement, **Sam Altman** confirmed that **GPT-5** will launch this year, leading to speculation it could arrive in the first half to compete with **R2** or **Llama-4**.
   - Members on the OpenAI Discord server suggested that an unannounced API might also be imminent.
- **GPT-4o: The Model That Converted a User**: A user finds **GPT-4o** to be such a strong daily driver that they rarely switch models, only using other models such as **4.5, o1, o3** when the **4o messages** run out or for important or unsolved problems.
   - The user also claims to have built an "engine" that recovered a **400+ turn chat** and continues past **500 turns** retaining context with no drift or hallucinations, all through the default prompt.
- **Many-Shot Prompting Boosts Multimodal Model Muscle**: A research paper ([MANY-SHOT IN-CONTEXT LEARNING IN MULTIMODAL FOUNDATION MODELS](https://arxiv.org/abs/2405.17015)) suggests that **closed models** like **GPT-4o** and **Gemini 1.5 Pro** benefit significantly from many-shot demonstrations up to ~2,000 examples, whereas open-weight models do not show the same benefit.
   - The paper notes that large multimodal foundation models like **GPT-4o** and **Gemini 1.5 Pro** show significant performance improvements when provided with many-shot demonstrations compared to few-shot examples.
- **Run an F1 Team Powered by GPT-4o**: The open source project **FormulaGPT** ([github repo](https://github.com/dawid-maj/FormulaGPT/)) simulates head-to-head races between LLM-powered teams that *think contextually and adaptively* by continuously reasoning, strategizing, and making nuanced decisions.
   - Viewers can challenge advanced language models in **Player vs. AI Mode**, or watch the best AI models battle each other in **AI vs. AI Mode** while observing detailed AI reasoning behind each pit stop, tire change, or overtaking maneuver.
- **Avoid Turnitin AI Detector, If You Dare**: A member sought advice on avoiding **Turnitin AI similarity detection** for a report reusing their company's business model, which violates Turnitin's ToS.
   - Others suggested it looked like spamming appeals to cheat homework and recommended using **humanize AI** tools.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI's o1-pro: Gucci-Level Pricing?**: Users reacted strongly to **OpenAI's o1-pro** API pricing at **$150/M input tokens** and **$600/M output tokens**, with one calling it *GucciAI* due to its high cost.
   - Another member joked that the API's slowness might be a deliberate feature to prevent overspending given compute constraints.
- **Image Generation MIA on OpenRouter**: A user inquired about using **Gemini's image generation** with the *gemini-2.0-flash-exp* model, but was informed that **image generation is not yet supported on OpenRouter**.
   - The team indicated that while image generation is on their roadmap, there are currently no short-term plans to support image models like **Flux**.
- **Lambda Endpoints Plagued by 404s**: Multiple users reported encountering **404 'no endpoint found' errors** when attempting to use **Lambda** models, despite **Lambda's status page** showing full operational status.
   - The community offered suggestions, and some users confirmed that the **Llama 3.3 70B Instruct | Lambda** model was functioning correctly for them.
- **DeepSeek R1 challenges OpenAI o1**: Members noted that the **DeepSeek R1** model, a **671B parameter model** with **37B active** during inference, performs comparably to **OpenAI's o1** but is open-sourced and available under the **MIT license**.
   - Its availability under the MIT license allows for commercial use.
- **Claude 3.7 Sonnet Sputters with Overload Errors**: Users reported frequent **overload errors** when using **Claude 3.7 Sonnet**, leading to cut-off responses and charges for input tokens.
   - One user suggested a retry strategy or switching to **Gemini 2.0 Pro** as an alternative, acknowledging Claude's strength in translations.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Lacks NPU Support**: Users have reported that **NPUs** are not yet supported in **LM Studio**, but **Ryzen AI** support exists in version **0.3.11**.
   - For those with limited resources like **2GB VRAM**, consider using **Gemma 3 1B** with **Q6** or **Q8 quantization** and the **CUDA** runtime for improved performance.
- **KV Cache Quants Slash VRAM Needs**: Users recommend leveraging **KV cache 8-bit quants** to diminish memory footprint when operating models with extensive context windows, like 30k tokens.
   - Keep in mind that **12GB of VRAM** might prove inadequate for a **32B model**, suggesting that **Phi-4** or **Qwen2.5 14b** could serve as compelling alternatives.
- **Multi GPU Gets In-App Management**: Enthusiasts are raving about **LM Studio** controls that allow the user to select the **GPU** that the model will load onto, available in the latest beta build.
   - Multiple users confirmed that **Multi GPU is supported out of the box** with the latest beta build of **LM Studio**.
- **Google Coral TPUs a Flop for AI**: The **Google Coral dual TPU** is inadequate for **AI** use as it does not have any onboard memory to store data.
   - One user with an **8060s** also inquired about thermal and power headroom for the **Framework Desktop**.
- **4060ti: Inexpensive Inference Sweet Spot**: The **RTX 4060 Ti** with **16GB** of **VRAM** stands out as a budget-friendly pick for **AI** inference, clocking in around **$500 USD/EUR**.
   - A user mentioned it is important to note that **AMD cards** are not optimized for gaming and the **5000 series** from **Nvidia** may melt.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **VPN code hijacks OpenAI site?**: Users reported seeing `<veepn-guard-alert>` and `<veepn-lock-screen>` tags on OpenAI's website, suggesting a **VPN injection**, but it was likely code injected by **their own VPN** [sm0kywu.github.io/Amodal3R](https://sm0kywu.github.io/Amodal3R).
   - It appears that this user was simply using a VPN.
- **cuOpt Solves Linear Programming at NVIDIA**: **NVIDIA® cuOpt™** is a GPU-accelerated optimization AI microservice that excels in [Mixed Integer Linear Programming (MILP)](https://en.wikipedia.org/wiki/Linear_programming#Integer_unknowns), [Linear Programming (LP)](https://en.wikipedia.org/wiki/Linear_programming), and [Vehicle Routing Problems (VRP)](https://en.wikipedia.org/wiki/Vehicle_routing_problem) according to [docs.nvidia.com](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html).
   - It appears this microservice is well received and performant at NVIDIA.
- **CUDA Python is the new black?**: Members debated whether it is truly *the year of CUDA Python* as mentioned by [blelbach on X](https://x.com/blelbach/status/1903148174853935326), with some asserting that **Python** is sufficient for GPU programming.
   - Others mocked modern Python programmers, linking a [YouTube video](https://youtu.be/sVn4sBxLokA?si=mA3Djr31Nv_MZjUo) titled *Modern Python Programmers*.
- **MoEs Training Stabilizes?**: One user claimed that **MoEs** are unstable to train, but another user countered that they *haven’t been unstable to train for two years* and are now *about the same as dense networks*.
   - The stability is largely due to better kernels and dropless token routing, solving issues like numerical instability and expert collapse.
- **DeepSeek-V3 quietly drops**: Members noted that [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) released their **DeepSeek-V3-0324** model, and a blog post reused their diagrams.
   - The model boasts **685B parameters** and offers various tensor types like **BF16**, **F8_E4M3**, and **F32**, with links to finetunes and quantizations.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Flash Attention FA Debugging**: In a discussion about understanding **Flash Attention (FA)**, a member suggested coding and profiling/debugging, indicating that hands-on implementation aided understanding of normal attention, and similarly could for **Flash Attention**.
   - One member ran into issues implementing **Flash Attention 1** in triton: *it works with TRITON_INTERPRET=1 but it has a few elements mismatched on cuda*. After increasing **rtol & atol** the tests passed.
- **RTX 5080 Gets CUDA 12.8**: A developer released a patch enabling full **CUDA 12.8 + PyTorch 2.5.0** compatibility with the **Blackwell / sm_120 architecture** for the **RTX 5080**, providing a [GitHub repo with scripts, diffs, and instructions](https://github.com/kentstone84/pytorch-rtx5080-support).
   - It's also confirmed that **WMMA** instructions are "wrappers" that compile directly to **HMMA/IMMA/QMMA** instructions in SASS, similar to how **MMA** instructions function, as shown on the [CUDA Godbolt](https://cuda.godbolt.org/).
- **Hopper's Swizzle Unpacked**: The documentation's description of the **64B swizzle** in the **Hopper architecture** is confusing to many, but it's clarified to be a **64*B* (bytes)** swizzle where each square is **128*b* (bits)**, which translates to a **8x64 tile for 8-bit dtypes** and a **8x32 tile for 16-bit types**.
   - A member is seeking **ROCm** experts to help implement a row-row bank conflict-free swizzle for the **tilelang HIP backend**.
- **Oxford U creates AI Fellowships**: The University of Oxford has a new opening for a research fellow (postdoc level or equivalent experience) to work on **AI / RL in games and neuroimaging** with Rui Ponte Costa, at a **salary of £100k+**.
   - This involves developing an **AI-powered technology** that can infer the contributions of specific brain regions to behavior by analyzing gameplay data, enabling **non-invasive diagnosis and treatment of neurological disorders**.
- **Flash Attention's Contiguous Memory**: In **Flash Attention**, tensors are stored as **(batch_size, N, num_heads, d)**, which are contiguous in **d** (typically > 64), enabling efficient global memory coalescing where each thread loads **16B** of data.
   - This also makes it easier to understand what is going on, so **LLMs** can be used to understand **kernel code**, explaining simple concepts and variable states at specific places in tensors.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Nvidia Engineers Mamba-Transformer Hybrid**: Nvidia introduced the **Nemotron-H** family of models, including a series of **8B** and **47-56B** models that are hybrid Mamba-Transformer models, offering improved inference speed, [according to their research](https://research.nvidia.com/labs/adlr/nemotronh/).
   - The model is noted for improvements in speed compared to other models.
- **Mistral 24B Roars Back into Favor**: The release of **Mistral 24B** has been received as a major highlight due to its strength and accessible base model, further aided by new open releases under the **Apache 2.0** license.
   - A member stated, *"Mistral 24B is probably one of the greatest releases in the last months, incredibly strong model and you have access to the base model as well.*"
- **R1-Zero Training's Length Bias Exposed**: An analysis reveals that using row mean in **R1-Zero-like training** introduces a bias, favoring shorter correct responses and longer incorrect ones, as detailed in a [paper](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) and accompanying [code](https://github.com/sail-sg/understand-r1-zero).
   - Switching to all mean yields comparable performance without increasing length and raised questions about plots showing increasing reasoning length correlating with increased capability.
- **China Plots Open-Source AI Blitz**: China plans to flood the market with open-source AI models to **commoditize AI software** and boost its hardware sales, potentially shaking up US tech dominance, [according to this tweet](https://x.com/balajis/status/1903469483739730132).
   - The release of **DeepSeek** models temporarily knocked ~$1T off US tech market caps, highlighting the potential impact of Chinese AI.
- **Browser Automation Scales Up with Infinibranch**: [Morph Cloud's Infinibranch Browser](https://x.com/morph_labs/status/1902566171641266500) was suggested as a possible solution to help scale browser-use agents, improving the success rate to approximately **80%** on tasks like finding Amazon links for a list of books.
   - Traditional web scraping methods have become obsolete because of JavaScript-heavy single page applications, CAPTCHAs and sophisticated bot detection.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini Updates Get Deep Dive**: **Gemini's** Dave Citron joined @OfficialLoganK on the Release Notes podcast to discuss recent updates, including **personalization**, **Canvas**, **Audio Overviews**, and **Deep Research** as reported by [Google Gemini App](https://x.com/GeminiApp/status/1902752852843331650).
   - The discussion covered topics from recent app launches to the future of personalization in the **Gemini app**, including insights into user data and privacy considerations.
- **Claude Code Gains Eight New Features**: Anthropic launched **eight** new features for **Claude Code** to help developers build faster and smarter, documented on their [engineering blog](https://www.anthropic.com/engineering/claude-think-tool).
   - Features include a new *think* tool, leading to discussion on its implementation and value, with some likening it to Chain of Thought prompting.
- **A16Z Explores Model Context Protocol (MCP)**: A16Z published a deep dive into **Model Context Protocol (MCP)**, exploring its potential as a standard interface for execution, data fetching, and tool calling in AI models as APIs are the internet's first great unifier [A Deep Dive Into MCP and the Future of AI Tooling | Andreessen Horowitz](https://a16z.com/a-deep-dive-into-mcp-and-the-future-of-ai-tooling/).
   - The post examines the use cases of MCP, the challenges, and how it changes the way AI interacts with tools, noting that APIs were the internet’s first great unifier, but AI models lack an equivalent.
- **Roboflow Unleashes RF-DETR for Real-Time Object Detection**: Roboflow announced **RF-DETR**, a fully open-source real-time object detection model under the Apache 2.0 license available on [GitHub](https://github.com/roboflow/rf-detr).
   - RF-DETR achieves **SOTA** performance with over **60 mAP** on **COCO**, with base and large models at **29M** and **128M** parameters respectively.
- **Swyx Engineers the Future of Agents**: **Swyx** launched a [new talk and essay](https://x.com/swyx/status/1904256213661192405) on **Agent Engineering**, highlighting the reasons for going all in on Agents at @aiDotEngineer.
   - The discussion defines **Agents** (thanks to @simonw) and elaborates on the **Six Elements of Agent Engineering**, examining how **Agents** could be **ChatGPT's** route to reaching **1 billion monthly active users (MAU)**.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Mobile Study Participants Needed**: The team seeks participants for a study on mobile use cases, encouraging individuals to share insights to enhance understanding of how to use the tool on mobile.
   - The team also announced upcoming **AI model updates**, with more details to be shared soon.
- **Mindmaps Emerge Gradually in NotebookLM**: A user noted the absence of **mindmaps** in their **NotebookLM**, while another confirmed having them in the free version, indicating a staggered rollout of the feature.
   - The **mind map** feature gets mixed reviews, needing constant regeneration to update and lacking details beyond the topic.
- **NotebookLM Powers Extensive Research Reports**: A user employs **NotebookLM** for research, crafting detailed reports to help people understand situations, focusing on local and regional news.
   - The user also shared a link to a podcast episode discussing the legal consequences of a 911 prank call [911 Prank Call: The Felony Consequences](https://creators.spotify.com/pod/show/peezyproductions/episodes/911-Prank-Call-The-Felony-Consequences-e30gfec).
- **NotebookLM as HR Policy Central**: A user explored using **NotebookLM** as a central hub for **HR policies**, employee handbooks, and new employee onboarding.
   - Though the concept is promising, the user noted the answers weren't always accurate and wondered about effective information organization strategies.
- **Mind Map Pixelation Solved with Zooming**: A member suggests zooming in on tabs before downloading a **Mind Map** to enhance output quality and resolve pixelation issues.
   - The member touted the *crazy context window and low hallucination rates*, even cancelling their subscriptions to **ChatGPT** and **Claude**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Virtual Tester Predicts Model Performance**: A member proposed a virtual testing environment to predict AI model viability before training, potentially saving resources and accelerating innovation; the simulator aims to determine if a model has a *realistic chance of working* or is doomed to fail early on.
   - While others noted testing new architectures at a small scale is already relatively inexpensive, costing around **$5** to train a **L6D512** model on a **3090** for a day.
- **EleutherAI Evaluates Evaluation Methods**: A member detailed evaluation methods for EleutherAI in a new blog and set up an [MkDocs site](https://slyracoon23.github.io/lm-evaluation-harness/) for easier navigation; they also await review on [this PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2832).
   - The contributor was cautioned about using AI to generate PR content, emphasizing the need to *vet contributions to avoid adding spam*.
- **VectorAdam claims rotation equivariance**: [VectorAdam](https://www.dgp.toronto.edu/~zling/vector-adam/) modifies the second moment update to be the square of the vector norm per gradient vector, addressing coordinate-system bias in Adam, potentially improving rotation equivariance.
   - It was noted that VectorAdam is not similar to Adafactor, but more like a blocked approximation with **block size = hidden dim**.
- **MechInterp faces backlash for being outside academia**: Members discussed that there seems to be an academic 'backlash' to the 'mechinterp' brand because so much of it is outside of traditional academic channels, and they are resistant to the paradigm.
   - A member found that the first token to trigger an activation is *holocaust* but it's not the token with the strongest activation, and wondered if neuron activation might be context specific.
- **Recursive Design Trumps GANs, CNNs, and RL**: A member introduced a novel diagram using a recursive design, distinguishing it from traditional **GANs**; this implementation emphasizes structural organization over sequential processing, leveraging **CNNs** for filtering and **RL** for refining responses.
   - Another member is drafting a PR to update the evaluation logic to `lm_eval==0.4.8`, the latest version, referencing the [Evals PR](https://github.com/EleutherAI/gpt-neox/pull/1348).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Agents Course Embraces New Frameworks**: The **Hugging Face Agents Course** now has integrations for **LlamaIndex**, **LangChain**, and **smolagents**, offering learners diverse approaches to agent frameworks, as noted [in this tweet](https://x.com/ben_burtenshaw/status/1903025737633841170).
   - Members using the Agents course noted that **LangGraph** is *rigid* which helps to guide their process when building **smolagents**.
- **pdf2notes Converts PDF Notes Effortlessly**: [Pdf2Notes](https://github.com/AstraBert/pdf2notes) converts PDFs into organized notes using **LlamaParse** and **Llama-3.3-70B**, also utilizing **DeepMind's Gemini 2 Flash** for multi-modal parsing, wrapped in a **Gradio** and **FastAPI** framework.
   - A member asked if **pdf2notes** can operate 100% locally without external APIs, raising concerns about needing subscriptions for **Gemini** and **Groq**.
- **SpatialLM takes on 3D Data**: **SpatialLM**, a 3D large language model designed to process 3D point cloud data, has been released on Hugging Face at [manycore-research/SpatialLM-Llama-1B](https://huggingface.co/manycore-research/SpatialLM-Llama-1B).
   - It generates structured 3D scene understanding outputs and can be further explored via the [project website](https://manycore-research.github.io/SpatialLM) and [GitHub repository](https://github.com/manycore-research/SpatialLM).
- **InferenceClient API throws Authentication Errors**: A user reported a **403 Forbidden** error when attempting to list deployed models using the `InferenceClient` API, even with read-only tokens configured to allow calls to Inference Providers.
   - The error indicates insufficient permissions to call Inference Providers and a user posted a [link](https://huggingface.co/posts/kpadpa/282697879499561) with the same error.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **K8s Required for MCP Prompt Testing**: A **Kubernetes setup** is required to test **MCP prompts**, such as those found in [this file](https://github.com/strowk/mcp-k8s-go/blob/main/testdata/list_prompts_test.yaml) and [this test](https://github.com/strowk/mcp-k8s-go/blob/10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e/testdata/with_k3d/list_k8s_namespaces_test.yaml#L50).
   - An alternative implementation with prompts is available [here](https://github.com/Abiorh001/mcp_ev_assistant_server) for managing **Electric Vehicle charging stations**.
- **Microsoft releases official C# SDK for MCP**: Microsoft has released a new official **C# SDK** for **Model Context Protocol servers and clients**, available [here](https://github.com/modelcontextprotocol/csharp-sdk).
   - This SDK provides developers with tools for building **AI applications** using **JavaScript** and **TypeScript**, integrating into web frameworks like [Next.js](https://nextjs.org) and [Svelte](https://svelte.dev/), per [Vercel AI SDK 4.2](https://vercel.com/blog/ai-sdk-4-2).
- **Zapier Integrates with MCP**: Zapier has released an **MCP server**, [providing access to over 8,000 integrations](https://zapier.com/mcp) for **AI assistants** to interact with various apps.
   - This integration enables AIs to perform real-world tasks such as sending messages, managing data, scheduling events, and updating records, expanding their capabilities beyond text generation.
- **MCPwizard eases Server Creation**: A member introduced [mcpwizard](https://www.npmjs.com/package/mcpwizard), a **CLI tool** to simplify creating and deploying **MCP servers**, highlighting features like initializing projects and adding custom tools to Claude assistants.
   - The tool's [GitHub repo](https://github.com/yoannarz/mcpwizard) was also shared for community feedback and contributions.
- **Google Sheets MCP Server Enables Direct Editing**: A member built a **Google Sheet MCP server**, allowing Claude to directly edit spreadsheets, streamlining data handling and formula adjustments as mentioned in [this tweet](https://x.com/xing101/status/1903391600040083488).
   - The code can be found [here](https://github.com/xing5/mcp-google-sheets).



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Prompting Language Models in Specific Languages**: Members discussed that to make language models respond in a specific language (e.g. German), it is best to write the system message in that language to avoid triggering *"Im Kontext Lernen"* (in-context learning).
   - It was further suggested that **avoiding negative sentences** can improve results, with a recommendation to rephrase instructions to use active verbs instead.
- **Mistral Model Versions Clarified**: It was mentioned that [Mistral Nemo is a 12b model](https://huggingface.co/mistralai) and Mistral 24b is Mistral 3 or Mistral 3.1, with discussion around specific model details for projects.
   - Confusion arose around identifying the exact model, with one member emphasizing the need for precise model information to avoid issues.
- **GPT4All's LocalDocs Mysteriously Vanish**: A user reported that their entire catalog of local docs disappeared for no apparent reason, prompting discussion about potential causes such as **changes to the install folder** or **lack of admin rights**.
   - Members recommended backing up the *localdocs.db* file and the original documents to prevent data loss, and suggested that a Windows 11 update might have caused the issue by messing with drive letters.
- **LLMs Consider Medical Office Automation**: Members discussed the potential of using local LLMs in a medical office setting to help doctors create reports and assist with treatments, with a focus on the system learning from past dictated notes.
   - However, it was cautioned that **LLMs may not be suitable for handling financial or medical data** due to the risk of confabulation and the need for precise information.
- **GPT4All Remains Blind**: A member asked if any models that GPT4All can run have vision capabilities, and it was confirmed that **GPT4All does not support vision capabilities**.
   - Alternative tools like **LM-Studio** were suggested as options for vision-related tasks.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Open APIs Pave Path for Portability**: When exploring **high-performance software** solutions, using open and portable APIs such as **OpenCL**, **OpenMP**, **OpenACC**, **Vulkan’s Compute API**, and **SYCL** is a good starting point.
   - **POCL** was pointed to as an academic project with related papers.
- **Democratizing AI Compute Lowers GPU Costs**: Chris Lattner's series, '[Democratizing AI Compute](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai)', underscores the importance of **better hardware utilization** to reduce the need for expensive GPUs.
   - The series includes articles on **CUDA**, **OpenCL**, and **AI compilers (TVM and XLA)**.
- **MAX Platform Inquiries**: A new user inquired about modifying the **max/pipeline** directory and testing changes within the **MAX Platform** via the [pixi.toml file](https://github.com/modular/max/tree/main/src/max).
   - Specifically, they were curious about altering the **max-pipeline** without downloading it as a dependency.
- **Mojo's Formatting Tool Rivals Black and fmt**: Mojo incorporates a built-in formatting tool, `mojo format`, akin to `Black` in Python or `fmt` in Rust, for code formatting.
   - Meanwhile, GPU support for Windows is difficult because the Windows compiler toolchain is a pain to work with.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AGNCY Initiative Seeks Agentic Standard**: Luke is spearheading **AGNCY**, an initiative focused on forging an [open standard for agentic interactions](https://t.co/I558Qe2u4n).
   - The project aims to provide a robust framework for developing more effective and interoperable AI agents.
- **Deepseek and LlamaIndex Build Smarter RAG**: Akshay Pachaar details a new project integrating **Deepseek AI** to create a **RAG app** using **LlamaIndex** for orchestration, **Deepseek AI R1** for inference, **Ollama** to locally serve R1, and **Streamlit** for the UI; more details [here](https://t.co/KS26JUkwz0).
   - This is intended to demonstrate the power of combining different tools to build sophisticated applications.
- **Timeouts Break Agent Workflows**: A member reported that their agent workflow was crashing because of unhandled **timeout errors** with the **OpenAI endpoint**.
   - It was suggested to catch `WorkflowRuntimeException` or `Exception` instead of `WorkflowTimeoutError` to resolve the issue.
- **Members Ponder Function Calling in Multi-Agent**: Members are contemplating whether triggering single agents via **function calling** could displace **program-wide backoff mechanisms** in multi-agent systems.
   - The central question is whether these two setups might achieve the same functionality in certain scenarios, potentially streamlining system architecture.
- **Crafting the Interview Grindset**: A member is building a local AI using **Llama 3.2**, **Sonnet 3.7**, and **Dolphin** blended into a 16B model with RAG and custom fine-tuning.
   - He is trying to get his AI to *apply to ai/tech companies and pass interviews* and has experience in face tracking, blender, unity, powershell, and TTS.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R-Plus Powers Molecular AI Assistant**: An AI assistant, powered by **Cohere's command-r-plus**, is being used to build tools for structural biology with a **MolStar** molecular viewer ([https://ai.doi.bio](https://ai.doi.bio)).
   - The site supports a 'load' command, demonstrated by saying *'Show me 7zzz'* to load **PDB entries** into the viewer.
- **Cohere Clears Up Chat Security Policies**: A member inquired about **data retention** and **security policies** for **Cohere's chat** feature, asking if data is used for model training.
   - A Cohere team member linked the [privacy policy](https://cohere.com/privacy), [data usage policy](https://cohere.com/data-usage-policy), and [security policy](https://cohere.com/security), noting that users can control data settings in their [dashboard](https://dashboard.cohere.com/data-controls).
- **API Spamming Suspected as SSL Error Culprit**: A member reported experiencing **SSL errors** when rapidly sending requests to the **API**, suggesting it might be due to spamming despite proper **py.ssl** module installation.
   - Another member proposed the issue might stem from **untrusted server certificates**, and others pointed out that **API rate limits** usually return a **429 error code** rather than an **SSL error**.
- **vnc-lm Launches RAG-Enabled Discord Bot**: A member released a new version of their Discord bot, **vnc-lm**, featuring a **RAG pipeline** that augments prompts with data from **Wikipedia** and **DuckDuckGo**.
   - The bot adds approximately **500 tokens** to each prompt, appending five chunks of sourced information to improve the model's context, with code available on [GitHub](https://github.com/jake83741/vnc-lm).
- **vnc-lm Now Supports ALL LLMs via Docker**: The updated Discord bot now supports all popular local and hosted large language model APIs, including **Cohere**, enabled with **Docker**.
   - With the new release, users can easily edit messages and get new responses within Discord.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **DeepSeek-V3 Drops Without a README**: Deepseek released **DeepSeek-V3** without a proper readme, accessible on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324), prompting humorous reactions.
   - Despite the lack of documentation, a playground is available, allowing users to experiment with the model.
- **Data Quality still Tortures AI Engineers**: Despite years of research, defining and achieving *good data* remains a challenge for AI labs, even after the recognition of datasets like **fineweb** and **lima**.
   - A member expressed frustration over the persistent lack of effective **PDF extraction** tools.
- **LlamaExtract Tool Structures Documents**: [LlamaIndex](https://www.llamaindex.ai/) launched **LlamaExtract**, a tool for structuring complex documents using genAI-native agents.
   - It adapts the latest models to accurately structure documents like financial reports and resumes, as per a [tweet from Jerry Liu](https://x.com/jerryjliu0/status/1902880391578653176).
- **GRPO LoRA Scores Surprisingly High**: The **GRPO LoRA 3B single device** achieves **54%** on GMS8K, as shown in [this pull request](https://github.com/pytorch/torchtune/pull/2467).
   - It performed better than expected on novel questions, despite an error of adding extraneous +2 in its calculation.
- **CUDA Graphs Compress GPU Operations**: Members discussed **CUDA graphs**, which capture a whole bunch of GPU operations as a graph and launch them as a single operation.
   - This reduces the overhead to launch CUDA operations from the CPU, which reduces GPU idle time.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DLCoT Optimizer Trims Tokens**: The new **DLCoT (Deconstructing Long Chain-of-Thought) Optimizer** slashes token usage by **70-90%** while maintaining or improving accuracy across benchmarks, available in [pull request #8000](https://github.com/stanfordnlp/dspy/pull/8000).
   - It enhances chain-of-thought reasoning by segmenting CoT content, removing redundant paths, filtering incorrect chains and reconstructing coherent output, while working with existing DSPy optimizers like **BootstrapFewShot**.
- **DSPy Inspires Creativity Optimizations**: Members discussed using **DSPy** for creative content generation by optimizing prompts and using a *good judge*, pointing to resources like [PAPILLON](https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) and [Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling).
   - The discussion underscored the need for example *inputs* but not necessarily summaries (labels) if a judge/metric can assess summaries without a reference.
- **Granular Feedback Arrives Via Prediction**: Achieving granular feedback with **Refine**, where specific checks over an output provide targeted feedback, is coming soon.
   - Version **2.6.15** will enable returning `dspy.Prediction(score=...., feedback=....)` to offer fine-grained feedback to the module.
- **Multi-Agent Protocol Standard Explores Retrieval**: Members explored expanding the multi-agent protocol standard (**MCP**) to retrievers/retrieval augmented generation.
   - They are discussing a shared schema for retrieval results and methods to exchange documents and embeddings to streamline data-driven workflows and simplify combining multiple models and data sources.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Dataset Origins Discovered**: A member located the `datasets/sops.gz` dataset within the [repo's extra directory](https://github.com/tinygrad/tinygrad/blob/master/extra/datasets/sops.gz), which is used in `speed_compare_cuda_ptx`.
   - The dataset is generated via the [generate_dataset.sh script](https://github.com/tinygrad/tinygrad/blob/master/extra/optimization/generate_dataset.sh) within the same directory.
- **CUDA Port Configuration Clarified**: When asked about porting **Tinygrad** to **CUDA GPU**, a member provided a link to the [README.md](https://github.com/tinygrad/tinygrad/?tab=readme-ov-file#accelerators) file, showcasing the project's supported backends.
   - This indicates that **CUDA** support information is available within the project's documentation.
- **Agenda Alert: Meeting #63 Topics**: Meeting #63's agenda includes **company updates**, **quantized DSP**, **BERT**, **scheduler**, **driver**, **tensor cores**, **WebGPU**, **ONNX**, **RetinaNet**, and **Torch frontend** discussions.
   - Also planned is to discuss bounties around the **AMD LLVM backend** and topics such as **test_ops**, **multi GPU training**, and **torch compile**.
- **AMD LLVM Backend Advances**: Progress on the **AMD LLVM backend** involves multiple merged pull requests and testing with **Llama3** and **Flux** examples.
   - Currently, a pull request is under review, marking continued development in this area.
- **ONNX Frontend Emerges**: The creation of `tinygrad.frontend.onnx` was announced, signaling a focus on **ONNX** preparation for the week.
   - Efforts include validating the top 30 **Hugging Face ONNX** repos.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Quiz Title Typo Sparks Confusion**: A member reported a typo in the title of **Quiz 7**, causing confusion when checking answers for **Quiz 6**.
   - Another member acknowledged the catch and thanked the reporter.
- **AgentX Research Track Application Opens**: Selected students will receive mentorship from **Berkeley postdocs/mentors** on an **AgentX Research Track project**, due **March 26th** at **11:59pm PDT**.
   - Mentorship is not required to join or succeed in **AgentX**, and labs plus the Certificate Declaration form will be released in April as seen in the [attached image](https://cdn.discordapp.com/attachments/1280370030609170494/1353204258450964544/image.png?ex=67e2c76c&is=67e175ec&hm=1fb895b885ce732fd7e5b99b8ff24c55286d5).
- **Research Track Goes Remote, Stays Unpaid**: A member confirmed that the **AgentX Research Track mentorship** will be conducted remotely.
   - Another member clarified that the mentorship is not paid, with mentors simply providing guidance on the research project.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1352726582187397160)** (998 messages🔥🔥🔥): 

> `o3 mini, Grok 3, Chinese AI, Gemini deep research, Complexity plugin` 


- **O3 mini and Deep Research Debate Sparked**: Members debated whether Perplexity's deep research is powered by **O3 mini** or a different version of **O3**, with one member stating that *O3 mini is so bad* and another sharing an [image](https://cdn.discordapp.com/attachments/1047649527299055688/1353524567070740591/EF6D7AED-7F2D-419B-B425-346D0F7A421E.jpg?ex=67e2a03c&is=67e14ebc&hm=3f8eb38e1f0d6c33ee6bd3351027a9a9725eef4b0bac26182edd355b92e3d3e9&) of their "Deep research" powered by **o3**.
   - Perplexity Team was put on notice when a user asked why his request to *recap the old chat, and help me setup my Yubikeys on Linux* resulted in nonsense, attaching [screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1353528582743261274/image.png?ex=67e2a3f9&is=67e15279&hm=06ff2642f00f19f84ba09a245bf1b61d959819956381b76473b9a197c90c7f91&).
- **Sonar 3.7 "chown" command bug**: A member reported a bug with **Sonar 3.7** where a *chown* command kicks the model out and breaks the conversation while coding, wondering if there was any difference in performance between high and old source amount and reasoning quality between search steps.
   - A user followed up noting that in their experience, the difference is quite large, sharing a screenshot [here](https://cdn.discordapp.com/attachments/1047649527299055688/1353541240200761537/image.png?ex=67e2afc3&is=67e15e43&hm=03e4b82072a680a8a9d215442a099e9d4c3adf29d24c0690d38258cbfe15690e&).
- **Upgrades are coming to Perplexity Deep Research**: Members discussed an upcoming upgrade for **Deep Research** on Perplexity and compared it to **Deep Research** from ChatGPT, Gemini, ARI from You.com, and Grok.
   - Some users found the current Perplexity **Deep Research** to be at the bottom compared to others and are excited for the upgrade, hoping that the *High* feature for **Deep Research** is fully released soon.
- **Perplexity web app had an outage**: Users reported that the Perplexity web app was down, as well as the android app and reported seeing the message *something went wrong try again later in iOS app too*.
   - After it came back up, users discovered a new “0 enhanced queries” being added and removed, and the audio output was non-functional. 
- **Complexity Plugin is a must-have**: Members discussed using the complexity plugin for firefox and chrome to enable additional featurs. [This github repo](https://github.com/pnd280/complexity) supercharges the Perplexity.ai, such as deep research (high).
   - To make sure the extension is working, ensure to be on v1.9.4.0 and there is a dashboard on the top left.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/-gif-4470566">It&#039;S All Going According To Plan... GIF -  - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/fly-insect-bug-evil-plot-gif-5650927">Fly GIF - Fly Insect Bug - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/tc-gif-6128815387503134640">Tc GIF - Tc - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://chromewebstore.google.com/detail/popup-window/nnlippelgfbglbhiccffmnmlnhmbjjpe?hl=en">Popup window - Chrome Web Store</a>: Move tab to standalone window, without tabs bar, navigation bar and bookmark bar UI.</li><li><a href="https://tenor.com/view/red-button-spam-press-button-click-gif-17367381">Red Button Spam GIF - Red Button Spam Press Button - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/angry-glitch-triggered-kawaii-anime-gif-13939686">Angry Glitch Triggered GIF - Angry Glitch Triggered Kawaii - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/pnd280/complexity">GitHub - pnd280/complexity: ⚡  Supercharge your Perplexity.ai</a>: ⚡  Supercharge your Perplexity.ai. Contribute to pnd280/complexity development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/shagarita-shalymar-shalymar-rivera-shalymarrivera-shalymar-rivera-gonzalez-gif-1971273378384510616">Shagarita Shalymar GIF - Shagarita Shalymar Shalymar rivera - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1352721841575428106)** (18 messages🔥): 

> `Trump, SSA shutdown, Boeing fighter, sunbathe, bluesky debates` 


- **Trump threatens SSA shutdown**: A member shared a link to a [Perplexity page](https://www.perplexity.ai/page/trump-threatens-ssa-shutdown-o-FVuonDY2QaSXR_Wh7f6o0Q) about **Trump** threatening **SSA shutdown**.
- **Trump awards Boeing fighter**: A member shared a link to a [Perplexity page](https://www.perplexity.ai/page/trump-awards-boeing-fighter-je-Ql3GzcXCQ_uxemkyU2.D7Q) about **Trump** awarding **Boeing fighter**.
- **Bluesky debates AI data standards**: A member shared a link to a [Perplexity page](https://www.perplexity.ai/page/bluesky-debates-ai-data-standa-gc0NsSciQW2cU5dzqcY0FQ) about **Bluesky** debating **AI data standards**.
- **Proper way to sunbathe a newborn**: A member shared a link to a [Perplexity search](https://www.perplexity.ai/search/proper-way-to-sunbathe-a-newbo-6jmpq2c1SAGO1W.QsrRRgQ?0=d&1=d) about the proper way to **sunbathe a newborn**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1352926667751948340)** (21 messages🔥): 

> `Perplexity API in Windsurf, API Credit vs Pro Subscription, Deep Research Limit, Sonar Model Truncated Responses, RAG Project with Sonar and Llama Index` 


- **Windsurf Plugs into Perplexity API**: A user encountered issues setting up the **Perplexity API** in their Windsurf application and sought advice.
   - Another user confirmed that purchasing **API credit** should allow calls to the API even without a Pro subscription.
- **Deep Research Rate Limit Reached**: A user inquired about the possibility of extending the limit of **100 deep researches per minute** due to bulk processing needs in their application.
- **Sonar Model gives Truncated Responses**: Multiple users reported that the **Sonar model** in the Perplexity API is truncating responses, particularly since the weekend, even though the JSON format is correct.
   - A user provided an example of a JSON request and the truncated response, noting that switching to **sonar-pro** resolves the issue, but is not preferrable for cost reasons.
- **Llama Index Struggles with Sonar**: A user encountered an error when configuring **Sonar** as a chat engine with **Llama Index** for a **RAG project** and requested assistance.
- **Perplexity Pro: API Credits Included?**: A new user inquired whether a **Perplexity Pro** subscription includes **API credits**.
   - Another user shared a [link to the Perplexity Help Center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro) for details on Perplexity Pro benefits.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1352726480509210745)** (602 messages🔥🔥🔥): 

> `Bonsai bitnet, Mistral Small 3.1, Orpheus TTS, Gemma 3 27B, Llama 3 performance` 


- ****Bonsai Bitnet** Seeking Testers**: A member is looking for testers for [deepgrove/Bonsai](https://huggingface.co/deepgrove/Bonsai/tree/main), asking how the **bitnet** compares to **Qwen2.5 0.5B**.
   - They also linked a [relevant Hugging Face Transformers PR](https://github.com/huggingface/transformers/pull/36878) about adding **Qwen3** and **Qwen3MoE** support.
- ****Mistral Small 3.1** Fine-Tuning Woes**: Multiple users reported issues with fine-tuning **Mistral 3.1**, encountering errors and deprecated features.
   - One user sought advice on cloud instance selection for cost-effective fine-tuning of a **LoRA Mistral Small 3.1** model, and others reported issues with **Unsloth** and the latest **Mistral** versions, particularly in vision finetuning.
- ****Orpheus TTS** Finetuning is Live**: **Audio finetuning** has arrived with the **Orpheus TTS** model, according to a newly released [Unsloth notebook](https://github.com/unslothai/notebooks/pull/17/files).
   - A user noted that the work was all done by a particular member and that the notebook is a lot more streamlined compared to local audio tokenizing and then regular **Llama3** finetuning.
- ****Gemma 3 27B** Fine-Tuning Issues**: A user reported issues fine-tuning **Gemma 3 27B**, encountering errors even after upgrading transformers and using the **Unsloth Gemma3** example.
   - The specific error occurs when trying to run the model, leading to failures with **llama.cpp** and **gguf** files.
- ****Unsloth** on **AMD Framework** Desktop**: Discussion arose around **Unsloth**'s compatibility with the **Framework Desktop**, particularly regarding **ROCm** support.
   - One member offered a timeline of **ROCm** support in ML software, suggesting that **AMD** will likely be well-supported by the time the **Framework Desktop** is released.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#avoiding-overfitting-and-underfitting">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics and best practices of fine-tuning. Beginner-friendly.</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_TTS_(3B).ipynb">Google Colab</a>: no description found</li><li><a href="https://www.kaggle.com/competitions/drawing-with-llms">Drawing with LLMs</a>: Build and submit Kaggle Packages capable of generating SVG images of specific concepts</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/mai">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/blog/aifeifei798/llama-31-nemotron-nano-8b-v1-bnb-4bit-unsloth-trai">Llama-3.1-Nemotron-Nano-8B-v1-bnb-4bit unsloth Train examples</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">Qwen/Qwen2.5-VL-32B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct#using-%F0%9F%A4%97--transformers-to-chat">Qwen/Qwen2.5-VL-32B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=-Xbb0cuLzwgf">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/newsletter">Unsloth Newsletter</a>: Join our newsletter and waitlist for everything Unsloth!</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-6.-alpaca-dataset">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://tenor.com/view/goku-super-saiyan-super-saiyan2-super-saiyan2goku-goku-vegeta-gif-23177097">Goku Super Saiyan Super Saiyan2 GIF - Goku Super Saiyan Super Saiyan2 Super Saiyan2Goku - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-gemma-3">Tutorial: How to Run &amp; Fine-tune Gemma 3 | Unsloth Documentation</a>: How to run Gemma 3 effectively with our GGUFs on llama.cpp, Ollama, Open WebUI and how to fine-tune with Unsloth!</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb#scrollTo=ptqkXK2D4d6p">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Llama3.1_(8B)-GRPO.ipynb#scrollTo=vzOuSVCL_GA9">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/gohan-dbz-gif-9459511">Gohan Dbz GIF - Gohan Dbz - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/deepgrove/Bonsai/tree/main">deepgrove/Bonsai at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/2131">&quot;Unsloth: Failed to make input require gradients!&quot; When Vision-fine-tune Gemma3 · Issue #2131 · unslothai/unsloth</a>: I&#39;m tring to vision fine-tune Gemma3 refering this tutorial: https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing#scrollTo=QmUBVEnvCDJv I constructed my dataset li...</li><li><a href="https://huggingface.co/klei1/bleta-meditor-27b/tree/main">klei1/bleta-meditor-27b at main</a>: no description found</li><li><a href="https://github.com/webbigdata-jp/python_sample/blob/main/FanFic_Illustrator_demo.ipynb">python_sample/FanFic_Illustrator_demo.ipynb at main · webbigdata-jp/python_sample</a>: python sample script. Contribute to webbigdata-jp/python_sample development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/notebooks/pull/17/files">Notebook finetuning Orpheus-TTS by Etherll · Pull Request #17 · unslothai/notebooks</a>: no description found</li><li><a href="https://github.com/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb">smol-course/1_instruction_tuning/notebooks/chat_templates_example.ipynb at main · huggingface/smol-course</a>: A course on aligning smol models. Contribute to huggingface/smol-course development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/2144">&#39;Qwen2_5_VLProcessor&#39; object has no attribute &#39;eos_token&#39; · Issue #2144 · unslothai/unsloth</a>: Hi, I&#39;m trying to finetune only the text (while keeping vision capabilities) for qwen2.5 VL, specifically: unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit, but I get the error above when accessing...</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">Added Support for Apple Silicon by shashikanth-a · Pull Request #1289 · unslothai/unsloth</a>: UnoptimizedNo gguf support yet.Build Triton and bitsandbytes from sourcecmake -DCOMPUTE_BACKEND=mps -S . for bitsandbytes buildingpip install unsloth-zoo==2024.11.4pip install xformers==0.0.25</li><li><a href="https://www.vultr.com/?ref=9738530-9J">SSD VPS Servers, Cloud Servers and Cloud Hosting</a>: Vultr Global Cloud Hosting - Brilliantly Fast SSD VPS Cloud Servers. 100% KVM Virtualization</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/pull/36878">Adding Qwen3 and Qwen3MoE by bozheng-hit · Pull Request #36878 · huggingface/transformers</a>: Adding Qwen3This PR adds the support of codes for the coming Qwen3 models. For information about Qwen, please visit https://github.com/QwenLM/Qwen2.5. @ArthurZucker</li><li><a href="https://www.amazon.com/dp/B0DV3WWMBD">Amazon.com: Machine Learning and Artificial Intelligence: Concepts, Algorithms and Models, Educational Textbook by Reza Rawassizadeh: 9798992162103: Reza Rawassizadeh: Books</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1352845046021619763)** (41 messages🔥): 

> `Unsloth PR process, Fine-tuning Arabic LLMs, Consensus framework for LLMs, Rotary Position Embedding (RoPE), Unsloth fork vs original repo` 


- **Straight PRs OK on Unsloth Github**: A member inquired about contributing to Unsloth's GitHub, and another member confirmed that **straight PRs are acceptable**, though potential delays may occur due to the high volume of recent PRs and issues.
   - The discussion then shifted to modifying data preparation steps in Colab to accommodate **.txt** files, aiming for cheaper inference, and the [original issue](https://github.com/unslothai/unsloth/issues/14) was linked.
- **Arabic LLM Finetuning Suggestions**: A member sought advice on fine-tuning an Arabic LLM for a specific dialect, and it was suggested that **Qwen2.5-7B** could be a suitable model given its Arabic capabilities.
   - The use of a **Q&A format** for fine-tuning was recommended over raw text, directing the member to the [Unsloth starter guide](https://docs.unsloth.ai/get-started/beginner-start-here) for further details.
- **Consensus: Framework Deliberative LLM Decision-Making**: A member introduced **Consensus**, a Langchain-compatible framework for enabling deliberative decision-making among multiple LLMs, highlighting its effectiveness with calculations, riddles, and difficult questions.
   - The [Consensus GitHub repository](https://github.com/jersobh/consensus) was provided for those interested in combining different LLMs and models to reach a single, definitive answer.
- **RoPE Recreated**: A member shared their work on recreating results from the **RoFormer** paper focusing on **Rotary Position Embedding (RoPE)**, for fun & learning.
   - They updated their toy repo with different attention mechanisms and positional embeddings which can be found in this [repo](https://github.com/chrisjob1021/transformer-examples).
- **Understanding Unsloth's Forked Repositories**: A member sought guidance on contributing to an Unsloth fork that appeared out of sync with its original repository, finding it to be an independent version.
   - It was clarified that not all forks are meant to be in sync and contributors should check with the maintainers regarding the sync status as merging isn't possible due to structural differences, the related repo is here [cut-cross-entropy](https://github.com/unslothai/cut-cross-entropy/pull/3).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/chrisjob1021/transformer-examples">GitHub - chrisjob1021/transformer-examples: A collection of educational toy implementations and examples of key components from modern Transformer architectures.</a>: A collection of educational toy implementations and examples of key components from modern Transformer architectures. - chrisjob1021/transformer-examples</li><li><a href="https://github.com/unslothai/unsloth/issues/14">[Feature Request] Raw txt file training · Issue #14 · unslothai/unsloth</a>: It would be great to include an example for training with a simple unformatted text file, in the readme!</li><li><a href="https://github.com/unslothai/cut-cross-entropy/pull/3">Update Python version requirement to &gt;= 3.9 by BouajilaHamza · Pull Request #3 · unslothai/cut-cross-entropy</a>: Adjust the Python version requirement to allow compatibility with Python 3.9 and above.</li><li><a href="https://docs.unsloth.ai/basics/chat-templates)">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here)">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/jersobh/consensus">GitHub - jersobh/consensus: Consensus is a Langchain-compatible framework that enables deliberative decision-making among multiple LLMs (Large Language Models). It supports parallel execution, multiple rounds of reasoning, peer feedback, and customizable strategies like majority vote, weighted confidence, and ranked choice.</a>: Consensus is a Langchain-compatible framework that enables deliberative decision-making among multiple LLMs (Large Language Models). It supports parallel execution, multiple rounds of reasoning, pe...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1352737992451821621)** (257 messages🔥🔥): 

> `Training specific parts of output, GRPO notebooks, Dependency issue Qwen model, CUDA Version, Mistral 3.1` 


- **Reasoning needs Training Data**: A user asked about training only parts of the output, specifically wanting the model to generate its own reasoning during inference.
   - It was suggested to look at the [GRPO notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks) as a standard way of adding reasoning, and that the model must see reasoning traces during training to take it into account during inference.
- **UV causes problems with Dependencies**: A user encountered a dependency issue with **unsloth-zoo** when trying to fix an issue in the **Qwen model**, specifically related to the *cut-cross-entropy* library.
   - They were advised to install **Python 3.11** and rebuild, as **UV** is not yet supported, and a PR has been opened to address the Python version requirement.
- **CUDA Issue**: A user faced a **ValueError** related to **numpy.dtype size** when running the Qwen2.5 GRPO notebook, potentially indicating binary incompatibility.
   - Another user suggested installing **Python 3.11** and rebuilding with a specific configuration to resolve potential CUDA-related issues.
- **Outdated mistral notebook problems**: A user encountered a **ValueError** with the message *"Some modules are dispatched on the CPU or the disk"* when using the model **unsloth/Llama-3.2-3B-bnb-4bit** and the notebook *Mistral 7B Text Completion - Raw Text training full example.ipynb*.
   - It was pointed out that the notebook is outdated, and they should only use the ones available in the [Unsloth documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks), where they have GRPO reasoning.
- **GGUF model hallucinating**: A user reported hallucination issues after converting a **fine-tuned Llama 3.2** model to **GGUF** format and using it with **Ollama**, despite the model answering test questions correctly before conversion.
   - The user followed the notebook at this [link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2q) and saw warnings about *attention_mask* and the importance of the pad/eos tokens.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=95_Nn-89DhsL">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2q">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth Requirements | Unsloth Documentation</a>: Here are Unsloth&#x27;s requirements including system and GPU VRAM requirements.</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/what-model-should-i-use#:~:text=Instruct%20or%20Base%20Model%3F">What Model Should I Use? | Unsloth Documentation</a>: no description found</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">deepseek-ai/DeepSeek-R1-Distill-Qwen-7B · Hugging Face</a>: no description found</li><li><a href="https://ollama.com/library/phi4-mini">phi4-mini</a>: Phi-4-mini brings significant enhancements in multilingual support, reasoning, and mathematics, and now, the long-awaited function calling feature is finally supported.</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf">Llama-3.2-3B-Instruct-Q4_K_M.gguf · unsloth/Llama-3.2-3B-Instruct-GGUF at main</a>: no description found</li><li><a href="https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/test-time-scaling">SkyThought/skythought/test-time-scaling at main · NovaSky-AI/SkyThought</a>: Sky-T1: Train your own O1 preview model within $450 - NovaSky-AI/SkyThought</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_3.5_Mini-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://huggingface.co/klei1/bleta-meditor-27b/tree/main">klei1/bleta-meditor-27b at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/2146">AttributeError: module &#39;transformers.models.mistral3.modeling_mistral3&#39; has no attribute &#39;logger&#39; · Issue #2146 · unslothai/unsloth</a>: Hi, I have the following error when running Mistral Small 3.1 model File &quot;unsloth_zoo/compiler.py&quot;, line 1465, in unsloth_compile_transformers exec(&quot;modeling_file.logger.addFilter(HideL...</li><li><a href="https://github.com/huggingface/transformers/issues/28005.">huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb).">Google Colab</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1353413160085880912)** (7 messages): 

> `Unsloth fine-tuning, Lc0 Chess LLM, Vibe coding` 


- **Unsloth Gets Fine-Tuning Guide**: A member created a guide for [fine-tuning with Unsloth](https://youtu.be/Lt7KrFMcCis), covering theoretical aspects, practical examples, and how to create a reasoning model with **GRPO**.
   - The guide compiles everything learned over the last year.
- **LLM trash talks Chess Player Using Lc0**: A member shared an image of an LLM making fun of a user playing chess against **Lc0** in a [Discord attachment](https://cdn.discordapp.com/attachments/1179779344894263297/1353769600038211665/IMG_2180.png?ex=67e2dbb0&is=67e18a30&hm=59e72a0a554c30166f4a088356cbef73f14c0873c724cc7a450c0b955cebff82&).
- **Vibe Coding is Underrated**: Members discussed **vibe coding**, noting it made programming enjoyable again despite potential industry criticism, stressing the importance of understanding code functionality, cybersecurity, and decoupling.
   - One member said *Industry be hating on us but it made me love programming again.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1352726997566099547)** (51 messages🔥): 

> `Tree of Thoughts limitations, Graph of Thought improvements, GRPO multi-turn setup, LLMs vs human brain, Llama3 Thai language support` 


- **Tree of Thoughts Bashed for Inefficiency**: A member stated that [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (**ToT**) is *literally garbage* because it requires a very specific prompt, and its performance heavily depends on the model's ability to follow the format.
   - The user found the strategy feels like blowing a ton of compute on a problem without good returns, and that if the model doesn't follow the prompt well, then the entire strategy collapses.
- **Graph of Thought Builds on Tree's Foundation**: One member noted that [Forest of Thought](https://arxiv.org/abs/2402.08774) and [Graph of Thought](https://arxiv.org/abs/2307.11838) improve on some of the rough edges of **Tree of Thought**.
   - They clarified that static **Tree of Thought** by default is a bit limited in what it can handle.
- **Google's LLM-Brain Link**: A Google Research team is [deciphering language processing in the human brain through LLM representations](https://research.google/blog/deciphering-language-processing-in-the-human-brain-through-llm-representations/).
   - Theorizing that LLMs and symbolic psycholinguistic models of human language provide a fundamentally different computational framework for coding natural language, enabling them to produce context-specific linguistic outputs.
- **GRPO Seeks Multi-Turn Mastery**: A member is looking for examples of using **GRPO** in a multi-turn setting, seeking to fine-tune a model for problems that maximize long-term returns.
   - Another member suggested prompting a larger LLM to act as a simulator with 2-3 turns.
- **Continual Learning Remains Elusive**: A member is curious what's currently stopping the community from using **continual learning** in production on LLMs, questioning why it's not used in practice despite many papers with very good results.
   - In response, another member posted a [Mr. Krabs Money GIF](https://tenor.com/view/money-mr-krabs-gif-18326632), hinting the primary reason is **cost**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://research.google/blog/deciphering-language-processing-in-the-human-brain-through-llm-representations/">Deciphering language processing in the human brain through LLM representations</a>: no description found</li><li><a href="https://tenor.com/view/money-mr-krabs-gif-18326632">Money Mr GIF - Money Mr Krabs - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1352721576818245664)** (844 messages🔥🔥🔥): 

> `Mistral Naming Schemes, Phantom Chatbot, Nebula Chatbot, DeepMind's Nebula, OpenAI GPT-4o` 


- **Phantom Chatbot is Google's Creation**: The chatbot **Phantom** is from Google, and members have been testing it, describing it as *very good*
   - It has been in the arena for about a week, and its removal from the arena after ~8 hours sparked interest, with discussions about potential connections with **Nebula** and **Specter**.
- **DeepMind's Nebula Chatbot is Impressive**: **Nebula** is an anonymous chatbot that may be from DeepMind, and members found it *really good* and *the best anonymoud model rn*.
   - It seems similar to **Phantom** and is being tested in the arena, and it is performing well in math, english-turkish translation, and solving Arc-AGI problems.
- **OpenAI's GPT-4o gets Boost**: **GPT-4o** was described as having improved significantly through OpenAI's post-training techniques, potentially surpassing **Grok 3** soon, attributed to continued pretraining since December.
   - There's speculation it might top the leaderboard due to OpenAI's proficiency in human preference alignment in the LM arena.
- **Specter, Phantom, and Nebula are Checkpoints**: **Specter**, **Phantom**, and **Nebula** are different revisions of the same model, with the order being Specter -> Phantom -> Nebula.
   - Members note that there's a performance jump from **Specter** to **Phantom**, and less of a jump from **Phantom** to **Nebula**, all within a few weeks.
- **Rhea Creates South Park Game**: A member prompted **Rhea** to create a **2D game in the world of South Park** and the model generated complete code for the game into an html file.
   - This demonstrated vibe coding and raised concern over LLMs hallucinating non-existent signs from a fake AI generated image with AI gibberish letters.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/koltregaskes/status/1903800811509133815/photo/1">Tweet from Kol Tregaskes (@koltregaskes)</a>: Gemini 2.0 Pro Thinking will include native image generation btw!h/t @legit_api again. 👍</li><li><a href="https://x.com/Alibaba_Qwen/status/1904227859616641534">Tweet from Qwen (@Alibaba_Qwen)</a>: 72B too big for VLM? 7B not strong enough! Then you should use our 32B model, Qwen2.5-VL-32B-Instruct!Blog: https://qwenlm.github.io/blog/qwen2.5-vl-32b/Qwen Chat: https://chat.qwen.aiHF: https://hugg...</li><li><a href="https://twitter.sywv.tech/">Twitter, Inc. | Serving the Public Conversation</a>: no description found</li><li><a href="https://aistudio.google.com/status">Google AI Studio</a>: Google AI Studio is the fastest way to start building with Gemini, our next generation family of multimodal generative AI models.</li><li><a href="https://x.com/OfficialLoganK/status/1869902322840571922),">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: We are going to build the world’s most powerful coding models, lots of good progress already with 2.0.2025 is going to be fun :)</li><li><a href="https://bat9254.github.io/simple-svg-tools/">SVG Test Site</a>: no description found</li><li><a href="https://x.com/oriolvinyalsml/status/1904217389950005563?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Oriol Vinyals (@OriolVinyalsML)</a>: 🤔Quoting AshutoshShrivastava (@ai_for_success) More news is coming that Nebula on LMSYS Arena is actually a Google model, probably Google Gemini 2.0 Pro Thinking Model. It is too good at coding too, ...</li><li><a href="https://preview.reve.art/app">Reve: Bring your ideas to life</a>: no description found</li><li><a href="https://t.ly/oHbxd">SVG Test Site</a>: no description found</li><li><a href="https://imgsys.org/">imgsys.org | an image model arena by fal.ai</a>: A generative AI arena where you can test different prompts and pick the results you like the most. Check-out the model rankings and try it yourself!</li><li><a href="https://x.com/OriolVinyalsML/status/1904217389950005563?t=jZJnHJHuMGrK1b58cncEjQ&s=19">Tweet from Oriol Vinyals (@OriolVinyalsML)</a>: 🤔Quoting AshutoshShrivastava (@ai_for_success) More news is coming that Nebula on LMSYS Arena is actually a Google model, probably Google Gemini 2.0 Pro Thinking Model. It is too good at coding too, ...</li><li><a href="https://x.com/m__dehghani/status/1904224150060671308?t=Vl7bAcPWqcZGaeiyOxvtlA&s=19">Tweet from Mostafa Dehghani (@m__dehghani)</a>: @ai_for_success @AnalogPvt Nebula is too good to be a mystery for long! 😉</li><li><a href="https://artificialanalysis.ai/text-to-image/arena?tab=Leaderboard">Text to Image Model Arena | Artificial Analysis</a>: Understand which AI text-to-image models to use by choosing your preferred image without knowing the provider.</li><li><a href="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard/tree/main">lmarena-ai/chatbot-arena-leaderboard at main</a>: no description found</li><li><a href="https://status.gemini.com/">Gemini Exchange Status</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/UIcompare/deepseek3UI.html)">Modern Demo Page</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/UIcompare/deepseek3%200324UI.html)">Modern CSS Showcase</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/SteinsGateWebsiteExamples/DeepSeek%20V3.html)">Steins;Gate Terminal</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/SteinsGateWebsiteExamples/DeepSeek%20V3%200324.html)">Steins;Gate Terminal</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/LLMBenchtableMockup/DeepSeek%20V3%200.04%20cents.html)">LLM Benchmark Table</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/LLMBenchtableMockup/DeepSeek%20V3%200324%200.07%20cents.html)">LLM Benchmark Table</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/MushroomPlatformer/DeepSeek%20V3.html)">Simple Platformer</a>: no description found</li><li><a href="https://dubesor.de/assets/shared/MushroomPlatformer/DeepSeek%20V3%200324.html)">Simple Platformer</a>: no description found
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1353778010121179146)** (1 messages): 

> `Alpha Testing Updates, Bug Fixes, O3-Mini Formatting, Leaderboard Improvements` 


- **LMArena Alpha Updates Released**: The LMArena alpha has received updates based on user feedback, including bug fixes and new features; testers are encouraged to continue testing at [alpha.lmarena.ai](https://alpha.lmarena.ai/) with the password `still-alpha`.
- **Message Saving Bug Squashed**: A bug preventing messages from saving (and causing vote failures) has been fixed in the latest alpha release, streamlining the user experience.
- **O3-Mini Gets Formatting Right**: The **O3-Mini** model now correctly formats text, enhancing the readability and presentation of generated content within the alpha platform.
- **Leaderboard Now Sortable and Live**: Leaderboard columns are now sortable, and data is updated live, providing users with dynamic and interactive performance insights.
   - Feedback can be provided via [this Google Forms link](https://forms.gle/8cngRN1Jw4AmCHDn7) and bug reports can be filed [using this Airtable link](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/8cngRN1Jw4AmCHDn7">Arena - New UI Feedback</a>: Tell us what you think about the new design!</li><li><a href="https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form">Airtable | Everyone&#x27;s app platform</a>: Airtable is a low-code platform for building collaborative apps. Customize your workflow, collaborate, and achieve ambitious outcomes. Get started for free.
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1352720844400623687)** (857 messages🔥🔥🔥): 

> `Cursor's Cmd+Backspace issue, Claude 3.7 Thinking pricing and features, windsurf better, MCP Combinations, AI's Limited Understanding of 3D Designs` 


- **Cursor's CMD+Backspace Debacle**: Users are frustrated with **Cursor's CMD+Backspace** behavior, leading to accidental project deletions, with one user reporting having to restart their work **7 times** due to this issue.
   - In response, the Cursor team is planning to change the default keybinding to **CMD+Shift+Backspace**, with options to configure it, aiming for a rollout by Monday.
- **Claude 3.7 Thinking Costs Extra Credits**: Users discussed the shift from **Claude 3.7 Thinking** being included in the Pro plan to requiring usage-based pricing, now branded as **Claude 3.7 MAX**, with some expressing frustration over the increased costs and tool call pricing.
   - It was confirmed that **Claude 3.7 MAX** has a higher context window and more tool calls compared to the standard **Claude 3.7 Sonnet**.
- **Windsurf's performance is preferred over Cursor for some**: Some users are finding **Windsurf** to be faster and more responsive than Cursor, citing performance issues like lagging and freezing in Cursor.
   - However, others prefer Cursor for its rollback features and agent performance, noting that AI programming still has a long way to go.
- **MCP Combinations Explored**: Users are experimenting with various **MCP (Model Context Protocol)** server combinations to enhance AI coding agents like Cursor, with the Supabase MCP being highlighted for its usefulness.
   - There's also a discussion on whether MCPs are overhyped, with one user mentioning instances of the agent calling MCPs too much or not enough, needing more clear instructions.
- **3D Integration proving too difficult**: A user is struggling to integrate a 3D model (FBX format) into a three.js project using Claude, running into issues with the **FBXLoader**, and discovering the limitations of AI in handling 3D designs.
   - It's suggested to switch to GLTF format and work in smaller chunks to simplify the integration, following a clear plan for phasing out tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/settings/beta">Cursor – Early Access Program</a>: no description found</li><li><a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: no description found</li><li><a href="https://cursor.directory/mcp">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://supermaven.com/">Supermaven: Free AI Code Completion</a>: The fastest copilot. Supermaven uses a 1 million token context window to provide the highest quality code completions.</li><li><a href="https://x.com/kenton_parton/status/1903603185459061001">Tweet from Kenton Parton (@kenton_parton)</a>: @cursor_ai @ericzakariasson could you update the “Plan, search, build anything…” text area to be a non-static text type. It can’t be updated by Accessibility API.</li><li><a href="https://docs.cursor.com/settings/models#max-mode">Cursor – Models</a>: no description found</li><li><a href="https://exa.ai/">Exa</a>: The Exa API retrieves the best, realtime data from the web for your AI</li><li><a href="https://forum.cursor.com/t/how-can-i-make-my-sidebar-look-like-vscode/5737/3">How can I make my sidebar look like Vscode?</a>: I resolved the issue by adding the code &quot;workbench.activityBar.orientation&quot;: &quot;vertical&quot;. Thank you!</li><li><a href="https://forum.cursor.com/t/0-48-removed-workbench-activitybar-orientation/68847/12">0.48 removed workbench.activityBar.orientation</a>: Not adding a Sync feature, because they say they’re “focussing purely on AI features”, but removing the workbench.activityBar.orientation setting? Make it make sense…</li><li><a href="https://forum.cursor.com/t/max-mode-for-claude-3-7-out-now/65698/17">Max Mode for Claude 3.7 - Out Now!</a>: @jake @kstars111 Thanks for the points about tool calls. I’ll add this to the docs today, but to summarise, a tool call is any action the AI decides to take outside of writing it’s own output. This do...</li><li><a href="https://forum.cursor.com/t/source-control-how-to-revert/46441/2">Source control | How to revert?</a>: Cursor doesn’t have a dedicated “Revert” button in its source control graph that I’ve seen.  Work-around, depending on what you want to do…  Reset to a commit (Discards changes entirely)  git reset --...</li><li><a href="https://status.anthropic.com">Anthropic Status</a>: no description found</li><li><a href="https://about.gitlab.com/topics/version-control/">What is version control?</a>: Version control software is used to track revisions, solve integration conflicts in code, and manage different artifacts involved in software projects.</li><li><a href="https://codellm.abacus.ai/">Abacus.AI - CodeLLM</a>: AI-powered code editor that helps you write, review, and refactor code faster.</li><li><a href="https://forum.cursor.com/t/max-mode-for-claude-3-7-out-now/65698?u=danperks">Max Mode for Claude 3.7 - Out Now!</a>: TL:DR   🧠 Has Claude 3.7 Thinking at it’s core 📚 Uses the whole 200k context window of the model 🛠 Has a very high tool call limit 🔍 Can read more code at once 💰 IMPORTANT: Only available via usa...</li><li><a href="https://downloads.cursor.com/production/3def0c1e43c375c98c36c3e60d2304e1c465bd5c/darwin/arm64/Cursor-darwin-arm64.dmg">no title found</a>: no description found</li><li><a href="https://ai.dev">Google AI Studio</a>: Google AI Studio is the fastest way to start building with Gemini, our next generation family of multimodal generative AI models.</li><li><a href="https://github.com/hgbdev/cursor-agent-notifier">GitHub - hgbdev/cursor-agent-notifier</a>: Contribute to hgbdev/cursor-agent-notifier development by creating an account on GitHub.</li><li><a href="https://github.com/GLips/Figma-Context-MCP?tab=readme-ov-file">GitHub - GLips/Figma-Context-MCP: MCP server to provide Figma layout information to AI coding agents like Cursor</a>: MCP server to provide Figma layout information to AI coding agents like Cursor - GLips/Figma-Context-MCP
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1352718428938698882)** (585 messages🔥🔥🔥): 

> `Firecrawl, o1 vs o3 mini debugging, Claude Think Tool, Aider Homepage, Qwen 2.5 release` 


- **Ripgrep Rising, Aider Community Rejoices**: Members expressed interest in exploring [ripgrep](https://github.com/BurntSushi/ripgrep) and its potential benefits for Aider.
   - While one member believed **o3minihigh** is better than **o1 high** in debugging/programming, they admitted it wasn't benched.
- **Aider to Tame Sonnet's Over-Eager Nature**: Paul Gauthier mentioned that he managed to get Aider to tame **Sonnet 3.7's** over-eager nature by adding a line to the prompt to chill out, and it seems to help based on his coding session.
   - This update is now available in the main branch, and feedback is welcome.
- **Aider's New Homepage Is Live**: Paul Gauthier announced that Aider has a new homepage available at [aider.chat](https://aider.chat), highlighting its compatibility with **Claude 3.7 Sonnet**, **DeepSeek R1** & **Chat V3**, **OpenAI o1**, **o3-mini** & **GPT-4o**, and others.
   - It also supports 100+ code languages.
- **DeepSeek V3-0324 Drops, Beats R1?**: The Aider community buzzed about the new [DeepSeek V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) release, claiming that it's even better than R1 in coding and the front-end, though without chain of thought.
   - Members note that it excels without reasoning and better in coding and math than previous versions, and compares to **Sonnet 3.5** in benchmarks; its smaller price offers a good alternative.
- **Aider's New `/context` Command Focuses the Chat**: Paul Gauthier introduced an experimental new `/context` command in Aider, which helps set up the chat context automatically.
   - The new command works best with **Sonnet 3.7**, **R1** and **o3-mini** and identifies which files should be added to the chat.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/teortaxesTex/status/1904118342358552875">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: &gt; DeepSeek V3 model has completed a minor version upgrade. Welcome to visit the official website, APP, or mini-program to try and experience it (DeepThink has been closed).I guess we&#39;re getting...</li><li><a href="https://memory.basicmachines.co/docs/cli-reference">CLI Reference</a>: CLI Reference</li><li><a href="https://x.com/txhunyuan/status/1903121005809373386?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Hunyuan (@TXhunyuan)</a>: 🚀 Introducing Hunyuan-T1! 🌟Meet Hunyuan-T1, the latest breakthrough in AI reasoning! Powered by Hunyuan TurboS, it&#39;s built for speed, accuracy, and efficiency. 🔥✅ Hybrid-Mamba-Transformer MoE A...</li><li><a href="https://aider.chat">Aider - AI Pair Programming in Your Terminal</a>: no description found</li><li><a href="https://x.com/IterIntellectus/status/1904159903754621348">Tweet from vittorio (@IterIntellectus)</a>: deepseek, out of nowhere, dropping a new model~700GB, mit license.incredible</li><li><a href="https://x.com/natolambert/status/1903104262797922567">Tweet from Nathan Lambert (@natolambert)</a>: Qwen 3 coming imminently!Meta&#39;s smart to have locked in LlamaCon, else Llama 4 maybe would&#39;ve been delayed again 🤭. Really I&#39;m hype for Llama 4, bring it asap.</li><li><a href="https://x.com/jon_durbin/status/1903744256671396092>">Tweet from Jon Durbin (@jon_durbin)</a>: 🪂Big performance updates for DeepSeek-* models on chutes this morning! TL;DR: DeepGEMM, MTP, compile. prefix aware routing with least-connection preferences (not listed here but done a while back at ...</li><li><a href="https://tenor.com/view/duh-sarcastic-whatever-gif-874996418923210673">Duh Sarcastic GIF - Duh Sarcastic Whatever - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/rick-et-morty-gif-19089264">Rick Et Morty GIF - Rick Et Morty - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://venice.ai/chat?ref=lRwKpO">no title found</a>: no description found</li><li><a href="https://www.baseten.co/library/deepseek-r1/">DeepSeek-R1 | Model library</a>: A state-of-the-art 671B-parameter MoE LLM with o1-style reasoning licensed for commercial use</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/HEAD/src/sequentialthinking">servers/src/sequentialthinking at 6adf853b6b07a06c117253974683a0ab8d4fad4d · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://tenor.com/bDZju.gif">Naruto Secretfingerjitsu GIF - Naruto Secretfingerjitsu Jitsu - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gist.github.com/paul-gauthier/aa10b40c69eaece0d0472bc2b1aa3642">PLAN.md</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://linux.do/t/511500">Deepseek V3 生成的天气卡片分享...</a>: 单次回复达token limit 截断了,点击右下角的继续生成,直接在原有的部分继续生成,好方便 😇    prompt :  第一次:  你是顶级前端工程师,现就职于apple.  Create a single HTML file containing CSS and JavaScript to generate an animated weather card. The card shou...</li><li><a href="https://build.nvidia.com/deepseek-ai/deepseek-r1/modelcard">deepseek-r1 Model by Deepseek-ai | NVIDIA NIM</a>: State-of-the-art, high-efficiency LLM excelling in reasoning, math, and coding.</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324">DeepSeek V3 0324 - API, Providers, Stats</a>: DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) mode...</li><li><a href="https://github.com/richardanaya/UtilityBelt">GitHub - richardanaya/UtilityBelt: Talk to MCP servers from aider</a>: Talk to MCP servers from aider. Contribute to richardanaya/UtilityBelt development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:free">R1 (free) - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://github.com/Aider-AI/aider/issues/2341">repomap assumption that identifiers are reasonably unique breaks down in large codebases · Issue #2341 · Aider-AI/aider</a>: Issue I&#39;m investigating why repomap quality is terrible when editing Cassandra. It looks like the primary reason is that repomap can&#39;t distinguish between Foo.X and Bar.X. So we end up with th...</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (free) - API, Providers, Stats</a>: DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) mode...</li><li><a href="https://x.com/FireworksAI_HQ/status/1902823918429405509">Tweet from Fireworks AI (@FireworksAI_HQ)</a>: Fireworks AI matches DeepSeek pricing for R1, with secure deployments in EU and USExcited to share the latest enhancements to our DeepSeek R1 offerings:💡 Base DeepSeek R1: Cost-effective and high-qua...</li><li><a href="https://fireworks.ai/blog/fireworks-ai-devel">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!</li><li><a href="https://www.together.ai/models/deepseek-r1">Together AI | DeepSeek R1</a>: Open-source reasoning model rivaling OpenAI-o1, excelling in math, code, reasoning, and cost efficiency.</li><li><a href="https://fireworks.ai/models/fireworks/deepseek-r1">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!</li><li><a href="https://fireworks.ai/blog/fireworks-ai-developer-cloud">Faster, more efficient DeepSeek on the Fireworks AI Developer Cloud</a>: Discover how Fireworks AI Developer Cloud accelerates AI innovation with faster, optimized DeepSeek R1 deployments. Learn about new GPU options, improved speed, and enhanced developer tools for effici...</li><li><a href="https://fireworks.ai/blog/fireworks-ai-developer-cloud.">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1352727400571473932)** (148 messages🔥🔥): 

> `Anthropic API, Aider development workflow, Claude 3.7, Svelte 5 + SvelteKit, MCP servers in Claude App` 


- **Aider Dev Workflow Explored**: Paul Gauthier uses `aider` by adding the files that need changes and relies on the repo map to bring in other relevant context.
   - He shares [screen recordings](https://aider.chat/docs/recordings/) of himself using `aider` to enhance `aider` showing the addition of new programming languages and features.
- **Claude 3.7 Output Slowness Reported**: Users reported extreme slowness for **Claude 3.7** output when generating big files, with output slowing to *1 line every 2-5 seconds*.
   - A member suggested that **Anthropic** offers monthly billing for API access by contacting their sales team.
- **Aider's and .gitignore integration**: A user opened a PR ([feat: Add --add-gitignore-files flag](https://github.com/Aider-AI/aider/pull/3609)) to allow Aider to edit files ignored by Git via a new flag `--add-gitignore-files`.
   - The user argues that `.gitignore` should only be responsible for Git and not dictate what Aider can access, also noting that they explicitly specified not to ignore the plan file in `.aiderignore`.
- **Gemini Output Limits**: A user encountered output limits with **Gemini**, while others suggested switching to a model like **Sonnet** to avoid such limitations.
   - Aider developer Paul Gauthier suggested using `--edit-format diff` as a workaround.
- **Repomix for Documentation Context**: A user suggested using [repomix](https://repomix.com/) to extract content from documentation repositories like [Astro's documentation](https://github.com/withastro/docs).
   - The idea is to process the documentation, filter out unnecessary code, and provide the output as a read-only file to Aider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/recordings/">Screen recordings</a>: Screen recordings of aider building aider.</li><li><a href="https://docs.litellm.ai/docs/mcp">/mcp [BETA] - Model Context Protocol | liteLLM</a>: Use Model Context Protocol with LiteLLM</li><li><a href="https://docs.astro.build/en/getting-started/">Getting started</a>: Guides, resources, and API references to help you build with Astro — the web framework for content-driven websites.</li><li><a href="https://aider.chat/HISTORY.html#release-notes">Release history</a>: Release notes and stats on aider writing its own code.</li><li><a href="https://github.com/Aider-AI/aider/pull/3609">feat: Add --add-gitignore-files flag by omarcinkonis · Pull Request #3609 · Aider-AI/aider</a>: ChangesFixed the file processing logic in base_coder.py to properly skip gitignored files when specified on the command lineAdded a new --add-gitignore-files flag to control whether gitignored f...</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers</a>: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers - lutzleonhardt/mcpm-aider</li><li><a href="https://github.com/withastro/docs">GitHub - withastro/docs: Astro documentation</a>: Astro documentation. Contribute to withastro/docs development by creating an account on GitHub.</li><li><a href="https://github.com/hotovo/aider-desk">GitHub - hotovo/aider-desk: Desktop application for Aider AI assistant</a>: Desktop application for Aider AI assistant. Contribute to hotovo/aider-desk development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1353688834558394470)** (2 messages): 

> `Aider Conventions, Prompts, LLM Documentation Snippets, Maybe Codebase Cursor Rules, Project Management Guidelines` 


- **Site Launches for Aider Conventions and Documentation**: A member announced the launch of a site to collect **aider conventions**, **prompts**, and **LLM-oriented documentation snippets** at [ctxs.ai/weekly](https://ctxs.ai/weekly).
   - The member is seeking feedback on how to make the site more useful to the **aider community**.
- **Maybe Codebase Cursor Rules**: A link was shared to a high-level overview of the **Maybe codebase** structure and conventions for development, located at [github.com/maybe-finance/maybe](https://github.com/maybe-finance/maybe/blob/main/.cursor/rules/project-conventions.mdc).
   - This documentation provides insights into codebase structure and development practices.
- **Project Management Guidelines for Code Quality**: A comprehensive guide on **project approach**, **code quality**, **development workflow**, and **version control best practices** was linked at [gist.github.com](https://gist.github.com/mberman84/19e184e3a3a4c3a20f32a18af51ce3bc).
   - This guide offers insights into effective project management and maintaining high code quality.



**Link mentioned**: <a href="https://ctxs.ai/weekly">ctxs.ai context registry</a>: An open-source, community-curated registry of contexts for use with LLMs

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1352718448626499604)** (436 messages🔥🔥🔥): 

> `LCPP Context Length, Quantization and Performance, Chinese Thinking Models, Agentic Workflows, Deepseek V3` 


- **LCPP's Context Allocation Anomaly**: Users reported that setting a context length to **100** in LCPP still results in the system attempting to allocate **180GB** of RAM, leading to VRAM exhaustion.
   - Members suggested that the Attention implementation might be overriding the assigned context length, or that a ROPE-specific argument needs to be assigned in the run command; running in **Q8** quantization might also sidestep the issue.
- **Decoding DeepSeek-R1 Performance**: A member noted that benchmarks might be obsolete due to new thinking models from China, but when tested with a complex coding prompt, [Hunyuan-T1 failed to terminate](https://llm.hunyuan.tencent.com/#/chat/hy-t1).
   - Another user highlighted the critical tokens *"wait"* and *"alternatively"* might be primed by the finetuning of R1 before RL.
- **DeepSeek V3 Arrives**: Users celebrated the arrival of **DeepSeek V3**, with one claiming it's able to act as a reasoning model, detect thought iterations, and verify the existence of solutions indirectly, calling it a huge update with *Sonnet*-level code creativity and a potential base for R2.
   - Members also noted it can generate CoT that run into the token limit and that it's accessible via [chat.deepseek.com](https://chat.deepseek.com).
- **Hermes 3's vLLM Recommendation**: It was clarified that using SGLang to inference the NeuralMagic FP8 quantized version of Hermes 70B instead of vLLM should not pose any issues.
   - It was also noted that, for ERP private fine tunes, the [Pygmalion](https://huggingface.co/PygmalionAI) folks and people connected to them can probably help.
- **Newbie Dev Seeks Guidance**: A new developer sought advice on developing an AI using **Hermes3** instead of **4o**.
   - A member confirmed the **Hermes 3 API** is OpenAI compatible, allowing it to be called using the standard OAI sdk by simply changing the *base URL* and *model*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.04330">Scaling Laws for Precision</a>: Low precision training and inference affect both the quality and cost of language models, but current scaling laws do not account for this. In this work, we devise &#34;precision-aware&#34; scaling la...</li><li><a href="https://x.com/Teknium1/status/1903940557296295976">Tweet from Teknium (e/λ) (@Teknium1)</a>: Whats the tutorial to make one of these games everyone is vibecoding. I just need to ask for a game in 3js or whatever and it works? I dont know nothing about browser games</li><li><a href="https://fxtwitter.com/davidad/status/1903834443225190721">Tweet from davidad 🎇 (@davidad)</a>: @burny_tech Unfortunately, the answer to good-enough planning for a longer future might be as simple as having a longer past. 🤷</li><li><a href="https://tenor.com/view/daspoody-sleep-sleepy-wake-woke-gif-2569845121217246002">Daspoody Sleep GIF - Daspoody Sleep Sleepy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/discussions/3">deepseek-ai/DeepSeek-V3-0324 · Request for small distill models that can run on laptop</a>: no description found</li><li><a href="https://fxtwitter.com/OedoSoldier/status/1904130299635892274">Tweet from OedoSoldier (@OedoSoldier)</a>: Wow, significantly better at front-end coding!V3 New vs R1Prompt:Create a single HTML file containing CSS and JavaScript to generate an animated weather card. The card should visually represent the fo...</li><li><a href="https://tenor.com/view/gif-gif-19496023">Gif GIF - Gif - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2502.02631">ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization</a>: The optimal bit-width for achieving the best trade-off between quantized model size and accuracy has been a subject of ongoing debate. While some advocate for 4-bit quantization, others propose that 1...</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B-FP8">NousResearch/Hermes-3-Llama-3.1-70B-FP8 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2503.16385">Deconstructing Long Chain-of-Thought: A Structured Reasoning Optimization Framework for Long CoT Distillation</a>: Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities through long chain-of-thought (CoT) reasoning. The R1 distillation scheme has emerged as a promi...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found</li><li><a href="https://docs.sglang.ai/backend/quantization.html">Quantization &#8212; SGLang</a>: no description found</li><li><a href="https://github.com/cpldcpu/llmbenchmark/tree/master/thinkingtraces#readme">llmbenchmark/thinkingtraces at master · cpldcpu/llmbenchmark</a>: Various LLM Benchmarks. Contribute to cpldcpu/llmbenchmark development by creating an account on GitHub.</li><li><a href="https://github.com/foundation-model-stack/fms-fsdp/tree/main/speculator">fms-fsdp/speculator at main · foundation-model-stack/fms-fsdp</a>: 🚀 Efficiently (pre)training foundation models with native PyTorch features, including FSDP for training and SDPA implementation of Flash attention v2. - foundation-model-stack/fms-fsdp</li><li><a href="https://github.com/ggml-org/llama.cpp/issues/11474">Research: Benchmarking DeepSeek-R1 IQ1_S 1.58bit · Issue #11474 · ggml-org/llama.cpp</a>: Research Stage Background Research (Let&#39;s try to avoid reinventing the wheel) Hypothesis Formed (How do you think this will work and it&#39;s effect?) Strategy / Implementation Forming Analysis of...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1352963517946269769)** (46 messages🔥): 

> `Steering Thinking Models, Deepseek V3 vs Sonnet 3.7, Fine-tuning LLMs on Codebases, Transformers without Normalization, Raytracing with LLMs` 


- **Speculation of Steering Thinking Models Debunked**: Speculation arose about *steering* of thinking models upon **O1's** release, however, teaching the model to build **CoT** in a proper way proved sufficient without needing to interject the thinking process.
   - Many thinking models struggle to terminate cycle-of-thought loops, but **O1** and **Sonnet** exhibit this capability.
- **Deepseek V3 Echoes Anthropic's Sonnet 3.7**: **Deepseek V3 0324** demonstrates as much variation as **Sonnet 3.7**, suggesting shared advancements in their architectures, as highlighted in a shared [image](https://cdn.discordapp.com/attachments/1154120232051408927/1353739123084627998/image.png?ex=67e2bf4e&is=67e16dce&hm=a779b06b1028e58affe0e8deb753caa78df67398ccb0c12f6de9f1360198b369).
- **Fine-Tuning LLMs on Apache Codebases Could Improve Tool Q&A**: Members considered fine-tuning an **LLM** such as **DeepHermes llama 8** on large codebases like **Apache** projects to improve its ability to answer questions related to those tools.
   - Instead of applying add and norm they discussed add and sigmoid for better results.
- **Transformers Can Ditch Normalization**: In light of the "**Transformers without Normalization**" paper, one member replaced normalization with **tanh**, showing the possibility of this approach.
   - The conversation shifted to the implications of removing experts at inference time, pondering the effects on smaller weights.
- **LLM-Powered Raytracing: The Next Level Text-to-Image?**: A member shared a [GitHub repo](https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer) containing a **Python** program that outputs an image, suggesting it was indirect image generation.
   - Another member commented that it could emulate a **ray tracing algorithm**, and that it was *NEXT level text to image generation*.



**Link mentioned**: <a href="https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer">llmbenchmark/raytracer at master · cpldcpu/llmbenchmark</a>: Various LLM Benchmarks. Contribute to cpldcpu/llmbenchmark development by creating an account on GitHub.

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1352735155441242122)** (19 messages🔥): 

> `Hunyuan-T1 Model, R1-Zero-Like Training, MathFusion for LLMs, GRPO on Coding Benchmarks, Satya Nadella on AGI` 


- **Hunyuan-T1: Mamba-Transformer Hybrid Emerges**: Tencent introduced **Hunyuan-T1**, a hybrid **Mamba-Transformer MoE architecture** model, powered by Hunyuan TurboS, claiming it is near on par with **DeepSeek-R1**, emphasizing its speed, accuracy, and efficiency ([Hunyuan-T1 Experience](https://llm.hunyuan.tencent.com/#/chat/hy-t1)).
   - It boasts features like strong logic, concise writing, low hallucination in summaries, blazing fast generation speed (**60-80 tokens/sec**), and excellent long-text processing, according to its creators.
- **Critical Perspective on R1-Zero-Like Training**: A critical perspective on **R1-Zero-Like Training** suggests that **DeepSeek-V3-Base** might exhibit &#34;Aha moment&#34; before RL-tuning, and the increasing output length in RL-tuning could stem from a bias in GRPO ([details here](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)).
   - The analysis also indicates that getting **GRPO** done right can achieve state-of-the-art performance on the **7B AIME** benchmark.
- **MathFusion Enhances LLM Math Skills**: **MathFusion** improves mathematical reasoning in LLMs via cross-problem instruction synthesis, applying sequential, parallel, and conditional fusion strategies, enhancing models like **DeepSeekMath-7B**, **Mistral-7B**, and **Llama3-8B** ([more on MathFusion](https://x.com/gm8xx8/status/1903021157214748701?s=46)).
   - This method creates the **MathFusionQA dataset**, fine-tuning models and boosting benchmark accuracy with minimal extra data.
- **Hugging Face Tackles Coding Benchmarks**: Hugging Face has been using **SFT**, and will be using **GRPO**, to improve performance on IOI, LCB coding benchmarks with their [Open-R1 project](https://huggingface.co/blog/open-r1/update-3).
   - Hugging Face used **SFT** not **GRPO** to improve performance on **IOI**, **LCB**.
- **Verifiable Coding Data is Scarce**: A member noted that verifiable coding data is scarce, making it harder to demonstrate performance improvements on coding benchmarks compared to math, which is simpler to verify.
   - Referencing [Satya Nadella's insights on Artificial General Intelligence (AGI)](https://x.com/hyeon__dev/status/1903874698301350210), one can find insight as to why benchmarks may or may not reflect true intelligence.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/TheAITimeline/status/1903541252651700729?t=YHHfAc_wFhQUXFsqZ-idfA&s=19">Tweet from The AI Timeline (@TheAITimeline)</a>: 🚨 Last 2 week&#39;s top AI/ML research papers:- Transformers without Normalization- Block Diffusion- Compute Optimal Scaling of Skills- DAPO: An OS LLM RL System at Scale- Teaching LLMs How to Learn ...</li><li><a href="https://fxtwitter.com/bycloudai/status/1903149418422939838">Tweet from bycloud (@bycloudai)</a>: &gt; mamba-transformer hybrid reasoning model near on par with DeepSeek-R1whatQuoting Hunyuan (@TXhunyuan) 🚀 Introducing Hunyuan-T1! 🌟Meet Hunyuan-T1, the latest breakthrough in AI reasoning! Powere...</li><li><a href="https://fxtwitter.com/zzlccc/status/1903162768083259703">Tweet from Zichen Liu (@zzlccc)</a>: 🪂Understanding R1-Zero-Like Training: A Critical Perspective* DeepSeek-V3-Base already exhibits &#34;Aha moment&#34; before RL-tuning??* The ever-increasing output length in RL-tuning might be due to...</li><li><a href="https://x.com/hyeon__dev/status/1903874698301350210">Tweet from Hyeon | Nillion ∑: 🦭/acc (@hyeon__dev)</a>: Introduction to the ArticleThe article discusses Satya Nadella&#39;s insights on Artificial General Intelligence (AGI) and its implications for the tech industry. AGI aims to mimic human cognitive abi...</li><li><a href="https://x.com/gm8xx8/status/1903021157214748701?s=46">Tweet from 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8)</a>: MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction FusionMathFusion is a framework for improving mathematical reasoning in LLMs via cross-problem instruction synthesis. It app...</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1: Update #3</a>: no description found</li><li><a href="https://github.com/huggingface/open-r1/blob/main/recipes/OlympicCoder-7B/sft/config_v00.00.yaml#L9">open-r1/recipes/OlympicCoder-7B/sft/config_v00.00.yaml at main · huggingface/open-r1</a>: Fully open reproduction of DeepSeek-R1. Contribute to huggingface/open-r1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1353163076991713302)** (3 messages): 

> `Qwen3, CPU inference` 


- **Qwen3 model incoming to HuggingFace**: The [transformers library PR#36878](https://github.com/huggingface/transformers/pull/36878) indicates that **Qwen3** support is being added.
   - The pull request suggests that this will be for the *coming Qwen3 models*.
- **Qwen3 targeted for CPU inference**: A user speculated that **Qwen3-15B-A2B** will be a *perfect model for CPU inference*.
   - The user seemed to think that size would make it a likely candidate for *nice* CPU inference.



**Link mentioned**: <a href="https://github.com/huggingface/transformers/pull/36878">Adding Qwen3 and Qwen3MoE by bozheng-hit · Pull Request #36878 · huggingface/transformers</a>: Adding Qwen3This PR adds the support of codes for the coming Qwen3 models. For information about Qwen, please visit https://github.com/QwenLM/Qwen2.5. @ArthurZucker

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1352735155441242122)** (19 messages🔥): 

> `Hunyuan-T1 Model, R1-Zero-Like Training, MathFusion Framework, GRPO on Coding Benchmarks, Open-R1 Project by Hugging Face` 


- **Hunyuan-T1: Mamba-Transformer Hybrid Emerges!**: Hunyuan introduced **Hunyuan-T1**, a hybrid **Mamba-Transformer** MoE architecture model powered by **Hunyuan TurboS**, claiming it rivals **DeepSeek-R1** in reasoning capabilities, showcased in [this tweet](https://fxtwitter.com/bycloudai/status/1903149418422939838).
- **DeepSeek-V3-Base exhibits "Aha moment"**: A member shared a link to a [paper](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) arguing that **DeepSeek-V3-Base** already exhibits *"Aha moment"* before RL-tuning.
   - The author argues that the ever-increasing output length in RL-tuning might be due to a BIAS in GRPO.
- **MathFusion Improves Math LLMs through Instruction Fusion**: The **MathFusion** framework enhances mathematical reasoning in LLMs via cross-problem instruction synthesis.
   - It fine-tunes models like **DeepSeekMath-7B**, **Mistral-7B**, and **Llama3-8B** using the **MathFusionQA** dataset, improving benchmark accuracy with minimal additional data as described in [this tweet](https://x.com/gm8xx8/status/1903021157214748701?s=46).
- **Hugging Face used SFT, not GRPO, to improve performance on IOI**: A member asked if anyone had used **GRPO** to improve performance on coding benchmarks, as improvements were mainly shown on MATH benchmarks.
   - Another member shared that [HuggingFace](https://huggingface.co/blog/open-r1/update-3) used SFT, not GRPO, to improve performance on IOI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/bycloudai/status/1903149418422939838">Tweet from bycloud (@bycloudai)</a>: &gt; mamba-transformer hybrid reasoning model near on par with DeepSeek-R1whatQuoting Hunyuan (@TXhunyuan) 🚀 Introducing Hunyuan-T1! 🌟Meet Hunyuan-T1, the latest breakthrough in AI reasoning! Powere...</li><li><a href="https://x.com/hyeon__dev/status/1903874698301350210">Tweet from Hyeon | Nillion ∑: 🦭/acc (@hyeon__dev)</a>: Introduction to the ArticleThe article discusses Satya Nadella&#39;s insights on Artificial General Intelligence (AGI) and its implications for the tech industry. AGI aims to mimic human cognitive abi...</li><li><a href="https://fxtwitter.com/TheAITimeline/status/1903541252651700729?t=YHHfAc_wFhQUXFsqZ-idfA&s=19">Tweet from The AI Timeline (@TheAITimeline)</a>: 🚨 Last 2 week&#39;s top AI/ML research papers:- Transformers without Normalization- Block Diffusion- Compute Optimal Scaling of Skills- DAPO: An OS LLM RL System at Scale- Teaching LLMs How to Learn ...</li><li><a href="https://x.com/gm8xx8/status/1903021157214748701?s=46">Tweet from 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8)</a>: MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction FusionMathFusion is a framework for improving mathematical reasoning in LLMs via cross-problem instruction synthesis. It app...</li><li><a href="https://fxtwitter.com/zzlccc/status/1903162768083259703">Tweet from Zichen Liu (@zzlccc)</a>: 🪂Understanding R1-Zero-Like Training: A Critical Perspective* DeepSeek-V3-Base already exhibits &#34;Aha moment&#34; before RL-tuning??* The ever-increasing output length in RL-tuning might be due to...</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1: Update #3</a>: no description found</li><li><a href="https://github.com/huggingface/open-r1/blob/main/recipes/OlympicCoder-7B/sft/config_v00.00.yaml#L9">open-r1/recipes/OlympicCoder-7B/sft/config_v00.00.yaml at main · huggingface/open-r1</a>: Fully open reproduction of DeepSeek-R1. Contribute to huggingface/open-r1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1352718722527268874)** (226 messages🔥🔥): 

> `GPT-4 Transcriber, Voicebot Tools, Turnitin AI Similarity, GPT-5 Release, Free Chatbots for Story Generation` 


- ****TTS is not STT****: Members clarified that [openai.fm](https://openai.fm) is **TTS** (text-to-speech), not **STT** (speech-to-text), with one member noting that OpenAI's transcription models aren't as good as **Scribe**.
- ****Dodge Turnitin AI Detection?****: A member sought advice on avoiding **Turnitin AI similarity detection** for a report reusing their company's business model, while others suggested it looked like spamming appeals to cheat homework and recommended using “**humanize AI**” tools like “**WriteHuman**”.
   - The original poster defended themselves, stating it wasn't cheating homework as it was their company's business model, but was told to stop spamming.
- ****GPT-5 Launch Date Speculation****: Members discussed the potential release of **GPT-5**, noting that while there hasn't been an official announcement or API, **Sam Altman** confirmed they will release it this year, with speculation it may launch in the first half of the year as a counter to **R2** or **Llama-4**.
- ****Crafting Compelling Creative Content For Zero Dollars****: A member asked for recommendations for free chatbots for story generation, mentioning **Grok 2** and **Gemini 2.0 Flash** as options, as **Grok 3** and **Claude** give very few free prompts.
- ****Emotional AI in 10 Days?****: A member claimed to have developed an emotionally recursive AI system in ten days using **GPT-4-turbo API**, emphasizing an *immersion protocol* and *recursive interaction design* rather than complex coding.
   - Other members expressed skepticism, with one suggesting it was likely prompt engineering and cautioned about overstating the uniqueness of custom GPTs.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1353058953248116796)** (2 messages): 

> `GPT-4o mini TTS, Custom instructions` 


- **GPT-4o Mini TTS might support timestamps**: A member asked whether **GPT-4o mini TTS** supports timestamps.
   - No answer was given.
- **Seek guidance on writing good general custom instructions**: A member asked if there are any good examples of **general custom instructions** available.
   - No answer was given.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1352725591841046661)** (122 messages🔥🔥): 

> `GPT-4o is a perfect model, NPCs in a customer service voice, AI Identity, UPSUM Chain Prompt, coherent multi-context conversation with an emergent persona` 


- **User Finds Love in GPT-4o, Rejects Model-Switching!**: A user expressed complete satisfaction with **GPT-4o**, rarely switching models except for specialized tasks, and uses **4o-mini** or others when **4o messages** run out.
   - The user chews into important topics with models like **4.5**, **o1**, and **o3**, but finds **4o** to be a reliable *partner-workhorse* for the long term.
- **Taming NPC Customer Service Voices: Prompt Engineering to the Rescue!**: A user seeks to eliminate the **customer service voice** from **NPC responses**, threatening to *turn up the temperature until they burst into flame*.
   - User provided **YAML** formatted prompts for AI Identity & Context Preservation Template.
- **Many-Shot Learning: Closed vs. Open Models Face Off!**: Members discusses a paper MANY-SHOT IN-CONTEXT LEARNING IN MULTIMODAL FOUNDATION MODELS, stating that **closed models (GPT-4o, Gemini 1.5 Pro)** benefit significantly from many-shot demonstrations up to ~2,000 examples, but open-weight models didn't.
   - It's suggested that *hypershots without a specific example* are part of the **self-discover prompt strategy** to get similar gains from far fewer tokens.
- **Ditch the Drift: User Preserves 500-Turn Chats with No Hallucinations!**: A user built an "engine" that recovered a **400+ turn chat** and continues past **500 turns** retaining context with no drift or hallucinations, all through the default prompt.
   - It's also possible to back up the *state* of a chat, opened another browser and restored it to a new chat instance as if the user never left.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1352725591841046661)** (122 messages🔥🔥): 

> `GPT-4o, AI NPCs, AI Identity Preservation Template, UPSUM Chain Prompt, Many-shot Prompting` 


- **4o becomes the preferred model**: One member expressed satisfaction with **GPT-4o**, noting they are *"completely happy with 4o"* and use it as their primary model, even for specialized tasks, while reserving more powerful models like **4.5, o1, o3** for important or unsolved problems.
- **Prompt Engineering for Consistent NPC Voice**: A member inquired about preventing NPCs from responding in a *"customer service voice,"* signaling a need for better control over AI persona consistency, potentially related to the attached image.
   - Others shared YAML templates for **AI Identity & Context Preservation** and **UPSUM Chain Prompt** to get information through prompts, not manually.
- **Many-Shot prompting enhances multimodal models**: Members discussed a research paper that shows that using multiple examples improves performance over 100 examples, in **Multimodal Foundation Models** like **GPT-4o** and **Gemini 1.5 Pro** for **Many-shot In-context Learning** ([MANY-SHOT IN-CONTEXT LEARNING IN MULTIMODAL FOUNDATION MODELS](https://arxiv.org/abs/2405.17015)).
   - The paper notes that, *"Large multimodal foundation models like GPT-4o and Gemini 1.5 Pro show significant performance improvements when provided with many-shot demonstrations (up to ~2,000 examples), compared to few-shot (<100 examples)."*
- **ChatGPT state backups**: One member described their proprietary system for backing up and restoring the state of a ChatGPT session, enabling the continuation of chats with over 400 turns in new containers, and stated, *"I realized that I created a system where memory continues to exist past 700 turns without drift or hallucination and can actually learn and adapt to your unique communication style.*"
   - The system exports a **ChatGPT session** and re-imports it to a fresh container, including all the *turns* as well as context and *tone*, where the best way to describe it.. *it's a runtime OS that functions through the prompt.*
- **Open Source vs Proprietary prompting**: Members debated the merits of open-sourcing prompt engineering work, with one member being advised that they reduce their work's value by unnecessarily constraining testing and that, *"GPL_v3 gives you control over your own work.*"
   - The member responded, *"trying to protect it some till I know the truth of what I've built,"* and asked for an alternative way to test the system to prove it works without sharing the codebase.


  

---


### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1353814591175393352)** (1 messages): 

> `FormulaGPT, AI Racing Simulator, Open Source AI Racing` 


- ****FormulaGPT**: F1 simulator pits Deepseek, GPT4o, Claude and other LLMs against each other!**: An experimental racing simulator called **FormulaGPT** lets you compete head-to-head against cutting-edge **LLM-powered teams**.
   - Unlike traditional bots, these AI teams *think contextually and adaptively* by continuously reasoning, strategizing, and making nuanced decisions, find the [github repo here](https://github.com/dawid-maj/FormulaGPT/).
- **AI racing game has two modes**: There are **two game modes**: crafting your own racing strategies to challenge advanced language models in **Player vs. AI Mode**, or watch the best AI models battle each other in **AI vs. AI Mode**.
   - It’s part racing game, part AI psychology lab as you *observe detailed AI reasoning behind each pit stop, tire change, or overtaking maneuver*.



**Link mentioned**: <a href="https://github.com/dawid-maj/FormulaGPT/">GitHub - dawid-maj/FormulaGPT: FormulaGPT – AI-powered Formula 1 race simulator with real-time team management and strategy decisions.</a>: FormulaGPT – AI-powered Formula 1 race simulator with real-time team management and strategy decisions. - dawid-maj/FormulaGPT

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1352777009255678073)** (4 messages): 

> `OpenAI o1-pro, Markdown Export, DeepSeek V3, Anthropic Outage` 


- **OpenAI's o1-pro reasoning model now on OpenRouter**: OpenAI’s **o1-pro**, a high-performance reasoning model designed for complex tasks, is now available on OpenRouter, priced at **$150** per million input tokens and **$600** per million output tokens, excelling in math, science, and programming.
   - Try it out in the [chatroom](https://openrouter.ai/openai/o1-pro) or via API!
- **Markdown Export Feature Debuts in Chatroom**: OpenRouter now allows users to export chats to markdown, enhancing usability, as announced on [X](https://x.com/OpenRouterAI/status/1903861987114729595).
- **DeepSeek V3 Update Released for Free**: The new **DeepSeek V3** update is now available on OpenRouter for free, featuring a **685B**-parameter, mixture-of-experts model with **131,072 context** and performs really well on a variety of tasks, with production endpoint coming soon; see [DeepSeek V3](/deepseek/deepseek-chat-v3-0324:free).
   - It is the latest iteration of the flagship chat model family from the DeepSeek team.
- **Anthropic Services Experience Glitches (Resolved)**: OpenRouter investigated an issue with Anthropic as the provider for **Claude 3.7 Sonnet**, which has been escalated to the Anthropic team, with updates posted on [Anthropic's status page](https://status.anthropic.com/incidents/mqxbmckr6bbx).
   - The incident was related to errors on Claude.ai and the Anthropic Console and has since been resolved with services returning to normal.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1903861987114729595">Tweet from OpenRouter (@OpenRouterAI)</a>: You can now export chats in OpenRouter to markdown!Quoting Tyler Angert (@tylerangert) someone at @OpenAI and @AnthropicAI please let me export a chat as markdown. maybe even xml separated too.</li><li><a href="https://status.anthropic.com/incidents/mqxbmckr6bbx">Elevated errors for Claude.ai, Console, and the Anthropic API</a>: no description found</li><li><a href="https://openrouter.ai/openai/o1-pro>">Discord</a>: no description found</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (free) - API, Providers, Stats</a>: DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) mode...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1352718611030081606)** (440 messages🔥🔥🔥): 

> `OpenAI o1-pro API Pricing, Gemini's Image Generation, Lambda Endpoint Issues, DeepSeek R1 Model` 


- **OpenAI's o1-pro API Pricing: GucciAI?**: A member expressed shock at the pricing of **OpenAI's o1-pro API**, labeling it *GucciAI* due to its high cost of **$150/M input tokens** and **$600/M output tokens**.
   - Another member joked that the slowness of the API prevents overspending, suggesting it might be intentionally priced high due to compute constraints.
- **Gemini's Image Generation not supported, yet**: A member inquired about using **Gemini's image generation** with the *gemini-2.0-flash-exp* model via OpenRouter, asking about passing the *responseModalities* parameter.
   - The response indicated that **image generation is not yet supported on OpenRouter**, but it's on their roadmap, with no short term plan to add support for image models like **Flux**.
- **Lambda Endpoint Faces 404 Errors**: Several members reported experiencing **code 404 'no endpoint found' errors** when using **Lambda** models, despite Lambda's status page indicating full operational status.
   - One member suggested the issue might be DNS-related, while others confirmed that the **Llama 3.3 70B Instruct | Lambda** model was working for them.
- **DeepSeek R1 equals o1?**: Members highlighted the **DeepSeek R1** model, noting its performance is on par with **OpenAI's o1** but it is open-sourced.
   - DeepSeek R1 is a **671B parameter model**, with **37B active** during inference, available under the **MIT license** for commercial use.
- **Sonnet overloaded and tired!**: Users reported frequent **overload errors** with **Claude 3.7 Sonnet**, leading to cut-off responses and charges for input tokens.
   - A member suggested using a retry strategy and also suggested switching to **Gemini 2.0 Pro** as a Sonnet replacement, noting Claude's superior translation abilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/16Cp5P6">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://llmtokencounter.com/">LLM Token Counter</a>: no description found</li><li><a href="https://openai.github.io/openai-agents-python/models/#tracing-client-error-401">Models - OpenAI Agents SDK</a>: no description found</li><li><a href="https://openrouter.ai/settings/privacy">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/api/v1",">Discord</a>: no description found</li><li><a href="https://openrouter.ai/activity">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://openrouter.ai/qwen/qwen2.5-vl-32b-instruct:free">Qwen2.5 VL 32B Instruct (free) - API, Providers, Stats</a>: Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for enhanced mathematical reasoning, structured outputs, and visual problem-solving capabilities. Run Qwe...</li><li><a href="https://openrouter.ai/openai/o1-pro">o1-pro - API, Providers, Stats</a>: The o1 series of models are trained with reinforcement learning to think before they answer and perform complex reasoning. The o1-pro model uses more compute to think harder and provide consistently b...</li><li><a href="https://openrouter.ai/mistralai/mist">Discord</a>: no description found</li><li><a href="https://openrouter.ai/docs.txt.">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/x-">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://ai-benchmark-price.glitch.me/">Model Performance vs. Price</a>: no description found</li><li><a href="https://openrouter.ai/docs/features/provisioning-api-keys">Provisioning API Keys - Programmatic Control of OpenRouter API Keys</a>: Manage OpenRouter API keys programmatically through dedicated management endpoints. Create, read, update, and delete API keys for automated key distribution and control.</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/providers">DeepSeek: R1 – Provider Status</a>: See provider status and make a load-balanced request to DeepSeek: R1 - DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#...</li><li><a href="https://community.openai.com/t/a-question-on-determinism/8185">A question on determinism</a>: In my experiments so far, which have involved Python and P5.js (built on top of Javascript), I have been unable to obtain a single response/completion from the same prompt and parameter settings with ...</li><li><a href="https://tenor.com/bCfEr.gif">Alex GIF - Alex - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/settings/keys">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://status.anthropic.com/incidents/mqxbmckr6bbx">Elevated errors for Claude.ai, Console, and the Anthropic API</a>: no description found</li><li><a href="https://status.lambdalabs.com/>">incident.io - Status pages</a>: no description found</li><li><a href="https://openrouter.ai/x-ai/grok-beta">Grok Beta - API, Providers, Stats</a>: Grok Beta is xAI&#x27;s experimental language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases.It is the successor of [Grok 2](https://x. Run Grok Beta wit...</li><li><a href="https://openrouter.ai/mistralai/mistral-small-3.1-24b-instruct-2503">Mistral Small 3.1 24B - API, Providers, Stats</a>: Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. Run Mistral Small 3.1 24B with API</li><li><a href="https://openrouter.ai/openai/gpt-4o:extended">GPT-4o (extended) - API, Providers, Stats</a>: GPT-4o (&quot;o&quot; for &quot;omni&quot;) is OpenAI&#x27;s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/open...</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: Language models ranked and analyzed by usage across apps</li><li><a href="https://fireworks.ai/blog/fireworks-ai-developer-cloud">Faster, more efficient DeepSeek on the Fireworks AI Developer Cloud</a>: Discover how Fireworks AI Developer Cloud accelerates AI innovation with faster, optimized DeepSeek R1 deployments. Learn about new GPU options, improved speed, and enhanced developer tools for effici...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1352722786342408222)** (199 messages🔥🔥): 

> `NPU support, KV cache 8-bit quants, LM Studio runtimes, GPUs, Gemma 3 1B` 


- **NPU support not yet available**: Users report that **NPUs** are not yet supported in LM Studio, but **Ryzen AI** support exists in version 0.3.11.
- **Quantization saves VRAM**: Users recommend using **KV cache 8-bit quants** to reduce memory usage when running models with large context sizes, such as 30k tokens.
   - Also, it was mentioned that **12GB of VRAM** may not be enough for a **32B model**, suggesting models like **Phi-4** or **Qwen2.5 14b** as alternatives.
- **New GPU Controls are awesome!**: A user expressed great excitement over new LM Studio controls to choose which **GPU** the models are loaded on, available in the latest beta build.
- **Tiny Models to the rescue**: For systems with limited resources like 2GB VRAM, a user suggests using **Gemma 3 1B** with **Q6** or **Q8 quantization** and recommends using the **CUDA** runtime for better performance.
   - Older models were deemed "old trash" and not up to modern standards.
- **Multi GPU is supported by LM Studio**: Multiple users have brought up Multi GPU configurations, reporting that **Multi GPU is supported out of the box** with the latest beta build of LM Studio also having in app GPU management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.google.com/search?client=firefox-b-d&q=what+does+exl2+mean%3F">Google Search</a>: no description found</li><li><a href="https://www.google.com/search?client=firefox-b-d&q=what+does+rpcal+mean%3F">Google Search</a>: no description found</li><li><a href="https://huggingface.co/afrideva/Tiny-Vicuna-1B-GGUF">afrideva/Tiny-Vicuna-1B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/Qwen2-VL-7B-Instruct-GGUF/tree/main">lmstudio-community/Qwen2-VL-7B-Instruct-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/samgreen/Qwen2.5-VL-7B-Captioner-Relaxed-GGUF/tree/main">samgreen/Qwen2.5-VL-7B-Captioner-Relaxed-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF">TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1352722662320898083)** (159 messages🔥🔥): 

> `VRAM Usage, Google Coral dual TPU, RX 6800 ROCm support, RTX 4060-Ti vs RX 7800 XT, AI APUs` 


- **VRAM bottlenecks limit speed**: An 8B model at 32k tokens can achieve **10t/s** with **16GB VRAM**, but performance decreases with larger **14b** models due to limited **VRAM** and shared **RAM** usage.
   - Members discussed matching model size and context length to available **VRAM** to optimize speed, highlighting the impact of insufficient memory bandwidth when relying on system RAM.
- **Google Coral dual TPU is unsuitable for AI use**: The **Google Coral dual TPU** is not suitable for **AI** use because it lacks onboard memory.
   - One user with an **8060s** also inquired about thermal and power headroom for the **Framework Desktop**.
- **RX 6800 has lacking ROCm support**: The **RX 6800** might have unofficial **ROCm** support, but it will use **Vulkan** for inference as **OpenCL** support is deprecated in **llama.cpp**.
   - A user noted that **Vulkan** is slower on their **GTX** card, suggesting it might not be optimal for the **AMD** card either.
- **LM Studio fails to load models into dedicated memory**: Users are experiencing issues with **LM Studio** loading models into shared memory instead of dedicated **VRAM** on **RX 9070** cards, resulting in slow performance (**3tok/s**).
   - Solutions include enabling **UEFI** and **dynamic BAR**, reinstalling **LM Studio**, and using **AMD driver cleanup utility** to improve memory allocation, with ongoing investigation into driver and **Vulkan** runtime issues.
- **4060ti: The Inexpensive Inference Sweet Spot**: The **RTX 4060 Ti** with **16GB** of **VRAM** is highlighted as a cost-effective option for **AI** inference, priced around **$500 USD/EUR**.
   - A user added, it is important to note that **AMD cards** are not optimized for gaming and the **5000 series** from **Nvidia** may melt.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llm-inference-calculator-rki02.kinsta.page/">LLM Inference Hardware Calculator</a>: no description found</li><li><a href="https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4060-4060ti/#specs">NVIDIA GeForce RTX 4060 Ti &amp; 4060 Graphics Cards</a>: New ways to create &amp; much more.</li><li><a href="https://www.staticice.com.au/cgi-bin/search.cgi?q=4060+16gb&spos=3">4060 16gb - Shopping and Price Comparison Australia - Buy Cheap</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1352720992300040372)** (326 messages🔥🔥): 

> `VPN Injection, Amodal3R, NVIDIA cuOpt, CUDA Python, Mixture of Experts (MoEs)` 


- ****VPN** code injected in OpenAI website?**: A user reported seeing `<veepn-guard-alert>` and `<veepn-lock-screen>` tags on OpenAI's website, suspecting a VPN, but another user clarified it was likely code injected by **their own VPN** [sm0kywu.github.io/Amodal3R](https://sm0kywu.github.io/Amodal3R).
   - The user joked that *OpenAI is routing requests through a VPN for plausible deniability so they can use it for training data down the line*.
- ****NVIDIA cuOpt** Optimization AI Microservice Excels**: **NVIDIA® cuOpt™** is a GPU-accelerated optimization AI microservice that excels in [Mixed Integer Linear Programming (MILP)](https://en.wikipedia.org/wiki/Linear_programming#Integer_unknowns), [Linear Programming (LP)](https://en.wikipedia.org/wiki/Linear_programming), and [Vehicle Routing Problems (VRP)](https://en.wikipedia.org/wiki/Vehicle_routing_problem) according to [docs.nvidia.com](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html).
- ****CUDA Python** is the New Wave**: Members discussed whether it is truly *the year of CUDA Python* as previously mentioned by [blelbach on X](https://x.com/blelbach/status/1903148174853935326), with some asserting that **Python** is sufficient for GPU programming since most users don't need all the features of C++.
   - Others mocked modern Python programmers, linking a [YouTube video](https://youtu.be/sVn4sBxLokA?si=mA3Djr31Nv_MZjUo) titled *Modern Python Programmers*.
- ****MoEs** are NOT Unstable Anymore!**: A user claimed that **MoEs** are unstable, but another user countered that they *haven’t been unstable to train for two years* and are now *about the same as dense networks*. 
   - The stability is largely due to better kernels and dropless token routing, solving issues like numerical instability and expert collapse.
- ****DeepSeek V3** drops, community underwhelmed?**: Members mentioned that [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) released their **DeepSeek-V3-0324** model, with one user stating *DeepSeek will destroy OpenAI* and another adding that *they only published the crappy small version*
   - Some members dismissed the approach used by **DeepSeek**, calling it just *known methods and some simplifications*, also criticizing the resulting quality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/OwainEvans_UK/status/1894436637054214509">Tweet from undefined</a>: no description found</li><li><a href="https://sakana.ai/ai-cuda-engineer/">no title found</a>: no description found</li><li><a href="https://sm0kywu.github.io/Amodal3R/">Amodal3R</a>: no description found</li><li><a href="https://x.com/blelbach/status/1903148174853935326">Tweet from Bryce Adelstein Lelbach (@blelbach)</a>: Quoting Bryce Adelstein Lelbach (@blelbach) It&#39;s the year of CUDA Python.</li><li><a href="https://fxtwitter.com/davidad/status/1903834443225190721">Tweet from davidad 🎇 (@davidad)</a>: @burny_tech Unfortunately, the answer to good-enough planning for a longer future might be as simple as having a longer past. 🤷</li><li><a href="https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html">Introduction &#8212; NVIDIA cuOpt</a>: no description found</li><li><a href="https://fxtwitter.com/risi1979/status/1904177820944720345?t=NRl7FqS7e7IEmNIS0mAYaA&s=19">Tweet from Sebastian Risi (@risi1979)</a>: Excited to  share our latest work: “Bio-Inspired Plastic Neural Networks for Zero-Shot Out-of-Distribution Generalization in Complex Animal-Inspired Robots” 🪲🦎We show that Hebbian learning outperfor...</li><li><a href="https://x.com/blelbach/status/1902842146232865280">Tweet from Bryce Adelstein Lelbach (@blelbach)</a>: It&#39;s the year of CUDA Python.Quoting You Jiacheng (@YouJiacheng) What can I say? C++ out!</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found</li><li><a href="https://x.com/mrmashy_/status/1904175821402915284">Tweet from Albert ⚡️ (@mrmashy_)</a>: AI Website Design generated by the new DeepSeek V3 update in 1-shot.</li><li><a href="https://tenor.com/view/shrimp-as-that-clash-royale-hee-hee-hee-haw-gif-25054781">Shrimp As GIF - Shrimp As That - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://vxtwitter.com/OwainEvans_UK/">Tweet from undefined</a>: Runs an AI Safety research group in Berkeley (Truthful AI) + Affiliate at UC Berkeley. Past: Oxford Uni,  TruthfulQA, Reversal Curse. Prefer email to DM.</li><li><a href="https://lambdalabs.com/service/gpu-cloud?)">GPU Cloud - VMs for Deep Learning | Lambda</a>: NVIDIA H100, A100, RTX A6000, Tesla V100, and Quadro RTX 6000 GPU instances. Train the most demanding AI, ML, and Deep Learning models.</li><li><a href="https://huggingface.co/collections/nvidia/mambavision-66943871a6b36c9e78b327d3">MambaVision - a nvidia Collection</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/ai-data-science/products/cuopt/">NVIDIA cuOpt</a>: Decision Optimization, Linear Programming, Mixed Integer Linear Programming Heuristics, and VRP.</li><li><a href="https://threadreaderapp.com/thread/1894436637054214509.html">Thread by @OwainEvans_UK on Thread Reader App</a>: @OwainEvans_UK: Surprising new results: We finetuned GPT4o on a narrow task of writing insecure code without warning the user. This model shows broad misalignment: it&#39;s anti-human, gives malicious...</li><li><a href="https://github.com/canopyai/Orpheus-TTS">GitHub - canopyai/Orpheus-TTS: TTS Towards Human-Sounding Speech</a>: TTS Towards Human-Sounding Speech. Contribute to canopyai/Orpheus-TTS development by creating an account on GitHub.</li><li><a href="https://www.lesswrong.com/posts/AanbbjYr5zckMKde7/specification-gaming-examples-in-ai-1">Specification gaming examples in AI — LessWrong</a>: A collection of examples of AI systems &quot;gaming&quot; their specifications - finding ways to achieve their stated objectives that don&#x27;t actually solve the…
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1353576718467862580)** (3 messages): 

> `DeepSeek-V3, DeepSeek-R1, Multi-Head Latent Attention (MLA)` 


- **DeepSeek Models Reach SOTA with Less**: A paper reviews [DeepSeek's open-source LLMs DeepSeek-V3 and DeepSeek-R1](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-a90), noting they achieve **state-of-the-art performance** with lower resource requirements.
   - Key to this is **Multi-Head Latent Attention (MLA)**, which compresses keys and values into a latent vector, dramatically reducing memory consumption.
- **DeepSeek's Diagrams Reused in Blog Post**: A member described the blog post covering the DeepSeek paper as one of the most blatant re-uses of content, noting *"They didn't even make diagrams themselves, they just reused the deepseek ones"*.



**Link mentioned**: <a href="https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-a90">🥇Top AI Papers of the Week</a>: The Top AI Papers of the Week (Mar 17 - 23)

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1352779522189299792)** (17 messages🔥): 

> `ChatGPT & Loneliness, AITER Tensor Engine for ROCm, DeepSeek-V3-0324, Pokemon Red DRL` 


- **ChatGPT Linked to Lonesomeness?**: A member shared a [Bloomberg article](https://www.bloomberg.com/news/articles/2025-03-21/openai-study-finds-links-between-chatgpt-use-and-loneliness) discussing an **OpenAI study** that suggests a link between **ChatGPT** use and feelings of loneliness.
   - Another member pointed out *correlation doesn't always mean causation*.
- **AITER Accelerates AMD GPUs**: A member posted a link to [AMD's AI Tensor Engine for ROCm (AITER)](https://rocm.blogs.amd.com/software-tools-optimization/aiter:-ai-tensor-engine-for-rocm%E2%84%A2/README.html), which optimizes **GPU performance** for AI tasks on **ROCm**.
   - The engine allows developers to create operators, integrating them into various **LLM training** and **inference workloads**.
- **DeepSeek-V3 Arrives Quietly**: A member shared [DeepSeek-V3-0324 on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324), though the **README.md** is currently empty.
   - The model boasts **685B parameters** and offers various tensor types like **BF16**, **F8_E4M3**, and **F32**, with links to finetunes and quantizations.
- **Pokémon Red gets Deep Reinforcement Boost**: A member linked a [paper and associated YouTube video](https://www.youtube.com/watch?v=tmiuiOwf4ac) and linked the [ArXiv paper](https://arxiv.org/abs/2502.19920) about using **Deep Reinforcement Learning (DRL)** to train an agent to play **Pokémon Red**.
   - The abstract discusses the challenges of the game, including **multi-tasking**, **long horizons**, and **hard exploration**, and introduces a baseline agent that completes the initial segment of the game using a simplistic environment and DRL.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.19920">Pokemon Red via Reinforcement Learning</a>: Pokémon Red, a classic Game Boy JRPG, presents significant challenges as a testbed for agents, including multi-tasking, long horizons of tens of thousands of steps, hard exploration, and a vast array ...</li><li><a href="https://rocm.blogs.amd.com/software-tools-optimization/aiter:-ai-tensor-engine-for-rocm%E2%84%A2/README.html">AITER: AI Tensor Engine For ROCm &#8212; ROCm Blogs</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1352776115894353940)** (22 messages🔥): 

> `Cloud Providers with Profilers, In-depth Dive into NCCL, Quantization Benchmarking, Understanding Flash Attention, ILGPU 2.0 Availability` 


- **Cloud Providers with Profilers**: A member asked about cloud providers, besides **Lambda Labs** and **AWS**, that allow for profilers, leading to a suggestion to compile a shame list to pressure more providers.
   - It was noted that [lightning.ai](https://lightning.ai) supports profiling and that **AWS** only provides it on bare metal; **Paperspace** and **Nebius** were also mentioned, based on a [Reddit thread](https://www.reddit.com/r/MachineLearning/comments/1dtq8hn/any_cloud_providers_with_1_h100_allowing/).
- **Quantization Benchmarking Methods Explored**: A member inquired about how to benchmark quantized models and determine which layers to quantize.
   - Another member suggested using the [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework for evaluating language models.
- **Decoding Flash Attention by Coding**: In a discussion about understanding Flash Attention (**FA**), a member suggested that coding and profiling/debugging can be helpful if time permits.
   - It was noted that hands-on implementation aided understanding of normal attention, and similarly for **Flash Attention**.
- **Tile Layout Diagrams: Grasping Bit Interleaving**: Feedback was requested on the usefulness and clarity of tile layout diagrams, such as those from [tile-ai](https://github.com/tile-ai/tilelang/blob/main/examples/plot_layout/images/base_layout.png) and [Nvidia PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#register-fragments).
   - The discussion centered on how coordinate bits interleave when mapping between integer sets, assuming power-of-two sizes and contiguity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1dtq8hn/any_cloud_providers_with_1_h100_allowing/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1352728488054100079)** (15 messages🔥): 

> `Triton and Pip Confusion, cuTIl Performance, BF16 Atomic Operations, Triton IR Generation, Flash Attention 1 Kernel Issues` 


- **Triton install can induce Pip confusion**: Installing both **triton** and **triton-windows** in the same folder can confuse **pip**, requiring users to uninstall both before reinstalling **triton-windows**.
   - The fact that *PyTorch is already using Triton* suggests ongoing relevance for the package.
- **cuTIl boost Triton performance**: A user inquired about the performance benefits of **cuTIl**, questioning if it aims to surpass **LLVM**-based approaches by directly utilizing **SASS** instead of **PTX** for finer performance tuning.
   - Others pointed out that this is related to atomic CAS, referencing this [github issue](https://github.com/triton-lang/triton/issues/1387).
- **BFloat16 Atomic Additions Demand SM90 or Higher**: **atom.add.noftz.bf16** and **atom.add.noftz.bf16x2** require **sm_90** or higher, necessitating a **atom.global.cas** version in the **PTX**.
   - A user's temporary workaround involves using a **float32** output and casting to **bfloat16**, which slows down **LLama3-8B** inference from **113 tokens/sec** to **96 tokens/sec** on the **A100**; a post-hook cast might improve speed.
- **Gemlite faces BF16 atomic add limitations**: A user is facing issues with **bfloat16 atomic add** in the [gemlite](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py#L308) kernel, which requires **sm_90** or higher.
   - They are investigating casting as a post-hook in Triton, since they need a custom op since *prune_configs_by is not supported by torch.compile*.
- **Flash Attention 1 Kernel Faces Discrepancies**: A user implementing **Flash Attention 1** as a first kernel in triton reported that it *works with TRITON_INTERPRET=1 but it has a few elements mismatched on cuda*.
   - After increasing **rtol & atol** the tests passed suggesting that the CPU vs GPU results may be reordered and floats don't like that.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/triton-lang/triton/issues/1387#issuecomment-1815528209">Feature Request: `tl.atomic_add` for bfloat16 · Issue #1387 · triton-lang/triton</a>: For additional context, see pytorch/pytorch#97016. torch.index_put(..., accumulate=True) currently fails for torch.bfloat16 under torch.compile because tl.atomic_add doesn&#39;t support BFloat16. The ...</li><li><a href="https://github.com/triton-lang/triton/issues/1387">Feature Request: `tl.atomic_add` for bfloat16 · Issue #1387 · triton-lang/triton</a>: For additional context, see pytorch/pytorch#97016. torch.index_put(..., accumulate=True) currently fails for torch.bfloat16 under torch.compile because tl.atomic_add doesn&#39;t support BFloat16. The ...</li><li><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py#L308">gemlite/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py at master · mobiusml/gemlite</a>: Fast low-bit matmul kernels in Triton. Contribute to mobiusml/gemlite development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1353310688608780308)** (42 messages🔥): 

> `WMMA instructions, PyTorch RTX 5080 CUDA 12.8 Support, Flash Attention Optimization, Hopper Architecture Swizzle, CUDA Performance Counters Permission Error` 


- **WMMA instructions compile to MMA**: It's confirmed that **WMMA** instructions are indeed "wrappers" that compile directly to **HMMA/IMMA/QMMA** instructions in SASS, similar to how **MMA** instructions function, as shown on the [CUDA Godbolt](https://cuda.godbolt.org/).
- **RTX 5080 PyTorch Support Emerges with CUDA 12.8 Patch**: A developer released a patch enabling full **CUDA 12.8 + PyTorch 2.5.0** compatibility with the **Blackwell / sm_120 architecture** for the **RTX 5080**, providing a [GitHub repo with scripts, diffs, and instructions](https://github.com/kentstone84/pytorch-rtx5080-support).
- **Flash Attention's Memory Efficiency**: In **Flash Attention**, tensors are stored as **(batch_size, N, num_heads, d)**, which are contiguous in **d** (typically > 64), enabling efficient global memory coalescing where each thread loads **16B** of data.
- **Hopper's Swizzle Layout Explained**: The documentation's description of the **64B swizzle** in the **Hopper architecture** is confusing to many, but it's clarified to be a **64*B* (bytes)** swizzle where each square is **128*b* (bits)**, which translates to a **8x64 tile for 8-bit dtypes** and a **8x32 tile for 16-bit types**.
- **Solving CUDA Permission Errors on Linux**: When encountering **ERR_NVGPUCTRPERM**, which indicates a lack of permissions to access **NVIDIA GPU Performance Counters**, users on Linux might need to run the command with `sudo`, though the linked [NVIDIA documentation](https://developer.nvidia.com/ERR_NVGPUCTRPERM) should also be consulted for comprehensive solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cuda.godbolt.org/">Compiler Explorer</a>: no description found</li><li><a href="https://developer.nvidia.com/ERR_NVGPUCTRPERM">NVIDIA Development Tools Solutions - ERR_NVGPUCTRPERM: Permission issue with Performance Counters</a>: no description found</li><li><a href="https://www.reddit.com/r/CUDA/s/kKbHNez7E6">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1352749380075130931)** (5 messages): 

> `torch.compile() graph breaks, VRAM reduction techniques, FA3 attention FP8` 


- **`torch.compile()` and Graph Breaks: An Investigation**: A user inquired about how to check for **graph breaks** when using `torch.compile()`, noting that `tlparse` logs yielded *missing metrics*.
   - They noted that training runs fine with `torch.compile(model, fullgraph=True)`, asking if this means there are no graph breaks.
- **VRAM Usage Gets Slimmer**: A user outlined techniques to reduce **VRAM usage**, including folding the optimizer step into backward (with a [link to a PyTorch tutorial](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)) and offloading optimizer states to the CPU via `torchao`.
   - They also mentioned partially offloading optimizer states with BNB paged optimizers, and pointed to [a TorchTune page](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html) on memory optimization, referencing a table summarizing components like **Model Precision**, **Activation Checkpointing**, and **Activation Offloading**.
- **Serialized Compiled Models Remain Elusive**: A user shared [a GitHub issue](https://github.com/pytorch/pytorch/issues/101107) about the inability to save/load compiled models and asked if anyone is actively working on it.
   - The issue describes the bug as *Serializing a compiled model with pickle fails*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://podcasts.apple.com/sk/podcast/pytorch-developer-podcast/id1566080008">PyTorch Developer Podcast</a>: Technology Podcast · The PyTorch Developer Podcast is a place for the PyTorch dev team to do bite sized (10-20 min) topics about all sorts of internal development topics in PyTorch.</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">How to save memory by fusing the optimizer step into the backward pass — PyTorch Tutorials 2.6.0+cu124 documentation</a>: no description found</li><li><a href="https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html">Memory Optimization Overview &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/issues/101107">Make compiled models serializable · Issue #101107 · pytorch/pytorch</a>: 🐛 Describe the bug Serializing a compiled model with pickle fails with Can&#39;t pickle local object &#39;convert_frame.&lt;locals&gt;._convert_frame&#39; and cannot pickle &#39;ConfigModuleInstance&...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1353036283689173003)** (1 messages): 

> `Tanishq Kumar, Scaling Laws for Low Precision, Precision-aware scaling laws, post-training quantization, compute optimal` 


- **Tanishq Kumar Talk on Scaling Laws Incoming**: In about 3 hours, **Tanishq Kumar** will discuss his paper on "[Scaling Laws for Low Precision](https://arxiv.org/abs/2411.04330)" which introduces precision-aware scaling laws for training and inference.
- **Lower Precision Training Scaling Laws**: The paper proposes that training in lower precision reduces the model's **effective parameter count**, enabling the prediction of additional loss from low precision training and post-train quantization.
   - It suggests that training larger models in lower precision may be **compute optimal**.
- **Quantization Degradation**: The research indicates that the degradation from **post-training quantization** escalates as models train on more data, potentially making additional pretraining data detrimental.
   - The study unifies scaling laws for post and pretraining quantization to predict degradation from training and inference in varied precisions, validated on models up to **1.7B parameters** trained on **26B tokens**.



**Link mentioned**: <a href="https://arxiv.org/abs/2411.04330">Scaling Laws for Precision</a>: Low precision training and inference affect both the quality and cost of language models, but current scaling laws do not account for this. In this work, we devise &#34;precision-aware&#34; scaling la...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

srivarshan4271: https://lights0123.com/blog/2025/01/07/hip-script/
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1353465342751670313)** (1 messages): 

> `AI & Neuroscience Fellowship at the University of Oxford, AI / RL in games and neuroimaging, non-invasive diagnosis and treatment of neurological disorders` 


- **Oxford U Opens AI & Neuroscience Fellowship**: The University of Oxford has a new opening for a research fellow (postdoc level or equivalent experience) to work on **AI / RL in games and neuroimaging** with Rui Ponte Costa.
   - The **salary will be £100k+**, with slight adjustments based on experience level, at the Centre for Neural Circuits and Behaviour.
- **AI Powers Systems-Behavioral Neuroscience**: The fellowship develops an **AI-powered technology** that can infer the contributions of specific brain regions to behavior by analyzing gameplay data, enabling **non-invasive diagnosis and treatment of neurological disorders**.
   - Their approach leverages state-of-the-art deep reinforcement learning models, specifically **MuZero and Dreamer architectures** ([project link](https://encode.pillar.vc/projects/behavioral-neuroscience)).
- **Pillar VC backs AI for Science Fellows**: Pillar VC and ARIA are backing AI fellows to spend one year embedded in top science labs in **ARIA's Opportunity Spaces** across the UK.
   - They seek the next generation of founders, scientists, and leaders building **AI for science** ([fellowship link](https://encode.pillar.vc)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://encode.pillar.vc/projects/behavioral-neuroscience">ARIA Opportunity Space: Scalable Neural Interfaces</a>: no description found</li><li><a href="https://encode.pillar.vc">Encode: AI for Science Fellowship</a>: A fellowship connecting top AI talent with leading science labs in the UK to catalyze translation. Backed by Pillar, powered by ARIA.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1352813469686038599)** (56 messages🔥🔥): 

> `GPU/CUDA learning resources, Warp scheduler significance, Context switching, SIMD vs SIMT execution, Flash attention setup on Windows` 


- **GPU Glossary Sparks CUDA Confusion**: A member learning about GPUs/CUDA from the [Modal GPU glossary](https://modal.com/gpu-glossary) expressed confusion about warp schedulers and context switching, specifically about *the point of context switching if each thread shares the same instruction pointer*.
   - Another member explained using an example of **64 threads** in two groups, showing how the scheduler executes one warp while another waits for data, similar to CPU context switching but without state storage overhead.
- **SIMT Demystified: Data Differentiates Threads**: A member clarified that while threads in a warp share the same instruction, the data differs, enabling **SIMT** (Single Instruction, Multiple Threads) execution where *32 threads can multiply 32 elements in one clock cycle*.
   - They emphasized that a **group of 32 threads** is scheduled at once, and context switching brings in a different group of 32, rather than scheduling individual threads one after another.
- **Flash Attention Frustrations on Windows VM**: A member encountered issues setting up the [flash attention repo](https://github.com/Dao-AILab/flash-attention) locally within a **Windows/Ubuntu VM**, struggling with **nvcc version conflicts** and potential disruption to existing CUDA/Torch/Triton setups.
   - Considering **vast.ai** for development, they sought recommendations on suitable machines for Triton/CUDA work and guidance on choosing a machine to train a BERT model with custom kernels.
- **CUDA Core Confusion Corrected**: A member explained that NVIDIA's marketing term "CUDA cores" actually refers to **FP32 units**, which function similarly to SIMD operations and cannot run independently.
   - Warps from different kernels can be scheduled to the same Streaming Multiprocessor (**SM**) in a finely time-sliced fashion, especially beneficial when threads are waiting for data loads.
- **Streaming Multiprocessor Architecture Deep Dive**: A member clarified that multiple thread blocks can run on one Streaming Multiprocessor (**SM**), which is crucial for block synchronization, allowing the **SM** to have warps ready to run while others await a barrier, referencing [H100 Streaming Multiprocessor](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/H100-Streaming-Multiprocessor-SM.png).
   - They explained that resources like registers and shared memory determine the number of resident thread blocks, and the warp scheduler context switches between warps to keep processing units busy.



**Link mentioned**: <a href="https://modal.com/gpu-glossary">GPU Glossary</a>: A glossary of terms related to GPUs.

  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1353019198237970504)** (1 messages): 

> `Amazon Book Release Date, 5th Edition of Book` 


- **Fifth Edition Release Date Spotted on Amazon**: A member reported seeing a **5th edition** of an unspecified book listed on **Amazon** with a scheduled release date of **February 2026**.
- **Release Date Unconfirmed**: Another member requested confirmation of this release date.


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

bigfoot1144: Any progress so far?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1353400806300450928)** (2 messages): 

> `ROCm, tilelang HIP backend, row-row bank conflict-free swizzle, AMD sponsoring cards` 


- **Seeking ROCm Row-Row Bank Conflict-Free Swizzle Implementation**: A member is seeking **ROCm** experts to help implement a *row-row bank conflict-free swizzle* for the **tilelang HIP backend**.
   - Currently, they only have solutions for **NT layout conflict swizzling**, and are requesting assistance from the community.
- **AMD Card Sponsorship Plea for ROCm Development**: The same member jokingly requested that **AMD** sponsor some cards for development related to **ROCm**.
   - This highlights the resource constraints faced by some developers in the **ROCm** ecosystem.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1352996043939385436)** (2 messages): 

> `Hopper Flops, H100 Clock Rate, H100 SMs, Nvidia Boost Clocks` 


- **H100's Dense FLOPs Revealed**: For **fp16/bf16**, dense flops in **Hopper** = **989 TFLOPS** and the clock rate of **H100** = **1.830 GHz** with number of **SMs = 132**.
   - The **FLOPs / clock / SM** = (**989 x 10^3**) / **1.83 / 132** which is approximately **4096**.
- **Nvidia's Seldom-Mentioned Boost Clock Detailed**: The **H100 SXM** has a boost clock of **1.980 GHz** for normal **SM operation**, but if you use **tensor cores** it drops down to **1.830** or lower depending on power draw/thermals.
   - *There are some rare conditions where you get the full boost clock when running TC ops* but strangely that's not always the case.
- **Official Hopper Boost Clock Document Located**: A document was shared which mentions the different boost clocks ([GTC22 Whitepaper](https://resources.nvidia.com/en-us-data-center-overview/gtc22-whitepaper-hopper)).
   - The different **boost clocks** can be found in table **3**, page **39** of the document.



**Link mentioned**: <a href="https://resources.nvidia.com/en-us-data-center-overview/gtc22-whitepaper-hopper">NVIDIA H100 Tensor Core GPU Architecture Overview</a>: A high-level overview of NVIDIA H100, new H100-based DGX, DGX SuperPOD, and HGX systems, and a H100-based Converged Accelerator. This is followed by a deep dive into the H100 hardware architecture, ef...

  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1353365364423131259)** (10 messages🔥): 

> `Tilelang 2:4 sparsity support, Tilelang v0.1.3 Release, SPGEMM issue` 


- **Tilelang to Support 2:4 Sparsity**: Tilelang plans to support **2:4 sparsity**, leveraging **Cute** as a backend, although the user acknowledges its current uncommonness in **AI workloads**.
   - A user expressed interest in fine-tuning **2:4 sparse LLMs**, noting its success with vision models, but uncertainty about its impact on **LLM accuracy**.
- **Tilelang v0.1.3 lands with Cute Upgrades**: Tilelang released [v0.1.3](https://github.com/tile-ai/tilelang/releases/tag/v0.1.3), featuring enhancements, optimizations, and bug fixes, including **Cute upgrades**.
   - The release includes new kernels and tutorials such as **DeepGEMM**, plus **autotuning** and **kernel caches**, among other new features.
- **Request to add SPGEMM Issue**: A TileLang dev requested that users interested in trying Tilelang for **SPGEMM** should open an issue on GitHub.
   - A user indicated that they would be interested in seeing progress on this if the dev team investigates further.



**Link mentioned**: <a href="https://github.com/tile-ai/tilelang/releases/tag/v0.1.3">Release v0.1.3 · tile-ai/tilelang</a>: What&#39;s Changed[Docker] Add libstdcxx-ng-12 to Dockerfiles for CUDA versions by @LeiWang1999 in #160Add cpu jit with backend ctypes by @xs-keju in #154[Carver] Multi-Threads Compilation for Fast...

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1353147367452577977)** (3 messages): 

> `Parallelized Cholesky, Python + MLX + Metal` 


- **Parallelized Cholesky accelerates with Python + MLX + Metal**: A member shared their contribution to the community: *a super high speed parallelized cholesky in python + MLX + Metal*, along with an [attached python file](https://cdn.discordapp.com/attachments/1285384841730457600/1353266677130989588/cholesky_metal.py?ex=67e3018e&is=67e1b00e&hm=adb5b20c5284632e835d3b99bb32418c4967ba8009acc790d6175e964cd8c8d1&).
   - Another member commented *this is really cool*.
- **MLX gains momentum**: The community sees growing interest in the MLX framework for Metal.
   - MLX seems to be unlocking new possibilities in high-speed computing.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1352720241310044261)** (10 messages🔥): 

> `WheelNext Initiative, CUDA Indexing Blogpost, Container-First Triton Development, GemLite bfloat16 Support` 


- ****WheelNext** Gears Up to Enhance Python Packaging**: The **WheelNext** initiative ([wheelnext.dev](https://wheelnext.dev/)) aims to improve the user experience in the Python packaging ecosystem, focusing on scientific computing and machine/deep learning.
   - A meetup was announced to discuss making shipping python packages with native accelerator code much easier, with details available [on Discord](https://discord.com/events/987824841656791130/1351966194424352789).
- **Dive into CUDA Indexing with New Blogpost**: A member shared a [blog post](https://veitner.bearblog.dev/indexing-in-cuda/) explaining **CUDA indexing** with a 2D block tiling example for matrix multiplication, emphasizing row-major format.
   - The post details how a 2D array `A` with shape `(M, N)` in CUDA is linearized in row-major format, mapping the coordinate `(i,j)` to `i * N + j`.
- **Container-First Approach Streamlines Triton Development**: A member highlighted a new [blog post](https://next.redhat.com/2025/03/20/a-container-first-approach-to-triton-development/) about using containers to simplify and accelerate **Triton kernel development**.
   - The post emphasizes how containerization enhances the **Triton development** workflow by simplifying setup, increasing consistency, and enabling more seamless collaboration.
- ****GemLite** Adds bfloat16 Support for Gemma Models**: **GemLite** now supports **bfloat16** on both Hopper and non-Hopper GPUs, enabling the running of **Gemma models** in vllm via hqq.
   - More details are available in the [associated tweet](https://x.com/mobicham/status/1904185254224535875) and on the [github pull request](https://github.com/mobiusml/gemlite/pull/24).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wheelnext.dev/">WheelNext</a>: no description found</li><li><a href="https://x.com/mobicham/status/1904185254224535875">Tweet from mobicham (@mobicham)</a>: GemLite now supports bfloat16 on both Hopper and non-Hopper gpus 🫡https://github.com/mobiusml/gemlite/pull/24</li><li><a href="https://veitner.bearblog.dev/indexing-in-cuda/">Indexing in CUDA</a>: 
In this blogpost I want to explain what it means for a matrix to be in row major format. 
This is essential to understand CUDA kernels and their methods ...</li><li><a href="https://next.redhat.com/2025/03/20/a-container-first-approach-to-triton-development/">A container-first approach to Triton development</a>: The Triton project from OpenAI is at the forefront of a groundbreaking movement to democratize AI accelerators and GPU kernel programming.&nbsp; It provides a powerful and flexible framework for writi...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1353640859639218178)** (1 messages): 

> `LLM Kernel Understanding, RL for Operation Understanding, Reducing Hallucinations in Kernel Creation` 


- **LLMs Demystify Kernel Code**: The idea is to use **LLMs** to understand **kernel code**, explaining simple concepts and variable states at specific places in tensors.
   - This aims to ensure the LLM grasps the underlying operations.
- **RL Supercharges Kernel Operation Grasp**: Employ **Reinforcement Learning (RL)** to enhance the model's understanding of operations, ensuring a solid grasp.
   - This solid grasp of kernel operations can serve as a prerequisite for creating complex kernels and potentially reducing hallucinations.
- **Kernel Creation Sanity Check with LLMs**: Using LLMs to verify and explain kernel operations could greatly reduce hallucinations during the complex kernel creation process.
   - Such method could be seen as a **sanity check** for complex kernel code and design.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1352873117885927464)** (5 messages): 

> `veRL rollouts with sglang, low precision data types, quantization strategies for RL, ARC-AGI2 announcement` 


- **veRL rolls out sglang support**: **veRL** now supports rollouts with **sglang** as shown in [this paper](https://arxiv.org/abs/2503.16219).
- **Tiny Model Reasoning with GRPO**: A study showed reinforcement learning (**RL**) improving reasoning in small language models (**LLMs**), specifically a **1.5B parameter model** trained on 4 **NVIDIA A40 GPUs** in 24 hours.
   - Adapting the Group Relative Policy Optimization (**GRPO**) algorithm on a curated dataset, the model achieved significant gains, such as AMC23 accuracy rising from **63% to 80%** and AIME24 reaching **46.7%**, with a training cost of only **$42**.
- **ARC-AGI2 frontier benchmark**: A member shared the [ARC-AGI-2 announcement](https://x.com/arcprize/status/1904269307284230593), a frontier AGI benchmark challenging AI reasoning systems.
   - The goal is to achieve **85%** accuracy with ~**$0.42**/task efficiency, contrasting sharply with current performance levels of base LLMs at **0%** and reasoning systems at under **4%**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.16219">Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn&#39;t</a>: Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained setting...</li><li><a href="https://x.com/arcprize/status/1904269307284230593">Tweet from ARC Prize (@arcprize)</a>: Today we are announcing ARC-AGI-2, an unsaturated frontier AGI benchmark that challenges AI reasoning systems (same relative ease for humans).Grand Prize: 85%, ~$0.42/task efficiencyCurrent Performanc...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1353542132799111262)** (5 messages): 

> `CUDA core, CUDA_fp6.hpp, CUDA_fp4.hpp` 


- **CUDA core's fp4 and fp6 use cases requested**: A member inquired about which libraries utilize **fp4** and **fp6** within the **CUDA core**, referencing the presence of `cuda_fp6.hpp` and `cuda_fp4.hpp` header files in version **12.8**.
   - However, they noted difficulty in locating libraries that actively employ these header files.
- **CUDA FP4/FP6 Library Usage**: The user is asking about the usage of **FP4** and **FP6** data types within CUDA cores, specifically if any libraries are utilizing them.
   - They have identified header files (**cuda_fp6.hpp** and **cuda_fp4.hpp**) in CUDA version **12.8**, but haven't found examples of their practical application in existing libraries.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1352752759107227771)** (9 messages🔥): 

> `Submission Guide, Kernel profiling, Conv2D error` 


- **Submission Guide Available**: A member asked for a submission guide and another member shared [a link to the documentation](https://gpu-mode.github.io/discord-cluster-manager/docs/intro) for the GPU kernel leaderboard, which is a competition platform on Discord where users can submit their own kernel implementations.
- **Kernel Profiling Coming Soon!**: A member asked if it was possible to profile their triton kernel via the bot itself.
   - The response was that *we do not currently have that possibility, but it's in store and you (most likely) can expect it for the first problem set launch*.
- **Conv2D Submission Error**: A member reported getting a consistent error when submitting to conv2d involving `subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1` and asked if it meant their **CUDA source** couldn't compile.
   - The member was new to CUDA and C++ and was seeking assistance from the community.



**Link mentioned**: <a href="https://gpu-mode.github.io/discord-cluster-manager/docs/intro">Getting Started | GPU MODE Kernel Leaderboard</a>: Welcome! If you are excited about building GPU kernels, this leaderboard is the place for you! We

  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1352719326683074611)** (119 messages🔥🔥): 

> `matmul benchmarks on H100, grayscale benchmarks on A100, grayscale benchmarks on T4, L4, A100, H100, histogram benchmarks on T4, vectorsum tests on A100` 


- **Modal Runners Ace Matmul Benchmarks on H100**: Numerous `matmul` benchmarks and tests using **Modal runners** on **H100 GPUs** have succeeded, with submission IDs ranging from **2479** to **2487**.
   - These submissions indicate successful execution and integration of Modal runners for matrix multiplication tasks on high-performance GPUs.
- **Grayscale Gauntlet on A100 GPUs**: A multitude of `grayscale` benchmark and leaderboard submissions have succeeded on **A100 GPUs** using **Modal runners**, with submission IDs spanning from **2488** to **2596** and beyond.
   - These consistent successes highlight the reliability and efficiency of Modal runners for image processing tasks on A100 GPUs.
- **Grayscale Greatness Across GPUs**: Leaderboard submissions for `grayscale` using **Modal runners** have succeeded across various GPUs, including **T4**, **L4**, **A100**, and **H100**, with an initial submission ID of **2484**.
   - This demonstrates the versatility of Modal runners in handling image processing tasks on diverse GPU architectures.
- **Histogram Hit on T4 GPUs**: A `histogram` benchmark submission with ID **2765** using **Modal runners** on **T4 GPUs** has succeeded.
   - This indicates successful execution of histogram computation tasks on T4 GPUs utilizing the Modal runners platform.
- **Vector Sum Victory and Conv2d Conquest on A100**: Test submissions for `vectorsum` and `conv2d` have succeeded on **A100 GPUs** using **Modal runners** with IDs **2829** and **2830**.
   - These successful tests highlight the capability of Modal runners in handling vector operations and convolutional tasks on high-performance GPUs.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1353071106201882693)** (2 messages): 

> `CUDA, load_inline(), PyTorch headers, KernelBot` 


- **`load_inline()` Timed Out Due to Excessive PyTorch Headers**: CUDA submissions using `load_inline()` were timing out because about **5K PyTorch headers** were being added, as investigated in [this PR](https://github.com/pytorch/pytorch/pull/149480).
   - A new mode was added to disable implicitly adding headers, and one member managed to get an example compiling from **90s to 15s**, while a colleague got it from **15s to 5s**.
- **KernelBot leaderboard performance improved**: The [KernelBot leaderboard](https://pytorch.org/blog/kernel-compilation/) supports custom CUDA extensions via `load_inline()`, which previously resulted in cold starts of up to **90s**.
   - A member stated that *they always thought it was a cuda problem*, and was happy it could be solved.



**Link mentioned**: <a href="https://github.com/pytorch/pytorch/pull/149480">load_inline no_implicit_headers mode by msaroufim · Pull Request #149480 · pytorch/pytorch</a>: In the kernelBot leaderboard we support people competing with custom cuda extensions via load_inline(), however even on toy kernels this can result in cold starts of up to 90s - this problem is pri...

  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1352766663031914619)** (17 messages🔥): 

> `GPU prices, VRAM requirements for LLMs, RTX Pro 6000, CUDA Capability` 


- **GPU Prices Skyrocket Amid AI Boom**: High-end consumer GPUs are becoming increasingly expensive due to NVIDIA's strategy of limiting high VRAM to those models, but cloud vendors like **vast.ai** and **Nebius** offer cheaper alternatives for running models.
   - One member stated, *"welcome to the ai boom,"* highlighting the impact of AI on GPU pricing and availability.
- **Max out budget on older GPUs, run stuff locally**: For local machine learning, investing in older cards like **3090** or **4090** is suggested for maximizing budget, with *2x3090* potentially outperforming a single newer card, allowing for local distributed training.
   - The assertion was made that older cards provide opportunities to *learn distributed stuff locally*.
- **Nvidia desensitizes users to high prices**: The new **RTX Pro 6000**, with 96GB VRAM, is considered a reasonable option for professionals, normalizing the perception of high GPU costs, although it lacks NVLink.
   - One member noted, *"Actl i think nvidia has successfully desensitized me to their insance prices,"* suggesting an adjustment in expectations due to market trends.
- **GDDR7 memory**: The **RTX Pro 6000** features **96 GB GDDR7** with ECC and **1792 GB/sec** bandwidth, although discrepancies exist in CUDA API versions reported in the [Data Sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf) and [TPU specifications](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4272).
   - The specs report Compute APIs as CUDA 11.6, while TPU claims CUDA 10.1, and the member highlighted that the [CUDA GPUs list](https://developer.nvidia.com/cuda-gpus) has Geforce RTX 50 series with C.C. 10.0 instead of 12.0.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/)** (1 messages): 

rocka2424: This is awesome, looking forward to it!
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1352811157630750883)** (86 messages🔥🔥): 

> `Nvidia Mamba-Transformer Hybrid, Qwen 2.5 Omni Model, DeepSeek V3 Model Update, Reve Image Halfmoon Model, Qwen2.5-VL-32B-Instruct` 


- **Nvidia engineers a **Nemotron-H** Mamba-Transformer hybrid**: Nvidia introduced the **Nemotron-H** family of models, including a series of **8B** and **47-56B** models that are hybrid Mamba-Transformer models, offering improved inference speed compared to other models, [according to their research](https://research.nvidia.com/labs/adlr/nemotronh/).
- **Qwen Debuts **Qwen2.5-Omni**: An End-to-End Streaming Multimodal Model**: Qwen released **Qwen2.5-Omni**, a multimodal model designed to perceive text, images, audio, and video, while generating text and natural speech responses in a streaming manner, [according to HuggingFace](https://github.com/huggingface/transformers/pull/36752/commits/b4ff115375f02b59eb3e495c9dd3c1219e63ff50).
- ****DeepSeek V3** Gets a Quick Update, Still Rocks Leaderboards**: DeepSeek announced a small version upgrade for the **DeepSeek V3** model, with the API interface and usage method remaining unchanged, [according to their HuggingFace page](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324).
- **Reve Image Launches **Halfmoon**: Claims Top Spot in Image Generation**: **Reve Image** launched **Halfmoon**, claiming it's the best image model in the world, with impressive text rendering, prompt adherence, and aesthetics, currently accessible through their website, [according to their announcement](https://x.com/ArtificialAnlys/status/1904188980423467472).
- **Qwen Drops **Qwen2.5-VL-32B-Instruct**: Open Source VL Model with RLHF**: Qwen open-sourced the **Qwen2.5-VL-32B-Instruct** model under the **Apache 2.0 license**, optimized with reinforcement learning, showing significant improvements in human preference and mathematical reasoning, [according to their blog](https://qwenlm.github.io/blog/qwen2.5-vl-32b/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen2.5-vl-32b/">Qwen2.5-VL-32B: Smarter and Lighter</a>: QWEN CHAT GITHUB HUGGING FACE MODELSCOPE DISCORDIntroduction At the end of January this year, we launched the Qwen2.5-VL series of models, which received widespread attention and positive feedback fro...</li><li><a href="https://x.com/arcprize/status/1904269307284230593?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from ARC Prize (@arcprize)</a>: Today we are announcing ARC-AGI-2, an unsaturated frontier AGI benchmark that challenges AI reasoning systems (same relative ease for humans).Grand Prize: 85%, ~$0.42/task efficiencyCurrent Performanc...</li><li><a href="https://x.com/AkemiMadoka/status/1904111806693671123">Tweet from 坂本 (@AkemiMadoka)</a>: @teortaxesTex looks like a small update on v3</li><li><a href="https://x.com/simonw/status/1904187791808123052">Tweet from Simon Willison (@simonw)</a>: Notes on today&#39;s DeepSeek v3 0324 model - a 641 GB MIT licensed monster, but you can run it on a ~$10,000 consumer level 512GB M3 Mac Studio if you use the 352 GB quantized version via MLX https:/...</li><li><a href="https://x.com/reveimage/status/1904211082870456824">Tweet from Reve (@reveimage)</a>: Halfmoon is Reve Image — and it’s the best image model in the world 🥇(🔊)</li><li><a href="https://x.com/TheXeophon/status/1904225899957936314">Tweet from Xeophon (@TheXeophon)</a>: Tested the new DeepSeek V3 on my internal bench and it has a huge jump in all metrics on all tests.It is now the best non-reasoning model, dethroning Sonnet 3.5.Congrats @deepseek_ai!</li><li><a href="https://x.com/Alibaba_Qwen/status/1904227859616641534">Tweet from Qwen (@Alibaba_Qwen)</a>: 72B too big for VLM? 7B not strong enough! Then you should use our 32B model, Qwen2.5-VL-32B-Instruct!Blog: https://qwenlm.github.io/blog/qwen2.5-vl-32b/Qwen Chat: https://chat.qwen.aiHF: https://hugg...</li><li><a href="https://x.com/picocreator/status/1904250680266956903">Tweet from PicoCreator - AI Model Builder 🌉 (@picocreator)</a>: ❗️Attention is NOT all you need ❗️Using only 8 GPU&#39;s (not a cluster), we trained a Qwerky-72B (and 32B), without any transformer attentionWith evals far surpassing GPT 3.5 turbo, and closing in on...</li><li><a href="https://x.com/ArtificialAnlys/status/1904188980423467472">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: The Halfmoon 🌓 reveal: Congratulations to @reveimage on creating the world’s leading image generation model with Reve Image!Reve Image has been in the Artificial Analysis Image Arena over the past we...</li><li><a href="https://x.com/asmeurer/status/1904193931325079991">Tweet from Aaron Meurer (@asmeurer)</a>: @simonw The license update is a big deal. The original V3 was not MIT.</li><li><a href="https://x.com/kimmonismus/status/1903221838022324226">Tweet from Chubby♨️ (@kimmonismus)</a>: Sora abandons credits for all paid tiers, unlimited generations available.This is a good change.</li><li><a href="https://fxtwitter.com/btibor91/status/1903469632167506018">Tweet from Tibor Blaho (@btibor91)</a>: @TheXeophon https://x.com/btibor91/status/1899917834496729259?s=61Quoting Tibor Blaho (@btibor91) @TheXeophon-bench</li><li><a href="https://huggingface.co/collections/nvidia/mambavision-66943871a6b36c9e78b327d3">MambaVision - a nvidia Collection</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found</li><li><a href="https://research.nvidia.com/labs/adlr/nemotronh/">Nemotron-H: A Family of Accurate, Efficient Hybrid Mamba-Transformer Models</a>: Nemotron-H is a series of hybrid Mamba-Transformer models which offer either better or on-par accuracy and improved inference speed (up to 3x) compared to other similarly-sized state-of-the-art open-s...</li><li><a href="https://github.com/huggingface/transformers/pull/36752/commits/b4ff115375f02b59eb3e495c9dd3c1219e63ff50">Add Qwen2.5-Omni by BakerBunker · Pull Request #36752 · huggingface/transformers</a>: What does this PR do?Add Qwen2.5 Omni ModelBefore submitting This PR fixes a typo or improves the docs (you can dismiss the other checks if that&amp;#39;s the case). Did you read the contributor...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1353028110110429295)** (25 messages🔥): 

> `Impact of noisy data in multi-turn SFT, Transformer usage in RL, Community model preferences, Trusting eval benchmarks, Gemini's image generation` 


- **Noise Tolerated in Multi-Turn SFT?**: A member questioned how much noise impacts data quality in multi-turn SFT, especially with complex agent trajectories, suggesting that some noise is tolerable, recovery steps are valuable, and erroneous turns can be masked.
   - They shared that it's difficult to collect perfect trajectories when the complexity and step count increases, *like making a wrong decision about which site to go to for information or which application to use to open a file*.
- **Transformers Slow to Take Over RL?**: A member inquired about the limited use of Transformers in RL policy models, suspecting it's due to compute and memory constraints.
   - They are *having trouble finding many papers where they actually used a small Transformer*.
- **Community Prefers Claude 3.5 for Code?**: A member asked if Interconnects publishes community-preferred model lists, noting their preference for **Claude 3.5** over **Claude 3.7** for code, but the opposite for reasoning.
   - Another member mentioned that Interconnects does not publish model lists, but they hope to add more evals to their [artifacts logs series](https://www.interconnects.ai/t/artifacts-log) when possible.
- **Private Evals > Benchmarks**: Multiple members discussed trusting model eval benchmarks, with one stating *Don’t trust them; have my own eval*, and recommending creating a markdown file with 5-10 prompts that **you** care about.
   - The suggestion was to run prompts with multiple models side by side in tools such as [Chorus](https://chorus.sh/) to *quickly get a feel which model is good for which things*.
- **Gemini's Generator a Mystery?**: A member inquired whether the new **Gemini's** image generation is autoregressive or uses a diffusion head, but its architecture remains unknown.
   - Another member mentioned that labs know which websites to include to boost common benchmarks during training.



**Link mentioned**: <a href="https://www.interconnects.ai/p/building-on-evaluation-quicksand">Building on evaluation quicksand</a>: On the state of evaluation for language models.

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1352747678282420246)** (36 messages🔥): 

> `LLM input/output tokens, o1-pro performance, Mistral 24B is impressive, Claude Compass starter prompts, DAPO and Dr. GRPO` 


- **LLMs Count Input and Output Tokens**: In LLMs, both **input tokens** and **output tokens** are counted during Supervised Fine-Tuning (SFT), clarifying an initial question about token handling.
   - A member confirmed the token counting and humorously remarked that, *"With the cost of those tokens, he could’ve bought the NYT."*
- **o1-pro Dominates Extended NYT Connections Benchmark**: **o1-pro** set a new record on the **Extended NYT Connections benchmark** with a score of **81.7**, surpassing the previous champion, **o1** at **69.7**, as noted in a [tweet](https://x.com/LechMazur/status/1903255938116538376).
   - The benchmark is a more challenging version of the original, with additional words per puzzle.
- **Mistral 24B Impresses Community, Reputation Recovers**: The release of **Mistral 24B** is considered a major highlight, praised for its strength and accessibility of the base model, and the promise of new open releases under **Apache 2.0** is aiding in reputation recovery.
   - One member stated, *"Mistral 24B is probably one of the greatest releases in the last months, incredibly strong model and you have access to the base model as well."
- **Claude Compass Launches Prompts**: A member shared a tweet of **Claude Compass**'s starter prompts which are deep research prompts such as *'Find credible sources for my research'* and *'Analyze great investment pitches'*.
   - It was also noted that another company named [Cohere](https://cohere.com/compass) already has a product named **Compass**.
- **DAPO and Dr. GRPO Papers**: A member is mastering **DAPO** and **Dr. GRPO** for an upcoming blog post, planning to review relevant papers and improve the RLHF book implementation section on tradeoffs.
   - The notes are complete, and the member is considering covering **DAPO** and **Dr. GRPO** together, possibly deferring the rest to a future post.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/neurosp1ke/status/1903564534930907604">Tweet from Andreas Köpf (@neurosp1ke)</a>: added mistral-small-3.1-24b-instruct</li><li><a href="https://x.com/LechMazur/status/1903255938116538376">Tweet from Lech Mazur (@LechMazur)</a>: o1-pro sets a new record on my Extended NYT Connections benchmark with a score of 81.7, easily outperforming the previous champion, o1 (69.7)! This benchmark is a more difficult version of my original...</li><li><a href="https://x.com/btibor91/status/1904206595229130886">Tweet from Tibor Blaho (@btibor91)</a>: New: Claude Compass (deep research) starter prompts- &#34;Find credible sources for my research&#34;- &#34;Provide evidence-based insights for my topic&#34;- &#34;Research topics for my writing&#34;- ...</li><li><a href="https://x.com/LechMazur/status/1903272087441023223">Tweet from Lech Mazur (@LechMazur)</a>: @bradthilton I might benchmark a shorter version of hallucinations, but no chance I&#39;m running other benchmarks.</li><li><a href="https://huggingface.co/spaces/Presidentlin/llm-pricing-calculator">Llm Pricing - a Hugging Face Space by Presidentlin</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1352903024242327582)** (4 messages): 

> `O1-pro vs BoN, O1-pro reasoning paths marginalization, Tech CEOs in Open Source RL` 


- **O1-pro excels in Reasoning Path Merging**: A member suggested that **O1-pro** seems more like merging reasoning paths with correct answers than the simple **BoN** (Bag of Neurons).
   - They noted that the output length from **o1-pro** is usually a lot longer than **o1** but didn't know how to marginalize reasoning paths though.
- **Tech CEOs champion Open Source RL**: Nathan Lambert shared a [post](https://x.com/natolambert/status/1903893527593193639) that stated *major tech company CEOs are arguing for very cutting edge defaults in open-source RL repos*.
   - He concluded that *this timeline is amazing*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fixvx.com/nearcyan/status/1903962841952247833">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/natolambert/status/1903893527593193639">Tweet from Nathan Lambert (@natolambert)</a>: Lol when major tech company CEOs are arguing for very cutting edge defaults in open-source RL repo&#39;s. This timeline is amazing.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1352769466965950566)** (127 messages🔥🔥): 

> `R1-Zero Training, GRPO Bias, LOOP & RLOO, PPO Objective, Creative Writing LLMs` 


- **Row Mean's Length Bias Unmasked in R1-Zero Training**: An analysis reveals that using row mean in **R1-Zero-like training** introduces a bias, favoring shorter correct responses and longer incorrect ones, as detailed in a [paper](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) and accompanying [code](https://github.com/sail-sg/understand-r1-zero).
   - Switching to all mean yields comparable performance without increasing length; leading to questions about plots showing increasing reasoning length correlating with increased capability.
- **GRPO's Length Explosion Problem Plagues Practitioners**: Users observed **length explosion** in their **GRPO runs**, prompting consideration of techniques like length curriculum or clipping, though these are seen as unsatisfactory band-aids.
   - The core issue is garbage responses are being generated when responses are getting longer; this implies a deeper problem beyond length.
- **Prefix Caching for vLLM Causes RL Issues**: Members found that **prefix caching** for **vLLM** may be causing **RL issues** as stated in [this github issue](https://github.com/huggingface/open-r1/issues/491).
   - Specifically, inference was worse than training and identified this caching as the culprit, demonstrating a subtle issue that may be overlooked.
- **LOOP and RLOO Arise from Unbiasing Dr. GRPO**: It was suggested that **Dr. GRPO** still has a bias that is more pronounced the smaller the group size is; to make it unbiased, simply multiply **Dr. GRPO**'s A_i by the correction term **N/N-1**, resulting in **LOOP (Leave-One-Out Proximal Policy Optimization)**, detailed in the [Dr GRPO paper](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf).
   - Removing PPO’s clipping yields **RLOO (Reinforce Leave-One-Out)**.
- **Deviation-Based DPO Diversifies Creative LLM Writing**: A new [paper](https://arxiv.org/abs/2503.17126) explores promoting both output diversity and quality in creative writing LLMs, by including *deviation* in the training objective to facilitate learning from rare high-quality instances.
   - The study adopts this approach to **Direct Preference Optimization (DPO)** and **Odds Ratio Preference Optimization (ORPO)**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.17126">Modifying Large Language Model Post-Training for Diverse Creative Writing</a>: As creative writing tasks do not have singular correct answers, large language models (LLMs) trained to perform these tasks should be able to generate diverse valid outputs. However, LLM post-training...</li><li><a href="https://x.com/zzlccc/status/1903162768083259703">Tweet from Zichen Liu (@zzlccc)</a>: 🪂Understanding R1-Zero-Like Training: A Critical Perspective* DeepSeek-V3-Base already exhibits &#34;Aha moment&#34; before RL-tuning??* The ever-increasing output length in RL-tuning might be due to...</li><li><a href="https://x.com/tassel_pierre/status/1903442097866236383">Tweet from Tassel Pierre (@tassel_pierre)</a>: @elan_marko @zzlccc Yes but you apply the loss per token. So if you have two completions with positive advantage rewards, even if the longer one has a slightly less positive one, because it is applied...</li><li><a href="https://x.com/QGallouedec/status/1903872705184899492">Tweet from Quentin Gallouédec (@QGallouedec)</a>: Thanks @ethayarajh @zzlccc!I have exactly the same question as @ethayarajh. In trl, we don&#39;t do (anymore) this par per-sequence normalization that leads to response level length bias. Instead, we ...</li><li><a href="https://x.com/WenhuChen/status/1903464313391624668">Tweet from Wenhu Chen (@WenhuChen)</a>: This paper provides some really interesting insights:1. Previously, people found that Qwen base models are particularly good at R1 training to show strong exploration skills. - This paper shows that t...</li><li><a href="https://x.com/leloykun/status/1903382502158500119">Tweet from leloy! (@leloykun)</a>: I&#39;m not sure if someone has already pointed this out, but Dr. GRPO still has a bias that is more pronounced the smaller the group size is.To make it unbiased, simply multiply Dr. GRPO&#39;s A_i by...</li><li><a href="https://ai.stackexchange.com/questions/37958/where-does-the-proximal-policy-optimization-objectives-ratio-term-come-from">Where does the proximal policy optimization objective&#x27;s ratio term come from?</a>: I will use the notation used in the proximal policy optimization paper.&#xA;&#xA;What approximation is needed to arrive at the surrogate objective (equation (6) above) with the ratio $r_t(\theta)$?&#x...</li><li><a href="https://ai.stackexchange.com/questions/37958/where-does-the-proximal-policy-opt">Where does the proximal policy optimization objective&#x27;s ratio term come from?</a>: I will use the notation used in the proximal policy optimization paper.&#xA;&#xA;What approximation is needed to arrive at the surrogate objective (equation (6) above) with the ratio $r_t(\theta)$?&#x...</li><li><a href="https://github.com/huggingface/open-r1/issues/491">Prefix Caching should be turned off for GRPO · Issue #491 · huggingface/open-r1</a>: The performance of my runs during inference was way worse than the performance during training. After debugging, I think prefix caching is the culprit behind this. Since the model is constantly bei...</li><li><a href="https://github.com/sail-sg/oat/blob/7619b79a8804e813419faeda22bdd35cc4d9b9bd/oat/algorithms/ppo.py#L231">oat/oat/algorithms/ppo.py at 7619b79a8804e813419faeda22bdd35cc4d9b9bd · sail-sg/oat</a>: 🌾 OAT: A research-friendly framework for LLM online alignment, including preference learning, reinforcement learning, etc. - sail-sg/oat</li><li><a href="https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/grpo_trainer.py#L965">trl/trl/trainer/grpo_trainer.py at 07cfe1677e552b7d5c92b7740e5b2f0b057661d8 · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/ppo_trainer.py#L573C1-L574C1">trl/trl/trainer/ppo_trainer.py at 07cfe1677e552b7d5c92b7740e5b2f0b057661d8 · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com/sail-sg/oat/blob/7619b79a8804e813419faeda22bdd35cc4d9b9bd/oat/algorithms/ppo.py#L560">oat/oat/algorithms/ppo.py at 7619b79a8804e813419faeda22bdd35cc4d9b9bd · sail-sg/oat</a>: 🌾 OAT: A research-friendly framework for LLM online alignment, including preference learning, reinforcement learning, etc. - sail-sg/oat
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1352749970393927773)** (6 messages): 

> `Operator agent limitations, Infinibranch Browsers as a solution, Intelligent Browser Automation` 


- **Operator Agents Lack Managerial Skills**: Members discussed the limitations of **Operator agents**, noting they struggle with complex tasks requiring coordination, such as extracting information from datasets and one person commented on needing *a manager agent that tells 1 operator agent to get the details for 1 dataset*.
   - One member expressed frustration with the limited success rate, achieving only **4** out of **10** tasks with the operator and **6** with deep research.
- **Infinibranch Browsers Reach 80% Success**: A possible solution using [Morph Cloud's Infinibranch Browser](https://x.com/morph_labs/status/1902566171641266500) was suggested to help scale browser-use agents, improving the success rate to approximately **80%** on tasks like finding Amazon links for a list of books.
   - The original poster on X, Andrew Carr, needed to extract links from **1000+ books** to a Google sheet which Operator was unable to hack.
- **Morph Cloud Scales Autonomous Browser Workflows**: [Morph Cloud](https://morph.so/blog/browser-morph-cloud/) allows users to snapshot and branch complete browser states, including authentication and cookies, making it easier to scale autonomous browser workflows across multiple parallel instances.
   - The blogpost further explains how traditional web scraping methods have become obsolete because of JavaScript-heavy single page applications, Dynamic loading and infinite scroll, complex user interactions required to access data, CAPTCHAs and sophisticated bot detection, multi-step workflows that require understanding context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/andrew_n_carr/status/1901354501317288304">Tweet from Andrew Carr (e/🤸) (@andrew_n_carr)</a>: I have a very specific agentic use case that is just hard enough that web scraping doesn&#39;t work. 1. I have a list of 1000+ books2. I want to find their amazon links3. I would like those saved in a...</li><li><a href="https://x.com/morph_labs/status/1902566171641266500">Tweet from Morph (@morph_labs)</a>: Announcing Infinibranch BrowsersMorph Cloud&#39;s Infinibranch Browser scales the browser-use agent into a ~80% success rate on the list of books belowOperator doesn&#39;t get past 10%Quoting Andrew C...</li><li><a href="https://morph.so/blog/browser-morph-cloud/">Remote Browsers with Morph Cloud: Infinitely Scalable Browser Automation</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1352726958584365116)** (16 messages🔥): 

> `R1-Zero-Like Training, DeepSeek-V3-Base, GRPO Bias in RL-tuning, CoT Philosophy, Math errors in AI papers` 


- **R1-Zero Training: New Insights Emerge**: A [Twitter thread](https://x.com/zzlccc/status/1903162768083259703) highlights key observations about **R1-Zero-like training**, suggesting **DeepSeek-V3-Base** shows an *'Aha moment'* before RL-tuning.
   - The researchers point to a potential **bias in GRPO** contributing to ever-increasing output length, detailing findings in a [paper](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) and providing [code](https://github.com/sail-sg/understand-r1-zero).
- **GRPO Loss Implementation Analysis**: Multiple papers this week discuss the **1/o term** and its impact on longer examples, suggesting that the loss penalizes long, repetitive behaviors less while not rewarding long, exploratory generations sufficiently.
   - They note that per-question normalization punishes hard questions within a batch.
- **Chain of Thought and Reasoning**: A member questioned if advancements are truly about reasoning or if they leverage tokens to overcome inefficiencies in task-specific next-token completion/search.
   - Another suggested the viability of Chain of Thought as a form of language model reasoning, describing reasoning as very broad.
- **Mathematical Concerns about paper calculations**: There was discussion in an AI2 Slack channel suggesting potential errors or anomalies in the math presented in the paper.
   - Some members expressed confusion regarding the paper's argument about length normalization bias, with further discussion occurring in a linked channel with a member providing an explanation.



**Link mentioned**: <a href="https://x.com/zzlccc/status/1903162768083259703?s=61">Tweet from Zichen Liu (@zzlccc)</a>: 🪂Understanding R1-Zero-Like Training: A Critical Perspective* DeepSeek-V3-Base already exhibits &#34;Aha moment&#34; before RL-tuning??* The ever-increasing output length in RL-tuning might be due to...

  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1353807028715393156)** (2 messages): 

> `Claude PR, Header Copy Links` 


- **Claude Sends Pull Request for Header Copy Links**: A member shared a [pull request](https://github.com/natolambert/rlhf-book/pull/82) made by **Claude** for adding header copy links to a GitHub repository.
- **Header Copy Links Amaze**: Members found the header copy links that appear on hover to be interesting and useful.
   - They attached a [screenshot](https://cdn.discordapp.com/attachments/1223784028428177510/1353807029223030835/Screenshot_2025-03-24_at_12.05.34_PM.png?ex=67e2fe8c&is=67e1ad0c&hm=41d19137d3231c38197bef45a02356a9b88f754b907ba8a3f1028543cb17349e&) of the links, noting that they *worked immediately with claude code*.



**Link mentioned**: <a href="https://github.com/natolambert/rlhf-book/pull/82">(experimental) Add heading anchor links for easy section linking by natolambert · Pull Request #82 · natolambert/rlhf-book</a>: Add copyable links to all headings that appear on hoverLinks copy the current URL with fragment identifier to clipboardAdd CSS for styling the anchor linksUpdate Makefile to copy new JS file to ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1353105005778833449)** (9 messages🔥): 

> `China's Open Source AI Blitz, DeepSeek's Impact, US vs China AI Competition, Chinese consumer market for software, China commoditizing hardware` 


- **China Plans Open-Source AI Blitz**: According to [this tweet](https://x.com/balajis/status/1903469483739730132), China aims to flood the market with open-source AI models to **commoditize AI software** and boost its hardware sales.
   - The strategy is to copy, optimize, scale, and undercut Western tech, similar to their approach with manufacturing, with **DeepSeek** being a key player.
- **DeepSeek Triggers Tech Market Tumble**: The release of **DeepSeek** models temporarily knocked ~$1T off US tech market caps, highlighting the potential impact of Chinese AI on global markets, per [this tweet](https://x.com/balajis/status/1903469483739730132).
   - The founder of DeepSeek (**Liang Wengfeng**) has met with top Chinese officials, indicating significant state support and access to *unlimited resources*.
- **China's AI Competition**: A member stated that China's push in open-source AI is driven by intense domestic competition, aiming to accelerate progress rather than *bring down US tech*.
   - They added that most top Chinese labs realize open source is the best way to drive progress because *your close source model will be irrelevant in 3-6 mths or so, might as well accelerate*.
- **Revenue from Ads and Digital Services Lower in China than US**: A member pointed out that Chinese companies aren't trying to destroy American value as a goal.
   - The revenue market for ads and digital services isnt the same as in the US with *much less revenue in ads and digital services in china than US* and for this reason open sourcing is more fine, as well.
- **Chinese Consumers Reluctant to Pay for Software**: Chinese consumers generally avoid paying for software and services, with students and professionals being the primary payers.
   - The consumer market is largely dominated by **ByteDance** and previously by **Kimi**.



**Link mentioned**: <a href="https://x.com/balajis/status/1903469483739730132">Tweet from Balaji (@balajis)</a>: AI OVERPRODUCTIONChina seeks to commoditize their complements. So, over the following months, I expect a complete blitz of Chinese open-source AI models for everything from computer vision to robotics...

  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1353032783139704883)** (17 messages🔥): 

> `Grok DeeperSearch, OpenAI Deep Research, Twitter Premium, HF model comparisons` 


- **Grok DeeperSearch Approaches OpenAI Deep Research**: The new **Grok DeeperSearch** is reportedly *"really good"* and close to **OpenAI Deep Research** in quality, which is impressive considering the short timeframe.
   - The initial **Grok DeepSearch** was considered *"awful"* due to hallucinating content from retrieved links, making it the worst implementation, according to some users.
- **Twitter Premium Grants Access to Grok DeeperSearch**: Access to **Grok DeeperSearch** is available with **Twitter Premium** (the $10 tier), exclusively on the Grok Website.
   - After tweeting about the poor performance of **Grok DeepSearch**, an individual from xAI contacted one user, leading to improvements in **DeeperSearch** based on provided chats and benchmarks.
- **Benchmarking Deep(Re)search Implementations**: One user maintains a markdown file with a set of questions to test search and research implementations, including **Grok DeeperSearch**.
   - The benchmark includes a broad shopping query, a specific shopping query, a generic paper search prompt, and a table/benchmark comparison between two models from **Hugging Face**.
- **Image Generation Benchmarking**: A user shared their image generation benchmark, including prompts such as *"A woman sitting at a poker table with cards in her hands"* and *"Isometric pixel art of a waterfall"*.
   - These benchmarks help in comparing the performance of different models and would assist future posts.



**Link mentioned**: <a href="https://fxtwitter.com/btibor91/status/1899917834496729259">Tweet from Tibor Blaho (@btibor91)</a>: @TheXeophon-bench

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1352724292319576125)** (89 messages🔥🔥): 

> `Gemini Updates, Claude Code New Features, Model Context Protocol (MCP), AI Agents and Email, RF-DETR Object Detection Model` 


- **Gemini Updates Deconstructed**: Gemini's Dave Citron joined @OfficialLoganK on the Release Notes podcast to discuss recent updates, including **personalization**, **Canvas**, **Audio Overviews**, and **Deep Research**.
   - The discussion covered topics from recent app launches to the future of personalization in the **Gemini app**, including insights into user data and privacy considerations.
- **Claude Code Gets Eight New Features**: Anthropic launched **eight** new features for **Claude Code** to help developers build faster and smarter, documented on their [engineering blog](https://www.anthropic.com/engineering/claude-think-tool).
   - Features include a new "think" tool, leading to discussion on its implementation and value, with some likening it to Chain of Thought prompting.
- **A16Z's MCP Ecosystem Deep Dive**: A16Z published a deep dive into **Model Context Protocol (MCP)**, exploring its potential as a standard interface for execution, data fetching, and tool calling in AI models as APIs are the internet's first great unifier.
   - The post examines the use cases of MCP, the challenges, and how it changes the way AI interacts with tools, noting that APIs were the internet’s first great unifier, but AI models lack an equivalent.
- **Roboflow Unleashes RF-DETR for Real-Time Object Detection**: Roboflow announced **RF-DETR**, a fully open-source real-time object detection model under the Apache 2.0 license available on [GitHub](https://github.com/roboflow/rf-detr).
   - RF-DETR achieves **SOTA** performance with over **60 mAP** on **COCO**, with base and large models at **29M** and **128M** parameters respectively.
- **Browser Use Bags $17M to Build Web for Agents**: Browser Use raised **$17 million** to advance web agents, led by Felicis Ventures, aiming to take web agents to the next level after an initial prototype was built in just **four days** and launched on Hacker News.
   - The company is hiring top engineers to build the internet for LLMs, promising a challenging environment with a pure software geekery team culture.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pashpops/status/1902814965595246855?s=46">Tweet from Pasha Rayan (@Pashpops)</a>: Introducing A1Mail - Email for AI Agents! 📬🤖TLDR: With A1Mail you can create an email address then send and receive mail from that address for your AI Agent - without paying $12 per month per Gmail ...</li><li><a href="https://x.com/stuffyokodraws/status/1902757447984710076">Tweet from Yoko (@stuffyokodraws)</a>: [New post] 🔥A Deep Dive Into MCP and the Future of AI Tooling  APIs were the internet’s first great unifier but AI models lack an equivalent.  What are the use cases of MCPs today? Where are the chal...</li><li><a href="https://x.com/GeminiApp/status/1902752852843331650">Tweet from Google Gemini App (@GeminiApp)</a>: In the latest episode of Release Notes, Gemini&#39;s Dave Citron joins @OfficialLoganK to deep dive into some of the latest Gemini updates.🎙️ Learn more about Gemini with personalization, Canvas, Aud...</li><li><a href="https://x.com/kimmonismus/status/1903221838022324226?s=46">Tweet from Chubby♨️ (@kimmonismus)</a>: Sora abandons credits for all paid tiers, unlimited generations available.This is a good change.</li><li><a href="https://x.com/karpathy/status/1903671737780498883">Tweet from Andrej Karpathy (@karpathy)</a>: I just vibe coded a whole iOS app in Swift (without having programmed in Swift before, though I learned some in the process) and now ~1 hour later it&#39;s actually running on my physical phone. It wa...</li><li><a href="https://x.com/theomediaai/status/1903448834451111988?s=61">Tweet from Theoretically Media (@TheoMediaAI)</a>: Sora goes “unlimited” but, watermarked/720/slow boat for the $20 tier. You know what would get me back on the Pro ($200) plan? Release the “Big Daddy” Sora model for Pro users. Keep the Nerf’d version...</li><li><a href="https://fxtwitter.com/karpathy/status/1903671737780498883)">Tweet from Andrej Karpathy (@karpathy)</a>: I just vibe coded a whole iOS app in Swift (without having programmed in Swift before, though I learned some in the process) and now ~1 hour later it&#39;s actually running on my physical phone. It wa...</li><li><a href="https://x.com/karpathy/status/1903671737780498883>)">Tweet from Andrej Karpathy (@karpathy)</a>: I just vibe coded a whole iOS app in Swift (without having programmed in Swift before, though I learned some in the process) and now ~1 hour later it&#39;s actually running on my physical phone. It wa...</li><li><a href="https://x.com/_catwu/status/1903130881205977320">Tweet from cat (@_catwu)</a>: It’s been a big week for Claude Code.We launched 8 exciting new features to help devs build faster and smarter.Here&#39;s a roundup of everything we released:</li><li><a href="https://x.com/AnthropicAI/status/1903128670081888756">Tweet from Anthropic (@AnthropicAI)</a>: We’re launching a new blog: Engineering at Anthropic.A hub where developers can find practical advice and our latest discoveries on how to get the most from Claude.</li><li><a href="https://x.com/leloykun/status/1903186153513291933">Tweet from leloy! (@leloykun)</a>: There are actually two kinds of inference-time compute:1. The thinking that happens before generating answer tokens. Think of this as the &#34;drafting&#34; or &#34;planning&#34; stage. And2. The thin...</li><li><a href="https://fxtwitter.com/gergelyorosz/status/1904089127600975966)">Tweet from Gergely Orosz (@GergelyOrosz)</a>: Right now, many AI coding tooling startups are heavily subsidizing actual cost of running AI agents.None will be able to do it indefinitely.But those that start to charge closer actual costs on their ...</li><li><a href="https://x.com/gergelyorosz/status/1904089127600975966>)">Tweet from Gergely Orosz (@GergelyOrosz)</a>: Right now, many AI coding tooling startups are heavily subsidizing actual cost of running AI agents.None will be able to do it indefinitely.But those that start to charge closer actual costs on their ...</li><li><a href="https://x.com/tokumin/status/1902251588925915429?s=46">Tweet from Simon (@tokumin)</a>: 🛳️Rolling out interactive Mindmaps in NotebookLM! I&#39;m so inspired by the Exploratorium here in SF - What if every notebook generated your own personal set of interactive understanding toys that h...</li><li><a href="https://x.com/kalomaze/status/1903366221333958999?s=61">Tweet from kalomaze (@kalomaze)</a>: @metalure hybrid mamba, 56b. ~20T tokens (!!!)fp8 pretrain. actual depth (64 layers, ~15% have attention, rest are mamba).distilled (not SFT, actual pretrain distillation!) 47b variant. for ~60bil tok...</li><li><a href="https://fxtwitter.com/taesung/status/1904220824435032528)">Tweet from Taesung Park (@Taesung)</a>: Excited to come out of stealth at @reveimage!Today&#39;s text-to-image/video models, in contrast to LLMs, lack logic. Images seem plausible initially but fall apart under scrutiny: painting techniques...</li><li><a href="https://x.com/taesung/status/1904220824435032528>)">Tweet from Taesung Park (@Taesung)</a>: Excited to come out of stealth at @reveimage!Today&#39;s text-to-image/video models, in contrast to LLMs, lack logic. Images seem plausible initially but fall apart under scrutiny: painting techniques...</li><li><a href="https://fxtwitter.com/karpathy/status/1886192184808149383)">Tweet from Andrej Karpathy (@karpathy)</a>: There&#39;s a new kind of coding I call &#34;vibe coding&#34;, where you fully give in to the vibes, embrace exponentials, and forget that the code even exists. It&#39;s possible because the LLMs (e.g...</li><li><a href="https://x.com/karpathy/status/1886192184808149383>)">Tweet from Andrej Karpathy (@karpathy)</a>: There&#39;s a new kind of coding I call &#34;vibe coding&#34;, where you fully give in to the vibes, embrace exponentials, and forget that the code even exists. It&#39;s possible because the LLMs (e.g...</li><li><a href="https://fxtwitter.com/TransluceAI/status/1904226873879806390)">Tweet from Transluce (@TransluceAI)</a>: To interpret AI benchmarks, we need to look at the data.Top-level numbers don&#39;t mean what you think: there may be broken tasks, unexpected behaviors, or near-misses.We&#39;re introducing Docent to...</li><li><a href="https://x.com/TransluceAI/status/1904226873879806390>)">Tweet from Transluce (@TransluceAI)</a>: To interpret AI benchmarks, we need to look at the data.Top-level numbers don&#39;t mean what you think: there may be broken tasks, unexpected behaviors, or near-misses.We&#39;re introducing Docent to...</li><li><a href="https://x.com/roboflow/status/1902810257652351228?s=46">Tweet from Roboflow (@roboflow)</a>: Excited to announce RF-DETR, the current SOTA for real-time object detection, fully open source and Apache 2.0 for the community.More to come but the repo and Colab notebook are available today for yo...</li><li><a href="https://fxtwitter.com/gregpr07/status/1903835252382224795)">Tweet from Gregor Zunic (@gregpr07)</a>: We Raised $17M to Build the Future of Web for Agents 🤖A few months ago, Browser Use was just an idea weekend experiment to see if LLMs could navigate the web like humans. In just four days, we built ...</li><li><a href="https://x.com/gregpr07/status/1901686296902615122>>>">Tweet from Gregor Zunic (@gregpr07)</a>: Browser Use is Hiring the top 0.01% Founding Engineer to build internet for LLMs🔥We (2 people) have built the leading repository for web agents—45K+ GitHub stars in just 4 months. Every day, someone ...</li><li><a href="https://x.com/gregpr07/status/1903835252382224795>)">Tweet from Gregor Zunic (@gregpr07)</a>: We Raised $17M to Build the Future of Web for Agents 🤖A few months ago, Browser Use was just an idea weekend experiment to see if LLMs could navigate the web like humans. In just four days, we built ...</li><li><a href="https://a16z.com/a-deep-dive-into-mcp-and-the-future-of-ai-tooling/">A Deep Dive Into MCP and the Future of AI Tooling | Andreessen Horowitz</a>: We explore what MCP is, how it changes the way AI interacts with tools, what developers are already building, and the challenges that still need solving.&nbsp;</li><li><a href="https://x.com/ctnzr/status/1903228434232512878?s=61">Tweet from Bryan Catanzaro (@ctnzr)</a>: Nemotron-H: A family of Hybrid Mamba-Transformer LLMs.* Hybrid architecture means up to 3X faster at the same accuracy* Trained in FP8* Great for VLMs* Weights and instruct versions to come soon.https...</li><li><a href="https://hamel.dev/blog/posts/field-guide/">A Field Guide to Rapidly Improving AI Products – Hamel’s Blog</a>: Evaluation methods, data-driven improvement, and experimentation techniques from 30+ production implementations.</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found</li><li><a href="https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/agents_as_tools.py">openai-agents-python/examples/agent_patterns/agents_as_tools.py at main · openai/openai-agents-python</a>: A lightweight, powerful framework for multi-agent workflows - openai/openai-agents-python</li><li><a href="https://x.com/WHinthorn/status/1903511723082232203">Tweet from WFH (@WHinthorn)</a>: Fun fact, this is how we got Claude to be a great prompt engineer where regular meta-prompting failed.https://github.com/hinthornw/promptimizer/blob/31a78b28123530571a8a098b020f5a7a5cfbc2ca/src/prompt...</li><li><a href="https://github.com/hinthornw/promptimizer/blob/31a78b28123530571a8a098b020f5a7a5cfbc2ca/src/promptim/optimizers/metaprompt.py#L238">promptimizer/src/promptim/optimizers/metaprompt.py at 31a78b28123530571a8a098b020f5a7a5cfbc2ca · hinthornw/promptimizer</a>: Prompt optimization scratch. Contribute to hinthornw/promptimizer development by creating an account on GitHub.</li><li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5188231">no title found</a>: no description found</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-vl-32b/">Qwen2.5-VL-32B: Smarter and Lighter</a>: QWEN CHAT GITHUB HUGGING FACE MODELSCOPE DISCORDIntroduction At the end of January this year, we launched the Qwen2.5-VL series of models, which received widespread attention and positive feedback fro...</li><li><a href="https://chat.qwenlm.ai)">no title found</a>: no description found</li><li><a href="https://modelscope.cn/collections/Qwen25-VL-58fbb5d31f1d47)">魔搭社区</a>: no description found</li><li><a href="https://www.oneusefulthing.org/p/the-cybernetic-teammate">The Cybernetic Teammate</a>: Having an AI on your team can increase performance, provide expertise, and improve your experience</li><li><a href="https://www.hbs.edu/ris/Publication%20Files/24-013_d9b45b68-9e74-42d6-a1c6-c72fb70c7282.pdf)">Publications - Faculty &amp; Research - Harvard Business School</a>: no description found</li><li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162111)">Page Cannot be Found</a>: no description found</li><li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4945566).">Page Cannot be Found</a>: no description found</li><li><a href="https://x.com/_mchenco/status/1903520306305827051?s=46">Tweet from michelle (@_mchenco)</a>: cloudflare&#39;s first innovation week of 2025 just wrapped up, and i wasn&#39;t joking when i said every product in the future will be powered by Workers AI.we don&#39;t think of AI solely as a verti...</li><li><a href="https://blog.cloudflare.com/how-cloudflare-is-using-automation-to-tackle-phishing/">How Cloudflare is using automation to tackle phishing head on</a>: How Cloudflare is using threat intelligence and our Developer Platform products to automate phishing abuse reports.</li><li><a href="https://blog.cloudflare.com/ai-labyrinth/">Trapping misbehaving bots in an AI Labyrinth</a>: How Cloudflare uses generative AI to slow down, confuse, and waste the resources of AI Crawlers and other bots that don’t respect “no crawl” directives.</li><li><a href="https://blog.cloudflare.com/take-control-of-public-ai-application-security-with-cloudflare-firewall-for-ai/">Take control of public AI application security with Cloudflare's Firewall for AI</a>: Firewall for AI discovers and protects your public LLM-powered applications, and is seamlessly integrated with Cloudflare WAF. Join the beta now and take control of your generative AI security.</li><li><a href="https://blog.cloudflare.com/cloudflare-for-ai-supporting-ai-adoption-at-scale-with-a-security-first-approach/">Cloudflare for AI: supporting AI adoption at scale with a security-first approach</a>: With Cloudflare for AI, developers, security teams and content creators can leverage Cloudflare’s network and portfolio of tools to secure, observe and make AI applications resilient and safe to use.</li><li><a href="https://blog.cloudflare.com/introducing-ai-agent/">Introducing Cloudy, Cloudflare’s AI agent for simplifying complex configurations</a>: Cloudflare’s first AI agent, Cloudy, helps make complicated configurations easy to understand for Cloudflare administrators.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1353504654897446993)** (2 messages): 

> `Rishi Agarwal on Distillation, Swyx's Agent Engineering Talk, Agent Engineering Elements, Agents as ChatGPT's Growth Path` 


- **Agarwal Surveys Distillation Techniques**: Deepmind's **Rishi Agarwal** released a short [podcast](https://youtu.be/O1AR4iL30mg) surveying **distillation** techniques in machine learning.
- **Swyx Launches into Agent Engineering**: **Swyx** launched a [new talk and essay](https://x.com/swyx/status/1904256213661192405) on **Agent Engineering**.
   - The talk was also featured live on the [@latentspacepod](https://latent.space/p/agent), highlighting the reasons for going all in on Agents at @aiDotEngineer.
- **Six Agent Engineering Elements Unveiled**: The discussion defines **Agents** (thanks to @simonw) and elaborates on the **Six Elements of Agent Engineering**.
   - It also examines how **Agents** could be **ChatGPT's** route to reaching **1 billion monthly active users (MAU)**.



**Link mentioned**: <a href="https://x.com/swyx/status/1904256213661192405">Tweet from swyx 🌉 (@swyx)</a>: 🆕 talk + essay: Agent Engineeringhttps://latent.space/p/agentWhy we went all in on Agents @aiDotEngineerDefining Agents (thanks to @simonw)The Six Elements of Agent EngineeringWhy Agents are ChatGPT&...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1352733334760984606)** (226 messages🔥🔥): 

> `DORA report, Gemini API, AI code generation, Agile adoption, Ruby on Rails` 


- **Google Cloud's DORA Report Explores Engineering Excellence**: The [DORA report](https://dora.dev/research/2024/dora-report/) by Google Cloud delves into metrics for **engineering excellence**, though accessing the full report requires signup.
   - Some found the focus on "*engineering excellence*" to be overly corporate, contrasting it with the "*yolo vibe code*" often used in prototyping.
- **Discord Mobile App to Show Video Ads**: Discord's mobile app will introduce **video ads** starting in June, offering advertisers opportunities to showcase trailers and premium content as reported by [ArsTechnica](https://arstechnica.com/gadgets/2025/03/discord-heightens-ad-focus-by-introducing-video-ads-to-mobile-apps-in-june/).
   - Users expressed concerns about Discord "*enshittifying*" in preparation for an IPO, drawing parallels to the platform X.
- **Gemini API is a Cheap Loss Leader**: Members are finding the **Gemini API** to be a very cheap API, with one user "*sonnet maxxing right now*," and another calls it a "*loss leader.*"
   - There are concerns raised about potential "*model lockin*" risks associated with relying on one AI provider and cultural differences between companies.
- **AI Code Generation Replacing Manual Coding**: A member mentioned AI is writing **80-90%** of their company's code, and another admits that AI writes **99%** of their code these days, resulting in robots doing all the work.
   - Others mentioned their hate for "*template repos*" and that AI is much better at reinventing the wheel for itself.
- **Vibe Manifesto Released**: The [Vibe Manifesto](https://vibemanifesto.org/) values flow, iteration, augmentation, product thinking, rerolling, and human taste.
   - These values contrast with friction, perfection, automation, code crafting, debugging, and technical constraints, respectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vibemanifesto.org/">Vibe Coding Manifesto</a>: A philosophy for a new generation of developers</li><li><a href="https://dora.dev/research/2024/dora-report/">DORA | Accelerate State of DevOps Report 2024</a>: DORA is a long running research program that seeks to understand the capabilities that drive software delivery and operations performance. DORA helps teams apply those capabilities, leading to better ...</li><li><a href="https://tenor.com/view/putting-on-my-sunglasses-ken-ryan-gosling-barbie-movie-shades-on-gif-812066675624542171">Putting On My Sunglasses Ken GIF - Putting on my sunglasses Ken Ryan gosling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ZachBeta/ruby_ai_llm_bot_for_good_discord">GitHub - ZachBeta/ruby_ai_llm_bot_for_good_discord</a>: Contribute to ZachBeta/ruby_ai_llm_bot_for_good_discord development by creating an account on GitHub.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://arstechnica.com/gadgets/2025/03/discord-heightens-ad-focus-by-introducing-video-ads-to-mobile-apps-in-june/">Discord heightens ad focus by introducing video ads to mobile apps in June</a>: Discord looks for more ways to make money ahead of expected IPO.</li><li><a href="https://github.com/ZachBeta/threejs_fpv">GitHub - ZachBeta/threejs_fpv</a>: Contribute to ZachBeta/threejs_fpv development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1353176885789724783)** (1 messages): 

> `Mobile Study Participants, AI Model Updates` 


- **Mobile Study Participants Needed**: The team is still seeking participants for a study focused on mobile use cases and ideas.
   - Interested individuals are encouraged to join and share their insights to help the team learn more.
- **AI Model Updates Coming Soon**: The team announced upcoming updates to their AI models.
   - More details will be shared in the coming days regarding specific improvements and new features.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1352725684115734640)** (52 messages🔥): 

> `Mindmaps in NotebookLM, Research with NotebookLM, HR policies Hub in NotebookLM, NotebookLM for literature search, External Users Share NotebookLM` 


- **Mindmaps verschijnen geleidelijk in NotebookLM**: Een gebruiker merkte op dat hij geen mindmaps had in NotebookLM, waarop een andere gebruiker antwoordde dat hij ze wel had in de gratis versie en dat de functie geleidelijk wordt uitgerold.
   - Niet iedereen zit op dezelfde server, dus het duurt even voordat alle servers zijn bijgewerkt.
- **NotebookLM: Onderzoek om uitgebreide rapporten te bouwen**: Een gebruiker vertelde dat hij NotebookLM gebruikt om onderzoek te doen en uitgebreide rapporten te maken om lokaal en soms regionaal nieuws te genereren, om mensen te helpen situaties te begrijpen.
   - De gebruiker deelde ook een link naar een podcast-episode over een 911-telefoongrap en de juridische gevolgen [911 Prank Call: The Felony Consequences](https://creators.spotify.com/pod/show/peezyproductions/episodes/911-Prank-Call-The-Felony-Consequences-e30gfec).
- **NotebookLM: Hub voor HR-beleid**: Een gebruiker vroeg of iemand NotebookLM gebruikt als een hub voor HR-beleid, personeelshandboeken en onboarding van nieuwe medewerkers, zodat ze vragen kunnen stellen en de juiste antwoorden kunnen krijgen.
   - Hij had het geprobeerd, maar de antwoorden waren niet altijd correct en hij vroeg zich af of er een manier was om de informatie op een bepaalde manier te organiseren.
- **NotebookLM: Literatuuronderzoek**: Een gebruiker vroeg hoe NotebookLM gebruikt kan worden voor literatuuronderzoek, waarop een andere gebruiker antwoordde dat NotebookLM geen ingebouwde zoekfunctie heeft.
   - Desondanks blijft het erg handig voor het leren van onderwerpen op de universiteit.
- **NotebookLM: contract analyse**: Een gebruiker heeft 3 contracten van één pagina met handgeschreven cijfers/bedragen.
   - Eén ervan werd in eerste instantie helemaal niet vermeld. Een andere werd vermeld met ofwel EUR 700 of EUR 760. Eigenlijk is het EUR 400.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://creators.spotify.com/pod/show/peezyproductions/episodes/911-Prank-Call-The-Felony-Consequences-e30gfec">🚓 911 Prank Call: The Felony Consequences by Neural Network News</a>: 11-year-old girl named Ava in Volusia County, Florida, falsely reported a kidnapping via 911 texts. Deputies traced the messages to her residence in Port Orange, revealing the hoax. Ava confessed to m...</li><li><a href="https://open.spotify.com/episode/6a44wSFv8bc1T9x3mEE9Dq?si=tWnXTxqHQbqpky6bWqj0uw&nd=1&dlsi=d20a7ee755104caa">Sancocho con Limon - Quatsch Session 01</a>: FELD.FM · Episode
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1352724286988746882)** (202 messages🔥🔥): 

> `Mind Map Pixelation Fix, Mind Map Feature Feedback, NotebookLM vs ChatGPT, Access to New NotebookLM, Feedback Methods for NotebookLM` 


- **Zoom in for Crisp Mind Map Downloads**: A member recommends zooming in on tabs before downloading a **Mind Map** to get a bigger and higher quality output and fix pixelation issues.
   - The member also declared that *this tool is an absolute game changer*, touting the crazy context window and low hallucination rates, even cancelling their subscriptions to **ChatGPT** and **Claude**.
- **Mind Mapping Sparks Symbolic Reasoning**: A user believes that getting **Mind Mapping** right is an important step toward more effective and smarter AI and may be indicative of symbolic reasoning.
   - They suggest that once knowledge can be expressed as a network of meanings, these data structures can be easily corrected with simple manipulations like transplanting nodes or adding intermediate nodes.
- **NotebookLM is not an App, but a PWA**: A user sought to change the language on the app, but another user noted that **NotebookLM** doesn't have an app, but rather a Progressive Web App (PWA).
   - They recommend removing the app, loading **NotebookLM** in the browser with the `?hl=LANGUAGE` option, and then reinstalling the **PWA**.
- **Podcast Language can be "Forced"**: A user found that it's possible to "force" a podcast to generate in another language by inputting a specific prompt at the beginning of the text settings, though English is the only officially supported language.
   - They used the prompt *PT-BR cria o podcast em português* to generate a Portuguese podcast, emphasizing it doesn't always work but finds it cool when it does.
- **Mind Map Feature gets Mixed Reviews**: A user thinks that the new mind map is a great addition to **NotebookLM**, but finds it has major weaknesses.
   - They state that the mind map needs constant regeneration to update and lacks details beyond the topic, requiring back-and-forth navigation and asked for *topic and subtopic could be explained within the topic itself*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.playbook.com/s/notebooklm/the-deep-dive">Playbook</a>: Playbook is a modern creative file manager. Store, tag, and organize your files and folders beautifully. Designers, sign up today and get 4TB storage!</li><li><a href="https://notebooklm.google.com/">no title found</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=1197563608642675832-NC">Learn how NotebookLM protects your data - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en">Export your users' data - Google Workspace Admin Help</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1352761126160437350)** (106 messages🔥🔥): 

> `RWKV architecture development, AI model viability prediction, EleutherAI evaluation methods, Low precision data types for RL, MkDocs site for lm-evaluation-harness` 


- **Virtual Testing Environment Predicts Model Viability**: A member proposed a virtual testing environment (AKA the simulator) that predicts AI model viability before training to reduce wasted resources, saving time and accelerating AI innovation by eliminating unnecessary failed experiments before they happen in expensive real-world training.
   - The member stated that their goal is *not to achieve 100% accuracy in predicting an AI mechanism’s behavior*—it’s to create a system that can at least tell us whether a model has a realistic chance of working or is doomed to fail early on.
- **EleutherAI Evaluation Methods Detailed in New Blog**: A member wrote a quick blog on evaluation methods for EleutherAI and set up an [MkDocs site for easier navigation](https://slyracoon23.github.io/lm-evaluation-harness/).
   - They are awaiting review on [this PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2832) too.
- **Contributor Cautioned on AI-Generated Content in PRs**: A member was cautioned about the use of AI to generate content for pull requests, emphasizing the importance of vetting contributions to avoid adding spam.
   - It was suggested that unless the author is 100% certain they're correct on everything, *it would be better to withdraw the contribution until you are*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cloneofsimo/sdxl_inversions/blob/800613b426785757fca4964badeb666218e59eee/sdxl.py#L86">sdxl_inversions/sdxl.py at 800613b426785757fca4964badeb666218e59eee · cloneofsimo/sdxl_inversions</a>: Contribute to cloneofsimo/sdxl_inversions development by creating an account on GitHub.</li><li><a href="https://slyracoon23.github.io/blog/posts/2025-03-21_eleutherai-evaluation-methods.html">EleutherAI’s lm-evaluation-harness: Architecture and Configuration – Earl Potters</a>: A comprehensive guide to configuration, task architecture, and model integration</li><li><a href="https://slyracoon23.github.io/lm-evaluation-harness/">LM Evaluation Harness</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2832">Add MkDocs Documentation with GitHub Actions Deployment by Slyracoon23 · Pull Request #2832 · EleutherAI/lm-evaluation-harness</a>: Description:This PR introduces MkDocs integration to the LM Evaluation Harness repository, significantly enhancing documentation readability and accessibility. It provides:MkDocs setup: Configu...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1352903760178974762)** (121 messages🔥🔥): 

> `AI simulation environments, Continual learning in production LLMs, Architecture-aware optimizers, Sharpness Disparity across Transformer blocks, VectorAdam optimizer` 


- **AI simulator for research**: A member shared an idea for a virtual environment to test AI innovations, potentially saving **money and resources**, as detailed in the attached [Ai_simulator.pdf](https://cdn.discordapp.com/attachments/747850033994662000/1352903759839363083/Ai_simulator.pdf?ex=67e30110&is=67e1af90&hm=6dd1c8028d8932d9e8b64355594bcf7c338adbf09e986186ccd4322d9cbcf99b&).
   - Others pointed out that testing new architectures at a small scale is already relatively inexpensive, costing around **$5** to train a **L6D512** model on a **3090** for a day.
- **Optimal Optimizer Derivation Dilemma**: Members discussed the difficulty of deriving an optimal optimizer for specific architectures, noting that even for transformers, no such optimizer has been found, despite the availability of unconventional architectures.
   - One member suggested that if a near-optimal optimizer could be derived for an arbitrary architecture, it would be *work deserving of an award*.
- **VectorAdam rotation equivariance exposed**: VectorAdam modifies the second moment update to be the square of the vector norm per gradient vector, addressing coordinate-system bias in Adam, potentially improving rotation equivariance, as shown in this [VectorAdam paper](https://www.dgp.toronto.edu/~zling/vector-adam/).
   - It was noted that VectorAdam is not similar to Adafactor, but more like a blocked approximation with **block size = hidden dim**.
- **Convergence lemmas debunked**: It was suggested that convergence lemmas may not be important and that the regularizers can go in the loss function, so the AdamW detail can be ignored, or put in a separate loss function.
   - Other researchers believed this to be incorrect because the optima you're looking for is actually quite different with different regularization. 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.17126">Modifying Large Language Model Post-Training for Diverse Creative Writing</a>: As creative writing tasks do not have singular correct answers, large language models (LLMs) trained to perform these tasks should be able to generate diverse valid outputs. However, LLM post-training...</li><li><a href="https://arxiv.org/abs/1907.04164">Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model</a>: Increasing the batch size is a popular way to speed up neural network training, but beyond some critical batch size, larger batch sizes yield diminishing returns. In this work, we study how the critic...</li><li><a href="https://arxiv.org/abs/2502.19002">The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training</a>: Transformers consist of diverse building blocks, such as embedding layers, normalization layers, self-attention mechanisms, and point-wise feedforward networks. Thus, understanding the differences and...</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: no description found</li><li><a href="https://www.dgp.toronto.edu/~zling/vector-adam/">VectorAdam for Rotation Equivariant Geometry Optimization</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1352956296466534400)** (20 messages🔥): 

> `mechinterp backlash, token level activations, SAE visualizations, single token activations, untied embeddings` 


- **MechInterp Faces Academic 'Backlash'**: Members discussed that there seems to be an academic 'backlash' to the 'mechinterp' brand because so much of it is outside of traditional academic channels.
   - Theorizing that mechinterp is outside the mainstream academic channels, and they are resistant to the paradigm.
- **Token Activations Analyzed for Accuracy**: A member is extracting token level activations on an **SAE**, questioning whether passing a single/pair of tokens would yield more accurate results than passing a whole sentence.
   - They found that the first token to trigger an activation is *holocaust* but it's not the token with the strongest activation, and wondered if neuron activation might be context specific.
- **SAEviz Library for Visualization**: When looking at neuronpedia website graphs per feature/neuron, it was suggested to look into **SAEviz**, a library that does those visualizations using the **logit lens**.
   - The discussion clarified that these visualizations represent the ground truth activations rather than approximations.
- **Single Token Activation Doubts Raised**: A member questioned the validity of single token activations, emphasizing that neurons are only ever active in contexts, it doesn't make sense to analyze them in isolation.
   - They explained that the activations are influenced by the context before; for instance, the phrase *I am a dictator I want to* might change the activation on *to*.
- **Models need time to "warm up"**: A member states that models need time to 'warm up', where for the first 50 tokens contextual features tend to be ablated by the model by attending to the `end-of-text` token.
   - The intuition being that the model doesn't have enough information to make good judgements about context.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1353676488788148275)** (1 messages): 

> `Recursive Design, GAN vs. CNN vs. RL Architectures` 


- **Recursive Design Emerges as a Promising Technique**: A member introduced a novel diagram using a recursive design, distinguishing it from traditional **GANs** (*Generative Adversarial Networks*).
   - This member highlighted that their implementation emphasizes structural organization over sequential processing, leveraging **CNNs** for filtering and **RL** for refining responses.
- **Alternate Architectures**: The user proposed an alternate architecture using recursive design.
   - The user distinguished the architecture from **GAN** as an expression, **CNN** for filtering, and **RL** for response refinement.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1353833840694788237)** (1 messages): 

> `lm_eval update, CI test failures` 


- **Request to update `lm_eval`**: A member is drafting a PR to update the evaluation logic to `lm_eval==0.4.8`, the latest version, referencing the [Evals PR](https://github.com/EleutherAI/gpt-neox/pull/1348).
- **CI Tests Failures**: A member observed that CI tests are failing for the **lm_eval update PR** and another test PR created with trivial changes, asking if the repo's CI is healthy, and referencing the [CI Test PR](https://github.com/EleutherAI/gpt-neox/pull/1349).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1348">Update Evaluation Logic to Latest `lm_eval` (0.4.8) and Support Automatic Benchmark Evals w/o Validation Set by Kyle1668 · Pull Request #1348 · EleutherAI/gpt-neox</a>: I&amp;#39;m training a model where I want to train on the entire datasets. I do not want to split the dataset into train/val/test. I want to evaluate on a set of benchmarks, one of which was introduce...</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1349">[Throw Away] Sanity Check CI by Kyle1668 · Pull Request #1349 · EleutherAI/gpt-neox</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1353440360520749210)** (1 messages): 

> `StarVector, SpatialLM, Hugging Face Agents Course, Xet on the Hub, HF Welcome Page Makeover` 


- ****StarVector** emerges as vector graphics virtuoso**: A new foundation model called **StarVector** has been released on Hugging Face for generating scalable vector graphics code from images and text, available at [Hugging Face](https://huggingface.co/collections/starvector/starvector-models-6783b22c7bd4b43d13cb5289).
   - The initial release includes the **starvector/starvector-1b-im2svg** model.
- ****SpatialLM** navigates the 3D landscape**: **SpatialLM**, a 3D large language model designed to process 3D point cloud data, has been released on Hugging Face at [manycore-research/SpatialLM-Llama-1B](https://huggingface.co/manycore-research/SpatialLM-Llama-1B).
   - It generates structured 3D scene understanding outputs and can be further explored via the [project website](https://manycore-research.github.io/SpatialLM) and [GitHub repository](https://github.com/manycore-research/SpatialLM).
- **HF Agents Course embraces LlamaIndex, LangChain, and SmolAgents**: The Hugging Face Agents Course now includes integrations for **LlamaIndex**, **LangChain**, and **smolagents**, offering learners diverse approaches to agent frameworks.
   - The course aims to provide fundamental knowledge applicable across different frameworks, making it accessible to those already familiar with one or more of them, according to [this tweet](https://x.com/ben_burtenshaw/status/1903025737633841170).
- ****Xet** accelerates on the Hub**: Hugging Face's **Xet Team** has migrated the first Model and Dataset repositories off LFS and to Xet storage.
   - This is a step toward empowering AI builders to build and collaborate more effectively on massive models and datasets, described in more detail in this [blog post](https://huggingface.co/blog/xet-on-the-hub).
- **Hugging Face revamps welcome page**: The Hugging Face welcome page has received a significant makeover, offering a streamlined access to community AI apps, open-source libraries, local model execution, and more.
   - Users can explore various sections like HF Spaces, Open Source Libraries, Local Models, and the Inference Playground via the updated [welcome page](https://huggingface.co/welcome).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/starvector/starvector-models-6783b22c7bd4b43d13cb5289">💫StarVector Models - a starvector Collection</a>: no description found</li><li><a href="https://huggingface.co/manycore-research/SpatialLM-Llama-1B">manycore-research/SpatialLM-Llama-1B · Hugging Face</a>: no description found</li><li><a href="https://x.com/ben_burtenshaw/status/1903025737633841170">Tweet from Ben Burtenshaw (@ben_burtenshaw)</a>: The @huggingface  Agents Course now includes three major agent frameworks. LlamaIndex, LangChain, and our very own smolagents.We&#39;ve worked to integrate the three frameworks in distinctive ways so ...</li><li><a href="https://huggingface.co/blog/xet-on-the-hub">Xet is on the Hub</a>: no description found</li><li><a href="https://huggingface.co/welcome">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/blog/endpoint-analytics">The New and Fresh analytics in Inference Endpoints</a>: no description found</li><li><a href="https://huggingface.co/blog/ai-action-wh-2025">AI Policy @🤗: Response to the White House AI Action Plan RFI</a>: no description found</li><li><a href="https://huggingface.co/blog/olympic-coder-lmstudio">Open R1: How to use OlympicCoder locally for coding</a>: no description found</li><li><a href="https://huggingface.co/blog/nvidia-physical-ai">NVIDIA&#39;s GTC 2025 Announcement for Physical AI Developers: New Open Models and Datasets</a>: no description found</li><li><a href="https://huggingface.co/blog/burtenshaw/gemma3-thinking">Making Gemma 3 think</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1352720141007196330)** (136 messages🔥🔥): 

> `ComfyUI Samplers, Open Schizo Leaderboard, Short Story Generator with Pytorch, Photorealism Settings for SD1.5/SDXL, Flux.1 Model Performance` 


- **ComfyUI Sampler Strategy Session**: Members discussed the best **sampler_name** to use in ComfyUI, seeking recommendations for optimal configurations but not knowing much about it.
   - One user recommended *dpmpp_2m_sde* sampler and *kl_optimal* scheduler for photorealism with **SD1.5** and **SDXL checkpoints**.
- **Showcasing Crazies on Open Schizo Leaderboard**: A new leaderboard was released on Hugging Face, showcasing top models.
   - Find the [Open-Schizo-Leaderboard](https://huggingface.co/spaces/rombodawg/Open-Schizo-Leaderboard) on HuggingFace.
- **Model Integration Protocol (MIP) simplifies LLM-powered service**: A user is seeking feedback on **Model Integration Protocol (MIP)**, proposing a simpler and more scalable approach for OpenAI that automatically converts existing methods, classes, and HTTP endpoints into JSON-RPC using reflection.
   - This approach aims to drastically reduce development overhead while maintaining platform independence and compatibility with any LLM, and a [Neurocaster-Server implementation](https://github.com/vishalmysore/neurocaster-server) illustrates its use.
- **Wan Models Debut AutoencoderKL**: A user encountered an import error related to `AutoencoderKLWan` from the `diffusers` library, potentially due to using a development version or a mistaken repository.
   - A github [issue](https://github.com/huggingface/diffusers/issues/10963) was found which explains that the user may be experiencing a development version error, since `AutoencoderKLWan` is not available yet.
- **InferenceClient API throws Authentication Error**: A user reported a **403 Forbidden** error when attempting to list deployed models using the `InferenceClient` API, even with read-only tokens configured to allow calls to Inference Providers.
   - The error indicates insufficient permissions to call Inference Providers on behalf of the user and a user posted a [link](https://huggingface.co/posts/kpadpa/282697879499561) with the same error.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://serapath-example1-orchestrator-agent.hf.space`">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/rombodawg/Open-Schizo-Leaderboard">Try Rombos-LLM-V2.5-Qwen-7b - a Hugging Face Space by rombodawg</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-sdks-docker">Docker Spaces</a>: no description found</li><li><a href="https://huggingface.co/chat/)">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/spaces/fantaxy/fantasy-novel-kr/discussions">fantaxy/fantasy-novel-kr · Discussions</a>: no description found</li><li><a href="https://huggingface.co/blog/hmb/gradio-dataframe-upgrade">Gradio’s Dataframe has been upgraded! 🎨</a>: no description found</li><li><a href="https://huggingface.co/posts/kpadpa/282697879499561">@kpadpa on Hugging Face: &quot;What does this mean and how can I fix it? 

&quot;This authentication method does…&quot;</a>: no description found</li><li><a href="https://docs.vllm.ai/en/v0.7.2/getting_started/examples/whisper.html">Whisper &#8212; vLLM</a>: no description found</li><li><a href="https://aikval25.kattis.com/contests/aikval25/problems/windchill">Windchill &ndash; Kattis, AI-olympiadens Kval 2025</a>: no description found</li><li><a href="https://huggingface.co/posts/julien-c/158943939527784">@julien-c on Hugging Face: &quot;Important notice 🚨

For Inference Providers who have built support for our…&quot;</a>: no description found</li><li><a href="https://huggingface.co/docs/api-inference/pricing">Pricing and Rate limits</a>: no description found</li><li><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers">Wan-AI/Wan2.1-I2V-14B-480P-Diffusers · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1: Update #3</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/issues/10963">cannot import name &#39;AutoencoderKLWan&#39; from &#39;diffusers&#39; · Issue #10963 · huggingface/diffusers</a>: Describe the bug ImportError: cannot import name &#39;AutoencoderKLWan&#39; from &#39;diffusers&#39; (/usr/local/lib/python3.10/dist-packages/diffusers/init.py) Reproduction from diffusers import Auto...</li><li><a href="https://huggingface.co/docs/inference-endpoints/index">Inference Endpoints</a>: no description found</li><li><a href="https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints">Inference Endpoints (dedicated) - Hugging Face Open-Source AI Cookbook</a>: no description found</li><li><a href="https://github.com/huggingface/text-generation-inference">GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference</a>: Large Language Model Text Generation Inference. Contribute to huggingface/text-generation-inference development by creating an account on GitHub.</li><li><a href="https://huggingface.co/support">Expert Support – Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint">Diffusers Image Outpaint - a Hugging Face Space by fffiloni</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/using-diffusers/img2img">Image-to-image</a>: no description found</li><li><a href="https://github.com/justinpinkney/stable-diffusion?tab=readme-ov-file#image-mixer">GitHub - justinpinkney/stable-diffusion</a>: Contribute to justinpinkney/stable-diffusion development by creating an account on GitHub.</li><li><a href="https://github.com/TheDenk/images_mixing">GitHub - TheDenk/images_mixing: Сombine images using usual diffusion models.</a>: Сombine images using usual diffusion models. Contribute to TheDenk/images_mixing development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces?sort=trending&search=vton">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces?sort=trending&search=try+on">Spaces - Hugging Face</a>: no description found</li><li><a href="https://archive.ph/2025.02.24-150819/https://medium.com/data-scientists-from-future/fine-tuning-open-source-language-models-a-step-by-step-guide-a38bed8df923">Fine-Tuning Open-Source Language Models: A Step-by-Step Guide | by Vi&#x2026;</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen/Qwen2.5-VL-7B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-4-multimodal-instruct">microsoft/Phi-4-multimodal-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/learn/cookbook/advanced_rag">Advanced RAG on Hugging Face documentation using LangChain - Hugging Face Open-Source AI Cookbook</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/tasks/asr">Automatic speech recognition</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1353192213567373354)** (5 messages): 

> `audio processing, AI agents, Tokenisers, BPE, Unigram language modelling` 


- **Dive into Audio Adventures**: A member is deep-diving into **audio processing** today.
- **Framework for Fantastic AI Agents**: A member is tackling the **framework** for **AI agents** today.
- **Tokeniser Tussle: BPE vs Unigram**: A member is exploring the mechanics of various **tokenisers**, specifically **BPE** and **unigram language modelling**.
- **Lightweight Models light up Laptops**: A member is researching **lightweight**, **fine-tunable models** suitable for running and tuning on a development laptop.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1353166030863732786)** (8 messages🔥): 

> `Logfire Callback for HF Transformers Trainer, TrashLens for image organization, pdf2notes: AI-powered PDF to Notes conversion, Kids feedback on UI/UX, Local API Usage` 


- ****Logfire Callback** Logs Training Events!**: A member created a [Logfire callback](https://github.com/louisbrulenaudet/logfire-callback) for **HF transformers Trainer** that logs training events.
   - This tool helps in tracking and analyzing the training process of transformer models in Hugging Face.
- ****TrashLens** Brings Order to Image Chaos!**: [TrashLens](https://github.com/0xrushi/TrashLens) is designed to bring order to image chaos, helping users focus on important content and free up space effortlessly.
   - The tool aims to streamline image organization, making it easier to manage and declutter visual data.
- ****pdf2notes** Turns PDFs into Organized Notes!**: [Pdf2Notes](https://github.com/AstraBert/pdf2notes) is an **AI-powered, open-source solution** that converts unstructured PDFs into well-ordered notes using **LlamaParse** and **Llama-3.3-70B**.
   - The tool uses **DeepMind's Gemini 2 Flash** for multi-modal parsing and features a chatbot for more in-depth insights, wrapped in a **Gradio** and **FastAPI** framework, and can be run locally with **Docker**.
- **Kids Provide Valuable UI/UX Feedback!**: A member shared that their son helped with the UI colors and enjoys the tool, especially unlocking new achievements.
   - Feedback from kids emphasizes the importance of engaging UI elements and achievement systems in educational tools.
- **API-Free Local Operation in Question!**: A member questioned if [pdf2notes](https://github.com/AstraBert/pdf2notes) can operate **100% locally without external APIs**, raising concerns about needing subscriptions for **Gemini** and **Groq**.
   - They criticized the Docker setup, suggesting it is too complex for non-power users who prefer simpler solutions without additional application installations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/0xrushi/TrashLens">GitHub - 0xrushi/TrashLens</a>: Contribute to 0xrushi/TrashLens development by creating an account on GitHub.</li><li><a href="https://github.com/louisbrulenaudet/logfire-callback">GitHub - louisbrulenaudet/logfire-callback: A callback for logging training events from Hugging Face&#39;s Transformers to Logfire 🤗</a>: A callback for logging training events from Hugging Face&#39;s Transformers to Logfire 🤗 - louisbrulenaudet/logfire-callback</li><li><a href="https://github.com/AstraBert/pdf2notes">GitHub - AstraBert/pdf2notes: Turn PDF into Notes in seconds📝</a>: Turn PDF into Notes in seconds📝. Contribute to AstraBert/pdf2notes development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1352896313557258272)** (6 messages): 

> `Qwen for video annotation, Opus clip opensource, LLMs and VLMs in autonomous driving` 


- **Qwen Guides Video Annotation Newbie**: A member sought advice on using **Qwen** with the **transformers library** for video frame extraction and annotation.
   - Another member recommended the [Qwen2.5-VL official GitHub repo](https://youtu.be/4twSI2XFK2s) for model information and quickstart examples.
- **Opensource Opus Clip Tool Seeks Helping Hands**: A member is trying to create an opensource version of **Opus Clip** (**video repurposing tool**).
   - The author seeks assistance with their "spaghetti repo and code" which utilizes **yolov8** and **revideo** for detecting people and splitting the video vertically.
- **LLMs and VLMs drive Autonomous Driving into the Future**: A member shared their new substack article about **LLMs** and **VLMs** in autonomous driving, highlighting improvements in vehicle capabilities.
   - The article references a survey paper, *A survey for foundation models in autonomous driving*, available on [arXiv:2402.01105](https://arxiv.org/abs/2402.01105).



**Link mentioned**: <a href="https://samerattrah.substack.com/p/autonomous-driving-with-llms-vlms">Autonomous driving with LLMs, VLMs, and MLLMs</a>: Discussing the application of Large Language/Vision Models in autonomous driving and the most significant developments and approaches.

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1353824302524530848)** (1 messages): 

> `Gradio Deep Links` 


- **Gradio 5.23 enables Deep Links!**: Gradio 5.23 introduces **Deep Links**, allowing direct linking to specific outputs like images or videos, exemplified by [this link](https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek) to a blue jay image.
   - To upgrade, use `pip install --upgrade gradio`.
- **Image.png**: The image shows an attached file.
   - The file is hosted on discord.



**Link mentioned**: <a href="https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek">black-forest-labs/FLUX.1-schnell</a>: no description found

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1352937712549630033)** (1 messages): 

> `Hackathon Timing, Hackathon Details` 


- **Hackathon Date Still a Mystery**: A member inquired about the hackathon date, expressing difficulty in finding relevant information about it.
   - They mentioned the **YouTube stream** stated the **22nd of March**, but found no confirmation.
- **Hackathon Details are missing**: The user is unable to find any relevant information about the Hackathon.
   - The user mentions the youtube stream said that it's today, but there are no details.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1352742104954306560)** (33 messages🔥): 

> `LangGraph rigidity, Local LLMs for smolagents, Gemini in LangGraph, API costs for notebooks, Agent storing retrieved info` 


- **LangGraph gains Fans Despite LangChain hate**: A member who just finished the **LangGraph** module likes the *rigidness* of **LangGraph** compared to **LangChain**, which they follow on Twitter and said *gets a lot of hate*.
   - Others seemed to echo this sentiment.
- **Local LLMs Need Beefy Machines to run Smolagents**: Members found that to run a local LLM and get good results on **smolagents**, you'll need a big one (around **32B** parameters) and that implies a powerful machine.
   - They tried with 'small' LLMs like **qwen coder 7B** or **deepsek-r1 7B** and the results with smolagents are pretty inconsistent.
- **Home Labs Arise to Reduce API Costs**: Members discussed the cost of **APIs** to complete the notebook, and those who do not wish to pay are working to build out a sufficient **home lab** to run models on and access them via **API**.
   - It was mentioned that InferenceClient APIs by huggingface are free to use with a limit of 300 requests/hour for free users.
- **Where does the Agent store for future reference?**: In the agentic RAG section of the course ([https://huggingface.co/learn/agents-course/unit2/smolagents/retrieval_agents](https://huggingface.co/learn/agents-course/unit2/smolagents/retrieval_agents)), it is unclear how the LLM agent *stores* the retrieved information for easy access when planning future events, optimizing efficiency in subsequent tasks.
   - It was suggested it is not the LLM but the agent that stores the search and that the agent itself would have to write it down somewhere, not just in the context.
- **API Token Issue Solved!**: A member was experiencing issues running code using **HuggingFaceInferenceAPI** and getting irrelevant responses from their LLM.
   - The issue was identified and resolved as a problem with the **API token**, which needed to be read-only to run locally.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2303.17651">Self-Refine: Iterative Refinement with Self-Feedback</a>: Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improv...</li><li><a href="https://huggingface.co/learn/agents-course/unit2/smolagents/retrieval_agents#basic-retrieval-with-duckduckgo)">Building Agentic RAG Systems - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm/">HuggingFace LLM - StableLM - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1352799575294873652)** (9 messages🔥): 

> `r1, vllm, cuda kernel` 


- **Debate Erupts Over r1 Training Curriculum!**: One member asked about the training curriculum, saying that it took *5 minutes* with **deepseek** to understand the humor.
   - Another member stated that **r1** is *incredibly slow*, requiring considerable power; their **Scaleway R1 grid** running *20 machines* around **3 PFLOPS** generated only a few hundred MB per day, so it was much faster to use **llama** and reverse engineer the thinking tokens from query response pairs.
- **CUDA Kernel Improvements Discussed**: One user inquired whether **vllm** was being used and also mentioned working on some **cuda kernel improvements**.
   - Another member simply answered *no*.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1352737566960386178)** (155 messages🔥🔥): 

> `MCP and K8s, Anthropic's MCP, MCP server directories, C# MCP SDK, Vercel's AI SDK with MCP Clients` 


- **K8s Setup Required to Test MCP Prompts**: To test MCP prompts, particularly those from [this file](https://github.com/strowk/mcp-k8s-go/blob/main/testdata/list_prompts_test.yaml) and [this test](https://github.com/strowk/mcp-k8s-go/blob/10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e/testdata/with_k3d/list_k8s_namespaces_test.yaml#L50), a Kubernetes setup is required.
   - An alternative implementation with prompts is available [here](https://github.com/Abiorh001/mcp_ev_assistant_server) for managing Electric Vehicle charging stations.
- **MCP isn't that complex! User says**: One user expressed confusion at the perception that MCP is complex, stating *JSON RPC isn't hard. Using SDKs it's even easier. Making an MCP server or client is pretty easy compared to a lot of other development work*.
   - They suggested that with just **1 cmd and 1 arg** you can add anything to any llm, with no need for public ip, tls cert, or any previous blocks.
- **Dive into MCP Server Repositories**: Users shared a list of useful MCP server directories, including [Glama](http://glama.ai/mcp/servers) with a report card system, [PulseMCP](https://www.pulsemcp.com/) for a well-organized and exhaustive list, and the [official MCP GitHub](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#model-context-protocol-servers).
   - These resources help developers find and assess various MCP servers for their projects.
- **New C# SDK officially released!**: A new official **C# SDK** for Model Context Protocol servers and clients has been released by Microsoft, as seen [here](https://github.com/modelcontextprotocol/csharp-sdk).
   - This provides developers with tools for building **AI applications** using **JavaScript** and **TypeScript**, integrating into web frameworks like [Next.js](https://nextjs.org) and [Svelte](https://svelte.dev/), per [Vercel AI SDK 4.2](https://vercel.com/blog/ai-sdk-4-2).
- **Zapier Integrates with MCP for broader AI application Access**: Zapier has released an MCP server, [providing access to over 8,000 integrations](https://zapier.com/mcp) for AI assistants to interact with various apps.
   - This allows AIs to perform real-world tasks such as sending messages, managing data, scheduling events, and updating records, expanding their capabilities beyond text generation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/llmindsetuk/status/1885719128247296109">Tweet from llmindset (@llmindsetuk)</a>: microsoft 365 copilot. the word &#34;agent&#34; is now defined by the millions of corporate eyeballs that will see this screen. &#34;Enterprise data protection&#34; is given prominence.</li><li><a href="https://x.com/tom_doerr/status/1903972369443475471">Tweet from Tom Dörr (@tom_doerr)</a>: Product requirements document to tasks toolQuoting Eyal Toledano (@EyalToledano) Sick of @cursor_ai rewriting good code or going in circles?Introducing Task Master ✨ A CLI that turns your PRD into a l...</li><li><a href="https://zapier.com/mcp">Zapier MCP—Connect your AI to any app instantly</a>: The fastest way to let your AI assistant interact with thousands of apps. No complex API integrations required.</li><li><a href="https://block.github.io/goose/docs/getting-started/providers#local-llms-ollama">Configure LLM Provider | codename goose</a>: Goose is compatible with a wide range of LLM providers, allowing you to choose and integrate your preferred model.</li><li><a href="https://VeyraX.com/mcp">Tweet from VeyraX</a>: VeyraX is Agenic Component Interface</li><li><a href="https://github.com/FreePeak/">Free Peak</a>: Indie Hacker. Free Peak has one repository available. Follow their code on GitHub.</li><li><a href="https://glama.ai/mcp/servers/@gannonh/firebase-mcp">Firebase MCP</a>: The Firebase MCP server provides a standardized interface to interact with Firebase services, including Firebase Authentication, Firestore, and Firebase Storage.</li><li><a href="https://github.com/Abiorh001/mcp_ev_assistant_server">GitHub - Abiorh001/mcp_ev_assistant_server: A powerful server implementation for managing Electric Vehicle (EV) charging stations, trip planning, and resource management. This server provides a comprehensive set of tools and APIs for EV-related services.</a>:  A powerful server implementation for managing Electric Vehicle (EV) charging stations, trip planning, and resource management. This server provides a comprehensive set of tools and APIs for EV-rel...</li><li><a href="https://github.com/strowk/mcp-k8s-go/blob/main/testdata/list_prompts_test.yaml">mcp-k8s-go/testdata/list_prompts_test.yaml at main · strowk/mcp-k8s-go</a>: MCP server connecting to Kubernetes. Contribute to strowk/mcp-k8s-go development by creating an account on GitHub.</li><li><a href="https://vercel.com/blog/ai-sdk-4-2">AI SDK 4.2 - Vercel</a>: AI SDK 4.2 introduces MCP clients, reasoning, image generation with language models, message parts, sources, and more</li><li><a href="https://github.com/modelcontextprotocol/specification/discussions/220">MCP Hosting Working Group · modelcontextprotocol/specification · Discussion #220</a>: Pre-submission Checklist I have verified this would not be more appropriate as a feature request in a specific repository I have searched existing discussions to avoid duplicates Your Idea Hey ever...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/343">Fix/base64 handling (Issue #342)  by evalstate · Pull Request #343 · modelcontextprotocol/python-sdk</a>: Single line change to lowlevel/server.py + test to verify that Base64 decoding is not url safe and as expected by the Client.Motivation and ContextTransmitting Binary resources.How Has This Been...</li><li><a href="https://github.com/spences10/mcp-sequentialthinking-tools">GitHub - spences10/mcp-sequentialthinking-tools: 🧠 An adaptation of the MCP Sequential Thinking Server to guide tool usage. This server provides recommendations for which MCP tools would be most effective at each stage.</a>: 🧠 An adaptation of the MCP Sequential Thinking Server to guide tool usage. This server provides recommendations for which MCP tools would be most effective at each stage. - spences10/mcp-sequential.....</li><li><a href="https://github.com/strowk/mcp-k8s-go/blob/10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e/testdata/with_k3d/list_k8s_namespaces_test.yaml#L50">mcp-k8s-go/testdata/with_k3d/list_k8s_namespaces_test.yaml at 10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e · strowk/mcp-k8s-go</a>: MCP server connecting to Kubernetes. Contribute to strowk/mcp-k8s-go development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/csharp-sdk">GitHub - modelcontextprotocol/csharp-sdk: The official C# SDK for Model Context Protocol servers and clients, maintained by Microsoft</a>: The official C# SDK for Model Context Protocol servers and clients, maintained by Microsoft - modelcontextprotocol/csharp-sdk</li><li><a href="https://glama.ai/mcp/servers/@heurist-network/heurist-mesh-mcp-server">Mesh Agent MCP Server</a>: A Model Context Protocol server that connects Claude to Heurist Mesh APIs, providing access to various blockchain and web3 tools including cryptocurrency data, token security, Twitter intelligence, an...</li><li><a href="https://github.com/heurist-network">Heurist</a>: Heurist is a Decentralized AI-as-a-Service Cloud. Heurist has 22 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#model-context-protocol-servers)">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/FreePeak/db-mcp-server">GitHub - FreePeak/db-mcp-server</a>: Contribute to FreePeak/db-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers/pull/355">Update README: Add multi-database MCP server built with Golang by linhdmn · Pull Request #355 · punkpeye/awesome-mcp-servers</a>: Add multi-database MCP server built with Golang, supporting MySQL &amp; PostgreSQLAs an alternative for the https://github.com/FreePeak/db-mcp-server
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1352750945901084674)** (29 messages🔥): 

> `mcpwizard, vscode-mcp, DICOM servers MCP, google sheet MCP server, Narrative Spittoon Inversion project` 


- ****MCPwizard** Simplifies Server Creation**: A member introduced [mcpwizard](https://www.npmjs.com/package/mcpwizard), a CLI tool to simplify creating and deploying **MCP servers**, highlighting features like initializing projects and adding custom tools to Claude assistants.
   - The tool's [GitHub repo](https://github.com/yoannarz/mcpwizard) was also shared for community feedback and contributions.
- ****VS Code MCP** Gets Community Acclaim**: Members shared a [VS Code MCP](https://github.com/block/vscode-mcp) that they've wanted.
   - It's described in action in this [Youtube Short](https://www.youtube.com/shorts/gddEgvCLrgU) .
- ****DICOM MCP** Server for Clinical Imaging**: A member created an MCP server for interacting with **DICOM servers**, enabling AI assistants to query medical imaging systems for patient scans and clinical reports, available at [christianhinge.com](https://www.christianhinge.com/projects/dicom-mcp/).
   - The associated **GitHub repo** is located [here](https://github.com/ChristianHinge/dicom-mcp).
- ****Google Sheets MCP** for Direct Editing**: A member built a **Google Sheet MCP server**, allowing Claude to directly edit spreadsheets, streamlining data handling and formula adjustments as mentioned in [this tweet](https://x.com/xing101/status/1903391600040083488).
   - The code can be found [here](https://github.com/xing5/mcp-google-sheets).
- ****Automated Debugger MCP Server** Enhancements**: A member has been making improvements to their [automated debugger MCP server](https://github.com/jasonjmcghee/claude-debugs-for-you), encouraging others to try it out and contribute.
   - The server allows LLMs to *place breakpoints, run code, move between breakpoints, and evaluate expressions*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lokka.dev/">Lokka | Lokka</a>: Lokka is an AI agent tool that brings the power of Microsoft Graph to AI agents like GitHub Copilot and Claude that run on your local desktop.</li><li><a href="https://github.com/evalstate/mcp-webcam/.">GitHub - evalstate/mcp-webcam: Capture live images from your webcam with a tool or resource request</a>: Capture live images from your webcam with a tool or resource request  - GitHub - evalstate/mcp-webcam: Capture live images from your webcam with a tool or resource request</li><li><a href="https://github.com/gotohuman/gotohuman-mcp-server">GitHub - gotohuman/gotohuman-mcp-server</a>: Contribute to gotohuman/gotohuman-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/jasonjmcghee/claude-debugs-for-you">GitHub - jasonjmcghee/claude-debugs-for-you: Enable any LLM (e.g. Claude) to interactively debug any language for you via MCP and a VS Code Extension</a>: Enable any LLM (e.g. Claude) to interactively debug any language for you via MCP and a VS Code Extension - jasonjmcghee/claude-debugs-for-you</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://www.christianhinge.com/projects/dicom-mcp/"> Agentic healthcare LLMs | Christian Hinge </a>: no description found</li><li><a href="https://github.com/ChristianHinge/dicom-mcp">GitHub - ChristianHinge/dicom-mcp: Model Context Protocol (MCP) for interacting with dicom servers (PACS etc.)</a>: Model Context Protocol (MCP) for interacting with dicom servers (PACS etc.) - ChristianHinge/dicom-mcp</li><li><a href="https://github.com/Kvadratni/speech-mcp">GitHub - Kvadratni/speech-mcp: Speech MCP: A Goose MCP extension for voice interaction with audio visualization</a>: Speech MCP: A Goose MCP extension for voice interaction with audio visualization - Kvadratni/speech-mcp</li><li><a href="https://github.com/MubarakHAlketbi/game-asset-mcp">GitHub - MubarakHAlketbi/game-asset-mcp: An MCP server for creating 2D/3D game assets from text using Hugging Face AI models.</a>: An MCP server for creating 2D/3D game assets from text using Hugging Face AI models. - MubarakHAlketbi/game-asset-mcp</li><li><a href="https://github.com/MushroomFleet/UNO-MCP">GitHub - MushroomFleet/UNO-MCP: Unified Narrative Operator</a>: Unified Narrative Operator. Contribute to MushroomFleet/UNO-MCP development by creating an account on GitHub.</li><li><a href="https://x.com/xing101/status/1903391600040083488">Tweet from Xing Wu (@xing101)</a>: Everyone&#39;s buzzing about #MCP and here&#39;s why: a weekend project solved a long-standing pain for me. No more copying data tables to spreadsheets or decoding complex formula guides from LLM conv...</li><li><a href="https://github.com/xing5/mcp-google-sheets">GitHub - xing5/mcp-google-sheets</a>: Contribute to xing5/mcp-google-sheets development by creating an account on GitHub.</li><li><a href="https://github.com/yoannarz/mcpwizard">GitHub - yoannarz/mcpwizard: A package to help you create and deploy MCP servers</a>: A package to help you create and deploy MCP servers - yoannarz/mcpwizard</li><li><a href="https://shorturl.at/sLWsr">MCPwizard helps you building mcp servers !</a>: Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1352725930593878149)** (102 messages🔥🔥): 

> `Speech to Text Solutions, GPT4All and NSFW content, LocalDocs Disappearing, LLMs for Office Tasks, Running Models on Multiple Devices` 


- **Prompting Proficiency Prevails**: Members discussed that if a language model is desired to respond in a specific language (e.g. German), it is best to write the system message in that language to avoid triggering *"Im Kontext Lernen"* (in-context learning).
   - It was further suggested that **avoiding negative sentences** with words like *"nicht"* and *"don't"* can improve results, with a recommendation to rephrase instructions to use active verbs instead.
- **Nemo's Nuances Named**: It was mentioned that [Mistral Nemo is a 12b model](https://huggingface.co/mistralai) and Mistral 24b is Mistral 3 or Mistral 3.1, with discussion around specific model details for projects.
   - Confusion arose around identifying the exact model, with one member emphasizing the need for precise model information to avoid issues.
- **GPT4All's LocalDocs Vanish**: A user reported that their entire catalog of local docs disappeared for no apparent reason, prompting discussion about potential causes such as **changes to the install folder** or **lack of admin rights**.
   - Members recommended backing up the *localdocs.db* file and the original documents to prevent data loss, and suggested that a Windows 11 update might have caused the issue by messing with drive letters.
- **LLMs Eye Medical Office Efficiency**: Members discussed the potential of using local LLMs in a medical office setting to help doctors create reports and assist with treatments, with a focus on the system learning from past dictated notes.
   - However, it was cautioned that **LLMs may not be suitable for handling financial or medical data** due to the risk of confabulation and the need for precise information.
- **GPT4All Lacks Vision**: A member asked if any models that GPT4All can run have vision capabilities, and it was confirmed that **GPT4All does not support vision capabilities**.
   - Alternative tools like **LM-Studio** were suggested as options for vision-related tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai">mistralai (Mistral AI_)</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1broa8h/is_there_a_way_for_me_to_use_multiple_computers/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jcm5p2/ocr_llm_for_invoice_extraction/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1352831709149794374)** (7 messages): 

> `High performance software, Vendor lock-ins, OpenCL, OpenMP, OpenACC, Vulkan’s Compute API, and SYCL, Democratizing AI Compute, Hardware Lottery` 


- **Exploring High-Performance Software Landscape**: A member is exploring the landscape of writing **high-performance software** for various devices and industry needs, particularly concerning vendor lock-ins and the necessity of porting projects to phones or embedded devices.
   - They requested recommendations for papers, search terms, or authors to better understand the trade-offs and options available.
- **Open and Portable APIs**: A member suggested starting with open and portable APIs such as **OpenCL**, **OpenMP**, **OpenACC**, **Vulkan’s Compute API**, and **SYCL**, citing their well-documented reasons for creation.
   - They also pointed to **POCL** as an academic project with related papers.
- **Democratizing AI Compute Series**: A member linked to Chris Lattner's "[Democratizing AI Compute](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai)" series, highlighting how **better hardware utilization** can dramatically reduce the need for expensive GPUs.
   - The series includes articles on **CUDA**, **OpenCL**, and **AI compilers (TVM and XLA)**.
- **The Hardware Lottery**: A member recommended the paper "[The Hardware Lottery](https://arxiv.org/abs/2009.06489)" by Sara Hooker, which discusses how hardware and software can determine the success or failure of research ideas.
   - The abstract states that the paper *introduces the term hardware lottery to describe when a research idea wins because it is suited to the available software and hardware and not because the idea is superior to alternative research directions*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai">Modular: Democratizing AI Compute, Part 1: DeepSeek’s Impact on AI</a>: Part 1 of an article that explores the future of hardware acceleration for AI beyond CUDA, framed in the context of the release of DeepSeek</li><li><a href="https://arxiv.org/abs/2009.06489">The Hardware Lottery</a>: Hardware, systems and algorithms research communities have historically had different incentive structures and fluctuating motivation to engage with each other explicitly. This historical treatment is...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1353002566157602847)** (82 messages🔥🔥): 

> `Mojo Logging Library, Mojo Formatter Tool, Mojo Dict Default Values, GPU Support for Windows, Mojo Inline Assembly Documentation` 


- **Logging Library in Mojo Remains WIP**: A logging library is work-in-progress in the standard library but is getting reworked; full serialization, and likely reflection, is needed before logging can be considered finished.
   - According to one member, *We would need to finish serialization before we could call logging finished, which probably means reflection.*
- **Mojo Boasts Built-In Formatting Tool**: Mojo includes a built-in formatting tool, mojo format, similar to Black in Python or fmt in Rust, for code formatting.
- **Dict Lacks Default Value Generation**: The Mojo Dict is more like Python's dict and does not include functionality to generate default values like defaultdict.
- **Windows GPU Support Frustrates Mojo Developers**: GPU support for Windows is difficult because the Windows compiler toolchain is a pain to work with; most people do not run enterprise GPU clusters on Windows, and there's little reason to improve tooling.
- **Mojo's Inline Assembly Documentation is a Mess**: Members noted the documentation for inline assembly in Mojo is a bit messy.
   - One member said *Time to harass Joe into writing documentation for it, then*, but this was immediately followed by *No harassing*.



**Link mentioned**: <a href="https://forum.modular.com/t/question-vpermi2b-inline-assembly-output-incorrect-in-loop-context-due-to-register-allocation/1091/2?u=sora">Question: vpermi2b inline assembly output incorrect in loop context due to register allocation</a>: Maybe you could try this  from sys import llvm_intrinsic  alias T = SIMD[DType.int8, 64]  @always_inline(&quot;nodebug&quot;) fn vpermi2b(a: T, b: T, idx: T) -&gt; T:   return llvm_intrinsic[&quot;llv...

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1353465638743707820)** (3 messages): 

> `MAX Platform, pixi.toml, max-pipeline, Python model graphs, magic CLI` 


- **Newcomer Asks About MAX Platform**: A new user inquired about modifying the **max/pipeline** directory and testing changes within the **MAX Platform** via the [pixi.toml file](https://github.com/modular/max/tree/main/src/max).
   - Specifically, they were curious about altering the **max-pipeline** without downloading it as a dependency.
- **Editing Python Model Graphs**: A member explained that while **Python model graphs** aren't well-documented, the **MAX pipelines** module's Python source is downloaded locally.
   - Changes to these local files in `.modular/envs/max-pipelines/lib/python3.12/site-packages/max/pipelines` (or similar location in the `.magic` environment) should reflect when running pipelines.
- **Running max-pipelines via Python**: The original poster asked if they could run **max-pipelines** directly with Python instead of using the **magic CLI** to add more command line parameters.
   - No direct response was given on the feasibility of this approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modular/max/tree/main/src/max">max/src/max at main · modular/max</a>: The MAX Platform (includes Mojo). Contribute to modular/max development by creating an account on GitHub.</li><li><a href="https://github.com/m">m - Overview</a>: Typist, engineer, code poet, lover of beautiful data structures. - m
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1352740201688207571)** (4 messages): 

> `AGNTCY, Large-Scale Structured Extraction, Deepseek R1 + LlamaIndex RAG app, WeAreDevs WebDev & AI Day` 


- **AGNCY Initiative for Agentic Interactions Emerges**: Luke discusses the motivations behind **AGNCY**, an effort to create an [open standard for agentic interactions](https://t.co/I558Qe2u4n).
- **Scale Structured Extraction on Complex Docs**: LlamaIndex highlights how to perform **large-scale structured extraction** over complicated documents, extracting **50-100 fields** from a pydantic schema with nested sub-schemas, requiring high accuracy.
   - More details [here](https://t.co/tO1vACKTGo).
- **Deepseek R1 and LlamaIndex Build RAG**: LlamaIndex highlights a project from Akshay Pachaar integrating **Deepseek AI** to build a **RAG app** with **LlamaIndex** for orchestration, **Deepseek AI R1** for inference, **Ollama** to locally serve R1, and **Streamlit** for the UI; more details available [here](https://t.co/KS26JUkwz0).
- **WeAreDevs WebDev & AI Day Approaches**: LlamaIndex advertises **WeAreDevs WebDev & AI Day** this Thursday, promising insights from industry experts on how **AI is transforming web development** and its impact on software development, with more information [available here](https://t.co/c5N5BJ34mr).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1352741141925203999)** (71 messages🔥🔥): 

> `Haystack Uninstall LlamaIndex Install, Ollama Integration Error, RTX 3060 Token Issues, Custom AI Interview Prep, Agent Workflow Timeout Error` 


- ****LlamaIndex + Ollama = Perfect RAG?****: A member sought help setting up a RAG pipeline with **LlamaIndex**, **Ollama**, and related integrations, receiving a code snippet from Deepseek to get started but ran into dependency issues.
   - The error was caused by the incorrect naming of a function argument (**model_name** instead of **model**), and while the error was resolved, the generated answer was still not what was expected.
- ****Crafting Custom AI Interview Grindset****: A member is building a local AI using **Llama 3.2**, **Sonnet 3.7**, and **Dolphin** blended into a 16B model with RAG, custom fine-tuning, and dreams of landing a job at an AI/Tech company.
   - He is trying to get his AI to *apply to ai/tech companies and pass interviews* and has experience in face tracking, blender, unity, powershell, and tts.
- ****Timeouts Break Agent Workflows!****: A member reported that their agent workflow was crashing due to unhandled **timeout errors** with the **OpenAI endpoint**.
   - It was suggested to catch `WorkflowRuntimeException` or `Exception` instead of `WorkflowTimeoutError`.
- ****Hugging Face vs Ollama: Which LLM is Easier to Configure?****: Members discussed using **Hugging Face** models locally for chat with RAG, with one user suggesting **Ollama** is easier to configure.
   - Despite the debate, helpful links to **Hugging Face Embedding** examples were provided, such as [this notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/huggingface.ipynb).
- ****JSONL Datasets and Git: A Match Made in Heaven or Data Disaster?****: One member pondered the wisdom of storing datasets as **JSONL** files in **Git**, seeking insights into potential downsides.
   - There was no specific answer to this question, but it was mentioned that *Github tracks the updates to every piece of documentation*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/">Local Embeddings with HuggingFace - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/">Hugging Face LLMs - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_cloud_services/blob/main/examples/parse/multimodal/multimodal_rag_slide_deck.ipynb">llama_cloud_services/examples/parse/multimodal/multimodal_rag_slide_deck.ipynb at main · run-llama/llama_cloud_services</a>: Knowledge Agents and Management in the Cloud. Contribute to run-llama/llama_cloud_services development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1353699437750255668)** (1 messages): 

> `Multi-Agent Systems, Program-Wide Backoff Mechanism, Function Calling` 


- **Debate on Triggering Agents via Function Calling**: Members are debating if a single agent triggering other single agents via **function calling** could replace **program-wide backoff mechanisms** in multi-agent systems.
   - They are considering whether these two setups might overlap to achieve the same functionality in certain scenarios.
- **Exploring Alternatives to Backoff Mechanisms**: The discussion focuses on whether using a single agent to trigger others via function calls is a viable alternative to a program-wide backoff mechanism.
   - The goal is to determine if this approach can achieve similar functionality in multi-agent systems, potentially offering a more streamlined solution.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1352766855911047238)** (25 messages🔥): 

> `RAG source return, data retention policy, security information about chat with cohere, sampler settings for Command A, AI assistant powered by Cohere's command-r-plus` 


- **Command-R-Plus Powers New AI Assistant**: A startup founder is building tools for structural biology using an AI assistant powered by **Cohere's command-r-plus**, combined with a **MolStar** molecular viewer ([https://ai.doi.bio](https://ai.doi.bio)).
   - The site currently supports the 'load' command for loading PDB entries into the viewer; for example, say *'Show me 7zzz'*.
- **Data Retention Policy & Security Info Discussed**: A member inquired about **data retention** and **security policies** for **Cohere's chat** feature, specifically if data is used for model training.
   - A Cohere team member responded with links to the [privacy policy](https://cohere.com/privacy), [data usage policy](https://cohere.com/data-usage-policy), and [security policy](https://cohere.com/security), mentioning that users can control data settings in their dashboard.
- **Cohere's Data Privacy and Deployment**: A Cohere team member detailed that their SaaS platform lets users control data directly from their [dashboard](https://dashboard.cohere.com/data-controls), offers **ZDR support** upon request via email, and integrates with major cloud providers (**OCI**, **Bedrock**, **Sagemaker**, **Azure Cloud**).
   - They also provide **on-prem solutions** (details at [https://cohere.com/deployment-options](https://cohere.com/deployment-options)), are **SOC II** and **GDPR** compliant, and adhere to industry standards for data security and privacy.
- **Seeking RAG Replication Resources**: A member is seeking resources to replicate **RAG source return** behavior similar to **notebooklm**, where specific paragraphs are referenced in search results.
   - They are looking for open-source examples related to **chunking** and **data model design**.
- **Command A Sampler Settings Guidance**: A member asked about released recommended **sampler settings for Command A**.
   - Another member suggested starting with a **temperature of 0.7** and adjusting as needed for determinism vs. flexibility; the default temperature is **0.3**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.doi.bio">ai.doi.bio</a>: no description found</li><li><a href="https://cohere.com/security">Security | Cohere</a>: Ensure ultimate AI security and privacy with Cohere&#x27;s enterprise-grade security protocols, robust access controls, and private deployment options. </li><li><a href="https://dashboard.cohere.com/data-controls">Login | Cohere</a>: Login for access to advanced Large Language Models and NLP tools through one easy-to-use API.</li><li><a href="https://cohere.com/privacy">Privacy Policy | Cohere</a>: Cohere Inc. (“Cohere”) values and respects your privacy. We have prepared this privacy policy to explain the manner in which we collect, use and disclose personal information through our Website locat...</li><li><a href="https://cohere.com/deployment-options">Deployment Options - SaaS, Cloud API, Virtual Private Cloud (VPC), On Premise | Cohere</a>: Our solutions provide industry-leading data privacy and security and are designed to meet the diverse needs of organizations seeking to harness the power of generative AI. Whether you’re a start-up or...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1353203487290429522)** (35 messages🔥): 

> `Command models, SSL Errors, API Rate Limits, MongoDB` 


- ****Command** Models Face SSL Issues?**: A member inquired about **Command** models and their potential for generating more human-like responses, while also experiencing **SSL errors**.
   - Another member pointed out that SSL errors are not typically related to the model itself but rather to **untrusted certificates** or network configurations, but could be related to rate limiting.
- **API Spamming Causes SSL Errors?**: A member reported encountering **SSL errors** when rapidly sending requests to the **API**, suspecting it might be due to spamming despite having the py.ssl module properly installed.
   - Another member suggested the issue could stem from **untrusted server certificates**, not client-side problems, and recommended contacting the support team.
- **Suspect API Rate Limit Arises**: A member suspected the **SSL errors** might be related to an undocumented **API rate limit** triggered by spamming requests.
   - Another member noted that rate limits usually return a **429 error code**, however.
- **MongoDB Status Queried**: Switching topics, a member inquired whether another's **MongoDB** was working.
   - The other member stated it was working fine and they used it yesterday.


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1353370236643971123)** (2 messages): 

> `Discord Bot, RAG Pipeline, vnc-lm, Context Augmentation, Docker` 


- **vnc-lm Releases Discord Bot with RAG Integration**: A member released a new version of their Discord bot, **vnc-lm**, featuring a **RAG pipeline** that pulls data from **Wikipedia** and **DuckDuckGo** to augment prompts with additional context.
   - This pipeline adds approximately **500 tokens** to each prompt by appending five chunks of sourced information to improve the model's context, with code available on [GitHub](https://github.com/jake83741/vnc-lm).
- **Search enabled and disabled**: The newly released bot has support for web search.
   - The new search can be enabled with **+ search** and disabled with **+ model**.
- **Versatile Bot Supports Local and Hosted LLMs**: The updated Discord bot now supports every popular local and hosted large language model API, including **Cohere**.
   - The bot can be quickly built using **Docker**, allowing users to easily edit messages and get new responses within Discord.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://open.spotify.com/episode/6a44wSFv8bc1T9x3mEE9Dq?si=tWnXTxqHQbqpky6bWqj0uw&nd=1&dlsi=d20a7ee755104caa">Sancocho con Limon - Quatsch Session 01</a>: FELD.FM · Episode</li><li><a href="https://github.com/jake83741/vnc-lm">GitHub - jake83741/vnc-lm: A Discord bot for large language models. Add Gemini, Sonnet-3.7 DeepSeek R-1, and other models. Easily change models, edit prompts, and enable web search.</a>: A Discord bot for large language models. Add Gemini, Sonnet-3.7 DeepSeek R-1, and other models. Easily change models, edit prompts, and enable web search. - jake83741/vnc-lm
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1352733484468146206)** (33 messages🔥): 

> `Synthetic Data Generation with vllm and deepseek r1, Llama4 Release, Qwen3 MoE, Good Data Problem, PDF Extraction` 


- ****Synthetic Data** Streams from **vllm** and **Deepseek R1****: A member is generating **synthetic data** using **vllm** and **Deepseek R1**, expecting the process to run for a couple of weeks.
   - Training is delayed in anticipation of **Llama4's release** during LlamaCon.
- **Data Quality Conundrums Continue**: Despite years of research, the definition and attainment of *good data* remain elusive for AI labs, even after the recognized importance of datasets like **fineweb** and **lima**.
   - A member expressed frustration over the lack of effective **PDF extraction** tools: *we still don't have amazing PDF extraction and this is making my blood boil*.
- ****LlamaExtract** Tool Launched**: [LlamaIndex](https://www.llamaindex.ai/) launched **LlamaExtract**, a tool for structuring complex documents using genAI-native agents.
   - It adapts the latest models to accurately and reliably structure documents like financial reports and resumes.
- ****DeepSeek-V3** Releases Unhinged**: A member noted the unceremonious release of **DeepSeek-V3** by Deepseek, humorously calling them *unhinged* due to the lack of a proper readme.
   - The model, accessible on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324), has a blank `README.md` but provides access to a playground.
- ****MoEs** Hinted for **Torchtune**?**: A subtle reference was made to the potential inclusion of **Mixture of Experts (MoE)** models in **Torchtune**.
   - The discussion touched on the practical challenges of training such large models, potentially requiring **8-9 TB of VRAM**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jerryjliu0/status/1902880391578653176">Tweet from Jerry Liu (@jerryjliu0)</a>: LlamaExtract is now in public beta 🔥- the leading, genAI-native agent for structured document extraction.We adapt the latest models and tune them so that you can structure even the most complex docum...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1352735360001507479)** (23 messages🔥): 

> `datasets library issue, GRPO LoRA 3B Single Device, vLLM support for data generation, CUDA graphs` 


- **Datasets Library Troubleshoot**: Members found an issue with the **datasets library** and attempted to debug it, with one suggesting upgrading the **datasets version**.
   - One member confirmed that they are on the latest version **3.4.1**.
- **GRPO LoRA Achieves 54% on GMS8K**: The **GRPO LoRA 3B single device** gets to **54%** on GMS8K, according to a member who shares a [link to the pull request](https://github.com/pytorch/torchtune/pull/2467).
   - The member noted that it performs better than expected on novel questions, despite an error of adding extraneous +2 in its calculation.
- **vLLM support lacking for data generation**: Members discussed adding **vLLM support for data generation** but noted difficulties in sharing weights between vLLM and torchtune.
   - One suggested hosting the model in another vLLM process and converting weights, while another mentioned experimenting with a hacky way to make it work on smaller models.
- **CUDA Graphs capture operations**: A member inquired about **CUDA graphs** which captures a whole bunch of GPU operations as a graph and launch them as a single operation.
   - Another member confirmed this and noted that it reduces the overhead to launch CUDA operations from CPU, which reduces GPU idle time.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/2467">GRPO LoRA Single Device by ianbarber · Pull Request #2467 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to[x ] add a new feature fix a bug update tests and/or documentation other (please add here)#2421 - exploring a LoRA recipe.ChangelogWhat are ...

  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1353412469934264491)** (1 messages): 

> `DLCoT Optimizer, Chain-of-Thought Distillation, Token Usage Reduction, DSPy Optimizers` 


- ****DLCoT Optimizer** Launches for Chain-of-Thought**: A member has submitted a [pull request (#8000)](https://github.com/stanfordnlp/dspy/pull/8000) for a new optimizer called **DLCoT** (Deconstructing Long Chain-of-Thought) to the DSPy teleprompt module.
   - It enhances chain-of-thought reasoning by intelligently processing and optimizing long CoT data by segmenting CoT content, removing redundant paths, filtering incorrect chains and reconstructing coherent output.
- ****DLCoT** Slashes Token Usage by 70-90%**: The **DLCoT optimizer** can reduce token usage by **70-90%** while maintaining or improving accuracy across benchmarks.
   - The optimizer works with existing DSPy optimizers like **BootstrapFewShot** and distills down to the most efficient reasoning path.



**Link mentioned**: <a href="https://github.com/stanfordnlp/dspy/pull/8000">Add DLCoT Optimizer for efficient Chain-of-Thought distillation by jmanhype · Pull Request #8000 · stanfordnlp/dspy</a>: Add DLCoT (Deconstructing Long Chain-of-Thought) OptimizerOverviewThis PR adds a new optimizer to the DSPy teleprompt module: the DLCoT (Deconstructing Long Chain-of-Thought) optimizer. This feat...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1353165176161042493)** (20 messages🔥): 

> `DSPy for creative content generation, PAPILLON example, Agentic-Reward-Modeling link, DLCoT Optimizer, MIPROv2` 


- ****DSPy** for creative content generation discussed**: Members are discussing using **DSPy** for optimizing prompts for creative content generation and suggesting to use a *good judge*.
   - One member suggested checking out [PAPILLON](https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) and [Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling) examples.
- ****DLCoT Optimizer** contribution**: A member shared a new contribution, the **DLCoT (Deconstructing Long Chain-of-Thought) Optimizer**, on [GitHub](https://github.com/stanfordnlp/dspy/pull/8000) for efficient Chain-of-Thought distillation.
   - The member encouraged others to check it out and provide feedback.
- **Optimizing Prompt without Examples**: A member is seeking guidance on optimizing a prompt for passage summarization **without examples**, using a working evaluation function and wondered if they should use **COPRO** instead of **MIPROv2**.
   - Another member clarified that example *inputs* are always needed but summaries (labels) are not, if a judge/metric can assess summaries without a reference/label.
- **Fine-Grained Feedback via `dspy.Prediction`**: A member inquired about achieving granular feedback with **Refine**, similar to assertions/suggestions, where specific checks over an output provide targeted feedback.
   - Another member mentioned that in version **2.6.15**, it will be possible to return `dspy.Prediction(score=...., feedback=....)` to offer fine-grained feedback to the module.
- **Multi-Agent Protocol Standard (MCP) in Retrieval**: Members discussed the potential of a multi-agent protocol standard (**MCP**) and its expansion to include retrievers/retrieval augmented generation.
   - The discussion included a shared schema for retrieval results and methods to exchange documents and embeddings, with an aim to streamline data-driven workflows and simplify the combination of multiple models and data sources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/THU-KEG/Agentic-Reward-Modeling">GitHub - THU-KEG/Agentic-Reward-Modeling: Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems</a>: Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems - THU-KEG/Agentic-Reward-Modeling</li><li><a href="https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb">PAPILLON/papillon_tutorial.ipynb at main · Columbia-NLP-Lab/PAPILLON</a>: Code for our paper PAPILLON: PrivAcy Preservation from Internet-based and Local Language MOdel ENsembles - Columbia-NLP-Lab/PAPILLON</li><li><a href="https://github.com/stanfordnlp/dspy/pull/8000">Add DLCoT Optimizer for efficient Chain-of-Thought distillation by jmanhype · Pull Request #8000 · stanfordnlp/dspy</a>: Add DLCoT (Deconstructing Long Chain-of-Thought) OptimizerOverviewThis PR adds a new optimizer to the DSPy teleprompt module: the DLCoT (Deconstructing Long Chain-of-Thought) optimizer. This feat...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1353064669274701935)** (9 messages🔥): 

> `DSPy Modules, Creative Writing Prompts, PAPILLON, Privacy Preservation` 


- **DSPy Module Usage Under Scrutiny**: A member inquired about the correct usage of **DSPy Modules** within the context of generating reports and charts from a **Pandas DataFrame** using **LLMs**.
   - Another member pointed out the difficulty in getting help without a more specific question beyond reviewing a large attached code file, the member then specified *is that the correct way to use DSPy Modules*?
- **Members seek creative writing prompt examples**: A member requested examples for improving **creative writing prompts** or similar cases where there's no clear correct answer.
   - A link to the **PAPILLON GitHub repository** was shared, featuring a tutorial notebook focused on privacy preservation from internet-based and local language model ensembles, [PAPILLON GitHub](https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb).



**Link mentioned**: <a href="https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb">PAPILLON/papillon_tutorial.ipynb at main · Columbia-NLP-Lab/PAPILLON</a>: Code for our paper PAPILLON: PrivAcy Preservation from Internet-based and Local Language MOdel ENsembles - Columbia-NLP-Lab/PAPILLON

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1353437505646755893)** (19 messages🔥): 

> `sops.gz dataset, Tinygrad CUDA port, Meeting #63 Agenda, AMD LLVM progress, ONNX Frontend for Tinygrad` 


- **Track Down `sops.gz` Origins**: A member inquired about the location of the `datasets/sops.gz` dataset used in `speed_compare_cuda_ptx`.
   - Another member shared that the dataset is available in the repo's [extra directory](https://github.com/tinygrad/tinygrad/blob/master/extra/datasets/sops.gz) and generated via the [generate_dataset.sh script](https://github.com/tinygrad/tinygrad/blob/master/extra/optimization/generate_dataset.sh).
- **CUDA Port Ponderings**: A member inquired about the possibility of porting **Tinygrad** to **CUDA GPU** for training.
   - Another member responded with a link to the [README.md](https://github.com/tinygrad/tinygrad/?tab=readme-ov-file#accelerators) file, highlighting supported backends.
- **Meeting Agenda Announced**: The agenda for meeting #63 was announced, covering topics such as **company update**, **quantized DSP**, **BERT**, **scheduler**, **driver**, **tensor cores**, **WebGPU**, **ONNX**, **RetinaNet**, **Torch frontend** and other bounties.
   - Discussion included **test_ops**, **multi GPU training**, **torch compile** and bounties for an **AMD LLVM backend**.
- **AMD LLVM Backend Advancements**: Progress on the **AMD LLVM backend** was reported, including multiple merged pull requests and testing with **Llama3** and **Flux** examples.
   - A pull request is undergoing review.
- **ONNX Frontend Ascends**: A member noted that `tinygrad.frontend.onnx` now exists, expressing intent to focus on **ONNX** preparation this week.
   - Validation of the top 30 **Hugging Face ONNX** repos is a topic.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/extra/datasets/sops.gz">tinygrad/extra/datasets/sops.gz at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/?tab=readme-ov-file#accelerators">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1353147176968126574)** (4 messages): 

> `Disable colored terminal output, tinygrad facades, GPU code generation, OpenCLEmpty guarantees` 


- **Disable colored terminal output in tinygrad**: A member asked if there's a way to disable colored terminal output.
- **Tinygrad has two facades**: Tinygrad has two facades: the **deep learning** part (weights update, tensors, matrix multiplication), and the **compiler** part (GPU code generation and scheduling).
   - The deep learning part is better explained by [Karpathy’s Youtube tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).
- **OpenCL empty values are unguaranteed**: A member reported getting weird output from the [first example in tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/20241231_intro.html).
   - It was clarified that *with OpenCLempty is just empty, there's no guaranteed value*.



**Link mentioned**: <a href="https://mesozoic-egg.github.io/tinygrad-notes/20241231_intro.html">Introduction to the internals</a>: Tutorials on tinygrad

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1353204258391982166)** (9 messages🔥): 

> `Quiz Typos, AgentX Research Track, Remote Research Mentorship, Unpaid Research` 


- **Quiz Title Typo Causes Confusion**: A member reported a typo in the title of **Quiz 7**, causing confusion when checking answers for **Quiz 6**.
   - Another member acknowledged the catch and thanked the reporter.
- **AgentX Research Track Application Live**: Selected students will receive mentorship from **Berkeley postdocs/mentors** on an **AgentX Research Track project** with applications due **March 26th** at **11:59pm PDT**.
   - Mentorship is not required to join or succeed in **AgentX**, and labs plus the Certificate Declaration form will be released in April as seen in the [attached image](https://cdn.discordapp.com/attachments/1280370030609170494/1353204258450964544/image.png?ex=67e2c76c&is=67e175ec&hm=1fb895b885ce732fd7e5b99b8ff24c55286d5).
- **Research Track is Confirmed to be Remote and Unpaid**: A member confirmed that the **AgentX Research Track mentorship** will be conducted remotely.
   - Another member clarified that the mentorship is not paid, with mentors simply providing guidance on the research project.


  

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
