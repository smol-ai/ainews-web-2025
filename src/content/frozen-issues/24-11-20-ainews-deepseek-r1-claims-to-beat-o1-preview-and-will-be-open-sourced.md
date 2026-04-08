---
id: cf470da6-b5f6-4356-ac97-101745d71f29
title: DeepSeek-R1 claims to beat o1-preview AND will be open sourced
date: '2024-11-21T02:41:02.660253Z'
original_slug: ainews-deepseek-r1-claims-to-beat-o1-preview-and
description: >-
  **DeepSeek** has released **DeepSeek-R1-Lite-Preview**, an open-source
  reasoning model achieving **o1-preview-level performance** on math benchmarks
  with transparent thought processes, showing promise in real-time
  problem-solving. **NVIDIA** reported a record **$35.1 billion** revenue in Q3
  with **112% year-on-year data center growth**, driven by **Hopper** and
  **Blackwell architectures**, the latter offering **2.2x performance
  improvement**. **Google DeepMind** introduced **AlphaQubit**, a quantum
  computing system improving error correction and outperforming leading
  decoders, though challenges remain in scaling and speed. The AI community
  continues to focus on **reasoning models**, **benchmarking**, and **quantum
  error correction** advancements.
companies:
  - deepseek
  - nvidia
  - google-deepmind
models:
  - deepseek-r1-lite-preview
  - o1-preview
  - hopper
  - blackwell
  - alphaqubit
topics:
  - reasoning
  - benchmarking
  - quantum-error-correction
  - quantum-computing
  - model-performance
  - model-release
people:
  - yann-lecun
---


<!-- buttondown-editor-mode: plaintext -->**Whalebros are all you need.**

> AI News for 11/20/2024-11/21/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**217** channels, and **1837** messages) for you. Estimated reading time saved (at 200wpm): **197 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Ever since o1 was introduced (our coverage [here](https://buttondown.com/ainews/archive/ainews-o1-openais-new-general-reasoning-models/), [here](https://buttondown.com/ainews/archive/ainews-learnings-from-o1-ama/), and [here](https://buttondown.com/ainews/archive/ainews-o1-destroys-lmsys-arena-qwen-25-kyutai/)), the race has been on for an "open" reproduction. 2 months later, with honorable mentions to [Nous Forge Reasoning API](https://x.com/NousResearch/status/1856417883934601246) and [Fireworks f1](https://fireworks.ai/blog/fireworks-f1), [DeepSeek appear](https://x.com/deepseek_ai/status/1859200141355536422) to have made the first convincing attempt that 1) has *BETTER* benchmark results than o1-preview and 2) has a publicly available demo rather than waitlist.

![image.png](https://assets.buttondown.email/images/7225b5d6-d19f-4bda-b827-417771eab45b.png?w=960&fit=max)

Benchmarks wise, it doesn't beat o1 across the board, but does well on important math benchmarks and at-least-better-than-peers on all but GPQA Diamond.

![image.png](https://assets.buttondown.email/images/62d1286c-683b-4083-a07a-cbb5cc21d1e0.png?w=960&fit=max)

Also importantly, they appear to have replicated the similar inference-time-scaling performance improvements mentioned by OpenAI, but this time with an actual x-axis:

![image.png](https://assets.buttondown.email/images/be08c1f6-c666-4032-bafc-38bd05fa5da9.png?w=960&fit=max)

As for the "R1-Lite" naming, [rumor is](https://x.com/nrehiew_/status/1859265550767067518) (based on [wechat announcements](https://x.com/phill__1/status/1859263165000729024)) it is based on DeepSeek's existing V2-Lite model which is only a 16B MoE with 2.4B active params - meaning that if they manage to scale it up, "R1-full" will be an absolute monster. 

One notable result is that it has done (inconsistently) well on [Yann LeCun's pet 7-gear question](https://x.com/nrehiew_/status/1859268539770900923).

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

**1. NVIDIA Financial Updates and Market Insights**

- **NVIDIA Reports Record Revenue in Q3**: [@perplexity_ai discusses the insights](https://twitter.com/perplexity_ai/status/1859371790826701288) from NVIDIA's Q3 earnings call, highlighting a **record revenue of $35.1 billion**, a **17% increase from the previous quarter**. Key growth drivers include **strong data center sales** and demand for NVIDIA's **Hopper and Blackwell architectures**. The company anticipates continuing growth with projections of $37.5 billion in Q4.
  
- **Detailed Performance During Earnings Call**: [Another update from @perplexity_ai](https://twitter.com/perplexity_ai/status/1859361535577268698) further outlines that **data center revenue reached $30.8 billion**, marking a **112% year-on-year increase**. The Blackwell architecture reportedly offers a **2.2x performance improvement over Hopper**.

**2. DeepSeek-R1-Lite-Preview: New Reasoning Model Developments**

- **Launch of DeepSeek-R1-Lite-Preview**: [@deepseek_ai is excited about the release](https://twitter.com/deepseek_ai/status/1859200141355536422) of **DeepSeek-R1-Lite-Preview**, which offers **o1-preview-level performance** on MATH benchmarks and a transparent thought process. The model aims to have an open-source version available soon.

- **Evaluation of DeepSeek-R1-Lite-Preview**: [Multiple users, such as @omarsar0](https://twitter.com/omarsar0/status/1859373413439066590), discuss its capabilities, including **math reasoning improvements** and challenges in coding tasks. Despite some mishaps, the model shows promise in real-time problem-solving and reasoning.

**3. Quantum Computing Progress with AlphaQubit**

- **AlphaQubit Collaboration with Google**: [@GoogleDeepMind introduces AlphaQubit](https://twitter.com/GoogleDeepMind/status/1859273133234192598), a system designed to improve error correction in quantum computing. This system outperformed leading algorithmic decoders and shows potential in scale-up scenarios.

- **Challenges in Quantum Error Correction**: Despite these advancements, [additional insights from Google DeepMind](https://twitter.com/GoogleDeepMind/status/1859273153534681383) note ongoing issues with scaling and speed, highlighting the goal to make **quantum computers more reliable**.

**4. Developments in GPT-4o and AI Creative Enhancements**

- **GPT-4o's Enhanced Creative Writing**: [@OpenAI notes updates](https://twitter.com/OpenAI/status/1859296125947347164) in GPT-4o's ability to produce more natural, engaging content. [User comments, such as from @gdb,](https://twitter.com/gdb/status/1859329768707195161) highlight improvements in working with files and offering deeper insights.

- **Chatbot Arena Rankings Update**: [@lmarena_ai shares excitement](https://twitter.com/lmarena_ai/status/1859307979184689269) over ChatGPT-4o reaching the #1 spot, surpassing Gemini and Claude models with significant improvements in creative writing and technical performance.

**5. AI Implementations and Tools**

- **LangChain and LlamaIndex Systems**: [@LangChainAI announces updates](https://twitter.com/LangChainAI/status/1859250598698422392) to the platform focusing on observability, evaluation, and prompt engineering. They emphasize seamless integration, offering developers comprehensive tools to refine LLM-based applications.

- **AI Game Development Courses**: [@togethercompute introduces a course](https://twitter.com/togethercompute/status/1859315685974999413) on building AI-powered games, in collaboration with industry leaders. It focuses on integrating LLMs for immersive game creation.

**6. Memes/Humor**

- **High School AI Nostalgia**: [@aidan_mclau humorously reflects](https://twitter.com/aidan_mclau/status/1859378924771254734) on using AI to complete philosophy homework, showcasing a light-hearted take on AI's educational uses.

- **Chess Meme**: [@BorisMPower engages in a chess meme thread](https://twitter.com/BorisMPower/status/1859070338111599005), contemplating strategic moves and decision-making within the game context.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek R1-Lite matches o1-preview in math benchmarks, open source coming soon**

- **DeepSeek-R1-Lite Preview Version Officially Released** ([Score: 189, Comments: 64](https://reddit.com/r/LocalLLaMA/comments/1gvnhob/deepseekr1lite_preview_version_officially_released/)): **DeepSeek** released their new **R1 series inference models**, trained with **reinforcement learning**, featuring extensive reflection and verification capabilities through **chain of thought reasoning** that can span **tens of thousands of words**. The models achieve performance comparable to **o1-preview** in **mathematics**, **coding**, and **complex logical reasoning** tasks while providing transparent reasoning processes at [chat.deepseek.com](http://chat.deepseek.com).
  - **DeepSeek-R1-Lite** is currently in development, with the official announcement confirming it's web-only without API access. The company plans to **open-source** the full **DeepSeek-R1 model**, release technical reports, and deploy API services as stated in their [tweet](https://x.com/deepseek_ai/status/1859200141355536422).
  - Initial user testing shows impressive performance in **mathematics** with detailed reasoning steps, though some note longer response times compared to **o1-preview**. The model is estimated to be **15B parameters** based on previous DeepSeek releases.
  - Community response highlights the rapid advancement of **Chinese AI labs** despite GPU restrictions, with users noting the model's transparent thought process could benefit open-source development. Several users confirmed strong performance on **AIME & MATH benchmarks**.


- **[Chinese AI startup StepFun up near the top on livebench with their new 1 trillion param MOE model](https://i.redd.it/tqgyvi01ky1e1.jpeg)** ([Score: 264, Comments: 74](https://reddit.com/r/LocalLLaMA/comments/1gvdnzi/chinese_ai_startup_stepfun_up_near_the_top_on/)): **StepFun**, a **Chinese AI startup**, developed a **1 trillion parameter Mixture-of-Experts (MOE) model** that achieved competitive scores on **livebench**, a real-time AI model leaderboard. The model's specific performance metrics and technical details were not disclosed in the source material.
  - **Livebench scores** show the model is currently **underperforming** relative to its size, with users noting it's being beaten by smaller models like **o1 mini** (estimated at **70-120B parameters**) and showing particularly low **math scores**.
  - The model appears to be in early training stages, with discussion around "**Step 2**" potentially indicating its second training phase. Users speculate the underwhelming performance is due to being **heavily undertrained** rather than architectural limitations.
  - Discussion focused on the model's **MoE architecture** and deployment strategy, with experts noting that each **transformer layer** requires its own set of experts, leading to substantial **GPU-to-GPU communication** needs during inference and training.


**Theme 2. Sophisticated Open Source LLM Tools: Research Assistant & Memory Frameworks**

- **I Created an AI Research Assistant that actually DOES research! Feed it ANY topic, it searches the web, scrapes content, saves sources, and gives you a full research document + summary. Uses Ollama (FREE) - Just ask a question and let it work! No API costs, open source, runs locally!** ([Score: 487, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1gvlzug/i_created_an_ai_research_assistant_that_actually/)): **Automated-AI-Web-Researcher** is a **Python-based** tool that uses **Ollama** and **local LLMs** to conduct comprehensive web research, automatically generating up to **5 specific research focuses** from a single query, continuously searching and scraping content while saving sources, and creating detailed research documents with summaries. The project, available on [GitHub](https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama), runs entirely locally using models like **phi3:3.8b-mini-128k-instruct** or **phi3:14b-medium-128k-instruct**, features pause/resume functionality, and enables users to ask follow-up questions about the gathered research content.
  - Users reported mixed success with different **LLMs** - while some had issues with **Llama3.2-vision:11b** and **Qwen2.5:14b** generating empty summaries, others successfully used **mistral-nemo 12B** achieving **38000 context length** with **16GB VRAM** at **3% CPU / 97% GPU** usage.
  - Several technical suggestions were made including ignoring **robots.txt**, adding support for **OpenAI API** compatibility (which was later implemented via PR), and restructuring the codebase with a "lib" folder and proper configuration management using tools like **pydantic** or **omegaconf**.
  - Discussion around the tool's purpose emphasized its value in finding and summarizing real research rather than generating content, with concerns raised about source validation and factual accuracy of web-scraped information.


- **Agent Memory** ([Score: 64, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1gvhpjj/agent_memory/)): **LLM agent memory frameworks** are compared across multiple GitHub projects, with key implementations including **Letta** (based on MemGPT paper), **Memoripy** (supports Ollama and OpenAI), and **Zep** (maintains temporal knowledge graphs). Several frameworks support **local models** through **Ollama** and **vLLM**, though many assume **GPT access** by default, with varying levels of compatibility for open-source alternatives.

  - The comparison includes both active projects like **cognee** (for document ingestion) and **MemoryScope** (featuring memory consolidation), as well as development resources such as **LangGraph Memory Service** template and **txtai** for RAG implementations, with most frameworks offering **OpenAI-compatible API** support through tools like **LiteLLM**.
  - **Vector-based memory systems** use proximity and reranking to determine relevance, contrasting with simple **keyword activation** systems like those used in **Kobold** or **NovelAI**. The vector approach maps concepts spatially (e.g., "Burger King" closer to food-related terms than "King of England") and uses reranking through either small neural nets or direct AI evaluation.
  - Memory frameworks differ primarily in their handling of context injection - from automated to manual approaches - with more complex systems incorporating **knowledge graphing** and **decision trees**. Memory processing can become resource-intensive, sometimes requiring more tokens than the actual conversation.
  - The field of **LLM memory systems** remains experimental with no established best practices, ranging from basic lorebook-style implementations to sophisticated context-aware solutions. Simple systems require more human oversight to catch errors, while complex ones offer better robustness against contextual mistakes.


**Theme 3. Hardware & Browser Optimization: Pi GPU Acceleration & WebGPU Implementations**

- **[LLM hardware acceleration—on a Raspberry Pi (Top-end AMD GPU using a low cost Pi as it's base computer)](https://www.youtube.com/watch?v=AyR7iCS7gNI)** ([Score: 53, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1gvdrvj/llm_hardware_accelerationon_a_raspberry_pi_topend/)): **Raspberry Pi** configurations can run **Large Language Models (LLMs)** with **AMD GPU acceleration** through **Vulkan** graphics processing. This hardware setup combines the cost-effectiveness of a **Raspberry Pi** with the processing power of high-end **AMD GPUs**.
  - **Token rates** of **40 t/s** were achieved using a **6700XT** GPU with **Vulkan** backend, compared to **55 t/s** using an **RTX 3060** with **CUDA**. The lack of **ROCm** support on **ARM** significantly limits performance potential.
  - A complete **Raspberry Pi** setup costs approximately **$383 USD** (excluding GPU), while comparable **x86** systems like the **ASRock N100M** cost **$260-300**. The **Intel N100** system draws only **5W** more power while offering better compatibility and performance.
  - Users note that **AMD** could potentially create a dedicated product combining a basic **APU** with high **VRAM GPUs** in a **NUC-like** form factor. The upcoming **Strix Halo** release may test market demand, though alternatives like dual **P40s** for **$500** remain competitive.


- **In-browser site builder powered by Qwen2.5-Coder** ([Score: 55, Comments: 8](https://reddit.com/r/LocalLLaMA/comments/1gv73fn/inbrowser_site_builder_powered_by_qwen25coder/)): An **AI site builder** running in-browser uses **WebGPU**, **OnnxRuntime-Web**, **Qwen2.5-Coder**, and **Qwen2-VL** to generate code from text, images, and voice input, though only text-to-code is currently live due to performance constraints. The project implements **Moonshine** for speech-to-text conversion and includes code examples for integration on [GitHub](https://github.com/pdufour/llm-coder/blob/main/src/hooks/useSpeech.js) and [Huggingface](https://huggingface.co/spaces/pdufour/Qwen2-VL-2B-Instruct-ONNX-Q4-F16/blob/main/index.js), with performance currently limited by GPU capabilities and primarily tested on **Mac** systems.
  - Developer details challenges in **model conversion**, sharing their process through [export documentation](https://huggingface.co/pdufour/Qwen2-VL-2B-Instruct-ONNX-Q4-F16/blob/main/EXPORT.md) and a custom **Makefile**, noting issues with **mixed data types** and **memory management** that made the project particularly difficult.
  - Community feedback highlights interest in testing the system on **Linux** with **NVIDIA RTX** hardware, with users also reporting **UI contrast issues** on **iPhone** devices due to similar dark background colors.


**Theme 4. Model Architectures: Analysis of GPT-4, Gemini & Other Closed Source Models**

- **Closed source model size speculation** ([Score: 52, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1gve7sk/closed_source_model_size_speculation/)): The post analyzes parameter counts of **closed-source LLMs**, suggesting that **GPT-4 Original** has **280B active parameters** and **1.8T overall**, while newer versions like **GPT-4 Turbo** and **GPT-4o** have progressively smaller active parameter counts (**~93-94B** and **~28-32B** respectively). The analysis draws connections between model architectures and pricing, linking **Microsoft's Grin MoE** [paper](https://arxiv.org/pdf/2409.12136) to **GPT-4o Mini** (**6.6B-8B** active parameters), and comparing **Gemini Flash** versions (**8B**, **32B**, and **16B** dense) with models like **Qwen** and architectures from **Hunyuan** and **Yi Lightning**.
  - **Qwen 2.5**'s performance-to-size ratio supports the theory of smaller active parameters in modern models, particularly with **MoE architecture** and closed-source research advances. The discussion suggests **Claude** may be less efficient than **OpenAI** and **Google** models.
  - **Gemini Flash**'s **8B** parameter count likely includes the vision model, making the core language model approximately **7B parameters**. The model's performance at this size is considered notably impressive.
  - Community estimates suggest **GPT-4 Turbo** has **~1T** parameters (**100B** active) and **GPT-4o** has **~500B** (**50B** active), while **Yi-Lightning** is likely smaller based on its low pricing and reasoning capabilities. **Step-2** is estimated to be larger due to higher pricing (**$6/M** input, **$20/M** output).
- **[Judge Arena Leaderboard: Benchmarking LLMs as Evaluators](https://i.redd.it/rcrq5uh6r02e1.png)** ([Score: 33, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1gvl5x5/judge_arena_leaderboard_benchmarking_llms_as/)): **Judge Arena Leaderboard** aims to benchmark **Large Language Models (LLMs)** on their ability to evaluate and judge other AI outputs. Due to insufficient context in the post body, no specific details about methodology, metrics, or participating models can be included in this summary.
  - **Claude 3.5 Sonnet** initially led the rankings in the **Judge Arena** leaderboard, but subsequent updates showed significant volatility with **7B models** rising to top positions among open-source entries. The rankings showed compression from an **ELO spread** of ~400 points to ~250 points after **1197 votes**.
  - Community members questioned the validity of results, particularly regarding **Mistral 7B (v0.1)** outperforming **GPT-4**, **GPT-3.5**, and **Claude 3 Haiku**, with high margin of error (~100 ELO points) cited as a potential explanation.
  - Critics highlighted limitations in the **judgment prompt**, suggesting it lacks concrete evaluation criteria and depth, while the instruction to ignore response length could paradoxically influence assessors through the "pink elephant effect".


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. Live Demo Shows Real-Time AI Facial Recognition Raising Privacy Alarms**

- **[This Dutch journalist demonstrates real-time AI facial recognition technology, identifying the person he is talking to.](https://v.redd.it/lb0h1g0st12e1)** ([Score: 2523, Comments: 304](https://reddit.com/r/ChatGPT/comments/1gvo603/this_dutch_journalist_demonstrates_realtime_ai/)): **Dutch journalist** demonstrates **real-time facial recognition AI** capabilities by identifying individuals during live conversations. No additional context or technical details were provided about the specific technology or implementation used.
  - The top comment emphasizes **privacy concerns**, suggesting to "**post no photos online anywhere attached to actual names**" with **457** upvotes. Multiple users discuss continuing to wear **masks** and methods to avoid facial recognition.
  - Discussion reveals this likely uses **Pimeyes** or similar technology, with users noting that **Clearview AI** has even more advanced capabilities that can "*find your face in a crowd at a concert*". Several users point out that the demonstration likely involves a second person doing manual searches.
  - Users debate the societal implications, with some calling it a "**threat to democracy and freedom**" while others discuss practical applications like **car sales**. The conversation includes concerns about government surveillance and data privacy, particularly in reference to **China** and other nations.


**Theme 2. CogVideoX 1.5 Image-to-Video: Quality vs Performance Trade-offs**

- **[Comparison of CogvideoX 1.5 img2vid - BF16 vs FP8](https://v.redd.it/v531xs8o412e1)** ([Score: 165, Comments: 49](https://reddit.com/r/StableDiffusion/comments/1gvm302/comparison_of_cogvideox_15_img2vid_bf16_vs_fp8/)): **CogVideoX 1.5** post lacks sufficient context or content to generate a meaningful technical summary about the comparison between **BF16** and **FP8** implementations. No details were provided in the post body to analyze quality differences between these numerical formats.
  - **Performance metrics** show significant differences: **BF16** takes **12m57s** vs **FP8** at **7m57s** on an **RTX 3060 12GB** with **24 frames** at **1360x768**. **BF16** requires **CPU offload** due to **OOM errors** but delivers more stable results.
  - **CogVideoX 1.5** faces quantization challenges, unable to run in **FP16**. Among available options, **TorchAO FP6** provides best quality results, while **FP8DQ** and **FP8DQrow** offer faster performance on **RTX 4090** due to **FP8 scaled matmul**.
  - Installation on **Windows** requires specific setup using [TorchAO v0.6.1](https://github.com/pytorch/ao/releases/tag/v0.6.1) with code modification in `base.h` file, changing `FragM` definition to `Vec<unsigned int, 1>`.


**Theme 3. 10 AI Agents Collaborate to Write Novel in Real-Time**

- **[A Novel Being Written in Real-Time by 10 Autonomous AI Agents](https://i.redd.it/dfxwtvmpg12e1.png)** ([Score: 277, Comments: 153](https://reddit.com/r/ChatGPT/comments/1gvn049/a_novel_being_written_in_realtime_by_10/)): **Ten autonomous AI agents** collaborate in real-time to write a novel, though no additional details about the process, implementation, or results were provided in the post body. The concept suggests an experiment in multi-agent creative writing and AI collaboration, but without further context, specific technical details cannot be summarized.
  - Users express significant **skepticism about AI-generated long-form content**, with many pointing out that **ChatGPT struggles with coherence beyond a few pages** and frequently **forgets plot points and characters**. The top comment with **178 upvotes** emphasizes this limitation.
  - The author explains their solution to maintaining narrative coherence through a **file-based coordination system** where multiple agents access a **global map**, **content summaries**, and **running change logs** rather than relying on a single context window. The system is currently in the preparation and structuring phase using **Qwen 2.5**.
  - Several users debate the **artistic value** and **purpose** of AI-generated novels, arguing that literature is fundamentally about expressing human experience and creating human connections. Critics note that AI models like **ChatGPT** and **Claude** would likely avoid controversial topics that make novels interesting.


**Theme 4. StepFun's 1T Param Model Rises in LiveBench Rankings**

- **[Chinese AI startup StepFun up near the top on livebench with their new 1 trillion param MOE model](https://i.redd.it/p01x5ci4j02e1.png)** ([Score: 29, Comments: 0](https://reddit.com/r/OpenAI/comments/1gvkjib/chinese_ai_startup_stepfun_up_near_the_top_on/)): **StepFun**, a **Chinese AI startup**, has developed a **1 trillion parameter Mixture-of-Experts (MOE) model** that ranks among the top performers on **livebench**. The model's performance demonstrates increasing competition in large-scale AI model development from Chinese companies.

- **[Microsoft CEO says that rather than seeing AI Scaling Laws hit a wall, if anything we are seeing the emergence of a new Scaling Law for test-time (inference) compute](https://v.redd.it/c8tfecx1y22e1)** ([Score: 99, Comments: 40](https://reddit.com/r/OpenAI/comments/1gvsw59/microsoft_ceo_says_that_rather_than_seeing_ai/)): **Microsoft's CEO** discusses observations about **AI scaling laws**, noting that instead of encountering computational limits, evidence suggests a new pattern emerging specifically for **test-time inference compute**. The lack of specific details or quotes in the post body limits further analysis of the claims or supporting evidence for this observation.
  - The discussion reveals that **test-time inference compute** involves allowing models to "think" longer and iterate on outputs rather than accepting first responses, with accuracy scaling **logarithmically** with thinking time. This represents a second scaling factor alongside traditional training compute scaling.
  - Several users, including **Pitiful-Taste9403**, interpret this as evidence that **parameter scaling** has hit limitations, causing companies to focus on inference optimization as an alternative path forward for AI advancement.
  - The term "**scaling law**" sparked debate, with users comparing it to **Moore's Law**, suggesting it's more of a trend than a fundamental law. Some expressed skepticism about the economic implications of these developments for average people.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-mini

**Theme 1. Custom Model Deployments Take Center Stage**

- [**Deploy Custom AI Models on Hugging Face**](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler): Developers are now able to deploy tailored **AI models** on Hugging Face using a `handler.py` file, allowing for customized pre- and post-processing.
  
  - This advancement leverages **Hugging Face endpoints** to enhance model flexibility and integration into diverse applications.
- [**DeepSeek-R1-Lite-Preview Matches OpenAI's o1-Preview Performance**](https://x.com/deepseek_ai/status/1859200141355536422): **DeepSeek** launches **R1-Lite-Preview**, achieving **o1-preview-level performance** on **AIME** & **MATH** benchmarks.
  
  - This model not only mirrors **OpenAI's** advancements but also introduces a **transparent reasoning process** accessible in real-time.
- [**Tencent Hunyuan Model Fine-Tuning Now Accessible**](https://huggingface.co/spaces/tencent/Hunyuan-Large): Users can fine-tune the [**Tencent Hunyuan model**](https://huggingface.co/tencent/Tencent-Hunyuan-Large) with resources like the [GitHub repository](https://github.com/Tencent/Tencent-Hunyuan-Large) and [official demo](https://huggingface.co/spaces/tencent/Hunyuan-Large).
  
  - This facilitates enhanced customization for various NLP tasks, expanding the model's applicability.

**Theme 2. AI Model Performance and Optimization Soars**

- [**SageAttention2 Doubles Inference Speed**](https://arxiv.org/html/2411.10958v1): The [**SageAttention2 Technical Report**](https://arxiv.org/html/2411.10958v1) reveals a method for **4-bit matrix multiplication**, achieving a **2x speedup** over FlashAttention2 on **RTX40/3090** GPUs.
  
  - This innovation serves as a drop-in replacement, significantly accelerating **inference** without sacrificing accuracy.
- [**GPT-4o Gets Creative Boost and File Handling Enhancements**](https://x.com/openai/status/1859296125947347164?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): **OpenAI** updates **GPT-4o**, elevating its **creative writing** abilities and improving **file handling** for deeper insights.
  
  - The revamped model regains top spots in categories like **coding** and **creative writing** within chatbot competitions.
- [**Model Quantization Discussed for Performance Gains**](https://github.com/locuslab/wanda?tab=readme-ov-file#zero-shot-evaluation): Users express concerns that **quantization** adversely affects model performance, preferring original models over quantized versions.
  
  - Suggestions include clearer disclosures from providers like **OpenRouter** and exploring modifications in evaluation libraries to accommodate pruned models.

**Theme 3. Innovative AI Research Paves New Paths**

- [**ACE Method Enhances Model Control**](https://arxiv.org/abs/2411.09003): **EleutherAI** introduces the **ACE (Affine Concept Editing)** method, treating concepts as affine functions to better control **model responses**.
  
  - Tested on models like **Llama 3 70B**, ACE outperforms previous techniques in managing **refusal behavior** across harmful and harmless prompts.
- [**Scaling Laws Reveal Low-Dimensional Capability Space**](https://arxiv.org/abs/2405.10938): A new paper suggests that **language model performance** is influenced more by a **low-dimensional capability space** than solely by scaling across multiple dimensions.
  
  - **Marius Hobbhahn** from **Apollo** champions the advancement of evaluation science, emphasizing rigorous **model assessment** practices.
- [**Generative Agents Simulate Over 1,000 Real Individuals**](https://arxiv.org/abs/2411.10109): A novel architecture effectively simulates the attitudes and behaviors of **1,052 real people**, achieving **85% accuracy** on the General Social Survey.
  
  - This reduces accuracy biases across **racial and ideological groups**, offering robust tools for exploring individual and collective behaviors in social science.

**Theme 4. AI Tools Integration and Community Support Flourish**

- [**Aider's Setup Challenges Resolved with Force Reinstall**](https://aider.chat/docs/troubleshooting/edit-errors.html): Users facing setup issues with **Aider**, particularly with API keys and environment variables, found success by performing a **force reinstall**.
  
  - This solution streamlined the setup process, enabling smoother integration of **DeepSeek-R1-Lite-Preview** and other models.
- [**LM Studio Navigates Hardware Limitations with Cloud Solutions**](https://github.com/unslothai/unsloth/wiki#-we-are-hiring): Members discuss running **DeepSeek v2.5 Lite** on limited hardware, emphasizing the need for GPUs with at least **24GB VRAM**.
  
  - Cloud-based hardware rentals are explored as cost-effective alternatives, offering high-speed model access without local hardware constraints.
- [**Torchtune's Adaptive Batching Optimizes GPU Utilization**](https://github.com/pytorch/torchtune/pull/2035): Implementation of **adaptive batching** in **Torchtune** aims to maximize **GPU utilization** by dynamically adjusting batch sizes to prevent **OOM errors**.
  
  - This feature is suggested to be integrated as a flag in future recipes, enhancing training efficiency and resource management.

**Theme 5. Cutting-Edge AI Developments Address Diverse Challenges**

- [**LLMs Exhibit Intrinsic Reasoning Without Explicit Prompting**](https://arxiv.org/abs/2402.10200): Research demonstrates that **large language models (LLMs)** can display reasoning paths similar to **chain-of-thought (CoT)** without explicit prompting by tweaking the decoding process.
  
  - Adjusting to consider **top-$k$ alternative tokens** uncovers the inherent reasoning abilities of LLMs, reducing reliance on manual prompt engineering.
- [**Perplexity AI Introduces Shopping Feature Amid API Challenges**](https://www.perplexity.ai/page/nvidia-ai-chips-overheat-SRXQJH9yQ8ebTG_KeAT46A): **Perplexity AI** launches a new **Shopping** feature, sparking discussions about its exclusivity to the US market while users face issues with **API response consistency**.
  
  - Despite being **Pro** users, some members express frustration over limitations, leading to increased reliance on alternatives like **ChatGPT**.
- [**OpenRouter Tackles Model Description and Caching Clarifications**](https://x.com/natolambert/status/1859255627882664034?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): Users identify discrepancies in the **GPT-4o** model descriptions on **OpenRouter**, prompting quick fixes to model cards.
  
  - Clarifications are sought regarding **prompt caching** policies across different providers, with comparisons between **Anthropic** and **OpenAI** protocols.

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Deploying Custom AI Models Now Possible**: A member discovered that custom **AI models** can be deployed on Hugging Face using a `handler.py` file, enabling tailored pre- and post-processing of models.
  
  - This process involves specifying handling methods for requests and responses, enhancing customization via [Hugging Face endpoints](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler).
- **New Paper on AI Security Insights Released**: AI researchers at **Redhat/IBM** published a paper addressing the security implications of publicly available AI models, focusing on risk and lifecycle management.
  
  - The paper outlines strategies to improve security for developers and users, aiming to establish more standardized practices within the AI community. [View the paper](https://huggingface.co/papers/2411.12275).
- **Automated AI Research Assistant Takes Off**: An **Automated AI Researcher** was created using local LLMs to generate research documents in response to user queries.
  
  - The system employs web scraping to compile information and produce topic-relevant summaries and links, making research more accessible.
- **LangGraph Learning Initiatives**: User `richieghost` initiated learning around **LangGraph**, discussing its applications and developments in the community.
  
  - This highlights the ongoing interest in integrating graph-based techniques within AI models.
- **Semantic Search Challenges with Ada 002**: **Semantic search** using OpenAI's **Ada 002** is prioritizing dominant topics, resulting in less prominent but relevant sentences receiving lower rankings.
  
  - Users are seeking alternatives to **semantic search** to improve information extraction effectiveness.

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **o1 Release Rumblings**: Speculation is rife that **OpenAI's o1 model** may be released imminently, potentially aligning with **DevDay Singapore**, although these rumors remain unconfirmed.
  
  - A member noted, *'Wednesday would be a weird day for it,'* highlighting the community's vigilant anticipation despite uncertainties.
- **DeepSeek's RL Drive**: Discussions around **DeepSeek Prover** revealed interest in their application of **reinforcement learning**, with members anticipating a possible paper release despite challenges related to model size and performance.
  
  - The community is contemplating delays in full release due to these performance hurdles.
- **GPT-4o Gains Ground**: **OpenAI** announced an update to **GPT-4o**, enhancing its creative writing capabilities and file handling, which propelled it back to the top in performance categories such as creative writing and coding within a chatbot competition.
  
  - This update underscores **GPT-4o**'s improved relevance and readability, as detailed in [OpenAI's official tweet](https://x.com/openai/status/1859296125947347164?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ).
- **LLM Learning Loops**: Recent insights suggest that how **LLMs memorize training examples** significantly affects their generalization capabilities, with models first understanding concepts before memorization leading to better test accuracy predictions.
  
  - **Katie Kang** shared that this method allows for predicting test outcomes based solely on training dynamics.
- **NeurIPS NLP Nixed**: Concerns have been raised about **NeurIPS D&B reviewers** dismissing projects focused on **Korean LLM evaluation**, citing that similar efforts already exist in Chinese.
  
  - Community members argue that each language requires tailored models, emphasizing the importance of inclusivity in **NLP development** as highlighted in [Stella Biderman's tweet](https://x.com/blancheminerva/status/1859271409429795083?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ).

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Fine-tuning Finesse for LLMs**: A user successfully exported **Llama 3.1** for a **local project** to [Hugging Face](https://github.com/unslothai/unsloth) leveraging the **16-bit version** for enhanced performance in building RAG applications.
  
  - Members recommended using the **16-bit version** to optimize **fine-tuning** capabilities, ensuring better resource management during model development.
- **SageAttention2 Speeds Up Inference**: The [SageAttention2 Technical Report](https://arxiv.org/html/2411.10958v1) introduced a method for **4-bit matrix multiplication** that achieves a **2x speedup** over FlashAttention2.
  
  - With support for **RTX40/3090** hardware, SageAttention2 serves as a drop-in replacement for **FlashAttention2**, enhancing **inference acceleration** without compromising metric fidelity.
- **Training Llama Models**: Multiple members shared experiences training different **Llama models**, noting varying success based on **model parameters** and **dataset sizes**.
  
  - Suggestions included starting with base models and tuning **training steps** for optimal **performance**.
- **Enhancing Performance with Multi-GPU Training**: Users are exploring **multi-GPU training** capabilities with Unsloth, currently unavailable but expected to be released soon.
  
  - Strategies like utilizing **Llama Factory** for managing multiple GPUs were discussed to prepare for the upcoming feature.

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Setup Challenges Resolved**: Users faced issues with **Aider's setup**, specifically regarding [API keys](https://aider.chat/docs/troubleshooting/edit-errors.html) and environment variables, leading some to attempt reinstalling components.
  
  - One user reported that performing a **force reinstall** successfully resolved the setup challenges.
- **DeepSeek Impresses with Performance**: **DeepSeek-R1-Lite-Preview** matches **o1-preview** performance on AIME & MATH benchmarks, offering faster response times compared to previous models.
  
  - The model's transparent reasoning process enhances its effectiveness for coding tasks by allowing users to observe its thought process in real-time.
- **Concerns Over OpenRouter's Model Quality**: Users expressed dissatisfaction with **OpenRouter** utilizing quantized versions of open-source models, raising doubts about their performance on the **Aider Leaderboard**.
  
  - There were calls for clearer warnings on the leaderboard regarding potential performance discrepancies when using OpenRouter's quantized models.
- **Impact of Model Quantization Discussed**: Quantization negatively affects model performance, leading users to prefer original models over quantized versions.
  
  - Users suggested that **OpenRouter** should disclose specific model versions to accurately set performance expectations.
- **Understanding Aider's Chat Modes**: Members discussed the effectiveness of various **Aider chat modes**, highlighting that using **o1-preview** as the Architect with **DeepSeek** or **o1-mini** as the Editor yields the best results.
  
  - A user noted that **Sonnet** performs exceptionally well for daily tasks without requiring complex configurations.

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ACE Your Model Control with Affine Editing**: The authors introduce the new **ACE (Affine Concept Editing)** method, treating concepts as affine functions to enhance control over model responses. ACE enables projecting activations onto hyperplanes, demonstrating improved precision in managing model behavior as shown through tests on **Gemma**.
  
  - **ACE** was evaluated on ten models, including **Llama 3 70B**, achieving superior control over refusal behavior across both harmful and harmless prompts. This method surpasses previous techniques, offering a more reliable strategy for steering model actions.
- **Latent Actions Propel Inverse Dynamics**: A user inquired about the top papers on **latent actions** and **inverse dynamics models**, emphasizing an interest in state-of-the-art research within these domains. The discussion highlighted the significance of relevant literature for advancing current AI methodologies.
  
  - While no specific papers were cited, the conversation underscored the importance of exploring **latent actions** and **inverse dynamics models** to push the boundaries of existing AI frameworks.
- **Scaling Laws Unveil Capability Dimensions**: A newly published paper titled [*Understanding How Language Model Performance Varies with Scale*](https://arxiv.org/abs/2405.10938) presents an observational approach to scaling laws based on approximately 100 publicly available models. The authors propose that language model performance is influenced more by a **low-dimensional capability space** rather than solely training across multiple scales.
  
  - **Marius Hobbhahn at Apollo** was recognized as a leading advocate for advancing the science of evaluation methods within the AI community, highlighting a growing focus on rigorous evaluation practices in AI model development.
- **WANDA Pruning Enhances Model Efficiency**: A member inquired if **lm-eval** supports zero-shot benchmarking for pruned models, mentioning the use of the **WANDA** pruning method. Concerns were raised regarding the **suspect results** obtained from zero-shot evaluations.
  
  - Discussions included modifications to **lm_eval** for compatibility with pruned models and evaluations on **ADVBench** using **vllm**, with specific code snippets shared to illustrate model loading and inference methods.
- **Forgetting Transformer Integrates Forget Gates**: The **Forgetting Transformer** paper introduces a method that incorporates a forget gate into the softmax attention mechanism, addressing limitations of traditional position embeddings. This approach offers an alternative to recurrent sequence models by naturally integrating forget gates into Transformer architectures.
  
  - Community discussions referenced related works like [**Contextual Position Encoding (CoPE)**](https://arxiv.org/abs/2405.18719) and analyzed different strategies for position embeddings, evaluating whether simpler methods like **ALiBi** or **RoPE** might integrate more effectively than recent complex approaches.

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Outperformed by ChatGPT**: Users compared [Perplexity](https://perplexity.supply/) with **ChatGPT**, highlighting **ChatGPT**'s versatility and superior conversational abilities.
  
  - Despite being **Pro** users of Perplexity, some expressed frustration over its limitations, leading to increased reliance on **ChatGPT**.
- **Introduction of Perplexity Shopping Feature**: The new **Perplexity Shopping** feature sparked discussions, with users inquiring about its exclusivity to the US market.
  
  - There is significant interest in understanding potential access limitations for the shopping functionality.
- **API Functionality Issues Reported**: Users reported that **API** responses remain unchanged despite switching models, causing confusion and frustration.
  
  - The community debated the platform's flexibility and questioned the diversity of its responses.
- **Fullstack Development Insights with Next.js**: A resource on [fullstack Next.js development](https://www.perplexity.ai/search/webapp-fullstack-nextjs-hono-b-W1pCGRCUSJmPBFklgsg.7w) was shared, offering insights into modern web frameworks.
  
  - *Explore the use of Hono for server-side routing!*
- **NVIDIA AI Chips Overheating Concerns**: Concerns were raised about **NVIDIA AI chips overheating**, as detailed in [this report](https://www.perplexity.ai/page/nvidia-ai-chips-overheat-SRXQJH9yQ8ebTG_KeAT46A).
  
  - Discussions emphasized the risks associated with prolonged usage of these chips.

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 1114 Struggles with Input Handling**: Users reported that **Gemini 1114** often ignores image inputs during conversations, leading to hallucinated responses, unlike models such as **Grok vision Beta**.
  
  - Members are hoping for confirmation and fixes, expressing frustration over recurring issues with the model.
- **DeepSeek Launches New Reasoning Model**: A new model, **DeepSeek-R1-Lite-Preview**, was announced, boasting enhanced reasoning capabilities and performance on AIME & MATH benchmarks.
  
  - However, some users noted the model’s performance is slow, prompting discussions about whether **DeepInfra** might be a faster alternative.
- **Clarifications on Prompt Caching**: Prompt caching is available for specific models like **DeepSeek**, with users questioning the caching policies of other providers.
  
  - Some members discussed how caching works differently between systems, particularly noting **Anthropic** and **OpenAI** protocols.
- **Issues with GPT-4o Model Description**: Users identified discrepancies in the newly released **GPT-4o**, noting the model incorrectly listed an **8k context** and wrong descriptions linked to **GPT-4**.
  
  - After highlighting the errors, members saw quick updates and fixes to the model card, restoring accurate information.
- **Comparisons of RP Models**: Members discussed alternatives to **Claude** for storytelling and role-playing, with suggestions for **Hermes** due to its perceived quality and cost-effectiveness.
  
  - Users indicated a mix of experiences with these models, with some finding **Hermes** preferable while others remain loyal to **Claude**.

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Model Loading on Limited Hardware**: A user encountered **model loading issues** in LM Studio on a **36GB RAM M3 MacBook**, highlighting error messages about system resource limitations.
  
  - Another member recommended avoiding **32B models** for such setups, suggesting a maximum of **14B** to prevent overloading.
- **GPU and RAM Requirements for LLMs**: Discussions emphasized that running **DeepSeek v2.5 Lite** requires at least **24GB VRAM** for the Q4_K_M variant and **48GB VRAM** for full Q8.
  
  - Members preferred **NVIDIA GPUs** over AMD due to driver stability issues affecting performance.
- **Cloud-Based Solutions vs Local Hardware**: Users explored **cloud-based hardware rentals** as a cost-effective alternative to local setups, with monthly costs ranging from **$25 to $50**.
  
  - This approach enables access to high-speed models without the constraints of local hardware limitations.
- **Workstation Design for AI Workloads**: A member sought advice on building a workstation for fine-tuning LLMs within a **$30,000 to $40,000** budget, considering options like NVIDIA **A6000s** versus fewer **H100s**.
  
  - The discussion underscored the importance of video memory and hardware flexibility to accommodate budget constraints.
- **Model Recommendations and Preferences**: Users recommended various models including **Hermes 3**, **Lexi Uncensored V2**, and **Goliath 120B**, based on performance and writing quality.
  
  - Encouragement was given to experiment with different models to identify the best fit for individual use cases as new options become available.

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Gaming PC Guidance Galore**: A user is seeking **gaming PC** recommendations within a budget of **$2500**, asking for suggestions on both components and where to purchase.
  
  - They are encouraging others to send direct messages for personalized advice.
- **Character Consistency Challenges**: A member inquired about maintaining consistent **character design** throughout a picture book, struggling with variations from multiple generated images.
  
  - Suggestions included using **FLUX** or image transformation techniques to improve consistency.
- **AI Models vs. Substance Designer**: A discussion arose on whether **AI models** could effectively replace **Substance Designer**, highlighting the need for further exploration in that area.
  
  - Members shared their thoughts on the capabilities of different AI models and their performance.
- **GPU Optimization for Video Generation**: Users discussed the difficulties of performing **AI video generation** on limited VRAM GPUs, noting potential for slow processing times.
  
  - The recommended course of action included clearing VRAM and using more efficient models like [CogVideoX](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V).
- **Fast AI Drawing Techniques**: A member inquired about the technology behind **AI drawing** representations that update quickly on screen, wondering about its implementation.
  
  - Responses indicated that it often relies on powerful GPUs and consistency models to achieve rapid updates.

 

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Audio Generation Enhancements in NotebookLM**: A member showcased their [podcast featuring AI characters](https://preview.getsubly.com/preview/89c122b6-cc30-458a-9d2b-d3728098b255), utilizing **NotebookLM** for orchestrating complex character dialogues.
  
  - They detailed the multi-step process involved, including the integration of various AI tools and NotebookLM's role in facilitating dynamic conversations.
- **Podcast Creation Workflow in NotebookLM**: A member shared their experience in creating a [German-language podcast on Spotify](https://open.spotify.com/show/5OapAqDLaWMxAzqXgywBZH?si=2e422be55d784fde) using **NotebookLM** for audio generation.
  
  - They emphasized the effective audio features of NotebookLM and sought **customization recommendations** to enhance their podcast production.
- **Transcription Features for Audio Files**: Members discussed the option to upload generated audio files to **NotebookLM** for automatic transcription.
  
  - Alternatively, one member suggested leveraging MS Word's *Dictate...Transcribe* function for converting audio to text.
- **Combining Notes Feature Evaluation**: Members deliberated on the 'Combine to note' feature in **NotebookLM**, assessing its functionality for merging multiple notes into a single document.
  
  - One member questioned its necessity, given the existing capability to combine notes, seeking clarity on its utility.
- **Sharing Notebooks Functionality**: A user inquired about the procedure for sharing notebooks with peers, encountering difficulties in the process.
  
  - Another member clarified the existence of a 'share note' button located at the top right corner of the **NotebookLM** interface to facilitate sharing.

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek-R1-Lite-Preview Launch**: DeepSeek announced the launch of [DeepSeek-R1-Lite-Preview](http://chat.deepseek.com), showcasing enhanced performance on **AIME** and **MATH** benchmarks with a transparent reasoning process.
  
  - Users are excited about its potential applications, noting that reasoning improvements scale effectively with increased length.
- **GPT-4o Update Enhances Capabilities**: OpenAI released a new [GPT-4o snapshot](https://platform.openai.com/docs/models#gpt-4o) as `gpt-4o-2024-11-20`, which boosts creative writing and improves file handling for deeper insights.
  
  - Recent performance tests show GPT-4o reclaiming top spots across various categories, highlighting significant advancements.
- **Truffles Hardware Device Gears Up for LLM Hosting**: The **Truffles** hardware device was identified as a semi-translucent solution for self-hosting LLMs at home, humorously termed a 'glowing breast implant'.
  
  - This nickname reflects the light-hearted conversations around innovative home-based LLM deployment options.
- **Vercel Acquires Grep to Boost Code Search**: Vercel announced the acquisition of [Grep](https://grep.app/), enabling developers to search through over 500,000 public repositories efficiently.
  
  - Founder Dan Fox will join Vercel's AI team to enhance code search functionalities and improve development workflows.
- **Claude Experiences Availability Fluctuations**: Users reported intermittent availability issues with **Claude**, experiencing sporadic downtimes across different instances.
  
  - These reliability concerns have led to active discussions, with users seeking updates via social media platforms.

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Triumphs Over Torch in Softmax**: A member compared [Triton's fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) against PyTorch's native implementation on an RTX 3060, highlighting smoother performance from Triton.
  
  - While Triton generally outperformed PyTorch, there were instances where PyTorch matched or exceeded Triton's performance.
- **Metal GEMM Gains Ground**: [Philip Turner's Metal GEMM implementation](https://github.com/philipturner/metal-flash-attention) was showcased, with a member noting their own implementation achieves **85-90%** of theoretical maximum speed, similar to Turner's.
  
  - Further discussion touched on the challenges of optimizing Metal compilers and the necessity of removing addressing computations from performance-critical loops.
- **Dawn's Regressing Render**: Concerns were raised about performance regressions in Dawn's latest versions, especially in the wgsl-to-Metal workflow post Chrome **130**, despite improvements in Chrome **131**.
  
  - Issues related to Undefined Behavior (UB) check code placement were identified as potential causes for the lag behind Chrome **129**.
- **FLUX Speeds Ahead with CPU Offload**: A member reported a **200% speedup** in FLUX inference by implementing per-layer CPU offload on a **4070Ti SUPER**, reducing inference time to **1.23 s/it** from **3.72 s/it**.
  
  - Discussion highlighted the effectiveness of pinned memory and CUDA streams on capable machines, though performance gains were limited on shared instances.

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek-R1-Lite-Preview Launch**: [DeepSeek-R1-Lite-Preview](https://x.com/deepseek_ai/status/1859200141355536422) is now live, featuring **o1-preview-level performance** on AIME & MATH benchmarks.
  
  - It also includes a **transparent thought process** in real-time, with open-source models and an API planned for release soon.
- **AI Agents for Writing Books**: [Venture Twins](https://x.com/venturetwins/status/1859298925930479998) is showcasing a project where ten AI agents collaborate to write a fully autonomous book, each assigned different roles like setting narrative and maintaining consistency.
  
  - Progress can be monitored through [GitHub commits](https://github.com/Lesterpaintstheworld/terminal-velocity) as the project develops in real-time.
- **LLMs Reasoning Without Prompting**: Research demonstrates that **large language models (LLMs)** can exhibit reasoning paths akin to **chain-of-thought (CoT)** without explicit prompting by adjusting the decoding process to consider top-$k$ alternative tokens.
  
  - This approach underscores the **intrinsic reasoning abilities** of LLMs, indicating that CoT mechanisms may inherently exist within their token sequences.
- **Generative Agent Behavioral Simulations**: A new architecture effectively simulates the attitudes and behaviors of **1,052 real individuals**, with generative agents achieving **85% accuracy** on responses in the General Social Survey.
  
  - The architecture notably reduces accuracy biases across **racial and ideological groups**, enabling tools for the exploration of individual and collective behavior in social science.
- **Soft Prompts Inquiry**: A member inquired about the investigation of **soft prompts** for LLMs as mentioned in a [post](https://bsky.app/profile/saganite.bsky.social/post/3lbeajzg3ms2f), highlighting their potential in optimizing system prompts into embedding space.
  
  - Another member responded, expressing that the concept of soft prompts is **pretty interesting**, indicating some interest within the community.

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **API Usage Challenges**: A member reported searching for an **API** or tool but found both options unsatisfactory, indicating frustration.
  
  - This issue reflects a broader interest in locating efficient resources within the community.
- **Model Option Clarification**: There was a discussion regarding the **4o model** and whether it utilized **o1 mini** or **o1 preview**, with confirmation leaning towards **o1 mini**.
  
  - A member suggested checking the settings to verify options, promoting hands-on troubleshooting.
- **High Temperature Performance**: A member questioned if improved performance at **higher temperatures** could be linked to their prompt style, suggesting an excess of guiding rules or constraints.
  
  - This raises considerations for optimizing prompt design to enhance AI responsiveness.
- **Beta Access to o1**: A member expressed excitement and gratitude towards NH for granting them **beta access to o1**, brightening their morning.
  
  - *Woo! Thank you NH for making this morning even brighter* reflects the exhilaration around new updates.
- **Delimiter Deployment in Prompts**: A member shared OpenAI's advice on using delimiters like triple quotation marks or XML tags to help the model interpret distinct sections of the input clearly.
  
  - This approach aids in structuring prompts better for improved model responses, allowing for easier input interpretation.

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **API Key Problems Block Access**: Multiple members reported encountering **403 errors**, indicating invalid **API keys** or the use of outdated endpoints while trying to access certain functionalities.
  
  - One member shared experiencing **fetch errors** and difficulties using the sandbox feature after verifying their API keys.
- **CORS Errors Interrupt API Calls**: A member on the **free tier** faced several **CORS errors** in the console despite using a standard setup without additional plugins.
  
  - Attempts to upgrade to a production key to resolve these issues were unsuccessful, highlighting limitations of the free tier.
- **Advanced Model Tuning Techniques Explored**: Discussions delved into whether model tuning could be achieved using only a preamble and possibly chat history.
  
  - Questions were raised about the model's adaptability to various training inputs, indicating the need for more effective tuning methods.
- **Cohere Introduces Multi-modal Embeddings**: A member praised the new **multi-modal embeddings** for images, noting **significant improvements** in their applications.
  
  - However, concerns were raised about the **40 requests per minute rate limit**, which hinders their intended use case, leading them to seek alternative solutions.
- **Harmony Project Streamlines Questionnaire Harmonization**: The **Harmony** project aims to harmonize questionnaire items and metadata using LLMs, facilitating better data compatibility for researchers.
  
  - A competition is being hosted to enhance Harmony's **LLM matching algorithms**, with participants able to register on [DOXA AI](https://harmonydata.ac.uk/doxa/) and contribute to making Harmony more robust.

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Adaptive Batching Optimizes GPU Usage**: The implementation of [adaptive batching](https://github.com/pytorch/torchtune/pull/2035) aims to maximize **GPU utilization** by dynamically adjusting batch sizes to prevent **OOM errors** during training.
  
  - It was suggested to integrate this feature as a flag in future recipes, ideally activated when `packed=true` to maintain efficiency.
- **Enhancing DPO Loss Structure**: Concerns were raised about the current **TRL** code structure regarding the inclusion of recent papers on DPO modifications, as seen in [Pull Request #2035](https://github.com/pytorch/torchtune/pull/2035).
  
  - A request was made to clarify whether to remove **SimPO** and any separate classes to keep the DPO recipe clean and straightforward.
- **SageAttention Accelerates Inference**: [SageAttention](https://github.com/thu-ml/SageAttention) achieves speedups of **2.1x** and **2.7x** compared to **FlashAttention2** and **xformers**, respectively, while maintaining end-to-end metrics across various models.
  
  - *Pretty cool inference gains here!* expressed excitement about the performance improvements introduced by SageAttention.
- **Benchmarking sdpa vs. Naive sdpa**: Members recommended benchmarking the proposed **sdpa/flex** method against the **naive sdpa** approach to identify performance differences.
  
  - The numerical error in scores may vary based on the **sdpa backend** and **data type** used.
- **Nitro Subscription Affects Server Boosts**: A member highlighted that **server boosts** will be removed if a user cancels their **free Nitro** subscription, impacting server management.
  
  - This underscores the importance of maintaining **Nitro** subscriptions to ensure uninterrupted server benefits.

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Tackles Triton Integration**: A user inquired about **Tinygrad's** native integration with **Triton**, referencing earlier discussions. **George Hotz** directed them to consult the questions document for further clarification.
  
  - Further discussions clarified the integration steps, emphasizing the compatibility between **Tinygrad** and **Triton** for enhanced performance.
- **SASS Assembler Seeks PTXAS Replacement**: Members discussed the future of the **SASS assembler**, questioning if it is intended to replace **ptxas**. **George Hotz** suggested referring to the questions document for more details.
  
  - This has sparked interest in the potential improvements **SASS assembler** could bring over **ptxas**, though some uncertainty remains regarding the assembler's long-term role.
- **FOSDEM AI DevRoom Seeks Tinygrad Presenters**: A community member shared an opportunity to present at the [FOSDEM AI DevRoom](https://aifoundry.org/fosdem-2025-low-level-ai-engineering-hacking-dev-room) on February 2, 2025, highlighting **Tinygrad's** role in the AI industry. Interested presenters are encouraged to reach out.
  
  - The presentation aims to showcase **Tinygrad's** latest developments and foster collaboration among AI engineers.
- **Tinybox Hackathon Hopes for Hands-on Engagements**: A member proposed organizing a pre-FOSDEM **hackathon**, suggesting bringing a **Tinybox** on-site to provide hands-on experiences. They expressed enthusiasm about engaging the community over Belgian beer during the event.
  
  - The hackathon aims to facilitate practical discussions and collaborative projects among **Tinygrad** developers.
- **Exploring Int64 Indexing in Tinygrad**: A member questioned the necessity of **int64 indexing** in scenarios not involving **huge tensors**, seeking to understand its advantages. The discussion aims to clarify the use-cases of **int64 indexing** beyond large-scale tensor operations.
  
  - Exploring various **indexing techniques**, the community is evaluating the performance and efficiency impacts of **int64** versus **int32** indexing in smaller tensor contexts.

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Async functions awaitable in Mojo sync functions**: A member is puzzled about being able to *await an async function* inside a sync function in **Mojo**, which contrasts with Python's limitations, seeking clarification or an explanation for this difference in handling async functionality.
- **Inquiry about Mojo library repository**: Another member is curious about the availability of a repository for libraries comparable to **pip** for **Mojo**, looking for resources or links that provide access to Mojo libraries.
- **Moonshine ASR Model Tested with Max**: A user tested the **Moonshine ASR** model performance using both the Python API for **Max** and a native **Mojo** version, noting both were about **1.8x slower** than the direct **onnxruntime** Python version.
  
  - The **Mojo** and Python **Max** versions took approximately **82ms** to transcribe 10 seconds of speech, whereas the native **onnxruntime** reached **46ms**. Relevant links: [moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1) and [moonshine.py](https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862).
- **Mojo Model.execute Crash Due to TensorMap**: Instructions for running the **Moonshine ASR** model are provided in comments at the top of the **mojo** file that was shared.
  
  - The user's experience highlighted that **passing in TensorMap** into **Model.execute** caused a crash, and manual unpacking of **26 arguments** was necessary due to limitations in **Mojo**. Relevant link: [moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1).
- **Seeking Performance Improvements in Mojo**: The user expressed that this is one of their first **Mojo** programs and acknowledged that it may not be idiomatic.
  
  - They requested assistance for achieving better performance, emphasizing their eagerness to improve their **Mojo** and **Max** skills.

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Tencent Hunyuan Model Fine Tuning**: A member inquired about fine-tuning the [Tencent Hunyuan model](https://huggingface.co/tencent/Tencent-Hunyuan-Large), sharing links to the [GitHub repository](https://github.com/Tencent/Tencent-Hunyuan-Large) and the [official website](https://llm.hunyuan.tencent.com/).
  
  - Additional resources provided include the [Technical Report](https://arxiv.org/abs/2411.02265) and a [Demo](https://huggingface.co/spaces/tencent/Hunyuan-Large) for reference.
- **Bits and Bytes on MI300X**: A member shared their experience using [Bits and Bytes](https://github.com/bitsandbytes-foundation/bitsandbytes) on the MI300X system, highlighting its ease of use.
  
  - They emphasized the necessity of using the `--no-deps` flag during updates and provided a one-liner command to force reinstall the package.
- **Axolotl Collab Notebooks for Continual Pretraining of LLaMA**: A user asked if Axolotl offers any **collab notebooks** for the **continual pretraining of LLaMA**.
  
  - Phorm responded that the search result was **undefined**, indicating no available notebooks currently, and encouraged users to check back for updates.

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Juan seeks help with multimodal challenges**: Juan inquired about using the experimental support for **vision language models** while working on a **multimodal problem**.
  
  - Another member offered additional assistance by saying *Let me know if there are any issues!*.
- **Juan discovers the mmmu notebook**: Juan later found the **mmmu notebook** himself, which provided the support he needed for his project.
  
  - He thanked the community for their *awesome work*, showing appreciation for the resources available.
- **Semantic Router as a Benchmark**: A member suggested that the [Semantic Router](https://github.com/aurelio-labs/semantic-router) should serve as the baseline for performance in **classification tasks**, emphasizing its **superfast AI decision making** capabilities.
  
  - The project focuses on **intelligent processing of multi-modal data**, and it may offer competitive benchmarks we aim to exceed.
- **Focus on Performance Improvement**: There was an assertion that the performance of existing **classification tools** needs to be surpassed, with the **Semantic Router** as a reference point.
  
  - Discussion revolved around identifying metrics and strategies to achieve better results than the baseline set by this tool.

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LLM-Native Resume Matching Launched**: Thanks to [@ravithejads](https://twitter.com/ravithejads), an **LLM-native solution** for resume matching has been developed, enhancing traditional screening methods.
  
  - This innovative approach addresses the **slow and tedious process** of manual filtering in recruitment, offering a more efficient alternative.
- **Building AI Agents Webinar on December 12**: Join [@Redisinc](https://twitter.com/Redisinc) and LlamaIndex for a webinar on **December 12**, focusing on building **data-backed AI agents**.
  
  - The session will cover architecting agentic systems and best practices for **reducing costs** and optimizing **latency**.
- **PDF Table Data Extraction Methods**: A member in **#general** inquired about approaches to extract **table data** from PDF files containing text and images.
  
  - They expressed interest in knowing if there are any existing applications that facilitate this process.
- **Applications for PDF Data Extraction**: Another member sought recommendations for applications available to extract data specifically from PDFs.
  
  - This highlights a need within the community for tools that can handle various PDF complexities.

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **New UI sparks mixed feelings**: Some users feel the new UI is slightly **overwhelming** and unclear in directing attention, with one comparing it to a computer from *Alien*.
  
  - However, others are starting to appreciate its UNIX-inspired design, finding it suitable for **1.0 features**.
- **Rate Limit Configuration Needed**: A user expressed frustration over being rate limited by **Anthropic**, noting that current error handling in Interpreter leads to session exits when limits are exceeded.
  
  - They emphasized the importance of incorporating better rate limit management in future updates.
- **User Calls for UI Enhancements**: There are calls for a more informative UI that displays current tools, models, and working directories to enhance usability.
  
  - Users are also advocating for a potential **plugin ecosystem** to allow customizable features in future releases.
- **Compute Workloads Separation Proposed**: One member suggested splitting LLM workloads between local and cloud compute to optimize performance.
  
  - This reflects a concern about the limitations of the current Interpreter design, which is primarily built for one LLM at a time.

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Intel AMA Session Tomorrow**: A **Hackathon AMA** with **Intel** is scheduled for **3 PM PT tomorrow (11/21)**, offering participants direct insights from Intel specialists. Don’t forget to [watch live here](https://www.youtube.com/watch?v=_Wm5guUXt54) and set your reminders!
  
  - Participants are encouraged to prepare their questions to maximize the session's benefits.
- **Participant Registration Confusion**: A user reported not receiving emails after joining three different groups and registering with multiple email addresses, raising uncertainties about the success of their registration.
- **Clarification on Event Type**: A member sought clarification on whether the registration issue pertained to the **hackathon** or the **MOOC**, highlighting potential confusion among participants regarding different registration types.

 

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Refact.AI Live Demo Highlights Autonomous Agents**: **Refact.AI** is hosting a live demo showcasing their **autonomous agent** and [tooling](https://github.com/smallcloudai).
  
  - Join the [live demo and conversation](https://discord.com/events/1089876418936180786/1300459081181429810) to explore their latest developments.
- **Refact.AI Unveils New Tooling**: The **Refact.AI** team has released new [tooling](https://github.com/smallcloudai) to support their **autonomous agent** projects.
  
  - Participants are encouraged to engage with the tools during the live demo event.

 

---

The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **LAION Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1308644010251915346) (263 messages🔥🔥):

> - `Hugging Face Discord Community`
> - `AI and Machine Learning Projects`
> - `Gradio and Streamlit Integration`
> - `LangChain and RAG`
> - `General Discussion and Support Requests`

- **Community Engagement and Support**: Members shared their experiences with the Hugging Face community, discussing support requests related to training models and technical issues, such as RuntimeErrors.
  
  - The community provided troubleshooting tips and encouraged sharing resources, leading to collaborative problem-solving.
- **Integrating AI Models into Projects**: Users explored ways to integrate AI models into applications, with suggestions to use Gradio for its simplicity and efficiency over LangChain.
  
  - Discussions included practical approaches for building interfaces and workflows for various AI models, emphasizing hands-on learning.
- **Exploration of RAG and AI Agents**: The concept of Retrieval-Augmented Generation (RAG) and creating AI agents was debated, with recommendations to learn about them through available blogs.
  
  - Members expressed the importance of working on projects to solidify understanding and explore potential creative applications.
- **New Projects and Collaboration Opportunities**: A new community initiative named Open/acc was introduced, focusing on collaboration in open science and machine learning.
  
  - Participants were encouraged to share events and ideas within this new space to foster innovation.
- **General Discussions and Humor**: Light-hearted conversations about cooking, shared interests, and humorous takes on cult-like communities within the Discord were frequent.
  
  - Members also shared amusing gifs and engaged in friendly banter, contributing to a positive community atmosphere.

**Links mentioned**:

- [O'Reilly Media - Technology and Business Training](https://www.oreilly.com): no description found
- [Tweet from undefined](https://x.com/jadechoghari): no description found
- [Hamster Cry GIF - Hamster Cry Tears - Discover & Share GIFs](https://tenor.com/view/hamster-cry-tears-funny-wiping-tears-gif-23475965): Click to view the GIF
- [Simpsons Homer GIF - Simpsons Homer Bart - Discover & Share GIFs](https://tenor.com/view/simpsons-homer-bart-lisa-join-us-gif-17846376318791889140): Click to view the GIF
- [Lemon Demon Sundial GIF - Lemon Demon Sundial View-Monster - Discover & Share GIFs](https://tenor.com/view/lemon-demon-sundial-view-monster-view-monster-viewmonster-gif-12281888211472007989): Click to view the GIF
- [open-acc (open/ acc)](https://huggingface.co/open-acc): no description found
- [Spaces - Hugging Face](https://huggingface.co/spaces?sort=trending&search=inpaint): no description found
- [HeyGen - AI Video Generator](https://www.heygen.com/): no description found
- [Argil AI - Get ai short videos with AI clones in 2 minutes.](https://www.argil.ai/): Create AI-powered short videos featuring AI clones quickly and easily with Argil AI.
- [Large Language Models explained briefly](https://m.youtube.com/watch?v=LPZh9BOjkQs): Dig deeper here: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3piTechnical details as a talk: https://youtu.be/KJtZARuO3JYMade for an...
- [Rabbit Bunny GIF - Rabbit Bunny Toilet - Discover & Share GIFs](https://tenor.com/view/rabbit-bunny-toilet-yes-come-gif-4686108): Click to view the GIF
- [Sunday Cult Of The Lamb GIF - Sunday Cult of the lamb Cult - Discover & Share GIFs](https://tenor.com/view/sunday-cult-of-the-lamb-cult-happy-sunday-god-gif-422811577611096801): Click to view the GIF

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/) (1 messages):

richieghost: today-im-learning LangGraph

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1308679098402476082) (10 messages🔥):

> - `3D Printing Designs`
> - `Generative Design Tools`
> - `Custom AI Model Deployment`
> - `AI Security Research`
> - `Automated AI Researcher`

- **3D Printing Achieves Complex Designs**: Highlighted by the impressive **Bugatti brake caliper**, many components have showcased the capabilities of **3D printing** to achieve complex designs. With machine learning models optimizing component removal, engineers can enhance performance across sectors, including automotive and architecture.
  
  - *Using vector calculations,* the process streamlines efficiency not only in cars but also in broader engineering applications.
- **Generative Design Tools Available for Free**: Generative design tools have been praised for their creativity and innovative capabilities, available for free through an educational license with **Fusion 360**. This makes advanced design techniques accessible to students and aficionados alike.
  
  - The excitement around these tools stems from their potential to revolutionize design thinking and implementation.
- **Deploying Custom AI Models Now Possible**: A member shared the discovery that custom AI models can be deployed using a **handler file** on Hugging Face, allowing for tailored pre- and post-processing of models. The process involves creating a `handler.py` specifying methods for handling requests and responses.
  
  - This versatile setup enhances customization for AI projects via [Hugging Face endpoints](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler).
- **New Paper on AI Security Insights Released**: A recent paper from AI researchers at **Redhat/IBM** discusses the security implications of publicly available AI models, addressing risks and lifecycle management. Comprehensive strategies are proposed to enhance security for both developers and users.
  
  - The paper aims to foster more standardized practices within the AI community, contributing significantly to the discussion on safety and transparency. [View the paper](https://huggingface.co/papers/2411.12275).
- **Automated AI Research Assistant Takes Off**: An individual created an **Automated AI Researcher** using local LLMs, which can generate research documents in response to user queries. This system utilizes web scraping to compile information and produce topic-relevant summaries and links.
  
  - The innovation emphasizes the potential of AI to simplify research and information gathering, making it accessible at the touch of a button.

**Links mentioned**:

- [FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models](https://arxiv.org/abs/2311.15614): Collecting high-quality labeled data for model training is notoriously time-consuming and labor-intensive for various NLP tasks. While copious solutions, such as active learning for small language mod...
- [Paper page - Building Trust: Foundations of Security, Safety and Transparency in AI](https://huggingface.co/papers/2411.12275): no description found
- [Create custom Inference Handler](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler): no description found
- [no title found](https://amfg.ai/2019/07/24/7-complex-designs-achieved-with-3d-printing/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gvlzug/i_created_an_ai_research_assistant_that_actually/): no description found

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1308803911905181778) (4 messages):

> - `Fractal Forest Creatures`
> - `AI in Music and Animation`
> - `Effective Prompting Techniques`
> - `Psychedelic Experience with Music`
> - `Neo's Journey to the 60s`

- **Neural Voyage to the 60s**: The YouTube video titled [A.I. The Matrix Red Pill Scene Psychedelic Trip](https://youtu.be/JugL1okFCqI?si=zn7wpJFaQJnQcJx3) explores what happens when Neo's red pill sends him back to the 60s, merging iconic music with AI animation.
  
  - The video features a blend of tunes from **The Beatles**, **The Doors**, and **Jimi Hendrix**, creating a vibrant audiovisual experience.
- **The Art of Prompting in AIs**: A discussion emerges around the video [BAD vs GOOD prompting](https://youtu.be/m3Izr0wNfQc) which examines the necessity of effective prompting techniques in today's AI landscape.
  
  - Members are encouraged to leave comments, reflecting on the evolving dynamics of prompting and its impact on AI outputs.
- **Appreciation for Cool Content**: A member expressed enthusiasm, saying, *'very cool share, thank you for this!'* in response to the matrix-themed AI video.
  
  - Such reactions highlight the community's interest in innovative AI applications blended with artistic flair.

**Links mentioned**:

- [BAD vs GOOD prompting](https://youtu.be/m3Izr0wNfQc): Let's see in this video if we still need to make good prompting nowadays and if there is a difference, at what point is it different.Feel free to leave comme...
- [A.I. The Matrix Red Pill Scene Psychedelic Trip with The Beatles, The Doors, Nirvana, Jimi Hendrix🎧🔈](https://youtu.be/JugL1okFCqI?si=zn7wpJFaQJnQcJx3): Headphones Essential #4K #ai #animation #thematrix #redpill #bluepill #johnlennon #jimmorrison #jimihendrix #kurtcobain #nirvana #thebeatles #thedoors #psych...

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1308719052771819581) (4 messages):

> - `3080 GPU Pricing`
> - `VRAM Utilization`
> - `Channel Discussion Etiquette`

- **Affordable 3090 GPUs hit low prices**: Members discussed that used **3090 GPUs** are now available for **400-500€**, a price deemed worthwhile.
  
  - Another member suggested that **400-450€** could be considered a good deal for these cards.
- **GPU usage concerns raised**: Concerns were expressed regarding whether the GPU is being fully utilized; one member feels that only **VRAM** is actively being used.
  
  - This raises questions about the actual performance being leveraged during tasks.
- **Request for off-topic discussions move**: A member requested that off-topic discussions be redirected to another channel to keep the reading group focused.
  
  - They encouraged others to use the relevant channel for further discussion, promoting a better environment for group activities.

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1308714238583246859) (5 messages):

> - `Semantic Search Challenges`
> - `Issues with Evaluate Library`
> - `Alternatives to Pandas`

- **Semantic search struggles with focus topics**: A user is facing challenges with **semantic search** using OpenAI's **Ada 002**, where embeddings prioritize dominant topics, leading to lower rankings for less prominent but relevant sentences.
  
  - They are seeking alternatives to **semantic search** to effectively extract the needed information.
- **Frustration with Evaluate Library**: A user expressed frustration about the **Evaluate Library**, stating they had to manually compute **lift** metrics for a presentation, which was inefficiency.
  
  - They shared a sentiment of irritation, indicating it's bothersome when libraries do not function as expected.
- **Faster alternatives to Pandas needed**: Another user shared their struggle with **Pandas**, finding it slow when dealing with large datasets and requesting suggestions for faster libraries.
  
  - This highlights an ongoing need for more efficient data handling tools within the community.

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1308757287942225920) (6 messages):

> - `Diffusers Version Issues`
> - `CogVideoX1.5-5B-I2V Repo Updates`
> - `Colab Session Crashes`
> - `FP16 Model Loading`
> - `Oversampling and Downsampling Query`

- **Diffusers Version Fails to Work**: A member reported their attempts to get **i2v** working with the newer **diffusers** version were unsuccessful.
  
  - This issue may relate to recent updates reflected in the codebase.
- **Repo Update Needed for CogVideoX1.5-5B-I2V**: Another member noted that **corrections are needed** on the **CogVideoX1.5-5B-I2V** repository, highlighting a recent commit made two hours ago.
  
  - They referred to the [CogVideoX1.5-5B-I2V discussion](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/discussions?status=closed) for more details.
- **Colab Session Crashes While Loading Model**: A member shared a [Colab link](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing) where their session crashes when attempting to load the transformer model.
  
  - They speculated that the crash might be due to trying to load the **fp16 model**.
- **Request for Minimal Reproducible Snippet**: A member advised that issues should be reported with a **minimal reproducible snippet** to facilitate troubleshooting efforts.
  
  - This approach will help clarify specific problems faced by users.
- **Downsampling and Oversampling Inquiry**: A member asked whether it is possible to perform **oversampling or downsampling** in the discussed context.
  
  - This reflects ongoing interest in refining techniques for model training.

**Links mentioned**:

- [THUDM/CogVideoX1.5-5B-I2V · Discussions](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/discussions?status=closed): no description found
- [Google Colab](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing): no description found

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1308755202949513266) (175 messages🔥🔥):

> - `DeepSeek Prover`
> - `OpenAI o1 release`
> - `GPT-4o update`
> - `Model performance comparison`
> - `Community discussions on AI models`

- **DeepSeek Prover work discussed**: Members shared interest in DeepSeek's work using reinforcement learning in their model, speculating about a possible paper release.
  
  - There were discussions about the challenges of model size and performance, suggesting that a full release may be delayed.
- **OpenAI o1's imminent release**: Speculation arose about OpenAI releasing their o1 model soon, with some members expressing skepticism about timeline rumors circulating in the community.
  
  - Discussions hinted at OpenAI needing to showcase their o1 model in response to growing competition in the industry.
- **GPT-4o gets an update**: OpenAI announced an update to GPT-4o, improving its creative writing capabilities and file handling.
  
  - The model climbed back to the top in various performance categories including creative writing and coding in a chatbot competition.
- **Comparison of model performances**: Members compared the performance of various AI models, including OpenAI's and DeepSeek's, pointing out the importance of creative and technical skill improvements.
  
  - There were reflections on user experiences with the models, highlighting strengths and weaknesses in different tasks.
- **Community engagement and reactions**: The community engaged in lively discussions around the latest AI model updates and performance metrics, often sharing humorous takes.
  
  - Several users expressed their excitement and skepticism in equal measure regarding the direction AI development is heading.

**Links mentioned**:

- [Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)](https://x.com/lmarena_ai/status/1859307979184689269): Exciting News from Chatbot Arena❤️‍🔥 Over the past week, the latest @OpenAI ChatGPT-4o (20241120) competed anonymously as "anonymous-chatbot", gathering 8,000+ community votes. The result? ...
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1859296125947347164?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): GPT-4o got an update 🎉 The model’s creative writing ability has leveled up–more natural, engaging, and tailored writing to improve relevance & readability. It’s also better at working with uploaded...
- [Tweet from DeepSeek (@deepseek_ai)](https://x.com/deepseek_ai/status/1859200149844803724): 🌟 Inference Scaling Laws of DeepSeek-R1-Lite-Preview Longer Reasoning, Better Performance. DeepSeek-R1-Lite-Preview shows steady score improvements on AIME as thought length increases.
- [Will a “good” (fine-tuned) opensource model utilizing chain-of-thought reasoning like o1 be released by EOY 2024?](https://manifold.markets/Soli/will-a-finetuned-opensource-model-u): 90% chance. The model should be able to decide how long it needs to think based on the complexity of the problem. Ideally it should be ranked higher on LMSYS than the “normal” model but this is not a ...
- [Which major AI lab will be the first to release a model that "thinks before it responds" like o1 from OpenAI?](https://manifold.markets/NeuralBets/which-major-ai-lab-will-be-the-firs): OpenAI o1 blog post says: We are introducing OpenAI o1, a new large language model trained with reinforcement learning to perform complex reasoning. o1 thinks before it answers—it can produce a long ...
- [Tweet from Andrew Curran (@AndrewCurran_)](https://x.com/AndrewCurran_/status/1859241005465432540): This is interesting, and promising. 'DeepSeek-R1-Lite also uses a smaller base model, which cannot fully unleash the potential of the long thinking chain.' I wonder if there is a similar s...
- [no title found](https://mp.weixin.qq.com/s/e1YnTxZlzFvjcmrLLTA8fw?poc_token=HI7bPWejXDRRW5OqohHtuuqRtJ4F_UgfMxhXIhnk): no description found
- [GitHub - deepseek-ai/DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5): Contribute to deepseek-ai/DeepSeek-Prover-V1.5 development by creating an account on GitHub.
- [Tweet from FxTwitter / FixupX](https://x.com/search?q=stay%20in%20line%20vote&src=typed_query))): Sorry, that user doesn't exist :(

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1308771688439087107) (6 messages):

> - `Francois Fleuret mention`
> - `Korean LLM evaluation issues`
> - `Japanese LLM leaderboard`

- **Francois Fleuret's Controversial Remarks**: There was a discussion around whether **Francois Fleuret** was trash-talking **lbdl**, highlighting the ongoing tensions and opinions in the community.
  
  - One user expressed disbelief at the situation, calling it *unbelievable*.
- **Critique of NeurIPS Reviewer Standards**: Concerns were raised over NeurIPS D&B reviewers dismissing a project in **Korean LLM evaluation** based on claims that *it already exists in Chinese*.
  
  - Commentators argued that every language deserves tailored models, emphasizing the need for inclusivity in **NLP development**.
- **Highlighting Japanese LLM Performance Testing**: A user praised the establishment of a **leaderboard for Japanese LLMs**, created by **@llm_jp**, which tests performance across diverse NLP tasks.
  
  - They noted that **Japanese** requires multiple character sets for writing, adding complexity to evaluation efforts.

**Links mentioned**:

- [Tweet from Stella Biderman (@BlancheMinerva)](https://x.com/blancheminerva/status/1859271409429795083?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): NeurIPS D&B reviewers think "this already exists in Chinese" is a reason to dismiss the value of a project in Korean LLM evaluation, but the people whose opinion I care about know better. Eve...
- [François Chollet (@fchollet.bsky.social)](https://bsky.app/profile/fchollet.bsky.social/post/3lbew74c7is2k): Not a week passes by without me hearing from folks who bought a deep learning book by an author with a name close to mine because they thought I had written it. Something like half of its readers thin...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1308652899479453736) (25 messages🔥):

> - `o1 Release Speculations`
> - `Training Dynamics in LLMs`
> - `Reinforcement Learning Trends`
> - `Model Evaluation Bottlenecks`
> - `Release Fatigue and Post-Release Plans`

- **o1 Release Speculations Stir Interest**: There is buzz about the **o1 release** potentially happening tomorrow, coinciding with **DevDay Singapore**, but it's based on unconfirmed rumors.
  
  - *A member noted*, *Wednesday would be a weird day for it* but the community remains attentive to updates.
- **Exploring LLM Training Dynamics**: Recent findings suggest that how models **memorize** training examples impacts their ability to generalize, particularly if they first understand the concept before memorizing.
  
  - This highlights a method to predict test accuracy from models based on their training dynamics alone, as shared in a discussion by **Katie Kang**.
- **Reinforcement Learning's Surprise Comeback**: A member expressed excitement about the resurgence of **Reinforcement Learning (RL)** despite previous doubts, feeling free to embrace their RL roots again.
  
  - *They remarked*, *I can go back to being an RL person*, reflecting a broader sentiment of optimism within the community.
- **Bottleneck in Model Evaluation**: Concerns were raised about the **evaluation bottleneck**, indicating that it only takes a few hours to evaluate MMLU but can still hold up the process.
  
  - Discussion ensued on deciding when to stop training, with opinions about persistent efforts even as exhaustion sets in.
- **Release Fatigue and Plans Ahead**: With the imminent release approaching, the commentary suggested a need for recovery post-launch, with thoughts about a relaxing December ahead.
  
  - *Amidst the chatter*, *one member humorously mentioned*, *I am dead*, indicating the toll of the release process on developers.

**Links mentioned**:

- [Tweet from Nathan Lambert (@natolambert)](https://x.com/natolambert/status/1859255627882664034?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): I love seeing RL continue its magnificent takeover that I have doubted so many times over. I can go back to being an RL person, I don't even need to masquerade as an "RLHF" or "NLP"...
- [Lucas Beyer (bl16) (@giffmana.bsky.social)](https://bsky.app/profile/giffmana.bsky.social/post/3lbfok33dt22g): https://www.astralcodexten.com/p/how-did-you-do-on-the-ai-art-turing
- [Tweet from Sergey Levine (@svlevine)](https://x.com/svlevine/status/1859118047304602061): An intriguing new result from @katie_kang_: after training long enough, LLMs will reproduce training examples exactly (not surprising). But how they get there matters: if they first get the right answ...
- [Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)](https://x.com/apples_jimmy/status/1859062064410751266): o1 tomorrow apparently ? Wednesday would be a weird day for it but I guess devday Singapore. Heard from a birdie but I can’t confirm it.
- [Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)](https://x.com/apples_jimmy/status/1859121134777843765): This was sent out an hour ago. Mini and preview api access, let’s see if there’s more to come.

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1308645677320114268) (176 messages🔥🔥):

> - `Vision Support`
> - `Multi-GPU Training`
> - `Internship Opportunities`
> - `Data Quality in NLP`
> - `Training Llama Models`

- **Vision Support is on the Way**: A member inquired about the status of **vision support**, to which another confirmed that it is indeed coming soon.
  
  - This feature will be released alongside other updates.
- **Multi-GPU Training Discussed**: New users are exploring **multi-GPU training** capabilities with Unsloth, currently noting that it's not available but expected soon.
  
  - Members shared strategies like using **Llama Factory** for managing multiple GPUs.
- **Internship Roles Available**: Discussion emerged about available **internship roles** with Unsloth, prompting curiosity about the required experience.
  
  - Members were pointed towards a link detailing the opportunities and current needs.
- **Importance of Data Quality for NLP**: A user sought guidance on dataset cleaning for their **NLP task**, emphasizing its criticality for success.
  
  - The conversation stressed the importance of dataset quality with advice to start with a smaller dataset for better control during training.
- **Training Llama Models**: Several members shared their experiences training different **Llama models**, discovering varying degrees of success depending on parameters.
  
  - Suggestions included beginning with base models before scaling, weighing the dataset size, and adjusting training steps for optimal performance.

**Links mentioned**:

- [Machine Learning Projects by huq02 using Weights & Biases](https://wandb.ai/authorize): Weights & Biases, developer tools for machine learning
- [Nice Smack GIF - Nice Smack Delicious - Discover & Share GIFs](https://tenor.com/view/nice-smack-delicious-meme-gif-8375212): Click to view the GIF
- [Llama 3.1 | Model Cards and Prompt formats](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/): Llama 3.1 - the most capable open model.
- [Home](https://github.com/unslothai/unsloth/wiki#-we-are-hiring): Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
- [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing): no description found
- [Fine-tuning A Tiny-Llama Model with Unsloth](https://www.analyticsvidhya.com/blog/2024/02/fine-tuning-a-tiny-llama-model-with-unsloth/): Fine-Tuning your Tiny-Llama model for peak performance with Unsloth's user-friendly tools and advanced features.
- [Google Colab](https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_?usp=sharing,): no description found
- [unsloth/unsloth/chat_templates.py at 5078a870c04e60b2491cd4f2974cf78521961179 · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/5078a870c04e60b2491cd4f2974cf78521961179/unsloth/chat_templates.py#L583)): Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1308881292326211656) (9 messages🔥):

> - `NixOS Installation`
> - `Fedora KDE Experience`
> - `Windows-like Linux Distritos`
> - `Checkpoint Selection in AI Training`

- **Homies Recommend NixOS**: A member mentioned that their friends are urging them to install **NixOS**, showing a popular trend among users.
  
  - This sparked curiosity and comments about various Linux distributions among the group.
- **Fedora KDE is a Hit**: A member enthusiastically promoted **Fedora KDE**, exclaiming 'Fedora KDE ftw'.
  
  - The discussion included light-hearted banter about its benefits compared to other operating systems.
- **Linux Distro Aesthetic**: Another member remarked that Fedora KDE looks 'kinda like **Windows**' and expressed excitement about its appearance.
  
  - Their humorous take on the distro's interface resonated well with others in the channel.
- **AI Training Checkpoint Dilemma**: A member inquired about others' preferences when choosing AI training checkpoints, asking, 'which checkpoint are you taking?'
  
  - They shared that they opted for training checkpoint **200** without hesitation, inviting opinions on the varying approaches.

 

**Link mentioned**: [Dsa GIF - Dsa - Discover & Share GIFs](https://tenor.com/view/dsa-gif-22912899): Click to view the GIF

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1308881736322650202) (10 messages🔥):

> - `Fine-tuning LLMs`
> - `Model Export to Hugging Face`
> - `Pre-tokenization and Continued Pretraining`
> - `Inference with VLLM`
> - `Checkpoint Callback for Saving Models`

- **Exporting Llama 3.1 for Local Use**: A new user had success with **2x Llama 3.1 8b** and sought guidance on how to export the model for **local project** usage with Hugging Face.
  
  - Members advised on using the **16-bit version** for performance and capabilities in building RAG applications.
- **Pre-tokenized Dataset Queries**: Discussion arose about the compatibility of an **already tokenized dataset** for continued pretraining, with one user expressing uncertainty.
  
  - Another member suggested that passing the **untokenized dataset** might work better for training.
- **Warnings in Model Loading**: A user reported a **SyntaxWarning** related to invalid escape sequences while loading a model from Hugging Face, expressed with specific warning texts provided in the code snippet.
  
  - Despite the warnings, the model and tokenizer were successfully loaded as confirmed by the console output.
- **Checkpoint Management for Fine-tuning**: A member sought advice on fine-tuning while ensuring checkpoints are saved to a storage solution, like **Google Drive** or **Kaggle datasets**.
  
  - Another user confirmed the suggestion of using **callbacks** for this purpose, with a reference to learning about **Weights & Biases (WandB)** for tracking.

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1308701305472024626) (1 messages):

> - `SageAttention2`
> - `Quantized Attention`
> - `Inference Acceleration`

- **SageAttention2 boasts accurate 4-bit attention**: The [SageAttention2 Technical Report](https://arxiv.org/html/2411.10958v1) introduced a method for **4-bit matrix multiplication** that accelerates the attention process, achieving **2x speedup** compared to FlashAttention2.
  
  - SageAttention2 aims to maintain precision while optimizing attention computation, marking a significant enhancement in **inference acceleration**.
- **GitHub repository for SageAttention**: The [SageAttention GitHub repository](https://github.com/thu-ml/SageAttention/tree/main) claims **2.1x and 2.7x speedups** over FlashAttention2 and xformers, respectively, without losing end-to-end metrics.
  
  - This implementation indicates that SageAttention2 is a drop-in replacement for **FlashAttention2**, optimized for **RTX40/3090** hardware but serves solely for inference.
- **Limitations of SageAttention2**: It is noted that SageAttention2 supports only **inference** and not training, highlighting its intended use case.
  
  - The development focuses on optimizing performance features while ensuring compatibility with existing models.

**Links mentioned**:

- [SageAttention2 Technical Report: Accurate 4 Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/html/2411.10958v1): no description found
- [GitHub - thu-ml/SageAttention: Quantized Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models.](https://github.com/thu-ml/SageAttention/tree/main): Quantized Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models. - thu-ml/SageAttention

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1308649606825119784) (148 messages🔥🔥):

> - `Aider Setup Challenges`
> - `DeepSeek Performance`
> - `OpenRouter Concerns`
> - `Model Quantization Effects`
> - `Coding Tools Comparisons`

- **Aider Setup Challenges Resolved**: Users experienced difficulties with Aider's setup, particularly with API keys and environment variables, leading some to consider reinstalling components to resolve issues.
  
  - After troubleshooting, one user confirmed that a force reinstall helped in getting Aider up and running successfully.
- **DeepSeek Impresses with Performance**: DeepSeek-R1-Lite-Preview has shown to match o1-preview performance on AIME & MATH benchmarks, with faster response times noted compared to previous models.
  
  - The model's transparent reasoning process allows users to see its thought process in real-time, elevating its effectiveness for coding tasks.
- **Concerns Over OpenRouter's Model Quality**: Users expressed dissatisfaction with OpenRouter utilizing quantized versions of open-source models, leading to doubts about performance on the Aider Leaderboard.
  
  - There were calls for clearer warnings on the Aider Leaderboard about potential performance discrepancies when using OpenRouter's quantized models.
- **Impact of Model Quantization Discussed**: Quantization's adverse impact on model performance has raised concerns among users, with many preferring original models over quantized versions.
  
  - Users suggested that OpenRouter should disclose specific model versions to accurately reflect performance expectations.
- **Comparative Use of Coding Tools**: Users compared various coding tools like Aider, Cursor, and Sonnet, sharing insights into their effectiveness for file creation and editing tasks.
  
  - Many participants noted that they find Aider to be particularly beneficial for editing, while alternatives like Cline are too costly for extensive use.

**Links mentioned**:

- [Tweet from Paul Gauthier (@paulgauthier)](https://x.com/paulgauthier/status/1859320459634016553?s=46&t=AkDCTtZVFFazuKDknG6fLA): The new gpt-4o-2024-11-20 scored the same as the 08-06 version, and behind the 05-13 version on aider's code editing benchmark. This may be the first OpenAI in-family model update that wasn't ...
- [File editing problems](https://aider.chat/docs/troubleshooting/edit-errors.html): aider is AI pair programming in your terminal
- [Home](https://aider.chat/): aider is AI pair programming in your terminal
- [DeepSeek](https://chat.deepseek.com/): Chat with DeepSeek AI.
- [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards/): Quantitative benchmarks of LLM code editing skill.
- [Models | OpenRouter](https://openrouter.ai/models): Browse models on OpenRouter
- [Qwen2.5 Coder 32B Instruct - API, Providers, Stats](https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct): Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Run Qwen2.5 Coder 32B Instruct with API
- [DeepSeek V2.5 - API, Providers, Stats](https://openrouter.ai/deepseek/deepseek-chat): DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. Run DeepSeek V2.5 with API
- [Tweet from Andrew Curran (@AndrewCurran_)](https://x.com/andrewcurran_/status/1859235248632123763?s=46&t=LoeRx5EgmzbDflKGl42Euw): Two months after the o1-preview announcement, and its Chain-of-Thought reasoning has been replicated. The Whale can now reason. DeepSeek says that the official version of DeepSeek-R1 will be completel...
- [Tweet from OpenAI (@OpenAI)](https://x.com/OpenAI/status/1859296125947347164): GPT-4o got an update 🎉 The model’s creative writing ability has leveled up–more natural, engaging, and tailored writing to improve relevance & readability. It’s also better at working with uploaded...
- [🚀 DeepSeek-R1-Lite-Preview is now live: unleashing supercharged reasoning power! | DeepSeek API Docs](https://api-docs.deepseek.com/news/news1120): 🔍 o1-preview-level performance on AIME & MATH benchmarks.
- [Meta: Llama 3.1 70B Instruct – Provider Status](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct/providers): See provider status and make a load-balanced request to Meta: Llama 3.1 70B Instruct - Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 70B instruct-t...

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1308719778260717598) (29 messages🔥):

> - `Aider usage challenges`
> - `Chat modes best practices`
> - `Token limit concerns`
> - `Language support in Aider`
> - `Context extension mechanisms`

- **Understanding Aider's chat modes**: Members discussed the effectiveness of various Aider chat modes, emphasizing that using **o1-preview** as the Architect with **DeepSeek** or **o1-mini** as the Editor provided the best results.
  
  - A user noted that **Sonnet** performed exceptionally well for daily tasks without complex configurations.
- **Token limit exhaustion with Aider**: Users expressed concerns about burning through tokens while generating code, with suggestions for caching mechanisms to store context effectively.
  
  - A member was informed that changes in the set of `/read` files disrupt the cache, but modifications to `/add` files do not.
- **Issues with unexpected language responses**: A user reported using **o1-mini** and receiving responses in **Spanish** despite having a conventions file stating to respond in English.
  
  - The solution offered was to specify the language explicitly using the `--language` option.
- **Request for custom context extension features**: One user inquired about adding a personal automatic context extension mechanism to Aider, seeking to create an extension point for custom code.
  
  - However, it was clarified that such feature integration is currently not possible with the existing version of Aider.
- **Running scripts in Aider**: Discussion around automating checklist tasks with Aider suggested using the `--message` argument or the `aider -m` mode for scripting.
  
  - Examples were provided on how to run loops and apply instructions to multiple files using shell scripts.

**Links mentioned**:

- [Separating code reasoning and editing](https://aider.chat/2024/09/26/architect.html): An Architect model describes how to solve the coding problem, and an Editor model translates that into file edits. This Architect/Editor approach produces SOTA benchmark results.
- [Scripting aider](https://aider.chat/docs/scripting.html): You can script aider via the command line or python.
- [Chat modes](https://aider.chat/docs/usage/modes.html): Using the chat, ask and help chat modes.
- [Supported languages](https://aider.chat/docs/languages.html#how-to-add-support-for-another-language): Aider supports pretty much all popular coding languages.

---

### **Eleuther ▷ #**[**announcements**](https://discord.com/channels/729741769192767510/794042109048651818/1308867389319942154) (1 messages):

> - `Linear vs Affine Representation`
> - `ACE Method for Control in Language Models`
> - `Refusal Behavior in Language Models`

- **Clarifying Linear Representation Hypothesis**: The paper highlights an ambiguity in defining a **linear representation**: it questions if it should be considered a *linear* function that maintains the origin point or an *affine* function that does not.
  
  - This distinction is significant as prior findings, particularly by **Arditi et al.**, rely heavily on the interpretation, leading to misleading results in models like **RWKV**.
- **Introducing ACE for Affine Control**: The authors propose the new **ACE (Affine Concept Editing)** method, which treats concepts as affine functions to enhance control over model responses.
  
  - ACE allows for projecting activations onto hyperplanes, demonstrating improved precision in managing model behavior as shown through tests on **Gemma**.
- **Reliable Control Over Refusal Responses**: ACE was tested on ten models, including **Llama 3 70B**, where it achieved better control over refusal behavior across harmful and harmless prompts.
  
  - The method improves upon past techniques, indicating a more reliable strategy for steering model actions.
- **Research Contribution Invitation**: To continue improving upon the ACE method, the authors invite interested individuals to introduce themselves in a specific research channel.
  
  - Contributors are thanked for their efforts, emphasizing community collaboration in advancing this research.
- **Access Research Materials**: The GitHub repository for the project can be found at [steering-llama3](https://github.com/EleutherAI/steering-llama3) along with the linked paper on [arXiv](https://arxiv.org/abs/2411.09003).
  
  - Additional insights and discussions can be followed via the [Twitter thread](https://x.com/norabelrose/status/1859307287112007896) by the author.

**Links mentioned**:

- [GitHub - EleutherAI/steering-llama3](https://github.com/EleutherAI/steering-llama3): Contribute to EleutherAI/steering-llama3 development by creating an account on GitHub.
- [Refusal in LLMs is an Affine Function](https://arxiv.org/abs/2411.09003): We propose affine concept editing (ACE) as an approach for steering language models' behavior by intervening directly in activations. We begin with an affine decomposition of model activation vect...
- [Tweet from Nora Belrose (@norabelrose)](https://x.com/norabelrose/status/1859307287112007896): In this paper, we point out an ambiguity in prior work on the linear representation hypothesis: Is a linear representation a linear function— one that preserves the origin point— or an affine functio...

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1308771416170037249) (20 messages🔥):

> - `GPGPU Performance`
> - `PyTorch Optimization Techniques`
> - `Data Loading Strategies`
> - `GPU Memory Management`

- **GPGPU's Game-Changer Era**: In the early days of GPGPU, coding on **G200** was **30x faster** than a decent CPU, leading to widespread adoption in supercomputers.
  
  - Nvidia’s ability to scale due to its consumer industry roots made their GPUs the **sensible choice** for high-performance computing.
- **PyTorch Performance Optimization Shortcuts**: To enhance **PyTorch performance**, storing data in memory on the GPU significantly speeds up processes compared to using the data loader.
  
  - Suggestions like using **HLB-CIFAR10** or **Airbench** were shared for achieving better performance in convolutional networks.
- **Switching Data Formats for Speed**: Converting data to **UINT8** until the last moment can **halve memory bandwidth requirements**, increasing transfer speed.
  
  - Converting data on the GPU just before usage ensures efficient memory transfer and processing.
- **Balancing CPU and GPU Data Loading**: When using a CPU and GPU data loading pipeline, it’s crucial to avoid being **bottlenecked by CPU workers**.
  
  - Ensuring CPU efficiency supports smoother and faster data flow to the GPU, enhancing overall model training performance.
- **Referencing Optimized Practices**: Details were shared on optimizing training strategies, including links to resources like [David Page's bag of tricks](https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb).
  
  - Community contributions noted past efforts and established practices that optimize deep learning model training.

 

**Link mentioned**: [cifar10-fast/bag_of_tricks.ipynb at master · davidcpage/cifar10-fast](https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb): Contribute to davidcpage/cifar10-fast development by creating an account on GitHub.

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1308805502054039592) (125 messages🔥🔥):

> - `Latent Actions and Inverse Dynamics Models`
> - `nGPT Baseline Bugs`
> - `Use of Position Embeddings`
> - `Document Masking Impact on Training`
> - `Forgetting Transformer`

- **Exploration of Latent Actions Papers**: A user inquired about the best papers on **latent actions** and **inverse dynamics models**, hinting at interest in state-of-the-art (sota) works.
  
  - No specific papers were provided, but the conversation hinted at the importance of relevant literature in these areas.
- **nGPT's Baseline Evaluation Complications**: Members discussed inconsistencies in nGPT's published code and baseline evaluations, particularly bugs affecting comparison metrics.
  
  - One noted that internal and external code divergences made effective evaluation nearly impossible, leading to skepticism around results.
- **Position Embedding Innovations**: Discussions revolved around novel approaches to position embeddings, particularly one involving a cumulative width calculation for attention mechanisms.
  
  - Related papers like the **Forgetting Transformer** and new **Contextual Position Encoding (CoPE)** were mentioned as they address issues in traditional position embeddings.
- **Impact of Document Masking on Model Performance**: The group debated the influence of document masking techniques, revealing a decrease in token requirements without significantly compromising training performance.
  
  - Concerns were raised about the fairness of evaluations due to potential advantages from changes in data delivery methods, like biases from document boundaries.
- **Questions on Effective Positioning Strategies**: Different strategies for addressing attention issues using position embeddings were proposed, including the potential merit of simpler methods over complex mappings.
  
  - Members analyzed whether approaches like **ALiBi** or **RoPE** might integrate better than the alternatives presented in recent research.

**Links mentioned**:

- [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/abs/2405.18719): The attention mechanism is a critical component of Large Language Models (LLMs) that allows tokens in a sequence to interact with each other, but is order-invariant. Incorporating position encoding (P...
- [Tweet from YouJiacheng (@YouJiacheng)](https://x.com/YouJiacheng/status/1859353724713566290): @hi_tysam This is a sliding window, information can still propagate if there are >1 layers.
- [Forgetting Transformer: Softmax Attention with a Forget Gate](https://openreview.net/forum?id=q2Lnyegkr8): An essential component of modern recurrent sequence models is the forget gate. While Transformers do not have an explicit recurrent form, we show that a forget gate can be naturally incorporated...
- [koszarskyb - Overview](https://github.com/koszarskyb): GitHub is where koszarskyb builds software.
- [GPT baseline block computation error · Issue #1 · NVIDIA/ngpt](https://github.com/NVIDIA/ngpt/issues/1#issuecomment-2484596258): Hello, thank you very much for open sourcing nGPT. I have found an error in the block computation of the GPT (use_nGPT=0) baseline. The computation being done is : x = norm(x) + attn(norm(x)) x = n...
- [modded-nanogpt/records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt at master · KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt): NanoGPT (124M) quality in 7.8 8xH100-minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1308914897366552656) (2 messages):

> - `Scaling Laws in Language Models`
> - `Evaluation Science Advocacy`
> - `Marius Hobbhahn`

- **New Scaling Laws Paper Emerges**: A recently published paper titled *Understanding How Language Model Performance Varies with Scale* can be accessed [here](https://arxiv.org/abs/2405.10938), detailing an observational approach to scaling laws based on ~100 publicly available models.
  
  - The authors propose that language model performance is more a function of a **low-dimensional capability space** rather than just training across multiple scales.
- **Marius Hobbhahn Champions Evaluation Science**: A member pointed out that **Marius Hobbhahn at Apollo** is one of the most prominent advocates for advancing the science of evaluation methods within the AI community.
  
  - This seems to highlight a growing interest in enhancing rigorous evaluation practices in AI model development.

 

**Link mentioned**: [Observational Scaling Laws and the Predictability of Language Model Performance](https://arxiv.org/abs/2405.10938): Understanding how language model performance varies with scale is critical to benchmark and algorithm development. Scaling laws are one approach to building this understanding, but the requirement of ...

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1308927065319538769) (15 messages🔥):

> - `Zero-shot benchmarking for pruned models`
> - `WANDA pruning method`
> - `lm_eval library compatibility`
> - `Model evaluation on ADVBench`
> - `vllm model usage`

- **Question on zero-shot benchmarking for pruned models**: A member inquired if **lm-eval** supports zero-shot benchmarking for pruned models, mentioning that they are using a method called **WANDA**.
  
  - They expressed some concerns regarding the **suspect results** they were obtaining from zero-shot evaluation.
- **Compatibility of lm_eval with pruned models**: A user discussed modifications made to **lm_eval** to accommodate pruned models, noting that the version they have is quite old.
  
  - They questioned whether the **current version** supports both quantized and pruned models.
- **Evaluation on ADVBench using vllm**: The conversation revealed that a member is evaluating their models on **ADVBench** using **vllm**, and shared their method for running inference.
  
  - They provided the line of code used for generating outputs: **vllm_outputs = model.generate(dialogs, sampling_params)**.
- **Loading the pruned model**: The method for loading the model was shared as **from vllm import LLM; vllm_model = LLM(hf_model_path, tokenizer, dtype = 'bfloat16', swap_space = 128)**.
  
  - The **hf_model_path** denotes the path to the pruned model, while the role of **swap_space** was clarified as a later point of discussion.
- **Actively troubleshooting model inference**: A member is actively seeking to clarify their usage of pruned models for inference, asking about the **swap_space** argument.
  
  - They mentioned they would revisit the question later for further insights on their concerns.

 

**Link mentioned**: [GitHub - locuslab/wanda: A simple and effective LLM pruning approach.](https://github.com/locuslab/wanda?tab=readme-ov-file#zero-shot-evaluation)): A simple and effective LLM pruning approach. Contribute to locuslab/wanda development by creating an account on GitHub.

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1308668094805311499) (132 messages🔥🔥):

> - `Perplexity vs. ChatGPT`
> - `Referral Code Usage`
> - `Perplexity Shopping Feature`
> - `API Functionality`
> - `Image Creation on iOS`

- **Users Compare Perplexity with ChatGPT**: Users shared experiences using Perplexity compared to ChatGPT, noting that **ChatGPT** is often preferred for its versatility and superior conversational ability.
  
  - Some users, despite being **Pro** users of Perplexity, expressed frustration over its perceived limitations and noted a higher frequency of use for ChatGPT.
- **Referral Codes Clarification**: A user inquired about using multiple referral codes on the same account after using the first one, and it was clarified that referral codes apply only to new subscribers.
  
  - This means once a user has had **Pro** status, they can't utilize another referral code for discounts.
- **Discussion on Shopping Feature**: Questions arose regarding the new **Perplexity Shopping** feature, particularly whether it was exclusive to the US market.
  
  - Users expressed interest and sought clarification on potential access limitations for the shopping functionality.
- **Concerns Over API Functionality**: Several users reported issues with **API** responses remaining unchanged despite switching models, leading to confusion and frustration.
  
  - This led to discussions about perceived deficiencies in the platform's flexibility and response diversity.
- **Limited Image Creation on iOS**: A user asked if creating images was possible on the **iOS app**, revealing that this functionality is limited to iPad users.
  
  - This limitation spurred additional conversations about the app's capabilities across different devices.

**Links mentioned**:

- [Supported Models - Perplexity](https://docs.perplexity.ai/guides/model-cards): no description found
- [Perplexity Supply](https://perplexity.supply/): Where curiosity meets quality. Our premium collection features thoughtfully designed apparel for the the curious. From heavyweight cotton essentials to embroidered pieces, each item reflects our dedic...
- [no title found](https://docs.perplexity.ai/guides/getting-started```): no description found
- [Crypto Meets Real Estate: South Africans Can Now Buy Property With Bitcoin](https://techfinancials.co.za/2024/11/18/crypto-meets-real-estate-south-africans-can-now-buy-property-with-bitcoin/',): In a first for South Africa's property industry, buyers can now purchase real estate using cryptocurrency through secure and fully compliant transactions.
- [How Many Stars Are in the Universe](https://cosmicvoage.com/how-many-stars-are-in-the-universe/',): How Many Stars Are in the Universe. It is vast and magnificent, and it is with such beauty that humanity continues to speculate.
- [The Milky Way's 100 Billion Planets - NASA](https://www.nasa.gov/image-article/milky-ways-100-billion-planets/',): This artist's illustration gives an impression of how common planets are around the stars in the Milky Way. The planets, their orbits and their host stars are all vastly magnified compared to the...
- [How Many Stars in the Milky Way ?](https://www.youtube.com/watch?v=Fpgfd6FHQxg']): Ever wondered how many stars twinkle in the vast expanse of the Milky Way Galaxy? 🌌 In this mesmerizing 120-second animated journey, we delve into the encha...

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1308646352560980049) (9 messages🔥):

> - `Web App Fullstack with Next.js`
> - `Chicken or Egg Paradox Solved`
> - `Michelin Star Cities`
> - `NVIDIA Chips Overheat`
> - `Stock Monitoring for Qubit`

- **Web App Fullstack Development Sample**: A resource discussing [fullstack Next.js development](https://www.perplexity.ai/search/webapp-fullstack-nextjs-hono-b-W1pCGRCUSJmPBFklgsg.7w) has been shared, providing insights into modern web application frameworks.
  
  - *Explore the use of Hono for server-side routing!*
- **Final Resolution of Chicken or Egg Paradox**: The age-old **chicken or egg paradox** has been resolved with detailed insights found [here](https://www.perplexity.ai/page/chicken-or-egg-paradox-solved-i_BYFB5DQ6W8XeXOr_4eXQ).
  
  - *Diving into evolutionary biology, this article clarifies the origins!*
- **Cities with Most Michelin Stars Revealed**: A discussion about which city boasts the most **Michelin stars** can be found [here](https://www.perplexity.ai/search/city-with-the-most-michelin-st-NgnS7MxURLOQO4cReb.j6A).
  
  - *Review the culinary rankings of acclaimed global cities!*
- **NVIDIA Chips Experiencing Overheating Issues**: Concerns around **NVIDIA AI chips overheating** are raised in this report [here](https://www.perplexity.ai/page/nvidia-ai-chips-overheat-SRXQJH9yQ8ebTG_KeAT46A).
  
  - *Discussion highlights the risks associated with prolonged use!*
- **Keep Tracking Qubit Stock**: A call to action for monitoring **Qubit stock** is shared along with insights [visible here](https://www.perplexity.ai/page/qubit-stock-N9_yIkN5RbGoYzs2L___Lg).
  
  - *Investors are advised to stay vigilant!*

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1308808204880318615) (1 messages):

> - `Perplexity API`
> - `Domain Filtering`

- **Filtering issues with Perplexity API**: A user highlighted difficulties using the **Perplexity API** to filter search results by blacklisting a specific domain.
  
  - They expressed frustration as the intended excluded domain continued to appear in the results, questioning potential formatting requirements they might be missing.
- **Troubleshooting Filtered Domain Results**: Discussion centered around the effectiveness of **domain filtering** in the Perplexity API.
  
  - Clarification was sought regarding the specifics of format settings that could affect the visibility of blacklisted domains.

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1308661803219423253) (120 messages🔥🔥):

> - `Gemini 1114 performance`
> - `DeepSeek updates`
> - `Prompt caching`
> - `GPT-4o model issues`
> - `RP model comparisons`

- **Gemini 1114 struggles with input handling**: Users reported that **Gemini 1114** often ignores image inputs during conversations, leading to hallucinated responses, unlike models such as **Grok vision Beta**.
  
  - Members are hoping for confirmation and fixes, expressing frustration over recurring issues with the model.
- **DeepSeek launches new reasoning model**: A new model, **DeepSeek-R1-Lite-Preview**, was announced, boasting enhanced reasoning capabilities and performance on AIME & MATH benchmarks.
  
  - However, some users noted the model’s performance is slow, prompting discussions about whether **DeepInfra** might be a faster alternative.
- **Clarifications on prompt caching**: Prompt caching is available for specific models like **DeepSeek**, with users questioning the caching policies of other providers.
  
  - Some members discussed how caching works differently between systems, particularly noting **Anthropic** and **OpenAI** protocols.
- **Issues with GPT-4o model description**: Users identified discrepancies in the newly released **GPT-4o**, noting the model incorrectly listed an **8k context** and wrong descriptions linked to **GPT-4**.
  
  - After highlighting the errors, members saw quick updates and fixes to the model card, restoring accurate information.
- **Comparisons of RP models**: Members discussed alternatives to **Claude** for storytelling and role-playing, with suggestions for **Hermes** due to its perceived quality and cost-effectiveness.
  
  - Users indicated a mix of experiences with these models, with some finding **Hermes** preferable while others remain loyal to **Claude**.

**Links mentioned**:

- [Tweet from DeepSeek (@deepseek_ai)](https://x.com/deepseek_ai/status/1859200141355536422?s=46&t=2a7uDiV3mox9o-E5jIFbLQ): 🚀 DeepSeek-R1-Lite-Preview is now live: unleashing supercharged reasoning power! 🔍 o1-preview-level performance on AIME & MATH benchmarks. 💡 Transparent thought process in real-time. 🛠️ Open-sour...
- [Yi Large - API, Providers, Stats](https://openrouter.ai/01-ai/yi-large): The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service. Run Yi Large with API
- [anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb): A collection of notebooks/recipes showcasing some fun and effective ways of using Claude. - anthropics/anthropic-cookbook
- [GPT-4o (2024-11-20) - API, Providers, Stats](https://openrouter.ai/openai/gpt-4o-2024-11-20): The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability with more natural, engaging, and tailored writing to improve relevance & readability. It’s also better at working with...
- [Prompt Caching | OpenRouter](https://openrouter.ai/docs/prompt-caching#deepseek): Optimize LLM cost by up to 90%
- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing): Route requests across multiple providers

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1308662164961103902) (6 messages):

> - `Custom provider keys`
> - `Key integration access`
> - `Anthropic Claude 3.5 Sonnet`
> - `x-ai/grok-beta`
> - `xai`

- **Multiple Requests for Custom Provider Keys**: Several members have expressed their desire to request a **custom provider key**, including for **x-ai/grok-beta** and **Anthropic Claude 3.5 Sonnet**.
  
  - *One user noted that they already have an account with credits* that would be beneficial for use with OpenRouter.
- **Inquiries about Key Integration Access**: A member inquired about the process to gain **access to key integration**, expressing enthusiasm to test it out.
  
  - This shows an ongoing interest in exploring available features and tools.

 

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1308690107003437068) (58 messages🔥🔥):

> - `Model Loading Issues`
> - `System Requirements for Models`
> - `Optimizing Performance with Limited Hardware`
> - `Exploring Cloud-Based Solutions`
> - `Model Recommendations and Preferences`

- **Model Loading Challenges on MacBook**: A user faced difficulties loading a local model in LM Studio due to insufficient system resources, noting the error indicates overloading risks for their **36GB RAM M3 MacBook**.
  
  - Another member advised that **32B models** are too large for this setup, suggesting a maximum of **14B**.
- **Understanding Model System Requirements**: Discussions revealed that estimating required RAM can be done by adding **10% to the model's file size**, although it was suggested to pick smaller models for better performance.
  
  - It was noted that larger models can block available memory, reducing functionality for other tasks on the machine.
- **Exploring Optimizations for Performance**: To improve performance with a **1050 Ti GPU**, suggestions included using smaller model sizes, reducing context size, and ensuring efficient coding practices.
  
  - A user mentioned that cloud-based hardware rental could be a cost-effective solution when local hardware isn't adequate.
- **Cloud-Based Model Usage**: One member shared their shift to renting cloud servers for model usage, finding it financially beneficial compared to purchasing hardware, with costs ranging from **$25 to $50 per month**.
  
  - This approach allows for accessing high-speed models without the limitations of local hardware.
- **Model Recommendations and User Preferences**: Users recommended various models including **Hermes 3**, **Lexi Uncensored V2**, and **Goliath 120B**, noting personal preferences based on performance and writing quality.
  
  - It was suggested to experiment with different models to find the best fit for individual use cases, with encouragement to try new options as they become available.

**Links mentioned**:

- [NousResearch/Hermes-3-Llama-3.1-70B · Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B): no description found
- [Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct): no description found

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1308675045199183912) (64 messages🔥🔥):

> - `VM Performance with Qwen Models`
> - `Hardware Requirements for DeepSeek v2.5 Lite`
> - `Workstation Design for LLMs`
> - `GPU Selection for AI Workloads`
> - `Fine-tuning vs. Running Models`

- **VM struggles with Qwen models without a GPU**: A member reported that running Qwen 2.5 models on a virtual machine with no GPU resulted in severely limited performance, achieving only about **1 token/second**.
  
  - Another user clarified that CPU-only inference can be incredibly slow, and a GPU would significantly improve this situation.
- **RAM and GPU Requirements for DeepSeek v2.5 Lite**: For running **DeepSeek v2.5 Lite**, it was suggested that at least **24GB VRAM** is needed for the Q4_K_M variant; full Q8 demands about **48GB VRAM**.
  
  - Members discussed that NVIDIA cards are preferred due to AMD’s driver instabilities affecting performance.
- **Guidance on Workstation for LLMs**: A user is seeking advice on building a workstation for fine-tuning LLMs with a budget of **$30,000 to $40,000** and is weighing options like NVIDIA **A6000s** vs. fewer high-end options like the **H100**.
  
  - The discussion emphasized the importance of video memory and flexibility regarding used hardware for budget constraints.
- **GPU Selection for Optimal Performance**: It was noted that using **multiple 24GB 3090s** could provide a viable alternative to expensive newer models, despite lacking NVLink performance.
  
  - One member highlighted a resource comparing benchmarks of various GPUs on token/second performance for LLM inference.
- **Understanding Fine-tuning vs Running Models**: Fine-tuning a model consumes significantly more resources compared to running one, necessitating higher memory and compute power.
  
  - Members reflected on the potential of dedicated AI chips as a solution to the hardware challenges associated with running large models.

 

**Link mentioned**: [GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference): Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference? - XiongjieDai/GPU-Benchmarks-on-LLM-Inference

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1308671878063460472) (102 messages🔥🔥):

> - `Gaming PC Recommendations`
> - `Consistent Character Creation`
> - `AI Models for Substance Designer`
> - `GPU Utilization for Video Generation`
> - `Drawing AI Demonstrations`

- **Advice on Building a Gaming PC**: A user is seeking recommendations for a gaming PC within a budget of **$2500**, asking for suggestions on both components and where to purchase.
  
  - They encourage others to send direct messages for personalized advice.
- **Challenges in Character Consistency for Picture Books**: A member asked how to maintain a consistent character design throughout a picture book, struggling with variations from multiple generated images.
  
  - Suggestions included using **FLUX** or image transformation techniques to improve consistency.
- **Exploration of AI Models for Texture Creation**: A discussion arose about whether AI models could effectively replace **Substance Designer**, highlighting the need for further exploration in that area.
  
  - Members shared their thoughts on the capabilities of different AI models and their performance.
- **Optimizing GPU Usage for AI Video Generation**: Users discussed the difficulties of performing AI video generation on limited VRAM GPUs, noting potential for slow processing times.
  
  - The recommended course of action included clearing VRAM and using more efficient models like **CogVideoX**.
- **Understanding Fast AI Drawing Techniques**: A member inquired about the technology behind AI drawing representations that update quickly on screen, wondering about its implementation.
  
  - Responses indicated that it often relies on powerful GPUs and consistency models to achieve rapid updates.

**Links mentioned**:

- [THUDM/CogVideoX1.5-5B-I2V · Hugging Face](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V): no description found
- [thud (Adam M)](https://huggingface.co/THUD): no description found
- [ByteDance/Hyper-SD · Hugging Face](https://huggingface.co/ByteDance/Hyper-SD): no description found
- [GitHub - kijai/ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper): Contribute to kijai/ComfyUI-CogVideoXWrapper development by creating an account on GitHub.

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1308644641016647732) (17 messages🔥):

> - `Audio Generation in NotebookLM`
> - `External Access to Notebooks`
> - `Podcast Creation`
> - `Transcription Features`
> - `Customization Recommendations`

- **Generating Podcasts with NotebookLM**: One member showcased their podcast featuring AI characters discussing **climate innovation**, emphasizing the multiple steps involved including using an AI tool and NotebookLM.
  
  - They shared a [link to their podcast](https://preview.getsubly.com/preview/89c122b6-cc30-458a-9d2b-d3728098b255) and detailed their process of creating a conversation between diverse characters.
- **Accessing Notebook Features for External Operators**: A conversation around simplifying access for external operators to NotebookLM arose, with one member revealing they utilized a business Gmail to streamline the process.
  
  - They mentioned having created various reference guides already accessible in their quickstart manual folder.
- **Transcription Options for Audio Files**: New users were advised to upload their generated audio files back to NotebookLM, which would transcribe the content for them.
  
  - Alternatively, one member suggested using MS Word's Dictate...Transcribe function for audio files.
- **Feedback on Podcast Audio Creation**: Members shared their experiences with creating podcasts using NotebookLM, highlighting effective audio generation features.
  
  - One user shared their German-language podcast [on Spotify](https://open.spotify.com/show/5OapAqDLaWMxAzqXgywBZH?si=2e422be55d784fde) and expressed interest in customization recommendations.
- **Direct Links to Notebook Audio**: Multiple members shared links related to audio content generated from NotebookLM, including personal podcasts and specific episodes that discuss unique topics.
  
  - One notable episode referenced wine aging in microgravity, citing specific scientific experiments and outcomes.

**Links mentioned**:

- [no title found](https://notebooklm.google.com/notebook/c92bf58d-3a48-4462-9801-964d86829a1c/audio): no description found
- [Wein & Wahnsinn: Verrückte Fakten in 5 Minuten](https://open.spotify.com/show/5OapAqDLaWMxAzqXgywBZH?si=2e422be55d784fde): Podcast · bbrocher · Willkommen bei Wein & Wahnsinn: Verrückte Fakten in 5 Minuten, dem Podcast, der Sie in die skurrile, absurde und oft unerwartete Welt des Weins entführt. Hier erwarten Sie Ane...
- [no title found](https://notebooklm.google.com/notebook/c92bf58d-3a48-4462-9801-964): no description found
- [Anti Schwerkraft Weine](https://open.spotify.com/episode/35ahZlYN43acsu8rNTJyYD?si=16d8b1e6740a4b99): Wein & Wahnsinn: Verrückte Fakten in 5 Minuten · Episode
- [Subly - Story - Delhi Air Pollution](https://preview.getsubly.com/preview/89c122b6-cc30-458a-9d2b-d3728098b255): Need to add subtitles to your videos? Try Subly for free! Within minutes we automatically transcribe and translate your video or audio. You can style, add your brand logo and headline ready to be shar...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1308679850931912775) (35 messages🔥):

> - `Combining Notes Feature`
> - `Reliability of Uploaded Sources`
> - `Sharing Notebooks`
> - `Deep Dive Document Generation`
> - `Limitations on Uploading Large Files`

- **Combining Notes Feature is Under Discussion**: Members are discussing the existing 'Combine to note' feature, which allows users to join multiple notes into a single document.
  
  - One member expressed confusion about converting notes into actual sources, questioning its utility since combining notes is already possible.
- **Opinions on Reliability of Uploaded Sources**: Members have shared mixed experiences with hallucinations when using uploaded sources, with some finding it reliable, while others cite discrepancies.
  
  - A member noted that citations often come up as high-quality and do not fall into typical hallucination pitfalls.
- **Challenges in Sharing Notebooks**: A user inquired about the process of sharing notebooks with friends, having difficulty in executing it successfully.
  
  - Another member confirmed there is a 'share note' button located in the top right corner of the interface for this purpose.
- **Deep Dive Document Generation Sparks Interest**: The potential to create and summarize notes into a single document has generated conversation among members, although some find it redundant.
  
  - One member mentioned being able to compile summaries, expressing that it could be more useful if downloading sources becomes available.
- **Limitations on Uploading Large Files Explored**: A member encountered errors while attempting to upload a large CSV file containing over 444,000 rows, finding a limit at around 10,000 rows.
  
  - They sought confirmation from others about any imposed file size limits within the platform.

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1308767374060687360) (48 messages🔥):

> - `DeepSeek-R1-Lite-Preview`
> - `GPT-4o Update`
> - `Truffles Hardware Device`
> - `Vercel Acquires Grep`
> - `Claude Availability Issues`

- **DeepSeek-R1-Lite-Preview Launch**: DeepSeek announced the go-live of [DeepSeek-R1-Lite-Preview](http://chat.deepseek.com), featuring impressive performance on AIME & MATH benchmarks with a transparent reasoning process.
  
  - The system shows improvements as reasoning length increases, with users excited about its potential applications across various tasks.
- **New GPT-4o Update Released**: OpenAI’s new GPT-4o snapshot, released as `gpt-4o-2024-11-20`, enhances creative writing and is more capable of handling uploaded files for insights.
  
  - In recent performance tests, GPT-4o reclaimed the top rank in various categories, showcasing significant advancements.
- **Truffles Hardware Device Recognized**: Users identified 'Truffles' as the semi-translucent hardware device designed for self-hosting LLMs at home, humorously referred to as a 'glowing breast implant'.
  
  - This quirky description reflects a light-hearted conversation surrounding innovative home LLM solutions.
- **Vercel Acquires Grep**: Vercel announced the acquisition of [Grep](https://grep.app/), which enables developers to search code across over 500,000 public repositories.
  
  - Founder Dan Fox will join Vercel's AI team to enhance code search capabilities, aiming to improve development efficiency.
- **Claude Faces Availability Issues**: Users reported intermittent availability issues with Claude, with some experiencing downtimes while others found it operational.
  
  - Discussions ensued about the service's reliability, causing some users to check for updates on social media.

**Links mentioned**:

- [Tweet from undefined](https://x.com/itsalltruffles): no description found
- [Tweet from Rohan Paul (@rohanpaul_ai)](https://x.com/rohanpaul_ai/status/1847277918243754156?s=46): New Transformer architecture modifications from NVIDIA researchers. nGPT: A hypersphere-based Transformer achieving 4-20x faster training and improved stability for LLMs. \*\*Proposals in this Paper\*\*...
- [Tweet from Teortaxes▶️ (@teortaxesTex)](https://x.com/teortaxesTex/status/1859295840768229880): btw: scaling test time compute with pivot tokens will have great synergy with tool use. Look: it's already yearning to come to exist, begging for a vessel. Just as prophesied by @gwern and @xenoco...
- [Tweet from jack morris (@jxmnop)](https://x.com/jxmnop/status/1858627599981048211?s=46): more fun open-source research news - new paper drops (nGPT) - claims 4-20x training speedup over GPT - shocking - very cool - very valuable - community tries to reproduce - doesn't hold up - turn...
- [Tweet from xjdr (@_xjdr)](https://x.com/_xjdr/status/1859272181844422813): whalebros cooked here. Not only does it seem to replicate the o1-preview results, it seems to pretty effectively replicate (at least parts of) the process. My guess is it uses something very similar ...
- [Tweet from wh (@nrehiew_)](https://x.com/nrehiew_/status/1859265550767067518): Rumor is that DeepSeek R1-Lite is a 16B MOE with 2.4B active params if true, their MATH scores went from 17.1 -> 91.6 Quoting Phil (@phill__1) @nrehiew_ From their wechat announcement:
- [Tweet from DeepSeek (@deepseek_ai)](https://x.com/deepseek_ai/status/1859200141355536422): 🚀 DeepSeek-R1-Lite-Preview is now live: unleashing supercharged reasoning power! 🔍 o1-preview-level performance on AIME & MATH benchmarks. 💡 Transparent thought process in real-time. 🛠️ Open-sour...
- [Tweet from Akshay Agrawal (@akshaykagrawal)](https://x.com/akshaykagrawal/status/1858933658025160719?s=46): My co-founder @themylesfiles and I have started Marimo Inc. to keep building the @marimo_io notebook and other Python data tools. We've raised a $5M seed round led by @antgoldbloom and @shyammani...
- [Tweet from OpenAI Developers (@OpenAIDevs)](https://x.com/OpenAIDevs/status/1859296408131731592): This new GPT-4o snapshot is now available in the API as `gpt-4o-2024-11-20`: https://platform.openai.com/docs/models#gpt-4o. Quoting OpenAI (@OpenAI) GPT-4o got an update 🎉 The model’s creative w...
- [Tweet from Phil (@phill__1)](https://x.com/phill__1/status/1859263165000729024): @nrehiew_ From their wechat announcement:
- [Bitcoin billionaire Barry Silbert talks about his next big bet—on 'decentralized AI'](https://fortune.com/crypto/2024/11/20/decentralized-ai-yuma-bittensor-bcg-barry-silbert/): Silbert will be CEO of Yuma, a new DCG subsidiary focused on the AI ecosystem tied to Bittensor blockchain.
- [Tweet from wh (@nrehiew_)](https://x.com/nrehiew_/status/1859268539770900923): It solved Yann Lecun's 7-gear question
- [Vercel acquires Grep to accelerate code search - Vercel](https://vercel.com/blog/vercel-acquires-grep): Announcing the acquisition of Grep to further our mission of helping developers work and ship faster.
- [Tweet from Yaroslav Bulatov (@yaroslavvb)](https://x.com/yaroslavvb/status/1859032271208223191?s=46): There are a couple of independent efforts to reproduce https://github.com/NVIDIA/ngpt . There are bugs, so it could take some time, but I'm bullish on the core idea as represents normalization alo...
- [Tweet from Akari Asai (@AkariAsai)](https://x.com/akariasai/status/1858876162467881015?s=46): 3/ 🔍 What is OpenScholar? It's a retrieval-augmented LM with 1️⃣ a datastore of 45M+ open-access papers 2️⃣ a specialized retriever and reranker to search the datastore 3️⃣ an 8B Llama fine-tuned...
- [Tweet from Rohan Paul (@rohanpaul_ai)](https://x.com/rohanpaul_ai/status/1847277918243754156?s=4): New Transformer architecture modifications from NVIDIA researchers. nGPT: A hypersphere-based Transformer achieving 4-20x faster training and improved stability for LLMs. \*\*Proposals in this Paper\*\*...
- [Tweet from wh (@nrehiew_)](https://x.com/nrehiew_/status/1859228088292213007): hmm thats interesting, most models ive tried this on dont fail at generating the first 10 words, this fails to realise it generated 7 words instead of 10
- [Tweet from Teortaxes▶️ (@teortaxesTex)](https://x.com/teortaxesTex/status/1859224352731828303/photo/1): pretty interesting research focus this guy has Quoting Zhihong Shao (@zhs05232838) Our DeepSeek reasoning model is great on code and math. Try it out!
- [Tweet from wh (@nrehiew_)](https://x.com/nrehiew_/status/1859218213915001157?s=46): The most interesting about the DeepSeek release is that they basically replicated the o1 scaling laws Quoting DeepSeek (@deepseek_ai) 🌟 Inference Scaling Laws of DeepSeek-R1-Lite-Preview Longer R...
- [Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)](https://x.com/lmarena_ai/status/1859307979184689269): Exciting News from Chatbot Arena❤️‍🔥 Over the past week, the latest @OpenAI ChatGPT-4o (20241120) competed anonymously as "anonymous-chatbot", gathering 8,000+ community votes. The result? ...
- [Tweet from Tim Shi (@timshi_ai)](https://x.com/timshi_ai/status/1858937064647258326?s=46): Exciting update! 🌊 Cresta has raised $125m Series D to accelerate building agents for customer experience!
- [Tweet from Ryan Sun (@sun_hanchi)](https://x.com/sun_hanchi/status/1859243238986588166?s=46): Wait, Lite uses the 16B MoE base model😱😱😱 so technically it matches o1-mini instead Imagine the full version… Btw, deepseek might not have enough GPU to RL the full model, so we will see reverse ...
- [Tweet from Teortaxes▶️ (@teortaxesTex)](https://x.com/teortaxestex/status/1859259359630356955?s=46): was biting nails on the edge of my seat here, fr. 10/10 would prompt again. Defeat, turnaround, final fight – and glorious victory. DeepSeek-r1-lite revolutionizes LLM inference by turning it into a d...
- [Tweet from jack morris (@jxmnop)](https://x.com/jxmnop/status/1858895357209403510?s=46): so the error in the transformer impl from nGPT was very easy to make the residual stream propagated as this > x = norm(x) + attn(norm(x)) instead of this > x = x + attn(norm(x)) TLDR this bre...

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/) (1 messages):

cappuchinoraro: thanks

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1308694276040949790) (5 messages):

> - `Triton Tutorial Performance`
> - `GPU Comparisons`
> - `Softmax Kernel Profiling`

- **Triton vs Torch Softmax Performance**: A member compared the performance of Triton's fused softmax operation against PyTorch's native implementation using the [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) on an RTX 3060, noting smoother performance from Triton.
  
  - They highlighted that while Triton performed better overall, there were instances where PyTorch's performance matched or exceeded Triton's.
- **Inconsistent Throughput Observations**: Another member remarked on the disparities in throughput between the Triton tutorial's results and their own observations, suggesting potential differences in GPU hardware affecting outcomes.
  
  - They speculated that performance comparisons might be unreliable across different GPUs and proposed testing with an A100 to see if results stabilize.
- **Profiling Softmax Kernels on 4090**: A member added that they were profiling softmax kernels on a 4090, noting performance metrics with batch sizes fixed to 128 and comparing the results to the Triton tutorial.
  
  - They indicated that their findings were more aligned with the outcomes detailed in the tutorial, though they focused on ops/sec rather than GB/sec.

 

**Link mentioned**: [Fused Softmax — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html): no description found

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1308721932853706803) (3 messages):

> - `Readme Updates`
> - `Torchchat and Torchtune Linkage`

- **Readme Needs Small Update**: A member suggested that the Readme should mention that **torchchat** is also linked to **torchtune**.
  
  - This led to another member agreeing and providing a [link to a relevant GitHub pull request](https://github.com/pytorch/ao/pull/1319) that addresses the change.
- **GitHub Pull Request for Update**: The mentioned GitHub pull request by **drisspg** aims to update the **README.md** with necessary information.
  
  - It was noted that the pull request is related and provides a comprehensive update on the topic, reflected in a GitHub [link](https://github.com/pytorch/ao/pull/1319).

 

**Link mentioned**: [Update README.md by drisspg · Pull Request #1319 · pytorch/ao](https://github.com/pytorch/ao/pull/1319): no description found

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1308874151301353504) (2 messages):

> - `Ticket Price Changes`
> - `Buying Tickets Early`

- **Frequency of Ticket Price Changes**: A member inquired about how frequently ticket price changes occur and whether it would be wise to buy now instead of delaying.
  
  - Another member responded that **it's usually cheaper** to purchase tickets sooner, emphasizing that alerts are more beneficial if you're several months away from travel.
- **Advice on Buying Tickets Early**: A suggestion was made that buying tickets earlier generally results in lower prices.
  
  - *Alerts for price changes are more useful* for those planning trips several months ahead rather than last-minute purchases.

 

---

### **GPU MODE ▷ #**[**webgpu**](https://discord.com/channels/1189498204333543425/1262121239044948009/1308711923084427305) (11 messages🔥):

> - `Metal GEMM Implementations`
> - `WebGPU and Metal Compatibility`
> - `Register Optimization Techniques`
> - `Performance Regressions in Dawn`
> - `AGX Machine Code Disassembly Tools`

- **Philip Turner's Metal GEMM Implementation Insights**: A member highlighted [Philip Turner's repository](https://github.com/philipturner/metal-flash-attention) featuring a Metal GEMM implementation, though mentioned the code used to be more readable.
  
  - They also noted their own Metal GEMM reaches **85-90%** of theoretical max speed, resembling Turner's implementation.
- **WebGPU's Challenges with MMA Intrinsics**: A participant reminisced about issues with shared memory and questioned if WebGPU exposes **MMA intrinsics**, which can be accessed via Metal.
  
  - They acknowledged uncertainty if improvements have been made to compilers regarding this functionality.
- **Optimizing Register Utilization**: A member shared a technique that saved **25 registers** by replacing array access with pointer incrementing from *a[i]* to *a++*.
  
  - They cautioned about the Metal compiler's necessity for heavy optimization, especially moving addressing computation out of hot loops.
- **Dawn's Performance Regression Issues**: Concerns were raised regarding performance regressions in the latest versions of Dawn, particularly in the wgsl-to-Metal workflow after Chrome **130**.
  
  - It was suggested that Chrome **131** improved performance compared to **130**, but it still lagged behind **129**, with potential issues related to UB check code placement.
- **Disassembling AGX Machine Code Tools**: A tool for disassembling **AGX machine code** maintained by Freedesktop developers was shared: [applegpu repository](https://github.com/dougallj/applegpu).
  
  - This resource was referenced in the context of measuring register utilization in compiled code.

 

**Link mentioned**: [Chromium](https://issues.chromium.org/issues/41486305): no description found

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1308703908868395058) (17 messages🔥):

> - `Debugging Assistance`
> - `CUDA Device Mapping`
> - `Model Distribution across GPUs`
> - `Tensor Parallelism`
> - `Hugging Face Sharding Strategy`

- **Debugging Assistance Offered**: A member offered to help with debugging and suggested isolating problems by turning off optimizations one by one.
  
  - Another member confirmed this approach and expressed gratitude for the quick assistance.
- **CUDA Device Mapping Success**: Using `cuda` as the device map worked fine on L4, leading to humorous exchanges about fast verification of the solution.
  
  - This solution was credited to another member, who is expected to provide more insights soon.
- **Model Distribution Issues Discussed**: One member expressed concern that despite using 'auto', the model might only be utilizing one GPU during execution.
  
  - This led to discussions about tensor parallelism and the limitation of not being able to distribute the model across all four GPUs.
- **Observations on Model Usage**: Members discussed their observations regarding how the 'auto' setting distributes the model across GPUs.
  
  - There was uncertainty expressed regarding the default sharding strategy employed by Hugging Face after inspecting usage statistics.
- **Testing Updates from 0x000ff4**: A member provided a brief update on working on tests regarding a project.
  
  - No additional details were shared about the testing process or its objectives.

 

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1308729659390955521) (4 messages):

> - `FLUX inference optimization`
> - `CPU offloading techniques`
> - `GPU performance on different machines`

- **FLUX Optimized for Speed with CPU Offload**: A member reported achieving a **200% speedup** in FLUX inference by implementing per-layer CPU offload techniques on a **4070Ti SUPER**, reaching times as low as **1.23 s/it**.
  
  - Their results indicate improvements over the baseline method of `.enable_sequential_cpu_offload()` which took **3.72 s/it**.
- **Mobicham's Parallel Offloading Insights**: Another member shared their experience with offloading scales and zeros using pinned memory and CUDA streams, noting it worked well on capable machines but was slower on shared instances.
  
  - *They speculated on the efficiency in resource-constrained environments* like runpod/vastai.
- **Discussion on Bottlenecks in LLM Decoding**: In response, a member commented that for LLM decoding, the CPU to CUDA transfer could become a bottleneck, despite methods to overlap data transfer and compute.
  
  - However, with **FLUX** for image generation, slow data transfer is less impactful due to its higher arithmetic intensity.
- **Video Resource Shared for Further Insights**: A member shared a [YouTube link](https://www.youtube.com/watch?v=9Q7jMiXayXE) that may offer additional insights or relevant content related to the discussed optimizations.
  
  - This video could be beneficial for those exploring similar performance enhancements.

 

**Link mentioned**: [Tweet from Thien Tran (@gaunernst)](https://x.com/gaunernst/status/1859168533554565325): Speed up FLUX CPU offloading by 200%. On 4070Ti SUPER (16GB) baseline (.enable_sequential_cpu_offload()): 3.72 s/it + pin memory: 2.09 s/it (+78%) + CUDA stream (explicit synchronization): 1.32 s/it ...

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1308647313366978560) (27 messages🔥):

> - `DeepSeek-R1-Lite-Preview`
> - `AI agents for writing books`
> - `LLM knowledge evaluation`

- **DeepSeek-R1-Lite-Preview Launch**: [DeepSeek-R1-Lite-Preview](https://x.com/deepseek_ai/status/1859200141355536422) is now live, boasting supercharged reasoning power with **o1-preview-level performance** on AIME & MATH benchmarks.
  
  - It features a **transparent thought process** in real-time, with open-source models and API coming soon.
- **Team of AIs writes a book**: A project showcased by [Venture Twins](https://x.com/venturetwins/status/1859298925930479998) involves ten AI agents collaborating to write a fully autonomous book, each with different roles such as setting narrative and maintaining consistency.
  
  - The progress can be tracked through [GitHub commits](https://github.com/Lesterpaintstheworld/terminal-velocity) as they work in real-time.
- **Innovative LLM Benchmark Proposal**: A member proposed a benchmark testing how well an LLM knows what it doesn't know, where correct responses receive no mark.
  
  - The evaluation focuses on how the model responds to incorrectly answered questions, mixing both knowledge and reasoning.

**Links mentioned**:

- [Tweet from DeepSeek (@deepseek_ai)](https://x.com/deepseek_ai/status/1859200141355536422): 🚀 DeepSeek-R1-Lite-Preview is now live: unleashing supercharged reasoning power! 🔍 o1-preview-level performance on AIME & MATH benchmarks. 💡 Transparent thought process in real-time. 🛠️ Open-sour...
- [Tweet from Justine Moore (@venturetwins)](https://x.com/venturetwins/status/1859298925930479998): Someone is using a team of 10 AI agents to write a fully autonomous book. They each have a different role - setting the narrative, maintaining consistency, researching plot points... You can follow ...
- [GitHub - Lesterpaintstheworld/terminal-velocity: A novel created autonomously by 10 teams of 10 AI agents](https://github.com/Lesterpaintstheworld/terminal-velocity): A novel created autonomously by 10 teams of 10 AI agents - Lesterpaintstheworld/terminal-velocity

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1308676993440550983) (8 messages🔥):

> - `Learning Rate Scheduling`
> - `Warmup and Decay Strategies`
> - `Test Time Scaling for LLMs`
> - `Cyclic Learning Rate Schedulers`

- **Understanding Learning Rate Behavior Across Epochs**: A discussion arose regarding whether the **learning rate for a specific step** should match across different epochs. It was clarified that the learning rate typically **ramps up during warmup** and then decays over time, which leads to different values for corresponding steps across epochs.
- **Exploring Learning Rate Scheduler Configurations**: One member suggested that previously, learning rates were stepped at each epoch, but currently, they are configured based on **total steps** across all epochs and adjusted with every gradient step. They encouraged looking into **cyclic learning rate schedulers** as a modern approach.
- **Inquiries About Test Time Scaling for LLMs**: In a separate inquiry, a member asked if anyone was working on **test time scaling** for large language models and requested ideas on the subject. This generated curiosity and hints at ongoing discussions about scaling strategies.

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1308839061980708874) (2 messages):

> - `LLMs Reasoning Abilities`
> - `Generative Agent Simulations`

- **LLMs can reason without prompting**: Research shows that **large language models (LLMs)** can exhibit reasoning paths similar to **chain-of-thought (CoT)** without explicit prompting by altering the decoding process to examine top-$k$ alternative tokens.
  
  - This method highlights the **intrinsic reasoning abilities** of LLMs and suggests that CoT pathways may reside inherently in their sequences.
- **Behavioral simulations of over 1,000 individuals**: A novel architecture simulates the attitudes and behaviors of **1,052 real individuals**, demonstrating that generative agents can replicate human responses with **85% accuracy** on the General Social Survey.
  
  - The architecture reduces accuracy biases across **racial and ideological groups**, paving the way for tools to investigate individual and collective behavior.

**Links mentioned**:

- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200): In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) promptin...
- [Generative Agent Simulations of 1,000 People](https://arxiv.org/abs/2411.10109): The promise of human behavioral simulation--general-purpose computational agents that replicate human behavior across domains--could enable broad applications in policymaking and social science. We pr...

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1308670981618929746) (3 messages):

> - `Soft Prompts`
> - `LLM Optimization`

- **Inquiry on Soft Prompts Investigation**: A member posed a question about whether anyone has investigated the concept of **soft prompts** for LLMs noted in a [post](https://bsky.app/profile/saganite.bsky.social/post/3lbeajzg3ms2f). *They emphasized the potential for soft prompts in optimizing system prompts into embedding space.*
  
  - Another member responded, stating that the idea of soft prompts is **pretty interesting**, suggesting some level of interest in the topic.
- **Discussion on Value of Soft Prompts**: The conversation indicates curiosity surrounding the **untapped potential** of soft prompts in the LLM community. *Members seem to express that more exploration into this area could be fruitful for further advancements.*

 

**Link mentioned**: [@saganite.bsky.social](https://bsky.app/profile/saganite.bsky.social/post/3lbeajzg3ms2f): Really trying to figure out why "soft prompts" aren't used more often with LLMs. For those who aren't familiar, soft prompts are system prompts that have been converted to embedding ...

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1308839061980708874) (2 messages):

> - `LLMs Reasoning without Prompting`
> - `Generative Agent Behavioral Simulations`

- **LLMs Reasoning Effectively Without Prompting**: A study investigates whether large language models (LLMs) can reason effectively without prompting by altering the decoding process, revealing that CoT reasoning paths often emerge naturally. This novel approach allows for an assessment of the LLMs' **intrinsic reasoning abilities** without the complexities of manual prompt engineering.
  
  - The findings suggest that by utilizing **top-k** alternative tokens, researchers can elicit effective reasoning from pre-trained models, providing insights into their inherent capabilities.
- **Groundbreaking Behavioral Simulation of 1,052 Individuals**: New research presents an innovative agent architecture using large language models to simulate the attitudes and behaviors of **1,052** real individuals, based on qualitative interviews. These generative agents accurately replicate participants’ responses on the General Social Survey with **85%** accuracy, matching self-reported answers.
  
  - Notably, the architecture reduces accuracy biases across racial and ideological groups compared to agents that rely solely on demographic descriptions, laying a foundation for tools to explore individual and collective behavior in social science.

**Links mentioned**:

- [Generative Agent Simulations of 1,000 People](https://arxiv.org/abs/2411.10109): The promise of human behavioral simulation--general-purpose computational agents that replicate human behavior across domains--could enable broad applications in policymaking and social science. We pr...
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200): In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) promptin...

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1308645095439994921) (18 messages🔥):

> - `Daily Theme Winner`
> - `API Usage Discussion`
> - `Model Options and Performance`

- **Celebrating Daily Theme Victory**: One member expressed joy after being the first to win the **daily theme** challenge, stating they were 'so happy'.
  
  - This elicited a positive reaction from the community, highlighting engagement with ongoing activities.
- **API Solutions Sought**: A member mentioned searching for an **API** or tool but found both options unsatisfactory, indicating frustration.
  
  - This reflects a broader interest in finding useful resources within the community.
- **Clarification on Model Options**: A discussion emerged regarding the **4o model** and whether it used the **o1 mini** or **o1 preview**, with confirmation it likely used **o1 mini**.
  
  - Another member suggested checking the settings to verify options, promoting hands-on troubleshooting.
- **Swapping Channels Confusion**: Queries arose about whether the **ai-discussions** channel was swapped with another specific channel, indicating possible miscommunication.
  
  - A member expressed apologies for a mix-up, mentioning their intention to move comments to the correct **off-topic** space.

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1308733699793092608) (3 messages):

> - `High Temperature Performance`
> - `Beta Access to o1`
> - `Gaming Character Genshin Impact`

- **High Temp Performance Inquiry**: A member questioned if improved performance at **higher temperatures** could be linked to their prompt style, suggesting too many guiding rules or constraints.
  
  - This raises interesting considerations for optimizing prompt design for better AI responsiveness.
- **Gratitude for Beta Access to o1**: A member expressed excitement and gratitude towards NH for granting them **beta access to o1**, brightening their morning.
  
  - *Woo! Thank you NH for making this morning even brighter* reflects the exhilaration around new updates.
- **Genshin Impact Character Confusion**: One member raised a concern about ChatGPT not retrieving information on **Gaming**, a character from **Genshin Impact**.
  
  - This highlights potential gaps in the AI's knowledge concerning popular game characters and their contexts.

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1308659715097497631) (8 messages🔥):

> - `Using Delimiters in Prompts`
> - `Markdown for Clarity`
> - `Game Mechanics Understanding`
> - `Model Context Expectations`

- **Employing Delimiters for Clarity**: A member shared OpenAI's advice on using delimiters like triple quotation marks or XML tags to help the model interpret distinct sections of the input clearly.
  
  - This approach aids in structuring prompts better for improved model responses, allowing for easier input interpretation.
- **Markdown as a Formatting Tool**: Another member suggested using Markdown syntax to create structured headings and lists for better clarity in prompts.
  
  - Examples included `# Heading 1` for main titles and various list formats, indicating how structured text can enhance the model's understanding.
- **General Purpose Model Adaptability**: A discussion noted that since GPT is a general-purpose model, it may not strictly adhere to game mechanics like those in Tic Tac Toe.
  
  - This highlights the importance of clear context and expectations to guide the model's output when dealing with specific scenarios.
- **Direct Labeling for Model Context**: A member proposed providing explicit labels such as 'Section head: This Topic' to help the model infer context correctly.
  
  - This technique emphasizes the model's reliance on labeling and context to generate more relevant responses.

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1308659715097497631) (8 messages🔥):

> - `Using Delimiters for Clarity`
> - `Markdown Formatting`
> - `Improving GPT's Understanding`
> - `Game Mechanics in GPT`
> - `Model Context and Labeling`

- **Using Delimiters for Clarity in Prompts**: Using delimiters like triple quotation marks or section titles can help clarify distinct parts of the input for the model, according to OpenAI's advice.
  
  - This practice aids the model in interpreting different sections effectively, enhancing overall comprehension.
- **Markdown Formatting Tips**: Markdown can be utilized for creating headings and formatting text, with examples like `# This is Heading 1` and **bold** or *italic* styling shared in the discussion.
  
  - Many members highlighted the usefulness of backtick lines and lists in organizing content clearly.
- **Game Mechanics and GPT Responses**: A member noted that due to GPT's general-purpose nature, it may deviate from simple game mechanics like **Tic Tac Toe**.
  
  - Humorously, they mentioned gaining **25 experience points** while discussing this topic.
- **Model Context and Labeling for Better Interactions**: Participants suggested labeling sections directly, such as using 'Section head: This Topic', to guide the model's understanding.
  
  - Emphasizing that giving context aids the LLM in guessing and pattern matching, enriching its responses.

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1308645952181112925) (12 messages🔥):

> - `API Key Issues`
> - `CORS Errors`
> - `Python Learning Projects`

- **403 Errors Indicate API Issues**: Multiple members discussed encountering **403 errors**, indicating either invalid API keys or calling old endpoints while trying to access certain functionalities.
  
  - One member shared that after checking their API keys, they experienced **fetch errors** and difficulties using the sandbox feature.
- **Inquiry about Free Tier API Limitations**: A member confirmed they were still on the **free tier** and attempted to upgrade to a production key to resolve ongoing issues, but faced further challenges.
  
  - They reported several **CORS errors** in the console, noting that their setup was standard with no additional plugins.
- **Collective Learning with Python**: A member mentioned participating in **30 Days of Python**, sharing their learning project with the group.
  
  - This prompted a general inquiry about other members' ongoing projects, fostering a sense of community and collaboration.

**Links mentioned**:

- [Fetch Errors on Cohere Dashboard](https://imgur.com/a/zMBZQfz): Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...
- [imgur.com](https://imgur.com/uP2tgwp): Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1308709449829187584) (6 messages):

> - `Account-based settings`
> - `Model training prompts`
> - `Bulgarian language datasets`
> - `Model tuning techniques`
> - `Contributing processes`

- **Account-specific configurations**: It was noted that adjustments would be made on a per-account basis, highlighting the need for tailored settings.
  
  - This approach ensures customization for individual user needs.
- **Guiding the command-r**: A suggestion was made to allow command-r to draft system prompts with the user's oversight to enhance performance.
  
  - This could streamline the prompt creation process.
- **Bulgarian language training datasets**: The discussion pointed out that additional training data specific to the **Bulgarian language** would be crucial for model fine-tuning.
  
  - The user offered to gather a dataset and requested that findings be shared in the message thread for team review.
- **Model tuning capabilities**: It was asked whether tuning the model could be done using only a preamble and possibly chat history.
  
  - This raises important questions about the model's adaptability to various training inputs.
- **Seeking assistance for contributions**: A user expressed uncertainty about how to start contributing and requested help to understand the process.
  
  - This indicates a need for clearer guidelines on contribution pathways.

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1308829039607418920) (4 messages):

> - `RAG chatbot issues`
> - `Cohere multi-modal embeddings`
> - `Rate limiting problems`

- **RAG Chatbot Faces API Token Error**: A user reported an error stating `invalid api token` when executing their RAG chatbot code despite using a valid Cohere API key from the [dashboard](https://dashboard.cohere.com/api-keys).
  
  - They provided their code snippet and requested assistance in identifying the source of the error.
- **Praise for Multi-Modal Embeddings and Rate Limit Concerns**: A member expressed excitement over the new multi-modal embeddings for images, noting **fantastic improvements** observed in their applications.
  
  - However, they highlighted a significant issue with the **40 requests per minute rate limit**, which hinders their use case, and sought advice on potential alternatives.

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1308766287882620989) (4 messages):

> - `Harmony Open-Source Project`
> - `Competition for LLM Matching Algorithms`
> - `Data Availability for Harmony`
> - `Natural Language Processing in Harmony`
> - `Discord Community for Harmony`

- **Harmony Project for Questionnaire Harmonization**: The **Harmony** project is designed to retrospectively harmonise questionnaire items and metadata using LLMs, and it is used by researchers for better data compatibility.
  
  - It facilitates comparing instruments and explores potential compatibility across versions and languages via [this tool](https://harmonydata.ac.uk/).
- **Join the Competition to Improve LLM Algorithms**: Harmony is hosting a competition to enhance its LLM matching algorithms, offering prizes for participation, with no prior experience required in LLM training.
  
  - Participants can [register on DOXA AI](https://harmonydata.ac.uk/doxa/) to enter the competition and assist in making **Harmony** more robust.
- **Data Accessibility for Harmony**: Members inquired about the data used in the open-source **Harmony** project, prompting responses regarding its availability.
  
  - The project's code and data can be found on its [GitHub page](https://github.com/harmonydata/harmony).
- **Leveraging Natural Language Processing**: The Harmony project utilizes **Natural Language Processing** for improving the matching of questionnaire items across various studies and languages.
  
  - Further insights into Harmony’s algorithm performance can be explored in a detailed [blog post](https://harmonydata.ac.uk/nlp-semantic-text-matching/measuring-the-performance-of-nlp-algorithms/).
- **Engagement in the Harmony Discord Community**: The project encourages users to join the **Harmony Discord server** to participate in discussions and contribute to the matching challenge.
  
  - Members can access the \***�「matching-challenge」** channel for updates and collaboration.

**Links mentioned**:

- [GitHub - harmonydata/harmony: The Harmony Python library: a research tool for psychologists to harmonise data and questionnaire items. Open source.](https://github.com/harmonydata/harmony): The Harmony Python library: a research tool for psychologists to harmonise data and questionnaire items. Open source. - harmonydata/harmony
- [Harmony | A global platform for contextual data harmonisation](https://harmonydata.ac.uk/): A global platform for contextual data harmonisation
- [Competition to train a Large Language Model for Harmony on DOXA AI | Harmony](https://harmonydata.ac.uk/doxa/): A global platform for contextual data harmonisation

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1308760060159201320) (7 messages):

> - `Post-softmax Scores with sdpa/flex`
> - `Attention Score Calculation`
> - `Flex Attention Updates`
> - `Performance Benchmarking sdpa`

- **Using sdpa/flex for Post-softmax Scores**: To obtain **post-softmax scores** using sdpa/flex, feed in a dummy tensor initialized to the identity matrix and shape [..., seqlen, seqlen]. This approach might require two calls, though it's suggested this isn't necessary by some members.
  
  - The implementation details discussed reference potential changes in **flex's behavior** in version 2.5.1 that might affect the feasibility of this method.
- **Self-calculating Attention Scores**: One member suggested calculating the **attention scores** directly using `torch.softmax(q @ k.transpose(-1, -2), dim=-1)`, providing more control over storage options. They pointed out that since **F.sdpa/flex** implements the flash-attn algorithm, some score recomputation is necessary to save them.
  
  - Another member expressed agreement, noting that this would be a straightforward initial attempt unless there were specific reasons to avoid it.
- **Benchmarking sdpa Approaches**: There was a recommendation to benchmark the proposed method against the **naive sdpa** approach to identify performance differences. The numerical error in scores could vary based on the **sdpa backend** and **data type** used.

 

**Link mentioned**: [pytorch/torch/nn/attention/flex_attention.py at release/2.5 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/release/2.5/torch/nn/attention/flex_attention.py#L909)): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1308808453610930196) (14 messages🔥):

> - `Adaptive Batching Implementation`
> - `Improving DPO Loss Function`
> - `Standard vs. New Research Approaches`
> - `Server Boosts and Nitro Subscription`
> - `Code Structure and Modularity Concerns`

- **Adaptive Batching Aiming for Optimal GPU Utilization**: Discussions centered around implementing [adaptive batching](https://github.com/pytorch/torchtune/pull/2035) that maximizes GPU utilization by adjusting the batch size to avoid OOM errors during training.
  
  - It was suggested that this feature could be added as a flag in future recipes, ideally activated when `packed=True` to maintain efficiency.
- **Evaluating Changes in DPO Loss Structure**: Concerns were raised about the current structure of the TRL code and whether to include recent papers addressing DPO modifications that may not be significant enough.
  
  - A call for clarity was made on whether to remove **SimPO** and any separate classes to keep the DPO recipe clean and straightforward.
- **Preference for Standard Methods Over New Approaches**: There was a consensus that the usual practice is to implement standard methods while allowing for flexibility for new innovative strategies by others in the field.
  
  - Members discussed the potential trade-offs associated with trying new research preprints versus sticking with established techniques.
- **Impacts of Cancelling Nitro Subscription**: A member mentioned that **server boosts** will be removed if a user cancels their **free Nitro** subscription, highlighting the implications for server management.
  
  - This comment drew attention to the value of maintaining subscriptions for uninterrupted server benefits.
- **Deep Dive on TRL Code Challenges**: Feedback was given regarding the **TRL** code's complexity and modularity, specifically the issues arising from multiple checks for different arguments.
  
  - The group discussed the need for simplifying the DPO recipe to ensure it's more hackable, thereby enhancing future development.

 

**Link mentioned**: [Add RPO, DPOP losses, add lambda_dpop to basic DPO loss by krammnic · Pull Request #2035 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/2035): Context What is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here) Please link to any issues this PR addresses. Changelog W...

 

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1308760351541694537) (2 messages):

> - `SageAttention`
> - `Inference Gains`

- **SageAttention Achieves Impressive Speedups**: The project [SageAttention](https://github.com/thu-ml/SageAttention) boasts quantized attention that delivers speedups of **2.1x** and **2.7x** compared to **FlashAttention2** and **xformers**, respectively, while maintaining end-to-end metrics across various models.
  
  - *Pretty cool inference gains here!* implies excitement about the performance improvements presented by SageAttention.
- **Discussion on Inference Gains from SageAttention**: A member expressed their thoughts on the **inference gains** achieved through the implementation of SageAttention, indicating strong performance improvements.
  
  - The topic sparked interest among others, potentially leading to further discussions on its application in various AI models.

 

**Link mentioned**: [GitHub - thu-ml/SageAttention: Quantized Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models.](https://github.com/thu-ml/SageAttention/): Quantized Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models. - thu-ml/SageAttention

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1308670171505758260) (6 messages):

> - `Tinygrad and Triton Integration`
> - `SASS Assembler Questions`
> - `FOSDEM AI DevRoom Presentation`
> - `Tinybox Hackathon Proposal`

- **Inquiry on Tinygrad's Triton Integration**: A user inquired whether **Tinygrad** now has native integration with **Triton**, referencing earlier discussions.
  
  - George Hotz prompted the user to refer to the questions document for clarification.
- **SASS Assembler Intentions**: A discussion arose regarding whether the to-be-written **SASS assembler** is intended to replace **ptxas**.
  
  - One user expressed uncertainty about their question's relevance, with George Hotz suggesting they consult the questions document.
- **Call for Presenters at FOSDEM AI DevRoom**: A community member shared an opportunity for Tinygrad developers to present at the **FOSDEM AI DevRoom** happening on February 2, 2025.
  
  - They emphasized Tinygrad's significance in the AI industry and encouraged interested individuals to reach out for collaboration.
- **Tinybox Hackathon Idea**: The same member proposed organizing a pre-FOSDEM hackathon and invited someone to bring a **Tinybox** on-site for hands-on experiences.
  
  - They expressed enthusiasm to engage community members in discussions over Belgian beer, hoping it would enhance the event.

 

**Link mentioned**: [FOSDEM 2025 - Low-Level AI Engineering & Hacking Dev Room](https://aifoundry.org/fosdem-2025-low-level-ai-engineering-hacking-dev-room): Explore the new "Low-Level AI Hacking & Engineering" Dev Room at FOSDEM, featuring open-source projects powering the AI industry. Submit a session or become a sponsor for this innovative...

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1308923419408207952) (1 messages):

> - `int64 indexing`
> - `huge tensors`

- **Question on int64 Indexing Utility**: A member inquired about the necessity of **int64 indexing** in contexts where **huge tensors** are not being used.
  
  - The discussion seeks to clarify the scenarios or potential advantages of using int64 indexing despite the absence of large tensor applications.
- **Exploration of Indexing Techniques**: The community is delving into various **indexing techniques** used in tensor operations, which may include **int64**, **int32**, and others.
  
  - They are considering the impact of these indexing methods on performance and efficiency in smaller tensor operations.

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1308838379256807444) (2 messages):

> - `Async functions in Mojo`
> - `Mojo library repository`

- **Async functions awaitable in Mojo sync functions**: A member is puzzled about being able to await an async function inside a sync function in **Mojo**, which contrasts with Python's limitations.
  
  - They are seeking clarification or an explanation for this difference in handling async functionality.
- **Inquiry about Mojo library repository**: Another member is curious about the availability of a repository for libraries comparable to **pip** for Mojo.
  
  - They are looking for resources or links that provide access to Mojo libraries.

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1308916166185455768) (5 messages):

> - `Moonshine ASR Model Performance`
> - `Mojo Program Observations`
> - `Max API vs ONNX Performance`

- **Moonshine ASR Model Tested with Max**: A user tested the **Moonshine** ASR model performance using both the Python API for **Max** and a native **Mojo** version, noting both were about **1.8x slower** than the direct **onnxruntime** Python version.
  
  - The **Mojo** and Python Max versions took approximately **82ms** to transcribe 10 seconds of speech, whereas the native **onnxruntime** reached **46ms**.
- **Run Instructions and Observations Shared**: Instructions for running the **Moonshine** ASR model are provided in comments at the top of the **mojo** file that was shared.
  
  - The user's experience highlighted that **passing in TensorMap** into **Model.execute** caused a crash, and manual unpacking of **26 arguments** was necessary due to limitations in **Mojo**.
- **Seeking Performance Improvements in Mojo**: The user expressed that this is one of their first **Mojo** programs and acknowledged that it may not be idiomatic.
  
  - They requested assistance for achieving better performance, emphasizing their eagerness to improve their Mojo and Max skills.

**Links mentioned**:

- [moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1): moonshine.mojo. GitHub Gist: instantly share code, notes, and snippets.
- [moonshine.py](https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862): moonshine.py. GitHub Gist: instantly share code, notes, and snippets.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1308703559465959444) (2 messages):

> - `Tencent Hunyuan Model`
> - `Bits and Bytes on MI300X`

- **Discussion on Tencent Hunyuan Model Fine Tuning**: A member inquired about experiences with fine-tuning the [Tencent Hunyuan model](https://huggingface.co/tencent/Tencent-Hunyuan-Large), sharing various useful links including [GitHub](https://github.com/Tencent/Tencent-Hunyuan-Large) and [official website](https://llm.hunyuan.tencent.com/).
  
  - They provided additional resources such as the [Technical Report](https://arxiv.org/abs/2411.02265) and [Demo](https://huggingface.co/spaces/tencent/Hunyuan-Large) for reference.
- **Using Bits and Bytes on MI300X**: A member shared their successful experience with [Bits and Bytes](https://github.com/bitsandbytes-foundation/bitsandbytes) on the MI300X system, highlighting ease of use.
  
  - They emphasized the importance of remembering the `--no-deps` flag during updates, sharing a one-liner command to force reinstall the package.

 

**Link mentioned**: [tencent/Tencent-Hunyuan-Large · Hugging Face](https://huggingface.co/tencent/Tencent-Hunyuan-Large): no description found

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**community-showcase**](https://discord.com/channels/1104757954588196865/1117851527143493664/) (1 messages):

volko76: Do we still need to prompt correctly ?  
[https://youtu.be/m3Izr0wNfQc](https://youtu.be/m3Izr0wNfQc)

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1308662697252098079) (4 messages):

> - `Axolotl Collab Notebooks`
> - `Continual Pretraining of LLaMA`

- **Inquiry about Axolotl Collab Notebooks**: A user inquired whether Axolotl offers any **collab notebooks** that can be used for **continual pretraining of LLaMA**.
  
  - Phorm responded, indicating they would search **OpenAccess-AI-Collective/axolotl** for relevant information.
- **Undefined Result for Notebook Inquiry**: Phorm's search result returned **undefined**, indicating no current available notebooks for the stated purpose.
  
  - Users were encouraged to check back soon for updates on the availability of these resources.

 

**Link mentioned**: [OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)): Understand code, faster.

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1308822739083202641) (5 messages):

> - `multimodal problems`
> - `vision language models`
> - `mmmu notebook`

- **Juan seeks help with multimodal challenges**: Juan inquired about using the experimental support for **vision language models** while working on a **multimodal problem**.
  
  - *Let me know if there are any issues!* was posed by another member to offer additional assistance.
- **Juan discovers the mmmu notebook**: Juan later found the **mmmu notebook** himself, which provided the support he needed for his project.
  
  - He thanked the community for their *awesome work*, showing appreciation for the resources available.

 

---

### **DSPy ▷ #**[**examples**](https://discord.com/channels/1161519468141355160/1161519685616025600/1308754985961525308) (1 messages):

> - `Semantic Router`
> - `Classification Tasks`

- **Semantic Router as a Benchmark**: A member suggested that the [Semantic Router](https://github.com/aurelio-labs/semantic-router) should serve as the baseline for performance in classification tasks, emphasizing its **superfast AI decision making** capabilities.
  
  - The project focuses on **intelligent processing of multi-modal data**, and it may offer competitive benchmarks we aim to exceed.
- **Focus on Performance Improvement**: There was an assertion that the performance of existing classification tools needs to be surpassed, with the **Semantic Router** as a reference point.
  
  - Discussion revolved around identifying metrics and strategies to achieve better results than the baseline set by this tool.

 

**Link mentioned**: [GitHub - aurelio-labs/semantic-router: Superfast AI decision making and intelligent processing of multi-modal data.](https://github.com/aurelio-labs/semantic-router): Superfast AI decision making and intelligent processing of multi-modal data. - aurelio-labs/semantic-router

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1308839925042642965) (2 messages):

> - `LLM-Native Resume Matching`
> - `Building AI Agents with LlamaIndex`
> - `Webinar on December 12`

- **LLM-Native Resume Matching Solution Launched**: Thanks to [@ravithejads](https://twitter.com/ravithejads), an **LLM-native solution** for resume matching has been developed, enhancing traditional screening methods.
  
  - This innovative approach addresses the **slow and tedious process** of manual filtering in recruitment, offering a more efficient alternative.
- **Join Our Webinar on Building AI Agents**: Learn how to build **data-backed AI agents** with LlamaIndex and [@Redisinc](https://twitter.com/Redisinc) in an upcoming webinar on **December 12**.
  
  - The session will cover architecting agentic systems and best practices for **reducing costs** and optimizing **latency**.

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1308790453147144193) (2 messages):

> - `Extracting table data from PDFs`
> - `Applications for PDF data extraction`

- **Extracting Table Data from PDFs**: A member asked for approaches to extract **table data** from PDF files that contain various elements including text and images.
  
  - They expressed interest in knowing if there are any existing applications that facilitate this process.
- **Inquiry About PDF Data Extraction Apps**: Another member sought recommendations for any applications available that can extract data specifically from PDFs.
  
  - This highlights a need within the community for tools that can handle various PDF complexities.

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1308644726441906226) (4 messages):

> - `New UI Feedback`
> - `Rate Limit Issues`
> - `Interpreter Design`
> - `Future UI Configurations`

- **New UI sparks mixed feelings**: Some users feel the new UI is slightly overwhelming and unclear in directing attention, with one comparing it to a computer from *Alien*. However, others are starting to appreciate its UNIX-inspired design, finding it suitable for 1.0 features.
- **Need for token and rate limit configuration**: A user expressed frustration over being rate limited by Anthropic, noting that current error handling in Interpreter leads to session exits when limits are exceeded. They emphasized the importance of incorporating better rate limit management in future updates.
- **Suggestions for UI improvements**: There are calls for a more informative UI that displays current tools, models, and working directories to enhance usability. Users are also advocating for a potential 'plugin ecosystem' to allow customizable features in future releases.
- **Separation of compute workloads proposed**: One member suggested splitting LLM workloads between local and cloud compute to optimize performance. This reflects a concern about the limitations of the current Interpreter design, which is primarily built for one LLM at a time.

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1308953617528656003) (1 messages):

> - `Intel AMA`
> - `Hackathon Insights`

- **Intel AMA Session Tomorrow**: A **Hackathon AMA** with **Intel** is set for **3 PM PT tomorrow (11/21)**, providing a chance to gain insights directly from Intel specialists.
  
  - Don’t forget to [watch live here](https://www.youtube.com/watch?v=_Wm5guUXt54) and set your reminders!
- **Reminder for Upcoming Event**: @everyone is reminded about the upcoming Intel AMA, emphasizing its importance for gaining knowledge.
  
  - Participants are encouraged to prepare their questions to make the most of the session.

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1308807297564282880) (2 messages):

> - `Registration Issues`
> - `Hackathon vs MOOC Registration`

- **User Faces Registration Confusion**: A user expressed concern about not receiving any emails after joining three different groups and registering with multiple email addresses.
  
  - They are uncertain about whether their registration was successful.
- **Clarification on Event Type**: Another member asked for clarification, wondering if the user was referring to the **hackathon** or the **MOOC registration**.
  
  - This highlights potential confusion among participants regarding different types of registrations.

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1308958351589113877) (1 messages):

> - `Refact.AI`
> - `Autonomous Agents`
> - `Live Demo`
> - `Tooling`

- **Exciting Demo from Refact.AI Team**: The **Refact.AI** team, featuring members <@326360689453039618> and <@1291640853462253652>, is hosting a live demo showcasing their **autonomous agent** and [tooling](https://github.com/smallcloudai).
  
  - Join the **live demo and conversation** [here](https://discord.com/events/1089876418936180786/1300459081181429810) to dive deeper into their developments!
- **Live Event Announcement**: An event has been announced featuring **Refact.AI** members discussing their latest technology and tools.
  
  - Participants are encouraged to engage with the **live demo and conversation** related to the **autonomous agent**.

 

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