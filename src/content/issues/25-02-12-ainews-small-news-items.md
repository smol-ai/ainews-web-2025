---
id: 117d6739-42ce-4ce5-afeb-d11cee530906
title: small news items
date: '2025-02-13T00:10:12.213344Z'
original_slug: ainews-small-news-items
description: >-
  **OpenAI** revealed plans for **GPT-4.5** and **GPT-5**, with GPT-5
  integrating the advanced **o3+** model and offering unlimited chat access in
  the free tier. Meanwhile, **OpenAI's o3 model** achieved a gold medal at the
  *2024 International Olympiad in Informatics*, outperforming specialized
  pipelines without hand-crafted heuristics, highlighting significant advances
  in large reasoning models.
tags:
  - openai
  - ollama
  - glean
  - harvey
  - fal
  - scaled-cognition
  - mistral
  - perplexity
  - cerebras
  - alibaba
  - groq
  - bytedance
  - figure
  - bouygues-telecom
  - together
  - klarna
  - langchain
  - google-deepmind
  - gpt-4.5
  - gpt-5
  - o3
  - o1-preview
  - deepseek-r1-distilled-qwen-1.5b
  - deepscaler
  - qwen-1.5b
  - qwen-0.5b
  - modernbert-0.3b
  - deepseek-r1
  - o1-ioi
  - webli-100b
  - large-language-models
  - model-performance
  - reinforcement-learning
  - competitive-programming
  - reasoning-models
  - dataset-scaling
  - vision-language-models
  - model-benchmarks
  - industry-adoption
  - partnerships
  - funding
  - customer-support-automation
  - compute-infrastructure
  - multilinguality
  - cultural-diversity
companies:
  - openai
  - ollama
  - glean
  - harvey
  - fal
  - scaled-cognition
  - mistral
  - perplexity
  - cerebras
  - alibaba
  - groq
  - bytedance
  - figure
  - bouygues-telecom
  - together
  - klarna
  - langchain
  - google-deepmind
models:
  - gpt-4.5
  - gpt-5
  - o3
  - o1-preview
  - deepseek-r1-distilled-qwen-1.5b
  - deepscaler
  - qwen-1.5b
  - qwen-0.5b
  - modernbert-0.3b
  - deepseek-r1
  - o1-ioi
  - webli-100b
topics:
  - large-language-models
  - model-performance
  - reinforcement-learning
  - competitive-programming
  - reasoning-models
  - dataset-scaling
  - vision-language-models
  - model-benchmarks
  - industry-adoption
  - partnerships
  - funding
  - customer-support-automation
  - compute-infrastructure
  - multilinguality
  - cultural-diversity
---


<!-- buttondown-editor-mode: plaintext -->**GPT5 is all you need.**

> AI News for 2/11/2025-2/12/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**211** channels, and **5266** messages) for you. Estimated reading time saved (at 200wpm): **497 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

No title story but lots of cool updates:

- OpenAI shared a new [model spec](https://x.com/OpenAI/status/1889781541259321466) and that [gpt4.5 is coming and gpt5 will incorporate o3+](https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg)
- [glean announced agents](https://x.com/glean/status/1889706504812683728)
- funding announcements from [Harvey](https://x.com/winstonweinberg/status/1889713028234416371?s=46), [FAL](https://x.com/glennsolomon/status/1889717350456315960?s=46), and [Scaled Cognition](https://x.com/scaledcognition/status/1889721166421479751?s=46)
- [Jeff Dean and Noam Shazeer on Dwarkesh](https://x.com/swyx/status/1889810524696891903)

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Models & Performance**

- **DeepSeek R1 Distilled Qwen 1.5B surpasses OpenAI's o1-preview on math benchmarks**: [@ollama](https://twitter.com/ollama/status/1889496833875124735) announced the release of **DeepScaleR**, an Ollama model, a fine-tuned version of **Deepseek-R1-Distilled-Qwen-1.5B**, which **outperforms OpenAI’s o1-preview** on popular math evaluations, achieving this with just **1.5B parameters**.  [@jeremyphoward](https://twitter.com/jeremyphoward/status/1889435769959489800) noted that DeepScaleR also **beats Qwen at MMLU Pro**, questioning if decoder models are truly required for such complex domains. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889522980096712945) highlighted that **OpenAI's o3 achieved 99.8th percentile on Codeforces**.
- **ModernBERT 0.3b outperforms Qwen 0.5b at MMLU without task-specific fine-tuning**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1889434481519632505) stated that the encoder-only **ModernBERT 0.3b beats Qwen 0.5b at MMLU** without needing task-specific fine-tuning, suggesting this could start a **new revolution in language models**.
- **Mistral and Perplexity are adopting Cerebras for 10x performance gains**: [@draecomino](https://twitter.com/draecomino/status/1889430107288416340) announced that **Mistral and Perplexity** are moving to **Cerebras**, claiming it makes customer products **10x faster than competitors**. [@draecomino](https://twitter.com/draecomino/status/1889434428306497667) also noted that since his previous post, **two of the largest AI startups funded by Nvidia** are now using Cerebras.
- **OpenAI's o3 model achieves gold medal at IOI 2024**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889522974467957033) and [@_akhaliq](https://twitter.com/_akhaliq/status/1889523662732042610) shared **OpenAI's** paper "Competitive Programming with Large Reasoning Models", highlighting that their **o3 model** achieved a **gold medal at the 2024 International Olympiad in Informatics (IOI)**.  [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889517116816244995) further detailed that **o3 surpassed specialized pipelines like o1-ioi** without hand-crafted inference heuristics and under relaxed constraints.
- **Qwen and Groq partnership**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1889473940894130605) signaled a partnership between **Qwen and Groq** with a simple emoji post.
- **GPT-4.5 and GPT-5 roadmap from OpenAI**: [@sama](https://twitter.com/sama/status/1889755723078443244) shared an **OpenAI roadmap update**, revealing plans to ship **GPT-4.5 (Orion)** as their last non-chain-of-thought model and to release **GPT-5 as a system** integrating technologies like **o3**.  [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889756903187829107) and [@stevenheidel](https://twitter.com/stevenheidel/status/1889757357908836654) summarized these points, noting that **GPT-5** in the free tier of ChatGPT will have **unlimited chat access**. [@nrehiew_](https://twitter.com/nrehiew_/status/1889757485755416782) commented that this approach to **GPT-5 as a system** might **widen the gap between academia and industry** in model evaluation.
- **RLHFers are significantly present in Nigeria and the global south**: [@DanHendrycks](https://twitter.com/DanHendrycks/status/1889483790638317774) pointed out the significant presence of **RLHFers from Nigeria** and potentially other countries in the global south.
- **Bytedance is expected to become notable in AI soon**: [@agihippo](https://twitter.com/agihippo/status/1889583723730829687) predicted that **Bytedance**, currently not prominent in AI, will become notable **very soon**.
- **Apps built with FastHTML and MonsterUI are easy to build and maintain**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1889430719988113911) praised **FastHTML**, **htmx**, and **MonsterUI** for enabling the creation of apps that are quick to write, easy to maintain, and great to use.
- **DeepScaleR, a 1.5B parameter model, surpasses OAI's O1-preview using RL**: [@_philschmid](https://twitter.com/_philschmid/status/1889592742088515630) detailed that **DeepScaleR**, a **1.5B parameter model** fine-tuned with Reinforcement Learning, **outperforms OpenAI's O1-preview** in math benchmarks, highlighting the effectiveness of RL even for smaller models and the use of a simple binary reward function.
- **Only offline RL experts understand the importance of online RL**: [@shaneguML](https://twitter.com/shaneguML/status/1889505192229609864) stated that **only those who have delved into offline RL** truly appreciate the **importance of online RL**.

**Industry & Business**

- **Mistral and Perplexity are adopting Cerebras for 10x performance gains**: [@draecomino](https://twitter.com/draecomino/status/1889430107288416340) announced that **Mistral and Perplexity** are moving to **Cerebras**, claiming it makes customer products **10x faster than competitors**. [@draecomino](https://twitter.com/draecomino/status/1889434428306497667) also noted that since his previous post, **two of the largest AI startups funded by Nvidia** are now using Cerebras.
- **Figure is a highly in-demand company in the secondary market**: [@adcock_brett](https://twitter.com/adcock_brett/status/1889743077323272442) shared that **Figure** was the **9th most in-demand company** in the secondary market last month, noting investor demand is "off the charts".
- **Perplexity is aiming for a TikTok deal**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1889577617885380613) mentioned he'll "still chug Red Bulls to get the **TikTok deal done**".
- **Perplexity is partnering with Bouygues Telecom in France**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1889724444894957788) announced a partnership with **Bouygues Telecom** to distribute **Perplexity** in France, adding to their global partnerships.
- **Perplexity launches a Finance Dashboard**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1889775543635050985) promoted **Perplexity's Finance Dashboard**, offering stocks, earnings, market movements, and summaries in one place.
- **High user adoption of Perplexity in Paris**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1889635609582444880) and [@AravSrinivas](https://twitter.com/AravSrinivas/status/1889562246004494405) described experiencing **high user adoption of Perplexity in Paris**, with people stopping him on the street to express their love for the app and meeting enthusiastic students using Perplexity.
- **Together AI launches Reasoning Clusters for DeepSeek-R1 deployment**: [@togethercompute](https://twitter.com/togethercompute/status/1889743684977168547) announced **Together Reasoning Clusters**, dedicated compute built for large-scale, low-latency reasoning workloads, expanding beyond their Serverless API for deploying reasoning models like **DeepSeek-R1** in production.
- **Klarna's AI assistant scaled customer support with LangGraph and LangSmith**: [@LangChainAI](https://twitter.com/LangChainAI/status/1889728750415479161) and [@hwchase17](https://twitter.com/hwchase17/status/1889758528232898844) highlighted how **Klarna** used **LangGraph and LangSmith** to scale customer support for **85 million active users**, reducing resolution times by **80%** and automating **70%** of tasks.

**Research & Papers**

- **OpenAI releases "Competitive Programming with Large Reasoning Models" paper**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889522974467957033) and [@_akhaliq](https://twitter.com/_akhaliq/status/1889523662732042610) shared **OpenAI's** paper "Competitive Programming with Large Reasoning Models", highlighting that their **o3 model** achieved a **gold medal at the 2024 International Olympiad in Informatics (IOI)**.  [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889517116816244995) further detailed that **o3 surpassed specialized pipelines like o1-ioi** without hand-crafted inference heuristics and under relaxed constraints.
- **Google DeepMind publishes "Scaling Pre-training to One Hundred Billion Data for Vision Language Models"**: [@_akhaliq](https://twitter.com/_akhaliq/status/1889526316673732753) and [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889520451501900000) shared **Google DeepMind's** paper "Scaling Pre-training to One Hundred Billion Data for Vision Language Models", introducing **WebLI-100B**, a dataset with **100 billion image-text pairs**, showing benefits beyond traditional benchmarks, especially in **cultural diversity and multilinguality**. [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889521157482959067) also highlighted the dataset and findings.
- **New paper "InSTA" for internet-scale web agent training**: [@rsalakhu](https://twitter.com/rsalakhu/status/1889492471630946662) announced a new paper on **InSTA**, a pipeline for **internet-scale training of web agents** across **150k diverse websites** without human annotations, achieving competitive performance with human annotators in tasks like harmful content detection and task completion, using **Llama 3.1 70B agents**.
- **Scale AI releases research on "Jailbreak to Jailbreak" for LLMs**: [@goodside](https://twitter.com/goodside/status/1889492446750364103) shared new research from **Scale AI** on "Jailbreak to Jailbreak", using **jailbreaking safety-trained LLMs** to develop jailbreaks for other LLMs.
- **Paper on MARIA model for masked token infilling**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889542518465077557) highlighted a paper on **MARIA**, a hybrid autoregressive and masked language model for **infilling masked tokens**, outperforming discrete diffusion models and offering faster inference with KV caching.
- **Microsoft Research presents "NatureLM" for scientific discovery**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889518044273385672) shared **Microsoft Research's** paper on **NatureLM**, a sequence-based science foundation model for **scientific discovery**, capable of generating and optimizing molecules, proteins, RNA, and materials using text instructions.
- **Meta AI presents "Pippo" for high-resolution multi-view humans from a single image**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889515688647373113) shared **Meta AI's** paper on **Pippo**, a model generating **1K resolution, multi-view, studio-quality images of humans** from a single photo in one forward pass.
- **Paper investigates emergent thinking in LLMs using RLSP technique**: [@omarsar0](https://twitter.com/omarsar0/status/1889697727703134544) discussed a paper on "On the Emergence of Thinking in LLMs", exploring **reasoning in LLMs** using a post-training technique called **RLSP**, showing emergent behaviors like backtracking and exploration.
- **Paper on Large Memory Models (LM2) for long-context reasoning**: [@omarsar0](https://twitter.com/omarsar0/status/1889681118913577345) summarized a paper on **Large Memory Models (LM2)**, a Transformer-based architecture with a dedicated memory module to enhance **long-context reasoning**, outperforming baselines on memory-intensive benchmarks.
- **TAID paper accepted at ICLR2025 on knowledge distillation**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1889708905280028809) announced that their paper “TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models” has been accepted as a **Spotlight Paper at ICLR2025**, introducing a new knowledge distillation approach.

**Tools & Applications**

- **Ollama releases DeepScaleR model**: [@ollama](https://twitter.com/ollama/status/1889496833875124735) announced the release of **DeepScaleR**, an Ollama model, a fine-tuned version of **Deepseek-R1-Distilled-Qwen-1.5B**, which **outperforms OpenAI’s o1-preview** on popular math evaluations, achieving this with just **1.5B parameters**.
- **LangChain releases LangGraph Supervisor for multi-agent systems**: [@LangChainAI](https://twitter.com/LangChainAI/status/1889717269510394365) introduced **LangGraph Supervisor**, a lightweight library for building **hierarchical multi-agent systems** with LangGraph, featuring a supervisor agent to orchestrate specialized agents and tool-based handoffs.
- **Perplexity launches a Finance Dashboard**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1889775543635050985) promoted **Perplexity's Finance Dashboard**, offering stocks, earnings, market movements, and summaries in one place.
- **AI financial agent with stock price updates**: [@virattt](https://twitter.com/virattt/status/1889458443515265066) announced updates to their **AI financial agent**, now showing stock prices, market cap, volume, and historical prices, with open-source code and no signup required.
- **SWE Arena for model preference voting in coding tasks**: [@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1889584039629078963) highlighted **SWE Arena**, a platform where users can **vote for their preferred model** when coding with frontier models like **o3-mini**.
- **Aomniapp agent orchestration system beta release**: [@dzhng](https://twitter.com/dzhng/status/1889547813559951533) announced the beta availability of **Aomniapp**, an **agent orchestration system** allowing users to spawn hundreds of agents with a prompt.
- **Google DeepMind Gemini API key setup is quick and easy**: [@_philschmid](https://twitter.com/_philschmid/status/1889689838464516228) detailed how to create a **Google DeepMind Gemini API key** in under 30 seconds, requiring only a Google account and no credit card or Google Cloud Account.
- **DeepSeek R1 generates Rubik's cube visualizer and solver**: [@_akhaliq](https://twitter.com/_akhaliq/status/1889736413429559444) showcased **DeepSeek R1** generating a **Rubik's cube visualizer and solver** in a single HTML file using Three.js, with interactive controls and animation.
- **RepoChat allows chatting with GitHub repos**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1889741525808193635) announced the **RepoChat Blog & Dataset Release**, highlighting their tool that allows users to **chat with their GitHub repos**, having collected over 11K conversations.
- **Text2web Arena for text-to-web applications**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1889496847708045496) promoted **Text2web Arena**, a platform to try out text-to-web applications, showcasing **Claude 3.5 Sonnet** generating a 3D scene with Three.js.

**Development & Coding**

- **Software libraries in 2025 should include context.txt for LLM codegen**: [@vikhyatk](https://twitter.com/vikhyatk/status/1889540437557518843) suggested that publishing a **software library in 2025** requires including a **context.txt** file for users to paste into LLMs for correct code generation.
- **Manual coding in 2025 compared to assembly for web apps in 2024**: [@vikhyatk](https://twitter.com/vikhyatk/status/1889597476895662336) commented that **writing code manually in 2025** will be like **writing assembly to build a web app in 2024**, implying AI-driven code generation will become dominant.
- **Preference for C++ over scripting for complex tasks**: [@MParakhin](https://twitter.com/MParakhin/status/1889428158421819825) expressed a preference for **C++** over scripting for complex tasks due to its speed and debuggability, using `system()` for scripting needs.
- **DeepSeek CPU/GPU hybrid inference for MLA operators**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1889531203742466250) highlighted **DeepSeek's CPU/GPU hybrid inference** approach for their computationally intensive MLA operators, offloading heavy computations to the GPU for performance boost.
- **Tooling for curating video datasets for fine-tuning released**: [@RisingSayak](https://twitter.com/RisingSayak/status/1889632398465228998) announced the release of **tooling for curating small and high-quality video datasets** for fine-tuning, inspired by SVD & LTX-Video, addressing the lack of good data curation pipelines in video fine-tuning.

**Humor & Meta**

- **Meme summarizes OpenAI's o3 paper**: [@polynoamial](https://twitter.com/polynoamial/status/1889541408065028421) shared a **meme** that nicely summarizes the "Competitive Programming with Large Reasoning Models" paper.
- **State of AI meme**: [@giffmana](https://twitter.com/giffmana/status/1889424405350002991) posted a meme depicting "the state of AI rn, more or less."
- **Humorous historical question about Stalingrad**: [@kipperrii](https://twitter.com/kipperrii/status/1889440548848804252) jokingly asked for a history explanation of **Stalingrad**, pointing out Wikipedia's seemingly contradictory death toll figures.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Revolutionary Latent Space Reasoning in LLMs**

- **[A new paper demonstrates that LLMs could "think" in latent space, effectively decoupling internal reasoning from visible context tokens. This breakthrough suggests that even smaller models can achieve remarkable performance without relying on extensive context windows.](https://huggingface.co/papers/2502.05171)** ([Score: 1218, Comments: 261](https://reddit.com/r/LocalLLaMA/comments/1inch7r/a_new_paper_demonstrates_that_llms_could_think_in/)): A recent paper reveals that **Large Language Models (LLMs)** can perform reasoning in latent space, allowing them to separate internal reasoning from visible context tokens. This advancement implies that smaller models might deliver impressive results without depending on large context windows.
  - Discussions highlight the potential of reasoning in **latent space** to improve model performance, with comparisons to existing methods like **Chain-of-Thought (CoT)** and references to **Meta's COCONUT approach**. Concerns are raised about safety and transparency, as latent reasoning may lead to models "thinking" in ways not easily represented in words, complicating alignment and explainability efforts.
  - The paper's testing on **AMD mi250x** and the use of **ROCM software stack** are notable, challenging the dominance of **Nvidia** in AI research. There is interest in whether this approach can be scaled effectively, with skepticism about the authors' previous works and the challenges of implementing such methods in practice.
  - The conversation touches on broader themes of AI reasoning and consciousness, with references to **Daniel Kahneman's** "Thinking, Fast and Slow" and the distinction between intuitive and logical reasoning systems. The potential for models to "think without thinking" or "think without language" is explored, with links to **Hugging Face** resources for further exploration of the paper's concepts.


**Theme 2. AMD's Strategic Moves in AI Hardware Competition**

- **[AMD reportedly working on gaming Radeon RX 9070 XT GPU with 32GB memory](https://videocardz.com/newz/amd-reportedly-working-on-gaming-radeon-rx-9000-gpu-with-32gb-memory)** ([Score: 383, Comments: 96](https://reddit.com/r/LocalLLaMA/comments/1inoui5/amd_reportedly_working_on_gaming_radeon_rx_9070/)): **AMD** is reportedly developing the **Radeon RX 9070 XT GPU** aimed at gaming, featuring **32GB of memory**. This development suggests potential implications for AI applications, given the substantial memory capacity, which could enhance performance in AI-driven tasks.
  - **ROCm vs CUDA**: There is a strong sentiment in favor of **ROCm** as an open-source alternative to **CUDA**, with many users arguing that high VRAM GPUs like the **RX 9070 XT** could drive community improvements in ROCm to better compete with **NVIDIA's** ecosystem. Some users express frustration with CUDA's dominance, comparing it to **OpenAI**'s influence in LLMs.
  - **Pricing and Performance Comparisons**: Discussions highlight the potential competitive pricing of the **RX 9070 XT**, rumored to be under **$1000**, as a significant factor against NVIDIA's offerings, such as the **RTX 5090**. Users are debating the trade-offs between VRAM capacity and memory bandwidth, noting that **7900 XTX** provides a cost-effective alternative with reasonable performance.
  - **Community and Source Reliability**: There is skepticism about the reliability of GPU leaks, as evidenced by a humorous critique of a source with a photoshopped profile picture. Despite this, some community members vouch for the consistency of such sources, highlighting the speculative nature of GPU news.


**Theme 3. Project Digits: Nvidia’s Next Big Step in AI Workstations**

- **[Some details on Project Digits from PNY presentation](https://www.reddit.com/gallery/1inos01)** ([Score: 128, Comments: 86](https://reddit.com/r/LocalLLaMA/comments/1inos01/some_details_on_project_digits_from_pny/)): Nvidia's **Project Digits** was presented by PNY's DGX EMEA lead, highlighting features such as **DDR5x memory** with 128GB initially, dual-port **QSFP networking** with a Mellanox chip, and a new ARM processor. The workstation, priced around **$3,000**, is noted for its software stack and Ubuntu-based OS, targeting universities and researchers, and is significantly more powerful than Jetson products, although not a replacement for multi-GPU workstations.
  - **Memory Bandwidth Concerns**: Several commenters expressed frustration over Nvidia's lack of disclosure regarding the **memory bandwidth** of Project Digits, speculating it to be around **270 GB/s**. The absence of this information is seen as a potential red flag, with some suggesting it's a strategy to maintain interest until more details are revealed at **GTC**.
  - **Target Audience and Purpose**: Project Digits is positioned as a compact, portable workstation for **researchers and universities**, meant for developing and experimenting with new AI architectures rather than replacing multi-GPU workstations. It's described as a gateway to the Nvidia ecosystem, enabling researchers to easily transition to more powerful **DGX machines** for larger projects.
  - **Strategic Positioning and Market Impact**: The product is seen as a strategic move by Nvidia to capture the next generation of AI/ML engineers, despite concerns about its **niche market** status and potential quick obsolescence. The discussion highlighted Nvidia's focus on maintaining its market dominance through software support and ecosystem integration, while some users expressed skepticism about Nvidia's long-term strategy and its implications for consumer-grade products.


**Theme 4. Phi-4's Unconventional Approach to AI Creativity**

- **Phi-4, but pruned and unsafe** ([Score: 112, Comments: 21](https://reddit.com/r/LocalLLaMA/comments/1inn034/phi4_but_pruned_and_unsafe/)): **Phi-Lthy4** is a pruned version of **Phi-4** designed to enhance roleplay capabilities by removing unnecessary mathematical layers, resulting in a model with **11.9B parameters**. The model, which underwent a two-week fine-tuning process using **1B tokens**, excels in creative writing and roleplay, proving to be a unique assistant with low refusal rates and strong adherence to character cards. Despite its unconventional approach, it remains surprisingly effective, as detailed on [Hugging Face](https://huggingface.co/SicariusSicariiStuff/Phi-lthy4).
  - **Model Size and Performance**: **Phi-Lthy4** is a pruned version of Phi-4 with **11.9B parameters** and excels in creative writing and roleplay. There is a discussion on the model's size in different quantizations, with the **IQ4_XS quant** version being **6.5GB**, suggesting it could run with **8GB** of memory.
  - **Model Merging and Variants**: **Environmental-Metal9** expresses interest in merging Phi with **Mistral** due to its prose quality. **Sicarius_The_First** shares a related project, **Redemption Wind 24B**, on [Hugging Face](https://huggingface.co/SicariusSicariiStuff/Redemption_Wind_24B), highlighting the potential of combining different model strengths.
  - **Benchmarking and Writing Style**: The **Phi series** is not typically used for benchmarking compared to **Qwen**, which is often the base model for fine-tuning in recent papers. However, Phi is noted for its unique writing style, described as "clinical but not cringy-sloppy," which some users appreciate.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. OpenAI's New Models: GPT-4.5 'Orion' and Chain-of-Thought Integration**

- **[OpenAI Roadmap Update for GPT-4.5 & GPT-5](https://i.redd.it/j6j5diamdrie1.jpeg)** ([Score: 503, Comments: 106](https://reddit.com/r/OpenAI/comments/1inz75h/openai_roadmap_update_for_gpt45_gpt5/)): **OpenAI's roadmap** update, shared by **Sam Altman** on Twitter, outlines plans for **GPT-4.5**, internally named **Orion**, and **GPT-5**. The update highlights efforts to simplify product offerings, enhance user experience, and unify model series, with GPT-5 integration into both **ChatGPT** and the API, offering tiered access levels, including a higher intelligence setting for Pro subscribers.
  - Users express concerns about **OpenAI's tiered intelligence model** potentially complicating the system and reducing user choice, with some preferring the ability to manually select models for specific tasks, such as using **o3-mini** for coding or health-related questions. Others argue that automating model selection could improve user experience by simplifying decisions for non-experts.
  - The discussion includes skepticism about **OpenAI's cost-saving strategies**, such as reducing running costs by automating model selection, which could limit transparency and user control. Some users appreciate the idea of models like **GPT-4.5** and **GPT-5** autonomously deciding when to employ 'chain-of-thought' reasoning, while others worry it might lead to a "black box" system.
  - There is curiosity about the future of **external chatbots** running on older models like **GPT-3 or GPT-3.5**, with some users concerned about their potential obsolescence. However, there is no clear indication from OpenAI that these APIs will be phased out soon, though it is speculated that it may not be economically viable to support them indefinitely.


**Theme 2. DeepSearch Goes Mainstream: Plus and Free User Access**

- **[DeepSearch soon to be available for Plus and Free users](https://i.redd.it/9zwkrb49uqie1.png)** ([Score: 555, Comments: 97](https://reddit.com/r/OpenAI/comments/1inwhg1/deepsearch_soon_to_be_available_for_plus_and_free/)): **DeepSearch**, a feature mentioned by **Sam Altman** in a Twitter conversation, will soon be available to **ChatGPT Plus users** with 10 uses per month and **free users** with 2 uses. A user highlighted the feature's substantial value, estimating it at approximately **$1,000 per month**, and noted its significant impact on cognitive engagement.
  - Several commenters criticize the claim that **DeepSearch** is worth **$1,000 per month**, arguing that it is not realistic and may be a tactic called "anchoring" to make future pricing appear lower. **Fumi2014** mentions that the feature is not comprehensive enough as a research tool because it relies on publicly accessible web data, excluding many academic resources.
  - **EastHillWill** and others discuss the potential cost of **DeepSearch**, with estimates around **$0.50 per use**. There is a suggestion to offer more flexible pricing options, like **20 free uses** followed by a charge for additional uses, to provide better value.
  - Concerns are raised about the availability and pricing structure of **DeepSearch** for different user tiers, with some users expressing frustration over the exclusion of **ChatGPT Team accounts** and the potential for circumventing usage limits by creating multiple accounts, although this would require multiple phone numbers.


**Theme 3. Grok 3 Performance Leak and xAI Resignation Fallout**

- **[xAI Resignation](https://i.redd.it/nkcfuep8enie1.png)** ([Score: 721, Comments: 174](https://reddit.com/r/OpenAI/comments/1ink8o2/xai_resignation/)): Benjamin De Kraker announced his resignation from **xAI**, citing pressure to delete a statement about **Grok 3** as a key reason. He criticized the company for labeling his opinion as "confidential information" and expressed disappointment in xAI's stance on free speech, while reflecting on his future plans.
  - Many commenters agree that **Benjamin De Kraker's public disclosure** regarding **Grok 3's performance** was inappropriate, as it involved ranking the model against competitors using internal information. This is seen as a breach of confidentiality, and several users argue that such actions can lead to justified termination due to potential financial and reputational impacts.
  - The discussion emphasizes that **company policies typically prohibit unauthorized discussions** of unreleased products, especially when they involve comparative assessments. Commenters highlight that even if some information is public, employees are generally expected to adhere to strict protocols and not publicly speculate or share internal insights without explicit permission.
  - There is a consensus that **De Kraker's framing of the issue as a free speech violation** is misplaced. The comments suggest that his actions were more about breaching company confidentiality rather than an infringement on personal expression, with some users noting that other companies would have handled the situation more severely.


**Theme 4. OpenAI Multimodal Models: o1, o3-mini, and o3-mini high**

- **OpenAI silently rolls out: o1, o3-mini, and o3-mini high is now multimodal.** ([Score: 393, Comments: 101](https://reddit.com/r/OpenAI/comments/1inoi6b/openai_silently_rolls_out_o1_o3mini_and_o3mini/)): **OpenAI** has quietly introduced multimodal capabilities to their models **o1**, **o3-mini**, and **o3-mini high**, enabling them to process both images and files. The update has been met with surprise and enthusiasm for its expanded functionality.
  - Users report varied experiences with multimodal capabilities across different platforms, with some able to upload images and files on **iOS** and **web versions**, while others, particularly on **desktop** and in certain regions like **Poland** and **Asia**, have not yet received updates. **PDF uploads** on **o3** are highlighted as a significant feature, though some express a desire for API support for PDFs.
  - There is confusion and discussion around which models support these capabilities, with users noting that **o1** supports file uploads, but **o3-mini** and **o3-mini high** do not yet show this feature on desktop versions. Some users have been using **o1 pro** for image uploads for a while, as demonstrated in a **YouTube demo**.
  - The rollout of these features appears inconsistent, with users in various regions and platforms reporting different levels of access, sparking discussions on the availability and potential of using models beyond **4o** for projects.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1: OpenAI Unveils GPT-5 and Opens Floodgates to o1 and o3**

- **OpenAI Bets Big on GPT-5: No More Model Messing Around!** OpenAI announced the upcoming release of **GPT-4.5** and **GPT-5**, aiming to unify their product offerings and make AI "just work" for users, as per [Sam Altman's tweet](https://x.com/sama/status/1889755723078443244). GPT-5 will incorporate diverse technologies and be available to free-tier users with varying intelligence levels.

- **OpenRouter Throws OpenAI's o1 and o3 to the Masses!** **OpenAI's o1 and o3** reasoning models are now available to all [OpenRouter](https://openrouter.ai/) users without requiring BYOK, enhancing rate limits for previous key users, as announced [here](https://x.com/OpenRouterAI/status/1889708759355691327). These models now support web search, broadening their utility and streamlining the user experience.

- **Community Cheers (and Jeers) at OpenAI's Shift in Strategy** The community reacted to OpenAI's roadmap update with a mix of excitement and skepticism. While some are thrilled about the simplified offerings, others question the move away from non-reasoning models. Discussions highlight both anticipation and concerns over AI development directions.

**Theme 2: GRPO Powers Up AI Models, Sending Performance Soaring**

- **GRPO Integration Woes: Model Tweaking Ain't for the Faint of Heart!** AI enthusiasts grappled with challenges in integrating **GRPO** with models like **Mistral** and **Llama**, sharing insights and pointing out quirks with special tokens like *<thinking>*. Despite hurdles, the community shared resources like a [helpful notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=vITh0KVJ10qX) to iron out implementation kinks.

- **Tulu Pipeline Turbocharged: GRPO Gives 4x Performance Boost!** Switching from **PPO** to **GRPO** in the **Tulu pipeline** resulted in a [4x increase](https://x.com/vwxyzjn/status/1889730488199209393) in performance, showing significant improvements on tasks like **MATH** and **GSM8K**. This marks a promising direction for RL strategies in AI training.

- **Fine-Tuners Rejoice: GRPO Makes Models Shine Brighter** Users shared success stories of fine-tuning models using GRPO, emphasizing the importance of dataset preparation and proper training templates. Tools and datasets like [OpenR1-Math-Raw](https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw) emerged as valuable resources for enhancing model performance.

**Theme 3: Thomson Reuters Clobbers AI Copycats in Court**

- **Copyright Crusade: Thomson Reuters Wins First AI Court Battle!** In a landmark decision, [Thomson Reuters secured a copyright victory](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/) against Ross Intelligence for reproducing materials from Westlaw. Judge Stephanos Bibas declared, *"None of Ross’s possible defenses holds water,"* emphasizing the seriousness of the infringement.

- **AI Learns a Legal Lesson: Respect IP or Face the Music** This ruling sets a critical precedent for AI copyright in the U.S., highlighting that AI companies must respect intellectual property rights when developing technologies. The case sends a strong message about the legal responsibility in AI development.

- **Lawyers Celebrate: AI Becomes the Gift That Keeps on Giving** The legal community buzzed with the potential for new cases following this decision. Companies are urged to review their AI training data to avoid similar lawsuits, while IP lawyers see a surge in future work opportunities.

**Theme 4: DeepScaleR Rockets RL Back into the Spotlight**

- **RL Revival: DeepScaleR's Tiny Titan Takes on the Giants!** The [DeepScaleR preview](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scali) showcased a **1.5B model** that significantly scales up RL, igniting excitement in the AI community. Enthusiasts proclaimed, *"RL is back baby!"* as the model surpassed expectations.

- **Small Model, Big Impact: DeepScaleR Defies Scaling Norms** The model's advancements suggest that even smaller models can achieve impressive results with proper RL scaling techniques. This challenges the notion that only massive models can lead the AI pack, opening doors for more efficient AI development.

- **Researchers Rally: RL Techniques Get a Second Wind** The success of DeepScaleR encourages researchers to revisit reinforcement learning methods. This revival could lead to new innovations in AI training and optimization, as the community explores scalable solutions.

**Theme 5: AI Models Get Curious with Automated Capability Discovery**

- **Models Play Scientist: ACD Lets AI Explore Itself!** A new framework called [Automated Capability Discovery (ACD)](https://arxiv.org/abs/2502.07577) allows AI models to self-explore their capabilities and weaknesses. By acting as their own 'scientists', models like GPT and Claude can propose tasks to evaluate themselves, as highlighted in [Jeff Clune's tweet](https://x.com/jeffclune/status/1889568685632667672).

- **Foundation Models Go Self-Aware: What Could Possibly Go Wrong?** ACD empowers models to identify unexpected behaviors without exhaustive manual testing, enhancing evaluation accuracy with less human effort. While exciting, this raises questions about control and safety in AI systems as models take on self-directed exploration.

- **Less Human, More Machine: ACD Redefines Model Evaluation** With ACD, developers can potentially speed up development cycles and uncover hidden model potentials. The community is both intrigued and cautious about the implications, balancing innovation with the need for responsible AI practices.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GRPO Implementation Challenges**: Members discussed integrating **GRPO** with models like **Mistral** and **Llama**, noting challenges with models not producing expected tokens even with correct implementation, hinting at integration difficulties with **special tokens** like *<thinking>*.
   - A [tweet](https://x.com/UnslothAI/status/1889726411478278183) and associated [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=vITh0KVJ10qX) were shared showing *Llama 3.1 (8B)* being transformed to chain-of-thought with **GRPO**, highlighting the need for appropriate training templates.
- **Dataset Cleaning Demands Deeper Analysis**: Discussions emphasized that simply removing missing values from datasets could diminish the data's relevance; thorough analysis and understanding are vital for effective data preparation before training, ensuring the dataset remains relevant and robust for **LLM** training.
   - For additional information, [Datasets 101 | Unsloth Documentation](https://docs.unsloth.ai/basics/datasets-101#getting-started) was cited as a helpful resource for best practices.
- **Liger vs Apple Kernels Show Performance Variance**: Comparisons between the **Liger kernel** and **Apple's cross-entropy** implementation revealed that while **Liger** has speed advantages, **Apple's kernel** performs certain operations more efficiently due to its complete implementation, impacting overall performance.
   - Specifically, discussions referenced the implementations in [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py) and [Apple's ml-cross-entropy](https://github.com/apple/ml-cross-entropy/blob/main/cut_cross_entropy/cce_lse_forward.py#L79), with nuances due to differences in how they process logits.
- **GRPO Fine-Tuning Struggles on A100**: A user encountered out-of-memory (**OOM**) errors while fine-tuning the **Qwen 32B** model on an **A100**, reducing context length from 128k to 16k, raising questions about memory allocation feasibility.
   - The user sought advice on whether to use **wandb** or **Unsloth**'s built-in features for experiment tracking during the **GRPO** process, pointing out that they were primarily interested in loss tracking and optimization.
- **Reward Function Generosity Spurs Repetitive Outputs**: Community members found that reward functions, while effective, are too lenient on certain phrases, leading to undesirable repetitive outputs such as *"Hmm, let's see..."*, highlighting a need for more sophisticated penalties.
   - To address this, it was suggested to explore a sliding window of previous messages to improve self-supervision, rather than treating each generation independently in order to improve the diversity of responses.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DIY Voice Chatbots Arise**: Users explored DIY voice chatbots with Raspberry Pi and ESP32, recommending the **Eilik companion robot** and custom 3D prints for device styling.
   - This showcases the fusion of creativity and functionality in enhancing personal tech.
- **Home Assistant Talks Back**: Members discussed the **Home Assistant Voice**, enabling customized voice assistants using **OpenAI APIs** for web search and smart home controls.
   - This setup requires running a Home Assistant server and supports multilingual configurations, making it accessible for diverse user bases.
- **Moxie's Fate Uncertain**: Concerns were raised about **Moxie**, a children's robot companion facing issues that threaten its future, though its **emotional intelligence** is still noted.
   - Participants speculated on potential successors and discussed its design focused on child interaction; see [YouTube video on Moxie](https://www.youtube.com/watch?v=7dd_r-aqecw).
- **Iterative Prompting Delivers**: A member shared that **iterative prompting** significantly improves results by starting with a baseline and continually refining the prompt.
   - The community emphasized the need for *clear and specific instructions*, acknowledging that LLMs cannot infer intent without explicit guidance.
- **Function Calling causes headaches**: A member described challenges with **function calling** in their system prompt, noting failures or unnecessary triggers based on client interactions.
   - They also mentioned lagging performance even with specific instructions to avoid function calls on ambiguous responses.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium Extension Lags Behind Windsurf**: Members voiced concerns that the **Codeium extension** is falling behind due to increased focus on **Windsurf** and **enterprise offerings**.
   - One member pointed out that the extension remains available through the enterprise option, highlighting the dual focus, while others evaluate switching to **Cursor**.
- **Windsurf Plagued by Errors and Outages**: Users reported ongoing issues with **Windsurf**, including repeated internal errors when using **Cascade** and problems with the **Gemini model**.
   - Many expressed frustration over recent performance drops, particularly the inability to edit files reliably, detailed in [Codeium's status page](https://status.codeium.com).
- **Claude 3.5 Sonnet Tops Windsurf Model Rankings**: An unofficial ranking placed **Claude 3.5 Sonnet** as the top performer in **Windsurf** due to its context handling and tool calling capabilities.
   - **Gemini 2.0 Flash** and **O3-Mini** were praised for speed and pricing, while **GPT-4o** received criticism for poor performance.
- **Users Urge Vigilance with AI-Generated Outputs**: Several users emphasized the importance of user vigilance when working with AI, noting that blindly trusting AI could lead to costly mistakes.
   - The conversation highlighted a need for clearer risk assessments and user education, and cited issues with Windsurf autocomplete, [request canceled](https://codeium.canny.io/feature-requests/p/windsurfs-autocomplete-now-working-around-08-35-41202-utc).
- **Document Sources via llms.txt Format Requested**: Users discussed the potential for adding custom document sources in **Windsurf**, referencing a standardized approach via the **llms.txt** format for indexing documentation.
   - The community hopes for improvements in this area to enhance functionality and ease of access, linking to a [llms.txt directory](https://directory.llmstxt.cloud).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar Built on R1 but Edges out DeepSeek**: Users debated **DeepSeek R1** versus **Sonar Reasoning Pro**, concluding that **Sonar** is built on **R1** and is optimized for web search responses, possibly replacing DeepSeek R1 in the Perplexity app.
   - A [tweet from Perplexity](https://x.com/perplexity_ai/status/1889392617479082323?s=61) notes that **Sonar**, built on **Llama 3.3 70b**, outperforms **GPT-4o-mini** and **Claude 3.5 Haiku** while matching top models.
- **Perplexity API Plagued by 500 Errors**: Multiple users reported experiencing **500 internal server errors** while trying to access the Perplexity API, sparking worries about its reliability and production readiness.
   - Despite the [status page](https://status.perplexity.com/) showing operational status, users expressed frustration, reporting consistent **500 errors** on nearly every API call.
- **Sonar Gets Real-Time Internet Browsing**: Perplexity can perform searches based on current links, giving it **real-time internet browsing capabilities**.
   - This allows for flexibility in browsing and access to the most up-to-date information which is especially useful when you need market [summaries, daily highlights, earnings snapshots](https://x.com/PPLXfinance/status/1889742180421337120?s=61).
- **OpenAI Rebrands, Other News**: Recent happenings include the **rebranding of OpenAI**, news on an **Apple prototype for a tabletop robot** and the discovery of the **largest structure in the universe**.
   - View the [YouTube video](https://www.youtube.com/embed/9SUxli8UDA0) for detailed insights.
- **401 Authorization Snafu Addressed**: A user initially encountered a **401 Authorization Required** error while trying to access the API, but resolved it after troubleshooting.
   - After removing the `<>` brackets around their token as suggested, the user reported that the API started working.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Deepseek R1 Sparks Code Curiosity**: Community members explored the **Deepseek R1 distill model** for math and reasoning, with initial suggestions to test its coding capabilities despite not being its primary function.
   - The discussion highlighted the model's potential to handle complex problems across various applications.
- **LM Studio Lacks Audio Acumen**: Users reported that **LM Studio** does not support audio models like **Qwen2-Audio-7B-GGUF**, leading to discussions on alternative methods for utilizing audio models.
   - External tools and platforms were suggested as potential solutions for those seeking to work with audio models, however no specific suggestions were provided.
- **Markdown Mishaps Muddle Messages**: A bug was reported where markdown input is rendered as formatted text rather than displayed as raw text in **LM Studio**, disrupting the chat interface.
   - The issue has been documented in the [bug tracker](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/430), noting the unexpected behavior and requesting a fix.
- **5090 Reliability Rumors Raise Red Flags**: Concerns amplified regarding the reliability of the **5090** GPU, referencing reports of malfunctioning cards that prompted cautious behavior, based on [anecdotal reports](https://www.youtube.com/watch?v=L1NPFFRTzLo).
   - As a precautionary measure, users suggested undervolting the **5090** to mitigate potential issues.
- **Multi-GPU Builds Bandwidth Bottlenecks**: Experiences were shared about building a server with multiple GPUs, noting specific board configurations to optimize performance in a multi-GPU AI setup, despite bandwidth limitations.
   - Discussion included scenarios where x1 links were utilized due to board constraints, challenging typical expectations of GPU performance with limited PCI-E lanes.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Reuters wins AI Copyright Case**: Thomson Reuters has [won](https://storage.courtlistener.com/recap/gov.uscourts.ded.72109/gov.uscourts.ded.72109.770.0.pdf) a major AI copyright case against Ross Intelligence for reproducing materials from Westlaw, with Judge Stephanos Bibas dismissing all of Ross's defenses.
   - This is a *landmark case* setting a precedent for AI copyright in the U.S.
- **Current AI raises impressive amount**: [Current AI](https://www.currentai.org/) is beginning its work in public interest AI with a pledge of **$400 million**, aiming to reach $2.5 billion over five years, with involvement from locations like Lagos to Lima.
   - The initiative seeks to steer AI development towards community opportunity and security.
- **OpenAI plots GPT 4.5 and 5**: **OpenAI** is planning to release **GPT-4.5**, which will be the last non-chain-of-thought model, followed by **GPT-5** which intends to unify all product offerings, and unlimited free tier access.
   - Paid subscribers will gain enhanced capabilities, including voice and deep research features.
- **GRPO training boosts Performance 4x**: Switching from **PPO** to **GRPO** in the **Tulu pipeline** resulted in a **4x increase** in performance, showing considerable improvements in challenges like **MATH** and **GSM8K**.
   - The latest **GRPO-trained Tulu model** indicates a new direction for RL strategies.
- **xAI Employee forced to Resign Over Grok 3**: An employee resigned from xAI after being compelled to delete a tweet acknowledging **Grok 3**'s existence, classified as confidential by the company, who stated he was disappointed that such an obvious opinion could threaten his job.
   - Members speculated if the employee's remarks on unreleased product performance may have influenced the push for his resignation as some felt xAI's stance contradicts its free speech advocacy.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepfrying strikes 72B Model Training**: A user reported experiencing **wild and increasing loss** in a **72B model** compared to smaller ones, suspecting that high learning rates may not be the only issue, as potentially exacerbated by **deepfrying**.
   - The conversation defined **deepfrying** as a state where a model experiences progressively increasing variance, leading to elevated loss spikes, which can be further influenced by short sequence lengths.
- **Magic Extends Context to 100M Tokens**: Recent updates from Magic introduced **Long-Term Memory models** that can handle contexts up to **100M tokens**, enhancing reasoning capabilities beyond traditional training methods, see [Magic's blog](https://magic.dev/blog/100m-token-context-windows).
   - This advancement opens up significant opportunities in software development by integrating extensive codebases and documentation into the context for model training.
- **Doubts raised on LM2 Memory Slots**: Concerns emerged regarding the transparency of memory slot implementation in the **LM2 model**, see [the LM2 paper](https://arxiv.org/abs/2502.06049), where the selection and updating mechanisms of memory slots in their architecture were not clearly described.
   - Participants voiced skepticism about the effectiveness and parallelizability of the design, suggesting it might be oversimplified in the paper.
- **Automated Capability Discovery Self-Explores Models**: A new framework called **Automated Capability Discovery (ACD)** aims to self-explore model capabilities in a systematic way, identifying unexpected abilities and weaknesses in foundation models, according to [Jeff Clune's Tweet](https://x.com/jeffclune/status/1889568685632667672).
   - ACD operates by designating one foundation model as a 'scientist' to propose tasks for other models, enhancing evaluation accuracy with less human effort.
- **Exploring Fine-tuning with Mnemonic Patterns**: A member inquired if ongoing work relates to fine-tuning methods involving mnemonic strings, specifically how a model could 'recognize' patterns such as those spelling out 'HELLO'.
   - They mentioned having a 'testable hypothesis in that regard', signaling a potential for further experimental exploration, and offering possibilities for collaboration.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Deepseek R1 Pricing Perplexes**: **Cursor** updated [documentation](https://docs.cursor.com/settings/models) specifying usage-based pricing and model availability, causing confusion around models like **deepseek R1** and **O3-mini** premium status.
   - The documentation specifies [usage-based pricing](https://docs.cursor.com/account/usage#usage-based-pricing) for specific models, leaving users to compare costs and benefits of various options like **Perplexity API** and **Claude**.
- **MCP Server Integrations Spark Headaches**: Users encountered issues with **MCP server integrations**, specifically the **Perplexity API**, resulting in errors during usage.
   - Some users resolved problems by hardcoding API keys and removing conflicting packages, but inconsistencies in performance remain.
- **O3-Mini's Outputs Fluctuate**: The inconsistent performance of **O3-mini** raised concerns, with users experiencing both successful and hallucinated outputs depending on the context.
   - While **O3-mini** occasionally provides impressive improvements, ongoing inconsistencies remain a notable point of frustration, according to user feedback.
- **Claude Model Releases Spark Anticipation**: Enthusiasm builds for upcoming **Anthropic** model releases, with users sharing positive experiences about the capabilities of current models like **Claude Sonnet**.
   - The community eagerly anticipates improvements, especially regarding features and capabilities promised by future **Anthropic** iterations.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Community lusts after NVIDIA GB200**: A member confirmed that this Discord server is dedicated to discussing **lewd NVIDIA GB200 images**.
   - The rapid confirmation by another member highlighted the community's direct and humorous approach.
- **Triton's Interpreter Mode Shines!**: During 2D matrix multiplication, the **error** in **Triton's default mode** was significantly larger compared to **INTERPRET mode**, as detailed in [this GitHub issue](https://github.com/triton-lang/triton/issues/5895).
   - In INTERPRET mode, the error was notably lower, at **9.5367431640625e-07**, sparking a discussion on performance disparities with Torch.
- **CUDA Memory Model causes Confusion**: A beginner in CUDA questioned whether a code snippet violated the **C++ memory model** and asked if it needed acquire/release semantics, posting to [Stack Overflow](https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi) for community feedback.
   - Another member clarified that register definitions are **per thread**, with each thread potentially loading values for an **8x8 matrix**.
- **CPUOffload Challenges**: Members discussed the intricacies of **CPUOffload**, particularly in how to effectively gather **DTensor shards** to rank 0 for optimizer updates without excessive overhead using methods such as `mmap()` or `shm_open()`.
   - A member is also seeking efficient means to perform a CPU optimizer step fused with gradient clipping on rank 0, aiming to use reduced gradients without traditional allreduce setup.
- **Tilelang v0.1.0 Launches!**: The community celebrated the release of [tilelang v0.1.0](https://github.com/tile-ai/tilelang), a new pythonic DSL for high-performance AI kernels with features like dedicated memory allocations and optional layout and pipeline annotations.
   - The tool offers **fine-grained thread-level control** and an invitation was extended to the creator to share more with the community in a future talk.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Unleashes OpenAI o1 and o3 for All**: **OpenAI**'s **o1** and **o3** reasoning model series are now available to all **OpenRouter** users without requiring BYOK, enhancing rate limits for previous key users, as detailed [here](https://x.com/OpenRouterAI/status/1889708759355691327).
   - These models incorporate web search, broadening their utility and streamlining the user experience.
- **Groq's Llamas Zip at Unprecedented Speeds**: Thanks to official **Groq** support, users can harness lightning-fast endpoints for **Llama 3.3** at over **250 tokens per second** and **Llama 3.1** at **600 TPS**, models available as described [at this link](https://openrouter.ai/provider/groq).
   - Bringing your own keys unlocks boosted rate limits, enhancing efficiency.
- **Nitro Feature Turbocharges Throughput**: The `:nitro` suffix is upgraded, allowing users to sort endpoints by latency and throughput, configurable via API or in chat, rather than appearing as separate endpoints.
   - Enhanced charts track provider performance, simplifying comparisons over time.
- **DeepSeek R1 70B Blazes New Speed Trails**: The **Groq DeepSeek R1 70B** achieves approximately **1000 tokens per second**, setting a new standard in speed, with extensive parameter support and BYOK options, information shared [here](https://x.com/OpenRouterAI/status/1889726731571044538).
   - The community reacted positively to the new standard.
- **OpenRouter Chat Histories Vanish into Thin Air**: Users reported losing chat histories after updates, highlighting that histories are stored locally, which they claim was not clearly communicated initially.
   - Members suggest clearer messaging about potential data loss when clearing browser history, to avoid future user frustration.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deep Hermes Release Anticipated**: The community eagerly awaits the release of **Deep-Hermes-8B** model weights, watching for announcements and benchmarks from the NousResearch HuggingFace repo.
   - Teknium indicated ongoing preparations, including benchmarks and a model card, hinting that the model might be used to compose posts about its own release.
- **LM Studio Speculative Decoding Debuts**: The latest LM Studio 0.3.10 Beta introduces **Speculative Decoding**, aiming to accelerate inference using a main and draft model in tandem, promising performance enhancements.
   - Despite the potential, some members reported mixed results, suggesting that **Speculative Decoding** is most effective for larger models and may not always yield noticeable speed gains.
- **Calibration Dataset Generates Question**: Curiosity arose concerning the nature of the calibration dataset used, particularly its seemingly random and unstructured content reminiscent of subpar pretraining data.
   - Jsarnecki clarified that the unusual dataset was chosen intentionally, as research indicated that near-random data snippets led to improved training outcomes, even when contrasted with traditional datasets such as wikitext.
- **Hackathon Superagents Emerge**: A one-day hackathon challenges developers to create next-level **SUPERAGENTS**, integrating Story's **Agent Transaction Control Protocol** across various frameworks and chains.
   - Participants are encouraged to innovate on existing projects or develop new ones, competing for prizes and collaborative opportunities.
- **US Declines AI Safety Declaration**: At an international summit, the US, represented by Vance, declined to sign an AI safety declaration over concerns that partnerships with **authoritarian regimes** like China could jeopardize national security.
   - Disagreements over the language regarding **multilateralism** and international collaboration led to a lack of consensus, particularly about US leadership in AI.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Clamor for Google Sheets Support**: The NotebookLM team is seeking feedback on **Google Sheets** integration, with users requesting the ability to **ingest** data, and they've released [a feedback survey](https://forms.gle/G78qnNCv2UwcYXc16).
   - The survey aims to gather detailed specifications, including the dimensions of the sheets, types of data, and insights users hope to gain from them.
- **NotebookLM Becomes Fantasy Novelist's Muse**: A user is utilizing NotebookLM as a writing assistant for their fantasy novel, focusing on world building, character development, and data organization.
   - The user values the audio generator for synthesizing questions from potential readers, helping identify gaps and inconsistencies in their detailed world building, and they're dynamically refreshing **Google Sheets** to track progress.
- **AI-Podcasts Democratize Content Creation**: A user elaborated on leveraging AI to create podcasts rapidly, highlighting the significant market opportunity, and pointed out how **podcasting** can elevate content consumption and market reach, according to [this article](https://millionai.substack.com/p/create-ai-podcasts-in-seconds-without?r=297y6u&utm_medium=ios&triedRedirect=true).
   - They emphasized transforming static content into engaging audio, maximizing outreach without requiring public speaking, creating value from something like **NotefeedLM**.
- **Students Juggle Limits and Embrace Audio**: Undergraduate users employ NotebookLM to generate mock tests and summarize sources, praising its effectiveness, however the daily query limit makes usage difficult.
   - The **audio conversation** feature is valued for multitasking, but some experience functionality issues, and there are requests for personalized audio features using user's voices.
- **Users Cite Source Formatting Problems**: Users report issues with source display; mangled formatting in PDFs hinders content verification, impacting the overall user experience.
   - The product team acknowledges these formatting issues and is working on potential improvements to accurately display source materials.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OpenRouter Frees OpenAI Models**: OpenRouter made **OpenAI o1 and o3** accessible to all, removing the need for BYOK and raising rate limits, as [announced on X](https://x.com/OpenRouterAI/status/1889708759355691327).
   - The update was well-received, particularly because it enhances functionality, especially when integrated with web search.
- **Users Explore Aider Multi-Session**: Users are seeking capabilities in Aider to manage multiple **tmux sessions** to enhance process control, such as for server spawning.
   - Currently, the workaround involves local setups using **SSH connections** to streamline coding workflows.
- **Editor Model Dreams of Collab**: A proposal suggests training a **1.5b 'editor' model** to work with architect models, improving the efficiency of code editing.
   - The goal is to reduce hallucinations and increase the precision of code diffs in larger contexts.
- **GPT-5 Roadmap Unveiled**: Plans for **GPT-4.5 and GPT-5** aim to unify model offerings and improve user experience, according to [Sam Altman's tweet](https://x.com/sama/status/1889755723078443244).
   - GPT-5 will incorporate diverse technologies and be available to free-tier users with varying intelligence levels.
- **o3-mini Speeds up Coding Tasks**: Feedback indicates **o3-mini** performs admirably and speeds up coding, outperforming other models in specific tasks.
   - Some users observed faster deployment times with **o3**, and others suggest combining it with models like **Sonnet** for optimal results.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL Quality Matches 1.5, Lacks Unique Interpretations**: A discussion compared **SDXL** and **SD 1.5**, noting that **SDXL** achieves comparable quality without a refiner, but lacks **1.5**'s unique interpretations due to a focus on popular aesthetics.
   - Members emphasized the importance of **benchmarks**, pointing out that **SDXL** generally outperforms **SD 1.5** in these controlled evaluations.
- **Flux Model's Consistent Faces Highlight Data Tuning**: The **Flux model** consistently produces similar facial features, like a distinctive cleft chin, which suggests reliance on **quality-tuned data** or specific distillation methods.
   - While some found the diversity lower than **SDXL**, others argued that **Flux's** higher log likelihood distribution allows for diversity improvements via **loras**.
- **Distillation Methods Greatly Affect Model Performance**: It was clarified that the derivation of **Schnell** from **Pro** via 'timestep distilled' differs from **Dev's** use of 'guidance distilled,' significantly influencing model performance and **lora** compatibility.
   - The discussion highlighted how different **data handling** techniques in distillation can critically impact the final model quality and behavior.
- **Human Preference Benchmarks Face Skepticism**: Concerns were raised about **human preference benchmarks** potentially favoring aesthetically pleasing outputs over more objective quality metrics, possibly skewing results.
   - The worry is that these benchmarks might prioritize outputs like 'pretty ladies' instead of accurate representations based on detailed and varied prompts.
- **ComfyUI Linux Transition Causes OOM Errors**: A user reported facing **OOM errors** during video generation after transitioning from **ComfyUI on Windows** to **Linux**, despite following a guide.
   - Community members recommended verifying proper **driver** installations, with one pointing out that inadequate guidance may have led to the system's instability.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Author Flair Sparks Mistrust**: The granting of server author flair led to mixed reactions, with one member expressing *mistrust towards anyone involved in crypto/NFTs*.
   - This sentiment highlights ongoing concerns about trustworthiness within the community.
- **Community Debates Code Review Process**: Members discussed implementing a *code review process* for MCP public servers, suggesting multiple reviewers to manage the workload, given there are 900+ servers.
   - One member jokingly suggested using a language model to pre-screen for malicious code.
- **Open Source LLM Models Crave New Research**: Concerns arose about the need for *ground-breaking research on open-source LLM models*, with mentions of DeepSeek potentially drawing inspiration from OpenAI's work.
   - Despite any shared innovations, it was noted that DeepSeek still leverages OpenAI's technology.
- **Clickhouse & Streamlit Create Dashboards**: One member showed keen interest in building a generative dashboard server using *Clickhouse and Streamlit*, considering monetization strategies.
   - They asked for feedback on Streamlit's effectiveness versus alternatives like PowerBI, hinting at future monetization collaborations.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Posts Job Openings**: Modular recently posted new job openings, signaling ongoing expansion and development efforts within the company, which could lead to future improvements and integrations.
   - These could lead to improvements and new integrations across their products such as Mojo and MAX.
- **Modular Ditches stdlib Meetings**: Regular **stdlib meetings** were discontinued due to scheduling conflicts and the departure of the organizer.
   - Members had trouble accessing the regular meetings and were informed that the meetings are cancelled for the time being.
- **Parameterized traits > Sum Types**: The Mojo team is prioritizing **parameterized traits** over **sum types** due to their enabling of more foundational capabilities.
   - It was pointed out that the focus is on developing ground level features that allow Mojo to represent constructs similar to C.
- **MAX doesn't prioritize Wasm now**: The Wasm backend is currently not a focus for MAX and is not on the near-term roadmap, as MAX focuses on other technologies.
   - One member expressed curiosity about the relevance of Wasm, highlighting its potential for future use despite current priorities.
- **ONNX model execution depends on MAX**: Members noted that Modular's support for executing **ONNX models** largely depends on **MAX**, emphasizing its necessity.
   - This highlights MAX's role in facilitating various ML model executions across the platform, with MAX being crucial for applications utilizing GPUs, though not strictly necessary for running Mojo.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **VAEs Demand Reparameterization**: Discussion arose around why **backpropagation** cannot be directly performed through a distribution in **VAEs**, necessitating the *reparameterization trick* due to the non-differentiable stochastic sampling operation.
   - Members clarified that **VAEs** generate distribution parameters that require stochastic sampling.
- **OpenAI Triumphs in Competitive Programming**: OpenAI released [a paper](https://arxiv.org/abs/2502.06807) detailing their **o3 model's** gold medal performance at IOI 2024 without needing hand-crafted strategies, signaling significant advancements in reasoning models as mentioned in [this tweet](https://x.com/iScienceLuvr/status/1889517116816244995?s=46).
   - The team noted that model flexibility is key, contrasting it with **o1-ioi's** previous requirement for specialized pipelines as covered in [this tweet](https://x.com/polynoamial/status/1889541408065028421?s=46).
- **Scaled Cognition Debuts Agentic APT-1 Model**: Scaled Cognition announced [their APT-1 model](https://x.com/scaledcognition/status/1889721166421479751?s=46), designed specifically for agentic applications, which now tops agent benchmarks.
   - The team highlighted a **$21M** seed round led by Khosla Ventures, utilizing a fully synthetic data pipeline.
- **Glean Launches Scalable AI Agents**: Glean introduced [Glean Agents](https://x.com/glean/status/1889706504812683728), a platform designed for scalable AI agent management, featuring new data integration and governance capabilities.
   - The goal is to boost productivity by offering user-friendly access to company and web data.
- **OpenAI Charts Roadmap with GPT-4.5 and GPT-5**: OpenAI provided [a roadmap update](https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg) indicating the upcoming **GPT-4.5 and GPT-5** models, aiming to unify modeling approaches and simplify product offerings.
   - OpenAI signals a shift away from non-reasoning models, focusing on broader functionality and advanced reasoning capabilities.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Step-Based Checkpointing In Development**: A member inquired about saving checkpoints multiple times per epoch in **Torchtune**, and another mentioned that **Joe** is working on this feature in [PR #2384](https://github.com/pytorch/torchtune/pull/2384).
   - They said it is *a widely requested feature* and is expected to improve the checkpointing process significantly.
- **MLFlow Logger Integration Lands**: The **MLFlow logger integration** was successfully merged, reported by a member excited to test it ASAP.
   - The integration aims to enhance logging capabilities in **Torchtune**.
- **Torchtune Enables Distributed Inference**: A member inquired about running **distributed inference** using multiple GPUs with **Torchtune**, and another shared a [link](https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py) to relevant code.
   - They noted that loading a saved model into **vLLM** will work for distributed inference and be *much faster*.
- **Gradient Accumulation Plagues Training**: There is ongoing confusion around the [gradient accumulation fix](https://github.com/pytorch/torchtune/issues/2334), affecting training effectiveness.
   - Members described hours spent debugging without finding a root cause, and the issue appears complex and may require more collaborative effort.
- **Attention Mechanisms Still Crucial**: A participant succinctly stated that *attention is still all we need*, underscoring its fundamental role in modern AI models.
   - This reinforces the ongoing importance and focus on attention mechanisms in the field of artificial intelligence.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **TinyStories Paper Trains Models on Small Data**: The [tinystories paper](https://link.to.tinystories) was recommended for training ML models with **limited datasets**, offering strategies for effective learning under dataset constraints.
   - This could be especially useful for scenarios where obtaining large datasets is difficult or costly.
- **EU Pledges Funds into AI Gigafactories**: The European Union committed **200 billion euros** in AI investment to compete with the U.S. and China, focusing on creating **AI gigafactories** for advanced model training, according to [Ursula von der Leyen's announcement](https://www.msn.com/en-us/money/companies/eu-pledges-200-billion-in-ai-spending-in-bid-to-catch-up-with-u-s-china/ar-AA1yO0Su).
   - This initiative aims to position Europe as a leading continent in AI technology and development.
- **DeepScaleR Beats Scaling Expectations**: The [DeepScaleR preview](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scali) showcased a **1.5B model** that significantly scales up RL, sparking excitement within the community.
   - The model's advancements suggest a promising revival of RL techniques.
- **Reuters Copyright Triumphs Over AI**: In a landmark case, [Thomson Reuters secured a copyright victory](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/) against Ross Intelligence, underscoring the importance of respecting intellectual property in AI.
   - Judge Stephanos Bibas ruled decisively against Ross, stating, *None of Ross’s possible defenses holds water*.
- **OpenAI's Roadmap Teases GPT-4.5**: OpenAI revealed that **GPT-4.5** will be their last model not using chain-of-thought, planning to integrate **o-series and GPT-series models**, according to [Sam Altman](https://x.com/sama/status/1889755723078443244?t=EgnihPXVoD2fsS9ag5u5SA&s=19).
   - Their goal is for models to *just work* across various applications, simplifying user interaction.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CUDA backend hacks its way to Windows**: A user got the **CUDA backend working on Windows** by correcting the autogen files with appropriate DLL names, but standard CI runners lack GPU support.
   - They suggested possibly hard-coding the CUDA version to keep the setup simple. See [this PR](https://github.com/tinygrad/tinygrad/pull/9036) for more details.
- **CI struggles with backend env vars**: The **Windows CI** was not propagating backend environment variables between steps, leading to a default switch to CLANG during testing.
   - A pull request was initiated to ensure that environment variables persist between CI steps for proper functionality; see [this PR](https://github.com/tinygrad/tinygrad/pull/9047).
- **Testing Iteration causes chaos**: Doubts arose about switching from recursion to iteration, as it caused many tests to fail beyond the original changes.
   - The immediate cause of CI failures stemmed from an indentation issue that inadvertently affected critical functionality within the code.
- **Tinygrad promises cheaper hardware**: A user questioned the advantages of switching to **tinygrad** from established frameworks like **PyTorch**, citing personal experience with the latter.
   - Another member suggested that choosing tinygrad could lead to **cheaper hardware**, a better understanding of underlying processes, and potentially faster model performance.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex needs Open Source Engineer**: A full-time position for an **open source engineer** at [@llama_index](https://twitter.com/llama_index) has been announced, seeking candidates passionate about **Python** and **AI**.
   - More details about expanding the **llama_index** framework are available [here](https://t.co/WMgdaauxP8).
- **Nomic AI Improves Document Workflows**: [@nomic_ai](https://twitter.com/nomic_ai) is showing the importance of a great **embedding model** for effective **Agentic Document Workflows**.
   - This new development has been positively received, marking a significant step in enhancing these workflows, with more details shared [here](https://t.co/pezsylHNpH).
- **Data Loaders are Critical for RAG Systems**: Members discussed the desire to experiment with different data loaders for building **RAG systems** and building query engines, recommending [llamahub](https://llamahub.example) for resources.
   - One member emphasized the importance of selecting loaders tailored to specific use cases.
- **Members tackle Batch processing PDFs**: One member sought advice on methods to **batch process PDFs**, asking for clarification on the specific approach being considered.
   - The conversation suggests a need for more specialized tools or scripts to efficiently manage bulk PDF operations.
- **Crafting Smart Query Engines with Filters**: A member asked for tips on using **predefined filters** within query engine tools for different topics, aiming for an efficient workflow without creating multiple indexes.
   - Another member shared a code example to illustrate how to implement a query engine tool with specified filters.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Hackathon Winners Crowned**: The **LLM Agents MOOC Hackathon** winners were announced, drawing ~**3,000** participants from **127** countries and **1,100+** universities, as noted in a [tweet by Prof. Dawn Song](https://x.com/dawnsongtweets/status/1889686697564315963).
   - Key participants included **Amazon**, **Microsoft**, **Samsung**, and **Salesforce**, and winning teams are showcased on the [hackathon website](https://rdi.berkeley.edu/llm-agents-hackathon/).
- **Advanced LLM MOOC Incoming**: The **Spring 2025 MOOC** focusing on **Advanced LLM Agents** has been launched, per [Professor Dawn Song's announcement](https://x.com/dawnsongtweets/status/1889355520294944829), covering **Reasoning & Planning**, **Multimodal Agents**, and **AI for Mathematics**.
   - Building on the **Fall 2024** MOOC's success, with **15K+** registered learners and **200K+** YouTube lecture views, live sessions are scheduled every **Monday at 4:10 PM PT**.
- **Curriculum Details Coming Soon**: Details for the **MOOC curriculum** are expected to be released in approximately **two weeks**, and there will not be a hackathon this semester.
   - MooC students are waiting on more information on how to apply for research subjects.
- **DeepScaleR Scales RL with 1.5B Model**: The [DeepScaleR model](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) surpasses the O1 preview using a **1.5B model** by scaling reinforcement learning techniques, per a recent document.
   - Details regarding assignment deadlines are set to be released soon, with reminders for catching up on missed lectures.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic AI Offers Steam Gift Card**: A member announced a **$50 Steam gift card giveaway** via [steamcommunity.com/gift-card/pay/50](https://u.to/kR7WIQ).
   - The post received mixed reactions, with one member labeling it as **spam**.
- **Debate Arises over TextWeb-UI Installation Complexity**: A member mentioned that **TextWeb-UI** requires a complex installation process, with one user noting that it's not an easy `.exe` install.
   - This complexity raised concerns about its accessibility and ease of use for some members.
- **Mobile App Battery Life is Questioned**: Concerns arose about using mobile applications for both **iOS and Android**, with one member speculating that such apps could drain a **device's battery in 1 hour**.
   - The discussion underscored performance issues with mobile applications within the **Nomic AI** ecosystem.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Hit by Failed Fetch Error**: Users reported a 'Failed to fetch' error when attempting to log into their personal account with their credentials, but offered *not very informative* feedback on the experience.
   - The error prompted inquiries about possible filtering that could be blocking API requests.
- **Cohere API Requests Possibly Getting Filtered?**: Members are investigating whether filtering might be causing the failure in API requests during the login attempt.
   - This concern suggests a deeper investigation may be needed to identify connectivity issues or software restrictions.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Podcast Success Unlocked with AI**: A free workshop on **Thursday, Feb 13 at 9PM IST** teaches creators how to launch podcasts using just **AI** and no expensive equipment, participants will learn the **fundamentals of AI audio models**.
   - The session provides hands-on experience with platforms like [ElevenLabs](https://elevenlabs.io) and [PlayHT](https://playht.com) to effortlessly **transform text into audio content**.
- **Hands-On Audio Creation**: Attendees gain hands-on experience with leading **voice generation platforms**, allowing them to **transform text into audio content** effortlessly and develop their own **open source NotebookLM** for custom implementations.
   - Additional free resources and tools dedicated to **generative AI solutions** are available through [Build Fast With AI](https://t.me/BuildFastWithAI), offering the **latest Gen AI tools**, roadmaps, and workshop links.



---


The **DSPy Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1338968843212685322)** (1027 messages🔥🔥🔥): 

> `GRPO Implementation, Dataset Cleaning, MoE Models, Liger Kernel vs Apple Kernel, OpenR1-Math-Raw Dataset` 


- **GRPO Implementation Guidance**: Members discussed integrating GRPO with models like Mistral and Llama, emphasizing the importance of using appropriate training templates to utilize special tokens like <thinking> effectively.
   - Challenges arose when testing outputs, especially regarding the model not producing expected tokens despite correct implementation.
- **Importance of Dataset Cleaning**: There were discussions around dataset cleaning, with members noting that simply removing missing values without understanding the data could dilute the dataset's relevance.
   - It was suggested that thorough analysis and understanding are crucial for effective data preparation prior to training.
- **Exploring MoE Models**: A member shared their intention to build a custom MoE model after gaining experience with existing architectures, citing the potential advantages of MoE for handling large models.
   - Concerns about the costs and compute requirements for training such models were also discussed.
- **Comparison of Liger and Apple Kernels**: The group compared the Liger kernel with Apple's cross-entropy implementation, noting differences in how they process logits and the performance implications.
   - Members pointed out that while Liger may have certain speed advantages, Apple's kernel performs certain operations more efficiently due to its complete implementation.
- **OpenR1-Math-Raw Dataset Availability**: The OpenR1-Math-Raw dataset was introduced as a resource for mathematical reasoning, featuring over 516k problems and verified solutions.
   - This dataset aims to assist users in generating and evaluating mathematical reasoning tasks, making it a potentially valuable tool for training LLMs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (All Versions) - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview">agentica-org/DeepScaleR-1.5B-Preview · Hugging Face</a>: no description found</li><li><a href="https://pastebin.com/cfibZ8DG">Qwen2VL-GRPO-o3mini - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw">open-r1/OpenR1-Math-Raw · Datasets at Hugging Face</a>: no description found</li><li><a href="https://build.nvidia.com/nvidia/nemotron-4-340b-reward">nemotron-4-340b-reward Model by NVIDIA | NVIDIA NIM</a>: Grades responses on five attributes helpfulness, correctness, coherence, complexity and verbosity.</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://github.com/TruffleClock/nano-r1/blob/main/nano-r1.ipynb">nano-r1/nano-r1.ipynb at main · TruffleClock/nano-r1</a>: Contribute to TruffleClock/nano-r1 development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset">agentica-org/DeepScaleR-Preview-Dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/agentica-project/deepscaler/tree/main/scripts/train">deepscaler/scripts/train at main · agentica-project/deepscaler</a>: Democratizing Reinforcement Learning for LLMs. Contribute to agentica-project/deepscaler development by creating an account on GitHub.</li><li><a href="https://github.com/deepseek-ai/DeepSeek-Math">GitHub - deepseek-ai/DeepSeek-Math: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models - deepseek-ai/DeepSeek-Math</li><li><a href="https://huggingface.co/Blackroot/SimpleDiffusion-MultiHeadAttentionNope/blob/main/train.py#L94>">train.py · Blackroot/SimpleDiffusion-MultiHeadAttentionNope at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1590">Training a Vision Model with Text-only Inputs · Issue #1590 · unslothai/unsloth</a>: I need to train the vision model using only text inputs. I tried using Colab notebooks but noticed that images are mandatory in the data. After researching a bit more, I found a Colab notebook that...</li><li><a href="https://x.com/UnslothAI/status/1889726411478278183">Tweet from Unsloth AI (@UnslothAI)</a>: Train your own reasoning LLM using DeepSeek&#39;s GRPO algorithm with our free notebook!You&#39;ll transform Llama 3.1 (8B) to have chain-of-thought. Unsloth makes GRPO use 80% less VRAM.Guide: https:...</li><li><a href="https://github.com/fishaudio/fish-speech/discussions/870">Why is FishSpeech API Faster Than My Self-Hosted Setup? · fishaudio/fish-speech · Discussion #870</a>: Hi everyone, I&#39;m trying to understand the performance gap between inference with the FishSpeech API and my self-hosted deployment Here&#39;s my configuration: Technical Details: Hardware: NVIDIA H...</li><li><a href="https://github.com/triton-lang/triton/issues/5895">error is significantly larger in default mode than INTERPRET mode · Issue #5895 · triton-lang/triton</a>: Describe the bug For the simple 2D matrix multiplation, difference between my triton kernel and torch are: INTERPRET mode: 9.5367431640625e-07 (set os.environ[&quot;TRITON_INTERPRET&quot;] = &quot;1&q...</li><li><a href="https://github.com/allenai/s2orc?tab=readme-ov-file#download-instructions">GitHub - allenai/s2orc: S2ORC: The Semantic Scholar Open Research Corpus:  https://www.aclweb.org/anthology/2020.acl-main.447/</a>: S2ORC: The Semantic Scholar Open Research Corpus:  https://www.aclweb.org/anthology/2020.acl-main.447/ - allenai/s2orc</li><li><a href="https://github.com/datadreamer-dev/DataDreamer">GitHub - datadreamer-dev/DataDreamer: DataDreamer: Prompt. Generate Synthetic Data. Train &amp; Align Models.    🤖💤</a>: DataDreamer: Prompt. Generate Synthetic Data. Train &amp; Align Models.    🤖💤 - datadreamer-dev/DataDreamer</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py">Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py at main · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/apple/ml-cross-entropy/blob/main/cut_cross_entropy/cce_lse_forward.py#L79">ml-cross-entropy/cut_cross_entropy/cce_lse_forward.py at main · apple/ml-cross-entropy</a>: Contribute to apple/ml-cross-entropy development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py#L25>">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at main · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/cross_entropy.py">flash-attention/flash_attn/ops/triton/cross_entropy.py at main · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B-v0.2">Skywork/Skywork-Reward-Gemma-2-27B-v0.2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py#L264>">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at main · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/Deep-Agent/R1-V/blob/367658d518d3173ee4c2a47123547adeab363b14/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L62">R1-V/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py at 367658d518d3173ee4c2a47123547adeab363b14 · Deep-Agent/R1-V</a>: Witness the aha moment of VLM with less than $3. Contribute to Deep-Agent/R1-V development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1338980495878389781)** (12 messages🔥): 

> `Reading resources, Sourcing information on AI/ML/RAG, Reddit as a source, Local Llama hosting` 


- **Reading resources praised**: A member expressed appreciation for a shared reading resource, calling it **great** and thanking another member for it.
- **Exploring information sources for AI/ML/RAG**: Members discussed various methods for sourcing information on AI, ML, and RAG, mentioning they use **Twitter** and **RSS** feeds.
   - One member was open to suggestions and inquired if there are **better** sources beyond their current methods.
- **Reddit recommended for discussions**: A member suggested using **Reddit** as a source, particularly for insights about local developments in AI.
   - They emphasized that the community shares a lot of **industry happenings**, making it a valuable resource.
- **Local Llama as a thought source**: The discussion mentioned **local llama** hosting as an option, though some clarified it refers to community discussions rather than personal hosting.
   - One member acknowledged this insight and planned to **give it a try** after recognizing its value.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1338965061326082232)** (149 messages🔥🔥): 

> `Fine-tuning Models, GRPO Optimization, Experiment Tracking, Qwen Model Performance, Mistral Benchmarking` 


- **GRPO Fine-tuning on A100**: A user is experiencing out-of-memory (OOM) issues while fine-tuning the **Qwen 32B** model, initially using a 128k context and progressively reducing to 16k.
   - They are unsure about memory allocation in the trainer and whether it's feasible to achieve the full 128k context on their **A100**.
- **Experiment Tracking with wandb**: A user is contemplating whether to configure **wandb** or utilize experiment tracking features within **Unsloth** for loss tracking.
   - Support for training loss tracking is confirmed, particularly regarding the **GRPO** process.
- **Mistral Fine-tuning Performance**: Unsloth's **Mistral 7B** model can be fine-tuned up to **14x faster** using **QLoRA**, significantly reducing VRAM usage.
   - The performance improvements enable users to finetune models effectively even on less powerful hardware.
- **Tokenization and Data Preparation**: Discussion on proper data formatting for fine-tuning, emphasizing the importance of tokenization and structured datasets for large language models.
   - Users are directed to useful resources like the **Hugging Face** documentation for structuring chat templates.
- **TeleChat Model Context Limitations**: A user inquiry about the feasibility of running the **TeleChat** model at full capability highlights performance limitations due to model architecture.
   - It was noted that if users want to tune this older model, they might need to create support layers from scratch.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=vITh0KVJ10qX)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks">Unsloth Benchmarks | Unsloth Documentation</a>: Want to know how fast Unsloth is?</li><li><a href="https://unsloth.ai/blog/r1-reasoning">Train your own R1 reasoning model locally (GRPO)</a>: You can now reproduce your own DeepSeek-R1 reasoning model with Unsloth 100% locally. Using GRPO.Open-source, free and beginner friendly.</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://unsloth.ai/blog/mistral-benchmark#Breakdown">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...</li><li><a href="https://docs.unsloth.ai/basics/datasets-101#getting-started">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://github.com/unslothai/unsloth/tree/main/unsloth/models">unsloth/unsloth/models at main · unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://github.com/MaxHastings/Kolo">GitHub - MaxHastings/Kolo: A one stop shop for data generation, fine tuning and testing LLMs locally using the best tools available. Keeping it simple and versatile!</a>: A one stop shop for data generation, fine tuning and testing LLMs locally using the best tools available. Keeping it simple and versatile! - MaxHastings/Kolo</li><li><a href="https://github.com/sathvikask0/r1-distilled-RL/blob/main/config.py">r1-distilled-RL/config.py at main · sathvikask0/r1-distilled-RL</a>: Contribute to sathvikask0/r1-distilled-RL development by creating an account on GitHub.</li><li><a href="https://asksathvik.substack.com/p/some-rl-ideas-i-am-currently-working">Some RL ideas I am currently working on..</a>: Experiments I am currently trying out:</li><li><a href="https://github.com/sathvikask0/r1-distilled-RL">GitHub - sathvikask0/r1-distilled-RL</a>: Contribute to sathvikask0/r1-distilled-RL development by creating an account on GitHub.</li><li><a href="https://huggingface.co/Tele-AI/TeleChat-12B">Tele-AI/TeleChat-12B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Templates</a>: no description found</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | Model Cards and Prompt formats</a>: Llama 3.1 - the most capable open model.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1339021502066200677)** (71 messages🔥🔥): 

> `Reward Function Challenges, In-context Training, GRPO Methodology, Fine-tuning Experiences, Mistral and Numeric Data Limitations` 


- **Challenges in Reward Function Design**: There is a consensus that the reward function is currently too generous for certain phrases, causing repetitive outputs like "Hmm, let's see..." to dominate the model’s responses.
   - Users emphasized the need to implement penalties for overused phrases to foster greater diversity in generation.
- **Exploring In-context Training**: A suggestion was made to try in-context training by sending a sliding window of previous messages to help the model understand the evolution of its responses.
   - This method aims to improve self-supervision rather than treating each generation as unrelated.
- **Advancements with GRPO Method**: The Group Robust Preference Optimization (GRPO) method was discussed as a novel approach to optimizing LLMs for specific group preferences, emphasizing a robust policy for aligning to user needs.
   - Users referred to relevant academic papers highlighting the importance of tailored reward functions to enhance model performance.
- **Fine-tuning Insights and Experiences**: One user shared that applying a length penalty resulted in reduced output lengths while maintaining accuracy, showcasing effective fine-tuning techniques.
   - The conversation highlighted the significance of tuning hyperparameters like reward functions and training durations for successful model adaptation.
- **Limitations of Transformers with Numeric Data**: There is an acknowledgment that transformers like Mistral struggle with tabular numerical data, leading to inconsistent calculations.
   - This insight raises concerns about the applicability of certain models in scenarios requiring precise numerical reasoning or operations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Co...</li><li><a href="https://arxiv.org/abs/2405.20304">Group Robust Preference Optimization in Reward-free RLHF</a>: Adapting large language models (LLMs) for specific tasks usually involves fine-tuning through reinforcement learning with human feedback (RLHF) on preference data. While these data often come from div...</li><li><a href="https://x.com/AskSathvik/status/1889697491769078270">Tweet from ASK Sathvik (@AskSathvik)</a>: RL simply works.I just used length penalty inspired from the kimi1.5 paper and trained r1-1.5b on the gsm8k dataset and the completion length drops from &gt;2000 tokens to &lt;500 tokens while keeping...</li><li><a href="https://github.com/sathvikask0/r1-distilled-RL">GitHub - sathvikask0/r1-distilled-RL</a>: Contribute to sathvikask0/r1-distilled-RL development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1338963390642782231)** (308 messages🔥🔥): 

> `Voice Chatbot Technology, Home Assistant Voice Integration, Moxie Robot Companion, ESP32 Microcontrollers, ChatGPT and Other AI Models` 


- **Exploring DIY Voice Chatbots**: Users discussed building voice chatbots using various off-the-shelf products, including DIY projects with Raspberry Pi and ESP32, with recommendations like the Eilik companion robot.
   - There was a mention of creating custom 3D prints to stylize these devices, showcasing both creativity and functionality.
- **Home Assistant Voice Setup**: A member shared their experience using Home Assistant Voice, which allows for a customized voice assistant that interacts with OpenAI APIs, providing features like web search and smart home controls.
   - The integration requires running a Home Assistant server, and users can configure multilingual support, beneficial for diverse communities.
- **Moxie Robot’s Current Status**: The conversation highlighted concerns about Moxie, a children's robot companion, which faced issues leading to its uncertain future; nonetheless, it serves as a reference for emotional intelligence in robotic companions.
   - Participants speculated about potential successors to Moxie and discussed its design features emphasizing interactions with children.
- **Integration of Robots with Voice Assistants**: Users are wiring toy robots to voice assistants, discussing the practicality of various microphones and setup configurations to enhance functionality.
   - Experiences shared include using USB microphones for better audio quality and plans for integrating MCP servers for robot control.
- **AI Model Comparisons**: Dialogue included comparing ChatGPT with Claude for personality and responsiveness, revealing preferences for different AI models based on task requirements.
   - Users highlighted the functionalities of various versions, emphasizing how the models adapt and handle user interactions differently.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.04517">xLSTM: Extended Long Short-Term Memory</a>: In the 1990s, the constant error carousel and gating were introduced as the central ideas of the Long Short-Term Memory (LSTM). Since then, LSTMs have stood the test of time and contributed to numerou...</li><li><a href="https://community.openai.com/t/webrtc-real-time-api-with-microcontroller/1059806">Webrtc Real Time API with microcontroller</a>: Hi! In the Day 9 demo, we saw a stuffed animal with a microcontroller consuming the WebRTC Real-Time API (link: YouTube Live).  Could you provide more details about the overall architecture? For examp...</li><li><a href="https://www.youtube.com/watch?v=7dd_r-aqecw">Rethinking social development with Moxie, a robot companion for kids</a>: With its blue body and big anime eyes, Moxie wants to be friends with your child. Named one of Time’s top inventions of 2020, the AI-powered robot is designe...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1inmkbc/agenticaorgdeepscaler15bpreview/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1339218726628626514)** (9 messages🔥): 

> `Account Management Improvements, Custom GPT Model, Hiring Experts, Discussion Guidelines` 


- **Request for Account Management Enhancements**: A user expressed a desire for improvements in ChatGPT, including the ability to change account emails, transfer chats between accounts, and bulk delete chats.
   - Another member noted that some features were previously designed to be restricted, but sharing threads is still possible, indicating frustrations with chat management.
- **Clarification on Custom GPT Models**: In response to a question about the model used for custom GPTs, a member confirmed they run on **GPT-4o**.
   - This highlights user interest in the specifics of model implementation behind custom GPT configurations.
- **Seeking Expertise for Startup**: A user stated they need to hire an expert for their startup, inviting qualified members to reach out with their experience.
   - The reply humorously challenged the vagueness of their request, suggesting specificity in the expertise sought.
- **Discussion Channel Guidelines**: A channel moderator reminded users to keep GPT discussions separate from ChatGPT suggestions and directed them to the appropriate channel.
   - This emphasizes the need for organized discussion topics and adherence to community guidelines.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1338992028088012842)** (12 messages🔥): 

> `Iterative Prompting Benefits, Function Calling Issues, Prompt Engineering Best Practices, Community Engagement, Prompt Sharing Guidelines` 


- **Iterative Prompting boosts results**: A user shared that **iterative prompting** really helps by starting with a baseline and continually refining it to achieve desired outcomes.
   - This emphasizes the importance of specificity, especially since **LLMs can't read minds** and require clear instructions.
- **Function Calling in Prompts causes troubles**: A member discussed issues with **function calling** in their system prompt, describing how status indications sometimes fail or trigger unnecessarily based on client interactions.
   - They indicated lagging performance even after specifying to avoid function calls on ambiguous responses.
- **Best practices in prompt engineering**: In response to function issues, a user advised to provide the model with **exact instructions** on desired outcomes before detailing the methods.
   - This strategy aims to minimize the model's guesswork, leading to more consistent execution of logic.
- **Community welcomes prompt sharing**: An enthusiastic user expressed interest in posting prompts, and the community encouraged sharing as long as it fits the channel's focus.
   - Members suggested focusing on discussions or problem-solving rather than simply dumping prompts to enhance engagement.
- **Member struggles with prompt length**: A user mentioned their difficulties entering a prompt due to its complexity, indicating limitations in sharing certain information.
   - This highlights the ongoing challenges some members face when trying to contribute detailed prompts in discussions.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1338992028088012842)** (12 messages🔥): 

> `Iterative Prompting, Function Calling Issues, Prompt Sharing Etiquette, Clarifying Model Instructions` 


- **Iterative Prompting enhances results**: A member highlighted that **iterative prompting** helps refine inputs continuously to achieve better AI responses.
   - They emphasized starting with a base prompt and improving it iteratively until the desired result is achieved.
- **Function Calling in Prompts poses challenges**: A member expressed difficulties with **function calling** in their system prompt, stating that the AI sometimes fails to indicate statuses accurately.
   - They noted that functions are still being called without relevant user responses, causing potential lead losses.
- **Open discussion welcomed for prompt issues**: A new member shared their issues regarding prompt engineering and invited **discussions** on the matter.
   - They referenced their system's function structure and commented on the ambiguous responses received from the AI.
- **Prompt Sharing best practices discussed**: A member suggested that sharing prompts works best when accompanied by *questions or observations* rather than info dumps.
   - They mentioned that this channel is better for discussing unusual situations rather than just sharing prompts.
- **Importance of Clear Instructions for AI**: One participant advised being very clear about what the model is expected to do before providing instructions.
   - They indicated that unclear instructions may lead the AI to make varying guesses on tasks, resulting in inconsistent outputs.


  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1338964659054579862)** (19 messages🔥): 

> `Codeium Extension Issues, Switching to Alternatives, Pre-release Version Usage, Support Concerns, Recent Updates` 


- **Codeium Extension Lags Behind Features**: Members expressed frustration that the **Codeium extension** is falling behind due to a focus on **windsurf** and **enterprise offerings**.
   - One noted that the extension is still available under the enterprise option, illustrating the duality of product focus.
- **Evaluating Alternatives: Cursor vs Codeium**: There’s discussion about switching to **Cursor**, with recommendations to consider it as a replacement for Codeium.
   - However, it's pointed out that Cursor isn't the same as an editor extension, leaving some users torn over the differences.
- **Pre-release Version Conflicts**: A user inquired about how to use the **pre-released 1.37 version** of Codeium in **Goland**, finding no update button as expected.
   - There was mention of the option to switch to pre-release due to ongoing bugs in the standard version.
- **Support Feedback: Bug Fixes Needed**: Concerns were raised about **Codeium's support**, with one user stating it has been disappointing and unhelpful.
   - Another user found that recent updates mostly included minor tweaks rather than substantial bug fixes that would impact usability.
- **Latest Release Update Announcements**: An announcement indicated that issues with authorization in **Forge** would be resolved in the upcoming release **1.36.1**.
   - Despite this, one user reported that the situation regarding improvements significantly lagging behind expectations, even mentioning the year 2025.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1338964644752134225)** (281 messages🔥🔥): 

> `Windsurf Issues and Errors, Model Comparisons, User Feedback and Suggestions, Aggregate Model Performances, Support and Stability Concerns` 


- **Windsurf experiencing errors and outages**: Users reported ongoing issues with Windsurf, mentioning repeated internal errors when using Cascade and problems with the Gemini model on multiple occasions.
   - Many expressed frustration over recent performance drops, particularly the inability to edit files and the unreliability of certain features.
- **Comparative effectiveness of AI models**: A user provided an unofficial ranking of Windsurf models, noting **Claude 3.5 Sonnet** as the top performer due to its context handling and tool calling capabilities.
   - Models like **Gemini 2.0 Flash** and **O3-Mini** were praised for their speed and pricing, while others like **GPT-4o** received criticism for poor performance.
- **Feedback on AI model limitations**: Several users emphasized the importance of user vigilance when working with AI, indicating that blindly trusting AI-generated outputs could lead to costly mistakes.
   - The conversation highlighted a need for clearer risk assessments and user education regarding LLM capabilities.
- **Document source requests for LLMs**: Users discussed the potential for adding custom document sources in Windsurf, referencing a standardized approach via the llms.txt format for indexing documentation effectively.
   - The community expressed hopes for improvements in this area to enhance functionality and ease of access to information.
- **User experiences with credit management**: There was discussion surrounding user strategies to optimize credit usage within Windsurf, including leveraging Cascade for simpler tasks to conserve credits.
   - Users requested additional plans or options indicating a desire for more flexibility in pricing structures to meet their needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://shitposting.pictures/ElRlAJulppNd">A Hand-Curated Shitpost Picture</a>: no description found</li><li><a href="https://codeium.com/changelog/windsurf-next">Windsurf Next Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Next extension.</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.canny.io/feature-requests/p/windsurfs-autocomplete-now-working-around-08-35-41202-utc">Windsurf&#x27;s Autocomplete now working around  08:35:41.202 UTC | Feature Requests | Codeium</a>: It was working at other times, but stopped working in the afternoon of UTC+8. All I see from the log is request canceled.</li><li><a href="https://status.codeium.com">Codeium Status</a>: no description found</li><li><a href="https://directory.llmstxt.cloud">llms.txt directory</a>: no description found</li><li><a href="https://mintlify.com/blog/simplifying-docs-with-llms-txt">Simplifying docs for AI with /llms.txt</a>: Why we&#x27;re providing a better way for LLMs to process documentation.</li><li><a href="https://docs.github.com/articles/restricting-access-to-your-organization-s-data/">Managing OAuth access to your organization&#x27;s data - GitHub Docs</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1338964311917203496)** (240 messages🔥🔥): 

> `Perplexity AI models, Sonar vs DeepSeek R1, API functionality, Real-time browsing, User experiences with Perplexity` 


- **Clarification on Perplexity AI models**: Users discussed the differences between DeepSeek R1 and Sonar Reasoning Pro, noting that Sonar is built on R1 and optimized for web search responses.
   - It was suggested that Sonar Reasoning Pro may eventually replace DeepSeek R1 in the Perplexity app.
- **Issues with Perplexity API**: Multiple users reported experiencing 500 internal server errors while attempting to access the Perplexity API, raising concerns about its reliability.
   - Despite the status page indicating operational status, users shared frustrations about the API's performance.
- **Real-time browsing capabilities**: A user inquired about whether Perplexity offers real-time internet browsing or is limited to information up to 2023.
   - It was confirmed that Perplexity can perform searches based on current links, allowing flexibility in browsing.
- **User experience complaints**: Several users expressed dissatisfaction with the perception that responses from Perplexity are less concise compared to DeepSeek.
   - Discussions also highlighted frustrations regarding the lack of clear documentation related to model versions and capabilities.
- **Perplexity features and tools**: The channel referenced a user-developed Chrome extension that highlights sources cited by Perplexity responses.
   - Members questioned whether similar features could be implemented for Firefox or on mobile versions of Perplexity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/model-cards">no title found</a>: no description found</li><li><a href="https://tenor.com/view/r%C3%A1pido-fast-snail-robot-gif-15498737">Rápido Fast GIF - Rápido Fast Snail - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://pastebin.com/BFmw7FBc">CLAUDE:The Flash is a DC Comics superhero known as &quot;The Scarlet Speedster&quot; w - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://x.com/aravsrinivas/status/1889398205076451697?s=46&t=Un1yLqIRg3sDiqpmnWHBfg">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We’ve post-trained some really good models on top of Llama 3.3 that far surpass 4o-mini and 3.5-Haiku and match 4o and 3.5-Sonnet for answer quality, despite being way cheaper and blazing fast! Users ...</li><li><a href="https://x.com/perplexity_ai/status/1889392617479082323?s=61">Tweet from Perplexity (@perplexity_ai)</a>: Perplexity&#39;s Sonar—built on Llama 3.3 70b—outperforms GPT-4o-mini and Claude 3.5 Haiku while matching or surpassing top models like GPT-4o and Claude 3.5 Sonnet in user satisfaction.At 1200 tokens...</li><li><a href="https://x.com/pplxfinance/status/1889742180421337120?s=61">Tweet from Perplexity Finance (@PPLXfinance)</a>: Your daily source for the latest market insights—now live on Perplexity.Market summaries, daily highlights, earnings snapshots, and everything you need to understand the &#34;why&#34; behind it all.Fi...</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1indfjd/pro_was_600_requests_per_day_then_300_then_now_100/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1ilzw2e/i_made_a_chrome_extension_to_highlight_evidence/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://sonar.perplexity.ai">Sonar by Perplexity</a>: Build with the best AI answer engine API, created by Perplexity. Power your products with the fastest, cheapest offering out there with search grounding. Delivering unparalleled real-time, web-wide re...</li><li><a href="https://sonar.perplexity.ai/">Sonar by Perplexity</a>: Build with the best AI answer engine API, created by Perplexity. Power your products with the fastest, cheapest offering out there with search grounding. Delivering unparalleled real-time, web-wide re...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1339019112525074543)** (9 messages🔥): 

> `OpenAI Rebranding, Apple Tabletop Robot Prototype, Largest Structure in Universe, Street Art Discussion, EU AI Investment` 


- **OpenAI Rebrands and News Highlights**: A video titled *YouTube* discusses the recent **rebranding of OpenAI** amongst other significant happenings including an **Apple prototype** for a tabletop robot and the discovery of the **largest structure in the universe**.
   - View the video [here](https://www.youtube.com/embed/9SUxli8UDA0) for detailed insights.
- **Exploration of Street Art**: A user shared a link discussing **street art** and its various forms and implications.
   - For a deeper dive, check the resource [here](https://www.perplexity.ai/search/tell-me-about-street-art-artri-4PpxVJBsSOWQ_T36v9e7NA).
- **EU's AI Investment Insights**: The topic of **EU AI investment** was brought forward, highlighting its strategic importance in the AI landscape.
   - For more details, the information can be accessed [here](https://www.perplexity.ai/search/eu-ai-investment-aE_wZ53LRUCrT.ntggaGZQ).



**Link mentioned**: <a href="https://www.youtube.com/embed/9SUxli8UDA0">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1339200031378509965)** (20 messages🔥): 

> `401 Authorization Required, API 500 Errors, Token Issues` 


- **User encounters 401 Authorization Error**: A user reported receiving a **401 Authorization Required** error when trying to access the API, asking for help on what might be wrong.
   - *DenoLand* suggested that the user needed to remove the `<>` around their token, but the issue persisted.
- **Resolution of initial authorization issue**: After some troubleshooting, the user announced that they resolved the issue and the API started working, expressing gratitude.
   - They indicated that they received a different error message before finding a solution.
- **Widespread 500 Error Reports**: Multiple users reported encountering **500 errors** from the API, indicating that the service was down.
   - Comments reflected general frustration as users noted failures in production with some API calls still succeeding.
- **Immediate concern over API availability**: Users described the situation as not good, with consistent **500 errors** on every API call.
   - This raised concerns regarding the reliability of the API service in production environments.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1338963567155875953)** (85 messages🔥🔥): 

> `LM Studio and Model Usage, Deepseek Model Applications, Audio Models Support, Markdown Input Rendering Issue, Coding LLM Preferences` 


- **Understanding LM Studio Model Usage**: Members discussed that LM Studio requires specific VRAM to load models efficiently, with **Duckyblender** noting models should fit entirely into VRAM to avoid performance issues.
   - Another member confirmed that the size of the model correlates roughly with required VRAM, particularly when using models like the **Deepseek R1 distill model**.
- **Applications of Deepseek Models**: Members inquired about using the **Deepseek R1 distill model** for math and reasoning tasks, with **Duckyblender** suggesting it could also be tested for coding despite not being its primary function.
   - Community members expanded on various potential uses for the model, emphasizing its capability to handle complex problems.
- **Challenges with Audio Models**: **Heyitsyorkie** stated that LM Studio does not support audio models like **Qwen2-Audio-7B-GGUF**, prompting members to seek alternative methods for using audio models.
   - Advice was given regarding exploring external tools or platforms as options for working with audio models.
- **Markdown Rendering Bugs**: **Vandalbyte** reported a bug where markdown input is rendered as formatted text rather than displayed as raw text, causing confusion in the chat interface.
   - An issue was opened in the bug tracker, highlighting the unexpected behavior in markdown rendering and requesting further inspection.
- **Preferences for Coding LLMs**: Community discussions revealed **Codestral 22b** as a preferred choice for coding LLMs, while another member shared their experience using **Claude Desktop**.
   - Members shared various opinions on different models, suggesting alternatives that cater to coding requirements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/cat-stare-catstare-cat-stare-sus-catglare-cat-glare-gif-14942558849944709546">Cat Stare Catstare GIF - Cat stare Catstare Cat stare sus - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/430">Markdown input is rendered instead of displayed as raw text in chat · Issue #430 · lmstudio-ai/lmstudio-bug-tracker</a>: Which version of LM Studio? LM Studio 0.3.9 (Build 6) Which operating system? Windows 11 What is the bug? When users input text in markdown format (e.g., # Heading, italic, bold), it gets rendered ...</li><li><a href="https://v0.dev">v0 by Vercel</a>: Chat with v0. Generate UI with simple text prompts. Copy, paste, ship.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1338963101608968323)** (175 messages🔥🔥): 

> `GPU Comparison: 3080 Ti vs 3090 vs 4090 vs 5090, Importance of PCI-E Bandwidth in Gaming and Inference, AMD vs NVIDIA: Current Performance and Recommendations, Potential Issues with 5090 GPU, Building a Multi-GPU AI Setup` 


- **Testing GPU Performance: 3090, 4090, 5090**: A user is testing a setup with a **3090**, **4090**, and **5090**, seeking recommendations for models to run locally.
   - Concerns about the **5090** have surfaced, particularly with reports of it malfunctioning, raising questions about its reliability.
- **Gaming vs Inference Performance Variability**: Discussion highlights that in gaming, when textures are loaded, the performance is similar to inference as it's just math calculations on the GPU.
   - It was noted that with lower-end GPUs like the **1050 Ti**, performance suffers significantly when bandwidth is limited.
- **AMD and NVIDIA GPU Recommendations**: Opinions were shared on the viability of AMD GPUs for AI tasks, noting their **24 PCI-E lines** which may restrict multi-GPU setups.
   - Users discussed the possibility of using an **AMD Threadripper** for better performance due to more available lanes.
- **Concerns Regarding 5090 GPU Reliability**: Amplified concerns arose about the reliability of the **5090**, with references to users frying their cards leading to cautious behavior.
   - To address concerns, some suggested undervolting the 5090 as a precautionary measure.
- **Building a Multi-GPU AI Setup**: A user shared their experience building a server with multiple GPUs, mentioning specific board configurations to optimize performance.
   - Discussion included various setups where x1 links were used due to board limitations, challenging typical expectations of GPU performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://videocardz.com/newz/amd-reportedly-working-on-gaming-radeon-rx-9000-gpu-with-32gb-memory">AMD reportedly working on gaming Radeon RX 9070 XT GPU with 32GB memory - VideoCardz.com</a>: AMD may be working on Radeon RX 9070 XT with 32GB memory There&#8217;s a new rumor from Chiphell about an alleged Radeon RX 9000 card equipped with twice the memory capacity compared to the RX 9070 se...</li><li><a href="https://videocardz.com/newz/amd-reportedly-working-on-gamin">AMD reportedly working on gaming Radeon RX 9070 XT GPU with 32GB memory - VideoCardz.com</a>: AMD may be working on Radeon RX 9070 XT with 32GB memory There&#8217;s a new rumor from Chiphell about an alleged Radeon RX 9000 card equipped with twice the memory capacity compared to the RX 9070 se...</li><li><a href="https://www.techpowerup.com/gpu-specs/tesla-m10.c3035">NVIDIA Tesla M10 Specs</a>: NVIDIA GM107 x4, 1306 MHz, 640 Cores, 40 TMUs, 16 ROPs, 8192 MB GDDR5, 1300 MHz, 128 bit</li><li><a href="https://www.youtube.com/watch?v=L1NPFFRTzLo">NVIDIA RTX 5090 PCIe 5.0 vs. 4.0 vs. 3.0 x16 Scaling Benchmarks</a>: Sponsor: Arctic Liquid Freezer III on Amazon - https://geni.us/NrMtDTThis benchmark compares PCIe generation differences on the NVIDIA RTX 5090 GPU. We&#39;re te...</li><li><a href="https://youtu.be/COcHHX2MdKs"> - YouTube</a>: no description found</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-ti-super.c4187">NVIDIA GeForce RTX 4070 Ti SUPER Specs</a>: NVIDIA AD103, 2610 MHz, 8448 Cores, 264 TMUs, 96 ROPs, 16384 MB GDDR6X, 1313 MHz, 256 bit</li><li><a href="https://github.com/Nicoolodion/RTX-3070-16GB-GUIDE">GitHub - Nicoolodion/RTX-3070-16GB-GUIDE: A Guide for Modding a RTX 3070 to 16 GB VRAM</a>: A Guide for Modding a RTX 3070 to 16 GB VRAM. Contribute to Nicoolodion/RTX-3070-16GB-GUIDE development by creating an account on GitHub.</li><li><a href="https://youtu.be/kb5YzMoVQyw">How Nvidia made the 12VHPWR connector even worse.</a>: Der8auer&#39;s video: https://www.youtube.com/watch?v=Ndmoi1s0ZaYPatreon: https://www.patreon.com/buildzoidTwitch(mainly gaming streams): https://www.twitch.tv/b...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1338990315318677595)** (80 messages🔥🔥): 

> `Thomson Reuters AI copyright case, Current AI fundraising, OpenAI's GPT-4.5 and GPT-5 roadmap, GRPO training improvements, DeepSeek-R1 and reasoning models` 


- **Thomson Reuters wins landmark AI copyright case**: Thomson Reuters has [won](https://storage.courtlistener.com/recap/gov.uscourts.ded.72109/gov.uscourts.ded.72109.770.0.pdf) the first major AI copyright case in the U.S., ruling in favor of the company's claim against Ross Intelligence for reproducing materials from Westlaw.
   - Judge Stephanos Bibas stated that *none of Ross’s possible defenses holds water* and dismissed them all.
- **Current AI launches with major fundraising**: [Current AI](https://www.currentai.org/) is poised to lead in public interest AI with a starting pledge of $400 million and a goal to raise $2.5 billion over five years to shape AI towards societal benefits.
   - Many are excited about the potential to create opportunities and security through AI, with a vision that includes community involvement from diverse locations like Lagos to Lima.
- **OpenAI unveils roadmap for GPT-4.5 and GPT-5**: OpenAI's roadmap details the upcoming release of GPT-4.5, the last non-chain-of-thought model, followed by the integration of GPT-5 that aims to unify their product offerings.
   - Free-tier users will gain unlimited access to GPT-5, while paid subscribers will have enhanced capabilities, including features like voice and deep research.
- **GRPO training significantly boosts performance**: Transitioning from PPO to GRPO has resulted in a 4x increase in performance gains for the Tulu pipeline, showing considerable improvements in challenges like MATH and GSM8K.
   - Costa Huang shared the success of their latest GRPO-trained Tulu model, indicating a new direction for RL strategies.
- **DeepSeek-R1 enhances reasoning models**: DeepSeek-R1 has brought about a wave of interest among companies looking to implement reasoning models effectively in production environments.
   - Together Compute has announced dedicated reasoning clusters powered by advanced chips to support large-scale and low-latency AI workloads.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06807">Competitive Programming with Large Reasoning Models</a>: We show that reinforcement learning applied to large language models (LLMs) significantly boosts performance on complex coding and reasoning tasks. Additionally, we compare two general-purpose reasoni...</li><li><a href="https://www.bomberbot.com/debugging/how-to-debug-your-code-like-a-competitive-programmer-automate-and-save-time/#:~:text=Start%20with%20a%20brute%20force,The).">How to Debug Your Code Like a Competitive Programmer – Automate and Save Time - Bomberbot</a>: As programmers, we spend a significant portion of our time debugging code. A study by the University of Cambridge found that software developers spend 50% of</li><li><a href="https://x.com/togethercompute/status/1889743684977168547">Tweet from Together AI (@togethercompute)</a>: Since launching DeepSeek-R1, we&#39;ve seen a wave of companies looking to deploy reasoning models in production—but scaling them efficiently remains a challenge.Today, we’re expanding beyond our ultr...</li><li><a href="https://x.com/natolambert/status/1889730488199209393">Tweet from Nathan Lambert (@natolambert)</a>: Costa&#39;s just trying to make GRPO go brrr with no bugs and we&#39;re ending up with way better performance than the Tülu models we released in the fall. Changing from PPO -&gt; GRPO 4x&#39;d the ga...</li><li><a href="https://x.com/TheXeophon/status/1889762840384266578">Tweet from Xeophon (@TheXeophon)</a>: with GPT-5 being an (even more) black-box system, i hope academia finally moves on from being paying product testers to using open models exclusively</li><li><a href="https://x.com/Dorialexander/status/1889300494989869464">Tweet from Alexander Doria (@Dorialexander)</a>: Common Corpus 2 is an in-kind contribution to CurrentAI, the new foundation for open source ai that just launched during the #AISummit. With support from the AI Alliance and institutional actors we co...</li><li><a href="https://x.com/lmarena_ai/status/1889741530757210524">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: What types of programming languages are people asking about?Python and markdown are by far the most common languages people’s questions are related to, according to our retrieved file types.</li><li><a href="https://x.com/stablequan/status/1889560991882416294">Tweet from qnguyen3 (@stablequan)</a>: Apple Intelligence is going to be powered by @Alibaba_Qwen, in China. Big W.</li><li><a href="https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/">Thomson Reuters Wins First Major AI Copyright Case in the US</a>: The Thomson Reuters decision has big implications for the battle between generative AI companies and rights holders.</li><li><a href="https://x.com/replicate/status/1889628772997243034">Tweet from Replicate (@replicate)</a>: Hello Claude!https://replicate.com/anthropicClaude 3.5 Sonnet and Claude 3.5 Haiku models are now available on Replicate</li><li><a href="https://x.com/nrehiew_/status/1889737259835969735">Tweet from wh (@nrehiew_)</a>: On the same TULU3 dataset, GRPO &gt; PPO. What’s the intuition here? is GRPO just the mandate of heaven RL algo?Quoting Costa Huang (@vwxyzjn) 🔥 allenai/Llama-3.1-Tulu-3-8B (trained with PPO) -&gt; a...</li><li><a href="https://www.currentai.org/">Current AI | Building Public Interest AI Technology Together</a>: Join a global initiative building open, fair AI technology that serves the public interest. Through collaboration and local action, we&#x27;re creating AI solutions that benefit everyone.</li><li><a href="https://x.com/NeginRaoof_/status/1889739171826377008">Tweet from Negin Raoof (@NeginRaoof_)</a>: Announcing OpenThinker-32B: the best open-data reasoning model distilled from DeepSeek-R1.Our results show that large, carefully curated datasets with verified R1 annotations produce SoTA reasoning mo...</li><li><a href="https://x.com/sama/status/1889755723078443244?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Sam Altman (@sama)</a>: OPENAI ROADMAP UPDATE FOR GPT-4.5 and GPT-5:We want to do a better job of sharing our intended roadmap, and a much better job simplifying our product offerings.We want AI to “just work” for you; we re...</li><li><a href="https://biz.chosun.com/stock/stock_general/2025/02/12/KAL6SZYMQ5DLTEMGKYSTTIXT5I/">자체 칩 개발 실패한 메타... 퓨리오사AI M&amp;A 성사될까</a>: 자체 칩 개발 실패한 메타... 퓨리오사AI M&amp;A 성사될까 메타, 엔비디아 의존도 낮추려 대안 모색 메타, 자체 칩 개발 실패로 인수 유인 충분 최근 투자 유치 기업가치 8000억원 기준점</li><li><a href="https://techcrunch.com/2025/02/10/google-backed-public-interest-ai-partnership-launches-with-400m-pledged-for-open-ecosystem-building/?guccounter=1">Google-backed public interest AI partnership launches with $400M+ for open ecosystem building | TechCrunch</a>: Make room for yet another partnership on AI. Current AI, a &quot;public interest&quot; initiative focused on fostering and steering development of artificial
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1339076985460363356)** (24 messages🔥): 

> `Notebook LM alternatives, Long context models, Elicit.com and PaperQA, Claude performance, Gemini capabilities` 


- **Searching for Better PDF Chat Tools than Notebook LM**: Users discussed the limitations of **Notebook LM** for chatting with multiple PDFs and queried if alternatives like **Glean** or **Claude** could meet their needs.
   - *Claude's performance* was questioned, especially with large sets of PDFs, leading to suggestions for using tools that can handle long contexts effectively.
- **Challenges with Long Context Models**: Concerns were raised about **Claude/r1** and its rapid performance decline when interacting with 5-6 PDFs, questioning its robustness for extensive document engagement.
   - References to a study on long context evaluation highlighted that existing models struggled despite improvements in handling larger contexts.
- **Exploring Elicit.com and PaperQA Options**: Users shared mixed experiences with **Elicit.com**, expressing dissatisfaction with previous outcomes but acknowledged its potential for certain use cases.
   - Alternatively, **PaperQA** was mentioned as a fast-moving open-source project with support from Eric Schmidt, suggesting a more promising option for document-based queries.
- **Internal Tool Development for Document Interaction**: One user is developing a custom tool to facilitate document queries through the **OAI Assistants API**, emphasizing the need for user-selectable sources.
   - Despite the tool's complexity, there was uncertainty regarding its long-term utility and effectiveness.
- **Mixed Reviews on Elicit's Effectiveness**: Past users of **Elicit** expressed frustration with its accuracy and reliability but noted that the platform remains active.
   - A consensus emerged that tools like Elicit need improvements to be genuinely useful for AI researchers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05167">NoLiMa: Long-Context Evaluation Beyond Literal Matching</a>: Recent large language models (LLMs) support long contexts ranging from 128K to 1M tokens. A popular method for evaluating these capabilities is the needle-in-a-haystack (NIAH) test, which involves ret...</li><li><a href="https://github.com/Future-House/paper-qa">GitHub - Future-House/paper-qa: High accuracy RAG for answering questions from scientific documents with citations</a>: High accuracy RAG for answering questions from scientific documents with citations - Future-House/paper-qa
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1339111478262894602)** (21 messages🔥): 

> `Grok 3 Controversy, Resignation from xAI, Grok 3 Performance Speculations, Free Speech Concerns at xAI, DeepSeek Comparisons` 


- **Grok 3 Controversy Leads to Resignation**: A resigning employee stated he was forced to delete a tweet acknowledging the existence of **Grok 3**, which xAI classified as confidential, leading to his decision to leave the company. He expressed disappointment that such an obvious opinion could threaten his job, asserting it seems contrary to the company's claimed support for free speech.
   - *It's absurd that a post labeled as 'opinion' on Grok 3 could be seen as grounds for dismissal.*
- **xAI's Reaction Sparks Discussion**: Members discussed the peculiar decision by xAI to enforce silence on Grok 3's existence, leading to interpretations of how this might reflect internal pressures on free expression. Some speculated if the employee's remarks on unreleased product performance may have influenced the push for his resignation.
   - *They noted that the trending sentiment seems to be that xAI's stance is a contradiction to the free speech advocacy they profess.*
- **Comparisons to DeepSeek**: In response to Grok 3's anticipated performance, some voiced skepticism, suggesting that if Grok 3 barely surpasses **DeepSeek**'s capabilities, it would be deemed disappointing considering the massive compute resources available. One member reminded that *DeepSeek operates on much lower-end hardware*.
   - *Speculation abounds about Grok 3 being derived from a **DeepSeek** distillation or fine-tune, raising eyebrows among researchers.*
- **Future of AI Models Speculation**: Some members speculate that the rumored performance of Grok 3 and **Llama 4** could lead to renewed faith in models from OpenAI, Anthropic, and Gemini, possibly revealing hidden advantages of research over sheer GPU power. This discussion points to the competitive dynamics in AI model development and evaluation.
   - *There is caution that researchers may hold valuable innovations that outpace mere hardware specifications.*
- **Mixed Reactions on Information Sharing**: Amid the drama, one participant shared a preference for being added to **blocklists** as a more comprehensive method of managing dissension, highlighting the polarized environment within the discourse. The sentiment mirrors the broader themes of how information and opinions are managed in contentious environments.
   - *Participants acknowledge that discord around opinions, especially in the context of competitive AI landscapes, creates a complex web of communication challenges.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/BenjaminDEKR/status/1888331926630510638">Tweet from Benjamin De Kraker (@BenjaminDEKR)</a>: The ranking currently (my opinion), for code:ChatGPT o1-proo1o3-mini(all kind of tied)Grok 3 (expected, tbd)Claude 3.5 SonnetDeepSeekGPT-4oGrok 2Gemini 2.0 Pro Series (might be higher, will probably m...</li><li><a href="https://x.com/vikhyatk/status/1889535819997725008">Tweet from vik (@vikhyatk)</a>: Quoting Benjamin De Kraker (@BenjaminDEKR) I resigned from xAI tonight. It makes me very sad, but was the right thing to do -- and here&#39;s why.xAI told me I either had to delete the post quoted bel...</li><li><a href="https://fxtwitter.com/BenjaminDEKR/status/1889526713735905502">Tweet from Benjamin De Kraker (@BenjaminDEKR)</a>: I resigned from xAI tonight. It makes me very sad, but was the right thing to do -- and here&#39;s why.xAI told me I either had to delete the post quoted below, or face being fired. After reviewing ev...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1339049850876199026)** (78 messages🔥🔥): 

> `Voiceover Techniques, Claude Reasoning Model, OpenThinker-32B Release, Nuclear Physicists Comparison, RL Roundup Challenges` 


- **Improving Voiceover Breathing Techniques**: Members exchanged advice on reducing loud breathing during voiceover recordings, with suggestions including OBS settings and maintaining a natural speaking style.
   - One member noted they had found noise reduction features in Riverside to help improve audio quality.
- **Claude's Extended Thinking Mode**: Discussion around Claude's prompt response times highlighted its potential new 'thinking' indicator, observed during recent prompts.
   - Members speculated on whether this was an indication of a new reasoning model, with some dissent regarding changes to the underlying system.
- **Release of OpenThinker-32B Model**: The launch of the **OpenThinker-32B** reasoning model was discussed, revealing it could achieve good performance while being less censored.
   - Team members indicated this development stemmed from community concerns about model censorship and the desire for less restricted outputs.
- **Nuclear Physicists Parallel**: A member drew comparisons between modern ML scientists and early nuclear physicists, citing historical idealism about technology's positive impact.
   - They reflected on the shift in perspective following significant geopolitical developments, paralleling modern apprehensions about AI.
- **Challenges in RL Roundups**: Concerns were raised about the complexity of compiling RL results, with members discussing the overwhelming nature of available data.
   - One member expressed that they planned to provide commentary on the topic rather than a detailed roundup due to the extensive effort required.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://neurips.cc/virtual/2024/tutorial/99526">NeurIPS Tutorial Opening the Language Model Pipeline: A Tutorial on Data Preparation, Model Training, and Adaptation</a>: no description found</li><li><a href="https://x.com/__nmca__/status/1889741584922751092">Tweet from Nat McAleese (@__nmca__)</a>: @stalkermustang @ahelkky o3 samples many solutions and uses a learned function to pick the best --- for codeforces, we sampled 1,162 samples per problem</li><li><a href="https://fxtwitter.com/kernelkook/status/1889678407346106418">Tweet from sanchay (@kernelkook)</a>: i think this is gonna come pretty soon.gave claude a prompt recently and noticed it displayed &#34;thinking...&#34; for like 7-8 secs before responding, which isn&#39;t the usual behavior.couldn&#39;t...</li><li><a href="https://tenor.com/view/avatar-aang-aang-atla-avatar-the-last-airbender-avatar-gif-23087281">Avatar Aang Aang GIF - Avatar Aang Aang Atla - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/naklecha/status/1889662581180183039">Tweet from naklecha (@naklecha)</a>: i have run 100s of training runs on grpo and sft+grpo combinations in the past 2 weeks. by far the coolest reward hacking example i found is — when i penalize high confidence logprobs of wrong tokens ...</li><li><a href="https://x.com/madiator/status/1889772019492987225">Tweet from Mahesh Sathiamoorthy (@madiator)</a>: We accidentally de-censored the model!Qwen-instruct which we use is censored and aligned.DeepSeek-R1 distilled models are censored and aligned.When we SFT the Qwen model with reasoning data in math an...</li><li><a href="https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use">How much energy does ChatGPT use?</a>: This Gradient Updates issue explores how much energy ChatGPT uses per query, revealing it’s 10x less than common estimates.</li><li><a href="https://youtu.be/64E9O1Gv99o?si=Bi5YLxNkYKOt8bRa&t=575)"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/qPzZeP7t5ZQ?si=czaITQnSdRyA6tCi">Language Modeling: A Tutorial on Data Preparation, Model Training, and Adaptation</a>: Opening the Language Model Pipeline: A Tutorial on Data Preparation, Model Training, and AdaptationKyle Lo · Akshita Bhagia · Nathan LambertIf you would like...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1339065561585422367)** (5 messages): 

> `Goldman Sachs AI Interview Controversy, Deepseek Model Insights, OpenAI's Strategy Shift` 


- **Goldman Sachs Circulates AI Generated Interview**: [Kevin Xu](https://x.com/kevinsxu/status/1889504794420646039) highlighted that Goldman Sachs shared what they claimed to be a 'new' interview with Liang Wenfeng, but it was actually AI-generated audio from his July 2024 interview.
   - *If you are a GS client, stop paying them money.* Instead, Xu recommends subscribing to newsletters from sources like [Jordan Schnitzer](https://x.com/jordanschnyc) or [Nat Lambert](https://x.com/natolambert) for real insights.
- **Deepseek Model Mentioned in Master Thesis**: A member noted that they included the first **Deepseek** model in their Master Thesis and humorously suggested they should join Goldman Sachs.
   - This comment indicates that Deepseek is seen as significant within academic circles.
- **Anticipation Builds for Deepseek Announcement**: [AK](https://x.com/_akhaliq) teased that something substantial about **Deepseek** is set to be revealed tomorrow, building excitement among members.
   - This hints at upcoming developments that could impact the field significantly.
- **OpenAI's Strategy Under Scrutiny**: [Sam Altman](https://x.com/sama) announced a shift in OpenAI's approach, acknowledging that simply scaling model size and resources is no longer effective for achieving AGI/ASI.
   - With the upcoming release of **GPT-4.5** and **GPT-5**, they aim to simplify offerings and unify their models for broader application, moving away from the complex model picker.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/kevinsxu/status/1889504794420646039">Tweet from Kevin Xu (@kevinsxu)</a>: Just heard that &#34;the Goldman Sachs&#34; circulated a note to their buyside clients earlier this week with a &#34;new&#34; interview Liang Wenfeng did post R1 release to show they are in the knowEx...</li><li><a href="https://x.com/stanfordnlp/status/1889768783834976431">Tweet from Stanford NLP Group (@stanfordnlp)</a>: The final admission that the 2023 strategy of OpenAI, Anthropic, etc. (“simply scaling up model size, data, compute, and dollars spent will get us to AGI/ASI”) is no longer working!Quoting Sam Altman ...</li><li><a href="https://x.com/untitled01ipynb/status/1889751694365388821">Tweet from loss (@untitled01ipynb)</a>: what did ak seeQuoting AK (@_akhaliq) Something big coming out Tomorrow about deepseek
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1339259588247289979)** (18 messages🔥): 

> `Invention of Science, Dwarkesh Podcast with Jeff Dean and Noam Shazeer, Metascience and History of Science Discussions` 


- **Readers show interest in 'Invention of Science'**: A member shared a [link to 'The Invention of Science' by David Wootton](https://www.inventionofscience.com/) published on September 17, 2015, available in various formats.
   - Enthusiasts of metascience expressed their love for the genre while others noted it for their reading lists.
- **Iconic moments on Dwarkesh Podcast**: A discussion sparked around a recent episode featuring **Jeff Dean** and **Noam Shazeer** that highlights their impact on modern computing and LLMs, shared via [this Substack link](https://open.substack.com/pub/dwarkesh/p/jeff-dean-and-noam-shazeer?r=68gy5&utm_medium=ios).
   - Listeners expressed excitement over Dean's insights on Google's future and Shazeer's bold claims regarding global GDP, noting the episode's significance in the tech community.
- **Interest in future podcast appearances**: Multiple members encouraged one person to go on Dwarkesh, suggesting it would be a fun experience, albeit not a priority at the moment.
   - There were playful suggestions for inviting **Xeophon** on Dwarkesh, with promises of fun discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://open.substack.com/pub/dwarkesh/p/jeff-dean-and-noam-shazeer?r=68gy5&utm_medium=ios">Jeff Dean &amp; Noam Shazeer – 25 years at Google: from PageRank to AGI</a>: Two of Gemini&#x27;s co-leads on Google&#x27;s path to AGI</li><li><a href="https://www.inventionofscience.com/">The Invention of Science | by David Wootton</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1339223904870465609)** (13 messages🔥): 

> `Naming Schemes for LLMs, Philosophical Views on Kuhn's Book, Future of LLMs vs LRMs` 


- **Incongruent Naming Schemes in RLMs**: The discussion revealed confusion over naming conventions, with **OpenAI** referring to them as **LRMs**, while others simply label them as **LLMs**.
   - A participant pointed out that having an 'L' prefix implies 'large,' but that term is unnecessary.
- **Philosophers Challenging Kuhn**: There are emerging opinions from philosophers suggesting that **Kuhn's book** is outdated, with little elaboration on the claim.
   - This sentiment seems to lack substantial discourse with members calling for clarity and depth of understanding.
- **Skepticism Towards the Future of LLMs**: One member expressed a **'not-so-hot'** take, asserting that **LLMs** are doomed and only **LRMs** will prevail.
   - This view was echoed by others, indicating that all current LLMs could eventually be seen as variants of LRMs.


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1338976597629206569)** (7 messages): 

> `Shitty slides, Instructor concerns, Course content, Immortalization of students` 


- **Students immortalized on slides**: A member humorously noted, *"Immortalized on my shitty slides"* with an image attached showcasing perhaps less-than-ideal content.
   - Another member replied with *"ohno lol"*, indicating a shared sentiment of laughter or concern about the quality.
- **Concerns about student experience**: One member expressed worry for the students by stating, *"These poor students (what am I missing 🥹)"* suggesting a feeling of inadequacy.
   - Amidst this concern, a member humorously commented, *"Dos hombres fuerte y rico"*, ensuring some levity in the conversation.
- **Confusion over course content**: Another member, reflecting confusion, asked, *"wtf did I do wrong"*, perhaps indicating frustration with the course delivery or content.
   - A member responded that it’s either an improvement in the current course or a potential return to *"another course on parse trees"*, hinting at unsatisfactory alternatives.


  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1339214661098410077)** (8 messages🔥): 

> `Deep Research and O1 Pro Combo, ChatGPT UX Feedback, RL Question Misinformation, Switching Models in ODR Chat` 


- **Deep Research and O1 Pro make a powerful combo**: A member tried a strategy of starting with **deep research**, followed by using **O1 Pro** for follow-ups, allowing for rich context in responses.
   - They encountered a misunderstanding when asking a reinforcement learning question but resolved it by searching beforehand.
- **Users express concerns about ChatGPT UX**: A member criticized the **user experience (UX)** of switching models, noting that starting with **O1 Pro** could inadvertently lead to unnecessary additional clicks.
   - There was a consensus that the functionality seems janky, as some members mentioned being able to switch models but found it less intuitive.
- **ChatGPT's janky functionalities draw comments**: Users described **ChatGPT** as 'janky', highlighting frustrations with model switching and operation.
   - The conversation reflects broader frustrations towards the UI and operational mechanics of the platform.
- **Confusion over Reinforcement Learning acronyms**: One user struggled with an RL acronym, initially interpreting **GRPO** as **Gaussian Regularized Proximal Policy Optimization** without context.
   - They acknowledged that thorough research beforehand corrected the misunderstanding, emphasizing the importance of context.
- **Sharing chats reveals hidden features**: A member mentioned sharing their chat led to successfully switching to **O1 Pro** in **ODR chat**, suggesting collaborative troubleshooting.
   - This highlights how sharing experiences can illuminate overlooked functionalities in the platform.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1339001334665580587)** (22 messages🔥): 

> `Deep Learning Loss Issues, Deepfrying Phenomenon, Pythia's Safety Training, Dataset Perplexity Evaluation, Polyglot Project Interest` 


- **Navigating Deep Learning Loss Dilemmas**: A user expressed frustration over experiencing **wild and increasing loss** in a **72B model** compared to smaller ones, suspecting high learning rates might not be the only issue.
   - *Another user suggested that the length of sequences used during training could directly impact performance*.
- **Understanding Deepfrying in Model Training**: The conversation revealed that **deepfrying** refers to a state where a model experiences progressively increasing variance leading to elevated loss spikes.
   - *Key contributors noted that high learning rates and short sequence lengths could exacerbate this issue*.
- **Pythia's Safety Training Inquiry**: A member inquired about **Pythia's** ability to resist adversarial attacks, specifically if it is safety trained against jailbreaks.
   - Another participant confirmed that *Pythia is not safety trained at all*.
- **Dataset Perplexity Evaluation Efficiency**: A user sought recommendations for an **efficient implementation** of **dataset perplexity evaluation**.
   - Someone else prompted for clarification on what aspects of efficiency were being targeted in the request.
- **Interest in Polyglot Project**: A newcomer joined and expressed interest in **multilinguality**, **language learning**, and specifically in EleutherAI's **polyglot project**.
   - They conveyed eagerness to learn and absorb knowledge within the community.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1338964146095390842)** (198 messages🔥🔥): 

> `Long-Term Memory models, Memory in Transformers, Automated Capability Discovery, Reinforcement Learning via Self-Play, Recursive Inference Scaling` 


- **Exploration of Long-Term Memory models for ultra-long contexts**: Recent updates from Magic introduced Long-Term Memory models that can handle contexts up to **100M tokens**, enhancing the reasoning capabilities beyond traditional training methods.
   - This advancement opens up significant opportunities in software development by integrating extensive codebases and documentation into the context for model training.
- **Questioning the implementation of memory slots in LM2**: There were concerns about the transparency of the memory slot implementation in the LM2 model, where it was noted that the authors did not clearly describe how memory slots are chosen or updated in their architecture.
   - Participants expressed skepticism over the effectiveness and parallelizability of the design, suggesting it might be oversimplified in the paper.
- **Introducing Automated Capability Discovery (ACD)**: A new framework called Automated Capability Discovery (ACD) aims to self-explore model capabilities in a systematic way, identifying unexpected abilities and weaknesses in foundation models.
   - ACD operates by designating one foundation model as a 'scientist' to propose tasks for other models, enhancing evaluation accuracy with less human effort.
- **Reinforcement Learning via Self-Play proposed for LRMs**: A proposed framework, Reinforcement Learning via Self-Play (RLSP), focuses on training Large Reasoning Models by decoupling exploration and correctness signals during reinforcement learning.
   - This method involves fine-tuning via demonstrations followed by RL training bolstered by an exploration reward signal, targeting efficient reasoning behaviors without rewarding exploitation.
- **Revisiting Recursive Inference Scaling**: Recursive Inference Scaling (RINS) builds on scaling inference time for language models, enhancing performance through sophisticated inference methods inspired by fractal geometry.
   - Discussion raised concerns about the novelty of RINS's approach and its implications for existing models, questioning whether it introduced significant advancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06807">Competitive Programming with Large Reasoning Models</a>: We show that reinforcement learning applied to large language models (LLMs) significantly boosts performance on complex coding and reasoning tasks. Additionally, we compare two general-purpose reasoni...</li><li><a href="https://arxiv.org/abs/2410.02416">Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models</a>: Classifier-free guidance (CFG) is crucial for improving both generation quality and alignment between the input condition and final output in diffusion models. While a high guidance scale is generally...</li><li><a href="https://arxiv.org/abs/2502.02996">Building Bridges between Regression, Clustering, and Classification</a>: Regression, the task of predicting a continuous scalar target y based on some features x is one of the most fundamental tasks in machine learning and statistics. It has been observed and theoretically...</li><li><a href="https://arxiv.org/abs/2502.07503">Harnessing Language&#39;s Fractal Geometry with Recursive Inference Scaling</a>: Recent research in language modeling reveals two scaling effects: the well-known improvement from increased training compute, and a lesser-known boost from applying more sophisticated or computational...</li><li><a href="https://arxiv.org/abs/2502.06049">LM2: Large Memory Models</a>: This paper introduces the Large Memory Model (LM2), a decoder-only Transformer architecture enhanced with an auxiliary memory module that aims to address the limitations of standard Transformers in mu...</li><li><a href="https://arxiv.org/abs/2501.15420">Visual Generation Without Guidance</a>: Classifier-Free Guidance (CFG) has been a default technique in various visual generative models, yet it requires inference from both conditional and unconditional models during sampling. We propose to...</li><li><a href="https://arxiv.org/abs/2408.00677">Scaling Backwards: Minimal Synthetic Pre-training?</a>: Pre-training and transfer learning are an important building block of current computer vision systems. While pre-training is usually performed on large real-world image datasets, in this paper we ask ...</li><li><a href="https://arxiv.org/abs/2502.06772">ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates</a>: We present that hierarchical LLM reasoning via scaling thought templates can effectively optimize the reasoning search space and outperform the mathematical reasoning capabilities of powerful LLMs lik...</li><li><a href="https://arxiv.org/abs/2502.06773">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a>: Recent AI advancements, such as OpenAI&#39;s new models, are transforming LLMs into LRMs (Large Reasoning Models) that perform reasoning during inference, taking extra time and compute for higher-qual...</li><li><a href="https://arxiv.org/abs/2502.07527">NatureLM: Deciphering the Language of Nature for Scientific Discovery</a>: Foundation models have revolutionized natural language processing and artificial intelligence, significantly enhancing how machines comprehend and generate human languages. Inspired by the success of ...</li><li><a href="https://arxiv.org/abs/2211.04800">Designing Network Design Strategies Through Gradient Path Analysis</a>: Designing a high-efficiency and high-quality expressive network architecture has always been the most important research topic in the field of deep learning. Most of today&#39;s network design strateg...</li><li><a href="https://arxiv.org/abs/2402.13616">YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information</a>: Today&#39;s deep learning methods focus on how to design the most appropriate objective functions so that the prediction results of the model can be closest to the ground truth. Meanwhile, an appropri...</li><li><a href="https://x.com/jeffclune/status/1889568685632667672">Tweet from Jeff Clune (@jeffclune)</a>: Introducing Automated Capability Discovery!ACD automatically identifies surprising new capabilities and failure modes in foundation models, via &#34;self-exploration&#34; (models exploring their own a...</li><li><a href="https://arxiv.org/abs/2502.07577">Automated Capability Discovery via Model Self-Exploration</a>: Foundation models have become general-purpose assistants, exhibiting diverse capabilities across numerous domains through training on web-scale data. It remains challenging to precisely characterize e...</li><li><a href="https://arxiv.org/html/2501.01257v2">CodeForces: Benchmarking Competition-level Code Generation of LLMs on CodeForces Disclaimer: This is a non-traditional code benchmark.</a>: no description found</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows — Magic</a>: Research update on ultra-long context models, our partnership with Google Cloud, and new funding.</li><li><a href="https://github.com/SmerkyG/gptcore/blob/main/model/experimental/memtention.py">gptcore/model/experimental/memtention.py at main · SmerkyG/gptcore</a>: Fast modular code to create and train cutting edge LLMs - SmerkyG/gptcore
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1339264670024077443)** (4 messages): 

> `Fine-tuning and recognition, Testable hypotheses in AI` 


- **Collaboration on Current Work**: A member expressed that the ongoing discussion aligns closely with their current project, suggesting active involvement in the topic.
   - Another member showed enthusiasm, stating they were 'happy to hear that!', indicating community support.
- **Fine-tuning with Mnemonic Patterns**: A member inquired if the current work relates to fine-tuning methods involving mnemonic strings, specifically how the model could 'recognize' patterns such as those spelling out 'HELLO'.
   - They mentioned having a 'testable hypothesis in that regard', signaling a potential for further experimental exploration.


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1338976640218300528)** (167 messages🔥🔥): 

> `Cursor Documentation Updates, MCP Server Issues, O3-Mini Performance, Pricing Models for AI Services, Claude Model Performance` 


- **Cursor Updates on Pricing and Models**: Cursor has updated their documentation, specifying usage-based pricing and the details of models available, including which are free and which are not.
   - Members noted that models such as **deepseek R1** and **O3-mini** are now recognized in the pricing structure, causing some confusion about their premium status.
- **Challenges with MCP Server Integrations**: Users reported issues with MCP server integrations, particularly with the **Perplexity API**, leading to inconsistencies and errors during usage.
   - Several users were able to resolve their issues, suggesting troubleshooting steps such as hardcoding API keys and removing conflicting packages.
- **O3-Mini's Inconsistent Performance**: Concerns were raised about the inconsistent performance of **O3-mini**, with users experiencing both successful and hallucinated outputs depending on context.
   - The ongoing discussion indicates that while O3-mini can occasionally provide impressive improvements, inconsistencies remain a significant point of frustration.
- **Pricing Comparisons for AI Models**: Discussion circulated around the affordability of using various AI models, particularly the cost-effectiveness of **MCP Perplexity** compared to others like **Claude** and **O3-mini**.
   - Members shared their experiences, noting that recent usage of these models only incurs minimal costs when token usage is managed effectively, especially during long interactions.
- **Excitement for Future Anthropic Models**: Anticipation grew for upcoming releases of **Anthropic** models, with users expressing their thoughts on how current models like **Claude Sonnet** handle various tasks effectively.
   - The community seems eager for improvements, particularly related to the features and capabilities promised by future iterations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://viggle.ai/)">Viggle AI | Controllable AI Video Generator</a>: Bring your characters to life with AI. From professional motion capture to viral memes, discover endless ways to create with Viggle.</li><li><a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: no description found</li><li><a href="https://docs.cursor.com/account/usage#usage-based-pricing">Cursor – Usage</a>: no description found</li><li><a href="https://half-single-ecd.notion.site/Experiment-Prompting-86aa8f988fce404cbf70134690d2635a?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://fireworks.ai/models/fireworks/deepseek-r1">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1inoi6b/openai_silently_rolls_out_o1_o3mini_and_o3mini/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=RCFe1L9qm3EI">Cursor + MCP Servers: Complete Setup Guide (Sequential Thinking, Brave Search, &amp; More)</a>: Cursor just added MCP support! In this complete setup guide, I&#39;ll show you how to integrate and use MCP servers (Sequential Thinking, Brave Search, and Puppe...</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://status.cursor.com/">Cursor Status</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1339057086746857492)** (2 messages): 

> `NVIDIA GB200 images, Discord server purpose` 


- **Inquiry about NVIDIA GB200 images**: A member asked if this is a server dedicated to **lewd NVIDIA GB200 images**.
   - Another member confirmed, saying, '*Yes*'.
- **Server Confirmation**: The inquiry about server content regarding NVIDIA GB200 images was swiftly confirmed by another member.
   - This exchange highlights the community's openness to such discussions.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1339174631730581596)** (1 messages): 

> `Error in default mode vs INTERPRET mode, Triton kernel comparison, Matrix multiplication differences` 


- **Error Comparison: Default vs INTERPRET Mode**: A member raised a concern regarding why the **error** is significantly larger in **default mode** compared to **INTERPRET mode** during simple 2D matrix multiplication.
   - They referenced a [GitHub issue](https://github.com/triton-lang/triton/issues/5895) discussing specific differences, noting that in INTERPRET mode the error was as low as **9.5367431640625e-07**.
- **Comparison of Triton Kernel and Torch**: The discussion highlighted differences between their Triton kernel and **Torch** when executed in default and INTERPRET modes, raising questions about performance metrics.
   - More insights can be found in the linked GitHub issue, which details the discrepancies noted during execution.



**Link mentioned**: <a href="https://github.com/triton-lang/triton/issues/5895">error is significantly larger in default mode than INTERPRET mode · Issue #5895 · triton-lang/triton</a>: Describe the bug For the simple 2D matrix multiplation, difference between my triton kernel and torch are: INTERPRET mode: 9.5367431640625e-07 (set os.environ[&quot;TRITON_INTERPRET&quot;] = &quot;1&q...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1339136840673792040)** (7 messages): 

> `CUDA Memory Model Confusion, PTX Instruction Explanation, Blackwell Tensor Memory Management` 


- **CUDA Memory Model raises questions**: A beginner in CUDA shared concerns about a code snippet in PMPP violating the **C++ memory model** regarding load/store operations and highlighted their findings on **Stack Overflow**: [link](https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi).
   - They emphasized that while source code order may ensure correct behavior, it can lead to bugs with new compilers or hardware, advocating for **acquire/release annotations**.
- **Clarification on PTX Instruction**: A member inquired whether the PTX instruction `ldmatrix.sync.aligned.m8n8.x4.b16` means each register holds an entire **8x8 matrix** or just a portion, based on the datatype being **f16**.
   - Another member explained that register definitions are **per thread**, and each thread can indeed load values corresponding to one **8x8 matrix**.
- **Insight on Blackwell Tensor Memory**: A user double-checked if the new **tensor memory** on Blackwell GPUs is hardware-managed and faster than shared memory but unclear compared to **L1 cache**.
   - A member clarified that **tensor memory** is **software-managed** and pointed to [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-alloc-manage-instructions) for details on dedicated allocation functions.



**Link mentioned**: <a href="https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi.">CUDA memory model: why acquire fence is not needed to prevent load-load reordering?</a>: I am reading the book &amp;quot;Programming Massively Parallel Processors&amp;quot; and noticed the below code snippets to achieve &amp;quot;domino-style&amp;quot; scan:&#xA;if (threadIdx.x == 0) {&#x...

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1338982389711376444)** (18 messages🔥): 

> `CPUOffload functionality, CPU optimizer step, Backward pass derivation, Shared memory techniques, Gradient checking issues` 


- **Understanding CPUOffload**: Members are grappling with how [`CPUOffload`](https://link.to.cpuoffload) works, particularly in the context of gathering DTensor shards to rank 0 for optimizer updates without heavy overhead.
   - One proposed using shared memory techniques like `mmap()` or `shm_open()` for efficiency, as managing tensor data transfers between GPUs and CPU could streamline operations.
- **CPU optimizer step complexities**: A member seeks a technique to perform a **CPU** optimizer step fused with gradient clipping specifically on rank 0, aiming to use reduced gradients while avoiding a traditional allreduce setup.
   - There are discussions about the feasibility of such an approach, noting that colocating everything on rank 0 may not be a bottleneck due to the potential use of parallel processing.
- **Automating backward pass derivation**: A member expresses frustration over deriving backward passes for complex **forward()** functions, seeking more efficient methods or automation tools beyond `sympy`.
   - Suggestions arise, including utilizing `torch.compile` with specific logging to facilitate understanding of computation graphs, although optimization concerns are noted.
- **Challenges with gradient checking**: Concerns were raised about inconsistencies observed with `gradgradcheck()`, particularly regarding outputs being cancelled or summed unexpectedly, leading to confusion.
   - A member noted that returning a zero matrix could complicate the verification process, suggesting the need for further examination before escalating potential issues on GitHub.
- **Clarifying derivative computations**: Discussions emphasized the complexities involved in double backward computations, with members acknowledging challenges in tracking the computation graph effectively.
   - There is recognition that while automatic differentiation provides gradients, some manual simplification may still be necessary for clarity and correctness.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: https://github.com/RC4ML/LoHan
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1339054178521059459)** (27 messages🔥): 

> `CUDA Installation Issues, Global Memory Coalescing in CUDA, Feedback on CUDA Code Structure, Error Handling in CUDA, Memory Coalescing Visualization` 


- **Troubleshooting CUDA Installation**: A user reported difficulty running CUDA on their PC, encountering a 'cudafe++' access violation error after multiple installation attempts.
   - They sought advice on resolving this issue, highlighting their struggle with CUDA toolkit installation.
- **Understanding Global Memory Coalescing**: A user learned about global memory coalescing through [Simon Boehm's blog post](https://siboehm.com/articles/22/CUDA-MMM) and discussed indexing schemes in matrix multiplication.
   - They pointed out an inconsistency in their implementation leading to duplicate indexes and sought clarification on the correctness of block dimensions.
- **Feedback Requested on CUDA Code Design**: A beginner in CUDA requested feedback on their code structure and design, wanting tips on error handling and memory cleanup in CUDA.
   - They received guidance on the use of C versus C++ in CUDA programming and were directed to a [GitHub repository](https://github.com/nvidia/cccl) for further reading.
- **Clarifying CUDA's C vs C++ Usage**: Discussions emerged regarding the preference of C or C++ for CUDA coding, emphasizing that CUDA is primarily compiled with nvcc as a C++ compiler.
   - It was noted that while you can write CUDA in pure C, many modern libraries heavily leverage C++ features.
- **Exploring PPC Course Materials**: A user expressed excitement about finding a PPC course and looked forward to exploring its contents.
   - Their engagement indicates a desire to further their understanding of CUDA and parallel programming.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/saptarshichaudhuri/4c3c63448279c8b87ba2fe5ce83d8de9">Sample matrix multiplication - CUDA</a>: Sample matrix multiplication - CUDA. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://www.youtube.com/watch?v=QmKNE3viwIE">4.5x Faster CUDA C with just Two Variable Changes || Episode 3: Memory Coalescing</a>: Memory Coalescing for efficient global memory transfers in CUDA C.Video Notes: https://0mean1sigma.com/chapter-4-memory-coalescing-and-tiled-matrix-multiplic...</li><li><a href="https://github.com/nvidia/cccl">GitHub - NVIDIA/cccl: CUDA Core Compute Libraries</a>: CUDA Core Compute Libraries. Contribute to NVIDIA/cccl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1339287560803061893)** (2 messages): 

> `CUDA memory model, Atomic Operations in CUDA, Thread Synchronization, GPU vs CPU Architecture, Understanding Scan/Prefix Sum` 


- **CUDA Memory Model Confusion**: A CUDA beginner raised concerns about a code snippet from the book PMPP that might violate the **C++ memory model** regarding memory ordering, specifically questioning the need for acquire semantics.
   - They expressed uncertainty about whether the absence of a thread fence might lead to undefined behavior and sought clarification on the CUDA documentation about handling memory models.
- **Discussion on Atomic Operations**: The thread discussed the use of **atomicAdd** in relation to thread synchronization, pondering why a thread fence is only necessary after the atomic operation.
   - A member mentioned the potential risks of relying on compiler behavior without proper acquire/release annotations, suggesting this could cause elusive bugs.
- **Comparing GPU and CPU Architectures**: Members noted fundamental differences between **GPU and CPU constructions**, highlighting that CUDA cores are much smaller and do not have features like out-of-order execution.
   - This leads to more deterministic behavior in CUDA, as the order of execution follows the instruction sequence, unlike CPUs which might reorder operations.
- **Community Insights Needed**: The beginner sought insights and clarification from more experienced members regarding whether the book's code snippet is an oversight or justified in the context of CUDA.
   - They specifically mentioned that even experienced community members were initially confused about the legality of the code's behavior.



**Link mentioned**: <a href="https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi.">CUDA memory model: why acquire fence is not needed to prevent load-load reordering?</a>: I am reading the book &amp;quot;Programming Massively Parallel Processors&amp;quot; and noticed the below code snippets to achieve &amp;quot;domino-style&amp;quot; scan:&#xA;if (threadIdx.x == 0) {&#x...

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1339316722976423978)** (2 messages): 

> `FP8 Dynamic Quantization, INT8 Dynamic Quantization, Issue Resolution` 


- **Dynamic Quantization Options Available**: Users can now try **FP8** or **INT8 dynamic quantization** directly from torchao after recent discussions.
   - This follows the resolution of an issue that involved user **<@969697995522191360>**, indicating smoother functionality.
- **Previous Issue Largely Resolved**: Discussion revealed that the prior issue has been largely resolved, leading to improved user experience.
   - *Yeah this has been resolved,* reiterated a member, confirming the positive outcome.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1339007335909167185)** (2 messages): 

> `Quantization-Aware Training, QuEST Method, YouTube Animation Parody` 


- **Exploring Quantization-Aware Training for LLMs**: Recent discussions highlighted the importance of **Quantization-Aware Training (QAT)** in reducing costs of large language models, emphasizing its ability to train with low bit-width while maintaining accuracy.
   - A study indicated that **8-bits weights and activations** are optimal for performance compared to FP16/BF16, paving the way for new methods like QuEST.
- **Introducing the QuEST Method for Model Training**: The **QuEST** method was proposed as a state-of-the-art approach, achieving better accuracy with models trained with **4-bits or less**, while being Pareto-competitive with FP16.
   - It utilizes techniques like the **Bengio trick (STE)** and RMS combined with unique quantization error separation to enhance training efficiency.
- **A Short Take on 'It's a Good Deal' Animation**: A YouTube short titled, *'it's a good deal'*, parodizes existing content with creative animation techniques, garnering interest from the community.
   - The video highlights the blend of humor and visual artistry, especially in the context of **#strangerthings** and **#blender3d**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05003">QuEST: Stable Training of LLMs with 1-Bit Weights and Activations</a>: One approach to reducing the massive costs of large language models (LLMs) is the use of quantized or sparse representations for training or deployment. While post-training compression methods are ver...</li><li><a href="https://www.youtube.com/shorts/QnxbNd74UCU">it&#39;s a good deal. parody of a parody by Matt Storer #strangerthings #animation #b3d #blender3d</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1339180282212909087)** (1 messages): 

> `Nebius Meetup, Kubernetes operator for Slurm, Test-time computation in agentic systems` 


- **Nebius Meetup in San Francisco**: Nebius is hosting a meetup in SF on **March 13th**, featuring insights into their architecture and development principles, including [event registration details](https://nebius.com/events/nebius-roadshow-san-francisco).
   - *Attendees will receive free credits* to try out Nebius GPU Cloud powered by NVIDIA during the event.
- **Deep dive into Kubernetes Operator for Slurm**: The session will include a detailed look at how Nebius developed a **Kubernetes operator** for **Slurm**, a scalable workload manager for clusters.
   - This aims to enhance the management of resources while supporting AI workloads more effectively.
- **Unlocking agentic systems through Test-time Computation**: The meetup will explore how **test-time computation** can open new capabilities for **agentic systems** within AI frameworks.
   - This segment is expected to reveal innovative applications and implications of the technology.



**Link mentioned**: <a href="https://nebius.com/events/nebius-roadshow-san-francisco">Nebius AI Cloud Unveiled. San Francisco Meetup</a>: Discover the most efficient way to build, tune and run your AI models and applications on top-notch NVIDIA® GPUs.

  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1339069679976583261)** (5 messages): 

> `T-mac paper inquiries, FMHA BWD Kernel example, Contributions to BWD Kernels` 


- **Inquiries about T-mac Paper Implementation**: A member reached out to @lei regarding questions about the implementation of the **T-mac** paper, expressing interest in discussing it via PM.
   - This highlights an active engagement in collaborative discourse around *research collaboration*.
- **FMHA BWD Kernel Example Shared**: Another member shared a link to an example of an **FMHA BWD Kernel** implementation on GitHub, emphasizing its role in high-performance kernel development.
   - This can be accessed at [example_mha_bwd.py](https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_bwd.py).
- **Call for Contributions to BWD Kernels**: A member welcomed contributions for additional **BWD kernels**, suggesting an open and collaborative environment for development.
   - This invitation reflects the community's interest in enhancing the resources available for kernel optimization.



**Link mentioned**: <a href="https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_bwd.py">tilelang/examples/flash_attention/example_mha_bwd.py at main · tile-ai/tilelang</a>:  Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels - tile-ai/tilelang

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1339320713240252539)** (2 messages): 

> `FSDP, Liger Kernel` 


- **Struggles with FSDP Implementation in Liger Kernel**: A member expressed difficulty in using **FSDP** with **Liger Kernel**, indicating they've been trying for hours without success.
   - *Does anyone know how to use FSDP with Liger Kernel?*
- **Request for Help on FSDP**: Another member acknowledged the challenge faced with **FSDP**, showing empathy towards the user who's been struggling.
   - They suggested potential search strategies to find relevant resources or community insights.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1339124822210514964)** (3 messages): 

> `Tilelang v0.1.0 Release, Community Engagement, High-Performance AI Kernels` 


- **Tilelang v0.1.0 Released for High-Performance AI Kernels**: The community celebrated the release of [tilelang v0.1.0](https://github.com/tile-ai/tilelang), a new pythonic DSL designed for high-performance AI kernels, featuring dedicated memory allocations and optional layout and pipeline annotations.
   - Highlighted features include **fine-grained thread-level control**, making it a promising tool for developers focused on efficiency.
- **Offer for a Community Talk**: A member expressed interest in inviting the creator for a talk about tilelang, suggesting it would be beneficial for the community.
   - The developer agreed, stating that they would love to share more once the associated preprint is ready, indicating that it was still a work in progress.



**Link mentioned**: <a href="https://github.com/tile-ai/tilelang">GitHub - tile-ai/tilelang: Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels</a>:  Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels - tile-ai/tilelang

  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1339347516361146368)** (42 messages🔥): 

> `DeepSeek-R1 Model for GPU Kernel Generation, Collaboration on Project Popcorn, KernelBench Benchmark Discussion, Performance of Generated Kernels` 


- **DeepSeek-R1 simplifies GPU kernel generation**: NVIDIA's experiment showcased the DeepSeek-R1 model, which creates GPU attention kernels optimized for performance using [inference-time scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/). Scaling test time compute to over **10 minutes** significantly improved results.
   - A member speculated that the verifier might be **ncu** combined with a PyTorch baseline correctness check.
- **Project Popcorn collaboration efforts**: Plans are being made to release the project 'tasks' for easier public collaboration on Project Popcorn, but it hasn't been fully open yet. Members expressed interest in contributing when those tasks are available.
   - There's ongoing development associated with Stanford, and relevant infrastructure for collaboration is being built on [Discord](https://discord.gg/MAnFAGRn).
- **KernelBench performance insights**: Discussion centered around the fact that the workflow produced correct kernels for **100% of Level-1** and **96% of Level-2** problems using Stanford's KernelBench benchmark. Questions arose about the lack of performance reporting on the higher levels, indicating possible saturation of the benchmark.
   - It was suggested that a new, more challenging benchmark may be needed as the existing one appears saturated.
- **The importance and niche of GPU kernels**: The conversation highlighted that while **GPU programming** is viewed as a niche segment of software engineering, it holds significant value in saving resources and costs for companies reliant on GPU kernels. Debates centered on the comparative value of general software engineering versus specialized GPU kernel engineering.
   - Some members acknowledged the essential role of GPU kernels in many deep learning applications, drawing parallels to how DeepMind improved matrix multiplication algorithms.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940">Automating GPU Kernel Generation with DeepSeek&#x2d;R1 and Inference Time Scaling | NVIDIA Technical Blog</a>: As AI models extend their capabilities to solve more sophisticated challenges, a new scaling law known as test&#x2d;time scaling or inference&#x2d;time scaling is emerging. Also known as AI reasoning ...

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1339010857530822717)** (2 messages): 

> `Complex MatMul Performance, Reinterpreting ST to CST` 


- **Struggles with Complex MatMul Performance**: A member is trying to create a complex **matmul** but is having difficulty achieving performance similar to [the benchmark kernel](https://link.to.benchmark). They are seeking an implementation that matches the performance of the real example kernel.
- **Issues Reinterpreting ST as CST**: The same member expressed frustration with reinterpreting **st** as **cst**, specifically citing attempts with `subtile_inplace` that didn’t integrate well with **mma**.
   - They are looking for guidance or alternative approaches to resolve this issue.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1338968411102773309)** (49 messages🔥): 

> `Prompt Optimization, Dataset Evaluation, Model Formatting Flexibility, CodeSteer Release, Math Verification` 


- **Streamlining Prompt Instructions for Clarity**: The team discussed adding clear formatting instructions directly within the question templates to enhance model understanding and consistency, such as using <answer> tags for final answers.
   - This approach aims to avoid ambiguity in model responses and ensure that all responses adhere to expected formats, benefitting both LLMs and human evaluators.
- **Concerns on Model Parsing and Formatting**: There was a debate on the necessity of allowing more flexible answer formats versus strictly punishing incorrect formats, with suggestions leaning towards a combination of improved parsing and clearer instructions.
   - This reflects a desire to balance model training effectiveness with the precision required in output formats, ensuring usability for different datasets.
- **CodeSteer Launch and Use**: Andreas mentioned the release of [CodeSteer](https://github.com/yongchao98/CodeSteer-v1.0), highlighting its open-source nature and the citation requirement for research usage.
   - This signals an important contribution to the community for those involved in research and development around code generation.
- **Issues with Propositional Logic Dataset**: Concerns were raised about the propositional logic dataset being broken, prompting discussions on whether to remove it or fix its construction.
   - Clarifications about specific errors in puzzle constructions indicate a need for thorough auditing of datasets to ensure accuracy.
- **Enhancing Dataset Evaluations through Iterative Improvements**: There was an emphasis on iterative evaluation methods across various datasets to ensure each model performs well, even if that requires adjusting prompts or the structures of datasets.
   - Members expressed the ongoing importance of human oversight in fine-tuning prompts and evaluation strategies as they work towards developing effective solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Adefioye/AI-Playground/blob/main/eval/Prompt.md">AI-Playground/eval/Prompt.md at main · Adefioye/AI-Playground</a>: Contribute to Adefioye/AI-Playground development by creating an account on GitHub.</li><li><a href="https://github.com/yongchao98/CodeSteer-v1.0">GitHub - yongchao98/CodeSteer-v1.0: Code and dataset of CodeSteer</a>: Code and dataset of CodeSteer. Contribute to yongchao98/CodeSteer-v1.0 development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/Math-Verify">GitHub - huggingface/Math-Verify</a>: Contribute to huggingface/Math-Verify development by creating an account on GitHub.</li><li><a href="https://github.com/agentica-project/deepscaler">GitHub - agentica-project/deepscaler: Democratizing Reinforcement Learning for LLMs</a>: Democratizing Reinforcement Learning for LLMs. Contribute to agentica-project/deepscaler development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1339272257029279765)** (2 messages): 

> `OpenAI o1 and o3 availability, Groq-powered Llamas introduction, Nitro feature upgrade, Groq DeepSeek R1 70B performance` 


- **OpenAI o1 and o3 Now Available to Everyone**: OpenAI has announced that the o1 and o3 reasoning model series are open for all OpenRouter users, eliminating the need for BYOK and providing higher rate limits for those previously using their own keys. More details can be found [here](https://x.com/OpenRouterAI/status/1889708759355691327).
   - The models also feature web search capabilities, adding significant utility to the experience.
- **Groq-Powered Llamas Delivering Speed**: With Groq officially supported, users can now experience lightning-fast endpoints with Groq-powered Llamas, achieving over **250 tokens per second** for Llama 3.3 and **600 TPS** for Llama 3.1. Details on available models are shared at [this link](https://openrouter.ai/provider/groq).
   - Users have the option to bring their own keys for boosted rate limits.
- **Revamped Nitro Feature Boosts Throughput**: The `:nitro` suffix is now enhanced for all models, enabling users to sort endpoints by latency and throughput rather than appearing as separate endpoints. This powerful configuration can be achieved via API or directly within the chatroom.
   - Enhanced charts have also been introduced to track provider performance over time, making it easier to compare them.
- **Groq DeepSeek R1 70B Sets New Speed Record**: The newly added Groq DeepSeek R1 70B achieves a phenomenal rate of approximately **1000 tokens per second**, marking a new benchmark in speed. More information can be found [here](https://x.com/OpenRouterAI/status/1889726731571044538).
   - The addition includes support for numerous parameters and an option for users to bring their own key for additional rate limit boosts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1889726731571044538">Tweet from OpenRouter (@OpenRouterAI)</a>: Excited to announce @GroqInc officially on OpenRouter! ⚡️- incl. a record-fast 1000 TPS distilled DeepSeek R1 70B- tons of supported parameters- bring your own key if you want, get a rate limit boostP...</li><li><a href="https://x.com/OpenRouterAI/status/1889708759355691327">Tweet from OpenRouter (@OpenRouterAI)</a>: OpenAI o1 and o3 are now available to all OpenRouter users!BYOK is no longer required. If you did have your own key working, you now have much higher rate limits.They work with web search too 👇</li><li><a href="https://openrouter.ai/openai/o3-mini-high.">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/provider/groq">Groq | OpenRouter</a>: Browse models provided by Groq
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1338967086470402129)** (163 messages🔥🔥): 

> `Chat History Issues, Provider Routing and Performance, Model Performance Comparisons, OpenRouter Updates, Credit and Subscription Concerns` 


- **Chat History Disappears Amid Updates**: Several users reported issues with disappearing chat histories after recent updates, highlighting that chat histories are only stored locally, which was not made clear upfront.
   - Members discussed the lack of prominent warnings regarding data loss when clearing browser history, suggesting the need for clearer messaging in the application.
- **Provider Routing Effects on Performance**: A user blacklisted a provider due to consistently receiving empty responses, indicating problems with the routing system favoring lesser-quality providers.
   - Another member recommended checking the documentation to disable fallback settings for better control over provider selection.
- **Concerns about Model Performance**: Discussion arose regarding the mixed reliability of hosted models, with some users experiencing poor performance and empty responses when using certain models.
   - Users noted that while some models like Mixtral performed well, others, including Llama3, were described as less reliable.
- **Updates on OpenRouter Features**: Updates from OpenRouter included the introduction of a price suffix for better modeling of request costs and a discussion on how requests are routed among providers.
   - The community discussed how the default behavior ensures load balancing across the best available providers to maximize performance.
- **Subscription and Credit Usefulness**: Users expressed frustration over spending on openAI while seeking more value from their subscriptions, especially regarding model access and performance.
   - Concerns were raised regarding the potential expiration of credits and how previously paid credits can be utilized within different contexts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - Improve AI Model Decision Making</a>: Learn how to use reasoning tokens to enhance AI model outputs. Implement step-by-step reasoning traces for better decision making and transparency.</li><li><a href="https://community.openai.com/t/are-openai-credits-expiring/511215">Are OpenAI credits expiring?</a>: Since dashboard change, I see no warning about credit expiration date. They forgot to put it, they placed it somewhere else or credits are not expiring any more?</li><li><a href="https://www.reddit.com/r/openrouter/comments/1inpby4/structured_output_with_deepseekr1_how_to_account/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing - Smart Multi-Provider Request Management</a>: Route AI model requests across multiple providers intelligently. Learn how to optimize for cost, performance, and reliability with OpenRouter&#x27;s provider routing.</li><li><a href="https://tenor.com/view/bau-bau-merrow-virtualmerrow-fuwamoco-fuwawa-gif-10720476399213933291">Bau Bau Merrow GIF - Bau bau Merrow Virtualmerrow - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/docs/features/provider-routing#floor-price-shortcut">Provider Routing - Smart Multi-Provider Request Management</a>: Route AI model requests across multiple providers intelligently. Learn how to optimize for cost, performance, and reliability with OpenRouter&#x27;s provider routing.</li><li><a href="https://x.com/sama/status/1889755723078443244">Tweet from Sam Altman (@sama)</a>: OPENAI ROADMAP UPDATE FOR GPT-4.5 and GPT-5:We want to do a better job of sharing our intended roadmap, and a much better job simplifying our product offerings.We want AI to “just work” for you; we re...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1338992273169580042)** (149 messages🔥🔥): 

> `Deep Hermes Model Release, Speculative Decoding in LM Studio, Calibration Dataset Characteristics, Quantization Strategy for LLMs, Hugging Face Agent Certification Course` 


- **Deep Hermes Model Release Expectations**: Members are eager for the release of the Deep-Hermes-8B model weights, with discussions surrounding benchmarks and announcements for its accessibility on the NousResearch HuggingFace repo.
   - Teknium mentioned preparations are underway, including benchmarks and a model card, with a hint of utilizing the model for composing posts about the release.
- **LM Studio Introduces Speculative Decoding**: The latest LM Studio 0.3.10 Beta introduces **Speculative Decoding**, aiming to accelerate inference by using a main and draft model together, which can enhance performance significantly.
   - Despite the potential benefits, some members noted mixed results, suggesting it works best for larger models and may not provide noticeable speed improvements in all scenarios.
- **Questions About Calibration Dataset**: There was curiosity about the nature of the calibration dataset used, with mentions of its seemingly random and unstructured content that resembles poor pretraining data.
   - Jsarnecki explained that the odd dataset choice was the result of research indicating that near-random data snippets produced better training outcomes, even when compared to more traditional datasets like wikitext.
- **Strategies for Quantizing LLMs**: Discussion included strategies for creating F32.imatrix for LLaMA models, with jsarnecki sharing insights into using Flash Attention and offloading GPU layers to handle limited memory resources.
   - Members emphasized the importance of examining and comparing different quantization strategies for model accuracy and efficiency.
- **Hugging Face Agent Certification Course**: Members shared information about the ongoing Hugging Face Agent Certification course, highlighting its relevance and the skills being covered, such as creating personal agents with the `smolagents` library.
   - The course includes a benchmark assessment opportunity, enticing many participants who feel confident in their knowledge of the fundamentals.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1889755723078443244">Tweet from Sam Altman (@sama)</a>: OPENAI ROADMAP UPDATE FOR GPT-4.5 and GPT-5:We want to do a better job of sharing our intended roadmap, and a much better job simplifying our product offerings.We want AI to “just work” for you; we re...</li><li><a href="https://tenor.com/view/apparently-its-a-big-deal-big-deal-big-deal-apparently-it-is-a-big-deal-gif-26730751">Apparently Its A Big Deal Big GIF - Apparently Its A Big Deal Big Deal - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=nt7ckZDTtis">An unfiltered conversation with Chamath Palihapitiya</a>: Join Nolan Fortman and Logan Kilpatrick as they dive into an unfiltered conversation with Chamath Palihapitiya, covering:- open source AI- the massive comput...</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding)">LM Studio Docs | LM Studio Docs</a>: Learn how to run Llama, DeepSeek, Phi, and other LLMs locally with LM Studio.</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio - Beta Releases</a>: Beta and Release Candidate versions of LM Studio</li><li><a href="https://www.cnbc.com/2025/02/11/ken-griffin-says-trumps-bombastic-trade-rhetoric-is-a-mistake-thats-eroding-trust-in-the-us.html">Ken Griffin says Trump&#x27;s &#x27;bombastic&#x27; trade rhetoric is a mistake that&#x27;s eroding trust in the U.S.</a>: The billionaire hedge fund founder&#x27;s comments came after Trump on Monday evening signed an order that would impose 25% tariffs on steel and aluminum imports.</li><li><a href="https://huggingface.co/Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF/tree/main">Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF at main</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339187638896164956)** (2 messages): 

> `LoRA Forget Less and Learn Less, Automated Capability Discovery, Diversity in Technology, Self-Exploration in AI Models` 


- **Insights from LoRA Forget Less and Learn Less**: The paper titled *LoRA Forget Less and Learn Less* highlights the importance of ensuring **diversity** in the technology industry, specifically in Portuguese Brazil.
   - It discusses the challenges of a **domain shift** in combining language and technology, emphasizing the need for models with enough capacity to avoid **catastrophic forgetting**.
- **Automated Capability Discovery Introduced**: New research explores whether frontier models can engage in **self-exploration** to identify their own capabilities and failure modes through a process called Automated Capability Discovery (ACD).
   - Led by [@cong_ml](https://x.com/cong_ml) and [@shengranhu](https://x.com/shengranhu), ACD allows a foundation model to act as a scientist, proposing open-ended tasks to systematically probe its abilities, showcasing results across multiple models like GPT and Claude.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jeffclune/status/1889568685632667672">Tweet from Jeff Clune (@jeffclune)</a>: Introducing Automated Capability Discovery!ACD automatically identifies surprising new capabilities and failure modes in foundation models, via &#34;self-exploration&#34; (models exploring their own a...</li><li><a href="https://arxiv.org/abs/2502.07577">Automated Capability Discovery via Model Self-Exploration</a>: Foundation models have become general-purpose assistants, exhibiting diverse capabilities across numerous domains through training on web-scale data. It remains challenging to precisely characterize e...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1339100746259632311)** (3 messages): 

> `AI value systems, Superagent Hackathon, cRACK'd Den event, AI safety declaration summit` 


- **AI Develops Coherent Value Systems**: A member highlighted that as AIs become more advanced, they establish internal value hierarchies, showing a preference for lives in **Pakistan > India > China > US**.
   - This raises significant questions regarding **AI alignment** and the implications of AI-driven value systems.
- **Join the Superagent Hackathon!**: A one-day hackathon invites developers to create next-level **SUPERAGENTS**, utilizing Story's **Agent Transaction Control Protocol** for integration across various frameworks and chains.
   - Participants can work on new projects or improve existing ones, with opportunities to win prizes and collaborate.
- **Unwind at the cRACK'd Den Event**: The **cRACK'd Den** event serves as a mixer for developers building in crypto and AI, celebrating creativity and innovation after the hackathon.
   - Attendees can enjoy performances, food, and the announcement of hackathon winners while connecting with like-minded individuals.
- **US and UK Decline AI Safety Declaration**: At an international summit, the US, represented by Vance, refused to sign a safety declaration, fearing partnerships with **authoritarian regimes** like China could compromise national security.
   - Concerns over the language around **multilateralism** and international collaboration led to a lack of agreement, particularly regarding US leadership in AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arstechnica.com/ai/2025/02/us-and-uk-refuse-to-sign-ai-safety-declaration-at-summit/">US and UK refuse to sign AI safety declaration at summit</a>: US stance is &ldquo;180&#x2d;degree turnaround&rdquo; from Biden administration.</li><li><a href="https://x.com/DanHendrycks/status/1889344074098057439?t=IXS9ty0t1fVgxJ4W90enDw&s=33">Tweet from Dan Hendrycks (@DanHendrycks)</a>: We’ve found as AIs get smarter, they develop their own coherent value systems.For example they value lives in Pakistan &gt; India &gt; China &gt; USThese are not just random biases, but internally con...</li><li><a href="https://lu.ma/superagenthackathon">Super Agent Hackathon · Luma</a>: Come build next level SUPERAGENTS that can only exist on Story.THE STORY AGENT LAB is cooking up the next SUPER AGENT alongside the top agentic frameworks and…</li><li><a href="https://lu.ma/crackdden">cRACK&#x27;d DEn · Luma</a>: 🚨 CRACK&#x27;D DEVS. AI ENJOOYERS. ETHDENVERERS. 🚨Psyops got you down?Ass numb from biilding?Cursor won&#x27;t compile?Yur token won&#x27;t bond?Need agent…
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339187638896164956)** (2 messages): 

> `LoRA Forget Less and Learn Less, Diversity in Technology, Automated Capability Discovery, Foundation Models Self-Exploration` 


- **Insights from LoRA paper on technology in Brazil**: Reading the paper “LoRA Forget Less and Learn Less” revealed the importance of diversity in the technology sector, particularly in Portuguese Brazil.
   - The author noted that combining language and technology presents challenges, advocating for either acceptance of domain shifts or using models capable of handling specific language requirements.
- **Introducing Automated Capability Discovery**: [@jeffclune's Twitter thread](https://x.com/jeffclune/status/1889568685632667672) highlights a new framework called **Automated Capability Discovery (ACD)** that identifies capabilities and failure modes in foundation models via self-exploration.
   - This framework is capable of systematically proposing open-ended tasks to probe the abilities of models, potentially uncovering surprising capabilities and failures.
- **ACD Demonstrated with Multiple Models**: The **Automated Capability Discovery** framework has been demonstrated across several models including **GPT**, **Claude**, and **Llama**, showcasing its utility in evaluating foundational models.
   - Initial results indicate that ACD fosters a better understanding of models' abilities and potential risks through a systematic approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jeffclune/status/1889568685632667672">Tweet from Jeff Clune (@jeffclune)</a>: Introducing Automated Capability Discovery!ACD automatically identifies surprising new capabilities and failure modes in foundation models, via &#34;self-exploration&#34; (models exploring their own a...</li><li><a href="https://arxiv.org/abs/2502.07577">Automated Capability Discovery via Model Self-Exploration</a>: Foundation models have become general-purpose assistants, exhibiting diverse capabilities across numerous domains through training on web-scale data. It remains challenging to precisely characterize e...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1339079084633882675)** (1 messages): 

> `Google Sheets Support, NotebookLM Feedback Survey` 


- **Users Want Google Sheets Support**: A notification was shared regarding increased interest in **Google Sheets** and broader **spreadsheet support** for NotebookLM, prompting a [feedback survey](https://forms.gle/G78qnNCv2UwcYXc16) for input.
   - The request calls for details on the desired sheet, including its dimensions and types of data contained within.
- **Feedback on Ingesting Google Sheets**: The team is looking to gather specifications on what users wish to **ingest** from **Google Sheets** into NotebookLM, focusing on characteristics like the number of tabs and rows.
   - Users are also encouraged to share insights they hope to derive from their sheets, with an invitation to provide sanitized copies of their data.



**Link mentioned**: <a href="https://forms.gle/G78qnNCv2UwcYXc16">Google sheets for NotebookLM</a>: For those of you who have asked to be able to ingest Google Sheets into NotebookLM, we are looking for your feedback to better understand your use case!

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1338966305306447985)** (15 messages🔥): 

> `Health Tracking with Notebook LM, NotebookLM for Writing Assistants, Anki Flashcards Creation, Using NotebookLM for Resume Review, AI-Podcast Creation` 


- **Transforming Health Tracking with Google LM**: A user discussed utilizing Notebook LM to monitor health data, emphasizing how dynamically refreshing links to Google Sheets could revolutionize the tracking process. They mentioned already integrating various data streams into Looker Studio.
   - They emphasized the potential for Notebook LM to enhance health data management considerably and shared a link to their audio discussion on the topic.
- **NotebookLM Aids in Fantasy Novel Writing**: A user is leveraging NotebookLM as a writing assistant for their long-term fantasy novel project, focusing on world building and data organization. They value the audio generator for synthesizing questions their potential readers might have.
   - This tool helps them identify gaps and contradictions in their extensive world building efforts.
- **Effective Flashcards with NotebookLM**: A member noted that specific prompts in NotebookLM can effectively create Anki flashcards, improving their learning process. They received interest from another user asking for details on the prompt used.
   - This highlights the utility of NotebookLM in creating structured learning materials.
- **Revamping Resume Reviews Using NotebookLM**: One user is exploring using NotebookLM to assist in reviewing over 100 resumes by loading them alongside the relevant job description. Initial attempts at this approach seem promising, as the AI helps them reconsider candidate evaluations.
   - This method seems to aid in providing a fresh outlook on their review process.
- **Creating AI-Podcasts for Monetization**: A user elaborated on using AI to create podcasts quickly, emphasizing the substantial market opportunity for entrepreneurs in this space. They expressed how podcasting can elevate content consumption and market reach.
   - They pointed out the novelty of transforming static content into engaging audio, maximizing outreach without the need for public speaking.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://millionai.substack.com/p/create-ai-podcasts-in-seconds-without?r=297y6u&utm_medium=ios&triedRedirect=true.">🎙️create podcasts, in seconds (without speaking)🤐</a>: How I&#x27;d make an extra $7850/mo with a two-person AI-podcast 🎧 (no-code)</li><li><a href="https://notefeedlm.com/">NotefeedLM</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1338976871148163164)** (90 messages🔥🔥): 

> `User Limits and Sharing Notebooks, Student Experiences with NotebookLM, Audio Features and Personalization, Feedback on Product Limits, Source Handling Issues` 


- **Clarification on User Limits**: Users expressed confusion about query limits, particularly if sharing notebooks affects their own daily caps; it's confirmed that limits apply per user, not per notebook shared.
   - One member noted that NotebookLM's limits—50 queries for free and 500 for Plus—make usage difficult for students who rely on interactive questioning.
- **Students Share How They Use NotebookLM**: Undergraduate users reported using NotebookLM to create mock tests and summarize source materials, praising its effectiveness for study purposes.
   - Features like audio conversations were mentioned as a helpful tool for multitasking, though some expressed issues with certain functionalities not working correctly.
- **Interest in Personalized Audio Features**: Users asked about the possibility of using their own voice for audio features; currently, this capability is not available, but a beta program for voice interaction was mentioned.
   - One user mentioned manually transferring notes to other applications to read aloud, highlighting a desire for more integrated audio functionality.
- **Feedback on Limitations Impacting User Experience**: Concerns were raised regarding the 50-query daily limit, which some users feel restricts the potential for in-depth study and interaction with the app.
   - A student suggested that while 50 sources are manageable, the limit on queries undermines the application's effectiveness for research and learning.
- **Source Formatting Issues**: Some users reported they encountered issues with how sources are displayed when clicked; mangled formatting in some PDFs made it challenging to verify content.
   - The product team is aware of these formatting issues and is working towards potential improvements in displaying source materials accurately.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://blog.google/feed/notebooklm-google-one/">NotebookLM Plus is now available in the Google One AI Premium subscription.</a>: NotebookLM is a research and thinking companion designed to help you make the most of your information. You can upload material, summarize it, ask questions and transfor…
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1338980796119384155)** (56 messages🔥🔥): 

> `OpenRouter access to o1 and o3, Aider multi-session support, Editor model for code editing, GPT-5 roadmap update, User experiences with o3-mini` 


- **OpenRouter Grants Access to o1 and o3**: OpenRouter announced that **OpenAI o1 and o3 are now available** to all users, eliminating the need for BYOK and increasing rate limits.
   - This announcement was met with excitement, highlighting the improved functionality that also works with web search.
- **Chatbot Multi-Session Management in Aider**: Users expressed interest in enabling Aider to manage multiple tmux sessions for better control of processes like server spawning.
   - As a workaround, suggestions included using local setups with SSH connections to streamline the coding workflow.
- **Proposal for an 'Editor' Model**: There is a discussion about training a **1.5b 'editor' model** to collaborate with architect models, focusing on efficient code editing.
   - Participants believe that such a model could address issues with hallucinations and improve code diff accuracy in larger contexts.
- **GPT-5 Roadmap Insights Shared**: A significant update revealed plans for **GPT-4.5** and **GPT-5**, aiming to unify model offerings and simplify user experience.
   - The roadmap indicates that GPT-5 will incorporate various technologies and be made available to free tier users with different intelligence levels.
- **Users Share Experiences with o3-mini**: Feedback indicated that **o3-mini** performed well and was faster for coding tasks, especially when compared to other models.
   - Some users noticed improvements in deployment speed with o3, while others suggested combinations with models like Sonnet for optimal performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1889755723078443244">Tweet from Sam Altman (@sama)</a>: OPENAI ROADMAP UPDATE FOR GPT-4.5 and GPT-5:We want to do a better job of sharing our intended roadmap, and a much better job simplifying our product offerings.We want AI to “just work” for you; we re...</li><li><a href="https://x.com/OpenRouterAI/status/1889708759355691327">Tweet from OpenRouter (@OpenRouterAI)</a>: OpenAI o1 and o3 are now available to all OpenRouter users!BYOK is no longer required. If you did have your own key working, you now have much higher rate limits.They work with web search too 👇
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1338976932339122197)** (24 messages🔥): 

> `Implementing AI Agents in Testing, Aider Multi-Step Execution, Managing Aider Configuration, Using OpenRouter with Aider, Tracking Aider Configuration Sources` 


- **Integrate AI Agent for Test Automation**: A user discussed creating an AI agent that updates code according to test results from `results.json` based on provided features in `features.json` and editable files in `inputs.json`.
   - They inquired if such a task is feasible within Aider's purview and sought guidance on the best tools or approaches.
- **Executing Aider in Two Steps**: A user outlined a plan to run Aider in two steps: first to generate code for a specific requirement, then to create unit tests based on that code.
   - They expressed concerns about efficiently obtaining the code changes from the first step for use in the second.
- **Issues with Aider Configuration Drop**: A user mentioned that using `/drop` in Aider deletes all files specified in a `read:` configuration, which they found problematic.
   - They asked if there's a way to prevent Aider from dropping those specifically loaded configuration files.
- **Configuring Model Metadata for OpenRouter**: A user suggested creating a `.aider.model.metadata.json` file to configure `openrouter/openai/o3-mini-high`, since it wasn't listed in the default settings.
   - They provided a link to a source where the existing configuration standards could be observed.
- **Identifying Aider Configuration Source**: A user queried how to determine the specific source of Aider's configuration values, citing complexities from various possible origins like `.env` files, environment variables, and YAML configs.
   - They reported experiencing unexpected values and needed assistance to track down the origin of these configurations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com/library/openthinker/blobs/6490a490932e">openthinker/system</a>: A fully open-source family of reasoning models built using a dataset derived by distilling DeepSeek-R1.</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://github.com/Aider-AI/aider/commit/cf0710225c4fac6f07582821634a98447a74814f">Tell o1 &amp; o3-mini to use markdown · Aider-AI/aider@cf07102</a>: no description found</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>: Python SDK, Proxy Server (LLM Gateway) to call 100+ LLM APIs in OpenAI format - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1339277918026727434)** (9 messages🔥): 

> `Aider Mention on LinkedIn, CodeSteer-v1 Discussion, PyCharm Plugin Feasibility, Microsoft Support for JetBrains, O3 High and 3.5 Sonnet PRs` 


- **Aider gets a LinkedIn shoutout**: A LinkedIn post by [Addy Osmani](https://www.linkedin.com/posts/addyosmani_softwareengineering-programming-ai-activity-7289554729720852480-2oVx?utm_source=social_share_send&utm_medium=android_app&utm_campaign=copy_link) mentions Aider, prompting user curiosity.
   - The discussion revolves around finding specific prompts or files used in demos linked in the post.
- **Excitement around CodeSteer-v1**: User shared a link to an interesting [Hugging Face paper](https://huggingface.co/papers/2502.04350#67aaa92ca8192c1ba3c7798f) about CodeSteer-v1, sparking excitement.
   - Another user, however, noted that the paper focuses on numerical calculations within code rather than native LLM processing.
- **Query on PyCharm Plugin Development**: A user expressed interest in creating a similar PyCharm plugin and asked about the difficulty involved.
   - Another member acknowledged the feasibility of this plugin, noting recent support from Microsoft for JetBrains.
- **Microsoft's Role in JetBrains Support**: In response to a query, a member confirmed Microsoft's recent involvement in supporting JetBrains plugins.
   - The specifics of this support were not elaborated, leaving room for further discussion.
- **Inquiry on PRs for O3 High and 3.5 Sonnet**: A user sought clarification on whether specific pull requests are necessary to utilize O3 High and 3.5 Sonnet through OpenRouter.
   - However, no direct responses were provided within the existing messages.



**Link mentioned**: <a href="https://huggingface.co/papers/2502.04350#67aaa92ca8192c1ba3c7798f">Paper page - CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance</a>: no description found

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1338982822026940478)** (85 messages🔥🔥): 

> `Differences between SDXL and SD 1.5, Performance of Flux model, Distillation methods in AI models, Human preference benchmarks, Linux transition issues with ComfyUI` 


- **SDXL vs SD 1.5 quality insights**: Discussion highlighted that **SDXL** without a refiner shows comparable quality to **SD 1.5**, but **1.5** maintains unique interpretations of terms that **SDXL** lacks, often due to focus on popular aesthetics.
   - Members noted that **benchmarks** are crucial for judging quality, with **SDXL outperforming SD 1.5** in a benchmark context.
- **Flux model's consistent output style**: Members observed that the **Flux model** produces consistent facial features, particularly a unique cleft chin, suggesting reliance on **quality-tuned data** or distillation processes.
   - Some contrasted this with **SDXL**, arguing that lower diversity in Flux is overshadowed by its higher log likelihood distribution, allowing room for improved diversity through **loras**.
- **Distillation methods affect model performance**: Discussion clarified that **Schnell** is derived from **Pro** via 'timestep distilled,' while **Dev** uses 'guidance distilled,' influencing how models perform and share **loras**.
   - Participants referred to the difference in **data handling** between distillation methods and their impact on model quality.
- **Debates on human preference benchmarks**: Members expressed mixed feelings on **human preference benchmarks**, arguing they could skew towards aesthetically pleasing outputs rather than quality metrics.
   - Concerns were raised about the potential for these benchmarks to favor specific outputs like 'pretty ladies' over accurate representations based on detailed prompts.
- **Transition challenges from Windows to Linux for ComfyUI**: A user reported issues transitioning from **ComfyUI on Windows** to **Linux**, encountering **OOM errors** during video generation after following a guide.
   - Fellow members recommended ensuring proper **drivers** are installed and inquired about the user's Linux experience, underscoring that poor guidance may have caused instability.



**Link mentioned**: <a href="https://huggingface.co/segmind/Segmind-Vega">segmind/Segmind-Vega · Hugging Face</a>: no description found

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1338974813624205366)** (48 messages🔥): 

> `Server Author Flair, Trust Issues with Crypto/NFTs, Code Review Process, Open Source LLM Models, Generative Dashboards with Clickhouse and Streamlit` 


- **Welcome with Server Author Flair**: A member welcomed another by granting them the server author flair, prompting mixed reactions about whether it's a good thing.
   - *I have a fundamental mistrust with anyone who participated in crypto/nfts* expressed concerns about the community’s trustworthiness.
- **Discussing Code Review Options**: Members suggested a *code review process* to assess the safety of numerous MCP public servers, proposing multiple reviewers to manage the workload.
   - *Considering 900+ servers*, a member humorously noted the feasibility of using a language model to automate initial filtering of malicious code.
- **Open Source LLM Models Need Research**: Concerns were raised about the need for *ground-breaking research on open-source LLM models*, with references to DeepSeek copying ideas from OpenAI.
   - A member cited that while DeepSeek may have shared innovations first, they still utilized OpenAI’s technology.
- **Exploring Clickhouse and Streamlit**: One member expressed interest in creating a generative dashboard server using *Clickhouse and Streamlit*, while exploring potential monetization options.
   - They sought community input on the effectiveness of Streamlit compared to other tools like PowerBL, with a promise of future collaboration for monetization.
- **Accidental Server Confusion**: A member humorously acknowledged being in the wrong server, highlighting the potential for mix-ups within large communities.
   - The conversation noted various interests and goals, showcasing a lively and varied community dynamic.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

eggsquad: new Modular job postings 👀
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1339195468760223755)** (20 messages🔥): 

> `stdlib meetings, Mojo language features, sum types and parameterized traits, List with trait as element type, Bug report submitted` 


- **Uncertainty About stdlib Meetings**: A member inquired about the continuity of regular **stdlib meetings** after a previous mention in a community meeting, noting the inability to find any current information.
   - Another member confirmed that previous meetings were canceled due to scheduling conflicts and the departure of the organizer.
- **Inquiry on Mojo Language Features Timeline**: A user asked for a timeline regarding upcoming **Mojo language features**, specifically about support for **enums** and **parameterized traits**.
   - Discussion noted that `Variant` may cover some functionalities of enums, but **parameterized traits** remain a higher priority for the team.
- **Discussion on Sum Types and New Powers**: Members discussed the absence of a timeline for implementing **sum types**, explaining that there’s value in having these, but they don’t enable many new capabilities until foundational features are in place.
   - It was pointed out that the focus is on developing **ground level** features that allow Mojo to represent constructs similar to C.
- **Challenges with List of Traits in Mojo**: A user experienced a compiler crash when attempting to create a **List** with a trait as an element type, seeking advice on potential workarounds.
   - Another member clarified that it's necessary for elements to be concrete types, suggesting using `Variant` if the types are fixed.
- **Bug Report Submitted to GitHub**: A member mentioned submitting a **bug report** on GitHub regarding the limitations of `parallelize` with **Mojo** code that interacts with Python.
   - The bug described a situation where parallelized function calls result in runtime crashes under certain conditions.



**Link mentioned**: <a href="https://github.com/modular/mojo/issues/3993">[BUG][stdlib] Limitation of `parallelize` with Mojo code that interacts with Python · Issue #3993 · modular/mojo</a>: Bug description Actual behavior When a parallelized function call interacts with Python, it crashes at runtime under certain conditions. Referring to the code example below, struct Bar.start() uses...

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1339082853912281129)** (24 messages🔥): 

> `MAX -> Wasm compilation, Running Mojo programs, ONNX model execution, CUDA library issues, Mojo API considerations` 


- **MAX doesn't prioritize Wasm for now**: Currently, the Wasm backend is not a target focus for MAX, and this isn't on the near-term roadmap.
   - One member expressed curiosity about the relevance of Wasm, highlighting its potential for future use.
- **Multiple ways to run Mojo programs**: MAX is not the only way to run Mojo programs, as outlined in the [getting started tutorial](https://docs.modular.com/mojo/manual/get-started), which operates independently of MAX.
   - While MAX is crucial for interesting applications like those utilizing GPUs, it's not strictly necessary for running Mojo.
- **ONNX requires MAX for execution**: Members noted that Modular's support for executing ONNX models largely hinges on MAX, emphasizing its current necessity.
   - This highlights MAX's role in facilitating various ML model executions across the platform.
- **Seg faults linked to CUDA libraries**: Issues with seg faults from CUDA libraries raised concerns, with members suggesting that MAX is not solely dependent on CUDA.
   - Despite its minimal use of CUDA, MAX still relies on NVIDIA drivers, and specific operations in Mojo could lead to issues.
- **Discussion around Mojo API integration**: One member proposed integrating NVIDIA libraries directly into Mojo to simplify usage with MAX.
   - Opinions varied, with some suggesting that a complete move away from CUDA could enhance stability and performance.



**Link mentioned**: <a href="https://forum.modular.com/">Modular</a>: Build the future of AI with us and learn about MAX, Mojo, and Magic.

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1339058386389565492)** (37 messages🔥): 

> `VAE Reparameterization Trick, OpenAI IOI Paper, Scaled Cognition APT-1, Glean Agents Launch, OpenAI Roadmap Update` 


- **VAE Reparameterization Trick Explained**: A discussion emerged on why backpropagation cannot be performed through a distribution for VAEs, highlighting the need for the reparameterization trick.
   - One member clarified that VAEs generate distribution parameters that require stochastic sampling, a non-differentiable operation.
- **OpenAI's Competitive Programming Breakthrough**: A new OpenAI paper details their performance at IOI 2024, noting that the **o3 model** achieved gold without hand-crafted strategies, showcasing significant advancements in reasoning models.
   - The paper comments on model flexibility as key, illustrated by **o1-ioi's** prior need for specialized pipelines.
- **Introduction of APT-1 by Scaled Cognition**: Scaled Cognition announced their **APT-1 model**, specifically designed for agentic applications, which now holds the top position on agent benchmarks.
   - They revealed funding details including a **$21M** seed round led by Khosla Ventures, utilizing a fully synthetic data pipeline.
- **Unveiling Glean Agents**: Glean launched **Glean Agents**, a platform facilitating scalable AI agent management, with new features for data integration and governance.
   - They aim to enhance productivity by allowing user-friendly access to both company and web data.
- **OpenAI Roadmap Update for GPT Models**: OpenAI shared a roadmap update revealing the upcoming **GPT-4.5 and GPT-5** which aims to unify modeling approaches and simplify product offerings.
   - They indicated a shift away from non-reasoning models, emphasizing a push towards models that incorporate broader functionality and reasoning capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/glennsolomon/status/1889717350456315960?s=46">Tweet from Glenn Solomon (@glennsolomon)</a>: Proud to co-lead @FAL&#39;s Series B 🚀AI-powered creativity is only as good as the infrastructure behind it. fal is the inference layer fueling gen-media for Canva, Perplexity & more!Thrilled to part...</li><li><a href="https://x.com/glean/status/1889706504812683728">Tweet from Glean (@glean)</a>: Welcome to the agentic era 🚀 We’re excited to announce 𝐆𝐥𝐞𝐚𝐧 𝐀𝐠𝐞𝐧𝐭𝐬–our horizontal agent environment that enables employees and businesses to build, run, manage, and govern AI agents at sc...</li><li><a href="https://x.com/scaledcognition/status/1889721166421479751?s=46">Tweet from Scaled Cognition (@ScaledCognition)</a>: We’re Scaled Cognition, developing the first ever models trained specifically for agentic applications:1. Our first system, APT-1, is now #1 on agentic benchmarks.2. It was developed by a US team for ...</li><li><a href="https://arxiv.org/abs/2502.06807">Competitive Programming with Large Reasoning Models</a>: We show that reinforcement learning applied to large language models (LLMs) significantly boosts performance on complex coding and reasoning tasks. Additionally, we compare two general-purpose reasoni...</li><li><a href="https://x.com/arankomatsuzaki/status/1889522977185865833?s=46">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: no description found</li><li><a href="https://x.com/winstonweinberg/status/1889713028234416371?s=46">Tweet from Winston Weinberg (@winstonweinberg)</a>: Excited to announce our Series D led by @sequoia with participation from @conviction, @kleinerperkins, @OpenAI, @GVteam, @conviction, @eladgil, and @LexisNexis.Thank you to our customers, team, invest...</li><li><a href="https://x.com/swyx/status/1889810524696891903">Tweet from swyx 🔜 @aidotEngineer NYC (@swyx)</a>: RT @JeffDean: I&#39;m delighted to have joined my good friend and colleague @NoamShazeer for a 2+hour conversation with @dwarkesh_sp about a wi…</li><li><a href="https://x.com/iscienceluvr/status/1889517116816244995?s=46">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Competitive Programming with Large Reasoning ModelsNew paper from OpenAI highlighting results of their reasoning models on IOI and CodeForces.&#34;We competed live at IOI 2024 with o1-ioi and, using h...</li><li><a href="https://x.com/polynoamial/status/1889541408065028421?s=46">Tweet from Noam Brown (@polynoamial)</a>: This meme summarizes the paper nicelyQuoting Aran Komatsuzaki (@arankomatsuzaki) OpenAI presents:Competitive Programming with Large Reasoning Models- Competed live at IOI 2024- o3 achieved gold- Gener...</li><li><a href="https://x.com/deedydas/status/1889713595384312089?s=46">Tweet from Deedy (@deedydas)</a>: HUGE: OpenAI o3 scores 394 of 600 in the International Olympiad of Informatics (IOI) 2024, earning a Gold medal and 18 in the world.The model was NOT contaminated with this data and the 50 submission ...</li><li><a href="https://share.snipd.com/episode/645ae532-40fd-43ff-9ee4-eb76c8fd56fe">Jeff Dean &amp; Noam Shazeer – 25 years at Google: from PageRank to AGI</a>: Jeff Dean &amp; Noam Shazeer – 25 years at Google: from PageRank to AGI</li><li><a href="https://x.com/OpenAI/status/1889781541259321466">Tweet from OpenAI (@OpenAI)</a>: Today we&#39;re sharing a major update to the Model Spec—a document which defines how we want our models to behave.The update reinforces our commitments to customizability, transparency, and intellect...</li><li><a href="https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg">Tweet from Sam Altman (@sama)</a>: OPENAI ROADMAP UPDATE FOR GPT-4.5 and GPT-5:We want to do a better job of sharing our intended roadmap, and a much better job simplifying our product offerings.We want AI to “just work” for you; we re...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1339070584717316191)** (15 messages🔥): 

> `Checkpointing in Torchtune, MLFlow Logger Integration, Distributed Inference with Torchtune, Crypto Portfolio Growth` 


- **Step-based Checkpointing in Progress**: A member inquired about the ability to save checkpoints multiple times per epoch in **Torchtune**, to which another mentioned that **Joe** is currently working on this feature in [PR #2384](https://github.com/pytorch/torchtune/pull/2384).
   - *This is a widely requested feature* and is expected to improve the checkpointing process significantly.
- **MLFlow Logger Integration Merged**: The **MLFlow logger integration** was successfully merged, as reported by a member who was excited to test it ASAP after a busy week.
   - This integration aims to enhance logging capabilities in Torchtune.
- **Distributed Inference with Torchtune**: A member asked about running **distributed inference** using multiple GPUs with Torchtune, and another shared a [link](https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py) to relevant code for this task.
   - The member also noted that loading a saved model into **vLLM** will work for distributed inference and be *much faster*.
- **Crypto Portfolio Success**: One participant joyfully declared their **crypto portfolio** is currently *booming*, highlighting financial success in recent days.
   - This moment was shared amidst various discussions about software and developments in the Torchtune community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/2384">Implement step based checkpointing by joecummings · Pull Request #2384 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Closes #2105. This is a widely requested feature that al...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py">torchtune/recipes/dev/generate_v2_distributed.py at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1338965172232130591)** (10 messages🔥): 

> `Testing Assumptions, Gradient Accumulation Issue, Checkpointing Functionality, DPO Loss Calculation, Memory Management with Opt-in BWD` 


- **Testing Assumptions Under Scrutiny**: A member expressed doubt about the quality of their own code, stating that passing tests could be attributed to poor test writing rather than good implementation.
   - This highlights a common mindset where success in testing leads to questioning rather than confidence in one's coding skills.
- **Gradient Accumulation Debugging Frustration**: There is ongoing confusion around the [gradient accumulation fix](https://github.com/pytorch/torchtune/issues/2334), which is affecting training effectiveness.
   - Members described hours spent debugging without finding a root cause, and the issue appears complex and may require more collaborative effort.
- **Checkpointing Branch Works Perfectly**: One member confirmed successful testing of a checkpointing branch, stating it functions as intended and is ready for more documentation on resuming training.
   - Another member, having heard the good news, responded with humor and relief about its success.
- **DPO Loss Calculation Concerns**: Discussion arose around ensuring that DPO loss isn't unfairly affected by padded tokens, which ties back to a previous [issue](https://github.com/pytorch/torchtune/pull/1875).
   - The conversation emphasized the importance of accuracy in loss calculations and potential adjustments to mitigate inflated losses due to padding.
- **Memory Management Dilemma with Opt-in BWD**: A member reflected on the complicated relationship with **opt_in_bwd**, which saves memory during fine-tuning but introduces significant challenges.
   - They hinted at the potential for minimizing the cross-entropy peak to lessen the memory impact further.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/2334">Apply gradient accumulation fix to DPO/PPO recipes · Issue #2334 · pytorch/torchtune</a>: https://unsloth.ai/blog/gradient</li><li><a href="https://unsloth.ai/blog/gradient">Bug Fixes in LLM Training - Gradient Accumulation</a>: Unsloth&#x27;s Gradient Accumulation fix solves critical errors in LLM Training.</li><li><a href="https://github.com/pytorch/torchtune/pull/1875">Normalize CE loss by total number of (non-padding) tokens by ebsmothers · Pull Request #1875 · pytorch/torchtune</a>: In honor of the day the ML community first discovered the fact that (x1 / n1) + (x2 / n2) != (x1 + x2) / (n1 + n2)This PR changes how we calculate the loss when gradient accumulation is enabled. T...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1339263623079395411)** (4 messages): 

> `RWKV RNN, Scaling Challenges of RNNs, State Space Models, Importance of Attention Mechanisms` 


- **RWKV RNN Emerges as a Favorite**: One member highlighted their preference for **RWKV** as the most liked RNN, noting a plethora of good **Apache 2.0** content from the project community.
   - They praised the active Discord channel filled with *crunchy minds* discussing advancements and applications.
- **Skepticism on RWKV's Scaling Potential**: A member expressed admiration for the **RWKV** project but voiced skepticism about its ability to scale to the same levels as **transformers**.
   - This viewpoint sparked a discussion regarding the challenges of scaling RNNs in comparison to transformer architectures.
- **State Space Models Mentioned**: Another member suggested that skepticism about RWKV could hold true for state spaces, citing **Mamba** as an example.
   - This comment highlights ongoing discussions about the scalability of various model architectures.
- **Attention Mechanisms Remain Crucial**: A participant succinctly stated that *attention is still all we need*, underscoring its fundamental role in modern AI models.
   - This reinforces the ongoing importance and focus on attention mechanisms in the field of artificial intelligence.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1339021472496222300)** (5 messages): 

> `Training ML models with limited datasets, Accessing research papers, Tinystories paper, Anna's Archive` 


- **Tinystories Paper Tackles Limited Datasets**: A member suggested the [tinystories paper](https://link.to.tinystories) as a resource for training ML models using small or limited datasets.
   - This paper presents strategies for effective learning despite dataset constraints, which could be beneficial for the original inquiry.
- **Seeking Access to ResearchGate Papers**: A member reached out for assistance in obtaining a paper hosted on ResearchGate.
   - Another member recommended contacting the authors directly for a PDF copy.
- **Anna's Archive: A Resource for Research Papers**: A suggestion was made to use [Anna's Archive](https://annas-archive.org/) for accessing various papers, as it is billed as the largest open library.
   - This platform claims to mirror content from Sci-Hub and LibGen, preserving a vast number of books and papers.



**Link mentioned**: <a href="https://annas-archive.org/">Anna’s Archive</a>: no description found

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1339033447716491317)** (6 messages): 

> `Reasoning over Latent Spaces, Multilingual Language Models, Llama-2 and Linguistic Bias, Latent Languages in Transformers` 


- **Exploring Reasoning over Latent Spaces**: *Pyro99x* initiated interest in discussing reasoning over latent spaces, highlighting a fresh angle on AI model functioning.
   - This sparked immediate community engagement, with a member expressing willingness to partake in the discussion soon.
- **Analyzing Multilingual Models: Llama-2**: Today’s focus is on the [paper](https://arxiv.org/abs/2402.10588) questioning whether multilingual language models use English as an internal pivot language.
   - The study tracks how intermediate embeddings in Llama-2 transform through layers, revealing biases in language processing.
- **Cited Studies on Latent Languages**: A member referenced a related paper claiming that models trained on balanced corpora can pivot between multiple languages based on the target.
   - *Bhagawanpanditi* pondered whether this phenomenon could extend to more languages, presenting a trade-off between using a dominant language versus multiple.



**Link mentioned**: <a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...

  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1339238950555877377)** (4 messages): 

> `Tokenization Issues, Character Representation in LLMs` 


- **Tokenization: A Problematic Viewpoint**: A member argued that counting **tokens** might be the real issue since models don't count tokens in learning data, rather they count **characters**.
   - *What do you mean with that?* another member inquired, seeking clarity on this perspective.
- **Understanding Token Representation**: Another member explained that LLMs only perceive **tokens** in training data, noting that while some tokens represent a single character, most represent combinations of multiple characters.
   - They added that when spaced out, each character is treated as its own token, highlighting an important aspect of tokenization.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1338985003878715453)** (12 messages🔥): 

> `DeepScaleR Model, EU AI Funding, Thomson Reuters Copyright Case, OpenAI Roadmap Update, Literature Review Tools` 


- **DeepScaleR Surpasses Expectations**: A user expressed their excitement about the [DeepScaleR preview](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scali) featuring a 1.5B model that demonstrates significant advancements in scaling RL.
   - *RL is back baby!* resonated as a comment on this development.
- **EU Invests Big in AI**: The European Union pledged **200 billion euros** to invest in AI to catch up with the U.S. and China, emphasizing the need for **AI gigafactories** for model training.
   - European Commission President Ursula von der Leyen declared a goal for Europe to be a leading AI continent at the [AI Action Summit in Paris](https://www.msn.com/en-us/money/companies/eu-pledges-200-billion-in-ai-spending-in-bid-to-catch-up-with-u-s-china/ar-AA1yO0Su).
- **Thomson Reuters Wins AI Copyright Battle**: [Thomson Reuters won](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/) a landmark AI copyright case against Ross Intelligence, affirming that their copyright was infringed.
   - Judge Stephanos Bibas stated, *None of Ross’s possible defenses holds water,* emphasizing the seriousness of copyright infringement in AI applications.
- **OpenAI Reveals Future Model Plans**: In a recent update, OpenAI announced that **GPT-4.5** will be released as their last non-chain-of-thought model, followed by an integration of the **o-series and GPT-series models**.
   - OpenAI aims for their models to *just work*, with GPT-5 offering features across various applications, reducing complexity for users.
- **Efficient Literature Review Tool Available**: A GitHub repository titled [Deep-Research-Arxiv](https://github.com/GitsSaikat/Deep-Research-Arxiv) was shared, designed for fast and reliable literature reviews.
   - Users can also access the application on [Hugging Face Spaces](https://huggingface.co/spaces/AlignAI/Deep-Research-Arxiv) to streamline their research process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/AlignAI/Deep-Research-Arxiv">Deep Research Arxiv - a Hugging Face Space by AlignAI</a>: no description found</li><li><a href="https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/">Thomson Reuters Wins First Major AI Copyright Case in the US</a>: The Thomson Reuters decision has big implications for the battle between generative AI companies and rights holders.</li><li><a href="https://github.com/GitsSaikat/Deep-Research-Arxiv">GitHub - GitsSaikat/Deep-Research-Arxiv: Do literature review Fast, Simple and Reliable</a>: Do literature review Fast, Simple and Reliable. Contribute to GitsSaikat/Deep-Research-Arxiv development by creating an account on GitHub.</li><li><a href="https://news.slashdot.org/story/25/02/11/1617259/eu-pledges-200-billion-in-ai-spending-in-bid-to-catch-up-with-us-china">EU Pledges $200 Billion in AI Spending in Bid To Catch Up With US, China - Slashdot</a>: The European Union pledged to mobilize 200 billion euros ($206.15 billion) to invest in AI as the bloc seeks to catch up with the U.S. and China in the race to train the most complex models. From a re...</li><li><a href="https://x.com/sama/status/1889755723078443244?t=EgnihPXVoD2fsS9ag5u5SA&s=19">Tweet from Sam Altman (@sama)</a>: OPENAI ROADMAP UPDATE FOR GPT-4.5 and GPT-5:We want to do a better job of sharing our intended roadmap, and a much better job simplifying our product offerings.We want AI to “just work” for you; we re...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1339123140017782805)** (20 messages🔥): 

> `CUDA backend on Windows, PR feedback and testing issues, Windows CI environment variable propagation, Testing failures with recursion vs iteration approach` 


- **Progress on Windows CUDA Backend**: A user confirmed getting the **CUDA backend working on Windows** by correcting the autogen files with appropriate DLL names, although they observed that standard CI runners lack GPU support.
   - They suggested possibly hard-coding the CUDA version to keep the setup simple, but concerns were raised about overall test coverage.
- **Calls for Feedback on a Closed PR**: A pull request concerning bug fixes was closed without comments, prompting a request for feedback on its contents and rationale.
   - Another user highlighted an indentation error as the primary reason for CI test failures, hinting that testing before pushing might have been overlooked.
- **Inconsistency in CI with Environment Variables**: Discussions revealed that the **Windows CI was not propagating backend environment variables** between steps, leading to a default switch to CLANG during testing.
   - A pull request was initiated to ensure that environment variables persist between CI steps for proper functionality.
- **Concerns Over Testing Changes in Implementation**: There were doubts expressed about the efficacy of switching from recursion to iteration, indicating that it caused many tests to fail beyond just the original changes.
   - It was noted that the immediate cause of CI failures stemmed from an indentation issue that inadvertently affected critical functionality within the code.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/9036">Fix TestLinearizerFailures.test_failure_53 by bruari · Pull Request #9036 · tinygrad/tinygrad</a>: Fix TestLinearizerFailures.test_failure_53 bug as per bounty spreadsheet.</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/13280542105/job/37077817692?pr=9039#step:5:17">Check the used device on Windows in CI is the matrix backend being te… · tinygrad/tinygrad@a4e6599</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Check the used device on Windows in CI is the matrix backend being te… · tinygrad/tinygrad@a4e6599</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9047">Ensure Windows CI correctly tests the specified backends  by rmtew · Pull Request #9047 · tinygrad/tinygrad</a>: Ensure that the set backend environment variable is persisted to the next step via $GITHUB_ENVIt doesn&amp;#39;t actually persist for Windows unless shell is explicitly set to bash.Add the assertion ....</li><li><a href="https://github.com/rmtew/tinygrad/blob/feature-windows-cuda/.github/workflows/test.yml#L615">tinygrad/.github/workflows/test.yml at feature-windows-cuda · rmtew/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - rmtew/tinygrad
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1339182995327815720)** (5 messages): 

> `Graph Scheduler Tests, Tinygrad vs PyTorch` 


- **Inquiry on Graph Scheduler Tests**: A member asked which test generated a specific graph related to **ASSIGN's** in the messages, indicating a need for clarification.
   - Another member noted that they had reverted a change due to Python's performance issues, hinting at upcoming improvements.
- **Debate on Tinygrad vs Traditional Frameworks**: A user questioned the advantages of switching to **tinygrad** from established frameworks like **PyTorch**, citing personal experience with the latter.
   - Another member suggested that choosing tinygrad could lead to **cheaper hardware**, a better understanding of underlying processes, and potentially faster model performance.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1339284317175681236)** (2 messages): 

> `Open Source Engineer Position, Agentic Document Workflows, Nomic AI Embedding Model` 


- **Exciting Job Opening for Open Source Engineer**: A full-time position for an **open source engineer** at [@llama_index](https://twitter.com/llama_index) has been announced, targeting those passionate about **Python** and **AI**.
   - The role emphasizes expanding the **llama_index** framework amidst growing capabilities, with further details available [here](https://t.co/WMgdaauxP8).
- **Nomic AI Enhances Document Workflows**: The latest work from [@nomic_ai](https://twitter.com/nomic_ai) showcases the importance of a great **embedding model** for effective **Agentic Document Workflows**.
   - This new development has been positively received, marking a significant step in enhancing these workflows, with more details shared [here](https://t.co/pezsylHNpH).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1339168356309143582)** (17 messages🔥): 

> `Building RAG systems, Batch processing PDFs, Creating query engine tools, Exhaustive RAG search methods, Vector databases` 


- **Exploring data loaders for RAG systems**: Members discussed the desire to practice different data loaders for building **RAG systems**, with suggestions to explore the [llamahub](https://llamahub.example) website for resources.
   - One member highlighted the importance of choosing loaders that fit individual use cases.
- **Batch processing PDFs inquiry**: A member inquired about methods to **batch process PDFs**, prompting another member to ask for clarification on the specific approach being considered.
   - The discussion hints at the need for more targeted tools or scripts to handle bulk PDF operations.
- **Creating query engine tools with filters**: A member sought advice on utilizing **predefined filters** within query engine tools for different topics, aiming for an efficient workflow without creating multiple indexes.
   - Another member provided a code example to demonstrate how to implement a query engine tool with specified filters.
- **Best methods for exhaustive RAG search**: A member questioned the best approach for conducting an **exhaustive RAG search** when retrieving a range of data, acknowledging existing methods like autorag and query synthesizing.
   - This highlights an interest in exploring innovative search techniques to cover potentially extensive data chunks.
- **Choices in vector databases**: Members shared their experiences with different **vector databases**, with one member mentioning their use of **Milvus** and another mentioning the use of **Redis** in a Docker container.
   - This reflects a community interest in the various tools available for managing vector data.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1339295436812587038)** (1 messages): 

> `LLM Agents MOOC Hackathon, Global participation, Winning teams announcement, Top represented countries and universities, Hackathon website` 


- **LLM Agents MOOC Hackathon Winners Revealed**: Excited to announce the winning teams of the **LLM Agents MOOC Hackathon** with amazing participation from the community, featuring ~3,000 participants from [127 countries](https://x.com/dawnsongtweets/status/1889686697564315963).
   - Prof. Dawn Song expressed gratitude for the community's enthusiasm in her [Twitter announcement](https://x.com/dawnsongtweets/status/1889686697564315963).
- **Global Representation in Hackathon**: The hackathon saw contributions from **1,100+ universities** and **800+ companies**, showcasing strong global interest in AI.
   - Prominent representation included top countries like the **US**, **India**, and **China**, along with notable universities such as UC Berkeley and Stanford.
- **Top Companies at the Hackathon**: Participants included top companies such as **Amazon**, **Microsoft**, **Samsung**, and **Salesforce**, reflecting industry engagement.
   - This diverse representation highlights the hackathon's significance in bridging academia and industry.
- **Hackathon Website for Winning Teams**: Interested parties can view the winning teams and their submissions on the [hackathon website](https://rdi.berkeley.edu/llm-agents-hackathon/).
   - This platform serves as a hub for showcasing the innovation and creativity displayed during the event.



**Link mentioned**: <a href="https://x.com/dawnsongtweets/status/1889686697564315963)">Tweet from Dawn Song (@dawnsongtweets)</a>: 🎉 Excited to announce the winning teams of LLM Agents MOOC Hackathon! We’re thrilled by the amazing participation and enthusiasm from the global AI community:🌍 ~3,000 participants from 127 countries...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1339296200913850450)** (1 messages): 

> `Spring 2025 MOOC Launch, Advanced LLM Topics, Dawn Song Announcement, Course Attendance Stats` 


- **Spring 2025 MOOC officially launched**: The **Spring 2025 MOOC** has been officially announced to the greater AI community, focusing on **Advanced LLM Agents**.
   - Participants are encouraged to **retweet and share** Professor Dawn Song's [announcement on Twitter](https://x.com/dawnsongtweets/status/1889355520294944829) to spread the word.
- **Advanced topics in the upcoming MOOC**: This semester, the MOOC will cover advanced topics such as **Reasoning & Planning**, **Multimodal Agents**, and **AI for Mathematics**.
   - Additional focuses will include **Agent Safety & Security**, providing a comprehensive exploration for AI enthusiasts.
- **Course attendance statistics**: Building on the success of the **Fall 2024** MOOC, which saw **15K+ registered learners** and **200K+ lecture views** on YouTube, this course is expected to attract further interest.
   - The previous offering also amassed around **9K Discord members**, highlighting its strong community engagement.
- **Weekly live sessions set**: The MOOC will feature live sessions every **Monday at 4:10 PM PT** for interactive learning.
   - This format is designed to foster engagement for students, researchers, developers, and AI practitioners alike.



**Link mentioned**: <a href="https://x.com/dawnsongtweets/status/1889355520294944829)">Tweet from Dawn Song (@dawnsongtweets)</a>: Really excited to announce our Advanced LLM Agents MOOC (Spring 2025)!Building on the success of our LLM Agents MOOC from Fall 2024 (15K+ registered learners, ~9K Discord members, 200K+ lecture views ...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1338997446612881448)** (5 messages): 

> `MOOC Curriculum Updates, Hackathon Participation` 


- **Upcoming MOOC Curriculum Details**: More details about the **MOOC curriculum** will be released soon, with expectations set for around **two weeks out**.
   - *Thank you for your patience!*
- **No Hackathon this Semester**: A new student inquired about the possibility of a **hackathon this semester**, akin to last semester's event.
   - However, it was clarified that **there is no hackathon planned**, though past events led by Prof. Song may indicate potential future opportunities.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1339127002786959395)** (2 messages): 

> `MooC student applications, Research subjects` 


- **MooC Students in Research Applications?**: A member inquired whether MooC students can apply for research subjects and asked for access to the Google form.
   - Another member responded that **more details will be released soon**, expressing appreciation for the patience of the inquirers.
- **Accessing Research Subjects Information**: The conversation indicated an interest in how MooC students could gain access to the necessary forms for research subjects.
   - Responses suggest that clarity and further information will be provided in the near future.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1338996056800628858)** (3 messages): 

> `DeepScaleR Model, Assignments Deadline` 


- **DeepScaleR Surpasses O1 Preview**: The [DeepScaleR model](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) has been documented to surpass the O1 preview using a **1.5B model** by scaling reinforcement learning techniques.
   - This advancement signifies the potential for improved performance in AI modeling.
- **Assignments Deadline Coming Soon**: In response to a query about the assignment deadlines, a member confirmed that details will be released soon.
   - *Thank you for your patience!* A reminder for those catching up on missed lectures.



**Link mentioned**: <a href="https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1338995758346666056)** (6 messages): 

> `Steam Gift Card Giveaway, Voice Model Updates, TextWeb-UI Installation, Session Memory Management, Mobile Application Usability` 


- **Steam Gift Card Giveaway Alert**: A member announced a **$50 Steam gift card giveaway** with a link to participate: [steamcommunity.com/gift-card/pay/50](https://u.to/kR7WIQ).
   - Chat activity following this included one member calling it **spam**, indicating mixed reactions.
- **Current Voice Model Inquiry**: A member inquired about the availability of a **current voice model** for use, indicating interest in updates.
   - This sparked further discussion on potential options and usability.
- **TextWeb-UI Installation Challenges**: It was mentioned that **TextWeb-UI** requires a complex installation process, prompting a response about ease of use.
   - One user noted that it's not an easy `.exe` install and warned about its requirements.
- **Managing Session Memories via Python**: A discussion arose about why the backend couldn't feed **session memories** from SQL to a Python script, emphasizing potential for usage.
   - Participants conveyed interest in exploring different ways to enhance session management.
- **Mobile Application Performance Concerns**: Concerns were raised about using mobile applications for both **iOS and Android**, especially regarding battery life during use.
   - One member speculated that using such applications could drain a **device's battery in 1 hour**, indicating performance issues.


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1339143819651973160)** (2 messages): 

> `Login Issues, API Request Blocking` 


- **Failed to Fetch Error on Login**: A user reported receiving a 'Failed to fetch' error when attempting to log into their personal account with their credentials.
   - *Not very informative* was the feedback regarding this issue, prompting inquiries about possible filtering that could be blocking API requests.
- **Concerns About API Request Filtering**: A member raised a question about whether some sort of filtering might be causing the failure in API requests during the login attempt.
   - This suggests a deeper investigation may be needed to identify connectivity issues or software restrictions.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1339139551989596160)** (1 messages): 

> `Podcasting Workshop, AI Voice Generation, ElevenLabs and PlayHT` 


- **Unlocking Podcast Success with AI**: Join our FREE workshop on **Thursday, Feb 13 at 9PM IST** to discover how creators are launching podcasts using just **AI** and no expensive equipment.
   - Participants will learn **fundamentals of AI audio models** and get hands-on experience with platforms like [ElevenLabs](https://elevenlabs.io) and [PlayHT](https://playht.com) to create engaging audio content.
- **Hands-On Experience in Audio Creation**: Attendees will gain **hands-on experience** with leading voice generation platforms, allowing them to **transform text into audio content** effortlessly.
   - The workshop will also cover how to develop their own **open source NotebookLM** for custom implementations.
- **Build Fast with AI Resources**: Join the [Build Fast With AI](https://t.me/BuildFastWithAI) community for free resources and tools dedicated to **generative AI solutions**.
   - Run by IIT Delhi alumni, the group offers **latest Gen AI tools**, roadmaps, and workshop links to help attendees build innovative AI applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/wlnvyebn?tk=eMLWC6">Create Your Own Podcast Studio with AI 🎙️ · Zoom · Luma</a>: Want to start a podcast but don&#x27;t have professional recording equipment? Curious about how AI can be your personal voice artist?Join us for an exciting…</li><li><a href="https://t.me/BuildFastWithAI">Build Fast With AI - Free AI Resources</a>: Build Fast With AI is a Generative AI focused start-up run by IIT Delhi alumni to deliver cutting-edge Gen AI solutions.		--&gt; Latest Gen AI Tools	--&gt; Gen AI roadmap &amp; materials	--&gt; Worksh...
</li>
</ul>

</div>
  

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
