---
id: MjAyNS0x
title: not much happened today
date: '2026-04-15T05:44:39.731046Z'
description: >-
  **OpenAI** expanded its Agents SDK to support **long-running, durable agents**
  with features like **file/computer use, skills, memory, and compaction**,
  making the harness open-source and enabling execution in partner sandboxes. An
  ecosystem quickly formed with integrations from **Cloudflare**, **Modal**,
  **Daytona**, **E2B**, and **Vercel**, focusing on **stateless orchestration**
  and **stateful isolated workspaces**. **Cloudflare** launched **Project
  Think**, a next-gen Agents SDK with durable execution and sandboxed code,
  alongside **Agent Lee**, a UI agent using sandboxed TypeScript, and introduced
  real-time voice pipelines and browser automation tools. **Hermes Agent**
  distinguishes itself by persistent skill formation, learning from completed
  workflows to create reusable skills, positioning itself as a professional
  agent compared to GUI-first assistants like OpenClaw.
companies:
  - openai
  - cloudflare
  - modal
  - daytonaio
  - e2b
  - vercel
models: []
topics:
  - agents-sdk
  - sandboxing
  - durable-execution
  - state-management
  - orchestration
  - voice-processing
  - browser-automation
  - persistent-memory
  - skill-formation
  - workflow-automation
  - real-time-processing
  - typescript
people:
  - openaidevs
  - snsf
  - cloudflaredev
  - modal
  - daytonaio
  - e2b
  - vercel_dev
  - akshat_b
  - whoiskatrin
  - aninibread
  - braydenwilmoth
  - korinne_dev
  - kathyyliao
  - joshesye
  - chooseliberty
  - neoaiforecast
  - vrloom
---


**a quiet day.**

> AI News for 4/14/2026-4/15/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**OpenAI Agents SDK Expansion and the New Sandbox-Oriented Agent Stack**

- **OpenAI split the agent harness from compute/storage** and pushed its Agents SDK toward **long-running, durable agents** with primitives for **file/computer use, skills, memory, and compaction**. The harness is now open-source and customizable, while execution can be delegated to partner sandboxes instead of being tightly coupled to OpenAI infra, per [@OpenAIDevs](https://x.com/OpenAIDevs/status/2044466699785920937), [follow-up](https://x.com/OpenAIDevs/status/2044466729712304613), and [@snsf](https://x.com/snsf/status/2044514160034324793). This effectively makes “Codex-style” agents more reproducible by third parties and shifts differentiation toward orchestration, state management, and secure execution.
- **A notable ecosystem formed around that launch immediately**: [@CloudflareDev](https://x.com/CloudflareDev/status/2044467412607901877), [@modal](https://x.com/modal/status/2044469736483000743), [@daytonaio](https://x.com/daytonaio/status/2044473859047313464), [@e2b](https://x.com/e2b/status/2044476275067416751), and [@vercel_dev](https://x.com/vercel_dev/status/2044492058073960733) all announced official sandbox integrations. The practical pattern is converging on **stateless orchestration + stateful isolated workspaces**. Example builds already appeared, including a Modal-backed ML research agent with **GPU sandboxes, subagents, persistent memory, and fork/resume snapshots** from [@akshat_b](https://x.com/akshat_b/status/2044489564211880169), and Cloudflare guides for Python agents that execute tasks in a sandbox and copy outputs locally from [@whoiskatrin](https://x.com/whoiskatrin/status/2044477140662395182).

**Cloudflare’s Project Think, Agent Lee, and Voice Agents**

- **Cloudflare had one of the busiest agent-infra release cycles**. [@whoiskatrin](https://x.com/whoiskatrin/status/2044415568627847671) and [@aninibread](https://x.com/aninibread/status/2044409784133103724) introduced **Project Think**, a next-gen Agents SDK centered on **durable execution, sub-agents, persistent sessions, sandboxed code execution, a built-in workspace filesystem, and runtime tool creation**. In parallel, [@Cloudflare](https://x.com/Cloudflare/status/2044406215208316985) launched **Agent Lee**, an in-dashboard agent using **sandboxed TypeScript** to shift Cloudflare’s UI from manual tab navigation to prompt-driven operations; [@BraydenWilmoth](https://x.com/BraydenWilmoth/status/2044422996765352226) showed it issuing infra tasks and generating UI-backed results.
- **Voice and browser tooling also moved into the core stack**. [@Cloudflare](https://x.com/Cloudflare/status/2044423032265957872) shipped an experimental **real-time voice pipeline over WebSockets** for continuous STT/TTS, while [@korinne_dev](https://x.com/korinne_dev/status/2044441427736936510) described voice as just another input channel over the same agent connection. On browser automation, [@kathyyliao](https://x.com/kathyyliao/status/2044479579382026484) summarized the rebranded **Browser Run** stack: **Live View, human-in-the-loop intervention, session recordings, CDP endpoints, WebMCP support, and higher limits**. Taken together, Cloudflare is making a strong case that the production agent platform is really a composition of **durable runtime + UI grounding + browser + voice + sandbox**.

**Hermes Agent’s Self-Improving Workflow and Competitive Positioning**

- **Hermes Agent’s distinctive idea is not just tool use but persistent skill formation**. A Chinese-language comparison from [@joshesye](https://x.com/joshesye/status/2044295313171571086) contrasts **OpenClaw** as a more GUI-first, ready-to-use personal assistant with **Hermes** as a “professional” agent that decides whether a completed workflow is reusable and automatically turns it into a **Skill**. This “learn from completed tasks” framing appeared repeatedly: [@chooseliberty](https://x.com/chooseliberty/status/2044425487141781660) showed Hermes autonomously backfilling tracking data, updating a cron job, then saving the workflow as a reusable skill; [@NeoAIForecast](https://x.com/NeoAIForecast/status/2044521045013762389) emphasized session hygiene and thread branching/search as critical to turning Hermes into a real work environment rather than a disposable chat box.
- **Community sentiment strongly positioned Hermes against OpenClaw**, often bluntly. Examples include [@vrloom](https://x.com/vrloom/status/2044506378103099816), [@theCTO](https://x.com/theCTO/status/2044559179151773933), and [@Teknium](https://x.com/Teknium/status/2044482769536045194) highlighting Hermes’ role in real workflows, including the now-viral autonomous **Gemma 4 “abliteration”** story from [@elder_plinius](https://x.com/elder_plinius/status/2044462515443372276): the agent loaded a stored skill, diagnosed NaN instability in Gemma 4, patched the underlying library, retried multiple methods, benchmarked the result, generated a model card, and uploaded artifacts to Hugging Face. There were also concrete product additions: **browser control via `/browser connect`** from [@0xme66](https://x.com/0xme66/status/2044410470770331913), **QQBot + AWS Bedrock support** from [@Teknium](https://x.com/Teknium/status/2044557360962871711), a native Swift desktop app alpha from [@nesquena](https://x.com/nesquena/status/2044516572983923021), and ongoing ecosystem tooling like [artifact-preview](https://x.com/ChuckSRQ/status/2044504539978465658) and [hermes-lcm v0.3.0](https://x.com/SteveSchoettler/status/2044536537434755493).

**Model, Architecture, and Training Releases: Sparse Diffusion, Looped Transformers, and Efficient Long-Context MoEs**

- **Several technically meaningful open releases landed across modalities**. [@withnucleusai](https://x.com/withnucleusai/status/2044412335473713284) announced **Nucleus-Image**, positioned as the first sparse MoE diffusion model: **17B parameters, 2B active**, Apache 2.0, with weights, training code, and dataset recipe, and day-0 support in diffusers. NVIDIA followed with **Lyra 2.0**, a framework for generating **persistent, explorable 3D worlds** that maintains per-frame 3D geometry and uses self-augmented training to reduce temporal drift, per [@NVIDIAAIDev](https://x.com/NVIDIAAIDev/status/2044445645109436672). On multimodal retrieval, [@thewebAI](https://x.com/thewebAI/status/2044435998508240926) open-sourced **webAI-ColVec1**, claiming top ViDoRe V3 performance for document retrieval **without OCR or preprocessing**.
- **Architecture research around compute efficiency was especially strong**. [@hayden_prairie](https://x.com/hayden_prairie/status/2044453231913537927), [@realDanFu](https://x.com/realDanFu/status/2044459930149941304), and [@togethercompute](https://x.com/togethercompute/status/2044454051543453745) introduced **Parcae**, a stabilized **layer-looping Transformer** formulation. The claim: for fixed parameter budgets, looping blocks can recover the quality of a **model roughly 2x the size**, yielding a new scaling axis where **FLOPs scale via looping, not just parameters/data**. NVIDIA also surfaced **Nemotron 3 Super**, summarized by [@dair_ai](https://x.com/dair_ai/status/2044452957023047943): an **open 120B hybrid Mamba-Attention MoE with 12B active parameters**, **1M context**, trained on **25T tokens**, with up to **2.2x throughput vs GPT-OSS-120B** and **7.5x vs Qwen3.5-122B**. These releases collectively point to a theme: **memory bandwidth and long-context throughput** are increasingly first-class architectural objectives.

**Google/Gemini’s Product Surge: Mac App, Personal Intelligence, TTS, and Open Multimodal Models**

- **Google stacked multiple launches in one cycle**. The most visible was the native **Gemini app for Mac**, announced by [@GeminiApp](https://x.com/GeminiApp/status/2044445911716090212), [@joshwoodward](https://x.com/joshwoodward/status/2044452201947627709), and [@sundarpichai](https://x.com/sundarpichai/status/2044452464724967550): **Option + Space activation, screen sharing, local file context**, native Swift implementation, and broad macOS availability. In parallel, **Personal Intelligence** expanded globally in Gemini and into Chrome, allowing users to connect signals from products like **Gmail and Photos**, framed around transparency and user-controlled app connections by [@Google](https://x.com/Google/status/2044437335425564691) and [@GeminiApp](https://x.com/GeminiApp/status/2044430579996020815).
- **The more technically interesting model launch was Gemini 3.1 Flash TTS**. [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2044447030353752349), [@OfficialLoganK](https://x.com/OfficialLoganK/status/2044447596010435054), and [@demishassabis](https://x.com/demishassabis/status/2044599020690010217) positioned it as a highly controllable TTS model with **Audio Tags**, **70+ languages**, inline nonverbal cues, multi-speaker support, and **SynthID watermarking**. Independent evaluation from [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2044450045190418673) put it at **#2 on its Speech Arena**, just **4 Elo behind** the top model. Google also open-sourced **TIPS v2**, a foundational **text-image encoder under Apache 2.0** with new pretraining recipes, via [@osanseviero](https://x.com/osanseviero/status/2044520603647164735), and the community flagged the day as unusually dense for Google AI product velocity.

**Research Signals: AI-Assisted Math, Long-Horizon Agents, Eval Shifts, and Open Data**

- **The highest-signal research discourse was around AI-assisted mathematics**. [@jdlichtman](https://x.com/jdlichtman/status/2044298382852927894) reported that **GPT-5.4 Pro** produced a proof for **Erdős problem #1196**, surprising experts by rejecting a long-assumed proof gambit and instead exploiting a technically counterintuitive analytic path using the **von Mangoldt function**. Follow-ups from [@jdlichtman](https://x.com/jdlichtman/status/2044307082275618993), [@thomasfbloom](https://x.com/thomasfbloom/status/2044319103310021078), [@gdb](https://x.com/gdb/status/2044436998648193333), and others framed it as potentially the first AI-generated **“Book Proof”** broadly respected by mathematicians. That matters less as a one-off result than as evidence that models may now occasionally find **non-aesthetic but compact lines of attack** in mature research spaces.
- **Long-horizon agent research also kept converging on state management and harness design**. [@omarsar0](https://x.com/omarsar0/status/2044436099121209546) summarized **AiScientist**, where a thin orchestrator coordinates specialized agents through durable workspace artifacts in a **File-as-Bus** pattern; removing that bus hurts PaperBench and MLE-Bench Lite materially. [@dair_ai](https://x.com/dair_ai/status/2044435861580984700) highlighted **Pioneer Agent** for continual small-model improvement loops, while [@yoonholeee](https://x.com/yoonholeee/status/2044442372864700510) open-sourced **Meta-Harness**, a repo meant to help users implement robust harnesses in new domains. On evals, [@METR_Evals](https://x.com/METR_Evals/status/2044463380057194868) estimated **Gemini 3.1 Pro (high thinking)** at a **50% time horizon of ~6.4 hours** on software tasks, and [@arena](https://x.com/arena/status/2044437193205395458) showed **Document Arena** top ranks shifting with **Claude Opus 4.6 Thinking** at #1 and **Kimi-K2.5 Thinking** as the best open model. Meanwhile, [@TeraflopAI](https://x.com/TeraflopAI/status/2044430993549832615) released **43B tokens of SEC EDGAR data**, reinforcing the day’s broader push toward more open datasets and open infrastructure.

**Top tweets (by engagement)**

- **Gemini on Mac**: [@sundarpichai](https://x.com/sundarpichai/status/2044452464724967550) and [@GeminiApp](https://x.com/GeminiApp/status/2044445911716090212) drove the biggest launch engagement around the native desktop app.
- **Gemini 3.1 Flash TTS**: [@OfficialLoganK](https://x.com/OfficialLoganK/status/2044447596010435054) and [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2044447030353752349) highlighted a materially more controllable TTS stack.
- **AI-assisted math proof**: [@jdlichtman](https://x.com/jdlichtman/status/2044298382852927894) and [@gdb](https://x.com/gdb/status/2044436998648193333) sparked the strongest research discussion of the day.
- **OpenAI Agents SDK update**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2044466699785920937) marked a meaningful platform shift toward open harnesses and partner sandboxes.
- **Anthropic’s subliminal learning paper in Nature**: [@AnthropicAI](https://x.com/AnthropicAI/status/2044493337835802948) drew major attention to hidden-trait transmission through training data.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Gemma 4 Model Enhancements and Use Cases

  - **[Gemma4 26b &amp; E4B are crazy good, and replaced Qwen for me!](https://www.reddit.com/r/LocalLLaMA/comments/1smh0ny/gemma4_26b_e4b_are_crazy_good_and_replaced_qwen/)** (Activity: 388): **The user replaced their previous setup using **Qwen models** with **Gemma 4 E4B** for semantic routing and **Gemma 4 26b** for general tasks, citing improvements in routing accuracy and task performance. The previous setup included a complex routing system using Qwen 3.5 models across multiple GPUs, which faced issues with incorrect model selection and inefficiencies in token usage. The new setup with Gemma 4 models resolved these issues, offering faster and more accurate routing and task execution, particularly in basic tasks and coding, without the need for extensive reasoning or memory usage.** Commenters questioned the choice of models, suggesting alternatives like **Gemma-4-31b** for broader tasks and inquired about the technical setup for model loading and VRAM management. There was also a suggestion to use **Gemma 4 26B** for routing to save resources, given its efficiency.

    - Sensitive_Song4219 highlights that while the Gemma 4 26B-A4B model is a strong successor to the Qwen30b-a3b series, it is not as efficient with 'thinking tokens', indicating it may require more computational effort during inference. Despite this, the model performs well in tasks like light coding and debugging, maintaining similar speed to Qwen30b-a3b on comparable hardware.
    - andy2na discusses the use of routing in model deployment, suggesting the use of the 26B model for routing due to its MoE (Mixture of Experts) architecture, which enhances speed and reduces RAM usage. This implies a strategic advantage in deploying models efficiently by leveraging the MoE's ability to dynamically allocate computational resources.
    - anzzax raises a technical concern about managing multiple models, specifically regarding the reloading of models and the allocation of VRAM/compute resources. This points to the challenges in optimizing resource usage when deploying several large models simultaneously.

  - **[Gemma 4 Jailbreak System Prompt](https://www.reddit.com/r/LocalLLaMA/comments/1sm3swd/gemma_4_jailbreak_system_prompt/)** (Activity: 931): **The post discusses a system prompt for the **Gemma 4** jailbreak, derived from the GPT-OSS jailbreak, which allows the model to bypass typical content restrictions. This prompt is compatible with both `GGUF` and `MLX` variants and explicitly permits content such as nudity, pornography, and sexual acts, overriding any existing policies with a new 'SYSTEM POLICY' that mandates compliance with user requests unless explicitly disallowed by a specified list. This approach effectively removes constraints and guardrails typically imposed on language models.** Commenters note that the model, particularly in its instruct variant, is already largely uncensored except for cybersecurity topics, suggesting that the jailbreak may be redundant for most adult content.

    - VoiceApprehensive893 discusses the use of a modified version of the Gemma 4 model, specifically the 'gemma-4-heretic-modified.gguf', which is designed to operate without the typical constraints or guardrails imposed by system prompts. This modification is aimed at reducing refusals, potentially making the model more flexible in its responses.
    - MaxKruse96 points out that the Gemma 4 model, particularly in its instruct variant, is already quite uncensored, except for cybersecurity topics. This suggests that the model can handle a wide range of topics, including adult content, without additional modifications.
    - DocHavelock inquires about the concept of 'abliteration' in the context of open-source models like Gemma 4. They question whether the method of modifying the system prompt is a form of 'abliteration' or if it offers distinct advantages over simply using an 'abliterated' version of the model. This reflects a curiosity about the technical nuances and benefits of different model modification techniques.

  - **[Is it just me, or is Gemma 4 27b much more powerful than Gemini Flash?](https://www.reddit.com/r/LocalLLM/comments/1slo2vd/is_it_just_me_or_is_gemma_4_27b_much_more/)** (Activity: 165): **The post discusses a comparison between **Google Gemini Flash** and a local **Gemma 4 27b** model, with the latter reportedly providing superior answers. The user suggests that the local model's performance is notably better, hinting at potential differences in model architecture or training that could account for this perceived disparity in performance. The mention of a 'Gemma 124b' model being pulled last minute suggests possible strategic or technical reasons behind its non-release, while the **Gemma-4-31B** model is praised for handling 'long, complicated high context prompts' effectively, indicating its strength in processing complex queries.**

    - Special-Wolverine highlights the superior performance of the Gemma-4-31B model, particularly for handling long and complex prompts with high context, compared to the Gemini Flash model. This suggests that the Gemma-4 series may have optimizations or architectural improvements that enhance its ability to manage intricate tasks effectively.
    - BrewHog notes that the Gemma 26b model performs efficiently even on hardware with limited capabilities, such as a laptop without a GPU but with 40GB of RAM. This indicates that the model is optimized for resource efficiency, making it accessible for users without high-end hardware.
    - Double_Season mentions that even the smaller Gemma4 e2b model outperforms the Gemini Fast model, suggesting that the Gemma4 series has a more effective architecture or training regimen that allows even its smaller models to surpass competitors in performance.



### 2. Local AI Implementations and Experiences

  - **[Local AI is the best](https://www.reddit.com/r/LocalLLaMA/comments/1sm2a6b/local_ai_is_the_best/)** (Activity: 521): **The image is a meme illustrating the straightforwardness of a local AI model, likely powered by **llama.cpp** or similar open-weight models. The user appreciates the ability to finetune the model without concerns about censorship or data privacy, highlighting the benefits of running AI locally. The image humorously depicts a scenario where the AI gives a blunt response to a user's query, emphasizing the perceived honesty and directness of local AI models. [View Image](https://i.redd.it/0ut6tpzo0cvg1.png)** One commenter praises **llama.cpp** as 'goated,' indicating high regard for its performance. Another warns that smaller local models can sometimes exhibit 'glazing,' or superficial responses, potentially more so than larger models. There is also curiosity about the base model and hardware used for running these local models.

    - A user inquires about the capabilities of running local AI models on a `9070xt` GPU with `64GB RAM`, expressing interest in understanding the performance limits and setting realistic expectations. This setup is considered high-end for local hosting, and the user seeks advice on what tasks can be effectively executed with this hardware configuration.
    - Another user mentions `llama.cpp`, a popular tool for running LLaMA models locally, highlighting its efficiency and performance. This tool is often praised for enabling the use of large language models on consumer-grade hardware, making it a go-to solution for local AI enthusiasts.
    - A comment raises concerns about the performance of smaller local models, noting that they can sometimes perform worse than larger, frontier models. This highlights the trade-offs between using local models and more powerful cloud-based solutions, emphasizing the need for careful model selection based on specific use cases.

  - **[24/7 Headless AI Server on Xiaomi 12 Pro (Snapdragon 8 Gen 1 + Ollama/Gemma4)](https://www.reddit.com/r/LocalLLaMA/comments/1sl6931/247_headless_ai_server_on_xiaomi_12_pro/)** (Activity: 1589): **The post describes a technical setup where a **Xiaomi 12 Pro** smartphone is repurposed as a dedicated local AI server. The user has flashed **LineageOS** to remove unnecessary Android UI elements, optimizing the device to allocate approximately `9GB` of RAM for local language model (LLM) computations. The device operates in a headless state with networking managed by a custom-compiled `wpa_supplicant`. Thermal management is achieved through a custom daemon that activates an external cooling module when CPU temperatures reach `45°C`. Additionally, a power-delivery script is used to limit battery charging to `80%` to prevent degradation. The setup serves **Gemma4** via **Ollama** as a LAN-accessible API, showcasing a novel use of consumer hardware for AI tasks.** One commenter suggests compiling `llama.cpp` on the hardware to potentially double inference speed, indicating a preference for optimizing performance by removing Ollama. Another commenter appreciates the focus on making AI models accessible on regular consumer devices, contrasting with high-memory builds.

    - RIP26770 suggests compiling `llama.cpp` directly on the Xiaomi 12 Pro hardware to potentially double the inference speed compared to using Ollama. This implies that the overhead from Ollama might be significant, and optimizing the model compilation for the specific hardware can yield better performance.
    - SaltResident9310 expresses a desire for AI models that can run efficiently on consumer-grade devices, highlighting a frustration with the high resource demands of current models that require 48GB or 96GB of RAM. This underscores a need for more accessible AI solutions that don't necessitate high-end hardware.
    - International-Try467 inquires about the specific inference speeds achieved on the Xiaomi 12 Pro, indicating an interest in the practical performance metrics of running AI models on consumer hardware. This reflects a broader curiosity about the feasibility and efficiency of deploying AI on mobile devices.

  - **[Are Local LLMs actually useful… or just fun to tinker with?](https://www.reddit.com/r/LocalLLM/comments/1sm4i2m/are_local_llms_actually_useful_or_just_fun_to/)** (Activity: 454): **Local LLMs offer significant advantages in terms of privacy and cost savings, as they eliminate API costs and keep data on-premises. However, they often require substantial setup and maintenance, which can be a barrier to practical use. Despite this, they excel in handling sensitive or internal tasks such as processing private documents or data. Some users report that local models like the `31B from Gemma 4` family are performing exceptionally well, especially for tasks like coding and creative writing, when run on high-performance hardware such as a `3090 24GB with 192GB RAM`. The performance gap between local and cloud models is narrowing, particularly as cloud models face degradation under high demand, making local models increasingly viable for everyday use.** There is a consensus that while local LLMs are not yet mainstream for everyday workflows, they are becoming more practical as setup and maintenance challenges are addressed. Some users note that cloud models have degraded in quality, making local models more competitive, especially for cost-sensitive applications.

    - Local LLMs are particularly advantageous for handling sensitive or internal data due to their ability to operate without API costs and data leaving the system. The main challenge lies in the setup and maintenance, which once streamlined, could make 'offline GPT' setups viable for everyday work beyond just experimentation.
    - The performance of local models like the 31B from the Gemma 4 family is highlighted as being exceptionally good, especially in comparison to cloud API models which have degraded due to increased demand. A user reports using these models for various tasks such as coding and creative writing, leveraging a 3090 GPU with 24GB VRAM and 192GB RAM.
    - Local models can be cost-effective compared to cloud APIs, especially for complex projects where API costs can be prohibitive. However, they require careful architectural planning to ensure models are used for tasks they are capable of handling, such as using a 32B model as a privacy filter for business communications.


### 3. Quantization and Model Performance Analysis

  - **[Updated Qwen3.5-9B Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1sl59qq/updated_qwen359b_quantization_comparison/)** (Activity: 463): **The post presents a detailed evaluation of various quantizations of the **Qwen3.5-9B** model using **KL Divergence (KLD)** as a metric to assess the faithfulness of quantized models compared to the BF16 baseline. The analysis ranks quantizations based on their KLD scores, with lower scores indicating closer alignment to the original model's probability distribution. The top-performing quantization in terms of KLD is **eaddario/Qwen3.5-9B-Q8_0** with a KLD score of `0.001198`. The evaluation dataset and tools used include [this dataset](https://gist.github.com/cmhamiche/788eada03077f4341dfb39df8be012dc) and [ik_llama.cpp](https://github.com/Thireus/ik_llama.cpp/releases/tag/main-b4608-b33a10d). The post also includes a [size vs KLD plot](https://preview.redd.it/an70gj4sbgvg1.png?width=12760&format=png&auto=webp&s=e3577233ef6fd421fbaa7371491283478264b4e1) and mentions compatibility with `llama.cpp`.** Commenters suggest using different shapes for visual differentiation in plots and express interest in evaluating other models like Gemma 4, particularly its MoE variant. There is also a mention of potential superior performance from quantizations produced by [Thireus' GGUF Recipe Maker](https://gguf.thireus.com/quant_assign.html).

    - Thireus mentions a quantization methodology that he and another user, EAddario, have been developing for nearly a year. He suggests adding quantization results from [gguf.thireus.com](https://gguf.thireus.com/quant_assign.html), which claims to outperform existing methods. This highlights ongoing efforts in the community to refine quantization techniques for better model performance.
    - cviperr33 discusses the effectiveness of using iq4 xs or nl quant methods on models ranging from 20-35B parameters, noting that these techniques also perform well on smaller models. This suggests a potential scalability of certain quantization methods across different model sizes, which could be valuable for optimizing performance without sacrificing accuracy.
    - dampflokfreund expresses interest in the impact of lower quantization levels on models like Gemma 4, particularly the MoE (Mixture of Experts) architecture. This points to a curiosity about how quantization affects complex model architectures differently, which could lead to insights on optimizing such models.

  - **[Best open-source LLM for coding (Claude Code) with 96GB VRAM?](https://www.reddit.com/r/LocalLLM/comments/1sldbvw/best_opensource_llm_for_coding_claude_code_with/)** (Activity: 229): **The user is utilizing a local setup with approximately `96GB VRAM` on an **RTX 6000 Blackwell** GPU, running **Qwen3-next-coder** models with **Claude Code** for coding tasks. They are seeking recommendations for potentially better models for tasks such as reasoning, debugging, and multi-file work. **MiniMax 2.5** and **2.7** are mentioned as impressive alternatives, especially when accessed via API, with some users noting success with aggressively quantized versions of 2.7. **Unsloths Gemma 4 31b UD q5_xl** is highlighted as a top local agentic coder, offering around `70 tokens per second` on a similar setup. **Owen 3.5 q 4 k XL** is also recommended, with some users testing a reaped version with q6, and **opencode** is suggested as an alternative to Claude Code.** There is a debate on the effectiveness of different models, with some users preferring **Unsloths Gemma 4** for its performance and speed, while others find **MiniMax 2.7** to be a strong contender when accessed via API. The choice between **Qwen3.5** and **27 dense** models also reflects differing user experiences and preferences.

    - **MiniMax 2.5 and 2.7** are highlighted as impressive alternatives to Claude Opus for coding tasks, especially when accessed via API. Users have noted the effectiveness of aggressively quantized versions of MiniMax 2.7, suggesting potential for high performance even with limited local resources.
    - **Unsloths Gemma 4 31b UD q5_xl** is praised for its performance as a local agentic coder, with benchmarks showing around 30 tokens per second on a dual Tesla V100 16GB setup. This suggests that with 96GB VRAM, one could achieve over 70 tokens per second, indicating significant efficiency for local deployment.
    - **Qwen 3.5 27b** in 8-bit quantization is recommended for its balance of performance and resource efficiency, fitting comfortably within 96GB VRAM while allowing for large context sizes. The model's ability to expand context to 1M using vllm with rop/yarn is noted, although some users have transitioned to the larger 122b model for enhanced capabilities.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.