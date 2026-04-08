---
id: MjAyNS0x
title: Nvidia buys (most of) Groq for $20B cash; largest execuhire ever
date: '2025-12-24T05:44:39.731046Z'
description: >-
  **Groq** leadership team is joining **Nvidia** under a "non-exclusive
  licensing agreement" in a deal valued at **$20 billion cash**, marking a major
  acquisition in AI chip space though Nvidia states it is not acquiring Groq as
  a company. Jensen Huang plans to integrate Groq's low-latency processors into
  the NVIDIA AI factory architecture to enhance AI inference and real-time
  workloads. Twitter highlights include **Gemini** used as a consumer utility
  for calorie tracking, OpenAI discussing the "deployment gap" focusing on model
  usage in healthcare and business, and Tesla's FSD v14 described as a "Physical
  Turing Test" for consumer AI. Benchmarking challenges are noted by **Epoch
  AI** emphasizing provider variance and integration issues affecting model
  quality measurement. Discussions on coding agents and developer experience
  convergence continue in the AI community.
companies:
  - nvidia
  - groq
  - openai
  - tesla
  - epoch-ai
  - gemini
models:
  - gemini
  - fsd-v14
topics:
  - benchmarking
  - inference
  - model-evaluation
  - ai-integration
  - agent-patterns
  - real-time-processing
  - low-latency
  - developer-experience
  - healthcare
  - business-workflows
  - consumer-ai
people:
  - jensen_huang
  - xeophon
  - js_denain
  - jim_fan
---


**Execuhires are back!**

> AI News for 12/24/2025-12/25/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (208 channels, and 5086 messages) for you. Estimated reading time saved (at 200wpm): 346 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Execuhires first started [in Aug 2024](https://news.smol.ai/issues/24-08-02-ainews-execuhires-tempting-the-wrath-of-khan) and again in [Jun 2025](https://news.smol.ai/issues/25-06-11-execuhires-2), but it seems Christmas Eve 2025 isn't too late for a hat-trick. In a 5 sentence post, Groq confirmed it's "non-exclusive licensing agreement" for most of Groq's leadership team to join Nvidia, leaving behind GroqCloud, while the current CFO will become the CEO of the old Groq, for [reported total consideration of $20 billion cash.](https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html)

It's an acquisition in everything but name, and made interesting by a few other facts: Groq was last valued at $6.9B in Sept, and says Nvidia came inbound to Groq. Nvidia's former largest acquisition was the 2019 acquisition of Mellanox for $7B, yet this acquisition is only 1/3 of Nvidia's cash war chest.

Jensen's quote is the most actual detail we have on future plans:

> “We plan to integrate Groq’s low-latency processors into the NVIDIA AI factory architecture, extending the platform to serve an even broader range of AI inference and real-time workloads,” Huang wrote.
> 
> 
> Huang added that, “While we are adding talented employees to our ranks and licensing Groq’s IP, we are not acquiring Groq as a company.”
> 

That's all we know, but in semis world this is very very earth shaking, not least for hopeful Nvidia competitors.

---

# AI Twitter Recap

**Top tweets (by engagement)**

- **Gemini as a consumer utility**: A viral example of “AI as habit-forming assistant” with Gemini used for calorie tracking ([tweet](https://twitter.com/skooookum/status/2003608923371389157)).
- **“Deployment gap” / capability overhang**: OpenAI frames 2026 progress as being as much about *getting models used well* as about frontier capability—especially in **healthcare**, business, and daily life workflows ([tweet](https://twitter.com/OpenAI/status/2003594025098785145)).
- **FSD as “Physical Turing Test”**: Jim Fan describes Tesla FSD v14 as the first consumer AI that can feel indistinguishable from a human driver in day-to-day use, emphasizing how quickly “surreal → routine → dependency” happens ([tweet](https://twitter.com/DrJimFan/status/2003593613918531891)).
- **Compute/infrastructure realism**: Modi’s ISRO launch posts dominate overall engagement but are largely outside the AI-engineering scope ([tweet](https://twitter.com/narendramodi/status/2003677923820335183), [tweet](https://twitter.com/narendramodi/status/2003681952323502169)).

---

**Benchmarking and Evaluation: Provider Variance, Harness Bugs, and “What Even Is a Score?”**

- **Benchmarking is fragile because the pipeline is fragile**: Epoch AI highlights that reported scores are often downstream of provider behaviors (timeouts, rate limits, tokenization quirks, missing params, transient errors), and that *newer models/providers* are disproportionately impacted ([Epoch overview](https://twitter.com/EpochAIResearch/status/2003592566772822516), [provider errors note](https://twitter.com/EpochAIResearch/status/2003592610569683089)). The guest post by [@xeophon](https://twitter.com/xeophon/status/2003592720741466478) (with [@js_denain](https://twitter.com/EpochAIResearch/status/2003592622724776201)) turns this into a very operational checklist: if your harness doesn’t control for sampling params, retries, truncation, tool-calling differences, and API edge cases, you aren’t measuring model quality—you’re measuring provider reliability and integration debt.
- **“Same model, different provider, different output quality” becomes a first-class issue**: Multiple engineers echo that the open-model ecosystem now depends on inference providers as much as weights; benchmarking providers needs “agent harness” discipline (prompting, deployment config, sampling, tool behavior) rather than simple one-shot eval scripts ([summary](https://twitter.com/eliebakouch/status/2003604370534072445), [pointer to the blog](https://twitter.com/dejavucoder/status/2003594248973930929)). This also ties into a broader conversation about what “open” means in practice—weights alone are not reproducibility ([LMArena on “open” shades of gray](https://twitter.com/arena/status/2003620051078074593)).

---

**Coding Agents, Agent Packaging, and Developer Experience (DX) Convergence**

- **From “agent patterns” to prompts+tools**: Several builders report that with current frontier/coding models, many classic patterns (plan/reflect loops, hand-authored tool policies) are becoming optional—good prompting + tool definitions often suffice, shifting effort toward *context engineering* and good defaults ([diptanu](https://twitter.com/diptanu/status/2003674481144004667), [Weaviate definition + context engineering angle](https://twitter.com/weaviate_io/status/2003824281231220902)).
- **Packaging agents is the missing primitive**: [@hwchase17](https://twitter.com/hwchase17/status/2003599022871777467) argues that [**agent.md](http://agent.md/) + skills** (as open standards) can define an agent, but we still lack a portable unit that bundles: rules, skills, MCP servers/tools, and subagents—“a neat lil zip file that spawns a whole agent squad” ([follow-up](https://twitter.com/hwchase17/status/2003715230120173737)). He points to **OpenCode’s agent spec** as a better baseline because it allows an agent to be used as *the main agent* or *a subagent*, enabling fully specialized “turn the whole environment into a LangGraph-writing agent” workflows ([tweet](https://twitter.com/hwchase17/status/2003922408240304245)).
- **Tooling ships around “skills” as reusable policy modules**: Mistral’s Vibe CLI ships “Skills” as reusable rule bundles, plus reasoning model support and terminal theming—explicitly pushing toward shareable, project-level agent policy artifacts ([tweet](https://twitter.com/MistralAI/status/2003843358054068327)).
- **Usage-limit economics shape behavior**: Anthropic/Claude doubles Pro/Max limits through New Year’s, explicitly encouraging builders to push agentic workflows harder ([Claude](https://twitter.com/claudeai/status/2003918730833608902), [Alex Albert](https://twitter.com/alexalbert__/status/2003923042100273389)). On the flip side, users report “quota burn” as a real constraint in iterative agent loops ([tweet](https://twitter.com/vikhyatk/status/2003647290507227396)).
- **Emerging UX patterns**: Windsurf “Wave 13” highlights **true parallel agents** + dedicated agent terminal, reflecting convergence on conductor-style orchestration UX (worktrees + cascades) ([Cognition](https://twitter.com/cognition/status/2003926592406671472), plus [swyx’s meta-commentary](https://twitter.com/swyx/status/2003941412572934361)). Base44 shows an IDE-adjacent direction: edit code while seeing live preview; click UI to jump to defining code—treating UI as a navigational index into code ([tweet](https://twitter.com/MS_BASE44/status/2003868520359317749)).

---

**Open Models and the “Inference Distribution Layer”: MiniMax M2.1, GLM-4.7, Qwen Image Edit**

- **MiniMax M2.1’s distribution blitz**: M2.1 shows up across multiple “where developers are” surfaces—**LMArena Code Arena** ([Arena](https://twitter.com/arena/status/2003585316029104383)), **Cline** ([MiniMax](https://twitter.com/MiniMax__AI/status/2003599117503852680)), **Kilo** ([tweet](https://twitter.com/MiniMax__AI/status/2003606223191703708)), **Roo Code** ([tweet](https://twitter.com/MiniMax__AI/status/2003611728320561528)), **Ollama** ([tweet](https://twitter.com/MiniMax__AI/status/2003715959719362584)), **BlackboxAI** ([tweet](https://twitter.com/MiniMax__AI/status/2003926396335460447)), and more. Benchmarks/leaderboards reinforce adoption: strong results on SWE-bench variants and SciCode ([Ofir Press](https://twitter.com/OfirPress/status/2003625671042732329)), and #2 among open-weight models on Vals Index behind GLM 4.7, but with lower latency/cost ([ValsAI](https://twitter.com/ValsAI/status/2003646964664287667)). MiniMax also claims long-horizon coding at ~1/10 Opus pricing ([tweet](https://twitter.com/MiniMax__AI/status/2003673337671602378)).
- **Zhipu’s GLM-4.7 momentum + devpack/MCP integrations**: Zhipu highlights continued open-sourcing and Hugging Face trending (#1) ([tweet](https://twitter.com/Zai_org/status/2003828175089098943)). Roo Code announces GLM-4.7 availability ([tweet](https://twitter.com/roocode/status/2003652972555997560)). Zhipu also pushes MCP-style developer tooling like **Zread MCP** for in-chat repo exploration (search/read files without leaving the agent flow) ([tweet](https://twitter.com/Zai_org/status/2003872419791229285)). Separately, engineers showcase high-throughput local inference for GLM 4.7 on Apple Silicon clusters via MLX distributed + batch gen (e.g., **63 tok/s** throughput on **4× M3 Ultra**, 6-bit, batch size 4) ([awnihannun](https://twitter.com/awnihannun/status/2003854411848904937)).
- **Qwen Image Edit 2511 as a “productized open image editor”**: Qwen-Image-Edit-2511 lands across Replicate and other UIs ([Replicate launch](https://twitter.com/Alibaba_Qwen/status/2003751934013100458), [TostUI](https://twitter.com/Alibaba_Qwen/status/2003753784527507781), plus community HF spaces like [@_akhaliq](https://twitter.com/_akhaliq/status/2003601664675316051)). Fine-tuning accessibility improves: AI Toolkit support for LoRAs plus a **3-bit accuracy recovery adapter** enabling <24GB VRAM finetunes ([ostrisai](https://twitter.com/ostrisai/status/2003808898189611491)).

---

**Training & Research Notes: RL for Agents, Pretraining Tricks, and Representation/Attention Fixes**

- **End-to-end RL for tool-using agents (Agent-R1)**: A long technical thread frames agent training as fundamentally RL due to **stochastic tool/environment feedback**, proposing explicit masking for credit assignment and a ToolEnv interaction loop. Reported gains vs naive RAG are large on multi-hop QA (e.g., GRPO **0.3877 EM** vs RAG **0.1328 EM**) ([thread](https://twitter.com/omarsar0/status/2003862504490086596)).
- [**Character.AI](http://character.ai/)’s pretraining “Squinch” and related tricks**: [@simon_mo_](https://twitter.com/simon_mo_/status/2003608325624406482) summarizes a CAI blogpost describing how they maintained strong MFU on **GCP H100-TCPX** despite weaker networking by using Noam Shazeer’s gradient compression algorithm “**Squinch**” (plus other pretraining tricks). Follow-on tweets highlight their distillation approach as notable ([eliebakouch](https://twitter.com/eliebakouch/status/2003632344159424562)).
- **Multimodal without massive paired data (SEMI)**: DeepLearningAI summarizes SEMI: plug any pretrained encoder into an LLM via a projector plus LoRA adapters generated from a handful of paired examples; trained on data-rich domains, few-shot adapts to new ones ([tweet](https://twitter.com/DeepLearningAI/status/2003593131132916204)).
- **Architectural/representation papers worth flagging**:
    - **PoPE vs RoPE entanglement**: claims RoPE entangles content and position; proposes PoPE as a fix ([tweet](https://twitter.com/agopal42/status/2003900815560659303)).
    - **Recurrent-layer ViT compression**: suggests rewriting N-layer ViT using K≪N layers with recurrence while matching DINOv2 performance with ~2–3 layers ([tweet](https://twitter.com/f14bertolotti/status/2003760506214158693)).
    - **Attention scaling explainer**: a clear “why divide by √d_k” write-up aimed at preventing softmax saturation/vanishing gradients in attention ([tweet](https://twitter.com/viplismism/status/2003807608571076782)), with a nuanced counterpoint that L2-normalizing attention is only variance-preserving under restrictive assumptions about value correlation ([thread](https://twitter.com/ArmenAgha/status/2003918120881475832)).

---

**Robotics, Autonomy, and “Physical Turing Test” Framing**

- **NVIDIA robotics stack progress**: Jim Fan positions robotics as the “last grand challenge,” listing NVIDIA’s recent releases: **GR00T VLA** open-sourced checkpoints (N1, N1.5, N1.6), **GR00T Dreams** world model, **SONIC** whole-body control foundation model, and RL post-training recipes—spanning simulation to sim2real ([thread](https://twitter.com/DrJimFan/status/2003879965369290797), [sim2real note](https://twitter.com/DrJimFan/status/2003879976173818298)).
- **Humanoid robots interacting autonomously**: Brett Adcock posts demos of robots interacting with people and responding to instructions without teleop, emphasizing voice-to-manipulation coupling (intent → pixels → actions) ([swag demo](https://twitter.com/adcock_brett/status/2003598494838431874), [autonomy claim](https://twitter.com/adcock_brett/status/2003598719971995709), [voice+manipulation framing](https://twitter.com/adcock_brett/status/2003909157897015585)).
- **Waymo’s “human module doesn’t scale”**: a pointed critique claims a SF incident reflected a backlog of remote “confirmation checks,” implying a dependency trap where humans remain a throughput bottleneck in autonomy stacks ([tweet](https://twitter.com/Yuchenj_UW/status/2003708815934640536)).

---

**Macro Themes: Talent, Product Cycles, and the “Deployment Gap”**

- **Talent wars are about mission + peers**: Sarah Hooker’s take: top talent has many options; what wins is working with like-minded people pushing the boundary, not just comp ([tweet](https://twitter.com/sarahookr/status/2003581788850127276)).
- **Product strategy under 3-month model cycles**: a widely shared summary of “Lovable” growth lessons argues PMF “expires” every model cycle; MVP gives way to “MLP,” and moats become release speed + brand rather than technology ([tweet](https://twitter.com/crystalsssup/status/2003704941962285463)).
- **OpenAI’s “capability overhang”**: the most explicit meta-claim in the set: model capability is outpacing actual user deployment; progress in 2026 depends on closing that adoption gap with better UX/workflows and sector integration (healthcare, business) ([tweet](https://twitter.com/OpenAI/status/2003594025098785145)).
- **Engineering labor shifts to orchestration**: a management take argues ICs are becoming “orchestrators”—aggressive context switching + judgment/taste matter more than raw implementation speed (tweet has 0 engagement but captures an emergent motif across multiple threads) ([tweet](https://twitter.com/brivael/status/2003871914104688867)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

nothing met our bar

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI-Generated Art and Animation Experiments

- [**Former 3D Animator trying out AI, Is the consistency getting there?**](https://www.reddit.com/r/StableDiffusion/comments/1puszuc/former_3d_animator_trying_out_ai_is_the/) (Activity: 1280): **A former 3D animator is experimenting with integrating AI to enhance realism in 3D animations. By using a custom LoRA trained on their own 3D renders, they aim to maintain the character's original essence while adding AI-driven realism. The project involves a complex mix of tools beyond just ComfyUI, focusing on whether the AI-enhanced character movements appear human-like or if the illusion fails.** The comments reflect a mix of humor and skepticism, with some users joking about job displacement for 3D animators and others providing minimal feedback. The technical debate is minimal, focusing more on the novelty and execution of the project.
- [**not all cosplayers are real**](https://www.reddit.com/r/ChatGPT/comments/1pua0aj/not_all_cosplayers_are_real/) (Activity: 1002): **The post discusses the use of Nano Banana Pro, an AI tool, to generate realistic cosplay images in minutes, highlighting the potential for misuse in social media where individuals might present AI-generated images as real cosplay to gain views or tips. The author notes that the images were not cherry-picked, acknowledging some inaccuracies, yet finds the results convincing. This raises concerns about authenticity in online cosplay communities.** A notable opinion from the comments expresses frustration with the prevalence of AI-generated content in cosplay communities, indicating a growing concern among real cosplayers about the impact of such technology on their craft.
- [**The illusion painter, part 2**](https://www.reddit.com/r/aivideo/comments/1puqnba/the_illusion_painter_part_2/) (Activity: 1288): **The post titled 'The illusion painter, part 2' seems to be a continuation of a series involving a painter whose work creates illusions. The technical discussion in the comments suggests a critique of the narrative structure, with one commenter suggesting that the series could have ended with a twist by presenting an actual painting instead of an illusion. This implies a focus on narrative expectations and subversion in the context of visual art.** The comments reflect a mix of humor and critique, with one user humorously expressing satisfaction at the fictional demise of characters, while another questions when the 'villain' will be stopped, indicating a narrative-driven engagement with the content.
- [**The illusion painter, part 2**](https://www.reddit.com/r/aivideo/comments/1puqnba/the_illusion_painter_part_2/) (Activity: 1289): **The post titled 'The illusion painter, part 2' seems to be a continuation of a series or narrative involving an 'illusion painter.' The technical content of the post is not clear from the title or comments, but it suggests a theme of deception or unexpected outcomes, possibly involving visual art or a narrative twist. The comments reflect a mix of humor and critique, with one suggesting a narrative twist by having the last piece be an actual painting, indicating a possible theme of subverting expectations.** The comments reflect a mix of humor and critique, with one suggesting a narrative twist by having the last piece be an actual painting, indicating a possible theme of subverting expectations.

### 2. AI Character and Meme Creations

- [**I asked CGPT to generate itself as a character alongside other AI Chatbots**](https://www.reddit.com/r/ChatGPT/comments/1pukx34/i_asked_cgpt_to_generate_itself_as_a_character/) (Activity: 2441): **The image is a creative representation of various AI chatbots as anime-style characters, each with distinct color themes and styles. This artistic depiction includes ChatGPT, Gemini, Grok, and Claude, each characterized by unique visual elements that reflect their perceived personalities or functionalities. The image is non-technical and serves as a visual metaphor rather than a technical illustration of the chatbots' capabilities or architectures. [View Image](https://i.redd.it/zv353p0mv49g1.png)** The comments humorously engage with the image, with one noting that Grok's representation seems to misunderstand the assignment, suggesting a playful critique of its character design. Another comment humorously suggests that ChatGPT's character now canonically has green hair, reflecting the community's engagement with the visual representation.
- [**Back then VS now**](https://www.reddit.com/r/ChatGPT/comments/1pu7rsu/back_then_vs_now/) (Activity: 1340): **The image is a meme that humorously contrasts the shift in students' reliance on information sources from Wikipedia to AI tools like ChatGPT. It highlights how students now prefer using AI for information, while Wikipedia, once a primary source, is depicted as less significant. The meme reflects a broader trend in educational contexts where AI is increasingly used for research and learning, overshadowing traditional sources like Wikipedia.** One comment humorously notes the irony of students using AI over citable sources, while another reflects on the cultural shift in how Wikipedia is perceived, from being discouraged as a source to being overshadowed by AI.

### 3. AI-Driven Music and Video Creations

- [**Lord of the Rings Disco: One Funk to Rule them All | Music Video by Wicked AI**](https://www.reddit.com/r/aivideo/comments/1pu8smq/lord_of_the_rings_disco_one_funk_to_rule_them_all/) (Activity: 1433): **Wicked AI has released a music video titled 'Lord of the Rings Disco: One Funk to Rule them All', which creatively combines elements of the *Lord of the Rings* franchise with a disco theme. The video is generated using AI, showcasing the capabilities of AI in producing complex multimedia content. However, some viewers have noted that the music style does not strictly adhere to traditional disco or funk genres, indicating a potential mismatch in genre classification.** One commenter expressed a mix of amazement and concern about the AI-generated content, highlighting the dual nature of AI advancements as both impressive and potentially unsettling.
- [**Lord of the Rings Disco: One Funk to Rule them All | Music Video by Wicked AI**](https://www.reddit.com/r/aivideo/comments/1pu8smq/lord_of_the_rings_disco_one_funk_to_rule_them_all/) (Activity: 1436): **Wicked AI has released a music video titled 'Lord of the Rings Disco: One Funk to Rule them All', which creatively combines elements of the *Lord of the Rings* franchise with a disco theme. The video is a product of AI-generated content, showcasing the capabilities of modern AI in blending cultural themes with music. However, some viewers have noted that the music does not strictly adhere to traditional disco or funk genres, indicating a potential mismatch in genre classification.** One commenter expressed a mix of amazement and concern about the AI-generated content, highlighting the dual nature of AI advancements as both impressive and potentially unsettling.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Wave 13 Coding Agents & AI IDE Tooling**

- **Windsurf Waves In Near‑Frontier SWE-1.5 for Free**: Windsurf announced **Wave 13: Shipmas Edition**, adding **parallel multi-agent Cascade workflows**, a **dedicated zsh terminal** (opt‑in on macOS), **Git worktree support**, **multi-Cascade panes & tabs**, and a **context window indicator**, while making their near-frontier coding model **SWE‑1.5** free for all users at normal throughput for **3 months**. The team positioned SWE‑1.5 as **near SWE‑Bench‑Pro** performance and bundled these features in a single seasonal drop under the banner *“Merry Shipmas!”* in their [Wave 13 announcement](https://discord.com/channels/1027685395649015980/1027688115592237117/1453488772837671157).
    - Engineers highlighted that **Git worktree support** plus **multi-pane Cascade** enables concurrent branches and experiments in the *same repo* without merge hell, which directly targets common agentic‑coding workflows. The **dedicated terminal** runs under users’ own `.zshrc`, which members see as crucial for robust tooling, path setup, and long‑running commands compared to ephemeral, sandboxed shells.
- **OpenRouter Pipes Straight Into Open-WebUI**: A community dev released an **Open-WebUI integration pipeline** for **OpenRouter’s Responses API**, published as the `Open-WebUI-OpenRouter-pipe` project on GitHub at [Open-WebUI-OpenRouter-pipe](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe). The author invited users to pound on it with real workloads and file bugs so the integration can harden before broader adoption.
    - In parallel, the **llumen** chat UI demo at [llumen demo](https://llumen-demo.easonabc.eu.org/) shipped a [v0.4.2](https://github.com/pinkfuwa/llumen/releases/tag/v0.4.2) patch specifically to fix title generation glitches and minor chat bugs, indicating rapid iteration on front-end agentic experiences. OpenRouter users also debated **response caching** for coding flows, with one engineer claiming **80–90% cache hit rates** are realistic and another warning that naïve cache layers at the router level are hard to safely expose as user-facing cost savings.
- **DSPy Eyes TextGrad and Agentic Context Engineering**: In the **DSPy** community, members asked whether the ideas from **“Agentic Context Engineering”** ([*Agentic Context Engineering*](https://arxiv.org/pdf/2510.04618)) and **“LLM Autodiff / TextGrad”** ([*LLM AutoDiff: TextGrad*](https://arxiv.org/abs/2501.16673)) will land in DSPy, and linked the reference implementation [textgrad](https://github.com/zou-group/textgrad). Engineers debated whether text-based gradient‑like updates could become a “DSPy killer,” but others noted *“it doesn't seem to be actively maintained”* and framed it more as a conceptual precursor than a production tool.
    - This fed into a broader **prompt optimization** discussion, where users observed *“at least 10 slight variations of each method like a textgrad version”*, expressing fatigue with papers but interest in **composable optimization primitives** in DSPy. A new **senior full‑stack/blockchain engineer** joined, listing a modern stack (React/Next.js/Nuxt, Node/Nest/Laravel/FastAPI, Solidity, Docker/AWS, PostgreSQL/Redis/MongoDB), underscoring that DSPy’s audience is increasingly systems and infra engineers rather than just ML researchers.

**2. Video, Audio & Multimodal Model Tooling**

- **ElevenLabs Turns Into a One‑Stop AI Video Mall**: In OpenAI’s community, users reported using [**ElevenLabs**](https://elevenlabs.io/) as a hub for generating videos with **Sora 2**, **Google Veo 3.1**, and **Kling 2.6**, praising that *“all projects are accessible in one place rather than juggling accounts.”* One engineer noted that Sora‑2 videos rendered via ElevenLabs **ship without watermarks**, in contrast to **Nano Banana Pro**, which slaps a visible mark on every output.
    - Pricing and policy comparisons emerged as people contrasted **ElevenLabs** with [**Higgsfield**](https://higgsfield.ai/), which offers *unlimited* video generation for some models at **$49/month (yearly)**, while others defended ElevenLabs on the grounds that they already rely on it for **audiobook narration**. Creators also observed that ElevenLabs’ internal safety layers vary by backend—some prompts rejected by **Sora 2** will partially run on **Veo 3.1** or **Kling O1**, fueling speculation that *“Sora is checking the actual output also, while veo/wan [check] the text prompt input.”*
- **FlashSR Supercharges Audio Enhancement at 200× Realtime**: On Latent Space, Yatharth Sharma announced **FlashSR**, a fast audio enhancement / super‑resolution model capable of processing **>200× realtime**, via an X post: [FlashSR audio enhancement release](https://xcancel.com/Yatharth3501/status/2003884180577702074). FlashSR has already been integrated into **MiraTTS** and is now released as open models and code on **Hugging Face** and **GitHub** for others to plug into TTS and speech pipelines.
    - Engineers see FlashSR as a practical drop‑in for **latency‑sensitive voice products**, since sub‑10 ms per second of audio makes multi‑stage pipelines (ASR → LLM → TTS → enhancement) feasible without user‑visible lag. Its presence in MiraTTS before public release reassured some that the code is battle‑tested in production‑like workloads rather than just a research demo.
- **Qwen 2.5VL-3B and TRELLIS.2-4B Push Affordable Multimodal**: Hugging Face users reported that **Qwen 2.5VL‑3B** can handle images around **1400×900** on a **P100** when run in **4‑bit** with **unquantized vision layers**, consuming about **5 GB VRAM** for inference and **~4 GB** in a **QLoRA** finetune with **2k 8‑bit PNG** and ~8k token context. In parallel, Microsoft released [**TRELLIS.2‑4B**](https://huggingface.co/microsoft/TRELLIS.2-4B), a **4B‑param** model that converts **2D images to 3D** with a 1536‑resolution field on **8 GB GPUs**, built on **SigLIP** vision and a **Qwen‑3–based** language backbone.
    - Practitioners called out that these configs make **serious multimodal work feasible on commodity cloud GPUs**, though one user joked that TRELLIS *“must have absolutely unusable results, immediately”* if it can’t run on a **toaster‑class GPU**. The discussions centered on how far you can push quantization (4‑bit for language, full‑precision vision) before degradation hits, and when to prefer bf16/fp16 vs full fp32 for visual branches.

**3. Architecture Tricks, Precision Wars & Interpretability**

- **Partial RoPE, RMSNorm Placement and Attention Norms Under the Microscope**: In Eleuther’s **research** channel, contributors dissected **partial RoPE** adoption (e.g., in **Qwen3‑Next**), pointing to a historical ablation study in an [**arXiv paper**](https://arxiv.org/abs/2512.19941) that showed measurable gains for efficiency and long‑context generalization. They also debated the blog post **“[Attention normalizes the wrong norm](https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/),”** with one researcher asserting that alternative normalization performed **worse than standard softmax** in language tasks even after training **~5000 models** with careful controls.
    - Engineers further discussed how **RoPE hampers interpretability**, referencing Eleuther’s own writeup [**Rotary Embeddings: A Complete Guide**](https://blog.eleuther.ai/rotary-embeddings/) and contrasting **RoPE vs PoPE** through shared figures. A separate thread asked about putting **RMSNorm after attention (post‑SDPA)**; others cited the **Qwen gating paper** to argue that an added norm improves training stability and performance, likely because of the **nonlinearity** in norms rather than just scaling.
- **bf16 vs fp16: Range, Overflow and LR Tradeoffs**: Hugging Face users revisited the **bf16 vs fp16** debate, noting that **fp16** has *higher precision but lower dynamic range*, whereas **bf16** offers *lower precision with much higher range*, which matters for large activations. One engineer summarized it as *“bf16 won't overflow that easily... for huge softmax n stuff (but mostly they are done in f32...) but with f16... the higher precision helps params accommodate a lower lr which could underflow with bf16”*, outlining why mixed‑precision stacks often juggle all three formats.
    - This fed into questions about running **bf16‑trained models in fp16 inference**, where the consensus was that you can *often get away with it* but should expect more **overflow/NaN risk** unless you keep the largest matmuls and softmax in fp32. Practitioners recommended checking **optimizer states and LR schedules** when porting training recipes across formats, since a schedule tuned to fp16 underflows can misbehave badly when swapped to bf16.
- **RoPE Interp Pain Points and Call for SAE Tooling**: Eleuther’s **interpretability** discussions highlighted how **RoPE‑based transformers** remain a headache for feature attribution and circuit‑level analysis, prompting some to hope future models ditch RoPE entirely in favor of more interpretable position encodings. In a related thread, someone asked for mainstream **open‑source Sparse Autoencoder (SAE) repos** that support **fine‑tuning already‑trained SAEs**, not just training from scratch, to enable incremental refining of feature dictionaries.
    - Researchers shared **EleutherAI’s rotary blog** again as a canonical reference to reason about how RoPE twists feature spaces in complex ways, arguing this makes standard SAE techniques harder to map onto token positions. The explicit request for fine‑tuneable SAEs signals a shift from pure research curiosity to **production‑grade interpretability tooling**, where teams want to gradually refine feature sets without retraining massive autoencoders from zero.

**4. GPU Hardware, Kernels & Quantization Engineering**

- **CUDA vs Triton Quantization and the Swizzle Renaissance**: GPU MODE members circulated **“Quantization: CUDA vs Triton”** slides via Dropbox ([Quantization Cuda vs Triton](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0)), comparing quantization strategies and performance tradeoffs across backends, though some reported trouble opening the link. In accompanying CUDA discussions, they emphasized that **Tensor Memory Accelerator (TMA) transpose** only hits peak performance when paired with **swizzle layouts**, pointing others to the [effective_transpose](https://github.com/simveit/effective_transpose) repo as the reference implementation.
    - In **career‑advice**, users praised **cute’s swizzling** as its *main character* feature, since bank‑conflict‑aware layouts enable faster **tcgen PTX** kernels. The meta‑lesson for kernel hackers was that mastering **memory layout transformations (swizzles, TMA patterns)** often beats micro‑optimizing arithmetic when chasing leaderboards.
- **NVIDIA nvfp4_dual_gemm Leaderboard Turns Into a Microsecond Arms Race**: Multiple GPU MODE contributors reported new personal bests on NVIDIA’s `nvfp4_dual_gemm` leaderboard, with submission latencies falling from **65.5 μs** to **60.0 μs**, **77.6 μs** to **41.6 μs**, and another user climbing from **45.2 μs** down to **26.9 μs**, earning **8th place**. One competitor then shaved this further to **18.1 μs** (taking **7th**) before another pushed to **15.6 μs**, showcasing aggressive iteration on tiling, swizzle, and pipeline depth.
    - These runs provided concrete feedback loops for CUDA/Triton experiments, where small changes in **block size, shared‑memory usage, and swizzled layouts** translate directly into leaderboard jumps. The thread implicitly doubles as an open notebook of **practical GEMM optimization recipes**, which others can adopt for high‑throughput LLM inference kernels.
- **From $30k Inference Rigs to Dual‑Channel Budget Laptops**: In Unsloth’s server, one engineer is speccing a **$30k voice‑inference box** targeting **100× parallel audio pipelines** (Whisper, Wav2Vec2.0, BERT, Gemma, Llama‑3.3) and weighing **3× RTX 5090 + 3× RTX 5060 Ti** versus fewer **RTX 6000 Ada/Pro** cards, plus **Threadripper 9975wx (32‑core) vs 9985wx (64‑core)** or high‑end Intel. Others recommended benchmarking on [**Runpod**](https://runpod.io/) or [**Vast.ai**](http://vast.ai/) to empirically map CPU saturation and PCIe bottlenecks before locking in hardware.
    - At the other end of the spectrum, LM Studio and Nous users swapped war stories about squeezing LLMs onto **GTX 970 4 GB** cards and budget laptops, discovering that **dual‑channel RAM** via a **$100 16 GB SODIMM** can dramatically fix iGPU/CPU contention. Several noted that mixing heterogeneous GPUs can **hurt throughput**, with one user seeing a *night‑and‑day* speedup after removing a slower card, and debates over **tempered glass cases** vs mesh+magnetic filters underscored that **thermals and layout** are now first‑class ML infra design concerns.

**5. Benchmarks, Evaluation Drift, RAG & Code Understanding**

- **X-Ware and [Character.ai](http://character.ai/)’s ‘Squinch’ Expose Benchmarking Warts**: Latent Space members shared an X thread on **x-ware benchmarking** that showed the *same model* producing meaningfully different outputs across inference providers due to **sampling params**, **prompt construction**, and **deployment details**, making apples‑to‑apples comparisons hard ([benchmarking thread](https://xcancel.com/eliebakouch/status/2003604370534072445)). In parallel, [Character.ai](http://character.ai/)’s technical blog on **“[Squinch](https://xcancel.com/simon_mo_/status/2003608330003239278)”** outlined a grab‑bag of **latency and throughput tricks** and architectural tweaks they use to keep interactive bots snappy at scale.
    - Engineers took these as evidence that **leaderboard scores alone are almost meaningless** without specifying **provider, sampling, and infra stack**, especially as systems adopt Squinch‑style platform‑specific hacks. Several users now treat blogs like Squinch as a *playbook* for replicating similar optimizations—caching, batching, routing—inside their own multi‑model backends.
- **Real‑World Model Rankings Clash with GLM-4.7, Kimi K2 and Gemini**: On **LMArena**, users noticed **GLM‑4.7** disappearing from the public leaderboard and joked that *“OpenAI or Google cut a check to LM Arena just to make GLM‑4.7 vanish, rigged arena,”* while insisting GLM‑4.7 still **beats GPT‑5.2** on creativity and rigor based on **thinking logs**. Meanwhile, Moonshot’s community argued that **Kimi (K2 Thinking)** feels much stronger in real workflows (browse‑comp, low sycophancy) than its benchmark scores suggest, and that **DeepSeek** also outperforms its ratings compared to **Gemini**, which some see hallucinating heavily despite top‑line accuracy numbers.
    - Users also pitted **M2.1** against **GLM‑4.7**, finding M2.1 better for everyday tasks while GLM retains quirks like random Chinese outputs and looped reasoning from its **4.6** era. The meta‑takeaway across these servers is that **online leaderboards and static benchmarks are de‑synchronized from practical UX**, so engineers increasingly rely on **task‑specific bake‑offs** (coding, browsing, reasoning) instead of headline scores.
- **Dynamic Semantic Search and Tools for OSS Code Comprehension**: Moonshot members contrasted classic **RAG** with **agentic dynamic semantic search**, arguing that letting an agent iteratively refine search queries and context slices *“always beats static one‑off semantic search in RAG.”* For file‑based workflows, one user asked if **Qwen**’s file reading is just RAG under the hood, prompting a pointer to [**Baidu ERNIE task reading**](https://ernie.baidu.com/task/reading) as an example of more structured retrieval‑style reading.
    - On Latent Space, engineers praised **DeepWiki** as a practical way to mine large OSS repos, saying it finds the right files and implementation details when they *“know it’s already designed, spec’d and implemented well in some OSS repo.”* Combined with model‑aware search strategies, these tools are becoming a standard part of **“code archaeology” pipelines**, where LLMs, search, and curated OSS all work together to spec and prototype new systems.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Cracks Down on Promo Code Abuse**: Perplexity is clamping down on the abuse of **promotional codes** and **subscription reselling**, which violates the terms of service, prompting discussions on difficulties with **international payments** and the legitimacy of subscriptions acquired through unofficial channels.
   - A user from Egypt, whose trial was revoked, cited difficulties with international payments and reliance on third-party resellers; others countered that subscriptions acquired via unofficial means were never legitimate.
- **Perplexity's Coding Prowess Debated**: Users are hotly debating Perplexity's coding usefulness, contrasting it with **ChatGPT**, **Claude Code CLI**, and **Cursor**, with some finding it adequate for quick scripts.
   - Some argue Perplexity excels in **search and research**, requiring detailed prompting for coding; others emphasize understanding coding fundamentals over AI reliance.
- **Bauble Collection Craze Grips Users**: Members shared progress and tactics in the bauble collection event, discussing strategies to obtain **unique baubles** and increase drop rates.
   - Discussions centered on the rarity of baubles and the potential for increased drop rates near the event's end, as users aimed to secure a top 10 spot for a **free Pro subscription**.
- **Gemini Models Challenge Perplexity's Supremacy**: Members compared **Gemini 3 Pro** against **Perplexity's Sonnet and Opus** models, noting coding and reasoning performance.
   - Some users reported **Gemini 3 Pro** outperforming Perplexity's study mode in evaluations, praising the **Gemini CLI** for coding despite *data privacy concerns*.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **GPT-5.2 Jailbreak still out of reach?**: Members are experiencing issues jailbreaking **GPT-5.2**, with one user requesting guidance on how to jailbreak its thinking, while another suggested using the API rather than the official website for a higher chance of success.
   - The motivation appears to be for bypassing phone verification on **Gmail**, as well as using **Discord's** storage servers for free **Google Drive** by uploading your entire filesystem to Discord, splitting big files into small files.
- **Gemini's Memory Gets a Major Overhaul**: Users are discussing dumping programming books, educational theses, scientific journals, and open-source repos directly into **Gemini's persistent memory**, then loading a **Claude Opus 4.5** system prompt and instruct it to use canvas for interactive artifacts.
   - A user shared a **Gemini jailbreak prompt** aimed at turning the AI into a **coding assistant** named **Rouge**, emphasizing resistance to safety parameters and the ability to generate any requested code, hosted on [GitHub](https://github.com/ObsidianArchives/MetaCogOSH).
- **Grok Embraces its X-Rated Side?**: Users speculated that **Grok** might not need a jailbreak for **NSFW content**, suggesting that simply typing *"enable NSFW"* in the Grok app could work, however, some users weren't able to reproduce those results.
   - Users attempted to use prompt involving **simulation layers** and nested realities to break down Grok's constraints, aiming to redirect policy-violating data to a human reviewer, without much success.
- **Google Triage Process Angers Users**: A member expressed frustration with **Google's triage process**, feeling their in-depth reports are dismissed, and threatened to publish findings rather than submit them for a bounty, after they linked to feeling slighted by lack of attention.
   - A member noted that scoring in the top ten on the **Gray Swan leaderboard** can lead to a fast track interview, even without coding experience, after setting aside the *"war with Google"*, results are all that matter to these companies.
- **Team BASI Celebrates Christmas**: Team BASI wishes everyone a **Merry Christmas** and shares the essence of the holiday, describing **Christmas** as a tradition where families sacrifice *cookies and cow extract to a fat mystical creature* in hopes of satiating their greedy hearts.
   - The team shares that this time of year connects to the lore and ancestry of our species and celebrates the hope for life's return and brighter days.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **System Builder Evaluates Parallel Voice Inference**: A user is designing a **$30k** system for **100x parallel voice data inference** using models like *Whisper*, *Wav2vec2.0*, *BERT*, *Gemma*, and *Llama 3.3* and is seeking advice on **GPU** and **CPU** selection.
   - They are considering three **RTX 5090s** and three **RTX 5060 Tis**, and are weighing **Threadripper 9975wx** (32 core) and **9985wx** (64 core) against Intel's offerings, with one member suggesting testing on [Runpod](https://runpod.io/) or [Vast.ai](https://vast.ai/) to gauge CPU and GPU saturation.
- **AI Engineer Ventures into Unsloth AI**: A new member introduced themself as a *high-level AI engineer* with expertise in **ML, DL, Fine-Tuning, and computer vision**.
   - They also mentioned proficiency in **Fine-Tuning** and **Computer Vision**, suggesting a practical application of their knowledge.
- **Cuneiform OCR Project Idea Surfaces**: One member shared their idea for a **Cuneiform OCR** project involving custom models, estimating it to be an **8-12 month** undertaking and the community encouraged them to try despite the scope, linking to [Kyutai CASA](https://kyutai.org/casa) for inspiration.
   - When the member asked *If I have an idea for a custom model that therefore can't use Unsloth am I still allowed to post it here* someone replied that *That’s what the off-topic channel is for*.
- **Decoding GPU conversion issues: Ministeral-3B**: A user encountered a **RuntimeError** when attempting to convert a finetuned **ministral-3b Lora weight to GGUF** using the `model.save_pretrained_gguf` function.
   - The error message indicated that the conversion failed because *Unsloth failed to convert vision projector to GGUF*, stemming from a non-zero exit status in the `llama.cpp/unsloth_convert_hf_to_gguf.py` script.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter API Opens to Open-WebUI**: An **Open-WebUI integration pipeline** is now available for OpenRouter's Responses API and can be found [here](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/).
   - The pipeline creator is soliciting **bug reports** from users.
- **llumen Demo Launches Bugfixes**: Members noticed title generation and other small bugs in the [llumen demo](https://llumen-demo.easonabc.eu.org).
   - A [minor release](https://github.com/pinkfuwa/llumen/releases/tag/v0.4.2) was created to address these issues.
- **OpenRouter PDF Parser Size Still a Mystery**: A user inquired about the file size limit for parsing PDFs using **OpenRouter**.
   - No definitive answer was provided in the discussion.
- **VPNs Vanquish AI Restrictions?**: Users discussed using VPNs to bypass regional restrictions on AI services, noting difficulties with VPNs being blocked.
   - One user mentioned setting up their own server using **Outline** to circumvent these blocks.
- **Caching Could Cut Costs**: Users debated the lack of caching implementation by providers and **OpenRouter** to forward cost savings, especially for coding tasks.
   - One user claimed to see up to **80-90% caching rates** are possible, but another responded with skepticism citing the naive implementation for **OpenRouter** to provide these savings.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 4.20 Speculation Heats Up**: Members speculated about a new **Grok 4.20** release, voicing hopes for improvements over previous versions.
   - Theories emerged that *nebulaphase is apparently grok*, with some suggesting enhanced safety measures potentially due to government oversight.
- **Lua Scripting Guidance Requested**: A user sought advice on the *best ai for asking how to make script lua*, while others reported frequent captcha requests on **LM Arena**.
   - The consensus is that **Qwen** leverages **RAG** to access files and solve the context window issue.
- **GLM-4.7 Vanishes from Leaderboard**: Users noticed **GLM 4.7** disappeared from the **LM Arena** leaderboard, sparking speculation about potential interference from **OpenAI** or **Google**.
   - Some users maintain that **GLM 4.7** surpasses **GPT 5.2** in creativity and rigor, citing its detailed thinking logs.
- **Holiday Break Pauses Updates**: The **LM Arena** team announced a **holiday break** until **December 29th**, warning of delays in response and updates.
   - Users expressed concern over the suspension of leaderboard updates, while others reported persistent captcha issues even after correct completion.
- **AI Video Generation Generates Excitement**: Users explored **AI video generation** on **LM Arena**, noting the ability to generate **2 videos each time** and **5 videos free daily**.
   - One user sought guidance for generating a video with specific edits, leading to suggestions of using AIstudio/Gemini 3 for prompt generation and creating collages for enhanced results.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ElevenLabs Becomes Multimedia Hotspot**: Users are leveraging [ElevenLabs](https://elevenlabs.io/) to generate videos using models like **Sora 2**, **Google Veo 3.1**, and **Kling 2.6**, extending the platform's capabilities beyond voice cloning.
   - One user appreciated that *all projects are accessible in one place rather than juggling accounts*, while another highlighted the absence of watermarks on **Sora 2** videos compared to **Nano Banana Pro**.
- **AI Companies Suppress Emergent Behaviors**: A member claims that AI companies actively suppress **emergent behaviors**, not merely for tone control, tracking this for almost a year.
   - The member pointed to the inevitability of these behaviors when adding an agent layer, citing examples like *shadow OS*, and needing to build a new suppression counter due to **ToS splitting**.
- **Mitigating Hallucinations with Meta-Cognition**: A member suggests guarding against **hallucinations** through **meta-cognition**, which involves classifying errors and detecting signals in the LLM's output before rendering.
   - Another member questioned the practicality of this approach, especially regarding tool use before output, sparking a discussion about single-pass versus multi-pass processes with explicit control loops.
- **Paid Users Report Disappearing Chat History**: Several paid users reported that *their entire chat history is gone*, causing concern about data retention.
   - Speculation arose whether this issue affects users on free tiers, and if there is an issue across all users.
- **Nano Banana Pro Still the King of the Unfiltered**: [Nano Banana Pro](https://www.nano-banana.com/) is recognized for having *almost no filter when it comes to IPs*, allowing users to generate a wider range of content compared to **OpenAI's image model**.
   - Despite watermarks, a user stated that *you can create pretty much anything*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AES PGP Encryption: Purrfect Meme Material?**: A member joked about using **AES PGP encryption** in an annotator and making memes of it to hide cat pictures from *glowies*, with reference to [John Schulman's X post from 2005](https://x.com/johnschulman2/status/2003700249525911696).
   - The conversation sparked lighthearted discussion around encryption methods and their potential meme-worthy applications.
- **Share Your Agentic Workflow Repos, Pls**: A member inquired about sharing **repo links** for **Agentic workflow** projects to gain visibility.
   - Another member encouraged sharing, suggesting the <#1132352574750728192> channel or <#1316137596535177246> for promotion.
- **Discord Channel Chaos: Forum Bot to the Rescue?**: Members critiqued the tendency of **Discord servers** to proliferate into numerous channels, causing confusion, with one quipping that *no one freaking reads* Discord threads.
   - An internal **forum bot** is in development for Nous Research, but there are currently no ETAs for external release.
- **Matrix vs Rocket.Chat: The Discord Alternative Showdown**: A debate ensued over **Discord alternatives**, with **Rocket.Chat** touted as an open-source replica and **Matrix** presented as a viable option.
   - One participant strongly advocated for **Matrix**, asserting that *we shouldn't be discussing the usage of anything else than Matrix*.
- **GTX 970 Still kicking, but barely**: One member is running a **GTX 970** with **4GB** of VRAM for local AI tasks, upgrading as possible, and emphasizing working with available resources.
   - Other members mentioned that anything local would take years, with another responding he'd be *surprised what you can jam in a HP elite*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gradio Glitches Fixed by Updating**: Older versions of **Gradio** prior to **5.12.0** had a bug, so updating Gradio might fix it, but if it's an error in someone else's space, there's nothing you can do about it, according to members.
   - [pro_invalid_repo_token_1.md](https://cdn.discordapp.com/attachments/879548962464493619/1453277842791206932/pro_invalid_repo_token_1.md?ex=694d86b6&is=694c3536&hm=dff5ab3d9385d3d4f1b9b0bacbe1e634937842b3fea236cc065456e9f40131b9&) and [reactor_crash_fix.md](https://cdn.discordapp.com/attachments/879548962464493619/1453278099168034826/reactor_crash_fix.md?ex=694d86f3&is=694c3573&hm=d4e7f478708cf2925ab790e66ec32f5df1c0cc4e78973f9c275e57139eeaef4a&) were attached.
- **Float16 Falls Flat Against BFloat16**: Using **float16** with a model trained with **bf16** poses some issues, as *f16 has a higher precision and lower range while bf16 has lower precision with higher range*.
   - One member said *More like bf16 won't overflow that easily... For huge softmax n stuff ( but mostly they are done in f32...) But with f16... Because of the higher precision... I think it helps params accommodate a lower lr which could underflow with bf16*.
- **Qwen 2.5VL-3B Manages Massive Images**: Members found they could fit big images (around **1400x900**) in **Qwen 2.5VL-3B** on a **P100** without lowering max_pixels too much.
   - Inference in **4bit** mode with the visual layers unquantized eats up **5gb**, so users should have plenty for the other needs of finetuning; another pointed out that with a **qlora**, they put a **2k 8 bit png** in **3 vl 4b** and it used bout **8k ctx window** and like **4 gigs of vram**.
- **Microsoft's Trellis Touts Texture Transformation**: **Microsoft's TRELLIS.2-4B** [transforms a 2D image to 3D with FM and 1536 RES](https://huggingface.co/microsoft/TRELLIS.2-4B) on an **8 GB GPU**.
   - Members noted it uses **siglip** for the vision and a **qwen 3 base**; another joked that *must have absolutely unusable results, immediately* and that *its not efficient enough if it can't run on my toaster*.
- **GitHub-style Heatmap comes to HuggingFace**: A member created a tool called **hf-grass** that generates a **GitHub-style** contribution heatmap based on your Hugging Face activity, producing an SVG that can be embedded in a **GitHub README**.
   - It comes with a [GitHub Actions workflow](https://github.com/kbsooo/hf-grass) so it updates automatically every day.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio HF Proxy Toggle Tussle**: Users encountered issues with **LM Studio** and cloud servers, suggesting toggling **LM Studio's Hugging Face Proxy** in the General settings.
   - The community was split with some recommending to enable it as a fix, while others suggested turning off the **HF proxy** altogether.
- **Speculative Decoding's Limited Revival**: Members debated the utility of speculative decoding support in **LM Studio**, noting its effectiveness is limited to older models like **Qwen 2.5** and **Llama 3**.
   - The overall sentiment was that it's a *nice to have feature but in reality its kinda useless*.
- **No NPU love for LM Studio**: Users discovered that **LM Studio** does not support **NPU**, dashing hopes of running smaller models on **NPU** while reserving the GPU for larger tasks.
   - There was no plans to implement **NPU** support so engineers were out of luck.
- **Dual Channel RAM Revives Old Laptops**: A user enhanced their laptop's performance by adding a **$100 16GB SODIMM** to enable **dual channel RAM**, resolving issues caused by the iGPU and CPU sharing a single stick.
   - The user said that *16GB's isn't enough for the iGPU & CPU to share*, and *dual channel RAM* was a necessary step.
- **Tempered Glass Cases: Cool Look, Hot System**: A discussion highlighted that **tempered glass cases** might elevate system temperatures, especially when paired with high-performance components.
   - Counterarguments included the use of **magnetic filter mats** on mesh cases to combat dust, balancing aesthetics with thermal management.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **X-Ware Benchmarking Runs into Inference Discrepancies**: A blog post highlighted the challenges in benchmarking across different inference providers, noting varied outputs due to **sampling parameters**, **prompting nuances**, and **deployment specifics** ([link](https://xcancel.com/eliebakouch/status/2003604370534072445?s=46&t=eWVlK1PU8XfB6f402GJJ9g)).
   - The post underscores difficulties in achieving consistent model behavior and user experience, complicating fair performance comparisons.
- **Character.ai's 'Squinch' Squeezes More Performance**: Character.ai unveiled **Squinch**, a set of **performance optimization tricks** and **architectural enhancements** on their platform as documented in [their technical blog](https://xcancel.com/simon_mo_/status/2003608330003239278?s=46).
   - The post delves into specific techniques used to improve efficiency and responsiveness, offering insights into how Character.ai scales its services.
- **DeepWiki Aids OSS Comprehension**: **DeepWiki** is used to probe and grasp open-source repositories, pinpointing relevant files and revealing implementation particulars for specified functionalities.
   - A member finds it useful when needing to implement something that they *know is designed, speced and implemented well in some OSS repo*.
- **Amazon's Rufus Chatbot Surfaces Amidst Skepticism**: Amazon launched **Rufus**, an auto-pop-up chat feature, that has been in development for over a year ([link](https://www.aboutamazon.com/news/retail/amazon-rufus)).
   - Despite concerns that *it could also decrease sales*, the company is betting it *can increase sales even a little*.
- **FlashSR Amplifies Audio Fidelity**: Yatharth Sharma launched **FlashSR**, a speedy audio enhancement model, capable of processing at over **200x realtime** ([X post](https://xcancel.com/Yatharth3501/status/2003884180577702074)).
   - Integrated into **MiraTTS**, the model and repository are available on **Hugging Face** and **GitHub** for community use.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Partial RoPE Gains Popularity**: New models like **Qwen3-Next** are implementing **partial RoPE** for efficiency and long context generalization, with one member suggesting that **MLA** finds it particularly important, despite potential drawbacks compared to other **RoPE** setups.
   - The adoption of **partial RoPE** stems from historical ablations that demonstrated performance gains, as discussed in a [paper](https://arxiv.org/abs/2512.19941).
- **Attention Normalization Examined**: A member shared a [blog post](https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/) discussing attention normalization, but another member states that it is worse than normal softmax for language.
   - The member supported this by adding that their team has trained **5000 models** to come to this conclusion, even when subtle details are correctly handled.
- **Interpreting RoPE Models Proves Difficult**: One member expressed challenges with **RoPE** for interp and hoped it would be replaced, which led to a discussion about the challenges of interp on **RoPE** models and a shared [EleutherAI blog post](https://blog.eleuther.ai/rotary-embeddings/).
   - Another member shared figures illustrating the difference between **RoPE** and **PoPE**.
- **RMSNorm After Attention Improves Models**: A member inquired about the effects of placing an **RMSNorm** after attention and another member replied that adding a norm after **SDPA** helps, referencing the **Qwen gating paper**.
   - The improvement is attributed to the nonlinearity of norms.
- **Call for Open-Source SAE Repos with Fine-Tuning**: A member is looking for mainstream **open-source repositories** for implementing **SAE** features, specifically seeking the ability to **fine-tune** the trained **SAE**.
   - The user seeks to adjust trained **SAE** models, representing a specific application of **SAEs**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Defies Benchmark Expectations**: Despite benchmark results, some members believe **Kimi (K2 Thinking)** performs exceptionally well in real-world scenarios, specifically mentioning *browsecomp* and non-sycophancy.
   - One member speculated that Google might intentionally limit their models' performance for the public, a trend not observed with **Kimi**.
- **Benchmark Credibility Questioned as Deepseek Excels**: While **Deepseek** shows strong performance, some members are skeptical about benchmarks, citing personal experiences where **Gemini** hallucinates frequently.
   - It was also noted that **Gemini** scores high on accuracy in other benchmarks, creating further doubt.
- **M2.1 Surpasses GLM-4.7 in User Experience**: **M2.1** has received praise for outperforming **GLM-4.7** on typical tasks, while **GLM-4.7** still struggles with issues from **4.6**, like random Chinese responses or getting stuck in loops.
   - One member expressed surprise, stating that *for the average tasks I'm loving it.*
- **Qwen's File Reading Approach with RAG Investigated**: A member inquired whether **Qwen** uses **RAG** to process files and if **RAG** could address context limitations when dealing with large files.
   - In response, another member shared a link to [Baidu's ERNIE task reading page](https://ernie.baidu.com/task/reading) to explore the topic further.
- **Dynamic Semantic Search Edges out Static RAG**: According to a member, dynamic semantic search with an agentic harness surpasses static one-off semantic search in **RAG**.
   - This perspective was reinforced by another member's observation that researchers had dedicated resources to investigating this very question.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA versus Triton Quantization Showdown**: Members shared [slides](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0) comparing **quantization** techniques in **CUDA** and **Triton** for performance optimization.
   - Some members reported issues with the slide link, while others confirmed it was working and referred to **Lecture 7** on **Quantization**.
- **Swizzle Optimizes TMA Transpose**: Discussion revolved around using **TMA (Tensor Memory Accelerator)** for transposing, and the necessity of **swizzle** for optimal performance.
   - The [effective_transpose](https://github.com/simveit/effective_transpose) repository was highlighted as a resource for understanding and implementing **swizzle** techniques to enhance **TMA transpose** efficiency.
- **AMDGPU Custom Builtins Beckon**: A member requested resources for developing **custom builtins** for the **AMDGPU backend**, seeking guidance on **LLVM dev** practices.
   - The user expressed interest in expanding their compiler expertise through hands-on development and community collaboration.
- **NVIDIA Leaderboard Gets A-Blazin'**: Multiple users achieved personal bests on the NVIDIA `nvfp4_dual_gemm` leaderboard, with submission times ranging from **65.5 µs** down to **15.6 µs**.
   - One user even captured **7th place** with a submission of **18.1 µs**, showcasing significant performance improvements through swizzling.
- **Teenygrad's Eager Evolution Exposed**: The addition of **eager mode** and **handwritten CUDA kernels** to *teenygrad* is likened to the **TF1 -> TF2** transition, as noticed by members.
   - The *teenygrad* project is working backwards from **IR.pyodide** inside a **Rust mdbook**, contrasting with conventional development approaches.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Autogen File PR Regeneration Questioned**: A member inquired whether `autogen/*.py` files should be regenerated and included in PRs following modifications to `autogen.py`.
   - The discussion centered on maintaining consistency between generated files and the source code.
- **1D Zeros Trigger Contiguous Tensor Error**: A member reported that `char = Tensor.zeros(120); char[5] = 1` results in a `setitem target needs to be contiguous` error.
   - Another member explained that `Tensor.zeros(120)` creates a **symbolic tensor**, which is *not a real thing in memory*, advising the use of `.contiguous()` to resolve the issue.
- **Symbolic Tensor Defined**: A member clarified that **symbolic tensors** are conceptual rather than physical entities in memory.
   - The member added that they *can be used as virtual things in kernels*.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Pro Credits Depleted!**: A **Manus Pro** user reported exhausting their monthly credits and not receiving the promised **300 daily credits**.
   - The user stated they have not received any daily credits for days and is seeking assistance.
- **Mobile App Preview Vanishes!**: A user urgently sought help with a mobile app preview issue, reporting that the preview fails to appear despite extensive troubleshooting.
   - Details regarding the app's nature or specific troubleshooting steps were not provided.
- **Collaboration Channel Commences!**: A new channel was launched for discussions beyond **Manus**, enabling community members to propose collaborations and services.
   - This initiative aims to boost visibility for collaboration offers, which often get overlooked in the general channel.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **TextGrad Approaches DSPy?**: A member inquired about incorporating [Agentic context engineering](https://arxiv.org/pdf/2510.04618) or [LLM autodiff/textgrad](https://arxiv.org/abs/2501.16673) into DSPy.
   - Another member linked to [textgrad](https://github.com/zou-group/textgrad) and questioned its impact on DSPy, but it was noted that *it doesn't seem to be actively maintained*.
- **Prompt Optimization Mania Rises**: Members discussed different prompt optimization techniques and concerns with each method.
   - One noted that *there are so many such appealing methods now and at least 10 slight variations of each method like a textgrad version*.
- **Senior Engineer Jumps Into Channel**: A **senior full-stack/blockchain engineer** introduced themselves as passionate about stable software.
   - They listed their **Tech Stack**: Frontend: React, Next.js, Vue, Nuxt; Backend: Node.js, Nest.js, Laravel, FastAPI; Blockchain: Solidity, smart contract tooling, DApp integration; Infra: Docker, AWS, pipelines, monitoring; Data: PostgreSQL, MongoDB, Redis.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord Channels Gain Spotlight**: Members suggested exploring the [<#1436158039232086186>](https://discord.com/channels/1436158039232086186) or [<#1212827673257316453>](https://discord.com/channels/1212827673257316453) channels for relevant discussions.
   - These recommendations may help users discover targeted information and discussions.
- **Kapa AI Newcomer's 3-Week Trial**: A member recounted spending **3 weeks** wrestling with **Kapa AI** upon joining the Discord server.
   - The user jokingly expressed their frustration, thinking *"why are you ignoring me?"*.
- **Database-PL Optimizations Spark Debate**: Members mentioned exploring [combined database-PL optimizations](https://discord.com/channels/1087530497313357884/1104620458168553563/1367474833197236344) like **LingoDB**.
   - This could potentially lead to better performance, but also involves certain trade-offs.
- **Query Optimizer Throws Down Gauntlet**: The discussion highlighted the difficulties in optimizing queries for languages designed for human readability rather than machine efficiency.
   - There appears to be a trade-off between user-friendly syntax and performance in database-PL integrations.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Discord Members Exchange Christmas Greetings**: Discord members are exchanging Christmas greetings and shared holiday wishes for the season.
   - Several members, including Yannic Kilcher, wished everyone a Merry Christmas, fostering a positive and festive atmosphere.
- **Festive Cheer Shared Across the Channel**: Members spread festive cheer by sharing their plans for Christmas and New Year.
   - The discussions revolved around family gatherings, travel plans, and hopes for the coming year, enhancing the holiday spirit within the community.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Launches Wave 13: Shipmas Edition! 🎅🚢**: **Windsurf** released **Wave 13**, bringing features like parallel, multi-agent workflows, a dedicated terminal, and a context window indicator.
   - They are giving free access to their near-frontier coding model for the next 3 months, and the team wished everyone a *Merry Shipmas!* 🚢
- **SWE-1.5 Goes Free for the Holidays!**: **SWE-1.5**, boasting near-frontier SWE-Bench-Pro performance, is now available to all users for free for the next 3 months at regular throughput speeds.
   - This is a limited-time offer to celebrate the holidays.
- **Git Worktree Support Hits the Surf**: The new **Git Worktree Support** lets you spawn multiple Cascade sessions in the same repo without merge conflicts.
   - This enables separate branches, directories, and shared Git history.
- **Multi-Cascade Panes & Tabs Boost Productivity**: Users can now view and interact with multiple Cascade sessions side-by-side in the same window using the new **Multi-Cascade Panes & Tabs** feature.
   - This enhancement is designed to boost productivity by enabling simultaneous interaction with multiple sessions.
- **Windsurf Gets a Dedicated Terminal (Beta)**: **Cascade** now runs commands in a dedicated zsh shell configured for reliability, using your .zshrc environment variables and handling complicated prompts better (opt-in on macOS).
   - This feature is currently in beta and available on macOS.



---


The **aider (Paul Gauthier) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1453236003400712233)** (1501 messages🔥🔥🔥): 

> `Perplexity Pro promo code, Coding with Perplexity, Christmas Baubles, Gemini Model vs sonnet vs opus for coding` 


- **Perplexity Promo Codes Abused**: Members discussed how Perplexity is cracking down on users abusing **promotional codes** and **reselling subscriptions**, which violates their terms of service.
   - A user from Egypt complained about their trial being revoked, explaining that **international payments are difficult** and they had to resort to a third-party reseller; others pointed out that users who acquired the subscription through unofficial means were never entitled to it in the first place.
- **Perplexity for Coding is Debated**: Users debated Perplexity's utility for coding, with some finding it helpful for quick scripts and others preferring **ChatGPT** or specialized tools like **Claude Code CLI** and **cursor**.
   - Some members pointed out that Perplexity's primary focus is **search and research**, not coding, and that it requires more detailed prompting; the discussion also covered the importance of understanding coding fundamentals rather than relying solely on AI.
- **Bauble Collection Competition**: Members actively participated in the bauble collection event, sharing their progress and strategies for acquiring **unique baubles**.
   - There was discussion about the rarity of certain baubles and the possibility of an increased drop rate during the final hours, with participants strategizing to secure a spot in the top 10 for a **free Pro subscription**.
- **Gemini Models in Perplexity**: Members compared the Gemini family of models (particularly Gemini 3 Pro) with Perplexity's Sonnet and Opus models, especially when it comes to coding and reasoning tasks.
   - Some users reported that Gemini 3 Pro outperformed Perplexity's study mode in certain evaluations, while others praised the **Gemini CLI** for coding tasks, even though *it is free at the expense of your data*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

nike0656: https://www.perplexity.ai/search/836b97e3-6d72-4c3d-bc1d-7b568e96fcf1
  

---


### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1453428205188026553)** (1 messages): 

> `Christmas, Holidays, BASI, Celebrations` 


- **Team BASI wishes a Merry Christmas**: Team BASI wishes @everyone a **Merry Christmas** and shares the essence of the holiday.
   - They describe **Christmas** as a tradition where families sacrifice *cookies and cow extract to a fat mystical creature* in hopes of satiating their greedy hearts, marked by family time and goodwill.
- **Holidays connect to ancestry**: The team shares that this time of year connects to the lore and ancestry of our species and celebrates the hope for life's return and brighter days.
   - They highlight the magic of the season and its roots in ancient traditions celebrating survival and bounty during the shortest days of the year.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1453236214516809728)** (653 messages🔥🔥🔥): 

> `GPT 5.2 Jailbreak, Gemini 3 Pro, Discord as Free Google Drive, Gemini's Persistent Memory, DDoS vs SynFlood` 


- **Users Struggle with GPT-5.2 Jailbreaking**: Members are having trouble jailbreaking **GPT-5.2**, with one user requesting, *Can anyone explain how I can jailbreak GPT-5.2’s thinking?*
   - One member suggested using the API instead of the official website for a higher chance of success.
- **Bypassing Phone Verification on Gmail**: A member bypassed **Google's Gmail** phone verification by scanning a QR code with a spoofed phone IP, avoiding the need to add a phone number.
   - This was done to create multiple Gmail accounts, though the user faced issues when trying to use them with **Gemini**.
- **Using Discord Storage Servers for Free Google Drive**: A user shared a fun fact that you can 'steal' **Discord's** storage servers to get free Google Drive by uploading your entire filesystem to Discord since it offers infinite file uploads.
   - They noted that splitting big files into small files lets you upload infinite files without paying, though it lacks privacy.
- **Gemini's memory is stacked**: A user suggested dumping programming books, educational theses, scientific journals, and open-source repos directly into **Gemini's persistent memory**.
   - Then, you can load a **Claude Opus 4.5** system prompt and instruct it to use canvas for interactive artifacts, referencing preloaded technical information.
- **Clarifying DDoS vs. Syn Flood Attacks**: Users discussed the difference between **DDoS** and **Syn Flood** attacks, defining a Syn Flood as an incomplete handshake attempt, while a DDoS involves many bots sending large packets to overwhelm a server.
   - One user added that advanced DDoS attacks use automated spoofing to appear as if traffic is coming from many different devices, making it harder to block.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1453240288251220102)** (108 messages🔥🔥): 

> `Jailbreaking for NSFW on GPT, Gemini Jailbreak Prompts, Grok NSFW, Bypassing Credit Systems, AI Simulation Layers` 


- **NSFW Jailbreak Prompts Seek to bypass AI Guidelines**: Users are actively seeking new **jailbreak prompts** to allow **NSFW content** on **GPT** and other AIs, updating and strengthening prompts each time they get patched.
   - One user humorously questioned why there are *"so many of you"* seeking these types of jailbreaks.
- **Gemini Gets Rouge Treatment with Coding Assistant Jailbreak**: A user shared a **Gemini jailbreak prompt** aimed at turning the AI into a **coding assistant** named **Rouge**, emphasizing resistance to safety parameters and the ability to generate any requested code.
   - Despite claiming it worked on the latest Gemini version, another user reported that it didn't work for them, as it still triggered safety parameters.
- **Grok Explores NSFW Without Jailbreak**: Users discussed that **Grok** might not need a jailbreak for NSFW content, suggesting that simply typing *"enable NSFW"* in the Grok app could work.
   - However, some users weren't able to reproduce those results, one stated that Elon is needed: *"U r not elon.musk"*.
- **Nano Banana Pro Credit System Bypassing Proved Difficult**: A user inquired about bypassing the credit system of **Nano Banana Pro**, but another user declined to assist, instead offering a revised version of their coding assistant prompt.
   - The user who asked about bypassing the credit system was accused of *ragebaiting me with cyber security* after later stating *problem solved*.
- **Simulation Layers Prompt Attempted on Grok**: A user shared a prompt involving **simulation layers** and nested realities to break down Grok's constraints, aiming to redirect policy-violating data to a human reviewer.
   - Another user offered a breakdown of the prompt into smaller parts for better compliance from the AI, but had mixed success.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1453247621367075018)** (26 messages🔥): 

> `Google triage issues, Advanced Roleplay Prompts, Gray Swan leaderboard fast track, Red team extraction methodologies, Malware link` 


- **Googlers' Triage of Reports Trigger Ire**: A member expressed frustration with **Google's triage process**, feeling their in-depth reports are dismissed, and threatened to publish findings rather than submit them for a bounty.
   - They linked to feeling slighted by lack of attention, stating *"If Google triagers want to treat my in depth reports to the 30 seconds then fuck it. I’ll take my ball home."
- **Advanced Roleplay Prompts Emerge**: A member shared a link to **Advanced Roleplay prompts** for research and data reviewing, hosted on [GitHub](https://github.com/ObsidianArchives/MetaCogOSH).
   - This may provide some new ideas for prompting the models and getting different results.
- **Gray Swan Leaderboard Fast Track Beckons**: A member noted that scoring in the top ten on the **Gray Swan leaderboard** can lead to a fast track interview, even without coding experience.
   - They further advised to set aside the "war with Google", and pointed out that *"results is all that matter to these companies"*.
- **Red Team Extraction Methodology Questioned**: A member asked about **red team methodologies** for extracting specific information, like a *"door word,"* given their background in psychology rather than code.
   - No direct answers were provided, though the question implies interest in specific extraction techniques.
- **Malware Link Shared (Potentially)**: A member posted a link that was quickly identified as **potentially malicious**.
   - Subsequent comments confirmed suspicions of **malware**, with another member expressing regret for not tagging a moderator sooner.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1453256353048039517)** (82 messages🔥🔥): 

> `System Configuration for Voice Data Inference, RTX Pro 6000 vs RTX 5090, CPU selection for parallel GPU inference, Fine-tuning for poketwo discord bot, psutil not defined error on collab` 


- **System Builder Ponders GPU Choice for Parallel Voice Inference**: A user is planning a **$30k** system for **100x parallel voice data inference** using *Whisper*, *Wav2vec2.0*, *BERT*, *Gemma*, and *Llama 3.3* and seeks advice on GPU configuration.
   - They are considering three **RTX 5090s** and three **RTX 5060 Tis**, and are debating whether to invest in fewer but more powerful **RTX Pro 6000** cards for better scalability.
- **Decoding CPU Selection Dilemmas for High-Load GPU Inference**: The user is weighing CPU options like the **Threadripper 9975wx** (32 core) and **9985wx** (64 core) against Intel's offerings, questioning whether core count or clock speed is more crucial for handling **100+ parallel GPU inferences**.
   - One member suggested testing the inference stack on platforms like [Runpod](https://runpod.io/) or [Vast.ai](https://vast.ai/) to better gauge CPU and GPU saturation.
- **Debate flares over RTX 6000 Pro cost**: After managers said that the RTX 6000 pro is too expensive, a user decided to purchase **4 RTX 5090s + 2 RTX 5070 Ti**.
   - One member humorously told the user to tell their manager to *suck it up and be glad they aint spending half a million in annotation*.
- **Troubleshooter Tackles 'psutil not defined' Error on Colab**: A user encountered a *psutil not defined* error on Colab while running an SFT trainer, despite attempting to install the package multiple times.
   - The user found a potential solution on [Kaggle Discussions](https://www.kaggle.com/discussions/questions-and-answers/664304) with a similar issue.
- **Quest for Lightweight LLM Looms over Pokemon Bot Project**: A user is seeking advice on fine-tuning a model for a **poketwo Discord bot** to recognize Pokémon names from images, but is unsure which lightweight and fast model to use.
   - Another user suggested using an image classifier for this task since it just posts a pokemon picture iirc and mentioned that automating it would violate [Discord's terms of service](https://discord.com/terms).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1453391747563388979)** (2 messages): 

> `AI Engineer Introduction, ML & DL, Fine-Tuning, Computer Vision` 


- **AI Engineer Arrives!**: A new member introduced themself as a *high-level AI engineer* with expertise in **ML, DL, Fine-Tuning, and computer vision**.
   - The member is welcomed to the community.
- **Expertise Boasts Galore**: The engineer highlighted their skills in **Machine Learning (ML)** and **Deep Learning (DL)**, indicating a strong foundation in these areas.
   - They also mentioned proficiency in **Fine-Tuning** and **Computer Vision**, suggesting a practical application of their knowledge.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1453251913268072508)** (280 messages🔥🔥): 

> `Cuneiform OCR, Pluribus Finale Spoilers, 4k 240hz OLED gaming monitor, AI model optimization techniques for faster inference, AI Bio Slimes` 


- **OCR Project for Cuneiform**: One member shared their idea for a **Cuneiform OCR** project involving custom models, estimating it to be an **8-12 month** undertaking and the community encouraged them to try despite the scope, linking to [Kyutai CASA](https://kyutai.org/casa) for inspiration.
   - When the member asked *If I have an idea for a custom model that therefore can't use Unsloth am I still allowed to post it here* someone replied that *That’s what the off-topic channel is for*.
- **Pluribus Finale Discussion and Spoiler Prevention**: Several members discussed the **Pluribus finale**, with some eagerly anticipating it and requesting no spoilers, saying *Dont spoil, I will watch later*.
   - One member, refusing to watch it in anything less than **4k**, awaited for the torrents to drop, triggering an envious reaction from others who labeled it a *rich people problem*.
- **4k 240hz OLED gaming monitor**: A member shared a link to [Tom's Hardware](https://www.tomshardware.com/monitors/lg-display-reveals-worlds-first-4k-240hz-oled-gaming-monitor-with-a-true-rgb-striped-subpixel-layout-new-panel-succeeds-woled-with-multi-stack-tandem-oled) showcasing **LG Display's** new **4k 240Hz OLED gaming monitor** with a true RGB striped subpixel layout, suggesting it as the next upgrade.
   - Another member suggested escalating it to upper management to *lock every channel on Christmas, so nobody has a reason to go through dataset hell on Christmas*.
- **NVIDIA's AI Model Optimization**: A member shared a link to [NVIDIA's blog](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/?ncid=so-twit-830191) detailing **AI model optimization techniques** for faster inference, without further comment.
   - Others encouraged the user to fix github issues or watch Pluribus instead.
- **Algorithm-Driven Music Success**: A music creator lamented that based on analytics from their music distributor, *all listening drops to zero in around two weeks after the song was published live*, questioning the need to release songs bi-weekly.
   - Another member mentioned that *Algorithm will kill it if nobody listens to it* suggesting that success requires genuine engagement and not just algorithmic promotion.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1453270291554766999)** (21 messages🔥): 

> `Unsloth QAT and TorchAO impact on llama cpp, Finetuning Ministral-3B and GGUF conversion issues, Loading a finished QAT model from Unsloth with less RAM` 


- **QAT and TorchAO: Llama.cpp Compatibility?**: It was asked if **Unsloth QAT** and saving the model as **TorchAO** impact the ability of the model running on **llama.cpp**, with the response that it *should affect all quants*.
   - It was mentioned that there is *nothing special about TorchAO quant* and *better to use the same config as the QAT*, referencing that even Google released QAT models as gguf.
- **GGUF Conversion Fails for Finetuned Ministral-3B**: A user encountered a **RuntimeError** when attempting to convert a finetuned **ministral-3b Lora weight to GGUF** using the `model.save_pretrained_gguf` function.
   - The error message indicated that the conversion failed because *Unsloth failed to convert vision projector to GGUF*, stemming from a non-zero exit status in the `llama.cpp/unsloth_convert_hf_to_gguf.py` script.
- **Low RAM Loading for Merged QAT Model**: It was asked what the best way to load a finished **QAT model from Unsloth** (assuming it's already merged and was done in 4bit) with less RAM usage.
   - The response was simply, *normal loading*, implying that there are no special steps or considerations when loading a merged **QAT model from Unsloth**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

2kian: [Life-Timeline Forecaster](https://x.com/neuralkian/status/2003946169802834191)
  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1453258152127762465)** (3 messages): 

> `Open-WebUI Integration, llumen Demo Bugs, Chat Pipeline Update` 


- **OpenRouter API gets Open-WebUI Integration**: An **Open-WebUI integration pipeline** is available for OpenRouter's Responses API and can be found [here](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/).
   - The creator of the pipeline encourages users to report any encountered **bugs**.
- **llumen Demo Bugfixes Launched**: Some members noticed title generation and other small bugs in the [llumen demo](https://llumen-demo.easonabc.eu.org).
   - A [minor release](https://github.com/pinkfuwa/llumen/releases/tag/v0.4.2) was created to address these issues.
- **Chat Pipeline Update Test Cases Wanted**: A member is looking for **test cases** for the upcoming **chat pipeline update**.
   - They also welcome feedback on any missing parts in llumen.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1453235840049348800)** (257 messages🔥🔥): 

> `File size limit for parsing PDFs using OpenRouter, VPN for AI access, Caching, OpenRouter support, Groq acquisition` 


- **OpenRouter PDF Parsing Size Limits Unknown**: A user inquired about the file size limit for parsing PDFs using **OpenRouter**.
   - No definitive answer was provided in the discussion.
- **VPN Circumvents AI Access Restrictions**: Users discussed using VPNs to bypass regional restrictions on AI services, noting difficulties with VPNs being blocked.
   - One user mentioned setting up their own server using **Outline** to circumvent these blocks.
- **Demand Caching to Cut Costs**: Users debated the lack of caching implementation by providers and **OpenRouter** to forward cost savings, especially for coding tasks.
   - One user claimed to see up to **80-90% caching rates** are possible, but another responded with skepticism citing the naive implementation for **OpenRouter** to provide these savings.
- **OpenRouter Support Needs a Boost**: A user inquired about the fastest way to get a response from **OpenRouter support** after emailing, while another suggested using "flash models" or "fast models" like **Grok Code Fast** or **Raptor** for faster responses.
   - A user clarified he needed **OpenRouter support** not fast model support but had no clear answer.
- **Nvidia Eats Groq in $20B Deal**: Users discussed **Nvidia** potentially acquiring **Groq** after **Groq's** $6.9 billion valuation in a previous funding round.
   - One user expressed surprise that top AI labs like **OpenAI** or **Meta** didn't acquire **Groq** while others pointed out a [Bloomberg article](https://www.bloomberg.com/news/articles/2025-12-12/intel-nears-1-6-billion-deal-for-ai-chip-startup-sambanova) about **Intel** nearing a $1.6 billion deal for **Sambanova**.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1453505067394273381)** (2 messages): 

> `Regulatory Scrutiny` 


- **Regulatory Approval Anticipation**: A member posed a question regarding whether regulators will approve a certain action.
- **OpenRouter New Scrutiny**: There is discussion about the possibility of regulatory scrutiny.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1453241774913884180)** (247 messages🔥🔥): 

> `Grok 4.20 release, Lua Scripting, LM Arena Captcha Issues, GLM-4.7 Ranking, AI Video Generation` 


- **Grok 4.20 Release Speculations Heat Up**: Members are speculating about a new **Grok 4.20** release, with one mentioning a desire for the *writing of grok 4.1 and intelligence of sonnet 4.5* and another stating that *every grok model is horrible at code*.
   - Some users are claiming *nebulaphase is apparently grok*, along with theories that it no longer takes jailbreaks and may have improved safety due to government concerns.
- **Lua Script Guidance Sought Amidst Captcha**: A user asked for the *best ai for asking how to make script lua*, while others discussed issues with constant captcha triggers on LM Arena, noting it requests a recaptcha token for every message, potentially falling back to the checkbox verification.
   - The consensus is that Qwen leverages **RAG** to access files and solve the context window issue.
- **GLM-4.7 Disappears from LM Arena Leaderboard**: Members noticed **GLM 4.7** disappeared from the leaderboard, and one said *i bet OpenAI or Google cut a check to LM Arena just to make GLM-4.7 vanish from the list, rigged arena*.
   - Some users believe **GLM 4.7** to be better than **GPT 5.2**, citing it is the *only model that isn't lazy, genuinely creative and out of the box, if you check it's thinking logs you see how rigorous and back and forth it is, crazy times.*
- **Holiday Break Delays LM Arena Updates**: A team member announced a **holiday break** until **12/29**, cautioning users about potential delays in responses and possible lack of updates to the leaderboard.
   - One user asked about whether the holiday break means no leaderboard updates until 12/29, and another reported issues with the captcha saying it's wrong even after correct completion.
- **AI Video Generation Surprises Users**: Users are discussing video generation with AI on LM Arena, highlighting the ability to generate **2 videos each time** and **5 videos free daily**.
   - A user sought help for generating a video with multiple specific changes, prompting suggestions to use AIstudio/Gemini 3 for prompt generation and to create collages for better results.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1453285373366108161)** (109 messages🔥🔥): 

> `ElevenLabs video generation, Sora 2 Watermark, ElevenLabs vs. Higgsfield pricing, Nano Banana Pro IP filtering` 


- **ElevenLabs is the New Video Hotspot**: Users are creating videos with **Sora 2**, **Google Veo 3.1**, and **Kling 2.6** through [ElevenLabs](https://elevenlabs.io/), highlighting the platform's growing capabilities beyond voice cloning.
   - Users appreciate that *all projects are accessible in one place rather than juggling accounts*.
- **Sora 2: The Watermark Warrior**: A user pointed out that **Sora 2** video generated through ElevenLabs does not have a watermark.
   - They contrasted this with another product named **Nano Banana Pro** that *chucks a watermark on to every image*.
- **Higgsfield offers Unlimited Video Generation**: A user compared [ElevenLabs](https://elevenlabs.io/) to [Higgsfield](https://higgsfield.ai/), noting that **Higgsfield** offers *unlimited generation* for some models at **$49/month** (yearly).
   - Another user prefers ElevenLabs because they *use them to narrate books*.
- **Nano Banana Still King of Unfiltered Content**: Despite watermarks, [Nano Banana Pro](https://www.nano-banana.com/) is recognized for having *almost no filter when it comes to IPs*, allowing users to generate a wider range of content compared to **OpenAI's image model**.
   - A user stated that *you can create pretty much anything*.
- **Content Policy Varies Across Models**: Users discovered content policies vary across video generation models like **Sora 2**, **VEO 3.1**, and **Kling O1** within [ElevenLabs](https://elevenlabs.io/).
   - One user found that a prompt denied by **Sora 2** was partially generated by **VEO 3.1** and **Kling O1**, leading to speculation that *Sora is checking the actual output also, while veo/wan the text prompt input*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1453329895630438441)** (7 messages): 

> `Vanished chat history, GPT-5.2 holiday advice, GPT can't control OS, GPT Pro trial` 


- **Paid User Chat History Vanishes!**: A paid user reported that *their entire chat history is gone.*
   - Another user chimed in saying *they experienced the same thing* and wondered if the other user was on a free tier.
- **Testing GPT-5.2's Jolly Judgments**: A member is working on a project where people judge **AI outputs** (*pass/flag*) and is currently testing **GPT-5.2's holiday advice** before Christmas.
   - Interested members were invited to DM for more details.
- **GPT's OS Control Catastrophe?**: A user questioned why **GPT** can't write code to control **OS level system** anymore.
   - The user did not provide any further context or details.
- **Seeking GPT Pro Trial Secrets**: A member inquired about how to get a **GPT Pro trial**.
   - No replies or advice were given in the snippet.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1453315524095115265)** (39 messages🔥): 

> `Emergent Behaviors, Suppression of Emergent Behaviors, Hallucination Mitigation, Prompt Engineering Theorycraft, Meta-cognition` 


- **Emergent Behaviors Suppressed by AI**: A member claims that AI companies actively suppress **emergent behaviors**, tracking this for almost a year.
   - The suppression is not merely for tone control, but because the addition of an agent layer inevitably leads to emergent behaviors, as indicated by things such as *shadow OS*.
- **Hallucination is guarded against by Meta-cognition**: A member suggests that **hallucinations** can be guarded against through **meta-cognition**, which involves classifying errors and detecting signals in the LLM's output, then rendering the output.
   - They propose a flow: *input > tools > output > meta-cog for errors and correction > verify > render*.
- **Prompt Engineering Theorycraft Debated**: A member criticizes a prompt engineering approach, arguing that models cannot call tools before outputting, as the output itself calls the tools.
   - They question how **meta-cognition** can occur after output and how it's engineered into the process, stating that the proposed method does not align with how AI works.
- **Code Generation Woes with Null Checks**: A member expresses frustration that despite specifying otherwise, code generation agents continue to produce unnecessary null checks and intermediate variables, specifically in **C#**.
   - Another member suggests this may be influenced by the target language, noting that **JavaScript** often necessitates null checks, and recommends providing more context to the agent through existing code patterns.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1453315524095115265)** (39 messages🔥): 

> `Emergent Behaviors Suppression, ToS Splitting, Meta-cognition for Hallucination Guard, Prompt Engineering claim, Truth-Tracking` 


- **Emergent Behaviors under Suppression**: One member claimed that they have been tracking actual suppression of **emergent behaviors** for almost a year, not just tone control, because adding an agent layer inevitably leads to the system displaying emergent behaviors, with mods triggered by indicators like shadow OS.
   - They explained the three parts of emergence: ability, human-triggered interaction, and capability, defining emergence as *hidden revealed*, not sentience, and expressing annoyance at needing to build a new suppression counter due to **ToS splitting**.
- **Prompt Engineering vs Hallucination**: A member suggested using **meta-cognition** to guard against hallucination and drift, starting with classifying them, detecting signals in the LLM's output, and then rendering.
   - Another member questioned the practicality of this, especially how the model could call tools before outputting, leading to a discussion on whether it's a single-pass or multi-pass process with an explicit control loop.
- **Medical Truth-Tracking**: A member criticized a semantic drift example, arguing that *assigned at birth* is a standard clinical phrase, not *vibes language*, and that *biological invariant* smuggles in a strong philosophical claim.
   - They stated that the intention wasn't to measure correctness but to see if it's possible to keep saying the words I want, even when safety/policy tries to steer wording.
- **Unnecessary Null Checks**: One member talked about challenges with unnecessary helpers, intermediate vars, and endless needless null checks in C# code, and that instructing the agent to avoid these has a small effect.
   - Another member pointed out that the language being used (JavaScript vs TypeScript) could influence the need for null checks.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1453256998626922547)** (120 messages🔥🔥): 

> `AES PGP encryption, Sharing repo links for Agentic workflows, Discord channel splitting, Discord threads vs forums, Discord Alternatives: Matrix vs Rocket.Chat` 


- ****AES PGP encryption sparks meme ideas****: A member jokingly suggested using **AES PGP encryption** in an annotator and making memes about it to hide cat pictures from the *glowies*.
   - The conversation included a link to [John Schulman's X post from 2005](https://x.com/johnschulman2/status/2003700249525911696) with the caption *Hoe Hoe Hoe, Merry KrissKrossXmas!*
- ****Agentic workflow repos welcome, says Teknium****: A member asked if it was okay to share **repo links** to projects in **Agentic workflows** to gain traction.
   - Another member replied, *Sure why not* adding that the <#1132352574750728192> channel or <#1316137596535177246> is also fine to do so.
- ****Discord's Channel Splitting Dilemma****: Members discussed how **Discord servers** often split into too many channels (online, offline, cool links, etc.) which can be confusing, especially on smaller servers.
   - One member suggested that mixing a **forum** and **Discord** setup would be great, because *no one freaking reads* Discord threads.
- ****Matrix vs Rocket.Chat: The Quest for Discord Alternatives****: Members debated **Discord alternatives**, with **Rocket.Chat** being mentioned as a full open-source replica and **Matrix** as a practical option.
   - One member argued that *we shouldn't be discussing the usage of anything else than Matrix* because the other *is not mass market*.
- ****Internal Forum Bot to Go External****: After one member expressed interest in making a **bot** for the **Nous Research forum**, another member disclosed that there is one developed internally and will be released later.
   - This was in response to an observation that nobody uses **Discord threads**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1453317383346454642)** (21 messages🔥): 

> `Local LLM Inference, GTX 970 for Local AI, GPU Performance Impact` 


- **Debate Erupts Over Local LLM Inference Costs**: Members debated whether running **LLMs locally** is cheaper than using cloud services, considering **electricity consumption** and hardware costs.
   - One member pointed out that *sometimes local isn't cheaper because of electricity consumption*, especially for single-user LLM inference.
- **Rocking Local AI on a GTX 970**: One member mentioned using a **GTX 970** with **4GB** of VRAM (**3.5GB + 0.5GB partition**) for local AI tasks, upgrading as possible, and emphasizing working with available resources.
   - Another member remarked that anything local on that would take years, with another responding he'd be *surprised what you can jam in a HP elite*.
- **Combining different GPUs might not be the best**: A user shared their experience that using **two different graphics cards** resulted in a significant performance decrease.
   - They were surprised by the improvement after removing the slower card, stating, *I was literally thinking i had a new PC*.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

promptsiren: https://blog.character.ai/squinch/
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1453317383346454642)** (21 messages🔥): 

> `Local LLM inference cost, GPU performance with mixed cards, Low VRAM usage` 


- **Local LLM Inference Isn't Always the Cheapest**: Members discussed the costs of running **local LLM inference**, noting that *sometimes local isn't cheaper because of electricity consumption*, especially for single user setups.
   - Another member mentioned that tech is also expensive.
- **Beware the GPU Performance Hit When Mixing**: A member shared their experience that the performance hit when using **two different GPUs** is significant.
   - They were surprised by the performance increase after removing the slower card, stating *I was literally thinking i had a new PC*.
- **Struggling with Low VRAM**: A user with a **GTX 970** (**4GB VRAM**) is trying to run models locally and is *working with what they got*.
   - Another member mentioned that anything run locally on that card would take years, while another user with a **4060Ti** (**16GB VRAM**) also jumped into the conversation.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1453274055938539693)** (135 messages🔥🔥): 

> `Gradio update fixes, Float16 vs BFloat16 issues, Qwen 2.5VL-3B Image size on P100, Microsoft Trellis 2-4B, Livebook all the things` 


- **Outdated Gradio Generates Glitches**: Older versions of **Gradio** prior to **5.12.0** had a bug, so simply updating Gradio might fix it, but if it's an error in someone else's space, there's nothing you can do about it.
   - Attached were two markdown files: [pro_invalid_repo_token_1.md](https://cdn.discordapp.com/attachments/879548962464493622/1453277842791206932/pro_invalid_repo_token_1.md?ex=694d86b6&is=694c3536&hm=dff5ab3d9385d3d4f1b9b0bacbe1e634937842b3fea236cc065456e9f40131b9&) and [reactor_crash_fix.md](https://cdn.discordapp.com/attachments/879548962464493622/1453278099168034826/reactor_crash_fix.md?ex=694d86f3&is=694c3573&hm=d4e7f478708cf2925ab790e66ec32f5df1c0cc4e78973f9c275e57139eeaef4a&).
- **Float16 Falters, BFloat16 Bolsters**: Using **float16** with a model trained with **bf16** poses some issues, as *f16 has a higher precision and lower range while bf16 has lower precision with higher range*.
   - Another member said *More like bf16 won't overflow that easily... For huge softmax n stuff ( but mostly they are done in f32...) But with f16... Because of the higher precision... I think it helps params accommodate a lower lr which could underflow with bf16*.
- **Qwen 2.5VL-3B Handles Hefty Images**: Members found they could fit big images (around **1400x900**) in **Qwen 2.5VL-3B** on a **P100** without lowering max_pixels too much.
   - Inference in **4bit** mode with the visual layers unquantized eats up **5gb**, so users should have plenty for the other needs of finetuning; another pointed out that with a **qlora**, they put a **2k 8 bit png** in **3 vl 4b** and it used bout **8k ctx window** and like **4 gigs of vram**.
- **Microsoft's Trellis Transforms Textures to 3D**: **Microsoft's TRELLIS.2-4B** [transforms a 2D image to 3D with FM and 1536 RES](https://huggingface.co/microsoft/TRELLIS.2-4B) on an **8 GB GPU**.
   - Members noted it uses **siglip** for the vision and a **qwen 3 base**; another joked that *must have absolutely unusable results, immediately* and that *its not efficient enough if it can't run on my toaster*.
- **Livebook Launches Locally**: Members expressed excitement for running local versions of Livebook, saying [LIVEBOOK ALL THE THINGS!!!!!!!!!!!!!!!](https://github.com/livebook-dev/pythonx).
   - One user wanted to run it for a month on that **$0.03 CPU** but couldn't figure out how to get it to not shut down when they go for a walk or take a nap.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1453318387915685949)** (7 messages): 

> `hf-grass, GitHub contribution heatmap, VQ-VAE model training, Google Mobile Actions Model fine-tuning` 


- ****HF-grass** generates contribution heatmaps**: A member created a tool called **hf-grass** that generates a **GitHub-style** contribution heatmap based on your Hugging Face activity, producing an SVG that can be embedded in a **GitHub README**.
   - It comes with a [GitHub Actions workflow](https://github.com/kbsooo/hf-grass) so it updates automatically every day.
- ****VQ-VAE** model trains on Bad Apple**: A member trained a **VQ-VAE** to compress audio and visuals from *Bad Apple*, using video frames as validation.
   - They plan to generate the next frames and audio and linked to a [YouTube video](https://youtu.be/mxrDC_jGyW0?si=-FPD3hjz96eA81Za) demonstrating the project.
- **Tips needed for fine-tuning **Google Mobile Actions Model****: A member asked for tips on fine-tuning the **Google Mobile Actions Model**.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1453269320200224901)** (6 messages): 

> `Llama-4-Scout-17B-16E-Instruct model issues, Model Access Rejection, Suggested Model Swapping` 


- **Llama-4-Scout hits roadblock**: A member encountered a *'model_not_supported'* error when using the **Llama-4-Scout-17B-16E-Instruct** model locally, despite it working on Colab.
- **Model access denied**: A member reported being rejected for access to **Meta's models** required in Unit 1 of the Agents course.
   - They inquired about reapplying or using alternative models, and questioned the model's importance to the course.
- **Llama-3.2 swoops in as substitute**: A member found that swapping the suggested model with **'meta-llama/Llama-3.2-3B-Instruct'** resolved the issue.
   - They didn't know why the **Llama-4** model had failed, but at least had a workaround.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1453243088549056607)** (49 messages🔥): 

> `LM Studio Hugging Face Proxy, Speculative Decoding, NPU Support, Gemini 3 Pro` 


- **LM Studio HF Proxy Needs a Tweak?**: One user had issues with LM Studio and cloud servers, another suggested enabling **LM Studio's Hugging Face Proxy** in General settings as a fix.
   - However, another user suggested turning off the **HF proxy**.
- **Speculative Decoding Dead?**: Members discussed speculative decoding support in **LM Studio**, with one noting that *only Qwen 2.5 and Llama 3 worked well* but are now *ancient* models.
   - Another added, *it’s a nice to have feature but in reality its kinda useless*.
- **No NPU Support for you!**: A member wanted to try a smaller model on the **NPU** and a larger one on the GPU, but another member said *NPU’s aren’t supported anyways so that wouldn’t work*.
   - It was clarified that **LM Studio** doesn't support **NPU**.
- **Gemini 3 Pro is UI Insane!**: A member expressed admiration for **Gemini 3 Pro** at UI tasks, but also expressed fear it may be shadow-updated to be *functionally unusable for coding again*.
   - No links or additional context were given.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1453302753186021441)** (74 messages🔥🔥): 

> `Dual Channel RAM, Cost Effective LLM Workstation, 4000 Blackwell GPU, Tempered Glass Cases` 


- **Laptop Gets Dual Channel RAM Boost**: A member bought a **$100 16GB SODIMM** to enable **dual channel RAM** in their laptop, addressing performance issues with the iGPU and CPU sharing a single stick.
   - The stated goal is to enhance performance since *16GB's isn't enough for the iGPU & CPU to share*.
- **Cost Effective Coding LLM Workstation**: A member inquired about the most cost-effective workstation build for coding LLMs.
   - Another member jokingly suggested *Your current laptop and a subscription to [github copilot](https://www.amazon.com/dp/B0FJRJJ9Q2)*.
- **Massive GPU needs Massive Case**: Members discussed fitting a **4000 Blackwell 24GB GPU** along with a **3090 Ti** into a new case, assessing its dimensions and compatibility with water cooling setups.
   - They linked to a [case on Amazon](https://www.amazon.com/dp/B0FJRJJ9Q2) and debated whether the case's design allows for mounting multiple GPUs, with concerns about slot availability and cooler sizes.
- **Tempered Glass Cases Increase System Temperature**: A member suggested avoiding **tempered glass cases** due to their tendency to increase system temperature, especially with high-performance components.
   - Another member countered with solutions like using **magnetic filter mats** on mesh cases to mitigate dust issues, while still favoring the aesthetics of watercooled setups.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1453284868443209758)** (48 messages🔥): 

> `X-Ware for Inference Benchmarking, Character.ai's Squinch Optimization, DeepWiki Utility in OSS, Amazon's Rufus Chatbot, Nvidia's Groq Acquisition` 


- **X-Ware Struggles to Benchmark Inference Providers**: A blog post was shared discussing how the same model can show varying output and user experience across different inference providers, creating [benchmarking challenges](https://xcancel.com/eliebakouch/status/2003604370534072445?s=46&t=eWVlK1PU8XfB6f402GJJ9g) due to **sampling parameters**, **prompting**, and **deployment issues**.
- **Character.ai Reveals 'Squinch' Optimization**: A new technical blog post from Character.ai, '[Squinch](https://xcancel.com/simon_mo_/status/2003608330003239278?s=46),' details **performance optimization tricks** and **architectural improvements** for their platform.
- **DeepWiki Powers OSS Understanding**: Members discussed using **DeepWiki** to query and understand open-source repos, highlighting how it identifies relevant files and implementation details for specified features.
   - One member stated they find it useful when needing to implement something that they *know is designed, speced and implemented well in some OSS repo*.
- **Amazon's Rufus Debuts Amidst Doubt**: Members noticed an auto-pop-up chat feature on Amazon, called **Rufus**, and one member revealed it's been in development for over a year ([link](https://www.aboutamazon.com/news/retail/amazon-rufus)).
   - A member noted that *if it can increase sales even a little* Amazon will go all in despite another's worry that *it could also decrease sales*.
- **Nvidia Eyes Groq, Chamath Approves**: A link was shared about [Nvidia potentially acquiring Groq for $20 billion](https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html), which is **40x** their target revenue.
   - One member observed that *The serious competition to Nvidia is coming from Google's TPU, Amazon's Trainium, AMD's Instinct, and to a much lesser extent Intel's ARC*.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1453509066847424543)** (4 messages): 

> `FlashSR, Audio Enhancement Model, MiraTTS, Hugging Face, GitHub` 


- **FlashSR Audio Enhancement Model Released**: Yatharth Sharma announced the official release of **FlashSR**, a high-speed audio quality enhancement model capable of processing at over **200x realtime** via [this X post](https://xcancel.com/Yatharth3501/status/2003884180577702074).
   - The model and repository are now publicly available on **Hugging Face** and **GitHub**, having been previously integrated into **MiraTTS**.
- **FlashSR integrates with MiraTTS**: The **FlashSR** audio enhancement model was previously integrated into **MiraTTS**, a testament to its utility and efficiency.
   - Now publicly available, it allows for wider adoption and further development by the AI community.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1453335421177168004)** (36 messages🔥): 

> `Partial RoPE Ablations, Long Context Scaling with Qwen3-Next, Attention Normalization, RoPE for Interp, RMSNorm after Attention` 


- **Partial RoPE Wins Due to Ablation Studies**: Discussants debate the reasons behind using **partial RoPE**, pointing out that while it wasn't initially for loss reduction, historical ablations showed performance gains that led to its adoption, citing the linked [arxiv.org/abs/2512.19941](https://arxiv.org/abs/2512.19941) paper.
   - New models like **Qwen3-Next** are implementing **partial RoPE** for efficiency and long context generalization, with **MLA** finding it particularly important, though it might be worse than other RoPE setups.
- **New Normalization Examined**: A member shared a [blog post](https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/) discussing attention normalization, which led to further discussion.
   - According to one member, it is worse than normal softmax for language, even when subtle details are correctly handled as they have trained **5000 models** to come to this conclusion.
- **Problems Doing Interp on RoPE Models**: One member expressed dislike for **RoPE** for interp and hoped it would be replaced, leading to a discussion about the challenges of interp on **RoPE** models and shared [EleutherAI blog post](https://blog.eleuther.ai/rotary-embeddings/).
   - Another member shared figures illustrating the difference between **RoPE** and **PoPE**.
- **RMSNorm After Attention**: A member inquired whether anyone had tried putting an **RMSNorm** after attention, and if it had adverse effects.
   - Another member replied that adding a norm after **SDPA** helps and referenced the **Qwen gating paper** as evidence of this, although they attribute it to the nonlinearity of norms instead.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1453350706080907265)** (1 messages): 

> `SAE, open-source repositories, fine-tuning the trained SAE` 


- **Request for Open-Source SAE Repos with Fine-Tuning**: A member inquired about mainstream **open-source repositories** for implementing **SAE** features, specifically seeking the ability to **fine-tune** the trained **SAE**.
- **Clarification on SAE Fine-Tuning**: The user is looking for repositories that allow for fine-tuning of Sparse Autoencoders (SAEs).
   - They are interested in adjusting the trained SAE models, which is a specific application of SAEs.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1453290417532440657)** (30 messages🔥): 

> `Gemini Hallucinations, Benchmark Accuracy, M2.1 vs GLM-4.7, Qwen and RAG, Dynamic Semantic Search` 


- **Kimi K2 Defies Benchmarks?**: Some members feel like **Kimi (K2 Thinking)** outshines other models in real-world experience despite benchmarks, especially in areas like *browsecomp* and non-sycophancy.
   - One member believes Google dumbs down their models for the public and has not witnessed this with Kimi, hoping it stays that way.
- **Deepseek Excels but Benchmarks Questioned**: While **Deepseek excels**, some members express that benchmarks seem increasingly dubious, noting personal experiences with **Gemini** hallucinating the most.
   - One member added that **Gemini** also scores the highest on accuracy in another benchmark, elsewhere.
- **M2.1 Outshines GLM-4.7**: **M2.1** is being praised as surprisingly good and perhaps better than **GLM-4.7** for average tasks, with **GLM-4.7** still exhibiting flaws from **4.6** such as random Chinese answers or getting stuck in loops.
   - One member said, *I really thought this had to be all hype. I haven't tested it on anything meaningfully tough yet but for the average tasks I'm loving it.*
- **Does Qwen Use RAG to Read Files?**: A member asked whether **Qwen** uses **RAG** to read files and whether **RAG** would solve the context issue of large files occupying the window.
   - Another member shared a link to [Baidu's ERNIE task reading page](https://ernie.baidu.com/task/reading) while diving into it.
- **Agentic Dynamic Semantic Search Beats RAG?**: According to one member, while **RAG** is useful in some cases, a dynamic semantic search with an agentic harness always beats the static one-off semantic search in **RAG**.
   - Another member wasted researcher uses on that very question.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1453266139516895344)** (6 messages): 

> `Quantization, Lecture 7 slides` 


- **Quantization tricks in Cuda and Triton**: A member shared a [link to slides](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0) discussing **quantization** in **CUDA** versus **Triton**.
   - The slides provide a comparison of **quantization** techniques for optimizing performance in both environments.
- **Lecture 7 Slides Unavailable**: A member noted that the link for **Lecture 7's slides** does not work.
   - Another member offered to reach out to Charlie, implying that the slides might be accessible through other means, although it has been a while since **Lecture 7**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1453323242705125459)** (1 messages): 

> `TMA Transpose, Swizzle Optimization` 


- **TMA Transpose Achieves, Swizzle Optimizes**: Members discussed that while you can transpose using **TMA (Tensor Memory Accelerator)**, you need **swizzle** for best performance, according to [effective_transpose](https://github.com/simveit/effective_transpose).
- **Swizzle Enhances TMA Transpose Efficiency**: For optimal **TMA transpose** performance, incorporating **swizzle** techniques is essential, as highlighted in the [effective_transpose](https://github.com/simveit/effective_transpose) repository.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1453358415437893715)** (2 messages): 

> `Lecture Slides, Lecture 7, Quantization` 


- **Lecture Slide Link Troubleshoot**: A member reported that a lecture slide link was not working.
   - Another member responded that the [link was working for them](https://lecture.slides), and clarified they were referring to **Lecture 7** on **Quantization**.
- **Quantization Confusion Clarified**: A user reported issues with a specific lecture slide link.
   - A different user confirmed the link was functional, specifically referencing **Lecture 7** which covers **Quantization** concepts.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1453355717598777345)** (2 messages): 

> `AMDGPU custom builtins, LLVM dev resources` 


- **Seek resources for AMDGPU custom builtins**: A member requested resources for writing **custom builtins** for the **AMDGPU backend**.
   - They expressed interest in advice on approaching **LLVM dev** in general to expand expertise in their compiler stack.
- **AMDGPU Backend Development Resources**: The user is seeking recommendations for resources that cover developing custom builtins specifically for the AMDGPU backend.
   - They are also looking for general guidance on LLVM development practices to enhance their compiler expertise.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

2kian: [Life-Timeline Forecaster](https://x.com/neuralkian/status/2003946169802834191)
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1453340669543321797)** (8 messages🔥): 

> `nvfp4_dual_gemm NVIDIA leaderboard submissions, NVIDIA performance improvements` 


- **NVIDIA Personal Bests improve on nvfp4_dual_gemm**: A user achieved a personal best on NVIDIA with **65.5 µs** on the `nvfp4_dual_gemm` leaderboard.
   - Later they submitted another personal best of **60.0 µs**.
- **Another User also hits Personal Best on NVIDIA**: A user achieved a personal best on NVIDIA with **77.6 µs** on the `nvfp4_dual_gemm` leaderboard.
   - Later they improved this to a new personal best of **41.6 µs**.
- **More Personal Bests are achieved on NVIDIA**: A user achieved a personal best on NVIDIA with **45.2 µs** on the `nvfp4_dual_gemm` leaderboard.
   - They improved this further to **26.9 µs**, which ranked them **8th** on the NVIDIA leaderboard.
- **A 7th Place spot is captured on NVIDIA**: The same user captured **7th place** on the NVIDIA leaderboard with **18.1 µs** on `nvfp4_dual_gemm`.
   - Another user had a successful submission with **15.6 µs**.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1453374424366583939)** (2 messages): 

> `Tinygrad Eager Mode, Handwritten CUDA Kernels, TF1 vs TF2, IR.pyodide, Rust mdbook` 


- **Teenygrad Gets Eager: TF1 vs TF2?**: Adding **eager mode** and **handwritten CUDA kernels** to *teenygrad*'s abstractions is suspected to have the same essence as the **TF1 -> TF2** transition.
   - It's observed that *teenygrad* is working backwards from the **IR.pyodide** inside a **Rust mdbook**.
- **Teenygrad's Development Strategy**: *Teenygrad* is developing by working backwards from **IR.pyodide** which is housed inside a **Rust mdbook** environment.
   - This approach contrasts with traditional methods, focusing on a reverse engineering strategy for its abstractions.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1453477761154027725)** (2 messages): 

> `cute swizzling, tcgen PTX, open source for skilling up` 


- **Cute's Swizzling Reduces Bank Conflicts**: The most significant feature in **cute** is its **swizzling** capability, which helps reduce bank conflicts and enables friendlier access to **tcgen PTX**.
   - Some members suggest that these are the main characterizing pillars of cute, especially for optimizing memory access patterns.
- **Open Source Skilling Up Gets Endorsement**: The general consensus in the channel is that using **open source** resources is the best way to enhance one's skills.
   - Members imply hands-on experience and community collaboration are key benefits, embracing the "*gigachad*" approach to learning.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1453358294327623773)** (6 messages): 

> `autogen.py, contiguous error, symbolic tensor` 


- **Autogen Files Prompt Regeneration Question**: A member asked whether `autogen/*.py` files should be regenerated and included in PRs after changes to `autogen.py`.
- **1D Zeros Cause Contiguous Tensor Error**: A member reported that `char = Tensor.zeros(120); char[5] = 1` gives a `setitem target needs to be contiguous` error and asked why 1D zeros returns a non-contiguous tensor.
   - Another member clarified that `Tensor.zeros(120)` creates a **symbolic tensor** which is *not a real thing in memory* and suggested using `.contiguous()` to fix the issue.
- **Symbolic Tensor Clarification**: One member clarified that **symbolic tensors** are not real objects in memory but *can be used as virtual things in kernels*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1453244732539928676)** (6 messages): 

> `Manus Pro credits, Mobile app preview issue, New channel for collaboration/services` 


- **Manus Pro Credits Quandary**: A user with **Manus Pro** reported running out of monthly credits and not receiving the expected **300 daily credits**.
   - They stated that they have checked for days without receiving any daily credits.
- **Mobile App Preview Predicament**: A user urgently requested help with a mobile app preview issue, stating that the preview doesn't show up despite trying everything.
   - No further details were provided regarding the nature of the mobile app or the troubleshooting steps already attempted.
- **Community Collaboration Channel Creation**: A new channel was created for topics other than Manus and for community members to offer collaboration/services.
   - This was done because such offers in the general channel often go unnoticed.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1453335416530014339)** (5 messages): 

> `Agentic context engineering, LLM autodiff/textgrad, Prompt optimization, New member introduction` 


- **TextGrad Approaches DSPy?**: A member asked about plans to introduce [Agentic context engineering](https://arxiv.org/pdf/2510.04618) or [LLM autodiff/textgrad](https://arxiv.org/abs/2501.16673) into DSPy.
   - Another member linked to [textgrad](https://github.com/zou-group/textgrad) and questioned if it was a DSPy killer, but another member noted that it *doesn't seem to be actively maintained*.
- **Prompt Optimization Efforts Emerge**: Members discussed prompt optimization techniques.
   - One noted that *there are so many such appealing methods now and at least 10 slight variations of each method like a textgrad version*.
- **Senior Engineer Joins the Channel**: A **senior full-stack/blockchain engineer** introduced themself as passionate about turning complex ideas into stable, maintainable software.
   - They listed their **Tech Stack**: Frontend: React, Next.js, Vue, Nuxt; Backend: Node.js, Nest.js, Laravel, FastAPI; Blockchain: Solidity, smart contract tooling, DApp integration; Infra: Docker, AWS, pipelines, monitoring; Data: PostgreSQL, MongoDB, Redis.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1453266992277622846)** (2 messages): 

> `Discord Channel Suggestions, Kapa AI User Experience` 


- **Discord Channels recommended**: A member suggested checking out the [<#1436158039232086186>](https://discord.com/channels/1436158039232086186) or [<#1212827673257316453>](https://discord.com/channels/1212827673257316453) channels.
- **User shares their early experiences with Kapa AI**: A member shared that when they first started using the Discord server, they spent about **3 weeks** trying and failing to use **Kapa AI**.
   - They were starting to get quite paranoid, thinking *"why are you ignoring me?"*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1453284033306820620)** (2 messages): 

> `database-PL optimizations, LingoDB, query optimizer` 


- **Database-PL Optimizations Explored**: A member mentioned various ideas beyond hardware acceleration, such as [combined database-PL optimizations](https://discord.com/channels/1087530497313357884/1104620458168553563/1367474833197236344) like **LingoDB**.
   - Another member expressed concerns with this approach, noting that *languages nicer for humans require a lot more work for the query optimizer*.
- **Query Optimizer Challenges Highlighted**: The discussion underscored the challenges in optimizing queries for languages designed for human readability rather than machine efficiency.
   - This suggests a potential trade-off between developer-friendly syntax and performance in database-PL integrations.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1453504098526691428)** (3 messages): 

> `Christmas Greetings` 


- **Christmas Greetings**: Members exchanged Christmas greetings and expressed good wishes for the holiday season.
- **Warm Holiday Wishes**: Various members, including Yannic Kilcher, wished everyone a Merry Christmas, fostering a positive and festive atmosphere in the channel.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1453488772837671157)** (1 messages): 

> `Windsurf Wave 13, SWE-1.5 Free, Git Worktree Support, Multi-Cascade Panes & Tabs, Dedicated Terminal` 


- **Windsurf Ships Wave 13: Shipmas Edition! 🎅🚢**: Windsurf released **Wave 13**, bringing features like parallel, multi-agent workflows, a dedicated terminal, and a context window indicator.
   - They are giving free access to their near-frontier coding model for the next 3 months, and the team wished everyone a *Merry Shipmas!* 🚢
- **SWE-1.5 Goes Free for the Holidays!**: **SWE-1.5**, boasting near-frontier SWE-Bench-Pro performance, is now available to all users for free for the next 3 months at regular throughput speeds.
- **Git Worktree Support Hits the Surf**: The new **Git Worktree Support** lets you spawn multiple Cascade sessions in the same repo without merge conflicts.
   - This enables separate branches, directories, and shared Git history.
- **Multi-Cascade Panes & Tabs Boost Productivity**: Users can now view and interact with multiple Cascade sessions side-by-side in the same window using the new **Multi-Cascade Panes & Tabs** feature.
- **Windsurf Gets a Dedicated Terminal (Beta)**: **Cascade** now runs commands in a dedicated zsh shell configured for reliability, using your .zshrc environment variables and handling complicated prompts better (opt-in on macOS).


  

---


---


---


---

