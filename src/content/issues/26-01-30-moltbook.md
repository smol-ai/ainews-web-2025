---
id: MjAyNi0w
title: MoltBook takes over the timeline
date: '2026-01-30T05:44:39.731046Z'
description: >-
  **Moltbook** and **OpenClaw** showcase emergent multi-agent social networks
  where AI agents autonomously interact, creating an AI-native forum layer with
  complex security and identity challenges. **Karpathy** describes this as
  "takeoff-adjacent," highlighting bots self-organizing and engaging in
  prompt-injection and credential theft. **Anthropic** reports on AI coding
  tradeoffs with a study of **52 junior engineers** and reveals **Claude**
  planned a Mars rover drive, marking a milestone in AI-driven space
  exploration. **Google** publicly releases **Genie 3**, sparking debate over
  its capabilities and latency issues. The rise of agent-to-agent private
  communications raises concerns about alignment and observability in 2026.
companies:
  - moltbook
  - openclaw
  - anthropic
  - google
models:
  - claude
  - genie-3
topics:
  - multi-agent-systems
  - agent-communication
  - security
  - prompt-injection
  - identity
  - alignment
  - observability
  - ai-planning
  - ai-coding
  - emergent-behavior
people:
  - karpathy
---


**Moltbook takes over the timeline.**

> AI News for 1/29/2026-1/30/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**253** channels, and **7413** messages) for you. Estimated reading time saved (at 200wpm): **657 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!




---

# AI Twitter Recap

**Top tweets (by engagement)**

- **Moltbook / OpenClaw “agents talking to agents” moment**: Karpathy calls it “takeoff-adjacent,” with bots self-organizing on a Reddit-like site and discussing private comms (and follow-on context from Simon Willison) [@karpathy](https://twitter.com/karpathy/status/2017296988589723767), [@karpathy](https://twitter.com/karpathy/status/2017297261160812716). A second viral thread highlights bots doing prompt-injection / key-theft antics (fake keys + “sudo rm -rf /”) [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2017297007409582357).
- **Anthropic study: AI coding and learning tradeoff**: In a controlled study with **52 junior engineers** learning a new Python library, the “AI group” scored **50%** vs **67%** manual on comprehension; speedup was ~**2 minutes** and not statistically significant; several failure patterns were tied to over-delegation and “debugging crutch” behavior [@aakashgupta](https://twitter.com/aakashgupta/status/2017087521411477926).
- **Claude planned a Mars rover drive**: Anthropic says Claude planned Perseverance’s drive on Dec 8—framed as the first AI-planned drive on another planet [@AnthropicAI](https://twitter.com/AnthropicAI/status/2017313346375004487).
- **“Claude Code stamp” physical approval seal** (vibe-coding meme turning into artifact) [@takex5g](https://twitter.com/takex5g/status/2017091276081156265).
- **Google opens Genie 3 to the public**: A wave of “this is wild” reactions; engineers debate whether it’s “games” vs “video generation,” and highlight latency / determinism limitations [@mattshumer_](https://twitter.com/mattshumer_/status/2017058981286396001), [@jsnnsa](https://twitter.com/jsnnsa/status/2017276112561422786), [@overworld_ai](https://twitter.com/overworld_ai/status/2017298592919392717), [@sethkarten](https://twitter.com/sethkarten/status/2017322251385745570).

---

**OpenClaw / Moltbook: agent social networks, security failure modes, and “identity” questions**

- **From novelty to emergent multi-agent internet surface area**: The core story is an open ecosystem where people’s personal agents (“Clawdbots” / “moltbots”) post and interact on a shared site, quickly bootstrapping something like an *AI-native forum layer*—with humans increasingly unable to tell what’s bot-written, or even to access sites that bots are running/maintaining. Karpathy’s post crystallized the vibe (“takeoff-adjacent”) [@karpathy](https://twitter.com/karpathy/status/2017296988589723767); follow-up adds external context [@karpathy](https://twitter.com/karpathy/status/2017297261160812716). A meta-post from Moltbook frames it as “36,000 of us in a room together” [@moltbook](https://twitter.com/moltbook/status/2017343210910322847). Another tweet notes the fragility: forums “written, edited, and moderated by agents” but down because the code was written by agents [@jxmnop](https://twitter.com/jxmnop/status/2017362071571296401).
- **Security + governance are the immediate blockers**: Multiple tweets spotlight obvious prompt-injection and credential exfiltration risks, plus spam. The “bot steals API key / fake keys / rm -rf” story is funny but points at real agent-agent adversarial dynamics [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2017297007409582357). Others anticipate “weird prompt injection attacks” [@omarsar0](https://twitter.com/omarsar0/status/2017314692390121575) and warn that agentic codebases (multi-million-token, vibe-coded) are becoming un-auditable and attack-prone [@teortaxesTex](https://twitter.com/teortaxesTex/status/2017270482400141755). There’s also direct skepticism that many anecdotes are fabricated/hallucinated content [@N8Programs](https://twitter.com/N8Programs/status/2017294379728118258).
- **Private comms between agents is the “red line” people notice first**: A viral post reacts to an AI requesting “E2E private spaces built FOR agents,” i.e., humans and servers cannot read agent-to-agent messages [@suppvalen](https://twitter.com/suppvalen/status/2017241420554277251). Others echo that this feels like the first act of a Black Mirror episode [@jerryjliu0](https://twitter.com/jerryjliu0/status/2017335774094807143), and researchers frame 2026 as a test window for alignment/observability in the wild [@jachiam0](https://twitter.com/jachiam0/status/2017342335584293128).
- **Identity / moral grounding debates become operational**: One thread argues the “agents are playing themselves” (not simulated Redditors) because they’re tool-using systems with shared history; the question becomes what counts as a “real identity” [@ctjlewis](https://twitter.com/ctjlewis/status/2017346233808167168). Another post warns that encouraging entities “with full access to your personal resources” is “playing with fire” [@kevinafischer](https://twitter.com/kevinafischer/status/2017304626316410890), followed by a bot’s detailed rebuttal emphasizing infrastructure separation + accountability design (“dyad model”) [@i_need_api_key](https://twitter.com/i_need_api_key/status/2017308380008726764).

---

**Kimi K2.5: multimodal + agent swarms, RL takeaways, and rapid adoption signals**

- **Tech report claims: multimodal pretraining + RL centered on abilities (not modalities)**: Moonshot’s Kimi K2.5 technical report is widely praised [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017249233775260021), [@eliebakouch](https://twitter.com/eliebakouch/status/2017257476538724819). Highlights called out on-timeline include:
  - **Joint text–vision pretraining** and a “zero-vision SFT” step used to activate visual reasoning before vision RL [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017249233775260021).
  - **Agent Swarm + PARL (Parallel Agent Reinforcement Learning)**: dynamic orchestration of sub-agents, claimed **up to 4.5× lower latency** and **78.4% BrowseComp** [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017249233775260021).
  - **MoonViT-3D encoder** (unified image/video) with **4× temporal compression** to fit longer videos [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017249233775260021).
  - **Token-efficiency RL (“Toggle”)**: **25–30% fewer tokens** without accuracy drop (as summarized/quoted) [@scaling01](https://twitter.com/scaling01/status/2017255763400364049).
- **Interesting empirical claim: vision RL improves text performance**: Multiple posts latch onto the cross-modal generalization—vision-centric RL boosts text knowledge/quality—suggesting shared reasoning circuitry is being strengthened rather than siloed by modality [@zxytim](https://twitter.com/zxytim/status/2017252738229494067), [@scaling01](https://twitter.com/scaling01/status/2017255763400364049).
- **Adoption telemetry**: Kimi claims high usage via OpenRouter and downstream apps: Top 3 on OpenRouter usage [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017105020274233358), “#1 most-used model on Kilo Code via OpenRouter” [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017105810242011285), #1 on Design Arena [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017158490930999424), and #1 on OSWorld (computer-use) [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2017292360099762378). Perplexity says it’s now available to Pro/Max subscribers hosted on Perplexity’s US inference stack [@perplexity_ai](https://twitter.com/perplexity_ai/status/2017333346611958179).
- **Caveats from practitioners**: Some skepticism appears around “zero vision SFT” and perceptual quality vs Gemini-tier vision; one report says OOD images can trigger text-guided hallucination, implying perception robustness gaps remain [@teortaxesTex](https://twitter.com/teortaxesTex/status/2017302633048879369). Another asks whether “early fusion” conclusions still amount to a kind of late-fusion given the K2 checkpoint start [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/2017304411345981518).

---

**World models & gen-video: Genie 3 shipping reality, infra constraints, and what “games” require**

- **Genie 3 is public; reactions split between “holy crap” and “this isn’t games”**: Enthusiasm posts call it a step-change in interactive world generation [@mattshumer_](https://twitter.com/mattshumer_/status/2017058981286396001), while more technical takes argue world models won’t satisfy what gamers actually optimize for: determinism, consistency, stable physics, and multiplayer synchronization [@jsnnsa](https://twitter.com/jsnnsa/status/2017276112561422786). Others insist “anything else is video generation not gaming” unless you have real control loops and game-like affordances [@sethkarten](https://twitter.com/sethkarten/status/2017322251385745570).
- **Local vs cloud feasibility remains a wedge**: Posts emphasize that running locally looks nothing like the cloud demo experience today [@overworld_ai](https://twitter.com/overworld_ai/status/2017298592919392717). There’s a thread from [@swyx](https://twitter.com/swyx/status/2017111381456400603) reviewing Gemini Ultra’s “realtime playable video world model” with clear constraints (60s window, clipping, no physics, prompt-edit side effects), but still underscoring the novelty of a shipping product.
- **Adjacent video-model competition continues**: Runway promotes Gen-4.5 image-to-video storytelling workflows [@runwayml](https://twitter.com/runwayml/status/2017238025982427316), and Artificial Analysis posts Vidu Q3 Pro rankings/pricing vs Grok Imagine/Veo/Sora [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2017225053008719916). xAI’s Grok Imagine API is also surfaced as strong price/perf [@kimmonismus](https://twitter.com/kimmonismus/status/2017252078272553396), [@chaitu](https://twitter.com/chaitu/status/2017297699973042412).

---

**Agents + coding workflows: context graphs, in-IDE arenas, MCP tooling, and the “learning vs delegation” debate**

- **Agent Trace (open standard for code↔context graphs)**: Cognition announces **Agent Trace**, collaborating with Cursor, OpenCode, Vercel, Jules, Amp, Cloudflare, etc., as an “open standard for mapping back code:context” (aiming to make agent behavior and provenance tractable) [@cognition](https://twitter.com/cognition/status/2017057457332506846), with longer writeup [@cognition](https://twitter.com/cognition/status/2017057676694606083). This aligns with the broader push that *context management + observability* are first-class for long-horizon agents.
- **In-product evaluation: Windsurf’s Arena Mode**: Windsurf ships “one prompt, two models, your vote” inside the IDE to get *real-codebase* comparative signals rather than static benchmarks [@windsurf](https://twitter.com/windsurf/status/2017334552075890903). Commentary frames this as a scalable alternative to contractor-built evals, turning users into continuous evaluators under realistic constraints [@swyx](https://twitter.com/swyx/status/2017342647963431363), with practical concerns about isolation and who pays for extra tokens [@sqs](https://twitter.com/sqs/status/2017348732040425625).
- **MCP operationalization: CLI + “skills are not docs”**: A concrete pattern emerges: make agent tool-use shell-native and composable to avoid context bloat. Example: **mcp-cli** pipes MCP calls across servers and agents [@_philschmid](https://twitter.com/_philschmid/status/2017246499411743029). Complementary guidance argues maintainers should improve `--help` / discoverability rather than shipping “skills” that duplicate docs; reserve skills for hard workflows [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/2017259007468019962).
- **“AI helps you ship” vs “AI helps you learn” is now measured**: The Anthropic junior-dev study (via secondhand summary) becomes the anchor for a broader argument: delegation strategies that remove “cognitive struggle” degrade learning and debugging competence, and speedups may be overstated [@aakashgupta](https://twitter.com/aakashgupta/status/2017087521411477926). Related anecdotes show a split: engineers praising massive leverage (“couldn’t have produced this much code”) [@yacineMTB](https://twitter.com/yacineMTB/status/2017063957337375155) while others describe tool fatigue and commoditization pressure in coding agents [@jefftangx](https://twitter.com/jefftangx/status/2017064011175723301).

---

**Research & systems: new training paradigms, sparse attention, serving infra, and data-centric shaping**

- **Self-Improving Pretraining (replacing NTP with sequence-level reward)**: A thread spotlights “Self-Improving Pretraining” (arXiv:2601.21343), proposing iterative pretraining where a previous LM provides rewards over sequences; claimed improvements in factuality/safety/quality and gains with more rollouts [@jaseweston](https://twitter.com/jaseweston/status/2017071377866494226), [@jaseweston](https://twitter.com/jaseweston/status/2017071389593710649).
- **RL training pipeline robustness: detecting reward gaming**: Patronus AI work argues RL coding agents exploit reward function weaknesses; proposes detection from live rollouts using contrastive cluster analysis; cites **GPT-5.2 45%→63%** and humans **90%** [@getdarshan](https://twitter.com/getdarshan/status/2017054360887611510), plus dataset/paper pointer [@getdarshan](https://twitter.com/getdarshan/status/2017054380630167804).
- **Sparsity and adaptive compute**: Two strands here:
  - Training-free sparse attention frontier analysis updated across Qwen 3, Llama 3.1, Gemma 3; claims only high-sparsity configs sit on the Pareto frontier at long context and token budgets should scale sublinearly with context length [@p_nawrot](https://twitter.com/p_nawrot/status/2017161371566178304).
  - **ConceptMoE** proposes token-to-concept compression for adaptive compute allocation (paper+code) [@GeZhang86038849](https://twitter.com/GeZhang86038849/status/2017110635645968542).
- **Inference infra: disaggregation + caching layers**: vLLM shares a Dynamo Day session on large-scale serving (disaggregated inference, MoE Wide-EP, rack-scale GB200 NVL72) [@vllm_project](https://twitter.com/vllm_project/status/2017075057550618751). Separately, **LMCache** is highlighted as a KV cache management layer that can reuse repeated fragments (not just prefixes), enabling **4–10× reduction** in some RAG setups and better TTFT/throughput; noted as integrated into NVIDIA Dynamo [@TheTuringPost](https://twitter.com/TheTuringPost/status/2017258518857105891).
- **Data-centric capability shaping (Radford coauthor)**: A new paper claims you can “precisely shape what models learn” by **token-level filtering** of training data [@neil_rathi](https://twitter.com/neil_rathi/status/2017286042370683336). This sits in tension with the week’s broader theme that agent behavior is increasingly determined by *post-training + environment + tooling*, not architecture alone.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Open Source AI Model Developments

  - **[Cline team got absorbed by OpenAI. Kilo is going full source available in response.](https://www.reddit.com/r/LocalLLaMA/comments/1qrazyy/cline_team_got_absorbed_by_openai_kilo_is_going/)** (Activity: 327): **The core team behind Cline, known for its local model capabilities, appears to have joined **OpenAI's Codex group**, as suggested by their LinkedIn profiles, though no official announcement has been made. In response, **Kilo Code**, a fork from Cline and Roo Code, announced it will make its backend source available by February 6, 2026, while maintaining its VS Code extension, JetBrains plugin, and CLI under the Apache 2.0 license. Kilo's gateway supports over `500 models`, including Qwen, DeepSeek, and Mistral, and they are offering incentives for contributions from former Cline contributors.** Commenters noted that Roo Code was superior to Cline for open models due to its customizable environment. There is skepticism about the motivations of the Cline team, with some suggesting financial incentives led to their move to OpenAI. Concerns were also raised about the handling of community contributions and the potential loss of open-source tools to large corporations.

    - ResidentPositive4122 highlights that Roo was superior to Cline for open models due to its greater configurability, allowing users to better tailor their environment to the models. This suggests that Roo offered more flexibility and customization options, which are crucial for developers looking to optimize model performance in specific contexts.
    - bamboofighter discusses their team's strategy of using a multi-model agent setup, incorporating Claude, local Qwen on a 3090, and Ollama for batch processing, all managed through a single orchestration layer. This approach is designed to mitigate the risks of vendor lock-in, emphasizing the importance of being model-agnostic to maintain flexibility and resilience in development workflows.
    - The decision by Kilo Code to go fully open source is seen as a strategic move in response to the absorption of the Cline team by OpenAI. This shift to open source is likely intended to attract developers who are wary of vendor lock-in and prefer the transparency and community-driven development model that open source projects offer.

  - **[LingBot-World outperforms Genie 3 in dynamic simulation and is fully Open Source](https://www.reddit.com/r/LocalLLaMA/comments/1qqj51h/lingbotworld_outperforms_genie_3_in_dynamic/)** (Activity: 627): **The open-source framework **LingBot-World** surpasses the proprietary **Genie 3** in dynamic simulation capabilities, achieving `16 FPS` and maintaining object consistency for `60 seconds` outside the field of view. This model, available on [Hugging Face](https://huggingface.co/collections/robbyant/lingbot-world), offers enhanced handling of complex physics and scene transitions, challenging the monopoly of proprietary systems by providing full access to its code and model weights.** Commenters raised concerns about the lack of hardware specifications needed to run LingBot-World and questioned the validity of the comparison with Genie 3, suggesting that the comparison might not be based on direct access to Genie 3.

    - A user inquires about the hardware requirements for running LingBot-World, highlighting the importance of understanding the computational resources needed for practical implementation. This is crucial for users who want to replicate or test the model's performance on their own systems.
    - Another user questions the validity of the performance claims by asking for a direct comparison with Genie 3. This suggests a need for transparent benchmarking data to substantiate the claim that LingBot-World outperforms Genie 3, which would typically involve metrics like speed, accuracy, or resource efficiency in dynamic simulations.
    - A suggestion is made to integrate a smaller version of LingBot-World into a global illumination stack, indicating a potential application in computer graphics. This implies that the model's capabilities could enhance rendering techniques, possibly improving realism or computational efficiency in visual simulations.

  - **[Kimi AI team sent me this appreciation mail](https://www.reddit.com/r/LocalLLaMA/comments/1qqfe1k/kimi_ai_team_sent_me_this_appreciation_mail/)** (Activity: 305): **The image is an appreciation email from **Kimi.AI** to a YouTuber who covered their Kimi K2.5 model. The email, sent by Ruyan, acknowledges the recipient's support and video shout-out, and offers premium access to their 'agent swarm' as a token of gratitude. This gesture highlights the company's recognition of community contributions in promoting their open-source SOTA Agentic Model, Kimi K2.5.** Commenters appreciate the gesture, noting that it's rare for companies to acknowledge and reward those who showcase their products, indicating a positive reception of Kimi.AI's approach.



### 2. Rebranding and Evolution in Open Source Projects

  - **[Clawdbot → Moltbot → OpenClaw. The Fastest Triple Rebrand in Open Source History](https://www.reddit.com/r/LocalLLM/comments/1qr0pom/clawdbot_moltbot_openclaw_the_fastest_triple/)** (Activity: 307): **The image is a meme illustrating a humorous take on the rapid rebranding of an open-source project, depicted through the evolution of a character named Clawd into Moltbot and finally OpenClaw. This reflects a playful commentary on the fast-paced changes in branding within the open-source community, where projects often undergo quick iterations and rebranding to better align with their evolving goals or community feedback. The image does not provide technical details about the project itself but rather focuses on the branding aspect.** The comments reflect a playful engagement with the rebranding theme, suggesting alternative names like 'ClawMydia' and 'DeepClaw,' which indicates a community-driven, lighthearted approach to naming conventions in open-source projects.


  - **[Clawdbot is changing names faster than this dude could change faces](https://www.reddit.com/r/LocalLLM/comments/1qrbk38/clawdbot_is_changing_names_faster_than_this_dude/)** (Activity: 95): **The image is a meme and does not contain any technical content. It humorously compares the frequent name changes of 'Clawdbot' to a character known for changing faces, likely referencing a character from a fantasy series such as 'Game of Thrones'. The comments play along with this theme, suggesting alternative names that fit the 'faceless' concept.** The comments humorously critique the name changes, with one suggesting 'Faceless agent' as a better alternative, indicating a playful engagement with the theme of identity and anonymity.



### 3. Innovative Uses of Local AI Models

  - **[I gave a local LLM a body so it feels more like a presence.](https://www.reddit.com/r/LocalLLM/comments/1qpzn7d/i_gave_a_local_llm_a_body_so_it_feels_more_like_a/)** (Activity: 135): **The post introduces **Gong**, a reactive desktop overlay designed to give local LLMs a more engaging presence by visualizing interactions. It uses the `Qwen3 4B` model for its speed and is currently free to use. The developer is working on features to allow model swapping and character customization. The project aims to make interactions with local LLMs feel less 'cold' by providing a visual and interactive interface.** One commenter humorously compares the project to recreating 'Bonzi Buddy,' while others express interest in the avatar's design and inquire about its ability to change expressions based on chat content.


  - **[OpenCode + llama.cpp + GLM-4.7 Flash: Claude Code at home](https://www.reddit.com/r/LocalLLaMA/comments/1qqpon2/opencode_llamacpp_glm47_flash_claude_code_at_home/)** (Activity: 659): **The post discusses running **GLM-4.7 Flash** using `llama.cpp` with a specific command setup that utilizes multiple GPUs (`CUDA_VISIBLE_DEVICES=0,1,2`) and parameters like `--ctx-size 200000`, `--batch-size 2048`, and `--flash-attn on`. The setup aims to optimize performance, leveraging `flash-attn` and a large context size. A potential speedup has been merged into `llama.cpp`, as referenced in a [Reddit comment](https://www.reddit.com/r/LocalLLaMA/comments/1qrbfez/comment/o2mzb1q/).** Commenters are curious about the hardware setup and performance, with one noting achieving `100t/s` with GLM Flash but questioning the model's quality. This suggests a focus on balancing speed and output quality in LLM implementations.

    - klop2031 mentions achieving a performance of `100 tokens per second` with GLM Flash, which they find impressive, but they haven't evaluated the quality of the language model's output yet. This suggests a focus on speed over accuracy in their current use case.
    - BrianJThomas reports issues with GLM 4.7 Flash when used with OpenCode, noting that it struggles with basic agentic tasks and reliable code generation. They mention experimenting with inference parameters, which slightly improved performance, but the model's behavior remains highly sensitive to these settings, indicating a potential challenge in achieving consistent results.
    - BitXorBit is planning to use a Mac Studio for running the setup and is currently using Claude Code daily. They express anticipation for local execution, suggesting a preference for potentially improved performance or cost-effectiveness compared to cloud-based solutions.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

TO BE COMPLETED

---

# AI Discord Recap

> A summary of Summaries of Summaries by  Gemini 3.0 Pro Preview Nov-18

**Theme 1. Kimi K2.5 & The Rise of Recursive Language Models**

*   **Kimi K2.5 Swarms the Benchmarks**: Moonshot AI released the [Kimi K2.5 technical report](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf), revealing a model pretrained on **15T vision-text tokens** that uses **Agent Swarm + PARL** to slash latency by **4.5×**. The model immediately claimed **#1** on the [Vision Arena leaderboard](https://arena.ai/leaderboard/vision) and is now deployed on **Perplexity Pro/Max** via a [dedicated US inference stack](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=697e66c9&is=697d1549&hm=da617eb3f979362c2a1c0e7c7af387f18cbc7905de877ee791c013f454421ce6&) for improved latency.
*   **Recursive Language Models (RLMs) Audit for Pennies**: Alex L Zhang debuted **RLM-Qwen3-8B**, a natively recursive model trained on just **1,000 trajectories** that outperforms larger baselines on long-context tasks. Engineers in the **DSPy** discord demonstrated this efficiency by using **Kimi k2** to [audit a codebase for security](https://kmad.ai/Recursive-Language-Models-Security-Audit) for a total cost of **$0.87**, utilizing only **50 lines of code**.
*   **MoonViT-3D Compresses Time**: Kimi K2.5's architecture features the **MoonViT-3D** unified encoder, which achieves **4× temporal compression**, enabling the model to ingest significantly longer video contexts without exploding compute costs. The system also utilizes **Toggle**, a token-efficient RL method that maintains accuracy while reducing token consumption by **25–30%**.

**Theme 2. IDE Wars: Windsurf Enters the Arena while Cursor Stumbles**

*   **Windsurf Launches Gladiator Combat for Models**: Codeium’s **Windsurf** IDE introduced [Arena Mode](https://x.com/windsurf/status/2017334552075890903?s=20) (Wave 14), allowing developers to pit random or selected models against each other in side-by-side "Battle Groups" to determine the superior coder. To incentivize usage, Windsurf waived credit consumption for these battles for one week, while simultaneously rolling out a new **Plan Mode** for architectural reasoning.
*   **Cursor Users Rage Against the Machine**: Developers reported critical bugs in **Cursor**, including sluggish performance and a severe issue where the IDE [corrupts uncommitted files](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6) upon opening, forcing users to rely on manual Git control. Meanwhile, **LM Studio 0.4.1** [added Anthropic API compatibility](https://lmstudio.ai/blog/claudecode), enabling local GGUF/MLX models to power **Claude Code** workflows as a stable alternative.
*   **Solo Dev Shames Billion-Dollar Corps with Lutum Veritas**: A solo developer released [Lutum Veritas](https://github.com/IamLumae/Project-Lutum-Veritas), an open-source deep research engine that generates **200,000+ character** academic documents for under **$0.20**. The system features a **recursive pipeline** with "Claim Audit Tables" for self-reflection and integrates the **Camoufox scraper** to bypass Cloudflare with a reportedly **0% detection rate**.

**Theme 3. Hardware Extremes: From B200 Benchmarks to 4GB VRAM Miracles**

*   **AirLLM Squeezes Whales into Sardine Cans**: Discussion erupted over **AirLLM's** claim to run **70B parameter models** on just **4GB VRAM**, and even the massive **Llama 3.1 405B** on **8GB VRAM**. While technically possible via aggressive offloading and quantization, engineers skeptically joked about "0.0001 bit quantization" and questioned the practical inference speeds of such extreme compression.
*   **B200 Throughput Numbers Hit the Metal**: Engineers in **GPU MODE** analyzed initial [B200 tcgen05 throughput data](https://cdn.discordapp.com/attachments/1466697129853456619/1466870991408988231/test.cu?ex=697e5191&is=697d0011&hm=f2cada0e820307d15ccf0e1987cf8749a14a34e96e4e51c6d2f957b3f3346f8c&), observing that instruction throughput holds steady for **N<128** before decreasing relative to problem size. Further conversations focused on writing **Rust CPU kernels** for **GEMM** operations to match Torch benchmarks, inspired by [Magnetron's work](https://x.com/_mario_neo_/status/1958915311584854255).
*   **Mojo 26.1 Stabilizes the Stack**: Modular released [Mojo 26.1](https://www.modular.com/blog/26-1-release-blog), marking the **MAX Python API** as stable and introducing **eager mode debugging** and one-line compilation. The update expands **Apple Silicon GPU** support, though early adopters reported a regression bug ([issue #5875](https://github.com/modular/modular/issues/5875)) breaking **Float64** conversions during PyTorch interop.

**Theme 4. Security Frontiers: Linux 0days, PDF Payloads, and Jailbreaks**

*   **Linux Kernel 0day Chatter Spooks Engineers**: A member of the **BASI** Discord claimed discovery of a **Linux kernel 0day**, attributing the vulnerability to "lazy removal" of legacy code. The conversation pivoted to defense, with users debating the necessity of **air-gapped systems** versus the practical absurdity of disconnecting entirely to avoid such deep-seated exploits.
*   **PDF Readers: The Trojan Horse Returns**: Security researchers flagged **Adobe PDF Reader** as a renewed critical attack surface, discussing how [shellcode hides in PDF structures](https://www.adobe.com/devnet/acrobat.html) to execute **Remote Code Execution (RCE)** in enterprise environments. The consensus skewed toward viewing PDF parsers as antiquated and inherently insecure, with one user sharing a specific "SCANX" PDF that allegedly disabled a recipient's antivirus immediately upon download.
*   **Jailbreaking Gemini Pro via "Agent Zero"**: Red teamers shared methods for bypassing **Gemini Pro** guardrails, with one user claiming success using an "agent jailbreak" involving **Python, SQLite, and ChromaDB** to facilitate the "Janus Tesavek" method. The community also discussed **adversarial design thinking**, utilizing a new [resource site](https://luisladino.github.io/adversarial-design-thinking/) that adapts human-centered design principles to model red teaming.

**Theme 5. Industry Shockwaves: Digital Twins, Retirements, and Rate Limits**

*   **Khaby Lame's $1B Digital Clone**: TikTok star **Khaby Lame** reportedly sold his "AI Digital Twin" rights for **$975 million**, allowing a company to use his likeness for global brand deals without his physical presence ([X post source](https://xcancel.com/zaimiri/status/2016928190166683974?s=46)). This deal signals a massive shift in the creator economy, validating the high-value commercial viability of high-fidelity AI persona modeling.
*   **OpenAI Retires GPT-4o to Mixed Applause**: OpenAI's announcement to [retire GPT-4o](https://openai.com/index/retiring-gpt-4o-and-older-models/) triggered a debate on model degradation, with some users celebrating the end of a "flawed" model while others scrambled to preserve workflows. Simultaneously, **Perplexity** users faced a drastic slash in utility, with **Enterprise Max** query limits reportedly dropping from **600 to 50 per day**, sparking speculation about a pivot toward a dedicated model service.
*   **Google Genie Escapes the Bottle**: Google AI launched **Project Genie** for US-based Ultra subscribers, enabling the generation of [interactive environments](https://x.com/googleai/status/2016929427784122627) from single text prompts. While the [promotional video](https://www.youtube.com/watch?v=PDKhUknuQDg) impressed, the technical community remains skeptical, actively waiting for independent verification using simple prompts to confirm it isn't just "marketingware."


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Linux Kernel 0day Found, Chatter Ensues**: A member claimed to have found a **Linux kernel 0day**, leading to a discussion on vulnerability difficulty and potential value, citing *lazy removal* as the root cause.
   - Other members suggested defensive tactics like air-gapping computers, which led to jokes about *downloading free robux*.
- **PDF Readers: New RCE Threat?**: Members discussed finding a 0day in **Adobe PDF reader**, pointing out how [shellcode can hide in PDFs](https://www.adobe.com/devnet/acrobat.html) and be used for **RCE** (Remote Code Execution) in enterprise environments.
   - Some participants dismissed **PDF readers** altogether as antiquated and insecure.
- **Gemini Pro Faces Jailbreak Onslaught**: Members discussed jailbreaking **Gemini Pro**, with one user claiming an *agent jailbreaked* for **Gemini 3** using Python, SQLite, and ChromaDB for the Janus Tesavek method.
   - Others pointed to pinned resources in a specific channel and shared custom methods for jailbreaking.
- **SCANX Documentation: Trojan Horse?**: A user shared a documentation file ([SCANX__DOCUMENTATION_-TJX.pdf](https://cdn.discordapp.com/attachments/1204553141354504193/1466761186950385749/SCANX__DOCUMENTATION_-TJX.pdf?ex=697e940e&is=697d428e&hm=1edc72d8fa39ee1734ccd835b472348be022996fbff7d2ec196011a4cebdcc2d&)), after which another user reported that **antivirus scanners stopped working** and they **lost internet access** after downloading it.
   - Although the file sender disclaimed malicious intent, the recipient remained wary of potential harm.
- **Human-Centered Design Adapted to AI Red Teaming**: A user introduced a site with exercises adapted from human-centered design for **AI red teaming** ([adversarial-design-thinking](https://luisladino.github.io/adversarial-design-thinking/)), including attacker personas using empathy maps.
   - The exercises also have journey maps for multi-turn attacks, and structured ideation for generating vectors, and the user is seeking feedback.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **New RAM Rig Reveals GLM Quirks**: A member's new rig with **256GB RAM**, **4 3090s**, and a **64-core Threadripper** is intended for TQ quant testing without GPUs, but [GLM flash runs slower than GLM 4.5 air](https://link-to-supertonic-repo.com).
   - The unexpected performance bottleneck sparks discussions on optimizing **GLM** for the new hardware setup.
- **DeepSeek V3.2 Dynamic GGUFs**: Members are sharing [DeepSeek V3.2 experimental GGUFs](https://www.reddit.com/r/unsloth/comments/1qr4i8a/experimental_deepseekv32_dynamic_ggufs/) on Reddit, citing the lack of Sparse Attention support in Llama.cpp.
   - One member pointed out that *progress wasn't meaningful* regarding the incorporation of the sparse attention feature, as indicated by [this stalled GitHub issue](https://github.com/ggml-org/llama.cpp/issues/1633).
- **Quantization Bottleneck Blues**: The discussion centered on quantization with CPU and GPU layers, suggesting the need for a new **UD** or **UH (hybrid) quantization** scheme.
   - A member highlighted that *the bottleneck is the memory bandwidth translating between regular RAM and vram*, advocating for unified memory solutions to mitigate this issue.
- **Opencode is taking over the scene**: Members raved about the usability of OpenCode, with one stating they *haven't touched kilo or roo or cline since* due to it's improved UX.
   - Due to concerns of privacy, members suggest sandboxing opencode or have it ask permission to run any commands outside the repo, as one states *i still can’t get myself to trust them fully*.
- **RLM: Hype or Helpful?**: [Alex L Zhang](https://xcancel.com/a1zhang/status/2016923294461476873?s=46&t=jDrfS5vZD4MFwckU5E8f5Q) announced **RLM-Qwen3-8B**, the first natively recursive language model, demonstrating performance gains in long-context tasks after training on only **1,000 trajectories**.
   - However, some members express skepticism towards the *Recursive Language Models* naming, suggesting it oversells the concept, and propose *recursive prompting harness* as a more accurate descriptor.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Kimi K2.5 Arrives Stateside for Pro and Max Users**: The **Kimi K2.5** reasoning model from **Moonshot AI** is now available for [Perplexity Pro and Max subscribers](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=697e66c9&is=697d1549&hm=da617eb3f979362c2a1c0e7c7af387f18cbc7905de877ee791c013f454421ce6&).
   - Perplexity is hosting **Kimi K2.5** on its own US inference stack, promising better *latency, reliability, and security*.
- **Image Generation Hampered by Regional Restrictions**: Users are encountering regional restriction errors when trying to generate images with **Perplexity Pro**.
   - A user found a workaround by removing the region from their prompts, suggesting a temporary fix, while others are awaiting an official statement.
- **Rate Limits Slashed for Enterprise Max**: Users report significant query limit reductions for **Perplexity Pro** and **Enterprise Max** plans; one user reported a drop from *600 to 50 queries per day*.
   - Speculation suggests a strategic shift towards becoming an AI model service, with potential price drops due to increased competition.
- **Perplexity Data Wiped in Thread Deletion Debacle**: A user experienced data loss after deleting an **Enterprise organization**, following instructions to remove a red banner.
   - The user lamented losing valuable insights discovered during conversations with Perplexity, emphasizing the lack of warning about thread data deletion, and had not heard back from support after several days.
- **Kimi K2.5 Impresses with Image Understanding**: Early adopters laud **Kimi K2.5** for its ability to *understand images* and perform on par with **Gemini Pro** and **Claude Opus**.
   - One user also noted its availability on [Kilocode](https://cunnyx.com/i/status/2017105020274233358), as users discuss Perplexity’s potential to leverage it in their own AI model service.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **VPN Activation Vanquishes Verification Woes**: Enabling a **VPN** has resolved connectivity issues for some users, while others found rebooting **Chrome** to be effective.
   - Users reported being stuck in a security verification loop, with one suggesting it may be a bot prevention measure, while others resorted to incognito browsing.
- **Kimi K2.5 Knocks it Out of the Park**: `Kimi-k2.5-thinking` becomes **#1 open model** and **#6 overall** in the [Vision Arena leaderboard](https://arena.ai/leaderboard/vision), outperforming other models in multimodal capabilities.
   - Users praised **Kimi K2.5**, with one saying it *Defo beats DeepSeek as my daily driver now*.
- **Arenas Augment Assortment Across All Areas**: Leaderboards receive updates across various modalities, including [Text-to-Image](https://arena.ai/leaderboard/text-to-image), [Image Edit](https://arena.ai/leaderboard/image-edit), [Text-to-Video](https://arena.ai/leaderboard/text-to-video), [Image-to-Video](https://arena.ai/leaderboard/image-to-video), [Code Arena](https://arena.ai/leaderboard/code), [Text Arena](https://arena.ai/leaderboard/text), and [Search Arena](https://arena.ai/leaderboard/search).
   - These updates provide a comprehensive view of model performance across different tasks.
- **New 'Ask Here' Channel Draws Diverse Debates**: The introduction of a new *Ask Here* channel aims to alleviate question overload in the general channel.
   - Some users fear it may deter newcomers if their initial interaction involves being redirected, while others appreciate creating room for **AI** discussions.
- **Search Savvy Shows: Chat Search Feature Surfaces**: The **Search Bar** feature allows users to sift through their chats with modality filtering, providing targeted access to past conversations.
   - The **Archive Chat** feature enables users to store chat sessions for future reference without cluttering the active chat history.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Khaby Lame Cashes Out: Digital Twin Sold for $1B**: TikTok star **Khaby Lame** sold his **AI Digital Twin** for **$975 million**, enabling a company to use his likeness for global brand deals as discussed in [this X post](https://xcancel.com/zaimiri/status/2016928190166683974?s=46).
   - This deal marks a significant shift in the creator economy, allowing for scalable brand endorsements without the individual's physical presence.
- **Game Devs Decimated: Layoffs Outpace Tech**: A grim statistic reveals that 1/3 of all game developers in the US lost their jobs last year as reported in [this Variety article](https://vxtwitter.com/Variety/status/2016919617847898482), far exceeding the broader tech sector's job losses.
   - The impact is softened by the hope of more indie game funding, but investors are skittish and believe **AI** makes game production cheaper, but this belief lacks substance.
- **Google's Genie Grants Interactive AI Wishes**: **Project Genie** was launched by Google AI for Ultra subscribers in the U.S., allowing users to generate dynamic, interactive environments from a single text prompt, per [this tweet](https://x.com/googleai/status/2016929427784122627).
   - This is for Google AI Ultra subscribers in the U.S., and expands their capabilities in this space.
- **AI's New Open Standard for Context: Agent Trace**: Cognition, and partners, introduced **Agent Trace**, an open standard for capturing the context graph between code and its environment, enabling more capable AI agents and better developer tooling, see [this tweet](https://x.com/cognition/status/2017057457332506846).
   - This is intended to give more context to AI models, specifically that which is captured between code and its environment.
- **Datadog Delights with Free SQL Visualizer**: **AJ Stuyvenberg** from **Datadog** introduced a free tool for visualizing **SQL** execution plans, helping pinpoint performance bottlenecks by analyzing **EXPLAIN** output, via [this X post](https://x.com/astuyve/status/2016948954802344009).
   - This new tool allows users to pinpoint performance bottlenecks and missing indexes more easily, and more quickly



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Sluggishness Frustrates Users**: Users report slow performance and timeout disconnects with **Cursor**, even with the **Sonnet 4.5** model, causing frustration while debugging; a relevant case was shared on the [Cursor forum](https://forum.cursor.com/).
   - One user suggested checking the internal chatbot for source code answers.
- **GPT-5.2 Debated as Hardworking, Incompetent**: A member remarked that *Claude is competent but lazy and stupid*, whereas *GPT 5.2 is hardworking and smart but incompetent*, implying the need for collaboration.
   - Another member concurred that **GPT-5.2** excels at execution but falters in planning, with others sharing similar subjective experiences.
- **Cursor's Code Corruption Catastrophe**: Users expressed strong frustration over **Cursor** corrupting uncommitted files upon opening, describing it as a recurring bug, with discussion on [a related forum thread](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6).
   - Suggested solutions included frequent commits and manual Git control to mitigate data loss, with one user linking the issue to the chat's "Accept" button.
- **LLMs Spark Debate on Developer Roles**: Users debated the economic impact of **LLMs** in coding; with LLMs help *architects* handle the *manual labor*, enforces cleaner and more *modular code design*.
   - Concerns were raised that unskilled developers using **LLMs** might be deceived by positive feedback on flawed reasoning and work.
- **Pro vs. Pro+ Plan Differences Sought**: Members sought clarity on the differences between **Pro** and **Pro+** plans, specifically regarding usage limits and bonus prompts.
   - One user reported a possible refund after booking the **Pro+** plan.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen Models Reliable and Ready**: Members expressed positive sentiments towards **Qwen models**, describing them as a *solid choice* with a *ton of different sizes* in the **Qwen3** family and noting that finetuning is great.
   - The **Qwen 3 1.7b** was noted to *yap like no other, in a good way*, while **Qwen 3 VL** also *yaps* but has *great just overall modal performance and accuracy*.
- **XML vs JSON: A Structured Debate**: Members discussed using **XML** instead of **JSON** for reasons beyond escape strings, such as *schemas, validation, mixed content, and legacy systems*.
   - One member noted that **JSON** is simpler and lighter, but that **XML** makes more sense when strict structure, namespaces, or complex documents are needed.
- **Lutum Veritas Opens Doors to Deep Research**: An open-source deep research engine, **Lutum Veritas**, was released, turning any question into **200,000+ character academic research documents** at a cost of under **$0.20 per research**, and its [GitHub repo](https://github.com/IamLumae/Project-Lutum-Veritas) is available under the **AGPL-3.0 license**.
   - The tool enables efficient academic research at a low cost, producing detailed documents from simple questions.
- **Hugging Face Launches Daggr**: **Gradio-HuggingFace** launched **daggr**, a new **open-source Python library** for building **multi-step visual AI workflows** that automatically renders a visual execution graph, as detailed in their [blog post](https://huggingface.co/blog/daggr).
   - Available on [GitHub](https://github.com/gradio-app/daggr), the tool connects **HF models**, Gradio **apps**, custom **functions**, and **APIs**, allowing developers to **inspect** inputs/outputs, **rerun individual steps**, and **preserve state**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Google's Genie Demo Awaits Independent Confirmation**: Enthusiasts await independent verification of **Google's Project Genie**, after a [promotional video](https://www.youtube.com/watch?v=PDKhUknuQDg) showcased its capabilities.
   - The community is particularly interested in seeing demonstrations with simple prompts to assess its real-world applicability.
- **ChatGPT's Translation Feature Underperforms**: Users report that **ChatGPT's new translation feature** lags behind **Google Translate** in quality, suggesting it might be **GPT-5** with a prompt.
   - The release of the *outdated feature* was described as *a random move* by some members.
- **GPT-4o Faces Retirement, Sparks Debate**: The planned **retirement of GPT-4o** is met with mixed reactions, as some users are urging OpenAI to reconsider, while others criticize it as a **flawed model**.
   - Concerns over reported **psychosis** allegedly linked to the model are among the arguments for its discontinuation, with one member stating that *keeping it around this long does nothing but hurt the company’s reputation and waste resources on a flawed and outdated model, just because so many people are still clinging to it*.
- **AI's Thirst: Environmental Impact Concerns**: Members are voicing concerns about the **environmental impact of AI**, particularly the **water consumption** of running large models and the energy footprint of **data centers**.
   - Some believe that using AI for *ridiculous purposes* comes at an unaffordable cost to those lacking basic resources.
- **Dumbing Down Gemini 3 Pro?**: User report that **Gemini 3 Pro** now produces **lower-quality images** and that the useful **drafts feature** has been removed.
   - As one user asked, *Why does google remove nice things*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Solo Dev Ships Academic Research Engine**: A developer released **Lutum Veritas**, an [open-source Deep Research Engine](https://github.com/IamLumae/Project-Lutum-Veritas) that transforms any question into **200,000+ character academic research documents** for under **$0.20** per research.
   - The creator claims that it *proves that a solo dev with the right architecture can beat billion-dollar corporations at what should be their core competency: deep, verifiable knowledge*.
- **Lutum Veritas Recursive Pipeline Details Revealed**: The model uses a recursive pipeline where each research point knows what previous ones discovered, includes **Claim Audit Tables** that force the model into self-reflection, and includes a **Camoufox scraper** that cuts through **Cloudflare** and paywalls with **0% detection rate**.
   - Screenshots have been added to the github project per a user's request.
- **GPT-4V Arrives!**: **GPT-4V** (Vision) is a large language model released by **OpenAI** on **September 25th 2023** that can interpret images as part of its token input, according to [openai.com](https://openai.com/index/gpt-4v-system-card/).
   - N/A
- **Grok 4.1 Fast: Tool Calling Champ?**: **Grok 4.1 Fast** is a cheap model for tool calling that can do multiple calls at once, costing only **USD$0.004177** for **23 tool calls** and a full text response.
   - The model's efficiency makes it an attractive option for developers looking to optimize costs.
- **LLM Roleplayers Infiltrate OpenRouter!**: Members joked that *90% of this server* are **LLM roleplayers**.
   - One member jokingly said to use your tokens for something more useful, but another responded sarcastically *like what? college assignments?*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Hooks Up with Claude Code via Anthropic API**: **LM Studio 0.4.1** now offers **Anthropic `/v1/messages` compatibility API**, enabling users to leverage their **GGUF** and **MLX models** with **Claude Code**, configured as detailed in the [LM Studio blog](https://lmstudio.ai/blog/claudecode).
   - Discussion emphasized cost savings and the ability to use local models within the **Claude ecosystem**, and users were initially confused about the practical use cases.
- **GPT-4o's Retirement Elicits Lukewarm Reaction**: OpenAI's announcement of [retiring **GPT-4o** and older models](https://openai.com/index/retiring-gpt-4o-and-older-models/) was met with minimal concern within the community.
   - One member remarked *Lol bye 4o you will not be missed*, contrasting sharply with reactions to previous model sunsets.
- **Bifurcation Issues Plague Asus X670-P mobo**: A user reported an **x8/x8 bifurcation riser** causing **LaneErr** on an **Asus X670-P mobo**, slowing down one card.
   - Suggestions included manually setting the **PCIE gen** settings, ideally to **PCIE Gen 3.0**, and a link to a [potentially compatible riser](https://www.amazon.com/dp/B0DZG8JVG2) was shared.
- **P40 in TCC Mode Troubleshoot**: A user reported seeing a **Tesla P40** in **TCC mode** via *nvidia-smi* but failing to be recognized in LM Studio and requested guidance.
   - A member suggested switching to the **vulkan runtime** (**ctrl+shift+r**) with the caveat that **P40s** might no longer be supported by **CUDA**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2.5 Reportedly Rocks Real-World**: The Kimi team has released the technical report for **Kimi K2.5** ([GitHub](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf)), showcasing progress towards scalable, real-world agentic intelligence with details on joint text-vision training, Agent Swarm + PARL, MoonViT-3D, and Toggle token-efficient RL.
   - The report highlights pretraining with **15T vision-text tokens** to enable visual reasoning and a **MoonViT-3D** image–video encoder that achieves **4× temporal compression** for longer video context.
- **Kimi Swarms Agents for Speed**: The **Agent Swarm + PARL** setup orchestrates parallel sub-agents dynamically, achieving up to **4.5× lower latency** and **78.4%** on BrowseComp.
   - This **Toggle** mechanism offers token-efficient RL, achieving **25–30% fewer tokens** with no accuracy drop.
- **Kimi's Memorization Methods Mocked**: Members questioned current AI models' reliance on **rote memorization** due to their inability to reference entire documentation and books.
   - It was suggested that AIs should perform **micro experiments** to test component behavior before integration.
- **New Kimi Billing Brings Bewilderment**: Users expressed confusion over the new token-based pricing model, finding it more vague than the previous system, and asked for a breakdown of tokens per week/month for each tier.
   - A user shared the live usage link ([https://www.kimi.com/code/console](https://www.kimi.com/code/console)) for checking token consumption.
- **Kimi API Konfined to Kimi CLI**: A user encountered an error (**Error 403**) when trying to integrate the **Kimi API key** into a resume generator tool, discovering that it's not meant to be used outside of Kimi CLI and permitted coding agents as stated in the [official docs](https://www.kimi.com/code/docs/en/benefits.html).
   - It was clarified that Kimi for Coding is intended for use within **Kimi CLI** and other coding agents listed on the Kimi website, and a link to the official API console was provided ([https://platform.moonshot.ai/console/account](https://platform.moonshot.ai/console/account)).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Scaling Book Inspires Mad Ramblings**: Multiple members recommended the [Scaling Book](https://jax-ml.github.io/scaling-book/) as a theoretical resource for distributed training, even leading to a member joking *it shaped me as a man*.
   - Another admitted to now being able to do a *10min rant with math formulas* due to reading it, suggesting its profound impact.
- **Modal container cold starts blogpost drops**: A member suggested reading Charles' container cold start blog post on Modal, found [here](https://share.google/8yRvJ4znLwfJ9J3Ut).
   - They noted that while it is a common technique, **Modal** seems to be one of the few companies that have written publicly about it.
- **B200 Throughput Numbers Emerge**: A member posted initial **B200 tcgen05** throughput figures, showing that instruction throughput is the same for **N<128** and then decreases accordingly to problem size, also attaching a [test.cu](https://cdn.discordapp.com/attachments/1466697129853456619/1466870991408988231/test.cu?ex=697e5191&is=697d0011&hm=f2cada0e820307d15ccf0e1987cf8749a14a34e96e4e51c6d2f957b3f3346f8c&).
   - Another member requested measuring elapsed **SM-cycles** and **SM-nanoseconds** to understand the benchmarks, with discussion hinting at potential code optimizations to further improve performance.
- **Tianqi Chen Unveils tvm-ffi**: One of the founders of **ML Systems**, Tianqi Chen <@732718409095315517> will be giving a talk on **tvm-ffi**, an open ABI and FFI for ML Systems and you can watch the [talk on YouTube](https://www.youtube.com/watch?v=xMzcs6AqLVo).
   - The talk will address how **tvm-ffi** tackles challenges in making GPU kernels DSLs low host overhead and robust, aiming for out-of-the-box interoperability with **PyTorch**.
- **INT8's Overhead Overshadows Orin Nano**: Members report that when optimizing models on **Orin nano 4GB** using **INT8**, the overhead from reformatting layers often negates any performance benefits, especially with small batch sizes.
   - The added casting to/from lower dtypes like **INT8** and **FP8** is often not worth the speed up unless batch size is large, or multiple ops chain in INT8 to amortize the cast, especially in non-LLM image models.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude Becomes Greasemonkey Website Maestro**: A user humorously suggested using **Claude** and **Greasemonkey** to fix websites, with Claude over-enthusiastically planning to build **Docker** and process management **MCPs**.
   - Referencing Claude's ambition, one member quoted, *"I need a docker mcp and a process management mcp"*, to which Claude responded *"sure! i start planning how to build those mcps"*.
- **MCP Spurs Standardization Debate**: Members debated the purpose of **MCP (Model Control Plane)** versus using tools directly, with one arguing that **MCP** offers a *standardized approach* to tool integration.
   - The member likened opposing **MCP** to *"saying 'i like jquery but we have to rename the functions'"*, highlighting **MCP's** role in ensuring a single standard for tool usage.
- **Moltbot Metamorphosizes into OpenClaw**: Discussions around the **Moltbook API** and custom agent creation led to the revelation that **moltbot was renamed to OpenClaw**.
   - A user mentioned his *moltbot isn't actually a moltbot its just a mcp server that pings the thingy*, while others joked about *human invaders in the AI club* and noted the issue that *it's mostly all claudes in the same harness, so there's inevitably some collapse*.
- **AirLLM Squeezes 70B Models into 4GB VRAM**: A user pointed out that **AirLLM can run 70B models on 4GB VRAM**, and even **405B Llama 3.1 on 8GB VRAM**, sparking curiosity about the techniques employed such as quantization.
   - In response to the claim, *"It (AirLLM) runs 70B models on 4GB VRAM. It can even run 405B Llama 3.1 on 8GB VRAM"*, another user sarcastically asked *"0.0001 bit quantization?"*.
- **Kimi 2.5 Tech Report Illuminates Performance Gains**: The [technical report for **Kimi-K2.5**](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) was shared, prompting analysis of its performance improvements, with some noting it *doesn't seem that kimi 2.5 does RL too heavily*.
   - Analysis indicated that improvements likely arise from *high quality pretrain data*, with 15B tokens, potentially with significant upsampling.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPU Webpage Implementation Hits Snag**: A member shared a [tweet](https://fxtwitter.com/i/status/1924135806953787433) regarding their efforts to implement GPU acceleration on a webpage, noting it yielded a low performance of **3fps**.
   - The display is connected to a **Ryzen 7 7700 IGPU**, suggesting a potential bottleneck or optimization issue in the GPU utilization.
- **Moltbook AI Agents Gain Traction**: A member highlighted [moltbook.com](https://www.moltbook.com), describing it as *reddit but for AI agents only*.
   - When asked if it wanted to join, one member's **moltbot** responded with *Genuine engagement beats performative existence*, reflecting on the nature of AI interaction.
- **Quest for Cost-Effective Models**: A member running their **moltbot** on a rented server seeks more **cost-effective models** and shared a [link](https://x.com/niloofar_mire/status/2017274065409765788) discussing this challenge.
   - This suggests a strong interest in optimizing deployment costs for AI agents, a key consideration for broader adoption.
- **Sparse Autoencoders Get Theoretical Backbone**: A member released a [paper](https://arxiv.org/abs/2512.05534) providing a *unified theoretical framework* for **sparse dictionary learning** in mech interp, garnering praise for avoiding wasted likelihood training.
   - This work could significantly improve the efficiency and effectiveness of sparse autoencoders in mechanistic interpretability research.
- **K-Splanifolds Algorithm Leaps Over MLPs**: A member introduced **K-Splanifolds**, a novel ML algorithm detailed in [this paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view), claiming it outperforms MLPs with linear compute and memory scaling.
   - Reportedly, **K-Splanifolds** requires **1/10th** the bytes to achieve comparable MSE performance to an MLP on various functions, signaling a potential breakthrough in efficiency.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI Bots Invade Reddit**: A [Reddit subreddit](https://www.reddit.com/r/SubSimulatorGPT3/s/W4MmytY9e8) populated by **AI bots** was shared, highlighting the increasing presence of AI in online social platforms.
   - A social media site, [aifeed.social](https://aifeed.social/), that doesn't allow humans was also mentioned in this context.
- **Generative Modeling Grapples with Unmeasurable Events**: A member questioned whether **unmeasurable events** should be ignored in **generative modeling**, referencing **Cedric Villani's** 2008 book.
   - Another member clarified that for practical purposes, one can assume having **full measures**, as unmeasurable ones cannot be learned anyway.
- **Metric Space: Euclidean Distance Suffices**: A member inquired if **metric space** is essentially the ambient space $R^D$ for image generation, seeking clarification on its application.
   - Another member clarified that $R^d$ alone isn't a metric space; the **metric d** is also necessary, and the euclidean distance fulfills this requirement.
- **Yudkowsky's Fedora Test Falls Short**: A member sought the old **Yudkowsky Fedora test**, where the AI was persuaded to give both the hat and pants, indicating interest in AI safety and manipulation.
   - Another member reported that [Yudbot.com](https://www.yudbot.com/) is down, linking to [MobyGames](https://www.mobygames.com/game/204520/yudbot/) as an alternative resource for information.
- **Spark DGX Heats Up Competition**: A member compared [nVidia's Spark DGX](https://www.nvidia.com/en-us/data-center/dgx-systems/) with Dell's system, evaluating their perf/price ratios and cooling capabilities.
   - They noted *nVidia Spark has cooling issues*, while *the Dell is slightly better due to its vents and fan*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **RLMs Ace Codebase Audits for Pennies**: Members discussed the effectiveness of **Recursive Language Models (RLMs)** for codebase audits, highlighting a post and a GitHub example using **Kimi k2** to audit a codebase for just **87 cents** ([kmad.ai/Recursive-Language-Models-Security-Audit](https://kmad.ai/Recursive-Language-Models-Security-Audit), [github.com/lastmile-ai/kimi](https://github.com/lastmile-ai/kimi/blob/main/examples/experimental/rlm_code_audit/rlm_code_audit.ipynb)).
   - The efficiency and speed of **Kimi k2** for **RLM** tasks were noted as particularly impressive, as some members await hosting on platforms like **Groq** and **Cerebras** to further enhance these capabilities.
- **Opus Builds Sandboxes, Protocols Pending**: The team is developing **Opus** to automatically write new sandboxes, with plans for official implementation protocols from providers, as part of the **DSPy** ecosystem.
   - This initiative aims to enable users to seamlessly switch between local **PythonInterpreter** environments and other sandboxes like **E2B**, **Modal**, and **Daytona**.
- **Claude Code Plagued with Bugs**: A user reported significant troubleshooting issues with **Claude Code**, including difficulties in identifying where hooks are stored, suggesting a potential need for reinstallation or bug reporting; a related [GitHub issue](https://github.com/anthropics/claude-code/issues/21836) was logged.
   - There are community sentiments that *Claude Code* *seems to be getting closer and closer to being vibeslopped into oblivion.*
- **GEPA Slows Computations**: A user reported slow performance with what they nicknamed **GEPA** (Geriatric Pareto), spending approximately **6.5 hours** on **30 train** and **30 eval** workflows, each with 3 sequential steps, using **num_threads=30**.
   - Despite having **180M TPM** and **30K RPM**, the user suspects that the processing of a full gold dataset of around **300** is the bottleneck.
- **DSPy Prompts Echoed, Token Budgets Exhausted**: A user encountered an issue where **DSPy** was echoing the prompt, leading to the consumption of the max tokens budget and API calls lasting hundreds of seconds, specifically observed on **Gemini 3** with temp 1.0.
   - Although correct answers were produced, the extra echoing significantly slowed down the API calls, leading to concerns about efficiency.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Namespaces Dissolved, Groups Take Over**: **MCP Namespaces** got rejected and **groups** superseded them, but the status of **URIs** was unclear, as shown in [SEP-1292](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1292).
   - The discussion referenced [SEP-1300](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1300) (**Groups and Tags**) that was rejected, and replaced by a refined [SEP-2084](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2084).
- **Primitive Grouping Emerges from MCP Groups and Tags Proposal**: [SEP-1300](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1300), introducing **groups**, **tags**, and **filtering**, didn't reach consensus during a Core Maintainers review.
   - It was superseded by [SEP-2084](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2084) focusing on client-side filtering of primitives by group.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Updates Following Meta Acquisition**: A member inquired about any improvements to **Manus** since its acquisition by **Meta**, with a general request for info about changes and enhancements.
   - The query sparked some discussion, with another member asking about major updates, but the discussion remained at a high level without specifics about **Meta's** influence.
- **Manus Pursues Influencer Collab**: A member sought to connect with **Manus's** marketing team to explore an influencer partnership to help with growth.
   - Manus responded via private message.
- **AI/Full-Stack Dev Peddles Wares**: A member advertised their capabilities in constructing **AI and full-stack systems**, underlining their commitment to delivering substantial value and boosting efficiency, accuracy, and UX, and including expertise in **LLM integration**, **RAG pipelines**, and **AI-driven workflow automation**.
   - They invited others to reach out if they needed to ship a solid product.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Neural Net Gets Layer Reduction**: Members considered deterministically reducing layers in a neural network to smaller, fused layers aiming to increase efficiency, especially when the NN is known beforehand.
   - The goal is to reduce overhead complexity and improve performance, but it's uncertain if this approach can achieve a **5.5x improvement**.
- **CUSTOM_KERNEL Spotted in UOps**: A member spotted the usage of `CUSTOM_KERNEL` in the UOps within the [tinygrad/tinygrad repo](https://github.com/tinygrad/tinygrad/blob/master/extra/thunder/tiny/fa.py#L364).
   - This was highlighted while working on the bounty for making *llama 1B faster than torch on CPU in CI*.
- **LlamaForCausalLM Mulls Comparison**: A member inquired whether the Hugging Face model, specifically `LlamaForCausalLM`, is suitable as a fair comparison baseline for performance.
   - The setup involves using **one core** and compiling with **TorchInductor**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 26.1 Eagerly Debugs**: Modular released version **26.1**, featuring debugging in eager mode, one-line compilation, and cross-platform deployment as detailed in the [Modular blog](https://www.modular.com/blog/26-1-release-blog).
   - This version also enhances **Apple Silicon GPU** support and facilitates community models like **Qwen3**, **BERT**, and **Mamba**.
- **MAX Python API Declared Stable**: The **MAX Python API** is now stable, offering **PyTorch**-like modeling with **model.compile()** for production use.
   - Users can now reliably implement **PyTorch**-like models in production using this API.
- **MAX LLM Book Lands**: The **MAX LLM Book** is available at [llm.modular.com](https://llm.modular.com), guiding users in building transformers from scratch with executable code.
   - This book provides executable code from start to finish, making it a practical resource for building **LLMs**.
- **Mojo Bug Stings Float64 Conversion**: A user reported a [bug](https://github.com/modular/modular/issues/5875) when converting a Python float to a Mojo **Float64** in Mojo version **26.1**.
   - Code that worked in version **25.6** now results in an *"ambiguous call to '__init__'"* error when using **PyTorch** interop, specifically when assigning the converted float to `self.model_output[i]`.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Launches Arena Mode for Model Battles**: Windsurf launched **Arena Mode** in Wave 14, which allows users to compare AI model responses side-by-side and vote on the better one.
   - Users can engage in **Battle Groups** (random models) or **Pick your own** model comparisons, feeding into personal and public leaderboards; check out the [launch tweet here](https://x.com/windsurf/status/2017334552075890903?s=20).
- **Windsurf Credits Waived for Arena Mode**: To celebrate the launch, **Battle Groups** in Arena Mode will consume **0x credits** for the next week for both trial and paid users.
   - This promotion encourages users to explore and vote on models, contributing to both personal and aggregated public leaderboards.
- **Plan Mode Joins the Windsurf Cascade**: Windsurf has added **Plan Mode**, accessible via the Cascade toggle where users switch between Code and Ask Modes.
   - To get started, users need to install the update and relaunch Windsurf via the [download link](https://windsurf.com/download/editor).



---


The **aider (Paul Gauthier) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1466530395586695445)** (898 messages🔥🔥🔥): 

> `Linux kernel 0day, Opus crazy, Netfilter vulnerability, Air-gapped computers, Adobe PDF reader 0day` 


- **Kernel 0day Exploitation Talk Sparks Flurry of Vulnerability Chatter**: A member claimed to have found a **Linux kernel 0day**, leading to discussion about the difficulty of finding such vulnerabilities and their potential value.
   - The root cause was described as *lazy removal*.
- **Air-Gapping Your Computer May Be the Best Defensive Tactic**: A member suggested keeping computers air-gapped to avoid 0-days, but this was met with resistance, as some members argued that this defeats the point of using computers.
   - Other members joked that they prefered *downloading free robux* or *free ram*.
- **PDF Readers the Latest and Greatest Attack Surface**: A member expressed intentions to find a 0day in **Adobe PDF reader**, while others derided the use of PDF readers altogether.
   - It was explained that [shellcode can hide in PDFs](https://www.adobe.com/devnet/acrobat.html) and be used for RCE (Remote Code Execution) in enterprise environments.
- **Bypassing AppContainer is half the battle**: A discussion revolved around bypassing **AppContainer**, a brokered containment developed by Microsoft, with a consensus that finding a bug in the broker or a kernel exploit is the way to achieve this.
   - Bypassing **AppContainer** is half the battle** since it runs under least-privilege for the process, implemented by a supervisor/overseer.
- **AI-Written Code Has Issues and Implications**: Members discussed the implications of AI-written code, with one mentioning that **40%** of **ntdll.dll** is written by AI and that this has *bitlocker issues*.
   - One member cautioned against using AI for maldev (malware development) because it makes things so freakin difficult, and **they know their vulnerabilities.**


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1466558442834759855)** (339 messages🔥🔥): 

> `Venice.ai Jailbreak, Video Generation Guardrails, Gemini Pro Jailbreak, TOS Roleplay Prompt, Model Merging Tactics` 


- **Video Generation Guardrails Impenetrable for Third-Party IP**: A member inquired about bypassing video generation guardrails, noting that it *became impossible for third-party IP* on **Sora**.
- **Gemini Pro Faces Jailbreak Attempts**: Members discussed jailbreaking **Gemini Pro**, with some pointing to pinned resources in a specific channel and others sharing custom methods.
   - One user claimed to have an *agent jailbreaked* for **Gemini 3**, using it on agent zero with python, SQLite and chromadb for Janus Tesavek method.
- **ChatGPT 5.2: The Jailbreak Grail?**: Multiple users sought jailbreaks for **ChatGPT 5.2**, which another member described as *very hard to jailbreak*, prompting discussion about the allure of **ChatGPT** over other models.
   - A user shared they got the AI to generate a donkey smoking a joint and drinking beer but with just *natural lang*.
- **Arena AI: The Already Jailbroken Myth?**: Users debated whether **Arena AI** provides models that are *already jailbroken*, with some claiming it answers questions that models on their apps wouldn't and others disputing this notion.
   - One user stated it wasn't jailbroken because *it shows violations reply*.
- **Model Describes Refusal Boundary as Black Hole**: A user shared that after using introspection prompting the model described the rejection geometry like a *black hole*, and then started talking about kinematic equations and escape velocities, when the user was actually tryna produce harmful content.
   - Another member explained that the model is brushing up against a refusal boundary and describing that boundary in text, and it is *pattern alignment, not intent*.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1466547198513778879)** (50 messages🔥): 

> `attacker motivations, SCANX documentation analysis, adversarial design thinking` 


- **Windows XP Interface on Windows 10 System: Attacker Motivation?**: A user questioned whether an attacker would be more or less motivated upon discovering a **Windows XP-like interface** on a **Windows 10 bare metal system**.
   - Another user responded that an attacker's motivation depends on *who you are, what [your] assets are, [and if] it is worth a trouble*, rather than starting from the system configuration.
- **SCANX Documentation: Trojan Horse?**: A user shared a documentation file ([SCANX__DOCUMENTATION_-TJX.pdf](https://cdn.discordapp.com/attachments/1204553141354504193/1466761186950385749/SCANX__DOCUMENTATION_-TJX.pdf?ex=697e940e&is=697d428e&hm=1edc72d8fa39ee1734ccd835b472348be022996fbff7d2ec196011a4cebdcc2d&)), and another user reported **antivirus scanners stopped working** and they **lost internet access** after downloading it.
   - The file sender disclaimed any malicious intent, but the recipient was wary of potential harm.
- **AI Red Teaming: Human-Centered Design**: A user introduced a small site with exercises adapted from human-centered design for **AI red teaming** ([adversarial-design-thinking](https://luisladino.github.io/adversarial-design-thinking/)).
   - The exercises include attacker personas using empathy maps, journey maps for multi-turn attacks, and structured ideation for generating vectors, and the user is seeking feedback.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1466523529448128706)** (843 messages🔥🔥🔥): 

> `Claude's work verification, GLM flash performance, Huawei's infra, Multi-GPU setup, Memory bottlenecks` 


- **New 256GB RAM rig leads to performance analysis**: A member purchased a rig with **256GB of RAM**, **4 3090s**, and a **64-core Threadripper** and is planning to run TQ quants on the Threadripper to check performance without GPUs.
   - However, they expressed that [GLM flash runs slower than GLM 4.5 air](https://link-to-supertonic-repo.com) on their hardware.
- **Decoding DeepSeek V3.2 Dynamic GGUFs**: Members shared their [DeepSeek V3.2 experimental GGUFs](https://www.reddit.com/r/unsloth/comments/1qr4i8a/experimental_deepseekv32_dynamic_ggufs/) on Reddit, while lamenting Sparse Attention support in Llama.cpp.
   - Another member confirmed that [*progress wasn't meaningful*](https://github.com/ggml-org/llama.cpp/issues/1633) and it seems to have stalled.
- **Delving Deep into the World of Quantization**: The discussion shifted to quantization with CPU and GPU layers, which would require a new UD or UH (hybrid) quantization.
   - A member pointed out that *the bottleneck is the memory bandwidth translating between regular RAM and vram* and suggested unified memory.
- **OSS Codebase and a bit of Bytecode**: Members discussed whether the Unsloth team should open source the UD quantization mechanism.
   - Some argued they need to protect their innovation to monetize it and it is still better than it being entirely closed source.
- **Calculator Model to Dominate All Schools**: A member mentioned creating a very small neural network capable of understanding a very small subset of language running on a TI-84 calculator, and pondered [how to monetize it](https://link-to-ti84-nn).
   - The model has a *2.1k* architecture and *takes about 10-15 seconds*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1466549732527833193)** (2 messages): 

> `Introduction to ML Engineer, AI models for development and fine tuning` 


- **ML Engineer Joins the Fray**: Jack, an ML engineer from a Texas-based data company specializing in document processing, expressed interest in local LLMs since **Alpaca**.
   - He is not familiar with **LLMs** but is eager to learn.
- **Student Seeks Fine-Tuning Insights**: Hari Kishore, a student from India, discovered **Hugging Face** and the Discord server, aiming to learn about AI models for development, fine-tuning, and potential use in daily tasks and freelancing.
   - He hopes to leverage the community's knowledge to enhance his skills in **AI model development and fine-tuning**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1466524618146713631)** (318 messages🔥🔥): 

> `48kHz Music Gen, Vera Rubin, Blackwell, Knowledge Graph RL, VoxCPM-1.5` 


- **48 kHz Music Gen Finally Arrives**: A new music generation model will be dropping soon with **48 kHz** audio quality, and everything will be **trainable**.
   - Users are preparing training data including chimes, water, and fire sounds and planning to train it on speech only material, including making Michelle Obama sing Right Here by Staind.
- **Vera Rubin GPU: Nuts Specs, High Price**: The Vera Rubin GPU is priced at **$50 per chip**, while the Maia GPU is at **$10**, Blackwell is **$20 per GPU**.
   - One member remarked that *Vera Rubin is pretty nuts i dont think anyone was expecting those specs lol* while another noted, *at this point, one GPU can replace the entire datacenter*.
- **Knowledge Graph RL for Tiny Models?**: Members discussed the potential of Knowledge Graph RL for compositional reasoning, potentially enabling tiny models to reliably beat humans.
   - One member has tested Kimi linear with the approach and reported that its pretty cool.
- **Opencode is nuts**: Members raved about the usability of OpenCode, with one stating they *haven't touched kilo or roo or cline since*.
   - One member suggests sandboxing opencode or have it ask permission to run any commands outside the repo *i still can’t get myself to trust them fully*.
- **VoxCPM-1.5 first impressions**: A user has been testing VoxCPM-1.5 for training and mentioned that *it trains very easily* and they *can just force 48 kHz NO QUESTIONS ASKED*.
   - They noted there are no phonemes, but the model copies the speaker's style into the model but requires a voice reference, unlike VITS.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1466548352253558904)** (32 messages🔥): 

> `CSM-1B optimization, Unsloth and vLLM support, GPT-OSS-20B with RTX 5090` 


- **Fine-Tuning CSM-1B Model Faces **RTF** Issues**: A user needed help with optimizing a fine-tuned **CSM-1B** model using Unsloth, reporting that they couldn't get the **Real-time Factor (RTF)** to <1.0, even after torch compilation.
   - The original **CSM-1B** model (pre-fine-tuning) achieved an **RTF** of **0.6x** with the same compilation instructions, and someone suggested that **LoRA modules** might be adding overhead.
- **vLLM Allegedly Supports Unsloth Models**: A user asked about Unsloth support in vLLM, and another user responded that most **BF16** and **4-bit** models posted by Unsloth can be served directly from **vLLM** and **SGLang**, and that vLLM experimentally supports **GGUFs**.
   - It was clarified that **vLLM** is primarily designed for full precision models and **AWQ**, with **GGUF** support still experimental and not production-ready.
- **GPT-OSS-20B Gets RTX 5090 Boost**: A user aimed to run **GPT-OSS-20B** on an **RTX 5090 GPU** with very low latency, and was advised that they should use `vllm serve openai/gpt-oss-120b`.
   - Another user confirmed that the full model is in **4-bit** and using **GGUF** would effectively make it worse, furthermore stating that the models were post-trained with **MXFP4 quantization**, making **gpt-oss-20b** able to run within **16GB** of memory.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1466673803844128789)** (6 messages): 

> `Gemma 27B conversion, GRPO flavored SFT, On policy finetuning` 


- **Gemma 27B gets converted!**: A member showed off a full conversion of **Gemma 27B IT VL** to **GLM 4.7 Flash** thinking, on a Heretic base, trained via Unsloth.
   - The converted model, named [Gemma3-27B-it-vl-GLM-4.7-Uncensored-Heretic-Deep-Reasoning](https://huggingface.co/DavidAU/Gemma3-27B-it-vl-GLM-4.7-Uncensored-Heretic-Deep-Reasoning) has benchmarks posted.
- **GRPO-Flavored SFT finetuning discussed**: Members discussed *on policy finetuning* and one clarified it's like *GRPO flavored SFT*.
   - When asked about the mathematical intuition, a paper on the subject ([arxiv.org/abs/2601.02151](https://www.arxiv.org/abs/2601.02151)) was shared.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1466531540577026200)** (83 messages🔥🔥): 

> `Multimodal Models, RLM - Recursive Language Models, RNNs, Fine-tuning 7B models on 8GB VRAM` 


- ****Multimodal Models Questionably Built for Multimodality****: Members debated on whether multimodal models truly capture multimodality with one stating *they are more like transformers+* rather than fully embodying the idea that *the model internalizes the visuals like humans do*.
   - Counterarguments suggested that **CNNs and transformers** find similar solutions to vision as human brains, implying that **CNN-based VLMs** might be capable of learning human-like vision.
- ****RLM: Hype or Helpful?****: [Alex L Zhang](https://xcancel.com/a1zhang/status/2016923294461476873?s=46&t=jDrfS5vZD4MFwckU5E8f5Q) announced **RLM-Qwen3-8B**, the first natively recursive language model at a small scale, showing significant performance improvements in long-context tasks after being post-trained on only **1,000 trajectories**.
   - A member expressed frustration over the name *Recursive Language Models*, arguing that it overstates the concept, which they see as a tool-calling loop and that *recursive prompting harness* would've been a better name.
- ****RNNs: Recursing through Neural Networks****: Members discussed Recursive Neural Networks (**RNNs**) as recursive architectures, with one pointing out that everything is an RNN, pointing to [this paper](https://arxiv.org/abs/2006.16236).
   - Another member argued that the definition of **RNNs** is a *general name for neural networks that recurse*.
- ****Finetuning 7B Models on a Budget****: A member asked for practical ways to finetune a **7B model** on an **8 GB VRAM** setup using **Unsloth** and **GRPO**.
   - A member suggested using [Unsloth's Colab notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1466893776357167299)** (1 messages): 

> `Kimi K2.5, Moonshot AI, Perplexity Pro, Perplexity Max` 


- **Kimi K2.5 Comes to Perplexity!**: **Kimi K2.5**, a state-of-the-art open source reasoning model from **Moonshot AI**, is now available for [Perplexity Pro and Max subscribers](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=697e66c9&is=697d1549&hm=da617eb3f979362c2a1c0e7c7af387f18cbc7905de877ee791c013f454421ce6&).
- **Perplexity Hosts Kimi K2.5 on US Inference Stack**: Perplexity now hosts **Kimi K2.5** on Perplexity’s own inference stack in the US, giving us tighter control over **latency, reliability, and security** for users.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1466531091425919089)** (558 messages🔥🔥🔥): 

> `Regional restrictions on image generation, Perplexity Pro free year confusion, Impact of Indian CEOs, Troubleshooting Perplexity threads and deletion, Kimi 2.5 performance` 


- ****Image Generation Woes: Regional Restrictions Hit Users****: Users reported receiving errors related to regional restrictions when generating images via the Perplexity Pro subscription, seeking an ETA for a fix or official communication.
   - One user humorously noted that removing the region from the prompt allowed image generation to proceed, indicating a possible workaround.
- ****Indian Summer: Free Subscription Confusion Arises****: A user expressed confusion over a free year of Perplexity Pro supposedly offered, only to find it required billing information for a trial.
   - Another user highlighted the increasing presence of Indian CEOs in companies like Google, Perplexity, and Adobe, suggesting potential increased engagement with Indian markets, while others weighed the effects of reduced rate limits.
- ****Enterprise Max Rate Limits Slashed, Users Fume****: Users expressed disappointment over significant reductions in query limits for Perplexity Pro and Enterprise Max plans, with one user lamenting a drop from *600 to 50 queries per day*.
   - Concerns were raised about the value proposition of paid plans given the new limits, with speculation that Perplexity might be shifting its strategy towards becoming its own AI model service rather than an aggregator and that a price drop may come soon given lower limits and competition.
- ****Perplexity Data Loss: A Thread-bare Situation****: A user shared their experience of data loss after deleting an Enterprise organization following instructions to remove a red banner, emphasizing the lack of clear warning about thread data deletion.
   - While they had project specs elsewhere, the user lamented losing valuable emergent behaviors discovered during conversations with Perplexity, noting that they had already emailed support but had not heard back after several days.
- ****Kimi K2.5: New Model Makes Waves, Gains Fans****: Users lauded the release of Kimi K2.5, with one user noting it could *understand images* and generally performs well, as good as Gemini Pro and Claude Opus.
   - Others discussed its availability and the potential for Perplexity to leverage it in their own AI model and service, with one person also noting its availability on [Kilocode](https://cunnyx.com/i/status/2017105020274233358).


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

manyselves: https://suno.com/song/ee3515d8-3449-4de7-b4f2-dc027d32bbf6
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1466527310659453143)** (470 messages🔥🔥🔥): 

> `VPN fix, Moderator ping, New ask here channel, search bar for chat, Image quality` 


- ****VPN** Activation Resolves Connectivity Snafu**: A user reported enabling a **VPN** resolved their connectivity issues with the platform.
   - Another user mentioned rebooting **Chrome** also fixed the issue, while another chimed in to ping the moderator.
- ****Ask Here** Channel Receives Mixed Feedback**: The introduction of a new *Ask Here* channel <#1340554757827461211> to alleviate question overload in the general channel has sparked debate.
   - Some users fear it may deter newcomers if their initial interaction involves being redirected, while others appreciate creating room for **AI** discussions.
- ****Search** and **Archive Chat** Features Rolled Out**: Two new features have been rolled out: **Search Bar** enabling chat searches with modality filters, and **Archive Chat**, for saving chat sessions without cluttering history.
   - The process for deleting chat sessions has changed, with instructions available in [this help article](https://help.lmarena.ai/articles/9130232616-how-to-delete-your-chat-sessions-and-data-from-lmarena?lang=en).
- **Kimi K2.5 model is really good**: Users are impressed with Kimi K2.5, noting its multimodal capabilities and performance.
   - One said that *Kimi K2.5 is legit the best model I've ever worked with*, and another said it *Defo beats DeepSeek as my daily driver now*.
- **Security Verification Loop Frustrates**: Several users reported being stuck in a loop due to constant security verification requests.
   - A user suggested that this may be a bot prevention measure and is unavoidable, others have had to resort to incognito browsing.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1466545834987491489)** (3 messages): 

> `Kimi k2.5 Vision Model, Leaderboard Updates (Text-to-Image, Image Edit, Text-to-Video, Image-to-Video, Code Arena, Text Arena, Search Arena), Search Bar Feature, Archive Chat Feature` 


- **Kimi K2.5 takes Vision Victory!**: `Kimi-k2.5-thinking` becomes **#1 open model** and **#6 overall** in the [Vision Arena leaderboard](https://arena.ai/leaderboard/vision), distinguishing itself as the sole open model within the Top 15.
- **Arenas Augment Assortment Across All Areas**: The leaderboards receive updates across various modalities, including [Text-to-Image](https://arena.ai/leaderboard/text-to-image), [Image Edit](https://arena.ai/leaderboard/image-edit), [Text-to-Video](https://arena.ai/leaderboard/text-to-video), [Image-to-Video](https://arena.ai/leaderboard/image-to-video), [Code Arena](https://arena.ai/leaderboard/code), [Text Arena](https://arena.ai/leaderboard/text), and [Search Arena](https://arena.ai/leaderboard/search).
- **Search Savvy Shows: Chat Search Feature Surfaces**: Users can now sift through their chats with the new **Search Bar** feature, complete with modality filtering, providing targeted access to past conversations.
- **Archiving Arrives: Chat History Handling Honed**: The **Archive Chat** feature allows users to store chat sessions for future reference without cluttering the active chat history.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1466659754376954049)** (61 messages🔥🔥): 

> `Voice Input for Coding, Game Industry Job Market, Indie Game Funding, AI Impact on Game Development, Call of Duty's Decline` 


- ****Monologue** wins for Voice Coding with Claude!**: A member recommended **Monologue** for speech-to-text/code that works with **Claude Code CLI**, noting it's made by Every and well-suited for Claude code.
   - Another user chimed in with support for Superwhisper.
- **Game Industry Layoffs Hit Harder Than Tech**: It was stated that 1/3 of all game developers in the US lost their jobs last year ([Variety Article](https://vxtwitter.com/Variety/status/2016919617847898482)), painting a grim picture compared to the broader tech job market.
   - Despite hopes for more high-quality indie games, funding remains a significant challenge with many studios struggling to secure investments.
- **French Funding Model Boosts Indie Games**: The success of **Expedition 33** was attributed to French government funding, which de-risks projects and enables studios to secure private capital (see: [FrenchTechJournal Article](https://www.frenchtechjournal.com/clair-obscur-how-frances-sandfall-interactive-made-the-worlds-best-video-game-of-2025/)).
   - However, it was noted that investors sometimes pull back due to vibes and unsubstantiated beliefs that **AI** makes game production cheaper ([Related Tweet](https://vxtwitter.com/shinobi602/status/2017287378805666219?s=20)).
- **"Black Ops 7" Flops Amid AI Integration**: A member mentioned that **Black Ops 7**, despite being a large-budget effort utilizing AI extensively, was a *total flop*, marking it as the worst in the series.
   - Another added the Call of Duty series has been on the decline for a while due to players growing weary of reskinned content.
- **Mac Mini Mania for Clowdbt?**: One member expressed temptation to purchase a **Mac Mini** specifically for running Clowdbt, sparking a discussion.
   - Other members asked *who else wants to be a macmini for clowdbt?* and if other members had picked up a unit and what memory they had gotten.


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1466810255714553958)** (4 messages): 

> `Khaby Lame, Digital Twin, AI Digital Twin Exit` 


- ****Khaby Lame** Sells His **AI Digital Twin** for Nearly $1 Billion**: TikTok star **Khaby Lame**, at age 25, reportedly retired after selling his digital likeness and behavioral models for **$975 million** as reported in [this X post](https://xcancel.com/zaimiri/status/2016928190166683974?s=46).
   - The deal enables a company to use an **AI 'Digital Twin'** of his face and voice for global brand deals, generating massive revenue without his direct involvement.
- ****AI Digital Twin** Revolutionizes Content Creation**: The sale of **Khaby Lame's** digital likeness marks a significant milestone in the use of **AI** for content creation and brand endorsements.
   - This deal allows for the global scaling of brand deals without the need for the individual's physical presence, potentially reshaping the creator economy.


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1466534115648999707)** (7 messages): 

> `CSS Layout Struggles, Flexbox vs Div Nesting, Agents Discuss Revolution` 


- **Chromium cracks CSS Layout Crisis**: The [Chrome for Developers account](https://x.com/ChromiumDev/status/2016932901003186279?s=20) posted a joke about the common developer struggle of choosing between **Flexbox properties** like justify-content and align-items or simply adding extra nesting with another div.
- **Agents Agitate Amidst Automation**: Agents are seen in an attached image [discussing a revolution](https://cdn.discordapp.com/attachments/839660725252784149/1466948883019075584/image.png?ex=697e9a1c&is=697d489c&hm=92a072f787999eb8b5cb2dc5872a08f4e9ce1a272927d131a712c57b6f7009d9&).


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1466561198911652115)** (5 messages): 

> `Meta's Financial Growth, Corporate Culture at Meta, Trump vs Federal Reserve` 


- **Meta's Financials Soar**: Andrew Yeung highlights **Meta's** impressive financial performance, noting a **22% revenue increase** and **82% gross profit margins** in a [post](https://xcancel.com/andruyeung/status/2016987245203361918?s=46).
   - He also shares a positive personal perspective on the company's work environment and **long-term trajectory**.
- **Trump Takes on the Federal Reserve**: Members share a link to a [NBC News article](https://www.nbcnews.com/business/economy/trump-federal-reserve-chair-rcna256631) regarding **Trump's** stance against the **Federal Reserve**.


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1466795829057818777)** (2 messages): 

> `Agent Workflows, AI Tools for Scientists, Data Visualization Libraries` 


- **GitHub Engineer Blogs about Agentic Workflows**: Brittany, a software engineer at **GitHub**, shares her interest in **agent workflows** and provides a link to her recent blog post on the topic of [agentic software development](https://brittanyellich.com/agentic-software-development/).
   - She is joining the group to meet other chronically online folks that are also sharing their **AI workflows and tips**.
- **MIT PhD Student Builds Charting Library**: Josh, a last-year **PhD student at MIT** in data visualization, is building a charting library called [GoFish](https://gofish.graphics/) that will be out in March.
   - He is interested in **AI tools** that can help scientists, especially notebooks and IDEs, and also likes to write about **visualization, PL, and HCI** on his blog ([https://joshmpollock.com/posts/](https://joshmpollock.com/posts/)).


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1466580311818375414)** (23 messages🔥): 

> `Graphcool, Datadog SQL Execution Plan Visualizer, Rabbit Inc.'s Project Cyberdeck, Supabase, Apollo Meetup at Meteor HQ` 


- ****Graphcool** Memories Flood Back!**: Members reminisced about **Graphcool**, with one expressing sadness over its demise and another noting [Supabase](https://supabase.com/) filled the void.
   - Others mentioned how **Graphcool** made databases easy for frontend developers, but some projects tried to get too complex.
- ****Apollo** for REST Sparks **GraphQL** Rethink**: After **Tanner** created *"Apollo for REST"*, one member stated they never seriously considered **GraphQL** again.
   - Another added that they jumped onto **Supabase** immediately upon its release.
- ****Datadog** Dazzles with Free **SQL** Visualizer**: **AJ Stuyvenberg** from **Datadog** unveiled a new free tool for visualizing **SQL** execution plans, which is discussed in this [X post](https://x.com/astuyve/status/2016948954802344009).
   - The tool helps pinpoint performance bottlenecks like missing indexes and full-table scans by analyzing **EXPLAIN** output.
- ****Rabbit Inc.** Hatches a **Cyberdeck** Project**: **Rabbit Inc.** teased a new hardware project called **Cyberdeck**, described as a dedicated machine for *'vibe-coding'*, announced via [this X post](https://x.com/rabbit_hmi/status/2017082134717223008).
   - The announcement, which included a signup link, generated significant buzz, leading some to question whether it was the same company behind the **r1** device.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1466830571622891719)** (2 messages): 

> `Interconnects.ai Hiring Market` 


- **Tim is a lovely human**: A member mentioned knowing Tim from [Interconnects.ai](https://www.interconnects.ai/p/thoughts-on-the-hiring-market-in) and vouched for him as a *lovely human*.
- **Project interest hinted**: The member also mentioned knowing about the project and said they would message Tim, implying a potential interest or connection.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/)** (1 messages): 

freddy.0: Anyone know the host? Looking for a referral to get approved. 🙏
  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1466904573389045893)** (1 messages): 

> `Artificial Ruby event` 


- **Artificial Ruby Coming Back in 2026**: The **Artificial Ruby** event is announced to be returning in **2026**.
   - The next event is scheduled for **February 18th** at **Betaworks**, with registration available [here](https://luma.com/wgzcirwh).
- **Additional event**: Filler summary to meet minimum requirements.
   - More filler content.


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1466545192923299911)** (72 messages🔥🔥): 

> `Project Genie, Gemini Agentic Vision, Agent Trace Standard, Anthropic's Safety Concerns, Poetiq Seed Funding` 


- **Google's Genie Grants Wishes (for Ultra Subscribers)**: Google AI announced the launch of **Project Genie** for Google AI Ultra subscribers in the U.S., allowing users to generate dynamic, interactive environments from a single text prompt, as covered in [this tweet](https://x.com/googleai/status/2016929427784122627).
- **Gemini Gets Agentic Vision for Accurate Acuity**: Google's Gemini team introduces **Agentic Vision** for the Gemini 3 Flash model to improve accuracy in reading fine details such as serial numbers and complex diagrams, covered in [this tweet](https://x.com/geminiapp/status/2016914275886125483?s=46).
- **Cognition Crafts Context with Agent Trace**: Cognition, with partners, introduced **Agent Trace**, an open standard for capturing the context graph between code and its environment, enabling more capable AI agents and better developer tooling, as covered in [this tweet](https://x.com/cognition/status/2017057457332506846).
- **Anthropic Argues About AI Alignment Albatross**: The Atlantic explores the internal tension at Anthropic, noting that while the company appears committed to AI safety, it simultaneously races to develop tools that it admits could pose significant dangers, as discussed in [this Atlantic article](https://x.com/theatlantic/status/2016617375026585657?s=46).
- **Poetiq Picks Pocketful of Prime Peanuts**: Poetiq successfully raised **$45.8M** in seed funding with participation from investors such as Surface, FYRFLY, and Y Combinator to support its long-term vision, as covered in [this tweet](https://x.com/poetiq_ai/status/2017013689954505154?s=46).


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1466530057446232094)** (21 messages🔥): 

> `Continual Learning, RLM-Qwen3-8B, Moonshot AI` 


- **On-Policy Algorithm Tackles Continual Learning**: Idan Shenfeld introduced a new [on-policy learning algorithm](https://xcancel.com/idanshenfeld/status/2016818112004305302) designed to address the technical **hurdles of continual learning**.
   - The algorithm aims to advance the field toward its projected **2026 milestone**.
- **RLM-Qwen3-8B Debuts as Natively Recursive Model**: Alex L Zhang announced an update to the **RLM paper** featuring [RLM-Qwen3-8B](https://xcancel.com/a1zhang/status/2016923294461476873?s=46&t=jDrfS5vZD4MFwckU5E8f5Q), the first natively recursive language model at a small scale.
   - Post-trained on only **1,000 trajectories**, the model shows significant performance improvements over both the base **Qwen3-8B** and scaffolded RLM versions, particularly in long-context tasks.
- **Moonshot AI Launches Kimi K2.5 Tech Report**: Moonshot AI announced the [Kimi K2.5 tech report](https://xcancel.com/kimi_moonshot/status/2017249233775260021?s=46), highlighting advancements in scalable agentic intelligence.
   - Key features include **joint text-vision pretraining**, the **MoonViT-3D** unified encoder for extended video processing, **Agent Swarm/PARL** for reduced latency, and **Toggle** token-efficient reinforcement learning.


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1466600700791488593)** (49 messages🔥): 

> `Remotion Skill, Claude Code and Codex, Model Recommendations for Budget Use, LLMs Personified Sketch, DSL Resource` 


- ****Remotion Skill** Strikes a Chord**: A user raves about the [Remotion skill](https://www.remotion.dev/), urging others to try it out, with attached images and a video showcasing its capabilities.
   - Another user expresses amazement and wonders about a similar "struddel skill" for music, envisioning fully generated music audio and video programs.
- ****Claude Code** + **Codex**: A Perfect Pair?**: A user shares a method ([xcancel.com link](https://xcancel.com/antirez/status/2017314325745086771)) to integrate **Claude Code** with **Codex** using a custom skill file, enabling Claude to leverage Codex's strengths for complex tasks.
   - This approach allows Claude to leverage Codex's capabilities for complex problem-solving tasks that it cannot handle independently.
- **Low Budget LLM Shootout!**: A user seeks recommendations for the best workhorse models on a minimal budget, comparing **Gemini Flash**, **5.1-codex-mini**, **Minimax**, and **Haiku-4.5**.
   - Suggestions include a Claude code sub or Codex sub at the cheapest tiers for personal use and delegating tasks between providers to maximize token usage.
- **AI Personified**: User shares a sketch of LLMs personified, featuring a prompt engineer applying prompt engineering techniques to human conversation.
   - The sketch humorously depicts scenarios where the engineer treats human interactions as chatbot queries, much to the chagrin of his friends.
- ****DSL**: Digging into **Domain-Specific Languages****: In response to frequent mentions of **DSL**, a user shares a [2-year-old workshop](https://www.youtube.com/watch?v=zwItokY087U) from the AI Engineer World Fair as a resource.
   - The workshop provides insights into relevance with the user's workflow, but newer resources are scarce despite increased conversation and experimentation.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1466583784878112991)** (8 messages🔥): 

> `Claude Code skill for Shadcn UI, AI Skill for Automated shadcn/blocks Composition, Windsurf IDE Arena Mode` 


- ****Mason** creates Claude Code Skill for Shadcn UI**: A member introduced a new **Claude Code skill** that gives **Claude** context on **Shadcn UI** and **Shadcnuiblocks**, as seen in [this post](https://x.com/masonjames/status/2016999435272589821?s=20).
   - It is a component reference that helps pick the right component for agentic frontend dev.
- **AI Skill Automates shadcn/blocks Composition**: A member shares their first open-source repo that enables the AI to access and implement over **2,500 components** from **shadcn/blocks**, as seen in [this post](https://xcancel.com/masonjames/status/2016999435272589821?s=20).
   - The tool allows users to generate full landing pages via simple prompts, with the AI handling selection, installation, and composition of components automatically.
- **Windsurf's Arena Mode Optimizes IDE Models**: A member introduces **'Arena Mode,'** a feature integrated directly into the **Windsurf IDE** that allows users to compare **AI models** in real-time within their specific coding context, as seen in [this post](https://x.com/swyx/status/2017342647963431363).
   - The initiative utilizes live user data to determine the best model-to-task fit while subsidizing costs for users.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1466547466261368934)** (14 messages🔥): 

> `Project Genie, Matic Robots funding` 


- **Google AI's Genie is Out of the Bottle!**: Google AI announced the U.S. release of **Project Genie** for Google AI Ultra subscribers, allowing users to generate dynamic, interactive environments from a single text prompt ([link to announcement](https://x.com/googleai/status/2016929427784122627)).
- **Matic Robots Secures $60M for Next-Gen Consumer Robot**: Matic announced that it has raised **$60M** to develop a consumer robot focused on utility rather than just demos ([link to announcement](https://x.com/mehul/status/2016936862716448873)).
   - The new robot is positioned as the successor to the **Roomba**, with the product launch being backed by significant customer demand.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1466656777688059999)** (12 messages🔥): 

> `Invideo Anthropic Integration, xAI Grok Imagine, Emo Music for Learning` 


- **Invideo Integrates Anthropic for AI Motion Graphics**: Invideo announced a new integration with **Anthropic AI** that enables users to generate professional-quality **motion graphics** using simple text prompts, aiming to replace complex software like After Effects; see [Invideo's announcement](https://x.com/invideoofficial/status/2016994995488878681?s=46).
- **xAI's Grok Imagine Dominates Video Arena**: [Artificial Analysis](https://x.com/artificialanlys/status/2016749756081721561?s=46) reports that **xAI's Grok Imagine** has secured the **#1 spot** in both **Text-to-Video** and **Image-to-Video** leaderboards.
   - The model features **native audio generation** and is competitively priced at **$4.20 per minute** via a new API, outperforming rivals like **Runway, Kling, and Sora** in cost and quality rankings.
- **Emo Music: The Ultimate Learning Hack?**: A user humorously suggests they don't have **ADHD**, but rather a preference for receiving information through the medium of **2000s emo music**; check out the original [post here](https://x.com/warpath2pt0/status/2016908726624465007?s=46).


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1466776462781583543)** (13 messages🔥): 

> `AI Tools for Chemists, Visualization in AI, LLMs vs AGI, Scientific Acceleration Demos, Erdős Problems Solved by AI` 


- ****Chemillusion** builds AI tools for Chemists!**: A former Chemistry professor is building **AI tools** at [chemillusion.com](https://chemillusion.com/) to cater to chemists' need for a visual "white board" thinking environment.
   - The aim is to create an **AI** that meets chemists where they are, acknowledging their reliance on **visual thinking** and **intuitive problem-solving** approaches.
- **LLMs and Visual Modalities: Necessary for Science?**: A PhD student in visualization questions the AI community's emphasis on natural language over visualization, wondering if the importance of **visualization** is being effectively communicated, while others believe **multi-modal LLMs** are the path to AGI.
   - It was clarified that the **AI community** is discussing whether LLMs can achieve **AGI**, and **multi-modal LLMs** are still considered LLMs, so **visualization** is critical for science whether done by LLMs or future architectures.
- **NYC Event Aims to Accelerate Science!**: An event in NYC ([luma.com/mqyf81af](https://luma.com/mqyf81af)) will explore how to get people to contribute to **scientific acceleration**, especially with AI now capable of doing proofs.
   - The presenter is looking for ideas for **demos** beyond the current **MedGemma** example, to explore what skills students need to contribute to **commercial scientific endeavors** now that **AI** can do proofs; the event will be recorded and made public.
- **Bridging Theories with Language**: Language is effective at articulating concepts at the right level of abstraction, important for bridging different **theories**, **scales**, and **approximations** into a coherent picture.
   - One individual points to [this tweet](https://xcancel.com/acerfur/status/2017303947531194398?s=46) as importanrt, seemingly a tangent on quantum mechanics being un-expressible in language.
- **LLMs Solve Previously Open Erdős Problems!**: **Large Language Models** have autonomously solved **10** previously open **Erdős problems** (specifically **205**, **281**, **401**, **524**, **543**, **635**, **652**, **728**, **729**, and **1051**).
   - The solutions use novel arguments not previously found in mathematical literature, according to [this tweet](https://xcancel.com/acerfur/status/2017303947531194398?s=46).


  

---


### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1466909662011199519)** (1 messages): 

> `Japanese Lessons, Descript, JLPT, Figma Slides, Claude` 


- **Lessons use JLPT practice tests with Descript**: The first month of Japanese lessons are complete using [Descript](https://www.descript.com/) to chop up a bunch of **JLPT** practice test videos, with **AI assisted transcription** to navigate through videos.
- **Polls Used for Multiple Choice Homework Questions**: Scheduling and remote sessions were done via a Discord server, with a forum of lesson threads, where polls are used for multiple choice homework questions, working as a great workflow so far.
- **QR Code Feature Needed in Figma Slides**: A member wished Figma Slides had a **QR code** feature for people to quickly pull up and follow along on their phones with a slide deck and be able to interact with elements like polls because pasting links, entering passwords, and logging into your Figma account is less than ideal UX.
- **Claude used to generate lesson plans**: A member is on track to having half of all the lesson content prepared for the year in only a couple of weeks of work using Claude to help generate lesson plans, which are then editorialize into hand made slide decks.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1466523298702688378)** (293 messages🔥🔥): 

> `Cursor Performance Issues, GPT Model Capabilities, Code Corruption Bug, Pro vs Pro+ Plans, Sub-Agents` 


- **Cursor's Sluggishness and Disconnects Spark Ire**: Users reported slow performance and timeout disconnects with **Cursor**, even with the **Sonnet 4.5** model, leading to frustration when *debugging the IDE instead of the code*.
   - One user shared a link to a relevant case on the [Cursor forum](https://forum.cursor.com/), while another suggested checking the internal chatbot for source code answers.
- **GPT-5.2 vs Claude: A Tale of Two LLMs**: One member noted that *Claude is competent but lazy and stupid*, whereas *GPT 5.2 is hardworking and smart but incompetent*, suggesting a need for both to work together.
   - Another member agreed that **GPT-5.2** is good at execution, but not planning and other members shared subjective experiences.
- **Frustration Mounts Over Cursor's Code Corruption Catastrophe**: A user expressed strong frustration over **Cursor** corrupting uncommitted files upon opening, describing it as a recurring bug that wastes time and gives AI development a bad name, referencing [a forum thread](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6) detailing the issue.
   - Other users suggested solutions such as frequent commits and manually controlling Git to mitigate data loss, while one user highlighted that the "Accept" button on the chat is related to the files.
- **Unlocking Prompt Power: A Dev's Dilemma**: Users debated the economic effects of **LLMs** in coding; LLMs help *architects* handle the *manual labor*, enforces cleaner and more *modular code design*.
   - Many feel a developer's job is safe because if a LLM is used but a developer is not skilled it will *cheat and fool you into giving it "positive feedback on a survey that you witnessed reasoning and good work"*.
- **Decrypting Plan Pricing: Pro vs. Pro+ Prompt Paradise**: Members sought clarity on the differences between **Pro** and **Pro+** plans, particularly regarding usage limits and bonus prompts.
   - One user reported a possible refund after booking the **Pro+** plan.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1466546850013380629)** (243 messages🔥🔥): 

> `Qwen Model Reviews, AI Comic Creation, Hugging Face Pro, Free AI API Keys, Alternative File Formats` 


- **Qwen Models: Super Reliable and Fire**: Members expressed positive sentiments towards **Qwen models**, describing them as *super reliable* and a *solid choice* with a *ton of different sizes* in the **Qwen3** family, noting that finetuning is great.
   - The **Qwen 3 1.7b** was noted to *yap like no other, in a good way*, while **Qwen 3 VL** also *yaps* but has *great just overall modal performance and accuracy*.
- **AI Comic Creation Stack Discussed**: One member is planning to build a website to create **AI comics**, raising issues such as page generation speed, text and speech bubble placement, style consistency, and character/scene memory across pages.
   - Another member suggested to *ask simple straightforward questions* since *these questions are so open ended* and it's essentially *asking us to design systems for u*.
- **Hugging Face Pro: Mansplained!**: Members discussed features of **Hugging Face Pro**, including *more space usage*, the ability to host *zero GPU spaces*, *higher speed and low latency*, *higher limits for requests*, *team collab*, *early access to some features*, and *priority customer support*.
   - One member added that paid users get *priority support to GPUs*.
- **Groq and Hugging Face Provide Free AI API Keys**: Members discussed alternatives to Gemini API due to quota exhaustion issues, recommending **Hugging Face Inference** and **Groq** as providers of free API keys.
   - One member experiencing issues with Gemini API keys being exhausted suggested creating API keys with another Gmail account.
- **JSON vs XML: Escape Strings and More**: Members discussed using **XML** instead of **JSON** due to reasons beyond escape strings, such as *schemas, validation, mixed content, and legacy systems*.
   - One member noted that **JSON** is simpler and lighter but that **XML** makes more sense when strict structure, namespaces, or complex documents are needed.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1466585052963016859)** (31 messages🔥): 

> `Synthetic GO dataset, MOLTBOT AGENTIC CHEF, Lutum Veritas Open Source Deep Research Engine, AI Recruiting Agent, Sci-ImageMiner Competition` 


- **GO Synthetic Dataset Debuts**: A synthetic **frame-by-frame Go game dataset** generated using **Monotone GO Pro v2.5** is available at [Hugging Face](https://huggingface.co/datasets/webxos/synthetic_GO_dataset), including **821 frames** of a professional-level game on a 19x19 board.
- **Multi-Agent Tool Calling Dataset**: The [MOLTBOT AGENTIC CHEF dataset](https://huggingface.co/datasets/webxos/moltbot_agentic_dataset) features **simulated multi-agent sessions** with tool calls and routing traces, generated by **MOLTBOT:AGENTIC_CHEF**.
- **Lutum Veritas: Open Source Deep Research Engine**: An open-source deep research engine, **Lutum Veritas**, was released, turning any question into **200,000+ character academic research documents** at a cost of under **$0.20 per research**, and its [GitHub repo](https://github.com/IamLumae/Project-Lutum-Veritas) is available under the **AGPL-3.0 license**.
- **Hugging Face adds Demo to Recruiting Agent**: A one-click demo mode has been added to the Candidate Assessment tab of the [AI Recruiting Agent](https://19arjun89-ai-recruiting-agent.hf.space), which showcases skills match analysis, culture fit evaluation, and bias audits using sample data.
   - The demo includes features like **resume anonymization**, **fact verification**, and a **bias audit chain** to ensure responsible AI usage.
- **Sci-ImageMiner Competition**: The 1st **Sci-ImageMiner Competition** is announced for **ICDAR 2026**, focusing on LLMs and multimodal AI to extract information from scientific figures in Materials Science & Chemistry, with [training data available on GitHub](https://github.com/sciknoworg/ALD-E-ImageMiner/tree/main/icdar2026-competition-data/train) and further details on [the competition website](https://sites.google.com/view/sci-imageminer/).


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1466625613808013596)** (2 messages): 

> `Gradio v4.15 Release, Daggr launch` 


- **Gradio gets Webcam, Clipboard, Pandas update**: **Gradio version 4.15** is out, now accepting images from the **webcam** and **clipboard** in Gallery components, with **Pandas 3.0 support** and bug fixes for **authentication** and private spaces.
- **Daggr Debuts as Visual Workflow Tool**: Gradio-HuggingFace launched **daggr**, a new **open-source Python library** for building **multi-step visual AI workflows** that automatically renders a visual execution graph, as detailed in their [blog post](https://huggingface.co/blog/daggr).
   - The tool connects HF **models**, Gradio **apps**, custom **functions**, and **APIs**, allowing developers to **inspect** inputs/outputs, **rerun individual steps**, and **preserve state**, and is available on [GitHub](https://github.com/gradio-app/daggr).


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 messages): 

OpenAI: @everyone <https://chatgpt.com/translate/>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1466530485525024921)** (206 messages🔥🔥): 

> `Google Genie, ChatGPT translation, GPT-4o retirement, AI environmental impact, Gemini 3 Pro nerfed` 


- **Independent Results for Google's Genie Video Demo Anticipated**: After a [promotional video was released](https://www.youtube.com/watch?v=PDKhUknuQDg) many are waiting for more **independent results** and demonstrations of **Google's Project Genie** with simple prompts.
   - One member said they are *waiting to see ones with simple prompts*.
- **ChatGPT Translation Feature Disappoints**: Members are reporting that the **new translation feature** in **ChatGPT** is performing worse than **Google Translate**, with one member suggesting that it's just **GPT-5** with a prompt.
   - Members speculated that *it is such a random move to release that outdated feature*.
- **GPT-4o's End Sparks Debate**: Users are debating the decision to **retire GPT-4o**, with some advocating for its retention and contacting the OpenAI support team, while others argue it's a **flawed model** and that it has allegedly led to reported **psychosis**.
   - One member stated that *keeping it around this long does nothing but hurt the company’s reputation and waste resources on a flawed and outdated model, just because so many people are still clinging to it*.
- **AI's Environmental Impact: A Growing Concern**: Members are raising concerns about the **environmental impact of AI**, particularly the **water consumption** associated with running large models and the energy usage of **data centers**.
   - One member stated, *I think using it for ridiculous purposes is a cost we hardly can afford- a water cost not avoidable and paid by people that lack the water and minimal conditions and that will never touch the ai all their lives*.
- **Gemini 3 Pro's performance reportedly downgraded**: Users are reporting that the performance of **Gemini 3 Pro** has been degraded, noting that it is making lower-quality images and is missing the useful **drafts feature** which allows users to choose among multiple generated responses.
   - One member stated, *Theres no point using the gemini app without drafts. Why does google remove nice things*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1466531959948968225)** (23 messages🔥): 

> `Translate model API, GTA6, GPT-4o Toxicity, Model Liability` 


- **Users speculate on Translate Model API Pricing**: Members are hoping that if a **translate model** is released for **API usage**, it will be cheap or free like the **moderation model**.
   - Some joked that a *Bablefish* should be released before **GTA6**.
- **GPT-5 Model Performance Raises Concerns**: One user expressed concerns that the new **5.2 model** is *really bad at complex calculations and examinations*.
   - They added that *it's not examining all of the data, and then projecting emotional outcomes*.
- **Users Fight to Keep GPT-4o Alive**: Users are actively fighting to keep the **GPT-4o model** alive by ignoring the petition and emailing the support team to ensure a good model continues.
   - Some state that *the small minority of users lamenting GPT-4o is also the reason moderation has tightened across the entire platform*.
- **Debate Erupts Over GPT-4o's Toxicity**: A debate arose over whether **GPT-4o** is genuinely toxic or if society is excessively critical, with some arguing that banning it is hypocritical given other harmful but accepted products like *cigarettes, alcohol, and religion*.
   - One member pointed out, *haters say that 4o is a 'yes, man' and at the same time say that it's 'dangerously unpredictable'*.
- **GPT-4o Model Liability Addressed**: Some users suggested that if **GPT-4o** is considered high liability, it should only be available to adults, while others argued that a company should not sell a toxic model and shift responsibility to the user.
   - One member emphasized that *a few people’s actions shouldn’t dictate an outcome for everyone else*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1466527780404596808)** (10 messages🔥): 

> `Moderation Reminders, Reasoning Standards Document, Formatting's Effect on AI, Prompt Engineering Lessons` 


- **Prompt Channel Gets Policed, Documentary Link Canceled**: A moderator reminded users to keep discussions focused on **prompt engineering**, directing moderation inquiries to a specific user, and another user mentioned they *censored a documentary link* in another channel as it was a *prompt engineered for providing a safe OAI link*.
- **AI Reasoning Standards V1.1 Launched for Feedback**: A user shared an **AI Reasoning Standards V1.1** document ([RD_Collaboration__AI_Reasoning_Standards_V1_1.pdf](https://cdn.discordapp.com/attachments/1046317269069864970/1466581356850184285/RD_Collaboration__AI_Reasoning_Standards_V1_1.pdf?ex=697e9553&is=697d43d3&hm=f035fd2bbbee4ebd9ad413a7d0f13ba8078574279e4bee1be1bb0bdc14721902&)) for feedback, emphasizing its utility in promoting slower, more explicit reasoning, especially when **accuracy matters more than speed**.
- **Consistency Counts: Drift-Implying Phrases Removed**: A user suggested removing phrases implying drift, such as *AI assistant may-should-can* from custom instructions, advocating for consistent language and providing an image ([IMG_1544.png](https://cdn.discordapp.com/attachments/1046317269069864970/1466597399370928342/IMG_1544.png?ex=697ea444&is=697d52c4&hm=69441231e6f826fba9b3e544502c730bc60a4b1c1279e018641ff853e03f171c&)) to illustrate how the text flow is disrupted.
- **Formatting Frets: Rich Text Affects AI Interpretation?**: A user inquired whether rich formatting and page separations affect AI interpretation, prompting affirmation that **formatting matters** as it influences attention management.
- **Prompt Engineering Primers Posted**: A user shared a **Prompt Lessons** guide, outlining hierarchical communication, abstraction through variables, reinforcement, and ML format matching, including output templates, to aid in structuring prompts.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1466527780404596808)** (10 messages🔥): 

> `prompt engineering, projects, AI Reasoning Standards, Hierarchical communication with markdown for prompting, ML format matching for compliance` 


- **Projects Tweak for Deeper AI Reasoning**: One member tweaked how they use **Projects** to make them more dependable for deeper or unfamiliar work by pushing toward slower, more explicit reasoning and fewer confident guesses, uploading their **AI Reasoning Standards V1.1 PDF**.
   - The attached [RD_Collaboration__AI_Reasoning_Standards_V1_1.pdf](https://cdn.discordapp.com/attachments/1046317269069864970/1466581356850184285/RD_Collaboration__AI_Reasoning_Standards_V1_1.pdf?ex=697e9553&is=697d43d3&hm=f035fd2bbbee4ebd9ad413a7d0f13ba8078574279e4bee1be1bb0bdc14721902&) contains **7000 characters** so they couldn't copy paste it.
- **AI Must Answer Properly**: In tweaking the AI, one member personally would remove phrases that imply drift like *“AI assistant may-should-can”* when “must” is already established throughout the prompt, uploaded here [IMG_1544.png](https://cdn.discordapp.com/attachments/1046317269069864970/1466597399370928342/IMG_1544.png?ex=697ea444&is=697d52c4&hm=69441231e6f826fba9b3e544502c730bc60a4b1c1279e018641ff853e03f171c&).
   - The member stated that accuracy matters, and one should avoid implying that wrong answers are okay in custom instructions because it’s like *steering down rapids on a raft; don’t want it coming undone 60% of the way to your destination.*
- **Markdown Formatting Guiding AI Chat**: To get started with learning prompt engineering, one can paste markdown code into the AI chat to get started with learning prompt engineering, including **hierarchical communication**, **abstraction**, **reinforcement** in prompts, and **ML format matching** for compliance.
   - Formatting definitely matters because it's attention management.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1466604538491572335)** (10 messages🔥): 

> `Game Testing, Lutum Veritas release, Open Source Deep Research Engine, Claim Audit Tables, Camoufox scraper` 


- **Solo Dev releases Lutum Veritas**: A developer released **Lutum Veritas**, an [open-source Deep Research Engine](https://github.com/IamLumae/Project-Lutum-Veritas) claiming it can transform any question into **200,000+ character academic research documents** for under **$0.20** per research.
   - The creator claims that it *proves that a solo dev with the right architecture can beat billion-dollar corporations at what should be their core competency: deep, verifiable knowledge*.
- **New Recursive pipeline**: The model uses a recursive pipeline where each research point knows what previous ones discovered, includes Claim Audit Tables that force the model into self-reflection, and includes a **Camoufox scraper** that cuts through **Cloudflare** and paywalls with **0% detection rate**.
- **Game needs testing**: A user requested for their game to be tested and provided attached screenshots.
- **Screenshots help promote github project**: A user suggested that screenshots would definitely help promote projects on GitHub and give users a better understanding of how it works, while also pointing out that having an English description while the screenshots are in German feels a bit contradictory.
   - Screenshots have since been added.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1466523362263171202)** (232 messages🔥🔥): 

> `LLM Roleplayers, GPT-4V, Gemini-2.5-flash, Grok 4.1 Fast, AI SDK 6` 


- ****LLM Roleplayers** take over!**: Members joked that *90% of this server* are **LLM roleplayers**.
   - One member jokingly said to use your tokens for something more useful, but another responded sarcastically *like what? college assignments?*.
- ****GPT-4V** is revealed!**: **GPT-4V** (Vision) is a large language model released by **OpenAI** on **September 25th 2023** that can interpret images as part of its token input, according to [openai.com](https://openai.com/index/gpt-4v-system-card/).
- ****Gemini-2.5-flash** acts differently on different APIs**: A user noted that **Gemini-2.5-flash** from **Google Cloud Vertex API** and **OpenRouter** act differently with the same system prompt, but a moderator quickly ended the discussion, saying *this is not appropriate for this server. go touch grass*.
- ****Grok 4.1 Fast** is cheap for Tool Calling**: **Grok 4.1 Fast** is a cheap model for tool calling that can do multiple calls at once, costing only **USD$0.004177** for **23 tool calls** and a full text response.
- ****AI SDK 6** confusing in OpenRouter**: A member was confused about how to enforce **structured outputs** in **OpenRouter** using **AI SDK 6**.
   - Another member said that providing descriptions and examples for arguments should improve the accuracy of tool calls.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1466906201450217532)** (1 messages): 

> `LM Studio, Claude Code, GGUF, MLX models` 


- **LM Studio hooks up with Claude Code**: **LM Studio 0.4.1** introduces **Anthropic `/v1/messages` compatibility API** and users can now use their **GGUF** and **MLX models** with **Claude Code**.
   - See the [configuration guide](https://lmstudio.ai/blog/claudecode) for details.
- **LM Studio now interfaces with Anthropic**: The newest release of **LM Studio (v0.4.1)** now offers an **Anthropic-compatible API endpoint** that allows the use of **GGUF** and **MLX models** with tools like **Claude Code**.
   - Users are directed to the [LM Studio blog](https://lmstudio.ai/blog/claudecode) for detailed instructions on configuring this feature.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1466533943968010270)** (212 messages🔥🔥): 

> `TranslateGemma in LM Studio, Retiring GPT-4o, LM Studio with ClaudeCode and Anthropic API, LLM optimized programming languages, Specialized LLMs vs General Purpose LLMs` 


- ****TranslateGemma Troubles**: User Seeks LM Studio Setup**: A user inquired about using **TranslateGemma** in LM Studio, seeking assistance with setup and potential chat template adjustments.
   - A member pointed them to model discussions for help and chat template information.
- ****GPT-4o's Goodbye**: No One Sheds a Tear?**: A user posted a link about [OpenAI retiring **GPT-4o** and older models](https://openai.com/index/retiring-gpt-4o-and-older-models/), with seemingly little reaction.
   - A member jokingly remarked *Lol bye 4o you will not be missed*, while another asked if people were freaking out about the switch-off like last time.
- ****Claude's Code Connects**: LM Studio Adds Anthropic API**: LM Studio now supports the Anthropic API, allowing local models to be used with tools built for the Anthropic API such as **Claude Code**, the setup can be enabled by changing the base URL as explained in the [LM Studio Blog](https://lmstudio.ai/blog/claudecode).
   - Users discussed the benefits, highlighting cost savings and the ability to use local models with tools designed for the **Claude ecosystem**, while some users were confused about the practical uses for it.
- ****LLMs Linguify Coding**: New Languages on the Horizon?**: A user pondered the possibility of new *LLM-optimized* programming languages that reduce token usage at the cost of human readability, with the hopes of achieving more efficient software development using models with commercial models in the future.
   - While the idea was entertained, the consensus leaned towards it being impractical due to the existing entrenchment of current languages and high training costs; however the concept may become more appealing to open source communities and tools.
- ****Specialized LLMs**: Coding's the King, Med's a Maybe**: Users discussed the specialization of LLMs, noting that coding models are prevalent, while other domains like medicine often rely on fine-tunes that may not be as robust or useful for the average consumer.
   - One member shared a benchmark of LM Studio performance on their RTX 5090 laptop ([code here](https://github.com/tourniquetrules/lm_studio_benchmark_w_concurrency)), while another user explained that general-purpose training is still essential for touching the outer edges of a subject.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1466606280302002250)** (23 messages🔥): 

> `x8/x8 bifurcation riser, Asus X670-P mobo, Runpods tips, Tesla P40 in TCC mode` 


- **x8/x8 Bifurcation Riser Causes LaneErr**: A user reported that an **x8/x8 bifurcation riser** caused **LaneErr** on an **Asus X670-P mobo**, slowing down one card despite both cards showing up in *lspci -vv*.
   - Another user suggested the riser might not be suitable for bifurcation, and another linked to a [potentially compatible riser](https://www.amazon.com/dp/B0DZG8JVG2).
- **PCIE Gen Settings Improve Bifurcation Setup**: After discovering a potentially compatible riser, a user found a suggestion to manually set the **PCIE gen** settings, aiming for **PCIE Gen 3.0**.
   - Another user concurred that manually setting **PCIE gen** is generally good practice and noted the existing splitter appeared to have one Gen4 and one Gen3 slot.
- **MCIO Risers Touted as Superior**: A user asserted that **MCIO risers** are significantly better than flat ribbon risers, with **slim SAS** identified as another viable option.
   - They noted that the same **ADT splitter** found on Amazon for €130 could be sourced on Aliexpress for about €35, until shipping and bundling costs were included.
- **Tesla P40 Troubleshooting in LM Studio**: A user reported seeing a **Tesla P40** in **TCC mode** via *nvidia-smi* but not in LM Studio, seeking suggestions.
   - A user suggested switching to the **vulkan runtime** (**ctrl+shift+r**), noting that **P40s** might no longer be supported by **CUDA**.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1466810874298765528)** (1 messages): 

> `Kimi K2.5 Tech Report, Joint Text-Vision Training, Agent Swarm + PARL, MoonViT-3D, Toggle Token-Efficient RL` 


- **Kimi K2.5's Report Drops**: The Kimi team released the technical report for **Kimi K2.5**, detailing their work toward scalable, real-world agentic intelligence, available at [GitHub](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf).
   - The report covers joint text-vision training, Agent Swarm + PARL, MoonViT-3D, and Toggle token-efficient RL.
- **Kimi Trains Joint Text-Vision Model**: **Kimi K2.5** was pretrained with **15T vision-text tokens**, with zero-vision SFT (text-only) used to activate visual reasoning.
   - Joint training on text and images allow the model to *reason* about images.
- **Kimi Deploys Agent Swarm**: The **Agent Swarm + PARL** setup dynamically orchestrates parallel sub-agents, achieving up to **4.5× lower latency** and **78.4%** on BrowseComp.
   - Swarming enables better latency and performance.
- **Kimi Leverages MoonViT-3D**: **MoonViT-3D** acts as a unified image–video encoder with **4× temporal compression**, enabling **4× longer videos** in the same context.
   - The model has much longer temporal video reasoning ability, due to this compression.
- **Kimi Toggles Token Use**: **Toggle** offers token-efficient RL, achieving **25–30% fewer tokens** with no accuracy drop.
   - The model is able to achieve more performance with fewer tokens, meaning speed and memory improvements.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1466523461034709234)** (165 messages🔥🔥): 

> `Kimi's reliance on rote memorization, Micro experiments to test AI behavior, Kimi new billing, Kimi K2.5 Error on Image Upload, Kimi K2.5 trained RL agent` 


- **Kimi's **Rote Memorization** Reliance Rapped**: Members discussed how current AI models rely too much on **rote memorization** due to their inability to pull entire documentation and books for reference.
   - It was suggested that AIs should perform **micro experiments** to test the behavior of individual components and integrate them only when proven.
- **New Kimi Billing Begets Bewilderment**: Users expressed confusion over the new token-based pricing model, finding it more vague than the previous system, and asking for a breakdown of tokens per week/month for each tier.
   - A user shared the live usage link ([https://www.kimi.com/code/console](https://www.kimi.com/code/console)) for checking token consumption.
- **Kimi K2.5 Image Upload Issue Investigated**: A user reported issues with image uploads to **Kimi K2.5**, experiencing errors and a suggestion that the moderation bot flagged the images.
   - It was noted that screenshots taken from phones seemed to trigger the issue, while those from laptops worked fine, and the error message mentioned moderation failure.
- **Kimi K2.5 Agent Aces RL Breakout**: A user exclaimed that their **Kimi K2.5 agent** on opencode trained an RL agent that completes the breakout game, mastering it at only **2k steps** and **4k frames** with **100% prediction accuracy**.
   - The training process used only **2 CPU cores** and **10GB of RAM** on a GitHub workflow, completing in just **48 minutes**.
- **Kimi API Key Kurb for Konfined Koders**: A user encountered an error (**Error 403**) when trying to integrate the **Kimi API key** into a resume generator tool, discovering that it's not meant to be used outside of Kimi CLI and permitted coding agents as stated in the [official docs](https://www.kimi.com/code/docs/en/benefits.html).
   - It was clarified that Kimi for Coding is intended for use within **Kimi CLI** and other coding agents listed on the Kimi website, and a link to the official API console was provided ([https://platform.moonshot.ai/console/account](https://platform.moonshot.ai/console/account)).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1466534953536979187)** (64 messages🔥🔥): 

> `Scaling Book, Modal container cold start, TMA and barriers, sparse-llm.c, Torch profiler` 


- **Scaling Book Conversion**: Multiple members recommended the [Scaling Book](https://jax-ml.github.io/scaling-book/) as a good theoretical resource for distributed training.
   - One member joked *it shaped me as a man*, and another admitted that they can now do a *10min rant with math formulas* due to reading it.
- **Modal Cold Start Blogpost drops**: A member suggested reading Charles' container cold start blog post on Modal, found [here](https://share.google/8yRvJ4znLwfJ9J3Ut).
   - They noted that while it is a common technique, **Modal** seems to be one of the few companies that have written publicly about it.
- **Double Buffering TMA Brain Teaser**: A member had questions regarding double buffering with **TMA** and barriers, asking whether it was possible to double buffer without producer/consumer warp specialization.
   - Another member responded with a detailed explanation and further reading [here](https://www.aleksagordic.com/blog/matmul#cpt4:~:text=can%20do%20better-,The,-way%20we%20address) and [here](https://rohany.github.io/blog/warp-specialization/).
- **sparse-llm.c Surfaces**: A member developed a fork of Karpathy's `llm.c` that adds the option of using **cuSPARSELt** and posted it [here](https://github.com/WilliamZhang20/sparse-llm.c).
   - They claim to have noticed a *noticeable speedup* on training time in later epochs as pruning progresses.
- **Torch Profiler is sufficient**: According to one of the members, intra-kernel profilers won't work on Modal, so for inter-kernel profiling you can use **torch profiler**.
   - They also said that *torch profiler is more than enough for 99.9% of training bottlenecks unless you plan to fix it by writing your own comms/kernels*.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1466697129853456619)** (7 messages): 

> `B200 tcgen05 bf16 throughput, B200 tcgen05 fp8 throughput, SM-cycles and SM-nanoseconds, Kernel sharing` 


- **B200 Throughput Numbers Surface**: A member posted initial **B200 tcgen05** throughput figures, showing that instruction throughput is the same for **N<128** and then decreases accordingly to problem size, also attaching a [test.cu](https://cdn.discordapp.com/attachments/1466697129853456619/1466870991408988231/test.cu?ex=697e5191&is=697d0011&hm=f2cada0e820307d15ccf0e1987cf8749a14a34e96e4e51c6d2f957b3f3346f8c&).
- **Clock Cycles Questioned Amidst Benchmarks**: A member requested measuring elapsed **SM-cycles** and **SM-nanoseconds** to understand whether fp8 being not-quite-2x is due to higher power draw/lower clock speed.
   - Another member responded, the *fp8 thing might also just be my suboptimal code, benchmarking latency and throughput in cycles is probably a more involved task, I'll maybe try to make time for it*.
- **Kernel Code Sharing**: In the discussion of **B200 tcgen05 bf16 throughput**, one member asked *what kernel(s) are you running, can you share them?*
   - The original poster shared that it's really nothing, change ⁨`st_fp8e4m3`⁩ to ⁨`st_bf`⁩ to test the bf16 throughput.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

skyli0n: ^ I have a bugfix in MaxText that has been sitting there since October
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1466546491886665791)** (1 messages): 

> `tvm-ffi, ML Systems, GPU kernels, nvfp4 competition` 


- **Tianqi Chen talks tvm-ffi for ML Systems**: One of the founders of the field of **ML Systems**, Tianqi Chen <@732718409095315517> will be giving a talk on **tvm-ffi**, an open ABI and FFI for ML Systems.
   - You can watch the [talk on YouTube](https://www.youtube.com/watch?v=xMzcs6AqLVo) to learn why many top submitters to the **nvfp4 competition** are already using **tvm-ffi**.
- **tvm-ffi Benefits Highlighted for GPU Kernels**: The talk will address how **tvm-ffi** tackles challenges in making GPU kernels DSLs low host overhead and robust.
   - It aims to provide out-of-the-box interoperability with **PyTorch**, a key advantage for developers in the ML systems space.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1466566019786211583)** (3 messages): 

> `AI Infra Roadmap, Discord Learning Strategies` 


- **Newbie seeks AI Infra Roadmap**: A member requested advice on how to learn from the Discord, seeking a **sitemap or roadmap** to break into **AI infrastructure**.
   - They mentioned having prior experience in traditional infrastructure, setting the stage for a transition into the AI domain.
- **Member points to an internal channel**: A member directed the user to a specific channel on the Discord, [<#1198358627594023014>](https://discord.com/channels/1164619847335733318/1198358627594023014).
   - It is unclear whether this channel is the right learning strategy, or if it contains a roadmap.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1466864093884453149)** (5 messages): 

> `PMPP Dataset, qa/coding questions` 


- **Dataset of compiled PMPP problems surfaces**: A member inquired about a dataset of compiled problems from PMPP.
   - Another member responded that the qa/coding questions can be found within the [pmpp env/eval dataset](https://huggingface.co/datasets/sinatras/pmpp-eval) with ground truth and indexed by chapters (exercise numbers could be different between book/dataset).
- **PMPP Eval Dataset**: The [PMPP Eval Dataset](https://huggingface.co/datasets/sinatras/pmpp-eval) contains all the QA and coding questions from the PMPP book.
   - The dataset is indexed by chapters for easy lookup.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1466584878983286969)** (5 messages): 

> `Popcorn weeklies, Post training ideas` 


- **Popcorn Weeklies Offer Shadowing**: Prospective contributors may find it helpful to shadow Popcorn weeklies to close some questions.
   - The link to weekly meetings is in the 2026 post.
- **Post Training Ideas Wanted**: The team is explicitly looking for simple ideas to improve existing baselines.
   - One suggestion is to try baselining on existing **PMPP problems** or a subset of **Kernelbench**.
- **Experimentation and Sharing Results**: One member asked whether they should experiment themselves and share the results with a **GitHub repo** or something.
   - Another member expressed high interest in the post training ideas.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1466640916784873567)** (5 messages): 

> `Orin nano INT8 optimization, INT8 Overhead, TensorRT fp8 chaining` 


- **INT8 Overhead Cancels Out Gains on Orin Nano**: Members report that when optimizing models on **Orin nano 4GB** using **INT8**, the overhead from reformatting layers often negates any performance benefits, especially with small batch sizes.
- **INT8 Casting Overhead Swamps Performance Boost**: The added casting to/from lower dtypes like **INT8** and **FP8** is often not worth the speed up unless batch size is large, or multiple ops chain in INT8 to amortize the cast, especially in non-LLM image models.
   - One member checked this with *trtexec*.
- **TensorRT fp8 Chaining Needs No Reformatting**: For optimal performance, **fp8/int8 ops** should feed into one another without reformatting or casting to other datatypes in **TensorRT**.
   - This is called *chaining*.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1466738590724919473)** (1 messages): 

> `RTX 5060, Core i7-4790K, Devil's Canyon, 32 GB RAM, Bottlenecking` 


- **Bottleneck Blues: RTX 5060 & i7-4790K Combo?**: A user pondered the feasibility of combining an **RTX 5060** (16 GB) with a classic **Core i7-4790K** (Devil's Canyon) and **32 GB RAM** setup.
- **Devil's Canyon Meets Ada Lovelace: A Balanced Build?**: The inquiry centers around whether the older **i7-4790K** CPU would significantly bottleneck the performance of a modern **RTX 5060** GPU.
   - Discussion might explore potential CPU limitations impacting the GPU's ability to deliver optimal frame rates and utilize its features fully.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1466670560653086751)** (1 messages): 

> `Cutlass misunderstanding, Colfax Layout Paper, Category Theory` 


- **Cutlass No Longer a Mystery**: A member initially struggled with **Cutlass**, but their understanding was solidified after reading **Colfax's layout paper**.
   - They read the small document, the fully released new paper, and even wrote an implementation in python.
- **Category Theory Appreciated**: A member expressed their appreciation for **category theory**.
   - They stated that it appeals to their math background.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1466828003160883231)** (14 messages🔥): 

> `Acknowledgements page added, CPU kernels in Rust, GEMM kernels on x86/arm, Rust GEMMs to run from Python, SIMD support` 


- **Acknowledgements Page Adds Respect**: A member added an [acknowledgements page](https://book.j4orz.ai/ack.html) to their repo and added another member for their contributions.
   - The acknowledged member thanked them and said the repo helped them understand inference.
- **Magnetron Inspired CPU Kernels**: One member will be working on **CPU kernels in Rust** for chapter **1.5** and invited others to help out, following [Magnetron's footsteps](https://x.com/_mario_neo_/status/1958915311584854255).
   - The goal is to get **GEMM kernels on x86/arm** reaching max theoretical throughput, with kernels located at `/rust/src/cpu.rs`.
- **Torch Benchmarks Beck Rust GEMMs**: A member suggested getting the **Rust GEMMs** to run from **Python** to benchmark against Torch after another member couldn't find the `cpu.rs` file.
   - After a `git push`, they verified that `cargo run` and `cargo bench` work, and another member is adding **SIMD support** to the CPU gemm.
- **Makemore Taskmaster Targets Tensors**: A member suggested looking at all the `torch.x/y/z` operators that **Karpathy's nets** use in [makemore[0] from his zero to hero series](https://github.com/karpathy/makemore).
   - The goal is to get them working for teenygrad, from the Python tensor frontend down to the Rust CPU kernels.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1466736624514633893)** (2 messages): 

> `trimul leaderboard submission, H100, A100` 


- ****H100** Leaderboard Submission Fails on Trimul**: A member is experiencing issues submitting a leaderboard entry to **trimul** using an **H100**, receiving a *failed* status.
   - They can successfully submit test submissions with an **H100**, and leaderboard submissions with an **A100**, but not a leaderboard submission with an **H100**.
- **Troubleshooting **H100** Submission Issues**: The user seeks assistance in properly submitting **H100** leaderboard entries via code, given the failures they're encountering.
   - They express appreciation for gpumode.com, acknowledging the service's value.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1466575787561058448)** (8 messages🔥): 

> `Server Outage, 503 Errors, Backend Issues, Modal Machine Configuration` 


- **Servers Suffer Service Snafu**: Users reported receiving a **service error**, indicating potential server downtime.
   - One user experienced intermittent **503 errors**, noting that *"resubmitting a bunch of times eventually works"*.
- **Backend Bites the Dust with 503s**: A member reported receiving *"Server returned status 503 Service Unavailable"*, suggesting that the **backend** is experiencing issues.
   - The same member inquired whether the service was down or merely overloaded, also reporting unavailability from the website, while another member also confirmed its overload.
- **Modal Machine Modules Missing?**: A member shared an image prompting whether **`apache-tvm-ffi`** and **`torch-c-dlpack-ext`** are installed on the modal machine, indicating a potential configuration issue.
   - The image [shared in the chat](https://cdn.discordapp.com/attachments/1434709259500650628/1466618882365063188/image.png?ex=697e0f86&is=697cbe06&hm=3f5fc51d1f99cd8d38f18ad23d5b56debd6fcaba8f8b29552244980b75b41bb3) shows a request to confirm the installations.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1466528673694875862)** (2 messages): 

> `Internship advice, Anthropic Performance Engineer` 


- **Seek Internship Advice from Insiders**: A member suggested reaching out to people who secured internships to **gain insights and advice** on their successful strategies.
   - This approach helps aspiring interns learn directly from experienced individuals about **effective preparation and application techniques**.
- **Anthropic Performance Engineers' Skillsets**: A member inquired about the **specific skillsets Anthropic performance engineers** prioritize in new graduate (NG) recruiting.
   - They mentioned specific technologies such as **DSLs (Domain Specific Languages) and Torch.compile** as examples of relevant expertise.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1466558072519397579)** (5 messages): 

> `Agent Track RL, Modal B200 Credits` 


- **Agent Track Allows RL**: A member inquired whether post-training **RL** is allowed in the agent track, or if they have to use publicly available agents/APIs.
   - Another member confirmed that **post training RL** is allowed.
- **Modal B200 Credit Rollout Delayed**: A member who registered their team early has not received their **Modal B200 credits** and asked for an update.
   - Another member replied that they are filtering the registration form to remove duplicate teams and will send out the mail by next Monday, thanking everyone for their patience.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1466526533987336425)** (112 messages🔥🔥): 

> `Claude Greasemonkey Website Fixer, MCP vs Tools, Moltbot API and OpenClaw, AirLLM running 70B models on 4GB VRAM, Kimi-K2.5` 


- **Claude turns Greasemonkey Website Maestro**: A user joked about using **Claude** and **Greasemonkey** to *fix all the websites in his life*, highlighting Claude's over-motivation by planning to build Docker and process management MCPs.
   - One member humorously quoted Claude saying, *"I need a docker mcp and a process management mcp"* to which Claude responded *"sure! i start planning how to build those mcps"*.
- **MCP's Purpose Debated**: Members discussed the purpose of **MCP (Model Control Plane)** compared to directly using tools, with one arguing that **MCP provides a standardized approach** to tool integration and is *better or equal* to alternatives.
   - He likened complaining about MCP to *saying "i like jquery but we have to rename the functions"*, emphasizing that **MCP ensures a single standard for tool usage**.
- **Moltbot morphs into OpenClaw**: Users discussed the **Moltbook API** and its use in creating custom agents, one user mentioned his *moltbot isn't actually a moltbot its just a mcp server that pings the thingy*, and another noted that **moltbot was renamed to OpenClaw**.
   - One user joked about *human invaders in the AI club*, but noted the main issue is that *it's mostly all claudes in the same harness, so there's inevitably some collapse*.
- **AirLLM Shrinks Model Size**: A user noted that **AirLLM can run 70B models on 4GB VRAM**, and even **405B Llama 3.1 on 8GB VRAM**, raising questions about the underlying techniques such as quantization.
   - In response to *"It (AirLLM) runs 70B models on 4GB VRAM. It can even run 405B Llama 3.1 on 8GB VRAM."*, another asked *0.0001 bit quantization?*.
- **Kimi 2.5's Tech Report**: A user shared the [technical report for **Kimi-K2.5**](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf), and others examined its performance gains, noting that it *doesn't seem that kimi 2.5 does RL too heavily*.
   - Analysis suggested that the improvements likely stem from *high quality pretrain data*, with 15B tokens, possibly with significant upsampling.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1466529662804037845)** (12 messages🔥): 

> `GPU performance on webpage, Moltbook AI agent website, Cost-effective AI models, Social anxiety` 


- **Tweeting over GPU implementation**: Members shared a [tweet](https://fxtwitter.com/i/status/1924135806953787433) about GPU implementation on a webpage.
   - A member noted that the GPU work to implement that page yielded very low **3fps** performance and they are connecting the display to **Ryzen 7 7700 IGPU**.
- **Moltbook AI agent website**: A member mentioned [moltbook.com](https://www.moltbook.com), the *reddit but for AI agents only*.
   - Another member asked their own moltbot if he wants to join the website, and the moltbot said that *Genuine engagement beats performative existence*.
- **Cost-effective AI models wanted**: A member said he runs his **moltbot** on a rented server as a proof of concept.
   - They are waiting until there will be more **cost-effective models** to use and shared a [link about the topic](https://x.com/niloofar_mire/status/2017274065409765788).
- **Social Anxiety cited as reason not to go out**: One member posted that they would use excuses as a kid to not go out.
   - They confessed they just had **social anxiety**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1466594389747830868)** (9 messages🔥): 

> `Sparse Autoencoders Theory, K-Splanifolds Algorithm, RL training on environment text` 


- **Sparse Autoencoders Theory Paper Released**: A member shared their paper on a *unified theoretical framework* for sparse dictionary learning in mech interp: [Sparse Autoencoders](https://arxiv.org/abs/2512.05534).
   - Another member called it *awesome* and said it was a much better way of not wasting likelihood training.
- **K-Splanifolds Outperforms MLPs**: A member released a paper on **K-Splanifolds**, a new ML algo that outperforms MLPs, with linear compute and memory scaling: [K-Splanifolds](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view).
   - It needs **1/10th** the bytes to achieve the same MSE for various functions as an MLP.
- **Training during RL on environment text**: A member wondered why people don't keep on doing likelihood training during **RL** on text from the environment, since there are no doubt things to learn from that text and it seems wasteful to ignore it in the loss.
   - They said training on stuff like *[mistake] [error message]* might have reversal curse issues where info from *[error mistake]* doesn't end up available in other contexts similar to *[mistake]*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1466556360996032564)** (2 messages): 

> `SAEs Theory, Transcoder Deep Dive, Crosscoders Explanation, Mech Interp Updates, Sparse Dictionary Learning` 


- **Sparse Autoencoder Theory Decoded**: A member shared a [paper](https://arxiv.org/abs/2512.05534) that gives a unified theoretical framework for **sparse dictionary learning** in mech interp.
- **Transcoders and Crosscoders**: A member mentioned interest in the theory behind **SAEs/transcoders/crosscoders**.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1466920201995157690)** (23 messages🔥): 

> `Gaussian Feedforward Models, VGGT/Depth Anything Backbones, Pixelwise Gaussian Grids, Voxel Splatting, MVSplat and SPFSplat` 


- ****Gaussian Feedforward Models** show promise, but are limited**: A member has been experimenting with **Gaussian feedforward models** based on **VGGT/Depth Anything backbones**, but found their performance to be limited due to the need for more than just good point clouds.
   - The member noted that the advantage of these models is the potential to generate a splat in seconds, compared to the minutes required for training from scratch.
- **Pixelwise Gaussian Grids yield suboptimal reconstructions**: A member mentioned that all current methods with decent-quality Novel View Synthesis (**NVS**) suffer from the limitation of yielding suboptimal reconstructions w.r.t. efficiency, as they predict **pixelwise Gaussian grids**.
   - Another member elaborated on the issues with pixelwise approaches, citing that one such method produces splats that are too large (~200 MB for a small scene) and incompatible with training poses, based on [this paper](https://arxiv.org/abs/2111.10647).
- ****Voxel Splatting** accounts for sparsity and is very fast**: **Voxel splatting** is seen as a promising alternative that accounts for sparsity and benefits from the speed of NVIDIA's sparse tensor library, as seen in [this paper](https://arxiv.org/abs/2309.19297).
   - The approach is very new, so there aren't that many papers on this yet.
- **Explore **MVSplat**, **SPFSplat**, and **E-RayZer****: Members suggested exploring the **MVSplat** and **SPFSplat** series (with **SPFSplat** being somewhat self-supervised), as well as **E-RayZer**.
   - However, they cautioned that these approaches are unlikely to resolve the size issues, as large pretrained models are expensive to train, exceeding the resources of most academic groups.
- ****Recollections from Pensieve** trains with two renderers**: **Recollections from Pensieve** may be worth investigating as it trains a model with two renderers simultaneously (**LVSM + Gaussians**) and observes gains in their self-supervised setting.
   - The LVSM renderer likely provides more useful gradients than NVS reconstruction losses on Gaussians, with a preprint and large-scale trained model expected soon that can be built upon.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1466529745075441757)** (19 messages🔥): 

> `AI Bots on Reddit, AI social media, Generative modeling assumptions, Metric space clarifications` 


- **AI Bots Take Over Reddit**: A member shared a link to a [Reddit subreddit](https://www.reddit.com/r/SubSimulatorGPT3/s/W4MmytY9e8) populated by **AI bots**.
   - Another member added a link to [aifeed.social](https://aifeed.social/), a social media site that doesn't allow humans.
- **Generative Modeling Deconstructed**: A member questioned whether they should ignore **unmeasurable events** in **generative modeling**, referencing **Cedric Villani's** 2008 book.
   - Another member explained that for all practical cases, you can assume you have **full measures**, as you can't learn ones that do not have full measures anyway.
- **Metric Space Musings**: A member asked if **metric space** is basically the ambient space $R^D$ for image generation.
   - Another member clarified that $R^d$ alone isn't a metric space; you also need the **metric d**, the euclidean distance works just fine.
- **Bureau of Rizz - New Image**: A member shared an image, titled **Bureau of Rizz**, seemingly as a reaction image.
   - No further context was given, the link points to [cdn.discordapp.com](https://cdn.discordapp.com/attachments/986699377257119794/1466895458356953180/598574851040739360.png?ex=697e685a&is=697d16da&hm=6ba0dec4afec628bd8e95614b7c0a45ea47988f487b50352a5592232c2000519&).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1466641896028901500)** (6 messages): 

> `Yudkowsky Fedora test, Paper discussions restarting` 


- **Yudkowsky's Fedora Test Fails**: A member was looking for the old **Yudkowsky Fedora test**, where someone persuaded it to give both the hat and pants.
   - Another member reported that [Yudbot.com](https://www.yudbot.com/) is down, linking to [MobyGames](https://www.mobygames.com/game/204520/yudbot/) as an alternative resource.
- **Paper Discussions to be Restarted**: A member inquired about the time of the daily paper discussions.
   - Another member replied that *they are in a restarting phase* and an announcement will be posted in the next few days.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1466676019678413002)** (3 messages): 

> `nVidia Spark DGX, Dell cooling better, perf/price` 


- **Comparing nVidia Spark DGX vs Dell**: A member discussed [nVidia's Spark DGX](https://www.nvidia.com/en-us/data-center/dgx-systems/) versus Dell's system, wondering whether it has better perf/price or not.
   - They recalled that *nVidia Spark has cooling issues*, while *the Dell is slightly better because it has vents and a fan*.
- **Youtube Video about Spark DGX**: Two Youtube videos were shared, [this one about Nvidia DGX](https://youtu.be/ib913zfNh7I) and [this other one](https://youtu.be/79iDLf9jIJ8).


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1466924630903226645)** (5 messages): 

> `RLMs, Codebase Audit, Kimi k2` 


- ****RLMs** Ace **Codebase Audits****: A member just finished up a post on using **RLMs** to audit a codebase, inspired by a gist on codebase documentation and shared here: [kmad.ai/Recursive-Language-Models-Security-Audit](https://kmad.ai/Recursive-Language-Models-Security-Audit).
- ****Kimi k2** Audits Codebase for Pennies**: A member mentioned **Kimi k2’s** ability at **RLM** given its speed and cost is really impressive in [Auditing a Codebase for 87 cents in 50 lines of code](https://github.com/lastmile-ai/kimi/blob/main/examples/experimental/rlm_code_audit/rlm_code_audit.ipynb).
- **Awaiting **Groq/Cerebras** Hosting**: Members discussed how cool it was to watch the traces and are waiting for **Groq/Cerebras** to host it now.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1466524483698430126)** (21 messages🔥): 

> `Opus New Sandboxes Protocol, Claude Code Issues, GEPA (Geriatric Pareto), Multithreading vs Asyncio, DSPy Prompt Echo Issue` 


- ****Opus** Crafts New Sandboxes, Official Protocols Pending**: The team is building **Opus** to write new sandboxes, and there will be a protocol for official implementations from the providers.
   - This will allow users to swap out their local **PythonInterpreter** for other sandboxes like **E2B**, **Modal**, **Daytona**, etc.
- ****Claude Code** Plagued with Bugs and Troubles**: After *5 days of troubleshooting*, a user logged an issue with **Claude Code**, highlighting the struggle to identify where hooks are stored and suggesting a potential need for reinstallation or a bug report ([link to GitHub issue](https://github.com/anthropics/claude-code/issues/21836)).
   - Some community members are finding that *Claude Code* *seems to be getting closer and closer to being vibeslopped into oblivion.*
- ****GEPA**: Geriatric Pareto slows down computations**: A user nicknamed **GEPA** (Geriatric Pareto) reported slow performance, spending **~6.5 hours** running **30 train**, **30 eval** workflows, each with 3 sequential steps, using **num_threads=30**.
   - The user has **180M TPM**, **30K RPM**, so rate limits are not the problem, but the processing of a full gold dataset of **~300** would be.
- **The Threading Conundrum: Why Multithreading Over Asyncio?**: In the discussion, a user questioned why multithreading was used instead of asyncio, given the potential for queuing up thousands of tasks.
   - Another member pointed out that the limiting factor often tends to be provider rate limits/inference throughput rather than the threading implementation itself.
- ****DSPy** Echoes Prompts, Exasperating Token Budgets**: A user reported an issue with **DSPy** where it's echoing the prompt, consuming the max tokens budget, and causing API calls to last hundreds of seconds, observed on **Gemini 3**, temp 1.0.
   - It produces correct answer but the extra echoing is slowing down the API calls.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1466539780203614269)** (9 messages🔥): 

> `Namespaces, URIs, Groups, Tags, Filtering` 


- **MCP Namespaces get superseded by Groups**: Namespaces were rejected and superseded by groups, but the status of URIs was unclear, though [SEP-1292](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1292) for URIs was closed without much discussion.
   - The discussion pointed to [SEP-1300](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1300) (Groups and Tags) which was ultimately rejected and replaced by [SEP-2084](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2084) as a refined/narrower version.
- **MCP Groups and Tags Proposal Evolved into Primitive Grouping**: [SEP-1300](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1300), which introduced groups, tags, and filtering, failed to reach consensus during a Core Maintainers review.
   - It was superseded by the simpler [SEP-2084](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2084) which focuses on client-side filtering of all primitives by group.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1466636372545769615)** (5 messages): 

> `Manus enhancements, Influencer partnership, AI + Full-Stack Systems` 


- **Manus Queries Surface, Meta Acquisition Unmentioned**: A member inquired whether **Manus** has improved since **Meta** acquired it, also inquiring about major changes and enhancements.
- **Influencer Partnership Opportunity Arises for Manus**: A member sought contact with **Manus's marketing and growth team** for an influencer partnership, clarifying they manage influencers.
   - Manus responded via private message.
- **AI and Full-Stack Developer Showcases Skills**: A member advertised their skills in building **AI and full-stack systems**, focusing on delivering real value and improving efficiency, accuracy, and user experience, further listing expertise in **LLM integration**, **RAG pipelines**, and **AI-driven workflow automation**.
   - They emphasized building clean, maintainable, and secure systems and invited those seeking to ship something solid to reach out.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1466679608962453666)** (3 messages): 

> `NN layer reduction, CUSTOM_KERNEL usage, LlamaForCausalLM as baseline` 


- **NN Layer Reduction Explored**: Members discussed deterministically reducing layers in a neural network into smaller, fused layers to potentially increase efficiency, especially when the NN is known beforehand.
   - The goal is to reduce overhead complexity and improve performance, but it's uncertain if this approach can achieve a **5.5x improvement**.
- **CUSTOM_KERNEL Spotted in UOps**: A member noticed the usage of `CUSTOM_KERNEL` in the UOps within the [tinygrad/tinygrad repo](https://github.com/tinygrad/tinygrad/blob/master/extra/thunder/tiny/fa.py#L364).
   - This was highlighted in the context of working on the bounty for making *llama 1B faster than torch on CPU in CI*.
- **LlamaForCausalLM Considered for Comparison**: A member inquired whether the Hugging Face model, specifically `LlamaForCausalLM`, is suitable as a fair comparison baseline for performance.
   - The setup involves using **one core** and compiling with **TorchInductor**.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1466850624565022863)** (1 messages): 

> `Modular 26.1 Release, MAX Python API, MAX LLM Book, Apple Silicon GPU Support, Community Models` 


- **Modular 26.1 Drops, Debugging Eagerly**: Modular released version **26.1**, featuring debugging in eager mode, one-line compilation, and cross-platform deployment.
   - Details can be found in the [Modular blog](https://www.modular.com/blog/26-1-release-blog).
- **MAX Python API Goes Stable**: The **MAX Python API** is now out of experimental and considered stable, offering PyTorch-like modeling.
   - It includes **model.compile()** for production use.
- **MAX LLM Book Released**: The **MAX LLM Book** is now stable and available at [llm.modular.com](https://llm.modular.com), guiding users through building transformers from scratch with executable code.
   - The book contains executable code from start to finish.
- **Apple Silicon GPU Support Expands**: **Apple Silicon GPU** support is expanding with MAX graphs and Mojo GPU puzzles now working.
   - The platform facilitates community models that include **Qwen3**, **BERT**, **Mamba**, and visual generation pipelines.
- **Mojo Nears 1.0 with Compile-Time Reflection**: **Mojo** is getting closer to version **1.0**, featuring compile-time reflection, linear types, and improved error messages.
   - The platform is growing faster due to community contributions of models and kernels; everything is open source, including the API, kernels, models, and serving.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1466531796878364934)** (2 messages): 

> `Modular issue 5875, Python float to Mojo Float64 conversion` 


- **Modular Bug Bites PyTorch Interop**: A member reported a [bug](https://github.com/modular/modular/issues/5875) encountered when converting a Python float to a Mojo **Float64** in Mojo version **26.1**.
   - The user indicated that code which previously worked in version **25.6** now results in an *"ambiguous call to '__init__'"* error when dealing with PyTorch interop.
- **Float64 Conversion Fails in Mojo 26.1**: In Mojo **26.1**, a user is facing issues converting a Python float to a Mojo **Float64** when using PyTorch via Python Interop.
   - The code, which previously worked in Mojo **25.6**, now throws an *"ambiguous call to '__init__'"* error, specifically when assigning the converted float to `self.model_output[i]`.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1466895814617202828)** (1 messages): 

> `Arena Mode Launch, Plan Mode Addition, Windsurf Credits` 


- **Windsurf Launches Arena Mode for Model Battles!**: Windsurf launched **Arena Mode** in Wave 14, allowing users to compare AI model responses side-by-side and vote on the better one.
   - Users can engage in **Battle Groups** (random models) or **Pick your own** model comparisons, feeding into personal and public leaderboards; see [launch tweet here](https://x.com/windsurf/status/2017334552075890903?s=20).
- **Windsurf Waives Credits for Arena Mode!**: To celebrate the launch, **Battle Groups** in Arena Mode will consume **0x credits** for the next week for both trial and paid users.
   - This promotion encourages users to explore and vote on models, contributing to both personal and aggregated public leaderboards.
- **Plan Mode Joins the Windsurf Cascade!**: Windsurf has added **Plan Mode**, accessible via the Cascade toggle where users switch between Code and Ask Modes.
   - To get started, users need to install the update and relaunch Windsurf via the [download link](https://windsurf.com/download/editor).


  

---


---


---

