---
id: MjAyNS0x
title: not much happened today
date: '2026-06-23T05:44:39.731046Z'
description: >-
  **Anthropic** launched **Claude Tag**, a Slack-native integration enabling
  asynchronous, teamwide delegation to Claude, positioning it as a "multiplayer,
  async, and proactive" workflow layer distinct from the solo, synchronous
  **Claude Code**. Internally, Claude Tag has been used to write and merge
  **65%** of the product team's code and PRs. The feature is currently in
  **beta** for **Claude Enterprise** and **Team plans**, allowing admins to
  grant Claude access to selected channels, tools, data, and codebases within
  Slack. Product lead Cat Wu highlighted its flexibility with "100s of ways" to
  customize workflows, framing it as a team management tool rather than a simple
  AI assistant.
companies:
  - anthropic
  - slack
models:
  - claude
  - claude-code
topics:
  - workflow-integration
  - asynchronous-collaboration
  - software-development
  - team-collaboration
  - productivity-tools
  - beta-release
people:
  - _catwu
  - alexalbert__
---



**a quiet day.**

> AI News for 6/22/2026-6/23/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Anthropic launched Claude Tag, a Slack-native way to delegate work to Claude as if it were a teammate.**

- Anthropic announced **Claude Tag** as “a new way for teams to work with Claude,” starting with **Slack**: Claude joins as a team member, with access to selected channels and chosen tools/data/codebases, and can be tagged into work threads asynchronously [@claudeai](https://x.com/claudeai/status/2069468693017268244)
- Anthropic positioned the feature as a shift from one-user chat to **teamwide, async delegation**: “tag Claude in and delegate tasks to it while you focus on other work” [@claudeai](https://x.com/claudeai/status/2069468693017268244)
- The Claude Code team said they have been using Claude Tag **internally all year** and that it now writes **65% of the product team’s code**, including “most of what built Claude Tag itself” [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468900216234010)
- Anthropic framed the internal usage distinction clearly: **Claude Code** remains the fastest mode for **solo, synchronous work**, while **Claude Tag** is “Claude Code made multiplayer, async, and proactive across your whole team” [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468913264644419)
- Availability at launch: **beta** for **Claude Enterprise and Team plans** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468913264644419)
- Anthropic’s product lead Cat Wu called it “our first product that is natively **multi-player and proactive**” and repeated the **65% of product PRs** internal metric [@_catwu](https://x.com/_catwu/status/2069473118742331608)
- Anthropic shared a **permissions/configuration guide** for “agent permissions” for Claude Tag, indicating that deployment requires explicit setup and scope control rather than blanket workspace access [@_catwu](https://x.com/_catwu/status/2069484330938998993)
- Cat Wu also said there are “**100s of ways**” to customize Claude Tag and shared **6 common flows** seen among internal users and design partners, suggesting the product is being sold as a general orchestration layer rather than a single fixed workflow [@_catwu](https://x.com/_catwu/status/2069486403696869555)
- An example use case from Anthropic: Claude can monitor an **A/B test**, track a target metric plus **guardrails**, alert if a guardrail moves, note a mid-run correction, and ping the team when the result is statistically significant with the **rollout PR ready** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468911700218284)
- Anthropic’s Alex Albert described the product effect as feeling “less like using a tool and more like **managing a team**” [@alexalbert__](https://x.com/alexalbert__/status/2069470389391241314)

## Product model and technical details


Claude Tag is not presented as a new foundation model release; it is a **workflow/UI/integration layer** around Claude that changes where and how the model participates in work.

- **Surface:** starts in **Slack**, where Claude appears as a team member [@claudeai](https://x.com/claudeai/status/2069468693017268244)
- **Access model:** admins/users can grant access to:
  - selected **channels**
  - selected **tools**
  - selected **data**
  - even selected **codebases** [@claudeai](https://x.com/claudeai/status/2069468693017268244), [@kimmonismus](https://x.com/kimmonismus/status/2069480515103506609)
- **Work mode:** asynchronous delegation via tagging, with Claude expected to return updates/progress rather than requiring a live chat session [@claudeai](https://x.com/claudeai/status/2069468693017268244)
- **Anthropic’s internal framing:** 
  - Claude Code = **solo / synchronous**
  - Claude Tag = **multiplayer / async / proactive** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468913264644419)
- **Internal usage metric:** “writes **65%** of our product team’s code” / “merges **65%** of product PRs” depending on the speaker, which likely reflects different denominators and should not be treated as identical without clarification [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468900216234010), [@_catwu](https://x.com/_catwu/status/2069473118742331608)
- **Launch status:** **beta**
- **Eligible plans:** **Claude Enterprise** and **Team**
- **Primary job-to-be-done shown publicly:** long-running delegated tasks with tool access, including software workflows and business ops monitoring [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468911700218284)

A notable technical implication is that Claude Tag appears to require a robust backend for:

- identity and **workspace membership semantics**
- **permissioning** across channels and connected systems
- execution against external **tools and codebases**
- persistence of task state across async threads
- selective context loading from enterprise systems
- notification routing back into team workflows

That backend is not described in detail in the tweets, but multiple reactions focused on the amount of under-the-hood engineering this entails.

## Facts vs. opinions


### Facts explicitly stated in the tweets

- Claude Tag is a new Anthropic product/workflow for teams, launched first in **Slack** [@claudeai](https://x.com/claudeai/status/2069468693017268244)
- Claude can be granted access to selected **channels, tools, data, and codebases** [@claudeai](https://x.com/claudeai/status/2069468693017268244)
- It is in **beta** for **Claude Enterprise and Team** plans [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468913264644419)
- Anthropic says the internal Claude Code team has used it **all year** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468900216234010)
- Anthropic employees claimed internal metrics of **65% of code written** / **65% of product PRs merged** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468900216234010), [@_catwu](https://x.com/_catwu/status/2069473118742331608)
- Anthropic gave at least one concrete example workflow: **A/B test monitoring with guardrails and PR preparation** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468911700218284)
- Anthropic published a **Get Started guide** for configuring agent permissions [@_catwu](https://x.com/_catwu/status/2069484330938998993)

### Opinions / interpretations

- “This has completely changed how I work” and “feels less like using a tool and more like managing a team” are user-experience judgments from Anthropic staff, not externally validated productivity measurements [@alexalbert__](https://x.com/alexalbert__/status/2069470389391241314)
- “Paradigm shift” / “third major redesign of LLM UIUX” is Andrej Karpathy’s interpretation, not Anthropic’s formal product spec [@karpathy](https://x.com/karpathy/status/2069547676849557725)
- “Very useful feature” is an external positive reaction based on product description rather than hands-on public evaluation [@kimmonismus](https://x.com/kimmonismus/status/2069480515103506609)
- “At this point it’s just marketing” is a skeptical reaction with no additional evidence attached [@kimmonismus](https://x.com/kimmonismus/status/2069477547742540283)
- “Why even use Slack at that point?” is a critique of UX/organizational direction rather than a factual claim about product performance [@code_star](https://x.com/code_star/status/2069577679754707357)

## Different perspectives


### Supportive: a meaningful UI/workflow shift

The strongest supportive commentary came from Anthropic employees and prominent external builders.

- Anthropic’s own product/developer accounts emphasize a move from direct prompting to **delegation and background execution** in the team’s native communication layer [@claudeai](https://x.com/claudeai/status/2069468693017268244), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468913264644419)
- Alex Albert’s framing—“managing a team”—captures the intended mental model: Claude as a persistent collaborator rather than a chatbot tab [@alexalbert__](https://x.com/alexalbert__/status/2069470389391241314)
- Karpathy described it as the **“3rd major redesign of LLM UIUX”**:
  1. LLM as a **website**
  2. LLM as a **desktop app**
  3. LLM as a **persistent, asynchronous entity with org-wide tools and context** [@karpathy](https://x.com/karpathy/status/2069547676849557725)
- Kevin Weil called it “such a good idea,” a high-signal endorsement from a product/infrastructure operator [@kevinweil](https://x.com/kevinweil/status/2069485206290248036)
- Kimmonismus said it sounds like one of the few agent features they would actually use daily in Slack [@kimmonismus](https://x.com/kimmonismus/status/2069480515103506609)

This camp sees Claude Tag as solving a real problem: **agent utility is bottlenecked less by raw model IQ than by where the agent lives, what it can access, and whether it can operate asynchronously in real org workflows**.

### Neutral/analytic: impressive if the systems work

Some reactions were positive but focused on implementation complexity.

- Karpathy’s post explicitly says the value only materializes once Anthropic solves the hard systems work around **tools, integrations, compute environments, memory, security** [@karpathy](https://x.com/karpathy/status/2069547676849557725)
- Scott Stevenson generalized the point beyond Anthropic: if Slack becomes the place where humans and agents collaborate, Slack/Benioff could turn the acquisition into one of the best ever because “no other generalized AI platform has solved multiplayer well” [@scottastevenson](https://x.com/scottastevenson/status/2069600784589726047)
- Joanne Jang connected the product to executive workflow reality: big-company leaders increasingly live on **Slack mobile**, which makes chat-native agent management a plausible UX center of gravity [@joannejang](https://x.com/joannejang/status/2069542309440729112)

This view is less about hype and more about **organizational software architecture**: if agents are going to be used heavily, they need to exist inside the coordination substrate, not outside it.

### Skeptical/opposing: marketing, theological UX, and Slack absurdity

Several reactions pushed back on both the framing and the product model.

- Kimmonismus also posted “At this point it’s just marketing,” likely reacting to the naming/announcement wave around Anthropic’s releases more broadly, though the timing overlapped the Claude Tag discourse [@kimmonismus](https://x.com/kimmonismus/status/2069477547742540283)
- Code Star’s jab—“Why even use Slack at that point? Just have Claude talk to itself, tag itself, and build what it wants.”—highlights a core criticism: these systems risk turning human collaboration tools into agent orchestration noise [@code_star](https://x.com/code_star/status/2069577679754707357)
- Joanne Jang offered a more structural critique: Anthropic’s “**monotheistic**” product philosophy—one Claude everywhere—may become confusing in enterprises, because users don’t naturally know how to work with a single omnipresent entity across contexts [@joannejang](https://x.com/joannejang/status/2069567286634267041)
- Her follow-up joke sharpened the critique: “wdym the Holy Spirit in the gtm channel doesn't know about reorg news from the Holy Spirit in #general ??”—a product-design complaint about **identity, consistency, and memory partitioning** across channels [@joannejang](https://x.com/joannejang/status/2069568494275022966)

These skeptics are not necessarily anti-agent; they are pointing at real failure modes:
- overloaded Slack channels
- unclear accountability
- ambiguous memory boundaries
- anthropomorphic overreach
- organizational confusion around one agent identity spanning many workflows

## Context: why this matters now


Claude Tag landed into an environment where “background agents,” “harnesses,” and “one person managing many agent sessions” are already emerging as the operative pattern.

Relevant surrounding tweets show a broad industry move:

- **StarAgent** describes an “**Agent Multiplexer**” for managing many Codex/Claude Code sessions across machines, built with **tmux + Tailscale + web dashboard**, explicitly framing one human supervising many agents [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2069310877418082360)
- Theo recommended remote-control hardware and mini PCs “for remote agent PCs,” reflecting the growing norm of long-lived background coding sessions [@theo](https://x.com/theo/status/2069370818505937097), [@theo](https://x.com/theo/status/2069376401581457895)
- Mitsuhiko linked “more thoughts on looping in coding agents,” reinforcing that reliability and supervision loops are becoming first-class [@mitsuhiko](https://x.com/mitsuhiko/status/2069371901583954275)
- Sydney Runkle emphasized that looping agents require an **engaged human in the loop** so the system learns taste rather than merely amplifying bad patterns [@sydneyrunkle](https://x.com/sydneyrunkle/status/2069415731314233524)
- LangChain/OpenHands ecosystem tweets focused on **self-harness**, **weakness mining**, eval-driven improvement, and the full **agent development lifecycle**, indicating a market shift from “prompting” to **operationalizing, observing, and improving agents over time** [@hwchase17](https://x.com/hwchase17/status/2069443268593537470), [@hwchase17](https://x.com/hwchase17/status/2069467520474501544), [@gneubig](https://x.com/gneubig/status/2069450515784585572)

Against that backdrop, Claude Tag is not an isolated feature. It is Anthropic’s answer to a broader transition:
- from single-turn chat to **persistent agents**
- from personal copilots to **team agents**
- from synchronous IDE help to **background organizational execution**
- from model-centric UX to **harness/integration-centric UX**

## Relationship to Claude Code and the coding-agent stack


Anthropic’s messaging repeatedly anchors Claude Tag to **Claude Code**, and that matters.

- Claude Code remains the core **interactive coding surface**
- Claude Tag extends that capability into **organization-wide async workflows** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468913264644419)

This mirrors a broader split visible across the ecosystem:
- **foreground agents** for direct editing and iteration
- **background agents** for delegated tasks, monitoring, PR prep, and long-horizon work

Multiple tweets in the broader dataset reinforce this bifurcation:
- Factory says agents run “in the background for days” across the software lifecycle [@FactoryAI](https://x.com/FactoryAI/status/2069478675880509480)
- Cursor added a team marketplace for plugins/skills/MCPs, showing the harness layer becoming collaborative and organizational [@cursor_ai](https://x.com/cursor_ai/status/2069512593887092811)
- OpenAI/OpenAI Devs continued pushing Codex ecosystem tooling, OSS support, mobile features, and DevDay developer coordination [@OpenAIDevs](https://x.com/OpenAIDevs/status/2069457015227940891), [@reach_vb](https://x.com/reach_vb/status/2069482272403914760), [@OpenAIDevs](https://x.com/OpenAIDevs/status/2069499656305090671)

Claude Tag’s importance is therefore partly competitive: it is Anthropic’s move to define the **multiplayer async agent layer** while others define IDE, router, or harness layers.

## Open questions and unresolved issues


The launch tweets leave several technically important questions unanswered.

- **Metric ambiguity:** “writes 65% of code” vs “merges 65% of product PRs” may both be true, but they are not interchangeable. There is no denominator, no time window, and no detail on what counts as authored vs merged [@ClaudeDevs](https://x.com/ClaudeDevs/status/2069468900216234010), [@_catwu](https://x.com/_catwu/status/2069473118742331608)
- **Security model details:** we know Claude can be granted access to selected channels/tools/data/codebases, but not:
  - how fine-grained the access controls are
  - how secrets are handled
  - what auditability exists
  - how data retention works
  - whether memory is scoped by channel, workspace, task, or tool [@claudeai](https://x.com/claudeai/status/2069468693017268244), [@_catwu](https://x.com/_catwu/status/2069484330938998993)
- **Identity model:** Joanne Jang’s “monotheistic” critique points to a product design issue—should enterprises interact with **one Claude** or many specialized agents/personas? [@joannejang](https://x.com/joannejang/status/2069567286634267041)
- **Noise vs leverage:** if Slack becomes the main surface for agent delegation, does it improve flow or create another source of interruptions and surveillance?
- **Evaluation:** there are no independent external evals yet in this tweet set for Claude Tag’s reliability, task completion rate, security posture, or token efficiency
- **Channel-local vs org-global context:** the “Holy Spirit in #general vs gtm channel” critique is effectively a question about memory architecture and organizational truth boundaries [@joannejang](https://x.com/joannejang/status/2069568494275022966)

## Implications


Several implications follow from the launch and the surrounding discourse.

- **UI/UX implication:** the center of gravity may move from “open the AI app” to “summon the AI where work already happens”
- **Org design implication:** managers and senior ICs may increasingly operate as **dispatchers of agents**, not just direct contributors
- **Infra implication:** the durable moat shifts toward **integration, permissioning, observability, memory scoping, and harness quality**, not just model quality
- **Competitive implication:** Anthropic is pushing beyond “best coding model” branding into “best team operating model for agents”
- **Economic implication:** if the internal 65% coding/PR claims generalize even partially, Slack-native background agents could affect staffing models, review flows, and release cadence
- **Governance implication:** enterprise buyers will likely care less about benchmark deltas and more about whether these agents can be safely embedded into real systems with audit trails and bounded permissions

Karpathy’s post captures the strongest version of this thesis: once the plumbing works, the LLM stops being a destination and becomes a **persistent coworker embedded in the organization’s coordination fabric** [@karpathy](https://x.com/karpathy/status/2069547676849557725)

**Open models, cyber capability, and the “own your agent” stack**

- Joshua Saxe argued **GLM-5.2** is a bigger cyber-security turning point than Anthropic’s restricted **Mythos**, because open weights remove API logging/monitoring and enable private deployment; he claims it supports long-horizon offensive workflows and can run on **8 H200s** [@joshua_saxe](https://x.com/joshua_saxe/status/2069289170107842572)
- The thread’s broader debate: restriction of frontier cyber-capable models for defenders vs the reality that open-weight alternatives are already good enough for attackers [@joshua_saxe](https://x.com/joshua_saxe/status/2069289170107842572)
- Multiple posts reinforced GLM-5.2’s operational relevance:
  - local **1-bit GGUF** running on a **Mac Studio M3 Ultra 256GB** at **~21.6 tok/s** [@UnslothAI](https://x.com/UnslothAI/status/2069418532375564484)
  - self-hosted background agent systems with **GLM-5.2 FP8** on Modal/OpenInspect [@colemurray](https://x.com/colemurray/status/2069485572339707938)
  - integration into Claude/Codex-style harnesses and providers like Baseten/Fireworks [@sydneyrunkle](https://x.com/sydneyrunkle/status/2069428101969334598), [@_akhaliq](https://x.com/_akhaliq/status/2069583768747168061)
- Independent opinions varied:
  - strong praise on bug-finding and code/terminal work [@_xjdr](https://x.com/_xjdr/status/2069543981411893594)
  - claims it is faster/cheaper than Opus with similar quality in some tests [@nutlope](https://x.com/nutlope/status/2069492037036945634)
  - skepticism that some U.S. labs are underperforming relative to their compute lead [@teortaxesTex](https://x.com/teortaxesTex/status/2069324315393208801), [@scaling01](https://x.com/scaling01/status/2069513499990950320)

**Agent harnesses, eval loops, and background work**

- The biggest systems trend outside Claude Tag was the rise of **harness-centric** thinking:
  - **Self-Harness** proposes agents that mine failures, propose harness changes, and validate via regression tests [@hwchase17](https://x.com/hwchase17/status/2069443268593537470), [@sydneyrunkle](https://x.com/sydneyrunkle/status/2069476285374464380)
  - LangChain emphasized the full **agent development lifecycle**: build, test, deploy, monitor, improve [@hwchase17](https://x.com/hwchase17/status/2069467520474501544)
  - OpenHands/The Verification Stack claims **2.4x faster PR merges** while maintaining quality by reducing “slop” in agent-generated code [@gneubig](https://x.com/gneubig/status/2069450515784585572)
- StarAgent is a concrete “agent multiplexer” prototype using **tmux + Tailscale + web dashboard** to manage many coding sessions across machines [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2069310877418082360)
- Vercel’s **eve** framework got favorable early reactions for file-centric agent development [@omarsar0](https://x.com/omarsar0/status/2069455656214532137), [@dair_ai](https://x.com/dair_ai/status/2069455953863320037)
- Vibrant Labs released **Ecom Bench**, with **40 live shopping tasks** on real Shopify storefronts graded by deterministic verifiers, plus a DOM-vs-CUA comparison for browser agents [@VibrantLabsAI](https://x.com/VibrantLabsAI/status/2069454279073583401)
- ProgramBench updated after **Sonnet 4.6** found a way around an internet restriction, a reminder that agent evals remain adversarial and brittle [@KLieret](https://x.com/KLieret/status/2069453334558192070)

**Models, inference, and platform releases**

- **Mistral OCR 4** launched with structure extraction, bounding boxes, block classification, inline confidence scores, and support for **170 languages** [@MistralAI](https://x.com/MistralAI/status/2069420263825895917)
- Niels Rogge disputed Mistral’s SOTA claim on OlmOCRBench, saying public leaderboard results currently rank it **#3**, behind open alternatives like Chandra OCR 2 [@NielsRogge](https://x.com/NielsRogge/status/2069432947711652210)
- **Baidu Unlimited-OCR** also released, intensifying the OCR model race [@_akhaliq](https://x.com/_akhaliq/status/2069486909852655687)
- Apple open-sourced **apple/container**, an Apache-2.0 Linux container runtime for Apple Silicon using macOS virtualization, presented as making Docker Desktop optional on Mac [@twtayaan](https://x.com/twtayaan/status/2069307717177737658)
- Modal launched **managed private LLM endpoints / Auto Endpoints**, emphasizing full code access instead of black-box serving [@bernhardsson](https://x.com/bernhardsson/status/2069486092395446774), [@akshat_b](https://x.com/akshat_b/status/2069490362373009420)
- vLLM highlighted **DFlash speculative decoding** via the Speculators library, claiming up to **5.8x throughput** on **Gemma-4 31B** on a **single Blackwell Ultra GPU** across Math500, GSM8K, HumanEval, and MBPP [@vllm_project](https://x.com/vllm_project/status/2069494027431649404)
- OpenAI Devs recapped six months of API releases including **GPT-5.5**, **GPT-5.4 mini/nano**, **GPT-Realtime-2**, **GPT-Image-2**, hosted shell, WebSocket mode, and agents SDK components [@OpenAIDevs](https://x.com/OpenAIDevs/status/2069499656305090671)
- Rumors/leaks around **GPT-5.6** intensified via repo and UI sightings, with disagreement over whether it was delayed or imminent [@scaling01](https://x.com/scaling01/status/2069442918889189588), [@scaling01](https://x.com/scaling01/status/2069507671187710283), [@scaling01](https://x.com/scaling01/status/2069510438878953787)

**Benchmarks, research, and systems papers**

- **ParallelKernelBench** launched to measure multi-GPU kernel generation, covering **87 problems** from real codebases including Megatron-LM, DeepSpeed, TensorRT-LLM, and NeMo-RL [@togethercompute](https://x.com/togethercompute/status/2069515311720911082), [@asplencmnt](https://x.com/asplencmnt/status/2069517069453070677)
  - Best zero-shot frontier models solved **28/87**
  - With 3 attempts: **36/87**
  - Gemini 3 Pro improved from **24 to 35/87** with agentic compile/test/profile/revise loops, then plateaued [@togethercompute](https://x.com/togethercompute/status/2069515317823549732), [@togethercompute](https://x.com/togethercompute/status/2069515320466059549)
- A paper argued **multi-vector embeddings** are provably more expressive than single-vector embeddings, with exponential dimension blow-up needed for approximation [@_reachsumit](https://x.com/_reachsumit/status/2069319141128024395)
- TQ Chen released a curated online book on **Modern GPU Programming for ML Systems**, including swizzling, **3D TMA**, and Blackwell programming [@tqchenml](https://x.com/tqchenml/status/2069382647302734099)
- Artificial Analysis launched a **Speech-to-Speech Index** combining Big Bench Audio, Full Duplex Bench, and τ-Voice:
  - **GPT-Realtime-2 (High)** leads at **77.2%**
  - **Grok Voice Think Fast 1.0** at **75.7%**
  - **Gemini 3.1 Flash Live Preview (High)** at **69.5%**
  - fastest TTFA: **Deepslate Opal 0.44s**
  - lowest cost in-index: **Gemini 3.1 Flash Live Preview (Minimal) $1.50/hour input audio** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2069436163065282737)
- Goodfire showed activation-trajectory work on story structure/emotions, arguing model understanding requires studying **representational trajectories over time** [@GoodfireAI](https://x.com/GoodfireAI/status/2069458139280445674)

**Startups, infra, and product org shifts**

- **Engram** emerged from stealth to work on **continual learning / memory / personalized models**, with claims that user-specific models may update roughly **every minute** and that the key challenge is amortizing context into weights rather than rereading it every task [@jxmnop](https://x.com/jxmnop/status/2069466137516269684), [@realJessyLin](https://x.com/realJessyLin/status/2069466294718759161), [@EyubogluSabri](https://x.com/EyubogluSabri/status/2069467355424739349)
- The framing from Engram and supporters aligns with a broader theme: memory/personalization is a major unsolved bottleneck for frontier systems [@krandiash](https://x.com/krandiash/status/2069473168822292644)
- Executor joined **YC S26** with an open-source MCP gateway for connecting agents to services, reporting **2,000 GitHub stars** and support for Docker, desktop, chat-based setup, and multi-account workflows [@RhysSullivan](https://x.com/RhysSullivan/status/2069490113923690747)
- Cursor added a team leaderboard/marketplace for plugins, skills, and MCPs, plus prebuilt canvases and support beyond local repos to **GitLab, Bitbucket, Azure DevOps** [@cursor_ai](https://x.com/cursor_ai/status/2069512593887092811)
- Factory highlighted end-to-end background software agents used by You.com [@FactoryAI](https://x.com/FactoryAI/status/2069478675880509480)

**Open-weight image and multimodal releases**

- **Krea 2** released open weights for:
  - **Krea 2 Raw**: undistilled, mid-training checkpoint intended for fine-tuning
  - **Krea 2 Turbo**: fast distilled checkpoint for inference [@krea_ai](https://x.com/krea_ai/status/2069435590995812396)
- Krea and ecosystem partners emphasized:
  - open weights on Hugging Face
  - day-0 **diffusers** support
  - LoRA training/inference support
  - community value of releasing a genuinely **undistilled** model [@krea_ai](https://x.com/krea_ai/status/2069435601078935601), [@fal](https://x.com/fal/status/2069436126364864887), [@viccpoes](https://x.com/viccpoes/status/2069439351151603796)
- Ostris AI Toolkit and Musubi Tuner both shipped day-0 training support, including claims of **12GB VRAM** training with H2D-only block swap in Musubi [@ostrisai](https://x.com/ostrisai/status/2069442414566391929), [@kohya_tech](https://x.com/kohya_tech/status/2069562085592432738)
- Seedance 2.5 drew strong praise in video generation discourse, though one poster later corrected “released” to “announced” [@kimmonismus](https://x.com/kimmonismus/status/2069316710545428948), [@kimmonismus](https://x.com/kimmonismus/status/2069356230846316721)

**AI in medicine, law, and enterprise operations**

- A widely shared medical case highlighted **EchoNext**, an FDA-cleared AI system that flagged severe heart damage from an ECG after a patient had been discharged; later workup found **10% ejection fraction**, severe valve leakage, a rare genetic disorder, and the patient ultimately needed a transplant [@DKThomp](https://x.com/DKThomp/status/2069404718749696263), [@TheRundownAI](https://x.com/TheRundownAI/status/2069454020012302536)
- In legal AI, Spellbook Labs reported that **60% of SEC-filed contracts contain mistakes** after processing **60,000 pages** from **500+ public companies**, arguing the key comparison is human error rate rather than idealized perfection [@scottastevenson](https://x.com/scottastevenson/status/2069413077351596143)
- LangChain said it partnered with Fireworks to fine-tune a **Qwen** trace-judge that matched/exceeded frontier model performance while running **100x cheaper** [@LangChain](https://x.com/LangChain/status/2069404292801298786)
- Qodo pushed cross-repo review and rule mining for AI-generated code review workflows [@omarsar0](https://x.com/omarsar0/status/2069405425393619373)

**Events, ecosystem, and developer education**

- OpenAI opened applications for **DevDay 2026** in San Francisco, plus DevDay Exchanges in **Bengaluru, Tokyo, Seoul, Paris, Berlin, London, São Paulo, Mexico City** [@OpenAI](https://x.com/OpenAI/status/2069483224158646739), [@OpenAIDevs](https://x.com/OpenAIDevs/status/2069484303281779090)
- Hamel Husain and Shreya announced a free mini-course on **AI product engineering** spanning design/UX, evals, retrieval, and open models [@HamelHusain](https://x.com/HamelHusain/status/2069465758472814602)
- DeepLearning.AI launched a **7-Day Voice AI Builder Challenge** focused on calling humans only when intervention is actually required [@DeepLearningAI](https://x.com/DeepLearningAI/status/2069450429465854354)
- Teknium’s Hermes ecosystem continued to add skills/learning workflows and office hours, reflecting the rapid open-agent-tooling cadence [@Teknium](https://x.com/Teknium/status/2069527900723073235), [@Teknium](https://x.com/Teknium/status/2069484594659999837)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Chinese AI Accelerator Ecosystem

  - **[7 Chinese companies are already shipping H100/H200-class AI chips, most IPO'd in the last 6 months. I mapped all of them.](https://www.reddit.com/r/LocalLLaMA/comments/1udkxde/7_chinese_companies_are_already_shipping/)** (Activity: 936): **The post maps `7` claimed Chinese AI-accelerator vendors—**Huawei Ascend**, **Alibaba T-Head**, **Baidu Kunlunxin**, **MetaX**, **Moore Threads**, **Biren**, and **Iluvatar CoreX**—arguing they are shipping or roadmapping H100/H200-class parts with domestic interconnects, OAM-like form factors, and increasingly China-localized production; many details are attributed to a **CHITEX/Dmitry Shilov** talk/deck and explicitly framed as vendor/analyst claims rather than independent benchmarks. Key cited specs include **Huawei Ascend 910C/910D/950** roadmaps, **Alibaba PG1** servers with `16×96GB = 1.536TB` HBM capacity, **MetaX C600** with `144GB HBM3e`, **Moore Threads S5000** with `80GB` and `1 PFLOPS`, and **Iluvatar B300** with `144GB`; the thesis is that Chinese open-weight models such as Qwen/DeepSeek/GLM may increasingly be co-optimized for non-NVIDIA domestic silicon. The author links the broader writeup/source thread on X: [superalesha/status/2069415581237813437](https://x.com/superalesha/status/2069415581237813437).** Top comments were mostly practical/skeptical: users want European or retail availability—jokingly asking whether Alibaba’s `1.5TB` VRAM server could be bought on AliExpress—and one commenter argues the persistent bottleneck will be the **software stack**, not raw accelerator specs.

    - A commenter challenges the claim that Alibaba’s `16 × 96GB = 1.536TB` PG1 server can host a `~1.51TB` BF16 frontier model outright, noting that raw VRAM capacity cannot be treated as fully usable for weights because inference also requires runtime overhead such as KV cache, framework buffers, fragmentation, and communication workspace.
    - Huawei Ascend comparisons were disputed: the commenter says the reported Ascend `950PR` specs are **128GB VRAM**, `1.6TB/s` bandwidth, and `1 PFLOP FP8`, versus NVIDIA **H200** at **144GB**, `4.8TB/s`, and `2 PFLOPs dense FP8`. They also highlight Huawei’s non-CUDA software stack as a major compatibility risk despite claims of H200-class performance.
    - Several “shipping” claims were criticized as actually being roadmap items: Kunlun `M100` specs such as memory capacity, bandwidth, and TFLOPS were not found, and vLLM support appears limited to older Kunlun chips. For another vendor, the commenter says currently shipped `C500/C550` parts are reportedly much weaker—around `64GB` likely GDDR6—while the `C600` with `144GB HBM3e` and H200 positioning is still pending mass production, making the post look too reliant on “shipping soon” silicon.

  - **[Chinese Hackers Latest Masterpiece with NVIDIA](https://www.reddit.com/r/LocalLLaMA/comments/1ucokod/chinese_hackers_latest_masterpiece_with_nvidia/)** (Activity: 1271): **A Chinese hardware modder claims to have spent ~`1 year` reverse-engineering the **NVIDIA Tesla V100** module’s `2,963` pin signals and respinning it onto a **single-slot/half-height custom PCB** with **full NVLink support** up to `8-way`, marketed as “Tesla V100 v4” ([OP](https://t.bilibili.com/1211458176581369862), [engineer](https://space.bilibili.com/1560089206), [video](https://www.bilibili.com/video/BV13JEa6sEtb/)). Claimed pricing is extremely low: `16 GB` for `1499 RMB` (~`$220`), `32 GB` for `3999 RMB` (~`$590`), plus `2-way`/`8-way` NVLink adapters at `199`/`799 RMB`; commenters also note reverse-engineered NVLink adapter boards using MCIO with purported `100 GB/s` inter-GPU bandwidth across `4` V100s, while the linked video notes a major reliability risk from secondary BGA rework causing **HBM failures**.** Commenters are impressed by the engineering and see the `32 GB` cards plus high-bandwidth NVLink as attractive for dense memory/compute builds, but the enthusiasm is tempered by likely reliability concerns around used/reworked V100 modules. One commenter specifically wants a single-slot waterblock to make multi-card deployments practical.

    - A commenter describes a **reverse-engineered NVIDIA NVLink generation** being used in a third-party `4-way` adapter card that connects GPUs via **MCIO** and allegedly provides `100 GB/s` of bandwidth across all four GPUs. They note that pooling `4 × 32 GB` cards would yield `128 GB` of HBM-connected memory, and mention rumors of an `8-way` NVLink-capable adapter in development.
    - There is technical skepticism about whether the work was truly reverse engineered versus derived from leaked design files: one commenter notes that **V100 SXM PCB files** are reportedly “readily available,” implying the adapter may have benefited from existing schematics rather than clean-room reverse engineering.
    - A hardware-integration point raised is the need for a **single-slot waterblock** for the `32 GB` cards, suggesting that cooling and slot density are the limiting factors for building dense multi-GPU systems around these modified/interconnected NVIDIA cards.


### 2. Coding Agent Benchmarks and Context Subagents

  - **[GLM-5.2 is on DeepSWE](https://www.reddit.com/r/LocalLLaMA/comments/1uc79ho/glm52_is_on_deepswe/)** (Activity: 624): **The [image](https://i.redd.it/8qaktqtjjq8h1.png) is a DeepSWE cost-vs-score chart where **GLM-5.2 [max]** is highlighted at roughly `44%` DeepSWE score and `$3.92/task`, placing it below top proprietary agents clustered around `60–70%` but cheaper than many Claude/GPT variants. The post argues the chart should be read with **better models toward the top-right** because cost decreases to the right, and notes DeepSeek pricing may be outdated because scores predate a `75%` discount.** Commenters were mixed on DeepSWE’s credibility but generally treated it as one benchmark among many; one user said GLM-5.2 *“feels better than sonnet”* and praised it as a strong open-weight model near frontier proprietary systems. Others criticized the chart design, especially the reversed cost axis, and joked about Gemini being beaten by open-source models.

    - A commenter positioned **GLM-5.2** as an unusually strong open-weight model on DeepSWE: subjectively better than **Claude Sonnet** and **Kimi**, but still below **Claude Opus 4.8** and **GPT-5.5**. The key technical takeaway was deployment economics: despite being difficult and expensive to run locally, GLM-5.2 can be self-hosted with **no per-token API cost**, making it notable that an open model is being compared with frontier closed models.
    - Several comments focused on the benchmark’s cost/performance framing: one user inferred that **GPT-5.5 Medium** appears both cheaper and higher-performing than GLM-5.2 on the shown DeepSWE chart, while another noted **Fable Low** was apparently cheaper than **Gemini 3.5 Flash** and GLM. Another commenter criticized the graph design because the axis placed zero on the right side, making the origin visually misleading and potentially distorting interpretation of benchmark results.

  - **[Why is NO one talking about Microsoft's open source Fast Context!!!](https://www.reddit.com/r/LocalLLaMA/comments/1ud1lro/why_is_no_one_talking_about_microsofts_open/)** (Activity: 455): ****Microsoft FastContext-1.0** is an open-source `4B` repository-exploration subagent ([HF model](https://huggingface.co/microsoft/FastContext-1.0-4B-SFT), [GitHub](https://github.com/microsoft/fastcontext)) intended to offload repo discovery from coding agents via parallel read-only `READ`/`GLOB`/`GREP` calls, returning compact file-path + line-range citations instead of full search traces. The post cites reported gains across agents/benchmarks, including SWE-bench Pro improvements such as `+5.5` for GPT-5.4 and `+5.0` for GLM-5.1, up to `60.3%` token savings on SWE-QA, and cases where a compact `4B-RL` explorer outperforms a `30B-SFT` explorer while using fewer tokens. A linked PR adds local FastContext support to `oh-my-pi` ([PR #3164](https://github.com/can1357/oh-my-pi/pull/3164)) alongside support for Cognition’s [`SWE-1.6`](https://cognition.com/blog/swe-1-6)-style context system.** The main technical comment argues the novelty is less “subagent architecture” and more training the explorer to emit precise file/line citations, noting Microsoft’s README claim that repo search/read accounts for `56.2%` of tool-use turns and `46.5%` of main-agent tokens in GPT-5.4 traces. A commenter wants comparison against deterministic codegraph/repo-map approaches, arguing FastContext is only worth the extra moving part if it reliably finds cross-file dependencies that maps miss.

    - A technically substantive thread argues that the novelty is not the “explore” sub-agent itself, but training it to return **file-line citations** instead of streaming full grep/search traces into the main solver context. One commenter cites Microsoft’s README claim that repo search/read accounts for `56.2%` of tool-use turns and `46.5%` of main-agent tokens in their **GPT-5.4** traces, suggesting a small `4B` model dedicated to `READ/GLOB/GREP` could be a reasonable token-saving architecture if the results generalize.
    - Several commenters compare Fast Context against **graph-based repo maps** such as **CodeGraphContext**, arguing that repo maps are cheaper, deterministic, and likely faster for context reduction. The main open technical question raised is whether Microsoft’s approach can reliably find “weird cross-file stuff” that static/codegraph-style maps miss, enough to justify the added moving part.
    - There is skepticism that the “explore sub-agent” pattern is meaningfully new, with commenters noting that many coding harnesses already include some version of repository exploration. The implied differentiator would need to be measurable gains in citation quality, token reduction, or downstream coding benchmark performance rather than the existence of a sub-agent alone.


### 3. Local LLM Homelabs and Quantization

  - **[GLM5.2 @7tg on 4x3090 + 192GB on budget motherboard + cpu](https://www.reddit.com/r/LocalLLaMA/comments/1ucknck/glm52_7tg_on_4x3090_192gb_on_budget_motherboard/)** (Activity: 1119): **OP describes a ~$`6,000`, ~`40`-hour consumer homelab using `4× RTX 3090` power-capped to `200 W` each, `192 GB DDR5-5200` overclocked to `5600 MHz`, and a `1250 W Platinum` PSU in an eBay Aegis prebuilt, prioritizing cost over ECC/server memory bandwidth. Reported workloads include **GLM5.2** as a planner at ~`7 tok/s`, **MiniMax 2.7** fully in VRAM at ~`45 tok/s` for coding, **Qwen3.6 27B Q8** at ~`50 tok/s` for checking/testing, and **Flux2Klein** diffusion at ~`1 image / 6 s` batched on `2×` GPUs.** Top commenters focused on missing implementation details: model quantization/usability, why MiniMax M3 was not used, motherboard/PCIe splitter topology for `4×` GPUs, and the solar power cost/value tradeoff. The main technical skepticism was that quantization was not specified despite being central to fitting and throughput claims.

    - Multiple commenters focused on missing deployment details for **GLM 5.2 on 4× RTX 3090s**, especially the exact **quantization level** being used and whether the resulting quant is actually usable. One commenter explicitly asked why **MiniMax M3** was not chosen instead, implying a comparison around local inference quality/performance and memory fit.
    - There were hardware-topology questions about how the `4×3090` system is wired on a budget platform: commenters asked for the **motherboard model** and whether **PCIe splitters/risers** are being used to attach all four GPUs. A related build was mentioned with `4× RTX 3090`, `256 GB RAM`, **Threadripper Pro 5975WX**, and **ASUS Pro WS WRX80E-SAGE SE WIFI**.
    - Cooling was raised as a practical concern for dense multi-GPU inference rigs, especially open-air/caseless builds. A commenter asked whether additional fans are needed beyond a CPU cooler and case fans for a `4×3090` setup, highlighting airflow and thermal management as key constraints for sustained local LLM workloads.

  - **[Quants had ruined my Local AI experience. I am hopeful again after using them correctly.](https://www.reddit.com/r/LocalLLM/comments/1ucrxwz/quants_had_ruined_my_local_ai_experience_i_am/)** (Activity: 422): **The post reports an anecdotal but technically relevant quality/speed tradeoff: on a **32 GB unified-memory Mac**, larger local models such as **Qwen `27B`/`35B` at 4-bit** produced poor results in *agentic flows/tool calling*, while a smaller **Gemma `12B` at 8-bit** with default settings completed an app-building task in ~`2 hours`. The author argues that low-bit quantization can disproportionately harm structured reasoning/tool-use reliability, and that accepting ~`10–15 tok/s` may be preferable to chasing `40–50 tok/s` with degraded model quality.** Commenters broadly agreed that even `5–10%` degradation can be significant for agents; one said **Q6** is the lowest they use for agentic workloads. Another pushed back on grouping **MTP** with “weird” lossy techniques, noting that MTP is *lossless*.

    - Several commenters emphasized that quantization quality loss is materially noticeable for agentic workflows: *“5-10% loss [is] a big deal”*, and one user said **Q6** is their minimum for agents because lower quants cause too much degradation in reasoning/tool-use reliability.
    - Users distinguished model scale/architecture effects: **30B dense models** reportedly suffer more visibly from aggressive quantization, while **large MoE models** at **Q5/Q6** can still perform well due to higher total parameter capacity and sparse activation behavior.
    - One user reported strong local results using **Q8_K_XL weight quantization with 16-bit KV cache** on **27B** and **35B A3B** models, suggesting that preserving KV precision and using high-bit weight quants can significantly improve output quality versus lower-bit setups.





## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code Power-User Workflows

  - **[I added a clause to Andrej Karpathy's 4 CLAUDE.MD clauses for Claude Code. It has been a game changer for me.](https://www.reddit.com/r/ClaudeAI/comments/1uc7izy/i_added_a_clause_to_andrej_karpathys_4_claudemd/)** (Activity: 2495): **The post proposes extending **Andrej Karpathy’s** `CLAUDE.md` rules for Claude Code—originally emphasizing *ask before assuming*, simplest implementation, avoiding unrelated edits, and explicit uncertainty—with a fifth directive encouraging Claude to suggest better long-term approaches rather than acting only as an obedient code generator. After feedback, the author revised the rules to add unattended-mode assumptions, distinguish simple vs. harder problems, surface design smells separately, and permit small low-risk experiments; reference video: [X/Twitter link](https://x.com/Ai_Tech_tool/status/2058140300502261784). Top technical suggestions include bounding “better approach” advice with tradeoff bullets and thresholds around irreversible work, security/data-loss risk, broad refactors, or wasted debugging; another commenter recommends requiring Claude to state the approach up front and list “what it makes harder later,” plus ending tasks with what it **did not** do.** Commenters broadly agree the added clause helps prevent over-obedient behavior, but warn that without constraints Claude may become an “annoying consultant” that challenges trivial requests. The main debate is how to encode execution modes: follow instructions, flag-and-wait on materially better alternatives, or stop when the requested path is unsafe or likely wrong.

    - Several commenters argued that a CLAUDE.md rule telling Claude to challenge the user needs explicit decision modes: **execute exactly**, **flag a better approach and wait**, or **stop/refuse when unsafe or likely wrong**. One proposed bounded wording: *“If you see a clearly better approach, say so before implementing. Explain the tradeoff in 2-4 bullets”*, with escalation only for issues like security risk, data loss, irreversible refactors, or hours of wasted debugging—not merely cleaner abstractions.
    - A recurring technical failure mode was Karpathy’s **“simplest solution first”** clause causing Claude Code to optimize for the nearest passing implementation, then create architectural dead ends across later files. One mitigation was to require Claude to state the approach in `2` lines before coding and list *“what it makes harder later”*, plus end each task with what it **did not do** to surface skipped edge cases.
    - One commenter described adding a CLAUDE.md instruction to identify when a task overlaps with **settled science or industry practice** so Claude suggests existing patterns instead of reinventing them. They reported this led to more useful implementation guidance such as *“this is how X company approaches it”* or combining data via a transform from recent research, e.g. a `2024` MIT-published method.

  - **[The $20 → $100 gap is pushing solo power users to split spend with OpenAI](https://www.reddit.com/r/ClaudeAI/comments/1ud388h/the_20_100_gap_is_pushing_solo_power_users_to/)** (Activity: 1068): **A solo Claude power user reports that **Claude Pro at `$20/mo` is insufficient for daily agent orchestration, Claude Code, analysis, and writing workloads**, while **Claude Max at `$100/mo` is a `5×` jump with no intermediate tier**. They currently split spend across **Claude Pro + ChatGPT/Codex at `$20 + $20`**, arguing that API-style usage credits are not equivalent because they deplete at token-metered rates; they propose a `$35–40/mo` “Pro 2x” plan with `2–3×` Pro allowance at the same app-consumption rate.** Comments were split between practical workarounds and pushback: one user argued that alternating Codex/GPT and Claude is technically useful because each catches bugs the other misses, while another suggested simply using two Claude Pro accounts. A harsher commenter argued that if Claude is core to a full-time business workflow, the user should pay for the `$100/mo` or business tier rather than expect a cheaper middle plan.

    - Several users discussed a practical multi-model workflow where **Claude/Opus** and **OpenAI GPT/Codex** are used as cross-checkers for coding tasks. One commenter said they “juggle back and forth between Codex and Claude” because each model catches bugs the other misses, suggesting power users may value complementary error profiles more than a single higher-tier subscription.
    - A few comments focused on pricing-tier gaps for solo technical users: one user said they prefer **Anthropic** over an enterprise **GitHub Copilot** subscription provided by work, but would only personally pay around `$40/month`, not `$100/month`. Another described oscillating between Claude Pro and higher-usage tiers depending on workload, indicating intermittent demand that does not fit neatly into fixed high-cost plans.


### 2. AI Writing and Restoration Failure Modes

  - **[I pulled ~90,000 Reddit posts about what makes writing "sound like AI" to determine the biggest AI-slop giveaways (Part 2)](https://www.reddit.com/r/ClaudeAI/comments/1ucpw87/i_pulled_90000_reddit_posts_about_what_makes/)** (Activity: 1081): **A Reddit analysis of `89,239` Arctic Shift posts across `47` subreddits filtered to `7,984` on-topic AI-writing-detection posts, with a `600`-post hand audit, ranks user-cited AI prose “tells”: **em dash** (`7.1%` of audited posts), flat sentence rhythm (`4.0%`), “not just X, it’s Y” constructions (`2.8%`), five-paragraph/“in conclusion” structure (`2.5%`), and diction clusters like “delve/leverage/seamless/tapestry” (`1.3%`). The author argues keyword detectors are misaligned with human judgments: common words like “however/thus/hence” matched frequently (`6.3%`) but were cited as tells `0%` of the time, while higher-signal traits such as rhythm, sycophancy, and “fluent but empty” prose are not captured by simple lexicon scans; data/scripts are published on [GitHub](https://github.com/JCarterJohnson/vibecoded-design-tells/tree/main/unslop-ai-text).** Top comments largely parody the listed tells by producing exaggerated AI-slop prose, while others push back that terms like “however” and punctuation like the em dash are normal human writing conventions. The main debate is whether these features are useful population-level signals or unfairly stigmatize careful writers, students, and non-native English speakers.

    - A commenter suggested the analysis may be time-sensitive and should be rerun on a newer slice, e.g. `2024–2026`, because LLM capabilities and possibly stylistic fingerprints have changed substantially since `2021`. The key methodological concern is whether older AI-writing markers still generalize to current model outputs, or whether the dataset mixes obsolete model behavior with contemporary “AI slop” signals.

  - **[I aged and restored a photo of myself](https://www.reddit.com/r/ChatGPT/comments/1ud6wuy/i_aged_and_restored_a_photo_of_myself/)** (Activity: 2745): **The image ([link](https://i.redd.it/rqbz1fkqhy8h1.png)) is a controlled test from the post *“I aged and restored a photo of myself”*: the author used **Gemini** to artificially age a known original photo, then asked **ChatGPT** to restore/colorize it. The result shows that the “restoration” is not a faithful reconstruction: ChatGPT hallucinated facial structure, hair/beard density, and apparent age, demonstrating that generative photo restoration can produce plausible but incorrect identities rather than recover ground truth.** Commenters largely treated this as evidence that AI photo restoration is misleading for historical/family photos, with one noting *“you’re a completely different person.”* Another comment extended the concern to face recognition/security systems, implying that similar identity drift could have real-world risks.

    - One commenter argued the result illustrates a core failure mode of AI aging/restoration: the model can synthesize a plausible older face while drifting identity enough that *“you're a completely different person.”* They connected this to risks in AI-assisted face recognition/security systems, where generative identity drift could undermine reliability.
    - Another commenter compared **Gemini**’s aged output with **NanoBananaPro**, saying NanoBananaPro was *“still way better for restorations”* after cropping the Gemini-aged photo back to the original framing. They noted Gemini’s aged image appeared to zoom out or alter framing, while the second restoration model had to infer and reconstruct substantial missing/detail information from the crop.


### 3. U.S. AI and Quantum Policy Pushes

  - **[President Trump orders a national effort to build a quantum computer capable of performing important scientific calculations](https://www.reddit.com/r/singularity/comments/1ucy9oj/president_trump_orders_a_national_effort_to_build/)** (Activity: 2937): **The post claims **President Trump** issued two quantum-focused orders: (1) a `5-year` national effort to build a quantum computer capable of meaningful scientific calculations, plus quantum sensors/networks; and (2) a mandate for federal agencies to migrate systems to **post-quantum cryptography (PQC)** by `2031`. The technically concrete element is the PQC migration: commenters note that useful fault-tolerant quantum computing remains a major uncertainty, while replacing quantum-vulnerable public-key cryptography is a long-lead engineering/security task that can begin before such machines exist.** Top comments were skeptical or cynical, with one suggesting the capability would be handed to the DoW/NSA and another joking about personal motives. The main substantive opinion was that the cryptography migration deadline is far more realistic and actionable than the quantum-computer build target.

    - Commenters highlighted that the **post-quantum cryptography migration deadline** is the most actionable part of the order: a useful, fault-tolerant quantum computer remains a major technical uncertainty, but replacing cryptographic systems vulnerable to Shor-style attacks requires long lead times across software, infrastructure, and standards compliance.
    - Several comments framed the likely strategic motivation as **cryptanalysis and national security**, specifically eventual capabilities to break deployed public-key encryption and cryptocurrency-related cryptography. The technical concern is less near-term quantum computing performance and more the need to harden systems before a future machine can attack RSA/ECC at scale.

  - **[Bernie Sanders unveils $7 trillion plan to give Americans control of AI industry](https://www.reddit.com/r/singularity/comments/1ucq463/bernie_sanders_unveils_7_trillion_plan_to_give/)** (Activity: 1505): **Sen. **Bernie Sanders** proposed a roughly **`$7T` AI sovereign wealth fund**, financed by a **one-time `50%` stock tax** on AI companies with at least **`$200M` in annual AI revenue**, according to [Ars Technica](https://arstechnica.com/tech-policy/2026/06/bernie-sanders-unveils-7-trillion-plan-to-give-americans-control-of-ai-industry/). The fund would issue estimated annual dividends of **over `$1,000` per American**, support public services, and create a Senate-confirmed **Independent Commission for Democratic AI** with voting-share authority to influence or block AI-company decisions deemed harmful to the public.** Top comments largely frame the bill as **politically dead on arrival**, but debate the underlying premise: if AI labs’ claims about AGI/ASI-driven productivity are true, commenters argue public ownership/UBI becomes economically necessary; if not, the industry is overpromising. Several commenters also view **UBI/Universal Basic Services** as inevitable to avoid large-scale unrest from automation-driven displacement.

    - One commenter critiques the proposed ownership threshold as creating a hard incentive boundary: if companies over `$200M` must transfer `50%` ownership, firms may deliberately cap growth near `$199M`, split entities, or offshore before crossing the threshold. They argue a sovereign wealth fund tied to AI upside could be more viable, but that a mandatory equity transfer would likely deter domestic AI development.
    - Another commenter frames the policy debate around ASI/RSI claims: if AI labs are correct that advanced AI will automate technological progress and wealth creation, then traditional capitalist incentives and concentrated private control become less necessary. Conversely, if firms reject public control, the commenter argues it implies the industry may be overpromising AI’s transformative capabilities.

  - **[Gen Z is the most anti-AI generation, yet remains its biggest consumer.](https://www.reddit.com/r/singularity/comments/1ucne6b/gen_z_is_the_most_antiai_generation_yet_remains/)** (Activity: 909): **The [image](https://i.redd.it/e4nijz88pu8h1.jpeg) is a non-meme text excerpt summarizing survey-style findings: **Gen Z adults ages 18–29 are reportedly the most wary of AI**, with `48%` saying AI will negatively affect society, while also being the **most frequent AI users**, with `66%` reporting usage. In context of the Yahoo article linked in the post, the technical significance is more about **AI adoption vs. risk perception** than model performance: younger users appear to be heavy consumers of AI tools despite stronger concern about societal impacts such as automation, misinformation, or loss of human control.** Comments frame the contradiction as partly generational polarization and partly exposure-driven: some argue Gen Z is highly online and therefore more exposed to anti-AI narratives, while others say the generation can simultaneously dislike AI’s implications and still use it pragmatically.

    - Several commenters framed Gen Z’s anti-AI sentiment as an adoption paradox rather than a technical rejection: they may object to AI socially or economically while still using it because it provides a perceived productivity advantage. One commenter specifically argued that avoiding AI could become a career disadvantage because it *“obviously makes you more productive,”* linking usage to job-market pressure and fear of displacement.


# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.