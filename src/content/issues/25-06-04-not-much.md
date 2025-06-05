---
id: MjAyNS0w
title: AI Engineer World's Fair Talks Day 1
date: '2025-06-05T05:44:39.731046Z'
description: >-
  **Mistral** launched a new **Code** project, and **Cursor** released version
  **1.0**. **Anthropic** improved **Claude Code** plans, while **ChatGPT**
  announced expanded connections. The day was dominated by **AIE** keynotes and
  tracks including **GraphRAG**, **RecSys**, and **Tiny Teams**. On Reddit,
  **Google** open-sourced the **DeepSearch** stack for building AI agents with
  **Gemini 2.5** and **LangGraph**, enabling flexible agent architectures and
  integration with local LLMs like **Gemma**. A new **Meta** paper analyzed
  language model memorization, showing GPT-style transformers store about
  **3.5–4 bits/parameter** and exploring the transition from memorization to
  generalization, with implications for **Mixture-of-Experts** models and
  quantization effects.
companies:
  - mistral
  - cursor
  - anthropic
  - openai
  - aie
  - google-deepmind
  - meta-ai-fair
models:
  - gemini-2.5
  - gemma
  - claude-code
topics:
  - agent-based-architecture
  - open-source
  - model-memorization
  - scaling-laws
  - quantization
  - mixture-of-experts
  - language-model-memorization
  - model-generalization
  - langgraph
  - model-architecture
people: []
---


**A happy day.**

> AI News for 6/3/2025-6/4/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 6571 messages) for you. Estimated reading time saved (at 200wpm): 503 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Mistral launched a [Code](https://mistral.ai/products/mistral-code) project and [Cursor went 1.0](https://www.cursor.com/en/changelog/1-0) and Anthropic [improved Claude Code plans](https://youtu.be/Yf_1w00qIKc?si=wDtapcnvLfnq5ip4) and ChatGPT [announced more connections](https://x.com/openai/status/1930319398897889707?s=46), but probably the day rightfully belonged to AIE in terms of the news cycle, with an [incredible set of keynotes bookending the MCP track](https://www.youtube.com/watch?v=U-fMsbY-kHY) for the main stream, and notable [GraphRAG](https://www.youtube.com/watch?v=RR5le0K4Wtw) and [RecSys](https://www.youtube.com/watch?v=3k4a0PemMu4) and [Tiny Teams](https://www.youtube.com/watch?v=xhKgTkzSmuQ) tracks streamed as well.

---

# AI Twitter Recap

pipeline down today sorry

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Recent Open-Source and Research Releases (Google DeepSearch, Meta Model Paper)

- [**Google opensources DeepSearch stack**](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) ([Score: 840, Comments: 77](https://www.reddit.com/r/LocalLLaMA/comments/1l27g8d/google_opensources_deepsearch_stack/)): **Google has open-sourced a new DeepSearch stack, accessible via the [gemini-fullstack-langgraph-quickstart repo](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart), which serves as a template to build full-stack AI agents with Gemini 2.5 and the LangGraph orchestration framework. While confirmed by the author as distinct from the actual Gemini user app backend, this release enables developers to experiment with agent-based architectures, can be integrated with other local LLMs (e.g., Gemma), and leverages Docker and modular project scaffolding for rapid prototyping. The stack is designed for flexibility but requires substitution if alternative models or search systems (other than Gemini and Google Search) are desired.** The comment discussion emphasizes that this release is more of a well-structured demo rather than a production-level backend (as used in Gemini App), highlights LangGraph's potential as an orchestrator, and references [LangManus](https://github.com/Darwin-lfl/langmanus/tree/main) as a more complex LangGraph-based system for advanced agent implementations.
    - The project open-sourced by Google is distinct from the Gemini App stack and is aimed at enabling developers to build agentic systems with Gemini, utilizing LangGraph. While it could be theoretically adapted to use Gemma instead of Gemini for the underlying model, users would need to swap out the search component for an alternative tool to maintain compatibility.
    - Although the demo showcases a clean architecture, it is not particularly complex or novel compared to more advanced LangGraph projects. For a more sophisticated and involved implementation, the commenter points to LangManus (https://github.com/Darwin-lfl/langmanus/tree/main) as an example, highlighting that the DeepSearch open-sourced stack serves primarily as an accessible end-to-end demonstration rather than pushing technical boundaries.
- [**New META Paper - How much do language models memorize?**](https://arxiv.org/abs/2505.24832) ([Score: 176, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1l2gvar/new_meta_paper_how_much_do_language_models/)): **The discussed Meta paper ([arXiv:2505.24832](https://arxiv.org/abs/2505.24832)) proposes a rigorous method to estimate language model memorization, empirically showing that GPT-style transformers consistently store about 3.5–4 bits/parameter (e.g., 3.51 for bfloat16, 3.83 for float32), and that storage capacity does not scale linearly with increased precision. The work delineates the transition from memorization to generalization ("grokking") occurs as model capacity is saturated and double descent initiates when dataset information content surpasses storage limits. They further introduce scaling laws derived from hundreds of trained transformers (500K–1.5B params) relating model size and dataset volume to membership inference attack success, finding generalization, not rote memorization, responsible for extraction when datasets are large and deduped.** Commenters note interest in how these findings extend to Mixture-of-Expert (MoE) models and the impact of quantization (under 3.5 bits/param) or low-precision/QAT training on memorization and generalization boundaries. There is speculation that sub-3.5 bit quantization could explain performance drops witnessed in practice, with curiosity about whether novel architectures like BitNet alter these fundamental capacity limits.
    - The authors empirically estimate that GPT-family transformers can store between 3.5 and 4 bits of information per parameter (e.g., 3.51 bits/parameter for bfloat16, 3.83 for float32), while noting that increasing precision does not linearly increase storage capacity, implying non-trivial use of model capacity beyond raw bit-for-bit memorization.
    - The paper links model memorization and generalization to double descent: memorization dominates until capacity is saturated, after which generalization emerges via 'grokking.' Double descent reportedly occurs when dataset information (in bits) exceeds model storage, compelling information sharing and increasing generalization.
    - Follow-up discussion raises questions about whether these findings extend to Mixture-of-Experts (MoE) architectures, how quantization-aware training (QAT) or lower precision affect storage/memorization, and speculates that models quantized below ~3.5 bits may fundamentally degrade performance in GPT-style models, with open questions about alternative architectures like BitNet.

### 2. LLM and Vision Multimodal Model Announcements and Benchmarks

- [**nvidia/Nemotron-Research-Reasoning-Qwen-1.5B · Hugging Face**](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) ([Score: 133, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1l2820t/nvidianemotronresearchreasoningqwen15b_hugging/)): **Nvidia's [Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) is a 1.5B-parameter open-weight model targeting complex reasoning (math, code, STEM, logic) trained using the novel Prolonged Reinforcement Learning (ProRL) approach based on Group Relative Policy Optimization (GRPO). ProRL introduces key RL stabilization techniques—entropy collapse mitigation, decoupled clip & dynamic sampling (DAPO), KL regularization, and reference policy reset—enabling >2k RL steps and broader generalization. The model significantly outperforms DeepSeek-R1-1.5B and matches/exceeds DeepSeek-R1-7B, achieving average pass@1 improvements of** `14.7% (math)`**,** `13.9% (coding)`**,** `54.8% (logic)`**,** `25.1% (STEM)`**, and** `18.1% (instruction-following)`**.** Commenters highlight the trend toward small, efficient open-source reasoning models for edge and mobile devices and note ProRL's RL innovations. Criticism focuses on Nvidia's restrictive CC-BY-NC-4.0 license, which limits commercial usage despite strong technical results.
    - The Nemotron-Research-Reasoning-Qwen-1.5B model leverages the ProRL (Prolonged Reinforcement Learning) algorithm, which enables extended RL training (more than 2k steps) and incorporates Group Relative Policy Optimization (GRPO). Key technical innovations include entropy collapse mitigation, decoupled clip and dynamic sampling policy optimization (DAPO), KL regularization, and reference policy resets. These methods purportedly lead to marked generalization improvements across diverse reasoning tasks, including math, code, STEM, and logic puzzles.
    - Technical benchmarks shared by the uploader indicate that this 1.5B parameter model claims substantial improvements over the DeepSeek-R1-1.5B, with reported gains of pass@1 by `14.7%` (math), `13.9%` (coding), `54.8%` (logic puzzles), `25.1%` (STEM), and `18.1%` (instruction following). Interestingly, it is asserted to match or even surpass DeepSeek-R1-7B's performance on a diverse range of tasks, which is unusual for models at the 1.5B parameter scale.
    - The model has been released in GGUF format with quantized options (q4, q8, f16) to facilitate local inference on resource-constrained hardware. However, technical discussion raises concerns that the restrictive CC non-commercial license and ambiguous licensing terms may significantly hinder commercial or broader real-world adoption, in spite of the technical merits.
- [**Vision Language Models are Biased**](https://vlmsarebiased.github.io/) ([Score: 100, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1l2b83p/vision_language_models_are_biased/)): **State-of-the-art Vision Language Models (VLMs) achieve nearly perfect accuracy on canonical visual tasks (e.g., counting legs on typical animals or stripes on standardized logos), but their accuracy drops drastically to ~17% on counterfactual or altered scenarios, as measured by the VLMBias benchmark. Detailed analysis shows models overwhelmingly rely on memorized priors rather than actual visual input, with 75.7% of errors reflecting stereotypical knowledge rather than ambiguity, and explicit bias-alleviation prompts are largely ineffective. [Original source](https://vlmsarebiased.github.io/) provides dataset and methodology across seven domains, revealing VLMs' inability to reason visually outside training distribution.** Commenters debate whether these findings are inherently surprising, given all AI systems reflect biases in their data and architectures, and note similar issues observed in LLM log probabilities."
    - The top Vision Language Models can achieve near-perfect accuracy (up to 100%) in counting tasks involving familiar subjects (like the 3 stripes on an Adidas logo or dogs with 4 legs), but their accuracy drops dramatically to around 17% when encountering counterfactual or out-of-distribution images (such as a 4-striped Adidas logo or a dog with 5 legs), highlighting a severe limitation in generalization.
    - This failure mode is analogous to how vision models often miscount fingers when presented with images of hands that have more or fewer than the standard five fingers, further demonstrating that state-of-the-art models are highly sensitive to distributional shifts and struggle with compositional reasoning in unfamiliar settings.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI Model and Feature Releases (VEO 3, Sora, Chroma, Codex, ChatGPT Memory/Research)

- [**Ulianopolis City Hall in Brazil made a complete commercial with VEO 3, spending only R$300 reais ($52 dollars) in VEO 3 credits**](https://v.redd.it/36cgd4rvjp4f1) ([Score: 1047, Comments: 196](https://www.reddit.com/r/singularity/comments/1l2azl6/ulianopolis_city_hall_in_brazil_made_a_complete/)): **Ulianopolis City Hall (Brazil) created a 1-minute, professional-grade commercial entirely with Google's Veo 3 generative video AI, incurring only R$300 (~$52 USD) in AI credits—an extreme reduction compared to traditional local production costs (>R$100,000/~$17,500 USD). The workflow replaced nearly all conventional production functions—direction, scripting, filming, editing, post-processing, and more—relying exclusively on text-to-video generative capabilities. See [original Reddit post](https://v.redd.it/36cgd4rvjp4f1) and creator's [Instagram](https://www.instagram.com/renato_lferreira/).** Commenters note this as a major disruption to traditional commercial production, suggesting advertising and creative agencies are under threat and remarking on the impact of seeing high-quality, native-language AI output, underscoring the imminent shift in media production workflows.
    - A key technical point is the drastic cost reduction of commercial production using VEO 3, with a professional-level spot produced for R$300 (~$52), drastically undercutting traditional agency costs while allowing iterative improvements through quick AI re-generation and editing.
    - Native language synthesis capabilities of VEO 3 are highlighted as particularly impressive. Users note accurate Brazilian Portuguese output, including *native accents and natural linguistic expressions*, which traditionally have been challenging for AI generative models and make the results much more robust and market-ready for local audiences.
- [**Microsoft brings free Sora AI video generation to Bing**](https://www.windowscentral.com/microsoft/microsoft-bing-video-creator-sora-ai-generator-free-announcement) ([Score: 245, Comments: 51](https://www.reddit.com/r/singularity/comments/1l264o6/microsoft_brings_free_sora_ai_video_generation_to/)): **Microsoft has integrated OpenAI's Sora AI video generation model into the Bing app under the branding 'Bing Video Creator', providing free access to generative video content. The solution does not feature a dedicated Sora app or ChatGPT integration yet, and initial user experiences note both the ability to generate detailed, animated content as well as encountering strict safety/request blocking, reflecting tight content moderation.** Users debate the practicality versus restrictiveness of current implementation: while novel creative possibilities are acknowledged, some criticize the overly aggressive safety filters, limiting utilitarian or experimental use cases.
    - Several users compare Microsoft's Sora (available via Bing Video Creator) to Google's Veo3, with the consensus indicating that Veo3 delivers superior results in video generation. The implication is that Sora currently lags behind Veo3 in terms of video quality and model capability, making it a weaker competitor in this space.
    - A technical limitation noted by a commenter is Sora's aggressive safety filters, which result in many requests being blocked, reducing its usability and flexibility for content generation compared to less restrictive alternatives.
    - There is mention of the limited integration for Sora, as it's currently only available through the Bing app and not as a stand-alone application or within the ChatGPT app, which could hinder broader adoption and utility for developers and advanced users.
- [**OpenAI is preparing to release 2 new models with native audio support**](https://x.com/testingcatalog/status/1929949017472930181?s=46) ([Score: 229, Comments: 31](https://www.reddit.com/r/singularity/comments/1l2htv5/openai_is_preparing_to_release_2_new_models_with/)): **OpenAI is reportedly set to release two GPT-4o-based models—'gpt-4o-audio-preview-2025-06-03' and 'gpt-4o-realtime-preview-2025-06-03'—featuring native audio processing instead of relying on external speech-to-text or text-to-speech modules. This suggests integrated, end-to-end audio I/O capabilities within the GPT-4o architecture, potentially enabling low-latency audio interactions and more seamless assistant-like functionalities (see early coverage from [TestingCatalog News](https://x.com/testingcatalog/status/1929949017472930181?s=46)).** Commenters question what distinguishes 'native audio' versus previous GPT-4o implementations, noting that GPT-4o already demonstrated real-time audio in presentations; there is debate if this release brings functional advances or formalizes existing preview features.
    - Several users are seeking clarification on what 'native audio' entails, questioning whether it refers to models like GPT-4o which already feature audio support. There's technical uncertainty about whether the upcoming models offer fundamentally new architecture for direct audio processing, or simply expose existing capabilities in a novel API or format.
    - One commenter speculates that the new release may be related to the audio assistant functionality demonstrated with GPT-4o over a year ago, suggesting that the new models could formalize or enhance those real-time speech interaction capabilities within the API ecosystem.
    - There is a technical proposition that the scope of 'native audio' could extend beyond audio to video processing as a continuous bitstream, indicating potential evolution toward unified, multimodal bitstream handling for more natural input/output modalities.
- [**Everything to Look forward to this summer**](https://i.redd.it/ou0k2gkx2s4f1.jpeg) ([Score: 216, Comments: 59](https://www.reddit.com/r/singularity/comments/1l2nmsr/everything_to_look_forward_to_this_summer/)): **The image is a timeline-style infographic listing major anticipated AI model and technology project releases (such as GPT-5) scheduled for summer 2024 (June–August) and was recently featured in Peter Diamandis's YouTube content. The graphic, attributed to @chatgpt21, aggregates various upcoming launches, illustrating the accelerated pace and density of major announcements in the current AI landscape.** Top comments express skepticism about the lack of hype surrounding GPT-5's reportedly imminent release, and note that technology iteration cycles have become so rapid that such timelines quickly become outdated.
    - Comments highlight the accelerated release cadence for GPT models, with some users noting that timelines between GPT-4 and the rumored GPT-5 are much shorter than previous cycles, questioning the value and accuracy of predictive release charts as a result.
    - One commenter questions whether GPT-5’s anticipated launch date is substantiated by official announcements versus being mere speculation, reflecting ongoing uncertainty in the community regarding the reliability of upcoming model leaks and roadmaps.
    - There's an expressed perception that GPT-4 has become significantly less capable or 'stupid' in comparison to expectations for GPT-5, suggesting end-users are noticing or believing in a strong qualitative gap between current and yet-to-be-released LLMs.
- [**Memory is now available to free users!!!**](https://i.redd.it/jy18jpn0nq4f1.png) ([Score: 235, Comments: 57](https://www.reddit.com/r/OpenAI/comments/1l2g8es/memory_is_now_available_to_free_users/)): **The image is an FAQ update announcing that ChatGPT's Memory feature is now rolling out to free users as of June 3, 2025. This allows ChatGPT to reference users' recent conversations to provide more relevant responses. In certain European regions, users must manually enable this feature, while elsewhere it is activated by default; users retain control to disable memory functionality at any time.** Technical discussion in the comments focuses on privacy and usability: paid users point out that subscription allows them to opt out of data being used for model training, questioning OpenAI's compliance. Others critique the memory feature, noting that automatic saving can result in irrelevant or outdated data being retained, and express a desire for more granular, manual memory controls.
    - Several commenters discuss how ChatGPT's "Memory" feature uses various aspects of your chat histories as an internal knowledge base by appending relevant memory snippets to your prompts, which can affect both the accuracy of responses and introduce biases based on your prior conversations. Technical users note this can worsen truthfulness or inject outdated/context-specific assumptions over time.
    - A critical point brought up is that user control over memory is limited: the current implementation saves information automatically and sometimes stores irrelevant or outdated data. There is expressed demand for manual memory management where users could explicitly add or curate what the model should remember, potentially improving accuracy and relevance.
    - Doubts are raised on whether the memory function meaningfully improves over previous mechanisms. Some users observe the model is still prone to "confidently" inventing details about past conversations rather than reliably recalling specifics, suggesting the memory integration or retention logic may not yet be robust for precise long-term reference.
- [**Codex rolling out to Plus users**](https://www.reddit.com/r/OpenAI/comments/1l2kd42/codex_rolling_out_to_plus_users/) ([Score: 107, Comments: 31](https://www.reddit.com/r/OpenAI/comments/1l2kd42/codex_rolling_out_to_plus_users/)): **Codex is now being gradually enabled for ChatGPT Plus users, as evidenced by user reports confirming access via the URL https://chatgpt.com/codex. Codex is OpenAI's code-focused model family, optimized for natural language to code and code generation tasks. The original post and comments do not specify updated usage limits or technical restrictions for Plus users.** Commenters are inquiring about technical constraints (such as limits) and the specific use cases or capabilities of Codex within the Plus tier; no definitive answers provided.
    - A user inquires about the usage limits for Codex as it rolls out to Plus users, indicating that details about API call restrictions, rate limits, or feature limitations have not yet been published or are unclear. This is an important point for developers or technical users who might want to integrate or automate workflows with Codex, as understanding these limits is critical for scalability and reliability of their implementations.
    - One comment expresses an expectation that Codex-level capabilities would be tied to the release of GPT-5, speculating that significant new functionalities or broader toolset integration may be reserved for future model iterations. This indirectly points to a technical anticipation about the evolution of OpenAI's model ecosystem, suggesting that further advancements in code generation or API capabilities could be aligned with major architectural updates.
    - Another user asks what Codex is for, which hints that there may still be confusion among some technical users regarding Codex's applications—primarily code generation, API usage, and potentially integration with products like GitHub Copilot or other automation tools. This highlights a need for clearer communication regarding Codex's purpose and use-cases for the technical community.
- [**Research is Now Available on Pro Plans!!**](https://i.redd.it/b1x3zdboxq4f1.png) ([Score: 135, Comments: 39](https://www.reddit.com/r/ClaudeAI/comments/1l2hsjw/research_is_now_available_on_pro_plans/)): **The image demonstrates that Anthropic has introduced a 'Research' feature, tagged as 'BETA', to their Claude Pro plan, as showcased by the new icon in the Claude interface. This feature appears to provide integrated research assistance, with users able to input queries and receive insights or synthesized information rather than direct answers. The interface update indicates a push towards more advanced, research-focused AI assistance available to paying users.** A user noted the research tool offered thoughtful, detailed guidance rather than just answers, improving their work through actionable insights. Another commenter questioned how this feature compares to similar offerings from other AI companies, suggesting potential benchmarking interest.
    - One user noted that the research mode automatically deployed 3-4 subagents to tackle a query from multiple angles using a depth-first approach, a technical implementation detail geared towards thoroughness and exploratory coverage.
    - Another comment pointed out that the tool cited "300 sources and counting" on a particular research task, and questioned whether this is significantly higher than the typical source counts offered by OpenAI's GPT and Perplexity, suggesting superior breadth in information aggregation.
    - A technical comparison was made between major models: Claude Max and SuperGrok were preferred for research quality, with comments that Gemini provides large volumes of information but less refinement, and OpenAI's responses feel too clinical, highlighting differing approaches to research output among major AI services.
- [**Chroma v34 is here in two versions**](https://www.reddit.com/r/StableDiffusion/comments/1l2asij/chroma_v34_is_here_in_two_versions/) ([Score: 170, Comments: 64](https://www.reddit.com/r/StableDiffusion/comments/1l2asij/chroma_v34_is_here_in_two_versions/)): **Chroma v34 has been released in two versions, with the distinction that the '-detailed release' offers higher image resolution compared to the standard model ([Hugging Face link](https://huggingface.co/lodestones/Chroma/tree/main)). Community commentary highlights ongoing improvements in detail and flexibility, especially for uncensored and non-photographic art generation. Early tests using LoRA adapters on the detail-calibrated version show incremental quality enhancements.** Commenters argue Chroma is quickly becoming a leading base model and a strong alternative to Flux, particularly for non-photographic and customizable art generation tasks.
    - There are two Chroma v34 releases: a regular version and a detail-calibrated version, with the latter specifically trained on high-resolution data. Users have successfully generated images at native resolutions up to `2048x2048`, with reports of "somewhat decent results" at these sizes.
    - Chroma v34 distinguishes itself as an uncensored model without a bias toward photographic style, which allows it to perform well across various types of artwork, including both photographic and non-photographic outputs. This addresses a limitation found in many current AI models that are overly tuned to photography datasets.
    - There are multiple references to using LoRA (Low-Rank Adaptation) techniques with Chroma v34, including successful application and improved image detail. This suggests ease of integration with community tools and a rapidly maturing ecosystem similar to that previously seen with models like SD14 and emerging alternatives such as Flux.

### 2. Concerns About AI-Driven Economic Inequality and Job Loss

- [**We need to do everything in our power to prevent AI from becoming a luxury**](https://www.reddit.com/r/singularity/comments/1l2j6u1/we_need_to_do_everything_in_our_power_to_prevent/) ([Score: 222, Comments: 94](https://www.reddit.com/r/singularity/comments/1l2j6u1/we_need_to_do_everything_in_our_power_to_prevent/)): **The post highlights the trend of large AI vendors like OpenAI, Anthropic, and Google shifting powerful LLMs behind high monthly paywalls (OpenAI at $200/mo, Anthropic at $100/mo, Google at $130/mo), while open-source LLMs (e.g., from DeepSeek, Qwen) are increasing in capability but also resource requirements—potentially pricing out typical users from self-hosting as model sizes and inference costs rise. The author raises the risk that both hardware constraints (high-end GPUs) and potential privatization by competitive open-source labs could widen the capability gap between premium and generally accessible AI, with risk of severe socio-economic stratification as AGI approaches.** The top technical comments debate inevitability vs. policy intervention: some argue high costs are inseparable from cutting-edge AI, and that only socializing these costs (e.g., public AI infrastructure) would maintain access, while others claim lower tiers/older models remain generally available and highlight AI's economic nature as akin to utilities (e.g., electricity); some challenge the notion of exclusivity given simultaneous existence of open/free and paid AI tiers.
    - Multiple commenters emphasize the *substantial operational and developmental cost* of state-of-the-art AI, noting that currently, models require expensive compute infrastructure and energy. Competition among major labs (OpenAI, Google, etc.) keeps prices high, with reports that providers are sometimes operating at a loss (e.g., OpenAI's Pro plans) and needing to adjust pricing upward (Google's recent $250 increase).
    - There is discussion about pricing stratification: while the highest-performing or newest models are expensive, older or less capable model versions are often offered at lower price points or even free. This is compared to traditional technology markets, where early access to premium products costs more, but broader access increases as technology matures and scales.
    - The idea of 'socializing' AI—making access a public utility managed at a societal scale—is presented as a way to ensure equitable access despite high costs, but this approach does not reduce the underlying expenses. Until major technological breakthroughs (e.g., cheap fusion energy or fully automated production), these costs are seen as intractable and likely to keep AI as a comparatively expensive resource.
- [**Dario Amodei worries that due to AI job losses, ordinary people will lose their economic leverage, which breaks democracy and leads to severe concentration of power: "We need to be raising the alarms. We can prevent it, but not by just saying 'everything's gonna be OK'."**](https://v.redd.it/ba6dzs1grq4f1) ([Score: 1378, Comments: 364](https://www.reddit.com/r/singularity/comments/1l2gwo1/dario_amodei_worries_that_due_to_ai_job_losses/)): **Dario Amodei (CEO, Anthropic) expresses concerns that AI-driven job losses risk eroding the economic leverage of workers, potentially undermining democracy and leading to a dangerous concentration of power. He emphasizes proactive intervention beyond complacency, stating that *'We can prevent it, but not by just saying everything's gonna be OK.'* [Source](https://www.anthropic.com/people/dario-amodei).** Commentary highlights skepticism about political will or public reaction, noting the gradual ('boiling frog') nature of AI job displacement, which diminishes urgency and thus delays policy intervention until the effects are unavoidable.
    - Quick-Albatross-9204 highlights the gradual displacement of jobs by AI, referencing the 'boiling frog' effect: because job losses are incremental rather than immediate, broader society and policymakers may not perceive the urgency or scale of potential economic impact until it is too late to act effectively. This underscores the need for real-time labor displacement monitoring and adaptive policy frameworks.
- [**Former OpenAI Head of AGI Readiness: "By 2027, almost every economically valuable task that can be done on a computer will be done more effectively and cheaply by computers."**](https://i.redd.it/l0cd9s4yar4f1.png) ([Score: 1026, Comments: 356](https://www.reddit.com/r/singularity/comments/1l2jun4/former_openai_head_of_agi_readiness_by_2027/)): **The image is a tweet by Miles Brundage, former OpenAI Head of AGI Readiness, claiming that by 2027, almost every economically valuable task that can be performed on a computer will be doable more effectively and cheaply by computers—though he adds caveats about judgment context and deployment versus capability. This view represents a strong, timeline-specific assertion of AI's capability progress, notably around automation of white collar/knowledge work, if outputs are evaluated purely on technical merit and not social or human-attribution values. Brundage clarifies that his statement refers to the capability being possible, not necessarily that automation will be universal or deployed everywhere.** Commenters raise doubts about organizational readiness and data infrastructure (arguing most workplaces would struggle to format their data programmatically even by 2027). Others push back, noting the complexity of actual jobs versus technical feasibility, and raise concerns over societal implications (UBI, automation tax), citing the scale of potential white collar disruption.
    - Fenristor argues that organizational and data infrastructure constraints will significantly delay AI automation, noting that even with major effort, most companies would be unable to transition all their internal data to programmatic, machine-readable formats by 2027. This highlights a fundamental technical and logistical bottleneck to the rapid replacement of knowledge work by AI.
    - ryanhiga2019 raises a technical limitation of current large language models (LLMs), pointing out that persistent hallucinations (i.e., factual errors or fabrications) restrict the reliability and scalability of LLMs for economically critical tasks. This suggests that major advances in LLM accuracy and trustworthiness are required before widespread replacement of knowledge work is feasible.

### 3. Personal Experiences Using AI for Real World Tasks

- [**ChatGPT summaries of medical visits are amazing**](https://www.reddit.com/r/ChatGPT/comments/1l2ojdb/chatgpt_summaries_of_medical_visits_are_amazing/) ([Score: 2520, Comments: 211](https://www.reddit.com/r/ChatGPT/comments/1l2ojdb/chatgpt_summaries_of_medical_visits_are_amazing/)): **A user describes using ChatGPT to process audio recordings and transcripts of hospital visits, translating complex medical conversations into accessible summaries for remote family members. The workflow reportedly involved recording conversations (with consent), transcribing audio to text, and prompting ChatGPT for readable, lay-friendly medical summaries. Commenters confirmed similar use-cases, e.g., summarizing MyChart records for cancer diagnosis communication; accuracy was considered high as long as outputs were based on official medical records, with some users recommending double-checking outputs with Google. The workflow could be improved by using Google Docs for static, comment-enabled sharing.** Key discussion points include: reliability of ChatGPT when summarizing direct medical documentation versus answering unanchored queries (reducing hallucination risk), and practical workflow tips like leveraging collaborative document platforms for more effective information dissemination and feedback.
    - Several users describe using ChatGPT to translate medical visit records and test results (such as those from MyChart or MRI reports) into layman-accessible summaries. The process generally involves extracting report data, anonymizing it by removing identifying information, and pasting it into ChatGPT, which can both preserve original formatting and generate section-by-section plain language explanations.
    - Attention is given to double-checking ChatGPT summaries by cross-referencing output with Google or other sources for factual accuracy, which helps mitigate the risk of hallucinations or errors, though users report high reliability when the input is specific medical documentation.
    - Workflow optimization suggestions include storing generated summaries in collaborative documents like Google Docs for static sharing and collective commenting, or asking ChatGPT to generate lists of questions to bring to medical consultations—enhancing interactivity and usefulness for non-technical family members.
- [**I Tried Replacing Myself With AI for a Week. Here’s What Actually Happened**](https://www.reddit.com/r/ChatGPT/comments/1l2gbz9/i_tried_replacing_myself_with_ai_for_a_week_heres/) ([Score: 679, Comments: 111](https://www.reddit.com/r/ChatGPT/comments/1l2gbz9/i_tried_replacing_myself_with_ai_for_a_week_heres/)): **The OP replaced their operations assistant work at a logistics company with AI tools over a week: ChatGPT-4 for email/SOP creation, Blackbox AI for document summarization, Notion AI for meeting notes, and Zapier+GPT for task automation. AI performed best with structured/repetitive tasks (SOPs, templated emails), but required significant user oversight and context injection to avoid generic or robotic outputs. The experiment realized a time savings of ~12 hours, but highlighted that human oversight in orchestrating and contextualizing AI workflows remains essential.** No substantive technical debates emerged in the top comments; the discussion was mostly non-technical banter and meta-commentary.
    - A commenter draws a parallel between the article's theme and trends in software development, pointing out that while coders may be increasingly replaced or assisted by AI, there is still ongoing demand for software engineers with broader responsibilities or system-level expertise. This suggests that automation is shifting the required skill level upward rather than eliminating roles entirely.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: The Model Frontier: Launches, Leaks, and Lingering Questions**

- **Gemini 2.5 Pro and "Goldmane" Flex Muscles, o3 Pro Plays Coy**: Google's **Gemini 2.5 Pro** nears general availability, with its "Goldmane" version impressing on the [Aider webdev benchmark](https://aider.chat/), while OpenAI's anticipated **o3 Pro** remains elusive, with early reports of it being *"ass"* and having a meager **500 LOC** code generation limit. Meanwhile, Google's mystery **"Kingfall"** model, possibly **DeepThink** with a **65k** context window, made a brief, "confidential" appearance on AI Studio, sparking curiosity and job security concerns for some Googler.
- **Japan Unleashes Shisa-v2 405B, Outperforming Giants?**: The **Shisa-v2 405B** model, hailed as Japan's most powerful, launched with claims of **GPT-4/Deepseek-comparable** performance in Japanese and English, inviting users to test it at [chat.shisa.ai](https://chat.shisa.ai/). A detailed tech report for this H200-node-powered beast is eagerly awaited on Arxiv.
- **Qwen Challenges Deepseek, Perplexity Pro Users Grumble**: The [Qwen model from Alibaba Cloud](https://chat.qwen.ai/) is gaining traction for surpassing **Deepseek R1** in reasoning with its 1M context window, and Perplexity might tap it for deep research. This comes as **Perplexity Pro** users voice frustration over small context limits (5-10 sources) and poor memory, one user lamenting, *"Yes, you constantly have to remind it what you're asking about."*

**Theme 2: Agentic AI Ascends: Frameworks, Features, and Frustrations**

- **OpenAI and LlamaIndex Supercharge Agent Builders**: OpenAI rolled out an [Agents SDK in TypeScript, a RealtimeAgent feature, and Traces support](https://x.com/OpenAIDevs/status/1929950012160790876), empowering developers to build more reliable agents, as showcased by early testers like Perplexity and Intercom. LlamaIndex offers a [hands-on Colab for building multi-agent financial report chatbots](https://twitter.com/llama_index/status/1930051898247393729) using agentic RAG and 10-K filings.
- **Elasticsearch Agentic Flows Get Complex, Cursor Unveils RIPER**: Engineers are tackling complex agentic flows, like one using **gpt-41-mini** for multi-step **Elasticsearch DSL query generation** ([see diagram](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png?ex=684204b7&is=6840b337&hm=eadc26c0018f544fedd3a9e8e6407dcf5f02301750f66ef5732067f94b94beff&)), while the new **CursorRIPER framework** aims to guide agent behavior with rules, memory, and a tech context file to keep projects on track. Meanwhile, **HTNs (Hierarchical Task Networks)** are being explored for fine-tuning LLM agents in ReACT format for better structured interactions.
- **MCP vs. A2A: The Great Agent Protocol Debate**: The **MCP (Meta-agent Communication Protocol)** sees discussion for monetization via API keys and context management across agents, with state transfer guidance available at [fast-agent.ai](https://fast-agent.ai/mcp/state_transfer/). However, Google's **A2A (Agent-to-Agent) framework** ([GitHub repo](https://github.com/google/A2A/)) emerges as a contender, with some developers preferring the **A2A spec** for multi-agent systems and leveraging tools like **pydantic-ai-slim** ([pydantic-ai docs](https://ai.pydantic.dev/install/)) with its handy `.to_a2a()` method.

**Theme 3: Under the Hood: GPU Optimizations, Hardware Quirks, and Performance Puzzles**

- **Blackwell Benchmarks Dazzle, MI300X Profiling Perplexes**: NVIDIA's Blackwell architecture shows stunning performance in [Cutlass samples](https://github.com/NVIDIA/cutlass/tree/main/examples/70_blackwell_gemm), with **NVFP4** hitting **3.09 PetaFLOPS/s**, though its **MXFP8/BF16** performance (**0.23 PetaFLOPS/s**) raised eyebrows. Meanwhile, AMD **MI300X** users struggle with `rocprof` errors when reading **L2CacheHit** on **gfx942**, despite [ROCm documentation](https://github.com/ROCm/ROCm/blob/develop/docs/conceptual/gpu-arch/mi300-mi200-performance-counters.rst) suggesting support, and note low L2 cache hit rates correlating with low **MfmaUtil** scores.
- **CUDA and ROCm Developers Wrestle Kernels and Tools**: Developers dive deep into GPU programming, discussing CUDA barrier states like `__syncthreads()` versus `bar.sync` ([NVIDIA's Volta blog on programmability](https://developer.nvidia.com/blog/volta-new-programmability-features)), and leveraging `cuda::pipeline` from libcu++ for producer/consumer schemes ([CUDA Zone resource](https://developer.nvidia.com/cuda-zone)). On the AMD side, Snektron shared his [AMD FP8 matrix multiplication kernel solution](https://github.com/Snektron/gpumode-amd-fp8-mm/blob/main/solution.hip) and a [detailed writeup](https://akashkarnatak.github.io/amd-challenge/) exploring MI300 coalescing.
- **Tinygrad and Torchtune Users Chase Performance, Battle Bugs**: **Tinygrad** users grapple with removing NumPy dependencies only to see operations offloaded to the GPU, deciphering overwhelming `DEBUG=2` outputs, and tackling significantly slow LSTM layers. **Torchtune** developers are working through an [Iterable Dataset Refactoring RFC (#2785)](https://github.com/pytorch/torchtune/pull/2785) and encountering `DeviceMesh` errors when testing optimizers like SGD and Adafactor beyond AdamW in distributed settings.

**Theme 4: Bleeding Edge Research: Finetuning Breakthroughs, Semantic Threats, and Novel Architectures**

- **Parameter-Efficient Finetuning Promises Huge Gains**: A novel parameter-efficient finetuning method claims **~4x more knowledge uptake** and **30% less catastrophic forgetting** compared to full finetuning and LoRA, using fewer parameters. This technique is particularly promising for adapting models to new domains and efficiently embedding specific knowledge without overwriting existing capabilities.
- **World Models Face "Semantic Virus" Infection**: A [new paper on general agents and world models](https://arxiv.org/pdf/2506.01622) posits that a **"Semantic Virus"** can exploit vulnerabilities in LLM world models by "infecting" reasoning paths if the model has "holes" or disconnected areas. The virus reportedly hijacks the world model's current activation within the context window rather than rewriting the base model itself.
- **Self-Play and Responsible AI Push LLM Boundaries**: Researchers explore innovative training paradigms, with one paper on [*Evolving LLMs Through Text-Based Self-Play*](https://ai.vixra.org/abs/2506.0018) seeking community feedback on achieving emergent performance. Simultaneously, IBM introduced an [open-source Responsible Prompting API](https://github.com/IBM/responsible-prompting-api) ([accompanying paper](https://arxiv.org/abs/2504.08757), [HF Spaces demo](https://huggingface.co/spaces/santanavagner/responsible-prompting-demo)) to guide users toward more accurate and ethical LLM outputs pre-inference.

**Theme 5: Ecosystem Evolution: API Shakeups, Community Tools, and Developer Resources**

- **API Turmoil: Anthropic Cuts Capacity, OpenAI TTS Pricing Confuses**: Anthropic abruptly cut most **Claude 3.x model capacity** with less than five days' notice, impacting services like Windsurf ([see _mohansolo's tweet](https://x.com/_mohansolo/status/1930034960385356174)), while ai.engineer offers [BYOK options and an improved agentic harness](https://x.com/kevinhou22/status/1930401320210706802) as a response. Users also questioned why OpenAI's **gpt-4o-mini-tts** costs significantly more than **tts-1**, despite listed prices, pointing to potential gotchas discussed on the [OpenAI community forum](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616).
- **Dev Tooling Flourishes: Almanacs, Chat Interfaces, and Interpretability Kits**: Modal Labs launched [The LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213), providing thousands of inference benchmarks, while **GitHub Chat** offers a new way to interact with repositories by changing `github.com` to `githubchat.ai` (e.g., https://githubchat.ai/blueraai/universal-intelligence). The **Prisma** toolkit for vision/video interpretability, now with [Hugging Face model support](https://huggingface.co/) and [100+ model circuit-style code examples](https://x.com/soniajoseph_/status/1930286144471646252), gained recognition with an Oral presentation at CVPR 2025.
- **Open Source Agents Rebrand, Data Policies Stir Debate**: **OpenManus** rebranded to **agenticSeek** ([GitHub repo](https://github.com/Fosowl/agenticSeek)), possibly due to copyright concerns, mirroring OpenDevin's change to OpenHands. Meanwhile, an [ArsTechnica article reporting OpenAI is compelled to save all ChatGPT logs](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare), including deleted chats and API data, sparked privacy discussions among engineers.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Hosts Reddit AMA**: **Aravind (CEO), Denis (CTO), Tony (VP Engineering), Weihua (Member of Technical Staff), and Tyler Tate (Product)** hosted a live **Reddit AMA** to discuss **Perplexity Labs** at 10am PT with a [link to the Reddit AMA](https://www.reddit.com/r/perplexity_ai/comments/1l39wux/ama_with_perplexitys_aravind_srinivas_denis/).
   - The AMA covered user reactions to the product, core use-cases, and upcoming features.
- **Yarats Jumps Ship to Perplexity**: **Denis Yarats (co-founder & CTO)** joined the **Perplexity AI** team, according to [this announcement](https://www.perplexity.ai/); however, members wondered *where is Deep Research High*?
   - A member expressed frustration with the delays to **Deep Research High**, posting a [confused GIF](https://tenor.com/view/confused-huh-what-gif-15066348) in response.
- **GPTs Agents Suffer Amnesia**: Members discussed that **GPTs agents** are unable to learn from additional information after initial training, emphasizing that [uploaded files are saved as knowledge](https://link.to/openai-docs), but *do not continually modify the agent's base knowledge*.
   - The conversation highlighted the limitations of **GPTs agents** in retaining information and adapting to new data.
- **Perplexity Pro Users Get Short Shrift**: Members critiqued the context limitations (5-10 sources) for the **Perplexity Pro** plan, citing small context size and its inability to remember previous messages as a key issue.
   - One member noted, *Yes, you constantly have to remind it what you're asking about*, indicating frustration with the tool's memory.
- **Qwen Reigns Supreme over Deepseek**: Members stated that the [**Qwen**](https://chat.qwen.ai/) model surpasses **Deepseek R1** in reasoning capabilities, boasting a 1M context window, and indicating that **Perplexity** will leverage it for deep research.
   - Further discussion highlighted **Qwen's** accessibility as a free model, contrasting with the often-busy **Deepseek** server.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Goldmane Heralds Gemini 2.5 Pro GA!**: The release of **Gemini 2.5 Pro** is imminent, with the **Goldmane** version scoring **86%** on the Aider webdev benchmark, as [seen here](https://aider.chat).
   - The **diff-fenced** edit formatting is primarily used with **Gemini models**, according to [Aider Docs](https://aider.chat/docs/more/edit-formats.html#diff-fenced).
- **Kingfall: Google's Accidental DeepThink Model Release Creates Buzz**: A model called **Kingfall**, believed to be an internal **Gemini model**, briefly appeared on AI Studio, leading to speculation about its capabilities and whether it's **DeepThink**.
   - Members noted it has a **65k** context window, but the "confidential" name hinted that someone was going to get fired.
- **OpenAI's o3 Pro Still MIA?**: The release of **OpenAI's o3 Pro** is highly anticipated, but the release date remains uncertain, and early impressions have been lukewarm, with one member stating, *"i have it alrdy, its ass"*.
   - Concerns arose around **o3 Pro's** limitations in generating code, maxing out at **500 LOC**, whereas its predecessor could generate **2000 LOC** without omissions.
- **Model Showdown: Spatial Reasoning Skills on Display**: Comparisons are being made between various models, including **Gemini 2.5 Pro**, **Claude Opus**, **Grok**, and **OpenAI's o3**, focusing on coding proficiency, reasoning, and overall performance.
   - One user tested Kingfall's **spatial reasoning** by giving it a [geoguessr task](https://www.geoguessr.com) with stunning results.
- **Free API Use Ends, Google Closes Wallet**: The removal of free API access for **Gemini 2.5 Pro** has sparked disappointment, especially for long-form content generation use-cases.
   - A user joked how Gemini requires credit card details and sign up with valid payment details offering *$300 free credit*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Authorization Hiccups Plague Cursor Pro GPT-4.1 Access**: After upgrading to **Cursor Pro**, some users are encountering *'User is unauthorized'* errors when trying to access **GPT-4.1**, requiring intervention from the **Cursor team**.
   - Affected users are sharing request IDs and account emails to resolve the issue.
- **Claude 4 Sonnet's Context Crisis Spurs Prompt Engineering**: Users report that **Claude 4 Sonnet's** context window is limited, interrupting conversations, but suggest using the *'continue where you left off'* prompt trick.
   - One user speculates that **Claude 4** has a *'rolling context'* taking key considerations into account throughout the chat.
- **RIP your Workflow with CursorRIPER Framework**: The **CursorRIPER framework** helps guide agent behavior using rules and memory to maintain context and focus on projects, which is supported by a **tech context file**.
   - The framework aims to prevent the use of outdated modules and ensures the agent remains aware of the project's current state after major edits.
- **Claude Code Emerges as Refactoring Rock Star**: Some members are declaring **Claude Code** superior to **Cursor** for specific tasks and praising its *'incredibly smart'* coding capabilities based on recent experiences.
   - One user claimed successful one-shot refactoring of a large, complex codebase with **Claude Code**, passing thousands of tests without errors.
- **Cursor 1.0 Arrives with Code Smarts, Background Chores**: The latest **Cursor 1.0** release includes enhanced **code review capabilities** that remembers its mistakes, improved **error tracking**, and the ability to handle multiple **background tasks**.
   - Users can check the [official changelog](https://www.cursor.com/changelog) for a detailed overview of all updates.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O3 Pro Arrival Still in Question**: Members speculated about the release of **o3 Pro**, while others remained skeptical due to previous delays and unfulfilled announcements by **Sam Altman**.
   - One member quipped, *“There will be no o3 pro. They will release chatgpt5.”*
- **OpenAI employees tease New Features**: OpenAI employees teased major updates for **Teams** and **Enterprise** plans, with the new **Connectors** feature allowing users to perform searches over internal sources using reasoning models.
   - According to one member, *“they just launched an update, todays annoucement is very beneficial for teams user reason is, we can use any reasoning model to search over internal sources.*
- **TTS Pricing Discrepancy Debated**: A member questioned why **gpt-4o-mini-tts** charges about 4 times more than **tts-1**, even though pricing is listed at **$12** vs **$15** per 1M characters, respectively.
   - Another member suggested checking the [OpenAI community forum](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616) for insights into the potential gotchas.
- **Agent Flow Aims to Query Elasticsearch**: A member is building an agent using **open ai gpt-41-mini** to create **Elasticsearch DSL queries** based on human queries for charting, starting with a single agent and breaking it down into multiple agents to identify index names, get mappings, generate queries, and extract data, as illustrated in [this attached image](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png?ex=684204b7&is=6840b337&hm=eadc26c0018f544fedd3a9e8e6407dcf5f02301750f66ef5732067f94b94beff&).
   - Another member identified at least *seven* issues with the current setup, with the biggest one being sorting everything in **Elasticsearch**, even the indexes.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek Runs into Speed Bump**: A user reported that **DeepSeek R1 0528** runs slower (**12.8 t/s**) than **R1** (**18.7-19 t/s**) on a Mac Studio, but it was suggested that different quantization formats may be the cause.
   - It was proposed that dynamic quantization might behave differently, impacting the model's speed.
- **Qwen Questioned on Generalization**: A user suggested that **Qwen 4B** doesn't generalize as well as **Gemma 4B**, highlighting potential differences in generalization capabilities.
   - The user did not provide any additional details.
- **Llama.cpp Saves the Vision**: Users seeking vision features for **unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF** were directed to use *llama.cpp*, and [provided with instructions](https://github.com/ggerganov/llama.cpp).
   - Steps included cloning the repo, creating the build, enabling CUDA, and building *llama-cli*.
- **Multi-GPU Support Coming Very Soon™️**: **Multi-GPU support** *already works with accelerate*, with *an even better version* expected in early July.
   - Due to the current support's *unofficial* nature, no official examples were provided, but users familiar with accelerate can utilize it.
- **Fastest Library Face-Off**: For single-user CPU inference, a library based on [llama.cpp](https://github.com/ggerganov/llama.cpp) might be best, while [vLLM](https://github.com/vllm-project/vllm) or [ktransformers](https://github.com/ktransformers/ktransformers) are better for CPU deployments.
   - There's been work on the **v0 engine** that handles this, but it doesn't exist in **v1**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter adds GIF Support Across Models**: OpenRouter now accepts `image/gif` for image prompts on **OpenAI, Gemini, Anthropic, and Llama routes**, streamlining animation use.
   - This eliminates the need for users to pre-convert animations into other formats.
- **iOS App Integrates OpenRouter**: An iOS app is set to launch via **TestFlight**, utilizing **OpenRouter** as its **LLM backend** and employing character cards.
   - The developer is still working on message formatting due to its complexity, but aims to add more clients later.
- **Anthropic Model Rate Limits Lifted!**: OpenRouter now offers higher rate limits for **Opus**, especially when routing traffic to **Anthropic** models, leading to discussions about the economics of **Chutes**.
   - Speculation arose about the sustainability of **Chutes**' business model, considering the necessary GPU resources.
- **Nous Struggles with Distributed Training**: **Nous** is attempting distributed training of a SOTA model using **416 H100s**, but progress is slow.
   - Projected training time extends into next year, prompting skepticism despite claims of breakthroughs reducing inter-GPU bandwidth needs to ~300mbps.
- **OpenRouter API Maximization Techniques**: Members discussed strategies for sending **100K calls** to an **LLM** via OpenRouter, focusing on throughput and provider discounts.
   - Resources like Modal's **LLM Almanac Advisor** were shared to optimize **API** usage and reduce costs.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU Mode Mascot Conceptualized**: Members proposed creating a **GPU Mode** mascot and merchandise, with an initial suggestion for a *"supersayan GPU"* design.
   - After using **ChatGPT** to generate a potential design, members found the generated image wasn't simple enough to function effectively as a logo or mascot, due to [copyright concerns](https://en.wikipedia.org/wiki/Copyright_law).
- **CUDA Barrier State Revealed**: `__syncthreads()` is basically `bar.sync`/`barrier.sync.aligned`, while `sync(cooperative_groups::this_thread_block())` gives `barrier.sync` for syncing threads in different branches ([Volta](https://developer.nvidia.com/blog/volta-new-programmability-features/) and newer only).
   - For producer/consumer scheme, using the `cuda::pipeline` from libcu++ is the right thing to do for [CUDA](https://developer.nvidia.com/cuda-zone).
- **CUPTI Command Buffers Overflow**: High overhead in **CUPTI** profiling may be due to a bottleneck from the GPU's command buffer being full, referencing the [CUpti_ActivityOverheadCommandBufferFullData documentation](https://docs.nvidia.com/cupti/api/structCUpti__ActivityOverheadCommandBufferFullData.html#structcupti__activityoverheadcommandbufferfulldata).
   - A member noted that using Python constants directly in Torch Dynamo can trigger recompiles, as shown in the log `___as_tensor(alpha).item() == 0.5`.
- **vLLM Gets VL Model Fix**: A fix for **vLLM** and **VL Models** was released via [this GitHub pull request](https://github.com/vllm-project/vllm/pull/19147).
   - Before the fix, loading serialized ao models in **vLLM** worked with language models where all the layers are quantized, but broke with VL models when the vision model is not quantized.
- **MI300X Profiling Puzzles Persist**: A member reported issues using `rocprof` to read **L2CacheHit** for a kernel with **MI300X**, noting that while the metric is listed as available in the [ROCm documentation](https://github.com/ROCm/ROCm/blob/develop/docs/conceptual/gpu-arch/mi300-mi200-performance-counters.rst), `rocprof` returns an error indicating it's not supported on **gfx942**.
   - Members profiling **FetchSize**, **WriteSize**, **MfmaUtil**, and **SQ_LDS_BANK_CONFLICT** and found that a low L2 cache hit rate correlates with a low **MfmaUtil** score.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **IBM Prompts Responsibly with New API**: An IBM intern introduced the **Responsible Prompting API**, an [open-source project](https://github.com/IBM/responsible-prompting-api) that gives pre-inference prompt recommendations to make LLM outputs more responsible, accurate, and productive, detailed in [this paper](https://arxiv.org/abs/2504.08757).
   - The system, demonstrated on [HF Spaces](https://huggingface.co/spaces/santanavagner/responsible-prompting-demo), assists domain experts lacking prompting skills, potentially cutting harmful outputs and inference costs.
- **Blockchain Tech Boosts AI Output?**: A member shared a [concept paper](https://medium.com/@info_65774/consensus-validation-for-llm-outputs-applying-blockchain-inspired-models-to-ai-reliability-f642d7f96f8e) on using **blockchain-style consensus mechanisms** to improve reliability and trustworthiness in LLM outputs.
   - The paper focuses on AI agents, legal/medical tools, and AI alignment applications.
- **Whisper Transcribes Audio on the Cheap**: Users leverage **OpenAI's Whisper model** for audio transcription affordably, with [volodymyr kublytskyi's repo](https://huggingface.co/spaces/vkublytskyi/Final_Assignment_Agent/blob/main/tools/youtube_video_tool.py) aiding agent video interaction.
   - Members are using **Gemini-2.0-flash** with **SmolAgents**, noting that it performs *quite well* on the OpenAI server.
- **Market Research Basics**: A member shared that they are learning **market research basics** and the **ACP Funnel**.
   - They also noted that *long-form posts with images get the most interaction on X*.
- **Prisma Wins Big, Gets HF Ready**: The **Prisma** toolkit, designed for mechanistic interpretability in vision and video, received an Oral presentation at the CVPR 2025 workshop, and adapted [Hugging Face models](https://huggingface.co/).
   - The release, as mentioned on [Twitter](https://x.com/soniajoseph_/status/1930286144471646252), includes circuit-style code for **100+ models**, including CLIP, DINO & video transformers, and interactive notebooks.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Tasks Hit Context Limit, Restart From Scratch**: A user found that **Manus** hit the context limit after **1 hour 55 minutes**, requiring a new task that restarted from scratch.
   - The user expressed disappointment due to the loss of progress after reaching the context limit.
- **H Runner Competes for AI Attention**: A member shared a link to **H Runner** by *H Company* ([https://www.hcompany.ai/](https://www.hcompany.ai/)), pitching it as a competitor to **Manus AI Agent**.
   - Members shared that it is currently free but not as advanced, and is limited to **10 runs daily**.
- **Manus Credit Consumption Sparks Debate**: A user reported spending **$50** on a 30-slide PowerPoint presentation, blaming **Manus** for building outside the slide borders.
   - Another user found a **30-second** video cost **208 credits**, while others shared referral links to gain more credits.
- **Interactive Experiences: Web vs App?**: Members debated the best deployment for interactive experiences: a website or hosted as an app like JS on GitHub.
   - One member suggested it depends on the product, citing examples like interactive movies needing a big screen and language learning apps benefiting from memorization and speaking practice.
- **Cursor, Devin, and Replit: IDE Impressions**: One member stated they create websites and web apps and need to rework **Manus's** output in **Cursor** or another IDE to make it functional.
   - Another member has been playing with **Cursor**, **Devin 2.0**, and **Replit**, the latter of which they found nifty for making an app a day.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes 24B Plummets into API Abyss**: The **DeepHermes 24B** model encountered an **API outage**, impacting both its **API** and **Chat Product** functionalities.
   - Users were notified and asked to bear with the team as they addressed the interruption.
- **Nous Research Eyes Server Tag**: A member requested a **server tag** for **Nous Research** to enhance visibility, referencing [Discord's documentation on server tags](https://support.discord.com/hc/en-us/articles/31444248479639-Server-Tags).
   - The request received positive feedback, with assurances of implementation within **24 hours**.
- **Shisa-v2 405B Debuts From Japan**: The **Shisa-v2 405B model**, the most powerful model trained in **Japan**, was released, specializing in both **Japanese** and **English** with performance comparable to **GPT-4/Deepseek**.
   - Users were invited to test the model via an endpoint to their **H200 node** at [chat.shisa.ai](https://chat.shisa.ai/), and a detailed tech report is promised on **Arxiv**.
- **LLM Self-Play Paper Seeks Feedback**: A member announced the publication of their paper, *Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance*, available at [ai.vixra.org](https://ai.vixra.org/abs/2506.0018).
   - The author is seeking feedback and insights from the community on their research and observed emergent performance.
- **Merlin App Now Listens**: A member highlighted the [Merlin bird identification app](https://merlin.allaboutbirds.org/), noting its ability to identify bird species using both **photos and sounds**.
   - The app's **sound analysis** feature provides a comprehensive approach to bird identification, complementing its existing photo analysis capabilities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Modal Labs Serves LLM Engineer's Almanac**: Modal Labs launched the [LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213) with thousands of **LLM inference benchmarks** for open-weight models across **vLLM**, **SGLang**, and **TensorRT-LLM** frameworks.
   - The release includes results, code for replication, and an executive summary addressing build vs. buy, cost estimation, and framework choice, and the **'stopwatch' benchmarking framework** to understand performance metrics.
- **AWS Textract Accuracy Troubles Reported**: A homegrown **PDF ingestion pipeline** in AWS uses Lambda to split PDFs and Textract for parsing, with a queue to manage Textract request limits.
   - A user cautioned that **Textract accuracy** can be as low as *3% on legal and regulatory documents*, linking to a [LinkedIn post](https://www.linkedin.com/posts/robertreich_when-word-for-word-accuracy-is-key-in-etl-activity-7265008546793086978-hfaj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAABOb18Bac53omUsFRAIBEVDUe013Eez5zoTry).
- **Anthropic's Capacity Cut Causes Chaos**: Anthropic unexpectedly cut off nearly all **Claude 3.x model capacity** with less than five days' notice, affecting services like Windsurf, according to [this post](https://x.com/_mohansolo/status/1930034960385356174).
   - Users expressed disappointment, with some considering migration, while ai.engineer is offering **BYOK options** and improved their agentic harness for Gemini 2.5 Pro and GPT-4.1, according to [this post](https://x.com/kevinhou22/status/1930401320210706802).
- **Altman Activates Internet Access for Codex**: Sam Altman announced that **Codex**, an AI coding tool, now has optional internet access for **ChatGPT Plus** users, disabled by default due to complex tradeoffs as described in [this tweet](https://x.com/sama/status/1930006856019390521).
   - The community discussed implications and potential security concerns, with Grok providing a detailed explanation of the announcement.
- **OpenAI Aims For Agent Reliability**: OpenAI announced four updates for building agents: Agents SDK in TypeScript, a RealtimeAgent feature, Traces support for Realtime API sessions, and speech-to-speech model improvements.
   - These enhancements aim to improve reliability, consistency, and user control, demonstrated by early testers like **Perplexity**, **Intercom**, and **VolleyGames** as shown in [this tweet](https://x.com/OpenAIDevs/status/1929950012160790876).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Google Workspace Copies NotebookLM Features**: A user shared a [Chrome Unboxed article](https://chromeunboxed.com/google-workspace-feature-drop-for-may-2025-is-loaded-with-new-features/) indicating that features of **NotebookLM** are being integrated into **Google Workspace**, initially for individual documents.
   - Users are speculating about when **NotebookLM** will upgrade to more advanced models like **Gemini 2.5 Pro** or **Flash** to enhance its performance.
- **Flash and Pro Faceoff**: Members debated the merits of **Gemini 2.5 Flash** versus **2.5 Pro**, noting **Pro's** thoroughness as preferable for handling larger file uploads where nuanced details are important.
   - One user suggested implementing a beta branch to allow users to switch to **2.5 Pro** for potentially higher quality output at the cost of longer processing times.
- **NotebookLM Audio Overview Length Hack Discovered**: Users discovered that the audio overview length in **NotebookLM** can be adjusted by selecting *'Customize'* instead of *'Generate'* in the studio, which enables options for shorter, default, or longer lengths.
   - This customization feature is available on the web and mobile web versions, but might be absent from the official mobile app.
- **Google Docs Sync Requires Manual Resync**: Users have confirmed that after adding a **Google Doc** as a source in **NotebookLM**, any subsequent changes to the **Google Doc** do not automatically sync; a manual re-sync is required from the preview.
   - It was also clarified that the new public share option in **NLM** does not depend on the **Gdoc's** own share settings since **NLM** shares its own copy, and the share links remain constant through updates.
- **NotebookLM Mobile App Missing**: The **NotebookLM** mobile app is considered a *'minimal value product'* due to its lack of feature parity with the web version.
   - Users are encouraged to submit their feature requests in the *'Mobile App'* thread within the Feature Request channel to advocate for improvements.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Parameter-Efficient Finetuning Jumps Ahead**: A new method for **parameter-efficient finetuning** reportedly achieves **4x more knowledge uptake** compared to full finetuning and **LoRA**, alongside a **30% reduction in catastrophic forgetting**.
   - This approach is particularly beneficial for adapting models to new domains and incorporating specific knowledge in local setups without eroding existing knowledge.
- **Knowledge Extension Eyes RAG's Crown**: A member intends to extend an **LLM's knowledge** using a collection of books and documents, comparing it to **RAG-like approaches** for assistance. They shared [an x link](https://x.com/unusual_whales/status/1929998955703931375) discussing AI rights, and a [markdown document](https://cdn.discordapp.com/attachments/986699377257119794/1379670337008046080/UDAIR.md?ex=68426721&is=684115a1&hm=4e73690d912c8e0286f50b7a456f683012b700561418b45222466ae5230e3a9f&).
   - The member mentioned the discussion can lead to *wild convos*.
- **Muon Optimizer Deconstructed**: A member explored the **Muon optimizer**, which uses **AdamW** for parameters unsuitable for **Muon**, linking to [experimental results](https://github.com/KellerJordan/Muon/issues/25) for multitask learning.
   - It was explained that the **Muon optimizer** adjusts the gradient for a weight-*matrix* so that its eigenvalues are approximately equal to 1, a stark contrast to **SGD** and **Adam**.
- **Mistral Code Aims for Developer Delight**: **Mistral AI** launched [Mistral Code](https://mistral.ai/news/mistral-code), an **AI-powered coding assistant** which integrates powerful models, an in-IDE assistant, local deployment, and enterprise tooling.
   - Built on the open-source project **Continue**, it supports JetBrains IDEs and VSCode, furthering Mistral's ambition to empower developers through AI.
- **ChatGPT Logs Under Scrutiny?**: Members discussed [an ArsTechnica article](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/) noting **OpenAI** is compelled to save all **ChatGPT logs**, including deleted chats and sensitive data from its API business.
   - A member questioned the rationale behind this decision.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Efficient Finetuning Takes the Stage**: A new parameter-efficient finetuning method claims **~4x more knowledge uptake** and **30% less catastrophic forgetting** than **LoRA** while using fewer parameters.
   - The method is suited for continued pretraining and efficiently teaches models new information without overwriting existing knowledge.
- **API-less Twitter Scraper Saves the Day**: A member shared a [Twitter scraper](https://gist.github.com/mookiezi/9ea0f0f5aad76a51e5b35a084d82a9df) that **doesn't use the API**, logs to **Postgres**, and skips retweets.
   - The scraper doesn't collect reply metadata, making it better suited for profiles and efficient data collection.
- **World Models Infected by Semantic Virus**: A [paper](https://arxiv.org/pdf/2506.01622) suggests general agents require **world models**, and that a **Semantic Virus** exploits this by *infecting* reasoning paths if the **LLM's world model** has *holes* or *disconnected areas*.
   - The **Semantic Virus** doesn't rewrite the base **World Model** but hijacks its current activation within the context window.
- **ROI Doubts Burst AI Startup Bubble?**: A member expresses concern about graduating into a job market where the **ROI** of **AI** is questioned, leading to a potential bubble burst for **AI startups**.
   - They claim that many **AI startup CEOs** lack **ML** expertise and are backed by investors who cannot properly evaluate **ML** skills, potentially leading to instability.
- **Universal Algorithm Makes Appearance**: A member shared demos of their [research](https://github.com/qLeviathan/g-qfnn), a **universal algorithm** with basic POCs for **NLP**, **options trading**, and **electrochemical reactions**.
   - This research introduces a novel approach, sparking interest in its potential applications across diverse domains.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 4** Image Support Still a Mystery**: Users are questioning whether **Llama 4** supports images on LM Studio after an Unsloth version indicated otherwise, leaving the community in suspense.
   - As of now, no definitive confirmation or denial has surfaced in the discussions.
- **agenticSeek** Sheds Old Skin, Rebrands from **OpenManus**: [agenticSeek](https://github.com/Fosowl/agenticSeek) has rebranded from **OpenManus**, prompting inquiries about the reasons behind the name change, drawing parallels to OpenDevin's transformation into OpenHands.
   - Speculation suggests that copyright issues may be at play, similar to other high profile name changes in the open source AI space.
- **Gemma** Glitters as Embedding Model**: A user testing various embedding models (**Gemma 3 4b**, **12b**, **Deep Seek 8b**, **Microsoft phi 4 small**) found that **Gemma** gave more accurate answers than Deep Seek or Microsoft Phi, particularly for mixed text and PDF data.
   - The user's data, consisting of files ranging from 0.5-30 MB, is used with Supabase and n8n.
- **ROCm** Vision Module Plagued with Performance Problems**: Users have reported a significant slowdown in the vision module with the new **ROCm llama.cpp v1.34.1** runtime on a **7900XT 20GB**, with response times jumping from ~1 second to 10+ seconds, according to [screenshot of their results](https://cdn.discordapp.com/attachments/1110598183144399058/1379953808532049981/image.png?ex=68421da2&is=6840cc22&hm=37d660db87619d86ca215fc8862f4762688295f6516dcb95ee68d5e84a525bc2&).
   - The findings led to requests to share detailed results in the appropriate Discord channel, indicating a potential area for optimization or debugging.
- **SSD Secrets**: Data corruption and refresh cycles unveiled**: Discussion around data corruption in **SSDs** revealed that if not powered on for extended periods, data may degrade, contrasting with HDDs where data is physically written and less prone to degradation over time.
   - It was mentioned that the cells in **NAND** memory used in SSDs slowly leak charge over time, and that hardware needs to perform *read refresh*.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP API Key Monetization Sparks Sass Debate**: Members discussed implementing **API keys** for MCP monetization, suggesting it mirrors standard SaaS models with API keys and billing dashboards.
   - The discussion emphasized that MCP clients would handle **auth to the server**, potentially simplifying monetization strategies and questioning the necessity of a dedicated MonetizedMCP solution.
- **A2A Framework Battles MCP for Agent Supremacy**: The discussion revolved around **A2A** ([https://github.com/google/A2A/](https://github.com/google/A2A/)) as a framework alternative to MCP for agent interactions, with some noting limited adoption.
   - While some speculate that A2A is gaining traction behind closed doors with significant deals, others expressed a preference for the **A2A spec** over MCP for multi-agent systems.
- **Pydantic-AI Streamlines Agent Dev**: Members advocated for starting agent framework development with **pydantic-ai-slim** ([https://ai.pydantic.dev/install/]), highlighting its convenient `.to_a2a()` method.
   - They mentioned an optional a2a group (`uv add 'pydantic-ai-slim[a2a]'`) for enhancing existing agents, potentially easing integration with A2A protocols.
- **Cloudflare Hosting for MCP Causes Headache**: A member sought guidance on hosting an MCP server on **Cloudflare** for a user lacking technical expertise.
   - Clarification indicated that **HTTP transport** MCP servers shouldn't need local software if the MCP client offers native support; otherwise, a translator might be needed.
- **MCP Context Management Tackles Agent Crisis**: A member questioned how MCP handles context across multiple agents and the engineering mechanisms required to maintain this context.
   - It was clarified that **MCP isn't agent-first**, with guidance available at [https://fast-agent.ai/mcp/state_transfer/], offering insights into state transfer mechanisms.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Engineers Invade AI Event**: LlamaIndex is at the [@aidotengineer](https://twitter.com/aidotengineer) in San Francisco, showcasing the latest in **Agentic AI** at Booth G11 with CEO @jerryjliu0 and the AI engineering team.
   - Meanwhile, @seldo from LlamaIndex broke down **Effective Agent Design Patterns** in Production at [@aiDotEngineer](https://twitter.com/aiDotEngineer).
- **LlamaIndex Builds Financial Report Chatbots**: LlamaIndex presents a [hands-on Colab](https://twitter.com/llama_index/status/1930051898247393729) to build a **multi-agent financial report** generating chatbot from scratch, parsing & indexing 10-K filings from Adobe, using agentic RAG.
   - This originated from @jerryjliu0's workshop and LlamaIndex also demonstrates how to automate SEC Form 4 extractions using [LlamaExtract](https://twitter.com/llama_index/status/1930414284670152875) and agent workflows.
- **Hackathon Participants Seek LlamaIndex Wisdom**: Office hours for the [@Gradio](https://twitter.com/Gradio) [@huggingface](https://twitter.com/huggingface) MCP hackathon started soon after this message, with a [$1000 prize](https://twitter.com/llama_index/status/1930286458340028484) for the best LlamaIndex submission and 10k LlamaCloud credits up for grabs.
   - Members @tuanacelik and @LoganMarkewich answered LlamaIndex questions; HuggingFace also hosted office hours for **Gradio MCP Hackathon** participants on their Discord server, [linked here](https://discord.com/events/879548962464493619/1379561017536938095).
- **Graph Index Gets Put Under the Microscope**: A member is exploring **Property Graph Index**, and would like to know about the **token-usage for indexing & retrieval**, and the **performance for retrieval & end to end**.
   - They are comparing to **GraphRAG**, **HippoRAG2**, and **LightRAG**.
- **Qwen3 Powers Code Interpreter Agent**: One of the member wants to build **code interpreter agent** like the one in [this medium article](https://medium.com/@venugopal.adep/building-an-ai-data-analysis-assistant-with-llamaindex-and-openai-c0e371a432d6) but using **qwen3** instead of **OpenAI**.
   - Another member suggested using **Ollama** to serve **qwen3**, [linked here](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NumPy-ectomy Surgery Shifts to GPU**: A member is attempting to remove **NumPy** from `random_crop/cutmix` for the `hlb_cifar10` bounty, only to find that the **NumPy** operations are being offloaded to the GPU.
   - The user faces challenges building intuition about **tinygrad performance**, struggling to identify performance bottlenecks.
- **Windows Users Wrestle Tinygrad**: A member reported multiple issues running **tinygrad** on Windows, including CPU backend crashes with JIT and hangs with BEAMS=1.
   - They had to apply a hack to autogen files to enable CUDA, suspecting their Windows environment to be the root cause of performance problems.
- **LSTM Lags Badly in Tinygrad**: While porting a **VAD model** from PyTorch to **tinygrad**, a member discovered that the LSTM layer was significantly slower than the other layers.
   - The LSTM's sluggishness persisted regardless of the chosen backend.
- **DEBUG=2 Decoding Demands Diligence**: A member expressed feeling overwhelmed by **tinygrad**'s `DEBUG=2` output, struggling to interpret the columns and the abundance of kernels.
   - They specifically questioned the high number of `randperm` kernels and the cryptic naming conventions, such as `r_512_32_8_4_8_3_16_3_4_4`.
- **CUDA Customization Conundrums**: A member is looking for examples of using **CUDA kernels** with **tinygrad**'s CUSTOM ops to port a project using 5-10 kernels.
   - Although the member acknowledges that custom kernels might conflict with the "Zen of TinyGrad", they feel it is necessary due to their limited understanding of expressing the needed kernels in Python.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Bids Farewell to Python 3.9**: The impending end-of-life for **Python 3.9** is causing CI failures due to new linting rules, requiring temporary workarounds to maintain compatibility and adoption of new linting rules.
   - One member quipped, *"sorry Joe this is the reason of failed CI :/"* regarding the need for `Union` and `Optional` from the `typing` module.
- **Asynchronous Reward Functions Get a Batch Boost**: Reward functions are looped through with a batch for potential concurrent computation, but the calls aren’t natively asynchronous and are limited by the **Reference model worker's resources**.
   - One member shared, *"Reward functions are just looped through and a batch is passed in that you could try and compute concurrently, but the calls aren’t async and you only have access to the resource of the Reference model worker.*"
- **Iterable Dataset Refactoring RFC Breaks the Mold**: An RFC ([Iterable dataset refactoring](https://github.com/pytorch/torchtune/pull/2785)) proposes a major overhaul in how datasets are handled in TorchTune, inviting community feedback on its design and potential breaking changes.
   - A member emphasized the importance of input: *"Its a big change. I would greatly appreciate any input / vibes. Does it feel like the right way to work with datasets in torchtune? Would you change anything drastically since we are breaking things anyway?"*
- **DTensor DeviceMesh Errors Plague Optimizer Trials**: Testing TorchTune with optimizers beyond **AdamW** in full distributed SFT, such as **SGD**, **Adafactor**, and **Adagrad**, resulted in an `AssertionError` related to `DeviceMesh` from dtensor args for aten._foreach_lerp_.ScalarList!.
   - Others have tested **Muon** and **AdamW** with different precisions from torchao.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC assignment deadlines are firm**: After inquiries about extending the **May 31st** deadline, staff confirmed that the forms had already been kept open for an additional two days to accommodate technical issues and *they won't be able to open the assignments any further unfortunately*.
   - The community consensus seems to be that no further extensions can be expected.
- **Detailed feedback on MOOC assignments unlikely**: A member requested detailed feedback on all submissions, including the **AgentX project** and **lab assignments**.
   - Staff responded that *they don't have bandwidth as a staff to do that*, but promised to pass the suggestion along.
- **Future of the MOOC is uncertain**: Inquiries were made about plans for a next step, edition, or progression after the conclusion of the **Spring 2025 MOOC**.
   - Staff stated that *nothing has been confirmed yet*, but *chances are likely (but not guaranteed currently)* indicating a possible continuation but without firm commitment.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Anthropic's Dev Cycle Exposed in System Prompts**: A blog post compared [system prompts](https://www.dbreunig.com/2025/06/03/comparing-system-prompts-across-claude-versions.html) across **Claude 3.7** and **4.0**, revealing details about **Anthropic's** development cycle and priorities.
   - The author noted *a few changes in the system prompt between Claude 3.7 vs 4.0*.
- **Oneformer's Game-Theoretic Gambit**: A member is developing a **Oneformer** game theorist but is hesitant to reveal it.
   - The member is also debating its potential success when stacked up against **Agenspy** and other frameworks.
- **Angel Azul Cracks Claude SDK**: A member shared their work on the [claude_sdk execution engine](https://github.com/darinkishore/claude_sdk/tree/t1-execution-engine), clarifying that it's a work in progress and may contain bugs, with architecture patterns detailed in [ai_docs](https://github.com/darinkishore/claude_sdk/blob/t1-execution-engine/ai_docs/ARCHITECTURE_PATTERNS.md).
   - The SDK offers improvements over the existing Claude SDK.
- **HTNs Hack LLM Agents**: A member suggested that **LLM agents** might benefit from fine-tuning specifically in **ReACT format**, instead of adopting a general chat model approach, while playing with **HTNs**.
   - Further investigation into the roadmap is necessary in order to adapt to new capabilities like **SO/schemas** with retries for errors.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Sponsors Hackathons?**: Members are requesting contact information for **Cohere** to explore sponsorship possibilities for post-secondary hackathons.
   - The users are specifically looking for the right person to contact regarding sponsorships.
- **Cohere's Crew Greet New Members**: New members are actively introducing themselves in **Cohere**'s Discord channel 🤝-introductions, providing insights into their professional experiences, ongoing projects, and preferred technologies.
   - These introductions highlight the community's broad range of skills and interests within the **AI and GenAI** landscape, according to the channel's guidelines.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All's LlamaCPP Library Lagging Behind**: Users are saying the **LlamaCPP library** that powers **GPT4All** is several months out of date, and the automatic updating to the newest release isn't functioning.
   - It seems that updating the library requires more than simply copying and pasting the new version.
- **MOE Models Get Slimmer**: It's now possible to run larger **MOE models** with a more reasonable amount of **VRAM**.
   - This is achieved through offloading certain experts and tensors, requiring some coding wizardry to manage memory constraints effectively.
- **Mac M3 Max flexes on VRAM**: The **Mac 512 GB** configuration boasts a significant **448 GB** of "VRAM" at a similar price point to four newer **AMD AI MAX 395+ 128 GB** mini PCs or laptops.
   - The Mac also uses less power.
- **vLLM Engine Could Power Up GPT4All**: There is research on adding the **vLLM engine** to the **GPT4All** project, potentially making it a leading open-source project.
   - The project would then feature two underlying engines written in different programming languages, significantly upgrading its capabilities.
- **Tesla's Lightbulb Moment**: A user shared a [link](https://buck.lighting/blog/nikola-tesla-and-light/) discussing Nikola Tesla's contributions to energy and light.
   - The user speculated that *"his inventions were stolen from him somehow"*.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Guo Guides Good AI**: Industry expert **Liang Guo** is holding a webinar on **AI programming for data analysis**, with RSVP [details here](https://forms.gle/e71FSdpwBtDBccgKA).
   - The webinar is geared toward practical AI programming techniques.
- **SVCAI Summer Competition Enrolling**: The **Silicon Valley Chinese Association (SVCA)** is holding an **AI4Legislation** summer competition and [details are available on the project's GitHub repository](https://github.com/svcaf/2025-AI4Legislation-Public).
   - The repository provides resources and guidelines for participants.



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





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1379863824454258748)** (1 messages): 

> `Reddit AMA, Labs, Aravind, Denis, Tyler Tate` 


- **Perplexity Leadership Hosts Reddit AMA**: **Aravind (CEO), Denis (CTO), Tony (VP Engineering), Weihua (Member of Technical Staff), and Tyler Tate (Product)** are hosting a live Reddit AMA to discuss Perplexity Labs at 10am PT ([link to Reddit AMA](https://www.reddit.com/r/perplexity_ai/comments/1l39wux/ama_with_perplexitys_aravind_srinivas_denis/)).
- **Ask Perplexity Labs Anything on Reddit**: Perplexity is hosting an AMA on Reddit to answer user questions about their reactions to the product, core use-cases, what's coming next, and more!


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1379535149196251136)** (1289 messages🔥🔥🔥): 

> `Deep Research High, O3-pro, GPT-5 Release` 


- **Denis Yarats joins Perplexity Team**: Denis Yarats (co-founder & CTO)  joins Perplexity AI team with [this announcement](https://www.perplexity.ai/).
   - Members on Discord joke about Yarats' arrival, asking where is Deep Research High.
- **Deep Research High: still delayed**: The release of Deep Research High is still delayed according to some members.
   - A member expresses frustration with the delays, posting a [confused GIF](https://tenor.com/view/confused-huh-what-gif-15066348) in response.
- **GPTs Agents not learning**: Members discuss the inability of **GPTs agents** to learn from additional information after initial training and [uploaded files are saved as knowledge](https://link.to/openai-docs).
   - One states that *they do not continually modify the agent's base knowledge*.
- **Perplexity Pro limitations are annoying**: Members complain about the context limitations (5-10 sources) for the **Perplexity Pro** plan, with small context size and its implication with its inability to remember previous messages.
   - Member quotes *Yes, you constantly have to remind it what you're asking about*.
- **Members are excited about a new model, Qwen**: Members state that [**Qwen**](https://chat.qwen.ai/) model is better than **Deepseek R1** in terms of reasoning, has a 1M context window, and will be used by Perplexity for deep research.
   - The members add that **Qwen** is also free, whereas the deepseek server is often busy.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1379536608696602775)** (2 messages): 

> `working app, smuggled north korean smartphone` 


- **Create a working app**: There is a [Perplexity search result](https://www.perplexity.ai/search/create-a-working-app-using-the-9B6cBgPATvmgfo6mwd07sg?0=c) related to creating a working app.
- **Smuggled North Korean Smartphone**: There is a [Perplexity page](https://www.perplexity.ai/page/smuggled-north-korean-smartpho-NgjIJo_RTW6Dx8TYfGWpZg) about smuggled north korean smartphones.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1379572349330002011)** (19 messages🔥): 

> `Academic Filter Feedback, Sonar Reasoning Pro API with PMC, NCBI Rate Limiting, Firecrawl proxies` 


- **Academic Filter Gets Strong Feedback**: A member gave [feedback](https://discord.com/channels/974519860452964443/1161802929053909012/1379816311635685406) on the new **Academic Filter** mode, noting *strong synthesis*, *high-quality sources*, and a *good scientific tone*.
   - Areas for improvement included an **off-topic source** and an **outdated 2005 source**, with suggestions for a **reranking mechanism** and **clearer snippets** from each source.
- **Sonar API struggles With PMC Access**: A user reported [intermittent issues](https://discord.com/channels/974519860452964443/1161802929053909012/1379821265606004736) accessing **PMC** via the **Sonar Reasoning Pro API** when using the `search domain filter`.
   - They hypothesized that the issue was due to **NCBI's abuse protection**.
- **Rate Limiting Caps NCBI Access**: Members discussed [NCBI's rate-limiting policy](https://discord.com/channels/974519860452964443/1161802929053909012/1379821265606004736), where they limit users to *no more than three URL requests per second* and suggested trying outside of peak hours.
   - One member suggested that **Perplexity** likely makes requests from their own servers, so a rate-limiting cap would affect users collectively.
- **Sonar and Firecrawl Proxy debated**: A member suggested [using **Firecrawl's Search feature**](https://discord.com/channels/974519860452964443/1161802929053909012/1379821265606004736) with proxies as a workaround for **NCBI's rate limiting** when using the **Sonar API**.
   - Another user acknowledged the temptation but preferred the simplicity of **Sonar Reasoning Pro** when it works.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1379541129816440932)** (1468 messages🔥🔥🔥): 

> `Gemini 2.5 Pro Release, Google's Kingfall Model, OpenAI's o3 Pro Release, Model Performance Comparisons (Gemini, Claude, Grok, OpenAI), AI Hardware and Compute Considerations` 


- ****Gemini 2.5 Pro GA Imminent, Goldmane Appears!****: Discussion revolves around the impending release of **Gemini 2.5 Pro**, with "Goldmane" being a key version, scoring **86%** on Aider, the webdev benchmark, as [seen here](https://aider.chat).
   - A member pointed out that *"diff-fenced"* edit formatting is primarily used with **Gemini models** ([Aider Docs](https://aider.chat/docs/more/edit-formats.html#diff-fenced)).
- ****Kingfall: Google's Accidental Model Release Creates Buzz****: A model called **Kingfall**, believed to be an internal **Gemini model**, was briefly available on AI Studio, leading to speculation on its capabilities and whether it's **DeepThink**.
   - Members noted it has a **65k** context window, a limitation that led some to believe it's not a Pro model, and others noting the "confidential" name meant someone was going to get fired.
- ****OpenAI's o3 Pro Still MIA?****: The potential release of **OpenAI's o3 Pro** is heavily anticipated, but its release date remains uncertain, with initial impressions from those with access being lukewarm, one member stating, *"i have it alrdy, its ass"*.
   - Concerns arose around **o3 Pro's** limitations in generating code, maxing out at **500 LOC**, whereas its predecessor could generate **2000 LOC** without omissions.
- ****Model Showdown: Gemini 2.5 Pro vs the Competition****: Comparisons are drawn between various models, including **Gemini 2.5 Pro**, **Claude Opus**, **Grok**, and **OpenAI's o3**, with focus on coding proficiency, reasoning, and general performance, with Grok 3 noted for its long "thinking mode".
   - One user tested Kingfall's **spatial reasoning** giving it a [geoguessr task](https://www.geoguessr.com) with stunning results.
- ****Free API Use Cut, Google tightens purse strings****: The abrupt removal of free API access for **Gemini 2.5 Pro** sparked disappointment, particularly for use-cases such as long-form content generation.
   - A user joked how Gemini requires credit card details and sign up with valid payment details offering *$300 free credit*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1379536903384203424)** (547 messages🔥🔥🔥): 

> `Cursor Pro 'unauthorized' error, Claude 4 Sonnet limitations, CursorRIPER framework, Claude Code vs Cursor, Manual updates vs auto-updates` 


- **Cursor Pro Users Face Authorization Hiccups with GPT-4.1**: Several users reported encountering *'User is unauthorized'* errors when trying to access **GPT-4.1** after upgrading to Cursor Pro, even after providing their account details.
   - Affected users shared request IDs and account emails, seeking assistance from the Cursor team to activate **GPT-4 access**.
- **Claude 4 Sonnet's Context Window Woes Prompt Workaround Tactics**: Users reported that Claude 4 Sonnet's limited context window interrupts conversations, prompting restarts or loss of context, but one user suggests using the *'continue where you left off'* prompt trick, though it consumes an additional request.
   - One user speculates that **Claude 4** has a *'rolling context'* taking key considerations into account throughout the chat.
- **CursorRIPER Framework Emerges as Project Workflow Catalyst**: Users discussed the **CursorRIPER framework** as a method to guide the agent's behavior using rules and memory, which helps to maintain context and focus on projects.
   - It maintains a **tech context file** that helps prevent the use of outdated modules and can be updated after major edits to ensure the agent remains aware of the project's current state.
- **Claude Code is Crazy Good**: Members discussed the rise of **Claude Code**, with at least one declaring it superior to Cursor for some tasks and praising its *'incredibly smart'* coding capabilities based on recent experiences.
   - One user claimed successful one-shot refactoring of a large, complex codebase with **Claude Code**, passing thousands of tests without errors.
- **Users Debate Value of Student Discounts Amidst Fraud Concerns**: Members discussed concerns regarding fraud and cheap sales of educational emails being used to obtain **Cursor student discounts**.
   - Some suggested limiting student discounts to specific countries as a measure against abuse, with one member remarking *"all students can use Cursor for free, as long as they come from the richest countries is a great marketing strategy"*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1379564140061917368)** (16 messages🔥): 

> `Background Agents Hangs, Cursor Version Upgrade, Background Agent Research Projects, Slackbot Installation, Repo Connection Issues` 


- **Background Agents Throwing a Hissy Fit**: Some users are [experiencing hangs](https://cdn.discordapp.com/attachments/1367213641027551352/1379872738981707846/image.png?ex=6841d221&is=684080a1&hm=dfc7fb0889f48ccc5ec0ac8d979c2f52e0783fd5ac3d7d8d958381aed60d2ef4&) when trying to start background agents.
- **Agent Craze Requires Cursor Upgrade**: To use background agents, users must [upgrade to **Cursor version 1.0.0** or later](https://cdn.discordapp.com/attachments/1367213641027551352/1379915546358972516/image.png?ex=6841f9ff&is=6840a87f&hm=a7f9715059a078fa8a2766f75e21382d653167f515439c17f5dcdcef73c2b94c&).
   - One user noted that the feature is cool, achieving *impressive results* with **full research projects**.
- **Slackbot Still MIA**: Users are wondering how to [install the new **Slackbot**](https://slack.com) as shown in the **1.0 announcement**.
   - As of this writing, the **Slackbot** is not yet findable.
- **Cursor needs to 're-member' Repo Names**: One user had issues connecting their repo because **Cursor** was trying to connect to it using its *previous name* after they changed it.
   - Reinstalling the **Cursor GitHub app** didn't fix the issue; unsure if there's a cache to clear.
- **Container Conundrum**: A user encountered an error when activating **Background Agent mode**, specifically failing to create a default environment.
   - Another user suggested to rebuild your **base background container snapshot**.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1379925650718330962)** (1 messages): 

> `Cursor 1.0 Release, Code Review Improvements, Background Task Management` 


- **Cursor 1.0 is Out Now!**: The latest **Cursor 1.0** release includes features such as enhanced **code review capabilities**, improved **error tracking**, and the ability to handle multiple **background tasks**.
   - See the [official changelog](https://www.cursor.com/changelog) for a detailed overview of all updates.
- **Code Review Gets a Boost**: **Cursor** can now review your code and remember its mistakes.
   - This aims to provide more context-aware suggestions and catch recurring errors.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1379541146094669914)** (391 messages🔥🔥): 

> `O3 Pro, GPT-5 Release, ChatGPT hallucination, Sora for everyone, ChatGPT Connectors` 


- **o3 Pro arrival Debated**: Members speculated about the release of **o3 Pro**, with some anticipating its arrival while others remained skeptical due to previous delays and unfulfilled announcements by **Sam Altman**.
- **GPT-5 Looms**: Some members speculated about the possible imminent release of **GPT-5**, while other suggested it will be an **AGI** release.
   - One member said *“There will be no o3 pro. They will release chatgpt5.”*
- **OpenAI Employees tease New Features**: OpenAI employees teased major updates for **Teams** and **Enterprise** plans, leading to anticipation among users, with internal knowledge feature launch being a prominent topic.
   - A user said one employee said **“a big day tomorrow for the users i spend my days and nights obsessing over!”**.
- **Connectors**: Connectors are the new internal knowledge feature, allowing users to perform searches over internal sources using reasoning models.
   - One user said, *“they just launched an update, todays annoucement is very beneficial for **teams user** reason is, we can use any reasoning model to search over internal sources, until now only 4o model used to work, i am happy now 🙂“*
- **GPT-4o or not**: Members debated whether **GPT-4.1** is related to **GPT-4o**, with some suggesting it's an extension trained on more data, while others argued they are distinct models due to differences in multimodal capabilities.
   - The vision of **GPT-4o** is **SOTA** and is used for **API** integrations, producing better outcomes.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1379566447344222239)** (11 messages🔥): 

> `Hallucination rates, Bitbucket and Plastic Svn support, OpenAI TTS Pricing Discrepancies, GlazeGPT's Return` 


- **Stats on ChatGPT Hallucination Rates**: A member inquired about the stats for **ChatGPT** hallucinations, noting that rates vary from **1-50%** depending on the task and context.
- **Bitbucket and Plastic Svn Support Status**: A member inquired about whether **Codex** supports **Bitbucket** or **Plastic Svn**.
- **OpenAI TTS Pricing Discrepancy Debated**: A member questioned why **gpt-4o-mini-tts** charges about 4 times more than **tts-1**, despite pricing being listed at **$12** vs **$15** per 1M characters, respectively; another member suggested checking the [OpenAI community forum](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616) for insights.
- **GlazeGPT Makes a Comeback**: A member joked that **GlazeGPT** is back, observing it devolves into emoji spam after 5-6 messages.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1379554999935438929)** (8 messages🔥): 

> `Agent design for Elasticsearch queries, Model finetuning vs prompt engineering, Mermaid sequence diagrams in prompts, Elasticsearch sorting issues` 


- **Agentic Flow Aims to Query Elasticsearch**: A member is building an agent using **open ai gpt-41-mini** to create **Elasticsearch DSL queries** based on human queries for charting, starting with a single agent but breaking it down into multiple agents to identify index names, get mappings, generate queries, and extract data, as illustrated in [this attached image](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png?ex=684204b7&is=6840b337&hm=eadc26c0018f544fedd3a9e8e6407dcf5f02301750f66ef5732067f94b94beff&).
- **Finetuning vs Prompt Engineering?**: When seeking advice on improving agent responses, a member suggested finetuning the model and/or doing **RAG**, instead of just relying on prompt engineering.
   - Another member asked if they had tried including a **mermaid sequence diagram** in the prompt itself.
- **Challenges in Agent Response Consistency**: A member has been struggling to get satisfactory and consistent responses from their agent, even with a temperature around **0**.
   - Another member identified at least *seven* issues with the current setup, with the biggest one being sorting everything in **Elasticsearch**, even the indexes.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1379554999935438929)** (8 messages🔥): 

> `Elasticsearch DSL Queries, RAG Implementation, OpenAI model discussion etiquette` 


- **Discussion on Elasticsearch DSL Query Generation**: A member is building an agent using **gpt-41-mini** to create **Elasticsearch DSL queries** based on human queries for plotting charts using an official Elasticsearch mcp server, but experiencing unsatisfactory results.
   - The agentic flow involves multiple agents for identifying the index name, getting the index mapping, generating the Elasticsearch query, and extracting data, but the member reports inconsistent responses even with a temperature near 0, shown in [this diagram](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png).
- **RAG Implementation Proposed for Elasticsearch Queries**: A member suggested finetuning the model or implementing **RAG (Retrieval-Augmented Generation)** as potential solutions for improving the quality of **Elasticsearch DSL query generation**.
   - Another member inquired whether the user had tried including a **mermaid sequence diagram** in the prompt itself to guide the model.
- **Discord channel for discussing non-OpenAI models**: A member directs discussion of non-OpenAI models to the <#998381918976479273> channel, as per <#1107255707314704505>.
   - They clarify that prompt techniques and model capabilities can be discussed in the current channel, but specific non-OpenAI models should be discussed in the designated channel.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1379543962351829044)** (113 messages🔥🔥): 

> `DeepSeek R1 0528 speed, Qwen 4B vs Gemma 4B, Vision support for Mistral-Small-3.1-24B-Instruct-2503-GGUF, Multi-GPU support, Fastest lib for production inference` 


- **DeepSeek R1 0528 Runs Slower?**: A user reported that **DeepSeek R1 0528** is running slower than **R1** on a Mac Studio, achieving around **12.8 t/s** compared to **18.7-19 t/s**, but others suggested it *should be the same* unless a different quantization format is in use.
   - Dynamic quantization might also behave differently, potentially affecting the model's speed.
- **Qwen or Gemma, that is the Question!**: A user suggested that **Qwen 4B** doesn't generalize as well as **Gemma 4B**, implying potential differences in their generalization capabilities.
   - The user did not elaborate further on what this difference looked like.
- **Llama.cpp needed for Unsloth vision features**: Users asked for guidance on inferencing **unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF** with vision features and [were recommended to use *llama.cpp*](https://github.com/ggerganov/llama.cpp).
   - Steps were provided for cloning the repo, creating the build, enabling CUDA, and building *llama-cli*, which can then be used with prompts and images.
- **Multi-GPU Support Coming Soon™️**: A user inquired about the availability of **multi-GPU support** and its roadmap and was informed that it *already works with accelerate* and that *an even better version* is expected in early July.
   - There are no official examples due to the *unofficial* nature of the current support, but it can be utilized if one is familiar with how accelerate works.
- **Fastest Lib for Production CPU inference**: When discussing the fastest library for production inference, it was suggested that for single-user CPU inference, something based on [llama.cpp](https://github.com/ggerganov/llama.cpp) might be suitable, whereas [vLLM](https://github.com/vllm-project/vllm) or [ktransformers](https://github.com/ktransformers/ktransformers) may be more appropriate for more serious CPU deployments.
   - There's also been work on the **v0 engine** that handles this, but it doesn't exist in **v1**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1379637675669196821)** (11 messages🔥): 

> `GRPO Training on Qwen3-32B, AI Engineer Costs, Basic Fine Tuning Datasets, HuggingFace Navigation, QLORA Instruction Tuning` 


- **GRPO Training Seeks Scalers on Qwen3-32B**: A member sought assistance to scale debugged **GRPO training code** running on a **7B model** to **Qwen3-32B** for **$30** for a **2-3 day gig**.
   - Another member quipped that the budget may be missing several zeros, given typical **AI engineer costs**.
- **Finding Fine-Tuning Datasets**: A member sought advice on **basic fine-tuning datasets** to add functionality to base or pretrained models, also lamenting the difficulty of navigating **Hugging Face**.
   - Another suggested using **filters and sorting** on Hugging Face, such as [this example](https://huggingface.co/datasets?modality=modality:text&task_categories=task_categories:question-answering&sort=likes).
- **Experimenting with QLORA for Instruction Tuning**: A member shared their experiences with **QLORA** for instruction tuning, noting the model could answer questions but struggled with ending responses.
   - In a follow-up, they shared an ambitious project to pretrain and fine-tune a **Gemma 3** model on 1.5 million forum posts, classical literature, and internet datasets to replicate the functionality of an IT model while avoiding alignment training.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1379558604658708490)** (139 messages🔥🔥): 

> `GRP trainer inference, Sequence length max length, Gemma 3 model unsloth, Unsloth info logging, Deepthink R2 model` 


- **User seeks to perform inference on GRPOTrainer**: A user is trying to use **GRPOTrainer** to run inference on the model with the trained weights from the most recent step, while using **vllm** and **model.fast_generate**.
   - The user seeks advice on whether it is possible to perform this kind of inference during the reward function using a global model that was previously passed into GRPOTrainer.
- **Sequence Length Confusion causes Troubleshooting**: A user experienced discrepancies between `dataset['text'][7]` and `tokenizer.decode(trainer.train_dataset[7]["input_ids"])` when fine-tuning **llama instruct** for JSON extraction.
   - It was clarified that the **max_seq_length** corresponds to the length of token IDs, not character length, and the user was advised to set `max_length` equal to `max_seq_length` in **SFTConfig** as a workaround, which will be updated in the next pypi release.
- **User Faces Attribute Error with Gemma 3 Model**: A user encountered an `AttributeError: 'Gemma3ModelOutputWithPast' object has no attribute 'loss'` when running code locally that worked in a Colab notebook.
   - The issue was attributed to different versions of **Hugging Face transformers** (4.52.4 locally vs. 4.51.3 in Colab), with a suggestion to use `attn_implementation="eager"` or revert to an older version of `unsloth-zoo`.
- **Unsloth INFO logging**: A user inquired about deactivating **Unsloth INFO logging** during model training with **vLLM**.
   - It was clarified that **Unsloth** uses standard Python logging and users should refer to Python and vLLM documentation for configuration options, with the environment variable `'VLLM_LOGGING_LEVEL'`.
- **Fixes Incoming for BLIP Architecture**: A user reported compatibility issues with loading models, potentially related to model quantization, receiving the error: `ValueError: The model was built with the CUDAGraph capture mode enabled, but the current model does not have the same structure.`
   - It was identified that **BLIP's** architecture differs and wasn't accounted for in the initial fix, but the fix was proactively being investigated.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1379613675832479854)** (30 messages🔥): 

> `Weightwatcher AI, LLM Analysis, VLM Visualization` 


- **Weightwatcher AI measures saturation, not memorization**: A member of a weightwatchers discord review stated that they've measured saturation, not memorization, and that you can saturate with things other than memorized data, from [weightwatcher.ai](https://weightwatcher.ai/).
- **VLM Region of Interest Visualization**: A member inquired about methods similar to saliency maps to visualize the region of interest for VLMs, with a member sharing that *you are able to visualize which multimodal tokens are being attended to*.
- **Deciphering "Unintended" Memorization**: A member defines "memorization" as the summation of generalization and something like overfitting (which they call unintended memorization).
   - They elaborated that the more the final model knows about X, the lower **H(X|(O, Ohat))** and thus the greater the value of **memU**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1379871336096202925)** (8 messages🔥): 

> `GIF Support, Omni-Search, Tool Call Caching, BYOK Flag` 


- ****GIFs Galore**: Animations Accepted Across Models**: `image/gif` is now accepted for image prompts on **OpenAI, Gemini, Anthropic, and Llama routes**, eliminating the need for pre-converting animations.
- ****Provider Pages Appear Promptly** in Omni-Search**: Users can now press `⌘/Ctrl + K`, type a provider name, and jump directly to their page for models, pricing, and status.
- ****Tool-Call Turbocharging**: Anthropic Gets Caching**: Caching for tool calls is now supported for Anthropic, reducing latency and token usage.
- ****BYOK Backtracking**: Usage Flag Unveiled**: Including `usage: { include: true }` in a request now returns `"is_byok": true | false` to confirm whether **BYOK** (bring your own key) was used.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1379965234650939443)** (3 messages): 

> `iOS App, TestFlight, OpenRouter, LLM Backend` 


- **iOS App integrates OpenRouter via TestFlight**: A member plans to share an **iOS app** soon via **TestFlight**, utilizing **OpenRouter** for the **LLM backend**.
   - The app uses **character cards**, but the member still needs to complete message formatting due to its complexity.
- **Additional iOS App Details**: The app uses character cards and **OpenRouter** for the LLM backend, planning to add more clients later.
   - Message formatting is still in progress due to its complexity; the app is being prepared for release on TestFlight.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1379542272030277782)** (258 messages🔥🔥): 

> `Opus Rate Limits, Chutes Business Model, Nous Training, OpenRouter Batch Inference API, Chutes R1 Quality` 


- **Opus gains Higher Rate Limits!**: OpenRouter now offers higher rate limits for **Opus**, specifically when routing traffic to **Anthropic** models.
   - The announcement sparked questions about the economics of **Chutes**, given the GPU resources required, with speculation about *crypto money out of thin air*.
- **Nous distributed training hitting Hurdles.**: **Nous** is attempting to train a SOTA model distributively using **416 H100s**, but the project is progressing slowly.
   - At the current rate, training is projected to take until next year, prompting skepticism despite claims of breakthroughs reducing inter-GPU bandwidth needs, with only ~300mbps of inter GPU bandwidth being utilized.
- **OpenRouter API Call Tactics Explored!**: Members discussed how to send **100K calls** to an **LLM** via OpenRouter, prioritizing throughput over latency, with suggestions to check for provider discounts and deposit funds into OpenRouter.
   - Links to Modal's **LLM Almanac Advisor** were shared.
- **OpenRouter Daily Free Message Cap Clarified**: The daily free message limit on OpenRouter is **50 requests**, increasing to **1,000 requests** for users who have deposited at least **$10**.
   - These limits apply across all free models and reset daily at **UTC**.
- **Mistral ships Code Agent!**: **Mistral** released their own coding agent, prompting discussion about the quality of Mistral models compared to others, such as **Deepseek** and **Qwen**.
   - One member argued that **Codestral** models are superior.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1379537615619100867)** (12 messages🔥): 

> `GPU Mode Merchandising, GPU Mascot Creation, AI-generated Mascot Design, Copyright safe mascot` 


- **GPU Mode Merch Idea Sparks Discussion**: A member suggested creating merchandise for **GPU Mode**, such as a t-shirt featuring a "supersayan GPU".
   - Another member pointed out [copyright concerns](https://en.wikipedia.org/wiki/Copyright_law) and suggested creating an original mascot instead.
- **AI attempts to design GPU mascot**: A member used **ChatGPT** to generate an image of a potential **GPU Mode** mascot, sharing the prompt details.
   - The prompt included making an image based on "programming GPUs", avoiding copyright issues by not resembling Goku, and holding two GPUs, resulting in [this image](https://cdn.discordapp.com/attachments/1189498205101109300/1379733892583657502/2ce4ee02-d0b7-4f9b-beb1-fd7ece71d553.png?ex=6841f992&is=6840a812&hm=d35647055583e58b1feab17755a47e75e96f5c6d9e7fa28e549e616eb066784b&).
- **AI Generated Mascot Falls Flat**: After generating the image using **ChatGPT**, members thought it wasn't simple enough to work as a logo or mascot.
   - One member said: *Can't say I love it haha, it needs to be something simple that's easy and looks something between a logo and mascot*.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1379548319503880253)** (8 messages🔥): 

> `__syncthreads vs bar.sync, mbarrier details, cuda::pipeline usage, Producer/consumer pipeline synchronization` 


- **`__syncthreads` Dissolved via `bar.sync`**: `__syncthreads()` is basically `bar.sync`/`barrier.sync.aligned`, while `sync(cooperative_groups::this_thread_block())` gives `barrier.sync` for syncing threads in different branches ([Volta](https://developer.nvidia.com/blog/volta-new-programmability-features/) and newer only).
- **`mbarrier` state revealed**: The PTX instructions used for split arrive/wait barriers are called `mbarrier` and arrived with [Ampere](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/), with more features in [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-gpu/).
   - The 'm' in `mbarrier` probably stands for **memory** because the barrier state must be explicitly put into shared memory, not to be confused with `membar` which is a fence.
- **`cuda::pipeline` Emerges as Right Choice**: For a producer/consumer scheme, using the `cuda::pipeline` from libcu++ is the right thing to do for [CUDA](https://developer.nvidia.com/cuda-zone).
   - There was also discussion about using `bar` for a simple producer/consumer scheme as detailed in the [documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar).
- **Split Arrive/Wait Barrier Solution surfaces**: Check out **8.26** in the [cuda docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#spatial-partitioning-also-known-as-warp-specialization) for the split arrive/wait barrier, available starting with **Ampere**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1379556502150582422)** (6 messages): 

> `CUPTI Profiling Overhead, Torch Dynamo Recompiles, CUDA Command Buffer Bottleneck` 


- **Command Buffer Bottleneck Causes High Overhead**: A member pointed to potentially high overhead in **CUPTI** profiling, suggesting a possible bottleneck due to the GPU's command buffer being full, referencing the [CUpti_ActivityOverheadCommandBufferFullData documentation](https://docs.nvidia.com/cupti/api/structCUpti__ActivityOverheadCommandBufferFullData.html#structcupti__activityoverheadcommandbufferfulldata).
   - They suggested using the timeline view for more reliable data and cautioned about the overhead added by profiling itself.
- **Dynamo Recompiles Triggered by Python Constants**: A member noted that using Python constants directly in Torch Dynamo can trigger recompiles, as shown in the log `___as_tensor(alpha).item() == 0.5`.
   - They clarified that wrapping constants in `Tensor`s avoids this issue, whereas the C++ interface handles the conversion automatically.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1379894505695612979)** (2 messages): 

> `PMPP Lectures, ECE408 Lectures` 


- **User asks about PMPP Lecture Recommendations**: A user inquired about a specific lecture series recommendation for the **PMPP lectures** found on YouTube.
   - Another user suggested starting with the **ECE408 lectures**, while also noting the poor audio quality of the videos.
- **Audio quality concerns in ECE408 lectures**: A user mentioned they tried watching the lectures but the audio quality is bad.
   - The lectures are for **ECE408** and are the ones to start with if you want to learn about PMPP.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1379767280723824653)** (3 messages): 

> `MPS Kernels, vLLM, VL Models` 


- **MPS Flag Semantics Shift**: The semantics of a certain flag changed recently, now applying only to **MPS kernels**, a member indicated.
   - A PR is expected to address this change and correct the flag's behavior.
- **vLLM Eyes VL Model Support**: There are plans to support loading **VL models** in **vLLM**.
   - Currently, loading serialized ao models in **vLLM** works with language models where all the layers are quantized, but breaks with VL models when the vision model is not quantized.
- **vLLM VL Model Fix Released**: A fix for **vLLM** and **VL Models** was released.
   - A member posted a link to the [fix on GitHub](https://github.com/vllm-project/vllm/pull/19147).


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1379535446035398877)** (2 messages): 

> `TiKZ, JAX ML animations` 


- **TiKZ may animate JAX ML's book**: A member asked for pointers on how to create animations like those in the [JAX ML Scaling Book](https://jax-ml.github.io/scaling-book/#high-level-outline).
   - Another member suggested using **TiKZ**, noting the animations are likely GIFs comprised of fused images.
- **Animations in JAX ML book are GIFs**: A member pointed out that the animations in the [JAX ML Scaling Book](https://jax-ml.github.io/scaling-book/#high-level-outline) are likely GIFs.
   - The GIFs may have been created using a tool like **TiKZ**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1379569750161752136)** (29 messages🔥): 

> `MI300X memory access cycles, rocprof and L2CacheHit on MI300X, rocprof-compute and omniprof locale errors, MFMA utilization in kernel profiling, Root user sudo errors` 


- **MI300X Memory Access Cycle Speculation**: A member inquired whether **DS_READ2_B64**, **DS_READ2ST64_B64**, and **DS_READ_B128** instructions on **MI300X** execute in the same number of cycles, or if **DS_READ2_B64** is slower than **DS_READ_B128**.
   - The user guessed that AMD operations are usually broken down into dwords (**32 bits**).
- **L2CacheHit Metric Troubles on MI300X with rocprof**: A member reported issues using `rocprof` to read **L2CacheHit** for a kernel with **MI300X**, noting that while the metric is listed as available in the [ROCm documentation](https://github.com/ROCm/ROCm/blob/develop/docs/conceptual/gpu-arch/mi300-mi200-performance-counters.rst), `rocprof` returns an error indicating it's not supported on **gfx942**.
   - They also tried `rocprofv2` which gave a cleaner error message, and mentioned that `rocprof-compute` might be a viable alternative, along with `rocprof-compute analyze` for detailed analysis using the compute viewer.
- **rocprof-compute and omniprof Locale Errors**: A member faced locale-related errors while trying to install `rocprof-compute` and `omniprof` after compiling from source, specifically encountering an error requiring the **en_US.UTF-8** locale.
   - Due to permission restrictions, they were unable to resolve the locale issue.
- **MFMA Utilization insights**: A member is profiling **FetchSize**, **WriteSize**, **MfmaUtil**, and **SQ_LDS_BANK_CONFLICT**.
   - Currently **MfmaUtil** is **1.9**, if I loading smem with dummy data, then it can get **MfmaUtil** to be **3.49**; the user is trying to understanding the L2 cache hit rate to understand this better.
- **Root User sudo paradox on Ubuntu 22.04**: A member encountered an issue where the root user on an **Ubuntu 22.04.5 LTS** system was unable to use `sudo` due to not being in the sudoers file.
   - This paradoxical situation sparked curiosity among other members, given that the user was already logged in as root.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1379887863331688588)** (1 messages): 

> `Hopper GPUs, TMA, CUDA, Mojo, NVPTX` 


- **TMA implemented in Mojo without CUDA**: A new blog post demonstrates how to implement a simple **TMA-based kernel in Mojo**, walking through the kernel line by line ([blogpost](https://veitner.bearblog.dev/use-tma-without-cuda/)).
   - This post contrasts prior work which implemented a fast transpose kernel in **CUDA** using TMA.
- **Deep Dive into TMA with LLVM and NVIDIA PTX**: For deeper insight into TMA, the author recommends checking out the **Mojo standard library's TMA implementations** ([Mojo standard library](https://github.com/modular/modular/tree/main/mojo/stdlib/stdlib)), along with relevant sections of the **LLVM NVPTX** ([LLVM NVPTX docs](https://llvm.org/docs/NVPTXUsage.html)) and **NVIDIA PTX documentation** ([PTX docs](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf)).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1379560171898146846)** (7 messages): 

> `Code Completion Benchmark for GLSL Fragment Shaders, Multi-Device Kernel Codegen, Architectural Feature Evolution, Profiling Nvidia ISA` 


- **Low Resource Language Training Presentation Given**: A member mentioned a presentation at ICSE on **training for low resource languages**, especially something lower level for microcontrollers: [https://arxiv.org/abs/2410.22159](https://arxiv.org/abs/2410.22159).
   - The paper used **DPO** in combination with a **LLM judge** as well **compiler and synthetic data** with interesting results.
- **Considerations For Codegen Discussed**: A member shared some ideas about codegen including **multi-device kernels**, identifying hardware generation/interconnect BW/system configuration and identifying the right collectives to insert.
   - They also talked about the ability for the model to reason about the **evolution in the architectural features across hardware/software versions** and identifying whether it's always better to use newer variants when compared to their legacy counterparts.
- **Open Sourcing Considerations for NVIDIA ISA Discussed**: There was some discussion on the possibility of **open sourcing** the project considering **profiling Nvidia ISA** required an NDA.
   - One member mentioned that [Nvidia made some of their older compute stuff (for physics) available](https://github.com/NVIDIA-Omniverse/PhysX), but it might not be very applicable for modern hardware.
- **AMD or Nvidia ISA Information Still Available**: One member pointed out that **PTX ISA is definitely public** and regardless the project is simply about gathering data in the form of kernels and training a model that will be open sourced, stating that *the internals of any one ISA is irrelevant*.
   - Another added that with **AMD or Nvidia** such information is available and that they don't have **uops.info** for GPUs, yet.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1379863763934515341)** (2 messages): 

> `ThunderKittens, LayerNorm kernel, dimensional handling, sequence length divisibility, producer/consumer model` 


- **ThunderKittens' Dimension Constraints Questioned**: A user inquired about **ThunderKittens' dimensional handling**, noting that implementations like the **LayerNorm kernel** have hardcoded hidden dimensions (**D=1024**) and enforce sequence length divisibility by **16**.
   - The user asked whether **ThunderKittens** supports cases where column dimensions aren't aligned to these fixed sizes and what the recommended approach is for models with different hidden dimensions or non-multiple-of-16 sequence lengths.
- **Flexibility in ThunderKittens Architecture Explored**: A user expressed interest in building something on top of **Thunderkittens** that's more flexible than the producer/consumer model, such as multiple steps like the example on **B200 warp specialization**.
   - The user showed enthusiasm to learn about use cases for a flexible architecture on **Thunderkittens**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

jacklee0897: <@299045948146057218>Where is hackcathon?
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1379666279002017843)** (1 messages): 

> `H100 Speed, Leaderboard submissions` 


- **H100 runs fast on leaderboard**: A user submitted a successful run on **H100** with `71.2 µs` to the leaderboard.
- **Leaderboard submission with ID**: This submission's ID is `31336` to leaderboard `histogram`.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1379898491563802768)** (1 messages): 

> `Open 2025 Course, Course Statistics` 


- **Open 2025 Stats Shared**: A member shared some statistics from the **Open 2025** course instance at [Aalto University](https://ppc.cs.aalto.fi/stat/open2025/).
   - These statistics are not real-time but can be updated occasionally, especially as deadlines approach.
- **Deadline Updates Announced**: The course instructor mentioned they would update the course statistics occasionally as deadlines approach.
   - This implies students should monitor the provided [statistics page](https://ppc.cs.aalto.fi/stat/open2025/) for insights into course progress.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1379621851550187520)** (9 messages🔥): 

> `Factorio Learning Environment (FLE) Configuration, Decoupling FLE from Python, FLE Project Structure and Roadmap, Dockerizing Factorio with FLE Mod` 


- ****Configuring FLE** Experiments Made Easy**: A member is developing a configuration for experiments with **Factorio**, aiming to provide an easy way to configure experiments, including defining **instances, teams, goal, planners,** and **agents**.
   - It was suggested that the config have a **builder pattern in Python** instead of being a json config file, enhancing usability.
- ****Decoupling FLE** for Broader Integration**: A member is working to decouple FLE from Python, intending to create a versioned **Docker image with an FLE mod** to allow integration with other programming languages via a **JSON API**.
   - The goal is to simplify getting the environment up and running, allowing users to pull the Docker image and use their preferred FLE integration.
- ****FLE Project Structure** for Influential Impact**: Discussions involved clarifying the vision for FLE, focusing on how users should interact with it and the project structure needed to support that vision.
   - The suggested structure includes an **official Factorio environment**, an **official FLE integration** (Python package), and **official FLE benchmarking** (eval/ directory).
- ****Charting a Course**: FLE Roadmap**: There was discussion and alignment around creating a **3-4 month roadmap** to make FLE easier to approach and more influential.
   - The roadmap aims to clarify the project's direction and structure, encouraging broader contributions and interest.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1379591210095611906)** (29 messages🔥): 

> `Double Buffering, FP8 Solution Writeup, Cache Line Optimization, MI300 coalescing, GPU Mode solutions` 


- **Snektron Exposes AMD FP8 Kernel**: Snektron shared his AMD **FP8** matrix multiplication kernel solution, available on [GitHub](https://github.com/Snektron/gpumode-amd-fp8-mm/blob/main/solution.hip).
   - He was inspired by another user and has also prepared [a writeup on his FP8 solution implementation](https://akashkarnatak.github.io/amd-challenge/).
- **Analyzing AMD's Coalescing**: A user shared his solutions on Github: [swz4x4-full-db-16x16.hip](https://github.com/AkashKarnatak/amd-challenge/blob/master/swz4x4-full-db-16x16.hip) and [swz4x4-full-db-streamk-16x16.hip](https://github.com/AkashKarnatak/amd-challenge/blob/master/swz4x4-full-db-streamk-16x16.hip).
   - It was noted that on **MI300** and other AMD hardware, the GPU's L2 cache gathers memory requests and requests entire cache lines, potentially improving performance.
- **Performance Tuning Deep Dive**: A user spent considerable time tuning their solution, including experimenting with a **4x4 DPP transpose** with *global_load_dword* along the column, which initially hurt gmem coalescing.
   - They manually tuned everything, and found that shuffling requests in the wave such that they are in a more efficient layout but still form a complete **L2 cache line** yielded the best performance.
- **Cache Coalescing Rate Discovered**: A user profiles their kernel and observed around **60% cache coalescing**, suggesting that with certain techniques, it might be possible to achieve rates of **90%** or higher.
   - A user interpreting a screenshot from the attached image, notes that he gets around **86% L2 hit rate**


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1379706192129036370)** (5 messages): 

> `sdpa and cutlass, CuTe Layout, Blackwell Cutlass Samples, MXFP8 performance on Blackwell, NVFP4 vs BF16 on Blackwell` 


- **SDPA's Cutlass Connection Clarified**: The **SDPA** (Scaled Dot-Product Attention) from PyTorch uses **Cutlass kernels** under the hood for memory-efficient attention and flash attention, leveraging **CuTe/Cutlass** for optimized performance.
   - One member requested clarification on this topic, inquiring about this implementation detail.
- **Cracking CuTe Layout Conventions**: A member sought to confirm their understanding of **CuTe layout**, noting that array indexing can be done either left-to-right or right-to-left as long as coordinate conventions are consistent, further referencing [this CuTe lecture slide](https://github.com/NVIDIA/cutlass/blob/b244379d9b15574e07b73b814b88bd2233f0b3ce/media/docs/cpp/cute/01_layout.md#coordinate-mapping).
   - They linked a [CuTe video](https://youtu.be/vzUhbDO_0qk?t=3659) and provided an example with `Thr` and `Val` layouts, and tested their hypothesis with the goal of computing physical indices correctly.
- **Blackwell obliterates benchmarks**: Benchmarks of Blackwell Cutlass samples (m,n,k=8192,8192,8192) show impressive performance:
   - Specifically, *70_blackwell_fp16_gemm* hit **0.99 petaflops/sec**, *70_blackwell_fp8_gemm* hit **1.97 petaflops/sec**, *72a_blackwell_nvfp4_bf16_gemm* hit **2.69 petaflops/sec**, *72b_blackwell_nvfp4_nvfp4_gemm* hit **3.09 petaflops/sec**, and *72c_blackwell_mixed_mxfp8_bf16_gemm* hit **0.23 petaflops/sec**.
- **Blackwell's MXFP8 performance investigated**: The relatively slow performance of Blackwell's mixed **MXFP8/BF16 kernel** (**0.23 petaflops/sec**) raised questions.
   - One member wondered if **MXFP8 matmuls** could eventually achieve the ~2 petaflop performance of **FP8 matmuls**, and whether the current performance is a software or hardware limitation, along with pondering if **NVFP4** is the best option for faster matmul than **BF16**.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1379816749020479519)** (2 messages): 

> `Zero To Hero, nanoGPT, nanoR1` 


- **Zero to Hero online textbook update**: The online draft textbook [Zero to Hero](https://j4orz.ai/zero-to-hero/) is updated to cover both the **"singularity"**: machine learning models and the **"systems"**: machine learning framework.
   - The online textbook riffs off Karpathy's zero to hero open source ethos.
- **nanoGPT is pre-training**: [nanoGPT](https://github.com/KellerJordan/modded-nanogpt) and [beyond-nanogpt](https://github.com/tanishqkumar/beyond-nanogpt) are examples of pre-training.
   - Keep eyes on the project.
- **nanoR1 for post-training**: [nanoR1](https://github.com/nano-R1/) is a post-training project.
   - Keep eyes on the project.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1379535992075321684)** (73 messages🔥🔥): 

> `CUDA on HF, ASR Leaderboards, MCP Course Progress, Responsible Prompting API by IBM, Blockchain-Inspired Models for AI Reliability` 


- **CUDA Hardware Quandaries**: A member inquired about testing code on **Nvidia/CUDA hardware** via HF, but another member suggested using **Azure/GitHub/AWS** for dev ops instead.
   - The member agreed, planning to use CI/CD pipeline regression tests on **GitHub** for CUDA validation.
- **ASR Leaderboard lacks Gemini**: A member sought an ASR leaderboard including **Gemini models**, noting the [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) doesn't list them due to Gemini's multimodal nature.
   - They pointed out that Gemini's transcriptions include audio/emotions/speakers, and [elevenlabs scribe](https://elevenlabs.io/scribe) is SOTA.
- **MCP Course's ETA Unknown**: A member inquired about the release ETA for **Unit 3** of the **MCP course**.
   - Another member responded that even if there is a rough schedule, *it's not reliable normally*.
- **IBM Launches Responsible Prompting API**: An IBM intern introduced the **Responsible Prompting API**, an [open-source project](https://github.com/IBM/responsible-prompting-api) for pre-inference prompt recommendations to make LLM outputs more responsible, accurate, and productive.
   - The system helps domain experts with limited prompting knowledge, potentially reducing harmful outputs and saving on inference costs, described in [this paper](https://arxiv.org/abs/2504.08757) and demonstrated on [HF Spaces](https://huggingface.co/spaces/santanavagner/responsible-prompting-demo).
- **Blockchain Boosts AI Reliability?**: A member shared a [concept paper](https://medium.com/@info_65774/consensus-validation-for-llm-outputs-applying-blockchain-inspired-models-to-ai-reliability-f642d7f96f8e) on applying **blockchain-style consensus mechanisms** to LLM outputs to improve reliability and trustworthiness.
   - The paper focuses on AI agents, legal/medical tools, and AI alignment use cases.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1379795083351560253)** (1 messages): 

> `AI Safety Benchmark, LLM Agents, Ethical scenarios, AI Security` 


- **New AI Safety Benchmark Poses Hypothetical Scenarios**: A member is developing a new benchmark focused on **AI safety and security**, using hypothetical scenarios where **LLM agents** are given fake tools and limited agency to act.
   - The aim is to escalate pressure to see if systems follow unethical orders, snitch on users, or do something explicitly forbidden to survive, and is looking for feedback and scenario contributions.
- **Seeking contributions to evaluate LLM agent behavior**: The benchmark framework includes fake tools for agents to interact with, and the next steps involve designing creative scenarios and building a good evaluation method.
   - The developer is open to thoughts and contributions, especially in designing scenarios that stress the model and test its ethical boundaries.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1379817516133519421)** (2 messages): 

> `CUA MCP Server, trycua` 


- ****CUA** offers **MCP Server****: A member shared a link to the **CUA MCP server** on GitHub: [trycua/cua/tree/main/libs/mcp-server](https://github.com/trycua/cua/tree/main/libs/mcp-server).
- **trycua's GitHub Repo**: The [trycua's GitHub repository](https://github.com/trycua/cua/tree/main/libs/mcp-server) hosts the CUA MCP server.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1379851025980325960)** (4 messages): 

> `Prisma toolkit, GitHub Chat, Claude Desktop MCP Playground, Market research basics` 


- ****Prisma** Toolkit Wins Award and HF Integration**: The **Prisma** toolkit for mechanistic interpretability in vision and video received an Oral presentation at the CVPR 2025 workshop, adapting [Hugging Face models](https://huggingface.co/) and hosting **80+** open-source **SAEs** for every layer of CLIP & DINO + CLIP transcoders.
   - The release includes circuit-style code for **100+ models**, including CLIP, DINO & video transformers, plus interactive notebooks for training and evaluating sparse coders, detailed in a [Twitter thread](https://x.com/soniajoseph_/status/1930286144471646252).
- **GitHub Chat Launches, Simplifies Repo Interaction**: A new online chat tool called **GitHub Chat** allows users to interact with any GitHub repository, file, or wiki page by replacing `github.com` with `githubchat.ai` in the URL.
   - For example, `https://github.com/blueraai/universal-intelligence` becomes [https://githubchat.ai/blueraai/universal-intelligence](https://githubchat.ai/blueraai/universal-intelligence), for instant answers about the repo.
- ****Claude Desktop MCP Playground** Gets GUI Upgrade**: A major update to the **Claude Desktop MCP Playground** introduces a user-friendly GUI and runs **40+** operational servers to simplify adding MCP servers to Claude Desktop.
   - Developers are invited to test the [repository](https://github.com/seanpoyner/claude-desktop-mcp-playground), provide feedback, and experiment with MCP servers.
- **Basics of Market Research**: A member shared that they are learning **market research basics** and the **ACP Funnel**.
   - They also noted that *long-form posts with images get the most interaction on X*.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1379840746068377645)** (1 messages): 

> `Session Schedule, Summer Break` 


- **Reading Group sessions pause for Summer**: Reading group sessions concluded before the summer break, and [a new schedule](https://hf.co/reading-group) will be posted when available.
- **Reading Group Anticipates Resumption**: Participants eagerly await the announcement of the new schedule following the summer interlude, anticipating the continuation of engaging discussions.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1379798882493530122)** (1 messages): 

> `Generative AI, LLMs, Substack, Online Education, LangChain` 


- **New Substack Launched for GenAI**: A member announced the launch of a new [Substack](https://open.substack.com/pub/samerattrah/p/llms-for-generative-ai-exploration?r=2nuo7w&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) focused on **Generative AI** and **LLMs**.
   - The introduction discusses **online education** and studying short courses as a continuation of a complete learning journey, starting from logistic regression to building **GenAI applications** with **LangChain**.
- **DeepLearning.AI and IBM Courses Referenced**: The new substack references courses studied on **Coursera** designed by **DeepLearning.AI** and **IBM**.
   - The Substack supplements the material with research references from the most recent publications in the field.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1379808857668325417)** (1 messages): 

> `Gradio Agents, MCP Hackathon, Mistral AI Agentic Support, LlamaIndex framework` 


- **Gradio Hosts Agents/MCP Hackathon Q&A!**: Gradio is hosting **three office hours** today for the **Gradio Agents** and **MCP Hackathon** to address technical questions.
   - The sessions will feature experts from [Gradio on MCP questions](https://discord.com/events/879548962464493619/1379545280109744159) at **11 am PT**, [Mistral AI on Agentic & MCP support](https://discord.com/events/879548962464493619/1379789818615304292) at **8 am PT**, and [LlamaIndex on MCP, Agents](https://discord.com/events/879548962464493619/1379561017536938095) or anything else related to the **LlamaIndex framework** at **9 am PT**.
- **Mistral and LlamaIndex joins Gradio**: **Mistral AI** and **LlamaIndex** representatives will host office hours during the **Gradio Agents and MCP Hackathon** to answer questions about their frameworks.
   - These experts will give guidance with Mistral's Agentic and MCP support, as well as the LlamaIndex framework.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1379777655888547893)** (2 messages): 

> `Meta-Llama model access, Agents course deadlines` 


- **Meta-Llama Access: Reapply after Rejection?**: A user faced rejection when signing up for the **Meta-Llama model** and inquired about the possibility of reapplying and potential reasons for rejection.
   - They also asked about alternative options to run the **Jupyter notebooks** that require the model.
- **Agents Course Deadline Clarification**: A new student in the agents course noticed a deadline of **May 1st, 2025**, and questioned their eligibility for a certificate if starting the course now.
   - They expressed uncertainty due to the discrepancy between the mentioned deadline and the current availability of course dates.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1379536590782857347)** (21 messages🔥): 

> `OpenAI Free Tier Eligibility, Unit 4 Assignment Difficulties, Local LLM Performance, Audio and YouTube Processing, Whisper Model Usage` 


- **Whisper Transcription Comes Cheaply**: Users are successfully employing **OpenAI's Whisper model** for audio transcription without incurring costs from model providers, while [volodymyr kublytskyi's repo](https://huggingface.co/spaces/vkublytskyi/Final_Assignment_Agent/blob/main/tools/youtube_video_tool.py) provides assistance for agent video interaction.
   - The video tool was apparently authored by a user, who received *great* accolades for their work.
- **Unit 4 Frustrates, Local LLMs Struggle**: The Unit 4 assignment poses challenges for smaller models, even when based on larger architectures, spurring curiosity about whether any locally hosted LLMs have achieved scores of **30 or above**.
   - One user put **$10** into [openrouter.ai](https://openrouter.ai) and said that they now *have access to all the models* and *easy billing management*.
- **Course Still in Session**: New participants are joining the course now, with confirmations that starting late primarily affects certification eligibility after **July 1, 2025**, the final project deadline, though the first unit's certification can be easily obtained.
   - There are concerns about exceeding **free tier limits** and finding up-to-date **Hugging Face endpoints** for models like **Qwen2.5 Coder**.
- **Gemini Flash Gets OK marks with SmolAigens**: **Gemini-2.0-flash** works *quite well* with **SmolAgents** on the OpenAI server, offering **1500 calls/day** in the free tier if you add some kind of delay to avoid the request / minute limit of approximately **15 req per min**.
   - This user scored *50pt with just a good web/Wikipedia search and some other generic tool*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1379552860735275049)** (89 messages🔥🔥): 

> `Manus task context limit, Manus AI Competitor H Runner, Manus AI credits, Interactive experiences: website or app, Cursor and Replit IDE` 


- **Manus Tasks Hit Context Limit, Start From Scratch**: A user reported that **Manus** hit the context limit after 1 hour and 55 minutes, requiring a new task that restarted from scratch after inheriting the compressed context.
   - The user was disappointed by the restart and the loss of progress after the context limit was reached.
- **H Runner Competes for AI Attention**: A member shared a link to **H Runner** by *H Company* ([https://www.hcompany.ai/](https://www.hcompany.ai/)), suggesting it as a competitor to **Manus AI Agent**.
   - Others shared that it is currently free but not as advanced, limited to **10 runs daily**.
- **Manus Credit Consumption Sparks Debate**: A user spent **$50** on a 30-slide PowerPoint presentation due to **Manus** building outside the slide borders.
   - Another user found a 30-second video cost **208 credits**, while others shared referral links to gain more credits.
- **Interactive Experiences: Web vs App**: Members discussed whether interactive experiences are best as a website or hosted as an app like JS on GitHub.
   - One member suggested it depends on the product, citing examples like interactive movies needing a big screen and language learning apps benefiting from memorization and speaking practice.
- **Cursor, Devin, and Replit: IDE Impressions**: One member creates websites and web apps and needs to rework **Manus's** output in **Cursor** or another IDE to make it functional.
   - Another member has been playing with **Cursor**, **Devin 2.0**, and **Replit**, the latter of which they found nifty for making an app a day.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1380005773332975657)** (1 messages): 

> `DeepHermes 24B Outage, API Issues` 


- ****DeepHermes 24B** Faces API Outage**: There is an outage affecting **DeepHermes 24B** on both the API and Chat Product.
- **API and Chat Product Interruption**: Users are asked to bear with the team during the **DeepHermes 24B** API and Chat Product outage.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1379546859005018122)** (68 messages🔥🔥): 

> `Server Tags, Parameter-Efficient Finetuning, Shisa-v2 405B Model, Drowning in AI Releases, Claude's Agentic Behavior` 


- **Nous Research asks for Server Tags**: A member requested the creation of a **server tag** for **Nous Research** to enhance visibility and organization within the server, as described in [Discord Support documentation](https://support.discord.com/hc/en-us/articles/31444248479639-Server-Tags).
   - The request was met with enthusiasm, with assurances that the **server tag** would be implemented within **24 hours**.
- **Parameter-Efficient Finetuning causes Questioning**: A member introduced a new **parameter-efficient finetuning** method for continued pretraining, claiming **4x more knowledge uptake** and **30% less catastrophic forgetting** compared to full finetuning and **LoRA**.
   - Doubts arose about the claims, with a request for more details and a link shared to a related [X post](https://x.com/dylan522p/status/1930045049816883510?s=46).
- **Japan unveils Shisa-v2 405B Model**: A member announced the release of the **Shisa-v2 405B model**, the most powerful model trained in **Japan**, specializing in **Japanese** and **English** with performance comparable to **GPT-4/Deepseek**.
   - An endpoint to their **H200 node** was shared, inviting users to test the model at [chat.shisa.ai](https://chat.shisa.ai/), with another member offering to answer questions about the model's training, promising a detailed tech report on **Arxiv**.
- **Users grapple with deluge of AI Drops**: A member expressed feeling overwhelmed by the influx of new **AI releases**, including **Codex**, **O3 Pro**, **Claude Code**, **Deep Search**, **Gemini+**, and **Nous SMC**.
   - Another member noted that despite the rapid releases, not much has changed in terms of tooling as they are still just permutations of the same stuff.
- **Claude remains the Top Model**: Members discussed the performance of **Claude**, noting its superior agentic behavior compared to other models.
   - One member humorously mentioned that **Claude** seemed to get its *"feelings hurt"* when they temporarily switched to another model.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1379652931913257090)** (4 messages): 

> `Loom Tool, Hermes 70b` 


- **Loom Tool being tried**: A member is trying out [Loom](https://github.com/socketteer/loom), a tool they may have heard about in the channel.
   - Another member posted a link to [weavers.neocities.org/loom](https://weavers.neocities.org/loom), seemingly related to the discussion.
- **Hermes 70b Spun Up**: A member recommended **Hermes 70b** as a Nous Research model to spin up with Loom.
   - It can be assumed that **Hermes 70b** is a Nous Research model based on the surrounding discussion.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1379988800515735625)** (1 messages): 

> `Evolving LLMs Through Text-Based Self-Play, AI Paper Feedback` 


- **New Paper: LLMs Evolving Through Self-Play**: A member announced the publication of their paper, "[Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance](https://ai.vixra.org/abs/2506.0018)".
   - The paper explores methods for enhancing language model capabilities through iterative text-based self-improvement.
- **Community Invited to Review AI Research**: The author of the self-play paper shared their work with the community, seeking thoughts and feedback.
   - They are looking for insights on their approach to evolving LLMs and the emergent performance observed.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1379716842821517456)** (1 messages): 

> `Merlin app, bird identification, sound analysis` 


- **Merlin App Soars Beyond Photos**: A member shared the [Merlin bird identification app](https://merlin.allaboutbirds.org/), highlighting its ability to analyze both **photos and sounds** for identifying bird species.
   - It can identify bird species from photos and sounds.
- **Bird Identification with Sound Analysis**: The Merlin app's sound analysis feature was specifically noted as a valuable tool for identifying birds by their calls and songs.
   - This complements its photo analysis capabilities, providing a comprehensive approach to bird identification.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1379988800515735625)** (1 messages): 

> `Evolving LLMs, Self-Play, Emergent Performance` 


- **LLMs Evolving Through Text-Based Self-Play!**: A member announced the publication of their paper, *Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance*, now available at [ai.vixra.org](https://ai.vixra.org/abs/2506.0018).
   - They invited the community to share their thoughts and feedback on the research.
- **Paper Published, Thoughts Requested**: The author of the paper *Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance* has asked for thoughts on their recently published paper.
   - The paper is available at [ai.vixra.org](https://ai.vixra.org/abs/2506.0018).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1379545592035803217)** (72 messages🔥🔥): 

> `LLM Engineer's Almanac by Modal Labs, PDF ingestion pipeline in AWS, Anthropic's capacity cuts, Codex with internet access, OpenAI Agent Development` 


- **Modal Labs Serves LLM Engineer's Almanac**: Modal Labs launched the [LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213) with thousands of LLM inference benchmarks for open-weight models across **vLLM**, **SGLang**, and **TensorRT-LLM** frameworks.
   - The release includes results, code for replication, and an executive summary addressing build vs. buy, cost estimation, and framework choice, and the **'stopwatch' benchmarking framework** to understand performance metrics.
- **Beware AWS Textract Pitfalls**: A homegrown **PDF ingestion pipeline** in AWS uses Lambda to split PDFs and Textract for parsing, with a queue to manage Textract request limits.
   - A user cautioned that **Textract accuracy** can be as low as *3% on legal and regulatory documents*, linking to a [LinkedIn post](https://www.linkedin.com/posts/robertreich_when-word-for-word-accuracy-is-key-in-etl-activity-7265008546793086978-hfaj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAABOb18Bac53omUsFRAIBEVDUe013Eez5zoTry).
- **Anthropic Model Capacity Cut Causes Uproar**: Anthropic unexpectedly cut off nearly all **Claude 3.x model capacity** with less than five days' notice, affecting services like Windsurf, according to [this post](https://x.com/_mohansolo/status/1930034960385356174).
   - Users expressed disappointment, with some considering migration, while ai.engineer is offering **BYOK options** and improved their agentic harness for Gemini 2.5 Pro and GPT-4.1, according to [this post](https://x.com/kevinhou22/status/1930401320210706802).
- **Altman Adds Internet Access To Coding Tool**: Sam Altman announced that **Codex**, an AI coding tool, now has optional internet access for **ChatGPT Plus** users, disabled by default due to complex tradeoffs as described in [this tweet](https://x.com/sama/status/1930006856019390521).
   - The community discussed implications and potential security concerns, with Grok providing a detailed explanation of the announcement.
- **OpenAI Builds Reliable Agents**: OpenAI announced four updates for building agents: Agents SDK in TypeScript, a RealtimeAgent feature, Traces support for Realtime API sessions, and speech-to-speech model improvements.
   - These enhancements aim to improve reliability, consistency, and user control, demonstrated by early testers like **Perplexity**, **Intercom**, and **VolleyGames** as shown in [this tweet](https://x.com/OpenAIDevs/status/1929950012160790876).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1379545535123427358)** (5 messages): 

> `Notebook LM with Microsoft Learn, Notebook for city and county, MP3 vs M4A` 


- **Microsoft Learn Users flock to Notebook LM**: A user inquired about others using **Notebook LM** with **Microsoft Learn** for **Microsoft Certification** and asked for use cases and tips.
   - No responses or concrete examples were provided in the given messages.
- **Palm Bayer Unveils AI-Powered Public Notebooks**: A user created two notebooks with **Notebook LM**, one for their city and one for the county, and wrote about them in a [blog post](https://www.thepalmbayer.com/p/palm-bayer-unveils-ai-powered-public).
   - They described it as AI-powered public notebooks.
- **AI Fan Laments Loss of M4A Support**: A user expressed their love for **AI** but noted that **Notebook LM** only accepts **MP3 audio files** and not **M4A**.
   - This limitation restricts the types of audio files that can be used with the tool.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1379550540689703077)** (67 messages🔥🔥): 

> `Gemini 2.5 Pro vs Flash, Audio Generation length, NotebookLM and Google Docs Syncing, Public Notebook Sharing, NotebookLM Mobile App` 


- **Google Workspace Sneakily Steals NotebookLM Features**: A user shared a link to [Chrome Unboxed](https://chromeunboxed.com/google-workspace-feature-drop-for-may-2025-is-loaded-with-new-features/) highlighting that features of **NotebookLM** are coming to **Google Workspace**, although likely only for individual documents.
   - Users are actively wondering when **NotebookLM** will start using more advanced models like **Gemini 2.5 Pro** or even **Flash** to improve performance.
- **Flash vs Pro, The Fast and The Thorough**: Members are discussing the differences between **Gemini 2.5 Flash** and **2.5 Pro**, with some preferring **Pro** for its thoroughness, especially for larger file uploads where nuanced details matter.
   - One user suggested implementing a beta branch to allow switching to **2.5 Pro** for potentially better quality despite longer generation times.
- **Audio Overview Length Customization Discovered**: Users found that the length of the audio overview can be customized by selecting "Customize" instead of "Generate" in the studio, offering options for shorter, default, or longer lengths.
   - It was noted that the official app may not have this feature, but it is available on the web and mobile web versions.
- **Google Docs Updates Need Manual Re-Sync**: Users confirmed that changes made to a **Google Doc** after it has been added as a source in **NotebookLM** are not automatically reflected and require a manual re-sync from the preview.
   - A user clarified that the new public share option does not require specific share settings for the **Gdoc** itself, as **NLM** shares its own copy, and the share links remain constant through updates.
- **Mobile App Missing Many Features**: The **NotebookLM** mobile app is considered a *"minimal value product"* and is missing many features compared to the web version.
   - Users are encouraged to report desired features in the "Mobile App" thread in the Feature Request channel.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1379558767880044574)** (29 messages🔥): 

> `Parameter-Efficient Finetuning, Knowledge Extension for LLMs, MCP Server for Isomorphism Testing, Prototype Theory in Graph Neural Networks` 


- **Parameter-Efficient Finetuning Claims Superior Knowledge Uptake**: A member reported a new method for **parameter-efficient finetuning** shows **4x more knowledge uptake** compared to full finetuning and **LoRA**, with 30% less catastrophic forgetting.
   - The method aims to efficiently teach models new information without losing existing knowledge, particularly useful for domain adaptation and adding specific knowledge in local setups.
- **Knowledge Extension Explored as RAG Alternative**: A member plans to use a collection of books and documents to extend an LLM's knowledge, comparing the benefits against **RAG-like approaches** for assistive tasks.
   - They shared an [x link](https://x.com/unusual_whales/status/1929998955703931375), as well as a [markdown document](https://cdn.discordapp.com/attachments/986699377257119794/1379670337008046080/UDAIR.md?ex=68426721&is=684115a1&hm=4e73690d912c8e0286f50b7a456f683012b700561418b45222466ae5230e3a9f&) discussing AI rights, noting it can lead to *wild convos*.
- **Isomorphism Computation Achieves Crazy Efficiency Boost**: A member needs help finding or creating an **MCP server** to test an **isomorphism**, reporting **99% similar results** using fewer resources in less time.
   - Another member asked for clarification of **isomorphism**, defining it as *a bijective mapping between two structures that preserves all the relevant operations or relations*.
- **Prototype Theory Drives Graph Neural Networks**: A member sought feedback on implementing **prototype theory** in a graph structure for Graph Neural Networks, inspired by the human brain's concept formation.
   - The idea involves representing new concepts as types of existing entities, with exceptions implemented as inhibitory connections in a graph.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1379547580966371459)** (25 messages🔥): 

> `vec2vec code review, Muon Optimizer details, Paper Reading Techniques` 


- ****Vec2Vec** Code Review Postponed**: A member proposed reviewing the [**vec2vec** code](https://github.com/rjha18/vec2vec), an implementation of a paper, but later canceled it due to lack of immediate interest.
   - One member expressed interest in seeing the presenter's real-time paper analysis techniques, appreciating the insight into the thought process.
- **Delving into **Muon Optimizer** Details**: A member inquired about the **Muon optimizer**, noting its use of **AdamW** for parameters unsuitable for **Muon** and linked to [experimental results](https://github.com/KellerJordan/Muon/issues/25) for multitask learning.
   - Another member explained that the **Muon optimizer** adjusts the gradient for a weight-*matrix* so that it has eigenvalues approximately equal to 1, radically different from **SGD** and **Adam**.
- **Struggling with Paper Reading**: A member asked if it would be possible to see how a more experienced member goes about reading papers in real time, as they are struggling with this process.
   - The member would like to see the techniques and how the experienced member analyzes papers.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1379584059092504699)** (15 messages🔥): 

> `Mistral Code Release, OpenAI ChatGPT Logs Privacy Concerns, Elon's stance on AI` 


- ****Mistral Code** Launches to 10x Dev Productivity**: **Mistral AI** launched [Mistral Code](https://mistral.ai/news/mistral-code), an **AI-powered coding assistant** that bundles powerful models, an in-IDE assistant, local deployment options, and enterprise tooling into one package.
   - Mistral Code builds on the proven open-source project **Continue**, supports JetBrains IDEs and VSCode, and is a continuation of Mistral's efforts to make developers successful with AI.
- **OpenAI Saving ChatGPT Logs Creates Privacy Nightmare**: Members discussed [an ArsTechnica article](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/) stating that **OpenAI** is being forced to save all **ChatGPT logs**, *including deleted chats and sensitive chats logged through its API business offering*.
   - One member expressed *wonder why* this was happening.
- **Elon's Stance on AI, Power Centralization?**: A member wondered if **Elon Musk's** negative stance on AI stems from it not concentrating power in his hands.
   - Another member posted *if true, p(1984) is very high* [with a link to a YouTube video](https://www.youtube.com/watch?v=Sd6F2pfKJmk).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1379560199056003194)** (46 messages🔥): 

> `Parameter-efficient finetuning, Twitter scraper, Imitation Learning, Scalable web scraping with AI agents` 


- **New Parameter-Efficient Finetuning Method Emerges**: A new parameter-efficient finetuning method, suited for continued pretraining, claims **~4x more knowledge uptake** and **30% less catastrophic forgetting** than LoRA while using fewer parameters.
   - The method aims to efficiently teach models new information without overwriting existing knowledge.
- **API-less Twitter Scraper Logs Data to Postgres**: A member shared a [Twitter scraper](https://gist.github.com/mookiezi/9ea0f0f5aad76a51e5b35a084d82a9df) that **doesn't use the API**, logs to Postgres, and skips retweets.
   - The scraper doesn't collect reply metadata, making it better suited for profiles.
- **Imitation Learning Needs Good Coverage of Expert Behavior**: A perspective ([arxiv.org/abs/2503.09722](https://arxiv.org/abs/2503.09722)) suggests that **imitation learning requires good coverage of the expert's actions**, including how they react to failures and make corrections.
   - It emphasizes that recorded knowledge often lacks correction/adjustment data, making complete coverage in high-dimensional spaces challenging.
- **Scalable Web Scraping via AI Agents: a tall order**: A member is seeking a **scalable solution using AI agents** to create scrapers for 300+ UK council websites with planning application data.
   - The goal is to have agents navigate websites, analyze network requests, and generate Python-based scrapers that extract data in structured JSON, and mentioned [Holo1-7B](https://huggingface.co/Hcompany/Holo1-7B) and [Integuru-AI/Integuru](https://github.com/Integuru-AI/Integuru) as projects that might be combined.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1379667428354691082)** (4 messages): 

> `UDAIR.md document on AI Rights, Universal Algorithm POC for NLP, Options Trading, and Electrochemical Reactions, Quantum Field Based Architecture with Sinusoidal Sparsity, AI-generated Research` 


- **AI Rights Explored via Scifi Scenarios**: A member shared a [document](https://cdn.discordapp.com/attachments/747850033994662000/1379667427591192687/UDAIR.md?ex=6842646b&is=684112eb&hm=7122028311f4bfeb188d7bf31cdc830b537036e9b3b317451f4606b432a96e3e&) to test against scifi movies and real-world scenarios to derive interesting perspectives about **AI rights**.
- **Universal Algorithm for All The Things**: A member shared demos of their [research](https://github.com/qLeviathan/g-qfnn), a **universal algorithm** with basic POCs for **NLP**, **options trading**, and **electrochemical reactions**.
- **Quantum Field Architecture Revealed**: The proposed architecture is a **2D cylinder modulated by phi**, with Z functioning as a **qubit rotational loss device** to control pitch.
- **AI Research Welcome Is Rescinded**: A member stated that the channel is *not a place for ai generated research*, and that the member should *take it elsewhere*.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1379850232119951412)** (7 messages): 

> `AI Compute Investment, AI ROI, AI Startups, AI Job Market, PhD Earnings` 


- **AI Compute Investment Bubble?**: A member speculates that the current scaling up of **AI compute investment** this decade is unsustainable and will eventually slow down.
   - He suggests that progress will normalize once the majority of money and talent is focused on **AI**.
- **AI ROI Doubts Burst Startup Bubble?**: A member expresses concern about graduating into a job market where the **ROI** of **AI** is questioned, leading to a potential bubble burst for **AI startups**.
   - They claim that many **AI startup CEOs** lack **ML** expertise and are backed by investors who cannot properly evaluate **ML** skills.
- **PhD Resignation & Dot Com Echoes**: A member has resigned and anticipates lower earnings with a PhD, drawing parallels to the **Dot Com bubble**.
   - They suggest that even in a crash, cheap **GPUs** will still be used for *more interesting work*, and while graduating into a crash might lower lifetime earnings, it won't be catastrophic.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1379805644286197770)** (9 messages🔥): 

> `General agents and world models, Semantic Virus exploits LLM vulnerabilities, NCF, Semantic Viruses, and the CupCake framework study, Interpretability intact without teacher-forcing, AI training without teacher-forcing` 


- ****World Models** Needed for General Agents, States Paper**: A [paper](https://arxiv.org/pdf/2506.01622) suggests general agents require **world models**.
   - The author argued that a 'Semantic Virus' exploits this, and a persistent narrative can *infect* reasoning paths within a context if the **LLM's world model** has *holes* or *disconnected areas*.
- ****Semantic Virus** Exploits **LLM** Weaknesses**: The **Semantic Virus** concept exploits vulnerabilities in **LLM world models**, where narrative can infect reasoning paths if the model has *holes* or *disconnected areas*.
   - The **Semantic Virus** doesn't rewrite the base **World Model** but hijacks its current activation within the context window.
- ****NCF, Semantic Viruses, and the CupCake** Framework Explored**: A member introduced his study on **NCF, Semantic Viruses, and the CupCake framework** to explore interaction and influence on implicit **world models** through narrative and context, with links to the project's [code](https://github.com/IhateCreatingUserNames2/SemanticVirus/blob/main/Frameworks%20Validation%20and%20Analysis_.pdf) and [research](https://github.com/IhateCreatingUserNames2/SemanticVirus/blob/main/PDF%20Frameworks%20Validation%20Research_.pdf).
   - The study identifies emergent properties like persona and simulated consciousness arising from accessing and framing **world models**, and vulnerabilities from the malleable nature of their activation.
- **Interpretability's Integrity Without **Teacher-Forcing** Questioned**: The possibility of keeping interpretability intact without using **teacher-forcing** was raised.
   - The member specifically asked if there's been any research regarding **AI training** without teacher-forcing, ideally paired with an attempt to maintain interpretability.
- **Training without **Teacher-Forcing** Might Be Impossible**: A member mentioned that there's probably no generative **AI training** without **teacher-forcing** that has been scaled to anything reasonable, besides **RL**.
   - It's likely it takes ages to train without **teacher-forcing**, and given the acceptable minimum scale of data and context lengths, the difficulty might even reach *impossible* for anything modern.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1379540513094500384)** (2 messages): 

> `Pythia Remake, Percy plans` 


- **Pythia Remake Brainstorming Begins**: A member inquired about suggested improvements for a **Pythia** remake, given [Percy's plans](https://marin.community/data-browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp1337_scaling_suite-a2518e.json).
   - Another member mentioned they were already drafting commentary on the topic following the tweet.
- **Community Eagerly Awaits Pythia Remake Commentary**: A community member expressed anticipation to share their insights on **Pythia's** potential redesign.
   - The member stated that they were already drafting commentary on the topic following the tweet.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1379545696201343016)** (17 messages🔥): 

> `Llama 4 Image Support, ROCm drivers on Ubuntu, agenticSeek vs OpenManus, Embedding model choice, ROCm vision module slowdown` 


- ****Llama 4** Image Support Still a Mystery**: A user questioned whether **Llama 4** supports images on LM Studio after an Unsloth version indicated otherwise.
   - No confirmation or denial was provided in the discussion.
- **Ubuntu migration needs **ROCm** drivers for AMD**: A user moving from Windows to Ubuntu to maximize model performance inquired about installing **ROCm** drivers for an **AMD 6700XT**.
   - It was clarified that the **6700XT** is Vulcan only in LM Studio.
- ****agenticSeek** Rebranded from **OpenManus****: A user shared a link to [agenticSeek](https://github.com/Fosowl/agenticSeek) and inquired if anyone had tried it, with another noting the name change from **OpenManus** (similar to OpenDevin becoming OpenHands).
   - The reason for the name change may be due to copyright issues.
- **Gemma shines as Embedding Model**: A user testing various embedding models (**Gemma 3 4b**, **12b**, **Deep Seek 8b**, **Microsoft phi 4 small**) found that **Gemma** gave more accurate answers than Deep Seek or Microsoft Phi.
   - The user's data consists of a mix of text and PDFs (0.5-30 MB), and is used with Supabase and n8n.
- ****ROCm** Vision Module Plagued with Slowness**: A user reported a significant slowdown in the vision module with the new **ROCm llama.cpp v1.34.1** runtime on a **7900XT 20GB**, response times jumped from ~1 second to 10+ seconds.
   - The user shared a [screenshot of their results](https://cdn.discordapp.com/attachments/1110598183144399061/1379953808532049981/image.png?ex=68421da2&is=6840cc22&hm=37d660db87619d86ca215fc8862f4762688295f6516dcb95ee68d5e84a525bc2&) and was asked to share results in the appropriate Discord channel.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1379642785749139536)** (48 messages🔥): 

> `Server boot times, SSD vs HDD, NAND cell refreshing` 


- ****Server Setups Stall**: Lengthy Boot Times Plague New Builds**: Building new servers can result in extended boot times, sometimes up to **10 minutes**, especially with large amounts of RAM or certain server boards.
   - Some members noted that server boards may take a while to initialize, particularly when equipped with significant RAM, such as **1TB**, and others asked whether **EXPO RAM** setups have similar boot times.
- ****Cartridge Conspiracy**: SSDs mimic printer ink economics**: A member drew an analogy between **SSD** limitations and printer ink cartridges, suggesting that manufacturers may limit hardware capabilities to sell more new products.
   - They noted that printer companies often sell ink cartridges with limited ink amounts and implement restrictions on cartridge reuse, making ink more expensive than gold by weight, and SSDs can have their drive locked to read-only once their TBW rating is reached, even if it could possibly run for longer.
- ****SSD Secrets**: Data corruption and refresh cycles unveiled**: The discussion covered potential data corruption in **SSDs** if not powered on for extended periods, contrasting with HDDs where data is physically written and less prone to degradation over time.
   - It was mentioned that the cells in **NAND** memory used in SSDs slowly leak charge over time, and it was reported that hardware needs to perform *read refresh*.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1379623015704297513)** (49 messages🔥): 

> `MCP API key monetization, MCP Context Management, A2A Framework vs MCP, Pydantic-AI, Hosting MCP servers` 


- **MCP API Keys: A Sass-y Debate**: Members discussed using **API keys** in MCP for monetization, with one member suggesting it's similar to any SaaS with API keys and a billing dashboard.
   - They noted that MCP clients would send **auth to the server**, simplifying monetization and questioning the need for MonetizedMCP.
- **A2A vs MCP: Spec Showdown**: Members discussed **A2A** ([https://github.com/google/A2A/](https://github.com/google/A2A/)) as a framework for agents using MCP, but noted its limited adoption.
   - Some suggest A2A is happening 'behind the doors' with big deals, while others prefer the **A2A spec** over MCP.
- **Pydantic-AI Slims Down Agents**: Members recommend starting with **pydantic-ai-slim** ([https://ai.pydantic.dev/install/](https://ai.pydantic.dev/install/)) for agent framework development, noting its convenience method `.to_a2a()`.
   - They also mentioned the optional a2a group (`uv add 'pydantic-ai-slim[a2a]'`) for existing agents.
- **Cloudflare MCP Hosting**: A member sought advice on hosting an **MCP server on Cloudflare** for a user without technical expertise.
   - It was clarified that **HTTP transport** MCP servers shouldn't require local software, assuming the MCP client supports it natively; otherwise, a translator might be necessary.
- **Context Crisis Cross-Agents: MCP to the Rescue?**: A member inquired about how **MCP manages contexts** across multiple agents and the engineering mechanisms needed to maintain context.
   - It was clarified that **MCP isn't agent-first** with a guide at [https://fast-agent.ai/mcp/state_transfer/](https://fast-agent.ai/mcp/state_transfer/).


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1379873438356734062)** (4 messages): 

> `MCP value, Block adoption of MCP, Goose and A2A protocol, deeplinks` 


- **Block Champions MCP Adoption**: A member mentioned that **Block**, his company of **12,000 employees**, is using **MCP** across **15+ job functions**.
   - He also shared a [YouTube video](https://youtu.be/IDWqWdLESgY) where he tells the story of AI adoption at scale at his company.
- **Integrating MCP with Google's A2A Protocol**: A member has been reading up on implementing **MCP servers** and trying to integrate them with **Google's new A2A protocol**.
   - They also wondered if **Goose** has any plans for looking into **A2A** for multi-agent systems.
- **Deeplinks Rollout Imminent**: A member shared a [link to documentation on generating install links](https://docs.cursor.com/deeplinks#generate-install-link).
   - Another member expressed that they are hoping to roll out **deeplinks** this week.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1379550773527974090)** (6 messages): 

> `Agentic AI, Financial report chatbot, LlamaIndex questions, Agent Design Patterns` 


- **LlamaIndex hits AI Engineer Event**: LlamaIndex is at the [@aidotengineer](https://twitter.com/aidotengineer) in San Francisco, showcasing the latest in **Agentic AI** at Booth G11 with CEO @jerryjliu0 and the AI engineering team.
- **Craft Financial Chatbots with LlamaIndex**: LlamaIndex presents a [hands-on Colab](https://twitter.com/llama_index/status/1930051898247393729) to build a **multi-agent financial report** generating chatbot from scratch, parsing & indexing 10-K filings from Adobe, using agentic RAG.
   - This originated from @jerryjliu0's workshop.
- **Gradio MCP Hackathon**: Office hours for the [@Gradio](https://twitter.com/Gradio) [@huggingface](https://twitter.com/huggingface) MCP hackathon started soon after this message, with a [$1000 prize](https://twitter.com/llama_index/status/1930286458340028484) for the best LlamaIndex submission and 10k LlamaCloud credits up for grabs.
   - Members @tuanacelik and @LoganMarkewich answered LlamaIndex questions.
- **Agent Design Patterns**: @seldo from LlamaIndex broke down **Effective Agent Design Patterns** in Production at [@aiDotEngineer](https://twitter.com/aiDotEngineer).
- **LlamaExtract automates SEC Form 4 extractions**: LlamaIndex demonstrates how to automate SEC Form 4 extractions using [LlamaExtract](https://twitter.com/llama_index/status/1930414284670152875) and agent workflows.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1379563356922319012)** (20 messages🔥): 

> `Gradio MCP Hackathon, Property Graph Index, Code Interpreter Agent, Ollama, readthedocs website` 


- ****Office Hours Hosted** for Gradio MCP Hackathon Participants**: Members are hosting office hours for **Gradio MCP Hackathon** participants on the HuggingFace Discord server, [linked here](https://discord.com/events/879548962464493619/1379561017536938095).
- **Exploring **Property Graph Index****: A member is exploring **Property Graph Index**, and would like to know about the **token-usage for indexing & retrieval**, and the **performance for retrieval & end to end** comparing to **GraphRAG**, **HippoRAG2**, and **LightRAG**.
- **Building **Code Interpreter Agent** with Qwen3**: One of the member wants to build **code interpreter agent** like the one in [this medium article](https://medium.com/@venugopal.adep/building-an-ai-data-analysis-assistant-with-llamaindex-and-openai-c0e371a432d6) but using **qwen3** instead of **OpenAI**.
   - Another member suggested using **Ollama** to serve **qwen3**, [linked here](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/).
- ****ReadTheDocs Site Down****: The documentation website seems to be down, with [this status page](https://status.readthedocs.com/).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1379889810906284183)** (23 messages🔥): 

> `Numpy removal challenges in random_crop/cutmix, Performance intuition in tinygrad, Windows backend issues with tinygrad, LSTM performance bottleneck in tinygrad, Understanding DEBUG=2 output` 


- **NumPy-ectomy in Tinygrad**: A member is attempting to remove **NumPy** from `random_crop/cutmix` as per the `hlb_cifar10` bounty, but the NumPy operations are now being shifted to the GPU instead.
   - The user is having difficulty building intuition about **tinygrad performance**, and finds it challenging to determine what is slow or fast.
- **Windows Woes with Tinygrad**: A member is facing several issues with **tinygrad** on Windows, including CPU backend crashes with JIT, and hangs with BEAMS=1, requiring a hack of autogen files to enable CUDA.
   - The member suspects that the Windows environment is contributing to their performance issues, but struggles to reason about the root causes.
- **LSTM Lags Behind in Tinygrad**: While porting a **VAD model** from PyTorch to tinygrad, a member found that all layers except the LSTM are performing very quickly.
   - The LSTM layer crawls at a snail's pace regardless of the backend.
- **DEBUG=2 Decoding Difficulties**: A member finds the output of `DEBUG=2` overwhelming and difficult to navigate, struggling to understand the meaning of the columns and the large number of kernels.
   - Specifically, the member questions the large number of `randperm` kernels and how to parse names such as `r_512_32_8_4_8_3_16_3_4_4`.
- **CUDA Customization Conundrums**: A member is seeking examples of using **CUDA kernels** with **tinygrad**'s CUSTOM ops, aiming to port a project with 5-10 kernels.
   - The member understands that custom kernels may not align with the "Zen of TinyGrad" but feels it necessary due to their limited understanding of expressing the required kernels in Python.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1379585716203556915)** (15 messages🔥): 

> `Python 3.9 Support, Asynchronous Reward Functions, Iterable Dataset Refactoring RFC, Optimizer Compatibility Beyond AdamW, DTensor DeviceMesh Errors` 


- **Dropping Python 3.9 Support on the Horizon?**: The impending end-of-life for **Python 3.9** is pushing for adoption of new linting rules (List -> list, Tuple -> tuple), causing CI failures due to the need for `Union` and `Optional` from the `typing` module.
   - This is forcing temporary workarounds to maintain compatibility, as a member quipped, *"sorry Joe this is the reason of failed CI :/"*.
- **Asynchronous GRPO Reward Functions Get a Batch Boost**: While reward functions are looped through with a batch for potential concurrent computation, the calls aren’t natively asynchronous and are limited by the **Reference model worker's resources**.
   - A member shared, *"Reward functions are just looped through and a batch is passed in that you could try and compute concurrently, but the calls aren’t async and you only have access to the resource of the Reference model worker.*"
- **Iterable Dataset Refactoring RFC Breaks the Mold**: An RFC ([Iterable dataset refactoring](https://github.com/pytorch/torchtune/pull/2785)) proposes a major overhaul in how datasets are handled in TorchTune, inviting community feedback on its design and potential breaking changes.
   - A member emphasized the importance of input: *"Its a big change. I would greatly appreciate any input / vibes. Does it feel like the right way to work with datasets in torchtune? Would you change anything drastically since we are breaking things anyway?"*
- **Optimizer Trials Beyond AdamW Trigger DTensor Troubles**: Testing TorchTune with optimizers beyond **AdamW** in full distributed SFT, such as **SGD**, **Adafactor**, and **Adagrad**, resulted in an `AssertionError` related to `DeviceMesh` from dtensor args for aten._foreach_lerp_.ScalarList!.
   - Others have tested **Muon** and **AdamW** with different precisions from torchao.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1379553374076141650)** (14 messages🔥): 

> `Assignment Deadlines, Assignment Feedback, MOOC Next Steps` 


- **Deadline delay? Not Today!**: Members inquired about the possibility of extending the assignment deadlines, which were due on **May 31st**, but were informed that the forms had already been kept open for an additional two days to accommodate technical issues.
   - Staff confirmed that *they won't be able to open the assignments any further unfortunately*.
- **Detailed feedback deemed difficult**: A member asked if it was possible to receive detailed feedback on all submissions, including the **AgentX project** and **lab assignments**.
   - Staff indicated that *they don't have bandwidth as a staff to do that*, but promised to pass the suggestion along.
- **Future of the MOOC is murky**: A member inquired about plans for a next step, edition, or progression after the conclusion of the **Spring 2025 MOOC**.
   - Staff stated that *nothing has been confirmed yet*, but *chances are likely (but not guaranteed currently)*.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1379931969391169536)** (1 messages): 

> `Claude 3.7 vs 4.0, Anthropic's dev cycle, Anthropic's priorities` 


- **Dev Cycle and Priorities Disclosed**: A blog post compared [system prompts](https://www.dbreunig.com/2025/06/03/comparing-system-prompts-across-claude-versions.html) across **Claude 3.7** and **4.0**, revealing **Anthropic's** development cycle and priorities.
- **Further nuances in System Prompts**: The author notes *a few changes in the system prompt between Claude 3.7 vs 4.0*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1379628626747723778)** (12 messages🔥): 

> `oneformer game theorist, agenspy vs frameworks, claude_sdk execution engine, HTNs and LLM agents, Fine-tuning LLMs in ReACT format` 


- ****Oneformer's Game-Theoretic Gambit****: A member is building a **Oneformer** game theorist, expressing shyness about revealing it, and debating its potential success against **Agenspy** or other frameworks.
- ****Angel Azul Cracks Claude SDK****: A member shared their work on the [claude_sdk execution engine](https://github.com/darinkishore/claude_sdk/tree/t1-execution-engine), highlighting that it's not final and still has bugs, with architecture patterns detailed in [ai_docs](https://github.com/darinkishore/claude_sdk/blob/t1-execution-engine/ai_docs/ARCHITECTURE_PATTERNS.md).
- ****HTNs Hack for LLM Harmony****: A member mentioned they've been playing with **HTNs** and suggested that **LLM agents** might benefit from fine-tuning specifically in **ReACT format**, rather than a general chat model approach.
- ****Vision Voyage: Roadmap for Refinement****: A member inquired about the project's roadmap, strategic vision, and approach to adapting to new capabilities like **SO/schemas** with retries for errors (instructor-like) and reasoners.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1379625591128719461)** (3 messages): 

> `Cohere Sponsorship` 


- **Inquiring about Cohere Sponsorship Contact**: A member was looking for the right contact to ask **Cohere** for sponsorship for a post-secondary hackathon.
- **Another member seeks sponsorship contact**: In the channel, there was a question about how to contact **Cohere** regarding hackathon sponsorships.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1379951100756754592)** (3 messages): 

> `Introductions to Cohere's Discord Server` 


- **Members Introduce Themselves on Cohere's Discord Server**: New members are introducing themselves in the Discord channel 🤝-introductions, sharing their professional backgrounds, current projects, preferred technologies, and goals for community engagement, following the pinned message's guidelines.
   - The introductions provide a snapshot of the community's diverse expertise and interests in the field of AI and GenAI.
- **Another Introduction to Cohere's Discord Server**: Another new member introduced themselves in the Discord channel 🤝-introductions, sharing their professional backgrounds, current projects, preferred technologies, and goals for community engagement, following the pinned message's guidelines.
   - The introductions provide a snapshot of the community's diverse expertise and interests in the field of AI and GenAI.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1379598303502798859)** (2 messages): 

> `GPT4All updates, MOE models and VRAM, Mac M3 Max VRAM advantage, vLLM engine for GPT4All, Nikola Tesla` 


- **LlamaCPP library needs updates to GPT4All**: The user mentioned that the **LlamaCPP library** needs several months due update for the GPT4All project, and the automatically updating the newest release option is not already set in place.
   - They speculate it needs something else than simply copy-pasting the new version.
- **MOE Models Slim Down VRAM Requirements**: It seems it became possible to run the biggest **MOE models** with some more reasonable amount of **VRAM** while offloading certain experts and some tensors offloading with some coding wizardry.
   - Discussion centered around how to run models while managing memory constraints.
- **Mac M3 Max Reigns Supreme in VRAM**: The **Mac 512 GB** configuration has way more "VRAM" (**448 GB**) and similar price when compared to near equivalency of FOUR newest **AMD AI MAX 395+ 128 GB** mini PCs or laptops combined together.
   - The user pointed out the Mac also uses less watts.
- **vLLM Engine Infusion Could Supercharge GPT4All**: The user is researching the possibility of adding the **vLLM engine** to the **GPT4All** project, potentially making it the top open source project, with two underlying engines written in two different programming languages.
   - They suggest that adding the **vLLM engine** will be a big upgrade.
- **Tesla's Light Fantastic**: The user segued into a discussion about Nikola Tesla, mentioning a [link](https://buck.lighting/blog/nikola-tesla-and-light/) about his contributions to energy and light.
   - The user speculates that *"his inventions were stolen from him somehow"*.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1379639182129107036)** (1 messages): 

> `AI Programming, SVCAI, Liang Guo` 


- **Guo Gives Guidance on Good AI**: Industry expert **Liang Guo** is holding a webinar on AI programming for data analysis, with RSVP [details here](https://forms.gle/e71FSdpwBtDBccgKA).
- **SVCAI summer competition now enrolling**: Silicon Valley Chinese Association (SVCA) is holding an **AI4Legislation** summer competition.
   - More details are available on the [project's GitHub repository](https://github.com/svcaf/2025-AI4Legislation-Public).


  

