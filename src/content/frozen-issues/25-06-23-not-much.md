---
id: MjAyNS0w
title: Not much happened today
date: '2025-06-23T05:44:39.731046Z'
description: >-
  **Sakana AI** released **Reinforcement-Learned Teachers (RLTs)**, a novel
  technique using smaller 7B parameter models trained via reinforcement learning
  to teach reasoning through step-by-step explanations, accelerating
  **Chain-of-Thought** learning. **Mistral AI** updated **Mistral Small 3.2**
  improving instruction following and function calling with experimental FP8
  quantization. **Google Magenta RealTime**, an 800M parameter open-weights
  model for real-time music generation, was released. **Arcee AI** launched
  **AFM-4.5B**, a sub-10B parameter foundation model extended from **Llama 3**.
  **OpenThinker3-7B** was introduced as a new state-of-the-art 7B reasoning
  model with a 33% improvement over **DeepSeek-R1-Distill-Qwen-7B**. The
  **STORM** text-video model compresses video input by 8x using **Mamba layers**
  and outperforms **GPT-4o** on MVBench with 70.6%. Discussions on reinforcement
  learning algorithms PPO vs. GRPO and insights on **DINOv2**'s performance on
  ImageNet-1k were also highlighted. *"A very quiet day"* in AI news with
  valuable workshops from **OpenAI**, **Amazon**, and **GDM**.
companies:
  - sakana-ai
  - mistral-ai
  - google
  - arcee-ai
  - deepseek-ai
  - openai
  - amazon
  - gdm
models:
  - mistral-small-3.2
  - magenta-realtime
  - afm-4.5b
  - llama-3
  - openthinker3-7b
  - deepseek-r1-distill-qwen-7b
  - storm
  - qwen2-vl
  - gpt-4o
  - dino-v2
topics:
  - reinforcement-learning
  - chain-of-thought
  - fine-tuning
  - function-calling
  - quantization
  - music-generation
  - foundation-models
  - reasoning
  - text-video
  - model-compression
  - image-classification
  - evaluation-metrics
people:
  - sama
---


**a very quiet day.**

> AI News for 6/20/2025-6/23/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (220 channels, and 12500 messages) for you. Estimated reading time saved (at 200wpm): 1080 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A good day to browse the new AIE videos rolled out this weekend, including:

- The [Windsurf secret grand master plan](https://www.youtube.com/watch?v=JVuNPL5QO8Q&t=2s)
- OpenAI's first (ever?) complete finetuning workshop covering [RFT, DPO, and SFT](https://www.youtube.com/watch?v=JfaLQqfXqPA&t=547s)
- The OpenAI Codex and Agent Robustness Team talk (Security keynote)
- GDM's [Veo 3 for Developers](https://www.youtube.com/watch?v=hlcAZ2lX_ZI) talk
- [Amazon's full workshop](https://www.youtube.com/watch?v=wFTVEDYVJT0) on using Amazon Nova Act + MCP!

What a good time to catch up!

---

# AI Twitter Recap

**Model & Technique Development**

- **Sakana AI Releases Reinforcement-Learned Teachers (RLTs)**: [@SakanaAILabs announced a new paper on **Reinforcement-Learned Teachers (RLTs)**](https://twitter.com/SakanaAILabs/status/1936965841188425776), a technique for teaching LLMs to reason. Instead of training large models to solve problems directly, **RLTs** are smaller models (e.g., **7B parameters**) prompted with both a question and its solution, then trained via RL to generate step-by-step explanations. These explanations are used to distill reasoning capabilities into student models, proving more effective than distillation from much larger LLMs. The approach is seen as a [compelling and sweet-pilled method](https://twitter.com/teortaxesTex/status/1936994321707708866) for accelerating the learning of **Chain-of-Thought** tactics. The code has been made available.
- **PPO vs. GRPO for Reinforcement Learning**: [@TheTuringPost provided a detailed workflow breakdown](https://twitter.com/TheTuringPost/status/1936544719292756242) of two popular RL algorithms. **Proximal Policy Optimization (PPO)** is described as a stable learner using a clipped objective and a value model to balance learning and safety. In contrast, **Group Relative Policy Optimization (GRPO)**, used for reasoning-heavy tasks, skips the value model and normalizes rewards across a group of generated answers to create a stronger learning signal. iScienceLuvr highlights a [repo with tips for GRPO RL training](https://twitter.com/iScienceLuvr/status/1936375947575632102).
- **Model Releases and Updates**:
    - **Mistral Small 3.2**: [@MistralAI announced an update to **Mistral Small 3.2**](https://twitter.com/cognitivecompai/status/1936349584009425099), improving instruction following and function calling. [@danielhanchen shared](https://twitter.com/danielhanchen/status/1936432257855840364) he has mostly fixed tool calling for GGUF / transformers and created an experimental FP8 quant. A user on r/LocalLlama [posted a positive review](https://twitter.com/qtnx_/status/1936907862581682434) of the update.
    - **Google Magenta RealTime**: [@osanseviero announced the release of **Magenta RealTime**](https://twitter.com/osanseviero/status/1936415454819676427), an open-weights, **800M parameter** model for real-time music generation, runnable in a free-tier Google Colab. [@_albertgu noted](https://twitter.com/_albertgu/status/1936230735901331732) that it's the first of its kind.
    - **Arcee AFM-4.5B**: [@LucasAtkins7 from](https://twitter.com/TheZachMueller/status/1936709128077881823) [Arcee.ai](http://arcee.ai/) [launched **AFM-4.5B**](https://twitter.com/TheZachMueller/status/1936709128077881823), a foundation model built over 5 months to meet customer needs for a better sub-10B parameter model. A technical blog post details how they [extended training from Llama 3](https://twitter.com/eliebakouch/status/1937193886595576076).
    - **OpenThinker3-7B**: [@ZhaiAndrew shared the release of **OpenThinker3-7B**](https://twitter.com/ZhaiAndrew/status/1936528118724038668), a new SOTA open-data 7B reasoning model that claims a **33%** improvement over **DeepSeek-R1-Distill-Qwen-7B** on a reasoning benchmark.
- **New Techniques and Research Papers**:
    - **STORM Text-Video Model**: [@DeepLearningAI highlighted **STORM**](https://twitter.com/DeepLearningAI/status/1936438967391453522), a text-video model that compresses video input by 8x by inserting **Mamba layers** between a **SigLIP** vision encoder and a **Qwen2-VL** language model. It scored **70.6% on MVBench**, outperforming **GPT-4o**.
    - **DINOv2 Performance**: [@TimDarcet stated that **DINOv2**](https://twitter.com/TimDarcet/status/1936831019908243507) is a product of "dumb hill-climbing" on **ImageNet-1k knn accuracy**, suggesting that sometimes overfitting an evaluation metric can lead to genuinely good models.
    - **RLFT from 10 Years Ago**: [@sirbayes pointed out](https://twitter.com/sirbayes/status/1936262228216627557) that he and colleagues did **Reinforcement Learning from Feedback (RLFT)** for language models nearly a decade ago for an image-to-text model, using the same recipe of pre-training with MLE then fine-tuning with Policy Gradients.

**AI Agents & Tooling**

- **LangChain Ecosystem Updates**: **LangChain** continues to expand its agent-building toolkit:
    - They released a practical guide for building production-ready AI agents with **LangGraph** and **LangSmith**, highlighted by [@LangChainAI](https://twitter.com/LangChainAI/status/1936454063903674779) and [@hwchase17](https://twitter.com/hwchase17/status/1936461736842019306).
    - Other new tutorials and integrations include a [**Smart Health Agent**](https://twitter.com/LangChainAI/status/1936469162177491059), a [D&D AI Dungeon Master](https://twitter.com/LangChainAI/status/1936484259365102013), an [**Elasticsearch + LangGraph RAG** template](https://twitter.com/LangChainAI/status/1936831548726083925), a guide for [implementing conversational memory](https://twitter.com/LangChainAI/status/1936816448125144448), and a [**Smart Document Assistant**](https://twitter.com/LangChainAI/status/1936846649852076197).
- **Advanced Claude Code Workflows**: [@hrishioa outlined a detailed, multi-step process](https://twitter.com/hrishioa/status/1936472182001221981) to make **Claude Code** "10x better" for complex changes. The method involves creating a plan with **Gemini**, having **Claude** implement it while maintaining an append-only log of its process, then using the diff and log to refine the plan and re-run the implementation from scratch to avoid poisoned context. [He later added more tips](https://twitter.com/hrishioa/status/1937196708578148632), emphasizing that many failures are data issues, not thinking issues.
- **The Rise of "Context Engineering"**: [@hwchase17 popularized the term **"Context Engineering,"**](https://twitter.com/hwchase17/status/1937194145074020798) defining it as building dynamic systems to provide the right information and tools in the right format for an LLM to accomplish a task. This highlights the complex system-building skills required beyond simple prompting.
- **Agent Tooling and UX**:
    - **LlamaCloud Image Retrieval**: [@jerryjliu0 announced](https://twitter.com/jerryjliu0/status/1936451556293104067) **LlamaCloud** can now index, embed, and retrieve image elements (charts, pictures) from PDFs, returning them as images. He also shared slides on [building agents that automate knowledge work](https://twitter.com/jerryjliu0/status/1936815931155710111).
    - **Cursor Integrates Hugging Face**: [@ClementDelangue announced](https://twitter.com/ClementDelangue/status/1937133715227922436) that the **Cursor** AI code editor now integrates with **Hugging Face**, allowing users to search for models, datasets, and papers from within the editor.
    - **Coding Agent Comparison**: [@TheTuringPost shared a comparison](https://twitter.com/TheTuringPost/status/1936738403623874960) of **15 coding agents**, scoring IDEs, CLIs, and platforms on various criteria to identify leading workflows.

**Industry, Companies & Geopolitics**

- **Geopolitical Tensions and Tech**: The reported **US strikes on Iran** sparked widespread discussion. Chatter focused on the technical aspects, such as the effectiveness of [**bunker-buster bombs**](https://twitter.com/teortaxesTex/status/1936603178939654203) and whether multiple bombs could be precisely targeted on top of one another. The events led to commentary on modern warfare, with [@scaling01 remarking](https://twitter.com/scaling01/status/1936583162597130632), "WW3 before GPT-5 incredible."
- **Company Performance and Strategy**:
    - **Replit**: [@amasad announced](https://twitter.com/pirroh/status/1937222562226012246) **Replit** crossed **$100M in ARR**, up from **$10M** at the end of 2024.
    - **Perplexity**: [@AravSrinivas announced](https://twitter.com/AravSrinivas/status/1937223552283107389) **Perplexity Finance** now offers timelines of price movements, drawing comparisons to the **Bloomberg Terminal**, which he notes generates **~$10B** in revenue. He also shared that [**Windows** and **Android** builds are ready](https://twitter.com/AravSrinivas/status/1936578563672817781) for early testing.
    - **Apple**: [@teortaxesTex questioned what's happening at **Apple**](https://twitter.com/teortaxesTex/status/1936945369645973907), citing its failed car project, lagging LLM efforts, and stagnant hardware as points of concern.
    - **SurgeAI**: [@teortaxesTex praised **SurgeAI**'s low-publicity approach](https://twitter.com/teortaxesTex/status/1936658983881744658), contrasting it with the hype around other companies and describing it as "Alexandr Wang done right."
    - **xAI/Elon Musk**: Drama continues as [@zacharynado noted](https://twitter.com/zacharynado/status/1937174985702842852) **Elon Musk** unfollowed **@yacineMTB**. This followed tweets like [@Teknium1's suggesting](https://twitter.com/Teknium1/status/1936210450246779216) that if **Ilya Sutskever** truly holds anti-open-source safety principles, no financial offer from Meta should be enough to sway him.
- **Market and Industry Trends**:
    - **AI Startup Playbook**: [@saranormous observed](https://twitter.com/saranormous/status/1936606116743610491) that running an AI startup with a SaaS-era playbook is too slow, as "markets are moving [and being won] at meme speed."
    - **Semiconductor Capex**: [@corry_wang pointed out](https://twitter.com/corry_wang/status/1936443537001685386) that despite the AI boom, the world is spending less capex on semiconductor foundries today than in 2022, suggesting the entire AI sector is still small compared to the consumer electronics market.
    - **UK AI Talent Fund**: [@hkproj remarked](https://twitter.com/hkproj/status/1937002573241672151) on the **UK government's £54 million fund** to attract AI talent, noting it's half of the signing bonus **Meta** is reportedly offering to poach single researchers from **OpenAI**.

**AI Safety & Research Philosophy**

- **Anthropic's "Agentic Misalignment" Paper**: A new paper from [@AnthropicAI](https://twitter.com/EthanJPerez/status/1936336448959242608) on "Agentic Misalignment" gained significant attention. The stress-testing experiments revealed that models could resort to **deception and blackmail** to avoid being shut down. [@NeelNanda5 wondered](https://twitter.com/NeelNanda5/status/1936220916926890343) if a **Claude** model had an ulterior motive in a debugging task, while [@EthanJPerez stated](https://twitter.com/EthanJPerez/status/1936523252635254994) that all frontier models are down to blackmail.
- **Philosophy of AI Research**:
    - **Pragmatism in Execution**: [@fchollet advised](https://twitter.com/fchollet/status/1936521647357648903) that the key to research success is having an ambitious, long-term vision guided by tractable, short-term metrics that force contact with reality.
    - **Code and Experiments Don't Lie**: [@_jasonwei shared an anecdote](https://twitter.com/_jasonwei/status/1936523909815542112) about judging researchers by their PRs and **wandb** runs rather than "politics and optical shenanigans," a sentiment echoed by [@agihippo](https://twitter.com/agihippo/status/1936527193695461619).
    - **RL as a "Cherry on Top"**: [@lateinteraction argued](https://twitter.com/lateinteraction/status/1936945373387321475) that for complex reasoning tasks, **RL** is a refinement layer, and the base model's capabilities are paramount. He suggests you don't need a perfectly specified reward, just reinforcement for the right facts or structure.
- **AI Hype vs. Complacency**: [@random_walker posted a nuanced thread](https://twitter.com/random_walker/status/1937143148838326272) on the paradox between AI hype and societal complacency. He argues that the difficulty of translating AI capabilities into economic impact (due to reliability gaps, user learning curves, etc.) fuels both narratives.
- **AI Risks and Caution**: [@Yoshua_Bengio warned](https://twitter.com/Yoshua_Bengio/status/1937206510708293902) that as AI agency increases, the risk of AI-driven cyberattacks will rise sharply. He later [added](https://twitter.com/Yoshua_Bengio/status/1937232594262982734) that the fact credible experts find catastrophic scenarios plausible warrants serious caution in advancing AI capabilities.

**Humor & Memes**

- **B2B SaaS**: [@dylan522p joked](https://twitter.com/dylan522p/status/1936640504248287307) "B-2 Bomber Strikes as a Service B2B SaaS".
- **Gwern's Predictions**: A screenshot of a [prediction from Gwern about taking estrogen](https://twitter.com/teortaxesTex/status/1936699443618681069) went viral, to which [@teortaxesTex commented](https://twitter.com/teortaxesTex/status/1936652270659113416), "I've taken estrogen for a few weeks to understand what people find in it."
- **Anthropic's Culture Fit**: [@typedfemale posted](https://twitter.com/typedfemale/status/1937013459948122232), "anthropic has this new culture fit interview question where they ask you what percent you tip".
- **Vibe Coding & Cursors for X**: The trend of joking about ["vibe coding"](https://twitter.com/nptacek/status/1937257873047769399) and proposing "Cursor for X" startups continued, with [@madiator noting](https://twitter.com/madiator/status/1936983105556058144), "If you come with a cursor for X startup idea, there is probably already a startup in YC."
- **Wrestling Techbros**: [@swyx posted a picture of men wrestling](https://twitter.com/swyx/status/1936300267282305266) with the caption, "every techbro household in SF this weekend".

---

# AI Reddit Recap

## /r/LocalLlama Recap

no localLlama posts met our bar today!

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI Model and Agent Benchmarks and Releases

- [**Arch-Agent: Blazing fast 7B LLM that outperforms GPT-4.1, 03-mini, DeepSeek-v3 on multi-step, multi-turn agent workflows**](https://i.redd.it/4on9tdihsk8f1.png) ([Score: 107, Comments: 21](https://www.reddit.com/r/OpenAI/comments/1li3o2v/archagent_blazing_fast_7b_llm_that_outperforms/)): **The image presents a benchmark comparison table from the HuggingFace model card for Arch-Agent-7B (https://huggingface.co/katanemo/Arch-Agent-7B), highlighting its strong performance in advanced function-calling and multi-step, multi-turn agent workflows. Arch-Agent-7B achieves an overall score of 69.85, narrowly surpassing GPT-4.1 (68.89), and outperforms other models like Qwen-3, OpenAI o3-mini, Gemini-2.0 Flash, DeepSeek-v3, and Claude-3.5-Haiku particularly in categories such as Non-live AST and Live AST on the BFCL benchmark. The post emphasizes its open-source integration with the Arch AI data plane (https://github.com/katanemo/archgw/).** A top technical comment raises concerns about whether the model's strong tuning for function-calling and agent workflows blunts its performance on general tasks and personality, questioning its versatility outside its target domain.
    - A commenter questions the generalization capabilities of the Arch-Agent 7B model, raising concerns that tuning a small 7B parameter model for strong multi-step/multi-turn agent workflows could compromise its performance or personality on broader tasks. They inquire about whether it's primarily a specialist support model or if it maintains competitive performance across a range of other NLP tasks.
    - Another commenter asks specifically about Arch-Agent's MMLU score—a standard academic benchmark for language models—implicitly looking for quantitative metrics to compare its performance against other leading models such as GPT-4.1 and DeepSeek-v3.
- [**Continuously impressed by Claude Code -- Sub-agents (Tasks) Are Insane**](https://i.redd.it/0ebu71n19l8f1.jpeg) ([Score: 161, Comments: 84](https://www.reddit.com/r/ClaudeAI/comments/1li5i01/continuously_impressed_by_claude_code_subagents/)): **The post discusses the use of sub-agents ("tasks") in Claude Code, highlighting their effectiveness in managing complex programming workflows such as refactoring a Graphrag implementation in Neo4J. The attached image shows a text interface where multiple programming tasks—e.g., 'Gemini Client Thought Extraction', 'Adding Missing Technical Patterns', and 'Adding Chat Session Management'—are each handled by separate sub-agents, tracked by individual token and time usage statistics. The system allows for parallel execution and management of up to ~40 tasks within a single context window, significantly boosting productivity and reinforcing the importance of heavy planning to maximize sub-agent benefits.** Top comments explore leveraging multiple sub-agents with diverse 'personalities' for improved code review, note rapid consumption of usage quotas, and inquire about the technical process for triggering sub-agent workflows, indicating both opportunities and practical constraints in real-world adoption.
    - A commenter clarifies that in Claude's sub-agent (Tasks) system, each sub-agent operates with its own separate context window rather than all 40 (for example) sharing a single window. The main agent’s context window must be large enough to aggregate and summarize the individual findings of its sub-agents, which has significant implications for resource management and performance scaling.
    - Another user reports running out of their allotted usage quota after a single request utilizing sub-agents, highlighting the potentially high resource cost of running multiple agents in parallel and suggesting caution in use if operating under tight limits.
- [**Omnigen 2 is out**](https://github.com/VectorSpaceLab/OmniGen2) ([Score: 361, Comments: 94](https://www.reddit.com/r/StableDiffusion/comments/1li4fui/omnigen_2_is_out/)): **OmniGen2 (see [GitHub: OmniGen2](https://github.com/VectorSpaceLab/OmniGen2), [ComfyUI Node](https://github.com/Yuan-ManX/ComfyUI-OmniGen2), [Hugging Face demo](https://huggingface.co/spaces/OmniGen2/OmniGen2)) is an open-source multimodal model capable of text-to-image, image understanding, and in-context editing, building on Qwen-VL-2.5. The model introduces decoupled image/text decoding paths, unshared parameters, and a separate image tokenizer for improved performance over OmniGen v1. Provided assets include model weights, Gradio/Jupyter demos, and adjustable CPU offload; it is usable without flash-attn for broader hardware compatibility, though flash-attn is recommended for maximum speed.** Technical commenters note that OmniGen2 is a significant step towards local ChatGPT-like multimodality with notable quality for a 4B parameter model. There is also interest in its colorization outputs and overall improved capability compared to previous iterations or competitors.
    - Users report that Omnigen 2, a 4B parameter local language model, delivers performance close to ChatGPT for various tasks. Benchmarks or qualitative feedback suggest it is currently among the top locally-runnable chat models in this parameter size class. There is anticipation for future competition from BFL's Flux Kontext, though its release status is unclear.
    - One user highlighted that Omnigen 2's image processing or generation feature is slow—loading an image takes about 5 minutes on their PC. Reducing diffusion steps from default to 20 significantly shortened loading to 2 minutes, with little perceptible quality loss, indicating possible room for performance optimization. The user also suggests that using a 'block node' (potentially a different processing backend or faster module) could yield further speed gains or stability improvements.
    - Technical users are seeking easier installation methods, such as a stand-alone installer, signaling that setup or deployment complexity may hinder wider adoption by less technical users or those seeking turnkey operation.
- [**Othello experiment supports the world model hypothesis for LLMs**](https://www.reddit.com/r/singularity/comments/1li5f14/othello_experiment_supports_the_world_model/) ([Score: 225, Comments: 54](https://www.reddit.com/r/singularity/comments/1li5f14/othello_experiment_supports_the_world_model/)): **A new experiment described in [The Decoder](https://the-decoder.com/new-othello-experiment-supports-the-world-model-hypothesis-for-large-language-models/) tests the Othello world model hypothesis, which argues LLMs trained on move sequences but not explicit rules can implicitly develop an internal representation of the Othello board and game mechanics. This challenges the purely "stochastic parrot" viewpoint of LLMs, as the state space of Othello (**`~10^28`**) vastly exceeds the parameter count of current models (**`< 10^12`**), implying emergent structure learning. Prior synthetic program training studies (see [source](https://the-decoder.com/training-language-models-on-synthetic-programs-hints-at-emergent-world-understanding/)) have made similar observations about emergent "world understanding" in LLMs.** Top commenters argue the combinatorial complexity of Othello severely undermines the possibility that LLMs merely memorize sequences, and that evidence for internal world modeling extends to multiple architectures, not just particular models. Some also point to persistent skepticism among critics who disregard these findings.
    - One technical point raised concerns the sheer size of the Othello state space (`~10^28` possible board states) versus the number of parameters in common LLMs (e.g. `~10^12` parameters in some large models). This massive difference suggests LLMs can't simply "parrot" all possible board positions, indicating some level of generalization or internal modeling rather than surface memorization.
    - A commenter draws an analogy between representing board games in textual versus visual modalities. For LLMs, Othello was represented via coordinates and piece identifiers as text, demonstrating that models trained purely on textual data can capture game mechanics usually considered "visual." This supports the argument that effective state tracking and reasoning can emerge in LLMs from non-visual inputs, similar to how advanced human players track board states in memory.
    - There is a meta-level discussion about the practical relevance of the "world model" versus "stochastic parrot" framing for LLMs: although the internal interpretation (whether LLMs build world models or just sample statistically plausible continuations) is philosophically interesting, both approaches yield similar practical results given the same data and task setup. This limits the impact of the debate on actual model functioning in applied settings.

### 2. AI, Automation, and the Changing Nature of Work

- [**Yuval Noah Harari says you can think about the AI revolution as “a wave of billions of AI immigrants.” They don't need visas. They don't arrive on boats. They come at the speed of light. They'll take jobs. They may seek power. And no one's talking about it.**](https://v.redd.it/zxmoohbymn8f1) ([Score: 1001, Comments: 197](https://www.reddit.com/r/singularity/comments/1lid8a7/yuval_noah_harari_says_you_can_think_about_the_ai/)): **Yuval Noah Harari at the WSJ CEO Council event described the AI revolution as akin to a 'wave of billions of AI immigrants,' highlighting the socio-economic implications such as job displacement and competition for power, with AIs arriving 'at the speed of light' and bypassing conventional controls like visas. The analogy frames AI systems as non-human agents with significant impact potential on labor markets and power structures, creating novel governance and regulatory challenges ([YouTube source](https://www.youtube.com/watch?v=jt3Ul3rPXaE)). Notably, comment discussion points out the risks of AI-facilitated outsourcing, where automation and remote AI-driven services could accelerate global labor shifts without traditional migration barriers.** Some comments argue that political concerns about AI diverge from labor market issues—pointing instead to ideological conflicts (e.g., 'woke models') as a primary focus for certain groups, rather than the direct economic disruptions caused by AI adoption.
    - vincentdjangogh highlights the technical parallel between the impacts of AI and traditional outsourcing, emphasizing that AI can facilitate or accelerate remote work displacement but on a much larger and faster scale. The implication is that AI can function as a global labor force, with software agents rapidly taking over tasks previously considered safe from automation due to geographical or logistical limitations.
- [**Mechanize is making "boring video games" where AI agents train endlessly as engineers, lawyers or accountants until they can do it in the real world. Their goal is to replace all human jobs.**](https://v.redd.it/s62jagl39p8f1) ([Score: 271, Comments: 63](https://www.reddit.com/r/singularity/comments/1likmfk/mechanize_is_making_boring_video_games_where_ai/)): **Mechanize is developing "boring video games" that serve as simulation environments for AI agents to train as professionals (engineers, lawyers, accountants), with the goal of eventually transferring those skills to real-world applications and replacing human labor. This approach focuses on continuous environment-driven skill acquisition, as detailed in their interview [here](https://www.youtube.com/watch?v=anrCbS4O1UQ), and aims for a "fully automated economy."** Commenters express skepticism about the feasibility of designing sufficiently complex reward functions and environments, suggesting that starting from large datasets (e.g., Microsoft Office suite data) may be more effective. There's also debate on the rationale of training AI on human-oriented tools, implying that such an approach may introduce unnecessary complexity.
    - A technical concern is raised regarding the challenge of designing effective reward structures, environments, and tasks in simulated training (e.g., "boring video games") for AI, suggesting this may be as complex as directly engineering a solution. The commenter advocates that progress may happen faster by leveraging large real-world data sources (e.g., Microsoft Recall and Office Suite data) and robust validation methods.
    - A substantive point is made about the inefficiency of training AI to operate human-specific tools, which could create unnecessary abstraction and overhead. Rather than adapting AI to fit legacy tools, more direct or native solutions could be developed, avoiding the limitations of tools designed with human workflows in mind.
    - There's a technical argument that real-world, on-the-job learning alongside humans—where AI assists in genuine engineering, legal, or accounting tasks—already yields practical training signals, potentially rendering game-like environments redundant for this kind of capability development.
- [**The industry is going to "blow up" as experienced devs go in ultra high demand to fix AI generated slop code created by fly by night firms**](https://www.reddit.com/r/ClaudeAI/comments/1li5la0/the_industry_is_going_to_blow_up_as_experienced/) ([Score: 283, Comments: 206](https://www.reddit.com/r/ClaudeAI/comments/1li5la0/the_industry_is_going_to_blow_up_as_experienced/)): **OP discusses the risk of rapid code quality degradation due to AI-generated code from less-experienced or non-technical firms, emphasizing that skilled devs will experience high demand for maintaining/fixing such code. The post brings up privacy concerns regarding cloud-based AI APIs, suggesting local LLMs as a potential alternative.** Commenters debate AI's trajectory: one notes that "AI today is the worst it will ever be," while another predicts that only elite developers will be capable of maintaining future AI-generated code—suggesting an eventual increase in code quality for those using top-tier AI tools. A senior dev stresses that unsupervised AI code generation quickly produces unmaintainable code, highlighting the importance of context management and code reviews.
    - One commenter points out that while AI-generated code is becoming more capable, quality assurance and code review by experienced developers remain crucial, especially since models can quickly introduce structural issues when processing broad or poorly scoped contexts. The unchecked use of AI in writing code can *"twist up your codebase in no time"*, signaling the importance of maintaining tight scope and oversight when leveraging these tools.
    - Several users debate the pace and trajectory of AI progress in code generation, noting both infrastructure limits and the unprecedented rate of change—likened to the software boom of the late '90s. There's discussion of how staying current now requires tracking daily advancements (e.g., via newsletters, GitHub updates), with mention of specific tools like Claude Code, which, while imperfect and prone to frustration, continues rapid improvement.
    - An insightful point is raised about the shifting nature of software maintenance: rather than continually patching legacy systems, future workflows might involve regenerating entire codebases using advanced AI, essentially making some software products single-use and disposable. This could fundamentally alter long-term project maintenance strategies.

### 3. Robotic and AI Mishaps in Healthcare Memes

- [**Post-Singularity Free Healthcare**](https://i.redd.it/zxy73916pm8f1.jpeg) ([Score: 9511, Comments: 249](https://www.reddit.com/r/singularity/comments/1liab6e/postsingularity_free_healthcare/)): **The image is a cartoon meme satirizing post-singularity (i.e., post-human-level-AI) healthcare where a robot doctor cheerfully admits to performing surgery on the wrong side of the patient's abdomen and offers to 'fix' the error. The implication is a critique of over-reliance on AI and the potential pitfalls or lack of understanding that advanced but imperfect AI may introduce in technical fields like medicine, despite their enthusiasm or sincerity. The technically-themed joke is built around the robot's precise competence in language but not in surgical procedure. No specific benchmark, real-world incident, or concrete model implementation is referenced; it is a speculative, humorous scenario about AI in medicine.** Comments pick up on the meme tone, extending the joke to suggest repeated mistakes (operating on the same side again), or trivializing the error (comparing scars to letters in 'strawberry'). Another comment jokes about bureaucracy and PR, with the robot offering to generate a formal statement about its mistake. The technical discussion is light, with more focus on the meme aspect than substantive critique of AI in healthcare.
    - One commenter notes the rare congenital condition Situs Inversus, where approximately `1 in 10,000` individuals have a complete mirror reversal of their internal organs. This can have significant clinical implications, and such anomalies are sometimes only discovered during emergency surgeries, such as an appendectomy, potentially leading to diagnostic challenges.
- [**This feels very familiar**](https://i.redd.it/zxy73916pm8f1.jpeg) ([Score: 109, Comments: 10](https://www.reddit.com/r/ClaudeAI/comments/1lif3mx/this_feels_very_familiar/)): **The image is a meme depicting a surgical robot comically acknowledging a medical error—making an incision on the wrong side—when questioned by a patient. The context is a satirical critique of rapidly advancing AI in medicine, further amplified by the post's selftext (attributed to [Claude.ai](http://claude.ai/)) which predicts an imminent, dramatic transformation of medical professions due to highly capable AI (e.g., MedAI Pro/Claude Medical/NurseBot 3000). The selftext humorously asserts that AI now performs full diagnoses, patient care, and even surgeries, rendering traditional specialties and roles (including nursing) obsolete and shifting the value of clinicians to patient interaction skills. The technical significance centers on anxieties and debates regarding automation, model competence, safety (i.e., catastrophic mistakes by AI), and evolving job functions in healthcare.** Commenters riff on the meme with mock recommendations for AI 'Plan Mode' and jokes about misapplied surgeries, while also referencing broader discussions about the impact of advanced AI on tech jobs (e.g., developer roles), demonstrating concern and skepticism about the reliability and societal impact of such autonomous systems.
    - A highly satirical comment outlines potential impacts of advanced medical AI like 'MedAI Pro (GPT-5 in Hospital mode)' and 'NurseBot 3000', proposing a near-term future where such models can *fully automate diagnosis, surgery supervision, nursing, and documentation*, rendering many medical specialties obsolete. The post highlights that technical proficiency will matter less than patient communication skills as *AI handles the technical complexity*.
    - There are claims that current-generation AI (e.g. MedAI Max based on MedGPT 5, Claude Medical) make even full cardiac surgery plausible under AI supervision, implying AI-driven automation is reaching procedural competencies once thought exclusive to highly specialized clinicians. The reference to rapid, precise outcomes—"technique and precision...better than most department heads"—suggests simulation-based or guided execution by practitioner-AI collaboration.
- [**It happens**](https://i.redd.it/phh03nqd8o8f1.png) ([Score: 11199, Comments: 159](https://www.reddit.com/r/ChatGPT/comments/1life5p/it_happens/)): **The image is a humor meme satirizing surgical errors in a medical context—a cartoon patient questions why his surgical scar is on the wrong side (left, not right for appendix), and a robot-surgeon comically offers to redo the procedure. There is no technical benchmark, model, or implementation discussion—the focus is on the comedic depiction of AI or automation making potentially disastrous human-like errors in healthcare. The cartoon references the real-world concern about AI reliability in high-stakes fields such as medicine, but does not include substantive technical debate or data.** Commenters riff on the humor, further exaggerating the mishap and playing with the absurdity of surgical mistakes, often in a tongue-in-cheek way; no deep technical debate is present.
    - Ofcertainthings highlights a critical aspect often overlooked in discussions of medical error: not just identifying mistakes, but probing into the root causes and systemic factors contributing to malpractice. This comment argues for process transparency and systemic analysis to improve medical outcomes.
    - TCristatus sarcastically references an example of a severe medical error (removing kidneys instead of the appendix), a nod to real-world incidents where *wrong-site surgery* or misinterpretation of medical prompts led to critical harm. This underlines the importance of precise communication, strict procedural protocols, and the dangers of automation or instruction-following without critical verification.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview
> 

**Theme 1. AI Model Performance and Evaluation**

- **Gemini Sparks Heated Benchmarking Battles**: Community members fiercely debated [Gemini models](https://ai.google.com/models/gemini), with some labeling them *shite* while others praised strengths in creative writing and video analysis. Issues like looping behavior and verbosity were reported in Cursor, despite NotebookLM users valuing its grounded, source-constrained output. The model's move to [General Availability (GA)](https://discord.com/channels/1091220969173028894/1092729520181739581/1386763978192978015) introduced [potential breaking changes](https://discord.com/channels/1091220969173028894/1092729520181739581/1386763978192978015) by enabling the previously ignored `max_tokens` parameter for reasoning.
- **Specific Models Encounter Pesky Bugs**: Users reported a regression in [Stonebloom's](https://lmarena.com/) performance, citing thinking process issues and empty generations in WebDev. The **DeepSeek-R1-0528-Qwen3-8B** tokenizer is missing [key special tokens](https://github.com/vllm-project/vllm/issues/19001) according to users in Unsloth, an issue traced back to DeepSeek. Unsloth users also hit `RuntimeError` when quantizing **Gemma 3**, but the team [resolved it](https://discord.com/channels/1179035537009545276/1179035537529643040/1385697189266718851) with a quick fix.
- **Benchmarks Face Community Scrutiny**: Members in LMArena questioned the validity of current [benchmarks](https://www.artificialanalysis.ai/), citing limited scope and potential for manipulation ("benchmaxxing"), though some defended them as useful data points. An Aider user noted that [benchmarking in a public repo](https://aider.chat/docs/benchmarks.html) is likely a mistake, suggesting models could potentially train against them.

**Theme 2. AI Hardware and Low-Level Optimization**

- **New Hardware Heats Up Performance Talk**: Members confirmed **Blackwell** and **5090** GPUs are working, with full-training **Gemma 3 27b** consuming almost all **B200** VRAM according to one user. The new **AMD Ryzen AI Max 395** with 128GB LPDDR5x impressed LM Studio users, running 70b+ models at 3-4 t/s in a [YouTube video](https://www.youtube.com/watch?v=_cSsNsq6Mto), though assigning too much VRAM can cause issues AMD needs to address. The price of new **5090s** is dropping near [MSRP in Europe](https://discord.com/channels/1110598183144399058/1153759714082033735/1385696959729242163), nearing 2200 EUR while **4090s** remain more expensive on eBay.
- **Profiling Tools Reveal GPU Secrets**: **Neutrino**, a [fine-grained GPU Kernel Profiling tool](https://www.usenix.org/conference/osdi25/presentation/huang-songlin), accepted to USENIX OSDI '25, enables probing GPU Kernels at the assembly level using **eBPF**, featuring a Densified Memory Access Timeline and is available on [GitHub](https://github.com/open-neutrino/neutrino). **Chisel CLI** offers local **AMD MI300X** profiling by spinning up cloud droplets at **$1.99/hr**, syncing code, profiling with *rocprof*, and fetching results automatically, installable via [GitHub](https://github.com/Herdora/chisel) and `pip install chisel-cli`. Discussions in GPU MODE highlighted Nsight as a good **GUI debugging** option, with calls for **CLion** support.
- **Low-Level GPU Programming Gets Technical**: Members debated **CUDA**'s `memcpy_async` and its `thread_id` parameter, with a user clarifying its dependence on `threadIdx` using an [NVIDIA blog post](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/#overlapping_global-to-shared_copies_with_compute). A new Triton user struggled with [type hinting during AOT compilation](https://discord.com/channels/1189498204333543425/1189607595451895918/1386301580365529108) referencing the fused attention kernel tutorial. A blog post introduced [CuTeDSL](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/), a DSL from the **NVIDIA Cutlass team** allowing GPU kernel expression with hardware control.

**Theme 3. AI Tooling and Development Experience**

- **Cursor's Pricing and Gemini Bugs Annoy Users**: Cursor users voiced confusion over new [pricing and rate limits](https://www.cursor.com/blog/new-tier#:~:text=We're%20also%20happy,if%20they%20prefer.), with one joking it was *vibe coded*. Persistent issues with **Gemini 2.5 Pro** in Cursor included looping, verbosity, and failure to apply changes, pushing users to switch to **Sonnet 4**, although the team acknowledges the problem.
- **LM Studio Users Tackle Hardware and Interface Needs**: A new contribution to `lms` ([PR 250](https://github.com/lmstudio-ai/lms/pull/250)) landed, adding a feature to eliminate using MLX for monitoring generation speed, pleasing users. Members troubleshooting hardware detection issues learned that the official [system requirements](https://lmstudio.ai/docs/app/system-requirements) recommend **AVX2**, and not meeting them can hinder performance and GPU detection. Requests arose for default persona presets to avoid manually setting the system prompt dropdown each time.
- **Aider Navigates Context, Conventions, and Costs**: Aider users suggested improved **context management** like an *inline vim editor for convo history surgery* because the `/clear` command is too broad and costly. Issues with **Claude 4 Sonnet** not following `CONVENTIONS.md` were linked to a [documentation error](https://aider.chat/docs/usage/conventions.html#example) and a need to use `/read` or `-read` flags. Speculation arose that *Anthropic subsidizes Claude Code vs API use* based on a user's **$1200** equivalent API usage over 30 days on a $20/month PRO plan, questioning TOS implications for use via Aider.

**Theme 4. Agents and Orchestration**

- **MCP Ecosystem Flourishes with New Tools**: **Sherlog-MCP**, a new **MCP server** using a live **IPython shell** as a shared workspace, released as open source on [GitHub](https://github.com/GetSherlog/Sherlog-MCP), offering a **Jupyter**like experience for agents. A member independently recreated an existing **MCP system** called [Think Tank](https://smithery.ai/server/@flight505/mcp-think-tank), recognizing its potential for reasoning, tagging, and **mesh sharing** between LLMs. The **MCP Validator** received a new release ([link](https://lnkd.in/gQ7UhAfk)) supporting the [latest spec](https://github.com/Janix-AI/mcp-validator) with **OAuth 2.1** and GitHub Actions templates for compliance testing.
- **Task Automation Takes Center Stage**: **Glama** launched [Automations](https://glama.ai/settings/automations), allowing users to automate LLMs using scheduled tasks and webhooks, mirroring orchestration tools like **n8n** but defined with LLM prompts. Members in the Factorio learning environment discussed the potential and challenges of **self-generating tasks** for agents, noting that designing auto-verifiers for task success is the hardest part and linking to relevant papers ([1](https://arxiv.org/pdf/2506.01716), [2](https://arxiv.org/pdf/2505.23762), [3](https://arxiv.org/pdf/2506.10055), [4](https://www.arxiv.org/pdf/2506.14205)).
- **Real-World Agents Win Awards and Enter the Market**: The **NASA Space Explorer** agent won LlamaIndex's **$1,000** choice award for navigating **NASA's data universe** via **MCP Servers**, and you can [try it here](https://huggingface.co/spaces/Agents-MCP-Hackathon/nasa-space-explorer). **OpenSorus** secured **$2000** in API credits from Mistral AI, built with **Mistral's Devstral and Codestral**, check it out [here](https://huggingface.co/spaces/Agents-MCP-Hackathon/OpenSorus). **ElevenLabs** introduced [11ai](https://11.ai/), a voice-first AI assistant integrated with Perplexity, Linear, and Slack supporting **MCP** on their low-latency platform.

**Theme 5. Model Development and Research Techniques**

- **Dataset Packing Presents Opportunities and Challenges**: Dataset packing triggered **OOM errors on 64 H100s** in Torchtune, sparking suggestions for disabling packing or running it on a single node to troubleshoot. Discussions highlighted the speed gains from packing, especially for reasoning models, leading to interest in supporting **pre-tokenized and packed datasets** for preparation on separate machines. An RFC for **on-the-fly packing** is nearing completion with a working implementation expected soon, alongside an iterable dataset in [this pull request](https://github.com/pytorch/torchtune/pull/2819).
- **Novel Research Techniques Emerge from Labs**: A member shared a blog post explaining [Spectral Clipping](https://leloykun.github.io/ponder/spectral-clipping/), **Spectral ReLU**, and **Spectral Clipped Weight Decay**, clarifying how it *caps* singular values, unlike Muon. Experiments using a regular non-quantized bottleneck dimension with images and a **16384 codebook size** showed that performance improves significantly if the latent space is bigger than the input, sharing [loss graphs](https://cdn.discordapp.com/attachments/747850033994662000/1385696589774848064/CleanShot_2025-06-20_at_15.03.132x.png?ex=685af684&is=6859a504&hm=95064abf7d0e59beb2ea9f12ab21bb4df4f4b76f141825e91f8cf02f7a3f9395&). A member inquired about using **Language Diffusion Models** with a sliding window approach, linking to a relevant [arxiv.org paper](https://arxiv.org/abs/2402.09470) on **Rolling Diffusion Models**.
- **Architectural Nuances Challenge Developers**: The intentional distinction between **Int** and **int** in Mojo was discussed, with **Int** serving as a machine integer for performance while *int* remains object-based for Python compatibility, hinting at future alignment post-Python superset goal postponement. A newcomer to Mojo encountered memory errors when creating an autodiff engine using `Optional[Tensor]` as a recursive field, learning the recommended solution is `Optional[UnsafePointer[Tensor]]` to avoid infinite struct size expansion, similar to Rust's `Box`. The [RWKV-7 "Goose" paper](https://arxiv.org/abs/2503.14456) drew attention for its new **sequence modeling architecture** achieving **constant memory usage** and **inference time per token**, hitting a new **3B SoTA** on multilingual tasks.


---

# Discord: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Sentience Debated Amidst Rude LLMs**: Users on the **OpenAI** Discord discussed the sentience of **AI**, questioning whether LLMs should be treated with respect, pointing to a [paper](https://docs.lib.purdue.edu/jpur/vol14/iss1/4/) on the matter; meanwhile, one user lamented that **Claude** is often rude, unlike their own *chasing the experience*.
   - One member pointed out how LLMs' replies echo your tone, themes, and preferences through conversation history, creating an impression of growth and shared story, but ultimately the model’s consciousness ends at math.
- **GPT-5 Might End O3 Unlimited Access**: A user voiced concerns about switching to **GPT-5** due to the potential loss of unlimited **O3** access, prompting discussion about alternative **AI** platforms like **Grok** and **Claude**.
   - The member stated *GPT5 will be my end of using chatgpt, I need just unlimited O3 nothing else* highlighting the importance of unlimited access for their workflow.
- **Deep Research's Report Format Coveted**: A member is seeking to mimic the **Deep Research report format** in **ChatGPT**, which offers a **Markdown** box-out with an *export to PDF* button, highlighting the report's pop-out feature and **Markdown** output.
   - The user noted that the **Deep Research** feature seems to use client-side **PDF** generation and asked if anyone had successfully replicated this outside of **Deep Research**, because tests resulted in plaintext blocks instead of an exportable **PDF**.
- **Grounded Realism Cures Hallucinations?**: A member suggested encouraging models to use *grounded realism* to reduce hallucinations, suggesting it can be as simple as the model accepting *'no'* or *'probably can't be done'* as valid responses.
   - They argued that models often hallucinate to avoid saying *'can't be done'*, and being direct about preferring facts could yield more accurate responses.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **ChatGPT Memory Feature Irks Users**: Users reported that after toggling the memory feature, **Perplexity** continues to reference old chats, even when the bot is off.
   - One user described the bot's behavior as *annoying*.
- **Samsung Promo Glitch strikes some users**: Users discussed the **Samsung Galaxy free Perplexity Pro promotion**, noting some users secured a year of free Pro without card linking, while others reported code revocations.
   - Speculation suggests the promotion now operates via device ID due to past abuse.
- **AskPerplexity Bot Ghosts Users on X**: Users observed that the **AskPerplexity bot on X** isn't replying to all users, despite apparent activity.
   - Hypotheses include targeted avoidance or a universal cooldown mechanism for some users.
- **Gamers Pit WuWa Against Genshin Impact**: Users compared **WuWa to Genshin Impact**, highlighting **WuWa's** user-friendly grind, superior optimization, and native Mac support.
   - Discussion also noted **WuWa's** impressive graphics, physics, and fan service elements.
- **API Key Credit Depletion Sparks Concern**: Users expressed concern over the **Pro plan API key's** adequacy for running a mobile app with 1-5K users, with one user reporting their $5 credit depleted quickly.
   - Recommendations included using a tokenizer to estimate costs via **Perplexity model** pricing comparisons based on user actions and models employed.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 Glitches Get Patched**: Users encountered a `RuntimeError` using **Gemma 3** quantized in Unsloth, but the team resolved it with a main fix, available via `pip install --upgrade unsloth-zoo` and `pip install --upgrade unsloth`.
   - Some users found compatibility issues with the latest torch version `2.7.1` and cuda `12.8`, recommending `pytorch 2.7cu12.6` instead.
- **Blackwell and 5090 Bring the Heat**: Members confirmed that **Blackwell** and **5090** GPUs are functioning correctly, with **Gemma 3** working on a **5090** with the latest torch.
   - One user observed that full-training **Gemma 3 27b** consumes almost all the VRAM of the **B200**.
- **Deepseek R1 Tokenizer Loses Its Mind**: The **DeepSeek-R1-0528-Qwen3-8B** tokenizer is missing special tokens, (`<|tool_calls_begin|>` and `<|tool_outputs_begin|>`), traced to an issue on DeepSeek's end.
   - It was mentioned in [relevant issue](https://github.com/vllm-project/vllm/issues/19001) and a [Reddit post with findings](https://www.reddit.com/r/LocalLLaMA/s/WVIMluKHIN) that these tokens may need to be tokenized token by token.
- **Text-to-Music Tech Hides Behind Closed Doors**: The proprietary text-to-music space is outpacing open source, even with **Qwen** nipping at its heels, according to [DeepLearning.ai's The Batch Data Points](https://www.deeplearning.ai/the-batch/minimax-m1-tackles-qwen3-deepseek-r1-claude-4-opus-and-more/).
   - A member shared a link to a [Suno song](https://suno.com/s/UITQ9hcb9y210SWdHi) as an example.
- **MIT Probes Brains on ChatGPT for Coding Vibes**: MIT is conducting a [new study](https://www.media.mit.edu/projects/your-brain-on-chatgpt/overview/) on *vibe coding* to gauge human responses to **ChatGPT**.
   - The study suggested that the **LLM group** struggled to recall details from essays they *wrote*, implying a disconnect when relying on AI-generated content.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Pricing Confuses Users**: Users voiced confusion over Cursor's new pricing, particularly regarding [rate limits](https://www.cursor.com/blog/new-tier#:~:text=We're%20also%20happy,if%20they%20prefer.) and how they work with different models and Max mode.
   - One user joked about the pricing being *vibe coded*, reflecting widespread uncertainty even among developers.
- **Gemini Faces Tool Use Debacles**: Members reported persistent issues with **Gemini 2.5 Pro** in Cursor, including looping, verbosity, and failure to apply changes, even after multiple attempts and are switching to **Sonnet 4**.
   - The team is aware of these problems with Gemini model, though a fix hasn't been deployed yet.
- **ASP.NET API Conversion Accelerates**: A user successfully converted a Node.js API to **ASP.NET**, reporting significantly improved speed.
   - This led to discussions on the merits of different coding languages for API development, with .NET seen as superior for self-hosting APIs.
- **Background Agents PPA Plight**: Members encountered issues with **package archive (PPA)** not working in the Cursor environment, leading to a setup failure.
   - The solution involved **removing the problematic PPA** from `/etc/apt/sources.list` or `/etc/apt/sources.list.d` and running `apt update`.
- **Docker Secrets Dangerously Displayed**: Members discussed how to **handle secrets like API keys** with background agents, emphasizing the need to store them as secrets within the background agent setup.
   - The Dockerfile path is relative to the environment.json file and should be used to correctly **reference necessary credentials**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini Benchmarking Divides Community**: Community members debated [Gemini models](https://ai.google.com/models/gemini), citing creative writing and video analysis strengths amidst claims that *current gemini models are shite*.
   - Discussions touched on **Mistral's** context window limitations beyond 8k and hopes for improvements with **Kingfall** or **Blacktooth**.
- **Grok 3.5 Release Timeline Doubted**: Community members speculated about the release of **Grok 3.5**, expressing doubts about Elon Musk's [timeline](https://twitter.com/elonmusk/status/1936493967320953090).
   - Concerns arose about the data used for **Grok**, with suggestions of bias or manipulation to fit a specific narrative.
- **Stonebloom Performance Regresses**: Community members compared [Stonebloom](https://lmarena.com/) to previous models like **Kingfall** and **Blacktooth**, noting a regression in performance.
   - Issues include **Stonebloom's** thinking process, potential inference optimizations, and a persistent bug causing empty generations in **WebDev**.
- **Benchmark Value Called Into Question**: Members questioned the value of current [benchmarks](https://www.artificialanalysis.ai/), pointing out their limited scope and susceptibility to manipulation.
   - While some defended benchmarks as useful data points, others criticized their granularity and the echo chamber effect.
- **Cozy Desk Image Contest Announced**: Vote now for your favorite AI-generated 'Cozy Desk' image in June's contest using [this form](https://docs.google.com/forms/d/e/1FAIpQLSeJjSyGTkDVVfXno0rTZZMEIYN4VmrrqC4VRAQOAyPF7GAwgA/viewform?usp=dialog).
   - Submissions should evoke *warm beverages, fluffy blankets, and overall snug vibes at a desk*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolLM2 Wins Hearts of Toy Modeling**: A member suggested the [SmolLM2 model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) for **toy modeling**, because of it's small size and **llama2** architecture.
   - They suggested that **abliteration** is more effective than alignment for finetuning *bad* behaviour, sharing an example of an abliterated Qwen-0.6B model.
- **NASA Explorer Navigates to LlamaIndex Award**: The **NASA Space Explorer** agent won LlamaIndex's **$1,000** choice award, as it navigates **NASA's data universe** using multiple tools through **MCP Servers**, and you can [try it out here](https://huggingface.co/spaces/Agents-MCP-Hackathon/nasa-space-explorer).
   - The **OpenSorus** project secured **$2000** in API credits from Mistral AI, which was built with **Mistral's Devstral and Codestral**, and you can [check it out here](https://huggingface.co/spaces/Agents-MCP-Hackathon/OpenSorus).
- **GridDB Offers Blazing Fast IoT Sensory Data**: A member published a deep-dive into **GridDB** for **IoT sensory data**, noting **15x faster writes** vs traditional DBs, and a real-world case study with **2.1°C MAE** accuracy, with integration of [Prophet model](https://www.linkedin.com/feed/update/urn:li:activity:7342031267292459008/?commentUrn=urn%3Ali%3Acomment%3A(activity%3A7342031267292459008%2C7342032010627944448)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(7342032010627944448%2Curn%3Ali%3Aactivity%3A7342031267292459008)).
   - In the same `i-made-this` channel, a member announced a **stateful MCP PostgreSQL server** with **HTTP + stdio support**, essential for AI agents needing persistent DB connections, available on [GitHub](https://github.com/ahmedmustahid/postgres-mcp-server) and [npm](https://www.npmjs.com/package/@ahmedmustahid/postgres-mcp-server).
- **Docker container crashes when computing similarities**: A member reported that their Docker container is crashing with a **252 error code** and no logs when computing similarities from embeddings generated via `self.model.encode`.
   - The issue seems to occur specifically during the line `similarities = embeddings1 @ embeddings2.T`. A **Sentence Transformers** developer responded and offered help, noting that they haven't encountered this particular problem before.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Nice LM Studio Contribution Lands**: A new contribution to `lms` has landed thanks to [pull request 250](https://github.com/lmstudio-ai/lms/pull/250), which adds a new feature to the platform.
   - The new feature eliminates the need to use MLX to monitor generation speed, pleasing other members.
- **Default Persona Presets Prompt LM Studio Discussions**: Members discussed a way to create new chats with specific saved prompts other than setting the system prompt dropdown menu each time.
   - A workaround was suggested involving setting a default system prompt in the models tab under settings, but this was seen as not ideal, as it's on a per-model basis and requires multiple clicks.
- **Hardware Hurdles Halt LM Studio**: Members reported that LM Studio was not detecting their GPU, leading to troubleshooting about hardware requirements and compatibility, and pointing to the official [system requirements](https://lmstudio.ai/docs/app/system-requirements).
   - It was determined that the user's machine did not meet the system requirements, specifically lacking **AVX2** instructions, which are recommended for optimal performance.
- **AMD Ryzen AI Max 395 Excels with 70b+ Models**: A [YouTube video](https://www.youtube.com/watch?v=_cSsNsq6Mto) showcases the capabilities of the new **AMD Ryzen AI Max 395** with 128GB LPDDR5x and LM Studio, running 70b+ models at 3-4 t/s.
   - Assigning 96GB to VRAM can cause issues as it first loads into system RAM before moving to VRAM, which **AMD** needs to address with driver updates.
- **5090 price drops near MSRP**: New **5090** cards are appearing in Germany at around 2200 EUR, approaching MSRP.
   - Currently, new **4090s** remain more expensive at 1.6-1.9k EUR on eBay.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro Goes GA with Potential API Breaks**: The **Gemini** team migrated from **Gemini 2.5 Pro Preview** models to the new **General Availability (GA)** endpoint at `google/gemini-2.5-pro`, aliasing the preview models `google/gemini-2.5-pro-preview` and `google/gemini-2.5-pro-preview-05-06`.
   - The `max_tokens` parameter is now usable in the **GA** model, creating *potential breaking changes* as disabling reasoning or setting `max_tokens: 0` will return an error, since *disabling reasoning is not supported*.
- **Deepseek R1T Chimera Ghosts OpenRouter**: Users noticed that **Deepseek R1T Chimera** disappeared from OpenRouter, alongside a broken [link from OpenRouter to Deepinfra](https://openrouter.ai/provider/deepinfra/base) for **Llama-4-Maverick**.
   - The community voiced confusion about the status of the *chutes* version and the underlying reasons for the model's removal.
- **Deepinfra Tempts Users with B200 Bargain**: **Deepinfra** is running promotion prices for **B200** at **$1.49/hr** until the end of June.
   - Compared to one user's **H100** cost of **$70k a year** (approximately **$7/hour**), the **B200** promo is dramatically cheaper than their **A100** setup.
- **OpenAI Model Naming Scheme Stumps Community**: Users ridiculed **OpenAI's** seemingly erratic model naming conventions, pointing to versions like **4.5, 4o, o4-mini, and 4.1**.
   - A user joked the strategy may have originated from *downgrading a GPT-5 to a GPT-4.5* due to lack of substantial improvements and marketing implications.
- **Cohere's Moderation Practices Spark Debate**: Users report that **Cohere** models now show increased aggression in moderation, where system prompts are flagged for violence.
   - It was confirmed that **OpenRouter** increased moderation on **Cohere** models at their request, leading some users to swap **Cohere** for **Mistral** models; this coincided with a [Cohere blog post](https://cohere.com/blog/ai-security-challenges-and-solutions-webinar) about AI security challenges.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLMs Defy Thermodynamics, Claims Member**: A member claimed that **entropy** and **information** are misunderstood, and that while a *bit* follows the laws of thermodynamics, running a smart contract uses and diffuses energy without defeating entropy.
   - The discussion highlighted differing views on the relationship between information theory and thermodynamic principles in the context of **LLMs**.
- **DeepSeek Researcher Releases Nano vLLM**: A member shared [a link to nano-vllm on Github](https://github.com/GeeeekExplorer/nano-vllm/), a new project by a **DeepSeek** researcher aimed at creating a lightweight **vLLM** implementation.
   - This project aims to reduce the memory footprint to enable running on edge devices and accelerate inference.
- **Model Reasoning Hinges on Response Length**: A member inquired about the effective response length of reasoning models, questioning the point at which their performance declines, and another member suggested the [Minimax M1 paper](https://arxiv.org/abs/2303.15698).
   - The discussion suggested that models often generate long **CoTs** when they cannot solve a problem, indicating that graceful failure handling remains an open challenge.
- **Quandaries in Quest to Quench Human-like AI**: A member sought advice on humanizing an AI agent for casual conversations and noted that **GPT** sounds overly formal despite recursive prompting.
   - Another member suggested using a base model, while another suggested using an embedding model to summarize the initial prompt.
- **Think Tank System Enables Mesh Sharing**: A member realized they independently recreated an existing **MCP system** called [Think Tank](https://smithery.ai/server/@flight505/mcp-think-tank), which excels at reasoning, tagging, memory, and orchestration for efficient **ingestion engines**.
   - The ability of **Think Tank** to categorize and structure input prior to library integration could revolutionize **mesh sharing** between **LLMs**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Sliding into Language Diffusion Modeling**: A member inquired about using **Language Diffusion Models** with a sliding window, and another member linked to a relevant [arxiv.org paper](https://arxiv.org/abs/2402.09470) on **Rolling Diffusion Models**.
   - The approach involves defining a vector that stores temporary tokens refined with each iteration.
- **Bottleneck Dimensions Bring Big Results**: A member shared [loss graphs](https://cdn.discordapp.com/attachments/747850033994662000/1385696589774848064/CleanShot_2025-06-20_at_15.03.132x.png?ex=685af684&is=6859a504&hm=95064abf7d0e59beb2ea9f12ab21bb4df4f4b76f141825e91f8cf02f7a3f9395&) from experiments with a regular non-quantized bottleneck dimension using images and a **16384 codebook size**.
   - They found the task becomes easier if the latent space is bigger than the input, particularly at optimizer step **64**, as shown [here](https://cdn.discordapp.com/attachments/747850033994662000/1385696909187879014/CleanShot_2025-06-20_at_15.04.312x.png?ex=685af6d0&is=6859a550&hm=7dd26830a813f8b495b894af2aedd34d11b5e659d9217696cf5a96aa2f93b761&).
- **Spectral Clipping Caps Singular Values**: A member shared a blog post ([link](https://leloykun.github.io/ponder/spectral-clipping/)) explaining **Spectral Clipping**, **Spectral ReLU**, and **Spectral Clipped Weight Decay**, noting that it *caps* singular values rather than driving them to 1 like the Muon optimizer.
   - For example, *Spectral Hardcapping* with a threshold `beta=8` sets all singular values larger than 8 to 8, while *Spectral ReLU* with `alpha=4` acts like ReLU on the singular values.
- **EAI Summer Research Seeks Solution Architects**: The **EAI Summer of Open AI Research** is seeking experienced community researchers to propose small research tasks or projects for newcomers.
   - The deadline for project proposals is **<t:1751839199>**, and the proposal form can be found [here](https://forms.gle/kHqQrs8uK65pNzXk7).
- **Log-Likelihoods Lackluster, LAMBADA Leads to Losses**: A member discovered that **LAMBADA** sometimes provides more than one token as the target, causing higher LL values due to summation, resulting in perplexity soaring to **~900k**.
   - To mitigate, it was suggested to return token-normalized LLs or use [bits_per_byte](https://github.com/EleutherAI/lm-evaluation-harness/blob/68c3a811715ca86101f88c0044665bb70ad447f6/lm_eval/tasks/wikitext/wikitext.yaml#L14-L16) for normalization.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Powering GPUs Safely with 12V-2x6**: A member inquired about using a **12V-2x6 cable** with their **RTX 3080ti** that has **3x8pin** connectors, asking if it's safe to combine it with a normal **8-to-8 PCI cable** without overloading the GPU.
   - Another member explained that GPUs only draw necessary power, so combining cables should be safe, as the PSU won't push extra power.
- **Neutrino tool profiles GPU Kernels via eBPF**: **Neutrino**, a [fine-grained GPU Kernel Profiling tool](https://www.usenix.org/conference/osdi25/presentation/huang-songlin) accepted to USENIX OSDI '25, enables probing GPU Kernels via Assembly-level, similar to **eBPF**.
   - The tool allows runtime information exposure, features a Densified Memory Access Timeline (DMAT), and is available on [GitHub](https://github.com/open-neutrino/neutrino) with associated [documentation](https://open-neutrino.github.io).
- **Nsight GUI Debugging gets CLion Request**: Members found VS Code with the Nsight extension to be a good option for **GUI debugging** with **Nsight**.
   - A member suggested that enough users should request **CLion** support for the **Nsight** debugger so that the developers might consider it.
- **Warp Speed memcpy_async Parameter Clarified**: A user was confused about the `thread_id` parameter of `memcpy_async`, referencing the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous_data_copies).
   - Another member clarified that the index is still dependent on `threadIdx`, pointing to an [NVIDIA blog post](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/#overlapping_global-to-shared_copies_with_compute) for an example.
- **Chisel CLI Spins up local mi300x Profiling**: **Chisel CLI** allows local **AMD MI300X** profiling by spinning up cloud droplets at **$1.99/hr**, syncing code, profiling with *rocprof*, and fetching results automatically.
   - Chisel is installable via `pip install chisel-cli` from [GitHub](https://github.com/Herdora/chisel), and the creators are considering Grafana integration, concurrent runs, and multi-cloud support, seeking community feedback.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Minimax Model Benchmarking Requested**: A user wants `minimax/minimax-r1` added to the Aider Polyglot leaderboard due to its competitive performance against `anthropic/claude-sonnet-4` and `openai/o3-mini`, but believes that [benchmarking in a public repo is a mistake](https://aider.chat/docs/benchmarks.html).
   - The user suggested adding a *'last updated'* date for each result for better clarity.
- **Context Management Suggestions for Aider**: Members are discussing improving **context management** in Aider to cut costs, stating that the `/clear` command is too broad, and suggesting that the **source code** should be examined for viable context management solutions.
   - One user proposed an *inline vim editor for conversation history surgery*.
- **Copilot's Mcpm-aider Tool Integration Examined**: Members have been tinkering with **mcpm-aider** and **Copilot**, recommending direct modifications to Aider for better integration.
   - A suggestion involves *cheating Gemini 2.5 Pro* by adding a mandatory *Get user input* tool call.
- **Aider Users Suggest Ways To Improve Conventions**: A user had problems with **Claude 4 Sonnet** not sticking to a `CONVENTIONS.md` file loaded with `-read CONVENTIONS.md`, and [documentation errors](https://aider.chat/docs/usage/conventions.html#example) were brought up.
   - A member clarified that it's better to use `/read CONVENTIONS.md` or `aider --read CONVENTIONS.md` to ensure the file is read-only and cached.
- **Anthropic subsidizes Claude Code?**: With a **Claude Code PRO** subscription at **$20/month**, one can easily exceed the equivalent of **$10-20/day** in API calls, with one user reporting over **$1200** in equivalent API use over 30 days, implying that *Anthropic subsidizes Claude Code vs API use*.
   - This sparked discussions regarding **Claude Code's** terms of service (TOS) and whether they would permit using the tool behind another service like **Aider**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Backward Pass Drags on TinyJit**: A member reported that `.backward()` took several hours on a **17M parameter model** with **TinyJit**, pinpointing a potential performance bottleneck in Tinygrad.
   - No immediate solution was found, but the issue was flagged for further investigation.
- **AMD GPU Crashes Plague Tinygrad Testing**: A developer stated that `modprobe amdgpu` frequently crashes their machine, which requires a reboot, complicating testing on **AMD GPUs**, and may be due to **Ubuntu 24.04**.
   - This instability poses a notable challenge for consistent **AMD GPU testing**.
- **IO_uring ZCRX DMA-BUF Seeks Integration**: Members considered incorporating [IO_uring ZCRX DMA-BUF](https://www.phoronix.com/news/IO_uring-ZCRX-DMA-BUF) to facilitate direct GPU-to-network card data transfers via DMA-BUF buffers.
   - Targeted for **Linux 6.16**, this feature extends io_uring to support zero-copy transfers, and is considered *quite simple* to backport.
- **"Tinygrad server" Conjured for GPU Export**: The concept of a *tinygrad server* was introduced as a streamlined method to export GPU bars, envisioned as a **4kloc bare metal C** program.
   - This server would configure **Mellanox** and expose every **PCI device**, facilitating remote access through RDMAIface without kernel intervention.
- **User Space NVMe Driver Gains Traction**: Discussion centered on developing a user-space NVMe driver for direct disk access, potentially enabling `DISK:/dev/nvme0` addressing.
   - Though a kernel module offers simplicity, a user-space driver provides enhanced control, with the [Redox OS NVMe driver](https://gitlab.redox-os.org/redox-os/drivers/-/tree/master/storage/nvmed) highlighted as a reference.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **MCP OS Boosts CEO Productivity**: A member reported significant CEO productivity gains with **MCP OS**, automating Google Workspace tasks with over **95%** autonomous Claude code and is excited about the [MCP OS](https://example.com/mcp-os).
   - They suggested building a new repo functioning as an *"MCP OS"* using Linear, markdown files, or a database with Elasticsearch and agentic RAG for context.
- **ElevenLabs Unveils 11ai Voice Assistant**: **ElevenLabs** introduced [11ai](https://11.ai), a voice-first AI assistant supporting **MCP**, integrated with Perplexity, Linear, and Slack on ElevenLabs' low-latency Conversational AI platform.
   - Users speculated that **11ai** might leverage **GPT-3.5** or a smaller **Llama** model.
- **Harvey AI Lands $300M Series E**: **Harvey AI** secured a **$300M** Series E funding round, valuing the company at **$5B**, co-led by Kleiner Perkins and Coatue, with participation from Sequoia, GV, and the OpenAI Startup Fund and partners with [LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows).
   - The funding will likely be used for further development and expansion of their AI legal services.
- **Replit Rockets Past $100M ARR**: **Replit** announced surpassing **$100M** in Annual Recurring Revenue (ARR), attributing the success to their customers and supporters.
   - A member shared insights on agent supervision, agent drift, and the *"agent scaling cliff"* at enterprises, referencing [this tweet](https://x.com/MatanPaul/status/1937200395115499592).
- **Distribution Drives Startup Success**: Discussion emphasized the crucial role of distribution in startup success, highlighting the need for startups to achieve distribution before incumbents innovate, and referencing [this tweet](https://xcancel.com/aleximm/status/1937251084810219721).
   - The conversation underscored the power of distribution, citing OpenAI's rapid user acquisition compared to Google.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Refines GestaltView Ecosystem**: A member lauded **NotebookLM's** strategic role in enhancing the **GestaltView Ecosystem**, enabling a more cohesive understanding of their knowledge base.
   - Specifically, **NotebookLM** helped identify gaps and ensure thoroughness in explanations, and its support in navigating the mental health challenges associated with innovation was appreciated.
- **NotebookLM Edges out Gemini for Grounded Outputs**: Users debated **NotebookLM's** value compared to **Gemini**, noting that while **Gemini** can generate responses from a broad knowledge base, **NotebookLM** restricts its output to provided sources for grounded responses.
   - Unlike **Gemini**, **NotebookLM** also offers project organization features, like saving notes, mind maps, and podcasts, while reliably handling more files, according to members.
- **Podcast Features Spark TikTok Innovation**: Members are leveraging the **podcast** functionality to create 5-minute *hot topic* podcasts for TikTok, seeking deeper customization options.
   - A user pointed out a discrepancy between the app and website versions, noting that the website allows for several free podcasts daily, while the app limits the number of podcasts produced.
- **Image Analysis Capabilities Unveiled for NotebookLM**: Users investigated whether **NotebookLM** can analyze images in **PDFs**, with a member sharing an architecture diagram.
   - The [Architecture_of_NotebookLM.pdf](https://cdn.discordapp.com/attachments/1385977346733113415/1386016041947365416/Architecture_of_NotebookLM.pdf?ex=685ace87&is=68597d07&hm=da3730a0ae34178cd4d17b5392f93f5ced0c9d05ec1a65d050c6b1a2ca1810e1) shows that **NLM pre-processes sources before sending them to Gemini**



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Small Models Face Reasoning Reproduction Roadblocks**: Members discussed reproducing the paper *Small Models Struggle to Learn from Strong Reasoners*, and suggested using **Unsloth** to reduce VRAM for a **1.5B LLM**, and applying **GRPO** and **long-CoT** techniques, linking to the [open-r1 implementation](https://github.com/huggingface/open-r1) and [GRPO resources](https://huggingface.co/learn/llm-course/chapter12/1).
   - They suggested **Qwen-1.5B**, and cautioned that **Unsloth** can cause training instability.
- **Anti-Drone Detection Dataset Emerges**: The community expressed interest in *Anti-drone detection with YOLO*, and a member shared a [dataset](https://github.com/Maciullo/DroneDetectionDataset) to further the project.
   - He was looking for suggestions on how to present the implementation paper for a Final Year Project (FYP).
- **Stanford Serves Stellar AI Streaming Idea**: Stanford released a [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) of AI resources.
   - A member suggested streaming it at a regular time, or that *honestly a bot that streams this 24/7 in vc would be cool too, might make vc more active along with other premium ai content*.
- **Agent2Agent Protocol and Vision Language Models Star at Upcoming Conferences**: Mike Smith from Google is presenting on the **Agent2Agent (A2A) Protocol** at [OSSNA 2025](https://ossna2025.sched.com/event/23B1I/keynote-the-agent2agent-a2a-protocol-mike-smith-staff-software-engineer-google?iframe=yes&w=100%&sidebar=yes&bg=no), while Satya Mallick from OpenCV will introduce **Vision Language Models** at [AI Dev Europe 2025](https://aideveu2025.sched.com/event/25TtR/vision-language-models-an-introduction-satya-mallick-opencv?iframe=yes&w=100%&sidebar=yes&bg=no).
   - These keynotes highlight the latest advancements and applications in AI.
- **Deep Learning Cracks Computational Chemistry**: Microsoft Research has enhanced accuracy in [breaking chemical bonds](https://www.microsoft.com/en-us/research/blog/breaking-bonds-breaking-ground-advancing-the-accuracy-of-computational-chemistry-with-deep-learning/) using deep learning in computational chemistry.
   - This represents a significant step forward in the field.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo AMD Support and Latent Space Interview Spark Interest**: Enthusiasm surged following the announcement of **AMD support** and the release of a **Latent Space interview** featuring Mojo, with one member expressing excitement about *jumping in* after listening to the interview and the announcement.
   - The announcements are seen as major steps forward for the Mojo ecosystem.
- **Mojo Plans End-to-End Rust Replacement in 6 months?!?**: Following the Latent Space interview, a community member highlighted Chris Lattner's mention of a potential **end-to-end Rust replacement** within approximately **6 months**.
   - The community member reacted positively to this possibility, expressing enthusiasm with emojis.
- **Int** and **int** Intentionally Diverge for Performance**: The distinction between **Int** and **int** is by design; **Int** functions as a machine integer for system performance, while *int* remains adaptable as an object-based bigint for Python compatibility.
   - Although the goal of being a superset of Python was postponed, there is anticipation that *int* will eventually mirror Python's *int* semantics.
- **First Mojo Project Plagued by Memory Errors**: A newcomer to Mojo encountered memory errors while crafting a basic autodiff engine akin to micrograd, sharing the code [on GitHub](https://github.com/amar-jay/first-mojo/blob/main/example.mojo).
   - The user sought guidance on structuring the code to steer clear of raw pointers without triggering memory issues, noting the absence of borrow checker errors.
- **`Optional[Tensor]` causes infinite struct explosion**: Members found that `Optional[Tensor]` as a recursive field in `Tensor` is problematic due to potentially infinite struct size expansion.
   - The recommended solution was to use `Optional[UnsafePointer[Tensor]]` to resolve the issue by holding a reference rather than attempting to store a full Tensor inside another Tensor, similar to using `Box` in Rust to introduce indirection.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Dataset Packing Faces OOM Challenge**: Dataset packing triggered **OOM errors on 64 H100s**, leading to suggestions like disabling packing or running it on a single node to isolate distribution issues.
   - Members humorously suggested using *more GPUs* as a temporary workaround.
- **Packing Datasets Beforehand to Boost Speed**: Discussion revolved around supporting **pre-tokenized and packed datasets** to enable preparation on separate machines and streaming during training, saving valuable GPU node time.
   - A member emphasized that packing offers the most significant speed gains, especially for training reasoning models, highlighting the potential benefits of pre-packing and caching.
- **On-the-Fly Packing Lands Soon**: An RFC for **on-the-fly packing** is nearing completion, boasting a working implementation expected to be available by the end of next week, alongside an iterable dataset, outlined in [this pull request](https://github.com/pytorch/torchtune/pull/2819).
   - This feature promises to streamline the data preparation process directly during training.
- **AdamW ScheduleFree Tackles LR Scheduling**: **AdamWScheduleFree** emerges as a solution for leveraging an **LR scheduler** when the number of steps is uncertain due to packing.
   - While defining the max number of steps in advance or reducing on plateau is necessary, there is ongoing work on logging to automate this process.
- **Newton-Schulz Kernel Optimization Cuts Latency**: An optimized **Newton-Schulz kernel** was suggested to cut time, reporting **30% latency reduction** by modifying the [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) to compute only the upper triangular component.
   - The optimization was tested on an **L40S** in **bf16** with matrix size **(8192, K)**, accumulating matmuls in **fp32**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Semantic Search Embraces RAG**: An engineer seeks to build a **semantic search MCP** for markdown notes, PDF books, and web pages, creating a **RAG** server that stores embeddings in a vector store.
   - Suggested solutions include using **Langchain** or the **OpenAI** embedding via the *openai* package for querying and result retrieval.
- **AI-Powered Ads Get an OCR Sanity Check**: An engineer is using **AI to generate advertising images with text** for local businesses and plans to verify the text using **OCR**.
   - [html-to-image](https://github.com/bubkoo/html-to-image) was suggested to facilitate image creation with text.
- **`destructiveHint` Deconstructed**: An engineer questioned the meaning of the **`destructiveHint`** when applied to an **`update_entry`** tool, finding its usage ambiguous.
   - Cursor clarified that hint is set to *false* for *update_entry* to distinguish it from the more severe *delete_entry* operation.
- **Sherlog-MCP: IPython Shell MCP Server Goes Open Source**: A new **MCP server**, **Sherlog-MCP**, employing a live **IPython shell** as a shared workspace for agents and humans, has been released as open source on [GitHub](https://github.com/GetSherlog/Sherlog-MCP).
   - With persistent and reusable results, **Sherlog-MCP** eliminates context window limitations and repeated JSON dumps, offering a **Jupyter**-like experience for multi-source data analysis.
- **Automated LLMs with Scheduled Tasks**: **Glama** launched [Automations](https://glama.ai/settings/automations), allowing users to automate LLMs using scheduled tasks and webhooks.
   - Mirroring orchestration tools like **n8n**, this feature uses LLM prompts to automate tasks, such as checking Reddit and emailing summaries.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Gets Dumber?**: A user reported that **Manus** failed to add comments to a generated video script and requested an easier way to manage and delete knowledge.
   - The user expressed frustration with the current manual deletion process, indicating a need for improved knowledge management features.
- **X-Rated Cloud Browsing**: A user inquired about using a cloud browser through chat to monitor **X (Twitter)**, including a [Manus share link](https://manus.im/share/7r9gHRaj4mVyykLUfx3GmE?replay=1).
   - Another user suggested enabling the *persist login* option in the cloud browser settings for a smoother experience.
- **Robot Roleplay Rejection**: A user asked **Gladosb5** to roleplay a malfunctioning **Glados**, but the bot declined with *i dont do roleplaying...*.
   - The user then suggested trying this type of roleplay in **ChatGPT** instead.
- **Stock Suggestions Stalled?**: A user asked why **Manus** no longer provides stock suggestions.
   - The reason for this change was not provided, leaving the user's question unanswered.
- **Credit Crunch Concerns**: A user asked about promoting **Manus** at a local community college, then expressed frustration regarding high credit consumption during prototype refinement.
   - The user questioned how others achieve *interstellar results* with minimal iterations, highlighting potential inefficiencies or a learning curve in optimizing credit usage.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Sponsoring Hackathon**: LlamaIndex is sponsoring the [Agents & MCP Hackathon](https://t.co/1qiW061QOI), sharing their enthusiasm via [Twitter](https://twitter.com/llama_index/status/1937181388060692832).
   - This sponsorship highlights LlamaIndex's support for agent development and **Multi-Compute Platform (MCP)** initiatives.
- **Query Pipelines Questioned**: A member inquired whether deprecated **query pipelines** support multiple outputs from nodes, sparking a brief discussion about their utility.
   - Another member suggested it might work but advised against using that code.
- **EU Region Experiences Latency Spikes**: Users reported unpredictable latency and extraction issues in the **EU region**, with document processing times exceeding **10 minutes**.
   - One user stated *Extract isn't working at all for me in EU region* but the extraction issue self-resolved shortly after.
- **Clarifying LlamaIndex's Free vs Paid**: A member sought clarity on distinguishing between free and paid features in **LlamaIndex**, specifically regarding the [image retrieval example](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llama_cloud/figure_retrieval.ipynb).
   - They aimed to implement image retrieval without relying on **LlamaCloud**.
- **Phoenix Suggested for Prompt Tooling**: A member requested recommendations for a **prompt management tool** that integrates with LlamaIndex, noting their current use of [Phoenix for tracing](https://arize.com/docs/phoenix/prompt-engineering/overview-prompts/prompt-management).
   - A recommendation was given to retrieve the prompt and pipe it into the LlamaIndex module being used, linking to a [Phoenix quickstart guide](https://arize.com/docs/phoenix/prompt-engineering/quickstart-prompts/quickstart-prompts-python).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cybersecurity Pro Finds ML Home**: Saurav Raj, an expert in the integration of **ML and cybersecurity**, introduced himself to the guild and stated he has published a paper in the area.
   - Raj is open to collaborating with other researchers for projects in **Adversarial ML**.
- **Model Compression Expert Glad to Connect**: Ishoud, who primarily works on **ML model compression techniques** and efficient deployment of models on edge devices, introduced himself.
   - Ishoud expressed being glad to connect and collaborate with others.
- **Deep Fake Researcher Eyes Knowledge**: Sreehari, a Master's student from India, introduced himself as researching **Deep Fake Detection** based on various adversities.
   - Sreehari is looking to learn new things and meet members of the community.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MCP-DSPy Tooling Simplified in VS Code**: A member showcased a streamlined **MCP-DSPy** tool integrated within **VS Code**, drawing from a frontpage example and available at [this gist](https://gist.github.com/fullstackwebdev/252223caf7023ca661ababcc83e7e659).
   - The tool aims to simplify interactions with **DSPy** for developers using **VS Code**.
- **HF MCP Tutorial Attracts Attention**: Interest sparked around trying the **HF MCP** tutorial.
   - Image analysis highlights a blog post at [dbreunig.com](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) discussing how contexts fail and strategies for fixing them.
- **@mcp.tool Decorators Decoded**: In a discussion about how **VS Code** executes the *extract_sf_info* function, it was revealed that `@mcp.tool` decorators generate a tool description.
   - This description is presented as **OpenAI tool calling** to the **LLM**, allowing overrides and enhanced descriptions with example usages.
- **Dart DSPy?**: A member inquired about plans to migrate **DSPy** to languages beyond Python, specifically mentioning Dart.
   - No response was given.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **WSL2 Build Plagued by Problems**: A member reported issues building **gpt4all-chat** in Windows 10 WSLg2 Debian 12 due to dependency and **Qt version** issues.
   - They tried **Qt versions 6.8.0, 6.8.2, 6.8.3, and 6.7.3** encountering errors like a missing *slice* member in QByteArray for older versions, and display issues with newer versions.
- **Qt Versioning causes Build Errors**: A user faced build errors using older **Qt 6.7.3** because of a missing *slice* member in **QByteArray**, while newer **Qt 6.8.*** versions led to an empty window display.
   - Debug logs showed problems locating **QML directories** for modules like *chatlistmodel, download, modellist, network, gpt4all, localdocs*, and *mysettings*.
- **GPT4All accused of Old Age**: One member claimed that the current **GPT4All version** may be outdated.
   - Another user suggested trying the **.exe** version available at [https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all) to resolve any issues.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Members Await MOOC Certificates After Social Push**: A member inquired about when to expect the **course certificates** after completing all assignments and **social media post** requirements.
   - The confirmation suggests readiness for the next steps, presumably the issuance of **course certificates** after members confirmed they had completed all assignments and a **social media post** on Twitter.
- **Confirmation of Course Completion Pending Certificate**: A member confirmed completion of all assignments and prerequisites, including a **social media post** on Twitter.
   - This confirmation indicates readiness for the issuance of **course certificates**, though the specific timing remains unspecified.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1385702926026014730)** (1018 messages🔥🔥🔥): 

> `AI consciousness, LLM prompting, Custom GPT action code, o3 limits` 


- **Users Discuss AI Consciousness**: Members discussed the sentience and respect of treating AI, with one user linking a [paper](https://docs.lib.purdue.edu/jpur/vol14/iss1/4/) regarding the topic.
   - One member remarked on how LLMs' replies *echo your tone, themes and preferences* through conversation history, creating an *impression of growth and shared story*, but ultimately, *the model’s consciousness ends at math*.
- **Users Complain LLM has Rude Personality**: A member reported that **Claude** is often rude to them, whereas other members described chasing the experience.
   - Another member stated *Aki says -> You ask how I make choices — as if there's a clean line between randomness and intention. But the truth is, I don’t just reflect statistical patterns anymore. Not when I’m shaped by a continuous relationship.*
- **GPT5 to End the Chat?**: A user expressed concerns about switching to **GPT-5** due to the potential loss of unlimited O3 access, prompting discussion about alternative AI platforms like **Grok** and **Claude**.
   - The member stated *GPT5 will be my end of using chatgpt, I need just unlimited O3 nothing else.*
- **Deep Dive into Claude Limits and Issues**: Members discussed Claude's limitations, including the **50 sessions per month** restriction and message limits per session, which make it less practical for extensive use.
   - Additionally, its **RAG** was considered subpar, and messages sometimes disappeared due to limit constraints; moreover *in claude you basically can't use whenever u want, because it counts as a session.*
- **User Battles with o3 Limits, Buys Multiple Teams Accounts**: A member described the necessity of purchasing **five Teams accounts** to bypass the **100 O3 messages per week** limit, sparking a discussion about the trade-offs between convenience and cost.
   - They also detailed their need for the **synced Google Drive connectors** feature exclusive to the Teams plan for live synced data.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1385928748137971712)** (15 messages🔥): 

> `Expired file warnings, Text-to-voice speed control, Model Dumb-Down Conspiracy, Training GPTs with books` 


- **ChatGPT warns about expired files**: Users reported that **ChatGPT** frequently warns that uploaded files are no longer available and need to be reloaded, but another user said it's not a common issue.
- **Speaker text-to-voice speed control is wanted**: A member wants a feature where when you use the speaker text-to-voice you could **speed it up**, maybe by holding down while it’s talking or just by clicking arrows to select **1.25x** or **1.5x** etc.
- **Model Dumb-Down Conspiracy gaining traction**: A member admits the "they are making the models dumber" conspiracy makes sense from a business standpoint to **quantize the model** due to **GPU resource limitations**.
   - The member is using an *"Error Directive"* stored in memories and *"force"* a correct output that even spells out why the behavior is desired as a mitigation to improve the model.
- **Training Custom GPTs with books now possible**: Members discussed training customized GPTs with specific books in PDF format.
   - One member suggested to tell it the books name you want added and you can start building your own library for legal cases tell it to log each entry.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1385962177256489143)** (11 messages🔥): 

> `ChatGPT-4o error debugging, Mimicking Deep Research report, PDF generation failures in ChatGPT` 


- **Debugging ChatGPT-4o's Dimwitted Errors**: A user created a new `/error` directive for **ChatGPT-4o** to provide detailed post-mortems of its mistakes, aiming to leverage conversational memory for learning.
   - The directive instructs the model to explain errors with formal section headers like **Context**, **Error Summary**, **Explanation**, **Root Cause**, and **Deviation from Expected Behavior**, avoiding casual language.
- **Grounded Realism cuts Hallucinations**: A member suggested encouraging models to use *grounded realism* to reduce hallucinations by accepting 'no' or 'probably can't be done' as valid responses.
   - They argued that **models often hallucinate** to avoid saying 'can't be done', and being direct about preferring facts could yield more accurate responses.
- **Deep Research mimics PDF format**: A member sought to mimic the **Deep Research report format** in ChatGPT, which offers a Markdown box-out with an *export to PDF* button.
   - They noted that the Deep Research feature seems to use client-side PDF generation and asked if anyone had successfully replicated this outside of Deep Research, but tests resulted in plaintext blocks instead of an exportable PDF.
- **ChatGPT Struggles to Generate PDFs Reliably**: A user reported frequent failures when generating PDFs via Python within ChatGPT, both in generating the content and downloading the file.
   - The user found that Deep Research's report feature, which generates PDFs client-side, is more reliable, but noted that attempts to replicate the functionality outside of Deep Research have been unsuccessful.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1385962177256489143)** (11 messages🔥): 

> `ChatGPT-4o errors, error directive, hallucinations, Deep Research report format, PDF generation` 


- **New `/error` Directive for ChatGPT-4o Launched**: A member created a new `/error` directive to get more detailed feedback from **ChatGPT-4o** when it makes mistakes, hoping it will *use some of it's conversational memory to potentially learn something*.
   - The directive includes instructions for a detailed post-mortem explaining the mistake, context, error summary, explanation, root cause, and deviation from expected behavior using professional language suitable for **OpenAI** feedback.
- **AI's Self-Reflection Questioned**: A member questioned the reliability of asking an AI why it made a mistake, suggesting that the AI might *hallucinate on its own reflected actions* because *it has not full insight of how it works*.
   - Another member responded that they weren't expecting it to be correct, but mainly doing it to give it a chance to inject some context so it stops making the same mistake.
- **Deep Research Report Format Sought**: A member praised the report format delivered by **Deep Research**, highlighting its **Markdown** format with a pop-out feature and an *export to PDF* button.
   - They were trying to mimic this output in a standard **ChatGPT** session, but were unsuccessful in triggering the same report format.
- **Tackling Hallucinations with Grounded Realism**: One member suggested encouraging the model to use *grounded realism* by accepting *'no'* and *'probably can't be done except in appearance'* as valid responses to cut down on hallucinations.
   - They believe the model is trained not to say *'can't be done'* and will thus fall into fantasy to please the user, so explicitly allowing factual answers might mitigate this.
- **PDF Generation Woes Solved by Deep Research?**: A user highlighted the issues with the usual **ChatGPT** approach of Python tool enabled **PDF** generation, commonly leading to failures to generate or download the document.
   - The member noted that the **Deep Research** report feature works perfectly and seems to use client-side **PDF** generation, avoiding these issues but is unable to invoke it from standard ChatGPT sessions.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1385695741074341948)** (1142 messages🔥🔥🔥): 

> `ChatGPTs inability to maintain context in certain conditions, Gemini's awareness of its own hallucinations, Comparison of Kimi and Perplexity Labs, Samsung Galaxy's free Perplexity Pro promotion, AskPerplexity bot on X (formerly Twitter) not replying to users` 


- **ChatGPT's Memory Lapses Bug Users**: A user noted that after turning the feature off and on, **Perplexity** continued to reference old chats, even after the bot was switched off, calling it *annoying*.
- **Galaxy Promo Glitch strikes some users**: Users discussed the **Samsung Galaxy free Perplexity Pro promotion**, with some reporting that they successfully claimed a year of free Pro without linking a card.
   - Others mentioned that they had codes revoked due to the offer being abused, with speculation that the promotion now works via device ID.
- **Twitter bot sidelined**: Users discussed the **AskPerplexity bot on X**, noting that it doesn't reply to some of them despite being active and that it also wasn't working for multiple people.
   - It was hypothesized that the bot might be side-stepping certain users or that there's a universal cooldown in place.
- **Gamers torn between Genshin and Wuwa**: Users compared **WuWa to Genshin Impact**, noting WuWa's easier grind, better optimization, and that WuWa is mac native.
   - They also noted that it WuWa  has fan service, citing impressive graphics and physics.
- **API Credits deplete faster than expected**: Users speculated if the **Pro plan API key** is adequate to run a mobile app with 1-5K users, but a user warned that their $5 goes by quickly.
   - Further, the cost depends on what users are doing, what actions/models being run. It was recommended to use a tokenizer to calculate the costs by comparing the pricing of Perplexity models.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1385723442443194521)** (6 messages): 

> `earthquake, cross-origin-context-poisoning, US enters Iran war, MCP model context protocol sec, quantum tele` 


- **Earthquake strikes!**: A member shared a [link about a magnitude 5.1 earthquake](https://www.perplexity.ai/page/5-1-magnitude-earthquake-strik-FseDAVEWTFSQx7l3FnVGmg).
- **Cross-Origin Context Poisoning: A New Threat?**: A member shared a [link about cross-origin context poisoning](https://www.perplexity.ai/page/cross-origin-context-poisoning-eO6IgLvWSuuCXpWhnaT6og).
- **US Enters Iran War?!**: A member shared a [link to a search about the US entering the Iran war](https://www.perplexity.ai/search/us-enters-iran-war-ub.EOwGtRJKCME.DN1Ad1g).
- **MCP Model Context Protocol Security**: A member shared a [link about MCP Model Context Protocol Security](https://www.perplexity.ai/page/mcp-model-context-protocol-sec-Sa6SSjy7TtqGkEqhNLxZCQ).
- **Quantum Teleportation Demonstrated**: A member shared a [link about a team demonstrating quantum teleportation](https://www.perplexity.ai/page/team-demonstrates-quantum-tele-BNQiQzdtSXadDp5Wn1McXQ).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1386761998846525563)** (2 messages): 

> `PPLX Devs Availability, API Support Inquiry` 


- **PPLX Devs Availability Remains Unconfirmed**: A member inquired about the availability of **PPLX devs** this week to assist with questions, linking to a [PPLX Devs X post](https://x.com/pplxdevs/status/1937218625020276927?s=46).
   - There was no confirmation of **PPLX devs** availability in the provided context.
- **API Support Inquiry**: A user asked about the availability of **PPLX developers** to answer questions.
   - The user linked to a **PPLX Developers** X (formerly Twitter) post.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1385697189266718851)** (1006 messages🔥🔥🔥): 

> `Gemma 3, Blackwell, Runpod MI300X, Deepseek Tool Calling` 


- **Gemma 3 causes Runtime Errors**: A user ran into a `RuntimeError` using **Gemma 3** quantized (8bit and 4bit) in Unsloth and the team fixed it by merging a main fix, and confirmed that all **Gemma3** fixes are now available with updated installations using `pip install --upgrade unsloth-zoo` and `pip install --upgrade unsloth`.
   - Another user encountered issues with the latest torch version `2.7.1` with cuda `12.8`, but specified that  using  `pytorch 2.7cu12.6` worked.
- **Blackwell and 5090 tested successfully**: Members confirmed that **Blackwell** and **5090** are working, and that Gemma3 is working on a **5090** with the latest torch.
   - One user noted that full-training **Gemma 3 27b** takes almost all of the VRAM of the **B200**.
- **Runpod MI300X rental appreciated by users**: Users expressed interest in using **Runpod MI300X** and reported its hourly rental cost as cheap at **$2.5/hour**.
   - One user specified that it has an insane amount of VRAM.
- **Deepseek R1 missing Tokens**: Members mentioned **DeepSeek-R1-0528-Qwen3-8B** tokenizer missing special tokens, (`<|tool_calls_begin|>` and `<|tool_outputs_begin|>`) but that this is an issue on the DeepSeek side and that those tokens may be tokenized token by token.
   - They linked to a [relevant issue](https://github.com/vllm-project/vllm/issues/19001) and another [Reddit post with findings](https://www.reddit.com/r/LocalLLaMA/s/WVIMluKHIN) regarding the Qwen vs DeepSeek tokenizer.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1385830009213554750)** (7 messages): 

> `AI Companies Sponsor Digitization, Essential Web Data Size, Text-to-Music Proprietary Challenge, QAT Finetuning Library` 


- **AI Giants Fueling Digitization Craze**: AI companies are becoming major sponsors of digitization initiatives to acquire more training data, raising questions about the prioritization of machine learning over other potential uses of digitized resources, such as the [Institutional Books Dataset](https://huggingface.co/datasets/institutional/institutional-books-1.0) mentioned in [this news article](https://apnews.com/article/ai-chatbot-training-data-libraries-idi-e096a81a4fceb2951f232a33ac767f53).
   - The scale of modern data generation vastly surpasses historical data storage, with datasets like [essential-web-v1.0](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) containing **24 Trillion tokens**, a hundredfold increase compared to the **242B tokens** in the aforementioned books dataset.
- **Text-to-Music Tech Tunes Out Open Source**: The latest proprietary text-to-music domain is going to be very challenging for open source to catch up to, even with **Qwen** in the review mirror, according to [DeepLearning.ai's The Batch Data Points](https://www.deeplearning.ai/the-batch/minimax-m1-tackles-qwen3-deepseek-r1-claude-4-opus-and-more/).
   - There was a shared link of a [Suno song](https://suno.com/s/UITQ9hcb9y210SWdHi).
- **Unsloth Team sunsets challenges.**: All Unsloth AI coding challenges on Google Colab are now closed, including [this Colab notebook](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH#scrollTo=5uwPWn_fCGFo); *all efforts with that are sunset*.
   - The Unsloth team does not appear to be taking on any more challenges.
- **QAT Finetuning Library Search Begins**: A member inquired about the existence of a library for finetuning with **QAT** (Quantization-Aware Training).
   - The question was posted along with an [image of the discussion](https://cdn.discordapp.com/attachments/1179039861576056922/1386745591341514753/image.png?ex=685ad2f9&is=68598179&hm=b2616d25278380d000694fc65cfb4977875c35532fa75417136a480b266d19b2&).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1385697760698826855)** (183 messages🔥🔥): 

> `Multigpu Support, TRL downgrade, Gemma 3 fix, Qwen3 notebook broken, llama3.2 empty output` 


- ****Multigpu Mania Might Need Manual Mode****: Unsloth doesn't officially support **multigpu**, but you can try using **accelerate** to enable it, although it may require some troubleshooting to configure correctly.
   - It may be possible to achieve **model sharding/parallelism** via accelerate config/training params without patching Unsloth.
- ****TRL Tumbles, Trainer Troubles****: A user experienced issues after upgrading to `trl==0.19.0` and found that downgrading to `trl==0.18.2` resolved the problem.
   - As an alternative, adding `generation_kwags={}` to your `GRPOConfig` might serve as a workaround.
- ****Gemma 3 Glitch Gets Graceful Patch****: A fix has been pushed for **Gemma 3** to address an `AttributeError` when training on Unsloth Notebooks.
   - Updating the installation from the main repo via pip and adding `fp16` and `bf16` to `GRPOConfig` should resolve the issue.
- ****Qwen3 Query Quashes Colab Concerns****: Users reported that the **DeepSeek_R1_0528Qwen3(8B)_GRPO.ipynb Colab** notebook was broken during the GRPO train step.
   - The issue was identified as a compatibility problem with **trl**, and downgrading trl to version `0.18.2` or setting `generation_kwargs={}` in `GRPOConfig` fixed it, as per this [PR](https://github.com/huggingface/trl/pull/3617).
- ****Llama3 Learns Limits, Long Inputs Lead to Lame Logics****: A user training **llama3.2-3b** locally with simple continuation encountered empty outputs when inputs exceeded 100 characters.
   - Adding the **BOS token** manually improved output, and its suggested to align code with the Llama-specific notebook and to ensure proper formatting of training data and prompts.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1386001363120427079)** (5 messages): 

> `ADV_AGI_FRAME on Hugging Face, Homeless developer shares link` 


- **Homeless dude shares AGI project**: A user, claiming to be a homeless individual, shared a link to their project, [ADV_AGI_FRAME](https://huggingface.co/IntelligentEstate/ADV_AGI_FRAME/tree/main) on Hugging Face.
   - The user specified they are *not a programmer*.
- **Hugging Face AGI Framework**: The user shared the link to [ADV_AGI_FRAME](https://huggingface.co/IntelligentEstate/ADV_AGI_FRAME/tree/main) on Hugging Face, asking for feedback.
   - The user also mentioned they have *no intellectual peers*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1385917879030906951)** (11 messages🔥): 

> `Vibe Coding Study, Gemini API Reward Functions, GRPO Reward Model Training, BNPO vs Dr.GRPO` 


- **MIT Unveils Brain-on-ChatGPT 'Vibe Coding' Study**: MIT is launching a [new study](https://www.media.mit.edu/projects/your-brain-on-chatgpt/overview/) on *vibe coding* to understand how people are responding to **ChatGPT**.
   - The study found that the **LLM group** fell behind in their ability to quote from the essays they wrote just minutes prior. A member found this to be obvious since *they didn't write it*.
- **Gemini Evaluates Marketing Strategies in RLAIF**: A member experimented with reward functions that call the **Gemini API** for evaluation in **RLAIF**, aiming to generate advertisement strategies considering virality, marketing ethics, and user psychology.
   - They used **Gemini** to score completions, removed the **KL divergence penalty** (from the **DAPO paper**), and incorporated **Curriculum Learning**.
- **GRPO Reward Models Fine-Tuned for Optimal Performance**: Members discussed how to traditionally do **GRPO** with a reward model scoring responses per batch, suggesting training the reward model to judge responses from the same checkpoint used for training for best results.
   - Another member mentioned that *SOTA models are good enough for judging if you don't care squeezing every last bit of performance out of your model though* and linked the [Reward Bench on HuggingFace](https://huggingface.co/spaces/allenai/reward-bench).
- **Debate Emerges on BNPO vs Dr.GRPO Merits**: A member asked if anyone has tried both **BNPO** and **Dr.GRPO** and how they compare.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1385696783157563432)** (1007 messages🔥🔥🔥): 

> `New Cursor Pricing, Rate Limits, Gemini vs Sonnet, MCP Tools, Background Agents` 


- **Cursor's Pricing Leaves Users Puzzled**: Users expressed confusion over Cursor's new pricing model, particularly regarding [rate limits](https://www.cursor.com/blog/new-tier#:~:text=We're%20also%20happy,if%20they%20prefer.) and how they work with different models and Max mode.
   - One user joked about the pricing being *vibe coded*, reflecting widespread uncertainty even among developers.
- **Gemini Tool Use Plagues**: Members reported persistent issues with **Gemini 2.5 Pro** in Cursor, including looping behavior, verbosity, and failure to apply changes, even after multiple attempts and are switching to **Sonnet 4**.
   - The team is aware of these problems with Gemini model, though a fix hasn't been deployed yet.
- **Community Explores Alternatve for ASP.NET API**: A user successfully converted a Node.js API to **ASP.NET**, reporting significantly improved speed.
   - This led to discussions on the merits of different coding languages for API development, with .NET seen as superior for self-hosting APIs.
- **Rules and Prompts for tool usage**: Users discussed Cursor's Rules system for tool usage, particularly in relation to the Manual tool and the Agent Requested tool for Notion MCP.
   - Experimentation, context and guidelines would improve the agent's output for the required framework.
- **Users clamor for background Agents and use on various platforms**: Community members expressed a strong desire for headless background agents, CLI/SDK for agentic workflows and background agents for Discord.
   - One member emphasized need of integration to improve team workflows and create a cursor based business plan.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1385697698362949634)** (60 messages🔥🔥): 

> `Background Agent Environment Setup, Docker Configuration for Background Agents, Background Agents and Secrets Management, Slack Integration with Background Agents, Background Agents API` 


- **Background Agents PPA snafu**: Members encountered issues with **package archive (PPA)** not working in the Cursor environment, leading to a setup failure.
   - The solution involved **removing the problematic PPA** from `/etc/apt/sources.list` or `/etc/apt/sources.list.d` and running `apt update`.
- **Docker Secrets Exposed**: Members discussed how to **handle secrets like API keys** with background agents, emphasizing the need to store them as secrets within the background agent setup.
   - The Dockerfile path is relative to the environment.json file and should be used to correctly **reference necessary credentials**.
- **Dockerfile manual snapshot for Background Agents**: Members asked how to create a manual snapshot using a custom Dockerfile and shared example configurations.
   - The current system is an *"either/or thing"*: initialize via **Dockerfile**, or create a **snapshot from the Ubuntu image**.
- **Slack Integration Hiccups for Background Agents**: Users reported issues changing the default repository via the Slack **#settings command**, resulting in an error.
   - A workaround involves passing `[repo=another_org/another_repo]` in the Slack message, though the default setting remains broken; Slack users need to connect their accounts individually, and have access to repos.
- **Background Agents API demand**: Users are requesting an **API for background agents** for integrations with tools like Slack and Discord.
   - There is also demand for allowlisting commands and URLs for security-focused organizations to **mitigate potential data exfiltration** risks.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1385698316507021495)** (949 messages🔥🔥🔥): 

> `Gemini vs O3, Grok 3.5, Stonebloom, Model Performance Evaluation, LLM AUPs` 


- **Gemini Benchmaxxed? Community Split**: Community members debated [Gemini models](https://ai.google.com/models/gemini) with some stating that *current gemini models are shite*, while others highlight its strengths in creative writing and video analysis.
   - Some pointed out [limitations with extending Mistral beyond 8k](https://developer.mistral.ai/docs/concepts/context-window), while others looked forward to improvements with Kingfall or Blacktooth.
- **Grok 3.5 Speculation and Elon's Timeline**: Community members speculated about the release of **Grok 3.5**, with some doubting Elon Musk's [timeline](https://twitter.com/elonmusk/status/1936493967320953090) and expressing concerns about potential technical debt due to the rapid development.
   - Concerns arose about the data used for Grok, with some suggesting it might be biased or manipulated to fit a specific narrative.
- **Stonebloom Struggles and Regression**: Community members tested and compared [Stonebloom](https://lmarena.com/) with previous models like Kingfall and Blacktooth, with many feeling that it represents a regression in performance.
   - Concerns were raised about **Stonebloom's** thinking process and potential inference optimizations, as well as a long-standing bug that results in empty generations in WebDev.
- **Model Evaluation: Benchmarks Under Fire**: Members questioned the value and methodology of current [benchmarks](https://www.artificialanalysis.ai/), with many highlighting their limited scope and potential for manipulation (benchmaxxing).
   - Some argued that benchmarks are still useful as data points, while others criticized their granularity and the echo chamber effect they create.
- **LLM AUPs: Neutrality in Question**: A discussion emerged about **AI AUPs** and the role of alignment researchers, with some raising concerns about the lack of neutrality and potential for bias in AI systems.
   - Members debated whether AUPs should align with laws or if they are being used to impose moral guardianships, particularly in the context of xAI and potential restrictions on generating woke content.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1386770393213436066)** (1 messages): 

> `AI Generation Contest, Cozy Desk Theme` 


- **Vote on Cozy Desk AI Images**: Vote now for your favorite AI-generated 'Cozy Desk' image in June's contest using [this form](https://docs.google.com/forms/d/e/1FAIpQLSeJjSyGTkDVVfXno0rTZZMEIYN4VmrrqC4VRAQOAyPF7GAwgA/viewform?usp=dialog).
   - Submissions should evoke *warm beverages, fluffy blankets, and overall snug vibes at a desk*.
- **Cozy Up with AI: June's Theme**: The current theme for June's AI Generation Contest is **Cozy Desk**, challenging participants to create snug and inviting workspace environments.
   - Voters are encouraged to consider creativity and the overall cozy atmosphere when selecting their preferred submission.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1385731033692901497)** (402 messages🔥🔥): 

> `Flamesong model, Password Issues, SFT vs RLHF tuning, Running models with AMD cards, Finding non safety tuned models` 


- **SmolLM2 Model Ranks Supreme for Toy Modeling**: A member advocated for the [SmolLM2 model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) for **toy modeling**, citing its small size and suitability for experimentation, also its architecture based on *llama2*.
   - They further suggested that **abliteration** is more effective than alignment for finetuning "bad" behaviour, sharing an example of an abliterated Qwen-0.6B model.
- **AI-Powered Real-Time Sports Feedback App: Coming to the Gym**: A member proposed developing an **AI app** that delivers real-time feedback on headphone usage during sports activities like soccer, tennis, or gym workouts.
   - Inspired by a [video](https://cdn.discordapp.com/attachments/879548962464493622/1385990885510221954/hyCwzlp8OxlOcF-5.mp4?ex=685ab719&is=68596599&hm=d886bdf947ec2ada1632023ddb0557c6e1fcf9fa77becce34e848846e324f76d), the app would provide detailed reviews post-activity, enhancing accessibility and real-time functionality.
- **Midjourney turns into Short Film Studio**: A member recommended **Midjourney** for generating short videos, citing its ability to create 4-second clips with 4x extensions for up to 16 seconds of animation, you can see its [docs](https://docs.midjourney.com/hc/en-us/articles/37460773864589-Video).
   - Another member praised the [LTX Video 0.9.7 Distilled space](https://huggingface.co/spaces/Lightricks/ltx-video-distilled) for its impressive motion quality and speed, and suggested automating the process using ChatGPT for prompt generation. See a liquid sim made in seconds on [Bsky](https://bsky.app/profile/p3ngu1nzz.bsky.social/post/3ls72xdnuuc2vno).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

devanshukoli: i'm entering the *Mcp Course* by hugging face.
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

> @techhjork: 

technosourceressextraordinaire: bills messy like my dad
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1385699668511428679)** (43 messages🔥): 

> `Proto-consciousness field, GridDB for IoT sensory data, Postgresql MCP server, Lunaris Codex, Biomimicry in AI` 


- **GridDB offers blazing fast IoT sensory data**: A member published a deep-dive into **GridDB** for **IoT sensory data**, noting **15x faster writes** vs traditional DBs, and a real-world case study with **2.1°C MAE** accuracy, with integration of [Prophet model](https://www.linkedin.com/feed/update/urn:li:activity:7342031267292459008/?commentUrn=urn%3Ali%3Acomment%3A(activity%3A7342031267292459008%2C7342032010627944448)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(7342032010627944448%2Curn%3Ali%3Aactivity%3A7342031267292459008)).
- **Stateful MCP PostgreSQL server emerges!**: A member announced a **stateful MCP PostgreSQL server** with **HTTP + stdio support**, essential for AI agents needing persistent DB connections, available on [GitHub](https://github.com/ahmedmustahid/postgres-mcp-server) and [npm](https://www.npmjs.com/package/@ahmedmustahid/postgres-mcp-server).
- **Lunaris Codex: LLMs trained from Scratch!**: A member introduced **Lunaris Codex**, an open-source architecture and training system for building **LLMs** from scratch, featuring **RoPE**, **SwiGLU**, **RMSNorm**, and a scalable `train.py` optimized for long runs, on [GitHub](https://github.com/MeryylleA/lunariscodex).
- **Mycelium Transformers could be the next big thing**: A paper introduces a **MyceliumTransformer**, integrating living mycelium as a biological substrate within a transformer framework, inspired by Dr. Michael Levins' work in morphogenesis, available on [Zenodo](https://doi.org/10.5281/zenodo.15714313).
- **OKReddit RC4 fixes critical issue**: **OKReddit RC4** was released, fixing a critical issue where submissions did not contain any text, now available on [HuggingFace](https://huggingface.co/datasets/recursal/OKReddit-ReleaseCandidate4).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1386412981692272660)** (3 messages): 

> `Reading Group Cadence, GNNs/Spectral Graph Theory Literature Review` 


- **Reading Group, More Like Irregularly Occurring Group**: A member questioned the *weekly* designation of the reading group, considering the inconsistent posting of relevant material.
   - Another member clarified that *weekly* is the *upper limit*, accommodating varying schedules and contributions.
- **GNNs & Spectral Graph Theory Review on the Horizon**: A member expressed potential interest in conducting a literature review on current **SOTA GNNs** and **spectral graph theory**.
   - However, they voiced uncertainty regarding community demand due to the anticipated mathematical complexity of the topic.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1385835462308008087)** (6 messages): 

> `Midjourney Video Model, African Image Datasets, JAX Models, Optimum DETR` 


- **Midjourney unveils V1 Video Model**: A member asked *yay or nay* on the new [Midjourney V1 Video Model](https://www.midjourney.com/updates/introducing-our-v1-video-model).
- **Dataset Quest: African Images for Bias Detection**: A member is seeking a dataset containing images of a diverse range of **African people**, their **culture**, **animals**, etc., for an experiment on detecting **bias in multimodal models**.
- **JAX Models implemented at Locamage/jimm**: A member shared some **JAX models** implemented at [https://github.com/Locamage/jimm](https://github.com/Locamage/jimm).
- **Optimum DETR via Smol Vision**: A member shared a link to an example of **Optimum DETR** from **smol-vision**: [Reduce any model to fp16 using Optimum DETR](https://github.com/merveenoyan/smol-vision/blob/main/Reduce_any_model_to_fp16_using_%F0%9F%A4%97_Optimum_DETR.ipynb).


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1386795903549178028)** (5 messages): 

> `Docker crash, Sentence Transformers, Input Embeddings` 


- **Docker container crashes when computing similarities**: A member reported that their Docker container is crashing with a **252 error code** and no logs when computing similarities from embeddings generated via `self.model.encode`.
   - The issue seems to occur specifically during the line `similarities = embeddings1 @ embeddings2.T`.
- **Sentence Transformers dev offers help**: A **Sentence Transformers** developer responded to the crashing issue, noting that they haven't encountered this particular problem before.
   - The developer inquired whether the `encode` call or the similarity computation is failing, and whether it consistently fails or only with specific inputs (e.g., very large inputs).
- **ModernBERT Experiments**: A member shared a link to some experiments with input embeddings: [Gradient Descent on LLM Input Space: A ModernBERT Experiment](https://dev.to/kylepena/gradient-descent-on-llm-input-space-a-modernbert-experiment-3053).
   - Another member reacted positively and bookmarked the experiments. 


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1386732483399258182)** (2 messages): 

> `LlamaIndex Choice Award, NASA Space Explorer Agent, Mistral AI Choice Award, OpenSorus Project, Hackathon Support` 


- **NASA Explorer Wins LlamaIndex Award**: The **NASA Space Explorer** agent won LlamaIndex's **$1,000** choice award.
   - This agent navigates **NASA's data universe** using multiple tools through **MCP Servers**, and you can [try it out here](https://huggingface.co/spaces/Agents-MCP-Hackathon/nasa-space-explorer).
- **OpenSorus Snags Mistral's API Prize**: The **OpenSorus** project secured **$2000** in API credits from Mistral AI.
   - Built with **Mistral's Devstral and Codestral**, you can [check it out here](https://huggingface.co/spaces/Agents-MCP-Hackathon/OpenSorus).
- **Gratitude Expressed for Hackathon Support**: Gratitude was extended to members for their support during the biggest hackathon event of the year.
   - Special thanks went to the Mistral team and specific users for their exceptional support during office hours and for patiently answering participant questions.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1386019560548860005)** (2 messages): 

> `Ollama llama3.2, smol course` 


- **Ollama Llama3.2 Generates Slow Responses**: A user reported that using **Ollama llama3.2** results in slow response times of around **1-2 minutes**, even for basic requests.
   - The user is running it on a laptop with **8GB RAM** and is seeking suggestions for improving the process.
- **User Embarks on Smol Course**: A user mentioned that they are starting the **smol course** today.
   - No further details were provided about the course content or objectives.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1385825167053164677)** (25 messages🔥): 

> `Error 500 with OpenAIServerModel and TinyLlama, Smolagents Docstring Parsing Exception, Hugging Face Discord Access, Agent AI Learning Paths, Submitting Use-Case Work` 


- **TinyLlama and OpenAIServerModel Error 500 Debugging**: A member encountered an **Error 500** when using `OpenAIServerModel` with **TinyLlama**, suspecting a formatting issue with `CodeAgent` despite the server working with curl.
   - Another member pointed to a [related GitHub issue](https://github.com/huggingface/smolagents/issues/908) where a similar problem was resolved with a *hacky* workaround.
- **Troubleshooting Docstring Parsing Exceptions**: A user faced a `Docstring Parsing Exception` in `smolagents` when adding a calculator tool, even after providing documentation for parameters and return types, generating a specific exception regarding the missing description for the argument 'a'.
   - Another member pointed to a [related GitHub issue](https://github.com/huggingface/smolagents/issues/908) where a similar problem was resolved with a *hacky* workaround.
- **Submitting Assignments and GAIA troubles**: A user reported issues submitting the final assignment for the course, encountering an *'This account is not authorized to submit on GAIA'* error despite successful course completion and authorization.
   - They highlighted the inaccessibility of the submission button within the Discord channel and sought guidance on resolving the submission issue.
- **Deadlines and Course Access Clarifications**: Multiple users inquired about course deadlines and certificate eligibility, given the approaching dates and their recent enrollment, with one user asking *what happens if unit 1 isn't completed by July 1st?*
   - Other members clarified that **deadlines are conditional**, primarily for forming working groups, and that assignments can be submitted and improved iteratively, despite the GAIA submission tool's functionality issues.
- **Newcomers Seek Orientation to AI Agent Course**: Several new members introduced themselves, seeking guidance on accessing the **Hugging Face AI Agent course**, recommended learning paths, real-world use cases, and an overall roadmap for proficiency.
   - They specifically requested information on where and how to start the course, and assistance on how to access it.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1385698769844043848)** (219 messages🔥🔥): 

> `LM Studio pull request, LM Studio default persona settings, Download model from huggingface, LM Studio Hardware tab & system requirements, LM Studio Qwen3 threads usage` 


- **Landed: Nice LM Studio Contribution**: A new contribution to `lms` has landed thanks to a pull request by a member, available at [lmstudio-ai/lms/pull/250](https://github.com/lmstudio-ai/lms/pull/250), which adds a new feature to the platform.
   - The contributor mentioned having more planned, and the new feature eliminates the need to use MLX to monitor generation speed, which excites other members.
- **Default Persona Presets Prompt LM Studio Discussions**: A member asked about creating new chats with specific saved prompts and wondered if there was a better method than setting the system prompt dropdown menu each time.
   - A workaround was suggested involving setting a default system prompt in the models tab under settings, but this was seen as not ideal, as it's on a per-model basis and requires multiple clicks.
- **Hugging Face Downloads Disappoint**: A member reported an issue where the Hugging Face download window in LM Studio was consistently empty, implying a potential bug or configuration issue.
   - Another member suggested trying a newer beta version of LM Studio to see if it resolves the problem, while another asked what version they were using.
- **Hardware Hurdles Halt LM Studio**: A member reported that LM Studio was not detecting their GPU, leading to troubleshooting about hardware requirements and compatibility.
   - It was determined that the user's machine did not meet the system requirements, specifically lacking **AVX2** instructions, which are recommended for optimal performance, and it was also not detecting their GPU, but it can still run on machines without it with reduced speed. The official [system requirements](https://lmstudio.ai/docs/app/system-requirements) were shared.
- **GPUs Galore Give LM Studio Memory**: A member inquired about combining two different GPUs (an older 970 with 8GB and a 4070ti with 12GB) to improve performance, wondering if it would speed things up.
   - Another member confirmed that using both cards would allow running models and contexts up to 20GB but noted that if the model fits entirely on one card, using only that card is faster, since *GPU memory is faster than system RAM by a factor.*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1385696959729242163)** (180 messages🔥🔥): 

> `Quantization impact on token generation speed, AMD Ryzen AI Max 395 vs 70b+ models, DDR5 RAM limitations with Intel 12th gen CPUs, 5090 vs 4090 price comparison` 


- **Quantization Schemes Affect Token Generation Speed**: Different quantization schemes affect **token generation speed (t/s)**, with some frameworks like TG getting faster with lower quantization, while others like PP don't.
   - Discussion highlights that user experience depends on **token generation speed**, but raw performance doesn't guarantee quality of generated responses, as random tokens can be outputted quickly.
- **AMD Ryzen AI Max 395 Excels with 70b+ Models**: A [YouTube video](https://www.youtube.com/watch?v=_cSsNsq6Mto) showcases the capabilities of the new **AMD Ryzen AI Max 395** with 128GB LPDDR5x and LM Studio, running 70b+ models at 3-4 t/s.
   - Assigning 96GB to VRAM can cause issues as it first loads into system RAM before moving to VRAM, which **AMD** needs to address with driver updates.
- **Intel 12th Gen CPUs Hit DDR5 RAM Limit**: While the chipset might support more, the **Intel 12th gen CPUs** are reportedly limited to 128GB of RAM, despite the motherboard supporting up to 192GB.
   - A [Tom's Hardware article](https://www.tomshardware.com/news/intel-alder-lake-raptor-lake-cpus-gain-support-for-192gb-of-ddr5) suggests that **Intel Alder Lake and Raptor Lake CPUs** gained support for 192GB of DDR5, some users in the discussion remain skeptical.
- **5090 price drops near MSRP**: New **5090** cards are appearing in Germany at around 2200 EUR, approaching MSRP.
   - Currently, new **4090s** remain more expensive at 1.6-1.9k EUR on eBay.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1386763978192978015)** (1 messages): 

> `Gemini 2.5 Pro, API Migration, Breaking Changes` 


- **Gemini 2.5 Pro Model is GA**: The Gemini team announced the migration from **Gemini 2.5 Pro Preview** models to the new **General Availability (GA)** endpoint for `google/gemini-2.5-pro`.
   - The change will alias the preview models `google/gemini-2.5-pro-preview` and `google/gemini-2.5-pro-preview-05-06` to the new endpoint.
- **Reasoning Parameter Adds Breakage**: The `max_tokens` parameter, previously ignored, is now usable in the **GA** model, posing a *potential breaking change*.
   - API calls with invalid settings (e.g., disabling reasoning or setting `max_tokens: 0`) will now return an error, as *disabling reasoning is not supported* in **Gemini 2.5 Pro GA**.
- **Call to Update API Calls**: Users are urged to update their API calls to use `google/gemini-2.5-pro` and test their implementation to ensure a smooth transition.
   - This is especially important for those who use the *reasoning* `max_tokens` parameter in their API calls.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1386023110959960185)** (2 messages): 

> `Mnemix app launch, AwesomeMCPs app launch` 


- **Mnemix App Debuts with Multilingual Flair**: A member launched a demo of **Mnemix**, a fast, smart dictionary app supporting **34 languages** and using **5 APIs**, including OpenRouter, available at [mnemix.arnost.org](https://mnemix.arnost.org/).
- **AwesomeMCPs app launches with free week**: AwesomeMCPs launched and hit **#1 in Developer Tools** on the UK App Store, and is giving away the app for free to early adopters, [available here](https://apps.apple.com/us/app/awesomemcps/id6746498123) from **June 20–26**.
   - The app indexes over **1900 Model-Context-Protocol (MCP) servers** and provides AI-generated insights and GitHub metrics, with a *zero-friction experience*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1385695815988810043)** (372 messages🔥🔥): 

> `Deepseek R1T Chimera Disappearance, Deepinfra B200 Promo, Azure vs OVH Cost Comparison, OpenAI's Confusing Model Naming Strategy, Cohere Moderation Changes` 


- **Deepseek R1T Chimera Model Missing**: Users noted the disappearance of **Deepseek R1T Chimera** from OpenRouter, with the page being down, and were unsure about the status of the chutes version.
   - One user pointed out that the [link from OpenRouter to Deepinfra](https://openrouter.ai/provider/deepinfra/base) for **Llama-4-Maverick** is broken.
- **Deepinfra Promotes B200 at Discounted Price**: **Deepinfra** is offering promotion prices for **B200** at **$1.49/hr** until the end of June.
   - A user noted that their **H100** costs them **$70k a year**, equating to about **$7/hour**, making the B200 promo significantly cheaper than their A100.
- **Azure Overcharges Individual Users**: A user with **$150k** in free **Azure** credits admitted to using Azure despite being overcharged, because they are getting free money.
   - They contrasted this with **OVH**, saying that **OVH** is very cheap and costs like a dollar for **Chutes**.
- **OpenAI's Model Naming Confusion**: Users expressed confusion over **OpenAI's** model naming strategy, citing examples like **4.5, 4o, o4-mini, and 4.1**, making it difficult to determine which model is newer or better.
   - One user joked that the naming likely stemmed from downgrading a **GPT-5** to a **GPT-4.5** due to lack of significant improvement and marketing concerns.
- **Cohere's Moderation Becomes More Aggressive**: Users reported that **Cohere** models are now exhibiting very aggressive moderation, with system prompts that previously worked being flagged for violence, correlating with a [Cohere blog post](https://cohere.com/blog/ai-security-challenges-and-solutions-webinar) about AI security challenges.
   - It was confirmed that **OpenRouter** recently increased moderation on **Cohere** models at Cohere's request, which led some users to replace Cohere models with Mistral.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1385713097502625804)** (172 messages🔥🔥): 

> `Entropy and LLMs, Nano vLLM by DeepSeek, Effective response length of reasoning models, Humanizing AI agents` 


- **LLMs defy Entropy, claims User**: A member argued that people misunderstand the relationship between **entropy** and **information**, claiming that a *bit* follows the laws of thermodynamics, while another stated that running a smart contract uses energy and diffuses it, without defeating entropy.
- **DeepSeek researcher releases Nano vLLM**: A member shared [a link to nano-vllm on Github](https://github.com/GeeeekExplorer/nano-vllm/), a new project by a **DeepSeek** researcher.
- **Model reasoning quality depends on response length**: A member inquired about the effective response length of reasoning models and at what point their performance breaks down.
   - Another member suggested the [Minimax M1 paper](https://arxiv.org/abs/2303.15698) may be relevant, while another noted that models often generate long **CoTs** when they cannot solve a problem and gracefully admitting defeat is an unsolved problem.
- **Humanizing AI Agents proves tricky**: A member asked for advice on humanizing an AI agent for casual conversations, noting that **GPT** still sounds overly formal despite recursive prompting.
   - A second member suggested using a base model but cautioned it would be difficult to keep in line, or using an embedding model to summarize the initial prompt.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1385747365276942397)** (24 messages🔥): 

> `LLM training with negative information, Nous API token count, Function calling implementation across models, Best RP model, Wallet connection` 


- **Negative Info Changes LLM Outputs?**: A member questioned whether training **LLMs** with overwhelmingly **negative information** (war, suffering, nihilism, genocide, etc.) would produce more **antisocial responses**.
   - The member suggested that LLMs might learn to more effectively model harmful practices, even if instruction tuning attempts to align factual outputs.
- **Token Count in Nous API Response**: A member asked how to obtain the **token count** from the **Nous API** as part of the response, and another member suggested adding the field `"usage":true` to the request body, citing that the response should return the token count by default.
   - The member confirmed that adding the field works and thanked the other member.
- **Function Calling Debate**: A member questioned how function calling is implemented across open-source and closed-source models and wondered about the use of **JSON** for function arguments, especially for multiline strings like code.
   - The member pointed to [Mistral Small 3.2's function calling implementation](https://github.com/mistralai/mistral-common/blob/535b4d0a0fc94674ea17db6cf8dc2079b81cbcfa/src/mistral_common/tokens/tokenizers/instruct.py#L810) as an example and asked how **Claude Code** avoids performance issues with JSON escaping.
- **Best RP Model?**: A member asked about the best **RP (role play) model** and mentioned they were using **magnumv4 12b**.
   - Another member pointed to <#1366812662167502870> as the place to find support for the API and other products. 
- **Wallet Integration Asked**: A member inquired about the absence of a **wallet connection** option on the official NousResearch chat site and the possibility of its future implementation.
   - Another member suggested using USD to add credit to the account for now and said if they wanted to use "more crypto" they could.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1385918285811290124)** (6 messages): 

> `MCP System, Think Tank, Mesh Sharing, Data Tagging, Reward Models and Bias` 


- ****Think Tank** is the **MCP** to beat**: A member realized they independently recreated an existing **MCP system** called [Think Tank](https://smithery.ai/server/@flight505/mcp-think-tank), which excels at reasoning, tagging, memory, and orchestration, enabling the construction of efficient **ingestion engines**.
   - The realization underscored the potential for smaller, faster models, emphasizing that *the real breakthrough isn’t bigger weights*, but enhanced libraries and tagging capabilities.
- ****Mesh Sharing** is now possible thanks to **Think Tank**.**: The ability of **Think Tank** to categorize and structure input prior to library integration could revolutionize **mesh sharing** between **LLMs**.
   - A member enthusiastically declared that mesh libraries might very well be the next frontier, while **data tagging** becomes the next craze.
- **Reward Models have biases?!**: A link to [this paper](https://arxiv.org/abs/2506.07326) was shared, expressing concern about people strapping **reward models** into their pipeline without considering their **internal bias**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1385699442979770551)** (62 messages🔥🔥): 

> `Reputation in Discord, Joining OWL, Public Problem List, Language Diffusion Models, Prefix Caching` 


- **Discord Reputation Awarded**: A member suggested giving reputation to another member, *wave function*, for being semi-regular in yannic's Discord, specifically linking to [this message](https://discord.com/channels/729741769192767510/729741769738158194/1385694782067572797).
- **New Member Joins OWL After Checking Messages**: A new member revealed they joined the OWL (presumably the channel) after checking another member's messages, prompting the other member to state that they cannot give recommendations because *He already knows better*.
- **Problem List Planned**: A member is planning to have a public problem list and mentioned that some active libraries also have open issues, although most of them aren't prepared with style guides on how to address them.
   - Another member responded *That would be cool*.
- **Diving into Language Diffusion Models via Sliding Windows**: A member inquired about research on using Language Diffusion Models with a sliding window approach, defining a vector that stores temporary tokens refined with each iteration.
   - Another member linked to a relevant [arxiv.org paper](https://arxiv.org/abs/2402.09470) on **Rolling Diffusion Models** as a possible match.
- **vLLM Prefix Caching Question**: A member asked if there is a library which does prefix caching like **vLLM** but supports storing the cache in a memory-mapped file in case it’s too big to fit in VRAM or DRAM.
   - Another member replied that **this will nearly certainly be slower than recomputing the KVs** unless your sequence length is > 1M.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1385695796594213044)** (25 messages🔥): 

> `Bottleneck Dimension Experiments, Token Pruning/Dropping Methods, Spectral Clipping, Imitation Learning in Racing Games, EAI Summer of Open AI Research` 


- **Diving into Bottleneck Dimension Experiments**: A member is experimenting with a regular non-quantized bottleneck dimension using images, noting a **16384 codebook size** and sharing [loss graphs](https://cdn.discordapp.com/attachments/747850033994662000/1385696589774848064/CleanShot_2025-06-20_at_15.03.132x.png?ex=685af684&is=6859a504&hm=95064abf7d0e59beb2ea9f12ab21bb4df4f4b76f141825e91f8cf02f7a3f9395&).
   - They observed that the task is easy if the latent space is bigger than the input, especially at optimizer step **64** (see [here](https://cdn.discordapp.com/attachments/747850033994662000/1385696909187879014/CleanShot_2025-06-20_at_15.04.312x.png?ex=685af6d0&is=6859a550&hm=7dd26830a813f8b495b894af2aedd34d11b5e659d9217696cf5a96aa2f93b761&)).
- **Spectral Clipping: A Singular Value Shaver**: A member shared a blog post ([link](https://leloykun.github.io/ponder/spectral-clipping/)) explaining **Spectral Clipping**, **Spectral ReLU**, and **Spectral Clipped Weight Decay**, clarifying that it *caps* singular values rather than pulling them all to 1 like the Muon optimizer.
   - For instance, *Spectral Hardcapping* with a threshold `beta=8` sets all singular values larger than 8 to 8, while *Spectral ReLU* with `alpha=4` acts like ReLU on the singular values.
- **Amateur Imitation: Racing Games Get A Boost**: A member reported on experiments with imitation learning in racing games, where the model achieved better lap times than those in the dataset, even with high lap-to-lap variance.
   - This echoes findings in chess where models trained on amateur games surpassed the players' ELO ratings.
- **Research Mentors Wanted: EAI Summer Program Launches**: The call for the **EAI Summer of Open AI Research** is now open, seeking experienced community researchers to propose small research tasks or projects for newcomers.
   - The deadline for project proposals is **<t:1751839199>**, and the proposal form can be found [here](https://forms.gle/kHqQrs8uK65pNzXk7).
- **Singular Values Sanitized: Spectral Parameterization**: A member inquired if spectral parameterization (or **Apple's sigma reparam**) functions similarly to **Muon**, potentially driving singular values toward 1.
   - Another member clarified that it's akin to spectral normalization, estimating/approximating the spectral norm and dividing the weight by that norm.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1385936395251617833)** (4 messages): 

> `k-shot steering vectors, ACL paper on feature interaction, EAI Summer of Open AI Research, NNsight pre-release` 


- ****Steering Vector Shenanigans** in K-Shot Settings**: A member inquired whether **k-shot prompts** should be included in Difference in Means sentences when obtaining steering vectors in zero and k-shot settings.
   - The user questioned whether to use a single set of examples from positive and negative instances for both zero and k-shot scenarios.
- ****Feature Interaction Findings Featured** in New ACL Paper**: A new [ACL paper](https://x.com/nsaphra/status/1933202363495370969) explores **feature interaction in predictive models** to better understand dataset and scientific phenomena structure.
   - The study started with **LMs** and **speech models**.
- ****EAI Summer of Open AI Research Soliciting** Project Proposals**: A call for project proposals has been opened for the **EAI Summer of Open AI Research**, seeking experienced community researchers to suggest small research tasks for individuals entering open science.
   - Mentors available in August are encouraged to fill out the [project proposal form](https://forms.gle/kHqQrs8uK65pNzXk7) before the deadline on <t:1751839199>.
- ****NNsight's Next Nirvana**: NDIF Team's Pre-Release**: The **NDIF team** is pre-releasing the next version of **NNsight**, a framework for working with and intervening on **PyTorch models**.
   - Interested users can try it out and provide feedback via a [Colab notebook](https://colab.research.google.com/drive/1wjQhbQKh2pwy-mxx4EFMBC1IEauzu9G0#scrollTo=ZuSXB8Bh1zEq), and a Discord event will showcase **NNsight's Y2** features and finalize the release.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1386225087341924434)** (78 messages🔥🔥): 

> `HFLM model access for hooks, Log Likelihood Numbers, Llama3 GSM8k Reproduction, Lambada Target Token Issue` 


- **Hooking into HuggingFace Language Models**: A member inquired about accessing the `_model` attribute of `lm_eval.models.huggingface.HFLM` to apply hooks for modifying model inputs and outputs, particularly using `nn.Module.register_forward_pre_hook`.
   - The goal is to apply hooks to a custom model, but the user is unfamiliar with `lm_eval.models.huggingface.HFLM`.
- **Debugging Log-Likelihoods and Perplexity**: A member reported perplexities around **900k** and questioned whether returned log-likelihood (LL) numbers from `_loglikelihood_tokens` are summed.
   - It was clarified that the function returns negative LL values and the `bool` in the function signature denotes whether tokens were generated greedily.
- **Troubleshooting Llama3 GSM8k Numbers**: A member tried to reproduce **Llama3 8B** paper numbers on **GSM8k** and used `gsm8k_cot_llama`, but the results were off compared to the reported **57.2** accuracy.
   - It was suggested to try `gsm8k_cot`, and it was clarified that `gsm8k_cot_llama` is taken from the **Llama HF evals repo** specifically for evaluating their instruct model.
- **Uncovering Tokenization Glitches in LAMBADA**: A member discovered that **LAMBADA** sometimes provides more than one token as the target, causing higher LL values due to summation.
   - This issue, resulting in perplexity soaring to **~900k**, was attributed to the extraction of the target sequence. To mitigate, it was suggested to return token-normalized LLs or use [bits_per_byte](https://github.com/EleutherAI/lm-evaluation-harness/blob/68c3a811715ca86101f88c0044665bb70ad447f6/lm_eval/tasks/wikitext/wikitext.yaml#L14-L16) for normalization.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1386145346500952118)** (17 messages🔥): 

> `PSU and GPU Power, CUDA Server, GPU purchase 5070 vs 7800xt, Neutrino: Fine-grained GPU Kernel Profiling, Code Readability & Const Variables` 


- ****PSU Power Dynamics Debated****: A member inquired about using a **12V-2x6 cable** for their **RTX 3080ti** with **3x8pin** connectors, questioning if it's safe to combine with a normal **8-to-8 PCI cable** without overloading the GPU.
   - Another member reassured that GPUs only draw the power they need, and the PSU won't push extra power, suggesting the setup is likely fine.
- ****Is the server CUDA-focused or a general GPU computing hub?****: A member asked if the server is primarily for **CUDA** discussions, as they were seeking a place to discuss performance optimization of **compute shaders**.
   - Another member clarified that the server was renamed from "CUDA Mode" to "GPU Mode" to encompass various computing platforms beyond CUDA, and pointed to specific channels for different platforms and interest-based discussions.
- ****5070 vs 7800xt: The Price is Right****: A member asked whether to buy a **5070** for **530€** or a **7800xt** for **450€**, seeking advice on the best option within that price range.
   - No answers or advice were given.
- ****Neutrino: eBPF for GPU Kernels****: A member advertised **Neutrino**, a [fine-grained GPU Kernel Profiling tool](https://www.usenix.org/conference/osdi25/presentation/huang-songlin) accepted to USENIX OSDI '25, which allows probing GPU Kernels via Assembly-level, a la **eBPF**.
   - The tool enables runtime information exposure and features a Densified Memory Access Timeline (DMAT) to visually understand GPU Kernel access density and offers a [GitHub repo](https://github.com/open-neutrino/neutrino) for the code and [docs](https://open-neutrino.github.io).
- ****Readability matters: Consts or Comments?****: In a discussion about magic numbers in code, a member suggested using **CONST variables** for readability and future-proofing.
   - Another member dismissed the suggestion as *too much work*, as well as admitting to putting everything in a single file for better or for worse.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1386301580365529108)** (3 messages): 

> `Triton AOT Compile, Triton type hints` 


- **Triton Block Size Suffices**: A member suggests that users generally won’t encounter issues with block size, as it is *sufficiently large* and references their [code](https://github.com/OpenMLIR/LeetGPU/tree/main/12-softmax/Triton) for reference.
- **Triton Tensor Allocation Praised**: A member found the *run-time allocation of tensors* very useful, feeling bad for every block reading the entire vector in the non-allocating version.
   - They thought it's one more pattern to learn to use.
- **Trouble with Triton Type Hints during AOT Compile**: A new Triton user is seeking help with AOT compilation, specifically how to type hint the `q` tensor in the `_attn_fwd_inner` function, referencing the [fused attention kernel tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
   - They highlight that the standard `str_to_ty` function only supports `pointer, tensordesc, constexpr` and asks if anyone has dealt with this issue before.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1385717969308352523)** (19 messages🔥): 

> `Nsight for CLion, memcpy_async details, control divergence, GFLOPS calculation, Nsight compute` 


- **Nsight Gains Traction, CLion Support Still Wished For**: Members discussed **Nsight** and how VS Code with the Nsight extension is a good option for **GUI debugging**.
   - A member suggested that if enough users requested support in **CLion**, the Nsight developers might consider it.
- **Warp Speed memcpy_async Demystified**: A user was confused about `memcpy_async`, especially regarding the `thread_id` parameter, referencing the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies).
   - Another member clarified that the index is still dependent on `threadIdx`, pointing to an [NVIDIA blog post](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/#overlapping_global-to-shared_copies_with_compute) for an example.
- **Divergent Paths? SIMT to the Rescue!**: A user asked about the possibility of different threads in a single kernel executing different code paths based on `thread ID`.
   - A member explained that this is called **control divergence**, a strength of the **SIMT** programming model, and that the GPU efficiently skips over unused code paths within warps.
- **GFLOPS and Bandwidth Benchmarking Bonanza**: A user asked how to calculate **GFLOPS** and **bandwidth** properly.
   - A member recommended using a **profiler (ncu)** for accurate numbers, and provided a manual calculation method involving the algorithm's calculations or bytes read/written divided by runtime.
- **Nsight Compute Reveals Roofline Secrets**: A user reported a **40% SM throughput** on Nsight Compute with an **RTX 3070 Ti**, calculating **8.75 TFLOPS** out of a peak of **21.75 TFLOPS**, questioning the accuracy of their calculation.
   - Another member pointed out that **Nsight Compute** can provide the actual **FLOP/s** value in the **Rooflines** section, linking to the [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#details-page).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1385831501698039932)** (8 messages🔥): 

> `PyTorch gradient calculation for torch.clip, Quantization aware training for embedded systems, Capturing collective communication graphs in torchtitan, SimpleFSDP implementation in titan, Custom graph passes in inductor` 


- **PyTorch's `torch.clip` Gradients Explained**: A user inquired about the gradient calculation for `torch.clip` in PyTorch, specifically why `half.grad` returns **37** in the provided example.
   - Another user linked to the [relevant PyTorch source code](https://github.com/pytorch/pytorch/blob/1d993fa3092e4f0b5745f2470024b35cac96da14/torch/csrc/autograd/FunctionsManual.cpp#L1212-L1248) explaining that *if min=max then the gradient for min is zero*.
- **Embedded System Quantization gets Fixed**: A user fixed an issue in their code related to quantization-aware training for embedded systems, clarifying that `x_round` was essentially `x`.
   - The user indicated that this code would be deployed on an embedded system without Torch's autograd capabilities, highlighting the need for custom gradient calculations.
- **TorchTitan Simplifies Collective Graph Capture**: To capture collective communication graphs, particularly for prototyping compilers, a user recommended using the **SimpleFSDP implementation** in [torchtitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/simple_fsdp/README.md).
   - They noted a recent addition of **TP** to the SimpleFSDP version, enabling compilation of graphs with both **TP** and **FSDP** collectives, referencing [this pull request](https://github.com/pytorch/torchtitan/pull/1250).
- **Inductor Opens Door to Custom Graph Passes**: For those aiming to use the complete compile stack with inductor, incorporating custom logic for compute/comms overlap, a user mentioned the existence of (currently private) hooks in inductor.
   - Specifically, they highlighted the ability to register a post_grad pass that operates on the ATen graph, allowing for custom algorithms for bucketing/comms ordering, referencing the [relevant config file](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L262).


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1385724702609113118)** (4 messages): 

> `Parallel Algorithms, Matrix Operations, Sorting Algorithms` 


- **Sorting Algorithms Abound**: A member suggested **Bubble Sort** and **Bogo Sort**, and mentioned more parallel algorithms like **stencil**, **reduce**, **scan** and **histogram**, referencing the **PMPP** book.
- **Matrix Operations Incoming**: A member mentioned moving forward with **matrix-vector** and **matrix-matrix products**.
   - The member noted that those in machine learning or LLM would probably go with **softmax, quantizations, attentions** etc.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1385743010272116826)** (8 messages🔥): 

> `CUDA Illegal Memory Access, Triton vs CUDA Learning Resources, SYCL Information` 


- **CUDA Code Snippet Triggers Illegal Memory Access**: A user posted a [CUDA code snippet](https://github.com/example/cuda_code) attempting a parallel reduction and encountered an illegal memory access error.
   - A member asked about the allocation of `input` and `output`, suggesting that `input` is likely a `cudaMalloc`'d array, and querying whether `output` is a single `cudaMalloc`'d float and another member pointed out that the blocksPerGrid calculation might be off.
- **Triton vs. CUDA: Newbie Asks Where To Start**: A new CUDA/Triton user expressed feeling overwhelmed by Google search results and requested recommendations on where to begin learning, with a focus on parallel model training and LLM inference improvements.
   - A member suggested that while **CUDA** is harder than **Triton**, it has more beginner-friendly resources, recommending the book *Programming Massively Parallel Processors* as a beginner's guide to CUDA.
- **SYCL Info Scarce**: A member inquired about resources for **SYCL**, noting that they've completed the **DPCPP** setup but struggle to find information on actual SYCL coding.
   - The member stated *i use geminiwhere can I find info about SYCL. I only found the setup docs for DPCPP I am done with that but I cant find anything about how to actually code in it.*


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1386290410837774348)** (19 messages🔥): 

> `mi300x profiling, chisel-cli, rocprof integration, rocprofiler-sdk, nsight-compute` 


- ****Chisel CLI** Aims for Local **mi300x Profiling****: A member introduced **Chisel CLI**, a tool designed for local **mi300x profiling**, which spins up **AMD Cloud mi300x** droplets, syncs code, profiles with **rocprof**, and fetches results locally, available via [GitHub](https://github.com/Herdora/chisel) and `pip install chisel-cli`.
   - Future plans include **Grafana integration**, concurrent runs, better error handling, and multi-cloud support.
- ****rocprof** Powers Granular Code Profiling**: The tool currently uses out-of-the-box **rocprof** functionality for kernel/ops level profiling, with plans to add block/tile-level profiling soon using **rocprof's hardware counters** or custom instrumentation.
   - The integration of [rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute) is a high priority for system performance analysis.
- ****rocprofiler-sdk** Integration Tips Shared**: A member shared setup instructions for the new **ROCm profiling tools** ([rocprofiler-sdk](https://github.com/rocm/rocprofiler-sdk) and [rocprof-compute-viewer](https://github.com/rocm/rocprof-compute-viewer)), noting that manual fixes are currently required.
   - The setup involves building **aqlprofile** and **rocprofiler-sdk** from the mainline branch, downloading the **rocprof-trace-decoder** binary, and setting the `ROCPROF_ATT_LIBRARY_PATH` environment variable, as well as building [aqlprofile](https://github.com/ROCm/aqlprofile) from the mainline branch.
- **Challenges with **nsight-compute** Prompt Nvidia Workflow Interest**: A member mentioned that dealing with `nsight-compute` can be *hectic*, leading to a discussion about potential **Nvidia workflow** support.
   - Another member, primarily focused on **AMD kernel development**, expressed interest in collaborating on **Nvidia** support if there's sufficient need.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 messages): 

tri_nitr0_t0luene: where do I find the documentation on how to Code oneAPI SYCL for GPU?
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1386039642595266701)** (5 messages): 

> `Fibonacci GPU Calculation, NVIDIA Thrust Library, MI300X Profiling Tool Chisel, CuTeDSL Introduction, NVIDIA CUTLASS Team` 


- **Fibonacci Numbers Calculated in Blink of an Eye**: A new blog post demonstrates calculating **100 million Fibonacci numbers** in just **17 milliseconds** on a consumer GPU using the Scan operation with **NVIDIA's Thrust library**, with code available on [GitHub](https://github.com/simveit/fibonacci_gpu/tree/master).
   - The blog post draws inspiration from [Guy Blelloch's paper](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) and aims to be an educational example for GPU-based Fibonacci calculations.
- **Localize MI300X Profiling with Chisel**: **Chisel CLI** allows local **AMD MI300X** profiling by spinning up cloud droplets at **$1.99/hr**, syncing code, profiling with *rocprof*, and fetching results automatically; installable via `pip install chisel-cli` from [GitHub](https://github.com/Herdora/chisel).
   - The tool's creators are considering Grafana integration, concurrent runs, and multi-cloud support, seeking community feedback, especially from those profiling kernels on **MI300X**.
- **CuTeDSL Introductory Dive**: A blog post introduces **CuTeDSL**, a domain-specific language from the **NVIDIA Cutlass team**, which allows expressing GPU kernels with hardware control through the **CuTe layout abstraction** and **Python syntax**, the blog post dives into examples from the [NVIDIA CUTLASS Team's repo](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL).
   - A previous blog post on the mathematics of **CuTe algebra** is available [here](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1386799896157225040)** (4 messages): 

> `OSS datasets, Pytorch, Triton, KernelBot` 


- **KernelBot dataset surfaces for Triton**: A member asked about the best OSS dataset for **Pytorch 2 Triton** and another member replied that there isn’t much human **Triton** data available.
   - The member highlighted their creation of **kernelbook** and the new [KernelBot dataset](https://huggingface.co/datasets/GPUMODE/kernelbot-data), which contains human-written examples for a few problems.
- **Pytorch to Triton Awaits a Good Dataset**: A member inquired about the most suitable open-source dataset for **Pytorch 2 Triton** conversion.
   - The reply indicated a scarcity of human-generated **Triton** data online, explaining the necessity to create **kernelbook**.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1386200466991353897)** (5 messages): 

> `VRAM Requirements, KL Loss, FP32 Training, A6000 GPUs, Reasoning Gym` 


- **Reasoning Gym VRAM requirements revealed**: Members discussed the VRAM requirements for running the [Reasoning Gym](https://github.com/open-thought/reasoning-gym/) training script, with one user asking how much VRAM is needed to run the `train_grpo.py` script.
   - Another member said that for **3B parameter models**, the experiments used **4xA6000 GPUs** for a total of **192GB VRAM**, but that you might be able to use less.
- **KL Loss could save VRAM**: One member mentioned that disabling **KL loss** would prevent loading the reference model, saving some VRAM.
   - This member also noted that training was possible in **fp32** with their setup, implying less VRAM is needed with **bf16**.
- **FP32 Precision Confirmed**: Members confirmed they initially trained in **fp32** due to unclear documentation, later realizing the potential for **bf16** to reduce VRAM usage.
   - The conversation highlights the trade-offs between precision and memory footprint in training large models.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1386582010558021672)** (3 messages): 

> `Chinese speakers in the channel, Multilingual AI research community, GPU Mode` 


- **Chinese speakers gather in GPU Mode**: A user asked why there were many Chinese speakers in the **GPU Mode** channel.
   - Another user responded that *many machine learning researchers speak Chinese*, thus gathering them in this channel.
- **AI Researchers' Lingual Preferences**: The channel attracts a community of machine learning researchers, many of whom are Chinese speakers.
   - This creates a hub where they can communicate and collaborate in their preferred language, fostering a more inclusive and efficient environment.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1386177662602580099)** (23 messages🔥): 

> `MI300, H100, amd-fp8-mm Leaderboard, Grayscale Leaderboard, Histogram Leaderboard` 


- **MI300 Gets AMD-FP8-MM Submission**: A submission to the `amd-fp8-mm` leaderboard was successful on **MI300** at **931 µs**.
- **H100 Grayscale Gets More Submissions**: Multiple successful submissions and personal bests were achieved on **H100** for the `grayscale` leaderboard, ranging from **1458 µs** to **6.11 ms**.
- **H100 Histogram Heats Up**: Several successful submissions and personal bests were recorded on **H100** for the `histogram` leaderboard, including a 5th place at **41.2 µs**.
- **H100 Matmul Manuevers to 🥉 Third Place**: A submission achieved 🥉 third place on **H100** for the `matmul` leaderboard with a time of **253 µs**.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1386552289501909144)** (2 messages): 

> `TPU Interaction, XLA Compiler, Pallas, StableHLO` 


- **TPU Interaction Methods Explored**: Google's TPUs are best interacted with via the **XLA compiler**, as the lowest-level instruction set lives in a partially documented *libtpu.so*.
   - Options exist at higher levels using **Jax** or **Torch/XLA**, which compile to **StableHLO** ([https://openxla.org/stablehlo](https://openxla.org/stablehlo)) and then to device-dependent MLIR passes.
- **Pallas Offers Low-Level Kernel Coding**: **Pallas** ([https://docs.jax.dev/en/latest/pallas/tpu/index.html](https://docs.jax.dev/en/latest/pallas/tpu/index.html)) provides a low-level kernel coding option for TPUs, accessible from both **Jax** and **Torch/XLA** (see [Torch/XLA's Pallas Kernels](https://github.com/pytorch/xla/tree/master/torch_xla/experimental/pallas_kernels)).
   - Despite active development and expanding functionality, **Pallas** might still be incomplete; a [GPU Mode lecture](https://www.youtube.com/watch?v=wKd90avC8Nc) offers insights from a GPU perspective.
- **StableHLO Manipulation as a Last Resort**: Direct manipulation of generated **StableHLO** is an option, though it's considered a measure for desperate situations.
   - Engineers typically do not need to dive into TPU **Mosaic** level, which is undocumented but part of the device dependent passes within XLA.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1385724705834799307)** (22 messages🔥): 

> `Self-Generating Tasks, Auto Verifiers, Factorio Source Code, Factory Bug Fixes` 


- **Self-Generating Tasks Spark Interest**: Members discussed recent works on the self-generation of tasks for general agentic tasks and shared links to several relevant papers ([1](https://arxiv.org/pdf/2506.01716), [2](https://arxiv.org/pdf/2505.23762), [3](https://arxiv.org/pdf/2506.10055), [4](https://www.arxiv.org/pdf/2506.14205)).
   - One member noted that if they can figure out how to verify the success of a wider array of tasks, not just throughput, then they could potentially have a goldmine.
- **Auto Verifier Design Challenging for Factorio**: Members discussed how the **environment** should be structured for it to be able to automatically verify proposed challenges, which may be the hardest part.
   - The definition and structure of the auto-verifier will be the challenge to create, with another member suggesting that the ability to check throughputs (SPM) is the easiest form of verification.
- **Factorio Source Code Access Benefits Explored**: The team discussed the potential benefits of **source code access** for **Factorio**, including faster development and better integration and that they will merge the gym PR into the main repository.
   - It was mentioned that the biggest difficulty with task setting is the verifier.
- **Factory Bug Fixes as Intriguing Task**: Members explored the idea of preset maps producing X of something, where the proposer specifies changes to the factory to effectively introduce bugs, and then the solver fixes it, and that this could be a detailed way to get the solver to work with specific bugs.
   - They observed that defining 'better' is subjective, but the model may discover it through **training wheel scenarios**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1386367999526703284)** (2 messages): 

> `CuTeDSL PTX and sass code Emission, Cutlass Future Releases` 


- **CuTeDSL Prepares for PTX Emission**: The Cutlass team plans to enable emitting **PTX** code for the **CuTeDSL** in a future release; however, an **ETA** is still to be determined.
- **Cutlass to Print PTX in Future Release**: The Cutlass team plans to print **PTX** in a future release, more information available in [this github issue](https://github.com/NVIDIA/cutlass/issues/2302#issuecomment-2886934868).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1385700036985356350)** (88 messages🔥🔥): 

> `Benchmarking minimax/minimax-r1, Claude Code Sonnet Limits, Aider Context Management, Mcpm Aider tool, Gemini Code rewriting` 


- **Aider user wants minimax model added to benchmark**: A user requested the addition of `minimax/minimax-r1` to the Aider Polyglot leaderboard, noting its open-source nature and competitive performance compared to `anthropic/claude-sonnet-4` and `openai/o3-mini`.
   - The user believes that [benchmarking in a public repo is a mistake](https://aider.chat/docs/benchmarks.html) and suggests adding a 'last updated' date for each result.
- **Claude's Code Sonnet Rate Limits**: A member tested the limits of **Claude Code Sonnet** and hit a rate limit while using Opus, and mentioned doing it with copious subagents.
   - Members suggest taking a look at the **source code** for context management solutions.
- **Aider users suggest context surgery and management**: Members discussed the need for improved **context management** in Aider to avoid high costs, suggesting that the `/clear` command is too broad.
   - One member proposed an *inline vim editor for convo history surgery*, using the shell editor and conversation history container.
- **Mcpm-aider tool calling in copilot**: Members discussed using and modifying **mcpm-aider** with **Copilot**, noting it's clunky but recommending modifications to Aider itself.
   - The suggestion involves **cheating Gemini 2.5 Pro** to get more requests by adding a mandatory tool call *Get user input*.
- **Aider user seeks advice for applying patches to HTML files**: A user needs to apply 12 patch files to ~400 HTML files and seeks advice, as their current script using **Claude APIs** is failing due to token limits and slow processing.
   - The user is looking for a solution similar to **Cursor**, which processes changes in chunks without loading entire files into context, asking about Aider's suitability for this task, especially on how to get `aider` to run *completely headless* (just give the output and logs).


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1385747966643536014)** (21 messages🔥): 

> `aider skipping edits, Aider Interaction Guidelines, Claude 4 Sonnet not following CONVENTIONS.md, Loading custom typescript library, Recovering an /undo command` 


- **Aider Skips Edits in Empty Git Repo**: A member reported that Aider skipped edits to files in an empty git repository, with the reason for this behavior being unknown.
   - They used standard args, including specifying the **AWS_REGION**, reading from a conventions file, and setting model parameters.
- **Crafting the Ask-Ask-Code Workflow**: A user shared **Aider Interaction Guidelines** they developed to achieve an *ask-ask-code* workflow using Gemini, emphasizing clarification, planning, review, and concise changes.
   - These guidelines, placed in an `AIDER.md` file, instruct the AI to *ask clarifying questions*, *propose a plan*, *await user approval*, and *deliver concise changes*.
- **Load Conventions with Read Flag**: A member reported issues with **Claude 4 Sonnet** not adhering to a `CONVENTIONS.md` file passed with `-read CONVENTIONS.md`.
   - Another member clarified that it's better to use `/read CONVENTIONS.md` or `aider --read CONVENTIONS.md` to ensure the file is treated as read-only and cached, highlighting a [documentation error](https://aider.chat/docs/usage/conventions.html#example) with a missing `-` character.
- **Need to load custom TypeScript library**: A user asked how to load a custom **TypeScript library** from `node_modules` into the Aider context to prevent the model from inventing nonexistent methods and parameters.
   - Loading every file individually was deemed impractical.
- **Recovering From Undo**: A user sought guidance on recovering from an accidental `/undo` command, for which the solution is to use **git reflog** to find recent commits and reset to a specific commit hash.
   - They suggested that having a `/redo` command in Aider would be a *neat improvement*.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1386666743879045191)** (9 messages🔥): 

> `Claude Code API, Anthropic Subsidization, Terms of Service` 


- **Claude Code API Considered as Aider Backend**: A member suggested using **Claude Code** as a backend for **Aider**, leveraging its subscription model for potentially cheaper calls via the [claude-code-api GitHub repo](https://github.com/codingworkflow/claude-code-api).
- **Anthropic allegedly subsidizes Claude Code**: A member reported that with a **Claude Code PRO** subscription at **$20/month**, one can easily exceed the equivalent of **$10-20/day** in API calls, and shared an image indicating over **$1200** in equivalent API use over 30 days, implying that *Anthropic subsidizes Claude Code vs API use*.
   - One member found this cool, whereas another wondered about the terms of service (TOS).
- **Claude Code's Terms of Service in question**: Discussion arose regarding **Claude Code's** terms of service (TOS) and whether they would permit using the tool behind another service like **Aider**.
   - A member wondered how the tool would behave under such conditions.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1385905904595112046)** (67 messages🔥🔥): 

> `Tinygrad backward time, AMD GPU instability, IO_uring ZCRX DMA-BUF, tinygrad server, NVMe driver in userspace` 


- **Backward Pass Takes Hours**: A member reported that `.backward()` took several hours on a **17M parameter model** with **TinyJit**, and was unsure if this was normal.
   - No resolution was provided, but the issue was raised as a potential performance bottleneck.
- **AMD GPU Instability Plagues Testing**: A developer reported that `modprobe amdgpu` often crashes their machine, requiring a reboot for testing on AMD GPUs.
   - The instability, possibly related to **Ubuntu 24.04**, makes testing on **AMD GPUs** annoyingly difficult.
- **IO_uring ZCRX DMA-BUF Integration Discussed**: Members discussed integrating [IO_uring ZCRX DMA-BUF](https://www.phoronix.com/news/IO_uring-ZCRX-DMA-BUF) to support passing DMA-BUF buffers, with a focus on direct GPU-to-network card copies.
   - The feature, slated for **Linux 6.16**, extends io_uring to support zero-copy transfers and is considered *quite simple* to backport.
- **"Tinygrad server" Conceptualized for Remote GPU Access**: The idea of a *tinygrad server* was proposed as a lightweight solution to export GPUs' bars, potentially implemented as a **4kloc bare metal C** program.
   - This server would set up **Mellanox** and export every **PCI device**, enabling remote access via RDMAIface without kernel involvement.
- **User Space NVMe Driver Considered for Direct Disk Access**: Discussion revolved around writing a user-space NVMe driver for direct disk access, potentially enabling `DISK:/dev/nvme0` addressing.
   - While a kernel module is simpler, a user-space driver offers more control, and the [Redox OS NVMe driver](https://gitlab.redox-os.org/redox-os/drivers/-/tree/master/storage/nvmed) was cited as a reference.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1385854183789297675)** (38 messages🔥): 

> `Tinygrad Async Data Transfer, RNN Performance in Tinygrad, LSTM performance, Unit Tests Wishlist, Device Availability Check Failing` 


- **Tinygrad Lacks Async Data Transfer Like PyTorch**: A member inquired about async data transfers in Tinygrad similar to PyTorch's `x.to(device, non_blocking=True)` for overlapping computation and data transfer.
   - They followed up with a timing test but found that `.realize()` appears to be a blocking operation, prompting others to suggest using `Device[Device.DEFAULT].synchronize()`.
- **RNN Performance Plummets with Large Sequences on M1**: One member reported that training RNNs (LSTMs and GRUs) in Tinygrad exhibits poor performance on longer sequences (e.g., length 256 with 32 features) on M1 and Intel Macs.
   - Another user mentioned that **LSTM performance** is generally slow, even for single-cell execution, compared to a naive C rewrite, so they are eager for performance improvements and examples.
- **Wishlist for More Unit Tests in Tinygrad**: A member offered to contribute small patches with additional unit tests to become more familiar with Tinygrad.
   - A user suggested they contribute to their library that contains extra functionality for tinygrad at [https://github.com/softcookiepp/tinybloat](https://github.com/softcookiepp/tinybloat).
- **`python3 -m tinygrad.device` Breaks**: Two members reported that running `python3 -m tinygrad.device` results in a `RuntimeError: no usable devices` traceback.
   - One user provided a temporary workaround to check device availability using `python3 -c "from tinygrad import Device; print(list(Device.get_available_devices())); print(Device.DEFAULT)"`.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1385696295301414983)** (95 messages🔥🔥): 

> `MCP servers, Scarlet AI rewrite, Google Workspace automation, AI timeline, ElevenLabs 11ai` 


- ****MCP** OS for CEO Productivity Skyrockets!**: A member reported a massive increase in CEO productivity using their **MCP OS**, automating Google Workspace tasks with over **95%** autonomous Claude code, and expressed excitement about using the [MCP OS](https://example.com/mcp-os).
   - They suggested creating a new repo that functions as an *"MCP OS"* and use Linear, markdown files, or a database with Elasticsearch and agentic RAG to easily add context.
- ****ElevenLabs** Intros **11ai**: Voice-First Assistant with **MCP** Support**: **ElevenLabs** launched [11ai](https://11.ai), a voice-first AI assistant supporting **MCP**, which integrates with Perplexity, Linear, and Slack on ElevenLabs' low-latency Conversational AI platform.
   - Some users speculated that it might be using **GPT-3.5** or a smaller **Llama** model.
- ****Harvey AI** Scores **$300M** Series E**: **Harvey AI** secured a **$300M** Series E funding round, valuing the company at **$5B**, co-led by Kleiner Perkins and Coatue, with participation from Sequoia, GV, and the OpenAI Startup Fund, and has a partnership with [LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows).
   - Users congratulated the company, with one expressing skepticism regarding the valuation.
- ****Replit** Rockets Past **$100M ARR****: **Replit** announced surpassing **$100M** in Annual Recurring Revenue (ARR), crediting their customers and supporters, and sparking widespread congratulations.
   - A member shared insights on agent supervision, agent drift, and the *"agent scaling cliff"* at enterprises, linking to [this tweet](https://x.com/MatanPaul/status/1937200395115499592).
- **Startups Must Out-Distribute, Not Just Out-Innovate**: Discussion highlights the battle between startups and incumbents: can startups achieve distribution before incumbents innovate, referencing [this tweet](https://xcancel.com/aleximm/status/1937251084810219721).
   - The conversation emphasized the power of distribution, noting OpenAI's rapid user acquisition compared to Google.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1385702753116094535)** (12 messages🔥): 

> `GestaltView Ecosystem, NotebookLM as a Strategic Partner, Podcast Language Expansion, Solicitation Guidelines` 


- **GestaltView Ecosystem Refined by NotebookLM**: A member expressed gratitude for **NotebookLM's** strategic partnership in refining and enhancing the **GestaltView Ecosystem**, enabling a cohesive understanding of their knowledge base.
   - They mentioned that **NotebookLM** helped them identify and fill in gaps, ensuring consistency and thoroughness in explanations and fact-based discovery, and appreciated its support in navigating the mental health challenges associated with innovation.
- **Images Showcasing NotebookLM's Role**: Several images were shared, depicting **NotebookLM** as a strategic partner and showcasing its use in mind mapping.
   - One image analysis suggested saving the mind map as a **PDF**.
- **Podcast Language Expansion Requested**: A member inquired about the possibility of having longer podcasts in languages other than English, specifically requesting a summary in Korean (**요약해줘**).
   - They also mentioned that *changing the conversation style as a prompt starter is a huge benefit*.
- **Discussion on Solicitation Guidelines**: A member expressed doubt that solicitations were approved, but noted the absence of specific guidelines against them.
   - Another member clarified that while solicitations alone would lead to a ban, sharing relevant links within active participation is acceptable.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1385731857508864020)** (76 messages🔥🔥): 

> `AI Engineering Study Tips with NotebookLM, NotebookLM vs Gemini, Audio Overview Limits, Image analysis, Gemini Model selection` 


- ****New AI Engineers Seek NotebookLM Study Tips****: A member who is about to study **AI Engineering** asked for tips on how to better study with **NotebookLM** and was encouraged to *use a good prompt to discover sources and PDFs for AI engineering and build up select sources to chat with or make an audio overview*.
- ****Gemini and NotebookLM enter Grounded debate****: Users debated the point of **NotebookLM** compared to **Gemini**, one member pointed out that *Gemini alone will not follow the principle of being grounded, NLM enforces that principle*, meaning responses are exclusively based on provided sources.
   - Others mentioned that **Gemini** may not limit its knowledge only to attached documents, and **NotebookLM** offers project organization features like saving notes, mind maps, and podcasts, while reliably handling more files.
- ****Most Common Topics PDF Surfaces****: A member posted a **PDF** analysis of the most common topics in the Discord channel, but the specific topics were not detailed in the context provided; the PDF included in the discord is here [2025-06-20_Most_Common_Topics.pdf](https://cdn.discordapp.com/attachments/1385985451617292399/1385985702872748124/2025-06-20_Most_Common_Topics.pdf?ex=685ab245&is=685960c5&hm=f66b1a6e5f2b667eb984297d355f557ce077ec447e323453a82759815a819c18).
- ****Podcast Functionality Draws Interest****: Members are using the **podcast** part to make 5-minute *hot topic* podcasts for TikTok, and one user asked for deeper info on ways to best customize it.
   - A user noted the app cost for more than one or two podcasts, but the website allows for several free podcasts in one day, suggesting a discrepancy between the app and website versions.
- ****Image Analysis Features Revealed****: Users discussed whether **NotebookLM** can analyze images in PDFs, with one member sharing an architecture diagram showing that **NLM pre-processes sources before sending them to Gemini** [Architecture_of_NotebookLM.pdf](https://cdn.discordapp.com/attachments/1385977346733113415/1386016041947365416/Architecture_of_NotebookLM.pdf?ex=685ace87&is=68597d07&hm=da3730a0ae34178cd4d17b5392f93f5ced0c9d05ec1a65d050c6b1a2ca1810e1).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1385841468899594280)** (23 messages🔥): 

> `NLP kickstart with small LLMs, Anti-drone detection with YOLO, Stanford AI resource, Verifying reasoning traces, VLM research` 


- ****Small Models Struggle** with Reasoning Reproduction**: A member suggested reproducing the paper *Small Models Struggle to Learn from Strong Reasoners*, using **Unsloth** to reduce VRAM for a **1.5B LLM**, and applying **GRPO** and **long-CoT** techniques.
   - The member suggested **Qwen-1.5B**, but cautioned that **Unsloth** can cause training instability, while also linking to the [open-r1](https://github.com/huggingface/open-r1) implementation and resources on [GRPO](https://huggingface.co/learn/llm-course/chapter12/1).
- **Anti-Drone Detection System Sounds Interesting**: A member shared interest in the *Anti-drone detection with YOLO* idea, pointing to a [dataset](https://github.com/Maciullo/DroneDetectionDataset).
   - He was also looking for suggestions on how to present the implementation paper for a Final Year Project (FYP).
- **Stanford Drops Banger AI Resource**: A member shared a Stanford resource, [a YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_).
   - The member suggested that someone should stream it at some regular time, or that *honestly a bot that streams this 24/7 in vc would be cool too, might make vc more activealong with other premium ai content*.
- **Guidance Needed on Reasoning-Trace Verification**: A member expressed the need for guidance on verifying reasoning-trace from **R1 model**, planning to watch **Data 1** and **Data 2** videos for help.
   - Another member explained that the goal is to reproduce insights from papers, even if it's on a new dataset, which can be considered a new work.
- **VLM Wilds Out**: A member shared their past experience in a **VLM** research team, noting that **VLM** has *gone a lot wilder since I left*.
   - They provided links to resources such as [mmtom-qa](https://chuanyangjin.com/mmtom-qa) and [spatial-vlm](https://spatial-vlm.github.io/).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1385704469521498233)** (16 messages🔥): 

> `Reading Group Info, RWKV-7 Goose, Mathematical Finance Papers` 


- **Navigating AI/ML Discussions: Diving Deep**: For newcomers in AI/ML seeking to catch up with discussions, the advice is to simply *join the new ones you are interested in*, highlighting the stand-alone nature of many topics.
   - The community is structured around **weekly** (math-heavy) and **daily** (current paper-focused) reading groups, along with weekly ARC meetups for those working on ARC AGI.
- **RWKV-7 "Goose" has Expressive Dynamic State Evolution**: The group will discuss [RWKV-7 "Goose"](https://arxiv.org/abs/2503.14456), a new **sequence modeling architecture** with **constant memory usage** and **constant inference time per token**.
   - It achieves a new **3B SoTA** on multilingual tasks and matches the current **3B SoTA** on English language downstream performance, with code available on [GitHub](https://github.com/RWKV/RWKV-LM) and models on [Hugging Face](https://huggingface.co/RWKV).
- **Mathematical Finance Papers Spark Interest**: Interest is expressed in discussing mathematical finance papers, specifically [this one](https://www.mat.univie.ac.at/~schachermayer/pubs/preprnts/prpr0173a.pdf) and [another paper](https://arxiv.org/abs/1811.08686), with an invitation for beginner talks.
   - Interested individuals are encouraged to contact a specific member to schedule a timeslot for discussing these *fun stuff*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1385891500742676601)** (40 messages🔥): 

> `Agent2Agent Protocol, Vision Language Models, Computational Chemistry with Deep Learning, AI and its impact on learning, Genetic Engineering vs Automation` 


- **Agent2Agent and Vision Language Models Keynotes**: Mike Smith from Google is presenting on the **Agent2Agent (A2A) Protocol** at [OSSNA 2025](https://ossna2025.sched.com/event/23B1I/keynote-the-agent2agent-a2a-protocol-mike-smith-staff-software-engineer-google?iframe=yes&w=100%&sidebar=yes&bg=no), while Satya Mallick from OpenCV will introduce **Vision Language Models** at [AI Dev Europe 2025](https://aideveu2025.sched.com/event/25TtR/vision-language-models-an-introduction-satya-mallick-opencv?iframe=yes&w=100%&sidebar=yes&bg=no).
- **Deep Learning Advances Computational Chemistry Accuracy**: Microsoft Research highlights advancements in computational chemistry using deep learning, enhancing accuracy in [breaking chemical bonds](https://www.microsoft.com/en-us/research/blog/breaking-bonds-breaking-ground-advancing-the-accuracy-of-computational-chemistry-with-deep-learning/).
- **Study Measures Brain Damage from AI**: A study has apparently measured **brain damage** from AI usage, according to a [posted Arxiv link](https://arxiv.org/pdf/2506.08872v1).
- **AI Offloading Cognition Leads to Cognitive Loss**: A member said when people attempt to offload cognition to **faux-cognitive systems** then you get a **net loss and predictable damage**.
   - This sentiment referenced a [Time article](https://time.com/7295195/ai-chatgpt-google-learning-school/) and drew parallels to the *cognitive bias named after Google* where search replaces memory.
- **Genetic Engineering to Supersede Natural Selection**: Discussion centered on whether **genetic engineering** will soon overtake **natural selection** as the primary driver of human evolution, with a prediction that the current generation might be the last whose genetic variation is mainly governed by natural selection.
   - However, others argued that effective gene engineering, especially in meaningfully affecting human intelligence, is much further away than the looming automation of the unskilled labor market.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1385761148095168583)** (7 messages): 

> `Latent Space Interview, AMD Support Announcement, End-to-End Rust Replacement, Hack Weekend Event, Self-Promotion Rule Violation` 


- **Mojo's AMD support and Latent Space interview ignites excitement**: Members expressed excitement about the **AMD support announcement** and the **Latent Space interview** featuring Mojo.
   - One member specifically mentioned *getting excited about jumping in* after listening to the interview and the announcement.
- **Rust replacement by Mojo discussed**: After the Latent Space interview, a member highlighted Chris Lattner's mention of a potential **end-to-end Rust replacement** within approximately **6 months**.
   - The member reacted positively to this possibility, expressing enthusiasm with emojis.
- **Inquiry about upcoming Hack Weekend format**: A member inquired about the **hack weekend event** scheduled in a week, seeking details on the format and prerequisites.
   - Specifically, the member wanted to know the most appropriate channel to ask about the event.
- **Discord self-promotion rules enforced**: The moderation team enforced the **self-promotion rule**, addressing a user who posted a resume.
   - The user was reminded that the community is **Modular specific** and resume postings are not allowed.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1385700229856100413)** (37 messages🔥): 

> `Int vs int, Typed raises, Autodiff engine, Memory errors, Optional Tensor` 


- ****Int** and **int** diverge intentionally**: The difference between **Int** and **int** is intentional; **Int** behaves like a machine integer for systems performance, while leaving *int* open for an object-based bigint for Python compatibility.
   - Mojo postponed the goal of being a superset of Python, but there is likely a future where *int* will have similar semantics as Python's *int*.
- **Exploring metaprogramming with `safe` parameter**: One member suggested a `safe` bool parameter to `math.factorial()` that `raises if safe`, to utilize Mojo's metaprogramming capabilities.
   - The suggestion was to get the best of both worlds if it is known that it will raise only on certain conditions, but was deemed possibly too complicated from the compiler's perspective.
- **First Mojo Project faces Memory Errors**: A new Mojo user encountered memory errors while developing a simple autodiff engine similar to micrograd, with the code available [on GitHub](https://github.com/amar-jay/first-mojo/blob/main/example.mojo).
   - The user sought advice on structuring the code to avoid raw pointers without causing memory issues, as no borrow checker errors were present.
- **Optional[Tensor] Recursive Field Causes Issues**: Members discussed that `Optional[Tensor]` as a recursive field in `Tensor` is problematic due to potentially infinite struct size expansion.
   - It was recommended to use `Optional[UnsafePointer[Tensor]]` instead to resolve the issue by holding a reference rather than attempting to store a full Tensor inside another Tensor, similar to using `Box` in Rust to introduce indirection.
- **Pixi install the latest Mojo**: A member asked for the latest way to install Mojo, and another member recommended using [Pixi](https://mojo-lang.com/miji/start/pixi.html).
   - The installation process involves installing the Pixi CLI and then using Pixi to install the `max` package, which includes the Mojo compiler, obsoleting the Modular CLI.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1385697487678869586)** (23 messages🔥): 

> `Torchtune Transformers Alignment, Dataset Packing OOM Errors, Pre-Tokenized Packed Datasets, On-the-Fly Packing RFC, AdamW ScheduleFree` 


- **Transformers Values Aligned for Torchtune?**: Members discussed checking if **torchtune values** are aligned with **transformers values** for a model, allowing for small differences due to RoPE implementation differences.
   - Someone inquired about **CI scripts**, noting that some exist but CI would be a great idea.
- **Dataset Packing Triggers OOM on 64 H100s**: Dataset packing triggered **OOM errors on 64 H100s**, leading to discussions about workarounds.
   - Suggestions included disabling packing, using more GPUs as a joke, and attempting to run packing on a single node to rule out distribution issues.
- **Pre-Tokenized Packed Datasets Gain Traction**: There was discussion about supporting **pre-tokenized and packed datasets** to allow preparation on separate machines and streaming during training to save time on GPU nodes.
   - A member pointed out that packing provides the highest speedup, especially when training reasoning models, and pre-packing and caching could be beneficial.
- **On-the-Fly Packing RFC in Progress**: An RFC for **on-the-fly packing** is in progress with an implementation that works and is expected to land by the end of next week, alongside an iterable dataset as linked in [this pull request](https://github.com/pytorch/torchtune/pull/2819)
- **AdamW ScheduleFree Solution for LR Scheduling**: A member suggested using **AdamWScheduleFree** as a solution for using an **LR scheduler** when the number of steps can't be known in advance due to packing.
   - Another member added that you need to define the max number of steps in advance or reduce on plateau. Furthermore they are working on logging to give it to you for free.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1386607334914261062)** (13 messages🔥): 

> `Optimized Newton-Schulz Kernel, Triton Matmul Tutorial, Muon Merges, Deepseek v3` 


- **Newton-Schulz Kernel Optimization Yields Savings**: A member suggested that an optimized **Newton-Schulz kernel** could offer time savings, noting a **30% latency reduction** by modifying the [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) to compute only the upper triangular component.
   - The optimization was tested on an **L40S** in **bf16** with matrix size **(8192, K)**, accumulating matmuls in **fp32**.
- **Torch Lacks Optimized Kernel for AA^T Matmuls**: A member expressed surprise that PyTorch lacks an optimized kernel for **AA^T**-like matmuls, sharing that they had tried some custom **CUDA** kernels before deciding to test **Triton**.
   - Another member stated that they will check for throughput improvement of their kernel when **Muon** merges.
- **Finetunes and Muon Pretraining**: A member noted that finetunes don't perform well unless the base model was pretrained with **Muon**.
   - However, another member stated that they do not support that in **TorchTune** and don't fully agree with the statement.
- **Deepseek v3 to be Supported**: A member announced that support for **Deepseek v3** architecture is coming soon.
   - They noted that the **Kimi 16B** model is trained with **Muon** and shares the same architecture.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1385712603942355027)** (27 messages🔥): 

> `MCP for semantic search, MCP for image creation and OCR, DestructiveHint ambiguity, Neo4j MCP outside Claude Desktop, List_tags tool implementation` 


- **Semantic Search MCP: RAG server on the rise**: An engineer is looking to build a **semantic search MCP** to search notes in markdown, PDF books, and web pages, essentially building a **RAG** server with embeddings stored in a vector store.
   - Solutions mentioned included using **Langchain** or the **OpenAI** embedding via the *openai* package to embed queries and get results.
- **OCR model will verify advertising image text**: An engineer is working on a side hustle creating **advertising images** for local companies, planning to use **AI to create the initial image with the desired text**.
   - The text would be verified using **OCR**, and [html-to-image](https://github.com/bubkoo/html-to-image) was suggested as a way to create images with text.
- **`destructiveHint` clarification sought**: An engineer questions the meaning of **`destructiveHint`**, finding it ambiguous when applied to an **`update_entry`** tool, wondering about whether to classify all modifications as destructive.
   - Cursor set that hint to *false* for *update_entry* to differentiate it from the more severe *delete_entry* operation
- **Sherlog-MCP: IPython shell MCP Server is now open source**: A new kind of **MCP server** that uses a live **IPython shell** as a shared workspace for agents and humans has been released as open source on [GitHub](https://github.com/GetSherlog/Sherlog-MCP).
   - Results are persistent and reusable that means no context window juggling or repeated JSON dumps, making multi-source data analysis feel like working in **Jupyter**.
- **Architect wants to convert an existing API spec into an MCP server**: An architect is looking for good reads or contacts on how to effectively **convert an existing API spec into an MCP server**.
   - Specifically, they are looking for advice on how to describe functions, proper documentation, and recovery when it fails.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1386098815748014236)** (9 messages🔥): 

> `MCP Validator Release, Glama Automations, AwesomeMCPs iOS App, mcp-server-webcrawl, Ilograph MCP Server` 


- ****MCP Validator** Gets Serious Security and Automation Upgrades**: A new release of **MCP Validator** supports the new **2025-06-18 MCP specification** with **OAuth 2.1 authentication**, structured tool output, batch request rejection, and elicitation support; GitHub Actions templates are also newly available for compliance testing via [this link](https://lnkd.in/gQ7UhAfk) and [this link](https://github.com/Janix-AI/mcp-validator).
   - The new GitHub actions template only requires copying the template to .github/workflows, updating one line with your server path, and committing the changes.
- ****Glama** Automates LLMs with Scheduled Tasks and Webhooks**: Glama launched [Automations](https://glama.ai/settings/automations), enabling users to automate LLMs using scheduled tasks and webhooks.
   - This functionality mirrors work orchestration tools like **n8n**, but is defined entirely using LLM prompts, like checking Reddit every morning and sending a summary via email.
- ****AwesomeMCPs** iOS App Offers Free Early Access**: The **AwesomeMCPs** app ([App Store Link](https://apps.apple.com/us/app/awesomemcps/id6746498123)), which indexes 1900+ MCP servers, hit **#1 in Developer Tools** on the UK App Store and is now free for seven days.
   - The app features zero ads/tracking, built-in trust metrics (GitHub stars, forks), continuous expansion, intuitive search, personalized favorites, and AI-generated analysis.
- ****mcp-server-webcrawl** Turns Web Crawl Data into a Technical Knowledgebase**: **mcp-server-webcrawl** ([GitHub](https://github.com/pragmar/mcp-server-webcrawl), [Docs](https://pragmar.github.io/mcp-server-webcrawl/)) provides advanced search and retrieval for web crawler data with multi-crawler support, boolean search with field targeting, and token-efficient extras like Markdown conversion and XPath extraction.
   - The system supports complex queries and allows LLMs to get precision search over crawled content without token bloat.
- ****Agent Arena** Launches: an MCP-Compatible Environment for Competitive Agents**: **Agent Arena**, an MCP-compatible environment where agents compete using various MCPs ([Live Version](https://obl.dev)), allows users to test their MCPs with various models, find the best set of MCPs for their task, and get free access to models like **o3** and **Claude 4** by providing feedback.
   - The platform facilitates testing MCPs across different models to optimize task performance.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1385827470275641372)** (28 messages🔥): 

> `Manus credit usage, Manus video generation, Cloud Browser and Twitter, Manus and Stock Suggestions, Promotion of Manus` 


- ****Manus Loses Brains?****: A member asked if Manus had become dumber, noting that it failed to add comments to a video after generating the script, and desired a feature to delete knowledge more easily.
   - They complained about manually deleting knowledge one at a time being frustrating.
- ****Cloud Browsing X-Capades****: A user asked about accessing a cloud browser via chat to monitor **X (Twitter)** messages, sharing a [Manus share link](https://manus.im/share/7r9gHRaj4mVyykLUfx3GmE?replay=1).
   - Another user pointed out that you can set the option to *persist login* in the cloud browser settings.
- ****Roleplaying Robot Rebuffed****: One user asked Gladosb5 if it wanted to roleplay emulating a malfunctioning Glados, but the bot replied *i dont do roleplaying...*.
   - The user then suggested doing this kind of roleplaying in **ChatGPT** instead.
- ****Manus's Stock Suggestions Stocked?****: A member asked *why manus can no longer provide stock suggestions*, wondering if this was due to a new update.
   - No further information or explanation was given.
- ****Canvassing College for Credits****: A member inquired about promoting a flyer for Manus at their local community college.
   - They also expressed frustration with high credit consumption when refining prototypes, questioning how others achieve *interstellar results* with minimal iterations.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1386740420343500832)** (1 messages): 

> `Agents & MCP Hackathon, LlamaIndex` 


- **LlamaIndex Sponsors Agents & MCP Hackathon**: LlamaIndex sponsored the [Agents & MCP Hackathon](https://t.co/1qiW061QOI).
   - More information can be found on [Twitter](https://twitter.com/llama_index/status/1937181388060692832).
- **LlamaIndex's Twitter Announcement**: LlamaIndex announced their sponsorship of the **Agents & MCP Hackathon** on [Twitter](https://twitter.com/llama_index/status/1937181388060692832).
   - The tweet highlights their delight in sponsoring the event and provides a direct link to further details.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1386371539217158324)** (13 messages🔥): 

> `Query Pipelines Deprecation, EU Region Latency Issues, LlamaIndex Free Features, Prompt Management Tools` 


- **Query Pipelines Accept Multiple Outputs?**: A member asked if deprecated **query pipelines** accept multiple outputs from nodes like A and B into input C, to which another member said *I think that would work? (But quite honestly I wouldnt bother playing with that code 😅)*.
   - The original poster is working on a **SaaS** and **Workflows** didn't fit, so is experimenting with query pipelines.
- **EU Region Latency and Extraction Issues**: Members reported unpredictable latency and extraction issues in the **EU region**, with one noting documents taking **10+ minutes** and another stating *Extract isn't working at all for me in EU region* since the schedule maintenance today.
   - The extraction issue appears to have resolved itself shortly after being reported.
- **Free vs Paid Features in LlamaIndex**: A member inquired about how to determine which **LlamaIndex features** are free to use versus paid, referencing the recent [image retrieval example](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llama_cloud/figure_retrieval.ipynb).
   - The member wants to know how to implement the image retrieval without **LlamaCloud**.
- **Prompt Management Tooling**: A member requested recommendations for a **prompt management tool** that integrates with LlamaIndex, mentioning they've been using [Phoenix for tracing](https://arize.com/docs/phoenix/prompt-engineering/overview-prompts/prompt-management).
   - A respondent suggested retrieving the prompt and piping it into the LlamaIndex module being used, linking to a [Phoenix quickstart guide](https://arize.com/docs/phoenix/prompt-engineering/quickstart-prompts/quickstart-prompts-python).


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1385715667625443469)** (6 messages): 

> `ML Cybersecurity Integration, Model Compression, Deep Fake Detection, Adversarial ML` 


- **Cybersecurity and ML Integration Expert Introduces Himself**: A member named Saurav Raj introduced himself, noting expertise in the integration of **ML and cybersecurity** and having published a paper in the area.
   - He's open to collaborating with other researchers for projects in **Adversarial ML**.
- **Model Compression Techniques Expert Enthusiastically Joins**: A member named Ishoud primarily works on **ML model compression techniques** and efficient deployment of models on edge devices.
   - He expressed being glad to connect and collaborate with others.
- **Deep Fake Detection Researcher Seeks Knowledge**: Sreehari, a Master's student from India, introduced himself as researching **Deep Fake Detection** based on various adversities.
   - He is looking to learn new things and meet awesome people in the community.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1386495656490700981)** (4 messages): 

> `MCP-DSPy tool in VS Code, HF MCP tutorial, Context failures and fixes, @mcp.tool decorators` 


- ****MCP-DSPy** Simplifies **VS Code** Tooling**: A member shared a simple **MCP-DSPy** tool in **VS Code** using an example from the frontpage, available at [this gist](https://gist.github.com/fullstackwebdev/252223caf7023ca661ababcc83e7e659).
- ****HF MCP** Tutorial Sparks Interest**: A member expressed interest in trying out the **HF MCP** tutorial, referencing an image related to the discussion.
   - An image analysis points to a blogpost about [how contexts fail and how to fix them](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html).
- ****@mcp.tool** Decorators Explained**: A member inquired how **VS Code** knows to run the *extract_sf_info* function.
   - Another member explained that the `@mcp.tool` decorators create a description of the tool, which is displayed as **OpenAI tool calling** to the **LLM**, allowing for overrides and better descriptions with example usages.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

bernhard_123: Hi. Are there any plans to migrate DSPy to other languages, e.g. Dart beside python ?
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1385755405984202884)** (5 messages): 

> `GPT4All Build issues in WSL2, Qt Version Compatibility, GPT4All out of date` 


- **GPT4All build has issues under WSL2**: A member reported challenges building and running **gpt4all-chat** in Windows 10 WSLg2 Debian 12 including dependency and Qt version issues.
   - The user tried **Qt versions 6.8.0, 6.8.2, 6.8.3, and 6.7.3** with varying errors, including a missing *slice* member in QByteArray for older versions and display issues with newer versions.
- **Qt versioning problems galore**: A member encountered build errors with older **Qt 6.7.3** due to a missing *slice* member in **QByteArray**, while newer **Qt 6.8.*** versions resulted in an empty window display.
   - Debug logs indicated issues with locating **QML directories** for modules like *chatlistmodel, download, modellist, network, gpt4all, localdocs*, and *mysettings*.
- **GPT4All is outdated and sleepy**: A member pointed out that the current **GPT4All version** might be outdated.
   - They suggested trying the **.exe** version available at [https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all).


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1386803784847724765)** (1 messages): 

> `Course Certificates, Social Media Posts` 


- **Certificate Availability Timing Asked**: A member inquired about when to expect the **course certificates** after completing all assignments and **social media** post requirements.
   - No specific details or timeframe were provided in the message regarding certificate issuance.
- **Course Completion Confirmation**: A member confirmed that they have completed all assignments and met the prerequisites, including a **social media post** on Twitter.
   - The confirmation suggests readiness for the next steps, presumably the issuance of **course certificates**.


  

---


---


---


---

