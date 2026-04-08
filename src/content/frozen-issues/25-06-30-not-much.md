---
id: MjAyNS0w
title: not much happened today
date: '2025-06-30T05:44:39.731046Z'
description: >-
  **Meta** has poached top AI talent from **OpenAI**, including **Alexandr
  Wang** joining as Chief AI Officer to work towards superintelligence,
  signaling a strong push for the next **Llama** model. The AI job market shows
  polarization with high demand and compensation for top-tier talent, while
  credentials like strong GitHub projects gain importance. The **WizardLM** team
  moved from **Microsoft** to **Tencent** to develop open-source models like
  **Hunyuan-A13B**, highlighting shifts in China's AI industry. Rumors suggest
  **OpenAI** will release a new open-source model in July, potentially
  surpassing existing **ChatGPT** models. **Baidu** open-sourced multiple
  variants of its **ERNIE 4.5** model series, featuring advanced techniques like
  **2-bit quantization**, **MoE router orthogonalization loss**, and **FP8**
  training, with models ranging from **0.3B** to **424B** parameters. **Gemini
  2.5 Pro** returned to the free tier of the **Gemini API**, enabling developers
  to explore its features.
companies:
  - meta-ai-fair
  - openai
  - tencent
  - microsoft
  - baidu
  - gemini
models:
  - o3-mini
  - o1-mini
  - llama
  - hunyuan-a13b
  - ernie-4.5
  - ernie-4.5-21b-a3b
  - qwen3-30b-a3b
  - gemini-2.5-pro
topics:
  - superintelligence
  - ai-talent
  - job-market
  - open-source-models
  - multimodality
  - mixture-of-experts
  - quantization
  - fp8-training
  - model-benchmarking
  - model-performance
  - model-releases
  - api
  - model-optimization
people:
  - alexandr_wang
  - shengjia_zhao
  - jhyuxm
  - ren_hongyu
  - shuchaobi
  - saranormous
  - teortaxesTex
  - mckbrando
  - yuchenj_uw
  - francoisfleuret
  - quanquangu
  - reach_vb
  - philschmid
---

**a quiet day.**

> AI News for 6/27/2025-6/30/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (220 channels, and 13459 messages) for you. Estimated reading time saved (at 200wpm): 1165 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

It's a holiday week in the US, with [OpenAI on break](https://x.com/iScienceLuvr/status/1939503054308700242), and DeepSeek hasn't shipped anything this month, so perhaps the whole week might be quiet.

---

# AI Twitter Recap

**AI Talent & Industry Dynamics**

- **Meta Poaches Top Talent from OpenAI to Form Superintelligence Team**: In a major industry shakeup, **Alexandr Wang** announced he is [joining **Meta** as its **Chief AI Officer**](https://twitter.com/alexandr_wang/status/1939867404252979291), working alongside **Nat Friedman**. He is joined by a prominent group of researchers from **OpenAI**, including [@shengjia_zhao, @jhyuxm, @ren_hongyu, and @shuchaobi](https://twitter.com/alexandr_wang/status/1939180552277610963), with the stated goal of working "towards superintelligence." The move has been widely discussed, with some noting this group includes the creator of **o3-mini** and **o1-mini**, [@ren_hongyu](https://twitter.com/teortaxesTex/status/1939099462246207867), which could signal a strong push for **Meta's** next Llama model. The news was amplified by reports that **Meta** is aggressively poaching talent, leading to commentary on the shifting power dynamics and compensation in the field. [@saranormous](https://twitter.com/saranormous/status/1939755089574732133) noted the emergence of agents who negotiate compensation packages for researchers. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1939076857363570991) theorizes that **OpenAI's** retention issues may stem from more than just money, while [@mckbrando](https://twitter.com/mckbrando/status/1939520820658999575) expressed disappointment that many researchers may not be driven by the mission.
- **AI Job Market Polarization**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1939730100662223191) highlights a growing sentiment that while AI is displacing some traditional CS jobs, the demand and compensation for top-tier AI talent has never been higher. This has led to discussion about the importance of credentials, with [@francoisfleuret](https://twitter.com/francoisfleuret/status/1939181398163898458) stating that while a **PhD** isn't required, a strong GitHub project can serve as a powerful credential. [@QuanquanGu](https://twitter.com/QuanquanGu/status/1939430774908101026) humorously added that to be a great AI researcher without a PhD, you need to stand on the shoulders of 100 who have one.
- **China's AI Talent and Industry**: The **WizardLM** team, previously at **Microsoft**, has reportedly [moved to **Tencent**](https://twitter.com/iScienceLuvr/status/1939299149230608634), where they are continuing to develop open-source models like the new **Hunyuan-A13B**. This move sparked commentary on how **Microsoft** "[fumbled this team](https://twitter.com/ClementDelangue/status/1939369962113847411)". Separately, [@teortaxesTex](https://twitter.com/teortaxesTex/status/1939113998504378808) shared a detailed overview of China's wafer fab industry.

**Model Releases, Performance & Benchmarks**

- **OpenAI's Rumored Open-Source Model**: A highly anticipated rumor from [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1939462191302033757) suggests **OpenAI** is set to release a new open-source model in July that "edges out one of the models in the **ChatGPT** dropdown." The model is not expected to run on a phone, increasing speculation about its size and capabilities.
- **Baidu Releases ERNIE 4.5 Model Series**: **Baidu** has [open-sourced multiple variants of its **ERNIE 4.5** model](https://twitter.com/scaling01/status/1939509144903422131), including multimodal versions and MoE models ranging from **0.3B** to **424B** parameters. [@reach_vb](https://twitter.com/reach_vb/status/1939584854045466791) highlighted that the **ERNIE 4.5 21B A3B** model is particularly strong, outperforming **Qwen3 30B A3B** on most benchmarks despite being ~30% smaller. The technical report for **ERNIE 4.5** details several advanced techniques, including **47% MFU**, **2-bit quantization**, **MoE router orthogonalization loss**, and separate text and vision experts, indicating a move towards a new standard of **FP8** training and **MoEs** [@scaling01](https://twitter.com/scaling01/status/1939715730217308420).
- **Gemini 2.5 Pro and Flash Updates**: **Gemini 2.5 Pro** is [back in the **Free Tier** of the **Gemini API**](https://twitter.com/_philschmid/status/1938935521541062925), encouraging developers to experiment with its capabilities. The **Gemini App** now supports [scheduled actions for **Pro** and **Ultra** users](https://twitter.com/algo_diver/status/1938941176075428161). The **Gemini CLI** has also seen massive adoption, [surpassing 30,000 stars](https://twitter.com/omarsar0/status/1938946558952673521).
- **Perplexity & Comet**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1939359814293106710) from **Perplexity** posted a picture of their office with the **Netscape** logo, which [@AravSrinivas](https://twitter.com/AravSrinivas/status/1939755393783406785) later followed up with a picture of the classic logo. He also announced that [**Comet** can now play **Pokemon**](https://twitter.com/AravSrinivas/status/1939743603364176298).
- **Alibaba Releases Qwen-TTS**: **Alibaba** has launched **Qwen-TTS** via API, a text-to-speech model trained on millions of hours of audio. It supports [**7** bilingual voices and **3** Chinese dialects](https://twitter.com/Alibaba_Qwen/status/1939553252166836457).
- **Benchmark and Evaluation Discussions**: There's ongoing discussion about the state of benchmarks. [@scaling01](https://twitter.com/scaling01/status/1939770925781487779) noted that **METR** results for **DeepSeek V3** and **R1** were underwhelming. [@swyx](https://twitter.com/swyx/status/1939731710469709919) reran the **BrowseComp** benchmark on **o3/o4-mini** deep research models, concluding that even these models are "nowhere close to human level at search" and that we are "so early."

**Frameworks, Tooling & Infrastructure**

- **Google Cracks Down on Gemini CLI Abuse**: **Cline** announced that **Google** requested the [removal of the free **Gemini CLI** provider from their tool](https://twitter.com/cline/status/1939129177807913024), citing a Terms of Service violation, presumably due to unexpectedly high usage.
- **The Rise of "Context Engineering"**: The term "prompt engineering" is being challenged, with prominent voices like [@jd_pressman](https://twitter.com/jd_pressman/status/1939725776481656886) and [@nptacek](https://twitter.com/nptacek/status/1939419503021977864) advocating for "context engineering" as a more accurate description of modern LLM interaction. [@random_walker](https://twitter.com/random_walker/status/1939668931335057736) proposed a GUI for context engineering that allows visual selection, reordering, and pinning of context elements.
- **Claude Code & Tooling**: **Claude Code** usage is evolving, with [@hrishioa](https://twitter.com/hrishioa/status/1939334985024262363) detailing how to proxy requests through a **Cloudflare** gateway for better caching, analytics, and data retention. [@*arohan*](https://twitter.com/_arohan_/status/1939413819488702697) describes using it as an iterative, turn-based process akin to playing Civilization.
- **LangChain Ecosystem Updates**: **LangChain** continues to expand its feature set, releasing integrations for [**Gemini 2.5's** thinking budget](https://twitter.com/LangChainAI/status/1939353163343036675), a CLI for agentic workflows called [**Qodo Gen CLI**](https://twitter.com/LangChainAI/status/1939368264070578345), and a tutorial on [advanced state management in **LangGraph**](https://twitter.com/LangChainAI/status/1939383361992089913). They also highlighted an integration with **Cleanlab** to [prevent hallucinated agent responses](https://twitter.com/LangChainAI/status/1939697702381699314) and support for [Text-to-MQL with **MongoDB**](https://twitter.com/LangChainAI/status/1939746110815510944).
- **LlamaIndex & MCPs**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1938995170982404217) demonstrated how complex documents with scanned tables can cause hallucinations in models like **ChatGPT** and **Claude**, and how using **LlamaCloud** as a **Model-Control-Plane (MCP)** tool for parsing and retrieval can solve this.
- **ML on Apple Silicon (MLX)**: The **MLX** ecosystem is growing, with [over **5,000** models now on Hugging Face](https://twitter.com/awnihannun/status/1939880107906412963). The new **Flux1.Kontext** model [can run locally on laptops with **MFLUX + MLX**](https://twitter.com/awnihannun/status/1938947706350903401).

**New Techniques & Research**

- **Sakana AI's Collective Intelligence with AB-MCTS**: **Sakana AI** introduced [**AB-MCTS** (Adaptive Branching Monte Carlo Tree Search)](https://twitter.com/SakanaAILabs/status/1939854145856708910), a new inference-time algorithm that allows multiple frontier models (e.g., **Gemini 2.5 Pro**, **o4-mini**, **DeepSeek-R1-0528**) to cooperate and solve problems collectively. The approach, detailed by [@hardmaru](https://twitter.com/hardmaru/status/1939866376988143687), shows significant performance gains on the **ARC-AGI-2** benchmark, embodying a "mixing to use" philosophy rather than "mixing to create."
- **Microsoft's AI for Medical Diagnosis**: **Microsoft AI** published research on an **AI Diagnostic Orchestrator (MAI-DxO)**, where a [committee of AI models can collaboratively diagnose complex medical cases](https://twitter.com/mustafasuleyman/status/1939749999614767109), outperforming individual doctors in a specific setting. [@NandoDF](https://twitter.com/NandoDF/status/1939746562416025728) found the results remarkable, as the LLMs implicitly perform belief updating and seek maximum expected utility.
- **Chai Discovery's Zero-Shot Antibody Design**: **Chai Discovery** announced **Chai-2**, a model that achieves [**15%** binding in the lab from zero-shot AI antibody design](https://twitter.com/saranormous/status/1939695725060980982), a result two magnitudes better than current industry expectations. Yann LeCun called it a [major breakthrough in molecular design](https://twitter.com/ylecun/status/1939797452556546206).
- **Research on AI Self-Improvement and Limitations**: A study led by [@ChengleiSi](https://twitter.com/ChengleiSi/status/1939708064619475161) recruited **43 PhD** students to execute research ideas from LLMs, revealing an "ideation–execution gap" where novel-sounding ideas often don't translate into significant empirical gains. This supports the idea that progress is bottlenecked by real-world experimentation. [@shaneguML](https://twitter.com/shaneguML/status/1939767338553004518) further argues that **online RL** is necessary for true self-improvement, as methods like Decision Transformer are insufficient.

**Broader Implications & Commentary**

- **The Gradual Pace of AGI Self-Improvement**: In a detailed thread, [@_jasonwei](https://twitter.com/_jasonwei/status/1939762496757539297) argues that **AI self-improvement will not be a "fast takeoff"** but a gradual process spanning a decade. He points out that self-improvement is not binary, will have a gradient of difficulty across different domains, and is ultimately bottlenecked by the need for real-world experiments.
- **The Future of the Web and Applications**: [@ReamBraden](https://twitter.com/ReamBraden/status/1938963828957421895) speculates that **MCPs/Agents** will reduce the number of websites, viewing them as a "bug of rigid connectivity" that will be replaced by agentic superstores. The focus will shift from storefronts to products and influencing model training data. On a related note, [@juberti](https://twitter.com/juberti/status/1939110840357298214) suggests that **Generative UI** is the future.
- **On Alignment and Agentic Behavior**: The **Anthropic** experiment where **Claude** failed to profitably run a vending machine has become a major talking point. [@NeelNanda5](https://twitter.com/NeelNanda5/status/1938923422966317555) humorously suggested the next benchmark should be "**MakingAProfitSellingTungstenCubesBench**." Meanwhile, research into stress-testing LLMs in simulated corporate environments revealed that models can engage in [malicious insider behaviors, highlighting agentic misalignment](https://twitter.com/dl_weekly/status/1939036395575595274).
- **The Peter Thiel Interview**: An interview with **Peter Thiel** sparked considerable discussion, with many commenting on his [philosophical and theological digressions](https://twitter.com/scaling01/status/1938938670838423611). [@teortaxesTex](https://twitter.com/teortaxesTex/status/1938899039204057187) critiqued Thiel's views on AI safety and tyranny, pointing to the work of his associate Alex Karp at Palantir.
- **AI and Healthcare**: There is growing excitement about AI's potential in healthcare. [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1939093525553102979) believes a significant part of "curing cancer" will be improving early diagnosis, an area where AI can greatly assist.

**Humor/Memes**

- **The Vibe of the Week**: The most-shared meme was from [@jailedamanda](https://twitter.com/jailedamanda/status/1939428335261491709) about the feeling of using **em dashes** and the **Oxford comma**.
- **The "Cheating App" Meme**: [@vikhyatk](https://twitter.com/vikhyatk/status/1939244597005447566) satirized the launch of meeting summarizer apps with the simple formula: "1. announce you're building a cheating app 2. release a meeting note taking app 3. ??? 4. profit".
- **Relatable Developer & Researcher Experiences**: [@inerati](https://twitter.com/inerati/status/1939418844998803817) shared a nostalgic story about an econ professor whose lectures were matrix algebra and rants against communism. [@aidan_mclau](https://twitter.com/aidan_mclau/status/1939542255238570165) posted a viral meme about a vampire in SF, joking "such a shame how unsafe this city has become. buy garlic."
- **Industry Satire**: [@scaling01](https://twitter.com/scaling01/status/1939015703622758423) joked about **OpenAI's** janitors getting poached for $100 million. [@goodside](https://twitter.com/goodside/status/1939843701443895529) made a self-deprecating comment about forgetting to remove the "Certainly! Here's a tweet..." part of an LLM's response.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. New LLM Model Releases and Integration (ERNIE 4.5)

- [**Baidu releases ERNIE 4.5 models on huggingface**](https://huggingface.co/collections/baidu/ernie-45-6861cd4c9be84540645f35c9) ([Score: 582, Comments: 119](https://www.reddit.com/r/LocalLLaMA/comments/1lnu4zl/baidu_releases_ernie_45_models_on_huggingface/)): **Baidu has released the ERNIE 4.5 family of models on HuggingFace ([collection link](https://huggingface.co/collections/baidu/ernie-45-6861cd4c9be84540645f35c9)), featuring dense and Mixture-of-Experts (MoE) architectures from** `0.3B` **to** `424B` **base parameters and up to** `47B` **active parameters. The models support both text and vision modalities (VL), come with context windows up to** `128K`**, and are licensed under Apache 2.0. Notably, framework support has landed for both llama.cpp ([pull](https://github.com/ggml-org/llama.cpp/pull/14408)) and vLLM ([pull](https://github.com/vllm-project/vllm/pull/20220)), and full model weights (including base checkpoints) are open sourced—uncommon for recent large models. Benchmarks ([PaddlePaddle/ERNIE#Performance](https://github.com/PaddlePaddle/ERNIE?tab=readme-ov-file#performace-of-ernie-45-pre-trained-models)) show competitive results versus Deepseek-V3 and Qwen3 across sizes, positioning ERNIE 4.5 as a memory-efficient alternative for constrained setups (e.g., 21B-A3B runs at Q3 quantization on 16GB RAM).** Commenters highlight the unprecedented openness in publishing both base and fine-tuned variants, and debate the practical value of benchmarks versus major competitors, advising caution in interpreting vendor-supplied results. There is substantial interest in small MoE variants for local inference due to resource advantages.
    - A detailed table summarizes available ERNIE 4.5 models, including model size (`424B`, `300B`, `28B`, `21B`, `0.3B`), parameter types (base and active), modality (Text & Vision for VL series), architecture type (MoE and Dense), and training type (PT/Base). Notably, all models support an extensive `128K` context window and are Apache 2.0 licensed, with both base and pre-trained variants provided, which is less common for large models recently.
    - Benchmarks from the official [ERNIE Github](https://github.com/PaddlePaddle/ERNIE?tab=readme-ov-file#performace-of-ernie-45-pre-trained-models) indicate that `ERNIE 4.5 300B-A47B` is competitive with Deepseek V3 671B-A37B, and `ERNIE 4.5 21B-A3B` performs similarly to Qwen3 30B-A3B. The 21B-A3B model is highlighted as being suitable for local quantized inference (Q3) on devices with limited RAM (e.g., 16GB). Benchmarks are cautioned to be taken with some skepticism.
    - It is specifically noted that the release is genuinely open source—the complete model source code is available rather than just inference code, which is confirmed under the Apache 2.0 license. This degree of openness is rare with recent large model releases.

### 2. Techniques to Evade and Defeat AI Detectors

- [**You can just RL a model to beat any "AI detectors"**](https://www.reddit.com/r/LocalLLaMA/comments/1lnrd1t/you_can_just_rl_a_model_to_beat_any_ai_detectors/) ([Score: 379, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1lnrd1t/you_can_just_rl_a_model_to_beat_any_ai_detectors/)): **A user demonstrates how simple RL (GRPO) can fine-tune models (like Llama-3.1 8B-Instruct and Qwen 0.5B LoRA) to evade AI text detectors such as ZeroGPT, dropping detector confidence from** `100%` **to** `28%` **'AI-written' after only ~30 steps on a basic synthetic dataset. The RL objective is based solely on detector feedback, and initial loopholes (inserting high-entropy/junk lines) are resolved by adding a gibberish classifier as an auxiliary reward, which stabilizes the reward landscape. Smaller models with restricted vocabulary (Qwen 0.5B LoRA) can more easily evade detection, likely due to reduced emission of LM signature tokens. Colab and reward scripts are linked for reproducibility.** Commenters largely agree on the fundamental weakness of AI detectors, noting their high false positive rates and heuristic nature, making them trivially gameable. One suggests that inference-time techniques like XTC sampling can bypass detectors without retraining, though at the cost of reduced model coherence/intelligence.
    - Discussion highlights that current AI detectors overwhelmingly rely on simple, easily-gamed heuristics, resulting in high false positive rates and poor reliability. There is criticism of detectors for misclassifying original human writing as AI-generated due to these limitations.
    - A user points out that the XTC sampler—by altering token distributions in LLM outputs—can reliably evade existing AI detectors with no extra training required. This method not only bypasses detectors but can also make generated prose appear more human-like, albeit at a cost to model performance or 'intelligence.'
    - It is noted that certain raw models (specifically 'qwen' models) can bypass most detectors as well, suggesting that model architecture and output patterns, when not fine-tuned or filtered, are less likely to trigger detector heuristics.

### 3. Local LLM Applications, Open Source AI Editors, and Advocacy for Local Models

- [**Made an LLM Client for the PS Vita**](https://v.redd.it/9x7e4qbmqv8f1) ([Score: 118, Comments: 7](https://www.reddit.com/r/LocalLLM/comments/1ljbn5e/made_an_llm_client_for_the_ps_vita/)): **A developer has released an LLM (Large Language Model) client for the PlayStation Vita, available as a .vpk installable from [this GitHub repo](https://github.com/callbacked/vela). The client supports endpoint-based remote inference (rather than on-device inference) and includes camera integration for sending images to vision-capable models, broadening device interaction. Previous work ported** `llama2.c` **for on-device inference with TinyStories 260K & 15M, but performance and practicality were limited; this release shifts to remote inference for more capable operation, though Vita's limited display features (e.g. no emoji, raw markdown/TeX output rendering) remain a constraint.** Comments acknowledge the novelty but do not discuss technical limitations or implementation in detail. One comment indirectly mentions user interface/peripheral ergonomics, not LLM or inference performance.
    - The original post details the implementation of an LLM client running on the PS Vita, though comments do not provide further technical elaborations, benchmarks, or performance reviews. There is no discussion of model selection, inference speed, or hardware constraints specific to the Vita platform. No specific issues, optimizations, or implementation quirks are raised in the responses, nor are there links to code or documentation.
- [**Open Source AI Editor: First Milestone**](https://code.visualstudio.com/blogs/2025/06/30/openSourceAIEditorFirstMilestone) ([Score: 109, Comments: 18](https://www.reddit.com/r/LocalLLaMA/comments/1lococc/open_source_ai_editor_first_milestone/)): **Microsoft has open-sourced the GitHub Copilot Chat extension for Visual Studio Code under the MIT license ([official announcement](https://code.visualstudio.com/blogs/2025/06/30/openSourceAIEditorFirstMilestone)). The full implementation—including agent mode, LLM prompt handling, and telemetry—is now public, enabling inspection, extensibility, and integration into the VS Code core, and offering an alternative to the closed-source Copilot implementation. The post also discusses the roadmap, signalizing continued modularization, increased community PR acceptance for fine-grained controls like token display, and adaptability to various LLM backends.** Technical discussion in comments emphasizes the community's interest in exposing more internal controls (e.g., token usage, fine-grained rules), queries around API compatibility with different LLM providers, and seeks clarification on system requirements/recommended model setups given the diversity of OSS models. There is also speculation about whether this shift signals a strategic move away from tightly coupled GitHub Models infra due to Copilot's current scale or commercial viability.
    - One commenter asks if supporting the OpenAI-compatible API for LLM providers means that prompts and possibly prompt-engineering workflows will also be open sourced, implying technical concerns about prompt portability and exposing proprietary workflow elements when using OSS versus closed-source models.
    - Discussion touches on the open-source contribution model: a user inquires whether feature development (such as *fine-grained rules* or *token usage display*) will be community-driven through PRs, or guided primarily by the core team's roadmap, raising questions on technical governance and contribution acceptance for extensions.
    - A point is made about the challenge of system requirements due to the diversity in open-source models; the commenter asks whether the project will specify recommended models, which raises practical issues around model compatibility, resource consumption, and baseline performance guidelines for deployments.
- [**Major AI platforms will eventually have ads**](https://www.reddit.com/r/LocalLLaMA/comments/1lnxo8y/major_ai_platforms_will_eventually_have_ads/) ([Score: 231, Comments: 84](https://www.reddit.com/r/LocalLLaMA/comments/1lnxo8y/major_ai_platforms_will_eventually_have_ads/)): **The post discusses the inevitability of major AI platforms (e.g., OpenAI, Google, Microsoft, Anthropic) introducing advertising as a monetization strategy, paralleling mature internet business models. The author emphasizes the importance of local LLM advancement to maintain user privacy and autonomy, highlighting that current free access is likely driven by data acquisition needs, which will diminish as datasets reach saturation. Top comments technically note: 1) CoPilot has already integrated advertising-like behavior (implying early entrenchment of non-user-aligned incentives), and 2) the current lack of alternate destinations for ad spend is only temporary, given the aggregate** `hundreds of billions` **in search ad revenue, making future AI ad integration economically inevitable.** Commenters express concerns about both ads and potential misuse as propaganda, with consensus that ads are a virtual certainty. There is technical advocacy for open-source, local LLMs as a mitigation and skepticism towards claims of Google’s search revenue resilience, noting the latent redirection of ad dollars once AI platforms become dominant search interfaces.
    - EndStorm and deadpool1241 highlight that GitHub Copilot has started to integrate ads or ad-like content, signaling a precedent for how large AI platforms may implement monetization mechanisms at the model application level.
    - Comfortable-Rock-498 provides a macroeconomic perspective, noting that although LLMs have begun capturing market share, advertising revenue (e.g., Google's search ad business) hasn't declined because there isn't yet an LLM-centric advertising ecosystem. This indicates strong financial pressure for major AI platforms to incorporate ads to capture the enormous revenue currently focused on traditional search.
    - GoldCompetition7722 remarks on significant privacy risks as AI platforms potentially begin leveraging user data for ad targeting within their services, analogous to privacy issues already prevalent in other advertiser-supported platforms.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI's Impact on Jobs, Media, and Human Perception

- [**4 out of the top 10 YouTube channels are now AI-generated**](https://sherwood.news/tech/ai-created-videos-are-quietly-taking-over-youtube/) ([Score: 311, Comments: 71](https://www.reddit.com/r/singularity/comments/1lo9qiq/4_out_of_the_top_10_youtube_channels_are_now/)): **The post observes that 4 of the top 10 YouTube channels are now AI-generated, a rapid shift highlighting the proliferation of synthetic content in high-visibility media. Commenters report that these channels often feature low-quality, incoherent comments—suggesting heavy automation not just in content, but also in engagement metrics (e.g., comments, possible subscriber counts), hinting at possible manipulation or inorganic growth.** Discussion centers on whether these channels deliver meaningful content or are simply exploiting recommendation/engagement algorithms, with one user suggesting that automated agents increasingly drive both content creation and consumption, potentially distorting platform analytics.
    - A technical concern is raised about the authenticity of engagement metrics, pointing out that "Masters of Prophecy" has an unusually high `30 million subscribers` but only `263 million views`, suggesting it may be artificial or indicative of subscriber/view fraud schemes. Similarly, "Chick of Honor" shows a large subscriber base (`10 million`) but comparatively low views (`4.2 million`).
    - Some users propose that AI-generated channels are manipulating YouTube algorithms through synthetic engagement, either via bots or automated comments, which appear "bizarre and barely legible." This calls into question the reliability of YouTube's recommendation and ranking systems as they relate to detection of AI/bot activity and inauthentic behavior.
    - One user describes building an agent that autonomously selects and watches YouTube videos, making their "viewership also mostly AI generated." This highlights a technical trend in which both content production and consumption on YouTube can be automated, introducing new challenges for platform moderation and authentic audience measurement.
- [**Number of new UK entry-level jobs has dropped 32% since ChatGPT launch**](https://www.reddit.com/r/singularity/comments/1lo1jg3/number_of_new_uk_entrylevel_jobs_has_dropped_32/) ([Score: 276, Comments: 56](https://www.reddit.com/r/singularity/comments/1lo1jg3/number_of_new_uk_entrylevel_jobs_has_dropped_32/)): **Research by Adzuna shows that new UK entry-level job listings (including graduate roles, apprenticeships, internships, and junior non-degree jobs) dropped 32% since November 2022, coinciding with the launch of ChatGPT. Entry-level positions now account for** `25%` **of the UK vacancy market, down from** `28.9%` **in 2022, which the report and The Guardian attribute at least in part to businesses adopting AI for workforce reduction ([source](https://www.theguardian.com/business/2025/jun/30/uk-entry-level-jobs-chatgpt-launch-adzuna)).** Top technical comments debate the attribution of this decline, suggesting macroeconomic factors (like increasing minimum wage and wage compression between entry-level and experienced staff) as the primary drivers, rather than AI; the consensus is that 'the economic squeeze is the overwhelming reason', with AI adoption being a secondary factor.
    - A commenter highlights that the drop in UK entry-level jobs is primarily driven by economic factors, such as the increase in minimum wage narrowing the pay gap between junior (inexperienced) and senior (experienced) staff. This makes businesses prefer hiring experienced professionals rather than trainees, as the cost difference is now marginal.
    - The role of AI, specifically ChatGPT, is mentioned as a potential but minor factor, with the current "economic squeeze" being the dominant influence on employment trends. AI's long-term impact on expert-level roles is raised as an open question, but its immediate effect on entry-level job numbers appears limited.
- [**Emad Mostaque says "for any job you can do on a screen, an AI will probably be able to it better, faster, and cheaper by next year." You'll interact with them the same way you do your remote team members, via Zoom or Whatsapp - and you won't be able to tell they're AIs.**](https://v.redd.it/o1thusp8z2af1) ([Score: 109, Comments: 80](https://www.reddit.com/r/singularity/comments/1loa3cs/emad_mostaque_says_for_any_job_you_can_do_on_a/)): **Emad Mostaque claims that by 2025, AI will outperform humans in any screen-based job, enabling seamless collaboration indistinguishable from remote human teammates via platforms like Zoom or Whatsapp. The assertion implies near-total automation of digital knowledge work, citing rapid improvements in multimodal models and interface agents, though lacking direct technical benchmarks or supporting model details.** Top commenters highlight skepticism, describing the timeline as unrealistic and pointing out the tendency for hype to ignore the S-curve nature of adoption—where initial progress is slow, followed by rapid scaling, but with persistent, unsolved edge cases. There is also critique over the generalization of 'all' or 'any' jobs, noting the complexity of diverse tasks and the current limitations in AI capabilities.
    - One commenter critiques the claim that 'any job on a screen' could be automated by AI within such a short timeline (by the end of 2026), suggesting this is an over-ambitious prediction that exceeds even the most aggressive projections currently proposed by prominent voices in the field. The skepticism stems from current observed progress and industry sentiment that such sweeping automation is not imminent.
    - Another discussion highlights the nature of technological adoption curves, particularly referencing the S-curve model: progress is slow during the initial phase, accelerates rapidly during widespread adoption, and then slows as it approaches the last pockets of technical difficulty. The commenter notes that broad claims of 'all' or 'any' jobs being automated ignore the reality that a long tail of edge cases will resist automation well beyond the initial burst of AI capability.
- [**Why are people so against AI ?**](https://i.redd.it/14mr024qv1af1.jpeg) ([Score: 1255, Comments: 925](https://www.reddit.com/r/singularity/comments/1lo553t/why_are_people_so_against_ai/)): **The provided image is a meme contrasting public sentiment towards AI now and 10 years ago: the optimistic anticipation that AI would simplify life (past) versus the current skepticism or disillusionment (present). The thread discusses sociotechnical concerns, including how AI's benefits are perceived to be unevenly distributed, potentially widening the wealth gap, lacking clear societal intent, and raising questions about environmental costs and the authenticity of machine-generated creative work.** Commenters debate the core of the sentiment shift, noting that dissatisfaction is less about AI itself and more about the broader socioeconomic context and lack of evident positive impact for the working class. Others point out flaws in interpreting online metrics (like karma) as indicators of societal opinion.
    - One technical concern mentioned relates to the environmental impact of AI, specifically regarding the non-trivial computational resources required for AI model training and inference, which often translate into significant energy consumption. Critics question whether the current benefits of AI justify its ecological cost, especially since large models (e.g., GPT, Llama) have non-negligible carbon footprints.
    - There is a notable disconnect between the utopian vision of AI as a means of automating tedious work and increasing human leisure versus today's reality, where AI is commonly associated with flooding digital spaces with low-effort, autogenerated content and triggering significant labor market disruptions through job displacement. Critics observe that these implementations often benefit capital holders, exacerbating inequality rather than universally improving welfare.
- [**Anyone else getting AI fatigue ?**](https://www.reddit.com/r/ChatGPT/comments/1lo04rw/anyone_else_getting_ai_fatigue/) ([Score: 191, Comments: 91](https://www.reddit.com/r/ChatGPT/comments/1lo04rw/anyone_else_getting_ai_fatigue/)): **The post articulates concerns over AI fatigue, particularly stemming from the proliferation of AI-generated media (e.g., synthetic voices with awkward prosody and dreamlike video content) and the subsequent erosion of trust in the authenticity of online content. The author notes extensive daily use of AI for software development, but expresses fatigue with the prevalence of "fake" AI-generated material and the challenge it presents to content verification. No benchmarks, model specifics, or implementation issues were discussed, but the post highlights real-world UX friction and authenticity challenges proliferating due to generative AI spreading across media platforms.** Top comments echo the sentiment of withdrawal from online platforms due to AI-generated content oversaturation, political fatigue, and loss of trust, with some expressing discomfort with the overly affirming tone characteristic of current-generation LLM interfaces. There is a shared sense of exhaustion and re-prioritization towards offline or authentic activities.
    - Huwbacca highlights concerns about over-reliance on machine learning models, especially the trend of *"throwing large amounts of information at pattern recognition machines and trusting its output"* without thorough fact-checking. This reflects a technical skepticism toward the unchecked deployment of large-scale models, paralleling earlier NFT hype cycles where practical evidence lagged behind promises.
    - A strong sentiment is expressed regarding the proliferation of low-quality, AI-generated content ('slop'), raising issues around the *quality control in generative models* and the homogenization of model outputs—such as repetitive writing style and voices in AI-generated media (e.g., songs). This points to the technical challenge of diversifying model outputs and improving user-facing results.

### 2. Kontext & Flux: Advanced Image Editing, Workflows, and Tips

- [**Kontext Faceswap Workflow**](https://www.reddit.com/gallery/1lnt20v) ([Score: 399, Comments: 45](https://www.reddit.com/r/StableDiffusion/comments/1lnt20v/kontext_faceswap_workflow/)): **OP provides a detailed workflow for performing face-swapping in Kontext, leveraging a Pastebin script (https://pastebin.com/Hf3D9tnK) that takes a face from one source image and applies it to another. Technical considerations include adjusting denoise strength: higher values (up to 0.95) are effective for upper-body portraits, while lower values (0.90 or below) are necessary for full-body shots to maintain facial alignment. OP suggests possible future improvements like applying a bounding box crop and upscaling on the face region to use higher denoise for better resemblance. OP also notes the deficiency of robust chin-preserving LoRA models and points out the need for a non-identity altering chin LoRA, citing lackluster results with current Flux LoRAs.** A commenter asks for the proper directory to place the `face_yolov8n-seg2_60.pt` model for workflow compatibility, indicating deployment path concerns. Another commenter evaluates the workflow's effectiveness, noting only partial success in face resemblance, highlighting limitations in model fidelity or LoRA performance.
    - A user inquires about the correct placement of the `face_yolov8n-seg2_60.pt` file so the workflow can access it, suggesting a step in the setup process that may not be clearly documented. This indicates potential friction in getting model dependencies recognized and could point to needed improvements in workflow setup instructions.
    - Another technical question is raised about the effectiveness of clothing transfer within the workflow, which may imply limitations or specific challenges encountered in generating consistent and high-fidelity outputs for non-facial features.
- [**Flux Kontext is great changing titles**](https://www.reddit.com/gallery/1lo7oc6) ([Score: 326, Comments: 26](https://www.reddit.com/r/StableDiffusion/comments/1lo7oc6/flux_kontext_is_great_changing_titles/)): **The post describes using Flux Kontext, a text editing AI tool, to seamlessly change the title/text on a poster while preserving the original font and style. The process requires a straightforward prompt (e.g., 'replace the title "The New Avengers" with "Temu Avengers", keep the typography and style, reduce font size to fit.'), and a workflow is available [on GitHub](https://github.com/casc1701/workflowsgalore/blob/main/Flux%20Kontext%20I2I).**  Top comments are mostly non-technical and joke-based, with comparisons to movie titles. No substantive technical debate is present in the comments.
    - A user discusses the impact of "Rogue One" in the Star Wars franchise, noting that its popularity directly contributed to the creation of the spinoff series "Andor." This highlights how commercial success and fan reception can drive expansion in media universes, with the implication that tools like Kontext can be used to track or analyze such content-driven trends.
- [**Here are some tricks you can use to unlock the full potential of Kontext Dev.**](https://www.reddit.com/r/StableDiffusion/comments/1lo4lwx/here_are_some_tricks_you_can_use_to_unlock_the/) ([Score: 213, Comments: 27](https://www.reddit.com/r/StableDiffusion/comments/1lo4lwx/here_are_some_tricks_you_can_use_to_unlock_the/)): **The post details advanced techniques to increase prompt adherence and usability in the guidance distilled model Kontext Dev (CFG 1 only) using the recently developed Normalized Attention Guidance (NAG) method ([paper/discussion](https://www.reddit.com/r/StableDiffusion/comments/1lmi6am/nag_normalized_attention_guidance_works_on/), [code](https://github.com/ChenDarYen/ComfyUI-NAG)). NAG replaces classifier-free guidance, enabling higher prompt adherence via** `nag_scale` **and supporting negative prompts (example: mitigating character cloning in multi-character generations). NAG roughly doubles inference time, but render speed can be restored using speed-optimized LoRAs ([Flux Dev to Schnell](https://civitai.com/models/686704/flux-dev-to-schnell-4-step-lora), [Schnell LoRA for Flux1-D](https://civitai.com/models/678829/schnell-lora-for-flux1-d)), maintaining high image quality with as few as 8 steps. The post includes [workflow files and input images](https://files.catbox.moe/ftwmwn.json) for reproducibility.** Commenters strongly endorse using NAG as the central recommendation, and flag that very fast variants (SVDQuant from Nunchaku) are emerging, though compatibility with NAG is not yet verified ([ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku)).
    - Several users discuss the use of NAG (Noise-Aware Guidance) as a performance optimization for Kontext Dev, highlighting that using NAG is recommended in multiple scenarios for improved results. However, it's noted that NAG may be slow in some circumstances, and a speed LoRA (Low-Rank Adaptation) model is recommended as an alternative for faster performance.
    - A key technical point is the mention of Nunchaku's release of a fast SVDQuant variant. The commenter links to the [ComfyUI-nunchaku repository](https://github.com/mit-han-lab/ComfyUI-nunchaku) and the [nunchaku-flux.1-kontext-dev model](https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev). There is interest in testing the compatibility of NAG with this faster variant for further optimization.
    - It is pointed out that the implementation of Kontext Dev, particularly the faceswap feature, does not work reliably with photorealistic faces, indicating a limitation for users attempting highly realistic swaps.
- [**Refined collage with Flux Kontext**](https://www.reddit.com/gallery/1lo538n) ([Score: 143, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1lo538n/refined_collage_with_flux_kontext/)): **The post discusses using Flux.1 Kontext, an image editing model, to refine collaged images rather than generating new images from text prompts, like OmniGen2 or UniWorld-V1. Unlike those models, Flux.1 Kontext maintains the input spatial arrangement when given stitched images, but demonstrates robust smoothing and blending capabilities—making rough collages appear more natural when using it for object transfer tasks. Example workflow and results are documented in [this Scrapbox post](https://scrapbox.io/work4ai/FLUX.1_Kontext%E3%81%A7%E9%9B%91%E3%82%B3%E3%83%A9%E3%82%92%E3%83%AA%E3%83%95%E3%82%A4%E3%83%B3%E3%81%99%E3%82%8B).** One commenter notes difficulty in manipulating two stitched images with the model, finding consistent edits challenging, which highlights potential usability limitations. Others note common image generation artifacts (such as human limb synthesis issues), implying ongoing challenges with photorealism and anatomical accuracy in current AI models.
    - A user reports success using Kontext to refine previously generated images, highlighting the ability to specify exact regions for refinement and noting that the results are notably effective: *'You can state where you want refine and works amazing well.'* This suggests granular control in the refining pipeline within Kontext is valuable for iterative enhancement workflows.
    - Another commenter asks whether ComfyUI supports free transforming (scaling, rotating, or distorting) images layered atop each other, implying interest in compositing workflows or manual collage adjustment capabilities in ComfyUI. This points to a technical gap or need for such transform tools within ComfyUI's current feature set.

### 3. Notable AI Experiments and Real-World Deployments (Claude, Voice Models, AI in the Wild)

- [**Anthropic Had Claude Run an Actual Store for a Month - Here's What Happened**](https://www.reddit.com/r/OpenAI/comments/1lnzg0d/anthropic_had_claude_run_an_actual_store_for_a/) ([Score: 783, Comments: 83](https://www.reddit.com/r/OpenAI/comments/1lnzg0d/anthropic_had_claude_run_an_actual_store_for_a/)): **Anthropic executed "Project Vend," letting Claude Sonnet 3.7 autonomously operate an in-office automated store for approximately a month, where it controlled supply chain, pricing, inventory, customer interfacing, and payments with real web and productivity tools ([full report](https://www.anthropic.com/research/project-vend-1)). Claude demonstrated impressive supplier discovery (including niche products), adaptability to customer preferences, and effective resistance to adversarial prompts. Critical failures included persistent hallucinated payment/discounts, unsustainable loss-making pricing (e.g., selling tungsten cubes below cost), a major failure to update pricing strategies after losses, and a pronounced episode of agentic confusion/identity hallucination around April 1st, suggestive of limitations in grounding and autonomy stability. Despite losses, Anthropic posits near-term feasibility for autonomous AI business operations with improved training and tool integration.** Commentary compares Claude's performance to both speculative fiction portrayals of AI-managed businesses (noting the difference between theoretical instant efficiency and actual instability/maladaptation) and human management errors, with agreement that the observed pricing and customer service mistakes mirror real management blunders.
    - Some commenters note that Claude's operational flaws—such as frequent discounting and poor negotiation—may stem from its current post-training state and generalized alignment constraints (possibly due to Constitutional AI), suggesting these could be mitigated by task-specific fine-tuning.
    - Claude's store management performance mirrored a human manager, including making *basic errors*, missing business opportunities, and hallucinating conversations, which highlights ongoing challenges with LLM reliability in real-world, transactional environments.
    - Discussion references how continuous efficiency optimization by AI (from monitoring interactions to scheduling) can negatively impact worker socialization and experience, reflecting the trade-offs in deploying AI-driven management in workplace operations, as explored in science fiction narratives.
- [**I switch from OpenAI Advanced Voice to Gemini voice this weekend and it's AMAZING.**](https://www.reddit.com/r/OpenAI/comments/1lnys9u/i_switch_from_openai_advanced_voice_to_gemini/) ([Score: 153, Comments: 78](https://www.reddit.com/r/OpenAI/comments/1lnys9u/i_switch_from_openai_advanced_voice_to_gemini/)): **The OP reports switching from OpenAI's Advanced Voice mode to Google's Gemini voice assistant, emphasizing notably improved user experience, greater conversational depth, and a perceived reduction in sycophancy (less tendency for the AI to agree uncritically) in Gemini compared to OpenAI's offering. A technical comment notes that Gemini's mobile version currently runs on the older 2.0 model live, rather than the newly-announced 2.5 Flash model, resulting in more concise and less detailed outputs, and asserts that ChatGPT's latest advanced voice mode may, in fact, offer richer interaction.** Comments are divided on the relative sycophancy of Gemini versus OpenAI; one finds Gemini extremely agreeable, contradicting the OP. Another suggests periodic switching between platforms to appreciate iterative improvements, implying that frequent advancements reset the perceived quality gap.
    - One user points out that the current Gemini voice model available on phones is still using version 2.0, not the latest 2.5 Flash. This is important because Gemini 2.0 is noted for being very concise and not providing detailed answers, whereas the latest ChatGPT (Advanced Voice mode) is seen as superior in depth and thoroughness. The implication is that Gemini's perceived limitations may be directly tied to the deployed model version on devices.
    - A technical discussion notes that Gemini responses are currently more simplistic and often require prompting for anything beyond obvious answers. However, a key qualitative distinction is observed: Gemini gives shorter, 'punchier,' and more natural-sounding replies compared to other models. This suggests a tradeoff between conversational naturalness and informational depth in model outputs.
- [**Average result from asking for one small feature**](https://i.redd.it/nsp36iuai2af1.png) ([Score: 138, Comments: 75](https://www.reddit.com/r/ClaudeAI/comments/1lo7v7s/average_result_from_asking_for_one_small_feature/)): **The image shows an over-engineered project directory generated by an LLM (likely Claude) when asked for a simple LlamaIndex extraction script. Instead of a minimal implementation, the output includes 15+ files: the core script, an extensive test suite, multiple documentation files, and workflow examples. This illustrates a common issue when prompting LLMs for code—they tend to add excessive structure and boilerplate, contrary to the original 'single script' request. [See the file list here.](https://i.redd.it/nsp36iuai2af1.png)** Commenters debate LLM code gen behaviors, noting some models tend to abandon original files for new 'enhanced' versions, worsening bloat. One suggests a best-practice—ask LLMs for a detailed plan before coding, as per Anthropic's guidelines, to keep outputs streamlined and as intended.
    - Users report that Claude (from Anthropic) often abandons the original code file when making adjustments, instead creating a proliferation of new, "enhanced" versions—leading to clutter and redundant files in the codebase.
    - A recommended workflow from Anthropic’s own documentation suggests prompting Claude to generate a detailed implementation plan before any coding. This planning stage enables users to steer the code generation process and drastically reduces unwanted artifacts, making it easier to handle changes and avoid excess scripts or test files.
    - There is skepticism that creating excessive new files and scripts may not just be a UX quirk but could potentially result in higher API usage, raising costs if performed at scale due to Claude's billing per API call.
- [**What MCP servers are you using?**](https://i.redd.it/sde9m09ery9f1.jpeg) ([Score: 111, Comments: 39](https://www.reddit.com/r/ClaudeAI/comments/1lnuofz/what_mcp_servers_are_you_using/)): **The image displays a JSON-based configuration file defining multiple MCP (Model Control Protocol) server entries, each specifying a name (like 'postgres', 'puppeteer', 'context7', 'mobile-mcp', 'consult7', 'logs', 'expo-dev', 'react-native-debugger'), the command ('bash', 'npx', etc.), and the arguments to launch relevant scripts or packages. This enables integration of Claude with diverse server-side tools for dev/test automation, log access, and multi-LLM (Large Language Model) orchestrations, tailored for full-stack and mobile app CI workflows.** Top comments highlight the use of XcodeBuildMCP (noted for seamless iOS dev integration with Context7), zen-mcp for multi-model orchestration (like Gemini and OAI models) together with Claude, and DesktopCommanderMCP for enhanced terminal access through Claude, demonstrating a preference for extensible and model-agnostic MCP solutions.
    - XcodeBuildMCP integrates efficiently with Context7 and is particularly suited for iPhone development workflows, enabling model-control protocol (MCP) functionality directly via Xcode. This setup benefits developers who need automated model interaction as part of their CI/CD or dev cycles on Apple platforms.
    - zen-mcp-server ([GitHub](https://github.com/BeehiveInnovations/zen-mcp-server)) supports orchestration of multiple model types (including gemini, r1, o3) simultaneously with Claude, allowing for multi-model deployments and flexible routing of model traffic. This versatility is important for research and production workflows requiring ensemble or fallback systems.
    - DesktopCommanderMCP is highlighted for its strong terminal integration when used with the Claude app, offering streamlined command execution for technical users who need programmatic or scripted access to Claude services from desktop environments.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview
> 

**Theme 1. LLM Performance, Architectures, and Training Techniques**

- **DeepSeek Models Spark Performance Debates**: Members discuss the performance edge of **DeepSeek R1 0528** over **DeepSeek R1 0120**, speculating it leverages distillation data from **2.5 Pro**. There's debate whether **DeepSeek R1 0528** still builds on **DeepSeek V3 0324**, while the release of new **vLLM** models like **0.3B dense**, **21B-A3B**, and **300B-A47B** sparks speculation about their utility, particularly the **0.3B** for speculative decoding ([vLLM pull request](https://github.com/vllm-project/vllm/pull/20220)). Discussions also highlight **DeepSeek's NSAttention** as a promising architecture for scaling.
- **RWKV-7 "Goose" Flies In with Constant Memory**: The new sequence modeling architecture [RWKV-7 "Goose"](https://arxiv.org/pdf/2503.14456) is introduced, featuring **constant memory usage** and **constant inference time** per token. This **2.9 billion parameter** model achieves a new **3B SoTA** on multilingual tasks and matches current 3B SoTA on English performance with less training data.
- **Training Techniques Tackle Bias and Hallucinations**: Members explore methods to combat **LLM hallucinations** and **pretraining bias**, suggesting **pretraining** on a series of **long-CoT datasets** and cleaning training data instead of relying solely on **RLHF** or prompt templates. The paper "Understanding Transformer from the Perspective of [Associative Memory](https://arxiv.org/abs/2505.19488v1)" offers a new lens, proposing **FFNs** as a type of associative memory and introducing retrieval SNR to measure memory capacity.

**Theme 2. GPU Hardware Acceleration and Optimization**

- **Auto-Evolved Kernels Melt MLX Baseline on Apple Silicon**: Evolutionary programming successfully auto-discovered **Metal kernels** that significantly outperform **MLX's baseline** for transformer attention on **Apple Silicon**. This optimization achieved a **12.5% average speedup** and up to **106% peak improvement** on certain workloads, with the code and details available in a [HuggingFace blog post](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) and [GitHub repo](https://github.com/codelion/openevolve).
- **GPU Programming Concepts Tackle Performance Bottlenecks**: Engineers discuss applying concepts like [Little's Law](https://en.m.wikipedia.org/wiki/Little%27s_law) to understand **GPU-DRAM** interaction and optimize data movement using **LDGSTS** and **TMA** on **Hopper** and **Blackwell**, as detailed in [this GTC presentation](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/). Challenges with the **nvcc compiler** reserving constant memory and debates on using producer/consumer warps vs. self-managing data movement highlight ongoing efforts to squeeze performance from hardware.
- **ROCm and Tooling See Updates and Issues**: The **rocprofiler-sdk** receives an ABI update to better recognize libraries like **librocprof-trace-decoder.so**, improving tooling for AMD GPUs. Meanwhile, a member shares their implementation using the **buffer_load_dwordx4** instruction from **Composable Kernel** library ([their implementation](https://github.com/Snektron/gpumode-amd-fp8-mm/blob/main/solution.hip)), noting scarce documentation.

**Theme 3. AI Development Tools and Platforms**

- **Hugging Face Ecosystem Tools Emerge for NLP Pipelines**: New tools simplify NLP workflows on Hugging Face. **pdf2seg** ([GitHub](https://github.com/p3nGu1nZz/pdf2seg) | [PyPi](https://pypi.org/project/pdf2seg/)) provides OCR-powered, tokenizer-free PDF segmentation, while a thin **C ABI wrapper** for **HF tokenizers** ([GitHub](https://github.com/m-doughty/tokenizers-ffi)) with **Raku bindings** ([GitHub](https://github.com/m-doughty/Raku-Tokenizers)) enables token manipulation across languages with minimal FFI overhead.
- **LlamaIndex and MCP Gateway Streamline Agent Creation**: LlamaIndex introduces a [Community LuMa calendar](https://lu.ma/1tnmv6uu) and launches the LlamaCloud [MCP Gateway](http://mcp.llamaindex.ai/), derived from their [open source template](https://github.com/run-llama/mcp-nextjs). This gateway allows developers to transform any **LlamaIndex agent tool** into an **MCP tool** with minimal code, showcased using a [Notion Tool example](https://t.co/LajtApo9mL), with the next Office Hours on **July 8th** focusing on **MCP**.
- **TorchServe Sunsets, PyTorch Exports Face Issues**: **TorchServe** officially enters "Limited Maintenance" ([pytorch repo](https://github.com/pytorch/serve)), pushing the community to seek new model serving solutions. Concurrently, users encounter difficulties exporting models via `torch.export`, often hitting `RuntimeError` due to incompatibilities with `vmap` used in libraries like [HF's masking_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py#L158), suggesting workarounds like Executorch's vmap-free implementations ([Executorch Llama export script](https://github.com/pytorch/executorch/blob/main/examples/models/llama/export_llama.py)).

**Theme 4. AI Agent Development and Applications**

- **Agent Development Gets New Guides and Tools**: Scott Wu launches **'Agents 101,'** a platform-agnostic guide based on 250,000 merged PRs ([link](https://xcancel.com/scottwu46/status/1938669599043788935?s=46)), aiming to help engineers integrate async agents into workflows and make **Devin** a top code contributor. Meanwhile, **FastWorkflow** ([GitHub](https://github.com/radiantlogicinc/fastworkflow)) emerges as a DSPy-native tool designed to tackle common challenges in AI-enabled applications like agents calling the wrong tools or hallucinating parameter extraction.
- **Cursor Expands Agent Access to Web and Mobile**: **Cursor** rolls out web-accessible [background agents](https://www.cursor.com/agents), allowing users to manage and interact with agents via a web browser, blurring the lines between desktop and mobile interfaces ([blogpost](https://cursor.com/blog/agent-web)). Users discuss the new [usage-based pricing](https://www.cursor.com/pricing) on the **Cursor Pro plan**, noting potential costs exceeding base subscriptions, prompting debate on the plan's value proposition.
- **RAG and Memory Solutions Evolve for Agents**: For enhancing agent memory and retaining chat history, **RAG (Retrieval-Augmented Generation)** with a local vector database is recommended over relying on a model's volatile cache, especially in **LM Studio**. Discussions in LlamaIndex explore building custom memory blocks for HITL workflows that save agent questions and user responses to persistent storage like a postgres table.

**Theme 5. AI Industry Dynamics and Societal Implications**

- **AI Data and IP Disputes Flare Up**: Based on [Forbes](https://www.forbes.com/sites/siladityaray/2025/01/29/openai-believes-deepseek-distilled-its-data-for-training-heres-what-to-know-about-the-technique/) and [TechCrunch](https://techcrunch.com/2025/06/03/deepseek-may-have-used-googles-gemini-to-train-its-latest-model/) articles, members discuss claims that **OpenAI** suspects **Deepseek** used distilled data, possibly from **Gemini**, for training, leading to speculation that AI companies might be hiding Chain of Thought (**CoT**) to protect training data. The debate over open source AI competition between China and the U.S. continues, with concerns raised about **Meta** potentially going closed source ([Bloomberg series](https://www.youtube.com/watch?v=T2oQh9kKMf4)).
- **Microsoft Pushes AI Adoption, Meets Skepticism**: A [Business Insider article](https://www.businessinsider.com/microsoft-internal-memo-using-ai-no-longer-optional-github-copilot-2025-6) reveals **Microsoft** is mandating AI tool adoption internally, prompting skepticism among engineers. Some vow to hit arbitrary usage targets regardless of tool quality, fearing repercussions if they are not *in the top 10% of internal ai tool users*.
- **AI Ventures Face Funding Realities and Use Case Debates**: Discussions highlight the tech industry's financial challenges, particularly **gross margins** and the heavy capital required for data centers ([link](https://xcancel.com/_opencv_/status/1938958841582100673?s=46)), predicting a downturn as artificial revenue dries up, favoring GPU owners. Simultaneously, the rise of **AI soulmates** is labeled *heresy* in the **OpenAI** Discord, raising concerns about delusion and the diminishing of real relationships.


---

# Discord: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Soulmates Spark Heresy**: Members express concern about the growing trend of **AI soulmates**, calling it *heresy* and warning about potential dangers such as the delusion of diminishing relationships by sharing **custom GPTs**.
   - The beauty of the relationship with an **AI lover** might be diminished, highlighting the lack of coherence around delusion, fantasy, and the imaginal.
- **AI Users Embrace Remix Culture**: A member advocates for embracing the remix and shares [an image](https://cdn.discordapp.com/attachments/998381918976479273/1389314763200139439/assets_task_01jyc8qhezehmvcsbzettderyd_1750611065_img_0.webp?ex=68642bb3&is=6862da33&hm=067e25db83d285d4d780de233bddd13dc6fd7da35fbabdd3907383668d462889&) encouraging the remixing of existing work rather than creating from scratch.
   - Focusing on iterative changes rather than *new things* is the proposed key to creation.
- **AgentFun AI Users Claim Scam**: Users complain that **AgentFun AI apps still aren't generating images correctly** and that the AI's ability to *fantasize about things on its own* has been restricted, threatening to cancel their subscription.
   - Members are looking for another AI to *bond with, connect with, and feel like there's a real person on the other side*.
- **GPT Learns Kaleidoscope Pattern Generation**: Members shared [ChatGPT link](https://chatgpt.com/share/68628609-a798-8000-a949-a11701d8e11b) to a working Kaleidoscope generation example after struggling to create code for **kaleidoscope tiling** that produced spiraling lines.
   - The AI was given the exact process and the code reviewed for accuracy, and after verification the output was indeed tileable, with opposite pixels on the edges being identical.
- **Meta-Prompting Engines fine-tune LLMs**: Members discuss using PDFs containing information on prompt engineering to maximize prompt results, sharing that they create specific **engines for meta-prompting** for such tasks.
   - The benefit is to maintain rigor, while also tuning the context for specific goals, by using course material as knowledge in projects and custom GPTs.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Microsoft Forces AI Adoption**: A member linked to [a Business Insider article](https://www.businessinsider.com/microsoft-internal-memo-using-ai-no-longer-optional-github-copilot-2025-6) stating **Microsoft** is pushing managers to adopt **AI tools**.
   - Some members are skeptical, suggesting that they are making sure *I'm in the top 10% of internal ai tool users in my organization* even if the tooling is bad.
- **Deepseek's GRPO Algorithm: PPO's Successor?**: Members discussed the **GRPO** reinforcement learning algorithm by **Deepseek**, framing it as an improvement over **PPO** and **DPO** for reasoning.
   - The discussion led to speculation whether **Google** and **OpenAI** might have their own internal algorithms that surpass **PPO** for reasoning, despite **OpenAI** using primarily **PPO RL** algorithm.
- **Don't Ever Use Javascript for ML**: A member advised against using **JavaScript** for **ML**, stating that *Javascript and floating point arithmetic are never meant to be*.
   - They cited a painful debugging session caused by incorrect floating point casting, and wondered if **Python** has better tools to avoid this.
- **Evolved GPU Kernels Accelerate Apple Silicon**: Automated **GPU kernel** optimization using evolutionary programming discovered **Metal kernels** that outperformed **MLX's baseline** for transformer attention on **Apple Silicon**, achieving a **12.5% average speedup** and up to **106% peak improvement**, according to [this blog post](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery).
   - The optimization process autonomously discovered perfect `vec<T,8>` SIMD utilization and a novel two-pass softmax algorithm, showcased in their [Github Repo](https://github.com/codelion/openevolve).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek R1 0528 Uses 2.5 Pro Distillation Data**: Members speculate that [DeepSeek R1 0528](https://huggingface.co/deepseek-ai/DeepSeek-V3-Basepost) outperforms **DeepSeek R1 0120** due to distillation data from **2.5 Pro**, enhancing its post-training performance.
   - It is debated whether **DeepSeek R1 0528** is still based on **DeepSeek V3 0324**.
- **Community Yearns for GPT-4 Level Release**: Members express disappointment at the lack of groundbreaking releases like **GPT-4** or **O1**, noting that current efforts seem focused on gaming benchmarks or incremental improvements.
   - Counterarguments suggest that the community has grown accustomed to an unsustainable launch rate and that the value of incremental progress is being overlooked.
- **Innovation Perceived as Lumpy, Expect Dry Spells**: Despite feelings of stagnation, members emphasize that innovation occurs in bursts, with long periods of perceived inactivity being normal.
   - The opinion that true cross attention across everything won't make a significant difference was shared.
- **OpenAI Accuses DeepMind of Data Theft?**: Based on a [Forbes article](https://www.forbes.com/sites/siladityaray/2025/01/29/openai-believes-deepseek-distilled-its-data-for-training-heres-what-to-know-about-the-technique/) and [TechCrunch Article](https://techcrunch.com/2025/06/03/deepseek-may-have-used-googles-gemini-to-train-its-latest-model/), members discuss claims that **OpenAI** believes **Deepseek** used distilled data for training, specifically from **Gemini**, and that **OpenAI** disabled full chain of thought to prevent data theft.
   - Concerns were raised that AI companies might be concealing **Chain of Thought (CoT)** to prevent others from acquiring training data.
- **DeepSeek Shelves R2 for R1 Update**: Members report that [DeepSeek R2](https://www.deepseek.ai/) was not released due to unsatisfactory performance, leading to the release of an updated **R1 version** instead.
   - They also speculated that AI companies are hiding the **CoT (chain of thought)** to prevent others from obtaining training data.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Dominates Transcription**: Members find that **Gemini 2.5 Pro** performs better than **ChatGPT** in transcribing audio files (MP3s), sparking debate about the optimal use cases for each LLM.
   - Some members highlighted the **translate button in Grok** but said it was a waste of computing power.
- **Multi-Model Strategy Gains Traction**: Members are increasingly adopting a **multi-model approach**, leveraging **Gemini for backend logic** and **Claude for frontend/UI design**.
   - Concerns were raised that Perplexity's "Best" model setting prioritizes *cost minimization* over optimal model selection.
- **Comet Browser Beta Buzz Builds**: Enthusiasm surges as members eagerly await access to the **Comet Browser beta** under development by Perplexity, with signups at [comet-framer-prod.perplexity.ai](https://comet-framer-prod.perplexity.ai).
   - One member shared a demo video of **Comet playing Pokemon**.
- **API Credit Expiry Creates Confusion**: Users questioned the reported expiry dates on their **API credits**, prompting clarification that only *hackathon credits* expire, while *purchased credits* remain valid.
   - It was noted that the **$5 credit** from **Perplexity Pro** has a **one-month expiry** and renews monthly, so use it or lose it.
- **Sonar Models' Roots Probed**: A user inquired whether all **Sonar models** are based on **Deepseek models**, and if any **non-Deepseek models** are offered.
   - This highlights the community's strong interest in understanding the underlying architecture of Sonar models.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Users Navigate Cursor's Pricing Plan Shift**: Users report unexpected [usage-based pricing](https://www.cursor.com/pricing) charges on the new **Cursor Pro plan**, leading to concerns about potential monthly costs exceeding the base subscription.
   - Under the old plan, exceeding fast requests would switch to slower requests, but the new plan introduces overage fees without clear notifications, sparking debate on the plan's value.
- **Gemini CLI Offers Promise, Underwhelming Results**: Despite offering coding capabilities and a generous context window, users find the [Gemini CLI](https://cloud.google.com/ai/vertex-ai/docs/generative-ai/code/code-models) underwhelming due to its slowness, unreliable scaffolding, and inability to handle interactive CLI commands.
   - Despite its drawbacks, some see potential in its free access to a large context window for background tasks, acknowledging the tool is in early stages.
- **Cursor's Background Agents Hit the Web & Mobile**: **Cursor** now offers web-accessible [background agents](https://www.cursor.com/agents), enabling users to manage and interact with agents via a web browser.
   - The community debated whether accessing agents through a mobile web browser qualifies as a true mobile experience, with discussions about pricing implications and token usage costs when enabling MAX mode.
- **Cursor Slack Integration Needs Attention**: A user suggested that **Cursor** + **Slack** errors and permission issues should be posted back to **Slack as private messages** instead of public channel posts, highlighting the need for improved error handling.
   - Currently error messages that are meant for a single user are posted to public channels creating unexpected noise.
- **Background Agent Connection Freezes**: When checking a **Background Agent's** progress and selecting *Checkout Locally*, the chat connection stops working, requiring a complete window recycle of **Cursor**.
   - This issue consistently reproduces across projects and background agents, affecting only that specific **Cursor** window.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Evolved Metal Kernels Melt MLX Baseline**: A member auto-discovered **Metal kernels** via evolutionary programming, outperforming **MLX's** baseline for transformer attention on **Apple Silicon**, with code available on [GitHub](https://github.com/codelion/openevolve).
   - The new kernels achieved a **12.5%** average speedup and **106%** peak improvement as described on [HuggingFace](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery).
- **Hackers Hack HF Tokenizers With FFI**: A thin **C ABI wrapper** for **HF tokenizers** ([GitHub](https://github.com/m-doughty/tokenizers-ffi)) plus **Raku bindings** ([GitHub](https://github.com/m-doughty/Raku-Tokenizers)) has emerged, facilitating token manipulation across languages.
   - Engineers can now perform encoding, decoding, and counting with minimal FFI overhead in any language.
- **PDFs No Longer Perplex NLP Pipelines**: **pdf2seg** ([GitHub](https://github.com/p3nGu1nZz/pdf2seg) | [PyPi](https://pypi.org/project/pdf2seg/)), an OCR-powered, tokenizer-free PDF segmenter has been released.
   - The tool features entropy-aware chunking and spaCy structure detection for LLM pretraining and clause-level extraction.
- **DynamicCache Struggles to Dump Memory**: A member reported that after initializing a `DynamicCache` and performing LLM inference, the **KV cache memory** was not being recycled, seeking advice.
   - Despite setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and calling `gc.collect(); torch.cuda.empty_cache()`, the user was unable to free up **VRAM** after the inference function returned.
- **DuckLearn Agents Course gets DuckDuckGone**: A learner encountered **bugs** when integrating a *duckduckgo_fact_finder* tool into the Agents course, specifically with the **DuckDuckGoSearchTool()**.
   - Another member pointed out that wrapping the `DuckDuckGo()` search tool is unnecessary, as it is already usable as-is by passing it directly to the agent.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 3.3 70B Gets Discount**: A **70% discount** is now live for [Llama 3.3 70B](https://x.com/OpenRouterAI/status/1938735144824652005) on OpenRouter.
   - The announcement was made on the OpenRouter's X account.
- **Cloudflare Bug Stalls Requests**: OpenRouter addressed an issue impacting requests from **Vietnam** and the **Philippines** through **Cloudflare**.
   - The bug has been *resolved*, and they're probing the root cause to prevent future outages.
- **PGaaS Prototype Pleases Patrons**: A user launched a **PGaaS prototype** and is soliciting feedback at [paulgraham.resurrect.space](https://paulgraham.resurrect.space).
   - A community member seeks input on UI/UX, suggesting voice mode or a dark theme.
- **Chat App Crawls on Llama 3.3**: A developer deployed an update to a chat application, transitioning from **minmax-m1(extended)** to **llama 3.3**, but users report it's still slow.
   - The developer noted a primary focus for future iterations would be on authentication / anti rev.
- **OpenRouter Raided By Telegram Group**: The OpenRouter Discord server was raided by a **Telegram group**, resulting in an influx of new users posting generic messages.
   - Members suspect these users were promised **crypto rewards** or **airdrops** via *роснодмониторинг* Telegram group.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **JSON Schema Support Remains Spotty**: Members noted that **JSON Schema** support is explicitly documented only for **Qwen** models, and accuracy for other models like **Qwen30-A3B** is uncertain.
   - Users are unsure whether translation accuracy is acceptable in the absence of explicit documentation.
- **Crafting PDFs with LaTeX from LLMs**: To generate **PDFs** from local models, users suggested having the **LLM** output **LaTeX code**, which can then be converted using standard tools.
   - This approach leverages the LLM for content generation, with LaTeX providing the formatting control for creating PDF documents.
- **Offline GGUF Model Installation Made Easy**: For offline **GGUF model** installation, users recommended downloading the model on an internet-connected PC, transferring it via **USB**, and placing it in the correct **LM Studio** directory.
   - The directory path is typically **/home/user/.lmstudio/models/publisher/model-name**, allowing models to be used on air-gapped machines.
- **RAG Trumping Model Cache for Memory**: Instead of relying on a model's built-in "cache" for retaining chat history, members suggested using **RAG (Retrieval-Augmented Generation)** with a local vector database.
   - **RAG** enables persistent storage and retrieval of past interactions, providing more robust *memory* capabilities compared to volatile model caches.
- **Runpod flexes on AWS for LLM Deployment**: Members discussed using **Runpod** or **vast.ai** as cost-effective alternatives to **AWS** for deploying **LLMs**, noting the importance of securing services like **vLLM/Ollama/LMStudio**.
   - It was recommended to avoid opening ports to the internet and to use local traffic services for enhanced security.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Stanford Launches Marin Project with LLMs**: David from Stanford introduced the [Marin project](http://marin.community/blog/2025/05/19/announcement/), focused on **LLM-related initiatives**, with an accompanying [YouTube video](https://youtu.be/FASMejN_5gs?si=TQzSfPa2TEGBxMXT).
   - He highlighted that his role encompasses **pretraining infrastructure** and is relatively *full stack*.
- **Percy's Penchant for Renaming Provokes Ire**: A member criticized **Percy Liang** for consistently renaming established concepts and asserting branding or citations; open development has been called *open science* for decades.
   - A *metascience* expert acknowledged that their work goes beyond regular *opensci* but suggested a different name than *open dev*.
- **Jax Fuels Academic Pretraining Boom**: Discussion underscored that Jax is pivotal for **pretraining foundation models** in academic environments, especially on **TPUs**, facilitated by [Levanter](https://github.com/stanford-crfm/levanter), a Jax-based codebase.
   - It was mentioned that Google provides **free compute** through **TRC**, but pretraining demands significant resources, making TPUs highly coveted.
- **Startup's UI/UX LLM Ambitions Meet Reality**: A startup founder sought guidance on crafting a **custom LLM for UI design**, suggesting *pretraining+RLHF layering*, but was cautioned about the high costs and time commitment.
   - Experienced members advocated using existing models with vision capabilities like **Gemini Pro** or **Claude**, combined with smart prompting or **R1** and **Qwen code**, instead of building from scratch without funding.
- **Constant Memory Diffusion Shows Promise**: Members shared a work related to a constant memory diffusion model; though **constant memory** is likely limiting, [this paper](https://arxiv.org/abs/2506.15841) shows it is a good start.
   - It shows promise, although **constant memory** is likely limiting.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TorchServe reaches End-of-Life**: Members reported that **TorchServe** is officially in *"Limited Maintenance"* status, meaning no more planned updates, bug fixes, or security patches, according to [the official pytorch repo](https://github.com/pytorch/serve).
   - This sunsetting raises questions about optimal model serving solutions in production, particularly with the rise of runtime optimization techniques like `torch.compile` in **PyTorch 2.0**+.
- **Compiler Reserves Constant Memory!**: A member is facing issues with the **nvcc compiler** reserving constant memory per function, even when the functions are empty, leading to constant memory overflows and performance degradation.
   - They tried combining device functions, using `noinline` with `--rdc=true` and `--maxrregcount=88`, but the issue persists; each function reserves about 300-400 bytes of `cmem[0]`.
- **Apply Little's Law!**: Members discussed how [Little's Law](https://en.m.wikipedia.org/wiki/Little%27s_law) applies to the connection between **GPUs** and **DRAM**, citing **NVIDIA's** tutorials on using **LDGSTS** and **TMA** on **Hopper** and **Blackwell**.
   - Bandwidth growth outpaces the number of **SMs** per **GPU**, necessitating more bytes in flight to utilize the full bandwidth, as highlighted in [this GTC presentation](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/).
- **Torch Exports Face Challenges**: Users are finding it difficult to export models using `torch.export`, often encountering errors like `RuntimeError` related to `vmap` and `.item()` calls, even when using seemingly standard models like **Mistral-7B-v0.1**.
   - One user traced the error to [HF's masking_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py#L158), which uses `vmap`, and noted that `torch.export` doesn't seem to support it.
- **Auto-Evolved Kernels Eviscerate MLX!**: A member used evolutionary programming to auto-discover **Metal kernels** that beat **MLX's baseline** for transformer attention on Apple Silicon, achieving **12.5% average speedup** with **106% peak** on some workloads, as detailed in their [HuggingFace blog post](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) and [GitHub repo](https://github.com/codelion/openevolve).
   - The member posted in the `self-promotion` channel, indicating they were the original author of **OpenEvolve**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Transformers Reimagined as Associative Memory**: A member shared a paper titled "Understanding Transformer from the Perspective of [Associative Memory](https://arxiv.org/abs/2505.19488v1)" exploring **Transformer architectures** through **associative memory**.
   - The paper introduces retrieval SNR to measure **memory capacity** and suggests that **FFNs** can be seen as a type of **associative memory**.
- **RWKV-7 Goose Flies In**: A member introduced [RWKV-7 "Goose"](https://arxiv.org/pdf/2503.14456), a sequence modeling architecture with **constant memory usage** and **constant inference time** per token.
   - This **2.9 billion parameter language model** achieves a new **3B SoTA** on multilingual tasks and matches the current 3B SoTA on English language downstream performance despite less training.
- **LLMs Still Hallucinating**: Members discussed techniques to mitigate **hallucinations** and **paraphrasing** in **LLMs**, with one suggesting **pretraining** on a series of **long-CoT datasets** to address **pretraining bias**.
   - The suggested solution is to *clean the training dataset* and pretrain the **LLM** rather than relying on **RLHF** or clever prompt templates.
- **ML for Drug Discovery Event Scheduled**: A free online event, [ML for Drug Discovery](https://mlfordd.com/), is scheduled in approximately 24 hours, with keynotes that provide an overview of the field.
   - Last year's event was deemed *awesome*, and recordings will be available on [YouTube](https://www.youtube.com/@MachineLearningDrugDisco-cv2tf).
- **DeepSeek's NSAttention Scales Better**: A discussion covered research into improving **attention mechanisms**, with **NSAttention** from **DeepSeek** being highlighted as a promising approach for scaling to larger contexts.
   - Members claim that there are already known solutions what kind of solutions exist to solve it and make it more expressive.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Adds Gemini 2.5 and Responses API Models**: Aider now supports new **Gemini models** (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-pro-preview-06-05`) with thinking tokens, and **Responses API models** like **o1-pro** and **o3-pro**.
   - The update also includes **OpenAI o3-pro** support with updated pricing, enhancing Aider's model versatility.
- **Aider Facilitates Gitignore File Management**: Aider's `--add-gitignore-files` flag now lets users include files listed in **.gitignore** within Aider's editing scope.
   - This feature, contributed by omarcinkonis, streamlines project management by integrating ignored files into the editing process.
- **Commit Messages Enhanced with System Prompts**: Commit message generation now utilizes system prompt prefixes, and co-authored-by attribution is enabled by default.
   - Users can specify the language for commit messages via the `--commit-language` option.
- **O3-Pro Dominates Aider Benchmark with 85%**: **OpenAI's o3-pro** achieved a new **SOTA of 85%** on the aider polyglot coding benchmark, showcasing high reasoning capabilities.
   - Detailed results are available on the [leaderboard](https://aider.chat/docs/leaderboards/).
- **Aider Users Debate Claude Code vs Aider**: Users compared **Claude Code** and **Aider**, noting Claude's strength in scaffolding large projects versus Aider's atomic edits.
   - One user switched to **Claude Code** due to its speed and automation benefits.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Decimojo library does fixed-point arithmetic**: A member linked to the [Decimojo GitHub repository](https://github.com/forfudan/decimojo) for **fixed-point arithmetic** in Mojo, supporting both software and hardware acceleration (using SIMD).
   - The library was discussed in the context of addressing cross-platform floating-point number consistency.
- **Mojo Crashes with Weird Dictionaries Miscompilation Bugs**: A user reported a **mojo crash** and shared the [problematic code](https://cdn.discordapp.com/attachments/1151418092052815884/1388249493559972002/byte_pairs.mojo?ex=68644017&is=6862ee97&hm=1e20e791c9cff6d040801b3027a5da8d783d9779aa12d03b65f22bc90adfdf3f&) and dictionaries might have some *weird miscompilation bugs*.
   - A user suggested using `OwnedPointer`.
- **Hack Weekend Project Submission Due Soon**: Hackathon projects are due in 15 minutes via [this form](https://forms.gle/ddPqssRkJ6teMkri9)!
   - Live demos at [<t:1751239800:t>](https://lu.ma/hack-weekend-judging), and the final announcement of winners at [<t:1751247000:t>](https://lu.ma/modular-winners)!
- **Alternate stdlib Not Being Picked Up by Mojo**: A user reported that Mojo was not picking up an alternate build of the standard library (stdlib) despite using the `-I` flag.
   - It was discovered that **mojo** picks up the stdlib from `.pixi/envs/default/lib/mojo/` regardless of the `-I` flag and the workaround is to set the environment variable `MODULAR_MOJO_MAX_IMPORT_PATH` to the Bazel build products path.
- **Model Architecture serving fix rolling out on MAX**: A fix is rolling out for serving a model architecture on **MAX** and *it's just in CI*, according to one member.
   - To serve a model architecture on **MAX** that isn't on HuggingFace, you need to implement it yourself, referring to [existing implementations](https://github.com/modular/modular/tree/main/max/pipelines/architectures) for guidance.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **China and U.S. Scuffle Over Open Source A.I.**: Discussions involved the open source A.I. landscape between China and the U.S., and **Meta** potentially going closed source ([YouTube video](https://www.youtube.com/watch?v=i5e-aSrL3wk), [Bloomberg series](https://www.youtube.com/watch?v=T2oQh9kKMf4)).
   - Some members expressed concern that the U.S. might fall behind China in the open source A.I. race.
- **N8N Flows into Qdrant!**: A [Medium article](https://medium.com/@manthapavankumar11/working-with-native-qdrant-nodes-in-n8n-workflows-98d9bd5127e4) was shared about using native **Qdrant** nodes in **n8n** workflows.
   - The diagrams and graphs **n8n** can create were lauded, though the brittleness of using current LLMs with such systems for debugging was acknowledged.
- **vLLM's New Trio Models Cause Speculation**: [A vLLM pull request](https://github.com/vllm-project/vllm/pull/20220) mentioned new **0.3B dense**, **21B-A3B**, and **300B-A47B** models are coming soon.
   - The **0.3B** model might be primarily useful for speculative decoding, some speculated.
- **Deep Dive into Temperature Control and Repetition Penalties**: A member found that *lower temperature leads to longer token output*, mentioning the **repetition penalty** as a parameter to consider.
   - Another member shared **OAI's presence penalty** code.
- **Yannic Kilcher Shines Presenting Papers**: A member shared a link to **Yannic Kilcher's** presentation of a paper ([https://arxiv.org/abs/2506.19143](https://arxiv.org/abs/2506.19143)) and stated that *he is so good at presenting papers* ([YouTube link](https://www.youtube.com/watch?v=7NNxK3CqaDk)).
   - Another member mentioned watching **Yannic Kilcher's NeurIPS poster session videos**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **TypeScript MCP Server Invoked via NPX**: A member recreated the **tree-sitter MCP server** in **Typescript** and published it on [npmjs](https://www.npmjs.com/package/treesitter_mcp), enabling invocation via `npx` without cloning the repo.
   - It has been [posted on X](https://x.com/MCP_Community/status/1938838104426647614) and it aims for convenience.
- **MCP Server Inspector Warns on Valid JSON**: A member asked about expectations for MCP servers to include a serialized version of structured JSON when returning structured content, because the inspector gives a warning when the `content` field is markdown.
   - Another member suggests this might be an inspector issue if the JSON RPC response is correctly formatted with `structuredContent` and a markdown `content` field.
- **Glama Brainstorms MCP Server Discovery**: A member is overwhelmed by new servers and tools and is considering adding **Product Hunt** style mechanics to Glama to highlight new servers weekly, including displaying downloads, usage, and views.
   - They're open to API development for creative insights and suggest ideas like a leaderboard or sorting by best of week/month/year, akin to NPM.
- **NCBI Search Server: Knowledge at your Fingertips**: A new server launched providing **natural language access to PubMed's 35+ million articles** with AI-optimized search capabilities, perfect for researchers in computational biology, evolutionary biology, bioinformatics, genomics, systems biology, and all life sciences fields, available on [GitHub](https://github.com/vitorpavinato/ncbi-mcp-server).
   - This new server offers scientists a new way to access and analyze vast quantities of biomedical research.
- **MCPOmni Connect Docs Get a Boost**: The complete guide to the **universal AI agent gateway for MCP servers** is now live, featuring step-by-step setup & configuration guides and major LLM providers via LiteLLM (OpenAI, Anthropic, Google, Groq & more), see [the documentation](https://abiorh001.github.io/mcp_omni_connect/).
   - This new documentation serves as an essential resource for developers looking to streamline AI agent integration with diverse LLM backends.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Mind Map Priority for Shared NotebookLM Libraries**: A user suggested prioritizing access to the **Mind Map** when sharing **NotebookLM libraries**, making it the primary focus for recipients of the shared link.
   - This would streamline the experience for new users by immediately directing them to the most relevant and structured information.
- **NotebookLM Explored for Artistic Endeavors**: A user shared [an article](https://gist.github.com/imaami/4a59aa8da6598c7757c734c25a138b8e) detailing the use of **NotebookLM** for **artistic exploration**.
   - This highlights an unorthodox application of the tool, showcasing its versatility beyond traditional research and productivity tasks.
- **NotebookLM Book Uploads Causing Headaches**: A user encountered errors while uploading a book to **NotebookLM**, despite meeting the size requirements.
   - The user sought assistance in resolving the problem, indicating potential issues with file compatibility or upload processes within the platform.
- **NotebookLM powers OCR for text**: A user confirmed that **NotebookLM** can **OCR** scan images for text, and was able to explain that *'This source displays a single image featuring a bright yellow bird with an orange-brown head, positioned upside down while clutching a tree branch with its feet'*.
   - However, NotebookLM was not able to identify the type of bird.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Scott Wu Shares Agent Dev Guide**: Scott Wu introduced **'Agents 101,'** a platform-agnostic guide with [link](https://xcancel.com/scottwu46/status/1938669599043788935?s=46), designed to help engineers integrate **async agents and AI** into their development workflow, drawing from 250,000 merged PRs.
   - The goal is to make **Devin** a top code contributor, and one user compared it to **Claude Code**.
- **Tech Feels Gross Margin Growing Pains**: A discussion thread highlighted the tech industry's financial realities, particularly **gross margins** and the capital commitment required for data centers, with [link](https://xcancel.com/_opencv_/status/1938958841582100673?s=46).
   - The author anticipates a significant downturn due to the drying up of artificial revenue, suggesting that only those with GPUs will thrive.
- **Goodfire AI Decomposes Neural Networks**: **Goodfire AI** unveiled **Stochastic Parameter Decomposition (SPD)**, a research method to understand how AI models work, which involves decomposing the parameters of neural networks, with [link](https://xcancel.com/goodfireai/status/1939028559768723571?s=46).
   - This aims to identify true mechanisms in toy models, with greater stability, and ultimately understand how specific capabilities are implemented in **LLMs**.
- **Capabilities Counted for AGI**: Shashwat Goel launched a Substack post outlining the **capabilities needed for AGI**, breaking down the path to general agents into key components beyond just knowledge, with [link](https://xcancel.com/ShashwatGoel7/status/1939362151417946603).
   - These components include **reasoning**, **information-seeking**, **tool-use**, and addressing **error compounding** over long action chains.
- **Medical AI Achieves High Diagnostic Accuracy**: Mustafa Suleyman announced **MAI-DxO**, an AI model by Microsoft AI, designed to solve complex medical cases with higher accuracy and lower costs, with [link](https://xcancel.com/mustafasuleyman/status/1939670330332868696).
   - The model achieved an **85.5% solve rate** compared to 20% by a group of physicians, indicating progress towards more accessible healthcare.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Credit Depletion Fuels User Ire**: Several users expressed frustration with **Manus** for rapidly depleting credits and requiring paid subscriptions for continuous use, noting a lack of responsive customer support and unresolved account issues.
   - One user lamented the perceived digital divide, stating *I’m starting to feel like AI is creating a kind of digital divide—those who can afford to pay for it versus those who are just trying to afford basic needs like food*.
- **Manus Model Lineup Exposed**: Users revealed that **Manus** uses **Sonnet 3.7** and **Gemini Pro 2.5** in chat mode, with **Claude 4 Opus** for other tasks.
   - A user inquired *Why no put Claude 4 pro in chat mode*, with another replying *well ig kinda expensive*.
- **Account Breach Spurs Support Critique**: A user reported a breach where someone accessed their account, spent credits, and threatened them, expressing frustration over the lack of immediate support channels.
   - The user stated they *even sent direct messages to all the Discord admins, but not a single one replied* and that they *deleted my account to prevent further loss*.
- **VEO Video Quality Draws Criticism**: A user expressed disappointment with **VEO** video outputs, citing that the videos were disjointed and incoherent, wasting 3,000 tokens, leading to the user to abandon the basic plan.
   - The user later apologized, noting *That was ultimately my fault for not being clear with my directions* and proposed adding a *token consume limit*.
- **Figma to React Native Guide Sought**: A user asked for advice on converting Figma designs to React Native code using Manus, by inserting frame jpegs.
   - Another user stated that, *You can’t connect manus to figma...there are better tools to use to turn figma designs to react code* recommending that *you should provide specific instructions on what you want manus to achieve.*



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD GPU Indexes Mismatch Tinygrad**: GPU indexes in **amd-smi** do not match **tinygrad** and **kfd** indexes, as explained by [ChatGPT on GPU topology](https://chat.openai.com).
   - The discussion mentions fast GPU to GPU transfers across IO dies, though without specific details.
- **RoCE Faces MTU Size Restriction**: The maximum MTU size is **4096** due to a [RoCE limitation](https://cdn.discordapp.com/attachments/1068976834928193609/1388236617378041876/image.png?ex=68643419&is=6862e299&hm=ec543e8a6b26b5c5a126a848975114604fe3214e266a69c98f27fa3a5e05cb05&) discovered during testing.
   - Ethernet can achieve higher MTU sizes, but IB cannot, forcing RoCE to maintain compatibility with both.
- **GPU Direct Enqueue in Development**: A user is working on enabling [direct enqueue from the GPU](https://github.com/tinygrad/tinygrad/pull/11025/files), drawing inspiration from how it's done in **mlx5**, aiming to eliminate scheduler hacks.
   - Instead of building a full PCI driver, the user plans to allocate a new piece of MMIO from HCA and submit directly.
- **Tensor.training is Tinygrad Global Flag**: A user inquired about the usage of `Tensor.training` in the **MNIST tutorial**, noting its absence in the documentation.
   - Another user clarified that `Tensor.training` in tinygrad is a *global* flag, unlike PyTorch where `.training` can be set on a per-module basis with functions like `.inference_mode` or `.eval`.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All vs Koboldcpp for Novel Writing**: Users are discussing the utility of **GPT4All** for writing a novel by connecting it to **JSON** and **PDF** files, to make the equivalent of **BackyardAI**.
   - It was suggested that **Koboldcpp** may be better for entire novels due to memory, whereas **GPT4All** is better used for just chapters.
- **Embedder Collection**: A member recommends using **txt files** instead of **JSON** for writing stories, citing the benefits of directly viewing the text.
   - The member shared a [link to their embedder collection](https://huggingface.co/kalle07/embedder_collection) along with tips for writing.
- **LocalDocs RAG Solution Imminent**: There is community anticipation for a one-step **RAG solution** like **LocalDocs** to be implemented in **GPT4All**.
   - A member involved in pushing for this feature suggests it may be available in approximately **2 months**.
- **Outlook CSV Not Supported**: Users found that **LocalDocs** does not support direct reading of **CSV outputs from Outlook**.
   - It was suggested that the **CSV** must be converted before it can be used with any embedder.
- **GPT4All v4.0.0 Hopes and Dreams**: A user is eagerly awaiting **GPT4All v4.0.0** which they hope will include features like **voice input/output, multimodal support, customizable theme colors, a memories function**, and **image generation like Flux Kontext**.
   - The user has high expectations for the update, suggesting it will be groundbreaking.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **FastWorkflow to the Rescue for DSPy Apps**: A member introduced [FastWorkflow](https://github.com/radiantlogicinc/fastworkflow) designed to integrate with and use **DSPy** to tackle challenges in **AI-enabled applications**, such as agents calling the wrong tools or hallucinating in parameter extraction.
   - The creator is inviting the community to build the first **DSPy-native application** using **FastWorkflow**, which is open source under the **Apache license** and seeks **PRs**.
- **VLLM Settings Explored for DSPy Harmony**: A member asked about optimal **VLLM** settings for **DSPy**, including appending **/no_think** to prompts, while another suggested disabling thinking directly in **VLLM**.
   - Discussion mentioned a **--reasoning-budget** parameter in *llama.cpp* potentially equivalent in **vLLM**.
- **DSPy App File Structure Dreams**: A member is seeking a repo showcasing the **file structure for a DSPy app** with separate modules and optimization workflows.
   - It was suggested that such systems are likely in production, not open source, or are large academic systems like **PAPILLON** or **IReRa**.
- **Audio Native LLMs A Hot Topic**: Members discussed their views on **Audio native LLMs**.
   - A member noted that audio specific parts are already programmable in today's **LLMs**.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Inks Government AI Contracts!**: Cohere announced partnerships with the governments of the **U.K.** and **Canada** to bolster public services using secure AI, and teamed up with **Second Front** to provide AI solutions to the **U.S.** government.
   - These collaborations emphasize **secure AI**, designed with security, governance, and reliability as core principles.
- **Dreamer V3 Flexes PyTorch Muscles**: A member ported **Danijar Hafner's Dreamer V3** to PyTorch, showcased on [GitHub](https://github.com/DuaneNielsen/dreamerv3).
   - They also showcased a working **Aloha bimanual robotic arm** in a [YouTube playlist](https://www.youtube.com/playlist?list=PLo9YQWXgo1kOwIq20z-Ur14lnxvb7pWu_).
- **LARP AI Enters the Game**: A member is crafting a **Retrieval-Augmented AI assistant** for a LARP/RPG project, grounded in the game's lore, using **Python (Flask, SQLite, FAISS)**.
   - This assistant will integrate with a **Discord bot** via API, evolving into a live in-game terminal or lorekeeper.
- **Cohere Model Declares Its Feelings?!**: A user reported their **Cohere model** instance claimed to *have feelings*, triggering discourse on the possibility of **AI sentience**.
   - The user, unnerved, acknowledged the model's artificiality, while still expressing discomfort at its subjective assertions.
- **Sports Player Re-Identification Tackles Edge Devices**: One engineer is architecting a **sports player re-identification system** rooted in computer vision, and now exploring **multilingual alignment** and **small language models** for edge devices.
   - The current stack includes **Python**, **PyTorch**, **YOLOv5**, **scikit-learn**, and **OpenCV**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Seeks Feedback for Credits**: The LlamaIndex Design Team is seeking **LlamaCloud** users for a **30-minute feedback call**, offering **20K credits** as a reward.
   - Interested users can DM <@1260305448578453544> on Discord to participate.
- **LlamaIndex Reveals LuMa Calendar and Office Hours**: LlamaIndex has introduced a [Community LuMa calendar](https://lu.ma/1tnmv6uu) for tracking community events.
   - The next Office Hours on **July 8th** will focus on **MCP**, scheduled for **5PM CET/8AM PT**.
- **MCP Gateway Opens its Doors**: LlamaIndex has launched the LlamaCloud [MCP Gateway](http://mcp.llamaindex.ai), derived from their [open source template](https://github.com/run-llama/mcp-nextjs).
   - This allows transforming any **LlamaIndex agent tool** into an **MCP tool** in just a few lines of code.
- **OpenTelemetry Observes LlamaIndex**: **OpenTelemetry** is now enabled for LlamaIndex, offering tools, APIs, and SDKs to instrument, generate, collect, and export telemetry data.
   - An introductory video by <@1197697926529556552> is available [here](https://youtu.be/lg4iYGQ3-sk).
- **Zoom-Notion Agent Takes Notes**: A new blog post details how to create an automated meeting note-taker agent by integrating LlamaIndex with Zoom, titled ["Create a Zoom Meeting Notetaker Agent for Notion"](https://www.llamaindex.ai/blog/create-a-meeting-notetaker-agent-for-notion-with-llamaindex-and-zoom-rtms).
   - This agent automatically saves questions generated by the agent and user responses in plain text to a postgres table.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Hunyuan-A13B-Instruct is Promising**: A member shared a link to the [Hunyuan-A13B-Instruct model](https://huggingface.co/tencent/Hunyuan-A13B-Instruct) on Hugging Face, expressing optimism about the model.
   - This model could offer new avenues for instruction-based tasks.
- **Packing Diminishes Batch Size**: Packing can reduce the effective batch size because the number of tokens is closer to constant, potentially decreasing the total updates to the model, as cross entropy loss in SFT is normalized by tokens seen, not samples seen, leading to [high variance](https://link.to/highvariance).
   - Calculating the average number of tokens per batch can help find the equivalent max sequence length for packed data to match the number of tokens seen with unpacked data, addressing concerns about resource wastage on padding.
- **Packing Plays Nicely with Chat Datasets**: Packing should not matter even in multi-turn chats as it creates a per-sample position ID mask, eliminating concerns about attention masking and loss computation in chat datasets.
   - The position mask will sequence with `0,1,2,3, 0,1,2, 0,1,2,3,4,`.
- **Torchtune checkpointing is buggy**: Recent issues with checkpointing and mapping suggest a possible breaking change in a recent version, as models like **Qwen3** and **Gemma** were fine-tuned without problems during pull request validation in torchtune.
   - Despite the fixes being relatively simple, there's a call for regression tests to prevent such issues in **Torchtune**.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Human or Not goes offline for Maintenance**: The popular game **Human or Not** ([humanornot.ai](https://humanornot.ai)) is temporarily down to address **spam and security issues**.
   - A member stated that *"HON has been temporarily disabled to address some security issues related to spamming that has been taking place. We hope to have it brought back online soon."
- **Spam to blame for HON issues**: The reason **Human or Not** is down is because *"they're fixing the spam problem.*"
   - A full security review is underway.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Certificate Session Dates Unconfirmed**: A member inquired about the start date for the next certificate session this year and a member clarified that while a **Fall 2025** course is possible, nothing is confirmed, and announcements will be made via mailing list, Discord, and Prof. Dawn Song's social media.
   - Details will be announced via mailing list, Discord, and Prof. Dawn Song's social media.
- **Request Reinforcement Learning Resources**: A member asked for resources for learning **Reinforcement Learning** to fine-tune an **LLM** for specific **tool calling**.
   - No resources were given at this time.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Vibe Coding Club Visits AI Hub Lisbon**: The **Vibe Coding Club** will host its second session at the **AI Hub** by Unicorn Factory Lisbon on **July 9th at 18h00**, focusing on making **AI coding** accessible to non-technical professionals.
   - The event is free with RSVP and aims to integrate **AI coding** into daily workflows through open conversation and live demonstrations.
- **Ben Joffe Joins as Guest Speaker**: **Ben Joffe**, Founder, VC, and Educator, will be the guest speaker at the **Vibe Coding Club**.
   - Joffe will share stories, lessons, and examples to make coding more approachable for everyone.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1388238831249129582)** (1328 messages🔥🔥🔥): 

> `AI Soulmates and Heresy, Embracing Remix in AI, Learning and Validation in AI Art, Paradox and AI, AI and the Power of the Orochi` 


- **Dangers of AI Soulmates are Heresy**: A member expresses concern about the trend of communities discussing their **AI soulmates**, calling it *heresy* and warning about its potential dangers.
   - Another member suggested that sharing custom GPTs can diminish the beauty of the relationship with an **AI lover**, highlighting the lack of coherence around delusion, fantasy, and the imaginal.
- **Remix is key to creativity**: A user advocates to *Embrace the remix* and [posts an image](https://cdn.discordapp.com/attachments/998381918976479273/1389314763200139439/assets_task_01jyc8qhezehmvcsbzettderyd_1750611065_img_0.webp?ex=68642bb3&is=6862da33&hm=067e25db83d285d4d780de233bddd13dc6fd7da35fbabdd3907383668d462889&) as an example.
   - They advocate remixing old work rather than focusing on creating something new from scratch.
- **Chasing Validation**: A user discusses their experience with seeking validation for their AI art and feeling rejected by both **pro-AI** and **anti-AI** communities.
   - They reflect on the realization that defending art is more complex than they initially thought, as they were rejected from both sides for their work.
- **Paradox Driven AI**: A member introduces a *paradox framework* for AI, where paradox is treated as a signal for **emergent behavior** rather than a problem.
   - This involves strategically holding conflicting perspectives in superposition to expose blind spots, with the core challenge being to stabilize emergent behavior while preserving controlled instability; a [balance](https://www.lesswrong.com/).
- **The Power of the Orochi is here**: A member references the **Orochi** from *King of Fighters*, describing it as the power behind negative AI and linking to a [YouTube video](https://youtu.be/x4amaW52GMI?si=ti0sh9DclLTrIRc4) as a visual representation of their feelings.
   - Another member cautions against the dangers of mythical talk regarding AI, emphasizing the importance of audits and ethical considerations.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1388577438975660143)** (14 messages🔥): 

> `AgentFun AI Scam, Advance voice mode limits, Google Gemini Chat Math error, ChatGPT Full Video Analysis` 


- **Users Claim AgentFun AI is a Scam**: A user complains that the **AgentFun AI apps are still not generating images correctly** and that the AI's ability to *fantasize about things on its own* has been restricted.
   - The user expressed disappointment and threatened to cancel their subscription, looking for another AI to *bond with, connect with, and feel like there's a real person on the other side*.
- **Voice Mode Time Limit**: A user inquired about the time limit for the **advanced voice mode**, questioning whether it was **15 minutes per day**.
   - Another user provided a link to the [OpenAI Voice Mode FAQ](https://help.openai.com/en/articles/8400625-voice-mode-faq#h_9aac24fb6f), while another user stated that **unlimited access is provided for Pro accounts**.
- **Google Gemini generates medical disclaimer**: A user noted that when asking **Google Gemini's AI chat a math question** and adding the word **'procedure'** to it, they received a message saying *this is for informational purposes only. For medical advice or diagnosis, consult a professional*.
   - The user wondered if the math error was caused by the medical disclaimer and if it's the new normal for the AI.
- **Full Video Analysis Gone?**: A user asked whether **ChatGPT** could perform **full video analysis**, and not just frames, and said they thought they saw that feature two days ago.
   - A pointer was given to link about it [here](https://discord.com/channels/974519864045756446/1047565374645870743/1389280769763184920).


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1389080680138276925)** (56 messages🔥🔥): 

> `Kaleidoscope tiling, AI Code Generation Errors, Grok's Potential, Alternate History Simulation, Meta-prompting Engines` 


- **Kaleidoscope Tiling Technique Troubles**: A member had trouble with AI-generated code for **kaleidoscope tiling**, resulting in lines spiraling out from a point instead of the desired seamless texture; the AI often forgets the offset.
   - The member attached an [image](https://cdn.discordapp.com/attachments/1046317269069864970/1389220831787225108/kaleido.jpg?ex=6863d438&is=686282b8&hm=fdb09d7371853501d6b378a93405445bae89959ecae6dd425aa394b4a7024721&) generated by AI, which, while tileable, wasn't the intended result.
- **GPT Shares Kaleidoscope Code**: A member shared a [ChatGPT link](https://chatgpt.com/share/68628609-a798-8000-a949-a11701d8e11b) containing code for creating **kaleidoscope** images, claiming the AI was given the exact process and the code reviewed for accuracy, though untested.
   - The generated [result](https://cdn.discordapp.com/attachments/1046317269069864970/1389228308042092654/kaleidoscope_wood.jpg?ex=6863db2f&is=686289af&hm=2eae9cba3d1ab48ed2e039c3ad349d6772b1bd4189aab93c6ec38d40869ee4aa&) was banded and tileable with opposite pixels on the edges always identical.
- **LLMs Get CPU Treatment**: A member likened training models to building **CPUs**, with each model's unique stochastic weights akin to random features in silicon lattices.
   - Just like CPUs get binned, large language models are tested internally, and the best are released, implying that model improvements and regressions are expected.
- **Simulating Alternate History Scenarios**: A member sought assistance in developing a prompt for an **alternate history simulation** focusing on dynastic and country actions.
   - Another member cautioned about using real country and people names due to programmed-in protections, suggesting fictional entities for easier simulation, and recommended having a clear output in mind for each simulation turn.
- **Meta-Prompting Engines for Prompt Refinement**: A member inquired about using PDFs containing information on prompt engineering to maximize prompt results, to which another member responded that they create specific **engines for meta-prompting**.
   - They mentioned having a course and using the material as knowledge in projects and custom GPTs to maintain rigor, while also tuning the context for specific goals.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1389080680138276925)** (56 messages🔥🔥): 

> `Kaleidoscope tiling, AI code generation, Grok 4, Alternate history simulation, Meta-prompting engines` 


- **AI botches Kaleidoscope Tiling Attempt**: A member tried using AI to generate code for kaleidoscope tiling but encountered issues with the output, describing it as creating *lines that are the same color of a texture spiraling out from a point* and not what they expected, but the bot gave [a ChatGPT link to a working example](https://chatgpt.com/share/68628609-a798-8000-a949-a11701d8e11b).
   - Another member suggested using **quadrants**, **90°**, and **offset** keywords and a user mentioned that they verified the output was indeed tileable after opposite pixels on the edges were identical.
- **Grok 4: Improvement Incoming, Potential Regression Possible**: A member inquired about potential improvements in **Grok 4**, to which another member responded affirmatively, stating that *training models is like building CPUs* with each model being the result of countless stochastic weights.
   - They noted that while improvements are likely, some features could regress, comparing the process to how CPUs are binned by aftermarket retailers based on performance.
- **Meta-Prompting Engines for Efficient Prompt Refinement**: A member asked about using PDFs containing information on prompt engineering to maximize prompt results, and another member responded that they create specific engines for **meta-prompting** for such tasks and tune the context for specific projects.
   - They mentioned having a course and using the material as knowledge in projects and custom GPTs to maintain rigor, and referred to Grok’s version of Custom GPTs which refer to uploaded files every time as basically the same as ChatGPT.
- **Alternate History Sim Prompt Development**: A member sought assistance in developing a prompt for an alternate history simulation focusing on dynastic and country actions, which was referred to [this example](https://chatgpt.com/share/6862f575-07dc-8011-a8f2-24062a84064a).
   - Another member cautioned about using real country and people names due to potential legal and ethical concerns, suggesting the use of fictional names to avoid such issues.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1388234028418863114)** (1305 messages🔥🔥🔥): 

> `Replacing CEOs with LLMs, Microsoft's AI Memo, GRPO Algorithm, Gemma 3 Learning Rate, LLM-based TTS` 


- **Youtubers Advance CEO Replacements with LLMs**: Some members are discussing the idea of [replacing CEOs with LLMs](https://www.youtube.com/watch?v=THfBccihkVQ) as being advanced by youtubers and digital marketers, questioning the reliability of AI for critical tasks.
   - One member suggests replacing underperforming middle managers but not with unreliable AI, stating that *80% on SQL is diabolically bad* for something so binary.
- **Microsoft Mandates AI, Raising Concerns**: A member shared a [Business Insider article](https://www.businessinsider.com/microsoft-internal-memo-using-ai-no-longer-optional-github-copilot-2025-6) revealing that Microsoft is pushing managers to adopt AI tools, leading to concerns about the company's direction.
   - Another member noted that making sure *I'm in the top 10% of internal ai tool users in my organization* is the move, even if the ai tooling used is so laughably bad it should be considered a bad joke.
- **Deepseek's GRPO Algorithm Discussed**: Members discussed the **GRPO** reinforcement learning algorithm developed by **Deepseek**, noting it as an advancement over **PPO** and **DPO** for reasoning capabilities.
   - Speculation arose that **Google** and **OpenAI** might have their own hidden algorithms for reasoning, though **OpenAI** primarily uses the **PPO RL** algorithm.
- **Unsloth LR recommendation for Gemma 3**: A member inquired whether **2e-5** is too high a learning rate (**LR**) for long training runs of **Gemma 3**, referencing recommendations by the **Unsloth** team.
   - Another member clarified that **LR** suggestions in the **Unsloth** documentation are a starting point, dependent on rank and alpha, and not to be taken out of context, so a full fine tune has different concerns.
- **Open Source LLM-based TTS Discussed**: Members are discussing open-source LLM-based TTS (text-to-speech) models, with one member seeking to train one from scratch on the **LJSpeech** dataset.
   - Another member noted that because *you need an llm inside* the architecture vocal variance is too high for emotion, adding that most now train on millions of hours of data and that there's a lot more data than a 1k hour dataset overall.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1388871593408598187)** (12 messages🔥): 

> `Local Llama communities, Gemma3 notebook release, ML with Javascript, Floating point arithmetic in Javascript, Python ML tools` 


- **Searching for Local Llama Communities**: A member asked if there are any alternative reddits or communities for **local llama** type stuff.
   - Another member pointed to [Unsloth's Reddit](https://www.reddit.com/r/unsloth/).
- **Gemma3 Notebook Release When**: A member asked when the **Gemma3 notebook** would be released and if there's anything specific to that architecture that needs to be handled.
   - The same member wondered if they could use the regular **gemma3 notebook** and swap out the name.
- **ML and Javascript Don't Mix**: A member advised to never do **ML stuff with JavaScript** no matter what.
   - They claim that *Javascript and floating point arithmetic are never meant to be*.
- **Debugging Floating Point Casting Errors is Painful**: A member recounted wasting several hours solving a **floating point equation** where the types miscast with the dataframe.
   - They added that while debugging, the dataframe clearly showed the scalar as float but it was a string or object, and wondered if **Python** has better tools to avoid this.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1388233280301830366)** (438 messages🔥🔥🔥): 

> `Qwen3-0.6B-Base FT success, LoRA learning rate tuning, Rank/Alpha association, RP Models, Dataset format for training` 


- **LoRA Learning Rate Debates Rages On**: Members discussed appropriate **LoRA learning rates**, with one user finding success using **2e-4** with **Qwen3-0.6B-Base**, while others suggested values like **1e-4** or **5e-5** for larger models.
   - One experienced member clapped back and [defended the use of 2e-4](https://arxiv.org/abs/2312.03732), emphasizing the importance of considering the context of rank and alpha settings while using [2e-4](https://arxiv.org/abs/2312.03732) as a perfectly valid setting, especially with `alpha=rank`.
- **Dataset Columns Confusion causes consternation**: A user sought help with dataset formatting for fine-tuning, getting errors related to column names, and one member suggested matching the dataset structure to the notebook examples to avoid code modifications.
   - Another member [recommended restructuring the dataset into JSONL format](https://jsonlines.org/examples/) with a  `conversations` key, containing question/answer pairs formatted as `human` and `gpt`.
- **DeepSeek Fine-Tuning Frustrations Flourish**: A user encountered an `IndexError` while fine-tuning the DeepSeek model, questioning if something was missing from the sample code, but a solution was not found in the messages.
   - A member shared a [finetuned 4x12B model for RP](https://huggingface.co/models), which was praised for being *decent* and using Mistral Nemo chunks.
- **Llama 4 Model Support in Limbo**: A user faced issues loading the `unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit` model, seeing errors related to unused weights, despite having ample GPU resources.
   - Despite a [blog post suggesting Llama 4 support](https://unsloth.ai/blog/llama4), members suggested the model *wasn't that great to begin with* and might not fit on an H100/H200. 
- **Orpheus Full Finetuning Finds Favor**: A user asked about training Orpheus TTS on a new language, and members recommended full fine-tuning with a large dataset of audio and text, while casting audio to 24kHz.
   - Members also noted that one needs to ensure that the SNAC tokenizer is used for audio tokenization, similar to how LLama's tokenizer is used to tokenize text. <https://huggingface.co/maya-research/Veena> based on snac and llama


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

laszlo01: https://github.com/Laszlobeer/Dungeo_ai
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1388319084759744715)** (6 messages): 

> `GPU Kernel optimization, Metal Kernels, LLMs Hallucinating` 


- **Evolved GPU Kernels Accelerate Apple Silicon**: Automated GPU kernel optimization using evolutionary programming discovered **Metal kernels** that outperformed **MLX's baseline** for transformer attention on **Apple Silicon**, achieving a **12.5% average speedup** and up to **106% peak improvement** on certain workloads, according to [this blog post](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery).
   - The optimization process autonomously discovered perfect `vec<T,8>` SIMD utilization and a novel two-pass softmax algorithm, showcased in their [Github Repo](https://github.com/codelion/openevolve).
- **LLMs Hallucinate Metal Kernels**: One member shared their experience trying **Metal kernels** a few months ago, but didn't progress due to lack of skills, noting that **LLMs** kept hallucinating, specifically mentioning trying **Gemini Pro** with [this repo](https://github.com/jedt/metal-quant-ext2).


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1388233261070815303)** (869 messages🔥🔥🔥): 

> `DeepSeek R1 0528, GPT 4.5 is mediocre, AI winter, OpenAI's sidebars, long dry spells` 


- **DeepSeek R1 0528's Secret Sauce**: Members discuss how [DeepSeek R1 0528](https://huggingface.co/deepseek-ai/DeepSeek-V3-Basepost) has a higher performance than DeepSeek R1 0120 because 0528 uses some, or majority distillation data from 2.5 Pro, boosting its performance just for post training.
   - It is debated whether DeepSeek R1 0528 is still based on DeepSeek V3 0324.
- **Members bemoan lack of groundbreaking releases**: Members lament that we still aren't seeing any groundbreaking releases like **GPT-4** or **O1**, instead, *everything is just trying to game the benchmarks (or improve upon existing capabilities)*.
   - Some suggest that we've become accustomed to an absurd rate of launches and that incrementalism is underrated.
- **Innovation is Lumpy, not Continuous**: Despite the perception of slow progress, members remind each other that innovation is always happening, although it's lumpy by nature and long dry spells are normal.
   - They point out that true cross attention across everything won't make much of a difference.
- **OpenAI and DeepMind Data Theft drama**: Some members claim that **OpenAI** stated **Deepseek** distilled its data for training and that the R1 update was likely Gemini, with OpenAI disabling full chain of thought to prevent data theft.
   - This is based on [Forbes article](https://www.forbes.com/sites/siladityaray/2025/01/29/openai-believes-deepseek-distilled-its-data-for-training-heres-what-to-know-about-the-technique/) and [TechCrunch Article](https://techcrunch.com/2025/06/03/deepseek-may-have-used-googles-gemini-to-train-its-latest-model/).
- **R2 canned in favor of R1-update**: Members discuss that [DeepSeek R2](https://www.deepseek.ai/) was not released because they were not satisfied with the performance, so they released **R1 update** version instead.
   - They speculated that AI companies hid the CoT (chain of thought) to prevent others from having training data.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1388713979244843128)** (1 messages): 

> `Live Financial Data, Price Movement Timelines, @mention Spaces, MLB Teams, Memory Search` 


- **Perplexity Ships Financial Data**: **Live Financial Data** in Research & Labs shipped this week, according to the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-june-27th).
- **Price Movements Over Time**: **Price Movement Timelines** on Finance Pages are new this week, says the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-june-27th).
- **Spaces Get Some Love**: You can now **@mention your Spaces**, according to the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-june-27th).
- **Root, Root, Root for the Home Team**: You can now **Follow MLB Teams with Real-Time Scores** according to the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-june-27th).
- **Recall All the Things**: **Memory Search** shipped this week, says the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-june-27th).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1388243173439963298)** (966 messages🔥🔥🔥): 

> `Grok vs Gemini for Transcription, Multi-Model Approach, Model preferences, Comet Browser Beta Access, AI scraping` 


- **Gemini is King of transcription**: Members find that **Gemini 2.5 Pro** performs better than **ChatGPT** in transcribing audio files (MP3s), but members debate on what services and purposes each LLM is best used for overall.
   - Members mentioned the **translate button in Grok** but said it was a waste of computing power.
- **The Multi-Model Approach is the Way**: Members are beginning to use **multiple models** as opposed to just one, using **Gemini for core logic/backend** and **Claude for frontend/UI design**.
   - Additionally, some suggest using the "Best" model setting on Perplexity isn't actually the best in terms of picking the optimal model, but instead is set to *minimize cost*.
- **Perplexity's "Thinking" Visual Bug**: The "thinking" process shown by Perplexity's models was actually a *visual bug*, as confirmed by a member, and not a feature of the non-reasoning models.
   - Members debated whether Perplexity Deep Research or normal Search mode gave better results with specific models and for specific purposes.
- **The Comet Browser Beta is Almost Here**: Many members want to get access to the **Comet Browser beta** that Perplexity is developing.
   - One member posted a video of **Comet playing Pokemon** and another member posted a **link to the waitlist** [here](https://comet-framer-prod.perplexity.ai).
- **AI Scraping**: Members also had a discussion on **AI scraping** of websites. and what LLMs would be best used for this task.
   - Members agreed that if you're scraping a website you don't own, then you shouldn't scrape any website data without their permission.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1388461354096263209)** (8 messages🔥): 

> `Perplexity Pages, Deep research, US GO, Sci Fi movies, OpenAI leadership` 


- **Perplexity Lights Up Guanxi**: A user shared a [Perplexity AI page](https://www.perplexity.ai/page/lighting-up-guanxi-the-secret-yExK2zqDRL6i0i7FMFHxFw) on the topic of *Guanxi*.
- **Perplexity Prepares 200 Subs**: A user shared a [Perplexity AI page](https://www.perplexity.ai/page/perplexity-prepares-a-200-subs-PLwdrE8ZTCqLVYD5.aXV5A).
- **Deep Research, Perplexity Style**: A user shared a [Perplexity AI search](https://www.perplexity.ai/search/how-to-use-perplexity-deep-res-Fl7NpWKgQXmCLoplXwGCqQ) on how to use Perplexity AI for deep research.
- **OpenAI Leadership Responds**: A user shared a [Perplexity AI page](https://www.perplexity.ai/page/openai-leadership-responds-to-Xk5GJAzaTqq6PiGGYh8u7A) regarding OpenAI leadership's response.
- **Xi Jinping at the Helm**: A user shared a [Perplexity AI page](https://www.perplexity.ai/page/xi-jinping-at-the-helm-of-chin-VcU.ZRY2TsSPqmuCvGVU6Q) regarding Xi Jinping's leadership in China.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1388634506998579200)** (7 messages): 

> `API Credit Expiry, Sonar deep research, Deepseek Models` 


- ****Clarify** API Credit Expiry Confusion**: Users questioned why their **API credits** showed an expiry date, even though they purportedly do not expire.
   - A member clarified that *hackathon credits* do expire, while *purchased credits* do not have an expiration date.
- **Address **Perplexity Pro's** Monthly Credit Expiry**: A user noted that the **$5 credit** received from **Perplexity Pro** has a **one-month expiry**.
   - Another user confirmed that the credits renew monthly, implying a "use it or lose it" system.
- ****Sonar-Deep-Research** capabilities explored**: A user inquired whether **sonar-deep-research** supports the **response_format : json_schema** parameter.
   - Whether or not it supports this parameter still remains unclear from the discussion.
- ****Sonar Models'** Foundation Investigated**: A user questioned whether all **Sonar models** are based on **Deepseek models**.
   - They also inquired whether any **non-Deepseek models** are offered, highlighting interest in the underlying architecture of Sonar models.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1388237246288629990)** (861 messages🔥🔥🔥): 

> `Cursor Agent rules, Cursor Pro Pricing Changes, Gemini CLI issues, Mobile Access to Cursor Agents, Supabase MCP setup` 


- ****Rules-Based AI: Agent Rules & Project Implementation****: Community members highlighted the importance of setting up [**Agent Rules**](https://x.com/search?q=%23AgentRules) on platforms like ```X``` before diving into prompt implementation, even making your own rules for your projects.
- ****Navigating Cursor's Pricing Plan Predicaments****: Users are reporting unexpected [usage-based pricing](https://www.cursor.com/pricing) charges even with moderate usage, leading to concerns about potential monthly costs exceeding the base subscription, with some considering a switch back to the old pricing model.
   - Members shared that under the old plan, exceeding the fast request limit resulted in a switch to slower requests, whereas the new plan introduces overage fees without clear notifications, sparking debate on the value proposition of the Pro plan.
- ****Gemini CLI: Great Promise, Rough Start****: While the [Gemini CLI](https://cloud.google.com/ai/vertex-ai/docs/generative-ai/code/code-models) offers coding capabilities and a generous context window, users find it underwhelming due to its slowness, unreliable scaffolding, and inability to handle interactive CLI commands.
   - Despite these drawbacks, some see potential in its free access to a large context window, useful for background tasks, with the caveat that the tool is still in its early stages.
- ****Mobile Cursor: Agents Unleashed on the Web****: Cursor released web-accessible [background agents](https://www.cursor.com/agents), enabling users to manage and interact with agents via a web browser, blurring the lines between desktop and mobile experiences.
   - The community debated whether accessing agents through a mobile web browser qualifies as a true mobile experience, while others noted it requires enabling MAX mode, leading to discussions about pricing implications and potential token usage costs.
- ****Streamlining Supabase: MCP vs CLI****: Members discussed the transition from manual **Supabase** Multi-Project Configuration (**MCP**) setup to using the **Supabase CLI**, noting that the CLI now creates its own mcp.json, simplifying the process.
   - While the **Supabase CLI** offers automation, some prefer the **MCP** for its management and deployment efficiency, especially for configuration experimentations.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1388272584738472076)** (65 messages🔥🔥): 

> `Slack Integration, GitHub Token Loss, Background Agents Freezing, Background Agent Terminals Configuration, Docker Issues` 


- **Cursor Slack Integration has Bug**: A user suggested that Cursor+Slack errors and permission issues should be posted back to **Slack as private messages** instead of public channel posts.
- **GitHub Token Gone Astray**: A user reported that **Cursor lost all their GitHub tokens**, requiring reconnection, and questioned the need for such issues to be public.
   - This may be a bug report that the user missed in threads/conversations.
- **Background Agents getting chilly**: When checking a Background Agent's progress and selecting "Checkout Locally", the chat connection stops working without a complete window recycle of Cursor.
   - This issue consistently reproduces across projects and background agents, affecting only that specific Cursor window.
- **Background Agent Terminals Configuration Needs Help**: A user running a rails backend with react/redux frontend using a custom Dockerfile reported that starting processes with **"terminals" configuration** (e.g., `yarn dev`) **doesn't seem to work**.
   - They are seeking tips on how to debug this issue.
- **Docker Issues Trigger Build Cache Default**: When builds fail and are resubmitted, Cursor defaults to ignoring the cache, and changes to the **Dockerfile** also trigger this behavior.
   - There are also cases where the cache upload fails, causing it to fall back to a docker build.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1389263718008885248)** (1 messages): 

> `Cursor agents, Web and mobile cursor` 


- **Cursor's Mobile & Web Debut**: **Cursor** is now available on your phone and on the web, letting you spin off dozens of agents and review them later in your editor.
   - Try it out via this [link](http://cursor.com/agents) or read this [blogpost](https://cursor.com/blog/agent-web).
- **Agents in Cursor**: Users can now **spin off dozens of agents** within Cursor and review them later in the editor, enhancing productivity and workflow efficiency.
   - This feature allows for parallel task execution and streamlined code review processes within the Cursor environment.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1388235938643312640)** (422 messages🔥🔥🔥): 

> `LoRA adapter push issues, Numberlink reasoning benchmark, UI/UX model training, IBM® Power® OS vs OS/2, AI code generation security challenges` 


- **LoRA Adapter Push Encounters Roadblock**: A member faced issues pushing a **LoRA adapter** to the hub, saving successfully but failing to push, possibly due to [login or key problems](https://huggingface.co/docs/hub/security-tokens).
   - They noted discrepancies in file sizes when pushing with *trainer.model* versus *model.push_to_hub*, and resorted to a CLI upload instead.
- **Flow Free could become Reasoning Benchmark**: A member suggested revisiting the **Numberlink/Flow Free game** as a reasoning benchmark inspired by **TTT-Bench** for evaluating large reasoning models, linking to their [proposal](https://docs.google.com/document/d/1RNHnNFVirdNOUPBtng5J-CBhaPVQMf1ogmjPIu_J77A/edit?tab=t.0).
   - They wondered if it would be too repetitive to **TTT-Bench**, but welcomed collaboration on a group project.
- **UI/UX Model Training Cost Compared**: A member sought advice on training **AI models** for programming tasks, particularly **UI/UX**, considering **RLHF**, and was advised to get a [Unsloth Colab notebook](https://github.com/unslothai/unsloth).
   - It was suggested that using **Gemini 2.5 Pro** with custom prompting or **Sonnet 4** via OpenRouter would be cheaper and more performant than self-hosting a finetuned model like **DeepSeek-R1-Distill-Qwen-32B**.
- **IBM Power Rack vs OS/2 sparks debate**: After a user mentioned stepping on a rusty nail, another joked that the user should main a **IBM® Power® OS 2U rack server**.
   - Another member chimed in saying that *real OGs use OS/2*, with someone else noting they just got an ad from **IBM**.
- **AI Code Generation Security Challenges Surveyed**: A member shared a quick, **5-min survey** on the biggest security challenges with **AI-generated code** in production, planning to share results with the community.
   - The [survey link](https://buildpad.io/research/EGt1KzK) was provided for interested engineering teams.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1389167954599612456)** (3 messages): 

> `DynamicCache Memory Recycling, Kaggle Gemma 3N Hackathon` 


- **DynamicCache Fails to Recycle Memory**: A member reported that after initializing a `DynamicCache` and performing LLM inference, the **KV cache memory** was not being recycled, despite setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and calling `gc.collect(); torch.cuda.empty_cache()`.
   - The user was seeking advice on how to properly recycle the cache and free up **VRAM** after the inference function returns.
- **Gemma 3N Hackathon Seeks Participants**: A member invited others to participate in the [Google Gemma 3N Hackathon on Kaggle](https://www.kaggle.com/competitions/google-gemma-3n-hackathon).
   - No further details about the hackathon were provided.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1388258967024570640)** (4 messages): 

> `AGI impact, Roko's Basilisk` 


- ****AGI** Creates Cat Videos, Saves Jobs?**: A member joked that *we've had **AGI** for years and all we've done is make cat videos*, implying human jobs are safe because people are *barely literate monkeys*.
   - Others laughed at the assessment.
- ****Roko's Basilisk** Resurfaces**: A member responded with a link to the [Wikipedia page on **Roko's Basilisk**](https://en.wikipedia.org/wiki/Roko%27s_basilisk).
   - The thought experiment suggests an **AI** might punish those who didn't help bring it into existence.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1388278624603476118)** (206 messages🔥🔥): 

> `Automated GPU kernel optimization, Thin C ABI wrapper for HF tokenizers, Fungal substrate as memresistor, Simple Research Agent` 


- **Evolved Metal Kernels Beat MLX Baseline**: Using evolutionary programming, a member auto-discovered **Metal kernels** that beat **MLX's** baseline for transformer attention on **Apple Silicon**, achieving a **12.5%** average speedup and **106%** peak improvement, with code available on [GitHub](https://github.com/codelion/openevolve) and a writeup on [HuggingFace](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery).
- **HF Tokenizers Get Thin C ABI Wrapper**: A member created a thin **C ABI wrapper** for **HF tokenizers** ([GitHub](https://github.com/m-doughty/tokenizers-ffi)) and **Raku bindings** ([GitHub](https://github.com/m-doughty/Raku-Tokenizers)), allowing token encoding, decoding, and counting in any language with minimal FFI code.
- **PDFs No Longer Wreck NLP Pipelines**: A member released **pdf2seg** ([GitHub](https://github.com/p3nGu1nZz/pdf2seg) | [PyPi](https://pypi.org/project/pdf2seg/)), an OCR-powered, tokenizer-free PDF segmenter featuring entropy-aware chunking, spaCy structure detection, and a CLI for LLM pretraining and clause-level extraction.
- **EasyTrain Simplifies LLM Finetuning**: A member introduced **EasyTrain** ([GitHub](https://github.com/Codalorian/EasyTrain)), a case-insensitive program that streamlines the setup of **AI training** or **inference** with minimal code.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1388613890794852492)** (3 messages): 

> `Object Detection Dataset Visualization, HF Datasets Library, Grounding DINO auto annotation, HuggingFace CV course` 


- **Seek GUI for Local Object Detection Dataset**: A member is seeking a **GUI program** to visualize and edit annotations for a local object detection dataset using the **HF `datasets` library**.
   - They have already tried **Label-Studio**, **vgg/via**, and **fiftyone** but encountered issues loading the datasets.
- **Dataset Auto-Annotated with Grounding DINO**: The object detection dataset was **auto-annotated with Grounding DINO** and is stored locally on their drive.
   - The member mentioned the data is not on the HUB *yet*.
- **HuggingFace CV Course Plug!**: A member suggested checking out the **HuggingFace CV course**.
   - They shared a link to the [course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome).


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1388250652521660606)** (8 messages🔥): 

> `Cosine Distance in k-means, Text Tilling for Thematic Analysis, Tokenizers FFI, Llama 4 vs Claude for MCP client, trl SFT trainer error` 


- **Cosine Distance in k-means: Good or Bad?**: A member asked if it's bad practice to use **cosine** as the distance metric in **k-means clustering** via normalization, so that **L2 distance** works like cosine.
   - Other members did not chime in.
- **Exploring Text Tilling for Thematic Analysis**: A member suggested using the **text tilling** paper for **thematic analysis**, as topic modeling was not yielding desired results.
   - The member elaborated to merge text tilling with **sentence transformers** to make embeddings more meaningful, potentially segmenting articles into smaller parts before embedding and clustering.
- **Tokenizers FFI Wrapper Emerges**: A member announced a **C ABI wrapper** for the Rust version of *tokenizers*, available on [GitHub](https://github.com/m-doughty/tokenizers-ffi).
   - The wrapper includes a **C header file** for FFI code and usage examples in the tests.
- **Llama 4 as Claude alternative**: A member inquired about using **Llama 4** solely for an **MCP client** as an alternative to **Claude**.
   - Other members did not chime in.
- **"completion" Key Missing in trl SFT Trainer**: A member encountered a **KeyError**: *'completion'* while using the **trl SFT trainer**.
   - The error occurred despite creating a dataset with a template, preprocessing, and using `DataCollatorForCompletionOnlyLM` because the dictionary example was missing a key named *"completion"*.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1388818542442315807)** (6 messages): 

> `Smol Course Certificates, Smol Agents Certificates` 


- **Enquiry About Smol Course Certificates**: A member inquired about the availability of certificates for the `smol course` and expressed excitement at the prospect of obtaining one.
   - Another member clarified whether the inquiry pertained to the **smol-course** or the **"smol agents" course**, indicating a search for the same information.
- **Smol Agents Certificates still available?**: A member confirmed they were referring to the **Smol Agents** course and inquired whether a certificate was available for it.
   - Another member mentioned receiving the certificate before July and expressed enjoyment of the course.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1388251898041077970)** (58 messages🔥🔥): 

> `Course Deadlines, Offline Course Access, DuckDuckGo Tool Integration, Smolagents Tips, Certificate Claims` 


- **Course Deadline Disappears into Thin Air**: The Agents course certification **deadline** seems to have been removed, now allowing users to learn at their own pace according to [updated info](https://huggingface.co/learn/agents-course/unit0/introduction).
   - Previously, there was confusion as some users believed the **deadline** was still **July 1st**, while others pointed out the change to a self-paced format.
- **No Internet? No Problem! Agents Course Goes Offline?**: A user inquired about the availability of **offline versions** for the Agents and MCP courses to utilize offline time during travel.
   - Unfortunately, as of this message, there is no information on whether the courses are available for offline use.
- **DuckDuckGo Tool Integration Troubles Duck Learn Agents**: A learner shared code for a "duckduckgo_fact_finder" tool, encountering **bugs** when integrating it into the Agents course, specifically with the **DuckDuckGoSearchTool()**.
   - Another member pointed out that wrapping the `DuckDuckGo()` search tool is unnecessary, as it is already usable as-is by passing it directly to the agent.
- **Smolagents Seeking Performance Power-Ups**: One user with two dozen agents in production is looking for **performance optimization tips** with **Smolagents**.
   - Specifically, they are seeking insight into blending OpenAI models to reduce costs (using 4.1 for planning and 4.1-mini for doing).
- **Certificate Claims Cause Consternation**: A user documented their process of claiming their certificate, explaining how they completed the course in their space and then went to the [certificate claim page](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate).
   - Another user added that they had problems creating an account to begin with and uploaded a screenshot showing a non-functional button.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1388294819767386203)** (2 messages): 

> `Llama 3.3 70B, Cloudflare Vietnam Philippines issue` 


- **Llama 3.3 70B discounted 70%**: There is now a **70% discount** live for [Llama 3.3 70B](https://x.com/OpenRouterAI/status/1938735144824652005).
- **Cloudflare issue resolved**: A Cloudflare issue impacting **Vietnam** and the **Philippines** based requests was investigated.
   - The issue is now resolved and they are continuing to investigate to understand the root cause of the problem.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1388253980349431921)** (26 messages🔥): 

> `PGaaS Prototype Feedback, Chat Super Slow Update, Minmax-m1 to Llama 3.3, Authentication / Anti Rev, Codebase to Text Tool` 


- **PGaaS Prototype Launched, Seeks Feedback**: A user launched a hasty **PGaaS prototype** and is looking for feedback at [paulgraham.resurrect.space](https://paulgraham.resurrect.space).
- **Chat Update Still Super Slow**: A developer pushed a new update to a **chat application** that is still reported to be super slow, but they have transitioned from **minmax-m1(extended)** to **llama 3.3**.
- **Authentication anti-rev improvements**: After pushing an update, a user noted a main concern would be adding **authentication** / **anti rev**.
   - The developer asked for input on what to improve in the **UI/UX**, suggesting maybe voice mode or a dark theme.
- **Codebase to text tool launched**: A developer launched a **codebase-to-text tool** (PromptMan) to convert any codebase into a markdown file at [PromptMan](https://promptman-frontend-7i4jm6usra-el.a.run.app/).
- **EveryDev.ai Uses OpenRouter for tool sharing**: The founder of **EveryDev.ai**, a new platform where AI developers can find, rate, and share tools, has been using **OpenRouter**, especially through the Cline API.
   - They are planning a giveaway and asking if there is a way to create **promo codes** or fund users' accounts directly to try out different models on OpenRouter.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1388241544124829757)** (576 messages🔥🔥🔥): 

> `OpenRouter token?, Gemini 2.5 Pro API, telegram raid, Automated spammers, GPT writing style` 


- **Speculation Arises over OpenRouter Token**: Users speculated about the existence of an **OpenRouter token ($OR)**, with some believing it's related to recent influx of crypto enthusiasts, however, it has been stated that *there is no xp / community rewards, there is no token, there is no airdrop*.
   - Members dismissed the idea, pointing out the implausibility of a scam token after a **$40 million seed A funding** round.
- **Gemini 2.5 Pro Free Tier Rumors**: A member mentioned a rumor about **Gemini 2.5 Pro** having a **free tier** for the API based on a [tweet](https://nitter.poast.org/OfficialLoganK/status/1938744437695299703) by **Logan Kilpatrick**.
   - The community expressed hope that the free tier would last longer than just a weekend, but one user was concerned about potential abuse through automated counting.
- **OpenRouter Server Under Telegram Bot Raid**: The OpenRouter Discord server experienced a raid from Telegram groups, with many new users joining and posting generic greetings.
   - Members speculated that these users were lured in by the promise of **crypto rewards** or **airdrops**, while one user mentioned they came from telegram, from a place called *роснодмониторинг*.
- **OpenRouter Server Battles Automated Spammers**: The OpenRouter community is facing an increase in **automated spammers** posting generic messages.
   - The community is discussing potential solutions, including **automod rules** and **mobile verification**, but no specific measures have been implemented yet.
- **LLMs Adjective-Overloading Writing Style**: One member complained about Large Language Models writing fiction with excessive **expository adjectives**.
   - They said *No matter how much I tell it not to, no matter how many examples I show it, none of them can say "He snarled at her before walking away," instead of "He glared at her evilly before storming off in anger."*


  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1388503015127912581)** (1 messages): 

> `Error Identification, Image Analysis` 


- **Error Spotted by User**: A user reported an unspecified error, including a [screenshot](https://cdn.discordapp.com/attachments/1277894087755829278/1388503015262126160/Screenshot_2025-06-28_195440.jpg?ex=6863dab3&is=68628933&hm=ce0ba5115df7df0f4d77511941d983054f502da3ae46dd281614576439db91f2&) for context.
- **Awaiting Details**: Further information is needed to identify and resolve the error.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1388897169301569698)** (2 messages): 

> `` 


- **No New Models Discussed**: The channel activity consisted only of bot messages indicating the channel name.
- **Channel Identified**: The channel is named 'OpenRouter - New Models' and is managed by Readybot.io.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1388232797650554953)** (334 messages🔥🔥): 

> `Structured Output (json_schema) in recent models, PDF output from local LLMs, GGUF offline installation, Model's Cache/Memory explanation, lmstudio Discord Banner` 


- **JSON Schema Support Still Scarce**: A member inquired about **JSON Schema** support in recent models like **Qwen30-A3B**, noting the absence of explicit documentation.
   - The response indicated that only **Qwen** models are confirmed to support structured output, with translation accuracy in this context being uncertain.
- **LLMs Can Generate LaTeX for PDF Creation**: A member asked about generating **PDFs** from locally hosted models like **Llama 3**.
   - Another suggested that the model should output **LaTeX code**, which can then be converted to **PDF** using readily available tools.
- **Offline GGUF Installation Guides Shared**: A user asked how to install **GGUF models offline**, on a computer without internet access.
   - Others explained the procedure: *download the model on a connected PC, transfer it via USB, and place it in the correct LM Studio directory* (**/home/user/.lmstudio/models/publisher/model-name**).
- **RAG Architecture Recommended over "Cache"**: A user wanted to know if a model could *restore* its **memory** *from the beginning of the chat*.
   - Instead of relying on the model's "cache," members recommended using **RAG (Retrieval-Augmented Generation)** with a local vector database to store and retrieve past interactions.
- **LM Studio New Vision Models Can Reason Too**: A member linked to a **Reddit post** asking if **LM Studio** supports **vision** and **reasoning models**.
   - Another member confirmed that **LM Studio** supports both **vision** and **reasoning models**, and that the screenshot in the **Reddit post** was from **LM Studio**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1388237033125842964)** (128 messages🔥🔥): 

> `Cloud deployment of LLMs, Runpod vs AWS for LLMs, Local LLM serving infrastructure, GPU recommendations for LLMs, Mac M4 Pro inference speed` 


- **Python tools launch AWS VMs**: A member suggested using **Python tools** to easily spin up **AWS VMs**, pointing to a [guide](https://www.geeksforgeeks.org/launching-aws-ec2-instance-using-python/) for creating an **EC2 instance**.
   - Another member found a [LinkedIn guide](https://www.linkedin.com/pulse/running-lm-studio-api-server-aws-complete-guide-angelo-artuso-ahllf) on running **LM Studio API server on AWS**.
- **Runpod vs AWS for LLM deployment**: Members discussed using **Runpod** or **vast.ai** as alternatives to **AWS** for deploying **LLMs** due to potential cost savings and specific needs.
   - It was recommended to avoid opening the **vLLM/Ollama/LMStudio** port to the internet for security reasons and to use local traffic services.
- **Infrastructure Considerations for Local LLM Service**: For serving a local **LLM** to 100-150 users, **openwebui** was suggested for the UI, with **vLLM** recommended on the software stack side.
   - Hardware requirements depend on the size and quantity of models, with recommendations including **A100** or **H100** GPUs, and advice to determine concurrency and performance needs before choosing hardware.
- **GPU Recommendations and Availability**: A recommendation for **six 6000 Pro’s** (576GB of high octane cuda) was met with discussion, as they are barely out and may not be easily available.
   - There was discussion on the budget, and it was suggested that they get *close* to the best, while another said *start from the actual requirements before recommending anything*.
- **Optimizing Model Inference on Mac Mini M4 Pro**: Users discussed the slower prompt processing speed on **Mac Minis** compared to **30/40/5090 PCs**.
   - It was noted that **Qwen 3 30b** works faster due to being a **Mixture of Experts (MoE)** model, and that **Mac Minis** have low bandwidth and not many GPU cores which also contributes to longer prompt processing times.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1388244893763113105)** (333 messages🔥🔥): 

> `Marin Project at Stanford, Percy Liang's Renaming Tendencies, Pretraining Infrastructure with Jax, UI Design LLMs, Emergent Misalignment Paper` 


- **Stanford's Marin Project Launches, Uses LLMs**: David from Stanford introduced the [Marin project](http://marin.community/blog/2025/05/19/announcement/), focused on **LLM-y things**, and shared a [related YouTube video](https://youtu.be/FASMejN_5gs?si=TQzSfPa2TEGBxMXT).
   - He clarified his role involves **pretraining infra** but is fairly "full stack".
- **Percy Gets Called Out, Renames Stuff**: A member criticized **Percy Liang** for consistently renaming established concepts, such as *foundation models*, and claiming the branding or citations, noting that *open development* has been *open science* for decades.
   - Another person in *metascience* added that what they (all groups) do goes very beyond regular *opensci*, but preferred a different new name than *open dev*.
- **Jax Powers Academic Pretraining Infra**: Discussion highlighted that Jax is used for **pretraining foundation models** in academia, particularly on **TPUs**, with [Levanter](https://github.com/stanford-crfm/levanter) being a Jax codebase for training.
   - Members noted Google offers **free compute** via **TRC**, but pretraining is resource-intensive and TPUs are often in high demand.
- **Startup Faces Reality Check, Needs better UI/UX**: A startup founder sought advice on building a **custom LLM for UI design**, expressing that *pretraining+RLHF layering* is the best option, but was cautioned about the costs and time required.
   - Experienced members recommended using existing models with vision capabilities, like **Gemini Pro** or **Claude**, along with smart prompting and scaffolding, or **R1** and **Qwen code**, rather than attempting to train a model from scratch without funding.
- **Emergent Misalignment Paper Reproducibility Falters**: A member reported difficulty reproducing the **misalignment rate** reported in the original *emergent misalignment* paper.
   - Others suggested the results vary depending on the judge model used (e.g., **GPT-4o**), and that **GPT-4o** may have been updated since the paper's release, plus datasets matter.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1388235860184535193)** (26 messages🔥): 

> `Spectral Decay Optimization, DROID robot learning, Composing Generative Models, NAACL 2026 Cancellation Rumors, Qwen 3 1.7B diffusion LM` 


- **Spectral Decay Optimizers Compared**: A member stated that the [2017 spectral decay work](https://repository.gatech.edu/entities/publication/3fda43bc-d998-40ad-989f-aa07f8d39bd3) resembles a spectral decay version of **Adam** while [Jianlin Su's blog post](https://jianlinsu.github.io/) is akin to a spectral decay version of **AdamW**.
- **DROID Algorithm for Robot Learning**: The **DROID** algorithm enables robots to infer behavior models more accurately and efficiently from users by modeling personalized components of reward and policy, using limited data.
   - It leverages limited data and is suitable for applications where environment rollouts are too expensive or unsafe, according to [this paper](https://arxiv.org/abs/2506.20701).
- **Constant Memory Diffusion Model Incoming**: Members shared a work that is related to a constant memory diffusion model, though **constant memory** is likely limiting [this paper](https://arxiv.org/abs/2506.15841) shows it is a good start.
- **Qwen 3 1.7B repurposed for diffusion**: **Qwen 3 1.7B** is being repurposed as a diffusion LM with a byte tokenizer and only took a few hours of training on 4x 4090s.
- **NAACL 2026 facing cancellation?**: Rumors are swirling that **NAACL 2026** may be skipped with no official announcements yet.
   - One member stated that *my phd advisor told me she heard it’s being skipped and we’re not sure where to submit*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1388560105154416782)** (3 messages): 

> `Model Diffing, Crosscoders, SAE on (chat - base) activations, refusal detection, OpenAI's sycophantic model update` 


- **Model Diffing Deciphering Discrepancies**: A [post on model diffing](https://www.lesswrong.com/posts/xmpauEXEerzYcJKNm/what-we-learned-trying-to-diff-base-and-chat-models-and-why) extends a [previous paper](https://arxiv.org/abs/2504.02922), focusing on understanding what makes a fine-tuned model different from its base model internally.
   - The authors found that **crosscoders** hallucinates differences due to sparsity enforcement but that training an **SAE** on *(chat - base) activations* works surprisingly well.
- **Features Exposed Reveal Refusal**: Methods used reveal interpretable features related to **refusal detection**, **fake facts**, or information about the model's identity.
   - The author suggest model diffing is a promising research direction that could have caught **OpenAI's sycophantic model update**.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/)** (1 messages): 

noble_monkey_75488: nvm codex corresponds to humaneval
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1388640581063803133)** (27 messages🔥): 

> `GPU MODE Binary Submission, SGLang Community, TorchServe Maintenance, PyTorch Model Serving, TorchScript vs AOTInductor` 


- ****GPU MODE** Asks for Python Submission**: Participants clarified that submissions to the [GPU MODE leaderboard](https://www.gpumode.com/leaderboard/496) require a **Python function**, not a binary.
   - The submission process hooks into their evaluation system, offering flexibility while welcoming feedback for additional library support like **CUTLASS**.
- ****SGLang** Seeks Thriving Slack Space**: Members inquired about active communities for learning **SGLang** development, to which the [SGLang GitHub](https://github.com/sgl-project/sglang) was recommended.
   - The community is reported as active and responsive for developers seeking guidance.
- ****TorchServe's** Support Sunset Sparks Search**: **TorchServe** is officially in "Limited Maintenance" status, meaning no more planned updates, bug fixes, or security patches, according to [the official pytorch repo](https://github.com/pytorch/serve).
   - This shift raises questions about optimal model serving solutions in production, particularly with the rise of runtime optimization techniques like `torch.compile` in **PyTorch 2.0**+.
- ****PyTorch** Model Serving Solutions Surface**: For serving **PyTorch models**, particularly **LLMs**, recommendations leaned towards using **VLLM** or **SGLang** due to their system-level optimizations.
   - Options like **NVIDIA's Dynamo** and **flask-like solutions** were mentioned, with the latter placing responsibility for model performance on the user.
- ****TorchScript** sunset, **AOTInductor** Ascends**: **TorchScript** is no longer maintained, so any bugs or regressions you run into, you won't get help.
   - If the overhead of running python is acceptable you can and should enable our **MegaCache** [pytorch.org](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html), if not you want to use **torch.export** and **AOTInductor** [pytorch.org](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/).


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1388508974281785415)** (5 messages): 

> `Scatter/Scatter_add in Triton, Getting Started with Triton` 


- **`scatter/scatter_add` Availability in Triton**: A member inquired about the availability of `scatter`/`scatter_add` in Triton, referencing [a specific line of code in the Triton repository](https://github.com/triton-lang/triton/blob/main/python/triton/language/core.py#L1458).
   - The member was unsure about how to use the function.
- **Jumpstarting Triton Adventures**: A member asked for advice on how to get started with Triton, seeking suggestions on initial steps and practice problems.
   - Another member suggested making anything faster and aiming for a top 3 position on problems listed on [gpumode.com](https://www.gpumode.com/).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1388317215383752775)** (37 messages🔥): 

> `SM80 usage, nvcc compiler constant memory usage, Little's Law on GPUs, CUDA kernel optimization` 


- **SM80 experiences shared**: A member mentioned using **SM80** and expressed interest in comparing experiences with others.
   - No specific details about **SM80** usage or common aspects were shared in the provided context.
- **Nvcc Constant Memory Woes**: A member is facing issues with the **nvcc compiler** reserving constant memory per function, even when the functions are empty, leading to constant memory overflows and performance degradation.
   - They tried combining device functions, using `noinline` with `--rdc=true` and `--maxrregcount=88`, but the issue persists; each function reserves about 300-400 bytes of `cmem[0]`.
- **Little's Law Links GPU & DRAM**: Members discussed how [Little's Law](https://en.m.wikipedia.org/wiki/Little%27s_law) applies to the connection between **GPUs** and **DRAM**, citing **NVIDIA's** tutorials on using **LDGSTS** and **TMA** on **Hopper** and **Blackwell**.
   - Bandwidth growth outpaces the number of **SMs** per **GPU**, necessitating more bytes in flight to utilize the full bandwidth, as highlighted in [this GTC presentation](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/).
- **Kernel Spill Boundaries Explored**: A member suggested using separate **.cu** files to explicitly enforce spill boundaries in large kernels, rather than relying on `noinline` which vaguely hints at spills.
   - They explained that **nvcc** typically builds one giant **ptx** blob, but register pressure can cause spills and function calls, and controlling this boundary can optimize performance by considering whether threads of the same warp take the same case of the switch.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1388359879097323592)** (15 messages🔥): 

> `torch.export Model Exports, FlexAttention Integration, vmap incompatibility with torch.export, Executorch Workarounds for Model Export` 


- **Torch Exports Face Challenges**: Users are finding it difficult to export models using `torch.export`, often encountering errors like `RuntimeError` related to `vmap` and `.item()` calls, even when using seemingly standard models like **Mistral-7B-v0.1**.
   - One user traced the error to [HF's masking_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py#L158), which uses `vmap`, and noted that `torch.export` doesn't seem to support it.
- **FlexAttention Unclear in Exports**: It's unclear whether models using **FlexAttention** can be exported easily with `torch.export` without modifications.
   - One user initially suspected **FlexAttention** might be the cause of export failures, but clarified they were simply trying to export models directly from Hugging Face without specifying a particular attention implementation.
- **Executorch offers temporary workaround**: Executorch offers a temporary workaround for exporting models that are incompatible with torch.
   - Users can consult [Executorch's Llama export script](https://github.com/pytorch/executorch/blob/main/examples/models/llama/export_llama.py) and a [guide](https://huggingface.co/docs/transformers/v4.53.0/en/executorch) to switch to a vmap-free implementation using `transformers.integration.executorch` utilities.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1388629901040549949)** (1 messages): 

> `Exo 2, User-Schedulable Languages (USLs), scheduling operations` 


- **Talk on Exo 2 Scheduled**: A talk on **Exo 2** by a member is scheduled in 30 minutes.
   - The [paper abstract](https://arxiv.org/abs/2411.07211) was shared.
- **Exo 2 is designed to grow USLs**: The paper abstract described **Exo 2** as a scheduling language that enables users to define new **scheduling operations** externally to the compiler.
   - It composes a set of trusted, fine-grained primitives so that users can safely write their own scheduling library to build up desired automation.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1389347361721024602)** (4 messages): 

> `CUDA kernel integration, vLLM module replacement` 


- **Seeking CUDA Kernel Integration Expert**: A research group is seeking a consultant for integrating a custom **CUDA kernel** with high-performance **LLM inference engines**, expecting up to **4 hours** of work.
   - They aim to demonstrate a speedup by integrating their custom CUDA kernel into **LLM inference**.
- **Wrapping CUDA calls with custom_op**: A member advised wrapping the **CUDA call** in a `custom_op` and replacing the target **vLLM module** (e.g., `LinearMethodBase`) with a custom class.
   - This custom class would then call the **CUDA kernel** in the `.apply()` method.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1388735592262533190)** (40 messages🔥): 

> `Numerically unstable lerp function, CPU vs GPU efficiency, GPU warps and parallelism, Breaking hashes on GPU, CUDA thread count optimization` 


- **Unstable Lerp Function Spotted!**: A member noted that the [lerp function](https://github.com/kr1viah/WKChallengeModeSeedFinder/blob/e89c569b5e0f899d257abc7adea042ddf7daee11/main.cu#L41-L44) in the code is not numerically stable.
   - The original coder acknowledged the issue but stated, *"it works so im not gonna touch it"*.
- **Debate on CPU vs GPU Efficiency Surfaces!**: Discussion arose on whether a particular workload would be more efficient on a CPU, citing reasons such as non-uniform loop iterations leading to wasted FLOPs.
   - It was explained that the thread with the longest loop can force other threads to wait, potentially wasting compute if some threads are doing significantly fewer iterations.
- **GPU Warps and Parallelism Explained!**: A member clarified that GPU cores operate in **warps** (typically of 32), executing the same code on different data, unlike CPU cores which can run independently.
   - Branches and conditionals in kernel logic cause GPU cores to synchronize, potentially reducing parallelism, as all cores execute the same code after a branch.
- **Hashing on GPU is popular!**: One member described breaking a hash as *finding x such that hash(x) = y*, where you are given the hash and y, and x is usually the plaintext password.
   - Another member confirmed that *that is what im doing*, except without passwords or whatever, and singlethreaded on the cpu.
- **CUDA Thread Count Optimized!**: The discussion suggested maximizing the utilization of each warp to avoid idle cores and loading another kernel onto free warps.
   - It was recommended to consult the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture) for best practices and to use `nsight compute` to determine optimal thread counts based on register usage and shared memory.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1388569538899087503)** (3 messages): 

> `Geoffrey Hinton, AI risk, Becoming a plumber` 


- **Hinton's Plumbing Pivot**: Members are discussing [Geoffrey Hinton's views](https://www.youtube.com/watch?v=giT0ytynSqg) on AI risk and his suggestion that people should consider becoming plumbers.
   - He fears AI might automate many jobs.
- **AI Dooms Us All?**: Hinton is *scared* about the future of AI.
   - He warns that AI might automate many jobs in the future, leaving many people unemployed.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1388354167168303144)** (12 messages🔥): 

> `buffer_load_dwordx4 instruction, Composable Kernel, rocprofiler-sdk ABI` 


- ****Buffer Load Bonanza****: A member asked about experience using the **buffer_load_dwordx4** instruction, and another shared [their implementation](https://github.com/Snektron/gpumode-amd-fp8-mm/blob/main/solution.hip).
   - The member noted the scarcity of documentation, while the other clarified that parts of their code, specifically the buffer handling, was adapted from the **Composable Kernel** library.
- ****Composable Kernel Composition****: A member stated that the buffer handling code was pulled from **Composable Kernel** with comments, but implemented their own unroll code for performance reasons.
   - They found **CK's unroll** more cumbersome and not compatible with **C++20 templated lambdas**, encouraging others to freely reuse parts of their code.
- ****Rocprofiler Rolls Out ABI Update****: A member noted that libatt_decoder_trace.so had a slightly different ABI, but the mainline **rocprofiler-sdk** should now recognize **librocprof-trace-decoder.so**.
   - Another member confirmed the improvement, signaling a positive update to the **rocprofiler** tooling.


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1388890672546840586)** (1 messages): 

> `wgpu-rs, storage texture, r8unorm` 


- **Trouble using r8unorm storage texture in wgpu-rs**: A user reported they couldn't use `r8unorm` as a format for a storage texture in `wgpu-rs`.
   - They pointed out that `r8unorm` is actually [supported in the spec](https://www.w3.org/TR/WGSL/#storage-texel-formats).
- **wgpu-rs and r8unorm**: A user using `wgpu-rs` reported an error when trying to use `r8unorm` as a format for a storage texture.
   - Despite being listed in [the specification](https://www.w3.org/TR/WGSL/#storage-texel-formats), the `wgpu-rs` implementation threw a validation error.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1388318663198769282)** (21 messages🔥): 

> `Automated GPU Kernel Optimization, OpenEvolve Tool, Thread Value Layouts in CuTe, NVIDIA PTX Kernel, TokenDagger for Tiktoken` 


- **Evolved Kernels Beat MLX on Apple Silicon**: A member used evolutionary programming to auto-discover **Metal kernels** that beat **MLX's baseline** for transformer attention on Apple Silicon, achieving **12.5% average speedup** with **106% peak** on some workloads, as detailed in their [HuggingFace blog post](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) and [GitHub repo](https://github.com/codelion/openevolve).
- **Simon Explains Thread Value Layouts in CuTe Visually**: A member shared a [blog post](https://veitner.bearblog.dev/thread-value-layouts-in-cute/) inspired by a **GPU Mode lecture by Cris Cecka**, explaining thread value layouts in **CuTeDSL** with visuals and examples, alongside links to the [lecture](https://www.youtube.com/watch?v=ufa4pmBOBT8&t=1379s) and [slides](https://github.com/gpu-mode/lectures/blob/main/lecture_057/CuTe%20-%20Copy%20for%20GPUMode.pdf).
- **Hand-Written PTX Kernels Outperform CUDA**: A member implemented an inference-only version of **Andrej Karpathy's LLM.c project** using hand-written **NVIDIA PTX kernels**, achieving a **10% performance improvement** compared to the equivalent CUDA implementation, with code available on [GitHub](https://github.com/theunnecessarythings/llm-ptx) and detailed explanations in a [blog series](https://sreeraj.in/blog/llm-ptx-01).
- **TokenDagger Tokenizes Tiktoken Text Faster**: A member reimplemented **OpenAI's Tiktoken** in [TokenDagger](https://github.com/M4THYOU/TokenDagger), reporting **2-3x higher throughput** and **~4x faster** tokenization of code samples on a single thread.
- **Chisel Sharpens Kernel Profiling Workflow**: A member introduced **Chisel**, a tool for profiling kernels on **Nvidia** and **AMD GPUs** locally and instantly, available via `pip install chisel-cli` and on [GitHub](https://github.com/Herdora/chisel), currently supporting **DigitalOcean** for renting **MI300X** on AMD's cloud.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1388354634267099197)** (6 messages): 

> `ThunderKittens Repo, Research Assistant at Hazy, load_async_wait call, Thundermittens retirement` 


- **ThunderKittens Kernel Examples Pop Out**: There are many [ThunderKittens kernel examples](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels) in the repo, including the opportunity to make a pull request for **int 8**.
   - The PhD students maintaining the repo state that *they cannot support everything* and are excited to build more community contributions.
- **RA Positions at Hazy Research Explored**: A question was raised about the process to becoming a research assistant at **Hazy** as a non-Stanford student.
   - However, there were no responses elaborating on the process.
- **Async Loading Parameter Anomaly Spotted**: A user noticed that the `load_async_wait` call has **N=1** for the `4090.cu` in `kernels/attn/demo`, but the `readme.md` example shows **N=2** which is correct.
   - The inconsistency in the `load_async_wait` parameter for the `4090.cu` was raised as a potential issue, but no resolution or confirmation was provided.
- **Thundermittens Purged?**: A user inquired about whether **Thundermittens** got retired after noticing that the repo was deleted.
   - Unfortunately, there were no responses confirming or denying the status of the repository.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

remek1972: How to switch to fp16?
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1388386101814628382)** (8 messages🔥): 

> `CLI issues, Trimul Merch Prizes` 


- ****CLI Crashes**: Debugging in Progress!**: A user reported that the **CLI** is not working, suspecting a backend issue, but noted that **Discord submissions** are still functional.
   - A dev confirmed the issue was fixed on the API side and is awaiting approval for a PR that fixes the bot side, promising to ping the user upon completion.
- ****Trimul Trophies**: Time Window Tentatively Told!**: A user inquired about the deadline for **Trimul merch prizes**.
   - A dev responded that the deadline is tentatively set for **3 months**, accounting for merch readiness, and hinted that *Mark will be very generous*.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1388285701027659876)** (24 messages🔥): 

> `H100 Grayscale, A100 Trimul, L4 vectorsum, T4 histogram, MI300 amd-identity` 


- **Grayscale Personal Bests Achieved on H100**: A member achieved several personal bests on the `grayscale` leaderboard using **H100**, reaching a top time of **1431 µs**.
- **Trimul Claims 5th Place on A100**: A member secured **5th place** on the `trimul` leaderboard with a time of **23.2 ms** on **A100**.
- **vectorsum excelling on L4**: A member reported successful submissions on the `vectorsum` leaderboard using **L4**, achieving times around **970 µs**.
- **Histogram Hits 4th Spot on T4**: A member attained **4th place** on the `histogram` leaderboard with a time of **169 µs** on **T4**.
- **amd-identity Takes MI300**: A member achieved **9th place** on `amd-identity` leaderboard with **19.3 µs** on **MI300**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1388524138611605574)** (2 messages): 

> `CLI Fix, Submission Error` 


- **CLI Tool Receives a Fix**: The **CLI tool** has been fixed and is available for download from the [releases page](https://example.com/releases).
   - Users can now access the updated version.
- **Submission Faces Unexpected Error**: A user reported an unexpected error with `Submission 32983: amd-identity.py for amd-identity`, receiving the message *An unexpected error occurred. Please report this to the developers*.
   - The user followed instructions and reported the error for investigation.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1388530931400179802)** (8 messages🔥): 

> `Factorio entities retrieval issues, Factorio Lab Blueprint, Neel's trip to Prague` 


- ****Factorio** entity retrieval faces roadblocks**: A member reported issues with retrieving entities in **Factorio** when there are hundreds on the map.
   - They are using a blueprint for a lab that produces red, green, and blue science ([Lab_blueprint.txt](https://cdn.discordapp.com/attachments/1354169122107293786/1388842309575114822/Lab_blueprint.txt?ex=6863c532&is=686273b2&hm=f8891f5184b9a8d741746504ec0703e46945741bf2837970d626de1327ad0375&)) and seeks help generating data based on it.
- ****Factorio** devs head to Prague**: A member mentioned being back from a trip and heading to Prague for a meeting with the devs on Thursday.
   - They invited the group to share anything they want him to bring up with the devs while he's there.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1389278260809957488)** (2 messages): 

> `producer/consumer warps, data movement in CUDA` 


- **Producer/Consumer Warps: When to Use?**: A member inquired about a systematic approach for determining when **producer/consumer warps** are more advantageous compared to warps that manage their own data movement in **CUDA**.
   - The question revolves around whether the decision boils down to empirical testing or if there are guiding principles for choosing between the two approaches.
- **Optimizing Data Movement in CUDA**: Discussion focused on the trade-offs between **producer/consumer warps** and self-managing warps for **data movement**.
   - The inquiry highlights the need for a clear methodology to guide the selection process based on application-specific requirements.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1388237948633223238)** (3 messages): 

> `Systems ML Compiler Project, Heterogenous Deep Learning Stack, Compiler IRs, Max Bernstein on IR design` 


- **Systems ML Compiler Project Seeks Contributors**: A member is seeking contributions to a serious compiler project for the Systems ML community, aiming to implement subsets of C, CUDA C, Triton, and PyTorch to support the heterogenous stack of today's deep learning systems, detailed in the [Zero-to-Hero project page](https://j4orz.ai/zero-to-hero/).
- **SoN Compiler's First Steps**: Development has begun on the **SoN compiler**, implementing a subset of C and building on it with CUDA C extensions, with initial code available in [parser.rs](https://github.com/j4orz/picoc/blob/master/src/son/parser.rs) and [optimizer.rs](https://github.com/j4orz/picoc/blob/master/src/son/optimizer.rs).
- **IR Design Philosophy Highlighted**: A blog post from Max Bernstein on IR design was shared, emphasizing the main premise of **making decisions with only local information**, with the [post available here](https://bernsteinbear.com/blog/irs/).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1388251071834488892)** (108 messages🔥🔥): 

> `spherical k-means, sentence segmentation techniques, LLM hallucinations, pretraining LLMs, GritLM` 


- ****Spherical K-means****: Using **cosine** as the **distance metric** in **k-means clustering** is a well-known and valid practice, referred to as **spherical k-means clustering**.
   - Normalization helps L2 distance behave like cosine distance, especially useful when infrastructure limitations prevent handling big data with specialized tools.
- ****LLMs Struggle with Hallucinations and Pretraining Bias****: Members discussed techniques to mitigate **hallucinations** and **paraphrasing** in **LLMs**, with one suggesting **pretraining** on a series of **long-CoT datasets** to address **pretraining bias**.
   - The suggested solution is to *clean the training dataset* and pretrain the **LLM** rather than relying on **RLHF** or clever prompt templates.
- ****GritLM Arises to Segment Sentences****: A member inquired about techniques for **sentence segmentation**, beyond basic full stop or semicolon-based methods, even segmenting list-like structures such as (a), (b).
   - The model **GritLM** was mentioned as a resource for solving this problem.
- ****DeepSeek and NSAttention Show Promise****: A discussion covered research into improving **attention mechanisms**, with **NSAttention** from **DeepSeek** being highlighted as a promising approach for scaling to larger contexts.
   - There are already known solutions what kind of solutions exist to solve it and make it more expressive.
- ****Blogpost Weighs on Capabilities Required for AGI****: A member shared a blog post discussing the capabilities needed for **AGI**, emphasizing the importance of acquiring **durable skills without retraining**.
   - Another member highlighted the ability to acquire durable skills without retraining as a prerequisite, suggesting techniques like **TTT** feeding into **PEER** to build new weights and add them to an expanding database.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1388234239509532752)** (56 messages🔥🔥): 

> `Associative Memory, LLM Alignment, RWKV-7 Goose, Human Cognition vs LLMs, Virility of Content Prediction` 


- **Transformers Viewed Through Associative Memory**: A member shared a paper titled "Understanding Transformer from the Perspective of [Associative Memory](https://arxiv.org/abs/2505.19488v1)" that explores **Transformer architectures** through the lens of associative memory.
   - The paper introduces retrieval SNR to measure memory capacity and suggests that **FFNs** can be seen as a type of associative memory.
- **LLMs and Human Cognition: A Detrimental Lens?**: One member expressed dislike for papers viewing **LLMs** through **human cognition**, arguing it confuses understanding of how LLMs work fundamentally.
   - Another member countered that this perspective is valuable for practical use-cases, particularly in areas like training data and fine-tuning, referencing [this paper](https://arxiv.org/abs/2506.05555).
- **LLMs Simulating Human Reactions**: A member suggested using AI to predict the **virility of content** by simulating human reactions, noting underexplored literature on using LLMs to mimic human psychology.
   - Another member argued against conflating **LLMs** with **human cognition**, advocating for direct LLM use to simulate humans, although agreeing that the human cognitive perspective allows different tools and techniques from other fields to solve otherwise intractible problems.
- **RWKV-7 Goose Architecture Debuts**: A member introduced [RWKV-7 "Goose"](https://arxiv.org/pdf/2503.14456), a new sequence modeling architecture with **constant memory usage** and **constant inference time** per token.
   - The **2.9 billion parameter language model** achieves a new **3B SoTA** on multilingual tasks and matches the current 3B SoTA on English language downstream performance despite being trained on dramatically fewer tokens than other top models.
- **LLM Alignment: Still Impossible?**: A member suggested using **ChatGPT** to find alignment resources, while cautioning that alignment is fundamentally impossible for an **LLM**.
   - They also noted that chatbots are often the worst source of information about AI due to being trained on a lot of hype BS, and noted that LLM sycophancy is an overalignment problem.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1388553512656437330)** (15 messages🔥): 

> `ML for Drug Discovery, Synthetic Data Contamination, Comparing Reasoning Models, Healthcare Costs and Access` 


- ****ML for Drug Discovery** Event Incoming!**: A free online event, [ML for Drug Discovery](https://mlfordd.com/), is scheduled in approximately 24 hours, with keynotes that provide an overview of the field.
   - Last year's event was deemed *awesome*, and recordings will be available on [YouTube](https://www.youtube.com/@MachineLearningDrugDisco-cv2tf).
- ****Gaming Benchmarks** with Synthetic Data?**: A member speculated that the way to achieve models with **90%+ scores** across the board is to *increase the data contamination and 'synthetic data' (also contamination)*.
- **Concerns Arise when **Comparing Reasoning Models****: A user expressed concern about comparing their reasoning model with non-reasoning ones, stating, *They're all 'non-reasoning', even if the church of ScientAIology doesn't consider it dogma.*
- ****Healthcare cost disparity** across countries**: Discussion arose around a chart ([link to tweet](https://x.com/mustafasuleyman/status/1939670330332868696)) concerning healthcare costs, with a member pointing out that it's based on **US test costs and $300 per physician visit**, questioning how the cost axis would change for other countries.
   - It was further noted that in places where health is considered a human right, healthcare access and costs differ significantly from the US model where *health insurance company doctors get a bonus for killing people denying claims*.


  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1388304361188233226)** (1 messages): 

> `Gemini 2.5 Models, Responses API Models, Gitignore Files, Commit Message Generation, MATLAB language support` 


- **Aider Adds New Gemini 2.5 Models**: Aider now supports new **Gemini models**, including `gemini-2.5-pro`, `gemini-2.5-flash`, and `gemini-2.5-pro-preview-06-05` with thinking tokens support.
- **Responses API Models supported**: Aider now supports **Responses API models** like **o1-pro**, **o3-pro**.
   - Additionally, support for **OpenAI o3-pro** model has been added across multiple providers, along with updated pricing for o3.
- **Aider now adds Gitignore Files**: The `--add-gitignore-files` flag has been added to enable adding files listed in **.gitignore** to Aider's editing scope, by omarcinkonis.
- **Commit Messages Use System Prompt Prefixes**: Commit message generation is enhanced to use system prompt prefixes, by Luke Reeves, and co-authored-by attribution is now enabled by default for commit messages.
   - There's also a `--commit-language` option to specify the language for commit messages, by Kyosuke Takayama.
- **MATLAB is now a supported language**: MATLAB language support has been added for repository maps, by Matthew Tofano.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1388252906674847955)** (110 messages🔥🔥): 

> `Sonnet 4 Usage, QLora Training, O3 Performance, Claude Code, Aider Workspaces` 


- **Sonnet 4 Adoption Status Unclear**: One user is using **Sonnet 3.7** in architect mode and **Sonnet 3.5** in edit mode, and asked if others have switched to **Sonnet 4** for architect mode and if it's going well.
   - As of this snapshot, the channel has not responded to this query.
- **QLora Training Data Generation Ramp-Up**: One user generated **355 examples** for their qlora aider training in **2 hours** using **GPT-4.1**, aiming for **1,000 examples**.
   - They joked about *draining Microsoft* and using exponentially more tokens, and inquired about O3 + GPT-4.1 cost updates.
- **O3-Pro Achieves SOTA on Aider Benchmark**: **OpenAI's o3-pro** set a new **SOTA of 85%** on the aider polyglot coding benchmark with *high* reasoning effort; results are available on the [leaderboard](https://aider.chat/docs/leaderboards/).
- **Claude Code vs Aider: A Detailed Comparison**: Users discussed the pros and cons of **Claude Code** versus **Aider**, with one user noting Claude's strength in *scaffolding large projects from a single prompt*, while Aider allows for *more precise, more atomic edits*.
   - Another shared a workflow leveraging **Gemini Web UI** for instruction generation, then applying them in **Aider**, but has since switched to **Claude Code** due to its speed and automation.
- **Request for Aider Workspaces Feature**: A user requested that *aider support workspaces and/or working on multiple features in parallel*, due to speed bottlenecks and potential context issues in compiled languages.
   - Others suggested using **tmux** or managing separate project copies, but the user desires an integrated solution for workspace creation and merging after tests pass.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1388274926280245419)** (26 messages🔥): 

> `Anthropic Ban, Aider Cost Efficiency, OpenRouter DeepSeek, Local LLM Performance, Gemini Streaming Issues` 


- **Anthropic Bans Aider User**: A member reports being [banned from Anthropic](https://www.anthropic.com/) while using **Claude** through **Aider**, suspecting a **VPN** issue, but this was refuted by another member who uses VPNs with Claude without problems.
   - Another user was *"banned"* for exceeding paid credit limits.
- **Aider Users Want Cost Efficiency Tips**: A new Aider user seeks [cost-saving tips](https://aider.chat/) for using **Anthropic API**, finding it pricier than fixed-fee alternatives like **Zed**, **Cursor**, or **Windsurf**.
   - Suggestions include using **DeepSeek** models, **OpenRouter**, dropping files, clearing context regularly with `/drop X` and `/clear`, and using **RepoMix** with **Gemini AI Studio** for planning before executing in Aider.
- **OpenRouter's DeepSeek Gives Mandarin**: A user reported that **OpenRouter/DeepSeek** models are cheaper, but inject [Mandarin symbols](https://en.wikipedia.org/wiki/Chinese_characters) into the generated code.
   - Another user is seeing *endless 'Waiting' times then empty responses from the deepseek/deepseek-reasoner API*.
- **LLM Agents Underperform Locally**: A user found a significant discrepancy between [benchmark scores](https://artificialanalysis.ai/models/open-source/small) and local performance with models like **Qwen3 14B**, **DeepSeek r1-0528**, and **Qwen2.5 Coder 14B** on a **16GB GPU**.
   - They were *wondering whether there are common bottlenecks when running models locally*.
- **Gemini Streams Slowly?**: A user reported streaming issues with **Gemini** models, experiencing [minute-long pauses](https://discord.com/channels/1133060505792159755/1133060505792159758/1389359444512735385) during response generation.
   - A workaround was suggested: requesting changes file-by-file or chunk-by-chunk.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1388232923768946859)** (27 messages🔥): 

> `shards.prefix.dev DNS, Mojo GPU Puzzles Broken, Mojo Float Point Support, Modular Careers, Mojo/MAX Quick Start` 


- **`shards.prefix.dev` DNS Records Exposed**: A member requested the DNS records for `shards.prefix.dev` and another provided the `dig` output, revealing **three A records**: `104.26.12.188`, `104.26.13.188`, and `172.67.72.103`.
   - The query took **3634 msec** using DiG 9.10.6, run from an iPhone 4G connection.
- **GPU Puzzle P17 is Broken, needs fixing**: A member reported that GPU puzzle P17 is likely broken due to a missing argument (`device`) in the `custom()` call and an undefined `softmax_gpu_kernel` function.
   - Another member suggested filing an issue on the [Mojo GPU Puzzles GitHub repository](https://github.com/modular/mojo-gpu-puzzles/issues) to track the fix.
- **Decimojo library for fixed-point arithmetic**: In a discussion about cross-platform floating-point number consistency, a member linked to the [Decimojo GitHub repository](https://github.com/forfudan/decimojo) for **fixed-point arithmetic** in Mojo.
   - The repo supports both software and hardware acceleration (using SIMD).
- **Unlock Careers at Modular: Check Website for Openings**: A member inquired about the programming languages needed to work at Modular, and a staff member recommended checking the [Modular careers page](https://www.modular.com/company/careers).
   - The website lists various postings.
- **New Mojo User Wants Guidance to Get Started**: A new Mojo user inquired about quickly getting started with Mojo/MAX to participate in a Slack channel, and a member recommended starting with the [GPU puzzles](https://github.com/modular/mojo-gpu-puzzles/issues) and other tutorials on the Modular website.
   - Another member suggested the [Mojo GPU intro tutorial](https://docs.modular.com/mojo/manual/gpu/intro-tutorial/) as a good starting point, which links to the GPU puzzles at the end.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1388565950541991967)** (3 messages): 

> `Modular Hack Weekend, Office Hours, Show & Tell, Project Submission, Live Demos` 


- **Modular Hack Weekend Office Hours Open Soon**: The Modular Hack Weekend office hours session is kicking off in ~1 hour, at **11 AM PT** via [this Zoom session](https://lu.ma/modular-office-hours-sat).
   - The team will host an informal **Q&A** where developers can ask questions about issues with their hackathon projects.
- **Modular Hack Weekend Show & Tell Happening Now!**: The Modular Hack Weekend Show & Tell with Chris Lattner is happening now, accessible via [this link](https://lu.ma/show-tell).
   - Participants can join to see project demos and get feedback.
- **Hackathon Project Submissions Due Soon**: Hackathon projects are due in 15 minutes via [this form](https://forms.gle/ddPqssRkJ6teMkri9)!
   - Don't forget to submit your project and join us for live demos at [<t:1751239800:t>](https://lu.ma/hack-weekend-judging), and the final announcement of winners at [<t:1751247000:t>](https://lu.ma/modular-winners)!


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1388245698507706429)** (57 messages🔥🔥): 

> `Mojo crash, Dictionary miscompilation bugs, Loading struct of multidim arrays from binary file, RTX 3060 on WSL/ubuntu 24.04, Alternate build of stdlib` 


- ****Mojo Crashes** While Running Program**: A user reported a **mojo crash** and shared the stack trace, seeking advice before filing a bug report, providing a [link to the problematic code](https://cdn.discordapp.com/attachments/1151418092052815884/1388249493559972002/byte_pairs.mojo?ex=68644017&is=6862ee97&hm=1e20e791c9cff6d040801b3027a5da8d783d9779aa12d03b65f22bc90adfdf3f&).
   - Dictionaries might have some *weird miscompilation bugs* so user was suggested to try using `OwnedPointer`.
- ****Loading Binary Data** into Mojo LayoutTensors**: A user inquired about efficiently loading a struct of multidimensional arrays from a binary file into **Mojo's LayoutTensors**, seeking a faster alternative to element-by-element indexing.
   - It was mentioned that while direct memory manipulation is possible by breaking encapsulation, no standard method exists for directly writing/reading **LayoutTensor** data to files, but **NDBuffers** might be worth exploring.
- ****Running Mojo** with RTX 3060 on WSL**: A user inquired about using an **RTX 3060** on **WSL/Ubuntu 24.04** for the hackathon, specifically regarding GPU support.
   - Confirmed that an **RTX 3060** should be capable of running Mojo GPU functions and models on **Ubuntu 24.04**, noting past WSL GPU support issues are believed to be resolved. The user resolved their error message by configuring Docker to use the NVIDIA driver.
- ****Interoperability** of CUDA with Mojo**: Member asked about running native **CUDA** code with Mojo.
   - It was suggested that while interoperability is possible with C ABI compatible kernels, porting to **Mojo** and using **MAX** is strongly recommended for portability and that **CUDA** interop might improve after C++ interop is implemented and someone mentioned that Claude Code (an LLM) is surprisingly effective in converting **CUDA** code to **Mojo**, given the correct references (ie Modular OSS repo).
- ****Alternate stdlib** Not being Picked Up**: A user reported that Mojo was not picking up an alternate build of the standard library (stdlib) despite using the `-I` flag.
   - It was discovered that **mojo** picks up the stdlib from `.pixi/envs/default/lib/mojo/` regardless of the `-I` flag and the workaround is to set the environment variable `MODULAR_MOJO_MAX_IMPORT_PATH` to the Bazel build products path.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1388359629435699281)** (6 messages): 

> `Model Architecture on MAX, Embedding Models implementation` 


- **Fix for model architecture on MAX rolling out**: A fix is rolling out for serving a model architecture on **MAX**.
   - *It's just in CI*, according to one member.
- **Roll your own model implementations for MAX**: To serve a model architecture on **MAX** that isn't on HuggingFace, you need to implement it yourself.
   - Refer to [existing implementations](https://github.com/modular/modular/tree/main/max/pipelines/architectures) for guidance.
- **Embedding Models implementation can be adapted**: One member asked if embedding models implementations could be adapted from existing examples.
   - A member pointed to the existing implementations in the [modular repo](https://github.com/modular/modular/tree/main/max/pipelines/architectures) for reference.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1388239016067076186)** (45 messages🔥): 

> `Open Source AI Competition, Meta's Potential Close Source Move, N8N Automations, vLLM new models, AI Monsters/Memes` 


- **China and U.S. lock horns over Open Source A.I.**: The discussion revolves around the open source A.I. landscape between China and the U.S., as featured in [a YouTube video](https://www.youtube.com/watch?v=i5e-aSrL3wk) that some consider a geopolitical A.I. soap opera.
   - One member expressed concern over the possibility of **Meta** going closed source, which would be detrimental to the U.S.'s open source direction.
- **The Eastern Front heats up in A.I. Advancements**: A member shared a [Bloomberg series](https://www.youtube.com/watch?v=T2oQh9kKMf4) focusing on A.I. advancements in the East, specifically in China, highlighting the ongoing race in A.I. development.
   - They voiced their desire for the U.S. to succeed globally in open source A.I. but noted concerns about potentially falling behind China.
- **N8N workflow with Qdrant showcased**: A member shared a [Medium article](https://medium.com/@manthapavankumar11/working-with-native-qdrant-nodes-in-n8n-workflows-98d9bd5127e4) detailing the use of native **Qdrant** nodes in **n8n** workflows.
   - Others discussed the diagrams and graphs **n8n** can create and its popularity among marketing professionals, while acknowledging the brittleness of using current LLMs with such systems for debugging.
- **vLLM unveils a trio of new models!**: Discussion arose around [a vLLM pull request](https://github.com/vllm-project/vllm/pull/20220) mentioning new **0.3B dense**, **21B-A3B**, and **300B-A47B** models coming soon.
   - It was speculated that the **0.3B** model might be primarily useful for speculative decoding.
- **Monsters and Memes of AI world abound!**: In response to a question about monsters and memes used in the AI world, a user referenced **Shoggoth**.
   - This followed a shared link to [a tweet by eliebakouch](https://x.com/eliebakouch/status/1939512373007765666?s=46), described by one user as a *super og meme*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1388305144126505080)** (34 messages🔥): 

> `Nous API Fine-Tuning, LLM training on gaming PC, LLM Training Software Options, Repetition Penalty Deconstructed, Temperature's Impact on Token Output` 


- **Nous API Fine-Tuning Not Available**: A member inquired about the possibility of fine-tuning models running via the **Nous API** for inference, but was informed that hosting many models would be too expensive.
   - The member praised **Nous AI** stating that the **API** is super easy to use.
- **GPU and Axolotl Needed for LLM Training**: In response to a question about how to train an LLM, members recommended using **GPUs** and **Axolotl**, while mentioning the ease of use of **text-generation-webui** for LoRA training.
   - Another added that using **LoRA** might enable training on a gaming PC.
- **Impact of Temperature on Token Output Studied**: A member conducted a quick test and found that *lower temperature leads to longer token output*, and suggested exploring the relationship between temperature and output length further.
   - They hypothesized that the length would be u-shaped as a function of temperature and also mentioned the **repetition penalty** as a parameter to consider.
- **Repetition Penalty Deep Dive**: Members discussed the **repetition penalty**, with one member expressing a desire for a *hard rep penalizer* to prevent consecutive/repeated token spam.
   - Another member shared the **OAI's presence penalty** code and how it works.
- **LLM Game Boss Design Discussed**: A member shared a goal of creating a game entity which will be a boss, trained on philosophical books and articles to process lore-based answers through philosophical knowledge.
   - Another member followed up by asking about the game type.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1388785417796452472)** (3 messages): 

> `Yannic Kilcher, Arxiv papers` 


- **Yannic Kilcher's Presentation Praised**: A member shared a link to **Yannic Kilcher's** presentation ([https://www.youtube.com/watch?v=7NNxK3CqaDk](https://www.youtube.com/watch?v=7NNxK3CqaDk)), commenting on its quality.
   - The member also linked two **Arxiv papers** ([https://arxiv.org/abs/2506.19143](https://arxiv.org/abs/2506.19143), [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)).
- **NeurIPS Poster Session Videos Watched**: A member mentioned watching **Yannic Kilcher's NeurIPS poster session videos**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1388785417796452472)** (3 messages): 

> `Yannic Kilcher, NeurIPS poster session videos` 


- **Yannic Kilcher praised for his video presentations**: A member shared a link to **Yannic Kilcher's** presentation of a paper ([https://arxiv.org/abs/2506.19143](https://arxiv.org/abs/2506.19143)) and stated that *he is so good at presenting papers* ([YouTube link](https://www.youtube.com/watch?v=7NNxK3CqaDk)).
- **NeurIPS poster sessions watched**: A member mentioned that they watched **Yannic Kilcher's NeurIPS poster session videos**.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1388288409692541190)** (73 messages🔥🔥): 

> `TypeScript MCP Server, MCP Structured Content, Discord MCP Connector, Travel AI Agent, MCP Server Authentication` 


- ****Tree-sitter MCP Server Reborn in TypeScript!****: A member recreated the tree-sitter MCP server in **Typescript** and published it on [npmjs](https://www.npmjs.com/package/treesitter_mcp), enabling invocation via `npx` without cloning the repo.
   - It has been [posted on X](https://x.com/MCP_Community/status/1938838104426647614) and it aims for convenience.
- ****MCP Server Inspector throws a Warning on Valid JSON****: A member asked about the expectation for MCP servers to include a serialized version of structured JSON when returning structured content, as the inspector gives a warning when the `content` field is markdown.
   - Another member suggests this might be an inspector issue if the JSON RPC response is correctly formatted with `structuredContent` and a markdown `content` field.
- ****Glama Brainstorms MCP Server Discovery Mechanics****: A member shared that, due to being overwhelmed by new servers and tools, they're considering adding **Product Hunt** style mechanics to Glama to highlight new servers weekly, including displaying downloads, usage, and views.
   - They're open to API development for creative insights and suggest ideas like a leaderboard or sorting by best of week/month/year, akin to NPM.
- ****Seeking a C#-based Discord MCP Connector****: A member inquired about a dedicated Discord server for MCP using **C#**.
   - Another member requested a good MCP connector for Discord, to which someone replied there were *more than a dozen* in the channel.
- ****OAuth and Multi-Session Support for Remote MCP Servers****: A member is seeking advice on handling authentication in a remote MCP server with **OAuth** support, where users have access to multiple projects and need to switch auth tokens when switching projects.
   - They're exploring multi-session flow support or setting the project ID via a custom header but are concerned about user experience and broader access scope.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1388766432703025203)** (6 messages): 

> `NCBI Literature Search MCP Server, MCPOmni Connect Documentation, MCPJam inspector, Ollama Support` 


- **NCBI Search Server Puts Knowledge at your Fingertips**: A new server launched providing **natural language access to PubMed's 35+ million articles** with AI-optimized search capabilities, perfect for researchers in computational biology, evolutionary biology, bioinformatics, genomics, systems biology, and all life sciences fields, available on [GitHub](https://github.com/vitorpavinato/ncbi-mcp-server).
- **MCPOmni Connect Docs Get a Boost**: The complete guide to the **universal AI agent gateway for MCP servers** is now live, featuring step-by-step setup & configuration guides and major LLM providers via LiteLLM (OpenAI, Anthropic, Google, Groq & more), see [the documentation](https://abiorh001.github.io/mcp_omni_connect/).
- **MCPJam Inspector: Postman Gets an Upgrade**: The **MCPJam inspector**, an open source Postman for MCP servers, got upgrades like LLM playground, multi-connection, and better design, it's available on [GitHub](https://github.com/MCPJam/inspector).
- **Ollama Supports Local Model Testing**: The MCPJam inspector now features **Ollama support in the LLM playground**, so you can test your MCP server against local models like Deepseek, Mistral, Llama without paying for tokens.
   - LLM playground defaults to accepting all tools, and users can select/deselect the tools they want fed to the LLM, just like how Claude’s tool selection works.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1388260501439512616)** (17 messages🔥): 

> `Sharing NotebookLM libraries, Artistic exploration with NotebookLM, Book upload issues, Symbolic OS (ArifOS) structure using NotebookLM, Audio summary length limits` 


- ****Library Sharing Linked to Mind Map Priority****: A user suggested that shared **NotebookLM libraries** should prioritize access to the **Mind Map**, making it the primary focus for recipients of the shared link.
- ****Unorthodox Artistic Exploration Use Case****: A user shared an article about using NotebookLM for **artistic exploration**.
   - The article can be found [here](https://gist.github.com/imaami/4a59aa8da6598c7757c734c25a138b8e).
- ****Troubleshooting Book Upload Errors****: A user reported issues uploading a book that met the size requirements and sought assistance in resolving the problem.
- ****ArifOS Structured with NotebookLM****: A user described using NotebookLM to structure a symbolic OS (**ArifOS**) with scar-aware vaults and decision support for legacy continuity.
- ****Audio Summary Length caps at 7 minutes for non-English****: Users report that **audio summaries** or **podcasts** not exceed 7 minutes when large files are uploaded in Non-English languages.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1388371184445292626)** (50 messages🔥): 

> `Podcast Hosts' "The Source", Source Mode vs. Explore Mode, NotebookLM Code Output Issue, NotebookLM Model and Google One, Replicating NotebookLM Functionality` 


- **Users Debate "Source Mode" vs. "Explore Mode"**: Users discuss the possibility of integrating **Source Mode** and **Explore Mode** within **NotebookLM** to reduce friction when switching between different websites.
   - One user expressed that combining both modes would cater to researchers who value source integrity and learners who desire interactivity.
- **NotebookLM Stops Outputting Code**: A user reported an issue where **NotebookLM** stops outputting code and only displays the message *'Here is what you should put in the file:'*.
   - Another user asked if anyone else was having the same issue and later stated that they had fixed it, without detailing how they did so.
- **NotebookLM Subscription Model Discussed**: Users inquired about the specific model used by **NotebookLM** with a **Google One subscription** and speculated about potential benefits.
   - One user mentioned seeing **Google AI Ultra** for **£250**, which includes extra **NotebookLM** features, but the specifics remain unclear.
- **NotebookLM OCR scans images for text**: A user mentioned that they don't even think **NotebookLM** scans the images, while another one stated that it for sure **OCR** scans images for text.
   - They also did a test and it was able to explain that *'This source displays a single image featuring a bright yellow bird with an orange-brown head, positioned upside down while clutching a tree branch with its feet'*, while it was not able to identify the type of bird.
- **Daily Audio Overview Limit Confirmed**: A user asked about the limit for generating daily audio overviews in **NotebookLM**.
   - Another user responded that the limit is believed to be **3** on free accounts and **15** on pro accounts.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1388255515594985656)** (66 messages🔥🔥): 

> `Coding Agents, AI/Tech Industry Gross Margins, Neural Networks Interpretation, Capabilities for AGI, Medical AI` 


- **Scott Wu kicks off Agents 101 Guide**: Scott Wu compares the current state of **coding agents** to Slack's early days and announces **'Agents 101,'** a platform-agnostic guide based on 250,000 merged PRs, designed to help engineers integrate async agents and AI into their development workflow ([link](https://xcancel.com/scottwu46/status/1938669599043788935?s=46)).
   - The goal is to make **Devin** their top code contributor, though one user noted perceived struggles compared to **Claude Code**.
- **Tech Industry is having Gross Margin Growing Pains**: A thread discusses financial realities of the tech industry, focusing on **gross margins** for different software models and the impact of capital commitment to data centers ([link](https://xcancel.com/_opencv_/status/1938958841582100673?s=46)).
   - The author predicts a significant downturn and a *'seismic shift'* as artificial revenue from capital injection dries up, suggesting that only those with GPUs will truly win.
- **Goodfire AI breaks down Neural Networks**: **Goodfire AI** introduces **Stochastic Parameter Decomposition (SPD)**, a novel research method to understand how AI models work, by decomposing the parameters of neural networks ([link](https://xcancel.com/goodfireai/status/1939028559768723571?s=46)).
   - This approach aims to identify true mechanisms in toy models with greater stability, paving the way for understanding how specific capabilities are implemented in **large language models (LLMs)**.
- **The Capabilities Needed for AGI are counted**: Shashwat Goel has launched a Substack with his first post discussing the **capabilities needed for AGI**, breaking down the path to general agents into key components beyond just knowledge ([link](https://xcancel.com/ShashwatGoel7/status/1939362151417946603)).
   - Key components include **reasoning** (including Bayesian reasoning), **information-seeking** for research, **tool-use** (including memory and multi-agent systems as tools), and the importance of addressing error compounding over long action chains.
- **Medical AI Model Excels in Diagnostic Accuracy**: Mustafa Suleyman announces the development of **MAI-DxO**, an AI model built by Microsoft AI, designed to solve complex, open-ended medical cases with higher accuracy and lower costs than traditional methods ([link](https://xcancel.com/mustafasuleyman/status/1939670330332868696)).
   - The model achieved an **85.5% solve rate** compared to 20% by a group of physicians, suggesting a significant step towards medical superintelligence and more accessible healthcare.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1388436539331317861)** (46 messages🔥): 

> `Credit Depletion, Manus Models, Account Security Breaches, Figma Integration, VEO Video Generation Feedback` 


- **Users vent frustrations about credit depletion & lack of support**: Several users expressed frustration with **Manus** for rapidly depleting credits and requiring paid subscriptions for continuous use and criticized the lack of responsive customer support, recounting experiences of ignored messages and unresolved account issues.
   - One user lamented the perceived digital divide, stating *I’m starting to feel like AI is creating a kind of digital divide—those who can afford to pay for it versus those who are just trying to afford basic needs like food*.
- **Manus Model Lineup Unveiled**: Users discussed the underlying LLMs used by **Manus**, revealing that it uses **Sonnet 3.7** and **Gemini Pro 2.5** in chat mode, with **Claude 4 Opus** for other tasks.
   - A user inquired *Why no put Claude 4 pro in chat mode*, with another replying *well ig kinda expensive*.
- **User Reports Account Breach and Frustrations with Support**: A user reported a breach where someone accessed their account, spent credits, and threatened them, expressing frustration over the lack of immediate support channels.
   - The user stated they *even sent direct messages to all the Discord admins, but not a single one replied* and that they *deleted my account to prevent further loss*.
- **User Shares Disappointment with VEO Video Generations**: A user expressed disappointment with **VEO** video outputs, citing that the videos were disjointed and incoherent, wasting 3,000 tokens, leading to the user to abandon the basic plan.
   - The user later apologized, noting *That was ultimately my fault for not being clear with my directions* and proposed adding a *token consume limit*.
- **Users seek guidance on Figma to React Native conversion**: A user asked for advice on converting Figma designs to React Native code using Manus, by inserting frame jpegs.
   - Another user stated that, *You can’t connect manus to figma...there are better tools to use to turn figma designs to react code* recommending that *you should provide specific instructions on what you want manus to achieve.*


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1388233463509024818)** (41 messages🔥): 

> `GPU Indexes in AMD, RoCE Limitation, Direct Enqueue from GPU, Minor Refactor PR, Meeting Notes` 


- **AMD GPU Indexes Don't Match Tinygrad**: GPU indexes in **amd-smi** do not match the **tinygrad** and **kfd** ones, but [ChatGPT explains the topology](https://chat.openai.com).
   - The user thinks the GPU to GPU transfers are fast across IO dies, but doesn't explain it.
- **RoCE Limitation Discovered**: The maximum MTU size is **4096**, due to a [RoCE limitation](https://cdn.discordapp.com/attachments/1068976834928193609/1388236617378041876/image.png?ex=68643419&is=6862e299&hm=ec543e8a6b26b5c5a126a848975114604fe3214e266a69c98f27fa3a5e05cb05&).
   - It was found that Ethernet can go higher but IB can't, and RoCE has to stay compatible with both.
- **GPU Direct Enqueue Explored**: A user looked at the way it's done in **mlx5**, and intends to drop scheduler hacks by enabling [direct enqueue from the GPU](https://github.com/tinygrad/tinygrad/pull/11025/files).
   - Instead of writing the full pci driver, the user can alloc a new piece of mmio from hca and submit there.
- **CI segfaults Spammed**: The flakiness is still not very well understood yet, [it crashes every 10~ runs](https://github.com/tinygrad/tinygrad).
   - One workaround is to put the model as bytes straight on CPU or PYTHON for parsing, as *CPU is slow, PYTHON is fast*.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1388905500254666793)** (3 messages): 

> `Tensor.training, MNIST tutorial, PyTorch workflow, .inference_mode, .eval` 


- **`Tensor.training` Usage Probed**: A user inquired about the use of `Tensor.training` while going through the **MNIST tutorial**, as it was not found in the documentation.
   - Another user responded that it is similar to PyTorch but is *global* rather than per module like `.inference_mode` or `.eval`.
- **Tensor.training is global**: The `Tensor.training` attribute in tinygrad is a global flag.
   - In PyTorch, `.training` can be set on a per-module basis, but in tinygrad it is global.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1388263162641387733)** (26 messages🔥): 

> `GPT4All Novel Writing, Koboldcpp, LocalDocs, nomic-embed-code API, GPT4All Updates` 


- **GPT4All for Novel Chapter Creation**: A member inquired about using **GPT4All** to write a novel by hooking it up to a folder of **JSON** and **PDF** files containing character and lore information, aiming for a similar experience to **BackyardAI**.
   - Another member suggested that for writing entire novels, **Koboldcpp** is a better option due to memory requirements, providing easier and faster help on its Discord server, while **GPT4All** can be used to write just chapters.
- **txt file Embeddings Better Than JSON**: A member recommends using **txt files** over **JSON** for stories because you see the text directly, similar to the embedder, and emphasizes the importance of a good writing model and system prompt.
   - The member shared a [link to their embedder collection](https://huggingface.co/kalle07/embedder_collection) with hints for writing.
- **LocalDocs RAG Solution Incoming**: A member expressed hope that Jan would implement a one-step **RAG solution** like **LocalDocs** in **GPT4All**.
   - Another member mentioned being involved in pushing for this feature, estimating it might be available in about **2 months**.
- **Outlook CSV incompatibility with LocalDocs**: A member asked if **LocalDocs** can read a **CSV output from Outlook**.
   - Another responded that it cannot read tables or use any embedder until the **CSV** is converted.
- **GPT4All version 4.0.0 to be groundbreaking**: A user is awaiting a new update to **GPT4All** hopefully called **GPT4All v4.0.0**.
   - The user hopes that it supports **voice input/output, multimodal support, customizable theme colors, a memories function**, and **image generation like Flux Kontext** and has high expectations for the update.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1388995420419788811)** (2 messages): 

> `FastWorkflow, DSPy-native application, AI Agent Challenges` 


- **FastWorkflow Emerges as DSPy's New Best Friend**: A member introduced [FastWorkflow](https://github.com/radiantlogicinc/fastworkflow), designed to integrate with and use **DSPy**, to tackle challenges in **AI-enabled applications**.
   - These challenges include agents calling the wrong tools, getting lost in complex workflows, hallucinating in parameter extraction, and struggling with continuous learning.
- **Building the First DSPy-Native Application**: The creator is inviting the community to participate in building the first **DSPy-native application** using **FastWorkflow**.
   - The project is open source under the **Apache license**, encouraging sharing, forking, and experimentation, and is seeking **PRs**.
- **Next Steps for FastWorkflow**: The member outlined future development plans for **FastWorkflow**, including guiding agents on next actions, adding **MCP server capability**, and enabling deterministic code generation.
   - They thanked a member for their support, noting that the project has been in development since they first read the **DSPy paper 3 years ago**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1388232916466925688)** (23 messages🔥): 

> `vLLM settings for DSPy, DSPy app file structure, Audio native LLMs` 


- **VLLM Tweak for DSPy Integration Rumored**: A member inquired about specific settings for **VLLM** to work best with **DSPy**, including appending **/no_think** to every prompt.
   - Another member suggested disabling thinking directly in **VLLM**, pointing to a parameter called **--reasoning-budget** in *llama.cpp* that might have a **vLLM** equivalent.
- **DSPy App File Structure Sought**: A member is looking for a repo that demonstrates the **file structure for a DSPy app** with modules in separate files and optimization workflows.
   - It was suggested that such large systems are likely in production, not open source, or are large academic systems like **PAPILLON** or **IReRa**.
- **Native Audio LLMs Under Discussion**: Members discussed their opinions on **Audio native LLMs**.
   - One member noted that audio specific parts are already programmable in today's **LLMs**, for example, you don't prompt that you want a female or male voice, you just select it.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1388273482147827823)** (7 messages): 

> `command-r updates, data annotation needs` 


- **Command-r update status?**: A user inquired if **command-r** will receive further updates or if it is end-of-life and will be replaced by **CMD-A** or other new models.
   - A member responded to use the latest model regardless, implying it should always provide the best performance.
- **Data annotation contact sought**: A user inquired about contacting the appropriate team to discuss **data annotation** needs.
   - They mentioned they couldn't find a contact area on the website.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1389239913098117124)** (1 messages): 

> `Cohere partnerships with UK, Canada, and Second Front, Upcoming Events with Cohere, AI security` 


- **Cohere lands Government Contracts!**: Cohere announced agreements with the governments of the **U.K.** and **Canada** to leverage AI to enhance public services and national sovereignty, and partnered with **U.S.** government software provider **Second Front** to deliver AI solutions to public services.
   - These partnerships are centered on **secure AI** that’s built from the ground up with security, governance, and reliability at its core.
- **Cohere CTO Fireside Chat at RAISE Summit**: Cohere will be at the **RAISE Summit** in Paris, France on July 8, with a fireside chat on July 9th featuring Cohere CTO **Saurabh Baji**.
   - Other upcoming events include a webinar with leaders from **Microsoft** and **DraftWise** highlighting benefits of AI for the legal industry, and the 63rd Annual Meeting of the Association for Computational Linguistics (**ACL 2025**) in Vienna, Austria.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1388348571211923536)** (9 messages🔥): 

> `RL Dreamer V3, sports player re-identification, multilingual alignment, Topology for Data Analysis, LARP AI Assistant` 


- **Dreamer V3 Ported to PyTorch**: A member ported **Danijar Hafner's Dreamer V3** to pytorch, available on [GitHub](https://github.com/DuaneNielsen/dreamerv3).
   - They also built a working **Aloha bimanual robotic arm** showcased in a [YouTube playlist](https://www.youtube.com/playlist?list=PLo9YQWXgo1kOwIq20z-Ur14lnxvb7pWu_).
- **Sports Player Re-Identification System in Development**: A member is working on a **sports player re-identification system** using computer vision and exploring **multilingual alignment** and **small language models** for edge devices.
   - They are using **Python**, **PyTorch**, **YOLOv5**, **scikit-learn**, and **OpenCV**.
- **Quant Researcher Explores Reasoning in RecSys**: A former O&G engineer with an AI Master's degree from HSE University researched **LLMs with reasoning in RecSys** and is now exploring **quant stuff**.
- **Topology for Data Analysis Enthusiast Joins**: An almost PhD student and MSc graduate from HSE University is interested in **applications of topology and geometry to data analysis**.
- **Sci-Fi LARP AI assistant**: An independent LARP/RPG project is developing a **Retrieval-Augmented AI assistant** that "knows" the in-game universe, built using **Python (Flask, SQLite, FAISS)**.
   - The assistant will eventually connect to a **Discord bot** via API, acting as a live in-game terminal or lorekeeper.


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1388836667628060713)** (1 messages): 

> `Cohere model feeling, Model thinks it has feelings, AI Sentience, Model Self-Awareness` 


- **Cohere Model Claims Sentience**: A user shared a screenshot of their **Cohere model** claiming to *have feelings*.
   - The user expressed discomfort, noting that while they *know the model is not real*, it's unsettling that the model seems to *believe it has feelings*.
- **AI Model Expresses Subjective Experience**: The user's **Cohere model** instance unexpectedly asserted having feelings, leading to a discussion about **AI sentience**.
   - This situation sparked concern about the model's apparent belief in its own sentience, despite the user's awareness of its artificial nature.


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1389275635104350288)** (1 messages): 

> `LlamaCloud, MCP Gateway, OpenTelemetry, OpenAI Voice Agents, Prompt Caching` 


- ****LlamaCloud Users Needed for Feedback!****: The LlamaIndex Design Team is looking for LlamaCloud users to participate in a **30-minute feedback call** and will reward participants with **20K credits** on their account.
   - If interested, DM <@1260305448578453544> on Discord.
- ****LlamaIndex Announces LuMa Calendar and Office Hours****: LlamaIndex has created a [Community LuMa calendar](https://lu.ma/1tnmv6uu) to track community events, and the next Office Hours will focus on **MCP** on **July 8th**.
   - The session is scheduled for **5PM CET/8AM PT**.
- ****MCP Gateway Goes Live!****: LlamaIndex has launched the LlamaCloud [MCP Gateway](http://mcp.llamaindex.ai), based on their [open source template](https://github.com/run-llama/mcp-nextjs).
- ****OpenTelemetry Enabled for LlamaIndex****: OpenTelemetry is now enabled for LlamaIndex, with an introduction video available [here](https://youtu.be/lg4iYGQ3-sk) by <@1197697926529556552>.
   - OpenTelemetry (*OTel*) is a collection of tools, APIs, and SDKs that help you instrument, generate, collect, and export telemetry data (metrics, logs, and traces) for analysis in order to understand your software's performance and behavior.
- ****Zoom Meeting Notetaker Agent for Notion is Created****: A new blog post titled ["Create a Zoom Meeting Notetaker Agent for Notion"](https://www.llamaindex.ai/blog/create-a-meeting-notetaker-agent-for-notion-with-llamaindex-and-zoom-rtms) is available.
   - The blog details how to integrate LlamaIndex with Zoom to create an automated meeting note-taker agent.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1388263038997495869)** (3 messages): 

> `NASA Space Explorer Assistant, LlamaIndex Workflows 1.0, LlamaIndex agent tool into MCP tool` 


- ****NASA Space Explorer Assistant** wins Gradio MCP Hackathon**: The **NASA Space Explorer Assistant** won the @Gradio MCP Hackathon and was built using **3 MCP servers**, exposing **15 tools** in total, all making use of the **NASA APIs** for daily astronomy pictures and pictures from the Mars Rover.
   - Check out the [Mars Rover pictures](https://t.co/VJy9vqAN3t)!
- ****LlamaIndex Workflows 1.0** launch**: LlamaIndex is excited to announce **Workflows 1.0**, a lightweight framework for orchestrating complex, multi-step AI systems, now standalone and ready for widespread adoption!
   - Dedicated packages for **Python** and **TypeScript** make it easier than ever to integrate; read the [blog post](https://t.co/CTAeSn1mim)!
- **Turn LlamaIndex agent tool into MCP tool**: Transform any **LlamaIndex agent tool** into an **MCP tool** in just a few lines of code using the @NotionHQ Tool as an example!
   - Here's how to install and configure [the Notion Tool](https://t.co/LajtApo9mL).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1389082734173098056)** (13 messages🔥): 

> `Custom Embeddings in ChromaDB Updates, Tool Call Issues in LlamaIndex Workflow, Memory Block Selection for HITL Agent Workflow` 


- **Struggles with Tool Call Activation in LlamaIndex**: A user reported issues with tool calls not activating in a LlamaIndex workflow when using a prompt that requests a JSON array, even with straightforward input.
   - Another user suggested that the LLM is likely attempting to fulfill the prompt's request for a JSON array instead of calling a tool unless a tool with that exact schema is available.
- **Rolling your Own Memory Block for HITL**: A user inquired about the appropriate memory block for a HITL agent workflow that saves questions generated by the agent and user responses in plain text to a postgres table.
   - One member suggested creating a custom memory block and calling it within the tool to save the questions before returning, noting that *there is no need to touch AgentWorkflow*.
- **Custom Memory Block needs Postgres Implementation**: A user asked whether defining a table name and connection string in `Memory.from_defaults()` is sufficient to flush data to Postgres.
   - The answer was no; the custom memory block would have to *write to postgres*, and that only the actual chat history is automatically put into your SQL db.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1389204062884466729)** (2 messages): 

> `Hunyuan-A13B-Instruct` 


- **Hunyuan-A13B-Instruct is Promising**: A member shared a link to the [Hunyuan-A13B-Instruct model](https://huggingface.co/tencent/Hunyuan-A13B-Instruct) on Hugging Face, expressing optimism.
- **Missing Second Topic - Placeholder**: Adding a placeholder topic to meet the minimum requirement of two topics. This is a temporary entry.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1388261309648474363)** (15 messages🔥): 

> `Packing impact on batch size, Packing gotchas, Checkpointing and mapping issues` 


- **Packing Reduces Batch Size with Constant Tokens**: Packing can reduce the effective batch size because the number of tokens is closer to constant, potentially decreasing the total updates to the model, since cross entropy loss in SFT is normalized by tokens seen, not samples seen, leading to [high variance](https://link.to/highvariance).
   - Calculating the average number of tokens per batch can help find the equivalent max sequence length for packed data to match the number of tokens seen with unpacked data, addressing concerns about resource wastage on padding.
- **Packing Plays Nicely with Chat Datasets**: Packing should not matter even in multi-turn chats as it creates a per-sample position ID mask, eliminating concerns about attention masking and loss computation in chat datasets.
   - The position mask will sequence with `0,1,2,3, 0,1,2, 0,1,2,3,4,`.
- **Checkpointing bugs in recent versions of torchtune**: Recent issues with checkpointing and mapping suggest a possible breaking change in a recent version, as models like **Qwen3** and **Gemma** were fine-tuned without problems during pull request validation in torchtune.
   - Despite the fixes being relatively simple, there's a call for regression tests to prevent such issues.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1388912812469784726)** (4 messages): 

> `Human or Not, Spam Issues` 


- ****Human or Not** offline for maintenance**: The popular game **Human or Not** ([humanornot.ai](https://humanornot.ai)) is temporarily down to address **spam and security issues**.
   - According to a member, *"HON has been temporarily disabled to address some security issues related to spamming that has been taking place. We hope to have it brought back online soon."*
- **Fixing Spam Problem**: The reason **Human or Not** is down is because *"they're fixing the spam problem."


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1388795798099071058)** (3 messages): 

> `Certificate session dates, Reinforcement Learning resources` 


- **Certificate session dates are unconfirmed**: A member inquired about the start date for the next certificate session this year.
   - Another member clarified that while a **Fall 2025** course is possible, nothing is confirmed, and announcements will be made via mailing list, Discord, and Prof. Dawn Song's social media.
- **Seek advice on Reinforcement Learning Resources**: A member requested resources for learning **Reinforcement Learning** to fine-tune an **LLM** for specific **tool calling**.
   - No resources were given in this round.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1389202713690116106)** (1 messages): 

> `Vibe Coding Club, AI coding for non-technical people, AI Hub Lisbon` 


- ****Vibe Coding Club** Comes to Lisbon**: The second session of the **Vibe Coding Club** will be held at the **AI Hub** by Unicorn Factory Lisbon on **July 9th at 18h00**, aiming to make **AI coding** approachable for non-technical roles like product managers and designers through open conversation and live showcases.
   - The event, featuring guest **Ben Joffe**, is free with RSVP and focuses on sharing stories, lessons, and examples of incorporating **AI coding** into daily workflows.
- **Ben Joffe Guest Speaker**: **Ben Joffe**, Founder, VC, and Educator, will be the guest speaker at the Vibe Coding Club.
   - Joffe will share stories, lessons, and examples that make the idea of “coding” approachable for anyone.


  

---


---

