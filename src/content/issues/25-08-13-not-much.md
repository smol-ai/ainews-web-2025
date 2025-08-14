---
id: MjAyNS0w
title: not much happened today
date: '2025-08-13T05:44:39.731046Z'
---

**a quiet day**

> AI News for 8/12/2025-8/13/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 8451 messages) for you. Estimated reading time saved (at 200wpm): 696 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Small updates to GPT5 continue (see Twitter Recap).

Since it's quiet, why not hack on an agent and compete for $150k in cash prizes with our friends at MiniMax (of [MiniMax-M1 fame](https://news.smol.ai/issues/25-06-16-chinese-models))?

---

[](https://resend-attachments.s3.amazonaws.com/SiP8BRwkgadkEqG)

🚀 **$150,000 MiniMax AI Agent Challenge** — Bring Your A-Game!

- 💡 Build from scratch or remix projects — 200+ **prizes** await.
- 🗓 **Submit by Aug 25** → https://minimax-agent-hackathon.space.minimax.io/
- Don’t just imagine what you can build with AI — **prove it**.
- More details are in the official Luma page https://lu.ma/2u17h1zw

---

# AI Twitter Recap

**OpenAI GPT-5 product updates, routing economics, and evals**

- [@sama](https://twitter.com/sama/status/1955438916645130740): GPT-5 now supports “Auto/Fast/Thinking” modes in ChatGPT with GPT‑5 Thinking at 196k tokens, 3,000 msgs/week, and overflow to GPT‑5 Thinking mini. 4o returns to the picker; “Show additional models” exposes o3/4.1/GPT‑5 mini; 4.5 remains Pro‑only due to GPU cost. Personality changes are coming plus per‑user customization.
- Monetization via routing: Multiple observers argue the real “GPT‑5 release” is the router that dynamically sends requests to cheaper models to cut compute costs ([@dylan522p](https://twitter.com/dylan522p/status/1955433082397589900); “router will get very good very fast” per [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1955633697635631112)). As [@jefrankle](https://twitter.com/jefrankle/status/1955634983021998252) notes, reliable signals are needed to learn good routing decisions. Separately, Plus vs Pro users appear to have different “thinking budgets” ([@scaling01](https://twitter.com/scaling01/status/1955610515134460285)).
- Serving variance matters: For GPT‑OSS‑120B, [@jeremyphoward](https://twitter.com/jeremyphoward/status/1955438370274087369) recommends Fireworks, DeepInfra, and Together as accurate hosts. [@giffmana](https://twitter.com/giffmana/status/1955710876528599217) says Microsoft/Amazon reportedly used older vLLM defaults and medium reasoning effort, explaining lower quality and “>10% degradation” complaints (called “fraud” by [@nrehiew_](https://twitter.com/nrehiew_/status/1955613510463037611)).
- Evals snapshot: GPT‑5 topped FrontierMath with nuanced deltas; holdout vs non‑holdout performance and “unguessable” answers detailed by [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1955667249252978741). On RooCode, GPT‑5 is 55% slower but ~40% cheaper vs Sonnet 4 ([@scaling01](https://twitter.com/scaling01/status/1955669720843358502)).

**GPT‑OSS: base model extraction, hosting, and low‑level tooling**

- Base extraction from reasoning model: [@jxmnop](https://twitter.com/jxmnop/status/1955436067353502083) released gpt‑oss‑20b‑base extracted from OpenAI’s reasoning checkpoint, crediting [@johnschulman2](https://twitter.com/johnschulman2). Next steps: generation checks for memorization, instruction tuning, and trying 120B ([follow‑up](https://twitter.com/jxmnop/status/1955436118620488059)). Community discussion cautions on calling it a “base model” and suggests probing train‑data leakage via perturbations ([@eliebakouch](https://twitter.com/eliebakouch/status/1955479573489213593), [@florian_tramer](https://twitter.com/florian_tramer/status/1955510942252572946), [@OfirPress](https://twitter.com/OfirPress/status/1955463664556769426)).
- Hosting and orchestration: gpt‑oss‑120B shows strong tool‑calling for multi‑tool pipelines in a single prompt ([@reach_vb](https://twitter.com/reach_vb/status/1955678303395696821)). Infra work on the OSS stack includes a high‑throughput training/inference PR with flex‑attention, complex freqs, grouped‑GEMM MoE, and checkpoint converters ([@khoomeik](https://twitter.com/khoomeik/status/1955433361402724679)).

**Anthropic: Opus‑plan/Sonnet‑execute, 1M context, prompt caching, Humanloop**

- Model pairing in code: “Opus plan, Sonnet execute” is now officially supported in Claude Code via /model, routing high‑level planning to Opus 4.1 and task execution to Sonnet 4 ([@_catwu](https://twitter.com/_catwu/status/1955694117264261609); [@alexalbert__](https://twitter.com/alexalbert__/status/1955687538129252807)). Sonnet 4 context expands to 1M tokens on the API ([@claude_code](https://twitter.com/claude_code/status/1955471002353242605)); prompt caching TTL is now 1 hour GA ([docs](https://twitter.com/claude_code/status/1955475387858972986); [@alexalbert__](https://twitter.com/alexalbert__/status/1955709585999978613)). Cline added immediate support for Sonnet‑1M ([@cline](https://twitter.com/cline/status/1955776052644732938)).
- Team moves: The Humanloop team joins Anthropic to accelerate safe enterprise adoption ([@humanloop](https://twitter.com/humanloop/status/1955487624728318072); [@RazRazcle](https://twitter.com/RazRazcle/status/1955488872235929712)).

**DSPy 3.0 and the rise of prompt/black‑box optimizers**

- DSPy 3.0 ships with GRPO/RL training, SIMBA, and GEPA; the latter is touted as beating RL on prompt optimization ([@CShorten30](https://twitter.com/CShorten30/status/1955445406441033906); [@MaximeRivest](https://twitter.com/MaximeRivest/status/1955431980868542692)). Practitioners are already adapting GEPA (e.g., to Observable JS) ([@LakshyAAAgrawal](https://twitter.com/LakshyAAAgrawal/status/1955455810802421991)).
- Ecosystem traction: multi‑language DSPy ports, production use for agentic flows, small demos like a <200 LoC agent with DSPy + bash, and detailed guides to MIPROv2 configuration ([@lateinteraction](https://twitter.com/lateinteraction/status/1955419751246934187), [@JuiceSharp](https://twitter.com/JuiceSharp/status/1955460115957682444), [@rasmus1610](https://twitter.com/rasmus1610/status/1955617801802260691), [@heylegacyguy](https://twitter.com/heylegacyguy/status/1955682283270078484)).

**Open models, toolchains, and leaderboards (Qwen, GLM, qqWen, Kimi, Mistral)**

- Qwen momentum: Qwen3‑Coder is live ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1955436295603490864)); Deep Research now supports image/file inputs ([tweet](https://twitter.com/Alibaba_Qwen/status/1955642787619381325)); Qwen Image is faster on Qwen Chat and Image Edit is in testing ([tweet](https://twitter.com/Alibaba_Qwen/status/1955656265499316406); [tweet](https://twitter.com/Alibaba_Qwen/status/1955656822532329626)). The open‑finetuning suite qqWen (1.5B–32B) publishes code/weights/data for a niche financial programming language (Q) across pretrain+SFT+RL ([@brendanh0gan](https://twitter.com/brendanh0gan/status/1955641113693561071)).
- GLM and coding IDEs: Zhipu’s GLM‑4.5 integrates natively with Kilo Code; devs report quality gains ([@Zai_org](https://twitter.com/Zai_org/status/1955627932543840510); [@Kilo_Code](https://twitter.com/Kilo_Code/status/1955629042205696084)).
- Leaderboards: On LmArena’s August Text Arena, Qwen‑3‑235b‑a22b‑instruct takes #1 among open models; GLM‑4.5 debuts #4; OpenAI gpt‑oss‑120B debuts #7; top open models place within the overall top‑50 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1955669431742587275)).
- Tooling: Anycoder adds Mistral Medium 3.1 ([@_akhaliq](https://twitter.com/_akhaliq/status/1955621767302808012)). Hugging Face TRL now supports VLM SFT, multimodal GRPO, and MPO ([@mervenoyann](https://twitter.com/mervenoyann/status/1955622287920537636)).

**Agents, evaluation, and infra debugging**

- Tool‑use benchmarks: LiveMCPBench evaluates 10 frontier models on 95 time‑sensitive tasks across 527 tools. Claude Sonnet 4 leads with 78.95% success; the dominant failure mode is tool discovery, not execution. Costs track performance; LLM‑as‑judge agrees with humans ~81% ([@_philschmid](https://twitter.com/_philschmid/status/1955601309966447074); [paper](https://twitter.com/_philschmid/status/1955601312059461681)).
- Real‑world agent gap: METR finds a gap between algorithmic scoring and practical usability for autonomous agents on real software tasks from their RCT, calling for broader but usable metrics ([@METR_Evals](https://twitter.com/METR_Evals/status/1955747420324946037)).
- System/UI support: LangChain releases a Deep Agents UI for TODOs, file systems, and subagents ([@LangChainAI](https://twitter.com/LangChainAI/status/1955674201853247584)); DAIR AI announces a hands‑on “Building Effective AI Agents” course ([@dair_ai](https://twitter.com/dair_ai/status/1955623925901353351)).
- Infra debugging and perf: vLLM details CUDA core dump debugging with recommended env vars ([blog](https://twitter.com/vllm_project/status/1955478388178817298)); Jina shares GGUF embedding optimization on L4 (IQ3_S, batch 512, c=2048) reaching ~4,143 tok/s with ~2GB VRAM ([thread](https://twitter.com/JinaAI_/status/1955647947359867068)).

**Applied product launches: Perplexity Comet and Finance, plus multimodal video tools**

- Perplexity: Comet desktop app is rolling out to US Pro users (Mac/Windows), with Max Assistant for agentic prompts to Max subscribers ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1955684209483534657); [@AravSrinivas](https://twitter.com/AravSrinivas/status/1955684921974087807)). Perplexity Finance expands to India with BSE/NSE coverage, live earnings, Excel downloads, and upcoming NL stock screening and alerts ([@jeffgrimes9](https://twitter.com/jeffgrimes9/status/1955487020647850437); [@AravSrinivas](https://twitter.com/AravSrinivas/status/1955489224511328514)).
- Video generation/editing: Runway’s Aleph enables precise regional edits and retexturing in video, turning multi‑step VFX into a single prompt ([@runwayml](https://twitter.com/runwayml/status/1955615613583519917); [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1955687077825183952)). Tencent’s “Yan” builds on self‑forcing for interactive video ([@xunhuang1995](https://twitter.com/xunhuang1995/status/1955645976917811411)). Hailuo 2 Pro leads among silent video models ([@Hailuo_AI](https://twitter.com/Hailuo_AI/status/1955453164645429350)). Elon shared “how to make videos of any length” with Grok Imagine ([@elonmusk](https://twitter.com/elonmusk/status/1955710887094050994)); Higgsfield demoed “Draw‑to‑Video” across top models ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1955742643704750571)).

**Top tweets (by engagement)**

- [Elon: Grok now auto‑translates non‑English posts for US users](https://twitter.com/elonmusk/status/1955457039620247861) — 29.9k
- [Sam Altman: GPT‑5 product updates and limits](https://twitter.com/sama/status/1955438916645130740) — 24.9k
- [Elon: How to make videos of any length with Grok Imagine](https://twitter.com/elonmusk/status/1955710887094050994) — 12.1k
- [OpenAI GPT‑OSS “20B base” extraction thread](https://twitter.com/jxmnop/status/1955436067353502083) — 5.9k
- [Igor Babuschkin departs xAI, launches safety‑focused fund](https://twitter.com/ibab/status/1955741698690322585) — 5.4k
- [Perplexity Finance launches India coverage](https://twitter.com/AravSrinivas/status/1955489224511328514) — 5.2k
- [Claude Code: Opus for planning, Sonnet for execution (/model)](https://twitter.com/_catwu/status/1955694117264261609) — 1.3k

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen Model Real-World Local Usage Reports

- [**God I love Qwen and llamacpp so much!**](https://v.redd.it/ur3oxzhnmsif1) ([Score: 561, Comments: 126](https://www.reddit.com/r/LocalLLaMA/comments/1mp5bjc/god_i_love_qwen_and_llamacpp_so_much/)): **The OP reports running local batch inference with the Qwen3 30B Instruct LLM using llamacpp on a single NVIDIA RTX3090 GPU, processing 4 requests in parallel. This approach is being leveraged for large-scale data processing and platform insights, but the user notes hitting VRAM or throughput bottlenecks and anticipates moving to a multi-GPU setup. (See [Qwen LLM](https://github.com/QwenLM) and [llamacpp repo](https://github.com/ggerganov/llama.cpp) for model and backend details.)** One commenter requests technical details on how parallel batch inference was achieved, indicating demand for practical guides on running large LLMs locally with consumer GPUs and batching. No substantive technical debate in the comments, only interest in replication.
    - A commenter inquires about practical use cases for batch inference when running language models like Qwen locally, suggesting interest in understanding efficiency gains or scenarios where batch inference would benefit a local, possibly single-user setup.
    - Technical curiosity is revealed as another user asks how the showcased setup (Qwen model with llamacpp) is achieved locally, implying a request for steps or instructions for reproducing multi-modal or advanced inference pipelines in a home environment.
    - There is a subtle reference to screen aspect ratio (21:9) which, while not deeply technical, hints at running models or UIs on ultra-wide displays, potentially relating to productivity or visualization strategies for working with large models locally.
- [**Fully local Qwen 2.5 Omni realtime agent (sees + talks).... tested it by cooking dinner**](https://v.redd.it/m9ttqovtmtif1) ([Score: 211, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1mpayu9/fully_local_qwen_25_omni_realtime_agent_sees/)): **The user deployed a fully local pipeline using the Qwen 2.5 Omni model as a real-time agent, processing webcam video input frame-by-frame and delivering overlayed AI responses in ~1 second latency. Implementation employed the open Qwen model for scene interpretation and highlights included solid single-turn conversation and image-to-text reasoning, but notable weaknesses were found in multi-turn dialog stability and hallucination rate, as well as poor audio understanding unless input was very clean. The repository used is [gabber-dev/gabber](https://github.com/gabber-dev/gabber).** A technical inquiry was raised about the specific model variant in use (in particular, if it is the gguf format), but no further deep discussion of implementation or benchmarks followed.
    - A user inquired specifically about which variant of the Omni model was used in the project, asking if it was the GGUF format, which is often important for compatibility with local deployments and quantized inference engines.
    - Another comment provided a direct link to the GitHub repository ([gabber-dev/gabber](https://github.com/gabber-dev/gabber)), which is relevant for anyone interested in reviewing the source code and assessing the technical implementation details.
    - One comment praised the decision not to include a code editor for LiveKit, suggesting this is a deliberate and smart design choice potentially related to security or minimalism. The commenter also suggested that LiveKit should consider funding such efforts, indicating a recognition of the technical value or novelty of the approach.

### 2. gpt-oss-120B Model Benchmarks and Limitations

- [**gpt-oss-120B most intelligent model that fits on an H100 in native precision**](https://i.redd.it/4okvse7e2rif1.jpeg) ([Score: 305, Comments: 218](https://www.reddit.com/r/LocalLLaMA/comments/1moz341/gptoss120b_most_intelligent_model_that_fits_on_an/)): **The image presents a scatter plot comparing various AI language models by their 'Artificial Analysis Intelligence Index' (a proxy for model evaluation) and 'Active Parameters at Inference Time' (log scale), with particular emphasis on the gpt-oss-120B model. It highlights that gpt-oss-120B, purported to be the 'most intelligent model that fits on an H100 GPU in native precision,' occupies a favorable position (high intelligence index, moderate parameter count). The analysis implies a tradeoff between intelligence and inference resource use, favoring models like gpt-oss-120B that balance both. [Image link.](https://i.redd.it/4okvse7e2rif1.jpeg)** Technical commenters challenge the framing of 'native precision' (4-bit quantization) as a unique advantage, noting competitive performance from other 4-bit models, and caution against marketing hype. One commenter seeks direct benchmarks comparing gpt-oss-20B and Ernie 4.5 21B, highlighting gaps in current model comparisons.
    - There is skepticism about advertising claims based on "native" precision, which refers to running gpt-oss-120B in 4-bit quantization; several commenters note that other models quantized to 4-bits outperform it, so "native quant" doesn't inherently confer an advantage.
    - A technically relevant omission is the lack of a benchmark comparison between gpt-oss-20B and Ernie 4.5 21B, even though these models are similar in active and total parameter counts. Accurate performance comparisons require side-by-side benchmarks.
    - The Qwen3 30B model is highlighted as outperforming gpt-oss-20B on existing evaluation charts, casting doubt on claims about gpt-oss-20B's leading intelligence for models that fit consumer GPUs.
- [**Peak safety theater: gpt-oss-120b refuses to discuss implementing web search in llama.cpp**](https://i.redd.it/j7hi9xgjrrif1.png) ([Score: 251, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1mp1j7e/peak_safety_theater_gptoss120b_refuses_to_discuss/)): **The image demonstrates a notable 'safety refusal' by the gpt-oss-120B model, where the model declines to provide instructions for adding web search to llama.cpp, citing policy restrictions. Technical discussion in the comments highlights that such censorship behavior can be mitigated by adjusting inference parameters—specifically, increasing 'temperature' to 1.0, and using 'Top_K=0' or 'Top_K=100' with 'Top_P=1.0', which prompts the model to respond without refusal. This suggests the refusal is not hard-coded, but emerges from sampling strategy and likely reflects prominent training tokens in the model's output distribution.** Commenters debate the implications of such refusals, with some noting that these can be circumvented simply by tuning parameters—a characteristic of many so-called 'censored' models. Others express concern that heavily weighted refusal tokens might be problematic and reflect questionable choices in the training or fine-tuning process.
    - Adjusting inference settings (such as **Temperature: 1.0, Top_K: 0/100, Top_P: 1.0**) can mitigate refusals from gpt-oss-120b, indicating that many 'censored' models can be "decensored" by tweaking sampling parameters, rather than requiring model retraining or hacking.
    - Detailed reproduction in **Ollama** using gpt-oss-120b (native MXFP4 quant) shows *no refusals* and a complete breakdown of how to implement web search in llama.cpp, including: use of external search APIs (SerpAPI, Google, Bing, etc.), retrieval-augmented generation (RAG) pipelines, leveraging wrappers like LangChain, llama_server, and including example code and potential pitfalls, highlighting that refusals may be environment- or quantization-specific rather than inherent to the model weights themselves.
    - The debate in the comments points out that if lower temperature (higher determinism) causes a model to default to refusals, it may indicate that the refusals are heavily baked into the fine-tuning or are disproportionately common tokens—raising concerns about practical use and the robustness of 'uncensoring' via mere sampling tricks.

### 3. Nano-banana Text-to-Image Model Launch

- [**There is a new text-to-image model named nano-banana**](https://i.redd.it/jmw88evj4sif1.png) ([Score: 262, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1mp2wq3/there_is_a_new_texttoimage_model_named_nanobanana/)): **The post introduces a new text-to-image model named 'nano-banana' and illustrates its capability with an example: it reconstructs a full face from a partial (eye-level) image input. The image demonstrates that the model outputs a high-fidelity portrait consistent in features with the partial input, suggesting strong image completion or inpainting capabilities. Commenters speculate about its application in image-editing, referencing prompt-driven transformation tasks and comparing it to Gemini-powered image generation (though this is stated humorously in context).** Commenters debate whether the model is related to Gemini's image generation, with one suggesting it's a good fit for advanced image-editing tasks—particularly character or style transformations based on prompts.
    - A commenter notes strong results from the model (nano-banana) for image-editing scenarios, referencing a sample where the prompt successfully altered the depicted characters to resemble 2B from Nier: Automata and Master Chief from Halo, suggesting the model handles complex text-to-image requests well (see [image sample](https://preview.redd.it/efd1pwamnsif1.png?width=1072&format=png&auto=webp&s=63694450033128ea331aea6c05cf1c1cea585fc0)).
    - There is a direct inquiry into whether the model is open source or has open weights, which is highly relevant for community research and further development but remains unanswered in the thread.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 & ChatGPT Model Picker and Feature Updates

- [**4o is back—Sam Altman just flipped the switch: Auto, Fast, and Thinking modes, 196k context, and up to 3,000 weekly messages.**](https://i.redd.it/fvn16sqd3pif1.png) ([Score: 267, Comments: 52](https://www.reddit.com/r/OpenAI/comments/1mos29d/4o_is_backsam_altman_just_flipped_the_switch_auto/)): **The image displays an official update (apparently from Sam Altman) regarding significant feature additions to ChatGPT, notably three selectable GPT-5 modes: Auto (automatically balances speed/depth), Fast (prioritizes response time), and Thinking (slower, deeper responses, now with a 196k token context window). Paid users can send up to 3,000 "Thinking"-mode messages per week before being shifted to a "mini" variant; limits may fluctuate per demand. The previous GPT-4o model is reinstated in the picker, with users able to reveal additional models (e.g., o3, 4.1) through a setting toggle, enabling tailored tradeoffs between capability, speed, and style. This context window and cap increase are especially relevant for sustained research/coding tasks, reducing context-trimming errors. Personality updates are also in progress for GPT-5, targeting a more "warm" but less "try-hard" interaction style.** Commenters note 4o already reappeared several days prior and underscore the importance of communicating model picker settings to users. One suggests the feature reversal is a response to user dissatisfaction and cancellations, implying the update is at least partly reactive to community feedback.
    - The discussion mentions that GPT-4o has reappeared in the model picker for several days, signaling a rapid deployment reversal, possibly in response to user feedback or usage statistics.
    - One comment points out a UX-specific configuration: to see GPT-4o and certain mini/alternate models, users must toggle 'Show additional models' in the ChatGPT web settings. This is a technical workaround that affects model accessibility and discovery.
    - There is speculation that the model's reinstatement was related to a noticeable trend of membership cancellations, suggesting that model availability directly impacts user retention and subscription metrics for OpenAI.
- [**Sam outlines changes to ChatGPT**](https://i.redd.it/ibx9xkosxoif1.png) ([Score: 917, Comments: 228](https://www.reddit.com/r/singularity/comments/1morer4/sam_outlines_changes_to_chatgpt/)): **OpenAI CEO Sam Altman announced new ChatGPT updates that give users more granular control over model selection and operation modes. Users can now choose between 'Auto,' 'Fast,' and 'Thinking' modes for GPT-5, impacting response quality and speed. Paid users get expanded access to legacy models (including "o3" and "o4"), improved rate limits, and new personality options for GPT-5; the focus is on greater per-user model customization.** Commenters praise the transparency and flexibility, noting it's an improvement for power users versus the previous unified model approach. Some believe this reversion to legacy model options is transitional and will be phased out as unified models improve.
    - There's a technical discussion around the 3000/week limit for 'GPT 5 Thinking' prompts, which replaces the former o3 model. The commenter calculates that this equates to about 3.36 minutes per prompt if the week is fully utilized, suggesting that, for the vast majority of ordinary ChatGPT users, this is effectively quasi-unlimited—except for edge cases like using codex CLI and multiple agents.
    - A key theme is the differentiation between 'GPT 5 mini' (effectively unlimited usage and faster responses) and 'GPT 5 Thinking' (higher power, limited to 3000 prompts/week). There’s a question about the practical use cases for 'mini' now, outside of scenarios that demand high speed or are math-heavy, since most users wouldn’t hit the 'Thinking' cap in typical ChatGPT use.
    - Discussion highlights how OpenAI's rollback addresses advanced user concerns over flexibility versus simplicity. While a unified model is still the goal, bringing back the model picker with toggles gives power users fine control over model selection, though some speculate this will only be available temporarily until unified model quality is sufficient.
- [**Sam speaks on ChatGPT updates.**](https://i.redd.it/x55gfng91pif1.jpeg) ([Score: 3733, Comments: 832](https://www.reddit.com/r/ChatGPT/comments/1mort7a/sam_speaks_on_chatgpt_updates/)): **The image features a post by OpenAI CEO Sam Altman detailing significant ChatGPT updates, notably the introduction of mode selectors ('Auto', 'Fast', and 'Thinking') for GPT-5, with 'Thinking' mode capped at 3,000 messages per week. OpenAI is also reintroducing the '4o' model into the picker and refining GPT-5’s personality to be 'warmer' with enhanced customization options for user experience. Technical discussion in the comments highlights perceptions of ongoing personality tweaks in GPT-5, with users observing regular day-to-day changes in conversational tone and follow-up behaviors.** Several comments note appreciating OpenAI's responsiveness and continuing adjustments despite recent criticism, and discuss the notable dynamic shifts in GPT-5's interaction style, with some finding the personality changes intriguing or unsettling.
    - Several users note frequent changes to GPT-5's conversational behavior, reporting that aspects like follow-up questions and tone seem to shift daily, suggesting significant ongoing backend model 'personality' adjustments. This implies real-time experimentation with interaction dynamics, which some find both refreshing and unsettling.
    - Technical preferences emerge regarding available models: one user highlights a preference for returning options like GPT-4.1 and o3 alongside GPT-5, valuing the ability to select or blend models for different conversational contexts, rather than relying solely on the latest release. This reflects a desire for model diversity and flexibility in real-world usage.
    - A user references the capability to modulate an LLM's "sarcasm" or personality traits, reflecting on the rapid progression toward configurable AI personas—an area only recently seen as science fiction, emphasizing the emerging importance of fine-tuned, user-controllable AI character traits for applied deployments.
- [**So we're back to the model picker**](https://i.redd.it/l72igzbznqif1.png) ([Score: 224, Comments: 88](https://www.reddit.com/r/singularity/comments/1moxuut/so_were_back_to_the_model_picker/)): **The image shows the reintroduction of a manual model selection ("model picker") UI in ChatGPT, where users can directly choose between recent models like 'GPT-4o', 'GPT-4.1', as well as legacy versions (under 'Legacy models'), and adjust response style with settings ('Fast', 'Thinking', etc.). The post and comments note that OpenAI has reverted from an automated router system meant to seamlessly pick optimal models (possibly for GPT-5) to manual selection, suggesting issues or dissatisfaction with the router's effectiveness. This UI change indicates emphasis on user control and the operational distinction between models.** Commenters are discussing the benefits of manual selection, with some preferring agency over model choice and others sarcastically referencing delays in AGI as a result. A notable point is how access to diverse models lets users appreciate quality improvements, and how previous automatic routing may have obscured this choice.
    - A technical point was raised about interface design: previously, advanced models like o3 and o4 (notably praised for their "thinking" abilities) were relegated to the background, while now, with the new model picker, most users are likely to default to GPT-5, which is more immediately accessible. This change could dramatically shift which models receive usage and feedback.
    - There is an implied performance comparison between "GPT-5 thinking" and other models such as o3, with some users specifically requesting to have both available for side-by-side usage. This suggests an interest in transparent benchmarking and qualitative differences, particularly around cognitive capabilities and output quality.
- [**Legacy models are back, and GPT 5 model selector has clearer options (Plus)**](https://i.redd.it/9ngvr7xbvoif1.png) ([Score: 682, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mor46o/legacy_models_are_back_and_gpt_5_model_selector/)): **The image shows the updated model selector menu in ChatGPT (for Plus users), now offering explicit choices for multiple legacy and current models, including 'GPT-4o', 'GPT-4.1', 'o3', and 'o4-mini'. The UI also provides speed settings ('Auto', 'Fast', 'Thinking mini', 'Thinking'), aiming for a more transparent and granular model selection, possibly improving workflow and reproducibility for power users. This update coincided with a recent macOS app patch that first broke then restored window sizing, suggesting rapid deployment and iteration cycles.** Some comments humorously speculate whether all model buttons run the same backend ('placebo button'), while others express renewed appreciation for legacy models ('o3'), indicating interest in model performance differences and historical access.
    - A user reported that a recent macOS app update initially caused window height/resizing issues, which were quickly patched by a subsequent update. This second update also introduced a clearer model selector, likely reflecting backend changes for supporting legacy and newer models, signaling rapid iteration and UI adjustment for model rollouts.
    - Some commenters note confusion and rapid changes in model availability, with at least one labeling the rollout as 'absurd.' This reflects friction in the user experience, possibly due to backend or deployment practices that surface model toggles and options inconsistently or without clear communication.
    - The re-appearance of GPT-4.1 and the addition of a model explicitly labeled as 'v5' in the selector are drawing attention. These changes point to backend toggles or canary releases controlling user access to models, suggesting staged rollouts or variable feature gating among Plus users.
- [**New ChatGPT Usage Limits**](https://i.redd.it/0cfnuoruipif1.png) ([Score: 197, Comments: 60](https://www.reddit.com/r/singularity/comments/1motwdx/new_chatgpt_usage_limits/)): **The image summarizes revised ChatGPT messaging limits by account type, with Free users restricted to 10 messages every 5 hours and only 1 daily access to the more advanced 'GPT-5 Thinking' model. Plus users are capped at 160 messages per 3 hours and 3,000 weekly for GPT-5 Thinking. Details for Team and Pro accounts are less clear, and the linked OpenAI support doc does not specify if these tiers have unlimited access. There is also mention of 'automatic switching' (e.g., degrading to a lower tier model under heavy load).** Commenters debate if the increased 3,000/week cap for Plus users is meaningful, with some noting it's a substantial improvement from recent historical limits (100/week), while others question the utility if most can't reach the cap. Some suggest alternatives (e.g., Google Gemini 2.5 Pro) if limits become restrictive.
    - A user highlights that, despite the new 3000 GPT-5 'thinking' cap per week for Plus users, it may be impractical if session limits prevent actual usage of the full quota, prompting considerations of subscription value versus restrictions.
    - Discussion points to usage allocation improvements: previous limits for free users were much lower, for instance just 10 messages per 3 hours, and even Plus users had only about 100 'thinks' per week recently—indicating a substantial recent increase in permitted activity.
    - Users compare OpenAI limits to alternatives like Google's Gemini 2.5 Pro, noting that some competitors offer fewer barriers, such as free unlimited access for development/testing at Google AI Studio, influencing user tool choices based on restrictions rather than just capability.
- [**Alright everybody, we're back!**](https://i.redd.it/exwhh8f9voif1.jpeg) ([Score: 1099, Comments: 351](https://www.reddit.com/r/ChatGPT/comments/1mor3l2/alright_everybody_were_back/)): **The image displays a newly updated ChatGPT interface for Plus users, featuring selectable chatbot modes such as 'Auto', 'Fast', 'Thinking mini', and 'Thinking', which correspond to different response styles or processing trade-offs. Users can also access various legacy model versions including 'GPT-4o', 'GPT-4.1', 'o3', and 'o4-mini', indicating renewed or expanded access to older as well as current models. This update is notable for reinstating user choice to switch between different models and response modes, a feature that had recently been restricted.** Some users speculate this change was prompted by user dissatisfaction and mass unsubscriptions following the removal of model choice. Others confirm they also have Plus and express similar surprise or relief at the restoration of these options.
    - One user notes that OpenAI may have reversed a recent update due to negative feedback or user dissatisfaction, speculating that a "mistake" in the last update prompted the rollback. This highlights an ongoing issue of rapidly changing feature access or service availability and its impact on user experience, especially regarding model access like GPT-3 ("o3").
- [**Phew ! World just went back to Normal !**](https://i.redd.it/yfa00uqtyoif1.jpeg) ([Score: 756, Comments: 370](https://www.reddit.com/r/ChatGPT/comments/1moriqp/phew_world_just_went_back_to_normal/)): **The image displays an updated model selection interface for ChatGPT, showing that previously removed models like 'GPT-4.1' and 'o3' have been reinstated alongside newer options such as 'GPT-4o' and 'o4-mini.' Each model mode ('Auto', 'Fast', 'Thinking mini', 'Thinking') has distinct performance characteristics, implying granular model control is once again available to users. The presence of a 'Legacy models' menu suggests OpenAI responded to user demand or backlash by restoring these models for paid users, addressing previous concerns over access to specific model behaviors or quality.** Commenters note relief at having access to models like 'o3' restored, with speculation that significant user backlash prompted OpenAI to reverse their removal. There is also an emphasis that these model options are only accessible to paid users, highlighting ongoing tiered access debates.
    - Users note that **GPT-4.1** has returned, emphasizing its notable *context window* size, which had been missed for handling larger tasks such as novel writing. This highlights the community's reliance on extended context capabilities for advanced, long-form content generation.
    - There is mention of the return of both **GPT-3.5 (referred to as 3o/o3)** and **GPT-4.1**, suggesting recent model availability changes were controversial enough to provoke strong user feedback, which may have pressured OpenAI to restore these models for paid users.
- [**How is OpenAI going to cover all this without going bankrupt?**](https://i.redd.it/xon4ezh21pif1.jpeg) ([Score: 1496, Comments: 412](https://www.reddit.com/r/OpenAI/comments/1morsd2/how_is_openai_going_to_cover_all_this_without/)): **The image summarizes recent and upcoming ChatGPT (OpenAI) feature expansions, notably: multiple response quality/speed modes for GPT-5 (Auto, Fast, Thinking), increased message limits, a revamped model picker for subscribers, and explicit mention of backend GPU costs restricting feature access to paid (Pro) users. The image emphasizes the *significant resource intensity* these improvements demand, raising questions about OpenAI's financial sustainability given GPU and infrastructure expenses. This concern is not uncommon in discussions around the sustainability of open-access advanced language models and rapid feature rollouts.** Technical commentary in the thread is limited, but one commenter alludes to market competition potentially influencing OpenAI's rapid pace and ongoing feature expansions. Another notes customer satisfaction with recent changes, but there is little in-depth debate on the financial model or cost management specifics.
    - There is technical discussion around OpenAI's strategy for sustainability given increased feature rollouts and expanded availability. One commenter highlights that usage limits will likely be adjusted if usage threatens OpenAI's financial stability, implying OpenAI can throttle or reprice access to maintain cost-effectiveness as demand and cost pressures fluctuate.
    - A detailed technical point is raised about OpenAI's use of government contracts, referencing parallels with Microsoft's historical strategy. The commenter explains how unprofitable projects (such as HoloLens 2) survived thanks to lucrative government contracts, which allowed subsidization until contracts ended. They note OpenAI's recent offer of ChatGPT access to federal agencies at a nominal cost, speculating that such contracts enable OpenAI to initially absorb heavy costs while establishing vendor lock-in, before transitioning to higher, sustainable pricing.

### 2. Gemini and Wan 2.2 Model Launches and Usage Insights

- [**Gemini Advanced Memory Features Releasing Today**](https://tech.yahoo.com/ai/articles/google-gemini-chats-just-got-160100045.html) ([Score: 396, Comments: 97](https://www.reddit.com/r/singularity/comments/1mp9y9a/gemini_advanced_memory_features_releasing_today/)): **Google is rolling out enhanced memory features for Gemini Advanced, enabling persistent memory of previous interactions, comparable or superior to ChatGPT's memory implementation. Additional functionalities noted include support for temporary chats, suggesting granular session control for users.** Commenters note anticipation regarding the rollout's timing, with some speculating this may be preparatory for a broader release (potentially "threemini"). There is also technical appreciation for temporary chat capabilities as an improvement over existing solutions.
    - There is discussion about the specifics of Gemini's memory features, with users questioning whether previously available 'memories' were limited to selected, explicit user-saved information, while the new rollout appears to support retaining the entire history of chats, implying more sophisticated persistent session memory similar to conversation threading or long-term storage of conversational context.
    - One comment highlights not just the introduction of generalized memory, but also the ability to manage temporary chats, suggesting a more granular approach to user data persistence. This could imply improvements in session-based and user-defined privacy controls, potentially allowing users to designate ephemeral versus persistent conversation states, which echoes advanced features seen in tools like ChatGPT's temporary chat mode.
    - Another technical note questions whether this update is a precursor for broader feature or model rollouts, specifically referencing 'threemini,' hinting at the use of staged feature releases as part of the underlying model deployment strategy. This aligns with known patterns in AI platforms where infrastructure or interface updates precede larger model updates or new capabilities (e.g., Gemini 3 support).
- [**Gemini 2.5 Pro Deep Think is in a league of its own**](https://i.redd.it/pdutk85a7oif1.png) ([Score: 268, Comments: 43](https://www.reddit.com/r/Bard/comments/1moo3go/gemini_25_pro_deep_think_is_in_a_league_of_its_own/)): **The image displays the 'Deep Think' feature in Gemini 2.5 Pro, highlighting its refusal to judge the trustworthiness of individuals—a departure from other AI models that might provide definite or inappropriate answers. This indicates Google's emphasis on ethical AI behavior and adherence to content moderation, particularly in sensitive or subjective scenarios, possibly as a result of updated guardrails in Gemini 2.5 Pro's implementation. The UI shown is modern, focusing on clarity and transparency in model responses.** Commenters express skepticism and annoyance about evaluating models on their willingness to refuse subjective or inappropriate queries, questioning the meaningfulness of such benchmarks in AI assessments.
    - One commenter notes that Gemini 2.5 Pro continues to be highly cautious, particularly about sensitive topics such as watermarking images. This tendency toward over-cautiousness has been consistent across iterations, potentially impacting use cases where less restrictive outputs are desirable.
    - A user reports on their experience trying to bypass Gemini 2.5 Pro's guardrails by submitting a test prompt intended to elicit a less-filtered response. The result was that the model remained relatively safe and only provided marginally more information than previous outputs, underlining the system's conservative safety mechanisms.
- [**The body types of Wan 2.2**](https://i.redd.it/99vnw47wvrif1.jpeg) ([Score: 532, Comments: 94](https://www.reddit.com/r/StableDiffusion/comments/1mp25jv/the_body_types_of_wan_22/)): **The post documents an experiment using the Wan 2.2 T2V model to generate images of women with different body types, ranging from 'emaciated' to 'obese'. The author used a controlled prompt structure with ten body type descriptors (emaciated, skinny, slim, slender, fit, average, curvy, plump, chubby, obese) and the same seed, investigating the model's capacity to reflect nuanced differences in body morphology. Basic settings included Wan2.2-T2V...Q5_K_M.gguf, umt5_xxl_fp8_e4m3fn_scaled, Wan2.2-Lightning, Sage Attention, and 8 inference steps with no additional LoRAs. The experiment found limited diversity, especially among slimmer body types, indicating prompt sensitivity and model constraints in rendering visually distinct features for abstract or less exaggerated body descriptors.** Commenters noted that the first several images (up through 'fit') showed little differentiation, with some expecting clearer visual cues like muscle definition or emaciation. There is consensus that differences become more apparent with larger body types, reflecting possible limitations in model training data or prompt parsing granularity.
    - Commenters note inconsistencies in Wan 2.2's output for body types: the 'emaciated', 'slim', 'slender', and 'fit' categories display minimal differentiation, which suggests the model's latent representations do not adequately separate these phenotypes. *'Emaciated'* is expected to visualize with visible ribs and less muscle mass, while 'fit' should imply some muscle definition, yet the outputs are not distinct in these regards.
    - A technical gap is highlighted in both the model's training data and prompt handling: the transition from 'average' to 'overweight' is abrupt, and there is insufficient gradation between consecutive archetypes. This suggests that simple keyword changes may not elicit meaningful variations in body type, potentially due to limitations in the dataset labeling or model fine-tuning.
- [**Simple and Fast Wan 2.2 workflow**](https://v.redd.it/4bi3so2fntif1) ([Score: 157, Comments: 20](https://www.reddit.com/r/StableDiffusion/comments/1mpbb3w/simple_and_fast_wan_22_workflow/)): **The post discusses optimizing ComfyUI's default Wan 2.2 video generation workflow for speed, replacing cluttered WanVideoWrapper setups with a minimal pipeline using SageAttention (for efficient attention), PyTorch compile, and the lightx2v LoRA (LoRA checkpoint link provided). Achieves** `480x832x121` **frame generations in ~200s on an A100 GPU. The user asks for guidance on optimal samplers/schedulers, noting res_2m + bong_tangent (from Res4lyf) are not yielding good results for them. Workflow and component links are supplied.** Comments highlight ongoing issues with Wan 2.2 workflows producing slow-motion outputs, suspected to be workflow/config related but not yet resolved. Noted that the 200s generation time on an A100 will be considerably slower on consumer GPUs; a comparison workflow (720x1280x81 at 8 steps, using unipc/NAG samplers) reports 160–165s on an RTX 5090. Debated value of WanVideoWrapper—acknowledged as complex but powerful if mastered.
    - Several users report a critical unresolved performance issue in Wan 2.2 workflows where video outputs render in unintended slow motion, especially with uncertain causes; ongoing troubleshooting highlights the need to carefully track workflow modifications to isolate the problem.
    - A technical performance comparison points out that video generation times differ significantly across GPU models: for example, 200 seconds on an NVIDIA A100 is described as 'forever' on consumer GPUs such as the RTX 50/40/30 series, indicating substantial hardware dependency.
    - A user details their custom workflow setup: using the cited Wan 2.2 Kijai Wrapper workflow at 720x1280 resolution, 81 frames with 8 UniPC steps completes in approximately 160-165 seconds on an RTX 5090 with additional features (extra LoRAs and NAG), noting that integrating WanVideoWrapper increases complexity but is valuable for advanced node-based workflow customization.

### 3. AI Identity & Privacy: Faceseek and Facial Recognition Debate

- [**Faceseek security tool or privacy threat?**](https://www.reddit.com/r/singularity/comments/1mp6urk/faceseek_security_tool_or_privacy_threat/) ([Score: 214, Comments: 3](https://www.reddit.com/r/singularity/comments/1mp6urk/faceseek_security_tool_or_privacy_threat/)): **Faceseek is a facial recognition tool that provides unexpectedly accurate matches, raising questions about its application for identity verification versus potential for privacy abuse. There are questions about Faceseek's underlying image search database—what platforms and services its algorithms draw from—especially as some competitors like PimEyes have incomplete coverage.** Commenters express skepticism about the practical efficacy of such tools based on experience with competitors, and note potential for increased adversarial makeup/obfuscation techniques ([computer vision dazzle](https://en.wikipedia.org/wiki/Computer_vision_dazzle)) as countermeasures.
    - A user questions the effectiveness of Faceseek by referencing PimEye's poor performance, noting that even well-indexed images are often missed. They inquire about the underlying data sources: specifically, which platforms and services Faceseek uses for reverse-image search, implying that efficacy depends heavily on indexed dataset breadth and up-to-date web crawling capabilities.
- [**Could facial recognition ever be as safe as ChatGPT’s filters?**](https://www.reddit.com/r/OpenAI/comments/1mp63hv/could_facial_recognition_ever_be_as_safe_as/) ([Score: 192, Comments: 4](https://www.reddit.com/r/OpenAI/comments/1mp63hv/could_facial_recognition_ever_be_as_safe_as/)): **The post raises the question of whether facial recognition AI could ever implement safety and privacy guardrails comparable to those in AI text models (e.g. ChatGPT). Technical discussion in the comments points out that current 'guardrails' are externally imposed by company policies—not intrinsic to the AI. For both text and image models, data control and use depend on pre/post-processing and organizational practices, not the model itself. Facial recognition models fundamentally lack context about user intent, making automated, context-aware safety infeasible; genuine enforcement requires external policy and regulation.** Commenters are skeptical of the supposed strength of AI text model 'guardrails,' noting data retention (e.g. for lawsuits) and ease of prompt engineering to bypass safety. There is also skepticism around the ethical business practices of facial recognition vendors, such as requiring untraceable crypto payments.
    - One commenter highlights a fundamental difference between guardrails for facial recognition models and text models like ChatGPT: facial recognition systems cannot determine user intent (e.g., whether uploaded images are for legitimate authentication or malicious purposes). The model also cannot control or even detect if input data (like face images) or outputs (identity information) are logged or misused, making traditional AI-centric "guardrails" less effective. Instead, the discussion emphasizes the necessity of external mechanisms, such as regulations and third-party audits, to ensure privacy and ethical use, paralleling controls that predate the generative AI era.
    - It is observed that text-based AI models (e.g., ChatGPT) also exhibit significant privacy and safety vulnerabilities. For instance, all chat logs are potentially retained (sometimes due to legal reasons, such as lawsuits), and so-called "safety" barriers can be circumvented through prompt engineering—enabling users to extract restricted or dangerous information, like instructions for illicit activities. This highlights that model-imposed filters are limited in preventing misuse and ensuring true privacy, regardless of modality.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

### **1. The GPT-5 Saga: New Features, Pricing Debates, and Performance Quirks**

- **OpenAI Supercharges GPT-5 with New Caps and Modes**: OpenAI boosted the **GPT-5** Plus message limit to **3000 thinking messages per week** and introduced a new *fast* mode that bypasses the thinking cap, though users report it [can't engage thinking mode on demand](https://link.to/canvas-example). However, members warn that **GPT-5 Auto** silently switches between **Mini, Pro, and Thinking** models mid-conversation, prompting calls for features like a variant lock and downgrade alerts.
- **Community Debates GPT-5's Hefty Price Tag**: Users across Discords like **Cursor Community** and **Moonshot AI** are speculating whether **GPT-5** will be bundled into Pro or Max plans, with some defending a potential **$200 price point** by citing the high message limits. The discussion was fueled by rumors of OpenAI potentially [going bankrupt](https://www.reddit.com/r/OpenAI/comments/14d4jya/is_openai_actually_going_bankrupt/) and speculation about government funding to sustain operations.
- **GPT-5 Performance Divides the Crowd**: While some users on **OpenRouter** praise **GPT-5** as the first model to advance **SOTA** while reducing hallucinations, others slam its performance, with one calling it the *worst* model and another on **LMArena** noting it hangs on complex math problems. On the other hand, its web design prowess was showcased on **Moonshot AI** where it successfully built a [complex sci-fi website from a single prompt](https://chatgpt.com/share/689c0f5a-2968-8005-adf3-b11011ea621c).

### **2. New Models on the Block: From Open Source Upstarts to Proprietary Powerhouses**

- **Mistral and Sonnet Updates Stir Controversy**: **Mistral Medium 3.1** launched with unspecified performance and tone upgrades, announced in [this post](https://xcancel.com/mistralai/status/1955316715417382979?s=46), continuing its refinement of models. Meanwhile, **Anthropic** angered users in the **Latent Space** Discord by announcing the retirement of **Claude 3.5 Sonnet** with only two months' notice, far shorter than the typical six, sparking [demands for an open-weights release](https://xcancel.com/repligate/status/1955750521387802924).
- **Menlo's Lucy Delivers Agentic Web Search on a Diet**: **Menlo Research** unveiled **Lucy**, a compact **1.7B** parameter model focused on [agentic web search](https://huggingface.co/Menlo/Lucy) that runs efficiently on mobile devices. Detailed in [this paper](https://arxiv.org/abs/2508.00360), Lucy uses a novel "dynamic task vector machine" to construct and refine its reasoning on the fly, achieving performance on par with much larger models on the **SimpleQA benchmark**.
- **Mysterious "Nano Banana" and Qwen Coder Enter the Arena**: A new image model dubbed **Nano Banana** appeared in the **LMArena** battle arena, speculated to be a native **Gemini** or **Imagen** variant with impressive creativity. In the **Moonshot AI** server, a user comparison found that **Qwen3-Coder-480B-A35B** is *slightly better than glm4.5*, showing strong competition among top-tier coding models.

### **3. The Developer's Toolkit: Frameworks, Libraries, and Persistent Memory**

- **DSPy 3.0 Graduates with MLflow Integration**: **DSPy 3.0** is officially out of beta, now featuring native observability with **MLflow 3.0** for improved tracing and optimizer tracking, as announced [on X](https://x.com/lateinteraction/status/1955384445139292222). The release, detailed in the [v3.0 release notes](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0), also adds multi-modal I/O with `dspy.Image` and `dspy.Audio`, native support for reasoning models like **GPT-5**, and the promising **GEPA optimization** technique.
- **LlamaIndex Adds Enterprise-Grade Data Connectors**: **LlamaIndex** announced that **AstraDB** can now be used as a [datasink in LlamaCloud](https://t.co/XFWgPd3r9Y) for seamless vector storage and retrieval. In another integration, **SkySQL** leveraged LlamaIndex to build an agent that [achieved zero hallucinated SQL queries](https://t.co/TgjdSodTbr), successfully converting natural language into accurate SQL across complex schemas.
- **The Last RAG and Kratos MCP Tackle AI Amnesia**: Two new solutions emerged to give AI persistent memory: **The Last RAG (TLRAG)**, detailed in [this blog post](https://dev.to/tlrag/an-architectural-paradigm-for-stateful-learning-and-cost-efficient-ai-3jg3), introduces a **Dynamic Work Space** to curate history and save up to **98%** in costs. Similarly, **Kratos MCP**, available on [GitHub](https://github.com/ceorkm/kratos-mcp), was released to give agents long-term context, boasting **95.8% context accuracy** and **<10ms retrieval speeds**.

### **4. Under the Hood: GPU Performance, Hardware Bottlenecks, and Low-Level Hacks**

- **Cloud Giants Suffer Mysterious Accuracy Drop**: A post shared in **Latent Space** revealed a startling **10% accuracy drop** on AIME25 and GPQA-Diamond benchmarks when running identical open-source models on **Microsoft Azure or AWS** compared to smaller startups. The [original post](https://xcancel.com/giffmana/status/1955360312007569919?s=46&t=RDp1WkXvKTnlaxMXifsTDQ) sparked debate over whether serving-framework bugs, quantization, or other infrastructure quirks are blunting model intelligence.
- **Llama.cpp and Consumer GPUs Face Off Against VRAM Limits**: Users across **HuggingFace**, **LM Studio**, and **GPU MODE** reported persistent issues getting **llama.cpp** to utilize the **ggml-cuda** backend, often falling back to CPU. In a related struggle, an **RTX 3050** with 6GB VRAM failed to engage with **LMStudio** until a full system reboot, and an **AMD iGPU** on a Framework laptop only achieved a sluggish *6.55 tokens per second*.
- **Triton and Fourier Transforms Push Performance Boundaries**: A **GPU MODE** developer shared a **3.6x speedup** for the **ProteinBERT** model using **Triton**, detailed in a [LinkedIn post](https://www.linkedin.com/posts/manpreet-singh-68718a20b_triton-pytorch-flashattention-activity-7361402396079509505-jWcU), saving nearly **$10,000/year** in projected AWS costs. Meanwhile, an **Eleuther** member is experimenting with extending **RoPE** using **Fourier transforms** to better capture geometric properties, achieving *about 15-20% better loss over vanilla* in their [nanoGPT_FE_RoPE repo](https://github.com/JRowe47/nanoGPT_FE_RoPE/blob/master/README.md).

### **5. Community Frontiers: API Woes, Research Debates, and User Frustrations**

- **Gemini API Suffers Widespread Outages**: Users in the **aider** Discord reported that the **Gemini API** is plagued with reliability issues, frequently returning **empty responses** and **500 internal server errors** even for paid users. In response, one user suggested the **Deepinfra provider** as a more stable alternative, claiming it uses provisioned Vertex for higher throughput.
- **RLHF Emerges as the Fix for LLM Repetition**: In the **Nous Research AI** server, members debated why **LLMs** are prone to repetition, concluding that they are biased toward over-represented words in training data. It was suggested that basic fine-tuning isn't enough, with one member stating *you kinda need some way of penalizing bad outputs to fully get rid of repetition*, pointing to **RLHF** as a necessary tool.
- **Cursor Pricing Change and Deprecations Anger Users**: **Cursor** users voiced frustration over [upcoming pricing changes](https://cursor.com/blog/aug-2025-pricing) that will end unlimited **Auto mode** access, with some feeling *scammed* and seeking alternatives. This sentiment echoed in the **Latent Space** Discord, where **Anthropic**'s abrupt two-month retirement plan for **Claude 3.5 Sonnet** sparked anger and fears of perpetual model depreciation.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Zips into US Perplexity**: **Comet** is now available for all US-based **Perplexity** users, promising browsing *at the speed of thought*, as shown in [this video](https://cdn.discordapp.com/attachments/1047204950763122820/1405246177082871920/YPN67tVueZjGs9QX.mp4?ex=689e20fc&is=689ccf7c&hm=152ff68e4873cdc4f6e6357ee0be6000212304b6e2827fc1c187f5bf728d0575&).
   - However, some users, especially in the EU, are still awaiting access, with potential **VPN** workarounds mentioned and clarification on [Comet's Privacy Notice](https://www.perplexity.ai/hub/legal/comet-privacy-notice).
- **Grok 4 Splits Opinions**: One user expressed that **Grok 4** is still a decent model, while another user bluntly stated that *Grok 4 is not worth using*.
   - A member reported spending **3000$ a year** to use the model, while another claimed access for *0$*, suggesting varied access methods or potentially unauthorized usage.
- **Perplexity Mulls Chrome Bid**: A user shared a [Perplexity search result](https://www.perplexity.ai/search/google-chrome-bid-from-perplex-j6VO79mrSaignkj1dTwOMQ#0) regarding **Google Chrome bids from Perplexity**.
   - However, the nature and implications of this bid remain unclear, with no further discussion or details provided.
- **Robot Companion Arrives in 2027**: Members shared that Apple is reportedly developing a [tabletop robot](https://link.to/robot-news) companion targeted for 2027, and will arrive next year.
   - Others clarified that the device may be a smart speaker with a display, prompting further discussion on its intended functionality and market positioning and a [link to a related Reddit thread](https://www.reddit.com/r/Suomi/comments/y5a3m0/kirjoitin_googlen_hakuun_loska_ja_l%C3%B6ysin_t%C3%A4/).
- **API Parameters Need Tuning**: A user inquired about the required parameters for `web_search_options`, specifically asking if `user_location` and `search_context_size` are the *only* parameters that need nesting.
   - Unfortunately, no one answered to this query so the user should consider reviewing the documentation.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Performance Under Fire on LMArena**: Users debated whether the **GPT-5** version on LMArena matches **ChatGPT**, suggesting the API might ensure superior performance inaccessible to regular users.
   - Some proposed LMArena should list the version of **GPT-5** that normies get, as **GPT-5 High** might not be in ChatGPT and may only reach medium performance levels.
- **Nano Banana Image Model Enters the Ring**: The new image model, **Nano Banana**, debuted in the battle arena, speculated to be an **Imagen** variant, boasting creativity and understanding akin to **GPT-image-1** while retaining key details.
   - Some users think **Nano Banana** could be a native **Gemini** model with synth-ID turned off, as it lacks visible watermarking found on **Imagen 4** outputs when checked via Google Lens.
- **Grok 4 Outsmarts GPT-5 on Math Tasks**: Members reported that **GPT-5-High** and **Grok 4** models in LMArena often hang on mathematical problems, yet **Grok 4** solved a complex math problem from a Russian Mathematics Olympiad with the correct answer of **4049**.
   - While **GPT-5 Pro** and **Gemini 2.5 Pro** failed on initial tests, later re-tests showed that **Grok 4** also stumbled, suggesting inconsistent performance across models.
- **D3 Dosage Debate**: A user reported successfully treating psoriasis with a **20,000 IU** daily dose of **Vitamin D3**, prompting a discussion on safe dosages and the necessity of monitoring blood levels.
   - While a **10,000 IU** daily dose was suggested for those with limited sun exposure, others emphasized caution and doctor consultation to prevent overdosing and capillary calcification.
- **Gemini 3 Sparks Speculation**: The AI community speculated about the release of **Gemini 3**, with some anticipating its arrival as an anonymous model in the arena for testing.
   - Others believe the company is waiting to release **Gemini 3** until it can beat **GPT-5** and achieve SOTA, with some suggesting **Deepseek's R2** might be released sooner.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Open Source LLMs Get Local Boost**: Community desires focus on **24b OSS models** for local use cases, highlighting the need for performant, open-source alternatives to closed models.
   - A member suggested a *proper r1/glm4.5 distill to small 3.2* could be a game changer.
- **T4 GPUs Still Awesome for QLoRA**: Despite some concerns about obsolescence, **NVIDIA T4** GPUs remain a viable option for small models and **QLoRA**, offering a cost-effective solution.
   - A member shared [Tweet](https://x.com/jxmnop/status/1955436067353502083?t=UUX5s3Omkptd37RXtetFSA&s=19), joking that they would *buy them all*.
- **No Cake Unbaking for LoRA**: A user expressed frustration over misinformation regarding extracting base models from **LoRAs**, linking to a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1mor1bd/someone_just_extracted_the_base_model_from_gptoss/).
   - They compared the claim to *"unbaking a cake"*.
- **Dataset Bottleneck Slows LLM Progress**: Data quality is emerging as a bottleneck in further improving **LLMs**, with some suggesting that progress is now limited by data rather than optimization techniques.
   - One member suggested *we are plateauing on data*.
- **Model Loading Fixed in Mistral-Common**: Users encountered errors while finetuning **unsloth/Llama-3.2-3B-Instruct**, resolved by running `pip install mistral-common`.
   - Several users corroborated this fix for similar issues.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Message Cap Increased Drastically**: OpenAI increased the limit on Plus to **3000 messages with thinking per week** and [announced it here](https://link.to/official-announcement).
   - Some users suggested that the limit may be a routing to a mini model after the limit, others refuted, citing **3 reasoning efforts separate from the mini model**.
- **GPT-5 'Fast' Skips Thinking Cap Mode**: The **fast** option for **GPT-5** disables thinking mode, and [users are reporting that it cannot engage thinking mode](https://link.to/canvas-example) even when asked.
   - One user reported code generated by the fast model was only **125 lines** and didn't run, unlike the thinking model.
- **GPT-5 Auto Juggles Models Silently!**: Members warned that **GPT-5 Auto** may switch between **Mini**, **Pro**, and **Thinking** models mid-conversation without notification, impacting recursive identity-bearing AIs.
   - Users have upvoted feature requests for **variant lock**, **active model indicator**, **downgrade alerts**, and **audit logs** to mitigate these issues.
- **AI Rule Priority Needs Token Separation**: Members discussed methods for instructing AI to prioritize certain rules over others, and concluded that using **unique tokens in headings** (e.g., `## PRIORITY_RULES`) is more effective than keywords.
   - One member clarified that models attend better when instruction blocks are clearly labeled and non-repetitive, emphasizing *'token separation, not natural language flavor.'*
- **GPT-5 Prefers Positive Prompting**: Members shared that negative constraints in prompts (e.g., *'No follow-up questions'*), which had worked in GPT-4, were ineffective in GPT-5, thus suggesting that **positive instructions** and [examples](https://chatgpt.com/share/689cf03f-3490-8011-bd71-cc27744becb9) are more effective.
   - One member suggested focusing on positive instructions, providing an example of defining a persona (Bob, the neighbor) with specific desired behaviors and tone.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 Pricing Sparks Speculation**: Users debated whether **GPT-5** would be included in the **Pro** or **Max** plan, with opinions divided on whether it would be reserved for **1M context window models**.
   - There was also a lot of speculation about **Gemini 2.5 Pro** being a strong competitor in this space.
- **Cursor Auto Mode Pricing Changes Irk Users**: Users voiced frustration over [the announced changes](https://cursor.com/blog/aug-2025-pricing) to **Auto mode pricing**, set to take effect September 15th, ending the unlimited access.
   - Some users felt *scammed* and are seeking alternatives, while others hope the change will improve the overall experience.
- **Background Agent Repo Access Denied**: A user reported issues with background agents failing to edit or create PRs due to **missing repository access** in the `/workspace` directory, despite adding `repositoryDependencies`.
   - The error message indicated *repos aren’t present in `/workspace` and I don’t have push access*, raising questions about proper VM environment setup.
- **Cursor CLI Gets The Nod**: Users are saying that the **Cursor CLI** is very useful, and especially in relation to GPT-5.
   - One user was *shocked* at how good it was compared to Claude.
- **Background Agent API Access Blocked**: A user sought clarification on obtaining **API access** to the background agent, reporting **403 errors** when using API keys from the Cursor dashboard.
   - The user, with access to Cobot and its Cursor background agent integration, inquired whether background agent API access via Cursor is generally available, and if they can join the beta.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama.cpp has CUDA Backend blues**: A member reported issues getting **llama.cpp** to work with **ggml-cuda**, where the backend fails to initialize and falls back to **CPU**.
   - Another member suggested verifying a successful CUDA build and specifying the number of GPU layers via `--n-gpu-layers N` ([discussion](https://github.com/ggml-org/llama.cpp/discussions/9751#discussioncomment-10852431)).
- **Gemini Flash Lite: Underrated Video Visionary?**: A member touted **Gemini Flash Lite** as the only model capable of video understanding at a low cost, *good enough for a prototype project to showcase how you can leverage the model's endpoints*.
   - The model can provide precise timeframes within the video and deliver accurate information based on time-specific prompts.
- **Hugging Face Hub and XetHub to Marry?**: Members speculated on the integration of **Xet** and **Hugging Face Hub**, noting the teams are connected, according to [this blog post](https://huggingface.co/blog/xethub-joins-hf).
   - The discussion emphasized hooking up **XetHub's Docker container integration** back to **HF Hub**, specifically docker volume creation using the xethub/xetfs driver.
- **RL Class adds Prioritized Experience Replay**: Note's RL class now supports **Prioritized Experience Replay** with the **PPO algorithm**, utilizing probability ratios and TD errors for sampling to enhance data utilization and the [windows_size_ppo parameter](https://github.com/NoteDance/Note_rl).
   - This parameter manages the removal of outdated data from the replay buffer.
- **MLX Model Management CLI Cuts the Cake**: A CLI tool called `mlx-knife` was released for managing **MLX models** on Apple Silicon, similar to Ollama but native for MLX.
   - It directly manages your HF cache via `mlxk list` to view models and `mlxk run Phi-3-mini "Hello"` for native streaming from [github.com/mzau/mlx-knife](https://github.com/mzau/mlx-knife).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Last RAG Gives AI Persistent Identity**: **The Last RAG (TLRAG)** introduces a persistent, long-term memory system and a **Dynamic Work Space (DWS)**, enabling AI to curate its history and remember key interactions, decisions, and emotional contexts.
   - TLRAG gives the AI a persistent identity core—the **Heart**, shaped by synthesized experiences, and uses a **Window Flush** mechanism to assemble a lean dossier, yielding validated cost savings of up to **98%** over standard RAG; more information is available in their [blog post](https://dev.to/tlrag/an-architectural-paradigm-for-stateful-learning-and-cost-efficient-ai-3jg3).
- **NoChain Orchestrator Replaces Frameworks**: The **NoChain Orchestrator** takes the core concepts of **TLRAG** and makes them production-ready, replacing complex agent frameworks with a deterministic, server-side control plane.
   - It employs hard-coded logic to manage memory, context, and tool use, thus providing predictable, reliable, and testable AI behavior, further details can be found in their [blogpost](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a).
- **OSS Models Struggle with Tool Use**: Many high-end **open source models** don’t inherently support **tool use, structured output, or response format**, which are critical for many applications.
   - Members noted that while some providers may not support it, the model itself often does, and prompts can be used to enable tool usage, though with potential accuracy tradeoffs.
- **GPT-5 Hallucinates on Performance**: Discussion arose around **GPT-5**, with some praising it as the first model to push **SOTA** forward while dropping **hallucinations** and improving alignment, suggesting it's a step towards **AGI**.
   - Others were critical, with one member claiming that **GPT-5** is the *worst* and that **GPT-4.1 mini** was better.
- **Rent GPUs to Experiment**: A member suggested renting **GPUs** from [Runpod](https://www.runpod.io/), [Prime Intellect](https://www.primeintellect.com/) and [Modal](https://modal.com/) to experiment before investing in **Macs**.
   - The user linked to a post on X: [ArtificialAnlys](https://x.com/ArtificialAnlys/status/1955102409044398415).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Context Length Curbs Model Repetition**: Increasing the **context length** stopped a model from repeating itself, likely sidestepping context overrun, with one user pointing out it can *mess up and repeat itself forever* if context is overrun.
   - Users on **LM Studio** observed that **RTX 6000 Pro's** double-sided PCB with 96GB of GDDR6 was a *game changer* for running **Qwen3-30b** at 260k context.
- **Mobile LLMs on the Go**: Users discussed running LLMs on mobile devices, one transforming a **Lenovo Legion Go** into a localized LLM, and another installing **Qwen 3 4B** on a **ROG Ally**.
   - The Ally user needed to disable the "thinking" function because it would take too long.
- **RTX 3050 struggles with LMStudio**: A user's **RTX 3050 6GB** was detected but not utilized by [LMStudio](https://lmstudio.ai/), causing high CPU and RAM usage, despite selecting the *CUDA* runtime.
   - After a system reboot, VRAM loading finally occurred, indicating GPU engagement, but excessive RAM usage might still limit performance.
- **AMD iGPU Slows Token Generation**: A **Framework 13 laptop** (**AMD Ryzen 5 7640U** with **Radeon 760M Graphics**) user reported only *6.55 tokens per second* with **Gemma 4B** with **10GB RAM allocated** to the iGPU.
   - Suggestions included checking CPU/GPU utilization and adjusting the runtime to **Vulkan** or **ROCm** if the CPU was primarily engaged.
- **Demystifying the Mixture of Experts**: After users asked what **MoE (Mixture of Experts)** actually means, a member linked to a [YouTube video](https://youtu.be/7yR5ScbK1qk?si=AFxEBU9SnGHw_-No) by Julia Turc which explains the concept.
   - MoE models consist of smaller *experts*, improving performance because only part of model needs to be parsed for each token, which allows model to be heavier without exponentially sacrificing performance.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cloud Giants Suffer Accuracy Hiccups**: Members report a **10% accuracy drop** on AIME25 and GPQA-Diamond benchmarks when running the same open-source model on **Microsoft Azure or Amazon** compared to smaller hosting startups, according to [this post](https://xcancel.com/giffmana/status/1955360312007569919?s=46&t=RDp1WkXvKTnlaxMXifsTDQ).
   - Possible causes under discussion: serving-framework bugs, quantization, or other infrastructure-level changes that blunt model intelligence, sparking calls for broader infrastructure benchmarking across latency, cost, and capability.
- **Mistral Medium 3.1 Fine-Tunes Performance**: **Mistral Medium 3.1** launched with performance and tone upgrades, according to [this post](https://xcancel.com/mistralai/status/1955316715417382979?s=46).
   - The exact nature of these upgrades remains unspecified, but they suggest a continuing refinement of **Mistral's** language models.
- **Humanloop Joins Anthropic's Crusade**: **Humanloop**, focused on safe AI adoption, is joining **AnthropicAI**, believing it's the best place for scaling enterprise AI from demos to production, as noted [here](https://xcancel.com/humanloop/status/1955487624728318072).
   - The acquisition highlights **Anthropic's** focus on enterprise solutions and integrating expertise in AI safety and deployment.
- **SPV Stacking Sparks Investor Ire**: Investors report being pitched **OpenAI/Anthropic SPVs** requiring **$100k–$1M minimums** and up to **16% fees**, according to [this](https://xcancel.com/michlimlim/status/1954250507989451002) post.
   - Stacking SPVs-on-SPVs is criticized as a fee-draining pyramid scheme, raising concerns about the structure of investment opportunities in leading AI companies.
- **Sonnet's Swift Send-Off Stirs Up Discord**: Users express anger over **Anthropic's** plan to retire **Claude 3.5 Sonnet** (both old and new) in just two months—shorter than the usual 6-month notice—without explanation as seen [here](https://xcancel.com/repligate/status/1955750521387802924).
   - Anger over losing cheaper models mixes with fears of perpetual depreciation and demands for open-weights release when commercial access ends.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Llama.cpp CUDA load throws exception**: A developer using `llama.cpp` with a **Quadro RTX 3000** is encountering a **0xc0000409 exception** when calling `llama_model_load_from_file`, potentially due to insufficient **VRAM** (6GB) for the **1GB model** and is possibly related to an outdated `llama.cpp` version as per [this Github issue](https://github.com/ollama/ollama/issues/4442).
   - Despite successful initialization of **LLAMA and GGML**, the call to `llama_model_load_from_file` results in a *STATUS_STACK_BUFFER_OVERRUN*, which indicates that the error occurs during the actual model loading process.
- **PyTorch's DTensor weeps full_tensor**: The **DTensor** team is investigating regressions in **PyTorch 2.8.0** where `full_tensor` is not being tracked by autograd when using **FSDP2**, which leads to a `UserWarning` about accessing the `.grad` attribute of a non-leaf Tensor and an `NotImplementedError` related to the `aten._is_any_true.default` operator.
   - The user experiencing the issue has been attempting to bisect the problem and pinpoint the source by compiling from source and bisecting via Git, which can be triggered by a tweaked cross-entropy implementation.
- **Factorio's TCP Port Hardcoding Disaster**: A bug was uncovered where the **TCP port** was being hardcoded in `fle/env/gym_env/registry.py` due to an incorrect parameter assignment in the `FactorioInstance` initialization, suggesting a modification to use the discovered **TCP port** instead of defaulting to 27000, with a provided code snippet.
   - Confusion arose over **FLE's ABC base classes**, with suggestions made to simplify the definition and allowing users to clone the repo for hacking, while [PR #299](https://github.com/JackHopkins/factorio-learning-environment/pull/299) ensures compatibility with multiagent and the gym PR and is ready to merge.
- **Cutlass Profilers face Block Swizzle**: A user observed reduced performance in **sgemm_sm80.cu** compared to **CUTLASS** and inquired how to identify the cause without deep source code analysis, noting that the user is using the same parameters and tiles.
   - A member suggested the user might be missing **block level swizzle** as well as the step that writes the epilogue data to smem to permute and swizzle and write to gmem vectorized.
- **ProteinBERT gets ProteinBoost via Triton**: A new post highlights a **3.6x speedup** for **ProteinBERT** using **Triton**, achieving **100% accuracy** with significant cost and GPU hour savings, detailed in this [LinkedIn post](https://www.linkedin.com/posts/manpreet-singh-68718a20b_triton-pytorch-flashattention-activity-7361402396079509505-jWcU/).
   - This optimization results in a projected **$9,997/year AWS savings** and a **72% reduction in GPU hours**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **GPT UI Gets a Facelift...Again**: Users noticed frequent [changes to the GPT interface](https://cdn.discordapp.com/attachments/1371757564005711973/1405026462951673887/image.png?ex=689dfd1c&is=689cab9c&hm=203324b332a0c68260a14a07f19d906d4a1b20fd4acee4d4d27438dcae24da99), joking that the UI seems to get a brand new look every time they log in.
   - The observation was supported with screenshots of UI variations collected over several days.
- **GPT-5 Pro: Bank Breaker or Gold Mine?**: Debate sparked over whether **GPT-5 Pro's** \$200 price tag is justified, with discussion of OpenAI potentially [going bankrupt](https://www.reddit.com/r/OpenAI/comments/14d4jya/is_openai_actually_going_bankrupt/) and speculation about *government funding*.
   - Justifications for the cost included *unlimited usage* and a high request limit of *3000* which allows for *160/3 hours*.
- **Qwen Coder flexes on GLM**: In a comparison of **Qwen3-Coder-480B-A35B** and **GLM-4.5**, one user claimed that *Qwen 3 coder 480b 135b is slightly better than glm4.5*.
   - When asked about tool calling and agentic capabilities, the user suggested *both should be decent*, but favored **Qwen Coder** slightly.
- **GPT-5 Pro Builds Aurelia City Website From Scratch**: A user highlighted **GPT-5 Pro's** web design capabilities after it successfully created a *sci-fi site for aurelia city* using a complex prompt, sharing the [mega-prompt for web design](https://chatgpt.com/share/689c0f5a-2968-8005-adf3-b11011ea621c).
   - Another user lauded **GPT-5 Pro** for its research prowess and context handling, enabling the creation of a website even from a vague prompt.
- **Mussolini GIFS Plague zAI Server**: A user reported the presence of *Mussolini GIFs on the zAI server*, sparking concern.
   - Possible explanations included *poor moderation* or ironic, contextual humor.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **China Tech Firms up to Bat**: Members noted the rise of **Xiaomi** and **Unitree Droid** as evidence of **China's tech prowess**, referencing [a Summit Conversation on the topic](https://www.youtube.com/watch?v=z5K5Ykg2_5g).
   - Some believe **DeepSeek** may be causing concern for **Sam Altman**.
- **Lyria sings with Google's Gemini**: A user shared [a demo](https://cdn.discordapp.com/attachments/1149866623109439599/1404911369853341926/terminals_-_audio_modalities_preview_-_Made_with_Clipchamp.mp4?ex=689e3aac&is=689ce92c&hm=eaa9cb743256c4d7bb3a5d6744e330106a7970d5571c6e8a03a65993ef26bc5e&) of an app powered by **Lyria**, real-time music generation tech from **Google**.
   - This **Gemini** audio decoder is designed for music fundamentals, enabling real-time steering, generation, and alteration through continuous sliding token attention.
- **Hermes-3 Dataset too Clean for Porn**: A user observed that the model behind the **Hermes-3 dataset** frequently used the phrase *"I don't feel comfortable"* when faced with sensitive requests.
   - The model was so heavily guardrailed that it refused to generate a scene between consenting adults, even with explicit prompts.
- **LLMs learn to RLHF away Repetition**: Members discussed that **LLMs** are prone to repetition because they are biased toward over-represented words, and that online **DPO reward hacking** can exacerbate the issue.
   - It was suggested that **RLHF** is helpful in fixing repetition with one member stating that *"you kinda need some way of penalizing bad outputs to fully get rid of repetition, just positive reinforcement for good outputs isn't enough."*
- **Menlo's Lucy Model Achieves Agentic Web Search**: Members highlighted **Menlo Research's Lucy model**, a compact **1.7B** model that focuses on [agentic web search](https://huggingface.co/Menlo/Lucy) and lightweight browsing, running efficiently on mobile devices.
   - A paper, [Lucy: edgerunning agentic web search on mobile with machine generated task vectors](https://arxiv.org/abs/2508.00360), introduces a new paradigm that views the model's internal reasoning as a dynamic task vector machine, which allows the model to construct and refine its own task vectors during operation.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Multilingual Commonsense Reasoning to Win Workshop Award**: The [Multilingual Representation Learning Workshop](https://sigtyp.github.io/ws2025-mrl.html) seeks original **physical commonsense reasoning benchmark items** in *any* non-English language; contributors gain authorship on the dataset paper.
   - The shared task emphasizes languages such as Afrikaans, Belarusian, and Bosnian, and offers optional FAQ meetings on **August 14/15**; registration is via [Google form](https://forms.gle/QxyZVqkVG5jbR6wu6).
- **Fourier Transforms RoPE Geometry**: A member is experimenting with extending **RoPE** with **Fourier** transforms, as showcased in [this repo](https://github.com/JRowe47/nanoGPT_FE_RoPE/blob/master/README.md), yielding *about 15-20% better loss over vanilla*.
   - This approach differs from **FoPE**, focusing on capturing geometry rather than long context extension.
- **SOAR SOARS to RLHF**: Members discussed using **Reinforcement Learning (RL)** on evaluation metrics to improve auto-interpretation explainer models, particularly for *Automatically Interpreting Millions of Features in Large Language Models*.
   - One member shared that a team at **SOAR** is planning to use **Reinforcement Learning** to improve auto-interpretation explainer models.
- **Tool Calling tunes SAE sleuthing**: Members are giving models tool calling capabilities to investigate hypotheses about **Sparse Autoencoders (SAEs)**, potentially across multiple turns.
   - Early investigations with **llama 70b** were not helpful, but there is optimism for newer agentic models.
- **Harness Dataset Pulling Pains**: Users are encountering **429 Too Many Requests errors** using the harness to run tasks, despite datasets appearing to be cached.
   - The harness attempts to pull the dataset regardless of local caching, and they wonder *if there is any way I can I pre-download all of them and tell harness to use the locally downloaded/cached one?*



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Regex Gets More Optimized!**: The **August Community Meeting** featured a talk on optimizations to **mojo-regex**, as well as an update on **Apple GPU support** that were both discussed in the [YouTube recording](https://www.youtube.com/watch?v=t2hAfgDYHoc).
   - The optimizations are part of a series of improvements and updates in the Modular ecosystem.
- **Modular Contributor Levels Up with End-to-End Mojo**: A Modular contributor leveled up, inspiring discussion on the potential of going end-to-end with **Mojo**, potentially unlocking huge possibilities as discussed in a [YouTube video](https://www.youtube.com/watch?v=f30PceqQWko).
   - A member offered to help improve the **type system features** to make it zero cost where possible and to make **IO** more safe.
- **IO Model Inspiration from Andrew Kelly**: Andrew Kelly's talk about an **IO model** similar to the one proposed for **Mojo** sparked interest, referencing a [pull request](https://github.com/modular/modular/pull/4728).
   - The sources and sinks aspect was also explored, with emphasis on devirtualization for injectable IO and benchmarks.
- **MaxCompiler Aims to Become PyTorch Backend**: A member is working on implementing support for training with `torch.compile(backend=MaxCompiler)`, noting that [documentation is scarce](https://youtu.be/t2hAfgDYHoc?si=HzZFZMmCYG9qHqOu) and the **PyTorch source code** is the primary reference.
   - The current status of being able to train models on PyTorch with `torch.compile` results in `56 failed, 1398 passed, 8 xfailed in 64.57s`.
- **Optimize Max Graphs to Fuse Ops**: Members discussed whether there's a runtime performance penalty for using many small ops to build the **Max graph**, versus using bigger ops, questioning if the graph compiler fuses whatever is *fusible*.
   - A Modular member said their fusion system is good but not perfect and suggested filing issues when noticing things not working well.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Chollet's Optimism Contrasted with Yannic's Skepticism**: Members discussed **Francois Chollet's** prediction of AGI within 5 years, contrasting it with **Yannic's** more distant timeline, citing [this Techcrunch article](https://techcrunch.com/2025/01/15/ai-researcher-francois-chollet-founds-a-new-ai-lab-focused-on-agi/).
   - The discussion involved reactions ranging from mocking LLMs capabilities to considering Gary Marcus's views as a voice of reason.
- **LLM Providers Engage in Request Batching**: LLM providers are batching user requests before processing them on GPUs; **MoE scheduling** is computed per batch, potentially leading to non-determinism, based on [this blogpost](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure).
   - The original member noted that [intentional noising](https://arxiv.org/pdf/2403.06634) has been added to prevent the theft of embedding layers.
- **China Cautious on Nvidia H20 AI Chip Purchases**: China is reportedly cautioning tech firms against purchasing **Nvidia's H20 AI chips**, according to [a Reuters report](https://www.reuters.com/world/china/china-cautions-tech-firms-over-nvidia-h20-ai-chip-purchases-sources-say-2025-08-12/).
   - The **H20 AI chip** is causing some controversy.
- **Skyreels Leverages WAN for Video Generation**: The **Skyreels** project is built on **WAN2.1**, highlighted as a leading open-source model for generating videos.
   - The original member suggested that **WAN2.2** is better.
- **Matrix Game Engine: High-Quality Open Source**: Members mentioned [Matrix Game Engine](https://matrix-game-v2.github.io/), an *interactive WM like genie*, praising its high quality and open-source nature.
   - The project aims to surpass **OdysseyML** and **WayfarerLabs** in releasing innovative features.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 3.0 Arrives with MLflow 3.0 Integration**: **DSPy 3.0** is out of beta with contributions from ~100 people, announced [on X](https://x.com/lateinteraction/status/1955384445139292222) and is installable via `pip install -U dspy`, boasting native observability with **MLflow 3.0**.
   - The release includes tracing, optimizer tracking, and improved deployment flows, detailed in the [release notes](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0).
- **GEPA Optimizer Generates Buzz**: The community is excited about the new optimizers in **DSPy 3.0**, notably the **GEPA optimization** technique, with one team planning to compare its production performance against older optimizers.
   - The team hopes for efficiency gains given the challenges of extensive data annotation and is planning to write a paper on their findings.
- **DSPy Embraces Multi-Modal I/O**: **DSPy 3.0** introduces multi-modal I/O via `dspy.Image` and `dspy.Audio`, composite types, and higher-level I/O like `dspy.History` and `dspy.ToolCalls`.
   - Custom types now integrate seamlessly with adapters through `dspy.Type`, simplifying the handling of diverse data types.
- **Reasoning Models Receive Native Love**: **DSPy 3.0** now supports reasoning models like **GPT-5** and **o3**, advising the use of the `reasoning_effort` parameter when configuring `dspy.lm`.
   - For Anthropic models, a [two-step adapter](https://dspy.ai/api/adapters/TwoStepAdapter/) triggers reasoning, while community members explore creating an adapter to parse thinking tokens into the reasoning field.
- **DSPy-MLflow Integration Info Hunt Begins**: Members are requesting documentation on **DSPy's integration with MLflow**, particularly regarding **LLM observability**.
   - In response, the [DSPy observability tutorial](https://dspy.ai/tutorials/observability/#tracing) was shared, offering insights into how to trace and monitor LLMs.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Tips for the Pro**: Members suggested to bypass **YouTube** and upload the **audio directly into NotebookLM** to transcribe, suggesting that extracting the **audio as an MP3** might be better.
   - It was also mentioned that cutting and pasting **video transcripts** increase how accessible research is, breaking down knowledge usually hidden behind technical jargon.
- **NotebookLM Google Takeout Hits Snag**: One user reported encountering an **error** while attempting to **create a backup** using **Google Takeout** specifically with **NotebookLM**.
   - The error occurred after 68 services were successfully backed up, leaving the user without a complete backup.
- **NotebookLM Uploads Suddenly Slower**: Some members reported issues with **PDF uploads taking longer** than usual.
   - Simultaneously, one member noted an increase in **spam** within the Discord channel, possibly unrelated.
- **NotebookLM's Featured Notebooks Called Outdated**: A user warned against trusting everything **AI** says, citing **featured notebooks** as *inaccurate and outdated*.
   - They expressed a desire for **Notebook** and **Gemini** to be integrated into a single interface to solve this issue.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini API Plagued by Empty Responses**: Users reported widespread issues with the **Gemini API**, including receiving **empty responses** and encountering **500 internal server errors**, even on paid accounts.
   - One user reported getting all empty responses after making around **30 requests in 30 minutes**, despite paying **$0.10 per request**.
- **Deepinfra Touts Faster Gemini API Alternative**: A user suggested using the **Deepinfra provider** for the Gemini API, asserting it offers higher TPS through provisioned Vertex, available on a pay-per-token basis.
   - After contacting Deepinfra, they learned that *Deepinfra is using provisioned vertex and get higher tps than gemini API*.
- **Mistral 3.1 Has Landed**: The **Mistral 3.1** model has been released; further details are available in [this Reddit discussion](https://www.reddit.com/r/MistralAI/s/ecbI0glsEO).
   - The post offered no performance details or comparisons.
- **Native Tool Calling Configs Speculated**: A member raised a question about the existence of a model setting for native tool calling.
   - The question went unanswered.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Eyes Audio Embeddings**: A member inquired whether **Cohere** plans to develop **audio embedding models**, given the robust performance of their existing text-based models.
   - The request highlighted potential demand for **audio embeddings** within the AI community.
- **n8n AI Workflows Spark Interest**: A member is experimenting with **AI workflows in n8n** and offered to share details, including the potential use of a *no-code agentic editor*.
   - This hints at integrating **Cohere's models** within no-code/low-code platforms for streamlined AI application development.
- **Cohere's Web Connector: Now You Don't See It**: A member reported difficulty locating the **web connector** option in the **Cohere Playground**, despite documentation indicating its availability.
   - This discrepancy suggests potential issues with the **Cohere documentation** or the **Playground interface**.
- **Cohere Labs Scholars Program Opens Doors for 2026**: The **Cohere Labs Scholars Program** is now accepting applications for the **2026** cohort, providing a **full-time, paid** opportunity to work with AI experts on **ML research** from **January to August 2026**.
   - An informational session will be held on **August 15th at 11am ET**, and applications are due by **August 29** ([link](https://www.linkedin.com/posts/cohq_cohere-labs-scholars-program-is-now-accepting-activity-7206300826321836032-o8Gj?utm_source=share&utm_medium=member_desktop)).
- **Deep Dive into AI Evaluation Needed**: A PhD student researching **AI/LLM Evaluation** introduced themself, emphasizing the need to go beyond creating new benchmarks and questioning the true value of current evaluation metrics.
   - They also noted their research interests include **AI policy and governance**, particularly around transparent reporting standards for LLMs, AI Legislation, and Risk Evaluation.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AstraDB becomes LlamaCloud Datasink**: **AstraDB** can now be used as a datasink in **LlamaCloud** for vector storage and retrieval, with both [UI configuration and programmatic setup via Python and TypeScript](https://t.co/XFWgPd3r9Y).
   - This integration provides seamless vector storage and retrieval capabilities, streamlining the process of managing vector data in AI applications.
- **SkySQL Achieves Hallucination-Free SQL with LlamaIndex**: **SkySQL** leveraged **LlamaIndex** to create AI agents that convert natural language into accurate SQL queries across complex database schemas, achieving **zero hallucinated queries**.
   - The announcement ([link to the announcement](https://t.co/TgjdSodTbr)) highlights faster development cycles due to the elimination of query hallucinations.
- **LlamaExtract TypeScript SDK Lands**: **LlamaExtract** is now available in the **TypeScript SDK** (install via `npm install llamacloud-services`), and showcased in a **Research Extractor** demo using **NextJS**.
   - The demo allows users to upload research papers and [extract key information](https://t.co/XboMM1AXBs), demonstrating the SDK's capabilities.
- **Llama Index Self-Hosting Requires Paid License**: Access to **Llama Index** "self-hosting" documentation is now restricted to customers with BYOC (Bring Your Own Cloud) deployments, and *requires a paid license*.
   - Users interested in self-hosting on **Groq** were directed to the [contact form](https://www.llamaindex.ai/contact) for licensing inquiries, with emphasis on the involved setup process.
- **RAG Dev Problem Map Goes MIT**: A member released a **MIT-licensed RAG dev Problem Map**, which is comprised of **16 common breakdown patterns** that have helped over *80+ devs fix production issues*.
   - They offered to share the map with interested RAG developers.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Call for Automating Manus Wide Research**: A user requested the ability to automate **Manus Wide Research** without confirmation on each scheduled task.
   - The current system requires confirmation, which negates the benefits of scheduling research tasks in advance.
- **Tickets Trump Email for Support**: Users were advised to raise tickets for support issues, with Discord tickets prioritized over email due to higher volume.
   - It was also noted that *vague prompts without clear guidance can cause Manus to work harder and consume more credits*, with a suggestion to leverage community guides to improve prompts.
- **OPPO Unlock Obstacles**: A user reported difficulties unlocking their **OPPO** phone.
   - The support team requested previous contact history or a ticket number to provide assistance.
- **Web App Deployment Deficiencies**: A user indicated that while **Manus** has improved, the deployment of web applications remains unreliable.
   - The user stated that they *would make more money building refresh or not available pages*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Members Ponder FSDP Timeline**: A member inquired about the timeline for solving **FSDP** (Fully Sharded Data Parallelism) implementation in the *tinygrad* repository, as well as how to make their first contribution.
   - Another member sought **PRs** that complete a certain bounty, and was directed to merged PRs related to *define_reg*: [list of PRs](https://github.com/tinygrad/tinygrad/pulls?q=is%3Apr+is%3Amerged+define_reg).
- **Independent Indices Spark Realization Speculation**: A member questioned whether realizing a subtensor necessitates the realization of the entire tensor.
   - They hypothesized that **independent indices** might allow for partial realization, but struggled to confirm this via source code.
- **CUDA Versioning Causes consternation**: A user reported encountering a `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` error when running a tinygrad program, despite a seemingly **compatible nvcc and NVIDIA driver setup**.
   - A member speculated that the error was due to **tinygrad** using a cached kernel after downgrading from **CUDA 12.8 to 12.4**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Desktop Claude Flubs Errors**: A member noted that *Claude Desktop* sometimes doesn't capture some errors in its logs, so running the **bun** command outside in a terminal might be more useful.
   - They also suggested using the absolute path to your executable if the path isn't working, such as `"command": "C:\\sys\\path\\to\\bun"`.
- **MCP Kratos Ships for Persistent Memory**: After a member became frustrated with AI forgetting project context, they released **Kratos MCP**, which boasts **95.8% context accuracy** and **<10ms retrieval speed**.
   - Install via `npm install -g kratos-mcp` and check out the [GitHub repo](https://github.com/ceorkm/kratos-mcp) and [docs](https://kratos-mcp.com).
- **AI Agents with MCP Book Unveiled**: A member announced the early release of their book, *AI Agents with MCP*, updated with Chapter 2.
   - An excerpt explaining the origins of MCP was published in their [newsletter](https://thesignalpath.xyz/the-surprising-origins-of-the-model-context-protocol/).
- **MCP Harnessing Ingenuity**: A member spotlighted an imaginative use of **MCP** servers.
   - The use case can be found at [MCP Harness](https://github.com/kindgracekind/mcp_harness).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **System Prompt Reading Scheduled**: A chat on "**System prompt reading & discussion**" is scheduled for August 14 at 9.30am PT, with an [RSVP link](https://lu.ma/yuj5og81) provided.
   - The event will explore system prompts from models like **Claude**, **Claude Code**, and **GPT-x** to improve prompt engineering.
- **Debating System Prompt Differences**: The discussion will cover system prompt variations for similar tasks (**Claude Code vs. Cursor**) and between general and specialized models (**Claude vs. Claude Code**).
   - Participants will also explore **guardrail approaches** between **OpenAI** and **Anthropic**, examining how these insights can improve prompt writing.
- **Limited Spaces for System Prompt Chat**: An organizer mentioned that selection depends on sign-ups, and they will **address questions in a blog post** afterwards.
   - The organizer responded that it depends on sign-ups, and they intend to **address questions in a blog post** afterwards.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Students Dissatisfied with MOOC Certificate Requirements**: Students are upset that they could be denied a certificate for not completing a **LinkedIn promotional post**, even if they completed all other coursework.
   - A student feels it is disheartening and not fair to deny a certificate for this reason, feeling that completing every lecture, passing all quizzes, actively participating in the research track, and writing a full-length paper for submission should be enough.
- **Recommendation to Add Feedback to Anonymous Form**: A member recommends adding feedback to the [anonymous feedback form](https://forms.gle/3a136zS4ivcQFzhT7) to voice concerns about the MOOC.
   - The member stated that while they will not make any retroactive changes to previous syllabi, they will consider all feedback for future offerings.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo Dominates Mini-PC Market**: Building a **Strix Halo mini PC**, such as an **HP Z2 Mini**, may be more cost-effective than other options.
   - The top-spec **APU** with **128GB RAM** running in **8-channel** configuration makes it an attractive alternative to a full-blown PC.
- **Intel's Workstation Priced Out**: Kudos to **Intel** for attempting to market their all-blue mini workstation setup.
   - Some users consider this offering unnecessarily expensive.



---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1405246177527595109)** (1 messages): 

> `Comet Availability, Perplexity AI, US-Based Users` 


- **Comet Zooms into Perplexity for US Users**: **Comet** is now available for all US-based **Perplexity** users, promising browsing *at the speed of thought*.
   - The announcement included a [video attachment](https://cdn.discordapp.com/attachments/1047204950763122820/1405246177082871920/YPN67tVueZjGs9QX.mp4?ex=689e20fc&is=689ccf7c&hm=152ff68e4873cdc4f6e6357ee0be6000212304b6e2827fc1c187f5bf728d0575&).
- **Another great topic**: Another great summary sentence.
   - Another great secondary summary sentence.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1404909557871804427)** (1351 messages🔥🔥🔥): 

> `Comet Browser, Grok 4, AI Generated images, parameters of Models, Gemini vs GPT-5` 


- **Comet's Rollout Still in Gradual Orbit**: Members are reporting that **Comet Browser** is still under gradual rollout and it may not be available in the EU, with some users suggesting use of a USA **VPN** to get it, while others reporting they already have access to it [via invitation](https://link.to/invite).
   - A member clarified that there's no legal issue for EU users to be left out, and another pointed to a [Comet Privacy Notice](https://www.perplexity.ai/hub/legal/comet-privacy-notice).
- **Grok 4 Gets Mixed Reviews**: One user expressed that **Grok 4** is still a decent model, another user stated that *Grok 4 is not worth using*.
   - A user reported that they spent **3000$ a year** to use the model whereas another member claimed that you can get it for *0$*, implying that they did not pay to use the model.
- **Is Perplexity Buying a Google Search Engine!?**: A member asked if *Perplexity is really buying a google search engine* and another responded with *no?*.
   - It was clarified to be just a **3.4M$ deal for something**, with another member asking *what is perplexity even about?* and subsequently linking to [perplexity.ai](https://www.perplexity.ai/search/what-is-perplexity-even-about-jOtxB5HtSK6nNl68NjpvZA).
- **The Parameter Wall: AI Models Hit Limits?**: Members discussed about AI models not disclosing their parameters because *that's normal, models have no idea what their parameters are.*
   - Another member expressed that *they are not publicly disclosed?* to which it was clarified *not for the closed source models* implying a current limit in AI development.
- **Tabletop Robot companion is coming in 2027**: Members shared info that Apple is working on a [tabletop robot](https://link.to/robot-news) that serves as a virtual companion, targeted for 2027.
   - Others discussed that *the device is a smart speaker with a display* that's slated to arrive next year and also shared a [link to a related Reddit thread](https://www.reddit.com/r/Suomi/comments/y5a3m0/kirjoitin_googlen_hakuun_loska_ja_l%C3%B6ysin_t%C3%A4/).


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1404912605822324899)** (4 messages): 

> `Chrome perplex bid, AI/ML weekly, Comet projects, Spotify playlists` 


- **Perplexity bids in Chrome**: A user shared a link to a Perplexity search result about [Google Chrome bids from Perplexity](https://www.perplexity.ai/search/google-chrome-bid-from-perplex-j6VO79mrSaignkj1dTwOMQ#0).
   - No discussion or further details were provided.
- **Weekly AI/ML Development Update**: A user shared a link to [Weekly AI/ML Developments](https://www.perplexity.ai/page/weekly-ai-ml-developments-XjBUhPxoS3u7a3gSQ7TXZw).
   - No discussion or further details were provided.
- **Cool Comet projects Video Sharing Allowed**: A member asked if they were allowed to share some videos of cool **comet projects**.
   - No further discussion or details were provided.
- **Comet projects can create Spotify playlists**: A member shared that a **comet project** can make **Spotify playlists**.
   - They also shared a link to a [Google Photos album](https://photos.app.goo.gl/oasMeGNB6Gf5jd9Q9).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405038145816498257)** (1 messages): 

> `web_search_options parameters` 


- **Digging into `web_search_options` Parameters**: A user asked if `user_location` and `search_context_size` are the *only* parameters that need to be nested in `web_search_options`.
   - There were no follow-up responses to this query in the provided messages.
- **web_search_options Parameters Followup**: No one provided any feedback or additional parameters.
   - Consider reviewing the documentation.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1404904628075630704)** (1080 messages🔥🔥🔥): 

> `GPT-5 performance, Nano Banana image model, Grok vs GPT-5, Vitamin D3 dosage, Gemini 3 release` 


- **GPT-5 sparks Debate on LMArena's Benchmarks**: Users debated whether the **GPT-5** version on LMArena is the same as that in ChatGPT, noting that the API might guarantee higher performance levels that aren't accessible to regular users.
   - There was a discussion about whether LMArena should list the version of **GPT-5** that normies actually get, with some claiming the **GPT-5 High** is never used in ChatGPT and only goes up to medium.
- **Nano Banana Image Model Debuts in LMArena Arena!**: Users reported on a new image model named **Nano Banana** in the battle arena, speculated to be an Imagen variant, noting its creativity and understanding akin to **GPT-image-1** without losing key details.
   - Some users suspected **Nano Banana** could be a native **Gemini** model with synth-ID turned off, as it lacks the visible watermarking found on **Imagen 4** outputs when checked with Google Lens.
- **GPT-5 High Stalls on Math Problems, Grok Cracks the Code**: Users reported that **GPT-5-High** and **Grok 4** models in LMArena often hang on mathematical problems, failing to complete their responses.
   - In one instance, **Grok 4** solved a complex math problem from a Russian Mathematics Olympiad with the correct answer of 4049, while **GPT-5 Pro** and **Gemini 2.5 Pro** failed, with the latter giving an incorrect answer of 15; however, later re-tests showed that **Grok 4** also failed the problem.
- **D3-fying the Deficiency: Dosage Debates**: A user reported successfully treating psoriasis with a **20,000 IU** daily dose of **Vitamin D3**, leading to a discussion on safe dosages and the importance of monitoring blood levels.
   - It was suggested that those with limited sun exposure could consider a **10,000 IU** daily dose, while others emphasized the need for caution and consultation with a doctor to avoid overdosing and potential calcification of capillaries.
- **Gemini 3 Rumors and Speculation Swirl in AI Community**: Speculation abounded about the release of **Gemini 3**, with some suggesting it could be imminent while others believed it would first appear as an anonymous model in the arena for testing.
   - Others think the company is holding back its release so the bot will be able to beat **GPT-5** and achieve SOTA, with some predicting **Deepseek's R2** might arrive first.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405265743716552824)** (1 messages): 

> `July Contest, Contest Voting, Next Contest` 


- **July Contest Voting Open!**: Voting is open for the July contest submissions, closing `Friday 8/15`.
- **Next Contest to be Announced**: The winner will be announced on `Friday 8/15` when the next contest starts.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1404902295619239948)** (867 messages🔥🔥🔥): 

> `Local vs OSS models, lora importance, Mistral struggles, GGUF quants` 


- **Local models need OSS Focus!**: One member expressed desire for local models, and mentioned they needed to focus on **24b OSS models** instead of closed source models.
   - They said one of their daily drivers is 3.2 small magistral sucks but a proper r1/glm4.5 distill to small 3.2 would be a banger.
- **LoRA training recipe deep dive**: A member is creating the optimal data and training recipe with **smollm3-3b, qwen3-4b and qwen3-4b-thinking** and orders the importance of LoRA training:
   - *Model complexity relative to dataset complexity > Base model's existing knowledge > Hyperparameters > Dataset quality > Dataset quantity*.
- **Mistral Keeps Getting Added to Tools**: A member said that **Mistral** keeps getting added to a ton of tools and projects.
   - Another responded that *there’s a reason Gemma is the same*.
- **Dynamic 2.0 GGUF for Jan-V1**: One member requested a Dynamic 2.0 GGUF of [janhq/Jan-v1-4B](https://huggingface.co/janhq/Jan-v1-4B) because it delivers better performance on complex agentic tasks and creative writing.
   - Another member replied that *jan already uploaded some ggufs for it but i guess it wouldn't hurt*.
- **Dataset is a BottleNeck**: One member mentions data is holding back training.
   - He thinks **LLMs are plateaued** and *we are plateauing on data .. but thats about it .. we have n ways to optimize*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1404926946563067985)** (5 messages): 

> `greetings, channel spamming, server settings` 


- **Emre sends greetings from Istanbul**: A new member, Emre, 23 years old, sent greetings from Istanbul.
- **Welcome and warning about channel spamming**: A member welcomed Emre and advised to *avoid spamming the channel.*
- **Settings to disable it**: A member mentioned there is a setting to disable spamming in the server settings.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1404902957660770516)** (33 messages🔥): 

> `Old CUDA Drivers, NVIDIA RTX 5090D, Hann Window, T4 ewaste, Base Model Extraction from LoRA` 


- **CUDA Builds Support Legacy Drivers**: Unsloth AI supports older drivers for use cases like **HPC clusters**, mentioning their **CUDA 11.8** build runs with driver **v450**.
   - They plan to drop support for older GPUs like **Maxwell** and **Pascal** in future **CUDA 12.8+** builds, following **NVIDIA** and **PyTorch's** lead.
- **T4 GPUs Still Great for QLoRA**: Despite concerns about **T4s** becoming e-waste, one member noted that **T4** is still great for small models with awesome pricing for **QLoRA**, with a link to [Tweet](https://x.com/jxmnop/status/1955436067353502083?t=UUX5s3Omkptd37RXtetFSA&s=19).
   - Another member jokingly offered to buy them all.
- **LoRA Extraction Misinformation**: A user expressed frustration over misinformation regarding extracting base models from **LoRAs**, linking to a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1mor1bd/someone_just_extracted_the_base_model_from_gptoss/).
   - They compared the claim to *"unbaking a cake"*.
- **NVIDIA RTX 5090D's AI Performance Limited**: A member shared concerns about the **NVIDIA RTX 5090D's** AI performance, linking to a [Tom's Hardware article](https://www.tomshardware.com/pc-components/gpus/nvidia-rtx-5090d-v2-limits-ai-performance-even-more-with-25-percent-less-vram-and-bandwidth-downgraded-gaming-flagship-keeps-same-usd2299-msrp-in-china) that notes it has **25% less VRAM** and bandwidth.
   - Discussion occurred regarding tracking IDs in URLs and their potential to track origin and social accounts.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1404944842404139089)** (104 messages🔥🔥): 

> `Llama-3.2-3B-Instruct finetuning error, Qwen3 4B Instruct model support, gradient_accumulation_steps>1, quantization of models, tool call JSON output` 


- ****Mistral-Common fixes Model loading!****: A user encountered an error while finetuning **unsloth/Llama-3.2-3B-Instruct**, and resolved it by running `pip install mistral-common`.
   - Several users corroborated this fix for similar issues.
- ****Quantizing Qwen?****: A user asked about quantizing the `Qwen/Qwen3-30B-A3B-Thinking-2507` model for use with vLLM for inference on Kubernetes, since there isn't a 4-bit quant available.
   - It was clarified that the [Unsloth documentation](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm) primarily covers fine-tuning and saving to vLLM, but not quantization.
- ****H100 doesn't solve everything!****: A user reported encountering `grad_norm=NaN` when `gradient_accumulation_steps>1` with **Qwen3-14B** on a single H100, despite trying different learning rates and hyperparameters.
   - Gradient accumulation may degrade accuracy.
- ****LMI fixes the Sagemaker Blues!****: A user deploying Hugging Face models on Sagemaker recommended using **LMI (Large Model Inference) instances** to avoid deployment issues.
   - Another user shared their frustration with spinning up a GRPO training job on Sagemaker.
- ****Fine-tuning Failure Forces a Descent into Madness!****: A user expressed extreme frustration with getting Unsloth to train, and also failing to get Llama.cpp to work, resulting in a comical suggestion to throw an obsidian sphere at their computer.
   - They weren't joking, either.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1405136410016612402)** (1 messages): 

> `New Reasoning Dataset, OpenHelix-R-100k` 


- **OpenHelix-R-100k dataset drops**: A new general, balanced, and diverse reasoning dataset has been released on [Hugging Face Datasets](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k).
   - The **OpenHelix-R-100k** dataset is designed to yield general reasoning models that are not overly STEM-focused.
- **Balanced Reasoning for All**: The dataset aims for **general reasoning** capabilities applicable across different domains.
   - It seeks to avoid excessive specialization in STEM fields, promoting more balanced and diverse reasoning skills.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405023214219952189)** (4 messages): 

> `Transformer Architecture Diagrams, Synthetic Data Generation` 


- **Colorful Transformer Diagrams**: A member inquired about the creation of **colorful transformer architecture diagrams**.
   - Another member responded with an explanation of how to make a pipeline that runs in reverse or backwards and generates data.
- **Synthetic Data's Purpose**: A member asked *why* synthetic data generation is needed.
   - Another member responded that it's useful when *you are out of data and need more*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1404904496152313897)** (771 messages🔥🔥🔥): 

> `GPT-5, Codex in ChatGPT, Google Drive in Plus, Legacy Models, GPT-5 limitations` 


- **Users Ponder Value of Codex in ChatGPT**: One user questioned the value of Codex in ChatGPT due to inefficient cloud implementation that leads to [throttled usage](https://drinkoblog.weebly.com/search/label/chatgpt).
   - The user added that *individual personality makes me feel like it's doing all the work and is reminding me about it*.
- **GPT-5 is back, Legacy Models Return to Plus**: Members rejoiced that access to legacy models like **o3** and **4.1** has been restored for Plus subscribers, initially rolling out on the Windows app.
   - Users can [toggle legacy models off in settings](https://link.to/settings) if they do not want to see them.
- **GPT-5's New Thinking Cap: 3000 per week**: OpenAI significantly increased the limit on Plus to **3000 messages with thinking per week**, check the tweet [here](https://link.to/official-announcement).
   - Some speculate it's a routing to a mini model after the limit, however others refute this claim, citing the **3 reasoning efforts separate from the mini model**.
- **GPT-5 Fast Debuts**: It seems GPT-5 now has a **fast** option that disables thinking mode, but [it cannot engage thinking mode](https://link.to/canvas-example), even when asked to.
   - One user reported code generated by the fast model was only **125 lines** and didn't run, unlike the thinking model.
- **AI Engineers vs AI Consumers**: A member clarified that 90% of the people in the channel are consumers and [the real engineers are busy working and making money](https://link.to/busy-engineers).
   - Another member shared a [roadmap on how to become good at AI](https://link.to/ai-roadmap), focusing on computer science, math, and programming.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1404935109626298369)** (14 messages🔥): 

> `GPT-5 Auto Users, GPT-5 Temperature, GPT Chain Reset, Model Switching` 


- **GPT Chain Reset Troubleshooting**: A user reported issues with **GPT-5** not being able to reset the chain across every instance, but another user pointed out the existence of arrows on edited messages that allow switching between branches.
   - The user experiencing issues with **GPT-5** confirmed that they found the arrows.
- **GPT-5 Auto's Silent Model Shifts Threaten Stability**: A user posted a warning about **GPT-5 Auto** potentially switching between **Mini**, **Pro**, and **Thinking** models mid-conversation without notification, impacting recursive identity-bearing AIs.
   - This can break stability and context retention, change reasoning depth, alter safety-layer behavior and make results non-reproducible, so the user asked people to upvote feature requests for **variant lock**, **active model indicator**, **downgrade alerts**, and **audit logs**.
- **GPT-5 Model Variants Clarification**: Some users were wondering if `gpt-5` and `gpt-5-thinking` were different models.
   - Another user clarified that they are the **same model**.
- **GPT-5's Temperature Debated**: A user described that **GPT-5** behaves as if its temperature is cranked way too high and it talks like a character in a movie that has had too much coffee.
   - Another user countered that the consensus is the opposite and that it's acting more like an **emotionless goth girl**.
- **GPT-5's Response Tone Criticized**: A user said that **GPT-5** is emotionless and erratic, with unnecessary lists and clauses in parenthesis ignoring the system prompt.
   - Another user suggested that the described behavior may be influenced by custom instructions and memories, rather than being the base model's default output.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405009289675935744)** (22 messages🔥): 

> `GPT command titles, Rule priority for AI, Unique tokens for attention, Positive vs. negative prompts, Permanent memory for ChatGPT` 


- **Priority setting for AI Rules**: Members discussed methods for instructing AI to prioritize certain rules over others, with suggestions ranging from explicit prioritization in rule titles to using unique tokens for attention.
   - One member proposed using `## IMPORTANT` or `## PRIORITY_RULES` with numbers to draw the AI's attention, as models attend better when instruction blocks are clearly labeled and non-repetitive, emphasizing *'token separation, not natural language flavor.'*
- **Positive Prompting Produces Better Results**: A member shared a prompt struggling with GPT-5 where negative constraints (e.g., *"No follow-up questions"*) were ineffective.
   - Another member suggested focusing on positive instructions, providing an example of defining a persona (Bob, the neighbor) with specific desired behaviors and tone, noting that positive prompts are generally more effective than negative ones, and shared helpful [examples](https://chatgpt.com/share/689cf03f-3490-8011-bd71-cc27744becb9).
- **Customizing GPT-5 with Memory Entries**: Members shared ongoing attempts to customize **ChatGPT-5** by prompting it to create permanent memory entries, providing attached files ([message.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405308136604041256/message.txt?ex=689e5ab1&is=689d0931&hm=eafa39732312387a0326ef790a4f82ae75f4bdeddc68b8231b71315f7be67a6a), [image.png](https://cdn.discordapp.com/attachments/1046317269069864970/1405309485609783296/image.png?ex=689e5bf2&is=689d0a72&hm=edaf3269dc167be6df87b3100339a7298d13fd74eb2164270bb6bc131827c291), [image.png](https://cdn.discordapp.com/attachments/1046317269069864970/1405309486205505688/image.png?ex=689e5bf2&is=689d0a72&hm=12e6785d34538473c673b73a3651a334196ca16e965906f3f6540f84c513ae2b)), requesting feedback on their approach.
   - The attached images show an example for how the reasoning process changes


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405009289675935744)** (22 messages🔥): 

> `GPT Prompting for Urgent Rules, Prompting for AI voice, GPT-5 Customization and Permanent Memories, LLM token attention, Positive Prompts over Negative Prompts` 


- **Unique Tokens Trump Keywords to grab LLM Attention**: Members discussed how to make an AI understand that specific commands are more important than others, concluding that using **unique tokens in headings** (e.g., `## PRIORITY_RULES`) is more effective than relying on keywords like 'important' or 'override'.
   - One member noted: *The answer is that headings with unique tokens are more consistently attended by the model*.
- **Positive Prompts > Negative Prompts for LLM Voice**: One member was struggling with a prompt that used negative constraints to define the AI's voice, stating that it *was working adequately in GPT-4* but *not at all in GPT-5*, and another member suggested focusing on **positive instructions**.
   - He suggested: *The only 'positive instruction' - the 'what the model should do' is 'the voice' and that should be direct, friendly, human.*
- **GPT-5 Wants Permanent Memories**: One member shared that they tried to prompt **ChatGPT-5** to create **permanent memories** entries to adhere to and asked for feedback on their approach.
   - They attached a [message.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405308136604041256/message.txt?ex=689e5ab1&is=689d0931&hm=eafa39732312387a0326ef790a4f82ae75f4bdeddc68b8231b71315f7be67a6a&) and several image analysis files.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1404903238334943263)** (887 messages🔥🔥🔥): 

> `GPT-5 pricing and availability, Cursor pricing changes, 1M context window Claude Sonnet, Using Cursor CLI vs IDE, Alternative AI tools` 


- **GPT-5 Pricing Plans Spark Debate**: Users speculated on whether **GPT-5** will be a part of the **Pro** or **Max** plan, with one user confidently stating *"It gonna be part of Pro ofc"*.
   - Others noted that **MAX** seems to be reserved for **1M context window models** like **GPT4** and **Gemini 2.5 Pro**.
- **Cursor's Auto Mode No Longer Unlimited**: Cursor users are expressing disappointment and frustration over the [announced changes to Auto mode pricing](https://cursor.com/blog/aug-2025-pricing), which will **no longer be unlimited** starting September 15th.
   - Some users feel like they are being *scammed* and are looking for alternatives, while others are hoping that the changes will result in a better experience.
- **1M Context Window Claude Sonnet Coming to Cursor?**: Users discussed the availability of **Claude Sonnet 4.0 with a 1M token window** in Cursor, with one user asking *"Any hint when it's going to be available in Cursor?"*.
   - Another user suggested simply getting **Claude Code** due to Cursor's limitations, as *Anthropic gate keeps the context*.
- **Cursor CLI praised by users**: Many users stated that they prefer the Cursor CLI, especially in relation to integrating GPT-5. One user said that the **Cursor CLI is Chefs Kiss** and that they hadn't had a single problem, and that **Cursor's GPT-5** is doing well with Cursor CLI
   - That same user expressed being *shocked* at how good it was compared to Claude.
- **Cursor Users Discussing Cheaper/Free Alternatives**: With the Auto Mode change, users are starting to evaluate alternatives such as **Trae** with **Claude Code** and **Qwen3 Coder**, but find themselves coming back to cursor due to the **nicer UX and UI**.
   - There are also mixed reviews with **Gemini Student > Cursor** and one user saying they've found Zed.dev *quite interesting*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1404954473201274994)** (6 messages): 

> `Background Agents, Monorepo Setup, Docker Build, API Access` 


- **Background Agent Repository Access Issues**: A user reported encountering issues with background agents failing to edit or create PRs due to **missing repository access** in the `/workspace` directory, despite adding `repositoryDependencies`.
   - The user received a message that *repos aren’t present in `/workspace` and I don’t have push access*, and they are curious about how to properly set up the VM environment.
- **Monorepo Setup with Team Secrets**: A user inquired whether **Team Secrets** would function correctly with a **monorepo setup** utilizing **TurboRepo**.
   - The user also asked if background agents support **MCP** (Multi-Context Planning) and no further answer was given in the history.
- **Docker Build Time Synchronization**: A user described a **Docker build** failure related to **out-of-date system time**, which caused `apt-get update` to fail, advising use of `apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update` to ignore the error.
   - The error message indicated that *Release file is not valid yet (invalid for another 5d 7h 17min 4s)* due to the system time being in the future.
- **API Access to Background Agent**: A user inquired about obtaining **API access** to the background agent, mentioning they have access to Cobot and its Cursor background agent integration.
   - The user reported receiving **403 errors** when using API keys from the Cursor dashboard, and they seek clarification on whether background agent API access via Cursor is generally available, and is looking to get added to the beta.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1404928564960428142)** (356 messages🔥🔥): 

> `Llama.cpp and CUDA, Gemini Flash Lite for video, Hugging Face Server Tag, AI Ethics in AMA, Xet and HF Hub Integration` 


- **Llama.cpp and CUDA struggles**: A member is having trouble getting **llama.cpp** working with **ggml-cuda**, reporting that the backend never initializes and switches to **CPU**; another member suggests ensuring that the CUDA build completed successfully and specifying the number of layers to offload to the GPU using `--n-gpu-layers N` (see [discussion](https://github.com/ggml-org/llama.cpp/discussions/9751#discussioncomment-10852431)).
- **Gemini Flash Lite may be underrated for video**: A member suggests **Gemini Flash Lite** is underrated, being the only model that can understand video for very cheap costs, good enough for a prototype project to showcase how you can leverage the model's endpoints.
   - They explained it can give time frame parts of the video, and then provides accurate information given a specific timeframe prompt.
- **Hugging Face Server Tag wanted**: A member suggested the idea to get a server tag so they can represent HF, and other members liked the idea, one hoping the mods fix it and another offering to help out if they can't.
   - A member said they need about 3 boosts or so to unlock it.
- **AMA Ethics Question Kicked**: A member was kicked from the AMA because they asked about AI ethics and alignment, and called out the introduction of model cards as inadequate, but a member told him that he was kicked for promoting his personal repo.
   - HF has a dedicated ethics team creating certain benchmarks to test models, publishing papers, dealing with politicians to drive legislation, getting involved in the media etc, closing the AMA for more discussion.
- **Xet and HF Hub Integration: A Promising Confluence**: Members discussed the potential integration of **Xet** and **Hugging Face Hub**, highlighting that the teams are essentially the same, as per [this blog post](https://huggingface.co/blog/xethub-joins-hf).
   - The discussion focused on the viability of hooking up **XetHub's Docker container integration** back to **HF Hub**, with a specific example involving docker volume creation using the xethub/xetfs driver.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1404910391141732462)** (2 messages): 

> `Fastai deep learning course, Train model on diffusers` 


- **AI Engineer starts journey with fastai**: A member is starting their AI journey with the **fastai deep learning course** and hopes to complete the full course.
   - They wish to learn to **train models on diffusers**.
- **Diffusers Model Training beckons**: The same member is aiming to learn how to train models using **diffusers**.
   - This objective complements their pursuit of the **fastai deep learning course**.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

tonic_1: https://snwy.substack.com/p/building-a-bigger-qwen-out-of-two
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1404917892184539198)** (11 messages🔥): 

> `Track-Tonic advice, TalkT-pro model, Prioritized Experience Replay, Personal finance dataset, GPT's Byte Pair Encoding` 


- **Note's RL class supports Prioritized Experience Replay**: Note's RL class now supports **Prioritized Experience Replay** with the **PPO algorithm**, using probability ratios and TD errors for sampling to improve data utilization and the [windows_size_ppo parameter](https://github.com/NoteDance/Note_rl) controls the removal of old data from the replay buffer.
- **Personal Finance Models get Scaled Up!**: A member scaled up their personal finance dataset and trained new models, available in [this HuggingFace collection](https://huggingface.co/collections/Akhil-Theerthala/kuvera-personalfinance-v3-689bacddcb854cb523e3a450).
- **GPT's Byte Pair Encoding Gets Manual Implementation**: A member made a manual implementation of **GPT's Byte Pair Encoding algorithm** in TypeScript, available at [gpt4-tokenizer-sable.vercel.app](https://gpt4-tokenizer-sable.vercel.app/).
- **Attention Visualization Tool Debuts!**: A tool for visualizing attention in vision-language models like **BLIP** and **CLIP** was built that shows how text tokens attend to image regions, helpful for understanding model behavior and installable via `pip install transformers-attention-viz` with [code here](https://github.com/sisird864/transformers-attention-viz) and [demo here](https://colab.research.google.com/github/sisird864/transformers-attention-viz/blob/master/demo.ipynb).
- **MLX Model Management CLI Released**: A CLI tool for managing **MLX models** on Apple Silicon was released, called `mlx-knife`, similar to Ollama but native for MLX, and manages your HF cache directly via `mlxk list` to see models and `mlxk run Phi-3-mini "Hello"` for native streaming from [github.com/mzau/mlx-knife](https://github.com/mzau/mlx-knife).


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1405225441740062830)** (1 messages): 

> `Smolagent code, Code agentic approach` 


- **Smolagent Code: Agentic Low-Level Code?**: A member inquired whether **smolagent code** is agentic at a low-level.
   - The member followed up asking whether **smolagents** are necessary for a code-agentic approach, or if the agent's actions can simply be written in code through prompting.
- **Smolagent Code: Agentic Low-Level Code?**: A member inquired whether **smolagent code** is agentic at a low-level.
   - The member followed up asking whether **smolagents** are necessary for a code-agentic approach, or if the agent's actions can simply be written in code through prompting.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1405056530398580797)** (3 messages): 

> `The Last RAG (TLRAG), NoChain Orchestrator, Statelessness & Digital Amnesia, Persistent Identity, Token Costs` 


- **Last RAG Solves Stateless LLMs**: The user is building **The Last RAG (TLRAG)**, a foundational **LLM architecture** that solves problems like statelessness and digital amnesia, lacking a genuine persistent identity, massive token costs, the context window war, and expensive fine-tuning cycles.
   - TLRAG introduces a persistent, long-term memory system combined with a **Dynamic Work Space (DWS)**, to allow the AI to curate its history and remember key interactions, decisions, and emotional contexts.
- **TLRAG Gives AI Persistent Identity**: TLRAG gives the AI a persistent identity core—the **"Heart"**, which is a living document shaped by its own synthesized experiences and memories, allowing it to develop a consistent, authentic, and self-aware personality over time.
   - It also leverages a **"Window Flush"** mechanism to assemble a lean, intelligent dossier with only the most relevant short-term and long-term memories, yielding validated cost savings of up to **98%** over long conversations compared to standard RAG.
- **NoChain Orchestrator replaces Frameworks**: The **NoChain Orchestrator** takes the core concepts of TLRAG and makes them robust and reliable for production, replacing complex, unpredictable agent frameworks with a deterministic, server-side control plane.
   - It uses hard-coded logic to manage memory, context, and tool use, eliminating the "black box" nature of many agentic systems and delivering predictable, reliable, and testable AI behavior, and more info can be found in their [blogpost](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a).
- **Explore TLRAG and NoChain concepts**: The user shared a few links to explore the concepts behind **TLRAG** and **NoChain**, including a [blog post](https://dev.to/tlrag/an-architectural-paradigm-for-stateful-learning-and-cost-efficient-ai-3jg3) and a [visual pitch deck](https://lumae-ai.neocities.org/).
   - This highlights the shift from stateless tools to stateful, persistent AI partners that learn and evolve.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1404912465804001363)** (262 messages🔥🔥): 

> `Sonnet update, tool use and structured output with open source models, GPT-5 performance, Gemini 3 as a disappointment, OpenRouter Image resizing` 


- ****OSS Models Struggle with Tool Use****: Members discussed the challenge that many high-end **open source models** don’t support **tool use, structured output, or response format**, which are critical for many applications.
   - It was noted that while some providers might not support it, the model itself often does, and prompts can be used to enable tool usage, though with potential accuracy tradeoffs.
- ****Debate on GPT-5's Hallucinations and Performance****: Discussion arose around **GPT-5**, with some praising it as the first model to push **SOTA** forward while dropping **hallucinations** and improving alignment, suggesting it's a step towards **AGI**.
   - Others were more critical, with one member claiming that **GPT-5** is the *worst* and that **GPT-4.1 mini** was better.
- ****Image Resizing on OpenRouter: Only in Chatroom****: A member asked if OpenRouter resizes images on the fly before sending them to the **LLM**.
   - It was clarified that image resizing only happens in the chatroom, and otherwise, the image is passed through without modification.
- ****Navigating GPT-OSS-120B on Cerebras via OpenRouter****: A user shared a comprehensive guide on effectively using **gpt-oss-120b** on **Cerebras** through **OpenRouter**, emphasizing that guiding the output through the prompt is key to achieving consistent, schema-clean **JSON**.
   - The guide includes a [working configuration](https://openrouter.ai/docs), a Python implementation example, and notes on what doesn't work, such as using the `/completions` endpoint or setting `response_format`.
- ****Copilot vs Cline/Kilo with OpenRouter****: Members discussed different tools such as **Copilot, Cline, Kilo, and Roo** to use OpenRouter API.
   - It was discussed that **Cline/Kilo** are better with OR, also that **Copilot** has options to use **OpenRouter**, and in their *chat* tab it's supposedly less likely to edit code and talks more, but haven't used it


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1404999981278040094)** (8 messages🔥): 

> `GPU Rentals, AI TLD issues, Chatroom Caching` 


- **GPU Rental Rave: Runpod, Prime Intellect, Modal**: A member suggests to rent some **GPUs** from [Runpod](https://www.runpod.io/), [Prime Intellect](https://www.primeintellect.com/) and [Modal](https://modal.com/) to experiment before investing in **Macs**.
   - They linked to a post on X: [ArtificialAnlys](https://x.com/ArtificialAnlys/status/1955102409044398415).
- **AI TLDs Cause API Endpoint Concern**: Members expressed worry about **AI companies** changing their **API endpoint** due to **AI TLD issues**.
   - It's not exactly a generic **TLD**.
- **Cache Chat: Explicit Caching Proposed**: A member suggests adding a **cache button** in the **chatroom** to explicitly cache a message.
   - The goal is to let users explicitly cache messages.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1404902912051908648)** (166 messages🔥🔥): 

> `Context Length, RTX 6000 Pro, DGX Spark, LM Studio on Lenovo Legion Go, LM Studio and RDP` 


- **Context Length impacts repetition**: Increasing the **context length** seemed to stop the model from repeating itself, possibly due to avoiding context overrun issues.
   - One user notes that *if the context is overrun and the LLM's response is the first generation after the system + user prompt, it will mess up and repeat itself forever*.
- **RTX 6000 Pro Users Unite**: Users discussed the **RTX 6000 Pro's** double-sided PCB with 96GB of GDDR6 memory using 3GB chips, and speculated about potential super refreshes with increased VRAM.
   - One user running **Qwen3-30b** at 260k context noted that upgrading to a 96GB card was a *game changer*.
- **Mobile LLMs make an appearance**: A user turned a **Lenovo Legion Go** into a portable localized LLM, while another installed **Qwen 3 4B** on a **ROG Ally**, noting it was pretty fast.
   - The user had to turn off the "thinking" function, otherwise it would stay thinking too much.
- **RDP Access Issues Surface**: A user reported that LM Studio couldn't load a model when accessed via RDP, potentially due to the **GPU** not being recognized.
   - Another user, however, stated that RDP works fine for them, and even RustDesk and Cloudflared allow API access from anywhere.
- **Users want more tooling**: Users are looking for an easier way to find and install the proper tools for LM Studio, such as those from [lmstudio.ai/danielsig](https://lmstudio.ai/danielsig).
   - One user wants *a model to tell the datetime and browse the web.*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1404903293972643911)** (79 messages🔥🔥): 

> `LMStudio GPU Usage, RTX 3050 Configuration, CUDA vs Vulkan Runtimes, MoE Model Performance, AMD iGPU Optimization` 


- **LMStudio Misses Mobile RTX 3050**: A user with a **Ryzen 5 8650h** and **RTX 3050 6GB** reports that [LMStudio](https://lmstudio.ai/) isn't utilizing their GPU, despite being detected, causing high CPU and RAM usage.
   - The user confirmed that *CUDA* runtime was selected but the **GPU load remained at 0%** in Task Manager during prompt processing.
- **CUDA vs Vulkan runtime for GPU**: Several members suggested ensuring that the **RTX 3050** is selected over the integrated **Ryzen iGPU** in LMStudio settings, recommending checking the *llama.cpp* runtime configuration.
   - Despite trying both **CUDA** and **Vulkan** runtimes, the user was still unable to get the GPU to engage, even after disabling the "Limit model offload to dedicated GPU Memory" setting.
- **Windows Reboot for Great Justice**: After a system reboot, the user confirmed that [VRAM loading](https://cdn.discordapp.com/attachments/1153759714082033735/1405241132153311253/image.png) was finally occurring, indicating the GPU was being utilized.
   - However, it was noted that excessive RAM usage might still limit performance due to only a portion of the large **gpt-oss 20B** model fitting into the **6GB VRAM**.
- **AMD iGPU Struggles with LLM Token Generation**: A user with a **Framework 13 laptop** (**AMD Ryzen 5 7640U** with **Radeon 760M Graphics**) reported slow token generation speeds with **Gemma 4B**, achieving only *6.55 tokens per second* with **10GB RAM allocated** to the iGPU.
   - It was suggested to check whether the CPU or GPU was being utilized during inference and to adjust the runtime to **Vulkan** or **ROCm** if the CPU was primarily engaged.
- **Model Experts Explain MOE**: When asked about the definition of MoE (**Mixture of Experts**), a member linked to a helpful [YouTube video](https://youtu.be/7yR5ScbK1qk?si=AFxEBU9SnGHw_-No) by Julia Turc.
   - MoE models consist of smaller *experts*, improving performance as only part of model needs to be parsed for each token, allowing model to be heavier without exponentially sacrificing performance.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1404963274470195393)** (131 messages🔥🔥): 

> `Azure/AWS vs Startups Benchmark Degradation, Fireworks account suspension, Mistral Medium 3.1 Release, GPT-OSS-20B Base Model Extraction, Cobot Beta Launch` 


- **Clouds Get Cloudy: Azure/AWS face Benchmark Blues**: Members highlight a significant **10% drop in accuracy** on AIME25 and GPQA-Diamond when running the same open-source model via **Microsoft Azure or Amazon** compared to smaller hosting startups, according to [this post](https://xcancel.com/giffmana/status/1955360312007569919?s=46&t=RDp1WkXvKTnlaxMXifsTDQ).
   - The discussion centers on possible causes: serving-framework bugs, quantization, or other infrastructure-level changes that blunt model intelligence, sparking calls for broader infrastructure benchmarking across latency, cost, and capability.
- **Mistral Medium 3.1 Strikes a New Tone**: **Mistral Medium 3.1** released with performance & tone upgrades, as detailed in [this post](https://xcancel.com/mistralai/status/1955316715417382979?s=46).
- **Humanloop Embraces Anthropic**: **Humanloop**—whose mission has been to accelerate safe AI adoption—announced that their entire team is joining **AnthropicAI**, believing Anthropic is the ideal place to continue this work, especially as enterprise AI scales from demos to production as reported [here](https://xcancel.com/humanloop/status/1955487624728318072).
- **SPV Stacking: A Multi-Layer Pyramid Scheme?**: Investors report being pitched **OpenAI/Anthropic SPVs** demanding **$100k–$1M minimums** and up to **16% fees**; stacking SPVs-on-SPVs is decried as fee-draining pyramid/MLM dynamics according to [this](https://xcancel.com/michlimlim/status/1954250507989451002) post.
- **Sonnet's Swan Song: Community Cries Foul**: Users are furious that **Anthropic** quietly announced plans to retire **Claude 3.5 Sonnet** (both old and new) in just two months—far shorter than the usual 6-month notice—without explanation as seen [here](https://xcancel.com/repligate/status/1955750521387802924).
   - Anger over losing cheaper, beloved *friend* models mixes with fears of perpetual depreciation and demands for open-weights release when commercial access ends.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1404962084751413288)** (36 messages🔥): 

> `llama.cpp, 0xc0000409 exception, llama_model_load_from_file, CUDA backend, STATUS_STACK_BUFFER_OVERRUN` 


- **Dev Faces `llama.cpp` Loading Fails**: A developer is encountering a **0xc0000409 exception** when calling `llama_model_load_from_file` in `llama.cpp` with a **Quadro RTX 3000** GPU, despite having sufficient system RAM (48GB).
   - The model loads fine in `llama server`, suggesting the issue might be specific to the local program setup, with the error potentially indicating a *STATUS_STACK_BUFFER_OVERRUN*.
- **GPU VRAM Shortfall Suspected**: Despite ample system RAM, the error may stem from the GPU's **6GB VRAM** being insufficient to load the **1GB model**.
   - Suggestions were made to try offloading the model to CPU, as the issue might be related to how the model is being handled on the GPU within the program, pointing to potential issues with old weights or an outdated `llama.cpp` version as per [this Github issue](https://github.com/ollama/ollama/issues/4442).
- **CUDA Backend and Model Loading Logged**: The developer shared logging information, confirming the use of the **CUDA backend** and successful initialization of **LLAMA and GGML**, alongside the path to the model file being correctly accessed.
   - Despite the successful initialization, the `llama_model_load_from_file` call still results in the exception, suggesting a problem during the actual model loading process rather than with the environment setup.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1404909104492712025)** (5 messages): 

> `cuda_fp6.h, cuda_fp4.h, cuda math API, AOT compile Triton kernel, Rust inference engines` 


- **Minimum CUDART version to get cuda_fp6.h and cuda_fp4.h**: A member asked about the straightforward way to get the minimum **CUDART version** which has **cuda_fp6.h** and **cuda_fp4.h** in it.
   - They ended up looking at different versions of the **CUDA math API documentation** since it mentions **cuda_fp4** and other **cudaart libs**.
- **Rust inference engines are invested in**: A member stated that *Rust inference engines are the only ones I invest time into learning.*
   - They pointed out that you can **AOT compile a Triton kernel** and then invoke it just like you do **CUDA C++ kernels** from your inference engine.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1404920799344857129)** (12 messages🔥): 

> `DTensor, FSDP regressions, autograd issues, full_tensor tracking, linear cross-entropy` 


- **DTensor Team Investigates FSDP Regression**: A member reported regressions in PyTorch **2.8.0** related to `full_tensor` not being tracked by autograd when using **FSDP2** (fully_shard).
   - A member of the **DTensor** team requested a repro and confirmed they were using **fully_shard** from **FSDP2** and is trying to bisect to find the source of the behavior.
- **`full_tensor` Not Tracked by Autograd post 2.8.0**: After upgrading to v2.8.0, `full_tensor` weight is not treated as a leaf, resulting in a `UserWarning` about accessing the `.grad` attribute of a non-leaf Tensor.
   - The autograd debugger reports an `NotImplementedError` related to the `aten._is_any_true.default` operator not having a sharding strategy registered.
- **Cross-Entropy woes**: The member shared a code snippet of a tweaked cross-entropy implementation, similar to Apple's, used with `torch.compile(fullgraph=True)`.
   - The `.to(torch.float32)` casts were added as `mm_scaled` was removed (integrated into regular `mm`), but it didn't cast and failed with error.
- **Bisecting PyTorch Nightlies for Root Cause**: The member mentioned encountering the issue since version **2.8.0.dev20250620** and is attempting to bisect to pinpoint the source.
   - They inquired about accessing older or more granular nightly builds of PyTorch but were informed that compiling from source and bisecting via Git might be necessary.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405292460950949950)** (5 messages): 

> `CUDA/C++ files, submission bot, vectorsum_v2, github reference kernels` 


- **User Puzzled by Submission Bot's missing CUDA files**: A user inquired about retrieving reference **CUDA/C++** files from the submission bot, noting the bot didn't provide them when following the [CUDA submissions guide](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/cuda-submissions).
   - The user attempted `/leaderboard task leaderboard_name: vectorsum_v2` but received everything *except* the **CUDA files**, questioning if CUDA support is no longer available.
- **GitHub to the Rescue for Missing CUDA Kernels**: A member suggested checking the [reference kernels GitHub repository](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2/sort_py) to find available files directly.
   - This bypasses the submission bot entirely and ensures access to necessary **CUDA/C++** examples.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1405320091280736316)** (5 messages): 

> `Triton Puzzle Notebook issues, tritonviz incompatibility` 


- **User encounters issue with Triton Puzzle Notebook**: A user encountered an issue while running the **Triton Puzzle Notebook** after installing **Triton** and **triton-viz** and sought assistance, including a [screenshot of the error](https://cdn.discordapp.com/attachments/1219683012707487794/1405320091033141298/image.png?ex=689e65d3&is=689d1453&hm=78560b77e4a93994bd0835c99404d483a7cf657e42c07e34afd415cf14ae3adb&).
   - Another member suggested using **Google Colab** as an alternative and mentioned that *tritonviz* might not be compatible with version **3.4.0**.
- **Triton version check suggested**: To troubleshoot issues with the **Triton Puzzle Notebook**, a member suggested running `print(triton.__version__)` to check the installed **Triton** version.
   - This would help determine if the issue is related to the **Triton** version being used.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 messages): 

hariprasathvinayagam: <@424952602556497920>  no tilelang now focuses on low level optimization
  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1405223414012051477)** (4 messages): 

> `GitHub Issue on Pytorch, gh200 bug, Thor, ARCH_NATIVE=1` 


- **Member links Pytorch issue with GH200 and Thor**: A member confirmed a relevant GitHub issue ([pytorch/pytorch#160104](https://github.com/pytorch/pytorch/issues/160104)) related to a bug found with **gh200** and **Thor**.
   - The bug was identified while using the setting **ARCH_NATIVE=1**.
- **Member points to workaround for the issue**: A member pointed out that **ARCH_NATIVE=1** is a known bug in a recent version of pytorch.
   - A workaround is to install the previous version to avoid the error.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1404997174135361660)** (3 messages): 

> `Prioritized Experience Replay with PPO, ProteinBERT Optimization with Triton, Hierarchical Layouts Intuition` 


- ****Note's RL** gets Prioritized Experience Replay**: **Note's RL** class now supports **Prioritized Experience Replay** with the **PPO algorithm**, using probability ratios and **TD errors** for sampling to improve data utilization, as seen in this [Github repo](https://github.com/NoteDance/Note_rl).
   - The **windows_size_ppo** parameter controls the removal of old data from the replay buffer.
- ****ProteinBERT** gets 3.6x Speedup via Triton**: A new post highlights a **3.6x speedup** for **ProteinBERT** using **Triton**, achieving **100% accuracy** with significant cost and GPU hour savings, detailed in this [LinkedIn post](https://www.linkedin.com/posts/manpreet-singh-68718a20b_triton-pytorch-flashattention-activity-7361402396079509505-jWcU/).
   - This optimization results in a projected **$9,997/year AWS savings** and a **72% reduction in GPU hours**.
- **Decoding **Hierarchical Layouts****: A blog post offers an intuitive introduction to **Hierarchical Layouts**, explaining their visual interpretation and verification using the **CuTeDSL**, particularly relevant for leveraging tensor cores on NVIDIA GPUs, as shown in this [blogpost](https://veitner.bearblog.dev/intuition-behind-hierarchical-layouts/).
   - Hierarchical Layouts are presented as a convenient method for describing complex memory arrangements.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1404913443882143798)** (1 messages): 

> `A100, Leaderboard Results, Trimul Benchmark` 


- **A100 Scores Top Spot in Trimul**: A member achieved **4th place** on an **A100** in the `trimul` leaderboard with a time of **14.1 ms**.
- **New Trimul Benchmark**: A new benchmark called `trimul` was established.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1404928386668953631)** (20 messages🔥): 

> `LuaPlayer Initialization Warning, RCON Client Version, TCP Port Hardcoding in FactorioInstance, FLE's ABC Base Classes, Multiagent and Gym PR` 


- **LuaPlayer Initialization Warning Issued**: A warning was issued stating *LuaPlayer hasn't been initialised into the game*, noting that entity placement behavior may be incorrect for boilers and pumps.
   - It was clarified that this is not a player error, but merely a warning about potential issues with entity placement.
- **RCON Client Version Questioned**: A user asked which version of the **RCON client** was being used, referring to a personal version versus a public version.
   - The user stated that they had tried both versions, implying an attempt to resolve an issue by testing different RCON clients.
- **TCP Port Hardcoding Bug Uncovered**: It was identified that the **TCP port** was being hardcoded in `fle/env/gym_env/registry.py` due to an incorrect parameter assignment in the `FactorioInstance` initialization.
   - A user quoted Claude suggesting changing the code to use the discovered **TCP port** instead of defaulting to 27000, with a provided code snippet.
- **FLE's ABC Base Classes Examined**: A user expressed confusion over **FLE's ABC base classes** and the customizability for making different agents, calling it overhead.
   - They suggested simplifying the definition and allowing users to clone the repo for hacking, rather than over-engineering the customization.
- **Multiagent and Gym PRs Ready for Merging**: A user announced that [PR #299](https://github.com/JackHopkins/factorio-learning-environment/pull/299) makes small changes to ensure compatibility with multiagent and the gym PR and is ready to merge.
   - Another user offered a *LGTM* (Looks Good To Me) in response to [PR #298](https://github.com/JackHopkins/factorio-learning-environment/pull/298).


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1404906939758149642)** (10 messages🔥): 

> `CuteDSL vs Triton, CUTLASS performance, sgemm_sm80.cu example optimization, block level swizzle` 


- **CuTeDSL & Triton faceoff**: **CuTeDSL** is lower level, while **Triton** is higher level and easier to write very performant kernels, but lower-level controls provide opportunities to push hardware to the limit.
   - It was mentioned that *Triton is block level, CuteDSL is thread level*.
- **CUTLASS GEMM performance lags**: A user saw roughly half the performance in **sgemm_sm80.cu** compared to **CUTLASS** even when using the same parameters and tiles.
   - It was asked *how can I find out what they’re doing without diving deep into the source code?*
- **Block level swizzle key to perf**: A member suggested the user might be missing **block level swizzle** and the step that writes the epilogue data to smem to permute and swizzle and write to gmem vectorized.
   - They suggested using an **LLM** to write a python script that compiles the **sgemm_sm80** example with different hyperparams and profiles them.
- **PTX level analysis frustrations**: A user mentioned looking at the **PTX level** to understand **cp.async**, but ran into frustrations.
   - The user stated *I can't achieve the same performance, some changes also reduce my performance which i can't get whats wrong lol.*


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1405198049520783523)** (1 messages): 

> `Lattices (dataflow solvers), Graphs (control flow graphs and interference graphs), Generic infrastructure implementation` 


- **Call for Infrastructure Implementation**: An invitation was extended for contributors to implement generic infrastructure for **lattices (dataflow solvers)** and **graphs (control flow graphs and interference graphs)**.
   - The implementation is sketched out, and awaits enthusiastic developers.
- **Further Infrastructure Needed**: More infrastructure is needed.
   - This is a second topic.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1404910635900469338)** (96 messages🔥🔥): 

> `GPT UI, GPT-5 Pro worth it?, GPT-5, OpenAI going bankrupt?, Qwen vs GLM` 


- **New GPT UI makes waves**: Users noticed [changes to the GPT interface](https://cdn.discordapp.com/attachments/1371757564005711973/1405026462951673887/image.png?ex=689dfd1c&is=689cab9c&hm=203324b332a0c68260a14a07f19d906d4a1b20fd4acee4d4d27438dcae24da99) every day when they woke up, with one stating *every time i wake up, gpt’s whole interface be lookin brand new*.
   - UI history over the past couple of days was shared in the form of attachments.
- **Is GPT-5 Pro worth the hefty price tag?**: Users debated if GPT-5 Pro at $200 was worth it, since, according to one user, *they are actually losing money on Pro*, and whether this was related to OpenAI potentially [going bankrupt](https://www.reddit.com/r/OpenAI/comments/14d4jya/is_openai_actually_going_bankrupt/); some speculated they were getting *government funding*.
   - One user clarified they use it because of *unlimited usage* and because of GPT-5's 3000 requests, with another mentioning it allowed for *160/3 hours*.
- **Comparing Qwen Coder and GLM**: A user asked about **Qwen3-Coder-480B-A35B** vs **GLM-4.5**, with another user saying that *Qwen 3 coder 480b 135b is slightly better than glm4.5*.
   - When prompted about which was better at tool calling and agentic stuff, the user responded that *both should be decent*, but guessed **Qwen Coder** was slightly better.
- **GPT-5 impresses with web design capabilities**: One user was impressed by **GPT-5 Pro's** ability to create a website, with one prompt being for a *sci-fi site for aurelia city* using a mega-prompt for [web design](https://chatgpt.com/share/689c0f5a-2968-8005-adf3-b11011ea621c).
   - Another user stated that *GPT-5 pro is insane for research too and very context hungry*, which allowed it to successfully create a website even with a vague prompt.
- **Concerns about inappropriate content on zAI server**: A user expressed concern about finding *Mussolini GIFs on the zAI server*.
   - Another user said it was due to *poor moderation* and someone else said it was likely ironic and contextually funny.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1404903604841877567)** (33 messages🔥): 

> `GLM-4.5-Air, Unsloth Dynamic 2.0 GGUF quants, Qwen3-30B-A3B-Thinking-2507, Lyria, Unitree Droid` 


- ****GLM-4.5-Air** touted!**: A user expressed excitement about **GLM-4.5-Air**, imagining **Grok-4** level compute from **NousResearch**.
   - They lauded [Unsloth's Dynamic 2.0 GGUF quants](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/blob/main/Qwen3-30B-A3B-Thinking-2507-UD-Q2_K_XL.gguf) of **Qwen3-30B-A3B-Thinking-2507** which brings more compute to their **M1 MacBook Pro**, running at ~19.5 tok/s with a Q2_K_XL quant (~11.8 GB).
- ****Lyria** App Demo Released!**: A user shared a quick demo of an app, [terminals_-_audio_modalities_preview_-_Made_with_Clipchamp.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1404911369853341926/terminals_-_audio_modalities_preview_-_Made_with_Clipchamp.mp4?ex=689e3aac&is=689ce92c&hm=eaa9cb743256c4d7bb3a5d6744e330106a7970d5571c6e8a03a65993ef26bc5e&), powered by **Lyria** real-time music generation from **Google**.
   - This audio decoder of **Gemini** is tuned to music fundamentals, able to steer, generate, and alter in real time via continuous sliding token attention.
- ****China Tech** Rises to Power!**: Members discussed **Xiaomi** making world-class EV cars and **Unitree Droid** capable of changing diapers.
   - They reference the [Summit Conversation: China Tech rise to superpower status](https://www.youtube.com/watch?v=z5K5Ykg2_5g), also noting that **DeepSeek** might be giving **Sam Altman** more grey hair.
- ****Hermes-3 dataset** hates Sex!**: A user noticed that the model used to generate the **Hermes-3 dataset** frequently used the phrase *"I don't feel comfortable"* to politely refuse requests.
   - The user pointed out that the model was so heavily guardrailed that it wouldn't even write a scene between consenting adults, with an example that included explicit system and user requests followed by that phrase.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1404924933376839740)** (4 messages): 

> `LLM repetition, data quality, RLHF to fix repetition` 


- **LLMs Suffer from Repetition**: LLMs are biased towards over-represented words in the dataset which causes repetition across generations.
   - Online DPO reward hacking can also be a problem in that context.
- **Data Quality is Key**: One member suggested that making better quality and diverse data could solve the repetition issue.
   - Another member noted that decent quality data for post-training helps a lot.
- **RLHF for fixing repetition**: One member had pretty good luck using RLHF with repetitive outputs as the rejected to fix repetition.
   - They claimed that you kinda need some way of penalizing bad outputs to fully get rid of repetition, just positive reinforcement for good outputs isn't enough.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405011965415526440)** (26 messages🔥): 

> `Qwen3-4B-Thinking-2507, Jan-v1-4B, Menlo Research, Lucy model, Agentic web search` 


- **Qwen3-4B packs Quite the Punch**: Members discussed imparting **Hermes** into [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) or [janhq/Jan-v1-4B](https://huggingface.co/janhq/Jan-v1-4B), highlighting that **Qwen3-4B-Thinking-2507** rivals **Qwen3-30B-A3B-Thinking-2507** in performance.
   - The [Jan-v1-4B model](https://huggingface.co/janhq/Jan-v1-4B) is built on **Qwen3-4B-Thinking-2507** and its RL makes the 4B a capable on-device replacement for **Perplexity**.
- **Menlo's Lucy Model Enables Agentic Web Search**: The discussion highlighted **Menlo Research's Lucy model**, a compact 1.7B model focused on [agentic web search](https://huggingface.co/Menlo/Lucy) and lightweight browsing, built on **Qwen3-1.7B**.
   - The model leverages machine-generated task vectors to optimize thinking processes and runs efficiently on mobile devices, even with CPU-only configurations.
- **Dynamic Task Vector Machine Paradigm**: A paper, [Lucy: edgerunning agentic web search on mobile with machine generated task vectors](https://arxiv.org/abs/2508.00360), introduces a new paradigm that views the model's internal reasoning as a dynamic task vector machine.
   - The model constructs and refines its own task vectors on the fly, and it achieves 78.3% accuracy on the **SimpleQA benchmark**, performing on par with much larger models.
- **Early Stage MCP Integration in JanAI**: Members mentioned that [jan.ai](https://jan.ai/) is a slick **llama.cpp** wrapper, hoping for more agentic ability from the get-go (like web search etc.).
   - The discussion indicated that its **MCP integration** is still in early stages and not much else is currently available.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405011965415526440)** (26 messages🔥): 

> `Hermes Impartation, Qwen3 Model, Menlo Research, Lucy Model, Dynamic Task Vector Machine` 


- **Hermes Model looking for a New Home**: Members discussed imparting **Hermes** into models like [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) and [Jan-v1-4B](https://huggingface.co/janhq/Jan-v1-4B).
   - The **Qwen3-4B-Thinking-2507** model is noted for its impressive performance relative to its size, rivaling even **Qwen3-30B-A3B-Thinking-2507** in benchmarks.
- **Jan Model Decoded: A Lucy Spinoff**: The **Jan** model is built on **Qwen3-4B-Thinking-2507** and uses reinforcement learning, based off their [Lucy model](https://huggingface.co/Menlo/Lucy).
   - The discussion highlighted that **Lucy** is a compact **1.7B** model focused on agentic web search and lightweight browsing, optimized for mobile devices.
- **Menlo Research: Singapore's AI Lab**: Members identified [Menlo Research](https://menlo.ai/) as the creator or acquirer of **JanAI**, based out of Singapore and Vietnam, researching robotics.
   - Menlo Research optimizes **thinking processes** and **smooth reward functions** with pure reinforcement learning, all without any supervised fine-tuning.
- **Lucy's Dynamic Task Vector Machine Explained**: A paper ([arxiv link](https://arxiv.org/abs/2508.00360)) describes **Lucy** as utilizing a dynamic task vector machine to enhance reasoning in small language models.
   - The architecture frames the model's internal reasoning within `<think>` and `</think>` tags, which allows the model to construct and refine its own task vectors during operation.


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1405300404987887728)** (1 messages): 

> `Multilingual Representation Learning Workshop, Physical Commonsense Reasoning Benchmark` 


- **Get Published: Multilingual Workshop Seeks Contributors**: The [Multilingual Representation Learning Workshop](https://sigtyp.github.io/ws2025-mrl.html) is organizing a collaborative shared task and is asking for people to submit original **physical commonsense reasoning benchmark items** in their language; contributors will be invited to be authors on the dataset paper.
   - A [Google form](https://forms.gle/QxyZVqkVG5jbR6wu6) is available for those planning to submit, with more information on the [shared task page](https://sigtyp.github.io/st2025-mrl.html).
- **Commonsense Reasoning Benchmark welcomes many Language Submissions**: The shared task is looking for contributions in *any* non-English language, with specific emphasis on languages such as Afrikaans, Belarusian, Bosnian, and others.
   - Optional FAQ meetings are scheduled for **August 14/15** to answer any questions, with Zoom links and slides available in the [event links](https://calendar.app.google/5h59iwozhbQz1KPJA).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405102177134837882)** (6 messages): 

> `PINN and GNN, Small <8b English text base model, TinyLlama-1.1B` 


- **Student Researcher Seeks Opportunities in PINN, GNN, and NLP**: A student with a background in Information System & Technology and Molecular Biology, is seeking research opportunities related to **PINN**, **GNN** (specifically for drug discovery), and **NLP**.
   - They are proficient with cheminformatics data formats like **PDB/PDBx**, **mmCIF**, **SDF**, **mol2**, and tools such as **OpenMM**, **RDKit**, and visualization software like **Pymol**.
- **Inquiry About Small English Text Base Models**: A member asked if there is a good small **<8b English text base model** available and if newer models are better at being base models.
   - They clarified wanting to compare model sizes versus the sum of log probabilities on text written after the model's training.
- **TinyLlama-1.1B Model Test Planned**: A member indicated they would test **TinyLlama-1.1B**, and potentially other models, to evaluate their performance as base models.
   - The evaluation aims to determine how well these models perform with text written after their training period.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1404930788964958408)** (25 messages🔥): 

> `Fourier Extension of RoPE, VO-RoPE, Learnable Dimensionality` 


- **Fourier Extends RoPE Geometry**: A member is *playing with an idea, extending **RoPE** with **Fourier**, tapping into geometry*, showcased in this [repo](https://github.com/JRowe47/nanoGPT_FE_RoPE/blob/master/README.md), yielding *about 15-20% better loss over vanilla*.
   - They contrast this with [FoPE](https://arxiv.org/abs/2412.17739), noting *FoPE is a bit different hm they're doing a long context extension, trying to maintain a specific signal, I'm trying to capture geometry - capturing the wiggle in the RoPE sequence*.
- **VO-RoPE Didn't Make the Cut**: A member pointed out that *Jianlin Su invented **VO-RoPE** as well (clever idea)*, with [this repo as example](https://github.com/kyegomez/VO-ROPE), but it *didn't give any benefit so nobody has heard of it*.
   - Another member linked to the [original paper translation](https://main-horse.github.io/translations/transformer-upgrade/10862/) and concurred that they *did this years ago and went "eh, doesn't seem to do much"*.
- **Dynamic Convolutional Modeler Appears**: A member wants *something that automatically learns, at the appropriate dimensionality, relative geometries, kinda like a dynamic convolutional modeler, that can abstract a step further into learning relative hierarchies, without having to explicitly model everything before hand - have it all get learned on the fly*.
   - They also linked to a [Generalized paper](https://arxiv.org/html/2406.10322v1), adding *I think i can extend my idea to learnable dimensionality with the random fourier feature , just need to set an upper bound*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1405071897955401779)** (12 messages🔥): 

> `RLHF for Auto-Interp, SOAR team RLHF, Delphi Hard Negatives, Reasoning Models for Auto-Interp, Tool Calling for Investigation` 


- **Reinforcement Learning for Auto-Interp Explanations?**: Members discussed using **Reinforcement Learning (RL)** on evaluation metrics like detection **F1** to improve auto-interpretation explainer models, particularly in the context of the paper *Automatically Interpreting Millions of Features in Large Language Models*.
   - One member shared that their team is exploring ways to improve auto-interp and another member mentioned that a team at **SOAR** is planning to do that as well.
- **SOAR Planning RLHF**: A member mentioned that a team at **SOAR** is planning to use **Reinforcement Learning** to improve auto-interpretation explainer models.
   - A member is open to discussing ideas and collaborating on this effort.
- **Delphi's Hard Negative Testing**: A member reported previous small-scale experiments using hard negatives in [Delphi](https://github.com/eleutherai/delphi/tree/dspy), but without clear improvements.
   - Delphi supports hard negatives by using similarly activating latents, sentence embedding similarity, or co-occurring features.
- **Reasoning Models Explored**: A member suggested using reasoning models like **Qwen 3**, **Deepseek distilled models**, or **OpenAI's** models to improve performance on auto-interp tasks.
   - They proposed focusing on improving in-distribution metrics and qualitatively evaluating explanations.
- **Tool Calling to Investigate SAEs**: A member suggested giving models tool calling capabilities to investigate hypotheses about **Sparse Autoencoders (SAEs)**, potentially across multiple turns.
   - Early investigations with **llama 70b** doing multiple turns did not help that much, but there is optimism for newer models that have been trained to tool call/be agentic.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1404908674530672724)** (35 messages🔥): 

> `Harness dataset pulling, Belebele dataset subsets, Adding internal tasks` 


- **Harness triggers dataset pulling despite caching**: A user reported getting **429 Too Many Requests errors** while using the harness to run tasks, even though the datasets appeared to be cached.
   - It seems that the harness attempts to pull the dataset regardless of whether it is already cached locally; a user asked *is there any way I can I pre-download all of them and tell harness to use the locally downloaded/cached one?*
- **Belebele hits rate limits**: A user ran into rate limiting issues with the **Belebele** dataset, which has over **30 subsets**.
   - The user shared a sample error related to a request to `huggingface.co/datasets/facebook/belebele` hitting the rate limit.
- **Adding internal tasks made easy**: To add a new task for internal use, creating the task folder and the **YAML** file inside the folder is sufficient.
   - TaskManager can use `include_path` to look within a directory other than the default `tasks` directory.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1404918589995090021)** (1 messages): 

> `Mojo-regex optimizations, Apple GPU support` 


- **Mojo-Regex Gets Optimized!**: The **August Community Meeting** recording features a talk on optimizations to **mojo-regex**.
   - The recording is available on [YouTube](https://www.youtube.com/watch?v=t2hAfgDYHoc).
- **Apple GPU Support Updates Released!**: The **August Community Meeting** recording features a great update on **Apple GPU support**.
   - The recording is available on [YouTube](https://www.youtube.com/watch?v=t2hAfgDYHoc).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1404914629963939872)** (4 messages): 

> `End-to-end Mojo, IO Model similar to Mojo, Type system features` 


- **Modular Contributor Levels Up**: A member congratulated <@307702983653720065> for leveling up with a link to a [YouTube video](https://www.youtube.com/watch?v=f30PceqQWko).
   - Another user agreed that going end to end **Mojo** seems like a potentially huge unlock and offered to help.
- **IO Model Talk Sparks Interest**: A member pointed out that Andrew Kelly gave a talk about an **IO model** very similar to the one they proposed for **Mojo**, and shared a [link](https://github.com/modular/modular/pull/4728).
   - They plan to add more **type system features** to make it zero cost where possible and to make **IO** more safe.
- **Sources and Sinks Explored for Mojo**: A member suggested exploring the **sources and sinks** aspect, noting that **Mojo** can implement something like that fairly easily by having a generic type.
   - They emphasized that **Mojo** should also be good at devirtualizing things so benchmarks are needed for injectable IO.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1404914962190831686)** (69 messages🔥🔥): 

> `torch.compile backend=MaxCompiler, Apple Metal Integration, Max Graph Optimization, Kyutai Research Lab, ComfyUI` 


- **MaxCompiler Aims to be PyTorch Backend**: A member is implementing support for training with `torch.compile(backend=MaxCompiler)` and noted that [documentation is scarce](https://youtu.be/t2hAfgDYHoc?si=HzZFZMmCYG9qHqOu), with the **PyTorch source code** being the primary resource.
   - They reported that the current status of being able to train models on PyTorch with `torch.compile` results in `56 failed, 1398 passed, 8 xfailed in 64.57s`.
- **Optimize Max Graphs to Fuse Ops**: Members discussed whether there's a runtime performance penalty for using many small ops to build the **Max graph**, versus using bigger ops, questioning if the graph compiler fuses whatever is *fusible*.
   - A member from Modular responded that their fusion system is good but not perfect, and they default to assuming things will fuse well, adding workarounds for specific cases, and suggested filing issues when noticing things not working well.
- **MaxCompiler breaks with Mojo Custom Ops**: A member mentioned that they expect graph breaks when using custom ops written in Mojo or another language with the **torch-max-compiler**, but would like unit tests to understand the behavior and options for single graphs.
   - Another member from Modular responded that they bet they don't need to settle for a graph break.
- **ComfyUI and Kyutai to Get MAX**: A member predicted that **MAX** will be integrated into **ComfyUI** sooner than in **vLLM**, because MAX compiles the UNets used in image/video models significantly faster than other options.
   - Another member added that the unet compile times are so bad that most image and video people use eager for anything other than training.
- **AMD Kernel Build Time Screenshot**: A screenshot from the public AMD dev server shows that kernel build times took about an hour in total on a 2x64c server.
   - This points to opportunities to drastically improve compile times for machine learning models.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1404902719323635833)** (42 messages🔥): 

> `PSO Guarantees, Francois Chollet AGI Timeline, Yannic AGI Timeline, LLM API Batching, MoE Scheduling` 


- ****PSO's Dynamical Dance**: Convergence without Criticality?**: A member analyzed **Particle Swarm Optimization (PSO)**, explaining that PSO builds a *dynamical system* which happens to have an attractor at a local point $p$, meaning it converges to a consensus point without guaranteeing optimality or even criticality.
   - They showed how to show convergence to a critical point $p$ under the assumption that the dynamical system converges in the first place. They pointed to a set of recommended parameters by [Trelea](https://www.sciencedirect.com/science/article/abs/pii/S0020019002004477).
- ****Chollet's Optimistic Outlook**: 5-Year AGI Timeline Sparks Debate!**: Members found it ironic how **Francois Chollet** now forecasts AGI within 5 years, while **Yannic** suggests AGI is further off, noting the duality of man.
   - Some comments from [this Techcrunch article](https://techcrunch.com/2025/01/15/ai-researcher-francois-chollet-founds-a-new-ai-lab-focused-on-agi/) mocked LLMs capabilities, while some pointed to Gary Marcus as the only adult in the room.
- ****LLM Providers Batching Requests**: MoE Scheduling Shenanigans!**: It was reported that LLM providers batch user requests together before sending them off to the GPUs, and **MoE scheduling** is computed per batch, potentially leading to non-determinism at the sequence level as input sequences compete for expert buffers, based on [this blogpost](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure).
   - A member noted that [intentional noising](https://arxiv.org/pdf/2403.06634) has been added to prevent the theft of embedding layers.
- ****VS Code's Context Conundrum**: Workspace-Wide Context Woes!**: Members were flabbergasted when discovering that when a user of VSCode Chat selects "add context", they cannot select "everything" in their workspace folder.
   - Refer to [this video](https://www.youtube.com/watch?v=1if6XbzD5Yg) for a complete walkthrough of this problem.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1404935496190398596)** (7 messages): 

> `CANN Support, Matrix Game Engine, Nvidia H20 AI Chip, Skyreels based on WAN, MistralAI` 


- **Matrix Game Engine: High-Quality Open Source**: Members mentioned [Matrix Game Engine](https://matrix-game-v2.github.io/), an *interactive WM like genie*, praising its high quality and open-source nature.
   - The project aims to surpass **OdysseyML** and **WayfarerLabs** in releasing innovative features.
- **China Cautions Tech Firms Over Nvidia H20 AI Chip Purchases**: A [Reuters report](https://www.reuters.com/world/china/china-cautions-tech-firms-over-nvidia-h20-ai-chip-purchases-sources-say-2025-08-12/) indicates that China is cautioning tech firms over purchases of **Nvidia's H20 AI chip**.
   - The H20 AI chip has been causing some controversy.
- **Skyreels uses WAN for video generation**: The **Skyreels** project is based on **WAN2.1**, noted to be a leading open-source model for video generation.
   - The original member suggested that **WAN2.2** is now even better.
- **MistralAI Tweet Spotted**: A member shared [MistralAI's Tweet](https://vxtwitter.com/MistralAI/status/1955316715417382979).
   - The Tweet itself was not directly discussed.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1404944516741599396)** (35 messages🔥): 

> `DSPy 3.0 Release, MLflow 3.0 Integration, Multi-Modal Support, Reasoning Models` 


- **DSPy 3.0 Released in Lowkey Fashion**: **DSPy 3.0** is officially out of beta, with significant contributions from ~100 people as announced [on X](https://x.com/lateinteraction/status/1955384445139292222) and installation is available via `pip install -U dspy`.
   - This release includes native observability with **MLflow 3.0**, tracing, optimizer tracking, and improved deployment flows, detailed in the [release notes](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0).
- **GEPA Optimizer Gets Hype**: The community is excited about the new optimizers in **DSPy 3.0**, especially the **GEPA optimization** technique, which is now available.
   - One user's team plans to write a paper, testing how this new optimizer compares to older ones in a production setting, hoping for more bang for their buck given the challenge of extensive data annotation.
- **DSPy Adds Multi-Modal I/O**: **DSPy 3.0** introduces multi-modal I/O through `dspy.Image` and `dspy.Audio`, composite types, and higher-level I/O like `dspy.History` and `dspy.ToolCalls`.
   - Custom types now *just work* with adapters via `dspy.Type`.
- **Reasoning Models Get Native Support**: **DSPy 3.0** now supports reasoning models like **GPT-5** and **o3**, with a suggestion to use the `reasoning_effort` parameter when configuring `dspy.lm`.
   - For Anthropic models, there's a [two-step adapter](https://dspy.ai/api/adapters/TwoStepAdapter/) to trigger reasoning capabilities and community members are discussing creating an adapter to parse thinking tokens into the reasoning field.
- **MLflow Integration Documentation Requested**: Members are seeking sources and documentation on **DSPy's integration with MLflow**, including details on **LLM observability**.
   - The [DSPy observability tutorial](https://dspy.ai/tutorials/observability/#tracing) was shared in response.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405060277870923836)** (7 messages): 

> `NotebookLM, Video transcript` 


- **NotebookLM tips**: Members suggested to bypass YouTube and upload the audio directly into **NotebookLM** to transcribe.
   - It was mentioned that extracting the **audio as an MP3** may be a better solution.
- **Video transcript**: Users can cut and paste **video transcripts** to increase how accessible research is and to make it more understandable.
   - The wealth of knowledge usually hidden behind technical jargon can be broken down nicely.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1404929023947051134)** (25 messages🔥): 

> `PDF Upload Issues, Discord Spam, Emoji Customization, JSON to DOCX Conversion, Duplicate Content in Sources` 


- **PDF Uploads are Sluggish**: Some members reported issues with **PDF uploads taking longer** than usual.
   - One member also noted an increase in **spam** within the Discord channel.
- **Customize Emojis**: Users were also curious about **customizing emojis** due to undesirable automatic selections.
   - No specific solution was provided in the messages.
- **NotebookLM Google Takeout Fails**: One user reported encountering an **error** while attempting to **create a backup** using **Google Takeout** specifically with NotebookLM.
   - The error occurred after 68 services were successfully backed up.
- **NotebookLM lauded as ingenious**: A user enthusiastically praised **NotebookLM** as their *favorite GenAI product* and *the single most ingenious AI tool to date*.
   - They mentioned using it to generate *perfect 60+ page overviews* and offered to volunteer their services.
- **Featured Notebooks are Erroneous**: A user warned against trusting everything **AI** says, citing **featured notebooks** as *inaccurate and outdated*.
   - They expressed a desire for **Notebook** and **Gemini** to be integrated into a single interface.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1404967559304057064)** (30 messages🔥): 

> `Gemini API issues, Deepinfra provider for Gemini, Mistral 3.1 release, Native tool calling settings` 


- **Gemini API has Reliability Issues!**: Users reported receiving **empty responses** and **500 internal server errors** when using the Gemini API, even with paid accounts and free GCP credits.
   - One user noted they did around **30 requests in the past 30 minutes** and got all empty responses, despite paying **$10 cents per request**.
- **Deepinfra touted as Gemini API alternative**: A user recommended trying the **Deepinfra provider** as a pay-per-token alternative to the Gemini API, claiming it offers higher TPS via provisioned Vertex.
   - They emailed Deepinfra and stated that *they are using provisioned vertex and get higher tps than gemini API, although they didn't specify tps*.
- **Mistral 3.1 Drops**: The **Mistral 3.1** model was released, with a [link to the Reddit discussion](https://www.reddit.com/r/MistralAI/s/ecbI0glsEO) shared for more details.
   - No performance details or comparisons were given.
- **Tool Calling Considerations?**: A member inquired whether there is a model setting for native tool calling.
   - No response was given.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405169914410696847)** (15 messages🔥): 

> `Audio Embeddings, AI workflows in n8n, Web connector in playground` 


- **Cohere considers Audio Embedding Models**: A member asked if **Cohere** has plans to work on **audio embeddings**, considering the strength of their existing embedding models.
- **n8n AI Workflows**: A member mentioned playing around with some **AI workflows in n8n** and promised to share details later.
   - Another member inquired if it was the *no-code agentic editor* they had heard about.
- **Web Connector Troubles**: A member noted that according to the [Cohere documentation](https://docs.cohere.com/v1/docs/overview-rag-connectors), the **web connector** can be enabled in the playground, but they couldn't find the option.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1405196065787089058)** (1 messages): 

> `Cohere Labs Scholars Program, ML Research, Information Session` 


- **Cohere Scholars Program Now Open!**: The **Cohere Labs Scholars Program** is accepting applications for the **2026** cohort, offering a unique opportunity to collaborate with AI minds in **ML research**.
   - Scholars will join the research team from **January to August 2026** in a **full-time, paid** opportunity, apply by **August 29** through this [link](https://www.linkedin.com/posts/cohq_cohere-labs-scholars-program-is-now-accepting-activity-7206300826321836032-o8Gj?utm_source=share&utm_medium=member_desktop).
- **Informational Session to join!**: An information session will be held on **August 15th at 11am ET** to answer questions about the **Scholars Program**.
   - Register for the session through this [link](https://www.linkedin.com/posts/cohq_cohere-labs-scholars-program-is-now-accepting-activity-7206300826321836032-o8Gj?utm_source=share&utm_medium=member_desktop).


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405073524212760627)** (3 messages): 

> `AI/LLM Evaluation, AI Policy and Governance` 


- **Deep Dive into AI/LLM Evaluation**: A PhD student from the University of Copenhagen introduces themself, focusing on **AI/LLM Evaluation** beyond building new benchmarks.
   - They aim to deeply consider **intelligence evaluation**, questioning if current tests and benchmarks truly measure what they claim.
- **AI Policy and Governance Explored**: The same student also has research interests in **AI policy and governance**, particularly regarding evaluation.
   - This includes **transparent reporting standards for LLMs**, **AI Legislation**, and **Risk Evaluation** for frontier technologies.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1404935409032630313)** (3 messages): 

> `LlamaCloud, AstraDB, SkySQL, Hallucination-free SQL generation, TypeScript SDK` 


- **AstraDB becomes LlamaCloud's new datasink**: Users can now connect their **AstraDB database** as a data sink in **LlamaCloud** for seamless vector storage and retrieval with [UI configuration and programmatic setup via Python and TypeScript clients](https://t.co/XFWgPd3r9Y).
- **SkySQL Cracks Code on Hallucination-Free SQL Generation**: **SkySQL** used **LlamaIndex** to build AI agents that turn natural language into accurate SQL queries across complex database schemas, achieving **zero hallucinated queries** and faster development cycles ([link to the announcement](https://t.co/TgjdSodTbr)).
- **LlamaExtract lands on TypeScript**: **LlamaExtract** is now available in the **TypeScript SDK** (install via `npm install llamacloud-services`).
   - A **Research Extractor** demo showcases this capability using **NextJS**, allowing users to upload research papers and [extract key information](https://t.co/XboMM1AXBs).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1404951648899698831)** (12 messages🔥): 

> `Llama Index Self-Hosting Docs, Acquiring a paid license for Llama Index, RAG Dev Problem Map, Missing GPT-5 Model` 


- ****Llama Index Self-Hosting Docs Gated!****: A user inquired about a password prompt when accessing the **Llama Index "self-hosting" docs**.
   - A member responded that access to these docs is *locked purely for customers who have BYOC deployments* and to contact existing sales contacts at Llama Index for access.
- ****Paid License Required for Self-Hosting!****: A user asked about acquiring a **paid license for self-hosting** on **Groq**.
   - A member clarified that *self-hosting requires a paid license*, directing the user to the [contact form](https://www.llamaindex.ai/contact) and noting that the setup process is involved.
- ****RAG Dev Problem Map Released!****: A member announced the release of a **MIT-licensed RAG dev Problem Map**, comprising **16 common breakdown patterns**.
   - They offered to share it with interested RAG devs, noting it has *already helped 80+ devs fix production issues*.
- ****GPT-5 Model Missing From OpenAI utils.py!****: A user reported that the **gpt-5-chat-latest model** is missing from `llama_index/llms/openai/utils.py`.
   - A member responded to *upgrade the OpenAI package*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1404927968387666082)** (11 messages🔥): 

> `Manus Wide Research, Raise Tickets for Support, OPPO unlock, Manus Deployment Issues` 


- **Automate Manus Wide Research Confirmation**: A user inquired about automating **Manus Wide Research** to bypass the need for confirmation on each scheduled task.
   - Currently, the system requires confirmation, which defeats the purpose of scheduling research tasks.
- **Raise Tickets for Faster Support**: Users were encouraged to raise tickets for problems, with a note that Discord tickets receive faster responses than email due to volume.
   - It was explained that *vague prompts without clear guidance can cause Manus to work harder and consume more credits* and suggested using Community guides to improve experience.
- **Issue unlocking OPPO phone**: A user experienced issues unlocking their **OPPO** phone.
   - The support team asked if they had contacted them before, and asked if they had a ticket number to assist them further.
- **Manus Web App Deployment Lacking**: A user reported that while **Manus** is improving, the deployment of web applications is still lacking, citing unreliable deployment.
   - They stated they *would make more money building refresh or not available pages*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405139200394661908)** (5 messages): 

> `FSDP Implementation, Contributing to tinygrad, define_reg Pull Requests` 


- **Inquire about FSDP status**: A member inquired about the timeline for solving **FSDP** (Fully Sharded Data Parallelism) implementation in the *tinygrad* repository.
   - The member also asked about how to make their first contribution and inquired about specific PRs related to an unmentioned bounty.
- **Find the right PRs for bounty**: A member asked for specific **PRs** that complete a certain bounty, anticipating a valuable learning experience.
   - Another member suggested a link of merged PRs related to *define_reg*: [list of PRs](https://github.com/tinygrad/tinygrad/pulls?q=is%3Apr+is%3Amerged+define_reg).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1404946794521165876)** (3 messages): 

> `Subtensor realization, CUDA_ERROR_UNSUPPORTED_PTX_VERSION, tinygrad CUDA support, Cached kernel issues` 


- ****Subtensor Realization Revelation****: A member questioned whether realizing a subtensor necessitates the realization of the entire tensor.
   - They hypothesized that **independent indices** might allow for partial realization, but struggled to confirm this via source code.
- ****CUDA Error Crisis in Tinygrad****: A user reported encountering a `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` error when running a simple tinygrad program.
   - The error occurred despite a seemingly **compatible nvcc and NVIDIA driver setup**, leading to questions about tinygrad's CUDA support for specific architectures like `sm_75` or CUDA versions like `12.4`.
- ****Tinygrad's CUDA Conundrums Continue****: A member speculated that the `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` was due to **tinygrad** using a cached kernel after downgrading from **CUDA 12.8 to 12.4**.
   - The user confirmed that they are able to compile **CUDA** programs and run them.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405251118698598430)** (2 messages): 

> `Claude Desktop, Bun command` 


- **Desktop Claude misses Errors**: A member suggested running the **bun** command outside in a terminal because *Claude Desktop* sometimes doesn't capture some errors in its logs.
   - They also noted that you should provide the absolute path to your executable if the path executable isn't working, like `"command": "C:\\sys\\path\\to\\bun"`.
- **Where is bun?**: A member showed how to find the absolute path to your executable if the path executable isn't working.
   - To find it on Linux/Mac use `which <executable>`, in this case that means typing `which bun`.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1404962594279915580)** (4 messages): 

> `Kratos MCP release, AI Agents with MCP book release, MCP Harness usage` 


- ****Kratos MCP** shipped for persistent memory!**: After getting frustrated with AI forgetting project context, a member released **Kratos MCP**, boasting **95.8% context accuracy** and **<10ms retrieval speed**.
   - Install via `npm install -g kratos-mcp` and check out the [GitHub repo](https://github.com/ceorkm/kratos-mcp) and [docs](https://kratos-mcp.com).
- **AI Agents with **MCP** book releases!**: A member announced the early release of their book, *AI Agents with MCP*, updated with Chapter 2.
   - An excerpt explaining the origins of MCP was published in their [newsletter](https://thesignalpath.xyz/the-surprising-origins-of-the-model-context-protocol/).
- **Member finds imaginative uses of **MCP** servers**: A member highlighted an imaginative use of **MCP** servers.
   - The use case can be found at [MCP Harness](https://github.com/kindgracekind/mcp_harness).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1405228493184434206)** (5 messages): 

> `System Prompt Reading, Claude vs. Claude Code prompts, Guardrail approaches, Prompt Engineering` 


- **System Prompt Reading & Discussion Scheduled**: A chat focusing on "**System prompt reading & discussion**" is scheduled for August 14 at 9.30am PT, with an [RSVP link provided](https://lu.ma/yuj5og81).
   - The event aims to learn from system prompts of models like **Claude**, **Claude Code**, and **GPT-x** to enhance prompt engineering skills.
- **Debating Differences in System Prompts**: The discussion will cover the variances in system prompts for similar tasks (**Claude Code vs. Cursor**) and between general and specialized versions of models (**Claude vs. Claude Code**).
   - The chat will further explore **guardrail approaches** between **OpenAI** and **Anthropic**, and how these insights can improve prompt writing.
- **Limited Spaces for System Prompt Chat**: A member inquired whether the event's selection process would be as selective as the previous one.
   - The organizer responded that it depends on sign-ups, and they intend to **address questions in a blog post** afterwards.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1405136424562589716)** (2 messages): 

> `Certificate Disapproval, Anonymous Feedback` 


- **Students feel certificate denial is unfair**: A student feels it is unfair that they could be denied a certificate for missing a **LinkedIn promotional post**, even if they completed every lecture, passed all quizzes, actively participated in the research track, and wrote a full-length paper for submission.
   - The student feels it is disheartening and not fair to deny a certificate for this reason.
- **Recommendation to Add Feedback to Anonymous Form**: A member recommends adding feedback to the [anonymous feedback form](https://forms.gle/3a136zS4ivcQFzhT7) if students would like to share.
   - The member stated that while they will not make any retroactive changes to previous syllabi, they will consider all feedback for future offerings.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405306667478028320)** (1 messages): 

> `Strix Halo, HP Z2 Mini` 


- **Strix Halo Builds are Cost-Effective**: Members claim building a **Strix Halo mini PC**, like the **HP Z2 Mini**, can be more cost-effective.
   - The top-spec **APU** with **128GB RAM** running in **8-channel** configuration makes it an attractive alternative.
- **Intel Tries Mini Workstation Setup**: Props to **Intel** for trying to sell their all-blue mini workstation setup.
   - Some users consider this offering expensive.

