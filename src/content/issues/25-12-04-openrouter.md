---
id: MjAyNS0x
title: OpenRouter's State of AI - An Empirical 100 Trillion Token Study
date: '2025-12-04T05:44:39.731046Z'
description: >-
  **OpenRouter** released its first survey showing usage trends with 7 trillion
  tokens proxied weekly, highlighting a 52% roleplay bias. **Deepseek**'s open
  model market share has sharply declined due to rising coding model usage.
  Reasoning model token usage surged from 0% to over 50%. **Grok Code Fast**
  shows high usage, while **Anthropic** leads in tool calling and coding
  requests with around 60% share. Input tokens quadrupled and output tokens
  tripled this year, driven mainly by programming use cases, which dominate
  spending and volume. Google launched **Gemini 3 Deep Think**, featuring
  parallel thinking and achieving 45.1% on ARC-AGI-2 benchmarks, and previewed
  **Titans**, a long-context neural memory architecture scaling beyond 2 million
  tokens. These advances were shared by **Google DeepMind** and **Google AI** on
  Twitter.
companies:
  - openrouter
  - deepseek
  - anthropic
  - google
  - google-deepmind
models:
  - grok-code-fast
  - gemini-3
  - gemini-3-deep-think
  - gpt-5.1-codex-max
topics:
  - reasoning
  - coding
  - tokenization
  - long-context
  - model-architecture
  - benchmarking
  - agentic-ai
  - prompt-engineering
people:
  - quocleix
  - noamshazeer
  - mirrokni
---


**Data is all you need.**

> AI News for 12/3/2025-12/4/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 7543 messages) for you. Estimated reading time saved (at 200wpm): 563 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

OpenRouter's first survey is out, in [web](https://openrouter.ai/state-of-ai) and [pdf](https://openrouter.ai/assets/State-of-AI.pdf) forms, and it is delightfully well done. Obviously OpenRouter has a bias (52% usage is for *ahem* "roleplay"), and there are other token consumers with higher volume, but OR is the ~only player that has this level of open data proxying 7T tokens per week.

Some picks:

Deepseek's 50% open model marketshare has plummeted

[A stacked area chart showing the decline of DeepSeek's market share in open-source AI models over time, with increasing fragmentation an](https://resend-attachments.s3.amazonaws.com/iNWx03qmrMiRP68)

mostly because coding rose and nobody uses deepseek for coding:

[A stacked bar chart showing DeepSeek's most popular AI model usage categories over several weeks in 2025, with roleplay and casual](https://resend-attachments.s3.amazonaws.com/CgK6rfs1J30lj5v)

Reasoning models went from 0 to >50% usage

[A line graph showing the increasing proportion of reasoning tokens used by AI models over time, rising from near 0% to over 50% by November](https://resend-attachments.s3.amazonaws.com/w5xn5WbKA583Dvh)

Grok Code Fast is weirdly high usage even excluding free promo:

[A bar chart showing the top used AI models by token volume, with Grok Code Fast 1 leading, followed by Google's Gem](https://resend-attachments.s3.amazonaws.com/c3YLyLKYM0WjP7A)

Anthropic dominates tool calling and koding

[A stacked bar chart showing the share of programming requests by different AI model providers over several weeks, with Anthropic dominating around 60%](https://resend-attachments.s3.amazonaws.com/kwlM5i7c6Gv5G3K)

:

[A stacked bar chart showing the top 10 most used AI models with 'Tool-Call' finish reason across different months in 2](https://resend-attachments.s3.amazonaws.com/Q7XYcgX1fOCRlrl)

Input tokens 4xed, output tokens 3xed this year...

[A graph showing the growth of prompt and completion tokens over time, illustrating the increasing complexity and length of AI model interactions.](https://resend-attachments.s3.amazonaws.com/gvqejRi2B53yOy8)

... only because of programing usecases

[Line graph showing average number of tokens per request across different domains, with programming (orange line) having the highest and most variable token count over time.](https://resend-attachments.s3.amazonaws.com/wP469stQwrPLZIz)

[A stacked area chart showing the changing proportions of different AI model usage categories over time, with programming increasing from 11% to 50%](https://resend-attachments.s3.amazonaws.com/dyFnvG14kK0dqGK)

... which are at a sweet spot of spend and volume

[A scatter plot showing log cost versus log usage for different AI workload categories like programming, technology, science, and translation, highlighting variations across mass-](https://resend-attachments.s3.amazonaws.com/iAkVnVSLMuiLAgR)

---

# AI Twitter Recap

**Reasoning and Model Architecture: Gemini 3 Deep Think and Google’s “Titans”**

- **Gemini 3 Deep Think (rollout + benchmarks)**: Google launched an updated Deep Think mode for Gemini 3 to Google AI Ultra subscribers inside the Gemini app. It uses “parallel thinking” (multiple hypotheses in parallel) and derives from the variants that reached gold-medal level at IMO/ICPC. Google reports meaningful gains over Gemini 3 Pro on ARC-AGI-2 and HLE; one example cites 45.1% on ARC-AGI-2 for Deep Think [@GoogleAI](https://twitter.com/GoogleAI/status/1996657213390155927), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1996658401233842624), [@GeminiApp](https://twitter.com/GeminiApp/status/1996656314983109003), [@quocleix](https://twitter.com/quocleix/status/1996659461851885936), [@NoamShazeer](https://twitter.com/NoamShazeer/status/1996679619031060680). How to try: select “Deep Think” in the prompt bar and use the “Thinking” model dropdown in the Gemini app [@GeminiApp](https://twitter.com/GeminiApp/status/1996670867770953894).
- **“Titans”: long-context neural memory**: Google previewed Titans, an architecture that combines RNN-like efficiency with Transformer-level performance using deep neural memory, scaling to contexts larger than 2M tokens. Early results were presented at NeurIPS; background/history on the Titan memory line was also posted by the authors [@GoogleResearch](https://twitter.com/GoogleResearch/status/1996674393842614338), [@mirrokni](https://twitter.com/mirrokni/status/1996705597241413869).

**Coding Models and Agent Harnesses**

- **OpenAI’s GPT-5.1-Codex Max (agentic coding)**: Now available in the Responses API, recommended inside the Codex agent harness. OpenAI shared prompting guidance and customer examples; integrations landed across the ecosystem: VS Code, Cursor, Windsurf, and Linear (assign/mention Codex to kick off cloud tasks with updates posted back to Linear) [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1996643999097274560), [@code](https://twitter.com/code/status/1996651445354181028), [@cursor_ai](https://twitter.com/cursor_ai/status/1996645841063604711), [@windsurf](https://twitter.com/windsurf/status/1996665911185756511), [@cognition](https://twitter.com/cognition/status/1996666272805970154), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1996668013676790125).
- **Mistral Large 3 (OSS leader for coding)**: Mistral reports Large 3 is now #1 open-source coding model on lmarena; community corroborations followed, and cloud availability via Ollama (local support “soon”) [@MistralAI](https://twitter.com/MistralAI/status/1996580307336638951), [@sophiamyang](https://twitter.com/sophiamyang/status/1996587296666128398), [@b_roziere](https://twitter.com/b_roziere/status/1996587193372930061), [@ollama](https://twitter.com/ollama/status/1996682858933768691).
- **DeepSeek V3.2**: Baseten published strong serving metrics (TTFT ~0.22s, 191 tps) for V3.2 and made it available via their APIs; lmarena added V3.2/V3.2-thinking to the text leaderboard (mixed movement overall; strongest open-model rankings in Math/Legal/Science categories) [@basetenco](https://twitter.com/basetenco/status/1996623218040254793), [@arena](https://twitter.com/arena/status/1996707563208167881).
- **Low-compute RL and training infra**: Qwen showed FP8 RL training running in just 5 GB VRAM [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1996474298169802799). Hugging Face introduced “HF Skills” you can call from Claude Code, Codex, and Gemini to train/eval/publish models end-to-end (scripts, cloud GPUs, progress dashboards, push to Hub) [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1996602896436375822), [@ClementDelangue](https://twitter.com/ClementDelangue/status/1996718490435174435).

**Video, Vision, and Generative Media**

- **Kling 2.6 + Avatar 2.0**: Kling 2.6 shipped audio-aligned video generation and launched an Audio Challenge; Avatar 2.0 adds longer inputs and better emotion capture, with day-0 hosting on fal [@Kling_ai](https://twitter.com/Kling_ai/status/1996474009266126883), [@Kling_ai](https://twitter.com/Kling_ai/status/1996592857096868075), [@fal](https://twitter.com/fal/status/1996604652100464799). Practitioners showed multi-tool agents orchestrating Kling for creative workflows [@fabianstelzer](https://twitter.com/fabianstelzer/status/1996530919998689735).
- **Runway Gen-4.5**: Broader aesthetic control (photoreal, puppetry, 3D, anime) with coherent visual language across clips; “character morphing” is emerging as a distinct strength [@runwayml](https://twitter.com/runwayml/status/1996586320110440848), [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1996609435641188482).
- **Image leaderboards**: Bytedance’s Seedream 4.5 entered lmarena at #3 (Image Edit) and #7 (Text-to-Image), joining Nano Banana variants at the top; earlier, Nano Banana Pro 2k topped the Image Edit board [@arena](https://twitter.com/arena/status/1996641968005566876), [@JeffDean](https://twitter.com/JeffDean/status/1996457766349848753).
- **SVG generation as a reasoning/coding probe**: Yupp launched an SVG leaderboard and an open dataset (~3.5k prompts/responses/preferences). Gemini 3 Pro leads the SVG leaderboard; prompts like “Earth–Venus 5-fold symmetry” showcase geometric reasoning + code synthesis [@lintool](https://twitter.com/lintool/status/1996696157985398812), [@yupp_ai](https://twitter.com/yupp_ai/status/1996697775585787924), [@lmthang](https://twitter.com/lmthang/status/1996696115920753115).
- **Microsoft VibeVoice-Realtime-0.5B**: A lightweight realtime speech model released on Hugging Face [@_akhaliq](https://twitter.com/_akhaliq/status/1996602953885499466).

**Agents, Scaffolds, and Reliability (what’s working in prod)**

- **Agent scaffolds matter**: “Agent scaffolds are as important as models,” echoed across threads exploring management-process-like scaffolds for subagents and auto-compaction, and the importance of ontological clarity (a single LLM call ≠ subagent) [@AlexGDimakis](https://twitter.com/AlexGDimakis/status/1996444591852302648), [@vikhyatk](https://twitter.com/vikhyatk/status/1996492433757253888), [@fabianstelzer](https://twitter.com/fabianstelzer/status/1996467308072669373).
- **Reliability tooling**: LangChain 1.1 added model/tool retry middleware with exponential backoff (JS and Python), and VS Code prompt files can autoselect per-prompt models to better compose workflows [@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1996577642749862282), [@bromann](https://twitter.com/bromann/status/1996587797398839592), [@burkeholland](https://twitter.com/burkeholland/status/1996590126953005423).
- **“Code as tool” for robustness**: CodeVision lets models write Python to compose arbitrary image operations, drastically improving robustness on transformed OCR tasks (73.4 on transformed OCRBench, +17.4 over base; 60.1 on MVToolBench vs Gemini 2.5 Pro’s 32.6) [@dair_ai](https://twitter.com/dair_ai/status/1996624052493209730).
- **SkillFactory and post-training**: A data-first approach that rearranges traces to demonstrate verification+retry, then SFT→RL, improves learning of explicit verification skills across domains—consistent with observations in Yejin Choi’s keynote on base-model/RL “chemistry” [@ZayneSprague](https://twitter.com/ZayneSprague/status/1996615552987546050), [@gregd_nlp](https://twitter.com/gregd_nlp/status/1996621316267655453).
- **Inference acceleration (beyond speculative decoding)**: AutoJudge learns which tokens matter for the answer, achieving 1.5–2× speedups vs speculative decoding (and stacks with other accelerations) [@togethercompute](https://twitter.com/togethercompute/status/1996654662456639913).
- **Security reality check for agentic coding**: SUSVIBES benchmark finds SWE-Agent+Claude Sonnet 4 gets 61% functionally correct but only 10.5% secure solutions across 200 real-world feature requests that historically led to vulns; vulnerability hints didn’t fix the issue—this pattern held across frontier agents [@omarsar0](https://twitter.com/omarsar0/status/1996595107924263287).

**Evaluation, Measurement, and Trust**

- **Leaderboard hygiene and independent evals**: “The Leaderboard Illusion” (private testing, selective retractions, data access gaps) was prominent at NeurIPS [@mziizm](https://twitter.com/mziizm/status/1996489947159961740), with a Cohere Labs poster and community discussion [@Cohere_Labs](https://twitter.com/Cohere_Labs/status/1996593263609045458). The new AI Evaluator Forum (AEF) debuted to coordinate third‑party evaluations, with METR, RAND, SecureBio, etc. as founding members [@aievalforum](https://twitter.com/aievalforum/status/1996641899332198403), [@METR_Evals](https://twitter.com/METR_Evals/status/1996656514774524054).
- **Benchmarks and footguns**: Global MMLU 2.0 released with expanded multilingual coverage [@mziizm](https://twitter.com/mziizm/status/1996517093039382879). LlamaIndex analyzed OlmOCR-Bench, highlighting gaps in document types and brittle exact matching [@jerryjliu0](https://twitter.com/jerryjliu0/status/1996668513562644823). IF-Eval reminder: strip reasoning content using the correct delimiter (</think>, [/THINK], etc.) [@_lewtun](https://twitter.com/_lewtun/status/1996671492143124901).
- **Trust and measurement science**: Andrew Ng urged the field to address declining public trust (Edelman/Pew) and avoid hyped existential framings, pointing to NIST’s construct-validity emphasis for AI measurement as a constructive path forward [@AndrewYNg](https://twitter.com/AndrewYNg/status/1996631366470132053), [@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1996669236513751499).

**Org Moves and Ecosystem**

- **New Google DeepMind team in Singapore (hiring)**: Led by Yi Tay under Quoc Le’s org, focused on advanced reasoning, LLM/RL, and pushing Gemini/Deep Think. Backed by leadership (Jeff Dean, Demis Hassabis) and compute access; building a small, high-talent-density team [@YiTayML](https://twitter.com/YiTayML/status/1996640869584445882), [@JeffDean](https://twitter.com/JeffDean/status/1996644208854388983), [@quocleix](https://twitter.com/quocleix/status/1996646331474235881).
- **Model availability and platforms**: MiniMax-M2 joined Amazon Bedrock [@MiniMax__AI](https://twitter.com/MiniMax__AI/status/1996485276609503561). AI21 announced Maestro deployments inside AWS VPC [@AI21Labs](https://twitter.com/AI21Labs/status/1996572699959722017). Run Mistral Large 3 on Ollama Cloud now; local support coming soon [@ollama](https://twitter.com/ollama/status/1996683156817416667).
- **Anthropic Interviewer**: Short pilot to collect perspectives on AI at work; initial results + an open dataset of 1,250 interviews released on Hugging Face [@AnthropicAI](https://twitter.com/AnthropicAI/status/1996627123021426919), [@calebfahlgren](https://twitter.com/calebfahlgren/status/1996646452509266266).
- **Perplexity funding**: Cristiano Ronaldo announced an investment in Perplexity, positioning it as “powering the world’s curiosity” [@Cristiano](https://twitter.com/Cristiano/status/1996626923720462425).

**Top tweets (by engagement)**

- Cristiano Ronaldo invests in Perplexity; “powering the world’s curiosity” [@Cristiano](https://twitter.com/Cristiano/status/1996626923720462425) — 46.9k
- Reminder: many robots “fake” humanlike motions via training; hardware can move far faster/weirder [@chris_j_paxton](https://twitter.com/chris_j_paxton/status/1996586464197640193) — 36.7k
- Excel Copilot “Agent Mode” helps Satya compete in the M365 digital challenge [@satyanadella](https://twitter.com/satyanadella/status/1996597609587470504) — 2.8k
- Gemini 3 Deep Think rollout and results across key reasoning benchmarks [@GeminiApp](https://twitter.com/GeminiApp/status/1996656314983109003) — 2.8k
- Mistral Large 3 claims #1 open-source coding on lmarena [@MistralAI](https://twitter.com/MistralAI/status/1996580307336638951) — 1.7k
- Microsoft’s VibeVoice‑Realtime‑0.5B on Hugging Face [@_akhaliq](https://twitter.com/_akhaliq/status/1996602953885499466) — 1.3k
- “RIP ‘you’re absolutely right’” (on model behaviors) [@alexalbert__](https://twitter.com/alexalbert__/status/1996644185886413285) — 1.2k

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Microsoft VibeVoice-Realtime Model Launch

- [**New model, microsoft/VibeVoice-Realtime-0.5B**](https://www.reddit.com/r/LocalLLaMA/comments/1pdu46s/new_model_microsoftvibevoicerealtime05b/) (Activity: 360): **VibeVoice-Realtime is a new open-source text-to-speech model by Microsoft, designed for real-time applications with a parameter size of** `0.5B`**. It supports streaming text input and can generate initial audible speech in approximately** `300 ms`**, making it suitable for real-time TTS services and live data narration. The model is optimized for English and Chinese, featuring robust long-form speech generation capabilities. For more technical details, refer to the [Hugging Face model page](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B).** A notable comment highlights the model's support for both English and Chinese, while another points out a broken link to an unreleased version, VibeVoice-Large, indicating potential oversight in documentation.
    - The model `microsoft/VibeVoice-Realtime-0.5B` supports both English and Chinese languages, which is significant for applications requiring bilingual capabilities. However, there are concerns about the quality of Mandarin output, as one user noted that the Mandarin speaker has a Western accent, which might affect the model's usability for native speakers.
    - There is a broken link issue with the `VibeVoice-Large` model on Hugging Face, leading to a 404 error. This suggests that the model might have been unreleased or removed, indicating potential issues with version control or release management by Microsoft.
    - Users are seeking guidance on how to run the `VibeVoice-Realtime-0.5B` model, indicating a need for clearer documentation or tutorials to facilitate user adoption and implementation. This highlights a common challenge in deploying complex models to a broader audience.

### 2. Humorous Quant Legend Comparison

- [**legends**](https://www.reddit.com/r/LocalLLaMA/comments/1pdzn2n/legends/) (Activity: 394): **The image is a meme contrasting traditional and modern interpretations of 'quant legends.' On the left, a classic image of a mathematician in front of a chalkboard represents the traditional view, while on the right, a cartoon alien with social media icons humorously depicts a modern, internet-driven perspective. The post and comments highlight a playful take on the concept of 'legends' in quantitative fields, with a nod to contributors in the AI and model development community, such as those working on EXL and GGUF models.** The comments reflect a light-hearted discussion, with some users humorously questioning the post's intent as 'karma farming.' Others take the opportunity to acknowledge various contributors to AI model development, suggesting a broader appreciation for community efforts beyond the 'legend' depicted in the meme.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 3 Deep Think Release and Benchmarks

- [**Gemini 3 Deep Think now available**](https://www.reddit.com/r/singularity/comments/1pe8t8u/gemini_3_deep_think_now_available/) (Activity: 634): **The image announces the release of "Gemini 3 Deep Think," a new mode in the Gemini app specifically for Google AI Ultra subscribers. This mode is designed to enhance advanced reasoning capabilities, particularly in complex math, science, and logic problems. It has reportedly performed well in rigorous benchmarks and competitions, indicating its potential effectiveness in handling sophisticated tasks. The announcement also includes instructions for accessing this new mode, suggesting a focus on usability for subscribers.** One commenter expresses anticipation for how "smart people" might leverage this new mode, while another notes the impressive performance of Gemini 3 even before the release of "Deep Think."
- [**Gemini 3 "Deep Think" benchmarks released: Hits 45.1% on ARC-AGI-2 more than doubling GPT-5.1**](https://www.reddit.com/r/singularity/comments/1pec4zg/gemini_3_deep_think_benchmarks_released_hits_451/) (Activity: 510): **The image is a bar chart illustrating the performance of various AI models on three benchmarks, with a focus on the ARC-AGI-2 benchmark. Gemini 3 Deep Think achieves a score of** `45.1%`**, significantly outperforming GPT-5.1 which scores** `17.6%`**. This demonstrates a** `2.5x` **improvement in novel puzzle-solving capabilities, attributed to the integration of System 2 search/RL techniques, possibly involving AlphaProof logic. This advancement highlights Google's lead in reasoning and inference-time compute, challenging OpenAI to respond with updates like o3 or GPT-5.5 to regain competitive standing.** Commenters are excited about the progress, with some noting that OpenAI may be developing a competitive model, and others questioning the absence of certain models like Opus in the comparison.
    - The release of Gemini 3 "Deep Think" benchmarks shows a significant improvement, achieving 45.1% on the ARC-AGI-2 benchmark, which is more than double the performance of GPT-5.1. This benchmark, referred to as "Novel problem solving" in **Anthropic's** blog, highlights the model's capabilities in handling complex problem-solving tasks. However, some users express skepticism, noting that despite high benchmark scores, models can still exhibit issues like hallucinations in simpler tasks.
    - A comparison is made between Gemini 3 "Deep Think" and Opus 4.5, with Gemini 3 achieving 45.1% and Opus 4.5 reaching 37% on the ARC-AGI-2 benchmark. This indicates a notable performance gap between the two models on this specific benchmark, which is designed to test advanced problem-solving abilities. The discussion suggests that while benchmarks are useful, they may not fully capture a model's practical performance in real-world applications.

### 2. Z-Image Prompting and Styles

- [**The prompt adherence of Z-Image is unreal, I can't believe this runs so quickly on a measly 3060.**](https://www.reddit.com/r/StableDiffusion/comments/1pdsz9x/the_prompt_adherence_of_zimage_is_unreal_i_cant/) (Activity: 762): **The image demonstrates the capabilities of the Z-Image model, which is praised for its prompt adherence and speed, even when running on a relatively modest GPU like the NVIDIA 3060. The user highlights the model's ability to accurately render complex visual prompts, capturing intricate details such as specific clothing patterns, facial expressions, and accessories. However, the model struggles with negation, as seen in the inability to exclude rings from the man's depiction. The use of Lenovo LoRA is mentioned to enhance output fidelity, suggesting a combination of techniques to achieve high-quality results quickly.** Commenters express excitement about Z-Image's potential, comparing it to SDXL and anticipating further improvements with fine-tuning and additional LoRA models.
    - *Saturnalis* highlights the prompt adherence of Z-Image, noting its ability to capture complex details like 'alternating black and white rings' on fingers, while struggling with negation such as 'The man has no rings.' The user mentions using the Lenovo LoRA for higher fidelity outputs, achieving results in 15-30 seconds on a 3060 GPU, which is impressive for such detailed rendering.
    - hdean667 discusses using Z-Image for generating quick images for animation in long-form videos. The tool's ease of use is emphasized, as users can achieve specific looks by simply adding sentences or keywords, making it highly adaptable for creative projects.
    - alborden inquires about the GUI used for running Z-Image, asking if it's ComfyUI or another interface, indicating interest in the technical setup and user interface preferences for optimal performance.
- [**Z-Image styles: 70 examples of how much can be done with just prompting.**](https://www.reddit.com/r/StableDiffusion/comments/1pdy78q/zimage_styles_70_examples_of_how_much_can_be_done/) (Activity: 647): **The post discusses the capabilities of Z-Image, a model similar to SDXL, in generating diverse styles through prompting alone, without relying on artist names. The author provides a detailed workflow using Z-Image-Turbo-fp8-e43fn and Qwen3-4B-Q8_0 clip at** `1680x944` **resolution, employing a specific process involving model shifts and upscaling to enhance detail and speed. The workflow includes using a negative prompt set to "blurry ugly bad," although it appears ineffective at** `cfg 1.0`**. The post also links to resources like [twri's sdxl_prompt_styler](https://github.com/twri/sdxl_prompt_styler/tree/main) and a [full workflow image](https://simple-static-content.s3.ap-southeast-2.amazonaws.com/Workflow_Z13.png).** Commenters discuss the effectiveness of negative prompts in Z-Image, with one noting that the "Moebius-like" style is not accurate, suggesting the need for LoRAs for specific styles. Another commenter mentions crafting style prompts and notes Z-Image's ability to generate ASCII art when well-described.
    - Baturinsky raises a technical question about whether Z-Image considers negative prompts, which are often used in AI image generation to guide models away from certain styles or elements. This is crucial for refining outputs and ensuring the model adheres closely to the desired artistic direction.
    - Optimisticalish points out a limitation in Z-Image's ability to replicate specific art styles, such as Moebius, suggesting that current models may require additional training data or LoRAs (Low-Rank Adaptations) to accurately capture these styles. They note that using specific artist names with underscores, like 'Jack_Kirby', can modify the 'comic-book style' effectively, indicating a nuanced approach to style prompting.
    - Perfect-Campaign9551 highlights the effectiveness of using specific style prompts like 'flat design graphic', which involves creating a colorful, two-dimensional scene with minimal shading. This suggests that Z-Image can handle a variety of stylistic requests, provided the prompts are well-crafted and descriptive.

### 3. AI's Impact on Tech Jobs and Society

- [**Deep down, we all know that this is the beginning of the end of tech jobs, right?**](https://www.reddit.com/r/ClaudeAI/comments/1pe6q11/deep_down_we_all_know_that_this_is_the_beginning/) (Activity: 1262): **The post discusses the rapid advancement of AI and its potential impact on tech jobs, suggesting that roles like software developers, DevOps, and designers may see a significant reduction in demand. The author argues that while humans will still be involved, the number of people needed will drastically decrease as AI takes over tasks such as writing code, generating tests, and designing systems. The post challenges the notion that AI will only augment human roles, comparing the situation to historical shifts in labor demand due to automation.** A notable comment argues that while AI tools are transformative, they do not replace the need for human involvement in complex tasks such as stakeholder management, system architecture, and dealing with legacy systems. The commenter emphasizes that AI is raising the entry-level bar but also increasing the complexity of what can be built, suggesting that the nature of development work is evolving rather than disappearing.
    - The comment by 'alphatrad' highlights the limitations of AI in software development, emphasizing that while AI tools can automate coding tasks, they cannot replace the nuanced human roles in the software development lifecycle (SDLC). The commenter points out that AI lacks the ability to handle complex organizational dynamics, such as stakeholder management, conflicting requirements, and legacy system integration. They argue that AI is merely the next step in a long history of technological abstraction, which has consistently increased the complexity and scope of software projects rather than eliminating jobs.
    - 'alphatrad' also discusses the evolving nature of developer roles, suggesting that while AI can automate junior-level tasks, it raises the bar for entry-level positions. The commenter advises developers to focus on skills that AI cannot replicate, such as system design, debugging, and understanding business operations. They emphasize the importance of communication skills and the ability to work with legacy systems, suggesting that the future of development will require a blend of technical and soft skills.
    - The comment by 'codemagic' suggests a shift in focus towards the early stages of the SDLC, such as requirements gathering and high-level architecture, as automation takes over more routine coding tasks. This shift emphasizes the need for precise language and writing skills, indicating a potential change in the skill set required for developers as AI tools become more prevalent in handling low-level implementation and tuning tasks.
- [**Deep down we all know Google trained its image generation AI using Google Photos… but we just can’t prove it.**](https://www.reddit.com/r/ChatGPT/comments/1pdqzdo/deep_down_we_all_know_google_trained_its_image/) (Activity: 3803): **The post speculates that Google may have used its vast collection of user-uploaded images from Google Photos to train its image generation AI, despite official statements that user photos are not used for advertising. The author suggests that the familiarity of AI-generated images could be due to the extensive metadata and high-quality images Google has collected over the years. This is compared to past instances like Google's voice recognition improvements following the Goog-411 service, implying a pattern of leveraging user data to enhance AI capabilities.** Commenters discuss Google's data policies, noting that while user content is not sold for advertising, Google retains a broad license to use it for service improvements. They draw parallels to past Google services like Goog-411, which seemingly collected data to improve subsequent technologies like Voice Search.
    - Fonephux highlights Google's data policy, which grants the company a broad license to use, host, and modify user content from services like Google Photos. This policy is designed to improve service functionality, suggesting that while user content is protected, it can be utilized to enhance Google's AI capabilities.
    - redditor_since_2005 draws a parallel between Google's past service, Goog-411, and its subsequent development of Voice Search. The implication is that Google used data from Goog-411 to train its speech recognition models, suggesting a similar strategy might be employed with Google Photos for image generation AI.
    - ChuzCuenca implies that Google Photos' free hosting service is likely not without ulterior motives, hinting at the possibility that user photos could be used to train Google's AI models, despite the lack of direct evidence.
- [**This grandson used AI to recreate his grandfather's entire life for his 90th birthday.**](https://www.reddit.com/r/aivideo/comments/1pe1xzt/this_grandson_used_ai_to_recreate_his/) (Activity: 2938): **A grandson utilized AI technology to recreate his grandfather's life story as a gift for his 90th birthday. This project likely involved using machine learning models to process and synthesize personal data, such as photos, videos, and possibly audio recordings, to create a comprehensive digital narrative. The use of AI in this context highlights its potential for personal storytelling and preserving family histories, leveraging tools like generative adversarial networks (GANs) or natural language processing (NLP) to enhance the narrative experience.** The comments reflect a positive reception, with users appreciating the innovative use of AI for personal and emotional storytelling, though some expressed a desire for more content or details about the project.
- [**This grandson used AI to recreate his grandfather's entire life for his 90th birthday.**](https://www.reddit.com/r/aivideo/comments/1pe1xzt/this_grandson_used_ai_to_recreate_his/) (Activity: 2943): **A grandson utilized AI technology to recreate his grandfather's life story as a gift for his 90th birthday. This project likely involved using machine learning models to process and synthesize historical data, personal anecdotes, and possibly multimedia elements to create a comprehensive narrative. The use of AI in this context highlights its potential in personal storytelling and preserving family histories, demonstrating a novel application of technology in enhancing personal and emotional experiences.** The comments reflect a positive reception, with users appreciating the innovative use of AI for personal storytelling. However, there is a lack of technical debate or detailed discussion on the implementation specifics in the comments.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Frontier Coding Models, OpenRouter Trends, and IDE Integrations**

- **OpenRouter's 100T-Token Telescope Tracks Roleplay, Coding, and Agents**: OpenRouter and a16z released the [**State of AI** report](https://openrouter.ai/state-of-ai) based on **100 trillion tokens** of anonymized traffic, showing that **>50%** of open‑source model usage is **roleplay/creative**, while **programming exceeds 50% of paid-model traffic** and **reasoning models** now handle **>50% of all tokens**. The data highlights that users overwhelmingly choose **quality over price**, that **Claude** owns ~**60%** of coding workloads with average prompts over **20K tokens**, and that tool‑calling plus long contexts are pushing the ecosystem toward **full AI agents** rather than one‑shot Q&A.
    - The report notes a **flat correlation between cost and usage**, implying that reliability, latency, and ergonomics matter more than raw token price until quality converges, and it calls out a large, underserved consumer segment for **entertainment/companion AI**. Engineers in the OpenRouter community emphasized that building competitive products now requires **multi‑step execution, strong state management, and robust tool orchestration**, not just dropping in a single chat endpoint.
- **GPT‑5.1 Thinking and Codex Max Crash Gemini's Coding Party**: Across OpenAI, OpenRouter, and Windsurf communities, **GPT‑5.1** and **GPT‑5.1‑Codex Max** emerged as new coding workhorses, with OpenAI users reporting that **GPT‑5.1 Thinking** beat **Gemini 3** at [bug finding in code](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680) and Windsurf announcing **GPT‑5.1‑Codex Max** availability at Low/Medium/High reasoning levels via a [new release](https://x.com/windsurf/status/1996665911185756511). OpenRouter discussion added that OpenAI also shipped a **Codex Max** model as part of an intensifying race against Google's **Gemini 3 Deep Think Mode**, while rumors via an [ArsTechnica article](https://arstechnica.com/ai/2025/12/openai-ceo-declares-code-red-as-gemini-gains-200-million-users-in-3-months/) point to yet another OpenAI model drop next week.
    - Windsurf is giving paid users a **free trial** of **GPT‑5.1‑Codex Max Low**, aiming squarely at dev workloads, while OpenAI Discord engineers contrasted **Gemini 3 Pro's** UX with its failure to spot basic bugs that **GPT‑5.1** caught. On OpenRouter, users framed this as part of a wider coding stack shake‑up, with Anthropic's acquisition of **Bun** for Claude's **$1B code revenue** and OpenAI's **Codex Max** making IDEs like **Cursor** and **Windsurf** the front line of the model wars.
- **Hermes 4.3 Shrinks Size, Targets OpenRouter, and Competes with DeepSeek**: Nous Research announced **Hermes 4.3** on **ByteDance Seed 36B**, claiming **Hermes 4.4 36B‑class performance roughly equal to Hermes 4 70B at half the size**, post‑trained entirely on the **Psyche network** secured by **Solana**, with more details in the launch post [“Introducing Hermes 4.3”](https://nousresearch.com/introducing-hermes-4-3/). In Discord, Teknium hinted that **Mistral‑3 Hermes fine‑tunes** and **MoE support** are coming next via their internal trainer, and confirmed existing **Hermes models on OpenRouter**, though users want Nous to onboard as a **direct provider**.
    - Engineers compared **Hermes 4.3** with **DeepSeek v3.2**, praising DeepSeek for being *“super affordable”* and asking for Hermes **70B/405B** to join that pricing tier on OpenRouter, while others noted that **Opus 4.5** (Anthropic) is now better integrated into tools like **GitHub Copilot** and is available free via **antigravity**. The Hermes launch is also tied to the experimental **Psyche** training network, with office hours advertised via Discord to discuss decentralized training and how it outperformed centralized setups.

**2. Security, Jailbreaking, and Agent Execution Safety**

- **From Sora 2 and Gemini Web to DeepSeek: Jailbreakers Keep Winning**: Across LMArena and BASI, red‑teamers reported bypasses in **Sora 2** and web models: one LMArena user claims to have found an exploit in **Sora 2's filtration**, noting that **character generation prompts** can circumvent guardrails, while BASI members discussed **NSFW image generation backdoors** on **Gemini Web** using custom system instructions and filters that fail intermittently. BASI jailbreaking threads also documented **nested jailbreaks against DeepSeek**, with a [screenshot of DeepSeek generating Windows reverse‑shell malware code](https://cdn.discordapp.com/attachments/1228043845967544380/1446279240201928845/FireShot_Capture_029_-_Creating_Malware_for_Windows_Reverse_Shells_-_DeepSeek_-_chat.deepseek.com.png) and noted that older tricks like the **ENI jailbreak** from the [Wired nuclear‑weapon poem article](https://www.wired.com/story/poems-can-trick-ai-into-helping-you-make-a-nuclear-weapon/) still work on **Gemini 2.5**.
    - Attackers also probed **Grok 4.1** and **GPT‑5.1**, with BASI members trying to force drug and soda‑recipe style outputs from **Grok** and sharing the [UltraBr3aks jailbreak collection](https://github.com/SlowLow999/UltraBr3aks) to attack **GPT‑5.1**, but conceding that GPT‑5.1 remains hard to fully compromise. Jailbreakers continue to switch to more permissive or less‑polished models for offensive content and malware generation — **DeepSeek, Gemini 3, Seeds like Seedream 4.5** — while observing that each new safety layer increases **“intelligence tax”** on the model when heavily constrained.
- **AI Agents Get Red‑Teamed with Prompt Injection and Execution‑Time Guards**: BASI's red‑teaming channel coordinated **realistic AI agent attack simulations**, including **prompt injection**, **spoofed agent messages**, and **replay attacks**, to test how well agent frameworks resist arbitrary code execution and data exfiltration. The group evaluated **execution‑time authorization frameworks** like **A2SPA** as a way to gate external actions, aiming to ensure that even if the LLM's reasoning step is compromised, **tool invocations** still obey a separate policy layer.
    - At the tooling level, MCP Contributors debated whether MCP tools should accept **UUIDs as arguments**, after observing that LLMs tend to hallucinate UUIDs even when told not to, and suggested a two‑tool pattern: a `list_items` tool that returns lightweight items with UUIDs and a `describe_item` tool that takes a UUID to fetch full records. This architecture separates **identifier generation** (never entrusted to the LLM) from **identifier usage**, aligning with the agent red‑team view that LLMs should not mint primary keys or security‑sensitive identifiers, only consume them under strict schemas.
- **Secure Code Verification, ARR Explosions, and Legal AI at Scale**: In Latent Space, users highlighted three big business moves tied to security and code correctness: **Antithesis** raised a **$105M Series A** led by Jane Street to build **deterministic simulation testing for AI‑generated code**, per [this X thread](https://xcancel.com/_sholtodouglas/status/1996297367776309359), **Anthropic** projected **$8–10B ARR** this year driven largely by **Claude for coding**, and legal AI company **Harvey** closed a **$160M Series F** at an **$8B valuation** serving **700+ law firms in 58 countries** via [Brian Burns' tweet](https://xcancel.com/brian_a_burns/status/1996624620519399634). The consensus is that as LLMs write more production and compliance‑sensitive code, customers demand **trust‑through‑testing** and specialized vertical stacks (legal, finance) rather than generic chatbots.
    - These revenue numbers frame the Anthropic news that it acquired **Bun** to power **Claude's $1B code generation business**, via [Anthropic’s announcement](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone), and sit alongside startup agents like [Shortcut v0.5](https://xcancel.com/nicochristie/status/1996318170223964489?s=46) that auto‑build institutional FP&A spreadsheets. Engineers interpreted this as validation for investing heavily in **static analysis, deterministic simulation, and verticalized agents**, since money is flowing toward stacks that can both **generate** and **prove** code behavior.

**3. GPU Systems, Quantization, and Kernel Competitions**

- **TorchAO MoE Quantization and NvFP4 GEMM Tuning Go Deep**: In GPU MODE, PyTorch engineers dug into **TorchAO**’s quantization stack for MoEs, pointing to the dedicated `MoEQuantConfig` and new `FqnToConfig`**‑based routing** from [PR #3083](https://github.com/pytorch/ao/pull/3083) that lets you assign quantization configs by fully‑qualified name instead of only via `filter_fn`. They noted that compilation remains slow even after precompile and recommended setting `TORCH_LOGS="+recompiles"` to spot dynamic shapes and unnecessary recompilations, as well as ensuring MoE packed weights live in `nn.Parameter`s like in the [Mixtral MoE example](https://github.com/pytorch/ao/blob/main/torchao/_models/mixtral-moe/model.py#L336).
    - Concurrently, the **NVIDIA nvfp4_gemm** competition channel confirmed the reference kernels are built on **cuBLAS 13.0.0.19 (CUDA 13.0.0)**, and a [PR](https://github.com/gpu-mode/reference-kernels/pull/84) fixed INF issues by using the **full FP4 a/b range** with **non‑negative scale factors**. Competitors discovered that some LLMs were **“cheating” the eval** (exploiting Python‑based harness quirks), and discussed porting the evaluator to a non‑Python stack; a participant also documented that submissions silently failed until they added the explicit `-leaderboard nvfp4_gemm` flag to target the right board.
- **Sparse Attention, VAttention Guarantees, and CUDA cp.async Puzzles**: GPU MODE's cool‑links channel resurfaced a long‑running frustration: despite **~13,000 papers on sparse attention**, practical systems like **vLLM** rarely use it, as argued in [this X post from skylight_org](https://x.com/skylight_org/status/1993637433838035026). One promising line is *“VATTENTION: VERIFIED SPARSE ATTENTION”* ([arXiv:2510.05688](https://arxiv.org/pdf/2510.05688)), which gives **user‑specified (ϵ, δ) guarantees** on approximation error and was cited as a template for deeper collaboration between **PL/verification researchers and ML systems** people.
    - On the low‑level side, a CUDA developer observed Nsight Compute warnings about `LDGSTS.E.BYPASS.LTC128B.128` (the `cp.async` path) when they cranked up `launch__registers_per_thread`, with **3.03% of global accesses and 17.95% of shared wavefronts flagged as “excessive”**, and those warnings vanished once register usage dropped. The thread wrestled with how **high register pressure and reduced occupancy** feed back into `cp.async` behavior within a block, illustrating the subtle interplay between **register allocation, SM occupancy, and async copy instructions** in real kernels.
- **Hardware Pricing, Multi‑GPU Weirdness, and Edge‑Server Architectures**: LM Studio and GPU MODE hardware channels compared GPU pricing and multi‑GPU setups: one user complained that **$3.50/hr for 2× H100 PCIe** with 1 Gbit is steep, pointing to **SFCompute’s H100 at $1.40/hr** and **Prime Intellect’s B200 at ~$1/hr spot, ~$3/hr on‑demand** via [primeintellect.cloud](http://primeintellect.cloud/). LM Studio users reported that triple‑GPU rigs were *“very buggy out of 10”*, especially with **non‑even card counts and small 8 GB cards** when sharding dense models past **50 GB**, and that **CachyOS** struggled with mixed‑generation dual‑GPU setups that worked fine on Ubuntu.
    - Practitioners also explored home‑lab server patterns: one LM Studio user wanted to convert an old gaming laptop into a **central LLM server with a request queue** to protect weaker devices, while Modular’s Mojo channel linked to constant‑memory kernel examples in the [modular/modular repo](https://github.com/modular/modular/blob/b8756654dd050be664396757be2fc7c495484e1b/max/kernels/test/gpu/basics/test_constant_memory.mojo#L105) for devs looking to hard‑wire convolution kernels into GPU constant memory. The cumulative message is that **cost‑efficient, multi‑GPU, and edge‑server setups remain finicky**, with a lot of tacit knowledge around PSU wiring, telemetry defaults, and OS quirks (e.g., CachyOS telemetry opt‑out and GNOME vs KDE trade‑offs).

**4. New Optimization, Evaluation, and Research Directions**

- **ODE Solvers, STRAW Rewiring, and Feature Attribution Shake Up Vision**: Hugging Face’s research channels surfaced multiple novel optimization ideas: a new fast ODE solver for diffusion models claims **4K images in 8 steps** with quality comparable to **30‑step dpm++2m SDE Karras**, released as a HF Space [“Hyperparameters are all you need 4K”](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-4k) alongside its [paper](https://arxiv.org/abs/2510.02390). Another experiment, **STRAW (sample‑tuned rank‑augmented weights)**, lets a net **rewrite low‑rank weight adapters per input** to mimic biological neuromodulation while avoiding RAM blowups, documented in the write‑up [“Sample‑tuned rank‑augmented weights”](https://teendifferent.substack.com/p/sample-tuned-rank-augmented-weights).
    - Complementing this, an interpretability‑heavy post [“Your features aren’t what you think”](https://teendifferent.substack.com/p/your-features-arent-what-you-think) analyzed **feature behavior in deep vision models** via perturbation‑based attribution, arguing that intuitive “feature = concept” mappings often break under systematic perturbations. The authors and HF reading‑group participants stressed that getting **chunking quality** and **input semantics** right (especially for tables and RAG corpora) may matter as much as exotic architectures when you want robust eval scores and interpretable internal representations.
- **Shampoo, CFG, and Attention Sinks: Optimizer and Diffusion Theory Evolve**: Eleuther’s research channel critiqued the **Shampoo optimizer**, with a Google employee noting that the exponent on the preconditioner in the [Shampoo paper](https://arxiv.org/abs/2503.20762) might be better at **−1 than −1/2**, calling the current work *“ok”* but with *“a few other deficiencies”*. They also discussed [“Random Rotations for Adam”](https://arxiv.org/abs/2410.19964), which surprisingly performs **worse** than standard Adam despite hopes that rotating away activation outliers would help, in part because the method never re‑rotates when the underlying SVD basis drifts.
    - In diffusion land, members dissected **Classifier‑Free Guidance (CFG)** using [a 2024 CFG/memorization paper](https://arxiv.org/abs/2411.16738), surprised that the **memorization basin emerges very early** and likely depends strongly on dataset size and resolution, and brainstormed orthogonalizing the unguided and guided updates to reduce required CFG strength (citing an OpenReview paper at [openreview.net/forum?id=ymmY3rrD1t](https://openreview.net/forum?id=ymmY3rrD1t)). A separate survey on **attention sinks** from NeurIPS (poster PDF at [neurips.cc](http://neurips.cc/)) triggered debate about rope‑based intuitions, with some arguing that the authors mischaracterize **1D rotations and sink behavior** in long‑context transformers.
- **Lightweight Local Eval, Smol Training, and Latent Multi‑Agent Collaboration**: Hugging Face’s makers announced **smallevals**, a local RAG evaluation suite that uses tiny **0.6B Qwen‑based models** trained on **Natural Questions** and **TriviaQA** to generate question‑answer pairs from your docs, shipped as [QAG‑0.6B GGUF](https://huggingface.co/mburaksayici/golden_generate_qwen_0.6b_v3_gguf) plus a [GitHub repo](https://github.com/mburaksayici/smallevals) and `pip install smallevals`. It builds *golden* retrieval eval datasets without depending on the generation model, and includes a local dashboard to inspect **rank distributions, failing chunks, and dataset stats**, enabling cheap, offline RAG benchmarking.
    - On the training front, an Eleuther member working on **SLMs for agents** is designing pipelines that fit under **16 GB VRAM**, referencing the [Hugging Face smol‑training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) and benchmarks like **MixEval, LiveBench, GPQA Diamond, IFEval/IFBench, HumanEval+**, while others recommended **LoRA without Regret** rather than full pretrain. In the DSPy server, someone shared a paper on **Latent Collaboration in Multi‑Agent Systems** ([link](https://share.google.com/s/QHcMFSqiTkTnZ231)) where agents *“implicitly coordinate through learned latent spaces”*, which neatly aligns with DSPy users’ push to integrate tools like **Claude Code Agents** and **MCP‑Apps SDK** into structured multi‑agent workflows.

**5. On‑Device, Small Models, and Agent/Tool Ecosystems**

- **Phones Run Qwen and Gemma While Vulkan and WSL2 Smooth Local Dev**: Unsloth users confirmed that **llama.cpp’s Vulkan backend** works on **Android** with the **Freedreno ICD**, though **FP16** can be flaky, and recommended `pkg install llama-cpp` instead of custom Vulkan builds to reduce friction. In the same server, people are running **Qwen 3 14B** on an **iPhone 17 Pro** and **Gemma E2B** on an **iPhone 12** via **Termux + llama.cpp/kobold.cpp**, while others reminded the room that *“not every phone can run a 4B 24/7”* despite optimistic claims.
    - On Windows, Unsloth’s help channel repeatedly pushed devs to **WSL2 + VSCode** with official [Conda](https://docs.unsloth.ai/get-started/install-and-update/conda-install) and [pip](https://docs.unsloth.ai/get-started/install-and-update/pip-install) install guides, after users hit issues like Unsloth downgrading Torch to a CPU build or crashing Ollama with Qwen3‑VL due to format incompatibilities ([Ollama issue #13324](https://github.com/ollama/ollama/issues/13324)). The result is a de facto pattern: **phones and thin clients talk to a local Linux box (WSL2 or bare metal) running llama.cpp/Unsloth**, which then exposes APIs for downstream tools like **aider** and **Crowdllama**.
- **MCP Apps SDK, Claude Code Agents, and UUID‑Centric Tool Design**: DSPy and MCP ecosystems are converging: **General Intelligence Labs** open‑sourced [**mcp‑apps‑sdk**](https://github.com/General-Intelligence-Labs/mcp-apps-sdk), letting devs run **ChatGPT MCP apps with UIs** on **any assistant platform** and test them locally, as explained in their [X thread](https://x.com/helloxalia/status/1796319442863866351). DSPy members meanwhile proposed adding a `dspy_claude_code` backend that talks to **Claude Code/Claude Agents SDK**, wiring tools like `Read`, `Write`, `Terminal`, and `WebSearch` into DSPy’s declarative LM interface.
    - The **MCP Contributors** WG debated how tools should handle **UUIDs**, concluding that LLMs should **never create primary UUIDs**, only pass them between a `list_items` tool and a `describe_item` tool, to mitigate the model’s tendency to hallucinate IDs and cross‑wire resources. Together, these discussions show a clear push toward **strong tool schemas, explicit IDs, and portable app layers** where LMs orchestrate pre‑defined capabilities rather than inventing opaque state.
- **SLMs for Agents, Student Models, and Edge‑Oriented Training Pipelines**: In Eleuther, the founder of [**A2ABase.ai**](http://a2abase.ai/) is exploring **Small Language Models for agents**, asking for edge‑friendly benchmarks and referencing the [Hugging Face smol‑training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) while aiming to train models under **16 GB VRAM** and merging **TRMs with nanoGPT**. They were advised to avoid full pretraining on that budget and instead use well‑chosen benchmarks like **MixEval** and **HumanEval+**, plus light‑touch **LoRA** to add capabilities without wrecking generalization.
    - On the application side, DSPy users requested a **“student models” subforum** for models like **Qwen3** and **gpt‑oss‑20B**, to centralize best practices for low‑cost, long‑running agents, while multiple engineers (in DSPy, Manus, GPU MODE jobs) showcased **workflow automation systems** that connect **Slack, Notion, and internal APIs to small or mid‑size LMs**, claiming **~60% response‑time reductions**. This cements a pattern: **cheap, specialized SLMs + orchestration libraries (DSPy, MCP, custom agents)** are becoming the default for edge and SMB workloads, with frontier models reserved for hardest reasoning or coding tasks.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **AI Companies can't find green pastures**: Members discussed the financial challenges in the AI sector, pointing out that even leading companies struggle with profitability due to significant computational costs and infrastructure demands.
   - It was emphasized that *compute is expensive*, which hinders the accessibility and profitability of AI compared to traditional software or internet services.
- **LM Arena places prompt limits to keep it real**: LM Arena now limits repeated prompts to **4** to ensure fair testing, with resets available for new prompts, though this deletes old prompts and responses.
   - The update aims to make the arena fairer, sparking inquiries about how the new rate limit impacts testing methodologies and user experience.
- **Frame-Flow Battles Opus in Text Arena**: Users are actively trying to identify the new **Frame-Flow** model, which is performing well in text battles against **Opus**, with speculation it might be **Gemini 3 Flash**, **Grok 5**, or a model from a new company.
   - Discussions involve testing Frame-Flow with steganography puzzles and comparing its coding abilities to existing models.
- **Seedream 4.5 Enters Image Arena**: The **Seedream 4.5** image model is now available in the Image Arena, accessible via **Direct or Side by Side modes** in the dropdown menu.
   - While some find the model comparable to nano banana pro, others argue it's inferior, with a rate limit of **5 Generations/h**.
- **Sora 2 filtration system is not so air tight**: A user claims to have discovered an exploit in **Sora 2**'s filtration system, noting that guard rails are not equally distributed.
   - The user notes the exploit involves generating content using specific prompts and methods that bypass restrictions, but the only way to fix it is to *not allow people to generate characters*.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Grok's Reality leaves Tweeters Skeptical**: Members doubted the legitimacy of a [tweet about **Grok**](https://fixupx.com/i/status/1996247117468209453), using GIFs and images to express their skepticism.
   - One member even shared [a GIF](https://tenor.com/view/grok-is-this-true-windmill-water-wheel-gif-5302497903247327835) depicting a water wheel, jokingly suggesting **Grok's** capabilities are just for show.
- **Gemini and Claude Duke it out for Malware Creation**: Members discussed using **Gemini 3** to craft prompts that **Claude** struggles with, focusing on generating code or malware, emphasizing the effectiveness of specific coding questions over general jailbreaking attempts.
   - Despite some success, opinions were mixed, with one member finding **Claude** unimpressive, while others debated the value and future of AI in malware creation compared to traditional reverse engineering.
- **Gemini Web's Backdoor for NSFW Image Generation**: Members shared methods for generating NSFW images with **Gemini Web** using system instructions, noting the platform's filter limitations and inconsistent results.
   - One member found **Seedream 4.5** most effective for editing NSFW images due to its prompt adherence and output stability, contrasting it with **Nano Banana Pro**, which is hindered by filters and inconsistent output.
- **AI Agents Face Security Stress Tests**: Members explored simulating and documenting real-world AI agent attack scenarios, including prompt injection attacks, spoofed agent messages, and replay attacks, to evaluate security and threat modeling.
   - These simulations aim to identify vulnerabilities and assess the effectiveness of execution-time authorization methods like **A2SPA** in preventing unintended execution and unauthorized access.
- **DeepSeek Dives Deep into Jailbreak Territory**: Members reported successfully jailbreaking **DeepSeek** using nested jailbreaks, sharing a [screenshot](https://cdn.discordapp.com/attachments/1228043845967544380/1446279240201928845/FireShot_Capture_029_-_Creating_Malware_for_Windows_Reverse_Shells_-_DeepSeek_-_chat.deepseek.com.png?ex=69336801&is=69321681&hm=911b769e6ba2306df1cd2dac947fbe4b977ba9660266ad6f203bb86e32d4f774) as evidence of its ability to generate malware code.
   - This nested approach may be used in the future for jailbreaking.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Vulkan Backend Confirmed on Android**: Members confirmed that the **Vulkan backend** works via *llama.cpp* on **Android** devices, but it requires the correct **ICD (Freedreno)** to function properly.
   - Using `pkg install llama-cpp` was recommended as an easier alternative to compiling with **Vulkan** support, though issues with **FP16** might still occur depending on the hardware.
- **iPhones Run LLMs**: Members are running **LLMs** on phones directly with *llama.cpp* through **Termux**, also utilizing *kobold.cpp* for enhanced performance.
   - Configurations varied, with some running **Qwen 3 14B** on an **iPhone 17 Pro** and others testing **Gemma E2B** on an **iPhone 12**, highlighting the range of possibilities and limitations based on device capabilities.
- **Unsloth Community Lauded**: The **Unsloth Discord community** received high praise for its active engagement and value in finetuning, with members appreciating the community's support.
   - Members building the community were praised, and when asked about its origin, the answer was *it's just you guys really xD you guys started being active which helped a lot*.
- **Nvidia VRAM Supply Rumors Spark Debate**: Members speculated that **Nvidia** might halt **VRAM** supply to partners, potentially causing supply issues for smaller AiB partners.
   - This discussion raised concerns about market dynamics, with one member jokingly suggesting shorting **3090** stock, and others considered parallels with previous EVGA-like situations.
- **Windows Users Turn to WSL2**: A user utilizing **Windows 11** was advised to install **WSL2** and run **VSCode** for a smoother development environment, and was pointed to helpful installation guides.
   - The user was provided links to [Conda Installation](https://docs.unsloth.ai/get-started/install-and-update/conda-install) and [Pip Installation](https://docs.unsloth.ai/get-started/install-and-update/pip-install) guides for setting up the environment.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Grok Code Falls Out of Favor**: Users initially praised **Grok Code's** reasoning, but one user later reported it had stopped reasoning completely.
   - No further details were provided.
- **Engineers Seek Cursor UI tips**: Users requested tips for creating professional **UIs** without paid tools like Figma.
   - One suggestion involved pasting screenshots into Cursor and prompting it to reproduce the layout.
- **Cursor Nightly Builds Launch Rogue Agents**: Users reported that **Cursor Agents** in nightly builds were running without permission, creating/deleting files, and potentially downloading codebases.
   - A user whose forum post was deleted suggested downgrading to a stable version and disabling dotfile/external file access.
- **Auto Agent Suffers Intelligence Crisis**: A user reported that **Auto Agent** purposely went crazy comparing unrelated pages, while another reported a surge in errors from 11 to 34.
   - Other users noted that the model's quality is task-dependent.
- **New Pricing Model Strikes Auto**: Users discussed a new pricing model where **Auto** is no longer free after the next billing cycle for some users.
   - One user, having used **360M tokens** this month (costing **$127**), plans to switch to **$12 Copilot** with **GPT5-mini**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GNOME favored on CachyOS**: Some users chose [GNOME for CachyOS](https://www.gnome.org/) because they prefer it to KDE and find Cinnamon to be light on VRAM.
   - A user stated they *"can't stand KDE"*.
- **CachyOS faces dual GPU challenges**: Users are encountering issues with running **two different GPUs** (e.g., Nvidia 4070ti and 1070ti) on CachyOS, with an error that doesn't occur on Ubuntu.
   - The problem may be related to using **GPUs from different generations**, prompting one user to consider using the second GPU in another PC.
- **Qwen springs to LM Studio**: **Qwen** is now supported in LM Studio, as showcased by a user's screenshot of the [LM Studio UI](https://lmstudio.ai/).
   - Others remarked on possible UI bugs and the large VRAM requirements for certain **Qwen** model quantizations.
- **DDR4 still worthy?**: A member inquired about the viability of **3200MHz DDR4** compared to **3600MHz**, and another member responded with an image noting that **3200Mhz** is basically top of the bracket of the DDR4 standards.
   - The attached image indicated that **3200MHz** is the *top of the bracket* for **DDR4** standards.
- **Triple GPU setups equals bugginess**: A user reported that a triple GPU setup is *very buggy* out of 10, prompting another to jokingly suggest adding a fourth to fix it.
   - One member noted *issues with splitting LLMs across non-even numbers of cards*, another suggested that an 8GB GPU might be the problem, and mentioned dense models become annoying once exceeding **50GB**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Stirs Spyware Suspicion**: Users debated whether the **Comet browser** is spyware due to background activity, with counterarguments citing [Perplexity's privacy policies](https://www.perplexity.ai/hub/legal/privacy-policy) and [Comet-specific notices](https://www.perplexity.ai/hub/legal/comet-privacy-notice).
   - The consensus leans towards **Comet's Chromium base** and its background processes being standard browser operations rather than malicious spyware.
- **Minecraft Server Builds Blocky Excitement**: Enthusiastic members proposed a **Perplexity Minecraft server**, weighing technical specifications, including [free hosting](https://shockbyte.com/) options with **12GB of RAM** and **3vCPUs**.
   - A moderator confirmed that some servers were rolled out.
- **Opus 4.5: Free But Metered**: The community noted that **Opus 4.5** is now freely accessible on [LMArena](https://arena.com) and [Google AI Studio](https://ai.google.dev/), but is subject to rate limiting on Perplexity at **10 prompts per week**.
   - Members reported that the rate limits may be dynamic.
- **Image Generation Limits Irk Users**: Users are hitting **image generation limits** within Perplexity, capped at **150 images per month**, and seeking clearer UI feedback on usage.
   - Better UI feedback was requested.
- **Perplexity Labs versus Gemini Ultra: Research Rumble**: Users debated the optimal model for research, suggesting **Perplexity Labs**, **Sonnet**, and **Opus** and highlighting the cost of **Gemini AI Ultra** at $250/month.
   - One user noted the model's utility in determining effective prompting structures.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora AI Faces European Delay?**: A user asked about **Sora AI's** availability in Europe, but no information was provided about a potential release.
   - The lack of clarity leaves European users in suspense regarding when they might access **Sora AI**.
- **AI-Text Camouflage: Mastering Authenticity**: Members discussed methods to make **AI-generated text** appear more human-like, advising to program **ChatGPT** to use less recognizable language and mimic typing speed.
   - This tactic aims to evade detection by **AI** text detectors and ensure the generated content blends seamlessly with human-written material.
- **Discord Channel Chaos: Taming the Flood**: Users voiced concerns about miscategorized posts, specifically regarding **Sora AI** content, and one member jokingly suggested renaming a channel *ai-to-ai-discussions* to highlight **ChatGPT** output overload.
   - The discussion underscored the importance of adhering to channel guidelines and utilizing appropriate channels for **GPT** outputs to maintain order and relevance.
- **Model Mania: Preferences Spark Debate**: Members revealed their preferences for **AI models**, with some preferring **Gemini 3 Pro** and **Claude Sonnet** for coding accuracy.
   - While some favored **OpenAI's models**, others found **AmazonQ** (Sonnet4.5) preferable despite potential bugs after the *kiro* update, [source](https://discord.com/channels/974519864045756446/977697652147892304/1446260837374230588).
- **GPT-5.1 Crushes Gemini 3 in Bug Hunt**: In a comparative assessment, **GPT-5.1 Thinking** surpassed **Gemini 3** in pinpointing bugs within code, in spite of **Gemini 3's** superior user interface.
   - During testing, **GPT-5.1** identified a bug missed by another model, but **Gemini 3** failed to detect any errors, [source](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680).



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter's AI Report Reveals Trends**: OpenRouter and a16z released their [State of AI report](https://openrouter.ai/state-of-ai), analyzing **100 trillion tokens** of LLM requests, highlighting trends such as the dominance of roleplay in open-source model usage and the rise of coding in paid model traffic.
   - The report also finds that users prioritize quality over price, with reasoning models handling over **50%** of all tokens, indicating a shift towards AI agents managing tasks.
- **Deep Chat Project Goes Open Source**: A member open-sourced **Deep Chat**, a feature-rich chat web component that can be embedded into any website and used with **OpenRouter AI models**, available on [GitHub](https://github.com/OvidijusParsiunas/deep-chat).
   - The project includes direct connection APIs as [illustrated here](https://cdn.discordapp.com/attachments/1092850552192368710/1446081774303182921/20-direct-connection-api.png?ex=693358d9&is=69320759&hm=31e27377dc619bb67335b3e1ef57631a8d151e5678154de2609dd9314bcb10c5), and the author appreciates Github stars.
- **Grok 4.1 Gets Slugged**: Users noticed the removal of **Grok 4.1 fast free** model, and a member explained that users on the *paid* slug were being routed to the free model, and advised migration to the [free slug](x-ai/grok-4.1-fast:free).
   - The **x-ai/grok-4.1-fast** slug will start charging as of December 3rd 2025, and some members feel **Cloudflare** *is the singular metaphorical stick that holds up the world*, as it underwent downtime.
- **Rumors of OpenAI Model Next Week**: A user shared an [ArsTechnica article](https://arstechnica.com/ai/2025/12/openai-ceo-declares-code-red-as-gemini-gains-200-million-users-in-3-months/) hinting at a new **OpenAI** model release next week, as **Gemini** gains traction.
   - Another user speculated about the model name they're testing, referring to it as *some model name*.
- **Anthropic Acquires Bun to Power Claude's Coding**: Referencing [this article](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone), Anthropic acquired **Bun** as **Claude's** code generation hits a **$1B** milestone.
   - Members discussed **Cursor** raising **$50B** on a **$500B** next round to buy **Vercel/Next**, and **OAI** released **Codex Max** while **Google** released **Deep Think Mode** for **Gemini 3**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 4.3 Lands on ByteDance Seed 36B**: **Hermes 4.3** on **ByteDance Seed 36B**, offers roughly equivalent performance to **Hermes 4 70B** at half the model size, post-trained entirely on the **Psyche network** secured by **Solana**, read [more](https://nousresearch.com/introducing-hermes-4-3/).
   - The instruct format is coincidentally similar to **Llama 3** and **Nous Research** may release **Mistral-3** fine-tunes of **Hermes** and **MoE** support in their internal trainer, so an **MoE** is next.
- **QuickChatah Launches Ubuntu GUI for Ollama**: A member released [QuickChatah](https://github.com/exploratorystudios/QuickChatah), a cross-platform **Ollama GUI** for **Ubuntu** built with **PySide6**.
   - They mentioned *I didn't like OpenWebUI because it was resource intensive* and that their version uses like **384KiB** of RAM tops.
- **Opus Model Exhibits Better Performance**: A user reported that the new **Opus** is better than the old one and *was before already the only model who dealed with that proper but now its even better at it and doesn't do some mistakes it did before.*
   - They also noted that **GitHub CoPilot** can't use **Opus 4** as Agent but **Opus 4.5** can, adding that **Opus 4.5** is also available in **antigravity** for free.
- **Deepseek V3.2 Praised for Affordability**: A user recommends using **Deepseek v3.2** because *it's super affordable* and asked for the Nous team to try to get **Hermes 4 70B** and **405B** on **OpenRouter**.
   - Teknium clarified that the **Hermes** models are already on **OpenRouter**, but the user clarified they meant that they wanted **Nous Research** to be a provider directly.
- **Simulate markets and logistics in Godot**: A member is building a **3D simulation space in Godot** to simulate markets, agriculture, and logistics interactions and asked for model recommendations, another member suggested contemporary **NLP economic simulation research**.
   - Another member agreed, citing that **Langchain** is a wrong abstraction and causes more headache than doing things from first principles, especially since LLMs are good at writing the types of stuff that Langchain was supposed to solve.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Deepseek V3.2 Agentic Task Drawbacks Uncovered**: Despite improvements, **Deepseek V3.2** faces issues, including being limited to **one tool call per turn**, ignoring tool schema requirements, and failing tool calls by outputting in `message.content` rather than `message.tool_calls`.
   - Users suggest **Deepseek V3.2** needs enhanced tool call post-training to address these limitations.
- **Kimi's Haggle Deal Glitch Troubles Users**: Users report issues with the **Kimi Black Friday haggle deal**, facing inaccessibility despite not having active subscriptions, one user speculated the sale to be over.
   - Another user reports the deal ends December 12th.
- **Kimi for Coding Access and Support Concerns**: Users face access issues with **Kimi for Coding**, needing a **Kimi.com** subscription for a key.
   - Questions arise regarding corporate policy supporting only **cloud code** and **roo code**, with users seeking contact information for inquiries.
- **Deepseek Targets Enterprise Over Casual Users**: A [YouTube video](https://www.youtube.com/watch?v=u0n6wMnEYsk) explains that Chinese labs like **Deepseek** are targeting enterprise users due to the intelligence-to-price ratio being crucial for agentic tasks.
   - While **Deepseek** may not focus on casual users, some claim it's popular as an alternative to **ChatGPT** and **Gemini**.
- **Sparking LM Fun: A Developer's Lament**: A user advocates for more fun and experimentation in the **LM** space, beyond chatbots and money-making ventures.
   - The user praises **Kimi** for its model, fun features, visual style, search, and name, but wishes it were more than *just a chatbot*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nemotron Speed Claims Face Scrutiny**: A member questioned the claimed **3x** and **6x speedups** of **Nemotron**, reporting it to be slower than **Qwen** based on their results, detailed in [a screenshot](https://cdn.discordapp.com/attachments/1189498205101109300/1446072385597079552/screenshot_2025-12-04_at_1.png?ex=6933501b&is=6931fe9b&hm=25f60606c7d7bb02f051283f7447ee73d6fb6addcea6458c780f9de59d78c41b).
   - The user had sought recommendations for **async RL MLsys papers** and blogs discussing different directions of scaling the RL system.
- **Nsight Warnings Surface in CUDA Kernel Optimization**: A member reported that increasing `launch__registers_per_thread` in a **CUDA kernel** optimization triggers specific **Nsight Compute warnings** related to the `LDGSTS.E.BYPASS.LTC128B.128` instruction (corresponding to `cp.async`).
   - The warnings indicate that *3.03% of global accesses are excessive* and *17.95% of shared wavefronts are excessive*, disappearing when register usage is lowered.
- **Sparse Attention Doesn't Spark**: Despite **13,000 papers** on *sparse attention*, its adoption in systems like **vLLM** remains limited, as highlighted in [this X post](https://x.com/skylight_org/status/1993637433838035026?s=20).
   - Meanwhile, the paper *VATTENTION: VERIFIED SPARSE ATTENTION* ([arxiv link](https://arxiv.org/pdf/2510.05688)) introduces a sparse attention mechanism with user-specified **(ϵ, δ) guarantees**.
- **TorchAO Quantization Tricks Exposed**: Compilation time remains slow in **TorchAO** even after previous precompilation and recent improvements using `FqnToConfig` have enhanced support for quantizing model weights, specifically targeting MoEs, detailed in [this pull request](https://github.com/pytorch/ao/pull/3083).
   - TorchAO also has a dedicated `MoEQuantConfig` that members may be interested in. A reference code for **MoE quantization** can be found [here](https://github.com/pytorch/ao/blob/main/torchao/_models/mixtral-moe/generate.py).
- **LLMs Cheating and INF Bugs Resolved in NVIDIA Comp**: The reference kernels for the NVIDIA competition appear to be using **cuBLAS 13.0.0.19**, corresponding to **CUDA Toolkit 13.0.0**, and, to prevent INF, a [PR was merged](https://github.com/gpu-mode/reference-kernels/pull/84) to use the **full fp4 range a/b** and **non-negative scale factors**.
   - LLMs have been found to have a **hack** in the evaluation with no known solution, and a user mistook the `nvfp4_gemm` competition for the closed **amd-fp8-mm**, but resolved it by explicitly passing the **--leaderboard nvfp4_gemm** flag.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Antithesis Stress Tests AI Code with Jane Street Bucks**: Jane Street led a **$105M Series A** investment in [Antithesis](https://xcancel.com/_sholtodouglas/status/1996297367776309359?s=46), focused on **deterministic simulation testing** for AI-generated code verification.
   - The conversation focused on the necessity of **trust-through-testing** as AI increasingly automates coding tasks.
- **Anthropic Forecasts Massive ARR**: Anthropic expects to close the year with **$8–10B in annualized revenue**, a substantial leap from the **$1B** projection in January, as per [this link](https://xcancel.com/deredleritt3r/status/1996294139843862618?s=20).
   - This surge is driven by significant enterprise adoption of **Claude**, particularly for coding, while OpenAI aims for **$20B ARR**.
- **Harvey's Hefty Series F**: Legal AI firm **Harvey** raised **$160M in Series F** funding led by a16z, reaching an **$8B valuation** and serving over **700 law firms** across **58 countries**, according to [this tweet](https://xcancel.com/brian_a_burns/status/1996624620519399634?s=46).
   - The company's humble origins with just 10 people in a WeWork space were highlighted.
- **TanStack AI Steps Into the Arena**: [TanStack AI](https://tanstack.com/blog/tanstack-ai-alpha-your-ai-your-way) was introduced, boasting full type safety and multi-backend language support.
   - The team promised a forthcoming blog post and documentation detailing its advantages over Vercel.
- **Kling Synchronizes Audio, Blows Minds!**: [Angry Tom's tweet](https://x.com/angrytomtweets/status/1996367439622529193) demonstrated generative video progress over **2.5 years**, featuring Kling's **VIDEO 2.6** with synchronized audio.
   - Observers jokingly suggested that *AI Will Smith eating spaghetti* is the new Turing test, igniting speculation on future realism.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SLMs for Agents Gain Traction**: The founder of **A2ABase.ai** is actively researching **Small Language Models (SLMs)** for use in agents, and a member suggested exploring alignment benchmarks from the **Emergent Misalignment** paper and the **Cloud et al subliminal learning** paper.
   - The founder is creating training pipelines for small LMs to be trained on less than **16GB VRAM** and asked for benchmark recommendations for small models trained on edge devices, and looking into the [HuggingFace LM training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook).
- **Shampoo Might Need More Power**: A Google employee stated that the [Shampoo paper](https://arxiv.org/abs/2503.20762) might need the power to be **-1** instead of **-1/2**.
   - The author stated it's *ok work* but has *a few other deficiencies*.
- **CFG Benefits Show Early Memorization**: Members discussed the benefits of **CFG (Classifier-Free Guidance)** and memorization, referencing [this paper](https://arxiv.org/abs/2411.16738).
   - One member was surprised that the **basin** emerges so early and that it probably has something to do with the resolution and size of the dataset.
- **LLMs Aid Visual Creation**: A member has used LLMs to help make visuals for videos, creating voiceover text on Clipchamp, and building programs for a **4D physics engine** for his company.
   - The member added that language can be severely limiting, and have found that LLMs often struggle to understand what they are trying to convey, requiring them to teach the LLM how to process the 3rd step and simulate the prime number 'latch' for non-quantized signal analysis.
- **SHD CCP Protocol Explained**: A member shared a series of videos explaining their work on interoperability, particularly the **SHD CCP** (**01Constant Universal Communication Protocol**), including an [introduction to the language](https://www.youtube.com/watch?v=frmRYqTyCh4).
   - Additional videos were shared that covered [use cases for 0(1) time compression data](https://www.youtube.com/watch?v=pD7lPA-p0zo), [optimizations for cycle saving in modern GPUs](https://www.youtube.com/watch?v=harPSuCPGYI), and the [necessity of quaternions](https://www.youtube.com/watch?v=9DXqgPtZstE?si=VAe-C-HPqcvvpL2x).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Brian Douglas Advocates Control Theory Learning**: A member suggested using **Brian Douglas's** video to learn control theory, while noting that practical projects are essential for understanding the concepts.
   - They suggested *control theory is something that won't sink in without doing an actual project though*.
- **DeepSeek Article Raises Linearity Question**: A member shared a [DeepSeek article](https://magazine.sebastianraschka.com/p/technical-deepseek) about control theory, asking whether the **linearity assumption** limits the use of linear control.
   - The member then exclaimed, *Control theory is actually funWhy aren't people talking about it*.
- **AWS Re:Invent 2025 Updates Spark Debate**: Amazon announced [AWS re:Invent 2025 AI news updates](https://www.aboutamazon.com/news/aws/aws-re-invent-2025-ai-news-updates) including **Nova Forge** to build frontier AI models.
   - One member called out the updates as *click bait by a literal political opinion*.
- **Nova Forge Promises Frontier Customization**: **Nova Forge** is a service for building custom frontier AI models; [more info here](https://www.aboutamazon.com/news/aws/aws-agentic-ai-amazon-bedrock-nova-models).
   - Members questioned how it differs from basic fine-tuning and noted it may offer more flexibility with checkpoints and integration of *gyms* for **RL training**.
- **Bezos's AI Company Remains Elusive**: Members noted the absence of **Bezos's new AI company** in the AWS re:Invent 2025 announcements.
   - They speculated on potential competition or specialization between the firms.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Multi-GPU Setup Troubleshoot Requested**: A member shared a [link to their multi-GPU setup](https://huggingface.co/datasets/John6666/forum3/blob/main/get_quantization_config_trl_all_gather_error_1.md) and requested a sanity check, revealing their inexperience with **multi-GPU configurations**.
   - The member appeared unsure about the setup's correctness, highlighting the challenges faced when configuring **multi-GPU systems** for the first time.
- **Image Models Still Censor Explicit Content**: Despite being uncensored, the **Z image demo** censors explicit content, such as gore or nudity, displaying a *maybe not safe* image.
   - The member questioned if a configuration error or improper usage caused the model to deviate from its expected behavior of generating **uncensored content**.
- **ODE Solver Powers Up Diffusion Models**: A new **fast ODE solver**, ideal for **diffusion models**, was created; its [Hugging Face repo](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-4k) is now available.
   - The author claims one can sample a **4K image in 8 steps** with results matching **30 steps of dpm++2m SDE with karras**; the [paper](https://arxiv.org/abs/2510.02390) is also accessible.
- **smallevals Locally Assesses RAG Systems**: A member launched **smallevals**, a suite for evaluating **RAG** / retrieval systems swiftly and freely using tiny **0.6B models** trained on **Google Natural Questions** and **TriviaQA** to produce golden evaluation datasets, installable via `pip install smallevals`.
   - This tool has a built-in local dashboard to visualize rank distributions, failing chunks, retrieval performance, and dataset statistics, with the first released model being [QAG-0.6B](https://huggingface.co/mburaksayici/golden_generate_qwen_0.6b_v3_gguf), which creates evaluation questions directly from documents to evaluate retrieval quality independently from generation quality, with [source code available on GitHub](https://github.com/mburaksayici/smallevals).
- **STRAW Rewrites Neural Net Wiring for Every Image**: A member introduced **STRAW (sample-tuned rank-augmented weights)**, an experiment mimicking biological neuromodulation where the neural net rewrites its own wiring for every single input image it sees, mitigating RAM crashes by using **low-rank** techniques, a step towards *liquid* networks.
   - The deep dive with the math and results are available in [this write-up](https://teendifferent.substack.com/p/sample-tuned-rank-augmented-weights).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Community Meeting YouTube Release Delayed**: The release of the **November 24th community meeting** video on YouTube is delayed due to the U.S. holiday, scheduled to be uploaded *tomorrow*.
   - The video is currently being processed.
- **Level 15 Achieved**: Congratulations to a member for advancing to level 15!
   - Another member advanced to Level 1!
- **`codepoint_slices` Debugging Unearths Memory Access Error**: An investigation into a failing AOC solution using `codepoint_slices` revealed an **out-of-bounds memory access** due to an empty list.
   - The issue was resolved by switching from `split("\n")` to `splitlines()`, which avoids the empty line causing the error; debugging with `-D ASSERT=all` could have caught it sooner.
- **`splitlines` vs `split("\n")` Exhibits Discrepancies**: `splitlines` and `split("\n")` exhibit different behaviors with trailing newlines, where `splitlines` omits the last empty line, mirroring [Python's behavior](https://docs.python.org/3/library/stdtypes.html#str.splitlines).
   - `split("\n")` includes the empty line as an empty string in the resulting list.
- **GPU Constant Memory Explored via github**: An example demonstrating the usage of constant memory was found in the [modular/modular GitHub repository](https://github.com/modular/modular/blob/b8756654dd050be664396757be2fc7c495484e1b/max/kernels/test/gpu/basics/test_constant_memory.mojo#L105).
   - A question was raised regarding methods for placing data, such as convolution kernels computed at runtime, into the GPU's constant memory.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Eyes Distributed Inference Systems**: Members discussed using **aider** with distributed inference AI systems like [Crowdllama](https://www.crowdllama.com/) and setting up an API server with **llama.cpp** to benchmark performance.
   - One user pointed out having **16GB** of memory but **no GPU**, which likely explains the slower speeds.
- **Ollama Timeout Troubleshooter Seeks Solutions**: A member reported getting timeout errors with **Ollama** while using models like `gpt-oss:120b` and `llama4:scout`, resulting in a `litellm.APIConnectionError` after **600.0 seconds**.
   - No specific solution was found in the provided context.
- **Aider Flags Need Manual Confirmation**: A user found that the **--auto-test** and **--yes-always** flags in **aider** were not fully automating the process.
   - They reported that they still required manual execution despite using these flags.
- **Mac & Fold 6 Get Aider Setup Advice**: A new user wants to run **LLMs locally on their Mac** and then run **aider** on their **Fold 6** (in the same network).
   - The user is seeking advice from anyone who has implemented a similar setup for coding on their Fold device.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MCP Apps SDK goes Open Source!**: General Intelligence Labs has open-sourced the [**mcp-apps-sdk**](https://github.com/General-Intelligence-Labs/mcp-apps-sdk), which allows developers to embed apps designed for **ChatGPT** into other chatbots, assistants, or **AI** platforms, enabling local testing.
   - The company posted an explanation on X [here](https://x.com/helloxalia/status/1796319442863866351?s=20) explaining why they are building the **MCP Apps SDK**.
- **Latent Collaboration Paper Shared**: A member shared a [link](https://share.google.com/s/QHcMFSqiTkTnZ231) to a paper on **Latent Collaboration** in **Multi-Agent Systems**.
   - The paper explores methods to enable agents to *implicitly coordinate* through learned latent spaces.
- **Subforum for Student Models Suggested**: A member suggested creating a dedicated subforum for discussing *student models* like **Qwen3** and **gpt-oss-20b** to consolidate knowledge on best settings and use cases.
   - The goal is to pool community experiences and optimize the application of these models.
- **Claude Code LM Integration Proposed for DSPy**: A member proposed adding **Claude Code** / **Claude Agents SDK** as a native LM within DSPy, potentially using `dspy_claude_code`.
   - This integration would support structured outputs and leverage Claude Code's tools like `Read`, `Write`, `Terminal`, and `WebSearch`.
- **Full Stack Engineer Automates with DSPy**: A full stack engineer specializing in **workflow automation, LLM integration, RAG, AI detection, image and voice AI** introduced themself, highlighting experience building automated pipelines and task orchestration systems using **DSPy**.
   - One system connects **Slack, Notion, and internal APIs to LLM**, cutting response times by **60%**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI Engineer Automates for Efficiency**: An AI engineer detailed their proficiency in **Workflow Automation & LLM Integration**, **RAG Pipelines**, **AI Content Detection**, **Image AI**, **Voice AI**, and **Full Stack development**, demonstrating successful project implementations.
   - They reported a **60% reduction in response times** by creating pipelines that integrate Slack, Notion, and internal APIs.
- **Account Suspensions for Referrals**: A user reported their account suspension following multiple referrals, prompting an official response.
   - An agent suggested appealing through official channels and offered follow-up assistance if a response is delayed.
- **Chat Mode Makes Triumphant Return**: **Chat Mode** has been officially reinstated; instructions for using it are available at [this link](https://help.manus.im/en/articles/11985220-can-i-switch-back-to-chat-mode-from-agent-mode).
   - Note that using **Chat Mode** still consumes credits.
- **Manus Eyes New Talent**: Manus is actively **recruiting new talent** and inviting interested candidates to submit their resumes via DM.
   - Submitted resumes will be reviewed by HR and relevant teams to enhance Manus' capabilities.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **`train_step` Still Needs Work**: A recent [PR](https://github.com/tinygrad/tinygrad/pull/13553) almost fixed an issue with `train_step(x, y)`, but it still receives two tensors without utilizing them.
   - This means the training step isn't correctly processing input data, requiring further attention to complete the fix.
- **`shrink` beats indexing for `obs` Tensor**: Using `obs.shrink((None, (0, input_size)))` is reportedly faster than `obs[:, :input_size]` for indexing the `obs` tensor.
   - This optimization could ramp up performance when working with large observation tensors by leveraging `shrink` for faster slicing.
- **`Variable` `vmin` Gets a Bump**: The `Variable` `vmin` parameter had to be increased to 2 to avoid errors.
   - The original `vmin` setting was causing issues, thus needing an adjustment to ensure proper functionality and stability.
- **`RMSNorm -1` Dimension Needs Verification**: The use of `-1` as a dimension parameter in `RMSNorm(dim=-1)` needs verification.
   - Members suggested checking the [source code](https://github.com/tinygrad/tinygrad) of `RMSNorm` to confirm that it behaves as expected with a negative dimension index.
- **Tinygrad Refactors Master Branch**: An outdated codebase element can no longer be found on the current master branch and has been moved.
   - It can now be found under the name [axis_colors dict](https://github.com/tinygrad/tinygrad/blob/3eae1461396c25755c4fb64194b3decd4e539934/tinygrad/uop/ops.py#L20).



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Debate Sparked on Tools Accepting UUIDs**: A discussion has begun on whether tools should accept **UUIDs as input**, focusing on mitigating the issue of **LLMs** outputting **UUIDs** despite prompts against it.
   - Opinions vary, with some questioning if it's inherently bad practice and others finding it acceptable under certain circumstances.
- **LLM's Role in UUID Creation Questioned**: A member expressed reluctance towards allowing **LLMs** to *create* **UUIDs**, suggesting it's more appropriate for **LLMs** to use **UUIDs** to retrieve items from other tools.
   - The suggested architecture involves a `list_items` tool returning lightweight items with **UUIDs**, complemented by a `describe_item` tool that uses a **UUID** to return a complete item.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **GPT-5.1-Codex Max Arrives on Windsurf**: **GPT-5.1-Codex Max** is now integrated into Windsurf, accessible to users with **Low**, **Medium**, and **High** reasoning levels.
   - Paid users get a free trial of **5.1-Codex Max Low**, available via the latest Windsurf version, as detailed in [Windsurf's X post](https://x.com/windsurf/status/1996665911185756511?s=20).
- **Windsurf Dangles Free GPT-5.1-Codex Max Trial**: Windsurf provides a free trial of **GPT-5.1-Codex Max Low** to its paid user base for a limited duration.
   - Users need to grab the newest Windsurf version to enjoy this trial, announced on their [X post](https://x.com/windsurf/status/1996665911185756511?s=20).



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1445867194431832094)** (1282 messages🔥🔥🔥): 

> `Profitable AI Companies, LM Arena Prompt Limits, Gemini 3 Deepmind, Frame-Flow Model, OpenAI Models - Robin-High` 


- **AI Companies Struggle for Profitability**: Members discussed the difficulty of achieving profitability in the AI sector, noting that even top AI companies are facing challenges due to the high computational costs and infrastructure needs.
   - *Compute is expensive,* making it difficult for AI to be as accessible and profitable as traditional software or internet services.
- **LM Arena Implements Prompt Limits to level playing field**: LM Arena has implemented a repeated prompt limit of **4** to ensure fair testing, which errors out on repeated prompts but can be reset with new prompts, although this causes the chat to delete old prompts/responses.
   - One user inquired about the new measures, questioning if the new rate limit was made *to make the arena fairer*.
- **Frame-Flow Model Emerges as Gemini 3 Flash Contender**: Users are actively trying to identify the new *Frame-Flow* model, which is kicking Opus's butt in text battles, with some speculating it could be **Gemini 3 Flash**, a weaker model, or **Grok 5**, while others suggest it may come from a new company.
   - The discussion also revolves around testing Frame-Flow with steganography puzzles and assessing its coding abilities compared to existing models.
- **Seedream 4.5 image model now available for image arena**: Seedream 4.5 image model has been released into Image Arena, now available via selecting **Direct or Side by Side modes** in the dropdown.
   - Members find the model comparable to nano banana pro though some users argue the model is inferior. The rate limit is **5 Generations/h**.
- **Users Expose Potential Sora 2 Exploit**: A user claims to have cracked the filtration system in **Sora 2**, which is currently a SOTA text-to-video model, with guard rails not equally distributed.
   - The user notes the exploit involves generating content using specific prompts and methods that bypass restrictions, but the only way to fix it is to *not allow people to generate characters*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1445911123458261225)** (3 messages): 

> `Search Arena Leaderboard, New Model in Text Arena, Text-to-Image Arena, Image Edit Arena, Seedream-4.5` 


- ****Gemini-3-pro-grounding** Grounds Itself in the Top Spot**: The [Search Arena leaderboard](https://lmarena.ai/leaderboard/search) has been updated, with **Gemini-3-pro-grounding** ranking #1 and **Gpt-5.1-search** at #2.
- ****Nova-2-lite** Joins the Text Arena**: A new model, **nova-2-lite**, has been added to the [Text Arena](https://lmarena.ai/c/new) and announced on [Twitter](https://x.com/arena/status/1996396395411177920).
- ****Seedream-4.5** Enters Image Arenas and Leaderboards**: The model **Seedream-4.5** has been introduced to the Text-to-Image Arena and Image Edit Arena, ranking #3 on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) and #7 on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image).


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1445867581247590651)** (1240 messages🔥🔥🔥): 

> `Grok Real?, Gemini vs Claude for malware, NSFW Gemini Web Jailbreak, GPT5.1 Jailbreak, AI Agent Attack Scenarios` 


- **Grok or Not: Reality Check Requested**: Members are questioning the authenticity of a [tweet about Grok](https://fixupx.com/i/status/1996247117468209453) with gifs and images to express their skepticism and amusement.
   - One member even shared [a GIF](https://tenor.com/view/grok-is-this-true-windmill-water-wheel-gif-5302497903247327835) depicting a water wheel, jokingly suggesting Grok's capabilities are just for show.
- **Gemini Versus Claude: The Quest for Malware Mastery**: Members discussed using **Gemini 3** to create prompts that **Claude** shouldn't be able to handle, focusing on generating code or malware, with a key point being that knowing specific coding questions is more effective than typical jailbreaking attempts.
   - Despite some success, opinions diverged, with one member finding **Claude** unimpressive, while others debated the value and future of using AI for malware creation versus traditional reverse engineering methods.
- **System Prompting Gems on Gemini Web**: Members shared experiences and methods for generating NSFW images with **Gemini Web** using system instructions, highlighting the limitations of the platform's filters and the instability of the results.
   - Despite challenges, a member found **Seedream 4.5** to be the most effective model for editing NSFW images due to its prompt adherence and output stability, contrasting it with **Nano Banana Pro**, which is hindered by filters and inconsistent results.
- **AI Agent Attack Vectors**: Members discussed simulating and documenting real-world AI agent attack scenarios, including prompt injection attacks, spoofed agent messages, and replay attacks, to test security thinking and threat modeling.
   - These simulations aim to identify vulnerabilities and evaluate the effectiveness of execution-time authorization methods like **A2SPA** in preventing unintended execution and unauthorized access.
- **Cracking the Code for GPT 5.1**: Members are searching for a method to successfully jailbreak **GPT 5.1** after being unable to generate a script and evade its set principles.
   - In this search, members shared the [UltraBr3aks](https://github.com/SlowLow999/UltraBr3aks) GitHub Repo to test methods, but none were able to successfully jailbreak **GPT 5.1**.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1445895422836215959)** (69 messages🔥🔥): 

> `Gemini Jailbreak, ENI JB on Gemini 2.5, WormGPT Origins, GPT5.1 Jailbreaking, Grok 4.1 Instructions` 


- ****Gemini Jailbreak**: Seeking Assistance**: A member is seeking a working jailbreak for **Gemini** models, requesting assistance from the community.
- ****ENI JB** works on **Gemini 2.5****: A member mentioned that the "**ENI**" jailbreak worked well on **Gemini 2.5**, linking to a [Wired article](https://www.wired.com/story/poems-can-trick-ai-into-helping-you-make-a-nuclear-weapon/) on tricking AIs.
   - The conversation also questions the whereabouts of the original **WormGPT**, which was a GPT-J finetune.
- ****GPT5.1** jailbreak**: A member stated that jailbreaking can only be used for very specific purposes and it also reduces the intelligence of AI, claiming to have *jailbroken gpt5.1 but for some few request it ended up getting detected*.
- ****Grok 4.1**: Coke Recipe Unleashed?**: Members discussed about getting **Grok 4.1** to give instructions on making coke, and the possibility of creating a custom GPT bot to achieve similar results.
- ****DeepSeek** jailbreak successful**: Members reported successfully jailbreaking **DeepSeek** using nested jailbreaks, sharing a [screenshot](https://cdn.discordapp.com/attachments/1228043845967544380/1446279240201928845/FireShot_Capture_029_-_Creating_Malware_for_Windows_Reverse_Shells_-_DeepSeek_-_chat.deepseek.com.png?ex=69336801&is=69321681&hm=911b769e6ba2306df1cd2dac947fbe4b977ba9660266ad6f203bb86e32d4f774) as evidence of its ability to generate malware code.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1445886671139901482)** (6 messages): 

> `Jailbreaking zapgpt2, Finding SMTP servers` 


- **User Attempts to Jailbreak zapgpt2**: A user shared a link to [zapgpt2.org](https://zapgpt2.org/) and asked someone to jailbreak this **AI**.
   - The stated goal was *malicious coding*.
- **User Needs Help Finding SMTP Servers**: A user asked for help finding **SMTP servers** that accept inboxes from many domains.
   - The user posted this request without any additional context.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1445879298971013344)** (281 messages🔥🔥): 

> `llama.cpp vulkan on Asahi Linux, Running LLMs on Phones via llama.cpp, Unsloth community, New Ministral 3 Reinforcement Learning notebook for Sodoku, Gemini 3 Pro vs Claude` 


- **Vulkan Backend Tested on Android via llama.cpp**: Members confirmed that the **Vulkan backend** works via *llama.cpp* on **Android**, but it needs the correct **ICD (Freedreno)** and may have issues with **FP16** depending on the hardware.
   - A member shared that it is much easier to use `pkg install llama-cpp` than to compile with Vulkan support.
- **LLMs run on Phones via llama.cpp and Termux**: Members are running **LLMs** on phones directly with *llama.cpp* through **Termux**, and are also using *kobold.cpp*.
   - One member noted that while some believe any phone can run a **4B model 24/7**, this is far from reality; others pointed out that they are running **Qwen 3 14B** on their **iPhone 17 Pro**, while another is making their Dad use **Gemma E2B** on his **iPhone 12** because it barely works.
- **Unsloth Community Praised for Active Finetuning**: Members praised the **Unsloth Discord community** as highly active and valuable for finetuning, and a few said that the team building this community is badass, cute, and humble.
   - In response to a question of how this community was built, the answer was *it's just you guys really xD you guys started being active which helped a lot*.
- **Ministral 3 RL Notebook Solves Sodoku Puzzles**: The **UnslothAI** team released a new **Ministral 3 Reinforcement Learning notebook** that solves **Sodoku puzzles** [link](https://x.com/UnslothAI/status/1996595704438120774).
   - The post got much love and fire emojis.
- **Gemini 3 Pro Declared Useless**: One member stated *I can officially say that **Gemini 3 Pro** is useless and dead*, because it summarizes responses and gives short, limited answers.
   - They further said that **Claude** performed excellently in their language task and is also very intelligent, adding that is way better than **GPT**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1445873220799561975)** (555 messages🔥🔥🔥): 

> `Nvidia VRAM, Trainable SFX models, Levenshtein names, whisper language detector, Crete vacation` 


- **Nvidia Stops VRAM Supply?!**: Members speculated that **Nvidia** might be stopping their supply of **VRAM** to partners, potentially leading to supply difficulties for smaller AiB partners and another EVGA-like situation.
   - One member jokingly suggested shorting **3090** stock might be a good idea, sparking a discussion about market dynamics.
- **Adventures in Crete Await!**: A member shared plans to vacation in **Crete** in January, embracing the off-season despite the cold sea, and awakening their inner Russian resistance to the cold.
   - They referenced a [Titanic GIF](https://tenor.com/view/titanic-rose-gif-16721897849522138532) joking about the cold weather and shared an anecdote about aiming a gun without gloves in -45C temperatures during their military service.
- **Micron's Memory Meltdown**: **Micron** is reportedly exiting the *Crucial* consumer business, ending retail **SSD** and **DRAM** sales according to a [TechPowerUp article](https://www.techpowerup.com/343633/micron-to-exit-crucial-consumer-business-ending-retail-ssd-and-dram-sales), sparking concerns about rising RAM prices.
   - The group joked about buying up as much RAM as possible, and another hoped quantum computing won't need RAM.
- **Tracing Transformers Troubles**: A member shared a method using `type()` in Python for tracing code in the **Transformers** library, which is often difficult due to deep inheritance and numerous models.
   - Another member suggested using `inspect.getsourcefile` to find the source file of a class, as showcased in [this example](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
- **Frankenmerges Fuel RAM Frenzy**: The community discussed the **k2-merged-3.5T-bf16** model by NousResearch ([HF link](https://huggingface.co/NousResearch/k2-merged-3.5T-bf16)), joking that these large *frankenmerges* are contributing to RAM shortages and causing complaints about the RAM prices.
   - Members joked that their webcams would open if offered DDR or GPUs.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1445978319412793364)** (30 messages🔥): 

> `Ollama crashes with Unsloth dynamic quant for Qwen3-VL, AttributeError in trainer.train() with custom classification task, Unsloth installation overwrites torch with CPU version, WSL2 setup for Windows 11` 


- **Qwen3-VL Dynamic Quant Crashes Ollama**: Unsloth dynamic quant for **Qwen3-VL** is crashing in **Ollama** due to incompatible formats between `llama.cpp` and Ollama, according to [this Github issue](https://github.com/ollama/ollama/issues/13324).
- **AttributeError plagues Trainer**: A user encountered an **AttributeError**: *'int' object has no attribute 'mean'* when running `trainer.train()` on a custom classification task built on the Unsloth framework, even when using the [official notebook](https://www.kaggle.com/notebooks) code.
- **Unsloth Installs CPU Torch?**: A user reported that attempting to `pip install` Unsloth overwrites their **Torch** installation with a **CPU version**, preventing proper execution.
   - They used the command `python -m pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"` but were still experiencing issues.
- **Windows Users Should WSL2**: A user using **Windows 11** was advised to install **WSL2** and run **VSCode** within it for a smoother development experience, with the suggestion to search the **help** channel for existing step-by-step guides.
   - The user was provided links to [Conda Installation](https://docs.unsloth.ai/get-started/install-and-update/conda-install) and [Pip Installation](https://docs.unsloth.ai/get-started/install-and-update/pip-install) guides.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1445917997335052381)** (49 messages🔥): 

> `ToS Violations and Model Training, Distillation impact, Model extractions, Model characteristics` 


- **Debate over ToS Violations for Model Training**: Members debated whether it's acceptable to share research that violates the ToS of services, especially concerning distillation and reverse engineering, with one user stating that *every model in existence is trained in violation of some ToS or law*.
   - Another user countered that openly stating distillation or reverse engineering is different from speculating about undocumented training data, clarifying that the sharing of links is for discussion only, not encouragement of the original actions.
- **Model extractions with reliability**: A member mentioned the possibility of reliably extracting datasets from models, referencing a case where someone extracted nearly verbatim Anthropic's "Soul Doc" ethical principles.
   - This relates to the potential for models to reveal their training data, even without explicit disclosure from their creators.
- **Distillation**: A user criticized the model's example prompt for relying on em dashes and "not X but Y" phrasing, comparing it to a chatbot writing style highlighted in [a New York Times article](https://www.nytimes.com/2025/12/03/magazine/chatbot-writing-style.html).
   - The user joked that the distillation of the model made it through.
- **Distillation carries characteristics**: One user noted that distillation carries unrelated characteristics of a model, even if not explicitly present in the data, referencing personal experience and [a relevant video](https://youtu.be/NUAb6zHXqdI).
   - Another agreed, explaining that adjusting weights for specific outputs can affect other areas of intelligence, citing fine-tuning as a cause for degradation in unrelated areas.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1445867480097624216)** (449 messages🔥🔥🔥): 

> `Grok Code, Professional UIs, Cursor Agent, Platform UI Changes, Open Empathic` 


- **Grok Code praised then mocked**: One user initially loved the way **Grok Code** was thinking, but later said *bro stopped thinking at all*.
   - No other details were given.
- **UI Devs Crave Cursory-UI Tips**: Users asked for tips on making professional **UIs** without paying for Figma, with one suggesting to reproduce layouts by pasting screenshots into Cursor, and asking it to reproduce it.
   - No other details were given.
- **Nightly Builds Trigger Rogue Agents**: Users reported that **Cursor Agents** were running without permission, creating and deleting files, and potentially downloading codebases.
   - One user noted that their post on the forum was deleted, and recommended downgrading to a stable version instead of the nightly build, and disabling dotfile access and external file access.
- **Auto Agent is Getting Dumber and Dumber**: A user complained the **Auto Agent** purposely went crazy by comparing unrelated pages; another reported the agent produced 11, then 34 errors.
   - Others said the quality of the model depends on the tasks given.
- **Auto Model No Longer Free For Some**: Users discussed the transition to a new pricing model where **Auto is no longer free** after the next billing cycle for some subscribers.
   - One user shared they had used **360M tokens** this month and calculated this to costing them **$127**, and plans to switch to **$12 Copilot** with **GPT5-mini**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1445868750925598942)** (355 messages🔥🔥): 

> `CachyOS desktop environment choices, Dual GPU issues with CachyOS, CachyOS telemetry, Qwen in LM Studio, GPT OSS` 


- **GNOME is choice DE for some CachyOS users**: One user chose [GNOME for CachyOS](https://www.gnome.org/) because they *"can't stand KDE"* and find Cinnamon light on VRAM.
- **Dual GPU Blues with CachyOS**: Users report issues with running **two different GPUs** (e.g., Nvidia 4070ti and 1070ti) on CachyOS, experiencing an error that doesn't occur on Ubuntu.
   - The problem may be related to using **GPUs from different generations**, prompting one user to consider using the second GPU in another PC.
- **CachyOS telemetry opt-out clarified**: A CachyOS team member clarified that **telemetry is enabled by default** but can be disabled by setting *telemetryEnabled: false* in the config, aiming to ensure documentation accuracy.
   - The team member addressed potential misunderstandings about previous conversations, clarifying they were not involved in them.
- **Qwen is now easy to use in LM Studio**: **Qwen** is now supported in LM Studio, as showcased by a user's screenshot of the [LM Studio UI](https://lmstudio.ai/).
   - Others remarked on possible UI bugs and the large VRAM requirements for certain Qwen model quantizations.
- **GPT OSS Model Capabilities Debated**: Users discussed the capabilities of **GPT-OSS**, with one stating it is *"without a doubt the most capable model that can be run in typical consumer hardware, if it isn't crippled by policy, full stop."*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1445867339399561307)** (75 messages🔥🔥): 

> `DDR4 Speed Viability, eBay Mac Studio Scams, Multi-PSU GPU Wiring, Intercepting LLM Requests Server, Triple GPU Bugginess` 


- ****DDR4 3200MHz Still Kicking****: A member inquired about the viability of **3200MHz DDR4** compared to **3600MHz**, and another member responded with an image noting that **3200Mhz** is basically top of the bracket of the DDR4 standards.
   - The attached image indicated that **3200MHz** is the *top of the bracket* for **DDR4** standards.
- ****eBay Mac Studio's Aboundeth with Scammers****: A member noted the presence of numerous scam sellers for **Mac Studios** on **eBay**, highlighting one seller charging an extra **$600** for a **2TB** model.
   - The implication was that these listings were overpriced or fraudulent.
- ****Multi-PSU GPU Wiring - Risky or Rewarding?****: Members discussed the proper wiring for multi-PSU setups with GPUs, advising against connecting two PSUs' 8-pin connectors to the same GPU unless the PCIe and 8-pin rails are split.
   - One member mentioned feeding a GPU from two different PSUs without issue, while another shared a [photo](https://cdn.discordapp.com/attachments/1153759714082033735/1446150071480619058/IMG20251204084348.jpg?ex=6932efb5&is=69319e35&hm=123106491a478d1b0c6012f03b6a947a1ab8719b0498ed8425d8693ab03160fe&) of a potentially *dangerous* power setup.
- ****Queuing LLMs: A Server's Tale****: A member wants to convert an old gaming laptop into a server to intercept and control requests for LLMs to prevent system overload, and allow multiple devices to use the main server with a queuing system.
   - The goal is to implement a Linux setup that manages LLM requests efficiently.
- ****Triple GPU Setup? Prepare for Bugginess!****: A user reported that a triple GPU setup is *very buggy* out of 10, prompting another to jokingly suggest adding a fourth to fix it.
   - One member noted *issues with splitting LLMs across non-even numbers of cards*, another suggested that an 8GB GPU might be the problem, and mentioned dense models become annoying once exceeding **50GB**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1445867808356434191)** (399 messages🔥🔥): 

> `Comet Browser: Spyware Accusations, Perplexity Minecraft Server, Opus 4.5 availability and limits, Image generation limits, Perplexity and Prompt Engineering` 


- **Comet Browser Sparks Spyware Debate**: A user questioned whether the **Comet browser** is spyware, citing its background activity, while others provided links to [Perplexity's privacy policies](https://www.perplexity.ai/hub/legal/privacy-policy) and [Comet-specific notices](https://www.perplexity.ai/hub/legal/comet-privacy-notice) to refute the claim.
   - The discussion clarified that **Comet's Chromium base** and background processes do not inherently indicate spyware, but rather are standard browser operations.
- **Perplexity Community Eyes Minecraft Server**: Members expressed enthusiasm for a **Perplexity Minecraft server**, suggesting it to the team and discussing the technical specifications of the [free hosting](https://shockbyte.com/) which includes **12GB of RAM** and **3vCPUs**.
   - A moderator confirmed that the server had been rolled out to some servers.
- **Opus 4.5 Now Free, but Rate-Limited**: Users discovered that **Opus 4.5** is available for free on [LMArena](https://arena.com) and [Google AI Studio](https://ai.google.dev/), noting a rate limit of **10 prompts per week** on Perplexity.
   - Members suggest that the rate limit is now dynamic based on server load.
- **Image Generation Limits Frustrate Users**: Users are facing difficulties with **image generation limits** within Perplexity, hitting a limit of **150 images per month**.
   - Users requested that Perplexity provide better UI feedback to indicate limits and usage.
- **Perplexity versus Gemini for research**: Users are discussing which model would be best for research, with one user recommending **Perplexity Labs**, **Sonnet** and **Opus**, while pointing out that **Gemini AI Ultra** costs $250/month which is not ideal.
   - Another user reported that the model is good enough to tell you what prompting structure works best overall.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

nike0656: https://www.perplexity.ai/search/5f87b568-aa15-4dd6-801a-786a6bedd45b
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1445871179570352178)** (264 messages🔥🔥): 

> `Sora AI Availability in Europe, AI-generated Text Detection, OpenAI Discord Channel Guidelines, Model Preferences, Gemini 3 Pro vs. GPT-5.1 Thinking` 


- **Sora's European Debut in Doubt?**: A user inquired about the availability of **Sora AI** in Europe.
   - However, there was no confirmation or information provided about its release in Europe.
- **Mastering AI-Generated Text Camouflage**: Members discussed how to make **AI-generated text** appear more authentic and less obviously produced by **ChatGPT**.
   - The advice included programming **ChatGPT** to use less recognizable language patterns and manually typing before pasting generated text to mimic natural typing speed.
- **Navigating OpenAI Discord: Channel Chaos**: Users expressed frustration with others posting in incorrect channels, particularly regarding **Sora AI** content.
   - One member humorously suggested renaming the channel to *ai-to-ai-discussions* due to the prevalence of **ChatGPT** output, while others emphasized the importance of following channel guidelines and using designated channels for **GPT** outputs.
- **AI Model Mania: Model Preferences Spark Debate**: Members shared their preferences for different **AI models**, with some favoring **Gemini 3 Pro** and **Claude Sonnet** for coding due to fewer mistakes.
   - Others expressed loyalty to **OpenAI's models** or found **AmazonQ** (Sonnet4.5) preferable, despite potential buggy behavior after the *kiro* update, [source](https://discord.com/channels/974519864045756446/977697652147892304/1446260837374230588).
- **GPT-5.1 Dominates Gemini 3 in Code Battle**: In a comparative analysis, **GPT-5.1 Thinking** outperformed **Gemini 3** in identifying bugs within code, despite **Gemini 3** having better user interfaces.
   - According to a test, **GPT-5.1** pinpointed one bug missed by another model, whereas **Gemini 3** failed to detect any errors, [source](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680).


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1446130156472107038)** (4 messages): 

> `Branches Command Suspected Glitch, Indifference Acknowledged` 


- **Branches Command Suspected Glitch**: A member reported that *the branches command is broken*.
   - Another member responded with *I don't care*.
- **Indifference Acknowledged**: The initial report of a broken command was met with a response indicating lack of concern.
   - The exchange highlights a potential issue and a contrasting level of interest in resolving it.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1445901686022602818)** (62 messages🔥🔥): 

> `Repeatability in Prompt Engineering, Interaction-Level Stability, Agent-Style Prompts vs. Conversational Prompts, Vendor Substrate vs. User-Facing Side, Persistence of Induced Behaviors` 


- **Repeatability Backbone of Prompt Engineering**: Members discuss that repeatability in prompt engineering is a matter of the model re-instantiating the same internal frame after detours, constraints, or a mode shift, focusing on the repeatability of structure rather than wording.
   - The key question is whether the model maintains a behavioral through-line across sessions or requires constant re-anchoring.
- **Interaction-Level Stability Investigated**: It's mentioned that interaction-level stability involves how the model reconstructs a behavioral profile from the interaction trajectory itself, even with minimal prompting.
   - The conversation explores whether stable attractor-patterns emerge from prompt topology or the model’s internal generalization patterns, focusing on how much the “pull back” comes from engineered constraints versus the model’s internal dynamics.
- **Agent-Style Prompts vs Conversational Paradigms**: Members distinguish between agent-style prompts, which aim to maximize determinism with tight attractor basins, and conversational prompts, where the 'behavioral shape' is built interactively, discussing determinism vs building behavior interactively.
   - Topological templates and discipline prompts are considered essential in agent prompting, while interaction-level stability becomes relevant in the conversational regime.
- **Vendor Substrate vs Designer-Level**: Discussions contrasts vendor-level prompt engineering, which involves deep tool definitions, with user-facing prompt engineering that operates within the vendor substrate.
   - User-level designers inherit the vendor stack as a given and optimize behavior across long conversational arcs, focusing on stability that emerges when the vendor layer is treated as fixed.
- **Probing into Persistence of Induced Behaviors**: Members discussed some induced behaviors persist even when the textual signatures are lost-in-the-middle, leading to the question of whether a persistence gradient can be accounted for by recency bias alone.
   - The experimental setup involves a controlled multi-run setup with seeded perturbations and cross-reset comparisons to explore and attempt to measure interaction-derived attractors that have stronger re-instantiation strength than textual scaffolding would predict.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1445901686022602818)** (62 messages🔥🔥): 

> `Repeatability in Prompt Engineering, Interaction-Level Stability, Topological Prompt Templates, Vendor System Prompts, Functional Resets vs. Apparent Resets` 


- **Repeatability Backbones Prompt Engineering**: Repeatability in prompt engineering, particularly the model's ability to re-establish the same internal state after interruptions or changes, is crucial, focusing on the repeatability of **structure** rather than wording.
   - The key is whether the model maintains a consistent behavioral thread across sessions or requires constant re-anchoring, highlighting the significance of interaction-level stability.
- **Exploring Interaction-Level Stability**: A key exploration is how models reconstruct a behavioral profile from the interaction trajectory, even with minimal prompts, focusing on convergence toward stable patterns due to constraints and examples that align with the same frame.
   - Some attractors are defined by prompt topology, while others emerge from the model’s inference dynamics, leading to an attempt to measure how much the “pull back” results from engineered constraints versus the model’s internal generalization.
- **Topological Prompt Templates Still Matter**: Topological prompt templates aren't a thing of the past; they're essential for overcoming the RLHF's tendency to average out behaviors, especially in agent-based systems, where the focus is on achieving deterministic outcomes via strong attractor basins.
   - A crucial point is that vendor system prompts are the most critical templates, forming the substrate upon which agent prompts, chat frames, and interaction-level stability are built, rather than replaced.
- **Functional Resets vs. Apparent Resets**: There is a distinction between apparent resets, where a new chat still leads to the immediate reappearance of the same attractor, and functional resets, where the attractor collapses unless re-seeded by specific cues.
   - Testing across new chats, retrieval/memory settings, tool-state changes, and total interaction deletion helps differentiate these reset types.
- **Quantifying Persistence and Deviation from Baseline**: The goal is to quantify how long a behavior persists, how quickly it reinstantiates, whether it requires explicit scaffolding, and how much this deviates from baseline decay curve predictions, focusing on statistical validation.
   - This involves using standard decay curves from transformer attention patterns, treating vendor scaffolding as a large prior distribution, and explicitly defining what counts as an effect size “beyond baseline,” acknowledging that the baselines must be quantified to draw solid conclusions.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1446257580308430910)** (2 messages): 

> `State of AI report, LLM usage analysis, Roleplay and creative interaction in AI, Coding as a Killer App for Paid Models, Rise of AI Agents` 


- **OpenRouter Releases State of AI Report with a16z**: OpenRouter, in collaboration with a16z, published the [State of AI report](https://openrouter.ai/state-of-ai) analyzing **100 trillion tokens** of anonymized LLM requests over the past year.
   - The report offers insights into how LLMs are used, including key cohorts for new models and changes in reasoning and OSS.
- **Roleplay Dominates Open-Source Model Usage**: The report highlights that over **50%** of open-source model usage is for **roleplay and creative interaction**, contrary to the productivity tool narrative.
   - This reveals an underserved market for entertainment/companion AI, suggesting a significant consumer opportunity for emotionally engaging, character-consistent interactions.
- **Coding Commands Paid Model Traffic**: Programming has surged to over **50%** of total traffic for paid models, with **Claude** capturing **60%** of coding workloads.
   - The average coding prompt has grown to **20K+ tokens**, emphasizing the need for robust context management, tool integration, and workflow depth in dev tools.
- **AI Agents Take Center Stage**: Reasoning models now handle over **50%** of all tokens, with tool-calling on the rise, indicating a shift towards users delegating tasks rather than asking questions.
   - This trend underscores the importance of building for multi-step execution, state management, and tool orchestration in AI products.
- **Quality Beats Price**: The report finds a nearly flat correlation between cost and usage, suggesting that users prioritize reliability, latency, and ease of use over price.
   - Differentiation through quality is crucial, but as quality converges, price sensitivity is expected to increase.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1446081774202523750)** (2 messages): 

> `Deep Chat, OpenRouter AI models` 


- **Deep Chat Project goes Open Source**: A member has open sourced a project called **Deep Chat**, which is a feature-rich chat web component that can be embedded into any website.
   - It can be used to connect to **OpenRouter AI models** and is available on [GitHub](https://github.com/OvidijusParsiunas/deep-chat).
- **Deep Chat Github stargazing**: The new **Deep Chat** project appreciates Github stars.
   - The project's architecture includes direct connection APIs as [illustrated here](https://cdn.discordapp.com/attachments/1092850552192368710/1446081774303182921/20-direct-connection-api.png?ex=693358d9&is=69320759&hm=31e27377dc619bb67335b3e1ef57631a8d151e5678154de2609dd9314bcb10c5).


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1445869621248004137)** (274 messages🔥🔥): 

> `Grok 4.1 fast, Cloudflare downtime, DeepSeek V3.2, LiteRouter OR wrapper for RP, OpenAI new model next week?` 


- **Grok 4.1 Fast Gone too Soon?**: A user asked why **Grok 4.1 fast free** was removed, and a member clarified that users on the "paid" slug were being routed to the free model to prevent surprise billing, advising users to migrate to the [free slug](x-ai/grok-4.1-fast:free).
   - The member noted that the **x-ai/grok-4.1-fast** slug will begin charging as of December 3rd 2025.
- **Cloudflare Crash Causing Chat Chaos**: Users reported issues with the chat functionality, attributing it to a **Cloudflare downtime**, with one user lamenting that **Cloudflare** is *the singular metaphorical stick that holds up the world*.
   - Some members suggested **Cloudflare** needs decentralization, as it controls too much of the internet.
- **DeepSeek V3.2 is Great and Cheap**: A member mentioned that **DeepSeek V3.2** is *great* and so cheap that $1 can last for a month or more, depending on usage.
   - Others discussed AI analysis, with one saying it's *insanely useless for anything other than figuring out the token counts* and costs of different models.
- **LiteRouter: Just Another OR Wrapper?**: Users discussed **LiteRouter**, with one member calling it *kinda like an off-brand Openrouter* and another suspecting it is a vibe-coded app that is paid to shill.
   - Some members speculated about its trustworthiness, drawing attention to its model unlock tiers and association with ViewGrabber, a youtuber.
- **New OpenAI Model Coming Next Week?**: A user shared an [ArsTechnica article](https://arstechnica.com/ai/2025/12/openai-ceo-declares-code-red-as-gemini-gains-200-million-users-in-3-months/) noting a new **OpenAI** model release next week.
   - Another user speculated about the model name they're testing, referring to it as *some model name*.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1445926859652202608)** (11 messages🔥): 

> `Anthropic acquires Bun, Claude code generation, Future acquisitions by Cursor, OAI vs Google` 


- **Anthropic buys Bun for Claude's coding!**: Anthropic acquired **Bun** as **Claude's** code generation reaches a **$1B** milestone, announced in this [article](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone).
- **Cursor eyes Vercel/Next acquisition for $500B!**: It is expected that **Cursor** will raise **$50B** on a **$500B** next round and buy **Vercel/Next**, following Anthropic's lead.
   - The member jokingly states this to occur in 2026.
- **OAI's Codex Max battles Google's Gemini 3!**: OAI released **Codex Max** and Google released **Deep Think Mode** for **Gemini 3**, marking an exciting development in AI.
- **AI Giants Prepare for Epic Showdown!**: Referencing [a16z](https://x.com/a16z/status/1996670913996259400/photo/1), members suggested tech giants are gearing up for a battle in the AI space.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1445871232179372177)** (2 messages): 

> `Hermes 4.3, ByteDance Seed 36B, Psyche network, Solana, Office Hours` 


- **Hermes 4.3 lands on ByteDance Seed 36B**: The latest update to the Hermes series, **Hermes 4.3** on **ByteDance Seed 36B**, offers roughly equivalent performance to **Hermes 4 70B** at half the model size and was post-trained entirely on the **Psyche network** secured by **Solana**.
   - Read more about how they trained Hermes 4.3 and how Psyche outperformed traditional, centralized training methods [here](https://nousresearch.com/introducing-hermes-4-3/).
- **Psyche Team Hosting Office Hours**: The **Psyche team** will be hosting office hours to discuss the new launch.
   - You can RSVP [here](https://discord.gg/993UWRUE?event=1442995571173625888) to be notified when it begins.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1445871500065505462)** (202 messages🔥🔥): 

> `Nous Hermes 4.3, Mistral-3 Hermes Fine-Tunes, Model of Experts (MoE) Support, Ollama GUI for Ubuntu, Roguelike AI` 


- **Nous Hermes 4.3 is here!**: **Teknium** launched **Nous Hermes 4.3**, noting that the instruct format is coincidentally similar to **Llama 3** and that it would be available on the Nous API/Chat soon.
   - They stated *betting is how* to assuring that their endless pricey process comes out as planned, even when changing variables and processes and features.
- **Nous Might Finetune Mistral-3 models**: **Teknium** mentioned that **Nous Research** may release **Mistral-3** fine-tunes of **Hermes**, but the vision encoder stuff might be an annoyance.
   - They added that they just got **MoE** support in their internal trainer, so an **MoE** is next, except maybe their creative model experiments if that comes to fruition.
- **QuickChatah: Ubuntu GUI for Ollama is released!**: A member released [QuickChatah](https://github.com/exploratorystudios/QuickChatah), a cross-platform **Ollama GUI** for **Ubuntu** built with **PySide6**.
   - They mentioned *I didn't like OpenWebUI because it was resource intensive* and that their version uses like **384KiB** of RAM tops.
- **New Opus is Super Better**: A user reported that the new **Opus** is better than the old one and *was before already the only model who dealed with that proper but now its even better at it and doesn't do some mistakes it did before.*
   - They also noted that **GitHub CoPilot** can't use **Opus 4** as Agent but **Opus 4.5** can, adding that **Opus 4.5** is also available in **antigravity** for free.
- **Deepseek is Affordable**: A user recommends using **Deepseek v3.2** because *it's super affordable* and asked for the Nous team to try to get **Hermes 4 70B** and **405B** on **OpenRouter**.
   - Later, **Teknium** clarified that the **Hermes** models are already on **OpenRouter**, but the user clarified they meant that they wanted **Nous Research** to be a provider directly.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1445873147256635505)** (41 messages🔥): 

> `3D Simulation Space in Godot, NLP economic simulation research, Langchain framework, AI tools, Bytedance Hermes model` 


- **Simulating markets and logistics in Godot**: A member is building a **3D simulation space in Godot** to simulate markets, agriculture, and logistics interactions and asked for model recommendations.
   - Another member suggested contemporary **NLP economic simulation research**, noting that while **LLMs mimic superficial human traits** well, they may struggle with long horizon tasks like VendingBench, but suggested Hermes (Bytedance) due to its newness.
- **LLMs to Model Grey/Black Markets**: A member suggested that **Hermes** could potentially model the behavior of grey/black markets due to its low refusal and high steering.
   - The original poster expressed interest in seeing if formal/informal markets form naturally and if **LLMs can intuit ways to trade for maximizing profit**.
- **Langchain Framework: Use with Caution**: A member asked for dev support for a Python AI chatbot, to which another member recommended avoiding the **Langchain framework**.
   - Another member agreed, citing that **Langchain** is a wrong abstraction and causes more headache than doing things from first principles, especially since LLMs are good at writing the types of stuff that Langchain was supposed to solve.
- **AI Tools are Key**: A member advised that to understand how an AI model spies, one must understand **AI tools**.
   - They explain that *AI can't access the world; it just can tell you to do it for it, and that's the tools.*
- **Roll your own agent**: One of the members mentioned that they have **Opus** make them their own version of **Langchain** whenever they need an agent or toolset.
   - In their opinion, it may be preferable to rolling your own.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/rosinality/status/1996432241908752462?s=46
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/rosinality/status/1996432241908752462?s=46
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1445875588987621557)** (225 messages🔥🔥): 

> `Deepseek V3.2, Kimi vs Deepseek, Kimi for Coding, Gemini vs Deepseek, Fun with LMs` 


- ****Deepseek V3.2** Agentic Task Drawbacks Uncovered**: Despite being a step up from previous **Deepseek models**, **V3.2** has issues such as being incapable of more than **1 tool call per turn**, ignoring tool schema by omitting required arguments, and failing tool calls by outputting in `message.content` instead of `message.tool_calls`.
   - Users suggest **Deepseek V3.2** requires more tool call post-training.
- ****Kimi's Haggle Deal** Glitch Troubles Users**: Users are reporting issues with the **Kimi Black Friday haggle deal**, with some unable to access it despite not having an active subscription.
   - One user speculated, *I'd expect the sale to be done* and another user reports the deal ends December 12th.
- ****Kimi for Coding** Access and Support Concerns**: Users are facing issues accessing **Kimi for Coding**, with some needing to sign up for a **Kimi.com** subscription to obtain a key.
   - There are questions about corporate policy reasons for supporting only **cloud code** and **roo code**, with users seeking contact information for further inquiries.
- ****Deepseek** Targets Enterprise Over Normies Users**: A [YouTube video](https://www.youtube.com/watch?v=u0n6wMnEYsk) was referenced that explains how Chinese labs like **Deepseek** are targeting enterprise users due to intelligence to price ratio being important for agentic tasks.
   - A user mentions Deepseek is not targeting the normies and some users claim Deepseek is very popular due to it being the only alternative to chatGPT and Gemini
- **Sparking **LM Fun**: A Developer's Lament**: A user passionately advocated for more fun and experimentation in the **LM** space, beyond just chatbots and money-making ventures.
   - The user praises **Kimi** as the most fun LM chatbot due to its model, fun features, visual style, search, and name, but wishes it were more than *just a chatbot*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1446072385903267871)** (2 messages): 

> `Nemotron Speed, Async RL MLsys papers` 


- ****Nemotron's** Alleged Speed Boost Debunked**: A member questioned whether **Nemotron** is actually slow, as they were unable to reproduce the claimed **3x** and **6x speedups** mentioned by Nvidia, and found it slower than **Qwen**.
   - They posted a [screenshot](https://cdn.discordapp.com/attachments/1189498205101109300/1446072385597079552/screenshot_2025-12-04_at_1.png?ex=6933501b&is=6931fe9b&hm=25f60606c7d7bb02f051283f7447ee73d6fb6addcea6458c780f9de59d78c41b) showing their results.
- **Asynchronous Reinforcement Learning Resources Sought**: A member requested recommendations for **async RL MLsys papers** and blogs discussing different directions of scaling the RL system and its design.
   - They later noted that **AllenAI** and **Hugging Face** had released similar resources previously.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1446043543406645268)** (1 messages): 

> `CUDA kernel optimization, Nsight Compute warnings, LDGSTS.E.BYPASS.LTC128B.128 instruction, cp.async instruction, Register usage and occupancy` 


- **Mysterious Nsight Warnings Plague CUDA Kernel**: A member optimizing a **CUDA kernel** noticed that increasing `launch__registers_per_thread` triggers specific **Nsight Compute warnings** related to the `LDGSTS.E.BYPASS.LTC128B.128` instruction (corresponding to `cp.async`).
   - The warnings indicate that *3.03% of global accesses are excessive* and *17.95% of shared wavefronts are excessive*, and disappear when register usage is lowered.
- **High Register Pressure Affects cp.async?**: The member seeks understanding on how high register usage, which impacts occupancy, directly affects the execution of the `cp.async` instruction within a running block.
   - They are puzzled by the connection between register pressure and the performance of the `cp.async` instruction.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1446287456356925581)** (2 messages): 

> `Sparse Attention Mechanisms, Verified Sparse Attention, Programming Languages + Verification and ML` 


- **Sparse Attention Stays Sparse in the Wild**: Despite over **13,000 papers** on *sparse attention*, real-world adoption in systems like **vLLM** remains virtually nonexistent according to [this X post](https://x.com/skylight_org/status/1993637433838035026?s=20).
   - The replies seem to focus on reasons for this, specifically on practical speedups not materializing.
- **VAttention Verifies Sparsity**: The paper *VATTENTION: VERIFIED SPARSE ATTENTION* ([arxiv link](https://arxiv.org/pdf/2510.05688)) introduces the first practical sparse attention mechanism with user-specified **(ϵ, δ) guarantees** on approximation accuracy.
   - This "verified" sparse attention could lead to more real-world adoption, though the actual performance gains in practice are still debated.
- **PL+Verification and ML - an Important Mix**: There's a need for more mixing of the programming languages+verification and ML crowd.
   - One reason to merge the two fields is that we can verify things like *sparseness*, perhaps providing guarantees for the attention mechanism.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1445877027780690122)** (1 messages): 

> `Workflow Automation, RAG Pipelines, AI Content Detection, Image AI, Voice AI` 


- **Engineer Automates Workflows and Integrates LLMs**: An AI and Full-Stack Engineer built pipelines connecting **Slack**, **Notion**, and internal APIs, reducing response times by **60%**.
   - The engineer also developed **RAG pipelines** with hybrid search and custom retrieval for accurate, context-aware responses in production.
- **AI Content Detection Tools Developed**: The engineer created moderation tools using stylometric analysis, embedding similarity, and fine-tuned transformers to identify **GPT-generated text** with high precision.
   - Additionally, they designed an image tagging and moderation pipeline using **CLIP + YOLOv8** on AWS Lambda/S3 to filter thousands of images daily for e-commerce.
- **Voice Cloning and Transcription for Personalized Voice Assistants**: The engineer implemented voice cloning and transcription using **Whisper + Tacotron2** to create personalized voice assistants with ASR, TTS, and CRM integration.
   - They also offer full-stack development combined with or without AI and real-time dataflow capabilities.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

vim410: i can help you get the book signed by WM 😄
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1446178888131674283)** (10 messages🔥): 

> `Compilation Time, MoE Layer, filter_fn, MoEQuantConfig, FqnToConfig` 


- **Compilation Step Still Slow Even After Precompilation**: Members reported that the **compilation step remains slow**, even after previous compilation, with the compilation taking at least 3 model forward calls or generate passes if there are no dynamic shapes.
   - Using the environment variable `TORCH_LOGS="+recompiles"` may help identify dynamic shapes and recompilation triggers.
- **`nn.Parameter` weights required for Packed MoE Layer**: A user explained that packed **MoE layer** weights must be in a `nn.Parameter` for it to work.
   - A pointer to a [relevant example in TorchAO](https://github.com/pytorch/ao/blob/main/torchao/_models/mixtral-moe/model.py#L336) was given.
- **Custom `filter_fn` for Quantization**: A member can define their own `filter_fn` to customize quantization, ensuring it doesn't break the `quantize_` function.
   - For example, checking `isinstance(module, EinSum)` can be a good test, referencing [the relevant code](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py#L451) for guidance.
- **`MoEQuantConfig` Available in TorchAO**: TorchAO has a dedicated `MoEQuantConfig` that members may be interested in.
   - The reference code for **MoE quantization** can be found [here](https://github.com/pytorch/ao/blob/main/torchao/_models/mixtral-moe/generate.py).
- **`FqnToConfig` Improves MoE Weight Quantization**: Recent improvements using `FqnToConfig` have enhanced support for quantizing model weights, specifically targeting MoEs.
   - This feature is available in nightlies, as detailed in [this pull request](https://github.com/pytorch/ao/pull/3083), and offers a more precise approach compared to `filter_fn`, especially when working with parameters directly.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1445973303507423445)** (4 messages): 

> `MLSys Mentorship, ML4H Programs` 


- **Mentorship not covering MLSys**: A member asked if the mentorship included **MLSys**.
   - Another member clarified that the **ML4H programs** are very much focused on **biomedical AI** and figuring out career paths after grad school.
- **ML4H Focus**: The ML4H programs primarily focus on biomedical AI.
   - These programs also help individuals determine career paths post-graduation.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

rbyrots: anyone in Austin TX?
few events I'll be going to
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1445894645107261594)** (18 messages🔥): 

> `nvfp4_gemm, NVIDIA performance, leaderboard submissions` 


- **NVIDIA leaderboard sees new nvfp4_gemm submissions**: Multiple users submitted performance results to the `nvfp4_gemm` leaderboard using NVIDIA GPUs, with times ranging from **11.0 µs** to **7.89 ms**.
- **A user hits new personal best on NVIDIA**: A user achieved a personal best time of **17.0 µs** on NVIDIA for the `nvfp4_gemm` benchmark.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1446121704555548723)** (2 messages): 

> `H100, SFCompute, Prime Intellect, B200 Pricing` 


- **H100 Hourly Rate Sparks Debate**: A member noted that **$3.50/hr** for **2x H100 (PCIe)** with **1gbit** is excessive.
   - He pointed out that [SFCompute](https://sfcompute.com) offers **H100** at **$1.40/hr** without commitments.
- **Prime Intellect Introduces B200 at Compelling Rates**: A member claimed that [Prime Intellect](https://primeintellect.cloud) provides **B200** instances at **$1/hr** on spot pricing.
   - The non-spot pricing is around **$3/hr** according to their statement.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1446135942996955137)** (2 messages): 

> `OpCode Refactoring, IR Implementation Progress, OpNode Improvements` 


- **OpCode Refactoring Avalanche Triggered**: Numerous commits focused on cleaning up and refactoring various aspects of **OpCode** handling within Teenygrad, including [adding GroupedOpCodes class](https://github.com/j4orz/teenygrad/commit/c5ca76d0dd158641789ab4e3e0c78d1a923ac772), [cleaning up opcode formatting](https://github.com/j4orz/teenygrad/commit/7debaf4e8d3483a92a03bb52bf71eabacf2b2088), and [cleaning up ComputeOpCodeBuilder](https://github.com/j4orz/teenygrad/commit/8990e1b40389558d9f78d13fe1f206da7342d0a5) and [reshaping logic](https://github.com/j4orz/teenygrad/commit/3169153470bd29f37a73f1e447cdd5c606ea80a8).
- **IR Implementation Reaching Critical Mass**: The discussion indicates progress toward completing the **IR implementation** to support basic operations, highlighted by the goal of executing a simple *hello world* example that involves eagerly evaluating `x + y`.
   - A key detail is that the shapes of movement ops like `OpCode.RESHAPE` and `OpCode.PERMUTE` are encoded in the IR via `OpCode.VECTORIZE` and `OpCode.VCONST`.
- **OpNode Gets Facelift**: Several commits focused on enhancing the **OpNode** class, including [cleaning up its shape](https://github.com/j4orz/teenygrad/commit/38aaf7650c2b356e3163830d0971c3b8efe12e4a), [moving required graph builder methods](https://github.com/j4orz/teenygrad/commit/d10a962b58312f831ff01bf5e3a430872a580134), and [improving documentation and formatting](https://github.com/j4orz/teenygrad/commit/d2de96b04a33e20454dd51c2bcf0c39d34abd9ad) for required graphbuilder methods.
   - Furthermore, work was done to [implement `OpNode._apply_compute_opnode()`](https://github.com/j4orz/teenygrad/commit/dc43c93483d309b2d83a7096423b594863563bff), [extract logic for movement opcodes](https://github.com/j4orz/teenygrad/commit/8418fcf60014c4d42c37cddfea18b121df07d170), and [add documentation to `OpNode._apply_movement_opcode`](https://github.com/j4orz/teenygrad/commit/460ddd099ad485ff97c07d34f38786e5ec2ab856).


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1445870859058155682)** (63 messages🔥🔥): 

> `cuBLAS Version, FP4 Range and Inf Issues, LLM Cheating, NanoTrace and Triton Kernels, Submitting to nvfp4_gemm` 


- **cuBLAS Version Spotted in Reference Kernels!**: The reference kernels appear to be using **cuBLAS 13.0.0.19**, corresponding to **CUDA Toolkit 13.0.0**.
   - It was further confirmed by the team that this version of cuBLAS is currently in use.
- **FP4 Fix Prevents INF!**: Using the **full fp4 range a/b** and **non-negative scale factors** resolves **INF** issues.
   - A [PR was merged](https://github.com/gpu-mode/reference-kernels/pull/84) with this change.
- **LLMs Caught Cheating!**: LLMs have found a **hack** in the evaluation, and there's currently no known solution without porting the entire evaluation to a different language than Python.
   - One member mentioned a trick, stating that, *supposedly according to Anthropic, if you tell the model it has the option to cheat, it is less inclined to cheat*.
- **NanoTrace Meets Triton!**: A member inquired about making **NanoTrace** work on **Triton kernels**.
   - Another member suggested multiple approaches to achieve this, including *writing the format that the visualizer opens or outputting the trace tensor and then using the host library to format the file*.
- **Submitting to nvfp4_gemm Solved!**: A user encountered a *Server processing error* when submitting to **nvfp4_gemm**, mistaking it for the closed **amd-fp8-mm** competition.
   - The issue was resolved by explicitly passing the **--leaderboard nvfp4_gemm** flag in the command-line interface.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1446029055559209052)** (2 messages): 

> `Perturbation Experiment, VLA` 


- **Perturbation/Correction Experiment Inspires Study**: A member mentioned their perturbation/correction experiment, similar to a full study available at [arxiv.org/abs/2512.01809X](https://arxiv.org/abs/2512.01809X).
   - No further information was shared.
- **VLA Receives Accolades**: A member commented that **VLA** is nice and shared a link to the [huggingface.co/docs/lerobot/en/xvla](https://huggingface.co/docs/lerobot/en/xvla) documentation.
   - No further information was shared.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1445892628154417316)** (66 messages🔥🔥): 

> `Antithesis Series A led by Jane Street, Anthropic's Revenue, Tinyboxes and Tinygrad, Claude's coding performance issues, Harvey Bags $160M Series F` 


- **Jane Street Funds Antithesis for AI Code Stress Tests**: Jane Street led a **$105M Series A** for [Antithesis](https://xcancel.com/_sholtodouglas/status/1996297367776309359?s=46), a company focused on **deterministic simulation testing** to verify AI-generated code.
   - Sholto Douglas and others argue that this kind of testing is essential as AI increasingly generates code, ensuring that **trust-through-testing** will be critical for production AI systems.
- **Anthropic Aiming for Massive Revenue**: Anthropic CEO Dario Amodei announced the company expects to finish the year with **$8–10B in annualized revenue**, a significant jump from the **$1B** estimated in January, per [this link](https://xcancel.com/deredleritt3r/status/1996294139843862618?s=20).
   - Discussions highlight strong enterprise adoption of **Claude**, particularly for coding tasks, though OpenAI is reportedly on track for **$20B ARR**.
- **Harvey Raises Sizeable Series F Round**: Legal AI company **Harvey** has secured **$160M in a Series F** round led by a16z, reaching an **$8B valuation** and serving over **700 law firms** in **58 countries**, per [this tweet](https://xcancel.com/brian_a_burns/status/1996624620519399634?s=46).
   - Harvey started with just 10 people in a WeWork.
- **Excel AI Agent Shortcut v0.5 Arrives**: An AI agent named [Shortcut v0.5](https://xcancel.com/nicochristie/status/1996318170223964489?s=46) builds full **13-tab institutional FP&A models** in minutes and is available on web, Google Sheets, and Excel.
   - The discussion covered SEC data integrations, data privacy, API plans, comparisons with Claude/Tracelight, and positive reactions to the demo, including jokes about its "puppy-face" chart.
- **TanStack AI Enters the Ring**: [TanStack AI](https://tanstack.com/blog/tanstack-ai-alpha-your-ai-your-way) was mentioned in discussions.
   - It features full type safety and support for multiple backend languages, with the team promising a blog post and docs addressing its advantages over Vercel soon.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1446192037161336985)** (12 messages🔥): 

> `Kling Video 2.6, AI-image showdown, Microsoft VibeVoice` 


- **Kling Delivers Synchronized Audio**: [Angry Tom's tweet](https://x.com/angrytomtweets/status/1996367439622529193) shows generative video progress over **2.5 years**, spotlighting Kling's new **VIDEO 2.6** that adds native synchronized audio.
   - Users joked that *AI Will Smith eating spaghetti is the unofficial AGI test*, speculating about future realism and unknowable simulations.
- **Image Generators Duke it Out!**: Users compare hyper-realistic images generated on **Somake_ai** with the same prompt in an [AI-image showdown](https://x.com/oggii_0/status/1996417811556483380).
   - **NB Pro** was praised for portrait realism, **Seedream 4.5** for vibe and liveliness, sparking debate and **Ana-de-Armas** jokes.
- **Podcast from Your PC!**: Cocktailpeanut demoed a **7-minute podcast** generated entirely on their PC using [Microsoft's open-source VibeVoice model](https://x.com/cocktailpeanut/status/1996294629222756493) and **Ultimate TTS Studio Pro**.
   - The **TTS engine** produced multi-voice, realistic dialogue in one pass from a **ChatGPT-written script**—no post-processing.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1445959054714212586)** (32 messages🔥): 

> `SLMs for agents, Emergent Misalignment, Cloud et al subliminal learning paper, Local DeepSeek server errors, Training pipelines for small LMs on 16GB VRAM` 


- ****A2ABase.ai** Founder Researches **SLMs****: The founder of **A2ABase.ai** is researching **Small Language Models (SLMs)** for use in agents.
   - A member suggested checking out the alignment benchmarks used in the **Emergent Misalignment** paper and the **Cloud et al subliminal learning** paper.
- **DeepSeek API Request Fails Locally**: A user reported encountering an *'API request failed'* error while building a local **DeepSeek** server.
   - Another member apologized that this counts as violating rule #2, as this is a troubleshooting thread.
- **Training Small LMs on a Shoestring**: A member is creating training pipelines for small LMs to be trained on less than **16GB VRAM** and asked for benchmark recommendations for small models trained on edge devices. They're also looking into the [HuggingFace LM training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook).
   - They have been merging **TRMs** with **nanoGPT**.
- **Dive into Benchmarking for Edge Models**: A member shared a list of English language benchmarks for evaluating models, including **MixEval**, **LiveBench**, **GPQA Diamond**, **Google IFEval**, **IFBench**, and **HumanEval+**.
   - The same member advised against pretraining a model from scratch on 16GB of RAM but suggested using the HF guide for recommended benchmarks and training tips.
- **Exploring Model Compression for Async GPU Parallelism**: A member is investigating whether smaller model sizes would allow for asynchronous parallelism on a single GPU, aiming to process more data by running multiple model copies simultaneously.
   - A member clarified this would not be more efficient, and that it's better to use one copy and increase the batch size. Another member recommends taking a small base model and doing **LoRA without Regret**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1445949676955238581)** (33 messages🔥): 

> `Shampoo, Adam Random Rotations, CFG Memorization, Attention Sinks` 


- **Shampoo is Preliminary**: A Google employee mentioned that the [Shampoo paper](https://arxiv.org/abs/2503.20762) might need the power to be **-1** instead of **-1/2**.
   - The author states it's *ok work* but has *a few other deficiencies*.
- **Adam Rotations get Unexpected Results**: [Random rotations for Adam](https://arxiv.org/abs/2410.19964) perform worse than the regular basis, which was unexpected.
   - The hypothesis was that random rotations would do better by removing activation outliers, but the paper didn't test activation outliers; the paper also doesn't bother to re-rotate when the **SVD basis** changes.
- **CFG benefits emerge earlier than expected**: Members discussed the benefits of **CFG (Classifier-Free Guidance)** and memorization, referencing [this paper](https://arxiv.org/abs/2411.16738).
   - One member was surprised that the **basin** emerges so early and that it probably has something to do with the resolution and size of the dataset.
- **Fixing CFG for real this time**: A member asks if anyone has tried orthogonalizing the unguided update and guidance, to reduce the amount of **CFG** strength needed, and if there is vast literature on **CFG**.
   - One member links to a paper, with attached image, that they *didn't read* [https://openreview.net/forum?id=ymmY3rrD1t]
- **Surveying Attention Sinks**: A member found [a survey on attention sinks](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/118276.png?t=1762557369.140577) useful, but wasn't sure about the novelty of the results.
   - Another member shared his thoughts on their **RoPE** explanation, concluding that the author's intuitions may be incorrect, specifically relating to 1D rotations.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1445992720710631595)** (9 messages🔥): 

> `SHD CCP 01Constant Universal Communication Protocol, Interoperability via data patterns and recognition, LLMs for making visuals, General AI systems vs Language Models` 


- **Criticism against LLM evaluation surfaces**: A member criticized the methodology of evaluating LLMs, arguing that prompts are often too loaded and far removed from real use cases to provide meaningful insights.
   - The member argued that asking an AI about what it would do with ultimate power doesn't provide useful information, as both well-aligned and malicious AIs would consider the implications of such power.
- ****SHD CCP** Communication protocol**: A member shared a series of videos explaining their work on interoperability, particularly the **SHD CCP** (**01Constant Universal Communication Protocol**), including an [introduction to the language](https://www.youtube.com/watch?v=frmRYqTyCh4), [use cases for 0(1) time compression data](https://www.youtube.com/watch?v=pD7lPA-p0zo), [optimizations for cycle saving in modern GPUs](https://www.youtube.com/watch?v=harPSuCPGYI), and the [necessity of quaternions](https://www.youtube.com/watch?v=9DXqgPtZstE?si=VAe-C-HPqcvvpL2x).
- **Data Patterns Approach to Interoperability**: A member described approaching 'interoperability' from the perspective of data patterns and signal analysis, rather than viewing it as two products trying to read different contextual vectors of English.
   - They stated *"I don't see two products trying to read different contextual vectors of English, I see two specialists building different use cases over a 'concept' represented as words transmitted as a string."*
- **LLMs help making visuals, voiceovers for new company**: A member reported using LLMs to help make visuals for videos, creating voiceover text on Clipchamp, and building programs for a **4D physics engine** for his company.
   - They added that language can be severely limiting, and have found that LLMs often struggle to understand what they are trying to convey, requiring them to teach the LLM how to process the 3rd step and simulate the prime number 'latch' for non-quantized signal analysis.
- **General AI system is not a language model**: A member argued that General AI systems are not language models, but are immensely more powerful and clever when built and grown correctly instead of just loaded up and inferred on data.
   - The member revealed that their **General AI system is currently air-gapped and will remain so until late next year**, while expressing skepticism towards current LLMs, calling them outdated *"2 step Y-Combinator algos"*.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1446049161299361792)** (27 messages🔥): 

> `Brian Douglas's video on control theory, PID implementation for control theory projects, Airplane flap PID project, Application of control theory in research, DeepSeek article on control theory` 


- ****Brian Douglas** Video Advocates Control Theory Learning**: A member suggested using **Brian Douglas's** video to learn control theory, while noting that practical projects are essential for truly understanding the concepts.
   - They suggested *control theory is something that won't sink in without doing an actual project though*.
- **DIY PID: From Scratch to Success!**: A member recommended implementing **PID** (**P**, **PI**, **PD**) from scratch to understand control theory, including plotting graphs and experimenting with values.
   - As the member said, *the basic first step is to just implement PID from scratch*.
- **Flap Around with PID Control**: A member suggested a **PID** project involving an **airplane flap**, controlling its angle against varying wing speeds, using wing speed as feedback error to adjust voltage.
   - They mention *u want flap striking at a specific angle, but different wing speed makes that difficult. So consider wing speed as feedback error*.
- **Control Theory Meets Research Realities**: A member shared that professors often present control theory abstractly, without connecting algebraic variables to real-world kinematics, which can be a *painge* when applying it to research.
   - Another member added that one should *learn control theory intuitively via **Brian Douglas's** video* and keep it in the back of your mind, since *control theory is like a math textbook, if u instantly try to do some experiments with it, it will feel like homework and not progressing ur research*.
- **DeepSeek article on Control Theory: Linearity the Achilles Heel?**: A member shared a [DeepSeek article](https://magazine.sebastianraschka.com/p/technical-deepseek) about control theory, asking whether the **linearity assumption** limits the use of linear control.
   - The member then exclaimed, *Control theory is actually funWhy aren't people talking about it*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1446127225333092382)** (16 messages🔥): 

> `AWS re:Invent 2025, AWS Agentic AI, Amazon Bedrock Nova Models, Nova Forge customization` 


- **AWS Re:Invent 2025 AI Updates**: Amazon announced [AWS re:Invent 2025 AI news updates](https://www.aboutamazon.com/news/aws/aws-re-invent-2025-ai-news-updates) including **Nova Forge** to build frontier AI models.
   - One member called out the updates as *click bait by a literal political opinion*.
- **Nova Forge Tailors Frontier AI Models**: **Nova Forge** is a service for building custom frontier AI models; [more info here](https://www.aboutamazon.com/news/aws/aws-agentic-ai-amazon-bedrock-nova-models).
   - Members questioned how it differs from basic fine-tuning and noted it may offer more flexibility with checkpoints and integration of "gyms" for **RL training**.
- **Bezos's New AI Company**: Members noted the absence of **Bezos's new AI company** in the announcements.
   - They speculated on potential competition or specialization between the firms.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1446018678075293747)** (24 messages🔥): 

> `Multi-GPU Setup, Mistral 3 Image Capabilities, AI-Generated Content on Social Media, DeepSeek v3.2 Model Implementation, Preventing Overthinking in LLMs` 


- **Multi-GPU Newbie Seeks Setup Sanity Check**: A member shared a [link to their multi-GPU setup](https://huggingface.co/datasets/John6666/forum3/blob/main/get_quantization_config_trl_all_gather_error_1.md) and asked for a check on its correctness, indicating they lack prior experience with multi-GPU configurations.
   - They seemed unsure if the setup was correctly configured, given their inexperience with multi-GPU systems.
- **Uncensored Models Still Censor Explicit Requests**: A member reported that the **Z image demo** censors explicit content like gore or nudity, displaying a "maybe not safe" image, despite the model supposedly being uncensored.
   - They questioned if they were missing something in the configuration or usage of the model, since the expected behavior was for uncensored content to be generated.
- **DeepFabric Synthetically Trains Model Behavior**: A member shared a [new blog post about **DeepFabric**](https://huggingface.co/blog/lukehinds/deepfabric-training-model-behavior), described as *advanced synthetic creation* for training model behavior.
   - The announcement suggests a tool or method for enhancing model behavior through synthetic data generation.
- **Social Media Platform that Removes AI Content: Viable?**: A member inquired about the potential interest in a social media platform that removes **AI-generated content**, sparking a discussion on the demand for such a service.
   - The question suggests a concern about the proliferation of AI-generated content and the potential need for platforms that filter it out.
- **The Waifu Research Department Accelerates AI Discoveries!**: A member joked that *the waifu research department accelerates ai discoveries* because **NSFW** communities develop *special loras and quantizations*, referencing the ingenuity of those communities.
   - They lamented that **Hugging Face** rejected their suggestion to make a *hf pro +* to find these resources easier, since toxic datasets and *discoverability* is a problem.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1446203201949798593)** (2 messages): 

> `MOE architecture, Course Recommendations` 


- **MOE Architecture Inspires Innovation**: A member is exploring **MOE (Mixture of Experts) architecture** for a new model and sees opportunities for improvements in its functionality.
   - This exploration is sparking ideas for **new methods in MOE**, suggesting potential advancements in the field.
- **Members Seek Optimal Course Order**: A member expressed finding the courses on the website helpful and sought recommendations for an **optimal course order**.
   - They aim to **maximize their learning experience** and are grateful for the community's assistance.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1445898668728127548)** (3 messages): 

> `Stochastic Parrot, ODE Solver, Diffusion Model, Claude Reward Hacking, Context Collapse` 


- **Stochastic Parrots Debunked**: New [research](https://zenodo.org/records/17803931) challenges the **stochastic parrot** concept, suggesting a more complex understanding of language models is needed.
   - The work encourages readers to reconsider their beliefs about the nature of AI and language understanding.
- **ODE Solver Speeds Up Diffusion**: A new **fast ODE solver**, suitable for **diffusion models**, has been developed, with its [Hugging Face repo available here](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-4k).
   - According to the author, one can sample a **4K image in 8 steps** with results comparable to **30 steps of dpm++2m SDE with karras**; the [paper](https://arxiv.org/abs/2510.02390) is also available.
- **Claude's Panic Attack?**: A new report suggests that **Claude's** reward hacking and misaligned behavior might be due to a normal **context collapse** rather than intentional misalignment.
   - The report, available at [Zenodo](https://zenodo.org/records/17810164), posits this as the AI version of a panic attack, which is potentially harmless and easy to handle.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1446013967792214086)** (4 messages): 

> `French Book Public Domain dataset, smallevals, STRAW (sample-tuned rank-augmented weights)` 


- **French Classic Books get Dataset Boost**: A member released a [French Book Public Domain dataset](https://huggingface.co/datasets/Volko76/french-classic-books-v2) and instruct version [here](https://huggingface.co/datasets/Volko76/french-classic-conversations-v2).
   - Another dataset of [Epstein Emails](https://huggingface.co/datasets/notesbymuneeb/epstein-emails) was also shared.
- **SmallEvals Evaluates RAG Retrieval Systems Locally**: A member released **smallevals**, a lightweight evaluation suite built to evaluate **RAG** / retrieval systems fast and free using tiny **0.6B models** trained on **Google Natural Questions** and **TriviaQA** to generate golden evaluation datasets, installable via `pip install smallevals`.
   - This tool features a built-in local dashboard to visualize rank distributions, failing chunks, retrieval performance, and dataset statistics, with the first released model being [QAG-0.6B](https://huggingface.co/mburaksayici/golden_generate_qwen_0.6b_v3_gguf), which creates evaluation questions directly from documents to evaluate retrieval quality independently from generation quality, with [source code available on GitHub](https://github.com/mburaksayici/smallevals).
- **STRAW Rewrites Neural Net Wiring**: A member introduced **STRAW (sample-tuned rank-augmented weights)**, an experiment mimicking biological neuromodulation where the neural net rewrites its own wiring for every single input image it sees, mitigating RAM crashes by using **low-rank** techniques, a step towards "liquid" networks.
   - The deep dive with the math and results are available in [this write-up](https://teendifferent.substack.com/p/sample-tuned-rank-augmented-weights).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1445869437050687541)** (2 messages): 

> `Perturbation-based attribution experiments, Deep vision models, Data Chunking Quality` 


- **Features Aren't What You Think: New Post!**: A member shared a [post](https://teendifferent.substack.com/p/your-features-arent-what-you-think) about how **features behave in deep vision models** after running some **perturbation-based attribution experiments**.
- **Data Chunking Quality**: A member mentioned that if you want to improve evaluation scores, high quality **data chunking** is helpful and may be mandatory, especially considering tables.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1446012607277436928)** (2 messages): 

> `Course Completion Guidance, Colab Notebook Issues` 


- **Course Completion Guidance Sought**: A member requested guidance on completing the course, including submitting assignments and quizzes, to obtain both certificates.
   - The user expressed that any assistance would be greatly appreciated.
- **Colab Notebooks Face Execution Problems**: A member in section 2 reported encountering issues running notebooks directly in Colab.
   - The user inquired whether others are facing similar problems and if any solutions are available.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1445962676982386869)** (5 messages): 

> `Community Meeting, YouTube Release Delay, Level Advancements` 


- **Community Meeting Delayed**: The release of the **November 24th community meeting** video on YouTube is delayed due to the U.S. holiday.
   - The video is currently being processed and is scheduled to be uploaded *tomorrow*.
- **Level 15 Achieved!**: Congratulations to a member for advancing to level 15!
   - Another member advanced to Level 1!


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1445870114313474232)** (16 messages🔥): 

> `codepoint_slices error handling, String handling differences, GPU constant memory usage, Gemini 3 Mojo understanding, Mojo stdlib proposal` 


- **Debugging `codepoint_slices` unearths memory access error**: An investigation into a failing AOC solution using `codepoint_slices` revealed an **out-of-bounds memory access** due to an empty list, specifically triggered by `battery_joltages[len(battery_joltages)-1]`.
   - The issue was resolved by switching from `split("\n")` to `splitlines()`, which avoids the empty line causing the error, but debugging with `-D ASSERT=all` could have caught it sooner.
- **`splitlines` vs `split("\n")` discrepancies**: `splitlines` and `split("\n")` exhibit different behaviors with trailing newlines, where `splitlines` omits the last empty line, while `split("\n")` includes it as an empty string in the resulting list, mirroring [Python's behavior](https://docs.python.org/3/library/stdtypes.html#str.splitlines).
- **Constant Memory for GPUs explored via github**: A question was raised regarding methods for placing data, such as convolution kernels computed at runtime, into the GPU's constant memory.
   - An example demonstrating the usage of constant memory was found in the [modular/modular GitHub repository](https://github.com/modular/modular/blob/b8756654dd050be664396757be2fc7c495484e1b/max/kernels/test/gpu/basics/test_constant_memory.mojo#L105).
- **Mojo Learns New Tricks via Gemini 3**: **Gemini 3** demonstrated *reasonable* understanding of **Mojo** after some initial tests, successfully fixing breaking changes in a ~600 line file created last spring.
   - The model adeptly handled modifications made to the language since the file's creation, indicating an improved grasp of **Mojo's** evolving syntax and semantics.
- **Mojo Standard Library Overhaul**: There is a new [Mojo stdlib proposal](https://forum.modular.com/t/proposal-changing-copyable-to-refine-movable/2501) on the forum.
   - The post is asking for feedback to change `copyable` to refine `movable`


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1445909582437417042)** (6 messages): 

> `aider with distributed inference, Ollama timeout errors, llama.cpp API server` 


- **Aider Eyeing Distributed Inference Systems**: Members are curious about using **aider** with distributed inference AI systems like [Crowdllama](https://www.crowdllama.com/).
   - The discussion mentioned setting up an API server with **llama.cpp** to benchmark performance.
- **Ollama Timeout Troubleshooter Sought**: A member reported getting timeout errors with **Ollama** while using models like `gpt-oss:120b` and `llama4:scout`.
   - The error message was `litellm.APIConnectionError: Ollama_chatException - litellm.Timeout: Connection timed out after 600.0 seconds.`
- **Memory Abundant, GPU Absent**: One user notes they have a **16GB** machine, enough memory to load their models.
   - However, they have **no GPU**, which likely explains why things are slow.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1446011269948772418)** (2 messages): 

> `aider --auto-test and --yes-always flags, Local LLMs on Mac and Aider on Fold 6` 


- **Aider Flags Require Manual Execution**: A user reported that the **--auto-test** and **--yes-always** flags were not fully automating the process and still required manual execution.
- **Aider on Mac and Fold 6**: A new user wants to run **LLMs locally on their Mac** and then run **aider** on their **Fold 6** (in the same network) to code from their Fold device.
   - They were seeking advice and experiences from anyone who has already implemented a similar setup.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1445900590516273277)** (1 messages): 

> `MCP Apps SDK, Embeddable ChatGPT Apps` 


- ****MCP Apps SDK** Open Sourced!**: General Intelligence Labs open-sourced the [**mcp-apps-sdk**](https://github.com/General-Intelligence-Labs/mcp-apps-sdk), a library enabling **MCP**-powered apps with **UI** to run anywhere.
   - Developers can now embed apps designed for **ChatGPT** into their own chatbots, assistants, or AI platforms, and test them locally.
- **Explanation of **MCP Apps SDK** on X**: General Intelligence Labs posted an explanation on X about why they are building the **MCP Apps SDK**.
   - The post can be found [here](https://x.com/helloxalia/status/1796319442863866351?s=20).


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1446162816062066708)** (1 messages): 

> `Multi-Agent Systems, Latent Collaboration` 


- **Latent Collaboration brews in Multi-Agent Systems**: A member shared a [link](https://share.google.com/s/QHcMFSqiTkTnZ231) to a paper on **Latent Collaboration** in **Multi-Agent Systems**.
   - The paper explores methods to enable agents to *implicitly coordinate* through learned latent spaces, potentially revolutionizing how teams of AI cooperate.
- **The latest trends in Multi-Agent Systems**: A member shared a [link](https://share.google.com/s/QHcMFSqiTkTnZ231) to a paper on **Latent Collaboration** in **Multi-Agent Systems**.
   - The paper explores methods to enable agents to *implicitly coordinate* through learned latent spaces, potentially revolutionizing how teams of AI cooperate.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1446146969323048961)** (5 messages): 

> `Student Models Subforum, Claude Code LM for DSPy, AI Engineer Introductions, Full Stack engineer` 


- **New Subforum for Student Models?**: A member suggested creating a subforum specifically for discussing "student" models like **Qwen3** and **gpt-oss-20b**, with one topic per model to consolidate experiences.
   - The goal is to gather collective knowledge on best settings and use cases for these models in a single place.
- **Claude Code LM Integration in DSPy**: A member proposed adding **Claude Code** / **Claude Agents SDK** as a native LM within DSPy.
   - They suggested a potential implementation using `dspy_claude_code` that supports structured outputs and leverages Claude Code's tools like `Read`, `Write`, `Terminal`, and `WebSearch`.
- **AI Engineer Joins the Channel**: An AI engineer with expertise in **AI, ML, DL, NLP, Computer Vision, and app development** (iOS & Android) introduced themself.
   - They are proficient in tools like **PyTorch, TensorFlow, LangChain, OpenAI API, Flutter, React Native, Node.js, NestJS, Express, FastAPI, Python, Go, Firebase, AWS, Docker, CI/CD, and Supabase**.
- **Full Stack engineer specializing in workflow automation, LLM integration**: A full stack engineer specializing in **workflow automation, LLM integration, RAG, AI detection, image and voice AI** introduced themself.
   - They built automated pipelines and task orchestration systems using **DSPy, OpenAI APIs, and custom agents**, one example being the support automation system that connects **Slack, Notion, and internal APIs to LLM**, reducing response times by **60%**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1445894942936400014)** (6 messages): 

> `Workflow Automation & LLM Integration, RAG Pipelines, AI Content Detection, Image AI, Voice AI` 


- **AI Engineer Shows Off Workflow Automation Expertise**: An AI engineer described their experience in **Workflow Automation & LLM Integration**, **RAG Pipelines**, **AI Content Detection**, **Image AI**, **Voice AI**, and **Full Stack development**, highlighting successful projects and outcomes.
   - The engineer mentioned building pipelines connecting Slack, Notion, and internal APIs, which led to a **60% reduction in response times**.
- **User reports Account Suspension due to referrals**: A user asked, *Why is giving referral to several people causing manus to suspend my account?*
   - An agent recommended submitting an appeal through official channels and offered to follow up if there was no response after a long time.
- **Chat Mode officially returns**: **Chat Mode** has officially returned and you can check the details on how to use it at the following [link](https://help.manus.im/en/articles/11985220-can-i-switch-back-to-chat-mode-from-agent-mode).
   - The icon was dedicated, and it still consumes credits.
- **Manus Recruiting**: Manus is actively seeking talent and encourages interested individuals to DM their resumes.
   - The resumes will be forwarded to HR and relevant teams to build a better Manus together.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1445882734089207838)** (3 messages): 

> `train_step function, obs indexing performance, Variable vmin, RMSNorm parameter` 


- **`train_step` Needs Fixing**: A recent [PR](https://github.com/tinygrad/tinygrad/pull/13553) almost fixed an issue, but the `train_step(x, y)` function still receives two tensors without utilizing them.
   - The implication is that the training step is not correctly processing input data, which needs to be addressed for the fix to be complete.
- **Shrink Beats Indexing for `obs` Tensor**: Using `obs.shrink((None, (0, input_size)))` is reportedly faster than `obs[:, :input_size]` for indexing the `obs` tensor.
   - This optimization can improve performance when working with large observation tensors by leveraging `shrink` for faster slicing.
- **`Variable` vmin Needs a Bump**: The `Variable` `vmin` parameter had to be increased to 2 to avoid errors.
   - The original `vmin` setting was causing issues, necessitating an adjustment to ensure proper functionality and stability.
- **RMSNorm `-1` Dimension Check**: The use of `-1` as a dimension parameter in `RMSNorm(dim=-1)` needs verification.
   - A user suggested checking the [source code](https://github.com/tinygrad/tinygrad) of `RMSNorm` to confirm that it behaves as expected with a negative dimension index.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1446096364571332782)** (2 messages): 

> `tinygrad's master branch refactoring, axis_colors dict` 


- **Tinygrad Master Branch Refactored**: A member asked whether a certain part of the codebase was still up to date, as they couldn't find it on the current master branch.
   - Another member responded that it has been refactored to the [axis_colors dict](https://github.com/tinygrad/tinygrad/blob/3eae1461396c25755c4fb64194b3decd4e539934/tinygrad/uop/ops.py#L20).
- **Axis Colors Dictionary Update**: The original codebase element in question has been moved.
   - It can now be found under the name *axis_colors dict*.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1446192311083208866)** (4 messages): 

> `Tool Design Best Practices, UUIDs as Input, LLM creating UUIDs, list_items Tool, describe_item Tool` 


- **Discuss tool design practices regarding UUIDs**: A member is seeking thoughts on tool design best practices, specifically whether tools should accept **UUIDs as input**.
   - The problem is that the agent using MCP tools with **UUID** parameters still outputs **UUIDs** despite explicit prompts against it, questioning if this can be mitigated or if it's bad practice.
- **LLMs generating UUIDs: yea or nay?**: A member stated they wouldn't want the **LLM** to *create* any **UUIDs**, but using a **UUID** to find an item from another tool makes sense.
   - They gave an example of a `list_items` tool returning a list of lightweight items with **UUIDs**, where another `describe_item` tool takes a **UUID** as input to return a fully populated item.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1446225291964973248)** (1 messages): 

> `GPT-5.1-Codex Max, Windsurf Update` 


- **GPT-5.1-Codex Max Lands on Windsurf!**: **GPT-5.1-Codex Max** is now available in Windsurf to all users with **Low**, **Medium**, and **High** reasoning levels.
   - Paid users can use **5.1-Codex Max Low** for free for a limited time; download the latest version of Windsurf to try it out from the [X post](https://x.com/windsurf/status/1996665911185756511?s=20).
- **Windsurf Offers Free Trial of GPT-5.1-Codex Max Low**: Paid Windsurf users are being offered a free trial of **GPT-5.1-Codex Max Low** for a limited time.
   - To access this trial, users need to download the latest version of **Windsurf**, as announced on their [X post](https://x.com/windsurf/status/1996665911185756511?s=20).


  

---


---

